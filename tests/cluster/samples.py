"""The test data of sd.cluster (docs/design/cluster.md §7.1): generated from formulas and fixed seeds, nothing
downloaded. The tests, the reference tool (tools/cluster_ref.py), the probe of scipy, scikit-learn and R
(tools/cluster_probe.py) and the measurements take their data from here, by id.

Kinds, each with its own getter:

    point sets  (X (n, p), weights, the true labels, a suggested k, evaluation points Xe) by tag:
                blobs      well separated Gaussian blobs (k = 2 to 10, p = 1 to 10), overlapping blobs, blobs
                           of very different sizes, elongated blobs (covariance condition 1e4), blobs of
                           different densities (HDBSCAN), rings and moons (density methods right, k-means wrong)
                ties       integer lattices (1-D to 3-D), an evenly spaced line, repeated points (all equal
                           too), a regular hexagon and its centre (equal distances to the centre), pairs exactly
                           eps apart
                degenerate n = 1, 2, 3, k = n, fewer distinct points than k, points on a line in 2-D
                           (singular covariance), a constant column
                scale      a very narrow range (1e-8), a large shift (1e8 + blobs: the expanded distance's
                           cancellation), a large scale (1e150), subnormal spacing
                inversion  triangles on which centroid and median linkage invert, and random sets of the kind
                weights    blobs with weights (zeros among them)
    labelings   pairs of labelings for the indices: equal partitions, one cluster, all singletons, random
                partitions (near the expectation), partitions that refine one another, string labels, a large
                n (10^6) for the counts
    mixtures    Gaussian mixture parameters (weights, means, covariances of the four covariance types) with the
                points they are evaluated at

    import tests.cluster.samples as S
    s = S.points("blobs/k3p2")       # cached; S.point_ids(tag="ties") lists ids
    s.X, s.w, s.truth, s.k, s.Xe      # numpy arrays (w, truth None when there are none; k None too)

Every record and reference names the data it was made from by `digest`.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib

import numpy as np

SEED = 20261124


def hash_int(k) -> int:
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "little")


def rng(*key) -> np.random.Generator:
    """The generator of a key: the same numbers on every machine (PCG64 of SEED and the key)."""
    return np.random.Generator(np.random.PCG64([SEED, *[hash_int(k) for k in key]]))


def digest(*arrays) -> str:
    """SHA-256 of arrays as given (dtype, shape and bytes, one after another; None counts)."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(repr((str(a.dtype), a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _frozen(*arrays):
    for a in arrays:
        if a is not None:
            a.setflags(write=False)


def lex_rank(X) -> np.ndarray:
    """The key of decision CL7: each point's rank when the points are sorted by their coordinates (the first
    column first) and then by their index. rank[i] is point i's place in that order."""
    X = np.asarray(X, float)
    order = np.lexsort(tuple([np.arange(X.shape[0])] + [X[:, j] for j in range(X.shape[1] - 1, -1, -1)]))
    rank = np.empty(X.shape[0], np.int64)
    rank[order] = np.arange(X.shape[0])
    return rank


def first_seen(labels) -> np.ndarray:
    """Labels renumbered 0 .. K - 1 by first appearance in the input's order, -1 kept (decision CL6)."""
    out = np.full(len(labels), -1, np.int64)
    seen = {}
    for i, v in enumerate(np.asarray(labels).tolist()):
        if v == -1:
            continue
        out[i] = seen.setdefault(v, len(seen))
    return out


# ---------------------------------------------------------------- point sets


@dataclasses.dataclass(frozen=True)
class Points:
    id: str
    X: np.ndarray  # (n, p) in the order given
    w: np.ndarray | None  # (n,) weights >= 0, or None
    truth: np.ndarray | None  # (n,) the generating labels (first-seen numbering), or None
    k: int | None  # the suggested number of clusters, or None
    Xe: np.ndarray  # (m, p) evaluation points: inside the data's box, at data points and outside it
    tags: frozenset
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def p(self) -> int:
        return int(self.X.shape[1])

    @property
    def digest(self) -> str:
        return digest(self.X, self.w, self.Xe)


def _xe(X, g, m=6):
    """Evaluation points of data X: random in the data's box, two data points, one outside each corner."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, np.maximum(np.abs(lo), 1.0))
    inside = lo + (hi - lo) * g.random((m, X.shape[1]))
    picks = X[np.unique(np.linspace(0, X.shape[0] - 1, 2).astype(int))]
    return np.vstack([inside, picks, lo - 0.25 * span, hi + 0.25 * span])


def _blobs(g, sizes, centers, scales):
    """Gaussian blobs: sizes[j] points around centers[j] with standard deviation scales[j] per axis; the
    points in a random order, and their blob numbers."""
    parts, labs = [], []
    for j, (m, c, s) in enumerate(zip(sizes, centers, scales)):
        parts.append(np.asarray(c, float) + np.asarray(s, float) * g.standard_normal((m, len(c))))
        labs.append(np.full(m, j))
    X, lab = np.vstack(parts), np.concatenate(labs)
    perm = g.permutation(len(lab))
    return X[perm], lab[perm]


def _point_table():
    t = {}

    def add(pid, X, tags, *, truth=None, k=None, w=None, note="", g=None):
        g = g or rng("xe", pid)
        X = np.ascontiguousarray(np.asarray(X, float))
        if X.ndim == 1:
            X = X[:, None]
        truth = None if truth is None else first_seen(truth)
        Xe = _xe(X, g)
        _frozen(X, w, truth, Xe)
        t[pid] = Points(pid, X, w, truth, k, Xe, frozenset(tags), note)

    # -- blobs
    for k, p, n_each, sep in ((2, 1, 20, 8.0), (3, 2, 30, 8.0), (5, 5, 30, 10.0), (10, 10, 30, 12.0), (4, 3, 15, 8.0)):
        g = rng("blobs", k, p)
        centers = sep * g.standard_normal((k, p))
        X, lab = _blobs(g, [n_each] * k, centers, [1.0] * k)
        add(f"blobs/k{k}p{p}", X, {"blobs", "separated"}, truth=lab, k=k, note=f"{k} blobs of {n_each}, sd 1")
    g = rng("blobs", "overlap")
    X, lab = _blobs(g, [40, 40, 40], [[0.0, 0.0], [2.0, 0.5], [0.5, 2.0]], [1.0, 1.0, 1.0])
    add("overlap/k3p2", X, {"blobs", "overlap"}, truth=lab, k=3, note="three blobs 2 apart, sd 1")
    g = rng("blobs", "sizes")
    X, lab = _blobs(g, [5, 30, 120], [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]], [0.3, 1.0, 1.5])
    add("sizes/k3p2", X, {"blobs", "sizes"}, truth=lab, k=3, note="blobs of 5, 30 and 120 points")
    g = rng("blobs", "elongated")
    A = np.array([[100.0, 0.0], [0.0, 1.0]]) @ np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    parts = [g.standard_normal((40, 2)) @ A.T * 0.05 + c for c in ([0.0, 0.0], [0.0, 3.0])]
    X = np.vstack(parts)
    lab = np.repeat([0, 1], 40)
    perm = g.permutation(80)
    add("elongated/k2p2", X[perm], {"blobs", "elongated"}, truth=lab[perm], k=2, note="covariance condition 1e4")
    g = rng("blobs", "density")
    X, lab = _blobs(g, [60, 30, 30], [[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]], [0.2, 1.0, 0.6])
    noise = np.column_stack([g.uniform(-3, 8, 12), g.uniform(-3, 8, 12)])
    add(
        "density/k3p2",
        np.vstack([X, noise]),
        {"blobs", "density"},
        truth=np.concatenate([lab, np.full(12, -1)]),
        k=3,
        note="blobs of sd 0.2, 1 and 0.6 and 12 uniform noise points (truth -1)",
    )
    g = rng("shapes", "rings")
    ang = 2 * np.pi * g.random(80)
    rad = np.where(np.arange(80) < 40, 1.0, 3.0) + 0.08 * g.standard_normal(80)
    add(
        "rings/p2",
        np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]),
        {"shapes"},
        truth=(np.arange(80) >= 40).astype(int),
        k=2,
        note="two concentric rings (radii 1 and 3)",
    )
    g = rng("shapes", "moons")
    s = np.pi * g.random(80)
    up = np.arange(80) < 40
    X = np.where(
        up[:, None], np.column_stack([np.cos(s), np.sin(s)]), np.column_stack([1 - np.cos(s), 0.5 - np.sin(s)])
    )
    add("moons/p2", X + 0.05 * g.standard_normal((80, 2)), {"shapes"}, truth=(~up).astype(int), k=2, note="two moons")
    g = rng("blobs", "big")
    X, lab = _blobs(g, [100, 100, 100], [[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]], [1.0, 1.0, 1.0])
    add("probe/300", X, {"blobs", "probe"}, truth=lab, k=3, note="the design probe's three blobs of 100")
    # -- ties
    add("line/7", np.arange(7.0), {"ties", "lattice"}, k=2, note="0, 1, ..., 6: every nearest distance 1")
    add("lattice/3x3", [[i, j] for i in range(3) for j in range(3)], {"ties", "lattice"}, k=3)
    add("lattice/4x4", [[i, j] for i in range(4) for j in range(4)], {"ties", "lattice"}, k=4)
    add("lattice/3x3x3", [[i, j, l] for i in range(3) for j in range(3) for l in range(3)], {"ties", "lattice"}, k=3)
    g = rng("ties", "lattice-shuffled")
    L = np.array([[i, j] for i in range(4) for j in range(4)], float)
    add("lattice/4x4-shuffled", L[g.permutation(16)], {"ties", "lattice", "shuffled"}, k=4, note="lattice/4x4 shuffled")
    g = rng("ties", "dups")
    base = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]])
    rep = np.repeat(np.arange(4), 5)
    add("dups/4x5", base[rep][g.permutation(20)], {"ties", "dups"}, k=4, note="four points, each five times")
    add("allsame/8", np.full((8, 2), 1.25), {"ties", "dups", "degenerate"}, k=1, note="one point eight times")
    hexagon = [[np.cos(np.pi * j / 3), np.sin(np.pi * j / 3)] for j in range(6)]
    add(
        "hexagon/7",
        np.vstack([[[0.0, 0.0]], np.round(np.array(hexagon), 12)]),
        {"ties", "polygon"},
        k=2,
        note="a regular hexagon's vertices (to 12 digits) and its centre",
    )
    add(
        "eps/6",
        [[0.0], [0.5], [1.0], [3.0], [3.5], [4.0]],
        {"ties", "eps"},
        k=2,
        note="dyadic points: with eps = 0.5 the neighbours are exactly eps apart",
    )
    # -- degenerate
    for n in (1, 2, 3):
        add(f"tiny/{n}", np.arange(n, dtype=float)[:, None] * [1.5, 0.5] + 0.25, {"degenerate", "tiny"}, k=n)
    g = rng("degenerate", "collinear")
    tline = g.random(30)
    add(
        "collinear/p2",
        np.column_stack([tline, 2.0 * tline + 1.0]),
        {"degenerate", "collinear"},
        k=2,
        note="points on y = 2x + 1 (singular covariance)",
    )
    g = rng("degenerate", "constcol")
    X, lab = _blobs(g, [15, 15], [[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]], [1.0, 1.0])
    X[:, 2] = 7.0
    add("constcol/p3", X, {"degenerate", "constcol"}, truth=lab, k=2, note="the third column constant")
    add(
        "fewdistinct/10",
        np.repeat([[0.0, 0.0], [1.0, 1.0]], 5, axis=0),
        {"degenerate", "dups"},
        k=3,
        note="two distinct points, k = 3 asked",
    )
    # -- scale
    g = rng("scale", "base")
    Xb, labb = _blobs(g, [20, 20, 20], [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], [1.0, 1.0, 1.0])
    add("narrow/k3p2", Xb * 1e-8, {"scale"}, truth=labb, k=3, note="scale-base blobs times 1e-8")
    add("shifted/k3p2", Xb + 1e8, {"scale", "shifted"}, truth=labb, k=3, note="the same blobs plus 1e8")
    add("huge/k3p2", Xb * 1e150, {"scale"}, truth=labb, k=3, note="the same blobs times 1e150")
    add(
        "subnormal/k3p2",
        np.round(Xb * 64) * 2.0**-1068,
        {"scale", "subnormal"},
        truth=labb,
        k=3,
        note="the blobs rounded to 1/64 times 2^-1068 (subnormal coordinates)",
    )
    # -- inversion
    add("inversion/3", [[0.0, 0.0], [2.0, 0.0], [1.0, 1.8]], {"inversion"}, note="centroid heights 2.0 then 1.8")
    add("inversion/4", [[0.0, 0.0], [2.0, 0.0], [1.0, 1.8], [8.0, 0.0]], {"inversion"}, k=3)
    g = rng("inversion", "random")
    add("inversion/random30", g.standard_normal((30, 2)), {"inversion", "random"}, note="random points: inversions")
    # -- weights
    g = rng("weights", "blobs")
    X, lab = _blobs(g, [20, 20], [[0.0, 0.0], [6.0, 0.0]], [1.0, 1.0])
    w = 0.5 + 1.5 * g.random(40)
    w[[3, 11, 29]] = 0.0
    _frozen(w)
    add("weights/k2p2", X, {"weights"}, truth=lab, k=2, w=w, note="weights in [0.5, 2], three of them 0")
    return t


@functools.cache
def _points():
    return _point_table()


def points(pid: str) -> Points:
    return _points()[pid]


def point_ids(tag=None, exclude=(), max_n=None) -> list[str]:
    return [
        k
        for k, s in _points().items()
        if (tag is None or tag in s.tags) and not (set(exclude) & s.tags) and (max_n is None or s.n <= max_n)
    ]


# ---------------------------------------------------------------- labelings (the external indices)


@dataclasses.dataclass(frozen=True)
class Labeling:
    id: str
    a: np.ndarray  # labels_true (int64, or str for the string kind)
    b: np.ndarray  # labels_pred
    tags: frozenset
    note: str = ""

    @property
    def n(self) -> int:
        return int(len(self.a))

    @property
    def digest(self) -> str:
        return digest(np.asarray(self.a).astype("U") if self.a.dtype.kind == "U" else self.a, self.b)


def _labeling_table():
    t = {}

    def add(lid, a, b, tags, note=""):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype.kind in "iub":
            a = a.astype(np.int64)
        if b.dtype.kind in "iub":
            b = b.astype(np.int64)
        _frozen(a, b)
        t[lid] = Labeling(lid, a, b, frozenset(tags), note)

    add("small/6", [0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 3, 3], {"small"}, note="the design probe's pair")
    add("equal/12", [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [5, 5, 5, 0, 0, 0, 9, 9, 9, 2, 2, 2], {"equal"})
    add("one-one/6", [0] * 6, [0] * 6, {"special"}, note="one cluster on both sides")
    add("unique-unique/6", list(range(6)), list(range(6)), {"special"}, note="all singletons on both sides")
    add("one-unique/6", [0] * 6, list(range(6)), {"special"}, note="one cluster against all singletons")
    add("two/2", [0, 1], [0, 0], {"special", "tiny"})
    add("single/1", [3], [7], {"special", "tiny"})
    g = rng("labels", "random")
    for n, ka, kb in ((50, 3, 4), (200, 5, 5), (1000, 10, 7)):
        add(f"random/{n}", g.integers(0, ka, n), g.integers(0, kb, n), {"random"}, note="independent random")
    g = rng("labels", "refine")
    a = g.integers(0, 4, 120)
    add("refine/120", a, a * 3 + g.integers(0, 3, 120), {"refine"}, note="b refines a")
    g = rng("labels", "strings")
    names = np.array(["setosa", "versicolor", "virginica"])
    add("strings/30", names[g.integers(0, 3, 30)], names[g.integers(0, 3, 30)], {"strings"})
    g = rng("labels", "large")
    a = g.integers(0, 20, 10**6)
    b = np.where(g.random(10**6) < 0.9, a, g.integers(0, 25, 10**6))
    add("large/1e6", a, b, {"large"}, note="10^6 points, b equal to a for about 90% of them")
    return t


@functools.cache
def _labelings():
    return _labeling_table()


def labeling(lid: str) -> Labeling:
    return _labelings()[lid]


def labeling_ids(tag=None, exclude=()) -> list[str]:
    return [k for k, s in _labelings().items() if (tag is None or tag in s.tags) and not (set(exclude) & s.tags)]


# ---------------------------------------------------------------- mixtures (Gaussian mixture parameters)


@dataclasses.dataclass(frozen=True)
class Mixture:
    id: str
    covariance_type: str
    weights: np.ndarray  # (k,)
    means: np.ndarray  # (k, p)
    covariances: np.ndarray  # sklearn's shapes: full (k, p, p), tied (p, p), diag (k, p), spherical (k,)
    X: np.ndarray  # (m, p) points the densities are evaluated at
    tags: frozenset
    note: str = ""

    @property
    def k(self) -> int:
        return int(self.weights.size)

    @property
    def p(self) -> int:
        return int(self.means.shape[1])

    @property
    def digest(self) -> str:
        return digest(self.weights, self.means, self.covariances, self.X)


def _spd(g, p, cond):
    """A symmetric positive definite p x p matrix of condition about cond, as doubles."""
    Q, _ = np.linalg.qr(g.standard_normal((p, p)))
    ev = np.geomspace(1.0, 1.0 / cond, p) if p > 1 else np.ones(1)
    S = (Q * ev) @ Q.T
    return (S + S.T) / 2


def _mixture_table():
    t = {}

    def add(mid, ctype, weights, means, covs, g, tags, note=""):
        weights = np.asarray(weights, float)
        means = np.asarray(means, float)
        covs = np.asarray(covs, float)
        k, p = means.shape
        full = {
            "full": lambda j: covs[j],
            "tied": lambda j: covs,
            "diag": lambda j: np.diag(covs[j]),
            "spherical": lambda j: np.eye(p) * covs[j],
        }[ctype]
        # two points per component drawn from it (0.7 of its spread), three far from every component
        near = [means[j] + 0.7 * np.linalg.cholesky(full(j)) @ g.standard_normal(p) for j in range(k) for _ in range(2)]
        X = np.vstack([np.array(near), means[0] + 12.0 * g.standard_normal((3, p))])
        _frozen(weights, means, covs, X)
        t[mid] = Mixture(mid, ctype, weights, means, covs, X, frozenset(tags), note)

    g = rng("mixture", "full")
    means = 3.0 * g.standard_normal((3, 2))
    add("full/k3p2", "full", [0.5, 0.3, 0.2], means, np.stack([_spd(g, 2, c) for c in (2, 10, 100)]), g, {"full"})
    g = rng("mixture", "full5")
    means = 3.0 * g.standard_normal((2, 5))
    add("full/k2p5", "full", [0.6, 0.4], means, np.stack([_spd(g, 5, c) for c in (10, 1e4)]), g, {"full", "ill"})
    g = rng("mixture", "tied")
    add("tied/k3p3", "tied", [0.2, 0.3, 0.5], 4.0 * g.standard_normal((3, 3)), _spd(g, 3, 50), g, {"tied"})
    g = rng("mixture", "diag")
    add("diag/k4p3", "diag", [0.25] * 4, 4.0 * g.standard_normal((4, 3)), 0.2 + g.random((4, 3)), g, {"diag"})
    g = rng("mixture", "spherical")
    add(
        "spherical/k3p2",
        "spherical",
        [0.4, 0.4, 0.2],
        3.0 * g.standard_normal((3, 2)),
        [0.5, 1.0, 2.0],
        g,
        {"spherical"},
    )
    g = rng("mixture", "one")
    add("full/k1p1", "full", [1.0], [[2.0]], [[[0.25]]], g, {"full", "tiny"})
    return t


@functools.cache
def _mixtures():
    return _mixture_table()


def mixture(mid: str) -> Mixture:
    return _mixtures()[mid]


def mixture_ids(tag=None) -> list[str]:
    return [k for k, s in _mixtures().items() if tag is None or tag in s.tags]
