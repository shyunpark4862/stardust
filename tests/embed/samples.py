"""The test data of sd.embed (docs/design/embed.md §7.1): generated from formulas and fixed seeds, nothing
downloaded. The tests, the reference tool (tools/embed_ref.py), the probe of the comparison libraries
(tools/embed_probe.py), the prototype measurements and the golden cases take their data from here, by id.

One kind, point sets (X (n, p), the generating labels or manifold coordinate when there are any, a suggested
n_components, evaluation points Xe), by tag:

    linear      low rank plus noise (rank 1 to 20, p = 2 to 200), equal variances on two axes (a repeated
                eigenvalue), a very small eigenvalue gap, a large shift (1e8 + data: the centring's cancellation),
                tiny and huge scales (1e-150, 1e150), a constant column, n < p, sparse-like data (mostly zeros),
                nonnegative data (NMF), mixed independent sources (uniform, Laplace, sine: ICA), data drawn from a
                factor model (factor analysis)
    manifold    the Swiss roll, the S curve, a circle and a ring, a sphere, two far apart blobs (a disconnected
                neighbour graph), an integer lattice (distance ties), repeated points, points on one line
    neighbour   Gaussian blobs (k = 3 to 10, p = 10 to 50), blobs of very different sizes, synthetic 8 x 8
                "digits" (ten stroke templates drawn with noise: a formula, not a dataset)
    degenerate  n = 1, 2, 3, fewer distinct points than n_components, every point the same
    large       sizes that make the parallel paths run (golden cases and measurements)

    import tests.embed.samples as S
    s = S.points("lowrank/r3p10")     # cached; S.point_ids(tag="manifold") lists ids
    s.X, s.truth, s.t, s.d, s.Xe       # numpy arrays (truth, t None when there are none)

Every record and reference names the data it was made from by `digest`. The key of decision EM12 (cluster CL7) is
lex_rank().
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib

import numpy as np

SEED = 20261127


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
    """The key of decision EM12 (cluster CL7): each point's rank when the points are sorted by their coordinates
    (the first column first) and then by their index. rank[i] is point i's place in that order."""
    X = np.asarray(X, float)
    order = np.lexsort(tuple([np.arange(X.shape[0])] + [X[:, j] for j in range(X.shape[1] - 1, -1, -1)]))
    rank = np.empty(X.shape[0], np.int64)
    rank[order] = np.arange(X.shape[0])
    return rank


@dataclasses.dataclass(frozen=True)
class Points:
    id: str
    X: np.ndarray  # (n, p) in the order given
    truth: np.ndarray | None  # (n,) generating labels (blobs, digits), or None
    t: np.ndarray | None  # (n,) or (n, 2) the manifold's own coordinates (Swiss roll, S curve, ...), or None
    d: int  # the suggested n_components
    Xe: np.ndarray  # (m, p) evaluation points (transform): inside the data's box, at data points, outside it
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
        return digest(self.X, self.Xe)


def _xe(X, g, m=6):
    """Evaluation points of data X: random in the data's box, two data points, one outside each corner."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, np.maximum(np.abs(lo), 1.0))
    inside = lo + (hi - lo) * g.random((m, X.shape[1]))
    picks = X[np.unique(np.linspace(0, X.shape[0] - 1, 2).astype(int))]
    return np.vstack([inside, picks, lo - 0.25 * span, hi + 0.25 * span])


def _blobs(g, sizes, centers, scales):
    """Gaussian blobs: sizes[j] points around centers[j] with standard deviation scales[j] per axis; the points in
    a random order, and their blob numbers."""
    parts, labs = [], []
    for j, (m, c, s) in enumerate(zip(sizes, centers, scales)):
        parts.append(np.asarray(c, float) + np.asarray(s, float) * g.standard_normal((m, len(c))))
        labs.append(np.full(m, j))
    X, lab = np.vstack(parts), np.concatenate(labs)
    perm = g.permutation(len(lab))
    return X[perm], lab[perm]


def _lowrank(g, n, p, r, noise, decay=0.7):
    """Rank-r data with singular values 10 decay^j along a random orthonormal basis, plus isotropic noise."""
    U = np.linalg.qr(g.standard_normal((n, r)))[0]
    V = np.linalg.qr(g.standard_normal((p, r)))[0]
    s = 10.0 * decay ** np.arange(r) * np.sqrt(n)
    return (U * s) @ V.T + noise * g.standard_normal((n, p)) + g.standard_normal(p)


def swiss_roll(n, g, noise=0.0, hole=False):
    """scikit-learn's make_swiss_roll formula (t = 1.5 pi (1 + 2 u), height 21 v): (x, y, z) = (t cos t, 21 v,
    t sin t); returns (X, (t, height))."""
    u = g.random(n)
    if hole:
        u = np.where((u > 0.4) & (u < 0.6), u * 0.4 / 0.6, u)
    t = 1.5 * np.pi * (1 + 2 * u)
    h = 21.0 * g.random(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)]) + noise * g.standard_normal((n, 3))
    return X, np.column_stack([t, h])


def s_curve(n, g, noise=0.0):
    """scikit-learn's make_s_curve formula: t = 3 pi (u - 1/2), (sin t, 2 v, sign(t) (cos t - 1))."""
    t = 3 * np.pi * (g.random(n) - 0.5)
    h = 2.0 * g.random(n)
    X = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1)]) + noise * g.standard_normal((n, 3))
    return X, np.column_stack([t, h])


_STROKES = {  # ten 8 x 8 stroke templates (row, column) of a seven-segment-like digit
    "top": [(1, c) for c in range(2, 6)],
    "mid": [(4, c) for c in range(2, 6)],
    "bot": [(7, c) for c in range(2, 6)],
    "ul": [(r, 1) for r in range(1, 5)],
    "ur": [(r, 6) for r in range(1, 5)],
    "ll": [(r, 1) for r in range(4, 8)],
    "lr": [(r, 6) for r in range(4, 8)],
}
_DIGITS = {
    0: "top ul ur ll lr bot",
    1: "ur lr",
    2: "top ur mid ll bot",
    3: "top ur mid lr bot",
    4: "ul ur mid lr",
    5: "top ul mid lr bot",
    6: "top ul mid ll lr bot",
    7: "top ur lr",
    8: "top ul ur mid ll lr bot",
    9: "top ul ur mid lr bot",
}


def digits(n, g, noise=2.0):
    """Synthetic 8 x 8 digits with values 0..16 like scikit-learn's load_digits: digit j's template (16 on its
    strokes), shifted by up to one pixel, plus Gaussian noise of standard deviation `noise`, clipped to [0, 16].
    Returns (X (n, 64), labels)."""
    tmpl = np.zeros((10, 8, 8))
    for j, names in _DIGITS.items():
        for s in names.split():
            for r, c in _STROKES[s]:
                tmpl[j, r, c] = 16.0
    lab = g.integers(0, 10, n)
    out = np.empty((n, 8, 8))
    shifts = g.integers(-1, 2, (n, 2))
    for i in range(n):
        out[i] = np.roll(np.roll(tmpl[lab[i]], shifts[i, 0], axis=0), shifts[i, 1], axis=1)
    out += noise * g.standard_normal(out.shape)
    return np.clip(out, 0.0, 16.0).reshape(n, 64), lab


def _point_table():
    t = {}

    def add(pid, X, tags, *, d=2, truth=None, tt=None, note="", g=None):
        g = g or rng("xe", pid)
        X = np.ascontiguousarray(np.asarray(X, float))
        if X.ndim == 1:
            X = X[:, None]
        truth = None if truth is None else np.ascontiguousarray(truth, np.int64)
        tt = None if tt is None else np.ascontiguousarray(tt, float)
        Xe = _xe(X, g)
        _frozen(X, truth, tt, Xe)
        t[pid] = Points(pid, X, truth, tt, d, Xe, frozenset(tags), note)

    # -- linear
    for r, n, p, noise in (
        (1, 40, 2, 0.1),
        (3, 60, 10, 0.05),
        (5, 120, 30, 0.1),
        (20, 300, 50, 0.01),
        (2, 50, 200, 0.1),
    ):
        g = rng("lowrank", r, p)
        add(
            f"lowrank/r{r}p{p}",
            _lowrank(g, n, p, r, noise),
            {"linear", "lowrank"},
            d=r,
            note=f"rank {r} + noise {noise}",
        )
    g = rng("linear", "equal")
    X = g.standard_normal((80, 4)) * [3.0, 3.0, 1.0, 0.2]
    add(
        "equalvar/p4",
        X @ np.linalg.qr(g.standard_normal((4, 4)))[0],
        {"linear", "repeated"},
        d=2,
        note="two axes of equal variance (a repeated eigenvalue), rotated",
    )
    g = rng("linear", "gap")
    X = g.standard_normal((100, 5)) * [2.0, 1.0 + 1e-7, 1.0, 0.5, 0.1]
    add("tinygap/p5", X, {"linear", "gap"}, d=2, note="standard deviations 1 + 1e-7 and 1: a very small gap")
    g = rng("linear", "base")
    Xb = _lowrank(g, 60, 6, 3, 0.1)
    add("shifted/p6", Xb + 1e8, {"linear", "scale", "shifted"}, d=3, note="lowrank r3 plus 1e8 (centring cancels)")
    add("tiny/p6", Xb * 1e-150, {"linear", "scale"}, d=3, note="lowrank r3 times 1e-150")
    add("huge/p6", Xb * 1e150, {"linear", "scale"}, d=3, note="lowrank r3 times 1e150")
    Xc = Xb.copy()
    Xc[:, 2] = 7.0
    add("constcol/p6", Xc, {"linear", "constcol"}, d=3, note="the third column constant")
    g = rng("linear", "wide")
    add("wide/n10p40", _lowrank(g, 10, 40, 4, 0.1), {"linear", "wide"}, d=4, note="n < p")
    g = rng("linear", "sparse")
    Xs = g.standard_normal((100, 50)) * (g.random((100, 50)) < 0.05)
    add("sparse/n100p50", Xs, {"linear", "sparse"}, d=5, note="5% of the entries nonzero")
    g = rng("linear", "nonneg")
    W, H = g.random((80, 4)), g.random((4, 20))
    add("nonneg/r4", W @ H + 0.01 * g.random((80, 20)), {"linear", "nonnegative"}, d=4, note="W H + small, >= 0")
    g = rng("linear", "ica")
    tt = np.linspace(0.0, 8.0, 400)
    Sx = np.column_stack([np.sin(2 * tt), np.sign(np.sin(3 * tt)), g.laplace(size=400), g.uniform(-1, 1, 400)])
    A = np.array([[1.0, 1.0, 1.0, 0.5], [0.5, 2.0, 1.0, 1.0], [1.5, 1.0, 2.0, 0.3], [0.2, 0.4, 0.6, 1.0]])
    add("ica/s4", Sx @ A.T, {"linear", "ica"}, d=4, tt=tt, note="sine, square, Laplace, uniform sources mixed")
    g = rng("linear", "factor")
    L = g.uniform(0.4, 0.9, (8, 2)) * np.sign(g.standard_normal((8, 2)))
    L[:4, 1] *= 0.2
    L[4:, 0] *= 0.2
    psi = 1 - (L**2).sum(axis=1)
    F = g.standard_normal((200, 2))
    add(
        "factor/p8k2",
        F @ L.T + np.sqrt(psi) * g.standard_normal((200, 8)),
        {"linear", "factor"},
        d=2,
        note="two-factor model, uniquenesses 1 - |l|^2",
    )
    # -- manifold
    g = rng("manifold", "swiss")
    X, tt = swiss_roll(300, g, noise=0.05)
    add("swiss/300", X, {"manifold", "swiss"}, tt=tt, note="Swiss roll, noise 0.05")
    g = rng("manifold", "scurve")
    X, tt = s_curve(300, g, noise=0.02)
    add("scurve/300", X, {"manifold", "scurve"}, tt=tt, note="S curve, noise 0.02")
    a = 2 * np.pi * np.arange(60) / 60
    add(
        "circle/60",
        np.column_stack([np.cos(a), np.sin(a)]),
        {"manifold", "circle", "ties"},
        tt=a,
        note="60 evenly spaced points on the unit circle (equal neighbour distances)",
    )
    g = rng("manifold", "ring")
    a = 2 * np.pi * g.random(150)
    rad = 2.0 + 0.2 * g.standard_normal(150)
    add(
        "ring/150",
        np.column_stack([rad * np.cos(a), rad * np.sin(a), 0.1 * g.standard_normal(150)]),
        {"manifold", "ring"},
        tt=a,
        note="a noisy ring in 3-D",
    )
    g = rng("manifold", "sphere")
    v = g.standard_normal((200, 3))
    add("sphere/200", v / np.linalg.norm(v, axis=1, keepdims=True), {"manifold", "sphere"}, note="uniform on S^2")
    g = rng("manifold", "twoblobs")
    X, lab = _blobs(g, [40, 40], [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], [1.0, 1.0])
    add(
        "twoblobs/80",
        X,
        {"manifold", "disconnected", "neighbour"},
        truth=lab,
        note="two blobs 100 apart: a disconnected neighbour graph",
    )
    add(
        "lattice/5x5",
        [[i, j] for i in range(5) for j in range(5)],
        {"manifold", "ties", "lattice"},
        note="integer lattice: distance ties everywhere",
    )
    g = rng("manifold", "lattice-shuffled")
    Lt = np.array([[i, j] for i in range(5) for j in range(5)], float)
    add(
        "lattice/5x5-shuffled",
        Lt[g.permutation(25)],
        {"manifold", "ties", "lattice", "shuffled"},
        note="lattice/5x5 shuffled (decision EM12)",
    )
    g = rng("manifold", "dups")
    base = g.standard_normal((10, 3))
    add(
        "dups/10x4",
        base[np.repeat(np.arange(10), 4)][g.permutation(40)],
        {"manifold", "ties", "dups"},
        note="ten points, each four times",
    )
    g = rng("manifold", "line")
    s = g.random(40)
    add(
        "collinear/p3",
        np.column_stack([s, 2 * s + 1, -s]),
        {"manifold", "degenerate", "collinear"},
        d=2,
        note="points on one line in 3-D (rank 1)",
    )
    # -- neighbour embedding
    for k, p, each in ((3, 10, 50), (5, 20, 40), (10, 50, 30)):
        g = rng("blobs", k, p)
        X, lab = _blobs(g, [each] * k, 6.0 * g.standard_normal((k, p)), [1.0] * k)
        add(f"blobs/k{k}p{p}", X, {"neighbour", "blobs"}, truth=lab, note=f"{k} blobs of {each}, sd 1")
    g = rng("blobs", "sizes")
    X, lab = _blobs(g, [10, 50, 240], 8.0 * g.standard_normal((3, 10)), [0.5, 1.0, 2.0])
    add("sizes/k3p10", X, {"neighbour", "blobs", "sizes"}, truth=lab, note="blobs of 10, 50 and 240 points")
    g = rng("digits", 500)
    X, lab = digits(500, g)
    add("digits/500", X, {"neighbour", "digits"}, truth=lab, note="synthetic 8 x 8 digits, 500 images")
    # -- degenerate
    for n in (1, 2, 3):
        add(f"tiny/{n}", np.arange(n, dtype=float)[:, None] * [1.5, 0.5, -1.0] + 0.25, {"degenerate", "tiny"}, d=1)
    add(
        "fewdistinct/12",
        np.repeat([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]], 6, axis=0),
        {"degenerate", "dups"},
        d=3,
        note="two distinct points, n_components = 3 asked",
    )
    add("allsame/8", np.full((8, 3), 1.25), {"degenerate", "dups"}, d=2, note="one point eight times")
    # -- large (the parallel paths)
    g = rng("large", "lowrank")
    add("large/lowrank", _lowrank(g, 20000, 40, 8, 0.1), {"large", "linear"}, d=8, note="20000 x 40, rank 8")
    g = rng("large", "blobs")
    X, lab = _blobs(g, [2000] * 5, 6.0 * g.standard_normal((5, 20)), [1.0] * 5)
    add("large/blobs", X, {"large", "neighbour"}, truth=lab, note="5 blobs of 2000 in 20-D")
    g = rng("large", "swiss")
    X, tt = swiss_roll(5000, g, noise=0.05)
    add("large/swiss", X, {"large", "manifold"}, tt=tt, note="Swiss roll of 5000")
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
