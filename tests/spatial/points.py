"""The test points of sd.spatial (docs/design/spatial.md §7.1): generated from formulas and fixed
seeds, nothing downloaded. The tests, the reference tool (tools/spatial_ref.py), the probe of
scipy, scikit-learn and R (tools/spatial_probe.py) and the measurements take their points from
here, by id.

A point set is an (m, n) array (float64; bool or int64 for the sets of the counting distances)
and, for the neighbour sets, query points (q, n). Kinds, by tag:

    random       standard normal, uniform, a Gaussian mixture, a 2-D manifold in 64 dimensions; the
                 normal family at every length where a sum changes shape (the pair kernel's block
                 of 32, the pairwise leaf of 512 and its tree up to 2048, and past it)
    adversarial  what stresses the screening margin of §3.2 and the careful path: small clusters
                 far from the origin (1e8 + [0, 1)^n, 1e15 + integers), a shell of points whose
                 distances to the query differ by one ULP, heaps of identical points, coordinates
                 near the overflow (1e150, 1e160) and underflow (1e-160) of the squares
    lattice      integer and dyadic grids: every difference and sum of squares is exact, so a
                 distance is the square root of an exact number, correctly rounded -- the data of
                 §4 neighbours 2 (exact ties by the thousand)
    neighbors    the sets with queries (`queries`), for the neighbour references
    sphere       (latitude, longitude) sets with queries for the haversine neighbour references (the KD
                 tree on the unit sphere): uniform, near-identical clusters with duplicates, antipodes,
                 the poles, the dateline, exact ties
    binary       bool and int64 sets for hamming and jaccard (int64 values past 2^53)

and the pair families of the distance functions: `edge_pairs()` (NaN and inf in every position,
zero vectors, constant rows, 0/0 terms, negative coordinates), `weights()`, `variances()`,
`inverse_covariances()` (positive definite and not) and `haversine_pairs()` (antipodes, poles,
identical points).

    import tests.spatial.points as P
    s = P.get("lattice/int2d")       # cached; P.ids(tag="lattice") lists ids
    s.x, s.queries                   # numpy arrays (queries None for a set without them)
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math

import numpy as np

SEED = 20261122


@dataclasses.dataclass(frozen=True)
class PointSet:
    id: str
    x: np.ndarray  # (m, n)
    tags: frozenset
    queries: np.ndarray | None = None  # (q, n), the neighbour sets'
    note: str = ""

    @property
    def m(self) -> int:
        return int(self.x.shape[0])

    @property
    def n(self) -> int:
        return int(self.x.shape[1])


def hash_int(k) -> int:
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "little")


def rng(*key) -> np.random.Generator:
    """The generator of a key: the same numbers on every machine (PCG64 of SEED and the key)."""
    return np.random.Generator(np.random.PCG64([SEED, *[hash_int(k) for k in key]]))


def digest(*arrays) -> str:
    """SHA-256 of arrays as given (dtype, shape and bytes): the records and references name the
    points they were made from."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(repr((str(a.dtype), a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- builders

# the normal family's lengths: 0 features, the short pair kernels, the pair kernel's block of 32
# and the pairwise sums' leaf (512), tree (to 2048) and beyond
NORMAL_N = (0, 1, 2, 3, 8, 15, 16, 17, 31, 32, 33, 64, 100, 512, 513, 1024, 2048, 2049, 4096)


def _rows(n: int) -> int:
    return 64 if n <= 64 else (32 if n <= 1024 else 12)


def _normal(n):
    g = rng("normal", n)
    return g.standard_normal((_rows(n), n))


def _uniform(n):
    return rng("uniform", n).uniform(0, 1, (64, n))


def _mixture(n, m=300):
    g = rng("mixture", n)
    centres = g.normal(0, 10, (5, n))
    lab = g.integers(0, 5, m)
    return centres[lab] + g.normal(0, 0.5, (m, n))


def _manifold(n=64, m=200):
    g = rng("manifold", n)
    t = g.uniform(-1, 1, (m, 2))
    basis = g.standard_normal((2, n))
    curve = np.sin(3 * t[:, :1]) * g.standard_normal((1, n)) * 0.1
    return t @ basis + curve


def _far(base, n=3, m=200, integers=False, key="data"):
    g = rng("far", base, n, integers, key)
    if integers:
        return base + g.integers(0, 100, (m, n)).astype(np.float64)
    return base + g.uniform(0, 1, (m, n))


def _shell():
    """Points whose distance to the origin query is 1, 1 + 2^-52, 1 + 2 * 2^-52, ...: on the axes
    (exact), and their negatives; the query is the origin and the point (0.5, 0, 0)."""
    rows = []
    for k in range(16):
        d = 1.0 + k * 2.0**-52
        for axis in range(3):
            for sign in (1.0, -1.0):
                p = [0.0, 0.0, 0.0]
                p[axis] = sign * d
                rows.append(p)
    x = np.array(rows)
    order = rng("shell").permutation(len(x))
    return x[order], np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _dup(n=3, distinct=20, copies=10):
    g = rng("dup", n)
    base = g.integers(-5, 6, (distinct, n)).astype(np.float64)
    x = np.repeat(base, copies, axis=0)
    return x[g.permutation(len(x))], np.vstack([base[:5], base[:5] + 0.5])


def _ties():
    """Every integer point on the circles x^2 + y^2 = 25, 50, 65, 85 (12, 12, 16, 16 points: equal
    distances to the origin), shuffled."""
    pts = [(a, b) for r2 in (25, 50, 65, 85) for a in range(-10, 11) for b in range(-10, 11) if a * a + b * b == r2]
    x = np.array(pts, dtype=np.float64)
    x = x[rng("ties").permutation(len(x))]
    return x, np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [5.0, 5.0], [-3.0, 4.0]])


def _scaled(scale, n=3, m=64):
    return rng("scaled", scale, n).standard_normal((m, n)) * scale


def _lattice_queries(n, lo, hi, q, key):
    g = rng("lattice-q", key)
    a = g.integers(lo, hi + 1, (q // 2, n)).astype(np.float64)
    b = g.integers(lo, hi, (q - q // 2, n)) + 0.5
    return np.vstack([a, b])


def _int_grid(n, half):
    axes = [np.arange(-half, half + 1, dtype=np.float64)] * n
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n)
    return grid[rng("grid", n, half).permutation(len(grid))]


def _big_int(n, m):
    """Integers whose squared differences sum below 2^53 (|x| < 2^24 in 3 dimensions, 2^23 in 8): the
    sums of squares are exact doubles."""
    half = 2**24 if n <= 3 else 2**23
    return rng("bigint", n).integers(-half + 1, half, (m, n)).astype(np.float64)


def _dyadic(n=3, m=300):
    return rng("dyadic", n).integers(-64, 65, (m, n)) / 8.0


def _binary(kind, n, m=64):
    g = rng("binary", kind, n)
    if kind == "bool":
        return g.random((m, n)) < 0.3
    if kind == "int":
        v = g.integers(-3, 4, (m, n))
        v[g.random((m, n)) < 0.4] = 0
        return v.astype(np.int64)
    # int64 values past 2^53: neighbours 2^60 and 2^60 + 1 are distinct integers, one double
    v = np.full((m, n), 2**60, dtype=np.int64) + g.integers(0, 2, (m, n))
    v[g.random((m, n)) < 0.3] = 0
    return v


def _neighbors_normal(n, m, q, key):
    g = rng("nn", key, n)
    return g.standard_normal((m, n)), g.standard_normal((q, n))


LAT_MAX = 1.5707963267948966  # fl(pi / 2)


def _on_sphere(g, m, lon=(-math.pi, math.pi)):
    return np.column_stack([np.arcsin(g.uniform(-1.0, 1.0, m)), g.uniform(*lon, m)])


def _sphere(kind, m=400, q=20):
    """(data, queries) on the unit sphere, (latitude, longitude) in radians, inside the domain of the
    sphere path's margin (|latitude| <= fl(pi / 2), |longitude| <= 8)."""
    g = rng("sphere", kind)
    if kind == "uniform":
        return _on_sphere(g, m, (0.0, 2.0 * math.pi)), _on_sphere(g, q, (0.0, 2.0 * math.pi))
    if kind == "close":  # clusters of spread 1e-12 to 1e-6 rad, a fifth of them duplicates
        c = _on_sphere(g, 10)
        lab = g.integers(0, 10, m)
        x = c[lab] + g.normal(0.0, 1.0, (m, 2)) * 10.0 ** g.integers(-12, -5, (m, 1))
        x[: m // 5] = x[m // 5 : 2 * (m // 5)]
        return x, c[g.integers(0, 10, q)] + g.normal(0.0, 1e-9, (q, 2))
    if kind == "antipode":  # the queries' antipodes, moved by 0 to 1e-3 rad
        qq = _on_sphere(g, q)
        anti = np.column_stack([-qq[:, 0], np.where(qq[:, 1] > 0, qq[:, 1] - math.pi, qq[:, 1] + math.pi)])
        near = np.repeat(anti, m // (2 * q), axis=0)
        near = near + g.normal(0.0, 1.0, near.shape) * 10.0 ** g.integers(-14, -2, (len(near), 1))
        x = np.vstack([near, _on_sphere(g, m - len(near))])
        x[:, 0] = np.clip(x[:, 0], -LAT_MAX, LAT_MAX)
        return x, qq
    if kind == "poles":  # at and next to both poles, every longitude
        lat = g.choice([LAT_MAX, -LAT_MAX, LAT_MAX * (1 - 1e-16), -(LAT_MAX - 1e-9)], m // 2)
        x = np.vstack([np.column_stack([lat, g.uniform(-8.0, 8.0, m // 2)]), _on_sphere(g, m - m // 2)])
        return x, np.column_stack([g.choice([LAT_MAX, -LAT_MAX, LAT_MAX - 1e-7], q), g.uniform(-4.0, 4.0, q)])
    if kind == "dateline":  # longitudes next to pi and -pi
        side = g.uniform(0.0, 1e-6, m)
        lon = np.where(g.random(m) < 0.5, math.pi - side, -math.pi + side)
        x = np.column_stack([g.uniform(-1.0, 1.0, m), lon])
        return x, np.column_stack([g.uniform(-1.0, 1.0, q), g.choice([math.pi, -math.pi], q)])
    if kind == "ties":  # mirror images about the equator and a meridian of queries on the equator: equal h
        qq = np.column_stack([np.zeros(q), g.uniform(-3.0, 3.0, q)])
        a = g.uniform(0.0, 0.05, (m // 4, 1))
        base = qq[g.integers(0, q, m // 4)]
        x = np.vstack([base + np.hstack([a * 0, a]), base - np.hstack([a * 0, a]), base + np.hstack([a, a * 0]),
                       base - np.hstack([a, a * 0])])  # fmt: skip
        return x, qq
    raise KeyError(kind)


_BUILDERS = {}


def _register(id_, tags, build, note=""):
    _BUILDERS[id_] = (frozenset(tags), build, note)


for _n in NORMAL_N:
    _register(f"normal/{_n}", {"random"}, lambda n=_n: (_normal(n), None))
for _n in (2, 3, 8, 16):
    _register(f"uniform/{_n}", {"random"}, lambda n=_n: (_uniform(n), None))
for _n in (2, 3, 8):
    _register(f"mixture/{_n}", {"random"}, lambda n=_n: (_mixture(n), None))
_register("manifold/64", {"random"}, lambda: (_manifold(), None), "a 2-D sheet in 64 dimensions")
_register(
    "far/1e8", {"adversarial", "neighbors"}, lambda: (_far(1e8), _far(1e8, m=20, key="queries")), "1e8 + [0, 1)^3"
)
_register("far/1e8x16", {"adversarial"}, lambda: (_far(1e8, n=16), None), "1e8 + [0, 1)^16")
_register(
    "far/1e15", {"adversarial", "neighbors"},
    lambda: (_far(1e15, integers=True), _far(1e15, m=20, integers=True, key="queries")),
    "1e15 + integers in [0, 100)^3: exact differences",
)  # fmt: skip
_register("shell/3", {"adversarial", "neighbors"}, _shell, "distances 1 + k 2^-52 to the origin")
_register("dup/3", {"adversarial", "neighbors", "lattice"}, _dup, "20 integer points, 10 copies each")
_register("ties/2", {"adversarial", "neighbors", "lattice"}, _ties, "integer points on four circles")
_register("scaled/1e150", {"adversarial"}, lambda: (_scaled(1e150), None), "squares near 1e300")
_register("scaled/1e160", {"adversarial"}, lambda: (_scaled(1e160), None), "squares overflow: the careful path")
_register("scaled/1e-160", {"adversarial"}, lambda: (_scaled(1e-160), None), "squares underflow: the careful path")
_register(
    "lattice/int2d", {"lattice", "neighbors"}, lambda: (_int_grid(2, 10), _lattice_queries(2, -11, 11, 40, "int2d")),
    "every integer point of [-10, 10]^2",
)  # fmt: skip
_register(
    "lattice/int3d", {"lattice", "neighbors"}, lambda: (_int_grid(3, 4), _lattice_queries(3, -5, 5, 40, "int3d")),
    "every integer point of [-4, 4]^3",
)  # fmt: skip
_register(
    "lattice/bigint3", {"lattice", "neighbors"},
    lambda: (_big_int(3, 300), _big_int(3, 330)[300:]), "integers below 2^24 in magnitude",
)  # fmt: skip
_register("lattice/bigint8", {"lattice"}, lambda: (_big_int(8, 200), None), "integers below 2^23 in magnitude")
_register(
    "lattice/dyadic3", {"lattice", "neighbors"}, lambda: (_dyadic(), _dyadic(m=330)[300:]),
    "multiples of 1/8 in [-8, 8]^3",
)  # fmt: skip
_register("nn/normal3", {"random", "neighbors"}, lambda: _neighbors_normal(3, 1000, 50, "a"))
_register("nn/normal16", {"random", "neighbors"}, lambda: _neighbors_normal(16, 500, 30, "b"))
_register("nn/mixture8", {"random", "neighbors"}, lambda: (_mixture(8, 500), _mixture(8, 530)[500:]))
for _kind in ("uniform", "close", "antipode", "poles", "dateline", "ties"):
    _register(f"sphere/{_kind}", {"sphere"}, lambda k=_kind: _sphere(k))
for _kind, _n in (("bool", 7), ("bool", 64), ("bool", 200), ("int", 10), ("int64big", 10)):
    _register(f"binary/{_kind}{_n}", {"binary"}, lambda k=_kind, n=_n: (_binary(k, n), None))

IDS = tuple(_BUILDERS)


@functools.cache
def get(id_: str) -> PointSet:
    if id_ not in _BUILDERS:
        raise KeyError(f"unknown point set {id_!r}")
    tags, build, note = _BUILDERS[id_]
    x, q = build()
    x = np.ascontiguousarray(x)
    if q is not None:
        q = np.ascontiguousarray(q, dtype=x.dtype)
    return PointSet(id_, x, tags, q, note)


def ids(tag: str | None = None) -> list[str]:
    return [i for i in IDS if tag is None or tag in _BUILDERS[i][0]]


def set_digest(s: PointSet) -> str:
    return digest(s.x, s.queries)


def pairs(s: PointSet, limit: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive rows (x[i], x[i + 1]) for i < limit: the distance references' pairs of a set."""
    k = max(0, min(limit, s.m - 1))
    return s.x[:k], s.x[1 : k + 1]


# ---------------------------------------------------------------- distance-function inputs

NAN, INF = math.nan, math.inf


def edge_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(id, u, v) pairs at the edges of the distance definitions (docs/design/spatial.md §2.2)."""
    out = []
    base_u, base_v = np.array([1.0, -2.0, 3.0, 0.5]), np.array([0.5, 2.0, -1.0, 0.5])
    for pos in range(4):
        for val, name in ((NAN, "nan"), (INF, "inf"), (-INF, "-inf")):
            u = base_u.copy()
            u[pos] = val
            out.append((f"{name}@u{pos}", u, base_v.copy()))
    out.append(("inf-inf", np.array([INF, 1.0]), np.array([INF, 2.0])))
    out.append(("nan-both", np.array([NAN, 1.0]), np.array([NAN, 1.0])))
    out.append(("zero-zero", np.zeros(4), np.zeros(4)))
    out.append(("zero-u", np.zeros(4), base_v.copy()))
    out.append(("const-rows", np.full(4, 3.0), np.full(4, -1.0)))
    out.append(("const-u", np.full(4, 3.0), base_v.copy()))
    out.append(("canberra-0/0", np.array([0.0, 1.0, 0.0, -2.0]), np.array([0.0, 3.0, 0.0, 2.0])))
    out.append(("braycurtis-cancel", np.array([1.0, -2.0]), np.array([-1.0, 2.0])))
    out.append(("negative", np.array([-1.0, -2.0, -3.0]), np.array([-4.0, -6.0, -8.0])))
    out.append(("equal", base_u.copy(), base_u.copy()))
    out.append(("empty", np.zeros(0), np.zeros(0)))
    out.append(("one", np.array([2.5]), np.array([-1.0])))
    out.append(("huge", np.array([1e200, -1e200]), np.array([-1e200, 1e200])))
    out.append(("tiny", np.array([1e-200, 3e-200]), np.array([-2e-200, 1e-200])))
    out.append(("subnormal", np.array([5e-324, 0.0]), np.array([0.0, 1e-320])))
    return out


def weights(n: int, kind: str = "positive") -> np.ndarray:
    """A weight vector of length n: positive, or with zeros ('zeros'), or all ones ('ones')."""
    g = rng("w", n, kind)
    if kind == "ones":
        return np.ones(n)
    w = g.uniform(0.1, 3.0, n)
    if kind == "zeros":
        w[g.random(n) < 0.4] = 0.0
    return w


def variances(n: int) -> np.ndarray:
    return rng("V", n).uniform(0.2, 5.0, n)


def inverse_covariances(n: int, kind: str = "pd") -> np.ndarray:
    """An (n, n) matrix VI: symmetric positive definite ('pd') or symmetric indefinite ('indefinite'),
    with dyadic entries (multiples of 2^-10), so the quadratic form is exact in rationals."""
    g = rng("VI", n, kind)
    a = np.round(g.standard_normal((n, n)) * 256) / 1024
    s = a @ a.T  # dyadic, exact
    if kind == "pd":
        s = s + np.eye(n) * 0.5
    else:
        s = s - np.eye(n) * (np.round(np.trace(s) / n * 1024) / 1024)  # about half the eigenvalues negative
    return s


def haversine_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(id, u, v) of (latitude, longitude) pairs in radians: random, near and at the antipode, the
    poles, identical points, very close points."""
    g = rng("haversine")
    out = []
    for i in range(24):
        u = np.array([g.uniform(-math.pi / 2, math.pi / 2), g.uniform(-math.pi, math.pi)])
        v = np.array([g.uniform(-math.pi / 2, math.pi / 2), g.uniform(-math.pi, math.pi)])
        out.append((f"random{i}", u, v))
    for i, eps in enumerate((0.0, 1e-8, 1e-4, 1e-2)):
        lat, lon = g.uniform(-1.2, 1.2), g.uniform(-3.0, 0.0)
        out.append((f"antipode{i}", np.array([lat, lon]), np.array([-lat + eps, lon + math.pi - eps])))
    out.append(("poles", np.array([math.pi / 2, 0.0]), np.array([-math.pi / 2, 1.0])))
    out.append(("same", np.array([0.3, -1.1]), np.array([0.3, -1.1])))
    out.append(("close", np.array([0.3, -1.1]), np.array([0.3 + 1e-12, -1.1 - 1e-12])))
    out.append(("equator", np.array([0.0, 0.0]), np.array([0.0, math.pi / 2])))
    return out
