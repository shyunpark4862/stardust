"""The test data of sd.interpolate (docs/design/interpolate.md §7.1): generated from formulas and fixed
seeds, nothing downloaded. The tests, the reference tool (tools/interpolate_ref.py), the probe of scipy,
numpy and R (tools/interpolate_probe.py) and the measurements take their data from here, by id.

Kinds, each with its own getter:

    curves   1-D data (x, y) with evaluation points xe: uniform, random and Chebyshev nodes, spacings
             that differ by 1e10, 2 to 5 nodes (the boundary conditions' special cases), monotone data,
             flat stretches, a step, Runge's function, values near 1e300 and 1e-300, cancellation
             (1e15 + small changes in x or y), subnormal spacing, data ready for 'periodic' (y[0] ==
             y[-1]), complex values and a batch of columns
    lerps    single intervals (x0, x1, a, b) and points x for the linear kernel (§3.2): the doubles next
             to both ends, endpoint values of opposite signs, a == b, signed zeros, inf and NaN ends, a
             subnormal width, a width that overflows, points outside the interval
    grids    regular grids (points, values, xi) in 1 to 4 dimensions, an axis of 1 and of 2 points, a
             decreasing axis, NaN values, trailing batch axes
    scatters scattered data (y, d, xi) in 2 and 3 dimensions: uniform, clustered, duplicated points,
             collinear points (the degree-1 monomials lose rank), a lattice with many equal distances
    roots    piecewise polynomials (c, x) for roots and solve: a tangent root, two roots 2^-30 apart,
             a piece that is zero, a sign change at a breakpoint, random cubic and quintic pieces
    polys    nodes and values for the polynomial interpolants: Chebyshev and uniform nodes, nodes
             1e-8 apart, repeated nodes (Krogh's derivative data)
    aaas     samples for AAA: tan, |x|, a function on the unit circle (complex), samples with NaN

    import tests.interpolate.samples as S
    c = S.curve("uniform/6")         # cached; S.curve_ids(tag="periodic") lists ids
    c.x, c.y, c.xe                   # numpy arrays

Every record and reference names the data it was made from by `digest`.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math

import numpy as np

SEED = 20261123
NAN, INF = math.nan, math.inf


def hash_int(k) -> int:
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "little")


def rng(*key) -> np.random.Generator:
    """The generator of a key: the same numbers on every machine (PCG64 of SEED and the key)."""
    return np.random.Generator(np.random.PCG64([SEED, *[hash_int(k) for k in key]]))


def digest(*arrays) -> str:
    """SHA-256 of arrays as given (dtype, shape and bytes, a tuple of arrays one after another)."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        if isinstance(a, (tuple, list)):
            h.update(b"seq%d" % len(a))
            for b in a:
                h.update(digest(b).encode())
            continue
        a = np.ascontiguousarray(a)
        h.update(repr((str(a.dtype), a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- curves


@dataclasses.dataclass(frozen=True)
class Curve:
    id: str
    x: np.ndarray  # (n,), strictly increasing
    y: np.ndarray  # (n,) or (n, m); float64 or complex128
    xe: np.ndarray  # evaluation points, sorted; some outside [x[0], x[-1]]
    tags: frozenset
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def digest(self) -> str:
        return digest(self.x, self.y, self.xe)


def _eval_points(x, key, *, node_limit=24, inside=24):
    """The nodes (up to node_limit of them, spread), the doubles on both sides of those nodes, the
    midpoints of their intervals, random points inside, and points outside both ends."""
    g = rng("xe", key)
    n = x.size
    pick = np.unique(np.linspace(0, n - 1, min(n, node_limit)).round().astype(int))
    nodes = x[pick]
    pts = [nodes, np.nextafter(nodes, -INF), np.nextafter(nodes, INF)]
    mids = pick[pick < n - 1]
    pts.append(x[mids] + (x[mids + 1] - x[mids]) / 2)
    lo, hi = x[0], x[-1]
    pts.append(lo + (hi - lo) * g.random(inside))
    span = hi - lo
    pts.append(np.array([lo - 0.25 * span, lo - 0.01 * span, hi + 0.01 * span, hi + 0.25 * span]))
    out = np.unique(np.concatenate(pts))
    return out[np.isfinite(out)]


def _uniform_x(n, lo=0.0, hi=1.0):
    return lo + (hi - lo) * (np.arange(n) / (n - 1))


def _random_x(n, key):
    g = rng("random-x", key)
    while True:
        x = np.sort(g.random(n))
        if np.all(np.diff(x) > 0):
            return x


def _normal_y(n, key, m=None):
    g = rng("normal-y", key)
    return g.standard_normal(n if m is None else (n, m))


def _spacing_x(n=12, ratio=1e10):
    h = np.where(np.arange(n - 1) % 2 == 0, 1.0, 1.0 / ratio)
    return np.concatenate([[0.0], np.cumsum(h)])


def _chebyshev_x(n):
    k = np.arange(n)
    return np.sort(np.cos(np.pi * (2 * k + 1) / (2 * n)))


def _monotone(n, key, sign=1.0, flats=False):
    g = rng("monotone", key)
    steps = g.exponential(1.0, n - 1)
    if flats:
        steps[g.random(n - 1) < 0.3] = 0.0
    y = np.concatenate([[0.0], np.cumsum(steps)]) * sign
    return _random_x(n, key), y


def _flat():
    x = _uniform_x(14)
    y = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.5, 0.0, 0.0, 0.0, 1.0, 1.0])
    return x, y


def _periodic_sin(n=16):
    x = _uniform_x(n, 0.0, 2 * math.pi)
    y = np.sin(x)
    y[-1] = y[0]
    return x, y


def _periodic_random(n=9):
    x, y = _random_x(n, "periodic"), _normal_y(n, "periodic")
    y[-1] = y[0]
    return x, y


def _subnormal_x(n=8):
    return np.arange(n) * 7 * 5e-324 + 3 * 5e-324


_CURVES: dict = {}


def _curve(id_, tags, build, note=""):
    _CURVES[id_] = (frozenset(tags), build, note)


for _n in (6, 50, 200):
    _curve(f"uniform/{_n}", {"random"}, lambda n=_n: (_uniform_x(n), _normal_y(n, f"u{n}")))
_curve("random/50", {"random"}, lambda: (_random_x(50, "r50"), _normal_y(50, "r50")))
_curve("spacing/1e10", {"adversarial"}, lambda: (_spacing_x(), np.sin(np.arange(12.0))), "h alternates 1 and 1e-10")
_curve("chebyshev/21", {"random"}, lambda: (_chebyshev_x(21), np.exp(_chebyshev_x(21))), "exp at Chebyshev nodes")
for _n in (2, 3, 4, 5):
    _curve(f"small/{_n}", {"small"}, lambda n=_n: (_random_x(n, f"s{n}"), _normal_y(n, f"s{n}")))
_curve("monotone/up30", {"monotone"}, lambda: _monotone(30, "up"))
_curve("monotone/down30", {"monotone"}, lambda: _monotone(30, "down", -1.0))
_curve("monotone/flats30", {"monotone", "flat"}, lambda: _monotone(30, "flats", 1.0, True), "steps of 0 among them")
_curve("flat/akima", {"flat"}, _flat, "runs of three and more equal values")
_curve("step/10", {"adversarial"}, lambda: (_uniform_x(10), (np.arange(10) >= 5).astype(np.float64)))
_curve("runge/21", {"random"}, lambda: (_uniform_x(21, -1, 1), 1 / (1 + 25 * _uniform_x(21, -1, 1) ** 2)))
_curve("large/1e300", {"adversarial"}, lambda: (_random_x(12, "l"), 1e300 * _normal_y(12, "l")))
_curve("tiny/1e-300", {"adversarial"}, lambda: (_random_x(12, "t"), 1e-300 * _normal_y(12, "t")))
_curve("cancel/y1e15", {"adversarial"}, lambda: (_uniform_x(16), 1e15 + _normal_y(16, "cy")), "1e15 + N(0, 1)")
_curve("cancel/x1e15", {"adversarial"}, lambda: (1e15 + 0.25 * np.arange(16), _normal_y(16, "cx")),
       "x = 1e15 + k/4: exact, the differences are exact too")  # fmt: skip
_curve("subnormal/x", {"adversarial"}, lambda: (_subnormal_x(), _normal_y(8, "sub")), "subnormal x and spacing")
_curve("periodic/sin16", {"periodic"}, _periodic_sin, "sin on [0, 2 pi], y[-1] = y[0]")
_curve("periodic/random9", {"periodic"}, _periodic_random)
_curve("complex/20", {"complex"}, lambda: (_uniform_x(20), _normal_y(20, "cre") + 1j * _normal_y(20, "cim")))
_curve("batch/20x3", {"batch"}, lambda: (_random_x(20, "b"), _normal_y(20, "b", 3)), "y has 3 columns")

CURVE_IDS = tuple(_CURVES)


@functools.cache
def curve(id_: str) -> Curve:
    if id_ not in _CURVES:
        raise KeyError(f"unknown curve {id_!r}")
    tags, build, note = _CURVES[id_]
    x, y = build()
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y)
    return Curve(id_, x, y, _eval_points(x, id_), tags, note)


def curve_ids(tag: str | None = None) -> list[str]:
    return [i for i in CURVE_IDS if tag is None or tag in _CURVES[i][0]]


def real_curve_ids() -> list[str]:
    """The curves with one real column: the ones the references cover."""
    return [i for i in CURVE_IDS if not ({"complex", "batch"} & _CURVES[i][0])]


# ---------------------------------------------------------------- linear kernel


@dataclasses.dataclass(frozen=True)
class Lerp:
    id: str
    x0: float
    x1: float
    a: float
    b: float
    x: np.ndarray  # the points

    @property
    def digest(self) -> str:
        return digest(np.array([self.x0, self.x1, self.a, self.b]), self.x)


def _neighbours(v, k):
    out, lo, hi = [v], v, v
    for _ in range(k):
        lo, hi = math.nextafter(lo, -INF), math.nextafter(hi, INF)
        out += [lo, hi]
    return out


def _lerp_points(x0, x1, key, k=6, inside=8, outside=False):
    g = rng("lerp-x", key)
    pts = _neighbours(x0, k) + _neighbours(x1, k) + list(x0 + (x1 - x0) * g.random(inside))
    if not outside:
        pts = [p for p in pts if x0 <= p <= x1]
    return np.unique(np.array(pts))


@functools.cache
def lerps() -> tuple[Lerp, ...]:
    out = []
    g = rng("lerps")
    for i in range(24):  # random intervals and values, both signs
        x0 = g.standard_normal() * 10.0 ** g.integers(-3, 4)
        x1 = x0 + g.exponential() * 10.0 ** g.integers(-3, 4)
        a, b = g.standard_normal(2) * 10.0 ** g.integers(-5, 6, 2)
        out.append(Lerp(f"random{i}", x0, x1, a, b, _lerp_points(x0, x1, ("r", i))))
    for i in range(8):  # same sign, close values: the a + t (b - a) branch
        a = g.uniform(1, 2)
        b = a * (1 + g.uniform(-1, 1) * 2.0 ** -g.integers(1, 50))
        out.append(Lerp(f"close{i}", 0.0, 1.0, a, b, _lerp_points(0.0, 1.0, ("c", i))))
    for i in range(8):  # opposite signs: the t b + (1 - t) a branch
        a, b = -g.exponential(), g.exponential()
        out.append(Lerp(f"opposite{i}", 1.0, 3.0, a, b, _lerp_points(1.0, 3.0, ("o", i))))
    out.append(Lerp("equal", -2.0, 5.0, 0.7, 0.7, _lerp_points(-2.0, 5.0, "eq")))
    out.append(Lerp("equal-tiny", 0.0, 1.0, 5e-324, 5e-324, _lerp_points(0.0, 1.0, "eqt")))
    for name, a, b in (("zero+-", 0.0, -0.0), ("zero-+", -0.0, 0.0), ("zero--", -0.0, -0.0), ("zero++", 0.0, 0.0),
                       ("zero-1", -0.0, 1.0), ("one-zero", 1.0, -0.0)):  # fmt: skip
        out.append(Lerp(name, 0.0, 1.0, a, b, _lerp_points(0.0, 1.0, name, k=2, inside=3)))
    for name, a, b in (("inf-1", INF, 1.0), ("1-inf", 1.0, INF), ("inf-inf", INF, INF), ("-inf-inf", -INF, INF),
                       ("nan-1", NAN, 1.0), ("1-nan", 1.0, NAN), ("-inf-1", -INF, 1.0)):  # fmt: skip
        out.append(Lerp(name, 0.0, 1.0, a, b, _lerp_points(0.0, 1.0, name, k=2, inside=3)))
    out.append(Lerp("subnormal-width", 1e-310, 1e-310 + 3e-320, -1.0, 2.0,
                    _lerp_points(1e-310, 1e-310 + 3e-320, "sw", k=4, inside=4)))  # fmt: skip
    out.append(Lerp("wide", -1e308, 1.5e308, 2.0, -3.0, _lerp_points(-1e308, 1.5e308, "wide")))
    out.append(Lerp("huge-values", 0.0, 1.0, -1.7e308, 1.7e308, _lerp_points(0.0, 1.0, "hv")))
    out.append(Lerp("huge-same", 0.0, 1.0, 1.7e308, 1.79e308, _lerp_points(0.0, 1.0, "hs")))
    out.append(Lerp("tiny-values", 0.0, 1.0, 3e-320, -7e-321, _lerp_points(0.0, 1.0, "tv")))
    out.append(Lerp("outside", 2.0, 3.0, 1.0, 4.0, np.array([-1e3, 0.0, 1.999, 3.001, 4.0, 1e3])))
    out.append(Lerp("outside-down", 2.0, 3.0, 4.0, 1.0, np.array([-1e3, 0.0, 1.999, 3.001, 4.0, 1e3])))
    return tuple(out)


def lerp_finite(L: Lerp) -> bool:
    return all(math.isfinite(v) for v in (L.x0, L.x1, L.a, L.b))


# ---------------------------------------------------------------- grids


@dataclasses.dataclass(frozen=True)
class Grid:
    id: str
    points: tuple  # one strictly monotone (n_i,) array per axis
    values: np.ndarray  # (n_1, ..., n_d, ...batch)
    xi: np.ndarray  # (q, d)
    tags: frozenset

    @property
    def digest(self) -> str:
        return digest(tuple(self.points), self.values, self.xi)


def _axis(n, key, desc=False):
    p = _uniform_x(n) if n > 1 else np.array([0.5])
    if n > 2:
        p = p + rng("axis", key).uniform(-0.3, 0.3, n) / (n - 1)
        p[0], p[-1] = 0.0, 1.0
        p = np.sort(p)
    return p[::-1].copy() if desc else p


def _grid_xi(points, key, q=40):
    g = rng("grid-xi", key)
    d = len(points)
    lo = np.array([p.min() for p in points])
    hi = np.array([p.max() for p in points])
    inside = lo + (hi - lo) * g.random((q, d))
    nodes = np.array([[p[g.integers(0, p.size)] for p in points] for _ in range(8)])
    edges = np.vstack([lo, hi])
    return np.vstack([inside, nodes, edges])


def _grid(id_, shape, key, desc=(), batch=(), nan=False, tags=()):
    points = tuple(_axis(n, (key, i), i in desc) for i, n in enumerate(shape))
    values = rng("grid-v", key).standard_normal(tuple(shape) + tuple(batch))
    if nan:
        values.flat[values.size // 3] = NAN
    return id_, points, values, _grid_xi(points, key), frozenset(tags)


_GRIDS = {}
for _args in (
    ("grid/1d7", (7,), "g1"),
    ("grid/2d", (6, 7), "g2"),
    ("grid/2d-desc", (6, 7), "g2d", (1,)),
    ("grid/3d", (5, 6, 4), "g3"),
    ("grid/4d", (4, 3, 5, 4), "g4"),
    ("grid/axis1", (1, 5), "ga1"),
    ("grid/axis2", (2, 2, 3), "ga2"),
    ("grid/2d-big", (8, 9), "gb"),
):
    _GRIDS[_args[0]] = functools.partial(_grid, *_args)
_GRIDS["grid/nan"] = functools.partial(_grid, "grid/nan", (5, 6), "gn", nan=True, tags={"nan"})
_GRIDS["grid/batch"] = functools.partial(_grid, "grid/batch", (6, 5), "gbt", batch=(2,), tags={"batch"})
GRID_IDS = tuple(_GRIDS)


@functools.cache
def grid(id_: str) -> Grid:
    i, points, values, xi, tags = _GRIDS[id_]()
    return Grid(i, points, values, xi, tags)


# ---------------------------------------------------------------- scattered data


@dataclasses.dataclass(frozen=True)
class Scatter:
    id: str
    y: np.ndarray  # (n, d) data points
    d: np.ndarray  # (n,) values
    xi: np.ndarray  # (q, d) evaluation points
    tags: frozenset
    note: str = ""

    @property
    def digest(self) -> str:
        return digest(self.y, self.d, self.xi)


def _field(y):
    return np.sin(3 * y[:, 0]) + np.cos(2 * y[:, -1]) + 0.1 * y.sum(axis=1)


def _scatter_build(kind):
    g = rng("scatter", kind)
    if kind == "uniform2d":
        y = g.random((40, 2))
    elif kind == "uniform3d":
        y = g.random((30, 3))
    elif kind == "cluster2d":
        centres = np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9]])
        y = centres[g.integers(0, 3, 40)] + g.normal(0, 0.03, (40, 2))
    elif kind == "duplicate2d":
        y = g.random((20, 2))
        y[7], y[13] = y[2], y[2]
    elif kind == "collinear2d":
        t = np.sort(g.random(12))
        y = np.column_stack([t, 0.5 * t + 0.25])
    else:  # lattice2d: integer points, scaled by 1/4 (exact), many equal distances
        a = np.arange(5) / 4
        y = np.stack(np.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2)
    lo, hi = y.min(axis=0), y.max(axis=0)
    xi = lo + (hi - lo) * g.random((16, y.shape[1]))
    if kind == "lattice2d":
        xi = np.vstack([xi, [[0.125, 0.125], [0.5, 0.5], [0.375, 0.625]]])
    return y, _field(y), xi


_SCATTER_TAGS = {
    "uniform2d": {"random"},
    "uniform3d": {"random"},
    "cluster2d": {"random"},
    "duplicate2d": {"duplicate"},
    "collinear2d": {"collinear"},
    "lattice2d": {"lattice"},
}
SCATTER_IDS = tuple(f"scatter/{k}" for k in _SCATTER_TAGS)


@functools.cache
def scatter(id_: str) -> Scatter:
    kind = id_.split("/", 1)[1]
    y, d, xi = _scatter_build(kind)
    return Scatter(id_, y, d, xi, frozenset(_SCATTER_TAGS[kind]))


# ---------------------------------------------------------------- piecewise polynomials for roots


@dataclasses.dataclass(frozen=True)
class Pieces:
    id: str
    c: np.ndarray  # (k + 1, m): highest degree first, in the local variable s = x - x[i]
    x: np.ndarray  # (m + 1,) breakpoints
    note: str = ""

    @property
    def digest(self) -> str:
        return digest(self.c, self.x)


def _poly_from_roots(roots, lead=1.0):
    """Coefficients (highest first) of lead * prod(s - r): exact for dyadic roots of few terms."""
    c = np.array([lead])
    for r in roots:
        c = np.convolve(c, np.array([1.0, -r]))
    return c


def _pieces_build(kind):
    g = rng("pieces", kind)
    if kind == "tangent":
        cols = [_poly_from_roots([0.25, 0.25, -1.0]), _poly_from_roots([0.5, 0.5, 2.0], -1.0)]
        return np.column_stack(cols), np.array([0.0, 1.0, 2.0]), "double roots at 0.25 and 1.5"
    if kind == "close":
        r = 0.5 + 2.0**-30
        return np.column_stack([_poly_from_roots([0.5, r, 3.0])]), np.array([0.0, 1.0]), "roots 2^-30 apart"
    if kind == "zero-piece":
        c = np.column_stack([_poly_from_roots([0.5, 2.0, -2.0]), np.zeros(4), _poly_from_roots([0.25, 5.0, 6.0])])
        return c, np.array([0.0, 1.0, 2.0, 3.0]), "the middle piece is 0"
    if kind == "breakpoint":
        c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, -1.0, 2.0]])
        return c, np.array([0.0, 1.0, 2.0, 3.0]), "constant pieces 1, -1, 2: sign changes at the breakpoints"
    if kind == "endpoint":
        c = np.column_stack([_poly_from_roots([0.0, 0.5, 4.0]), _poly_from_roots([1.0, 7.0, 8.0])])
        return c, np.array([0.0, 1.0, 2.0]), "roots at a breakpoint and at the right end"
    if kind == "cubic":
        m = 40
        return g.standard_normal((4, m)), np.concatenate([[0.0], np.cumsum(g.uniform(0.5, 1.5, m))]), "random"
    m = 20  # quintic
    return g.standard_normal((6, m)), np.concatenate([[0.0], np.cumsum(g.uniform(0.5, 1.5, m))]), "random"


PIECES_IDS = tuple(f"roots/{k}" for k in ("tangent", "close", "zero-piece", "breakpoint", "endpoint", "cubic",
                                          "quintic"))  # fmt: skip


@functools.cache
def pieces(id_: str) -> Pieces:
    c, x, note = _pieces_build(id_.split("/", 1)[1])
    return Pieces(id_, np.ascontiguousarray(c, dtype=np.float64), np.ascontiguousarray(x), note)


# ---------------------------------------------------------------- polynomial interpolation


@dataclasses.dataclass(frozen=True)
class Poly:
    id: str
    xi: np.ndarray  # nodes (repeated nodes: Krogh's derivative data)
    yi: np.ndarray
    x: np.ndarray  # evaluation points
    note: str = ""

    @property
    def digest(self) -> str:
        return digest(self.xi, self.yi, self.x)

    @property
    def distinct(self) -> bool:
        return bool(np.unique(self.xi).size == self.xi.size)


def _poly_build(kind):
    """(nodes, values, evaluation points, note); the evaluation points include the nodes."""
    g = rng("poly", kind)
    x = np.sort(np.concatenate([np.linspace(-1, 1, 41), g.uniform(-1, 1, 20)]))
    if kind.startswith("cheb"):
        n = int(kind[4:])
        xi = _chebyshev_x(n)
        return xi, np.exp(xi), np.union1d(x, xi), "exp at Chebyshev nodes"
    if kind.startswith("uniform"):
        n = int(kind[7:])
        xi = _uniform_x(n, -1, 1)
        return xi, 1 / (1 + 25 * xi**2), np.union1d(x, xi), "Runge at uniform nodes"
    if kind == "close":
        xi = np.array([0.0, 1e-8, 0.5, 0.5 + 1e-8, 1.0])
        return xi, np.cos(xi), np.union1d(np.linspace(-0.2, 1.2, 29), xi), "two pairs of nodes 1e-8 apart"
    # hermite: repeated nodes carry the successive derivatives (scipy's Krogh)
    xi = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0])
    yi = np.array([1.0, 2.0, 0.0, -1.0, 3.0, 1.0])
    return xi, yi, np.linspace(-0.5, 2.5, 25), "f(0)=1, f'(0)=2, f(1)=0, f'(1)=-1, f''(1)=3, f(2)=1"


POLY_IDS = tuple(f"poly/{k}" for k in ("cheb10", "cheb30", "uniform10", "uniform30", "close", "hermite"))


@functools.cache
def poly(id_: str) -> Poly:
    xi, yi, x, note = _poly_build(id_.split("/", 1)[1])
    return Poly(id_, np.ascontiguousarray(xi), np.ascontiguousarray(yi), np.ascontiguousarray(x), note)


# ---------------------------------------------------------------- AAA samples


@dataclasses.dataclass(frozen=True)
class Samples:
    id: str
    z: np.ndarray  # points (float64 or complex128)
    f: np.ndarray  # values
    note: str = ""

    @property
    def digest(self) -> str:
        return digest(self.z, self.f)


def _aaa_build(kind):
    if kind == "tan":
        z = np.linspace(-1, 1, 1000)
        return z, np.tan(np.pi * z / 2), "tan(pi z / 2) on 1000 points of [-1, 1]"
    if kind == "abs":
        z = np.linspace(-1, 1, 1000)
        return z, np.abs(z), "|x|"
    if kind == "circle":
        z = np.exp(2j * np.pi * np.arange(256) / 256)
        return z, np.exp(z) / (z - 1.5), "exp(z) / (z - 1.5) on the unit circle"
    z = np.linspace(0, 1, 200)  # nonfinite
    f = 1 / (z - 0.5 + 1e-3)
    f[[10, 50]] = [NAN, INF]
    return z, f, "a pole near 0.5; one NaN and one inf sample"


AAA_IDS = tuple(f"aaa/{k}" for k in ("tan", "abs", "circle", "nonfinite"))


@functools.cache
def aaa(id_: str) -> Samples:
    z, f, note = _aaa_build(id_.split("/", 1)[1])
    return Samples(id_, z, f, note)
