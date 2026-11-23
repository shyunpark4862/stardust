"""Test points of sd.spatial's computational geometry (docs/design/spatial_geometry.md §7.1; SPT117):
2-D and 3-D point sets for Delaunay / ConvexHull / Voronoi, and rows of points for the predicates.

A module of its own beside points.py (the distance and neighbour sets, whose ids other tests and the
reference files enumerate): the same deterministic generators (`points.rng`), the same PointSet.

Point sets: `get(id)`, `ids(tag)`, tags
- 'random': uniform, normal, clustered, in a disk / ball;
- 'degenerate': exact ties (integer grids, integer points on one circle / sphere, a cube with its inner
  grid, boundary points on hull edges and faces, duplicates, every point extreme, d + 1 and d + 2 points);
- 'near': almost degenerate (a grid with 1e-13 noise, y = 1e-9 x^2 and 1e-14 x^2, 1e8 + a small grid,
  1e6 + 1e-3 * uniform, points one ULP off a circle / sphere);
- 'extreme': the wide path (1e150 with 1e-150, subnormals, near DBL_MAX);
- 'invalid': inputs the classes reject (all collinear / coplanar, too few distinct points);
- 'small': at most 40 (d = 2) or 16 (d = 3) distinct points, for the exhaustive references;
- 'd2', 'd3'.

Query points: `queries(id, q)`, uniform in a set's box (the barycentric references of tools/spatial_ref.py).

Predicate rows: `predicate_rows(name, kind, m)` -> (m, points, d) float64 for name in PREDICATES and kind
in ROW_KINDS (uniform, grid, zero, near, far, dyadic, extreme).
"""

import functools
import itertools
import math

import numpy as np

from .points import PointSet, rng

PREDICATES = {"orient2d": (3, 2), "orient3d": (4, 3), "incircle": (4, 2), "insphere": (5, 3)}
ROW_KINDS = ("uniform", "grid", "zero", "near", "far", "dyadic", "extreme")
DBL_MAX = float(np.finfo(np.float64).max)
TINY = 5e-324

# ---------------------------------------------------------------- point sets


def _uniform(d, n, key="u"):
    return rng("geom", key, d, n).random((n, d))


def _normal(d, n):
    return rng("geom", "normal", d, n).standard_normal((n, d))


def _cluster(d, n, k=8):
    g = rng("geom", "cluster", d, n)
    centers = g.random((k, d)) * 10.0
    return centers[g.integers(0, k, n)] + 0.05 * g.standard_normal((n, d))


def _in_ball(d, n):
    g = rng("geom", "ball", d, n)
    v = g.standard_normal((n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * g.random((n, 1)) ** (1.0 / d)


def _grid(d, side):
    return np.array(list(itertools.product(range(side), repeat=d)), dtype=np.float64)


def _shuffled(x, key):
    return x[rng("geom", "shuffle", key).permutation(len(x))]


@functools.cache
def _on_circle_int(r2):
    """Every integer point with x^2 + y^2 = r2."""
    r = math.isqrt(r2)
    return [(x, y) for x in range(-r, r + 1) for y in range(-r, r + 1) if x * x + y * y == r2]


@functools.cache
def _on_sphere_int(r2):
    r = math.isqrt(r2)
    out = []
    for x in range(-r, r + 1):
        for y in range(-r, r + 1):
            z2 = r2 - x * x - y * y
            if z2 >= 0 and math.isqrt(z2) ** 2 == z2:
                z = math.isqrt(z2)
                out.extend({(x, y, z), (x, y, -z)})
    return sorted(out)


def _circle_int():
    return _shuffled(np.array(_on_circle_int(1105), dtype=np.float64), "circle")  # r^2 = 1105: 32 points


def _knight():
    return np.array([(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1), (0, 0)], dtype=np.float64)


def _sphere_int():
    return _shuffled(np.array(_on_sphere_int(66), dtype=np.float64), "sphere")  # 96 points


def _cube(side):
    return _shuffled(_grid(3, side) / (side - 1), ("cube", side))


def _boundary2():
    g = rng("geom", "boundary2")
    edge = np.arange(1, 8) / 8.0
    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    pts += [(t, 0) for t in edge] + [(1, t) for t in edge] + [(t, 1) for t in edge] + [(0, t) for t in edge]
    inner = g.integers(1, 16, (20, 2)) / 16.0
    return _shuffled(np.concatenate([np.array(pts, dtype=np.float64), inner]), "boundary2")


def _boundary3():
    g = rng("geom", "boundary3")
    corners = _grid(3, 2)
    face = g.integers(0, 9, (60, 3)) / 8.0
    face[np.arange(60), g.integers(0, 3, 60)] = g.integers(0, 2, 60)  # one coordinate on a face
    inner = g.integers(1, 8, (20, 3)) / 8.0
    return _shuffled(np.concatenate([corners, face, inner]), "boundary3")


def _dup(d):
    g = rng("geom", "dup", d)
    base = g.integers(-4, 5, (25, d)).astype(np.float64)
    x = base[g.integers(0, 25, 120)]
    x[::7] *= -1.0  # -0.0 against 0.0 and mirrored copies
    return x


def _parabola(scale, n=300):
    x = np.arange(n, dtype=np.float64)
    return np.stack([x, scale * x * x], axis=1)


def _moment3(n=12):
    t = np.arange(1, n + 1, dtype=np.float64)
    return np.stack([t, t * t, t * t * t], axis=1)


def _circle_ulp(n, d):
    g = rng("geom", "circle_ulp", n, d)
    if d == 2:
        a = g.random(n) * 2 * math.pi
        x = np.stack([np.cos(a), np.sin(a)], axis=1)
    else:
        x = g.standard_normal((n, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    k = g.integers(-2, 3, x.shape)
    for _ in range(2):
        x = np.where(k > 0, np.nextafter(x, np.inf), np.where(k < 0, np.nextafter(x, -np.inf), x))
        k = k - np.sign(k)
    return x


def _polygon(n):
    a = 2 * math.pi * np.arange(n) / n
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def _mixed(d, n, big, small):
    g = rng("geom", "mixed", d, n, big)
    x = g.standard_normal((n, d))
    scale = np.where(g.random((n, d)) < 0.5, big, small)
    return x * scale


def _subnormal(d, n):
    g = rng("geom", "subnormal", d, n)
    return g.integers(-(2**20), 2**20, (n, d)).astype(np.float64) * TINY


def _dblmax(d, n):
    g = rng("geom", "dblmax", d, n)
    return (g.random((n, d)) * 2.0 - 1.0) * DBL_MAX


def _line(d, n):
    t = np.arange(n, dtype=np.float64)
    return np.stack([t * (k + 1) for k in range(d)], axis=1)


def _plane3(n):
    g = rng("geom", "plane3", n)
    uv = g.integers(-8, 9, (n, 2)).astype(np.float64)
    return np.stack([uv[:, 0], uv[:, 1], uv[:, 0] + 2 * uv[:, 1]], axis=1)


_BUILDERS = {}


def _register(id_, tags, build, note=""):
    _BUILDERS[id_] = (frozenset(tags), build, note)


for _d in (2, 3):
    _t = {f"d{_d}", "random"}
    _register(f"uniform{_d}/12", _t | {"small"}, lambda d=_d: _uniform(d, 12))
    _register(f"uniform{_d}/1000", _t, lambda d=_d: _uniform(d, 1000))
    _register(f"uniform{_d}/20000", _t | {"large"}, lambda d=_d: _uniform(d, 20000))
    _register(f"normal{_d}/1000", _t, lambda d=_d: _normal(d, 1000))
    _register(f"cluster{_d}/1000", _t, lambda d=_d: _cluster(d, 1000))
    _register(f"ball{_d}/1000", _t, lambda d=_d: _in_ball(d, 1000), "inside the unit disk / ball")
    _register(f"dup{_d}", {f"d{_d}", "degenerate"}, lambda d=_d: _dup(d), "25 integer points, 120 rows, -0.0")
    _register(f"line{_d}", {f"d{_d}", "invalid"}, lambda d=_d: _line(d, 9), "every point on one line")
    _register(
        f"line{_d}+1", {f"d{_d}", "degenerate" if _d == 2 else "invalid", "small"},
        lambda d=_d: np.concatenate([_line(d, 9), np.eye(d)[:1] * 3.0]), "a line and one point off it",
    )  # fmt: skip
    _register(f"few{_d}", {f"d{_d}", "invalid"}, lambda d=_d: np.repeat(np.eye(d)[: d - 1], 3, axis=0) + 0.0,
              "fewer than d + 1 distinct points")  # fmt: skip
    _register(f"mixed{_d}/1e150", {f"d{_d}", "extreme"}, lambda d=_d: _mixed(d, 40, 1e150, 1e-150))
    _register(f"mixed{_d}/1e300", {f"d{_d}", "extreme"}, lambda d=_d: _mixed(d, 40, 1e300, 1e-300))
    _register(f"subnormal{_d}", {f"d{_d}", "extreme"}, lambda d=_d: _subnormal(d, 40), "integers times 2^-1074")
    _register(f"dblmax{_d}", {f"d{_d}", "extreme"}, lambda d=_d: _dblmax(d, 40), "uniform in (-DBL_MAX, DBL_MAX)")
    _register(
        f"ulp{_d}/200", {f"d{_d}", "near"}, lambda d=_d: _circle_ulp(200, d), "0 to 2 ULP off the unit circle / sphere"
    )
    _register(
        f"minimal{_d}", {f"d{_d}", "degenerate", "small"}, lambda d=_d: np.concatenate([np.zeros((1, d)), np.eye(d)])
    )
    _register(
        f"minimal{_d}+1", {f"d{_d}", "degenerate", "small"},
        lambda d=_d: np.concatenate([np.zeros((1, d)), np.eye(d), np.ones((1, d))]),
        "d + 2 points on one circle / sphere",
    )  # fmt: skip

_register("grid2/30", {"d2", "degenerate"}, lambda: _shuffled(_grid(2, 30), "g30"), "every cell cocircular")
_register("grid2/5", {"d2", "degenerate", "small"}, lambda: _shuffled(_grid(2, 5), "g5"))
_register("grid3/6", {"d3", "degenerate"}, lambda: _shuffled(_grid(3, 6), "g6"), "every cell cospherical")
_register("grid3/2", {"d3", "degenerate", "small"}, lambda: _grid(3, 2), "the cube: 8 cospherical points")
_register("cube/5", {"d3", "degenerate"}, lambda: _cube(5), "the unit cube with its 5^3 grid (multiples of 1/4)")
_register("circle2/int", {"d2", "degenerate", "small"}, _circle_int, "32 integer points on x^2 + y^2 = 1105")
_register("knight2", {"d2", "degenerate", "small"}, _knight, "(+-2, +-1), (+-1, +-2) and the centre")
_register("sphere3/int", {"d3", "degenerate"}, _sphere_int, "96 integer points on x^2 + y^2 + z^2 = 66")
_register("sphere3/int9", {"d3", "degenerate", "small"},
          lambda: np.array(_on_sphere_int(9)[::2][:14], dtype=np.float64),
          "14 integer points on the sphere of radius 3")  # fmt: skip
_register("boundary2", {"d2", "degenerate"}, _boundary2, "a square with points on its edges")
_register("boundary3", {"d3", "degenerate"}, _boundary3, "a cube with points on its faces and edges")
_register("parabola2/int", {"d2", "degenerate"}, lambda: _parabola(1.0, 200), "(x, x^2): every point extreme")
_register("moment3", {"d3", "degenerate", "small"}, _moment3, "(t, t^2, t^3): the quadratic case of 3-D Delaunay")
_register("plane3", {"d3", "invalid"}, lambda: _plane3(30), "every point on one plane")
_register(
    "plane3+1", {"d3", "degenerate"}, lambda: np.concatenate([_plane3(30), [[0.0, 0.0, 5.0]]]), "a plane and one point"
)
_register("polygon2/200", {"d2", "near"}, lambda: _polygon(200), "200 computed points of the unit circle")
_register(
    "grid2/30+1e-13", {"d2", "near"}, lambda: _grid(2, 30) + 1e-13 * rng("geom", "noise").standard_normal((900, 2))
)
_register("parabola2/1e-9", {"d2", "near"}, lambda: _parabola(1e-9))
_register("parabola2/1e-14", {"d2", "near"}, lambda: _parabola(1e-14))
_register("far2/1e8grid", {"d2", "near"}, lambda: 1e8 + 0.1 * _grid(2, 30))
_register("far2/1e6uniform", {"d2", "near"}, lambda: 1e6 + 1e-3 * _uniform(2, 500, "far"))
_register("far3/1e8grid", {"d3", "near"}, lambda: 1e8 + 0.1 * _grid(3, 6))

IDS = tuple(_BUILDERS)


@functools.cache
def get(id_: str) -> PointSet:
    if id_ not in _BUILDERS:
        raise KeyError(f"unknown point set {id_!r}")
    tags, build, note = _BUILDERS[id_]
    return PointSet(id_, np.ascontiguousarray(build(), dtype=np.float64), tags, None, note)


def ids(*tags: str) -> list[str]:
    """The ids carrying every one of `tags`."""
    return [i for i in IDS if all(t in _BUILDERS[i][0] for t in tags)]


def queries(id_: str, q: int = 64) -> np.ndarray:
    """q points uniform in the bounding box of set id_ (the barycentric references, SPT187)."""
    x = get(id_).x
    lo, hi = x.min(axis=0), x.max(axis=0)
    t = rng("geom", "queries", id_).random((q, x.shape[1]))
    with np.errstate(over="ignore"):
        span = hi - lo
    return lo + t * span if np.all(np.isfinite(span)) else lo * (1 - t) + hi * t


# ---------------------------------------------------------------- predicate rows


def _ulps(x, g, most=2):
    """x moved by -most..most ULP per element."""
    k = g.integers(-most, most + 1, x.shape)
    for _ in range(most):
        x = np.where(k > 0, np.nextafter(x, np.inf), np.where(k < 0, np.nextafter(x, -np.inf), x))
        k = k - np.sign(k)
    return x


def _degenerate_rows(name, g, m, lo=-50, hi=51):
    """Integer rows on which the predicate is exactly 0: the last point of orient* is an integer affine
    combination of the others; incircle / insphere rows are images of fixed cocircular / cospherical sets."""
    k, d = PREDICATES[name]
    if name.startswith("orient"):
        base = g.integers(lo, hi, (m, k - 1, d))
        w = g.integers(-3, 4, (m, k - 1))
        w[:, 0] = 1 - w[:, 1:].sum(axis=1)
        last = (w[:, :, None] * base).sum(axis=1, keepdims=True)
        return np.concatenate([base, last], axis=1).astype(np.float64)
    pool = np.array(_on_circle_int(1105) if d == 2 else _on_sphere_int(66), dtype=np.int64)
    rows = np.empty((m, k, d))
    for i in range(m):
        pick = g.choice(len(pool), k, replace=False)
        rows[i] = pool[pick] * g.integers(1, 5) + g.integers(lo, hi, d)
    return rows


def predicate_rows(name: str, kind: str, m: int) -> np.ndarray:
    """(m, points, d) float64 rows for the predicate `name`.
    uniform: [0, 1); grid: integers in [-3, 3] (many exact zeros, repeated points); zero: exactly degenerate
    integer rows; near: those rows scaled by a random float and moved 0-2 ULP (the filter cannot decide);
    far: 2^27 + small multiples of 2^-20 (exact differences, huge cancellation); dyadic: coordinates
    +-2^k, k in [-40, 40]; extreme: 1e300, 1e-300, subnormals and DBL_MAX mixed (the wide path)."""
    k, d = PREDICATES[name]
    g = rng("geom", "rows", name, kind, m)
    if kind == "uniform":
        return g.random((m, k, d))
    if kind == "grid":
        return g.integers(-3, 4, (m, k, d)).astype(np.float64)
    if kind == "zero":
        return _degenerate_rows(name, g, m)
    if kind == "near":
        rows = _degenerate_rows(name, g, m) * g.random((m, 1, 1)) + g.random((m, 1, d))
        return _ulps(rows, g)
    if kind == "far":
        return 2.0**27 + g.integers(-8, 9, (m, k, d)) * 2.0**-20
    if kind == "dyadic":
        return np.ldexp(g.choice([-1.0, 1.0], (m, k, d)), g.integers(-40, 41, (m, k, d)))
    if kind == "extreme":
        scale = g.choice([1e300, 1e-300, TINY * 2**30, DBL_MAX / 4, 1.0, 1e150], (m, k, d))
        rows = g.integers(-4, 5, (m, k, d)) * scale
        rows[: m // 4] = _degenerate_rows(name, g, m // 4, -3, 4) * g.choice(
            [TINY, 2.0**1000, 2.0**-600], (m // 4, 1, 1)
        )
        return rows
    raise KeyError(kind)
