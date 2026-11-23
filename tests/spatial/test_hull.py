"""ConvexHull (docs/design/spatial_geometry.md §2.4, §3.6, §4 item 5 and the real-output contracts; milestone
SPT11): the combinatorial structure is exact (checked in rational arithmetic by exactgeom.check_hull), the
arrays are in the canonical form of §2.2, and area, volume and equations are within the §4 bounds of their
exact values (rationals; Arb for the square roots)."""

import copy
import math
import pickle
from fractions import Fraction

import flint
import numpy as np
import pytest

import stardust as sd

from . import exactgeom as X
from . import geompoints as G

U = Fraction(1, 2**53)
HULL = sd.spatial.ConvexHull if hasattr(sd.spatial, "ConvexHull") else None
if HULL is None:  # before sd.spatial exports it
    from stardust.spatial._hull import ConvexHull as HULL

VALID = [i for i in G.IDS if "invalid" not in G.get(i).tags]
WIDE = (
    "mixed2/1e150",
    "mixed3/1e150",
    "mixed2/1e300",
    "mixed3/1e300",
    "dblmax2",
    "dblmax3",
    "subnormal2",
    "subnormal3",
)  # area or volume overflow, or terms underflow


def hull(x):
    return HULL(sd.asarray(np.ascontiguousarray(x, dtype=np.float64)))


def arrays(h):
    return {k: np.asarray(getattr(h, k)) for k in ("vertices", "simplices", "neighbors", "equations")}


def gamma(k):
    return k * U / (1 - k * U)


def s_of(n):
    """The most additions a term passes in the sum of 4,096-term partial sums (§4)."""
    return min(n, 4096) + -(-n // 4096) - 2


def to_frac(a):
    """An arb's midpoint as a Fraction (the radius is far below the bounds checked)."""
    m, e = a.mid().man_exp()
    m, e = int(m), int(e)
    return Fraction(m * 2**e) if e >= 0 else Fraction(m, 2**-e)


flint.ctx.prec = 256


def fsqrt(q):
    return to_frac(flint.arb(flint.fmpq(q.numerator, q.denominator)).sqrt())


def rel(got, want):
    return abs(Fraction(got) - want) / abs(want)


# ---------------------------------------------------------------- §4 item 5 and the canonical form


def check_canonical(x, h):
    a = arrays(h)
    v, s, nb, eq = a["vertices"], a["simplices"], a["neighbors"], a["equations"]
    d = x.shape[1]
    assert v.dtype == s.dtype == nb.dtype == np.int64 and eq.dtype == np.float64
    assert s.shape == nb.shape == (len(s), d) and eq.shape == (len(s), d + 1)
    assert [tuple(r) for r in s] == sorted(tuple(r) for r in s), "rows are not in lexicographic order"
    if d == 2:
        assert v[0] == v.min() and len(s) == len(v)
        ring = {int(v[k]): int(v[(k + 1) % len(v)]) for k in range(len(v))}
        assert {(int(i), int(j)) for i, j in s} == set(ring.items()), "edges are not the ring's"
    else:
        assert list(v) == sorted(v)
        assert all(r[0] == r.min() for r in s)
    # neighbors[i, k]: the facet across from vertex k
    for i, row in enumerate(s):
        for k in range(d):
            edge = {int(row[j]) for j in range(d) if j != k}
            j = int(nb[i, k])
            assert j != i and edge <= {int(t) for t in s[j]}, f"neighbors[{i}, {k}]"
    # the representative of equal points is the smallest index
    rep = X.representatives(X.fpoints(x))
    assert all(rep[int(i)] == int(i) for i in v)
    return v, s


@pytest.mark.parametrize("id_", VALID)
def test_structure_is_exact(id_):
    x = G.get(id_).x
    h = hull(x)
    v, s = check_canonical(x, h)
    if len(x) <= 2000:
        X.check_hull(x, v, s, reference=30)
    else:  # every point inside every facet by the (tested) exact predicate, the surface closed
        name = "orient2d" if x.shape[1] == 2 else "orient3d"
        sign = 1 if x.shape[1] == 2 else -1
        for f in s:
            rows = np.concatenate([np.broadcast_to(x[f], (len(x), *x[f].shape)), x[:, None, :]], axis=1)
            o = np.asarray(sd._core.spatial._predicate(name, sd.asarray(np.ascontiguousarray(rows)), "auto"))
            assert (o * sign >= 0).all()


@pytest.mark.parametrize("id_", ["uniform2/1000", "ball3/1000", "cube/5", "boundary3", "sphere3/int", "dup3"])
def test_rows_shuffled_give_the_same_hull(id_):
    x = G.get(id_).x
    perm = np.random.default_rng(5).permutation(len(x))
    a, b = hull(x), hull(x[perm])

    def facets(h, pts):  # the flat polygons as point sets (the fans start at the smallest index, which moves)
        poly = {}
        for f, e in zip(np.asarray(h.simplices), np.asarray(h.equations)):
            poly.setdefault(tuple(np.round(e, 12)), set()).update(map(tuple, pts[f]))
        return {frozenset(v) for v in poly.values()}

    assert {tuple(p) for p in x[np.asarray(a.vertices)]} == {tuple(p) for p in x[perm][np.asarray(b.vertices)]}
    assert facets(a, x) == facets(b, x[perm])


def test_flat_facets_are_fans_from_the_smallest_index():
    x = G.get("cube/5").x  # the unit cube with its grid: 6 square facets, corners only
    h = hull(x)
    v = np.asarray(h.vertices)
    assert sorted(map(tuple, x[v])) == sorted({(a, b, c) for a in (0.0, 1.0) for b in (0.0, 1.0) for c in (0.0, 1.0)})
    s, eq = np.asarray(h.simplices), np.asarray(h.equations)
    assert len(s) == 12
    for f, e in zip(s, eq):  # the two triangles of a square share the smallest corner and the row of bits
        twins = [g for g, q in zip(s, eq) if g[0] == f[0] and np.array_equal(q.view(np.uint64), e.view(np.uint64))]
        assert len(twins) == 2
    assert h.volume == 1.0 and h.area == 6.0


def test_two_dimensional_boundary_points_are_not_vertices():
    x = G.get("boundary2").x
    h = hull(x)
    assert len(h.vertices) == 4 and h.nsimplex == 4


# ---------------------------------------------------------------- the real-output contracts (§4)


def exact_measures(x, s, v):
    """(area, volume) exactly: d = 2 (perimeter, enclosed area), d = 3 (surface area, volume)."""
    P = X.fpoints(x)
    if x.shape[1] == 2:
        per = sum(fsqrt(X.dist2(P[i], P[j])) for i, j in s)
        o = P[v[0]]
        area = sum(X.orient_value(o, P[v[k]], P[v[k + 1]]) for k in range(1, len(v) - 1)) / 2
        return per, area
    o = P[v[0]]
    surf, vol = Fraction(0), Fraction(0)
    for a, b, c in s:
        n = X._cross(X._sub(P[b], P[a]), X._sub(P[c], P[a]))
        surf += fsqrt(X._dot(n, n)) / 2
        vol += X.orient_value(o, P[a], P[b], P[c]) / 6
    return surf, vol


CONTRACT = [i for i in VALID if i not in WIDE and len(G.get(i).x) <= 2000] + ["uniform2/20000", "uniform3/20000"]


@pytest.mark.parametrize("id_", CONTRACT)
def test_area_and_volume_contracts(id_):
    x = G.get(id_).x
    h = hull(x)
    s, v = np.asarray(h.simplices), np.asarray(h.vertices)
    area, vol = exact_measures(x, s, v)
    n = len(s) if x.shape[1] == 3 else len(s)
    if x.shape[1] == 2:
        assert rel(h.area, area) <= gamma(s_of(len(s)) + 4)
        assert rel(h.volume, vol) <= gamma(s_of(len(v) - 2) + 2)
    else:
        assert rel(h.area, area) <= gamma(s_of(n) + 6)
        assert rel(h.volume, vol) <= gamma(s_of(n) + 3)


@pytest.mark.parametrize("id_", [i for i in VALID if len(G.get(i).x) <= 2000])
def test_equations_contract(id_):
    x = G.get(id_).x
    h = hull(x)
    P = X.fpoints(x)
    s, eq = np.asarray(h.simplices), np.asarray(h.equations)
    for f, e in zip(s, eq):
        if x.shape[1] == 2:
            i, j = int(f[0]), int(f[1])
            n = (P[j][1] - P[i][1], P[i][0] - P[j][0])
        else:
            n = X._cross(X._sub(P[f[1]], P[f[0]]), X._sub(P[f[2]], P[f[0]]))
        L = fsqrt(sum(t * t for t in n))
        c = [t / L for t in n]
        for k in range(len(c)):
            assert abs(Fraction(e[k]) - c[k]) <= 8 * U * abs(c[k]), (f, k)
        p = P[int(min(f))]
        b = -sum(ck * pk for ck, pk in zip(c, p))
        tol = 12 * U * sum(abs(ck * pk) for ck, pk in zip(c, p)) + len(c) * Fraction(1, 2**1074)
        big = Fraction(np.finfo(float).max)
        if abs(b) > big:  # the offset itself overflows
            assert np.isinf(e[-1])
        elif np.isinf(e[-1]):  # it may round past DBL_MAX only within the bound
            assert abs(b) + tol >= big, (f, "offset")
        else:
            assert abs(Fraction(e[-1]) - b) <= tol, (f, "offset")


@pytest.mark.parametrize("id_", WIDE)
def test_wide_range_points(id_):
    """Coordinates whose products overflow or underflow: the normals are still unit vectors, area and
    volume overflow to inf only when the exact value does, and tiny ones are within 2^-1074 per term."""
    x = G.get(id_).x
    h = hull(x)
    eq = np.asarray(h.equations)
    assert np.isfinite(eq[:, :-1]).all()
    assert np.allclose(np.linalg.norm(eq[:, :-1], axis=1), 1.0, rtol=1e-15, atol=0)
    area, vol = exact_measures(x, np.asarray(h.simplices), np.asarray(h.vertices))
    for got, want, nterms in ((h.area, area, h.nsimplex), (h.volume, vol, h.nsimplex)):
        if abs(want) > Fraction(np.finfo(float).max):
            assert math.isinf(got)
        else:
            tol = gamma(s_of(nterms) + 6) * abs(want) + nterms * Fraction(1, 2**1073)
            assert abs(Fraction(got) - want) <= tol


# ---------------------------------------------------------------- attributes, errors, pickle


def test_attributes():
    x = G.get("uniform3/1000").x
    h = hull(x)
    assert h.ndim == 3 and h.npoints == 1000 and h.nsimplex == len(np.asarray(h.simplices))
    assert np.array_equal(np.asarray(h.points), x)
    assert np.array_equal(np.asarray(h.min_bound), x.min(axis=0))
    assert np.array_equal(np.asarray(h.max_bound), x.max(axis=0))
    assert isinstance(h.area, float) and isinstance(h.volume, float)
    assert repr(h) == f"ConvexHull(n=1000, d=3, nvertices={len(np.asarray(h.vertices))})"
    for name in ("points", "vertices", "simplices", "neighbors", "equations", "min_bound", "max_bound"):
        with pytest.raises(ValueError):
            getattr(h, name)[0] = 0


def test_points_are_copied():
    x = sd.asarray(G.get("uniform2/1000").x.copy())
    h = HULL(x)
    before = np.asarray(h.points).copy()
    x[0, 0] = 5.0
    assert np.array_equal(np.asarray(h.points), before)


def test_inputs_of_every_kind():
    pts = [[0, 0], [2, 0], [0, 2], [1, 1], [2, 2]]
    want = arrays(hull(np.array(pts, dtype=float)))
    for inp in (pts, np.array(pts), sd.asarray(np.array(pts)), np.array(pts) != 0):
        got = arrays(HULL(inp)) if not isinstance(inp, np.ndarray) or inp.dtype != bool else None
        if got is not None:
            assert all(np.array_equal(got[k], want[k]) for k in want)
    assert HULL(np.array([[0, 0], [1, 0], [0, 1]], dtype=bool)).volume == 0.5


def test_pickle_and_copy():
    h = hull(G.get("ball3/1000").x)
    for g in (pickle.loads(pickle.dumps(h)), copy.copy(h), copy.deepcopy(h)):
        assert all(np.array_equal(arrays(g)[k], arrays(h)[k]) for k in arrays(h))
        assert g.area == h.area and g.volume == h.volume


@pytest.mark.parametrize(
    ("x", "exc", "msg"),
    [
        (np.zeros(3), ValueError, "ConvexHull(): 'points' must have 2 dimensions, not 1"),
        (np.zeros((5, 4)), ValueError, "ConvexHull(): 'points' must have 2 or 3 columns, not 4"),
        (np.zeros((5, 1)), ValueError, "ConvexHull(): 'points' must have 2 or 3 columns, not 1"),
        (np.array([[0.0, 0.0], [1.0, np.nan], [0.0, 1.0]]), ValueError, "ConvexHull(): 'points' must be finite"),
        (np.array([[0.0, 0.0], [1.0, np.inf], [0.0, 1.0]]), ValueError, "ConvexHull(): 'points' must be finite"),
        (G.get("line2").x, ValueError, "ConvexHull(): 'points' are collinear"),
        (G.get("line3").x, ValueError, "ConvexHull(): 'points' are coplanar"),
        (G.get("plane3").x, ValueError, "ConvexHull(): 'points' are coplanar"),
        (G.get("line3+1").x, ValueError, "ConvexHull(): 'points' are coplanar"),
        (G.get("few2").x, ValueError, "ConvexHull(): needs at least 3 distinct points, got 1"),
        (G.get("few3").x, ValueError, "ConvexHull(): needs at least 4 distinct points, got 2"),
        (np.zeros((0, 2)), ValueError, "ConvexHull(): needs at least 3 distinct points, got 0"),
        (np.zeros((7, 3)), ValueError, "ConvexHull(): needs at least 4 distinct points, got 1"),
        (np.array([[0, 0], [1, 0], [0, 1j]]), TypeError, "ConvexHull(): 'points' must be real, not a complex128 array"),
    ],
)
def test_errors(x, exc, msg):
    with pytest.raises(exc) as e:
        HULL(x)
    assert str(e.value) == msg


def test_call_shape_errors():
    x = G.get("uniform2/12").x
    with pytest.raises(TypeError, match=r"^ConvexHull\(\): unexpected keyword argument 'qhull_options'$"):
        HULL(x, qhull_options="Qt")
    with pytest.raises(TypeError, match=r"^ConvexHull\(\): unexpected keyword argument 'incremental'$"):
        HULL(x, incremental=True)


def test_matches_delaunay_boundary():
    """§4 item 5: the extreme points are Delaunay's boundary vertices that are extreme."""
    if not hasattr(sd.spatial, "Delaunay"):
        pytest.skip("Delaunay comes with milestones SPT9 and SPT10")
    for id_ in ("uniform2/1000", "boundary2", "grid2/30", "uniform3/1000", "boundary3", "cube/5"):
        x = G.get(id_).x
        try:
            tri = sd.spatial.Delaunay(sd.asarray(x))
        except ValueError, NotImplementedError:
            continue
        bnd = {int(i) for f in np.asarray(tri.convex_hull) for i in f}
        ext = set(X.extreme_points(x)) if len(x) <= 30 or x.shape[1] == 2 else {int(i) for i in hull(x).vertices}
        assert {int(i) for i in hull(x).vertices} == bnd & ext
