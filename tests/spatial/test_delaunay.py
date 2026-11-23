"""Delaunay for d = 2 (docs/design/spatial_geometry.md §2.3, §3.3, §3.5, §4 items 1-4; milestone SPT9): the
triangulation by the exact checkers of exactgeom.py, the tie rule against the exhaustive reference, the
canonical form, the attributes by their definitions, find_simplex, errors, pickle and repr, and scipy's
simplices on points in general position."""

import copy
import itertools
import pickle
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd

from . import exactgeom as X
from . import geompoints as G
from .conftest import LIVE

Delaunay = sd.spatial.Delaunay
U = 2.0**-53

VALID2 = [i for i in G.ids("d2") if "invalid" not in G.get(i).tags and "large" not in G.get(i).tags]
SMALL2 = [i for i in G.ids("d2", "small")]


def _np(a):
    return np.asarray(a)


@pytest.fixture(scope="module")
def built():
    return {i: Delaunay(G.get(i).x) for i in VALID2}


# ---------------------------------------------------------------- §4 items 1 and 2


@pytest.mark.parametrize("pid", VALID2)
def test_triangulation_is_delaunay(built, pid):
    """§4 item 1: positive, a cover of the hull, every distinct point a vertex, empty circles; the
    neighbours and coplanar rows by their definitions."""
    x = G.get(pid).x
    D = built[pid]
    X.check_delaunay(x, _np(D.simplices), neighbors=_np(D.neighbors), coplanar=_np(D.coplanar))


@pytest.mark.parametrize("pid", VALID2)
def test_tie_rule_local(built, pid):
    """§4 item 2: across every inner edge the opposite vertex is outside the perturbed circle."""
    D = built[pid]
    X.check_perturbed(G.get(pid).x, _np(D.simplices))


@pytest.mark.parametrize("pid", SMALL2)
def test_tie_rule_reference(built, pid):
    """§4 item 2: the simplices are those of the exhaustive reference of the perturbed point set."""
    x = G.get(pid).x
    assert [tuple(r) for r in _np(built[pid].simplices).tolist()] == X.delaunay_reference(x)


def test_large_uniform():
    x = np.random.default_rng(17).random((4000, 2))
    D = Delaunay(x)
    X.check_delaunay(x, _np(D.simplices), neighbors=_np(D.neighbors), exhaustive=0)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 13, 17, 33, 64, 65, 200])
def test_small_counts(n):
    """Every size of the recursion's base cases and splits (two and three points, odd and even halves)."""
    x = np.random.default_rng(n).integers(-6, 7, (n, 2)).astype(np.float64)
    if len({tuple(r) for r in x.tolist()}) < 3 or X.orient(*X.fpoints(x[:3])) == 0 and n == 3:
        x[0] = [100.0, -37.0]
        x[1] = [-50.0, 81.0]
    D = Delaunay(x)
    X.check_delaunay(x, _np(D.simplices), neighbors=_np(D.neighbors), coplanar=_np(D.coplanar))
    X.check_perturbed(x, _np(D.simplices))


def test_collinear_halves():
    """Halves of the recursion that are whole lines: a vertical and a horizontal line of points with one
    point off them, and points on the lines of the cuts."""
    xs = [np.array([[0.0, float(i)] for i in range(40)] + [[1.0, 0.5]]),
          np.array([[float(i), 0.0] for i in range(40)] + [[0.5, 1.0]]),
          np.array([[float(i % 7), float(i // 7)] for i in range(49)] + [[3.0, 100.0]]),
          np.array([[float(i), float(2 * i)] for i in range(30)] + [[0.0, 1.0], [5.0, 0.0]])]  # fmt: skip
    for x in xs:
        D = Delaunay(x)
        X.check_delaunay(x, _np(D.simplices), neighbors=_np(D.neighbors))
        X.check_perturbed(x, _np(D.simplices))


def test_cocircular_large():
    """Many points on one circle (exact integer points) and a fine integer grid: every cell cocircular."""
    pts = sorted({(a, b) for a in range(-65, 66) for b in range(-65, 66) if a * a + b * b == 4225})
    x = np.array(pts, dtype=np.float64)
    D = Delaunay(x)
    X.check_delaunay(x, _np(D.simplices), neighbors=_np(D.neighbors))
    X.check_perturbed(x, _np(D.simplices))
    g = np.array([[i, j] for i in range(60) for j in range(60)], dtype=np.float64)
    g = g[np.random.default_rng(3).permutation(len(g))]
    D = Delaunay(g)
    X.check_delaunay(g, _np(D.simplices), neighbors=_np(D.neighbors), exhaustive=0)
    X.check_perturbed(g, _np(D.simplices))


# ---------------------------------------------------------------- §4 item 3: the canonical form


def _canonical_rows(S):
    for r in S:
        a, b, c = r
        assert a < b and a < c, r
    assert S.tolist() == sorted(S.tolist())


@pytest.mark.parametrize("pid", VALID2)
def test_canonical_form(built, pid):
    D = built[pid]
    S = _np(D.simplices)
    _canonical_rows(S)
    assert S.dtype == np.int64 and _np(D.neighbors).dtype == np.int64
    H = _np(D.convex_hull)
    assert H.tolist() == sorted(H.tolist())
    pts = X.fpoints(G.get(pid).x)
    for i, j in H.tolist():  # outward: every point on the left of (i, j) or on its line
        assert all(X.orient(pts[i], pts[j], p) >= 0 for p in pts[:: max(1, len(pts) // 50)])
    bound = {(int(S[t, (k + 1) % 3]), int(S[t, (k + 2) % 3])) for t, k in zip(*np.nonzero(_np(D.neighbors) < 0))}
    assert bound == {tuple(r) for r in H.tolist()}


@pytest.mark.parametrize("pid", ["uniform2/1000", "grid2/30", "dup2", "circle2/int", "boundary2", "far2/1e8grid"])
def test_shuffled_rows(pid):
    """§4 item 3: shuffling the input's rows renames the indices and changes nothing else."""
    x = G.get(pid).x
    perm = np.random.default_rng(7).permutation(len(x))
    A, B = Delaunay(x), Delaunay(x[perm])
    rep = X.representatives(X.fpoints(x))
    repB = X.representatives(X.fpoints(x[perm]))

    def key(sim, p, r):
        return {frozenset(tuple(x[p[r[v]]].tolist()) for v in row) for row in _np(sim).tolist()}

    ident = np.arange(len(x))
    assert key(A.simplices, ident, rep) == key(B.simplices, perm, repB)
    _canonical_rows(_np(B.simplices))


# ---------------------------------------------------------------- the attributes


@pytest.mark.parametrize("pid", VALID2)
def test_attributes(built, pid):
    x = G.get(pid).x
    D = built[pid]
    n = len(x)
    S = _np(D.simplices)
    assert D.ndim == 2 and D.npoints == n and D.nsimplex == len(S)
    assert _np(D.points).tobytes() == x.tobytes()
    np.testing.assert_array_equal(_np(D.min_bound), x.min(axis=0))
    np.testing.assert_array_equal(_np(D.max_bound), x.max(axis=0))
    rep = X.representatives(X.fpoints(x))
    v2s = _np(D.vertex_to_simplex)
    first = {}
    for t, row in enumerate(S.tolist()):
        for v in row:
            first.setdefault(v, t)
    assert v2s.tolist() == [first[rep[i]] for i in range(n)]
    indptr, indices = (_np(a) for a in D.vertex_neighbor_vertices)
    nbrs = [set() for _ in range(n)]
    for a, b, c in S.tolist():
        for p, q in ((a, b), (b, c), (c, a)):
            nbrs[p].add(q)
            nbrs[q].add(p)
    assert indptr.tolist() == np.concatenate([[0], np.cumsum([len(s) for s in nbrs])]).tolist()
    assert indices.tolist() == [v for s in nbrs for v in sorted(s)]
    C = _np(D.coplanar)
    assert C.shape == (sum(rep[i] != i for i in range(n)), 3)


@pytest.mark.parametrize("pid", ["uniform2/1000", "grid2/30+1e-13", "parabola2/1e-14", "far2/1e6uniform",
                                 "mixed2/1e150", "subnormal2", "ulp2/200"])  # fmt: skip
def test_transform_contract(built, pid):
    """§4: every entry of T^-1 within 4u of the exact one (relative), the last row exact."""
    x = G.get(pid).x
    D = built[pid]
    S, T = _np(D.simplices), _np(D.transform)
    assert T.shape == (len(S), 3, 2)
    for q in range(0, len(S), max(1, len(S) // 300)):
        a, b, r = (X.frac(x[v]) for v in S[q])
        m = [[a[0] - r[0], b[0] - r[0]], [a[1] - r[1], b[1] - r[1]]]
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        inv = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]
        for i, j in itertools.product(range(2), range(2)):
            got = T[q, i, j]
            if not np.isfinite(got) or (got == 0 and inv[i][j] != 0):
                continue  # outside the contract: overflow or underflow of the result
            assert abs(Fraction(float(got)) - inv[i][j]) <= Fraction(4 * U) * abs(inv[i][j]), (q, i, j)
        assert T[q, 2].tolist() == x[S[q, 2]].tolist()


def test_attributes_read_only_and_cached():
    D = Delaunay(G.get("uniform2/12").x)
    for name in ("points", "simplices", "neighbors", "vertex_to_simplex", "convex_hull", "coplanar", "transform",
                 "min_bound", "max_bound"):  # fmt: skip
        a = getattr(D, name)
        assert a is getattr(D, name)
        assert not a._flags()["writeable"], name
    for a in D.vertex_neighbor_vertices:
        assert not a._flags()["writeable"]


def test_points_are_copied():
    x = G.get("uniform2/12").x.copy()
    s = sd.asarray(x.copy())
    D = Delaunay(s)
    s[0, 0] = 99.0
    assert _np(D.points).tobytes() == x.tobytes()


def test_inputs():
    pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
    a = Delaunay(pts)
    for v in (np.array(pts), np.array(pts, dtype=np.float32), sd.asarray(pts), np.array(pts) == 1):
        b = Delaunay(v)
        assert _np(b.points).dtype == np.float64
    assert _np(a.simplices).tolist() == _np(Delaunay(np.array(pts, dtype=float)).simplices).tolist()
    fr = sd.frame.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]})
    assert _np(Delaunay(fr).simplices).tolist() == _np(a.simplices).tolist()


def test_duplicates_and_negative_zero():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [-0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -0.0]])
    D = Delaunay(x)
    assert _np(D.simplices).tolist() == [[0, 1, 3]]
    assert _np(D.coplanar).tolist() == [[2, 0, 0], [4, 0, 1], [5, 0, 0]]
    assert _np(D.vertex_to_simplex).tolist() == [0] * 6
    indptr, indices = D.vertex_neighbor_vertices
    assert _np(indptr).tolist() == [0, 2, 4, 4, 6, 6, 6]
    assert _np(indices).tolist() == [1, 3, 0, 3, 0, 1]


# ---------------------------------------------------------------- §4 item 4: find_simplex


def _queries(x, rng):
    """Vertices, midpoints of point pairs, points of the box and of a box 20% larger (convex combinations of
    the corners: no overflow near DBL_MAX)."""
    lo, hi = x.min(axis=0), x.max(axis=0)
    k = min(40, len(x) - 1)
    inner = rng.random((200, 2))
    outer = rng.random((60, 2)) * 1.2 - 0.1
    with np.errstate(over="ignore", invalid="ignore"):
        q = [x[:50], x[:k] / 2 + x[1 : k + 1] / 2, lo * (1 - inner) + hi * inner, lo * (1 - outer) + hi * outer]
        q = np.concatenate(q)
    return q[np.isfinite(q).all(axis=1)]


@pytest.mark.parametrize("pid", VALID2)
def test_find_simplex(built, pid):
    x = G.get(pid).x
    D = built[pid]
    q = _queries(x, np.random.default_rng(5))
    if len(_np(D.simplices)) * len(q) > 3_000_000:
        q = q[::4]
    X.check_find_simplex(x, _np(D.simplices), q, _np(D.find_simplex(q)))


def test_find_simplex_on_edges_and_vertices():
    """Points on edges and vertices of an integer grid (exact): the smallest of the simplices holding them."""
    x = G.get("grid2/30").x
    D = Delaunay(x)
    q = np.array([[a / 2, b / 2] for a in range(-1, 60) for b in range(-1, 60, 3)])
    X.check_find_simplex(x, _np(D.simplices), q, _np(D.find_simplex(q)))
    v = _np(D.find_simplex(x))
    assert v.tolist() == _np(D.vertex_to_simplex).tolist()


def test_find_simplex_shapes_and_non_finite():
    D = Delaunay([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert D.find_simplex([0.25, 0.25]) == 0 and type(D.find_simplex([0.25, 0.25])) is int
    assert D.find_simplex([2.0, 2.0]) == -1
    assert D.find_simplex([1.0, 0.0]) == 0 and D.find_simplex([1.0, 1.0]) == 1  # vertices: the smallest
    assert D.find_simplex([0.5, 0.5]) == 0  # the diagonal
    q = np.array([[[0.25, 0.25], [np.nan, 0.5]], [[0.5, np.inf], [0.75, 0.75]]])
    out = D.find_simplex(q)
    assert _np(out).tolist() == [[0, -1], [-1, 1]] and _np(out).dtype == np.int64
    assert _np(D.find_simplex(np.zeros((0, 2)))).shape == (0,)
    assert _np(D.find_simplex(np.zeros((3, 0, 2)))).shape == (3, 0)
    assert D.find_simplex([1.0 + 2**-52, 0.0]) == -1  # just outside: no tolerance (SPT103)
    with pytest.raises(ValueError, match=r"find_simplex\(\): 'xi' must have 2 entries on its last axis, not 3"):
        D.find_simplex(np.zeros((4, 3)))
    with pytest.raises(TypeError, match="complex"):
        D.find_simplex(np.zeros((4, 2), dtype=complex))


def test_find_simplex_wide_range():
    """Query points whose bits fall outside the points' range promise take the wide path."""
    x = G.get("uniform2/1000").x
    D = Delaunay(x)
    q = np.array([[0.5 + 2**-80, 0.5], [1e200, 0.5], [5e-324, 0.3], [0.3, 1e-300]] + x[:5].tolist())
    X.check_find_simplex(x, _np(D.simplices), q, _np(D.find_simplex(q)))


# ---------------------------------------------------------------- errors, pickle, repr


@pytest.mark.parametrize(
    "x, exc, msg",
    [
        ([[0.0, 0.0], [1.0, 1.0]], ValueError, r"Delaunay\(\): needs at least 3 distinct points, got 2"),
        ([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], ValueError, "got 2"),
        (np.zeros((0, 2)), ValueError, "got 0"),
        ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [-3.0, -3.0]], ValueError, r"Delaunay\(\): 'points' are collinear"),
        ([[0.0, 0.0], [1.0, np.nan], [2.0, 0.0]], ValueError, r"Delaunay\(\): 'points' must be finite"),
        ([[0.0, 0.0], [1.0, np.inf], [2.0, 0.0]], ValueError, "must be finite"),
        (np.zeros((4, 4)), ValueError, r"Delaunay\(\): 'points' must have 2 or 3 columns, not 4"),
        (np.zeros((4, 1)), ValueError, "not 1"),
        (np.zeros(4), ValueError, r"Delaunay\(\): 'points' must have 2 dimensions, not 1"),
        (np.zeros((4, 2), dtype=complex), TypeError, r"Delaunay\(\): 'points' must be real"),
        ("abc", TypeError, r"Delaunay\(\): 'points' must be"),
    ],
)
def test_errors(x, exc, msg):
    with pytest.raises(exc, match=msg):
        Delaunay(x)


def test_invalid_sets():
    for pid in G.ids("d2", "invalid"):
        with pytest.raises(ValueError, match=r"collinear|distinct"):
            Delaunay(G.get(pid).x)


def test_near_collinear_is_triangulated():
    x = G.get("parabola2/1e-14").x
    D = Delaunay(x)
    X.check_delaunay(x, _np(D.simplices), exhaustive=0)


def test_call_shape():
    with pytest.raises(TypeError, match=r"Delaunay\(\): unexpected keyword argument 'qhull_options'"):
        Delaunay([[0, 0], [1, 0], [0, 1]], qhull_options="Qbb")
    with pytest.raises(TypeError, match=r"Delaunay\(\): unexpected keyword argument 'furthest_site'"):
        Delaunay([[0, 0], [1, 0], [0, 1]], furthest_site=True)
    D = Delaunay([[0, 0], [1, 0], [0, 1]])
    with pytest.raises(TypeError, match=r"find_simplex\(\): unexpected keyword argument 'tol'"):
        D.find_simplex([0.1, 0.1], tol=1e-9)


def test_pickle_copy_repr():
    x = G.get("dup2").x
    D = Delaunay(x)
    assert repr(D) == f"Delaunay(n={len(x)}, d=2, nsimplex={D.nsimplex})"
    for E in (pickle.loads(pickle.dumps(D)), copy.copy(D), copy.deepcopy(D)):
        assert type(E) is Delaunay and E is not D
        for name in ("points", "simplices", "neighbors", "vertex_to_simplex", "coplanar", "convex_hull"):
            assert _np(getattr(E, name)).tobytes() == _np(getattr(D, name)).tobytes(), name
    assert repr(Delaunay(G.get("uniform2/1000").x)) == "Delaunay(n=1000, d=2, nsimplex=1982)"


def test_three_d_not_yet():
    with pytest.raises(ValueError, match="SPT10"):
        Delaunay(np.random.default_rng(0).random((10, 3)))


# ---------------------------------------------------------------- scipy on points in general position


@LIVE
@pytest.mark.parametrize("pid", ["uniform2/1000", "normal2/1000", "cluster2/1000", "ball2/1000", "uniform2/20000"])
def test_same_simplices_as_scipy(pid):
    import scipy.spatial as S

    x = G.get(pid).x
    ref = S.Delaunay(x)
    D = Delaunay(x)
    assert {frozenset(r) for r in ref.simplices.tolist()} == {frozenset(r) for r in _np(D.simplices).tolist()}
    tr = ref.transform
    order = {frozenset(r): k for k, r in enumerate(ref.simplices.tolist())}
    S_ = _np(D.simplices)
    T = _np(D.transform)
    q = np.random.default_rng(1).random((500, 2)) * (x.max(0) - x.min(0)) + x.min(0)
    found = _np(D.find_simplex(q))
    inside = found >= 0
    assert ((ref.find_simplex(q) >= 0) == inside).all()
    for k in range(0, len(S_), 37):  # barycentric coordinates agree to the conditioning's rounding
        t = order[frozenset(S_[k].tolist())]
        p = x[S_[k]].mean(axis=0)
        c_sd = T[k, :2] @ (p - T[k, 2])
        c_sc = tr[t, :2] @ (p - tr[t, 2])
        lam_sd = dict(zip(S_[k].tolist(), [*c_sd, 1 - c_sd.sum()]))
        lam_sc = dict(zip(ref.simplices[t].tolist(), [*c_sc, 1 - c_sc.sum()]))
        for v in lam_sd:
            assert abs(lam_sd[v] - lam_sc[v]) < 1e-9
