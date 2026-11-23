"""Delaunay for d = 3 (docs/design/spatial_geometry.md §2.3, §3.4, §3.5, §4 items 1-4; milestone SPT10): the
triangulation by the exact checkers of exactgeom.py (and, for thousands of points, a float filter whose
uncertain rows go to the same exact predicates), the tie rule against the exhaustive reference, the canonical
form, the attributes by their definitions (vertex_neighbor_vertices, transform's contract), find_simplex,
the MemoryError of the build's limit, errors, pickle, and scipy's simplices on points in general position."""

import copy
import itertools
import pickle
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd
from stardust.spatial import _delaunay

from . import exactgeom as X
from . import geompoints as G
from .conftest import LIVE

Delaunay = sd.spatial.Delaunay
U = 2.0**-53

VALID3 = [i for i in G.ids("d3") if "invalid" not in G.get(i).tags and "large" not in G.get(i).tags]
SMALL3 = [i for i in G.ids("d3", "small") if i in VALID3]
# Exact checks of every pair in Fractions: the sets of a few hundred points; the others take the filter.
EXACT3 = [i for i in VALID3 if len(G.get(i).x) <= 250]
FILTER3 = [i for i in VALID3 if i not in EXACT3]


def _np(a):
    return np.asarray(a)


@pytest.fixture(scope="module")
def built():
    return {i: Delaunay(G.get(i).x) for i in VALID3}


# The filtered exact checker for large sets is exactgeom.check_delaunay3_large (the accuracy sweep delaunay_exact
# uses it too).
check_large = X.check_delaunay3_large


# ---------------------------------------------------------------- §4 items 1 and 2


@pytest.mark.parametrize("pid", EXACT3)
def test_triangulation_is_delaunay(built, pid):
    """§4 item 1 in Fractions: positive, a cover of the hull, every distinct point a vertex, empty spheres (every
    (simplex, point) pair up to 40 distinct points); the neighbours and coplanar rows by their definitions."""
    x = G.get(pid).x
    D = built[pid]
    X.check_delaunay(x, _np(D.simplices), neighbors=_np(D.neighbors), coplanar=_np(D.coplanar), exhaustive=40)


@pytest.mark.parametrize("pid", EXACT3)
def test_tie_rule_local(built, pid):
    """§4 item 2: across every inner facet the opposite vertex is outside the perturbed sphere."""
    X.check_perturbed(G.get(pid).x, _np(built[pid].simplices))


@pytest.mark.parametrize("pid", FILTER3)
def test_triangulation_filtered(built, pid):
    """§4 items 1 and 2 on the sets of a thousand points (the filter; open rows exact)."""
    D = built[pid]
    check_large(G.get(pid).x, _np(D.simplices), _np(D.neighbors))


@pytest.mark.parametrize("pid", SMALL3)
def test_tie_rule_reference(built, pid):
    """§4 item 2: the simplices are those of the exhaustive reference of the perturbed point set."""
    x = G.get(pid).x
    assert [tuple(r) for r in _np(built[pid].simplices).tolist()] == X.delaunay_reference(x)


@pytest.mark.parametrize("seed", range(6))
def test_tie_rule_reference_cospherical(seed):
    """Small integer sets full of cospherical and coplanar points (a 3 x 3 x 2 grid, points of one sphere, a
    plane with points off it): the exhaustive reference of the perturbed set."""
    g = np.random.default_rng(seed)
    grid = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(2)], dtype=np.float64)
    sphere = np.array([p for p in itertools.product(range(-3, 4), repeat=3) if sum(v * v for v in p) == 9], float)
    sets = [grid, sphere[g.permutation(len(sphere))[:13]],
            np.concatenate([g.integers(-2, 3, (10, 2)), np.zeros((10, 1))], axis=1).astype(float)]  # fmt: skip
    sets[2] = np.concatenate([sets[2], [[0.0, 0.0, 1.0], [1.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])
    for x in sets:
        x = x[g.permutation(len(x))]
        D = Delaunay(x)
        assert [tuple(r) for r in _np(D.simplices).tolist()] == X.delaunay_reference(x)


def test_large_uniform():
    x = np.random.default_rng(17).random((20000, 3))
    D = Delaunay(x)
    check_large(x, _np(D.simplices), _np(D.neighbors))


def test_degenerate_large():
    """A shuffled 12^3 grid (every cell cospherical: the tie rule everywhere) and 2,000 points on a sphere of
    integer points with duplicates: the filtered checker, whose open rows here are many."""
    g = np.random.default_rng(23)
    grid = np.array(list(itertools.product(range(12), repeat=3)), dtype=np.float64)[g.permutation(1728)]
    D = Delaunay(grid)
    check_large(grid, _np(D.simplices), _np(D.neighbors))
    sphere = np.array([p for p in itertools.product(range(-9, 10), repeat=3) if sum(v * v for v in p) == 81], float)
    x = np.concatenate([sphere, sphere[:40], g.random((200, 3)) * 8 - 4])[g.permutation(len(sphere) + 240)]
    D = Delaunay(x)
    check_large(x, _np(D.simplices), _np(D.neighbors))


def test_coplanar_and_collinear_parts():
    """Points on a few planes and lines with points off them: the first tetrahedron's search and the hull
    facets that new points are coplanar with (the ghosts' tie rule)."""
    xs = [np.array([[float(i), 0.0, 0.0] for i in range(20)] + [[0.0, 1.0, 0.0], [3.0, 2.0, 1.0]]),
          np.array([[float(i % 5), float(i // 5), 0.0] for i in range(25)] + [[2.0, 2.0, 3.0]]),
          np.array([[float(i % 5), float(i // 5), 0.0] for i in range(25)] + [[2.0, 2.0, 3.0], [1.0, 1.0, -2.0]]),
          np.array([[float(i), float(i), float(i)] for i in range(10)] + [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                                                         [0.0, 0.0, 1.0]])]  # fmt: skip
    for x in xs:
        D = Delaunay(x)
        X.check_delaunay(x, _np(D.simplices), neighbors=_np(D.neighbors), exhaustive=30)
        X.check_perturbed(x, _np(D.simplices))


# ---------------------------------------------------------------- §4 item 3: the canonical form


def _canonical_rows(S):
    for r in S.tolist():
        a, b, c, d = r
        assert a < b and a < c and a < d and b < c and b < d, r
    assert S.tolist() == sorted(S.tolist())


@pytest.mark.parametrize("pid", VALID3)
def test_canonical_form(built, pid):
    D = built[pid]
    S = _np(D.simplices)
    _canonical_rows(S)
    assert S.dtype == np.int64 and _np(D.neighbors).dtype == np.int64
    H = _np(D.convex_hull)
    assert H.shape[1] == 3 and H.tolist() == sorted(H.tolist())
    assert all(r[0] < r[1] and r[0] < r[2] for r in H.tolist())
    pts = X.fpoints(G.get(pid).x)
    inner = pts[S[0, 0]], pts[S[0, 1]], pts[S[0, 2]], pts[S[0, 3]]
    g = tuple(sum(c) / 4 for c in zip(*inner))
    for i, j, k in H.tolist()[:: max(1, len(H) // 60)]:  # outward: the interior on the negative side
        assert X.orient(pts[i], pts[j], pts[k], g) < 0
    nb = _np(D.neighbors)
    bound = set()
    for t, k in zip(*np.nonzero(nb < 0)):
        f = [int(v) for j, v in enumerate(S[t]) if j != k]
        bound.add(frozenset(f))
    assert bound == {frozenset(r) for r in H.tolist()}


@pytest.mark.parametrize("pid", ["uniform3/1000", "grid3/6", "dup3", "sphere3/int", "boundary3", "far3/1e8grid",
                                 "plane3+1", "mixed3/1e300"])  # fmt: skip
def test_shuffled_rows(pid):
    """§4 item 3: shuffling the input's rows renames the indices and changes nothing else."""
    x = G.get(pid).x
    perm = np.random.default_rng(7).permutation(len(x))
    A, B = Delaunay(x), Delaunay(x[perm])
    rep = X.representatives(X.fpoints(x))
    repB = X.representatives(X.fpoints(x[perm]))

    def key(sim, p, r):
        return {frozenset(tuple(x[p[r[v]]].tolist()) for v in row) for row in _np(sim).tolist()}

    assert key(A.simplices, np.arange(len(x)), rep) == key(B.simplices, perm, repB)
    _canonical_rows(_np(B.simplices))


# ---------------------------------------------------------------- the attributes


@pytest.mark.parametrize("pid", VALID3)
def test_attributes(built, pid):
    x = G.get(pid).x
    D = built[pid]
    n = len(x)
    S = _np(D.simplices)
    assert D.ndim == 3 and D.npoints == n and D.nsimplex == len(S) and S.shape[1] == 4
    assert _np(D.points).tobytes() == x.tobytes()
    np.testing.assert_array_equal(_np(D.min_bound), x.min(axis=0))
    np.testing.assert_array_equal(_np(D.max_bound), x.max(axis=0))
    rep = X.representatives(X.fpoints(x))
    first = {}
    for t, row in enumerate(S.tolist()):
        for v in row:
            first.setdefault(v, t)
    assert _np(D.vertex_to_simplex).tolist() == [first[rep[i]] for i in range(n)]
    indptr, indices = (_np(a) for a in D.vertex_neighbor_vertices)
    nbrs = [set() for _ in range(n)]
    for row in S.tolist():
        for p, q in itertools.combinations(row, 2):
            nbrs[p].add(q)
            nbrs[q].add(p)
    assert indptr.tolist() == np.concatenate([[0], np.cumsum([len(s) for s in nbrs])]).tolist()
    assert indices.tolist() == [v for s in nbrs for v in sorted(s)]
    assert indptr.dtype == np.int64 and indices.dtype == np.int64
    assert _np(D.coplanar).shape == (sum(rep[i] != i for i in range(n)), 3)


@pytest.mark.parametrize("pid", ["uniform3/1000", "cluster3/1000", "ulp3/200", "far3/1e8grid", "mixed3/1e150",
                                 "subnormal3", "moment3", "grid3/6"])  # fmt: skip
def test_transform_contract(built, pid):
    """§4: every entry of T^-1 within 4u of the exact one (relative), the last row exact."""
    x = G.get(pid).x
    D = built[pid]
    S, T = _np(D.simplices), _np(D.transform)
    assert T.shape == (len(S), 4, 3) and T.dtype == np.float64
    for q in range(0, len(S), max(1, len(S) // 150)):
        v = [X.frac(x[i]) for i in S[q]]
        cols = [[v[k][c] - v[3][c] for k in range(3)] for c in range(3)]  # T[c][k] = (v_k - v_3)_c
        det = X.det(cols)
        for i, j in itertools.product(range(3), range(3)):
            minor = [[cols[r][c] for c in range(3) if c != i] for r in range(3) if r != j]
            inv = (-1) ** (i + j) * X.det(minor) / det  # (T^-1)[i][j] = cofactor(j, i) / det
            got = T[q, i, j]
            if not np.isfinite(got) or (got == 0 and inv != 0) or (inv != 0 and abs(inv) < 2.0**-1022):
                continue  # outside the contract: overflow or underflow of the result
            assert abs(Fraction(float(got)) - inv) <= Fraction(4 * U) * abs(inv), (q, i, j)
        assert T[q, 3].tolist() == x[S[q, 3]].tolist()


def test_transform_barycentric():
    """The barycentric coordinates T^-1 (x - r) and 1 - their sum give back a point from its simplex."""
    x = G.get("uniform3/1000").x
    D = Delaunay(x)
    S, T = _np(D.simplices), _np(D.transform)
    lam = np.random.default_rng(3).dirichlet(np.ones(4), len(S))
    p = (x[S] * lam[:, :, None]).sum(axis=1)
    c = np.einsum("tij,tj->ti", T[:, :3], p - T[:, 3])
    np.testing.assert_allclose(c, lam[:, :3], atol=1e-9)


def test_attributes_read_only_and_cached():
    D = Delaunay(G.get("uniform3/12").x)
    for name in ("points", "simplices", "neighbors", "vertex_to_simplex", "convex_hull", "coplanar", "transform",
                 "min_bound", "max_bound"):  # fmt: skip
        a = getattr(D, name)
        assert a is getattr(D, name)
        assert not a._flags()["writeable"], name
    for a in D.vertex_neighbor_vertices:
        assert not a._flags()["writeable"]


def test_points_are_copied_from_any_strides():
    x = G.get("uniform3/1000").x
    f = np.asfortranarray(x)
    s = sd.asarray(x.copy())
    D = Delaunay(s)
    s[0, 0] = 99.0
    assert _np(D.points).tobytes() == x.tobytes()
    assert _np(Delaunay(f).simplices).tobytes() == _np(D.simplices).tobytes()
    assert _np(Delaunay(sd.asarray(np.concatenate([x, x], axis=1))[:, ::2]).simplices).shape[1] == 4


def test_duplicates_and_negative_zero():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.0, 0.0, -0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0], [0.0, -0.0, 0.0]])  # fmt: skip
    D = Delaunay(x)
    assert _np(D.simplices).tolist() == [[0, 1, 3, 4]]
    assert _np(D.coplanar).tolist() == [[2, 0, 0], [5, 0, 1], [6, 0, 0]]
    assert _np(D.vertex_to_simplex).tolist() == [0] * 7
    indptr, indices = D.vertex_neighbor_vertices
    assert _np(indptr).tolist() == [0, 3, 6, 6, 9, 12, 12, 12]
    assert _np(indices).tolist() == [1, 3, 4, 0, 3, 4, 0, 1, 4, 0, 1, 3]
    assert _np(D.convex_hull).tolist() == [[0, 1, 4], [0, 3, 1], [0, 4, 3], [1, 3, 4]]


# ---------------------------------------------------------------- §4 item 4: find_simplex


def _queries(x, rng):
    """Vertices, midpoints of point pairs (edges, faces and insides), points of the box and of a box 20% larger."""
    lo, hi = x.min(axis=0), x.max(axis=0)
    k = min(40, len(x) - 1)
    inner = rng.random((150, 3))
    outer = rng.random((50, 3)) * 1.2 - 0.1
    with np.errstate(over="ignore", invalid="ignore"):
        q = [x[:40], x[:k] / 2 + x[1 : k + 1] / 2, lo * (1 - inner) + hi * inner, lo * (1 - outer) + hi * outer]
        q = np.concatenate(q)
    return q[np.isfinite(q).all(axis=1)]


@pytest.mark.parametrize("pid", VALID3)
def test_find_simplex(built, pid):
    x = G.get(pid).x
    D = built[pid]
    q = _queries(x, np.random.default_rng(5))
    S = _np(D.simplices)
    if len(S) * len(q) > 250_000:  # the reference tries every simplex in Fractions
        q = q[:: -(-len(S) * len(q) // 250_000)]
    X.check_find_simplex(x, S, q, _np(D.find_simplex(q)))


def test_find_simplex_on_edges_faces_vertices():
    """Points on edges, faces and vertices of an integer grid (exact): the smallest simplex holding them."""
    x = np.array(list(itertools.product(range(4), repeat=3)), dtype=np.float64)
    D = Delaunay(x)
    q = np.array([[a / 2, b / 2, c / 2] for a in range(-1, 8, 2) for b in range(0, 7) for c in range(0, 7, 3)])
    X.check_find_simplex(x, _np(D.simplices), q, _np(D.find_simplex(q)))
    assert _np(D.find_simplex(x)).tolist() == _np(D.vertex_to_simplex).tolist()


def test_find_simplex_shapes_and_non_finite():
    D = Delaunay([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    assert D.find_simplex([0.1, 0.1, 0.1]) == 0 and type(D.find_simplex([0.1, 0.1, 0.1])) is int
    assert D.find_simplex([2.0, 2.0, 2.0]) == -1
    q = np.array([[[0.1, 0.1, 0.1], [np.nan, 0.5, 0.5]], [[0.5, np.inf, 0.0], [0.9, 0.9, 0.9]]])
    out = _np(D.find_simplex(q))
    assert out.shape == (2, 2) and out.dtype == np.int64 and out[0, 1] == -1 and out[1, 0] == -1
    assert out[0, 0] == 0 and out[1, 1] >= 0
    assert _np(D.find_simplex(np.zeros((0, 3)))).shape == (0,)
    assert D.find_simplex([1.0 + 2**-52, 0.0, 0.0]) == -1  # just outside: no tolerance (SPT103)
    with pytest.raises(ValueError, match=r"find_simplex\(\): 'xi' must have 3 entries on its last axis, not 2"):
        D.find_simplex(np.zeros((4, 2)))
    with pytest.raises(TypeError, match="complex"):
        D.find_simplex(np.zeros((4, 3), dtype=complex))


def test_find_simplex_wide_range():
    """Query points whose bits fall outside the points' range promise take the wide path."""
    x = G.get("uniform3/1000").x
    D = Delaunay(x)
    q = np.array([[0.5 + 2**-80, 0.5, 0.5], [1e200, 0.5, 0.5], [5e-324, 0.3, 0.4], [0.3, 1e-300, 0.5]] + x[:5].tolist())
    X.check_find_simplex(x, _np(D.simplices), q, _np(D.find_simplex(q)))


# ---------------------------------------------------------------- memory, errors, pickle, repr


def test_memory_limit():
    """The build counts its tetrahedra (ghosts included): past the limit a MemoryError, and nothing is kept (an
    injected limit stands for physical memory's share)."""
    x = np.random.default_rng(2).random((3000, 3))
    with pytest.raises(MemoryError, match=r"^Delaunay\(\): the triangulation is larger than this machine's physical"):
        _delaunay._limited(x, 1000)
    with pytest.raises(MemoryError):
        _delaunay._limited(x, 5)  # not even the first tetrahedron and its ghosts
    D = _delaunay._limited(x, 40_000)
    assert _np(D.simplices).tobytes() == _np(Delaunay(x).simplices).tobytes()
    moment = np.array([[t, t * t, t * t * t] for t in np.linspace(-1, 1, 300)])  # quadratic: ~n^2 / 2 tetrahedra
    with pytest.raises(MemoryError):
        _delaunay._limited(moment, 10_000)
    assert Delaunay(moment).nsimplex > 10_000


@pytest.mark.parametrize(
    "x, exc, msg",
    [
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], ValueError,
         r"Delaunay\(\): needs at least 4 distinct points, got 3"),
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-0.0, 0.0, 0.0]], ValueError, "got 2"),
        (np.zeros((0, 3)), ValueError, "got 0"),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [5.0, -2.0, 0.0]], ValueError,
         r"Delaunay\(\): 'points' are coplanar"),
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-3.0, -3.0, -3.0]], ValueError, "coplanar"),
        ([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]], ValueError, "must be finite"),
        (np.zeros((4, 3), dtype=complex), TypeError, r"Delaunay\(\): 'points' must be real"),
    ],
)  # fmt: skip
def test_errors(x, exc, msg):
    with pytest.raises(exc, match=msg):
        Delaunay(x)


def test_invalid_sets():
    for pid in G.ids("d3", "invalid"):
        with pytest.raises(ValueError, match=r"coplanar|distinct"):
            Delaunay(G.get(pid).x)


def test_near_coplanar_is_triangulated():
    """z = 1e-14 * (x^2 + y^2): not coplanar, so triangulated exactly (scipy: QH6154 for such flat sets)."""
    g = np.random.default_rng(9).random((300, 2))
    x = np.column_stack([g, 1e-14 * (g * g).sum(axis=1)])
    D = Delaunay(x)
    check_large(x, _np(D.simplices), _np(D.neighbors))


def test_pickle_copy_repr():
    x = G.get("dup3").x
    D = Delaunay(x)
    assert repr(D) == f"Delaunay(n={len(x)}, d=3, nsimplex={D.nsimplex})"
    for E in (pickle.loads(pickle.dumps(D)), copy.copy(D), copy.deepcopy(D)):
        assert type(E) is Delaunay and E is not D
        for name in ("points", "simplices", "neighbors", "vertex_to_simplex", "coplanar", "convex_hull"):
            assert _np(getattr(E, name)).tobytes() == _np(getattr(D, name)).tobytes(), name


# ---------------------------------------------------------------- scipy on points in general position


@LIVE
@pytest.mark.parametrize("pid", ["uniform3/1000", "normal3/1000", "cluster3/1000", "ball3/1000", "uniform3/20000"])
def test_same_simplices_as_scipy(pid):
    import scipy.spatial as S

    x = G.get(pid).x
    ref = S.Delaunay(x)
    D = Delaunay(x)
    assert {frozenset(r) for r in ref.simplices.tolist()} == {frozenset(r) for r in _np(D.simplices).tolist()}
    order = {frozenset(r): k for k, r in enumerate(ref.simplices.tolist())}
    S_, T, tr = _np(D.simplices), _np(D.transform), ref.transform
    q = np.random.default_rng(1).random((500, 3)) * (x.max(0) - x.min(0)) + x.min(0)
    assert ((ref.find_simplex(q) >= 0) == (_np(D.find_simplex(q)) >= 0)).all()
    for k in range(0, len(S_), 97):  # barycentric coordinates agree to the conditioning's rounding
        t = order[frozenset(S_[k].tolist())]
        p = x[S_[k]].mean(axis=0)
        c_sd = T[k, :3] @ (p - T[k, 3])
        c_sc = tr[t, :3] @ (p - tr[t, 3])
        lam_sd = dict(zip(S_[k].tolist(), [*c_sd, 1 - c_sd.sum()]))
        lam_sc = dict(zip(ref.simplices[t].tolist(), [*c_sc, 1 - c_sc.sum()]))
        for v in lam_sd:
            assert abs(lam_sd[v] - lam_sc[v]) < 1e-8
    indptr, indices = (np.asarray(a) for a in ref.vertex_neighbor_vertices)
    mine_p, mine_i = (_np(a) for a in D.vertex_neighbor_vertices)
    for i in range(0, len(x), 53):
        assert sorted(indices[indptr[i] : indptr[i + 1]].tolist()) == mine_i[mine_p[i] : mine_p[i + 1]].tolist()
