"""Voronoi (docs/design/spatial_geometry.md §2.5, §3.7, §4 item 6 and the vertex contract; milestone SPT12): the
diagram against an exact reference built from the Delaunay triangulation in rational arithmetic (clusters by
their exact circumcentre, the ridges' sizes by exact collinearity, and for small sets the definition by
exactgeom.voronoi_ridge_exists), the vertex numbering and the §4 bound of every vertex (Fractions), the vertex
kernel's numerators (1.002 u), the orders of §2.5 (counterclockwise checked with exact centres), equal points,
shuffled rows, errors, pickle, the MemoryError of the 3-D build, and scipy on points in general position."""

import copy
import itertools
import pickle
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd
from stardust.spatial import _voronoi

from . import exactgeom as X
from . import geompoints as G
from .conftest import LIVE

Voronoi = sd.spatial.Voronoi
_core = sd._core.spatial
U = Fraction(1, 2**53)
TINY = Fraction(1, 2**1074)
DBL_MAX = Fraction(float(np.finfo(np.float64).max))

VALID = [i for i in G.IDS if "invalid" not in G.get(i).tags and "large" not in G.get(i).tags]
SMALL = [i for i in G.ids("small") if i in VALID]


def _np(a):
    return np.asarray(a)


def _rows(pair, k):
    indptr, indices = (_np(a) for a in pair)
    return indices[indptr[k] : indptr[k + 1]].tolist()


def _all_rows(pair):
    indptr, indices = (_np(a) for a in pair)
    return [indices[indptr[k] : indptr[k + 1]].tolist() for k in range(len(indptr) - 1)]


def build(x):
    """The diagram and the Voronoi vertex of every simplex of the canonical Delaunay triangulation (the test hook
    of _voronoi), with that triangulation's simplices."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    V = Voronoi(x)
    sv = _np(_core._voronoi(sd.asarray(x), -1, True)[8])
    S = _np(sd.spatial.Delaunay(x).simplices)
    return V, sv, S


@pytest.fixture(scope="module")
def built():
    return {}


def get(built, pid):
    if pid not in built:
        built[pid] = build(G.get(pid).x)
    return built[pid]


# ---------------------------------------------------------------- the exact reference


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _sub(p, q):
    return tuple(a - b for a, b in zip(p, q))


def _collinear(P):
    """Are the 3-D points P on one line (or one point)?"""
    p0 = P[0]
    far = next((p for p in P if p != p0), None)
    if far is None:
        return True
    u = _sub(far, p0)
    return all(not any(_cross(u, _sub(p, p0))) for p in P)


def reference(x, simplices):
    """The diagram as the dual of the triangulation, exactly and by geometry (§3.7): clusters of simplices by
    their exact circumcentre; the ridge of an edge with its clusters; a closed ridge kept with d or more
    clusters; an unbounded 3-D ridge kept when its finite vertices and the outward normals of its two hull
    triangles (from the centres of their tetrahedra) are not on one line. Returns (cluster of each simplex,
    {(i, j): set of clusters, -1 for unbounded}, {point: set of clusters of its star, -1 on the hull})."""
    pts = X.fpoints(x)
    d = len(pts[0])
    S = [tuple(int(v) for v in s) for s in simplices]
    centres = [X.circumcenter(*(pts[i] for i in s)) for s in S]
    key = {}
    cluster = [key.setdefault(c, len(key)) for c in centres]
    edges = defaultdict(list)
    for t, s in enumerate(S):
        for i, j in itertools.combinations(sorted(s), 2):
            edges[(i, j)].append(t)
    hull = defaultdict(list)  # edge -> [(hull facet, its tetrahedron)]
    on_hull = set()
    for f, owners in X.adjacency(S).items():
        if len(owners) == 1:
            on_hull.update(f)
            for i, j in itertools.combinations(f, 2):
                hull[(i, j)].append((f, owners[0][0]))
    ridges = {}
    for e, ts in edges.items():
        cs = {cluster[t] for t in ts}
        if e not in hull:
            if len(cs) >= d:
                ridges[e] = cs
        elif d == 2:
            ridges[e] = cs | {-1}
        else:
            P = [centres[t] for t in ts]
            for f, t in hull[e]:
                a, b, c = (pts[i] for i in f)
                n = _cross(_sub(b, a), _sub(c, a))
                other = next(i for i in S[t] if i not in f)
                if X._dot(n, _sub(pts[other], a)) > 0:
                    n = tuple(-v for v in n)
                P.append(tuple(p + q for p, q in zip(centres[t], n)))
            if not _collinear(P):
                ridges[e] = cs | {-1}
    regions = defaultdict(set)
    for t, s in enumerate(S):
        for i in s:
            regions[i].add(cluster[t])
    for i in on_hull:
        regions[i].add(-1)
    return cluster, centres, ridges, regions


def check_structure(x, V, sv, S):
    """§4 item 6 and the numbering against reference()."""
    cluster, centres, ridges, regions = reference(x, S)
    pairs = {}
    for t, (c, v) in enumerate(zip(cluster, sv.tolist())):
        assert pairs.setdefault(c, v) == v, f"simplex {t}: cospherical simplices have different vertices"
    assert len(set(pairs.values())) == len(pairs) == len(_np(V.vertices)), "a vertex for two spheres, or extra"
    first = {}
    for t, v in enumerate(sv.tolist()):
        first.setdefault(v, t)
    assert sorted(first, key=first.get) == list(range(len(first))), "vertices not in their smallest simplex's order"
    rp = _np(V.ridge_points).tolist()
    got = {tuple(e): set(vs) for e, vs in zip(rp, _all_rows(V.ridge_vertices))}
    want = {e: {(-1 if c == -1 else pairs[c]) for c in cs} for e, cs in ridges.items()}
    assert got == want, "the ridges differ from the reference"
    rep = X.representatives(X.fpoints(x))
    for i, row in enumerate(_all_rows(V.regions)):
        if rep[i] != i:
            assert row == [], f"equal point {i} has a region"
        else:
            assert set(row) == {(-1 if c == -1 else pairs[c]) for c in regions[i]}, f"region {i}"
            assert len(row) == len(set(row)), f"region {i} repeats a vertex"
    assert _np(V.point_region).tolist() == rep
    return pairs, centres, cluster


@pytest.mark.parametrize("pid", VALID)
def test_structure_is_exact(built, pid):
    x = G.get(pid).x
    V, sv, S = get(built, pid)
    check_structure(x, V, sv, S)


@pytest.mark.parametrize("pid", SMALL)
def test_ridges_by_definition(built, pid):
    """For the small sets every pair of distinct points against the definition (exactgeom.voronoi_ridge_exists:
    a point closer to the two than to every other, by exact Fourier-Motzkin), through exactgeom.check_voronoi."""
    x = G.get(pid).x
    V, sv, S = get(built, pid)
    X.check_voronoi(x, S, _np(V.ridge_points), [set(r) for r in _all_rows(V.ridge_vertices)], sv, definition=20)


def test_reference_agrees_with_the_definition():
    """The collinearity rule of reference() for unbounded 3-D ridges is the definition's: a flat hull face's
    diagonal has no ridge (its two hull triangles are coplanar), on cubes with points on their faces."""
    cube = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=float)
    pts = [cube, cube[[0, 2, 6, 8, 18, 20, 24, 26, 13]], np.concatenate([cube[:9], [[1.0, 1.0, 1.0], [0.5, 0.25, -1]]])]
    for x in pts:
        S = _np(sd.spatial.Delaunay(x).simplices)
        _, _, ridges, _ = reference(x, S)
        reps = sorted(set(X.representatives(X.fpoints(x))))
        for i, j in itertools.combinations(reps, 2):
            assert ((i, j) in ridges) == X.voronoi_ridge_exists(x, i, j), (i, j)


# ---------------------------------------------------------------- the vertex contract (§4)


def check_vertices(x, V, sv, S):
    """|v^_k - v_k| <= u |v_k| + 4 u |v_k - a_k| (a: the first vertex of the cluster's smallest simplex), with
    2^-1074 per term near the subnormals; an overflowing centre is inf."""
    pts = X.fpoints(x)
    verts = _np(V.vertices)
    first = {}
    for t, v in enumerate(sv.tolist()):
        first.setdefault(v, t)
    worst = Fraction(0)
    for v, t in first.items():
        s = [int(i) for i in S[t]]
        c, a = X.circumcenter(*(pts[i] for i in s)), pts[s[0]]
        for k in range(len(c)):
            got = float(verts[v, k])
            if not np.isfinite(got):  # the bound reaches past DBL_MAX
                reach = abs(c[k]) + U * abs(c[k]) + 4 * U * abs(c[k] - a[k])
                assert reach >= DBL_MAX and (got > 0) == (c[k] > 0), (v, k, got)
                continue
            err = abs(Fraction(got) - c[k])
            bound = U * abs(c[k]) + 4 * U * abs(c[k] - a[k]) + 2 * TINY
            assert err <= bound, (v, k, got, float(c[k]))
            scale = U * abs(c[k]) + U * abs(c[k] - a[k])
            if scale:
                worst = max(worst, err / scale)
    return worst


@pytest.mark.parametrize("pid", VALID)
def test_vertex_contract(built, pid):
    x = G.get(pid).x
    V, sv, S = get(built, pid)
    check_vertices(x, V, sv, S)


def _simplex_rows(kind, d, m, g):
    """(m, d + 1, d) positive simplices: 'uniform', 'thin' (needles and slivers), 'apex' (a centre coordinate
    within a few ULPs of the first vertex's: the numerators near 0), 'far' (a small simplex at 1e8), 'grid'
    (integers: exact differences), 'wide' (1e150 with 1e-150: the exact path)."""
    if kind == "uniform":
        P = g.random((m, d + 1, d))
    elif kind == "thin":
        P = g.random((m, d + 1, d))
        P[:, -1] = P[:, 0] + (P[:, 1] - P[:, 0]) * g.random((m, 1)) + 1e-9 * g.standard_normal((m, d))
    elif kind == "apex":
        P = g.random((m, d + 1, d))
        P[:, 1:, 0] = P[:, :1, 0] + np.array([[-1.0, 1.0, *([0.5] if d == 3 else [])]]) * g.random((m, 1))
        if d == 2:
            P[:, 2, 1] = P[:, 1, 1]  # an isoceles triangle with its apex at vertex 0: N_x = 0 exactly
        P[:, 1:, :] = np.nextafter(P[:, 1:, :], P[:, 1:, :] + g.choice([-1.0, 1.0], size=P[:, 1:, :].shape))
    elif kind == "far":
        P = 1e8 + 1e-3 * g.random((m, d + 1, d))
    elif kind == "grid":
        P = g.integers(-6, 7, size=(m, d + 1, d)).astype(float)
    else:
        P = g.random((m, d + 1, d)) * np.where(g.random((m, d + 1, d)) < 0.5, 1e150, 1e-150)
    keep = []
    for k in range(m):
        o = X.orient(*X.fpoints(P[k]))
        if o < 0:
            P[k, [-2, -1]] = P[k, [-1, -2]]
        if o != 0:
            keep.append(k)
    return np.ascontiguousarray(P[keep])


def _exact_nd(rows):
    """The exact numerators N (rows, d) and denominators D of v - a = N / (2 D)."""
    out = []
    for r in rows:
        p = X.fpoints(r)
        a, M = p[0], [_sub(q, p[0]) for q in p[1:]]
        L = [sum(v * v for v in m) for m in M]
        d = len(a)
        Dn = X.det([list(m) for m in M])
        N = [X.det([[L[i] if j == k else M[i][j] for j in range(d)] for i in range(d)]) for k in range(d)]
        out.append((N, Dn))
    return out


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("kind", ["uniform", "thin", "apex", "far", "grid", "wide"])
def test_vertex_kernel(d, kind):
    """The numerators' fast path is within 1.002 u when it answers (or leaves the row to the exact path), the exact
    path rounds N and D correctly, and the centres keep the §4 bound on both paths."""
    g = np.random.default_rng([d, len(kind)])
    rows = _simplex_rows(kind, d, 400, g)
    fast = [_np(a) for a in _core._circumcenter(sd.asarray(rows), False)]
    wide = [_np(a) for a in _core._circumcenter(sd.asarray(rows), True)]
    answered = 0
    for r, (N, Dn), v, Nf, Df, vw, Nw, Dw in zip(rows, _exact_nd(rows), fast[0], fast[1], fast[2], *wide):
        if not np.isnan(Df):
            answered += 1
            assert abs(Fraction(float(Df)) - Dn) <= Fraction(1002, 1000) * U * abs(Dn) + TINY / 2
            for k in range(d):
                assert abs(Fraction(float(Nf[k])) - N[k]) <= Fraction(1002, 1000) * U * abs(N[k]) + TINY / 2
        if kind != "wide":
            assert all(float(Nw[k]) == float(N[k]) for k in range(d)) and float(Dw) == float(Dn)
        a = X.frac(r[0])
        c = [a[k] + N[k] / (2 * Dn) for k in range(d)]
        for got in (v, vw):
            for k in range(d):
                if np.isfinite(got[k]):
                    err = abs(Fraction(float(got[k])) - c[k])
                    assert err <= U * abs(c[k]) + 4 * U * abs(c[k] - a[k]) + 2 * TINY, (kind, k)
                else:
                    assert abs(c[k]) + U * abs(c[k]) + 4 * U * abs(c[k] - a[k]) >= DBL_MAX
    if kind in ("uniform", "far", "thin"):
        assert answered == len(rows)
    if kind == "wide":
        assert answered < len(rows)


def test_overflowing_centre_is_inf():
    """A nearly flat triangle of huge points: the centre overflows to inf (SPT175), not an error; a centre whose
    distance from the first vertex passes DBL_MAX while it does not stays finite."""
    for x, finite in ((np.array([[-1e300, 0.0], [1e300, 0.0], [0.0, 1e-300]]), [True, False]),
                      (np.array([[-1.5e308, 0.0], [1.5e308, 0.0], [0.0, 1.5e308]]), [True, True])):  # fmt: skip
        V = Voronoi(x)
        assert np.isfinite(_np(V.vertices)[0]).tolist() == finite
        check_vertices(x, V, _np(_core._voronoi(sd.asarray(x), -1, True)[8]), _np(sd.spatial.Delaunay(x).simplices))


# ---------------------------------------------------------------- the orders of §2.5


def _exact_centres(x, V, sv, S):
    pts = X.fpoints(x)
    first = {}
    for t, v in enumerate(sv.tolist()):
        first.setdefault(v, t)
    return {v: X.circumcenter(*(pts[int(i)] for i in S[t])) for v, t in first.items()}


@pytest.mark.parametrize("pid", [i for i in VALID if len(G.get(i).x) <= 1000])
def test_orders(built, pid):
    x = G.get(pid).x
    V, sv, S = get(built, pid)
    d = x.shape[1]
    rp = _np(V.ridge_points)
    assert rp.dtype == np.int64 and (rp[:, 0] < rp[:, 1]).all()
    assert [tuple(r) for r in rp.tolist()] == sorted(set(tuple(r) for r in rp.tolist()))
    centres = _exact_centres(x, V, sv, S)
    pts = X.fpoints(x)
    for (i, j), row in zip(rp.tolist(), _all_rows(V.ridge_vertices)):
        assert len(row) == len(set(row)) and row.count(-1) <= 1
        if -1 in row:
            assert row[0] == -1
        elif d == 2:
            assert row == sorted(row) and len(row) == 2
        else:
            assert row[0] == min(row)
        if d == 2:
            assert len(row) == 2
            continue
        fin = [centres[v] for v in row if v >= 0]
        if len(fin) < 3:
            continue
        g = tuple(sum(c[k] for c in fin) / len(fin) for k in range(3))
        axis = _sub(pts[j], pts[i])
        pairs = zip(fin, fin[1:] + fin[:1]) if row[0] != -1 else zip(fin, fin[1:])
        for p, q in pairs:  # counterclockwise seen from i towards j: det[p - g, q - g, j - i] < 0
            assert X.det([list(_sub(p, g)), list(_sub(q, g)), list(axis)]) < 0, (i, j, row)
    for i, row in enumerate(_all_rows(V.regions)):
        if not row:
            continue
        assert row.count(-1) <= 1 and (row[0] == -1 if -1 in row else True)
        fin = [v for v in row if v >= 0]
        if d == 3:
            assert row == sorted(row)
            continue
        if -1 not in row:
            assert row[0] == min(row)
        cyc = list(zip(fin, fin[1:] + fin[:1])) if -1 not in row else list(zip(fin, fin[1:]))
        for p, q in cyc:  # counterclockwise around point i
            assert X.orient(pts[i], centres[p], centres[q]) > 0, (i, row)


def test_three_by_three_grid():
    """§2.5's example: 4 vertices and the grid's 12 edges as ridges (scipy's too)."""
    x = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
    V = Voronoi(x)
    assert _np(V.vertices).tolist() == [[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]]
    want = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6), (4, 5), (4, 7), (5, 8), (6, 7), (7, 8)]
    assert [tuple(r) for r in _np(V.ridge_points).tolist()] == want
    assert _rows(V.regions, 4) == [0, 2, 3, 1]
    assert _rows(V.regions, 0) == [-1, 0]
    assert _rows(V.ridge_vertices, 3) == [0, 1]  # (1, 4): between the vertices (0.5, 0.5) and (0.5, 1.5)
    assert _rows(V.ridge_vertices, 0) == [-1, 0]


def test_cube_grid():
    """A 3 x 3 x 3 grid: 8 vertices, a ridge for every pair of axis neighbours (54; scipy leaves out the 24
    unbounded ones with one finite vertex), none for the diagonals of the flat hull faces."""
    x = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=float)
    V = Voronoi(x)
    assert len(_np(V.vertices)) == 8
    rp = _np(V.ridge_points)
    assert len(rp) == 54 and all(np.abs(x[i] - x[j]).sum() == 1 for i, j in rp.tolist())
    assert _rows(V.regions, 13) == list(range(8))
    assert sum(len(r) == 2 for r in _all_rows(V.ridge_vertices)) == 24


def test_equal_points_and_negative_zero():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [-0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, -0.0], [0.4, 0.6]])
    V = Voronoi(x)
    assert _np(V.point_region).tolist() == [0, 1, 0, 3, 1, 5, 0, 7]
    rows = _all_rows(V.regions)
    assert rows[2] == rows[4] == rows[6] == [] and all(rows[i] for i in (0, 1, 3, 5, 7))
    assert not {2, 4, 6} & set(_np(V.ridge_points).ravel().tolist())


@pytest.mark.parametrize("pid", ["uniform2/1000", "grid2/30", "dup2", "uniform3/1000", "grid3/6", "dup3", "cube/5"])
def test_shuffled_rows(pid):
    """The diagram is a function of the point set: shuffled rows give the same ridges and regions (as point
    coordinates) and vertices within twice the §4 bound."""
    x = G.get(pid).x
    perm = np.random.default_rng(5).permutation(len(x))
    A, B = Voronoi(x), Voronoi(x[perm])

    def geometry(V, y):
        verts = _np(V.vertices)
        key = [tuple(np.round(v, 6).tolist()) for v in verts]  # vertex identity for these well-separated sets
        ridges = {frozenset((tuple(y[i]), tuple(y[j]))): frozenset(key[v] if v >= 0 else None for v in r)
                  for (i, j), r in zip(_np(V.ridge_points).tolist(), _all_rows(V.ridge_vertices))}  # fmt: skip
        regions = {tuple(y[i]): frozenset(key[v] if v >= 0 else None for v in r)
                   for i, r in enumerate(_all_rows(V.regions)) if r}  # fmt: skip
        return len(verts), ridges, regions

    assert geometry(A, x) == geometry(B, x[perm])


# ---------------------------------------------------------------- attributes, errors, pickle, memory


def test_attributes():
    x = G.get("uniform3/1000").x
    V = Voronoi(x)
    assert V.ndim == 3 and V.npoints == 1000 and repr(V) == f"Voronoi(n=1000, d=3, nvertices={len(_np(V.vertices))})"
    assert _np(V.points).tobytes() == x.tobytes()
    assert _np(V.min_bound).tolist() == x.min(0).tolist() and _np(V.max_bound).tolist() == x.max(0).tolist()
    for a in (V.points, V.vertices, V.ridge_points, *V.ridge_vertices, *V.regions, V.point_region, V.min_bound):
        with pytest.raises(ValueError):
            a[0] = 0
    assert _np(V.vertices).dtype == np.float64 and _np(V.point_region).dtype == np.int64
    assert all(_np(a).dtype == np.int64 for a in (*V.ridge_vertices, *V.regions))
    indptr = _np(V.regions[0])
    assert indptr[0] == 0 and len(indptr) == 1001 and (np.diff(indptr) > 0).all()
    assert len(_np(V.ridge_vertices[0])) == len(_np(V.ridge_points)) + 1


def test_points_are_copied_and_inputs():
    x = G.get("uniform2/1000").x.copy()
    s = sd.asarray(x.copy())
    V = Voronoi(s)
    s[0, 0] = 9.0
    assert _np(V.points).tobytes() == x.tobytes()
    pts = [[0, 0], [2, 0], [0, 2], [1, 1], [2, 2]]
    want = Voronoi(np.array(pts, dtype=float))
    for inp in (pts, np.array(pts), sd.asarray(np.array(pts, dtype=float)), np.asfortranarray(np.array(pts, float))):
        got = Voronoi(inp)
        assert _np(got.vertices).tobytes() == _np(want.vertices).tobytes()
        assert _np(got.ridge_points).tobytes() == _np(want.ridge_points).tobytes()


def test_pickle_copy_repr():
    V = Voronoi(G.get("dup3").x)
    for W in (pickle.loads(pickle.dumps(V)), copy.copy(V), copy.deepcopy(V)):
        assert type(W) is Voronoi and W is not V
        for name in ("points", "vertices", "ridge_points", "point_region"):
            assert _np(getattr(W, name)).tobytes() == _np(getattr(V, name)).tobytes(), name
        for name in ("ridge_vertices", "regions"):
            assert all(_np(a).tobytes() == _np(b).tobytes() for a, b in zip(getattr(W, name), getattr(V, name)))


def test_memory_limit():
    x = np.random.default_rng(2).random((3000, 3))
    with pytest.raises(MemoryError, match=r"^Voronoi\(\): the triangulation is larger than this machine's physical"):
        _voronoi._limited(x, 1000)
    W = _voronoi._limited(x, 200_000)
    assert _np(W.vertices).tobytes() == _np(Voronoi(x).vertices).tobytes()


@pytest.mark.parametrize(
    ("x", "exc", "msg"),
    [
        (np.zeros(3), ValueError, "Voronoi(): 'points' must have 2 dimensions, not 1"),
        (np.zeros((5, 4)), ValueError, "Voronoi(): 'points' must have 2 or 3 columns, not 4"),
        (np.array([[0.0, 0.0], [1.0, np.nan], [0.0, 1.0]]), ValueError, "Voronoi(): 'points' must be finite"),
        (np.array([[0.0, 0.0], [1.0, np.inf], [0.0, 1.0]]), ValueError, "Voronoi(): 'points' must be finite"),
        (G.get("line2").x, ValueError, "Voronoi(): 'points' are collinear"),
        (G.get("line3").x, ValueError, "Voronoi(): 'points' are coplanar"),
        (G.get("plane3").x, ValueError, "Voronoi(): 'points' are coplanar"),
        (G.get("few2").x, ValueError, "Voronoi(): needs at least 3 distinct points, got 1"),
        (G.get("few3").x, ValueError, "Voronoi(): needs at least 4 distinct points, got 2"),
        (np.zeros((0, 2)), ValueError, "Voronoi(): needs at least 3 distinct points, got 0"),
        (np.array([[0, 0], [1, 0], [0, 1j]]), TypeError, "Voronoi(): 'points' must be real, not a complex128 array"),
    ],
)
def test_errors(x, exc, msg):
    with pytest.raises(exc) as e:
        Voronoi(x)
    assert str(e.value) == msg


def test_call_shape_errors():
    x = G.get("uniform2/12").x
    for kw in ("qhull_options", "furthest_site", "incremental"):
        with pytest.raises(TypeError, match=rf"^Voronoi\(\): unexpected keyword argument '{kw}'$"):
            Voronoi(x, **{kw: None})


# ---------------------------------------------------------------- scipy on points in general position


@LIVE
@pytest.mark.parametrize("pid", ["uniform2/1000", "normal2/1000", "cluster2/1000", "ball2/1000", "uniform2/20000",
                                 "uniform3/1000", "normal3/1000", "cluster3/1000", "ball3/1000"])  # fmt: skip
def test_matches_scipy(pid):
    """The same vertices (within the §4 bound and scipy's error), ridges and regions; in 3-D scipy leaves out the
    unbounded ridges with one finite vertex (Delaunay edges in one tetrahedron), which sd has."""
    import scipy.spatial as S

    x = G.get(pid).x
    V, q = Voronoi(x), S.Voronoi(x)
    mine, theirs = _np(V.vertices), q.vertices
    assert len(mine) == len(theirs)
    order_m, order_t = np.lexsort(mine.T[::-1]), np.lexsort(theirs.T[::-1])
    scale = np.abs(mine).max()
    assert np.allclose(mine[order_m], theirs[order_t], rtol=1e-9, atol=1e-9 * scale)
    vmap = np.empty(len(theirs), dtype=np.int64)  # scipy's vertex -> sd's
    vmap[order_t] = order_m
    got = {tuple(e): set(r) for e, r in zip(_np(V.ridge_points).tolist(), _all_rows(V.ridge_vertices))}
    want = {tuple(sorted(e)): {int(vmap[v]) if v >= 0 else -1 for v in r}
            for e, r in zip(q.ridge_points.tolist(), q.ridge_vertices)}  # fmt: skip
    assert want.items() <= got.items()
    extra = [r for e, r in got.items() if e not in want]
    assert all(r == {-1, next(iter(r - {-1}))} and len(r) == 2 for r in extra) and (x.shape[1] == 3 or not extra)
    for i in range(0, len(x), 7):
        r = q.regions[q.point_region[i]]
        assert set(_rows(V.regions, i)) == {int(vmap[v]) if v >= 0 else -1 for v in r}
