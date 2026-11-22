"""KDTree.query_ball_point, query_ball_tree and sparse_distance_matrix (docs/api/spatial.md §4, docs/design/spatial.md
§3.4, §4 neighbours 1-2; milestone SPT4).

1. The result is the brute force with the same kernel: the rows of cdist(x, data) (minkowski for p not 1,
   2, inf) with distance <= r, in index order, the distances cdist's bits; so it is radius_neighbors'
   RadiusResult byte for byte. On random, duplicated, tied, lattice, clustered, far, huge, tiny,
   subnormal and spanning data, at every leafsize and p, with radii at the computed distances (the
   boundary is included, the next double below is not), and with hypothesis.
2. On the lattice sets of tests/spatial/ref/neighbors.npz (the computed distance is the true one rounded)
   the rows are the Arb references within r.
Plus return_length, the shapes and the broadcast r, query_ball_tree as radius_neighbors(self.data,
other.data), sparse_distance_matrix as its tocsr() with stored zeros, scipy 1.18.1 live and its records,
the errors and the same bytes for 1, 3 and 10 threads."""

import math
import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import stardust as sd

from .._bits import bits_equal
from . import points as P
from . import refdata
from .conftest import LIVE
from .test_kdtree_query import PS, cdist_p

S = sd.spatial
INF = math.inf


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


def brute(x, data, r, p=2.0):
    """(indptr, indices, distances) of the rows of cdist(x, data) within r (one radius per row of x)."""
    data = np.asarray(data, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(math.prod(x.shape[:-1]), data.shape[1])
    r = np.broadcast_to(np.asarray(r, dtype=np.float64), (len(x),))
    D = cdist_p(x, data, p) if len(x) and len(data) else np.zeros((len(x), len(data)))
    indptr, indices, dist = [0], [], []
    for q in range(len(x)):
        j = np.nonzero(D[q] <= r[q])[0]
        indices.extend(j.tolist())
        dist.extend(D[q, j].tolist())
        indptr.append(len(indices))
    return np.array(indptr, np.int64), np.array(indices, np.int64), np.array(dist, np.float64)


def same(res, want):
    got = (np.asarray(res.indptr), np.asarray(res.indices), np.asarray(res.distances))
    return all(g.dtype == w.dtype and g.shape == w.shape and bits_equal(g, w) for g, w in zip(got, want))


def check(data, x, r, p=2.0, leafsizes=(1, 3, 10, 64)):
    want = brute(x, data, r, p)
    for leafsize in leafsizes:
        t = S.KDTree(data, leafsize=leafsize)
        res = t.query_ball_point(x, r, p=p)
        assert same(res, want), (leafsize, p)
        assert res.shape == (len(want[0]) - 1, len(data))
        n = t.query_ball_point(x, r, p=p, return_length=True)
        n = np.asarray([n]) if isinstance(n, int) else np.asarray(n).ravel()
        assert np.array_equal(n, np.diff(want[0])), (leafsize, p)


def radii(D):
    """Radii at the computed distances of a row (the boundary cases) and their neighbours."""
    v = np.sort(D[np.isfinite(D)])
    out = [0.0, INF]
    for d in v[[0, len(v) // 7, len(v) // 3, len(v) // 2]] if len(v) else []:
        out += [float(d), math.nextafter(float(d), -INF), math.nextafter(float(d), INF)]
    return [max(o, 0.0) for o in out]


@pytest.mark.parametrize("p", PS)
def test_random_points_are_the_brute_force(p):
    g = np.random.default_rng(30)
    for n, m in ((500, 3), (300, 1), (200, 8), (150, 16), (80, 64)):
        data, x = g.standard_normal((n, m)), g.standard_normal((20, m))
        D = cdist_p(x[:1], data, p)[0]
        for r in radii(D):
            check(data, x, r, p=p, leafsizes=(1, 10))
        check(data, data[:15], radii(cdist_p(data[:1], data, p)[0])[3], p=p)  # the data points themselves


@pytest.mark.parametrize("p", PS)
def test_duplicates_ties_and_lattices(p):
    g = np.random.default_rng(31)
    dup = np.repeat(g.integers(-3, 4, (15, 3)).astype(float), 9, axis=0)[g.permutation(135)]
    for r in (0.0, 1.0, 2.0, 3.0, INF):
        check(dup, g.integers(-4, 5, (20, 3)).astype(float), r, p=p)
    grid = np.array(np.meshgrid(*[np.arange(-3.0, 4.0)] * 2)).reshape(2, -1).T
    for r in (0.0, 1.0, math.sqrt(2.0), 2.0, 2.5):
        check(grid, np.array([[0.0, 0.0], [0.5, 0.5], [3.0, -3.0], [10.0, 0.0]]), r, p=p)
    check(np.ones((6, 2)), np.ones((1, 2)), 0.0, p=p)


@pytest.mark.parametrize("p", (1.0, 2.0, INF, 2.5))
def test_clustered_far_and_extreme_data(p):
    g = np.random.default_rng(32)
    centers = g.standard_normal((6, 4)) * 50
    clustered = (centers[:, None, :] + g.standard_normal((6, 70, 4)) * 1e-3).reshape(-1, 4)
    check(clustered, centers + 0.01, 0.02, p=p)
    check(clustered, g.standard_normal((10, 4)) * 50, 40.0, p=p)
    sets = [
        (1e8 + g.random((200, 3)), 1e8 + g.random((15, 3))),
        (g.standard_normal((150, 3)) * 1e200, g.standard_normal((10, 3)) * 1e200),
        (g.standard_normal((150, 3)) * 1e-200, g.standard_normal((10, 3)) * 1e-200),
        (g.integers(-50, 50, (100, 2)) * 5e-324, g.integers(-50, 50, (10, 2)) * 5e-324),
    ]
    for data, x in sets:
        for r in radii(cdist_p(x[:1], data, p)[0]):
            check(data, x, r, p=p, leafsizes=(1, 10))
    span = np.array([[-1.7e308, 0.0], [1.7e308, 0.0], [0.0, 1.0], [1e308, -1e308]])
    for r in (1.0, 1e308, 1.7e308, INF):  # distances overflow to inf: in only for r = inf
        check(span, np.array([[0.0, 0.0], [1.6e308, 1.0]]), r, p=p, leafsizes=(1, 10))


def test_the_reference_shell_sets_and_far_clusters():
    for sid in ("shell/3", "far/1e8", "far/1e15", "ties/2", "dup/3"):
        s = P.get(sid)
        for p in (1.0, 2.0, INF):
            D = cdist_p(s.queries, s.x, p)
            for r in radii(D[0]):
                check(s.x, s.queries, r, p=p, leafsizes=(1, 10))


def test_radius_per_point_and_broadcast():
    g = np.random.default_rng(33)
    data, x = g.random((300, 3)), g.random((24, 3))
    r = g.random(24) * 0.3
    check(data, x, r)
    t = S.KDTree(data)
    xb = x.reshape(2, 3, 4, 3)
    rb = r.reshape(2, 3, 4)
    assert same(t.query_ball_point(xb, rb), brute(x, data, r))
    col = g.random((3, 1)) * 0.3  # broadcast over the last batch axis
    assert same(t.query_ball_point(xb, col), brute(x, data, np.broadcast_to(col, (2, 3, 4)).ravel()))
    assert same(t.query_ball_point(xb, [0.1]), brute(x, data, 0.1))
    assert same(t.query_ball_point(x, sd.asarray(r)), brute(x, data, r))
    counts = t.query_ball_point(xb, rb, return_length=True)
    assert counts.shape == (2, 3, 4) and counts.dtype == sd.int64
    assert np.array_equal(np.asarray(counts).ravel(), np.diff(brute(x, data, r)[0]))


def test_query_ball_point_equals_radius_neighbors():
    g = np.random.default_rng(34)
    data, x = g.standard_normal((2000, 5)), g.standard_normal((300, 5))
    t = S.KDTree(data)
    for p, kw in ((2.0, {}), (1.0, {"metric": "cityblock"}), (INF, {"metric": "chebyshev"}), (3.0,
                  {"metric": "minkowski", "p": 3.0})):  # fmt: skip
        for r in (0.5, 1.5, 3.0):
            a, b = t.query_ball_point(x, r, p=p), S.radius_neighbors(x, data, r, **kw)
            assert bits_equal(a.indptr, b.indptr) and bits_equal(a.indices, b.indices), (p, r)
            assert bits_equal(a.distances, b.distances) and a.shape == b.shape, (p, r)


def test_shapes_types_and_empty_trees():
    data = np.random.default_rng(35).random((20, 2))
    t = S.KDTree(data)
    one = t.query_ball_point([0.5, 0.5], 0.3)
    assert type(one).__name__ == "RadiusResult" and one.shape == (1, 20)
    assert type(t.query_ball_point([0.5, 0.5], 0.3, return_length=True)) is int  # scipy: numpy.int64
    assert t.query_ball_point(np.zeros((0, 2)), 0.3).shape == (0, 20)
    assert t.query_ball_point(np.zeros((0, 2)), 0.3, return_length=True).shape == (0,)
    for a in (one.indptr, one.indices, one.distances):
        assert not a._flags()["writeable"]
    e = S.KDTree(np.zeros((0, 3)))
    res = e.query_ball_point(np.zeros((2, 3)), 1.0)
    assert res.shape == (2, 0) and res.indptr.tolist() == [0, 0, 0]
    z = S.KDTree(np.zeros((4, 0)))  # every distance 0
    assert z.query_ball_point(np.zeros((2, 0)), 0.0).indices.tolist() == [0, 1, 2, 3] * 2
    r1, r2 = t.query_ball_point(np.zeros((3, 2), dtype=np.int64), 0.5), t.query_ball_point(np.zeros((3, 2)), 0.5)
    assert same(r1, (np.asarray(r2.indptr), np.asarray(r2.indices), np.asarray(r2.distances)))


@pytest.mark.parametrize("p", (1.0, 2.0, INF))
def test_lattice_references_exactly(p):
    """§4 neighbours 2: on the lattice sets the computed distance is the true one rounded, so the rows are
    the Arb references within r (and at the boundary the boundary is in)."""
    r = refdata.ref("neighbors")
    method = {1.0: "p1", 2.0: "p2", INF: "pinf"}[p]
    for sid in sorted(set(P.ids("lattice")) & set(P.ids("neighbors"))):
        s = P.get(sid)
        t = S.KDTree(s.x, leafsize=4)
        rows = [k for k in range(r.points) if r.inputs["set"][k] == sid]
        for k in rows:
            hi = r.methods[method].hi[k]
            for rad in np.unique(np.sort(hi)[[0, 3, 10, len(hi) // 2]]):
                res = t.query_ball_point(r.inputs["query"][k], float(rad), p=p)
                want = np.nonzero(hi <= rad)[0]
                assert np.array_equal(np.asarray(res.indices), want), (sid, k, rad)
                assert bits_equal(res.distances, hi[want]), (sid, k, rad)


# ---- query_ball_tree and sparse_distance_matrix ----


def test_query_ball_tree_is_radius_neighbors_of_the_data():
    g = np.random.default_rng(36)
    a, b = g.standard_normal((700, 3)), g.standard_normal((900, 3))
    a[::5] = np.round(a[::5], 1)
    b[::3] = np.round(b[::3], 1)
    b[:40] = a[:40]  # duplicates across the trees: distance 0
    for la, lb in ((10, 10), (1, 64), (64, 1)):
        A, B = S.KDTree(a, leafsize=la), S.KDTree(b, leafsize=lb)
        for p in PS:
            for r in (0.0, 0.1, 0.45, 2.0, INF):
                got = A.query_ball_tree(B, r, p=p)
                assert same(got, brute(a, b, r, p)), (la, lb, p, r)
                assert got.shape == (700, 900)
                if (la, lb) == (10, 10) and r == 0.45:
                    other = B.query_ball_point(a, r, p=p)
                    assert bits_equal(got.indices, other.indices) and bits_equal(got.distances, other.distances)
    A = S.KDTree(a)
    assert same(A.query_ball_tree(A, 0.3), brute(a, a, 0.3))  # a tree with itself: every point is its own
    e = S.KDTree(np.zeros((0, 3)))
    assert A.query_ball_tree(e, 1.0).shape == (700, 0) and A.query_ball_tree(e, 1.0).indices.shape == (0,)
    assert e.query_ball_tree(A, 1.0).indptr.tolist() == [0]


@pytest.mark.parametrize("p", (1.0, 2.0, INF, 1.5))
def test_query_ball_tree_extreme_data(p):
    g = np.random.default_rng(37)
    for a, b in ((1e8 + g.random((300, 3)), 1e8 + g.random((200, 3))),
                 (g.standard_normal((200, 3)) * 1e200, g.standard_normal((150, 3)) * 1e200),
                 (g.standard_normal((200, 3)) * 1e-200, g.standard_normal((150, 3)) * 1e-200),
                 (g.integers(-20, 20, (100, 2)) * 5e-324, g.integers(-20, 20, (90, 2)) * 5e-324)):  # fmt: skip
        D = cdist_p(a[:1], b, p)[0]
        for r in radii(D):
            for ls in (1, 10):
                assert same(S.KDTree(a, leafsize=ls).query_ball_tree(S.KDTree(b, leafsize=ls), r, p=p),
                            brute(a, b, r, p)), (r, ls)  # fmt: skip


def test_sparse_distance_matrix_is_the_tocsr_with_stored_zeros():
    g = np.random.default_rng(38)
    a = np.round(g.standard_normal((300, 2)), 1)
    b = np.vstack([a[:30], np.round(g.standard_normal((200, 2)), 1)])
    A, B = S.KDTree(a), S.KDTree(b)
    for p in (1.0, 2.0, INF, 3.0):
        m = A.sparse_distance_matrix(B, 0.4, p=p)
        want = A.query_ball_tree(B, 0.4, p=p).tocsr()
        assert type(m) is sd.sparse.csr_array and m.shape == (300, 230) and m.dtype == sd.float64
        assert bits_equal(m.indptr, want.indptr) and bits_equal(m.indices, want.indices)
        assert bits_equal(m.data, want.data)
    m = A.sparse_distance_matrix(B, 0.4)
    zeros = int(np.sum(np.asarray(m.data) == 0.0))
    assert zeros >= 30 and m.nnz == m.count_nonzero() + zeros  # the stored zeros (SPT3, scipy's record)
    two = S.KDTree([[0.0, 0.0], [0.0, 0.0]])
    rec = refdata.data("scipy")["kdtree/sdm_zero_dok"]["value"]
    m2 = two.sparse_distance_matrix(two, 1.0)
    assert [m2.nnz, m2.count_nonzero()] == rec == [4, 0]
    m2.data[0] = 5.0  # a new, writable array
    assert two.sparse_distance_matrix(two, 1.0).data.tolist() == [0.0] * 4


@given(
    n=st.integers(0, 60),
    nb=st.integers(0, 50),
    m=st.integers(0, 4),
    p=st.sampled_from(PS),
    leafsize=st.integers(1, 12),
    spread=st.sampled_from([1.0, 4.0, 1e-300, 1e150]),
    integers=st.booleans(),
    radius=st.sampled_from([0.0, 0.5, 1.0, 2.0, 3.5, INF]),
    seed=st.integers(0, 2**32 - 1),
)
def test_hypothesis_brute_force(n, nb, m, p, leafsize, spread, integers, radius, seed):
    g = np.random.default_rng(seed)
    data = g.integers(-3, 4, (n, m)).astype(float) if integers else g.standard_normal((n, m))
    x = g.integers(-4, 5, (nb, m)).astype(float) if integers else g.standard_normal((nb, m))
    if n and m and nb >= 3 and g.random() < 0.5:
        x[:3] = data[g.integers(0, n, 3)]
    data, x = data * spread, x * spread
    r = radius * spread
    if n and nb and m and g.random() < 0.5:  # a radius at a computed distance: the boundary
        D = cdist_p(x, data, p)
        r = float(D.ravel()[g.integers(0, D.size)])
    check(data, x, r, p=p, leafsizes=(leafsize,))
    got = S.KDTree(x, leafsize=leafsize).query_ball_tree(S.KDTree(data, leafsize=leafsize), r, p=p)
    assert same(got, brute(x, data, r, p))


# ---- scipy ----


@LIVE
def test_scipy_agrees_on_sets_without_boundary_ties():
    import scipy.spatial as SS

    g = np.random.default_rng(39)
    for n, m in ((3000, 3), (500, 8)):
        data, x = g.random((n, m)), g.random((60, m))
        T, U = S.KDTree(data), SS.KDTree(data)
        for p in (1.0, 2.0, INF, 3.0):
            r = 0.15 if m == 3 else 0.6
            got = T.query_ball_point(x, r, p=p)
            want = U.query_ball_point(x, r, p=p)
            for q in range(len(x)):
                assert got.indices[got.indptr[q] : got.indptr[q + 1]].tolist() == sorted(want[q]), (p, q)
            assert np.asarray(T.query_ball_point(x, r, p=p, return_length=True)).tolist() == \
                U.query_ball_point(x, r, p=p, return_length=True).tolist()  # fmt: skip
            other = g.random((400, m))
            bt = T.query_ball_tree(S.KDTree(other), r, p=p)
            st_ = U.query_ball_tree(SS.KDTree(other), r, p=p)
            for i in range(n):
                assert bt.indices[bt.indptr[i] : bt.indptr[i + 1]].tolist() == sorted(st_[i]), (p, i)
            sm = T.sparse_distance_matrix(S.KDTree(other), r, p=p)
            ss = U.sparse_distance_matrix(SS.KDTree(other), r, p=p, output_type="coo_array").tocsr()
            ss.sort_indices()
            assert np.array_equal(np.asarray(sm.indptr), ss.indptr) and np.array_equal(np.asarray(sm.indices),
                                                                                      ss.indices)  # fmt: skip
            np.testing.assert_allclose(np.asarray(sm.data), ss.data, rtol=8e-16 * m, atol=0)


@LIVE
def test_scipy_records_of_the_differences():
    """The [scipy와 다름] of the ball queries (docs/api/spatial.md §4): scipy takes r = -1 as r = 1 and gives
    an empty list for NaN; one point gives a list, many an object array, one point's list unsorted; here
    a negative or NaN r is an error and every result is a RadiusResult in index order."""
    rec = refdata.data("scipy")
    assert rec["kdtree/r_negative_same_as_1"]["value"] is True
    assert rec["kdtree/r_nan"]["value"] == []
    assert rec["kdtree/ball_point_one"]["value"] == "list"
    assert rec["kdtree/ball_point_many"]["value"] == "ndarray"
    assert rec["kdtree/ball_point_one_sorted"]["value"] is False
    t = S.KDTree(P.get("nn/normal3").x[:50])
    with pytest.raises(ValueError):
        t.query_ball_point(np.zeros(3), -1.0)
    with pytest.raises(ValueError):
        t.query_ball_point(np.zeros(3), math.nan)
    res = t.query_ball_point(np.zeros(3), 1.5)
    assert type(res).__name__ == "RadiusResult"
    assert res.indices.tolist() == sorted(res.indices.tolist())


# ---- threads ----


@pytest.mark.usefixtures("restore_threads")
def test_one_three_ten_threads():
    g = np.random.default_rng(40)
    a = g.standard_normal((30_000, 3))
    a[::5] = np.round(a[::5], 1)
    b = g.standard_normal((20_000, 3))
    b[:2000] = a[:2000]
    x = g.standard_normal((3000, 3))
    got = {}
    for th in (1, 3, 10):
        sd.set_num_threads(th)
        A, B = S.KDTree(a), S.KDTree(b)
        out = []
        for p in (2.0, 1.0, INF, 3.0):
            r = A.query_ball_point(x, 0.08, p=p)
            out += [r.indptr, r.indices, r.distances, A.query_ball_point(x, 0.08, p=p, return_length=True)]
            r = A.query_ball_tree(B, 0.03, p=p)
            out += [r.indptr, r.indices, r.distances]
        m = A.sparse_distance_matrix(B, 0.03)
        out += [m.indptr, m.indices, m.data]
        got[th] = out
    for th in (3, 10):
        for k, (u, v) in enumerate(zip(got[1], got[th])):
            assert bits_equal(u, v), (th, k)


# ---- errors ----


def test_errors():
    t = S.KDTree(np.random.default_rng(41).random((10, 3)))
    u = S.KDTree(np.random.default_rng(42).random((8, 2)))
    x = np.zeros(3)
    with raises(TypeError, "query_ball_point(): takes 2 positional arguments but 3 were given"):
        t.query_ball_point(x, 1.0, 2.0)  # p is keyword only
    with raises(TypeError, "query_ball_point(): unexpected keyword argument 'workers'"):
        t.query_ball_point(x, 1.0, workers=2)
    with raises(TypeError, "query_ball_point(): unexpected keyword argument 'return_sorted'"):
        t.query_ball_point(x, 1.0, return_sorted=True)
    with raises(TypeError, "query_ball_point(): unexpected keyword argument 'eps'"):
        t.query_ball_point(x, 1.0, eps=0.1)
    with raises(ValueError, "query_ball_point(): 'r' must be a nonnegative number or inf, not -1.0"):
        t.query_ball_point(x, -1.0)  # scipy: as r = 1
    with raises(ValueError, "query_ball_point(): 'r' must be a nonnegative number or inf, not nan"):
        t.query_ball_point(x, math.nan)  # scipy: an empty list
    with raises(ValueError, "query_ball_point(): every entry of 'r' must be a nonnegative number or inf"):
        t.query_ball_point(np.zeros((2, 3)), [0.5, -0.5])
    with raises(ValueError, "query_ball_point(): 'r' of shape (2,) does not broadcast to the batch shape (3,) of 'x'"):
        t.query_ball_point(np.zeros((3, 3)), [0.5, 0.5])
    with raises(ValueError, "query_ball_point(): 'r' of shape (1,) does not broadcast to the batch shape () of 'x'"):
        t.query_ball_point(x, [0.5])  # scipy also refuses it
    with raises(
        ValueError, "query_ball_point(): 'r' of shape (2, 1) does not broadcast to the batch shape (2,) of 'x'"
    ):
        t.query_ball_point(np.zeros((2, 3)), [[0.5], [0.5]])
    with raises(TypeError, "query_ball_point(): 'r' must be a number or an array of numbers, not NoneType"):
        t.query_ball_point(x, None)
    with raises(TypeError, "query_ball_point(): 'r' must be a number or an array of numbers, not str"):
        t.query_ball_point(x, "1")
    with raises(TypeError, "query_ball_point(): 'return_length' must be True or False, not int"):
        t.query_ball_point(x, 1.0, return_length=1)
    with raises(ValueError, "query_ball_point(): 'p' must be a number of at least 1 or inf, not 0.5"):
        t.query_ball_point(x, 1.0, p=0.5)
    with raises(ValueError, "query_ball_point(): 'x' must have 3 entries on its last axis (the tree's m), not 2"):
        t.query_ball_point([0.0, 0.0], 1.0)
    with raises(ValueError, "query_ball_point(): 'x' must be finite"):
        t.query_ball_point([0.0, math.nan, 0.0], 1.0)
    with raises(TypeError, "query_ball_point(): 'x' must be real, not a complex128 array"):
        t.query_ball_point(np.zeros(3, complex), 1.0)
    for name, call in (("query_ball_tree", lambda o, r, **k: t.query_ball_tree(o, r, **k)),
                       ("sparse_distance_matrix", lambda o, r, **k: t.sparse_distance_matrix(o, r, **k))):  # fmt: skip
        arg = "r" if name == "query_ball_tree" else "max_distance"
        with raises(TypeError, f"{name}(): 'other' must be a KDTree, not numpy.ndarray"):
            call(np.zeros((2, 3)), 1.0)
        with raises(ValueError, f"{name}(): 'other' must have the same m as this tree (3), not 2"):
            call(u, 1.0)  # scipy: "Trees passed to query_ball_tree have different dimensionality"
        with raises(ValueError, f"{name}(): '{arg}' must be a nonnegative number or inf, not -0.5"):
            call(t, -0.5)
        with raises(ValueError, f"{name}(): '{arg}' must be a nonnegative number or inf, not nan"):
            call(t, math.nan)
        with raises(TypeError, f"{name}(): '{arg}' must be an int or a float, not list"):
            call(t, [1.0])
        with raises(ValueError, f"{name}(): 'p' must be a number of at least 1 or inf, not 0"):
            call(t, 1.0, p=0)
    with raises(TypeError, "query_ball_tree(): takes 2 positional arguments but 3 were given"):
        t.query_ball_tree(t, 1.0, 2.0)
    with raises(TypeError, "sparse_distance_matrix(): unexpected keyword argument 'output_type'"):
        t.sparse_distance_matrix(t, 1.0, output_type="dok_matrix")
    assert t.query_ball_point(x, np.float32(0.5), p=np.int64(1)).shape == (1, 10)
    assert t.query_ball_point(x, np.array(0.5)).shape == (1, 10)
    assert t.query_ball_tree(t, np.float64(0.5)).shape == (10, 10)
