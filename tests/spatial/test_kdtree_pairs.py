"""KDTree.query_pairs and count_neighbors (docs/api/spatial.md §4, docs/design/spatial.md §3.4, §4 neighbours 1;
milestone SPT4; decision SPT25).

1. query_pairs is the brute force with the same kernel: the pairs (i, j), i < j, whose pdist entry (minkowski
   for p not 1, 2, inf) is <= r, sorted, as a (k, 2) int64 array.
2. count_neighbors counts the entries of cdist(self.data, other.data) <= r exactly, for one radius and for
   many (cumulative, binned, unsorted, repeated, 0 and inf), self with itself included; weighted it sums
   wa_i wb_j over the same pairs (exactly when the sums are exact, else within rounding), in scipy's weight
   forms, and the sums are the same bytes for 1, 3 and 10 threads (SPT25).
On random, duplicated, lattice, clustered, far, huge, tiny and subnormal data, every leafsize and p, with
radii at the computed distances and with hypothesis; plus scipy 1.18.1 live and its records and the errors."""

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
from .test_kdtree_radius import radii

S = sd.spatial
INF = math.inf


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


def pdist_p(x, p):
    if p == 1:
        return np.asarray(S.pdist(x, metric="cityblock"))
    if p == 2:
        return np.asarray(S.pdist(x))
    if p == INF:
        return np.asarray(S.pdist(x, metric="chebyshev"))
    return np.asarray(S.pdist(x, metric="minkowski", p=p))


def brute_pairs(x, r, p=2.0):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return np.zeros((0, 2), np.int64)
    i, j = np.triu_indices(n, 1)  # pdist's order: (i, j) sorted
    keep = pdist_p(x, p) <= r
    return np.stack([i[keep], j[keep]], axis=1).astype(np.int64)


def check_pairs(x, r, p=2.0, leafsizes=(1, 3, 10, 64)):
    want = brute_pairs(x, r, p)
    for leafsize in leafsizes:
        got = S.KDTree(x, leafsize=leafsize).query_pairs(r, p=p)
        assert got.dtype == sd.int64 and got.shape == want.shape, (leafsize, p, r)
        assert np.array_equal(np.asarray(got), want), (leafsize, p, r)


def brute_count(a, b, r, p=2.0, wa=None, wb=None, cumulative=True):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    D = cdist_p(a, b, p) if len(a) and len(b) else np.zeros((len(a), len(b)))
    W = (np.ones(len(a)) if wa is None else wa)[:, None] * (np.ones(len(b)) if wb is None else wb)[None, :]
    rs = np.atleast_1d(np.asarray(r, dtype=np.float64))
    if cumulative:
        out = [W[D <= v].sum() for v in rs]
    else:
        out = [W[(D <= v) & ((D > rs[k - 1]) if k else True)].sum() for k, v in enumerate(rs)]
    weighted = wa is not None or wb is not None
    return np.array(out, np.float64 if weighted else np.int64)


# ---- query_pairs ----


@pytest.mark.parametrize("p", PS)
def test_query_pairs_random_points(p):
    g = np.random.default_rng(50)
    for n, m in ((600, 3), (300, 1), (200, 8), (120, 16), (60, 64)):
        x = g.standard_normal((n, m))
        for r in radii(pdist_p(x[:40], p)):
            check_pairs(x, r, p=p, leafsizes=(1, 10))


@pytest.mark.parametrize("p", PS)
def test_query_pairs_duplicates_lattices_and_extremes(p):
    g = np.random.default_rng(51)
    dup = np.repeat(g.integers(-3, 4, (15, 3)).astype(float), 9, axis=0)[g.permutation(135)]
    for r in (0.0, 1.0, 2.0, INF):
        check_pairs(dup, r, p=p)
    grid = np.array(np.meshgrid(*[np.arange(-3.0, 4.0)] * 2)).reshape(2, -1).T
    for r in (0.0, 1.0, math.sqrt(2.0), 2.0):
        check_pairs(grid, r, p=p)
    span = np.array([[-1.7e308, 0.0], [1.7e308, 0.0], [0.0, 1.0]])
    for x in (1e8 + g.random((250, 3)), g.standard_normal((200, 3)) * 1e200, g.standard_normal((200, 3)) * 1e-200,
              g.integers(-30, 30, (120, 2)) * 5e-324, span):  # fmt: skip
        for r in radii(pdist_p(x, p)):
            check_pairs(x, r, p=p, leafsizes=(1, 10))
    check_pairs(np.ones((7, 2)), 0.0, p=p)
    for s in ("shell/3", "far/1e15", "ties/2", "lattice/int3d"):
        check_pairs(P.get(s).x, radii(pdist_p(P.get(s).x, p))[4], p=p, leafsizes=(1, 10))


def test_query_pairs_small_and_empty():
    assert S.KDTree(np.zeros((0, 3))).query_pairs(1.0).shape == (0, 2)
    assert S.KDTree(np.zeros((1, 3))).query_pairs(1.0).shape == (0, 2)
    z = S.KDTree(np.zeros((4, 0))).query_pairs(0.0)  # every distance 0
    assert z.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    got = S.KDTree(np.random.default_rng(52).random((30, 2))).query_pairs(INF)
    assert got.shape == (435, 2) and got._flags()["writeable"]  # a plain new array


# ---- count_neighbors ----


@pytest.mark.parametrize("p", PS)
def test_count_random_points(p):
    g = np.random.default_rng(53)
    for na, nb, m in ((500, 400, 3), (300, 200, 1), (200, 150, 8), (80, 90, 64)):
        a, b = g.standard_normal((na, m)), g.standard_normal((nb, m))
        b[:20] = a[:20]
        rs = radii(cdist_p(a[:1], b, p)[0])
        for leafsize in (1, 10):
            A, B = S.KDTree(a, leafsize=leafsize), S.KDTree(b, leafsize=leafsize)
            for r in rs:
                got = A.count_neighbors(B, r, p=p)
                assert type(got) is int and got == brute_count(a, b, r, p)[0], (leafsize, r)
            got = A.count_neighbors(B, rs, p=p)
            assert got.dtype == sd.int64 and np.array_equal(np.asarray(got), brute_count(a, b, rs, p))
            srt = sorted(rs)
            got = A.count_neighbors(B, srt, p=p, cumulative=False)
            assert np.array_equal(np.asarray(got), brute_count(a, b, srt, p, cumulative=False))
            got = A.count_neighbors(A, rs, p=p)  # self with itself: (i, i) and both (i, j), (j, i)
            assert np.array_equal(np.asarray(got), brute_count(a, a, rs, p))


@pytest.mark.parametrize("p", (1.0, 2.0, INF, 2.5))
def test_count_duplicates_lattices_and_extremes(p):
    g = np.random.default_rng(54)
    dup = np.repeat(g.integers(-3, 4, (15, 3)).astype(float), 9, axis=0)[g.permutation(135)]
    rs = [0.0, 1.0, 1.0, 2.0, 3.0, INF]
    for leafsize in (1, 4, 10):
        A = S.KDTree(dup, leafsize=leafsize)
        assert np.array_equal(np.asarray(A.count_neighbors(A, rs, p=p)), brute_count(dup, dup, rs, p))
        assert np.array_equal(np.asarray(A.count_neighbors(A, rs, p=p, cumulative=False)),
                              brute_count(dup, dup, rs, p, cumulative=False))  # fmt: skip
    for a, b in (
        (1e8 + g.random((200, 3)), 1e8 + g.random((150, 3))),
        (g.standard_normal((150, 3)) * 1e200, g.standard_normal((100, 3)) * 1e200),
        (g.standard_normal((150, 3)) * 1e-200, g.standard_normal((100, 3)) * 1e-200),
        (g.integers(-30, 30, (120, 2)) * 5e-324, g.integers(-30, 30, (90, 2)) * 5e-324),
        (np.array([[-1.7e308, 0.0], [1.7e308, 0.0], [0.0, 1.0]]), np.array([[1.7e308, 1.0], [0.0, 0.0]])),
    ):
        rs = radii(cdist_p(a[:2], b, p).ravel())  # fmt: skip
        for leafsize in (1, 10):
            A, B = S.KDTree(a, leafsize=leafsize), S.KDTree(b, leafsize=leafsize)
            assert np.array_equal(np.asarray(A.count_neighbors(B, rs, p=p)), brute_count(a, b, rs, p))
            for r in rs:
                assert A.count_neighbors(B, r, p=p) == brute_count(a, b, r, p)[0]


def test_count_weighted_forms_and_types():
    g = np.random.default_rng(55)
    a, b = g.random((300, 2)), g.random((200, 2))
    A, B = S.KDTree(a), S.KDTree(b)
    wa, wb = g.integers(-4, 9, 300).astype(float), g.integers(0, 5, 200) * 0.25  # exact sums
    rs = [0.02, 0.1, 0.1, 0.3, INF]
    got = A.count_neighbors(B, rs, weights=(wa, wb))
    assert got.dtype == sd.float64 and np.array_equal(np.asarray(got), brute_count(a, b, rs, wa=wa, wb=wb))
    got = A.count_neighbors(B, sorted(rs), weights=(wa, wb), cumulative=False)
    assert np.array_equal(np.asarray(got), brute_count(a, b, sorted(rs), wa=wa, wb=wb, cumulative=False))
    one = A.count_neighbors(B, 0.1, weights=(wa, None))
    assert type(one) is float and one == brute_count(a, b, 0.1, wa=wa, wb=np.ones(200))[0]
    assert A.count_neighbors(B, 0.1, weights=(None, wb)) == brute_count(a, b, 0.1, wa=np.ones(300), wb=wb)[0]
    plain = A.count_neighbors(B, 0.1, weights=(None, None))  # scipy: an int, as without weights
    assert type(plain) is int and plain == A.count_neighbors(B, 0.1)
    assert A.count_neighbors(A, 0.1, weights=wa) == brute_count(a, a, 0.1, wa=wa, wb=wa)[0]  # one array: self
    assert A.count_neighbors(A, 0.1, weights=list(wa)) == A.count_neighbors(A, 0.1, weights=(wa, wa))
    assert A.count_neighbors(B, 0.1, weights=(sd.asarray(wa), wb.tolist())) == A.count_neighbors(
        B, 0.1, weights=(wa, wb)
    )
    w = g.random(300) * 3  # rounding: within it of the exact sum  # fmt: skip
    got = A.count_neighbors(A, rs, weights=w)
    np.testing.assert_allclose(np.asarray(got), brute_count(a, a, rs, wa=w, wb=w), rtol=1e-12)
    assert type(A.count_neighbors(B, np.float64(0.1))) is int
    assert type(A.count_neighbors(B, np.array(0.1))) is int  # numpy's 0-d array is one radius (scipy)
    assert A.count_neighbors(B, [0.1]).shape == (1,)
    assert A.count_neighbors(B, []).shape == (0,) and A.count_neighbors(B, [], weights=(wa, wb)).dtype == sd.float64
    assert A.count_neighbors(B, sd.asarray([0.1, 0.2])).tolist() == A.count_neighbors(B, [0.1, 0.2]).tolist()
    assert A.count_neighbors(B, np.array([0.1, 0.2])).tolist() == A.count_neighbors(B, [0.1, 0.2]).tolist()
    e = S.KDTree(np.zeros((0, 2)))
    assert e.count_neighbors(B, 1.0) == 0 and A.count_neighbors(e, [1.0, 2.0]).tolist() == [0, 0]
    assert e.count_neighbors(e, 1.0, weights=(sd.zeros(0), None)) == 0.0
    z = S.KDTree(np.zeros((4, 0)))  # every distance 0
    assert z.count_neighbors(z, 0.0) == 16


def test_many_radii_cut_the_tasks_less_deep():
    # SPT25: tasks x (R + 1) sums are bounded, so many radii take shallower pieces; the counts are the same
    g = np.random.default_rng(56)
    a, b = g.random((3000, 2)), g.random((2500, 2))
    A, B = S.KDTree(a), S.KDTree(b)
    rs = np.sort(g.random(20_000) * 0.05)
    got = np.asarray(A.count_neighbors(B, rs))
    D = np.sort(cdist_p(a, b, 2.0).ravel())
    assert np.array_equal(got, np.searchsorted(D, rs, side="right"))
    w = g.random(3000)
    np.testing.assert_allclose(np.asarray(A.count_neighbors(B, rs[::50], weights=(w, None))),
                               brute_count(a, b, rs[::50], wa=w, wb=None), rtol=1e-12)  # fmt: skip


@given(
    na=st.integers(0, 50),
    nb=st.integers(0, 50),
    m=st.integers(0, 4),
    p=st.sampled_from(PS),
    leafsize=st.integers(1, 12),
    spread=st.sampled_from([1.0, 4.0, 1e-300, 1e150]),
    integers=st.booleans(),
    nr=st.integers(0, 5),
    weighted=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_hypothesis_brute_force(na, nb, m, p, leafsize, spread, integers, nr, weighted, seed):
    g = np.random.default_rng(seed)
    a = g.integers(-3, 4, (na, m)).astype(float) if integers else g.standard_normal((na, m))
    b = g.integers(-3, 4, (nb, m)).astype(float) if integers else g.standard_normal((nb, m))
    if na and nb and m and g.random() < 0.5:
        b[: min(nb, 3)] = a[g.integers(0, na, min(nb, 3))]
    a, b = a * spread, b * spread
    pool = [0.0, 0.5, 1.0, 2.0, 3.5, INF]
    if na and nb and m:
        pool += cdist_p(a, b, p).ravel()[g.integers(0, na * nb, 4)].tolist()  # the boundaries
    rs = [float(g.choice(pool)) if v > 3 else v * spread for v in g.choice(pool, nr)] if nr else []
    rs = [max(v, 0.0) for v in rs]
    A, B = S.KDTree(a, leafsize=leafsize), S.KDTree(b, leafsize=leafsize)
    wa = g.integers(-3, 4, na) * 0.5 if weighted else None  # exact sums
    wb = g.integers(0, 4, nb) * 0.25 if weighted else None
    w = (wa, wb) if weighted else None
    assert np.array_equal(np.asarray(A.count_neighbors(B, rs, p=p, weights=w)), brute_count(a, b, rs, p, wa, wb))
    srt = sorted(rs)
    assert np.array_equal(np.asarray(A.count_neighbors(B, srt, p=p, weights=w, cumulative=False)),
                          brute_count(a, b, srt, p, wa, wb, cumulative=False))  # fmt: skip
    if m:
        r = rs[0] if rs else 1.0
        assert np.array_equal(np.asarray(A.query_pairs(r, p=p)), brute_pairs(a, r, p))


# ---- scipy ----


@LIVE
def test_scipy_agrees():
    import scipy.spatial as SS

    g = np.random.default_rng(57)
    a, b = g.random((2000, 3)), g.random((1500, 3))
    A, B, SA, SB = S.KDTree(a), S.KDTree(b), SS.KDTree(a), SS.KDTree(b)
    rs = [0.01, 0.05, 0.2, 0.05]
    wa, wb = g.random(2000), g.random(1500)
    for p in (1.0, 2.0, INF, 3.0):
        assert sorted(map(tuple, A.query_pairs(0.04, p=p).tolist())) == sorted(SA.query_pairs(0.04, p=p))
        nd = SA.query_pairs(0.04, p=p, output_type="ndarray")  # the same pairs in an order of scipy's own
        assert np.array_equal(np.asarray(A.query_pairs(0.04, p=p)), nd[np.lexsort((nd[:, 1], nd[:, 0]))])
        assert A.count_neighbors(B, 0.05, p=p) == SA.count_neighbors(SB, 0.05, p=p)
        assert np.asarray(A.count_neighbors(B, rs, p=p)).tolist() == SA.count_neighbors(SB, rs, p=p).tolist()
        assert np.asarray(A.count_neighbors(B, sorted(rs), p=p, cumulative=False)).tolist() == \
            SA.count_neighbors(SB, sorted(rs), p=p, cumulative=False).tolist()  # fmt: skip
        assert A.count_neighbors(A, 0.05, p=p) == SA.count_neighbors(SA, 0.05, p=p)
        np.testing.assert_allclose(np.asarray(A.count_neighbors(B, rs, p=p, weights=(wa, wb))),
                                   SA.count_neighbors(SB, rs, p=p, weights=(wa, wb)), rtol=1e-12)  # fmt: skip
    assert type(SA.count_neighbors(SB, 0.05, weights=(None, None))) is int  # as here
    rec = refdata.data("scipy")["kdtree/count_self"]["value"]
    t = S.KDTree(P.get("nn/normal3").x[:50])
    assert t.count_neighbors(t, 0.5) == rec


@LIVE
def test_scipy_differences():
    """The [scipy와 다름] of query_pairs and count_neighbors (docs/api/spatial.md §4): scipy gives a set by
    default, counts every pair for r = NaN and counts pairs for a negative r; here the pairs
    are a sorted array and a NaN or negative r is an error."""
    import scipy.spatial as SS

    g = np.random.default_rng(58)
    a, b = g.random((50, 2)), g.random((40, 2))
    SA, SB = SS.KDTree(a), SS.KDTree(b)
    assert type(SA.query_pairs(0.2)) is set
    assert SA.count_neighbors(SB, math.nan) == 50 * 40
    assert SA.count_neighbors(SB, [-0.2])[0] > 0  # pairs "within" a negative radius
    A, B = S.KDTree(a), S.KDTree(b)
    with pytest.raises(ValueError):
        A.count_neighbors(B, math.nan)
    with pytest.raises(ValueError):
        A.count_neighbors(B, [-0.2, 0.1])
    with pytest.raises(ValueError):
        A.query_pairs(-0.2)


# ---- threads ----


@pytest.mark.usefixtures("restore_threads")
def test_one_three_ten_threads():
    g = np.random.default_rng(59)
    a = g.standard_normal((30_000, 3))
    a[::5] = np.round(a[::5], 1)
    b = g.standard_normal((20_000, 3))
    b[:2000] = a[:2000]
    wa, wb = g.random(30_000), g.standard_normal(20_000)  # rounding: the order of the sums shows
    rs = np.linspace(0.0, 0.2, 25)
    got = {}
    for th in (1, 3, 10):
        sd.set_num_threads(th)
        A, B = S.KDTree(a), S.KDTree(b)
        out = []
        for p in (2.0, 1.0, INF, 3.0):
            out.append(A.query_pairs(0.03, p=p))
            out.append(A.count_neighbors(B, rs, p=p))
            out.append(A.count_neighbors(B, rs, p=p, weights=(wa, wb)))
            out.append(A.count_neighbors(B, rs, p=p, weights=(wa, wb), cumulative=False))
            out.append(A.count_neighbors(A, rs, p=p, weights=wa))
        out.append(np.array([A.count_neighbors(B, 0.1, weights=(wa, None))]))
        got[th] = out
    for th in (3, 10):
        for k, (u, v) in enumerate(zip(got[1], got[th])):
            assert bits_equal(u, v), (th, k)


# ---- errors ----


def test_errors():
    t = S.KDTree(np.random.default_rng(60).random((10, 3)))
    u = S.KDTree(np.random.default_rng(61).random((8, 3)))
    v = S.KDTree(np.random.default_rng(62).random((8, 2)))
    with raises(TypeError, "query_pairs(): takes 1 positional argument but 2 were given"):
        t.query_pairs(1.0, 2.0)
    with raises(TypeError, "query_pairs(): unexpected keyword argument 'output_type'"):
        t.query_pairs(1.0, output_type="ndarray")
    with raises(TypeError, "query_pairs(): unexpected keyword argument 'eps'"):
        t.query_pairs(1.0, eps=0.1)
    with raises(ValueError, "query_pairs(): 'r' must be a nonnegative number or inf, not -1"):
        t.query_pairs(-1)
    with raises(ValueError, "query_pairs(): 'r' must be a nonnegative number or inf, not nan"):
        t.query_pairs(math.nan)
    with raises(TypeError, "query_pairs(): 'r' must be an int or a float, not list"):
        t.query_pairs([1.0])
    with raises(ValueError, "query_pairs(): 'p' must be a number of at least 1 or inf, not 0.5"):
        t.query_pairs(1.0, p=0.5)
    name = "count_neighbors"
    with raises(TypeError, f"{name}(): 'other' must be a KDTree, not list"):
        t.count_neighbors([[0.0, 0.0, 0.0]], 1.0)
    with raises(ValueError, f"{name}(): 'other' must have the same m as this tree (3), not 2"):
        t.count_neighbors(v, 1.0)
    with raises(TypeError, f"{name}(): takes 2 positional arguments but 3 were given"):
        t.count_neighbors(u, 1.0, 2.0)
    with raises(TypeError, f"{name}(): unexpected keyword argument 'workers'"):
        t.count_neighbors(u, 1.0, workers=2)
    with raises(ValueError, f"{name}(): 'r' must be a nonnegative number or inf, not -1.0"):
        t.count_neighbors(u, -1.0)
    with raises(ValueError, f"{name}(): 'r' must be a nonnegative number or inf, not nan"):
        t.count_neighbors(u, math.nan)  # scipy: every pair
    with raises(ValueError, f"{name}(): every entry of 'r' must be a nonnegative number or inf"):
        t.count_neighbors(u, [0.1, -0.1])
    with raises(ValueError, f"{name}(): every entry of 'r' must be a nonnegative number or inf"):
        t.count_neighbors(u, [0.1, math.nan])
    with raises(ValueError, f"{name}(): 'r' must be a number or a 1-D array, not a 2-D array"):
        t.count_neighbors(u, [[0.1, 0.2]])
    with raises(TypeError, f"{name}(): 'r' must be a number or an array of numbers, not str"):
        t.count_neighbors(u, "1")
    with raises(TypeError, f"{name}(): 'r' must be an int or a float, not bool"):
        t.count_neighbors(u, True)
    with raises(ValueError, f"{name}(): 'r' must be nondecreasing when cumulative=False"):
        t.count_neighbors(u, [0.2, 0.1], cumulative=False)  # scipy: "r must be non-decreasing ..."
    with raises(TypeError, f"{name}(): 'cumulative' must be True or False, not int"):
        t.count_neighbors(u, 0.1, cumulative=0)
    with raises(ValueError, f"{name}(): one array of 'weights' is for 'other' being this tree; give a tuple "
                            "(self's, other's) for two trees"):  # fmt: skip
        t.count_neighbors(u, 0.1, weights=np.ones(10))
    with raises(ValueError, f"{name}(): 'weights' as a tuple must have 2 entries (self's and other's), not 3"):
        t.count_neighbors(u, 0.1, weights=(None, None, None))
    with raises(ValueError, f"{name}(): the weights of this tree must have 10 entries (one per point), not 9"):
        t.count_neighbors(u, 0.1, weights=(np.ones(9), None))
    with raises(ValueError, f"{name}(): the weights of 'other' must have 8 entries (one per point), not 10"):
        t.count_neighbors(u, 0.1, weights=(None, np.ones(10)))
    with raises(ValueError, f"{name}(): the weights of this tree must have 1 dimension, not 2"):
        t.count_neighbors(t, 0.1, weights=np.ones((10, 1)))
    with raises(ValueError, f"{name}(): 'weights' must be finite"):
        t.count_neighbors(t, 0.1, weights=np.full(10, math.nan))  # scipy: a NaN count
    with raises(TypeError, f"{name}(): 'weights' must be real, not a complex128 array"):
        t.count_neighbors(t, 0.1, weights=np.ones(10, complex))
    with raises(ValueError, f"{name}(): 'p' must be a number of at least 1 or inf, not 0.5"):
        t.count_neighbors(u, 0.1, p=0.5)
