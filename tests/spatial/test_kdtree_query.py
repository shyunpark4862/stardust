"""KDTree.query (docs/api/spatial.md §4, docs/design/spatial.md §3.4, §4 neighbours 1-3; decisions SPT50-SPT62).

1. The result is the brute force with the same kernel: cdist(x, data) (minkowski for p not 1, 2, inf) row
   by row sorted by (distance, index), the first k with distance < distance_upper_bound, the rest inf and
   n; the distances are cdist's bits. On random, duplicated, tied, clustered, far, huge and tiny and high-m
   data, for every leafsize, p and k form, and with hypothesis.
2. On the lattice sets of tests/spatial/ref/neighbors.npz (every difference and sum exact, so the computed
   distance is the true one rounded) the result is the Arb references sorted by (distance, index).
3. On the other reference sets the result is a true kNN within the distance contract.
Plus the shapes, the fill, the strict upper bound, scipy 1.18.1 live and its records, the errors and the
same bytes for 1, 3 and 10 threads."""

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

S = sd.spatial
INF = math.inf


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


def cdist_p(x, data, p):
    """The kernel's distances: cdist of the metric of p (SPT2: the same bits as the tree)."""
    if p == 1:
        return np.asarray(S.cdist(x, data, metric="cityblock"))
    if p == 2:
        return np.asarray(S.cdist(x, data))
    if p == INF:
        return np.asarray(S.cdist(x, data, metric="chebyshev"))
    return np.asarray(S.cdist(x, data, metric="minkowski", p=p))


def brute(data, x, k, p=2.0, ub=INF):
    """(distances, indices) (Q, len(ranks)) of the brute force: rows sorted by (distance, index)."""
    data = np.asarray(data, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(math.prod(x.shape[:-1]), data.shape[1])
    ranks = list(range(1, k + 1)) if isinstance(k, int) else list(k)
    n = data.shape[0]
    D = cdist_p(x, data, p) if n and len(x) else np.zeros((len(x), n))
    dist = np.full((len(x), len(ranks)), INF)
    idx = np.full((len(x), len(ranks)), n, dtype=np.int64)
    for q in range(len(x)):
        order = np.lexsort((np.arange(n), D[q]))
        order = order[D[q][order] < ub]
        for j, r in enumerate(ranks):
            if r <= len(order):
                dist[q, j], idx[q, j] = D[q, order[r - 1]], order[r - 1]
    return dist, idx


def check(data, x, k=1, p=2.0, ub=INF, leafsizes=(1, 3, 10, 64)):
    want_d, want_i = brute(data, x, k, p, ub)
    for leafsize in leafsizes:
        t = S.KDTree(data, leafsize=leafsize)
        d, i = t.query(x, k=k if not isinstance(k, int) else list(range(1, k + 1)), p=p, distance_upper_bound=ub)
        d, i = np.asarray(d).reshape(want_d.shape), np.asarray(i).reshape(want_i.shape)
        assert bits_equal(d, want_d), (leafsize, p, k, ub)
        assert np.array_equal(i, want_i), (leafsize, p, k, ub)


PS = (1.0, 2.0, INF, 1.5, 3.0)


@pytest.mark.parametrize("p", PS)
def test_random_points_are_the_brute_force(p):
    g = np.random.default_rng(10)
    for n, m in ((500, 3), (300, 1), (200, 8), (150, 16), (80, 64)):
        data, x = g.standard_normal((n, m)), g.standard_normal((25, m))
        check(data, x, k=7, p=p)
        check(data, data[:20], k=3, p=p)  # the data points themselves: distance 0 first


@pytest.mark.parametrize("p", PS)
def test_duplicates_ties_and_lattices(p):
    g = np.random.default_rng(11)
    dup = np.repeat(g.integers(-3, 4, (15, 3)).astype(float), 9, axis=0)[g.permutation(135)]
    check(dup, g.integers(-4, 5, (20, 3)).astype(float), k=20, p=p)
    grid = np.array(np.meshgrid(*[np.arange(-3.0, 4.0)] * 2)).reshape(2, -1).T
    check(grid, np.array([[0.0, 0.0], [0.5, 0.5], [3.0, -3.0], [10.0, 0.0]]), k=12, p=p)
    six = np.ones((6, 2))
    check(six, np.ones((1, 2)), k=6, p=p)


def test_tie_order_against_scipy_and_its_record():
    # SPT4: (distance, index); scipy's record for six identical points is [1, 2, 3, 4, 5, 0]
    rec = refdata.data("scipy")["kdtree/tie_order"]["value"]["array"]
    assert rec == [1, 2, 3, 4, 5, 0]
    assert S.KDTree(np.ones((6, 2))).query(np.ones(2), k=6).indices.tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("p", (1.0, 2.0, INF, 2.5))
def test_clustered_far_and_extreme_data(p):
    g = np.random.default_rng(12)
    centers = g.standard_normal((6, 4)) * 50
    clustered = (centers[:, None, :] + g.standard_normal((6, 70, 4)) * 1e-3).reshape(-1, 4)
    check(clustered, centers + 0.01, k=5, p=p)
    check(clustered, g.standard_normal((10, 4)) * 50, k=9, p=p)
    far = 1e8 + g.random((200, 3))
    check(far, 1e8 + g.random((15, 3)), k=5, p=p)
    huge = g.standard_normal((150, 3)) * 1e200  # the careful path (sums overflow)
    check(huge, g.standard_normal((10, 3)) * 1e200, k=4, p=p)
    tiny = g.standard_normal((150, 3)) * 1e-200  # the careful path (sums underflow)
    check(tiny, g.standard_normal((10, 3)) * 1e-200, k=4, p=p)
    sub = g.integers(-50, 50, (100, 2)) * 5e-324  # subnormal coordinates
    check(sub, g.integers(-50, 50, (10, 2)) * 5e-324, k=6, p=p)
    span = np.array([[-1.7e308, 0.0], [1.7e308, 0.0], [0.0, 1.0], [1e308, -1e308]])
    check(span, np.array([[0.0, 0.0], [1.6e308, 1.0]]), k=4, p=p, leafsizes=(1, 10))


def test_the_reference_shell_sets_and_far_clusters():
    for sid in ("shell/3", "far/1e8", "far/1e15", "ties/2", "dup/3"):
        s = P.get(sid)
        for p in (1.0, 2.0, INF):
            check(s.x, s.queries, k=6, p=p, leafsizes=(1, 10))


def test_k_forms_and_upper_bound():
    g = np.random.default_rng(13)
    data, x = g.random((400, 3)), g.random((30, 3))
    for k in ([1], [3, 1], [2, 2, 2], [5, 400, 401, 10**6], list(range(1, 30))):
        check(data, x, k=k, leafsizes=(10,))
    d = cdist_p(x, data, 2.0)
    for ub in (0.0, 0.05, 0.2, float(np.sort(d[0])[4]), INF):
        check(data, x, k=10, ub=ub, leafsizes=(1, 10))
    # strict: a point at exactly the bound is left out (scipy's rule, SPT19)
    t = S.KDTree(data)
    exact = float(np.sort(d[0])[4])
    assert t.query(x[0], k=10, distance_upper_bound=exact).distances.tolist()[4] == INF
    assert t.query(x[0], k=10, distance_upper_bound=math.nextafter(exact, INF)).distances.tolist()[4] == exact


def test_brute_force_path_of_high_m():
    # SPT21, SPT75: for p = 2 and many columns for the number of points, query takes kneighbors' brute force
    # over the data in input order; the result is the tree's, bit for bit, for every k form and bound
    from stardust.spatial import _kdtree as K

    g = np.random.default_rng(15)
    for n, m in ((3000, 12), (500, 8), (40, 3)):
        assert K._brute_pays(n, m, 1)
        data, x = g.random((n, m)), g.random((70, m))
        data[5] = data[3]  # a duplicate: a tie broken by the index
        t = S.KDTree(data)
        d = cdist_p(x, data, 2.0)
        for k in (1, 7, [3, 1], [2, n + 5]):
            for ub in (INF, float(np.sort(d[0])[3])):
                ranks = sd.asarray(
                    [k] if isinstance(k, int) and k == 1 else (list(range(1, k + 1)) if isinstance(k, int) else k),
                    dtype=sd.int64,
                )
                td, ti = t._tree.query(sd.asarray(x), ranks, 2.0, ub)
                got = t.query(x, k=k, distance_upper_bound=ub)
                gd, gi = np.asarray(got.distances).reshape(len(x), -1), np.asarray(got.indices).reshape(len(x), -1)
                assert bits_equal(gd, np.asarray(td)) and np.array_equal(gi, np.asarray(ti)), (n, m, k, ub)
    assert not K._brute_pays(10**6, 3, 10) and not K._brute_pays(10**5, 10, 1) and not K._brute_pays(0, 16, 1)


def test_overflowing_distances_are_never_neighbours():
    # SPT37 (kneighbors' rule, the same here): a neighbour has distance < distance_upper_bound, so a pair
    # whose distance overflows to inf is padding even with the default bound inf
    data = [[-1.7e308, 0.0], [1.7e308, 0.0]]
    for p in (1.0, 2.0, 3.0):
        d, i = S.KDTree(data).query([1.7e308, 0.0], k=2, p=p)
        assert d.tolist() == [0.0, INF] and i.tolist() == [1, 2], p
        kd, ki = S.kneighbors([[1.7e308, 0.0]], data, k=2, **({} if p == 2.0 else {"metric": "minkowski", "p": p}))
        assert bits_equal(kd[0], d) and bits_equal(ki[0], i), p


def test_k_larger_than_n_and_empty_trees():
    data = np.random.default_rng(14).random((3, 2))
    d, i = S.KDTree(data).query(np.zeros(2), k=5)
    assert d.tolist()[3:] == [INF, INF] and i.tolist()[3:] == [3, 3]  # scipy's fill: inf and n
    e = S.KDTree(np.zeros((0, 3)))
    rec = refdata.data("scipy")["kdtree/n0_query"]["value"]
    d, i = e.query(np.zeros(3), k=2)
    assert d.tolist() == rec[0]["array"] and i.tolist() == rec[1]["array"]
    z = S.KDTree(np.zeros((5, 0)))  # scipy: IndexError
    d, i = z.query(np.zeros((2, 0)), k=3)
    assert d.tolist() == [[0.0] * 3] * 2 and i.tolist() == [[0, 1, 2]] * 2


def test_shapes_and_types():
    data = np.random.default_rng(15).random((20, 2))
    t = S.KDTree(data)
    d, i = t.query([0.5, 0.5])
    assert type(d) is float and type(i) is int  # scipy: float and numpy.int64
    r = t.query([0.5, 0.5], k=[1])
    assert r.distances.shape == (1,) and r.indices.dtype == sd.int64
    assert t.query([[0.5, 0.5]]).distances.shape == (1,)
    assert t.query([0.5, 0.5], k=3).distances.shape == (3,)
    assert t.query(np.zeros((2, 3, 2)), k=[2, 5]).distances.shape == (2, 3, 2)
    assert t.query(np.zeros((2, 3, 2)), k=1).distances.shape == (2, 3)
    assert t.query(np.zeros((0, 2)), k=3).distances.shape == (0, 3)
    assert t.query(np.zeros((0, 2))).indices.shape == (0,)
    res = t.query(np.zeros((4, 2)), k=2)
    assert type(res) is sd._results.QueryResult and res._fields == ("distances", "indices")
    for a in res:
        assert not a._flags()["writeable"]
    assert bits_equal(t.query(sd.asarray(np.zeros((4, 2))), k=2), res)
    assert bits_equal(t.query(np.zeros((4, 2), dtype=np.int64), k=2), res)


@pytest.mark.parametrize("p", (1.0, 2.0, INF))
def test_lattice_references_exactly(p):
    """§4 neighbours 2: on the lattice sets the computed distances are the true ones rounded, so the result
    is the Arb references sorted by (distance, index), ties and all."""
    r = refdata.ref("neighbors")
    method = {1.0: "p1", 2.0: "p2", INF: "pinf"}[p]
    for sid in sorted(set(P.ids("lattice")) & set(P.ids("neighbors"))):
        s = P.get(sid)
        t = S.KDTree(s.x, leafsize=4)
        rows = [k for k in range(r.points) if r.inputs["set"][k] == sid]
        d, i = t.query(np.stack([r.inputs["query"][k] for k in rows]), k=15, p=p)
        d, i = np.asarray(d), np.asarray(i)
        for q, k in enumerate(rows):
            hi = r.methods[method].hi[k]
            order = np.lexsort((np.arange(hi.size), hi))[:15]
            assert np.array_equal(i[q], order), (sid, q)
            assert bits_equal(d[q], hi[order]), (sid, q)


@pytest.mark.parametrize(("p", "metric"), [(1.0, "cityblock"), (2.0, "euclidean"), (INF, "chebyshev")])
def test_general_references_within_the_contract(p, metric):
    """§4 neighbours 3: every returned distance is within the distance contract of the true one, no point
    left out is nearer than the k-th returned by more than the contract, and none returned is farther."""
    r = refdata.ref("neighbors")
    method = {1.0: "p1", 2.0: "p2", INF: "pinf"}[p]
    K = 8
    for sid in P.ids("neighbors"):
        s = P.get(sid)
        t = S.KDTree(s.x)
        rows = [k for k in range(r.points) if r.inputs["set"][k] == sid]
        d, i = t.query(np.stack([r.inputs["query"][k] for k in rows]), k=K, p=p)
        d, i = np.asarray(d), np.asarray(i)
        n = s.x.shape[1]
        for q, k in enumerate(rows):
            hi, lo = r.methods[method].hi[k], r.methods[method].lo[k]
            true = hi + lo
            for j in range(K):
                assert refdata.units(metric, d[q, j], hi[i[q, j]], lo[i[q, j]], n=n) <= 1.0, (sid, q, j)
            dk = d[q, -1]
            eta = refdata.bound(metric, dk, n=n) * 2
            chosen = np.zeros(true.size, bool)
            chosen[i[q]] = True
            assert np.all(true[~chosen] >= dk - eta), (sid, q)
            assert np.all(true[chosen] <= dk + eta), (sid, q)


@LIVE
def test_scipy_agrees_on_sets_without_ties():
    import scipy.spatial as SS

    g = np.random.default_rng(16)
    for n, m in ((2000, 3), (500, 8)):
        data, x = g.random((n, m)), g.random((40, m))
        for p in (1.0, 2.0, INF, 3.0):
            sd_d, sd_i = S.KDTree(data).query(x, k=6, p=p)
            sp_d, sp_i = SS.KDTree(data).query(x, k=6, p=p)
            assert np.array_equal(np.asarray(sd_i), sp_i), p
            # the same points; the values may differ in the last bits (scipy's tree sums in another order)
            np.testing.assert_allclose(np.asarray(sd_d), sp_d, rtol=8e-16 * m, atol=0)
    s = P.get("dup/3")  # ties: the same distances, scipy's order of equal ones is its own
    sd_d, _ = S.KDTree(s.x).query(s.queries, k=12)
    sp_d, _ = SS.KDTree(s.x).query(s.queries, k=12)
    assert np.array_equal(np.asarray(sd_d), sp_d)


@LIVE
def test_scipy_records_of_the_differences():
    """The [scipy와 다름] of query (docs/api/spatial.md §4): scipy takes k=True and k=2.0, k=0 is numpy's
    message, k=[0] kills the process, a negative bound acts as its absolute value, a NaN bound fills
    everything; here they are errors."""
    rec = refdata.data("scipy")
    assert rec["kdtree/k_true"]["value"][1] == {"np_int": 31}
    assert rec["kdtree/k_list_zero"]["crash"] == -11
    assert rec["kdtree/dub_negative_same_as_1"]["value"] is True
    assert rec["kdtree/dub_nan"]["value"][1]["array"] == [50, 50, 50]
    assert rec["kdtree/leafsize_float"]["value"] == 2
    data = P.get("nn/normal3").x[:50]
    t = S.KDTree(data)
    with pytest.raises(TypeError):
        t.query(np.zeros(3), k=True)
    with pytest.raises(TypeError):
        t.query(np.zeros(3), k=2.0)
    with pytest.raises(ValueError):
        t.query(np.zeros(3), k=[0])
    with pytest.raises(ValueError):
        t.query(np.zeros(3), k=3, distance_upper_bound=-1.0)
    gt = rec["kdtree/k_gt_n"]["value"]
    d, i = S.KDTree(data[:2]).query(np.zeros(3), k=4)
    assert i.tolist() == gt[1]["array"]
    assert d.tolist()[2:] == [INF, INF]
    assert d.tolist()[:2] == pytest.approx(gt[0]["array"][:2], rel=1e-15)


@given(
    n=st.integers(0, 70),
    m=st.integers(0, 4),
    k=st.integers(1, 12),
    p=st.sampled_from(PS),
    leafsize=st.integers(1, 12),
    spread=st.sampled_from([1.0, 4.0, 1e-300, 1e150]),
    integers=st.booleans(),
    bound=st.sampled_from([INF, 0.0, 0.5, 2.0]),
    seed=st.integers(0, 2**32 - 1),
)
def test_hypothesis_brute_force(n, m, k, p, leafsize, spread, integers, bound, seed):
    g = np.random.default_rng(seed)
    data = g.integers(-3, 4, (n, m)).astype(float) if integers else g.standard_normal((n, m))
    x = g.integers(-4, 5, (6, m)).astype(float) if integers else g.standard_normal((6, m))
    if n and m and g.random() < 0.5:
        x[:3] = data[g.integers(0, n, 3)]  # queries on data points
    data, x = data * spread, x * spread
    check(data, x, k=k, p=p, ub=bound * spread, leafsizes=(leafsize,))


@pytest.mark.usefixtures("restore_threads")
def test_one_three_ten_threads():
    g = np.random.default_rng(17)
    data = g.standard_normal((30_000, 3))
    data[::5] = np.round(data[::5], 1)
    x = g.standard_normal((3000, 3))
    x[::4] = data[::40][: len(x[::4])]
    got = {}
    for th in (1, 3, 10):
        sd.set_num_threads(th)
        t = S.KDTree(data)
        got[th] = [t.query(x, k=k, p=p) for k, p in ((1, 2.0), (10, 2.0), ([3, 1], 1.0), (5, INF), (4, 3.0))]
        got[th].append(t.query(x, k=8, distance_upper_bound=0.05))
    for th in (3, 10):
        for a, b in zip(got[1], got[th]):
            assert bits_equal(a, b), th


# ---- errors ----


def test_query_errors():
    t = S.KDTree(np.random.default_rng(18).random((10, 3)))
    x = np.zeros(3)
    with raises(TypeError, "query(): takes 1 positional argument but 2 were given"):
        t.query(x, 3)  # SPT15: k is keyword only
    with raises(TypeError, "query(): unexpected keyword argument 'workers'"):
        t.query(x, workers=2)
    with raises(TypeError, "query(): unexpected keyword argument 'eps'"):
        t.query(x, eps=0.1)
    with raises(TypeError, "query(): 'k' must be an int or a sequence of ints, not bool"):
        t.query(x, k=True)
    with raises(TypeError, "query(): 'k' must be an int or a sequence of ints, not float"):
        t.query(x, k=2.0)
    with raises(TypeError, "query(): 'k' must be an int or a sequence of ints, not str"):
        t.query(x, k="3")
    with raises(TypeError, "query(): 'k' must be an int or a sequence of ints, not a sequence holding float"):
        t.query(x, k=[1, 2.0])
    with raises(TypeError, "query(): 'k' must be an int or a sequence of ints, not a sequence holding bool"):
        t.query(x, k=[True])
    with raises(TypeError, "query(): 'k' must be an int or a sequence of ints, not a 2-D array"):
        t.query(x, k=np.ones((1, 1), dtype=np.int64))
    with raises(ValueError, "query(): 'k' must be at least 1, not 0"):
        t.query(x, k=0)
    with raises(ValueError, "query(): 'k' must be at least 1, not -2"):
        t.query(x, k=-2)
    with raises(ValueError, "query(): every entry of 'k' must be at least 1, not 0"):
        t.query(x, k=[0])  # scipy: a segmentation fault
    with raises(ValueError, "query(): 'k' must not be an empty sequence"):
        t.query(x, k=[])
    with raises(OverflowError, "query(): 'k' is out of range"):
        t.query(x, k=2**63)
    with raises(TypeError, "query(): 'p' must be an int or a float, not str"):
        t.query(x, p="2")
    with raises(ValueError, "query(): 'p' must be a number of at least 1 or inf, not 0.5"):
        t.query(x, p=0.5)
    with raises(ValueError, "query(): 'p' must be a number of at least 1 or inf, not nan"):
        t.query(x, p=math.nan)
    with raises(ValueError, "query(): 'distance_upper_bound' must be a number >= 0 or inf, not -1.0"):
        t.query(x, distance_upper_bound=-1.0)
    with raises(ValueError, "query(): 'distance_upper_bound' must be a number >= 0 or inf, not nan"):
        t.query(x, distance_upper_bound=math.nan)
    with raises(TypeError, "query(): 'distance_upper_bound' must be an int or a float, not NoneType"):
        t.query(x, distance_upper_bound=None)
    with raises(ValueError, "query(): 'x' must have 3 entries on its last axis (the tree's m), not 2"):
        t.query([0.0, 0.0])
    with raises(ValueError, "query(): 'x' must be finite"):
        t.query([[0.0, 0.0, 0.0], [0.0, math.nan, 0.0]])
    with raises(ValueError, "query(): 'x' must be finite"):
        t.query([math.inf, 0.0, 0.0])
    with raises(TypeError, "query(): 'x' must be real, not a complex128 array"):
        t.query(np.zeros(3, complex))
    with raises(MemoryError, "query(): cannot allocate 147573952589676412912 bytes: more than this machine's "
                             "physical memory"):  # fmt: skip
        t.query(x, k=2**63 - 1)
    assert t.query(x, k=np.int64(2), p=np.float32(2.0), distance_upper_bound=np.float64(9)).indices.shape == (2,)
    assert t.query(x, k=np.array([2, 1])).indices.shape == (2,)
    assert t.query(x, k=sd.asarray([2, 1])).indices.shape == (2,)
