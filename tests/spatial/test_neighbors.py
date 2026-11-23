"""kneighbors and radius_neighbors are their definition (docs/api/spatial.md §3, docs/design/spatial.md
§2.3, §3.2, §4 neighbours 1): the k smallest (d, j) of cdist's row with d < distance_upper_bound, padded
with (inf, mB); the j with d <= r, increasing. The distances are cdist's bits, and the answer is the same
on every path (direct, the GEMM pre-selection of euclidean and sqeuclidean, and the KD tree over XB of the
Minkowski metrics, each forced on and off),
for every metric, on the adversarial sets of §7.1 (far-from-origin clusters where the expanded form loses
every digit, duplicates, ties, a shell of distances 1 ULP apart, squares near overflow and underflow), and
on hypothesis' clusters."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from . import points as P

S = sd.spatial
N = S._neighbors
INF = math.inf

METRICS = (("euclidean", {}), ("sqeuclidean", {}), ("cityblock", {}), ("chebyshev", {}), ("minkowski", {"p": 3.0}),
           ("minkowski", {"p": 0.5}), ("seuclidean", {}), ("mahalanobis", {}), ("cosine", {}),
           ("correlation", {}), ("canberra", {}), ("braycurtis", {}), ("hamming", {}), ("jaccard", {}))  # fmt: skip
WEIGHTED = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "cosine", "correlation", "canberra",
            "braycurtis", "hamming", "jaccard")  # fmt: skip


def brute_knn(D, k, ub=INF):
    """(distances, indices) of the definition from a cdist matrix D (numpy)."""
    ma, mb = D.shape
    d = np.full((ma, k), INF)
    i = np.full((ma, k), mb, dtype=np.int64)
    for r in range(ma):
        j = np.nonzero(D[r] < ub)[0]
        o = j[np.lexsort((j, D[r, j]))][:k]
        d[r, : len(o)], i[r, : len(o)] = D[r, o], o
    return d, i


def brute_radius(D, r):
    r = np.broadcast_to(np.asarray(r, dtype=np.float64), (D.shape[0],))
    rows = [np.nonzero(D[q] <= r[q])[0] for q in range(D.shape[0])]
    indptr = np.concatenate([[0], np.cumsum([len(x) for x in rows])]).astype(np.int64)
    idx = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64)
    dist = np.concatenate([D[q, x] for q, x in enumerate(rows)]) if rows else np.zeros(0)
    return indptr, idx, dist


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_knn(A, B, k, metric="euclidean", ub=INF, paths=(0, 1, 2, 3), **kw):
    D = np.asarray(S.cdist(A, B, metric=metric, **kw))
    wd, wi = brute_knn(D, k, ub)
    for path in paths:
        got = N._knn("kneighbors", A, B, k, metric, kw.get("p"), kw.get("w"), kw.get("V"), kw.get("VI"), ub, path)
        d, i = np.asarray(got.distances), np.asarray(got.indices)
        if k == 1:
            d, i = d[:, None], i[:, None]
        assert np.array_equal(i, wi), (metric, path)
        assert same_bits(d, wd), (metric, path)


def check_radius(A, B, r, metric="euclidean", paths=(0, 1, 2, 3), **kw):
    D = np.asarray(S.cdist(A, B, metric=metric, **kw))
    wp, wi, wd = brute_radius(D, r)
    for path in paths:
        got = N._radius("radius_neighbors", A, B, r, metric, kw.get("p"), kw.get("w"), kw.get("V"), kw.get("VI"),
                        False, path)  # fmt: skip
        assert np.array_equal(np.asarray(got.indptr), wp), (metric, path)
        assert np.array_equal(np.asarray(got.indices), wi), (metric, path)
        assert same_bits(np.asarray(got.distances), wd), (metric, path)
        cnt = N._radius("radius_neighbors", A, B, r, metric, kw.get("p"), kw.get("w"), kw.get("V"), kw.get("VI"),
                        True, path)  # fmt: skip
        assert np.array_equal(np.asarray(cnt), np.diff(wp)), (metric, path)


def _normal(ma, mb, n, seed):
    g = np.random.default_rng(seed)
    return g.standard_normal((ma, n)), g.standard_normal((mb, n))


# ---- every metric, both paths -------------------------------------------------------------------


@pytest.mark.parametrize("metric,kw", METRICS, ids=[m + str(kw.get("p", "")) for m, kw in METRICS])
def test_every_metric_is_the_definition(metric, kw):
    A, B = _normal(37, 211, 9, 1)
    if metric in ("hamming", "jaccard"):
        A, B = np.round(A), np.round(B)  # equal entries and zeros
    for k in (1, 4, 300):
        check_knn(A, B, k, metric, **kw)
    check_knn(A, B, 7, metric, ub=float(np.median(np.asarray(S.cdist(A, B, metric=metric, **kw)))), **kw)
    D = np.asarray(S.cdist(A, B, metric=metric, **kw))
    check_radius(A, B, float(np.quantile(D, 0.1)), metric, **kw)


@pytest.mark.parametrize("metric", WEIGHTED)
def test_weighted(metric):
    A, B = _normal(23, 150, 6, 2)
    w = np.random.default_rng(3).uniform(0.0, 2.0, 6)
    w[2] = 0.0
    kw = {"w": w, **({"p": 3.0} if metric == "minkowski" else {})}
    if metric in ("hamming", "jaccard"):
        A, B = np.round(A), np.round(B)
    check_knn(A, B, 5, metric, **kw)
    D = np.asarray(S.cdist(A, B, metric=metric, **kw))
    check_radius(A, B, float(np.quantile(D, 0.2)), metric, **kw)


def test_counting_dtypes_and_haversine():
    g = np.random.default_rng(4)
    a, b = g.random((30, 70)) < 0.3, g.random((120, 70)) < 0.3
    for metric in ("hamming", "jaccard"):
        check_knn(a, b, 6, metric)
        check_knn(a.astype(np.int64), b, 6, metric)
        check_radius(a, b, 0.25, metric)
    h, q = g.uniform(-1.5, 1.5, (300, 2)), g.uniform(-1.5, 1.5, (20, 2))
    check_knn(q, h, 5, "haversine")
    check_radius(q, h, 0.3, "haversine")


def test_many_blocks_and_segments():
    # more XB rows than a GEMM block (2048) and a direct block, few XA rows (XB cut into segments), many
    # XA rows (chunks), k past a block
    A, B = _normal(5, 9000, 24, 5)
    check_knn(A, B, 3)
    check_knn(A, B, 2500, paths=(2,))
    check_knn(A, B, 40, "sqeuclidean")
    check_radius(A, B, 5.5)
    A2, B2 = _normal(700, 3000, 17, 6)
    check_knn(A2, B2, 2)
    check_radius(A2, B2, 4.0, paths=(1, 2))


def test_far_to_near_order():
    # XB sorted from far to near the queries: every GEMM block lowers T, so the block's candidates are many
    # and the k of smallest L~ go first (SPT72); the block-wide bound (SPT71) drops the rest
    g = np.random.default_rng(11)
    B, A = g.standard_normal((9000, 16)), g.standard_normal((40, 16))
    B = B[np.argsort(-np.linalg.norm(B - A.mean(axis=0), axis=1), kind="stable")]
    for k in (1, 3, 10):
        check_knn(A, B, k, paths=(1, 2))
        check_knn(A, B, k, "sqeuclidean", paths=(2,))
    check_knn(A[:1], B, 5, paths=(2,))


def test_paths_chosen_by_shape():
    # shapes where the automatic path is the GEMM pre-selection (n = 3, many rows) and the KD tree over XB
    # (SPT73, SPT74): the same answer as the forced paths
    A, B = _normal(2100, 2100, 3, 12)
    check_knn(A, B, 4)
    check_knn(A, B, 4, "sqeuclidean")
    check_knn(A, B, 2, "cityblock")
    check_radius(A, B, 0.1)
    A, B = _normal(300, 1100, 5, 13)
    check_knn(A, B, 3, "minkowski", p=1.5)
    check_radius(A, B, 1.0, "minkowski", p=1.5)


def test_shapes_and_edges():
    A, B = _normal(6, 10, 3, 7)
    check_knn(A, B, 15)  # k > mB: padded with (inf, mB)
    check_knn(A, np.zeros((0, 3)), 3)
    check_knn(np.zeros((0, 3)), B, 3)
    check_knn(np.zeros((4, 0)), np.zeros((5, 0)), 2)  # no columns: every distance 0
    check_radius(A, np.zeros((0, 3)), 1.0)
    check_radius(np.zeros((0, 3)), B, 1.0)
    r = S.kneighbors(A, B, k=3)
    assert isinstance(r, sd._results.QueryResult)
    d, i = r
    assert d.shape == (6, 3) and i.shape == (6, 3) and d.dtype == sd.float64 and i.dtype == sd.int64
    for x in (d, i):
        with pytest.raises(ValueError):
            x[0, 0] = 0
    d1, i1 = S.kneighbors(A, B)
    assert d1.shape == (6,) and i1.shape == (6,)
    assert np.array_equal(np.asarray(i1), np.asarray(i)[:, 0])
    # k as ranks (scipy's meaning): any order, repeats, past mB
    dk, ik = S.kneighbors(A, B, k=[3, 1, 1, 12])
    assert np.array_equal(np.asarray(ik)[:, :3], np.asarray(i)[:, [2, 0, 0]])
    assert np.all(np.asarray(ik)[:, 3] == 10) and np.all(np.asarray(dk)[:, 3] == INF)
    dk2, _ = S.kneighbors(A, B, k=np.array([2]))
    assert dk2.shape == (6, 1)


def test_upper_bound_is_strict_and_excludes_inf_and_nan():
    A = np.array([[0.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 1.0], [3.0, 0.0]])
    d, i = S.kneighbors(A, B, k=4, distance_upper_bound=1.0)
    assert np.all(np.asarray(i) == 4)  # d = 1 is not < 1
    d, i = S.kneighbors(A, B, k=4, distance_upper_bound=2.0)
    assert np.asarray(i).tolist() == [[0, 2, 4, 4]]
    # NaN distances (a zero row's cosine) and inf ones (an overflowing square) never count
    d, i = S.kneighbors([[1.0, 0.0]], [[0.0, 0.0], [2.0, 1.0], [1.0, 0.0]], k=3, metric="cosine")
    assert np.asarray(i).tolist() == [[2, 1, 3]]
    d, i = S.kneighbors([[0.0]], [[1e300], [1.0]], k=2, metric="sqeuclidean")
    assert np.asarray(i).tolist() == [[1, 2]] and np.asarray(d)[0, 1] == INF
    r = S.radius_neighbors([[0.0]], [[1e300], [1.0]], INF, metric="sqeuclidean")
    assert np.asarray(r.indices).tolist() == [0, 1]  # d <= inf holds for inf


# ---- the adversarial sets of §7.1 -----------------------------------------------------------------

ADVERSARIAL = ("far/1e8", "far/1e8x16", "far/1e15", "shell/3", "dup/3", "ties/2", "scaled/1e150", "scaled/1e160",
               "scaled/1e-160", "lattice/int2d", "lattice/bigint3", "lattice/dyadic3", "manifold/64")  # fmt: skip


@pytest.mark.parametrize("sid", ADVERSARIAL)
def test_adversarial_sets(sid):
    s = P.get(sid)
    X = s.x
    Q = s.queries if s.queries is not None else X[::7]
    for metric in ("euclidean", "sqeuclidean"):
        check_knn(Q, X, 1, metric)
        check_knn(Q, X, 7, metric)
        check_knn(X, X, 12, metric)  # every point finds itself (and its duplicates) at 0
        D = np.asarray(S.cdist(Q, X, metric=metric))
        finite = D[np.isfinite(D)]
        if finite.size:
            check_radius(Q, X, float(np.quantile(finite, 0.05)), metric)
            check_radius(Q, X, D[:, 3].copy(), metric)  # r is a returned distance: the boundary


def test_far_cluster_finds_the_true_neighbours():
    # sklearn's expanded form gives 0 for every distance here and misses all 20 neighbours (§3.2)
    s = P.get("far/1e8")
    d, i = S.kneighbors(s.queries, s.x)
    D = np.asarray(S.cdist(s.queries, s.x))
    assert np.array_equal(np.asarray(i), np.argmin(D, axis=1))
    assert np.all(np.asarray(d) > 0.01)


def test_ties_order_by_index():
    # six copies of one point (scipy [1, 2, 3, 4, 5, 0], sklearn [3, 4, 1, 5, 2, 0]): (d, j) order
    X = np.ones((6, 3))
    for path in (1, 2):
        r = N._knn("kneighbors", np.ones((1, 3)), X, 6, "euclidean", None, None, None, None, INF, path)
        assert np.asarray(r.indices).tolist() == [[0, 1, 2, 3, 4, 5]]
    lat = P.get("ties/2")
    check_knn(lat.queries, lat.x, 9)


# ---- hypothesis: clusters anywhere --------------------------------------------------------------


@st.composite
def clusters(draw):
    n = draw(st.integers(1, 24))
    mb = draw(st.integers(1, 150))
    ma = draw(st.integers(1, 12))
    centre = draw(st.sampled_from([0.0, 3.0, -1e3, 1e8, -1e12, 1e15]))
    spread = draw(st.sampled_from([1e-6, 1.0, 1e3]))
    seed = draw(st.integers(0, 2**31))
    g = np.random.default_rng(seed)
    B = centre + spread * g.standard_normal((mb, n))
    A = centre + spread * g.standard_normal((ma, n))
    if draw(st.booleans()):
        B, A = np.round(B / spread) * spread, np.round(A / spread) * spread  # ties
    if mb and draw(st.booleans()):
        B = np.concatenate([B, B[: max(1, mb // 3)]])  # duplicates
    if mb and ma and draw(st.booleans()):
        A[0] = B[0]
    k = draw(st.integers(1, 15))
    return A, B, k


@settings(deadline=None)
@given(
    clusters(), st.sampled_from([("euclidean", {}), ("cityblock", {}), ("chebyshev", {}), ("minkowski", {"p": 3.0})])
)
def test_hypothesis_tree_path_is_the_definition(case, mk):
    A, B, k = case
    metric, kw = mk
    check_knn(A, B, k, metric, paths=(3,), **kw)
    if B.shape[0] and A.shape[0]:
        D = np.asarray(S.cdist(A, B, metric=metric, **kw))
        check_radius(A, B, D[:, min(k, D.shape[1]) - 1].copy(), metric, paths=(3,), **kw)


@settings(deadline=None)
@given(clusters(), st.sampled_from(["euclidean", "sqeuclidean"]))
def test_hypothesis_gemm_path_is_the_definition(case, metric):
    A, B, k = case
    check_knn(A, B, k, metric, paths=(2,))
    if B.shape[0] and A.shape[0]:
        D = np.asarray(S.cdist(A, B, metric=metric))
        check_radius(A, B, D[:, min(k, D.shape[1]) - 1].copy(), metric, paths=(2,))


def test_radius_stops_past_the_pair_budget():
    """Every path counts the pairs it finds and stops with the same MemoryError once they pass what
    physical memory holds (here a lowered number); the counts alone gather nothing and never stop."""
    g = np.random.default_rng(5)
    A, B = g.random((300, 3)), g.random((400, 3))
    msg = r"^radius_neighbors\(\): the result is larger than this machine's physical memory$"
    for path in (1, 2, 3):
        with pytest.raises(MemoryError, match=msg):
            N._radius("radius_neighbors", A, B, math.inf, "euclidean", None, None, None, None, False, path, 1000)
        full = N._radius("radius_neighbors", A, B, math.inf, "euclidean", None, None, None, None, False, path, 120000)
        assert full.indices.shape == (120000,)
        cnt = N._radius("radius_neighbors", A, B, math.inf, "euclidean", None, None, None, None, True, path, 1000)
        assert np.asarray(cnt).tolist() == [400] * 300
    with pytest.raises(MemoryError, match=msg):  # a metric without a tree or a GEMM path
        N._radius("radius_neighbors", A, B, math.inf, "cosine", None, None, None, None, False, 0, 1000)


# ---- RadiusResult ---------------------------------------------------------------------------------


def test_radius_result():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    r = S.radius_neighbors(X, X, 1.0)
    assert repr(r) == "RadiusResult(shape=(4, 4), nnz=10)"
    assert r.shape == (4, 4)
    assert np.asarray(r.indptr).tolist() == [0, 3, 6, 9, 10]
    assert np.asarray(r.indices).tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 3]
    for a in (r.indptr, r.indices, r.distances):
        with pytest.raises(ValueError):
            a[0] = 1
    assert r.indptr.dtype == sd.int64 and r.indices.dtype == sd.int64 and r.distances.dtype == sd.float64
    m = r.tocsr()
    assert isinstance(m, sd.sparse.csr_array) and m.shape == (4, 4)
    assert m.nnz == 10 and m.count_nonzero() == 4  # the zero distances are stored zeros (SPT3)
    assert m.eliminate_zeros() is None and m.nnz == 4
    assert r.tocsr().nnz == 10  # a new array every call
    c = r.tocsr()
    c.data *= 2  # its data is a writable copy: the result's distances stay as they were
    assert c.data is not r.distances and np.asarray(c.data).tolist() == [2 * d for d in np.asarray(r.distances)]
    assert np.asarray(r.distances).tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert np.asarray(r.tocsr().data).tolist() == np.asarray(r.distances).tolist()
    assert np.array_equal(np.asarray(r.tocsr().toarray()), np.where(np.asarray(S.cdist(X, X)) <= 1.0,
                                                                     np.asarray(S.cdist(X, X)), 0.0))  # fmt: skip
    n = S.radius_neighbors(X, X, [1.0, 0.0, 0.5, 10.0], return_length=True)
    assert np.asarray(n).tolist() == [3, 1, 2, 4] and n.dtype == sd.int64
    n[0] = 7  # a plain new array
    assert np.asarray(S.radius_neighbors(X, X, [0.5], return_length=True)).tolist() == [2, 1, 2, 1]


def test_inputs():
    A, B = [[0.0, 0.0], [2.0, 2.0]], [[1, 1], [3, 3], [0, 0]]
    d, i = S.kneighbors(A, B, k=2)
    assert np.asarray(i).tolist() == [[2, 0], [0, 1]]
    df = sd.frame.DataFrame({"x": [0.0, 2.0], "y": [0.0, 2.0]})
    assert np.array_equal(np.asarray(S.kneighbors(df, sd.asarray(B)).indices), [2, 0])
    assert np.asarray(S.radius_neighbors(np.asarray(A), B, 1.5).indices).tolist() == [0, 2, 0, 1]
