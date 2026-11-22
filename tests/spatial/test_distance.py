"""pdist, cdist and the distance functions (docs/api/spatial.md §2, docs/design/spatial.md §2.2): the
definitions against scipy 1.18.1 live and against its records (tests/spatial/data), the weights, V and VI,
the values at the edges (NaN, inf, zero rows, constant rows, no columns, fewer than two rows), the batch
shapes of the distance functions, the dtypes of hamming and jaccard (SPT23), the inputs taken (SPT13) and
the result types."""

import math

import numpy as np
import pytest
import scipy.spatial.distance as D

import stardust as sd

from .._bits import bits_equal, bits_equal_any_nan
from . import points as P
from . import refdata
from .conftest import LIVE

# scipy's and numpy's own warnings at NaN and inf (sd warns about nothing)
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

S = sd.spatial
SCIPY = refdata.data("scipy")
SKLEARN = refdata.data("sklearn")
R = refdata.data("r")
METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "seuclidean", "mahalanobis", "cosine",
           "correlation", "canberra", "braycurtis", "hamming", "jaccard")  # fmt: skip


def _np(x):
    return np.asarray(x)


def _kw(metric, n, scipy=False):
    if metric == "minkowski":
        return {"p": 3.0}
    if metric == "seuclidean":
        return {"V": P.variances(n)}
    if metric == "mahalanobis":
        return {"VI": P.inverse_covariances(n)}
    return {}


# ---- the definitions --------------------------------------------------------------------------


@LIVE
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("n", [1, 3, 17, 64])
def test_cdist_and_pdist_agree_with_scipy(metric, n):
    g = np.random.default_rng(n)
    a, b = g.standard_normal((20, n)), g.standard_normal((15, n))
    a[:, ::2] = np.round(a[:, ::2])  # ties for hamming, zeros for jaccard
    b[:, ::2] = np.round(b[:, ::2])
    kw = _kw(metric, n)
    assert np.allclose(_np(S.cdist(a, b, metric=metric, **kw)), D.cdist(a, b, metric, **kw), rtol=1e-13, atol=1e-15,
                       equal_nan=True)  # fmt: skip
    assert np.allclose(_np(S.pdist(a, metric=metric, **kw)), D.pdist(a, metric, **kw), rtol=1e-13, atol=1e-15,
                       equal_nan=True)  # fmt: skip
    if metric not in ("seuclidean", "mahalanobis"):
        w = P.weights(n, "zeros")
        assert np.allclose(_np(S.cdist(a, b, metric=metric, w=w, **kw)), D.cdist(a, b, metric, w=w, **kw), rtol=1e-12,
                           atol=1e-15, equal_nan=True)  # fmt: skip


@LIVE
def test_default_V_and_VI_are_scipys():
    g = np.random.default_rng(1)
    a, b = g.standard_normal((30, 4)), g.standard_normal((12, 4))
    for metric in ("seuclidean", "mahalanobis"):
        assert np.allclose(_np(S.pdist(a, metric=metric)), D.pdist(a, metric), rtol=1e-12)
        assert np.allclose(_np(S.cdist(a, b, metric=metric)), D.cdist(a, b, metric), rtol=1e-12)
    # bit for bit the documented defaults: var(correction=1) and inv(cov(correction=1))
    sa = sd.asarray(a)
    V = sd.var(sa, axis=0, correction=1)
    assert bits_equal(S.pdist(a, metric="seuclidean"), S.pdist(a, metric="seuclidean", V=V))
    VI = sd.linalg.inv(sd.cov(sa, rowvar=False, correction=1))
    assert bits_equal(S.pdist(a, metric="mahalanobis"), S.pdist(a, metric="mahalanobis", VI=VI))
    nan = a.copy()
    nan[3, 1] = math.nan  # SPT34: a NaN in the data makes the default VI NaN, every distance NaN (as scipy)
    assert np.all(np.isnan(_np(S.pdist(nan, metric="mahalanobis"))))
    assert np.all(np.isnan(D.pdist(nan, "mahalanobis")))
    both = sd.concat([sa, sd.asarray(b)], axis=0)
    V2 = sd.var(both, axis=0, correction=1)
    assert bits_equal(S.cdist(a, b, metric="seuclidean"), S.cdist(a, b, metric="seuclidean", V=V2))


@LIVE
def test_haversine_is_sklearns_meaning():
    from sklearn.metrics.pairwise import haversine_distances

    g = np.random.default_rng(2)
    h = np.column_stack([g.uniform(-1.5, 1.5, 30), g.uniform(-3, 3, 30)])
    assert np.allclose(_np(S.cdist(h, h, metric="haversine")), haversine_distances(h), rtol=1e-12, atol=1e-15)
    assert abs(S.haversine([0.0, 0.0], [0.0, math.pi / 2]) - math.pi / 2) <= math.ulp(math.pi / 2)
    assert S.haversine([0.3, -1.1], [0.3, -1.1]) == 0.0


def test_haversine_keeps_its_digits_at_the_antipode():
    """sklearn's 2 asin(sqrt(h)) loses them there (the records); sd's form does not (SPT31)."""
    r = refdata.ref("haversine")
    m = r.methods["haversine"]
    for j, k in enumerate(m.index):
        sid = str(r.inputs["set"][k])
        if "antipode" not in sid:
            continue
        got = S.haversine(r.inputs["u"][k], r.inputs["v"][k])
        assert abs(got - m.hi[j]) <= 2 * math.ulp(m.hi[j]), sid
    rec = SKLEARN["haversine/antipode1"]["value"]  # 1e-8 from the antipode
    j = next(j for j, k in enumerate(m.index) if str(r.inputs["set"][k]) == "haversine/antipode1")
    assert abs(rec - m.hi[j]) > 1000 * math.ulp(m.hi[j])


# ---- the records of scipy's edges ---------------------------------------------------------------

# where sd departs from scipy at the edge pairs (docs/api/spatial.md §1.5): chebyshev propagates a NaN
# (scipy's C loop skips it), weighted canberra propagates (scipy's nansum), and the careful paths keep
# huge and tiny pairs finite and accurate where scipy overflows or underflows
CAREFUL = ("huge", "tiny", "subnormal")


def _scipy_value(rec):
    return rec.get("value", math.nan) if isinstance(rec.get("value", 0.0), float) else math.nan


@pytest.mark.parametrize("metric", METRICS)
def test_edge_pairs_agree_with_scipys_records(metric):
    for eid, u, v in P.edge_pairs():
        n = len(u)
        kw = _kw(metric, n)
        for weighted in (False, True):
            if weighted and metric in ("seuclidean", "mahalanobis"):
                continue
            key = f"pdist{'_w' if weighted else ''}/{metric}/{eid}"
            rec = SCIPY[key]
            assert rec["digest"] == P.digest(u, v)
            if "error" in rec:  # scipy refuses (no columns for its seuclidean V, ...)
                continue
            if weighted:
                kw = {**kw, "w": P.weights(n, "zeros")}
            ours = S.pdist(np.vstack([u, v]), metric=metric, **kw)[0]
            theirs = rec["value"]
            if eid in CAREFUL and metric in ("euclidean", "minkowski", "seuclidean", "cosine", "correlation",
                                             "mahalanobis", "sqeuclidean"):  # fmt: skip
                continue  # sd's careful paths: tests/spatial/test_reference.py holds them to Arb
            nan_u = bool(np.isnan(u).any())
            nan_z = np.isnan(u - v)
            if metric == "chebyshev" and nan_z.any() and not (weighted and _zero_weight_at(nan_z, n)):
                assert math.isnan(ours) and not math.isnan(theirs), key
                continue
            if metric == "canberra" and weighted and nan_u:
                assert math.isnan(ours), key
                continue
            if math.isnan(theirs) or math.isnan(ours):
                assert math.isnan(ours) and math.isnan(theirs), (key, ours, theirs)
            elif metric in ("cosine", "correlation"):  # scipy's 1 - c: absolute error n u (sd: SPT29)
                assert ours == pytest.approx(theirs, rel=0, abs=1e-15), (key, ours, theirs)
            else:
                assert ours == pytest.approx(theirs, rel=1e-14, abs=1e-300), (key, ours, theirs)


def _zero_weight_at(mask, n):
    w = P.weights(n, "zeros")
    return bool(np.all(w[mask] == 0))


def test_the_documented_departures():
    assert math.isnan(S.cdist([[math.nan, 0.0]], [[1.0, 5.0]], metric="chebyshev")[0, 0])
    assert SCIPY["cdist/chebyshev_nan"]["value"] == 5.0
    assert math.isnan(S.canberra([math.nan, 1.0], [1.0, 2.0]))  # scipy's function: nansum
    assert math.isnan(S.pdist([[math.nan, 1.0], [1.0, 2.0]], metric="canberra", w=[1.0, 1.0])[0])
    assert S.minkowski([0.0, 0.0], [3.0, 4.0], p=2000) == 4.0 and math.isinf(SCIPY["minkowski/p2000"]["value"])
    assert S.euclidean([1e200, 0.0], [-1e200, 0.0]) == 2e200 and math.isinf(SCIPY["euclidean/1e200"]["value"])
    big = np.array([2**60, 0], dtype=np.int64)
    assert S.hamming(big, big + np.array([1, 0])) == 0.5 and SCIPY["hamming/int_2^60"]["value"] in (0.0, 1.0)
    assert S.jaccard(np.array([2**60]), np.array([2**60 + 1])) == 0.0  # both nonzero
    assert S.cosine(np.ones((2, 3)), np.ones((2, 3))).shape == (2,)  # scipy: ValueError (one pair only)
    assert SCIPY["cosine/batch"]["error"] == "ValueError"


def test_r_dist_agrees():
    """R 4.6.2's stats::dist on its recorded sets (euclidean, manhattan, maximum, minkowski p = 3)."""
    for sid in ("lattice/int3d", "normal/8", "uniform/3"):
        s = P.get(sid)
        x = s.x[:20]
        for method, metric, kw in (("euclidean", "euclidean", {}), ("manhattan", "cityblock", {}),
                                   ("maximum", "chebyshev", {}), ("minkowski", "minkowski", {"p": 3.0})):  # fmt: skip
            rec = R[f"dist/{sid}/{method}"]
            assert rec["digest"] == P.set_digest(s)
            full = _np(S.squareform(S.pdist(x, metric=metric, **kw)))
            want = [full[a, b] for b in range(20) for a in range(b + 1, 20)]  # R: the lower triangle by column
            assert np.allclose(want, rec["values"], rtol=1e-14, atol=0), (sid, method)


# ---- edges, shapes, types ---------------------------------------------------------------------


def test_no_columns():
    """n = 0: the definition on an empty sum (scipy's pdist)."""
    x = np.zeros((3, 0))
    want = {"euclidean": 0.0, "sqeuclidean": 0.0, "cityblock": 0.0, "chebyshev": 0.0, "canberra": 0.0,
            "minkowski": 0.0, "jaccard": 0.0, "cosine": math.nan, "correlation": math.nan, "braycurtis": math.nan,
            "hamming": math.nan}  # fmt: skip
    for metric, v in want.items():
        d = _np(S.pdist(x, metric=metric))
        assert d.shape == (3,) and (np.all(np.isnan(d)) if math.isnan(v) else np.all(d == v)), metric
        f = getattr(S, metric)(np.zeros(0), np.zeros(0))
        assert math.isnan(f) if math.isnan(v) else f == v, metric
    assert bits_equal(S.pdist(x), D.pdist(x))


def test_fewer_than_two_rows_and_empty_sides():
    for m in (0, 1):
        d = S.pdist(np.zeros((m, 3)))
        assert isinstance(d, sd.ndarray) and d.shape == (0,) and d.dtype == sd.float64
    assert S.cdist(np.zeros((0, 3)), np.zeros((4, 3))).shape == (0, 4)
    assert S.cdist(np.zeros((2, 3)), np.zeros((0, 3))).shape == (2, 0)


def test_values_at_the_edges():
    nan, inf = math.nan, math.inf
    assert math.isnan(S.euclidean([nan, 1.0], [0.0, 1.0]))
    assert S.euclidean([inf, 1.0], [0.0, 1.0]) == inf
    assert math.isnan(S.euclidean([inf, 1.0], [inf, 1.0]))  # inf - inf
    assert math.isnan(S.cosine([0.0, 0.0], [1.0, 2.0]))
    assert S.cosine([1.0, 2.0], [2.0, 4.0]) == 0.0  # the rows scaled by powers of two: the same unit row
    assert S.cosine([1.0, 2.0], [-2.0, -4.0]) == pytest.approx(2.0, abs=4.5e-16)  # 2 ||x^||^2: within an ULP
    assert math.isnan(S.correlation([3.0, 3.0, 3.0], [1.0, 2.0, 4.0]))
    c = [0.1] * 7  # a constant row whose plain mean is not exactly 0.1: still NaN
    assert math.isnan(S.correlation(c, [1.0, 2.0, 4.0, 0.0, 1.0, 2.0, 3.0]))
    assert S.canberra([0.0, 1.0], [0.0, 3.0]) == 0.5  # the 0/0 term is 0
    assert math.isnan(S.braycurtis([0.0, 0.0], [0.0, 0.0])) and S.braycurtis([1.0, -2.0], [-1.0, 2.0]) == inf
    assert S.hamming([nan, 1.0], [nan, 1.0]) == 0.5  # NaN != NaN
    assert S.jaccard([0.0, 0.0], [0.0, 0.0]) == 0.0 and S.jaccard([nan, 0.0], [0.0, 0.0]) == 1.0
    assert math.isnan(S.mahalanobis([0.0, 0.0], [1.0, 0.0], [[-1.0, 0.0], [0.0, 1.0]]))  # a negative form
    assert S.chebyshev([1.0, nan], [2.0, 3.0], w=[1.0, 0.0]) == 1.0  # over w > 0 only
    assert math.isnan(S.euclidean([1.0, nan], [2.0, 3.0], w=[1.0, 0.0]))  # 0 * NaN is NaN
    assert S.correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], centered=False) == S.cosine([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])


def test_batch_shapes():
    g = np.random.default_rng(3)
    a, b = g.standard_normal((3, 4)), g.standard_normal((2, 1, 4))
    d = S.euclidean(a, b)
    assert isinstance(d, sd.ndarray) and d.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert d[i, j] == S.euclidean(a[j], b[i, 0])
    assert isinstance(S.cityblock([1.0, 2.0], [0.0, 0.0]), float)
    assert S.sqeuclidean(a, a[0]).shape == (3,)
    assert S.haversine(np.zeros((5, 2)), np.ones(2)).shape == (5,)
    assert S.hamming(np.zeros((0, 3)), np.zeros(3)).shape == (0,)


def test_results_are_new_writable_float64_arrays():
    x = np.arange(12.0).reshape(4, 3)
    for r in (S.pdist(x), S.cdist(x, x), S.euclidean(x, x)):
        assert isinstance(r, sd.ndarray) and r.dtype == sd.float64
        r[0] = 1.0  # writable


def test_inputs_taken():
    """SPT13: an sd.ndarray, a sequence, numpy arrays of any real dtype, a frame DataFrame (NaN for a
    missing value); bool and int64 become float64 (hamming and jaccard keep them)."""
    rows = [[0, 1, 2], [3, 1, 0], [1, 1, 1]]
    ref = S.pdist(np.asarray(rows, dtype=np.float64))
    assert bits_equal(S.pdist(rows), ref)
    assert bits_equal(S.pdist(sd.asarray(rows)), ref)
    assert bits_equal(S.pdist(np.asarray(rows, dtype=np.int32)), ref)
    df = sd.frame.DataFrame({"a": [0, 3, 1], "b": [1, 1, 1], "c": [2, 0, 1]})
    assert bits_equal(S.pdist(df), ref)
    dfn = sd.frame.DataFrame({"a": [0.0, None, 1.0], "b": [1.0, 1.0, 1.0]})
    d = _np(S.pdist(dfn))
    assert math.isnan(d[0]) and math.isnan(d[2]) and d[1] == 1.0
    big = sd.frame.DataFrame({"a": [2**60, 2**60 + 1]})
    assert S.pdist(big, metric="hamming")[0] == 1.0  # int64 kept


def test_hamming_and_jaccard_dtypes():
    """SPT23: compared as given; bool rows through the bit-packed path, the same values as float64."""
    g = np.random.default_rng(4)
    for n in (1, 63, 64, 65, 200):
        b = g.random((9, n)) < 0.3
        for metric in ("hamming", "jaccard"):
            assert bits_equal(S.pdist(b, metric=metric), S.pdist(b.astype(np.float64), metric=metric))
            assert bits_equal(S.pdist(b, metric=metric), S.pdist(b.astype(np.int64), metric=metric))
            assert np.array_equal(_np(S.pdist(b, metric=metric)), D.pdist(b, metric))
    i = np.array([[2**60, 1, 0], [2**60 + 1, 1, 0]], dtype=np.int64)
    assert S.pdist(i, metric="hamming")[0] == 1 / 3
    assert S.pdist(i.astype(np.float64), metric="hamming")[0] == 0.0  # the same double
    assert S.hamming(np.array([True, False]), np.array([1, 0])) == 0.0  # bool with int64: int64


def test_symmetry_and_pdist_is_cdist():
    g = np.random.default_rng(5)
    x = g.standard_normal((7, 5))
    x[2, 1] = math.nan
    for metric in METRICS + ("haversine",):
        xx = x[:, :2] if metric == "haversine" else x
        kw = _kw(metric, xx.shape[1])
        c = _np(S.cdist(xx, xx, metric=metric, **kw))
        assert bits_equal_any_nan(c, c.T), metric
        assert bits_equal_any_nan(S.pdist(xx, metric=metric, **kw), c[np.triu_indices(7, 1)]), metric
