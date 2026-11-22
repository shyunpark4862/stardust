"""The tile order of sd.spatial's pair kernel (docs/design/spatial.md §3.1; decisions SPT22, SPT26, SPT27):
the batched kernel behind pdist and cdist (4 x 8 or 2 x 8 pairs side by side, the remainder rows and
columns padded) gives every pair the bits of the one-pair kernel behind the distance functions, at every
length around the block of 32 and the old summation boundaries, for every metric, weighted or not, with
NaN and inf; below 16 terms the euclidean distance is sd.linalg.vector_norm(x - y)'s bits, chebyshev is
vector_norm(ord=inf)'s at every length, and minkowski's p = 1, 2, inf are cityblock, euclidean and
chebyshev bit for bit. The careful paths keep huge and tiny coordinates finite and accurate."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import stardust as sd

from .._bits import bits_equal, bits_equal_any_nan

S = sd.spatial
LENGTHS = (0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 511, 512, 513, 1024, 2049)
FLOAT_METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra", "braycurtis", "cosine",
                 "correlation", "hamming", "jaccard")  # fmt: skip
WEIGHTED = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra", "braycurtis", "cosine", "correlation",
            "hamming", "jaccard", "minkowski")  # fmt: skip


def _data(n, ma=13, mb=11, seed=0, specials=False):
    g = np.random.default_rng([seed, n])
    a, b = g.standard_normal((ma, n)), g.standard_normal((mb, n))
    if n:  # ties for hamming and jaccard, zeros for canberra's 0/0
        a[:, ::3] = np.round(a[:, ::3])
        b[:, ::3] = np.round(b[:, ::3])
    if specials and n:
        a[1, 0], b[2, -1], a[3, n // 2], b[4, 0] = np.nan, np.inf, -np.inf, np.inf
        a[5] = 0.0
    return a, b


def _kw(metric, n, weighted, seed=0):
    kw = {}
    if metric == "minkowski":
        kw["p"] = 3.0
    if weighted:
        g = np.random.default_rng([seed, n, 7])
        w = g.uniform(0.1, 2.0, n)
        w[::4] = 0.0
        kw["w"] = w
    return kw


def _one(metric, x, y, kw):
    return getattr(S, metric)(x, y, **kw)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("metric", FLOAT_METRICS + ("minkowski",))
def test_batched_pairs_are_the_one_pair_kernel(metric, n):
    a, b = _data(n, specials=True)
    for weighted in (False, True):
        if weighted and metric not in WEIGHTED:
            continue
        kw = _kw(metric, n, weighted)
        c = np.asarray(S.cdist(a, b, metric=metric, **kw))
        d = np.asarray(S.pdist(a, metric=metric, **kw))
        one = np.array([[_one(metric, x, y, kw) for y in b] for x in a])
        assert bits_equal_any_nan(c, one), (metric, n, weighted)
        iu = np.triu_indices(a.shape[0], 1)
        selfc = np.asarray(S.cdist(a, a, metric=metric, **kw))
        assert bits_equal_any_nan(d, selfc[iu]), (metric, n, weighted)
        assert bits_equal_any_nan(selfc, selfc.T), (metric, n, weighted)  # fl(x - y) = -fl(y - x)


@pytest.mark.parametrize("n", (3, 16, 33))
def test_batched_pairs_of_the_scalar_metrics(n):
    """minkowski, seuclidean, mahalanobis, haversine (n = 2) and the int64 and bool counting paths."""
    a, b = _data(n)
    V = np.linspace(0.5, 2.0, n)
    g = np.random.default_rng(n)
    q = g.standard_normal((n, n))
    VI = q @ q.T + np.eye(n)
    for metric, kw, one in (
        ("seuclidean", {"V": V}, lambda x, y: S.seuclidean(x, y, V)),
        ("mahalanobis", {"VI": VI}, lambda x, y: S.mahalanobis(x, y, VI)),
        ("minkowski", {"p": 0.7}, lambda x, y: S.minkowski(x, y, p=0.7)),
    ):
        c = np.asarray(S.cdist(a, b, metric=metric, **kw))
        assert bits_equal(c, np.array([[one(x, y) for y in b] for x in a])), metric
    h = g.uniform(-1.5, 1.5, (9, 2))
    c = np.asarray(S.cdist(h, h[:5], metric="haversine"))
    assert bits_equal(c, np.array([[S.haversine(x, y) for y in h[:5]] for x in h]))
    ia, ib = g.integers(-2, 3, (9, n)), g.integers(-2, 3, (7, n))
    ba, bb = g.random((9, n)) < 0.4, g.random((7, n)) < 0.4
    w = g.uniform(0, 2, n)
    for metric in ("hamming", "jaccard"):
        for x, y in ((ia, ib), (ba, bb)):
            for kw in ({}, {"w": w}):
                c = np.asarray(S.cdist(x, y, metric=metric, **kw))
                one = np.array([[getattr(S, metric)(p, r, **kw) for r in y] for p in x])
                assert bits_equal(c, one), (metric, x.dtype, kw)
                # the same values compared as float64: the same counts, the same bits (SPT23)
                f = np.asarray(S.cdist(x.astype(float), y.astype(float), metric=metric, **kw))
                assert bits_equal(c, f), (metric, x.dtype, kw)


@pytest.mark.parametrize("rows", [(1, 1), (3, 5), (4, 8), (5, 9), (9, 17), (2, 70)])
def test_remainder_rows_and_columns(rows):
    ma, mb = rows
    a, b = _data(37, ma, mb, seed=3)
    for metric in ("euclidean", "braycurtis", "jaccard"):
        c = np.asarray(S.cdist(a, b, metric=metric))
        assert c.shape == (ma, mb)
        assert bits_equal_any_nan(c, np.array([[_one(metric, x, y, {}) for y in b] for x in a]))


@pytest.mark.parametrize("n", [1, 2, 3, 8, 15])
def test_short_rows_are_vector_norm_bits(n):
    g = np.random.default_rng(n)
    x, y = g.standard_normal((40, n)), g.standard_normal((40, n))
    for i in range(40):
        z = sd.asarray(x[i]) - sd.asarray(y[i])
        assert S.euclidean(x[i], y[i]) == sd.linalg.vector_norm(z)


@pytest.mark.parametrize("n", [1, 15, 16, 33, 512, 2049])
def test_chebyshev_is_vector_norm_inf(n):
    g = np.random.default_rng(n)
    x, y = g.standard_normal((10, n)), g.standard_normal((10, n))
    for i in range(10):
        assert S.chebyshev(x[i], y[i]) == sd.linalg.vector_norm(sd.asarray(x[i]) - sd.asarray(y[i]), ord=math.inf)


@pytest.mark.parametrize("n", [1, 5, 33, 100])
def test_minkowski_p_1_2_inf_are_the_three_metrics(n):
    a, b = _data(n, specials=True)
    w = np.random.default_rng(n).uniform(0.0, 2.0, n)
    for p, metric in ((1, "cityblock"), (2.0, "euclidean"), (math.inf, "chebyshev")):
        for kw in ({}, {"w": w}):
            assert bits_equal_any_nan(S.cdist(a, b, metric="minkowski", p=p, **kw), S.cdist(a, b, metric=metric, **kw))
            assert bits_equal_any_nan(S.minkowski(a, b[:13] if b.shape[0] >= 13 else a, p=p, **kw),
                                      getattr(S, metric)(a, b[:13] if b.shape[0] >= 13 else a, **kw))  # fmt: skip


def test_careful_paths_keep_huge_and_tiny_coordinates():
    """SPT27: the sum of squares overflows or underflows; the distance is still accurate."""
    for scale in (1e200, 1e160, 1e-160, 1e-200, 5e-324):
        x, y = np.array([3.0, 0.0]) * scale, np.array([0.0, 4.0]) * scale
        d = S.euclidean(x, y)
        assert math.isfinite(d) and d > 0
        assert abs(d - 5 * scale) <= 2 * math.ulp(5 * scale) or scale == 5e-324
        assert S.minkowski(x, y, p=3.0) == pytest.approx(scale * 91.0 ** (1 / 3), rel=1e-15) or scale == 5e-324
    assert S.minkowski([0.0, 0.0], [3.0, 4.0], p=2000) == 4.0
    assert S.sqeuclidean([1e200], [0.0]) == math.inf  # the true value overflows
    assert S.sqeuclidean([3e-160], [0.0]) == pytest.approx(9e-320, rel=1e-3)  # subnormal, careful path
    big = np.full(5, 1e300)
    assert S.cosine(big, 2 * big) == 0.0 and S.cosine(big, -big) == pytest.approx(2.0, abs=1e-15)
    tiny = np.array([1e-310, 3e-310, -2e-310])
    assert S.cosine(tiny, tiny * 2) == pytest.approx(0.0, abs=1e-15)
    assert S.correlation([5e-324, 0.0], [0.0, 1e-320]) == pytest.approx(2.0, abs=1e-15)  # the mean of subnormals


@given(
    st.integers(0, 40),
    st.integers(1, 9),
    st.integers(1, 9),
    st.sampled_from(FLOAT_METRICS + ("minkowski",)),
    st.booleans(),
    st.integers(0, 2**32 - 1),
)
def test_tile_and_one_pair_agree_on_random_data(n, ma, mb, metric, weighted, seed):
    g = np.random.default_rng(seed)
    a = g.choice([0.0, 1.0, -2.5, np.nan, np.inf, 1e300, 1e-300], (ma, n))
    a = np.where(g.random((ma, n)) < 0.5, g.standard_normal((ma, n)), a)
    b = np.where(g.random((mb, n)) < 0.5, g.standard_normal((mb, n)), a[g.integers(0, ma, mb)])
    if weighted and metric not in WEIGHTED:
        weighted = False
    kw = _kw(metric, n, weighted, seed)
    c = np.asarray(S.cdist(a, b, metric=metric, **kw))
    assert bits_equal_any_nan(c, np.array([[_one(metric, x, y, kw) for y in b] for x in a]))
    if ma >= 2:
        assert bits_equal_any_nan(S.pdist(a, metric=metric, **kw),
                                  np.asarray(S.cdist(a, a, metric=metric, **kw))[np.triu_indices(ma, 1)])  # fmt: skip
