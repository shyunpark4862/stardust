"""Descriptive statistics (T8, docs/api/stats.md §4; decisions T28): scipy.stats' values on
every axis and nan_policy, the exact values (rational arithmetic) on ill-conditioned data,
and the argument rules."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

S = sd.stats


def _close(a, b, rtol=1e-13, atol=1e-14):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True), (a, b)


def _data(shape=(30, 5, 4), nan=True, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, size=shape)
    if nan:
        flat = x.reshape(-1)
        flat[[17, 101]] = np.nan
    return x


AXES = [None, 0, 1, 2, -1, (0, 2), (1, 2)]
POLICIES = ["propagate", "omit"]


def _scipy_axis(axis):
    return axis  # scipy takes None, an int or a tuple


@pytest.mark.parametrize(("axis", "pol"), list(itertools.product(AXES, POLICIES)))
def test_moments_match_scipy(axis, pol):
    x = _data()
    X = sd.asarray(x)
    for bias in (True, False):
        _close(S.skew(X, axis=axis, bias=bias, nan_policy=pol), st.skew(x, axis=axis, bias=bias, nan_policy=pol))
        for fisher in (True, False):
            _close(S.kurtosis(X, axis=axis, fisher=fisher, bias=bias, nan_policy=pol),
                   st.kurtosis(x, axis=axis, fisher=fisher, bias=bias, nan_policy=pol))
    for corr in (0, 1, 2):
        _close(S.sem(X, axis=axis, correction=corr, nan_policy=pol), st.sem(x, axis=axis, ddof=corr, nan_policy=pol))
    for order in (0, 1, 2, 3, 4, 5):
        _close(S.moment(X, order=order, axis=axis, nan_policy=pol), st.moment(x, order=order, axis=axis, nan_policy=pol))
    _close(S.moment(X, order=[2, 3], center=1.5, axis=axis, nan_policy=pol),
           st.moment(x, order=[2, 3], center=1.5, axis=axis, nan_policy=pol))
    _close(S.iqr(X, axis=axis, nan_policy=pol), st.iqr(x, axis=axis, nan_policy=pol))
    _close(S.iqr(X, axis=axis, q=(0.1, 0.6), scale="normal", method="nearest", nan_policy=pol),
           st.iqr(x, axis=axis, rng=(10, 60), scale="normal", interpolation="nearest", nan_policy=pol))
    _close(S.entropy(X, axis=axis, nan_policy=pol), st.entropy(x, axis=axis, nan_policy=pol))
    _close(S.differential_entropy(X, axis=axis, nan_policy=pol),
           st.differential_entropy(x, axis=axis, nan_policy=pol))


@pytest.mark.parametrize(("axis", "pol"), list(itertools.product([None, 0, 1, -1], POLICIES)))
def test_zscore_rankdata_mode_match_scipy(axis, pol):
    x = _data((40, 6))
    X = sd.asarray(x)
    for corr in (0, 1):
        _close(S.zscore(X, axis=axis, correction=corr, nan_policy=pol),
               st.zscore(x, axis=axis, ddof=corr, nan_policy=pol))
    xi = np.round(x * 2)
    Xi = sd.asarray(xi)
    for method in ("average", "min", "max", "dense", "ordinal"):
        _close(S.rankdata(Xi, method=method, axis=axis, nan_policy=pol),
               st.rankdata(xi, method=method, axis=axis, nan_policy=pol))
    a, b = S.mode(Xi, axis=axis, nan_policy=pol), st.mode(xi, axis=axis, nan_policy=pol)
    _close(a.mode, b.mode)
    _close(a.count, b.count)


def test_describe_matches_scipy():
    x = _data((50, 3))
    for pol in POLICIES:
        for axis in (0, 1, None):
            for corr, bias in ((1, True), (0, False)):
                a = S.describe(sd.asarray(x), axis=axis, correction=corr, bias=bias, nan_policy=pol)
                b = st.describe(x, axis=axis, ddof=corr, bias=bias, nan_policy=pol)
                _close(a.nobs, b.nobs)
                _close(a.minmax[0], b.minmax[0])
                _close(a.minmax[1], b.minmax[1])
                for f in ("mean", "variance", "skewness", "kurtosis"):
                    _close(getattr(a, f), getattr(b, f))
    d = S.describe(sd.asarray([1.0, 2.0, 3.0, 10.0]))
    assert isinstance(d, S.DescribeResult) and d.nobs == 4 and d.minmax == (1.0, 10.0)
    assert isinstance(d.mean, float)


@pytest.mark.parametrize("method", ["vasicek", "van es", "ebrahimi", "correa", "auto"])
@pytest.mark.parametrize("n", [5, 9, 50, 2000])
def test_differential_entropy_methods(method, n):
    y = np.random.default_rng(n).normal(size=n)
    for m in (None, 1, 2):
        if m is not None and not 2 * m < n:
            continue
        _close(S.differential_entropy(sd.asarray(y), method=method, window_length=m, base=2.0),
               st.differential_entropy(y, method=method, window_length=m, base=2.0), rtol=1e-12)


def _exact_moments(xs):
    q = [Fraction(v) for v in xs]
    n = len(q)
    m = sum(q) / n
    mk = [sum((v - m) ** k for v in q) / n for k in (2, 3, 4)]
    return m, mk


def test_ill_conditioned_data_against_exact_values():
    # mean 1e8, spread 1: subtracting the mean rounded to a double costs 1e-8 of the skewness
    # (scipy: 1.2e-8); the mean as hi + lo and compensated sums keep every digit
    rng = np.random.default_rng(3)
    x = 1e8 + rng.gamma(2.0, size=2000)
    m, (m2, m3, m4) = _exact_moments(x)
    X = sd.asarray(x)
    sk = float(m3) / float(m2) ** 1.5
    ku = float(m4) / float(m2) ** 2 - 3.0
    assert abs(S.skew(X) - sk) <= 1e-14 * abs(sk)
    assert abs(S.kurtosis(X) - ku) <= 1e-14 * abs(ku)
    assert abs(S.moment(X, order=2) - float(m2)) <= 1e-14 * float(m2)
    assert abs(S.moment(X, order=3) - float(m3)) <= 1e-14 * abs(float(m3))
    assert abs(S.moment(X, order=4) - float(m4)) <= 1e-14 * float(m4)
    assert abs(S.describe(X).mean - float(m)) <= 1e-16 * float(m)
    z = np.asarray(S.zscore(X))
    exact = np.array([float((Fraction(v) - m)) for v in x[:5]]) / math.sqrt(float(m2))
    assert np.allclose(z[:5], exact, rtol=1e-14, atol=0)


def test_edge_cases():
    assert math.isnan(S.skew(sd.asarray([1.0, 1.0, 1.0])))
    assert S.kurtosis(sd.asarray([1.0, 2.0])) == -2.0
    assert math.isnan(S.sem(sd.asarray([1.0])))
    assert np.isnan(np.asarray(S.zscore(sd.asarray([1.0, 1.0, 1.0])))).all()
    assert S.moment(sd.asarray([1.0, 2.0, 3.0, 4.0]), order=0) == 1.0
    assert S.moment(sd.asarray([1.0, 2.0, 3.0, 4.0]), order=1) == 0.0
    assert S.moment(sd.asarray([1.0, 2.0, 3.0, 4.0]), order=3, center=0.0) == 25.0
    m = S.mode(sd.asarray([1.0, 1.0, np.nan, np.nan, np.nan, 2.0]))
    assert math.isnan(m.mode) and m.count == 3
    m = S.mode(sd.asarray([1.0, 1.0, np.nan, np.nan, np.nan, 2.0]), nan_policy="omit")
    assert m == (1.0, 2)
    m = S.mode(sd.asarray([3.0, 1.0, 3.0, 1.0]))  # ties: the smallest
    assert m == (1.0, 2)
    assert math.isnan(S.mode(sd.zeros((0,))).mode) and S.mode(sd.zeros((0,))).count == 0
    assert math.isnan(S.entropy(sd.asarray([0.0, 0.0])))
    assert S.entropy(sd.asarray([1.0, 1.0]), base=2) == 1.0
    assert math.isnan(S.differential_entropy(sd.asarray([1.0, 2.0, 3.0])))  # 2m >= n
    with pytest.raises(ValueError, match="empty"):
        S.describe(sd.zeros((0,)))
    # Python numbers and sequences
    assert S.skew([1.0, 2.0, 10.0]) == pytest.approx(st.skew([1.0, 2.0, 10.0]), rel=1e-14)
    assert isinstance(S.sem(sd.asarray([1.0, 2.0, 4.0])), float)


def test_keepdims_and_shapes():
    x = sd.asarray(_data((6, 5, 4), nan=False))
    for f in (S.skew, S.kurtosis, S.sem, S.iqr, S.differential_entropy, S.entropy):
        assert f(x, axis=(0, 2), keepdims=True).shape == (1, 5, 1)
        assert f(x, axis=1).shape == (6, 4)
    assert S.moment(x, order=2, axis=0, keepdims=True).shape == (1, 5, 4)
    assert S.mode(x, axis=2, keepdims=True).mode.shape == (6, 5, 1)
    assert S.moment(x, order=[1, 2, 3], axis=0).shape == (3, 5, 4)
    assert S.zscore(x, axis=1).shape == (6, 5, 4)
    assert S.rankdata(x, axis=1).shape == (6, 5, 4)
    assert S.rankdata(x).shape == (120,)  # axis=None ranks the flattened array (scipy)
    assert isinstance(S.skew(x), float)
    # int and bool inputs are float64 inside
    assert S.skew(sd.asarray([1, 2, 10])) == pytest.approx(st.skew([1, 2, 10]), rel=1e-14)


def test_nan_policy_raise_and_errors():
    x = sd.asarray([1.0, float("nan"), 3.0])
    for f in (S.skew, S.kurtosis, S.sem, S.zscore, S.rankdata, S.iqr, S.entropy, S.differential_entropy):
        with pytest.raises(ValueError, match="NaN"):
            f(x, nan_policy="raise")
    with pytest.raises(ValueError, match="NaN"):
        S.mode(x, nan_policy="raise")
    with pytest.raises(ValueError, match="NaN"):
        S.moment(x, order=2, nan_policy="raise")
    with pytest.raises(ValueError, match="nan_policy"):
        S.skew(x, nan_policy="ignore")
    with pytest.raises(TypeError):
        S.moment(x)  # order is required
    with pytest.raises(ValueError, match="integral"):
        S.moment(x, order=1.5)
    with pytest.raises(ValueError, match="method"):
        S.rankdata(x, method="first")
    with pytest.raises(ValueError, match="scale"):
        S.iqr(x, scale="raw")
    with pytest.raises(ValueError, match="q"):
        S.iqr(x, q=(25, 75))
    with pytest.raises(ValueError, match="base"):
        S.entropy(x, base=-1.0)
    with pytest.raises(ValueError, match="window_length"):
        S.differential_entropy(x, window_length=0)
    with pytest.raises(ValueError, match="axis"):
        S.skew(sd.zeros((3, 3)), axis=2)
    with pytest.raises(ValueError, match="repeated"):
        S.skew(sd.zeros((3, 3)), axis=(0, 0))
    with pytest.raises(TypeError, match="complex"):
        S.skew(sd.asarray([1 + 1j, 2 + 0j]))
    with pytest.raises(TypeError):
        S.skew(sd.zeros((3,)), 0)  # options are keyword-only


def test_entropy_relative_and_broadcast():
    p = np.array([[0.2, 0.5], [0.3, 0.1], [0.5, 0.4]])
    q = np.array([0.1, 0.2, 0.7])[:, None]
    _close(S.entropy(sd.asarray(p), sd.asarray(q), axis=0), st.entropy(p, q, axis=0))
    _close(S.entropy(sd.asarray(p), sd.asarray(q), axis=0, base=2), st.entropy(p, q, axis=0, base=2))


def test_long_lane_sorts_on_any_thread_count():
    """A single long lane sorts on the pool (a samplesort from 65536 elements): the ranks, the
    mode and the entropy estimate have the same bits on 1, 3 and 10 threads, and the ranks are
    numpy's."""
    rng = np.random.default_rng(12)
    before = sd.get_num_threads()
    try:
        for n in (65535, 65536, 200003):
            for x in (rng.normal(size=n), rng.integers(0, 40, n).astype(float), np.zeros(n),
                      np.where(rng.random(n) < 0.3, -0.0, 0.0), np.sort(rng.normal(size=n))[::-1].copy()):
                want = None
                for th in (1, 3, 10):
                    sd.set_num_threads(th)
                    X = sd.asarray(x)
                    got = (np.asarray(S.rankdata(X)), np.asarray(S.rankdata(X, method="ordinal")), tuple(S.mode(X)),
                           S.differential_entropy(X))
                    if want is None:
                        want = got
                        np.testing.assert_array_equal(got[0], st.rankdata(x))
                        np.testing.assert_array_equal(got[1], st.rankdata(x, method="ordinal"))
                        m = st.mode(x)
                        assert got[2][0] == m.mode and got[2][1] == m.count
                    else:
                        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
                        assert got[2] == want[2] and math.copysign(1.0, got[2][0]) == math.copysign(1.0, want[2][0])
                        assert got[3] == want[3] or (math.isnan(got[3]) and math.isnan(want[3]))
    finally:
        sd.set_num_threads(before)


def test_lanes_side_by_side_match_lanes_alone():
    """Reducing over an axis that is not the last (the lanes side by side in memory) gives the
    bits of each lane reduced alone, NaNs and all, on any number of threads."""
    rng = np.random.default_rng(13)
    x = rng.normal(size=(20011, 131)) * 3.0 + 1e6
    x[rng.integers(0, 20011, 40), rng.integers(0, 131, 40)] = np.nan
    x[:, 7] = np.inf
    X = sd.asarray(x)
    before = sd.get_num_threads()
    try:
        for pol in ("propagate", "omit"):
            sd.set_num_threads(1)
            alone = [np.array([f(sd.asarray(np.ascontiguousarray(x[:, j])), nan_policy=pol) for j in range(131)])
                     for f in (S.sem, S.skew, S.kurtosis)]
            for th in (1, 3, 10):
                sd.set_num_threads(th)
                for f, want in zip((S.sem, S.skew, S.kurtosis), alone):
                    np.testing.assert_array_equal(np.asarray(f(X, axis=0, nan_policy=pol)), want)
                z = np.asarray(S.zscore(X, axis=0, nan_policy=pol))
                for j in (0, 7, 64, 130):
                    np.testing.assert_array_equal(z[:, j], np.asarray(S.zscore(sd.asarray(np.ascontiguousarray(x[:, j])),
                                                                                nan_policy=pol)))
    finally:
        sd.set_num_threads(before)


@pytest.mark.parametrize("c", [1e-300, 1e-200, 1e-160, 1e160, 1e200, 1e300])
def test_moments_of_tiny_and_huge_data(c):
    # the powers of x - mean overflow or underflow at these scales (sem of 1e-200-sized data
    # was 0, the skewness of 1e160-sized data NaN): the lane is rescaled by a power of two
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 6)) + 0.3
    a, b = sd.asarray(x), sd.asarray(x * c)
    for axis in (None, 0, 1):
        for pol in ("propagate", "omit"):
            kw = dict(axis=axis, nan_policy=pol)
            for f in (S.skew, S.kurtosis, S.zscore, lambda v, **k: S.skew(v, bias=False, **k),
                      lambda v, **k: S.describe(v, **k).kurtosis, lambda v, **k: S.describe(v, **k).skewness):
                np.testing.assert_allclose(np.asarray(f(b, **kw)), np.asarray(f(a, **kw)), rtol=1e-12)
            np.testing.assert_allclose(np.asarray(S.sem(b, **kw)), np.asarray(S.sem(a, **kw)) * c, rtol=1e-12)
            for order in (2, 3):
                with np.errstate(over="ignore", under="ignore"):
                    want = np.asarray(S.moment(a, order=order, **kw)) * c**order if abs(math.log10(c)) * order < 300 else None
                if want is not None:
                    np.testing.assert_allclose(np.asarray(S.moment(b, order=order, **kw)), want, rtol=1e-12)


def test_moments_of_extreme_data():
    # subnormal data: integers times 5e-324
    x = sd.asarray(np.array([1.0, 2.0, 0.0, 4.0, 3.0, 6.0]) * 5e-324)
    assert S.skew(x) == pytest.approx(st.skew([1.0, 2.0, 0.0, 4.0, 3.0, 6.0]), rel=1e-13)
    assert S.kurtosis(x) == pytest.approx(st.kurtosis([1.0, 2.0, 0.0, 4.0, 3.0, 6.0]), rel=1e-13)
    # the largest doubles, mixed scales: the range itself overflows
    z = np.asarray(S.zscore(sd.asarray([1e-300, 1.0, 1e300, -1e300, 0.0, 1e150, -1e150])))
    assert z[2] == pytest.approx(math.sqrt(3.5), rel=1e-12) and z[3] == pytest.approx(-math.sqrt(3.5), rel=1e-12)
    assert S.kurtosis(sd.asarray([1.7e308, -1.7e308, 1.7e308, -1.7e308])) == pytest.approx(-2.0, abs=1e-12)
    # a constant lane stays NaN at any scale
    for v in (5e-324, 1e-300, 1e300):
        assert math.isnan(S.skew(sd.asarray([v, v, v]))) and np.isnan(np.asarray(S.zscore(sd.asarray([v, v, v])))).all()


@pytest.mark.parametrize("method", ["vasicek", "van es", "ebrahimi", "auto"])
def test_differential_entropy_of_tied_values_is_minus_inf(method):
    # a zero spacing is log 0 (scipy: -inf); the compensated sum made it NaN
    for x in ([1.0] * 10, [1.0] * 20 + list(range(30)), [0.0] * 100 + list(np.linspace(-3, 3, 1900))):
        assert float(sd.stats.differential_entropy(sd.asarray(x), method=method)) == -math.inf
    # Correa divides 0 by 0 on a constant window: NaN, as in scipy
    assert math.isnan(float(sd.stats.differential_entropy(sd.ones((10,)), method="correa")))
