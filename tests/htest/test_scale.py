"""Statistics that do not change with the scale of the data (docs/api/htest.md §3, §5, §6;
docs/design/htest.md H160, H153): the moment tests, shapiro and anderson, pearsonr,
pointbiserialr and epps_singleton_2samp give the same values from 1e-310 to 1e300 times a
sample, a sample summed in its unit of 2^e where its fourth powers would leave the doubles. The
Jarque-Bera kernel is sd.regression's too: the same bits on a fit's residuals at any scale."""

import math

import pytest

import stardust as sd

from .._bits import bits_equal

h = sd.htest
pytestmark = pytest.mark.usefixtures("restore_threads")

XS = [0.3, -1.2, 0.8, 2.1, -0.4, 0.6, 1.9, -0.5, 0.1, 3.0, 0.7, -0.9]
SCALES = [1e77, 1e-90, 1e150, 1e200, 1e300, 1e-160, 1e-310]
TESTS = {
    "kurtosistest": lambda x: h.kurtosistest(x).statistic,
    "jarque_bera": lambda x: h.jarque_bera(x).statistic,
    "skewtest": lambda x: h.skewtest(x).statistic,
    "normaltest": lambda x: h.normaltest(x).statistic,
    "shapiro": lambda x: h.shapiro(x).statistic,
    "anderson": lambda x: h.anderson(x).statistic,
}


@pytest.mark.parametrize("name", list(TESTS))
@pytest.mark.parametrize("s", SCALES)
def test_moment_tests_at_any_scale(name, s):
    f = TESTS[name]
    base = f(XS)
    # the scaled values are themselves rounded (and below 2^-1022 lose digits): a few ulps, more
    # for the subnormal sample
    tol = 1e-11 if s < 1e-300 else 1e-14
    assert f([v * s for v in XS]) == pytest.approx(base, rel=tol)


def test_powers_of_two_keep_the_bits():
    """A power of two scales every double-double step exactly: the unit changes no bit."""
    for e in (-1000, -500, -130, 130, 500, 1000):
        x = [math.ldexp(v, e) for v in XS]
        for f in TESTS.values():
            assert bits_equal(f(x), f(XS))


def test_ordinary_samples_unchanged():
    # well inside 2^-120 .. 2^120: summed as given (the same bits as before the unit)
    assert h.jarque_bera(XS).statistic == 0.7225276864803648
    assert h.kurtosistest(XS).statistic == -0.08187644068726543


@pytest.mark.parametrize("s", [1.0, 1e80, 1e-85, 1e-100])
def test_regression_jarque_bera_same_bits(s):
    X = sd.asarray([[1.0, float(i)] for i in range(12)])
    r = sd.regression.ols(sd.asarray([v * s for v in XS]), X)
    a, b = h.jarque_bera(r.resid), r.jarque_bera()
    assert not math.isnan(a.statistic)
    assert bits_equal(a.statistic, b.statistic) and bits_equal(a.pvalue, b.pvalue)


@pytest.mark.parametrize("s", [1e-310, 1e300])
def test_correlations_at_extreme_scales(s):
    y = [v * v for v in XS]
    g = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    x2 = XS[:6] + [v * 1.1 for v in XS]
    y2 = [v * 0.7 for v in XS]
    tol = 1e-11 if s < 1e-300 else 1e-14
    assert h.pearsonr([v * s for v in XS], [v * s for v in y]).statistic == pytest.approx(
        h.pearsonr(XS, y).statistic, rel=tol
    )
    assert h.pointbiserialr(g, [v * s for v in XS]).statistic == pytest.approx(
        h.pointbiserialr(g, XS).statistic, rel=tol
    )
    assert h.epps_singleton_2samp([v * s for v in x2], [v * s for v in y2]).statistic == pytest.approx(
        h.epps_singleton_2samp(x2, y2).statistic, rel=tol
    )


def test_epps_constant_samples():
    """Each sample constant (H153): the g vectors are constant, their covariance 0 but for the
    means' rounding; NaN, not W = 1e65 (scipy 4.9e32)."""
    for x, y in (([2.0] * 9, [3.0] * 8), ([2e300] * 9, [3e300] * 8), ([-1.0] * 6, [5.0] * 5)):
        r = h.epps_singleton_2samp(x, y)
        assert math.isnan(r.statistic) and math.isnan(r.pvalue) and math.isnan(r.df)
    assert h.epps_singleton_2samp([2.0] * 9, [3.0] * 7 + [3.5]).statistic == pytest.approx(39.3983, rel=1e-5)


def test_scaled_lanes_in_a_batch():
    rows = [[v * s for v in XS] for s in (1.0, 1e200, 1e-200)]
    sd.set_num_threads(1)
    one = [h.jarque_bera(r).statistic for r in rows] + [h.shapiro(r).statistic for r in rows]
    sd.set_num_threads(10)
    b = sd.asarray(rows)
    both = sd.concat([h.jarque_bera(b, axis=1).statistic, h.shapiro(b, axis=1).statistic])
    assert bits_equal(both, sd.asarray(one))
