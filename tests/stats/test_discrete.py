"""The discrete distributions' behaviour (docs/api/stats.md §2, docs/design/stats.md T25):
pmf/logpmf, the cdfs at non-integers, the quantile conventions, modes with ties, degenerate
parameters, and agreement with scipy in the body of each distribution."""

import inspect
import math

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

S = sd.stats

# (class, parameters, scipy frozen distribution, support)
CASES = [
    (S.Binomial, {"n": 20, "p": 0.3}, st.binom(20, 0.3), (0.0, 20.0)),
    (S.Poisson, {"mu": 4.5}, st.poisson(4.5), (0.0, math.inf)),
    (S.Geometric, {"p": 0.2}, st.geom(0.2), (1.0, math.inf)),
    (S.NegativeBinomial, {"n": 2.5, "p": 0.3}, st.nbinom(2.5, 0.3), (0.0, math.inf)),
    (S.Hypergeometric, {"total": 50, "successes": 12, "draws": 20}, st.hypergeom(50, 12, 20), (0.0, 12.0)),
    (S.Bernoulli, {"p": 0.3}, st.bernoulli(0.3), (0.0, 1.0)),
    (S.DiscreteUniform, {"low": 2, "high": 9}, st.randint(2, 9), (2.0, 8.0)),
    (S.BetaBinomial, {"n": 20, "a": 2.5, "b": 1.7}, st.betabinom(20, 2.5, 1.7), (0.0, 20.0)),
    (S.LogSeries, {"p": 0.7}, st.logser(0.7), (1.0, math.inf)),
    (S.Zipf, {"a": 6.5}, st.zipf(6.5), (1.0, math.inf)),
]
IDS = [c[0].__name__ for c in CASES]
INPUT = ["pmf", "logpmf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf"]
PARAM = ["mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy"]


def test_exports():
    for c in CASES:
        assert issubclass(c[0], S.DiscreteDistribution)
        assert c[0].__name__ in S.__all__
    assert not hasattr(S.Binomial(n=3, p=0.5), "pdf")  # pmf in place of pdf (T9)


@pytest.mark.parametrize("cls, params, _, __", CASES, ids=IDS)
def test_keyword_only_and_required(cls, params, _, __):
    sig = inspect.signature(cls)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())
    assert all(p.default is inspect.Parameter.empty for p in sig.parameters.values())
    with pytest.raises(TypeError):
        cls(*params.values())


@pytest.mark.parametrize("cls, params, _, __", CASES, ids=IDS)
def test_scalars_are_floats(cls, params, _, __):
    X = cls(**params)
    for m in INPUT:
        assert type(getattr(X, m)(0.3 if not m.startswith("ilog") else -0.3)) is float, m
    for m in PARAM:
        assert type(getattr(X, m)()) is float, m


@pytest.mark.parametrize("cls, params, ref, support", CASES, ids=IDS)
def test_support_and_quantile_conventions(cls, params, ref, support):
    X = cls(**params)
    lo, hi = support
    assert X.support() == support
    assert (X.icdf(0.0), X.icdf(1.0), X.iccdf(0.0), X.iccdf(1.0)) == (lo, hi, hi, lo)
    assert (X.ilogcdf(0.0), X.ilogcdf(-math.inf), X.ilogccdf(0.0), X.ilogccdf(-math.inf)) == (hi, lo, lo, hi)
    assert math.isnan(X.icdf(1.5)) and math.isnan(X.ilogcdf(0.5)) and math.isnan(X.pmf(math.nan))
    assert X.pmf(lo - 1.0) == 0.0 and X.logpmf(lo - 1.0) == -math.inf
    assert (X.cdf(lo - 0.5), X.ccdf(lo - 0.5)) == (0.0, 1.0)
    # a non-integer: pmf 0, the cdfs those at floor(x)
    k = lo + 1.0
    assert X.pmf(k + 0.5) == 0.0 and X.logpmf(k + 0.5) == -math.inf
    assert X.cdf(k + 0.5) == X.cdf(k) and X.ccdf(k + 0.5) == X.ccdf(k) and X.logcdf(k + 0.7) == X.logcdf(k)
    # icdf(p) is the smallest k with cdf(k) >= p; iccdf(p) the smallest with ccdf(k) <= p
    for p in (0.05, 0.3, 0.5, 0.77, 0.99):
        q = X.icdf(p)
        assert X.cdf(q) >= p and (q == lo or X.cdf(q - 1.0) < p)
        r = X.iccdf(p)
        assert X.ccdf(r) <= p and (r == lo or X.ccdf(r - 1.0) > p)
    assert X.median() == X.icdf(0.5)  # T25


@pytest.mark.parametrize("cls, params, ref, support", CASES, ids=IDS)
def test_body_agrees_with_scipy(cls, params, ref, support):
    X = cls(**params)
    p = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    k = ref.ppf(p)
    for ours, theirs in (("pmf", "pmf"), ("logpmf", "logpmf"), ("cdf", "cdf"), ("logcdf", "logcdf"), ("ccdf", "sf"),
                         ("logccdf", "logsf")):
        np.testing.assert_allclose(np.asarray(getattr(X, ours)(sd.asarray(k))), getattr(ref, theirs)(k), rtol=1e-12,
                                   err_msg=ours)
    assert np.asarray(X.icdf(sd.asarray(p))).tolist() == ref.ppf(p).tolist()
    assert np.asarray(X.iccdf(sd.asarray(p))).tolist() == ref.isf(p).tolist()
    assert np.asarray(X.ilogcdf(sd.asarray(np.log(p)))).tolist() == ref.ppf(p).tolist()
    m, v, s, kk = (float(t) for t in ref.stats(moments="mvsk"))
    np.testing.assert_allclose([X.mean(), X.variance(), X.skewness(), X.kurtosis()], [m, v, s, kk], rtol=1e-11)
    assert X.median() == ref.median()
    if cls is not S.Zipf:  # scipy's zipf entropy is a sum that does not converge (7 digits)
        assert math.isclose(X.entropy(), ref.entropy(), rel_tol=1e-12)


def test_modes_with_ties_are_nan():
    # two equal modes (T25 with T21): (n + 1) p an integer, an integer mu, a uniform
    assert math.isnan(S.Binomial(n=9, p=0.5).mode())
    assert S.Binomial(n=10, p=0.5).mode() == 5.0
    assert math.isnan(S.Poisson(mu=3.0).mode()) and S.Poisson(mu=3.5).mode() == 3.0 and S.Poisson(mu=0.5).mode() == 0.0
    assert math.isnan(S.NegativeBinomial(n=3.0, p=0.5).mode())  # (n - 1) q / p = 2
    assert math.isnan(S.DiscreteUniform(low=0, high=6).mode()) and S.DiscreteUniform(low=4, high=5).mode() == 4.0
    assert math.isnan(S.Bernoulli(p=0.5).mode()) and S.Bernoulli(p=0.6).mode() == 1.0
    assert math.isnan(S.BetaBinomial(n=10, a=1.0, b=1.0).mode())  # uniform on 0..10
    assert math.isnan(S.Hypergeometric(total=10, successes=5, draws=3).mode())  # (n + 1)(K + 1) / (N + 2) = 2
    assert S.Geometric(p=0.3).mode() == 1.0 and S.Zipf(a=2.0).mode() == 1.0 and S.LogSeries(p=0.9).mode() == 1.0


def test_degenerate_parameters():
    # point masses: p = 0 or 1, n = 0, mu = 0
    for X, at in ((S.Binomial(n=7, p=0.0), 0.0), (S.Binomial(n=7, p=1.0), 7.0), (S.Binomial(n=0, p=0.4), 0.0),
                  (S.Poisson(mu=0.0), 0.0), (S.Bernoulli(p=1.0), 1.0), (S.Geometric(p=1.0), 1.0),
                  (S.NegativeBinomial(n=2.0, p=1.0), 0.0)):
        assert X.pmf(at) == 1.0 and X.logpmf(at) == 0.0 and X.cdf(at) == 1.0 and X.ccdf(at - 1.0) == 1.0
        assert X.mean() == at and X.variance() == 0.0 and X.mode() == at and X.entropy() == 0.0
        assert math.isnan(X.skewness()) and math.isnan(X.kurtosis())  # undefined for a point mass


def test_parameters_outside_the_domain():
    for X in (S.Binomial(n=5.5, p=0.3), S.Binomial(n=5, p=1.5), S.Poisson(mu=-1.0), S.Geometric(p=0.0),
              S.NegativeBinomial(n=0.0, p=0.5), S.Hypergeometric(total=10, successes=11, draws=3),
              S.DiscreteUniform(low=3, high=3), S.DiscreteUniform(low=0.5, high=3), S.BetaBinomial(n=4, a=0.0, b=1.0),
              S.LogSeries(p=1.0), S.Zipf(a=1.0)):
        assert math.isnan(X.pmf(1.0)) and math.isnan(X.cdf(1.0)) and math.isnan(X.mean()) and math.isnan(X.icdf(0.5))


def test_infinite_quantiles_and_moments():
    # a quantile beyond float64 is inf; moments that diverge are inf (the rule of §2)
    assert S.Zipf(a=1.001).iccdf(1e-10) == math.inf
    assert S.Zipf(a=1.5).mean() == math.inf and math.isnan(S.Zipf(a=2.0).variance())
    assert S.Zipf(a=2.5).variance() == math.inf and S.Zipf(a=3.5).skewness() == math.inf
    assert S.Zipf(a=4.5).kurtosis() == math.inf and math.isnan(S.Zipf(a=3.0).kurtosis())
