"""The distribution classes' behaviour (docs/api/stats.md §1-§2, docs/design/stats.md
T6-T9, T18-T23): construction, attributes, broadcasting, out=, the domain and support
conventions, and agreement with scipy in the body of each distribution."""

import inspect
import math

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

S = sd.stats

# (class, example parameters, the scipy frozen distribution, support)
CASES = [
    (S.Normal, {"mu": 1.5, "sigma": 2.0}, st.norm(1.5, 2.0), (-math.inf, math.inf)),
    (S.StudentT, {"df": 3.5}, st.t(3.5), (-math.inf, math.inf)),
    (S.ChiSquared, {"df": 4.5}, st.chi2(4.5), (0.0, math.inf)),
    (S.F, {"dfn": 3.0, "dfd": 11.0}, st.f(3.0, 11.0), (0.0, math.inf)),
    (S.Gamma, {"a": 2.5, "scale": 3.0}, st.gamma(2.5, scale=3.0), (0.0, math.inf)),
    (S.Beta, {"a": 2.5, "b": 0.7}, st.beta(2.5, 0.7), (0.0, 1.0)),
    (S.Exponential, {"scale": 2.0}, st.expon(scale=2.0), (0.0, math.inf)),
    (S.LogNormal, {"mu": 0.5, "sigma": 0.8}, st.lognorm(0.8, scale=math.exp(0.5)), (0.0, math.inf)),
    (S.Uniform, {"a": -1.0, "b": 3.0}, st.uniform(-1.0, 4.0), (-1.0, 3.0)),
    (S.Cauchy, {"mu": 0.5, "sigma": 2.0}, st.cauchy(0.5, 2.0), (-math.inf, math.inf)),
    (S.Laplace, {"mu": 0.5, "sigma": 2.0}, st.laplace(0.5, 2.0), (-math.inf, math.inf)),
    (S.Weibull, {"c": 1.7, "scale": 2.0}, st.weibull_min(1.7, scale=2.0), (0.0, math.inf)),
]
IDS = [c[0].__name__ for c in CASES]

INPUT_METHODS = ["pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf"]
PARAM_METHODS = ["mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy"]


def test_exports():
    names = {"Normal", "StudentT", "ChiSquared", "F", "Gamma", "Beta", "Exponential", "LogNormal", "Uniform", "Cauchy",
             "Laplace", "Weibull"}
    assert names <= set(S.__all__)
    for n in names:
        assert issubclass(getattr(S, n), S.ContinuousDistribution)


# ---- construction (T19)


@pytest.mark.parametrize("cls, params, _, __", CASES, ids=IDS)
def test_keyword_only(cls, params, _, __):
    sig = inspect.signature(cls)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())
    assert tuple(sig.parameters) == cls._params
    with pytest.raises(TypeError):
        cls(*params.values())


def test_defaults_are_location_zero_scale_one():
    assert S.Normal().cdf(0.0) == 0.5
    assert (S.Normal().mu, S.Normal().sigma) == (0.0, 1.0)
    assert S.Exponential().scale == 1.0
    assert (S.Cauchy().mu, S.Laplace().sigma, S.LogNormal().sigma) == (0.0, 1.0, 1.0)
    assert S.Gamma(a=2.0).scale == 1.0 and S.Weibull(c=2.0).scale == 1.0
    for cls in (S.Gamma, S.Beta, S.Uniform, S.StudentT, S.ChiSquared, S.F, S.Weibull):
        with pytest.raises(TypeError):
            cls()  # shape parameters (and Uniform's a, b) are required


def test_attributes_are_read_only_and_as_given():
    mu = sd.asarray([0.0, 1.0])
    X = S.Normal(mu=mu, sigma=2)
    assert X.mu is mu and X.sigma == 2 and isinstance(X.sigma, int)
    with pytest.raises(AttributeError):
        X.mu = 3.0
    with pytest.raises(AttributeError):
        X.other = 1  # no instance dictionary
    assert repr(S.Gamma(a=2.0, scale=3.0)) == "Gamma(a=2.0, scale=3.0)"


def test_parameter_types():
    with pytest.raises(TypeError, match="'mu' must be a real number or an sd.ndarray, not str"):
        S.Normal(mu="1")
    with pytest.raises(TypeError, match="'sigma'"):
        S.Normal(sigma=[1.0, 2.0])
    with pytest.raises(TypeError, match="complex"):
        S.Normal(mu=sd.asarray([1j]))
    with pytest.raises(ValueError):
        S.Normal(mu=sd.zeros(2), sigma=sd.ones(3))  # shapes that cannot broadcast fail at construction
    assert S.Normal(mu=True, sigma=sd.asarray([1, 2])).cdf(1.0).tolist() == [0.5, 0.5]  # bool and int64 are real


# ---- results: scalars, broadcasting, out= (T22)


@pytest.mark.parametrize("cls, params, _, __", CASES, ids=IDS)
def test_scalar_results_are_python_floats(cls, params, _, __):
    X = cls(**params)
    for m in INPUT_METHODS:
        assert type(getattr(X, m)(0.3 if not m.startswith("ilog") else -0.3)) is float, m
    for m in PARAM_METHODS:
        assert type(getattr(X, m)()) is float, m
    lo, hi = X.support()
    assert type(lo) is float and type(hi) is float


def test_broadcasting():
    X = S.Gamma(a=sd.asarray([[1.0], [2.0]]), scale=sd.asarray([1.0, 2.0, 3.0]))
    assert X.cdf(1.0).shape == (2, 3)
    assert X.mean().shape == (2, 3)
    assert X.mean().tolist() == [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]
    assert S.Normal().cdf(sd.zeros((4, 1, 2))).shape == (4, 1, 2)
    lo, hi = X.support()
    assert lo.shape == (2, 3) and hi.tolist() == [[math.inf] * 3] * 2


def test_out():
    X = S.Normal(mu=sd.asarray([0.0, 1.0, 2.0]))
    out = sd.empty(3)
    r = X.cdf(1.0, out=out)
    assert r is out
    assert out.tolist() == [X.cdf(1.0)[i] for i in range(3)]
    with pytest.raises(TypeError):
        X.mean(out=out)  # parameter-only methods take no out=


# ---- domain and support (T8, T21)


@pytest.mark.parametrize("cls, params, _, __", CASES, ids=IDS)
def test_parameters_outside_the_domain_give_nan(cls, params, _, __):
    bad = dict(params)
    key = next(k for k in cls._params if k not in ("mu", "a") or cls is S.Uniform)
    bad[key] = -1.0 if cls is not S.Uniform else bad["a"] - 10.0  # Uniform: b < a
    if cls is S.Uniform:
        bad = {"a": 3.0, "b": -1.0}
    X = cls(**bad)
    for m in INPUT_METHODS:
        assert math.isnan(getattr(X, m)(0.3 if not m.startswith("ilog") else -0.3)), m
    for m in PARAM_METHODS:
        assert math.isnan(getattr(X, m)()), m
    assert all(math.isnan(v) for v in X.support())
    assert math.isnan(cls(**{**params, cls._params[-1]: math.nan}).cdf(0.5))


@pytest.mark.parametrize("cls, params, _, support", CASES, ids=IDS)
def test_support_conventions(cls, params, _, support):
    X = cls(**params)
    lo, hi = support
    assert X.support() == support
    assert (X.icdf(0.0), X.icdf(1.0)) == (lo, hi)
    assert (X.iccdf(0.0), X.iccdf(1.0)) == (hi, lo)
    assert (X.ilogcdf(0.0), X.ilogcdf(-math.inf)) == (hi, lo)
    assert (X.ilogccdf(0.0), X.ilogccdf(-math.inf)) == (lo, hi)
    for m in ("icdf", "iccdf"):
        assert math.isnan(getattr(X, m)(-0.1)) and math.isnan(getattr(X, m)(1.1)) and math.isnan(getattr(X, m)(math.nan))
    for m in ("ilogcdf", "ilogccdf"):
        assert math.isnan(getattr(X, m)(0.1)) and math.isnan(getattr(X, m)(math.nan))
    assert (X.cdf(-math.inf), X.cdf(math.inf)) == (0.0, 1.0)
    assert (X.ccdf(-math.inf), X.ccdf(math.inf)) == (1.0, 0.0)
    assert (X.logcdf(-math.inf), X.logcdf(math.inf)) == (-math.inf, 0.0)
    assert (X.pdf(-math.inf), X.pdf(math.inf), X.logpdf(math.inf)) == (0.0, 0.0, -math.inf)
    for m in INPUT_METHODS:
        assert math.isnan(getattr(X, m)(math.nan)), m
    if lo > -math.inf:
        below = lo - 1.0
        assert (X.pdf(below), X.logpdf(below), X.cdf(below), X.ccdf(below)) == (0.0, -math.inf, 0.0, 1.0)
    if hi < math.inf:
        above = hi + 1.0
        assert (X.pdf(above), X.cdf(above), X.ccdf(above), X.logccdf(above)) == (0.0, 1.0, 0.0, -math.inf)


def test_mode_conventions():
    assert math.isnan(S.Uniform(a=0.0, b=1.0).mode())  # every point (T21)
    assert S.Gamma(a=0.5).mode() == 0.0 and S.Gamma(a=1.0).mode() == 0.0  # one point, at the end
    assert S.Weibull(c=0.5).mode() == 0.0 and S.Exponential().mode() == 0.0
    assert math.isnan(S.Beta(a=0.5, b=0.5).mode()) and math.isnan(S.Beta(a=1.0, b=1.0).mode())
    assert (S.Beta(a=0.5, b=2.0).mode(), S.Beta(a=2.0, b=0.5).mode()) == (0.0, 1.0)
    assert (S.Beta(a=1.0, b=0.5).mode(), S.Beta(a=0.5, b=1.0).mode()) == (1.0, 0.0)
    assert S.F(dfn=2.0, dfd=5.0).mode() == 0.0 and S.ChiSquared(df=1.5).mode() == 0.0


def test_moment_conventions():
    # NaN where undefined, inf where infinite (docs/design/stats.md §2.3)
    t = S.StudentT
    assert math.isnan(t(df=1.0).mean()) and t(df=1.5).mean() == 0.0
    assert t(df=2.0).variance() == math.inf and math.isnan(t(df=1.0).variance())
    assert math.isnan(t(df=3.0).skewness()) and t(df=4.0).kurtosis() == math.inf and math.isnan(t(df=2.0).kurtosis())
    f = S.F
    assert f(dfn=3.0, dfd=2.0).mean() == math.inf and f(dfn=3.0, dfd=4.0).variance() == math.inf
    assert f(dfn=3.0, dfd=6.0).skewness() == math.inf and f(dfn=3.0, dfd=8.0).kurtosis() == math.inf
    for m in ("mean", "variance", "standard_deviation", "skewness", "kurtosis"):
        assert math.isnan(getattr(S.Cauchy(), m)())
    assert S.Normal().kurtosis() == 0.0  # excess kurtosis (T23)


def test_density_at_the_ends():
    assert S.Gamma(a=0.5).pdf(0.0) == math.inf and S.Gamma(a=1.0, scale=2.0).pdf(0.0) == 0.5
    assert S.Gamma(a=2.0).pdf(0.0) == 0.0 and S.Gamma(a=0.5).logpdf(0.0) == math.inf
    assert S.Beta(a=1.0, b=3.0).pdf(0.0) == 3.0 and S.Beta(a=3.0, b=1.0).pdf(1.0) == 3.0
    assert S.F(dfn=2.0, dfd=7.0).pdf(0.0) == 1.0 and S.F(dfn=1.0, dfd=7.0).pdf(0.0) == math.inf
    assert S.Weibull(c=1.0, scale=4.0).pdf(0.0) == 0.25
    assert S.Uniform(a=0.0, b=2.0).pdf(0.0) == 0.5 and S.Uniform(a=0.0, b=2.0).pdf(2.0) == 0.5


# ---- agreement with scipy in the body (the accuracy itself: test_reference.py)


@pytest.mark.parametrize("cls, params, ref, support", CASES, ids=IDS)
def test_body_agrees_with_scipy(cls, params, ref, support):
    X = cls(**params)
    p = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    x = ref.ppf(p)
    got = {m: np.asarray(getattr(X, m)(sd.asarray(x))) for m in ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf")}
    np.testing.assert_allclose(got["pdf"], ref.pdf(x), rtol=1e-12)
    np.testing.assert_allclose(got["logpdf"], ref.logpdf(x), rtol=1e-12)
    np.testing.assert_allclose(got["cdf"], ref.cdf(x), rtol=1e-12)
    np.testing.assert_allclose(got["logcdf"], ref.logcdf(x), rtol=1e-12)
    np.testing.assert_allclose(got["ccdf"], ref.sf(x), rtol=1e-12)
    np.testing.assert_allclose(got["logccdf"], ref.logsf(x), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(X.icdf(sd.asarray(p))), x, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(X.iccdf(sd.asarray(p))), ref.isf(p), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(X.ilogcdf(sd.asarray(np.log(p)))), x, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(X.ilogccdf(sd.asarray(np.log(p)))), ref.isf(p), rtol=1e-12)
    m, v, s, k = ref.stats(moments="mvsk")
    if cls is not S.Cauchy:
        np.testing.assert_allclose([X.mean(), X.variance(), X.skewness(), X.kurtosis()], [m, v, s, k], rtol=1e-12)
        assert math.isclose(X.standard_deviation(), math.sqrt(v), rel_tol=1e-12)
    assert math.isclose(X.median(), ref.median(), rel_tol=1e-12, abs_tol=1e-300)
    assert math.isclose(X.entropy(), ref.entropy(), rel_tol=1e-12)


# ---- determinism (docs/api/stats.md §1)


def test_same_bits_on_any_thread_count():
    rng = np.random.default_rng(1)
    x = sd.asarray(rng.gamma(2.0, size=200_000))
    a = sd.asarray(rng.uniform(0.5, 5.0, size=200_000))
    p = sd.asarray(rng.uniform(size=200_000))
    results = []
    for n in (1, 3, 10):
        sd.set_num_threads(n)
        try:
            X = S.Gamma(a=a, scale=1.5)
            results.append([np.asarray(X.cdf(x)).tobytes(), np.asarray(X.icdf(p)).tobytes(),
                            np.asarray(S.StudentT(df=a).logpdf(x)).tobytes()])
        finally:
            sd.set_num_threads(10)
    assert results[0] == results[1] == results[2]
