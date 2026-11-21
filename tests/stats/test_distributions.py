"""The distribution classes' behaviour (docs/api/stats.md §1-§2, docs/design/stats.md
T6-T9, T18-T23): construction, attributes, broadcasting, out=, the domain and support
conventions, and agreement with scipy in the body of each distribution."""

import inspect
import math
import time

import numpy as np
import pytest
import scipy.stats as st
from flint import arb, ctx

import stardust as sd
from tests._bits import bits_equal_any_nan

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
    # T6
    (S.Logistic, {"mu": 0.5, "sigma": 2.0}, st.logistic(0.5, 2.0), (-math.inf, math.inf)),
    (S.Gumbel, {"mu": 0.5, "sigma": 2.0}, st.gumbel_r(0.5, 2.0), (-math.inf, math.inf)),
    (S.GEV, {"c": 0.25, "mu": 1.0, "sigma": 2.0}, st.genextreme(0.25, 1.0, 2.0), (-math.inf, 9.0)),
    (S.Pareto, {"b": 5.5, "scale": 1.5}, st.pareto(5.5, scale=1.5), (1.5, math.inf)),
    (S.Rayleigh, {"scale": 2.0}, st.rayleigh(scale=2.0), (0.0, math.inf)),
    (S.InverseGamma, {"a": 5.5, "scale": 2.0}, st.invgamma(5.5, scale=2.0), (0.0, math.inf)),
    (S.NoncentralChiSquared, {"df": 3.5, "nc": 2.0}, st.ncx2(3.5, 2.0), (0.0, math.inf)),
    (S.NoncentralT, {"df": 9.0, "nc": 1.5}, st.nct(9.0, 1.5), (-math.inf, math.inf)),
    (S.NoncentralF, {"dfn": 4.0, "dfd": 13.0, "nc": 3.0}, st.ncf(4.0, 13.0, 3.0), (0.0, math.inf)),
]
NONCENTRAL = (S.NoncentralChiSquared, S.NoncentralT, S.NoncentralF)
IDS = [c[0].__name__ for c in CASES]

INPUT_METHODS = ["pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf"]
PARAM_METHODS = ["mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy"]


def test_exports():
    names = {
        "Normal",
        "StudentT",
        "ChiSquared",
        "F",
        "Gamma",
        "Beta",
        "Exponential",
        "LogNormal",
        "Uniform",
        "Cauchy",
        "Laplace",
        "Weibull",
        "Logistic",
        "Gumbel",
        "GEV",
        "Pareto",
        "Rayleigh",
        "InverseGamma",
        "NoncentralChiSquared",
        "NoncentralT",
        "NoncentralF",
    }
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
    # an array parameter is kept as a read-only copy (D-5): the caller's array may change
    assert X.mu is not mu and X.mu.tolist() == [0.0, 1.0] and X.sigma == 2 and isinstance(X.sigma, int)
    mu[0] = 5.0
    assert X.mu.tolist() == [0.0, 1.0] and float(X.cdf(0.0)[0]) == 0.5
    with pytest.raises(ValueError, match="read-only"):
        X.mu[0] = 1.0
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
    key = "sigma" if cls is S.GEV else next(k for k in cls._params if k not in ("mu", "a") or cls is S.Uniform)
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
        assert (
            math.isnan(getattr(X, m)(-0.1)) and math.isnan(getattr(X, m)(1.1)) and math.isnan(getattr(X, m)(math.nan))
        )
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
    if cls not in NONCENTRAL:  # scipy integrates those numerically (1e-10 off; ours: Arb, test_reference)
        assert math.isclose(X.entropy(), ref.entropy(), rel_tol=1e-12)


def test_t6_conventions():
    # moments by the rule of docs/api/stats.md §2 (T26)
    assert S.Pareto(b=1.0).mean() == math.inf and math.isnan(S.Pareto(b=1.0).variance())
    assert S.Pareto(b=1.5).variance() == math.inf and S.Pareto(b=2.5).skewness() == math.inf
    assert S.Pareto(b=3.5).kurtosis() == math.inf and math.isnan(S.Pareto(b=2.0).kurtosis())
    assert S.InverseGamma(a=1.0).mean() == math.inf and S.InverseGamma(a=1.5).variance() == math.inf
    assert S.GEV(c=-1.0).mean() == math.inf and S.GEV(c=-0.6).variance() == math.inf
    assert math.isnan(S.GEV(c=-1.2).variance()) and S.GEV(c=-0.4).skewness() == math.inf
    assert S.GEV(c=-0.3).kurtosis() == math.inf and S.GEV(c=0.0).kurtosis() == 2.4
    assert math.isnan(S.NoncentralT(df=1.0, nc=1.0).mean()) and S.NoncentralT(df=2.0, nc=1.0).variance() == math.inf
    assert math.isnan(S.NoncentralT(df=3.0, nc=1.0).skewness()) and S.NoncentralT(df=4.0, nc=1.0).kurtosis() == math.inf
    assert S.NoncentralF(dfn=3.0, dfd=2.0, nc=1.0).mean() == math.inf
    assert S.NoncentralF(dfn=3.0, dfd=8.0, nc=1.0).kurtosis() == math.inf
    # GEV at c = 0 is the Gumbel; nc = 0 the central distributions, to the bit
    x = sd.asarray([-3.0, 0.2, 5.0])
    for m in ("pdf", "cdf", "logccdf"):
        assert (
            np.asarray(getattr(S.GEV(c=0.0, mu=1.0, sigma=2.0), m)(x)).tolist()
            == np.asarray(getattr(S.Gumbel(mu=1.0, sigma=2.0), m)(x)).tolist()
        )
    for nc_cls, cen, kw in (
        (S.NoncentralChiSquared, S.ChiSquared(df=3.0), {"df": 3.0}),
        (S.NoncentralT, S.StudentT(df=3.0), {"df": 3.0}),
        (S.NoncentralF, S.F(dfn=3.0, dfd=7.0), {"dfn": 3.0, "dfd": 7.0}),
    ):
        X = nc_cls(**kw, nc=0.0)
        y = sd.asarray([0.3, 1.7, 9.0])
        for m in ("pdf", "logcdf", "ccdf", "icdf"):
            arg = sd.asarray([0.1, 0.5, 0.9]) if m == "icdf" else y
            assert np.asarray(getattr(X, m)(arg)).tolist() == np.asarray(getattr(cen, m)(arg)).tolist(), (nc_cls, m)
    # modes at the ends (T21) and densities there
    assert S.GEV(c=1.5).mode() == S.GEV(c=1.5).support()[1] and S.GEV(c=0.0, mu=2.0).mode() == 2.0
    assert (
        S.NoncentralChiSquared(df=1.5, nc=3.0).mode() == 0.0 and S.NoncentralF(dfn=1.5, dfd=9.0, nc=3.0).mode() == 0.0
    )
    assert S.NoncentralChiSquared(df=2.0, nc=1.0).mode() == 0.0 and S.NoncentralChiSquared(df=2.0, nc=5.0).mode() > 0.0
    assert (S.GEV(c=0.5).pdf(2.0), S.GEV(c=1.0).pdf(1.0), S.GEV(c=2.0).pdf(0.5)) == (0.0, 1.0, math.inf)
    assert S.Pareto(b=3.0, scale=2.0).pdf(2.0) == 1.5 and S.Rayleigh().pdf(0.0) == 0.0


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
            results.append(
                [
                    np.asarray(X.cdf(x)).tobytes(),
                    np.asarray(X.icdf(p)).tobytes(),
                    np.asarray(S.StudentT(df=a).logpdf(x)).tobytes(),
                ]
            )
        finally:
            sd.set_num_threads(10)
    assert results[0] == results[1] == results[2]


def test_noncentral_bulk_paths_same_bits_and_continuous():
    """The noncentral three's paths of the bulk (series from j = 0, Newton inverses; their
    per-thread memory of the parameters' constants): the same bits for a scalar parameter
    and an array of it, element by element, and on any thread count; and no jump where a
    path hands over to the precise definition (no step in the logs beyond their rounding)."""
    rng = np.random.default_rng(7)
    n = 3000
    u = rng.uniform(size=n)
    cases = [
        (S.NoncentralChiSquared, {"df": 3.5, "nc": 2.0}),
        (S.NoncentralT, {"df": 9.0, "nc": 1.5}),
        (S.NoncentralT, {"df": 4.0, "nc": -0.75}),
        (S.NoncentralF, {"dfn": 4.0, "dfd": 13.0, "nc": 3.0}),
    ]
    methods = ("pdf", "logpdf", "cdf", "ccdf", "logcdf", "logccdf", "icdf", "iccdf")
    for cls, params in cases:
        X = cls(**params)
        x = np.asarray(X.icdf(sd.asarray(u)))
        # interleaved with other parameters, so the remembered constants change between elements
        other = {k: np.where(np.arange(n) % 2 == 0, v, v * 1.25 + 0.5) for k, v in params.items()}
        Y = cls(**{k: sd.asarray(v) for k, v in other.items()})
        even = np.arange(n) % 2 == 0
        runs = []
        for threads in (1, 3, 10):
            sd.set_num_threads(threads)
            try:
                run = []
                for m in methods:
                    arg = u if m in ("icdf", "iccdf") else x
                    whole = np.asarray(getattr(X, m)(sd.asarray(arg)))
                    mixed = np.asarray(getattr(Y, m)(sd.asarray(arg)))
                    assert whole[even].tobytes() == mixed[even].tobytes(), (cls.__name__, m, threads)
                    for i in (0, 1, n // 2, n - 1):
                        assert getattr(X, m)(float(arg[i])) == whole[i], (cls.__name__, m, i)
                    run.append(whole.tobytes() + mixed.tobytes())
                runs.append(run)
            finally:
                sd.set_num_threads(10)
        assert runs[0] == runs[1] == runs[2], cls.__name__

    # across the hand-over: a walk along x through the bulk's limits (x / 2 = 340 for the
    # chi-square's sums, the series' term count for the densities)
    def smooth(v, tol):  # no step: third differences at the values' rounding
        return np.max(np.abs(np.diff(v, 3))) < tol

    X = S.NoncentralChiSquared(df=5.0, nc=40.0)
    for lo, hi in ((679.0, 681.0), (2990.0, 3010.0), (5990.0, 6010.0)):
        x = np.linspace(lo, hi, 401)
        for m in ("logpdf", "logccdf"):
            v = np.asarray(getattr(X, m)(sd.asarray(x)))
            assert np.all(np.diff(v) < 0) and smooth(v, 2e-11), (m, lo)
    T = S.NoncentralT(df=7.0, nc=3.0)
    t = np.linspace(-6.0, 0.5, 2001)  # the density's series ends where its two families cancel
    v = np.asarray(T.logpdf(sd.asarray(t)))
    assert np.all(np.diff(v) > 0) and smooth(v, 1e-6)  # (the function's own third difference: 3e-7)
    # the inverses return to their probability on both sides of Newton's range
    for cls, params in cases:
        X = cls(**params)
        for p in (1e-250, 1e-30, 1e-3, 0.3, 0.5, 0.9):
            assert X.cdf(X.icdf(p)) == pytest.approx(p, rel=1e-12), (cls.__name__, p)
            assert X.ccdf(X.iccdf(p)) == pytest.approx(p, rel=1e-12), (cls.__name__, p)


# ---- the chunk kernels of the closed-form distributions (docs/design/stats.md §3.4)

CLOSED_FORM = [
    (S.Normal, {"mu": 1.5, "sigma": 2.0}),
    (S.LogNormal, {"mu": 0.5, "sigma": 0.8}),
    (S.Exponential, {"scale": 2.0}),
    (S.Uniform, {"a": -1.0, "b": 3.0}),
    (S.Laplace, {"mu": 0.5, "sigma": 2.0}),
    (S.Cauchy, {"mu": 0.5, "sigma": 2.0}),
    (S.Weibull, {"c": 1.7, "scale": 2.0}),
    (S.Logistic, {"mu": 0.5, "sigma": 2.0}),
    (S.Gumbel, {"mu": 0.5, "sigma": 2.0}),
    (S.GEV, {"c": 0.25, "mu": 1.0, "sigma": 2.0}),
    (S.GEV, {"c": -0.4, "mu": 0.0, "sigma": 1.0}),
    (S.Pareto, {"b": 3.0, "scale": 1.5}),
    (S.Rayleigh, {"scale": 2.0}),
]


def test_closed_form_kernels_same_bits_for_any_layout_and_thread_count():
    """A chunk kernel's result for an element depends on nothing but the element: the same
    bits for a scalar parameter and an array of it (interleaved with other parameters, so
    constants computed once per chunk and the gathered second passes are exercised), for a
    strided input, for one element alone, and on 1, 3 and 10 threads. The inputs cover the
    bulk, both tails and the special values, so every grade of a kernel and its hand-over to
    the scalar definition take part."""
    rng = np.random.default_rng(11)
    n = 60_000
    u = rng.uniform(size=n)
    u[::7] = 10.0 ** rng.uniform(-300, 0, size=u[::7].size)  # the lower tail
    u[3::7] = 1.0 - 10.0 ** rng.uniform(-16, 0, size=u[3::7].size)  # the upper tail
    u[:6] = [0.0, 1.0, 0.5, 0.25, 0.75, 5e-324]
    even = np.arange(n) % 2 == 0
    for cls, params in CLOSED_FORM:
        X = cls(**params)
        with np.errstate(all="ignore"):
            x = np.asarray(X.icdf(sd.asarray(u)))
            x[1::11] = np.nextafter(x[1::11], np.inf)
            x[6:12] = [0.0, -1.0, math.inf, -math.inf, 1e300, -1e300]
        other = {k: np.where(even, v, v * 1.25 + 0.5) for k, v in params.items()}
        Y = cls(**{k: sd.asarray(v) for k, v in other.items()})
        runs = []
        for threads in (1, 3, 10):
            sd.set_num_threads(threads)
            try:
                run = []
                for m in INPUT_METHODS:
                    with np.errstate(all="ignore"):
                        arg = u if m in ("icdf", "iccdf") else (np.log(u) if m.startswith("ilog") else x)
                        whole = np.asarray(getattr(X, m)(sd.asarray(arg)))
                        mixed = np.asarray(getattr(Y, m)(sd.asarray(arg)))
                        strided = np.asarray(getattr(X, m)(sd.asarray(np.repeat(arg, 2))[::2]))
                    assert whole[even].tobytes() == mixed[even].tobytes(), (cls.__name__, m, threads)
                    assert whole.tobytes() == strided.tobytes(), (cls.__name__, m, threads)
                    if threads == 1:
                        for i in list(range(12)) + list(range(12, n, 997)):
                            one = getattr(X, m)(float(arg[i]))
                            assert one == whole[i] or (math.isnan(one) and math.isnan(whole[i])), (cls.__name__, m, i)
                    run.append(whole.tobytes() + mixed.tobytes())
                runs.append(run)
            finally:
                sd.set_num_threads(10)
        assert runs[0] == runs[1] == runs[2], cls.__name__


def test_closed_form_kernels_agree_with_scipy_in_the_bulk_and_keep_special_values():
    """The fast paths against an independent implementation (1e-12: scipy's own accuracy; the
    Arb references of test_reference.py are the contract), and the values outside them."""
    rng = np.random.default_rng(12)
    p = rng.uniform(0.01, 0.99, size=5000)
    refs = {
        S.Normal: st.norm(1.5, 2.0),
        S.LogNormal: st.lognorm(0.8, scale=math.exp(0.5)),
        S.Exponential: st.expon(scale=2.0),
        S.Uniform: st.uniform(-1.0, 4.0),
        S.Laplace: st.laplace(0.5, 2.0),
        S.Cauchy: st.cauchy(0.5, 2.0),
        S.Weibull: st.weibull_min(1.7, scale=2.0),
        S.Logistic: st.logistic(0.5, 2.0),
        S.Gumbel: st.gumbel_r(0.5, 2.0),
        S.Pareto: st.pareto(3.0, scale=1.5),
        S.Rayleigh: st.rayleigh(scale=2.0),
    }
    names = {
        "pdf": "pdf",
        "logpdf": "logpdf",
        "cdf": "cdf",
        "logcdf": "logcdf",
        "ccdf": "sf",
        "logccdf": "logsf",
        "icdf": "ppf",
        "iccdf": "isf",
    }
    for cls, params in CLOSED_FORM:
        if cls not in refs or (cls is S.GEV):
            continue
        X, ref = cls(**params), refs[cls]
        x = ref.ppf(p)
        for m, sm in names.items():
            arg = p if m in ("icdf", "iccdf") else x
            got = np.asarray(getattr(X, m)(sd.asarray(arg)))
            want = getattr(ref, sm)(arg)
            # a quantile near 0 (mu + sigma z cancelling) is only as good as its condition number
            # (and a log density crossing 0 as its absolute error: the mixed bound, T20)
            atol = 1e-11 if m in ("icdf", "iccdf") else (1e-12 if m == "logpdf" else 1e-300)
            assert np.allclose(got, want, rtol=1e-12, atol=atol), (cls.__name__, m)
        # NaN in, NaN out; invalid parameters NaN; outside the support 0 / -inf
        bad = cls(**{k: (-1.0 if k in ("sigma", "scale", "c", "b") else v) for k, v in params.items()})
        for m in INPUT_METHODS:
            ok = -0.3 if m.startswith("ilog") else 0.3
            v = np.asarray(getattr(X, m)(sd.asarray([math.nan, ok])))
            assert math.isnan(v[0]) and not math.isnan(v[1]), (cls.__name__, m)
            if cls is not S.Uniform:
                assert np.all(np.isnan(np.asarray(getattr(bad, m)(sd.asarray([ok, 2.0 * ok]))))), (cls.__name__, m)
        lo, hi = (float(v) for v in X.support())
        if math.isfinite(lo):
            below = sd.asarray([lo - 1.0, lo - 1e-9])
            assert np.asarray(X.pdf(below)).tolist() == [0.0, 0.0] and np.asarray(X.cdf(below)).tolist() == [0.0, 0.0]
            assert np.asarray(X.logpdf(below)).tolist() == [-math.inf] * 2 and np.asarray(X.ccdf(below)).tolist() == [
                1.0,
                1.0,
            ]
        ends = sd.asarray([-math.inf, math.inf])
        assert np.asarray(X.cdf(ends)).tolist() == [0.0, 1.0] and np.asarray(X.ccdf(ends)).tolist() == [1.0, 0.0]
        assert np.asarray(X.pdf(ends)).tolist() == [0.0, 0.0]
        q = np.asarray(X.icdf(sd.asarray([0.0, 1.0, -0.1, 1.1])))
        assert q[0] == lo and q[1] == hi and math.isnan(q[2]) and math.isnan(q[3]), cls.__name__


def test_exponential_where_x_over_scale_overflows():
    # x / scale = inf for finite x: the density is 0, not NaN (the quotient's double-double
    # remainder is inf - inf), in the scalar form and the kernel alike
    X = S.Exponential(scale=1e-3)
    assert X.pdf(1e306) == 0.0 and X.logpdf(1e306) == -math.inf
    assert X.cdf(1e306) == 1.0 and X.ccdf(1e306) == 0.0 and X.logccdf(1e306) == -math.inf
    x = sd.asarray([1e306, 1.0, 1e308])
    assert np.asarray(X.pdf(x)).tolist()[::2] == [0.0, 0.0]
    assert np.asarray(X.logpdf(x)).tolist()[::2] == [-math.inf, -math.inf]
    # one logarithm of the scale, whether it is a broadcast scalar or an array
    x = sd.asarray(np.random.default_rng(3).exponential(size=1000))
    for s in (0.3, 2.0, 1e-200, 7e150):
        a = np.asarray(S.Exponential(scale=s).logpdf(x))
        b = np.asarray(S.Exponential(scale=sd.asarray(np.full(1000, s))).logpdf(x))
        assert a.tobytes() == b.tobytes(), s


# ---- the gamma and beta families' fast paths (stats/dist/gamma.hpp, beta.hpp, t_f.hpp) ----


GB_CASES = [
    (S.Gamma, {"a": 2.5, "scale": 3.0}),
    (S.Gamma, {"a": 7.3, "scale": 0.4}),
    (S.Gamma, {"a": 300.0, "scale": 1.0}),
    (S.ChiSquared, {"df": 4.5}),
    (S.ChiSquared, {"df": 7.0}),
    (S.ChiSquared, {"df": 1.0}),
    (S.InverseGamma, {"a": 4.5, "scale": 2.0}),
    (S.Beta, {"a": 2.5, "b": 0.7}),
    (S.Beta, {"a": 2.0, "b": 3.0}),
    (S.Beta, {"a": 40.0, "b": 7.5}),
    (S.StudentT, {"df": 3.5}),
    (S.StudentT, {"df": 400.0}),
    (S.F, {"dfn": 3.0, "dfd": 11.0}),
    (S.F, {"dfn": 30.0, "dfd": 5.0}),
]
GB_METHODS = ("pdf", "logpdf", "cdf", "ccdf", "logcdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf")


def test_gamma_beta_families_same_bits_for_any_layout_and_thread_count():
    """A scalar parameter and an array of it, an array and its elements one by one, and 1, 3
    and 10 threads give the same bits: through the densities' two tiers and their fallback
    (bulk, tails, ends of the support), the finite sums and Boost's kernels of the cdfs, and
    the inverses' per-thread constants."""
    rng = np.random.default_rng(11)
    n = 4000
    u = rng.uniform(size=n)
    for cls, params in GB_CASES:
        X = cls(**params)
        x = np.asarray(X.icdf(sd.asarray(u)))
        x = x.copy()
        lo, hi = (float(v) for v in X.support())
        # the bulk, both far tails and the ends
        x[::7] = np.asarray(X.icdf(sd.asarray(rng.uniform(size=x[::7].size) * 1e-12)))
        x[3::7] = np.asarray(X.iccdf(sd.asarray(rng.uniform(size=x[3::7].size) * 1e-12)))
        x[:6] = [lo, hi, 0.0, 1.0, -1.0, math.inf]
        other = {k: np.where(np.arange(n) % 2 == 0, v, v * 1.25 + 0.5) for k, v in params.items()}
        Y = cls(**{k: sd.asarray(v) for k, v in other.items()})
        even = np.arange(n) % 2 == 0
        runs = []
        for threads in (1, 3, 10):
            sd.set_num_threads(threads)
            try:
                run = []
                for m in GB_METHODS:
                    arg = u if m in ("icdf", "iccdf") else (-rng.exponential(size=n) if m.startswith("ilog") else x)
                    if threads > 1:
                        arg = runs[0][GB_METHODS.index(m)][1]
                    whole = np.asarray(getattr(X, m)(sd.asarray(arg)))
                    mixed = np.asarray(getattr(Y, m)(sd.asarray(arg)))
                    assert whole[even].tobytes() == mixed[even].tobytes(), (cls.__name__, params, m, threads)
                    for i in (0, 1, 2, 3, 4, 5, 7, 10, n // 2, n - 1):
                        one = getattr(X, m)(float(arg[i]))
                        assert np.array([one]).tobytes() == whole[i : i + 1].tobytes(), (cls.__name__, params, m, i)
                    run.append((whole.tobytes() + mixed.tobytes(), arg))
                runs.append(run)
            finally:
                sd.set_num_threads(10)
        for k, m in enumerate(GB_METHODS):
            assert runs[0][k][0] == runs[1][k][0] == runs[2][k][0], (cls.__name__, params, m)


def test_density_kernels_agree_with_closed_forms_and_scipy():
    x = np.array([1e-300, 1e-8, 0.03, 0.4, 1.0, 2.5, 9.0, 40.0, 300.0])
    # exponential and half-normal-square special cases, to the last bits
    np.testing.assert_allclose(
        np.asarray(S.Gamma(a=1.0, scale=2.0).pdf(sd.asarray(x))), np.exp(-x / 2.0) / 2.0, rtol=3e-15
    )
    np.testing.assert_allclose(
        np.asarray(S.ChiSquared(df=2.0).logpdf(sd.asarray(x))), -x / 2.0 - math.log(2.0), rtol=3e-15, atol=3e-15
    )
    xb = np.array([1e-300, 1e-9, 0.01, 0.3, 0.5, 0.9, 1.0 - 1e-12])
    np.testing.assert_allclose(
        np.asarray(S.Beta(a=2.0, b=3.0).pdf(sd.asarray(xb))), 12.0 * xb * (1.0 - xb) ** 2, rtol=3e-15
    )
    t = np.array([0.0, 1e-200, 0.3, -2.0, 15.0, 1e6])
    np.testing.assert_allclose(
        np.asarray(S.StudentT(df=1.0).pdf(sd.asarray(t))), 1.0 / (math.pi * (1.0 + t * t)), rtol=3e-15
    )
    # large shapes (the second tier and the double-double definition) against scipy
    for X, ref, pts in (
        (S.Gamma(a=300.0, scale=1.0), st.gamma(300.0), np.array([150.0, 280.0, 300.0, 330.0, 500.0])),
        (S.Beta(a=40.0, b=7.5), st.beta(40.0, 7.5), np.array([0.3, 0.7, 0.85, 0.95, 0.999])),
        (S.F(dfn=30.0, dfd=5.0), st.f(30.0, 5.0), np.array([0.05, 0.5, 1.0, 3.0, 80.0])),
        (S.InverseGamma(a=25.0, scale=2.0), st.invgamma(25.0, scale=2.0), np.array([0.03, 0.07, 0.09, 0.2, 2.0])),
    ):
        np.testing.assert_allclose(np.asarray(X.pdf(sd.asarray(pts))), ref.pdf(pts), rtol=1e-11)
        np.testing.assert_allclose(np.asarray(X.logpdf(sd.asarray(pts))), ref.logpdf(pts), rtol=1e-11)


def test_student_t_density_at_zero():
    # ilogb(0) overflowed an int difference for df >= 4: the density at t = 0 was NaN
    for df in (0.5, 3.0, 4.0, 13.5, 1e6):
        X = S.StudentT(df=df)
        assert X.pdf(0.0) == pytest.approx(st.t(df).pdf(0.0), rel=1e-13)
        assert X.logpdf(0.0) == pytest.approx(st.t(df).logpdf(0.0), rel=1e-13)
        assert X.logcdf(0.0) == pytest.approx(-math.log(2.0), rel=1e-15)
        assert np.asarray(X.pdf(sd.asarray([0.0, -0.0]))).tolist() == [X.pdf(0.0)] * 2


def test_gamma_logpdf_tiny_x_huge_shape_is_finite():
    # a / y overflowed in the large-shape form's first-order term
    v = S.Gamma(a=4548098621.0228386, scale=1.0).logpdf(7.2992975175516095e-300)
    assert v == pytest.approx(-3229267019528.685, rel=1e-14)


def test_gamma_finite_sums():
    """Integer and half-integer shapes: the upper tail by its finite sum where the root is at
    or above a, against the closed forms."""
    x = np.array([1.0, 1.5, 3.0, 10.0, 80.0, 600.0])
    np.testing.assert_allclose(np.asarray(S.Gamma(a=1.0, scale=1.0).ccdf(sd.asarray(x))), np.exp(-x), rtol=2e-15)
    np.testing.assert_allclose(
        np.asarray(S.Gamma(a=3.0, scale=1.0).ccdf(sd.asarray(x[2:]))),
        np.exp(-x[2:]) * (1.0 + x[2:] + 0.5 * x[2:] ** 2),
        rtol=2e-15,
    )
    # erfc(sqrt x): squares, so the reference's own square root is exact
    z = np.array([1.0, 1.25, 1.5, 3.0, 9.0, 25.0])
    erfc = np.array([math.erfc(v) for v in z])
    np.testing.assert_allclose(np.asarray(S.Gamma(a=0.5, scale=1.0).ccdf(sd.asarray(z * z))), erfc, rtol=2e-15)
    np.testing.assert_allclose(np.asarray(S.ChiSquared(df=1.0).ccdf(sd.asarray(2.0 * z * z))), erfc, rtol=2e-15)
    np.testing.assert_allclose(np.asarray(S.ChiSquared(df=2.0).logccdf(sd.asarray(2.0 * x))), -x, rtol=2e-15)
    # a scale that does not divide exactly, and the inverse gamma's mirrored tails
    for a in (0.5, 1.0, 2.5, 7.0, 29.5, 30.0):
        y = a + np.array([0.0, 0.3, 2.0, 11.0, 150.0])
        for s in (1.0, 3.0, 0.37):
            X, ref = S.Gamma(a=a, scale=s), st.gamma(a, scale=s)
            np.testing.assert_allclose(np.asarray(X.ccdf(sd.asarray(y * s))), ref.sf(y * s), rtol=2e-12)
            np.testing.assert_allclose(np.asarray(X.cdf(sd.asarray(y * s))), ref.cdf(y * s), rtol=2e-12)
            np.testing.assert_allclose(np.asarray(X.logcdf(sd.asarray(y * s))), np.log1p(-ref.sf(y * s)), rtol=2e-12)
            assert abs(X.ccdf(X.iccdf(0.2)) - 0.2) <= 1e-13 * 0.2
            assert abs(X.ccdf(X.iccdf(1e-9)) - 1e-9) <= 1e-13 * 1e-9
            Y, refi = S.InverseGamma(a=a, scale=s), st.invgamma(a, scale=s)
            np.testing.assert_allclose(np.asarray(Y.cdf(sd.asarray(s / y))), refi.cdf(s / y), rtol=2e-12)
            np.testing.assert_allclose(np.asarray(Y.logcdf(sd.asarray(s / y))), refi.logcdf(s / y), rtol=2e-12)


def test_beta_family_small_shapes():
    """Shapes up to 6: the cdfs through betainc and betaincc, the inverses by Halley's method on
    them; continuous across the switch at 6 and into the deep tails."""
    xb = np.array([1e-100, 1e-9, 0.01, 0.3, 0.5, 0.9])
    X = S.Beta(a=2.0, b=3.0)
    exact = xb * xb * (6.0 - 8.0 * xb + 3.0 * xb * xb)  # I_x(2, 3)
    np.testing.assert_allclose(np.asarray(X.cdf(sd.asarray(xb))), exact, rtol=4e-15)
    np.testing.assert_allclose(np.asarray(X.logcdf(sd.asarray(xb[:5]))), np.log(exact[:5]), rtol=4e-15)
    np.testing.assert_allclose(np.asarray(X.cdf(sd.asarray(xb))) + np.asarray(X.ccdf(sd.asarray(xb))), 1.0, rtol=4e-16)
    for a, b in ((2.5, 0.7), (0.3, 0.4), (6.0, 6.0), (5.9, 1.0)):
        lo, hi = S.Beta(a=a, b=b), S.Beta(a=a * (1.0 + 1e-9) + 1e-9, b=b)
        for x in (1e-250, 1e-30, 0.02, 0.5, 0.97, 1.0 - 1e-10):
            for m in ("cdf", "ccdf", "logcdf", "logccdf"):
                assert getattr(lo, m)(x) == pytest.approx(getattr(hi, m)(x), rel=1e-6), (a, b, x, m)

        def close(got, want):  # (pytest.approx adds an absolute 1e-12: no test for a tiny p)
            return abs(got - want) <= 1e-12 * abs(want)

        for p in (1e-250, 1e-150, 1e-30, 1e-3, 0.3, 0.5, 0.9, 1.0 - 1e-9):
            # (where x is a well-conditioned root: not 0 below the range, not within 1e-3 of 1)
            x = lo.icdf(p)
            if 1e-300 < x < 0.999:
                assert close(lo.cdf(x), p), (a, b, p)
                assert close(lo.logcdf(lo.ilogcdf(math.log(p))), math.log(p)), (a, b, p)
            x = lo.iccdf(p)
            if 1e-300 < x < 0.999:
                assert close(lo.ccdf(x), p), (a, b, p)
        Fd = S.F(dfn=2.0 * a, dfd=2.0 * b)
        for p in (1e-250, 1e-30, 1e-3, 0.3, 0.5, 0.9, 1.0 - 1e-9):
            x = Fd.icdf(p)
            if 1e-300 < x < 1e300:
                assert close(Fd.cdf(x), p), (a, b, p)
            x = Fd.iccdf(p)
            if 1e-300 < x < 1e300:
                assert close(Fd.ccdf(x), p), (a, b, p)
        for x in (1e-200, 0.01, 0.8, 1.0, 1.3, 40.0, 1e30):
            c, cc = Fd.cdf(x), Fd.ccdf(x)
            if 1e-290 < c <= 0.5:
                assert Fd.logcdf(x) == pytest.approx(math.log(c), rel=1e-13), (a, b, x)
                assert Fd.logccdf(x) == pytest.approx(math.log1p(-c), rel=1e-13), (a, b, x)
            if 1e-290 < cc <= 0.5:
                assert Fd.logccdf(x) == pytest.approx(math.log(cc), rel=1e-13), (a, b, x)
                assert Fd.logcdf(x) == pytest.approx(math.log1p(-cc), rel=1e-13), (a, b, x)


# Tiny shapes, tiny and subnormal arguments: the incomplete beta near 1 while x is far from 1,
# where 1 - I formed from I lost up to a digit (Boost's betainc and our own BPSER both, 6e-15 to
# 1e-14), x below the normal doubles (Boost: 1e-4 off;
# the F argument dfn x / (dfn x + dfd) below the doubles altogether), and a result just above
# the subnormals (Boost: 2e-13). (x, a, b, cdf, ccdf, logcdf, logccdf): true values from Arb.
BETA_TINY = [
    (
        2.5577760077856607e-36,
        0.0012901596489084573,
        0.5449374823701244,
        0.8982974005410846,
        0.10170259945891533,
        -0.10725408450482585,
        -2.2857024161862785,
    ),
    (
        0.9999999999999957,
        0.5485563366642973,
        0.0037738739150651214,
        0.12119467925889547,
        0.8788053207411045,
        -2.110357106814741,
        -0.1291918839370924,
    ),
    (
        1.4262239968790307e-188,
        0.00027128344102526805,
        0.014207970097647911,
        0.872627967604178,
        0.12737203239582193,
        -0.1362459679190381,
        -2.060643085878883,
    ),
    (0.0172, 0.04, 0.7, 0.8293472759952524, 0.17065272400474765, -0.18711630204111349, -1.768124641232662),
    (
        1e-323,
        0.0013100809860233137,
        4.633908918194403,
        0.37841975016239504,
        0.621580249837605,
        -0.9717512492912156,
        -0.47549025353900126,
    ),
    (
        4.4748450513e-313,
        0.0007423607384228735,
        3.7530915798634688,
        0.5870724424560937,
        0.4129275575439063,
        -0.5326070554294219,
        -0.8844831068691381,
    ),
    (
        2.088498046372631e-53,
        5.856991998430899,
        5.7812750757189955,
        1.027714430495751e-306,
        1.0,
        -704.5637011190827,
        -1.027714430495751e-306,
    ),
]
# (x, dfn, dfd, cdf, ccdf, logcdf, logccdf)
F_TINY = [
    (
        3.6676358459647984e-299,
        0.0002713527552604415,
        3.6912760173190287,
        0.9099158920823638,
        0.0900841079176362,
        -0.09440311002593013,
        -2.407011512637245,
    ),
    (
        8.848448094366578e22,
        0.6201762912142642,
        0.0037809344523372207,
        0.10846838008096853,
        0.8915316199190315,
        -2.2212965763001384,
        -0.11481437408691324,
    ),
    (
        7.582839393594999e-298,
        2.1494246133543263e-05,
        10.390877909021208,
        0.9925574288272209,
        0.007442571172779163,
        -0.007470405296571315,
        -4.90053890209969,
    ),
    (
        1.3365244111007632e299,
        0.013321077854721223,
        2.5936018609059427e-05,
        0.0108979635112521,
        0.989102036488748,
        -4.519179341032736,
        -0.01095778130736473,
    ),
    (1e-320, 2e-05, 6.0, 0.9925485131681745, 0.007451486831825489, -0.0074793878489457484, -4.899341691761382),
    (1e-320, 10.0, 3.0, 0.0, 1.0, -3677.1204878785297, -0.0),
    (
        1.647e-320,
        5.2389704897305405e-05,
        5.868791808107427e-05,
        0.5182557867505337,
        0.4817442132494663,
        -0.6572863617844793,
        -0.7303419836965642,
    ),
]


def _within(got, want, rel=5e-15):
    return got == want if want == 0.0 else abs(got - want) <= rel * abs(want)


@pytest.mark.parametrize("row", BETA_TINY)
def test_beta_tails_at_tiny_shapes_and_arguments(row):
    x, a, b, *want = row
    X = S.Beta(a=a, b=b)
    got = [float(getattr(X, m)(x)) for m in ("cdf", "ccdf", "logcdf", "logccdf")]
    for m, g, w in zip(("cdf", "ccdf", "logcdf", "logccdf"), got, want):
        assert _within(g, w), (m, row, g)
    assert _within(float(sd.special.betainc(a, b, x)), want[0]), row
    assert _within(float(sd.special.betaincc(a, b, x)), want[1]), row
    # an array of the point gives the same bits as the scalar call
    xs = sd.asarray(np.full(3, x))
    for m, g in zip(("cdf", "ccdf", "logcdf", "logccdf"), got):
        assert np.asarray(getattr(X, m)(xs)).tolist() == [g] * 3, (m, row)


@pytest.mark.parametrize("row", F_TINY)
def test_f_tails_at_tiny_shapes_and_arguments(row):
    x, dfn, dfd, *want = row
    X = S.F(dfn=dfn, dfd=dfd)
    got = [float(getattr(X, m)(x)) for m in ("cdf", "ccdf", "logcdf", "logccdf")]
    for m, g, w in zip(("cdf", "ccdf", "logcdf", "logccdf"), got, want):
        assert _within(g, w), (m, row, g)
    assert _within(float(sd.special.fdtr(dfn, dfd, x)), want[0]), row
    assert _within(float(sd.special.fdtrc(dfn, dfd, x)), want[1]), row
    xs = sd.asarray(np.full(3, x))
    for m, g in zip(("cdf", "ccdf", "logcdf", "logccdf"), got):
        assert np.asarray(getattr(X, m)(xs)).tolist() == [g] * 3, (m, row)


def test_f_where_one_shape_is_past_2_900_review_regression():
    # The F distribution with dfd (or dfn) from 1.7e271 up is its chi-square limit; cdf was 1.0
    # at F(2, 1e308), x = 1 (true: 1 - 1/e) and the log forms NaN. Values: the incomplete beta
    # series in Arb. (x, dfn, dfd, cdf, ccdf, logcdf, logccdf)
    rows = [
        (1.0, 2.0, 1e308, 0.6321205588285577, 0.36787944117144233, -0.4586751453870819, -1.0),
        (1.0, 200.0, 1e308, 0.5132987982791487, 0.48670120172085135, -0.6668971505852894, -0.720104893025474),
        (1e-320, 2e-5, 1e300, 0.992550258327848, 0.007449741672151913, -0.007477629589156729, -4.899575922070794),
        (40.0, 0.5, 1e305, 0.9999979169695914, 2.083030408649431e-06, -2.0830325781602858e-06, -13.081686797304512),
        (1.0, 1e308, 2.0, 0.36787944117144233, 0.6321205588285577, -1.0, -0.4586751453870819),
        (1e305, 1e300, 2e-5, 0.007106867638463068, 0.992893132361537, -4.9466936894190034, -0.007132241713878829),
        (1e-05, 7.0, 1e300, 2.1806304891469145e-17, 1.0, -38.36433253070743, -2.1806304891469148e-17),
        (
            5.776808771711861,
            16.419704325584377,
            7.37960052026974e277,
            0.9999999999995333,
            4.667669003019092e-13,
            -4.667669003020181e-13,
            -28.3929464046775,
        ),
    ]
    for x, dfn, dfd, *want in rows:
        X = S.F(dfn=dfn, dfd=dfd)
        for m, w in zip(("cdf", "ccdf", "logcdf", "logccdf"), want):
            assert _within(float(getattr(X, m)(x)), w), (m, x, dfn, dfd)


def test_noncentral_tiny_noncentrality_and_argument_review_regression():
    # chndtr's far-end walk made NaN (returned as 1.0) for nc below about 1e-155 at a tiny x
    # (tests/special/test_noncentral.py); the distributions' methods on the same points. Arb.
    X = S.NoncentralChiSquared(df=2.0, nc=1e-200)
    assert _within(float(X.cdf(1e-160)), 5e-161) and float(X.ccdf(1e-160)) == 1.0
    assert _within(float(X.logcdf(1e-160)), -369.10676205960726) and _within(float(X.logccdf(1e-160)), -5e-161)
    Y = S.NoncentralF(dfn=2.0, dfd=3.0, nc=1e-200)
    assert _within(float(Y.cdf(1e-160)), 1e-160) and _within(float(Y.logcdf(1e-160)), -368.4136148790473)
    Z = S.NoncentralT(df=3.0, nc=1e-200)
    assert float(Z.cdf(1e-160)) == 0.5 and float(Z.ccdf(1e-160)) == 0.5
    assert _within(float(Z.logcdf(1e-160)), math.log(0.5))
    # arrays give the scalar's bits
    xs = sd.asarray(np.full(3, 1e-160))
    assert np.asarray(X.cdf(xs)).tolist() == [float(X.cdf(1e-160))] * 3
    assert np.asarray(Y.logcdf(xs)).tolist() == [float(Y.logcdf(1e-160))] * 3


def test_noncentral_far_arguments_have_true_values():
    # Inside the guaranteed parameters every finite x has its value. The chi-square's density
    # far out did not return (its series peaks at index sqrt(nc x) / 2: from 1e100 the loop
    # never ended): now its closed form. Arb (tools/stats_ref.py's _ncx2_logpdf).
    for df, nc, x, want in [
        (3.5, 2.0, 3.45e10, -17249737321.501545),
        (3.5, 2.0, 1e30, -4.999999999999986e29),
        (3.5, 2.0, 1e100, -5e99),
        (3.5, 2.0, 1e300, -5e299),
        (1e4, 1e4, 1e10, -4989970479.051994),
        (1e4, 1e4, 1e30, -4.999999999999e29),
        (0.01, 1e-3, 1e100, -5e99),
        (40.0, 900.0, 1e16, -4999997000000177.0),
    ]:
        X = S.NoncentralChiSquared(df=df, nc=nc)
        assert _within(float(X.logpdf(x)), want, 2e-15), (df, nc, x)
        v = float(X.pdf(x))
        assert v == 0.0 and math.copysign(1.0, v) > 0, (df, nc, x)
        assert np.asarray(X.logpdf(sd.asarray([x, x]))).tolist() == [float(X.logpdf(x))] * 2
    # Its tails there: cdf 1, ccdf 0, logcdf -0 (it was NaN), and the upper tail's log, which the
    # sum does not reach (it was -inf, or -36.7 at nc = 1e-10): between log(2 f (1 - 2 / x)) and
    # log(2 f / (1 - eps)), the same double here. Arb.
    for df, nc, x, want in [
        (3.5, 2.0, 3e16, -1.4999999755051024e16),
        (3.5, 2.0, 1e30, -4.999999999999986e29),
        (1e4, 1e4, 1e13, -4999683725446.925),
        (0.01, 1e-3, 1e21, -4.99999999999e20),
        (40.0, 900.0, 1e14, -49999700000219.055),
        (0.01, 1e-10, 1e100, -5e99),
    ]:
        X = S.NoncentralChiSquared(df=df, nc=nc)
        assert _within(float(X.logccdf(x)), want, 2e-15), (df, nc, x)
        assert float(X.ccdf(x)) == 0.0 and abs(float(X.cdf(x)) - 1.0) <= 2.3e-16 and float(X.logcdf(x)) == 0.0
    X = S.NoncentralChiSquared(df=3.5, nc=2.0)
    assert _within(float(X.ilogccdf(-1.4999999755051024e16)), 3e16, 2e-15)  # (it stopped at 2e16, where the sum ends)
    assert _within(float(X.ilogccdf(-1e15)), 2000000126491114.5, 2e-15)
    # The t's far tail on the side away from nc at |t| next to the largest double (it was NaN,
    # and with it every inverse whose search came there). Arb.
    T = S.NoncentralT(df=0.1, nc=1.5)
    assert _within(float(T.cdf(-1e308)), 8.255481062550066e-33) and float(T.ccdf(-1e308)) == 1.0
    assert _within(float(T.logcdf(-1e308)), -73.87443071783558)
    assert _within(float(S.NoncentralT(df=0.1, nc=-1.5).ccdf(1e308)), 8.255481062550066e-33)
    assert _within(float(S.NoncentralT(df=1e5, nc=100.0).logcdf(-1e308)), -70378235.39970005)
    assert float(T.icdf(1e-300)) == -math.inf  # (the root is beyond the doubles)
    # cdf(1e100) was 0 where df/2 + j is not exact (e^-(5e99 + its low part) came out NaN)
    for df, nc in [(0.01, 0.5), (1000.0, 1e4)]:
        X = S.NoncentralChiSquared(df=df, nc=nc)
        assert float(X.cdf(1e100)) == 1.0 and float(X.logcdf(1e100)) == 0.0 and float(X.ccdf(1e100)) == 0.0


def test_noncentral_methods_return_outside_the_guaranteed_region():
    # Every loop is bounded: outside the guaranteed region a method is NaN where its sum
    # cannot be finished (these calls did not return, or took seconds; the chi-square's cdf
    # at nc = 1e15 was 2.28e-9 at every x), and a value only where a form is right there.
    import time

    start = time.perf_counter()
    nan = [
        (
            S.NoncentralChiSquared(df=3.5, nc=1e15),
            ("cdf", 1e15),
            ("ccdf", 1e15),
            ("logcdf", 1e15),
            ("icdf", 0.5),
            ("entropy",),
        ),
        (
            S.NoncentralChiSquared(df=3.5, nc=1e100),
            ("icdf", 0.01),
            ("ilogcdf", -1.0),
            ("median",),
        ),
        (S.NoncentralChiSquared(df=1e100, nc=2.0), ("pdf", 1e300), ("logpdf", 1e300)),
        (S.NoncentralT(df=9.0, nc=1e100), ("cdf", 0.5), ("logccdf", 1e30), ("icdf", 0.5), ("median",)),
        (S.NoncentralT(df=9.0, nc=-1e8), ("cdf", -50.0), ("ccdf", -1e100), ("iccdf", 0.01)),
        (
            S.NoncentralF(dfn=4.0, dfd=13.0, nc=1e100),
            ("pdf", 1.0),
            ("cdf", 1.0),
            ("icdf", 0.5),
            ("mode",),
            ("entropy",),
        ),
        (S.NoncentralF(dfn=4.0, dfd=13.0, nc=1e15), ("pdf", 50.0), ("cdf", 1e16), ("iccdf", 0.5), ("mode",)),
        (S.NoncentralF(dfn=4.0, dfd=1e100, nc=3.0), ("pdf", 1e100), ("logpdf", 1e300)),
        # in the region, an integral that does not fit the range of x (it took 86 s)
        (S.NoncentralF(dfn=0.01, dfd=0.01, nc=1e4), ("entropy",)),
    ]
    for X, *calls in nan:
        for m, *a in calls:
            assert math.isnan(float(getattr(X, m)(*a))), (type(X).__name__, m, a)
    # tails far below e^-1000 by the Chernoff bound are exact 0 and 1
    far = S.NoncentralChiSquared(df=3.5, nc=1e100)
    assert float(far.cdf(1.0)) == 0.0 and float(far.ccdf(1e308)) == 0.0
    # ... and values where they are right: the density's closed form, the mode from it, a tail
    # beyond every term (Arb for the density; the mode is nc + df - 3 to 1e-15 here)
    X = S.NoncentralChiSquared(df=3.5, nc=1e15)
    assert _within(float(X.pdf(1e15)), 6.307831305050399e-09) and _within(
        float(X.logpdf(1.0000001e15)), -20.131473836219964
    )
    assert _within(float(X.mode()), 1e15 + 0.5, 1e-15)
    assert float(S.NoncentralChiSquared(df=3.5, nc=2.0).ccdf(1e300)) == 0.0
    # Degrees of freedom far beyond the region: the limit distributions (the t's normal, the
    # F's noncentral chi-square at dfn x; Arb). These were 0, or Phi(-nc) at every small t:
    # the test for a tail below e^-2000 took three log Gammas of 5e24 and more.
    assert _within(float(S.NoncentralT(df=1e25, nc=-1000.0).ccdf(-998.0)), 0.02275013194817921)
    assert _within(float(S.NoncentralT(df=1e300, nc=1.5).cdf(0.01)), 0.06811211796672545)
    assert _within(float(S.NoncentralT(df=1e300, nc=1.5).ccdf(0.3)), 0.8849303297782918)
    assert _within(float(S.NoncentralT(df=1e100, nc=1e4).cdf(9997.0)), 0.0013498980316300946)
    assert _within(float(S.NoncentralF(dfn=4.0, dfd=1e300, nc=3.0).cdf(1.75)), 0.5786191906312934)
    assert math.isnan(float(S.NoncentralF(dfn=4.0, dfd=1e300, nc=3.0).cdf(1e-10)))  # (it was 1e-10 off)
    assert math.isnan(float(S.NoncentralF(dfn=4.0, dfd=1e100, nc=3.0).pdf(1e-300)))  # (it was inf)
    # moments whose products pass the doubles, and the t's mode beyond what its equation resolves
    assert _within(float(S.NoncentralChiSquared(df=3.5, nc=1e300).skewness()), 3e-150)
    assert _within(float(S.NoncentralT(df=1e300, nc=-1e300).variance()), 5e299)
    assert math.isnan(float(S.NoncentralT(df=1e10, nc=-1e5).kurtosis())) and math.isnan(
        float(S.NoncentralT(df=9.0, nc=1e15).mode())
    )
    assert math.isnan(float(S.NoncentralF(dfn=1e100, dfd=1e100, nc=1e100).kurtosis()))
    for X in (
        S.NoncentralChiSquared(df=1e300, nc=1e300),
        S.NoncentralT(df=1e300, nc=1e300),
        S.NoncentralF(dfn=1e300, dfd=1e300, nc=1e300),
    ):
        for m in ("pdf", "logpdf", "cdf", "ccdf", "logcdf", "logccdf"):
            for x in (1e-300, 1.0, 1e300):
                float(getattr(X, m)(x))
        for m in ("mode", "median", "entropy"):
            float(getattr(X, m)())
    assert time.perf_counter() - start < 5.0


def test_central_extreme_arguments():
    import time

    # The 18 central distributions at the corners of the double range (parameters, arguments,
    # probabilities and log-probabilities from 5e-324 to 1.7e308): every value against Arb
    # (tools/stats_ref.py's definitions). Each of these was NaN, inf or wrong before: a
    # standardized z or a quotient x / s beyond the range (the double-double arithmetic gave
    # NaN), a product with a subnormal shape, moments whose factors leave the range, the F
    # density with a beta argument below the doubles (F(13, 1e5).logpdf(5e-324) was +774.7).
    points = [
        ("Normal", "ilogcdf", dict(mu=0.0, sigma=1.0), -1e300, -1.4142135623730951e150),
        ("Normal", "ilogccdf", dict(mu=1e300, sigma=1e100), -1e300, 1e300),
        ("Normal", "icdf", dict(mu=1.0, sigma=1.7e308), 1e-300, -math.inf),
        ("Laplace", "logcdf", dict(mu=0.0, sigma=5e-324), -1.0, -math.inf),
        ("Laplace", "pdf", dict(mu=1e-10, sigma=1e-100), 1e300, 0.0),
        ("Laplace", "logccdf", dict(mu=-1e-100, sigma=1e-10), 1.7e308, -math.inf),
        ("Logistic", "logcdf", dict(mu=1.0, sigma=5e-324), -1e-10, -math.inf),
        ("Logistic", "pdf", dict(mu=1e300, sigma=1e-100), 0.0, 0.0),
        ("Gumbel", "logccdf", dict(mu=-1.0, sigma=1e-100), 1.7e308, -math.inf),
        ("Gumbel", "ccdf", dict(mu=1e-10, sigma=1e100), -1.7e308, 1.0),
        ("Gumbel", "logpdf", dict(mu=-10000000000.0, sigma=1e-300), 1e16, -math.inf),
        ("Cauchy", "logcdf", dict(mu=-1e300, sigma=1e-300), -2e300, -1382.695785682277),
        ("Cauchy", "logccdf", dict(mu=-1e300, sigma=1e-300), 1000.0, -1382.695785682277),
        ("Cauchy", "cdf", dict(mu=1e-10, sigma=1e-300), -1e16, 3.183099e-317),
        ("Cauchy", "logpdf", dict(mu=1e-10, sigma=5e-324), -1e100, -1206.10182040604),
        ("Cauchy", "icdf", dict(mu=0.0, sigma=5e-324), 5e-324, -0.3183098861837907),
        ("Cauchy", "ilogccdf", dict(mu=0.0, sigma=1.0), -1e300, math.inf),
        ("LogNormal", "ilogccdf", dict(mu=1e-300, sigma=1e-100), -1e300, math.inf),
        ("LogNormal", "variance", dict(mu=1e300, sigma=1e-300), None, math.inf),
        ("LogNormal", "skewness", dict(mu=0.0, sigma=1e-300), None, 3e-300),
        ("LogNormal", "mean", dict(mu=-1e300, sigma=10000000000.0), None, 0.0),
        ("LogNormal", "kurtosis", dict(mu=0.0, sigma=10000000000.0), None, math.inf),
        ("LogNormal", "logpdf", dict(mu=-1e100, sigma=1e-100), 1.0, -math.inf),
        ("LogNormal", "mode", dict(mu=1e300, sigma=1e-300), None, math.inf),
        ("Uniform", "logccdf", dict(a=-1.7e308, b=1e-100), -1e-10, -732.7526878231687),
        ("Uniform", "ilogcdf", dict(a=-1e300, b=0.0), -745.0, -1e300),
        ("Exponential", "logcdf", dict(scale=1e100), 2.822350730471937e-224, -745.0),
        ("Weibull", "pdf", dict(c=10000000000.0, scale=1e-300), 1.000000000152718e-300, math.inf),
        ("Weibull", "skewness", dict(c=1e100, scale=1.7e308), None, -1.1395470994046486),
        ("Weibull", "kurtosis", dict(c=10000000000.0, scale=1e300), None, 2.3999999971082313),
        ("Weibull", "mean", dict(c=1e-300, scale=1.0), None, math.inf),
        ("Weibull", "variance", dict(c=1e100, scale=1.0), None, 1.6449340668482265e-200),
        ("Weibull", "iccdf", dict(c=1e-100, scale=5e-324), 0.01, math.inf),
        ("Weibull", "logcdf", dict(c=1.7e308, scale=1e100), 1.0, -math.inf),
        ("Weibull", "entropy", dict(c=5e-324, scale=1.0), None, -math.inf),
        ("Pareto", "ilogcdf", dict(b=5e-324, scale=1e-300), -745.0, 1.7704790283768383e-300),
        ("Pareto", "skewness", dict(b=1.7e308, scale=1.0), None, 2.0),
        ("Pareto", "kurtosis", dict(b=1e300, scale=1.0), None, 6.0),
        ("Pareto", "variance", dict(b=1e200, scale=1e300), None, 1.0000000000000001e200),
        ("Pareto", "logccdf", dict(b=1.7e308, scale=1e-10), 3.0, -math.inf),
        ("Rayleigh", "cdf", dict(scale=5e-324), 0.5, 1.0),
        ("Rayleigh", "logpdf", dict(scale=1e-100), 1.7e308, -math.inf),
        ("Rayleigh", "logccdf", dict(scale=1e-10), 1.7e308, -math.inf),
        ("GEV", "mean", dict(c=5e-324, mu=0.0, sigma=1.0), None, 0.5772156649015329),
        ("GEV", "iccdf", dict(c=-5e-324, mu=1e-300, sigma=1.0), 1e-10, 23.025850929890456),
        ("GEV", "mode", dict(c=-1e-300, mu=1.0, sigma=1e300), None, -7.756385209041318e-17),
        ("GEV", "cdf", dict(c=1.7e308, mu=0.0, sigma=1.7e308), 0.9999999999999999, 0.36787944117144233),
        ("GEV", "logccdf", dict(c=1.0, mu=1e-300, sigma=1e300), 1e300, -1381.5510557964274),
        ("GEV", "ilogccdf", dict(c=1.0, mu=-1e300, sigma=1e300), -745.0, -2.8223507304719374e-24),
        ("GEV", "variance", dict(c=100.0, mu=0.0, sigma=1e-300), None, 7.886578673647905e-230),
        ("GEV", "mean", dict(c=1.0000001, mu=0.0, sigma=1.0), None, -4.227843342509175e-08),
        ("GEV", "standard_deviation", dict(c=-1e-300, mu=1.0, sigma=1e300), None, 1.2825498301618642e300),
        ("GEV", "skewness", dict(c=-1e-300, mu=1.0, sigma=1.0), None, 1.1395470994046486),
        ("GEV", "logpdf", dict(c=-1e300, mu=1e-300, sigma=1.7e308), -169999999.99999997, -673.7441552867782),
        ("GEV", "cdf", dict(c=-0.5, mu=1.7e308, sigma=1.7e308), -1.1390447050961305e308, 1.1102230246251546e-16),
        ("GEV", "logccdf", dict(c=1e-300, mu=1e-300, sigma=1e-300), 1.0, -6.907755278982137e302),
        ("GEV", "cdf", dict(c=1.7e308, mu=1.7e308, sigma=1e300), 1e100, 0.36787944117144233),
        ("Gamma", "pdf", dict(a=0.5, scale=1e100), 5e-324, 2.538240300160582e111),
        ("Gamma", "cdf", dict(a=1e-10, scale=1e300), 5e-324, 0.9999998565361718),
        ("Gamma", "logccdf", dict(a=1e-10, scale=1e300), 5e-324, -15.757182902339945),
        ("Gamma", "ccdf", dict(a=3.7, scale=1e-10), 1e300, 0.0),
        ("Gamma", "logpdf", dict(a=1e-10, scale=1.7e308), 5e-324, 721.4142208460819),
        ("Gamma", "logcdf", dict(a=1000.0, scale=1e300), 5e-324, -1441127.7279980832),
        ("ChiSquared", "cdf", dict(df=1.0), 5e-324, 1.7735048886036274e-162),
        ("ChiSquared", "logcdf", dict(df=2.0), 1e-323, -744.4400719213812),
        ("InverseGamma", "ccdf", dict(a=1e-10, scale=5e-324), 1e300, 0.9999998565361718),
        ("InverseGamma", "pdf", dict(a=0.5, scale=5e-324), 1e100, 1.2540573332e-312),
        ("InverseGamma", "logcdf", dict(a=3.7, scale=1e300), 5e-324, -math.inf),
        # (a below 2^-1021: the limit's two atoms, P(1) = a / (a + b) = 1e-323, sd sqrt(q (1 - q)),
        # T50; the Beta's own value is 2.57e-162, sqrt(1 + b) apart)
        ("Beta", "standard_deviation", dict(a=5e-324, b=0.5), None, 3.1434555694052576e-162),
        ("Beta", "entropy", dict(a=5e-324, b=1.0), None, -math.inf),
        ("Beta", "skewness", dict(a=1e-300, b=1e-100), None, 1e100),
        ("Beta", "mean", dict(a=1.7e308, b=1.7e308), None, 0.5),
        ("StudentT", "cdf", dict(df=1e-300), -1.7e308, 0.5),
        ("StudentT", "ccdf", dict(df=1e-50), -1e300, 0.5),
        ("StudentT", "logccdf", dict(df=1e-100), 1e300, -0.6931471805599453),
        ("F", "logpdf", dict(dfn=13.0, dfd=100000.0), 5e-324, -4087.915885991893),
        ("F", "pdf", dict(dfn=0.5, dfd=13.0), 1e-320, 1.922065934334491e239),
        ("F", "logpdf", dict(dfn=1e100, dfd=13.0), 1e300, -5174.3123071466),
        ("F", "pdf", dict(dfn=1e100, dfd=13.0), 1e300, 0.0),
        ("F", "logpdf", dict(dfn=100000.0, dfd=13.0), 1.7e308, -5316.446767123508),
        ("F", "logcdf", dict(dfn=2.2e281, dfd=49.0), 1.8e-292, -3.0628413525357505e282),
        ("F", "logpdf", dict(dfn=13.0, dfd=1e100), 1e-300, -3792.761251350172),
        # moments whose factors leave the range (a subnormal shape; dfn, dfd next to the ends)
        ("Beta", "kurtosis", dict(a=5e-324, b=5e-324), None, -2.0),
        ("F", "kurtosis", dict(dfn=0.5, dfd=1.7e308), None, 24.0),
        ("F", "kurtosis", dict(dfn=10000000000.0, dfd=1.7e308), None, 1.2e-09),
        ("F", "kurtosis", dict(dfn=1.7e308, dfd=1.7e308), None, 3.8823529411764706e-307),
        ("F", "kurtosis", dict(dfn=1e-10, dfd=1e300), None, 120000000000.0),
        ("F", "skewness", dict(dfn=1e-300, dfd=1e300), None, 2.82842712474619e150),
        ("F", "skewness", dict(dfn=5e-324, dfd=1.7e308), None, 1.2724849808380786e162),
        ("F", "skewness", dict(dfn=1e300, dfd=13.0), None, 2.424366106925306),
        ("F", "skewness", dict(dfn=1e300, dfd=100000.0), None, 0.017889259401108203),
    ]
    start = time.perf_counter()
    for name, method, params, x, want in points:
        X = getattr(S, name)(**params)
        got = float(getattr(X, method)() if x is None else getattr(X, method)(x))
        if math.isinf(want) or want == 0.0:
            assert got == want, (name, method, params, x, got, want)
        else:
            assert abs(got - want) <= 1e-14 * abs(want), (name, method, params, x, got, want)
        if x is not None:  # the array path gives the same bits
            arr = float(np.asarray(getattr(X, method)(sd.asarray([x, x])))[1])
            assert arr == got or (math.isnan(arr) and math.isnan(got)), (name, method, params, x)
    # Outside the guaranteed shapes a value that cannot be vouched for is NaN: the gamma family's
    # cdfs and inverses past a = 2^100 (their values left [0, 1]: Gamma(a=1e100, scale=1e-300)
    # .cdf(1e-200) was -2.3e33), the beta family's with both shapes past 2^100 (F(1.7e308,
    # 1.7e308).cdf(0.5) was 1), a df below 2^-1021 (no exact half: StudentT(df=5e-324)
    # .logccdf(40) was +105), the F density with both shapes huge and a beta argument below the
    # doubles.
    nans = [
        S.Gamma(a=1e100, scale=1e-300).cdf(1e-200),
        S.Gamma(a=1e100).icdf(0.5),
        S.Gamma(a=1.7e308).iccdf(5e-324),
        S.InverseGamma(a=1e300, scale=1e-300).ccdf(1e-100),
        S.ChiSquared(df=1e31).cdf(1e31),
        S.ChiSquared(df=5e-324).cdf(1.0),
        S.StudentT(df=5e-324).logccdf(40.0),
        S.F(dfn=5e-324, dfd=1.0).cdf(1.0),
        S.F(dfn=1.7e308, dfd=1.7e308).cdf(0.5),
        S.F(dfn=1e100, dfd=1e100).logpdf(1e-310),
        S.Beta(a=1e300, b=1e300).icdf(0.5),
    ]
    assert all(math.isnan(float(v)) for v in nans), [float(v) for v in nans]
    # an inverse outside the measured shapes is returned only if the forward tail reproduces
    # the target: inside the support, or NaN (Beta(1e-300, 1e-300).icdf(0.5) was 2.8e283)
    for X, lo, hi in (
        (S.Beta(a=1e-300, b=1e-300), 0.0, 1.0),
        (S.Beta(a=1e-10, b=1e-10), 0.0, 1.0),
        (S.F(dfn=1.0, dfd=1e-300), 0.0, math.inf),
        (S.Gamma(a=1e-100), 0.0, math.inf),
        (S.ChiSquared(df=1e15), 9e14, 1.1e15),
    ):
        for p in (1e-300, 0.5, 1 - 1e-10):
            v = float(X.icdf(p))
            assert math.isnan(v) or lo <= v <= hi, (X, p, v)
            if lo < v < hi:
                assert abs(float(X.cdf(v)) - p) <= 1e-6 * p, (X, p, v)
    # bounded time and sane ranges at the far corners (special's inverses at shapes of 1e308
    # take up to 0.25 s a call: one of each here)
    for X in (
        S.Gamma(a=1.7e308),
        S.Beta(a=5e-324, b=1e300),
        S.StudentT(df=1e-300),
        S.F(dfn=1e-300, dfd=1.7e308),
        S.GEV(c=1.7e308, mu=1.7e308, sigma=5e-324),
        S.Weibull(c=5e-324, scale=1.7e308),
        S.LogNormal(mu=-1.7e308, sigma=1.7e308),
    ):
        for x in (5e-324, 1.0, 1.7e308, -1.7e308):
            c, cc, f = float(X.cdf(x)), float(X.ccdf(x)), float(X.pdf(x))
            assert math.isnan(c) or 0.0 <= c <= 1.0, (X, x, c)
            assert math.isnan(cc) or 0.0 <= cc <= 1.0, (X, x, cc)
            assert math.isnan(f) or f >= 0.0, (X, x, f)
            assert math.isnan(float(X.logcdf(x))) or float(X.logcdf(x)) <= 0.0, (X, x)
            assert math.isnan(float(X.logccdf(x))) or float(X.logccdf(x)) <= 0.0, (X, x)
        float(X.icdf(1e-300)), float(X.iccdf(0.5)), float(X.ilogcdf(-745.0)), float(X.ilogccdf(-1e300))
    assert time.perf_counter() - start < 30.0


def test_gamma_family_inverses_outside_the_measured_shapes():
    import time

    # The gamma family's inverses outside the measured shapes (a < 1e-6 or a > 1e6) return a
    # root only if the forward log tail at it reproduces the target; these calls returned wrong
    # numbers through three holes, each closed at its cause:
    #   - the check's tolerance came from the tail under test, so a tail of -inf made it
    #     infinite (Gamma(a=1e-300).ilogccdf(-1e50) was 1.27e80, Gamma(a=5e-324)
    #     .ilogccdf(-1e10) was 1.27e40: the deep iteration's 100th doubling);
    #   - the deep iteration returned its last iterate when it had not settled;
    #   - the log upper tail lost its bits where a / y is below the normal range (-inf for
    #     Gamma(a=1e-200).logccdf(1e150)) and for a subnormal shape (Gamma(a=5e-324)
    #     .logccdf(1.0) was -745.44, and .ilogccdf(-745) passed the check against it with 0.56).
    # Every value against Arb (the inverses certified: the true log tail brackets the target
    # within 2^-56 of the root).
    points = [
        ("Gamma", "ilogccdf", dict(a=5e-324, scale=1.0), -10000000000.0, 9999999232.534077),
        ("Gamma", "ilogccdf", dict(a=5e-324, scale=1e-300), -10000000000.0, 9.999999232534077e-291),
        ("Gamma", "ilogccdf", dict(a=5e-324, scale=1e100), -745.0, 4.906714973236744e99),
        ("Gamma", "ilogccdf", dict(a=5e-324, scale=1.0), -1e200, 1e200),
        ("Gamma", "iccdf", dict(a=1e-320, scale=1.0), 1e-322, 3.2200095811354794),
        ("Gamma", "ilogccdf", dict(a=1e-310, scale=1e-300), -100000.0, 9.92746929651357e-296),
        ("InverseGamma", "ilogcdf", dict(a=5e-324, scale=1.0), -745.0, 2.038023413738956),
        ("InverseGamma", "ilogcdf", dict(a=5e-324, scale=1e-100), -10000000000.0, 1.0000000767465982e-110),
        ("InverseGamma", "ilogcdf", dict(a=5e-324, scale=1.7e308), -10000000000.0, 1.7000001304692168e298),
        ("InverseGamma", "icdf", dict(a=1e-320, scale=1e300), 1e-322, 3.105580821431493e299),
        ("Gamma", "ilogccdf", dict(a=2.3e-308, scale=1.0), -1e20, 1e20),
        ("Gamma", "ilogccdf", dict(a=1e-300, scale=1.0), -1e50, 1e50),
        ("InverseGamma", "ilogcdf", dict(a=1e-300, scale=1e300), -1e100, 1e200),
        ("ChiSquared", "ilogccdf", dict(df=2e-300), -1e50, 2e50),
        ("ChiSquared", "ilogccdf", dict(df=4.6e-308), -1e200, 2e200),
        # roots beyond the doubles: a target Q above a E1(y) for every double s y; P(a, y) = p
        # at log y = log(p) / a
        ("Gamma", "ilogccdf", dict(a=1e-200, scale=1e300), -1e200, math.inf),
        ("Gamma", "ilogccdf", dict(a=2.3e-308, scale=1.0), -700.0, 0.0),
        ("Gamma", "ilogccdf", dict(a=5e-324, scale=1e300), -700.0, 0.0),
        ("Gamma", "iccdf", dict(a=5e-324, scale=1.0), 0.01, 0.0),
        ("Gamma", "icdf", dict(a=5e-324, scale=1.7e308), 0.5, 0.0),
        ("InverseGamma", "icdf", dict(a=5e-324, scale=1.0), 0.01, math.inf),
        ("InverseGamma", "ilogcdf", dict(a=2.3e-308, scale=1.0), -700.0, math.inf),
        # the forward tails the check reads
        ("Gamma", "logccdf", dict(a=5e-324, scale=1.0), 30.0, -777.873077706088),
        ("Gamma", "logccdf", dict(a=5e-324, scale=1.0), 1.0, -745.9570038803834),
        ("Gamma", "logccdf", dict(a=5e-324, scale=1.0), 1e-300, -737.9030929560503),
        ("Gamma", "logccdf", dict(a=1e-320, scale=1e-300), 7e-298, -1443.3797467487816),
        ("Gamma", "logccdf", dict(a=1e-310, scale=1.0), 100000.0, -100725.31431429298),
        ("Gamma", "ccdf", dict(a=1e-310, scale=1.0), 1e-05, 1.093571980004366e-309),
        ("Gamma", "logcdf", dict(a=1e-310, scale=1.0), 1e-05, -1.093571980004366e-309),
        ("Gamma", "cdf", dict(a=5e-324, scale=1.0), 1e-300, 1.0),
        ("InverseGamma", "logcdf", dict(a=5e-324, scale=1.0), 0.01, -849.0550962718867),
        ("Gamma", "logccdf", dict(a=1e-200, scale=1.0), 1e150, -1e150),
        ("Gamma", "logccdf", dict(a=1e-100, scale=1.0), 1e250, -1e250),
        ("Gamma", "logccdf", dict(a=1e-300, scale=1.0), 10000000000.0, -10000000713.801378),
        ("InverseGamma", "logcdf", dict(a=1e-200, scale=1.0), 1e-150, -1e150),
        ("ChiSquared", "logccdf", dict(df=2e-200), 2e150, -1e150),
    ]
    start = time.perf_counter()
    for name, method, params, x, want in points:
        X = getattr(S, name)(**params)
        got = float(getattr(X, method)(x))
        if math.isinf(want) or want == 0.0:
            assert got == want, (name, method, params, x, got, want)
        else:
            assert abs(got - want) <= 1e-14 * abs(want), (name, method, params, x, got, want)
        arr = float(np.asarray(getattr(X, method)(sd.asarray([x, x])))[1])  # the array path: the same bits
        assert arr == got, (name, method, params, x)
    # The right value or NaN, never another number: over shapes on both sides of the measured
    # range every finite normal root reproduces its target through the forward log tail (the
    # tails above: at v (1 -+ 1e-13) they bracket the target, to 1e-9), and the roots are
    # monotone in the target.
    forward = {"icdf": "logcdf", "iccdf": "logccdf", "ilogcdf": "logcdf", "ilogccdf": "logccdf"}
    logps = (-1e200, -1e50, -1e10, -745.0, -700.0, -1.0)
    ps = (1e-300, 1e-10, 0.5)
    for a in (5e-324, 1e-320, 1e-310, 2.3e-308, 1e-300, 1e-100, 1e-20, 1e-8, 1e7, 1e10, 1e12, 1e20):
        for scale in (1e-300, 1.0, 1e300):
            for X in (S.Gamma(a=a, scale=scale), S.InverseGamma(a=a, scale=scale)):
                for method, targets in (("ilogcdf", logps), ("ilogccdf", logps), ("icdf", ps), ("iccdf", ps)):
                    roots = [float(getattr(X, method)(t)) for t in targets]
                    for t, v in zip(targets, roots):
                        assert math.isnan(v) or v >= 0.0, (X, method, t, v)
                        if 2.3e-308 <= v < math.inf:  # (a subnormal root has lost bits of its own)
                            lt = t if method.startswith("ilog") else math.log(t)
                            lo, hi = sorted(float(getattr(X, forward[method])(v * (1.0 + d))) for d in (-1e-13, 1e-13))
                            tol = 1e-9 * max(1.0, abs(lt))
                            assert lo - tol <= lt <= hi + tol, (X, method, t, v, lo, hi)
                    known = [v for v in roots if not math.isnan(v)]
                    rising = sorted(known) if method in ("icdf", "ilogcdf") else sorted(known, reverse=True)
                    assert known == rising, (X, method, roots)
    assert time.perf_counter() - start < 30.0


def test_probabilities_and_densities_are_never_negative_zero():
    # a formula may round to -0.0 (Weibull.pdf(1e100), Gumbel.cdf(-50), Rayleigh.ccdf(1e100));
    # the engine returns +0.0 for pdf, cdf and ccdf, as an array and as a scalar
    xs = np.array([-1e308, -1e10, -50.0, 0.0, 0.5, 50.0, 1e5, 1e10, 1e100, 1e300, 1e308])
    for X in (S.Weibull(c=2.0), S.Gumbel(), S.Rayleigh(), S.Normal(), S.Gamma(a=2.5), S.GEV(c=0.1), S.Logistic()):
        for m in ("pdf", "cdf", "ccdf"):
            v = np.asarray(getattr(X, m)(sd.asarray(xs)))
            assert not np.any((v == 0) & np.signbit(v)), (type(X).__name__, m)
            for x in xs:
                y = float(getattr(X, m)(float(x)))
                assert not (y == 0 and math.copysign(1.0, y) < 0), (type(X).__name__, m, x)


def test_errors_name_the_method_not_the_kernel():
    n = sd.stats.Normal()
    with pytest.raises(TypeError, match=r"^cdf\(\): 'x' must be an sd.ndarray or a real number, not str$"):
        n.cdf("a")
    with pytest.raises(TypeError, match=r"^icdf\(\): 'p' must be"):
        n.icdf("a")
    with pytest.raises(TypeError, match=r"^pdf\(\): 'x' must be real, not a complex128 array$"):
        n.pdf(sd.asarray([1j]))
    with pytest.raises(TypeError, match=r"^pmf\(\): 'k' must be"):
        sd.stats.Binomial(n=5, p=0.3).pmf("a")
    with pytest.raises(OverflowError, match=r"^cdf\(\): "):
        n.cdf(10**400)
    with pytest.raises(ValueError, match=r"^cdf\(\): shapes could not be broadcast together"):
        sd.stats.Normal(mu=sd.ones((2,))).cdf(sd.ones((3,)))
    mvn = sd.stats.MultivariateNormal(mu=sd.zeros((2,)), cov=sd.eye(2))
    with pytest.raises(ValueError, match=r"^cdf\(\): 'x' must have 2 components"):
        mvn.cdf(sd.ones((3,)))
    with pytest.raises(ValueError, match=r"^logpdf\(\): 'x' must be \(2, 2\)"):
        sd.stats.Wishart(df=3.0, scale=sd.eye(2)).logpdf(sd.ones((3, 3)))


# ---- limits and agreements between distributions

A = sd.asarray
sp = sd.special
inf = math.inf
DENSITY_METHODS = INPUT_METHODS[:6]  # pdf to logccdf
INVERSE_METHODS = INPUT_METHODS[6:8]  # icdf, iccdf
LOG_INVERSE_METHODS = INPUT_METHODS[8:]  # ilogcdf, ilogccdf
_DFS = (0.01, 0.5, 1.0, 2.5, 3.0, 4.5, 6.5, 9.0, 11.0, 1e3, 1e17, 1e45, math.inf)


def _same(a, b, rel=0.0):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b or abs(a - b) <= rel * max(1.0, abs(b))


def _agree(X, Y, xs, rel):
    """X's methods equal Y's within rel (0: exactly), the chunk path too."""
    for m in DENSITY_METHODS:
        for x in xs:
            assert _same(getattr(X, m)(x), getattr(Y, m)(x), rel), (m, x)
            assert float(getattr(X, m)(A([x, x]))[1]) == getattr(X, m)(x)  # the chunk path too
    for m in INVERSE_METHODS:
        for p in (1e-300, 0.1, 0.5, 0.9):
            assert _same(getattr(X, m)(p), getattr(Y, m)(p), rel), (m, p)
    for m in LOG_INVERSE_METHODS:
        for lp in (-1e3, -2.0, -0.1):
            assert _same(getattr(X, m)(lp), getattr(Y, m)(lp), rel), (m, lp)
    for m in PARAM_METHODS:
        assert _same(getattr(X, m)(), getattr(Y, m)(), rel), m
    assert X.support() == Y.support() or rel


def _same_everywhere(a, b):
    """a and b give the same bits (any NaN for any NaN) in every method and in their samples."""
    for m in DENSITY_METHODS:
        for x in (-3.0, -0.5, 0.0, 0.3, 1.0, 2.5, 40.0, 1e10):
            assert bits_equal_any_nan(getattr(a, m)(x), getattr(b, m)(x)), (a, m, x)
    for m in INVERSE_METHODS:
        for p in (1e-300, 1e-10, 0.1, 0.5, 0.9, 1 - 1e-12):
            assert bits_equal_any_nan(getattr(a, m)(p), getattr(b, m)(p)), (a, m, p)
    for m in LOG_INVERSE_METHODS:
        for lp in (-1e5, -30.0, -0.7, -1e-8):
            assert bits_equal_any_nan(getattr(a, m)(lp), getattr(b, m)(lp)), (a, m, lp)
    for m in [*PARAM_METHODS, "support"]:
        assert bits_equal_any_nan(getattr(a, m)(), getattr(b, m)()), (a, m)
    sa = a.sample(6, rng=sd.random.default_rng(3)).tolist()
    sb = b.sample(6, rng=sd.random.default_rng(3)).tolist()
    assert bits_equal_any_nan(sa, sb), a


# ---- an infinite degree of freedom is the limit (user decision 2026-11-19)


def test_t_inf_is_the_normal():
    _agree(S.StudentT(df=inf), S.Normal(), (-40.0, -1.5, 0.0, 0.3, 7.0), 0.0)
    assert S.StudentT(df=inf).cdf(1.3) == sp.stdtr(inf, 1.3)
    assert S.StudentT(df=inf).icdf(0.3) == sp.stdtrit(inf, 0.3)


def test_noncentral_t_inf_is_the_shifted_normal():
    _agree(S.NoncentralT(df=inf, nc=1.5), S.Normal(mu=1.5), (-3.0, 0.0, 1.5, 4.0), 0.0)
    assert S.NoncentralT(df=inf, nc=1.0).cdf(1.5) == sp.nctdtr(inf, 1.0, 1.5)


def test_f_one_infinite_df():
    # dfn = inf: dfd / chi-square(dfd), the inverse gamma (dfd/2, dfd/2)
    _agree(S.F(dfn=inf, dfd=5.0), S.InverseGamma(a=2.5, scale=2.5), (0.3, 1.0, 4.0), 0.0)
    _agree(S.NoncentralF(dfn=inf, dfd=5.0, nc=2.0), S.InverseGamma(a=2.5, scale=2.5), (0.3, 1.0, 4.0), 0.0)
    # dfd = inf: chi-square(dfn) / dfn and the noncentral chi-square(dfn, nc) / dfn
    X = S.F(dfn=3.0, dfd=inf)
    for x in (0.2, 1.0, 3.0):
        assert X.cdf(x) == sp.fdtr(3.0, inf, x)
        assert X.ccdf(x) == pytest.approx(sp.fdtrc(3.0, inf, x), rel=2e-16)
        assert X.pdf(x) == pytest.approx(3.0 * S.ChiSquared(df=3.0).pdf(3.0 * x), rel=1e-15)
    assert X.icdf(0.3) == pytest.approx(sp.fdtri(3.0, inf, 0.3), rel=1e-15)
    assert (X.mean(), X.variance()) == (1.0, pytest.approx(2.0 / 3.0, rel=1e-15))
    Y = S.NoncentralF(dfn=3.0, dfd=inf, nc=2.0)
    assert Y.cdf(1.0) == sp.ncfdtr(3.0, inf, 2.0, 1.0)
    assert Y.icdf(0.3) == pytest.approx(sp.ncfdtri(3.0, inf, 2.0, 0.3), rel=1e-15)
    assert Y.mean() == pytest.approx(5.0 / 3.0, rel=1e-15)
    assert Y.entropy() == pytest.approx(S.NoncentralChiSquared(df=3.0, nc=2.0).entropy() - math.log(3.0), rel=1e-14)


def test_not_distributions_stay_nan():
    for X in (
        S.F(dfn=inf, dfd=inf),
        S.NoncentralF(dfn=inf, dfd=inf, nc=1.0),
        S.ChiSquared(df=inf),
        S.NoncentralChiSquared(df=inf, nc=1.0),
        S.NoncentralT(df=inf, nc=inf),
    ):
        assert math.isnan(X.cdf(1.0)) and math.isnan(X.mean())


def test_infinite_df_samples_and_arrays():
    rng = sd.random.default_rng(4)
    z = np.asarray(S.StudentT(df=inf).sample(200000, rng=rng))
    assert abs(z.mean()) < 0.01 and abs(z.var() - 1.0) < 0.01
    f = np.asarray(S.F(dfn=3.0, dfd=inf).sample(200000, rng=rng))
    assert abs(f.mean() - 1.0) < 0.01 and abs(f.var() - 2.0 / 3.0) < 0.02
    g = np.asarray(S.NoncentralF(dfn=inf, dfd=10.0, nc=1.0).sample(200000, rng=rng))
    assert abs(g.mean() - 10.0 / 8.0) < 0.02
    # a batch mixing finite and infinite df: each element its own
    v = S.StudentT(df=A([3.0, inf])).cdf(1.0)
    assert float(v[0]) == S.StudentT(df=3.0).cdf(1.0) and float(v[1]) == S.Normal().cdf(1.0)
    # the multivariate t agrees
    mvt = S.MultivariateT(mu=A([0.0]), shape=A([[1.0]]), df=inf)
    assert mvt.logpdf(A([0.7])) == pytest.approx(S.StudentT(df=inf).logpdf(0.7), rel=1e-15)


def test_f_kurtosis_diverges_where_the_variance_is_finite():
    for d in (4.5, 5.0, 6.0, 6.5, 8.0):
        assert S.F(dfn=5.0, dfd=d).kurtosis() == inf
    assert math.isnan(S.F(dfn=5.0, dfd=4.0).kurtosis())
    assert math.isfinite(S.F(dfn=5.0, dfd=9.0).kurtosis())


def test_noncentral_chi_square_nc0_moments_are_the_central_ones():
    for df in (0.5, 3.0, 17.0):
        X, Y = S.NoncentralChiSquared(df=df, nc=0.0), S.ChiSquared(df=df)
        assert (X.skewness(), X.kurtosis(), X.mean(), X.variance()) == (
            Y.skewness(),
            Y.kurtosis(),
            Y.mean(),
            Y.variance(),
        )


# ---- the noncentral NaN boundaries (docs/api/stats.md §2.4)


@pytest.mark.deep
def test_noncentral_chi_square_large_nc():
    for nc in (1.5e9, 2.2e9):
        X = S.NoncentralChiSquared(df=3.0, nc=nc)
        s = math.sqrt(2.0 * (3.0 + 2.0 * nc))
        xs = A(np.linspace(nc - 8 * s, nc + 8 * s, 101))
        assert not np.isnan(np.asarray(X.cdf(xs))).any() and not np.isnan(np.asarray(X.ccdf(xs))).any()
        ps = A(np.linspace(0.01, 0.99, 25))
        q = np.asarray(X.icdf(ps))
        assert not np.isnan(q).any() and (np.diff(q) > 0).all()
        assert X.median() == pytest.approx(nc + 3.0 - 2.0 / 3.0, rel=1e-9)


# ---- messages


def test_constructor_messages_name_the_class():
    with pytest.raises(ValueError, match=r"^Normal\(\): shapes could not be broadcast together: \(2,\) \(3,\)$"):
        S.Normal(mu=A([1.0, 2.0]), sigma=A([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match=r"^Gamma\(\): 'a' must be a real number or an sd.ndarray, not str$"):
        S.Gamma(a="x")
    with pytest.raises(OverflowError, match=r"^Normal\(\): 'mu' is out of range$"):
        S.Normal(mu=10**400)
    with pytest.raises(ValueError, match=r"^MultivariateNormal\(\): 'cov' must be \(2, 2\)"):
        S.MultivariateNormal(mu=A([0.0, 0.0]), cov=A([[1.0]]))
    with pytest.raises(TypeError, match=r"^Multinomial\(\): 'n' must be a real number or an sd.ndarray, not str$"):
        S.Multinomial(n="a", p=A([0.5, 0.5]))
    with pytest.raises(ValueError, match=r"^Wishart\(\): shapes could not be broadcast together: "):
        S.Wishart(df=A([3.0, 4.0, 5.0]), scale=A(np.stack([np.eye(2)] * 2)))
    with pytest.raises(TypeError, match=r"^ContinuousDistribution\(\): a base class"):
        S.ContinuousDistribution()


def test_method_messages():
    M = S.MultivariateNormal(mu=A([0.0, 0.0]), cov=A([[1.0, 0.2], [0.2, 1.0]]))
    with pytest.raises(ValueError, match=r"^cdf\(\): 'atol' must be a finite number > 0$"):
        M.cdf(A([0.0, 0.0]), atol=-1.0)
    with pytest.raises(TypeError, match=r"^cdf\(\): 'atol' must be an int or a float, not str$"):
        M.cdf(A([0.0, 0.0]), atol="a")
    with pytest.raises(TypeError, match=r"^cdf\(\): 'max_points' must be an int or None, not float$"):
        M.cdf(A([0.0, 0.0]), max_points=1.5)
    with pytest.raises(ValueError, match=r"^cdf\(\): 'max_points' must be >= 1$"):
        M.cdf(A([0.0, 0.0]), max_points=0)
    with pytest.raises(TypeError, match=r"^cdf\(\): 'return_error' must be True or False, not int$"):
        M.cdf(A([0.0, 0.0]), return_error=1)
    with pytest.raises(TypeError, match=r"^cdf\(\): 'x' must be real, not complex$"):
        S.Normal().cdf(1j)
    with pytest.raises(TypeError, match=r"^pdf\(\): 'x' must be real, not a complex128 array$"):
        S.Normal().pdf(A([1j]))
    # Python's own call errors name the method alone, not a private class
    with pytest.raises(TypeError, match=r"^sample\(\) missing 1 required keyword-only argument: 'rng'$"):
        S.Normal().sample()
    with pytest.raises(TypeError, match=r"^sample\(\) missing 1 required keyword-only argument: 'rng'$"):
        M.sample()
    with pytest.raises(TypeError, match=r"^sample\(\) missing 1 required keyword-only argument: 'rng'$"):
        S.Wishart(df=3.0, scale=A([[1.0]])).sample()
    with pytest.raises(TypeError, match=r"^mean\(\) got an unexpected keyword argument 'out'$"):
        S.Normal().mean(out=None)
    with pytest.raises(TypeError, match=r"^sample\(\): 'shape' must be an int or a tuple of ints, not str$"):
        S.Normal().sample("a", rng=sd.random.default_rng(0))


@pytest.mark.parametrize("x", [2**63, 2**64, -(2**63) - 1, 10**400])
def test_large_int_input_overflows_as_special(x):
    for m in ("cdf", "pdf", "logccdf", "icdf", "ilogcdf"):
        with pytest.raises(OverflowError, match=rf"^{m}\(\): Python int too large to convert to int64$"):
            getattr(S.Normal(), m)(x)
    with pytest.raises(OverflowError, match=r"^pmf\(\): Python int too large to convert to int64$"):
        S.Poisson(mu=1.0).pmf(x)
    with pytest.raises(OverflowError, match=r"^ndtr\(\): Python int too large to convert to int64$"):
        sp.ndtr(x)
    # int64 itself is a number; parameters keep their rule (a double)
    assert S.Normal().cdf(2**63 - 1) == 1.0
    assert S.Normal(mu=2**64).cdf(0.0) == 0.0


# ---- the beta family's deep inverse ends in bounded time


def test_beta_deep_inverse_converges_fast():
    X = S.Beta(a=1.16e5, b=0.063)
    t0 = time.perf_counter()
    x = X.ilogcdf(-1627.7)
    dt = time.perf_counter() - t0
    assert 0.0 < x < 1.0 and abs(X.logcdf(x) / -1627.7 - 1.0) < 1e-13
    assert dt < 0.05
    for a, b, lp in ((3.0, 0.5, -1e3), (0.2, 40.0, -100.0), (1e5, 1e-3, -2e3), (2e4, 7.0, -5e4)):
        x = S.Beta(a=a, b=b).ilogcdf(lp)
        assert abs(S.Beta(a=a, b=b).logcdf(x) / lp - 1.0) < 1e-12


def test_t_deep_inverse_near_the_centre():
    # p below 2^-1000 at a large df: the root's x = df / (df + t^2) is near 1, solved in log(1 - x)
    for df, p in ((9.959e5, 5.113e-305), (9.758e5, 3.148e-319), (1e6, 1e-310)):
        X = S.StudentT(df=df)
        t = X.icdf(p)
        assert abs(X.logcdf(t) / math.log(p) - 1.0) < 1e-15
        assert X.iccdf(p) == -t
        assert abs(X.ilogcdf(math.log(p)) / t - 1.0) < 1e-15


@pytest.mark.parametrize("df", _DFS)
def test_noncentral_t_at_nc_zero(df):
    _same_everywhere(S.NoncentralT(df=df, nc=0.0), S.StudentT(df=df))


@pytest.mark.parametrize("df", _DFS[:-1])
def test_noncentral_chi_square_at_nc_zero(df):
    _same_everywhere(S.NoncentralChiSquared(df=df, nc=0.0), S.ChiSquared(df=df))


@pytest.mark.deep
@pytest.mark.parametrize("dfn", (0.01, 2.0, 3.0, 1e5, 1e55, math.inf))
@pytest.mark.parametrize("dfd", (0.5, 3.0, 5.0, 6.5, 9.0, 11.0, 1e45, math.inf))
def test_noncentral_f_at_nc_zero(dfn, dfd):
    _same_everywhere(S.NoncentralF(dfn=dfn, dfd=dfd, nc=0.0), S.F(dfn=dfn, dfd=dfd))


def test_noncentral_f_moments_were_off_by_a_bit():
    assert S.NoncentralF(dfn=3, dfd=11, nc=0).variance() == S.F(dfn=3, dfd=11).variance() == 1.707231040564374


@pytest.mark.parametrize(
    "call,msg",
    [
        (lambda: S.Normal(cool=1), r"Normal\(\): unexpected keyword argument 'cool'"),
        (lambda: S.Normal(1), r"Normal\(\): takes 0 positional arguments but 1 was given"),
        (lambda: S.Normal(1, 2), r"Normal\(\): takes 0 positional arguments but 2 were given"),
        (lambda: S.Gamma(), r"Gamma\(\): missing 1 required keyword-only argument: 'a'"),
        (lambda: S.Binomial(n=3, p=0.5, k=2), r"Binomial\(\): unexpected keyword argument 'k'"),
        (
            lambda: S.MultivariateNormal(mu=A([0.0, 0.0])),
            r"MultivariateNormal\(\): missing 1 required keyword-only argument: 'cov'",
        ),
        (lambda: S.Dirichlet(A([1.0, 1.0])), r"Dirichlet\(\): takes 0 positional arguments but 1 was given"),
        (lambda: S.Wishart(df=3.0, scale=A(np.eye(2)), q=1), r"Wishart\(\): unexpected keyword argument 'q'"),
        (lambda: S.InverseWishart(df=3.0), r"InverseWishart\(\): missing 1 required keyword-only argument: 'scale'"),
        (
            lambda: S.MatrixNormal(mu=A(np.zeros((2, 2)))),
            r"MatrixNormal\(\): missing 2 required keyword-only arguments",
        ),
    ],
)
def test_constructor_call_shape(call, msg):
    with pytest.raises(TypeError, match="^" + msg):
        call()


def test_constructor_messages_kept():
    # the parameters' own checks, the base classes, the signature
    with pytest.raises(TypeError, match=r"^Normal\(\): 'mu' must be a real number or an sd.ndarray, not str$"):
        S.Normal(mu="a")
    with pytest.raises(TypeError, match=r"^ContinuousDistribution\(\): a base class"):
        S.ContinuousDistribution(1)
    assert str(inspect.signature(S.Normal)) == "(*, mu=0.0, sigma=1.0)"
    assert str(inspect.signature(S.Wishart)) == "(*, df, scale)"


@pytest.mark.parametrize("df", [1e-4, 1e-3, 1e6, 1.1e6, 1e9, math.inf])
def test_student_t_median_on_the_log_side(df):
    X = S.StudentT(df=df)
    for v in (X.ilogcdf(math.log(0.5)), X.ilogccdf(math.log(0.5)), X.icdf(0.5), X.iccdf(0.5)):
        assert v == 0.0 and math.copysign(1.0, v) == 1.0, (df, v)
    assert S.NoncentralT(df=2e6, nc=0.0).ilogcdf(math.log(0.5)) == 0.0


def test_signed_zeros():
    def pos0(v):
        return v == 0.0 and math.copysign(1.0, v) == 1.0

    assert pos0(S.StudentT(df=5.0).iccdf(0.5)) and pos0(S.NoncentralT(df=5.0, nc=0.0).iccdf(0.5))
    assert pos0(S.StudentT(df=5.0).ilogccdf(math.log(0.5)))
    P = S.Poisson(mu=-0.0)
    for v in (P.variance(), P.standard_deviation(), P.mean(), P.median(), S.Binomial(n=3.0, p=-0.0).variance()):
        assert pos0(v), v
    assert pos0(S.Bernoulli(p=-0.0).mean())
    # a support on the whole line keeps the sign of the location
    assert math.copysign(1.0, S.Normal(mu=-0.0).mean()) == -1.0


# ---- ST-3, CX-3: numpy scalars -----------------------------------------------------------


def test_numpy_scalar_parameters():
    assert repr(S.Normal(mu=np.int64(1))) == "Normal(mu=1.0, sigma=1.0)"
    assert repr(S.Normal(mu=np.float64(2.0))) == "Normal(mu=2.0, sigma=1.0)"
    assert S.Normal(mu=np.float32(1.5)).mean() == 1.5
    assert S.Binomial(n=np.int64(10), p=0.3).mean() == S.Binomial(n=10, p=0.3).mean()
    assert S.Normal(mu=True).mu is True and S.Normal(mu=np.bool_(True)).mu is True
    assert S.Wishart(df=np.int64(3), scale=sd.eye(2)).mean().tolist() == [[3.0, 0.0], [0.0, 3.0]]
    assert S.Multinomial(n=np.int64(3), p=A([0.5, 0.5])).mean().tolist() == [1.5, 1.5]
    assert math.isnan(S.skew(np.int64(3))) and math.isnan(S.skew(np.array(2.0)))
    with pytest.raises(TypeError, match=r"^Normal\(\): 'mu' must be a real number or an sd.ndarray, not complex$"):
        S.Normal(mu=1j)


def test_beta_deep_logcdf_next_to_one():
    ctx.prec = 2500
    for a, b, x in ((1.0, 1e-300, 0.95), (1.0, 1e-300, 1 - 1e-10), (2.0, 1e-306, 0.9000001), (0.5, 1e-295, 0.99)):
        got = S.Beta(a=a, b=b).logcdf(x)
        want = float(arb(x).beta_lower(arb(a), arb(b), 1).log().mid())
        assert abs(got - want) <= 1e-14 * abs(want), (a, b, x, got, want)
        # the upper side by symmetry
        assert S.Beta(a=b, b=a).logccdf(1.0 - x) == pytest.approx(got, rel=1e-13)
    assert S.Beta(a=1.0, b=1e-300).logcdf(0.95) == math.log(S.Beta(a=1.0, b=1e-300).cdf(0.95))
    # where the tail is subnormal the log is still NaN
    assert math.isnan(S.Beta(a=1e3, b=2.0**-1021).logcdf(0.95))
    # an inverse there: the forward check now has the log tail (Arb: the closest double)
    X = S.Beta(a=1e5, b=1e-10)
    r = X.icdf(1e-300)
    assert r == 0.9934092416074045
    assert abs(float(arb(r).beta_lower(arb(1e5), arb(1e-10), 1).mid()) / 1e-300 - 1.0) < 1e-11


def test_parameters_are_read_only_copies():
    mu, cov = sd.zeros(2), sd.eye(2)
    X = S.MultivariateNormal(mu=mu, cov=cov)
    mu[0], cov[0, 0] = 3.0, 100.0
    assert X.mean().tolist() == [0.0, 0.0] and X.cov.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert X.covariance().tolist() == X.cov.tolist() and X.logpdf(sd.zeros(2)) == pytest.approx(
        -math.log(2 * math.pi), rel=1e-15
    )
    a = A([2.0])
    G = S.Gamma(a=a)
    a[0] = -1.0
    assert G.mean().tolist() == [2.0] and G.a.tolist() == [2.0]
    for setter in (lambda: G.a.__setitem__(0, 1.0), lambda: X.mu.__setitem__(0, 1.0)):
        with pytest.raises(ValueError, match="read-only"):
            setter()
    # dtype as given (an int64 parameter stays int64)
    assert S.Binomial(n=A([10, 20]), p=0.5).n.dtype == sd.int64


def test_summaries_are_read_only():
    G = S.Gamma(a=A([2.0, 3.0]))
    lo, hi = G.support()
    summaries = [G.mean(), G.variance(), G.standard_deviation(), G.skewness(), G.kurtosis(), G.median(), G.mode()]
    summaries += [G.entropy(), lo, hi]
    M = S.MultivariateT(mu=sd.zeros((2, 3)), shape=sd.eye(3), df=5.0)
    summaries += [M.mean(), M.covariance(), M.entropy(), S.Dirichlet(alpha=A([1.0, 2.0])).covariance()]
    W = S.Wishart(df=A([3.0, 4.0]), scale=sd.eye(2))
    summaries += [W.mean(), W.mode(), W.variance(), W.entropy()]
    for s in summaries:
        with pytest.raises(ValueError, match="read-only"):
            s[(0,) * s.ndim] = 1.0
        assert s.copy().flags["WRITEABLE"] if hasattr(s, "flags") else True
    # the elementwise methods' results stay writable (as out=)
    c = G.cdf(A([1.0, 2.0]))
    c[0] = 0.5
    assert type(S.Gamma(a=2.0).mean()) is float
