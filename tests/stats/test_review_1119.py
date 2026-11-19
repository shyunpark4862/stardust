"""sd.stats: the reported anomalies and the user decisions of 2026-11-19 (df = inf as a limit,
the family's support, NegativeBinomial(n=0), lists refused, constructor messages, large
int inputs, sample(shape=None), Wishart at a large df, ...)."""

import math
import time

import numpy as np
import pytest
from flint import arb, arb_mat, ctx

import stardust as sd

S = sd.stats
sp = sd.special
A = sd.asarray
inf = math.inf

INPUT = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf")
PARAM = ("mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy")


def _same(a, b, rel=0.0):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b or abs(a - b) <= rel * max(1.0, abs(b))


def _agree(X, Y, xs, rel):
    for m in INPUT:
        for x in xs:
            assert _same(getattr(X, m)(x), getattr(Y, m)(x), rel), (m, x)
            assert float(getattr(X, m)(A([x, x]))[1]) == getattr(X, m)(x)  # the chunk path too
    for m in ("icdf", "iccdf"):
        for p in (1e-300, 0.1, 0.5, 0.9):
            assert _same(getattr(X, m)(p), getattr(Y, m)(p), rel), (m, p)
    for m in ("ilogcdf", "ilogccdf"):
        for lp in (-1e3, -2.0, -0.1):
            assert _same(getattr(X, m)(lp), getattr(Y, m)(lp), rel), (m, lp)
    for m in PARAM:
        assert _same(getattr(X, m)(), getattr(Y, m)(), rel), m
    assert X.support() == Y.support() or rel


# ---- S4: an infinite degree of freedom is the limit (user decision 2026-11-19)


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


# ---- S3, S5, S7b, S19


def test_subnormal_df_samples_and_median():
    rng = sd.random.default_rng(0)
    for X in (
        S.StudentT(df=1e-310),
        S.StudentT(df=5e-324),
        S.NoncentralT(df=1e-310, nc=1.0),
        S.NoncentralF(dfn=1e-310, dfd=1e-310, nc=1.0),
    ):
        assert np.isnan(np.asarray(X.sample(8, rng=rng))).all()
    assert math.isnan(S.StudentT(df=1e-310).median())
    assert S.StudentT(df=1e-300).median() == 0.0


def test_f_kurtosis_diverges_where_the_variance_is_finite():
    for d in (4.5, 5.0, 6.0, 6.5, 8.0):
        assert S.F(dfn=5.0, dfd=d).kurtosis() == inf
    assert math.isnan(S.F(dfn=5.0, dfd=4.0).kurtosis())
    assert math.isfinite(S.F(dfn=5.0, dfd=9.0).kurtosis())


def test_binomial_mode_at_2_52():
    assert S.Binomial(n=2.0**52, p=0.5).mode() == 2.0**51
    assert math.isnan(S.Binomial(n=2.0**52 + 2, p=0.5).mode())


def test_noncentral_chi_square_nc0_moments_are_the_central_ones():
    for df in (0.5, 3.0, 17.0):
        X, Y = S.NoncentralChiSquared(df=df, nc=0.0), S.ChiSquared(df=df)
        assert (X.skewness(), X.kurtosis(), X.mean(), X.variance()) == (
            Y.skewness(),
            Y.kurtosis(),
            Y.mean(),
            Y.variance(),
        )


# ---- S6: the family's support; S7a: NegativeBinomial(n=0)


def test_support_is_the_familys():
    cases = [
        (S.Binomial(n=10, p=0.0), (0.0, 10.0)),
        (S.Binomial(n=10, p=1.0), (0.0, 10.0)),
        (S.Bernoulli(p=0.0), (0.0, 1.0)),
        (S.Poisson(mu=0.0), (0.0, inf)),
        (S.Geometric(p=1.0), (1.0, inf)),
        (S.NegativeBinomial(n=3.0, p=1.0), (0.0, inf)),
        (S.NegativeBinomial(n=0.0, p=0.4), (0.0, inf)),
    ]
    for X, sup in cases:
        assert X.support() == sup
        assert X.icdf(1.0) == sup[1] and X.iccdf(0.0) == sup[1]
        assert X.icdf(0.0) == sup[0] and X.iccdf(1.0) == sup[0]
    # inside (0, 1) the quantile is the point mass
    assert S.Poisson(mu=0.0).icdf(0.5) == 0.0 and S.Geometric(p=1.0).iccdf(0.3) == 1.0
    # as scipy's support and ppf(1)
    assert S.Poisson(mu=0.0).icdf(1.0) == inf


def test_negative_binomial_n0_is_the_point_mass():
    X = S.NegativeBinomial(n=0.0, p=0.5)
    assert (X.pmf(0.0), X.pmf(1.0), X.logpmf(0.0), X.cdf(2.0), X.ccdf(2.0)) == (1.0, 0.0, 0.0, 1.0, 0.0)
    assert X.cdf(2.0) == sp.nbdtr(2, 0.0, 0.5)
    assert (X.mean(), X.variance(), X.median(), X.mode(), X.entropy()) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isnan(X.skewness()) and math.isnan(X.kurtosis())
    assert list(np.asarray(X.sample(5, rng=sd.random.default_rng(1)))) == [0.0] * 5
    assert float(S.NegativeBinomial(n=A([0.0, 2.0]), p=0.5).pmf(A([0.0, 0.0]))[0]) == 1.0
    # p = 0 and an infinite mean are not distributions: NaN (special's values are the limits)
    assert math.isnan(S.NegativeBinomial(n=3.0, p=0.0).cdf(2.0)) and sp.nbdtr(2, 3, 0.0) == 0.0
    assert math.isnan(S.Poisson(mu=inf).cdf(2.0)) and sp.pdtr(2, inf) == 0.0


# ---- S2: the noncentral NaN boundaries (docs/api/stats.md §2.4)


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


# ---- S8, S9


def test_summaries_read_the_lower_triangle():
    C = A([[1.0, 99.0], [0.5, 1.0]])
    sym = [[1.0, 0.5], [0.5, 1.0]]
    assert np.asarray(S.MultivariateNormal(mu=A([0.0, 0.0]), cov=C).covariance()).tolist() == sym
    t = np.asarray(S.MultivariateT(mu=A([0.0, 0.0]), shape=C, df=5.0).covariance())
    assert t[0, 1] == t[1, 0]
    W = S.Wishart(df=5.0, scale=C)
    for m in (W.mean(), W.mode(), W.var(), S.InverseWishart(df=5.0, scale=C).mean()):
        v = np.asarray(m)
        assert v[0, 1] == v[1, 0]
    assert np.asarray(W.mean()).tolist() == [[5.0, 2.5], [2.5, 5.0]]


def test_multinomial_point_mass_entropy_and_tiny_inverse_wishart():
    assert S.Multinomial(n=0, p=A([0.3, 0.7])).entropy() == 0.0
    assert S.Multinomial(n=0, p=A([0.2, 0.3, 0.5])).entropy() == 0.0
    x = A([[1e-310, 0.0], [0.0, 1.0]])
    assert S.InverseWishart(df=3.0, scale=A([[1.0, 0.0], [0.0, 1.0]])).logpdf(x) == -inf
    assert S.InverseWishart(df=2.0, scale=A([[2.0]])).logpdf(A([[1e-310]])) == S.InverseGamma(a=1.0, scale=1.0).logpdf(
        1e-310
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: S.MultivariateNormal(mu=[0.0, 0.0], cov=A([[1.0, 0.0], [0.0, 1.0]])),
        lambda: S.MultivariateNormal(mu=A([0.0, 0.0]), cov=[[1.0, 0.0], [0.0, 1.0]]),
        lambda: S.MultivariateT(mu=A([0.0]), shape=A([[1.0]]), df=[3.0]),
        lambda: S.Multinomial(n=3, p=[0.5, 0.5]),
        lambda: S.Dirichlet(alpha=[1.0, 2.0]),
        lambda: S.Wishart(df=3.0, scale=[[1.0]]),
        lambda: S.InverseWishart(df=3.0, scale=np.eye(2)),
        lambda: S.MatrixNormal(mu=[[0.0]], rowcov=A([[1.0]]), colcov=A([[1.0]])),
    ],
)
def test_matrix_and_multivariate_parameters_refuse_lists(make):
    with pytest.raises(
        TypeError, match=r"^[A-Za-z]+\(\): '\w+' must be (a real number or )?an sd.ndarray, not (list|numpy.ndarray)$"
    ):
        make()


# ---- S10: Wishart and the inverse Wishart at a large df


def _wishart_ref(df, V, X, inverse):
    ctx.prec = 400
    p = len(V)
    n = arb(df)
    Va = arb_mat([[arb(float(v)) for v in r] for r in V])
    Xa = arb_mat([[arb(float(v)) for v in r] for r in X])
    lmg = arb(p * (p - 1)) / 4 * arb.pi().log() + sum(((n - j) / 2).lgamma() for j in range(p))
    if not inverse:
        tr = Va.inv() * Xa
        t = sum((tr[i, i] for i in range(p)), arb(0))
        v = (n - p - 1) / 2 * Xa.det().log() - t / 2 - n * p / 2 * arb(2).log() - n / 2 * Va.det().log() - lmg
    else:
        tr = Va * Xa.inv()
        t = sum((tr[i, i] for i in range(p)), arb(0))
        v = n / 2 * Va.det().log() - n * p / 2 * arb(2).log() - lmg - (n + p + 1) / 2 * Xa.det().log() - t / 2
    return v


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("df", [1e9, 1e12, 1e15, 1e17])
def test_wishart_large_df_against_arb(inverse, df):
    V = np.array([[1.0, 0.3, 0.1], [0.3, 2.0, -0.2], [0.1, -0.2, 0.5]])
    for scale in (1.0, 1.001, 0.97):
        X = (V / (df + 4) if inverse else (df - 4) * V) * scale
        X[0, 1] = X[1, 0] = X[0, 1] * (1.0 + 1e-6)
        Dist = S.InverseWishart if inverse else S.Wishart
        v = Dist(df=df, scale=A(V)).logpdf(A(X))
        ref = _wishart_ref(df, V, X, inverse)
        err = float((abs(arb(v) - ref) / max(arb(1), abs(ref))).mid())
        assert err <= 1e-14, (scale, v, err)


@pytest.mark.parametrize("df", [1e10, 1e13, 1e15, 1e17])
def test_wishart_p1_is_gamma(df):
    for x in (df - 2.0, df * 0.999, df * 1.001):
        w = S.Wishart(df=df, scale=A([[1.0]])).logpdf(A([[x]]))
        g = S.Gamma(a=df / 2, scale=2.0).logpdf(x)
        assert abs(w - g) <= 2e-15 * max(1.0, abs(g)), (x, w, g)
        iw = S.InverseWishart(df=df, scale=A([[1.0]])).logpdf(A([[1.0 / x]]))
        ig = S.InverseGamma(a=df / 2, scale=0.5).logpdf(1.0 / x)
        assert abs(iw - ig) <= 2e-15 * max(1.0, abs(ig)), (x, iw, ig)


# ---- S12: messages


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
    with pytest.raises(ValueError, match=r"^Wishart\(\): the parameters could not be broadcast together"):
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
    with pytest.raises(ValueError, match=r"^cdf\(\): 'max_points' must be positive$"):
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


# ---- S13: descriptive statistics


def test_descriptive_edges_as_scipy():
    x = A([1.0, 2.0, 3.0, 4.0, 5.0])
    assert S.iqr(x, scale=0) == inf
    assert math.isnan(S.iqr(A([1.0, 1.0, 1.0]), scale=0))
    assert S.entropy(A([0.2, 0.8]), base=1) == inf
    assert S.differential_entropy(x, base=1) == inf
    assert S.differential_entropy(A([1.0, inf, 3.0, 4.0, 5.0])) == inf
    with pytest.raises(ValueError, match=r"^iqr\(\): 'method' must be "):
        S.iqr(x, method="foo")
    with pytest.raises(ValueError, match=r"^moment\(\): 'order' must not be empty$"):
        S.moment(x, order=[])
    with pytest.raises(TypeError, match=r"^entropy\(\): 'base' must be a positive number or None, not str$"):
        S.entropy(A([0.2, 0.8]), base="e")
    with pytest.raises(
        ValueError,
        match=r"^differential_entropy\(\): 'method' must be 'vasicek', 'van es', 'ebrahimi', 'correa' or 'auto', not 'VASICEK'$",
    ):
        S.differential_entropy(x, method="VASICEK")


def test_axis_lists_and_repeated_axis_as_the_array():
    X = A(np.arange(24.0).reshape(2, 3, 4) ** 1.5)
    for fn in ("skew", "kurtosis", "sem", "iqr", "entropy", "differential_entropy"):
        f = getattr(S, fn)
        a, b = f(X, axis=[0, 2]), f(X, axis=(0, 2))
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), fn
        with pytest.raises(ValueError, match=rf"^{fn}\(\): repeated axis$"):
            f(X, axis=(1, 1))
    assert np.array_equal(np.asarray(S.zscore(X, axis=[2])), np.asarray(S.zscore(X, axis=2)))
    assert np.array_equal(np.asarray(S.describe(X, axis=[1, 2]).mean), np.asarray(S.describe(X, axis=(1, 2)).mean))
    assert np.array_equal(np.asarray(S.mode(X, axis=[0]).mode), np.asarray(S.mode(X, axis=0).mode))
    with pytest.raises(ValueError, match=r"^var\(\): repeated axis$"):
        sd.var(X, axis=(1, 1))
    with pytest.raises(TypeError, match=r"^skew\(\): 'axis' must be an int or a tuple or list of ints, not str$"):
        S.skew(X, axis="a")
    with pytest.raises(TypeError, match=r"^rankdata\(\): 'axis' must be an int or None, not list$"):
        S.rankdata(X, axis=[0])


# ---- S14: the beta family's deep inverse ends in bounded time


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


# ---- S15, S20


def test_entropy_doc_and_sample_shape():
    assert "Shannon" in S.Poisson.entropy.__doc__ and "differential" in S.Normal.entropy.__doc__
    rng = sd.random.default_rng(0)
    assert isinstance(S.Normal().sample(rng=rng), float)
    assert isinstance(S.Normal().sample(None, rng=rng), float)
    assert S.Normal(mu=A([0.0, 1.0])).sample(rng=rng).shape == (2,)
    assert S.Normal(mu=A([0.0, 1.0])).sample(3, rng=rng).shape == (3, 2)
    msg = r"^sample\(\): 'shape' must not be \(\) \(stardust has no 0-d arrays; None gives one sample\)$"
    for X in (
        S.Normal(),
        S.Poisson(mu=2.0),
        S.MultivariateNormal(mu=A([0.0]), cov=A([[1.0]])),
        S.Wishart(df=3.0, scale=A([[1.0]])),
        S.Dirichlet(alpha=A([1.0, 1.0])),
    ):
        with pytest.raises(ValueError, match=msg):
            X.sample((), rng=rng)
    M = S.MultivariateNormal(mu=A([0.0, 0.0]), cov=A([[1.0, 0.0], [0.0, 1.0]]))
    assert M.sample(rng=rng).shape == (2,)
    assert S.Wishart(df=3.0, scale=A([[1.0]])).sample(rng=rng).shape == (1, 1)
    # one sample is the first of a call's (the same stream element)
    r1, r2 = sd.random.default_rng(9), sd.random.default_rng(9)
    assert S.Gamma(a=2.0).sample(rng=r1) == float(S.Gamma(a=2.0).sample(1, rng=r2)[0])


# ---- 2026-11-19 follow-up: T39 (multivariate inputs), T41 (string options), T42 (t's deep corner)


@pytest.mark.parametrize("x", [[0.1, 0.2], (0.1, 0.2), np.array([0.1, 0.2])])
def test_multivariate_and_matrix_inputs_refuse_lists(x):
    M = S.MultivariateNormal(mu=A([0.0, 0.0]), cov=A([[1.0, 0.0], [0.0, 1.0]]))
    for m in ("pdf", "logpdf", "cdf"):
        with pytest.raises(TypeError, match=rf"^{m}\(\): 'x' must be an sd.ndarray, not (list|tuple|numpy.ndarray)$"):
            getattr(M, m)(x)
    with pytest.raises(TypeError, match=r"^logpmf\(\): 'x' must be an sd.ndarray, not"):
        S.Multinomial(n=3, p=A([0.5, 0.5])).logpmf(x)
    with pytest.raises(TypeError, match=r"^pdf\(\): 'x' must be an sd.ndarray, not"):
        S.Dirichlet(alpha=A([1.0, 2.0])).pdf(x)
    with pytest.raises(TypeError, match=r"^logpdf\(\): 'x' must be an sd.ndarray, not"):
        S.Wishart(df=3.0, scale=A([[1.0, 0.0], [0.0, 1.0]])).logpdf([[1.0, 0.0], [0.0, 1.0]])
    # descriptive statistics keep taking sequences (scipy's convenience; T40)
    assert S.skew([1.0, 2.0, 4.0]) == S.skew(A([1.0, 2.0, 4.0]))


def test_string_option_messages():
    x = A([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"^skew\(\): 'nan_policy' must be 'propagate', 'omit' or 'raise', not 'x'$"):
        S.skew(x, nan_policy="x")
    with pytest.raises(
        TypeError, match=r"^skew\(\): 'nan_policy' must be 'propagate', 'omit' or 'raise', not NoneType$"
    ):
        S.skew(x, nan_policy=None)
    with pytest.raises(
        ValueError,
        match=r"^rankdata\(\): 'method' must be 'average', 'min', 'max', 'dense' or 'ordinal', not 'Average'$",
    ):
        S.rankdata(x, method="Average")
    with pytest.raises(TypeError, match=r"^differential_entropy\(\): 'method' must be .* or 'auto', not int$"):
        S.differential_entropy(x, method=1)


def test_t_deep_inverse_near_the_centre():
    # p below 2^-1000 at a large df: the root's x = df / (df + t^2) is near 1, solved in log(1 - x)
    for df, p in ((9.959e5, 5.113e-305), (9.758e5, 3.148e-319), (1e6, 1e-310)):
        X = S.StudentT(df=df)
        t = X.icdf(p)
        assert abs(X.logcdf(t) / math.log(p) - 1.0) < 1e-15
        assert X.iccdf(p) == -t
        assert abs(X.ilogcdf(math.log(p)) / t - 1.0) < 1e-15
