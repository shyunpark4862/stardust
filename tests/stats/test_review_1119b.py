"""stats (2026-11-19, second pass): nc = 0 bit for bit with the central distributions, one
df boundary 2^-1021 for the values and the samples (t, noncentral t and F, MultivariateT),
rankdata's axis message, the constructors' call-shape errors (T36), numeric options, and
where the large-df Wishart density turns NaN (T35)."""

import inspect
import math
import re
import struct

import numpy as np
import pytest

import stardust as sd

S = sd.stats
A = sd.asarray
_IN = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf")
_INV = ("icdf", "iccdf")
_LOGINV = ("ilogcdf", "ilogccdf")
_PAR = ("mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy", "support")


def _bits(v):
    if isinstance(v, tuple):
        return tuple(_bits(u) for u in v)
    return "nan" if math.isnan(v) else struct.pack("<d", v)


def _same_everywhere(a, b):
    for m in _IN:
        for x in (-3.0, -0.5, 0.0, 0.3, 1.0, 2.5, 40.0, 1e10):
            assert _bits(getattr(a, m)(x)) == _bits(getattr(b, m)(x)), (a, m, x)
    for m in _INV:
        for p in (1e-300, 1e-10, 0.1, 0.5, 0.9, 1 - 1e-12):
            assert _bits(getattr(a, m)(p)) == _bits(getattr(b, m)(p)), (a, m, p)
    for m in _LOGINV:
        for lp in (-1e5, -30.0, -0.7, -1e-8):
            assert _bits(getattr(a, m)(lp)) == _bits(getattr(b, m)(lp)), (a, m, lp)
    for m in _PAR:
        assert _bits(getattr(a, m)()) == _bits(getattr(b, m)()), (a, m)
    sa = a.sample(6, rng=sd.random.default_rng(3)).tolist()
    sb = b.sample(6, rng=sd.random.default_rng(3)).tolist()
    assert [_bits(v) for v in sa] == [_bits(v) for v in sb], a


_DFS = (0.01, 0.5, 1.0, 2.5, 3.0, 4.5, 6.5, 9.0, 11.0, 1e3, 1e17, 1e45, math.inf)


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


_TINY = (5e-324, 1e-310, 2.0**-1022, math.nextafter(2.0**-1021, 0.0))


@pytest.mark.parametrize("df", _TINY)
def test_t_family_below_the_half_boundary(df):
    for D in (S.StudentT(df=df), S.NoncentralT(df=df, nc=1.0), S.NoncentralT(df=df, nc=0.0)):
        for m in _IN:
            assert math.isnan(getattr(D, m)(-1.0)), (D, m)
        for m in _INV + _LOGINV:
            assert math.isnan(getattr(D, m)(0.3 if m in _INV else -1.0)), (D, m)
        assert math.isnan(D.median()) and math.isnan(D.entropy()), D
        assert all(math.isnan(v) for v in D.sample(2, rng=sd.random.default_rng(1)).tolist()), D
    # either df below: the noncentral F's values and samples (user decision 2026-11-19), nc = 0 too
    for nc in (1.0, 0.0):
        for D in (
            S.NoncentralF(dfn=df, dfd=3.0, nc=nc),
            S.NoncentralF(dfn=3.0, dfd=df, nc=nc),
            S.NoncentralF(dfn=df, dfd=df, nc=nc),
        ):
            for m in _IN:
                assert math.isnan(getattr(D, m)(1.0)), (D, m)
            assert math.isnan(D.icdf(0.3)) and math.isnan(D.ilogccdf(-1.0)) and math.isnan(D.median()), D
            assert all(math.isnan(v) for v in D.sample(3, rng=sd.random.default_rng(1)).tolist()), D
    # an infinite other df is the limit distribution's (its sample too)
    assert math.isfinite(S.NoncentralF(dfn=3.0, dfd=math.inf, nc=1.0).sample(rng=sd.random.default_rng(1)))


def test_noncentral_chi_square_small_df_keeps_values():
    # a Poisson mixture with a point mass at 0 (user decision 2026-11-19: the intended exception)
    X = S.NoncentralChiSquared(df=2.0**-1022, nc=1.0)
    assert X.cdf(1.0) == pytest.approx(0.7328798037968203, rel=1e-14)
    # (cdf(0) holds the point mass, T46: test_review_1119c.py)
    assert X.cdf(0.0) == math.exp(-0.5) and math.isfinite(X.pdf(1.0)) and math.isfinite(X.icdf(0.9))
    assert math.isnan(S.NoncentralChiSquared(df=2.0**-1022, nc=0.0).cdf(1.0))  # ChiSquared's


def test_noncentral_f_samples_at_the_half_boundary():
    df = 2.0**-1021
    v = S.NoncentralF(dfn=df, dfd=3.0, nc=1.0).sample(3, rng=sd.random.default_rng(1)).tolist()
    assert not any(math.isnan(u) for u in v)


def test_t_family_at_the_half_boundary():
    df = 2.0**-1021
    assert S.StudentT(df=df).cdf(-1.0) == 0.5 and S.StudentT(df=df).median() == 0.0
    assert S.StudentT(df=df).sample(2, rng=sd.random.default_rng(1)).tolist() == [math.inf, math.inf]
    assert math.isfinite(S.StudentT(df=df).entropy())
    # above the boundary the noncentral t's median is +inf: T has the sign of Z + nc and an
    # unbounded size (P(T <= t) -> Phi(-nc) < 1/2 for every finite t)
    T = S.NoncentralT(df=1e-300, nc=1.0)
    assert T.median() == math.inf and T.cdf(-1.0) == pytest.approx(0.15865525393146, rel=1e-12)


def test_rankdata_axis_message():
    x = sd.ones((3, 4))
    for ax, name in (((0,), "tuple"), ([0], "list"), (1.5, "float"), (True, "bool"), ("a", "str")):
        with pytest.raises(TypeError, match=rf"^rankdata\(\): 'axis' must be an int or None, not {name}$"):
            S.rankdata(x, axis=ax)
    assert S.rankdata(x, axis=np.int64(1)).shape == (3, 4)
    with pytest.raises(ValueError, match=r"^rankdata\(\): axis 5 is out of bounds"):
        S.rankdata(x, axis=5)
    # the reductions keep theirs (they take tuples)
    with pytest.raises(TypeError, match=r"^skew\(\): 'axis' must be an int or a tuple or list of ints, not float$"):
        S.skew(x, axis=1.5)


@pytest.mark.parametrize(
    "call,msg",
    [
        (lambda: S.Normal(cool=1), r"Normal\(\): unexpected keyword argument 'cool'"),
        (lambda: S.Normal(1), r"Normal\(\): takes no positional arguments but 1 was given"),
        (lambda: S.Normal(1, 2), r"Normal\(\): takes no positional arguments but 2 were given"),
        (lambda: S.Gamma(), r"Gamma\(\): missing 1 required keyword-only argument: 'a'"),
        (lambda: S.Binomial(n=3, p=0.5, k=2), r"Binomial\(\): unexpected keyword argument 'k'"),
        (
            lambda: S.MultivariateNormal(mu=A([0.0, 0.0])),
            r"MultivariateNormal\(\): missing 1 required keyword-only argument: 'cov'",
        ),
        (lambda: S.Dirichlet(A([1.0, 1.0])), r"Dirichlet\(\): takes no positional arguments but 1 was given"),
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


def test_mvcdf_atol():
    M = S.MultivariateNormal(mu=A([0.0, 0.0]), cov=A(np.eye(2)))
    x = A([0.0, 0.0])
    for bad in (math.inf, -1.0, 0, math.nan):
        with pytest.raises(ValueError, match=r"^cdf\(\): 'atol' must be a finite number > 0$"):
            M.cdf(x, atol=bad)
    with pytest.raises(OverflowError, match=r"^cdf\(\): 'atol' is out of range$"):
        M.cdf(x, atol=10**400)
    with pytest.raises(TypeError, match=r"^cdf\(\): 'atol' must be an int or a float, not bool$"):
        M.cdf(x, atol=True)
    assert M.cdf(x, atol=np.int64(1)) == M.cdf(x, atol=1) == 0.25


def test_descriptive_numeric_options():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert S.sem(x, correction=np.int64(2)) == S.sem(x, correction=2)
    assert S.entropy(x, base=np.int64(2)) == S.entropy(x, base=2)
    assert S.iqr(x, scale=np.int64(2)) == S.iqr(x, scale=2)
    assert S.moment(x, order=2, center=np.int64(2)) == S.moment(x, order=2, center=2)
    for f, arg in (
        (lambda v: S.sem(x, correction=v), "sem(): 'correction'"),
        (lambda v: S.entropy(x, base=v), "entropy(): 'base'"),
        (lambda v: S.iqr(x, scale=v), "iqr(): 'scale'"),
        (lambda v: S.moment(x, order=2, center=v), "moment(): 'center'"),
    ):
        with pytest.raises(OverflowError, match="^" + re.escape(arg) + " is out of range$"):
            f(10**400)


def _spd(p, kappa, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    v = (q * np.logspace(0, math.log10(kappa), p)) @ q.T
    return A((v + v.T) / 2)


def test_wishart_large_df_has_values_past_the_guarantee():
    # W - I is formed from B - n A exactly and log|W| - tr W + p has no rounding of size
    # |W - I|: the bound stays below 1e-14 |value| near the mode at any df, so no NaN
    for D in (S.Wishart, S.InverseWishart):
        X = D(df=1e40, scale=A(np.eye(2)))
        assert math.isfinite(X.logpdf(X.mode())) and math.isfinite(X.logpdf(X.mean()))
    # one ULP off the mean (W - I ~ 1e-16), and as far as 1e100
    for df in (1e33, 1e35, 1e100):
        X = S.Wishart(df=df, scale=A(np.eye(2)))
        m = np.nextafter(np.asarray(X.mean().tolist()), np.inf)
        assert math.isfinite(X.logpdf(A((m + m.T) / 2))), df
    # the mode, the mean and samples, inside the guarantee (df <= 1e30) and past it
    for D in (S.Wishart, S.InverseWishart):
        for p, kappa in ((2, 1e2), (3, 1e4), (10, 1e4), (5, 1e9)):
            for df in (1e30, 1e40):
                X = D(df=df, scale=_spd(p, kappa, 1))
                for pt in (X.mode(), X.mean(), X.sample(rng=sd.random.default_rng(1))):
                    assert math.isfinite(X.logpdf(pt)), (D, p, kappa, df)
