"""Shapes below 2^-1021 (user decisions 2026-11-19, T49 and T50): F's samples are NaN as its
values; Beta and BetaBinomial are their shapes' limit at 0, the two atoms with
P(upper) = a / (a + b) whichever shape is below (one mass rounds to 0 when the other shape is
not small: a single point mass), in every method and the samples. Arb checks the atoms."""

import math

import numpy as np
import pytest
from flint import arb, ctx

import stardust as sd

S = sd.stats
T = 2.0**-1022  # below 2^-1021
METHODS_IN = ("cdf", "ccdf", "logcdf", "logccdf")


def _arb_cdf(x, a, b):
    ctx.prec = 300
    try:
        return float(arb(x).beta_lower(arb(a), arb(b), 1))
    finally:
        ctx.prec = 53


@pytest.mark.parametrize("dfn,dfd", [(T, 3.0), (3.0, T), (T, T), (5e-324, 0.5), (T, math.inf)])
def test_f_samples_nan_below_the_boundary(dfn, dfd):
    X = S.F(dfn=dfn, dfd=dfd)
    assert all(math.isnan(v) for v in X.sample(5, rng=sd.random.default_rng(1)).tolist())
    assert math.isnan(X.cdf(1.0))  # (the values, as before)


def test_f_samples_at_the_boundary_and_the_inverse_gamma_limit():
    v = S.F(dfn=2.0**-1021, dfd=3.0).sample(5, rng=sd.random.default_rng(1)).tolist()
    assert not any(math.isnan(u) for u in v)
    # dfn = inf with a small dfd is InverseGamma(dfd/2, dfd/2), which has values and samples
    X = S.F(dfn=math.inf, dfd=T)
    assert math.isfinite(X.pdf(0.5)) and not any(
        math.isnan(u) for u in X.sample(3, rng=sd.random.default_rng(1)).tolist()
    )


@pytest.mark.parametrize("a,b,q", [(5e-324, 2.0, 0.0), (5e-324, 1e10, 0.0), (2.0, 5e-324, 1.0), (1e10, 5e-324, 1.0)])
def test_beta_point_masses(a, b, q):
    # the upper (lower) mass a / (a + b) rounds to 0: exactly the single point mass
    X = S.Beta(a=a, b=b)
    at = q  # the atom: 0 (a below) or 1 (b below)
    assert X.pdf(at) == math.inf and X.logpdf(at) == math.inf and X.pdf(0.5) == 0.0 and X.pdf(1.0 - at) == 0.0
    assert X.cdf(0.5) == (1.0 if at == 0.0 else 0.0) and X.ccdf(0.5) == 1.0 - X.cdf(0.5)
    assert X.cdf(at) == 1.0 and X.cdf(-1.0) == 0.0 and X.ccdf(2.0) == 0.0
    assert X.icdf(0.0) == 0.0 and X.icdf(0.3) == at and X.icdf(0.9) == at and X.icdf(1.0) == 1.0  # T32 ends
    assert X.iccdf(1.0) == 0.0 and X.iccdf(0.3) == at and X.iccdf(0.0) == 1.0
    assert X.ilogcdf(-1.0) == at and X.ilogccdf(-1.0) == at
    assert X.mean() == at and X.variance() == 0.0 and X.standard_deviation() == 0.0
    assert math.isnan(X.skewness()) and math.isnan(X.kurtosis())
    assert X.median() == at and X.mode() == at and X.entropy() == -math.inf and X.support() == (0.0, 1.0)
    assert X.sample(6, rng=sd.random.default_rng(1)).tolist() == [at] * 6


def test_beta_tiny_second_atom_and_the_boundary():
    # the other shape not small: a mass of order a at 1 (P(1) = a / (a + b), here 2^-1023)
    X = S.Beta(a=T, b=2.0)
    assert X.ccdf(0.5) == T / 2 and X.mean() == T / 2 and X.variance() == T / 2 and X.pdf(1.0) == math.inf
    assert X.cdf(0.5) == 1.0 and X.icdf(0.9) == 0.0 and X.median() == 0.0 and X.mode() == 0.0  # b >= 1: T21
    assert X.sample(6, rng=sd.random.default_rng(1)).tolist() == [0.0] * 6
    # the boundary pair: a below, b at 2^-1021, P(1) = 1/3 (was the point mass at 0)
    X = S.Beta(a=T, b=2.0**-1021)
    assert X.cdf(0.5) == pytest.approx(2 / 3, rel=1e-15) and X.mean() == pytest.approx(1 / 3, rel=1e-15)
    assert X.variance() == pytest.approx(2 / 9, rel=1e-15) and math.isnan(X.mode())
    # continuous across 2^-1021: the limit just below against the Beta itself at 2^-1021
    below, at = math.nextafter(2.0**-1021, 0.0), 2.0**-1021
    for m in ("cdf", "ccdf"):
        lim, own = getattr(S.Beta(a=below, b=2 * at), m)(0.5), getattr(S.Beta(a=at, b=2 * at), m)(0.5)
        assert lim == pytest.approx(own, rel=1e-15), m
    assert S.Beta(a=below, b=2 * at).mean() == pytest.approx(S.Beta(a=at, b=2 * at).mean(), rel=1e-15)
    Y = S.BetaBinomial(n=6, a=T, b=2.0**-1021)
    assert Y.pmf(6) == pytest.approx(1 / 3, rel=1e-15) and Y.mean() == pytest.approx(2.0, rel=1e-15)
    assert S.BetaBinomial(n=6, a=below, b=2 * at).pmf(6) == pytest.approx(
        S.BetaBinomial(n=6, a=at, b=2 * at).pmf(6), rel=1e-15
    )


@pytest.mark.parametrize("a,b", [(1e-310, 3e-310), (5e-324, 1.5e-323), (T, 1e-310), (3e-320, T), (5e-324, 5e-324)])
def test_beta_two_atoms_against_arb(a, b):
    X = S.Beta(a=a, b=b)
    ctx.prec = 300
    try:
        q = float(arb(a) / (arb(a) + arb(b)))
        qc = float(arb(b) / (arb(a) + arb(b)))
        var = float(arb(a) * arb(b) / ((arb(a) + arb(b)) ** 2 * (arb(a) + arb(b) + 1)))
    finally:
        ctx.prec = 53
    # the law of the atoms is the distribution itself to the doubles: Arb's incomplete beta
    for x in (1e-300, 0.5):
        assert X.cdf(x) == pytest.approx(_arb_cdf(x, a, b), rel=1e-15)
    assert X.ccdf(0.999) == pytest.approx(_arb_cdf(0.001, b, a), rel=1e-15)  # I_x(a, b) = 1 - I_(1-x)(b, a)
    assert X.cdf(0.5) == pytest.approx(qc, rel=1e-15) and X.ccdf(0.5) == pytest.approx(q, rel=1e-15)
    assert X.logcdf(0.5) == pytest.approx(math.log1p(-q), rel=1e-14)
    assert X.mean() == pytest.approx(q, rel=1e-15) and X.variance() == pytest.approx(var, rel=4e-16)
    assert X.pdf(0.0) == X.pdf(1.0) == math.inf and X.pdf(0.5) == 0.0 and math.isnan(X.mode())
    assert X.skewness() == pytest.approx((qc - q) / math.sqrt(q * qc), rel=1e-14)
    assert X.icdf(qc) == 0.0 and X.icdf(min(1.0, qc + 1e-9)) == 1.0 and X.entropy() == -math.inf
    v = X.sample(4000, rng=sd.random.default_rng(1)).tolist()
    assert set(v) <= {0.0, 1.0} and abs(sum(v) / 4000 - q) < 0.03


def test_beta_binomial_limits():
    # point masses (the other mass rounds to 0): the Shannon entropy exactly 0 (was 8.8e-19)
    X = S.BetaBinomial(n=5, a=5e-324, b=2.0)
    assert X.pmf(0) == 1.0 and X.pmf(2) == 0.0 and X.entropy() == 0.0 and X.mean() == 0.0 and X.variance() == 0.0
    # a tiny second mass: its entropy -q log q - (1 - q) log1p(-q)
    X = S.BetaBinomial(n=5, a=T, b=2.0)
    q = T / 2
    assert X.pmf(5) == q and X.mean() == 5 * q and X.entropy() == pytest.approx(-q * math.log(q) + q, rel=1e-15)
    X = S.BetaBinomial(n=5, a=2.0, b=5e-324)
    assert X.pmf(5) == 1.0 and X.mean() == 5.0 and X.icdf(0.3) == 5.0 and X.median() == 5.0 and X.mode() == 5.0
    assert X.sample(4, rng=sd.random.default_rng(1)).tolist() == [5.0] * 4
    # two atoms 0 and n, P(n) = a / (a + b)
    X = S.BetaBinomial(n=7, a=1e-310, b=3e-310)
    assert X.pmf(7) == 0.25 and X.pmf(0) == 0.75 and X.pmf(3) == 0.0 and X.cdf(3) == 0.75 and X.ccdf(3) == 0.25
    assert X.mean() == 1.75 and X.variance() == 49 * 0.1875 and X.mode() == 0.0
    assert X.entropy() == pytest.approx(-(0.25 * math.log(0.25) + 0.75 * math.log(0.75)), rel=1e-15)
    assert X.icdf(0.75) == 0.0 and X.icdf(0.76) == 7.0 and X.iccdf(0.25) == 0.0 and X.iccdf(0.2) == 7.0
    assert X.support() == (0.0, 7.0)
    v = X.sample(4000, rng=sd.random.default_rng(1)).tolist()
    assert set(v) <= {0.0, 7.0} and abs(v.count(7.0) / 4000 - 0.25) < 0.03
    # a tie: no single mode (T25)
    assert math.isnan(S.BetaBinomial(n=4, a=5e-324, b=5e-324).mode())


# ---- degrees of freedom below 2^-1021: the t family, the noncentral chi-square's point mass

TINY = (5e-324, 1e-310, 2.0**-1022, math.nextafter(2.0**-1021, 0.0))  # below 2^-1021
DENSITY_METHODS = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf")
INVERSE_METHODS = ("icdf", "iccdf")
LOG_INVERSE_METHODS = ("ilogcdf", "ilogccdf")


def _arb_q(a, y):
    ctx.prec = 4000
    try:
        return arb(y).gamma_upper(arb(a), 1)
    finally:
        ctx.prec = 53


def _arb_q_rel(a, y, s, q):
    ctx.prec = 3000
    try:
        return (
            float(arb(y).gamma_upper(arb(a), 1) / arb(q)) - 1
            if s is None
            else float((arb(y) / arb(s)).gamma_upper(arb(a), 1) / arb(q)) - 1
        )
    finally:
        ctx.prec = 53


def _ref(x, nc):
    """The df = 0 limit (Arb): the j >= 1 terms' density, cdf with the point mass e^-mu, ccdf."""
    ctx.prec = 300
    try:
        mu = arb(nc) / 2
        y = arb(x) / 2
        p, term, q = (-mu).exp(), (-y).exp(), arb(0)
        f, F, Q = arb(0), (-mu).exp(), arb(0)
        for j in range(1, 400):
            p = p * mu / j
            q = q + term  # Q(j, y)
            f += p * term / 2
            Q += p * q
            F += p * (1 - q)
            term = term * y / j
        return float(f), float(F), float(Q)
    finally:
        ctx.prec = 53


sp = sd.special


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


@pytest.mark.parametrize("df", TINY)
def test_t_family_below_the_half_boundary(df):
    for D in (S.StudentT(df=df), S.NoncentralT(df=df, nc=1.0), S.NoncentralT(df=df, nc=0.0)):
        for m in DENSITY_METHODS:
            assert math.isnan(getattr(D, m)(-1.0)), (D, m)
        for m in INVERSE_METHODS + LOG_INVERSE_METHODS:
            assert math.isnan(getattr(D, m)(0.3 if m in INVERSE_METHODS else -1.0)), (D, m)
        assert math.isnan(D.median()) and math.isnan(D.entropy()), D
        assert all(math.isnan(v) for v in D.sample(2, rng=sd.random.default_rng(1)).tolist()), D
    # either df below: the noncentral F's values and samples (user decision 2026-11-19), nc = 0 too
    for nc in (1.0, 0.0):
        for D in (
            S.NoncentralF(dfn=df, dfd=3.0, nc=nc),
            S.NoncentralF(dfn=3.0, dfd=df, nc=nc),
            S.NoncentralF(dfn=df, dfd=df, nc=nc),
        ):
            for m in DENSITY_METHODS:
                assert math.isnan(getattr(D, m)(1.0)), (D, m)
            assert math.isnan(D.icdf(0.3)) and math.isnan(D.ilogccdf(-1.0)) and math.isnan(D.median()), D
            assert all(math.isnan(v) for v in D.sample(3, rng=sd.random.default_rng(1)).tolist()), D
    # an infinite other df is the limit distribution's (its sample too)
    assert math.isfinite(S.NoncentralF(dfn=3.0, dfd=math.inf, nc=1.0).sample(rng=sd.random.default_rng(1)))


def test_noncentral_chi_square_small_df_keeps_values():
    # a Poisson mixture with a point mass at 0 (user decision 2026-11-19: the intended exception)
    X = S.NoncentralChiSquared(df=2.0**-1022, nc=1.0)
    assert X.cdf(1.0) == pytest.approx(0.7328798037968203, rel=1e-14)
    # (cdf(0) holds the point mass, T46: test_point_mass_distribution)
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


@pytest.mark.parametrize("df", TINY)
@pytest.mark.parametrize("nc", [0.1, 1.0, 7.0])
def test_point_mass_distribution(df, nc):
    X = S.NoncentralChiSquared(df=df, nc=nc)
    m = math.exp(-0.5 * nc)
    # at 0 the point mass e^-mu (R72)
    assert X.cdf(0.0) == m and X.ccdf(0.0) == -math.expm1(-0.5 * nc)
    assert X.logcdf(0.0) == -0.5 * nc and X.logccdf(0.0) == pytest.approx(math.log(-math.expm1(-0.5 * nc)), rel=1e-15)
    assert X.cdf(-1.0) == 0.0 and X.ccdf(-1.0) == 1.0
    assert X.pdf(0.0) == math.inf and X.logpdf(0.0) == math.inf and X.mode() == 0.0
    # off 0: the df = 0 limit's values (the j >= 1 terms), against Arb
    for x in (5e-324, 1e-300, 1e-10, 0.5, 3.0, 60.0):
        f, F, Q = _ref(x, nc)
        assert X.pdf(x) == pytest.approx(f, rel=1e-14) and X.cdf(x) == pytest.approx(F, rel=1e-14)
        assert X.ccdf(x) == pytest.approx(Q, rel=1e-14)
        assert X.logccdf(x) == pytest.approx(math.log(Q), rel=1e-14, abs=1e-14)
        assert X.logcdf(x) == pytest.approx(math.log(F), rel=1e-14, abs=1e-14)
    # the inverses: a target inside the mass gives 0, beyond it the root
    assert X.icdf(0.0) == 0.0 and X.icdf(m) == 0.0 and X.icdf(0.5 * m) == 0.0 and X.icdf(1.0) == math.inf
    assert X.iccdf(1.0) == 0.0 and X.iccdf(-math.expm1(-0.5 * nc)) == 0.0 and X.iccdf(0.0) == math.inf
    assert X.ilogcdf(-0.5 * nc) == 0.0 and X.ilogcdf(-0.5 * nc - 1.0) == 0.0
    for p in (m + (1 - m) * 0.25, m + (1 - m) * 0.9):
        x = X.icdf(p)
        assert x > 0.0 and X.cdf(x) == pytest.approx(p, rel=1e-13)
        assert X.iccdf(1.0 - p) == pytest.approx(x, rel=1e-12)
        assert X.ilogcdf(math.log(p)) == pytest.approx(x, rel=1e-12)
    assert X.median() == (0.0 if m >= 0.5 else X.icdf(0.5))
    # no finite differential entropy (R73)
    assert X.entropy() == -math.inf
    # the moments are the mixture's
    assert X.mean() == df + nc and X.variance() == 2.0 * (df + 2.0 * nc)


@pytest.mark.parametrize("df", TINY)
def test_chi_square_samples_below_the_boundary(df):
    # R74: NaN as the values (was 0.0); nc > 0 is the point mass, its samples are values
    for D in (S.ChiSquared(df=df), S.NoncentralChiSquared(df=df, nc=0.0)):
        assert all(math.isnan(v) for v in D.sample(4, rng=sd.random.default_rng(1)).tolist()), D
        assert math.isnan(D.sample(rng=sd.random.default_rng(1)))
    v = S.NoncentralChiSquared(df=df, nc=1.0).sample(2000, rng=sd.random.default_rng(1)).tolist()
    assert all(u >= 0.0 for u in v) and 0.55 < sum(u == 0.0 for u in v) / 2000 < 0.66  # e^-1/2 = 0.61
    assert S.ChiSquared(df=2.0**-1021).sample(2, rng=sd.random.default_rng(1)).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("dfn,nc", [(2.0**-1021, 2.0), (1e-307, 1.0), (2.0**-1000, 2.0), (1e-300, 2.0)])
def test_noncentral_f_just_above_the_boundary(dfn, nc):
    # R69: the j = 0 term alone (its F part is 1 to rounding): e^-mu, at a dfn where the test of
    # the next term divided by dfn / 2 and gave NaN
    Y = S.NoncentralF(dfn=dfn, dfd=5.0, nc=nc)
    m = math.exp(-0.5 * nc)
    assert Y.cdf(1.0) == pytest.approx(m, rel=1e-15) and Y.ccdf(1.0) == pytest.approx(-math.expm1(-0.5 * nc), rel=1e-15)
    assert Y.logcdf(1.0) == pytest.approx(-0.5 * nc, rel=1e-15)
    assert sp.ncfdtr(dfn, 5.0, nc, 1.0) == pytest.approx(m, rel=1e-15)
    x = Y.icdf(0.5 * (1.0 + m))
    assert x > 1.0 and Y.cdf(x) == pytest.approx(0.5 * (1.0 + m), rel=1e-12)


@pytest.mark.deep
def test_inverse_gamma_subnormal_shape():
    # R87 (T51): the roots beyond the doubles are inf (were NaN), the finite ones solve the tail
    for a in (5e-324, 5e-311, 1e-308):
        X = S.InverseGamma(a=a, scale=a)
        assert X.median() == math.inf and X.icdf(0.9) == math.inf and X.iccdf(0.5) == math.inf
        assert X.ilogcdf(-1.0) == math.inf and X.iccdf(1.0) == 0.0
    for a, s, p in ((5e-311, 5e-311, 1e-310), (1e-308, 1e-308, 1e-310), (1e-320, 1.0, 1e-319)):
        x = S.InverseGamma(a=a, scale=s).icdf(p)
        assert 0.0 < x < math.inf and float(_arb_q(a, arb(s) / arb(x))) == pytest.approx(p, rel=1e-13)
    # the entropy 1/a + log(s/a) + 1 + euler below 2^-1000, against Arb
    ctx.prec = 4000
    try:
        for a, s in ((1e-308, 1e-308), (2.0**-1000, 1.0), (1e-200, 3.0)):
            A, Sv = arb(a), arb(s)
            want = float(A + (Sv * A.gamma()).log() - (1 + A) * A.digamma())
            assert S.InverseGamma(a=a, scale=s).entropy() == pytest.approx(want, rel=1e-15)
    finally:
        ctx.prec = 53
    assert S.InverseGamma(a=1e-310, scale=1e-310).entropy() == math.inf
    # F's dfn = inf limit is that inverse gamma
    F = S.F(dfn=math.inf, dfd=1e-310)
    assert F.median() == math.inf and F.icdf(0.9) == math.inf and F.entropy() == math.inf
    assert F.cdf(1.0) == pytest.approx(3.5695865517192284e-308, rel=1e-14)


# ---- small shapes' upper roots below 2^-960 (T52)


@pytest.mark.parametrize("a", [3e-7, 1e-8, 1e-100, 1e-310, 5e-324])
@pytest.mark.parametrize("k", [600, 665, 680, 700, 720, 740])
def test_gamma_upper_root_band(a, k):
    # the root e^(-k - euler): a normal or subnormal double, found (was NaN for k ~ 665-745)
    y = S.Gamma(a=a).iccdf(k * a)
    assert 0.0 < y < 1e-250
    if y >= 2.0**-1022:
        assert abs(_arb_q_rel(a, y, None, k * a)) <= 1e-14 * k  # kappa ~ k
    else:  # a subnormal root: its rounding, the tail's slope a e^-y / y over it
        assert abs(_arb_q_rel(a, y, None, k * a)) * k <= 2 * (2.0**-1074 / y)
    assert S.Gamma(a=a).iccdf(760 * a) == 0.0  # below half the smallest subnormal
    assert S.Gamma(a=a).ilogccdf(math.log(k * a)) == pytest.approx(y, rel=1e-12 * k)


def test_gamma_upper_root_with_a_scale():
    # the root below 2^-1074 itself, s y a double: from logs (0 or NaN before)
    for a, s, k in ((1e-8, 1e300, 800.0), (1e-100, 1e300, 900.0)):
        r = S.Gamma(a=a, scale=s).iccdf(k * a)
        assert 0.0 < r < 1e-40 and abs(_arb_q_rel(a, r, s, k * a)) <= 1e-14 * k
    # R102: InverseGamma's root inside the doubles, its gamma's below them
    X = S.InverseGamma(a=1e-100, scale=1e-300)
    for x in (X.icdf(800e-100), X.ilogcdf(math.log(800e-100))):
        assert 1e47 < x < 1e48
        ctx.prec = 3000
        try:
            assert float((arb(1e-300) / arb(x)).gamma_upper(arb(1e-100), 1) / arb(800e-100)) == pytest.approx(
                1.0, abs=1e-14 * 800
            )
        finally:
            ctx.prec = 53
