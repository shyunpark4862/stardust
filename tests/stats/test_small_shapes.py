"""Shapes below 2^-1021 (user decisions 2026-11-19, T49 and T50): F's samples are NaN as its
values; Beta and BetaBinomial are their shapes' limit at 0, the two atoms with
P(upper) = a / (a + b) whichever shape is below (one mass rounds to 0 when the other shape is
not small: a single point mass), in every method and the samples. Arb checks the atoms."""

import math

import pytest
import stardust as sd
from flint import arb, ctx

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
    assert math.isfinite(X.pdf(0.5)) and not any(math.isnan(u) for u in X.sample(3, rng=sd.random.default_rng(1)).tolist())


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
    assert S.BetaBinomial(n=6, a=below, b=2 * at).pmf(6) == pytest.approx(S.BetaBinomial(n=6, a=at, b=2 * at).pmf(6), rel=1e-15)


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
