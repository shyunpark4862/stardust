"""stats and special (2026-11-19, third pass, R63-R74): the noncentral chi-square's point mass
for df < 2^-1021 (T44, T46, T47), the chi-square samples there (T48), the noncentral F just
above the boundary, moment's order, the gelsd path with tiny singular values."""

import inspect
import math

import numpy as np
import pytest
from flint import arb, ctx

import stardust as sd

S = sd.stats
sp = sd.special
TINY = (5e-324, 1e-310, 2.0**-1022, math.nextafter(2.0**-1021, 0.0))


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


def test_special_underneath():
    d = 2.0**-1022
    # the upper tail was NaN where df / 2 is subnormal; now the df = 0 values
    for x in (1e-300, 1e-10, 1.0):
        assert sp.chndtrc(d, 1.0, x) == sp.chndtrc(0.0, 1.0, x)
        assert sp.chndtr(d, 1.0, x) == sp.chndtr(0.0, 1.0, x)
    # df > 0 has no point mass: chndtr(df, nc, 0) stays 0 (the function, not the T44 distribution)
    assert sp.chndtr(d, 1.0, 0.0) == 0.0


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


def test_moment_order():
    x = [1.0, 2.0, 3.0, 4.0]
    assert S.moment(x, order=np.int64(2)) == S.moment(x, order=2) == 1.25  # R66
    assert S.moment(x, order=[np.int64(2), 3]).tolist() == [1.25, 0.0]
    with pytest.raises(OverflowError, match=r"^moment\(\): 'order' is out of range$"):  # R67
        S.moment(x, order=10**400)
    with pytest.raises(OverflowError, match=r"^moment\(\): 'order' is out of range$"):
        S.moment(x, order=[2, 10**400])
    with pytest.raises(TypeError, match=r"^moment\(\): 'order' must be an int or a sequence of ints, not bool$"):
        S.moment(x, order=True)
    for bad in (2.5, math.inf, math.nan):
        with pytest.raises(ValueError, match=r"^moment\(\): every 'order' must be integral$"):
            S.moment(x, order=bad)


def test_gelsd_path_is_numpys_for_tiny_systems():
    # R63: rtol >= eps keeps gelsd (it scales a tiny matrix itself): numpy's bits, no NaN
    rng = np.random.default_rng(7)
    for _ in range(200):
        m, n = rng.integers(1, 5, size=2)
        a = rng.standard_normal((m, n)) * 10.0 ** rng.uniform(-323, -295)
        a[rng.random((m, n)) < 0.3] = 0.0
        b = rng.standard_normal(m)
        b[rng.random(m) < 0.3] = 0.0
        x = np.asarray(sd.linalg.lstsq(sd.asarray(a), sd.asarray(b)).solution.tolist())
        want = np.linalg.lstsq(a, b, rcond=None)[0]
        assert not np.isnan(x).any()
        assert np.array_equal(x, want), (a, b)


# ---- R75, R77, R78 (2026-11-19, fourth pass)


@pytest.mark.parametrize("df", [5e-324, 2.0**-1022, 2.0**-1021, 1e-307, 1e-305, 1e-292])
def test_chndtrc_tiny_df(df):
    # R77: NaN for df up to ~1e-305 at small x; now the df = 0 law plus the j = 0 term's tail
    for x in (1e-300, 1e-100, 1e-20):
        assert sp.chndtrc(df, 1.0, x) == pytest.approx(-math.expm1(-0.5), rel=1e-15)
        assert sp.chndtr(df, 1.0, x) == pytest.approx(math.exp(-0.5), rel=1e-12)
    # nc tiny too: both terms count, mu e^-y and e^-mu (df/2) E1(y)
    want = 5e-301 + 0.5 * df * sp.expn(1, 5e-301)
    assert sp.chndtrc(df, 1e-300, 1e-300) == pytest.approx(want, rel=1e-14, abs=45 * 2.0**-1074)
    X = S.NoncentralChiSquared(df=max(df, 2.0**-1021), nc=1.0)
    assert X.ccdf(1e-300) == pytest.approx(-math.expm1(-0.5), rel=1e-15)
    assert X.logccdf(1e-20) == pytest.approx(math.log(-math.expm1(-0.5)), rel=1e-14)


def test_ncfdtri_root_beyond_the_doubles():
    # R78: the tail at the largest double is still short of the target: inf (was NaN)
    assert sp.ncfdtri(5e-324, 5.0, 2.0, 0.5) == math.inf and sp.ncfdtri(1e-320, 5.0, 2.0, 0.5) == math.inf
    assert sp.ncfdtri(5e-324, 5.0, 2.0, 0.2) == 0.0  # inside the j = 0 mass, below every double
    assert sp.ncfdtri(2.0**-1021, 5.0, 2.0, 0.5) == pytest.approx(1.8609659255940713e307, rel=1e-12)


def test_descriptive_call_shape_errors():
    # R75: the library's form, as the constructors' (T36)
    with pytest.raises(TypeError, match=r"^moment\(\): missing 1 required keyword-only argument: 'order'$"):
        S.moment([1.0, 2.0, 4.0])
    with pytest.raises(TypeError, match=r"^moment\(\): 'order' must be an int or a sequence of ints, not NoneType$"):
        S.moment([1.0, 2.0, 4.0], order=None)
    with pytest.raises(TypeError, match=r"^skew\(\): unexpected keyword argument 'ddof'"):
        S.skew([1.0, 2.0, 3.0], ddof=1)
    with pytest.raises(TypeError, match=r"^describe\(\): takes 1 positional argument but 2 were given$"):
        S.describe([1.0, 2.0], 0)
    assert S.moment.__name__ == "moment" and str(inspect.signature(S.moment)).startswith("(x, /, *, order")


# ---- R82, R83, R87 (2026-11-19, fifth pass)


def _arb_q(a, y):
    ctx.prec = 4000
    try:
        return arb(y).gamma_upper(arb(a), 1)
    finally:
        ctx.prec = 53


def test_ncfdtr_smallest_dfn():
    # R82: dfn = 5e-324 halves to 0; NaN for f ~ 1e23-1e173
    for f in (1e10, 1e23, 1e100, 1e173, 1e200):
        assert sp.ncfdtr(5e-324, 5.0, 2.0, f) == pytest.approx(math.exp(-1.0), rel=1e-15)
        assert sp.ncfdtrc(5e-324, 5.0, 2.0, f) == pytest.approx(-math.expm1(-1.0), rel=1e-15)
        assert sp.ncfdtr(5e-324, 5.0, 2.0, f) == sp.ncfdtr(1e-323, 5.0, 2.0, f)


@pytest.mark.parametrize(
    "df,nc,x",
    [(5e-324, 5e-324, 1e-300), (1.5e-323, 1e-300, 1e-300), (1e-310, 1e-310, 1e-20), (2.0**-1022, 1e-300, 1e-300)],
)
def test_chndtrc_subnormal_df_against_arb(df, nc, x):
    # R83: the j = 0 term's half after the scaling (0.5 * df rounded first: 0 for 1.71e-321)
    ctx.prec = 4000
    try:
        a, mu, y = arb(df) / 2, arb(nc) / 2, arb(x) / 2
        want, p = arb(0), (-mu).exp()
        for j in range(60):
            want += p * y.gamma_upper(a + j, 1)
            p = p * mu / (j + 1)
        want = float(want)
    finally:
        ctx.prec = 53
    assert abs(sp.chndtrc(df, nc, x) - want) <= max(2.0**-1074, 1e-14 * want)


@pytest.mark.parametrize("v,x", [(1.5e-323, 1e-300), (5e-324, 1e-300), (2.5e-323, 1.0), (1e-310, 1e-300)])
def test_chdtrc_odd_subnormal_df(v, x):
    want = float(_arb_q(arb(v) / 2, arb(x) / 2))
    assert abs(sp.chdtrc(v, x) - want) <= 2.0**-1074


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


# ---- R101, R102 (2026-11-19, sixth pass; T52): small shapes' upper roots below 2^-960


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


@pytest.mark.parametrize("a", [1e-310, 2.0**-1023, 5e-324, 2.0**-1022, 1e-307, 1e-303])
@pytest.mark.parametrize("c", [1.0, 5.0, 39.0, 41.0, 300.0, 665.0, 700.0])
def test_gammainccinv_subnormal_shape(a, c):
    # E1(y) = q / a (Q(a, y) = a E1(y)): was NaN for every subnormal a. R105: c = q / a in one
    # division (log q - log a carried 1e-13 into c: Q 5e-14 off), and the normal a near
    # 2^-1022 with q < 1e-300 by the same equation (NaN or 7e-13 off); stats' Gamma too
    y = sp.gammainccinv(a, c * a)
    assert y > 0.0 and abs(_arb_q_rel(a, y, None, c * a)) <= 1e-15
    ys = S.Gamma(a=a).iccdf(c * a)
    assert ys > 0.0 and abs(_arb_q_rel(a, ys, None, c * a)) <= 1e-15
    assert sp.gammainccinv(a, 760 * a) == 0.0 and sp.gammaincinv(a, 0.3) == 0.0
    # the lower side's root is below every double (it was NaN in stats for a normal a here)
    assert S.Gamma(a=a).icdf(0.3) == 0.0 and S.Gamma(a=a).icdf(1e-20) == 0.0
