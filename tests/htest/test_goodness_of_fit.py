"""sd.htest's tests of goodness of fit to a given distribution (milestone H2; docs/api/htest.md §3,
docs/design/htest.md §3.3, §4): the accuracy contract against the Arb references
(tests/htest/ref), batches lane by lane, the thread count, nan_policy, the degenerate inputs
(H11, H20), the H2 decisions (H34-H38), the Jarque-Bera kernel shared with sd.regression (H17),
the null distributions of the Kolmogorov-Smirnov statistics, the results and the messages."""

import math

import numpy as np
import pytest
import scipy.stats as st
from scipy import special

import stardust as sd

from .._bits import bits_equal, value_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H2
H2 = (
    "chisquare",
    "power_divergence",
    "skewtest",
    "kurtosistest",
    "normaltest",
    "jarque_bera",
    "shapiro",
    "anderson",
    "ks_1samp",
    "cramervonmises",
)
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
call = R.call
CDF = {"norm": sd.stats.Normal(), "uniform": sd.stats.Uniform(a=0.0, b=1.0)}


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """The statistic to 1e-14 relative (times its condition number in the moment or sum it is
    formed from, kappa_stat: the D'Agostino transforms near 0, A^2's cancelling sum), the p-value to
    1e-14 max(1, kappa_p) (times kappa_stat), the Cramér-von Mises p-value (1 - cdf of its
    approximation) to 1e-14 relative or 1e-15 absolute (tests/htest/refdata.py)."""
    rec = REFS[fam][pid]
    r = call(P.PROBLEMS[pid])
    for field, (e, c) in R.errors(fam, pid, r).items():
        assert c <= 1.0, (field, e)
    if "method" in rec:
        assert r.method == rec["method"]


def test_every_h2_function_has_references():
    assert {P.PROBLEMS[pid].func for _, pid in CASES} == set(H2)


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H2" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch (axis=-1) gives each lane's bits of the lane alone (axis=None)."""
    p = P.PROBLEMS[pid]
    r = call(p)
    kw = {k: v for k, v in p.kwargs.items() if k != "axis"}
    lanes = p.args[0].shape[0]
    assert r.statistic.shape == (lanes,)
    for i in range(lanes):
        args = [a[i] if isinstance(a, np.ndarray) and a.ndim == 2 else a for a in p.args]
        one = call(P.Problem(p.id, p.func, tuple(args), kw))
        for f in type(one)._FIELDS:
            got, want = getattr(r, f), getattr(one, f)
            if isinstance(want, str):  # a batch's methods: the lanes' names (H48)
                assert got[i] == want
            else:
                assert bits_equal(np.asarray(got)[i], want), (i, f)


def test_threads_do_not_change_the_bits():
    """The same bits on 1, 3 and 10 threads: long lanes (chunked sums, the Birnbaum-Tingey sum, the
    Durbin matrix's columns) and batches."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=300_001)
    b = rng.normal(size=(2000, 40))
    calls = [
        lambda: sd.htest.jarque_bera(x),
        lambda: sd.htest.normaltest(x),
        lambda: sd.htest.shapiro(x[:30_000]),
        lambda: sd.htest.anderson(x[:50_000], dist=sd.stats.Logistic),
        lambda: sd.htest.ks_1samp(x, sd.stats.Normal(mu=0.003)),
        lambda: sd.htest.ks_1samp(x[:900], sd.stats.Normal(mu=0.07)),
        lambda: sd.htest.cramervonmises(x[:100_000], sd.stats.Normal()),
        lambda: sd.htest.power_divergence(np.abs(x[:100_000]) + 1, lambda_="cressie-read"),
        lambda: sd.htest.skewtest(b, axis=-1),
        lambda: sd.htest.shapiro(b, axis=-1),
        lambda: sd.htest.ks_1samp(b, sd.stats.Normal(), axis=-1),
    ]
    before = sd.get_num_threads()
    try:
        want = None
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            got = [np.asarray([np.asarray(r.statistic), np.asarray(r.pvalue)]) for r in (c() for c in calls)]
            if want is None:
                want = got
            else:
                for g, w in zip(got, want):
                    assert bits_equal(g, w)
    finally:
        sd.set_num_threads(before)


# ---- scipy's definitions ------------------------------------------------------------------


def _close(a, b, rtol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_tests_are_scipys(seed):
    """Away from scipy's losses (docs/design/htest.md §7) the values are scipy 1.18.1's to rounding."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.2, 1.3, size=60)
    obs = rng.integers(5, 40, size=7).astype(float)
    exp = obs.sum() * rng.dirichlet(np.ones(7))
    for lam in (None, "log-likelihood", "freeman-tukey", "neyman", 0.37, "cressie-read"):
        a, b = sd.htest.power_divergence(obs, exp, lambda_=lam), st.power_divergence(obs, exp, lambda_=lam)
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), lam
    a, b = sd.htest.chisquare(obs, ddof=2), st.chisquare(obs, ddof=2)
    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-13) and a.df == 4.0
    for name in ("skewtest", "kurtosistest"):
        for alt in ("two-sided", "less", "greater"):
            a, b = getattr(sd.htest, name)(x, alternative=alt), getattr(st, name)(x, alternative=alt)
            assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (name, alt)
    for name in ("normaltest", "jarque_bera", "shapiro"):
        a, b = getattr(sd.htest, name)(x), getattr(st, name)(x)
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), name
    for d, sdd in (("norm", None), ("expon", sd.stats.Exponential), ("gumbel_r", sd.stats.Gumbel)):
        data = np.abs(x) if d == "expon" else x
        a, b = sd.htest.anderson(data, dist=sdd), st.anderson(data, dist=d, method="interpolate")
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-9), d
    # scipy's logistic fit stops at fsolve's xtol 1e-5: the statistic agrees to that fit's error
    a, b = sd.htest.anderson(x, dist=sd.stats.Logistic), st.anderson(x, dist="logistic", method="interpolate")
    assert _close(a.statistic, b.statistic, 1e-6)
    for alt in ("two-sided", "less", "greater"):
        for meth in ("auto", "exact", "approx", "asymp"):
            a = sd.htest.ks_1samp(x, sd.stats.Normal(), alternative=alt, method=meth)
            b = st.ks_1samp(x, st.norm.cdf, alternative=alt, method=meth)
            assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (alt, meth)
            assert a.statistic_location == b.statistic_location and a.statistic_sign == b.statistic_sign
    a, b = sd.htest.cramervonmises(x, sd.stats.Normal()), st.cramervonmises(x, "norm")
    assert _close(a.statistic, b.statistic, 1e-13) and abs(a.pvalue - b.pvalue) < 1e-7  # scipy's series stop at 1e-7


def test_p_values_are_sd_stats_distributions():
    """The p-values are sd.stats' distributions, the bits of the Python calls (H17's ccdf)."""
    x = np.random.default_rng(4).standard_t(5, size=200)
    r = sd.htest.jarque_bera(x)
    assert r.pvalue == float(sd.stats.ChiSquared(df=2.0).ccdf(r.statistic))
    r = sd.htest.normaltest(x)
    assert r.pvalue == float(sd.stats.ChiSquared(df=2.0).ccdf(r.statistic))
    r = sd.htest.chisquare([10.0, 14, 9, 22, 5])
    assert r.pvalue == float(sd.stats.ChiSquared(df=4.0).ccdf(r.statistic))
    r = sd.htest.skewtest(x)
    assert r.pvalue == min(1.0, 2.0 * float(sd.stats.Normal().ccdf(abs(r.statistic))))
    r = sd.htest.kurtosistest(x, alternative="less")
    assert r.pvalue == float(sd.stats.Normal().cdf(r.statistic))


# ---- H17, H18, H20: Jarque-Bera with sd.regression ----------------------------------------


def test_jarque_bera_is_regressions_kernel():
    """htest.jarque_bera(r.resid) and the OLS fit's r.jarque_bera() are the same bits (one kernel
    and one ChiSquared(df=2).ccdf), for one regression and for a batch."""
    rng = np.random.default_rng(3)
    for n in (30, 500, 70_000):
        X = rng.normal(size=(n, 3))
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.standard_t(4, size=n)
        r = sd.regression.ols(y, X)
        a, b = sd.htest.jarque_bera(r.resid), r.jarque_bera()
        assert bits_equal([a.statistic, a.pvalue], list(b))
    Y, Xb = rng.normal(size=(4, 300)), rng.normal(size=(4, 300, 2))
    rb = sd.regression.ols(Y, Xb)
    hb, gb = sd.htest.jarque_bera(rb.resid, axis=-1), rb.jarque_bera()
    assert bits_equal(np.asarray(hb.statistic), np.asarray(gb.statistic))
    assert bits_equal(np.asarray(hb.pvalue), np.asarray(gb.pvalue))


def test_jarque_bera_zero_variance_rule():
    """H20: constant residuals (by sd.stats skew's rule, m2 <= (eps mean)^2) give NaN in both areas,
    as skew does; the regression's old moments gave 245.66 (p 4.5e-54) for this fit."""
    x = np.tile([-1.0, 1.0], 10) * np.linspace(0.3, 2.9, 20)
    x = x - x.mean()
    r = sd.regression.ols(0.37 * x + 5.0, x[:, None], intercept=False)
    assert math.isnan(float(sd.stats.skew(r.resid)))
    assert all(math.isnan(v) for v in r.jarque_bera())
    assert all(math.isnan(v) for v in sd.htest.jarque_bera(r.resid))
    assert all(math.isnan(v) for v in sd.htest.jarque_bera(np.full(10, 1e8 + 0.5)))


def test_jarque_bera_result_class():
    """One class for the one test (H18, H39, regression's G165): htest's result is
    sd.regression.JarqueBeraResult, read-only, unpacking as (statistic, pvalue); no longer a tuple
    (no tuple equality, hash, _asdict, _replace or _fields)."""
    r = sd.htest.jarque_bera([1.0, 2.0, 4.0, 8.0, 3.0])
    X = np.random.default_rng(1).normal(size=(30, 2))
    g = sd.regression.ols(X @ np.array([1.0, -1.0]) + np.random.default_rng(2).normal(size=30), X).jarque_bera()
    assert type(r) is sd.regression.JarqueBeraResult and type(g) is sd.regression.JarqueBeraResult
    s, p = r
    assert (s, p) == (r[0], r[1]) == (r.statistic, r.pvalue) and len(r) == 2
    assert not isinstance(g, tuple) and not hasattr(g, "_asdict") and not hasattr(g, "_fields")
    with pytest.raises(AttributeError):
        g.statistic = 0.0


# ---- the null distributions ---------------------------------------------------------------


def test_kolmogorov_smirnov_distributions_against_closed_forms():
    k = sd._core.htest
    # n = 1: P(D >= d) = 2 (1 - d) above 1/2; P(D+ >= d) = 1 - d
    for d in (0.5, 0.6, 0.93):
        assert k.dist_sf(0, 1, d) == min(1.0, 2.0 * (1.0 - d))
        assert k.dist_sf(1, 1, d) == 1.0 - d
    # n d <= 1/2: 1; Ruben-Gambino's n d >= n - 1: 2 (1 - d)^n
    assert k.dist_sf(0, 10, 0.05) == 1.0
    assert math.isclose(k.dist_sf(0, 10, 0.95), 2.0 * 0.05**10, rel_tol=1e-14)
    # d >= 1/2: exactly twice the one-sided tail
    for n, d in ((7, 0.5), (50, 0.6), (3000, 0.51)):
        assert k.dist_sf(0, n, d) == 2.0 * k.dist_sf(1, n, d)
    # the deep tails go on into the subnormals (Arb: 7e-323; scipy's kstwo is 0 there)
    assert k.dist_sf(0, 5000, math.sqrt(365 / 5000)) == 7e-323 and st.kstwo.sf(math.sqrt(365 / 5000), 5000) == 0.0
    assert 0.0 < k.dist_sf(1, 10**6, math.sqrt(365 / 1e6)) < 1e-300
    assert 0.0 < k.dist_sf(2, 0, 19.2) < 1e-300
    # the limits: Kolmogorov's series and scipy's kolmogorov, Smirnov's tail for large n
    for x in (0.2, 0.7, 1.0, 1.5, 3.0, 6.0):
        assert math.isclose(k.dist_sf(2, 0, x), special.kolmogorov(x), rel_tol=1e-14)
    for n, d in ((10, 0.3), (1000, 0.03), (10**5, 0.004), (10**6, 0.002)):
        assert math.isclose(k.dist_sf(1, n, d), special.smirnov(n, d), rel_tol=1e-13)


def test_two_sided_tail_where_scipy_drifts():
    """scipy's kstwo uses Pelz and Good's expansion for n > 140 (4.4e-6 relative at n = 141) and
    2 smirnov from n d^2 = 2.2 (1.6e-6): here the double-double Durbin matrix, checked against the
    references and, for n = 2, Ruben and Gambino's closed form."""
    k = sd._core.htest
    # n = 2, 1/4 < d <= 1/2: P(D_2 < d) = 2!/2^2 (4d - 1)^2 = 2 (2d - 1/2)^2
    for d in (0.3, 0.4, 0.45):
        assert math.isclose(1.0 - k.dist_sf(0, 2, d), 2.0 * (2.0 * d - 0.5) ** 2, rel_tol=1e-14)
    ref = REFS["ks_1samp"]["ks_1samp/n141"]["pvalue"][0]
    r = call(P.PROBLEMS["ks_1samp/n141"])
    assert (
        abs(r.pvalue / ref - 1) < 1e-14
        and abs(st.ks_1samp(P.PROBLEMS["ks_1samp/n141"].args[0], st.norm.cdf).pvalue / ref - 1) > 1e-7
    )


def test_cramervonmises_support():
    k = sd._core.htest
    assert k.dist_sf(3, 10, 1.0 / 120) == 1.0 and k.dist_sf(3, 10, 10.0 / 3) == 0.0
    assert k.dist_sf(3, 5, 1.5) == 0.0  # Csörgő and Faraway's form goes below 0 there (scipy: max(0, ...))


# ---- degenerate inputs, nan_policy, decisions ---------------------------------------------


def test_degenerate_inputs():
    nan = math.isnan
    assert nan(sd.htest.skewtest(np.arange(7.0)).statistic) and not nan(
        sd.htest.skewtest(np.arange(8.0) ** 2).statistic
    )
    assert nan(sd.htest.kurtosistest(np.arange(4.0)).statistic) and not nan(
        sd.htest.kurtosistest(np.arange(5.0) ** 2).statistic
    )
    assert nan(sd.htest.normaltest(np.arange(7.0)).pvalue)
    assert nan(sd.htest.shapiro([1.0, 2.0]).statistic) and not nan(sd.htest.shapiro([1.0, 2.0, 4.0]).statistic)
    assert nan(sd.htest.anderson([3.0]).statistic)
    assert nan(sd.htest.cramervonmises([0.3], sd.stats.Normal()).statistic)
    for f in (
        sd.htest.skewtest,
        sd.htest.kurtosistest,
        sd.htest.normaltest,
        sd.htest.jarque_bera,
        sd.htest.shapiro,
        sd.htest.anderson,
    ):
        r = f(np.full(30, 2.5))
        assert nan(r.statistic) and nan(r.pvalue), f.__name__
        r = f([])
        assert nan(r.statistic) and nan(r.pvalue), f.__name__
    r = sd.htest.ks_1samp([], sd.stats.Normal())
    assert nan(r.statistic) and r.statistic_sign == 0
    r = sd.htest.chisquare([4.0])
    assert r.statistic == 0.0 and nan(r.pvalue) and r.df == 0.0
    # a real scalar is a sample of one (H25)
    assert nan(sd.htest.shapiro(2.0).statistic)


def test_zero_counts_are_limits():
    """O = 0 contributes its limit (H38's API note): finite for lambda > -1 where scipy gives NaN
    (0 * inf) for -1 < lambda < 0, inf for lambda <= -1."""
    obs = np.array([12.0, 0, 9, 15, 4])
    r = sd.htest.power_divergence(obs, lambda_="freeman-tukey")
    with np.errstate(all="ignore"):
        scipy_ft = st.power_divergence(obs, lambda_="freeman-tukey").statistic
    assert math.isfinite(r.statistic) and math.isnan(scipy_ft)
    assert sd.htest.power_divergence(obs, lambda_="neyman").statistic == math.inf
    assert sd.htest.power_divergence(obs, lambda_="mod-log-likelihood").statistic == math.inf
    # lambda near 0 and -1: continuous through the G test's and the modified one's values
    g = sd.htest.power_divergence(obs + 1, lambda_=0.0).statistic
    assert math.isclose(sd.htest.power_divergence(obs + 1, lambda_=1e-9).statistic, g, rel_tol=1e-8)
    m = sd.htest.power_divergence(obs + 1, lambda_=-1.0).statistic
    assert math.isclose(sd.htest.power_divergence(obs + 1, lambda_=-1.0 + 1e-9).statistic, m, rel_tol=1e-8)


def test_skewtest_of_zero_skewness():
    """A sample skewness of exactly 0 gives Z = 0, p = 1 (scipy 1.18.1 puts 1 for y = 0: Z != 0)."""
    x = np.array([-4.0, -3, -2, -1, 0, 1, 2, 3, 4])
    r = sd.htest.skewtest(x)
    assert r.statistic == 0.0 and r.pvalue == 1.0
    assert st.skewtest(x).statistic != 0.0


def test_nan_policy():
    rng = np.random.default_rng(9)
    x = rng.normal(size=40)
    xn = np.insert(x, 5, np.nan)
    for f, extra in (
        (sd.htest.skewtest, ()),
        (sd.htest.kurtosistest, ()),
        (sd.htest.normaltest, ()),
        (sd.htest.jarque_bera, ()),
        (sd.htest.shapiro, ()),
        (sd.htest.anderson, ()),
        (sd.htest.ks_1samp, (sd.stats.Normal(),)),
        (sd.htest.cramervonmises, (sd.stats.Normal(),)),
        (sd.htest.chisquare, ()),
    ):
        data, datan = (np.abs(x) + 1, np.abs(xn) + 1) if f is sd.htest.chisquare else (x, xn)
        r = f(datan, *extra)
        assert math.isnan(r.statistic) and math.isnan(r.pvalue), f.__name__
        a, b = f(datan, *extra, nan_policy="omit"), f(data, *extra)
        assert bits_equal([a.statistic, a.pvalue], [b.statistic, b.pvalue]), f.__name__
        with pytest.raises(ValueError, match=r"the input contains NaN \(nan_policy='raise'\)"):
            f(datan, *extra, nan_policy="raise")
    # f_exp pairs: omit drops the pair
    o, e = np.array([5.0, np.nan, 7, 9]), np.array([6.0, 1.0, 7, 8])
    a = sd.htest.chisquare(o, e, nan_policy="omit", sum_check=False)
    b = sd.htest.chisquare(o[[0, 2, 3]], e[[0, 2, 3]], sum_check=False)
    assert value_equal([a.statistic, a.pvalue, a.df], [b.statistic, b.pvalue, b.df])


def test_axis_and_keepdims():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(3, 50, 2))
    r = sd.htest.shapiro(x, axis=1)
    assert r.statistic.shape == (3, 2)
    assert r.statistic[1, 0] == sd.htest.shapiro(x[1, :, 0]).statistic
    r = sd.htest.ks_1samp(x, sd.stats.Normal(), axis=1, keepdims=True)
    assert r.statistic.shape == (3, 1, 2) and r.statistic_sign.dtype == sd.int64
    one = sd.htest.ks_1samp(x[2, :, 1], sd.stats.Normal())
    assert r.statistic_location[2, 0, 1] == one.statistic_location
    r = sd.htest.chisquare(np.abs(x) + 1, axis=(1, 2))
    assert r.statistic.shape == (3,) and r.df.tolist() == [99.0] * 3
    # f_exp broadcasts against f_obs
    o = np.abs(rng.normal(size=(4, 6))) + 1
    e = o.sum(axis=-1, keepdims=True) / 6
    r = sd.htest.chisquare(o, e, axis=-1)
    assert np.allclose(np.asarray(r.statistic), st.chisquare(o, axis=-1).statistic, rtol=1e-13)


def test_cdf_callables_and_distributions():
    """cdf: an sd.stats distribution object or a callable of the sorted samples in the batch's shape
    (H35)."""
    x = np.random.default_rng(12).normal(size=(3, 25))
    seen = []

    def cdf(v):
        seen.append(tuple(v.shape))
        return sd.stats.Normal().cdf(v)

    a = sd.htest.ks_1samp(x, cdf, axis=-1)
    b = sd.htest.ks_1samp(x, sd.stats.Normal(), axis=-1)
    assert seen == [(3, 25)] and bits_equal(np.asarray(a.pvalue), np.asarray(b.pvalue))
    r = sd.htest.cramervonmises(x[0], lambda v: np.asarray(st.norm.cdf(np.asarray(v))))
    assert math.isclose(r.statistic, sd.htest.cramervonmises(x[0], sd.stats.Normal()).statistic, rel_tol=1e-14)


def test_results():
    r = sd.htest.ks_1samp([0.1, 0.5, -0.3], sd.stats.Normal())
    assert r.method == "exact" and isinstance(r.statistic_sign, int)
    with pytest.raises(AttributeError):
        r.statistic = 1.0
    stat, p = r
    assert len(r) == 2
    assert sd.htest.ks_1samp([0.1, 0.5], sd.stats.Normal(), method="asymp").method == "asymp"
    assert sd.htest.ks_1samp([0.1, 0.5], sd.stats.Normal(), alternative="less", method="asymp").method == "exact"
    r = sd.htest.cramervonmises([0.1, 0.5, -0.3], sd.stats.Normal())
    with pytest.raises(TypeError):
        iter(r)
    assert repr(sd.htest.chisquare([16, 18, 16, 14, 12, 12])) == "ChisquareResult(statistic=2, pvalue=0.849145, df=5)"
    b = sd.htest.normaltest(np.zeros((2, 3, 10)) + np.arange(10.0) ** 2, axis=-1)
    assert repr(b) == "NormaltestResult(batch shape (2, 3))"
    with pytest.raises(ValueError):
        b.statistic[0, 0] = 1.0  # a read-only array (H21)


def test_messages():
    x = [0.1, 0.4, -0.2, 1.3]
    with pytest.raises(
        TypeError, match=r"ks_1samp\(\): 'cdf' must be an sd.stats distribution object or a callable cdf\(x\), not str"
    ):
        sd.htest.ks_1samp(x, "norm")
    with pytest.raises(TypeError, match=r"not the class Normal"):
        sd.htest.cramervonmises(x, sd.stats.Normal)
    with pytest.raises(TypeError, match=r"ks_1samp\(\): unexpected keyword argument 'args'"):
        sd.htest.ks_1samp(x, sd.stats.Normal(), args=())
    with pytest.raises(ValueError, match=r"'cdf' must return an array of the shape of its input \(4,\), not \(\)"):
        sd.htest.ks_1samp(x, lambda v: 0.5)
    with pytest.raises(ValueError, match=r"ks_1samp\(\): 'method' must be"):
        sd.htest.ks_1samp(x, sd.stats.Normal(), method="exactly")
    with pytest.raises(TypeError, match=r"anderson\(\): 'dist' must be a distribution class .* not str"):
        sd.htest.anderson(x, dist="norm")
    with pytest.raises(TypeError, match=r"not an instance of Normal"):
        sd.htest.anderson(x, dist=sd.stats.Normal())
    with pytest.raises(ValueError, match=r"anderson\(\): 'dist' must be sd.stats.Normal, .* not sd.stats.Gamma"):
        sd.htest.anderson(x, dist=sd.stats.Gamma)
    with pytest.raises(ValueError, match=r"anderson\(\): 'method' must be 'interpolate'"):
        sd.htest.anderson(x, method="interpolated")
    with pytest.raises(ValueError, match=r"power_divergence\(\): 'lambda_' must be"):
        sd.htest.power_divergence([3.0, 4.0], lambda_="g-test")
    with pytest.raises(TypeError, match=r"'lambda_' must be None, a real number or one of"):
        sd.htest.power_divergence([3.0, 4.0], lambda_=[1])
    with pytest.raises(ValueError, match=r"chisquare\(\): the sums of 'f_obs' and 'f_exp' must agree"):
        sd.htest.chisquare([3.0, 4.0], [3.0, 5.0])
    assert sd.htest.chisquare([3.0, 4.0], [3.0, 5.0], sum_check=False).df == 1.0
    with pytest.raises(ValueError, match=r"power_divergence\(\): the sums"):
        sd.htest.power_divergence([3.0, 4.0], [3.0, 5.0])
    with pytest.raises(ValueError, match=r"'f_obs' and 'f_exp' could not be broadcast together: \(3,\), \(2,\)"):
        sd.htest.chisquare([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(TypeError, match=r"chisquare\(\): 'ddof' must be an int, not float"):
        sd.htest.chisquare([1.0, 2.0, 3.0], ddof=1.5)
    with pytest.raises(ValueError, match=r"skewtest\(\): 'alternative' must be"):
        sd.htest.skewtest(x, alternative="larger")


# ---- R's goodness-of-fit packages (H44) ---------------------------------------------------

# nortest's and goftest's values farther than 1e-12 (relative) from sd's where they compute the
# same quantity: R's mean of the offset data rounded before the subtraction, the long sums of
# n = 5000 and 20000 (the Anderson-Darling sum cancels), goftest's ad.test where a fitted cdf
# value rounds to 1 (it caps the statistic), and goftest's cvm.test p-value in the tail (0 where
# the approximation's value is 5.2e-11)
R_GOF_KNOWN = {
    ("shapiro/offset", "ad"), ("shapiro/offset", "gof_ad"), ("shapiro/offset", "cvm"), ("shapiro/offset", "lillie"),
    ("shapiro/n3000", "gof_ad"), ("shapiro/n5000", "ad"), ("shapiro/n5000", "gof_ad"), ("shapiro/n20000", "ad"),
    ("shapiro/n20000", "gof_ad"), ("anderson/norm-large", "gof_ad"), ("shapiro/tail", "gof_ad"),
    ("cramervonmises/tail", "gof_cvm.p"),
}  # fmt: skip


def test_r_nortest_and_goftest():
    """nortest's ad.test, cvm.test, lillie.test and pearson.test (estimated normal parameters) have sd's
    anderson, cramervonmises, ks_1samp and chisquare(ddof=2) statistics against the fitted normal;
    goftest's cvm.test is cramervonmises (statistic and p-value, Csörgő and Faraway's) and its ad.test
    on the fitted normal anderson's statistic. nortest's ad.test p-value is Stephens' formula, not
    scipy's table (another definition), and sf.test (Shapiro-Francia) has no sd counterpart: both
    are recorded only (tests/htest/data/r_gof.json)."""
    gof = R.data("r_gof")
    assert gof and all(rec["digest"] == P.digest(P.PROBLEMS[pid]) for pid, rec in gof.items())
    seen = set()

    def check(pid, key, got, want):
        e = abs(got / want - 1) if want else abs(got)
        if (pid, key) in R_GOF_KNOWN:
            seen.add((pid, key))
            assert e > 1e-12, (pid, key, e)
        else:
            assert e <= 1e-12, (pid, key, e)

    for pid, rec in gof.items():
        res, p = rec["result"], P.PROBLEMS[pid]
        x = np.asarray(p.args[0], dtype=float)
        if p.func in ("anderson", "shapiro"):
            fitted = sd.stats.Normal(mu=float(x.mean()), sigma=float(x.std(ddof=1)))
            a = sd.htest.anderson(x).statistic
            check(pid, "ad", a, res["ad.statistic"][0])
            check(pid, "gof_ad", a, res["gof_ad.statistic"][0])
            check(pid, "cvm", sd.htest.cramervonmises(x, fitted).statistic, res["cvm.statistic"][0])
            check(pid, "lillie", sd.htest.ks_1samp(x, fitted).statistic, res["lillie.statistic"][0])
            counts = np.asarray(res["pearson.counts"])
            if counts.sum() == x.size:  # (nortest drops a value whose fitted cdf rounds to 1)
                c = sd.htest.chisquare(counts, ddof=2)
                check(pid, "pearson", c.statistic, res["pearson.statistic"][0])
                check(pid, "pearson.p", c.pvalue, res["pearson.p.value"][0])
            if x.size <= 5000:
                assert 0.0 < res["sf.statistic"][0] <= 1.0
        else:
            c = sd.htest.cramervonmises(x, CDF[p.args[1]] if p.args[1] in CDF else None)
            check(pid, "gof_cvm", c.statistic, res["gof_cvm.statistic"][0])
            ref = REFS["cramervonmises"][pid]["pvalue"][0]
            check(pid, "gof_cvm.p", res["gof_cvm.p.value"][0], ref)
    assert seen == R_GOF_KNOWN
