"""sd.htest's tests of goodness of fit between samples (milestone H4; docs/api/htest.md §5,
docs/design/htest.md §3.5, §4): the accuracy contract against the Arb and exact-rational
references (tests/htest/ref), batches lane by lane, the thread count, nan_policy, the degenerate
inputs (H11), the H4 decisions (H55-H61: the exact distributions conditional on ties as R's
psmirnov, the Cramér-von Mises dynamic programme past scipy's sizes, anderson_ksamp's default
and unpacking, epps_singleton_2samp's df), scipy's definitions, R's ks.test, the results and the messages."""

import math
import time

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from .._bits import bits_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H4
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
call = R.call
h = sd.htest


def _scipy(p):
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(st, p.func)(*p.args, **p.kwargs)


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """The statistics to 1e-14 relative (D, T and the Anderson-Darling sums are exact rationals
    rounded once), the exact p-values to 1e-14 (counts over the label arrangements), the
    asymptotic and table p-values to 1e-14 max(1, kappa_p), the asymptotic Cramér-von Mises one
    (1 - V) also within 1e-15 absolute (H41)."""
    rec = REFS[fam][pid]
    r = call(P.PROBLEMS[pid])
    for field, (e, c) in R.errors(fam, pid, r).items():
        assert c <= 1.0, (field, e)
    if "method" in rec:
        assert r.method == rec["method"]
    if "statistic_location" in rec:
        assert r.statistic_location == rec["statistic_location"][0] and r.statistic_sign == rec["statistic_sign"]


def test_every_h4_function_has_references():
    assert {P.PROBLEMS[pid].func for _, pid in CASES} == {
        "ks_2samp",
        "cramervonmises_2samp",
        "anderson_ksamp",
        "epps_singleton_2samp",
    }


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H4" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch (axis=-1) gives each lane's bits of the lane alone (axis=None), its methods too (H48)."""
    p = P.PROBLEMS[pid]
    r = call(p)
    kw = {k: v for k, v in p.kwargs.items() if k != "axis"}
    lanes = p.args[0].shape[0]
    assert r.statistic.shape == (lanes,)
    for i in range(lanes):
        one = call(P.Problem(p.id, p.func, tuple(a[i] for a in p.args), kw))
        for f in type(one)._FIELDS:
            if not hasattr(one, f):
                continue
            got, want = getattr(r, f), getattr(one, f)
            if isinstance(want, str):
                assert got[i] == want
            else:
                assert bits_equal(np.asarray(got)[i], want), (i, f)


def test_threads_do_not_change_the_bits():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=60_000), rng.normal(0.01, 1.0, size=50_001)
    bx, by = rng.normal(size=(3000, 30)), rng.normal(0.2, 1.0, size=(3000, 25))
    calls = [
        lambda: h.ks_2samp(x[:9000], y[:7000]),
        lambda: h.ks_2samp(np.round(x[:5000], 1), np.round(y[:4000], 1)),
        lambda: h.ks_2samp(x, y),
        lambda: h.cramervonmises_2samp(x, y),
        lambda: h.cramervonmises_2samp(x[:60], y[:50], method="exact"),
        lambda: h.anderson_ksamp([x, y, x[:1000] + 0.1]),
        lambda: h.epps_singleton_2samp(x, y),
        lambda: h.ks_2samp(bx, by, axis=-1),
        lambda: h.cramervonmises_2samp(bx, by, axis=-1),
        lambda: h.epps_singleton_2samp(bx, by, axis=-1),
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
    """Without ties and away from scipy's losses (docs/design/htest.md §7) the values are scipy
    1.18.1's to rounding: the exact KS distributions where scipy's counts hold (small sizes),
    the asymptotic forms, the Cramér-von Mises exact and asymptotic p-values, the Anderson-Darling
    variants and the Epps-Singleton statistic."""
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=25), rng.normal(0.4, 1.2, size=31)
    for alt in ("two-sided", "less", "greater"):
        for meth in ("auto", "exact", "asymp"):
            a, b = h.ks_2samp(x, y, alternative=alt, method=meth), st.ks_2samp(x, y, alternative=alt, method=meth)
            assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (alt, meth)
            assert a.statistic_location == b.statistic_location and a.statistic_sign == b.statistic_sign
    for meth in ("auto", "exact", "asymptotic"):
        a, b = h.cramervonmises_2samp(x[:15], y[:12], method=meth), st.cramervonmises_2samp(x[:15], y[:12], method=meth)
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), meth
    a, b = h.cramervonmises_2samp(x, y), st.cramervonmises_2samp(x, y)
    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12)
    z = rng.normal(-0.3, 0.8, size=19)
    for v in ("midrank", "right", "continuous"):
        a, b = h.anderson_ksamp([x, y, z], variant=v), st.anderson_ksamp([x, y, z], variant=v)
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), v
    for t in ((0.4, 0.8), (0.3, 0.6, 0.9), 0.5):
        a, b = h.epps_singleton_2samp(x, y, t=t), st.epps_singleton_2samp(x, y, t=t)
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-10), t


def test_methods_and_thresholds():
    """'auto' as scipy's (H46): ks_2samp exact up to max(n1, n2) = 10000, cramervonmises_2samp up to
    20; the result's method says which, lane by lane in a batch (H48)."""
    rng = np.random.default_rng(5)
    assert h.ks_2samp(rng.normal(size=10_000), rng.normal(size=30)).method == "exact"
    assert h.ks_2samp(rng.normal(size=10_001), rng.normal(size=30)).method == "asymp"
    assert h.cramervonmises_2samp(rng.normal(size=20), rng.normal(size=20)).method == "exact"
    assert h.cramervonmises_2samp(rng.normal(size=21), rng.normal(size=5)).method == "asymptotic"
    a = rng.normal(size=(2, 22))
    a[0, 2:] = np.nan
    r = h.cramervonmises_2samp(a, rng.normal(size=(2, 20)), axis=-1, nan_policy="omit")
    assert r.method == ["exact", "asymptotic"]
    # under 'propagate' a NaN lane's method is decided on its other values (H53)
    r = h.cramervonmises_2samp(a, rng.normal(size=(2, 20)), axis=-1)
    assert r.method == ["exact", "asymptotic"] and math.isnan(r.pvalue[0])


# ---- H55: the exact distribution conditional on the ties ---------------------------------


def test_ks_2samp_ties_are_rs_psmirnov():
    """With ties method='exact' (and 'auto') is the distribution conditional on the observed ties,
    R's psmirnov (R >= 4.3): x = [1, 2, 2, 3, 3, 3], y = [2, 3, 4, 4, 5] gives R's 0.1082251 where
    scipy 1.18.1's (ties ignored) gives 0.238; without ties the two are the same distribution."""
    x, y = [1.0, 2, 2, 3, 3, 3], [2.0, 3, 4, 4, 5]
    r = h.ks_2samp(x, y)
    assert r.method == "exact" and math.isclose(r.pvalue, 25 / 231, rel_tol=1e-15)  # 0.10822510822510822
    assert math.isclose(st.ks_2samp(x, y).pvalue, 0.23809523809523814, rel_tol=1e-15)
    x2, y2 = [1.0, 2.1, 2.2, 3.1, 3.2, 3.3], [2.3, 3.4, 4.1, 4.2, 5.0]
    assert math.isclose(h.ks_2samp(x2, y2).pvalue, st.ks_2samp(x2, y2).pvalue, rel_tol=1e-14)


R_KS_KNOWN = {"ks_2samp/separated", "ks_2samp/separated-greater"}  # ks.test's exact tail 1 - cdf: 0


def test_r_ks_test_exact():
    """R's ks.test(exact = TRUE) (tests/htest/data/r.json; tools/gen_htest_r.py) on every exact
    ks_2samp problem: the same D and p-value to 1e-12, with ties too (H55), but in the deep tail
    where R's 1 - cdf is 0."""
    data = R.data("r")
    seen = set()
    for pid, p in P.PROBLEMS.items():
        if p.func != "ks_2samp" or pid not in data or "result" not in data[pid]:
            continue
        r = call(p)
        if r.method != "exact":
            continue
        got = data[pid]["result"]
        assert math.isclose(got["statistic"][0], r.statistic, rel_tol=1e-12), pid
        if pid in R_KS_KNOWN:
            assert got["p.value"][0] == 0.0 and r.pvalue > 0.0
            seen.add(pid)
        else:
            assert math.isclose(got["p.value"][0], r.pvalue, rel_tol=1e-12), pid
            seen.add(pid)
    assert R_KS_KNOWN <= seen and len(seen) >= 15


def test_exact_tails_against_closed_forms():
    k = sd._core.htest
    # equal sizes without ties: the lattice programme and Hodges' closed form agree
    for n, hh in ((20, 4), (50, 9), (200, 30), (200, 120)):
        for alt in (0, 2):
            a, b = k.exact_tail(0, n, n, hh, alt), k.exact_tail(1, n, n, hh, alt)
            assert math.isclose(a, b, rel_tol=1e-14), (n, hh, alt)
    # P(D+ >= 1) = 1/C(n1 + n2, n1) (complete separation); the gaps that cannot be reached
    assert math.isclose(k.exact_tail(0, 7, 9, 63, 2), 1.0 / math.comb(16, 7), rel_tol=1e-15)
    assert k.exact_tail(0, 7, 9, 64, 2) == 0.0 and k.exact_tail(0, 7, 9, 0, 0) == 1.0
    # Cramér-von Mises: the largest U, nx^2 ny^2 (x all above y, or all below), twice in C(N, nx);
    # 4U/G at 0 is certain
    nx, ny = 6, 9
    u4 = nx * sum((2 * (ny + i) - 2 * i) ** 2 for i in range(1, nx + 1))
    assert u4 == 4 * nx**2 * ny**2
    assert math.isclose(k.exact_tail(2, nx, ny, u4 // 3, 0), 2.0 / math.comb(15, 6), rel_tol=1e-15)
    assert k.exact_tail(2, nx, ny, 0, 0) == 1.0


# ---- H56: the Cramér-von Mises dynamic programme ------------------------------------------


def test_cramervonmises_exact_far_past_scipys_sizes():
    """method='exact' counts the arrangements by a dynamic programme that keeps only the partial
    sums still undecided: 100 x 90 at p = 0.3 in about 0.3 s (scipy's enumeration takes seconds at
    40 x 35); past its cap (1e9 entries merged, 1e7 held) a ValueError that names 'asymptotic'
    (H13)."""
    rng = np.random.default_rng(23)
    x, y = rng.normal(size=100), rng.normal(0.1, 1.0, size=90)
    t0 = time.perf_counter()
    r = h.cramervonmises_2samp(x, y, method="exact")
    assert time.perf_counter() - t0 < 5.0 and r.method == "exact"
    a = h.cramervonmises_2samp(x, y, method="asymptotic")
    assert abs(r.pvalue - a.pvalue) < 0.01 and r.statistic == a.statistic
    with pytest.raises(
        ValueError, match=r"cramervonmises_2samp\(\): method='exact' needs about .* use method='asymptotic'"
    ):
        h.cramervonmises_2samp(rng.normal(size=3000), rng.normal(size=2999), method="exact")
    with pytest.raises(ValueError, match=r"ks_2samp\(\): method='exact' needs about .* use method='asymp'"):
        h.ks_2samp(rng.normal(size=40_000), rng.normal(size=39_999) + 0.5, method="exact", alternative="greater")


# ---- H57: anderson_ksamp ------------------------------------------------------------------


def test_anderson_ksamp_default_and_unpacking():
    """variant='midrank' by default (scipy 1.18.1 warns without it and gives the old form), and the
    result unpacks into (statistic, pvalue) as scipy 1.18.1's does (its documentation says it does
    not)."""
    rng = np.random.default_rng(29)
    s = [rng.normal(size=20), rng.normal(0.4, 1.0, size=25)]
    r = h.anderson_ksamp(s)
    stat, p = r
    assert (stat, p) == (r.statistic, r.pvalue) == (r[0], r[1]) and len(r) == 2
    assert r.statistic == h.anderson_ksamp(s, variant="midrank").statistic
    b = st.anderson_ksamp(s, variant="midrank")
    assert len(b) == 2 and math.isclose(r.statistic, b.statistic, rel_tol=1e-13)
    # the table's range: 0.25 above it, 0.001 below
    assert h.anderson_ksamp([s[0], s[0] + 1e-9]).pvalue == 0.25
    assert h.anderson_ksamp([s[0], s[1] + 5.0]).pvalue == 0.001


# ---- H59: epps_singleton_2samp's degrees of freedom ---------------------------------------


def test_epps_singleton_df():
    """df is the p-value's chi^2 degrees of freedom, numpy's matrix_rank of the covariance's pinv
    (H59, the H29/H43/H51 form: not in the tuple): 2 len(t) at full rank, less where the
    covariance is singular (values 0 and 1 only: rank 1, scipy's p-value too)."""
    rng = np.random.default_rng(41)
    x, y = rng.normal(size=30), rng.normal(0.3, 1.0, size=35)
    for t, df in (((0.4, 0.8), 4.0), ((0.2, 0.5, 1.0), 6.0), (0.7, 2.0)):
        r = h.epps_singleton_2samp(x, y, t=t)
        assert r.df == df and isinstance(r.df, float) and len(r) == 2, t
        assert r.pvalue == float(sd.stats.ChiSquared(df=df).ccdf(r.statistic))
    p = P.PROBLEMS["epps_singleton_2samp/rank-deficient"]
    r = h.epps_singleton_2samp(*p.args)
    assert r.df == 1.0 and math.isclose(r.pvalue, st.chi2(1).sf(r.statistic), rel_tol=1e-13)
    assert math.isclose(r.pvalue, _scipy(p).pvalue, rel_tol=1e-12)
    b = h.epps_singleton_2samp(rng.normal(size=(3, 20)), rng.normal(size=(3, 22)), axis=-1)
    assert b.df.tolist() == [4.0, 4.0, 4.0]
    assert math.isnan(h.epps_singleton_2samp(x[:4], y).df)


# ---- degenerate inputs, nan_policy --------------------------------------------------------


def test_degenerate_inputs():
    nan = math.isnan
    r = h.ks_2samp([], [1.0, 2.0])
    assert nan(r.statistic) and nan(r.pvalue) and r.statistic_sign == 0  # (scipy: ValueError)
    r = h.ks_2samp(np.full(4, 2.0), np.full(5, 2.0))
    assert r.statistic == 0.0 and r.pvalue == 1.0
    assert nan(h.cramervonmises_2samp([0.5], [1.0, 2.0, 3.0]).statistic)
    assert nan(h.epps_singleton_2samp([1.0, 2, 3, 4], np.arange(10.0)).statistic)
    assert nan(h.epps_singleton_2samp([1.0, 2, 3, np.inf, 5], np.arange(10.0)).pvalue)
    assert nan(h.epps_singleton_2samp(np.zeros(30), np.zeros(30)).statistic)  # an IQR of 0
    for s in ([np.full(5, 1.0), np.full(6, 1.0)], [[], [1.0, 2.0, 3.0]], [[1.0], [2.0]], [[1.0, np.nan], [2.0, 3.0]]):
        r = h.anderson_ksamp(s)
        assert nan(r.statistic) and nan(r.pvalue), s  # (scipy raises for the first three)


def test_nan_policy():
    rng = np.random.default_rng(31)
    x, y = rng.normal(size=30), rng.normal(0.3, 1.0, size=28)
    xn = np.insert(x, 4, np.nan)
    for f in (h.ks_2samp, h.cramervonmises_2samp, h.epps_singleton_2samp):
        r = f(xn, y)
        assert math.isnan(r.statistic) and math.isnan(r.pvalue), f.__name__
        a, b = f(xn, y, nan_policy="omit"), f(x, y)
        assert bits_equal([a.statistic, a.pvalue], [b.statistic, b.pvalue]), f.__name__
        with pytest.raises(ValueError, match=r"the input contains NaN \(nan_policy='raise'\)"):
            f(xn, y, nan_policy="raise")


def test_axis_and_keepdims():
    rng = np.random.default_rng(37)
    x, y = rng.normal(size=(3, 40, 2)), rng.normal(0.2, 1.0, size=(1, 35, 2))
    r = h.ks_2samp(x, y, axis=1)
    assert r.statistic.shape == (3, 2) and r.statistic_sign.dtype == sd.int64
    one = h.ks_2samp(x[2, :, 1], y[0, :, 1])
    assert r.pvalue[2, 1] == one.pvalue and r.statistic_location[2, 1] == one.statistic_location
    assert r.method == [["exact", "exact"]] * 3
    r = h.cramervonmises_2samp(x, y, axis=1, keepdims=True)
    assert r.statistic.shape == (3, 1, 2)
    r = h.epps_singleton_2samp(x, y, axis=(1, 2))
    assert r.statistic.shape == (3,)


# ---- results and messages -----------------------------------------------------------------


def test_results():
    r = h.ks_2samp([0.1, 0.5, -0.3], [0.2, 0.9])
    stat, p = r
    assert len(r) == 2 and isinstance(r.statistic_sign, int) and r.method == "exact"
    with pytest.raises(AttributeError):
        r.statistic = 1.0
    r = h.cramervonmises_2samp([0.1, 0.5, -0.3], [0.2, 0.9])
    with pytest.raises(TypeError):
        iter(r)
    assert repr(r).startswith("Cramervonmises2sampResult(statistic=")
    r = h.epps_singleton_2samp(np.arange(8.0), np.arange(9.0) ** 1.1)
    stat, p = r
    b = h.ks_2samp(np.zeros((2, 3, 10)) + np.arange(10.0), np.arange(5.0), axis=-1)
    assert repr(b) == "Ks2sampResult(batch shape (2, 3))"
    with pytest.raises(ValueError):
        b.statistic[0, 0] = 1.0  # a read-only array (H21)


def test_messages():
    x, y = [0.1, 0.4, -0.2, 1.3, 0.7], [0.3, 1.1, 0.0, -0.5, 2.0]
    with pytest.raises(ValueError, match=r"ks_2samp\(\): 'method' must be"):
        h.ks_2samp(x, y, method="asymptotic")
    with pytest.raises(ValueError, match=r"cramervonmises_2samp\(\): 'method' must be"):
        h.cramervonmises_2samp(x, y, method="asymp")
    with pytest.raises(ValueError, match=r"ks_2samp\(\): 'alternative' must be"):
        h.ks_2samp(x, y, alternative="larger")
    with pytest.raises(TypeError, match=r"ks_2samp\(\): unexpected keyword argument 'mode'"):
        h.ks_2samp(x, y, mode="exact")
    with pytest.raises(ValueError, match=r"anderson_ksamp\(\): 'variant' must be"):
        h.anderson_ksamp([x, y], variant="mid")
    with pytest.raises(TypeError, match=r"anderson_ksamp\(\): at least two samples are required, not 1"):
        h.anderson_ksamp([x])
    with pytest.raises(TypeError, match=r"anderson_ksamp\(\): 'samples' must be a list or tuple of samples"):
        h.anderson_ksamp(np.array([x, y]))
    with pytest.raises(TypeError, match=r"anderson_ksamp\(\): 'method' must be None or a PermutationMethod, not str"):
        h.anderson_ksamp([x, y], method="permutation")
    with pytest.raises(TypeError, match=r"anderson_ksamp\(\): unexpected keyword argument 'midrank'"):
        h.anderson_ksamp([x, y], midrank=True)
    with pytest.raises(ValueError, match=r"epps_singleton_2samp\(\): 't' must hold positive finite numbers"):
        h.epps_singleton_2samp(x, y, t=(0.4, -0.8))
    with pytest.raises(ValueError, match=r"epps_singleton_2samp\(\): 't' must be a number or a 1-D sequence"):
        h.epps_singleton_2samp(x, y, t=[[0.4, 0.8]])
    with pytest.raises(TypeError, match=r"ks_2samp\(\): 'data1' must be"):
        h.ks_2samp({"a": 1}, y)
