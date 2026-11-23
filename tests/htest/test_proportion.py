"""sd.htest's proportions and binomial tests (milestone H7; docs/api/htest.md §8,
docs/design/htest.md §3.8, §4): the accuracy contract against the Arb and exact-rational
references (tests/htest/ref), batches lane by lane, the thread count, the H7 decisions (H85 the
method names, H86 the alternatives, H87 prop_var, H88 the result of test_proportions_2indep, H89
compare, H90 the defaults, H91 no binom_test, H92 no Yates correction, H93 the unpacking of
proportions_chisquare, H95 the one-sided intervals, H96 no proportion_estimate, H97 the null
values, H98 the score test's constrained estimate), scipy's and statsmodels' definitions, R's
binom.test and prop.test, the degenerate inputs, the results and the messages."""

import math
import warnings

import numpy as np
import pytest
import scipy.stats as st
from statsmodels.stats import proportion as smp

import stardust as sd

from .._bits import bits_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H7
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
call = R.call
h = sd.htest
SM = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning:scipy", "ignore::RuntimeWarning:statsmodels")


def _quiet(f, *a, **k):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def _close(a, b, rtol, atol=0.0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """Statistics to 1e-14 (exact rationals rounded once, or Arb balls), p-values to 1e-14
    max(1, kappa_p) (the exact binomial and E-test sums, the normal and chi^2 tails), the
    intervals' ends to 1e-14 (each against itself, or against the interval's size where the ends
    subtract), the expected counts exactly rounded."""
    rec = REFS[fam][pid]
    r = call(P.PROBLEMS[pid])
    errs = R.errors(fam, pid, r)
    assert errs, pid
    for field, (e, c) in errs.items():
        assert c <= 1.0, (field, e)
    if "expected_freq" in rec:
        assert _close(r.expected_freq, np.asarray(rec["expected_freq"])[..., 0], 2.3e-16)


def test_every_h7_function_has_references():
    assert {P.PROBLEMS[pid].func for _, pid in CASES} == {
        "binomtest",
        "quantile_test",
        "poisson_means_test",
        "proportions_ztest",
        "proportions_chisquare",
        "proportion_confint",
        "test_proportions_2indep",
        "confint_proportions_2indep",
    }


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H7" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch (broadcast counts, the samples' leading axes, quantile_test's axis) gives each
    lane's bits of the lane alone."""
    p = P.PROBLEMS[pid]
    r = call(p)
    if p.func in ("proportion_confint", "confint_proportions_2indep"):
        args = [np.broadcast_to(np.asarray(a, dtype=float), np.broadcast_shapes(*[np.shape(x) for x in p.args]))
                for a in p.args]  # fmt: skip
        for i in range(args[0].shape[0]):
            one = call(P.Problem(p.id, p.func, tuple(float(a[i]) for a in args), p.kwargs))
            assert bits_equal(np.asarray(r)[i], np.asarray(one)), i
        return
    if p.func == "test_proportions_2indep":
        args = [np.broadcast_to(np.asarray(a, dtype=float), (3,)) for a in p.args]
        lanes = [call(P.Problem(p.id, p.func, tuple(float(a[i]) for a in args), p.kwargs)) for i in range(3)]
    elif p.func == "quantile_test":
        x = np.asarray(p.args[0])
        kw = {k: v for k, v in p.kwargs.items() if k != "axis"}
        lanes = [call(P.Problem(p.id, p.func, (x[i],), kw)) for i in range(x.shape[0])]
    else:  # proportions_ztest, proportions_chisquare: the leading axis
        c, n = (np.asarray(a) for a in p.args)
        lanes = [call(P.Problem(p.id, p.func, (c[i], n[i]), p.kwargs)) for i in range(c.shape[0])]
    for i, one in enumerate(lanes):
        for f in type(one)._FIELDS:
            if not hasattr(one, f):
                continue
            got, want = getattr(r, f), getattr(one, f)
            if isinstance(want, str) or not isinstance(got, sd.ndarray):  # (the lanes' common fields)
                assert got == want
            elif isinstance(want, sd.ndarray):
                assert bits_equal(np.asarray(got)[i], np.asarray(want)), (i, f)
            else:
                assert bits_equal(np.asarray(got)[i], want), (i, f)
        if hasattr(one, "conf_int"):
            assert bits_equal(np.asarray(r.conf_int())[i], np.asarray(one.conf_int()))


def test_threads_do_not_change_the_bits():
    rng = np.random.default_rng(71)
    n = rng.integers(1, 200, 3000).astype(float)
    c = np.floor(rng.uniform(0, 1, 3000) * (n + 1))
    n2 = rng.integers(1, 200, 3000).astype(float)
    c2 = np.floor(rng.uniform(0, 1, 3000) * (n2 + 1))
    x = rng.normal(size=(2000, 30))
    calls = [lambda m=m: h.proportion_confint(c, n, method=m) for m in ("normal", "exact", "wilson", "jeffreys")]
    calls += [
        lambda: h.proportion_confint(c[:40], n[:40], method="binom_test"),
        lambda: h.test_proportions_2indep(c, n, c2, n2).pvalue,
        lambda: h.test_proportions_2indep(c, n, c2, n2, method="score", compare="odds-ratio", value=1.3).pvalue,
        lambda: h.confint_proportions_2indep(c[:200], n[:200], c2[:200], n2[:200], method="score"),
        lambda: h.confint_proportions_2indep(c, n, c2, n2, compare="ratio", method="score"),
        lambda: h.proportions_ztest(np.c_[c, c2], np.c_[n, n2]).pvalue,
        lambda: h.quantile_test(x, q=0.1, axis=-1).pvalue,
        lambda: h.quantile_test(x, q=0.1, axis=-1).conf_int(),
    ]
    before = sd.get_num_threads()
    try:
        want = None
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            got = [np.asarray(f()) for f in calls]
            if want is None:
                want = got
            else:
                for g, w in zip(got, want):
                    assert bits_equal(g, w)
    finally:
        sd.set_num_threads(before)


# ---- scipy's and statsmodels' definitions -------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_scipy_binomtest_and_quantile_test(seed):
    """binomtest's p-value and its three intervals, quantile_test's statistic, p-value and interval
    are scipy 1.18.1's to rounding (scipy's binomtest is 5.9e-14 off at n = 1e5; the references
    hold sd to 1e-14)."""
    rng = np.random.default_rng(seed)
    for _ in range(150):
        n = int(rng.integers(1, 80))
        k = int(rng.integers(0, n + 1))
        p = float(rng.choice([0.5, 0.3, rng.uniform(0.01, 0.99)]))
        for alt in ("two-sided", "less", "greater"):
            a, b = h.binomtest(k, n, p=p, alternative=alt), st.binomtest(k, n, p=p, alternative=alt)
            assert _close(a.pvalue, b.pvalue, 1e-13) and a.statistic == b.statistic, (k, n, p, alt)
            for m in ("exact", "wilson", "wilsoncc"):
                ci = b.proportion_ci(method=m)
                assert _close(a.conf_int(method=m), [ci.low, ci.high], 2e-14, 1e-16), (k, n, p, alt, m)
        x = np.round(rng.normal(size=int(rng.integers(1, 60))), 1)  # (ties at q)
        q, pp = float(np.round(rng.normal(), 1)), float(rng.uniform(0.05, 0.95))
        for alt in ("two-sided", "less", "greater"):
            a, b = h.quantile_test(x, q=q, p=pp, alternative=alt), st.quantile_test(x, q=q, p=pp, alternative=alt)
            assert (a.statistic, a.statistic_type) == (b.statistic, b.statistic_type)
            assert _close(a.pvalue, b.pvalue, 1e-13)
            ci = b.confidence_interval()
            assert _close(a.conf_int(), [ci.low, ci.high], 0.0)


def _e_test_brute(k1, n1, k2, n2, d, alt):
    """The E-test's p-value by the double sum over x1, x2 up to 15 sd past the means with T compared exactly (squares of
    rationals), the pmfs in floats: a check of sd's one sum of tails."""
    from fractions import Fraction as F

    n1f, n2f, df = F(n1), F(n2), F(d)
    lam2 = (k1 + k2 - df * n1f) / (n1f + n2f)
    if lam2 <= 0:
        return 1.0
    mu1, mu2 = float(n1f * (lam2 + df)), float(n2f * lam2)

    def key(x1, x2):  # sign(D) D^2 / V, None where T is undefined, -inf for V = 0 < -D
        D, V = F(x1) / n1f - F(x2) / n2f - df, F(x1) / n1f**2 + F(x2) / n2f**2
        if V == 0:
            return None if D == 0 else -math.inf
        return (1 if D > 0 else -1) * D * D / V

    t = key(k1, k2)
    a = abs(t)
    p = 0.0
    top1, top2 = (int(m + 15 * math.sqrt(m) + 40) for m in (mu1, mu2))
    lp1 = [x * math.log(mu1) - mu1 - math.lgamma(x + 1) for x in range(top1)]
    lp2 = [x * math.log(mu2) - mu2 - math.lgamma(x + 1) for x in range(top2)]
    for x1 in range(top1):
        for x2 in range(top2):
            v = key(x1, x2)
            if v is None:
                continue
            hit = v <= t if alt == "less" else (v >= t if alt == "greater" else (v >= a or v <= -a))
            if hit:
                p += math.exp(lp1[x1] + lp2[x2])
    return p


def test_poisson_means_test_against_the_double_sum():
    """The E-test is the double sum over every outcome (scipy's definition without its truncation):
    against the double sum with exact comparisons to 1e-12 (the sum's own float rounding). scipy's
    truncated sums leave out up to some 2e-10 of mass, and its T of the grid and of the observed
    counts are different float expressions, so an outcome as extreme as the observed one (the
    observed itself) can drop out: (26, 2.89, 30, 5.86) 'less' is 6.6e-4 low; past-truncation (the
    observed (15, 34) beyond scipy's range of X2) 9000 times low."""
    rng = np.random.default_rng(5)
    far = 0
    for _ in range(40):
        k1, k2 = (int(v) for v in rng.integers(0, 40, 2))
        n1, n2 = (float(v) for v in np.round(rng.uniform(0.5, 6, 2), 2))
        d = float(rng.choice([0.0, 0.0, 0.5, 1.3]))
        for alt in ("two-sided", "less", "greater"):
            a = h.poisson_means_test(k1, n1, k2, n2, diff=d, alternative=alt)
            b = st.poisson_means_test(k1, n1, k2, n2, diff=d, alternative=alt)
            assert _close(a.statistic, b.statistic, 1e-13)
            assert math.isclose(a.pvalue, _e_test_brute(k1, n1, k2, n2, d, alt), rel_tol=1e-12, abs_tol=1e-300), (
                k1, n1, k2, n2, d, alt)  # fmt: skip
            far += abs(a.pvalue - b.pvalue) > 3e-10
    assert far <= 20  # scipy agrees but where its rounded T drops an outcome (some 8% of these)
    a = h.poisson_means_test(26, 2.89, 30, 5.86, alternative="less")
    b = st.poisson_means_test(26, 2.89, 30, 5.86, alternative="less")
    assert math.isclose(a.pvalue - b.pvalue, 6.555e-4, rel_tol=1e-3)
    a = h.poisson_means_test(15, 5.24, 34, 0.64, alternative="less")
    b = st.poisson_means_test(15, 5.24, 34, 0.64, alternative="less")
    assert a.pvalue > 9000 * b.pvalue


@pytest.mark.parametrize("seed", [0, 1])
def test_statsmodels_definitions(seed):
    """proportions_ztest, proportions_chisquare, proportion_confint (statsmodels' names mapped,
    H85, H95), test_proportions_2indep and confint_proportions_2indep are statsmodels 0.15.0's to
    its rounding and its root finders' tolerances (brentq's 2e-12; its 'binom_test' interval's 27
    halvings), but for the score tests of a difference away from 0 (H98: statsmodels' cubic is not
    the constrained estimate's) and where statsmodels gives NaN, inf or raises."""
    rng = np.random.default_rng(seed)
    for _ in range(60):
        n1, n2 = (int(v) for v in rng.integers(1, 60, 2))
        c1, c2 = int(rng.integers(0, n1 + 1)), int(rng.integers(0, n2 + 1))
        for alt in ("two-sided", "less", "greater"):
            for val in (None, 0.1):
                b = _quiet(smp.proportions_ztest, [c1, c2], [n1, n2], value=val, alternative=SM[alt])
                a = h.proportions_ztest([c1, c2], [n1, n2], value=val, alternative=alt)
                if np.isfinite(b[0]):
                    assert _close([a.statistic, a.pvalue], b, 1e-11, 1e-14)
            for pv in (None, 0.3):
                b = _quiet(smp.proportions_ztest, c1, n1, value=0.4, alternative=SM[alt], prop_var=pv or False)
                a = h.proportions_ztest(c1, n1, value=0.4, alternative=alt, prop_var=pv)
                if np.isfinite(b[0]):
                    assert _close([a.statistic, a.pvalue], b, 1e-11, 1e-14)
        for meth, smm in (("normal", "normal"), ("agresti_coull", "agresti_coull"), ("exact", "beta"),
                          ("wilson", "wilson"), ("jeffreys", "jeffreys"), ("binom_test", "binom_test")):  # fmt: skip
            for alt, sma in (("two-sided", "two-sided"), ("less", "larger"), ("greater", "smaller")):
                b = _quiet(smp.proportion_confint, c1, n1, method=smm, alternative=sma)
                a = h.proportion_confint(c1, n1, method=meth, alternative=alt)
                if meth == "binom_test" and alt == "two-sided" and not _close(a, b, 1e-7, 1e-12):
                    # the two-sided p-value is not monotone in q: statsmodels' brentq may land on a
                    # crossing past the acceptance region about count / nobs (48 of 48: 0.92675 for
                    # sd's 0.93007, where p jumps from 0.048 to 0.079); its end then rejects and lies
                    # outside sd's
                    assert b[0] <= a[0] and b[1] >= a[1], (c1, n1)
                    for q in b:
                        if 0 < q < 1:
                            assert st.binomtest(c1, n1, p=float(q)).pvalue <= 0.05 * (1 + 1e-9)
                    continue
                tol = 1e-7 if (meth == "binom_test" and alt == "two-sided") else 3e-14
                assert _close(a, b, tol, 1e-16), (c1, n1, meth, alt)
        cs = rng.integers(0, 30, 3).astype(float)
        ns = cs + rng.integers(1, 30, 3)
        for val in (None, 0.3):
            a, b = h.proportions_chisquare(cs, ns, value=val), smp.proportions_chisquare(cs, ns, value=val)
            assert _close([a.statistic, a.pvalue], b[:2], 1e-13) and _close(a.expected_freq, b[2][1], 1e-15)
            assert np.asarray(a.table).tolist() == b[2][0].tolist()
        for cmp, meths in (("diff", ("wald", "agresti-caffo", "score")), ("ratio", ("log", "log-adjusted", "score")),
                           ("odds-ratio", ("logit", "logit-adjusted", "logit-smoothed", "score"))):  # fmt: skip
            for m in meths:
                for alt in ("two-sided", "less", "greater"):
                    for val in (None,) + (
                        (0.1,) if cmp == "diff" and m != "score" else (1.7,) if cmp != "diff" else ()
                    ):
                        for corr in (True, False):
                            try:
                                b = _quiet(smp.test_proportions_2indep, c1, n1, c2, n2, value=val, method=m,
                                           compare=cmp, alternative=SM[alt], correction=corr)  # fmt: skip
                            except ZeroDivisionError:  # (statsmodels' smoothing of a zero margin)
                                continue
                            a = h.test_proportions_2indep(c1, n1, c2, n2, value=val, method=m, compare=cmp,
                                                          alternative=alt, correction=corr)  # fmt: skip
                            if np.isfinite(b.statistic) and np.isfinite(b.variance):
                                assert _close([a.statistic, a.pvalue, a.variance], [b.statistic, b.pvalue, b.variance],
                                              1e-10, 1e-13), (c1, n1, c2, n2, cmp, m, alt, val, corr)  # fmt: skip
                            assert _close([a.diff, a.ratio, a.odds_ratio], [b.diff, b.ratio, b.odds_ratio], 1e-13)
            for m in meths + (("newcomb",) if cmp == "diff" else ()):
                for corr in (True, False):
                    try:
                        b = _quiet(
                            smp.confint_proportions_2indep, c1, n1, c2, n2, method=m, compare=cmp, correction=corr
                        )
                    except ValueError, ZeroDivisionError:  # statsmodels' brentq without a sign change
                        continue
                    if m == "score" and cmp == "diff":
                        continue  # (H98; the inversion of the corrected test, test_score_difference_estimate)
                    a = h.confint_proportions_2indep(c1, n1, c2, n2, method=m, compare=cmp, correction=corr)
                    b = np.real(np.asarray(b))  # (numpy's roots of Koopman's double root: a tiny imaginary part)
                    if np.all(np.isfinite(b)):
                        # (count2 = nobs2: p0 = 1 is a double root, which numpy finds to some 1e-8; sd
                        # takes it exactly)
                        tol = 1e-6 if (m == "score" and c2 == n2) else (2e-9 if m == "score" else 1e-12)
                        # (numpy's roots put Koopman's end at 0 up to some 1e-11 off)
                        if cmp != "diff" and m != "score":
                            tol = 1e-11  # (statsmodels' exp(log r -+ d) of a large d)
                        assert _close(a, b, tol, 1e-9 if m == "score" else 1e-12), (c1, n1, c2, n2, cmp, m, corr)


def test_statsmodels_recorded():
    """statsmodels' recorded results (tests/htest/data/statsmodels.json) on the H7 problems: the
    same to statsmodels' rounding where its definition is sd's (not the score tests of a
    difference away from 0, H98) and it gives a number."""
    data = R.data("statsmodels")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.milestone != "H7" or p.func not in P.STATSMODELS or "result" not in data.get(pid, {}):
            continue
        if "batch" in p.tags:  # (statsmodels has no batches)
            continue
        if "offset" in p.tags:  # statsmodels' difference cancels (near-null: 5.4e-15 against 8.9e-16)
            continue
        got = data[pid]["result"]
        r = call(p)
        if p.func in ("proportion_confint", "confint_proportions_2indep"):
            if p.kwargs.get("method") == "score" and p.kwargs.get("compare", "diff") == "diff":
                continue
            want = np.asarray(got["confidence_interval"], dtype=float)
            if np.all(np.isfinite(want)):
                tol = 1e-7 if p.kwargs.get("method") in ("binom_test", "score") else 1e-12
                assert _close(np.asarray(r), want, tol, 1e-13), pid
                seen += 1
            continue
        score_diff = p.kwargs.get("method") == "score" and p.kwargs.get("compare", "diff") == "diff"
        if p.func == "test_proportions_2indep" and score_diff and p.kwargs.get("value") not in (None, 0):
            continue
        if not np.all(np.isfinite(np.asarray(got["statistic"], dtype=float))):
            continue
        assert _close(np.ravel(r.statistic), np.ravel(got["statistic"]), 1e-10), pid
        assert _close(np.ravel(r.pvalue), np.ravel(got["pvalue"]), 1e-10), pid
        seen += 1
    assert seen >= 100


# ---- the H7 decisions ---------------------------------------------------------------------


def test_method_names_and_defaults():
    """H85: scipy's names where scipy has the method ('exact' for statsmodels' 'beta', 'wilson',
    'wilsoncc'), statsmodels' elsewhere; the default 'normal' (statsmodels'). H90: the defaults of
    statsmodels 0.15.0, fixed (agresti-caffo, log-adjusted, logit-adjusted for the tests; newcomb for
    the difference's interval). H91: no binom_test (binomtest)."""
    assert h.proportion_confint(7, 30).tolist() == h.proportion_confint(7, 30, method="normal").tolist()
    exact = h.proportion_confint(7, 30, method="exact")
    assert _close(exact, smp.proportion_confint(7, 30, method="beta"), 3e-15)
    msg = (
        r"proportion_confint\(\): 'method' must be 'normal', 'agresti_coull', 'exact', 'wilson', 'wilsoncc', "
        r"'jeffreys' or 'binom_test', not 'beta'"
    )
    with pytest.raises(ValueError, match=msg):
        h.proportion_confint(7, 30, method="beta")
    for cmp, m in (("diff", "agresti-caffo"), ("ratio", "log-adjusted"), ("odds-ratio", "logit-adjusted")):
        r = h.test_proportions_2indep(15, 40, 25, 45, compare=cmp)
        assert r.method == m and r.compare == cmp
    assert (
        h.confint_proportions_2indep(15, 40, 25, 45).tolist()
        == h.confint_proportions_2indep(15, 40, 25, 45, method="newcomb").tolist()
    )
    assert not hasattr(h, "binom_test")
    # wilsoncc is scipy's (statsmodels has none): binomtest's proportion_ci(method='wilsoncc')
    b = st.binomtest(7, 30).proportion_ci(method="wilsoncc")
    assert _close(h.proportion_confint(7, 30, method="wilsoncc"), [b.low, b.high], 2e-15)


def test_alternatives_are_scipys():
    """H86: 'less' and 'greater' (statsmodels' 'smaller' and 'larger' are not taken); H95:
    proportion_confint's 'less' is [0, upper] and 'greater' [lower, 1] (scipy's direction, the
    tests' duals); statsmodels' 'larger' is [0, upper]."""
    for f, args in ((h.proportions_ztest, (15, 40)), (h.test_proportions_2indep, (15, 40, 25, 45)),
                    (h.proportion_confint, (7, 30))):  # fmt: skip
        with pytest.raises(ValueError, match=r"'alternative' must be 'two-sided', 'less' or 'greater', not 'larger'"):
            f(*args, alternative="larger")
    less = h.proportion_confint(7, 30, alternative="less")
    assert less[0] == 0.0 and _close(less, smp.proportion_confint(7, 30, alternative="larger"), 1e-15)
    greater = h.proportion_confint(7, 30, method="exact", alternative="greater")
    assert greater[1] == 1.0
    assert _close(greater, st.binomtest(7, 30, alternative="greater").proportion_ci(), 2e-15)
    assert h.proportions_ztest(30, 40, value=0.6, alternative="greater").pvalue == pytest.approx(
        smp.proportions_ztest(30, 40, value=0.6, alternative="larger")[1], rel=1e-13
    )


def test_prop_var_and_unpacking():
    """H87: prop_var None or a number in [0, 1] (statsmodels' False is not taken); H88: the result
    of test_proportions_2indep is statsmodels' fields, unpacking as (statistic, pvalue), with no
    return_results; H93: proportions_chisquare unpacks as statsmodels' (chi2, p, (table,
    expected)); H96: binomtest has no proportion_estimate (statistic is it)."""
    with pytest.raises(TypeError, match=r"proportions_ztest\(\): 'prop_var' must be an int or a float, not bool"):
        h.proportions_ztest(15, 40, value=0.5, prop_var=False)
    with pytest.raises(ValueError, match=r"proportions_ztest\(\): 'prop_var' must be in \[0, 1\]"):
        h.proportions_ztest(15, 40, value=0.5, prop_var=1.5)
    a = h.proportions_ztest(15, 40, value=0.5, prop_var=0.45)
    assert _close([a.statistic, a.pvalue], smp.proportions_ztest(15, 40, value=0.5, prop_var=0.45), 1e-14)
    stat, p = h.proportions_ztest([15, 25], [40, 45])
    r = h.test_proportions_2indep(15, 40, 25, 45, method="score")
    z, pv = r
    assert (z, pv) == (r.statistic, r.pvalue) and len(r) == 2
    assert type(r)._FIELDS == ("statistic", "pvalue", "compare", "method", "diff", "ratio", "odds_ratio",
                               "variance", "alternative", "value", "prop1_null", "prop2_null")  # fmt: skip
    assert not hasattr(h.test_proportions_2indep(15, 40, 25, 45), "prop1_null")
    with pytest.raises(TypeError, match=r"test_proportions_2indep\(\): unexpected keyword argument 'return_results'"):
        h.test_proportions_2indep(15, 40, 25, 45, return_results=False)
    c = h.proportions_chisquare([15, 25], [40, 45])
    stat, p, (table, expected) = c
    assert len(c) == 3 and table is c.table and expected is c.expected_freq and c.df == 1.0
    b = h.binomtest(7, 20)
    assert not hasattr(b, "proportion_estimate") and b.statistic == 0.35
    with pytest.raises(TypeError):
        iter(b)


def test_compare_values_and_null_values():
    """H89: compare is 'diff', 'ratio' or 'odds-ratio' (statsmodels' 'or' alias is not taken);
    H97: a null difference in (-1, 1), a ratio or odds ratio positive and finite."""
    with pytest.raises(ValueError, match=r"'compare' must be 'diff', 'ratio' or 'odds-ratio', not 'or'"):
        h.test_proportions_2indep(15, 40, 25, 45, compare="or")
    with pytest.raises(ValueError, match=r"'value' must be between -1 and 1 \(exclusive\) for compare='diff'"):
        h.test_proportions_2indep(15, 40, 25, 45, value=1.0)
    with pytest.raises(ValueError, match=r"'value' must be positive and finite for compare='ratio'"):
        h.test_proportions_2indep(15, 40, 25, 45, compare="ratio", value=0.0)
    with pytest.raises(ValueError, match=r"'method' must be 'wald', 'agresti-caffo' or 'score', not 'newcomb'"):
        h.test_proportions_2indep(15, 40, 25, 45, method="newcomb")
    r = h.test_proportions_2indep(15, 40, 25, 45, compare="ratio")
    assert r.value == 1.0 and h.test_proportions_2indep(15, 40, 25, 45).value == 0.0


def test_no_yates_correction():
    """H92: proportions_chisquare has no continuity correction (statsmodels'): R's prop.test with
    correct = FALSE, and chi2_contingency(table, correction=False) to the bit."""
    a = h.proportions_chisquare([15, 25], [40, 45])
    b = h.chi2_contingency([[15.0, 25], [25, 20]], correction=False)
    assert bits_equal(a.statistic, b.statistic) and bits_equal(a.pvalue, b.pvalue)
    data = R.data("r")["proportions_chisquare/two"]
    assert "correct = FALSE" in data["call"]
    assert math.isclose(a.pvalue, data["result"]["p.value"][0], rel_tol=1e-13)
    z = h.proportions_ztest([15, 25], [40, 45])
    assert math.isclose(z.statistic**2, a.statistic, rel_tol=1e-14) and math.isclose(z.pvalue, a.pvalue, rel_tol=1e-13)


def test_score_difference_estimate():
    """H98: the score test of a difference v (Miettinen and Nurminen) takes the constrained maximum
    likelihood estimate: the root of N x^3 + L2 x^2 + L1 x + L0 with L1 = (n0 v - N - 2 c0) v + C
    in the feasible interval -- Farrington and Manning's cubic in p1 gives the same -- or its end
    where the concave likelihood has no stationary point. statsmodels 0.15.0 writes c0 v for n0 v
    in L1, so its root is another one unless v = 0 (and its form is NaN where the estimate is at an
    end)."""
    for c1, n1, c2, n2, d in ((15, 40, 25, 45, 0.1), (10, 30, 20, 40, -0.2), (22, 34, 7, 20, 0.3)):
        p1, p2, th = c1 / n1, c2 / n2, n2 / n1
        fm = np.roots([1 + th, -(1 + th + p1 + th * p2 + d * (th + 2)), d * d + d * (2 * p1 + th + 1) + p1 + th * p2,
                       -p1 * d * (1 + d)])  # fmt: skip
        fm = [r.real - d for r in fm if abs(r.imag) < 1e-12 and max(0, -d) < r.real - d < min(1, 1 - d)]
        r = h.test_proportions_2indep(c1, n1, c2, n2, value=d, method="score")
        assert len(fm) == 1 and math.isclose(r.prop2_null, fm[0], rel_tol=1e-12)
        assert math.isclose(r.prop1_null, r.prop2_null + d, rel_tol=1e-15)
        b = smp.test_proportions_2indep(c1, n1, c2, n2, value=d, method="score")
        assert abs(b.prop2_null - fm[0]) > 1e-4  # statsmodels' root
    # at v = 0 the pooled proportion: statsmodels' own
    a, b = (
        h.test_proportions_2indep(15, 40, 25, 45, method="score"),
        smp.test_proportions_2indep(15, 40, 25, 45, method="score"),
    )
    assert _close([a.statistic, a.pvalue, a.prop1_null], [b.statistic, b.pvalue, b.prop1_null], 1e-13)
    # the estimate at an end (statsmodels: NaN): c1 = 0 at v = -0.6 gives p1 = 0, p0 = 0.6
    r = h.test_proportions_2indep(0, 40, 25, 45, value=-0.6, method="score")
    assert (r.prop1_null, r.prop2_null) == (0.0, 0.6) and math.isfinite(r.statistic)
    assert math.isnan(_quiet(smp.test_proportions_2indep, 0, 40, 25, 45, value=-0.6, method="score").statistic)
    # the interval inverts the corrected test: |z| = z_{alpha/2} at its ends
    lo, hi = h.confint_proportions_2indep(15, 40, 25, 45, method="score").tolist()
    zc = st.norm.isf(0.025)
    assert math.isclose(
        h.test_proportions_2indep(15, 40, 25, 45, value=lo, method="score").statistic, zc, rel_tol=1e-12
    )
    assert math.isclose(
        h.test_proportions_2indep(15, 40, 25, 45, value=hi, method="score").statistic, -zc, rel_tol=1e-12
    )


def test_binom_test_interval_is_the_acceptance_region():
    """'binom_test' (two-sided): the ends of the acceptance region of scipy's two-sided test about
    count / nobs: p(q) > alpha just inside each end (on a grid) and p <= alpha just outside;
    statsmodels' root finder (brentq and 27 halvings) lands within 1e-7."""
    for k, n in ((7, 30), (1, 50), (61, 200), (0, 12), (12, 12), (3, 7)):
        lo, hi = h.proportion_confint(k, n, method="binom_test").tolist()
        for q in np.linspace(lo, hi, 41)[1:-1]:
            assert st.binomtest(k, n, p=float(q)).pvalue > 0.05, (k, n, q)
        if lo > 0:
            assert st.binomtest(k, n, p=lo * (1 - 1e-9)).pvalue <= 0.05
        if hi < 1:
            assert st.binomtest(k, n, p=min(1.0, hi * (1 + 1e-9))).pvalue <= 0.05
        assert _close([lo, hi], smp.proportion_confint(k, n, method="binom_test"), 1e-7, 1e-12)


# ---- degenerate inputs --------------------------------------------------------------------


def test_degenerate_inputs():
    nan = math.isnan
    # binomtest's point masses
    assert h.binomtest(0, 10, p=0.0).pvalue == 1.0 and h.binomtest(3, 10, p=0.0).pvalue == 0.0
    assert h.binomtest(7, 10, p=1.0).pvalue == 0.0 and h.binomtest(10, 10, p=1.0).pvalue == 1.0
    assert h.binomtest(5, 10).pvalue == 1.0  # k = p n
    assert math.isclose(h.binomtest(0, 1000).pvalue, 2.0**-999, rel_tol=1e-15)  # a normal double
    # quantile_test: an empty sample NaN; NaN by the policy
    r = h.quantile_test([], q=0.0)
    assert nan(r.statistic) and nan(r.pvalue) and np.isnan(np.asarray(r.conf_int())).all()
    x = [0.4, math.nan, -1.2, 0.9, 1.7]
    assert nan(h.quantile_test(x).pvalue)
    assert h.quantile_test(x, nan_policy="omit").pvalue == h.quantile_test([0.4, -1.2, 0.9, 1.7]).pvalue
    with pytest.raises(ValueError, match=r"quantile_test\(\): the input contains NaN \(nan_policy='raise'\)"):
        h.quantile_test(x, nan_policy="raise")
    # the E-test: a non-positive null estimate (0, 1) (scipy's); T = 0 two-sided: all but (0, 0)
    r = h.poisson_means_test(0, 2.0, 0, 3.0)
    assert (r.statistic, r.pvalue) == (0.0, 1.0)
    r = h.poisson_means_test(5, 1.0, 5, 1.0)
    assert r.statistic == 0.0 and math.isclose(r.pvalue, 1 - math.exp(-10.0), rel_tol=1e-15)
    # a zero standard error NaN (H24; statsmodels inf and 0)
    r = h.proportions_ztest(0, 30, value=0.5)
    assert nan(r.statistic) and nan(r.pvalue)
    # a zero column of the proportions' table NaN (chi2_contingency's rule), one sample df 0
    r = h.proportions_chisquare([5, 7], [5, 7])
    assert nan(r.statistic) and nan(r.pvalue)
    r = h.proportions_chisquare(12, 40)
    assert (r.statistic, r.pvalue, r.df) == (0.0, 1.0, 0.0)
    # nobs = 0: NaN intervals and tests
    assert np.isnan(np.asarray(h.proportion_confint([0, 1], [0, 2])[0])).all()
    assert nan(h.test_proportions_2indep(0, 0, 3, 5).statistic)
    # intervals at the ends: exactly 0 and 1 where the definitions put them
    for m in ("exact", "wilson", "wilsoncc", "binom_test"):
        assert h.proportion_confint(0, 30, method=m)[0] == 0.0 and h.proportion_confint(30, 30, method=m)[1] == 1.0
    # zero counts: the logs' IEEE ends (statsmodels'), Koopman's ratio 0, the score odds ratio 0
    lo, hi = h.confint_proportions_2indep(0, 40, 25, 45, compare="ratio", method="log").tolist()
    assert lo == 0.0 and nan(hi)
    assert h.confint_proportions_2indep(0, 40, 25, 45, compare="ratio", method="score")[0] == 0.0
    assert h.confint_proportions_2indep(0, 40, 25, 45, compare="odds-ratio", method="score")[0] == 0.0


# ---- R ------------------------------------------------------------------------------------


def test_r_recorded():
    """R's binom.test (its p-value, scipy's rule, and Clopper-Pearson's conf.int), prop.test with
    correct = FALSE (the two-sample z test squared, the k-sample chi^2, Wilson's interval) and with
    correct = TRUE for wilsoncc (Newcombe's interval, scipy's), and poisson.test (another test of
    two rates: its conditional binomial p-value, recorded for comparison only) agree to 1e-12 where
    they are the same quantity (tests/htest/data/r.json)."""
    data = R.data("r")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.milestone != "H7" or "result" not in data.get(pid, {}):
            continue
        got, r = data[pid]["result"], call(p)
        if any(v is None for vs in got.values() for v in vs):  # R's NA (a zero margin)
            continue
        if p.func == "binomtest":
            assert math.isclose(r.pvalue, got["p.value"][0], rel_tol=1e-12), pid
            if p.kwargs.get("alternative", "two-sided") == "two-sided":
                assert _close(r.conf_int(), got["conf.int"], 1e-12, 1e-15), pid
        elif p.func in ("proportions_ztest", "proportions_chisquare"):
            assert math.isclose(r.pvalue, got["p.value"][0], rel_tol=1e-12), pid
        elif p.func == "proportion_confint":
            assert _close(np.asarray(r), got["conf.int"], 1e-12, 1e-15), pid
        else:
            continue
        seen += 1
    assert seen >= 25


# ---- results and messages -----------------------------------------------------------------


def test_results():
    b = h.binomtest(7, 20, p=0.3)
    assert repr(b) == "BinomtestResult(k=7, n=20, alternative='two-sided', statistic=0.35, pvalue=0.629498)"
    assert isinstance(b.k, int) and isinstance(b.pvalue, float)
    ci = b.conf_int()
    ci[0] = 5.0  # a new array at each call
    assert b.conf_int()[0] != 5.0
    with pytest.raises(AttributeError):
        b.pvalue = 0.5
    q = h.quantile_test([1.0, 2, 3, 4, 5], q=2.5)
    assert q.statistic == 2.0 and q.statistic_type == 1.0
    with pytest.raises(TypeError):
        iter(q)
    x = np.arange(20.0)[::-1].copy()
    r = h.quantile_test(x, q=2.5)
    want = h.quantile_test(np.arange(20.0), q=2.5).conf_int().tolist()
    x[:] = 100.0  # the interval is the sample's at the call
    assert r.conf_int().tolist() == want == [5.0, 14.0]
    s, p = h.poisson_means_test(10, 5.0, 20, 6.0)
    z = h.proportions_ztest(np.array([[15.0, 25], [3, 9]]), np.array([[40.0, 45], [20, 22]]))
    assert repr(z) == "ProportionsZtestResult(batch shape (2,))" and z.pvalue.shape == (2,)
    c = h.proportions_chisquare([15, 25], [40, 45])
    with pytest.raises(ValueError):
        c.table[0, 0] = 1.0  # a read-only array (H21)
    assert repr(c).startswith("ProportionsChisquareResult(statistic=2.77103, pvalue=0.0959846, df=1, table=<array")
    t = h.test_proportions_2indep(np.array([15.0, 5]), 40, np.array([25.0, 9]), 45)
    assert t.method == "agresti-caffo" and t.statistic.shape == (2,) and t.value == 0.0
    ci = h.proportion_confint([7, 8], 30)
    assert ci.shape == (2, 2) and h.proportion_confint(7, 30).shape == (2,)
    low, upp = h.proportion_confint(7, 30)
    assert low < upp
    assert h.confint_proportions_2indep(np.array([15.0, 5, 30]), 40, np.array([25.0, 9, 30]), 45).shape == (3, 2)


def test_summaries():
    s = h.binomtest(7, 20, p=0.3).summary()
    assert (
        "Exact binomial test" in s
        and "No. Successes:" in s
        and "[0.025      0.975]" in s.replace("     0.975", "      0.975")
    )
    assert "Alternative:" in s and s.count("Alternative:") == 1 and "proportion" in s
    s = h.quantile_test(np.arange(10.0), q=2.5, alternative="greater").summary()
    assert "Quantile test" in s and "Statistic Type:" in s and "1 (x <= q)" in s
    s = h.test_proportions_2indep(15, 40, 25, 45, method="score", compare="odds-ratio", value=2.0).summary()
    assert "Compare:" in s and "odds-ratio" in s and "Prop1 null:" in s and "Correction:" in s
    s = h.proportions_chisquare([15, 25, 9], [40, 45, 30]).summary()
    assert "No. Samples:" in s and "expected_freq" in s
    with pytest.raises(TypeError, match=r"summary\(\): unexpected keyword argument 'alpha'"):
        h.poisson_means_test(10, 5.0, 20, 6.0).summary(alpha=0.05)


def test_messages():
    with pytest.raises(TypeError, match=r"binomtest\(\): 'k' must be an int, not float"):
        h.binomtest(3.0, 10)
    with pytest.raises(ValueError, match=r"binomtest\(\): 'k' must not exceed 'n'"):
        h.binomtest(11, 10)
    with pytest.raises(ValueError, match=r"binomtest\(\): 'n' must be >= 1"):
        h.binomtest(0, 0)
    with pytest.raises(ValueError, match=r"binomtest\(\): 'p' must be in \[0, 1\]"):
        h.binomtest(3, 10, p=1.5)
    with pytest.raises(ValueError, match=r"conf_int\(\): 'method' must be 'exact', 'wilson' or 'wilsoncc', not 'beta'"):
        h.binomtest(3, 10).conf_int(method="beta")
    with pytest.raises(ValueError, match=r"quantile_test\(\): 'p' must be in \(0, 1\)"):
        h.quantile_test([1.0, 2.0], p=1.0)
    with pytest.raises(ValueError, match=r"poisson_means_test\(\): 'n1' must be positive and finite"):
        h.poisson_means_test(3, 0.0, 2, 1.0)
    with pytest.raises(ValueError, match=r"poisson_means_test\(\): 'diff' must be a finite number >= 0"):
        h.poisson_means_test(3, 1.0, 2, 1.0, diff=-1.0)
    with pytest.raises(ValueError, match=r"proportions_ztest\(\): 'value' must be given for a one-sample test"):
        h.proportions_ztest(15, 40)
    with pytest.raises(ValueError, match=r"proportions_ztest\(\): 'count' must hold one or two samples along its "
                                         r"last axis, not 3"):  # fmt: skip
        h.proportions_ztest([1, 2, 3], [5, 5, 5])
    with pytest.raises(ValueError, match=r"proportions_ztest\(\): 'count' must not exceed 'nobs'"):
        h.proportions_ztest(41, 40, value=0.5)
    with pytest.raises(ValueError, match=r"proportions_chisquare\(\): 'count' must hold integer counts >= 0"):
        h.proportions_chisquare([1.5, 2], [4, 4])
    with pytest.raises(ValueError, match=r"proportion_confint\(\): 'alpha' must be between 0 and 1 \(exclusive\)"):
        h.proportion_confint(7, 30, alpha=0.0)
    with pytest.raises(ValueError, match=r"confint_proportions_2indep\(\): 'count1', 'nobs1', 'count2', 'nobs2' "
                                         r"could not be broadcast together"):  # fmt: skip
        h.confint_proportions_2indep([1, 2], [3, 4, 5], 1, 3)
    with pytest.raises(TypeError, match=r"test_proportions_2indep\(\): 'correction' must be"):
        h.test_proportions_2indep(15, 40, 25, 45, correction=1)
