"""sd.htest's contingency tables (milestone H6; docs/api/htest.md §7, docs/design/htest.md §3.7,
§4): the accuracy contract against the Arb and exact-rational references (tests/htest/ref),
batches lane by lane, the thread count, the H6 decisions (H77 the exact r x c Fisher test, H78
Barnard's and Boschloo's certified supremum whatever n, H79 odds_ratio and relative_risk, H80
weighted tables, H81 mcnemar's exact default, H82 df a float), scipy's and statsmodels'
definitions, R's chisq.test, fisher.test and mcnemar.test, the degenerate tables, the results
and the messages."""

import math
import time
import warnings

import numpy as np
import pytest
import scipy.stats as st
from scipy.stats import contingency as ct

import stardust as sd

from .._bits import bits_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H6
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
call = R.call
h = sd.htest
# scipy's and statsmodels' own warnings on the degenerate tables they are compared on (sd warns never)
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning:scipy", "ignore::RuntimeWarning:statsmodels")


def _quiet(f, *a, **k):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def _close(a, b, rtol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """Statistics to 1e-14 (exact rationals rounded once, or Arb balls), p-values to 1e-14
    max(1, kappa_p) (the exact null distributions: hypergeometric, r x c tables, binomial; the
    certified suprema of Barnard and Boschloo), the expected counts exactly rounded, the odds
    ratio's and the relative risk's interval ends to 1e-14 each."""
    rec = REFS[fam][pid]
    r = call(P.PROBLEMS[pid])
    for field, (e, c) in R.errors(fam, pid, r).items():
        assert c <= 1.0, (field, e)
    if "expected_freq" in rec:
        assert np.asarray(r.expected_freq).tolist() == rec["expected_freq"] or _close(
            r.expected_freq, np.asarray(rec["expected_freq"])[..., 0], 2.3e-16
        )


def test_every_h6_function_has_references():
    assert {P.PROBLEMS[pid].func for _, pid in CASES} == {
        "chi2_contingency",
        "fisher_exact",
        "barnard_exact",
        "boschloo_exact",
        "mcnemar",
        "cochrans_q",
        "odds_ratio",
        "relative_risk",
    }


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H6" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch of tables (the leading axes) gives each table's bits of the table alone (H9)."""
    p = P.PROBLEMS[pid]
    r = call(p)
    tables = np.asarray(p.args[0])
    lanes = tables.shape[0]
    assert np.asarray(r.statistic).shape == (lanes,)
    for i in range(lanes):
        one = call(P.Problem(p.id, p.func, (tables[i],), p.kwargs))
        for f in type(one)._FIELDS:
            got, want = getattr(r, f), getattr(one, f)
            if isinstance(want, sd.ndarray):
                assert bits_equal(np.asarray(got)[i], np.asarray(want)), (i, f)
            else:
                assert bits_equal(np.asarray(got)[i], want), (i, f)
        if hasattr(one, "conf_int"):
            assert bits_equal(np.asarray(r.conf_int())[i], np.asarray(one.conf_int()))


def test_threads_do_not_change_the_bits():
    rng = np.random.default_rng(23)
    t22 = rng.integers(0, 30, (4000, 2, 2)).astype(float)
    t35 = rng.integers(0, 20, (2000, 3, 5)).astype(float)
    small = rng.integers(0, 12, (12, 2, 2)).astype(float) + 1.0
    rxc = rng.integers(0, 5, (8, 3, 3)).astype(float)
    q = rng.integers(0, 2, (500, 40, 4)).astype(float)
    calls = [
        lambda: h.chi2_contingency(t35),
        lambda: h.chi2_contingency(t22, lambda_="log-likelihood"),
        lambda: h.fisher_exact(t22),
        lambda: h.fisher_exact(rxc),
        lambda: h.barnard_exact(small),
        lambda: h.boschloo_exact(small),
        lambda: h.mcnemar(t22),
        lambda: h.cochrans_q(q),
        lambda: h.odds_ratio(small),
    ]
    before = sd.get_num_threads()
    try:
        want = None
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            got = [np.asarray([np.asarray(r.statistic), np.asarray(getattr(r, "pvalue", r.statistic))])
                   for r in (c() for c in calls)]  # fmt: skip
            got.append(np.asarray(h.odds_ratio(small).conf_int()))
            if want is None:
                want = got
            else:
                for g, w in zip(got, want):
                    assert bits_equal(g, w)
    finally:
        sd.set_num_threads(before)


# ---- scipy's and statsmodels' definitions -------------------------------------------------


def _scipy_barnard_region_is_exact(t, alt, pooled):
    """scipy's region (rounded Wald statistics) is the exact one: no mirror-image tie is lost."""
    n1, n2 = int(t[0, 0] + t[1, 0]), int(t[0, 1] + t[1, 1])
    if n1 == 0 or n2 == 0:
        return True
    x1, x2 = np.arange(n1 + 1).reshape(-1, 1), np.arange(n2 + 1).reshape(1, -1)
    p1, p2 = x1 / n1, x2 / n2
    if pooled:
        v = (x1 + x2) / (n1 + n2) * (1 - (x1 + x2) / (n1 + n2)) * (1 / n1 + 1 / n2)
    else:
        v = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    with np.errstate(all="ignore"):
        w = (p1 - p2) / np.sqrt(v)
    w[p1 == p2] = 0
    # exact: T^2 as (D^2, den) cross-multiplied
    D = x1 * n2 - x2 * n1
    den = (x1 + x2) * (n1 + n2 - x1 - x2) if pooled else x1 * (n1 - x1) * n2**3 + x2 * (n2 - x2) * n1**3
    a, b = int(t[0, 0]), int(t[0, 1])
    Do, do = D[a, b], den[a, b]
    num = D.astype(object) ** 2 * int(do)
    ref = int(Do) ** 2 * den.astype(object)
    with np.errstate(all="ignore"):
        if alt == "two-sided":
            exact = np.where(D == 0, Do == 0, np.where(den == 0, True, (do != 0) & (num >= ref)))
            got = np.abs(w) >= abs(w[a, b])
        else:
            return True  # (the one-sided regions are checked through the p-values only)
    return bool(np.all(exact == got))


@pytest.mark.parametrize("seed", [0, 1])
def test_the_tests_are_scipys(seed):
    """Away from scipy's losses the values are scipy 1.18.1's and statsmodels 0.15.0's to rounding:
    chi2_contingency with every lambda_ and weighted counts, fisher_exact 2 x 2, Boschloo's (scipy's
    shgo finds the supremum to ~1e-9 here), Barnard's where scipy's rounded region is the exact one
    (elsewhere sd's p-value is larger: it has the mirror-image tables), the odds ratio (scipy's
    brentq to 1e-13 absolute), the relative risk, McNemar and Cochran's Q."""
    rng = np.random.default_rng(seed)
    for _ in range(60):
        r, c = rng.integers(2, 5, 2)
        t = rng.integers(0, 15, (r, c)).astype(float)
        w = t + np.round(rng.uniform(0, 1, t.shape), 2)
        for tab in (t, w):
            for corr in (True, False):
                for lam in (None, "log-likelihood", "freeman-tukey", "cressie-read", 0.4):
                    try:
                        b = _quiet(st.chi2_contingency, tab, correction=corr, lambda_=lam)
                    except ValueError:  # a zero margin: NaN here (H11)
                        assert math.isnan(h.chi2_contingency(tab, correction=corr, lambda_=lam).statistic)
                        continue
                    a = h.chi2_contingency(tab, correction=corr, lambda_=lam)
                    if not (lam == "freeman-tukey" and (tab == 0).any()):  # (scipy's NaN of a zero count)
                        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-10), (tab, corr, lam)
                    assert _close(a.expected_freq, b.expected_freq, 1e-14) and a.df == b.dof
        t2 = t[:2, :2]
        for alt in ("two-sided", "less", "greater"):
            a, b = h.fisher_exact(t2, alternative=alt), st.fisher_exact(t2, alternative=alt)
            assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-13)
            a, b = h.boschloo_exact(t2, alternative=alt), _quiet(st.boschloo_exact, t2, alternative=alt, n=64)
            assert _close(a.statistic, b.statistic, 1e-13) and _close(a.pvalue, b.pvalue, 1e-8), (t2, alt)
            assert not a.pvalue < b.pvalue * (1 - 1e-12)  # a supremum: never below scipy's maximum
            for pooled in (True, False):
                a = h.barnard_exact(t2, alternative=alt, pooled=pooled)
                b = _quiet(st.barnard_exact, t2, alternative=alt, pooled=pooled, n=64)
                assert _close(a.statistic, b.statistic, 1e-13)
                if _scipy_barnard_region_is_exact(t2, alt, pooled) and alt == "two-sided":
                    assert _close(a.pvalue, b.pvalue, 1e-8), (t2, alt, pooled)
                assert not a.pvalue < b.pvalue * (1 - 1e-12), (t2, alt, pooled)
        for kind in ("conditional", "sample"):
            a, b = h.odds_ratio(t2, kind=kind), ct.odds_ratio(t2.astype(np.int64), kind=kind)
            assert _close(a.statistic, b.statistic, 1e-12)
            for alt in ("two-sided", "less", "greater"):
                ci = b.confidence_interval(alternative=alt)
                assert _close(a.conf_int(alternative=alt), [ci.low, ci.high], 1e-10 if kind == "conditional" else 1e-14)
        n1, n2 = rng.integers(1, 40, 2)
        e1, e2 = rng.integers(0, n1 + 1), rng.integers(0, n2 + 1)
        a, b = h.relative_risk(e1, n1, e2, n2), ct.relative_risk(e1, n1, e2, n2)
        ci = b.confidence_interval()
        assert _close(a.relative_risk, b.relative_risk, 1e-15) and _close(a.conf_int(), [ci.low, ci.high], 1e-14)
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

    for _ in range(40):
        t2 = rng.integers(0, 40, (2, 2)).astype(float)
        for ex in (True, False):
            for corr in (True, False):
                a, b = h.mcnemar(t2, exact=ex, correction=corr), _quiet(mcnemar, t2, exact=ex, correction=corr)
                if t2[0, 1] + t2[1, 0] > 0 or ex:
                    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-13)
        shape = (int(rng.integers(3, 40)), int(rng.integers(2, 6)))
        x = (rng.uniform(size=shape) < rng.uniform(0.2, 0.8)).astype(float)
        a, b = h.cochrans_q(x), _quiet(cochrans_q, x)
        assert _close([a.statistic, a.pvalue, a.df], [b.statistic, b.pvalue, b.df], 1e-13)


def test_statsmodels_recorded():
    """statsmodels' recorded mcnemar and cochrans_q (tests/htest/data/statsmodels.json): the same to
    1e-13 (the binomial tail of mcnemar/large: 3.2e-14) but where b + c = 0 under the chi-squared form
    (statsmodels inf and 0, NaN here: H11)."""
    data = R.data("statsmodels")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.func not in ("mcnemar", "cochrans_q") or "result" not in data.get(pid, {}):
            continue
        got, r = data[pid]["result"], call(p)
        if pid == "mcnemar/zero-off-chi2":
            assert got["statistic"] == math.inf and math.isnan(r.statistic)
            continue
        assert _close(np.ravel(r.statistic), np.ravel(got["statistic"]), 1e-14), pid
        assert _close(np.ravel(r.pvalue), np.ravel(got["pvalue"]), 1e-13), pid
        seen += 1
    assert seen >= 12


# ---- the H6 decisions ---------------------------------------------------------------------


def test_fisher_rxc_is_the_network_algorithm():
    """H77: an r x c table's p-value is exact (R's fisher.test rule, every table at most (1 + 1e-7)
    times as probable as the observed one), no Monte Carlo and no rng; its statistic is the observed
    table's probability (scipy's); R's fisher.test (its FEXACT merges past values within 3.45e-7 in
    log) agrees to 1e-12 on the problems, and tables past R's default workspace come out exact in
    well under its time; past the work cap a ValueError that names chi2_contingency."""
    data = R.data("r")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.func != "fisher_exact" or "rxc" not in p.tags or "batch" in p.tags:
            continue
        got = (data.get(pid) or {}).get("result")
        r = call(p)
        if "error" in data[pid]:  # fisher.test refuses a table of one row
            assert (r.statistic, r.pvalue) == (1.0, 1.0)
            continue
        assert math.isclose(r.pvalue, got["p.value"][0], rel_tol=1e-12), pid
        seen += 1
    assert seen >= 7
    # the statistic: K / prod x! of the observed table
    t = np.array([[3.0, 1, 4], [1, 5, 9], [2, 6, 5]])
    rows, cols = t.sum(axis=1).astype(int), t.sum(axis=0).astype(int)
    lp = sum(math.lgamma(v + 1) for v in rows) + sum(math.lgamma(v + 1) for v in cols)
    lp -= math.lgamma(t.sum() + 1) + sum(math.lgamma(v + 1) for v in t.ravel())
    assert math.isclose(h.fisher_exact(t).statistic, math.exp(lp), rel_tol=1e-12)
    # tables R's default workspace refuses (FEXACT error 7), R's values with workspace = 2e8 (7.3 s
    # and 0.42 s there): 1.7e-10 and 4.8e-13 apart (no exact reference of these sizes; the exact
    # rational ones of the smaller tables agree with sd to 1e-14 and with R to 1e-12)
    for big, want, tol in (
        ([[1.0, 2, 5, 20, 2], [9, 0, 1, 10, 9], [0, 0, 10, 0, 1], [3, 8, 5, 5, 9]], 2.8429560682259534e-10, 3e-10),
        ([[5.0, 13, 6, 8, 4, 3, 4, 6, 1, 6], [1, 7, 1, 5, 18, 10, 4, 23, 17, 8]], 4.0755503111255204e-06, 1e-12),
    ):
        t0 = time.perf_counter()
        p = h.fisher_exact(big).pvalue
        assert time.perf_counter() - t0 < 10.0
        assert math.isclose(p, want, rel_tol=tol)
    with pytest.raises(ValueError, match=r"fisher_exact\(\): the exact r x c test needs more than its cap of 2e\+08 "
                                         r"operations for this table; use chi2_contingency"):  # fmt: skip
        h.fisher_exact([[0.0, 0, 2, 0, 1, 1], [2, 0, 2, 0, 0, 3], [4, 1, 3, 1, 2, 2], [1, 2, 3, 0, 1, 3],
                        [1, 3, 4, 0, 1, 2], [1, 2, 1, 5, 2, 4]])  # fmt: skip
    # the 2 x 2 table is the hypergeometric (scipy's), not the r x c rule
    assert h.fisher_exact([[3.0, 7], [9, 2]]).pvalue == call(P.PROBLEMS["fisher_exact/small"]).pvalue
    one = h.fisher_exact([[3.0, 4, 5]])
    assert (one.statistic, one.pvalue) == (1.0, 1.0)


def test_nuisance_supremum_whatever_n():
    """H78: n is only the first grid's size; the p-value is the supremum certified to 2^-48
    whatever n (scipy's shgo with n = 32 misses it by 1e-7 on some tables; here n = 1 is enough)."""
    for t in ([[3.0, 7], [9, 2]], [[20.0, 12], [8, 25]], [[7.0, 12], [8, 3]], [[25.0, 2], [1, 22]]):
        for f, kw in ((h.barnard_exact, {}), (h.barnard_exact, {"pooled": False}), (h.boschloo_exact, {})):
            ps = [f(t, n=n, **kw).pvalue for n in (1, 4, 32, 300)]
            assert max(ps) <= min(ps) * (1 + 8e-15), (t, ps)
    # scipy's grid misses the maximum: [[3, 8], [2, 2]] unpooled two-sided (scipy n = 64: 1.2e-7 low)
    a = h.barnard_exact([[3.0, 8], [2, 2]], pooled=False)
    b = _quiet(st.barnard_exact, [[3.0, 8], [2, 2]], pooled=False, n=64)
    assert 1e-8 < (a.pvalue - b.pvalue) / a.pvalue < 1e-6
    # past (n1 + 1)(n2 + 1) = 1e7 tables: NaN (the statistic is still the observed one)
    r = h.barnard_exact([[2000.0, 1], [3000, 4000]])
    assert math.isnan(r.pvalue) and math.isfinite(r.statistic)
    t0 = time.perf_counter()
    r = h.boschloo_exact([[150.0, 120], [150, 180]])
    assert time.perf_counter() - t0 < 10.0 and 0 < r.pvalue < 1


def test_barnard_mirror_ties():
    """[[7, 12], [8, 3]]: the tables (3, 8) and (8, 3) are as extreme as the observed one exactly
    (scipy's rounded Wald statistics are one ulp below and leave them out: its p-value 9.5e-4 low)."""
    t = [[7.0, 12], [8, 3]]
    a, b = h.barnard_exact(t), _quiet(st.barnard_exact, t)
    assert a.pvalue > b.pvalue * 1.0009
    # the same region as Boschloo's here: the same supremum
    assert math.isclose(a.pvalue, h.boschloo_exact(t).pvalue, rel_tol=1e-14)


def test_odds_ratio_and_relative_risk():
    """H79: the estimates of scipy.stats.contingency; the conditional odds ratio and its exact
    interval from Fisher's noncentral hypergeometric (R's fisher.test estimate and conf.int are the
    same quantities, found by uniroot to its default tolerance: 6e-4 off on odds_ratio/conditional's
    lower end); conf_int's alternative; relative_risk's Katz interval."""
    data = R.data("r")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.func != "odds_ratio" or p.kwargs.get("kind", "conditional") != "conditional":
            continue
        got = (data.get(pid) or {}).get("result")
        if not got or "estimate" not in got:
            continue
        r = call(p)
        if math.isnan(r.statistic):  # a zero margin (R: 0 and (0, inf))
            assert r.conf_int().tolist() == got["conf.int"]
            continue
        if math.isfinite(r.statistic) and r.statistic > 0:
            assert math.isclose(r.statistic, got["estimate"][0], rel_tol=2e-3), pid
        ci = r.conf_int()
        for i in range(2):
            want = got["conf.int"][i]
            assert (ci[i] == want) if not math.isfinite(want) or want == 0 else math.isclose(ci[i], want, rel_tol=2e-3)
        seen += 1
    assert seen >= 4
    r = h.odds_ratio([[3.0, 7], [9, 2]])
    lo, hi = r.conf_int().tolist()
    assert r.conf_int(alternative="less")[0] == 0.0 and r.conf_int(alternative="greater")[1] == math.inf
    assert lo < r.conf_int(alternative="greater")[0] < r.statistic < r.conf_int(alternative="less")[1] < hi
    assert r.conf_int(alpha=0.1)[0] > lo
    rr = h.relative_risk(9, 20, 4, 21)
    assert rr.relative_risk == 9 * 21 / (20 * 4) and rr.exposed_cases == 9 and rr.control_total == 21
    assert not hasattr(rr, "pvalue") and not hasattr(r, "pvalue")


def test_weighted_tables():
    """H80: chi2_contingency takes non-integer counts (scipy's), the exact tests do not."""
    w = np.array([[3.5, 4.25, 1.0], [1.5, 6.0, 2.75]])
    a, b = h.chi2_contingency(w), st.chi2_contingency(w)
    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-13)
    for f in (h.fisher_exact, h.barnard_exact, h.boschloo_exact, h.mcnemar, h.odds_ratio):
        with pytest.raises(ValueError, match=r"'table' must hold non-negative integer counts"):
            f(w[:, :2])
    with pytest.raises(ValueError, match=r"chi2_contingency\(\): 'observed' must hold non-negative finite counts"):
        h.chi2_contingency([[1.0, -1.0], [2.0, 3.0]])


def test_mcnemar_is_exact_by_default():
    """H81: exact=True (statsmodels'): the binomial test of b of b + c; exact=False the chi-squared
    form with statsmodels' correction (|b - c| - 1)^2 / (b + c) (R's mcnemar.test does not correct
    b = c)."""
    t = [[30.0, 12], [4, 25]]
    r = h.mcnemar(t)
    assert r.statistic == 4.0 and r.pvalue == min(1.0, 2 * sd.stats.Binomial(n=16, p=0.5).cdf(4.0))
    r = h.mcnemar(t, exact=False)
    assert r.statistic == (12 - 4 - 1) ** 2 / 16
    assert h.mcnemar([[10.0, 6], [6, 12]], exact=False).statistic == 1 / 12
    assert h.mcnemar([[10.0, 0], [0, 12]]).pvalue == 1.0


def test_df_is_a_float():
    """H82: chi2_contingency's df (scipy's int dof) is a float, as every df (H22); cochrans_q's too."""
    r = h.chi2_contingency([[3.0, 7, 1], [9, 2, 4]])
    assert isinstance(r.df, float) and r.df == 2.0
    stat, p, df, exp = r
    assert isinstance(df, float) and exp.shape == (2, 3)
    assert isinstance(h.cochrans_q(np.eye(3)).df, float)


# ---- degenerate tables --------------------------------------------------------------------


def test_degenerate_tables():
    nan = math.isnan
    # chi2_contingency: a zero margin NaN (scipy raises), df = 0 gives 0 and 1 (scipy's)
    r = h.chi2_contingency([[0.0, 0, 0], [3, 4, 5]])
    assert nan(r.statistic) and nan(r.pvalue) and r.df == 2.0
    r = h.chi2_contingency([[3.0, 4, 5]])
    assert (r.statistic, r.pvalue, r.df) == (0.0, 1.0, 0.0)
    r = h.chi2_contingency([[0.0, 0], [0, 0]])
    assert nan(r.statistic) and np.isnan(np.asarray(r.expected_freq)).all()
    # Yates within half: |O - E| <= 1/2 corrects to 0
    assert h.chi2_contingency([[5.0, 5], [5, 6]]).statistic == 0.0
    # a zero count with lambda <= -1: +inf and 0 (the limit, power_divergence's)
    r = h.chi2_contingency([[0.0, 5], [6, 2]], lambda_="neyman", correction=False)
    assert r.statistic == math.inf and r.pvalue == 0.0
    # fisher_exact 2 x 2 with a zero margin: NaN and 1 (scipy's)
    r = h.fisher_exact([[0.0, 4], [0, 5]])
    assert nan(r.statistic) and r.pvalue == 1.0
    # Barnard NaN and 1, Boschloo NaN and NaN for a zero column (scipy's)
    r = h.barnard_exact([[0.0, 3], [0, 4]])
    assert nan(r.statistic) and r.pvalue == 1.0
    r = h.boschloo_exact([[0.0, 3], [0, 4]])
    assert nan(r.statistic) and nan(r.pvalue)
    # mcnemar b + c = 0: exact (0, 1), chi-squared NaN (H11; statsmodels inf and 0)
    r = h.mcnemar([[3.0, 0], [0, 4]], exact=False)
    assert nan(r.statistic) and nan(r.pvalue)
    # cochrans_q: every block all 0 or all 1
    r = h.cochrans_q([[1.0, 1, 1], [0, 0, 0]])
    assert nan(r.statistic) and nan(r.pvalue) and r.df == 2.0
    # odds_ratio with a zero margin: NaN and (0, inf)
    r = h.odds_ratio([[0.0, 0], [3, 4]])
    assert nan(r.statistic) and r.conf_int().tolist() == [0.0, math.inf]
    # the deep tail of a 2 x 2 table: 2 / C(1000, 500) = 7.4e-300, a normal double
    p = h.fisher_exact([[500.0, 0], [0, 500]]).pvalue
    assert math.isclose(p, 2 / math.comb(1000, 500), rel_tol=1e-14)


# ---- R ------------------------------------------------------------------------------------


def test_r_recorded():
    """R's chisq.test (but where it reads a one-row matrix as a goodness-of-fit test), fisher.test
    (2 x 2 and r x c) and mcnemar.test (but b = c, which R does not correct, and b + c = 0) agree to
    1e-12 (tests/htest/data/r.json); the p-values against the references are
    test_reference_data.py's."""
    data = R.data("r")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.func not in ("chi2_contingency", "fisher_exact", "mcnemar") or pid.startswith("mcnemar/equal-chi2"):
            continue
        got = (data.get(pid) or {}).get("result")
        if not got or pid in ("chi2_contingency/one-row", "chi2_contingency/zero-row", "mcnemar/zero-off-chi2"):
            continue
        r = call(p)
        if "binom.test" in data[pid]["call"]:
            assert math.isclose(r.pvalue, got["p.value"][0], rel_tol=1e-12), pid
        else:
            assert math.isclose(r.pvalue, got["p.value"][0], rel_tol=1e-12), pid
            if p.func != "fisher_exact":
                assert math.isclose(r.statistic, got["statistic"][0], rel_tol=1e-12), pid
        seen += 1
    assert seen >= 30


# ---- results and messages -----------------------------------------------------------------


def test_results():
    t = [[3.0, 7], [9, 2]]
    r = h.chi2_contingency(t)
    stat, p, df, exp = r
    assert len(r) == 4 and r[3] is r.expected_freq and isinstance(stat, float)
    with pytest.raises(ValueError):
        r.expected_freq[0, 0] = 1.0  # a read-only array (H21)
    with pytest.raises(AttributeError):
        r.statistic = 1.0
    stat, p = h.fisher_exact(t)
    assert len(h.fisher_exact(t)) == 2
    for res in (h.barnard_exact(t), h.boschloo_exact(t), h.mcnemar(t), h.cochrans_q(np.eye(3)), h.odds_ratio(t),
                h.relative_risk(9, 20, 4, 21)):  # fmt: skip
        with pytest.raises(TypeError):
            iter(res)
    assert repr(r).startswith("Chi2ContingencyResult(statistic=3.82216, pvalue=0.050579, df=1, expected_freq=<array")
    b = h.fisher_exact(np.ones((3, 2, 2)))
    assert repr(b) == "FisherExactResult(batch shape (3,))" and b.pvalue.shape == (3,)
    ci = h.odds_ratio(t).conf_int()
    ci[0] = 5.0  # a new array at each call
    assert h.odds_ratio(t).conf_int()[0] != 5.0
    a = np.array([[3.0, 7], [9, 2]])
    o = h.odds_ratio(a)
    a[0, 0] = 30.0  # the interval is the table's, not the caller's array later
    assert o.conf_int().tolist() == h.odds_ratio([[3.0, 7], [9, 2]]).conf_int().tolist()


def test_messages():
    t = [[3.0, 7], [9, 2]]
    with pytest.raises(TypeError, match=r"chi2_contingency\(\): 'method' must be None, not str"):
        h.chi2_contingency(t, method="permutation")
    with pytest.raises(TypeError, match=r"fisher_exact\(\): 'method' must be None, not str"):
        h.fisher_exact(t, method="exact")
    with pytest.raises(ValueError, match=r"chi2_contingency\(\): 'observed' must have at least 2 dimensions"):
        h.chi2_contingency([3.0, 4.0])
    with pytest.raises(ValueError, match=r"fisher_exact\(\): 'table' must be at least a 1 x 1 table, not 0 x 2"):
        h.fisher_exact(np.zeros((0, 2)))
    with pytest.raises(ValueError, match=r"fisher_exact\(\): 'alternative' must be 'two-sided' for a table that is "
                                         r"not 2 x 2, not 'less'"):  # fmt: skip
        h.fisher_exact([[1.0, 2, 3], [4, 5, 6]], alternative="less")
    with pytest.raises(ValueError, match=r"fisher_exact\(\): 'alternative' must be 'two-sided', 'less' or 'greater'"):
        h.fisher_exact(t, alternative="smaller")
    for f in (h.barnard_exact, h.boschloo_exact, h.mcnemar, h.odds_ratio):
        with pytest.raises(ValueError, match=rf"{f.__name__}\(\): 'table' must be a 2 x 2 table, not 2 x 3"):
            f([[1.0, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match=r"barnard_exact\(\): 'n' must be >= 1"):
        h.barnard_exact(t, n=0)
    with pytest.raises(TypeError, match=r"boschloo_exact\(\): 'n' must be an int, not float"):
        h.boschloo_exact(t, n=32.0)
    with pytest.raises(TypeError, match=r"barnard_exact\(\): 'pooled' must be"):
        h.barnard_exact(t, pooled=1)
    with pytest.raises(TypeError, match=r"mcnemar\(\): 'exact' must be"):
        h.mcnemar(t, exact="yes")
    with pytest.raises(ValueError, match=r"cochrans_q\(\): 'x' must hold 0 and 1 values \(the successes\)"):
        h.cochrans_q([[0.0, 2.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match=r"odds_ratio\(\): 'kind' must be 'conditional' or 'sample', not 'exact'"):
        h.odds_ratio(t, kind="exact")
    with pytest.raises(ValueError, match=r"conf_int\(\): 'alternative' must be"):
        h.odds_ratio(t).conf_int(alternative="up")
    with pytest.raises(ValueError, match=r"conf_int\(\): 'alpha' must be between 0 and 1 \(exclusive\)"):
        h.odds_ratio(t).conf_int(alpha=0.0)
    with pytest.raises(ValueError, match=r"relative_risk\(\): 'exposed_cases' must not exceed 'exposed_total'"):
        h.relative_risk(5, 4, 1, 2)
    with pytest.raises(ValueError, match=r"relative_risk\(\): 'control_total' must be >= 1"):
        h.relative_risk(1, 4, 0, 0)
    with pytest.raises(TypeError, match=r"relative_risk\(\): 'exposed_cases' must be an int, not float"):
        h.relative_risk(1.0, 4, 1, 2)
    with pytest.raises(TypeError, match=r"chi2_contingency\(\): unexpected keyword argument 'dof'"):
        h.chi2_contingency(t, dof=1)
    with pytest.raises(TypeError, match=r"chi2_contingency\(\): 'lambda_' must be None, a real number or one of"):
        h.chi2_contingency(t, lambda_=[1])
