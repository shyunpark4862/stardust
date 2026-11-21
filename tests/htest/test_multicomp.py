"""sd.htest's multiple comparisons and corrections (milestone H8; docs/api/htest.md §9,
docs/design/htest.md §3.9, §4): the accuracy contract against the references (tests/htest/ref:
Arb's integrals for tukey_hsd and dunnett, exact rationals for the corrections, Arb for
combine_pvalues), batches lane by lane, the thread count, the H8 decisions (H102 one kernel for
false_discovery_control and multipletests, H103 the default method, H104 the fields and their
unpacking, H105 axis, H106 and H109 the methods left out, H108 no rng, H110 max_iter, H111 the
rejections, H112 the p-values' range, H113 df), the identities (two groups are the t test, one
treatment is the t test, Dunnett's one-sided tail is sd.stats' multivariate t), scipy's,
statsmodels' and R's results, the degenerate inputs, the results and the messages."""

import math
import warnings

import numpy as np
import pytest
import scipy.stats as st
from statsmodels.stats import multitest as smt

import stardust as sd

from .._bits import bits_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H8
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
call = R.call
h = sd.htest
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning:scipy", "ignore::RuntimeWarning:statsmodels")


def _close(a, b, rtol, atol=0.0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def _groups(seed, sizes, locs, scales=None):
    rng = np.random.default_rng(seed)
    scales = scales or [1.0] * len(sizes)
    return [rng.normal(m, s, n) for n, m, s in zip(sizes, locs, scales)]


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """The mean differences and t statistics to 1e-14, the studentized range's and the multivariate
    t's tails to 1e-14 max(1, kappa_p) (the references' outer quadrature is estimated to 2^-80),
    the interval ends to 1e-14 of the interval's size, the corrected p-values to 1e-14 (exact
    rationals), the rejections exactly, the combinations to 1e-14 max(1, kappa_p)."""
    errs = R.errors(fam, pid)
    assert errs, pid
    for field, (e, c) in errs.items():
        assert c <= 1.0, (field, e)


def test_every_h8_function_has_references():
    assert {P.PROBLEMS[pid].func for _, pid in CASES} == {
        "tukey_hsd",
        "dunnett",
        "multipletests",
        "false_discovery_control",
        "combine_pvalues",
    }
    assert sum("conf_int" in rec for rec in REFS["tukey_hsd"].values()) >= 3
    assert sum("conf_int" in rec for rec in REFS["dunnett"].values()) >= 2


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H8" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch (axis=-1) gives each lane's bits of the lane alone."""
    p = P.PROBLEMS[pid]
    r = call(p)
    x = np.asarray(p.args[0])
    kw = {k: v for k, v in p.kwargs.items() if k != "axis"}
    for i in range(x.shape[0]):
        one = call(P.Problem(p.id, p.func, (x[i],), kw))
        if p.func == "false_discovery_control":
            assert bits_equal(np.asarray(r)[i], np.asarray(one)), i
        elif p.func == "multipletests":
            assert bits_equal(np.asarray(r.pvalues_corrected)[i], np.asarray(one.pvalues_corrected)), i
            assert bits_equal(np.asarray(r.reject)[i], np.asarray(one.reject)), i
            assert r.alpha_sidak == one.alpha_sidak and r.alpha_bonferroni == one.alpha_bonferroni
        else:
            assert bits_equal(float(r.statistic[i]), one.statistic) and bits_equal(float(r.pvalue[i]), one.pvalue)


def test_threads_do_not_change_the_bits():
    g = _groups(31, [8, 9, 7, 10, 8, 9], [0.0, 0.4, 0.8, -0.3, 1.2, 0.1])
    rng = np.random.default_rng(32)
    pv = rng.uniform(0, 0.3, (300, 40))
    calls = [
        lambda: h.tukey_hsd(*g).pvalue,
        lambda: h.tukey_hsd(*g, equal_var=False).conf_int(),
        lambda: h.dunnett(*g[1:], control=g[0]).pvalue,
        lambda: h.dunnett(*g[1:], control=g[0], alternative="less").conf_int(),
        lambda: h.multipletests(pv, method="hommel", axis=-1).pvalues_corrected,
        lambda: h.multipletests(pv, method="fdr_tsbky", axis=1).reject,
        lambda: h.false_discovery_control(pv, axis=0),
        lambda: h.combine_pvalues(pv, axis=-1, method="stouffer").pvalue,
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
                for a, b in zip(got, want):
                    assert bits_equal(a, b)
    finally:
        sd.set_num_threads(before)


# ---- identities ---------------------------------------------------------------------------


def test_two_groups_are_the_t_test():
    """With k = 2 the studentized range is sqrt(2) |T|: tukey_hsd's p-value is ttest_ind's
    (Student's; Games-Howell: Welch's) and its interval the t interval."""
    for seed in range(4):
        a, b = _groups(seed, [9 + seed, 12], [0.0, 0.7])
        t = h.ttest_ind(a, b)
        r = h.tukey_hsd(a, b)
        assert math.isclose(float(r.pvalue[0, 1]), t.pvalue, rel_tol=2e-14), seed
        assert _close(np.asarray(r.conf_int())[0, 1], t.conf_int(), 1e-13, 1e-15)
        w = h.ttest_ind(a, b, equal_var=False)
        g = h.tukey_hsd(a, b, equal_var=False)
        assert math.isclose(float(g.pvalue[0, 1]), w.pvalue, rel_tol=2e-14), seed
        assert math.isclose(float(g.df[0, 1]), w.df, rel_tol=1e-14)


def test_one_treatment_is_the_t_test():
    """With one treatment Dunnett's test is the pooled two-sample t test, each alternative."""
    for seed in range(3):
        a, c = _groups(10 + seed, [8, 11], [0.6, 0.0])
        for alt in ("two-sided", "less", "greater"):
            t = h.ttest_ind(a, c, alternative=alt)
            r = h.dunnett(a, control=c, alternative=alt)
            assert math.isclose(float(r.statistic[0]), t.statistic, rel_tol=1e-15)
            assert math.isclose(float(r.pvalue[0]), t.pvalue, rel_tol=2e-14), (seed, alt)
            assert _close(np.asarray(r.conf_int())[0], t.conf_int(), 1e-13, 1e-15)


def test_one_sided_dunnett_is_the_multivariate_t():
    """Dunnett's one-sided tail is 1 - the multivariate t's cdf at (t, ..., t) with the
    one-factor correlations (sd.stats' MultivariateT: exact formulas for d <= 3)."""
    for d in (2, 3):
        g = _groups(40 + d, [9, 10, 8, 11][: d + 1], [0.0, 0.5, 0.9, 0.2][: d + 1])
        r = h.dunnett(*g[1:], control=g[0], alternative="greater")
        n = np.array([len(x) for x in g[1:]], dtype=float)
        lam = np.sqrt(n / (n + len(g[0])))
        cov = np.outer(lam, lam)
        np.fill_diagonal(cov, 1.0)
        df = sum(len(x) for x in g) - d - 1
        mvt = sd.stats.MultivariateT(mu=sd.zeros(d), shape=sd.asarray(cov), df=float(df))
        for i in range(d):
            t = float(r.statistic[i])
            want = 1.0 - float(mvt.cdf(sd.full((d,), t)))
            assert math.isclose(float(r.pvalue[i]), want, rel_tol=1e-12, abs_tol=1e-15), (d, i)


def test_studentized_range_tail_is_accurate_where_scipy_is_not():
    """scipy's studentized_range.sf loses the tail: P(Q > 30; 3, 27) is 0 in scipy, 6.79e-18 (Arb:
    the reference tukey_hsd problems hold the kernel to 1e-14); P(Q > 12; 4, 200) is 1.1e-13 in
    scipy, 2.8705e-14."""
    k = sd._core.htest
    assert math.isclose(k.srange_sf(30.0, 3.0, 27.0), 6.790744229598e-18, rel_tol=1e-11)
    assert st.studentized_range.sf(30, 3, 27) == 0.0
    assert math.isclose(k.srange_sf(12.0, 4.0, 200.0), 2.870465236004e-14, rel_tol=1e-11)
    assert st.studentized_range.sf(12, 4, 200) > 3 * k.srange_sf(12.0, 4.0, 200.0)
    # k = 2: Q = sqrt(2) |T|
    for q in (0.5, 3.0, 10.0):
        assert math.isclose(k.srange_sf(q, 2.0, 10.0), 2 * st.t.sf(q / math.sqrt(2), 10), rel_tol=1e-14)


# ---- scipy's, statsmodels' and R's results ------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_scipy_tukey_hsd(seed):
    """tukey_hsd's mean differences are scipy's to rounding, its p-values to scipy's quadrature
    (studentized_range.sf, some 1e-9; above 1e-8: scipy's tail is lost below), its intervals to 1e-9
    (scipy's ppf), df scipy's private _df."""
    rng = np.random.default_rng(seed)
    for _ in range(4):
        k = int(rng.integers(2, 6))
        g = [rng.normal(rng.normal(), 1.0, int(rng.integers(3, 15))) for _ in range(k)]
        for ev in (True, False):
            a, b = h.tukey_hsd(*g, equal_var=ev), st.tukey_hsd(*g, equal_var=ev)
            assert _close(a.statistic, b.statistic, 1e-14, 1e-15)
            big = b.pvalue > 1e-8
            assert _close(np.asarray(a.pvalue)[big], b.pvalue[big], 1e-7)
            ci = b.confidence_interval()
            assert _close(np.asarray(a.conf_int())[..., 0], ci.low, 1e-9, 1e-12)
            assert _close(np.asarray(a.df), b._df, 1e-14)


def test_scipy_dunnett():
    """dunnett's statistics are scipy's to rounding; scipy's p-values are its randomized
    quasi-Monte Carlo's (abseps ~1e-6 for d >= 2) and its intervals minimize_scalar's to tol 1e-3."""
    for seed in range(3):
        g = _groups(50 + seed, [10, 9, 12, 8], [0.0, 0.6, -0.2, 1.0])
        for alt in ("two-sided", "less", "greater"):
            a = h.dunnett(*g[1:], control=g[0], alternative=alt)
            b = st.dunnett(*g[1:], control=g[0], alternative=alt, rng=7)
            assert _close(a.statistic, b.statistic, 1e-14)
            assert _close(a.pvalue, b.pvalue, 1e-3, 3e-5), (seed, alt)
            ci = b.confidence_interval()
            fin = np.isfinite(ci.low)
            assert _close(np.asarray(a.conf_int())[:, 0][fin], np.asarray(ci.low)[fin], 1e-3, 1e-3)


def test_scipy_recorded():
    """scipy's recorded results (tests/htest/data/scipy_h8.json): tukey_hsd's mean differences and
    p-values (away from scipy's lost tail; the offset problem's 1.9e-7 is scipy's cancelling mean
    difference), dunnett's statistics, false_discovery_control and combine_pvalues to scipy's
    rounding (Stouffer's deep tail: scipy's statistic's own rounding, 5e-14); dunnett's p-values to
    scipy's quasi-Monte Carlo at rng=12345 (9e-5)."""
    data = R.data("scipy_h8")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.milestone != "H8" or p.func in P.STATSMODELS or "result" not in data.get(pid, {}):
            continue
        if "degenerate" in p.tags:  # (scipy's 0/0 and inf; NaN here, H24)
            continue
        got, r = data[pid]["result"], call(p)
        if p.func == "false_discovery_control":
            assert _close(np.asarray(r), got["value"], 1e-15, 1e-300), pid
        elif p.func == "combine_pvalues":
            assert _close(r.statistic, got["statistic"], 1e-14), pid
            assert _close(r.pvalue, got["pvalue"], 1e-12, 1e-300), pid
        elif p.func == "tukey_hsd":
            if "offset" not in p.tags:  # (scipy's np.mean of values at 1e8: 2.5e-8)
                assert _close(r.statistic, got["statistic"], 1e-12, 1e-14), pid
            pw = np.asarray(got["pvalue"], dtype=float)
            ok = pw > 1e-8
            assert _close(np.asarray(r.pvalue)[ok], pw[ok], 1e-6 if "offset" in p.tags else 1e-7), pid
        else:
            if "offset" not in p.tags:  # (scipy's np.mean at 1e9: 5e-8)
                assert _close(r.statistic, got["statistic"], 1e-12), pid
            assert _close(r.pvalue, got["pvalue"], 1e-3, 2e-4), pid  # (scipy's QMC: 9e-5 off)
        seen += 1
    assert seen >= 35


def test_statsmodels_multipletests():
    """multipletests is statsmodels' on every method, alpha and maxiter to rounding (its sidak
    forms are the same expm1/log1p; its alphacSidak 1 - (1 - alpha)^(1/m) cancels, 1e-14), and its
    rejections are statsmodels' but at an exact tie of a rounded comparison."""
    rng = np.random.default_rng(3)
    for t in range(60):
        m = int(rng.integers(1, 40))
        p = rng.uniform(0, 1, m) ** rng.uniform(1, 5)
        if t % 4 == 0:
            p = np.round(p, 2)
        alpha = float(rng.choice([0.05, 0.1, 0.2]))
        for meth in P.MULTITEST + P.MULTITEST_TWO_STAGE:
            mi = int(rng.integers(0, 4))
            a = h.multipletests(p, method=meth, alpha=alpha, max_iter=mi)
            b = smt.multipletests(p, method=meth, alpha=alpha, maxiter=mi)
            assert _close(a.pvalues_corrected, b[1], 1e-14, 1e-300), (t, meth)
            # (statsmodels' 1 - (1 - alpha)^(1/m) cancels: 1.3e-14 off at m = 32)
            assert math.isclose(a.alpha_sidak, b[2], rel_tol=1e-12) and a.alpha_bonferroni == b[3]
            if not np.array_equal(np.asarray(a.reject), b[0]):  # only where a value sits on alpha
                diff = np.asarray(a.reject) != b[0]
                assert np.allclose(np.asarray(a.pvalues_corrected)[diff], alpha, rtol=1e-14), (t, meth)


def test_statsmodels_recorded():
    data = R.data("statsmodels")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.func != "multipletests" or "result" not in data.get(pid, {}):
            continue
        got, r = data[pid]["result"], call(p)
        assert _close(r.pvalues_corrected, got["pvals_corrected"], 1e-14, 1e-300), pid
        if pid != "multipletests/fdr_tsbh":  # statsmodels' rounded 14/20 x 0.1 (H111)
            assert np.array_equal(np.asarray(r.reject), np.asarray(got["reject"])), pid
        seen += 1
    assert seen >= 25


def test_r_recorded():
    """R's p.adjust (every shared method) to 1e-15, TukeyHSD's diff to rounding and its p adj (R's
    ptukey, about 1e-9) to 1e-7, its interval (qtukey) to 1e-7; multcomp's Dunnett statistics to
    rounding, its p-values and intervals (mvtnorm's randomized quasi-Monte Carlo, abseps 1e-7) to
    1e-5; Fisher's combination (pchisq) to 1e-13."""
    data = R.data("r")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.milestone != "H8" or "result" not in data.get(pid, {}) or "degenerate" in p.tags:
            continue  # (R's constant groups: q or t of 1e15, p adj 5.4e-10; NaN here, H24)
        got, r = data[pid]["result"], call(p)
        if p.func in ("multipletests", "false_discovery_control"):
            v = r.pvalues_corrected if p.func == "multipletests" else r
            assert _close(v, got["value"], 1e-15, 1e-300), pid
        elif p.func == "tukey_hsd":
            if {"offset", "tail"} & set(p.tags):  # R's mean difference cancels (1.3e-7); ptukey's tail
                continue  # is 1 - cdf (3e-10 for 7.8e-21)
            k = np.asarray(r.statistic).shape[0]
            pairs = [(j, i) for i in range(k) for j in range(i + 1, k)]  # R's "j-i" rows
            if any(v is None for v in got["p adj"]):
                continue
            assert _close([float(r.statistic[a, b]) for a, b in pairs], got["diff"], 1e-12, 1e-14), pid
            assert _close([float(r.pvalue[a, b]) for a, b in pairs], got["p adj"], 1e-6, 1e-9), pid
            ci = np.asarray(r.conf_int())
            assert _close([ci[a, b, 0] for a, b in pairs], got["lwr"], 1e-6, 1e-6), pid  # (qtukey: eps 1e-4)
        elif p.func == "dunnett":
            if any(v is None for v in got["p.value"]):
                continue
            if "offset" not in p.tags:  # (R's aov at 1e9: 2.2e-7)
                assert _close(r.statistic, got["statistic"], 1e-12), pid
            assert _close(r.pvalue, got["p.value"], 1e-4, 1e-5), pid
        elif p.func == "combine_pvalues":
            assert _close(r.pvalue, got["p.value"], 1e-13, 1e-300), pid
        seen += 1
    assert seen >= 30


# ---- the H8 decisions ---------------------------------------------------------------------


def test_one_kernel_for_the_fdr_corrections():
    """H102: false_discovery_control(method='bh' or 'by') and multipletests(method='fdr_bh' or
    'fdr_by').pvalues_corrected are the same bits, batches too."""
    rng = np.random.default_rng(8)
    for p in (P.PVALS, rng.uniform(0, 1, 300) ** 4, np.round(rng.uniform(0, 0.2, 50), 2)):
        for a, b in (("bh", "fdr_bh"), ("by", "fdr_by")):
            assert bits_equal(np.asarray(h.false_discovery_control(p, method=a)),
                              np.asarray(h.multipletests(p, method=b).pvalues_corrected))  # fmt: skip
    x = rng.uniform(0, 0.3, (4, 6, 5))
    assert bits_equal(np.asarray(h.false_discovery_control(x, axis=1)),
                      np.asarray(h.multipletests(x, method="fdr_bh", axis=1).pvalues_corrected))  # fmt: skip
    assert not hasattr(h, "fdrcorrection")


def test_default_method_fields_and_unpacking():
    """H103: statsmodels' default 'holm-sidak'. H104: reject, pvalues_corrected, alpha_sidak,
    alpha_bonferroni, unpacking as statsmodels' four. H113: df."""
    a = h.multipletests(P.PVALS)
    b = h.multipletests(P.PVALS, method="holm-sidak")
    assert bits_equal(np.asarray(a.pvalues_corrected), np.asarray(b.pvalues_corrected))
    reject, corrected, s, bonf = a
    assert len(a) == 4 and a[1] is not None
    assert reject.dtype == sd.bool and isinstance(s, float) and bonf == 0.05 / 20
    assert type(a).__name__ == "MultipletestsResult"
    with pytest.raises(AttributeError):
        a.reject = None
    with pytest.raises(ValueError):
        np.asarray(a.pvalues_corrected)[0] = 1.0  # (a read-only array)
    t = h.tukey_hsd([1.0, 2, 3], [2.0, 3, 5], [0.0, 1, 1.5])
    assert t.df == 6.0 and isinstance(t.df, float)
    with pytest.raises(TypeError):
        iter(t)
    d = h.dunnett([1.0, 2, 3], control=[2.0, 3, 5])
    assert d.df == 4.0
    with pytest.raises(TypeError):
        iter(d)
    s_, p_ = h.combine_pvalues([0.1, 0.2])
    assert isinstance(s_, float) and isinstance(p_, float)


def test_axis():
    """H105: multipletests reads one family per lane along axis (an int), axis=None one family of
    every value (1-D, as scipy's false_discovery_control and sd.stats' rankdata); a scalar is one
    p-value."""
    x = np.random.default_rng(9).uniform(0, 0.2, (3, 4))
    flat = h.multipletests(x)
    assert tuple(flat.pvalues_corrected.shape) == (12,)
    assert bits_equal(np.asarray(flat.pvalues_corrected), np.asarray(h.multipletests(x.ravel()).pvalues_corrected))
    cols = h.multipletests(x, axis=0)
    assert tuple(cols.pvalues_corrected.shape) == (3, 4)
    assert bits_equal(np.asarray(cols.pvalues_corrected)[:, 2], np.asarray(h.multipletests(x[:, 2]).pvalues_corrected))
    assert cols.alpha_bonferroni == 0.05 / 3
    assert h.false_discovery_control(0.3) == 0.3 and isinstance(h.false_discovery_control(0.3), float)
    one = h.multipletests(0.01, method="bonferroni")
    assert one.pvalues_corrected == 0.01 and one.reject is True
    e = h.multipletests([])
    assert tuple(e.pvalues_corrected.shape) == (0,) and math.isnan(e.alpha_sidak)
    with pytest.raises(TypeError, match=r"^multipletests\(\): 'axis' must be an int or None, not tuple$"):
        h.multipletests(x, axis=(0, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        h.false_discovery_control(x, axis=2)


def test_methods():
    """H102, H106, H109: statsmodels' ten long names only (no abbreviations, no fdr_gbs, no lfdr)."""
    for m in P.MULTITEST + P.MULTITEST_TWO_STAGE:
        h.multipletests(P.PVALS, method=m)
    for bad in ("hs", "b", "fdr_gbs", "lfdr", "Holm", "fdr_i"):
        with pytest.raises(ValueError, match=r"^multipletests\(\): 'method' must be"):
            h.multipletests(P.PVALS, method=bad)
    with pytest.raises(ValueError, match=r"^false_discovery_control\(\): 'method' must be 'bh' or 'by', not 'BH'$"):
        h.false_discovery_control(P.PVALS, method="BH")


def test_max_iter():
    """H110: max_iter (statsmodels' maxiter) is an int >= 0, the two stages' count; the iteration
    stops when the rejections stop changing, so a large max_iter is statsmodels' -1."""
    p = P.PVALS
    full = smt.multipletests(p, method="fdr_tsbh", maxiter=-1)[1]
    assert _close(h.multipletests(p, method="fdr_tsbh", max_iter=len(p)).pvalues_corrected, full, 1e-14)
    with pytest.raises(ValueError, match=r"^multipletests\(\): 'max_iter' must be >= 0$"):
        h.multipletests(p, max_iter=-1)
    with pytest.raises(TypeError, match=r"^multipletests\(\): 'max_iter' must be an int, not bool$"):
        h.multipletests(p, max_iter=True)


def test_rejections_are_the_procedures():
    """H111: the rejection is the exact corrected p-value against alpha, not the rounded one: fdr_tsbh
    on the problems' p-values corrects 0.07 to 0.05 + 5e-18 (rounded: 0.05 = alpha) and does not
    reject it; an exact tie (0.01 x 5 = 0.05 exactly) rejects."""
    r = h.multipletests(P.PVALS, method="fdr_tsbh")
    assert float(r.pvalues_corrected[13]) == 0.05 and not bool(r.reject[13])
    t = h.multipletests([0.01, 0.3, 0.6, 0.8, 0.02], method="bonferroni")
    assert float(t.pvalues_corrected[0]) == 0.05 and bool(t.reject[0])
    for m in P.MULTITEST:  # away from ties the rule is the corrected value against alpha (0.1 is one:
        x = h.multipletests(P.PVALS, method=m, alpha=0.13)  # fdr_bh's 0.07 x 20 / 14)
        assert np.array_equal(np.asarray(x.reject), np.asarray(x.pvalues_corrected) <= 0.13)


def test_p_value_range():
    """H112: the corrections' p-values lie in [0, 1] without NaN; combine_pvalues' NaN follow
    nan_policy, the rest must lie in [0, 1]."""
    for bad in ([0.1, 1.2], [-0.1, 0.5], [0.1, np.nan]):
        with pytest.raises(ValueError, match=r"^multipletests\(\): 'pvals' must hold p-values between 0 and 1"):
            h.multipletests(bad)
        with pytest.raises(ValueError, match=r"^false_discovery_control\(\): 'ps' must hold p-values"):
            h.false_discovery_control(bad)
    with pytest.raises(ValueError, match=r"^combine_pvalues\(\): 'pvalues' must hold p-values between 0 and 1$"):
        h.combine_pvalues([0.1, 1.5])
    x = [0.1, np.nan, 0.3]
    assert math.isnan(h.combine_pvalues(x).pvalue)
    assert bits_equal(h.combine_pvalues(x, nan_policy="omit").pvalue, h.combine_pvalues([0.1, 0.3]).pvalue)
    with pytest.raises(ValueError, match=r"nan_policy='raise'"):
        h.combine_pvalues(x, nan_policy="raise")
    w = h.combine_pvalues([0.1, 0.2, 0.3], method="stouffer", weights=[1.0, np.nan, 2.0], nan_policy="omit")
    assert bits_equal(w.pvalue, h.combine_pvalues([0.1, 0.3], method="stouffer", weights=[1.0, 2.0]).pvalue)


def test_dunnett_takes_no_rng():
    """H108: deterministic (the one-factor quadrature), no rng; the same bits every call."""
    g = _groups(60, [10, 9, 11, 8, 12, 9], [0.0, 0.3, 0.6, 0.1, -0.4, 0.9])  # d = 5 (scipy: QMC)
    a = h.dunnett(*g[1:], control=g[0])
    b = h.dunnett(*g[1:], control=g[0])
    assert bits_equal(np.asarray(a.pvalue), np.asarray(b.pvalue))
    with pytest.raises(TypeError, match=r"^dunnett\(\): unexpected keyword argument 'rng'$"):
        h.dunnett(*g[1:], control=g[0], rng=sd.random.default_rng(1))


# ---- degenerate inputs --------------------------------------------------------------------


def test_degenerate_inputs():
    """H11, H24: constant groups (the standard error 0) give NaN p-values (scipy: 0/0 or inf), the
    mean differences stand; too few observations for the degrees of freedom give NaN; NaN
    propagates (tukey_hsd's pooled variance: every pair; Games-Howell's: that group's pairs)."""
    c = h.tukey_hsd([1.0, 1, 1], [2.0, 2], [3.0, 3, 3])
    assert np.all(np.isnan(np.asarray(c.pvalue))) and float(c.statistic[0, 2]) == -2.0
    assert bits_equal(np.asarray(c.conf_int())[0, 2], np.array([-2.0, -2.0]))
    one = h.tukey_hsd([1.0], [2.0], [3.0])  # N - k = 0
    assert np.all(np.isnan(np.asarray(one.pvalue)))
    n = h.tukey_hsd([1.0, np.nan, 2], [2.0, 3, 4], [0.0, 1, 3])
    assert np.all(np.isnan(np.asarray(n.pvalue))) and not math.isnan(float(n.statistic[1, 2]))
    gh = h.tukey_hsd([1.0, np.nan, 2], [2.0, 3, 4], [0.0, 1, 3], equal_var=False)
    assert math.isnan(float(gh.pvalue[0, 1])) and not math.isnan(float(gh.pvalue[1, 2]))
    small = h.tukey_hsd([1.0], [2.0, 3, 4], [0.0, 1, 3])  # a group of one: Tukey-Kramer still has N - k
    assert not math.isnan(float(small.pvalue[0, 1]))
    assert math.isnan(float(h.tukey_hsd([1.0], [2.0, 3, 4], [0.0, 1, 3], equal_var=False).pvalue[0, 1]))
    assert np.all(np.isnan(np.asarray(h.tukey_hsd([], [2.0, 3, 4], [0.0, 1, 3]).pvalue)))
    assert float(h.tukey_hsd([1.0, 2, 4], [2.0, 3, 4]).pvalue[0, 0]) == 1.0  # (the diagonal, scipy's)
    d = h.dunnett([1.0, 1, 1], control=[2.0, 2, 2])
    assert math.isnan(float(d.statistic[0])) and math.isnan(float(d.pvalue[0]))
    assert math.isnan(float(h.dunnett([1.0], control=[2.0]).pvalue[0]))  # df = 0
    assert math.isnan(h.combine_pvalues([]).pvalue)
    assert h.combine_pvalues([0.0, 0.5]).pvalue == 0.0 and h.combine_pvalues([1.0, 0.5], method="pearson").pvalue == 1.0


# ---- results and messages -----------------------------------------------------------------


def test_results_and_intervals():
    g = _groups(70, [8, 10, 9], [0.0, 0.8, 0.3])
    t = h.tukey_hsd(*g)
    ci = t.conf_int(alpha=0.1)
    assert tuple(ci.shape) == (3, 3, 2) and tuple(t.statistic.shape) == (3, 3)
    st_ = np.asarray(t.statistic)
    assert np.allclose(st_, -st_.T, rtol=0, atol=0) and np.allclose(np.asarray(t.pvalue), np.asarray(t.pvalue).T)
    assert np.all(np.asarray(ci)[..., 0] <= st_) and np.all(st_ <= np.asarray(ci)[..., 1])
    assert repr(t) == "TukeyHsdResult(statistic=<array of shape (3, 3)>, pvalue=<array of shape (3, 3)>, df=24)"
    d = h.dunnett(*g[1:], control=g[0], alternative="greater")
    dc = np.asarray(d.conf_int())
    assert tuple(dc.shape) == (2, 2) and np.all(np.isinf(dc[:, 1]))
    assert repr(d).startswith("DunnettResult(statistic=[")
    m = h.multipletests([0.01, 0.04, 0.2], method="holm")
    assert repr(m) == (
        "MultipletestsResult(reject=[True, False, False], pvalues_corrected=[0.03, 0.08, 0.2], "
        "alpha_sidak=0.0169524, alpha_bonferroni=0.0166667)"
    )
    with pytest.raises(ValueError, match=r"^conf_int\(\): 'alpha' must be between 0 and 1 \(exclusive\)$"):
        t.conf_int(alpha=0.0)
    b = h.combine_pvalues(np.random.default_rng(1).uniform(0, 1, (3, 5)), axis=-1)
    assert repr(b) == "CombinePvaluesResult(batch shape (3,))"


def test_summaries():
    g = _groups(71, [8, 10, 9], [0.0, 0.8, 0.3])
    s = h.tukey_hsd(*g).summary(alpha=0.1)
    assert s.splitlines()[0].strip() == "Tukey's HSD test"
    assert "(0 - 1)" in s and "(1 - 2)" in s and "(1 - 0)" not in s and "[0.050     0.950]" in s
    assert "Df:                                 24" in s and "No. Groups:" in s
    gh = h.tukey_hsd(*g, equal_var=False).summary()
    assert gh.splitlines()[0].strip() == "Games-Howell test" and "        df " in gh
    d = h.dunnett(*g[1:], control=g[0], alternative="less").summary()
    assert d.splitlines()[0].strip() == "Dunnett's test" and "(1 - control)" in d and "[0.000     0.950]" in d
    assert "Control Observations:                8" in d
    m = h.multipletests(np.linspace(0.001, 0.3, 30), method="fdr_bh").summary()
    assert "Method:                         fdr_bh" in m and "No. Tests:                          30" in m
    assert "[29]" in m and "[15]" not in m and sum(1 for line in m.splitlines() if line.startswith("...")) == 1
    c = h.combine_pvalues([0.01, 0.2, 0.3], method="stouffer").summary()
    assert c.splitlines()[0].strip() == "Combined p-values (Stouffer)" and "No. P-values:" in c
    with pytest.raises(TypeError, match=r"^summary\(\): unexpected keyword argument 'alpha'$"):
        h.multipletests([0.1]).summary(alpha=0.1)


def test_messages():
    with pytest.raises(TypeError, match=r"^tukey_hsd\(\): at least two samples are required, not 1$"):
        h.tukey_hsd([1.0, 2.0])
    with pytest.raises(TypeError, match=r"^tukey_hsd\(\): 'equal_var' must be True or False, not int$"):
        h.tukey_hsd([1.0, 2.0], [2.0, 3.0], equal_var=1)
    with pytest.raises(TypeError, match=r"^dunnett\(\): at least one sample besides the control is required, not 0$"):
        h.dunnett(control=[1.0, 2.0])
    with pytest.raises(TypeError, match=r"^dunnett\(\): missing 1 required keyword-only argument: 'control'$"):
        h.dunnett([1.0, 2.0])
    with pytest.raises(ValueError, match=r"^dunnett\(\): 'alternative' must be 'two-sided', 'less' or 'greater'"):
        h.dunnett([1.0, 2.0], control=[2.0, 3.0], alternative="larger")
    with pytest.raises(TypeError, match=r"^tukey_hsd\(\): 'samples\[1\]' must be an sd.ndarray"):
        h.tukey_hsd([1.0, 2.0], {"a": 1})
    with pytest.raises(ValueError, match=r"^multipletests\(\): 'alpha' must be between 0 and 1 \(exclusive\)$"):
        h.multipletests([0.1], alpha=1.0)
    with pytest.raises(ValueError, match=r"^combine_pvalues\(\): 'method' must be"):
        h.combine_pvalues([0.1], method="stouffers")
    with pytest.raises(ValueError, match=r"^combine_pvalues\(\): the paired samples must agree in size, not 3 and 2$"):
        h.combine_pvalues([0.1, 0.2, 0.3], method="stouffer", weights=[1.0, 2.0])
    with pytest.raises(TypeError, match=r"^multipletests\(\): unexpected keyword argument 'maxiter'"):
        h.multipletests([0.1], maxiter=2)
    with pytest.raises(TypeError, match=r"^false_discovery_control\(\): takes 1 positional argument but 2 were given$"):
        h.false_discovery_control([0.1], 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h.tukey_hsd([1.0, 1, 1], [2.0, 2, 2])  # no warnings (H11)
