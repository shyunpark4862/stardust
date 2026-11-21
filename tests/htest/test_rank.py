"""sd.htest's rank tests (milestone H3; docs/api/htest.md §4, docs/design/htest.md §3.4, §4): the
accuracy contract against the exact and Arb references (tests/htest/ref), the exact null
distributions against brute force and closed forms, the H3 decisions (H45-H50: the conditional
exact distribution with ties, scipy's 'auto' thresholds, zstatistic wherever the normal
approximation is used, the batch's per-lane methods, median_test's axis, ranksums), the work cap
(H13), batches lane by lane, the thread count, nan_policy, the degenerate inputs (H11), scipy's
definitions, the results and the messages."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from .._bits import bits_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H3
H3 = (
    "mannwhitneyu",
    "ranksums",
    "wilcoxon",
    "kruskal",
    "friedmanchisquare",
    "brunnermunzel",
    "ansari",
    "mood",
    "median_test",
    "page_trend_test",
)
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
call = R.call
h = sd.htest


def _ranks(v):
    return st.rankdata(v)


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """The statistics to 1e-14 (the rank statistics U, T+, AB, L exactly), the p-values to 1e-14
    max(1, kappa_p) (the exact ones are integer counts' quotients: 1e-14), z, df and the median to
    1e-14; the method the references follow (scipy's 'auto', H46) and median_test's table."""
    rec = REFS[fam][pid]
    r = call(P.PROBLEMS[pid])
    for field, (e, c) in R.errors(fam, pid, r).items():
        assert c <= 1.0, (field, e)
    if "method" in rec:
        assert r.method == rec["method"]
    if "table" in rec:
        assert np.asarray(r.table).tolist() == rec["table"]
    # zstatistic wherever the normal approximation was used (H47)
    if fam in ("mannwhitneyu", "wilcoxon"):
        assert hasattr(r, "zstatistic") == (r.method == "asymptotic")


def test_every_h3_function_has_references():
    assert {P.PROBLEMS[pid].func for _, pid in CASES} == set(H3)


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H3" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch (axis=-1) gives each lane's bits of the lane alone (axis=None); its method field is
    the lanes' names (H48) and zstatistic is NaN in its exact lanes."""
    p = P.PROBLEMS[pid]
    r = call(p)
    kw = {k: v for k, v in p.kwargs.items() if k != "axis"}
    lanes = p.args[0].shape[0]
    for i in range(lanes):
        args = [a[i] if isinstance(a, np.ndarray) and a.ndim == 2 else a for a in p.args]
        one = call(P.Problem(p.id, p.func, tuple(args), kw))
        for f in type(one)._FIELDS:
            if not hasattr(one, f):
                if hasattr(r, f):
                    assert math.isnan(float(np.asarray(getattr(r, f))[i])), (i, f)
                continue
            got, want = getattr(r, f), getattr(one, f)
            if isinstance(want, str):
                assert got[i] == want
            else:
                assert bits_equal(np.asarray(got)[i], want), (i, f)


def test_threads_do_not_change_the_bits():
    """The same bits on 1, 3 and 10 threads: long lanes (the sort), batches, the exact paths."""
    rng = np.random.default_rng(7)
    x, y, z = rng.normal(size=200_001), rng.normal(0.01, 1, 150_000), rng.normal(size=200_001)
    b, c = rng.normal(size=(2000, 30)), rng.normal(0.2, 1, (2000, 25))
    calls = [
        lambda: h.mannwhitneyu(x, y),
        lambda: h.wilcoxon(x, z),
        lambda: h.kruskal(x, y, z[:1000]),
        lambda: h.friedmanchisquare(x, z, x[::-1]),
        lambda: h.brunnermunzel(x, y),
        lambda: h.ansari(x, y),
        lambda: h.mood(x, y),
        lambda: h.median_test(x, y, z),
        lambda: h.mannwhitneyu(b, c, axis=-1),
        lambda: h.mannwhitneyu(b[:, :8], c, axis=-1),
        lambda: h.wilcoxon(b, axis=-1),
        lambda: h.ansari(b, c, axis=-1),
        lambda: h.page_trend_test(b.reshape(200, 10, 30)[:, :, :6]),
    ]
    before = sd.get_num_threads()
    try:
        want = None
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            got = [np.asarray([np.asarray(r.statistic), np.asarray(r.pvalue)]) for r in (f() for f in calls)]
            if want is None:
                want = got
            else:
                for g, w in zip(got, want):
                    assert bits_equal(g, w)
    finally:
        sd.set_num_threads(before)


# ---- the exact distributions --------------------------------------------------------------


def _brute_two_sample(scores, n1, observed):
    """(P(S <= s), P(S >= s)) of the sum of n1 of the scores over all subsets, exactly."""
    tot = le = ge = 0
    for c in itertools.combinations(range(len(scores)), n1):
        v = sum(scores[i] for i in c)
        tot += 1
        le += v <= observed
        ge += v >= observed
    return Fraction(le, tot), Fraction(ge, tot)


def test_exact_counts_against_brute_force():
    """The integer paths: #{U <= b} of [N, c]_q (the product formula), Ansari-Bradley's #{AB <= b}
    from the two score halves' Gaussian binomials, and C(N, c)."""
    k = sd._core.htest
    for N, c in ((7, 3), (12, 5), (13, 6), (9, 4)):
        us = [sum(s) - c * (c + 1) // 2 for s in itertools.combinations(range(1, N + 1), c)]
        sc = [min(r, N + 1 - r) for r in range(1, N + 1)]
        ab = [sum(sc[i] for i in s) for s in itertools.combinations(range(N), c)]
        assert k.exact_count(2, N, c, 0) == math.comb(N, c)
        for b in range(-1, max(us) + 2):
            assert k.exact_count(0, N, c, b) == sum(u <= b for u in us), (N, c, b)
        for b in range(min(ab) - 1, max(ab) + 2):
            assert k.exact_count(1, N, c, b) == sum(v <= b for v in ab), (N, c, b)
    # past 2^53: C(130, 65) (the integer path's largest sizes) and the symmetric middle
    big = k.exact_count(2, 130, 65, 0)
    assert big == float(math.comb(130, 65))
    assert k.exact_count(0, 130, 65, 65 * 65) == big


def test_conditional_exact_with_ties_is_the_permutation_distribution():
    """H45: with ties, method='exact' is the distribution of the statistic over the arrangements of
    the observed midranks (all subsets; all sign choices), by brute force here. scipy's 'exact'
    takes the distribution without ties: another p-value."""
    rng = np.random.default_rng(11)
    for _ in range(6):
        x, y = np.round(rng.normal(size=6) * 2), np.round(rng.normal(size=7) * 2 + 0.6)
        r = _ranks(np.concatenate([x, y]))
        le, ge = _brute_two_sample(list(2 * r), 6, 2 * r[:6].sum())
        for alt, want in (("less", le), ("greater", ge), ("two-sided", min(Fraction(1), 2 * min(le, ge)))):
            got = h.mannwhitneyu(x, y, method="exact", alternative=alt).pvalue
            assert abs(got - float(want)) <= 1e-15 * float(want), (alt, got, want)
        sym = np.minimum(r, 14 - r)
        le, ge = _brute_two_sample(list(2 * sym), 6, 2 * sym[:6].sum())
        got = h.ansari(x, y, method="exact")
        assert got.method == "exact" and abs(got.pvalue - float(min(1, 2 * min(le, ge)))) <= 1e-15
        d = np.round(rng.normal(size=10) * 2 + 0.5)
        d = d[d != 0]
        rk = _ranks(np.abs(d))
        tp = rk[d > 0].sum()
        sums = [sum(rk[i] for i in range(len(d)) if m >> i & 1) for m in range(2 ** len(d))]
        le = Fraction(sum(v <= tp for v in sums), len(sums))
        got = h.wilcoxon(d, method="exact", alternative="less").pvalue
        assert abs(got - float(le)) <= 1e-15
    # scipy 1.18.1's 'exact' ignores the ties
    x, y = np.array([1.0, 1, 2, 3, 3, 3]), np.array([2.0, 3, 4, 4, 5, 5, 6])
    assert abs(h.mannwhitneyu(x, y, method="exact").pvalue - st.mannwhitneyu(x, y, method="exact").pvalue) > 1e-3


def test_with_ties_the_null_need_not_be_symmetric():
    """Tie runs need not mirror each other, so the conditional distribution of U is not symmetric:
    both tails are counted (a reflection would give 0.183 here; the truth is 0.233)."""
    x, y = np.array([-2.0, -1, 0, 0, 1, 2, 2]), np.array([0.0, 3, 3])
    r = _ranks(np.concatenate([x, y]))
    le, ge = _brute_two_sample(list(2 * r), 7, 2 * r[:7].sum())
    assert le != ge
    got = h.mannwhitneyu(x, y, method="exact").pvalue
    assert abs(got - float(min(1, 2 * min(le, ge)))) <= 1e-15


def test_deep_exact_tails():
    """The exact tails as counts, not 1 - cdf: scipy's signed-rank sf is 1 - cdf above the mean (0
    where the truth is 2 / 2^60); complete separation of 30 and 30 is 2 / C(60, 30)."""
    x = np.random.default_rng(3).normal(3.0, 1.0, 60)
    r = h.wilcoxon(x, method="exact")
    assert r.pvalue == 2.0**-59 and st.wilcoxon(x, method="exact").pvalue == 0.0
    a, b = np.arange(30.0), np.arange(30.0) + 100
    assert math.isclose(h.mannwhitneyu(a, b, method="exact").pvalue, 2 / math.comb(60, 30), rel_tol=1e-15)
    # n = 200 (counts past 2^126: double-double probabilities)
    r = h.wilcoxon(np.arange(1.0, 201.0), method="exact", alternative="greater")
    assert r.pvalue == 2.0**-200


def test_ansari_where_scipy_drifts():
    """scipy's gscale (AS 93) is off by 1e-9-1e-7 for N of 40-108; here the integer counts, and
    the agreement with them through a brute-force enumeration at a smaller size."""
    x, y = P.PROBLEMS["ansari/exact-54"].args
    ref = REFS["ansari"]["ansari/exact-54"]["pvalue"][0]
    assert abs(h.ansari(x, y).pvalue / ref - 1) < 1e-15 and abs(st.ansari(x, y).pvalue / ref - 1) > 1e-8


# ---- H46: the 'auto' thresholds are scipy's -----------------------------------------------


def test_auto_thresholds():
    rng = np.random.default_rng(5)
    n = rng.normal
    assert h.mannwhitneyu(n(size=8), n(size=500)).method == "exact"
    assert h.mannwhitneyu(n(size=9), n(size=9)).method == "asymptotic"
    assert h.mannwhitneyu([1.0, 1, 2], [3.0, 4]).method == "asymptotic"  # ties
    assert h.wilcoxon(n(size=50) + 0.2).method == "exact"
    assert h.wilcoxon(n(size=51) + 0.2).method == "asymptotic"
    assert h.wilcoxon(np.round(n(size=13) * 2) + 0.5).method == "exact"  # ties, n <= 13: the signs' enumeration
    assert h.wilcoxon(np.round(n(size=14) * 2) + 0.5).method == "asymptotic"
    assert h.ansari(n(size=54), n(size=54)).method == "exact"
    assert h.ansari(n(size=55), n(size=10)).method == "asymptotic"
    assert h.page_trend_test(n(size=(12, 8))).method == "exact"
    assert h.page_trend_test(n(size=(13, 4))).method == "asymptotic"
    assert h.page_trend_test(n(size=(20, 3))).method == "exact"
    assert h.page_trend_test(n(size=(2, 9))).method == "asymptotic"
    # scipy's choices on the same inputs (its methods where it records them)
    d = n(size=(12, 8))
    assert h.page_trend_test(d).method == st.page_trend_test(d).method


def test_exact_work_cap():
    """H13: method='exact' past the cap (1e9 operations, 1e7 counts held) is a ValueError naming
    the way out; 'auto' takes the normal approximation there."""
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=8), rng.normal(size=300_000)
    assert h.mannwhitneyu(x, y).method == "asymptotic"
    with pytest.raises(
        ValueError,
        match=r"mannwhitneyu\(\): method='exact' needs about .* more than its cap of 1e\+09; use method='asymptotic'",
    ):  # noqa: E501
        h.mannwhitneyu(x, y, method="exact")
    with pytest.raises(ValueError, match=r"wilcoxon\(\): method='exact' needs about"):
        h.wilcoxon(rng.normal(size=3000), method="exact")
    with pytest.raises(ValueError, match=r"ansari\(\): method='exact' needs about"):
        h.ansari(np.round(rng.normal(size=400)), np.round(rng.normal(size=400)), method="exact")
    # a count past the doubles (page's work is inf here): "needs more than <cap>", no "about inf"
    with pytest.raises(
        ValueError, match=r"page_trend_test\(\): method='exact' needs more than 1e\+09 operations for this input; use"
    ):
        h.page_trend_test(rng.normal(size=(5, 25)), method="exact")
    with pytest.raises(ValueError) as e:
        h.page_trend_test(np.random.default_rng(2).normal(size=(200, 12)), method="exact")
    assert "inf" not in str(e.value)
    with pytest.raises(ValueError, match=r"needs ranks that are integers or halves"):
        h.page_trend_test(np.array([[1.0, 2.3, 3], [2.0, 1, 3]]), ranked=True, method="exact")


# ---- H47, H48: zstatistic and the batch's methods -----------------------------------------


def test_zstatistic_wherever_the_normal_approximation_is_used():
    """scipy 1.18.1 gives zstatistic only when the call asks for 'asymptotic'; here also when 'auto'
    picks it (H47), and not for the exact test."""
    rng = np.random.default_rng(8)
    x = rng.normal(0.3, 1, 70)
    r = h.wilcoxon(x)
    assert r.method == "asymptotic" and not hasattr(st.wilcoxon(x), "zstatistic")
    assert math.isclose(r.zstatistic, st.wilcoxon(x, method="asymptotic").zstatistic, rel_tol=1e-13)
    assert not hasattr(h.wilcoxon(x[:20]), "zstatistic")
    a, b = rng.normal(size=20), rng.normal(size=25)
    assert math.isclose(
        h.mannwhitneyu(a, b).zstatistic, st.mannwhitneyu(a, b, method="asymptotic").zstatistic, rel_tol=1e-13
    )
    assert not hasattr(h.mannwhitneyu(a[:5], b), "zstatistic")


def test_batch_methods_are_per_lane_lists():
    """H48: a batch's method is the lanes' names in nested lists of its shape, a new list at each
    read (G113); zstatistic is there when a lane used the normal approximation (NaN elsewhere)."""
    x = np.zeros((2, 3, 20))
    rng = np.random.default_rng(9)
    x[0] = rng.normal(0.2, 1, (3, 20))
    x[1] = np.round(rng.normal(0.2, 1, (3, 20)) * 2) + 0.5  # ties: 'auto' is normal for n = 20
    r = h.wilcoxon(x, axis=-1)
    assert r.method == [["exact"] * 3, ["asymptotic"] * 3]
    z = np.asarray(r.zstatistic)
    assert np.isnan(z[0]).all() and np.isfinite(z[1]).all()
    m = r.method
    m[0][0] = "changed"
    assert r.method[0][0] == "exact"
    r = h.wilcoxon(x[0, 0], keepdims=True)
    assert r.method == ["exact"]
    assert h.ks_1samp(rng.normal(size=(2, 10)), sd.stats.Normal(), axis=-1).method == ["exact", "exact"]


# ---- H49, H50: median_test's axis, ranksums -----------------------------------------------


def test_median_test_axis_and_batches():
    rng = np.random.default_rng(10)
    a, b, c = rng.normal(size=(4, 15)), rng.normal(0.5, 1, (4, 18)), rng.normal(-0.3, 1, (4, 12))
    r = h.median_test(a, b, c, axis=-1)
    assert np.asarray(r.statistic).shape == (4,) and np.asarray(r.table).shape == (4, 2, 3)
    assert r.table.dtype == sd.int64
    for i in range(4):
        one = h.median_test(a[i], b[i], c[i])
        s = st.median_test(a[i], b[i], c[i])
        assert bits_equal(np.asarray(r.statistic)[i], one.statistic)
        assert np.asarray(r.table)[i].tolist() == np.asarray(one.table).tolist() == s.table.tolist()
        assert math.isclose(one.statistic, s.statistic, rel_tol=1e-13) and one.median == s.median
    stat, p, med, table = h.median_test(a[0], b[0])
    assert np.asarray(table).shape == (2, 2) and r.df.tolist() == [2.0] * 4
    k = h.median_test(a, b, c, axis=-1, keepdims=True)
    assert np.asarray(k.statistic).shape == (4, 1) and np.asarray(k.table).shape == (4, 1, 2, 3)


def test_median_test_exact_median():
    """The grand median of an even count is the exact mean of the two middle values: a value equal
    to its rounding is above or below it, not tied with it."""
    lo = 1.0
    hi = np.nextafter(1.0, 2.0)
    a, b = np.array([lo, 0.0, 0.5]), np.array([hi, 3.0, 4.0])
    r = h.median_test(a, b, ties="above")
    assert r.median == (lo + hi) / 2 == lo  # (its rounding)
    assert np.asarray(r.table).tolist() == [[0, 3], [3, 0]]
    assert st.median_test(a, b, ties="above").table.tolist() == [[1, 3], [2, 0]]  # 1.0 "equal" to it


def test_ranksums_is_kept():
    """H50: scipy's legacy ranksums (no tie or continuity correction), the same ranks as
    mannwhitneyu's normal approximation without them on data without ties."""
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=30), rng.normal(0.4, 1, 40)
    r = h.ranksums(x, y)
    m = h.mannwhitneyu(x, y, method="asymptotic", use_continuity=False)
    assert math.isclose(r.statistic, m.zstatistic, rel_tol=1e-13) and math.isclose(r.pvalue, m.pvalue, rel_tol=1e-13)


# ---- scipy's definitions ------------------------------------------------------------------


def _close(a, b, rtol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_tests_are_scipys(seed):
    """On data without ties (and with ties for the normal approximations) the values are scipy
    1.18.1's to rounding: the exact distributions are scipy's where it has them right (Ansari only
    up to N = 30: scipy's gscale drifts above)."""
    rng = np.random.default_rng(seed)
    for ties in (False, True):
        x, y = rng.normal(size=12), rng.normal(0.4, 1.3, 15)
        if ties:
            x, y = np.round(2 * x), np.round(2 * y)
        three = [x, y, rng.normal(size=10) if not ties else np.round(rng.normal(size=10))]
        for alt in ("two-sided", "less", "greater"):
            for meth in ("auto", "asymptotic") + (() if ties else ("exact",)):
                for cont in (True, False):
                    a = h.mannwhitneyu(x, y, alternative=alt, method=meth, use_continuity=cont)
                    b = st.mannwhitneyu(x, y, alternative=alt, method=meth, use_continuity=cont)
                    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (alt, meth, cont)
                a, b = h.ansari(x, y, alternative=alt, method=meth), st.ansari(x, y, alternative=alt, method=meth)
                assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (alt, meth)
                for zm in ("wilcox", "pratt", "zsplit"):
                    if ties and meth != "asymptotic":
                        continue
                    a = h.wilcoxon(x, y[:12], zero_method=zm, alternative=alt, method=meth, correction=cont)
                    b = st.wilcoxon(x, y[:12], zero_method=zm, alternative=alt, method=meth, correction=cont)
                    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (zm, alt, meth)
            for name in ("ranksums", "mood"):
                a, b = getattr(h, name)(x, y, alternative=alt), getattr(st, name)(x, y, alternative=alt)
                assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), name
            for dist in ("t", "normal"):
                a = h.brunnermunzel(x, y, alternative=alt, distribution=dist)
                b = st.brunnermunzel(x, y, alternative=alt, distribution=dist)
                assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), dist
        for f in ("kruskal", "median_test"):
            a, b = getattr(h, f)(*three), getattr(st, f)(*three)
            assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), f
        a, b = h.friedmanchisquare(x[:10], y[:10], three[2]), st.friedmanchisquare(x[:10], y[:10], three[2])
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12)
        for t in ("below", "above", "ignore"):
            for corr in (True, False):
                for lam in (1, "log-likelihood", 0.3):
                    a = h.median_test(x, y, ties=t, correction=corr, lambda_=lam)
                    b = st.median_test(x, y, ties=t, correction=corr, lambda_=lam)
                    assert _close([a.statistic, a.pvalue, a.median], [b.statistic, b.pvalue, b.median], 1e-12)
    d = rng.normal(size=(10, 6)) + np.arange(6) * 0.2
    for meth in ("auto", "exact", "asymptotic"):
        for pr in (None, [2, 1, 3, 4, 6, 5]):
            a = h.page_trend_test(d, method=meth, predicted_ranks=pr)
            b = st.page_trend_test(d, method=meth, predicted_ranks=pr)
            assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12) and a.method == b.method


def test_brunnermunzel_degrees_of_freedom():
    """df is Satterthwaite's (scipy's, not in its result) and only with distribution='t'."""
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=10), rng.normal(0.5, 2, 14)
    r = h.brunnermunzel(x, y)
    assert r.pvalue == min(1.0, 2.0 * float(sd.stats.StudentT(df=r.df).ccdf(abs(r.statistic))))
    assert not hasattr(h.brunnermunzel(x, y, distribution="normal"), "df")


# ---- degenerate inputs, nan_policy, results, messages -------------------------------------


def test_degenerate_inputs():
    nan = math.isnan
    for f in (h.mannwhitneyu, h.ranksums, h.brunnermunzel, h.ansari, h.mood):
        r = f([], [1.0, 2.0])
        assert nan(r.statistic) and nan(r.pvalue), f.__name__
    # every value tied: no variance (scipy: NaN with a warning, or 0 and 1)
    r = h.mannwhitneyu(np.full(10, 2.0), np.full(12, 2.0))
    assert r.statistic == 60.0 and nan(r.pvalue) and nan(r.zstatistic)
    assert nan(h.kruskal(np.ones(3), np.ones(4)).statistic)
    assert nan(h.friedmanchisquare(np.ones(4), np.ones(4), np.ones(4)).pvalue)
    assert nan(h.mood([1.0], [2.0]).statistic)  # N < 3
    r = h.wilcoxon(np.zeros(8))
    assert nan(r.statistic) and nan(r.pvalue)
    # complete separation: both placement variances 0 (scipy: inf and a warning)
    r = h.brunnermunzel(np.arange(5.0), np.arange(5.0) + 10)
    assert nan(r.statistic) and nan(r.pvalue) and nan(r.df)
    assert nan(h.brunnermunzel([1.0], [2.0, 3.0]).statistic)
    # median_test: an empty row or column (scipy: ValueError)
    r = h.median_test(np.ones(3), np.ones(4))
    assert nan(r.statistic) and r.median == 1.0 and np.asarray(r.table).tolist() == [[0, 0], [3, 4]]
    r = h.median_test([1.0, 2.0, 3.0], [2.0], ties="ignore")
    assert nan(r.statistic)
    # a real scalar is a sample of one (H25)
    assert h.mannwhitneyu(1.0, [0.0, 2.0, 3.0]).statistic == 1.0


def test_nan_policy():
    rng = np.random.default_rng(14)
    x, y = rng.normal(size=20), rng.normal(0.3, 1, 18)
    xn = np.insert(x, 4, np.nan)
    for f in (h.mannwhitneyu, h.ranksums, h.wilcoxon, h.brunnermunzel, h.ansari, h.mood, h.kruskal, h.median_test):
        args = (xn, y) if f is not h.wilcoxon else (xn,)
        clean = (x, y) if f is not h.wilcoxon else (x,)
        r = f(*args)
        assert math.isnan(r.statistic) and math.isnan(r.pvalue), f.__name__
        a, b = f(*args, nan_policy="omit"), f(*clean)
        assert bits_equal([a.statistic, a.pvalue], [b.statistic, b.pvalue]), f.__name__
        with pytest.raises(ValueError, match=r"the input contains NaN \(nan_policy='raise'\)"):
            f(*args, nan_policy="raise")
    # the method of a NaN lane is the one its other values lead to (H12)
    assert h.wilcoxon(xn).method == "exact"
    # paired: omit drops the pair (wilcoxon) or the block (friedmanchisquare)
    y2 = rng.normal(size=21)
    a = h.wilcoxon(xn, y2, nan_policy="omit")
    b = h.wilcoxon(np.delete(xn, 4), np.delete(y2, 4))
    assert bits_equal([a.statistic, a.pvalue], [b.statistic, b.pvalue])
    z = rng.normal(size=21)
    a = h.friedmanchisquare(xn, y2, z, nan_policy="omit")
    b = h.friedmanchisquare(np.delete(xn, 4), np.delete(y2, 4), np.delete(z, 4))
    assert bits_equal([a.statistic, a.pvalue], [b.statistic, b.pvalue])


def test_wilcoxon_exact_differences():
    """d = x - y is exact (a double-double): x - y of 1e16 and a small y would round to a multiple
    of 2, merging or reordering the |d|; here they rank as the exact differences do."""
    x = np.array([1e16, 1e16, -1e16, 1e16, -1e16])
    y = np.array([-1.0, -3.0, 5.0, 7.0, -9.0])
    # exact |d| = 1e16 + 1, + 3, + 5, - 7, - 9: ranks 3, 4, 5, 2, 1 with signs +, +, -, +, -
    same = np.array([3.0, 4.0, -5.0, 2.0, -1.0])
    a, b = h.wilcoxon(x, y), h.wilcoxon(same)
    assert a.method == b.method == "exact" and (a.statistic, a.pvalue) == (b.statistic, b.pvalue)


def test_results():
    r = h.mannwhitneyu([0.1, 0.5, -0.3], [1.2, 0.8])
    assert r.method == "exact" and not hasattr(r, "zstatistic")
    stat, p = r
    assert len(r) == 2
    with pytest.raises(AttributeError):
        r.statistic = 1.0
    r = h.page_trend_test(np.arange(12.0).reshape(4, 3))
    with pytest.raises(TypeError):
        iter(r)
    assert r.method == "exact" and r.statistic == 56.0
    assert repr(h.kruskal([1.0, 2, 3], [4.0, 5])) == "KruskalResult(statistic=3, pvalue=0.0832645, df=1)"
    b = h.kruskal(np.arange(20.0).reshape(2, 10), np.arange(16.0).reshape(2, 8), axis=-1)
    assert repr(b) == "KruskalResult(batch shape (2,))"
    with pytest.raises(ValueError):
        b.statistic[0] = 1.0  # a read-only array (H21)
    assert len(h.median_test([1.0, 2, 3], [4.0, 5, 6])) == 4
    r = h.page_trend_test(np.arange(24.0).reshape(2, 4, 3))
    assert np.asarray(r.statistic).shape == (2,) and r.method == ["exact", "exact"]


def test_messages():
    x, y = [0.1, 0.4, -0.2, 1.3], [0.5, 0.2, 1.1]
    with pytest.raises(ValueError, match=r"mannwhitneyu\(\): 'method' must be"):
        h.mannwhitneyu(x, y, method="permutation")
    with pytest.raises(TypeError, match=r"mannwhitneyu\(\): 'use_continuity' must be"):
        h.mannwhitneyu(x, y, use_continuity=1)
    with pytest.raises(ValueError, match=r"wilcoxon\(\): 'zero_method' must be"):
        h.wilcoxon(x, zero_method="drop")
    with pytest.raises(ValueError, match=r"wilcoxon\(\): the paired samples must agree in size, not 4 and 3"):
        h.wilcoxon(x, y)
    with pytest.raises(ValueError, match=r"brunnermunzel\(\): 'distribution' must be"):
        h.brunnermunzel(x, y, distribution="z")
    with pytest.raises(TypeError, match=r"kruskal\(\): at least two samples are required, not 1"):
        h.kruskal(x)
    with pytest.raises(TypeError, match=r"friedmanchisquare\(\): at least three samples are required, not 2"):
        h.friedmanchisquare(x, x)
    with pytest.raises(TypeError, match=r"median_test\(\): at least two samples are required, not 1"):
        h.median_test(x)
    with pytest.raises(ValueError, match=r"median_test\(\): 'ties' must be"):
        h.median_test(x, y, ties="split")
    with pytest.raises(ValueError, match=r"median_test\(\): 'lambda_' must be"):
        h.median_test(x, y, lambda_="g-test")
    with pytest.raises(ValueError, match=r"page_trend_test\(\): 'data' must be at least a 2 x 3 table"):
        h.page_trend_test(np.ones((3, 2)))
    with pytest.raises(ValueError, match=r"page_trend_test\(\): 'data' must have at least 2 dimensions"):
        h.page_trend_test(np.ones(5))
    with pytest.raises(ValueError, match=r"page_trend_test\(\): 'data' must not contain NaN"):
        h.page_trend_test(np.array([[1.0, 2, np.nan], [1.0, 2, 3]]))
    with pytest.raises(ValueError, match=r"'predicted_ranks' must hold each integer from 1 to 3 once"):
        h.page_trend_test(np.ones((2, 3)), predicted_ranks=[1, 1, 2])
    with pytest.raises(ValueError, match=r"must hold ranks between 1 and 3 \(ranked=True\)"):
        h.page_trend_test(np.array([[1.0, 2, 4], [1.0, 2, 3]]), ranked=True)
    with pytest.raises(TypeError, match=r"page_trend_test\(\): unexpected keyword argument 'axis'"):
        h.page_trend_test(np.ones((2, 3)), axis=0)
    with pytest.raises(ValueError, match=r"ansari\(\): 'alternative' must be"):
        h.ansari(x, y, alternative="larger")


# ---- R's coin (H54) -----------------------------------------------------------------------

COIN = R.DATA / "r_coin.json"


@pytest.mark.skipif(not COIN.exists(), reason="no record of R's coin yet (r_coin.json beside r.json; H54)")
def test_coin_conditional_exact():
    """coin's exact tests conditional on the ties (ansari_test, wilcox_test, wilcoxsign_test with
    distribution = 'exact'; Page's test as friedman_test of an ordered treatment where coin gives an
    exact distribution) against sd's method='exact' (H45), one-sided (coin's two-sided rule is another):
    the same p-values to 1e-12, coin's direction one convention per test."""
    rec = R.data("r_coin")
    assert rec and all(r["digest"] == P.digest(P.PROBLEMS[pid]) for pid, r in rec.items())
    convention = {}
    seen = 0
    for pid, r in rec.items():
        res = r.get("result") or {}
        p = P.PROBLEMS[pid]
        kw = {k: v for k, v in p.kwargs.items() if k not in ("alternative", "method")}
        if p.func == "page_trend_test":
            got = (res.get("exact.p.value") or [None])[0]
            if got is None:
                continue  # (coin has no exact distribution for this one)
            n = p.args[0].shape[1]
            mine = (
                h.page_trend_test(p.args[0], method="exact").pvalue,
                h.page_trend_test(p.args[0], method="exact", predicted_ranks=list(range(n, 0, -1))).pvalue,
            )
            pair = (got,)
        else:
            f = getattr(h, p.func)
            mine = tuple(f(*p.args, **kw, alternative=a, method="exact").pvalue for a in ("less", "greater"))
            pair = tuple((res.get(f"{a}.p.value") or [None])[0] for a in ("less", "greater"))
            if None in pair:
                continue
        options = [mine, mine[::-1]]
        ok = [all(math.isclose(g, w, rel_tol=1e-12) for g, w in zip(pair, o)) for o in options]
        assert any(ok), (pid, pair, mine)
        convention.setdefault(p.func, ok.index(True))
        assert ok[convention[p.func]], (pid, "coin's direction changed")
        seen += 1
    assert seen >= 8


def test_batch_zstatistic_always_there():
    """H156: a batch of a test whose 'auto' may take the normal approximation always has
    zstatistic (NaN in its exact lanes), so the field does not come and go with the data; one
    test has it only when the approximation was used (H47)."""
    A = sd.asarray([[1.0, 2, 3, 4], [1.0, 2, 3, 4]])
    r = h.mannwhitneyu(A, sd.asarray([[5.0, 6, 7], [2.5, 2.2, 5]]), axis=1)
    assert all(math.isnan(v) for v in r.zstatistic.tolist())
    r = h.mannwhitneyu(A, sd.asarray([[5.0, 6, 7], [2.0, 2, 5]]), axis=1)
    z = r.zstatistic.tolist()
    assert math.isnan(z[0]) and z[1] == pytest.approx(-0.18345, rel=1e-4)
    assert h.mannwhitneyu(sd.zeros((0, 5)), sd.zeros((0, 4)), axis=1).zstatistic.shape == (0,)
    w = h.wilcoxon(sd.asarray([[0.5, -0.25, 1.0, 2.0], [0.5, 0.75, 1.0, 2.0]]), axis=1)
    assert w.zstatistic.shape == (2,)
    with pytest.raises(AttributeError):
        _ = h.mannwhitneyu([1.0, 2, 3, 4], [5.0, 6, 7]).zstatistic
    # each lane's other fields as alone
    one = h.mannwhitneyu([1.0, 2, 3, 4], [2.0, 2, 5])
    assert r.pvalue[1] == one.pvalue and z[1] == one.zstatistic


def test_all_values_equal_is_nan():
    """H157 (the §4 rule; scipy gives p = 0.2207 for ansari's normal approximation, 1 for its
    exact one, and 0, 1 for ranksums): every value of the row equal gives NaN."""
    for meth in ("auto", "exact", "asymptotic"):
        r = h.ansari([1.0, 1, 1], [1.0, 1], method=meth)
        assert r.statistic == 9.0 and math.isnan(r.pvalue)
    assert math.isnan(h.ansari([2.0] * 4, [2.0] * 3).pvalue)
    r = h.ranksums([1.0] * 3, [1.0] * 2)
    assert math.isnan(r.statistic) and math.isnan(r.pvalue)
    assert h.ranksums([1.0, 2, 3], [4.0, 5]).statistic == pytest.approx(-1.73205, rel=1e-5)
    b = h.ansari(sd.asarray([[1.0, 1, 1], [1.0, 2, 3]]), sd.asarray([[1.0, 1], [1.0, 5]]), axis=1)
    assert math.isnan(b.pvalue[0]) and b.pvalue[1] == h.ansari([1.0, 2, 3], [1.0, 5]).pvalue
