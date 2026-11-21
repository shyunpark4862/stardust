"""summary() of the sd.htest results (H74-H76): the test's name as the title, the header of its
numbers and of the call (alternative, method, sample sizes), the interval as a table (one-sided
ends included), the array fields of one test, a batch one row per lane (cut when long), and the
one-line reprs."""

import numpy as np
import pytest

import stardust as sd

h = sd.htest


def _r(seed=3):
    return np.random.default_rng(seed)


def _row(text, stub):
    for line in text.splitlines():
        if line.startswith(stub + " "):
            return line[len(stub) :].split()
    raise AssertionError(f"no row {stub!r} in\n{text}")


def test_titles_name_the_test_and_its_variant():
    r = _r()
    a, b = r.normal(size=20), r.normal(0.4, size=18)
    for res, title in (
        (h.ttest_1samp(a, 0.0), "One-sample t-test"),
        (h.ttest_rel(a, a + r.normal(size=20)), "Paired t-test"),
        (h.ttest_ind(a, b), "Two-sample t-test (Student)"),
        (h.ttest_ind(a, b, equal_var=False), "Two-sample t-test (Welch)"),
        (h.ttest_ind(a, b, trim=0.2, equal_var=False), "Two-sample trimmed t-test (Yuen, Welch)"),
        (h.f_oneway(a, b, equal_var=False), "One-way ANOVA (Welch)"),
        (h.kendalltau(a[:18], b, variant="c"), "Kendall rank correlation test (tau-c)"),
        (h.shapiro(a), "Shapiro-Wilk normality test"),
        (h.mannwhitneyu(a, b), "Mann-Whitney U test"),
        (h.pearsonr(a[:18], b), "Pearson correlation test"),
    ):
        assert res.summary().splitlines()[0].strip() == title


def test_header_numbers_and_call():
    r = _r()
    a, b = r.normal(size=20), r.normal(0.4, size=18)
    t = h.ttest_ind(a, b, equal_var=False, alternative="less")
    s = t.summary()
    assert f"Statistic:{t.statistic:#.4g}".replace(" ", "") in s.replace(" ", "")
    assert f"{t.pvalue:#.3g}" in s and f"{t.df:#.4g}" in s
    assert "Alternative:                      less" in s and "No. Observations:               20, 18" in s
    k = h.kruskal(a, b, a[:5], b[:6], a[:7]).summary()
    assert "No. Observations:                   56" in k and "No. Samples:                         5" in k
    assert "Df:                                  4" in k
    m = h.mannwhitneyu(a, b, method="asymptotic").summary()
    assert "Method:                     asymptotic" in m and "Z-statistic:" in m
    lv = h.levene(a, b, center="trimmed", proportiontocut=0.1).summary()
    assert "Center:                        trimmed" in lv and "Proportion Cut:" in lv
    c = h.chisquare([16.0, 18, 16, 14, 12, 12]).summary()
    assert "No. Categories:                      6" in c and "No. Observations" not in c


def test_sizes_only_where_known():
    x = np.array([0.4, np.nan, -1.2, 0.9, 1.7, -0.3])
    assert "No. Observations" not in h.ttest_1samp(x, 0.0, nan_policy="omit").summary()
    assert "No. Observations:                    6" in h.ttest_1samp(x, 0.0).summary()


def test_interval_table_and_alpha():
    r = _r()
    a, b = r.normal(size=20), r.normal(0.4, size=18)
    t = h.ttest_ind(a, b)
    lo, hi = t.conf_int(alpha=0.1).tolist()
    s = t.summary(alpha=0.1)
    assert "[0.050     0.950]" in s
    assert [float(v) for v in _row(s, "mean difference")] == pytest.approx([lo, hi], abs=5e-4)
    g = h.ttest_ind(a, b, alternative="greater").summary()
    assert "[0.050     1.000]" in g and _row(g, "mean difference")[1] == "inf"
    p = h.pearsonr(a[:18], b, alternative="less").summary()
    assert "[0.000     0.950]" in p and _row(p, "correlation")[0] == "-1.000"
    with pytest.raises(ValueError, match=r"^summary\(\): 'alpha' must be between 0 and 1 \(exclusive\)$"):
        t.summary(alpha=1.5)
    with pytest.raises(TypeError, match=r"^summary\(\): unexpected keyword argument 'alpha'$"):
        h.shapiro(a).summary(alpha=0.1)


def test_array_fields_of_one_test_are_tables():
    r = _r()
    s = h.median_test(r.normal(size=12), r.normal(size=14), r.normal(size=9)).summary()
    assert "table             [0]        [1]        [2]" in s and "Median:" in s
    big = h.somersd(np.arange(400.0) % 31, np.arange(400.0) % 37).summary()  # a 31 x 37 table, cut
    head = next(line for line in big.splitlines() if line.startswith("table"))
    assert head.split()[1:] == ["[0]", "[1]", "[2]", "[3]", "[4]", "...", "[32]", "[33]", "[34]", "[35]", "[36]"]
    assert sum(1 for line in big.splitlines() if line.startswith("...")) == 1


def test_batch_one_row_per_lane():
    r = _r()
    A, B = r.normal(size=(4, 20)), r.normal(size=(4, 22))
    t = h.ttest_ind(A, B, axis=-1, equal_var=False)  # Welch: df varies per lane, a column
    s = t.summary()
    assert "Batch shape:                      (4,)" in s
    assert "statistic     pvalue         df     [0.025     0.975]" in s
    # the pooled df every lane shares is in the header (regression G195, API.md §2.3)
    pooled = h.ttest_ind(A, B, axis=-1).summary()
    assert "Df:                                 40" in pooled and "pvalue     [0.025     0.975]" in pooled
    assert "         df " not in pooled
    ci = t.conf_int()
    for i in range(4):
        cells = _row(s, f"[{i}]")
        assert float(cells[0]) == pytest.approx(float(t.statistic[i]), abs=5e-4)
        assert [float(v) for v in cells[3:]] == pytest.approx(ci[i].tolist(), abs=5e-4)
    long = h.ttest_1samp(r.normal(size=(3, 9, 6)), 0.0, axis=-1).summary()  # 27 lanes: cut
    assert "[0, 0]" in long and "[2, 8]" in long and "[1, 4]" not in long
    assert sum(1 for line in long.splitlines() if line.startswith("...")) == 1


def test_batch_methods_per_lane():
    x = np.vstack([np.arange(8.0), np.arange(80.0)[:8] * 0.5])
    y = np.vstack([np.arange(8.0) + 0.5, np.r_[np.arange(7.0), 3.0]])  # the second lane has ties
    m = h.mannwhitneyu(x, y, axis=-1)
    assert m.method == ["exact", "asymptotic"]
    s = m.summary()  # the lanes' methods differ: a column, and the exact lane's zstatistic is NaN
    assert s.splitlines()[s.splitlines().index("-" * 78) - 1].split() == ["statistic", "pvalue", "zstatistic", "method"]
    assert _row(s, "[0]")[2:] == ["nan", "exact"] and _row(s, "[1]")[3] == "asymptotic"
    ks = h.ks_2samp(_r().normal(size=(2, 15)), _r(4).normal(size=(2, 15)), axis=-1).summary()
    assert "Method:                          exact" in ks and "statistic_location" in ks


def test_keepdims_reads_as_one_test():
    a = _r().normal(size=15)
    s = h.ttest_1samp(a, 0.0, keepdims=True).summary()
    assert s == h.ttest_1samp(a, 0.0).summary()


def test_reprs_show_short_arrays_and_shapes():
    r = _r()
    assert repr(h.ttest_1samp(r.normal(size=10), 0.0, keepdims=True)).startswith("Ttest1sampResult(statistic=[")
    t = repr(h.median_test(r.normal(size=12), r.normal(size=14)))
    assert "table=<array of shape (2, 2)>" in t and "\n" not in t
    assert (
        repr(h.ttest_ind(r.normal(size=(3, 9)), r.normal(size=(3, 8)), axis=-1)) == "TtestIndResult(batch shape (3,))"
    )


def test_spearmanr_matrix_lanes():
    s = h.spearmanr(_r().normal(size=(15, 3)), axis=0).summary()
    assert "Batch shape:                    (3, 3)" in s and _row(s, "[1, 1]")[0] == "1.000"


def test_contingency_tables():
    """H6's results: the expected counts of one table as a table, the table's shape and Yates'
    correction in the header, the variants' titles, the odds ratio's and the relative risk's
    intervals as one row each."""
    t = [[3.0, 7], [9, 2]]
    s = h.chi2_contingency(t).summary()
    assert s.splitlines()[0].strip() == "Chi-squared test of independence"
    assert "Table:                           2 x 2" in s and "Correction:                      Yates" in s
    assert "expected_freq         [0]        [1]" in s and "Df:                                  1" in s
    assert "Correction" not in h.chi2_contingency([[3.0, 7, 1], [9, 2, 4]]).summary()
    for res, title in (
        (h.fisher_exact(t), "Fisher's exact test"),
        (h.barnard_exact(t), "Barnard's exact test (pooled)"),
        (h.barnard_exact(t, pooled=False), "Barnard's exact test (unpooled)"),
        (h.boschloo_exact(t), "Boschloo's exact test"),
        (h.mcnemar(t), "McNemar's test (exact)"),
        (h.mcnemar(t, exact=False), "McNemar's test (chi-squared)"),
        (h.cochrans_q(np.eye(3)), "Cochran's Q test"),
        (h.odds_ratio(t), "Odds ratio (conditional maximum likelihood)"),
        (h.odds_ratio(t, kind="sample"), "Odds ratio (sample)"),
        (h.relative_risk(9, 20, 4, 21), "Relative risk"),
    ):
        assert res.summary().splitlines()[0].strip() == title
    f = h.fisher_exact([[3.0, 1, 4], [1, 5, 9], [2, 6, 5]]).summary()
    assert "Table:                           3 x 3" in f and "Alternative" not in f
    o = h.odds_ratio(t)
    lo, hi = o.conf_int(alpha=0.1).tolist()
    assert [float(v) for v in _row(o.summary(alpha=0.1), "odds ratio")] == pytest.approx([lo, hi], abs=5e-4)
    rr = h.relative_risk(9, 20, 4, 21).summary()
    assert "Exposed cases:                       9" in rr and _row(rr, "relative risk")
    b = h.chi2_contingency(np.ones((3, 2, 2)) + np.eye(2)).summary()  # a batch: the tables are left out
    assert "Batch shape:                      (3,)" in b and "expected_freq" not in b
    q = h.cochrans_q(np.eye(3)).summary()
    assert "No. Observations:                    3" in q and "No. Treatments:                      3" in q
