"""The reference and comparison data of sd.htest (docs/design/htest.md §7): every file belongs to
the problem collection as it is (digests), the reference tool's exact distributions pass checks
of their own, and the recorded scipy, statsmodels and R results agree with the references except
where they are known not to (the lists below, measured when the data was made: the comparison
highlights of H0). The recorded files are compared, not the installed libraries, so the tests
do not change with them."""

import importlib.util
import itertools
import json
import math
import pathlib
import sys
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest

from . import problems as P

ROOT = pathlib.Path(__file__).parents[2]
REF = ROOT / "tests" / "htest" / "ref"
DATA = ROOT / "tests" / "htest" / "data"


def load(path):
    d = json.loads(path.read_text())
    meta = d.pop("_provenance")
    return meta, d


def tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("htest_ref", ROOT / "tools" / "htest_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REFS = {p.stem: load(p)[1] for p in sorted(REF.glob("*.json"))}
SCIPY = {}
for _p in sorted(DATA.glob("scipy_h*.json")):
    SCIPY.update(load(_p)[1])
STATSMODELS = load(DATA / "statsmodels.json")[1]
R = load(DATA / "r.json")[1]


def rel(v, ref):
    """|v - ref| / |ref| (NaN against NaN is 0; anything else against NaN is inf)."""
    if isinstance(v, list):
        v = v[0]
    if v is None and ref == ref:  # (R's NA against a number; against NaN it stays a difference)
        v = math.nan
    if ref != ref:
        return 0.0 if v != v else math.inf
    if v != v:
        return math.inf
    if math.isinf(ref):
        return 0.0 if v == ref else math.inf
    if ref == 0:
        return 0.0 if v == 0 else abs(v)
    return abs(v - ref) / abs(ref)


# ---- the files belong to the collection ---------------------------------------------------


def test_data_cover_the_collection_with_its_digests():
    assert set(SCIPY) | set(STATSMODELS) == set(P.PROBLEMS)
    for pid, rec in list(SCIPY.items()) + list(STATSMODELS.items()) + list(R.items()):
        assert rec["digest"] == P.digest(P.PROBLEMS[pid]), pid
    for fam, recs in REFS.items():
        for pid, rec in recs.items():
            assert rec["digest"] == P.digest(P.PROBLEMS[pid]), (fam, pid)


def test_versions_and_provenance():
    meta = load(DATA / "scipy_h1.json")[0]
    assert meta["scipy"] == "1.18.1" and meta["statsmodels"] == "0.15.0"
    assert meta["tool"] == "tools/gen_htest_scipy.py"
    assert load(DATA / "r.json")[0]["R"].startswith("R version 4.6.2")
    for path in sorted(REF.glob("*.json")):
        prov = load(path)[0]
        assert prov["arb"].startswith("python-flint ") and prov["target_bits"] == 110, path
        assert prov["tool"] == "tools/htest_ref.py" and prov["dropped"] == [], path
        assert prov["n_requested"] == len(load(path)[1]), path


def test_reference_fields_are_double_doubles():
    for recs in REFS.values():
        for pid, rec in recs.items():
            for k in ("statistic", "pvalue"):
                if k in rec:
                    hi, lo = rec[k]
                    assert isinstance(hi, float) and isinstance(lo, float), (pid, k)
                    assert hi != hi or not math.isfinite(hi) or abs(lo) <= abs(hi) * 2.0**-52, (pid, k)


def test_every_problem_kind_is_present():
    tags = {t for p in P.PROBLEMS.values() for t in p.tags}
    assert {"small", "large", "ties", "tail", "offset", "degenerate", "batch", "exact"} <= tags
    assert {p.milestone for p in P.PROBLEMS.values()} == {f"H{i}" for i in range(1, 9)}


# ---- the tool's exact distributions -------------------------------------------------------


def test_exact_counts_sum_to_their_totals():
    T = tool()
    for m, n in ((1, 1), (3, 4), (5, 7), (8, 8)):
        c = T.mwu_counts(m, n)
        assert sum(c) == math.comb(m + n, m) and c == c[::-1]
    for n in (1, 5, 13, 20):
        c = T.signed_rank_counts(n)
        assert sum(c) == 2**n and c == c[::-1]
    for n in (2, 5, 9):
        top = n * (n - 1) // 2
        c = T.mahonian(n, top)
        assert sum(c) == math.factorial(n) and c == c[::-1]


def test_rank_exact_distributions():
    """H3's counts (tools/htest_ref.py): the conditional subset sums total C(N, c) and without ties
    are the Gaussian binomial's; the signed sums total 2^n and are symmetric; a Page row's orders
    total n!; Friedman's exact null for k = 3, n = 2 against its enumeration by hand."""
    T = tool()
    for N, c in ((7, 3), (10, 4)):
        d = T.subset_sum_counts([2 * r for r in range(1, N + 1)], c)
        assert sum(d.values()) == math.comb(N, c)
        u = T.mwu_counts(c, N - c)
        assert [d.get(2 * (v + c * (c + 1) // 2), 0) for v in range(len(u))] == u
    scores = [3, 3, 6, 8, 8, 8, 12]  # doubled midranks of 1, 1, 2, 3, 3, 3, 4
    assert sum(T.subset_sum_counts(scores, 3).values()) == math.comb(7, 3)
    sc = T.signed_counts(scores)
    tot = sum(scores)
    assert sum(sc.values()) == 2**7 and all(sc[v] == sc[tot - v] for v in sc)
    row = T.page_row_counts((2, 4, 4, 8))
    assert sum(row.values()) == math.factorial(4)
    # k = 3, n = 2: the 36 pairs of orders; chi^2_F = (sum R_j^2) / 2 - 24
    cnt = Counter()
    for a in itertools.permutations((1, 2, 3)):
        for b in itertools.permutations((1, 2, 3)):
            cnt[Fraction(sum((x + y) ** 2 for x, y in zip(a, b)), 2) - 24] += 1
    for q in sorted(cnt):
        assert T.friedman_exact_sf(3, 2, q) == Fraction(sum(v for k, v in cnt.items() if k >= q), 36)


def test_two_sample_ks_paths_against_the_closed_form():
    # equal sizes: P(D >= h/n) = 2 sum_k (-1)^(k-1) C(2n, n - k h) / C(2n, n) (Gnedenko-Korolyuk)
    T = tool()
    for n in (5, 12, 30):
        for h in range(1, n + 1):
            closed = sum((-1) ** (k - 1) * 2 * math.comb(2 * n, n - k * h) for k in range(1, n // h + 1))
            assert T.ks2_paths(n, n, 1, 1, h, "two-sided") == closed, (n, h)
            assert T.ks2_paths(n, n, 1, 1, h, "greater") == math.comb(2 * n, n - h), (n, h)


def _labelings(n1, n2):
    """Every assignment of n1 x's among n1 + n2 positions (the label arrangements, equally likely)."""
    return itertools.combinations(range(n1 + n2), n1)


def test_two_sample_tail_counts_against_enumeration():
    # the block counts (ties conditional on the observed blocks, H55) against every arrangement of
    # the labels over the pooled sorted values: the KS gap's tail and the Cramér-von Mises 4U's
    T = tool()
    cases = [
        ([1, 2, 2, 3, 3, 3], [2, 3, 4, 4, 5]),
        ([0, 0, 1, 1, 2], [1, 1, 2, 3]),
        ([0.5, 1.5, 2.5, 3.5], [1.0, 2.0, 3.0, 4.0, 5.0]),
        ([1, 1, 1, 2], [1, 2, 2, 2, 3, 3]),
    ]
    for xs, ys in cases:
        x, y = [Fraction(v) for v in xs], [Fraction(v) for v in ys]
        n1, n2 = len(x), len(y)
        pooled = sorted(x + y)
        blocks = [cx + cy for _, cx, cy in T.pooled_blocks(x, y)]
        u4_obs, cblocks = T.cvm2_u4(x, y)
        h_obs = {alt: T._ks2_stat(x, y, alt)[0] for alt in ("two-sided", "greater", "less")}
        ks_count = Counter()
        cvm_count = 0
        for pos in _labelings(n1, n2):
            xx = [pooled[i] for i in pos]
            yy = [pooled[i] for i in range(n1 + n2) if i not in pos]
            for alt, h in h_obs.items():
                ks_count[alt] += T._ks2_stat(xx, yy, alt)[0] >= h
            cvm_count += T.cvm2_u4(xx, yy)[0] >= u4_obs
        for alt, h in h_obs.items():
            assert T.ks2_tail_count(n1, n2, blocks, h, alt) == ks_count[alt], (xs, ys, alt)
        assert T.cvm2_tail_count(n1, n2, cblocks, u4_obs) == cvm_count, (xs, ys)
    # without ties the block count is the lattice-path count
    for n1, n2 in ((7, 5), (9, 12), (10, 10)):
        L = math.lcm(n1, n2)
        for h in range(1, L + 1, max(1, L // 9)):
            for alt in ("two-sided", "greater", "less"):
                got = T.ks2_tail_count(n1, n2, [1] * (n1 + n2), h, alt)
                if alt == "less":  # the reflection of 'greater' (the samples' roles swapped)
                    assert got == T.ks2_paths(n2, n1, L // n2, L // n1, h, "greater"), (n1, n2, h)
                else:
                    assert got == T.ks2_paths(n1, n2, L // n1, L // n2, h, alt), (n1, n2, h, alt)


def test_two_sample_cvm_against_scipys_exact_count():
    # without ties the count is scipy's _pval_cvm_2samp_exact (Xiao, Gordon and Yakovlev's
    # frequency table), at every attained value of U for small sizes
    from scipy.stats import _hypotests as H

    T = tool()
    for nx, ny in ((3, 4), (5, 5), (6, 4)):
        N = nx + ny
        seen = set()
        for pos in _labelings(nx, ny):
            xx = [Fraction(i) for i in pos]
            yy = [Fraction(i) for i in range(N) if i not in pos]
            u4 = T.cvm2_u4(xx, yy)[0]
            if u4 in seen:
                continue
            seen.add(u4)
            mine = Fraction(T.cvm2_tail_count(nx, ny, [(1, 0)] * N, u4), math.comb(N, nx))
            assert float(mine) == pytest.approx(float(H._pval_cvm_2samp_exact(u4 // 4, nx, ny)), rel=1e-15), (
                nx,
                ny,
                u4,
            )


def test_anderson_ksamp_variants_without_ties():
    # without ties Scholz and Stephens' 'right' (eq. 6) and 'continuous' (eq. 3) are the same A2kN
    T = tool()
    s = [[Fraction(v) for v in P.normal(7101, 9)], [Fraction(v) for v in P.normal(7102, 11, 0.4)]]
    assert T.ad_ksamp_a2kn(s, "right")[0] == T.ad_ksamp_a2kn(s, "continuous")[0]
    assert T.ad_ksamp_a2kn(s, "midrank")[0] != T.ad_ksamp_a2kn(s, "right")[0]


def test_one_sample_ks_small_n():
    from flint import arb

    T = tool()
    # n = 1: D = max(F, 1 - F), P(D < d) = 2d - 1; P(D+ >= d) = 1 - d
    for d in (0.6, 0.75, 0.9):
        assert abs(float(T.kolmogorov_cdf_exact(1, arb(d))) - (2 * d - 1)) < 1e-15
        assert abs(float(T.smirnov_sf(1, arb(d))) - (1 - d)) < 1e-15
    # n = 2, 3/4 <= d <= 1: P(D < d) = 1 - 2 (1 - d)^2
    for d in (0.75, 0.8, 0.95):
        assert abs(float(T.kolmogorov_cdf_exact(2, arb(d))) - (1 - 2 * (1 - d) ** 2)) < 1e-15


def test_exact_binomial_and_fisher_rules():
    T = tool()
    # a symmetric binomial: the two tails of k = 3 of 10 are 2 P(X <= 3) = 2 (176/1024)
    assert T.binomtest(P.PROBLEMS["binomtest/3-10"])["pvalue"] == Fraction(352, 1024)
    # the balanced table is the mode: p = 1
    assert T.fisher_exact(P.PROBLEMS["fisher_exact/balanced"])["pvalue"] == 1


# ---- the recorded libraries against the references ----------------------------------------

# scipy's results farther than 1e-12 from the references (relative), measured on these problems:
# the statistics of data with a large offset (the mean rounded before the subtraction), Bartlett's
# statistic when the variances are nearly equal (two large logarithms cancel), the G statistic
# when O ~ E, Pearson's p-value near |r| = 1 (1 - r^2 from a rounded r), and the inf and 0 of a
# zero standard error (NaN here, H11).
SCIPY_KNOWN = {
    ("bartlett/near-equal", "statistic"): 52,
    ("power_divergence/g-near-equal", "statistic"): 9.9e-8,
    ("pearsonr/near-1-eps4", "pvalue"): 5.4e-7,
    ("pearsonr/near-1-eps6", "pvalue"): 8.3e-3,
    ("pearsonr/near-1-eps7", "pvalue"): 0.11,
    ("ttest_1samp/offset", "statistic"): 1.7e-8,
    ("ttest_1samp/offset", "pvalue"): 1.0e-8,
    ("ttest_ind/offset", "statistic"): 3.4e-8,
    ("ttest_ind/offset", "pvalue"): 3.2e-7,
    ("ttest_1samp/constant", "statistic"): math.inf,
    ("ttest_1samp/constant", "pvalue"): math.inf,
    # H7: the E-test's sums cut at [ppf(1e-10), ppf(1 - 1e-16)] (the observed counts of
    # past-truncation beyond the range of X2), and its T of the grid and of the observed counts are
    # different float expressions (an outcome as extreme as the observed one can drop out)
    ("poisson_means_test/large", "pvalue"): 4.9e-9,
    ("poisson_means_test/tail", "pvalue"): 2.1e-3,
    ("poisson_means_test/deep-tail", "pvalue"): 0.41,
    ("poisson_means_test/past-truncation", "pvalue"): 1.0,
    # H1's problems: the same offsets (Welch's t and ANOVA, Alexander-Govern), a - b rounded before
    # the paired test, the inf and 0 of zero variances (NaN here, user decision 3 and H11), and
    # the centres of levene and fligner rounded to a double: |x - c| of the two middle values of
    # an even group no longer tie, which moves two ranks of fligner by 1/2
    ("ttest_rel/offset-wide", "statistic"): 1.1e-9,
    ("ttest_rel/offset-wide", "pvalue"): 2.2e-9,
    ("ttest_ind/constant-both", "statistic"): math.inf,
    ("ttest_ind/constant-both", "pvalue"): math.inf,
    ("ttest_ind/offset-welch", "statistic"): 2.0e-8,
    ("ttest_ind/offset-welch", "pvalue"): 3.4e-8,
    ("ttest_ind_from_stats/zero-std", "statistic"): math.inf,
    ("ttest_ind_from_stats/zero-std", "pvalue"): math.inf,
    ("f_oneway/constant-groups", "statistic"): math.inf,
    ("f_oneway/constant-groups", "pvalue"): math.inf,
    ("f_oneway/welch-offset", "statistic"): 5.6e-8,
    ("f_oneway/welch-offset", "pvalue"): 1.1e-7,
    ("alexandergovern/offset", "statistic"): 2.7e-8,
    ("alexandergovern/offset", "pvalue"): 8.4e-8,
    ("levene/offset", "statistic"): 1.5e-11,
    ("levene/offset", "pvalue"): 7.4e-11,
    ("fligner/large", "statistic"): 8.5e-11,
    ("fligner/large", "pvalue"): 6.4e-9,
    ("fligner/tail", "statistic"): 2.2e-9,
    ("fligner/tail", "pvalue"): 2.9e-7,
    # H2's problems: the Cressie-Read statistics cancelling to the first order when O ~ E (the G
    # test's above, lambda near 0 and -1, and two O ~ E lambdas), the NaN of a zero count (0 inf;
    # the limit here), scipy's y = 0 replaced by 1 in skewtest, the D'Agostino moments of offset
    # data, 1 - W formed from a rounded W (shapiro, large n), the logistic fit's fsolve xtol 1e-5
    # (anderson), the two-sided Kolmogorov-Smirnov tail of Pelz-Good above n = 140 and the
    # Cramér-von Mises series cut at 1e-7
    ("power_divergence/zero-freeman-tukey", "statistic"): math.inf,
    ("power_divergence/zero-freeman-tukey", "pvalue"): math.inf,
    ("power_divergence/lambda-small", "statistic"): 2.4e-8,
    ("power_divergence/lambda-small", "pvalue"): 7.8e-9,
    ("power_divergence/lambda-near-minus-1", "statistic"): 7.0e-12,
    ("power_divergence/lambda-near-minus-1", "pvalue"): 2.3e-12,
    ("power_divergence/cr-near-equal", "statistic"): 66,
    ("power_divergence/cr-near-equal", "pvalue"): math.inf,
    ("power_divergence/ft-near-equal", "statistic"): 110,
    ("power_divergence/ft-near-equal", "pvalue"): math.inf,
    ("skewtest/symmetric", "statistic"): 1.0,
    ("skewtest/symmetric", "pvalue"): 0.69,
    ("skewtest/offset", "statistic"): 7.7e-10,
    ("skewtest/offset", "pvalue"): 1.6e-8,
    ("kurtosistest/offset", "statistic"): 2.8e-8,
    ("kurtosistest/offset", "pvalue"): 1.7e-8,
    ("shapiro/n5000", "pvalue"): 1.8e-12,
    ("shapiro/n20000", "pvalue"): 1.6e-12,
    ("anderson/logistic", "statistic"): 2.2e-11,
    ("anderson/logistic-normal", "statistic"): 7.8e-9,
    ("anderson/logistic-normal", "pvalue"): 3.9e-8,
    ("ks_1samp/n141", "pvalue"): 3.5e-6,
    ("ks_1samp/n400", "pvalue"): 1.7e-7,
    ("ks_1samp/expon", "pvalue"): 1.9e-6,
    ("cramervonmises/tail", "pvalue"): 0.016,
    # H3's problems: Ansari-Bradley's exact counts from gscale (AS 93) off by 6.5e-9-2.6e-7 at
    # N = 45-108; the exact signed-rank tail as 1 - cdf (0 where the truth is 1.7e-18); scipy's
    # 'exact' with ties or zeros, the distribution without them (H45: here the conditional one);
    # page_trend_test's 'auto' exact with ties (the same); the inf and the 0, 1 of a zero variance
    # or of no nonzero difference (NaN here, H11)
    ("ansari/exact-54", "pvalue"): 2.6e-7,
    ("ansari/exact-odd", "pvalue"): 6.5e-9,
    ("ansari/exact-greater", "pvalue"): 6.5e-9,
    ("ansari/ties-exact", "pvalue"): 0.49,
    ("mannwhitneyu/ties-exact", "pvalue"): 0.041,
    ("mannwhitneyu/ties-exact-greater", "pvalue"): 0.041,
    ("wilcoxon/ties-exact", "pvalue"): 0.041,
    ("wilcoxon/zeros-exact-pratt", "pvalue"): 1.8,
    ("wilcoxon/zeros-exact-zsplit", "pvalue"): 0.029,
    ("wilcoxon/exact-deep", "pvalue"): 1.0,
    ("page_trend_test/ties", "pvalue"): 3.5,
    ("brunnermunzel/separated", "statistic"): math.inf,
    ("wilcoxon/constant-zeros", "statistic"): math.inf,
    ("wilcoxon/constant-zeros", "pvalue"): math.inf,
    # H4's problems: the two-sample exact distributions with ties, scipy's without them (H55: here
    # conditional on the ties, as R's psmirnov; the Cramér-von Mises one the same, §9 of the design);
    # the asymptotic two-sided Kolmogorov-Smirnov p-value through kstwo at round(n1 n2 / N) > 140
    # (Pelz-Good); the asymptotic Cramér-von Mises tail from the series cut at 1e-7; the
    # Epps-Singleton characteristic functions of offset data and of t = 0.2 (cos of a rounded
    # argument, its covariance's inverse)
    ("ks_2samp/ties", "pvalue"): 1.4,
    ("ks_2samp/ties-small", "pvalue"): 1.2,
    ("ks_2samp/ties-small-greater", "pvalue"): 0.28,
    ("ks_2samp/ties-small-less", "pvalue"): 0.28,
    ("ks_2samp/ties-greater", "pvalue"): 0.35,
    ("ks_2samp/ties-few-values", "pvalue"): 8.5,
    ("ks_2samp/ties-large", "pvalue"): 0.22,
    ("ks_2samp/asymp", "pvalue"): 4.8e-8,
    ("ks_2samp/asymp-auto", "pvalue"): 3e-7,
    ("cramervonmises_2samp/ties-exact", "pvalue"): 0.012,
    ("cramervonmises_2samp/ties-auto", "pvalue"): 0.074,
    ("cramervonmises_2samp/tail", "pvalue"): 6.1e13,
    ("epps_singleton_2samp/three-t", "statistic"): 3.8e-12,
    ("epps_singleton_2samp/three-t", "pvalue"): 1.4e-11,
    ("epps_singleton_2samp/tail", "pvalue"): 4.2e-12,
    ("epps_singleton_2samp/offset", "statistic"): 9.6e-10,
    ("epps_singleton_2samp/offset", "pvalue"): 4.5e-9,
    # H5's problems: Pearson's 1 - r^2 from a rounded r (0 where the truth is 1.2e-74 at
    # r = 1 - 1e-18; the near -1 r and the deep tail of n = 400 too), and the normal tail of
    # chatterjeexi's xi / sd (its variance's sums in doubles)
    ("pearsonr/near-1-eps9", "pvalue"): 1.0,
    ("pearsonr/near-minus-1", "pvalue"): 3.3e-3,
    ("pearsonr/deep-tail", "pvalue"): 1.2e-12,
    ("chatterjeexi/large", "pvalue"): 1.6e-12,
    # H6's problems: the Cressie-Read terms of a zero count (scipy's NaN where the limits are 0 and
    # +inf), the G statistic when O ~ E, and Barnard's region from rounded Wald statistics: the
    # mirror-image tables (3, 8) and (8, 3) of [[7, 12], [8, 3]] have |T| = |T_obs| exactly but one
    # ulp below it in doubles, so scipy leaves them out (here exact T^2 comparisons)
    ("chi2_contingency/freeman-tukey-zero", "statistic"): math.inf,
    ("chi2_contingency/freeman-tukey-zero", "pvalue"): math.inf,
    ("chi2_contingency/neyman-zero", "statistic"): math.inf,
    ("chi2_contingency/neyman-zero", "pvalue"): math.inf,
    ("chi2_contingency/near-expected-g", "statistic"): 1.6e-11,
    ("barnard_exact/mirror", "pvalue"): 9.5e-4,
    ("barnard_exact/mirror-unpooled", "pvalue"): 9.5e-4,
    ("barnard_exact/unpooled-less", "pvalue"): 9.5e-4,
}
# R's: the same cancellations (and ttest_rel's x - y), its own methods where they differ from scipy's
# (ks.test is exact only below n = 100 (one sample), wilcox.test is exact up to n = 50 and conditional with
# ties, cor.test's Spearman p-value is AS 89's exact distribution), and two p-values of 0 in
# the deep tail (true 1.7e-21 and 6.1e-51).
R_KNOWN = {
    ("bartlett/near-equal", "statistic"): 54,
    ("f_oneway/offset", "statistic"): 1.6e-8,
    ("f_oneway/offset", "pvalue"): 4.3e-8,
    ("ttest_1samp/offset", "statistic"): 1.8e-8,
    ("ttest_1samp/offset", "pvalue"): 1.1e-8,
    ("ttest_rel/offset", "statistic"): 1.2e-6,
    ("ttest_rel/offset", "pvalue"): 1.2e-4,
    ("ttest_ind/offset", "statistic"): 1.1e-8,
    ("ttest_ind/offset", "pvalue"): 1.0e-7,
    ("pearsonr/near-1-eps4", "pvalue"): 2.4e-7,
    ("pearsonr/near-1-eps6", "pvalue"): 1.4e-3,
    ("pearsonr/near-1-eps7", "pvalue"): 0.088,
    ("pearsonr/offset", "pvalue"): 9.3e-8,
    ("ks_1samp/n100", "pvalue"): 0.06,
    ("ks_1samp/n1000", "pvalue"): 0.056,
    ("ks_1samp/tail", "pvalue"): 1.0,
    ("ks_1samp/asymp", "pvalue"): 9.4e-8,
    ("ks_1samp/approx", "pvalue"): 0.018,
    ("mannwhitneyu/ties", "pvalue"): 7.1e-3,
    ("mannwhitneyu/no-continuity", "pvalue"): 2.4e-3,
    ("wilcoxon/zeros-wilcox", "pvalue"): 0.16,
    ("wilcoxon/tail", "pvalue"): 1.0,
    # H1's problems (as scipy's above; fligner.test's centre is rounded as scipy's)
    ("ttest_rel/offset-wide", "statistic"): 1.1e-9,
    ("ttest_rel/offset-wide", "pvalue"): 2.2e-9,
    ("ttest_ind/offset-welch", "statistic"): 1.9e-8,
    ("ttest_ind/offset-welch", "pvalue"): 3.2e-8,
    ("f_oneway/constant-groups", "statistic"): math.inf,
    ("f_oneway/constant-groups", "pvalue"): math.inf,
    ("f_oneway/welch-offset", "statistic"): 7.9e-9,
    ("f_oneway/welch-offset", "pvalue"): 1.6e-8,
    ("bartlett/offset", "statistic"): 2.6e-9,
    ("bartlett/offset", "pvalue"): 3.2e-8,
    ("fligner/large", "statistic"): 8.5e-11,
    ("fligner/large", "pvalue"): 6.4e-9,
    ("fligner/tail", "statistic"): 2.2e-9,
    ("fligner/tail", "pvalue"): 2.9e-7,
    # car::leveneTest (H33): R's median of an even group rounded to a double, as scipy's
    ("levene/offset", "statistic"): 8.4e-9,
    ("levene/offset", "pvalue"): 4.2e-8,
    # H2's problems: ks.test's exact tail as 1 - cdf (0 where the truth is 1e-113, 1e-251, 1e-107)
    # and its asymptotic form above n = 100; shapiro.test's W of offset data and 1 - W
    ("shapiro/offset", "statistic"): 3.3e-10,
    ("shapiro/offset", "pvalue"): 2.8e-8,
    ("shapiro/n5000", "pvalue"): 4.4e-11,
    ("shapiro/n3-low", "pvalue"): 3.1e-11,
    ("ks_1samp/n141", "pvalue"): 0.043,
    ("ks_1samp/n400", "pvalue"): 0.043,
    ("ks_1samp/deep", "pvalue"): 1.1e97,
    ("ks_1samp/deep-400", "pvalue"): 4.9e43,
    ("ks_1samp/greater-deep", "pvalue"): 1.8e10,
    ("ks_1samp/approx-deep", "pvalue"): 1.1e97,
    ("ks_1samp/expon", "pvalue"): 0.037,
    # H3's problems: R's own switches (wilcox.test exact only for n < 50 and, with ties, its
    # conditional exact where scipy's 'auto' is normal; its zeros ranked as Pratt's; its rank-sum z
    # with the tie correction ranksums has not; ansari.test exact only below 50 and its own tie
    # variance), and the 1 or NA of every value tied (NaN here)
    ("ansari/ties", "pvalue"): 0.17,
    ("ansari/exact-54", "pvalue"): 0.015,
    ("ansari/ties-exact", "pvalue"): 0.0043,
    ("kruskal/constant", "statistic"): math.inf,
    ("kruskal/constant", "pvalue"): math.inf,
    ("mannwhitneyu/ties-small-auto", "pvalue"): 0.0087,
    ("mannwhitneyu/exact-8-300", "pvalue"): 0.02,
    ("mannwhitneyu/constant", "pvalue"): math.inf,
    ("ranksums/ties", "pvalue"): 0.0012,
    ("wilcoxon/zeros-auto-small", "pvalue"): 0.071,
    ("wilcoxon/constant-zeros", "pvalue"): math.inf,
    # H4's problems: ks.test runs exact = TRUE wherever sd's ks_2samp is exact (the conditional
    # distribution with ties, the same as sd's: 1e-15); its exact tail is 1 - cdf (0 where the
    # truth is 1e-133), and its asymptotic p-values are Kolmogorov's limit and exp(-2 n D^2), not
    # scipy's kstwo at round(n) and Hodges' formula
    ("ks_2samp/separated", "pvalue"): 1.0,
    ("ks_2samp/separated-greater", "pvalue"): 1.0,
    ("ks_2samp/asymp", "pvalue"): 0.02,
    ("ks_2samp/asymp-less", "pvalue"): 0.0031,
    ("ks_2samp/asymp-small", "pvalue"): 0.25,
    ("ks_2samp/asymp-auto", "pvalue"): 0.032,
    # H5's problems: cor.test's Pearson p-value from its t of a rounded r near |r| = 1 (and of
    # offset data, its r rounded in the centring); its exact Kendall p-value (ckendall's counts in
    # doubles: 2.1e-9 at n = 33) as 1 - cdf in the deep tail (6.8e-14 where the truth is 2.4e-82,
    # and -4.1e-14 at n = 100). Spearman is called with exact = FALSE (the t approximation, H63);
    # its default (AS 89's exact distribution) is recorded beside it (test_correlation.py).
    ("pearsonr/near-1-eps9", "pvalue"): 5.6e10,
    ("pearsonr/near-minus-1", "pvalue"): 5.5e-3,
    ("pearsonr/deep-tail", "pvalue"): 1.2e-12,
    ("pearsonr/offset-both", "pvalue"): 2.3e-7,
    ("kendalltau/exact-33", "pvalue"): 2.1e-9,
    ("kendalltau/exact-none-discordant", "pvalue"): 2.8e68,
    ("kendalltau/exact-one-discordant", "pvalue"): 1.9e142,
    ("kendalltau/exact-forced-200", "pvalue"): math.inf,  # NA (n = 200)
    ("pearsonr/scale-huge", "pvalue"): math.inf,  # NA: its squares of 1e200 overflow
    ("pearsonr/constant", "pvalue"): math.inf,  # NA, NaN here (as the others' constant samples)
    ("spearmanrho/constant", "pvalue"): math.inf,
    ("kendalltau/constant", "pvalue"): math.inf,
    # H6's problems: chisq.test of a one-row matrix is the goodness-of-fit test of its counts (scipy's
    # and sd's df = 0 test is 0 and 1); mcnemar.test does not correct b = c (statsmodels'
    # (|b - c| - 1)^2 / (b + c) still subtracts 1: 1/12 here)
    ("chi2_contingency/one-row", "statistic"): 0.5,
    ("chi2_contingency/zero-row", "statistic"): math.inf,  # NA (its 0/0), NaN here (H11)
    ("chi2_contingency/zero-row", "pvalue"): math.inf,
    ("chi2_contingency/one-row", "pvalue"): 0.22,
    ("mcnemar/equal-chi2", "pvalue"): 0.29,
    ("mcnemar/zero-off-chi2", "pvalue"): math.inf,  # NA (0/0), NaN here (H11)
    # H7: prop.test of a zero column (NA; NaN here, chi2_contingency's rule)
    ("proportions_chisquare/zero-column", "pvalue"): math.inf,
}
# the families whose R statistic is scipy's (R's W, V, S, T and z differ by definition elsewhere)
R_SAME_STATISTIC = {
    "ttest",
    "f_oneway",
    "bartlett",
    "fligner",
    "levene",
    "power_divergence",
    "ks_1samp",
    "ks_2samp",
    "mannwhitneyu",
    "shapiro",
    "kruskal",
    "friedmanchisquare",
    "ansari",
    "mood",
    "chi2_contingency",
}
R_FIELD = {"statistic": "statistic", "pvalue": "p.value"}
TOL = 1e-12


def _check(known, got_of, families=None):
    seen = set()
    for fam, recs in REFS.items():
        for pid, rec in recs.items():
            for k in ("statistic", "pvalue"):
                if k not in rec or (families is not None and k == "statistic" and fam not in families):
                    continue
                got = got_of(pid, k)
                if got is None:
                    continue
                e = rel(got, rec[k][0])
                if (pid, k) in known:
                    seen.add((pid, k))
                    want = known[(pid, k)]
                    assert e > TOL, (pid, k, e)
                    assert e == want if math.isinf(want) else 0.5 * want <= e <= 2 * want, (pid, k, e, want)
                else:
                    assert e <= TOL, (pid, k, e)
    assert seen == set(known)


def test_scipy_against_the_references():
    def got(pid, k):
        r = SCIPY.get(pid, {}).get("result", {})
        v = r.get(k)
        return None if v is None or isinstance(v, list) and len(v) != 1 else v

    _check(SCIPY_KNOWN, got)


def test_r_against_the_references():
    def got(pid, k):
        r = R.get(pid, {}).get("result")
        return None if not r or R_FIELD[k] not in r else r[R_FIELD[k]]

    _check(R_KNOWN, got, R_SAME_STATISTIC)


# The 95% intervals of the t tests (scipy's confidence_interval(), R's conf.int) farther than 1e-12
# from the references: the offsets again (R's paired test rounds a - b)
CI_KNOWN = {
    ("scipy", "ttest_ind/offset", 0): 9.8e-8,
    ("scipy", "ttest_ind/offset", 1): 2.1e-8,
    ("scipy", "ttest_ind/offset-welch", 0): 2.4e-8,
    ("scipy", "ttest_ind/offset-welch", 1): 7.1e-9,
    ("R", "ttest_rel/offset", 0): 3.9e-7,
    ("R", "ttest_rel/offset", 1): 3.8e-7,
    ("R", "ttest_ind/offset", 0): 3.3e-8,
    ("R", "ttest_ind/offset", 1): 7.9e-9,
    ("R", "ttest_ind/offset-welch", 0): 2.2e-8,
    ("R", "ttest_ind/offset-welch", 1): 7.7e-9,
}


def test_confidence_intervals_against_the_references():
    seen = set()
    for pid, rec in REFS["ttest"].items():
        ref = rec["conf_int"]
        libs = (
            ("scipy", SCIPY.get(pid, {}).get("result", {}).get("confidence_interval")),
            ("R", (R.get(pid, {}).get("result") or {}).get("conf.int")),
        )
        for lib, got in libs:
            if got is None:
                continue
            for j in (0, 1):
                want = ref[j][0]
                e = (0.0 if got[j] == want else math.inf) if math.isinf(want) else rel(got[j], want)
                key = (lib, pid, j)
                if key in CI_KNOWN:
                    seen.add(key)
                    assert 0.5 * CI_KNOWN[key] <= e <= 2 * CI_KNOWN[key], (key, e)
                else:
                    assert e <= TOL, (key, e)
    assert seen == set(CI_KNOWN)


def test_condition_numbers():
    # kappa (the p-value's condition number) is given wherever the p-value is a positive number
    for fam in ("ttest", "f_oneway", "alexandergovern", "bartlett", "levene", "fligner", "power_divergence",
                "jarque_bera", "dagostino", "shapiro", "anderson", "ks_1samp", "cramervonmises", "ranksums",
                "kruskal", "friedmanchisquare", "brunnermunzel", "mood", "median_test", "anderson_ksamp",
                "epps_singleton_2samp"):  # fmt: skip
        for pid, rec in REFS[fam].items():
            p, k = rec["pvalue"][0], rec["kappa"]
            # (a p-value below the doubles, 0 here, keeps its Arb value's condition number)
            assert (k == k and k >= 0) if p == p and p > 0 else (k != k or p == 0), (pid, p, k)
            if "kappa_stat" in rec:
                ks = rec["kappa_stat"]
                assert ks != ks or ks >= 0, (pid, ks)


# ---- the tool's H1 definitions against independent forms ----------------------------------


def test_h1_identities():
    T = tool()
    # F of two groups is Student's t^2, and the same p-value
    f, t = REFS["f_oneway"]["f_oneway/two-groups"], REFS["ttest"]["ttest_ind/small"]
    assert rel(f["statistic"][0], t["statistic"][0] ** 2) <= 4e-16 and rel(f["pvalue"][0], t["pvalue"][0]) <= 2e-16
    # Yuen with nothing trimmed is Student's (the pieces exactly)
    x = T.fvec(P.normal(7001, 9))
    v, m, h = T.trimmed(x, 0.0)
    assert (m, v, h) == (*T.mean_var(x), 9)
    # Welch's ANOVA of two groups is Welch's t^2 with Welch's degrees of freedom (exactly)
    a, b = P.normal(7002, 7), P.normal(7003, 11, 0.5, 2.0)
    fw = T.f_oneway(P.Problem("f_oneway/w2", "f_oneway", (a, b), {"equal_var": False}))
    tw = T.ttest(P.Problem("ttest_ind/w2", "ttest_ind", (a, b), {"equal_var": False}))
    assert fw["df_denom"] == tw["df"]
    (ma, va), (mb, vb) = T.mean_var(T.fvec(a)), T.mean_var(T.fvec(b))
    assert fw["statistic"] == (ma - mb) ** 2 / (va / 7 + vb / 11)
    # Levene's W of two groups is Student's t^2 on the deviations from the centres
    g = [P.normal(7004, 8), P.normal(7005, 10, 0, 2)]
    w = T.levene(P.Problem("levene/w2", "levene", tuple(g), {"center": "mean"}))["statistic"]
    z = [[abs(v - sum(T.fvec(x)) / len(x)) for v in T.fvec(x)] for x in g]
    (m1, v1), (m2, v2) = T.mean_var(z[0]), T.mean_var(z[1])
    sp2 = (7 * v1 + 9 * v2) / 16
    assert w == (m1 - m2) ** 2 / (sp2 * (Fraction(1, 8) + Fraction(1, 10)))
    # the even median: the exact average of the two middle values
    assert T.center([Fraction(1), Fraction(4), Fraction(2), Fraction(7)], "median", 0.05) == 3
    assert T.center([Fraction(v) for v in range(10)], "trimmed", 0.2) == Fraction(9, 2)


def test_fligner_scores_are_symmetric():
    from flint import arb

    T = tool()
    N = 11
    for r in (Fraction(1), Fraction(5, 2), Fraction(6)):
        # Phi^-1(1/2 + u) = -Phi^-1(1/2 - u), u = r/(2(N + 1)): the tool's scores mirror
        up = T.norm_isf((N + 1 - r) / (2 * (N + 1)))
        dn = T.norm_isf((N + 1 + r) / (2 * (N + 1)))
        assert (up + dn).contains(arb(0)) and up > 0


def test_multiple_testing_corrections():
    # statsmodels' multipletests, scipy's false_discovery_control and R's p.adjust against the exact
    # corrections
    for pid, rec in REFS["multipletests"].items():
        ref = [v[0] for v in rec["pvalues_corrected"]]
        if pid.startswith("multipletests/"):
            got = STATSMODELS[pid]["result"]["pvals_corrected"]
        else:
            got = SCIPY[pid]["result"]["value"]
        assert max(rel(a, b) for a, b in zip(got, ref)) <= 1e-15, pid
        if pid in R and "result" in R[pid]:
            assert max(rel(a, b) for a, b in zip(R[pid]["result"]["value"], ref)) <= 1e-15, pid


@pytest.mark.parametrize("fam", sorted(REFS))
def test_reference_methods_are_scipys(fam):
    # where the reference records the method it follows, it is the one scipy's 'auto' picked (H12)
    for pid, rec in REFS[fam].items():
        m = SCIPY.get(pid, {}).get("result", {}).get("method")
        if "method" in rec and m is not None:
            assert rec["method"] == m, pid


def test_r_levene_degrees_of_freedom():
    # car::leveneTest's Df column is the references' (df_num, df_denom) (H29, H33)
    seen = 0
    for pid, rec in REFS["levene"].items():
        r = R.get(pid, {}).get("result")
        if r:
            assert r["parameter"] == [rec["df_num"][0], rec["df_denom"][0]], pid
            seen += 1
    assert seen == len(REFS["levene"])


# ---- the tool's H5 definitions against independent forms ----------------------------------


def test_mahonian_numbers():
    T = tool()
    for n in range(1, 8):
        # every permutation's inversions counted directly
        want = Counter(sum(1 for i in range(n) for j in range(i + 1, n) if p[i] > p[j])
                       for p in itertools.permutations(range(n)))  # fmt: skip
        top = n * (n - 1) // 2
        got = T.mahonian(n, top)
        assert got == [want[k] for k in range(top + 1)]
        assert sum(got) == math.factorial(n) and got == got[::-1]


def test_weighted_tau_definitions():
    T = tool()
    x = [Fraction(v) for v in P.normal(7201, 12)]
    y = [Fraction(v) for v in P.rounded(7202, 12)]
    # equal weights: the additive weighted tau is Kendall's tau-b
    num, den = T._wtau_parts(x, y, [Fraction(1)] * 12, True)
    k = T.kendalltau(P.Problem("kendalltau/w", "kendalltau", (np.asarray(x, float), np.asarray(y, float)),
                               {"method": "asymptotic"}))  # fmt: skip
    from flint import arb

    tb = k["statistic"]()
    assert abs(float((arb(num) / arb(den).sqrt() - tb).mid())) < 1e-15  # (53-bit arbs)
    # the lexicographic rank: the largest (x, y) first, equal pairs by position
    assert T._lex_rank([Fraction(1), Fraction(3), Fraction(3), Fraction(2)], [Fraction(0)] * 4) == [3, 1, 0, 2]


def test_somers_d_and_xi_definitions():
    T = tool()
    # a 2 x 2 table: D(Y|X) = (ad - bc) / ((a + b)(c + d)), the difference of two proportions
    t = np.array([[7.0, 3.0], [2.0, 8.0]])
    d = T.somersd(P.Problem("somersd/t", "somersd", (t,), {}))["statistic"]
    assert d == Fraction(7 * 8 - 3 * 2, 10 * 10)
    # y a strictly increasing function of x: xi = 1 - 3 (n - 1) / (n^2 - 1) = (n - 2)/(n + 1)
    x = P.normal(7203, 15)
    for cont in (True, False):
        r = T.chatterjeexi(P.Problem("chatterjeexi/t", "chatterjeexi", (x, np.exp(x)), {"y_continuous": cont}))
        assert r["statistic"] == Fraction(13, 16)


# ---- the tool's H6 definitions against independent forms ----------------------------------


def test_rxc_dynamic_programme_against_enumeration():
    # the r x c tail by the columns' dynamic programme is the sum over every table (enumerated)
    T = tool()
    for t in ([[3, 1, 4], [1, 5, 9], [2, 6, 5]], [[3, 1, 0, 5], [0, 4, 6, 1]], [[2, 0, 3], [1, 4, 0], [0, 2, 2]]):
        rows, cols = [sum(r) for r in t], [sum(c) for c in zip(*t)]
        N = sum(rows)
        num = math.prod(math.factorial(v) for v in rows + cols)

        def prob(tab):
            return Fraction(num, math.factorial(N) * math.prod(math.factorial(v) for r in tab for v in r))

        tabs = list(T._tables(rows, cols))
        assert sum(prob(x) for x in tabs) == 1
        p0 = prob(t)
        want = sum(q for q in (prob(x) for x in tabs) if q <= p0 * T.RERR)
        got, stat = T._rxc_tail(t, T.RERR)
        assert got == min(Fraction(1), want) and stat == p0


def test_nuisance_supremum_against_a_dense_grid():
    # the supremum at the critical points (flint's roots) is at least every value of a dense grid of
    # pi and exceeds the largest by no more than the grid's curvature allows
    T = tool()
    for pid in ("barnard_exact/small", "barnard_exact/mirror", "boschloo_exact/large", "barnard_exact/tail"):
        rec = T.FAMILIES[P.PROBLEMS[pid].func].compute(P.PROBLEMS[pid])
        sup = float(T.rigorous(rec["pvalue"]).mid())
        t = np.asarray(P.PROBLEMS[pid].args[0]).astype(int)
        n1, n2 = int(t[0, 0] + t[1, 0]), int(t[0, 1] + t[1, 1])
        assert 0 < sup <= 1
        # Barnard's region (two-sided) again, by floats away from the ties: the grid's maximum
        if pid.startswith("barnard") and P.PROBLEMS[pid].kwargs == {}:
            obs = T._wald2(int(t[0, 0]), int(t[0, 1]), n1, n2, True)
            region = [(a, b) for a in range(n1 + 1) for b in range(n2 + 1) if T._wald2(a, b, n1, n2, True)[0] >= obs[0]]
            grid = np.linspace(1e-9, 1 - 1e-9, 20001)
            f = sum(math.comb(n1, a) * math.comb(n2, b) * grid ** (a + b) * (1 - grid) ** (n1 + n2 - a - b)
                    for a, b in region)  # fmt: skip
            assert f.max() <= sup * (1 + 1e-12) and f.max() >= sup * (1 - 1e-6), pid


def test_odds_ratio_roots_are_their_equations():
    # the conditional MLE solves E X = x, the interval's ends P(X >= x) = alpha/2 and P(X <= x) = alpha/2
    T = tool()
    t = [[3, 7], [9, 2]]
    rec = T.odds_ratio(P.PROBLEMS["odds_ratio/conditional"])
    (a, b), (c, d) = t
    M, n, N = a + b + c + d, a + b, a + c
    ks = range(max(0, n + N - M), min(n, N) + 1)
    C = {k: math.comb(n, k) * math.comb(M - n, N - k) for k in ks}

    def moments(w):
        tot = sum(C[k] * w**k for k in ks)
        return sum(k * C[k] * w**k for k in ks) / tot, tot

    w = Fraction(float(T.rigorous(rec["statistic"]).mid()))
    mean, _ = moments(w)
    assert abs(float(mean) - a) < 1e-12
    lo, hi = (Fraction(float(T.rigorous(v).mid())) for v in rec["conf_int"])
    _, tot = moments(lo)
    assert abs(float(sum(C[k] * lo**k for k in ks if k >= a) / tot) - 0.025) < 1e-14
    _, tot = moments(hi)
    assert abs(float(sum(C[k] * hi**k for k in ks if k <= a) / tot) - 0.025) < 1e-14


def test_binomial_inversion_and_score_estimates():
    """The H7 tool's own definitions against independent forms: the inverted binomial test's ends
    reject just outside and accept just inside (scipy's two-sided rule in exact rationals), and the
    score test's constrained estimate of a difference is the stationary point of the likelihood."""
    T = tool()
    for k, n in ((7, 30), (1, 50), (12, 12), (3, 7)):
        inv = T._BinomInversion(k, n, Fraction(0.05))
        lo, hi = inv.lower(), inv.upper()
        al = Fraction(0.05)
        if lo > 0:
            assert T.binom_two_sided(k, n, lo * (1 - Fraction(1, 10**12))) <= al
        for q in (lo + (hi - lo) * Fraction(i, 8) for i in range(1, 8)):
            assert T.binom_two_sided(k, n, q) > al
        if hi < 1:
            assert T.binom_two_sided(k, n, hi * (1 + Fraction(1, 10**12))) <= al
    from flint import arb

    for c1, n1, c0, n0, v in ((15, 40, 25, 45, Fraction(1, 10)), (10, 30, 20, 40, Fraction(-1, 5))):
        r = T._mn_diff_root(c1, n1, c0, n0, v)
        dl = c1 / (r + T.A(v)) - (n1 - c1) / (1 - r - T.A(v)) + c0 / r - (n0 - c0) / (1 - r)
        assert abs(dl) < arb(2) ** -40
