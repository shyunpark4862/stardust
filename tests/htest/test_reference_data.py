"""The reference and comparison data of sd.htest (docs/design/htest.md §7): every file belongs to
the problem collection as it is (digests), the reference tool's exact distributions pass checks
of their own, and the recorded scipy, statsmodels and R results agree with the references except
where they are known not to (the lists below, measured when the data was made: the comparison
highlights of H0). The recorded files are compared, not the installed libraries, so the tests
do not change with them."""

import importlib.util
import json
import math
import pathlib
import sys
from fractions import Fraction

import pytest

from . import problems as P

ROOT = pathlib.Path(__file__).parents[2]
REF = ROOT / "tests" / "htest" / "ref"
DATA = ROOT / "tests" / "htest" / "data"


def load(path):
    d = json.loads(path.read_text())
    meta = d.pop("_meta")
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
    if ref != ref:
        return 0.0 if v != v else math.inf
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


def test_versions():
    meta = load(DATA / "scipy_h1.json")[0]
    assert meta["scipy"] == "1.18.1" and meta["statsmodels"] == "0.15.0"
    assert load(DATA / "r.json")[0]["R"].startswith("R version 4.6.2")


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


def test_two_sample_ks_paths_against_the_closed_form():
    # equal sizes: P(D >= h/n) = 2 sum_k (-1)^(k-1) C(2n, n - k h) / C(2n, n) (Gnedenko-Korolyuk)
    T = tool()
    for n in (5, 12, 30):
        for h in range(1, n + 1):
            closed = sum((-1) ** (k - 1) * 2 * math.comb(2 * n, n - k * h) for k in range(1, n // h + 1))
            assert T.ks2_paths(n, n, 1, 1, h, "two-sided") == closed, (n, h)
            assert T.ks2_paths(n, n, 1, 1, h, "greater") == math.comb(2 * n, n - h), (n, h)


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
}
# R's: the same cancellations (and ttest_rel's x - y), its own methods where they differ from scipy's
# (ks.test is exact only below n = 100 (one sample) or m n = 10^4 (two), its two-sample exact
# distribution is conditional on the ties, wilcox.test is exact up to n = 50 and conditional with
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
    ("ks_2samp/exact-100-120", "pvalue"): 0.22,
    ("ks_2samp/exact-less", "pvalue"): 0.012,
    ("ks_2samp/exact-greater", "pvalue"): 0.22,
    ("ks_2samp/ties", "pvalue"): 0.58,
    ("ks_2samp/tail", "pvalue"): 137,
    ("mannwhitneyu/ties", "pvalue"): 7.1e-3,
    ("mannwhitneyu/no-continuity", "pvalue"): 2.4e-3,
    ("spearmanrho/small", "pvalue"): 1.5,
    ("spearmanrho/large", "pvalue"): 1.1e-4,
    ("spearmanr/small", "pvalue"): 1.5,
    ("wilcoxon/zeros-wilcox", "pvalue"): 0.16,
    ("wilcoxon/tail", "pvalue"): 1.0,
}
# the families whose R statistic is scipy's (R's W, V, S, T and z differ by definition elsewhere)
R_SAME_STATISTIC = {"ttest", "f_oneway", "bartlett", "power_divergence", "ks_1samp", "ks_2samp", "mannwhitneyu"}
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
