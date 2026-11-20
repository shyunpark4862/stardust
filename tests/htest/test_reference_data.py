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
    if ref != ref:
        return 0.0 if v != v else math.inf
    if v != v:
        return math.inf
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
                "jarque_bera", "dagostino", "shapiro", "anderson", "ks_1samp", "cramervonmises"):  # fmt: skip
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
