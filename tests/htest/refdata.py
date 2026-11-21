"""sd.htest's references and their contract (docs/design/htest.md §4, §7), as the tests and the
accuracy measurement (bench/accuracy/htest.py) both judge them.

tests/htest/ref/<family>.json (tools/htest_ref.py) holds, per problem of tests/htest/problems.py,
the inputs' digest and each reference field as a double-double [hi, lo] ("kappa", "kappa_stat":
plain condition numbers); its "_provenance" says what computed it. tests/htest/data holds the
other libraries' records (tools/gen_htest_scipy.py: scipy_h*.json, statsmodels.json;
tools/gen_htest_r.py: r.json), each with its own "_provenance".

The contract, per field, in units of its bound (1 or less is within it):
  - statistic: relative error / (1e-14 max(1, kappa_stat)), kappa_stat the statistic's condition
    number in the quantity it transforms (D'Agostino's moments, Anderson-Darling's cancelling
    sum; 1 elsewhere);
  - pvalue: relative error / (1e-14 max(1, kappa) max(1, kappa_stat)) (Pearson's and Spearman's
    kappa is the condition number in q = 1 - r^2, which sd forms without cancellation); the Cramér-von Mises
    p-values (1 - cdf of Csörgő and Faraway's approximation, one sample; 1 - V of the limit,
    the two-sample asymptotic one) also pass within 1e-15 absolute (H41);
  - zstatistic (the normal approximation's z of the rank tests), df, df_num, df_denom, median
    (median_test's grand median): relative error / 1e-14;
  - conf_int (the t tests' and pearsonr's 95% interval): each end's error / (1e-14 times the larger of the end
    and the estimate's and half width's sizes: the ends subtract them); odds_ratio's and relative_risk's
    ends (positive, no subtraction) each against itself, odds_ratio's one-sided ones too
    (conf_int_less, conf_int_greater); H7's: binomtest's three (conf_int Clopper-Pearson's,
    conf_int_wilson, conf_int_wilsoncc) and quantile_test's order statistics each against itself,
    proportion_confint's (the result itself) against itself but 'normal' and 'agresti_coull'
    (q -+ z se: the interval's size), confint_proportions_2indep's ratios and odds ratios against
    themselves and its differences against the interval's size;
  - H8's array fields element by element (points()): tukey_hsd's and dunnett's mean differences or t
    statistics, p-values (with kappa per comparison) and interval ends, multipletests' corrected
    p-values (false_discovery_control's result itself) and its rejections exactly;
  - H9's fixed answers: permutation_test's and bws_test's statistics and exact p-values (counts
    of the enumeration), multiscale_graphcorr's statistic, goodness_of_fit's fitted parameters
    (param0, param1: relative error / 1e-14) and statistic (its p-value is random: none).
"""

import json
import math
import pathlib

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
BOUND = 1e-14
CVM_ABS = 1e-15

# the families by milestone
H1 = ("ttest", "f_oneway", "alexandergovern", "bartlett", "levene", "fligner")
H2 = ("power_divergence", "jarque_bera", "dagostino", "shapiro", "anderson", "ks_1samp", "cramervonmises")
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
H4 = ("ks_2samp", "cramervonmises_2samp", "anderson_ksamp", "epps_singleton_2samp")
H5 = ("pearsonr", "spearmanrho", "kendalltau", "weightedtau", "somersd", "chatterjeexi")
H6 = (
    "chi2_contingency",
    "fisher_exact",
    "barnard_exact",
    "boschloo_exact",
    "mcnemar",
    "cochrans_q",
    "odds_ratio",
    "relative_risk",
)
H7 = (
    "binomtest",
    "quantile_test",
    "poisson_means_test",
    "proportions_ztest",
    "proportions_chisquare",
    "proportion_confint",
    "test_proportions_2indep",
    "confint_proportions_2indep",
)
H8 = ("tukey_hsd", "dunnett", "multipletests", "combine_pvalues")
H9 = ("permutation_test", "bws_test", "multiscale_graphcorr", "goodness_of_fit")
PUBLIC = (
    H1 + H2 + H3 + H4 + H5 + H6 + H7 + H8 + H9
)  # the families of the public functions (the accuracy measurement's sweeps)
# goodness_of_fit's sd parameter names by scipy's family name, in the references' order (param0, ...)
GOF_PARAMS = {
    "norm": ("mu", "sigma"),
    "expon": ("scale",),
    "logistic": ("mu", "sigma"),
    "gumbel_r": ("mu", "sigma"),
    "weibull_min": ("c", "scale"),
    "lognorm": ("mu", "sigma"),
    "laplace": ("mu", "sigma"),
    "rayleigh": ("scale",),
}
# scipy's known_params names as sd's: the location-scale families' loc and scale are mu and sigma;
# a location held at 0 (sd's families without one) is left out
_GOF_KNOWN = {"loc": "mu", "scale": "sigma"}
# H8's families whose fields are arrays (a value per comparison or per p-value): measured element by
# element (points())
ARRAY_FIELDS = ("tukey_hsd", "dunnett", "multipletests")
# the estimates whose interval ends are measured each against itself (no estimate to subtract): the
# ratios, binomtest's intervals (Clopper-Pearson's beta quantiles, Wilson's roots formed without
# cancellation) and quantile_test's order statistics
OWN_SCALE = ("odds_ratio", "relative_risk", "binomtest", "quantile_test")
# proportion_confint's methods whose ends subtract (q -+ z se): measured against the interval's size
SUBTRACTING = ("normal", "agresti_coull")
# the families whose result is the interval itself, a (..., 2) array (statsmodels' (low, upp))
ARRAY_INTERVALS = ("proportion_confint", "confint_proportions_2indep")
# binomtest's intervals by conf_int's method (scipy's proportion_ci)
BINOMTEST_INTERVALS = (("conf_int", "exact"), ("conf_int_wilson", "wilson"), ("conf_int_wilsoncc", "wilsoncc"))


def load(fam) -> dict:
    """{problem: record} of a reference family (the file's "_provenance" left out)."""
    d = json.loads((REF / f"{fam}.json").read_text())
    d.pop("_provenance", None)
    return d


def provenance(fam) -> dict:
    return json.loads((REF / f"{fam}.json").read_text())["_provenance"]


def families() -> list:
    return sorted(p.stem for p in REF.glob("*.json"))


def data(name) -> dict:
    """{problem: record} of another library's file in tests/htest/data ("_provenance" left out)."""
    d = json.loads((DATA / f"{name}.json").read_text())
    d.pop("_provenance", None)
    return d


def _sd_args(p):
    import stardust as sd

    cdf = {"norm": sd.stats.Normal(), "uniform": sd.stats.Uniform(a=0.0, b=1.0), "expon": sd.stats.Exponential()}
    dist = {"norm": None, "expon": sd.stats.Exponential, "logistic": sd.stats.Logistic, "gumbel_r": sd.stats.Gumbel}
    args = [cdf[a] if isinstance(a, str) and a in cdf else a for a in p.args]
    kw = dict(p.kwargs)
    if p.func == "anderson":
        kw["dist"] = dist[kw.get("dist", "norm")]
    # statsmodels' names (H85, H86): 'beta' is 'exact'; its 'larger' and 'smaller' are 'greater' and
    # 'less', but proportion_confint's 'larger' is the interval [0, upper], sd's 'less' (H95)
    if kw.get("alternative") in ("larger", "smaller"):
        flip = p.func == "proportion_confint"
        kw["alternative"] = {"larger": "less" if flip else "greater", "smaller": "greater" if flip else "less"}[
            kw["alternative"]
        ]
    if p.func == "proportion_confint" and kw.get("method") == "beta":
        kw["method"] = "exact"
    if "maxiter" in kw:  # statsmodels' multipletests (H110)
        kw["max_iter"] = kw.pop("maxiter")
    if p.milestone == "H9":
        from . import problems as P

        if p.func == "permutation_test":
            args = [list(p.args[0]), P.statistic(p.args[1], sd)]
        elif p.func == "goodness_of_fit":
            dist = p.args[0]
            args = [getattr(sd.stats, P.GOF_FAMILIES[dist][0]), p.args[1]]
            known = {}
            for k, v in kw.pop("known_params", {}).items():
                if dist in ("norm", "logistic", "gumbel_r", "laplace"):
                    known[_GOF_KNOWN[k]] = v
                elif k != "loc":
                    known[k] = v
            if known:
                kw["known_params"] = known
            kw["rng"] = sd.random.default_rng(12345)
        elif p.func == "multiscale_graphcorr":
            kw["rng"] = sd.random.default_rng(12345)
    return args, kw


def call(p, **extra):
    """sd's call of a problem: scipy's distribution names as sd.stats objects (cdf) or classes
    (anderson's dist)."""
    import stardust as sd

    args, kw = _sd_args(p)
    return getattr(sd.htest, p.func)(*args, **{**kw, **extra})


def rel(v, ref):
    """|v - ref| / |ref| for ref a double-double [hi, lo] (NaN against NaN is 0, inf against the
    same inf 0)."""
    hi, lo = ref
    if hi != hi:
        return 0.0 if v != v else math.inf
    if math.isinf(hi):
        return 0.0 if v == hi else math.inf
    if v != v:
        return math.inf
    if hi == 0.0:
        return abs(v)
    return abs((v - hi) - lo) / abs(hi)


def factor(rec, key):
    k = rec.get(key, 1.0)
    return 1.0 if k != k else max(1.0, k)


def contract(fam, rec, field, value):
    """value's error in units of the field's bound (module docstring)."""
    ks = factor(rec, "kappa_stat")
    e = rel(value, rec[field])
    if field == "statistic":
        return e / (BOUND * ks)
    if field == "pvalue":
        c = e / (BOUND * factor(rec, "kappa") * ks)
        if (
            fam in ("cramervonmises", "cramervonmises_2samp")
            and rec[field][0] == rec[field][0]
            and abs(value - rec[field][0]) <= CVM_ABS
        ):
            c = min(c, 1.0)
        return c
    return e / BOUND


def conf_int_contract(rec, ci, scale):
    """The interval's ends against the reference: error / (1e-14 max(|end|, scale)) each."""
    out = []
    for i in range(2):
        hi, lo = rec["conf_int"][i]
        v = float(ci[i])
        if hi != hi or math.isinf(hi) or (hi == 0.0 and scale == 0.0):
            out.append(0.0 if v == hi or (v != v and hi != hi) else math.inf)
        else:
            out.append(abs((v - hi) - lo) / (BOUND * max(abs(hi), scale)))
    return out


def ci_scale(p):
    """The size an interval's ends are measured against: the two-sided 95% (or 90% for a
    one-sided test, the same end) interval's estimate and half width."""
    alt = p.kwargs.get("alternative", "two-sided")
    two = call(p) if alt == "two-sided" else call(p, alternative="two-sided")
    lo2, hi2 = two.conf_int(alpha=0.05 if alt == "two-sided" else 0.1).tolist()
    return abs(lo2 + hi2) / 2 + abs(hi2 - lo2) / 2


FIELDS = (
    "statistic",
    "pvalue",
    "zstatistic",
    "df",
    "df_num",
    "df_denom",
    "median",
    "relative_risk",
    # H7: quantile_test's, and test_proportions_2indep's (statsmodels' fields)
    "statistic_type",
    "variance",
    "diff",
    "ratio",
    "odds_ratio",
    "prop1_null",
    "prop2_null",
)
# odds_ratio's intervals by the alternative of conf_int (its own argument, scipy's)
INTERVALS = (("conf_int", "two-sided"), ("conf_int_less", "less"), ("conf_int_greater", "greater"))


def interval_keys(fam):
    """The reference's interval fields of a family."""
    if fam == "binomtest":
        return [k for k, _ in BINOMTEST_INTERVALS]
    if fam in ARRAY_INTERVALS:
        return ["conf_int"]
    return [k for k, _ in INTERVALS]


def interval_of(fam, key, r):
    """sd's interval for the reference field `key` of result r (an array family's result is it)."""
    if fam in ARRAY_INTERVALS:
        return r
    if fam == "binomtest":
        return r.conf_int(method=dict(BINOMTEST_INTERVALS)[key])
    return r.conf_int() if key == "conf_int" else r.conf_int(alternative=dict(INTERVALS)[key])


def _ref_scale(ends):
    """The size of an interval from its reference ends (estimate and half width)."""
    lo, hi = ends[0][0], ends[1][0]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0.0
    return abs(lo + hi) / 2 + abs(hi - lo) / 2


def interval_scale(fam, p, ends):
    if fam in OWN_SCALE:
        return 0.0
    if fam == "proportion_confint":
        return _ref_scale(ends) if p.kwargs.get("method", "normal") in SUBTRACTING else 0.0
    if fam == "confint_proportions_2indep":
        return _ref_scale(ends) if p.kwargs.get("compare", "diff") == "diff" else 0.0
    return ci_scale(p)


def _pairs(k):
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def _elements(fam, rec, r, field):
    """sd's values of an array field in the reference's order: tukey_hsd's pairs i < j (row-major),
    dunnett's comparisons, multipletests' p-values (false_discovery_control's result itself)."""
    import numpy as np

    if fam == "multipletests":
        v = r if not hasattr(r, "pvalues_corrected") else (r.reject if field == "reject" else r.pvalues_corrected)
        return [float(x) for x in np.ravel(np.asarray(v, dtype=float))]
    v = getattr(r, field)
    if fam == "tukey_hsd" and not isinstance(v, float):
        a = np.asarray(v, dtype=float)
        return [float(a[i, j]) for i, j in _pairs(a.shape[0])]
    if isinstance(v, float):
        return [v] * len(rec[field]) if isinstance(rec[field][0], list) else [v]
    return [float(x) for x in np.ravel(np.asarray(v, dtype=float))]


def _interval_ends(fam, r, n):
    """sd's 95% interval ends per comparison (tukey_hsd's pairs i < j, dunnett's comparisons)."""
    import numpy as np

    ci = np.asarray(r.conf_int(), dtype=float)
    if fam == "tukey_hsd":
        return [(float(ci[i, j, 0]), float(ci[i, j, 1])) for i, j in _pairs(ci.shape[0])]
    return [(float(ci[i, 0]), float(ci[i, 1])) for i in range(n)]


def points(fam, pid, r=None) -> list:
    """H8's array families element by element: [(label, field, sd's value, the reference's hi, relative
    error, error in bound units)]. The p-values against 1e-14 max(1, kappa) (kappa per element: the
    studentized range's and the multivariate t's condition number in the statistic), the corrected
    p-values and the statistics (mean differences, t's) against 1e-14 of themselves, degrees of
    freedom too, the interval ends against 1e-14 of the larger of the end and the estimate's and
    half width's sizes, and the rejections exactly (an unequal one is inf)."""
    from . import problems as P

    rec = load(fam)[pid]
    p = P.PROBLEMS[pid]
    r = call(p) if r is None else r
    out = []
    fields = ("reject", "pvalues_corrected") if fam == "multipletests" else ("statistic", "pvalue", "df")
    for field in fields:
        if field not in rec:
            continue
        refs = rec[field] if isinstance(rec[field][0], list) or field == "reject" else [rec[field]]
        got = _elements(fam, rec, r, field)
        kap = rec.get("kappa")
        for i, (g, ref) in enumerate(zip(got, refs)):
            if field == "reject":
                bad = float(g != float(ref))
                out.append((f"{pid}[{i}]", field, g, float(ref), bad, math.inf if bad else 0.0))
                continue
            e = rel(g, ref)
            k = 1.0 if field != "pvalue" or kap is None else max(1.0, kap[i] if isinstance(kap, list) else kap)
            if k != k:
                k = 1.0
            out.append((f"{pid}[{i}]", field, g, ref[0], e, e / (BOUND * k)))
    if "conf_int" in rec:
        ends = _interval_ends(fam, r, len(rec["conf_int"]))
        for i, ((lo, hi), ref) in enumerate(zip(ends, rec["conf_int"])):
            scale = _ref_scale(ref)
            c = conf_int_contract({"conf_int": ref}, (lo, hi), scale)
            e = max(rel(lo, ref[0]), rel(hi, ref[1]))
            out.append((f"{pid}[{i}]", "conf_int", lo, ref[0][0], e, max(c)))
    return out


def errors(fam, pid, r=None) -> dict:
    """{field: (relative error, error in bound units)} of sd's result for a reference problem
    (conf_int: its two ends)."""
    from . import problems as P

    if fam in ARRAY_FIELDS:
        out = {}
        for _, field, _, _, e, c in points(fam, pid, r):
            old = out.get(field, (0.0, 0.0))
            out[field] = (max(old[0], e), max(old[1], c))
        return out
    rec = load(fam)[pid]
    p = P.PROBLEMS[pid]
    r = call(p) if r is None else r
    out = {}
    if fam == "goodness_of_fit":  # the fitted parameters (the fit_result's, sd's names)
        names = GOF_PARAMS[p.args[0]]
        for i, k in enumerate(names):
            if f"param{i}" in rec:
                v = getattr(r.fit_result, k)
                out[f"param{i}"] = (rel(v, rec[f"param{i}"]), rel(v, rec[f"param{i}"]) / BOUND)
    if fam not in ARRAY_INTERVALS:
        for f in FIELDS:
            if f in rec and hasattr(r, f):
                v = getattr(r, f)
                out[f] = (rel(v, rec[f]), contract(fam, rec, f, v))
    for key in interval_keys(fam):
        if key not in rec:
            continue
        ci = interval_of(fam, key, r)
        scale = interval_scale(fam, p, rec[key])
        c = conf_int_contract({"conf_int": rec[key]}, ci, scale)
        out[key] = (max(rel(float(ci[i]), rec[key][i]) for i in range(2)), max(c))
    return out
