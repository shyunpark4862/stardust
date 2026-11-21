"""sd.htest's accuracy sweeps: one per reference family of the public tests
(tests/htest/refdata.PUBLIC: tests/htest/ref/<family>.json, tools/htest_ref.py), one region per
field (statistic, pvalue, zstatistic, df, df_num, df_denom, median, relative_risk, statistic_type,
variance, diff, ratio, odds_ratio, prop1_null, prop2_null, conf_int and odds_ratio's one-sided
conf_int_less and conf_int_greater, binomtest's conf_int_wilson and conf_int_wilsoncc; the interval
functions' whole result is conf_int), each problem of the family a point; H8's array families
(tukey_hsd, dunnett, multipletests: refdata.ARRAY_FIELDS) a point per comparison or p-value
(problem[i]); H9's fixed answers (permutation_test's and bws_test's exact enumerations,
multiscale_graphcorr's statistic, goodness_of_fit's statistic and fitted parameters param0 and
param1).

The contract is tests/htest/refdata.py's, as the tests judge it (docs/design/htest.md §4): the
statistic to 1e-14 times its condition number in what it transforms (kappa_stat), the p-value
to 1e-14 max(1, kappa_p) (times kappa_stat), degrees of freedom to 1e-14, the t tests' interval
ends to 1e-14 of the estimate's and half width's sizes. ULP is ulp_distance from the
reference's double (common.ulps).

The comparators are the recorded results of scipy 1.18.1 and statsmodels 0.15.0
(tests/htest/data/scipy_h*.json, statsmodels.json; tools/gen_htest_scipy.py, with statsmodels'
counterparts of jarque_bera, normaltest, Welch's f_oneway and tukey_hsd) and of R 4.6.2 (r.json,
nortest's and goftest's r_gof.json, coin's exact one-sided p-values r_coin.json and r_h9.json;
tools/gen_htest_r.py), measured on the statistic, the p-value and the intervals (LIB_INTERVALS)
the same way: the recorded files, so a table does not change with the installed libraries. A
library without the problem, or a field it does not give (R's statistic where its definition
differs, docs/design/htest.md §7), has no point there; nor does R on combine_pvalues, whose
record is a formula written in R, not an R function.
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region, ulps

# the families whose R statistic is scipy's (R's W, V, S, T and z differ by definition elsewhere)
R_SAME_STATISTIC = (
    "ttest", "f_oneway", "bartlett", "fligner", "levene", "power_divergence", "ks_1samp", "ks_2samp", "shapiro",
    "mannwhitneyu",
    "kruskal", "friedmanchisquare", "ansari", "mood", "chi2_contingency",
)  # fmt: skip
R_FIELD = {"statistic": "statistic", "pvalue": "p.value"}
# cor.test's statistic is its t, S or T: the correlation itself is its estimate; fisher.test's
# estimate is the conditional maximum likelihood odds ratio (odds_ratio's kind 'conditional', the
# only problems R has)
R_ESTIMATE = ("pearsonr", "spearmanrho", "kendalltau", "odds_ratio")
# R records that are no R function's result (a formula in R)
R_NOT_A_FUNCTION = ("combine_pvalues",)
# coin's exact distributions (r_coin.json, r_h9.json): one-sided p-values only (coin's two-sided
# rule is another definition)
R_COIN = ("r_coin", "r_h9")
# a library's recorded interval for each reference interval field
LIB_INTERVALS = {
    "scipy": {"conf_int": "confidence_interval", "conf_int_less": "confidence_interval_less",
              "conf_int_greater": "confidence_interval_greater"},
    "statsmodels": {"conf_int": "confidence_interval"},
    "R": {"conf_int": "conf.int"},
}  # fmt: skip
# binomtest's intervals in scipy's record: proportion_ci by method
SCIPY_BINOMTEST_INTERVALS = {"conf_int": "proportion_ci_exact", "conf_int_wilson": "proportion_ci_wilson",
                             "conf_int_wilsoncc": "proportion_ci_wilsoncc"}  # fmt: skip
# R's goodness-of-fit packages (tests/htest/data/r_gof.json, H44) in the families they share with
# sd: goftest's cvm.test is the same test (Csörgő and Faraway's p-value); nortest's ad.test has
# Anderson-Darling's statistic for the normal family, its p-value from Stephens' formula, not the
# table interpolation (another definition: the statistic only)
R_GOF = {"cramervonmises": "gof_cvm", "anderson": "ad"}
R_GOF_FIELDS = {"cramervonmises": ("statistic", "pvalue"), "anderson": ("statistic",)}


def _regions(rows) -> dict:
    """rows: {field: [(problem, value got, relative error, contract units)]} -> {field: entry}."""
    from tests.htest import refdata as R

    out = {}
    for field, pts in rows.items():
        if not pts:
            continue
        where = np.array([p for p, *_ in pts])
        got = np.array([g for _, g, _, _, _ in pts], dtype=np.float64)
        hi = np.array([h for _, _, h, _, _ in pts], dtype=np.float64)
        rel = np.array([e for *_, e, _ in pts], dtype=np.float64)
        units = np.array([c for *_, c in pts], dtype=np.float64)
        out[field] = region(where, ulp=ulps(got, hi), rel=rel, contract=units * R.BOUND, bound=R.BOUND)
    return out


def _sd_regions(fam) -> dict:
    from tests.htest import problems as P
    from tests.htest import refdata as R

    if fam in R.ARRAY_FIELDS:  # H8: a point per comparison or p-value
        rows = {}
        for pid in R.load(fam):
            for label, field, got, hi, e, c in R.points(fam, pid):
                if field != "reject":
                    rows.setdefault(field, []).append((label, got, hi, e, c))
        return _regions(rows)
    rows = {}
    for pid, rec in R.load(fam).items():
        r = R.call(P.PROBLEMS[pid])
        keys = R.interval_keys(fam)
        for field, (e, c) in R.errors(fam, pid, r).items():
            if field in keys:  # an interval: its lower end
                ci = R.interval_of(fam, field, r)
                got, hi = float(ci[0]), rec[field][0][0]
            elif field.startswith("param"):  # goodness_of_fit's fitted parameters (H9)
                got = float(getattr(r.fit_result, R.GOF_PARAMS[P.PROBLEMS[pid].args[0]][int(field[5:])]))
                hi = rec[field][0]
            else:
                got, hi = float(getattr(r, field)), rec[field][0]
            rows.setdefault(field, []).append((pid, got, hi, e, c))
    return _regions(rows)


def _recorded(lib):
    from tests.htest import refdata as R

    if lib == "R":
        return R.data("r")
    if lib == "statsmodels":
        return R.data("statsmodels")
    out = {}
    for path in sorted(R.DATA.glob("scipy_h*.json")):
        out.update(R.data(path.stem))
    return out


def _pairs_of(k, r_order=False):
    """tukey_hsd's pairs i < j (the references' order); R's TukeyHSD rows j-i in its order."""
    if r_order:
        return [(j, i) for i in range(k) for j in range(i + 1, k)]
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def _lib_array_values(fam, lib, pid, res, field):
    """A comparator's values of an H8 array field in the reference's order, or None."""
    import math as _m

    if fam == "multipletests":
        if field != "pvalues_corrected":
            return None
        v = res.get("pvals_corrected") if lib == "statsmodels" else res.get("value")
        return [float(x) for x in np.ravel(np.asarray(v, dtype=float))] if v is not None else None
    if lib == "statsmodels" and fam == "tukey_hsd":  # pairwise_tukeyhsd: mean_j - mean_i for the pairs i < j
        if field not in ("statistic", "pvalue") or "meandiffs" not in res:
            return None
        v = res["meandiffs"] if field == "statistic" else res["pvalues"]
        return [(-float(x) if field == "statistic" else float(x)) for x in v]
    if lib == "R":
        if fam == "tukey_hsd":
            key = {"statistic": "diff", "pvalue": "p adj"}.get(field)
            v = res.get(key)
            if v is None or any(x is None for x in v):
                return None
            k = int(round((1 + _m.sqrt(1 + 8 * len(v))) / 2))
            order = _pairs_of(k, r_order=True)
            vals = dict(zip(order, v))
            # R's diff is mean_j - mean_i for its row j-i: the reference's i < j is its negative
            return [(-vals[(j, i)] if field == "statistic" else vals[(j, i)]) for i, j in _pairs_of(k)]
        key = {"statistic": "statistic", "pvalue": "p.value"}.get(field)
        v = res.get(key)
        return None if v is None or any(x is None for x in v) else [float(x) for x in v]
    v = res.get(field)
    if v is None or isinstance(v, (int, float)):
        return None
    a = np.asarray(v, dtype=float)
    if fam == "tukey_hsd":
        return [float(a[i, j]) for i, j in _pairs_of(a.shape[0])]
    return [float(x) for x in np.ravel(a)]


def _lib_array_regions(fam, lib, data):
    from tests.htest import refdata as R

    rows = {}
    fields = ("pvalues_corrected",) if fam == "multipletests" else ("statistic", "pvalue")
    for pid, rec in R.load(fam).items():
        res = data.get(pid, {}).get("result") or {}
        for field in fields:
            if field not in rec:
                continue
            refs = rec[field] if isinstance(rec[field][0], list) else [rec[field]]
            vals = _lib_array_values(fam, lib, pid, res, field)
            if vals is None or len(vals) != len(refs):
                continue
            kap = rec.get("kappa")
            for i, (v, ref) in enumerate(zip(vals, refs)):
                e = R.rel(v, ref)
                k = 1.0
                if field == "pvalue" and kap is not None:
                    kk = kap[i] if isinstance(kap, list) else kap
                    k = 1.0 if kk != kk else max(1.0, kk)
                rows.setdefault(field, []).append((f"{pid}[{i}]", v, ref[0], e, e / (R.BOUND * k)))
    return _regions(rows) or None


def _lib_regions(fam, lib):
    """A comparator's regions (statistic, pvalue) from its recorded results; None without any."""
    from tests.htest import problems as P
    from tests.htest import refdata as R

    if lib == "R" and fam in R_NOT_A_FUNCTION:
        return None
    data = _recorded(lib)
    if fam in R.ARRAY_FIELDS:
        return _lib_array_regions(fam, lib, data)
    gof = R.data("r_gof") if lib == "R" else {}
    coin = {k: v for name in R_COIN for k, v in R.data(name).items()} if lib == "R" else {}
    rows = {}
    for pid, rec in R.load(fam).items():
        res = data.get(pid, {}).get("result") or {}
        alt = P.PROBLEMS[pid].kwargs.get("alternative")
        if lib == "R" and "p.value" not in res and pid in coin and alt in ("less", "greater"):
            p_one = (coin[pid].get("result") or {}).get(f"{alt}.p.value")
            res = {"p.value": p_one} if p_one is not None else res
        if lib == "R" and fam in R_GOF and pid in gof:  # nortest's and goftest's tests (H44)
            g = gof[pid].get("result") or {}
            res = {R_FIELD[f]: g.get(f"{R_GOF[fam]}.{R_FIELD[f]}") for f in R_GOF_FIELDS[fam]}
        for field in ("statistic", "pvalue", "relative_risk"):
            if field not in rec or (lib == "R" and field == "relative_risk"):
                continue
            if lib == "R":
                key = "estimate" if field == "statistic" and fam in R_ESTIMATE else R_FIELD[field]
                if key == "statistic" and fam not in R_SAME_STATISTIC and fam not in R_GOF:
                    continue
                v = res.get(key)
                v = v[0] if isinstance(v, list) and len(v) == 1 else None
            else:
                v = res.get(field)
                v = v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            if v is None:
                continue
            v = float(v)
            rows.setdefault(field, []).append(
                (pid, v, rec[field][0], R.rel(v, rec[field]), R.contract(fam, rec, field, v))
            )
        for key in R.interval_keys(fam):  # the intervals, by sd's interval contract
            if key not in rec:
                continue
            name = (SCIPY_BINOMTEST_INTERVALS if lib == "scipy" and fam == "binomtest" else LIB_INTERVALS[lib]).get(key)
            ci = res.get(name) if name else None
            if not isinstance(ci, list) or len(ci) != 2 or any(not isinstance(x, (int, float)) for x in ci):
                continue
            ci = [float(x) for x in ci]
            scale = R.interval_scale(fam, P.PROBLEMS[pid], rec[key])
            c = R.conf_int_contract({"conf_int": rec[key]}, ci, scale)
            e = max(R.rel(ci[i], rec[key][i]) for i in range(2))
            rows.setdefault(key, []).append((pid, ci[0], rec[key][0][0], e, max(c)))
    return _regions(rows) or None


def sweeps():
    from tests.htest import refdata as R

    out = []
    for fam in R.PUBLIC:

        def run(fam=fam):
            return _sd_regions(fam)

        def measure(lib, fam=fam):
            return _lib_regions(fam, lib)

        out.append(Custom("htest", fam, run, compare_libs(measure, ("scipy", "statsmodels", "R"))))
    return out
