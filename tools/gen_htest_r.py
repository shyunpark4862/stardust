"""R's base stats results on the sd.htest problem collection (tests/htest/problems.py;
docs/design/htest.md §7). Writes tests/htest/data/r.json, so the tests and the accuracy tables
compare with R on machines without R (as tools/gen_regression_r.py does for sd.regression).

R's base `stats` package (decision H15), and the `car` package's leveneTest for levene (decision
H33; car is the regression area's comparison package, installed for G7a): t.test,
oneway.test, bartlett.test, fligner.test, chisq.test, shapiro.test, ks.test, wilcox.test,
kruskal.test, friedman.test, ansari.test, mood.test, cor.test, fisher.test, mcnemar.test,
binom.test, prop.test, TukeyHSD, p.adjust, pchisq, car::leveneTest. A problem is run where R has
the same test with the same options; the others (Yuen's trim, alexandergovern, the Cressie-Read family,
cramervonmises, anderson, brunnermunzel, median_test, page_trend_test, epps_singleton, batches,
...) are left out. Where R's definition differs from scipy's (ks.test's exact distribution with
ties, wilcox.test's exact switch at n < 50, fisher.test's conditional MLE odds ratio, prop.test's
one-sample variance), the record keeps R's own value; the comparison says which differ.

Each record: the R call (its data vectors shown as c(...)), the inputs' digest, and R's htest fields: statistic,
parameter (df), p.value, conf.int, estimate; for TukeyHSD the matrix of diff, lwr, upr, p adj;
for p.adjust the adjusted vector. Doubles go to R as %.17g (exact) and come back as %.17g.

Also writes tests/htest/data/r_gof.json: the goodness-of-fit packages nortest and goftest (decision
H44; GOF_PRELUDE below).

Run: .venv/bin/python tools/gen_htest_r.py      (needs Rscript on PATH, and nortest and goftest)
"""

import json
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "htest" / "data" / "r.json"
OUT_GOF = ROOT / "tests" / "htest" / "data" / "r_gof.json"  # nortest and goftest (H44)

PRELUDE = r"""
out <- function(id, name, v) {
  v <- as.numeric(v)
  cat(id, "\t", name, "\t", paste(ifelse(is.na(v), "NA", sprintf("%.17g", v)), collapse = " "), "\n", sep = "")
}
htest <- function(id, h) {
  out(id, "statistic", h$statistic)
  if (!is.null(h$parameter)) out(id, "parameter", h$parameter)
  out(id, "p.value", h$p.value)
  if (!is.null(h$conf.int)) out(id, "conf.int", h$conf.int)
  if (!is.null(h$estimate)) out(id, "estimate", h$estimate)
}
run <- function(id, expr) {
  r <- tryCatch(suppressWarnings(expr), error = function(e) e)
  if (inherits(r, "error")) cat(id, "\terror\t", gsub("[\t\n]", " ", conditionMessage(r)), "\n", sep = "")
  else if (inherits(r, "htest")) htest(id, r)
  else if (is.list(r)) for (nm in names(r)) out(id, nm, r[[nm]])
  else out(id, "value", r)
}
"""

ALT = {"two-sided": "two.sided", "less": "less", "greater": "greater"}


def vec(a):
    a = np.ravel(np.asarray(a, dtype=np.float64))
    return "c(" + ", ".join(f"{v:.17g}" for v in a) + ")"


def mat(a):
    a = np.asarray(a, dtype=np.float64)
    return f"matrix({vec(a.T)}, nrow = {a.shape[0]})"


def groups(xs):
    return "list(" + ", ".join(vec(x) for x in xs) + ")"


def alt(p):
    return ALT[p.kwargs.get("alternative", "two-sided")]


def exact(method, default="NULL"):
    return {"exact": "TRUE", "asymptotic": "FALSE", "asymp": "FALSE", "auto": default}.get(method, default)


def r_call(p):
    """R's call for problem p, or None where base R has no matching test."""
    a, kw, f = p.args, p.kwargs, p.func
    if "batch" in p.tags or "axis" in kw:
        return None
    if f == "ttest_1samp":
        return f"t.test({vec(a[0])}, mu = {a[1]!r}, alternative = '{alt(p)}')"
    if f == "ttest_rel":
        return f"t.test({vec(a[0])}, {vec(a[1])}, paired = TRUE, alternative = '{alt(p)}')"
    if f == "ttest_ind" and "trim" not in kw:
        ve = "TRUE" if kw.get("equal_var", True) else "FALSE"
        return f"t.test({vec(a[0])}, {vec(a[1])}, var.equal = {ve}, alternative = '{alt(p)}')"
    if f == "f_oneway":
        ve = "TRUE" if kw.get("equal_var", True) else "FALSE"
        g = "factor(rep(seq_along(L), lengths(L)))"
        return f"local({{L <- {groups(a)}; oneway.test(unlist(L) ~ {g}, var.equal = {ve})}})"
    if f == "bartlett":
        return f"bartlett.test({groups(a)})"
    if f == "fligner" and kw.get("center", "median") == "median":
        return f"fligner.test({groups(a)})"
    if f == "levene":
        # car's center: R's median, mean, or mean(trim = p) (floor(n p) from each end: scipy's g)
        c = kw.get("center", "median")
        ctr = {"median": "median", "mean": "mean"}.get(
            c, f"function(x) mean(x, trim = {kw.get('proportiontocut', 0.05)!r})"
        )
        g = "factor(rep(seq_along(L), lengths(L)))"
        return (
            f"local({{L <- {groups(a)}; r <- car::leveneTest(unlist(L), {g}, center = {ctr}); "
            "list(statistic = r[1, 'F value'], parameter = r[, 'Df'], p.value = r[1, 'Pr(>F)'])})"
        )
    if f == "chisquare" and "ddof" not in kw:
        if len(a) > 1:
            return f"chisq.test({vec(a[0])}, p = {vec(a[1])}, rescale.p = TRUE)"
        return f"chisq.test({vec(a[0])})"
    if f == "shapiro" and 3 <= np.size(a[0]) <= 5000:
        return f"shapiro.test({vec(a[0])})"
    if f == "ks_1samp" and a[1] in ("norm", "uniform", "expon") and np.ndim(a[0]) == 1:
        cdf = {"norm": "pnorm", "uniform": "punif", "expon": "pexp"}[a[1]]
        return f"ks.test({vec(a[0])}, '{cdf}', alternative = '{alt(p)}', exact = {exact(kw.get('method', 'auto'))})"
    if f == "ks_2samp":
        e = exact(kw.get("method", "auto"))
        return f"ks.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}', exact = {e})"
    if f == "mannwhitneyu":
        c = "TRUE" if kw.get("use_continuity", True) else "FALSE"
        e = exact(kw.get("method", "auto"))
        return f"wilcox.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}', correct = {c}, exact = {e})"
    if f == "ranksums":
        return f"wilcox.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}', correct = FALSE, exact = FALSE)"
    if f == "wilcoxon" and kw.get("zero_method", "wilcox") == "wilcox":
        c = "TRUE" if kw.get("correction", False) else "FALSE"
        e = exact(kw.get("method", "auto"))
        if len(a) > 1:
            return f"wilcox.test({vec(a[0])}, {vec(a[1])}, paired = TRUE, alternative = '{alt(p)}', correct = {c}, exact = {e})"  # noqa: E501
        return f"wilcox.test({vec(a[0])}, alternative = '{alt(p)}', correct = {c}, exact = {e})"
    if f == "kruskal":
        return f"kruskal.test({groups(a)})"
    if f == "friedmanchisquare":
        return f"friedman.test(cbind({', '.join(vec(x) for x in a)}))"
    if f == "ansari":
        e = exact(kw.get("method", "auto"))
        return f"ansari.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}', exact = {e})"
    if f == "mood":
        return f"mood.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}')"
    if f in ("pearsonr", "pointbiserialr"):
        return f"cor.test({vec(a[0])}, {vec(a[1])}, method = 'pearson', alternative = '{alt(p)}')"
    if f in ("spearmanrho", "spearmanr"):
        return f"cor.test({vec(a[0])}, {vec(a[1])}, method = 'spearman', alternative = '{alt(p)}')"
    if f == "kendalltau" and kw.get("variant", "b") == "b":
        e = exact(kw.get("method", "auto"))
        return f"cor.test({vec(a[0])}, {vec(a[1])}, method = 'kendall', alternative = '{alt(p)}', exact = {e})"
    if f == "chi2_contingency" and "lambda_" not in kw:
        c = "TRUE" if kw.get("correction", True) else "FALSE"
        return f"chisq.test({mat(a[0])}, correct = {c})"
    if f == "fisher_exact":
        return f"fisher.test({mat(a[0])}, alternative = '{alt(p)}')"
    if f == "mcnemar":
        t = np.asarray(a[0])
        if kw.get("exact", True):
            return f"binom.test({float(t[0, 1])!r}, {float(t[0, 1] + t[1, 0])!r}, 0.5)"
        c = "TRUE" if kw.get("correction", True) else "FALSE"
        return f"mcnemar.test({mat(t)}, correct = {c})"
    if f == "binomtest":
        return f"binom.test({a[0]}, {a[1]}, p = {kw.get('p', 0.5)!r}, alternative = '{alt(p)}')"
    if f == "proportions_ztest" and np.ndim(a[0]) == 1:
        return f"prop.test({vec(a[0])}, {vec(a[1])}, correct = FALSE)"
    if f == "proportions_chisquare":
        return f"prop.test({vec(a[0])}, {vec(a[1])}, correct = FALSE)"
    if f == "proportion_confint" and kw.get("method") in ("beta", "wilson"):
        if kw["method"] == "beta":
            return f"binom.test({a[0]}, {a[1]})"
        return f"prop.test({a[0]}, {a[1]}, correct = FALSE)"
    if f == "tukey_hsd" and kw.get("equal_var", True):
        g = "factor(rep(seq_along(L), lengths(L)))"
        return f"local({{L <- {groups(a)}; T <- TukeyHSD(aov(unlist(L) ~ {g}), conf.level = 0.95)[[1]]; as.list(as.data.frame(T))}})"  # noqa: E501
    if f == "multipletests":
        rm = {
            "bonferroni": "bonferroni",
            "holm": "holm",
            "simes-hochberg": "hochberg",
            "hommel": "hommel",
            "fdr_bh": "BH",
            "fdr_by": "BY",
        }.get(kw.get("method"))
        if rm and "alpha" not in kw:
            return f"p.adjust({vec(a[0])}, method = '{rm}')"
    if f == "false_discovery_control":
        return f"p.adjust({vec(a[0])}, method = '{'BH' if kw['method'] == 'bh' else 'BY'}')"
    if f == "combine_pvalues" and kw.get("method", "fisher") == "fisher":
        k = np.size(a[0])
        return f"local({{s <- -2 * sum(log({vec(a[0])})); list(statistic = s, p.value = pchisq(s, {2 * k}, lower.tail = FALSE))}})"  # noqa: E501
    return None


def shown(call):
    """The call with its data vectors written c(...): the inputs are the problem's (digest)."""
    return re.sub(r"c\([^()]*\)", "c(...)", call)


# The goodness-of-fit packages (H44: installed for comparison only; nortest 1.0-4 and goftest 1.2-3,
# both importing only stats): on the normal-data problems of anderson and shapiro, nortest's five
# tests of normality with estimated parameters (ad.test, cvm.test, lillie.test, pearson.test with
# its class counts, sf.test) and goftest's ad.test with the fitted normal taken as given; on the
# cramervonmises problems, goftest's cvm.test and ad.test against the problem's distribution.
GOF_PRELUDE = """
nt_pearson <- function(x) {
  n <- length(x); k <- ceiling(2 * (n^(2/5)))
  num <- floor(1 + k * pnorm(x, mean(x), sd(x)))
  h <- nortest::pearson.test(x)
  list(statistic = h$statistic, p.value = h$p.value, counts = tabulate(num, k))
}
gof_list <- function(...) {
  L <- list(...)
  out <- list()
  for (nm in names(L)) {
    r <- tryCatch(L[[nm]], error = function(e) NULL)
    if (is.null(r)) next
    out[[paste0(nm, ".statistic")]] <- r$statistic
    out[[paste0(nm, ".p.value")]] <- r$p.value
    if (!is.null(r$counts)) out[[paste0(nm, ".counts")]] <- r$counts
  }
  out
}
"""
GOF_CDF = {"norm": "pnorm", "uniform": "punif", "expon": "pexp"}


def r_gof_call(p):
    """The nortest and goftest calls of problem p as one named list, or None."""
    if "batch" in p.tags or "axis" in p.kwargs or p.func not in ("anderson", "shapiro", "cramervonmises"):
        return None
    x = vec(p.args[0])
    n = int(np.size(p.args[0]))
    if (p.func == "anderson" and p.kwargs.get("dist", "norm") == "norm") or p.func == "shapiro":
        if n < 8 or np.ptp(p.args[0]) == 0:
            return None
        sf = "nortest::sf.test(x)" if n <= 5000 else "NULL"
        return (
            f"local({{x <- {x}; gof_list(ad = nortest::ad.test(x), cvm = nortest::cvm.test(x), "
            f"lillie = nortest::lillie.test(x), pearson = nt_pearson(x), sf = {sf}, "
            "gof_ad = goftest::ad.test(x, 'pnorm', mean = mean(x), sd = sd(x)))})"
        )
    if p.func == "cramervonmises" and p.args[1] in GOF_CDF and n >= 2:
        f = GOF_CDF[p.args[1]]
        return (
            f"local({{x <- {x}; gof_list(gof_cvm = goftest::cvm.test(x, '{f}'), gof_ad = goftest::ad.test(x, '{f}'))}})"
        )
    return None


def run_r(prelude, calls, P):
    """{problem: record} of R's results for {problem: call}, and R's version."""
    script = prelude + "\n".join(f"run({json.dumps(pid)}, {c})" for pid, c in calls.items())
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "htest.R"
        path.write_text(script)
        res = subprocess.run(["Rscript", "--vanilla", str(path)], capture_output=True, text=True, check=True)
        version = subprocess.run(
            ["Rscript", "--vanilla", "-e", "cat(R.version.string)"], capture_output=True, text=True, check=True
        ).stdout.strip()
    recs = {pid: {"call": shown(c), "digest": P.digest(P.PROBLEMS[pid])} for pid, c in calls.items()}
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        pid, field, vals = line.split("\t", 2)
        rec = recs[pid].setdefault("result", {})
        if field == "error":
            recs[pid]["error"] = vals.strip()
            continue
        rec[field] = [None if t == "NA" else float(t) for t in vals.split()]
    return recs, version


def write(path, prov, recs):
    lines = [f"{json.dumps('_provenance')}: {json.dumps(prov)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in recs.items()]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def main():
    from tests.htest import problems as P

    calls = {pid: r_call(p) for pid, p in P.PROBLEMS.items()}
    calls = {k: v for k, v in calls.items() if v is not None}
    recs, version = run_r(PRELUDE, calls, P)
    write(OUT, {"tool": "tools/gen_htest_r.py", "R": version}, recs)
    print(f"{len(recs)} problems run in R ({version})")
    gof = {pid: r_gof_call(p) for pid, p in P.PROBLEMS.items()}
    gof = {k: v for k, v in gof.items() if v is not None}
    recs, _ = run_r(PRELUDE + GOF_PRELUDE, gof, P)
    pkgs = subprocess.run(
        ["Rscript", "--vanilla", "-e", 'cat(sapply(c("nortest", "goftest"), function(p) paste(p, packageVersion(p))))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    write(OUT_GOF, {"tool": "tools/gen_htest_r.py", "R": version, "packages": pkgs}, recs)
    print(f"{len(recs)} problems run with nortest and goftest ({pkgs})")


if __name__ == "__main__":
    main()
