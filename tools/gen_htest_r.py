"""R's base stats results on the sd.htest problem collection (tests/htest/problems.py;
docs/design/htest.md §7). Writes tests/htest/data/r.json, so the tests and the accuracy tables
compare with R on machines without R (as tools/gen_regression_r.py does for sd.regression).

R's base `stats` package (decision H15), and the `car` package's leveneTest for levene (decision
H33; car is the regression area's comparison package, installed for G7a): t.test,
oneway.test, bartlett.test, fligner.test, chisq.test, shapiro.test, ks.test, wilcox.test,
kruskal.test, friedman.test, ansari.test, mood.test, cor.test (pearson, kendall, spearman), fisher.test, mcnemar.test,
binom.test, prop.test, poisson.test, TukeyHSD, p.adjust, pchisq, car::leveneTest, and multcomp's
Dunnett contrasts (glht with mcp(g = 'Dunnett'), installed; mvtnorm's seeded quasi-Monte Carlo) for
dunnett. A problem is run where R has
the same test with the same options; the others (Yuen's trim, alexandergovern, the Cressie-Read family,
cramervonmises, anderson, brunnermunzel, median_test, page_trend_test, epps_singleton, batches,
...) are left out. Where R's definition differs from scipy's (ks.test's exact distribution with
ties, which is sd's: H55, so ks.test is called with exact = TRUE wherever sd's ks_2samp is
exact; wilcox.test's exact switch at n < 50 and its zeros ranked as Pratt's, fisher.test's
conditional MLE odds ratio (odds_ratio's conditional problems take fisher.test's estimate and
interval, both found by R's uniroot to its default tolerance), mcnemar.test's continuity correction
(none when b = c; statsmodels' (|b - c| - 1)^2 / (b + c) still subtracts 1), prop.test's one-sample
variance (the null proportion's; statsmodels' z test takes the sample's), poisson.test's two rates (the
conditional binomial test, not the E-test of poisson_means_test)), the record keeps R's own value; the
comparison says which differ. wilcox.test's exact p-values with ties are conditional on the
midranks, as sd's method='exact' (H45): the H3 problems with ties and method='exact' compare
the two. cor.test's Kendall is called with exact as scipy's (and sd's) method (R's own default is
exact below n = 50), its Spearman with exact = FALSE (the t approximation, sd's and scipy's: H63)
and its default (AS 89's exact distribution) recorded beside it as p.value.default.

Each record: the R call (its data vectors shown as c(...)), the inputs' digest, and R's htest fields: statistic,
parameter (df), p.value, conf.int, estimate; for TukeyHSD the matrix of diff, lwr, upr, p adj;
for p.adjust the adjusted vector; for multcomp's Dunnett the t statistics, the adjusted p-values and the
interval ends (lwr, upr). Doubles go to R as %.17g (exact) and come back as %.17g.

Also writes tests/htest/data/r_gof.json: the goodness-of-fit packages nortest and goftest (decision
H44; GOF_PRELUDE below).

Also writes r_coin.json beside r.json: the `coin` package's conditional exact tests (decision H54;
COIN_PRELUDE below) on the H3 problems with ties or zeros, where base R has none: ansari_test,
wilcox_test, wilcoxsign_test (zeros 'Pratt' and 'Wilcoxon') with distribution = "exact", and Page's
test as friedman_test of an ordered treatment (each distribution coin takes; one that it refuses is
recorded as its error). One-sided p-values only ("less" and "greater"): coin's two-sided p-value is
P(|T - mu| >= |t - mu|), another rule than scipy's 2 min(...), and its direction is that of its own
standardized statistic. Without coin the file is left as it is.

Run: .venv/bin/python tools/gen_htest_r.py      (needs Rscript on PATH, and nortest, goftest and coin)
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
OUT_COIN = ROOT / "tests" / "htest" / "data" / "r_coin.json"  # coin (H54)

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


def _r_num(v):
    if v != v:
        return "NaN"
    if np.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return f"{v:.17g}"


def vec(a):
    a = np.ravel(np.asarray(a, dtype=np.float64))
    return "c(" + ", ".join(_r_num(v) for v in a) + ")"


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
        # exact wherever sd's method is (H55: R's psmirnov with ties is the conditional distribution
        # sd's 'exact' is); ks.test's own default would be exact only for n1 n2 < 10^4
        m = kw.get("method", "auto")
        if np.size(a[0]) == 0 or np.size(a[1]) == 0:
            return None
        e = "TRUE" if m == "exact" or (m == "auto" and max(np.size(a[0]), np.size(a[1])) <= 10000) else "FALSE"
        return f"ks.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}', exact = {e})"
    if f == "mannwhitneyu":
        c = "TRUE" if kw.get("use_continuity", True) else "FALSE"
        e = exact(kw.get("method", "auto"))
        return f"wilcox.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}', correct = {c}, exact = {e})"
    if f == "ranksums":
        return f"wilcox.test({vec(a[0])}, {vec(a[1])}, alternative = '{alt(p)}', correct = FALSE, exact = FALSE)"
    # R 4.6's wilcox.test ranks the zeros with the other differences and leaves them out of V
    # (Pratt's): its exact p-value is zero_method='pratt''s conditional one (H45)
    if f == "wilcoxon" and (
        kw.get("zero_method", "wilcox") == "wilcox" or (kw["zero_method"] == "pratt" and kw.get("method") == "exact")
    ):
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
    if f in ("spearmanrho", "spearmanr") and np.ndim(a[0]) == 1:
        # exact = FALSE: the t approximation, scipy's and sd's (H63: no exact distribution); R's
        # default (AS 89's exact distribution for n < 1290 without ties) as p.value.default
        xy = f"{vec(a[0])}, {vec(a[1])}, method = 'spearman', alternative = '{alt(p)}'"
        return (
            f"local({{h <- cor.test({xy}, exact = FALSE); d <- cor.test({xy}); "
            "list(statistic = h$statistic, p.value = h$p.value, estimate = h$estimate, p.value.default = d$p.value)})"
        )
    if f == "kendalltau" and kw.get("variant", "b") == "b":
        # exact where scipy's (and sd's) method is: R's own default is exact up to n = 49
        m = kw.get("method", "auto")
        if m == "auto":
            x, y = np.asarray(a[0]), np.asarray(a[1])
            ties = len(np.unique(x)) < x.size or len(np.unique(y)) < y.size
            dis = int(np.sum(np.triu(np.sign(x[:, None] - x[None, :]) * np.sign(y[:, None] - y[None, :]) < 0, 1)))
            tot = x.size * (x.size - 1) // 2
            m = "exact" if not ties and (x.size <= 33 or min(dis, tot - dis) <= 1) else "asymptotic"
        e = exact(m)
        return f"cor.test({vec(a[0])}, {vec(a[1])}, method = 'kendall', alternative = '{alt(p)}', exact = {e})"
    if f == "chi2_contingency" and "lambda_" not in kw:
        c = "TRUE" if kw.get("correction", True) else "FALSE"
        return f"chisq.test({mat(a[0])}, correct = {c})"
    if f == "fisher_exact":
        return f"fisher.test({mat(a[0])}, alternative = '{alt(p)}')"
    if f == "odds_ratio" and kw.get("kind", "conditional") == "conditional":
        # fisher.test's estimate and interval are the conditional MLE and the exact interval
        return f"fisher.test({mat(a[0])})"
    if f == "mcnemar":
        t = np.asarray(a[0])
        if kw.get("exact", True):
            return f"binom.test({float(t[0, 1])!r}, {float(t[0, 1] + t[1, 0])!r}, 0.5)"
        c = "TRUE" if kw.get("correction", True) else "FALSE"
        return f"mcnemar.test({mat(t)}, correct = {c})"
    if f == "binomtest":
        return f"binom.test({a[0]}, {a[1]}, p = {kw.get('p', 0.5)!r}, alternative = '{alt(p)}')"
    if f == "proportions_ztest" and np.ndim(a[0]) == 1 and "value" not in kw:
        ra = {"two-sided": "two.sided", "larger": "greater", "smaller": "less"}[kw.get("alternative", "two-sided")]
        return f"prop.test({vec(a[0])}, {vec(a[1])}, alternative = '{ra}', correct = FALSE)"
    if f == "proportions_chisquare":
        k = np.size(a[0])
        if "value" in kw:
            nobs = vec(np.broadcast_to(np.asarray(a[1], dtype=float), (k,)))
            return f"prop.test({vec(a[0])}, {nobs}, p = rep({kw['value']!r}, {k}), correct = FALSE)"
        if k < 2:
            return None
        return f"prop.test({vec(a[0])}, {vec(a[1])}, correct = FALSE)"
    if f == "proportion_confint" and kw.get("method") in ("beta", "wilson", "wilsoncc"):
        # statsmodels' one-sided 'larger' interval is [0, upper]: R's alternative = 'less' (H95)
        ra = {"two-sided": "two.sided", "larger": "less", "smaller": "greater"}[kw.get("alternative", "two-sided")]
        cl = 1 - kw.get("alpha", 0.05)
        if kw["method"] == "beta":
            return f"binom.test({a[0]}, {a[1]}, alternative = '{ra}', conf.level = {cl!r})"
        cc = "TRUE" if kw["method"] == "wilsoncc" else "FALSE"
        return f"prop.test({a[0]}, {a[1]}, alternative = '{ra}', conf.level = {cl!r}, correct = {cc})"
    if f == "poisson_means_test":
        # R's poisson.test of two rates is another test (the conditional binomial), recorded beside
        ra = {"two-sided": "two.sided", "less": "less", "greater": "greater"}[kw.get("alternative", "two-sided")]
        return (
            f"local({{h <- poisson.test(c({a[0]}, {a[2]}), c({a[1]!r}, {a[3]!r}), alternative = '{ra}'); "
            "list(conditional.p.value = h$p.value, conditional.estimate = h$estimate)})"
        )
    if f == "tukey_hsd" and kw.get("equal_var", True):
        g = "factor(rep(seq_along(L), lengths(L)))"
        return f"local({{L <- {groups(a)}; T <- TukeyHSD(aov(unlist(L) ~ {g}), conf.level = 0.95)[[1]]; as.list(as.data.frame(T))}})"  # noqa: E501
    if f == "dunnett":
        # multcomp's single-step Dunnett contrasts (the treatments minus the control, the control the
        # factor's first level): mvtnorm's pmvt and qmvt (Genz and Bretz's randomized quasi-Monte
        # Carlo, seeded, abseps 1e-7)
        ra = {"two-sided": "two.sided", "less": "less", "greater": "greater"}[kw.get("alternative", "two-sided")]
        L = groups([kw["control"], *a])
        alg = "mvtnorm::GenzBretz(maxpts = 2e6, abseps = 1e-7)"
        return (
            f"local({{set.seed(1); L <- {L}; g <- factor(rep(seq_along(L), lengths(L))); fit <- aov(unlist(L) ~ g); "
            f"h <- multcomp::glht(fit, linfct = multcomp::mcp(g = 'Dunnett'), alternative = '{ra}'); "
            f"s <- summary(h, test = multcomp::adjusted('single-step', algorithm = {alg})); "
            "ci <- confint(h, level = 0.95); "
            "list(statistic = s$test$tstat, p.value = as.numeric(s$test$pvalues), lwr = ci$confint[, 'lwr'], "
            "upr = ci$confint[, 'upr'])})"
        )
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


# coin (H54): one problem's two samples as a data frame of y and the group g (x first), or the
# differences' pair; each test with each alternative, the p-value by coin::pvalue
COIN_PRELUDE = """
coin_two <- function(test, x, y, ...) {
  d <- data.frame(y = c(x, y), g = factor(rep(c("x", "y"), c(length(x), length(y))), levels = c("x", "y")))
  out <- list()
  for (alt in c("less", "greater")) {
    r <- tryCatch(test(y ~ g, data = d, alternative = alt, ...), error = function(e) e)
    out[[paste0(alt, ".p.value")]] <- if (inherits(r, "error")) NA else coin::pvalue(r)
  }
  out
}
coin_sign <- function(x, y, zm) {
  out <- list()
  for (alt in c("less", "greater")) {
    r <- tryCatch(coin::wilcoxsign_test(x ~ y, alternative = alt, distribution = "exact", zero.method = zm),
                  error = function(e) e)
    out[[paste0(alt, ".p.value")]] <- if (inherits(r, "error")) NA else coin::pvalue(r)
  }
  out
}
coin_page <- function(m) {
  d <- data.frame(y = as.vector(t(m)), trt = factor(rep(seq_len(ncol(m)), nrow(m)), ordered = TRUE),
                  blk = factor(rep(seq_len(nrow(m)), each = ncol(m))))
  out <- list()
  for (dist in c("exact", "asymptotic")) {
    r <- tryCatch(coin::friedman_test(y ~ trt | blk, data = d, distribution = dist, alternative = "greater"),
                  error = function(e) e)
    out[[paste0(dist, ".p.value")]] <- if (inherits(r, "error")) NA else coin::pvalue(r)
  }
  out
}
"""
COIN = {
    "mannwhitneyu": ("ties-exact", "ties-exact-greater", "exact-5-7", "exact-forced"),
    "ansari": ("ties-exact", "exact", "exact-odd"),
    "wilcoxon": ("ties-exact", "zeros-exact-pratt", "zeros-auto-small", "exact-10", "paired"),
    "page_trend_test": ("ties", "exact", "exact-20x3"),
}


def r_coin_call(p):
    """coin's call for problem p (COIN), or None."""
    func, case = p.id.split("/", 1)
    if case not in COIN.get(func, ()):
        return None
    a = p.args
    if func == "mannwhitneyu":
        return f"coin_two(coin::wilcox_test, {vec(a[0])}, {vec(a[1])}, distribution = 'exact')"
    if func == "ansari":
        return f"coin_two(coin::ansari_test, {vec(a[0])}, {vec(a[1])}, distribution = 'exact')"
    if func == "wilcoxon":
        y = vec(a[1]) if len(a) > 1 else f"rep(0, {np.size(a[0])})"
        zm = "Pratt" if p.kwargs.get("zero_method", "wilcox") == "pratt" else "Wilcoxon"
        return f"coin_sign({vec(a[0])}, {y}, '{zm}')"
    return f"coin_page({mat(a[0])})"


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
    has_coin = subprocess.run(
        ["Rscript", "--vanilla", "-e", 'cat(requireNamespace("coin", quietly = TRUE))'], capture_output=True, text=True
    ).stdout.strip()
    if has_coin != "TRUE":
        print("coin is not installed: its record (r_coin.json) is left as it is (H54)")
        return
    calls = {pid: r_coin_call(p) for pid, p in P.PROBLEMS.items()}
    calls = {k: v for k, v in calls.items() if v is not None}
    recs, _ = run_r(PRELUDE + COIN_PRELUDE, calls, P)
    pkgs = subprocess.run(
        ["Rscript", "--vanilla", "-e", 'cat(paste("coin", packageVersion("coin")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    write(OUT_COIN, {"tool": "tools/gen_htest_r.py", "R": version, "packages": pkgs}, recs)
    print(f"{len(recs)} problems run with coin ({pkgs})")


if __name__ == "__main__":
    main()
