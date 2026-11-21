"""sd.htest against scipy, statsmodels and R: the public tests (milestones H1-H7; docs/design/htest.md §5) at
n = 10, 10^3 and 10^6 per sample and a batch of 10^4 lanes of 30 (axis=-1); page_trend_test on
tables (m, n) of (12, 8) (exact), (1000, 30) (normal) and a batch 10^3 x (10, 6).

The samples are normal values from fixed seeds (the second shifted by 0.1, the third by -0.05
with spread 1.2; counts for chisquare and power_divergence, positive values for the exponential
families), made in the setup as numpy arrays for scipy and R and as sd arrays for sd. scipy runs
at its defaults (one thread); R (base stats, car's leveneTest) in its one session, without the
batches (one call per test there) and where its test is the same: t.test, oneway.test,
bartlett.test, fligner.test, car::leveneTest, chisq.test, shapiro.test (n <= 5000), ks.test,
wilcox.test, kruskal.test, friedman.test, ansari.test, mood.test, ks.test (two samples) and
cor.test (pearson, spearman with exact = FALSE, kendall up to n = 10^3). anderson_ksamp (a list
of samples, no axis), spearmanr (its axis is the variables') and somersd (no axis) have no batch;
somersd takes rounded values (a table of some 30 x 30 distinct values).
The contingency tables (H6) on fixed tables (2 x 2 small and N = 2 10^4, 5 x 5 N = 10^4, r x c for
the exact Fisher test, (1000, 4) for Cochran's Q) and a batch of 10^4 2 x 2 tables; scipy,
statsmodels (mcnemar, cochrans_q) and R (chisq.test, fisher.test, mcnemar.test) where they have the
test (scipy's r x c Fisher test is a Monte Carlo: none).
The proportions and the binomial (H7) on fixed counts (n=COUNT their total) and batches of 10^4
counts, against scipy, statsmodels and R (binom.test, prop.test, poisson.test: another test).
R's ks.test is exact only below n = 100 and its exact p-value is 1 - cdf (differs, noted);
scipy's anderson fits the logistic to fsolve's xtol 1e-5 (a tolerance, not a difference worth a
note: the statistics agree to 1e-6).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..registry import Case, Check, Ref, RRef, count

SIZES = ((10, "10"), (1000, "10^3"), (10**6, "10^6"))
BATCH = (10**4, 30)
LOCS = ((0.0, 1.0), (0.1, 1.0), (-0.05, 1.2))


def _pair(r):
    """statistic and p-value, raveled (a batch's lanes in order)."""
    return np.concatenate([np.ravel(np.asarray(r.statistic, dtype=float)), np.ravel(np.asarray(r.pvalue, dtype=float))])


def _ref_pair(label, r):
    if label.startswith("R"):
        return np.asarray(r, dtype=float)
    return _pair(r)


CHECK = Check(sd=_pair, ref=_ref_pair)
R_PAIR = "c(.v$statistic, .v$p.value)"


def _data(kind, k, n, batch=False, seed=2026):
    import stardust as sd

    g = np.random.Generator(np.random.PCG64(seed))
    shape = BATCH if batch else (n,)
    xs = []
    for i in range(k):
        m, s = LOCS[i]
        if kind == "counts":
            x = np.floor(g.uniform(20, 40, shape))
        elif kind == "ints":
            x = np.round(g.normal(m, s, shape) * 3.0)
        elif kind == "positive":
            x = g.exponential(1.0 + 0.1 * i, shape)
        else:
            x = g.normal(m, s, shape)
        xs.append(x)
    d = {f"x{i}": x for i, x in enumerate(xs)}
    d.update({f"s{i}": sd.asarray(x) for i, x in enumerate(xs)})
    return d


def _quiet(f):
    def g():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f()

    return g


# (group, name, k samples, data kind, sd call, scipy call, R call or None, R differs, options)
def _specs():
    import scipy.stats as st

    import stardust as sd

    h = sd.htest
    N = sd.stats.Normal()
    R2 = "x0, x1"
    return [
        ("means", "ttest_1samp", 1, "normal", lambda *a, **k: h.ttest_1samp(a[0], 0.0, **k),
         lambda *a, **k: st.ttest_1samp(a[0], 0.0, **k), "t.test(x0, mu = 0)", "", {}),
        ("means", "ttest_rel", 2, "normal", h.ttest_rel, st.ttest_rel, f"t.test({R2}, paired = TRUE)", "", {}),
        ("means", "ttest_ind", 2, "normal", h.ttest_ind, st.ttest_ind, f"t.test({R2}, var.equal = TRUE)", "", {}),
        ("means", "ttest_ind Welch", 2, "normal", h.ttest_ind, st.ttest_ind, f"t.test({R2})", "",
         {"equal_var": False}),
        ("means", "ttest_ind trim=0.2", 2, "normal", h.ttest_ind, st.ttest_ind, None, "", {"trim": 0.2}),
        ("means", "f_oneway (3)", 3, "normal", h.f_oneway, st.f_oneway,
         "oneway.test(c(x0, x1, x2) ~ factor(rep(1:3, c(length(x0), length(x1), length(x2)))), var.equal = TRUE)",
         "", {}),
        ("means", "f_oneway (3) Welch", 3, "normal", h.f_oneway, st.f_oneway,
         "oneway.test(c(x0, x1, x2) ~ factor(rep(1:3, c(length(x0), length(x1), length(x2)))))", "",
         {"equal_var": False}),
        ("means", "alexandergovern (3)", 3, "normal", h.alexandergovern, st.alexandergovern, None, "", {}),
        ("variances", "bartlett (3)", 3, "normal", h.bartlett, st.bartlett, "bartlett.test(list(x0, x1, x2))", "",
         {}),
        ("variances", "levene (3)", 3, "normal", h.levene, st.levene,
         "car::leveneTest(c(x0, x1, x2), factor(rep(1:3, c(length(x0), length(x1), length(x2)))))", "", {}),
        ("variances", "fligner (3)", 3, "normal", h.fligner, st.fligner, "fligner.test(list(x0, x1, x2))", FLIGNER, {}),
        ("gof", "chisquare", 1, "counts", h.chisquare, st.chisquare, "chisq.test(x0)", "", {}),
        ("gof", "power_divergence G", 1, "counts", h.power_divergence, st.power_divergence, None, "",
         {"lambda_": "log-likelihood"}),
        ("gof", "power_divergence cressie-read", 1, "counts", h.power_divergence, st.power_divergence, None, "",
         {"lambda_": "cressie-read"}),
        ("gof", "skewtest", 1, "normal", h.skewtest, st.skewtest, None, "", {}),
        ("gof", "kurtosistest", 1, "normal", h.kurtosistest, st.kurtosistest, None, "", {}),
        ("gof", "normaltest", 1, "normal", h.normaltest, st.normaltest, None, "", {}),
        ("gof", "jarque_bera", 1, "normal", h.jarque_bera, st.jarque_bera, None, "", {}),
        ("gof", "shapiro", 1, "normal", h.shapiro, st.shapiro, "shapiro.test(x0)", "", {}),
        ("gof", "anderson", 1, "normal", h.anderson, lambda *a, **k: st.anderson(*a, method="interpolate", **k),
         None, "", {}),
        ("gof", "anderson logistic", 1, "normal", lambda *a, **k: h.anderson(*a, dist=sd.stats.Logistic, **k),
         lambda *a, **k: st.anderson(*a, dist="logistic", method="interpolate", **k), None, "", {}),
        ("gof", "ks_1samp", 1, "normal", lambda *a, **k: h.ks_1samp(a[0], N, **k),
         lambda *a, **k: st.ks_1samp(a[0], st.norm.cdf, **k), "ks.test(x0, 'pnorm')",
         "ks.test's exact p-value is 1 - cdf and asymptotic from n = 100", {}),
        ("gof", "cramervonmises", 1, "normal", lambda *a, **k: h.cramervonmises(a[0], N, **k),
         lambda *a, **k: st.cramervonmises(a[0], "norm", **k), None, "", {}),
        ("rank", "mannwhitneyu", 2, "normal", h.mannwhitneyu, st.mannwhitneyu, f"wilcox.test({R2})", "", {}),
        ("rank", "ranksums", 2, "normal", h.ranksums, st.ranksums, None, "", {}),
        ("rank", "wilcoxon", 2, "normal", h.wilcoxon, st.wilcoxon, f"wilcox.test({R2}, paired = TRUE)", WILCOXON, {}),
        ("rank", "kruskal (3)", 3, "normal", h.kruskal, st.kruskal, "kruskal.test(list(x0, x1, x2))", "", {}),
        ("rank", "friedmanchisquare (3)", 3, "normal", h.friedmanchisquare, st.friedmanchisquare,
         "friedman.test(cbind(x0, x1, x2))", "", {}),
        ("rank", "brunnermunzel", 2, "normal", h.brunnermunzel, st.brunnermunzel, None, "", {}),
        ("rank", "ansari", 2, "normal", h.ansari, st.ansari, f"ansari.test({R2})", ANSARI, {}),
        ("rank", "mood", 2, "normal", h.mood, st.mood, f"mood.test({R2})", "", {}),
        ("rank", "median_test (3)", 3, "normal", h.median_test, st.median_test, None, "", {}),
        ("samples", "ks_2samp", 2, "normal", h.ks_2samp, st.ks_2samp, f"ks.test({R2})", KS2, {}),
        ("samples", "cramervonmises_2samp", 2, "normal", h.cramervonmises_2samp, st.cramervonmises_2samp, None, "",
         {}),
        ("samples", "anderson_ksamp (3)", 3, "normal", lambda *a, **k: h.anderson_ksamp(list(a), **k),
         lambda *a, **k: st.anderson_ksamp(list(a), variant="midrank", **k), None, "", {}),
        ("samples", "epps_singleton_2samp", 2, "normal", h.epps_singleton_2samp, st.epps_singleton_2samp, None, "",
         {}),
        ("corr", "pearsonr", 2, "normal", h.pearsonr, st.pearsonr, f"cor.test({R2})", "", {}),
        ("corr", "spearmanrho", 2, "normal", h.spearmanrho, st.spearmanrho,
         f"cor.test({R2}, method = 'spearman', exact = FALSE)", "", {}),
        ("corr", "spearmanr", 2, "normal", h.spearmanr, st.spearmanr, None, "", {}),
        ("corr", "pointbiserialr", 2, "normal", h.pointbiserialr, st.pointbiserialr, None, "", {}),
        ("corr", "kendalltau", 2, "normal", h.kendalltau, st.kendalltau, f"cor.test({R2}, method = 'kendall')",
         KENDALL, {}),
        ("corr", "weightedtau", 2, "normal", h.weightedtau, st.weightedtau, None, "", {}),
        ("corr", "somersd", 2, "ints", h.somersd, st.somersd, None, "", {}),
        ("corr", "chatterjeexi", 2, "normal", h.chatterjeexi, st.chatterjeexi, None, "", {}),
    ]  # fmt: skip


# cor.test's Kendall statistic counts the pairs one by one (O(n^2): no 10^6 case), and its exact
# p-value (n < 50, scipy's and sd's up to 33) is 1 - cdf
KENDALL = "cor.test's Kendall is O(n^2) and exact below n = 50 (1 - cdf)"


# R's wilcox.test(paired = TRUE) reports V = T+ (scipy's statistic is min(T+, T-)); ansari.test's
# two-sided exact p-value doubles the tail on the side of the distribution's centre
WILCOXON = "R's V is T+, scipy's statistic min(T+, T-)"
ANSARI = "R's two-sided p-value doubles the tail on the side of its centre, scipy's the smaller tail"


# ks.test is exact (its psmirnov) only for n1 n2 < 10^4 by default and asymptotic (Kolmogorov's
# limit) above, where scipy's 'auto' is exact up to max(n1, n2) = 10^4 and then kstwo at round(n)
KS2 = "ks.test is exact only for n1 n2 < 10^4 and its asymptotic p-value is Kolmogorov's limit"


# fligner's centre: scipy and R round the median of an even group to a double, which breaks the
# tie of the two middle values' deviations (docs/design/htest.md §3.2): another rank sum
FLIGNER = "the median rounded to a double breaks the tie of an even group's middle deviations"


def _no_batch_in_scipy(name):
    # scipy's anderson and median_test take no axis (one sample a call)
    return name.startswith("anderson") or name.startswith("median_test")


def _no_batch(name):
    # anderson_ksamp takes a list of samples, no axis (scipy's and sd's); spearmanr's axis is that
    # of its variables and somersd has none
    return name.startswith("anderson_ksamp") or name in ("spearmanr", "somersd")


def cases() -> list[Case]:
    out = []
    for group, name, k, kind, sdf, spf, rexpr, rdiff, kw in _specs():
        for n, size in SIZES:
            refs = [
                Ref(
                    "scipy",
                    lambda d, spf=spf, k=k, kw=kw: _quiet(lambda: spf(*(d[f"x{i}"] for i in range(k)), **kw)),
                    differs=FLIGNER if rdiff == FLIGNER else "",
                )
            ]
            if rexpr and not (name == "shapiro" and n > 5000) and not (name == "kendalltau" and n > 1000):
                data = {f"x{i}": f"x{i}" for i in range(k)}
                pk = ("car",) if "car::" in rexpr else ()
                refs.append(RRef("R", rexpr, data=data, packages=pk, differs=rdiff, value=R_PAIR if "car::" not in rexpr
                                 else "c(.v[1, 'F value'], .v[1, 'Pr(>F)'])"))  # fmt: skip
            out.append(
                Case(
                    area="htest",
                    group=group,
                    name=name,
                    size=size,
                    n=n * k,
                    setup=lambda kind=kind, k=k, n=n: _data(kind, k, n),
                    sd=lambda d, sdf=sdf, k=k, kw=kw: lambda: sdf(*(d[f"s{i}"] for i in range(k)), **kw),
                    refs=tuple(refs),
                    quick=n == 1000,
                    check=CHECK,
                )
            )
        if _no_batch(name):
            continue
        refs = ()
        if not _no_batch_in_scipy(name):
            refs = (
                Ref(
                    "scipy",
                    lambda d, spf=spf, k=k, kw=kw: _quiet(lambda: spf(*(d[f"x{i}"] for i in range(k)), axis=-1, **kw)),
                    differs=FLIGNER if rdiff == FLIGNER else "",
                ),
            )
        out.append(
            Case(
                area="htest",
                group=f"{group}-batched",
                name=name,
                size=f"({BATCH[0]}, {BATCH[1]})",
                n=BATCH[0] * BATCH[1] * k,
                setup=lambda kind=kind, k=k: _data(kind, k, 0, batch=True),
                sd=lambda d, sdf=sdf, k=k, kw=kw: lambda: sdf(*(d[f"s{i}"] for i in range(k)), axis=-1, **kw),
                refs=refs,
                check=CHECK,
            )
        )
    out.extend(_page_cases())
    out.extend(_table_cases())
    out.extend(_proportion_cases())
    return out


# ---- the contingency tables (H6) ----------------------------------------------------------

TABLES = {
    "2x2": np.array([[12.0, 5], [7, 16]]),
    "2x2 N=2*10^4": np.array([[5200.0, 4800], [4700, 5300]]),
    "5x5 N=10^4": np.array([[420.0, 380, 410, 395, 405], [390, 415, 400, 405, 390], [405, 400, 385, 410, 400],
                            [395, 405, 410, 390, 400], [390, 400, 395, 400, 415]]),
    "3x3": np.array([[3.0, 1, 4], [1, 5, 9], [2, 6, 5]]),
    "4x5 N=100": np.array([[1.0, 2, 5, 20, 2], [9, 0, 1, 10, 9], [0, 0, 10, 0, 1], [3, 8, 5, 5, 9]]),
    "2x2 N=200": np.array([[60.0, 45], [40, 55]]),
}  # fmt: skip


def _table_data(key, batch=False, seed=2026):
    import stardust as sd

    if batch:
        g = np.random.Generator(np.random.PCG64(seed))
        x = np.floor(g.uniform(1, 30, (10**4, 2, 2)))
    elif key == "cochran":
        g = np.random.Generator(np.random.PCG64(seed))
        x = (g.uniform(size=(1000, 4)) < np.array([0.3, 0.4, 0.5, 0.6])).astype(np.float64)
    else:
        x = TABLES[key]
    return {"x0": x, "s0": sd.asarray(x), "xv": np.ravel(x.T), "xi": x.astype(np.int64)}


def _first(r):
    return np.ravel(np.asarray(r.statistic if hasattr(r, "statistic") else r.relative_risk, dtype=float))


TABLE_CHECK = Check(sd=_first, ref=lambda label, r: np.asarray(r, dtype=float) if label.startswith("R") else _first(r))


def _table_size(key):
    """The id notation of a table key (bench/README.md): "(r, c)" for the small tables, "n=N" for
    the ones named by their total count ("5x5 N=10^4" -> "n=10^4")."""
    shape, _, total = key.partition(" N=")
    if total:
        n = 1
        for factor in total.split("*"):  # "2*10^4" -> 20000
            base, _, exp = factor.partition("^")
            n *= int(base) ** int(exp or 1)
        return "n=" + count(n)
    r, c = shape.split("x")
    return f"({r}, {c})"


def _table_cases():
    import scipy.stats as st
    from scipy.stats import contingency as ctg
    from statsmodels.stats import contingency_tables as smct

    import stardust as sd

    h = sd.htest
    specs = [  # (name, table key, sd call, comparators [(label, call)], R expression or None, R differs)
        ("chi2_contingency", "2x2", h.chi2_contingency, st.chi2_contingency, "chisq.test(matrix(xv, nrow = 2))", ""),
        ("chi2_contingency", "5x5 N=10^4", h.chi2_contingency, st.chi2_contingency,
         "chisq.test(matrix(xv, nrow = 5))", ""),
        ("fisher_exact", "2x2", h.fisher_exact, st.fisher_exact, "fisher.test(matrix(xv, nrow = 2))", ""),
        ("fisher_exact", "2x2 N=2*10^4", h.fisher_exact, st.fisher_exact, "fisher.test(matrix(xv, nrow = 2))", ""),
        ("fisher_exact", "3x3", h.fisher_exact, None, "fisher.test(matrix(xv, nrow = 3))", ""),
        ("fisher_exact", "4x5 N=100", h.fisher_exact, None, "fisher.test(matrix(xv, nrow = 4), workspace = 2e8)",
         "R needs workspace = 2e8 (its default fails) and merges past values (1.7e-10 off)"),
        ("barnard_exact", "2x2", h.barnard_exact, st.barnard_exact, None, ""),
        ("barnard_exact", "2x2 N=200", h.barnard_exact, st.barnard_exact, None, ""),
        ("boschloo_exact", "2x2", h.boschloo_exact, st.boschloo_exact, None, ""),
        ("boschloo_exact", "2x2 N=200", h.boschloo_exact, st.boschloo_exact, None, ""),
        ("mcnemar", "2x2", h.mcnemar, lambda x: smct.mcnemar(x), None, ""),
        ("mcnemar", "2x2 N=2*10^4", lambda x: h.mcnemar(x, exact=False), lambda x: smct.mcnemar(x, exact=False),
         "mcnemar.test(matrix(xv, nrow = 2))", ""),
        ("cochrans_q", "cochran", h.cochrans_q, lambda x: smct.cochrans_q(x), None, ""),
        ("odds_ratio", "2x2", lambda x: h.odds_ratio(x).conf_int(), None, None, ""),
        ("odds_ratio", "2x2 N=200", h.odds_ratio, None, None, ""),
    ]  # fmt: skip
    out = []
    for name, key, sdf, spf, rexpr, rdiff in specs:
        refs = []
        if spf is not None:
            lib = "statsmodels" if name in ("mcnemar", "cochrans_q") else "scipy"
            refs.append(Ref(lib, lambda d, spf=spf: _quiet(lambda: spf(d["x0"]))))
        if name == "odds_ratio":
            refs.append(Ref("scipy", lambda d: _quiet(lambda: ctg.odds_ratio(d["xi"]).confidence_interval())))
        if rexpr:
            val = ".v$p.value" if name == "fisher_exact" else "c(.v$statistic)"
            refs.append(RRef("R", rexpr, data={"xv": "xv"}, differs=rdiff, value=val))
        check = CHECK if name not in ("odds_ratio",) else Check(skip="the interval, not a statistic")
        if name == "fisher_exact" and rexpr:
            check = Check(sd=lambda r: np.ravel(np.asarray(r.pvalue, dtype=float)),
                          ref=lambda label, r: np.asarray(r, dtype=float) if label.startswith("R")
                          else np.ravel(np.asarray(r.pvalue, dtype=float)))  # fmt: skip
        elif name in ("chi2_contingency", "mcnemar") and rexpr:
            check = TABLE_CHECK
        out.append(
            Case(
                area="htest",
                group="tables",
                name=name,
                size="(1000, 4)" if key == "cochran" else _table_size(key),
                n=int(TABLES[key].sum()) if key in TABLES else 4000,
                setup=lambda key=key: _table_data(key),
                sd=lambda d, sdf=sdf: lambda: sdf(d["s0"]),
                refs=tuple(refs),
                quick=key == "2x2",
                check=check,
            )
        )
    for name in ("chi2_contingency", "fisher_exact", "barnard_exact", "mcnemar", "odds_ratio"):
        out.append(
            Case(
                area="htest",
                group="tables-batched",
                name=name,
                size="10^4 x (2, 2)",
                n=4 * 10**4,
                setup=lambda: _table_data("", batch=True),
                sd=lambda d, f=getattr(h, name): lambda: f(d["s0"]),
                refs=(),
                check=CHECK,
            )
        )
    return out


PAGE = (((12, 8), "(12, 8)"), ((1000, 30), "(1000, 30)"))


def _page_data(shape, seed=2026):
    import stardust as sd

    g = np.random.Generator(np.random.PCG64(seed))
    x = g.normal(size=shape) + np.arange(shape[-1]) * 0.05
    return {"x0": x, "s0": sd.asarray(x)}


def _page_cases():
    import scipy.stats as st

    import stardust as sd

    out = []
    for shape, size in PAGE:
        out.append(
            Case(
                area="htest",
                group="page",
                name="page_trend_test",
                size=size,
                n=shape[0] * shape[1],
                setup=lambda shape=shape: _page_data(shape),
                sd=lambda d: lambda: sd.htest.page_trend_test(d["s0"]),
                refs=(Ref("scipy", lambda d: _quiet(lambda: st.page_trend_test(d["x0"]))),),
                quick=shape[0] == 12,
                check=CHECK,
            )
        )
    out.append(
        Case(
            area="htest",
            group="page",
            name="page_trend_test",
            size="10^3 x (10, 6)",
            n=60_000,
            setup=lambda: _page_data((1000, 10, 6)),
            sd=lambda d: lambda: sd.htest.page_trend_test(d["s0"]),
            refs=(),
            check=CHECK,
        )
    )
    return out


TITLES = {
    "means": "평균",
    "means-batched": "평균 일괄",
    "variances": "분산의 동질성",
    "variances-batched": "분산의 동질성 일괄",
    "gof": "적합도",
    "gof-batched": "적합도 일괄",
    "rank": "순위",
    "rank-batched": "순위 일괄",
    "page": "Page의 추세",
    "samples": "표본끼리의 적합도",
    "samples-batched": "표본끼리의 적합도 일괄",
    "corr": "상관",
    "corr-batched": "상관 일괄",
    "tables": "분할표",
    "tables-batched": "분할표 일괄",
    "proportions": "비율과 이항",
    "proportions-batched": "비율과 이항 일괄",
}


# ---- the proportions and the binomial (H7) ------------------------------------------------


def _interval(r):
    """An interval as its two ends (sd's (2,) or (..., 2) array, scipy's ConfidenceInterval,
    statsmodels' (low, upp))."""
    if hasattr(r, "low"):
        return np.ravel([np.asarray(r.low, dtype=float), np.asarray(r.high, dtype=float)])
    a = np.asarray(r if not isinstance(r, tuple) else np.stack([np.asarray(v, dtype=float) for v in r], axis=-1),
                   dtype=float)  # fmt: skip
    return np.ravel(a)


INTERVAL_CHECK = Check(sd=_interval, ref=lambda label, r: np.asarray(r, dtype=float) if label.startswith("R")
                       else _interval(r))  # fmt: skip
SM_PAIR = Check(sd=_pair, ref=lambda label, r: np.asarray(r, dtype=float) if label.startswith("R")
                else (np.ravel(np.asarray(r[:2], dtype=float)) if isinstance(r, tuple) else _pair(r)))  # fmt: skip


def _proportion_data(seed=2026):
    import stardust as sd

    g = np.random.Generator(np.random.PCG64(seed))
    n = np.floor(g.uniform(20, 200, 10**4))
    c = np.floor(g.uniform(0, 1, 10**4) * (n + 1))
    n2 = np.floor(g.uniform(20, 200, 10**4))
    c2 = np.floor(g.uniform(0, 1, 10**4) * (n2 + 1))
    d = {"n": n, "c": c, "n2": n2, "c2": c2}
    d.update({f"s{k}": sd.asarray(v) for k, v in list(d.items())})
    for size in (10**3, 10**6):
        x = g.normal(0.0, 1.0, size)
        d[f"x{size}"], d[f"sx{size}"] = x, sd.asarray(x)
    return d


def _proportion_cases():
    import scipy.stats as st
    from statsmodels.stats import proportion as smp

    import stardust as sd

    h = sd.htest
    TWO = (15, 40, 25, 45)
    # (name, size, n, sd call, comparators [(label, call)], R expression or None, R differs, check)
    specs = [
        ("binomtest", "n=10", 10, lambda d: h.binomtest(3, 10), [("scipy", lambda d: st.binomtest(3, 10))],
         "binom.test(3, 10)", "", SM_PAIR),
        ("binomtest", "n=10^4", 10**4, lambda d: h.binomtest(5030, 10**4, p=0.49),
         [("scipy", lambda d: st.binomtest(5030, 10**4, p=0.49))], "binom.test(5030, 10000, p = 0.49)", "", SM_PAIR),
        ("binomtest conf_int", "n=10^4", 10**4, lambda d: h.binomtest(5030, 10**4).conf_int(),
         [("scipy", lambda d: st.binomtest(5030, 10**4).proportion_ci())], "binom.test(5030, 10000)$conf.int", "",
         INTERVAL_CHECK),
        ("quantile_test", "10^3", 10**3, lambda d: h.quantile_test(d["sx1000"], q=0.1),
         [("scipy", lambda d: st.quantile_test(d["x1000"], q=0.1))], None, "", SM_PAIR),
        ("quantile_test", "10^6", 10**6, lambda d: h.quantile_test(d["sx1000000"], q=0.1),
         [("scipy", lambda d: st.quantile_test(d["x1000000"], q=0.1))], None, "", SM_PAIR),
        ("poisson_means_test", "n=30", 30, lambda d: h.poisson_means_test(10, 5.0, 20, 6.0),
         [("scipy", lambda d: st.poisson_means_test(10, 5.0, 20, 6.0))], "poisson.test(c(10, 20), c(5, 6))",
         "poisson.test is the conditional binomial test of the two rates, not the E-test", SM_PAIR),
        ("poisson_means_test", "n=2300", 2300, lambda d: h.poisson_means_test(1200, 10.0, 1100, 10.0),
         [("scipy", lambda d: st.poisson_means_test(1200, 10.0, 1100, 10.0))], None, "", SM_PAIR),
        ("proportions_ztest", "n=85", 85, lambda d: h.proportions_ztest([15, 25], [40, 45]),
         [("statsmodels", lambda d: smp.proportions_ztest([15, 25], [40, 45]))], None, "", SM_PAIR),
        ("proportions_chisquare", "n=115", 115, lambda d: h.proportions_chisquare([15, 25, 9], [40, 45, 30]),
         [("statsmodels", lambda d: smp.proportions_chisquare([15, 25, 9], [40, 45, 30]))],
         "c(prop.test(c(15, 25, 9), c(40, 45, 30), correct = FALSE)$statistic, "
         "prop.test(c(15, 25, 9), c(40, 45, 30), correct = FALSE)$p.value)", "", SM_PAIR),
        ("test_proportions_2indep", "n=85", 85, lambda d: h.test_proportions_2indep(*TWO),
         [("statsmodels", lambda d: smp.test_proportions_2indep(*TWO))], None, "", SM_PAIR),
        ("test_proportions_2indep score", "n=85", 85,
         lambda d: h.test_proportions_2indep(*TWO, compare="odds-ratio", method="score", value=2.0),
         [("statsmodels", lambda d: smp.test_proportions_2indep(*TWO, compare="odds-ratio", method="score", value=2.0))],
         None, "", SM_PAIR),
        ("confint_proportions_2indep", "n=85", 85, lambda d: h.confint_proportions_2indep(*TWO),
         [("statsmodels", lambda d: smp.confint_proportions_2indep(*TWO))], None, "", INTERVAL_CHECK),
        ("confint_proportions_2indep score ratio", "n=85", 85,
         lambda d: h.confint_proportions_2indep(*TWO, compare="ratio", method="score"),
         [("statsmodels", lambda d: smp.confint_proportions_2indep(*TWO, compare="ratio", method="score"))], None, "",
         INTERVAL_CHECK),
        ("confint_proportions_2indep score odds-ratio", "n=85", 85,
         lambda d: h.confint_proportions_2indep(*TWO, compare="odds-ratio", method="score"),
         [("statsmodels", lambda d: smp.confint_proportions_2indep(*TWO, compare="odds-ratio", method="score"))], None,
         "", INTERVAL_CHECK),
    ]  # fmt: skip
    for meth, smm in (("normal", "normal"), ("agresti_coull", "agresti_coull"), ("exact", "beta"), ("wilson", "wilson"),
                      ("jeffreys", "jeffreys"), ("binom_test", "binom_test")):  # fmt: skip
        rexpr = {"exact": "binom.test(7, 30)$conf.int", "wilson": "prop.test(7, 30, correct = FALSE)$conf.int"}.get(
            meth
        )
        specs.append((f"proportion_confint {meth}", "n=30", 30, lambda d, m=meth: h.proportion_confint(7, 30, method=m),
                      [("statsmodels", lambda d, m=smm: smp.proportion_confint(7, 30, method=m))], rexpr, "",
                      INTERVAL_CHECK))  # fmt: skip
    out = []
    for name, size, n, sdf, comps, rexpr, rdiff, check in specs:
        refs = [Ref(label, lambda d, f=f: _quiet(lambda: f(d))) for label, f in comps]
        if rexpr:
            val = ".v" if "conf.int" in rexpr or rexpr.startswith("c(") else "c(.v$statistic, .v$p.value)"
            refs.append(RRef("R", rexpr, differs=rdiff, value=val))
        if name.startswith("binomtest") and rexpr and "conf.int" not in rexpr:
            check = Check(sd=lambda r: np.asarray([r.pvalue]),
                          ref=lambda label, r: np.asarray(r, dtype=float)[1:] if label.startswith("R")
                          else np.asarray([r.pvalue]))  # fmt: skip
        out.append(
            Case(
                area="htest",
                group="proportions",
                name=name,
                size=size,
                n=n,
                setup=_proportion_data,
                sd=lambda d, sdf=sdf: lambda: sdf(d),
                refs=tuple(refs),
                quick=size in ("n=10", "n=30", "n=85"),
                check=check,
            )
        )
    batched = [
        ("proportion_confint wilson", lambda d: h.proportion_confint(d["sc"], d["sn"], method="wilson"),
         lambda d: smp.proportion_confint(d["c"], d["n"], method="wilson")),
        ("proportion_confint exact", lambda d: h.proportion_confint(d["sc"], d["sn"], method="exact"),
         lambda d: smp.proportion_confint(d["c"], d["n"], method="beta")),
        ("test_proportions_2indep", lambda d: h.test_proportions_2indep(d["sc"], d["sn"], d["sc2"], d["sn2"]),
         lambda d: smp.test_proportions_2indep(d["c"], d["n"], d["c2"], d["n2"])),
        ("confint_proportions_2indep", lambda d: h.confint_proportions_2indep(d["sc"], d["sn"], d["sc2"], d["sn2"]),
         lambda d: smp.confint_proportions_2indep(d["c"], d["n"], d["c2"], d["n2"])),
        ("proportions_ztest", lambda d: h.proportions_ztest(sd.stack([d["sc"], d["sc2"]], axis=-1),
                                                            sd.stack([d["sn"], d["sn2"]], axis=-1)), None),
    ]  # fmt: skip
    for name, sdf, smf in batched:
        check = SM_PAIR if name.startswith(("test", "proportions_z")) else INTERVAL_CHECK
        refs = (Ref("statsmodels", lambda d, f=smf: _quiet(lambda: f(d))),) if smf else ()
        out.append(
            Case(
                area="htest",
                group="proportions-batched",
                name=name,
                size="10^4",
                n=10**4,
                setup=_proportion_data,
                sd=lambda d, sdf=sdf: lambda: sdf(d),
                refs=refs,
                check=check,
            )
        )
    return out
