"""sd.htest against scipy and R: the public tests (milestones H1-H4; docs/design/htest.md §5) at
n = 10, 10^3 and 10^6 per sample and a batch of 10^4 lanes of 30 (axis=-1); page_trend_test on
tables (m, n) of (12, 8) (exact), (1000, 30) (normal) and a batch 10^3 x (10, 6).

The samples are normal values from fixed seeds (the second shifted by 0.1, the third by -0.05
with spread 1.2; counts for chisquare and power_divergence, positive values for the exponential
families), made in the setup as numpy arrays for scipy and R and as sd arrays for sd. scipy runs
at its defaults (one thread); R (base stats, car's leveneTest) in its one session, without the
batches (one call per test there) and where its test is the same: t.test, oneway.test,
bartlett.test, fligner.test, car::leveneTest, chisq.test, shapiro.test (n <= 5000), ks.test,
wilcox.test, kruskal.test, friedman.test, ansari.test, mood.test and ks.test (two samples).
anderson_ksamp (a list of samples, no axis) has no batch.
R's ks.test is exact only below n = 100 and its exact p-value is 1 - cdf (differs, noted);
scipy's anderson fits the logistic to fsolve's xtol 1e-5 (a tolerance, not a difference worth a
note: the statistics agree to 1e-6).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..registry import Case, Check, Ref, RRef

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
    ]  # fmt: skip


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
    # anderson_ksamp takes a list of samples, no axis (scipy's and sd's)
    return name.startswith("anderson_ksamp")


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
            if rexpr and not (name == "shapiro" and n > 5000):
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
}
