"""Golden cases for sd.htest (tools/golden.py has the contract).

Every public test on fixed data sets (small samples, a large offset, ties, constant samples, a
sample of one, NaN under the three nan_policy values, empty samples), over its options
(alternative, equal_var, trim, center and proportiontocut, lambda_ and ddof, dist, method, variant, t,
zero_method, correction, use_continuity, distribution, ties, ranked, predicted_ranks, keepdims),
batches along an axis and a tuple of axes (the rank tests' per-lane methods), long lanes
(n = 10^5: the multithreaded kernels), the exact distributions (with ties, deep tails, past
their work cap), scales of 1e+-300, and the wrong calls (their errors are results). A result
object is stored whole: every field and conf_int(); conf_int(alpha=0.1) as its own case. The
contingency tables (H6) over their options too (correction, lambda_, alternative, pooled, n, exact,
kind, conf_int's alternative), weighted tables, zero margins, r x c tables (and one past the work
cap), batches of tables. The proportions and the binomial (H7) over their methods, sides, null
values and corrections, the binomial rule's edges, zero counts and batches.
"""

import math

import numpy as np

NAN, INF = math.nan, math.inf


def data(L):
    r = L.rng(401)
    d = {
        "a": r.normal(0.3, 1.0, 12),
        "b": r.normal(0.0, 1.5, 15),
        "c": r.normal(-0.2, 0.8, 9),
        "off_a": r.normal(1e8, 1.0, 40),
        "off_b": r.normal(1e8 + 0.5, 1.0, 41),
        "ties": np.round(r.normal(0.0, 2.0, 30)),
        "const": np.full(8, 2.5),
        "nan": np.array([0.4, NAN, -1.2, 0.9, 1.7, -0.3, 0.2, 2.2, -0.8]),
        "long_a": r.normal(0.0, 1.0, 100_000),
        "long_b": r.normal(0.01, 1.1, 100_000),
        "batch_a": r.normal(0.0, 1.0, (4, 20)),
        "batch_b": r.normal(0.2, 1.3, (4, 25)),
        "cube": r.normal(0.0, 1.0, (3, 10, 4)),
        "expo": r.exponential(1.0, 60),
        "logis": r.logistic(1.0, 2.0, 50),
        "gumb": r.gumbel(0.5, 1.5, 50),
        "counts": np.array([16.0, 18, 16, 14, 12, 12]),
        "zero_counts": np.array([12.0, 0, 9, 15, 4]),
        "near": np.array([1e6 + 1e-3, 1e6 - 1e-3, 1e6]),
        "count_batch": np.floor(r.uniform(5, 40, (4, 6))),
    }
    return d


def cases(L):
    sd, h = L.sd, L.sd.htest
    d = data(L)
    a, b, c = d["a"], d["b"], d["c"]
    N = sd.stats.Normal()

    # ---- H1: means and variances
    for alt in ("two-sided", "less", "greater"):
        yield f"ttest_1samp {alt}", lambda alt=alt: h.ttest_1samp(a, 0.1, alternative=alt)
        yield f"ttest_rel {alt}", lambda alt=alt: h.ttest_rel(a, b[:12], alternative=alt)
        for ev in (True, False):
            yield (
                f"ttest_ind equal_var={ev} {alt}",
                lambda ev=ev, alt=alt: h.ttest_ind(a, b, equal_var=ev, alternative=alt),
            )
    yield "ttest_1samp conf_int alpha=0.1", lambda: h.ttest_1samp(a, 0.0).conf_int(alpha=0.1)
    yield "ttest_ind conf_int alpha=0.1", lambda: h.ttest_ind(a, b, equal_var=False).conf_int(alpha=0.1)
    yield "ttest_ind trim=0.2", lambda: h.ttest_ind(a, b, trim=0.2)
    yield "ttest_ind trim=0.2 ties", lambda: h.ttest_ind(d["ties"], b, trim=0.2, equal_var=False)
    yield "ttest_1samp offset", lambda: h.ttest_1samp(d["off_a"], 1e8)
    yield "ttest_ind offset", lambda: h.ttest_ind(d["off_a"], d["off_b"])
    yield "ttest_rel offset", lambda: h.ttest_rel(d["off_a"], d["off_b"][:40])
    yield "ttest_1samp constant", lambda: h.ttest_1samp(d["const"], 1.0)
    yield "ttest_ind constant", lambda: h.ttest_ind(d["const"], d["const"] + 1.0)
    yield "ttest_ind one element", lambda: h.ttest_ind([1.5], b)
    yield "ttest_ind Welch one element", lambda: h.ttest_ind([1.5], b, equal_var=False)
    yield "ttest_1samp empty", lambda: h.ttest_1samp([], 0.0)
    for pol in ("propagate", "omit", "raise"):
        yield f"ttest_1samp nan {pol}", lambda pol=pol: h.ttest_1samp(d["nan"], 0.0, nan_policy=pol)
        yield f"ttest_rel nan {pol}", lambda pol=pol: h.ttest_rel(d["nan"], a[:9], nan_policy=pol)
    yield "ttest_ind batch", lambda: h.ttest_ind(d["batch_a"], d["batch_b"], axis=-1)
    yield "ttest_ind batch keepdims", lambda: h.ttest_ind(d["batch_a"], d["batch_b"], axis=-1, keepdims=True)
    yield "ttest_1samp axis tuple", lambda: h.ttest_1samp(d["cube"], 0.0, axis=(0, 2))
    yield "ttest_1samp popmean array", lambda: h.ttest_1samp(d["batch_a"], np.array([0.0, 0.1, -0.1, 0.2]), axis=-1)
    yield "ttest_ind long", lambda: h.ttest_ind(d["long_a"], d["long_b"], equal_var=False)
    yield "ttest_1samp scale 1e300", lambda: h.ttest_1samp(a * 1e300, 1e299)
    yield "ttest_1samp scale 1e-300", lambda: h.ttest_1samp(a * 1e-300, 1e-301)
    yield "ttest_ind_from_stats", lambda: h.ttest_ind_from_stats(0.2, 1.1, 12, -0.1, 1.4, 15)
    yield (
        "ttest_ind_from_stats Welch less",
        lambda: h.ttest_ind_from_stats(0.2, 1.1, 12, -0.1, 1.4, 15, equal_var=False, alternative="less"),
    )
    yield (
        "ttest_ind_from_stats broadcast",
        lambda: h.ttest_ind_from_stats(np.array([0.0, 0.5, 1.0]), 1.0, 10, 0.2, np.array([[1.0], [2.0]]), 12),
    )
    yield "ttest_ind_from_stats zero std", lambda: h.ttest_ind_from_stats(1.0, 0.0, 5, 2.0, 0.0, 5)
    for ev in (True, False):
        yield f"f_oneway equal_var={ev}", lambda ev=ev: h.f_oneway(a, b, c, equal_var=ev)
        yield f"f_oneway offset equal_var={ev}", lambda ev=ev: h.f_oneway(d["off_a"], d["off_b"], equal_var=ev)
    yield "f_oneway constant groups", lambda: h.f_oneway(d["const"], d["const"] + 1.0)
    yield "f_oneway batch", lambda: h.f_oneway(d["batch_a"], d["batch_b"], axis=-1)
    yield "f_oneway long", lambda: h.f_oneway(d["long_a"], d["long_b"], a)
    yield "alexandergovern", lambda: h.alexandergovern(a, b, c)
    yield "alexandergovern offset", lambda: h.alexandergovern(d["off_a"], d["off_b"])
    yield "bartlett", lambda: h.bartlett(a, b, c)
    yield "bartlett constant", lambda: h.bartlett(a, d["const"])
    yield "bartlett batch", lambda: h.bartlett(d["batch_a"], d["batch_b"], axis=-1)
    for center in ("median", "mean", "trimmed"):
        yield f"levene {center}", lambda center=center: h.levene(a, b, c, center=center, proportiontocut=0.2)
        yield f"fligner {center}", lambda center=center: h.fligner(a, b, c, center=center, proportiontocut=0.2)
    yield "levene offset", lambda: h.levene(d["off_a"], d["off_b"])
    yield "fligner ties", lambda: h.fligner(d["ties"], b)
    yield "levene long", lambda: h.levene(d["long_a"], d["long_b"])
    yield "fligner long", lambda: h.fligner(d["long_a"], d["long_b"])
    yield "levene one sample", lambda: h.levene(a)
    yield "ttest_ind wrong keyword", lambda: h.ttest_ind(a, b, equal_variance=True)
    yield "ttest_ind trim 0.5", lambda: h.ttest_ind(a, b, trim=0.5)

    # ---- H2: goodness of fit
    obs = d["counts"]
    yield "chisquare", lambda: h.chisquare(obs)
    yield "chisquare f_exp", lambda: h.chisquare(obs, np.array([16.0, 16, 16, 16, 16, 8]))
    yield "chisquare ddof=2", lambda: h.chisquare(obs, ddof=2)
    yield "chisquare ddof=-1 no sum_check", lambda: h.chisquare(obs, obs + 1.0, ddof=-1, sum_check=False)
    yield "chisquare sum_check", lambda: h.chisquare(obs, obs + 1.0)
    yield "chisquare one cell", lambda: h.chisquare([7.0])
    yield "chisquare batch", lambda: h.chisquare(d["count_batch"], axis=-1)
    yield (
        "chisquare batch f_exp broadcast",
        lambda: h.chisquare(d["count_batch"], d["count_batch"].sum(axis=-1, keepdims=True) / 6, axis=-1),
    )
    for pol in ("propagate", "omit", "raise"):
        yield f"chisquare nan {pol}", lambda pol=pol: h.chisquare([4.0, NAN, 6.0, 9.0], nan_policy=pol)
    for lam in (None, "pearson", "log-likelihood", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"):
        yield f"power_divergence {lam}", lambda lam=lam: h.power_divergence(obs, lambda_=lam)
        yield f"power_divergence zero count {lam}", lambda lam=lam: h.power_divergence(d["zero_counts"], lambda_=lam)
    for lam in (0.3, 1e-9, -0.999, -1.5, 4.0):
        yield f"power_divergence lambda={lam}", lambda lam=lam: h.power_divergence(obs, lambda_=lam)
    yield "power_divergence near-equal G", lambda: h.power_divergence(d["near"], np.full(3, 1e6), lambda_=0.0)
    yield "power_divergence lambda_ name", lambda: h.power_divergence(obs, lambda_="g-test")
    yield "power_divergence long", lambda: h.power_divergence(np.abs(d["long_a"]) + 1.0, lambda_="cressie-read")
    samples = {"small": a, "ties": d["ties"], "offset": d["off_a"], "expo": d["expo"], "long": d["long_a"]}
    for name, x in samples.items():
        for alt in ("two-sided", "less", "greater"):
            yield f"skewtest {name} {alt}", lambda x=x, alt=alt: h.skewtest(x, alternative=alt)
            yield f"kurtosistest {name} {alt}", lambda x=x, alt=alt: h.kurtosistest(x, alternative=alt)
        yield f"normaltest {name}", lambda x=x: h.normaltest(x)
        yield f"jarque_bera {name}", lambda x=x: h.jarque_bera(x)
        yield f"shapiro {name}", lambda x=x: h.shapiro(x[:20_000])
        yield f"anderson {name}", lambda x=x: h.anderson(x)
    for name, x in (("constant", d["const"]), ("empty", np.zeros(0)), ("small n", a[:4]), ("one", 2.0)):
        for f in ("skewtest", "kurtosistest", "normaltest", "jarque_bera", "shapiro", "anderson"):
            yield f"{f} {name}", lambda f=f, x=x: getattr(h, f)(x)
    yield "skewtest zero skewness", lambda: h.skewtest(np.arange(-4.0, 5.0))
    yield "shapiro n=3", lambda: h.shapiro([0.0, 0.0001, 1.0])
    for pol in ("propagate", "omit", "raise"):
        for f in ("normaltest", "jarque_bera", "shapiro", "anderson"):
            yield f"{f} nan {pol}", lambda f=f, pol=pol: getattr(h, f)(d["nan"], nan_policy=pol)
    for f in ("skewtest", "normaltest", "jarque_bera", "shapiro", "anderson"):
        yield f"{f} batch", lambda f=f: getattr(h, f)(d["batch_b"], axis=-1)
        yield f"{f} batch keepdims", lambda f=f: getattr(h, f)(d["batch_b"], axis=-1, keepdims=True)
    for dist, x in ((sd.stats.Normal, a), (sd.stats.Exponential, d["expo"]), (sd.stats.Logistic, d["logis"]),
                    (sd.stats.Gumbel, d["gumb"])):  # fmt: skip
        yield f"anderson dist={dist.__name__}", lambda dist=dist, x=x: h.anderson(x, dist=dist)
        yield f"anderson dist={dist.__name__} normal data", lambda dist=dist: h.anderson(d["off_a"] - 1e8, dist=dist)
    yield "anderson dist string", lambda: h.anderson(a, dist="norm")
    yield "anderson dist instance", lambda: h.anderson(a, dist=sd.stats.Normal())
    yield "anderson dist Weibull", lambda: h.anderson(a, dist=sd.stats.Weibull)
    yield "anderson method", lambda: h.anderson(a, method="interpolated")
    for alt in ("two-sided", "less", "greater"):
        for meth in ("auto", "exact", "approx", "asymp"):
            yield f"ks_1samp {alt} {meth}", lambda alt=alt, meth=meth: h.ks_1samp(a, N, alternative=alt, method=meth)
    for n in (1, 2, 30, 141, 400, 2000):
        yield f"ks_1samp n={n}", lambda n=n: h.ks_1samp(d["long_a"][:n] + 0.05, N)
    yield "ks_1samp deep tail", lambda: h.ks_1samp(b + 4.0, N)
    yield "ks_1samp deep tail one-sided", lambda: h.ks_1samp(b - 3.0, N, alternative="greater")
    yield "ks_1samp long", lambda: h.ks_1samp(d["long_a"], sd.stats.Normal(mu=0.003))
    yield "ks_1samp exponential", lambda: h.ks_1samp(d["expo"], sd.stats.Exponential())
    yield "ks_1samp batch", lambda: h.ks_1samp(d["batch_a"], N, axis=-1)
    yield "ks_1samp batch keepdims", lambda: h.ks_1samp(d["batch_a"], N, axis=-1, keepdims=True)
    yield "ks_1samp callable", lambda: h.ks_1samp(a, lambda v: sd.stats.Logistic().cdf(v))
    yield "ks_1samp empty", lambda: h.ks_1samp([], N)
    yield "ks_1samp ties", lambda: h.ks_1samp(d["ties"], sd.stats.Normal(sigma=2.0))
    for pol in ("propagate", "omit", "raise"):
        yield f"ks_1samp nan {pol}", lambda pol=pol: h.ks_1samp(d["nan"], N, nan_policy=pol)
        yield f"cramervonmises nan {pol}", lambda pol=pol: h.cramervonmises(d["nan"], N, nan_policy=pol)
    yield "ks_1samp cdf string", lambda: h.ks_1samp(a, "norm")
    yield "ks_1samp cdf class", lambda: h.ks_1samp(a, sd.stats.Normal)
    yield "ks_1samp cdf scalar", lambda: h.ks_1samp(a, lambda v: 0.5)
    yield "ks_1samp method", lambda: h.ks_1samp(a, N, method="exactly")
    for n in (2, 3, 10, 200, 2000):
        yield f"cramervonmises n={n}", lambda n=n: h.cramervonmises(d["long_b"][:n], N)
    yield "cramervonmises tail", lambda: h.cramervonmises(a + 2.0, N)
    yield "cramervonmises long", lambda: h.cramervonmises(d["long_a"], N)
    yield "cramervonmises batch", lambda: h.cramervonmises(d["batch_a"], N, axis=-1)
    yield (
        "cramervonmises uniform",
        lambda: h.cramervonmises(np.linspace(0.01, 0.95, 40) ** 1.2, sd.stats.Uniform(a=0.0, b=1.0)),
    )
    yield "cramervonmises one", lambda: h.cramervonmises([0.3], N)

    # ---- H3: rank tests
    t1, t2 = d["ties"], np.round(d["b"] * 2.0)
    zd = np.round(d["a"] * 2.0)  # zeros and ties among the differences
    for alt in ("two-sided", "less", "greater"):
        for meth in ("auto", "exact", "asymptotic"):
            yield f"mannwhitneyu {alt} {meth}", lambda alt=alt, m=meth: h.mannwhitneyu(a, b, alternative=alt, method=m)
            yield (
                f"mannwhitneyu ties {alt} {meth}",
                lambda alt=alt, m=meth: h.mannwhitneyu(t1, t2, alternative=alt, method=m),
            )
            yield f"wilcoxon {alt} {meth}", lambda alt=alt, m=meth: h.wilcoxon(a, b[:12], alternative=alt, method=m)
            yield (
                f"wilcoxon ties {alt} {meth}",
                lambda alt=alt, m=meth: h.wilcoxon(t1[:20] + 0.5, alternative=alt, method=m),
            )
            yield f"ansari {alt} {meth}", lambda alt=alt, m=meth: h.ansari(a, b, alternative=alt, method=m)
            yield f"ansari ties {alt} {meth}", lambda alt=alt, m=meth: h.ansari(t1, t2, alternative=alt, method=m)
            if alt == "two-sided":
                yield f"page_trend_test {meth}", lambda m=meth: h.page_trend_test(d["cube"][0].T[:8], method=m)
        yield f"ranksums {alt}", lambda alt=alt: h.ranksums(a, b, alternative=alt)
        yield f"mood {alt}", lambda alt=alt: h.mood(a, b, alternative=alt)
        yield f"mood ties {alt}", lambda alt=alt: h.mood(t1, t2, alternative=alt)
        for dist in ("t", "normal"):
            yield (
                f"brunnermunzel {alt} {dist}",
                lambda alt=alt, dist=dist: h.brunnermunzel(a, b, alternative=alt, distribution=dist),
            )
            yield (
                f"brunnermunzel ties {alt} {dist}",
                lambda alt=alt, dist=dist: h.brunnermunzel(t1, t2, alternative=alt, distribution=dist),
            )
    for zm in ("wilcox", "pratt", "zsplit"):
        for meth in ("auto", "exact", "asymptotic"):
            for corr in (False, True):
                yield (
                    f"wilcoxon zeros {zm} {meth} correction={corr}",
                    lambda zm=zm, m=meth, corr=corr: h.wilcoxon(zd, zero_method=zm, method=m, correction=corr),
                )
    yield "wilcoxon zeros small auto", lambda: h.wilcoxon(zd[:10])
    yield "mannwhitneyu no continuity", lambda: h.mannwhitneyu(t1, t2, use_continuity=False)
    yield "mannwhitneyu exact 8 x 2000", lambda: h.mannwhitneyu(a[:8], d["long_b"][:2000])
    yield "mannwhitneyu exact deep", lambda: h.mannwhitneyu(a, b + 10.0, method="exact")
    yield "wilcoxon exact 200", lambda: h.wilcoxon(d["long_a"][:200] + 0.1, method="exact")
    yield "wilcoxon exact deep", lambda: h.wilcoxon(np.abs(d["long_a"][:70]) + 0.1, method="exact")
    yield "ansari exact 54", lambda: h.ansari(d["long_a"][:54], d["long_b"][:54] * 1.4)
    yield "ansari exact odd", lambda: h.ansari(d["long_a"][:20], d["long_b"][:25] * 2.0)
    for name, gs in (("three", (a, b, c)), ("ties", (t1, t2, np.round(c * 2))), ("constant", (d["const"], d["const"]))):
        yield f"kruskal {name}", lambda gs=gs: h.kruskal(*gs)
        yield f"median_test {name}", lambda gs=gs: h.median_test(*gs)
    yield "friedmanchisquare", lambda: h.friedmanchisquare(a, b[:12], c[:9].tolist() + [0.1, 0.2, 0.3])
    yield "friedmanchisquare ties", lambda: h.friedmanchisquare(t1, t2[:30], np.round(d["long_a"][:30]))
    yield "friedmanchisquare constant", lambda: h.friedmanchisquare(d["const"], d["const"], d["const"])
    for ties in ("below", "above", "ignore"):
        for corr in (True, False):
            for lam in (1, "log-likelihood", "freeman-tukey", 0.4):
                yield (
                    f"median_test {ties} correction={corr} lambda={lam}",
                    lambda ties=ties, corr=corr, lam=lam: h.median_test(
                        t1, t2, ties=ties, correction=corr, lambda_=lam
                    ),
                )
    # H83: a zero cell with lambda <= -1 is +inf and p = 0 (the shared Cressie-Read rule)
    yield "median_test zero cell neyman", lambda: h.median_test([1.0, 2, 3, 4], [10.0, 11, 12, 13], lambda_="neyman",
                                                                 correction=False)  # fmt: skip
    yield "median_test batch", lambda: h.median_test(d["batch_a"], d["batch_b"], axis=-1)
    yield "median_test batch keepdims", lambda: h.median_test(d["batch_a"], d["batch_b"], axis=-1, keepdims=True)
    yield "page_trend_test ties", lambda: h.page_trend_test(np.round(d["cube"][1].T[:6]))
    yield (
        "page_trend_test ranked",
        lambda: h.page_trend_test(np.array([[1.0, 2, 3, 4], [2, 1, 3, 4], [1, 3, 2, 4]]), ranked=True),
    )
    yield "page_trend_test predicted", lambda: h.page_trend_test(d["cube"][2].T, predicted_ranks=[4, 3, 2, 1])
    yield "page_trend_test batch", lambda: h.page_trend_test(d["cube"].transpose(0, 2, 1))
    yield "page_trend_test 20 x 3", lambda: h.page_trend_test(d["long_a"][:60].reshape(20, 3))
    yield "page_trend_test large", lambda: h.page_trend_test(d["long_a"][:30000].reshape(1000, 30))
    for f in ("mannwhitneyu", "ranksums", "brunnermunzel", "ansari", "mood"):
        yield f"{f} long", lambda f=f: getattr(h, f)(d["long_a"], d["long_b"])
        yield f"{f} batch", lambda f=f: getattr(h, f)(d["batch_a"], d["batch_b"], axis=-1)
        yield f"{f} batch keepdims", lambda f=f: getattr(h, f)(d["batch_a"], d["batch_b"], axis=-1, keepdims=True)
        yield f"{f} empty", lambda f=f: getattr(h, f)([], b)
        for pol in ("propagate", "omit", "raise"):
            yield f"{f} nan {pol}", lambda f=f, pol=pol: getattr(h, f)(d["nan"], b, nan_policy=pol)
    yield "wilcoxon long", lambda: h.wilcoxon(d["long_a"], d["long_b"])
    yield "wilcoxon batch", lambda: h.wilcoxon(d["batch_a"], axis=-1)
    yield "wilcoxon batch mixed", lambda: h.wilcoxon(np.stack([a, np.round(a * 2) + 0.5]), axis=-1)
    yield "kruskal long", lambda: h.kruskal(d["long_a"], d["long_b"], a)
    yield "kruskal batch", lambda: h.kruskal(d["batch_a"], d["batch_b"], axis=-1)
    yield "friedmanchisquare long", lambda: h.friedmanchisquare(d["long_a"], d["long_b"], d["long_a"][::-1])
    yield (
        "friedmanchisquare axis tuple",
        lambda: h.friedmanchisquare(d["cube"], d["cube"] * 2.0, -d["cube"], axis=(0, 2)),
    )
    yield "median_test long", lambda: h.median_test(d["long_a"], d["long_b"], a)
    for pol in ("propagate", "omit", "raise"):
        yield f"wilcoxon nan {pol}", lambda pol=pol: h.wilcoxon(d["nan"], nan_policy=pol)
        yield f"kruskal nan {pol}", lambda pol=pol: h.kruskal(d["nan"], a, nan_policy=pol)
        yield f"friedmanchisquare nan {pol}", lambda pol=pol: h.friedmanchisquare(d["nan"], a[:9], c, nan_policy=pol)
        yield f"median_test nan {pol}", lambda pol=pol: h.median_test(d["nan"], a, nan_policy=pol)
    yield "wilcoxon all zeros", lambda: h.wilcoxon(np.zeros(6))
    yield "brunnermunzel separated", lambda: h.brunnermunzel(a, a + 100.0)
    yield "mannwhitneyu exact cap", lambda: h.mannwhitneyu(a[:8], d["long_a"], method="exact")
    yield "wilcoxon exact cap", lambda: h.wilcoxon(d["long_a"][:3000], method="exact")
    yield "page_trend_test exact cap", lambda: h.page_trend_test(d["long_a"][:125].reshape(5, 25), method="exact")
    yield "page_trend_test nan", lambda: h.page_trend_test(d["nan"][:6].reshape(2, 3))
    yield "page_trend_test shape", lambda: h.page_trend_test(np.ones((3, 2)))
    yield "kruskal one sample", lambda: h.kruskal(a)
    yield "friedmanchisquare two samples", lambda: h.friedmanchisquare(a, a)
    yield "wilcoxon zero_method", lambda: h.wilcoxon(a, zero_method="drop")
    yield "mannwhitneyu method", lambda: h.mannwhitneyu(a, b, method="permutation")

    # ---- H4: goodness of fit between samples
    t3 = np.round(c * 2)  # ties
    for alt in ("two-sided", "less", "greater"):
        for meth in ("auto", "exact", "asymp"):
            yield f"ks_2samp {alt} {meth}", lambda alt=alt, m=meth: h.ks_2samp(a, b, alternative=alt, method=m)
            yield f"ks_2samp ties {alt} {meth}", lambda alt=alt, m=meth: h.ks_2samp(t1, t2, alternative=alt, method=m)
        yield f"ks_2samp equal sizes {alt}", lambda alt=alt: h.ks_2samp(a, b[:12] + 0.5, alternative=alt)
        yield (
            f"ks_2samp long {alt}",
            lambda alt=alt: h.ks_2samp(d["long_a"][:9000], d["long_b"][:7000], alternative=alt),
        )
    yield "ks_2samp separated", lambda: h.ks_2samp(a, b + 100.0)
    yield "ks_2samp asymp long", lambda: h.ks_2samp(d["long_a"], d["long_b"])
    yield "ks_2samp ties long", lambda: h.ks_2samp(np.round(d["long_a"][:4000], 1), np.round(d["long_b"][:3000], 1))
    yield "ks_2samp offset", lambda: h.ks_2samp(d["off_a"], d["off_b"])
    yield "ks_2samp constant", lambda: h.ks_2samp(d["const"], d["const"][:5])
    yield "ks_2samp batch", lambda: h.ks_2samp(d["batch_a"], d["batch_b"], axis=-1)
    yield "ks_2samp batch keepdims", lambda: h.ks_2samp(d["batch_a"], d["batch_b"], axis=-1, keepdims=True)
    yield "ks_2samp exact cap", lambda: h.ks_2samp(d["long_a"][:40000], d["long_b"][:39999], method="exact",
                                                     alternative="greater")  # fmt: skip
    for meth in ("auto", "exact", "asymptotic"):
        yield f"cramervonmises_2samp {meth}", lambda m=meth: h.cramervonmises_2samp(a, b, method=m)
        yield f"cramervonmises_2samp ties {meth}", lambda m=meth: h.cramervonmises_2samp(t1[:15], t3, method=m)
    yield "cramervonmises_2samp exact 60 x 50", lambda: h.cramervonmises_2samp(d["long_a"][:60], d["long_b"][:50],
                                                                               method="exact")  # fmt: skip
    yield "cramervonmises_2samp exact separated", lambda: h.cramervonmises_2samp(a, c + 50.0)
    yield "cramervonmises_2samp long", lambda: h.cramervonmises_2samp(d["long_a"], d["long_b"])
    yield "cramervonmises_2samp batch", lambda: h.cramervonmises_2samp(d["batch_a"], d["batch_b"], axis=-1)
    yield "cramervonmises_2samp exact cap", lambda: h.cramervonmises_2samp(d["long_a"][:3000], d["long_b"][:2999],
                                                                           method="exact")  # fmt: skip
    yield "cramervonmises_2samp one", lambda: h.cramervonmises_2samp([0.3], b)
    for v in ("midrank", "right", "continuous"):
        yield f"anderson_ksamp {v}", lambda v=v: h.anderson_ksamp([a, b, c], variant=v)
        yield f"anderson_ksamp ties {v}", lambda v=v: h.anderson_ksamp([t1, t2, t3], variant=v)
    yield "anderson_ksamp default", lambda: h.anderson_ksamp((a, b))
    yield "anderson_ksamp capped", lambda: h.anderson_ksamp([a, b + 10.0])
    yield "anderson_ksamp long", lambda: h.anderson_ksamp([d["long_a"], d["long_b"]])
    yield "anderson_ksamp offset", lambda: h.anderson_ksamp([d["off_a"], d["off_b"]])
    yield "anderson_ksamp constant", lambda: h.anderson_ksamp([d["const"], d["const"]])
    yield "anderson_ksamp one sample", lambda: h.anderson_ksamp([a])
    yield "anderson_ksamp array", lambda: h.anderson_ksamp(np.stack([a, a]))
    yield "anderson_ksamp variant", lambda: h.anderson_ksamp([a, b], variant="mid")
    for t in ((0.4, 0.8), (0.2, 0.5, 1.0), 0.7):
        yield f"epps_singleton_2samp t={t}", lambda t=t: h.epps_singleton_2samp(a, b, t=t)
    yield "epps_singleton_2samp ties", lambda: h.epps_singleton_2samp(t1, t2)
    yield "epps_singleton_2samp long", lambda: h.epps_singleton_2samp(d["long_a"], d["long_b"])
    yield "epps_singleton_2samp offset", lambda: h.epps_singleton_2samp(d["off_a"], d["off_b"])
    yield "epps_singleton_2samp batch", lambda: h.epps_singleton_2samp(d["batch_a"], d["batch_b"], axis=-1)
    yield "epps_singleton_2samp four", lambda: h.epps_singleton_2samp(a[:4], b)
    yield "epps_singleton_2samp t", lambda: h.epps_singleton_2samp(a, b, t=(0.4, 0.0))
    yield (
        "epps_singleton_2samp rank-deficient",
        lambda: h.epps_singleton_2samp(np.round(np.abs(a)) % 2, np.round(np.abs(b)) % 2),
    )
    for f in ("ks_2samp", "cramervonmises_2samp", "epps_singleton_2samp"):
        yield f"{f} empty", lambda f=f: getattr(h, f)([], b)
        for pol in ("propagate", "omit", "raise"):
            yield f"{f} nan {pol}", lambda f=f, pol=pol: getattr(h, f)(d["nan"], b, nan_policy=pol)

    # ---- H5: correlation
    r5 = L.rng(405)
    u = r5.normal(0.0, 1.0, 25)
    v = 0.4 * u + r5.normal(0.0, 1.0, 25)
    ut, vt = np.round(u * 2.0), np.round(v * 2.0)
    near = u + 1e-8 * r5.normal(0.0, 1.0, 25)
    bu, bv = r5.normal(0.0, 1.0, (4, 18)), r5.normal(0.0, 1.0, (4, 18))
    la, lb = d["long_a"], d["long_a"] * 0.01 + d["long_b"]
    for alt in ("two-sided", "less", "greater"):
        yield f"pearsonr {alt}", lambda alt=alt: h.pearsonr(u, v, alternative=alt)
        yield f"pearsonr near 1 {alt}", lambda alt=alt: h.pearsonr(u, near, alternative=alt)
        yield f"spearmanrho {alt}", lambda alt=alt: h.spearmanrho(u, v, alternative=alt)
        yield f"spearmanrho ties {alt}", lambda alt=alt: h.spearmanrho(ut, vt, alternative=alt)
        yield f"spearmanr {alt}", lambda alt=alt: h.spearmanr(np.stack([u, v, ut], axis=1), axis=0, alternative=alt)
        for var in ("b", "c"):
            for meth in ("auto", "exact", "asymptotic"):
                yield (
                    f"kendalltau {var} {meth} {alt}",
                    lambda alt=alt, var=var, m=meth: h.kendalltau(u, v, variant=var, method=m, alternative=alt),
                )
            yield (
                f"kendalltau ties {var} {alt}",
                lambda alt=alt, var=var: h.kendalltau(ut, vt, variant=var, alternative=alt),
            )
        yield f"somersd {alt}", lambda alt=alt: h.somersd(ut, vt, alternative=alt)
    yield "pearsonr conf_int alpha=0.1", lambda: h.pearsonr(u, v).conf_int(alpha=0.1)
    yield "pearsonr offset", lambda: h.pearsonr(d["off_a"], d["off_a"] * 0.5 + d["off_b"][:40])
    yield "pearsonr scales", lambda: h.pearsonr(u * 1e300, v * 1e-300)
    yield "pearsonr exact line", lambda: h.pearsonr(u, 2.0 * u)
    yield "pearsonr two", lambda: h.pearsonr([1.0, 2.0], [3.0, 1.0])
    yield "pearsonr three", lambda: h.pearsonr([1.0, 2.0, 4.0], [0.3, 0.1, 0.9])
    yield "pearsonr constant", lambda: h.pearsonr(d["const"], u[:8])
    yield "pearsonr inf", lambda: h.pearsonr([1.0, INF, 3.0], [1.0, 2.0, 3.0])
    yield "pearsonr long", lambda: h.pearsonr(la, lb)
    yield "pearsonr batch", lambda: h.pearsonr(bu, bv, axis=-1)
    yield "pearsonr batch keepdims", lambda: h.pearsonr(bu, bv, axis=-1, keepdims=True)
    yield "pearsonr cube axis", lambda: h.pearsonr(d["cube"], d["cube"][:, ::-1], axis=1)
    yield "pearsonr method", lambda: h.pearsonr(u, v, method="permutation")
    yield "pointbiserialr", lambda: h.pointbiserialr(u > 0, v)
    yield "pointbiserialr batch", lambda: h.pointbiserialr(bu > 0, bv, axis=-1)
    yield "spearmanrho long", lambda: h.spearmanrho(la, lb)
    yield "spearmanrho long ties", lambda: h.spearmanrho(np.round(la, 1), np.round(lb, 1))
    yield "spearmanrho inf", lambda: h.spearmanrho([1.0, INF, 3.0, -INF], [1.0, 2.0, 3.0, 0.0])
    yield "spearmanrho batch", lambda: h.spearmanrho(bu, bv, axis=-1)
    yield "spearmanr two", lambda: h.spearmanr(u, v)
    yield "spearmanr rows", lambda: h.spearmanr(np.stack([u, v, ut]), axis=1)
    yield "spearmanr b", lambda: h.spearmanr(np.stack([u, v], axis=1), np.stack([ut, vt], axis=1), axis=0)
    un = u.copy()
    un[3] = NAN
    for pol in ("propagate", "omit", "raise"):
        yield f"spearmanr nan {pol}", lambda pol=pol: h.spearmanr(np.stack([un, v, ut], axis=1), axis=0,
                                                                  nan_policy=pol)  # fmt: skip
    yield "spearmanr one variable", lambda: h.spearmanr(u)
    yield "spearmanr axis", lambda: h.spearmanr(np.stack([u, v]), axis=2)
    yield "kendalltau exact 60 concordant", lambda: h.kendalltau(np.arange(60.0), np.arange(60.0) * 3.0)
    yield "kendalltau exact forced 150", lambda: h.kendalltau(la[:150], lb[:150], method="exact")
    yield "kendalltau exact cap", lambda: h.kendalltau(la[:3000], lb[:3000], method="exact")
    yield "kendalltau exact ties", lambda: h.kendalltau(ut, vt, method="exact")
    yield "kendalltau long", lambda: h.kendalltau(la, lb)
    yield "kendalltau batch", lambda: h.kendalltau(bu, bv, axis=-1)
    yield "kendalltau variant", lambda: h.kendalltau(u, v, variant="a")
    for add in (True, False):
        for rank in (True, None, False):
            yield (
                f"weightedtau additive={add} rank={rank}",
                lambda a=add, r=rank: h.weightedtau(u, v, rank=r, additive=a),
            )
        yield f"weightedtau ties additive={add}", lambda a=add: h.weightedtau(ut, vt, additive=a)
    yield "weightedtau rank array", lambda: h.weightedtau(u, v, rank=np.round(np.abs(v) * 4))
    yield "weightedtau weigher", lambda: h.weightedtau(u, v, weigher=lambda r: 1.0 / (r + 1.0) ** 2)
    yield "weightedtau long", lambda: h.weightedtau(la[:20000], lb[:20000])
    yield "weightedtau batch", lambda: h.weightedtau(bu, bv, axis=-1)
    yield "weightedtau weigher type", lambda: h.weightedtau(u, v, weigher=2.0)
    yield "somersd table", lambda: h.somersd([[8.0, 3, 1], [2, 9, 4], [0, 3, 7]])
    yield "somersd one row", lambda: h.somersd([[8.0, 3, 1]])
    yield "somersd balanced diagonal", lambda: h.somersd([[5.0, 0.0], [0.0, 5.0]])
    yield "somersd counts", lambda: h.somersd([[1.0, 2.5], [3.0, 1.0]])
    yield "somersd cap", lambda: h.somersd(la[:20000], lb[:20000])
    for cont in (True, False):
        yield f"chatterjeexi y_continuous={cont}", lambda c=cont: h.chatterjeexi(u, v, y_continuous=c)
        yield f"chatterjeexi ties y_continuous={cont}", lambda c=cont: h.chatterjeexi(ut, vt, y_continuous=c)
    yield "chatterjeexi long", lambda: h.chatterjeexi(la, np.sin(la) + lb)
    yield "chatterjeexi constant y", lambda: h.chatterjeexi(u, np.full(25, 1.0))
    yield "chatterjeexi batch", lambda: h.chatterjeexi(bu, bv, axis=-1)
    yield "chatterjeexi method", lambda: h.chatterjeexi(u, v, method="permutation")
    for f in ("pearsonr", "spearmanrho", "pointbiserialr", "kendalltau", "weightedtau", "chatterjeexi"):
        yield f"{f} empty", lambda f=f: getattr(h, f)([], [])
        yield f"{f} lengths", lambda f=f: getattr(h, f)(u, v[:20])
        for pol in ("propagate", "omit", "raise"):
            yield f"{f} nan {pol}", lambda f=f, pol=pol: getattr(h, f)(d["nan"], u[:9], nan_policy=pol)

    # ---- H6: contingency tables
    r6 = L.rng(406)
    t22 = np.array([[3.0, 7], [9, 2]])
    t34 = np.array([[10.0, 12, 8, 5], [7, 15, 9, 11], [4, 6, 14, 13]])
    weighted = np.array([[3.5, 4.25, 1.0], [1.5, 6.0, 2.75]])
    tb = np.floor(r6.uniform(0, 20, (4, 2, 2)))
    trc = np.floor(r6.uniform(0, 6, (3, 3, 3)))
    mirror = np.array([[7.0, 12], [8, 3]])
    for corr in (True, False):
        for lam in (None, "log-likelihood", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read", 0.3):
            yield (
                f"chi2_contingency correction={corr} lambda={lam}",
                lambda c=corr, lam=lam: h.chi2_contingency(t22, correction=c, lambda_=lam),
            )
            yield (
                f"chi2_contingency 3x4 correction={corr} lambda={lam}",
                lambda c=corr, lam=lam: h.chi2_contingency(t34, correction=c, lambda_=lam),
            )
    yield "chi2_contingency weighted", lambda: h.chi2_contingency(weighted)
    yield "chi2_contingency zero cell neyman", lambda: h.chi2_contingency([[0.0, 5], [6, 2]], lambda_="neyman",
                                                                           correction=False)  # fmt: skip
    yield "chi2_contingency yates within half", lambda: h.chi2_contingency([[5.0, 5], [5, 6]])
    yield "chi2_contingency zero row", lambda: h.chi2_contingency([[0.0, 0, 0], [3, 4, 5]])
    yield "chi2_contingency zero table", lambda: h.chi2_contingency([[0.0, 0], [0, 0]])
    yield "chi2_contingency one row", lambda: h.chi2_contingency([[3.0, 4, 5]])
    yield "chi2_contingency near expected", lambda: h.chi2_contingency([[1e6, 1e6 + 1], [1e6 + 1, 1e6]],
                                                                        lambda_="log-likelihood")  # fmt: skip
    yield "chi2_contingency batch", lambda: h.chi2_contingency(tb)
    yield "chi2_contingency 1-D", lambda: h.chi2_contingency([3.0, 4.0])
    yield "chi2_contingency negative", lambda: h.chi2_contingency([[1.0, -1.0], [2.0, 3.0]])
    yield "chi2_contingency method", lambda: h.chi2_contingency(t22, method="permutation")
    yield "chi2_contingency lambda type", lambda: h.chi2_contingency(t22, lambda_=[1])
    for alt in ("two-sided", "less", "greater"):
        yield f"fisher_exact {alt}", lambda alt=alt: h.fisher_exact(t22, alternative=alt)
        yield f"fisher_exact mirror {alt}", lambda alt=alt: h.fisher_exact([[2.0, 8], [8, 2]], alternative=alt)
        yield f"fisher_exact tail {alt}", lambda alt=alt: h.fisher_exact([[300.0, 10], [10, 300]], alternative=alt)
        for pooled in (True, False):
            yield (
                f"barnard_exact pooled={pooled} {alt}",
                lambda alt=alt, p=pooled: h.barnard_exact(mirror, alternative=alt, pooled=p),
            )
        yield f"boschloo_exact {alt}", lambda alt=alt: h.boschloo_exact(mirror, alternative=alt)
        yield f"boschloo_exact small {alt}", lambda alt=alt: h.boschloo_exact(t22, alternative=alt)
        for kind in ("conditional", "sample"):
            yield (
                f"odds_ratio {kind} conf_int {alt}",
                lambda alt=alt, k=kind: h.odds_ratio(t22, kind=k).conf_int(alternative=alt),
            )
    yield "fisher_exact balanced", lambda: h.fisher_exact([[5.0, 5], [5, 5]])
    yield "fisher_exact zero column", lambda: h.fisher_exact([[0.0, 4], [0, 5]])
    yield "fisher_exact deep tail", lambda: h.fisher_exact([[500.0, 0], [0, 500]])
    yield "fisher_exact 3x3", lambda: h.fisher_exact([[3.0, 1, 4], [1, 5, 9], [2, 6, 5]])
    yield "fisher_exact 2x6", lambda: h.fisher_exact([[9.0, 3, 7, 1, 6, 12], [4, 10, 2, 8, 9, 3]])
    yield "fisher_exact 4x5", lambda: h.fisher_exact([[1.0, 2, 5, 20, 2], [9, 0, 1, 10, 9], [0, 0, 10, 0, 1],
                                                       [3, 8, 5, 5, 9]])  # fmt: skip
    yield "fisher_exact one row", lambda: h.fisher_exact([[3.0, 4, 5]])
    yield "fisher_exact batch", lambda: h.fisher_exact(tb)
    yield "fisher_exact batch rxc", lambda: h.fisher_exact(trc)
    cap = [[0.0, 0, 2, 0, 1, 1], [2, 0, 2, 0, 0, 3], [4, 1, 3, 1, 2, 2], [1, 2, 3, 0, 1, 3], [1, 3, 4, 0, 1, 2],
           [1, 2, 1, 5, 2, 4]]  # fmt: skip
    yield "fisher_exact cap", lambda: h.fisher_exact(cap)
    yield "fisher_exact rxc alternative", lambda: h.fisher_exact(trc[0], alternative="less")
    yield "fisher_exact counts", lambda: h.fisher_exact([[1.0, 2.5], [3, 1]])
    yield "fisher_exact empty", lambda: h.fisher_exact(np.zeros((0, 2)))
    yield "barnard_exact n=1", lambda: h.barnard_exact(t22, n=1)
    yield "barnard_exact tail", lambda: h.barnard_exact([[25.0, 2], [1, 22]])
    yield "barnard_exact zero column", lambda: h.barnard_exact([[0.0, 3], [0, 4]])
    yield "barnard_exact large", lambda: h.barnard_exact([[60.0, 45], [40, 55]])
    yield "barnard_exact cap", lambda: h.barnard_exact([[2000.0, 1], [3000, 4000]])
    yield "barnard_exact batch", lambda: h.barnard_exact(tb + 1.0)
    yield "barnard_exact shape", lambda: h.barnard_exact(t34)
    yield "barnard_exact n", lambda: h.barnard_exact(t22, n=0)
    yield "boschloo_exact tail", lambda: h.boschloo_exact([[25.0, 2], [1, 22]])
    yield "boschloo_exact equal", lambda: h.boschloo_exact([[9.0, 3], [3, 9]])
    yield "boschloo_exact zero column", lambda: h.boschloo_exact([[0.0, 3], [0, 4]])
    yield "boschloo_exact batch", lambda: h.boschloo_exact(tb + 1.0)
    for ex in (True, False):
        for corr in (True, False):
            for case, tab in (("", [[30.0, 12], [4, 25]]), (" equal", [[10.0, 6], [6, 12]]),
                              (" zero", [[10.0, 0], [0, 12]])):  # fmt: skip
                yield (
                    f"mcnemar{case} exact={ex} correction={corr}",
                    lambda e=ex, c=corr, t=tab: h.mcnemar(t, exact=e, correction=c),
                )
    yield "mcnemar large", lambda: h.mcnemar([[500.0, 1200], [900, 400]])
    yield "mcnemar batch", lambda: h.mcnemar(tb)
    yield "mcnemar exact type", lambda: h.mcnemar(t22, exact=1)
    q = np.floor(r6.uniform(0, 2, (30, 4)))
    yield "cochrans_q", lambda: h.cochrans_q(q)
    yield "cochrans_q degenerate", lambda: h.cochrans_q([[1.0, 1, 1], [0, 0, 0]])
    yield "cochrans_q batch", lambda: h.cochrans_q(np.floor(r6.uniform(0, 2, (3, 15, 3))))
    yield "cochrans_q values", lambda: h.cochrans_q([[0.0, 2.0], [1.0, 1.0]])
    for kind in ("conditional", "sample"):
        yield f"odds_ratio {kind}", lambda k=kind: h.odds_ratio(t22, kind=k)
        yield f"odds_ratio {kind} alpha=0.1", lambda k=kind: h.odds_ratio(t22, kind=k).conf_int(alpha=0.1)
        for case, tab in (("zero cell", [[0.0, 5], [3, 4]]), ("infinite", [[5.0, 0], [3, 4]]),
                          ("zero margin", [[0.0, 0], [3, 4]]), ("large", [[100.0, 50], [30, 120]])):  # fmt: skip
            yield f"odds_ratio {kind} {case}", lambda k=kind, t=tab: h.odds_ratio(t, kind=k)
        yield f"odds_ratio {kind} batch", lambda k=kind: h.odds_ratio(tb + 1.0, kind=k)
    yield "odds_ratio kind", lambda: h.odds_ratio(t22, kind="exact")
    yield "odds_ratio alternative", lambda: h.odds_ratio(t22).conf_int(alternative="up")
    for args in ((9, 20, 4, 21), (0, 10, 3, 12), (3, 10, 0, 12), (0, 10, 0, 12), (3000, 10000, 2000, 10000)):
        yield f"relative_risk {args}", lambda a=args: h.relative_risk(*a)
        yield f"relative_risk {args} alpha=0.1", lambda a=args: h.relative_risk(*a).conf_int(alpha=0.1)
    yield "relative_risk exceeds", lambda: h.relative_risk(5, 4, 1, 2)
    yield "relative_risk float", lambda: h.relative_risk(1.0, 4, 1, 2)
    yield from proportions(L)
    yield from summaries(L)


def proportions(L):
    """H7: binomtest over its alternatives, p and conf_int's methods (and the rule's edges: p = 0
    and 1, k = p n, a deep tail), quantile_test over q, p, nan_policy and batches, the E-test
    (diff, one-sided, past scipy's truncation, a non-positive null estimate), statsmodels'
    proportions_ztest, proportions_chisquare, proportion_confint (every method and side, batches,
    nobs = 0), test_proportions_2indep and confint_proportions_2indep over compare, method, value,
    alternative and correction (zero counts, batches), and the wrong calls."""
    h = L.sd.htest
    d = data(L)
    r = L.rng(411)
    for alt in ("two-sided", "less", "greater"):
        for k, n, p in ((3, 10, 0.5), (7, 20, 0.3), (0, 10, 0.5), (31, 100, 0.3), (19, 40, 0.2)):
            yield f"binomtest {k} {n} {p} {alt}", lambda k=k, n=n, p=p, alt=alt: h.binomtest(k, n, p=p, alternative=alt)
            for m in ("wilson", "wilsoncc"):
                yield (
                    f"binomtest {k} {n} {p} {alt} conf_int {m}",
                    lambda k=k, n=n, p=p, alt=alt, m=m: h.binomtest(k, n, p=p, alternative=alt).conf_int(method=m),
                )
    for args, kw in (((5, 10), {}), ((0, 10), {"p": 0.0}), ((7, 10), {"p": 1.0}), ((0, 1000), {}),
                     ((1, 10), {"p": 0.1}), ((50300, 100000), {})):  # fmt: skip
        yield f"binomtest {args} {kw}", lambda a=args, k=kw: h.binomtest(*a, **k)
    yield "binomtest conf_int alpha=0.1", lambda: h.binomtest(7, 20).conf_int(alpha=0.1)
    yield "binomtest k float", lambda: h.binomtest(3.0, 10)
    yield "binomtest k > n", lambda: h.binomtest(11, 10)
    yield "binomtest p range", lambda: h.binomtest(3, 10, p=1.5)
    yield "binomtest conf_int method", lambda: h.binomtest(3, 10).conf_int(method="beta")
    xq = d["ties"]
    for alt in ("two-sided", "less", "greater"):
        yield f"quantile_test {alt}", lambda alt=alt: h.quantile_test(xq, q=0.0, p=0.4, alternative=alt)
        yield (
            f"quantile_test {alt} alpha=0.1",
            lambda alt=alt: h.quantile_test(xq, q=1.0, alternative=alt).conf_int(alpha=0.1),
        )
    yield "quantile_test outside", lambda: h.quantile_test(d["a"], q=10.0)
    yield "quantile_test batch", lambda: h.quantile_test(d["batch_a"], q=0.1, axis=-1)
    yield "quantile_test batch keepdims", lambda: h.quantile_test(d["batch_a"], q=0.1, axis=-1, keepdims=True)
    for pol in ("propagate", "omit", "raise"):
        yield f"quantile_test nan {pol}", lambda pol=pol: h.quantile_test(d["nan"], q=0.5, nan_policy=pol)
    yield "quantile_test empty", lambda: h.quantile_test([], q=0.0)
    yield "quantile_test p range", lambda: h.quantile_test(d["a"], p=1.0)
    for args, kw in (((10, 5.0, 20, 6.0), {}), ((30, 2.0, 10, 3.0), {"diff": 2.0}),
                     ((10, 5.0, 20, 6.0), {"alternative": "less"}), ((10, 5.0, 20, 6.0), {"alternative": "greater"}),
                     ((15, 5.24, 34, 0.64), {"alternative": "less"}),
                     ((0, 2.0, 0, 3.0), {}), ((5, 1.0, 5, 1.0), {}), ((400, 1.0, 200, 1.0), {})):  # fmt: skip
        yield f"poisson_means_test {args} {kw}", lambda a=args, k=kw: h.poisson_means_test(*a, **k)
    yield "poisson_means_test diff negative", lambda: h.poisson_means_test(3, 1.0, 2, 1.0, diff=-1.0)
    yield "poisson_means_test n zero", lambda: h.poisson_means_test(3, 0.0, 2, 1.0)
    cnt, nob = np.array([15.0, 25]), np.array([40.0, 45])
    for alt in ("two-sided", "less", "greater"):
        yield f"proportions_ztest two {alt}", lambda alt=alt: h.proportions_ztest(cnt, nob, alternative=alt)
        yield f"proportions_ztest one {alt}", lambda alt=alt: h.proportions_ztest(15, 40, value=0.5, alternative=alt)
    yield "proportions_ztest prop_var", lambda: h.proportions_ztest(15, 40, value=0.5, prop_var=0.45)
    yield "proportions_ztest value two", lambda: h.proportions_ztest(cnt, nob, value=-0.05)
    yield "proportions_ztest zero var", lambda: h.proportions_ztest(0, 30, value=0.5)
    yield (
        "proportions_ztest batch",
        lambda: h.proportions_ztest(np.array([[15.0, 25], [3, 9]]), np.array([[40.0, 45], [20, 22]])),
    )
    yield "proportions_ztest one value", lambda: h.proportions_ztest(15, 40)
    yield "proportions_ztest three", lambda: h.proportions_ztest([1, 2, 3], [5, 5, 5])
    yield "proportions_ztest prop_var bool", lambda: h.proportions_ztest(15, 40, value=0.5, prop_var=False)
    yield "proportions_ztest count exceeds", lambda: h.proportions_ztest(41, 40, value=0.5)
    yield "proportions_chisquare", lambda: h.proportions_chisquare(cnt, nob)
    yield "proportions_chisquare three", lambda: h.proportions_chisquare([15, 25, 9], [40, 45, 30])
    yield "proportions_chisquare value", lambda: h.proportions_chisquare(cnt, nob, value=0.4)
    yield "proportions_chisquare zero column", lambda: h.proportions_chisquare([5, 7], [5, 7])
    yield "proportions_chisquare one", lambda: h.proportions_chisquare(12, 40)
    yield (
        "proportions_chisquare batch",
        lambda: h.proportions_chisquare(np.array([[15.0, 25], [3, 9]]), np.array([[40.0, 45], [20, 22]])),
    )
    yield "proportions_chisquare non-integer", lambda: h.proportions_chisquare([1.5, 2], [4, 4])
    for meth in ("normal", "agresti_coull", "exact", "wilson", "wilsoncc", "jeffreys", "binom_test"):
        for alt in ("two-sided", "less", "greater"):
            yield (
                f"proportion_confint {meth} {alt}",
                lambda m=meth, alt=alt: h.proportion_confint(7, 30, method=m, alternative=alt),
            )
        yield f"proportion_confint {meth} ends", lambda m=meth: h.proportion_confint([0, 30, 1, 29], 30, method=m)
        yield f"proportion_confint {meth} alpha=0.1", lambda m=meth: h.proportion_confint(7, 30, method=m, alpha=0.1)
    yield "proportion_confint large", lambda: h.proportion_confint(30123, 100000, method="wilson")
    yield "proportion_confint binom_test 200", lambda: h.proportion_confint(61, 200, method="binom_test")
    yield (
        "proportion_confint batch",
        lambda: h.proportion_confint(np.floor(r.uniform(0, 20, (3, 4))), 20, method="exact"),
    )
    yield "proportion_confint nobs zero", lambda: h.proportion_confint([0, 1], [0, 2])
    yield "proportion_confint beta", lambda: h.proportion_confint(7, 30, method="beta")
    yield "proportion_confint larger", lambda: h.proportion_confint(7, 30, alternative="larger")
    for comp, meths in (("diff", ("wald", "agresti-caffo", "score")), ("ratio", ("log", "log-adjusted", "score")),
                        ("odds-ratio", ("logit", "logit-adjusted", "logit-smoothed", "score"))):  # fmt: skip
        for meth in meths:
            yield (
                f"test_proportions_2indep {comp} {meth}",
                lambda c=comp, m=meth: h.test_proportions_2indep(15, 40, 25, 45, compare=c, method=m),
            )
            yield (
                f"test_proportions_2indep {comp} {meth} zero",
                lambda c=comp, m=meth: h.test_proportions_2indep(0, 40, 25, 45, compare=c, method=m),
            )
        val = 0.1 if comp == "diff" else 1.5
        for alt in ("two-sided", "less", "greater"):
            yield (
                f"test_proportions_2indep {comp} score value {alt}",
                lambda c=comp, v=val, alt=alt: h.test_proportions_2indep(
                    15, 40, 25, 45, compare=c, method="score", value=v, alternative=alt
                ),
            )
        yield (
            f"test_proportions_2indep {comp} score nocorr",
            lambda c=comp: h.test_proportions_2indep(15, 40, 25, 45, compare=c, method="score", correction=False),
        )
        for meth in meths + (("newcomb",) if comp == "diff" else ()):
            yield (
                f"confint_proportions_2indep {comp} {meth}",
                lambda c=comp, m=meth: h.confint_proportions_2indep(15, 40, 25, 45, compare=c, method=m),
            )
            yield (
                f"confint_proportions_2indep {comp} {meth} zero",
                lambda c=comp, m=meth: h.confint_proportions_2indep(0, 40, 25, 45, compare=c, method=m),
            )
            yield (
                f"confint_proportions_2indep {comp} {meth} alpha=0.1",
                lambda c=comp, m=meth: h.confint_proportions_2indep(15, 40, 25, 45, compare=c, method=m, alpha=0.1),
            )
        yield (
            f"confint_proportions_2indep {comp} batch",
            lambda c=comp: h.confint_proportions_2indep(
                np.array([15.0, 5, 30]), 40, np.array([25.0, 9, 30]), 45, compare=c
            ),
        )
    yield (
        "test_proportions_2indep batch",
        lambda: h.test_proportions_2indep(np.array([15.0, 5, 30]), 40, np.array([25.0, 9, 30]), 45),
    )
    yield "test_proportions_2indep newcomb", lambda: h.test_proportions_2indep(15, 40, 25, 45, method="newcomb")
    yield "test_proportions_2indep or", lambda: h.test_proportions_2indep(15, 40, 25, 45, compare="or")
    yield "test_proportions_2indep value range", lambda: h.test_proportions_2indep(15, 40, 25, 45, value=1.0)
    yield "test_proportions_2indep larger", lambda: h.test_proportions_2indep(15, 40, 25, 45, alternative="larger")
    yield (
        "test_proportions_2indep return_results",
        lambda: h.test_proportions_2indep(15, 40, 25, 45, return_results=False),
    )


def summaries(L):
    """summary() of every test once (H74-H76; the result cases store it too, as a method without
    arguments): small samples, a one-sided interval, a batch, a long batch (cut), keepdims, the
    alpha check and the reprs of array fields."""
    sd, h = L.sd, L.sd.htest
    d = data(L)
    a, b, c = d["a"], d["b"], d["c"]
    N = sd.stats.Normal()
    r = L.rng(409)
    u, v = r.normal(0.0, 1.0, 25), r.normal(0.0, 1.0, 25)
    calls = {
        "ttest_1samp": lambda: h.ttest_1samp(a, 0.1),
        "ttest_rel": lambda: h.ttest_rel(a, b[:12]),
        "ttest_ind": lambda: h.ttest_ind(a, b),
        "ttest_ind welch less": lambda: h.ttest_ind(a, b, equal_var=False, alternative="less"),
        "ttest_ind trim": lambda: h.ttest_ind(a, b, trim=0.2, alternative="greater"),
        "ttest_ind_from_stats": lambda: h.ttest_ind_from_stats(0.2, 1.0, 12, -0.1, 1.4, 15),
        "f_oneway": lambda: h.f_oneway(a, b, c),
        "f_oneway welch": lambda: h.f_oneway(a, b, c, equal_var=False),
        "alexandergovern": lambda: h.alexandergovern(a, b, c),
        "bartlett": lambda: h.bartlett(a, b, c),
        "levene": lambda: h.levene(a, b, c, center="trimmed"),
        "fligner": lambda: h.fligner(a, b, c),
        "chisquare": lambda: h.chisquare(d["counts"], ddof=1),
        "power_divergence": lambda: h.power_divergence(d["counts"], lambda_="log-likelihood"),
        "skewtest": lambda: h.skewtest(b),
        "kurtosistest": lambda: h.kurtosistest(b, alternative="greater"),
        "normaltest": lambda: h.normaltest(b),
        "jarque_bera": lambda: h.jarque_bera(b),
        "shapiro": lambda: h.shapiro(b),
        "anderson": lambda: h.anderson(d["expo"], dist=sd.stats.Exponential),
        "ks_1samp": lambda: h.ks_1samp(b, N),
        "cramervonmises": lambda: h.cramervonmises(b, N),
        "mannwhitneyu": lambda: h.mannwhitneyu(a, b),
        "ranksums": lambda: h.ranksums(a, b),
        "wilcoxon": lambda: h.wilcoxon(a, b[:12]),
        "kruskal": lambda: h.kruskal(a, b, c),
        "kruskal five samples": lambda: h.kruskal(a, b, c, a[:5], b[:7]),
        "friedmanchisquare": lambda: h.friedmanchisquare(a[:9], b[:9], c),
        "brunnermunzel": lambda: h.brunnermunzel(a, b),
        "ansari": lambda: h.ansari(a, b),
        "mood": lambda: h.mood(a, b),
        "median_test": lambda: h.median_test(a, b, c),
        "page_trend_test": lambda: h.page_trend_test(r.normal(0.0, 1.0, (6, 4))),
        "ks_2samp": lambda: h.ks_2samp(a, b),
        "cramervonmises_2samp": lambda: h.cramervonmises_2samp(a, b),
        "anderson_ksamp": lambda: h.anderson_ksamp([a, b, c]),
        "epps_singleton_2samp": lambda: h.epps_singleton_2samp(a, b),
        "pearsonr": lambda: h.pearsonr(u, v),
        "pearsonr greater": lambda: h.pearsonr(u, v, alternative="greater"),
        "spearmanrho": lambda: h.spearmanrho(u, v),
        "spearmanr matrix": lambda: h.spearmanr(np.c_[u, v, u + v], axis=0),
        "pointbiserialr": lambda: h.pointbiserialr((u > 0).astype(np.float64), v),
        "kendalltau": lambda: h.kendalltau(u, v, variant="c"),
        "weightedtau": lambda: h.weightedtau(u, v),
        "somersd": lambda: h.somersd([[8.0, 3, 1], [2, 9, 4], [0, 3, 7]]),
        "chatterjeexi": lambda: h.chatterjeexi(u, v),
        "chi2_contingency": lambda: h.chi2_contingency([[3.0, 7], [9, 2]]),
        "chi2_contingency lambda": lambda: h.chi2_contingency([[10.0, 12, 8], [7, 15, 9]], lambda_="log-likelihood"),
        "fisher_exact": lambda: h.fisher_exact([[3.0, 7], [9, 2]], alternative="less"),
        "fisher_exact rxc": lambda: h.fisher_exact([[3.0, 1, 4], [1, 5, 9], [2, 6, 5]]),
        "barnard_exact": lambda: h.barnard_exact([[3.0, 7], [9, 2]], pooled=False),
        "boschloo_exact": lambda: h.boschloo_exact([[3.0, 7], [9, 2]]),
        "mcnemar": lambda: h.mcnemar([[30.0, 12], [4, 25]]),
        "mcnemar chi2": lambda: h.mcnemar([[30.0, 12], [4, 25]], exact=False),
        "cochrans_q": lambda: h.cochrans_q(np.eye(4)[[0, 1, 2, 3, 0, 1]]),
        "odds_ratio": lambda: h.odds_ratio([[3.0, 7], [9, 2]]),
        "relative_risk": lambda: h.relative_risk(9, 20, 4, 21),
        "fisher_exact batch": lambda: h.fisher_exact(np.array([[[3.0, 7], [9, 2]], [[5.0, 5], [5, 5]]])),
        "ttest_ind batch": lambda: h.ttest_ind(d["batch_a"], d["batch_b"], axis=-1),
        "ttest_1samp long batch": lambda: h.ttest_1samp(r.normal(0.0, 1.0, (30, 8)), 0.0, axis=-1),
        "mannwhitneyu batch methods": lambda: h.mannwhitneyu(d["cube"], d["cube"][::-1] + 0.3, axis=1),
        "ks_2samp batch": lambda: h.ks_2samp(d["batch_a"], d["batch_b"], axis=-1),
        "median_test batch": lambda: h.median_test(d["batch_a"], d["batch_b"], axis=-1),
        "ttest_1samp keepdims": lambda: h.ttest_1samp(a, 0.1, keepdims=True),
        "ttest_1samp omit": lambda: h.ttest_1samp(d["nan"], 0.0, nan_policy="omit"),
        "binomtest": lambda: h.binomtest(7, 20, p=0.3),
        "binomtest less": lambda: h.binomtest(7, 20, p=0.3, alternative="less"),
        "quantile_test": lambda: h.quantile_test(b, q=0.2, p=0.4),
        "quantile_test batch": lambda: h.quantile_test(d["batch_a"], q=0.1, axis=-1),
        "poisson_means_test": lambda: h.poisson_means_test(10, 5.0, 20, 6.0, diff=0.5),
        "proportions_ztest": lambda: h.proportions_ztest([15, 25], [40, 45]),
        "proportions_ztest one": lambda: h.proportions_ztest(15, 40, value=0.5, prop_var=0.45),
        "proportions_chisquare": lambda: h.proportions_chisquare([15, 25, 9], [40, 45, 30]),
        "proportions_chisquare value": lambda: h.proportions_chisquare([15, 25], [40, 45], value=0.4),
        "test_proportions_2indep": lambda: h.test_proportions_2indep(15, 40, 25, 45),
        "test_proportions_2indep score": lambda: h.test_proportions_2indep(
            15, 40, 25, 45, compare="odds-ratio", method="score", value=2.0
        ),
        "test_proportions_2indep batch": lambda: h.test_proportions_2indep(
            np.array([15.0, 5, 30]), 40, np.array([25.0, 9, 30]), 45
        ),
    }
    for name, call in calls.items():
        yield f"summary {name}", lambda call=call: call().summary()
    yield "summary ttest_ind alpha=0.1", lambda: h.ttest_ind(a, b).summary(alpha=0.1)
    yield "err summary alpha", lambda: h.ttest_ind(a, b).summary(alpha=1.0)
    yield "err summary alpha type", lambda: h.pearsonr(u, v).summary(alpha="0.05")
    yield "err summary alpha without interval", lambda: h.kruskal(a, b).summary(alpha=0.05)
    yield "repr median_test", lambda: repr(h.median_test(a, b, c))
    yield "repr somersd", lambda: repr(h.somersd([[8.0, 3, 1], [2, 9, 4], [0, 3, 7]]))
    yield "repr ttest_1samp keepdims", lambda: repr(h.ttest_1samp(a, 0.1, keepdims=True))
    yield "repr chi2_contingency", lambda: repr(h.chi2_contingency([[3.0, 7], [9, 2]]))
    yield "repr relative_risk", lambda: repr(h.relative_risk(9, 20, 4, 21))
    yield "summary odds_ratio alpha=0.1", lambda: h.odds_ratio([[3.0, 7], [9, 2]]).summary(alpha=0.1)
    yield "repr binomtest", lambda: repr(h.binomtest(7, 20, p=0.3))
    yield "repr proportions_chisquare", lambda: repr(h.proportions_chisquare([15, 25], [40, 45]))
    yield "repr test_proportions_2indep", lambda: repr(h.test_proportions_2indep(15, 40, 25, 45, method="score"))
    yield "summary binomtest alpha=0.1", lambda: h.binomtest(7, 20).summary(alpha=0.1)
