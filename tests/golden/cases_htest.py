"""Golden cases for sd.htest (tools/golden.py has the contract).

Every public test on fixed data sets (small samples, a large offset, ties, constant samples, a
sample of one, NaN under the three nan_policy values, empty samples), over its options
(alternative, equal_var, trim, center and proportiontocut, lambda_ and ddof, dist, method, variant, t,
zero_method, correction, use_continuity, distribution, ties, ranked, predicted_ranks, keepdims),
batches along an axis and a tuple of axes (the rank tests' per-lane methods), long lanes
(n = 10^5: the multithreaded kernels), the exact distributions (with ties, deep tails, past
their work cap), scales of 1e+-300, and the wrong calls (their errors are results). A result
object is stored whole: every field and conf_int(); conf_int(alpha=0.1) as its own case.
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
