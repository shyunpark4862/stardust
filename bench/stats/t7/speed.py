"""T7: speed of X.sample against numpy's Generator (PCG64; scipy.stats' rvs calls it), 10^6
samples, 1 and 10 threads, and one Python float per call.

    .venv/bin/python bench/stats/t7/speed.py > bench/stats/t7/speed.md
"""

import timeit

import numpy as np

import stardust as sd

S = sd.stats
CASES = [
    ("Normal", {}, lambda g, n: g.standard_normal(n)),
    ("LogNormal", {}, lambda g, n: g.lognormal(size=n)),
    ("Exponential", {}, lambda g, n: g.standard_exponential(n)),
    ("Uniform", dict(a=0.0, b=1.0), lambda g, n: g.random(n)),
    ("Laplace", {}, lambda g, n: g.laplace(size=n)),
    ("Cauchy", {}, lambda g, n: g.standard_cauchy(n)),
    ("Logistic", {}, lambda g, n: g.logistic(size=n)),
    ("Gumbel", {}, lambda g, n: g.gumbel(size=n)),
    ("GEV", dict(c=0.1), None),
    ("Weibull", dict(c=2.0), lambda g, n: g.weibull(2.0, n)),
    ("Pareto", dict(b=3.0), lambda g, n: g.pareto(3.0, n) + 1.0),
    ("Rayleigh", {}, lambda g, n: g.rayleigh(size=n)),
    ("Gamma", dict(a=3.0), lambda g, n: g.standard_gamma(3.0, n)),
    ("Gamma", dict(a=0.5), lambda g, n: g.standard_gamma(0.5, n)),
    ("ChiSquared", dict(df=5.0), lambda g, n: g.chisquare(5.0, n)),
    ("InverseGamma", dict(a=3.0), None),
    ("Beta", dict(a=2.0, b=3.0), lambda g, n: g.beta(2.0, 3.0, n)),
    ("Beta", dict(a=0.5, b=0.5), lambda g, n: g.beta(0.5, 0.5, n)),
    ("StudentT", dict(df=5.0), lambda g, n: g.standard_t(5.0, n)),
    ("F", dict(dfn=5.0, dfd=7.0), lambda g, n: g.f(5.0, 7.0, n)),
    ("NoncentralChiSquared", dict(df=5.0, nc=2.0), lambda g, n: g.noncentral_chisquare(5.0, 2.0, n)),
    ("NoncentralT", dict(df=5.0, nc=1.0), None),
    ("NoncentralF", dict(dfn=5.0, dfd=7.0, nc=2.0), lambda g, n: g.noncentral_f(5.0, 7.0, 2.0, n)),
    ("Bernoulli", dict(p=0.3), lambda g, n: g.binomial(1, 0.3, n)),
    ("DiscreteUniform", dict(low=0.0, high=10.0), lambda g, n: g.integers(0, 10, n)),
    ("Geometric", dict(p=0.1), lambda g, n: g.geometric(0.1, n)),
    ("Poisson", dict(mu=3.0), lambda g, n: g.poisson(3.0, n)),
    ("Poisson", dict(mu=100.0), lambda g, n: g.poisson(100.0, n)),
    ("Binomial", dict(n=20.0, p=0.3), lambda g, n: g.binomial(20, 0.3, n)),
    ("Binomial", dict(n=1e4, p=0.3), lambda g, n: g.binomial(10000, 0.3, n)),
    ("NegativeBinomial", dict(n=5.0, p=0.3), lambda g, n: g.negative_binomial(5.0, 0.3, n)),
    (
        "Hypergeometric",
        dict(total=1000.0, successes=300.0, draws=100.0),
        lambda g, n: g.hypergeometric(300, 700, 100, n),
    ),
    ("BetaBinomial", dict(n=20.0, a=2.0, b=3.0), None),
    ("LogSeries", dict(p=0.9), lambda g, n: g.logseries(0.9, n)),
    ("Zipf", dict(a=2.0), lambda g, n: g.zipf(2.0, n)),
]
N = 10**6


def best(f, repeat=5):
    return min(timeit.repeat(f, number=1, repeat=repeat))


def main():
    before = sd.get_num_threads()
    print(f"Arrays of {N:,} samples; ms. numpy: Generator(PCG64), the same parameters.\n")
    print("| distribution | parameters | sd 1T | sd 10T | numpy | numpy / sd 1T | numpy / sd 10T |")
    print("|---|---|---:|---:|---:|---:|---:|")
    scal = []
    try:
        for name, kw, npf in CASES:
            D = getattr(S, name)(**kw)
            rng = sd.random.default_rng(1)
            sd.set_num_threads(1)
            t1 = best(lambda: D.sample(N, rng=rng))
            sd.set_num_threads(10)
            t10 = best(lambda: D.sample(N, rng=rng))
            g = np.random.default_rng(1)
            tn = best(lambda: npf(g, N)) if npf else float("nan")
            r1 = f"{tn / t1:.2f}" if npf else "—"
            r10 = f"{tn / t10:.1f}" if npf else "—"
            tns = f"{tn * 1e3:.2f}" if npf else "—"
            params = ", ".join(f"{k}={v:g}" for k, v in kw.items())
            print(f"| {name} | {params} | {t1 * 1e3:.2f} | {t10 * 1e3:.2f} | {tns} | {r1} | {r10} |", flush=True)
            one = timeit.timeit(lambda: D.sample(rng=rng), number=20000) / 20000
            onen = timeit.timeit(lambda: npf(g, None), number=20000) / 20000 if npf else float("nan")
            scal.append((name, params, one, onen))
    finally:
        sd.set_num_threads(before)
    print("\nOne Python float per call (µs):\n")
    print("| distribution | parameters | sd | numpy | numpy / sd |")
    print("|---|---|---:|---:|---:|")
    for name, params, one, onen in scal:
        rn = f"{onen / one:.2f}" if onen == onen else "—"
        on = f"{onen * 1e6:.2f}" if onen == onen else "—"
        print(f"| {name} | {params} | {one * 1e6:.2f} | {on} | {rn} |")


if __name__ == "__main__":
    main()
