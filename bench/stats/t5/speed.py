"""T5: speed of the 10 discrete distributions against scipy.stats (1e5 points, 1 and 10 threads).

scipy is timed in its classic form (scipy.stats.<name>.<method>(x, params), the fast path
underneath its new-style classes) with the matching method names (sf for ccdf, ppf for
icdf, isf for iccdf); its new-style objects call the same functions with more Python
around them. ilogcdf and ilogccdf have no classic form and are not timed. Scalar calls:
one Python float.

    .venv/bin/python bench/stats/t5/speed.py > bench/stats/t5/speed.md
"""

import math
import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats


def best(f, n=1):
    return min(timeit.repeat(f, number=n, repeat=5)) / n * 1e3


SCIPY = {"pmf": "pmf", "logpmf": "logpmf", "cdf": "cdf", "logcdf": "logcdf", "ccdf": "sf", "logccdf": "logsf",
         "icdf": "ppf", "iccdf": "isf"}

# (class, parameters, scipy frozen classic distribution)
CASES = [
    (S.Binomial, {"n": 40, "p": 0.3}, st.binom(40, 0.3)),
    (S.Poisson, {"mu": 4.5}, st.poisson(4.5)),
    (S.Geometric, {"p": 0.2}, st.geom(0.2)),
    (S.NegativeBinomial, {"n": 2.5, "p": 0.3}, st.nbinom(2.5, 0.3)),
    (S.Hypergeometric, {"total": 500, "successes": 120, "draws": 200}, st.hypergeom(500, 120, 200)),
    (S.Bernoulli, {"p": 0.3}, st.bernoulli(0.3)),
    (S.DiscreteUniform, {"low": 2, "high": 90}, st.randint(2, 90)),
    (S.BetaBinomial, {"n": 40, "a": 2.5, "b": 1.7}, st.betabinom(40, 2.5, 1.7)),
    (S.LogSeries, {"p": 0.7}, st.logser(0.7)),
    (S.Zipf, {"a": 2.5}, st.zipf(2.5)),
]


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    N = 100_000
    p = rng.uniform(0, 1, N)
    print(f"Arrays of {N} points (k spread over the distribution by its quantiles); times in ms.\n")
    print("| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |")
    print("|---|---|---:|---:|---:|---:|---:|")
    before = sd.get_num_threads()
    scalar_rows = []
    try:
        for cls, params, ref in CASES:
            X = cls(**params)
            x = np.asarray(X.icdf(sd.asarray(p)))  # scipy's zipf ppf stops with "endless loop"
            sx, sp = sd.asarray(x), sd.asarray(p)
            for m in ("pmf", "logpmf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf"):
                arg = sp if m in ("icdf", "iccdf") else sx
                npa = p if m in ("icdf", "iccdf") else x
                f = getattr(X, m)
                sd.set_num_threads(1)
                t1 = best(lambda: f(arg))
                sd.set_num_threads(10)
                t10 = best(lambda: f(arg))
                g = getattr(ref, SCIPY[m])
                try:
                    ts = best(lambda: g(npa))
                except RuntimeError:  # scipy's zipf ppf: "updating stopped, endless loop"
                    print(f"| {cls.__name__} | `{m}` | {t1:.2f} | {t10:.2f} | — | — | — |", flush=True)
                    continue
                print(f"| {cls.__name__} | `{m}` | {t1:.2f} | {t10:.2f} | {ts:.2f} | {ts / t1:.2f} | {ts / t10:.1f} |",
                      flush=True)
            v = float(x[N // 3])
            ts1 = best(lambda: X.cdf(v), 2000)
            tsc = best(lambda: ref.cdf(v), 2000)
            scalar_rows.append((cls.__name__, ts1 * 1e3, tsc * 1e3))
    finally:
        sd.set_num_threads(before)
    print("\nOne Python float (cdf), in microseconds:\n")
    print("| distribution | sd | scipy | scipy / sd |")
    print("|---|---:|---:|---:|")
    for name, a, b in scalar_rows:
        print(f"| {name} | {a:.2f} | {b:.2f} | {b / a:.1f} |")


if __name__ == "__main__":
    main()
