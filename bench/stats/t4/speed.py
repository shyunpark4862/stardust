"""T4: speed of the 12 continuous distributions against scipy.stats (1e5 points, 1 and 10 threads).

scipy is timed in its classic form (scipy.stats.<name>.<method>(x, params), the fast path
underneath its new-style classes) with the matching method names (sf for ccdf, ppf for
icdf, isf for iccdf); its new-style objects call the same functions with more Python
around them. ilogcdf and ilogccdf have no classic form: scipy.stats' new-style
Normal / make_distribution objects are timed for them. Scalar calls: one Python float.

    .venv/bin/python bench/stats/t4/speed.py > bench/stats/t4/speed.md
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


SCIPY = {
    "pdf": "pdf",
    "logpdf": "logpdf",
    "cdf": "cdf",
    "logcdf": "logcdf",
    "ccdf": "sf",
    "logccdf": "logsf",
    "icdf": "ppf",
    "iccdf": "isf",
}

# (class, parameters, scipy frozen classic distribution)
CASES = [
    (S.Normal, {"mu": 1.5, "sigma": 2.0}, st.norm(1.5, 2.0)),
    (S.StudentT, {"df": 3.5}, st.t(3.5)),
    (S.ChiSquared, {"df": 4.5}, st.chi2(4.5)),
    (S.F, {"dfn": 3.0, "dfd": 11.0}, st.f(3.0, 11.0)),
    (S.Gamma, {"a": 2.5, "scale": 3.0}, st.gamma(2.5, scale=3.0)),
    (S.Beta, {"a": 2.5, "b": 0.7}, st.beta(2.5, 0.7)),
    (S.Exponential, {"scale": 2.0}, st.expon(scale=2.0)),
    (S.LogNormal, {"mu": 0.5, "sigma": 0.8}, st.lognorm(0.8, scale=math.exp(0.5))),
    (S.Uniform, {"a": -1.0, "b": 3.0}, st.uniform(-1.0, 4.0)),
    (S.Cauchy, {"mu": 0.5, "sigma": 2.0}, st.cauchy(0.5, 2.0)),
    (S.Laplace, {"mu": 0.5, "sigma": 2.0}, st.laplace(0.5, 2.0)),
    (S.Weibull, {"c": 1.7, "scale": 2.0}, st.weibull_min(1.7, scale=2.0)),
]


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    N = 100_000
    p = rng.uniform(0, 1, N)
    print(f"Arrays of {N} points (x spread over the distribution by its quantiles); times in ms.\n")
    print("| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |")
    print("|---|---|---:|---:|---:|---:|---:|")
    before = sd.get_num_threads()
    scalar_rows = []
    try:
        for cls, params, ref in CASES:
            X = cls(**params)
            x = ref.ppf(p)
            sx, sp = sd.asarray(x), sd.asarray(p)
            for m in ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf"):
                arg = sp if m in ("icdf", "iccdf") else sx
                npa = p if m in ("icdf", "iccdf") else x
                f = getattr(X, m)
                sd.set_num_threads(1)
                t1 = best(lambda: f(arg))
                sd.set_num_threads(10)
                t10 = best(lambda: f(arg))
                g = getattr(ref, SCIPY[m])
                ts = best(lambda: g(npa))
                print(
                    f"| {cls.__name__} | `{m}` | {t1:.2f} | {t10:.2f} | {ts:.2f} | {ts / t1:.2f} | {ts / t10:.1f} |",
                    flush=True,
                )
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
