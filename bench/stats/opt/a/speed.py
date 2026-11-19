"""Group A (closed-form continuous distributions): ns per element of every method with an
input, one thread, against scipy.stats' classic call on the same 1e5 points (x spread over
the distribution by its quantiles, p uniform; the cases of bench/stats/t4 and t6).

    .venv/bin/python bench/stats/opt/a/speed.py
"""

import math
import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats
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
CASES = [
    (S.Normal, {"mu": 1.5, "sigma": 2.0}, st.norm(1.5, 2.0)),
    (S.LogNormal, {"mu": 0.5, "sigma": 0.8}, st.lognorm(0.8, scale=math.exp(0.5))),
    (S.Exponential, {"scale": 2.0}, st.expon(scale=2.0)),
    (S.Uniform, {"a": -1.0, "b": 3.0}, st.uniform(-1.0, 4.0)),
    (S.Laplace, {"mu": 0.5, "sigma": 2.0}, st.laplace(0.5, 2.0)),
    (S.Cauchy, {"mu": 0.5, "sigma": 2.0}, st.cauchy(0.5, 2.0)),
    (S.Weibull, {"c": 1.7, "scale": 2.0}, st.weibull_min(1.7, scale=2.0)),
    (S.Logistic, {"mu": 0.5, "sigma": 2.0}, st.logistic(0.5, 2.0)),
    (S.Gumbel, {"mu": 0.5, "sigma": 2.0}, st.gumbel_r(0.5, 2.0)),
    (S.GEV, {"c": 0.25, "mu": 1.0, "sigma": 2.0}, st.genextreme(0.25, 1.0, 2.0)),
    (S.Pareto, {"b": 3.0, "scale": 1.0}, st.pareto(3.0, scale=1.0)),
    (S.Rayleigh, {"scale": 2.0}, st.rayleigh(scale=2.0)),
]


def best(f, n):
    return min(timeit.repeat(f, number=3, repeat=7)) / 3 / n * 1e9


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    n = 100_000
    p = rng.uniform(0, 1, n)
    before = sd.get_num_threads()
    sd.set_num_threads(1)
    print("| distribution | method | sd 1T (ns) | scipy (ns) | scipy / sd |")
    print("|---|---|---:|---:|---:|")
    try:
        for cls, params, ref in CASES:
            X = cls(**params)
            x = ref.ppf(p)
            sx, sp, slp = sd.asarray(x), sd.asarray(p), sd.asarray(np.log(p))
            for m in ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf"):
                arg = slp if m.startswith("ilog") else (sp if m in ("icdf", "iccdf") else sx)
                f = getattr(X, m)
                t = best(lambda: f(arg), n)
                if m in SCIPY:
                    g = getattr(ref, SCIPY[m])
                    npa = p if m in ("icdf", "iccdf") else x
                    ts = best(lambda: g(npa), n)
                    print(f"| {cls.__name__} | `{m}` | {t:.1f} | {ts:.1f} | {ts / t:.2f} |", flush=True)
                else:
                    print(f"| {cls.__name__} | `{m}` | {t:.1f} | — | — |", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
