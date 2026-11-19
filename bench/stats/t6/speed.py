"""T6: speed of the extreme value, lifetime and noncentral distributions against scipy.stats
(1e5 points, 1 and 10 threads).

scipy is timed in its classic form (scipy.stats.<name>.<method>(x, params), the fast path
underneath its new-style classes) with the matching method names (sf for ccdf, ppf for
icdf, isf for iccdf); its new-style objects call the same functions with more Python
around them. ilogcdf and ilogccdf have no classic form and are not timed. A call slower
than a second is timed once. The points are sd's own quantiles (scipy's noncentral ppf
takes minutes for 1e5). Scalar calls: one Python float.

    .venv/bin/python bench/stats/t6/speed.py > bench/stats/t6/speed.md
"""

import math
import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats


def best(f, n=1):
    first = timeit.timeit(f, number=n)
    if first > 1.0:
        return first / n * 1e3
    return min([first] + timeit.repeat(f, number=n, repeat=4)) / n * 1e3


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
    (S.Logistic, {"mu": 0.5, "sigma": 2.0}, st.logistic(0.5, 2.0)),
    (S.Gumbel, {"mu": 0.5, "sigma": 2.0}, st.gumbel_r(0.5, 2.0)),
    (S.GEV, {"c": 0.25, "mu": 1.0, "sigma": 2.0}, st.genextreme(0.25, 1.0, 2.0)),
    (S.Pareto, {"b": 2.5, "scale": 1.5}, st.pareto(2.5, scale=1.5)),
    (S.Rayleigh, {"scale": 2.0}, st.rayleigh(scale=2.0)),
    (S.InverseGamma, {"a": 4.5, "scale": 2.0}, st.invgamma(4.5, scale=2.0)),
    (S.NoncentralChiSquared, {"df": 3.5, "nc": 2.0}, st.ncx2(3.5, 2.0)),
    (S.NoncentralT, {"df": 9.0, "nc": 1.5}, st.nct(9.0, 1.5)),
    (S.NoncentralF, {"dfn": 4.0, "dfd": 13.0, "nc": 3.0}, st.ncf(4.0, 13.0, 3.0)),
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
            x = np.asarray(X.icdf(sd.asarray(p)))
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
