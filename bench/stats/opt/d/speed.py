"""The noncentral three against scipy.stats (1e5 points, one thread; ns per element), the
rows of bench/stats/t6/speed.py plus wider parameters and the parameter-only methods.

    .venv/bin/python bench/stats/opt/d/speed.py
"""

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
    (S.NoncentralChiSquared, {"df": 3.5, "nc": 2.0}, st.ncx2(3.5, 2.0)),
    (S.NoncentralT, {"df": 9.0, "nc": 1.5}, st.nct(9.0, 1.5)),
    (S.NoncentralF, {"dfn": 4.0, "dfd": 13.0, "nc": 3.0}, st.ncf(4.0, 13.0, 3.0)),
    (S.NoncentralChiSquared, {"df": 40.0, "nc": 25.0}, st.ncx2(40.0, 25.0)),
    (S.NoncentralT, {"df": 40.0, "nc": 4.0}, st.nct(40.0, 4.0)),
    (S.NoncentralF, {"dfn": 20.0, "dfd": 60.0, "nc": 25.0}, st.ncf(20.0, 60.0, 25.0)),
]


def best(f, n=1):
    first = timeit.timeit(f, number=n)
    if first > 1.0:
        return first / n
    return min([first] + timeit.repeat(f, number=n, repeat=4)) / n


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    N = 100_000
    p = rng.uniform(0, 1, N)
    sd.set_num_threads(1)
    print("| distribution | method | sd ns | scipy ns | scipy / sd |")
    print("|---|---|---:|---:|---:|")
    for cls, params, ref in CASES:
        X = cls(**params)
        x = np.asarray(X.icdf(sd.asarray(p)))
        sx, sp = sd.asarray(x), sd.asarray(p)
        label = cls.__name__ + str(tuple(params.values()))
        for m in ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf"):
            arg, npa = (sp, p) if m in ("icdf", "iccdf") else (sx, x)
            f, g = getattr(X, m), getattr(ref, SCIPY[m])
            t1 = best(lambda: f(arg)) / N * 1e9
            n2 = 2000 if m in ("icdf", "iccdf") else N  # scipy's inverses: seconds per 1e5
            ts = best(lambda: g(npa[:n2])) / n2 * 1e9
            print(f"| {label} | `{m}` | {t1:.0f} | {ts:.0f} | {ts / t1:.2f} |", flush=True)
        for m in ("mean", "variance", "skewness", "kurtosis", "median", "mode", "entropy"):
            f = getattr(X, m)
            t1 = best(f, 20) * 1e6
            print(f"| {label} | `{m}` (one call, us) | {t1:.1f} | | |", flush=True)


if __name__ == "__main__":
    main()
