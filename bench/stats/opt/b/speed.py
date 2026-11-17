"""Group B (gamma, chi-square, inverse gamma, beta, t, F): single-thread ns per element against
scipy.stats, at the parameters of bench/stats/t4/speed.py and t6/speed.py and at a second set
(integer chi-square df, larger shapes) that takes the kernels' other branches.

    .venv/bin/python bench/stats/opt/b/speed.py
"""

import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats
SCIPY = {"pdf": "pdf", "logpdf": "logpdf", "cdf": "cdf", "logcdf": "logcdf", "ccdf": "sf", "logccdf": "logsf",
         "icdf": "ppf", "iccdf": "isf"}
CASES = [
    (S.Gamma, {"a": 2.5, "scale": 3.0}, st.gamma(2.5, scale=3.0)),
    (S.Gamma, {"a": 7.3, "scale": 1.0}, st.gamma(7.3)),
    (S.ChiSquared, {"df": 4.5}, st.chi2(4.5)),
    (S.ChiSquared, {"df": 7.0}, st.chi2(7.0)),
    (S.InverseGamma, {"a": 4.5, "scale": 2.0}, st.invgamma(4.5, scale=2.0)),
    (S.Beta, {"a": 2.5, "b": 0.7}, st.beta(2.5, 0.7)),
    (S.Beta, {"a": 2.0, "b": 3.0}, st.beta(2.0, 3.0)),
    (S.StudentT, {"df": 3.5}, st.t(3.5)),
    (S.F, {"dfn": 3.0, "dfd": 11.0}, st.f(3.0, 11.0)),
]


def best(f):
    return min(timeit.repeat(f, number=1, repeat=7))


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    n = 100_000
    p = rng.uniform(0, 1, n)
    before = sd.get_num_threads()
    sd.set_num_threads(1)
    print("| distribution | method | sd ns | scipy ns | scipy / sd |")
    print("|---|---|---:|---:|---:|")
    try:
        for cls, params, ref in CASES:
            X = cls(**params)
            x = ref.ppf(p)
            sx, sp = sd.asarray(x), sd.asarray(p)
            name = f"{cls.__name__}({', '.join(str(v) for v in params.values())})"
            for m in SCIPY:
                arg, npa = (sp, p) if m in ("icdf", "iccdf") else (sx, x)
                f, g = getattr(X, m), getattr(ref, SCIPY[m])
                t1, ts = best(lambda: f(arg)) / n * 1e9, best(lambda: g(npa)) / n * 1e9
                print(f"| {name} | `{m}` | {t1:.1f} | {ts:.1f} | {ts / t1:.2f} |", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
