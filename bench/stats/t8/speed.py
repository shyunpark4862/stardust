"""T8: speed of the descriptive statistics against scipy.stats, 10^6 float64 (gamma(2) data),
1 and 10 threads, over the whole array and along axis 0 of a (1000, 1000) array.

    .venv/bin/python bench/stats/t8/speed.py > bench/stats/t8/speed.md
"""

import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats


def best(f, repeat=5):
    return min(timeit.repeat(f, number=1, repeat=repeat))


CASES = [
    ("skew", lambda X, ax: S.skew(X, axis=ax), lambda x, ax: st.skew(x, axis=ax)),
    ("kurtosis", lambda X, ax: S.kurtosis(X, axis=ax), lambda x, ax: st.kurtosis(x, axis=ax)),
    ("describe", lambda X, ax: S.describe(X, axis=ax), lambda x, ax: st.describe(x, axis=ax)),
    ("moment(3)", lambda X, ax: S.moment(X, order=3, axis=ax), lambda x, ax: st.moment(x, order=3, axis=ax)),
    ("sem", lambda X, ax: S.sem(X, axis=ax), lambda x, ax: st.sem(x, axis=ax)),
    ("zscore", lambda X, ax: S.zscore(X, axis=ax), lambda x, ax: st.zscore(x, axis=ax)),
    ("rankdata", lambda X, ax: S.rankdata(X, axis=ax), lambda x, ax: st.rankdata(x, axis=ax)),
    ("iqr", lambda X, ax: S.iqr(X, axis=ax), lambda x, ax: st.iqr(x, axis=ax)),
    ("mode", lambda X, ax: S.mode(X, axis=ax), lambda x, ax: st.mode(x, axis=ax)),
    ("entropy", lambda X, ax: S.entropy(X, axis=ax), lambda x, ax: st.entropy(x, axis=ax)),
    ("differential_entropy", lambda X, ax: S.differential_entropy(X, axis=ax),
     lambda x, ax: st.differential_entropy(x, axis=ax)),
    ("skew omit", lambda X, ax: S.skew(X, axis=ax, nan_policy="omit"),
     lambda x, ax: st.skew(x, axis=ax, nan_policy="omit")),
]


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    x1 = rng.gamma(2.0, size=10**6)
    x2 = x1.reshape(1000, 1000)
    xm = np.round(x1 * 4).reshape(1000, 1000)  # for mode: repeated values
    before = sd.get_num_threads()
    print("10^6 float64; ms. 'all': axis=None over a vector; 'axis 0': a (1000, 1000) array.\n")
    print("| function | data | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |")
    print("|---|---|---:|---:|---:|---:|---:|")
    try:
        for name, f, g in CASES:
            for label, x, ax in (("all", x1, None), ("axis 0", x2, 0)):
                if name == "mode":
                    x = xm.reshape(-1) if ax is None else xm
                if name == "skew omit":
                    x = x.copy()
                    x.reshape(-1)[::1000] = np.nan
                X = sd.asarray(x)
                sd.set_num_threads(1)
                t1 = best(lambda: f(X, ax))
                sd.set_num_threads(10)
                t10 = best(lambda: f(X, ax))
                ts = best(lambda: g(x, ax), repeat=3)
                print(f"| {name} | {label} | {t1 * 1e3:.2f} | {t10 * 1e3:.2f} | {ts * 1e3:.2f} | {ts / t1:.2f} | "
                      f"{ts / t10:.1f} |", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
