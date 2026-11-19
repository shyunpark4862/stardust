"""T1: speed of the distribution functions against scipy (1e6 points, 1 and 10 threads).

.venv/bin/python bench/special/t1/speed.py > bench/special/t1/speed.md
"""

import timeit
import warnings

import numpy as np
import scipy.special as sc

import stardust as sd


def best(f, n=3):
    return min(timeit.repeat(f, number=n, repeat=5)) / n * 1e3


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    N = 1_000_000
    p = rng.uniform(0, 1, N)
    df = np.exp(rng.uniform(-1, 4, N))
    t = rng.standard_normal(N) * 3
    v, x = np.exp(rng.uniform(-1, 4, N)), np.exp(rng.uniform(-2, 4, N))
    d1, d2, fx = np.exp(rng.uniform(-1, 4, N)), np.exp(rng.uniform(-1, 4, N)), np.exp(rng.uniform(-2, 2, N))
    k = np.floor(np.exp(rng.uniform(0, 4, N)))
    m = np.exp(rng.uniform(-1, 4, N))
    n = k + np.floor(np.exp(rng.uniform(0, 4, N)))
    pp = rng.uniform(0, 1, N)
    y = -np.exp(rng.uniform(-5, 5, N))
    cases = [
        ("stdtr", (df, t)),
        ("stdtrit", (df, p)),
        ("chdtr", (v, x)),
        ("chdtrc", (v, x)),
        ("chdtri", (v, p)),
        ("fdtr", (d1, d2, fx)),
        ("fdtrc", (d1, d2, fx)),
        ("fdtri", (d1, d2, p)),
        ("pdtr", (k, m)),
        ("pdtrc", (k, m)),
        ("bdtr", (k, n, pp)),
        ("bdtrc", (k, n, pp)),
        ("nbdtr", (k, n, pp)),
        ("nbdtrc", (k, n, pp)),
        ("ndtri_exp", (y,)),
    ]
    print("| function | sd 1T (ms) | sd 10T (ms) | scipy (ms) | scipy / sd 1T | scipy / sd 10T |")
    print("|---|---:|---:|---:|---:|---:|")
    before = sd.get_num_threads()
    try:
        for name, args in cases:
            sargs = [sd.asarray(a) for a in args]
            f = getattr(sd.special, name)
            sd.set_num_threads(1)
            t1 = best(lambda: f(*sargs))
            sd.set_num_threads(10)
            t10 = best(lambda: f(*sargs))
            ts = best(lambda: getattr(sc, name)(*args))
            print(f"| `{name}` | {t1:.1f} | {t10:.1f} | {ts:.1f} | {ts / t1:.2f} | {ts / t10:.1f} |", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
