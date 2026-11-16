"""T10: speed of the matrix distributions' logpdf and samples against scipy.stats, 10^4 matrices,
1 and 10 threads.

    .venv/bin/python bench/stats/t10/speed.py > bench/stats/t10/speed.md
"""

import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats
N = 10**4


def best(f, repeat=3):
    return min(timeit.repeat(f, number=1, repeat=repeat))


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    rows = []
    for p in (3, 10):
        a = rng.normal(size=(p, p))
        V = a @ a.T + p * np.eye(p)
        df = p + 2.5
        xw = st.wishart(df, V).rvs(N, random_state=1)
        rows.append((f"Wishart p={p}", S.Wishart(df=df, scale=sd.asarray(V)), st.wishart(df, V), xw,
                     lambda Sp, x: Sp.logpdf(np.moveaxis(x, 0, -1))))
        xi = st.invwishart(df, V).rvs(N, random_state=1)
        rows.append((f"InverseWishart p={p}", S.InverseWishart(df=df, scale=sd.asarray(V)), st.invwishart(df, V), xi,
                     lambda Sp, x: Sp.logpdf(np.moveaxis(x, 0, -1))))
    for n, p in ((3, 2), (10, 5)):
        M = rng.normal(size=(n, p))
        U, V = np.eye(n) + 0.3, np.eye(p) + 0.2
        Sp = st.matrix_normal(M, U, V)
        rows.append((f"MatrixNormal {n}x{p}", S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U),
                                                             colcov=sd.asarray(V)), Sp, Sp.rvs(N, random_state=1),
                     lambda Sp, x: Sp.logpdf(x)))
    before = sd.get_num_threads()
    print(f"{N:,} matrices; ms.\n")
    print("| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |")
    print("|---|---|---:|---:|---:|---:|---:|")
    try:
        for name, D, Sp, x, g in rows:
            X = sd.asarray(x)
            sd.set_num_threads(1)
            t1 = best(lambda: D.logpdf(X))
            sd.set_num_threads(10)
            t10 = best(lambda: D.logpdf(X))
            ts = best(lambda: g(Sp, x))
            print(f"| {name} | logpdf | {t1 * 1e3:.2f} | {t10 * 1e3:.2f} | {ts * 1e3:.2f} | {ts / t1:.2f} | {ts / t10:.1f} |",
                  flush=True)
            r = sd.random.default_rng(1)
            sd.set_num_threads(1)
            t1 = best(lambda: D.sample(N, rng=r))
            sd.set_num_threads(10)
            t10 = best(lambda: D.sample(N, rng=r))
            ts = best(lambda: Sp.rvs(N, random_state=1))
            print(f"| {name} | sample | {t1 * 1e3:.2f} | {t10 * 1e3:.2f} | {ts * 1e3:.2f} | {ts / t1:.2f} | {ts / t10:.1f} |",
                  flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
