"""M6 baseline: eig, eigvals (vs numpy 2.5.4); schur, hessenberg, ldl (vs scipy 1.18.1).
Usage: .venv/bin/python bench/m6_nonsym.py  (writes bench/results/m6_nonsym.txt)
"""

import pathlib
import timeit

import numpy as np
import scipy.linalg as sl
import stardust_numpy as sdnp

import stardust_linalg as sdla


def best_us(fn):
    fn()
    n = 1
    while True:
        t = timeit.timeit(fn, number=n)
        if t > 0.05 or n > 1_000_000:
            break
        n *= 4
    return min(timeit.repeat(fn, number=n, repeat=5)) / n * 1e6


def main():
    rng = np.random.default_rng(0)
    out = []

    def emit(s):
        print(s)
        out.append(s)

    nan = float("nan")
    emit(f"numpy {np.__version__}, scipy {__import__('scipy').__version__}")
    emit(
        f"{'func':11} {'shape':>14} {'sdla us':>11} {'other us':>11} {'other/sdla':>11}  (other: numpy for eig/eigvals, scipy for the rest)"
    )
    for shape in [(3, 3), (32, 32), (300, 300), (1000, 1000), (1000, 3, 3), (100, 32, 32)]:
        a = rng.standard_normal(shape)
        s = a + np.swapaxes(a, -1, -2)
        A, S = sdnp.asarray(a), sdnp.asarray(s)
        batched = len(shape) > 2
        rows = [
            ("eig", lambda: sdla.eig(A), lambda: np.linalg.eig(a)),
            ("eigvals", lambda: sdla.eigvals(A), lambda: np.linalg.eigvals(a)),
            ("schur", lambda: sdla.schur(A), lambda: sl.schur(a)),
            ("hessenberg", lambda: sdla.hessenberg(A), lambda: sl.hessenberg(a, calc_q=True)),
            ("ldl", lambda: sdla.ldl(S), None if batched else (lambda: sl.ldl(s))),
        ]
        for name, f_sd, f_o in rows:
            t_sd = best_us(f_sd)
            t_o = best_us(f_o) if f_o else nan
            emit(f"{name:11} {str(shape):>14} {t_sd:11.2f} {t_o:11.2f} {t_o / t_sd:11.2f}")
    path = pathlib.Path(__file__).parent / "results" / "m6_nonsym.txt"
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
