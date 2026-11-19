"""M3 baseline: solve_triangular, cholesky, cholesky_solve, lu, lu_solve
against numpy 2.5.4 and scipy 1.18.1 (all on Accelerate LAPACK).

scipy's cho_solve and lu_solve take their packed factor formats
((c, lower) and (lu, piv)); they are timed with those, which is the work
the same call does there. Usage: .venv/bin/python bench/m3_factor.py
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
        if t > 0.02 or n > 1_000_000:
            break
        n *= 4
    return min(timeit.repeat(fn, number=n, repeat=7)) / n * 1e6


def main():
    rng = np.random.default_rng(0)
    out = []

    def emit(s):
        print(s)
        out.append(s)

    emit(f"numpy {np.__version__}, scipy {__import__('scipy').__version__}, sdnp threads {sdnp.get_num_threads()}")
    emit(f"{'func':16} {'shape':>14} {'sdla us':>10} {'numpy us':>10} {'scipy us':>10} {'np/sdla':>8} {'sp/sdla':>8}")
    nan = float("nan")
    for shape in [(3, 3), (16, 16), (256, 256), (1024, 1024), (1000, 3, 3), (100, 32, 32)]:
        n = shape[-1]
        a = rng.standard_normal(shape)
        s = a @ np.swapaxes(a, -1, -2) + n * np.eye(n)
        t = np.triu(a) + n * np.eye(n)
        b = rng.standard_normal((*shape[:-1], 1))
        A, S, T, B = (sdnp.asarray(np.ascontiguousarray(v)) for v in (a, s, t, b))
        L = sdla.cholesky(S)
        LU = sdla.lu(A)
        cf = sl.cho_factor(s, lower=True)
        lf = sl.lu_factor(a)
        rows = [
            (
                "solve_triangular",
                lambda: sdla.solve_triangular(T, B, upper=True),
                None,
                lambda: sl.solve_triangular(t, b, lower=False),
            ),
            ("cholesky", lambda: sdla.cholesky(S), lambda: np.linalg.cholesky(s), lambda: sl.cholesky(s, lower=True)),
            ("cholesky_solve", lambda: sdla.cholesky_solve(L, B), None, lambda: sl.cho_solve(cf, b)),
            ("lu", lambda: sdla.lu(A), None, lambda: sl.lu(a, p_indices=True)),
            ("lu_solve", lambda: sdla.lu_solve(*LU, B), None, lambda: sl.lu_solve(lf, b)),
        ]
        for name, f_sd, f_np, f_sp in rows:
            t_sd = best_us(f_sd)
            t_np = best_us(f_np) if f_np else nan
            t_sp = best_us(f_sp) if f_sp else nan
            emit(
                f"{name:16} {str(shape):>14} {t_sd:10.2f} {t_np:10.2f} {t_sp:10.2f} {t_np / t_sd:8.2f} {t_sp / t_sd:8.2f}"
            )
    path = pathlib.Path(__file__).parent / "results" / "m3_factor.txt"
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
