"""M5 baseline: eigh, eigvalsh, svd, svdvals, matrix_rank, pinv, cond, matrix_norm(2)
against numpy 2.5.4 (and scipy 1.18.1 for the generalized eigh).
Usage: .venv/bin/python bench/m5_spectral.py  (writes bench/results/m5_spectral.txt)
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
        f"{'func':14} {'shape':>14} {'sdla us':>10} {'other us':>10} {'other/sdla':>11}  (other = numpy, or scipy for eigh(a, b))"
    )
    for shape in [(3, 3), (32, 32), (300, 300), (1000, 1000), (1000, 3, 3), (100, 32, 32)]:
        n = shape[-1]
        a = rng.standard_normal(shape)
        h = (a + np.swapaxes(a, -1, -2)) / 2
        bpd = a @ np.swapaxes(a, -1, -2) + n * np.eye(n)
        A, H, B = sdnp.asarray(a), sdnp.asarray(h), sdnp.asarray(bpd)
        batched = len(shape) > 2
        rows = [
            ("eigh", lambda: sdla.eigh(H), lambda: np.linalg.eigh(h)),
            ("eigvalsh", lambda: sdla.eigvalsh(H), lambda: np.linalg.eigvalsh(h)),
            ("eigh(a, b)", lambda: sdla.eigh(H, B), None if batched else (lambda: sl.eigh(h, bpd))),
            ("svd", lambda: sdla.svd(A), lambda: np.linalg.svd(a, full_matrices=False)),
            ("svdvals", lambda: sdla.svdvals(A), lambda: np.linalg.svdvals(a)),
            ("matrix_rank", lambda: sdla.matrix_rank(A), lambda: np.linalg.matrix_rank(a)),
            ("pinv", lambda: sdla.pinv(A), lambda: np.linalg.pinv(a)),
            ("cond", lambda: sdla.cond(A), lambda: np.linalg.cond(a)),
            ("matrix_norm 2", lambda: sdla.matrix_norm(A, ord=2), lambda: np.linalg.matrix_norm(a, ord=2)),
        ]
        for name, f_sd, f_o in rows:
            if shape == (1000, 1000) and name in ("pinv",):
                pass
            t_sd = best_us(f_sd)
            t_o = best_us(f_o) if f_o else nan
            emit(f"{name:14} {str(shape):>14} {t_sd:10.2f} {t_o:10.2f} {t_o / t_sd:11.2f}")
    path = pathlib.Path(__file__).parent / "results" / "m5_spectral.txt"
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
