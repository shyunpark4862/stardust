"""M4 baseline: qr, qr_pivoted, lstsq against numpy 2.5.4 and scipy 1.18.1.

numpy's lstsq is not batched, so batched shapes compare with scipy only.
Usage: .venv/bin/python bench/m4_qr.py  (writes bench/results/m4_qr.txt)
"""

import pathlib
import timeit

import numpy as np
import scipy.linalg as sl

import stardust_linalg as sdla
import stardust_numpy as sdnp


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

    nan = float("nan")
    emit(f"numpy {np.__version__}, scipy {__import__('scipy').__version__}")
    emit(f"{'func':18} {'shape':>14} {'sdla us':>10} {'numpy us':>10} {'scipy us':>10} {'np/sdla':>8} {'sp/sdla':>8}")
    for shape in [(3, 3), (10, 3), (64, 64), (1000, 100), (500, 500), (1000, 1000), (1000, 3, 3), (100, 50, 10)]:
        a = rng.standard_normal(shape)
        b = rng.standard_normal((*shape[:-1], 1))
        A, B = sdnp.asarray(a), sdnp.asarray(b)
        batched = len(shape) > 2
        mode_sp = "economic"
        rows = [
            ("qr", lambda: sdla.qr(A), lambda: np.linalg.qr(a), lambda: sl.qr(a, mode=mode_sp)),
            (
                "qr complete",
                lambda: sdla.qr(A, mode="complete"),
                lambda: np.linalg.qr(a, mode="complete"),
                lambda: sl.qr(a),
            ),
            ("qr_pivoted", lambda: sdla.qr_pivoted(A), None, lambda: sl.qr(a, mode=mode_sp, pivoting=True)),
            (
                "lstsq",
                lambda: sdla.lstsq(A, B),
                None if batched else (lambda: np.linalg.lstsq(a, b)),
                lambda: sl.lstsq(a, b),
            ),
        ]
        for name, f_sd, f_np, f_sp in rows:
            t_sd = best_us(f_sd)
            t_np = best_us(f_np) if f_np else nan
            t_sp = best_us(f_sp) if f_sp else nan
            emit(
                f"{name:18} {str(shape):>14} {t_sd:10.2f} {t_np:10.2f} {t_sp:10.2f} {t_np / t_sd:8.2f} {t_sp / t_sd:8.2f}"
            )
    path = pathlib.Path(__file__).parent / "results" / "m4_qr.txt"
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
