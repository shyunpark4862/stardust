"""M1 baseline: solve, inv, det, slogdet against numpy 2.5.4 and scipy 1.18.1.

All three call the same Accelerate LAPACK (DESIGN.md §2.9), so the gaps come
from the wrapping: argument handling, copies, LAPACK path, result objects.
Usage: .venv/bin/python bench/m1_lu.py  (writes bench/results/m1_lu.txt)
"""

import pathlib
import sys
import timeit

import numpy as np
import scipy.linalg as sl
import stardust_numpy as sdnp

import stardust_linalg as sdla


def best_us(fn, budget=0.2):
    fn()
    n = 1
    while True:
        t = timeit.timeit(fn, number=n)
        if t > 0.02 or n > 1_000_000:
            break
        n *= 4
    reps = max(3, int(budget / max(t, 1e-9)))
    return min(timeit.repeat(fn, number=n, repeat=min(reps, 9))) / n * 1e6


def cases():
    rng = np.random.default_rng(0)
    for shape in [
        (2, 2),
        (3, 3),
        (4, 4),
        (8, 8),
        (16, 16),
        (64, 64),
        (256, 256),
        (1024, 1024),
        (1000, 3, 3),
        (100, 32, 32),
        (10, 256, 256),
    ]:
        n = shape[-1]
        a = rng.standard_normal(shape)  # general: LU pivots (no diagonal boost)
        b = rng.standard_normal(shape[:-1])  # one vector per matrix: (..., n)
        yield shape, a, b


def main():
    np.seterr(all="ignore")  # numpy warns on det overflow; we only time it
    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit(
        f"numpy {np.__version__}, scipy {sl.__name__ and __import__('scipy').__version__}, "
        f"sdnp threads {sdnp.get_num_threads()}"
    )
    emit(f"{'func':8} {'shape':>16} {'sdla us':>10} {'numpy us':>10} {'scipy us':>10} {'np/sdla':>8} {'sp/sdla':>8}")
    for shape, a, b in cases():
        A, B = sdnp.asarray(a), sdnp.asarray(b)
        bm = b[..., None]  # numpy 2 and sdla read a 1-D b as one vector; batched b needs a column
        Bm = sdnp.asarray(np.ascontiguousarray(bm))
        rows = {
            "solve": (lambda: sdla.solve(A, Bm), lambda: np.linalg.solve(a, bm), lambda: sl.solve(a, bm)),
            "inv": (lambda: sdla.inv(A), lambda: np.linalg.inv(a), lambda: sl.inv(a)),
            "det": (lambda: sdla.det(A), lambda: np.linalg.det(a), lambda: sl.det(a)),
            "slogdet": (lambda: sdla.slogdet(A), lambda: np.linalg.slogdet(a), None),
        }
        for name, (f_sd, f_np, f_sp) in rows.items():
            t_sd, t_np = best_us(f_sd), best_us(f_np)
            t_sp = best_us(f_sp) if f_sp else float("nan")
            emit(
                f"{name:8} {str(shape):>16} {t_sd:10.2f} {t_np:10.2f} {t_sp:10.2f} "
                f"{t_np / t_sd:8.2f} {t_sp / t_sd:8.2f}"
            )
    path = pathlib.Path(__file__).parent / "results" / "m1_lu.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    sys.exit(main())
