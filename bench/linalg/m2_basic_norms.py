"""M2 baseline: the non-LAPACK functions against numpy 2.5.4.

These loops are our own (no LAPACK), written as simply as possible in M2;
M8 optimizes them. Usage: .venv/bin/python bench/m2_basic_norms.py
(writes bench/results/m2_basic_norms.txt)
"""

import math
import pathlib
import timeit

import numpy as np
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

    s3 = rng.standard_normal((3, 3))
    b3 = rng.standard_normal((1000, 3, 3))
    m256 = rng.standard_normal((256, 256)) / 16
    m1k = rng.standard_normal((1000, 1000))
    v3, v1m = rng.standard_normal(3), rng.standard_normal(1_000_000)
    rows = rng.standard_normal((1000, 3))
    x1k = rng.standard_normal(1000)
    S = {
        k: sdnp.asarray(v)
        for k, v in dict(s3=s3, b3=b3, m256=m256, m1k=m1k, v3=v3, v1m=v1m, rows=rows, x1k=x1k).items()
    }
    cases = [
        ("vecdot", "(3,)", lambda: sdla.vecdot(S["v3"], S["v3"]), lambda: np.vecdot(v3, v3)),
        ("vecdot", "(1e6,)", lambda: sdla.vecdot(S["v1m"], S["v1m"]), lambda: np.vecdot(v1m, v1m)),
        ("vecdot", "(1000,3) rows", lambda: sdla.vecdot(S["rows"], S["rows"]), lambda: np.vecdot(rows, rows)),
        ("outer", "(3,)x(3,)", lambda: sdla.outer(S["v3"], S["v3"]), lambda: np.outer(v3, v3)),
        ("outer", "(1000,)x(1000,)", lambda: sdla.outer(S["x1k"], S["x1k"]), lambda: np.outer(x1k, x1k)),
        ("trace", "3x3", lambda: sdla.trace(S["s3"]), lambda: np.linalg.trace(s3)),
        ("trace", "(1000,3,3)", lambda: sdla.trace(S["b3"]), lambda: np.linalg.trace(b3)),
        ("diagonal", "3x3", lambda: sdla.diagonal(S["s3"]), lambda: np.linalg.diagonal(s3).copy()),
        ("diagonal", "(1000,3,3)", lambda: sdla.diagonal(S["b3"]), lambda: np.linalg.diagonal(b3).copy()),
        ("matrix_power", "3x3, n=5", lambda: sdla.matrix_power(S["s3"], 5), lambda: np.linalg.matrix_power(s3, 5)),
        ("matrix_power", "3x3, n=-1", lambda: sdla.matrix_power(S["s3"], -1), lambda: np.linalg.matrix_power(s3, -1)),
        (
            "matrix_power",
            "(1000,3,3), n=3",
            lambda: sdla.matrix_power(S["b3"], 3),
            lambda: np.linalg.matrix_power(b3, 3),
        ),
        (
            "matrix_power",
            "256x256, n=10",
            lambda: sdla.matrix_power(S["m256"], 10),
            lambda: np.linalg.matrix_power(m256, 10),
        ),
        ("vector_norm", "(3,)", lambda: sdla.vector_norm(S["v3"]), lambda: np.linalg.vector_norm(v3)),
        ("vector_norm", "(1e6,) ord=2", lambda: sdla.vector_norm(S["v1m"]), lambda: np.linalg.vector_norm(v1m)),
        (
            "vector_norm",
            "(1e6,) ord=inf",
            lambda: sdla.vector_norm(S["v1m"], ord=math.inf),
            lambda: np.linalg.vector_norm(v1m, ord=math.inf),
        ),
        (
            "vector_norm",
            "(1e6,) ord=3",
            lambda: sdla.vector_norm(S["v1m"], ord=3),
            lambda: np.linalg.vector_norm(v1m, ord=3),
        ),
        (
            "vector_norm",
            "(1000,3) axis=-1",
            lambda: sdla.vector_norm(S["rows"], axis=-1),
            lambda: np.linalg.vector_norm(rows, axis=-1),
        ),
        ("matrix_norm", "3x3 fro", lambda: sdla.matrix_norm(S["s3"]), lambda: np.linalg.matrix_norm(s3)),
        ("matrix_norm", "1000x1000 fro", lambda: sdla.matrix_norm(S["m1k"]), lambda: np.linalg.matrix_norm(m1k)),
        (
            "matrix_norm",
            "1000x1000 ord=1",
            lambda: sdla.matrix_norm(S["m1k"], ord=1),
            lambda: np.linalg.matrix_norm(m1k, ord=1),
        ),
        (
            "matrix_norm",
            "1000x1000 ord=inf",
            lambda: sdla.matrix_norm(S["m1k"], ord=math.inf),
            lambda: np.linalg.matrix_norm(m1k, ord=math.inf),
        ),
        ("matrix_norm", "(1000,3,3) fro", lambda: sdla.matrix_norm(S["b3"]), lambda: np.linalg.matrix_norm(b3)),
    ]
    emit(f"numpy {np.__version__}, sdnp threads {sdnp.get_num_threads()}")
    emit(f"{'func':13} {'case':>20} {'sdla us':>10} {'numpy us':>10} {'np/sdla':>8}")
    for name, case, f_sd, f_np in cases:
        a, b = best_us(f_sd), best_us(f_np)
        emit(f"{name:13} {case:>20} {a:10.2f} {b:10.2f} {b / a:8.2f}")
    path = pathlib.Path(__file__).parent / "results" / "m2_basic_norms.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
