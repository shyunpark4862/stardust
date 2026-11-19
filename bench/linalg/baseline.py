"""The baseline (ROADMAP.md M7): every function on one fixed set of shapes,
against numpy 2.5.4 or scipy 1.18.1. M8 and M9 rerun this same script, so
their results line up row by row with this one.

Usage: .venv/bin/python bench/baseline.py TAG   (writes bench/results/baseline_TAG.txt)
       .venv/bin/python bench/baseline.py TAG --only eig,svd   (a subset, printed only)

The reference ("ref") is numpy where numpy has the function, else scipy;
"-" where neither takes the input (e.g. a batch). ref/sdla > 1: sdla is faster.
"""

import datetime
import math
import pathlib
import platform
import sys
import timeit

import numpy as np
import scipy
import scipy.linalg as sl
import stardust_numpy as sdnp

import stardust_linalg as sdla

MATRIX_SHAPES = [(3, 3), (32, 32), (300, 300), (1000, 1000), (1000, 3, 3), (100, 32, 32)]


def best_us(fn):
    """Best of 5 rounds, each long enough (>= 50 ms or 1e6 calls) to time reliably."""
    fn()
    n = 1
    while True:
        t = timeit.timeit(fn, number=n)
        if t > 0.05 or n > 1_000_000:
            break
        n *= 4
    return min(timeit.repeat(fn, number=n, repeat=5)) / n * 1e6


def matrix_rows(shape, rng):
    """(name, sdla call, ref call or None, ref name) for one matrix shape."""
    n = shape[-1]
    batch = shape[:-2]
    eye = np.eye(n)
    a = rng.standard_normal(shape)
    b = rng.standard_normal((*batch, n, 1))
    h = (a + np.swapaxes(a, -1, -2)) / 2
    spd = a @ np.swapaxes(a, -1, -2) + n * eye
    t = np.triu(a) + n * eye
    chol = np.linalg.cholesky(spd)
    tri = [
        rng.standard_normal((*batch, n - 1)),
        rng.standard_normal((*batch, n)) + 10,
        rng.standard_normal((*batch, n - 1)),
    ]
    ab_tri = np.zeros((*batch, 3, n))
    ab_tri[..., 0, 1:], ab_tri[..., 1, :], ab_tri[..., 2, :-1] = tri[2], tri[1], tri[0]
    ab5 = rng.standard_normal((*batch, 5, n))
    ab5[..., 2, :] += 10  # kl = ku = 2, a dominant diagonal
    c = rng.standard_normal(shape)
    x0 = a[..., 0, :]
    A, B, H, SPD, T, CHOL, C, X0 = map(sdnp.asarray, (a, b, h, spd, t, chol, c, x0))
    TRI = [sdnp.asarray(v) for v in tri]
    AB_TRI, AB5 = sdnp.asarray(ab_tri), sdnp.asarray(ab5)
    LU = sdla.lu(A)
    lu_f = sl.lu_factor(a) if not batch else None
    k3 = rng.standard_normal((3, 3))  # kron's second operand: (..., 3n, 3n) results
    v3, w3 = a[..., :3], c[..., :3]  # cross: the rows' first three entries
    K3, V3, W3 = sdnp.asarray(k3), sdnp.asarray(v3), sdnp.asarray(w3)
    m_ = a.shape[-2]

    def np_(f):
        return ("numpy", f)

    def sp(f):
        return ("scipy", f)

    single = not batch
    return [
        # M1
        ("solve", lambda: sdla.solve(A, B), np_(lambda: np.linalg.solve(a, b))),
        ("inv", lambda: sdla.inv(A), np_(lambda: np.linalg.inv(a))),
        ("det", lambda: sdla.det(A), np_(lambda: np.linalg.det(a))),
        ("slogdet", lambda: sdla.slogdet(A), np_(lambda: np.linalg.slogdet(a))),
        # M2
        ("vecdot rows", lambda: sdla.vecdot(A, A), np_(lambda: np.vecdot(a, a))),
        ("outer", lambda: sdla.outer(X0, X0), np_(lambda: x0[..., :, None] * x0[..., None, :])),
        ("trace", lambda: sdla.trace(A), np_(lambda: np.linalg.trace(a))),
        ("diagonal", lambda: sdla.diagonal(A), np_(lambda: np.linalg.diagonal(a).copy())),
        ("matrix_power 3", lambda: sdla.matrix_power(A, 3), np_(lambda: np.linalg.matrix_power(a, 3))),
        ("vector_norm rows", lambda: sdla.vector_norm(A, axis=-1), np_(lambda: np.linalg.vector_norm(a, axis=-1))),
        ("matrix_norm fro", lambda: sdla.matrix_norm(A), np_(lambda: np.linalg.matrix_norm(a))),
        ("matrix_norm 1", lambda: sdla.matrix_norm(A, ord=1), np_(lambda: np.linalg.matrix_norm(a, ord=1))),
        ("matrix_norm 2", lambda: sdla.matrix_norm(A, ord=2), np_(lambda: np.linalg.matrix_norm(a, ord=2))),
        # M3
        (
            "solve_triangular",
            lambda: sdla.solve_triangular(T, B, upper=True),
            sp(lambda: sl.solve_triangular(t, b, lower=False)),
        ),
        ("cholesky", lambda: sdla.cholesky(SPD), np_(lambda: np.linalg.cholesky(spd))),
        ("cholesky_solve", lambda: sdla.cholesky_solve(CHOL, B), sp(lambda: sl.cho_solve((chol, True), b))),
        ("lu", lambda: sdla.lu(A), sp(lambda: sl.lu(a, p_indices=True))),
        ("lu_solve", lambda: sdla.lu_solve(*LU, B), sp(lambda: sl.lu_solve(lu_f, b)) if single else None),
        # M4
        ("qr", lambda: sdla.qr(A), np_(lambda: np.linalg.qr(a))),
        ("qr complete", lambda: sdla.qr(A, mode="complete"), np_(lambda: np.linalg.qr(a, mode="complete"))),
        ("qr_pivoted", lambda: sdla.qr_pivoted(A), sp(lambda: sl.qr(a, mode="economic", pivoting=True))),
        (
            "lstsq",
            lambda: sdla.lstsq(A, B),
            np_(lambda: np.linalg.lstsq(a, b)) if single else sp(lambda: sl.lstsq(a, b)),
        ),
        # M5
        ("eigh", lambda: sdla.eigh(H), np_(lambda: np.linalg.eigh(h))),
        ("eigvalsh", lambda: sdla.eigvalsh(H), np_(lambda: np.linalg.eigvalsh(h))),
        ("eigh(a, b)", lambda: sdla.eigh(H, SPD), sp(lambda: sl.eigh(h, spd))),
        ("svd", lambda: sdla.svd(A), np_(lambda: np.linalg.svd(a, full_matrices=False))),
        ("svdvals", lambda: sdla.svdvals(A), np_(lambda: np.linalg.svdvals(a))),
        ("matrix_rank", lambda: sdla.matrix_rank(A), np_(lambda: np.linalg.matrix_rank(a))),
        ("pinv", lambda: sdla.pinv(A), np_(lambda: np.linalg.pinv(a))),
        ("cond", lambda: sdla.cond(A), np_(lambda: np.linalg.cond(a))),
        # M6
        ("eig", lambda: sdla.eig(A), np_(lambda: np.linalg.eig(a))),
        ("eigvals", lambda: sdla.eigvals(A), np_(lambda: np.linalg.eigvals(a))),
        ("schur", lambda: sdla.schur(A), sp(lambda: sl.schur(a))),
        ("hessenberg", lambda: sdla.hessenberg(A), sp(lambda: sl.hessenberg(a, calc_q=True))),
        ("ldl", lambda: sdla.ldl(H), sp(lambda: sl.ldl(h)) if single else None),
        # M7
        (
            "solve_tridiagonal",
            lambda: sdla.solve_tridiagonal(*TRI, B),
            sp(lambda: sl.solve_banded((1, 1), ab_tri, b)) if single else None,
        ),
        (
            "solve_banded 2,2",
            lambda: sdla.solve_banded(AB5, B, lower_bandwidth=2, upper_bandwidth=2),
            sp(lambda: sl.solve_banded((2, 2), ab5, b)) if single else None,
        ),
        (
            "solve_banded 1,1",
            lambda: sdla.solve_banded(AB_TRI, B, lower_bandwidth=1, upper_bandwidth=1),
            sp(lambda: sl.solve_banded((1, 1), ab_tri, b)) if single else None,
        ),
        ("solve_sylvester", lambda: sdla.solve_sylvester(A, A, C), sp(lambda: sl.solve_sylvester(a, a, c))),
        # M10 (a batched kron is numpy's broadcast product, reshaped)
        (
            "kron with 3x3",
            lambda: sdla.kron(A, K3),
            np_(lambda: np.kron(a, k3))
            if single
            else np_(lambda: (a[..., :, None, :, None] * k3[:, None, :]).reshape(*batch, 3 * m_, 3 * n)),
        ),
        ("cross rows", lambda: sdla.cross(V3, W3), np_(lambda: np.cross(v3, w3))),
        ("expm", lambda: sdla.expm(A), sp(lambda: sl.expm(a))),
    ]


def long_rows(rng):
    """Long vectors and long structured systems, where the per-element loops show."""
    v = rng.standard_normal(1_000_000)
    n = 100_000
    tri = [rng.standard_normal(n - 1), rng.standard_normal(n) + 10, rng.standard_normal(n - 1)]
    ab_tri = np.zeros((3, n))
    ab_tri[0, 1:], ab_tri[1], ab_tri[2, :-1] = tri[2], tri[1], tri[0]
    ab5 = rng.standard_normal((5, n))
    ab5[2] += 10
    b = rng.standard_normal(n)
    V, TRI, AB5, B = sdnp.asarray(v), [sdnp.asarray(t) for t in tri], sdnp.asarray(ab5), sdnp.asarray(b)
    p3, q3 = rng.standard_normal((1_000_000, 3)), rng.standard_normal((1_000_000, 3))
    P3, Q3 = sdnp.asarray(p3), sdnp.asarray(q3)
    return [
        ("vecdot", "(1000000,)", lambda: sdla.vecdot(V, V), ("numpy", lambda: np.vecdot(v, v))),
        ("vector_norm", "(1000000,)", lambda: sdla.vector_norm(V), ("numpy", lambda: np.linalg.vector_norm(v))),
        (
            "vector_norm inf",
            "(1000000,)",
            lambda: sdla.vector_norm(V, ord=math.inf),
            ("numpy", lambda: np.linalg.vector_norm(v, ord=math.inf)),
        ),
        (
            "solve_tridiagonal",
            f"n={n}",
            lambda: sdla.solve_tridiagonal(*TRI, B),
            ("scipy", lambda: sl.solve_banded((1, 1), ab_tri, b)),
        ),
        (
            "solve_banded 2,2",
            f"n={n}",
            lambda: sdla.solve_banded(AB5, B, lower_bandwidth=2, upper_bandwidth=2),
            ("scipy", lambda: sl.solve_banded((2, 2), ab5, b)),
        ),
        ("cross", "(1000000,3)", lambda: sdla.cross(P3, Q3), ("numpy", lambda: np.cross(p3, q3))),
    ]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tag = sys.argv[1]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[2] == "--only" else None
    rng = np.random.default_rng(0)
    np.seterr(all="ignore")  # numpy's det/slogdet raise spurious FP warnings on Accelerate (seen in M1)
    out = []

    def emit(s):
        print(s, flush=True)
        out.append(s)

    emit(
        f"baseline {tag}: {datetime.date.today()}, {platform.machine()}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, sdnp threads {sdnp.get_num_threads()}"
    )
    emit(f"{'func':18} {'shape':>14} {'sdla us':>12} {'ref us':>12} {'ref/sdla':>9}  ref")

    def run(name, shape, f_sd, ref):
        if only and name.split()[0] not in only:
            return
        t_sd = best_us(f_sd)
        if ref is None:
            emit(f"{name:18} {shape:>14} {t_sd:12.2f} {'-':>12} {'-':>9}  -")
            return
        t_ref = best_us(ref[1])
        emit(f"{name:18} {shape:>14} {t_sd:12.2f} {t_ref:12.2f} {t_ref / t_sd:9.2f}  {ref[0]}")

    for shape in MATRIX_SHAPES:
        for name, f_sd, ref in matrix_rows(shape, rng):
            run(name, str(shape).replace(" ", ""), f_sd, ref)
    for name, shape, f_sd, ref in long_rows(rng):
        run(name, shape, f_sd, ref)
    if only is None:
        path = pathlib.Path(__file__).parent / "results" / f"baseline_{tag}.txt"
        path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
