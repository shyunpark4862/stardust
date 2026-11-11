"""M9: batches and long sums on the thread pool.

Whatever the thread count (sdnp.set_num_threads), every result is bit-for-bit
the same and every error message names the same (smallest) batch index
(DESIGN.md §2.5, §2.18).
"""

import math
import threading

import numpy as np
import pytest
import stardust_numpy as sdnp
from conftest import npy, sd

import stardust_linalg as sdla

DEFAULT = sdnp.get_num_threads()
COUNTS = sorted({1, 3, DEFAULT})


@pytest.fixture(autouse=True)
def restore_threads():
    yield
    sdnp.set_num_threads(DEFAULT)


def as_tuple(r):
    return r if isinstance(r, tuple) else (r,)


def bits(r):
    return [np.asarray(v).tobytes() for v in as_tuple(r)]


def same_for_all_counts(call):
    results = []
    for n in COUNTS:
        sdnp.set_num_threads(n)
        results.append(bits(call()))
    assert all(r == results[0] for r in results[1:])


def matrices(rng, batch, n, dtype):
    a = rng.standard_normal((*batch, n, n))
    if dtype == "complex128":
        a = a + 1j * rng.standard_normal((*batch, n, n))
    return a + np.conj(np.swapaxes(a, -1, -2)) + 4 * n * np.eye(n)  # Hermitian, safely nonsingular


CALLS = {
    "solve": lambda a: sdla.solve(a, a),
    "inv": lambda a: sdla.inv(a),
    "det": lambda a: sdla.det(a),
    "slogdet": lambda a: sdla.slogdet(a),
    "lu": lambda a: sdla.lu(a),
    "lu_solve": lambda a: sdla.lu_solve(*sdla.lu(a), a),
    "cholesky": lambda a: sdla.cholesky(a),
    "cholesky_solve": lambda a: sdla.cholesky_solve(sdla.cholesky(a), a),
    "solve_triangular": lambda a: sdla.solve_triangular(a, a, upper=True),
    "qr": lambda a: sdla.qr(a),
    "qr_pivoted": lambda a: sdla.qr_pivoted(a),
    "lstsq": lambda a: sdla.lstsq(a, a),
    "eigh": lambda a: sdla.eigh(a),
    "eigh_ab": lambda a: sdla.eigh(a, a),
    "eigvalsh": lambda a: sdla.eigvalsh(a),
    "svd": lambda a: sdla.svd(a),
    "svdvals": lambda a: sdla.svdvals(a),
    "matrix_rank": lambda a: sdla.matrix_rank(a),
    "pinv": lambda a: sdla.pinv(a),
    "cond": lambda a: sdla.cond(a),
    "eig": lambda a: sdla.eig(a),
    "eigvals": lambda a: sdla.eigvals(a),
    "schur": lambda a: sdla.schur(a),
    "hessenberg": lambda a: sdla.hessenberg(a),
    "ldl": lambda a: sdla.ldl(a),
    "sylvester": lambda a: sdla.solve_sylvester(a, a, a),
    "matrix_power": lambda a: sdla.matrix_power(a, 5),
    "matrix_power_neg": lambda a: sdla.matrix_power(a, -2),
    "trace": lambda a: sdla.trace(a),
    "diagonal": lambda a: sdla.diagonal(a),
    "vecdot": lambda a: sdla.vecdot(a, a),
    "outer": lambda a: sdla.outer(a, a),
    "vector_norm": lambda a: sdla.vector_norm(a, axis=(-2, -1)),
    "fro": lambda a: sdla.matrix_norm(a),
    "norm1": lambda a: sdla.matrix_norm(a, ord=1),
    "norm2": lambda a: sdla.matrix_norm(a, ord=2),
}


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_batch_same_for_any_thread_count(name, dtype):
    rng = np.random.default_rng(abs(hash((name, dtype))) % 2**32)
    for batch, n in (((400,), 3), ((4, 30), 5), ((48,), 16)):  # all large enough to split
        a = sd(matrices(rng, batch, n, dtype))
        same_for_all_counts(lambda: CALLS[name](a))


def test_banded_same_for_any_thread_count():
    rng = np.random.default_rng(3)
    dl, du = rng.standard_normal((500, 9)), rng.standard_normal((500, 9))
    d = rng.standard_normal((500, 10)) + 10
    b = rng.standard_normal((500, 10, 2))
    same_for_all_counts(lambda: sdla.solve_tridiagonal(sd(dl), sd(d), sd(du), sd(b)))
    ab = rng.standard_normal((500, 5, 10))
    ab[:, 2] += 10
    same_for_all_counts(lambda: sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=2, upper_bandwidth=2))


def test_errors_name_the_smallest_index():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2000, 3, 3)) + 6 * np.eye(3)
    x[1500, 0, 0] = np.nan
    x[777] = 0.0  # singular
    x[1234, 1, 1] = np.inf
    for n in COUNTS:
        sdnp.set_num_threads(n)
        with pytest.raises(sdla.LinAlgError, match=r"singular matrix \(at batch index \(777,\)\)"):
            sdla.inv(sd(x))
        y = x.copy()
        y[777] = np.eye(3)
        with pytest.raises(ValueError, match=r"'a' contains NaN or infinity \(at batch index \(1234,\)\)"):
            sdla.solve(sd(y), sd(y))


@pytest.mark.parametrize("n", [40_000, 1_000_003])
def test_long_sums_same_for_any_thread_count(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = x + 1j * rng.standard_normal(n)
    X, Y, Z = sd(x), sd(y), sd(z)
    m = sd(x[: (n // 1000) * 1000].reshape(-1, 1000))
    for call in (
        lambda: sdla.vecdot(X, Y),
        lambda: sdla.vecdot(Z, Z),
        lambda: sdla.vector_norm(X),
        lambda: sdla.vector_norm(Z),
        lambda: sdla.vector_norm(X, ord=1),
        lambda: sdla.vector_norm(X, ord=3),
        lambda: sdla.vector_norm(X, ord=math.inf),
        lambda: sdla.vector_norm(X, ord=-math.inf),
        lambda: sdla.vector_norm(X, ord=0),
        lambda: sdla.matrix_norm(m),
        lambda: sdla.matrix_norm(m, ord=1),
        lambda: sdla.matrix_norm(m, ord=math.inf),
    ):
        same_for_all_counts(call)
    # still numpy's pairwise bits (M8)
    assert sdla.vecdot(X, Y) == np.sum(x * y)
    assert sdla.vector_norm(X, ord=1) == np.sum(np.abs(x))


def test_long_norm_rescaling_and_nan_on_threads():
    x = np.full(200_000, 1e300)  # the plain sum overflows: the rescaled pass runs, on the pool
    assert sdla.vector_norm(sd(x)) == pytest.approx(1e300 * math.sqrt(200_000))
    x[123_456] = np.nan
    for n in COUNTS:
        sdnp.set_num_threads(n)
        assert math.isnan(sdla.vector_norm(sd(x)))
        assert math.isnan(sdla.vector_norm(sd(x), ord=math.inf))


def test_python_threads_at_once():
    # several Python threads calling at once: the pool serves one job at a time,
    # the others run serially; every result is still the same
    rng = np.random.default_rng(5)
    a = sd(matrices(rng, (300,), 4, "float64"))
    v = sd(rng.standard_normal(300_000))
    want = (bits(sdla.svd(a)), bits(sdla.vecdot(v, v)))
    got = []

    def work():
        for _ in range(5):
            got.append((bits(sdla.svd(a)), bits(sdla.vecdot(v, v))))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 20 and all(g == want for g in got)


def test_thread_count_follows_sdnp():
    rng = np.random.default_rng(6)
    a = sd(matrices(rng, (500,), 3, "float64"))
    ref = bits(sdla.qr(a))
    for n in (2, 1, 5, DEFAULT, 1):
        sdnp.set_num_threads(n)
        assert bits(sdla.qr(a)) == ref
