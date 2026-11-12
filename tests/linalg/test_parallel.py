"""Batches and long sums on the thread pool.

Whatever the thread count (sd.set_num_threads), every result is bit-for-bit
the same and every error message names the same (smallest) batch index
(docs/design/linalg.md §8.5).
"""

import math
import threading

import numpy as np
import pytest
import stardust as sd
from .conftest import npy, as_sd


DEFAULT = sd.get_num_threads()
COUNTS = sorted({1, 3, DEFAULT})


@pytest.fixture(autouse=True)
def restore_threads():
    yield
    sd.set_num_threads(DEFAULT)


def as_tuple(r):
    return r if isinstance(r, tuple) else (r,)


def bits(r):
    return [np.asarray(v).tobytes() for v in as_tuple(r)]


def same_for_all_counts(call):
    results = []
    for n in COUNTS:
        sd.set_num_threads(n)
        results.append(bits(call()))
    assert all(r == results[0] for r in results[1:])


def matrices(rng, batch, n, dtype):
    a = rng.standard_normal((*batch, n, n))
    if dtype == "complex128":
        a = a + 1j * rng.standard_normal((*batch, n, n))
    return a + np.conj(np.swapaxes(a, -1, -2)) + 4 * n * np.eye(n)  # Hermitian, safely nonsingular


CALLS = {
    "solve": lambda a: sd.linalg.solve(a, a),
    "inv": lambda a: sd.linalg.inv(a),
    "det": lambda a: sd.linalg.det(a),
    "slogdet": lambda a: sd.linalg.slogdet(a),
    "lu": lambda a: sd.linalg.lu(a),
    "lu_solve": lambda a: sd.linalg.lu_solve(*sd.linalg.lu(a), a),
    "cholesky": lambda a: sd.linalg.cholesky(a),
    "cholesky_solve": lambda a: sd.linalg.cholesky_solve(sd.linalg.cholesky(a), a),
    "solve_triangular": lambda a: sd.linalg.solve_triangular(a, a, upper=True),
    "qr": lambda a: sd.linalg.qr(a),
    "qr_pivoted": lambda a: sd.linalg.qr_pivoted(a),
    "lstsq": lambda a: sd.linalg.lstsq(a, a),
    "eigh": lambda a: sd.linalg.eigh(a),
    "eigh_ab": lambda a: sd.linalg.eigh(a, a),
    "eigvalsh": lambda a: sd.linalg.eigvalsh(a),
    "svd": lambda a: sd.linalg.svd(a),
    "svdvals": lambda a: sd.linalg.svdvals(a),
    "matrix_rank": lambda a: sd.linalg.matrix_rank(a),
    "pinv": lambda a: sd.linalg.pinv(a),
    "cond": lambda a: sd.linalg.cond(a),
    "eig": lambda a: sd.linalg.eig(a),
    "eigvals": lambda a: sd.linalg.eigvals(a),
    "schur": lambda a: sd.linalg.schur(a),
    "hessenberg": lambda a: sd.linalg.hessenberg(a),
    "ldl": lambda a: sd.linalg.ldl(a),
    "sylvester": lambda a: sd.linalg.solve_sylvester(a, a, a),
    "matrix_power": lambda a: sd.linalg.matrix_power(a, 5),
    "matrix_power_neg": lambda a: sd.linalg.matrix_power(a, -2),
    "trace": lambda a: sd.linalg.trace(a),
    "diagonal": lambda a: sd.linalg.diagonal(a),
    "vecdot": lambda a: sd.linalg.vecdot(a, a),
    "outer": lambda a: sd.linalg.outer(a, a),
    "vector_norm": lambda a: sd.linalg.vector_norm(a, axis=(-2, -1)),
    "fro": lambda a: sd.linalg.matrix_norm(a),
    "norm1": lambda a: sd.linalg.matrix_norm(a, ord=1),
    "norm2": lambda a: sd.linalg.matrix_norm(a, ord=2),
}


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_batch_same_for_any_thread_count(name, dtype):
    rng = np.random.default_rng(abs(hash((name, dtype))) % 2**32)
    for batch, n in (((400,), 3), ((4, 30), 5), ((48,), 16)):  # all large enough to split
        a = as_sd(matrices(rng, batch, n, dtype))
        same_for_all_counts(lambda: CALLS[name](a))


def test_banded_same_for_any_thread_count():
    rng = np.random.default_rng(3)
    dl, du = rng.standard_normal((500, 9)), rng.standard_normal((500, 9))
    d = rng.standard_normal((500, 10)) + 10
    b = rng.standard_normal((500, 10, 2))
    same_for_all_counts(lambda: sd.linalg.solve_tridiagonal(as_sd(dl), as_sd(d), as_sd(du), as_sd(b)))
    ab = rng.standard_normal((500, 5, 10))
    ab[:, 2] += 10
    same_for_all_counts(lambda: sd.linalg.solve_banded(as_sd(ab), as_sd(b), lower_bandwidth=2, upper_bandwidth=2))


def test_errors_name_the_smallest_index():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2000, 3, 3)) + 6 * np.eye(3)
    x[1500, 0, 0] = np.nan
    x[777] = 0.0  # singular
    x[1234, 1, 1] = np.inf
    for n in COUNTS:
        sd.set_num_threads(n)
        with pytest.raises(sd.linalg.LinAlgError, match=r"singular matrix \(at batch index \(777,\)\)"):
            sd.linalg.inv(as_sd(x))
        y = x.copy()
        y[777] = np.eye(3)
        with pytest.raises(ValueError, match=r"'a' contains NaN or infinity \(at batch index \(1234,\)\)"):
            sd.linalg.solve(as_sd(y), as_sd(y))


@pytest.mark.parametrize("n", [40_000, 1_000_003])
def test_long_sums_same_for_any_thread_count(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = x + 1j * rng.standard_normal(n)
    X, Y, Z = as_sd(x), as_sd(y), as_sd(z)
    m = as_sd(x[: (n // 1000) * 1000].reshape(-1, 1000))
    for call in (
        lambda: sd.linalg.vecdot(X, Y),
        lambda: sd.linalg.vecdot(Z, Z),
        lambda: sd.linalg.vector_norm(X),
        lambda: sd.linalg.vector_norm(Z),
        lambda: sd.linalg.vector_norm(X, ord=1),
        lambda: sd.linalg.vector_norm(X, ord=3),
        lambda: sd.linalg.vector_norm(X, ord=math.inf),
        lambda: sd.linalg.vector_norm(X, ord=-math.inf),
        lambda: sd.linalg.vector_norm(X, ord=0),
        lambda: sd.linalg.matrix_norm(m),
        lambda: sd.linalg.matrix_norm(m, ord=1),
        lambda: sd.linalg.matrix_norm(m, ord=math.inf),
    ):
        same_for_all_counts(call)
    # still numpy's pairwise bits (docs/design/linalg.md §8.2)
    assert sd.linalg.vecdot(X, Y) == np.sum(x * y)
    assert sd.linalg.vector_norm(X, ord=1) == np.sum(np.abs(x))


def test_long_norm_rescaling_and_nan_on_threads():
    x = np.full(200_000, 1e300)  # the plain sum overflows: the rescaled pass runs, on the pool
    assert sd.linalg.vector_norm(as_sd(x)) == pytest.approx(1e300 * math.sqrt(200_000))
    x[123_456] = np.nan
    for n in COUNTS:
        sd.set_num_threads(n)
        assert math.isnan(sd.linalg.vector_norm(as_sd(x)))
        assert math.isnan(sd.linalg.vector_norm(as_sd(x), ord=math.inf))


def test_python_threads_at_once():
    # several Python threads calling at once: the pool serves one job at a time,
    # the others run serially; every result is still the same
    rng = np.random.default_rng(5)
    a = as_sd(matrices(rng, (300,), 4, "float64"))
    v = as_sd(rng.standard_normal(300_000))
    want = (bits(sd.linalg.svd(a)), bits(sd.linalg.vecdot(v, v)))
    got = []

    def work():
        for _ in range(5):
            got.append((bits(sd.linalg.svd(a)), bits(sd.linalg.vecdot(v, v))))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 20 and all(g == want for g in got)


def test_thread_count_is_library_wide():
    rng = np.random.default_rng(6)
    a = as_sd(matrices(rng, (500,), 3, "float64"))
    ref = bits(sd.linalg.qr(a))
    for n in (2, 1, 5, DEFAULT, 1):
        sd.set_num_threads(n)
        assert bits(sd.linalg.qr(a)) == ref
