"""Bugs found by the second code review: result sizes past int64 and past 32 axes,
cond and det at the edges of double, the GIL of unsplit batches, messages."""

import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import stardust as sd

LA = sd.linalg
INF = float("inf")

# Each size case runs in a child process: before the fix some of them crashed the
# interpreter (SIGBUS, SIGSEGV) or wrote past an array's shape.
_SIZE_CHILD = r"""
import sys
import stardust as sd
LA = sd.linalg
def bv(shape):  # a 0-stride view: a huge shape over one element
    return sd.broadcast_to(sd.reshape(sd.asarray([2.0]), (1,) * len(shape)), shape)
x32 = sd.ones((1,) * 31 + (2,))
cases = {
    "solve": lambda: LA.solve(bv((2**32, 1, 1, 1)), bv((2**32, 1, 1))),
    "vecdot": lambda: LA.vecdot(bv((2**32, 1, 4)), bv((1, 2**32, 4))),
    "outer batch": lambda: LA.outer(bv((2**32, 1)), bv((2**32, 1, 1))),
    "outer": lambda: LA.outer(bv((2**32,)), bv((2**32,))),
    "kron rows": lambda: LA.kron(bv((2**31, 1)), bv((2**33, 1))),
    "kron bytes": lambda: LA.kron(bv((2**32, 1)), bv((2**32 - 1, 1))),
    "solve_triangular": lambda: LA.solve_triangular(bv((2**32, 1, 1, 1)), bv((2**32, 1, 1)), upper=True),
    "cross": lambda: LA.cross(bv((2**32, 1, 3)), bv((1, 2**32, 3))),
    "outer 32 axes": lambda: LA.outer(x32, x32),
}
try:
    r = cases[sys.argv[1]]()
    print("returned", getattr(r, "shape", r))
except Exception as e:
    print(type(e).__name__, e)
"""

SIZE_CASES = {
    "solve": "ValueError solve(): array is too big",
    "vecdot": "ValueError vecdot(): array is too big",
    "outer batch": "ValueError outer(): array is too big",
    "outer": "ValueError outer(): array is too big",
    "kron rows": "ValueError kron(): array is too big",
    "kron bytes": "ValueError kron(): array is too big",
    "solve_triangular": "ValueError solve_triangular(): array is too big",
    "cross": "ValueError cross(): array is too big",
    "outer 32 axes": "ValueError outer(): arrays may have at most 32 dimensions",
}


@pytest.mark.parametrize("case", list(SIZE_CASES))
def test_results_past_int64_or_32_axes_raise(case):
    r = subprocess.run([sys.executable, "-c", _SIZE_CHILD, case], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == SIZE_CASES[case]


def test_batch_with_an_empty_axis_is_empty():
    a = sd.broadcast_to(sd.reshape(sd.asarray([2.0]), (1, 1, 1, 1)), (0, 3, 1, 1))
    assert LA.det(a).shape == (0, 3)
    assert LA.solve(a, sd.ones((0, 3, 1, 1))).shape == (0, 3, 1, 1)


# --------------------------------------------------------------- cond, det --

NEAR_SINGULAR = [
    [[1.0, 0.0], [0.0, 5e-324]],
    [[5e-324, 1.0, -1.0], [-1.0, -1.0, 5e-324], [5e-324, 5e-324, 5e-324]],
]


@pytest.mark.parametrize("a", NEAR_SINGULAR)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("p", [1, -1, INF, -INF, "fro", "nuc"])
def test_cond_is_inf_when_the_inverse_overflows(a, dtype, p):
    x = sd.asarray(np.array(a, dtype=dtype))
    assert LA.cond(x, ord=p) == INF
    assert np.linalg.cond(np.array(a, dtype=dtype), p) == INF  # numpy's value too


def test_cond_of_a_finite_inverse_is_unchanged():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    for p in (1, -1, INF, -INF, "fro", "nuc"):
        assert LA.cond(sd.asarray(a), ord=p) == pytest.approx(np.linalg.cond(a, p), rel=1e-14)


@pytest.mark.parametrize(
    "d, want",
    [
        ([1e200, 1e200, 1e-300], 1e100),
        ([1e-200, 1e-200, 1e300], 1e-100),
        ([1e300, 1e10, 1e-300], 1e10),
        ([1e300, 1e300], INF),  # the determinant itself overflows
        ([1e-300, 1e-300], 0.0),
        ([-1e200, 1e200, 1e-300], -1e100),
    ],
)
def test_det_survives_an_overflowing_partial_product(d, want):
    got = LA.det(sd.asarray(np.diag(d)))
    assert got == pytest.approx(want, rel=1e-15, abs=0.0)


def test_det_with_swaps_and_complex_past_the_range():
    a = np.array([[0.0, 1e200, 0.0], [1e200, 0.0, 0.0], [0.0, 0.0, 1e-300]])  # one row swap
    assert LA.det(sd.asarray(a)) == pytest.approx(-1e100, rel=1e-15)
    c = np.diag([1e200 + 1e200j, 1e200, 1e-300j])
    assert LA.det(sd.asarray(c)) == pytest.approx(-1e100 + 1e100j, rel=1e-15)


def test_det_bits_within_the_range_are_the_plain_product():
    # No partial product leaves the normal range: the determinant is U's diagonal
    # multiplied in order (the swaps only flip the sign), bit for bit.
    rng = np.random.default_rng(3)
    for n in (2, 5, 9, 30):
        a = rng.standard_normal((n, n))
        u = np.asarray(LA.lu(sd.asarray(a)).U)
        ref = 1.0
        for i in range(n):
            ref = ref * u[i, i]
        got = LA.det(sd.asarray(a))
        assert abs(got) == abs(ref) and np.sign(got) == LA.slogdet(sd.asarray(a)).sign, n


# ------------------------------------------------------------------- GIL --


class Watcher:
    """A Python thread that measures how long it was kept from running."""

    def __init__(self):
        self.longest = 0.0
        self.turns = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            self.longest = max(self.longest, now - last)
            last = now
            self.turns += 1

    def __enter__(self):
        self._thread.start()
        while self.turns == 0:
            time.sleep(0.001)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def held_fraction(self, call):
        best = 1.0
        for _ in range(4):  # a pause from the machine being busy can hit one attempt
            time.sleep(0.002)
            self.longest = 0.0
            start = time.perf_counter()
            call()
            took = time.perf_counter() - start
            time.sleep(0.002)
            best = min(best, self.longest / took)
            if best < 0.25:
                break
        return best


@pytest.mark.serial
@pytest.mark.parametrize("threads", [1, 10])
def test_unsplit_and_many_rhs_solves_release_the_gil(threads):
    rng = np.random.default_rng(0)
    n, batch = 10, 100_000
    ab = sd.asarray(rng.standard_normal((batch, 5, n)) + np.array([0, 0, 10, 0, 0])[:, None])
    b = sd.asarray(rng.standard_normal((batch, n, 1)))
    a3 = sd.asarray(rng.standard_normal((3, 3)) + 3 * np.eye(3))
    rhs = sd.asarray(np.asfortranarray(rng.standard_normal((3, 3_000_000))))
    d = sd.asarray(rng.standard_normal(4) + 5)
    dl = sd.asarray(rng.standard_normal(3))
    rhs4 = sd.asarray(np.asfortranarray(rng.standard_normal((4, 2_000_000))))
    calls = {
        # a small band is never split (docs/design/linalg.md §7.4), yet long
        "solve_banded 2,2 batch": lambda: LA.solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2),
        # one small matrix, many right-hand sides
        "solve many rhs": lambda: LA.solve(a3, rhs),
        "solve_triangular many rhs": lambda: LA.solve_triangular(a3, rhs, upper=True),
        "solve_tridiagonal many rhs": lambda: LA.solve_tridiagonal(dl, d, dl, rhs4),
    }
    before, interval = sd.get_num_threads(), sys.getswitchinterval()
    sd.set_num_threads(threads)
    sys.setswitchinterval(0.0005)
    try:
        for call in calls.values():
            call()
        with Watcher() as w:
            held = {name: f for name, call in calls.items() if (f := w.held_fraction(call)) >= 0.25}
    finally:
        sys.setswitchinterval(interval)
        sd.set_num_threads(before)
    assert not held, held


# ------------------------------------------------- lstsq alignment, messages --


@pytest.mark.parametrize("m, n, k", [(3, 3, 1), (5, 3, 3), (3, 5, 3), (9, 5, 1), (31, 29, 3)])
def test_lstsq_small_rtol_batch_is_one_by_one(m, n, k):
    # The SVD path writes the solution through gemm; an odd n * k puts every other
    # batch element's slot off 16 bytes (docs/design/linalg.md §6.4).
    rng = np.random.default_rng(m * 100 + n * 10 + k)
    a = rng.standard_normal((7, m, n))
    b = rng.standard_normal((7, m, k))
    whole = np.asarray(LA.lstsq(sd.asarray(a), sd.asarray(b), rtol=0.0).solution)
    for i in range(7):
        one = np.asarray(LA.lstsq(sd.asarray(a[i]), sd.asarray(b[i]), rtol=0.0).solution)
        assert np.array_equal(one, whole[i])


def test_messages_show_the_value_received():
    x = sd.asarray(np.eye(2))
    with pytest.raises(ValueError, match=r"^matrix_norm\(\): 'ord' must be .* or -inf, not 3$"):
        LA.matrix_norm(x, ord=3)
    with pytest.raises(ValueError, match=r"^cond\(\): 'ord' must be .* or -inf, not 'FRO'$"):
        LA.cond(x, ord="FRO")
    f = LA.lu(x)
    with pytest.raises(TypeError, match=r"^lu_solve\(\): 'perm' must be an int64 array, not float64$"):
        LA.lu_solve(f.L, f.U, sd.asarray([0.0, 1.0]), x)


# ---------------------------------------------------- read-only result fields --


def _results():
    a = sd.asarray(np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]))
    batch = sd.asarray(np.stack([np.eye(3) * 2, np.eye(3) * 3]))
    return {
        "slogdet": LA.slogdet(batch),
        "eigh": LA.eigh(a),
        "eigh batch": LA.eigh(batch),
        "svd": LA.svd(a),
        "qr": LA.qr(a),
        "qr_pivoted": LA.qr_pivoted(a),
        "lu": LA.lu(a),
        "lstsq": LA.lstsq(batch, sd.ones((2, 3, 1))),
        "eig": LA.eig(a),
        "schur": LA.schur(a),
        "hessenberg": LA.hessenberg(a),
        "ldl": LA.ldl(a),
    }


@pytest.mark.parametrize("name", list(_results()))
def test_namedtuple_array_fields_are_read_only(name):
    r = _results()[name]
    arrays = [f for f in r if isinstance(f, sd.ndarray)]
    assert arrays, name
    for f in arrays:
        before = np.asarray(f).copy()
        with pytest.raises(ValueError, match="read-only"):
            f[(0,) * f.ndim] = 7
        with pytest.raises(ValueError, match="read-only"):
            f[...] = 7  # a view of the field is read-only too
        assert np.array_equal(np.asarray(f), before)
        c = f.copy()
        c[(0,) * c.ndim] = 7  # a copy is the caller's to change


def test_single_array_results_stay_writable():
    x = LA.inv(sd.asarray(np.eye(2) * 2))
    x[0, 0] = 1.0
    s = LA.solve(sd.asarray(np.eye(2)), sd.ones(2))
    s[0] = 3.0
    assert s.tolist() == [3.0, 1.0]


# ------------------------------------------------- paths chosen by shape (§1.3) --
# LU of n = 32-46 by getf2 (docs/design/linalg.md §7.5), solve_banded of n <= 8 as a dense
# matrix (§7.4): each side of each switch gives right answers with the invariances kept.


def _bits(r):
    return [
        np.asarray(f).tobytes() if isinstance(f, sd.ndarray) else repr(f) for f in (r if isinstance(r, tuple) else (r,))
    ]


@pytest.mark.parametrize("n", [31, 32, 46, 47])
def test_lu_family_at_the_getf2_boundaries(n):
    rng = np.random.default_rng(n)
    a = rng.standard_normal((5, n, n))
    b = rng.standard_normal((5, n, 3))
    A, B = sd.asarray(a), sd.asarray(b)
    x = np.asarray(LA.solve(A, B))
    assert np.max(np.abs(a @ x - b)) <= 1e-10 * np.max(np.abs(b)) * np.linalg.cond(a).max()
    with np.errstate(all="ignore"):  # numpy's det raises spurious flags at these sizes
        want_det = np.linalg.det(a)
    np.testing.assert_allclose(np.asarray(LA.det(A)), want_det, rtol=1e-11)
    f = LA.lu(A[0])
    L_, U_, p = np.asarray(f.L), np.asarray(f.U), np.asarray(f.perm)
    np.testing.assert_allclose(L_ @ U_, a[0][p], atol=1e-13 * n)
    for call in (LA.inv, LA.det, LA.slogdet, LA.lu, lambda m: LA.solve(m, B)):
        whole = _bits(call(A))
        # the same bits for an F-order input, one by one, and on 1 or 10 threads
        assert _bits(call(sd.asarray(np.asfortranarray(a)))) == whole
        before = sd.get_num_threads()
        try:
            sd.set_num_threads(1)
            assert _bits(call(A)) == whole
        finally:
            sd.set_num_threads(before)
    one = np.asarray(LA.inv(A))
    for i in range(5):
        assert np.array_equal(np.asarray(LA.inv(sd.asarray(a[i]))), one[i])


@pytest.mark.parametrize("m, n", [(32, 46), (32, 47), (31, 40), (46, 33)])
def test_rectangular_lu_near_the_getf2_range(m, n):
    a = np.random.default_rng(m * n).standard_normal((m, n))
    f = LA.lu(sd.asarray(a))
    np.testing.assert_allclose(np.asarray(f.L) @ np.asarray(f.U), a[np.asarray(f.perm)], atol=1e-13 * max(m, n))


@pytest.mark.parametrize("n", [7, 8, 9, 12])
@pytest.mark.parametrize("kl, ku", [(0, 0), (2, 2), (0, 3), (3, 1)])
def test_solve_banded_at_the_dense_boundary(n, kl, ku):
    if max(kl, ku) >= n:
        pytest.skip("band wider than the matrix")
    rng = np.random.default_rng(n * 100 + kl * 10 + ku)
    ab = rng.standard_normal((4, kl + ku + 1, n))
    ab[:, ku, :] += 8.0
    b = rng.standard_normal((4, n, 2))
    # the dense matrix of each band
    a = np.zeros((4, n, n))
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            a[:, i, j] = ab[:, ku + i - j, j]
    x = np.asarray(LA.solve_banded(sd.asarray(ab), sd.asarray(b), lower_bandwidth=kl, upper_bandwidth=ku))
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-12, atol=1e-12)
    # corners outside the matrix are never read, on either path
    abn = ab.copy()
    for r in range(ku):
        abn[:, r, : ku - r] = np.nan
    for r in range(ku + 1, kl + ku + 1):
        abn[:, r, n - (r - ku) :] = np.nan
    same = np.asarray(LA.solve_banded(sd.asarray(abn), sd.asarray(b), lower_bandwidth=kl, upper_bandwidth=ku))
    assert np.array_equal(same, x)
    # layout, batch position and thread count do not change the bits
    f = np.asarray(
        LA.solve_banded(sd.asarray(np.asfortranarray(ab)), sd.asarray(b), lower_bandwidth=kl, upper_bandwidth=ku)
    )
    assert np.array_equal(f, x)
    for i in range(4):
        xi = LA.solve_banded(sd.asarray(ab[i]), sd.asarray(b[i]), lower_bandwidth=kl, upper_bandwidth=ku)
        assert np.array_equal(np.asarray(xi), x[i])
    # an exact zero pivot is singular on both paths
    sing = ab.copy()
    sing[0, :, 0] = 0.0  # column 0 is zero inside the band
    with pytest.raises(LA.LinAlgError, match=r"singular matrix \(at batch index \(0,\)\)"):
        LA.solve_banded(sd.asarray(sing), sd.asarray(b), lower_bandwidth=kl, upper_bandwidth=ku)
