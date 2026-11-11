"""After M11 (docs/learn/next-optimizations.md): three changes that must not
change any result -- solve_sylvester's two Schur decompositions at the same
time, a cap on tasks for the LU family near n = 32, and large C-order inputs
transposed on the pool. Every result is the same for any thread count."""

import numpy as np
import pytest
import stardust_numpy as sdnp
from conftest import sd

import stardust_linalg as sdla
from stardust_linalg import _core

DEFAULT = sdnp.get_num_threads()
COUNTS = sorted({1, 3, DEFAULT})


@pytest.fixture(autouse=True)
def restore():
    yield
    sdnp.set_num_threads(DEFAULT)
    _core._set_schur_pair_min(8)
    _core._set_pack_parallel_min(1 << 17)
    _core._set_lu_task_cap(True)


def bits(r):
    return [np.asarray(v).tobytes() for v in (r if isinstance(r, tuple) else (r,))]


def same_for_all_counts(call):
    res = []
    for n in COUNTS:
        sdnp.set_num_threads(n)
        res.append(bits(call()))
    assert all(r == res[0] for r in res[1:])
    return res[0]


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("m,n", [(8, 8), (40, 9), (9, 40), (120, 120)])
def test_sylvester_schur_pair(dtype, m, n):
    rng = np.random.default_rng(m * 1000 + n)
    mk = lambda r, c: rng.standard_normal((r, c)) + (1j * rng.standard_normal((r, c)) if dtype == "complex128" else 0)  # noqa: E731
    a, b, c = sd(mk(m, m)), sd(mk(n, n) + 3 * n * np.eye(n)), sd(mk(m, n))
    got = same_for_all_counts(lambda: sdla.solve_sylvester(a, b, c))
    _core._set_schur_pair_min(1 << 40)  # one after the other, as before
    assert bits(sdla.solve_sylvester(a, b, c)) == got


def test_sylvester_error_order_unchanged():
    rng = np.random.default_rng(1)
    a, b, c = rng.standard_normal((50, 50)), rng.standard_normal((50, 50)), rng.standard_normal((50, 50))
    bad = b.copy()
    bad[3, 4] = np.nan
    with pytest.raises(ValueError, match="'b' contains NaN"):
        sdla.solve_sylvester(sd(a), sd(bad), sd(c))
    both = a.copy()
    both[0, 0] = np.inf
    with pytest.raises(ValueError, match="'a' contains NaN"):  # a is reported before b
        sdla.solve_sylvester(sd(both), sd(bad), sd(c))
    cc = c.copy()
    cc[1, 1] = np.nan
    with pytest.raises(ValueError, match="'c' contains NaN"):
        sdla.solve_sylvester(sd(a), sd(b), sd(cc))


@pytest.mark.parametrize("n", [24, 32, 36])
def test_lu_task_cap_changes_nothing(n):
    rng = np.random.default_rng(n)
    g = rng.standard_normal((300, n, n))
    a = sd(g @ np.swapaxes(g, -1, -2) + n * np.eye(n))
    t = sd(np.triu(g) + n * np.eye(n))
    b = sd(rng.standard_normal((300, n, 12)))
    lu = sdla.lu(a)
    calls = [lambda: sdla.solve(a, b), lambda: sdla.inv(a), lambda: sdla.det(a), lambda: sdla.lu(a),
             lambda: sdla.lu_solve(*lu, b), lambda: sdla.solve_triangular(t, b, upper=True),
             lambda: sdla.cholesky_solve(sdla.cholesky(a), b), lambda: sdla.matrix_power(a, -2),
             lambda: sdla.cond(a, ord=1)]
    for call in calls:
        capped = same_for_all_counts(call)
        _core._set_lu_task_cap(False)
        assert bits(call()) == capped
        _core._set_lu_task_cap(True)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_pool_packing_changes_nothing(layout):
    rng = np.random.default_rng(7)
    n = 400  # above the 2^17-element line
    g = rng.standard_normal((n, n))
    f = {"C": np.ascontiguousarray, "F": np.asfortranarray,
         "strided": lambda x: np.repeat(x, 2, axis=1)[:, ::2]}[layout]
    t = f(np.triu(g) + n * np.eye(n))
    t[5, 3] = np.nan  # below the diagonal: never read by an upper solve
    lu = sdla.lu(sd(g))
    L, U = (sd(f(np.asarray(v))) for v in (lu.L, lu.U))
    A, T, B = sd(f(g)), sd(t), sd(rng.standard_normal((n, 3)))
    calls = [lambda: sdla.solve(A, B), lambda: sdla.solve_triangular(T, B, upper=True),
             lambda: sdla.lu_solve(L, U, lu.perm, B), lambda: sdla.qr(A), lambda: sdla.kron(A, sd(np.eye(2)))]
    for call in calls:
        pooled = same_for_all_counts(call)
        _core._set_pack_parallel_min(1 << 60)  # always serial
        assert bits(call()) == pooled
        _core._set_pack_parallel_min(1 << 17)
    # NaN in a read element is still found when the packing is split
    t2 = t.copy()
    t2[n - 1, n - 2] = 0.0
    t2[n - 7, n - 3] = np.nan  # upper triangle, read
    with pytest.raises(ValueError, match="'a' contains NaN"):
        sdla.solve_triangular(sd(f(t2) if layout != "C" else np.ascontiguousarray(t2)), B, upper=True)
