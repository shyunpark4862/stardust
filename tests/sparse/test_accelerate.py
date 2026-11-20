"""The Accelerate LU wrapper (csrc/accelerate/sparse.hpp) through the private entry point
stardust._core.sparse._lu_solve (csrc/sparse/direct.cpp): docs/design/sparse.md §1.4, §3.6.

What SP0 fixes and the tests hold it to: LU with AMD ordering solves real and complex systems
from CSC and CSR (the transpose flag), several right-hand sides at once; the same input gives the
same bits on every call, every sd thread count and from several Python threads at once (the GIL
is released); a structurally singular matrix is LinAlgError; a parameter Accelerate rejects comes
back as ValueError with Accelerate's own message (the reportError callback is set: with NULL the
process would stop); an exact zero pivot gives inf (zeroTolerance 0) instead of a finite wrong
solution."""

import threading

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import matrices as M

lu_solve = sd.sparse._core._lu_solve


def solve(a, b, fmt="csc"):
    indptr, indices, data = a.csc() if fmt == "csc" else a.csr()
    x = lu_solve(a.shape, sd.asarray(indptr), sd.asarray(indices), sd.asarray(data), sd.asarray(b), csr=fmt == "csr")
    return np.asarray(x)


def backward_error(a, x, b):
    d = a.dense()
    r = b - d @ x
    return np.abs(r).max() / (np.abs(d).sum(axis=1).max() * np.abs(x).max() + np.abs(b).max())


@pytest.mark.parametrize(
    "mid",
    ["lap2d/10", "lap3d/6", "convdiff2d/40", "random_unsym/1000x8", "random_indef/60", "saddle/40x10",
     "blockdiag/20x8", "banded/200x2x3", "random_complex/40", "random_herm/40", "graded/100x8"],
)  # fmt: skip
@pytest.mark.parametrize("fmt", ["csc", "csr"])
def test_lu_solves(mid, fmt):
    a = M.get(mid)
    b = M.rhs(a)
    x = solve(a, b, fmt)
    assert x.dtype == a.dtype and x.shape == b.shape
    assert backward_error(a, x, b) < 1e-14


def test_several_right_hand_sides():
    # Accelerate solves the columns together: each column agrees with its own solve to rounding,
    # not to the bit (an SP4 question, docs/design/sparse.md §9), and the batched result is the
    # same on every call.
    a = M.get("convdiff2d/10")
    b = M.rhs(a, 4)
    x = solve(a, b)
    assert x.shape == (100, 4)
    assert bits_equal(solve(a, b), x)
    for j in range(4):
        xj = solve(a, np.ascontiguousarray(b[:, j]))
        assert np.abs(x[:, j] - xj).max() <= 4e-16 * np.abs(xj).max()


def test_same_bits_on_every_call_and_thread_count(restore_threads):
    a = M.get("random_unsym/1000x8")
    b = M.rhs(a)
    first = solve(a, b)
    for t in (1, 2, 3, 7, 10):
        sd.set_num_threads(t)
        for _ in range(3):
            assert bits_equal(solve(a, b), first)


def test_same_bits_from_concurrent_python_threads():
    a = M.get("lap2d/30")
    b = M.rhs(a)
    first = solve(a, b)
    out = [None] * 8

    def run(k):
        out[k] = solve(a, b)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(bits_equal(x, first) for x in out)


@pytest.mark.parametrize("mid", ["struct_singular/40", "struct_singular/1000"])
def test_structurally_singular_is_linalg_error(mid):
    a = M.get(mid)
    with pytest.raises(sd.linalg.LinAlgError, match="singular matrix"):
        solve(a, M.rhs(a))


def test_exact_zero_pivot_gives_inf():
    # [[1, 2], [2, 4]]: numerically singular with an exactly zero pivot. Accelerate reports success;
    # with zeroTolerance 0 the solution is infinite (the SP4 check turns that into LinAlgError).
    a = M.Matrix("rank1", (2, 2), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([1.0, 2.0, 2.0, 4.0]),
                 frozenset())  # fmt: skip
    x = solve(a, np.array([1.0, 1.0]))
    assert not np.isfinite(x).any()


@pytest.mark.parametrize("mid", ["lap2d/10n", "lap2d/60n", "dupcol/40"])
def test_singular_in_value_gives_a_finite_huge_solution(mid):
    # Why SP4 needs a check beyond the status and inf (docs/design/sparse.md §3.6): these are
    # exactly singular, but the rounded pivots are not exactly zero; Accelerate reports success
    # and returns a finite solution of size about 1/eps along the null space (its normwise
    # backward error is small, so a residual check alone does not see it).
    a = M.get(mid)
    b = M.rhs(a)
    x = solve(a, b)
    assert np.isfinite(x).all() and np.abs(x).max() > 1e12 * np.abs(b).max()


def test_rejected_parameter_is_value_error_with_accelerates_message():
    a = M.get("rect/60x20")
    with pytest.raises(ValueError, match="only supports square matrices") as e:
        solve(a, M.rhs(a))
    assert not isinstance(e.value, sd.linalg.LinAlgError)


def test_empty_matrix_is_rejected_by_accelerate():
    # Accelerate takes no 0 x 0 matrix; SP4's spsolve answers it before calling (scipy returns
    # an empty solution).
    a = M.get("empty/0x0")
    with pytest.raises(ValueError, match="rowCount must be > 0"):
        solve(a, np.zeros(0))
