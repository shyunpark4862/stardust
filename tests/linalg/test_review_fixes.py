"""Bugs found by a code review (docs/design/linalg.md §10.5, §11.4)."""

import subprocess
import sys

import numpy as np
import pytest
import stardust as sd


# Each expm case runs in a child process with a time limit: before the fix
# these never returned (about 2^31 squarings), which would hang the suite.
_EXPM_CHILD = r"""
import sys, warnings
warnings.simplefilter("error")
import numpy as np, stardust as sd
a = eval(sys.argv[1])
r = np.asarray(sd.linalg.expm(sd.asarray(a)).tolist())
print(repr(r.tolist()))
"""


def expm_in_child(a):
    try:
        r = subprocess.run([sys.executable, "-c", _EXPM_CHILD, repr(a)], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail(f"expm did not return for {a!r}")
    assert r.returncode == 0, r.stderr
    return np.array(eval(r.stdout, {"nan": np.nan, "inf": np.inf, "nanj": np.nan}))


@pytest.mark.parametrize(
    "a",
    [
        [[1e-310, 1e-310], [1e-310, 1e-310]],  # ||A||_1 subnormal: 2^-e overflowed
        [[5e-324, 0.0, 5e-324], [5e-324, 5e-324, 0.0], [0.0, 5e-324, 5e-324]],
        [[1e-310 + 1e-310j, 1e-310], [1e-310j, 1e-310]],
    ],
)
def test_expm_subnormal_norm_returns(a):
    r = expm_in_child(a)
    assert np.all(np.isfinite(r))
    # exp(A) = I + A + O(A^2): A^2 underflows, so I + A up to a subnormal ulp or so
    want = np.eye(len(a)) + np.array(a)
    assert np.max(np.abs(r - want)) <= 1e-322


@pytest.mark.parametrize(
    "a",
    [
        [[1e308, 1e308], [1e308, 1e308]],  # finite entries, ||A||_1 = inf: the guard was skipped
        [[1.7e308] * 3] * 3,
        [[1e308 + 0j, 1e308j], [1e308j, -1e308]],
    ],
)
def test_expm_overflowing_norm_returns(a):
    # It returns now. The values are inf/NaN, as for ||A|| ~ 1e20 and up before
    # (scaling and squaring's limit; scipy does not return at all there).
    expm_in_child(a)


def test_expm_overflowing_norm_true_overflow_is_inf():
    r = expm_in_child([[1e308, 1e308], [1e308, 1e308]])
    assert np.all(np.isinf(r))


@pytest.mark.parametrize("func", [sd.linalg.cond, sd.linalg.matrix_norm])
@pytest.mark.parametrize("ord_", [10**400, -(10**400)])
def test_matrix_ord_too_large_for_float(func, ord_):
    # was: SystemError (OverflowError left pending, -1.0 taken as ord=-1);
    # now the OverflowError itself, as vector_norm raises
    a = sd.asarray([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(OverflowError):
        func(a, ord=ord_)
    with pytest.raises(OverflowError):
        sd.linalg.vector_norm(a, ord=ord_)


# ---- 2026-11-18: lstsq(rtol=0), the error's module path, the bits of matrix_norm ----


def test_lstsq_rtol_zero_counts_every_positive_singular_value():
    # xLALSD replaces rcond <= 0 by machine epsilon; rtol = 0 must mean what it means in
    # matrix_rank (only exact zeros are zero)
    a = sd.ones((3, 2))
    b = sd.ones((3,))
    r = sd.linalg.lstsq(a, b, rtol=0.0)
    assert min(r.singular_values.tolist()) > 0.0
    assert r.rank == sd.linalg.matrix_rank(a, rtol=0.0) == 2
    assert sd.linalg.lstsq(a, b, rtol=1e-300).rank == 2
    assert sd.linalg.lstsq(a, b).rank == sd.linalg.matrix_rank(a) == 1
    z = sd.linalg.lstsq(sd.zeros((3, 2)), b, rtol=0.0)
    assert z.rank == 0 and z.solution.tolist() == [0.0, 0.0]


def test_linalg_error_shows_its_public_path():
    assert sd.linalg.LinAlgError.__module__ == "stardust.linalg"
    with pytest.raises(sd.linalg.LinAlgError, match=r"^inv\(\): singular matrix"):
        sd.linalg.inv(sd.zeros((2, 2)))


@pytest.mark.parametrize("order", ["C", "F"])
def test_matrix_norm_sums_each_line_pairwise(order):
    # Each column (1, -1) or row (inf, -inf) is summed pairwise in its logical order, whatever
    # the memory layout: the bits of numpy's sum over that line as a contiguous vector, and of
    # sqrt(sum(x * x)) for 'fro'. numpy's own matrix norm adds the rows one after another when
    # the summed axis is not the contiguous one, so it differs in the last bits there.
    rng = np.random.default_rng(7)
    m = rng.standard_normal((300, 500))
    x = sd.asarray(np.asarray(m, order=order))
    cols = np.array([np.abs(np.ascontiguousarray(m[:, j])).sum() for j in range(m.shape[1])])
    rows = np.array([np.abs(np.ascontiguousarray(m[i, :])).sum() for i in range(m.shape[0])])
    assert float(sd.linalg.matrix_norm(x, ord=1)) == cols.max()
    assert float(sd.linalg.matrix_norm(x, ord=-1)) == cols.min()
    assert float(sd.linalg.matrix_norm(x, ord=float("inf"))) == rows.max()
    assert float(sd.linalg.matrix_norm(x, ord=float("-inf"))) == rows.min()
    assert float(sd.linalg.matrix_norm(x, ord="fro")) == float(np.sqrt((m * m).sum()))


def test_bits_shared_with_numpy_and_scipy():
    # docs/design/linalg.md §12.2: the same driver and the same call
    sl = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(3)
    for n in (3, 7, 40):
        a = rng.standard_normal((n, n))
        s = a @ a.T + n * np.eye(n)
        b = rng.standard_normal((n, 3))
        lower = sd.linalg.cholesky(sd.asarray(s))
        assert np.array_equal(np.asarray(lower.tolist()), np.linalg.cholesky(s))
        x = sd.linalg.cholesky_solve(lower, sd.asarray(b))
        assert np.array_equal(np.asarray(x.tolist()), sl.cho_solve((np.linalg.cholesky(s), True), b))
        got = sd.linalg.qr_pivoted(sd.asarray(a))
        q, r, perm = sl.qr(a, pivoting=True)
        assert np.array_equal(np.asarray(got[0].tolist()), q) and np.array_equal(np.asarray(got[1].tolist()), r)
        assert got[2].tolist() == perm.tolist()
        sol = sd.linalg.lstsq(sd.asarray(a), sd.asarray(b)).solution
        assert np.array_equal(np.asarray(sol.tolist()), np.linalg.lstsq(a, b, rcond=None)[0])
