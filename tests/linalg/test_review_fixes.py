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
