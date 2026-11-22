"""The errors of sd.interp (docs/api/array.md §8.5; API.md §2, §2.1): kind and message, named by the function."""

import math

import numpy as np
import pytest

import stardust as sd

A = sd.asarray
NOT_X = "interp(): 'x' must be an sd.ndarray or a bool, int or float, not "
XP, FP = A([0.0, 1.0]), A([0.0, 1.0])

CASES = [
    # (call, exception, message)
    (lambda: sd.interp(A([0.5]), A([0.0, 2.0, 1.0]), A([0.0, 1.0, 2.0])), ValueError,
     "interp(): 'xp' must be nondecreasing"),
    (lambda: sd.interp(A([0.5]), A([0.0, math.nan]), FP), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A([0.0, math.inf]), FP), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A([math.nan, 0.0]), FP, period=1.0), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A(np.zeros(0)), A(np.zeros(0))), ValueError, "interp(): 'xp' must not be empty"),
    (lambda: sd.interp(A([0.5]), XP, A([1.0])), ValueError, "interp(): 'fp' must be as long as 'xp'"),
    (lambda: sd.interp(A([0.5]), A([[0.0, 1.0]]), A([[0.0, 1.0]])), ValueError, "interp(): 'xp' must be 1-D"),
    (lambda: sd.interp(A([0.5]), XP, A([[0.0], [1.0]])), ValueError, "interp(): 'fp' must be 1-D"),
    (lambda: sd.interp(A([0.5j]), XP, FP), TypeError, "interp(): 'x' must be real, not complex128"),
    (lambda: sd.interp(0.5j, XP, FP), TypeError, "interp(): 'x' must be real, not complex"),
    (lambda: sd.interp(0.5, A([0j, 1.0]), FP), TypeError, "interp(): 'xp' must be real, not complex128"),
    (lambda: sd.interp([0.5], XP, FP), TypeError, NOT_X + "list"),
    (lambda: sd.interp(np.array([0.5]), XP, FP), TypeError, NOT_X + "numpy.ndarray"),
    (lambda: sd.interp(None, XP, FP), TypeError, NOT_X + "NoneType"),
    (lambda: sd.interp(0.5, [0.0, 1.0], FP), TypeError, "interp(): 'xp' must be an sd.ndarray, not list"),
    (lambda: sd.interp(0.5, XP, (0.0, 1.0)), TypeError, "interp(): 'fp' must be an sd.ndarray, not tuple"),
    (lambda: sd.interp(0.5, XP, FP, period=0), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=-0.0), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=math.nan), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=math.inf), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period="1"), TypeError, "interp(): 'period' must be an int or a float, not str"),
    (lambda: sd.interp(0.5, XP, FP, period=True), TypeError, "interp(): 'period' must be an int or a float, not bool"),
    (lambda: sd.interp(0.5, XP, FP, left=1j), TypeError, "interp(): 'left' must be an int or a float, not complex"),
    (lambda: sd.interp(0.5, XP, FP, right="a"), TypeError, "interp(): 'right' must be an int or a float, not str"),
    (lambda: sd.interp(0.5, XP, A([0j, 1.0]), left="a"), TypeError, "interp(): 'left' must be a number, not str"),
    (lambda: sd.interp(0.5, XP, FP, period=10**400), OverflowError, "interp(): 'period' is out of range"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES)))
def test_error(k):
    call, exc, msg = CASES[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_call_shape():
    # numpy takes left and right by position; sd only by keyword (docs/api/array.md §8.5)
    with pytest.raises(TypeError, match=r"^interp\(\): incompatible function arguments"):
        sd.interp(A([0.5]), XP, FP, 0.0, 1.0)
    with pytest.raises(TypeError, match=r"^interp\(\): incompatible function arguments"):
        sd.interp(x=A([0.5]), xp=XP, fp=FP)


def test_unsorted_xp_with_a_period_is_sorted_not_refused():
    assert sd.interp(0.5, A([1.0, 0.0]), A([1.0, 0.0]), period=2.0) == 0.5
