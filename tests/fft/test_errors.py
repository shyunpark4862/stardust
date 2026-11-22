"""The errors of sd.fft's FT1 and FT2 functions (docs/api/fft.md §1.4, §2, §5; API.md §2, §2.1): every wrong
argument, its type and its message, which starts with the function's name; the call shape (keyword-only
options, no workers/plan/overwrite_x/out/device) and numpy's and scipy's silent or internal cases that
are errors here."""

import re

import numpy as np
import pytest

import stardust as sd

F = sd.fft
X = sd.asarray(np.arange(6.0) + 0j)


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


@pytest.mark.parametrize("name", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"])
def test_transform_arguments(name):
    f = getattr(F, name)
    X = sd.asarray(np.arange(6.0)) if name in ("rfft", "ihfft") else globals()["X"]
    with raises(TypeError, f"{name}(): 'x' must be an sd.ndarray, not list"):
        f([1.0, 2.0])
    with raises(TypeError, f"{name}(): 'x' must be an sd.ndarray, not numpy.ndarray"):
        f(np.ones(4))
    with raises(TypeError, f"{name}(): 'n' must be an int or None, not float"):
        f(X, n=2.0)
    with raises(TypeError, f"{name}(): 'n' must be an int or None, not bool"):
        f(X, n=True)  # [scipy와 다름]: scipy takes True as 1
    with raises(ValueError, f"{name}(): 'n' must be at least 1, not 0"):
        f(X, n=0)
    with raises(ValueError, f"{name}(): 'n' must be at least 1, not -1"):
        f(X, n=-1)
    with raises(OverflowError, f"{name}(): 'n' is out of range"):
        f(X, n=2**64)
    with raises(TypeError, f"{name}(): 'axis' must be an int, not NoneType"):
        f(X, axis=None)
    with raises(ValueError, f"{name}(): axis 1 is out of bounds for an array of dimension 1"):
        f(X, axis=1)  # [numpy와 다름]: IndexError
    with raises(ValueError, f"{name}(): axis -3 is out of bounds for an array of dimension 2"):
        f(sd.ones((2, 3)), axis=-3)
    with raises(TypeError, f"{name}(): 'norm' must be 'backward', 'ortho' or 'forward', not NoneType"):
        f(X, norm=None)  # [numpy와 다름]: numpy takes None as 'backward'
    with raises(ValueError, f"{name}(): 'norm' must be 'backward', 'ortho' or 'forward', not 'bad'"):
        f(X, norm="bad")
    with raises(ValueError, f"{name}(): cannot transform an axis of length 0"):
        f(sd.zeros(0))
    with raises(ValueError, f"{name}(): cannot transform an axis of length 0"):
        f(sd.zeros((3, 0)))
    assert f(sd.zeros((3, 0)), n=2).shape == (3, 2)  # n gives the length: zeros padded


@pytest.mark.parametrize("name", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"])
def test_call_shape(name):
    f = getattr(F, name)
    with raises(TypeError, f"{name}(): takes 1 positional argument but 2 were given"):
        f(X, 4)  # FT6: options are keyword-only
    for kw in ("workers", "overwrite_x", "plan", "out", "device"):
        with raises(TypeError, f"{name}(): unexpected keyword argument '{kw}'"):
            f(X, **{kw: None})


@pytest.mark.parametrize("name", ["rfft", "ihfft"])
def test_real_transforms_refuse_complex_input(name):
    # numpy and scipy refuse it too (scipy: "x must be a real sequence"; tests/fft/data/scipy.json)
    with raises(TypeError, f"{name}(): 'x' must be real, not complex128"):
        getattr(F, name)(X)


@pytest.mark.parametrize("name", ["irfft", "hfft"])
def test_hermitian_default_length(name):
    # the default n = 2 (m - 1): an axis of length 1 gives 0 (numpy and scipy: ValueError too)
    f = getattr(F, name)
    with raises(ValueError, f"{name}(): the default 'n' = 2 (m - 1) is 0 for an axis of length m = 1; give 'n'"):
        f(sd.asarray(np.array([2.0 + 1j])))
    with raises(ValueError, f"{name}(): cannot transform an axis of length 0"):
        f(sd.zeros((2, 0)))
    with raises(ValueError, f"{name}(): 'n' must be at least 1, not 0"):
        f(X, n=0)  # [scipy와 다름]: scipy's irfft(n=0) returns one value
    assert f(sd.asarray(np.array([2.0 + 1j])), n=1).shape == (1,)
    assert f(sd.zeros((2, 0)), n=3).shape == (2, 3)  # n gives the length: zeros padded


def test_memory():
    with pytest.raises(
        MemoryError, match=r"^fft\(\): cannot allocate \d+ bytes: more than this machine's physical memory$"
    ):
        F.fft(sd.ones(4), n=2**58)
    for name in ("rfft", "irfft", "hfft", "ihfft"):
        with pytest.raises(MemoryError, match=rf"^{name}\(\): cannot allocate \d+ bytes"):
            getattr(F, name)(sd.ones(4), n=2**58)


def test_frequency_arguments():
    for name in ("fftfreq", "rfftfreq"):
        f = getattr(F, name)
        with raises(ValueError, f"{name}(): 'n' must be at least 1, not 0"):
            f(0)  # [numpy와 다름]: ZeroDivisionError
        with raises(ValueError, f"{name}(): 'n' must be at least 1, not -3"):
            f(-3)
        with raises(TypeError, f"{name}(): 'n' must be an int, not float"):
            f(5.0)
        with raises(TypeError, f"{name}(): 'n' must be an int, not bool"):
            f(True)
        with raises(OverflowError, f"{name}(): 'n' is out of range"):
            f(2**63)
        with raises(ValueError, f"{name}(): 'd' must be finite and nonzero, not 0.0"):
            f(4, d=0.0)  # [numpy와 다름]: ZeroDivisionError
        with raises(ValueError, f"{name}(): 'd' must be finite and nonzero, not 0.0"):
            f(4, d=0)
        with raises(ValueError, f"{name}(): 'd' must be finite and nonzero, not inf"):
            f(4, d=float("inf"))
        with raises(ValueError, f"{name}(): 'd' must be finite and nonzero, not nan"):
            f(4, d=float("nan"))
        with raises(TypeError, f"{name}(): 'd' must be an int or a float, not str"):
            f(4, d="1")
        with raises(TypeError, f"{name}(): 'd' must be an int or a float, not complex"):
            f(4, d=1j)
        with raises(TypeError, f"{name}(): takes 1 positional argument but 2 were given"):
            f(4, 0.1)
        with raises(TypeError, f"{name}(): unexpected keyword argument 'device'"):
            f(4, device="cpu")


def test_shift_arguments():
    x = sd.arange(6).reshape((2, 3))
    for name in ("fftshift", "ifftshift"):
        f = getattr(F, name)
        with raises(TypeError, f"{name}(): 'x' must be an sd.ndarray, not list"):
            f([1, 2, 3])
        with raises(ValueError, f"{name}(): repeated axis"):
            f(x, axes=(0, 0))  # [numpy와 다름]: numpy rolls twice
        with raises(ValueError, f"{name}(): repeated axis"):
            f(x, axes=[1, -1])
        with raises(ValueError, f"{name}(): axis 2 is out of bounds for an array of dimension 2"):
            f(x, axes=2)
        with raises(TypeError, f"{name}(): 'axes' must be an int or a tuple or list of ints, not float"):
            f(x, axes=1.0)
        with raises(TypeError, f"{name}(): 'axes' must be an int or a tuple or list of ints, not bool"):
            f(x, axes=(True,))
        with raises(TypeError, f"{name}(): takes 1 positional argument but 2 were given"):
            f(x, 0)


def test_fast_len_arguments():
    for name in ("next_fast_len", "prev_fast_len"):
        f = getattr(F, name)
        with raises(ValueError, f"{name}(): 'target' must be at least 1, not 0"):
            f(0)  # [scipy와 다름]: scipy returns 0
        with raises(ValueError, f"{name}(): 'target' must be at least 1, not -1"):
            f(-1)
        with raises(TypeError, f"{name}(): 'target' must be an int, not bool"):
            f(True)  # [scipy와 다름]: scipy takes True as 1
        with raises(TypeError, f"{name}(): 'target' must be an int, not float"):
            f(7.0)
        with raises(OverflowError, f"{name}(): 'target' is out of range"):
            f(2**63)
        with raises(TypeError, f"{name}(): 'real' must be True or False, not int"):
            f(5, real=1)
        with raises(TypeError, f"{name}(): 'real' must be True or False, not NoneType"):
            f(5, real=None)
        with raises(TypeError, f"{name}(): takes 1 positional argument but 2 were given"):
            f(5, True)
