"""Exact comparisons for the tests, one name per meaning of "the same":

    bits_equal(a, b)          the same bytes: shape, dtype and every bit, NaN payloads and the
                              sign of zero included
    bits_equal_any_nan(a, b)  as bits_equal, except that any NaN equals any NaN (a complex
                              value part by part); +0.0 and -0.0 still differ
    value_equal(a, b)         == element by element with NaN equal to NaN: +0.0 equals -0.0,
                              and the dtypes may differ (1 equals 1.0)

and one measure of "how far": max_rel_error(got, want), the largest element-wise relative error.

Each comparison takes Python or numpy scalars, numpy arrays, sd.ndarray (through the buffer protocol) or
tuples and lists of them (a result tuple, compared field by field; a list or tuple of numbers
is one 1-d array), and returns a bool. Shapes must agree; nothing broadcasts.
"""

import numpy as np


def _is_seq(x):
    return isinstance(x, (tuple, list))


def _number(x):
    return isinstance(x, (bool, int, float, complex, np.number, np.bool_))


def _array(x):
    a = np.asarray(x)
    if a.dtype == object:
        raise TypeError(f"cannot compare the bits of {type(x).__name__}")
    return np.ascontiguousarray(a)


def _compare(a, b, same):
    numbers_a = _is_seq(a) and all(map(_number, a))
    numbers_b = _is_seq(b) and all(map(_number, b))
    if (_is_seq(a) and not numbers_a) or (_is_seq(b) and not numbers_b):  # a result tuple
        if not (_is_seq(a) and _is_seq(b)) or len(a) != len(b):
            return False
        return all(_compare(x, y, same) for x, y in zip(a, b))
    if a is None or b is None or isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return a == b and type(a) is type(b)
    return _compare_arrays(_array(a), _array(b), same)


def _compare_arrays(a, b, same):
    return a.shape == b.shape and same(a, b)


def _bytes_same(a, b):
    return a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _bits_any_nan(a, b):
    if a.dtype != b.dtype:
        return False
    if a.dtype.kind == "c":
        a, b = a.view(a.real.dtype), b.view(b.real.dtype)  # the parts side by side
    if a.dtype.kind != "f":
        return a.tobytes() == b.tobytes()
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb)) and a[~na].tobytes() == b[~nb].tobytes()


def _values(a, b):
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        both_nan = _isnan(a) & _isnan(b)
        return bool(np.all((a == b) | both_nan))
    return bool(np.all(a == b))


def _isnan(a):
    if a.dtype.kind == "c":
        return np.isnan(a.real) | np.isnan(a.imag)
    return np.isnan(a) if a.dtype.kind == "f" else np.zeros(a.shape, dtype=bool)


def bits_equal(a, b):
    """The same shape, dtype and bytes (NaN payloads and signed zeros included)."""
    return _compare(a, b, _bytes_same)


def bits_equal_any_nan(a, b):
    """The same shape, dtype and bits, except that any NaN matches any NaN; -0.0 != +0.0."""
    return _compare(a, b, _bits_any_nan)


def value_equal(a, b):
    """The same shape and == element by element, NaN equal to NaN; -0.0 == +0.0, 1 == 1.0."""
    return _compare(a, b, _values)


def max_rel_error(got, want):
    """max |got - want| / |want| over the elements, as a float. Both are read as float64 arrays
    (sd.ndarray, numpy, lists; None reads as NaN) of one shape. A NaN anywhere, or a zero in
    `want` that `got` misses, makes it NaN or inf, so a bound on it fails."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if got.shape != want.shape:
        raise ValueError(f"max_rel_error: shapes {got.shape} and {want.shape} differ")
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.max(np.abs(got - want) / np.abs(want)))
