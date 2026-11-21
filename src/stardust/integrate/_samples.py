"""The sample rules: trapezoid, simpson, romb, cumulative_trapezoid and cumulative_simpson
(docs/api/integrate.md §8; docs/design/integrate.md §3.7 and decisions C39-C41, C44-C50;
scipy 1.18.1's integrate/_quadrature.py).

This module checks and converts the arguments; the rules run in
stardust._core.integrate.samples (csrc/integrate/samples.cpp): scipy's terms, each lane summed
with the library's pairwise sum and the cumulative ones scanned as cumulative_sum scans.

The data follow the data-taking functions' rule (API.md §5, C39): an sd.ndarray, a sequence of
numbers, a numpy array or a frame Series. y is float64 (bool and int64 converted) or complex128;
x is real. x is 1-dimensional with y's length along `axis`, or has y's number of dimensions and
broadcasts to y's shape (C45). dx is a number or an array of y's shape with 1 along `axis` (one
spacing per lane).
"""

import math
import numbers
import operator

from .. import _core as _sd
from ..optimize._common import _real, _type_name

_k = _sd.integrate

TRAPEZOID, SIMPSON, ROMB, CUMULATIVE_TRAPEZOID, CUMULATIVE_SIMPSON, WEIGHTED = range(6)

_TAKES = "an sd.ndarray, a sequence of numbers or a frame Series"


def _shape_text(shape):
    return "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"


def _is_number(v):
    return isinstance(v, numbers.Number) and not isinstance(v, bool)


def as_data(v, func, arg, complex_ok):
    """v as an sd.ndarray of float64 (or complex128 when complex_ok): the data-taking functions'
    inputs (API.md §5). A scalar is a ValueError (the rules need points along an axis)."""
    if isinstance(v, _sd.ndarray):
        a = v
    elif isinstance(v, (numbers.Number, bool)) or (type(v).__module__ == "numpy" and getattr(v, "ndim", None) == 0):
        if isinstance(v, bool) or type(v).__name__ == "bool":
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {_type_name(v)}")
        raise ValueError(f"{func}(): '{arg}' must be at least 1-dimensional, not a scalar")
    elif type(v).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if not isinstance(v, _frame.Series):
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {_type_name(v)}")
        try:
            a = v.to_array(dtype=_sd.float64, na_value=math.nan)
        except TypeError, ValueError:  # a string column is a ValueError of to_array
            raise TypeError(f"{func}(): '{arg}' must be a numeric Series") from None
    else:
        if isinstance(v, (str, bytes)) or v is None:
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {_type_name(v)}")
        try:
            a = _sd.asarray(v)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
        except (TypeError, ValueError) as e:
            msg = str(e)
            if isinstance(e, ValueError) and "0-dimensional" not in msg:  # a ragged sequence
                raise ValueError(f"{func}(): {msg.partition('(): ')[2] or msg}") from None
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {_type_name(v)}") from None
    if a.dtype == _sd.string:
        raise TypeError(f"{func}(): '{arg}' must be numeric, not a string array")
    if a.dtype == _sd.complex128:
        if not complex_ok:
            raise TypeError(f"{func}(): '{arg}' must be real, not a complex128 array")
        return a
    return a if a.dtype == _sd.float64 else a.astype(_sd.float64)


def _axis(axis, ndim, func):
    if isinstance(axis, bool) or not isinstance(axis, numbers.Integral):
        raise TypeError(f"{func}(): 'axis' must be an int, not {_type_name(axis)}")
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f"{func}(): axis {axis} is out of bounds for an array of dimension {ndim}")
    return axis % ndim


def lane_shape(shape, axis):
    return tuple(1 if d == axis else s for d, s in enumerate(shape))


def _x(x, y, axis, func):
    """x as a float64 view of y's shape, and x as given (converted), for the checks."""
    a = as_data(x, func, "x", complex_ok=False)
    n = y.shape[axis]
    if a.ndim == 1:
        if a.shape[0] != n:
            raise ValueError(f"{func}(): 'x' must have y's length {n} along 'axis', not {a.shape[0]}")
        if y.ndim == 1:
            return a, a, 0
        shape = tuple(n if d == axis else 1 for d in range(y.ndim))
        return _sd.broadcast_to(_sd.reshape(a, shape), tuple(y.shape)), a, 0
    if a.ndim != y.ndim:
        raise ValueError(f"{func}(): 'x' must be 1-dimensional or have the {y.ndim} dimensions of 'y', not {a.ndim}")
    if a.shape[axis] != n:
        raise ValueError(f"{func}(): 'x' must have y's length {n} along 'axis', not {a.shape[axis]}")
    try:
        view = _sd.broadcast_to(a, tuple(y.shape))
    except ValueError:
        raise ValueError(
            f"{func}(): 'x' of shape {_shape_text(a.shape)} does not broadcast to the shape "
            f"{_shape_text(y.shape)} of 'y'"
        ) from None
    return view, a, axis


def _per_lane(v, y, axis, func, arg, complex_ok):
    """A per-lane value: a number (returned as a float or complex), or an array of y's shape with
    1 along the axis (returned as an sd.ndarray of y's dtype)."""
    if _is_number(v) or (type(v).__module__ == "numpy" and getattr(v, "ndim", None) == 0 and not isinstance(v, bool)):
        if isinstance(v, complex) or (hasattr(v, "dtype") and getattr(v.dtype, "kind", "") == "c"):
            if not complex_ok:
                raise TypeError(
                    f"{func}(): '{arg}' must be real{' for a real y' if arg == 'initial' else ''}, not complex"
                )
            return complex(v)
        if type(v).__name__ == "bool":
            raise TypeError(f"{func}(): '{arg}' must be a number or an array, not numpy.bool")
        return _real(v, func, arg)
    if isinstance(v, bool):
        raise TypeError(f"{func}(): '{arg}' must be a number or an array, not bool")
    if v is None or isinstance(v, (str, bytes)):
        raise TypeError(f"{func}(): '{arg}' must be a number or an array, not {_type_name(v)}")
    a = as_data(v, func, arg, complex_ok=True)
    if a.dtype == _sd.complex128 and not complex_ok:
        raise TypeError(
            f"{func}(): '{arg}' must be real{' for a real y' if arg == 'initial' else ''}, not a complex128 array"
        )
    want = lane_shape(tuple(y.shape), axis)
    if tuple(a.shape) != want:
        raise ValueError(
            f"{func}(): '{arg}' must be a number or an array of shape {_shape_text(want)} (the shape of 'y' with 1 "
            f"along 'axis'), not shape {_shape_text(a.shape)}"
        )
    return a if a.dtype == y.dtype else a.astype(y.dtype)


def _inputs(func, y, x, dx, axis):
    ya = as_data(y, func, "y", complex_ok=True)
    ax = _axis(axis, ya.ndim, func)
    xv = xa = None
    x_axis = 0
    if x is not None:
        if not (_is_number(dx) and dx == 1.0):
            raise ValueError(f"{func}(): 'dx' is not used with 'x' (the spacing comes from 'x')")
        xv, xa, x_axis = _x(x, ya, ax, func)
        dxv = None
    else:
        dxv = _per_lane(dx, ya, ax, func, "dx", complex_ok=False)
    return ya, xv, xa, x_axis, dxv, ax


def trapezoid(y, /, x=None, *, dx=1.0, axis=-1):
    """The integral of the samples y along `axis` by the composite trapezoidal rule
    (scipy.integrate.trapezoid; docs/api/integrate.md §8)."""
    func = "trapezoid"
    ya, xv, _, _, dxv, ax = _inputs(func, y, x, dx, axis)
    return _k.samples(func, TRAPEZOID, ya, xv, dxv, ax, None)


def simpson(y, /, x=None, *, dx=1.0, axis=-1):
    """The integral of the samples y along `axis` by the composite Simpson rule, the last
    interval of an even number of points by Cartwright's correction (scipy 1.11's
    scipy.integrate.simpson; docs/api/integrate.md §8)."""
    func = "simpson"
    ya, xv, _, _, dxv, ax = _inputs(func, y, x, dx, axis)
    return _k.samples(func, SIMPSON, ya, xv, dxv, ax, None)


def romb(y, /, *, dx=1.0, axis=-1):
    """Romberg integration of 2^k + 1 equally spaced samples along `axis`
    (scipy.integrate.romb; docs/api/integrate.md §8)."""
    func = "romb"
    ya, _, _, _, dxv, ax = _inputs(func, y, None, dx, axis)
    n = ya.shape[ax]
    if n < 2 or (n - 1) & (n - 2):
        raise ValueError(f"{func}(): 'y' must have 2^k + 1 samples along 'axis' (k >= 0), not {n}")
    return _k.samples(func, ROMB, ya, None, dxv, ax, None)


def _at_least_one(ya, ax, func):
    if ya.shape[ax] == 0:
        raise ValueError(f"{func}(): 'y' must have at least one point along 'axis'")


def cumulative_trapezoid(y, /, x=None, *, dx=1.0, axis=-1, initial=None):
    """The running integral of the samples y along `axis` by the trapezoidal rule
    (scipy.integrate.cumulative_trapezoid; docs/api/integrate.md §8)."""
    func = "cumulative_trapezoid"
    ya, xv, _, _, dxv, ax = _inputs(func, y, x, dx, axis)
    _at_least_one(ya, ax, func)
    if initial is not None:
        if not _is_number(initial) or isinstance(initial, complex):
            raise TypeError(f"{func}(): 'initial' must be None or 0, not {_type_name(initial)}")
        if initial != 0:
            raise ValueError(f"{func}(): 'initial' must be None or 0, not {initial!r}")
        initial = complex(initial) if ya.dtype == _sd.complex128 else float(initial)
    return _k.samples(func, CUMULATIVE_TRAPEZOID, ya, xv, dxv, ax, initial)


def cumulative_simpson(y, /, x=None, *, dx=1.0, axis=-1, initial=None):
    """The running integral of the samples y along `axis` by Simpson's rule on every interval
    (Cartwright's formulas; scipy.integrate.cumulative_simpson; docs/api/integrate.md §8)."""
    func = "cumulative_simpson"
    ya, xv, xa, x_axis, dxv, ax = _inputs(func, y, x, dx, axis)
    _at_least_one(ya, ax, func)
    if xa is not None and ya.shape[ax] >= 3 and bool(_sd.any(_sd.diff(xa, axis=x_axis) <= 0)):
        raise ValueError(f"{func}(): 'x' must be strictly increasing along 'axis'")
    if initial is not None:
        initial = _per_lane(initial, ya, ax, func, "initial", complex_ok=ya.dtype == _sd.complex128)
        if isinstance(initial, float) and ya.dtype == _sd.complex128:
            initial = complex(initial)
    return _k.samples(func, CUMULATIVE_SIMPSON, ya, xv, dxv, ax, initial)
