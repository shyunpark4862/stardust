"""The argument checks of sd.optimize, in the message forms of the other areas (API.md §2.1,
docs/api/optimize.md §1.1, §1.5): 'func(): ...'."""

import math
import numbers
import operator

from .. import _core as _sd

_EPS = 2.0**-52  # the machine epsilon


def _type_name(v):
    return _sd._type_name(v)  # (numpy's types as numpy.bool: API.md §2.1)


def _callable(v, func, arg):
    if not callable(v):
        raise TypeError(f"{func}(): '{arg}' must be callable, not {_type_name(v)}")
    return v


def _check_choice(v, func, arg, choices):
    listed = ", ".join(f"'{c}'" for c in choices[:-1]) + f" or '{choices[-1]}'"
    if not isinstance(v, str):
        raise TypeError(f"{func}(): '{arg}' must be {listed}, not {_type_name(v)}")
    if v not in choices:
        raise ValueError(f"{func}(): '{arg}' must be {listed}, not {v!r}")
    return v


def _check_method(v, func, choices):
    """A method name: lowercase only (O15); another case names the lowercase form."""
    if isinstance(v, str) and v not in choices and v.lower() in choices:
        raise ValueError(f"{func}(): 'method' must be lowercase: {v.lower()!r}, not {v!r}")
    return _check_choice(v, func, "method", choices)


def _real(v, func, arg):
    """v (an int or a float; numpy's too) as a float: TypeError for anything else, OverflowError
    'out of range' for an int beyond the doubles (API.md §2)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {_type_name(v)}")
    try:
        return float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None


def _check_tol(v, func, arg):
    x = _real(v, func, arg)
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): '{arg}' must be a finite number >= 0")
    return x


def _check_int(v, func, arg, lo):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {_type_name(v)}")
    v = operator.index(v)
    if v < lo:
        raise ValueError(f"{func}(): '{arg}' must be >= {lo}")
    if v >= 2**31:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _as_array(v, func, arg):
    """v as a float64 sd.ndarray: sd.ndarrays, sequences of numbers and numpy arrays; bool and
    int64 are converted, complex is a TypeError."""
    if isinstance(v, _sd.ndarray):
        a = v
    elif isinstance(v, numbers.Number) or (hasattr(v, "ndim") and v.ndim == 0):
        if isinstance(v, complex) or (hasattr(v, "dtype") and getattr(v.dtype, "kind", "") == "c"):
            raise TypeError(f"{func}(): '{arg}' must be real, not complex")
        raise ValueError(f"{func}(): '{arg}' must be 1-dimensional, not a scalar")
    else:
        if isinstance(v, (str, bytes)) or v is None:
            raise TypeError(f"{func}(): '{arg}' must be an sd.ndarray or a sequence of numbers, not {_type_name(v)}")
        try:
            a = _sd.asarray(v)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
        except TypeError, ValueError:
            raise TypeError(
                f"{func}(): '{arg}' must be an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
            ) from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not a complex128 array")
    if a.dtype == _sd.string:
        raise TypeError(f"{func}(): '{arg}' must be real, not a string array")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    return a


def _x0(v, func, arg="x0"):
    """The starting point: a 1-dimensional, non-empty, finite float64 sd.ndarray (a copy)."""
    a = _as_array(v, func, arg)
    if a.ndim != 1:
        raise ValueError(f"{func}(): '{arg}' must be 1-dimensional, not shape {tuple(a.shape)}")
    if a.shape[0] == 0:
        raise ValueError(f"{func}(): '{arg}' is empty (no variables)")
    if not bool(_sd.all(_sd.isfinite(a))):
        raise ValueError(f"{func}(): '{arg}' must be finite")
    return a.copy()


def _real_vector(v, func, arg, n, positive=False, like="'x0'"):
    """A number (repeated n times) or n numbers, as a list of floats; finite (and > 0 when
    positive)."""
    if isinstance(v, bool):
        raise TypeError(f"{func}(): '{arg}' must be a number or a sequence of numbers, not bool")
    if isinstance(v, numbers.Real):
        out = [_real(v, func, arg)] * n
    else:
        if isinstance(v, (str, bytes)) or v is None:
            raise TypeError(f"{func}(): '{arg}' must be a number or a sequence of numbers, not {_type_name(v)}")
        try:
            a = _as_array(v, func, arg)
        except TypeError, ValueError:
            raise TypeError(
                f"{func}(): '{arg}' must be a number or a sequence of numbers, not {_type_name(v)}"
            ) from None
        if a.ndim != 1:
            raise ValueError(f"{func}(): '{arg}' must be a number or 1-dimensional")
        out = [float(t) for t in a.tolist()]
        if len(out) != n:
            raise ValueError(f"{func}(): '{arg}' must have {n} values like {like}, not {len(out)}")
    if positive and not all(math.isfinite(t) and t > 0 for t in out):
        raise ValueError(f"{func}(): '{arg}' must be finite and > 0")
    return out


def _bounds(bounds, func, n):
    """None (no bounds) or a pair (lower, upper) of a number or n numbers each (regression.nls's
    form): lists of floats, +-inf for none."""
    if bounds is None:
        return [-math.inf] * n, [math.inf] * n
    if isinstance(bounds, (str, bytes)) or not hasattr(bounds, "__len__"):
        raise TypeError(f"{func}(): 'bounds' must be None or a pair (lower, upper), not {_type_name(bounds)}")
    if len(bounds) != 2:  # the right kind, the wrong length
        raise ValueError(f"{func}(): 'bounds' must be a pair (lower, upper), not {len(bounds)} values")
    sides = []
    for side in bounds:
        if isinstance(side, bool):
            raise TypeError(f"{func}(): 'bounds' must be a number or a sequence of numbers, not bool")
        if isinstance(side, (str, bytes)) or side is None:
            raise TypeError(f"{func}(): 'bounds' must be a sequence of numbers, not {_type_name(side)}")
        if isinstance(side, numbers.Real):
            sides.append([float(side)] * n)
            continue
        try:
            a = _as_array(side, func, "bounds")
        except TypeError, ValueError:
            raise TypeError(f"{func}(): 'bounds' must be a sequence of numbers, not {_type_name(side)}") from None
        if a.ndim != 1:
            raise ValueError(f"{func}(): 'bounds' must be 1-dimensional")
        out = [float(t) for t in a.tolist()]
        if len(out) != n:
            raise ValueError(f"{func}(): 'bounds' must have {n} values like 'x0', not {len(out)}")
        sides.append(out)
    lb, ub = sides
    if any(math.isnan(v) for v in lb + ub):
        raise ValueError(f"{func}(): 'bounds' contains NaN")
    if not all(a < b for a, b in zip(lb, ub)):
        raise ValueError(f"{func}(): each lower bound must be less than its upper bound")
    return lb, ub
