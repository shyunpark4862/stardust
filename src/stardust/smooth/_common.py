"""The argument checks and messages sd.smooth's functions share (docs/api/smooth.md §1.1, §1.2; decisions SM6, SM9).

- Data (API.md §5): an sd.ndarray, a sequence of numbers, a numpy array or a frame Series, computed in float64;
  bool and int64 become float64; complex is a TypeError (Savitzky-Golay excepted).
- nan_policy: 'propagate' (every value of the result NaN), 'omit' (the points with a NaN in x, y or the weights
  left out), 'raise'. +-inf is always a ValueError, and so is a negative weight; at least one weight is positive.
- Options are keyword only; numbers follow the regression area's forms (an int or a float, operator.index for
  ints, OverflowError 'out of range').
"""

import math
import numbers
import operator

from .. import _core as _sd

_TAKES = "an sd.ndarray, a sequence of numbers, a numpy array or a frame Series"
POLICIES = ("propagate", "omit", "raise")


def type_name(v):
    return _sd._type_name(v)  # numpy's types keep their module (API.md §2.1)


def contiguous(a):
    """a as a C-ordered array (a view when it already is one)."""
    if a._flags()["c_contiguous"]:
        return a
    out = _sd.empty(tuple(a.shape), dtype=a.dtype)
    out[...] = a
    return out


def _is_numpy_scalar(x):
    return type(x).__module__ == "numpy" and getattr(x, "ndim", None) == 0


def array(x, func, arg):
    """x as a real float64 sd.ndarray of at least one dimension."""
    if isinstance(x, _sd.ndarray):
        a = x
    elif type(x).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if not isinstance(x, _frame.Series):
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {type_name(x)}")
        try:
            a = x.to_array(dtype=_sd.float64, na_value=math.nan)
        except TypeError, ValueError:
            raise TypeError(f"{func}(): '{arg}' must have a numeric column") from None
    else:
        if x is None or isinstance(x, (str, bytes)):
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {type_name(x)}")
        try:
            a = _sd.asarray(x)
        except (TypeError, ValueError) as e:
            msg = str(e)
            if isinstance(e, ValueError) and "0-dimensional" in msg:
                raise ValueError(f"{func}(): '{arg}' must be 1-D, not 0-D") from None
            if isinstance(e, ValueError):  # a ragged sequence, ...
                raise ValueError(f"{func}(): {msg.partition('(): ')[2] or msg}") from None
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {type_name(x)}") from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not complex128")
    if a.dtype == _sd.string:
        raise TypeError(f"{func}(): '{arg}' must be numeric, not string")
    return a if a.dtype == _sd.float64 else a.astype(_sd.float64)


def vector(x, func, arg):
    """A 1-D float64 C-contiguous array."""
    if isinstance(x, numbers.Number) or _is_numpy_scalar(x):
        raise ValueError(f"{func}(): '{arg}' must be 1-D, not 0-D")
    a = array(x, func, arg)
    if a.ndim != 1:
        raise ValueError(f"{func}(): '{arg}' must be 1-D, not {a.ndim}-D")
    return contiguous(a)


def points(x, func, arg="x_new"):
    """Evaluation points: (a C-contiguous 1-D float64 array, their shape); a number has shape ()."""
    if isinstance(x, numbers.Number) or _is_numpy_scalar(x):
        if isinstance(x, bool) or (isinstance(x, numbers.Complex) and not isinstance(x, numbers.Real)):
            raise TypeError(f"{func}(): '{arg}' must be real, not {type_name(x)}")
        try:
            v = float(x)
        except TypeError:
            raise TypeError(f"{func}(): '{arg}' must be real, not {type_name(x)}") from None
        except OverflowError:
            raise OverflowError(f"{func}(): '{arg}' is out of range") from None
        return _sd.asarray([v]), ()
    if x is None or isinstance(x, (str, bytes)):
        raise TypeError(f"{func}(): '{arg}' must be {_TAKES} or a number, not {type_name(x)}")
    a = array(x, func, arg)
    shape = tuple(a.shape)
    return contiguous(_sd.reshape(a, (a.size,))), shape


def real(v, func, arg):
    """v (an int or a float; numpy's too) as a float."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {type_name(v)}")
    try:
        return float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None


def integer(v, func, arg):
    if isinstance(v, bool) or (type(v).__module__ == "numpy" and type(v).__name__ == "bool"):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {type_name(v)}")
    try:
        k = operator.index(v)
    except TypeError:
        raise TypeError(f"{func}(): '{arg}' must be an int, not {type_name(v)}") from None
    if not -(2**63) <= k < 2**63:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return k


def choice(v, func, arg, choices):
    listed = ", ".join(f"'{c}'" for c in choices[:-1]) + f" or '{choices[-1]}'"
    if not isinstance(v, str):
        raise TypeError(f"{func}(): '{arg}' must be {listed}, not {type_name(v)}")
    if v not in choices:
        raise ValueError(f"{func}(): '{arg}' must be {listed}, not {v!r}")
    return v


def policy(v, func):
    return choice(v, func, "nan_policy", POLICIES)


def alpha(v, func):
    x = real(v, func, "alpha")
    if not 0 < x < 1:
        raise ValueError(f"{func}(): 'alpha' must be between 0 and 1 (exclusive)")
    return x


def scatter(func, x, y, weights, nan_policy):
    """The checked 1-D data of a scatterplot smoother: (x, y, w or None, keep or None, has_nan).

    x, y and w are C-contiguous float64 of one length n >= 1. keep: the bool mask of the points kept by
    nan_policy='omit' when it dropped some (x, y, w are then the kept points); has_nan: True when
    nan_policy='propagate' met a NaN (the caller returns NaN everywhere)."""
    nan_policy = policy(nan_policy, func)
    x = vector(x, func, "x")
    y = vector(y, func, "y")
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"{func}(): 'x' and 'y' must have the same length, not {n} and {y.shape[0]}")
    w = None
    if weights is not None:
        w = vector(weights, func, "weights")
        if w.shape[0] != n:
            raise ValueError(f"{func}(): 'weights' must have one value per point ({n}), not {w.shape[0]}")
    if n == 0:
        raise ValueError(f"{func}(): 'x' and 'y' must have at least 1 point")
    named = [("x", x), ("y", y)] + ([("weights", w)] if w is not None else [])
    for arg, a in named:
        if bool(_sd.any(_sd.isinf(a))):
            raise ValueError(f"{func}(): '{arg}' contains infinity")
    if w is not None and bool(_sd.any(w < 0)):
        raise ValueError(f"{func}(): 'weights' must be >= 0")
    bad = _sd.logical_or(_sd.isnan(x), _sd.isnan(y))
    if w is not None:
        bad = _sd.logical_or(bad, _sd.isnan(w))
    keep = None
    if bool(_sd.any(bad)):
        if nan_policy == "raise":
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
        if nan_policy == "propagate":
            return x, y, w, None, True
        keep = _sd.logical_not(bad)
        x, y = contiguous(x[keep]), contiguous(y[keep])
        w = contiguous(w[keep]) if w is not None else None
        if x.shape[0] == 0:
            raise ValueError(f"{func}(): no point is left without NaN (nan_policy='omit')")
    if w is not None and not bool(_sd.any(w > 0)):
        raise ValueError(f"{func}(): at least one weight must be positive")
    return x, y, w, keep, False


def scatter_back(values, keep, n):
    """values of the kept points at their places in an (n,) array of NaN (nan_policy='omit')."""
    if keep is None:
        return values
    out = _sd.full((n,), math.nan)
    out[keep] = values
    return out
