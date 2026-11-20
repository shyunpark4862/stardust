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


def _check_bool(v, func, arg):
    if v is not True and v is not False:
        raise TypeError(f"{func}(): '{arg}' must be True or False, not {_type_name(v)}")
    return v


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
    """An int >= lo within the kernels' int64 (a default such as least_squares' max_nfev can pass
    2^31 for large n, so the user's value may too)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {_type_name(v)}")
    v = operator.index(v)
    if v < lo:
        raise ValueError(f"{func}(): '{arg}' must be >= {lo}")
    if v >= 2**63:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _values_text(n):
    """'1 value' or 'n values' (messages that count what a function returned)."""
    return "1 value" if n == 1 else f"{n} values"


def _is_number(v):
    """v is a real number or a 0-d array (one value, not a vector or a matrix)."""
    if isinstance(v, numbers.Number):
        return True
    return getattr(v, "ndim", None) == 0


def _not_a_pair(v):
    """What a function returned in place of a pair, for the message: 'a list of 3 numbers' (a
    vector where a pair was expected), '3 values', or the type's name."""
    if not isinstance(v, (tuple, list)):
        return _type_name(v)
    if len(v) > 0 and all(_is_number(t) for t in v):
        kind = "a list" if isinstance(v, list) else "a tuple"
        return f"{kind} of {len(v)} number" + ("" if len(v) == 1 else "s")
    return _values_text(len(v))


def _show_x(xs, limit=10):
    """A point for a message: all its values up to limit, else the first few and the count, so
    that a message about a large problem stays short."""
    xs = list(xs)
    if len(xs) <= limit:
        return str(xs)
    return "[" + ", ".join(repr(v) for v in xs[:5]) + f", ...] ({len(xs)} values)"


def _as_array(v, func, arg):
    """v as a float64 sd.ndarray: sd.ndarrays, sequences of numbers and numpy arrays; bool and
    int64 are converted, complex is a TypeError."""
    a = _as_array_raw(v, func, arg)
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    return a


def _as_array_raw(v, func, arg):
    """_as_array before the conversion to float64: the array of v in its own real dtype (bool,
    int64 or float64)."""
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
    return a


def _returned_real(a, func, what):
    """A value a user function returned (an sd.ndarray from _as_array_raw) as float64: a bool
    array is a TypeError (a comparison returned where numbers were meant; a Python bool is one
    too)."""
    if a.dtype == _sd.bool:
        raise TypeError(f"{func}(): '{what}' must return real values, not a bool array")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    return a


def _x0(v, func, arg="x0"):
    """The starting point: a 1-dimensional, non-empty, finite float64 sd.ndarray (a copy), from an
    sd.ndarray, a sequence of numbers, a numpy array or a frame Series (a data-taking argument,
    API.md §2)."""
    from .. import frame as _frame

    if isinstance(v, _frame.Series):
        try:
            v = v.to_array(dtype=_sd.float64, na_value=math.nan)  # a missing value fails below
        except TypeError, ValueError:
            raise TypeError(f"{func}(): '{arg}' must be numeric, not a Series of {v.dtype}") from None
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
            raise ValueError(f"{func}(): '{arg}' must have {_values_text(n)} like {like}, not {len(out)}")
    if positive and not all(math.isfinite(t) and t > 0 for t in out):
        raise ValueError(f"{func}(): '{arg}' must be finite and > 0")
    return out


def _is_pair(v):
    """v is a sequence of two items (an sd.ndarray or numpy array of shape (2,) too)."""
    if isinstance(v, (str, bytes, dict)) or not hasattr(v, "__len__"):
        return False
    try:
        return len(v) == 2 and getattr(v, "ndim", 1) == 1
    except TypeError:  # (a 0-d array)
        return False


def _bounds(bounds, func, n, allow_equal=False, like="like 'x0'"):
    """None (no bounds) or a pair (lower, upper) of a number or n numbers each (regression.nls's
    form): lists of floats, +-inf for none. allow_equal: a lower bound may equal its upper bound
    (minimize: the variable is fixed, as in scipy). like: what n counts, for the message (e.g.
    "(one per column of 'A')").

    scipy's other form, a sequence of (low, high) pairs one per variable, is not taken: with
    other than two items it is named in the error; two pairs are read as (lower, upper)."""
    if bounds is None:
        return [-math.inf] * n, [math.inf] * n
    if isinstance(bounds, (str, bytes)) or not hasattr(bounds, "__len__"):
        raise TypeError(f"{func}(): 'bounds' must be None or a pair (lower, upper), not {_type_name(bounds)}")
    if len(bounds) != 2:  # the right kind, the wrong length
        if len(bounds) > 0 and all(_is_pair(bounds[i]) for i in range(len(bounds))):
            raise ValueError(f"{func}(): 'bounds' must be a pair (lower, upper), not a sequence of (low, high) pairs")
        raise ValueError(f"{func}(): 'bounds' must be a pair (lower, upper), not {_values_text(len(bounds))}")
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
            raise ValueError(f"{func}(): 'bounds' must have {_values_text(n)} {like}, not {len(out)}")
        sides.append(out)
    lb, ub = sides
    if any(math.isnan(v) for v in lb + ub):
        raise ValueError(f"{func}(): 'bounds' contains NaN")
    if allow_equal:
        if not all(a <= b for a, b in zip(lb, ub)):
            raise ValueError(f"{func}(): each lower bound must not be greater than its upper bound")
    elif not all(a < b for a, b in zip(lb, ub)):
        raise ValueError(f"{func}(): each lower bound must be less than its upper bound")
    return lb, ub


# scipy's names of the caps and the name each takes here (O14), by what the name caps
_CAP_HINTS = {
    "maxiter": " (the iteration cap is 'max_iter')",
    "maxfev": " (the cap on function evaluations is 'max_nfev')",
    "maxfun": " (the cap on function evaluations is 'max_nfev')",
    "maxls": " (the cap on line-search steps is 'max_ls')",
    "subproblem_maxiter": " (the cap on the subproblem's iterations is 'subproblem_max_iter')",
}


def _check_options(options, func, method, known, hints=None):
    """options (None or a dict) as a new dict, every key one the method knows (O16): another key
    is a ValueError, scipy's names of the caps with the name used here (O14; hints: a function's
    own, e.g. root's only cap is 'max_nfev'). A key whose value is None is left out: None means
    the default for every option, and conflicts (with tol) are only between values given."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError(f"{func}(): 'options' must be a dict or None, not {_type_name(options)}")
    table = _CAP_HINTS if hints is None else {**_CAP_HINTS, **hints}
    for key in options:
        if key not in known:
            raise ValueError(f"{func}(): method={method!r} has no option {key!r}{table.get(key, '')}")
    return {k: v for k, v in options.items() if v is not None}


def _finite_real(v, func, arg):
    """A finite real number (an int or a float) as a float."""
    x = _real(v, func, arg)
    if not math.isfinite(x):
        raise ValueError(f"{func}(): '{arg}' must be finite")
    return x


def _real_pair(v, func, arg, names, counts=(2,)):
    """A sequence (tuple, list, sd.ndarray, numpy array) of len in counts finite real numbers,
    as a list of floats; names is how the message calls the pair, e.g. '(a, b)'."""
    if isinstance(v, (str, bytes, dict)) or not hasattr(v, "__len__") or not hasattr(v, "__getitem__"):
        raise TypeError(f"{func}(): '{arg}' must be a {names}, not {_type_name(v)}")
    if len(v) not in counts:
        raise ValueError(f"{func}(): '{arg}' must be a {names}, not {_values_text(len(v))}")
    out = []
    for t in (v[i] for i in range(len(v))):
        if isinstance(t, bool) or not isinstance(t, numbers.Real):
            raise TypeError(f"{func}(): the values of '{arg}' must be real numbers, not {_type_name(t)}")
        t = _real(t, func, arg)  # (an int beyond the doubles: "'arg' is out of range")
        if not math.isfinite(t):
            raise ValueError(f"{func}(): the values of '{arg}' must be finite")
        out.append(t)
    return out


def _scalar_check(func):
    """check(value, what) for the scalar kernels: a value the user's function returned that is not
    a Python float, as one (a real number or a 0-d numpy array), or the error."""

    def check(v, what):
        if isinstance(v, bool) or v is None or isinstance(v, (str, bytes)):
            raise TypeError(f"{func}(): '{what}' must return a real number, not {_type_name(v)}")
        if isinstance(v, numbers.Real):
            return float(v)
        if isinstance(v, numbers.Complex):
            raise TypeError(f"{func}(): '{what}' must return a real number, not complex")
        shape = getattr(v, "shape", None)
        if shape is None and isinstance(v, (list, tuple)):
            shape = (len(v),)
        if shape is not None:
            shape = tuple(shape)
            kind = getattr(getattr(v, "dtype", None), "kind", "")
            if shape == () and kind and kind in "iuf":
                return float(v)
            if shape == () and kind == "c":
                raise TypeError(f"{func}(): '{what}' must return a real number, not complex")
            if shape == ():
                raise TypeError(f"{func}(): '{what}' must return a real number, not {_type_name(v)}")
            raise ValueError(f"{func}(): '{what}' must return a real number, not shape {shape}")
        raise TypeError(f"{func}(): '{what}' must return a real number, not {_type_name(v)}")

    return check
