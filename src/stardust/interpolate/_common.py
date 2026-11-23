"""The argument checks and messages sd.interpolate's functions share (docs/api/interpolate.md §1; decisions
IP6, IP11, IP13, IP14, IP19 in docs/design/interpolate.md §1.1).

- Data (API.md §5, IP6): an sd.ndarray, a sequence of numbers, a numpy array, a frame Series or a frame
  DataFrame (its columns side by side, as sd.spatial takes a point set), computed in float64; bool and int64
  become float64. The abscissae (nodes, breakpoints, evaluation points) are real:
  complex128 is a TypeError. The values may be complex128 where scipy takes them.
- Nodes are 1-D, finite and strictly increasing (breakpoints of a PPoly may strictly decrease).
- extrapolate= (IP13): True, False, 'periodic' or None.
- Options are keyword only (IP14); ints follow operator.index (API.md §2).
"""

import collections
import math
import numbers
import operator

from .. import _core as _sd

_core = _sd.interpolate

_TAKES = "an sd.ndarray, a sequence of numbers, a numpy array or a frame Series or DataFrame"
EXTRAPOLATE = ("True", "False", "'periodic'", "None")


def type_name(v):
    return _sd._type_name(v)  # numpy's types keep their module (API.md §2.1)


def contiguous(a):
    """a as a C-ordered array (a view when it already is one)."""
    if a._flags()["c_contiguous"]:
        return a
    out = _sd.empty(tuple(a.shape), dtype=a.dtype)
    out[...] = a
    return out


# ---- pickling: sd.ndarray does not pickle, so an object's slots go out with their arrays as bytes

_ArrayState = collections.namedtuple("_ArrayState", "raw shape kind readonly")
_KINDS = {"float64": "d", "int64": "q", "bool": "?", "complex128": "c"}


def _pack(v):
    sd = _sd
    if isinstance(v, sd.ndarray):
        a = contiguous(v)
        raw = memoryview(a).tobytes() if a.size else b""
        return _ArrayState(raw, tuple(v.shape), _KINDS[str(v.dtype)], not v._flags()["writeable"])
    if isinstance(v, tuple):
        return tuple(_pack(e) for e in v)
    if isinstance(v, list):
        return [_pack(e) for e in v]
    if isinstance(v, dict):
        return {k: _pack(e) for k, e in v.items()}
    return v


def _unpack(v):
    sd = _sd
    if isinstance(v, _ArrayState):
        dtype = {"d": sd.float64, "q": sd.int64, "?": sd.bool, "c": sd.complex128}[v.kind]
        if not v.raw:
            a = sd.zeros(v.shape, dtype=dtype)
        elif v.kind == "c":  # the parts side by side (memoryview has no complex format)
            parts = sd.reshape(sd.asarray(memoryview(v.raw).cast("d")), (-1, 2))
            a = sd.empty((parts.shape[0],), dtype=dtype)
            sd.real(a)[...] = parts[:, 0]
            sd.imag(a)[...] = parts[:, 1]
            a = sd.reshape(a, v.shape)
        else:
            a = sd.reshape(sd.asarray(memoryview(v.raw).cast(v.kind)).copy(), v.shape)
        if v.readonly:
            a._set_readonly()
        return a
    if type(v) is tuple:
        return tuple(_unpack(e) for e in v)
    if isinstance(v, list):
        return [_unpack(e) for e in v]
    if isinstance(v, dict):
        return {k: _unpack(e) for k, e in v.items()}
    return v


def reduce(obj):
    """obj's __reduce__: its class and its slots (every slot of every class of its mro that is set), the arrays as
    bytes. The object comes back with the same arrays to the bit, so it computes the same bits."""
    state = {}
    for cls in type(obj).__mro__:
        for name in getattr(cls, "__slots__", ()):
            if hasattr(obj, name):
                state[name] = _pack(getattr(obj, name))
    return _restore, (type(obj), state)


def _restore(cls, state):
    obj = cls.__new__(cls)
    for name, v in state.items():
        object.__setattr__(obj, name, _unpack(v))
    return obj


def _is_numpy_scalar(x):
    return type(x).__module__ == "numpy" and getattr(x, "ndim", None) == 0


def array(x, name, arg, *, complex_ok=False):
    """x as an sd.ndarray of at least one dimension: float64, or complex128 when complex_ok and x is complex."""
    sd = _sd
    if isinstance(x, sd.ndarray):
        a = x
    elif type(x).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if not isinstance(x, (_frame.Series, _frame.DataFrame)):
            raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {type_name(x)}")
        try:
            a = x.to_array(dtype=sd.float64, na_value=math.nan)
        except TypeError, ValueError:
            what = "a numeric column" if isinstance(x, _frame.Series) else "numeric columns"
            raise TypeError(f"{name}(): '{arg}' must have {what}") from None
        a = contiguous(a)  # a DataFrame's array is column-major
    else:
        try:
            a = sd.asarray(x)
        except (TypeError, ValueError) as e:
            msg = str(e)
            if isinstance(e, ValueError) and "0-dimensional" in msg:
                raise ValueError(f"{name}(): '{arg}' must have at least 1 dimension") from None
            if isinstance(e, ValueError):  # a ragged sequence, ...
                raise ValueError(f"{name}(): {msg.partition('(): ')[2] or msg}") from None
            raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {type_name(x)}") from None
    if a.dtype == sd.complex128:
        if not complex_ok:
            raise TypeError(f"{name}(): '{arg}' must be real, not complex128")
        return a
    if a.dtype == sd.string:
        raise TypeError(f"{name}(): '{arg}' must be numeric, not string")
    return a if a.dtype == sd.float64 else a.astype(sd.float64)


def points(x, name, arg="x"):
    """Evaluation points: (a C-contiguous 1-D float64 array, their shape); a scalar has shape ()."""
    sd = _sd
    if isinstance(x, numbers.Number) or _is_numpy_scalar(x):
        if isinstance(x, numbers.Complex) and not isinstance(x, numbers.Real):
            raise TypeError(f"{name}(): '{arg}' must be real, not {type_name(x)}")
        try:
            v = float(x)
        except TypeError:
            raise TypeError(f"{name}(): '{arg}' must be real, not {type_name(x)}") from None
        return sd.asarray([v]), ()
    a = array(x, name, arg)
    shape = tuple(a.shape)
    return contiguous(sd.reshape(a, (a.size,))), shape


def nodes(x, name, arg="x", *, decreasing_ok=False):
    """Nodes or breakpoints: a C-contiguous 1-D float64 copy, at least 2 points, finite, strictly monotone
    (increasing unless decreasing_ok); returns (array, ascending)."""
    a = array(x, name, arg)
    if a.ndim != 1:
        raise ValueError(f"{name}(): '{arg}' must be 1-D, not {a.ndim}-D")
    if a.shape[0] < 2:
        raise ValueError(f"{name}(): '{arg}' must have at least 2 points, not {a.shape[0]}")
    a = contiguous(a.copy())
    code = _core.check_x(a, decreasing_ok)
    if code == -1:
        raise ValueError(f"{name}(): '{arg}' must be finite")
    if code == -2:
        what = "strictly increasing or strictly decreasing" if decreasing_ok else "strictly increasing"
        raise ValueError(f"{name}(): '{arg}' must be {what}")
    a._set_readonly()
    return a, code == 0


def finite(a, name, arg):
    if not _core.all_finite(contiguous(a)):
        raise ValueError(f"{name}(): '{arg}' must be finite")


def extrapolate(v, name, *, none_ok=True):
    if v is None and none_ok:
        return None
    if v is True or v is False or (isinstance(v, str) and v == "periodic"):
        return v
    listed = ", ".join(EXTRAPOLATE[:-1]) + f" or {EXTRAPOLATE[-1]}" if none_ok else "True, False or 'periodic'"
    if isinstance(v, str):
        raise ValueError(f"{name}(): 'extrapolate' must be {listed}, not {v!r}")
    raise TypeError(f"{name}(): 'extrapolate' must be {listed}, not {type_name(v)}")


RULE = {False: 0, True: 1, "periodic": 2}


def integer(v, name, arg):
    if isinstance(v, bool) or type(v).__module__ == "numpy" and type(v).__name__ == "bool":
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}")
    try:
        k = operator.index(v)
    except TypeError:
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}") from None
    if not -(2**63) <= k < 2**63:
        raise OverflowError(f"{name}(): '{arg}' is out of range")
    return k


def order(v, name, arg="nu"):
    """A derivative order: an int, 0 or more."""
    k = integer(v, name, arg)
    if k < 0:
        raise ValueError(f"{name}(): '{arg}' must be >= 0, not {k}")
    return k


def real(v, name, arg):
    """A real scalar argument (int or float, numpy's too, not bool), as a float."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or type(v).__name__ == "bool":
        raise TypeError(f"{name}(): '{arg}' must be an int or a float, not {type_name(v)}")
    try:
        return float(v)
    except OverflowError:
        raise OverflowError(f"{name}(): '{arg}' is out of range") from None


def axis_index(axis, ndim, name):
    """axis in [-ndim, ndim) as an index in [0, ndim)."""
    k = integer(axis, name, "axis")
    if not -ndim <= k < ndim:
        raise ValueError(f"{name}(): axis {k} is out of bounds for an array of dimension {ndim}")
    return k % ndim


def choice(v, name, arg, choices):
    quoted = [f"'{c}'" for c in choices]
    listed = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" or {quoted[-1]}"
    if not isinstance(v, str):
        raise TypeError(f"{name}(): '{arg}' must be {listed}, not {type_name(v)}")
    if v not in choices:
        raise ValueError(f"{name}(): '{arg}' must be {listed}, not {v!r}")
    return v
