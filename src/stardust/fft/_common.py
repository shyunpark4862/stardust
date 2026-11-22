"""The argument checks and messages sd.fft's helpers share (docs/api/fft.md §1; decisions FT5, FT6, FT13,
FT18, FT26 in docs/design/fft.md §1.1). The transforms check theirs natively (csrc/fft/basic.cpp), with
the same messages.

- Data (FT5): an sd.ndarray only; lists and numpy arrays are TypeErrors.
- int arguments (API.md §2): operator.index's rule, bool is not an int, outside int64 is OverflowError.
- axes (FT13): None, an int, or a tuple or list of ints; out of range or repeated is ValueError (the
  messages of the array's axes, bind/pyutil.cpp parse_axes).
"""

import math
import operator

from .. import _core as _sd

_core = _sd.fft
_INT64_MAX = 2**63 - 1


def type_name(v):
    return _sd._type_name(v)  # numpy's types keep their module (API.md §2.1)


def array(x, name, arg="x"):
    if not isinstance(x, _sd.ndarray):
        raise TypeError(f"{name}(): '{arg}' must be an sd.ndarray, not {type_name(x)}")
    return x


def integer(v, name, arg):
    """v as an int by operator.index (bool refused), within int64."""
    if v is True or v is False or isinstance(v, _sd.ndarray):
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}")
    try:
        i = operator.index(v)
    except TypeError:
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}") from None
    if type(v).__module__ == "numpy" and type(v).__name__ == "bool":
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}")
    if not -_INT64_MAX - 1 <= i <= _INT64_MAX:
        raise OverflowError(f"{name}(): '{arg}' is out of range")
    return i


def real(v, name, arg):
    """v as a float: a Python int or float, or a numpy integer or floating scalar."""
    if v is True or v is False or isinstance(v, _sd.ndarray):
        raise TypeError(f"{name}(): '{arg}' must be an int or a float, not {type_name(v)}")
    if isinstance(v, float):
        return float(v)
    if isinstance(v, int):
        try:
            return float(v)
        except OverflowError:
            raise OverflowError(f"{name}(): '{arg}' is out of range") from None
    mod = type(v).__module__
    kind = getattr(getattr(v, "dtype", None), "kind", "")
    if mod == "numpy" and kind in "iuf" and kind:
        return float(v)
    raise TypeError(f"{name}(): '{arg}' must be an int or a float, not {type_name(v)}")


def axes(v, ndim, name):
    """The axes argument (None: every axis) as a tuple of axes in [0, ndim), checked."""
    if v is None:
        return tuple(range(ndim))
    if isinstance(v, (tuple, list)):
        items = list(v)
    else:
        items = [v]
    out = []
    for a in items:
        if a is True or a is False or isinstance(a, _sd.ndarray):
            raise TypeError(f"{name}(): 'axes' must be an int or a tuple or list of ints, not {type_name(a)}")
        try:
            i = operator.index(a)
        except TypeError:
            raise TypeError(f"{name}(): 'axes' must be an int or a tuple or list of ints, not {type_name(a)}") from None
        if not -_INT64_MAX - 1 <= i <= _INT64_MAX:
            raise OverflowError(f"{name}(): 'axes' is out of range")
        if not -ndim <= i < ndim:
            raise ValueError(f"{name}(): axis {i} is out of bounds for an array of dimension {ndim}")
        i %= ndim
        if i in out:
            raise ValueError(f"{name}(): repeated axis")
        out.append(i)
    return tuple(out)


def finite_nonzero(d, name, arg):
    if d == 0 or not math.isfinite(d):
        raise ValueError(f"{name}(): '{arg}' must be finite and nonzero, not {d!r}")
    return d
