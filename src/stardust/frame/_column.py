"""Turning user data into Columns, and Columns into numpy arrays (docs/api/frame.md §2)."""

import sys

from . import _core
from ._na import NA

Column = _core.Column

_NO_VALUE = object()  # to_numpy's "na_value not given"


def is_ndarray(obj):
    np = sys.modules.get("numpy")  # an ndarray means numpy is already imported
    return np is not None and isinstance(obj, np.ndarray)


def is_scalar(obj):
    np = sys.modules.get("numpy")
    return obj is None or obj is NA or isinstance(obj, (bool, int, float, str)) or (
        np is not None and isinstance(obj, np.generic)
    )


def column_from(data, dtype=None):
    """A Column holding `data` (a list, tuple, range or 1-D numpy array), as `dtype` if given."""
    if is_ndarray(data):
        return _from_ndarray(data, dtype)
    if isinstance(data, (list, tuple, range)):
        return Column.from_list(data, NA, dtype)
    raise TypeError(f"cannot make a column from {type(data).__name__}; expected a list, tuple, range or numpy array")


def _from_ndarray(arr, dtype):
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D array, got {arr.ndim} dimensions")
    kind = arr.dtype.kind
    if kind in "UO":  # strings and Python objects: the list rules (docs/api/frame.md §2.4)
        return Column.from_list(arr.tolist(), NA, dtype)
    if kind not in "biuf":
        raise TypeError(f"unsupported numpy dtype {arr.dtype}")
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="))
    col = Column.from_numpy(arr)  # always a copy, widened to int64/float64
    return col.astype(dtype) if dtype is not None else col  # dtype= is explicit (docs/api/frame.md §2.3)


# Implicit conversions an assignment or fillna makes on its own: by dtype
# only, never by value, as in the array area (docs/api/frame.md §5.7).
_UPCAST = {("bool", "int64"), ("bool", "float64"), ("int64", "float64")}


def implicit(col, dtype):
    """`col` as `dtype` for an assignment or fillna: the same dtype, an all-NA
    column, or a conversion up bool -> int64 -> float64; TypeError otherwise
    (int64 into bool, float64 into int64 -- even 2.0 -- and strings)."""
    if col.dtype == dtype:
        return col
    if col.null_count == len(col):
        return full_na(len(col), dtype)
    if (col.dtype, dtype) in _UPCAST:
        return col.astype(dtype)
    if col.dtype != "string" and dtype != "string":
        raise TypeError(f"cannot implicitly convert {col.dtype} to {dtype} (that would lose information); "
                        "convert explicitly with astype")
    raise TypeError(f"cannot store {col.dtype} values in a column of dtype {dtype}")


def implicit_scalar(value, dtype):
    """A one-row Column of `dtype` holding the scalar `value` (implicit, as above)."""
    if value is None or value is NA:
        return Column.from_list([None], NA, dtype)
    if dtype == "float64" and isinstance(value, int) and not isinstance(value, bool):
        return Column.from_list([value], NA, "float64")  # an int outside int64 still becomes a float
    return implicit(Column.from_list([value], NA, None), dtype)


def exact(value, dtype):
    """A one-row Column of `dtype` equal to `value`, for lookups (`value in s`):
    ValueError, TypeError or OverflowError when no value of `dtype` equals it."""
    col = Column.from_list([value], NA, None)
    return col if col.dtype == dtype else col.convert(dtype)  # lossless only


def full_na(length, dtype=None):
    """A column of `length` NA rows (float64 unless `dtype` says otherwise, docs/api/frame.md §2.2)."""
    return Column.from_list([None] * length, NA, dtype)


def _check_na_value(na_value, dtype):
    """Raises unless `na_value` can stand in an array of `dtype` (docs/api/frame.md §2.7)."""
    ok = {
        "bool": isinstance(na_value, bool),
        "int64": isinstance(na_value, int) and not isinstance(na_value, bool) and -(2**63) <= na_value < 2**63,
        "float64": isinstance(na_value, (int, float)) and not isinstance(na_value, bool),
        "string": na_value is None or isinstance(na_value, str),
    }[dtype]
    if not ok:
        raise TypeError(f"na_value={na_value!r} does not fit an array of dtype {dtype}; pass dtype= as well")


def to_numpy(col, dtype, copy, na_value):
    """The numpy array for one column; see Series.to_numpy."""
    import numpy as np

    if dtype is not None and dtype != col.dtype:
        col = col.astype(dtype)  # dtype= is explicit (docs/api/frame.md §2.7)
    if na_value is not _NO_VALUE:
        _check_na_value(na_value, col.dtype)
    elif col.null_count > 0:
        raise ValueError(f"cannot convert a {col.dtype} column with NA to numpy; pass na_value= to fill the NA rows")

    if col.dtype == "string":
        arr = np.empty(len(col), dtype=object)  # filled element-wise so numpy cannot reinterpret the strs
        arr[:] = col.to_list(None if na_value is _NO_VALUE else na_value)
        return arr
    view = col.to_numpy_view()
    if col.null_count == 0:
        return view.copy() if copy else view
    arr = view.copy()
    arr[~col.validity_mask()] = na_value
    return arr


def head_bounds(length, n):
    """[start, stop) of head(n): the first n rows, or all but the last -n."""
    stop = min(n, length) if n >= 0 else max(length + n, 0)
    return 0, stop


def tail_bounds(length, n):
    """[start, stop) of tail(n): the last n rows, or all but the first -n."""
    start = max(length - n, 0) if n >= 0 else min(-n, length)
    return start, length
