"""Turning user data into Columns, and Columns into numpy arrays and
sd.ndarrays (docs/api/frame.md §2)."""

import sys

from .. import _core as _sd_core
from . import _core, _dtypes
from ._errors import named, type_name
from ._na import NA

Column = _core.Column

_NO_VALUE = object()  # to_numpy's "na_value not given"


def is_ndarray(obj):
    np = sys.modules.get("numpy")  # an ndarray means numpy is already imported
    return np is not None and isinstance(obj, np.ndarray)


def is_scalar(obj):
    np = sys.modules.get("numpy")
    # complex too: a scalar the frame cannot store, so its message says that (not "cannot make a column")
    return (
        obj is None
        or obj is NA
        or isinstance(obj, (bool, int, float, complex, str))
        or (np is not None and isinstance(obj, np.generic))
    )


def is_sdarray(obj):
    return isinstance(obj, _sd_core.ndarray)


def column_from(data, dtype=None, func=None):
    """A Column holding `data` (a list, tuple, range, 1-D numpy array or 1-D
    sd.ndarray), as `dtype` if given. An sd.ndarray is always copied
    (docs/api/frame.md §2.8). `func` names the public function in messages."""
    if is_sdarray(data):
        col = Column.from_array(data, func=func)
        return col.astype(dtype, func=func) if dtype is not None else col  # dtype= is explicit
    if is_ndarray(data):
        return _from_ndarray(data, dtype, func)
    if isinstance(data, (list, tuple, range)):
        return Column.from_list(data, NA, dtype, func=func)
    raise TypeError(
        named(
            func,
            f"cannot make a column from {type_name(data)}; expected a list, tuple, range, numpy array or sd.ndarray",
        )
    )


def _from_ndarray(arr, dtype, func):
    if arr.ndim != 1:
        raise ValueError(named(func, f"expected a 1-D array, got {arr.ndim} dimensions"))
    kind = arr.dtype.kind
    if kind in "UO":  # strings and Python objects: the list rules (docs/api/frame.md §2.4)
        return Column.from_list(arr.tolist(), NA, dtype, func=func)
    if kind not in "biuf":
        raise TypeError(named(func, f"unsupported numpy dtype {arr.dtype}"))
    if kind == "u" and arr.dtype.itemsize == 8:  # by dtype, not by value, as sd.asarray (DESIGN.md §10.1 #10)
        raise TypeError(named(func, "uint64 buffers cannot be converted to int64 without loss"))
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="))
    col = Column.from_numpy(arr, func=func)  # always a copy, widened to int64/float64
    return col.astype(dtype, func=func) if dtype is not None else col  # dtype= is explicit (docs/api/frame.md §2.3)


# Implicit conversions an assignment or fillna makes on its own: by dtype
# only, never by value, as in the array area (docs/api/frame.md §5.7).
_UPCAST = {("bool", "int64"), ("bool", "float64"), ("int64", "float64")}


def implicit(col, dtype, func=None):
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
        raise TypeError(
            named(
                func,
                f"cannot implicitly convert {col.dtype} to {dtype} (that would lose information); "
                "convert explicitly with astype",
            )
        )
    raise TypeError(named(func, f"cannot store {col.dtype} values in a column of dtype {dtype}"))


def implicit_scalar(value, dtype, func=None):
    """A one-row Column of `dtype` holding the scalar `value` (implicit, as above)."""
    if value is None or value is NA:
        return Column.from_list([None], NA, dtype)
    if dtype == "float64" and isinstance(value, int) and not isinstance(value, bool):
        return Column.from_list([value], NA, "float64", func=func)  # an int outside int64 still becomes a float
    return implicit(Column.from_list([value], NA, None, func=func), dtype, func)


_NUMBERS = ("bool", "int64", "float64")


def exact(value, dtype):
    """A one-row Column of `dtype` equal to `value`, for lookups (`value in s`):
    ValueError, TypeError or OverflowError when no value of `dtype` equals it.
    bool is a number here as everywhere (bool < int64 < float64): True is 1
    and 1.0, and 0 or 1 find False or True in a bool column."""
    col = Column.from_list([value], NA, None)
    if col.dtype == dtype:
        return col
    if col.dtype in _NUMBERS and dtype in _NUMBERS and "bool" in (col.dtype, dtype):
        if col.dtype == "bool":
            return col.astype(dtype)  # 0 and 1: exact in int64 and float64
        if col.null_count == 0 and col.to_list(None)[0] not in (0, 1):  # NaN too
            raise ValueError(f"{value!r} is not a bool value")
        return col.astype("bool")
    return col.convert(dtype)  # lossless only


def full_na(length, dtype=None):
    """A column of `length` NA rows (float64 unless `dtype` says otherwise, docs/api/frame.md §2.2)."""
    return Column.from_list([None] * length, NA, dtype)


def article(word):
    """'a bool', 'an int64': the article a dtype name takes in a message."""
    return f"an {word}" if word[:1] in "aeiou" else f"a {word}"


def _plain(value):
    """A numpy scalar as the Python scalar of its kind (numpy.bool_ -> bool,
    numpy integers -> int, numpy floats -> float), as fillna reads them;
    anything else unchanged."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.generic):
        for kind, plain in ((np.bool_, bool), (np.integer, int), (np.floating, float)):
            if isinstance(value, kind):
                return plain(value)
    return value


def _check_na_value(na_value, dtype, func):
    """Raises unless `na_value` can stand in an array of `dtype` (docs/api/frame.md §2.7).
    A numpy scalar counts as the Python scalar of its kind. An int outside
    int64 for an int64 array is an OverflowError, as when any int must become
    an int64 (API.md §2)."""
    shown, na_value = na_value, _plain(na_value)
    if (
        dtype == "int64"
        and isinstance(na_value, int)
        and not isinstance(na_value, bool)
        and not -(2**63) <= na_value < 2**63
    ):
        raise OverflowError(named(func, "'na_value' is out of range for int64"))
    ok = {
        "bool": isinstance(na_value, bool),
        "int64": isinstance(na_value, int) and not isinstance(na_value, bool) and -(2**63) <= na_value < 2**63,
        "float64": isinstance(na_value, (int, float)) and not isinstance(na_value, bool),
        "string": na_value is None or isinstance(na_value, str),
    }[dtype]
    if not ok:
        raise TypeError(named(func, f"na_value={shown!r} does not fit an array of dtype {dtype}; pass dtype= as well"))


def common_dtype(cols, func, what):
    """The dtype a 2-D array of `cols` takes (DataFrame.to_numpy, to_array): the
    columns' dtypes promoted, bool < int64 < float64, as in the array area;
    TypeError when a string column meets another dtype."""
    dtypes = {c.dtype for c in cols}
    if not dtypes:
        return None
    dtype = _dtypes.common(dtypes)
    if dtype is None:
        raise TypeError(named(func, f"columns of dtypes {sorted(dtypes)} have no common {what} dtype; pass dtype="))
    return dtype


def check_copy(copy, func):
    """copy= as in the array area's asarray: True, False or None."""
    if copy is not None and not isinstance(copy, bool):
        raise TypeError(named(func, f"'copy' must be True, False or None, not {type_name(copy)}"))
    return copy


def to_numpy(col, dtype, copy, na_value, func):
    """The numpy array for one column; see Series.to_numpy. copy=None copies
    only when it must, True always, False never (ValueError instead)."""
    import numpy as np

    check_copy(copy, func)
    fresh = False  # whether `col` was just made here (nobody else holds its Buffer)
    if dtype is not None and dtype != col.dtype:
        if copy is False:
            raise ValueError(named(func, "copy=False, but a dtype conversion requires a copy"))
        col, fresh = col.astype(dtype, func=func), True  # dtype= is explicit (docs/api/frame.md §2.7)
    if na_value is not _NO_VALUE:
        _check_na_value(na_value, col.dtype, func)
    elif col.null_count > 0:
        raise ValueError(
            named(
                func, f"cannot convert {article(col.dtype)} column with NA to numpy; pass na_value= to fill the NA rows"
            )
        )

    if col.dtype == "string":
        if copy is False:
            raise ValueError(named(func, "copy=False, but a string column can only be converted by copying"))
        arr = np.empty(len(col), dtype=object)  # filled element-wise so numpy cannot reinterpret the strs
        arr[:] = col.to_list(None if na_value is _NO_VALUE else na_value)
        return arr
    if fresh and col.null_count == 0:
        # The converted column is the new array: writable, as to_array's (docs/api/frame.md §2.8).
        return np.asarray(col.to_array_view(True))
    view = col.to_numpy_view()
    if col.null_count == 0:
        return view.copy() if copy else view
    if copy is False:
        raise ValueError(named(func, "copy=False, but filling the NA rows requires a copy"))
    arr = view.copy()
    arr[~col.validity_mask()] = na_value
    return arr


def array_column(col, dtype, copy, na_value, func):
    """(column, fresh): `col` ready to become an sd.ndarray -- of `dtype`,
    without NA -- and whether it was just made (nobody else holds its Buffer).
    The checks of to_numpy (docs/api/frame.md §2.8)."""
    fresh = False
    if dtype == "string":  # the dtype argument, before any conversion (the column need not be string)
        raise TypeError(
            named(
                func,
                "'dtype' must be sd.bool, sd.int64 or sd.float64 (or their names), not sd.string "
                "(the array area has no string dtype)",
            )
        )
    if dtype is not None and dtype != col.dtype:
        if copy is False:
            raise ValueError(named(func, "copy=False, but a dtype conversion requires a copy"))
        col, fresh = col.astype(dtype, func=func), True  # dtype= is explicit
    if col.dtype == "string":
        raise TypeError(named(func, "a string column has no array (the array area has no string dtype)"))
    if na_value is not _NO_VALUE:
        _check_na_value(na_value, col.dtype, func)
    elif col.null_count > 0:
        raise ValueError(
            named(
                func,
                f"cannot convert {article(col.dtype)} column with NA to an array; pass na_value= to fill the NA rows",
            )
        )
    if col.null_count > 0:
        if copy is False:
            raise ValueError(named(func, "copy=False, but filling the NA rows requires a copy"))
        col, fresh = col.fill_na(Column.from_list([na_value], NA, col.dtype)), True
    return col, fresh


def to_array(col, dtype, copy, na_value, func):
    """The sd.ndarray for one column; see Series.to_array."""
    check_copy(copy, func)
    col, fresh = array_column(col, dtype, copy, na_value, func)
    if copy and not fresh:
        return _sd_core.asarray(col.to_array_view(False), copy=True)
    return col.to_array_view(fresh)  # shares the column's Buffer; read-only unless just made


def head_bounds(length, n):
    """[start, stop) of head(n): the first n rows, or all but the last -n."""
    stop = min(n, length) if n >= 0 else max(length + n, 0)
    return 0, stop


def tail_bounds(length, n):
    """[start, stop) of tail(n): the last n rows, or all but the first -n."""
    start = max(length - n, 0) if n >= 0 else min(-n, length)
    return start, length
