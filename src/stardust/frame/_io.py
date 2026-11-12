"""read_csv (docs/api/frame.md §10)."""

import os

from . import _core
from ._dtypes import resolve
from ._index import Index, RangeIndex


def _str_list(value, what):
    if value is None:
        return []
    if isinstance(value, str):
        raise TypeError(f"{what} takes a list of str")
    values = list(value)
    for v in values:
        if not isinstance(v, str):
            raise TypeError(f"{what} takes str, got {type(v).__name__}")
    return values


def read_csv(path, /, *, sep=",", header=True, names=None, usecols=None, dtype=None, na_values=None, nrows=None,
             index_col=None):
    """A DataFrame from a UTF-8 CSV file (RFC 4180 quoting).

    Types are inferred per column (int64, then float64, then bool, then
    string) unless dtype= gives them; only unquoted empty fields and the
    na_values texts are NA (docs/api/frame.md §10.3)."""
    from ._frame import DataFrame

    if not isinstance(sep, str) or len(sep) != 1 or sep in '"\r\n':
        raise ValueError(f"sep must be one character other than a quote or a line break, got {sep!r}")
    if not isinstance(header, bool):
        raise TypeError("header must be True or False")
    names, usecols, na_values = _str_list(names, "names"), _str_list(usecols, "usecols"), _str_list(na_values, "na_values")
    if not header and not names:
        raise ValueError("header=False needs names= (API.md §19)")
    if usecols and len(set(usecols)) != len(usecols):
        raise ValueError("usecols must not repeat a name")
    dtypes, all_dtype = {}, None
    if isinstance(dtype, dict):
        dtypes = {k: resolve(v) for k, v in dtype.items()}
    elif dtype is not None:
        all_dtype = resolve(dtype)
    if nrows is not None and (not isinstance(nrows, int) or isinstance(nrows, bool) or nrows < 0):
        raise ValueError(f"nrows must be an int >= 0, got {nrows!r}")

    with open(os.fspath(path), "rb") as f:
        data = f.read()
    out_names, cols = _core.read_csv(data, sep, header, names, usecols, dtypes, all_dtype, na_values,
                                     -1 if nrows is None else nrows)
    unknown = [k for k in dtypes if k not in out_names]
    if unknown:
        raise KeyError(f"dtype names columns that are not read: {unknown}")
    columns = dict(zip(out_names, cols))
    n = len(cols[0]) if cols else 0
    index = RangeIndex(n)
    if index_col is not None:
        if index_col not in columns:
            raise KeyError(f"index_col {index_col!r} is not a column that is read")
        labels = columns.pop(index_col)
        if labels.dtype not in ("int64", "string") or labels.null_count:
            raise ValueError(f"index_col {index_col!r} must be int64 or string without NA (API.md §6), got {labels.dtype}")
        index = Index._new(labels, index_col)
    return DataFrame._new(columns, index)
