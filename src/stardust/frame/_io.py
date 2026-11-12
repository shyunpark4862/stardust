"""read_csv (docs/api/frame.md §10)."""

import os

from . import _core
from ._dtypes import resolve
from ._errors import check_bool, check_int, type_name
from ._index import Index, RangeIndex


def _str_list(value, what):  # for read_csv()
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):  # no generators, as the array area's list arguments
        raise TypeError(f"read_csv(): '{what}' must be a list of str, not {type_name(value)}")
    for v in value:
        if not isinstance(v, str):
            raise TypeError(f"read_csv(): '{what}' must be a list of str, not a list holding {type_name(v)}")
    return list(value)


def read_csv(path, /, *, sep=",", header=True, names=None, usecols=None, dtype=None, na_values=None, nrows=None,
             index_col=None):
    """A DataFrame from a UTF-8 CSV file (RFC 4180 quoting).

    Types are inferred per column (int64, then float64, then bool, then
    string) unless dtype= gives them; only unquoted empty fields and the
    na_values texts are NA (docs/api/frame.md §10.3)."""
    from ._frame import DataFrame

    if not isinstance(sep, str):
        raise TypeError(f"read_csv(): 'sep' must be a str, not {type_name(sep)}")
    if len(sep) != 1 or sep in '"\r\n':
        raise ValueError(f"read_csv(): 'sep' must be one character other than a quote or a line break, not {sep!r}")
    check_bool(header, "header", "read_csv")
    if index_col is not None and not isinstance(index_col, str):
        raise TypeError(f"read_csv(): 'index_col' must be a column name (str) or None, not {type_name(index_col)}")
    names, usecols, na_values = _str_list(names, "names"), _str_list(usecols, "usecols"), _str_list(na_values, "na_values")
    if not header and not names:
        raise ValueError("read_csv(): 'names' is required when header=False")
    if usecols and len(set(usecols)) != len(usecols):
        raise ValueError("read_csv(): 'usecols' must not repeat a name")
    dtypes, all_dtype = {}, None
    if isinstance(dtype, dict):
        dtypes = {k: resolve(v, "read_csv") for k, v in dtype.items()}
    elif dtype is not None:
        all_dtype = resolve(dtype, "read_csv")
    if nrows is not None and check_int(nrows, "nrows", "read_csv") < 0:
        raise ValueError(f"read_csv(): 'nrows' must be >= 0, not {nrows}")

    with open(os.fspath(path), "rb") as f:
        data = f.read()
    out_names, cols = _core.read_csv(data, sep, header, names, usecols, dtypes, all_dtype, na_values,
                                     -1 if nrows is None else nrows)
    unknown = [k for k in dtypes if k not in out_names]
    if unknown:
        raise KeyError(f"read_csv(): 'dtype' names columns that are not read: {unknown}")
    columns = dict(zip(out_names, cols))
    n = len(cols[0]) if cols else 0
    index = RangeIndex(n)
    if index_col is not None:
        if index_col not in columns:
            raise KeyError(f"read_csv(): 'index_col' names a column that is not read: {index_col!r}")
        labels = columns.pop(index_col)
        if labels.dtype not in ("int64", "string") or labels.null_count:
            raise ValueError(f"read_csv(): the 'index_col' column {index_col!r} must be int64 or string without NA, "
                             f"got {labels.dtype}")
        index = Index._new(labels, index_col)
    return DataFrame._new(columns, index)
