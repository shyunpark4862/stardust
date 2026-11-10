"""DataFrame: named Columns of one length plus row labels (ARCHITECTURE.md §3.1)."""

from . import _column, _core, _format, _ops
from ._column import _NO_VALUE
from ._dtypes import resolve
from .index import RangeIndex
from .series import Series, fill_column, mask_positions


def _check_columns(columns):
    columns = list(columns)
    for c in columns:
        if not isinstance(c, str):
            raise TypeError(f"column names must be str, got {type(c).__name__}")  # API.md §6
    if len(set(columns)) != len(columns):
        raise ValueError("column names must be unique")
    return columns


class DataFrame:
    __slots__ = ("_cols", "_index")

    def __init__(self, data=None, *, columns=None, dtype=None):
        dtype = resolve(dtype) if dtype is not None else None
        if columns is not None:
            columns = _check_columns(columns)
        if data is None:
            cols, index = {}, RangeIndex(0)
            if columns is not None:
                cols = dict.fromkeys(columns)
        elif isinstance(data, DataFrame):
            cols, index = dict(data._cols), data._index  # shares the Buffers
            if dtype is not None:
                cols = {k: c.convert(dtype) for k, c in cols.items()}
            if columns is not None:
                cols = {k: cols.get(k) for k in columns}
        elif isinstance(data, dict):
            cols, index = _from_dict(data, dtype)
            if columns is not None:
                cols = {k: cols.get(k) for k in columns}
        elif _column.is_ndarray(data):
            if data.ndim != 2:
                raise ValueError(f"a DataFrame from a numpy array needs 2 dimensions, got {data.ndim}")
            names = _require_columns(columns, data.shape[1])
            cols = {k: _column.column_from(data[:, j], dtype) for j, k in enumerate(names)}
            index = RangeIndex(data.shape[0])
        elif isinstance(data, (list, tuple)):
            cols, index = _from_rows(data, columns, dtype)
        else:
            raise TypeError(f"cannot make a DataFrame from {type(data).__name__}")
        # Requested columns that the data lacks are all NA (as in pandas).
        self._cols = {k: c if c is not None else _column.full_na(len(index), dtype) for k, c in cols.items()}
        self._index = index

    @classmethod
    def _new(cls, cols, index):
        obj = cls.__new__(cls)
        obj._cols = cols
        obj._index = index
        return obj

    @property
    def columns(self):
        # Provisional: a list until M4 brings Index.
        return list(self._cols)

    @property
    def dtypes(self):
        # Provisional: a dict until M4 brings a string Index to label a Series with.
        return {k: c.dtype for k, c in self._cols.items()}

    @property
    def index(self):
        return self._index

    @property
    def shape(self):
        return (len(self._index), len(self._cols))

    def __len__(self):
        return len(self._index)

    def __bool__(self):
        raise ValueError("the truth value of a DataFrame is ambiguous")

    def __getitem__(self, key):
        if isinstance(key, Series):  # a bool mask selects rows (API.md §7)
            return self._take(mask_positions(key, self._index))
        if not isinstance(key, str):
            raise TypeError(f"column names are str, got {type(key).__name__}")
        try:
            col = self._cols[key]
        except KeyError:
            raise KeyError(key) from None
        return Series._new(col, self._index, key)

    def _take(self, positions, ignore_index=False):
        index = RangeIndex(len(positions)) if ignore_index else self._index._take(positions)
        return DataFrame._new({k: c.take(positions) for k, c in self._cols.items()}, index)

    def _map_columns(self, f, per_column=None):
        """A DataFrame with f(column) (or f(column, per_column[name])) for each column."""
        if per_column is None:
            return DataFrame._new({k: f(c) for k, c in self._cols.items()}, self._index)
        unknown = [k for k in per_column if k not in self._cols]
        if unknown:
            raise KeyError(f"no such columns: {unknown}")
        return DataFrame._new({k: f(c, per_column[k]) if k in per_column else c for k, c in self._cols.items()}, self._index)

    # --- element-wise (API.md §12) ---

    __hash__ = None
    __array_ufunc__ = None

    def _binary(self, other, op, reflected=False):
        if isinstance(other, DataFrame):
            if set(other._cols) != set(self._cols):
                raise NotImplementedError("DataFrames with different columns need alignment, which is not supported yet")
            _ops.check_same_index(self._index, other._index)
            pairs = {k: (c, other._cols[k], 0) for k, c in self._cols.items()}
        elif isinstance(other, Series):
            raise TypeError("operations between a DataFrame and a Series are not supported (API.md §12)")
        elif _ops.is_operand_scalar(other):
            pairs = {k: (c, _ops.scalar_column(other, c.dtype), 2) for k, c in self._cols.items()}
        else:
            return NotImplemented
        cols = {}
        for k, (a, b, side) in pairs.items():
            if reflected:
                a, b, side = b, a, 1 if side == 2 else side
            cols[k] = _ops.apply(op, a, b, side)
        return DataFrame._new(cols, self._index)

    def __neg__(self):
        return self._map_columns(lambda c: _core.unary_op("neg", c))

    def __abs__(self):
        return self._map_columns(lambda c: _core.unary_op("abs", c))

    def __pos__(self):
        bad = [k for k, c in self._cols.items() if c.dtype not in ("int64", "float64")]
        if bad:
            raise TypeError(f"unsupported operand dtype for unary + in columns {bad}")
        return self

    def __invert__(self):
        return self._map_columns(lambda c: _core.unary_op("invert", c))

    # --- NA and dtypes ---

    def astype(self, dtype):
        """Every column, or the columns of a {name: dtype} dict, as that dtype (API.md §12)."""
        if isinstance(dtype, dict):
            return self._map_columns(lambda c, dt: c.astype(resolve(dt)), dtype)
        dtype = resolve(dtype)
        return self._map_columns(lambda c: c.astype(dtype))

    def isna(self):
        return self._map_columns(lambda c: c.isna())

    def notna(self):
        return self._map_columns(lambda c: c.notna())

    def fillna(self, value):
        """NA rows set to `value`, or per column from a {name: value} dict (API.md §12)."""
        if isinstance(value, dict):
            return self._map_columns(fill_column, value)
        return self._map_columns(lambda c: fill_column(c, value))

    def dropna(self, *, how="any", subset=None, ignore_index=False):
        """The rows without NA in `subset` (default: all columns); with how="all",
        only rows where every `subset` value is NA are dropped."""
        if how not in ("any", "all"):
            raise ValueError(f"how must be 'any' or 'all', got {how!r}")
        names = list(self._cols) if subset is None else _subset(subset)
        for k in names:
            if k not in self._cols:
                raise KeyError(k)
        keep = None
        combine = "and" if how == "any" else "or"
        for k in names:
            valid = self._cols[k].notna()
            keep = valid if keep is None else _core.binary_op(combine, keep, valid, 0)
        if keep is None:  # no columns to look at: every row stays
            keep = _core.Column.from_list([True] * len(self), None, None)
        return self._take(keep.nonzero(), ignore_index)

    def to_numpy(self, *, dtype=None, copy=False, na_value=_NO_VALUE):
        """A 2-D numpy array (API.md §10): all columns must share a dtype,
        except that int64 and float64 together give float64. NA rows follow
        the Series.to_numpy rules."""
        import numpy as np

        dtype = resolve(dtype) if dtype is not None else None
        if dtype is None:
            kinds = set(self.dtypes.values())
            if len(kinds) == 1:
                (dtype,) = kinds
            elif kinds == {"int64", "float64"}:
                dtype = "float64"
            elif kinds:
                raise TypeError(f"columns of dtypes {sorted(kinds)} have no common numpy dtype; pass dtype=")
        arrays = [_column.to_numpy(c, dtype, True, na_value) for c in self._cols.values()]
        if not arrays:
            return np.empty((len(self), 0))
        return np.stack(arrays, axis=1)

    def _slice(self, start, stop):
        cols = {k: c.slice(start, stop - start) for k, c in self._cols.items()}
        return DataFrame._new(cols, self._index._slice(start, stop))

    def head(self, n=5):
        """The first n rows; with n < 0, all rows but the last -n."""
        return self._slice(*_column.head_bounds(len(self), n))

    def tail(self, n=5):
        """The last n rows; with n < 0, all rows but the first -n."""
        return self._slice(*_column.tail_bounds(len(self), n))

    def __repr__(self):
        return _format.frame_repr(self)


def _subset(subset):
    return [subset] if isinstance(subset, str) else list(subset)


def _make_binary(op, reflected):
    def method(self, other):
        return self._binary(other, op, reflected)

    return method


for _name, _op in {**_ops.ARITH, **_ops.LOGIC}.items():
    setattr(DataFrame, f"__{_name}__", _make_binary(_op, False))
    setattr(DataFrame, f"__r{_name}__", _make_binary(_op, True))
for _name in _ops.COMPARE:
    setattr(DataFrame, f"__{_name}__", _make_binary(_name, False))


def _require_columns(columns, ncols):
    if columns is None:
        raise ValueError("columns= is required to name the columns of row or 2-D array data")  # API.md §10
    if len(columns) != ncols:
        raise ValueError(f"{len(columns)} column names for {ncols} columns")
    return columns


def _from_dict(data, dtype):
    names = _check_columns(data.keys())
    length, index = None, None
    cols = {}
    for k in names:
        v = data[k]
        if _column.is_scalar(v):
            cols[k] = v  # broadcast below, once the length is known
            continue
        if isinstance(v, Series):
            if index is None:
                index = v._index
            elif not index.equals(v._index):
                raise NotImplementedError("Series with different indexes need alignment, which is not supported yet")
            col = v._col.convert(dtype) if dtype is not None else v._col
        else:
            col = _column.column_from(v, dtype)
        if length is None:
            length = len(col)
        elif len(col) != length:
            raise ValueError("all columns must have the same length")
        cols[k] = col
    if length is None:
        if cols:
            raise ValueError("cannot make a DataFrame from scalars alone; give at least one column as a list")
        length = 0
    for k, v in cols.items():
        if _column.is_scalar(v):
            cols[k] = _column.column_from([v] * length, dtype)
    return cols, index if index is not None else RangeIndex(length)


def _from_rows(rows, columns, dtype):
    if len(rows) == 0:
        return dict.fromkeys(columns or []), RangeIndex(0)
    if all(isinstance(r, dict) for r in rows):  # records: missing keys are NA
        names = {}
        for r in rows:
            names.update(dict.fromkeys(r))
        names = _check_columns(names)
        cols = {k: _column.column_from([r.get(k) for r in rows], dtype) for k in names}
        if columns is not None:
            cols = {k: cols.get(k) for k in columns}
        return cols, RangeIndex(len(rows))
    if all(isinstance(r, (list, tuple)) or _column.is_ndarray(r) for r in rows):
        names = _require_columns(columns, len(rows[0]))
        for r in rows:
            if len(r) != len(names):
                raise ValueError(f"every row must have {len(names)} values, one per column")
        values = [list(c) for c in zip(*rows)]
        return {k: _column.column_from(v, dtype) for k, v in zip(names, values)}, RangeIndex(len(rows))
    raise TypeError("a DataFrame from a list needs rows: all dicts, or all lists/tuples")
