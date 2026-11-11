"""DataFrame: named Columns of one length plus row labels (ARCHITECTURE.md §3.1)."""

from . import _column, _core, _format, _ops, _select, _unique
from ._column import _NO_VALUE
from ._core import Column
from ._dtypes import resolve
from ._reduce import Reductions
from ._na import NA
from .index import Index, RangeIndex, ensure_index
from .series import Series, assignment_values, fill_column


def _check_columns(columns):
    columns = list(columns)
    for c in columns:
        if not isinstance(c, str):
            raise TypeError(f"column names must be str, got {type(c).__name__}")  # API.md §6
    if len(set(columns)) != len(columns):
        raise ValueError("column names must be unique")
    return columns


class DataFrame(Reductions):
    __slots__ = ("_cols", "_index")

    def __init__(self, data=None, *, index=None, columns=None, dtype=None):
        dtype = resolve(dtype) if dtype is not None else None
        target = ensure_index(index) if index is not None else None
        if columns is not None:
            columns = _check_columns(columns)
        if data is None:
            cols, natural = {}, target if target is not None else RangeIndex(0)
            if columns is not None:
                cols = dict.fromkeys(columns)
        elif isinstance(data, DataFrame):
            cols, natural = dict(data._cols), data._index  # shares the Buffers
            if dtype is not None:
                cols = {k: c.convert(dtype) for k, c in cols.items()}
            if columns is not None:
                cols = {k: cols.get(k) for k in columns}
            if target is not None:  # labels given: rearrange to them (API.md §14)
                cols = {k: c if c is None else _select.reindexed(c, natural, target) for k, c in cols.items()}
                natural = target
        elif isinstance(data, dict):
            cols, natural = _from_dict(data, dtype, target)
            if columns is not None:
                cols = {k: cols.get(k) for k in columns}
        elif _column.is_ndarray(data):
            if data.ndim != 2:
                raise ValueError(f"a DataFrame from a numpy array needs 2 dimensions, got {data.ndim}")
            names = _require_columns(columns, data.shape[1])
            cols = {k: _column.column_from(data[:, j], dtype) for j, k in enumerate(names)}
            natural = RangeIndex(data.shape[0])
        elif isinstance(data, (list, tuple)):
            cols, natural = _from_rows(data, columns, dtype)
        else:
            raise TypeError(f"cannot make a DataFrame from {type(data).__name__}")
        if target is not None and natural is not target:
            if len(natural) != len(target) and cols:
                raise ValueError(f"{len(natural)} rows for an index of {len(target)} labels")
            natural = target
        # Requested columns that the data lacks are all NA (as in pandas).
        self._cols = {k: c if c is not None else _column.full_na(len(natural), dtype) for k, c in cols.items()}
        self._index = natural

    @classmethod
    def _new(cls, cols, index):
        obj = cls.__new__(cls)
        obj._cols = cols
        obj._index = index
        return obj

    @property
    def columns(self):
        """The column names as a string Index (API.md §14)."""
        return Index._new(Column.from_list(list(self._cols), NA, "string"))

    @property
    def dtypes(self):
        """The dtype of each column, as a string Series labelled by column name."""
        return Series._new(Column.from_list([c.dtype for c in self._cols.values()], NA, "string"), self.columns, None)

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

    def __iter__(self):
        """The column names (API.md §14)."""
        return iter(list(self._cols))

    def __contains__(self, name):
        return name in self._cols

    def _column(self, name):
        if not isinstance(name, str):
            raise TypeError(f"column names are str, got {type(name).__name__}")
        try:
            return self._cols[name]
        except KeyError:
            raise KeyError(name) from None

    def __getitem__(self, key):
        """df["a"] (a Series), df[["a", "b"]] (a DataFrame) or df[mask] (rows) (API.md §7)."""
        if isinstance(key, Series):
            return self._take(_select.mask_positions(key, self._index))
        if isinstance(key, list):
            names = _check_columns(key)
            return DataFrame._new({k: self._column(k) for k in names}, self._index)
        return Series._new(self._column(key), self._index, key)

    def __setitem__(self, name, value):
        """df["c"] = value: a new column at the end, or a whole column replaced (API.md §14)."""
        if not isinstance(name, str):
            raise TypeError(f"column names are str, got {type(name).__name__}")
        n = len(self._index)
        if isinstance(value, DataFrame):
            raise TypeError("assign a Series, a list or a scalar to a column")
        if isinstance(value, Series):
            col = _select.reindexed(value._col, value._index, self._index)  # matched by label
        elif _column.is_scalar(value):
            col = _column.column_from([value] * n)
        else:
            col = _column.column_from(value)
            if not self._cols and n == 0:  # the first column of an empty frame sets the length
                self._index = RangeIndex(len(col))
            elif len(col) != n:
                raise ValueError(f"{len(col)} values for {n} rows")
        self._cols[name] = col

    @property
    def loc(self):
        """Rows (and columns) by label: df.loc[rows] or df.loc[rows, columns] (API.md §14)."""
        return _FrameIndexer(self, by_label=True)

    @property
    def iloc(self):
        """Rows (and columns) by position: df.iloc[rows] or df.iloc[rows, columns]."""
        return _FrameIndexer(self, by_label=False)

    def reindex(self, labels):
        """The rows at `labels`, in that order; absent labels are NA (API.md §14)."""
        target = ensure_index(labels)
        cols = {k: _select.reindexed(c, self._index, target) for k, c in self._cols.items()}
        return DataFrame._new(cols, target)

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

    def _reduce(self, op, skipna, ddof):
        """One row, labelled by the operation, each column keeping its own result
        dtype (API.md §15). A column the operation does not apply to raises."""
        cols = {k: _core.reduce(op, c, skipna, ddof) for k, c in self._cols.items()}
        return DataFrame._new(cols, Index._new(Column.from_list([op], NA, "string")))

    def groupby(self, by, *, sort=True, dropna=True):
        """Group the rows by the values of the key column(s) `by` (API.md §16)."""
        from .groupby import GroupBy

        return GroupBy(self, by, sort=sort, dropna=dropna)

    # --- duplicates and unique values (API.md §13) ---

    def _key_columns(self, subset):
        names = list(self._cols) if subset is None else _subset(subset)
        if not names:
            raise ValueError("subset must name at least one column")
        for k in names:
            if k not in self._cols:
                raise KeyError(k)
        return names

    def nunique(self, *, dropna=True):
        """How many distinct values each column has, as an int64 Series labelled by column name."""
        counts = [self[k].nunique(dropna=dropna) for k in self._cols]
        return Series._new(Column.from_list(counts, NA, "int64"), self.columns, None)

    def duplicated(self, subset=None, *, keep="first"):
        """True for rows whose values in `subset` (default: all columns) repeat (API.md §13)."""
        cols = [self._cols[k] for k in self._key_columns(subset)]
        return Series._new(_unique.duplicated(cols, keep), self._index, None)

    def drop_duplicates(self, subset=None, *, keep="first", ignore_index=False):
        """The rows that duplicated(subset, keep=keep) does not mark."""
        keep_rows = _core.unary_op("invert", self.duplicated(subset, keep=keep)._col)
        return self._take(keep_rows.nonzero(), ignore_index)

    def value_counts(self, subset=None, *, normalize=False, sort=True, ascending=False, dropna=True):
        """A DataFrame of each distinct row of `subset` and how often it appears (API.md §13)."""
        names = self._key_columns(subset)
        count_name = "proportion" if normalize else "count"
        if count_name in names:
            raise ValueError(f"a key column cannot be named {count_name!r}")
        frame = self.dropna(subset=names) if dropna else self
        keys, counts = _unique.value_counts([frame._cols[k] for k in names], normalize, sort, ascending)
        cols = dict(zip(names, keys))
        cols[count_name] = counts
        return DataFrame._new(cols, RangeIndex(len(counts)))

    # --- element-wise (API.md §12) ---

    __hash__ = None
    __array_ufunc__ = None

    def _binary(self, other, op, reflected=False):
        if isinstance(other, DataFrame):
            # Rows by label, columns by name; what one side lacks is NA there (API.md §14).
            index, a_pos, b_pos = _select.align(self._index, other._index)
            names = list(self._cols) + [k for k in other._cols if k not in self._cols]
            n = len(index)
            pairs = {}
            for k in names:
                a, b = self._cols.get(k), other._cols.get(k)
                a = a.take(a_pos) if a is not None and a_pos is not None else a
                b = b.take(b_pos) if b is not None and b_pos is not None else b
                a = a if a is not None else _column.full_na(n, b.dtype)
                b = b if b is not None else _column.full_na(n, a.dtype)
                pairs[k] = (a, b, 0)
        elif isinstance(other, Series):
            raise TypeError("operations between a DataFrame and a Series are not supported (API.md §12)")
        elif _ops.is_operand_scalar(other):
            index = self._index
            pairs = {k: (c, _ops.scalar_column(other, c.dtype), 2) for k, c in self._cols.items()}
        else:
            return NotImplemented
        cols = {}
        for k, (a, b, side) in pairs.items():
            if reflected:
                a, b, side = b, a, 1 if side == 2 else side
            cols[k] = _ops.apply(op, a, b, side)
        return DataFrame._new(cols, index)

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
            kinds = {c.dtype for c in self._cols.values()}
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


def _from_dict(data, dtype, target):
    """Columns from {name: list | array | Series | scalar}. Series are matched
    by label: to `target` when given, else to the union of their indexes
    (API.md §14). Lists must have one value per row; scalars fill every row."""
    names = _check_columns(data.keys())
    series = [v for v in data.values() if isinstance(v, Series)]
    index = target
    if index is None and series:
        index = series[0]._index
        for v in series[1:]:
            index, _, _ = _select.align(index, v._index)
    length = len(index) if index is not None else None
    cols = {}
    for k in names:
        v = data[k]
        if _column.is_scalar(v):
            cols[k] = v  # filled in below, once the length is known
            continue
        if isinstance(v, Series):
            col = _select.reindexed(v._col, v._index, index)
            col = col.convert(dtype) if dtype is not None else col
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


class _FrameIndexer:
    """df.loc / df.iloc. A single row gives a one-row DataFrame, since a row
    of mixed dtypes has no dtype of its own (API.md §14)."""

    __slots__ = ("_df", "_by_label")

    def __init__(self, df, by_label):
        self._df = df
        self._by_label = by_label

    def _split(self, key):
        rows, cols = key if isinstance(key, tuple) and len(key) == 2 else (key, slice(None))
        df = self._df
        sel = _select.by_label(df._index, rows) if self._by_label else _select.by_position(len(df), rows)
        return sel, self._column_names(cols)

    def _column_names(self, key):
        names = list(self._df._cols)
        if self._by_label:
            if isinstance(key, str):
                self._df._column(key)
                return key
            if isinstance(key, slice):
                if key.step not in (None, 1):
                    raise ValueError("a column slice cannot have a step")
                start = names.index(key.start) if key.start is not None else 0
                stop = names.index(key.stop) + 1 if key.stop is not None else len(names)  # both ends included
                return names[start:stop]
            names_key = _check_columns(key)
            for k in names_key:
                self._df._column(k)
            return names_key
        if isinstance(key, slice):
            return names[key]
        if isinstance(key, list):
            return [names[_select.by_position(len(names), j)[1]] for j in key]
        return names[_select.by_position(len(names), key)[1]]

    def __getitem__(self, key):
        df = self._df
        sel, names = self._split(key)
        if isinstance(names, str):  # one column
            return Series._new(df._cols[names], df._index, names)._select(sel)
        if sel[0] == "one":  # one row: a one-row DataFrame keeps every column's dtype
            sel = ("slice", sel[1], sel[1] + 1, 1)
        cols = {k: _select.select_column(df._cols[k], sel) for k in names}
        return DataFrame._new(cols, _select.select_index(df._index, sel))

    def __setitem__(self, key, value):
        df = self._df
        sel, name = self._split(key)
        if not isinstance(name, str):
            raise TypeError("assign to one column at a time: df.loc[rows, 'c'] = value (API.md §14)")
        target = Series._new(df._cols[name], df._index, name)
        target._assign(sel, value)  # Copy-on-Write: target takes its own Column first
        df._cols[name] = target._col
