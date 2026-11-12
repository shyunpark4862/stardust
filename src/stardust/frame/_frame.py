"""DataFrame: named Columns of one length plus row labels (docs/design/frame.md §5.1)."""

from . import _column, _core, _dtypes, _format, _ops, _select, _sort, _unique
from ._column import _NO_VALUE
from ._errors import check_bool, check_choice, check_int, name_list, named, type_name
from ._dtypes import resolve
from ._reduce import Reductions
from ._na import NA
from ._index import Index, RangeIndex, ensure_index
from ._series import Series, assignment_values, fill_column

Column = _core.Column


def _check_columns(columns, func=None):
    """`func`: the public function, or None for indexing (df[[...]], .loc)."""
    from ._index import _IndexBase

    # a list-like of names; not a generator (as the array area's list arguments)
    if not isinstance(columns, (list, tuple, dict, type({}.keys()), _IndexBase)) and not _column.is_ndarray(columns):
        raise TypeError(named(func, f"column names must be given as a list, not {type_name(columns)}"))
    columns = list(columns)
    for c in columns:
        if not isinstance(c, str):
            raise TypeError(named(func, f"column names must be str, not {type_name(c)}"))  # docs/api/frame.md §1.7
    if len(set(columns)) != len(columns):
        raise ValueError(named(func, "column names must be unique"))
    return columns


class DataFrame(Reductions):
    __slots__ = ("_cols", "_index")

    def __init__(self, data=None, /, *, index=None, columns=None, dtype=None):
        dtype = resolve(dtype, "DataFrame") if dtype is not None else None
        target = ensure_index(index, "DataFrame") if index is not None else None
        if columns is not None:
            columns = _check_columns(columns, "DataFrame")
        if data is None:
            cols, natural = {}, target if target is not None else RangeIndex(0)
            if columns is not None:
                cols = dict.fromkeys(columns)
        elif isinstance(data, DataFrame):
            cols, natural = dict(data._cols), data._index  # shares the Buffers
            if dtype is not None:
                cols = {k: c.astype(dtype, func="DataFrame") for k, c in cols.items()}
            if columns is not None:
                cols = {k: cols.get(k) for k in columns}
            if target is not None:  # labels given: rearrange to them (docs/api/frame.md §2.1)
                cols = {k: c if c is None else _select.reindexed(c, natural, target, "DataFrame") for k, c in cols.items()}
                natural = target
        elif isinstance(data, dict):
            cols, natural = _from_dict(data, dtype, target)
            if columns is not None:
                cols = {k: cols.get(k) for k in columns}
        elif _column.is_sdarray(data):  # always copied (docs/api/frame.md §2.8)
            if data.ndim != 2:
                raise ValueError(f"DataFrame(): a DataFrame from an sd.ndarray needs 2 dimensions, got {data.ndim}")
            names = _require_columns(columns, data.shape[1])
            cols = {k: _column.column_from(data[:, j], dtype, "DataFrame") for j, k in enumerate(names)}
            natural = RangeIndex(data.shape[0])
        elif _column.is_ndarray(data):
            if data.ndim != 2:
                raise ValueError(f"DataFrame(): a DataFrame from a numpy array needs 2 dimensions, got {data.ndim}")
            names = _require_columns(columns, data.shape[1])
            cols = {k: _column.column_from(data[:, j], dtype, "DataFrame") for j, k in enumerate(names)}
            natural = RangeIndex(data.shape[0])
        elif isinstance(data, (list, tuple)):
            cols, natural = _from_rows(data, columns, dtype)
        else:
            raise TypeError(f"DataFrame(): cannot make a DataFrame from {type(data).__name__}")
        if target is not None and natural is not target:
            if len(natural) != len(target) and cols:
                raise ValueError(f"DataFrame(): {len(natural)} rows for an index of {len(target)} labels")
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

    def __reduce__(self):  # pickling: each Column pickles its own rows (DESIGN.md §10.1 #12)
        return (DataFrame._new, (self._cols, self._index))

    @property
    def columns(self):
        """The column names as a string Index (docs/api/frame.md §2.6)."""
        return Index._new(Column.from_list(list(self._cols), NA, "string"))

    @property
    def dtypes(self):
        """The dtype of each column: {column name: dtype object}, in column order."""
        return {k: _dtypes.obj(c.dtype) for k, c in self._cols.items()}

    @property
    def index(self):
        return self._index

    @property
    def shape(self):
        return (len(self._index), len(self._cols))

    def __len__(self):
        return len(self._index)

    def __bool__(self):
        raise TypeError("the truth value of a DataFrame is ambiguous")  # as an array's (DESIGN.md §10.1 #9)

    def __iter__(self):
        """The column names (docs/api/frame.md §5.8)."""
        return iter(list(self._cols))

    def __contains__(self, name):
        return name in self._cols

    def _column(self, name, func=None):
        """The column `name`. For indexing (func None) a missing one is
        KeyError(name), as in a dict; a public function names itself."""
        if not isinstance(name, str):
            raise TypeError(named(func, f"column names must be str, not {type_name(name)}"))
        try:
            return self._cols[name]
        except KeyError:
            raise KeyError(name if func is None else f"{func}(): no column {name!r}") from None

    def __getitem__(self, key):
        """df["a"] (a Series), df[["a", "b"]] (a DataFrame) or df[mask] (rows) (docs/api/frame.md §5.3)."""
        if isinstance(key, Series):  # rows by a bool mask: each column compacted directly (docs/design/frame.md §7.3)
            sel = ("mask", _select.aligned_mask(key, self._index))
            return DataFrame._new({k: _select.select_column(c, sel) for k, c in self._cols.items()},
                                  _select.select_index(self._index, sel))
        if isinstance(key, list):
            names = _check_columns(key)
            return DataFrame._new({k: self._column(k) for k in names}, self._index)
        return Series._new(self._column(key), self._index, key)

    def __setitem__(self, name, value):
        """df["c"] = value: a new column at the end, or a whole column replaced (docs/api/frame.md §5.7)."""
        if not isinstance(name, str):
            raise TypeError(f"column names must be str, not {type_name(name)}")
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
        """Rows (and columns) by label: df.loc[rows] or df.loc[rows, columns] (docs/api/frame.md §5.4)."""
        return _FrameIndexer(self, by_label=True)

    @property
    def iloc(self):
        """Rows (and columns) by position: df.iloc[rows] or df.iloc[rows, columns]."""
        return _FrameIndexer(self, by_label=False)

    def reindex(self, labels, /):
        """The rows at `labels`, in that order; absent labels are NA (docs/api/frame.md §5.2)."""
        target = ensure_index(labels, "reindex")
        cols = {k: _select.reindexed(c, self._index, target, "reindex") for k, c in self._cols.items()}
        return DataFrame._new(cols, target)

    def _take(self, positions, ignore_index=False):
        index = RangeIndex(len(positions)) if ignore_index else self._index._take(positions)
        return DataFrame._new({k: c.take(positions) for k, c in self._cols.items()}, index)

    def _map_columns(self, f, per_column=None, func=None):
        """A DataFrame with f(column) (or f(column, per_column[name])) for each column."""
        if per_column is None:
            return DataFrame._new({k: f(c) for k, c in self._cols.items()}, self._index)
        for k in per_column:  # the first missing one, as every other column argument (_column)
            self._column(k, func)
        return DataFrame._new({k: f(c, per_column[k]) if k in per_column else c for k, c in self._cols.items()}, self._index)

    def _reduce(self, op, skipna, correction):
        """One row, labelled by the operation, each column keeping its own result
        dtype (docs/api/frame.md §6.2). A column the operation does not apply to raises."""
        cols = {k: _core.reduce(op, c, skipna, correction, op) for k, c in self._cols.items()}
        return DataFrame._new(cols, Index._new(Column.from_list([op], NA, "string")))

    def sort_values(self, by, /, *, descending=False, stable=True, na_position="last", ignore_index=False):
        """The rows ordered by the column(s) `by`, first key first (docs/api/frame.md §8.1).
        descending and na_position take one value or one per key. Always stable."""
        names = name_list(by, "by", "sort_values")
        if not names:
            raise ValueError("sort_values(): 'by' must name at least one column")
        keys = [self._column(k, "sort_values") for k in names]
        check_bool(ignore_index, "ignore_index", "sort_values")
        return self._take(_sort.order(keys, descending, na_position, stable, "sort_values"), ignore_index)

    def sort_index(self, *, descending=False, stable=True, ignore_index=False):
        """The rows ordered by label."""
        check_bool(ignore_index, "ignore_index", "sort_index")
        return self._take(_sort.order([self._index._column()], descending, "last", stable, "sort_index"), ignore_index)

    def merge(self, right, /, *, how="inner", on=None, left_on=None, right_on=None, sort=False, suffixes=("_x", "_y")):
        """sd.frame.merge(self, right, ...): a hash join on key columns (docs/api/frame.md §9.1)."""
        from ._merge import merge

        return merge(self, right, how=how, on=on, left_on=left_on, right_on=right_on, sort=sort, suffixes=suffixes)

    def groupby(self, by, /, *, sort=True, dropna=True):
        """Group the rows by the values of the key column(s) `by` (docs/api/frame.md §7.1)."""
        from ._groupby import GroupBy

        return GroupBy(self, by, sort=sort, dropna=dropna)

    # --- duplicates and unique values (docs/api/frame.md §4) ---

    def _key_columns(self, subset, func):
        names = list(self._cols) if subset is None else name_list(subset, "subset", func)
        if not names:
            raise ValueError(f"{func}(): 'subset' must name at least one column")
        for k in names:
            if k not in self._cols:
                raise KeyError(f"{func}(): no column {k!r}")
        return names

    def nunique(self, *, dropna=True):
        """How many distinct values each column has, as an int64 Series labelled by column name."""
        check_bool(dropna, "dropna", "nunique")
        counts = [self[k].nunique(dropna=dropna) for k in self._cols]
        return Series._new(Column.from_list(counts, NA, "int64"), self.columns, None)

    def duplicated(self, subset=None, /, *, keep="first"):
        """True for rows whose values in `subset` (default: all columns) repeat (docs/api/frame.md §4.1)."""
        return Series._new(self._duplicated(subset, keep, "duplicated"), self._index, None)

    def _duplicated(self, subset, keep, func):
        return _unique.duplicated([self._cols[k] for k in self._key_columns(subset, func)], keep, func)

    def drop_duplicates(self, subset=None, /, *, keep="first", ignore_index=False):
        """The rows that duplicated(subset, keep=keep) does not mark."""
        keep_rows = _core.unary_op("invert", self._duplicated(subset, keep, "drop_duplicates"))
        return self._take(keep_rows.nonzero(), check_bool(ignore_index, "ignore_index", "drop_duplicates"))

    def value_counts(self, subset=None, /, *, normalize=False, sort=True, descending=True, dropna=True):
        """A DataFrame of each distinct row of `subset` and how often it appears (docs/api/frame.md §4.3)."""
        names = self._key_columns(subset, "value_counts")
        check_bool(normalize, "normalize", "value_counts")
        check_bool(dropna, "dropna", "value_counts")
        count_name = "proportion" if normalize else "counts"  # the field of unique_counts
        if count_name in names:
            raise ValueError(f"value_counts(): a key column cannot be named {count_name!r}")
        frame = self.dropna(subset=names) if dropna else self
        keys, counts = _unique.value_counts([frame._cols[k] for k in names], normalize, sort, descending, "value_counts")
        cols = dict(zip(names, keys))
        cols[count_name] = counts
        return DataFrame._new(cols, RangeIndex(len(counts)))

    # --- element-wise (docs/api/frame.md §3) ---

    __hash__ = None
    __array_ufunc__ = None

    def _binary(self, other, op, reflected=False):
        if isinstance(other, DataFrame):
            # Rows by label, columns by name; what one side lacks is NA there (docs/api/frame.md §5.6).
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
            raise TypeError("operations between a DataFrame and a Series are not supported")
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
        bad = [k for k, c in self._cols.items() if c.dtype not in ("bool", "int64", "float64")]
        if bad:
            raise TypeError(f"unsupported operand dtype for unary + in columns {bad}")
        if all(c.dtype != "bool" for c in self._cols.values()):
            return self
        return self._map_columns(lambda c: c.astype("int64") if c.dtype == "bool" else c)  # bool as int64, as with -

    def __invert__(self):
        return self._map_columns(lambda c: _core.unary_op("invert", c))

    # --- NA and dtypes ---

    def astype(self, dtype, /):
        """Every column, or the columns of a {name: dtype} dict, as that dtype (docs/api/frame.md §3.5)."""
        if isinstance(dtype, dict):
            return self._map_columns(lambda c, dt: c.astype(resolve(dt, "astype"), func="astype"), dtype, "astype")
        dtype = resolve(dtype, "astype")
        return self._map_columns(lambda c: c.astype(dtype, func="astype"))

    def isna(self):
        return self._map_columns(lambda c: c.isna())

    def notna(self):
        return self._map_columns(lambda c: c.notna())

    def isnan(self):
        """Series.isnan for every column (docs/api/frame.md §3.6)."""
        return self._map_columns(_core.isnan)

    def fillna(self, value, /):
        """NA rows set to `value`, or per column from a {name: value} dict (docs/api/frame.md §3.6)."""
        if isinstance(value, dict):
            return self._map_columns(lambda c, v: fill_column(c, v, "fillna"), value, "fillna")
        return self._map_columns(lambda c: fill_column(c, value, "fillna"))

    def dropna(self, *, how="any", subset=None, ignore_index=False):
        """The rows without NA in `subset` (default: all columns); with how="all",
        only rows where every `subset` value is NA are dropped."""
        check_choice(how, "how", ("any", "all"), "dropna")
        check_bool(ignore_index, "ignore_index", "dropna")
        names = list(self._cols) if subset is None else name_list(subset, "subset", "dropna")
        for k in names:
            if k not in self._cols:
                raise KeyError(f"dropna(): no column {k!r}")
        keep = None
        combine = "and" if how == "any" else "or"
        for k in names:
            valid = self._cols[k].notna()
            keep = valid if keep is None else _core.binary_op(combine, keep, valid, 0)
        if keep is None:  # no columns to look at: every row stays
            keep = _core.Column.from_list([True] * len(self), None, None)
        return self._take(keep.nonzero(), ignore_index)

    def to_numpy(self, *, dtype=None, copy=None, na_value=_NO_VALUE):
        """A 2-D numpy array (docs/api/frame.md §2.7) of the columns' common dtype:
        bool < int64 < float64, as in the array area; a string column with
        another dtype has none. NA rows follow the Series.to_numpy rules. The
        array is always new (the columns are separate buffers), so copy=False
        is a ValueError."""
        return self._to_numpy(dtype, copy, na_value, "to_numpy")

    def _to_numpy(self, dtype, copy, na_value, func):
        import numpy as np

        if _column.check_copy(copy, func) is False:
            raise ValueError(named(func, "copy=False, but a 2-D array of the columns requires a copy"))
        dtype = resolve(dtype, func) if dtype is not None else None
        if dtype is None:
            dtype = _column.common_dtype(self._cols.values(), func, "numpy")
        arrays = [_column.to_numpy(c, dtype, True, na_value, func) for c in self._cols.values()]
        if not arrays:
            return np.empty((len(self), 0))
        return np.stack(arrays, axis=1)

    def __array__(self, dtype=None, copy=None):
        """numpy's protocol: to_numpy's array (DESIGN.md §10.1 #6), no function name in messages."""
        arr = self._to_numpy(None, copy, _NO_VALUE, None)
        if dtype is None:
            return arr
        import numpy as np

        return np.asarray(arr, dtype=dtype, copy=False if copy is False else None)

    def to_array(self, *, dtype=None, copy=None, na_value=_NO_VALUE):
        """A new 2-D sd.ndarray in F order (docs/api/frame.md §2.8): one column after
        another in memory, each copied once, the layout sd.linalg works in.
        The dtype rules are to_numpy's; copy=False is a ValueError."""
        if _column.check_copy(copy, "to_array") is False:
            raise ValueError("to_array(): copy=False, but a 2-D array of the columns requires a copy")
        dtype = resolve(dtype, "to_array") if dtype is not None else None
        if dtype is None:
            dtype = _column.common_dtype(self._cols.values(), "to_array", "array")
        cols = [_column.array_column(c, dtype, True, na_value, "to_array")[0] for c in self._cols.values()]
        if cols and cols[0].dtype == "string":
            raise TypeError("to_array(): a string column has no array (the array area has no string dtype)")
        return _core.columns_to_array(cols, len(self))

    def _slice(self, start, stop):
        cols = {k: c.slice(start, stop - start) for k, c in self._cols.items()}
        return DataFrame._new(cols, self._index._slice(start, stop))

    def head(self, n=5, /):
        """The first n rows; with n < 0, all rows but the last -n."""
        return self._slice(*_column.head_bounds(len(self), check_int(n, "n", "head")))

    def tail(self, n=5, /):
        """The last n rows; with n < 0, all rows but the first -n."""
        return self._slice(*_column.tail_bounds(len(self), check_int(n, "n", "tail")))

    def __repr__(self):
        return _format.frame_repr(self)


_ops.attach(DataFrame, lambda: ())  # DataFrame._binary itself rejects a Series


def _require_columns(columns, ncols):  # for DataFrame()
    if columns is None:
        raise ValueError("DataFrame(): 'columns' is required to name the columns of row or 2-D array data")  # docs/api/frame.md §2.5
    if len(columns) != ncols:
        raise ValueError(f"DataFrame(): {len(columns)} column names for {ncols} columns")
    return columns


def _from_dict(data, dtype, target):
    """Columns from {name: list | array | Series | scalar}. Series are matched
    by label: to `target` when given, else to the union of their indexes
    (docs/api/frame.md §2.5). Lists must have one value per row; scalars fill every row.
    For DataFrame(): its messages name it."""
    names = _check_columns(data.keys(), "DataFrame")
    series = [v for v in data.values() if isinstance(v, Series)]
    index = target
    if index is None and series:
        index = series[0]._index
        for v in series[1:]:
            index, _, _ = _select.align(index, v._index, "DataFrame")
    length = len(index) if index is not None else None
    cols = {}
    for k in names:
        v = data[k]
        if _column.is_scalar(v):
            cols[k] = v  # filled in below, once the length is known
            continue
        if isinstance(v, Series):
            col = _select.reindexed(v._col, v._index, index, "DataFrame")
            col = col.astype(dtype, func="DataFrame") if dtype is not None else col
        else:
            col = _column.column_from(v, dtype, "DataFrame")
        if length is None:
            length = len(col)
        elif len(col) != length:
            if target is not None:  # index= sets the length: say so (DESIGN.md §8.4)
                raise ValueError(f"DataFrame(): column {k!r} has {len(col)} values, but 'index' has {length} labels")
            raise ValueError("DataFrame(): all columns must have the same length")
        cols[k] = col
    if length is None:
        if cols:
            raise ValueError("DataFrame(): cannot make a DataFrame from scalars alone; give at least one column as a list")
        length = 0
    for k, v in cols.items():
        if _column.is_scalar(v):
            cols[k] = _column.column_from([v] * length, dtype, "DataFrame")
    return cols, index if index is not None else RangeIndex(length)


def _from_rows(rows, columns, dtype):  # for DataFrame()
    if len(rows) == 0:
        return dict.fromkeys(columns or []), RangeIndex(0)
    if all(isinstance(r, dict) for r in rows):  # records: missing keys are NA
        names = {}
        for r in rows:
            names.update(dict.fromkeys(r))
        names = _check_columns(names, "DataFrame")
        cols = {k: _column.column_from([r.get(k) for r in rows], dtype, "DataFrame") for k in names}
        if columns is not None:
            cols = {k: cols.get(k) for k in columns}
        return cols, RangeIndex(len(rows))
    if all(isinstance(r, (list, tuple)) or _column.is_ndarray(r) for r in rows):
        names = _require_columns(columns, len(rows[0]))
        for r in rows:
            if len(r) != len(names):
                raise ValueError(f"DataFrame(): every row must have {len(names)} values, one per column")
        values = [list(c) for c in zip(*rows)]
        return {k: _column.column_from(v, dtype, "DataFrame") for k, v in zip(names, values)}, RangeIndex(len(rows))
    raise TypeError("DataFrame(): a DataFrame from a list needs rows: all dicts, or all lists/tuples")


class _FrameIndexer:
    """df.loc / df.iloc. A single row gives a one-row DataFrame, since a row
    of mixed dtypes has no dtype of its own (docs/api/frame.md §5.4)."""

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
        if isinstance(key, list) or _select.is_array(key):  # a numpy array or sd.ndarray of positions too
            return [names[_select.by_position(len(names), j, "columns")[1]] for j in _select.array_items(key)]
        return names[_select.by_position(len(names), key, "columns")[1]]

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
            raise TypeError("assign to one column at a time: df.loc[rows, 'c'] = value")
        target = Series._new(df._cols[name], df._index, name)
        target._assign(sel, value)  # Copy-on-Write: target takes its own Column first
        df._cols[name] = target._col
