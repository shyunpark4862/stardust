"""Series: one Column plus its row labels (ARCHITECTURE.md §1)."""

from . import _column, _core, _format, _ops, _select, _unique
from ._reduce import Reductions
from ._column import _NO_VALUE
from ._core import Column
from ._dtypes import resolve
from ._na import NA
from .index import RangeIndex, check_name, ensure_index


class Series(Reductions):
    __slots__ = ("_col", "_index", "_name")

    def __init__(self, data=None, *, index=None, dtype=None, name=None):
        dtype = resolve(dtype) if dtype is not None else None
        target = ensure_index(index) if index is not None else None
        if isinstance(data, Series):
            col = data._col.convert(dtype) if dtype is not None else data._col  # shares the Buffers
            if target is not None:  # labels given: rearrange to them (API.md §14)
                col = _select.reindexed(col, data._index, target)
            else:
                target = data._index
            if name is None:
                name = data._name
        else:
            if data is None:
                col = _column.full_na(len(target), dtype) if target is not None else Column.from_list([], NA, dtype)
            elif isinstance(data, dict):
                raise TypeError("a Series from a dict is not supported; pass the labels as index=")
            else:
                col = _column.column_from(data, dtype)
            if target is None:
                target = RangeIndex(len(col))
            elif len(target) != len(col):
                raise ValueError(f"{len(col)} values for an index of {len(target)} labels")
        self._col = col
        self._index = target
        self._name = check_name(name)

    @classmethod
    def _new(cls, col, index, name):
        obj = cls.__new__(cls)
        obj._col = col
        obj._index = index
        obj._name = name
        return obj

    @property
    def dtype(self):
        return self._col.dtype

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = check_name(value)

    @property
    def index(self):
        return self._index

    @property
    def shape(self):
        return (len(self._col),)

    def __len__(self):
        return len(self._col)

    def __bool__(self):
        raise ValueError("the truth value of a Series is ambiguous")

    def __iter__(self):
        """The values, as in a list (API.md §14)."""
        return iter(self.to_list())

    def __contains__(self, value):
        """Whether some row holds `value`, by the key rules of API.md §13 (NaN
        matches NaN, NA matches NA). Labels: `label in s.index`."""
        try:
            probe = Column.from_list([value], NA, self.dtype)
        except (TypeError, ValueError, OverflowError):
            return False  # a value of another type is in no row
        return self._col.contains_key(probe)

    def to_list(self):
        """The values as Python scalars; NA rows are sdpd.NA."""
        return self._col.to_list(NA)

    def to_numpy(self, *, dtype=None, copy=False, na_value=_NO_VALUE):
        """The values as a numpy array (API.md §10).

        Without NA this is a read-only view of the column (copy=True gives a
        writable copy). With NA, na_value= fills those rows; it must fit the
        result dtype. dtype= converts first, without loss only. A string
        column gives an object array of str.
        """
        return _column.to_numpy(self._col, resolve(dtype) if dtype is not None else None, copy, na_value)

    def __array__(self, dtype=None, copy=None):
        arr = self.to_numpy(copy=bool(copy))
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def _slice(self, start, stop):
        return Series._new(self._col.slice(start, stop - start), self._index._slice(start, stop), self._name)

    def head(self, n=5):
        """The first n rows; with n < 0, all rows but the last -n."""
        return self._slice(*_column.head_bounds(len(self), n))

    def tail(self, n=5):
        """The last n rows; with n < 0, all rows but the first -n."""
        return self._slice(*_column.tail_bounds(len(self), n))

    def __repr__(self):
        return _format.series_repr(self)

    # --- element-wise (API.md §12) ---

    __hash__ = None  # == returns a Series, so a Series cannot be a dict key
    __array_ufunc__ = None  # numpy defers to our reflected operators instead of converting us

    def _binary(self, other, op, reflected=False):
        if isinstance(other, Series):
            index, a_pos, b_pos = _select.align(self._index, other._index)
            a = self._col.take(a_pos) if a_pos is not None else self._col
            b = other._col.take(b_pos) if b_pos is not None else other._col
            side, name = 0, _select.merged_name(self._name, other._name)
        elif _ops.is_operand_scalar(other):
            a, b, side = self._col, _ops.scalar_column(other, self._col.dtype), 2
            index, name = self._index, self._name
        else:
            return NotImplemented
        if reflected:
            a, b, side = b, a, 1 if side == 2 else side
        return Series._new(_ops.apply(op, a, b, side), index, name)

    def _unary(self, op):
        return Series._new(_core.unary_op(op, self._col), self._index, self._name)

    def __neg__(self):
        return self._unary("neg")

    def __abs__(self):
        return self._unary("abs")

    def __pos__(self):
        if self.dtype not in ("int64", "float64"):
            raise TypeError(f"unsupported operand dtype for unary +: {self.dtype}")
        return self

    def __invert__(self):
        return self._unary("invert")

    # --- NA and dtypes ---

    def astype(self, dtype):
        """The values as `dtype` (API.md §12): rounds int -> float, truncates
        float -> int, formats to string like str(), parses strings strictly."""
        return Series._new(self._col.astype(resolve(dtype)), self._index, self._name)

    def isna(self):
        """True where the value is NA. NaN is a value, not NA (API.md §1)."""
        return Series._new(self._col.isna(), self._index, self._name)

    def notna(self):
        return Series._new(self._col.notna(), self._index, self._name)

    def fillna(self, value):
        """NA rows set to `value`, which must convert to the dtype without loss (API.md §12)."""
        return Series._new(fill_column(self._col, value), self._index, self._name)

    def dropna(self, *, ignore_index=False):
        """The rows that are not NA."""
        return self._take(self._col.notna().nonzero(), ignore_index)

    # --- duplicates and unique values (API.md §13) ---

    def unique(self):
        """The distinct values in order of first appearance, NA included (API.md §11 C6)."""
        _, first, _ = _unique.codes([self._col])
        return Series._new(self._col.take(first), RangeIndex(len(first)), self._name)

    def nunique(self, *, dropna=True):
        """How many distinct values. NaN counts as a value; dropna drops NA only (API.md §13)."""
        _, _, ngroups = _unique.codes([self._col])
        return ngroups - (1 if dropna and self._col.null_count else 0)

    def duplicated(self, *, keep="first"):
        """True for rows that repeat a value (see API.md §13 for keep)."""
        return Series._new(_unique.duplicated([self._col], keep), self._index, self._name)

    def drop_duplicates(self, *, keep="first", ignore_index=False):
        """The rows that duplicated(keep=keep) does not mark."""
        keep_rows = _core.unary_op("invert", _unique.duplicated([self._col], keep))
        return self._take(keep_rows.nonzero(), ignore_index)

    def value_counts(self, *, normalize=False, sort=True, ascending=False, dropna=True):
        """A DataFrame of each value and how often it appears (API.md §13)."""
        from .frame import DataFrame

        name = "value" if self._name is None else self._name
        count_name = "proportion" if normalize else "count"
        if name == count_name:
            raise ValueError(f"the value column cannot be named {count_name!r}")
        col = self._col.take(self._col.notna().nonzero()) if dropna and self._col.null_count else self._col
        (keys,), counts = _unique.value_counts([col], normalize, sort, ascending)
        return DataFrame._new({name: keys, count_name: counts}, RangeIndex(len(counts)))

    def _reduce(self, op, skipna, ddof):  # API.md §15: a Python scalar or NA
        return _core.reduce(op, self._col, skipna, ddof).to_list(NA)[0]

    # --- selection and assignment (API.md §14) ---

    def _take(self, positions, ignore_index=False):
        index = RangeIndex(len(positions)) if ignore_index else self._index._take(positions)
        return Series._new(self._col.take(positions), index, self._name)

    def _select(self, sel):
        if sel[0] == "one":
            return self._col.slice(sel[1], 1).to_list(NA)[0]
        return Series._new(_select.select_column(self._col, sel), _select.select_index(self._index, sel), self._name)

    def _assign(self, sel, value):
        labels = None if sel[0] == "one" else _select.select_index(self._index, sel)
        values = assignment_values(value, self.dtype, labels)
        # A Column object of our own, so the write cannot show through another
        # object holding the old one; set_rows copies shared Buffers (CoW).
        self._col = self._col.share()
        self._col.set_rows(_select.positions_column(sel), values)

    @property
    def loc(self):
        """Rows by label: a label (a scalar), a list of labels, a label slice
        (both ends included) or a bool Series mask (API.md §14)."""
        return _Indexer(self, by_label=True)

    @property
    def iloc(self):
        """Rows by position: an int, a slice or a list of ints."""
        return _Indexer(self, by_label=False)

    def __getitem__(self, key):
        return self._select(_select.by_label(self._index, key))  # s[x] is s.loc[x] (API.md §7)

    def __setitem__(self, key, value):
        self._assign(_select.by_label(self._index, key), value)

    def reindex(self, labels):
        """The values at `labels`, in that order; absent labels are NA (API.md §14)."""
        target = ensure_index(labels)
        return Series._new(_select.reindexed(self._col, self._index, target), target, self._name)


class _Indexer:
    __slots__ = ("_obj", "_by_label")

    def __init__(self, obj, by_label):
        self._obj = obj
        self._by_label = by_label

    def _sel(self, key):
        obj = self._obj
        return _select.by_label(obj._index, key) if self._by_label else _select.by_position(len(obj), key)

    def __getitem__(self, key):
        return self._obj._select(self._sel(key))

    def __setitem__(self, key, value):
        self._obj._assign(self._sel(key), value)


def assignment_values(value, dtype, labels):
    """`value` as a Column of `dtype` for rows labelled `labels` (None: one
    row). Converts without loss only (API.md §14); a Series is matched to the
    rows by label."""
    if isinstance(value, Series):
        if labels is None:
            raise TypeError("assign a scalar to a single row")
        col = _select.reindexed(value._col, value._index, labels)
        return col.convert(dtype) if col.dtype != dtype else col
    if _column.is_scalar(value):
        return Column.from_list([value], NA, dtype)
    if labels is None:
        raise TypeError("assign a scalar to a single row")
    col = _column.column_from(value, dtype)
    if len(col) != len(labels):
        raise ValueError(f"{len(col)} values for {len(labels)} rows")
    return col


def fill_column(col, value):
    if value is NA or value is None:
        return col
    if not _ops.is_operand_scalar(value):
        raise TypeError(f"fillna takes a scalar, got {type(value).__name__}")
    return col.fill_na(Column.from_list([value], NA, col.dtype))  # lossless rule of dtype= (API.md §10)


def _make_binary(op, reflected):
    def method(self, other):
        return self._binary(other, op, reflected)

    return method


for _name, _op in {**_ops.ARITH, **_ops.LOGIC}.items():
    setattr(Series, f"__{_name}__", _make_binary(_op, False))
    setattr(Series, f"__r{_name}__", _make_binary(_op, True))
for _name in _ops.COMPARE:
    setattr(Series, f"__{_name}__", _make_binary(_name, False))
