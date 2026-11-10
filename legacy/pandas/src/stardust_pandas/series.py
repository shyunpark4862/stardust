"""Series: one Column plus its row labels (ARCHITECTURE.md §1)."""

from . import _column, _format
from ._column import _NO_VALUE
from ._core import Column
from ._dtypes import resolve
from ._na import NA
from .index import RangeIndex


def check_name(name):
    if name is not None and not isinstance(name, str):
        raise TypeError(f"a Series name must be a str or None, got {type(name).__name__}")  # API.md §10
    return name


class Series:
    __slots__ = ("_col", "_index", "_name")

    def __init__(self, data=None, *, dtype=None, name=None):
        dtype = resolve(dtype) if dtype is not None else None
        if isinstance(data, Series):
            col = data._col.convert(dtype) if dtype is not None else data._col  # shares the Buffers
            index = data._index
            if name is None:
                name = data._name
        else:
            if data is None:
                col = Column.from_list([], NA, dtype)
            elif isinstance(data, dict):
                raise NotImplementedError("a Series from a dict needs a label Index, which is not supported yet")
            else:
                col = _column.column_from(data, dtype)
            index = RangeIndex(len(col))
        self._col = col
        self._index = index
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
