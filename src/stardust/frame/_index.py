"""Row labels: RangeIndex (three integers) and Index (an immutable int64 or
string Column), each with an optional name (docs/api/frame.md §5.1).

Label lookups go through a LabelTable, the hash table built on first use
and kept by the Index (docs/design/frame.md §6.2): an Index never changes, so the
table never goes stale.
"""

import operator
import sys

from .. import _calls
from . import _core, _dtypes, _format
from ._errors import check_int, named, type_name
from ._na import NA

Column = _core.Column
LabelTable = _core.LabelTable


def check_name(name, what="a Series", func=None):
    """`func` is None for the name setter (attribute assignment keeps Python's form)."""
    if name is not None and not isinstance(name, str):
        # docs/api/frame.md §2.6, §5.1
        if func is None:
            raise TypeError(f"the name of {what} must be a str or None, not {type_name(name)}")
        raise TypeError(f"{func}(): 'name' must be a str or None, not {type_name(name)}")
    return name


def _is_int_label(x):
    np = sys.modules.get("numpy")
    if isinstance(x, bool) or (np is not None and isinstance(x, np.bool_)):
        return False
    return isinstance(x, int) or (np is not None and isinstance(x, np.integer))


def check_label(index, label, func=None):
    """Raises TypeError unless `label` has the type of the index's labels (docs/api/frame.md §5.1)."""
    ok = _is_int_label(label) if index._dtype == "int64" else isinstance(label, str)
    if not ok:
        kind = "an int" if index._dtype == "int64" else "a str"
        raise TypeError(named(func, f"a label of {_an(index._dtype)} index must be {kind}, not {type_name(label)}"))


def mixed_labels(ours, theirs):
    """The message when labels of one dtype meet labels of the other (docs/api/frame.md §5.6):
    one wording for operators, reindex, masks, assignment and the constructors."""
    return f"cannot align {ours} labels with {theirs} labels"


def _an(dtype):
    return f"an {dtype}" if dtype == "int64" else f"a {dtype}"


def _an_obj(obj):
    return f"an {type_name(obj)}" if type_name(obj)[0] in "AEIOU" else f"a {type_name(obj)}"


def _probe(label, dtype):
    """A one-row Column of `dtype` equal to `label`, or None when no label of
    `dtype` equals it (an Index holds no NA)."""
    from ._column import exact

    if label is NA or label is None:
        return None
    try:
        return exact(label, dtype)
    except TypeError, ValueError, OverflowError:
        return None


def label_column(index, labels, func=None):
    """`labels` (a list-like, Index or RangeIndex) as a Column of the index's dtype."""
    if isinstance(labels, (Index, RangeIndex)):
        col = labels._column()
        if col.dtype != index._dtype:
            if len(col) == 0:  # no labels: nothing to mismatch (an empty list makes an int64 Index)
                return Column.from_list([], NA, index._dtype)
            raise TypeError(named(func, mixed_labels(index._dtype, col.dtype)))
        return col
    labels = list(labels)
    for x in labels:
        check_label(index, x, func)
    return Column.from_list(labels, NA, index._dtype)


class _IndexBase:
    __slots__ = ()

    @property
    def name(self):
        return self._name

    @property
    def is_unique(self):
        return self._table().is_unique

    def __iter__(self):
        return iter(self.tolist())

    def __contains__(self, label):
        """Whether some label equals `label`, by the rule of `x in s` (docs/api/frame.md
        §5.8): a value of another type is in no row unless it converts to the
        labels' dtype without loss (1.0 in an int Index is found)."""
        probe = _probe(label, self._dtype)
        return probe is not None and self._first_positions(probe).to_list(None)[0] >= 0

    # An Index is not a value: == and != are TypeErrors, as < is (labels are
    # compared with equals()), and an Index cannot be a dict key or a truth
    # value (DESIGN.md §10.1 #1b, #9, #12).
    __hash__ = None

    def __eq__(self, other):
        raise TypeError(
            f"'==' not supported between instances of {type_name(self)!r} and {type_name(other)!r}; "
            "compare labels with equals()"
        )

    def __ne__(self, other):
        raise TypeError(
            f"'!=' not supported between instances of {type_name(self)!r} and {type_name(other)!r}; "
            "compare labels with equals()"
        )

    def __bool__(self):
        raise TypeError(f"the truth value of {_an_obj(self)} is ambiguous")

    def __array__(self, dtype=None, copy=None):
        """numpy's protocol: the labels, by the rules of Series.to_numpy (docs/api/frame.md §2.7)."""
        from ._column import _NO_VALUE, to_numpy

        arr = to_numpy(self._column(), None, copy, _NO_VALUE, None)
        if dtype is None:
            return arr
        import numpy as np

        return np.asarray(arr, dtype=dtype, copy=False if copy is False else None)

    def equals(self, other, /):
        """True when `other` holds the same labels in the same order (names are not compared)."""
        if isinstance(self, RangeIndex) and isinstance(other, RangeIndex):
            return self._range == other._range
        return isinstance(other, _IndexBase) and self._column().equals(other._column())

    def _with_name(self, name):
        if name == self._name:
            return self
        obj = self._copy()
        obj._name = name
        return obj

    # --- lookups (positions are int64 Columns) ---

    def _first_positions(self, labels_col):
        return self._table().first_positions(labels_col)

    def _position_of(self, label):
        """The one position holding `label`: KeyError when absent or held by several rows (docs/api/frame.md §5.4)."""
        check_label(self, label)
        positions = self._table().all_positions(Column.from_list([label], NA, self._dtype)).to_list(None)
        if not positions:
            raise KeyError(label)
        if len(positions) > 1:
            raise KeyError(f"label {label!r} is not unique; select with a list ([{label!r}]) to get every row")
        return positions[0]

    def _positions_of(self, labels):
        """Every position holding each label, label by label (KeyError for an absent label)."""
        col = label_column(self, labels)
        missing = _core.binary_op("eq", self._first_positions(col), Column.from_list([-1], None, None), 2).nonzero()
        if len(missing):
            raise KeyError(f"labels not in the index: {col.take(missing).to_list(None)}")
        return self._table().all_positions(col)

    def _get_indexer(self, target, func=None):
        """Per label of `target` (an index), its position here or -1. Needs unique labels here."""
        if not self.is_unique:
            raise ValueError(named(func, "cannot reindex or align from an index with duplicate labels"))
        return self._first_positions(label_column(self, target, func))


class RangeIndex(_IndexBase):
    """The labels start, start + step, ... below stop, stored as three integers (docs/design/frame.md §6.1)."""

    __slots__ = ("_range", "_name", "_labels")

    @_calls.method("RangeIndex")
    def __init__(self, start, stop=None, step=1, /, *, name=None):
        # int arguments, as arange's start/stop/step are required and take ints (API.md §2)
        start, step = check_int(start, "start", "RangeIndex"), check_int(step, "step", "RangeIndex")
        if stop is not None:
            stop = check_int(stop, "stop", "RangeIndex")
        if step == 0:
            raise ValueError("RangeIndex(): 'step' must not be zero")
        if stop is None:
            start, stop = 0, start
        self._range = range(start, stop, step)
        try:
            len(self._range)
        except OverflowError:  # past Py_ssize_t, which is int64: no int64 position reaches the last labels
            raise ValueError("RangeIndex(): the range has more than 2**63 - 1 labels") from None
        self._name = check_name(name, "an Index", "RangeIndex")
        self._labels = None

    @classmethod
    def _from_range(cls, r, name=None):
        obj = cls.__new__(cls)
        obj._range = r
        obj._name = name
        obj._labels = None
        return obj

    def _copy(self):
        return RangeIndex._from_range(self._range, self._name)

    def __reduce__(self):  # pickled without the lookup table
        return (RangeIndex._from_range, (self._range, self._name))

    @property
    def start(self):
        return self._range.start

    @property
    def stop(self):
        return self._range.stop

    @property
    def step(self):
        return self._range.step

    @property
    def dtype(self):
        return _dtypes.obj("int64")

    @property
    def _dtype(self):  # the name, for the code here
        return "int64"

    @property
    def is_unique(self):
        return True

    def __len__(self):
        return len(self._range)

    def tolist(self):
        return list(self._range)

    def _column(self):
        r = self._range
        return _core.arange(r.start, r.step, len(r))

    def _table(self):
        if self._labels is None:  # built on first use (docs/design/frame.md §6.2)
            self._labels = LabelTable(self._column())
        return self._labels

    def __contains__(self, label):
        """Arithmetic, like _position_of: no labels or table are built (docs/design/frame.md §6.1).
        The probe follows Index.__contains__ (1.0 is found)."""
        probe = _probe(label, "int64")
        return probe is not None and probe.to_list(None)[0] in self._range

    def _position_of(self, label):
        check_label(self, label)
        r, i = self._range, operator.index(label)  # a Python int: range answers by arithmetic only for int
        if i not in r:  # arithmetic, no table needed
            raise KeyError(label)
        return r.index(i)

    def _slice(self, start, stop, step=1):  # the labels at positions range(start, stop, step)
        # Not self._range[start:stop:step]: the positions come from slice.indices(), where a
        # stop of -1 (a backward slice down to position 0) means "before 0", not "the last one".
        r, count = self._range, len(range(start, stop, step))
        first = r.start + start * r.step
        return RangeIndex._from_range(range(first, first + count * step * r.step, step * r.step), self._name)

    def _take(self, positions):  # an Index of the labels at `positions` (an int64 Column)
        r = self._range
        if r.start == 0 and r.step == 1:
            return Index._new(positions, self._name)  # the positions are the labels
        labels = _core.binary_op("mul", positions, Column.from_list([r.step], None, None), 2)
        return Index._new(_core.binary_op("add", labels, Column.from_list([r.start], None, None), 2), self._name)

    def _labels_at(self, positions):  # Python ints, for repr
        return [self._range[p] for p in positions]

    def __repr__(self):
        name = "" if self._name is None else f", name={_format.quote(self._name)}"
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step}{name})"


class Index(_IndexBase):
    """Labels in an immutable int64 or string Column without NA (docs/api/frame.md §1.7, §5.1)."""

    __slots__ = ("_col", "_name", "_labels")

    @_calls.method("Index")
    def __init__(self, data, /, *, name=None):
        self._col = _labels_from(data, "Index")
        self._name = check_name(name, "an Index", "Index")
        self._labels = None

    @classmethod
    def _new(cls, col, name=None):
        obj = cls.__new__(cls)
        obj._col = col
        obj._name = name
        obj._labels = None
        return obj

    def _copy(self):
        obj = Index._new(self._col, self._name)
        obj._labels = self._labels  # same labels, same table
        return obj

    def __reduce__(self):  # pickled without the lookup table
        return (Index._new, (self._col, self._name))

    @property
    def dtype(self):
        return _dtypes.obj(self._col.dtype)

    @property
    def _dtype(self):  # the name, for the code here
        return self._col.dtype

    def __len__(self):
        return len(self._col)

    def tolist(self):
        return self._col.to_list(None)

    def _column(self):
        return self._col

    def _table(self):
        if self._labels is None:  # built on first use (docs/design/frame.md §6.2)
            self._labels = LabelTable(self._col)
        return self._labels

    def _slice(self, start, stop, step=1):
        if step == 1:
            return Index._new(self._col.slice(start, max(stop - start, 0)), self._name)
        return self._take(_core.arange(start, step, len(range(start, stop, step))))

    def _take(self, positions):
        return Index._new(self._col.take(positions), self._name)

    def _labels_at(self, positions):
        return _format.values_at(self._col, positions)

    def __repr__(self):
        return _format.index_repr(self)


def _labels_from(data, func):
    """The label Column of an Index made from `data` by the public function `func`."""
    from ._column import column_from, is_ndarray, is_sdarray

    if isinstance(data, _IndexBase):
        col = data._column()
    else:
        if not isinstance(data, (list, tuple, range)) and not is_ndarray(data) and not is_sdarray(data):
            raise TypeError(
                named(
                    func,
                    f"labels must be a list, tuple, range, numpy array, sd.ndarray or Index, not {type_name(data)}",
                )
            )
        data = list(data) if isinstance(data, range) else data
        # No labels to infer from: int64, since float64 (the empty list's dtype) cannot label.
        col = column_from(data, "int64" if len(data) == 0 else None, func)
    if col.null_count and col.null_count == len(col):  # all NA ([None]): the dtype says nothing
        raise ValueError(f"{func}(): an index cannot hold NA")
    if col.dtype not in ("int64", "string"):
        raise TypeError(f"{func}(): index labels must be int64 or string, not {col.dtype}")
    if col.null_count:
        raise ValueError(f"{func}(): an index cannot hold NA")
    return col


def ensure_index(labels, func, like=None):
    """An Index or RangeIndex for `labels` (kept as is when it already is one),
    made by the public function `func` (for messages). No labels at all take
    the dtype of the index `like`, whatever holds them (reindex([]) of string
    labels keeps string labels, as .loc[[]] does); an empty range stays a
    RangeIndex when `like` has int64 labels."""
    from ._column import is_ndarray, is_sdarray

    if isinstance(labels, _IndexBase):
        return labels
    if (
        like is not None
        and (
            isinstance(labels, (list, tuple, range))
            or ((is_ndarray(labels) or is_sdarray(labels)) and labels.ndim == 1)
        )
        and len(labels) == 0
        and not (isinstance(labels, range) and like._dtype == "int64")
    ):
        return Index._new(Column.from_list([], NA, like._dtype))
    if isinstance(labels, range):
        return RangeIndex._from_range(labels)
    return Index._new(_labels_from(labels, func))
