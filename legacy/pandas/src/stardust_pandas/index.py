"""Row labels: RangeIndex, and Index (labels held in an immutable int64 Column).

M2 has what filtering needs: storing labels, taking rows of them, equals and
repr. The label lookup engine, .loc and alignment arrive in M4.
"""

from . import _core, _format


class RangeIndex:
    """The labels start, start + step, ... below stop, stored as three integers (ARCHITECTURE.md §4)."""

    __slots__ = ("_range",)

    def __init__(self, start=0, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        self._range = range(start, stop, step)

    @classmethod
    def _from_range(cls, r):
        obj = cls.__new__(cls)
        obj._range = r
        return obj

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
        return "int64"

    def __len__(self):
        return len(self._range)

    def equals(self, other):
        """True when `other` holds the same labels in the same order."""
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return isinstance(other, Index) and self._column().equals(other._col)

    def to_list(self):
        return list(self._range)

    def _column(self):
        r = self._range
        return _core.arange(r.start, r.step, len(r))

    def _slice(self, start, stop):  # the labels at positions [start, stop)
        return RangeIndex._from_range(self._range[start:stop])

    def _take(self, positions):  # an Index of the labels at `positions` (an int64 Column)
        r = self._range
        if r.start == 0 and r.step == 1:
            return Index._new(positions)  # the positions are the labels
        labels = _core.binary_op("mul", positions, _core.Column.from_list([r.step], None, None), 2)
        return Index._new(_core.binary_op("add", labels, _core.Column.from_list([r.start], None, None), 2))

    def _labels_at(self, positions):  # Python ints, for repr
        return [self._range[p] for p in positions]

    def __repr__(self):
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"


class Index:
    """Labels in an immutable int64 Column without NA (API.md §6, §12)."""

    __slots__ = ("_col",)

    def __init__(self, data):
        from ._column import column_from

        col = column_from(data, "int64")  # only int64 labels until M4 (API.md §6)
        if col.null_count:
            raise ValueError("an Index cannot hold NA (API.md §6)")
        self._col = col

    @classmethod
    def _new(cls, col):
        obj = cls.__new__(cls)
        obj._col = col
        return obj

    @property
    def dtype(self):
        return self._col.dtype

    def __len__(self):
        return len(self._col)

    def equals(self, other):
        """True when `other` holds the same labels in the same order."""
        if isinstance(other, RangeIndex):
            return other.equals(self)
        return isinstance(other, Index) and self._col.equals(other._col)

    def to_list(self):
        return self._col.to_list(None)

    def _column(self):
        return self._col

    def _slice(self, start, stop):
        return Index._new(self._col.slice(start, stop - start))

    def _take(self, positions):
        return Index._new(self._col.take(positions))

    def _labels_at(self, positions):
        return _format.values_at(self._col, positions)

    def __repr__(self):
        return _format.index_repr(self)
