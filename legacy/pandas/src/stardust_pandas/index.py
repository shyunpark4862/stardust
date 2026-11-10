"""Row labels. M1 has only RangeIndex; Index (labels in a Column) arrives in M4."""


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
        return isinstance(other, RangeIndex) and self._range == other._range

    def to_list(self):
        return list(self._range)

    def _slice(self, start, stop):  # the labels at positions [start, stop)
        return RangeIndex._from_range(self._range[start:stop])

    def __repr__(self):
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"
