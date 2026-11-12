"""Reductions shared by Series and DataFrame (docs/api/frame.md §6).

The signatures are the same everywhere (docs/api/frame.md §6.1): sum, mean, min and
max take (*, skipna=True); std and var add correction=0, as in the array
area (API.md §1); count takes nothing. Each class supplies
_reduce(op, skipna, correction); the methods here only check arguments and
name the operation.
"""


def _check_correction(correction):
    # the array area's rule: an int or a float, bool excluded
    if isinstance(correction, bool) or not isinstance(correction, (int, float)):
        raise TypeError("correction must be an int or a float")
    return float(correction)


class Reductions:
    __slots__ = ()

    def sum(self, *, skipna=True):
        """Sum of the values; 0 when there are none (docs/api/frame.md §6.4)."""
        return self._reduce("sum", skipna, 0)

    def mean(self, *, skipna=True):
        return self._reduce("mean", skipna, 0)

    def min(self, *, skipna=True):
        return self._reduce("min", skipna, 0)

    def max(self, *, skipna=True):
        return self._reduce("max", skipna, 0)

    def std(self, *, skipna=True, correction=0):
        """Standard deviation with divisor (count - correction); NA when that is <= 0."""
        return self._reduce("std", skipna, _check_correction(correction))

    def var(self, *, skipna=True, correction=0):
        """Variance with divisor (count - correction); NA when that is <= 0."""
        return self._reduce("var", skipna, _check_correction(correction))

    def count(self):
        """How many values are not NA."""
        return self._reduce("count", True, 0)
