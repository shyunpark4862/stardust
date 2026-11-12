"""Reductions shared by Series and DataFrame (docs/api/frame.md §6).

The signatures are the same everywhere (docs/api/frame.md §6.1): sum, mean, min and
max take (*, skipna=True); std and var add correction=0, as in the array
area (API.md §1); count takes nothing. Each class supplies
_reduce(op, skipna, correction); the methods here only check arguments and
name the operation.
"""


from ._errors import check_bool, check_real


def _check_correction(correction, func):
    # the array area's rule: an int or a float (numpy numbers too), bool excluded
    return check_real(correction, "correction", func)


def _skipna(skipna, func):
    return check_bool(skipna, "skipna", func)


class Reductions:
    __slots__ = ()

    def sum(self, *, skipna=True):
        """Sum of the values; 0 when there are none (docs/api/frame.md §6.4)."""
        return self._reduce("sum", _skipna(skipna, "sum"), 0)

    def mean(self, *, skipna=True):
        return self._reduce("mean", _skipna(skipna, "mean"), 0)

    def min(self, *, skipna=True):
        return self._reduce("min", _skipna(skipna, "min"), 0)

    def max(self, *, skipna=True):
        return self._reduce("max", _skipna(skipna, "max"), 0)

    def std(self, *, skipna=True, correction=0):
        """Standard deviation with divisor (count - correction); NA when that is <= 0."""
        return self._reduce("std", _skipna(skipna, "std"), _check_correction(correction, "std"))

    def var(self, *, skipna=True, correction=0):
        """Variance with divisor (count - correction); NA when that is <= 0."""
        return self._reduce("var", _skipna(skipna, "var"), _check_correction(correction, "var"))

    def count(self):
        """How many values are not NA."""
        return self._reduce("count", True, 0)
