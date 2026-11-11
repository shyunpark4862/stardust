"""Reductions shared by Series and DataFrame (API.md §15).

The signatures are the same everywhere (API.md §11 C3): sum, mean, min and
max take (*, skipna=True); std and var add ddof=1; count takes nothing.
Each class supplies _reduce(op, skipna, ddof); the methods here only check
arguments and name the operation.
"""


def _check_ddof(ddof):
    if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof < 0:
        raise ValueError(f"ddof must be an int >= 0, got {ddof!r}")
    return ddof


class Reductions:
    __slots__ = ()

    def sum(self, *, skipna=True):
        """Sum of the values; 0 when there are none (API.md §2, §15)."""
        return self._reduce("sum", skipna, 0)

    def mean(self, *, skipna=True):
        return self._reduce("mean", skipna, 0)

    def min(self, *, skipna=True):
        return self._reduce("min", skipna, 0)

    def max(self, *, skipna=True):
        return self._reduce("max", skipna, 0)

    def std(self, *, skipna=True, ddof=1):
        """Standard deviation with divisor (count - ddof); NA when that is <= 0."""
        return self._reduce("std", skipna, _check_ddof(ddof))

    def var(self, *, skipna=True, ddof=1):
        """Variance with divisor (count - ddof); NA when that is <= 0."""
        return self._reduce("var", skipna, _check_ddof(ddof))

    def count(self):
        """How many values are not NA."""
        return self._reduce("count", True, 0)
