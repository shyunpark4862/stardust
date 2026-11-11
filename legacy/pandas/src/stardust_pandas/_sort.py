"""Sort orders for sort_values and sort_index (API.md §17)."""

from . import _core


def _per_key(value, n, what, check):
    """`value` for every key, or a list with one value per key."""
    values = list(value) if isinstance(value, (list, tuple)) else [value] * n
    if len(values) != n:
        raise ValueError(f"{what} has {len(values)} values for {n} sort keys")
    for v in values:
        check(v)
    return values


def _check_ascending(v):
    if not isinstance(v, bool):
        raise TypeError(f"ascending must be True or False, got {v!r}")


def _check_na_position(v):
    if v not in ("first", "last"):
        raise ValueError(f"na_position must be 'first' or 'last', got {v!r}")


def order(keys, ascending=True, na_position="last"):
    """Positions that sort rows by `keys` (Columns), first key first. Stable."""
    n = len(keys)
    asc = _per_key(ascending, n, "ascending", _check_ascending)
    nas = _per_key(na_position, n, "na_position", _check_na_position)
    return _core.argsort(keys, asc, [p == "last" for p in nas])
