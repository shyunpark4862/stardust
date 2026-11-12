"""Sort orders for sort_values and sort_index (docs/api/frame.md §8)."""

from . import _core


def _per_key(value, n, what, check):
    """`value` for every key, or a list with one value per key."""
    values = list(value) if isinstance(value, (list, tuple)) else [value] * n
    if len(values) != n:
        raise ValueError(f"{what} has {len(values)} values for {n} sort keys")
    for v in values:
        check(v)
    return values


def check_bool(what, v):
    """A bool option takes True or False only (API.md §2)."""
    if not isinstance(v, bool):
        raise TypeError(f"{what} must be True or False, got {v!r}")
    return v


def _check_na_position(v):
    if v not in ("first", "last"):
        raise ValueError(f"na_position must be 'first' or 'last', got {v!r}")


def order(keys, descending=False, na_position="last", stable=True):
    """Positions that sort rows by `keys` (Columns), first key first.

    Always stable: stable=False only allows an unstable sort, as in the array
    area, so it gives the same order (docs/api/frame.md §8.1)."""
    check_bool("stable", stable)
    n = len(keys)
    desc = _per_key(descending, n, "descending", lambda v: check_bool("descending", v))
    nas = _per_key(na_position, n, "na_position", _check_na_position)
    return _core.argsort(keys, [not d for d in desc], [p == "last" for p in nas])
