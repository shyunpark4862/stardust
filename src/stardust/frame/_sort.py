"""Sort orders for sort_values and sort_index (docs/api/frame.md §8)."""

from . import _core
from ._errors import check_bool, check_choice, named


def _per_key(value, n, what, check, func):
    """`value` for every key, or a list with one value per key."""
    values = list(value) if isinstance(value, (list, tuple)) else [value] * n
    if len(values) != n:
        raise ValueError(named(func, f"'{what}' has {len(values)} values for {n} sort keys"))
    for v in values:
        check(v)
    return values


def _check_na_position(v, func):
    check_choice(v, "na_position", ("first", "last"), func)


def order(keys, descending=False, na_position="last", stable=True, func=None):
    """Positions that sort rows by `keys` (Columns), first key first. `func`
    names the public function in messages.

    Always stable: stable=False only allows an unstable sort, as in the array
    area, so it gives the same order (docs/api/frame.md §8.1)."""
    check_bool(stable, "stable", func)
    n = len(keys)
    desc = _per_key(descending, n, "descending", lambda v: check_bool(v, "descending", func), func)
    nas = _per_key(na_position, n, "na_position", lambda v: _check_na_position(v, func), func)
    return _core.argsort(keys, [not d for d in desc], [p == "last" for p in nas])
