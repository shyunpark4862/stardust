"""unique, duplicated, value_counts on top of factorize (docs/api/frame.md §4).

factorize gives every row the code of its key (0, 1, ... in order of first
appearance) and the first row of each code. Everything here is column
operations on those codes, so the rules for what counts as the same key
live in one place: the C++ factorize.
"""

from . import _core

Column = _core.Column

KEEP = ("first", "last", "none")


def codes(cols):
    """(codes, first, ngroups) for rows of `cols` taken together."""
    return _core.factorize_rows(cols)


def duplicated(cols, keep):
    """bool Column: True for rows whose key already appeared (keep="first"),
    appears again later (keep="last"), or appears more than once ("none")."""
    if keep not in KEEP:
        raise ValueError(f"keep must be 'first', 'last' or 'none', got {keep!r}")
    n = len(cols[0])
    row_codes, first, ngroups = codes(cols)
    if keep == "none":
        counts, _ = _core.group_counts(row_codes, ngroups)
        return _core.binary_op("gt", counts.take(row_codes), Column.from_list([1], None, None), 2)
    if keep == "last":
        _, first = _core.group_counts(row_codes, ngroups)  # the last row of each code
    positions = _core.arange(0, 1, n)
    return _core.binary_op("ne", positions, first.take(row_codes), 0)


def value_counts(cols, normalize, sort, ascending):
    """(key columns, count column): one row per key."""
    row_codes, first, ngroups = codes(cols)
    counts, _ = _core.group_counts(row_codes, ngroups)
    order = _core.stable_order(counts, ascending) if sort else _core.arange(0, 1, ngroups)
    keys = [c.take(first.take(order)) for c in cols]
    counts = counts.take(order)
    if normalize:
        total = Column.from_list([len(cols[0])], None, None)
        counts = _core.binary_op("truediv", counts, total, 2)
    return keys, counts
