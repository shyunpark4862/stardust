"""merge (a hash join) and concat (docs/api/frame.md §9)."""

from . import _column, _core, _select, _sort
from ._errors import name_list
from ._index import Index, RangeIndex, _IndexBase

Column = _core.Column

HOWS = ("inner", "left", "right", "outer")


def _names(value, what):  # for merge()
    names = name_list(value, what, "merge")
    if not names:
        raise ValueError(f"merge(): '{what}' must name at least one column")
    for k in names:
        if not isinstance(k, str):
            raise TypeError(f"merge(): '{what}' takes column names (str), got {type(k).__name__}")
    return names


def merge(left, right, /, *, how="inner", on=None, left_on=None, right_on=None, sort=False, suffixes=("_x", "_y")):
    """Join two DataFrames on key columns with a hash join (docs/api/frame.md §9.1).

    Keys are explicit: on= (the same names on both sides) or left_on= and
    right_on=. Rows come in left order, each with its matches in right
    order; outer appends unmatched right rows; right is the mirror image
    (docs/api/frame.md §9.1). NA keys match nothing; NaN matches NaN."""
    from ._frame import DataFrame

    if not isinstance(left, DataFrame) or not isinstance(right, DataFrame):
        raise TypeError("merge(): 'left' and 'right' must be DataFrames")
    if how not in HOWS:
        raise ValueError(f"merge(): 'how' must be one of {', '.join(HOWS)}, got {how!r}")
    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("merge(): pass on= or left_on=/right_on=, not both")
        lkeys = rkeys = _names(on, "on")
    elif left_on is not None and right_on is not None:
        lkeys, rkeys = _names(left_on, "left_on"), _names(right_on, "right_on")
        if len(lkeys) != len(rkeys):
            raise ValueError(f"merge(): {len(lkeys)} left keys for {len(rkeys)} right keys")
    else:
        raise ValueError("merge(): the keys are required: on= or left_on= and right_on=")
    if not (isinstance(suffixes, tuple) and len(suffixes) == 2 and all(isinstance(x, str) for x in suffixes)):
        raise TypeError("merge(): 'suffixes' must be a tuple of two str")

    lcols = [left._column(k, "merge") for k in lkeys]
    rcols = [right._column(k, "merge") for k in rkeys]
    lpos, rpos = _core.join_indices(lcols, rcols, how, func="merge")

    merged = set(lkeys) if on is not None else set()  # one key column per name with on=
    overlap = (set(left._cols) & set(right._cols)) - merged
    out = {}

    def put(name, col):
        if name in out:
            raise ValueError(f"merge(): result column {name!r} appears twice; choose other suffixes")
        out[name] = col

    no_left = _core.binary_op("eq", lpos, Column.from_list([-1], None, None), 2).nonzero()
    for k, c in left._cols.items():
        col = c.take(lpos)
        if k in merged and len(no_left):  # rows that only the right side has take the right key value
            col = col.share()
            col.set_rows(no_left, right._cols[k].take(rpos).take(no_left))
        put(k + suffixes[0] if k in overlap else k, col)
    for k, c in right._cols.items():
        if k in merged:
            continue
        put(k + suffixes[1] if k in overlap else k, c.take(rpos))

    result = DataFrame._new(out, RangeIndex(len(lpos)))
    if sort:
        keys = [out[k] for k in lkeys] if on is not None else _sort_keys(out, lkeys, rkeys, overlap, suffixes, no_left)
        result = result._take(_sort.order(keys), ignore_index=True)
    return result


def _sort_keys(out, lkeys, rkeys, overlap, suffixes, no_left):
    """With left_on/right_on: the left key, or the right key on rows with no left row."""
    keys = []
    for lk, rk in zip(lkeys, rkeys):
        col = out[lk + suffixes[0] if lk in overlap else lk].share()
        if len(no_left):
            col.set_rows(no_left, out[rk + suffixes[1] if rk in overlap else rk].take(no_left))
        keys.append(col)
    return keys


def concat(objs, /, *, join="outer", ignore_index=False):
    """DataFrames (or Series) one after another, rows only (docs/api/frame.md §9.2)."""
    from ._frame import DataFrame
    from ._series import Series

    try:
        objs = list(objs)
    except TypeError:
        raise TypeError(f"concat(): 'objs' must be a list of DataFrames or Series, not {type(objs).__name__}") from None
    if not objs:
        raise ValueError("concat(): 'objs' must hold at least one object")
    if join not in ("outer", "inner"):
        raise ValueError(f"concat(): 'join' must be 'outer' or 'inner', got {join!r}")
    index = None if ignore_index else _concat_labels([o.index for o in objs])
    n = sum(len(o) for o in objs)
    if all(isinstance(o, Series) for o in objs):
        dtypes = {o._dtype for o in objs}
        if len(dtypes) != 1:
            raise TypeError(f"concat(): Series of dtypes {sorted(dtypes)} cannot be joined")
        names = {o.name for o in objs}
        name = names.pop() if len(names) == 1 else None
        return Series._new(_core.concat([o._col for o in objs]), index or RangeIndex(n), name)
    if not all(isinstance(o, DataFrame) for o in objs):
        raise TypeError("concat(): 'objs' must be all DataFrames or all Series, not a mix")
    if join == "outer":
        names = list(dict.fromkeys(k for o in objs for k in o._cols))
    else:
        names = [k for k in objs[0]._cols if all(k in o._cols for o in objs)]
    cols = {}
    for k in names:
        dtypes = {o._cols[k].dtype for o in objs if k in o._cols}
        if len(dtypes) != 1:
            raise TypeError(f"concat(): column {k!r} has dtypes {sorted(dtypes)}; make them one with astype")
        (dtype,) = dtypes
        parts = [o._cols[k] if k in o._cols else _column.full_na(len(o), dtype) for o in objs]
        cols[k] = _core.concat(parts)
    return DataFrame._new(cols, index or RangeIndex(n))


def _concat_labels(indexes):
    dtypes = {i._dtype for i in indexes}
    if len(dtypes) != 1:
        raise TypeError("concat(): cannot join int labels with string labels")
    names = {i.name for i in indexes}
    return Index._new(_core.concat([i._column() for i in indexes]), names.pop() if len(names) == 1 else None)
