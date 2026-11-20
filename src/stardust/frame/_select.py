"""Turning .loc / .iloc / [] keys into row positions, and aligning labels
(docs/api/frame.md §5, docs/design/frame.md §6.3, §6.4).

A row selection is one of three shapes, so callers can keep zero-copy
slices and scalar results apart from gathers:
  ("one", position)                 a single row: a scalar for a Series
  ("slice", start, stop, step)      positions range(start, stop, step)
  ("take", positions)               an int64 Column of positions
  ("mask", mask)                    a bool Column aligned with the rows (columns are
                                    compacted directly, docs/design/frame.md §7.3)
"""

import sys

from .. import _core as _sd_core
from . import _core
from ._errors import named, type_name
from ._index import Index, _IndexBase, mixed_labels

Column = _core.Column


def _is_int(x):
    np = sys.modules.get("numpy")
    return (isinstance(x, int) and not isinstance(x, bool)) or (np is not None and isinstance(x, np.integer))


def is_array(key):
    """A numpy array or an sd.ndarray: keys that select as lists do (DESIGN.md §10.1 #7)."""
    from ._column import is_ndarray, is_sdarray

    return is_ndarray(key) or is_sdarray(key)


def array_items(key):
    """The elements of a list-like key. A 1-D sd.ndarray gives Python scalars
    (as its tolist()); a numpy array numpy scalars, which select the same."""
    from ._column import is_sdarray

    return key.tolist() if is_sdarray(key) and key.ndim == 1 else list(key)


def _is_listlike(key):
    return isinstance(key, (list, _IndexBase)) or is_array(key)


# --- rows by label (.loc) and by position (.iloc) ---


def by_label(index, key):
    from ._series import Series

    if isinstance(key, Series):
        return ("mask", aligned_mask(key, index))
    if isinstance(key, slice):
        return ("slice", *label_slice(index, key))
    if _is_listlike(key):
        return ("take", index._positions_of(array_items(key) if is_array(key) else key))
    if isinstance(key, tuple):
        raise TypeError("a tuple is not a row label (there is no MultiIndex)")
    return ("one", index._position_of(key))


def label_slice(index, key):
    """Positions of a label slice a:b: both ends included, each end present
    and unique; no search over sorted labels (docs/design/frame.md §6.4)."""
    n = len(index)
    step = 1 if key.step is None else key.step
    if not _is_int(step) or step == 0:
        raise ValueError("a slice step must be a non-zero int")
    start = index._position_of(key.start) if key.start is not None else (0 if step > 0 else n - 1)
    stop_pos = index._position_of(key.stop) if key.stop is not None else (n - 1 if step > 0 else 0)
    stop = stop_pos + 1 if step > 0 else stop_pos - 1  # the stop label is included
    if stop < 0:
        stop = None
    return slice(start, stop, step).indices(n)


def by_position(n, key, what="rows"):
    """A selection of `n` positions (`what`: rows, or columns for df.iloc[:, j]).
    A key of the wrong kind is an IndexError, as an array index of the wrong
    kind (API.md §2)."""
    if _is_int(key):
        pos = int(key) + n if key < 0 else int(key)
        if not 0 <= pos < n:
            raise IndexError(f"position {key} is out of range for {n} {what}")
        return ("one", pos)
    if isinstance(key, slice):
        for end in (key.start, key.stop, key.step):
            if end is not None and not hasattr(type(end), "__index__"):
                raise IndexError(
                    f"cannot select {what} by position with a slice of {type_name(end)}; "
                    "the ends and the step must be ints or None"
                )
        return ("slice", *key.indices(n))
    if isinstance(key, list) or is_array(key):
        col = _int_positions(key)
        if col is not None:  # all ints, read in one pass; the range checked in one kernel
            wrapped, bad = _core.wrap_positions(col, n)
            if bad < 0:
                return ("take", wrapped)
            raise IndexError(f"position {col.slice(bad, 1).to_list(None)[0]} is out of range for {n} {what}")
        positions = array_items(key)  # other keys: element by element, for the message
        for p in positions:
            if not _is_int(p):
                raise IndexError(f"positions must be ints, not {type_name(p)}")
            if not -n <= p < n:
                raise IndexError(f"position {p} is out of range for {n} {what}")
        return ("take", Column.from_list([int(p) + n if p < 0 else int(p) for p in positions], None, "int64"))
    raise IndexError(f"cannot select {what} by position with {type_name(key)}; use an int, a slice or a list of ints")


def _int_positions(key):
    """The int64 Column of a position key whose elements are all ints and can
    be read in one pass: a list of Python ints, a 1-D numpy integer array
    (not uint64) or a 1-D int64 sd.ndarray. None for any other key, which
    by_position then checks element by element."""
    from ._column import column_from, is_ndarray, is_sdarray

    if isinstance(key, list):
        if not set(map(type, key)) <= {int}:  # the types, not the values: one pass in C
            return None
        try:
            return Column.from_list(key, None, "int64")
        except OverflowError:  # an int outside int64: out of range, said element by element
            return None
    if key.ndim != 1:
        return None
    if is_ndarray(key):
        if key.dtype.kind not in "iu" or (key.dtype.kind == "u" and key.dtype.itemsize == 8):
            return None
        return column_from(key)
    if is_sdarray(key) and key.dtype == _sd_core.int64:
        return Column.from_array(key)
    return None


def positions_column(sel):
    """The positions of any selection as an int64 Column."""
    kind = sel[0]
    if kind == "one":
        return Column.from_list([sel[1]], None, "int64")
    if kind == "slice":
        start, stop, step = sel[1:]
        return _core.arange(start, step, len(range(start, stop, step)))
    if kind == "mask":
        return sel[1].nonzero()
    return sel[1]


def select_column(col, sel):
    """The rows of `col` picked by a non-"one" selection (slices with step 1 share the Buffers)."""
    if sel[0] == "slice" and sel[3] == 1:
        start, stop = sel[1], max(sel[2], sel[1])
        return col.slice(start, stop - start)
    if sel[0] == "mask":
        return col.filter(sel[1])
    return col.take(positions_column(sel))


def select_index(index, sel):
    if sel[0] == "slice":
        start, stop, step = sel[1:]
        return index._slice(start, stop, step)
    return index._take(positions_column(sel))


# --- masks and alignment ---


def mask_positions(mask, index):
    """Positions of the True rows of a bool Series mask (see aligned_mask)."""
    return aligned_mask(mask, index).nonzero()


def aligned_mask(mask, index):
    """The bool Column of a Series mask, in the order of `index` (NA counts
    as False, docs/api/frame.md §5.5). A mask with other labels is matched to `index` by
    label; the two must hold the same set of labels (docs/api/frame.md §5.5)."""
    if mask._dtype != "bool":
        raise TypeError(f"a mask must be a bool Series, got dtype {mask._dtype}")
    col = mask._col
    if not same_index(index, mask._index):
        # Same length, unique labels here, and every label found in the mask
        # (unique there too: _get_indexer checks) is the same set. With
        # duplicates here, the same length means the mask has a label this
        # index lacks (docs/design/frame.md §6.3).
        if len(mask) != len(index) or not index.is_unique:
            raise ValueError("the mask's labels must be the index's labels")
        positions = mask._index._get_indexer(index)  # needs unique mask labels
        if len(_core.binary_op("eq", positions, Column.from_list([-1], None, None), 2).nonzero()):
            raise ValueError("the mask's labels must be the index's labels")
        col = col.take(positions)
    return col


def same_index(a, b):
    return a is b or a.equals(b)


def merged_name(a, b):
    """The name of a result made from two named things: kept when equal (docs/api/frame.md §3.1, §5.1)."""
    return a if a == b else None


def align(a, b, func=None):
    """(union index, positions into a, positions into b) for labels a and b.
    Positions are None when that side needs no reordering; -1 means NA.
    The union keeps a's labels in order, then b's other labels (docs/api/frame.md §1.7).
    `func`: the public function, or None for an operator."""
    name = merged_name(a.name, b.name)
    if same_index(a, b):
        return a._with_name(name), None, None
    if a._dtype != b._dtype:
        raise TypeError(named(func, mixed_labels(a._dtype, b._dtype)))
    if not a.is_unique or not b.is_unique:
        raise ValueError(named(func, "cannot align indexes with duplicate labels"))
    a_labels, b_labels = a._column(), b._column()
    only_b = _core.binary_op("eq", a._get_indexer(b), Column.from_list([-1], None, None), 2).nonzero()
    union = Index._new(_core.concat([a_labels, b_labels.take(only_b)]), name)
    a_pos = None if len(only_b) == 0 else _core.concat([_core.arange(0, 1, len(a)), _minus_ones(len(only_b))])
    b_pos = b._get_indexer(union)
    return union, a_pos, b_pos


def _minus_ones(n):
    return _core.full(Column.from_list([-1], None, "int64"), n)


def reindexed(col, index, target, func=None):
    """`col` (labelled by `index`) rearranged to the labels of `target`; absent labels are NA."""
    if same_index(index, target):
        return col
    return col.take(index._get_indexer(target, func))
