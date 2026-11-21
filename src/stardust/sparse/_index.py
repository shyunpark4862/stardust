"""Indexing and assignment of sd.sparse, and the diagonal (docs/api/sparse.md §4.1,
docs/design/sparse.md §3.3; milestone SP3).

A key is read by the array's rules (docs/api/array.md §5): ints (negative from the end), slices,
`...`, None, integer arrays and boolean masks (any of sd.ndarray, a nested list or tuple, a
buffer such as a numpy array), with the array's messages. The result has at most 2 dimensions.

  - Every axis an int: a Python scalar (the stored value, a COO array's repeats summed in stored
    order, or a zero of the dtype).
  - Otherwise a sparse array. When it has as many dimensions as A it is in A's format (a CSR
    stays CSR); when the number of dimensions changes (A[i] of a 2-D array, a None on a 1-D one,
    picking elements A[rows, cols]) it is a COO array, as scipy's. [scipy differs: its CSR and CSC
    give a dense numpy array for A[rows, cols] and for a 2-D boolean mask.]
  - Arrays on both axes of a 2-D array pick elements (they broadcast together, as the array's);
    an array on one axis selects whole lines (outer). An integer next to an array counts as an
    array (the array's rule), and advanced indices separated by a slice, None or `...` are an
    IndexError, as for the array.
  - A COO array keeps its repeated positions and stored order in the result (scipy's).

Assignment A[key] = value sets every indexed position (a position named twice takes the last
value, as the array's scatter). The value is a Python scalar for a single element (the array's
rule), otherwise anything the array's assignment takes (broadcast strictly to the indexed shape)
or a sparse array. Values convert by the array's rules (an implicit downcast is a TypeError).
What is stored follows scipy per format:
  - CSR, CSC: a stored position changes in place; a new one is inserted (the arrays are rebuilt,
    O(nnz)), zeros included (an assigned 0 is a stored 0). A sparse value: the stored positions
    of the indexed region become 0 and the value's stored entries are set.
  - DOK: a zero erases the key (the region of a sparse value is set from its dense values).
  - COO: the entries at the indexed positions go (repeats too) and the nonzero values are
    appended in row-major order (a single element and a sparse value's stored entries are
    appended even when 0).
"""

import math
import operator

from .. import _core as _sd
from . import _common as C
from ._common import _core

_INVALID = (
    "only integers, slices (`:`), ellipsis (`...`), newaxis (`None`) and integer or boolean arrays are valid indices"
)
_SEPARATED = (
    "advanced indices (arrays, and ints next to them) separated by a slice, newaxis or ellipsis are not "
    "supported; index in two steps, e.g. x[0][:, idx]"
)


def _classes():
    from ._compressed import csc_array, csr_array
    from ._coo import coo_array
    from ._dok import dok_array

    return coo_array, csr_array, csc_array, dok_array


def _is_bool_scalar(v):
    return isinstance(v, bool) or (type(v).__name__ == "bool" and type(v).__module__ == "numpy")


def _readonly(x):
    x._set_readonly()
    return x


def _flat(x):
    """x as a contiguous 1-D int64 array (a copy when needed)."""
    if not C.contiguous(x):
        x = x.copy()
    return x.reshape((x.size,)) if x.ndim != 1 else x


# ---- the key ----


class _Key:
    """A parsed key: per axis ('int', i) | ('slice', start, step, count) | ('array', int64 array
    of any shape, normalized); `shape` the shape of the result (or of the indexed region);
    `scalar` whether every axis is an int (and no None)."""

    __slots__ = ("axes", "shape", "scalar", "inner")


def _as_index_array(item):
    from ._base import sparray

    if isinstance(item, sparray) or C.is_scipy_sparse(item):
        raise IndexError("a sparse array is not a valid index; use a dense integer or boolean array")
    if not isinstance(item, (_sd.ndarray, list, tuple, range)) and not C._has_buffer(item):
        raise IndexError(_INVALID)
    if isinstance(item, (list, tuple)) and len(item) == 0:
        return _sd.empty(0, dtype=_sd.int64)
    try:
        x = _sd.asarray(item)
    except ValueError as e:
        raise IndexError(_INVALID) from e
    if x.dtype not in (_sd.int64, _sd.bool):
        if x.size == 0:
            return _sd.empty(x.shape, dtype=_sd.int64)
        raise IndexError(f"arrays used as indices must be of integer or boolean type, not {x.dtype}")
    return x


def parse(key, shape):
    nd = len(shape)
    items = list(key) if type(key) is tuple else [key]
    parsed = []
    ellipsis = None
    used = 0
    for it in items:
        if it is None:
            parsed.append(("none",))
        elif it is Ellipsis:
            if ellipsis is not None:
                raise IndexError("an index can only have a single ellipsis ('...')")
            ellipsis = len(parsed)
        elif isinstance(it, slice):
            parsed.append(("slice", it))
            used += 1
        elif _is_bool_scalar(it):
            raise IndexError("a bool is not a valid index; use an int, or a boolean array")
        elif C.is_int(it):
            parsed.append(("int", operator.index(it)))
            used += 1
        else:
            x = _as_index_array(it)
            if x.dtype == _sd.bool:
                parsed.append(("mask", x))
                used += x.ndim
            else:
                parsed.append(("array", x))
                used += 1
    if used > nd:
        raise IndexError(f"too many indices for array: array is {nd}-dimensional, but {used} were indexed")
    fill = [("slice", slice(None))] * (nd - used)
    if ellipsis is None:
        parsed += fill
    else:
        parsed[ellipsis:ellipsis] = fill

    axes = [None] * nd
    out = []  # ('dim', n) | ('new',) | ('adv',)
    a = 0
    for it in parsed:
        kind = it[0]
        if kind == "none":
            out.append(("new",))
        elif kind == "slice":
            start, stop, step = it[1].indices(shape[a])
            count = len(range(start, stop, step))
            axes[a] = ("slice", start, step, count)
            out.append(("dim", count))
            a += 1
        elif kind == "int":
            n = shape[a]
            v = it[1]
            if not -n <= v < n:
                raise IndexError(f"index {v} is out of bounds for axis {a} with size {n}")
            axes[a] = ("int", v + n if v < 0 else v)
            out.append(("adv",))
            a += 1
        elif kind == "array":
            x = it[1]
            flat = _core._normalize_index(_flat(x), shape[a], a)
            axes[a] = ("array", flat.reshape(tuple(x.shape)) if x.ndim != 1 else flat)
            out.append(("adv",))
            a += 1
        else:  # a mask over x.ndim axes
            x = it[1]
            for d in range(x.ndim):
                if x.shape[d] != shape[a + d]:
                    raise IndexError(
                        f"boolean index did not match indexed array along axis {a + d}; size of axis is "
                        f"{shape[a + d]} but size of corresponding boolean axis is {x.shape[d]}"
                    )
            for d, idx in enumerate(_sd.nonzero(x)):
                axes[a + d] = ("array", _readonly(idx))
                out.append(("adv",))
            a += x.ndim

    k = _Key()
    k.axes = axes
    arrays = [s[1] for s in axes if s[0] == "array"]
    if not arrays:
        k.shape = tuple(o[1] if o[0] == "dim" else 1 for o in out if o[0] != "adv")
        k.scalar = not k.shape and all(o[0] == "adv" for o in out)
        k.inner = False
        return k
    pos = [i for i, o in enumerate(out) if o[0] == "adv"]
    if pos[-1] - pos[0] + 1 != len(pos):
        raise IndexError(_SEPARATED)
    bshape = _broadcast_shape([tuple(x.shape) for x in arrays])
    shape_out = []
    for i, o in enumerate(out):
        if o[0] == "adv":
            if i == pos[0]:
                shape_out.extend(bshape)
        else:
            shape_out.append(o[1] if o[0] == "dim" else 1)
    k.shape = tuple(shape_out)
    k.scalar = False
    k.inner = nd == 2 and axes[0][0] == "array" and axes[1][0] == "array"
    return k


def _broadcast_shape(shapes):
    nd = max(len(s) for s in shapes)
    out = []
    for i in range(nd):
        dims = {s[len(s) - nd + i] for s in shapes if len(s) - nd + i >= 0} - {1}
        if len(dims) > 1:
            raise IndexError(
                "shape mismatch: indexing arrays could not be broadcast together with shapes "
                + " ".join(str(s) for s in shapes)
            )
        out.append(dims.pop() if dims else 1)
    return tuple(out)


def _check_result_ndim(shape):
    if len(shape) > 2:
        raise IndexError(f"this index gives a result of {len(shape)} dimensions; a sparse array has 1 or 2")


# ---- one axis as a kernel selection ----


def _spec(sel, n):
    """None (all), (start, step, count), or a contiguous 1-D int64 array."""
    kind = sel[0]
    if kind == "int":
        return (sel[1], 1, 1)
    if kind == "slice":
        _, start, step, count = sel
        if start == 0 and step == 1 and count == n:
            return None
        return (start, step, count)
    return _flat(sel[1])


def _spec_len(spec, n):
    if spec is None:
        return n
    if isinstance(spec, tuple):
        return spec[2]
    return spec.shape[0]


def _spec_positions(spec, n):
    """The source indices of a selection as an int64 array."""
    if spec is None:
        return _sd.arange(n, dtype=_sd.int64)
    if isinstance(spec, tuple):
        start, step, count = spec
        return (
            _sd.arange(start, start + step * count, step, dtype=_sd.int64) if count else _sd.empty(0, dtype=_sd.int64)
        )
    return spec


# ---- reading ----


def scalar_value(A, i, j):
    """A[i, j] (j for 1-D; i, j already in range) as a Python scalar."""
    fmt = A.format
    if fmt == "dok":
        return A._map.get((i, j) if A.ndim == 2 else j)
    if fmt in ("csr", "csc"):
        if A.ndim == 1:
            return _core._compressed_value(A._indptr, A._indices, A._data, 0, j)
        line, pos = (i, j) if A._by_rows else (j, i)
        return _core._compressed_value(A._indptr, A._indices, A._data, line, pos)
    major, minor = A._split()
    v = _core._coo_value(major, minor, A._data, i, j).tolist()
    return v[0] if v else _sd.zeros(1, dtype=A.dtype).tolist()[0]


def select(A, row_spec, col_spec):
    """The outer selection: rows x cols of a 2-D array (row_spec unused for 1-D) in A's format."""
    coo_array, csr_array, csc_array, dok_array = _classes()
    fmt = A.format
    if A.ndim == 1:
        n = A.shape[0]
        shape = (_spec_len(col_spec, n),)
    else:
        shape = (_spec_len(row_spec, A.shape[0]), _spec_len(col_spec, A.shape[1]))
    if fmt in ("csr", "csc"):
        if A.ndim == 1:
            parts = _core._select_compressed(A._indptr, A._indices, A._data, A.shape[0], None, col_spec)
            return csr_array._new(shape, *parts)
        if A._by_rows:
            parts = _core._select_compressed(A._indptr, A._indices, A._data, A.shape[1], row_spec, col_spec)
            return csr_array._new(shape, *parts)
        parts = _core._select_compressed(A._indptr, A._indices, A._data, A.shape[0], col_spec, row_spec)
        return csc_array._new(shape, *parts)
    coo = A if fmt == "coo" else A._tocoo(None, "index")
    major, minor = coo._split()
    rows, cols, data = _core._select_coo(major, minor, coo._data, A.shape, row_spec, col_spec)
    coords = (rows, cols) if rows is not None else (cols,)
    out = coo_array._new(shape, coords, data, True if fmt == "dok" else None)
    return out._todok(None, "index") if fmt == "dok" else out


def pick(A, rows, cols):
    """The elements at the pairs (rows[k], cols[k]) as a 1-D COO array of len(rows)."""
    coo_array = _classes()[0]
    K = rows.shape[0]
    fmt = A.format
    if fmt in ("csr", "csc"):
        line, pos = (rows, cols) if A._by_rows else (cols, rows)
        where, vals = _core._pick_compressed(A._indptr, A._indices, A._data, line, pos)
        return coo_array._new((K,), (where,), vals, True)
    coo = A if fmt == "coo" else A._tocoo(None, "index")
    major, minor = coo._split()
    where, vals = _core._pick_coo(major, minor, coo._data, rows, cols)
    return coo_array._new((K,), (where,), vals, True if fmt == "dok" else None)


def _finish(A, res, shape):
    """res reshaped to the result shape: A's format when the number of dimensions is A's."""
    if res.shape == shape:
        return res
    out = res.reshape(shape)
    if len(shape) == A.ndim and A.format != "coo":
        out = out._convert(A.format, None, "index")
    return out


def _outer_pair(axes):
    """(rows, cols) of an element pick that is an outer product: rows (n, 1) and cols (m,) or
    (1, m); else None."""
    r, c = axes[0][1], axes[1][1]
    if r.ndim == 2 and r.shape[1] == 1 and (c.ndim == 1 or (c.ndim == 2 and c.shape[0] == 1)):
        return r, c
    return None


def getitem(A, key):
    k = parse(key, A.shape)
    if k.scalar:
        if A.ndim == 2:
            return scalar_value(A, k.axes[0][1], k.axes[1][1])
        return scalar_value(A, 0, k.axes[0][1])
    _check_result_ndim(k.shape)
    if k.inner:
        pair = _outer_pair(k.axes)
        if pair is not None:
            r, c = pair
            res = select(A, _flat(r), _flat(c))
            return _finish(A, res, k.shape)
        rows, cols = _sd.broadcast_arrays(k.axes[0][1], k.axes[1][1])
        res = pick(A, _flat(rows), _flat(cols))
        return _finish(A, res, k.shape)
    if A.ndim == 1:
        res = select(A, None, _spec(k.axes[0], A.shape[0]))
    else:
        res = select(A, _spec(k.axes[0], A.shape[0]), _spec(k.axes[1], A.shape[1]))
    return _finish(A, res, k.shape)


# ---- assignment ----


def _region(A, k):
    """(rows or None, cols): the indexed positions as flat int64 arrays in C order of k.shape."""
    if k.inner:
        rows, cols = _sd.broadcast_arrays(k.axes[0][1], k.axes[1][1])
        return _flat(rows), _flat(cols)
    if A.ndim == 1:
        return None, _spec_positions(_spec(k.axes[0], A.shape[0]), A.shape[0])
    r = _spec_positions(_spec(k.axes[0], A.shape[0]), A.shape[0])
    c = _spec_positions(_spec(k.axes[1], A.shape[1]), A.shape[1])
    nr, nc = r.shape[0], c.shape[0]
    rows = _flat(_sd.broadcast_to(r.reshape((nr, 1)), (nr, nc)))
    cols = _flat(_sd.broadcast_to(c.reshape((1, nc)), (nr, nc)))
    return rows, cols


def _dense_values(A, value, shape):
    """value as a dense array of the indexed shape in A's dtype, by the array's assignment rules."""
    out = _sd.empty(shape if shape else (1,), dtype=A.dtype)
    if shape:
        out[...] = value
    else:
        out[0] = value
    return _flat(out) if len(shape) != 1 else out


def setitem(A, key, value):
    from ._base import sparray

    k = parse(key, A.shape)
    if k.scalar and A.format == "dok":
        A._map.set(tuple(s[1] for s in k.axes) if A.ndim == 2 else k.axes[0][1], value)
        return
    if isinstance(value, sparray):
        _set_sparse(A, k, value)
        return
    if k.scalar:
        vals = _dense_values(A, value, ())
        if A.ndim == 2:
            rows = _sd.asarray([k.axes[0][1]], dtype=_sd.int64)
            cols = _sd.asarray([k.axes[1][1]], dtype=_sd.int64)
        else:
            rows, cols = None, _sd.asarray([k.axes[0][1]], dtype=_sd.int64)
        _apply(A, rows, cols, vals, keep_zeros=True)
        return
    vals = _dense_values(A, value, k.shape)
    if vals.shape[0] == 0:
        return
    rows, cols = _region(A, k)
    _apply(A, rows, cols, vals, keep_zeros=False)


def _set_sparse(A, k, x):
    shape = k.shape
    xs = tuple(x.shape)
    if len(xs) > len(shape) or any(d not in (1, s) for d, s in zip(reversed(xs), reversed(shape))):
        raise ValueError(f"cannot broadcast a value of shape {xs} to the indexed shape {shape}")
    probe = _sd.empty(1, dtype=A.dtype)
    probe[...] = _sd.zeros(1, dtype=x.dtype)  # the array's rule: an implicit downcast is a TypeError
    if math.prod(shape) == 0:
        return
    rows, cols = _region(A, k)
    dense = _flat(_sd.broadcast_to(_sd.astype(x.toarray(), A.dtype), shape))
    stored = _flat(_sd.broadcast_to(x._with_data(_sd.ones(x.nnz, dtype=_sd.bool)).toarray(), shape))
    fmt = A.format
    if fmt == "dok":
        A._map.set_many(rows, cols, dense)
        return
    sr = rows[stored] if rows is not None else None
    sc = cols[stored]
    sv = _sd.astype(dense[stored], A.dtype)
    if fmt in ("csr", "csc"):
        _assign(A, rows, cols, _sd.zeros(cols.shape[0], dtype=A.dtype), insert=False)
        _assign(A, sr, sc, sv, insert=True)
        return
    _coo_assign(A, rows, cols, sr, sc, sv, keep_zeros=True)


def _apply(A, rows, cols, vals, keep_zeros):
    fmt = A.format
    if fmt in ("csr", "csc"):
        _assign(A, rows, cols, vals, insert=True)
    elif fmt == "dok":
        A._map.set_many(rows, cols, vals)
    else:
        _coo_assign(A, rows, cols, rows, cols, vals, keep_zeros=keep_zeros)


def _assign(A, rows, cols, vals, insert):
    if A.ndim == 1:
        line, pos, n_minor = None, cols, A.shape[0]
    elif A._by_rows:
        line, pos, n_minor = rows, cols, A.shape[1]
    else:
        line, pos, n_minor = cols, rows, A.shape[0]
    parts = _core._assign_compressed(A._indptr, A._indices, A._data, n_minor, line, pos, vals, insert=insert)
    if parts is not None:
        A._indptr, A._indices, A._data = parts
        A._long = None


def _coo_assign(A, rows, cols, new_rows, new_cols, vals, keep_zeros):
    """The entries of A at (rows, cols) go; the (new_rows, new_cols) with vals come in, a repeat
    taking its last value, in row-major order (the nonzero ones unless keep_zeros)."""
    major, minor = A._split()
    kept = _core._coo_remove(major, minor, A._data, rows, cols)
    if kept is not None:
        A._set_entries(kept)
    # the distinct new positions, last value winning, row-major: an assignment into nothing
    n_lines = A.shape[0] if A.ndim == 2 else 1
    n_minor = A.shape[-1]
    empty_ptr = _readonly(_sd.zeros(n_lines + 1, dtype=_sd.int64))
    empty_idx = _readonly(_sd.empty(0, dtype=_sd.int64))
    ip, ix, dv = _core._assign_compressed(
        empty_ptr, empty_idx, _sd.empty(0, dtype=A.dtype), n_minor, new_rows, new_cols, vals, insert=True
    ) if vals.shape[0] else (empty_ptr, empty_idx, _sd.empty(0, dtype=A.dtype))  # fmt: skip
    if not keep_zeros:
        f = _core._filter_compressed(n_lines, ip, ix, dv, n_lines, n_minor, drop_zeros=True)
        if f is not None:
            ip, ix, dv = f
    if dv.shape[0] == 0:
        if kept is not None:
            A._canonical = None
        return
    new_major = _core._expand(ip, n_lines, dv.shape[0]) if A.ndim == 2 else None
    major, minor = A._split()
    data = _sd.concat([A._data, dv])
    cols = _readonly(_sd.concat([minor, ix]))
    if A.ndim == 2:
        A._coords = (_readonly(_sd.concat([major, new_major])), cols)
    else:
        A._coords = (cols,)
    A._data = data
    A._canonical = None


# ---- the diagonal ----


def _k_arg(k, func):
    if not C.is_int(k):
        raise TypeError(f"{func}(): 'k' must be an int, not {C.type_name(k)}")
    return operator.index(k)


def _check_2d(A, func):
    if A.ndim != 2:
        raise ValueError(f"{func}(): the array must have 2 dimensions, got {A.ndim}")


def diagonal(A, k, func="diagonal"):
    _check_2d(A, func)
    k = _k_arg(k, func)
    m, n = A.shape
    fmt = A.format
    if fmt in ("csr", "csc"):
        return _core._diagonal_compressed(A._indptr, A._indices, A._data, m, n, k, by_rows=A._by_rows)
    coo = A if fmt == "coo" else A._tocoo(None, func)
    major, minor = coo._split()
    return _core._diagonal_coo(major, minor, coo._data, m, n, k)


def trace(A, k):
    """The sum of the k-th diagonal: sd.sum of diagonal(k) (the array's sum: bool and int64 give
    int64)."""
    return _sd.sum(diagonal(A, k, "trace"))


def setdiag(A, values, k):
    func = "setdiag"
    _check_2d(A, func)
    k = _k_arg(k, func)
    m, n = A.shape
    if (k > 0 and k >= n) or (k < 0 and -k >= m):
        raise ValueError(f"{func}(): k {k} exceeds the array's dimensions {A.shape}")
    length = min(m, n - k) if k >= 0 else min(m + k, n)
    if C.is_scalar(values) or _is_bool_scalar(values):
        vals = _sd.empty(max(length, 0), dtype=A.dtype)
        vals[...] = values
    else:
        from ._base import sparray

        if isinstance(values, sparray):
            raise TypeError(f"{func}(): 'values' must be a number or a 1-D dense array, not {C.type_name(values)}")
        v = C.renamed(func, _sd.asarray, values)
        if v.ndim != 1:
            raise ValueError(f"{func}(): 'values' must be a number or 1-D, got {v.ndim} dimensions")
        length = min(length, v.shape[0])
        vals = _sd.empty(length, dtype=A.dtype)
        vals[...] = v[:length]
    if length <= 0:
        return
    r0, c0 = max(-k, 0), max(k, 0)
    rows = _sd.arange(r0, r0 + length, dtype=_sd.int64)
    cols = _sd.arange(c0, c0 + length, dtype=_sd.int64)
    _apply(A, rows, cols, vals, keep_zeros=True)
