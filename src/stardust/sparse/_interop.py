"""The constructors' inputs and the way out to scipy (docs/api/sparse.md §2, docs/design/sparse.md §2.5).

build(fmt, arg, ...) makes the array of format `fmt` from any accepted `arg`:
  - another sd sparse array: a format conversion (copy=None shares what it can);
  - a scipy.sparse array or matrix of any format: always copied (scipy may change its arrays in
    place later, and int32 indices widen to int64);
  - a shape tuple: an empty array (dtype=None is float64);
  - (data, (row, col)) or (data, (i,)) for every format, (data, indices, indptr) for CSR and CSC;
  - a dense array (an sd.ndarray, a numpy array, nested sequences): its nonzero values.
The values follow asarray's rules (numpy's lossless dtypes such as float32 widen; dtype= converts
explicitly; copy as asarray's). The index arrays are always copied and checked, so that changing
the arrays given later cannot break the canonical form.
"""

from .. import _core as _sd
from . import _common as C
from ._common import _core

_CLASSES = None


def _classes():
    global _CLASSES
    if _CLASSES is None:  # imported on first use: the classes' modules import this one
        from ._base import sparray
        from ._compressed import csc_array, csr_array
        from ._coo import coo_array
        from ._dok import dok_array

        _CLASSES = {"coo": coo_array, "csr": csr_array, "csc": csc_array, "dok": dok_array, "base": sparray}
    return _CLASSES


def _ndims(fmt):
    return (2,) if fmt == "csc" else (1, 2)


def build(fmt, arg, shape, dtype, copy, func):
    try:
        return _build(fmt, arg, shape, dtype, copy, func)
    except (MemoryError, ValueError) as e:  # a kernel's allocation failure, named by the constructor
        r = C.allocation_error(func, e)
        if r is None:
            raise
        raise r from None


def _build(fmt, arg, shape, dtype, copy, func):
    copy = C.copy_arg(copy, func)
    dtype = C.dtype_arg(dtype, func)
    if shape is not None:
        shape = C.shape_arg(shape, func, _ndims(fmt))
    if isinstance(arg, _classes()["base"]):
        return _from_sparse(fmt, arg, shape, dtype, copy, func)
    if C.is_scipy_sparse(arg):
        return _from_scipy(fmt, arg, shape, dtype, copy, func)
    if isinstance(arg, tuple):
        if arg and C.is_int(arg[0]) and all(C.is_int(v) for v in arg):
            return _empty(fmt, C.shape_arg(arg, func, _ndims(fmt), what="arg"), shape, dtype, func)
        if len(arg) == 2 and _is_coords(arg[1]) and not C.is_scalar(arg[0]):
            return _from_coords(fmt, arg[0], arg[1], shape, dtype, copy, func)
        if len(arg) == 3 and fmt in ("csr", "csc") and not any(C.is_scalar(p) for p in arg):
            return _from_parts(fmt, arg, shape, dtype, copy, func)
    return _from_dense(fmt, arg, shape, dtype, copy, func)


def _is_coords(v):
    return isinstance(v, tuple) and 1 <= len(v) <= 2 and not any(C.is_scalar(c) for c in v)


def _check_shape(shape, have, func):
    if shape is not None and shape != have:
        raise ValueError(f"{func}(): 'shape' {shape} does not match the input's shape {have}")


def _empty(fmt, dims, shape, dtype, func):
    _check_shape(shape, dims, func)
    dtype = dtype or _sd.float64
    cls = _classes()[fmt]
    if fmt == "dok":
        return cls._empty(dims, dtype)
    data = _sd.empty(0, dtype=dtype)
    idx = _sd.empty(0, dtype=_sd.int64)
    idx._set_readonly()
    if fmt == "coo":
        return cls._new(dims, tuple(idx for _ in dims), data, True)
    lines = 1 if len(dims) == 1 else (dims[0] if fmt == "csr" else dims[1])
    if lines == 2**63 - 1:  # an indptr of 2**63 entries (build names the constructor)
        raise ValueError("array is too big")
    indptr = _sd.zeros(lines + 1, dtype=_sd.int64)
    indptr._set_readonly()
    return cls._new(dims, indptr, idx, data)


def _check_inferred(shape, func):
    if any(d > 2**63 - 1 for d in shape):  # an index of 2**63 - 1 needs a length of 2**63
        raise OverflowError(f"{func}(): the shape inferred from the indices is out of range; pass 'shape'")


def _from_sparse(fmt, A, shape, dtype, copy, func):
    _check_shape(shape, A.shape, func)
    if fmt == "csc" and A.ndim != 2:
        raise ValueError(f"{func}(): a csc_array has 2 dimensions, the input has {A.ndim}")
    if dtype is not None and dtype != A.dtype:
        C.check_copy_false(copy, func, "a dtype conversion")
        A = A._astype(dtype)
        copy = None  # the values are new already
    return A._convert(fmt, copy, func)


def _from_dense(fmt, arg, shape, dtype, copy, func):
    x = C.as_dense(arg, func)
    C.check_ndim(x.ndim, func, _ndims(fmt), what="a dense input")
    _check_shape(shape, tuple(x.shape), func)
    C.check_copy_false(copy, func, "a dense input")
    indptr, indices, data = _core._from_dense(x, by_rows=fmt != "csc")
    if dtype is not None:  # after the pattern, as scipy: a value that becomes 0 stays stored
        data = C.renamed(func, _sd.astype, data, dtype)
    shape = tuple(x.shape)
    cls = _classes()
    if fmt in ("csr", "csc"):
        return cls[fmt]._new(shape, indptr, indices, data)
    if x.ndim == 1:
        major = None
    else:
        major = _core._expand(indptr, shape[0], data.shape[0])
    if fmt == "coo":
        return cls["coo"]._new(shape, (major, indices) if major is not None else (indices,), data, True)
    return cls["dok"]._from_coo(shape, major, indices, data)


def _from_coords(fmt, data, coords, shape, dtype, copy, func):
    nd = len(coords)
    if fmt == "csc" and nd != 2:
        raise ValueError(f"{func}(): a csc_array has 2 dimensions; the coordinates are for {nd}")
    idx = [C.index_array(c, func, "coords") for c in coords]
    data = C.data_array(data, func, dtype, copy)
    for i in idx:
        if i.shape[0] != data.shape[0]:
            raise ValueError(
                f"{func}(): 'data' and the index arrays must have the same length, got {data.shape[0]} and {i.shape[0]}"
            )
    if shape is None:
        if data.shape[0] == 0:
            raise ValueError(f"{func}(): cannot infer the shape from empty index arrays; pass 'shape'")
        shape = tuple(_core._index_max(i, axis=a, func=func) + 1 for a, i in enumerate(idx))
        _check_inferred(shape, func)
    elif len(shape) != nd:
        raise ValueError(f"{func}(): 'shape' has {len(shape)} dimensions, the coordinates are for {nd}")
    owned = tuple(_core._index_copy(i, shape[a], axis=a, func=func) for a, i in enumerate(idx))
    coo = _classes()["coo"]._new(shape, owned, data, None)
    if fmt == "coo":
        return coo
    if fmt == "dok":
        C.check_copy_false(copy, func, "a dok_array")
        return coo._todok(None, func)
    return coo._convert(fmt, False if copy is False else None, func)


def _from_parts(fmt, parts, shape, dtype, copy, func):
    data_in, indices_in, indptr_in = parts
    indices = C.index_array(indices_in, func, "indices")
    indptr = C.index_array(indptr_in, func, "indptr")
    data = C.data_array(data_in, func, dtype, copy)
    by_rows = fmt == "csr"
    word = "rows" if by_rows else "columns"
    if shape is None:
        if indptr.shape[0] == 0:
            raise ValueError(f"{func}(): 'indptr' must not be empty")
        if indices.shape[0] == 0:
            raise ValueError(f"{func}(): cannot infer the shape from empty index arrays; pass 'shape'")
        lines = indptr.shape[0] - 1
        positions = _core._index_max(indices, axis=1 if by_rows else 0, func=func) + 1
        shape = (lines, positions) if by_rows else (positions, lines)
        _check_inferred(shape, func)
    if len(shape) == 1:
        lines, positions, word = 1, shape[0], "lines (1 for a 1-D array)"
    else:
        lines, positions = (shape[0], shape[1]) if by_rows else (shape[1], shape[0])
    if indptr.shape[0] != lines + 1:
        raise ValueError(
            f"{func}(): 'indptr' must have {lines + 1} entries (one more than the number of {word}), "
            f"got {indptr.shape[0]}"
        )
    if data.shape[0] != indices.shape[0]:
        raise ValueError(
            f"{func}(): 'data' and 'indices' must have the same length, got {data.shape[0]} and {indices.shape[0]}"
        )
    minor_axis = 0 if len(shape) == 1 else (1 if by_rows else 0)
    indptr, indices, out = _core._compressed_in(
        lines, positions, indptr, indices, data, minor_axis=minor_axis, share=True, func=func
    )
    if copy is False and out is not data:
        C.check_copy_false(copy, func, "sorting or summing duplicates")
    return _classes()[fmt]._new(shape, indptr, indices, out)


def _from_scipy(fmt, S, shape, dtype, copy, func):
    C.check_copy_false(copy, func, "an input from scipy")
    if S.ndim not in (1, 2):
        raise ValueError(f"{func}(): the input must have 1 or 2 dimensions, got {S.ndim}")
    _check_shape(shape, tuple(S.shape), func)
    if S.format in ("csr", "csc") and (S.format == fmt or fmt not in ("csr", "csc")):
        A = _from_parts(S.format, (S.data, S.indices, S.indptr), tuple(S.shape), dtype, True, func)
    else:
        coo = S.tocoo()
        A = _from_coords("coo", coo.data, tuple(coo.coords), tuple(S.shape), dtype, True, func)
    return A._convert(fmt, None, func)


# ---- out to scipy ----


def to_scipy(A, copy):
    import numpy as np
    import scipy.sparse as sp

    def view(x):
        a = np.asarray(x)  # zero copy; read-only for the read-only index arrays
        return a.copy() if copy else a

    fmt = A.format
    if fmt == "dok":
        C.check_copy_false(copy, "to_scipy", "a scipy dok_array")
        coo = A._tocoo(None, "to_scipy")
        return sp.coo_array((np.asarray(coo._data), tuple(np.asarray(c) for c in coo._coords)), shape=A.shape).todok()
    if fmt == "coo":
        S = sp.coo_array(A.shape, dtype=np.asarray(A._data).dtype)
        S.coords = tuple(view(c) for c in A._coords)
        S.data = view(A._data)
        S.has_canonical_format = bool(A.has_canonical_format)
        return S
    cls = sp.csr_array if fmt == "csr" else sp.csc_array
    S = cls(A.shape, dtype=np.asarray(A._data).dtype)
    S.indptr = view(A._indptr)
    S.indices = view(A._indices)
    S.data = view(A._data)
    S.has_canonical_format = True
    return S
