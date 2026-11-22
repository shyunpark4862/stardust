"""The builders of sd.sparse (docs/api/sparse.md §4, docs/design/sparse.md §3.4; milestone SP3):
eye_array, diags_array, random_array, kron, kronsum, hstack, vstack, block_array, block_diag,
tril, triu, find.

  - Data arguments are sparse arrays or dense sd.ndarrays (a dense one stands for its nonzero
    values, as coo_array(x)); lists and numpy arrays are a TypeError, as for the array's
    functions (API.md §5).
  - format=None gives scipy's result format, except where scipy gives DIA or BSR, which are CSR
    here (decision SP51): eye_array, diags_array and kron are CSR; kronsum is CSR; hstack, vstack
    and block_array are CSR when every block is CSR, CSC when every block is CSC, else COO;
    block_diag, tril, triu and random_array are COO.
  - dtype=None is the array's promotion of the inputs (diags_array of int64 diagonals is int64:
    decision SP52); dtype= converts explicitly (the array's astype).
  - Every value is an input value or the array's own product of two (kron), so toarray() is the
    dense result bit for bit.
"""

import math

from .. import _core as _sd
from . import _common as C
from ._common import _core
from ._ops import promote, to_dtype

_CLASSES = None


def _classes():
    global _CLASSES
    if _CLASSES is None:  # imported on first use: the classes' modules import this one
        from ._compressed import csc_array, csr_array
        from ._coo import coo_array
        from ._dok import dok_array

        _CLASSES = (coo_array, csr_array, csc_array, dok_array)
    return _CLASSES


def _readonly(x):
    x._set_readonly()
    return x


def _format_arg(format, func):
    if format is None:
        return None
    if not isinstance(format, str):
        raise TypeError(f"{func}(): 'format' must be 'coo', 'csr', 'csc', 'dok' or None, not {C.type_name(format)}")
    if format not in C.FORMATS:
        raise ValueError(f"{func}(): 'format' must be 'coo', 'csr', 'csc', 'dok' or None, not {format!r}")
    return format


def _sparse_arg(A, func, name, ndims=(1, 2)):
    """A sparse array, or a dense sd.ndarray as the COO array of its nonzero values."""
    from ._base import sparray

    if isinstance(A, sparray):
        out = A
    elif isinstance(A, _sd.ndarray):
        C.check_ndim(A.ndim, func, (1, 2), what=f"'{name}'")
        out = _classes()[0](A)
    else:
        raise TypeError(f"{func}(): '{name}' must be a sparse array or an sd.ndarray, not {C.type_name(A)}")
    C.check_ndim(out.ndim, func, ndims, what=f"'{name}'")
    return out


def _as(A, format, func):
    return A if format is None or format == A.format else A._convert(format, None, func)


def _dtype_or(dtype, func, default):
    d = C.dtype_arg(dtype, func)
    return d if d is not None else default


# ---- eye_array, diags_array ----


def eye_array(n_rows, n_cols=None, /, *, k=0, dtype=None, format=None):
    """The n_rows x n_cols array with ones on the k-th diagonal (sd.eye's arguments), CSR by
    default."""
    func = "eye_array"
    m = C.int64_arg(n_rows, func, "n_rows")
    n = m if n_cols is None else C.int64_arg(n_cols, func, "n_cols")
    k = C.int64_arg(k, func, "k")
    if m < 0 or n < 0:
        raise ValueError(f"{func}(): negative dimensions are not allowed")
    dtype = _dtype_or(dtype, func, _sd.float64)
    format = _format_arg(format, func)
    length = max(0, min(m, n - k) if k >= 0 else min(m + k, n))
    r0, c0 = max(-k, 0), max(k, 0)

    def parts():
        if m == 2**63 - 1:  # an indptr of 2**63 entries
            raise ValueError("array is too big")
        indptr = _sd.clip(_sd.arange(-r0, m + 1 - r0, dtype=_sd.int64), min=0, max=length)
        return _readonly(indptr), _readonly(_sd.arange(c0, c0 + length, dtype=_sd.int64)), _sd.ones(length, dtype=dtype)

    indptr, indices, data = C.sized(func, parts)
    return _as(_classes()[1]._new((m, n), indptr, indices, data), format, func)


def _diagonal_arg(d, func, j):
    if isinstance(d, _sd.ndarray):
        if d.ndim != 1:
            raise ValueError(f"{func}(): diagonal {j} must have 1 dimension, got {d.ndim}")
        return d
    if isinstance(d, bool) or isinstance(d, (int, float, complex)):
        return _sd.asarray([d])
    raise TypeError(f"{func}(): diagonal {j} must be an sd.ndarray or a number, not {C.type_name(d)}")


def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None):
    """The array whose diagonals offsets[j] hold diagonals[j] (one diagonal and an int offset, or
    a list of each); CSR by default. A diagonal of length 1 fills its whole diagonal, a longer one
    is cut. Zeros are not stored (scipy's)."""
    func = "diags_array"
    if C.is_int(offsets):
        if isinstance(diagonals, (list, tuple)):
            raise TypeError(
                f"{func}(): with one offset, 'diagonals' must be one diagonal (an sd.ndarray or a number), "
                f"not {C.type_name(diagonals)}; for several give a list of offsets"
            )
        diags = [_diagonal_arg(diagonals, func, 0)]
        offs = [C.int64_arg(offsets, func, "offsets")]
    elif isinstance(offsets, (list, tuple)):
        offs = [C.int64_arg(o, func, "offsets") for o in offsets]
        if not isinstance(diagonals, (list, tuple)):
            raise TypeError(
                f"{func}(): with a list of offsets, 'diagonals' must be a list or a tuple, not {C.type_name(diagonals)}"
            )
        diags = [_diagonal_arg(d, func, j) for j, d in enumerate(diagonals)]
    else:
        raise TypeError(f"{func}(): 'offsets' must be an int or a list of ints, not {C.type_name(offsets)}")
    if len(diags) != len(offs):
        raise ValueError(f"{func}(): {len(diags)} diagonals but {len(offs)} offsets")
    if not diags:
        raise ValueError(f"{func}(): no diagonals")
    if len(set(offs)) != len(offs):
        raise ValueError(f"{func}(): the offsets repeat a value")
    if shape is None:
        m = diags[0].shape[0] + abs(offs[0])
        C.check_shape_range((m,), func)
        shape = (m, m)
    else:
        shape = C.shape_arg(shape, func, (2,))
    m, n = shape
    out_dtype = _dtype_or(dtype, func, None)
    if out_dtype is None:
        out_dtype = diags[0].dtype
        for d in diags[1:]:
            out_dtype = promote(out_dtype, d.dtype)
    format = _format_arg(format, func)
    return _as(C.sized(func, _diags, func, diags, offs, shape, out_dtype), format, func)


def _diags(func, diags, offs, shape, out_dtype):
    m, n = shape
    parts = []
    for j, (d, off) in enumerate(zip(diags, offs)):
        length = min(m + off, n - off, m, n)  # scipy's: an offset just past the corner is an empty diagonal
        if length < 0:
            raise ValueError(f"{func}(): offset {off} (diagonal {j}) is outside an array of shape {shape}")
        if d.shape[0] != 1 and d.shape[0] < length:
            raise ValueError(
                f"{func}(): diagonal {j} has {d.shape[0]} values, the diagonal at offset {off} of an array "
                f"of shape {shape} has {length}"
            )
        if d.dtype != out_dtype:
            d = C.renamed(func, _sd.astype, d, out_dtype)
        parts.append(_contiguous(d))
    direct = _core._diags(m, n, offs, parts)  # straight to CSR; None when most diagonals miss most rows
    if direct is not None:
        ip, ix, dv, zeros = direct
        if zeros:  # scipy stores no zeros
            kept = _core._filter_compressed(m, ip, ix, dv, m, n, drop_zeros=True)
            if kept is not None:
                ip, ix, dv = kept
        return _classes()[1]._new((m, n), ip, ix, dv)
    return _diags_coo(func, diags, offs, shape, out_dtype)


def _diags_coo(func, diags, offs, shape, out_dtype):
    m, n = shape
    rows, cols, vals = [], [], []
    for j, (d, off) in enumerate(zip(diags, offs)):
        length = min(m + off, n - off, m, n)  # scipy's: an offset just past the corner is an empty diagonal
        if length < 0:
            raise ValueError(f"{func}(): offset {off} (diagonal {j}) is outside an array of shape {shape}")
        if d.shape[0] == 1:
            v = _sd.broadcast_to(d, (length,))
        elif d.shape[0] >= length:
            v = d[:length]
        else:
            raise ValueError(
                f"{func}(): diagonal {j} has {d.shape[0]} values, the diagonal at offset {off} of an array "
                f"of shape {shape} has {length}"
            )
        r0, c0 = max(-off, 0), max(off, 0)
        rows.append(_sd.arange(r0, r0 + length, dtype=_sd.int64))
        cols.append(_sd.arange(c0, c0 + length, dtype=_sd.int64))
        vals.append(C.renamed(func, _sd.astype, v, out_dtype).copy())
    coo_array, csr_array = _classes()[:2]
    data = _sd.concat(vals) if len(vals) > 1 else _contiguous(vals[0])
    r = _sd.concat(rows) if len(rows) > 1 else rows[0]
    c = _sd.concat(cols) if len(cols) > 1 else cols[0]
    ip, ix, dv = _core._compress(m, _readonly(r), _readonly(c), data, share=True)
    kept = _core._filter_compressed(m, ip, ix, dv, m, n, drop_zeros=True)  # scipy stores no zeros
    if kept is not None:
        ip, ix, dv = kept
    return csr_array._new((m, n), ip, ix, dv)


def _contiguous(x):
    return x if C.contiguous(x) else x.copy()


# ---- random_array ----


def random_array(shape, /, *, density=0.01, format="coo", dtype=None, rng, data_sampler=None):
    """A random sparse array: exactly round(density * size) positions drawn without replacement,
    values from data_sampler(n) or uniform on [0, 1). `rng` (an sd.random.Generator) is required (no
    default: SP112);
    the same state gives the same array whatever the thread count. COO by default."""
    func = "random_array"
    from ..random import Generator

    if not isinstance(rng, Generator):
        raise TypeError(
            f"{func}(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not {C.type_name(rng)}"
        )
    shape = C.shape_arg(shape, func)
    if isinstance(density, bool) or not isinstance(density, (int, float)) and not _is_real_scalar(density):
        raise TypeError(f"{func}(): 'density' must be a real number, not {C.type_name(density)}")
    density = float(density)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"{func}(): 'density' must be in [0, 1], got {density}")
    dtype = C.dtype_arg(dtype, func)
    format = _format_arg(format, func)
    if data_sampler is not None and not callable(data_sampler):
        raise TypeError(f"{func}(): 'data_sampler' must be callable or None, not {C.type_name(data_sampler)}")
    total = math.prod(shape)
    size = int(round(density * total))
    if total <= 2**63 - 1:
        flat = rng.choice(total, shape=size, replace=False) if size else _sd.empty(0, dtype=_sd.int64)
        if len(shape) == 2:
            n = shape[1]
            coords = (_readonly(_sd.floor_divide(flat, n)), _readonly(_sd.remainder(flat, n)))
        else:
            coords = (_readonly(flat),)
    else:  # more positions than an int64 counts: draw coordinates until enough distinct ones
        seen, order = set(), []
        while len(order) < size:
            draw = [rng.integers(d, shape=size - len(order)).tolist() for d in shape]
            for key in zip(*draw):
                if key not in seen:
                    seen.add(key)
                    order.append(key)
        coords = tuple(_readonly(_sd.asarray([key[a] for key in order], dtype=_sd.int64)) for a in range(len(shape)))
    if data_sampler is not None:
        vals = data_sampler(size)
        vals = C.renamed(func, _sd.asarray, vals)
        if tuple(vals.shape) != (size,):
            raise ValueError(f"{func}(): data_sampler({size}) must give {size} values, got shape {tuple(vals.shape)}")
        if dtype is not None:
            vals = C.renamed(func, _sd.astype, vals, dtype)
    elif dtype == _sd.int64:
        vals = rng.integers(-(2**63), 2**63 - 1, shape=size)
    elif dtype == _sd.complex128:
        re = rng.random(size)
        vals = re + rng.random(size) * 1j
    else:
        vals = rng.random(size)
        if dtype == _sd.bool:
            vals = _sd.astype(vals, _sd.bool)
    vals = _contiguous(vals)
    A = _classes()[0]._new(shape, coords, vals, None)
    return _as(A, format, func)


def _is_real_scalar(v):
    t = type(v)
    return t.__module__ == "numpy" and t.__name__.startswith(("float", "int", "uint"))


# ---- kron, kronsum ----


def _as_2d_csr(A):
    """A (1-D: one row) as CSR lines."""
    csr_array = _classes()[1]
    if A.ndim == 1:
        a = A if A.format == "csr" else A._tocsr(None, "kron")
        return csr_array._new((1, A.shape[0]), a._indptr, a._indices, a._data)
    return A if A.format == "csr" else A._convert("csr", None, "kron")


def kron(A, B, /, *, format=None):
    """The Kronecker product: the blocks a_ij * B. CSR by default (scipy: BSR). 1-D arrays take
    part as rows; the result is 1-D when both are."""
    func = "kron"
    A = _sparse_arg(A, func, "A")
    B = _sparse_arg(B, func, "B")
    format = _format_arg(format, func)
    loop = C.renamed(func, _sd.multiply, _sd.zeros(1, dtype=A.dtype), _sd.zeros(1, dtype=B.dtype)).dtype
    a, b = _as_2d_csr(A), _as_2d_csr(B)
    if A.ndim == 1 and B.ndim == 1:
        shape = (A.shape[0] * B.shape[0],)
    else:
        shape = (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])
    C.check_shape_range(shape, func)
    ip, ix, dv = C.sized(
        func,
        _core._kron,
        a._indptr,
        a._indices,
        to_dtype(a._data, loop),
        b._indptr,
        b._indices,
        to_dtype(b._data, loop),
        b.shape[1],
    )
    return _as(_classes()[1]._new(shape, ip, ix, dv), format, func)


def kronsum(A, B, /, *, format=None):
    """kron(I_n, A) + kron(B, I_m) for square A (m x m) and B (n x n): CSR by default (scipy's)."""
    func = "kronsum"
    A = _sparse_arg(A, func, "A", (2,))
    B = _sparse_arg(B, func, "B", (2,))
    format = _format_arg(format, func)
    for name, X in (("A", A), ("B", B)):
        if X.shape[0] != X.shape[1]:
            raise ValueError(f"{func}(): '{name}' must be square, got shape {X.shape}")
    dtype = promote(A.dtype, B.dtype)
    loop = C.renamed(func, _sd.multiply, _sd.zeros(1, dtype=dtype), _sd.zeros(1, dtype=dtype)).dtype  # kron's
    m, n = A.shape[0], B.shape[0]
    C.check_shape_range((m * n,), func)
    return _as(C.sized(func, _kronsum, A, B, m, n, loop, func), format, func)


def _kronsum(A, B, m, n, loop, func):
    # The entries placed, not multiplied by the identity's ones (SP111): A's in every diagonal
    # block, b_ij on the diagonal of block (i, j); a position both hold (a_pp and b_ii) is their one
    # sum a + b, the rest keep their values (complex inf, -0.0 parts), as sd.linalg.kronsum.
    a, b = _as_2d_csr(A), _as_2d_csr(B)
    if m > 0 and n > 0:  # straight to CSR, row by row
        ip, ix, dv = _core._kronsum(
            a._indptr, a._indices, to_dtype(a._data, loop), b._indptr, b._indices, to_dtype(b._data, loop)
        )
        return _classes()[1]._new((m * n, m * n), ip, ix, dv)
    a, b = a.tocoo(), b.tocoo()
    (ra, ca), (rb, cb) = a.coords, b.coords
    base = _sd.arange(n, dtype=_sd.int64)[:, None] * m
    q = _sd.arange(m, dtype=_sd.int64)[None, :]
    na, nb = ra.shape[0], rb.shape[0]
    rows = _sd.concat([(base + ra[None, :]).reshape(n * na), (rb[:, None] * m + q).reshape(nb * m)])
    cols = _sd.concat([(base + ca[None, :]).reshape(n * na), (cb[:, None] * m + q).reshape(nb * m)])
    av, bv = to_dtype(a.data, loop), to_dtype(b.data, loop)
    vals = _sd.concat(
        [_sd.broadcast_to(av[None, :], (n, na)).reshape(n * na), _sd.broadcast_to(bv[:, None], (nb, m)).reshape(nb * m)]
    )
    coo = _classes()[0]._new((m * n, m * n), (_readonly(rows), _readonly(cols)), vals, None)
    out = coo._tocsr(None, func)  # a + b in that order (left to right, as the old kron(I, A) + kron(B, I))
    out.eliminate_zeros()  # the exact zeros go, as a sum of two sparse arrays drops them
    return out


# ---- stacking ----


def _block_arg(b, func, where, arg="blocks"):
    if b is None:
        return None
    from ._base import sparray

    if isinstance(b, sparray):
        A = b
    elif isinstance(b, _sd.ndarray):
        C.check_ndim(b.ndim, func, (1, 2), what=f"{arg}[{where}]")
        A = _classes()[0](b)
    else:
        raise TypeError(
            f"{func}(): {arg}[{where}] must be "
            f"{'a sparse array, an sd.ndarray or None' if arg == 'blocks' else 'a sparse array or an sd.ndarray'}"
            f", not {C.type_name(b)}"
        )
    if A.ndim == 1:  # a row, as scipy's
        A = A.reshape((1, A.shape[0])) if A.format != "csr" else _as_2d_csr(A)
    return A


def _grid(blocks, func):
    if not isinstance(blocks, (list, tuple)):
        raise TypeError(f"{func}(): 'blocks' must be a list or a tuple of lists, not {C.type_name(blocks)}")
    if not blocks or not all(isinstance(row, (list, tuple)) for row in blocks):
        raise ValueError(f"{func}(): 'blocks' must be 2-D (a list of rows of blocks), and some block must be given")
    width = len(blocks[0])
    if any(len(row) != width for row in blocks) or width == 0:
        raise ValueError(f"{func}(): every row of 'blocks' must have the same number of blocks")
    return [[_block_arg(b, func, f"{i},{j}") for j, b in enumerate(row)] for i, row in enumerate(blocks)]


def _block(grid, format, dtype, func):
    format = _format_arg(format, func)
    dtype = C.dtype_arg(dtype, func)
    M, N = len(grid), len(grid[0])
    heights, widths = [0] * M, [0] * N
    for i in range(M):
        for j in range(N):
            A = grid[i][j]
            if A is None:
                continue
            h, w = A.shape
            if heights[i] == 0:
                heights[i] = h
            elif heights[i] != h:
                raise ValueError(
                    f"{func}(): blocks[{i},:] has incompatible row dimensions: blocks[{i},{j}].shape[0] is {h}, "
                    f"expected {heights[i]}"
                )
            if widths[j] == 0:
                widths[j] = w
            elif widths[j] != w:
                raise ValueError(
                    f"{func}(): blocks[:,{j}] has incompatible column dimensions: blocks[{i},{j}].shape[1] is "
                    f"{w}, expected {widths[j]}"
                )
    given = [A for row in grid for A in row if A is not None]
    if not given:
        raise ValueError(f"{func}(): some block must be given (not None)")
    if dtype is None:
        dtype = given[0].dtype
        for A in given[1:]:
            dtype = promote(dtype, A.dtype)

    def values(A):
        return C.renamed(func, _sd.astype, A._data, dtype) if A.dtype != dtype else A._data

    shape = (sum(heights), sum(widths))
    C.check_shape_range(shape, func)
    return C.sized(func, _assemble, grid, shape, heights, widths, given, values, format, func)


def _assemble(grid, shape, heights, widths, given, values, format, func):
    coo_array, csr_array, csc_array, _ = _classes()
    M, N = len(grid), len(grid[0])
    full = all(A is not None for row in grid for A in row)
    fmts = {A.format for A in given}
    if full and fmts == {"csr"} and format in (None, "csr"):
        rows = []
        for row in grid:
            offs = [sum(widths[:j]) for j in range(N)]
            parts = [(A._indptr, A._indices, values(A)) for A in row]
            rows.append(_core._stack_minor(parts, offs) if N > 1 else parts[0])
        parts = _core._stack_major(rows) if M > 1 else rows[0]
        return csr_array._new(shape, *parts)
    if full and fmts == {"csc"} and format in (None, "csc"):
        cols = []
        for j in range(N):
            offs = [sum(heights[:i]) for i in range(M)]
            parts = [(grid[i][j]._indptr, grid[i][j]._indices, values(grid[i][j])) for i in range(M)]
            cols.append(_core._stack_minor(parts, offs) if M > 1 else parts[0])
        parts = _core._stack_major(cols) if N > 1 else cols[0]
        return csc_array._new(shape, *parts)
    rs, cs, vs = [], [], []
    r_off = 0
    for i in range(M):
        c_off = 0
        for j in range(N):
            A = grid[i][j]
            if A is not None:
                coo = A if A.format == "coo" else A._tocoo(None, func)
                rs.append(_sd.add(coo._coords[0], r_off))
                cs.append(_sd.add(coo._coords[1], c_off))
                vs.append(values(coo))
            c_off += widths[j]
        r_off += heights[i]
    out = coo_array._new(
        shape, (_readonly(_sd.concat(rs)), _readonly(_sd.concat(cs))), _contiguous(_sd.concat(vs)), None
    )
    return _as(out, format, func)


def _list_arg(blocks, func):
    if not isinstance(blocks, (list, tuple)):
        raise TypeError(f"{func}(): 'blocks' must be a list or a tuple, not {C.type_name(blocks)}")
    if not blocks:
        raise ValueError(f"{func}(): 'blocks' must hold at least one block")
    return list(blocks)


def hstack(blocks, /, *, format=None, dtype=None):
    """The blocks side by side (1-D blocks are rows)."""
    func = "hstack"
    blocks = _list_arg(blocks, func)
    return _block([[_block_arg(b, func, f"{j}") for j, b in enumerate(blocks)]], format, dtype, func)


def vstack(blocks, /, *, format=None, dtype=None):
    """The blocks one above the other (1-D blocks are rows)."""
    func = "vstack"
    blocks = _list_arg(blocks, func)
    return _block([[_block_arg(b, func, f"{i}")] for i, b in enumerate(blocks)], format, dtype, func)


def block_array(blocks, /, *, format=None, dtype=None):
    """The array of blocks: a list of rows of sparse arrays, dense sd.ndarrays or None (a zero
    block, its size taken from the others of its row and column)."""
    func = "block_array"
    return _block(_grid(blocks, func), format, dtype, func)


def block_diag(arrays, /, *, format=None, dtype=None):
    """The blocks along the diagonal (1-D blocks are rows), COO by default. The argument is
    `arrays`, sd.linalg.block_diag's name (SP119; scipy's is mats)."""
    func = "block_diag"
    if not isinstance(arrays, (list, tuple)):
        raise TypeError(f"{func}(): 'arrays' must be a list or a tuple, not {C.type_name(arrays)}")
    if not arrays:
        raise ValueError(f"{func}(): 'arrays' must hold at least one array")
    blocks = []
    for j, b in enumerate(arrays):
        if b is None:
            raise TypeError(f"{func}(): arrays[{j}] must be a sparse array or an sd.ndarray, not NoneType")
        blocks.append(_block_arg(b, func, f"{j}", "arrays").tocoo())
    coo_array = _classes()[0]
    out_dtype = C.dtype_arg(dtype, func)
    if out_dtype is None:
        out_dtype = blocks[0].dtype
        for A in blocks[1:]:
            out_dtype = promote(out_dtype, A.dtype)
    format = _format_arg(format, func)
    C.check_shape_range((sum(A.shape[0] for A in blocks), sum(A.shape[1] for A in blocks)), func)
    rs, cs, vs = [], [], []
    r_off = c_off = 0
    for A in blocks:
        rs.append(_sd.add(A._coords[0], r_off))
        cs.append(_sd.add(A._coords[1], c_off))
        vs.append(C.renamed(func, _sd.astype, A._data, out_dtype) if A.dtype != out_dtype else A._data)
        r_off += A.shape[0]
        c_off += A.shape[1]
    out = coo_array._new(
        (r_off, c_off), (_readonly(_sd.concat(rs)), _readonly(_sd.concat(cs))), _contiguous(_sd.concat(vs)), None
    )
    return _as(out, format, func)


# ---- tril, triu, find ----


def _band(A, k, format, lower, func):
    A = _sparse_arg(A, func, "A", (2,))
    k = C.int64_arg(k, func, "k")
    format = _format_arg(format, func)
    coo_array = _classes()[0]
    if A.format in ("csr", "csc"):  # each sorted line keeps a prefix or a suffix
        by_rows = A.format == "csr"
        r, c, v = _core._band_compressed(A._indptr, A._indices, A._data, k, lower=lower, by_rows=by_rows)
        return _as(coo_array._new(A.shape, (r, c), v, True if by_rows else None), format, func)
    coo = A if A.format == "coo" else A._tocoo(None, func)
    kept = _core._band_coo(coo._coords[0], coo._coords[1], coo._data, k, lower=lower)
    if kept is None:
        out = coo_array._new(A.shape, coo._coords, coo._data.copy(), coo._canonical)
    else:
        r, c, v = kept
        out = coo_array._new(A.shape, (r, c), v, True if coo._canonical is True else None)
    return _as(out, format, func)


def tril(A, /, *, k=0, format=None):
    """The entries on and below the k-th diagonal (col <= row + k); COO by default."""
    return _band(A, k, format, True, "tril")


def triu(A, /, *, k=0, format=None):
    """The entries on and above the k-th diagonal (col >= row + k); COO by default."""
    return _band(A, k, format, False, "triu")


def find(A, /):
    """(row, col, data) of the nonzero values in row-major order (a COO array's repeats summed
    first): three dense sd.ndarrays."""
    func = "find"
    A = _sparse_arg(A, func, "A", (2,))
    coo = A if A.format == "coo" else A._tocoo(None, func)
    major, minor, data = coo._summed()
    kept = _core._filter_coo(major, minor, data, 2**63 - 1, 2**63 - 1, drop_zeros=True)
    if kept is not None:
        major, minor, data = kept
    elif data is coo._data:
        data = data.copy()  # it may be A's own values
    return C.writable(major), C.writable(minor), data
