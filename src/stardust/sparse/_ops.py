"""The operators, the element-wise functions and the products of sd.sparse (docs/api/sparse.md §3.1,
§3.2; docs/design/sparse.md §2.4, §3.2).

The rules, in short:
  - Every value is the array's own operation on the stored values (sd.multiply, sd.sin, ... or the
    same scalar formulas in the native kernels), so it is bit for bit what the dense operation gives
    at that position. Dtypes follow the array's promotion.
  - Sparse with sparse: the shapes must be equal. +, -, maximum, minimum and the comparisons merge
    the two patterns (a position stored on one side meets the other side's implicit zero);
    multiply keeps the positions both store (an implicit zero is a strong zero). Exact zeros of the
    result are dropped and the result is CSR, or CSC when the left operand is CSC (scipy's rule).
    A comparison true between two implicit zeros (==, <=, >=) stores True everywhere else.
  - Sparse with a scalar: * and / keep the pattern and the format (A / 0 gives +-inf and NaN at the
    stored values only); A + c and A - c are a ValueError for c != 0; maximum, minimum and
    comparisons drop the zeros they make (CSR or CSC), or store every position (maximum with a
    positive scalar, a comparison true at 0) as scipy does.
  - Sparse with a dense sd.ndarray (it must broadcast to the sparse array's shape): * and / keep
    the pattern and the format of the sparse array; +, -, maximum, minimum and comparisons are
    dense: the dense operation on A.toarray().
  - A numpy array or a list is not an operand (the operators give NotImplemented, so Python raises
    TypeError; the methods raise it themselves).
  - @: sparse @ dense and dense @ sparse are dense, sparse @ sparse is sparse; 1-D @ 1-D is a
    Python scalar. The sums run in the order of docs/design/sparse.md §2.6 (native kernels).
"""

import numbers
import operator

from .. import _core as _sd
from . import _common as C
from ._common import _core

# ---- operands ----


def is_number(v):
    """A scalar operand, as the array's operators take it: bool, int, float, complex and numpy's
    scalars (numpy.bool included)."""
    if isinstance(v, numbers.Number):
        return True
    t = type(v)
    return t.__module__ == "numpy" and t.__name__ == "bool"


def _sparse_classes():
    from ._compressed import csc_array, csr_array
    from ._coo import coo_array
    from ._dok import dok_array

    return coo_array, csr_array, csc_array, dok_array


def is_sparse(v):
    from ._base import sparray

    return isinstance(v, sparray)


def compressed(A, func="op"):
    """A as a canonical compressed array: itself for CSR and CSC, the CSR of any other format."""
    if A.format in ("csr", "csc"):
        return A
    return A._tocsr(None, func)


def oriented(A, by_rows, func="op"):
    """A as CSR (by_rows) or CSC (2-D only)."""
    if by_rows:
        return A if A.format == "csr" else A._convert("csr", None, func)
    return A if A.format == "csc" else A._convert("csc", None, func)


def new_compressed(by_rows, shape, indptr, indices, data):
    _, csr_array, csc_array, _ = _sparse_classes()
    return (csr_array if by_rows else csc_array)._new(shape, indptr, indices, data)


def to_dtype(x, dtype):
    return x if x.dtype == dtype else _sd.astype(x, dtype)


def promote(a, b):
    order = (_sd.bool, _sd.int64, _sd.float64, _sd.complex128)
    return order[max(order.index(a), order.index(b))]


def one(dtype):
    return _sd.zeros(1, dtype=dtype)


# ---- the same pattern, new values ----


def with_values(A, f, *, dedupe, dok_csr=False):
    """A new array of A's pattern whose values are f(values). dedupe: a COO array's repeated
    positions are summed first (a nonlinear f; scipy does it in place, this does not change A).
    A dok_array stays DOK unless dok_csr (scipy's result for abs, power and the functions)."""
    coo_array, csr_array, _, dok_array = _sparse_classes()
    fmt = A.format
    if fmt in ("csr", "csc"):
        return A._with_data(f(A._data))
    if fmt == "coo":
        if dedupe and not A.has_canonical_format:
            major, minor, data = A._summed()
            coords = (major, minor) if major is not None else (minor,)
            return coo_array._new(A._shape, coords, f(data), True)
        return A._with_data(f(A._data))
    major, minor, data = A._parts()
    values = f(data)
    if dok_csr:
        n_major = A._shape[0] if major is not None else 1
        return csr_array._new(A._shape, *_core._compress(n_major, major, minor, values, share=True))
    return dok_array._from_coo(A._shape, major, minor, values)


def dropped_zeros(cs, values):
    """The compressed array cs's pattern with `values`, its exact zeros dropped (a new array)."""
    lines, positions = cs._lines()
    kept = _core._filter_compressed(lines, cs._indptr, cs._indices, values, lines, positions, drop_zeros=True)
    if kept is None:
        return cs._with_data(values)
    return new_compressed(cs._by_rows, cs._shape, *kept)


def filled_true(cs, flags):
    """A bool array of cs's orientation: True at the stored positions whose flag is True and at
    every position cs does not store."""
    _, positions = cs._lines()
    return new_compressed(cs._by_rows, cs._shape, *_core._fill_true(cs._indptr, cs._indices, flags, positions))


# ---- sparse with sparse ----

_OPERATORS = {
    "add": operator.add, "subtract": operator.sub, "multiply": operator.mul, "equal": operator.eq,
    "not_equal": operator.ne, "less": operator.lt, "less_equal": operator.le, "greater": operator.gt,
    "greater_equal": operator.ge,
}  # fmt: skip
_COMPARISONS = ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal")
_TRUE_AT_ZERO = ("equal", "less_equal", "greater_equal")


def _probe(op, da, db, func):
    """The array's result dtype of op on dtypes da and db (raising the array's error when there is
    no loop): an operator with the operator's message, maximum/minimum with the method's name."""
    if op in _OPERATORS:
        return _OPERATORS[op](one(da), one(db)).dtype
    return C.renamed(func, getattr(_sd, op), one(da), one(db)).dtype


def check_same_shape(A, B, func=None):
    if A.shape != B.shape:
        head = f"{func}(): " if func else ""
        raise ValueError(f"{head}shapes could not be broadcast together: {A.shape} {B.shape}")


def sparse_sparse(A, B, op, func=None):
    check_same_shape(A, B, func)
    out = _probe(op, A.dtype, B.dtype, func or op)
    loop = promote(A.dtype, B.dtype) if op in _COMPARISONS else out
    by_rows = A.format != "csc"
    a, b = oriented(A, by_rows), oriented(B, by_rows)
    fill = op in _TRUE_AT_ZERO
    parts = _core._binop(
        op, a._indptr, a._indices, to_dtype(a._data, loop), b._indptr, b._indices, to_dtype(b._data, loop),
        keep_all=fill,
    )  # fmt: skip
    res = new_compressed(by_rows, A.shape, *parts)
    if fill:
        res = filled_true(res, res._data)
    return res


# ---- sparse with a scalar ----


def scalar_compare(A, c, op):
    cs = compressed(A)
    flags = _OPERATORS[op](cs._data, c)
    if _OPERATORS[op](one(A.dtype), c).tolist()[0]:  # true at 0: every unstored position is True
        return filled_true(cs, flags)
    return dropped_zeros(cs, flags)


def scalar_extreme(A, c, op, func):
    """maximum/minimum with a scalar: the zeros made are dropped (CSR or CSC); when op(0, c) is
    not 0, every position is stored, in A's format (scipy densifies the same way)."""
    f = getattr(_sd, op)
    at_zero = C.renamed(func, f, one(A.dtype), c).tolist()[0]
    if at_zero != 0 or at_zero != at_zero:
        dense = f(A.toarray(), c)
        return type(A)(dense)
    cs = compressed(A, func)
    return dropped_zeros(cs, f(cs._data, c))


# ---- sparse with a dense array ----


def check_dense(A, D, func=None):
    """D must broadcast to A's shape (the result has A's shape)."""
    head = f"{func}(): " if func else ""
    if D.ndim > A.ndim or any(d not in (1, s) for d, s in zip(reversed(D.shape), reversed(A.shape))):
        raise ValueError(f"{head}shapes could not be broadcast together: {A.shape} {tuple(D.shape)}")


def dense_values(A, D):
    """D (broadcast to A's shape) at A's stored positions, in A's stored order."""
    Db = D if tuple(D.shape) == A.shape else _sd.broadcast_to(D, A.shape)
    fmt = A.format
    if fmt in ("csr", "csc"):
        return _core._gather_compressed(A._indptr, A._indices, Db, by_rows=A._by_rows)
    if fmt == "coo":
        major, minor = A._split()
        return _core._gather_coo(major, minor, Db)
    major, minor, _ = A._parts()
    return _core._gather_coo(major, minor, Db)


def dense_pattern_op(A, D, f, reflected=False, func=None):
    """f(values, D at the stored positions) on A's pattern and format (f(D, values) reflected):
    A * D and A / D."""
    check_dense(A, D, func)
    g = dense_values(A, D)  # in the order with_values hands the values over (a DOK's: row-major)
    if reflected:
        return with_values(A, lambda v: f(g, v), dedupe=False)
    return with_values(A, lambda v: f(v, g), dedupe=False)


def dense_result(A, D, f, reflected=False, func=None):
    check_dense(A, D, func)
    return f(D, A.toarray()) if reflected else f(A.toarray(), D)


# ---- @ ----


def _matmul_dtype(da, db):
    return (_sd.zeros((1, 1), dtype=da) @ _sd.zeros((1, 1), dtype=db)).dtype


def _mismatch(n1, n2):
    raise ValueError(f"the last axis of 'x1' ({n1}) must match the second-to-last axis of 'x2' ({n2})")


def _python_scalar(y):
    return y.tolist()[0]


def matmul_dense(A, X):
    """A @ X for a dense X: (m,) or (m, k); 1-D A gives a scalar or (k,)."""
    n = A.shape[-1]
    if X.shape[0] != n:
        _mismatch(n, X.shape[0])
    loop = _matmul_dtype(A.dtype, X.dtype)
    cs = compressed(A)
    data, X = to_dtype(cs._data, loop), to_dtype(X, loop)
    if cs._by_rows:
        y = _core._spmv(cs._indptr, cs._indices, data, X, n_minor=n, scatter=False)
    else:
        y = _core._spmv(cs._indptr, cs._indices, data, X, n_minor=A.shape[0], scatter=True, track=cs._long_across())
    if A.ndim == 1:
        return _python_scalar(y) if X.ndim == 1 else y.reshape((y.shape[1],))
    return y


def rmatmul_dense(A, X):
    """X @ A for a dense X: X (m,) gives (n,), X (p, m) gives (p, n); a 1-D A (n,) is a column."""
    if A.ndim == 1:
        n = A.shape[0]
        if X.shape[-1] != n:
            _mismatch(X.shape[-1], n)
        if X.ndim == 1:
            return matmul_dense(A, X)
        return matmul_dense(A, X.mT)  # y[r] = sum_j a_j X[r, j]
    m = A.shape[0]
    if X.shape[-1] != m:
        _mismatch(X.shape[-1], m)
    loop = _matmul_dtype(X.dtype, A.dtype)
    cs = compressed(A)
    data, X = to_dtype(cs._data, loop), to_dtype(X, loop)
    two = X.ndim == 2
    Xt = X.mT if two else X  # rows of Xt are indexed by A's row index
    if cs._by_rows:  # the terms of output j come from the rows of A: a scatter
        return _core._spmv(
            cs._indptr, cs._indices, data, Xt, n_minor=A.shape[1], scatter=True, transpose_out=two,
            track=cs._long_across(),
        )  # fmt: skip
    return _core._spmv(cs._indptr, cs._indices, data, Xt, n_minor=m, scatter=False, transpose_out=two)


def _column_lines(v):
    """A 1-D sparse vector v (n,) as canonical lines of an (n, 1) CSR: line k holds v_k at 0."""
    cs = compressed(v)
    n = v.shape[0]
    nnz = cs._data.shape[0]
    zeros = _sd.zeros(nnz, dtype=_sd.int64)
    zeros._set_readonly()
    indptr, indices, data = _core._compress(n, cs._indices, zeros, cs._data, share=True)
    return indptr, indices, data


def matmul_sparse(A, B):
    """A @ B, both sparse: CSR (CSC when A is CSC), a 1-D result COO (scipy), 1-D @ 1-D a scalar."""
    coo_array, csr_array, csc_array, _ = _sparse_classes()
    n = A.shape[-1]
    if B.shape[0] != n:
        _mismatch(n, B.shape[0])
    loop = _matmul_dtype(A.dtype, B.dtype)
    if B.ndim == 1:
        a = oriented(A, True)
        bp, bi, bd = _column_lines(B)
        ip, ix, d = _core._spgemm(a._indptr, a._indices, to_dtype(a._data, loop), bp, bi, to_dtype(bd, loop), 1)
        if A.ndim == 1:
            vals = d.tolist()
            return vals[0] if vals else _sd.zeros(1, dtype=loop).tolist()[0]
        rows = _core._expand(ip, A.shape[0], d.shape[0])
        return coo_array._new((A.shape[0],), (rows,), d, True)
    if A.ndim == 1:
        a, b = compressed(A), oriented(B, True)
        ip, ix, d = _core._spgemm(a._indptr, a._indices, to_dtype(a._data, loop), b._indptr, b._indices,
                                  to_dtype(b._data, loop), B.shape[1])  # fmt: skip
        return coo_array._new((B.shape[1],), (ix,), d, True)
    shape = (A.shape[0], B.shape[1])
    if A.format == "csc":
        b = oriented(B, False)
        parts = _core._spgemm(b._indptr, b._indices, to_dtype(b._data, loop), A._indptr, A._indices,
                              to_dtype(A._data, loop), A.shape[0], swap=True)  # fmt: skip
        return csc_array._new(shape, *parts)
    a, b = oriented(A, True), oriented(B, True)
    parts = _core._spgemm(a._indptr, a._indices, to_dtype(a._data, loop), b._indptr, b._indices,
                          to_dtype(b._data, loop), B.shape[1])  # fmt: skip
    return csr_array._new(shape, *parts)


# ---- the element-wise functions (methods only, f(0) = 0) ----

# the method names are the array's (decision SP46; scipy's arcsin, arctan, arcsinh, arctanh and rint
# are asin, atan, asinh, atanh and round here, with no aliases); each is the array function of that name
FUNCTIONS = {
    "sin": "sin", "tan": "tan", "asin": "asin", "atan": "atan", "sinh": "sinh", "tanh": "tanh",
    "asinh": "asinh", "atanh": "atanh", "sqrt": "sqrt", "expm1": "expm1", "log1p": "log1p",
    "floor": "floor", "ceil": "ceil", "trunc": "trunc", "round": "round", "sign": "sign",
    "deg2rad": "deg2rad", "rad2deg": "rad2deg",
}  # fmt: skip


def function(A, name):
    g = getattr(_sd, FUNCTIONS[name])
    f = lambda v: C.renamed(name, g, v)  # noqa: E731
    return with_values(A, f, dedupe=True, dok_csr=True)
