"""The direct solves of sd.sparse (docs/api/sparse.md §5, docs/design/sparse.md §3.6): spsolve,
splu (its factor SparseLU), factorized, inv, spsolve_triangular, is_sptriangular, spbandwidth.

The LU is Accelerate's sparse LU with AMD ordering (decisions SP18, SP59; no ordering option and
none of SuperLU's options, SP61), behind stardust._core.sparse._lu_factor (csrc/sparse/direct.cpp).
A factor is checked for singularity once (SP62): Accelerate's structural singularity, a solve that
is not finite, and the estimated condition number kappa_1 with kappa_1 * eps >= 1, all
LinAlgError("<func>(): singular matrix"). Several right-hand sides are solved together (SP64).

The matrix is a sparse array; the LU always factors its CSC (every format converts, so the format
never changes the bits), and spsolve_triangular reads its CSR. Values become float64 (bool, int64)
or stay complex128; NaN and infinity are a ValueError (as in sd.linalg; spsolve_triangular looks
only at the values it reads, as sd.linalg.solve_triangular). A right-hand side is a dense
sd.ndarray of shape (n,) or (n, k), or for spsolve also a sparse array (the solution is then
sparse, its exact zeros dropped).
"""

from .. import _core as _sd
from . import _common as C
from ._base import sparray
from ._common import _core

# The componentwise backward error at or below which a solve is not refined (SP63).
_REFINE_BERR = 8 * 2.0**-52

_DENSE = {
    "spsolve": "sd.linalg.solve",
    "splu": "sd.linalg.lu",
    "factorized": "sd.linalg.lu",
    "inv": "sd.linalg.inv",
    "spsolve_triangular": "sd.linalg.solve_triangular",
    "expm": "sd.linalg.expm",
    "spbandwidth": "sd.linalg.bandwidth",
    "is_sptriangular": "sd.linalg.bandwidth",
}


def _flag(v, func, what):
    if v is True or v is False:
        return v
    raise TypeError(f"{func}(): '{what}' must be True or False, not {C.type_name(v)}")


def _matrix(A, func, square=True):
    """A 2-D (square) sparse array, else the error that names what to use."""
    if not isinstance(A, sparray):
        hint = f"; for a dense array use {_DENSE[func]}" if func in _DENSE and isinstance(A, _sd.ndarray) else ""
        raise TypeError(f"{func}(): 'A' must be a sparse array, not {C.type_name(A)}{hint}")
    if A.ndim != 2:
        raise ValueError(f"{func}(): 'A' must have 2 dimensions, got {A.ndim}")
    if square and A.shape[0] != A.shape[1]:
        raise ValueError(f"{func}(): 'A' must be square, got {A.shape[0]}x{A.shape[1]}")
    return A


def _float(dtype):
    return _sd.complex128 if dtype == _sd.complex128 else _sd.float64


def _finite(x, func, what):
    if x.size and not bool(_sd.all(_sd.isfinite(x))):
        raise ValueError(f"{func}(): '{what}' contains NaN or infinity")


def _compressed(A, func, dtype=None, by_rows=None, check=True):
    """(A as CSR or CSC, its values in dtype (default: A's float dtype), csr flag). by_rows None
    keeps CSR and CSC as they are (others become CSC); True asks for CSR. check: every stored value
    must be finite."""
    if by_rows is True:
        A = A.tocsr()
    elif A.format not in ("csr", "csc"):
        A = A.tocsc()
    dtype = dtype or _float(A.dtype)
    data = A.data if A.data.dtype == dtype else A.data.astype(dtype)
    if check:
        _finite(data, func, "A")
    return A, data, A.format == "csr"


def _dense_rhs(b, n, func, what="A", rows=None):
    """b: a dense sd.ndarray (n,) or (n, k), finite; bool and int64 become float64."""
    if not isinstance(b, _sd.ndarray):
        raise TypeError(f"{func}(): 'b' must be an sd.ndarray, not {C.type_name(b)}")
    if b.ndim not in (1, 2):
        raise ValueError(f"{func}(): 'b' must have 1 or 2 dimensions, got {b.ndim}")
    if b.shape[0] != n:
        raise ValueError(f"{func}(): {rows or repr(what)} is {n}x{n} but 'b' has {b.shape[0]} rows")
    if b.dtype not in (_sd.float64, _sd.complex128):
        b = b.astype(_sd.float64)
    _finite(b, func, "b")
    return b


class _Factor:
    """An LU factor of a square sparse matrix, or of the 0 x 0 one (no native factor), and the
    matrix itself (its float values) for the refinement step."""

    __slots__ = ("core", "n", "dtype", "A", "_Ac")

    def __init__(self, A, func):
        n = A.shape[0]
        # always CSC: Accelerate's factor of a CSR (its transpose flag) differs in the last bits,
        # so the format would change the result, and its blocked solve is wrong when transposed
        A, data, _ = _compressed(A.tocsc(), func)
        if data is not A.data:
            A = type(A)._new(A.shape, A.indptr, A.indices, data)
        self.n, self.dtype, self.A, self._Ac = n, data.dtype, A, None
        self.core = _core._lu_factor(A.shape, A.indptr, A.indices, data, csr=False, func=func) if n else None

    def solve(self, b, trans, func, refine=True):
        """op(A) x = b for a checked dense b (float64 or complex128), with iterative refinement in
        working precision when the first solve needs it (decision SP63): x += solve(b - op(A) x)
        unless the componentwise backward error max_i |r_i| / (|op(A)| |x| + |b|)_i (LAPACK
        dgerfs's test) is already at most _REFINE_BERR = 8 eps. Measured in SP4: threshold
        pivoting alone left a normwise backward error of 404 eps on saddle/600x150, which one step
        brings under 1 eps; the Laplacians solve to 2-8 eps componentwise at once, and skip the
        second solve (half the time). refine=False: the LU solve alone (the eigensolvers'
        shift-invert operator, applied once a Lanczos or Arnoldi step as scipy's and RSpectra's
        are)."""
        if self.dtype == _sd.complex128 and b.dtype != _sd.complex128:
            b = b.astype(_sd.complex128)
        if self.core is None:
            return b.copy()
        x = self.core.solve(b, trans=trans, func=func)
        if not refine:
            return x
        r, berr = _core._residual(
            self.A.indptr, self.A.indices, self._data_as(x.dtype), x, b if C.contiguous(b) else b.copy(), trans=trans
        )
        if berr <= _REFINE_BERR:
            return x
        return x + self.core.solve(r, trans=trans, func=func)

    def _data_as(self, dtype):
        """A's values in dtype (a complex right-hand side of a real factor: complex values, kept)."""
        data = self.A.data
        if data.dtype == dtype:
            return data
        if self._Ac is None:
            self._Ac = data.astype(dtype)
        return self._Ac


class SparseLU:
    """The LU factorization of a square sparse matrix (splu(A)): Accelerate's LU with AMD
    ordering. `solve(b, trans='N' | 'T' | 'H')` solves A x = b, A^T x = b or A^H x = b for b of
    shape (n,) or (n, k); `shape` is A's. The factors themselves (scipy SuperLU's L, U, perm_r,
    perm_c, nnz) are not available: Accelerate keeps them opaque."""

    __slots__ = ("_factor", "_shape")

    def __init__(self, *args, **kwargs):
        raise TypeError("SparseLU(): use sd.sparse.splu(A)")

    @classmethod
    def _new(cls, factor, shape):
        obj = cls.__new__(cls)
        obj._factor, obj._shape = factor, shape
        return obj

    @property
    def shape(self):
        return self._shape

    def solve(self, b, /, *, trans="N"):
        func = "solve"
        if not isinstance(trans, str):
            raise TypeError(f"{func}(): 'trans' must be 'N', 'T' or 'H', not {C.type_name(trans)}")
        if trans not in ("N", "T", "H"):
            raise ValueError(f"{func}(): 'trans' must be 'N', 'T' or 'H', not {trans!r}")
        n = self._shape[0]
        b = _dense_rhs(b, n, func, rows="the factor")
        return self._factor.solve(b, trans, func)

    def __repr__(self):
        return f"SparseLU(shape={self._shape}, dtype={self._factor.dtype})"


def splu(A, /):
    """The LU factorization of the square sparse matrix A (a SparseLU); a LinAlgError when A is
    singular (structurally, a solve that is not finite, or an estimated kappa_1 >= 1/eps)."""
    func = "splu"
    A = _matrix(A, func)
    return SparseLU._new(_Factor(A, func), A.shape)


def factorized(A, /):
    """splu(A).solve: a function of b (and trans=) that solves A x = b."""
    func = "factorized"
    A = _matrix(A, func)
    return SparseLU._new(_Factor(A, func), A.shape).solve


def _block_columns(n):
    """Columns of a sparse right-hand side solved together: fixed by n alone (so the bits are),
    about 2^22 values a block, 1 to 64 columns."""
    return max(1, min(64, (1 << 22) // max(n, 1)))


def _sparse_solve(factor, B, fmt, func):
    """The sparse solution of A X = B for a sparse B: B's columns solved in fixed blocks, each
    block's exact zeros dropped (NaN stays); the format of A (CSR or CSC; CSR for a 1-D B)."""
    from ._compressed import csc_array, csr_array
    from ._construct import hstack

    dtype = factor.dtype if B.dtype != _sd.complex128 else _sd.complex128
    if B.ndim == 1:
        x = factor.solve(_dense_rhs(B.toarray(), factor.n, func), "N", func)
        return csr_array(x)
    B = B.tocsc()
    m, k = B.shape
    if k == 0 or m == 0:
        out = csc_array((m, k), dtype=dtype)
        return out if fmt == "csc" else out.tocsr()
    step = _block_columns(m)
    blocks = []
    for j0 in range(0, k, step):
        part = _dense_rhs(B[:, j0 : min(k, j0 + step)].toarray(), factor.n, func)
        blocks.append(csc_array(factor.solve(part, "N", func)))
    out = blocks[0] if len(blocks) == 1 else hstack(blocks, format="csc")
    if out.dtype != dtype:
        out = out.astype(dtype)
    return out if fmt == "csc" else out.tocsr()


def spsolve(A, b, /):
    """x with A x = b for the square sparse A: b dense (n,) or (n, k) gives x of b's shape; b
    sparse gives a sparse x in A's format (CSR or CSC; COO and DOK become CSC)."""
    func = "spsolve"
    A = _matrix(A, func)
    n = A.shape[0]
    if isinstance(b, sparray):
        if b.ndim not in (1, 2):
            raise ValueError(f"{func}(): 'b' must have 1 or 2 dimensions, got {b.ndim}")
        if b.shape[0] != n:
            raise ValueError(f"{func}(): 'A' is {n}x{n} but 'b' has {b.shape[0]} rows")
        _finite(b.data, func, "b")
        fmt = "csr" if A.format == "csr" else "csc"
        return _sparse_solve(_Factor(A, func), b, fmt, func)
    if not isinstance(b, _sd.ndarray):
        raise TypeError(f"{func}(): 'b' must be an sd.ndarray or a sparse array, not {C.type_name(b)}")
    b = _dense_rhs(b, n, func)
    return _Factor(A, func).solve(b, "N", func)


def inv(A, /):
    """The inverse of the square sparse A as a sparse array of A's format (CSR or CSC; COO and
    DOK give CSC), its exact zeros dropped: the solution of A X = I. Usually nearly dense."""
    func = "inv"
    from ._construct import eye_array

    A = _matrix(A, func)
    n = A.shape[0]
    fmt = "csr" if A.format == "csr" else "csc"
    factor = _Factor(A, func)
    return _sparse_solve(factor, eye_array(n, format="csc", dtype=factor.dtype), fmt, func)


def spsolve_triangular(A, b, /, *, upper, unit_diagonal=False):
    """x with A x = b for the triangular sparse A, by substitution: the upper (upper=True) or lower
    triangle of A is read, the other entries are ignored; unit_diagonal=True takes the diagonal as
    ones without reading it. `upper` is required, as in sd.linalg.solve_triangular (decision SP68;
    scipy's is lower=True). b dense (n,) or (n, k); x has b's shape. A missing or zero diagonal
    entry is LinAlgError("spsolve_triangular(): singular matrix")."""
    func = "spsolve_triangular"
    upper = _flag(upper, func, "upper")
    unit_diagonal = _flag(unit_diagonal, func, "unit_diagonal")
    A = _matrix(A, func)
    n = A.shape[0]
    b = _dense_rhs(b, n, func)
    dtype = _sd.complex128 if _sd.complex128 in (A.dtype, b.dtype) else _sd.float64
    A, data, _ = _compressed(A, func, dtype=dtype, by_rows=True, check=False)
    if data.size and not bool(_sd.all(_sd.isfinite(data))):
        # only the values read must be finite: the chosen triangle, and the diagonal unless
        # unit_diagonal (sd.linalg.solve_triangular's rule; the others may be NaN)
        rows = _core._expand(A.indptr, n, data.shape[0])
        cols = A.indices
        read = (cols != rows) if unit_diagonal else (cols == cols)
        read = read & ((cols >= rows) if upper else (cols <= rows))
        if bool(_sd.any(read & ~_sd.isfinite(data))):
            raise ValueError(f"{func}(): 'A' contains NaN or infinity")
    if b.dtype != dtype:
        b = b.astype(dtype)
    if not C.contiguous(b):
        b = b.copy()
    return _core._sptrsv(A.indptr, A.indices, data, b, lower=not upper, unit_diagonal=unit_diagonal, func=func)


def _pattern(A, func):
    A = _matrix(A, func, square=False)
    if A.format == "csc":
        return A.indptr, A.indices, True
    A = A.tocsr()
    return A.indptr, A.indices, False


def _bandwidth(A, func):
    indptr, indices, swapped = _pattern(A, func)
    below, above = _core._bandwidth(indptr, indices)
    return (above, below) if swapped else (below, above)


def spbandwidth(A, /):
    """BandwidthResult(lower, upper), sd.linalg.bandwidth's result (a tuple, scipy's (below,
    above)): the farthest stored diagonal below and above the main one (0 when none; stored zeros
    count)."""
    from .._results import BandwidthResult

    return BandwidthResult(*_bandwidth(A, "spbandwidth"))


def is_sptriangular(A, /):
    """(lower, upper): whether every stored entry is on or below, on or above the diagonal
    (stored zeros count; both for a diagonal pattern)."""
    indptr, indices, swapped = _pattern(A, "is_sptriangular")
    lower, upper = _core._triangular(indptr, indices)
    return (upper, lower) if swapped else (lower, upper)


__all__ = [
    "spsolve", "splu", "factorized", "inv", "spsolve_triangular", "is_sptriangular", "spbandwidth", "SparseLU",
]  # fmt: skip
