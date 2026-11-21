"""LinearOperator and aslinearoperator (docs/api/sparse.md §5): a matrix known only by what it does
to vectors, scipy.sparse.linalg's interface on sd arrays.

    LinearOperator(shape, matvec, /, *, rmatvec=None, matmat=None, rmatmat=None, dtype=None)

shape and matvec are positional, the rest keywords (decision SP65). matvec(x) takes an sd.ndarray of
shape (N,) (or (N, 1)) and returns the product (M,) (or (M, 1)); rmatvec the product with A^H;
matmat and rmatmat the same for (N, k) and (M, k) (by default column by column). dtype=None asks
matvec once, on zeros, as scipy does. A subclass may instead define _matvec / _matmat (and
_rmatvec / _rmatmat) and call LinearOperator.__init__(self, shape, None, dtype=...).

The operators combine as scipy's: A @ x (a vector or a matrix), A @ B, A + B, A - B, -A, c * A,
A * c, A ** p (p an int >= 0, square A), A.T, A.H and A.adjoint(). `*` takes a scalar only (A @ x
is the product). aslinearoperator(A) takes a sparse array, a 2-D sd.ndarray or an operator.
"""

import numbers

from .. import _calls
from .. import _core as _sd
from . import _common as C
from ._base import sparray

_DTYPES = (_sd.bool, _sd.int64, _sd.float64, _sd.complex128)


def _promote(*types):
    """The result dtype of the operands' dtypes and Python scalars (the array's rules)."""
    return _sd.result_type(*types)


def _conj(x):
    return _sd.conj(x) if x.dtype == _sd.complex128 else x


def _as_result(y, func, shape):
    """The user's function's result as an sd.ndarray of `shape` (an equal size reshaped)."""
    if not isinstance(y, _sd.ndarray):
        y = C.renamed(func, _sd.asarray, y)
    if tuple(y.shape) != shape:
        size = 1
        for d in shape:
            size *= d
        if y.size != size:
            raise ValueError(f"{func}(): the operator's function returned shape {tuple(y.shape)}, expected {shape}")
        y = y.reshape(shape)
    return y


def _check_vector(x, n, func):
    if not isinstance(x, _sd.ndarray):
        raise TypeError(f"{func}(): 'x' must be an sd.ndarray, not {C.type_name(x)}")
    if tuple(x.shape) not in ((n,), (n, 1)):
        raise ValueError(f"{func}(): 'x' must have shape ({n},) or ({n}, 1), got {tuple(x.shape)}")


def _check_matrix(X, n, func):
    if not isinstance(X, _sd.ndarray):
        raise TypeError(f"{func}(): 'X' must be an sd.ndarray, not {C.type_name(X)}")
    if X.ndim != 2:
        raise ValueError(f"{func}(): 'X' must have 2 dimensions, got {X.ndim}")
    if X.shape[0] != n:
        raise ValueError(f"{func}(): the operator has {n} columns but 'X' has {X.shape[0]} rows")


def _is_scalar(c):
    return isinstance(c, numbers.Number) and not isinstance(c, bool) and not isinstance(c, _sd.ndarray)


class LinearOperator:
    """A linear map of shape (M, N) given by its product with vectors (scipy's LinearOperator)."""

    @_calls.method("LinearOperator")
    def __init__(self, shape, matvec, /, *, rmatvec=None, matmat=None, rmatmat=None, dtype=None):
        func = "LinearOperator"
        self._shape = C.shape_arg(shape, func, ndims=(2,))
        own = {}
        for name, f in (("matvec", matvec), ("rmatvec", rmatvec), ("matmat", matmat), ("rmatmat", rmatmat)):
            if f is not None and not callable(f):
                raise TypeError(f"{func}(): '{name}' must be callable or None, not {C.type_name(f)}")
            own[name] = f
        cls = type(self)
        defines = cls._matvec is not LinearOperator._matvec or cls._matmat is not LinearOperator._matmat
        if matvec is None and not defines:
            raise TypeError(f"{func}(): 'matvec' must be callable, not NoneType")
        self._f = own
        self._matrix = None
        if dtype is None:
            n = self._shape[1]
            dtype = self.matvec(_sd.zeros(n)).dtype
        else:
            dtype = C.dtype_arg(dtype, func)
        self._dtype = dtype

    # ---- what a subclass may define ----

    def _matvec(self, x):
        f = self._f.get("matvec")
        if f is not None:
            return f(x)
        return self._matmat(x.reshape((self._shape[1], 1)))

    def _matmat(self, X):
        f = self._f.get("matmat")
        if f is not None:
            return f(X)
        m, k = self._shape[0], X.shape[1]
        if k == 0:
            return _sd.zeros((m, 0), dtype=self.dtype)
        cols = [_as_result(self._matvec(X[:, j]), "matmat", (m,)) for j in range(k)]
        return _sd.stack(cols, axis=1)

    def _rmatvec(self, x):
        f = self._f.get("rmatvec")
        if f is not None:
            return f(x)
        if type(self)._rmatmat is not LinearOperator._rmatmat or self._f.get("rmatmat") is not None:
            return self._rmatmat(x.reshape((self._shape[0], 1)))
        raise ValueError("rmatvec(): the operator was made without 'rmatvec'")

    def _rmatmat(self, X):
        f = self._f.get("rmatmat")
        if f is not None:
            return f(X)
        n, k = self._shape[1], X.shape[1]
        if k == 0:
            return _sd.zeros((n, 0), dtype=self.dtype)
        cols = [_as_result(self._rmatvec(X[:, j]), "rmatmat", (n,)) for j in range(k)]
        return _sd.stack(cols, axis=1)

    # ---- attributes ----

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def T(self):
        return _transposed(self)

    @property
    def H(self):
        return _adjoint(self)

    def adjoint(self):
        return _adjoint(self)

    # ---- products ----

    def matvec(self, x, /):
        func = "matvec"
        m, n = self._shape
        _check_vector(x, n, func)
        out = (m,) if x.ndim == 1 else (m, 1)
        return _as_result(self._matvec(x), func, out)

    def rmatvec(self, x, /):
        func = "rmatvec"
        m, n = self._shape
        _check_vector(x, m, func)
        out = (n,) if x.ndim == 1 else (n, 1)
        return _as_result(self._rmatvec(x), func, out)

    def matmat(self, X, /):
        func = "matmat"
        m, n = self._shape
        _check_matrix(X, n, func)
        return _as_result(self._matmat(X), func, (m, X.shape[1]))

    def rmatmat(self, X, /):
        func = "rmatmat"
        m, n = self._shape
        _check_matrix(X, m, func)
        return _as_result(self._rmatmat(X), func, (n, X.shape[1]))

    # ---- operators ----

    def __matmul__(self, other):
        if isinstance(other, LinearOperator):
            return _product(self, other)
        if isinstance(other, _sd.ndarray):
            if other.ndim == 1 or (other.ndim == 2 and other.shape[1] == 1):
                return self.matvec(other)
            if other.ndim == 2:
                return self.matmat(other)
            raise ValueError(f"matmul: expected a 1-D or 2-D operand, got {other.ndim} dimensions")
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, _sd.ndarray):
            if other.ndim == 1 or (other.ndim == 2 and other.shape[0] == 1):
                y = self.T.matvec(other.mT if other.ndim == 2 else other)
                return y.mT if other.ndim == 2 else y
            if other.ndim == 2:
                return self.T.matmat(other.mT).mT
            raise ValueError(f"matmul: expected a 1-D or 2-D operand, got {other.ndim} dimensions")
        return NotImplemented

    def __add__(self, other):
        if not isinstance(other, LinearOperator):
            return NotImplemented
        if self._shape != other._shape:
            raise ValueError(f"cannot add operators of shapes {self._shape} and {other._shape}")
        return _sum(self, other)

    def __sub__(self, other):
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return self + (-other)

    def __neg__(self):
        return _scaled(self, -1)

    def __mul__(self, c):
        return _scaled(self, c) if _is_scalar(c) else NotImplemented

    __rmul__ = __mul__

    def __pow__(self, p):
        if not C.is_int(p):
            return NotImplemented
        if p < 0:
            raise ValueError(f"the power must be an int >= 0, got {p}")
        if self._shape[0] != self._shape[1]:
            raise ValueError(f"a power needs a square operator, got shape {self._shape}")
        return _power(self, int(p))

    def __repr__(self):
        return f"{type(self).__name__}(shape={self._shape}, dtype={self._dtype})"


def _make(shape, dtype, matvec, rmatvec=None, matmat=None, rmatmat=None, matrix=None):
    op = LinearOperator.__new__(LinearOperator)
    op._shape, op._dtype = shape, dtype
    op._f = {"matvec": matvec, "rmatvec": rmatvec, "matmat": matmat, "rmatmat": rmatmat}
    op._matrix = matrix
    return op


def _transposed(A):
    m, n = A._shape
    return _make(
        (n, m), A._dtype,
        lambda x: _conj(A._rmatvec(_conj(x))), lambda x: _conj(A._matvec(_conj(x))),
        lambda X: _conj(A._rmatmat(_conj(X))), lambda X: _conj(A._matmat(_conj(X))),
    )  # fmt: skip


def _adjoint(A):
    m, n = A._shape
    return _make((n, m), A._dtype, A._rmatvec, A._matvec, A._rmatmat, A._matmat)


def _sum(A, B):
    return _make(
        A._shape, _promote(A._dtype, B._dtype),
        lambda x: A.matvec(x) + B.matvec(x), lambda x: A.rmatvec(x) + B.rmatvec(x),
        lambda X: A.matmat(X) + B.matmat(X), lambda X: A.rmatmat(X) + B.rmatmat(X),
    )  # fmt: skip


def _scaled(A, c):
    cc = c.conjugate()
    return _make(
        A._shape, _promote(A._dtype, c),
        lambda x: c * A.matvec(x), lambda x: cc * A.rmatvec(x),
        lambda X: c * A.matmat(X), lambda X: cc * A.rmatmat(X),
    )  # fmt: skip


def _product(A, B):
    if A._shape[1] != B._shape[0]:
        raise ValueError(f"cannot multiply operators of shapes {A._shape} and {B._shape}")
    return _make(
        (A._shape[0], B._shape[1]), _promote(A._dtype, B._dtype),
        lambda x: A.matvec(B.matvec(x)), lambda x: B.rmatvec(A.rmatvec(x)),
        lambda X: A.matmat(B.matmat(X)), lambda X: B.rmatmat(A.rmatmat(X)),
    )  # fmt: skip


def _power(A, p):
    def apply(f, x):
        for _ in range(p):
            x = f(x)
        return x

    return _make(
        A._shape, A._dtype,
        lambda x: apply(A.matvec, x), lambda x: apply(A.rmatvec, x),
        lambda X: apply(A.matmat, X), lambda X: apply(A.rmatmat, X),
    )  # fmt: skip


def aslinearoperator(A, /):
    """A as a LinearOperator: an operator is itself; a sparse array or a 2-D sd.ndarray gives the
    operator of its products (A @ x, and A^H @ x for rmatvec)."""
    func = "aslinearoperator"
    if isinstance(A, LinearOperator):
        return A
    if isinstance(A, sparray):
        if A.ndim != 2:
            raise ValueError(f"{func}(): 'A' must have 2 dimensions, got {A.ndim}")
        AH = A.T.conj() if A.dtype == _sd.complex128 else A.T
    elif isinstance(A, _sd.ndarray):
        if A.ndim != 2:
            raise ValueError(f"{func}(): 'A' must have 2 dimensions, got {A.ndim}")
        AH = _sd.conj(A.mT) if A.dtype == _sd.complex128 else A.mT
    else:
        raise TypeError(
            f"{func}(): 'A' must be a sparse array, an sd.ndarray or a LinearOperator, not {C.type_name(A)}"
        )
    return _make(
        tuple(A.shape), A.dtype,
        lambda x: A @ x, lambda x: AH @ x, lambda X: A @ X, lambda X: AH @ X, matrix=A,
    )  # fmt: skip


__all__ = ["LinearOperator", "aslinearoperator"]
