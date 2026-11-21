"""The iterative solvers of sd.sparse (docs/api/sparse.md §6, docs/design/sparse.md §3.7): cg,
bicgstab, gmres, minres, lsqr, lsmr.

scipy 1.18.1's algorithms and stopping tests (SP16), written by this library on its own SpMV and
vector kernels (stardust._core.sparse._cg ... _lsmr, csrc/sparse/iterative.cpp); Accelerate's
iterative solvers are not used. The names follow this library (SP73: max_iter; SP74: no callback,
show, check or calc_var); restart and max_iter of gmres mean what they mean in scipy (SP75); the
result is a read-only object that unpacks into scipy's tuple (SP72, sparse/_result.py).

The operator A (and the preconditioner M) is a sparse array, a 2-D sd.ndarray or a LinearOperator.
A matrix (also one inside a LinearOperator from aslinearoperator) runs in C++ for the whole solve:
its CSR arrays go to the kernel, a dense matrix as the CSR of all its entries (SP80). Another
LinearOperator is called once per product through Python (its matvec, and rmatvec for lsqr and
lsmr). The system's dtype is complex128 when any of A, b, x0, M is complex, float64 otherwise
(SP78); bool and int64 become float64.
"""

import math
import numbers
import operator

from .. import _core as _sd
from . import _common as C
from ._base import sparray
from ._common import _core
from ._interface import LinearOperator
from ._result import BicgstabResult, CgResult, GmresResult, LsmrResult, LsqrResult, MinresResult

# ---------------------------------------------------------------- arguments


def _real(v, func, arg):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {C.type_name(v)}")
    try:
        return float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None


def _tol(v, func, arg):
    """A tolerance: a finite number >= 0 (sd.optimize's rule and message)."""
    x = _real(v, func, arg)
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): '{arg}' must be a finite number >= 0")
    return x


def _finite(v, func, arg):
    x = _real(v, func, arg)
    if not math.isfinite(x):
        raise ValueError(f"{func}(): '{arg}' must be finite")
    return x


def _limit(v, func, arg):
    """conlim: a number >= 0, infinity allowed (0 and infinity: no limit, as scipy's ctol = 0)."""
    x = _real(v, func, arg)
    if not x >= 0:
        raise ValueError(f"{func}(): '{arg}' must be a number >= 0")
    return x


def _count(v, func, arg, default):
    """None (the default) or an int >= 1."""
    if v is None:
        return default
    if not C.is_int(v):
        raise TypeError(f"{func}(): '{arg}' must be an int or None, not {C.type_name(v)}")
    v = operator.index(v)
    if v < 1:
        raise ValueError(f"{func}(): '{arg}' must be >= 1")
    if v >= 2**63:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _check_finite(x, func, what):
    if x.size and not bool(_sd.all(_sd.isfinite(x))):
        raise ValueError(f"{func}(): '{what}' contains NaN or infinity")


def _float_dtype(dtype):
    return _sd.complex128 if dtype == _sd.complex128 else _sd.float64


class _Operand:
    """A, M: a matrix (sparse or dense; also inside an aslinearoperator) or another LinearOperator."""

    __slots__ = ("matrix", "op", "shape", "dtype")

    def __init__(self, A, func, what):
        self.matrix = self.op = None
        if isinstance(A, LinearOperator):
            inner = getattr(A, "_matrix", None)
            if inner is not None:
                A = inner
            else:
                self.op = A
                self.shape, self.dtype = tuple(A.shape), _float_dtype(A.dtype)
                return
        if isinstance(A, (sparray, _sd.ndarray)):
            if isinstance(A, sparray) and A.format == "dok":
                A = A.tocsr()
            if A.ndim != 2:
                raise ValueError(f"{func}(): '{what}' must have 2 dimensions, got {A.ndim}")
            if A.dtype == _sd.string:
                raise TypeError(f"{func}(): '{what}' must hold numbers, not strings")
            _check_finite(A.data if isinstance(A, sparray) else A, func, what)
            self.matrix = A
            self.shape, self.dtype = tuple(A.shape), _float_dtype(A.dtype)
            return
        raise TypeError(
            f"{func}(): '{what}' must be a sparse array, an sd.ndarray or a LinearOperator, not {C.type_name(A)}"
        )

    def native(self, dtype, adjoint=False):
        """The operator for the kernel: the CSR arrays (indptr, indices, data) of the matrix (or
        of its adjoint) in dtype, or a Python function of a vector."""
        A = self.matrix
        if A is None:
            return _python(self.op.rmatvec if adjoint else self.op.matvec, dtype)
        if isinstance(A, sparray):
            if adjoint:
                A = A.tocsr().T.tocsr()  # the transpose's CSR (its values in A's order of columns)
                if A.dtype == _sd.complex128:
                    A = A.conj()
            else:
                A = A.tocsr()
            data = A.data if A.data.dtype == dtype else A.data.astype(dtype)
            if not C.contiguous(data):
                data = data.copy()
            return (A.indptr, A.indices, data)
        # a dense matrix: the CSR of all its entries (zeros stored), so its products are SpMVs
        if adjoint:
            A = _sd.conj(A.mT) if A.dtype == _sd.complex128 else A.mT
        m, n = A.shape
        indptr = _sd.arange(0, m * n + 1, max(n, 1), dtype=_sd.int64) if n else _sd.zeros(m + 1, dtype=_sd.int64)
        indices = _sd.reshape(_sd.tile(_sd.arange(n, dtype=_sd.int64), (m,)), (m * n,))
        data = _sd.reshape(A.astype(dtype), (m * n,)).copy()
        return (indptr, indices, data)


def _python(f, dtype):
    """A LinearOperator's product for the kernel: a new C-contiguous vector of the system's dtype
    (a real result widens to complex; a complex one in a real system is a TypeError)."""

    def call(x):
        y = f(x)
        if y.ndim != 1:
            y = _sd.reshape(y, (y.size,))
        if y.dtype != dtype:
            if y.dtype == _sd.complex128:
                raise TypeError(
                    "the operator returned complex128 values for a float64 system; give it dtype=complex128"
                )
            y = y.astype(dtype)
        if not C.contiguous(y):
            y = y.copy()
        return y

    return call


def _vector(v, n, func, what, rows_of="A"):
    """b or x0: a dense sd.ndarray of shape (n,) or (n, 1)."""
    if not isinstance(v, _sd.ndarray):
        raise TypeError(f"{func}(): '{what}' must be an sd.ndarray, not {C.type_name(v)}")
    if v.dtype == _sd.string:
        raise TypeError(f"{func}(): '{what}' must hold numbers, not strings")
    if tuple(v.shape) not in ((n,), (n, 1)):
        raise ValueError(f"{func}(): '{what}' must have shape ({n},) or ({n}, 1) for {rows_of}, got {tuple(v.shape)}")
    _check_finite(v, func, what)
    return v


def _as(v, dtype):
    v = _sd.reshape(v, (v.shape[0],)) if v.ndim == 2 else v
    if v.dtype != dtype:
        v = v.astype(dtype)
    return v if C.contiguous(v) else v.copy()


def _dtype(*dtypes):
    return _sd.complex128 if any(d == _sd.complex128 for d in dtypes) else _sd.float64


def _square(A, b, x0, M, func):
    """The checked system of cg, bicgstab, gmres, minres: (A, M operands, b, x0, n, dtype)."""
    A = _Operand(A, func, "A")
    m, n = A.shape
    if m != n:
        raise ValueError(f"{func}(): 'A' must be square, got {m}x{n}")
    b = _vector(b, n, func, "b", f"'A' of shape {A.shape}")
    if x0 is not None:
        x0 = _vector(x0, n, func, "x0", f"'A' of shape {A.shape}")
    if M is not None:
        M = _Operand(M, func, "M")
        if M.shape != A.shape:
            raise ValueError(f"{func}(): 'M' must have the shape of 'A' {A.shape}, got {M.shape}")
    dtype = _dtype(A.dtype, b.dtype, *(() if x0 is None else (x0.dtype,)), *(() if M is None else (M.dtype,)))
    return A, M, _as(b, dtype), None if x0 is None else _as(x0, dtype), n, dtype


# ---------------------------------------------------------------- the results

_EXIT = "the residual norm is within the tolerance"
_LIMIT = "the iteration limit was reached"
_ZERO_B = "the right-hand side is zero: x = 0"

_MINRES = {
    -1: "beta2 = 0: if M = I, b and x are eigenvectors",
    0: "beta1 = 0: the exact solution is x0",
    1: "a solution to Ax = b was found, given rtol",
    2: "a least-squares solution was found, given rtol",
    3: "reasonable accuracy achieved, given eps",
    4: "the estimate of cond(A) has exceeded 0.1/eps",
    6: _LIMIT,
}

# lsqr's and lsmr's messages (scipy's), by istop
_LSQ = (
    "the exact solution is x = 0",
    "Ax - b is small enough, given atol, btol",
    "the least-squares solution is good enough, given atol",
    "the estimate of cond(Abar) has exceeded conlim",
    "Ax - b is small enough for this machine",
    "the least-squares solution is good enough for this machine",
    "cond(Abar) seems to be too large for this machine",
    _LIMIT,
)


def _message(func, d, bnorm_zero):
    status = d["status"]
    if func in ("lsqr", "lsmr"):
        if status == 0 and func == "lsmr":
            return "the exact solution is x = 0, or x = x0 if x0 was given"
        return _LSQ[status]
    if bnorm_zero:
        return _ZERO_B
    if func == "minres":
        return _MINRES[d["stop"]]
    if status == 0:
        return _EXIT
    if status == -10:
        return "breakdown: rho (the inner product of the shadow and the residual) is zero"
    if status == -11:
        return "breakdown: omega or the inner product of the shadow and A p is zero"
    if func == "gmres" and d["stop"] == 1:
        return "breakdown: the Krylov space holds the solution, but the residual norm is above the tolerance"
    return _LIMIT


def _fields(func, d, bnorm_zero=False):
    status = int(d["status"])
    if func in ("lsqr", "lsmr"):
        converged = status in (0, 1, 2, 4, 5)
    else:
        converged = status == 0
    out = {"x": d["x"], "converged": converged, "status": status, "message": _message(func, d, bnorm_zero),
           "n_iter": int(d["n_iter"])}  # fmt: skip
    return out


def _run(func, A, b, /, **kw):
    """(result, history): the solver's result and the residual norms it tracked step by step (a
    float64 array; the tests compare it with scipy's), after func's checks."""
    return _IMPL[func](A, b, **kw)


# ---------------------------------------------------------------- the solvers


def _finish(cls, func, d, extra, bnorm_zero=False):
    return cls._make(**_fields(func, d, bnorm_zero), **extra), d["history"]


def _bzero(b):
    return not bool(_sd.any(b != 0))


def _cg(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None):
    """Solve A x = b by the conjugate gradient method, for a Hermitian positive definite A (scipy's
    cg): stops when ||b - A x|| < max(rtol ||b||, atol) (its recurrence's residual) or after
    max_iter iterations (default 10 n). M approximates the inverse of A (Hermitian positive
    definite). The result unpacks as (x, status): status 0 when converged, else max_iter."""
    func = "cg"
    rtol, atol = _tol(rtol, func, "rtol"), _tol(atol, func, "atol")
    A, M, b, x0, n, dtype = _square(A, b, x0, M, func)
    max_iter = _count(max_iter, func, "max_iter", 10 * n)
    d = _core._cg(A.native(dtype), None if M is None else M.native(dtype), b, x0, rtol=rtol, atol=atol,
                  max_iter=max_iter)  # fmt: skip
    return _finish(CgResult, func, d, {"normr": d["normr"]}, _bzero(b))


def _bicgstab(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None):
    """Solve A x = b by BiCGSTAB (scipy's bicgstab): stops when ||b - A x|| < max(rtol ||b||, atol)
    (its recurrence's residual) or after max_iter iterations (default 10 n). The result unpacks
    as (x, status): 0 when converged, max_iter at the limit, -10 or -11 at a breakdown."""
    func = "bicgstab"
    rtol, atol = _tol(rtol, func, "rtol"), _tol(atol, func, "atol")
    A, M, b, x0, n, dtype = _square(A, b, x0, M, func)
    max_iter = _count(max_iter, func, "max_iter", 10 * n)
    d = _core._bicgstab(A.native(dtype), None if M is None else M.native(dtype), b, x0, rtol=rtol, atol=atol,
                        max_iter=max_iter)  # fmt: skip
    return _finish(BicgstabResult, func, d, {"normr": d["normr"]}, _bzero(b))


def _gmres(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, restart=None, max_iter=None, M=None):
    """Solve A x = b by restarted GMRES (scipy's gmres, its left preconditioning by M): stops when
    ||b - A x|| <= max(rtol ||b||, atol) (the true residual, after each restart cycle) or after
    max_iter cycles (default 10 n) of restart inner iterations (default 20, at most n). The result
    unpacks as (x, status): 0 when converged, else max_iter."""
    func = "gmres"
    rtol, atol = _tol(rtol, func, "rtol"), _tol(atol, func, "atol")
    A, M, b, x0, n, dtype = _square(A, b, x0, M, func)
    max_iter = _count(max_iter, func, "max_iter", 10 * n)
    restart = min(_count(restart, func, "restart", 20), max(n, 1))
    d = _core._gmres(A.native(dtype), None if M is None else M.native(dtype), b, x0, rtol=rtol, atol=atol,
                     restart=restart, max_iter=max_iter)  # fmt: skip
    return _finish(GmresResult, func, d, {"n_inner": int(d["n_inner"]), "normr": d["normr"]}, _bzero(b))


def _minres(A, b, /, *, x0=None, rtol=1e-05, shift=0.0, max_iter=None, M=None):
    """Solve (A - shift I) x = b by MINRES for a Hermitian (possibly indefinite) A (scipy's minres,
    Paige and Saunders): stops at scipy's tests on its estimates of ||r|| / (||A|| ||x||) and
    ||A r|| / (||A|| ||r||) against rtol, or after max_iter iterations (default 5 n). M is a
    Hermitian positive definite preconditioner. The result unpacks as (x, status): 0, or max_iter
    at the limit; the message says which test stopped it."""
    func = "minres"
    rtol, shift = _tol(rtol, func, "rtol"), _finite(shift, func, "shift")
    A, M, b, x0, n, dtype = _square(A, b, x0, M, func)
    max_iter = _count(max_iter, func, "max_iter", 5 * n)
    d = _core._minres(A.native(dtype), None if M is None else M.native(dtype), b, x0, rtol=rtol, shift=shift,
                      max_iter=max_iter)  # fmt: skip
    extra = {k: d[k] for k in ("normr", "norma", "conda", "normar", "normx")}
    return _finish(MinresResult, func, d, extra, _bzero(b))


def _lsq(func, A, b, damp, atol, btol, conlim, max_iter, x0):
    """(result, history) of lsqr or lsmr."""
    damp, atol, btol = _tol(damp, func, "damp"), _tol(atol, func, "atol"), _tol(btol, func, "btol")
    conlim = _limit(conlim, func, "conlim")
    A = _Operand(A, func, "A")
    m, n = A.shape
    b = _vector(b, m, func, "b", f"'A' of shape {A.shape}")
    if x0 is not None:
        x0 = _vector(x0, n, func, "x0", f"the columns of 'A' of shape {A.shape}")
    dtype = _dtype(A.dtype, b.dtype, *(() if x0 is None else (x0.dtype,)))
    b = _as(b, dtype)
    x0 = None if x0 is None else _as(x0, dtype)
    max_iter = _count(max_iter, func, "max_iter", 2 * n if func == "lsqr" else min(m, n))
    kernel = _core._lsqr if func == "lsqr" else _core._lsmr
    d = kernel(A.native(dtype), A.native(dtype, adjoint=True), b, x0, n=n, damp=damp, atol=atol, btol=btol,
               conlim=conlim, max_iter=max_iter)  # fmt: skip
    if func == "lsqr":
        extra = {"normr": d["normr"], "normr_damped": d["normr2"], "norma": d["norma"], "conda": d["conda"],
                 "normar": d["normar"], "normx": d["normx"], "var": d["var"]}  # fmt: skip
        return _finish(LsqrResult, func, d, extra)
    extra = {k: d[k] for k in ("normr", "normar", "norma", "conda", "normx")}
    return _finish(LsmrResult, func, d, extra)


def lsqr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None):
    """The least-squares solution of A x = b, min ||A x - b||^2 + damp^2 ||x - x0||^2, by LSQR
    (scipy's lsqr, Paige and Saunders 1982) for any A (m x n). Stops at scipy's tests with atol,
    btol and conlim, or after max_iter iterations (default 2 n). The result unpacks as scipy's
    (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var) = (x, status, n_iter, normr,
    normr_damped, norma, conda, normar, normx, var); var (the diagonal of (A^H A + damp^2 I)^-1's
    estimate) is always computed."""
    return _lsq("lsqr", A, b, damp, atol, btol, conlim, max_iter, x0)[0]


def lsmr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None):
    """The least-squares solution of A x = b, min ||A x - b||^2 + damp^2 ||x - x0||^2, by LSMR
    (scipy's lsmr, Fong and Saunders 2011) for any A (m x n). Stops at scipy's tests with atol,
    btol and conlim, or after max_iter iterations (default min(m, n)). The result unpacks as
    scipy's (x, istop, itn, normr, normar, norma, conda, normx) = (x, status, n_iter, normr,
    normar, norma, conda, normx)."""
    return _lsq("lsmr", A, b, damp, atol, btol, conlim, max_iter, x0)[0]


def _lsqr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None):
    return _lsq("lsqr", A, b, damp, atol, btol, conlim, max_iter, x0)


def _lsmr(A, b, /, *, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, max_iter=None, x0=None):
    return _lsq("lsmr", A, b, damp, atol, btol, conlim, max_iter, x0)


def cg(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None):
    return _cg(A, b, x0=x0, rtol=rtol, atol=atol, max_iter=max_iter, M=M)[0]


def bicgstab(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, max_iter=None, M=None):
    return _bicgstab(A, b, x0=x0, rtol=rtol, atol=atol, max_iter=max_iter, M=M)[0]


def gmres(A, b, /, *, x0=None, rtol=1e-05, atol=0.0, restart=None, max_iter=None, M=None):
    return _gmres(A, b, x0=x0, rtol=rtol, atol=atol, restart=restart, max_iter=max_iter, M=M)[0]


def minres(A, b, /, *, x0=None, rtol=1e-05, shift=0.0, max_iter=None, M=None):
    return _minres(A, b, x0=x0, rtol=rtol, shift=shift, max_iter=max_iter, M=M)[0]


for _public, _private in ((cg, _cg), (bicgstab, _bicgstab), (gmres, _gmres), (minres, _minres)):
    _public.__doc__ = _private.__doc__

_IMPL = {"cg": _cg, "bicgstab": _bicgstab, "gmres": _gmres, "minres": _minres, "lsqr": _lsqr, "lsmr": _lsmr}

__all__ = ["cg", "gmres", "minres", "bicgstab", "lsqr", "lsmr"]
