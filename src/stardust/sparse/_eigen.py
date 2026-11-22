"""The eigensolvers of sd.sparse (docs/api/sparse.md §7, docs/design/sparse.md §3.8): eigsh, eigs, svds.

eigsh and eigs are ARPACK (implicitly restarted Lanczos and Arnoldi) ported in its order of
operations, as scipy 1.18.1 runs it (decision SP83), on this library's SpMV and vector kernels
(stardust._core.sparse._eigsh, _eigs; csrc/sparse/arpack.cpp). svds is Golub-Kahan-Lanczos
bidiagonalization with thick restarts (SP85; _svds, csrc/sparse/svds.cpp), with no solver= option.

The start vector (SP84): v0 when given; else u = 2 g.random(n) - 1 (real and imaginary parts from
consecutive draws for a complex problem: scipy's uniform on [-1, 1)) with g = rng, or
sd.random.default_rng(0) when rng is None, so the same call gives the same result. The vectors
ARPACK asks for later (an invariant subspace found) come from the same generator.

Spectral transformations are scipy's (eigsh modes 'normal', 'buckling', 'cayley': SP86): the
inverse (A - sigma M)^-1 and M^-1 is this library's sparse LU (§5; a dense matrix is factored as
the sparse array of its nonzeros), a LinearOperator is inverted by gmres as scipy does, and Minv /
OPinv replace them. A result that has not converged reports it in its status (SP87): the
converged pairs are the result (scipy's ArpackNoConvergence).
"""

import math
import numbers

from .. import _core as _sd
from . import _common as C
from ._interface import LinearOperator
from ._iterative import _count, _dtype, _Operand, _python, _tol, _vector
from ._result import EigshResult, EigsResult, SvdsResult

_EPS = 2.0**-52
_SYM_WHICH = ("LM", "SM", "LA", "SA", "BE")
_NONSYM_WHICH = ("LM", "SM", "LR", "SR", "LI", "SI")
_SVD_WHICH = ("LM", "SM")
_MODES = ("normal", "buckling", "cayley")


def _choice(v, options, func, arg):
    text = ", ".join(repr(o) for o in options[:-1]) + f" or {options[-1]!r}"
    if not isinstance(v, str):
        raise TypeError(f"{func}(): '{arg}' must be {text}, not {C.type_name(v)}")
    if v not in options:
        raise ValueError(f"{func}(): '{arg}' must be {text}, not {v!r}")
    return v


def _flag(v, func, arg):
    if v is True or v is False:
        return v
    raise TypeError(f"{func}(): '{arg}' must be True or False, not {C.type_name(v)}")


def _int(v, func, arg):
    if not C.is_int(v):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {C.type_name(v)}")
    return int(v)


def _sigma(v, func, real):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, numbers.Number) or isinstance(v, _sd.ndarray):
        kind = "an int or a float" if real else "a number"
        raise TypeError(f"{func}(): 'sigma' must be {kind} or None, not {C.type_name(v)}")
    if real:
        if not isinstance(v, numbers.Real):
            raise TypeError(f"{func}(): 'sigma' must be an int or a float or None, not {C.type_name(v)}")
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"{func}(): 'sigma' must be finite")
        return v
    v = complex(v)
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        raise ValueError(f"{func}(): 'sigma' must be finite")
    return v


def _rng(rng, func):
    if rng is None:
        from ..random import default_rng

        return default_rng(0)
    from ..random import Generator

    if not isinstance(rng, Generator):
        raise TypeError(f"{func}(): 'rng' must be an sd.random.Generator or None, not {C.type_name(rng)}")
    return rng


def _uniform(g, n, dtype):
    """scipy's start vector: uniform on [-1, 1) (a complex one from pairs of draws)."""
    if dtype == _sd.complex128:
        u = g.random(2 * n) * 2.0 - 1.0
        pairs = _sd.reshape(u, (n, 2))
        return (pairs[:, 0] + pairs[:, 1] * 1j).copy()
    return g.random(n) * 2.0 - 1.0


def _drawer(g, n, dtype):
    def draw(size=None):
        return _uniform(g, n if size is None else size, dtype)

    return draw


def _start(v0, rng, n, dtype, func):
    """(v0 as the problem's dtype, the generator of the later vectors)."""
    g = _rng(rng, func)
    if v0 is None:
        v = _uniform(g, n, dtype)
    else:
        v = _vector(v0, n, func, "v0", f"'A' of shape ({n}, {n})" if func != "svds" else "min(A.shape)")
        v = _sd.reshape(v, (n,)) if v.ndim == 2 else v
        if v.dtype != dtype:
            v = v.astype(dtype)
        if not C.contiguous(v):
            v = v.copy()
        if not bool(_sd.any(v != 0)):
            raise ValueError(f"{func}(): 'v0' must not be zero")
    return v, _drawer(g, n, dtype)


def _native(op, dtype, adjoint=False):
    return None if op is None else op.native(dtype, adjoint)


# ---------------------------------------------------------------- the inverses


class _Inverse:
    """x -> M^-1 x for a matrix by the sparse LU, for another operator by gmres (scipy's IterInv
    and IterOpInv: rtol max(tol, 1000 sqrt(n) eps), tol = 2 eps when 0)."""

    def __init__(self, func, matrix=None, op=None, tol=0.0, part=None):
        self.func, self.factor, self.op, self.part = func, None, op, part
        self.rtol = None
        if matrix is not None:
            from ._direct import _Factor

            if isinstance(matrix, _sd.ndarray):  # a dense matrix: the sparse array of its nonzeros
                from ._compressed import csc_array

                matrix = csc_array(matrix)
            self.factor = _Factor(matrix.tocsc(), func)
        else:
            n = op.shape[0]
            t = tol if tol > 0 else 2 * _EPS
            self.rtol = max(t, 1000 * math.sqrt(n) * _EPS)

    def __call__(self, x):
        if self.factor is not None:
            y = self.factor.solve(x, "N", self.func, refine=False)
        else:
            from ._iterative import gmres

            r = gmres(self.op, x, rtol=self.rtol, atol=0.0)
            if not r.converged:
                raise ValueError(
                    f"{self.func}(): solving with the inverse by gmres did not converge (status {r.status})"
                )
            y = r.x
        if self.part == "real":
            y = _sd.real(y)
        elif self.part == "imag":
            y = _sd.imag(y)
        return y


def _shifted(A, M, sigma, dtype):
    """A - sigma M (or A - sigma I) as a sparse array of dtype, from matrices."""
    from ._compressed import csr_array
    from ._construct import eye_array

    def sp(X):
        X = csr_array(X) if isinstance(X, _sd.ndarray) else X.tocsr()
        return X.astype(dtype) if X.dtype != dtype else X

    if sigma == 0:
        return sp(A)
    B = sp(M) if M is not None else eye_array(A.shape[0], dtype=dtype, format="csr")
    return sp(A) - B * (sigma if dtype == _sd.complex128 else sigma.real)


def _op_inverse(A, M, sigma, dtype, func, tol, part=None):
    """x -> (A - sigma M)^-1 x (scipy's get_OPinv_matvec)."""
    if A.matrix is not None and (M is None or M.matrix is not None):
        return _Inverse(func, matrix=_shifted(A.matrix, None if M is None else M.matrix, sigma, dtype), part=part)
    from ._interface import aslinearoperator

    LA = aslinearoperator(A.matrix) if A.matrix is not None else A.op
    if sigma == 0:
        S = LA
    else:
        LM = None if M is None else (aslinearoperator(M.matrix) if M.matrix is not None else M.op)
        s = sigma if dtype == _sd.complex128 else sigma.real
        S = LA - s * LM if LM is not None else LA - s * _identity(A.shape[0], dtype)
    return _Inverse(func, op=S, tol=tol, part=part)


def _identity(n, dtype):
    return LinearOperator((n, n), lambda x: x, rmatvec=lambda x: x, dtype=dtype)


def _operand(X, func, arg, shape):
    if X is None:
        return None
    op = _Operand(X, func, arg)
    if op.shape != shape:
        raise ValueError(f"{func}(): '{arg}' must have the shape of 'A' {shape}, got {op.shape}")
    return op


def _square_a(A, func):
    A = _Operand(A, func, "A")
    m, n = A.shape
    if m != n:
        raise ValueError(f"{func}(): 'A' must be square, got {m}x{n}")
    return A, n


def _counts(k, ncv, n, func, low, dense):
    """k and ncv (eigsh: 1 <= k < n, k < ncv <= n; eigs: 1 <= k < n - 1, k + 1 < ncv <= n)."""
    k = _int(k, func, "k")
    if k < 1:
        raise ValueError(f"{func}(): 'k' must be >= 1")
    top = n if low == 0 else n - 1
    if k >= top:
        raise ValueError(
            f"{func}(): 'k' must be < {'n' if low == 0 else 'n - 1'} = {top} for 'A' of shape ({n}, {n}); "
            f"for all eigenvalues use {dense}(A.toarray())"
        )
    if ncv is None:
        ncv = min(max(2 * k + 1, 20), n)
    else:
        ncv = _int(ncv, func, "ncv")
        if not (k + low < ncv <= n):
            bound = "k" if low == 0 else "k + 1"
            raise ValueError(f"{func}(): 'ncv' must satisfy {bound} < ncv <= n = {n}, got {ncv}")
    return k, ncv


_STATUS = {
    1: "the iteration limit was reached",
    3: "no shifts could be applied in a restart (a larger ncv may help)",
    -9999: "the Krylov factorization could not be extended (an invariant subspace, then three failed restarts)",
}


def _message(status, got, k, what="eigenvalue"):
    """The result's message; `what` singular ("eigenvalue", "singular value"), plural after a count
    other than 1."""
    if status == 0:
        return f"the {k} wanted {what if k == 1 else what + 's'} converged"
    return f"{_STATUS[status]}: {got} of {k} {what if k == 1 else what + 's'} converged"


def _finish(cls, d, k, func, vals_dtype=None):
    info, extract = int(d["info"]), int(d["extract"])
    if info == -9:
        raise ValueError(f"{func}(): the start vector OP v0 is zero (v0 lies in the null space of OP)")
    if info == -8 or extract != 0:
        from ..linalg import LinAlgError

        raise LinAlgError(f"{func}(): the small eigenvalue problem inside ARPACK failed (info {info}, {extract})")
    w, z = d["d"], d["z"]
    if vals_dtype is not None:  # a complex Hermitian eigsh: real values, ascending as the real path's
        w = _sd.real(w).copy()
        order = _sd.argsort(w, stable=True)
        w = w[order]
        if z is not None:
            z = z[:, order]
    got = w.shape[0]
    status = 0 if info == 0 else info
    return cls._make(eigenvalues=w, eigenvectors=z, converged=status == 0, status=status,
                     message=_message(status, got, k), n_iter=int(d["iter"]), n_converged=got,
                     n_op=int(d["nopx"]))  # fmt: skip


# ---------------------------------------------------------------- eigsh, eigs


def eigsh(A, /, *, k=6, M=None, sigma=None, which="LM", v0=None, ncv=None, max_iter=None, tol=0,
          return_eigenvectors=True, Minv=None, OPinv=None, mode="normal", rng=None):  # fmt: skip
    """k eigenvalues and eigenvectors of a real symmetric or complex Hermitian A (A x = w M x with
    M Hermitian positive (semi)definite) by ARPACK's implicitly restarted Lanczos (scipy's eigsh):
    which 'LM', 'SM', 'LA', 'SA', 'BE' (largest / smallest magnitude, largest / smallest
    algebraic, half from each end); sigma: shift-invert near sigma (mode 'normal', 'buckling' or
    'cayley'). The result unpacks as (w, v): w ascending, v's columns the eigenvectors (None when
    return_eigenvectors is False); converged, status and message tell whether the iteration
    ended (max_iter restarts, default 10 n). tol: the relative accuracy (0: machine epsilon)."""
    func = "eigsh"
    A, n = _square_a(A, func)
    k, ncv = _counts(k, ncv, n, func, 0, "sd.linalg.eigh")
    which = _choice(which, _SYM_WHICH, func, "which")
    mode = _choice(mode, _MODES, func, "mode")
    vectors = _flag(return_eigenvectors, func, "return_eigenvectors")
    max_iter = _count(max_iter, func, "max_iter", 10 * n)
    tol = _tol(tol, func, "tol")
    sigma = _sigma(sigma, func, real=True)
    M = _operand(M, func, "M", (n, n))
    Minv = _operand(Minv, func, "Minv", (n, n))
    OPinv = _operand(OPinv, func, "OPinv", (n, n))
    _check_inverses(func, sigma, M, Minv, OPinv)
    if sigma is None and mode != "normal":
        raise ValueError(f"{func}(): mode={mode!r} needs 'sigma'")
    dtypes = [A.dtype] + [o.dtype for o in (M, Minv, OPinv) if o is not None]
    if v0 is not None and isinstance(v0, _sd.ndarray):
        dtypes.append(v0.dtype)
    dtype = _dtype(*dtypes)
    if dtype == _sd.complex128:
        # a complex Hermitian problem runs on eigs's complex Arnoldi (scipy)
        if mode != "normal":
            raise ValueError(f"{func}(): mode={mode!r} cannot be used with a complex problem")
        if which == "BE":
            raise ValueError(f"{func}(): which='BE' cannot be used with a complex problem")
        which = {"LA": "LR", "SA": "SR"}.get(which, which)
        d = _arnoldi(func, A, n, k, ncv, which, tol, max_iter, vectors, sigma, M, Minv, OPinv, None, v0, rng, dtype)
        return _finish(EigshResult, d, k, func, vals_dtype=_sd.float64)
    v, draw = _start(v0, rng, n, dtype, func)
    if sigma is None:
        imode = 1 if M is None else 2
        inv = None
        if M is not None:
            inv = _native(Minv, dtype) if Minv is not None else _python(_inverse_of(M, func, tol), dtype, func)
        Mn = _native(M, dtype)
    else:
        imode = {"normal": 3, "buckling": 4, "cayley": 5}[mode]
        if OPinv is not None:
            inv = _native(OPinv, dtype)
        else:
            inv = _python(_op_inverse(A, M, sigma, dtype, func, tol), dtype, func)
        Mn = None if imode == 4 else _native(M, dtype)
    d = _sd_core_sparse()._eigsh(A.native(dtype), Mn, inv, v, draw, k=k, ncv=ncv, which=which,
                                 tol=tol if tol > 0 else _EPS, max_iter=max_iter, mode=imode,
                                 sigma=0.0 if sigma is None else sigma, vectors=vectors)  # fmt: skip
    return _finish(EigshResult, d, k, func)


def eigs(A, /, *, k=6, M=None, sigma=None, which="LM", v0=None, ncv=None, max_iter=None, tol=0,
         return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None, rng=None):  # fmt: skip
    """k eigenvalues and eigenvectors of a square A (A x = w M x with M Hermitian positive
    (semi)definite) by ARPACK's implicitly restarted Arnoldi (scipy's eigs): which 'LM', 'SM',
    'LR', 'SR', 'LI', 'SI' (magnitude, real part, imaginary part); sigma: shift-invert near sigma
    (OPpart 'r' or 'i' for a real A and a complex sigma). The result unpacks as (w, v), w
    complex128 in ARPACK's order, v's columns the eigenvectors (None when return_eigenvectors is
    False); converged, status and message tell whether the iteration ended (max_iter restarts,
    default 10 n). tol: the relative accuracy (0: machine epsilon)."""
    func = "eigs"
    A, n = _square_a(A, func)
    k, ncv = _counts(k, ncv, n, func, 1, "sd.linalg.eig")
    which = _choice(which, _NONSYM_WHICH, func, "which")
    vectors = _flag(return_eigenvectors, func, "return_eigenvectors")
    max_iter = _count(max_iter, func, "max_iter", 10 * n)
    tol = _tol(tol, func, "tol")
    sigma = _sigma(sigma, func, real=False)
    M = _operand(M, func, "M", (n, n))
    Minv = _operand(Minv, func, "Minv", (n, n))
    OPinv = _operand(OPinv, func, "OPinv", (n, n))
    _check_inverses(func, sigma, M, Minv, OPinv)
    if OPpart is not None:
        OPpart = _choice(OPpart, ("r", "i"), func, "OPpart")
        if sigma is None:
            raise ValueError(f"{func}(): 'OPpart' needs 'sigma'")
    dtypes = [A.dtype] + [o.dtype for o in (M, Minv, OPinv) if o is not None]
    if v0 is not None and isinstance(v0, _sd.ndarray):
        dtypes.append(v0.dtype)
    dtype = _dtype(*dtypes)
    if OPpart is not None and dtype == _sd.complex128:
        raise ValueError(f"{func}(): 'OPpart' is for a real problem with a complex 'sigma'")
    if OPpart == "i" and sigma.imag == 0:
        raise ValueError(f"{func}(): OPpart='i' needs a complex 'sigma'")
    d = _arnoldi(func, A, n, k, ncv, which, tol, max_iter, vectors, sigma, M, Minv, OPinv, OPpart, v0, rng, dtype)
    return _finish(EigsResult, d, k, func)


def _check_inverses(func, sigma, M, Minv, OPinv):
    if sigma is None:
        if OPinv is not None:
            raise ValueError(f"{func}(): 'OPinv' needs 'sigma'")
        if Minv is not None and M is None:
            raise ValueError(f"{func}(): 'Minv' needs 'M'")
    elif Minv is not None:
        raise ValueError(f"{func}(): 'Minv' cannot be given with 'sigma' (give 'OPinv')")


def _inverse_of(M, func, tol):
    if M.matrix is not None:
        return _Inverse(func, matrix=M.matrix)
    return _Inverse(func, op=M.op, tol=tol)


def _arnoldi(func, A, n, k, ncv, which, tol, max_iter, vectors, sigma, M, Minv, OPinv, OPpart, v0, rng, dtype):
    v, draw = _start(v0, rng, n, dtype, func)
    sr = si = 0.0
    if sigma is None:
        imode = 1 if M is None else 2
        inv = None
        if M is not None:
            inv = _native(Minv, dtype) if Minv is not None else _python(_inverse_of(M, func, tol), dtype, func)
    else:
        sigma = complex(sigma)
        sr, si = sigma.real, sigma.imag
        imode = 4 if OPpart == "i" else 3
        if OPinv is not None:
            inv = _native(OPinv, dtype)
            if dtype != _sd.complex128 and si != 0:  # the user's inverse is complex: take its part
                f = OPinv.op.matvec if OPinv.op is not None else (lambda x, m=OPinv.matrix: m @ x)
                inv = _part_of(f, "imag" if imode == 4 else "real", dtype, func)
        else:
            sdtype = _sd.complex128 if (si != 0 or dtype == _sd.complex128) else dtype
            part = None
            if dtype != _sd.complex128 and si != 0:
                part = "imag" if imode == 4 else "real"
            shift = sigma if sdtype == _sd.complex128 else sr
            inv = _python(_op_inverse(A, M, shift, sdtype, func, tol, part), dtype, func)
    # eigs with a real A and a complex sigma needs the vectors for its Rayleigh quotients
    rvec = vectors or (dtype != _sd.complex128 and sigma is not None and si != 0)
    d = _sd_core_sparse()._eigs(A.native(dtype), _native(M, dtype), inv, v, draw, k=k, ncv=ncv, which=which,
                                tol=tol if tol > 0 else _EPS, max_iter=max_iter, mode=imode, sigma_real=sr,
                                sigma_imag=si, vectors=rvec)  # fmt: skip
    if rvec and not vectors:
        d["z"] = None
    return d


def _part_of(f, part, dtype, func):
    def call(x):
        y = f(x)
        return _sd.imag(y) if part == "imag" else _sd.real(y)

    return _python(call, dtype, func)


def _sd_core_sparse():
    from ._common import _core

    return _core


# ---------------------------------------------------------------- svds


def svds(A, /, *, k=6, ncv=None, tol=0, which="LM", v0=None, max_iter=None, return_singular_vectors=True, rng=None):
    """k singular values and vectors of A (m x n) by Golub-Kahan-Lanczos bidiagonalization with
    thick restarts: which 'LM' (largest) or 'SM' (smallest). The result unpacks as (U, S, Vh): S
    ascending (scipy's order), U (m, k) and Vh (k, n); return_singular_vectors True, False, 'u'
    or 'vh' says which of U and Vh are computed (None otherwise). A triplet converges when
    ||A^H u - s v|| <= tol max(s, eps^(2/3) s_max) (tol 0: machine epsilon); max_iter restarts
    (default 10 min(m, n)). v0 has min(m, n) entries."""
    func = "svds"
    A = _Operand(A, func, "A")
    m, n = A.shape
    if m == 0 or n == 0:
        raise ValueError(f"{func}(): 'A' must not be empty, got shape {A.shape}")
    mn = min(m, n)
    k = _int(k, func, "k")
    if not (1 <= k < mn):
        raise ValueError(f"{func}(): 'k' must satisfy 1 <= k < min(A.shape) = {mn}, got {k}")
    if ncv is None:
        ncv = min(max(2 * k + 1, 20), mn)
    else:
        ncv = _int(ncv, func, "ncv")
        if not (k < ncv <= mn):
            raise ValueError(f"{func}(): 'ncv' must satisfy k < ncv <= min(A.shape) = {mn}, got {ncv}")
    tol = _tol(tol, func, "tol")
    which = _choice(which, _SVD_WHICH, func, "which")
    max_iter = _count(max_iter, func, "max_iter", 10 * mn)
    rsv = return_singular_vectors
    if not (rsv is True or rsv is False or rsv in ("u", "vh")):
        if isinstance(rsv, str):
            raise ValueError(f"{func}(): 'return_singular_vectors' must be True, False, 'u' or 'vh', not {rsv!r}")
        raise TypeError(f"{func}(): 'return_singular_vectors' must be True, False, 'u' or 'vh', not {C.type_name(rsv)}")
    dtype = _dtype(A.dtype, *((v0.dtype,) if isinstance(v0, _sd.ndarray) else ()))
    transpose = m < n  # bidiagonalize X = A^H (n x m) so X has at least as many rows as columns
    v, draw = _start(v0, rng, mn, dtype, func)
    plain, adjoint = A.native(dtype), A.native(dtype, adjoint=True)
    Xn, XHn = (adjoint, plain) if transpose else (plain, adjoint)
    rows = max(m, n)
    d = _sd_core_sparse()._svds(Xn, XHn, v, draw, rows=rows, k=k, ncv=ncv, which=which, tol=tol if tol > 0 else _EPS,
                                max_iter=max_iter, vectors=rsv is not False)  # fmt: skip
    s = d["s"]
    U = Vh = None
    if rsv is not False:
        uu, vv = d["u"], d["v"]  # the left (rows) and right (min dim) vectors of X, as columns
        if transpose:  # X = A^H: A's left vectors are X's right ones
            uu, vv = vv, uu
        if rsv in (True, "u"):
            U = uu
        if rsv in (True, "vh"):
            Vh = _sd.conj(vv.mT).copy() if dtype == _sd.complex128 else vv.mT.copy()
    status = int(d["status"])
    got = s.shape[0]
    return SvdsResult._make(U=U, S=s, Vh=Vh, converged=status == 0, status=status,
                            message=_message(status, got, k, "singular value"), n_iter=int(d["iter"]),
                            n_converged=got, n_op=int(d["nmv"]))  # fmt: skip


__all__ = ["eigsh", "eigs", "svds"]
