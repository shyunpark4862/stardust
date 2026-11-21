"""The matrix functions and norms of sd.sparse (docs/api/sparse.md §8, docs/design/sparse.md §3.9):
expm, expm_multiply, onenormest, norm.

They are written in Python on this library's kernels (SpMV and SpMM, SpGEMM, the Accelerate LU of
§3.6 with its refinement and singularity check, the reductions of §2.6); every one of those is
deterministic, so a result is the same bit for bit for any thread count (and, for expm, for any
input format: the matrix becomes CSR first).

  - onenormest: Higham and Tisseur's block 1-norm estimator (2000, Algorithm 2.4), scipy 1.18.1's
    steps. Its random +-1 columns come from sd.random.default_rng(0) on every call (decision SP98:
    scipy's are unseeded), and h is ordered by value, ties by index (scipy's argsort is not stable).
    Its result is an OnenormestResult (decision SP102; scipy returns the bare estimate).
  - expm: Al-Mohy and Higham's scaling and squaring (2009, Algorithm 5.1), the choices of the dense
    sd.linalg.expm (docs/design/linalg.md §11): exact norms with bounds first, ||A^8|| and
    ||A^10|| estimated (here by onenormest) from n = 100 on, ell() exact, I + 2 (V - U)^-1 U, the
    diagonal and triangular shortcuts and fix-ups, the 2^100 guard. The products are SpGEMM, the
    solve is the sparse LU (spsolve's), the result in A's format (decision SP99).
  - expm_multiply: Al-Mohy and Higham's truncated Taylor series (2011), scipy 1.18.1's steps and
    tolerance 2^-53. A LinearOperator without traceA gets scipy's trace estimate on fixed draws.
  - norm: scipy's meaning with sd.linalg.matrix_norm's / vector_norm's ord rules and messages
    (SP14); a matrix's 'nuc' and -2 are not accepted (decision SP101).
"""

import cmath
import math
import numbers
import operator

from .. import _core as _sd
from . import _common as C
from ._base import sparray
from ._iterative import _Operand
from ._result import OnenormestResult


def _default_rng(seed):
    from ..random import default_rng

    return default_rng(seed)


_INF = math.inf

# ---------------------------------------------------------------- arguments


def _int_arg(v, func, arg, minimum):
    if not C.is_int(v):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {C.type_name(v)}")
    v = operator.index(v)
    if v < minimum:
        raise ValueError(f"{func}(): '{arg}' must be >= {minimum}")
    if v >= 2**63:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _flag(v, func, arg):
    if v is True or v is False:
        return v
    raise TypeError(f"{func}(): '{arg}' must be True or False, not {C.type_name(v)}")


def _square(shape, func, what="A"):
    if shape[0] != shape[1]:
        raise ValueError(f"{func}(): '{what}' must be square, got {shape[0]}x{shape[1]}")
    return shape[0]


def _real_dtype(dtype):
    return _sd.complex128 if dtype == _sd.complex128 else _sd.float64


def _cast(y, dtype, func):
    """An operator's product as the system's dtype (a real result widens; a complex one in a real
    system is a TypeError, as the iterative solvers')."""
    if y.dtype != dtype:
        if y.dtype == _sd.complex128:
            raise TypeError(
                f"{func}(): the operator returned complex128 values for a float64 system; give it dtype=complex128"
            )
        y = y.astype(dtype)
    return y


def _compressed(A):
    """A sparse matrix for repeated products: CSR, or a CSC as it is on one thread. Every format's
    product has the same bits (docs/design/sparse.md §2.6); a CSC's is a scatter on one thread,
    about as fast there as CSR's, so the conversion (dozens of SpMVs' time at nnz 10^6) pays only
    when the pool can split CSR's rows."""
    if A.format == "csr" or (A.format == "csc" and _sd.get_num_threads() == 1):
        return A
    return A.tocsr()


class _Map:
    """X -> A X and X -> A^H X for a dense X of shape (n,) or (n, k), in the system's dtype: a sparse
    matrix by its CSR (SpMV, SpMM), a dense one by matmul, another LinearOperator by its own
    products. A^H is formed on its first use."""

    __slots__ = ("fwd", "_A", "_op", "_AH", "_dtype", "_func")

    def __init__(self, operand, dtype):
        A, self._op, self._AH, self._dtype, self._func = operand.matrix, operand.op, None, dtype, operand.func
        if A is None:
            op, func = self._op, operand.func
            self.fwd = lambda X: _cast(op @ X, dtype, func)
        else:
            if isinstance(A, sparray):
                A = _compressed(A)
            if A.dtype != dtype:
                A = A.astype(dtype)
            self.fwd = lambda X: A @ X
        self._A = A

    def adj(self, X):
        if self._A is None:
            if self._AH is None:
                if not self._op._has_adjoint():
                    raise ValueError(f"{self._func}(): the operator was made without 'rmatvec'")
                self._AH = self._op.H
            return _cast(self._AH @ X, self._dtype, self._func)
        if self._AH is None:
            A, cplx = self._A, self._dtype == _sd.complex128
            if isinstance(A, sparray):
                self._AH = _compressed(A.T.conj() if cplx else A.T)
            else:
                self._AH = (_sd.conj(A.mT) if cplx else A.mT).copy()
        return self._AH @ X


# ---------------------------------------------------------------- onenormest


def _parallel(v, w):
    """Whether two sign columns are parallel: v_i w_i = 1 everywhere (the same +-1 column when real;
    scipy's dot(v, w) == n)."""
    return bool(_sd.all(v * w == 1))


def _needs_resampling(i, X, Y=None):
    v = X[:, i]
    if any(_parallel(v, X[:, j]) for j in range(i)):
        return True
    return Y is not None and any(_parallel(v, Y[:, j]) for j in range(Y.shape[1]))


def _top(h, r):
    """The indices of the r largest entries of h, largest first, ties by index (sd.argsort's stable
    descending order; for few, r passes of argmax, the same indices)."""
    if r > 16:
        return [int(i) for i in _sd.argsort(h, descending=True)[:r].tolist()]
    h = h.copy()
    out = []
    for _ in range(r):
        j = int(_sd.argmax(h))
        out.append(j)
        h[j] = -_INF
    return out


def _unit(n, i):
    v = _sd.zeros(n)
    v[i] = 1.0
    return v


def _estimate(fwd, adj, n, t, max_iter, dtype):
    """Higham and Tisseur's Algorithm 2.4 (scipy's _onenormest_core) for 1 <= t < n: (est, v, w,
    iterations, products), v the unit vector of the column that gave w = A v; the products count
    the blocks A X and A^H S (scipy's nmults)."""
    g = _default_rng(0)

    def resample(X, i):
        X[:, i] = (g.integers(0, 2, shape=n) * 2 - 1).astype(_sd.float64)

    X = _sd.ones((n, t))
    for i in range(1, t):
        resample(X, i)
    for i in range(t):
        while _needs_resampling(i, X):
            resample(X, i)
    X = X / float(n)
    hist, seen_hist = [], set()
    est_old = 0.0
    S = _sd.zeros((n, t), dtype=dtype)
    k = 1
    ind = None
    ind_best = None
    w = None
    n_op = 0
    while True:
        Y = fwd(X)
        n_op += 1
        mags = _sd.sum(_sd.abs(Y), axis=0)
        best_j = int(_sd.argmax(mags))
        est = float(mags[best_j])
        if est > est_old or k == 2:
            if k >= 2:
                ind_best = ind[best_j]
            w = Y[:, best_j].copy()
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        S_old = S
        if k > max_iter:
            break
        S = _sd.where(Y == 0, 1, Y / _sd.abs(Y)).astype(dtype)  # sign(Y), sign(0) = 1
        if all(any(_parallel(S[:, i], S_old[:, j]) for j in range(t)) for i in range(t)):
            break
        if t > 1:
            for i in range(t):
                while _needs_resampling(i, S, S_old):
                    resample(S, i)
        Z = adj(S)
        n_op += 1
        h = _sd.max(_sd.abs(Z), axis=1)
        if k >= 2 and float(_sd.max(h)) == float(h[ind_best]):
            break
        ind = _top(h, min(t + len(hist), n))
        if t > 1:
            if all(i in seen_hist for i in ind[:t]):
                break
            ind = [i for i in ind if i not in seen_hist] + [i for i in ind if i in seen_hist]
        X = _sd.zeros((n, t))
        for j in range(t):
            X[ind[j], j] = 1.0
        for i in ind[:t]:
            if i not in seen_hist:
                hist.append(i)
                seen_hist.add(i)
        k += 1
    return est, _unit(n, ind_best), w, k, n_op


def _exact(E, n):
    """(est, v, w, 0, 1) of an explicit n x n matrix (the product A I): its largest column sum of
    magnitudes."""
    sums = _sd.sum(_sd.abs(E), axis=0)
    j = int(_sd.argmax(sums))
    return float(sums[j]), _unit(n, j), E[:, j].copy(), 0, 1


def _norm1_of(fwd, adj, n, dtype, t=2, max_iter=5):
    """onenormest of the operator (fwd, adj), (est, v, w, iterations, products): exact (an explicit
    matrix) when t >= n."""
    if n == 0:
        return 0.0, _sd.zeros(0), _sd.zeros(0, dtype=dtype), 0, 0
    if t >= n:
        return _exact(fwd(_sd.eye(n, dtype=dtype)), n)
    return _estimate(fwd, adj, n, t, max_iter, dtype)


def onenormest(A, /, *, t=2, max_iter=5, compute_v=False, compute_w=False):
    """A lower bound of ||A||_1 (Higham and Tisseur's block estimator, scipy's): A a square sparse
    array, 2-D sd.ndarray or LinearOperator; t columns at a time (t >= n: the exact norm), at most
    max_iter iterations. The random +-1 columns are fixed (sd.random.default_rng(0)), so the same
    call gives the same estimate. A read-only OnenormestResult that unpacks into (est, v, w), v the
    unit vector of the column found and w = A v (each None unless compute_v / compute_w), with the
    counts n_iter and n_op."""
    func = "onenormest"
    t = _int_arg(t, func, "t", 1)
    max_iter = _int_arg(max_iter, func, "max_iter", 2)
    compute_v = _flag(compute_v, func, "compute_v")
    compute_w = _flag(compute_w, func, "compute_w")
    op = _Operand(A, func, "A")
    n = _square(op.shape, func)
    m = _Map(op, op.dtype)
    est, v, w, n_iter, n_op = _norm1_of(m.fwd, m.adj, n, op.dtype, t, max_iter)
    return OnenormestResult._make(est=est, v=v if compute_v else None, w=w if compute_w else None, n_iter=n_iter,
                                  n_op=n_op)  # fmt: skip


def _product_norm1(mats, dtype):
    """onenormest (t = 2, five iterations) of the product of the sparse CSR matrices `mats`."""
    adjs = [(M.T.conj() if dtype == _sd.complex128 else M.T).tocsr() for M in mats]

    def fwd(X):
        for M in reversed(mats):
            X = M @ X
        return X

    def adj(X):
        for M in adjs:
            X = M @ X
        return X

    return _norm1_of(fwd, adj, mats[0].shape[0], dtype)[0]


# ---------------------------------------------------------------- expm

# From this n on, ||A^8|| and ||A^10|| are estimated, not formed: sd.linalg.expm's threshold
# (kExpmEstimateMin, docs/design/linalg.md §11.2).
_ESTIMATE_MIN = 100

# theta_m: the largest eta for which r_m's backward error is below 2^-53 (Al-Mohy & Higham 2009,
# Table 3.1; theta_13 = 4.25 as scipy and sd.linalg.expm).
_THETA3 = 1.495585217958292e-2
_THETA5 = 2.539398330063230e-1
_THETA7 = 9.504178996162932e-1
_THETA9 = 2.097847961257068e0
_THETA13 = 4.25
# The Padé coefficients b_0 .. b_m.
_B = {
    3: (120.0, 60.0, 12.0, 1.0),
    5: (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0),
    7: (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0),
    9: (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0,
        1.0),
    13: (64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0, 129060195264000.0,
         10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0),
}  # fmt: skip


def _log2_err_coeff(m):
    """log2 |c_(2m+1)| = log2 (m!)^2 / ((2m)! (2m+1)!), the leading coefficient of r_m's backward
    error series (sd.linalg.expm's sum, term by term)."""
    v = 0.0
    for k in range(2, m + 1):
        v += 2 * math.log2(k)
    for k in range(2, 2 * m + 1):
        v -= math.log2(k)
    for k in range(2, 2 * m + 2):
        v -= math.log2(k)
    return v


def _abs(M):
    """|M| as float64, M a CSR or CSC: its pattern's long-line flag found on M first, so that |M|
    (which shares it) does not search again on every call (_compressed._long_across)."""
    M._long_across()
    a = abs(M)
    return a if a.dtype == _sd.float64 else a.astype(_sd.float64)


def _norm1(M):
    """||M||_1 exactly: the largest column sum of magnitudes (§2.6's sums)."""
    if M.shape[1] == 0:
        return 0.0
    return float(_sd.max(_abs(M).sum(axis=0)))


def _scale2(M, k):
    """M * 2^k, exactly (up to underflow): one product while 2^k is a normal double."""
    while k < -1022:
        M = M * math.ldexp(1.0, -1022)
        k += 1022
    return M * math.ldexp(1.0, k)


class _AbsPowers:
    """log2 || |A|^p ||_1 for p = 1, 2, ... in turn: e^T |A|^p from B = 2^-e |A| (||B||_1 <= 1, so no
    power overflows), log2 max(e^T B^p) + p e."""

    def __init__(self, A, e):
        a = abs(A)
        if e >= -1022:
            a = a * math.ldexp(1.0, -e)
        else:  # 2^-e itself would overflow: two exact products, every result <= 1
            a = (a * math.ldexp(1.0, 1000)) * math.ldexp(1.0, -e - 1000)
        self.bt = a.T.tocsr()
        self.e = e
        self.r = _sd.ones(A.shape[0])
        self.p = 0

    def log2norm(self, p):
        while self.p < p:
            self.r = self.bt @ self.r
            self.p += 1
        top = float(_sd.max(self.r))
        return (math.log2(top) if top > 0 else -_INF) + self.p * self.e


def expm(A, /):
    """exp(A) of the square sparse A: Al-Mohy and Higham's scaling and squaring (scipy's algorithm,
    with sd.linalg.expm's choices), as a sparse array of A's format, its exact zeros dropped. Usually
    nearly dense."""
    func = "expm"
    from ._construct import diags_array, eye_array, tril, triu
    from ._direct import _Factor, _matrix, _sparse_solve, is_sptriangular

    A = _matrix(A, func)
    fmt, n = A.format, A.shape[0]
    dtype = _real_dtype(A.dtype)
    X0 = A.tocsr()
    X0 = X0.astype(dtype) if X0.dtype != dtype else X0.copy()
    if X0.nnz and not bool(_sd.all(_sd.isfinite(X0.data))):
        raise ValueError(f"{func}(): 'A' contains NaN or infinity")
    X0.eliminate_zeros()
    if n == 0:
        return X0.asformat(fmt)
    lower, upper = is_sptriangular(X0)
    if lower and upper:  # diagonal (1 x 1 too): exp of the diagonal, exactly
        d = X0.diagonal()
        out = diags_array(C._core._expm_fix(d, d[1:], 0)[0], format="csr")  # libm exp, as sd.linalg.expm
        out.eliminate_zeros()
        return out.asformat(fmt)
    shape = "upper" if upper else ("lower" if lower else "full")
    if shape != "full":  # for the fix-ups: A's diagonal and first off-diagonal
        lam = X0.diagonal()
        tt = X0.diagonal(k=1 if shape == "upper" else -1)

    # ---- guard: ||A|| <= 2^100, so that A^2 .. A^10 cannot overflow
    a = X0
    nA = _norm1(a)
    s0 = 0
    if math.isinf(nA):  # finite entries whose column sum overflows: 2^k >= 4n makes it finite
        k = 2
        while (1 << (k - 2)) < n:
            k += 1
        s0 = k
        a = _scale2(a, -k)
        nA = _norm1(a)
    e = math.frexp(nA)[1]
    if e > 100:
        s0 += e - 100
        a = _scale2(a, -(e - 100))
        nA = _norm1(a)
        e = math.frexp(nA)[1]
    absp = _AbsPowers(a, e)
    log2nA = math.log2(nA)

    def ell(m, shift):
        la = _log2_err_coeff(m) + absp.log2norm(2 * m + 1) - log2nA - 2.0 * m * shift
        v = math.ceil((la + 53.0) / (2.0 * m)) if la > -_INF else 0
        return v if v > 0 else 0

    # ---- degree and scaling (sd.linalg.expm's tests: bounds from known norms first, a power
    # formed only when a test needs its exact norm; ||A^8|| and ||A^10|| estimated from n = 100)
    estimate = n >= _ESTIMATE_MIN
    P = {2: a @ a}
    nrm = {2: _norm1(P[2])}

    def need(k):
        if k not in P:
            P[k] = P[4] @ P[2] if k == 6 else P[k // 2] @ P[k // 2]
            if k not in nrm:
                nrm[k] = _norm1(P[k])

    def need6():
        need(4)
        need(6)

    def norm8():
        need(4)
        if 8 not in nrm:
            if estimate:
                nrm[8] = _product_norm1([P[4], P[4]], dtype)
            else:
                need(8)

    def root(v, k):
        return v ** (1.0 / k)

    def d4():
        return root(nrm[4], 4) if 4 in nrm else root(nrm[2], 2)

    def d6():
        if 6 in nrm:
            return root(nrm[6], 6)
        return root(nrm[4] * nrm[2], 6) if 4 in nrm else root(nrm[2], 2)

    def d8():  # ||A^8|| <= ||A^4||^2: d8 <= ||A^4||^(1/4)
        if 8 in nrm:
            return root(nrm[8], 8)
        return root(nrm[4], 4) if 4 in nrm else root(nrm[2], 2)

    def passes(theta, dx, dy, form):
        if max(dx(), dy()) <= theta:
            return True
        form()
        return max(dx(), dy()) <= theta

    s1 = 0
    if passes(_THETA3, d4, d6, need6) and ell(3, 0) == 0:
        m = 3
    elif (need(4), passes(_THETA5, d4, d6, need6))[1] and ell(5, 0) == 0:
        m = 5
    elif (need6(), passes(_THETA7, d6, d8, norm8))[1] and ell(7, 0) == 0:
        m = 7
    elif (norm8(), max(d6(), d8()) <= _THETA9)[1] and ell(9, 0) == 0:
        m = 9
        need(8)
    else:
        m = 13
        n10 = _product_norm1([P[4], P[6]], dtype) if estimate else _norm1(P[4] @ P[6])
        d10 = root(n10, 10)
        eta3 = max(d6(), d8())
        eta5 = min(eta3, max(d8(), d10))
        s1 = 0 if eta5 <= _THETA13 else math.ceil(math.log2(eta5 / _THETA13))
        s1 += ell(13, s1)
        if s1 > 0:
            a = _scale2(a, -s1)
            for k in (2, 4, 6):
                P[k] = _scale2(P[k], -k * s1)
    sq = s0 + s1

    # ---- U and V (Higham 2005's evaluation schemes, sd.linalg.expm's order of terms)
    b = _B[m]
    eye = eye_array(n, format="csr", dtype=dtype)
    if m == 13:
        A2, A4, A6 = P[2], P[4], P[6]
        W = A6 * b[13] + A4 * b[11] + A2 * b[9]
        U = a @ (A6 @ W + A6 * b[7] + A4 * b[5] + A2 * b[3] + eye * b[1])
        W = A6 * b[12] + A4 * b[10] + A2 * b[8]
        V = A6 @ W + A6 * b[6] + A4 * b[4] + A2 * b[2] + eye * b[0]
    else:
        top = (m - 1) // 2
        odd = P[2 * top] * b[2 * top + 1]
        even = P[2 * top] * b[2 * top]
        for k in range(top - 1, 0, -1):
            odd = odd + P[2 * k] * b[2 * k + 1]
            even = even + P[2 * k] * b[2 * k]
        U = a @ (odd + eye * b[1])
        V = even + eye * b[0]
    # r_m = (V - U)^-1 (V + U) = I + 2 (V - U)^-1 U: I added exactly to a small correction
    factor = _Factor(V - U, func)
    X = _sparse_solve(factor, U, "csr", func) * 2.0 + eye

    # ---- squaring, with the triangular fix-ups (Code Fragment 2.1)
    def fix(X, k):  # X ~ exp(2^-k A): its exact diagonal and first off-diagonal, sd.linalg.expm's bits
        diag, off = C._core._expm_fix(lam, tt, k)  # libm exp and sinh (Al-Mohy & Higham 2009, (2.2))
        X.setdiag(diag)
        X.setdiag(off, k=1 if shape == "upper" else -1)

    if shape != "full":
        fix(X, sq)
    for k in range(1, sq + 1):
        X = X @ X
        if shape != "full":
            fix(X, sq - k)
    if shape == "upper":  # the other triangle exactly 0
        X = triu(X, format="csr")
    elif shape == "lower":
        X = tril(X, format="csr")
    X.eliminate_zeros()
    return X.asformat(fmt)


# ---------------------------------------------------------------- expm_multiply

_TOL = 2.0**-53  # the unit roundoff: scipy's tolerance
_M_MAX = 55
_ELL = 2
# theta_m of Al-Mohy & Higham 2011, Table 3.1 (the double-precision column), scipy's table
_THETA = {
    1: 2.29e-16, 2: 2.58e-8, 3: 1.39e-5, 4: 3.40e-4, 5: 2.40e-3, 6: 9.07e-3, 7: 2.38e-2, 8: 5.00e-2,
    9: 8.96e-2, 10: 1.44e-1, 11: 2.14e-1, 12: 3.00e-1, 13: 4.00e-1, 14: 5.14e-1, 15: 6.41e-1, 16: 7.81e-1,
    17: 9.31e-1, 18: 1.09, 19: 1.26, 20: 1.44, 21: 1.62, 22: 1.82, 23: 2.01, 24: 2.22, 25: 2.43, 26: 2.64,
    27: 2.86, 28: 3.08, 29: 3.31, 30: 3.54, 35: 4.7, 40: 6.0, 45: 7.2, 50: 8.5, 55: 9.9,
}  # fmt: skip


def _inf_norm(B):
    """numpy's norm(B, inf): max |b_i| of a vector, the largest row sum of magnitudes of a matrix."""
    if B.ndim == 1:
        return _sd.linalg.vector_norm(B, ord=_INF)
    return _sd.linalg.matrix_norm(B, ord=_INF)


class _NormInfo:
    """scipy's LazyOperatorNormInfo of t A: ||t A||_1 and d_p = onenormest((t A)^p)^(1/p), cached,
    times a scale."""

    def __init__(self, fwd, adj, n, dtype, t, norm1):
        self.fwd, self.adj, self.n, self.dtype, self.t = fwd, adj, n, dtype, t
        self.norm1 = norm1
        self.d_ = {}
        self.scale = 1.0

    def onenorm(self):
        return self.scale * self.norm1

    def d(self, p):
        if p not in self.d_:
            t = self.t

            def fwd(X):
                for _ in range(p):
                    X = self.fwd(X) * t
                return X

            def adj(X):
                for _ in range(p):
                    X = self.adj(X) * t
                return X

            est = _norm1_of(fwd, adj, self.n, self.dtype)[0]
            self.d_[p] = est ** (1.0 / p)
        return self.scale * self.d_[p]

    def alpha(self, p):
        return max(self.d(p), self.d(p + 1))


def _p_max(m_max):
    sqrt_m = math.sqrt(m_max)
    lo, hi = int(math.floor(sqrt_m)), int(math.ceil(sqrt_m + 1))
    return max(p for p in range(lo, hi + 1) if p * (p - 1) <= m_max + 1)


def _fragment_3_1(info, n0):
    """(m_star, s): the cheapest Taylor degree and number of steps (Al-Mohy & Higham 2011, Code
    Fragment 3.1, scipy's)."""
    best_m = best_s = None
    p_max = _p_max(_M_MAX)
    if info.onenorm() <= 2 * _ELL * p_max * (p_max + 3) * (_THETA[_M_MAX] / float(n0 * _M_MAX)):
        for m, theta in _THETA.items():
            s = int(math.ceil(info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m, best_s = m, s
    else:
        for p in range(2, p_max + 1):
            for m in range(p * (p - 1) - 1, _M_MAX + 1):
                if m in _THETA:
                    s = int(math.ceil(info.alpha(p) / _THETA[m]))
                    if best_m is None or m * s < best_m * best_s:
                        best_m, best_s = m, s
        best_s = max(best_s, 1)
    return best_m, best_s


def _exp(z):
    """exp(z), inf (each part ±inf) when it overflows, as expm and scipy (math.exp raises)."""
    if not isinstance(z, complex):
        return _exp_real(z)
    try:
        return cmath.exp(z)
    except OverflowError:  # |exp(z)| past the largest double: magnitude and phase apart
        h = _exp_real(z.real / 2)  # exp(re) = h * h, so a part with a tiny cos or sin may stay finite
        c, s = math.cos(z.imag), math.sin(z.imag)
        return complex(h * c * h, h * s * h if s != 0 else z.imag)


def _exp_real(x):
    try:
        return math.exp(x)
    except OverflowError:
        return _INF


def _simple_core(prod, B, t, mu, m_star, s):
    """exp(t (A + mu I)) B by s steps of the degree-m_star Taylor series of t A / s, each stopped
    early once two terms are below the tolerance (scipy's _expm_multiply_simple_core)."""
    F = B
    eta = _exp(t * mu / float(s))
    for _ in range(s):
        c1 = _inf_norm(B)
        for j in range(m_star):
            coeff = t / float(s * (j + 1))
            B = prod(B) * coeff
            c2 = _inf_norm(B)
            F = F + B
            if c1 + c2 <= _TOL * _inf_norm(F):
                break
            c1 = c2
        F = F * eta
        B = F
    return F


def _interval_blocks(prod, X, h, mu, m_star, s, q):
    """X[k] = exp(k h (A + mu I)) X[0] for k = 1 .. q when q > s: blocks of d = q // s points from
    the block's first, the terms K[p] = (h A)^p Z / p! kept (scipy's _expm_multiply_interval_core_1
    and _2, which compute the same values)."""
    d = q // s
    j = q // d
    r = q - d * j
    for i in range(j + 1):
        K = [X[i * d]]
        for k in range(1, (d if i < j else r) + 1):
            F = K[0]
            c1 = _inf_norm(F)
            for p in range(1, m_star + 1):
                if p == len(K):
                    K.append((prod(K[p - 1]) * h) / float(p))
                coeff = float(pow(k, p))
                F = F + K[p] * coeff
                c2 = coeff * _inf_norm(K[p])
                if c1 + c2 <= _TOL * _inf_norm(F):
                    break
                c1 = c2
            X[k + i * d] = F * _exp(k * h * mu)


def _trace_estimate(fwd, n, m3, dtype, func):
    """scipy's traceest (Hutch++, Meyer et al. 2021) on fixed draws (sd.random.default_rng(0))."""
    g = _default_rng(0)
    signs = _sd.asarray([-1.0, 1.0])
    S = g.choice(signs, shape=(n, m3))
    Y = fwd(S.astype(dtype))
    if not bool(_sd.all(_sd.isfinite(Y))):  # the QR below would name itself
        raise ValueError(f"{func}(): the operator returned NaN or infinity")
    Q = _sd.linalg.qr(Y)[0]
    QH = _sd.conj(Q.mT) if dtype == _sd.complex128 else Q.mT
    tr_q = _sd.linalg.trace(QH @ fwd(Q))
    G = g.choice(signs, shape=(n, m3)).astype(dtype)
    right = G - Q @ (QH @ G)
    RH = _sd.conj(right.mT) if dtype == _sd.complex128 else right.mT
    tr_g = _sd.linalg.trace(RH @ fwd(right))
    return tr_q + tr_g / m3


def _time(v, func, arg):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {C.type_name(v)}")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{func}(): '{arg}' must be finite")
    return x


def _trace_arg(v, func):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, numbers.Number) or isinstance(v, _sd.ndarray):
        raise TypeError(f"{func}(): 'traceA' must be a number or None, not {C.type_name(v)}")
    if isinstance(v, numbers.Real):
        x = float(v)
        if not math.isfinite(x):
            raise ValueError(f"{func}(): 'traceA' must be finite")
        return x
    z = complex(v)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise ValueError(f"{func}(): 'traceA' must be finite")
    return z


def expm_multiply(A, B, /, *, start=None, stop=None, num=None, endpoint=None, traceA=None):
    """exp(A) B without forming exp(A) (Al-Mohy and Higham 2011, scipy's algorithm and tolerance
    2^-53): A a square sparse array, 2-D sd.ndarray or LinearOperator, B a dense sd.ndarray (n,) or
    (n, k). With start and stop: exp(t A) B at the num points t of linspace(start, stop, num,
    endpoint=endpoint) (num 50 and endpoint True by default), stacked: shape (num,) + B.shape.
    traceA (trace(A), exact for a matrix when None) only shifts A; a LinearOperator without it gets
    scipy's trace estimate on fixed draws."""
    func = "expm_multiply"
    op = _Operand(A, func, "A")
    n = _square(op.shape, func)
    if isinstance(B, sparray):
        raise TypeError(f"{func}(): 'B' must be an sd.ndarray, not {C.type_name(B)}; for a sparse B use B.toarray()")
    if not isinstance(B, _sd.ndarray):
        raise TypeError(f"{func}(): 'B' must be an sd.ndarray, not {C.type_name(B)}")
    if B.dtype == _sd.string:
        raise TypeError(f"{func}(): 'B' must hold numbers, not strings")
    if B.ndim not in (1, 2):
        raise ValueError(f"{func}(): 'B' must have 1 or 2 dimensions, got {B.ndim}")
    if B.shape[0] != n:
        raise ValueError(f"{func}(): 'A' is {n}x{n} but 'B' has {B.shape[0]} rows")
    if B.size and not bool(_sd.all(_sd.isfinite(B))):
        raise ValueError(f"{func}(): 'B' contains NaN or infinity")
    traceA = _trace_arg(traceA, func)
    interval = not all(v is None for v in (start, stop, num, endpoint))
    if interval:
        if start is None or stop is None:
            raise ValueError(f"{func}(): a time interval needs both 'start' and 'stop'")
        start, stop = _time(start, func, "start"), _time(stop, func, "stop")
        num = 50 if num is None else _int_arg(num, func, "num", 2)
        endpoint = True if endpoint is None else _flag(endpoint, func, "endpoint")
    complex_ = op.dtype == _sd.complex128 or B.dtype == _sd.complex128 or isinstance(traceA, complex)
    dtype = _sd.complex128 if complex_ else _sd.float64
    B = B.astype(dtype) if B.dtype != dtype else B.copy()
    if n == 0 or B.ndim == 2 and B.shape[1] == 0:  # nothing to compute: an empty result (B (n, 0) too)
        return _sd.zeros(((num,) if interval else ()) + tuple(B.shape), dtype=dtype)

    # A - mu I, mu = trace(A) / n
    M = op.matrix
    if traceA is None:
        if M is None:
            traceA = _trace_estimate(_Map(op, dtype).fwd, n, 5 if interval else 1, dtype, func)
        elif isinstance(M, sparray):
            traceA = M.trace()
        else:
            traceA = _sd.linalg.trace(M)
    mu = traceA / float(n)
    if isinstance(mu, complex) and dtype != _sd.complex128:
        mu = mu.real  # (a real system: an estimate's rounding only)
    if M is None:
        base = _Map(op, dtype)

        def prod(X):
            return base.fwd(X) + X * (-mu)

        def rprod(X):
            return base.adj(X) + X * (-mu.conjugate() if isinstance(mu, complex) else -mu)

        shifted = None
    else:
        if isinstance(M, sparray):
            from ._construct import eye_array

            M = _compressed(M)
            M = M.astype(dtype) if M.dtype != dtype else M
            shifted = M - eye_array(n, format=M.format, dtype=dtype) * mu
        else:
            M = M.astype(dtype) if M.dtype != dtype else M
            shifted = M - _sd.eye(n, dtype=dtype) * mu
        sop = _Map(_Operand(shifted, func, "A"), dtype)
        prod, rprod = sop.fwd, sop.adj
    if shifted is None:
        norm1 = _norm1_of(prod, rprod, n, dtype)[0]
    elif isinstance(shifted, sparray):
        norm1 = _norm1(shifted)
    else:
        norm1 = _sd.linalg.matrix_norm(shifted, ord=1)
    n0 = 1 if B.ndim == 1 else B.shape[1]

    if not interval:
        t = 1.0
        if t * norm1 == 0:
            m_star, s = 0, 1
        else:
            m_star, s = _fragment_3_1(_NormInfo(prod, rprod, n, dtype, t, t * norm1), n0)
        return _simple_core(prod, B, t, mu, m_star, s)

    div = num - 1 if endpoint else num
    h = (stop - start) / div
    t0 = start
    tq = stop if endpoint else (num - 1) * h + start
    if h == 0:  # numpy's linspace for a zero step: every point is start
        t0 = tq = start
    q = num - 1
    t = tq - t0
    info = _NormInfo(prod, rprod, n, dtype, t, t * norm1)
    if t * norm1 == 0:
        m_star, s = 0, 1
    else:
        m_star, s = _fragment_3_1(info, n0)
    X = [None] * num
    X[0] = _simple_core(prod, B, t0, mu, m_star, s)
    if q <= s:
        if info.onenorm() == 0:
            m_star, s = 0, 1
        else:
            info.scale = 1.0 / q
            m_star, s = _fragment_3_1(info, n0)
            info.scale = 1.0
        for k in range(q):
            X[k + 1] = _simple_core(prod, X[k], h, mu, m_star, s)
    else:
        _interval_blocks(prod, X, h, mu, m_star, s, q)
    return _sd.stack(X)


# ---------------------------------------------------------------- norm

_MATRIX_ORDS = "'fro', 1, -1, 2, inf or -inf"


def _axis(axis, ndim, func):
    """axis: None (every axis), an int, or a tuple or list of 1 or 2 distinct ints."""
    if axis is None:
        return tuple(range(ndim))
    if C.is_int(axis):
        axes = (operator.index(axis),)
    elif isinstance(axis, (tuple, list)):
        bad = next((a for a in axis if not C.is_int(a)), None)
        if bad is not None:
            raise TypeError(f"{func}(): 'axis' must be an int or a tuple or list of ints, not {C.type_name(bad)}")
        axes = tuple(operator.index(a) for a in axis)
    else:
        raise TypeError(f"{func}(): 'axis' must be an int or a tuple or list of ints, not {C.type_name(axis)}")
    out = []
    for a in axes:
        if not -ndim <= a < ndim:
            raise ValueError(f"{func}(): axis {a} is out of bounds for an array of dimension {ndim}")
        out.append(a % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"{func}(): repeated axis")
    if len(out) not in (1, 2):
        raise ValueError(f"{func}(): 'axis' must name 1 or 2 axes, got {len(out)}")
    return tuple(out)


def _is_bool(v):
    return isinstance(v, bool) or (type(v).__name__ == "bool" and type(v).__module__ == "numpy")


def _matrix_ord(ord, func):
    if ord is None:
        return "fro"
    if _is_bool(ord) or not isinstance(ord, (str, numbers.Real)):
        raise TypeError(f"{func}(): 'ord' must be a str, an int or a float, not {C.type_name(ord)}")
    if isinstance(ord, str):
        if ord == "fro":
            return "fro"
    else:
        v = float(ord)
        if v in (1.0, -1.0, 2.0, _INF, -_INF):
            return v
    shown = repr(ord) if isinstance(ord, str) else _show(ord)
    raise ValueError(f"{func}(): 'ord' must be {_MATRIX_ORDS}, not {shown}")


def _show(v):
    """A number as sd.linalg's messages print it (inf, -inf, 3, 1.5)."""
    if C.is_int(v):
        return str(operator.index(v))
    x = float(v)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(x)


def _vector_ord(ord, func):
    if ord is None:
        return 2.0
    if _is_bool(ord) or not isinstance(ord, numbers.Real):
        raise TypeError(f"{func}(): 'ord' must be an int or a float, not {C.type_name(ord)}")
    v = float(ord)
    if math.isnan(v):
        raise ValueError(f"{func}(): 'ord' must not be NaN")
    return v


def _powered(a, p):
    """|x|^p entry by entry of the stored magnitudes a (x * x for p = 2, as vector_norm's terms)."""
    d = a.data
    return a._with_data(d * d if p == 2 else d**p)


def _line_norms(X, p, axis):
    """The vector p-norms of the lines of the 2-D sparse X along `axis` (a dense float64 array):
    sd.linalg.vector_norm's meaning (the implicit zeros count), the sums §2.6's; a p > 0 norm whose
    sum overflowed or whose every term is tiny is computed again from the magnitudes over the
    line's largest (vector_norm's second pass)."""
    length = X.shape[axis]
    if length == 0:  # an empty line's norm is 0 (vector_norm's)
        return _sd.zeros(X.shape[1 - axis])
    a = _abs(X)
    if p == _INF:
        return a.max(axis=axis)
    if p == -_INF:
        return a.min(axis=axis)
    if p == 0:
        return (X != 0).sum(axis=axis).astype(_sd.float64)
    if p == 1:
        return a.sum(axis=axis)
    if p < 0:  # a zero term is inf: such a line's norm is 0
        s = _powered(a, p).sum(axis=axis)
        nonzero = (X != 0).sum(axis=axis)
        s = _sd.where((nonzero < length) & ~_sd.isnan(s), _INF, s)
        return s ** (1.0 / p)
    s = _powered(a, p).sum(axis=axis)
    out = _sd.sqrt(s) if p == 2 else s ** (1.0 / p)
    amax = a.max(axis=axis)
    top = amax * amax if p == 2 else amax**p
    redo = (_sd.isinf(s) | ((amax > 0) & (top < 2.0**-969))) & _sd.isfinite(amax)
    if bool(_sd.any(redo)):
        div = _sd.where(redo & (amax > 0), amax, 1.0)
        # each stored magnitude over its line's largest (a division, as vector_norm: a reciprocal
        # of a subnormal largest overflows, and its product rounds once more)
        line = C._core._expand(a._indptr, a.shape[0], a.nnz) if axis == 1 else a._indices
        r = a._with_data(a.data / div[line])
        t = _powered(r, p).sum(axis=axis)
        again = div * (_sd.sqrt(t) if p == 2 else t ** (1.0 / p))
        out = _sd.where(redo, again, out)
    return _sd.where(_sd.isinf(amax), _INF, out)


def norm(x, /, *, ord=None, axis=None):
    """The norm of the sparse array x (scipy's meaning): a matrix norm ('fro' (the default), 1, -1,
    2, inf, -inf) of a 2-D x over both axes, a vector norm (2 (the default), inf, -inf, 0 or any
    other p) of a 1-D x or of every line of a 2-D x along one axis (a dense float64 array). ord's
    rules and messages are sd.linalg.matrix_norm's and vector_norm's; ord=2 of a matrix is its
    largest singular value (svds)."""
    func = "norm"
    if not isinstance(x, sparray):
        hint = "; for a dense array use sd.linalg.matrix_norm or sd.linalg.vector_norm" if isinstance(
            x, _sd.ndarray) else ""  # fmt: skip
        raise TypeError(f"{func}(): 'x' must be a sparse array, not {C.type_name(x)}{hint}")
    axes = _axis(axis, x.ndim, func)
    X = x.tocsr() if x.format != "csr" else x
    if X.dtype == _sd.bool or X.dtype == _sd.int64:
        X = X.astype(_sd.float64)
    if len(axes) == 2:
        o = _matrix_ord(ord, func)
        transposed = axes == (1, 0)
        return _matrix_norm(X.T.tocsr() if transposed else X, o, func)
    o = _vector_ord(ord, func)
    if X.ndim == 1:
        return float(_line_norms(X.reshape((1, X.shape[0])).tocsr(), o, 1)[0])
    return _line_norms(X, o, axes[0])


def _matrix_norm(X, o, func):
    m, n = X.shape
    if o == "fro":
        return float(_sd.linalg.vector_norm(X.data)) if X.nnz else 0.0
    if m == 0 or n == 0:
        return 0.0
    if o == 2:
        if X.nnz and not bool(_sd.all(_sd.isfinite(X.data))):
            raise ValueError(f"{func}(): 'x' contains NaN or infinity")
        if not X.nnz or not bool(_sd.any(X.data != 0)):
            return 0.0
        if min(m, n) == 1:  # a single row or column: its vector 2-norm
            return float(_sd.linalg.vector_norm(X.data))
        from ._eigen import svds

        r = svds(X, k=1, return_singular_vectors=False)
        if not r.converged:
            from ..linalg import LinAlgError

            raise LinAlgError(f"{func}(): the largest singular value did not converge")
        return float(r.S[0])
    a = _abs(X)
    sums = a.sum(axis=0 if abs(o) == 1 else 1)
    return float(_sd.max(sums) if o > 0 else _sd.min(sums))


__all__ = ["expm", "expm_multiply", "onenormest", "norm"]
