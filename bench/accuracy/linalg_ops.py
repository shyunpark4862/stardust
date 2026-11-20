"""The linalg operations of each library the accuracy measurement compares, on numpy arrays.

ops(lib) -> an object whose methods return numpy results in one shape for every library; a
method returns None where the library has no such function. lu returns the reconstruction
P L U (each library orders the permutation its own way), qr and cholesky likewise the factors
the residual needs.
"""

from __future__ import annotations

import numpy as np


class _SD:
    def __init__(self):
        import stardust as sd

        self.L, self.a = sd.linalg, sd.asarray

    def solve(self, a, b):
        return np.asarray(self.L.solve(self.a(a), self.a(b)))

    def lu_solve(self, a, b):
        f = self.L.lu(self.a(a))
        return np.asarray(self.L.lu_solve(f.L, f.U, f.perm, self.a(b)))

    def inv(self, a):
        return np.asarray(self.L.inv(self.a(a)))

    def lstsq(self, a, b):
        return np.asarray(self.L.lstsq(self.a(a), self.a(b)).solution)

    def lu(self, a):
        f = self.L.lu(self.a(a))
        lo, up, p = np.asarray(f.L), np.asarray(f.U), np.asarray(f.perm)
        rec = np.empty_like(a)
        rec[p] = lo @ up  # a[perm] = L U
        return rec

    def qr(self, a):
        f = self.L.qr(self.a(a))
        return np.asarray(f.Q), np.asarray(f.R)

    def cholesky(self, a):
        return np.asarray(self.L.cholesky(self.a(a)))

    def svd(self, a):
        f = self.L.svd(self.a(a))
        return np.asarray(f.U), np.asarray(f.S), np.asarray(f.Vh)

    def eigh(self, a):
        f = self.L.eigh(self.a(a))
        return np.asarray(f.eigenvalues), np.asarray(f.eigenvectors)

    def eig(self, a):
        f = self.L.eig(self.a(a))
        return np.asarray(f.eigenvalues), np.asarray(f.eigenvectors)

    def det(self, a):
        return float(self.L.det(self.a(a)))

    def eigvalsh(self, a):
        return np.asarray(self.L.eigvalsh(self.a(a)))

    def svdvals(self, a):
        return np.asarray(self.L.svdvals(self.a(a)))

    def eigvals(self, a):
        return np.asarray(self.L.eigvals(self.a(a)))

    def expm(self, a):
        return np.asarray(self.L.expm(self.a(a)))


class _NumPy:
    """numpy.linalg; scipy.linalg for what numpy has not (lu, lu_solve, expm)."""

    def solve(self, a, b):
        return np.linalg.solve(a, b)

    def lu_solve(self, a, b):
        import scipy.linalg as sl

        return sl.lu_solve(sl.lu_factor(a), b)

    def inv(self, a):
        return np.linalg.inv(a)

    def lstsq(self, a, b):
        return np.linalg.lstsq(a, b, rcond=None)[0]

    def lu(self, a):
        import scipy.linalg as sl

        p, lo, up = sl.lu(a)
        return p @ lo @ up

    def qr(self, a):
        return np.linalg.qr(a, mode="reduced")

    def cholesky(self, a):
        return np.linalg.cholesky(a)

    def svd(self, a):
        return np.linalg.svd(a, full_matrices=False)

    def eigh(self, a):
        return np.linalg.eigh(a)

    def eig(self, a):
        return np.linalg.eig(a)

    def det(self, a):
        return float(np.linalg.det(a))

    def eigvalsh(self, a):
        return np.linalg.eigvalsh(a)

    def svdvals(self, a):
        return np.linalg.svd(a, compute_uv=False)

    def eigvals(self, a):
        return np.linalg.eigvals(a)

    def expm(self, a):
        import scipy.linalg as sl

        return sl.expm(a)


class _R:
    """base R (solve, qr, chol, svd, eigen, det) and Matrix::expm (Matrix is one of R's
    recommended packages). qr is LAPACK's (geqp3; R's default LINPACK qr stops at a rank
    tolerance of 1e-7, another computation), least squares its qr.coef. R has no LU in base:
    lu and lu_solve are None."""

    def _ev(self, expr, **data):
        from . import rsession

        return rsession.evaluate_named(expr, data)

    @staticmethod
    def _m(name, a):
        return f"{name} <- matrix({name}, nrow = {a.shape[0]}, byrow = TRUE); "

    def solve(self, a, b):
        return self._ev(self._m("A", a) + "solve(A, B)", A=a, B=b)

    def lu_solve(self, a, b):
        return None

    def inv(self, a):
        n = a.shape[0]
        return self._ev(self._m("A", a) + "as.numeric(t(solve(A)))", A=a).reshape(n, n)

    def lstsq(self, a, b):
        return self._ev(self._m("A", a) + "qr.coef(qr(A, LAPACK = TRUE), B)", A=a, B=b)

    def lu(self, a):
        return None

    def qr(self, a):
        m, n = a.shape
        v = self._ev(self._m("A", a) + "q <- qr(A, LAPACK = TRUE); Q <- qr.Q(q); R <- qr.R(q); P <- q$pivot; "
                     "R2 <- R; R2[, P] <- R; c(as.numeric(t(Q)), as.numeric(t(R2)))", A=a)  # fmt: skip
        return v[: m * n].reshape(m, n), v[m * n :].reshape(n, n)  # R's column pivoting undone in R

    def cholesky(self, a):
        n = a.shape[0]
        return self._ev(self._m("A", a) + "as.numeric(chol(A))", A=a).reshape(n, n)  # t(U) row-major = L

    def svd(self, a):
        m, n = a.shape
        k = min(m, n)
        v = self._ev(self._m("A", a) + "s <- svd(A); c(as.numeric(t(s$u)), s$d, as.numeric(s$v))", A=a)
        return v[: m * k].reshape(m, k), v[m * k : m * k + k], v[m * k + k :].reshape(k, n)

    def eigh(self, a):
        n = a.shape[0]
        v = self._ev(self._m("A", a) + "e <- eigen(A, symmetric = TRUE); c(e$values, as.numeric(t(e$vectors)))", A=a)
        return v[:n], v[n:].reshape(n, n)

    def eig(self, a):
        n = a.shape[0]
        v = self._ev(self._m("A", a) + "e <- eigen(A); c(Re(e$values), Im(e$values), "
                     "as.numeric(t(Re(e$vectors))), as.numeric(t(Im(e$vectors))))", A=a)  # fmt: skip
        w = v[:n] + 1j * v[n : 2 * n]
        vec = v[2 * n : 2 * n + n * n].reshape(n, n) + 1j * v[2 * n + n * n :].reshape(n, n)
        return w, vec

    def det(self, a):
        return float(self._ev(self._m("A", a) + "det(A)", A=a)[0])

    def eigvalsh(self, a):
        return np.sort(self._ev(self._m("A", a) + "eigen(A, symmetric = TRUE, only.values = TRUE)$values", A=a))

    def svdvals(self, a):
        return self._ev(self._m("A", a) + "svd(A, 0, 0)$d", A=a)

    def eigvals(self, a):
        n = a.shape[0]
        v = self._ev(self._m("A", a) + "e <- eigen(A, only.values = TRUE)$values; c(Re(e), Im(e))", A=a)
        return v[:n] + 1j * v[n:]

    def expm(self, a):
        n = a.shape[0]
        expr = self._m("A", a) + "suppressMessages(library(Matrix)); as.numeric(t(as.matrix(expm(Matrix(A)))))"
        return self._ev(expr, A=a).reshape(n, n)


def ops(lib):
    return {"sd": _SD, "numpy": _NumPy, "R": _R}[lib]()
