"""Test matrices and Arb reference values of sd.linalg (tools/linalg_ref.py; layout:
tests/reffile.py), shared by the accuracy measurement (bench/accuracy/linalg.py) and
tests/linalg/test_reference.py.

tests/linalg/ref/<family>.npz: one point per matrix a (and right-hand side b), its kappa_2,
and per method the Arb reference hi + lo with its condition number cond:

    square     solve (x = A^-1 b), det, eigvals (complex, for n <= 50)
    symmetric  eigvalsh (spd: positive definite, for cholesky)
    rect       lstsq (the least-squares solution), svdvals
    expm       expm (exp(A) entry by entry)
"""

import pathlib

import numpy as np

from .. import reffile
from .conftest import RESIDUAL_C  # noqa: F401 (the contract's c: the measurement reads it here)

REF = pathlib.Path(__file__).parent / "ref"
FAMILIES = ("square", "symmetric", "rect", "expm")
EPS = float(np.finfo(np.float64).eps)


def matrices(family):
    """The family's points as dicts: the inputs (a, b, kappa, spd as the family has them) and
    per method "<method>_hi", "_lo" and "<method>_cond" where the point has that reference."""
    r = reffile.read(REF / f"{family}.npz")
    out = []
    for i in range(r.points):
        g = {k: v[i] for k, v in r.inputs.items()}
        for m, mm in r.methods.items():
            hit = np.flatnonzero(mm.index == i)
            if hit.size:
                j = int(hit[0])
                g[f"{m}_hi"], g[f"{m}_lo"] = mm.hi[j], mm.lo[j]
                if "cond" in mm.fields:
                    g[f"{m}_cond"] = mm.fields["cond"][j]
        out.append(g)
    return out


# ------------------------------------------------------------------ the residual contract
# docs/design/linalg.md §13: every residual in units of n eps times its scale is at most
# RESIDUAL_C (n = max(m, n), infinity norms):
#     solve, lu_solve      ||A x - b|| / (n eps (||A|| ||x|| + ||b||))
#     inv                  ||A X - I|| / (n eps (||A|| ||X|| + 1))
#     lstsq                ||A^T (A x - b)|| / (n eps ||A^T|| (||A|| ||x|| + ||b||))
#     lu, qr, cholesky,    ||A - the factors' product|| / (n eps ||A||) (eigh: ||A V - V diag(w)||;
#       svd, eigh,         qr_pivoted: ||A[:, perm] - Q R||), and ||Q^H Q - I|| / (n eps) for the
#       qr_pivoted         orthogonal factors
#     eig                  ||A V - V diag(w)|| / (n eps ||A|| ||V||)

RESIDUAL = ("solve", "lu_solve", "inv", "lstsq", "lu", "qr", "qr_pivoted", "cholesky", "svd", "eigh", "eig")


def inf_norm(a) -> float:
    """The infinity norm: max |v_i| of a vector, the largest row sum of |a_ij| of a matrix."""
    a = np.asarray(a)
    if not a.size:
        return 0.0
    if a.ndim == 1:
        return float(np.abs(a).max())
    return float(np.abs(a).sum(axis=-1).max())


def residual_matrices(fn):
    """The points a residual sweep runs on: (label, a, b or None)."""
    fams = {"eigh": ("symmetric",), "cholesky": ("symmetric",), "lstsq": ("rect",),
            "svd": ("square", "rect"), "qr": ("square", "rect"), "qr_pivoted": ("square", "rect")}  # fmt: skip
    out = []
    for fam in fams.get(fn, ("square",)):
        for g in matrices(fam):
            if fn == "cholesky" and not bool(g["spd"]):
                continue
            out.append(g)
    return out


def residual(fn, a, b, result):
    """(backward, orth) of a residual sweep's result (the output of SD's method or a
    comparator's in the same shape): the residual over n eps and its scale, and the orthogonal
    factors' ||Q^H Q - I|| / (n eps) (None where there is none). The contract is
    max(backward, orth) <= RESIDUAL_C."""
    m, n = a.shape
    nn = max(m, n)
    na = inf_norm(a)
    orth = None
    if fn in ("solve", "lu_solve"):
        x = result
        res = inf_norm(a @ x - b) / (nn * EPS * (na * inf_norm(x) + inf_norm(b)))
    elif fn == "inv":
        x = result
        res = inf_norm(a @ x - np.eye(n)) / (nn * EPS * (na * inf_norm(x) + 1))
    elif fn == "lstsq":
        x = result
        res = inf_norm(a.T @ (a @ x - b)) / (nn * EPS * inf_norm(a.T) * (na * inf_norm(x) + inf_norm(b)))
    elif fn == "lu":
        res = inf_norm(a - result) / (nn * EPS * na)
    elif fn in ("qr", "qr_pivoted"):
        q, r, *perm = result
        ap = a[:, perm[0]] if perm else a
        res = inf_norm(ap - q @ r) / (nn * EPS * na)
        orth = inf_norm(q.T @ q - np.eye(q.shape[1])) / (nn * EPS)
    elif fn == "cholesky":
        lo = result
        res = inf_norm(a - lo @ lo.T) / (nn * EPS * na)
    elif fn == "svd":
        u, s, vh = result
        res = inf_norm(a - (u * s) @ vh) / (nn * EPS * na)
        orth = max(inf_norm(u.T @ u - np.eye(u.shape[1])), inf_norm(vh @ vh.T - np.eye(vh.shape[0]))) / (nn * EPS)
    elif fn == "eigh":
        w, v = result
        res = inf_norm(a @ v - v * w) / (nn * EPS * na)
        orth = inf_norm(v.T @ v - np.eye(n)) / (nn * EPS)
    elif fn == "eig":
        w, v = result
        res = inf_norm(a @ v - v * w) / (nn * EPS * na * inf_norm(v))
    else:
        raise ValueError(fn)
    return res, orth


def call(ops, fn, a, b):
    """ops' result of a residual sweep's function at (a, b), or None where it has none."""
    if fn in ("solve", "lu_solve", "lstsq"):
        return getattr(ops, fn)(a, b)
    return getattr(ops, fn)(a)


class SD:
    """sd.linalg on numpy arrays, in the shapes the residuals take (bench/accuracy/linalg_ops.py
    has the comparators' adapters)."""

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

    def qr_pivoted(self, a):
        f = self.L.qr_pivoted(self.a(a))
        return np.asarray(f.Q), np.asarray(f.R), np.asarray(f.perm)

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
