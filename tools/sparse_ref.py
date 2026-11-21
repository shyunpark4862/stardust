"""Reference values for sd.sparse (docs/design/sparse.md §4, §7) on the test matrices of
tests/sparse/matrices.py: exact rationals and Arb (python-flint).

    .venv/bin/python tools/sparse_ref.py                  # every family, on 8 processes
    .venv/bin/python tools/sparse_ref.py spmv solve       # only these families
    .venv/bin/python tools/sparse_ref.py --jobs 4 eig     # 4 processes
    .venv/bin/python tools/sparse_ref.py --list           # the families and their points

writes tests/sparse/ref/<family>.npz (layout: tests/reffile.py), read by tests/sparse/refdata.py.
Every point names its matrix (input "matrix", the id in tests/sparse/matrices.py) and the
matrix's digest (input "digest", matrices.digest), so a changed generator shows. The matrix is
its canonical form (duplicates summed left to right in input order, docs/design/sparse.md §2.3):
the stored doubles, taken as exact rationals. References are double-doubles hi + lo.

    spmv    y = A x and z = A^T w (plain transpose) for two x, w each: ones and a fixed uniform
            vector (complex for a complex matrix); exact dyadic sums, rounded once
    sums    the row sums, the column sums and the total of the stored values; exact
    norms   the 1-, -1-, inf-, -inf- and Frobenius norms (exact sums; |a| of a complex entry and
            the square root in Arb), and the 2-norm (sqrt of the largest eigenvalue of A^H A,
            Arb) for n <= 220
    solve   x = A^-1 b of square nonsingular matrices (n <= 2000), b = matrices.rhs(a):
            residual-verified (below); with the condition number kappa_inf ~ ||A|| ||A^-1|| (cond)
    lstsq   the least-squares solution of the full-rank rectangular matrices, from the normal
            equations A^T A x = A^T b formed exactly and solved as in solve
    eig     the 6 smallest and 6 largest eigenvalues of symmetric / Hermitian matrices: Arb's
            acb_mat.eig for n <= 220, the closed form sum_d (2 - 2 cos(k_d pi / (g + 1))) for
            the Dirichlet Laplacians; every singular value of small rectangular matrices
            (sqrt of the eigenvalues of the smaller Gram matrix); and every eigenvalue of small
            nonsymmetric real and complex matrices (acb_mat.eig, sorted by real then imaginary
            part: method "eigvals", complex; milestone SP6)
    expm    exp(s A) entry by entry (arb_mat.exp / acb_mat.exp) for n <= 200, s a power of two
            putting ||s A||_1 near 1 (input "scale"), and the same at ||s A||_1 near 30
    expmv   exp(s A) b on expm's points, b = matrices.rhs(a) (input "b"): t = 1 (method "expmv")
            and t = 0, 1/4, 1/2, 3/4, 1 (method "interval", the times one after another), the
            products in Arb (milestone SP7, expm_multiply)

Settling: exact sums are exact; an Arb value is settled to 2^-110 relative (eigenvalues,
singular values and the 2-norm to 2^-110 ||A||, expm's entries to 2^-110 of the largest entry,
expmv's to 2^-110 max_i (|exp(t s A)| |b|)_i:
tools/arbref.py's floor). A solve is verified as Rump's: R = inv(A) in double, alpha >=
||I - R A||_inf bounded with the rounding errors of the double products (gamma_n |R||A|), then
||A^-1||_inf <= ||R||_inf / (1 - alpha); x is refined in double steps x1 + x2 + ... whose residual
r = b - A x is computed exactly, until ||A^-1|| ||r|| <= 2^-110 ||x||_inf: the stored hi + lo is
x rounded, within 2^-110 ||x|| of the solution (normwise, like tests/linalg/ref). A point that
does not settle (alpha >= 1, no convergence in 12 steps, the time limit) is dropped and
reported. Workers are forked, replaced every 8 points (maxtasksperchild), each point under a
time limit (SIGALRM). No mpmath (the project's rule). The console output also goes to
build/logs/sparse_ref/.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import signal
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arbref  # noqa: E402
from arbref import NotSettled, rigorous, to_dd  # noqa: E402

OUT = ROOT / "tests" / "sparse" / "ref"
TOOL = "tools/sparse_ref.py"
TARGET = 2.0**-110
U = 2.0**-53
SECONDS = 300  # a point's time limit
EIG_MAX = 220  # the largest n whose eigenvalues / 2-norm Arb computes densely
EXPM_MAX = 200
K = 6  # eigenvalues at each end


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def M():
    from tests.sparse import matrices

    return matrices


# ---------------------------------------------------------------- exact dyadic arithmetic
# A double (or a product, or a sum of them) is n * 2^e exactly: (n, e) with Python ints.


def dy(v: float):
    n, d = float(v).as_integer_ratio()
    return n, -(d.bit_length() - 1)


def dsum(terms):
    """The exact sum of dyadics [(n, e)] as a dyadic."""
    if not terms:
        return 0, 0
    emin = min(e for _, e in terms)
    return sum(n << (e - emin) for n, e in terms), emin


def dfrac(d) -> Fraction:
    n, e = d
    return Fraction(n << e) if e >= 0 else Fraction(n, 1 << -e)


def dd_of(q: Fraction):
    """(hi, lo) of an exact rational: hi rounded to nearest (ties to even), lo the rest rounded."""
    hi = float(q)  # int / int true division rounds correctly
    if math.isinf(hi):
        return hi, 0.0
    return hi, float(q - Fraction(hi))


def up(q: Fraction) -> float:
    """A double >= |q|."""
    q = abs(q)
    f = float(q)
    return f if Fraction(f) >= q else math.nextafter(f, math.inf)


class Exact:
    """A matrix with exact dyadic entries (complex: real and imaginary parts), by rows.

    rows[i] = [(j, re, im)], re / im dyadics (im None for a real matrix)."""

    def __init__(self, shape, rows, complex_):
        self.shape, self.rows, self.complex = shape, rows, complex_

    @classmethod
    def of(cls, a):
        r, c, v = a.canonical()
        cplx = np.iscomplexobj(v)
        v = v.astype(np.complex128 if cplx else np.float64)
        rows = [[] for _ in range(a.shape[0])]
        for i, j, x in zip(r.tolist(), c.tolist(), v.tolist()):
            rows[i].append((j, dy(x.real), dy(x.imag)) if cplx else (j, dy(x), None))
        return cls(a.shape, rows, cplx)

    def transpose(self):
        rows = [[] for _ in range(self.shape[1])]
        for i, row in enumerate(self.rows):
            for j, re, im in row:
                rows[j].append((i, re, im))
        return Exact((self.shape[1], self.shape[0]), rows, self.complex)

    def dense(self) -> np.ndarray:
        """The entries rounded to doubles (exact for a matrix of doubles)."""
        out = np.zeros(self.shape, np.complex128 if self.complex else np.float64)
        for i, row in enumerate(self.rows):
            for j, re, im in row:
                out[i, j] = float(dfrac(re)) + (1j * float(dfrac(im)) if im is not None else 0)
        return out

    def matvec(self, x):
        """A x exactly for x a list of (re, im) dyadics: a list of (re, im) dyadics (im None if
        both are real)."""
        cplx = self.complex or x_complex(x)
        out = []
        for row in self.rows:
            tr, ti = [], []
            for j, ar, ai in row:
                xr, xi = x[j]
                tr.append(mul(ar, xr))
                if ai is not None and xi is not None:
                    tr.append(neg(mul(ai, xi)))
                if ai is not None:
                    ti.append(mul(ai, xr))
                if xi is not None:
                    ti.append(mul(ar, xi))
            out.append((dsum(tr), dsum(ti) if cplx else None))
        return out


def mul(a, b):
    return a[0] * b[0], a[1] + b[1]


def neg(a):
    return -a[0], a[1]


def x_complex(x):
    return any(xi is not None for _, xi in x)


def vec_dy(v):
    v = np.asarray(v)
    if np.iscomplexobj(v):
        return [(dy(z.real), dy(z.imag)) for z in v.tolist()]
    return [(dy(z), None) for z in v.tolist()]


def dd_vec(ys):
    """(hi, lo) arrays of a list of (re, im) dyadics."""
    cplx = any(im is not None for _, im in ys)
    hi = np.zeros(len(ys), np.complex128 if cplx else np.float64)
    lo = np.zeros_like(hi)
    for k, (re, im) in enumerate(ys):
        h, lw = dd_of(dfrac(re))
        if cplx:
            hi_, lo_ = dd_of(dfrac(im)) if im is not None else (0.0, 0.0)
            hi[k], lo[k] = complex(h, hi_), complex(lw, lo_)
        else:
            hi[k], lo[k] = h, lw
    return hi, lo


# ---------------------------------------------------------------- families


def _vector(a, length, kind, seed):
    if kind == "ones":
        return np.ones(length, np.complex128 if a.data.dtype == np.complex128 else np.float64)
    g = np.random.Generator(np.random.PCG64([20261120, seed, length]))
    v = g.uniform(-1, 1, length)
    if a.data.dtype == np.complex128:
        v = v + 1j * g.uniform(-1, 1, length)
    return v


def spmv_points():
    m = M()
    keep = [i for i in m.ids(small=True) if "coo" not in m.get(i).tags or "nan_inf" not in i]
    keep += ["lap2d/100", "long_rows/9000", "lap3d/22"]
    out = []
    for i in keep:
        a = m.get(i)
        for kind in ("ones", "random"):
            out.append(
                {"matrix": i, "kind": kind, "x": _vector(a, a.shape[1], kind, 1), "w": _vector(a, a.shape[0], kind, 2)}
            )
    return out


def spmv_ref(p):
    a = M().get(p["matrix"])
    e = Exact.of(a)
    y = dd_vec(e.matvec(vec_dy(p["x"])))
    z = dd_vec(e.transpose().matvec(vec_dy(p["w"])))
    return {"matvec": y, "rmatvec": z}


def sums_points():
    m = M()
    keep = [i for i in m.ids(small=True) if i != "coo/nan_inf"] + ["lap2d/100", "long_rows/9000"]
    return [{"matrix": i} for i in keep]


def _sums_rows(e):
    out = []
    for row in e.rows:
        tr = [re for _, re, _ in row]
        ti = [im for _, _, im in row if im is not None]
        out.append((dsum(tr), dsum(ti) if e.complex else None))
    return out


def sums_ref(p):
    a = M().get(p["matrix"])
    e = Exact.of(a)
    rows = _sums_rows(e)
    cols = _sums_rows(e.transpose())
    tot_r = dsum([re for re, _ in rows])
    tot = (tot_r, dsum([im for _, im in rows]) if e.complex else None)
    return {"row_sums": dd_vec(rows), "col_sums": dd_vec(cols), "total": dd_vec([tot])}


def norms_points():
    m = M()
    keep = [i for i in m.ids(small=True) if i != "coo/nan_inf" and "bool" not in m.get(i).tags]
    keep += ["lap2d/100"]
    return [{"matrix": i} for i in keep]


def _abs_arb(re, im):
    from flint import arb, fmpq

    def A(d):
        q = dfrac(d)
        return arb(fmpq(q.numerator, q.denominator))

    if im is None:
        return abs(A(re))
    return (A(re) ** 2 + A(im) ** 2).sqrt()


def norms_ref(p):
    from flint import arb, fmpq

    a = M().get(p["matrix"])
    e = Exact.of(a)
    m, n = a.shape
    out = {}

    def line_norm(ex, largest):
        def f():  # the largest (smallest) sum_j |a_ij| over the rows of ex, as an arb
            sums = [sum((_abs_arb(re, im) for _, re, im in row), arb(0)) for row in ex.rows]
            best = max(sums, key=lambda v: float(v.mid())) if largest else min(sums, key=lambda v: float(v.mid()))
            for v in sums:  # a ball the precision cannot order against the pick joins it
                if v.overlaps(best):
                    best = best.union(v)
            return best

        return f

    if m and n:
        et = e.transpose()
        for name, ex, largest in (("1", et, True), ("-1", et, False), ("inf", e, True), ("-inf", e, False)):
            out[name] = _settle1(line_norm(ex, largest))
    else:
        for name in ("1", "-1", "inf", "-inf"):
            out[name] = (np.array([0.0]), np.array([0.0]))

    def fro():
        t = []
        for row in e.rows:
            for _, re, im in row:
                t.append(mul(re, re))
                if im is not None:
                    t.append(mul(im, im))
        q = dfrac(dsum(t))
        return arb(fmpq(q.numerator, q.denominator)).sqrt()

    out["fro"] = _settle1(fro)
    if a.nnz == 0:
        out["2"] = (np.array([0.0]), np.array([0.0]))
    elif min(m, n) <= EIG_MAX:
        d = e.dense()
        scale = max(abs(float(out["fro"][0][0])), 1e-300)

        def two():
            return largest_singular(d)

        v = rigorous(two, floor=scale)
        out["2"] = tuple(np.array([x]) for x in to_dd(v))
    return out


def largest_singular(d: np.ndarray):
    """||A||_2 as an arb: the largest eigenvalue of G = A^H A (or A A^H, the smaller) by Rayleigh
    quotients refined with two inverse iterations at the current precision, enclosed with
    Kato-Temple's bound lambda_1 in [rho, rho + eps^2 / (rho - a)], eps = ||G v - rho v|| / ||v||
    and a >= lambda_2 from the double eigenvalues plus their error bound (Weyl: 4 n u ||G||_F). A
    largest eigenvalue that is not simple to that margin raises NotSettled."""
    from flint import acb_mat, arb

    g = d.conj().T @ d if d.shape[1] <= d.shape[0] else d @ d.conj().T
    n = g.shape[0]
    w, vecs = np.linalg.eigh(g)
    gnorm = float(np.linalg.norm(g))
    a_up = (w[-2] if n > 1 else -gnorm) + 4 * (n + 2) * U * gnorm + 1e-300
    if w[-1] - a_up <= 1e-6 * gnorm:
        raise NotSettled("the largest singular value is not simple")
    A = _acb(d)
    G = (A.conjugate().transpose() * A) if d.shape[1] <= d.shape[0] else (A * A.conjugate().transpose())
    v = _acb(vecs[:, -1:])
    for _ in range(2):  # inverse iteration at the Rayleigh quotient's midpoint: v is any test vector
        vh = v.conjugate().transpose()
        rho = ((vh * G * v)[0, 0] / (vh * v)[0, 0]).real.mid()
        sh = G - _acb(np.eye(n)) * rho
        try:
            z = sh.solve(v)
        except ZeroDivisionError:  # rho is an eigenvalue exactly at this precision: v is its vector
            break
        mx = max(abs(float(z[i, 0].real.mid())) + abs(float(z[i, 0].imag.mid())) for i in range(n))
        v = acb_mat([[z[i, 0].mid() / mx] for i in range(n)])  # exact midpoints, full precision
    vh = v.conjugate().transpose()
    nv = (vh * v)[0, 0].real
    rho = (vh * G * v)[0, 0].real / nv
    r = G * v - v * rho
    eps2 = (r.conjugate().transpose() * r)[0, 0].real / nv
    gap = rho - arb(a_up)
    if not gap > 0:
        raise NotSettled("no gap")
    lam = rho.union(rho + eps2 / gap)
    return lam.sqrt()


def _acb(d):
    from flint import acb, acb_mat

    if np.iscomplexobj(d):
        return acb_mat([[acb(z.real, z.imag) for z in row] for row in d.tolist()])
    return acb_mat(d.tolist())


def _settle1(fn):
    v = rigorous(fn)
    h, lo = to_dd(v)
    return np.array([h]), np.array([lo])


# ---------------------------------------------------------------- verified solves


def _gamma(k):
    return k * U / (1 - k * U)


def inverse_norm_bound(ad: np.ndarray, exact_delta: bool):
    """(an upper bound of ||A^-1||_inf, kappa_inf estimate, R) for the exact matrix A whose
    doubles are ad (exact_delta: |A - ad| <= u |ad| entrywise, the entries were rounded). None
    when the bound fails (alpha >= 1)."""
    n = ad.shape[0]
    with np.errstate(all="ignore"):
        try:
            r = np.linalg.inv(ad)
        except np.linalg.LinAlgError:
            return None
        c = r @ ad
        e = np.eye(n) - c
        mabs = np.abs(r) @ np.abs(ad)
    k = 4 * n + 8 if np.iscomplexobj(ad) else n + 2  # the inner products' terms (complex: 4 real each)
    g = _gamma(k)
    bound = np.abs(e) * (1 + 2 * U) + mabs * (1 + g) * (g + (U if exact_delta else 0.0) * (1 + g))
    alpha = float(bound.sum(axis=1).max()) * (1 + _gamma(n + 2))
    if not np.isfinite(alpha) or alpha >= 0.5:
        return None
    rn = float(np.abs(r).sum(axis=1).max()) * (1 + _gamma(n + 2))
    an = float(np.abs(ad).sum(axis=1).max())
    return rn / (1 - alpha), an * rn, r


def verified_solve(ex: Exact, b_dy, steps=12):
    """x of ex x = b (b a list of (re, im) dyadics) to 2^-110 ||x||_inf, as a list of Fractions
    (complex: pairs), and kappa_inf; NotSettled when it cannot be verified."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = ex.shape[0]
    ad = ex.dense()
    exact_delta = any(
        dfrac(re) != Fraction(float(dfrac(re))) or (im is not None and dfrac(im) != Fraction(float(dfrac(im))))
        for row in ex.rows
        for _, re, im in row
    )
    got = inverse_norm_bound(ad, exact_delta)
    if got is None:
        raise NotSettled("alpha >= 1/2: the inverse cannot be bounded in double")
    inv_bound, kappa, _ = got
    lu = spla.splu(sp.csc_array(ad))
    cplx = ex.complex or x_complex(b_dy)
    x = [((0, 0), (0, 0) if cplx else None) for _ in range(n)]
    for _ in range(steps):
        ax = ex.matvec(x)
        res = []
        for (br, bi), (yr, yi) in zip(b_dy, ax):
            rr = dsum([br, neg(yr)])
            ri = dsum(([bi] if bi is not None else []) + ([neg(yi)] if yi is not None else [])) if cplx else None
            res.append((rr, ri))
        # ||r||_inf bounded above exactly (|re + i im| <= |re| + |im|), ||x||_inf below
        rnorm = max(((up(dfrac(rr)) + (up(dfrac(ri)) if ri is not None else 0.0)) * (1 + 2 * U) for rr, ri in res),
                    default=0.0)  # fmt: skip
        xnorm = max((abs(complex(float(dfrac(xr)), float(dfrac(xi)) if xi is not None else 0.0)) for xr, xi in x),
                    default=0.0) * (1 - 4 * U)  # fmt: skip
        if inv_bound * rnorm <= TARGET * xnorm * 0.5 or rnorm == 0.0:
            return x, kappa
        rd = np.array([complex(float(dfrac(rr)), float(dfrac(ri))) if cplx else float(dfrac(rr)) for rr, ri in res])
        d = lu.solve(rd)
        if not np.all(np.isfinite(d)):
            break
        x = [
            (dsum([xr, dy(float(np.real(dv)))]), dsum([xi, dy(float(np.imag(dv)))]) if cplx else None)
            for (xr, xi), dv in zip(x, d.tolist())
        ]
    raise NotSettled("the refinement did not reach 2^-110 ||x||")


def solve_points():
    m = M()
    out = []
    for i in m.ids(small=True):
        a = m.get(i)
        t = a.tags
        if a.shape[0] != a.shape[1] or a.shape[0] == 0 or t & {"singular", "struct_singular", "coo", "cancel", "empty"}:
            continue
        out.append({"matrix": i, "b": m.rhs(a)})
    return out


def solve_ref(p):
    a = M().get(p["matrix"])
    x, kappa = verified_solve(Exact.of(a), vec_dy(p["b"]))
    return {"solve": dd_vec(x), "cond": np.array([kappa])}


def lstsq_points():
    m = M()
    return [{"matrix": i, "b": m.rhs(m.get(i))} for i in m.ids("rect") if "rank_deficient" not in m.get(i).tags
            and m.get(i).shape[0] >= m.get(i).shape[1]]  # fmt: skip


def lstsq_ref(p):
    a = M().get(p["matrix"])
    e = Exact.of(a)
    et = e.transpose()
    n = a.shape[1]
    # G = A^T A by exact column dot products (a real matrix), c = A^T b
    cols = et.rows
    colmap = [dict((i, re) for i, re, _ in col) for col in cols]
    rows = []
    for j in range(n):
        row = []
        for k in range(n):
            common = colmap[j].keys() & colmap[k].keys()
            if common:
                row.append((k, dsum([mul(colmap[j][i], colmap[k][i]) for i in common]), None))
        rows.append(row)
    g = Exact((n, n), rows, False)
    c = et.matvec(vec_dy(p["b"]))
    x, kappa = verified_solve(g, c)
    return {"lstsq": dd_vec(x), "cond_normal": np.array([kappa])}


# ---------------------------------------------------------------- eigenvalues


def _lap_dims(i):
    fam, _, arg = i.partition("/")
    return int(fam[3]), int(arg)


def eig_points():
    m = M()
    out = []
    for i in m.ids(small=True) + ["lap2d/100", "lap3d/22"]:
        a = m.get(i)
        t = a.tags
        if not (t & {"symmetric", "hermitian"}) or a.shape[0] == 0:
            continue
        closed = i.startswith("lap") and "spd" in t
        if not closed and a.shape[0] > EIG_MAX:
            continue
        out.append({"matrix": i, "kind": "closed" if closed else "arb"})
    for i in m.ids("rect"):
        if min(m.get(i).shape) <= EIG_MAX:
            out.append({"matrix": i, "kind": "svd"})
    for i in NONSYM:  # eigs (SP6)
        out.append({"matrix": i, "kind": "nonsym"})
    return out


NONSYM = ("convdiff2d/10", "random_unsym/50x5", "random_complex/40")


def eig_ref(p):
    from flint import arb

    a = M().get(p["matrix"])
    n = a.shape[0]
    scale = max(float(np.abs(a.dense()).sum(axis=1).max()), 1e-300) if n <= 4000 else 1.0
    if p["kind"] == "closed":
        dim, g = _lap_dims(p["matrix"])
        scale = 4.0 * dim

        def all_vals():
            c = [2 - 2 * (arb.pi() * k / (g + 1)).cos() for k in range(1, g + 1)]
            mids = np.array([float(v.mid()) for v in c])
            grids = np.meshgrid(*([np.arange(g)] * dim), indexing="ij")
            s = sum(mids[gr] for gr in grids).ravel()
            order = np.argsort(s, kind="stable")
            idx = np.stack([gr.ravel() for gr in grids], axis=1)
            pick = list(order[:K]) + list(order[::-1][:K])
            return [sum((c[int(k)] for k in idx[q]), arb(0)) for q in pick]

        vals = _settle_list(all_vals, scale)
        return {"smallest": _pairs(vals[:K]), "largest": _pairs(vals[K:])}
    if p["kind"] == "svd":
        d = a.dense()
        side = d.T @ d if a.shape[1] <= a.shape[0] else d @ d.T
        k = side.shape[0]
        scale = max(float(np.abs(d).sum(axis=1).max()), 1e-300)

        def sv():
            A = _acb(d)
            G = (A.transpose() * A) if a.shape[1] <= a.shape[0] else (A * A.transpose())
            ev = sorted((z.real for z in G.eig(multiple=True)), key=lambda v: -float(v.mid()))
            if len(ev) != k:
                raise NotSettled("eig")
            return [v.sqrt() if v > 0 else arb(0) for v in ev]

        vals = _settle_list(sv, scale)
        return {"svdvals": _pairs(vals)}
    if p["kind"] == "nonsym":
        d = a.dense()

        def every():
            e = _acb(d).eig(multiple=True)
            if len(e) != n:
                raise NotSettled("eig")
            return sorted(e, key=lambda z: (float(z.real.mid()), float(z.imag.mid())))

        return {"eigvals": _cpairs(every, scale)}

    d = a.dense()

    def ev():
        e = _acb(d).eig(multiple=True)
        if len(e) != n:
            raise NotSettled("eig")
        return sorted((z.real for z in e), key=lambda v: float(v.mid()))

    vals = _settle_list(ev, scale)
    return {"smallest": _pairs(vals[:K]), "largest": _pairs(vals[::-1][:K])}


def _settle_list(fn, floor):
    from arbref import precision, settled

    p = arbref.START_PREC
    while p <= arbref.MAX_PREC:
        with precision(p):
            try:
                vs = fn()
            except ValueError, ZeroDivisionError, NotSettled:
                vs = None
            if vs is not None and all(settled(v, floor=floor) for v in vs):
                return [to_dd(v) for v in vs]
        p *= 2
    raise NotSettled(fn)


def _pairs(dds):
    return np.array([h for h, _ in dds]), np.array([lo for _, lo in dds])


# ---------------------------------------------------------------- expm


def expm_points():
    m = M()
    out = []
    for i in m.ids(small=True):
        a = m.get(i)
        if (
            a.shape[0] != a.shape[1]
            or not 0 < a.shape[0] <= EXPM_MAX
            or a.data.dtype == bool
            or a.data.dtype == np.int64
        ):
            continue
        if "coo" in a.tags or "cancel" in a.tags:
            continue
        norm1 = float(np.abs(a.dense()).sum(axis=0).max())
        if norm1 == 0:
            continue
        for target in (1.0, 30.0):
            s = 2.0 ** round(math.log2(target / norm1))
            out.append({"matrix": i, "scale": s})
    return out


def expm_ref(p):
    a = M().get(p["matrix"])
    d = a.dense() * p["scale"]  # a power of two: exact
    n = d.shape[0]

    def ex():
        if np.iscomplexobj(d):
            E = _acb(d).exp()
        else:
            from flint import arb_mat

            E = arb_mat(d.tolist()).exp()
        return [E[i, j] for i in range(n) for j in range(n)]

    import scipy.linalg as sl

    floor = float(np.abs(sl.expm(d)).max())  # the largest entry: the scale of the measure
    if np.iscomplexobj(d):
        return {"expm": _cpairs(ex, floor)}
    return {"expm": _pairs(_settle_list(ex, floor))}


def _cpairs(fn, floor):
    re = _settle_list(lambda: [v.real for v in fn()], floor)
    im = _settle_list(lambda: [v.imag for v in fn()], floor)
    hi = np.array([complex(a, b) for (a, _), (b, _) in zip(re, im)])
    lo = np.array([complex(a, b) for (_, a), (_, b) in zip(re, im)])
    return hi, lo


# ---------------------------------------------------------------- expmv (SP7)

TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)  # linspace(0, 1, 5): exact doubles


def expmv_points():
    """expm's points, each with its right-hand side b = matrices.rhs(a)."""
    m = M()
    return [{**p, "b": m.rhs(m.get(p["matrix"]))} for p in expm_points()]


def expmv_ref(p):
    """exp(t s A) b for t = 1 (method "expmv") and for t in TIMES (method "interval", t by t), each
    entry settled to 2^-110 of max_i (|exp(s A)| |b|)_i over the times (the scale of the measure)."""
    import scipy.linalg as sl

    a = M().get(p["matrix"])
    d = a.dense() * p["scale"]  # a power of two: exact
    b = np.asarray(p["b"])
    n = d.shape[0]
    cplx = np.iscomplexobj(d) or np.iscomplexobj(b)
    floor = max(float((np.abs(sl.expm(t * d)) @ np.abs(b)).max()) for t in TIMES)

    def values(times):
        def f():
            from flint import acb, acb_mat, arb, arb_mat

            out = []
            for t in times:
                if cplx:
                    E = (_acb(d) * acb(t)).exp()
                    y = E * acb_mat([[acb(z.real, z.imag)] for z in b.astype(np.complex128).tolist()])
                else:
                    E = (arb_mat(d.tolist()) * arb(t)).exp()
                    y = E * arb_mat([[v] for v in b.tolist()])
                out += [y[i, 0] for i in range(n)]
            return out

        return f

    if cplx:
        return {"expmv": _cpairs(values((1.0,)), floor), "interval": _cpairs(values(TIMES), floor)}
    return {"expmv": _pairs(_settle_list(values((1.0,)), floor)),
            "interval": _pairs(_settle_list(values(TIMES), floor))}  # fmt: skip


# ---------------------------------------------------------------- driver


FAMILIES = {
    "spmv": (spmv_points, spmv_ref, ("matrix", "digest", "kind", "x", "w")),
    "sums": (sums_points, sums_ref, ("matrix", "digest")),
    "norms": (norms_points, norms_ref, ("matrix", "digest")),
    "solve": (solve_points, solve_ref, ("matrix", "digest", "b")),
    "lstsq": (lstsq_points, lstsq_ref, ("matrix", "digest", "b")),
    "eig": (eig_points, eig_ref, ("matrix", "digest", "kind")),
    "expm": (expm_points, expm_ref, ("matrix", "digest", "scale")),
    "expmv": (expmv_points, expmv_ref, ("matrix", "digest", "scale", "b")),
}
# fields that are per-point extras of a method (attached to the method's entry, not methods)
EXTRAS = {"cond": "solve", "cond_normal": "lstsq"}


def _point(task):
    fam, k, p = task
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SECONDS)
    t0 = time.monotonic()
    try:
        return fam, k, FAMILIES[fam][1](p), time.monotonic() - t0, None
    except Exception as e:  # a point that does not settle: dropped and reported
        return fam, k, None, time.monotonic() - t0, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def write_family(fam, points, results, dropped):
    from tests import reffile

    m = M()
    _, _, ins = FAMILIES[fam]
    for p in points:
        p["digest"] = m.digest(m.get(p["matrix"]))
    inputs = {k: [np.asarray(p[k]) for p in points] for k in ins}
    names = []
    for r in results.values():
        for name in r:
            if name not in names and name not in EXTRAS:
                names.append(name)
    methods = {}
    for name in names:
        idx = [k for k in sorted(results) if name in results[k]]
        e = {"over": "point", "index": np.array(idx, np.int64), "hi": [results[k][name][0] for k in idx],
             "lo": [results[k][name][1] for k in idx], "dropped": np.array(sorted(dropped), np.int64)}  # fmt: skip
        for extra, owner in EXTRAS.items():
            if owner == name:
                e["cond"] = [float(results[k][extra][0]) for k in idx]
        methods[name] = e
    reffile.write(OUT / f"{fam}.npz", **arbref.provenance(extra="exact dyadic rationals; verified solves"), tool=TOOL,
                  n_requested=len(points), inputs=inputs, methods=methods)  # fmt: skip


def build(names, jobs):
    import resource

    start = time.monotonic()
    plans = {f: FAMILIES[f][0]() for f in names}
    tasks = [(f, k, p) for f in names for k, p in enumerate(plans[f])]
    tasks.sort(key=lambda t: -M().get(t[2]["matrix"]).nnz)  # big ones first
    results = {f: {} for f in names}
    dropped = {f: {} for f in names}
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        for fam, k, res, secs, err in pool.imap_unordered(_point, tasks):
            if res is None:
                dropped[fam][k] = err
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]['matrix']}: {secs:.0f} s", flush=True)
    for f in names:
        write_family(f, plans[f], results[f], dropped[f])
        print(f"{f}: {len(results[f])} of {len(plans[f])} points -> {(OUT / f'{f}.npz').relative_to(ROOT)}", flush=True)
        for k, err in sorted(dropped[f].items()):
            print(f"  dropped {plans[f][k]['matrix']}: {err}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"{len(tasks)} points, {(time.monotonic() - start):.0f} s, largest worker RSS {rss:.2f} GB", flush=True)


def main(argv=None):
    from bench import cli

    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their points")
    cli.add_jobs(ap)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    names = a.names or list(FAMILIES)
    if a.list:
        for f in names:
            pts = FAMILIES[f][0]()
            print(f"{f} ({len(pts)}): " + ", ".join(sorted({p["matrix"] for p in pts})))
        return 0
    with arbref.logged("sparse_ref"):
        build(names, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
