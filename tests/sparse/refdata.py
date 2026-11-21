"""sd.sparse's references and their contract (docs/design/sparse.md §4, §7), as the tests and
the accuracy measurement (bench/accuracy) both judge them.

tests/sparse/ref/<family>.npz (tools/sparse_ref.py; layout: tests/reffile.py) holds one point per
(matrix, case): the inputs "matrix" (the id in tests/sparse/matrices.py) and "digest" (that
matrix's matrices.digest when the file was made), the family's own inputs, and per method the
reference hi + lo (a double-double; complex for complex results):

    spmv    matvec (A x), rmatvec (A^T w); inputs kind ('ones' | 'random'), x, w
    sums    row_sums, col_sums, total (of the stored values)
    norms   1, -1, inf, -inf, fro, 2 (2: n <= 220)
    solve   solve (x = A^-1 b; field cond: kappa_inf); input b
    lstsq   lstsq (the least-squares solution; field cond: kappa_inf of A^T A); input b
    eig     smallest, largest (6 each, ascending / descending), svdvals (descending), eigvals (every
            eigenvalue of a nonsymmetric matrix, complex, by real then imaginary part); input kind
    expm    expm (exp(scale A), row-major); input scale
    expmv   expmv (exp(scale A) b), interval (exp(t scale A) b for t in TIMES, one after another);
            inputs scale, b (expm's points, in the same order)

tests/sparse/data holds the other libraries' records (tools/gen_sparse_scipy.py: scipy.json;
tools/gen_sparse_r.py: r.json), each with its own "_provenance".

The contract (draft, docs/design/sparse.md §4), in units of its bound (1 or less is within it):
  - sums of products (spmv, sums): |y - ref| / (gamma_m sum_j |a_ij x_j|), m = min(len, L) +
    ceil(log2(len / L)), L = 4096 (the summation chunk), per output entry: sum_abs() gives the
    sum of magnitudes;
  - solve: the normwise backward error ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf) over
    C_BACKWARD eps (backward_error) is the contract; its forward consequence, the normwise
    relative error ||x - ref||_inf / ||ref||_inf over C_SOLVE kappa eps, is checked against the
    references. Both fixed by SP4's measurements (docs/design/sparse.md §4: at most 0.50 eps and
    0.39 kappa eps over the test matrices, after the one refinement step); lstsq: the forward form;
  - eigenvalues, singular values, the 2-norm: |v - ref| / (C_EIG eps ||A||);
  - the other norms (sums of magnitudes, SP7): |v - ref| / ((gamma_m + 2 eps) ref), m the terms of
    the longest line (sum_terms; Frobenius: of every stored entry), 2 eps for the magnitude of a
    complex entry and the last rounding (norm_error);
  - expm: max |E - ref| / (C_EXPM eps max(1, ||s A||_1) max |ref|); expm_multiply the same form,
    max |y - ref| / (C_EXPM eps max(1, ||t s A||_1) max_i (|exp(t s A)| |b|)_i) (expm_error).
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
EPS = float(np.finfo(np.float64).eps)
L = 4096  # the summation chunk of docs/design/sparse.md §2.6
FAMILIES = ("spmv", "sums", "norms", "solve", "lstsq", "eig", "expm", "expmv")
TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)  # expmv's interval: linspace(0, 1, 5)
COMPLEX_METHODS = ("eigvals",)  # complex for a real matrix too
# the contract's constants (drafts until the milestones that implement the functions measure them)
C_SOLVE = 4.0
C_BACKWARD = 4.0
C_EIG = 1000.0  # SP6: ARPACK's eigenvalues after many restarts, up to 497 (lap1d/1000 'LA'; scipy 217)
# SP7: expm and expm_multiply in units of eps max(1, ||s A||_1) scale, measured over expm's and
# expmv's points: sd's expm up to 2.4 (scipy 4.8, the dense sd.linalg.expm 3.5), expm_multiply up
# to 2.0 (2.5 on the interval; scipy the same)
C_EXPM = 10.0


def names() -> list[str]:
    return [f for f in FAMILIES if (REF / f"{f}.npz").exists()]


def load(family) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def provenance(family) -> dict:
    return reffile.provenance(REF / f"{family}.npz")


def points(family):
    """The family's points as dicts: the inputs and per method "<method>_hi", "_lo" (and
    "<method>_cond") where the point has that reference."""
    from . import matrices

    r = load(family)
    out = []
    for i in range(r.points):
        g = {k: v[i] for k, v in r.inputs.items()}
        g["matrix"], g["digest"] = str(g["matrix"]), str(g["digest"])
        # ragged arrays of a family share one dtype in the file: a real matrix's vectors come back
        # complex with zero imaginary parts
        real = matrices.get(g["matrix"]).data.dtype != np.complex128
        for k in ("x", "w", "b"):
            if k in g and real:
                g[k] = np.real(g[k]).astype(np.float64)
        for m, mm in r.methods.items():
            hit = np.flatnonzero(mm.index == i)
            if hit.size:
                j = int(hit[0])
                hi, lo = mm.hi[j], mm.lo[j]
                if real and np.iscomplexobj(hi) and m not in COMPLEX_METHODS:
                    hi, lo = np.real(hi).astype(np.float64), np.real(lo).astype(np.float64)
                g[f"{m}_hi"], g[f"{m}_lo"] = hi, lo
                if "cond" in mm.fields:
                    g[f"{m}_cond"] = float(mm.fields["cond"][j])
        out.append(g)
    return out


def data(name) -> dict:
    """{case: record} of another library's file in tests/sparse/data ("_provenance" left out)."""
    d = json.loads((DATA / f"{name}.json").read_text())
    d.pop("_provenance", None)
    return d


def data_provenance(name) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())["_provenance"]


# ------------------------------------------------------------------ the contract's scales


def gamma(m: int) -> float:
    return m * EPS / 2 / (1 - m * EPS / 2)


def sum_terms(length: int) -> int:
    """m of the summation bound for a sum of `length` terms: sequential up to L, then chunks of L
    joined by a pairwise tree."""
    if length <= L:
        return max(length, 1)
    return L + math.ceil(math.log2(length / L))


def spmv_error(y, ref_hi, ref_lo, abs_sum, lengths) -> np.ndarray:
    """|y - ref| per entry in units of gamma_m sum_j |a_ij x_j| (0 where both are 0)."""
    y = np.asarray(y)
    ref = np.asarray(ref_hi, dtype=y.dtype) + np.asarray(ref_lo, dtype=y.dtype)
    bound = np.array([gamma(sum_terms(int(k))) for k in lengths]) * np.asarray(abs_sum, dtype=float)
    err = np.abs(y - ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(err == 0, 0.0, err / bound)
    return out


def backward_error(A, x, b) -> float:
    """||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf) in units of eps, A dense or scipy sparse
    (the residual in float64: its own rounding is a small part of eps for these sizes; 0 when
    both norms vanish)."""
    x, b = np.asarray(x), np.asarray(b)
    r = b - A @ x
    anorm = float(abs(A).sum(axis=1).max()) if A.shape[0] else 0.0
    scale = anorm * float(np.abs(x).max(initial=0)) + float(np.abs(b).max(initial=0))
    return 0.0 if scale == 0 else float(np.abs(r).max()) / scale / EPS


def normwise_error(x, ref_hi) -> float:
    x, ref = np.asarray(x), np.asarray(ref_hi)
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    if scale == 0:
        return 0.0 if not np.abs(x).max(initial=0) else math.inf
    return float(np.abs(x - ref).max()) / scale


def norm_error(v, ref, terms) -> float:
    """|v - ref| of a norm made of sums of magnitudes, in units of (gamma_m + 2 eps) ref, m =
    sum_terms(terms) (0 when both are 0)."""
    err = abs(v - ref)
    if err == 0:
        return 0.0
    return err / ((gamma(sum_terms(int(terms))) + 2 * EPS) * abs(ref))


def expm_error(y, ref, norm1, scale) -> float:
    """max |y - ref| in units of C_EXPM eps max(1, norm1) scale (expm: scale max |ref|; expm_multiply:
    max_i (|exp(t s A)| |b|)_i)."""
    y, ref = np.asarray(y), np.asarray(ref)
    err = float(np.abs(y - ref).max()) if y.size else 0.0
    if err == 0:
        return 0.0
    return err / (C_EXPM * EPS * max(1.0, float(norm1)) * float(scale))
