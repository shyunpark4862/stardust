"""The linalg area: residuals (the contract) and forward errors against Arb (recorded).

The matrices are tests/linalg/ref/<family>.npz (tools/linalg_ref.py): square, symmetric (and
positive definite), rectangular and expm cases at condition numbers 1 to 1e12, read through
tests/linalg/refdata.py, whose residual() is the contract (docs/design/linalg.md §13,
RESIDUAL_C of tests/linalg/conftest.py) the tests judge with too: every residual in units of
n eps times its scale is at most c = 10, n = max(m, n), in the infinity norm.
Recorded only (no documented bound; the baseline judges): the error against the Arb value,
relative, and relative over u times the problem's condition number (sum over the inputs of
|d log f / d log a|: kappa_inf(A) for solve and lstsq's x -- for lstsq the 2-norm kappa(A)^2
bound is not used --, sum |a_ij (A^-1)_ji| for det, ||A|| / |lambda| for symmetric eigenvalues
and singular values, ||A|| ||x_l|| ||x_r|| / (|lambda| |x_l^H x_r|) for eigenvalues, ||A|| for
expm (its relative condition number is at least ||A||)).
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region

EPS = float(np.finfo(np.float64).eps)
U = EPS / 2


def _groups(family):
    from tests.linalg import refdata

    return refdata.matrices(family)


def _label(g):
    a = g["a"]
    k = float(g["kappa"]) if "kappa" in g else None
    shape = f"{a.shape[0]}x{a.shape[1]}"
    return shape if k is None else f"{shape} kappa={k:g}"


def _by_kappa(rows):
    """rows: [(kappa, where, {measure: value})] -> regions 'kappa=..' with the per-row measures."""
    from tests.linalg import refdata

    out = {}
    for k in sorted({r[0] for r in rows}):
        sel = [r for r in rows if r[0] == k]
        where = [r[1] for r in sel]
        ms = {m: np.array([r[2][m] for r in sel]) for m in sel[0][2]}
        contract = ms.pop("contract", None)
        out[f"kappa={k:g}"] = region(
            where, contract=contract, bound=refdata.RESIDUAL_C if contract is not None else None, **ms
        )
    return out


def _resid(fn_name, lib="sd"):
    """Residual sweeps: the contract measure per matrix, in units of n eps (the column) and over c."""
    from tests.linalg import refdata

    from .linalg_ops import ops

    O = ops(lib)
    rows = []
    for g in refdata.residual_matrices(fn_name):
        res = refdata.call(O, fn_name, g["a"], g.get("b"))
        if res is None:
            return None
        backward, orth = refdata.residual(fn_name, g["a"], g.get("b"), res)
        ms = {"backward": backward}
        if orth is not None:
            ms["orth"] = orth
        ms["contract"] = max(backward, orth or 0.0)
        rows.append((float(g.get("kappa", 0.0)), _label(g), ms))
    return _by_kappa(rows)


def _rel_dd(got, hi, lo):
    from tests.special import refdata as R

    with np.errstate(all="ignore"):
        return R.rel_error(np.atleast_1d(np.asarray(got, dtype=float)), np.atleast_1d(hi), np.atleast_1d(lo))


def _norm_rel(got, hi, lo):
    got = np.asarray(got, dtype=float).ravel()
    return float(np.abs((got - hi) - lo).max() / np.abs(hi).max())


def _forward(fn_name, lib="sd"):
    """Forward errors against Arb, recorded only."""
    from .linalg_ops import ops

    O = ops(lib)
    rows = []
    fam = {"eigvalsh": "symmetric", "svdvals": "rect", "lstsq_x": "rect", "expm": "expm"}.get(fn_name, "square")
    for g in _groups(fam):
        a = g["a"]
        with np.errstate(all="ignore"):
            if fn_name in ("solve_x", "lstsq_x"):
                b = g["b"]
                x = O.solve(a, b) if fn_name == "solve_x" else O.lstsq(a, b)
                m = "solve" if fn_name == "solve_x" else "lstsq"
                rel = _norm_rel(x, g[f"{m}_hi"], g[f"{m}_lo"])
                cond = float(g[f"{m}_cond"])
                ms = {"norm_inf_rel": rel, "rel_cond": rel / (U * cond)}
            elif fn_name == "det":
                rel = float(_rel_dd(O.det(a), g["det_hi"], g["det_lo"])[0])
                cond = float(g["det_cond"])
                ms = {"rel": rel, "rel_cond": rel / (U * cond)}
            elif fn_name in ("eigvalsh", "svdvals"):
                key = fn_name
                got = np.asarray(O.eigvalsh(a) if fn_name == "eigvalsh" else O.svdvals(a))
                if fn_name == "eigvalsh":
                    hi, lo = g[f"{key}_hi"], g[f"{key}_lo"]  # ascending, as eigvalsh
                else:
                    hi, lo = g[f"{key}_hi"], g[f"{key}_lo"]  # descending, as svdvals
                rel = _rel_dd(got, hi, lo)
                n2 = float(np.abs(hi).max())  # ||A||_2
                rc = np.abs((got - hi) - lo) / (U * n2)
                ms = {"rel": float(rel.max()), "rel_cond": float(rc.max())}
            elif fn_name == "eigvals":
                if "eigvals_hi" not in g:
                    continue
                got = np.asarray(O.eigvals(a))
                ref = g["eigvals_hi"]
                used = np.zeros(len(ref), bool)
                err, cond = np.empty(len(got)), np.empty(len(got))
                for i, z in enumerate(got):  # match each eigenvalue to the nearest reference
                    d = np.where(used, np.inf, np.abs(ref - z))
                    j = int(np.argmin(d))
                    used[j] = True
                    err[i] = d[j] / abs(ref[j]) if ref[j] != 0 else d[j]
                    cond[i] = g["eigvals_cond"][j]
                ms = {"rel": float(err.max()), "rel_cond": float(np.max(err / (U * cond)))}
            elif fn_name == "expm":
                e = O.expm(a)
                if e is None:
                    return None
                rel = _norm_rel(e, g["expm_hi"], g["expm_lo"])
                ms = {"norm_inf_rel": rel, "rel_cond": rel / (U * float(g["expm_cond"]))}
            else:
                raise ValueError(fn_name)
        rows.append((float(g.get("kappa", 0.0)), _label(g), ms))
    if fn_name == "expm":  # no kappa: one region
        where = [r[1] for r in rows]
        return {"cases": region(where, **{m: np.array([r[2][m] for r in rows]) for m in rows[0][2]})}
    return _by_kappa(rows)


RESIDUAL = ("solve", "lu_solve", "inv", "lstsq", "lu", "qr", "qr_pivoted", "cholesky", "svd", "eigh", "eig")
FORWARD = ("solve_x", "lstsq_x", "det", "eigvalsh", "svdvals", "eigvals", "expm")


def sweeps():
    out = []
    for name in RESIDUAL:
        fn = lambda lib="sd", name=name: _resid(name, lib)  # noqa: E731
        out.append(Custom("linalg", name, fn, compare_libs(fn, ("numpy", "R"), seconds=120)))
    for name in FORWARD:
        fn = lambda lib="sd", name=name: _forward(name, lib)  # noqa: E731
        out.append(Custom("linalg", name, fn, compare_libs(fn, ("numpy", "R"), seconds=120)))
    return out
