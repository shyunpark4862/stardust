"""Test matrices and Arb reference values for the linalg accuracy measurement.

    .venv/bin/python tools/linalg_ref.py               # every family, on 10 processes
    .venv/bin/python tools/linalg_ref.py expm --jobs=4

writes tests/linalg/ref/<family>.npz, read by bench/accuracy (python -m bench.accuracy, area
linalg). The matrices come from seeded generators and are stored, so the file is the sweep.
Group i of a family is the keys "g<i>_<name>": the inputs, and per reference value hi (and lo
where it is a double-double).

    square     A = U diag(s) V^T, n = 3, 10, 50, 200, kappa_2 = 1, 1e4, 1e8, 1e12 (s log-spaced,
               U, V random orthogonal), b normal. x = A^-1 b, det A, and for n <= 50 the
               eigenvalues (complex: re, im)
    symmetric  Q diag(l) Q^T (made exactly symmetric), the same n and kappa, eigenvalues of
               both signs; and the same with positive eigenvalues (spd, for cholesky). The
               eigenvalues
    rect       m x n, (20, 5), (100, 10), (500, 50), kappa = 1, 1e4, 1e8, b normal: the singular
               values and the least-squares solution
    expm       normal matrices scaled to norms 1e-2 .. 50, n = 3, 10, 30, non-normal triangular
               ones, and two classic cases: exp(A) entry by entry

Every reference is Arb on the matrix as stored (the doubles are exact balls): LU solves and
determinants, acb_mat.eig for eigenvalues (singular values: the square roots of the
eigenvalues of A^T A formed exactly), arb_mat.exp. A vector's entries are settled to 2^-110 of
the vector's largest entry (the measures are normwise), eigenvalues and singular values to
2^-110 ||A||, determinants to 2^-110 relative.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np
from flint import acb_mat, arb, arb_mat

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from arbref import NotSettled, precision, settled, to_dd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "linalg" / "ref"
START, MAX_PREC = 128, 1 << 13


def orth(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def amat(a):
    return arb_mat(np.asarray(a, dtype=float).tolist())


def settle(fn):
    """fn() -> {name: (list of arb, floor)} at doubling precision until every ball is settled
    (relative, or below floor * 2^-110); {name: (hi, lo)} as arrays."""
    p = START
    while p <= MAX_PREC:
        with precision(p):
            try:
                out = fn()
            except ValueError, ZeroDivisionError:  # eig could not isolate the eigenvalues here
                out = None
            if out is not None and all(settled(v, floor=f) for vs, f in out.values() for v in vs):
                res = {}
                for k, (vs, _) in out.items():
                    dd = [to_dd(v) for v in vs]
                    res[k] = (np.array([h for h, _ in dd]), np.array([lo for _, lo in dd]))
                return res
        p *= 2
    raise NotSettled(fn)


def _eig(M, n):
    """The eigenvalues of an arb_mat or acb_mat as lists of real and imaginary arb parts."""
    ev = acb_mat(M).eig()
    if len(ev) != n:
        raise ValueError("eig")
    return [e.real for e in ev], [e.imag for e in ev]


# ------------------------------------------------------------------------------ families


def square_groups(rng):
    out = []
    for n in (3, 10, 50, 200):
        for kappa in (1.0, 1e4, 1e8, 1e12):
            s = np.exp(np.linspace(0.0, -math.log(kappa), n)) * 10.0 ** rng.uniform(-2, 2)
            a = (orth(rng, n) * s) @ orth(rng, n).T
            out.append({"a": a, "b": rng.standard_normal(n), "kappa": kappa})
    return out


def square_ref(g):
    a, b = g["a"], g["b"]
    n = len(b)
    A = amat(a)
    nrm = float(np.abs(a).max())
    x = A.solve(amat(b[:, None]))
    xs = [x[i, 0] for i in range(n)]
    xf = max(abs(float(v.mid())) for v in xs)
    out = {"x": (xs, xf), "det": ([A.det()], 0)}
    if n <= 50:
        re, im = _eig(A, n)
        out["eig_re"], out["eig_im"] = (re, nrm), (im, nrm)
    return out


def symmetric_groups(rng):
    out = []
    for spd in (False, True):
        for n in (3, 10, 50, 200):
            for kappa in (1.0, 1e4, 1e8, 1e12):
                lam = np.exp(np.linspace(0.0, -math.log(kappa), n)) * 10.0 ** rng.uniform(-2, 2)
                if not spd:
                    lam = lam * np.where(rng.random(n) < 0.5, -1.0, 1.0)
                q = orth(rng, n)
                a = (q * lam) @ q.T
                a = np.tril(a) + np.tril(a, -1).T
                out.append({"a": a, "spd": spd, "kappa": kappa})
    return out


def symmetric_ref(g):
    a = g["a"]
    n = len(a)
    re, _ = _eig(amat(a), n)
    return {"eigvalsh": (sorted(re, key=lambda v: float(v.mid())), float(np.abs(a).sum(axis=1).max()))}


def rect_groups(rng):
    out = []
    for m, n in ((20, 5), (100, 10), (500, 50)):
        for kappa in (1.0, 1e4, 1e8):
            s = np.exp(np.linspace(0.0, -math.log(kappa), n)) * 10.0 ** rng.uniform(-2, 2)
            u = orth(rng, m)[:, :n]
            a = (u * s) @ orth(rng, n).T
            out.append({"a": a, "b": rng.standard_normal(m), "kappa": kappa})
    return out


def rect_ref(g):
    a, b = g["a"], g["b"]
    n = a.shape[1]
    A = amat(a)
    At = A.transpose()
    G = At * A
    x = G.solve(At * amat(b[:, None]))
    xs = [x[i, 0] for i in range(n)]
    re, _ = _eig(G, n)
    sv = sorted((v.sqrt() if v > 0 else arb(0) for v in re), key=lambda v: -float(v.mid()))
    return {"x": (xs, max(abs(float(v.mid())) for v in xs)), "svdvals": (sv, float(np.abs(a).max()))}


def expm_groups(rng):
    out = []
    for n in (3, 10, 30):
        for norm in (1e-2, 1.0, 10.0, 50.0):
            a = rng.standard_normal((n, n))
            out.append({"a": a * (norm / np.abs(a).sum(axis=1).max())})
        t = np.triu(rng.standard_normal((n, n)) * 10.0, 1) + np.diag(rng.uniform(-2, 0, n))
        out.append({"a": t})  # non-normal
    out.append({"a": np.array([[-49.0, 24.0], [-64.0, 31.0]])})  # Moler and Van Loan
    out.append({"a": np.array([[10.0, 1.0], [0.0, 10.0 + 1e-12]])})  # close eigenvalues
    return out


def expm_ref(g):
    a = g["a"]
    E = amat(a).exp()
    es = [E[i, j] for i in range(len(a)) for j in range(len(a))]
    return {"expm": (es, max(abs(float(v.mid())) for v in es))}


def conds(family, g, ref):
    """The condition numbers the measurement reports beside each forward error (sum over the
    inputs of |d log f / d log a|, or the bound named), from numpy/scipy in double: a few digits
    are all a condition number needs."""
    import scipy.linalg as sl

    a = g["a"]
    with np.errstate(all="ignore"):
        if family == "square":
            ai = np.linalg.inv(a)
            out = {"cond_solve": np.abs(a).sum(axis=1).max() * np.abs(ai).sum(axis=1).max(),  # kappa_inf
                   "cond_det": np.abs(a * ai.T).sum()}  # fmt: skip
            if "eig_re" in ref:  # ||A||_2 / (|lambda| |y^H x|), per reference eigenvalue
                w, vl, vr = sl.eig(a, left=True, right=True)
                vl, vr = vl / np.linalg.norm(vl, axis=0), vr / np.linalg.norm(vr, axis=0)
                lam = ref["eig_re"][0] + 1j * ref["eig_im"][0]
                k = [int(np.argmin(np.abs(w - v))) for v in lam]
                s_ = np.array([abs(np.vdot(vl[:, j], vr[:, j])) for j in k])
                out["cond_eig"] = np.linalg.norm(a, 2) / (np.abs(lam) * s_)
            return out
        if family == "rect":
            sv = np.linalg.svd(a, compute_uv=False)
            return {"cond_lstsq": sv[0] / sv[-1]}  # kappa_2
        if family == "expm":
            return {"cond_expm": max(1.0, np.abs(a).sum(axis=1).max())}  # the relative condition is >= ||A||
    return {}


FAMILIES = {
    "square": (square_groups, square_ref),
    "symmetric": (symmetric_groups, symmetric_ref),
    "rect": (rect_groups, rect_ref),
    "expm": (expm_groups, expm_ref),
}
SEEDS = {name: i + 1 for i, name in enumerate(FAMILIES)}


def _work(args):
    family, g = args
    return settle(lambda: FAMILIES[family][1](g))


def write(family, jobs):
    gen, _ = FAMILIES[family]
    groups = gen(np.random.default_rng(SEEDS[family]))
    order = sorted(range(len(groups)), key=lambda i: -groups[i]["a"].size)  # big ones first
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:
        refs = dict(zip(order, pool.map(_work, [(family, groups[i]) for i in order], chunksize=1)))
    arrays = {}
    for i, g in enumerate(groups):
        for k, v in g.items():
            arrays[f"g{i}_{k}"] = np.asarray(v)
        for k, (hi, lo) in refs[i].items():
            arrays[f"g{i}_{k}_hi"], arrays[f"g{i}_{k}_lo"] = hi, lo
        for k, v in conds(family, g, refs[i]).items():
            arrays[f"g{i}_{k}"] = np.asarray(v)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{family}.npz", **arrays)
    print(f"{family}: {len(groups)} groups -> {(OUT / f'{family}.npz').relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("families", nargs="*", help=", ".join(FAMILIES))
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    bad = set(a.families) - set(FAMILIES)
    if bad:
        ap.error(f"unknown families {sorted(bad)}; choose from {', '.join(FAMILIES)}")
    for f in a.families or FAMILIES:
        write(f, min(a.jobs, 10))


if __name__ == "__main__":
    main()
