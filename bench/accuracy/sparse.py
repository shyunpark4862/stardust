"""sd.sparse's accuracy sweeps: one per reference family of tests/sparse/ref (tools/sparse_ref.py) whose
functions are public. SP2 brings two:

  spmv  A @ x (region matvec) and w @ A (rmatvec) on the points of spmv.npz,
  sums  the row sums (row_sums), the column sums (col_sums) and the whole sum (total) of sums.npz;

one point per output entry, the matrix's id and the entry its description. SP4 adds

  solve  spsolve(A, b) (region solve) on the points of solve.npz: |x_i - ref_i| over
         C_SOLVE kappa_inf eps ||ref||_inf (refdata's normwise contract, entry by entry).

SP5 adds

  iterative  cg, minres (the symmetric and Hermitian points), gmres, bicgstab (all) on solve.npz
         and lsqr, lsmr on lstsq.npz, at tight tolerances (RTOL) and scipy's caps (gmres: n cycles,
         _cap); one region per solver, its
         converged results only (a result past max_iter is counted in `excluded`). The contract
         is what the stopping test promises (docs/design/sparse.md §4): a square solve stops at
         ||b - A x||_2 <= rtol ||b||_2, so |x_i - ref_i| <= ||A^-1||_inf rtol ||b||_2 (||A^-1||_inf
         = kappa_inf / ||A||_inf from the reference). minres' test is on its estimate of the
         residual, ||r|| <= rtol ||A|| ||x||, and the true residual stays above the estimate by up
         to eps kappa(A) ||A|| ||x|| (MINRES' residual gap, Sleijpen, van der Vorst and Modersitzki
         2000: graded/100x8, kappa 1e10, stops with a true relative residual of 7e-2 in scipy too),
         so its bound is ||A^-1||_inf (rtol + eps kappa_inf) ||A||_F ||x||_2. A least-squares one
         stops at ||A^T r||_2 <= atol
         ||A|| ||r||_2, so |x_i - ref_i| <= ||(A^T A)^-1||_inf atol ||A||_F ||r||_2 (the result's
         r). A value over 1 is a converged result whose true residual misses the test (the
         recurrence's residual parted from it). The comparator is scipy 1.18.1 at the same
         tolerances (R's Matrix has no iterative solver).

SP6 adds

  eig    eigsh (the 6 smallest and largest eigenvalues of the symmetric and Hermitian points: 'SA',
         'LA'; shift-invert for the smallest of the large grids and of graded/100x8, where plain
         Lanczos does not converge), svds (the 6 largest and smallest singular values of the
         rectangular points) and eigs (the 6 of each `which` 'LM', 'SR' of the nonsymmetric points)
         on eig.npz: one point per value, |v - ref| over C_EIG eps ||A||_2 (eigs: times the
         eigenvalue's condition number kappa = 1 / |y^H x|, the nearest reference taken), the
         refdata contract. The comparators are scipy 1.18.1 (eigsh / eigs, svds's default solver;
         v0 = sin(1..n)) and R 4.6.2's RSpectra 0.16.2 (eigs_sym, eigs, svds; real matrices;
         the same initvec).

 The contract of the others is
tests/sparse/refdata.py's (docs/design/sparse.md §4), as the tests judge it: |y - ref| in units of
gamma_m sum_j |a_ij x_j| (m = min(len, L) + ceil(log2(len / L)), L = 4096), so a region's
contract is the largest of those over 1 (refdata.spmv_error, the bound 1). The relative error and
the ULP distance from the reference's double go along. bool and int64 matrices are exact and
left out.

The comparators are scipy 1.18.1 (live: scipy.sparse's csr_array @ x, x @ A and sum(axis=)) and R
4.6.2's Matrix 1.7.6 (%*%, crossprod, rowSums, colSums, sum), on the same points and measured the
same way (solve: scipy's spsolve, SuperLU; R's solve, CSparse's LU). The other families (lstsq,
eig, norms, expm) join with the milestones that make their functions public (SP5-SP7).
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region, ulps

FAMILIES = ("spmv", "sums", "solve", "iterative", "eig")
RTOL = 1e-10  # the iterative family's rtol (square) and atol = btol (least squares)


def _points(family):
    from tests.sparse import matrices as M
    from tests.sparse import refdata as R

    for p in R.points(family):
        m = M.get(p["matrix"])
        if m.data.dtype.kind in "bi":
            continue
        yield p, m


def _abs_terms(m, x, by_row):
    """sum_j |a_ij x_j| per output (rows when by_row, else columns) and the term counts."""
    keys = m.row if by_row else m.col
    other = m.col if by_row else m.row
    n = m.shape[0] if by_row else m.shape[1]
    w = np.abs(m.data * (x[other] if x is not None else 1.0)) if m.nnz else np.zeros(0)
    return np.bincount(keys, weights=w, minlength=n), np.bincount(keys, minlength=n)


def _r_value(expr, m, data):
    from . import rsession

    r, c, v = m.canonical()
    setup = {"I": (r + 1).astype(np.float64), "J": (c + 1).astype(np.float64), "V": v, "D": np.array(m.shape, float)}
    head = "{ suppressMessages(library(Matrix)); A <- sparseMatrix(i = I, j = J, x = V, dims = D); "
    return np.asarray(rsession.evaluate_named(head + expr + " }", {**setup, **data}), dtype=np.float64).ravel()


def _iter_methods(m):
    """The solvers of the iterative family for a square matrix of solve.npz."""
    t = m.tags
    out = ["gmres", "bicgstab"]
    if "spd" in t:
        out.insert(0, "cg")
    if t & {"symmetric", "hermitian"}:
        out.insert(0, "minres")
    return out


def _iter_solve(lib, method, m, b):
    """x of method on (m, b) with library lib at the family's tolerances, or None when it stops
    without converging."""
    n = m.shape[1]
    if lib == "sd":
        import stardust as sd
        from tests.sparse.conftest import coo

        f = getattr(sd.sparse, method)
        A = coo(m).tocsr()
        if method in ("lsqr", "lsmr"):
            r = f(A, sd.asarray(b), atol=RTOL, btol=RTOL)
        else:
            r = f(A, sd.asarray(b), rtol=RTOL, max_iter=_cap(method, n))
        return np.asarray(r.x) if r.converged else None
    import scipy.sparse.linalg as spla

    f = getattr(spla, method)
    A = m.scipy("csr")
    if method in ("lsqr", "lsmr"):
        r = f(A, b, atol=RTOL, btol=RTOL)
        return r[0] if r[1] in (0, 1, 2, 4, 5) else None
    x, info = f(A, b, rtol=RTOL, maxiter=_cap(method, n))
    return x if info == 0 else None


def _cap(method, n):
    """The iteration cap: scipy's default (cg, bicgstab 10 n; minres 5 n), gmres n cycles of 20
    (its default 10 n cycles makes the unconverged graded matrices the bulk of the sweep)."""
    return n if method == "gmres" else (5 if method == "minres" else 10) * n


def _iterative_regions(lib):
    from tests.sparse import refdata as R

    rows, excluded = {}, {}
    for family in ("solve", "lstsq"):
        for p, m in _points(family):
            ref = p[f"{family}_hi"] + p[f"{family}_lo"]
            A = m.scipy("csr")
            methods = _iter_methods(m) if family == "solve" else ["lsqr", "lsmr"]
            for method in methods:
                x = _iter_solve(lib, method, m, p["b"])
                if x is None:
                    excluded[method] = excluded.get(method, 0) + 1
                    continue
                fro = float(np.sqrt((abs(A).power(2)).sum()))
                if family == "solve":
                    inv = p["solve_cond"] / float(abs(A).sum(axis=1).max())
                    if method == "minres":  # ||r|| <= (rtol + eps kappa) ||A|| ||x||: its test and gap
                        scale = inv * (RTOL + R.EPS * p["solve_cond"]) * fro * float(np.linalg.norm(x))
                    else:
                        scale = inv * RTOL * float(np.linalg.norm(p["b"]))
                else:
                    G = (A.T @ A).toarray()
                    inv = p["lstsq_cond"] / float(np.abs(G).sum(axis=1).max())
                    scale = inv * RTOL * fro * float(np.linalg.norm(p["b"] - A @ x))
                err = np.abs(x - ref) / scale
                with np.errstate(all="ignore"):
                    rel = np.where(ref == 0, np.abs(x - ref), np.abs(x - ref) / np.abs(ref))
                where = [f"{p['matrix']} {method}[{i}]" for i in range(x.size)]
                rows.setdefault(method, []).append((where, err, rel, ulps(x, p[f"{family}_hi"])))
    out = {}
    for method, parts in rows.items():
        where = np.concatenate([np.asarray(w) for w, *_ in parts])
        err = np.concatenate([e for _, e, _, _ in parts])
        rel = np.concatenate([r for _, _, r, _ in parts])
        u = np.concatenate([np.asarray(v, dtype=float) for *_, v in parts])
        out[method] = region(where, ulp=u, rel=rel, contract=err, bound=1.0, excluded=excluded.get(method, 0))
    return out or None


def _results(family, lib, p, m):
    """{method: values} of library lib on point p (None: not run there)."""
    if lib == "sd":
        import stardust as sd
        from tests.sparse.conftest import coo

        A = coo(m).tocsr()
        if family == "solve":
            return {"solve": np.asarray(sd.sparse.spsolve(A.tocsc(), sd.asarray(p["b"])))}
        if family == "spmv":
            return {"matvec": np.asarray(A @ sd.asarray(p["x"])), "rmatvec": np.asarray(sd.asarray(p["w"]) @ A)}
        return {"row_sums": np.asarray(A.sum(axis=1)), "col_sums": np.asarray(A.sum(axis=0)),
                "total": np.asarray([A.sum()])}  # fmt: skip
    if lib == "scipy":
        s = m.scipy("csr")
        if family == "solve":
            import scipy.sparse.linalg as spla

            return {"solve": spla.spsolve(m.scipy("csc"), p["b"])}
        if family == "spmv":
            return {"matvec": s @ p["x"], "rmatvec": p["w"] @ s}
        return {"row_sums": s.sum(axis=1), "col_sums": s.sum(axis=0), "total": np.asarray([s.sum()])}
    if m.data.dtype.kind == "c":  # R's Matrix has no complex sparse class
        return None
    if family == "solve":
        return {"solve": _r_value("as.numeric(solve(A, b))", m, {"b": p["b"]})}
    if family == "spmv":
        return {"matvec": _r_value("as.numeric(A %*% x)", m, {"x": p["x"]}),
                "rmatvec": _r_value("as.numeric(crossprod(A, w))", m, {"w": p["w"]})}  # fmt: skip
    return {"row_sums": _r_value("rowSums(A)", m, {}), "col_sums": _r_value("colSums(A)", m, {}),
            "total": _r_value("sum(A)", m, {})}  # fmt: skip


def _regions(family, lib):
    from tests.sparse import refdata as R

    if family == "iterative":
        return None if lib == "R" else _iterative_regions(lib)
    if family == "eig":
        return _eig_regions(lib)

    rows = {}
    for p, m in _points(family):
        got = _results(family, lib, p, m)
        if got is None:
            continue
        for method, y in got.items():
            hi, lo = np.atleast_1d(p[f"{method}_hi"]), np.atleast_1d(p[f"{method}_lo"])
            if method == "solve":
                y = np.asarray(y).ravel()
                ref = hi + lo
                scale = R.C_SOLVE * p["solve_cond"] * R.EPS * float(np.abs(ref).max())
                err = np.abs(y - ref) / scale
                with np.errstate(all="ignore"):
                    rel = np.where(ref == 0, np.abs(y - ref), np.abs(y - ref) / np.abs(ref))
                where = [f"{p['matrix']} solve[{i}]" for i in range(y.size)]
                rows.setdefault(method, []).append((where, err, rel, ulps(y, hi)))
                continue
            if method in ("matvec", "row_sums"):
                abs_sum, lengths = _abs_terms(m, p.get("x") if method == "matvec" else None, True)
            elif method in ("rmatvec", "col_sums"):
                abs_sum, lengths = _abs_terms(m, p.get("w") if method == "rmatvec" else None, False)
            else:
                abs_sum, lengths = np.asarray([np.abs(m.data).sum()]), np.asarray([max(m.nnz, 1)])
            y = np.asarray(y).ravel()
            ref = hi + lo
            with np.errstate(all="ignore"):
                err = R.spmv_error(y, hi, lo, abs_sum, lengths)
                rel = np.where(ref == 0, np.abs(y - ref), np.abs(y - ref) / np.abs(ref))
            u = ulps(y, hi)
            where = [f"{p['matrix']} {method}[{i}]" for i in range(y.size)]
            rows.setdefault(method, []).append((where, err, rel, u))
    out = {}
    for method, parts in rows.items():
        where = np.concatenate([np.asarray(w) for w, *_ in parts])
        err = np.concatenate([e for _, e, _, _ in parts])
        rel = np.concatenate([r for _, _, r, _ in parts])
        u = np.concatenate([np.asarray(x, dtype=float) for *_, x in parts])
        out[method] = region(where, ulp=u, rel=rel, contract=err, bound=1.0)
    return out or None


# ---------------------------------------------------------------- eig (SP6)


def _eig_values(lib, kind, m, which):
    """The values of library lib on matrix m: eigsh's / svds's / eigs's k = 6 for `which` (None:
    not run there, or not converged)."""
    n = min(m.shape)
    k = min(6, n - 1)
    big_or_graded = m.shape[0] > 2000 or m.id.startswith("graded")
    sigma = (-1.0 if m.shape[0] > 2000 else 0.0) if kind != "svd" and which == "SA" and big_or_graded else None
    if lib == "sd":
        import stardust as sd
        from tests.sparse.conftest import coo

        A = coo(m).tocsr()
        if kind == "svd":
            r = sd.sparse.svds(A, k=k, which=which)
            return np.asarray(r.S) if r.converged else None
        if kind == "nonsym":
            r = sd.sparse.eigs(A, k=k, which=which)
            return np.asarray(r.eigenvalues) if r.converged else None
        kw = {"sigma": sigma} if sigma is not None else {"which": which}
        r = sd.sparse.eigsh(A, k=k, **kw)
        return np.asarray(r.eigenvalues) if r.converged else None
    if lib == "scipy":
        import scipy.sparse.linalg as spla

        A = m.scipy("csr")
        v0 = np.sin(np.arange(1.0, m.shape[0] + 1))  # (ones lies in the null space of the Neumann grids)
        try:
            if kind == "svd":
                return np.asarray(spla.svds(A, k=k, which=which, rng=0, return_singular_vectors=False))
            if kind == "nonsym":
                return spla.eigs(A, k=k, which=which, v0=v0, return_eigenvectors=False)
            kw = {"sigma": sigma} if sigma is not None else {"which": which}
            return spla.eigsh(A, k=k, v0=v0, return_eigenvectors=False, **kw)
        except spla.ArpackError:  # (ArpackNoConvergence included)
            return None
    if m.data.dtype.kind == "c":  # R's Matrix has no complex sparse class
        return None
    opts = "opts = list(tol = 1e-12, maxitr = 100000, initvec = sin(seq_len(nrow(A))))"
    if kind == "svd":
        if which == "SM":  # RSpectra's svds has no smallest singular values
            return None
        expr = "suppressMessages(library(RSpectra)); svds(A, 6, opts = list(tol = 1e-12, maxitr = 100000))$d"
    elif kind == "nonsym":
        expr = (f'suppressMessages(library(RSpectra)); e <- eigs(A, 6, which = "{which}", {opts})$values; '
                "c(Re(e), Im(e))")  # fmt: skip
        v = _r_value(expr, m, {})
        return v[:6] + 1j * v[6:]
    elif sigma is not None:
        expr = f"suppressMessages(library(RSpectra)); eigs_sym(A, 6, sigma = {sigma!r}, {opts})$values"
    else:
        expr = f'suppressMessages(library(RSpectra)); eigs_sym(A, 6, which = "{which}", {opts})$values'
    try:
        return _r_value(expr, m, {})
    except RuntimeError:
        return None


def _eig_regions(lib):
    from tests.sparse import refdata as R

    rows, excluded = {}, {}
    for p, m in _points("eig"):
        kind = p["kind"]
        norm = 4.0 * int(p["matrix"][3]) if kind == "closed" else float(np.linalg.norm(m.dense(), 2))
        region_name = {"svd": "svds", "nonsym": "eigs"}.get(kind, "eigsh")
        cases = {"svd": ("LM", "SM"), "nonsym": ("LM", "SR")}.get(kind, ("SA", "LA"))
        for which in cases:
            v = _eig_values(lib, kind, m, which)
            if v is None:
                excluded[region_name] = excluded.get(region_name, 0) + 1
                continue
            v = np.asarray(v)
            if kind == "nonsym":
                import scipy.linalg as sl

                ref_all = p["eigvals_hi"] + p["eigvals_lo"]
                lam, Y, X = sl.eig(m.dense(), left=True, right=True)
                idx = [int(np.argmin(np.abs(ref_all - z))) for z in v]
                ref = ref_all[idx]
                hi = p["eigvals_hi"][idx]
                kappa = []
                for z in v:
                    j = int(np.argmin(np.abs(lam - z)))
                    y, x = Y[:, j], X[:, j]
                    kappa.append(np.linalg.norm(y) * np.linalg.norm(x) / abs(np.vdot(y, x)))
                scale = R.C_EIG * R.EPS * norm * np.asarray(kappa)
            else:
                if kind == "svd":
                    allv = p["svdvals_hi"] + p["svdvals_lo"]
                    allhi = p["svdvals_hi"]
                    sel = slice(0, 6) if which == "LM" else slice(len(allv) - 6, len(allv))
                    ref, hi = np.sort(allv[sel]), np.sort(allhi[sel])
                    v = np.sort(v)
                else:
                    key = "smallest" if which == "SA" else "largest"
                    k = v.size
                    ref, hi = np.sort((p[f"{key}_hi"] + p[f"{key}_lo"])[:k]), np.sort(p[f"{key}_hi"][:k])
                    v = np.sort(np.real(v))
                scale = R.C_EIG * R.EPS * norm
            err = np.abs(v - ref) / scale
            with np.errstate(all="ignore"):
                rel = np.where(ref == 0, np.abs(v - ref), np.abs(v - ref) / np.abs(ref))
            where = [f"{p['matrix']} {region_name} {which}[{i}]" for i in range(v.size)]
            rows.setdefault(region_name, []).append((where, err, rel, ulps(v, hi)))
    out = {}
    for name, parts in rows.items():
        where = np.concatenate([np.asarray(w) for w, *_ in parts])
        err = np.concatenate([e for _, e, _, _ in parts])
        rel = np.concatenate([r for _, _, r, _ in parts])
        u = np.concatenate([np.asarray(x, dtype=float) for *_, x in parts])
        out[name] = region(where, ulp=u, rel=rel, contract=err, bound=1.0, excluded=excluded.get(name, 0))
    return out or None


def sweeps():
    out = []
    for fam in FAMILIES:

        def run(fam=fam):
            return _regions(fam, "sd")

        def measure(lib, fam=fam):
            return _regions(fam, lib)

        # (scipy's iterative solvers take about 40 s over the family's points)
        seconds = 300.0 if fam in ("iterative", "eig") else 60.0
        out.append(Custom("sparse", fam, run, compare_libs(measure, ("scipy", "R"), seconds=seconds)))
    return out
