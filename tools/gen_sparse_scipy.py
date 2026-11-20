"""scipy 1.18.1's scipy.sparse and scipy.sparse.linalg on the test matrices of
tests/sparse/matrices.py: the comparison record of sd.sparse (docs/design/sparse.md §7).

    .venv/bin/python tools/gen_sparse_scipy.py              # writes tests/sparse/data/scipy.json
    .venv/bin/python tools/gen_sparse_scipy.py --jobs 4

The record is what API.md §4's rule 2 holds sd.sparse to (the non-rounding behavior) and what
the later milestones compare with:

    ops/<matrix>/<op>       the result of an operation: its type (sparse format, dense, scalar,
                            exception), dtype, shape, nnz and the stored zeros / NaN it keeps
                            (A - A, A * 0, A / 0, A + 1, comparisons, A.multiply(dense with NaN),
                            reductions with and without axis, argmax ties, NaN in max / min / sum)
    dups/<case>             how scipy sums duplicate entries: tocsr, tocsc and
                            coo.sum_duplicates against the sum in input order, left to right
    spmv/<matrix>/<kind>    A @ x, A.T @ w (CSR, CSC, COO) and A @ X (k = 3) on the x, w of
                            tests/sparse/ref/spmv.npz's points: SHA-256 of the result bytes
                            (and the values themselves for n <= 12)
    construct/<function>    the default formats and dtypes of the construction functions
    direct/<matrix>/<fn>    spsolve, splu, spsolve_triangular, inv, factorized: the result
                            (SHA-256 and the normwise residual), warnings and exceptions on
                            singular input
    iterative/<fn>/<matrix> cg, gmres, minres, bicgstab, lsqr, lsmr at scipy's defaults and at a
                            tight tolerance: info / istop, the iteration count, the residual,
                            SHA-256 of x
    eigen/<fn>/<matrix>     eigsh, eigs, svds with v0 = ones (fixed) and rng = default_rng(0)
    matfuncs/<fn>/<matrix>  expm, expm_multiply, onenormest (numpy's global seed 0), norm

Every record of a matrix carries that matrix's digest (tests/sparse/matrices.digest). Large
vectors are kept as hashes; the tests recompute what they compare. "_provenance" names the tool
and the versions.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
DATA = ROOT / "tests" / "sparse" / "data"
TOOL = "tools/gen_sparse_scipy.py"


def M():
    from tests.sparse import matrices

    return matrices


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def plain(v):
    """A JSON value: floats as floats (NaN, inf as strings), complex as [re, im]."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else repr(f)
    if isinstance(v, (np.complexfloating, complex)):
        return [plain(v.real), plain(v.imag)]
    if isinstance(v, np.ndarray):
        return [plain(x) for x in v.tolist()] if v.ndim else plain(v[()])
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v


def describe(r):
    """The kind of a result: sparse format, dense array, scalar; dtype, shape, nnz, stored zeros
    and NaN."""
    import scipy.sparse as sp

    if sp.issparse(r):
        d = r.data if r.format != "dok" else np.array(list(r.values()))
        return {"type": type(r).__name__, "format": r.format, "dtype": str(r.dtype), "shape": list(r.shape),
                "nnz": int(r.nnz), "stored_zeros": int(np.count_nonzero(d == 0)),
                "stored_nan": int(np.count_nonzero(np.isnan(d))) if d.dtype.kind in "fc" else 0}  # fmt: skip
    if isinstance(r, np.ndarray):
        return {"type": "ndarray", "dtype": str(r.dtype), "shape": list(r.shape)}
    return {"type": type(r).__name__, "value": plain(r)}


def attempt(fn):
    """{result description} or {"error": "Type: message"}; warnings by category name."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = describe(fn())
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    if caught:
        out["warnings"] = sorted({w.category.__name__ for w in caught})
    return out


# ---------------------------------------------------------------- groups


OPS_MATRICES = (
    "random_unsym/50x5",
    "coo/dups",
    "coo/int",
    "coo/bool",
    "random_complex/40",
    "coo/nan_inf",
    "empty/rows",
)


def ops():
    out = []
    for mid in OPS_MATRICES:
        a = M().get(mid)
        A = a.scipy("csr")
        B = A.T.tocsr() if A.shape[0] == A.shape[1] else A
        dense_nan = np.full(A.shape, np.nan)
        cases = {
            "A+B": lambda: A + B, "A-A": lambda: A - A, "A*B": lambda: A * B, "A*0": lambda: A * 0,
            "A*2": lambda: A * 2, "A/2": lambda: A / 2, "A/0": lambda: A / 0, "-A": lambda: -A,
            "abs(A)": lambda: abs(A), "A+1": lambda: A + 1, "A+0": lambda: A + 0, "A**0": lambda: A**0,
            "A**2": lambda: A**2, "A@A.T": lambda: A @ A.T, "A==A": lambda: A == A, "A!=A": lambda: A != A,
            "A==0": lambda: A == 0, "A<0": lambda: A < 0, "A>0": lambda: A > 0, "A<=1": lambda: A <= 1,
            "A.multiply(nan)": lambda: A.multiply(dense_nan), "A*nan": lambda: A * np.nan,
            "A*inf": lambda: A * np.inf, "A/B": lambda: A / B, "A+dense": lambda: A + np.ones(A.shape),
            "A.multiply(dense)": lambda: A.multiply(np.ones(A.shape)),
            "A@x": lambda: A @ np.ones(A.shape[1]), "x@A": lambda: np.ones(A.shape[0]) @ A,
            "A.T": lambda: A.T, "A.tocsc().T": lambda: A.tocsc().T, "A.sum()": lambda: A.sum(),
            "A.sum(axis=0)": lambda: A.sum(axis=0), "A.sum(axis=1)": lambda: A.sum(axis=1),
            "A.mean()": lambda: A.mean(), "A.mean(axis=0)": lambda: A.mean(axis=0),
            "A.max()": lambda: A.max(), "A.min()": lambda: A.min(), "A.max(axis=0)": lambda: A.max(axis=0),
            "A.min(axis=1)": lambda: A.min(axis=1), "A.nanmax()": lambda: A.nanmax(),
            "A.argmax()": lambda: A.argmax(), "A.argmin()": lambda: A.argmin(),
            "A.argmax(axis=0)": lambda: A.argmax(axis=0), "A.count_nonzero()": lambda: A.count_nonzero(),
            "A.diagonal()": lambda: A.diagonal(), "A.trace()": lambda: A.trace(), "A[1]": lambda: A[1],
            "A[1, 2]": lambda: A[1, 2], "A[:2]": lambda: A[:2], "A.nonzero()": lambda: np.array(A.nonzero()),
            "A.astype(int64)": lambda: A.astype(np.int64) if A.dtype.kind != "c" else A.real,
            "A.sin()": lambda: A.sin() if A.dtype.kind != "b" else A, "A.power(0.5)": lambda: abs(A).power(0.5),
            "A.eliminate_zeros()": lambda: (lambda c: (c.eliminate_zeros(), c)[1])(A.copy()),
        }  # fmt: skip
        dg = M().digest(a)
        for name, fn in cases.items():
            rec = {"digest": dg, **attempt(fn)}
            if name in ("A.sum()", "A.max()", "A.min()", "A.argmax()", "A.argmin()", "A.nanmax()", "A.trace()"):
                pass  # the value is in the description
            out.append((f"ops/{mid}/{name}", rec))
    # argmax ties and NaN positions on small fixed matrices
    import scipy.sparse as sp

    small = {
        "ties": np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0]]),
        "nan": np.array([[0.0, np.nan, 1.0], [np.nan, 0.0, -1.0]]),
        "all_negative_full": np.array([[-1.0, -2.0], [-3.0, -4.0]]),
        "implicit_zero_max": np.array([[-1.0, 0.0], [0.0, -4.0]]),
    }
    for key, d in small.items():
        A = sp.csr_array(d)
        for name, fn in {"max": A.max, "min": A.min, "argmax": A.argmax, "argmin": A.argmin, "nanmax": A.nanmax,
                         "sum": A.sum, "max(axis=0)": lambda: A.max(axis=0).toarray(),
                         "argmax(axis=1)": lambda: A.argmax(axis=1)}.items():  # fmt: skip
            rec = attempt(fn)
            if rec.get("type") == "ndarray":
                rec["value"] = plain(np.asarray(fn()))
            out.append((f"ops/small:{key}/{name}", rec))
    return out


def _input_order_sum(shape, r, c, v, by_row=True):
    a = M().Matrix("x", shape, np.asarray(r, np.int64), np.asarray(c, np.int64), np.asarray(v), frozenset())
    return a.dense()


def dups():
    import scipy.sparse as sp

    out = []
    for mid in ("coo/cancel", "coo/dups", "coo/int", "coo/complex"):
        a = M().get(mid)
        want = a.dense()
        rec = {"digest": M().digest(a)}
        A = sp.coo_array((a.data, (a.row, a.col)), shape=a.shape)
        rec["tocsr_equals_input_order"] = A.tocsr().toarray().tobytes() == want.tobytes()
        rec["tocsc_equals_input_order"] = A.tocsc().toarray().tobytes() == want.tobytes()
        S = sp.coo_array((a.data, (a.row, a.col)), shape=a.shape)
        S.sum_duplicates()
        rec["coo_sum_duplicates_equals_input_order"] = S.toarray().tobytes() == want.tobytes()
        if mid == "coo/cancel":
            rec["value_at_0_1"] = plain(A.tocsr()[0, 1])
        out.append((f"dups/{mid}", rec))
    # rows with many duplicates: where the three paths agree with the input-order sum, and with
    # scipy's own rule for coo.sum_duplicates (first + np.add.reduce(rest), numpy's pairwise sum)
    g = np.random.default_rng(7)
    vals = np.array([1.0, 1e16, -1e16, 3.0, 1e-3, 2.0**60, -(2.0**60), 0.1])
    stats = {"short": [0, 0, 0, 0, 0], "long": [0, 0, 0, 0, 0]}
    for _ in range(400):
        long = bool(g.integers(0, 2))
        per = int(g.integers(17, 60)) if long else int(g.integers(2, 16))
        r = np.zeros(per, np.int64)  # one row: every entry in row 0, columns 0..2
        c = g.integers(0, 3, per)
        v = g.choice(vals, per) * g.choice([1.0, -1.0], per)
        want = _input_order_sum((1, 3), r, c, v)
        key = "long" if long else "short"
        s = stats[key]
        s[0] += 1
        s[1] += sp.coo_array((v, (r, c)), shape=(1, 3)).tocsr().toarray().tobytes() == want.tobytes()
        s[2] += sp.coo_array((v, (r, c)), shape=(1, 3)).tocsc().toarray().tobytes() == want.tobytes()
        S = sp.coo_array((v, (r, c)), shape=(1, 3))
        S.sum_duplicates()
        s[3] += S.toarray().tobytes() == want.tobytes()
        rule = np.zeros((1, 3))
        for j in range(3):
            sel = v[c == j]
            if sel.size:
                rule[0, j] = sel[0] + np.add.reduce(sel[1:]) if sel.size > 1 else sel[0]
        s[4] += S.toarray().tobytes() == rule.tobytes()
    for key, s in stats.items():
        out.append((f"dups/random_one_row/{key}", {"rows": s[0], "row_entries": "17-59" if key == "long" else "2-15",
                    "tocsr_equals_input_order": s[1], "tocsc_equals_input_order": s[2],
                    "coo_sum_duplicates_equals_input_order": s[3],
                    "coo_sum_duplicates_equals_first_plus_reduce_rest": s[4]}))  # fmt: skip
    return out


def spmv():
    import scipy.sparse as sp

    from tests.sparse import refdata

    out = []
    for p in refdata.points("spmv"):
        a = M().get(p["matrix"])
        if a.data.dtype == bool:
            continue
        rec = {"digest": M().digest(a)}
        x, w = np.asarray(p["x"]), np.asarray(p["w"])
        for fmt in ("csr", "csc", "coo"):
            A = a.scipy(fmt)
            if fmt == "coo":  # the canonical coordinates
                r, c, v = a.canonical()
                A = sp.coo_array((v, (r, c)), shape=a.shape)
            with np.errstate(all="ignore"):
                y, z = A @ x, A.T @ w
            rec[f"{fmt}_matvec"] = sha(y)
            rec[f"{fmt}_rmatvec"] = sha(z)
            if fmt == "csr":
                X = np.stack([x, 2 * x, x[::-1]], axis=1) if x.size else np.zeros((0, 3), x.dtype)
                rec["csr_matmat3"] = sha(np.ascontiguousarray(A @ X))
                if y.size <= 12:
                    rec["matvec"] = plain(y)
        out.append((f"spmv/{p['matrix']}/{p['kind']}", rec))
    return out


def construct():
    import scipy.sparse as sp

    A = M().get("random_unsym/50x5").scipy("csr")
    cases = {
        "eye_array": lambda: sp.eye_array(4), "eye_array(k=1)": lambda: sp.eye_array(4, 5, k=1),
        "diags_array": lambda: sp.diags_array([1.0, 2.0], offsets=[0, 1], shape=(4, 4)),
        "diags_array(int)": lambda: sp.diags_array([1, 2], offsets=[0, 1], shape=(4, 4)),
        "kron": lambda: sp.kron(sp.eye_array(2), A), "kron(csr,csr)": lambda: sp.kron(A, A[:3, :3]),
        "kronsum": lambda: sp.kronsum(A[:3, :3], A[:2, :2]), "hstack": lambda: sp.hstack([A, A]),
        "vstack": lambda: sp.vstack([A, A]), "vstack(csc)": lambda: sp.vstack([A.tocsc(), A.tocsc()]),
        "block_array": lambda: sp.block_array([[A, None], [None, A]]),
        "block_diag": lambda: sp.block_diag([A, A]), "tril": lambda: sp.tril(A), "triu": lambda: sp.triu(A.tocsc()),
        "random_array": lambda: sp.random_array((5, 5), density=0.4, rng=np.random.default_rng(0)),
        "find": lambda: np.array(sp.find(A)[0]), "coo_array(dense)": lambda: sp.coo_array(np.eye(3)),
        "csr_array(dense with nan)": lambda: sp.csr_array(np.array([[0.0, np.nan], [0.0, 1.0]])),
        "csr_array(float32)": lambda: sp.csr_array(np.eye(2, dtype=np.float32)),
        "dok_array": lambda: sp.dok_array((3, 3)),
    }  # fmt: skip
    return [(f"construct/{k}", attempt(f)) for k, f in cases.items()]


def _resid(A, x, b):
    r = b - A @ x
    return float(np.abs(r).max() / (np.abs(A).sum(axis=1).max() * np.abs(x).max() + np.abs(b).max()))


def direct():
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    out = []
    for mid in ("lap2d/30", "convdiff2d/40", "random_complex/800", "saddle/600x150", "graded/400x12", "blockdiag/200x8",
                "struct_singular/40", "dupcol/40", "lap2d/10n", "near_singular/20"):  # fmt: skip
        a = M().get(mid)
        A = a.scipy("csc")
        b = M().rhs(a)
        dg = M().digest(a)

        def solved(fn):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    x = fn()
                    rec = {"sha": sha(x), "finite": bool(np.all(np.isfinite(x))), "max_abs_x": plain(np.abs(x).max())}
                    if rec["finite"]:
                        rec["backward"] = _resid(A, x, b)
                except Exception as e:
                    rec = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if caught:
                rec["warnings"] = sorted({w.category.__name__ for w in caught})
            return rec

        out.append((f"direct/{mid}/spsolve", {"digest": dg, **solved(lambda: spla.spsolve(A, b))}))
        out.append((f"direct/{mid}/splu", {"digest": dg, **solved(lambda: spla.splu(A).solve(b))}))
        out.append((f"direct/{mid}/factorized", {"digest": dg, **solved(lambda: spla.factorized(A)(b))}))
        if a.shape[0] <= 200:
            out.append((f"direct/{mid}/inv", {"digest": dg, **attempt(lambda: spla.inv(A))}))
        L = sp.tril(A, format="csr")
        out.append(
            (f"direct/{mid}/spsolve_triangular", {"digest": dg, **solved(lambda: spla.spsolve_triangular(L, b))})
        )
    out.append(("direct/spsolve(b sparse)", attempt(lambda: spla.spsolve(M().get("lap2d/10").scipy("csc"),
                                                                       sp.csc_array(np.ones((100, 2)))))))  # fmt: skip
    out.append(
        ("direct/spsolve(b (n,1))", attempt(lambda: spla.spsolve(M().get("lap2d/10").scipy("csc"), np.ones((100, 1)))))
    )
    out.append(("direct/spsolve(csr)", attempt(lambda: spla.spsolve(M().get("lap2d/10").scipy("csr"), np.ones(100)))))
    return out


def iterative():
    import scipy.sparse.linalg as spla

    out = []
    probs = {
        "cg": ("lap2d/30", "random_spd/1500", "random_herm/800"),
        "minres": ("lap2d/30", "random_indef/1500", "saddle/600x150"),
        "gmres": ("convdiff2d/40", "random_unsym/1000x8", "random_complex/800"),
        "bicgstab": ("convdiff2d/40", "random_unsym/1000x8", "random_complex/800"),
        "lsqr": ("rect/1200x300", "rect/60x20", "rect/80x30d", "lap2d/30"),
        "lsmr": ("rect/1200x300", "rect/60x20", "rect/80x30d", "lap2d/30"),
    }
    for fn, mids in probs.items():
        for mid in mids:
            a = M().get(mid)
            A = a.scipy("csr")
            b = M().rhs(a)
            for tol in ("default", "tight"):
                count = [0]

                def cb(*_):
                    count[0] += 1

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if fn in ("lsqr", "lsmr"):
                        kw = {} if tol == "default" else {"atol": 1e-12, "btol": 1e-12}
                        r = getattr(spla, fn)(A, b, **kw)
                        x, istop, itn = r[0], int(r[1]), int(r[2])
                        rec = {
                            "istop": istop,
                            "n_iter": itn,
                            "normr": plain(r[3]),
                            "normar": plain(r[4] if fn == "lsmr" else r[7]),
                        }
                    else:
                        kw = {} if tol == "default" else {"rtol": 1e-10}
                        if fn == "gmres":
                            kw["callback_type"] = "pr_norm"
                        x, info = getattr(spla, fn)(A, b, callback=cb, **kw)
                        rec = {"info": int(info), "n_callbacks": count[0]}
                rec["relres"] = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
                rec["sha"] = sha(x)
                if caught:
                    rec["warnings"] = sorted({w.category.__name__ for w in caught})
                out.append((f"iterative/{fn}/{mid}/{tol}", {"digest": M().digest(a), **rec}))
    return out


def eigen():
    import scipy.sparse.linalg as spla

    out = []
    for mid, which in (("lap2d/30", "LM"), ("lap2d/30", "SA"), ("lap2d/30", "LA"), ("random_spd/1500", "LM"),
                       ("random_indef/60", "BE"), ("random_herm/800", "LA")):  # fmt: skip
        a = M().get(mid)
        A = a.scipy("csr")
        try:
            w = spla.eigsh(A, k=6, which=which, v0=np.ones(a.shape[0]), return_eigenvectors=False)
            rec = {"eigenvalues": plain(np.sort(np.real(w)))}
        except Exception as e:
            rec = {"error": f"{type(e).__name__}: {e}"}
        out.append((f"eigen/eigsh/{mid}/{which}", {"digest": M().digest(a), **rec}))
    for mid, which in (("convdiff2d/40", "LM"), ("random_unsym/1000x8", "LR"), ("random_complex/800", "LM")):
        a = M().get(mid)
        w = spla.eigs(a.scipy("csr"), k=6, which=which, v0=np.ones(a.shape[0]), return_eigenvectors=False)
        w = w[np.lexsort((w.imag, w.real))]
        out.append(
            (f"eigen/eigs/{mid}/{which}", {"digest": M().digest(a), "eigenvalues": plain(w), "dtype": str(w.dtype)})
        )
    for mid in ("rect/1200x300", "rect/60x20", "lap2d/30"):
        a = M().get(mid)
        s = spla.svds(a.scipy("csr"), k=6, rng=np.random.default_rng(0), return_singular_vectors=False)
        out.append((f"eigen/svds/{mid}/LM", {"digest": M().digest(a), "singular_values": plain(np.sort(s)[::-1])}))
    return out


def matfuncs():
    import scipy.sparse.linalg as spla

    from tests.sparse import refdata

    out = []
    for p in refdata.points("expm")[::3]:
        a = M().get(p["matrix"])
        A = a.scipy("csc") * float(p["scale"])
        E = spla.expm(A)
        rec = {"digest": M().digest(a), "scale": float(p["scale"]), **describe(E)}
        ref = p["expm_hi"]
        rec["max_rel_error"] = float(np.abs(E.toarray().ravel() - ref).max() / np.abs(ref).max())
        out.append((f"matfuncs/expm/{p['matrix']}/{p['scale']}", rec))
    for mid in ("lap2d/30", "convdiff2d/40", "random_complex/800", "random_unsym/1000x8"):
        a = M().get(mid)
        A = a.scipy("csr")
        np.random.seed(0)
        est = spla.onenormest(A)
        exact = float(np.abs(a.dense()).sum(axis=0).max())
        b = M().rhs(a)
        y = spla.expm_multiply(-0.1 * A.tocsc(), b)
        rec = {"digest": M().digest(a), "onenormest": plain(est), "exact": exact, "expm_multiply_sha": sha(y)}
        for o in ("fro", 1, -1, np.inf, -np.inf):
            rec[f"norm_{o}"] = plain(spla.norm(A, o))
        rec["norm_2"] = attempt(lambda: spla.norm(A, 2))
        out.append((f"matfuncs/{mid}", rec))
    return out


GROUPS = {"ops": ops, "dups": dups, "spmv": spmv, "construct": construct, "direct": direct, "iterative": iterative,
          "eigen": eigen, "matfuncs": matfuncs}  # fmt: skip


def _run(name):
    return name, GROUPS[name]()


def main(argv=None):
    import argparse

    import scipy

    from bench import cli

    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    cli.add_jobs(ap)
    jobs = ap.parse_args(argv).jobs
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
        res = dict(pool.imap_unordered(_run, list(GROUPS)))
    meta = {"tool": TOOL, "scipy": scipy.__version__, "numpy": np.__version__}
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    for g in GROUPS:
        lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in res[g]]
    (DATA / "scipy.json").write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"{len(lines) - 1} records -> {(DATA / 'scipy.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
