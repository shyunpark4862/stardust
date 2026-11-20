"""R's Matrix package on the test matrices of tests/sparse/matrices.py: the comparison record of
sd.sparse (docs/design/sparse.md §7). Writes tests/sparse/data/r.json, so the tests and the
accuracy tables compare with R on machines without R.

    .venv/bin/python tools/gen_sparse_r.py          (needs Rscript on PATH with Matrix)

R 4.6.2 with Matrix 1.7.6, real matrices only (Matrix has no complex sparse class). Each matrix
goes to R as its canonical triplets (%.17g, exact) through sparseMatrix(index1 = FALSE), a
dgCMatrix. Per matrix:

    matvec, rmatvec   A %*% x, crossprod(A, w) with the x, w of tests/sparse/ref/spmv.npz
                      (kind 'random')
    rowSums, colSums, sum
    norm_1, norm_I, norm_F, norm_M   norm(A, type)
    solve_lu          solve(A, b): Matrix's sparse LU (CSparse) for a general dgCMatrix
    solve_chol        solve(forceSymmetric(A), b): CHOLMOD's Cholesky, for symmetric positive
                      definite matrices (an error otherwise is recorded)
    rcond             rcond(A) (1-norm, through Matrix's LU)
    lstsq             qr.coef(qr(A), b): the sparse QR's least-squares solution (tall matrices)
    expm              expm(scale A) (Matrix's Padé with scaling and squaring, dense) at the
                      scales of tests/sparse/ref/expm.npz, n <= 100
    error             R's message where a call fails (singular matrices)

b is tests/sparse/matrices.rhs(a). A vector of more than 300 entries is kept as its maximum
relative difference to the Arb reference (max_rel_*) with its SHA-256; shorter ones as values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "sparse" / "data" / "r.json"
TOOL = "tools/gen_sparse_r.py"
KEEP = 300  # vectors up to this length are stored

MATRICES = (
    "lap1d/10", "lap2d/10", "lap2d/30", "lap3d/6", "convdiff2d/40", "random_unsym/50x5", "random_unsym/1000x8",
    "random_spd/60", "random_spd/1500", "random_indef/60", "banded/200x2x3", "blockdiag/20x8", "saddle/40x10",
    "graded/100x8", "near_singular/20", "long_rows/300", "cancel/50", "coo/dups", "rect/60x20", "rect/1200x300",
    "struct_singular/40", "dupcol/40", "lap2d/10n",
)  # fmt: skip

PRELUDE = r"""
suppressMessages(library(Matrix))
out <- function(id, name, v) {
  v <- as.numeric(v)
  cat(id, "\t", name, "\t", paste(ifelse(is.na(v), "NA", sprintf("%.17g", v)), collapse = " "), "\n", sep = "")
}
run <- function(id, name, expr) {
  r <- tryCatch(suppressWarnings(expr), error = function(e) e)
  if (inherits(r, "error")) cat(id, "\terror_", name, "\t", gsub("[\t\n]", " ", conditionMessage(r)), "\n", sep = "")
  else out(id, name, as.numeric(as.matrix(r)))
}
rd <- function(f) scan(f, quiet = TRUE)
"""


def fmt(v):
    return " ".join(repr(float(x)) for x in np.ravel(v))


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def main(argv=None):
    from tests.sparse import matrices as M
    from tests.sparse import refdata

    spmv = {p["matrix"]: p for p in refdata.points("spmv") if p["kind"] == "random"}
    solve = {p["matrix"]: p for p in refdata.points("solve")}
    lstsq = {p["matrix"]: p for p in refdata.points("lstsq")}
    expm = {}
    for p in refdata.points("expm"):
        expm.setdefault(p["matrix"], []).append(p)
    sums = {p["matrix"]: p for p in refdata.points("sums")}
    norms = {p["matrix"]: p for p in refdata.points("norms")}
    script = [PRELUDE]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for k, mid in enumerate(MATRICES):
            a = M.get(mid)
            r, c, v = a.canonical()
            base = tmp / f"m{k}"
            for name, arr in (("i", r), ("j", c), ("x", v.astype(float)), ("b", M.rhs(a))):
                (base.parent / f"{base.name}_{name}.txt").write_text(fmt(arr))
            m, n = a.shape
            f = lambda s: str(tmp / f"m{k}_{s}.txt")  # noqa: E731
            script.append(f'id <- "{mid}"\nA <- sparseMatrix(i = rd("{f("i")}"), j = rd("{f("j")}"), '
                          f'x = rd("{f("x")}"), dims = c({m}, {n}), index1 = FALSE)\nb <- rd("{f("b")}")')  # fmt: skip
            if mid in spmv:
                (tmp / f"m{k}_vx.txt").write_text(fmt(spmv[mid]["x"]))
                (tmp / f"m{k}_vw.txt").write_text(fmt(spmv[mid]["w"]))
                script.append(
                    f'run(id, "matvec", A %*% rd("{f("vx")}"))\nrun(id, "rmatvec", crossprod(A, rd("{f("vw")}")))'
                )
            script.append('run(id, "rowSums", rowSums(A))\nrun(id, "colSums", colSums(A))\nrun(id, "sum", sum(A))')
            for t in ("1", "I", "F", "M"):
                script.append(f'run(id, "norm_{t}", norm(A, "{t}"))')
            if m == n:
                script.append('run(id, "solve_lu", solve(A, b))\nrun(id, "rcond", rcond(A))')
                if "spd" in a.tags:
                    script.append('run(id, "solve_chol", solve(forceSymmetric(A), b))')
            if m > n:
                script.append('run(id, "lstsq", qr.coef(qr(A), b))')
            for e in expm.get(mid, []):
                if n <= 100:
                    s = float(e["scale"])
                    script.append(f'run(id, "expm@{s!r}", t(as.matrix(expm(A * {s!r}))))')
        (tmp / "run.R").write_text("\n".join(script) + "\n")
        env = dict(os.environ, LANGUAGE="en", LC_ALL="C")  # R's messages in English
        p = subprocess.run(["Rscript", str(tmp / "run.R")], capture_output=True, text=True, env=env)
        if p.returncode:
            sys.exit(p.stderr[-3000:])
        ver = subprocess.run(["Rscript", "-e", 'cat(R.version.string, "|", as.character(packageVersion("Matrix")))'],
                             capture_output=True, text=True).stdout.split("|")  # fmt: skip
    got = {}
    for line in p.stdout.splitlines():
        mid, name, vals = line.split("\t", 2)
        if name.startswith("error_"):
            got.setdefault(mid, {})[name] = vals.strip()
        else:
            got.setdefault(mid, {})[name] = np.array([math.nan if s == "NA" else float(s) for s in vals.split()])
    refs = {"matvec": (spmv, "matvec"), "rmatvec": (spmv, "rmatvec"), "rowSums": (sums, "row_sums"),
            "colSums": (sums, "col_sums"), "sum": (sums, "total"), "solve_lu": (solve, "solve"),
            "solve_chol": (solve, "solve"), "lstsq": (lstsq, "lstsq"), "norm_1": (norms, "1"),
            "norm_I": (norms, "inf"), "norm_F": (norms, "fro")}  # fmt: skip
    records = []
    for mid in MATRICES:
        rec = {"digest": M.digest(M.get(mid))}
        for name, v in got.get(mid, {}).items():
            if isinstance(v, str):
                rec[name] = v
                continue
            src = refs.get(name)
            ref = None
            if name.startswith("expm@"):
                s = float(name[5:])
                ref = next((e["expm_hi"] for e in expm.get(mid, []) if float(e["scale"]) == s), None)
            elif src and mid in src[0] and f"{src[1]}_hi" in src[0][mid]:
                ref = np.real(np.asarray(src[0][mid][f"{src[1]}_hi"]))
            if v.size <= KEEP:
                rec[name] = [x if math.isfinite(x) else repr(x) for x in v.tolist()]
            else:
                rec[name + "_sha"] = sha(v)
            if ref is not None and np.size(ref) == v.size and v.size:
                scale = float(np.abs(ref).max())
                rec["max_rel_" + name] = float(np.abs(v - ref).max() / scale) if scale else float(np.abs(v).max())
        records.append((mid, rec))
    meta = {"tool": TOOL, "R": ver[0].strip(), "Matrix": ver[1].strip()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in records]
    OUT.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"{len(records)} matrices -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
