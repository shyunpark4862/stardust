"""sd.sparse against scipy.sparse 1.18.1 and R Matrix 1.7.6 (docs/design/sparse.md §5, §7).

The plan of the area's cases, milestone by milestone: every case below names the sd.sparse
names it needs, and cases() keeps a case only once they are public (SP0 has none, so the area
has no case yet; SP1 brings the conversions, SP2 the products and reductions, and so on). The
comparators are the ones the accuracy measurement will use: scipy.sparse and scipy.sparse.linalg
at their defaults (one thread), R's Matrix in its one session.

Matrices (the same as docs/design/sparse.md §5 and tools/sparse_probe.py):
  - random n x n with 10 stored entries a row (distinct uniform columns, values uniform in
    [-1, 1)) at nnz = 10^4, 10^5, 10^6, 10^7 (n = nnz / 10); the COO input of the conversion case
    has random coordinates (some duplicates) in random order;
  - the 2-D 5-point and 3-D 7-point Dirichlet Laplacians of tests/sparse/matrices.py (n = 10^4 to
    10^6) for the direct and iterative solves and the eigenvalues.
The data are numpy arrays made in the setup; scipy gets its arrays (int32 indices, as it makes
them), sd its own (int64), R a dgCMatrix built in its setup from 1-based coordinates.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable

import numpy as np

from ..registry import Case, Ref, RRef

NNZ = ((10**4, "10^4"), (10**5, "10^5"), (10**6, "10^6"), (10**7, "10^7"))
GRIDS = ("lap2d/100", "lap2d/300", "lap2d/1000", "lap3d/22", "lap3d/45")


def _grid_size(mid):
    """The matrix shape of a grid id: (n, n) with n = g^dim."""
    kind, g = mid.split("/")
    n = int(g) ** int(kind[3])
    return f"({n}, {n})"


def _random(nnz, seed):
    g = np.random.Generator(np.random.PCG64(seed))
    n = nnz // 10
    cols = np.sort(g.integers(0, n, (n, 10)), axis=1)
    keep = np.ones_like(cols, dtype=bool)
    keep[:, 1:] = cols[:, 1:] != cols[:, :-1]
    rows = np.repeat(np.arange(n), 10).reshape(n, 10)
    return n, rows[keep].astype(np.int64), cols[keep].astype(np.int64), g.uniform(-1, 1, int(keep.sum()))


def _array_setup(nnz):
    def setup():
        import scipy.sparse as sp

        n, r, c, v = _random(nnz, 20261120 + nnz)
        _, r2, c2, v2 = _random(nnz, 20261121 + nnz)
        g = np.random.Generator(np.random.PCG64(nnz))
        a = sp.csr_array((v, (r, c)), shape=(n, n))
        b = sp.csr_array((v2, (r2, c2)), shape=(n, n))
        cr, cc = g.integers(0, n, nnz), g.integers(0, n, nnz)
        return {
            "n": n, "row": r, "col": c, "val": v, "row2": r2, "col2": c2, "val2": v2,
            "a": a, "b": b, "csc": a.tocsc(), "coo": sp.coo_array((g.uniform(-1, 1, nnz), (cr, cc)), shape=(n, n)),
            "x": g.uniform(-1, 1, n), "X": g.uniform(-1, 1, (n, 16)),
            "r_i": (r + 1).astype(np.float64), "r_j": (c + 1).astype(np.float64), "r_x": v,
            "r_n": np.array([float(n)]),
        }  # fmt: skip

    return setup


def _grid_setup(mid):
    def setup():
        from tests.sparse import matrices as M

        a = M.get(mid)
        n = a.shape[0]
        b = np.sin(0.37 * np.arange(n)) + 0.1
        return {"m": a, "a": a.scipy("csc"), "b": b, "n": n}

    return setup


def _sd_csr(d, key="a"):
    import stardust as sd

    a = d[key].tocsr()
    return sd.sparse.csr_array(
        (sd.asarray(a.data), sd.asarray(a.indices.astype(np.int64)), sd.asarray(a.indptr.astype(np.int64))),
        shape=a.shape,
    )


def _sd(x):
    import stardust as sd

    return sd.asarray(x)


R_SETUP = "suppressMessages(library(Matrix)); A <- sparseMatrix(i = r_i, j = r_j, x = r_x, dims = c(n, n))"
R_DATA = {"r_i": "r_i", "r_j": "r_j", "r_x": "r_x"}


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]  # sd.sparse names the sd side calls
    sd: Callable  # d -> the call
    scipy: Callable  # d -> the call
    r: str = ""  # an R expression (A, x and n are defined), or none
    limit: int | None = None  # the largest nnz the case runs at
    quick: bool = False
    note: str = ""


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


ARRAY_SPECS = (
    Spec("SP1", "convert", "COO to CSR", ("coo_array",), lambda d: (lambda A: lambda: A.tocsr())(
        __import__("stardust").sparse.coo_array((_sd(d["coo"].data), (_sd(d["coo"].coords[0].astype(np.int64)),
            _sd(d["coo"].coords[1].astype(np.int64)))), shape=d["coo"].shape)),
         lambda d: lambda: d["coo"].tocsr(), r="sparseMatrix(i = r_i, j = r_j, x = r_x, dims = c(n, n))",
         quick=True, note="중복이 있는 무작위 자리"),
    Spec("SP1", "convert", "CSR to CSC", ("csr_array",), lambda d: (lambda A: lambda: A.tocsc())(_sd_csr(d)),
         lambda d: lambda: d["a"].tocsc(), r="t(A)"),
    Spec("SP1", "convert", "toarray", ("csr_array",), lambda d: (lambda A: lambda: A.toarray())(_sd_csr(d)),
         lambda d: lambda: d["a"].toarray(), limit=10**5),
    Spec("SP2", "products", "A @ x", ("csr_array",), lambda d: (lambda A, x: lambda: A @ x)(_sd_csr(d), _sd(d["x"])),
         lambda d: lambda: d["a"] @ d["x"], r="A %*% x", quick=True),
    Spec("SP2", "products", "A @ X (k = 16)", ("csr_array",),
         lambda d: (lambda A, X: lambda: A @ X)(_sd_csr(d), _sd(d["X"])), lambda d: lambda: d["a"] @ d["X"]),
    Spec("SP2", "products", "A.T @ x", ("csr_array",), lambda d: (lambda A, x: lambda: A.T @ x)(_sd_csr(d), _sd(d["x"])),
         lambda d: lambda: d["a"].T @ d["x"], r="crossprod(A, x)"),
    Spec("SP2", "products", "A @ B", ("csr_array",), lambda d: (lambda A, B: lambda: A @ B)(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: d["a"] @ d["b"], r="A %*% A", limit=10**6),
    Spec("SP2", "arithmetic", "A + B", ("csr_array",), lambda d: (lambda A, B: lambda: A + B)(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: d["a"] + d["b"], r="A + A"),
    Spec("SP2", "arithmetic", "A * B", ("csr_array",), lambda d: (lambda A, B: lambda: A * B)(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: d["a"] * d["b"]),
    Spec("SP2", "arithmetic", "A * 2.0", ("csr_array",), lambda d: (lambda A: lambda: A * 2.0)(_sd_csr(d)),
         lambda d: lambda: d["a"] * 2.0),
    Spec("SP2", "reductions", "sum(axis=0)", ("csr_array",), lambda d: (lambda A: lambda: A.sum(axis=0))(_sd_csr(d)),
         lambda d: lambda: d["a"].sum(axis=0), r="colSums(A)"),
    Spec("SP2", "reductions", "sum(axis=1)", ("csr_array",), lambda d: (lambda A: lambda: A.sum(axis=1))(_sd_csr(d)),
         lambda d: lambda: d["a"].sum(axis=1), r="rowSums(A)"),
    Spec("SP2", "reductions", "sum()", ("csr_array",), lambda d: (lambda A: lambda: A.sum())(_sd_csr(d)),
         lambda d: lambda: d["a"].sum(), r="sum(A)"),
    Spec("SP2", "reductions", "max(axis=1)", ("csr_array",), lambda d: (lambda A: lambda: A.max(axis=1))(_sd_csr(d)),
         lambda d: lambda: d["a"].max(axis=1), note="scipy는 COO 희소를 돌려준다"),
    Spec("SP3", "indexing", "A[rows 1/10]", ("csr_array",),
         lambda d: (lambda A, k: lambda: A[:k])(_sd_csr(d), d["n"] // 10), lambda d: lambda: d["a"][: d["n"] // 10]),
    Spec("SP3", "indexing", "A[:, cols 1/10]", ("csr_array",),
         lambda d: (lambda A, k: lambda: A[:, :k])(_sd_csr(d), d["n"] // 10), lambda d: lambda: d["a"][:, : d["n"] // 10]),
    Spec("SP3", "construct", "vstack", ("csr_array", "vstack"),
         lambda d: (lambda A, B: lambda: __import__("stardust").sparse.vstack([A, B]))(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: __import__("scipy.sparse", fromlist=["vstack"]).vstack([d["a"], d["b"]])),
    Spec("SP3", "construct", "kron(eye(10), A)", ("csr_array", "eye_array", "kron"),
         lambda d: (lambda A: lambda: __import__("stardust").sparse.kron(__import__("stardust").sparse.eye_array(10), A))(_sd_csr(d)),
         lambda d: lambda: __import__("scipy.sparse", fromlist=["kron"]).kron(
             __import__("scipy.sparse", fromlist=["eye_array"]).eye_array(10), d["a"], format="csr"),
         r="kronecker(Diagonal(10), A)", limit=10**6),
    Spec("SP7", "norms", "norm(A, 1)", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: __import__("stardust").sparse.norm(A, 1))(_sd_csr(d)),
         lambda d: lambda: __import__("scipy.sparse.linalg", fromlist=["norm"]).norm(d["a"], 1), r='norm(A, "1")'),
    Spec("SP7", "norms", "onenormest", ("csr_array", "onenormest"),
         lambda d: (lambda A: lambda: __import__("stardust").sparse.onenormest(A))(_sd_csr(d)),
         lambda d: lambda: __import__("scipy.sparse.linalg", fromlist=["onenormest"]).onenormest(d["a"]),
         note="scipy는 시드 없는 무작위 열"),
)  # fmt: skip


def _la(name):
    return getattr(__import__("scipy.sparse.linalg", fromlist=[name]), name)


def _sds(name):
    return getattr(__import__("stardust").sparse, name)


SOLVER_SPECS = (
    Spec("SP4", "direct", "spsolve", ("csc_array", "spsolve"),
         lambda d: (lambda A, b: lambda: _sds("spsolve")(A, b))(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("spsolve")(d["a"], d["b"]), r="solve(A, b)", quick=True,
         note="sd는 Accelerate LU/AMD, scipy는 SuperLU/COLAMD, R은 CSparse LU"),
    Spec("SP4", "direct", "splu", ("csc_array", "splu"), lambda d: (lambda A: lambda: _sds("splu")(A))(_sd_csc(d)),
         lambda d: lambda: _la("splu")(d["a"]), r="lu(A)"),
    Spec("SP5", "iterative", "cg (rtol 1e-10)", ("csc_array", "cg"),
         lambda d: (lambda A, b: lambda: _sds("cg")(A, b, rtol=1e-10, max_iter=20000))(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("cg")(d["a"], d["b"], rtol=1e-10, maxiter=20000)),
    Spec("SP5", "iterative", "minres (rtol 1e-10)", ("csc_array", "minres"),
         lambda d: (lambda A, b: lambda: _sds("minres")(A, b, rtol=1e-10, max_iter=20000))(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("minres")(d["a"], d["b"], rtol=1e-10, maxiter=20000)),
    Spec("SP6", "eigen", "eigsh(k=6)", ("csc_array", "eigsh"),
         lambda d: (lambda A: lambda: _sds("eigsh")(A, k=6))(_sd_csc(d)),
         lambda d: lambda: _la("eigsh")(d["a"], k=6, v0=np.ones(d["n"])), note="같은 시작 벡터(1의 벡터)"),
    Spec("SP7", "matfuncs", "expm_multiply(-0.1 A, b)", ("csc_array", "expm_multiply"),
         lambda d: (lambda A, b: lambda: _sds("expm_multiply")(A, b))(_sd_csc(d, -0.1), _sd(d["b"])),
         lambda d: (lambda A: lambda: _la("expm_multiply")(A, d["b"]))(-0.1 * d["a"])),
)  # fmt: skip


def _sd_csc(d, scale=1.0):
    import stardust as sd

    a = d["a"]
    return sd.sparse.csc_array(
        (sd.asarray(scale * a.data), sd.asarray(a.indices.astype(np.int64)), sd.asarray(a.indptr.astype(np.int64))),
        shape=a.shape,
    )


R_GRID_SETUP = (
    "suppressMessages(library(Matrix)); A <- sparseMatrix(i = r_i, j = r_j, x = r_x, dims = c(length(b), length(b)))"
)


def _grid_r(setup):
    def wrapped():
        d = setup()
        r, c, v = d["m"].canonical()
        d.update({"r_i": (r + 1).astype(np.float64), "r_j": (c + 1).astype(np.float64), "r_x": v})
        return d

    return wrapped


def _public(names) -> bool:
    import stardust as sd

    return all(n in sd.sparse.__all__ for n in names)


def cases() -> list[Case]:
    out = []
    for s in ARRAY_SPECS:
        if not _public(s.needs):
            continue
        for nnz, size in NNZ:
            if s.limit and nnz > s.limit:
                continue
            refs = [Ref("scipy", lambda d, f=s.scipy: _quiet(lambda: f(d)))]
            if s.r:
                refs.append(RRef("R", s.r, data={**R_DATA, "r_n": "n", "x": "x"}, setup=R_SETUP, packages=("Matrix",)))
            out.append(Case(area="sparse", group=s.group, name=s.name, size=size, n=nnz, setup=_array_setup(nnz),
                            sd=s.sd, refs=tuple(refs), quick=s.quick and nnz == 10**6, note=s.note))  # fmt: skip
    for s in SOLVER_SPECS:
        if not _public(s.needs):
            continue
        for mid in GRIDS:
            refs = [Ref("scipy", lambda d, f=s.scipy: _quiet(lambda: f(d)))]
            if s.r:
                refs.append(RRef("R", s.r, data={**R_DATA, "b": "b"}, setup=R_GRID_SETUP, packages=("Matrix",)))
            out.append(Case(area="sparse", group=s.group, name=f"{s.name} {mid.split('/')[0]}", size=_grid_size(mid),
                            setup=_grid_r(_grid_setup(mid)), sd=s.sd, refs=tuple(refs),
                            quick=s.quick and mid == "lap2d/300", note=s.note))  # fmt: skip
    return out


TITLES = {
    "convert": "형식 변환",
    "products": "곱 (SpMV, SpMM, SpGEMM)",
    "arithmetic": "산술",
    "reductions": "리덕션",
    "indexing": "인덱싱",
    "construct": "만들기",
    "norms": "노름",
    "direct": "직접 풀이",
    "iterative": "반복법",
    "eigen": "고유값",
    "matfuncs": "행렬 함수",
}
