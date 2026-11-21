"""sd.sparse against scipy.sparse 1.18.1 and R Matrix 1.7.6 (docs/design/sparse.md §5, §7).

The plan of the area's cases, milestone by milestone: every case below names its milestone and
the sd.sparse names it needs, and cases() keeps a case once its milestone is in DONE and the
names are public (SP1 brings the formats and conversions, SP2 the products and reductions, SP3
indexing and the builders, and so on; the operators of SP2 need no new name, hence the milestone). The
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

from ..registry import Case, Check, Ref, RRef

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
        cx = g.uniform(-1, 1, nnz)
        return {
            "n": n, "row": r, "col": c, "val": v, "row2": r2, "col2": c2, "val2": v2,
            "a": a, "b": b, "csc": a.tocsc(), "coo": sp.coo_array((cx, (cr, cc)), shape=(n, n)), "coo_x": cx,
            "x": g.uniform(-1, 1, n), "X": g.uniform(-1, 1, (n, 16)),
            "r_i": (r + 1).astype(np.float64), "r_j": (c + 1).astype(np.float64), "r_x": v,
            "r_ci": (cr + 1).astype(np.float64), "r_cj": (cc + 1).astype(np.float64),
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


def _picks(d, part):
    """n / part random row (or column) indices, repeats possible (fixed seed)."""
    g = np.random.Generator(np.random.PCG64(d["n"] + part))
    return g.integers(0, d["n"], d["n"] // part)


def _sd_coo(d, key="a"):
    import stardust as sd

    a = d[key].tocoo()
    return sd.sparse.coo_array(
        (sd.asarray(a.data), (sd.asarray(a.coords[0].astype(np.int64)), sd.asarray(a.coords[1].astype(np.int64)))),
        shape=a.shape,
    )


R_SETUP = "suppressMessages(library(Matrix)); A <- sparseMatrix(i = r_i, j = r_j, x = r_x, dims = c(n, n))"
R_DATA = {"r_i": "r_i", "r_j": "r_j", "r_x": "r_x"}


def _summary(x):
    """A sparse result for the smoke check: the sums of the values, of value x (row + 1) and of
    value x (col + 1) -- the matrix up to rounding, whatever the format, order or repeats. Dense
    and other results pass."""
    import scipy.sparse as sp

    import stardust as sd

    if isinstance(x, sd.sparse.sparray) or sp.issparse(x):
        c = x.tocoo()
        r, j, v = (np.asarray(a) for a in (c.coords[0], c.coords[1], c.data))
        return np.array([v.sum(), (v * (r + 1)).sum(), (v * (j + 1)).sum()])
    return x


SPARSE_CHECK = Check(sd=_summary, ref=lambda label, x: x if label == "R" else _summary(x), rtol=1e-9)
# The same three sums of an R sparse matrix (1-based indices).
R_SUMMARY = "{ s <- summary(.v); c(sum(s$x), sum(s$x * s$i), sum(s$x * s$j)) }"


def _densified(label, x):
    """A comparator's result for the SP2 smoke check: a sparse result as the same three sums as sd's
    (_summary), except where sd gives a dense array for a sparse one (scipy's max(axis=) is COO)."""
    import scipy.sparse as sp

    if sp.issparse(x) and x.ndim == 1:
        return x.toarray()
    return x if label == "R" else _summary(x)


SP2_CHECK = Check(sd=_summary, ref=_densified, rtol=1e-9)


def _size(x):
    return np.array([float(x.shape[0]), float(x.shape[1]), float(x.nnz)])


# A factorization object (SparseLU, scipy's SuperLU): its shape.
FACTOR_CHECK = Check(sd=lambda f: np.array(f.shape, float), ref=lambda label, f: np.array(f.shape, float), rtol=0.0)

# An iterative solution: the same point to what the tolerances allow (the iterates of two
# implementations part at rounding, amplified by the condition: up to 1e-3 relative here).
ITER_CHECK = Check(rtol=1e-2, agree=0.9)

# Random results of different generators: the shape and the number of stored entries.
SIZE_CHECK = Check(sd=_size, ref=lambda label, x: _size(x), rtol=0.0)


def _summed(A):
    B = A.copy()
    B.sum_duplicates()
    return B


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
    r_differs: str = ""  # the R expression computes something else by design: why
    r_data: tuple = ()  # more (case data key, R variable) pairs the R expression reads
    r_sparse: bool = False  # the R result is a sparse matrix: compared by the three sums (R_SUMMARY)
    check: object = None  # the smoke check, when not the milestone's
    grids: tuple = GRIDS  # the grids a solver case runs on


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


ARRAY_SPECS = (
    Spec("SP1", "convert", "COO to CSR", ("coo_array",), lambda d: (lambda A: lambda: A.tocsr())(
        __import__("stardust").sparse.coo_array((_sd(d["coo"].data), (_sd(d["coo"].coords[0].astype(np.int64)),
            _sd(d["coo"].coords[1].astype(np.int64)))), shape=d["coo"].shape)),
         lambda d: lambda: d["coo"].tocsr(), r="sparseMatrix(i = ci, j = cj, x = cx, dims = c(n, n))",
         r_data=(("r_ci", "ci"), ("r_cj", "cj"), ("coo_x", "cx")), quick=True, note="중복이 있는 무작위 자리"),
    Spec("SP1", "convert", "CSR to CSC", ("csr_array",), lambda d: (lambda A: lambda: A.tocsc())(_sd_csr(d)),
         lambda d: lambda: d["a"].tocsc(), r="t(A)", r_differs="t(A): the transpose, the same reordering of a dgCMatrix"),
    Spec("SP1", "convert", "COO to CSR (sorted)", ("coo_array",), lambda d: (lambda A: lambda: A.tocsr())(_sd_coo(d)),
         lambda d: (lambda A: lambda: A.tocsr())(d["a"].tocoo()), note="표준 꼴의 COO: 정렬 없이 indptr만"),
    Spec("SP1", "convert", "CSR to COO", ("csr_array",), lambda d: (lambda A: lambda: A.tocoo())(_sd_csr(d)),
         lambda d: lambda: d["a"].tocoo(), r='as(A, "TsparseMatrix")'),
    Spec("SP1", "convert", "COO sum_duplicates", ("coo_array",),
         lambda d: (lambda A: lambda: _summed(A))(_sd_coo(d, "coo")),
         lambda d: lambda: _summed(d["coo"]), note="복사 포함(제자리 연산이라)"),
    Spec("SP1", "construct", "coo_array((data, (row, col)))", ("coo_array",),
         lambda d: (lambda v, r, c, n: lambda: __import__("stardust").sparse.coo_array((v, (r, c)), shape=(n, n)))(
             _sd(d["val"]), _sd(d["row"]), _sd(d["col"]), d["n"]),
         lambda d: lambda: __import__("scipy.sparse", fromlist=["coo_array"]).coo_array(
             (d["val"], (d["row"], d["col"])), shape=(d["n"], d["n"])),
         r="sparseMatrix(i = r_i, j = r_j, x = r_x, dims = c(n, n), repr = \"T\")",
         note="sd는 색인을 복사하고 검사한다, scipy는 그대로 쓴다"),
    Spec("SP1", "construct", "csr_array((data, indices, indptr))", ("csr_array",),
         lambda d: (lambda p, n: lambda: __import__("stardust").sparse.csr_array(p, shape=(n, n)))(
             (_sd(d["a"].data), _sd(d["a"].indices.astype(np.int64)), _sd(d["a"].indptr.astype(np.int64))), d["n"]),
         lambda d: (lambda p, n: lambda: __import__("scipy.sparse", fromlist=["csr_array"]).csr_array(p, shape=(n, n)))(
             (d["a"].data, d["a"].indices.astype(np.int64), d["a"].indptr.astype(np.int64)), d["n"]),
         note="sd는 표준 꼴을 검사하고 색인을 복사한다, scipy는 검사 없이 그대로 쓴다"),
    Spec("SP1", "construct", "csr_array(dense)", ("csr_array",),
         lambda d: (lambda x: lambda: __import__("stardust").sparse.csr_array(x))(_sd(d["a"].toarray())),
         lambda d: (lambda x: lambda: __import__("scipy.sparse", fromlist=["csr_array"]).csr_array(x))(d["a"].toarray()),
         limit=10**5),
    Spec("SP1", "convert", "toarray", ("csr_array",), lambda d: (lambda A: lambda: A.toarray())(_sd_csr(d)),
         lambda d: lambda: d["a"].toarray(), limit=10**5),
    Spec("SP1", "convert", "to_scipy", ("csr_array",), lambda d: (lambda A: lambda: A.to_scipy())(_sd_csr(d)),
         lambda d: lambda: d["a"].copy(), note="sd는 복사 없는 뷰, scipy 쪽은 같은 객체의 copy()"),
    Spec("SP2", "products", "A @ x", ("csr_array",), lambda d: (lambda A, x: lambda: A @ x)(_sd_csr(d), _sd(d["x"])),
         lambda d: lambda: d["a"] @ d["x"], r="A %*% x", quick=True),
    Spec("SP2", "products", "A @ X (k = 16)", ("csr_array",),
         lambda d: (lambda A, X: lambda: A @ X)(_sd_csr(d), _sd(d["X"])), lambda d: lambda: d["a"] @ d["X"]),
    Spec("SP2", "products", "A.T @ x", ("csr_array",), lambda d: (lambda A, x: lambda: A.T @ x)(_sd_csr(d), _sd(d["x"])),
         lambda d: lambda: d["a"].T @ d["x"], r="crossprod(A, x)"),
    Spec("SP2", "products", "A @ B", ("csr_array",), lambda d: (lambda A, B: lambda: A @ B)(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: d["a"] @ d["b"], r="A %*% A", limit=10**6, r_sparse=True,
         r_differs="A %*% A: the session holds one matrix"),
    Spec("SP2", "arithmetic", "A + B", ("csr_array",), lambda d: (lambda A, B: lambda: A + B)(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: d["a"] + d["b"], r="A + A", r_sparse=True, r_differs="A + A: the session holds one matrix"),
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
    Spec("SP2", "products", "x @ A", ("csr_array",), lambda d: (lambda A, x: lambda: x @ A)(_sd_csr(d), _sd(d["x"])),
         lambda d: lambda: d["x"] @ d["a"], note="CSR의 열 방향: 1스레드는 흩뿌리기, 여러 스레드는 전치 뒤 모으기"),
    Spec("SP2", "products", "CSC @ x", ("csr_array",),
         lambda d: (lambda A, x: lambda: A @ x)(_sd_csr(d).tocsc(), _sd(d["x"])), lambda d: lambda: d["csc"] @ d["x"]),
    Spec("SP2", "arithmetic", "A != B", ("csr_array",),
         lambda d: (lambda A, B: lambda: A != B)(_sd_csr(d), _sd_csr(d, "b")), lambda d: lambda: d["a"] != d["b"],
         note="==는 두 암시적 0 사이가 참이라 결과가 밀집에 가깝다(scipy도 같다): 재지 않는다"),
    Spec("SP2", "arithmetic", "A.multiply(X dense)", ("csr_array",),
         lambda d: (lambda A, X: lambda: A.multiply(X))(_sd_csr(d), _sd(d["a"].toarray())),
         lambda d: (lambda X: lambda: d["a"].multiply(X))(d["a"].toarray()), limit=10**5,
         note="sd는 A의 형식(CSR), scipy는 COO"),
    Spec("SP2", "elementwise", "A.sin()", ("csr_array",), lambda d: (lambda A: lambda: A.sin())(_sd_csr(d)),
         lambda d: lambda: d["a"].sin()),
    Spec("SP2", "reductions", "mean(axis=0)", ("csr_array",), lambda d: (lambda A: lambda: A.mean(axis=0))(_sd_csr(d)),
         lambda d: lambda: d["a"].mean(axis=0), r="colMeans(A)"),
    Spec("SP2", "reductions", "argmax(axis=0)", ("csr_array",),
         lambda d: (lambda A: lambda: A.argmax(axis=0))(_sd_csr(d)), lambda d: lambda: d["a"].argmax(axis=0),
         limit=10**6, note="scipy는 줄마다 Python 반복"),
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
         r='kronecker(as(Diagonal(10), "CsparseMatrix"), A)', limit=10**6, r_sparse=True),
    Spec("SP3", "indexing", "A[random rows 1/10]", ("csr_array",),
         lambda d: (lambda A, k: lambda: A[k])(_sd_csr(d), _sd(_picks(d, 10))), lambda d: lambda: d["a"][_picks(d, 10)]),
    Spec("SP3", "indexing", "A[:, random cols 1/10]", ("csr_array",),
         lambda d: (lambda A, k: lambda: A[:, k])(_sd_csr(d), _sd(_picks(d, 10))),
         lambda d: lambda: d["a"][:, _picks(d, 10)]),
    Spec("SP3", "indexing", "A[::-1, ::2]", ("csr_array",), lambda d: (lambda A: lambda: A[::-1, ::2])(_sd_csr(d)),
         lambda d: lambda: d["a"][::-1, ::2]),
    Spec("SP3", "indexing", "diagonal()", ("csr_array",), lambda d: (lambda A: lambda: A.diagonal())(_sd_csr(d)),
         lambda d: lambda: d["a"].diagonal()),
    Spec("SP3", "construct", "hstack", ("csr_array", "hstack"),
         lambda d: (lambda A, B: lambda: __import__("stardust").sparse.hstack([A, B]))(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: __import__("scipy.sparse", fromlist=["hstack"]).hstack([d["a"], d["b"]])),
    Spec("SP3", "construct", "tril(A)", ("csr_array", "tril"),
         lambda d: (lambda A: lambda: __import__("stardust").sparse.tril(A))(_sd_csr(d)),
         lambda d: lambda: __import__("scipy.sparse", fromlist=["tril"]).tril(d["a"]), r="tril(A)", r_sparse=True),
    Spec("SP3", "construct", "random_array (density 10/n)", ("random_array",),
         lambda d: lambda: __import__("stardust").sparse.random_array(
             (d["n"], d["n"]), density=10 / d["n"], rng=__import__("stardust").random.default_rng(1)),
         lambda d: lambda: __import__("scipy.sparse", fromlist=["random_array"]).random_array(
             (d["n"], d["n"]), density=10 / d["n"], rng=1),
         note="난수 생성기가 달라 값은 다르다(검사는 모양과 저장 수만)", check="size"),
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


# The direct solves have no R comparator: Matrix keeps a matrix's factorization in the object
# (A@factors), so every timed call after the first is a lookup (0.01-0.4 ms in SP4's smoke run),
# and the first, CSparse's LU in the setup, took about 11 minutes and 6.5 GB on lap2d/1000 and
# lap3d/45. docs/design/sparse.md §5 has R's times measured once (SP0).
SOLVER_SPECS = (
    Spec("SP4", "direct", "spsolve", ("csc_array", "spsolve"),
         lambda d: (lambda A, b: lambda: _sds("spsolve")(A, b))(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("spsolve")(d["a"], d["b"]), quick=True,
         note="sd는 Accelerate LU/AMD, scipy는 SuperLU/COLAMD"),
    Spec("SP4", "direct", "splu", ("csc_array", "splu"), lambda d: (lambda A: lambda: _sds("splu")(A))(_sd_csc(d)),
         lambda d: lambda: _la("splu")(d["a"]), check=FACTOR_CHECK,
         note="sd는 특이 판정(조건수 추정)까지"),
    Spec("SP4", "direct", "splu solve", ("csc_array", "splu"),
         lambda d: (lambda f, b: lambda: f.solve(b))(_sds("splu")(_sd_csc(d)), _sd(d["b"])),
         lambda d: (lambda f: lambda: f.solve(d["b"]))(_quiet(lambda: _la("splu")(d["a"]))),
         note="sd는 반복 다듬기 한 번(풀이 둘과 SpMV 하나)"),
    Spec("SP4", "direct", "spsolve_triangular", ("csr_array", "tril", "spsolve_triangular"),
         lambda d: (lambda L, b: lambda: _sds("spsolve_triangular")(L, b, upper=False))(
             _sds("tril")(_sd_csc(d), format="csr"), _sd(d["b"])),
         lambda d: (lambda L: lambda: _la("spsolve_triangular")(L, d["b"]))(
             __import__("scipy.sparse", fromlist=["tril"]).tril(d["a"], format="csr"))),
    Spec("SP5", "iterative", "cg (rtol 1e-10)", ("csc_array", "cg"),
         lambda d: (lambda A, b: lambda: _sds("cg")(A, b, rtol=1e-10, max_iter=20000).x)(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("cg")(d["a"], d["b"], rtol=1e-10, maxiter=20000)[0], quick=True, check=ITER_CHECK),
    Spec("SP5", "iterative", "minres (rtol 1e-10)", ("csc_array", "minres"),
         lambda d: (lambda A, b: lambda: _sds("minres")(A, b, rtol=1e-10, max_iter=20000).x)(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("minres")(d["a"], d["b"], rtol=1e-10, maxiter=20000)[0], check=ITER_CHECK),
    Spec("SP5", "iterative", "bicgstab (rtol 1e-8)", ("csc_array", "bicgstab"),
         lambda d: (lambda A, b: lambda: _sds("bicgstab")(A, b, rtol=1e-8, max_iter=20000).x)(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("bicgstab")(d["a"], d["b"], rtol=1e-8, maxiter=20000)[0], check=ITER_CHECK),
    Spec("SP5", "iterative", "gmres (rtol 1e-8)", ("csc_array", "gmres"),
         lambda d: (lambda A, b: lambda: _sds("gmres")(A, b, rtol=1e-8, max_iter=2000).x)(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("gmres")(d["a"], d["b"], rtol=1e-8, maxiter=2000)[0], check=ITER_CHECK,
         grids=GRIDS[:2] + GRIDS[3:], note="restart 20(기본)"),
    Spec("SP5", "iterative", "lsqr (atol = btol = 1e-8)", ("csc_array", "lsqr"),
         lambda d: (lambda A, b: lambda: _sds("lsqr")(A, b, atol=1e-8, btol=1e-8, max_iter=20000).x)(
             _sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("lsqr")(d["a"], d["b"], atol=1e-8, btol=1e-8, iter_lim=20000)[0], check=ITER_CHECK,
         grids=GRIDS[:2] + GRIDS[3:]),
    Spec("SP5", "iterative", "lsmr (atol = btol = 1e-8)", ("csc_array", "lsmr"),
         lambda d: (lambda A, b: lambda: _sds("lsmr")(A, b, atol=1e-8, btol=1e-8, max_iter=20000).x)(
             _sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("lsmr")(d["a"], d["b"], atol=1e-8, btol=1e-8, maxiter=20000)[0], check=ITER_CHECK,
         grids=GRIDS[:2] + GRIDS[3:]),
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


# The milestones whose cases run.
DONE = ("SP1", "SP2", "SP3", "SP4", "SP5")


def _public(spec) -> bool:
    import stardust as sd

    return spec.milestone in DONE and all(n in sd.sparse.__all__ for n in spec.needs)


def cases() -> list[Case]:
    out = []
    for s in ARRAY_SPECS:
        if not _public(s):
            continue
        for nnz, size in NNZ:
            if s.limit and nnz > s.limit:
                continue
            refs = [Ref("scipy", lambda d, f=s.scipy: _quiet(lambda: f(d)))]
            if s.r:
                refs.append(
                    RRef(
                        "R",
                        s.r,
                        data={**R_DATA, "r_n": "n", "x": "x", **dict(s.r_data)},
                        setup=R_SETUP,
                        packages=("Matrix",),
                        differs=s.r_differs,
                        value=R_SUMMARY if s.milestone == "SP1" or s.r_sparse else "",
                    )
                )
            check = SPARSE_CHECK if s.milestone == "SP1" else SP2_CHECK
            if s.check == "size":
                check = SIZE_CHECK
            out.append(Case(area="sparse", group=s.group, name=s.name, size=size, n=nnz, setup=_array_setup(nnz),
                            sd=s.sd, refs=tuple(refs), quick=s.quick and nnz == 10**6, note=s.note, check=check))  # fmt: skip
    for s in SOLVER_SPECS:
        if not _public(s):
            continue
        for mid in s.grids:
            refs = [Ref("scipy", lambda d, f=s.scipy: _quiet(lambda: f(d)))]
            if s.r:
                refs.append(RRef("R", s.r, data={**R_DATA, "b": "b"}, setup=R_GRID_SETUP, packages=("Matrix",)))
            out.append(Case(area="sparse", group=s.group, name=f"{s.name} {mid.split('/')[0]}", size=_grid_size(mid),
                            setup=_grid_r(_grid_setup(mid)), sd=s.sd, refs=tuple(refs),
                            quick=s.quick and mid == "lap2d/300", note=s.note, check=s.check or Check()))  # fmt: skip
    return out


TITLES = {
    "convert": "형식 변환",
    "products": "곱 (SpMV, SpMM, SpGEMM)",
    "arithmetic": "산술",
    "reductions": "리덕션",
    "elementwise": "원소별 함수",
    "indexing": "인덱싱",
    "construct": "만들기",
    "norms": "노름",
    "direct": "직접 풀이",
    "iterative": "반복법",
    "eigen": "고유값",
    "matfuncs": "행렬 함수",
}
