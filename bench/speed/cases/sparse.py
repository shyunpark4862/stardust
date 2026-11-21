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
            "x": g.uniform(-1, 1, n), "X": g.uniform(-1, 1, (n, 16)), "XT": g.uniform(-1, 1, (16, n)),
            "a64": _int64(a), "pr": g.integers(0, n, max(1, n // 10)), "pc": g.integers(0, n, max(1, n // 10)),
            "r_i": (r + 1).astype(np.float64), "r_j": (c + 1).astype(np.float64), "r_x": v,
            "r_ci": (cr + 1).astype(np.float64), "r_cj": (cc + 1).astype(np.float64),
            "r_n": np.array([float(n)]),
        }  # fmt: skip

    return setup


def _int64(a):
    """A scipy CSR with int64 index arrays (scipy makes int32 ones when they fit)."""
    import scipy.sparse as sp

    b = sp.csr_array((a.data, a.indices.astype(np.int64), a.indptr.astype(np.int64)), shape=a.shape)
    assert b.indices.dtype == np.int64
    return b


def _skewed_setup(nnz):
    """A random n x n matrix (n = nnz / 10) whose first rows hold far more than the 4096 entries
    where a row's sum changes order (docs/api/sparse.md: products): 10 rows of nnz / 20 entries
    each, the rest about 5 a row; the same keys as _array_setup."""

    def setup():
        import scipy.sparse as sp

        g = np.random.Generator(np.random.PCG64(20261122 + nnz))
        n = nnz // 10
        long = nnz // 20
        rows = [np.full(long, i) for i in range(10)]
        cols = [g.choice(n, long, replace=False) for _ in range(10)]
        rest = nnz - 10 * long
        rows.append(g.integers(10, n, rest))
        cols.append(g.integers(0, n, rest))
        r, c = np.concatenate(rows), np.concatenate(cols)
        a = sp.csr_array((g.uniform(-1, 1, r.size), (r, c)), shape=(n, n))
        a.sum_duplicates()
        b = sp.csr_array((g.uniform(-1, 1, r.size), (c, r)), shape=(n, n))
        b.sum_duplicates()
        return {"n": n, "a": a, "b": b, "x": g.uniform(-1, 1, n), "X": g.uniform(-1, 1, (n, 16))}

    return setup


def _dense5_setup(nnz):
    """A random matrix with 5 % of its entries stored (n = sqrt(nnz / 0.05)); the same keys."""

    def setup():
        import scipy.sparse as sp

        n = int(round((nnz / 0.05) ** 0.5))
        a = sp.random_array((n, n), density=0.05, rng=nnz, format="csr")
        b = sp.random_array((n, n), density=0.05, rng=nnz + 1, format="csr")
        g = np.random.Generator(np.random.PCG64(nnz))
        return {"n": n, "a": a, "b": b, "x": g.uniform(-1, 1, n), "X": g.uniform(-1, 1, (n, 16))}

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


# exp(A) b: the values (the steps chosen from scipy's random estimates may differ by rounding).
EXPMV_CHECK = Check(rtol=1e-12)


# Eigenvalues: the sorted values (real parts; scipy returns (w, v)); singular values sorted.
def _eigvals(x):
    w = x.eigenvalues if hasattr(x, "eigenvalues") else (x[0] if isinstance(x, tuple) else x)
    return np.sort(np.asarray(w).real)


def _svals(x):
    s = x.S if hasattr(x, "S") else x[1]
    return np.sort(np.asarray(s))


EIG_CHECK = Check(sd=_eigvals, ref=lambda label, x: _eigvals(x), rtol=1e-8)
SVD_CHECK = Check(sd=_svals, ref=lambda label, x: np.sort(np.asarray(x)) if label == "R" else _svals(x), rtol=1e-8)


def _dense1(x):
    """A 1-D result, sparse (sd's A[rows, cols]) or dense (scipy's), as a dense array."""
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x)


PAIRS_CHECK = Check(sd=_dense1, ref=lambda label, x: _dense1(x), rtol=0.0)

# A norm: the value (2-norm: the singular value to the eigensolvers' tolerance).
NORM_CHECK = Check(rtol=1e-12)
# A lower-bound estimate with other random columns: the same size (both usually within a factor 3
# of the norm, Higham and Tisseur's experience: a gross difference fails).
ESTIMATE_CHECK = Check(sd=lambda r: r.est, rtol=0.7)  # (sd's result object: its estimate; SP102)

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
    tiny: bool = False  # also at nnz = 10^3 (n = 100): the fixed cost of a call
    matrix: str = ""  # "skewed" (rows past 4096 entries) or "dense5" (5 % stored) instead of the random set
    r_packages: tuple = ("Matrix",)
    r_setup: str = ""  # more R setup after the matrix A is built


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
    Spec("SP6", "eigen", "svds(k=6)", ("csr_array", "svds"),
         lambda d: (lambda A: lambda: _sds("svds")(A, k=6))(_sd_csr(d)),
         lambda d: lambda: _la("svds")(d["a"], k=6, rng=0), limit=10**6, check=SVD_CHECK,
         note="sd는 Golub-Kahan-Lanczos(두꺼운 재시작), scipy는 A^T A의 eigsh(ARPACK)"),
    Spec("SP7", "norms", "norm(A, 1)", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: _sds("norm")(A, ord=1))(_sd_csr(d)),
         lambda d: lambda: _la("norm")(d["a"], 1), r='norm(A, "1")', quick=True, check=NORM_CHECK),
    Spec("SP7", "norms", "norm(A, inf)", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: _sds("norm")(A, ord=np.inf))(_sd_csr(d)),
         lambda d: lambda: _la("norm")(d["a"], np.inf), r='norm(A, "I")', check=NORM_CHECK),
    Spec("SP7", "norms", "norm(A) Frobenius", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: _sds("norm")(A))(_sd_csr(d)),
         lambda d: lambda: _la("norm")(d["a"]), r='norm(A, "F")', check=NORM_CHECK),
    Spec("SP7", "norms", "norm(A, axis=1)", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: _sds("norm")(A, axis=1))(_sd_csr(d)),
         lambda d: lambda: _la("norm")(d["a"], axis=1), check=NORM_CHECK, note="행마다의 2-노름(밀집 결과)"),
    Spec("SP7", "norms", "norm(A, 2)", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: _sds("norm")(A, ord=2))(_sd_csr(d)),
         lambda d: lambda: _la("norm")(d["a"], 2), limit=10**6, check=NORM_CHECK,
         note="가장 큰 특이값: sd는 svds(GKL), scipy는 A^T A의 ARPACK"),
    Spec("SP7", "norms", "onenormest", ("csr_array", "onenormest"),
         lambda d: (lambda A: lambda: _sds("onenormest")(A))(_sd_csr(d)),
         lambda d: lambda: _la("onenormest")(d["a"]), check=ESTIMATE_CHECK,
         note="scipy는 시드 없는 무작위 열, sd는 고정 열(SP98): 추정값은 참값 이하에서 다를 수 있다"),
)  # fmt: skip


def _sp(name):
    return getattr(__import__("scipy.sparse", fromlist=[name]), name)


def _sdm(name):
    return getattr(__import__("stardust").sparse, name)


def _dok_fill(cls, n, r, c, v):
    """A DOK of shape (n, n) filled one entry at a time (the first 10^4 entries)."""
    D = cls((n, n))
    for i, j, x in zip(r, c, v):
        D[i, j] = x
    return D


# The rest of the API, the formats, sizes and matrices the cases above leave out (the review of
# 2026-11-22). R is the comparator where Matrix has the same operation, as above.
ARRAY_MORE = (
    Spec("SP1", "convert", "CSC to CSR", ("csc_array",), lambda d: (lambda A: lambda: A.tocsr())(_sd_csr(d).tocsc()),
         lambda d: lambda: d["csc"].tocsr()),
    Spec("SP1", "convert", "CSR to DOK", ("csr_array", "dok_array"),
         lambda d: (lambda A: lambda: A.todok())(_sd_csr(d)), lambda d: lambda: d["a"].todok(), limit=10**5,
         note="scipy의 DOK는 Python dict"),
    Spec("SP3", "construct", "dok_array: 10^4 entries set one at a time", ("dok_array",),
         lambda d: (lambda r, c, v: lambda: _dok_fill(_sdm("dok_array"), d["n"], r, c, v))(
             d["row"][:10**4].tolist(), d["col"][:10**4].tolist(), d["val"][:10**4].tolist()),
         lambda d: (lambda r, c, v: lambda: _dok_fill(_sp("dok_array"), d["n"], r, c, v))(
             d["row"][:10**4].tolist(), d["col"][:10**4].tolist(), d["val"][:10**4].tolist()),
         limit=10**5, note="원소 하나씩 D[i, j] = v(Python 반복)", check="size"),
    Spec("SP3", "construct", "diags_array(3 diagonals)", ("diags_array",),
         lambda d: (lambda x: lambda: _sdm("diags_array")([x[1:], x, x[1:]], offsets=[-1, 0, 1]))(_sd(d["x"])),
         lambda d: lambda: _sp("diags_array")([d["x"][1:], d["x"], d["x"][1:]], offsets=[-1, 0, 1]),
         r="bandSparse(n, k = c(-1, 0, 1), diagonals = list(x[-1], x, x[-1]))", r_sparse=True),
    Spec("SP3", "construct", "block_array([[A, B], [B, A]])", ("csr_array", "block_array"),
         lambda d: (lambda A, B: lambda: _sdm("block_array")([[A, B], [B, A]]))(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: _sp("block_array")([[d["a"], d["b"]], [d["b"], d["a"]]], format="csr"), limit=10**6,
         r="rbind(cbind(A, A), cbind(A, A))", r_sparse=True, r_differs="R: the session holds one matrix"),
    Spec("SP3", "construct", "block_diag(A, B)", ("csr_array", "block_diag"),
         lambda d: (lambda A, B: lambda: _sdm("block_diag")([A, B]))(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: _sp("block_diag")([d["a"], d["b"]], format="csr"),
         r="bdiag(A, A)", r_sparse=True, r_differs="R: the session holds one matrix"),
    Spec("SP3", "construct", "kronsum(A, eye(10))", ("csr_array", "eye_array", "kronsum"),
         lambda d: (lambda A: lambda: _sdm("kronsum")(A, _sdm("eye_array")(10)))(_sd_csr(d)),
         lambda d: lambda: _sp("kronsum")(d["a"], _sp("eye_array")(10), format="csr"), limit=10**6),
    Spec("SP3", "construct", "triu(A)", ("csr_array", "triu"),
         lambda d: (lambda A: lambda: _sdm("triu")(A))(_sd_csr(d)), lambda d: lambda: _sp("triu")(d["a"]),
         r="triu(A)", r_sparse=True),
    Spec("SP3", "construct", "find(A)", ("csr_array", "find"),
         lambda d: (lambda A: lambda: _sdm("find")(A))(_sd_csr(d)), lambda d: lambda: _sp("find")(d["a"]),
         check=Check(sd=lambda t: np.array([float(np.sum(np.asarray(t[2])))]),
                     ref=lambda label, t: np.array([float(np.sum(t[2]))]), rtol=1e-9)),
    Spec("SP3", "construct", "matrix_transpose(A)", ("csr_array", "matrix_transpose"),
         lambda d: (lambda A: lambda: _sdm("matrix_transpose")(A))(_sd_csr(d)),
         lambda d: lambda: _sp("matrix_transpose")(d["a"])),
    Spec("SP3", "construct", "is_sptriangular(A)", ("csr_array", "is_sptriangular"),
         lambda d: (lambda A: lambda: _sdm("is_sptriangular")(A))(_sd_csr(d)),
         lambda d: lambda: _la("is_sptriangular")(d["a"]), check=Check(rtol=0.0)),
    Spec("SP3", "construct", "spbandwidth(A)", ("csr_array", "spbandwidth"),
         lambda d: (lambda A: lambda: _sdm("spbandwidth")(A))(_sd_csr(d)),
         lambda d: lambda: _la("spbandwidth")(d["a"]), check=Check(rtol=0.0)),
    Spec("SP2", "arithmetic", "A - B", ("csr_array",), lambda d: (lambda A, B: lambda: A - B)(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: d["a"] - d["b"]),
    Spec("SP2", "arithmetic", "A * 2.0 (R)", ("csr_array",), lambda d: (lambda A: lambda: A * 2.0)(_sd_csr(d)),
         lambda d: lambda: d["a"] * 2.0, r="A * 2", r_sparse=True, note="R과 견주는 줄(위의 A * 2.0과 같은 일)"),
    Spec("SP2", "arithmetic", "A + X dense", ("csr_array",),
         lambda d: (lambda A, X: lambda: A + X)(_sd_csr(d), _sd(d["a"].toarray())),
         lambda d: (lambda X: lambda: d["a"] + X)(d["a"].toarray()), limit=10**5, note="결과는 밀집"),
    Spec("SP2", "arithmetic", "A.maximum(B)", ("csr_array",),
         lambda d: (lambda A, B: lambda: A.maximum(B))(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: d["a"].maximum(d["b"])),
    Spec("SP2", "elementwise", "abs(A)", ("csr_array",), lambda d: (lambda A: lambda: abs(A))(_sd_csr(d)),
         lambda d: lambda: abs(d["a"]), r="abs(A)", r_sparse=True),
    Spec("SP2", "elementwise", "A ** 2", ("csr_array",), lambda d: (lambda A: lambda: A**2)(_sd_csr(d)),
         lambda d: lambda: d["a"] ** 2, r="A^2", r_sparse=True, note="원소별 거듭제곱"),
    Spec("SP2", "elementwise", "astype(complex128)", ("csr_array",),
         lambda d: (lambda A: lambda: A.astype("complex128"))(_sd_csr(d)),
         lambda d: lambda: d["a"].astype(np.complex128),
         check=Check(sd=lambda x: np.abs(_summary(x)), ref=lambda label, x: np.abs(_summary(x)), rtol=1e-9)),
    Spec("SP2", "reductions", "min(axis=1)", ("csr_array",), lambda d: (lambda A: lambda: A.min(axis=1))(_sd_csr(d)),
         lambda d: lambda: d["a"].min(axis=1)),
    Spec("SP2", "reductions", "count_nonzero()", ("csr_array",),
         lambda d: (lambda A: lambda: A.count_nonzero())(_sd_csr(d)), lambda d: lambda: d["a"].count_nonzero(),
         check=Check(rtol=0.0)),
    Spec("SP2", "reductions", "nonzero()", ("csr_array",), lambda d: (lambda A: lambda: A.nonzero())(_sd_csr(d)),
         lambda d: lambda: d["a"].nonzero(), check=Check(sd=lambda t: np.array([float(len(t[0]))]),
                                                     ref=lambda label, t: np.array([float(len(t[0]))]), rtol=0.0)),
    Spec("SP2", "products", "x @ A (R)", ("csr_array",), lambda d: (lambda A, x: lambda: x @ A)(_sd_csr(d), _sd(d["x"])),
         lambda d: lambda: d["x"] @ d["a"], r="as.numeric(crossprod(x, A))", note="R과 견주는 줄"),
    Spec("SP2", "products", "A @ X (k = 16, R)", ("csr_array",),
         lambda d: (lambda A, X: lambda: A @ X)(_sd_csr(d), _sd(d["X"])), lambda d: lambda: d["a"] @ d["X"],
         r="as.matrix(A %*% X)", r_data=(("X", "X"),), note="R과 견주는 줄"),
    Spec("SP2", "products", "X dense (16 rows) @ A", ("csr_array",),
         lambda d: (lambda A, X: lambda: X @ A)(_sd_csr(d), _sd(d["XT"])), lambda d: lambda: d["XT"] @ d["a"]),
    Spec("SP2", "products", "COO @ x", ("coo_array",), lambda d: (lambda A, x: lambda: A @ x)(_sd_coo(d), _sd(d["x"])),
         lambda d: (lambda A: lambda: A @ d["x"])(d["a"].tocoo())),
    Spec("SP2", "products", "A @ x (scipy with int64 indices)", ("csr_array",),
         lambda d: (lambda A, x: lambda: A @ x)(_sd_csr(d), _sd(d["x"])), lambda d: lambda: d["a64"] @ d["x"],
         note="두 쪽 모두 int64 색인(위의 A @ x에서 scipy는 int32)"),
    Spec("SP2", "products", "CSC @ CSC", ("csc_array",),
         lambda d: (lambda A, B: lambda: A @ B)(_sd_csr(d).tocsc(), _sd_csr(d, "b").tocsc()),
         lambda d: (lambda A, B: lambda: A @ B)(d["a"].tocsc(), d["b"].tocsc()), limit=10**6),
    Spec("SP2", "products", "A @ x (rows past 4096 entries)", ("csr_array",),
         lambda d: (lambda A, x: lambda: A @ x)(_sd_csr(d), _sd(d["x"])), lambda d: lambda: d["a"] @ d["x"],
         matrix="skewed", note="10행이 nnz/20개씩, 나머지는 행마다 약 5개(행의 합 순서가 바뀌는 길이 넘음)"),
    Spec("SP2", "reductions", "sum(axis=1) (rows past 4096 entries)", ("csr_array",),
         lambda d: (lambda A: lambda: A.sum(axis=1))(_sd_csr(d)), lambda d: lambda: d["a"].sum(axis=1),
         matrix="skewed"),
    Spec("SP2", "products", "A @ x (5 % stored)", ("csr_array",),
         lambda d: (lambda A, x: lambda: A @ x)(_sd_csr(d), _sd(d["x"])), lambda d: lambda: d["a"] @ d["x"],
         matrix="dense5", note="n = sqrt(nnz / 0.05)"),
    Spec("SP2", "products", "A @ B (5 % stored)", ("csr_array",),
         lambda d: (lambda A, B: lambda: A @ B)(_sd_csr(d), _sd_csr(d, "b")), lambda d: lambda: d["a"] @ d["b"],
         matrix="dense5", limit=10**6),
    Spec("SP3", "indexing", "A[i, j] scalar (10^4 reads)", ("csr_array",),
         lambda d: (lambda A, r, c: lambda: [A[i, j] for i, j in zip(r, c)])(
             _sd_csr(d), d["pr"][:10**4].tolist(), d["pc"][:10**4].tolist()),
         lambda d: (lambda r, c: lambda: [d["a"][i, j] for i, j in zip(r, c)])(
             d["pr"][:10**4].tolist(), d["pc"][:10**4].tolist()),
         limit=10**6, note="Python 반복에서 원소 하나씩", check=Check(sd=lambda v: np.array([float(np.sum(v))]),
                                                                     ref=lambda label, v: np.array([float(np.sum(v))]),
                                                                     rtol=1e-9)),
    Spec("SP3", "indexing", "A[rows, cols] (n/10 pairs)", ("csr_array",),
         lambda d: (lambda A, r, c: lambda: A[r, c])(_sd_csr(d), _sd(d["pr"]), _sd(d["pc"])),
         lambda d: lambda: d["a"][d["pr"], d["pc"]], check=PAIRS_CHECK,
         note="sd는 1차원 COO(SP54), scipy는 밀집 배열: 값은 toarray()로 견준다"),
    Spec("SP3", "indexing", "A[rows 1/10] (R)", ("csr_array",),
         lambda d: (lambda A, k: lambda: A[:k])(_sd_csr(d), d["n"] // 10), lambda d: lambda: d["a"][: d["n"] // 10],
         r="A[seq_len(n %/% 10), , drop = FALSE]", r_sparse=True, note="R과 견주는 줄"),
    Spec("SP3", "indexing", "diagonal() (R)", ("csr_array",), lambda d: (lambda A: lambda: A.diagonal())(_sd_csr(d)),
         lambda d: lambda: d["a"].diagonal(), r="diag(A)", note="R과 견주는 줄"),
    Spec("SP3", "construct", "vstack (R)", ("csr_array", "vstack"),
         lambda d: (lambda A, B: lambda: _sdm("vstack")([A, B]))(_sd_csr(d), _sd_csr(d, "b")),
         lambda d: lambda: _sp("vstack")([d["a"], d["b"]]), r="rbind(A, A)", r_sparse=True,
         r_differs="R: the session holds one matrix", note="R과 견주는 줄"),
    Spec("SP7", "norms", "norm(A, -1)", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: _sds("norm")(A, ord=-1))(_sd_csr(d)), lambda d: lambda: _la("norm")(d["a"], -1),
         check=NORM_CHECK),
    Spec("SP7", "norms", "norm(A, -inf)", ("csr_array", "norm"),
         lambda d: (lambda A: lambda: _sds("norm")(A, ord=-np.inf))(_sd_csr(d)),
         lambda d: lambda: _la("norm")(d["a"], -np.inf), check=NORM_CHECK),
    Spec("SP6", "eigen", "svds(k=6) (R)", ("csr_array", "svds"),
         lambda d: (lambda A: lambda: _sds("svds")(A, k=6))(_sd_csr(d)),
         lambda d: lambda: _la("svds")(d["a"], k=6, rng=0), limit=10**6, check=SVD_CHECK,
         r="RSpectra::svds(A, 6)$d", r_packages=("Matrix", "RSpectra"),
         note="R은 RSpectra(정확도 측정과 같다)"),
    Spec("SP6", "eigen", "svds(k=6) (R irlba)", ("csr_array", "svds"),
         lambda d: (lambda A: lambda: _sds("svds")(A, k=6))(_sd_csr(d)),
         lambda d: lambda: _la("svds")(d["a"], k=6, rng=0), limit=10**6, check=SVD_CHECK,
         r="irlba::irlba(A, nv = 6, tol = 1e-12, maxit = 10000, v = sin(seq_len(ncol(A))))$d", r_packages=("Matrix", "irlba"),
         note="R은 irlba(sd의 svds와 같은 방법, 정확도 측정과 같다)"),  # fmt: skip
)  # fmt: skip
# the tiny size (nnz = 10^3, n = 100) of a few cases: the fixed cost of a call
TINY = {"A @ x", "A + B", "sum(axis=0)", "A[rows 1/10]", "COO to CSR", "norm(A, 1)", "A @ B"}


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
         lambda d: (lambda A, v: lambda: _sds("eigsh")(A, k=6, v0=v))(_sd_csc(d), _sd(np.ones(d["n"]))),
         lambda d: lambda: _la("eigsh")(d["a"], k=6, v0=np.ones(d["n"])), check=EIG_CHECK,
         grids=("lap2d/100", "lap3d/22"), note="같은 시작 벡터(1의 벡터): 같은 ARPACK, 같은 OP 횟수"),
    Spec("SP6", "eigen", "eigsh(k=6, sigma=0)", ("csc_array", "eigsh"),
         lambda d: (lambda A, v: lambda: _sds("eigsh")(A, k=6, sigma=0.0, v0=v))(_sd_csc(d), _sd(np.ones(d["n"]))),
         lambda d: lambda: _la("eigsh")(d["a"], k=6, sigma=0.0, v0=np.ones(d["n"])), check=EIG_CHECK,
         grids=("lap2d/100", "lap2d/300", "lap3d/22"), quick=True,
         note="shift-invert: sd는 Accelerate LU/AMD, scipy는 SuperLU"),
    Spec("SP6", "eigen", "eigs(k=6)", ("csc_array", "eigs"),
         lambda d: (lambda A, v: lambda: _sds("eigs")(A, k=6, v0=v))(_sd_csc(d), _sd(np.ones(d["n"]))),
         lambda d: lambda: _la("eigs")(d["a"], k=6, v0=np.ones(d["n"])), check=EIG_CHECK,
         grids=("lap2d/100", "lap3d/22"), note="같은 시작 벡터(1의 벡터)"),
    Spec("SP7", "matfuncs", "expm_multiply(-0.1 A, b)", ("csc_array", "expm_multiply"),
         lambda d: (lambda A, b: lambda: _sds("expm_multiply")(A, b))(_sd_csc(d, -0.1), _sd(d["b"])),
         lambda d: (lambda A: lambda: _la("expm_multiply")(A, d["b"]))(-0.1 * d["a"]), quick=True,
         check=EXPMV_CHECK),
    Spec("SP7", "matfuncs", "expm_multiply(-0.1 A, b, 0..1, 11 points)", ("csc_array", "expm_multiply"),
         lambda d: (lambda A, b: lambda: _sds("expm_multiply")(A, b, start=0.0, stop=1.0, num=11))(
             _sd_csc(d, -0.1), _sd(d["b"])),
         lambda d: (lambda A: lambda: _la("expm_multiply")(A, d["b"], start=0.0, stop=1.0, num=11))(-0.1 * d["a"]),
         check=EXPMV_CHECK, grids=GRIDS[:2] + GRIDS[3:]),
    Spec("SP7", "matfuncs", "expm(0.1 A)", ("csc_array", "expm"),
         lambda d: (lambda A: lambda: _sds("expm")(A))(_sd_csc(d, 0.1)),
         lambda d: (lambda A: lambda: _la("expm")(A))(0.1 * d["a"]), check=SPARSE_CHECK, grids=("lap2d/30", "lap3d/10"),
         note="결과는 거의 밀집(n² 원소): 작은 격자만"),
)  # fmt: skip


# R's direct solve with Matrix's cache cleared inside the timed call (B@factors <- list(): a
# factorization each call, as sd and scipy do), on the grids where CSparse's LU takes seconds,
# not minutes (lap2d/1000 and lap3d/45 took about 11 minutes and 6.5 GB, SP4).
R_SOLVE = "{ B <- A; B@factors <- list(); as.numeric(solve(B, b)) }"
SMALL_GRIDS = ("lap2d/100", "lap2d/300", "lap3d/22")
# The start vector of the rows compared with R: not a constant one. A grid Laplacian's eigenvalues
# come in pairs (lambda_i + lambda_j), and a start vector symmetric in the two axes spans only one
# of each pair: RSpectra then returns the wrong six (ARPACK's restarts recover them by rounding).
R_EIG_OPTS = "opts = list(initvec = sin(seq_along(b)))"


def _v0(d):
    return np.sin(np.arange(1, d["n"] + 1, dtype=float))


SOLVER_MORE = (
    Spec("SP4", "direct", "spsolve (R)", ("csc_array", "spsolve"),
         lambda d: (lambda A, b: lambda: _sds("spsolve")(A, b))(_sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("spsolve")(d["a"], d["b"]), r=R_SOLVE, grids=SMALL_GRIDS,
         note="R은 CSparse LU, 분해 캐시를 지우고 매번 분해한다"),
    Spec("SP4", "direct", "spsolve, 16 right sides", ("csc_array", "spsolve"),
         lambda d: (lambda A, B: lambda: _sds("spsolve")(A, B))(_sd_csc(d), _sd(_rhs16(d))),
         lambda d: (lambda B: lambda: _la("spsolve")(d["a"], B))(_rhs16(d)), grids=SMALL_GRIDS),
    Spec("SP4", "direct", "splu solve(trans='T')", ("csc_array", "splu"),
         lambda d: (lambda f, b: lambda: f.solve(b, trans="T"))(_sds("splu")(_sd_csc(d)), _sd(d["b"])),
         lambda d: (lambda f: lambda: f.solve(d["b"], trans="T"))(_quiet(lambda: _la("splu")(d["a"])))),
    Spec("SP4", "direct", "spsolve_triangular (R)", ("csr_array", "tril", "spsolve_triangular"),
         lambda d: (lambda L, b: lambda: _sds("spsolve_triangular")(L, b, upper=False))(
             _sds("tril")(_sd_csc(d), format="csr"), _sd(d["b"])),
         lambda d: (lambda L: lambda: _la("spsolve_triangular")(L, d["b"]))(_sp("tril")(d["a"], format="csr")),
         r="as.numeric(solve(L, b))", r_setup="L <- tril(A)", note="R은 삼각 행렬(dtCMatrix)의 solve"),
    Spec("SP4", "direct", "factorized(A)", ("csc_array", "factorized"),
         lambda d: (lambda A: lambda: _sds("factorized")(A))(_sd_csc(d)), lambda d: lambda: _la("factorized")(d["a"]),
         check=Check(skip="a solve function"), grids=SMALL_GRIDS),
    Spec("SP4", "direct", "inv(A)", ("csc_array", "inv"), lambda d: (lambda A: lambda: _sds("inv")(A))(_sd_csc(d)),
         lambda d: lambda: _la("inv")(d["a"]), check=SPARSE_CHECK, grids=("lap2d/30", "lap3d/10"),
         note="결과는 밀집에 가깝다: 작은 격자만"),
    Spec("SP5", "iterative", "cg with a LinearOperator (Python matvec)", ("csc_array", "cg", "LinearOperator"),
         lambda d: (lambda A, b: lambda: _sds("cg")(
             _sdm("LinearOperator")(A.shape, lambda v: A @ v), b, rtol=1e-10, max_iter=20000).x)(
             _sd_csc(d), _sd(d["b"])),
         lambda d: lambda: _la("cg")(_la("LinearOperator")(d["a"].shape, matvec=lambda v: d["a"] @ v), d["b"],
                                     rtol=1e-10, maxiter=20000)[0],
         check=ITER_CHECK, grids=("lap2d/100", "lap2d/300"), note="반복마다 Python 호출 하나"),
    Spec("SP6", "eigen", "eigsh(k=6) (R)", ("csc_array", "eigsh"),
         lambda d: (lambda A, v: lambda: _sds("eigsh")(A, k=6, v0=v))(_sd_csc(d), _sd(_v0(d))),
         lambda d: lambda: _la("eigsh")(d["a"], k=6, v0=_v0(d)), check=EIG_CHECK,
         r=f"RSpectra::eigs_sym(A, 6, which = 'LM', {R_EIG_OPTS})$values", r_packages=("Matrix", "RSpectra"),
         grids=("lap2d/100", "lap3d/22"), note="R은 RSpectra(정확도 측정과 같다)"),
    Spec("SP6", "eigen", "eigsh(k=6, sigma=0) (R)", ("csc_array", "eigsh"),
         lambda d: (lambda A, v: lambda: _sds("eigsh")(A, k=6, sigma=0.0, v0=v))(_sd_csc(d), _sd(_v0(d))),
         lambda d: lambda: _la("eigsh")(d["a"], k=6, sigma=0.0, v0=_v0(d)), check=EIG_CHECK,
         r=f"RSpectra::eigs_sym(A, 6, sigma = 0, {R_EIG_OPTS})$values", r_packages=("Matrix", "RSpectra"),
         grids=SMALL_GRIDS, note="R은 RSpectra의 shift-invert(Eigen의 희소 LU)"),
    Spec("SP6", "eigen", "eigsh(k=6, which='LA')", ("csc_array", "eigsh"),
         lambda d: (lambda A, v: lambda: _sds("eigsh")(A, k=6, which="LA", v0=v))(_sd_csc(d), _sd(np.ones(d["n"]))),
         lambda d: lambda: _la("eigsh")(d["a"], k=6, which="LA", v0=np.ones(d["n"])), check=EIG_CHECK,
         grids=("lap2d/100",)),
    Spec("SP6", "eigen", "eigs(k=6) (R)", ("csc_array", "eigs"),
         lambda d: (lambda A, v: lambda: _sds("eigs")(A, k=6, v0=v))(_sd_csc(d), _sd(_v0(d))),
         lambda d: lambda: _la("eigs")(d["a"], k=6, v0=_v0(d)), check=EIG_CHECK,
         r=f"Re(RSpectra::eigs(A, 6, which = 'LM', {R_EIG_OPTS})$values)", r_packages=("Matrix", "RSpectra"),
         grids=("lap2d/100", "lap3d/22"), note="R은 RSpectra"),
    Spec("SP6", "eigen", "eigs(k=6, sigma=0)", ("csc_array", "eigs"),
         lambda d: (lambda A, v: lambda: _sds("eigs")(A, k=6, sigma=0.0, v0=v))(_sd_csc(d), _sd(np.ones(d["n"]))),
         lambda d: lambda: _la("eigs")(d["a"], k=6, sigma=0.0, v0=np.ones(d["n"])), check=EIG_CHECK,
         grids=("lap2d/100", "lap3d/22")),
    Spec("SP7", "matfuncs", "expm_multiply(-0.1 A, B (n, 4))", ("csc_array", "expm_multiply"),
         lambda d: (lambda A, B: lambda: _sds("expm_multiply")(A, B))(_sd_csc(d, -0.1), _sd(_rhs16(d)[:, :4])),
         lambda d: (lambda A, B: lambda: _la("expm_multiply")(A, B))(-0.1 * d["a"], _rhs16(d)[:, :4]),
         check=EXPMV_CHECK, grids=SMALL_GRIDS),
    Spec("SP7", "matfuncs", "expm(0.1 A) (R)", ("csc_array", "expm"),
         lambda d: (lambda A: lambda: _sds("expm")(A))(_sd_csc(d, 0.1)),
         lambda d: (lambda A: lambda: _la("expm")(A))(0.1 * d["a"]), check=SPARSE_CHECK, grids=("lap2d/30", "lap3d/10"),
         r="{ M <- as.matrix(Matrix::expm(0.1 * A)); c(sum(M), sum(M * row(M)), sum(M * col(M))) }",
         note="R은 Matrix::expm(정확도 측정과 같다)"),
)  # fmt: skip


def _rhs16(d):
    n = d["n"]
    return np.sin(0.37 * np.arange(n)[:, None] * (1 + np.arange(16))[None, :]) + 0.1


# The SuiteSparse matrices (tools/sparse_suitesparse.py; build/sparse/suitesparse, outside version
# control): the case runs where the matrix is fetched. (matrix, what is timed)
REAL = (
    ("Boeing/msc10848", ("A @ x", "spsolve", "splu", "cg", "eigsh")),
    ("Goodwin/goodwin", ("A @ x", "spsolve", "splu", "gmres")),
    ("HB/bcsstk14", ("spsolve", "cg", "eigsh")),
    ("Hamm/memplus", ("spsolve", "gmres", "eigs")),
    ("GHS_indef/cvxqp3", ("spsolve (LinAlgError)", "minres")),
    ("HB/sherman3", ("spsolve (LinAlgError)", "gmres (Jacobi)")),
    ("HB/well1850", ("lsqr", "lsmr", "svds")),
)


def _real_setup(name):
    def setup():
        import scipy.sparse as sp

        from tools import sparse_suitesparse as S

        shape, r, c, v = S.load(name)
        a = sp.csc_array((v, (r, c)), shape=shape)
        a.sum_duplicates()
        m, n = shape
        return {"a": a, "n": n, "m": m, "b": np.sin(0.37 * np.arange(m)) + 0.1, "x": np.sin(0.37 * np.arange(n)) + 0.1}

    return setup


def _real_calls(what):
    """(sd factory, scipy factory, check) of one timed call on a real matrix."""
    v0 = lambda d: np.ones(d["n"])  # noqa: E731
    calls = {
        "A @ x": (lambda d: (lambda A, x: lambda: A @ x)(_sd_csc(d), _sd(d["x"])), lambda d: lambda: d["a"] @ d["x"],
                  Check(rtol=1e-12)),
        "spsolve": (lambda d: (lambda A, b: lambda: _sds("spsolve")(A, b))(_sd_csc(d), _sd(d["b"])),
                    lambda d: lambda: _la("spsolve")(d["a"], d["b"]), Check(rtol=1e-6, agree=0.9)),
        # kappa_1 * eps >= 1 (sherman3 6.9e16, cvxqp3 2.2e16): sd treats the matrix as singular and
        # raises (SP66), scipy returns a solution with no correct digits. The sd side times the
        # factorization that finds it out; the check is that sd raised.
        "spsolve (LinAlgError)": (lambda d: (lambda A, b: lambda: _raises_linalg(lambda: _sds("spsolve")(A, b)))(
            _sd_csc(d), _sd(d["b"])), lambda d: lambda: _la("spsolve")(d["a"], d["b"]),
            Check(sd=lambda v: v, ref=lambda label, x: np.array([1.0]), rtol=0.0)),
        "splu": (lambda d: (lambda A: lambda: _sds("splu")(A))(_sd_csc(d)), lambda d: lambda: _la("splu")(d["a"]),
                 FACTOR_CHECK),
        "cg": (lambda d: (lambda A, b: lambda: _sds("cg")(A, b, rtol=1e-8, max_iter=20000).x)(_sd_csc(d), _sd(d["b"])),
               lambda d: lambda: _la("cg")(d["a"], d["b"], rtol=1e-8, maxiter=20000)[0], ITER_CHECK),
        "minres": (lambda d: (lambda A, b: lambda: _sds("minres")(A, b, rtol=1e-8, max_iter=20000).x)(
            _sd_csc(d), _sd(d["b"])), lambda d: lambda: _la("minres")(d["a"], d["b"], rtol=1e-8, maxiter=20000)[0],
            ITER_CHECK),
        "gmres": (lambda d: (lambda A, b: lambda: _sds("gmres")(A, b, rtol=1e-8, max_iter=2000).x)(
            _sd_csc(d), _sd(d["b"])), lambda d: lambda: _la("gmres")(d["a"], d["b"], rtol=1e-8, maxiter=2000)[0],
            ITER_CHECK),
        # sherman3 stalls without a preconditioner (both stop at 2,000 restarts, residual 0.65)
        "gmres (Jacobi)": (lambda d: (lambda A, b, M: lambda: _sds("gmres")(A, b, rtol=1e-8, max_iter=2000, M=M).x)(
            _sd_csc(d), _sd(d["b"]), _sd_csc({"a": _jacobi(d)})),
            lambda d: (lambda M: lambda: _la("gmres")(d["a"], d["b"], rtol=1e-8, maxiter=2000, M=M)[0])(_jacobi(d)),
            ITER_CHECK),
        "lsqr": (lambda d: (lambda A, b: lambda: _sds("lsqr")(A, b, atol=1e-10, btol=1e-10, max_iter=20000).x)(
            _sd_csc(d), _sd(d["b"])),
            lambda d: lambda: _la("lsqr")(d["a"], d["b"], atol=1e-10, btol=1e-10, iter_lim=20000)[0], ITER_CHECK),
        "lsmr": (lambda d: (lambda A, b: lambda: _sds("lsmr")(A, b, atol=1e-10, btol=1e-10, max_iter=20000).x)(
            _sd_csc(d), _sd(d["b"])),
            lambda d: lambda: _la("lsmr")(d["a"], d["b"], atol=1e-10, btol=1e-10, maxiter=20000)[0], ITER_CHECK),
        "eigsh": (lambda d: (lambda A, v: lambda: _sds("eigsh")(A, k=6, v0=v))(_sd_csc(d), _sd(v0(d))),
                  lambda d: lambda: _la("eigsh")(d["a"], k=6, v0=v0(d)), EIG_CHECK),
        "eigs": (lambda d: (lambda A, v: lambda: _sds("eigs")(A, k=6, v0=v))(_sd_csc(d), _sd(v0(d))),
                 lambda d: lambda: _la("eigs")(d["a"], k=6, v0=v0(d)), EIG_CHECK),
        "svds": (lambda d: (lambda A: lambda: _sds("svds")(A, k=6))(_sd_csc(d)),
                 lambda d: lambda: _la("svds")(d["a"], k=6, rng=0), SVD_CHECK),
    }  # fmt: skip
    return calls[what]


def _raises_linalg(f):
    """1.0 when f raises stardust's LinAlgError, 0.0 when it returns."""
    import stardust as sd

    try:
        f()
    except sd.linalg.LinAlgError:
        return np.array([1.0])
    return np.array([0.0])


def _jacobi(d):
    """The inverse of A's diagonal as a CSC matrix: the Jacobi preconditioner."""
    import scipy.sparse as sp

    return sp.diags_array(1.0 / d["a"].diagonal()).tocsc()


def _real_cases():
    from tools import sparse_suitesparse as S

    out = []
    for name, whats in REAL:
        if not S.available(name):
            continue
        rows, cols, nnz = next((m[2], m[3], m[4]) for m in S.MATRICES if m[0] == name)
        for what in whats:
            sdf, spf, check = _real_calls(what)
            out.append(Case(area="sparse", group="real", name=f"{what} {name.split('/')[1]}", size=f"({rows}, {cols})",
                            n=nnz, setup=_real_setup(name), sd=sdf, refs=(Ref("scipy", lambda d, f=spf: _quiet(lambda: f(d))),),
                            note=f"SuiteSparse {name}", check=check))  # fmt: skip
    return out


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
DONE = ("SP1", "SP2", "SP3", "SP4", "SP5", "SP6", "SP7")


def _public(spec) -> bool:
    import stardust as sd

    return spec.milestone in DONE and all(n in sd.sparse.__all__ for n in spec.needs)


def cases() -> list[Case]:
    out = []
    setups = {"": _array_setup, "skewed": _skewed_setup, "dense5": _dense5_setup}
    for s in ARRAY_SPECS + ARRAY_MORE:
        if not _public(s):
            continue
        sizes = (((10**3, "10^3"),) if s.name in TINY and s.matrix == "" else ()) + NNZ
        for nnz, size in sizes:
            if s.limit and nnz > s.limit:
                continue
            if s.matrix and nnz > 10**6:  # the skewed and 5 % matrices up to 10^6 stored entries
                continue
            refs = [Ref("scipy", lambda d, f=s.scipy: _quiet(lambda: f(d)))]
            if s.r and not s.matrix:
                refs.append(
                    RRef(
                        "R",
                        s.r,
                        data={**R_DATA, "r_n": "n", "x": "x", **dict(s.r_data)},
                        setup=R_SETUP + ("; " + s.r_setup if s.r_setup else ""),
                        packages=s.r_packages,
                        differs=s.r_differs,
                        value=R_SUMMARY if s.milestone == "SP1" or s.r_sparse else "",
                    )
                )
            check = SPARSE_CHECK if s.milestone == "SP1" else SP2_CHECK
            if s.check == "size":
                check = SIZE_CHECK
            elif isinstance(s.check, Check):
                check = s.check
            out.append(Case(area="sparse", group=s.group, name=s.name, size=size, n=nnz, setup=setups[s.matrix](nnz),
                            sd=s.sd, refs=tuple(refs), quick=s.quick and nnz == 10**6, note=s.note, check=check))  # fmt: skip
    for s in SOLVER_SPECS + SOLVER_MORE:
        if not _public(s):
            continue
        for mid in s.grids:
            refs = [Ref("scipy", lambda d, f=s.scipy: _quiet(lambda: f(d)))]
            if s.r:
                setup = R_GRID_SETUP + ("; " + s.r_setup if s.r_setup else "")
                refs.append(RRef("R", s.r, data={**R_DATA, "b": "b"}, setup=setup, packages=s.r_packages))
            out.append(Case(area="sparse", group=s.group, name=f"{s.name} {mid.split('/')[0]}", size=_grid_size(mid),
                            setup=_grid_r(_grid_setup(mid)), sd=s.sd, refs=tuple(refs),
                            quick=s.quick and mid == "lap2d/300", note=s.note, check=s.check or Check()))  # fmt: skip
    out += _real_cases()
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
    "real": "실제 행렬 (SuiteSparse)",
}
