"""sd.linalg against numpy and scipy. The base set is the old baseline's (bench/linalg/ (at commit 4151d2a)
baseline.py matrix_rows and long_rows, timed by bench/linalg/stage_compare.py, summed up
by bench/linalg/final_table.py, rerun after the merge through bench/merge/rerun.sh):
every function on single small and large matrices, batches, long vectors and long
banded systems. On top of it: rectangular matrices, several right-hand sides, batches
near the split threshold (linalg.md §7), complex matrices, batched banded systems, and
inputs that take sd's own paths (a triangular or small-norm expm, the generalized
eigvalsh, cond with ord=1).

The comparator is numpy where numpy has the function, else scipy, at their default
threads (their BLAS on its default pool), as stage_compare.py --ref did. scipy 1.18
takes batches for lu_factor / lu_solve, ldl, cho_solve, solve_banded and lstsq, so every
row has a comparator. Every comparator does sd's work: numpy's full_matrices and mode
follow sd's mode, scipy's cho_solve gets cho_factor's factor as sd gets its cholesky.
Every call is a functools.partial over the prepared arrays where the call allows it, so
the timed callable adds no Python frame of its own on either side.

R joins where the accuracy measurement compares with R (bench/accuracy/linalg_ops.py, _R)
and R's call does sd's work (R_ROWS below): single real matrices only (R has no batched
or, on the accuracy side, complex calls), with R's matrices built from the numpy arrays
before the timing; and vecdot(v, v) of the long vectors (the accuracy side's vecdot is
the array area's, of two vectors: R's crossprod). Not qr (R's LAPACK qr pivots the columns: it is qr_pivoted's work;
its default LINPACK qr is another algorithm), not lstsq (R's qr.coef solves by pivoted
QR; sd's lstsq also returns the singular values and the rank), not the generalized
eigh, not svd(mode="complete") (the accuracy side calls svd(A) only).

A single small matrix has nothing to split (the batch split is the only parallel path
below the BLAS), so (3,3) and (32,32) run at one thread only, except solve_sylvester and
two rows that catch a misuse of the pool at 10 threads."""

from __future__ import annotations

import functools
import importlib

from ..registry import Case, Ref, RRef

MATRIX_SHAPES = [(3, 3), (32, 32), (300, 300), (1000, 1000), (1000, 3, 3), (100, 32, 32), (16, 3, 3), (8, 300, 300)]
KIND = {
    (3, 3): "small",
    (32, 32): "small",
    (300, 300): "large",
    (1000, 1000): "large",
    (1000, 3, 3): "batched",
    (100, 32, 32): "batched",
    (16, 3, 3): "batched",
    (8, 300, 300): "batched",
}
COMPLEX_GROUP = "complex"
COMPLEX_SHAPES = [(300, 300), (100, 32, 32)]
RECT_GROUP = "rectangular"
LONG_GROUP = "vectors-banded"
LONG_N = 1_000_000
BAND_N = 100_000
BAND_BATCH = (100, 128)  # batched banded systems: 100 systems of 128 unknowns

# Single small matrices that still run at 10 threads: the pool must stay out of the way.
POOL_CHECK = {("solve(a, b)", (3, 3)), ("svd(a)", (32, 32))}


def _resolve(path: str):
    """'sd.linalg.solve', 'numpy.linalg.qr', 'scipy.linalg.lu', ... -> the function."""
    head, *rest = path.split(".")
    obj = importlib.import_module("stardust" if head == "sd" else head)
    for i, part in enumerate(rest):
        try:
            obj = getattr(obj, part)
        except AttributeError:  # a submodule numpy / scipy load lazily
            obj = importlib.import_module(".".join(["stardust" if head == "sd" else head, *rest[: i + 1]]))
    return obj


class Call:
    """path(*args, **kw), timed as a functools.partial; a str arg names a data key, any
    other arg is passed as it is."""

    def __init__(self, path, *keys, **kw):
        self.path, self.keys, self.kw = path, keys, kw

    def make(self, d):
        return functools.partial(
            _resolve(self.path), *(d[k] if isinstance(k, str) else k for k in self.keys), **self.kw
        )


class Expr:
    """fn(*data[keys]) returns the callable to time (for a ref that is not one call)."""

    def __init__(self, keys, fn):
        self.keys, self.fn = tuple(keys.split()), fn

    def make(self, d):
        return self.fn(*(d[k] for k in self.keys))


def S(name, *keys, **kw):  # an sd.linalg call
    return Call(f"sd.linalg.{name}", *keys, **kw)


def N(path, *keys, **kw):  # a numpy call
    return ("numpy", Call(f"numpy.{path}", *keys, **kw))


def P(path, *keys, **kw):  # a scipy.linalg call
    return ("scipy", Call(f"scipy.linalg.{path}", *keys, **kw))


# ---------------------------------------------------------------- data

# numpy keys that have an sd copy under the upper-case name.
_SD_COPIES = (
    "a",
    "b",
    "b16",
    "c",
    "h",
    "spd",
    "t",
    "tf",
    "ut",
    "a001",
    "x0",
    "dl",
    "dd",
    "du",
    "ab5",
    "ab_tri",
    "k3",
    "v3",
    "w3",
)


def _matrix_data(shape, keys, quiet=False, cplx=False):
    """The inputs of baseline.py's matrix_rows at one shape (m, n or batch, m, n); only
    `keys` (and what they need) are built. Lower case: numpy, upper case: the sd copy.
    cplx: complex128 a, b, c and what is built from them (h Hermitian, spd = a·aᴴ + n·I).

    quiet: numpy's det and slogdet raise spurious floating-point warnings on Accelerate
    (baseline.py silenced them with np.seterr for the whole run); each warning costs
    numpy a trip through the warnings module per call. The registry has no teardown,
    so every other linalg setup puts numpy's default error state back."""
    import numpy as np

    import stardust as sd

    if quiet:
        np.seterr(all="ignore")
    else:
        np.seterr(divide="warn", over="warn", under="ignore", invalid="warn")

    m, n = shape[-2], shape[-1]
    batch = shape[:-2]
    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape)
    b = rng.standard_normal((*batch, m, 1))
    c = rng.standard_normal(shape)
    tri = [
        rng.standard_normal((*batch, n - 1)),
        rng.standard_normal((*batch, n)) + 10,
        rng.standard_normal((*batch, n - 1)),
    ]
    ab5 = rng.standard_normal((*batch, 5, n))
    ab5[..., 2, :] += 10  # kl = ku = 2, a dominant diagonal
    k3 = rng.standard_normal((3, 3))  # kron's second operand: (..., 3n, 3n) results
    if cplx:
        a = a + 1j * rng.standard_normal(a.shape)
        b = b + 1j * rng.standard_normal(b.shape)
        c = c + 1j * rng.standard_normal(c.shape)
    memo = {"a": a, "b": b, "c": c, "dl": tri[0], "dd": tri[1], "du": tri[2], "ab5": ab5, "k3": k3}

    def ct(x):
        return np.swapaxes(x, -1, -2).conj()

    def get(k):
        if k in memo:
            return memo[k]
        if k == "h":
            v = (a + ct(a)) / 2
        elif k == "spd":
            v = a @ ct(a) + n * np.eye(n)
        elif k == "t":
            v = np.triu(a) + n * np.eye(n)
        elif k == "tf":
            v = np.asfortranarray(get("t"))
        elif k == "ut":
            v = np.triu(a)  # expm of a triangular matrix (no shift: exp(n) would overflow)
        elif k == "a001":
            v = 0.01 * a
        elif k == "b16":
            v = np.random.default_rng(1).standard_normal((*batch, m, 16))
        elif k == "cho_f":
            import scipy.linalg as sl

            v = sl.cho_factor(get("spd"), lower=True)
        elif k == "CHOL":
            v = sd.linalg.cholesky(get("SPD"))  # sd's own output (column-major per matrix)
        elif k == "x0":
            v = a[..., 0, :]
        elif k == "v3":
            v = a[..., :3]  # cross: the rows' first three entries
        elif k == "w3":
            v = c[..., :3]
        elif k == "ab_tri":
            v = np.zeros((*batch, 3, n))
            v[..., 0, 1:], v[..., 1, :], v[..., 2, :-1] = tri[2], tri[1], tri[0]
        elif k == "lu_f":
            import scipy.linalg as sl

            v = sl.lu_factor(a)
        elif k in ("LU_L", "LU_U", "LU_P"):
            r = sd.linalg.lu(get("A"))
            memo["LU_L"], memo["LU_U"], memo["LU_P"] = r
            return memo[k]
        elif k[0].isupper() and k.lower() in _SD_COPIES:
            v = sd.asarray(get(k.lower()))
        else:
            raise KeyError(k)
        memo[k] = v
        return v

    return {k: get(k) for k in keys}


def _long_data(keys):
    """baseline.py's long_rows inputs, and the batched banded systems."""
    import numpy as np

    import stardust as sd

    np.seterr(divide="warn", over="warn", under="ignore", invalid="warn")  # see _matrix_data

    rng = np.random.default_rng(0)
    v = rng.standard_normal(LONG_N)
    n = BAND_N
    tri = [rng.standard_normal(n - 1), rng.standard_normal(n) + 10, rng.standard_normal(n - 1)]
    ab_tri = np.zeros((3, n))
    ab_tri[0, 1:], ab_tri[1], ab_tri[2, :-1] = tri[2], tri[1], tri[0]
    ab5 = rng.standard_normal((5, n))
    ab5[2] += 10
    b = rng.standard_normal(n)
    p3, q3 = rng.standard_normal((LONG_N, 3)), rng.standard_normal((LONG_N, 3))
    base = {"v": v, "dl": tri[0], "dd": tri[1], "du": tri[2], "ab_tri": ab_tri, "ab5": ab5, "b": b, "p3": p3, "q3": q3}
    if any(k.lower().startswith("q_") for k in keys):
        nb, m = BAND_BATCH
        qtri = [rng.standard_normal((nb, m - 1)), rng.standard_normal((nb, m)) + 10, rng.standard_normal((nb, m - 1))]
        qab_tri = np.zeros((nb, 3, m))
        qab_tri[:, 0, 1:], qab_tri[:, 1], qab_tri[:, 2, :-1] = qtri[2], qtri[1], qtri[0]
        qab5 = rng.standard_normal((nb, 5, m))
        qab5[:, 2] += 10
        base.update(
            q_dl=qtri[0],
            q_dd=qtri[1],
            q_du=qtri[2],
            q_ab_tri=qab_tri,
            q_ab5=qab5,
            q_b=rng.standard_normal((nb, m, 1)),  # scipy's batched solve_banded wants b as (…, n, k)
        )
    return {k: base[k] if k in base else sd.asarray(base[k.lower()]) for k in keys}


# ---------------------------------------------------------------- rows


def _outer(x0):
    return lambda: x0[..., :, None] * x0[..., None, :]


def _diagonal(a):
    import numpy as np

    return lambda: np.linalg.diagonal(a).copy()


def _kron_batch(a, k3):
    batch, m, n = a.shape[:-2], a.shape[-2], a.shape[-1]
    return lambda: (a[..., :, None, :, None] * k3[:, None, :]).reshape(*batch, 3 * m, 3 * n)


def matrix_rows(shape):
    """(name, sd call, (lib, ref call), note) at one square shape: baseline.py's rows and
    the ones added since. ROW_SHAPES limits a row to some shapes."""
    single = len(shape) == 2
    return [
        ("solve(a, b)", S("solve", "A", "B"), N("linalg.solve", "a", "b"), ""),
        ("solve(a, b16)", S("solve", "A", "B16"), N("linalg.solve", "a", "b16"), "오른쪽 변 16개"),
        ("inv(a)", S("inv", "A"), N("linalg.inv", "a"), ""),
        ("det(a)", S("det", "A"), N("linalg.det", "a"), ""),
        ("slogdet(a)", S("slogdet", "A"), N("linalg.slogdet", "a"), ""),
        ("vecdot(a, a)", S("vecdot", "A", "A"), N("vecdot", "a", "a"), "행마다 내적"),
        ("outer(x, x)", S("outer", "X0", "X0"), ("numpy", Expr("x0", _outer)), "numpy는 브로드캐스트 곱"),
        ("trace(a)", S("trace", "A"), N("linalg.trace", "a"), ""),
        ("diagonal(a)", S("diagonal", "A"), ("numpy", Expr("a", _diagonal)), "numpy는 뷰를 복사"),
        ("matrix_power(a, 3)", S("matrix_power", "A", 3), N("linalg.matrix_power", "a", 3), ""),
        ("vector_norm(a, axis=-1)", S("vector_norm", "A", axis=-1), N("linalg.vector_norm", "a", axis=-1), ""),
        ("matrix_norm(a)", S("matrix_norm", "A"), N("linalg.matrix_norm", "a"), ""),
        ("matrix_norm(a, ord=1)", S("matrix_norm", "A", ord=1), N("linalg.matrix_norm", "a", ord=1), ""),
        ("matrix_norm(a, ord=2)", S("matrix_norm", "A", ord=2), N("linalg.matrix_norm", "a", ord=2), ""),
        (
            "solve_triangular(t, b, upper=True)",
            S("solve_triangular", "T", "B", upper=True),
            P("solve_triangular", "t", "b", lower=False),
            "",
        ),
        (
            "solve_triangular(tF, b, upper=True)",
            S("solve_triangular", "TF", "B", upper=True),
            P("solve_triangular", "tf", "b", lower=False),
            "t는 F 순서",
        ),
        (
            "solve_triangular(t, b16, upper=True)",
            S("solve_triangular", "T", "B16", upper=True),
            P("solve_triangular", "t", "b16", lower=False),
            "오른쪽 변 16개",
        ),
        ("cholesky(spd)", S("cholesky", "SPD"), N("linalg.cholesky", "spd"), ""),
        (
            "cholesky_solve(l, b)",
            S("cholesky_solve", "CHOL", "B"),
            P("cho_solve", "cho_f", "b"),
            "l은 각자의 cholesky 결과",
        ),
        (
            "cholesky_solve(l, b16)",
            S("cholesky_solve", "CHOL", "B16"),
            P("cho_solve", "cho_f", "b16"),
            "오른쪽 변 16개",
        ),
        ("lu(a)", S("lu", "A"), P("lu", "a", p_indices=True), ""),
        ("lu_solve(l, u, perm, b)", S("lu_solve", "LU_L", "LU_U", "LU_P", "B"), P("lu_solve", "lu_f", "b"), ""),
        (
            "lu_solve(l, u, perm, b16)",
            S("lu_solve", "LU_L", "LU_U", "LU_P", "B16"),
            P("lu_solve", "lu_f", "b16"),
            "오른쪽 변 16개",
        ),
        ("qr(a)", S("qr", "A"), N("linalg.qr", "a"), ""),
        ("qr_pivoted(a)", S("qr_pivoted", "A"), P("qr", "a", mode="economic", pivoting=True), ""),
        ("lstsq(a, b)", S("lstsq", "A", "B"), N("linalg.lstsq", "a", "b") if single else P("lstsq", "a", "b"), ""),
        ("eigh(h)", S("eigh", "H"), N("linalg.eigh", "h"), ""),
        ("eigvalsh(h)", S("eigvalsh", "H"), N("linalg.eigvalsh", "h"), ""),
        ("eigh(h, spd)", S("eigh", "H", "SPD"), P("eigh", "h", "spd"), ""),
        ("eigvalsh(h, spd)", S("eigvalsh", "H", "SPD"), P("eigh", "h", "spd", eigvals_only=True), ""),
        ("svd(a)", S("svd", "A"), N("linalg.svd", "a", full_matrices=False), ""),
        ("svdvals(a)", S("svdvals", "A"), N("linalg.svdvals", "a"), ""),
        ("matrix_rank(a)", S("matrix_rank", "A"), N("linalg.matrix_rank", "a"), ""),
        ("pinv(a)", S("pinv", "A"), N("linalg.pinv", "a"), ""),
        ("cond(a)", S("cond", "A"), N("linalg.cond", "a"), ""),
        ("cond(a, ord=1)", S("cond", "A", ord=1), N("linalg.cond", "a", 1), ""),
        ("eig(a)", S("eig", "A"), N("linalg.eig", "a"), ""),
        ("eigvals(a)", S("eigvals", "A"), N("linalg.eigvals", "a"), ""),
        ("schur(a)", S("schur", "A"), P("schur", "a"), ""),
        ("hessenberg(a)", S("hessenberg", "A"), P("hessenberg", "a", calc_q=True), ""),
        ("ldl(h)", S("ldl", "H"), P("ldl", "h"), ""),
        (
            "solve_tridiagonal(dl, d, du, b)",
            S("solve_tridiagonal", "DL", "DD", "DU", "B"),
            P("solve_banded", (1, 1), "ab_tri", "b"),
            "scipy는 solve_banded((1, 1), ...)",
        ),
        (
            "solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2)",
            S("solve_banded", "AB5", "B", lower_bandwidth=2, upper_bandwidth=2),
            P("solve_banded", (2, 2), "ab5", "b"),
            "",
        ),
        (
            "solve_banded(ab, b, lower_bandwidth=1, upper_bandwidth=1)",
            S("solve_banded", "AB_TRI", "B", lower_bandwidth=1, upper_bandwidth=1),
            P("solve_banded", (1, 1), "ab_tri", "b"),
            "",
        ),
        ("solve_sylvester(a, a, c)", S("solve_sylvester", "A", "A", "C"), P("solve_sylvester", "a", "a", "c"), ""),
        (
            "kron(a, k)",
            S("kron", "A", "K3"),
            N("kron", "a", "k3") if single else ("numpy", Expr("a k3", _kron_batch)),
            "k는 3×3" if single else "k는 3×3, numpy는 브로드캐스트 곱을 reshape",
        ),
        ("cross(v, w)", S("cross", "V3", "W3"), N("cross", "v3", "w3"), "행마다 앞 세 원소"),
        ("expm(a)", S("expm", "A"), P("expm", "a"), ""),
        ("expm(t)", S("expm", "UT"), P("expm", "ut"), "위 삼각 행렬"),
        ("expm(0.01·a)", S("expm", "A001"), P("expm", "a001"), "노름이 작은 행렬"),
    ]


# R's version of a row on a single real matrix, as the accuracy measurement calls R:
# name -> (R expression, numpy data keys R reads (same names in R), R setup, packages).
# eigen gets symmetric = FALSE where the accuracy side leaves it to isSymmetric (an
# all.equal of the matrix and its transpose, not part of the eigenvalue problem).
R_ROWS = {
    "solve(a, b)": ("solve(a, b)", ("a", "b")),
    "solve(a, b16)": ("solve(a, b16)", ("a", "b16")),
    "inv(a)": ("solve(a)", ("a",)),
    "det(a)": ("det(a)", ("a",)),
    "cholesky(spd)": ("chol(spd)", ("spd",)),
    "qr_pivoted(a)": ("q <- qr(a, LAPACK = TRUE); list(qr.Q(q), qr.R(q), q$pivot)", ("a",)),
    "eigh(h)": ("eigen(h, symmetric = TRUE)", ("h",)),
    "eigvalsh(h)": ("eigen(h, symmetric = TRUE, only.values = TRUE)", ("h",)),
    "svd(a)": ("svd(a)", ("a",)),
    "svdvals(a)": ("svd(a, 0, 0)", ("a",)),
    "eig(a)": ("eigen(a, symmetric = FALSE)", ("a",)),
    "eigvals(a)": ("eigen(a, symmetric = FALSE, only.values = TRUE)", ("a",)),
    "expm(a)": ("expm(A)", ("a",), "A <- Matrix(a)", ("Matrix",)),
    "expm(t)": ("expm(A)", ("ut",), "A <- Matrix(ut)", ("Matrix",)),
    "expm(0.01·a)": ("expm(A)", ("a001",), "A <- Matrix(a001)", ("Matrix",)),
    "vecdot(v, v)": ("crossprod(v, v)", ("v",)),  # long vectors: the accuracy side's array vecdot
}


def _r_ref(name):
    spec = R_ROWS.get(name)
    if spec is None:
        return None
    expr, keys, setup, packages = (*spec, "", ())[:4]
    return RRef("R", expr, data={k: k for k in keys}, setup=setup, packages=tuple(packages))


_BASE_SHAPES = {(3, 3), (32, 32), (300, 300), (1000, 1000), (1000, 3, 3), (100, 32, 32)}
_SPLIT = {
    (16, 3, 3): {
        "solve(a, b)",
        "inv(a)",
        "det(a)",
        "cholesky(spd)",
        "qr(a)",
        "eigh(h)",
        "svd(a)",
        "eig(a)",
        "expm(a)",
    },
    (8, 300, 300): {"solve(a, b)", "inv(a)", "eigh(h)", "svd(a)"},
}
_RHS16 = {(100, 32, 32), (1000, 1000)}
_SVALS = {(3, 3), (1000, 3, 3)}  # rows that share svdvals' kernel
# Rows limited to some shapes; a row not named here runs at every base shape.
ROW_SHAPES = {
    "solve(a, b16)": _RHS16,
    "solve_triangular(t, b16, upper=True)": _RHS16,
    "cholesky_solve(l, b16)": _RHS16,
    "lu_solve(l, u, perm, b16)": _RHS16,
    "solve_triangular(tF, b, upper=True)": {(1000, 1000)},
    "matrix_rank(a)": _SVALS,
    "cond(a)": _SVALS,
    "matrix_norm(a, ord=2)": _SVALS,
    # the same LAPACK on both sides at 1000² (ratio about 1.00) and 18 s a round
    "eig(a)": _BASE_SHAPES - {(1000, 1000)},
    "eigvals(a)": _BASE_SHAPES - {(1000, 1000)},
    "schur(a)": _BASE_SHAPES - {(1000, 1000)},
    "solve_banded(ab, b, lower_bandwidth=1, upper_bandwidth=1)": {(100, 32, 32)},
}
COMPLEX_ROWS = {"solve(a, b)", "inv(a)", "cholesky(spd)", "qr(a)", "eigh(h)", "svd(a)", "eig(a)", "expm(a)"}


def _wanted(name, shape):
    if shape in _SPLIT:
        return name in _SPLIT[shape]
    return shape in ROW_SHAPES.get(name, _BASE_SHAPES)


def rect_rows():
    """(name, shape, sd call, (lib, ref call), note) on rectangular matrices."""
    tall, wide = (10_000, 100), (200, 1000)
    return [
        ("lstsq(a, b)", tall, S("lstsq", "A", "B"), N("linalg.lstsq", "a", "b"), ""),
        ("lstsq(a, b)", wide, S("lstsq", "A", "B"), N("linalg.lstsq", "a", "b"), "식보다 미지수가 많다"),
        ("lstsq(a, b)", (1000, 10, 3), S("lstsq", "A", "B"), P("lstsq", "a", "b"), ""),
        ("qr(a)", tall, S("qr", "A"), N("linalg.qr", "a"), ""),
        (
            'qr(a, mode="complete")',
            (2000, 100),
            S("qr", "A", mode="complete"),
            N("linalg.qr", "a", mode="complete"),
            "",
        ),
        ("qr_pivoted(a)", tall, S("qr_pivoted", "A"), P("qr", "a", mode="economic", pivoting=True), ""),
        ("svd(a)", (2000, 200), S("svd", "A"), N("linalg.svd", "a", full_matrices=False), ""),
        (
            'svd(a, mode="complete")',
            (2000, 200),
            S("svd", "A", mode="complete"),
            N("linalg.svd", "a", full_matrices=True),
            "",
        ),
        ("svd(a)", wide, S("svd", "A"), N("linalg.svd", "a", full_matrices=False), ""),
        ("pinv(a)", (2000, 200), S("pinv", "A"), N("linalg.pinv", "a"), ""),
        ("matrix_rank(a)", (2000, 200), S("matrix_rank", "A"), N("linalg.matrix_rank", "a"), ""),
        ("svdvals(a)", (2000, 200), S("svdvals", "A"), N("linalg.svdvals", "a"), ""),
    ]


def long_rows():
    """(name, size, n, sd call, (lib, ref call), note), as baseline.py's long_rows, and
    the batched banded systems."""
    band = "n=10^5"
    qsize = f"({BAND_BATCH[0]}, n={BAND_BATCH[1]})"
    qn = BAND_BATCH[0] * BAND_BATCH[1]
    return [
        ("vecdot(v, v)", "(10^6,)", LONG_N, S("vecdot", "V", "V"), N("vecdot", "v", "v"), ""),
        ("vector_norm(v)", "(10^6,)", LONG_N, S("vector_norm", "V"), N("linalg.vector_norm", "v"), ""),
        (
            "vector_norm(v, ord=inf)",
            "(10^6,)",
            LONG_N,
            S("vector_norm", "V", ord=float("inf")),
            N("linalg.vector_norm", "v", ord=float("inf")),
            "",
        ),
        (
            "solve_tridiagonal(dl, d, du, b)",
            band,
            BAND_N,
            S("solve_tridiagonal", "DL", "DD", "DU", "B"),
            P("solve_banded", (1, 1), "ab_tri", "b"),
            "scipy는 solve_banded((1, 1), ...)",
        ),
        (
            "solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2)",
            band,
            BAND_N,
            S("solve_banded", "AB5", "B", lower_bandwidth=2, upper_bandwidth=2),
            P("solve_banded", (2, 2), "ab5", "b"),
            "",
        ),
        (
            "solve_tridiagonal(dl, d, du, b)",
            qsize,
            qn,
            S("solve_tridiagonal", "Q_DL", "Q_DD", "Q_DU", "Q_B"),
            P("solve_banded", (1, 1), "q_ab_tri", "q_b"),
            "연립 100개, scipy는 solve_banded((1, 1), ...)",
        ),
        (
            "solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2)",
            qsize,
            qn,
            S("solve_banded", "Q_AB5", "Q_B", lower_bandwidth=2, upper_bandwidth=2),
            P("solve_banded", (2, 2), "q_ab5", "q_b"),
            "연립 100개",
        ),
        ("cross(p, q)", "(10^6,3)", LONG_N, S("cross", "P3", "Q3"), N("cross", "p3", "q3"), ""),
    ]


# The representative rows of bench/linalg/final_table.py (PICK), less the slowest, and
# one rectangular and one complex row: (name, shape or size).
QUICK = {
    ("solve(a, b)", (3, 3)),
    ("solve(a, b)", (1000, 1000)),
    ("matrix_norm(a, ord=1)", (1000, 3, 3)),
    ("matrix_power(a, 3)", (100, 32, 32)),
    ("solve_triangular(t, b, upper=True)", (1000, 1000)),
    ("lu_solve(l, u, perm, b)", (1000, 1000)),
    ("qr(a)", (1000, 3, 3)),
    ("eigh(h)", (1000, 3, 3)),
    ("svd(a)", (100, 32, 32)),
    ("eig(a)", (100, 32, 32)),
    ("solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2)", (1000, 3, 3)),
    ("expm(a)", (1000, 3, 3)),
    ("expm(a)", (300, 300)),
    ("vecdot(v, v)", "(10^6,)"),
}
QUICK_RECT = {("lstsq(a, b)", (10_000, 100))}
QUICK_COMPLEX = {("solve(a, b)", (300, 300))}


def _keys(*calls):
    return sorted({k for c in calls if c is not None for k in c.keys if isinstance(k, str)})


def _ref(r, rref=None):
    out = () if r is None else (Ref(r[0], r[1].make),)
    return out if rref is None else (*out, rref)


def _size(shape):
    return "(" + ",".join(map(str, shape)) + ")"


def _matrix_case(group, shape, name, sdc, ref, note, quick, cplx=False, threads=(1, 10)):
    rref = _r_ref(name) if not cplx and len(shape) == 2 else None  # R: single real matrices
    keys = sorted({*_keys(sdc, ref[1] if ref else None), *(rref.data if rref else ())})
    quiet = name.startswith(("det(", "slogdet("))
    return Case(
        area="linalg",
        group=group,
        name=name,
        size=_size(shape),
        setup=lambda: _matrix_data(shape, keys, quiet, cplx),
        sd=sdc.make,
        refs=_ref(ref, rref),
        note=note,
        quick=quick,
        threads=threads,
    )


# The section title of each group in bench/linalg/speed.md (the group is the English key in
# the case ids).
TITLES = {
    "small": "작은 행렬",
    "large": "큰 행렬",
    "batched": "행렬 묶음",
    "complex": "복소 행렬",
    "rectangular": "직사각 행렬",
    "vectors-banded": "긴 벡터·띠 행렬",
}


def cases() -> list[Case]:
    out = []
    for shape in MATRIX_SHAPES:
        for name, sdc, ref, note in matrix_rows(shape):
            if not _wanted(name, shape):
                continue
            small = len(shape) == 2 and shape[0] <= 32
            threads = (
                (1,)
                if small and not name.startswith("solve_sylvester") and (name, shape) not in POOL_CHECK
                else (1, 10)
            )
            out.append(_matrix_case(KIND[shape], shape, name, sdc, ref, note, (name, shape) in QUICK, threads=threads))
    for shape in COMPLEX_SHAPES:
        for name, sdc, ref, note in matrix_rows(shape):
            if name in COMPLEX_ROWS:
                out.append(
                    _matrix_case(COMPLEX_GROUP, shape, name, sdc, ref, note, (name, shape) in QUICK_COMPLEX, cplx=True)
                )
    for name, shape, sdc, ref, note in rect_rows():
        out.append(_matrix_case(RECT_GROUP, shape, name, sdc, ref, note, (name, shape) in QUICK_RECT))
    for name, size, n, sdc, ref, note in long_rows():
        rref = _r_ref(name) if name == "vecdot(v, v)" else None
        keys = sorted({*_keys(sdc, ref[1]), *(rref.data if rref else ())})
        out.append(
            Case(
                area="linalg",
                group=LONG_GROUP,
                name=name,
                size=size,
                n=n,
                setup=lambda keys=keys: _long_data(keys),
                sd=sdc.make,
                refs=_ref(ref, rref),
                note=note,
                quick=(name, size) in QUICK,
            )
        )
    return out
