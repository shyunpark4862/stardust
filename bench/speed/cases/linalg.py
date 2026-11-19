"""sd.linalg against numpy and scipy: every function on the fixed shapes of the old
baseline (single small and large matrices, batches, long vectors and long structured
systems). From bench/linalg/baseline.py (matrix_rows, long_rows), timed by
bench/linalg/stage_compare.py and summed up by bench/linalg/final_table.py; after the
merge rerun through bench/merge/rerun.sh (bench/merge/m8/final/).

The comparator is numpy where numpy has the function, else scipy, at their default
threads (their BLAS on its default pool), as stage_compare.py --ref did; none where
neither takes the input (a batch for lu_solve, ldl, the banded solvers). Every call is
a functools.partial over the prepared arrays where the call allows it, so the timed
callable adds no Python frame of its own on either side."""

from __future__ import annotations

import functools
import importlib

from ..registry import Case, Ref

MATRIX_SHAPES = [(3, 3), (32, 32), (300, 300), (1000, 1000), (1000, 3, 3), (100, 32, 32)]
KIND = {
    (3, 3): "작은 행렬", (32, 32): "작은 행렬",
    (300, 300): "큰 행렬", (1000, 1000): "큰 행렬",
    (1000, 3, 3): "행렬 묶음", (100, 32, 32): "행렬 묶음",
}
LONG_GROUP = "긴 벡터·띠 행렬"
LONG_N = 1_000_000
BAND_N = 100_000


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
        return functools.partial(_resolve(self.path), *(d[k] if isinstance(k, str) else k for k in self.keys),
                                 **self.kw)


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


def _matrix_data(shape, keys, quiet=False):
    """The inputs of baseline.py's matrix_rows at one shape; only `keys` (and what they
    need) are built. Lower case: numpy, upper case: the sd copy.

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

    n = shape[-1]
    batch = shape[:-2]
    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape)
    b = rng.standard_normal((*batch, n, 1))
    c = rng.standard_normal(shape)
    tri = [rng.standard_normal((*batch, n - 1)), rng.standard_normal((*batch, n)) + 10,
           rng.standard_normal((*batch, n - 1))]
    ab5 = rng.standard_normal((*batch, 5, n))
    ab5[..., 2, :] += 10  # kl = ku = 2, a dominant diagonal
    k3 = rng.standard_normal((3, 3))  # kron's second operand: (..., 3n, 3n) results
    memo = {"a": a, "b": b, "c": c, "dl": tri[0], "dd": tri[1], "du": tri[2], "ab5": ab5, "k3": k3}

    def get(k):
        if k in memo:
            return memo[k]
        if k == "h":
            v = (a + np.swapaxes(a, -1, -2)) / 2
        elif k == "spd":
            v = a @ np.swapaxes(a, -1, -2) + n * np.eye(n)
        elif k == "t":
            v = np.triu(a) + n * np.eye(n)
        elif k == "chol":
            v = np.linalg.cholesky(get("spd"))
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
        elif k[0].isupper() and k.lower() in ("a", "b", "c", "h", "spd", "t", "chol", "x0", "dl", "dd", "du",
                                               "ab5", "ab_tri", "k3", "v3", "w3"):
            v = sd.asarray(get(k.lower()))
        else:
            raise KeyError(k)
        memo[k] = v
        return v

    return {k: get(k) for k in keys}


def _long_data(keys):
    """baseline.py's long_rows inputs."""
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
    base = {"v": v, "dl": tri[0], "dd": tri[1], "du": tri[2], "ab_tri": ab_tri, "ab5": ab5, "b": b,
            "p3": p3, "q3": q3}
    return {k: base[k] if k in base else sd.asarray(base[k.lower()]) for k in keys}


# ---------------------------------------------------------------- rows


def _outer(x0):
    return lambda: x0[..., :, None] * x0[..., None, :]


def _diagonal(a):
    import numpy as np

    return lambda: np.linalg.diagonal(a).copy()


def _cho_solve(chol, b):
    import scipy.linalg as sl

    return functools.partial(sl.cho_solve, (chol, True), b)


def _kron_batch(a, k3):
    batch, m, n = a.shape[:-2], a.shape[-2], a.shape[-1]
    return lambda: (a[..., :, None, :, None] * k3[:, None, :]).reshape(*batch, 3 * m, 3 * n)


def matrix_rows(shape):
    """(name, sd call, (lib, ref call) or None, note) at one shape, as baseline.py."""
    single = len(shape) == 2
    return [
        ("solve(a, b)", S("solve", "A", "B"), N("linalg.solve", "a", "b"), ""),
        ("inv(a)", S("inv", "A"), N("linalg.inv", "a"), ""),
        ("det(a)", S("det", "A"), N("linalg.det", "a"), ""),
        ("slogdet(a)", S("slogdet", "A"), N("linalg.slogdet", "a"), ""),
        ("vecdot(a, a)", S("vecdot", "A", "A"), N("vecdot", "a", "a"), "행마다 내적"),
        ("outer(x, x)", S("outer", "X0", "X0"), ("numpy", Expr("x0", _outer)), "numpy는 브로드캐스트 곱"),
        ("trace(a)", S("trace", "A"), N("linalg.trace", "a"), ""),
        ("diagonal(a)", S("diagonal", "A"), ("numpy", Expr("a", _diagonal)), "numpy는 뷰를 복사"),
        ("matrix_power(a, 3)", S("matrix_power", "A", 3),
         N("linalg.matrix_power", "a", 3), ""),
        ("vector_norm(a, axis=-1)", S("vector_norm", "A", axis=-1), N("linalg.vector_norm", "a", axis=-1), ""),
        ("matrix_norm(a)", S("matrix_norm", "A"), N("linalg.matrix_norm", "a"), ""),
        ("matrix_norm(a, ord=1)", S("matrix_norm", "A", ord=1), N("linalg.matrix_norm", "a", ord=1), ""),
        ("matrix_norm(a, ord=2)", S("matrix_norm", "A", ord=2), N("linalg.matrix_norm", "a", ord=2), ""),
        ("solve_triangular(t, b, upper=True)", S("solve_triangular", "T", "B", upper=True),
         P("solve_triangular", "t", "b", lower=False), ""),
        ("cholesky(spd)", S("cholesky", "SPD"), N("linalg.cholesky", "spd"), ""),
        ("cholesky_solve(l, b)", S("cholesky_solve", "CHOL", "B"), ("scipy", Expr("chol b", _cho_solve)), ""),
        ("lu(a)", S("lu", "A"), P("lu", "a", p_indices=True), ""),
        ("lu_solve(l, u, perm, b)", S("lu_solve", "LU_L", "LU_U", "LU_P", "B"),
         P("lu_solve", "lu_f", "b") if single else None, ""),
        ("qr(a)", S("qr", "A"), N("linalg.qr", "a"), ""),
        ('qr(a, mode="complete")', S("qr", "A", mode="complete"), N("linalg.qr", "a", mode="complete"), ""),
        ("qr_pivoted(a)", S("qr_pivoted", "A"), P("qr", "a", mode="economic", pivoting=True), ""),
        ("lstsq(a, b)", S("lstsq", "A", "B"), N("linalg.lstsq", "a", "b") if single else P("lstsq", "a", "b"), ""),
        ("eigh(h)", S("eigh", "H"), N("linalg.eigh", "h"), ""),
        ("eigvalsh(h)", S("eigvalsh", "H"), N("linalg.eigvalsh", "h"), ""),
        ("eigh(h, spd)", S("eigh", "H", "SPD"), P("eigh", "h", "spd"), ""),
        ("svd(a)", S("svd", "A"), N("linalg.svd", "a", full_matrices=False), ""),
        ("svdvals(a)", S("svdvals", "A"), N("linalg.svdvals", "a"), ""),
        ("matrix_rank(a)", S("matrix_rank", "A"), N("linalg.matrix_rank", "a"), ""),
        ("pinv(a)", S("pinv", "A"), N("linalg.pinv", "a"), ""),
        ("cond(a)", S("cond", "A"), N("linalg.cond", "a"), ""),
        ("eig(a)", S("eig", "A"), N("linalg.eig", "a"), ""),
        ("eigvals(a)", S("eigvals", "A"), N("linalg.eigvals", "a"), ""),
        ("schur(a)", S("schur", "A"), P("schur", "a"), ""),
        ("hessenberg(a)", S("hessenberg", "A"), P("hessenberg", "a", calc_q=True), ""),
        ("ldl(h)", S("ldl", "H"), P("ldl", "h") if single else None, ""),
        ("solve_tridiagonal(dl, d, du, b)", S("solve_tridiagonal", "DL", "DD", "DU", "B"),
         P("solve_banded", (1, 1), "ab_tri", "b") if single else None,
         "scipy는 solve_banded((1, 1), ...)" if single else ""),
        ("solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2)",
         S("solve_banded", "AB5", "B", lower_bandwidth=2, upper_bandwidth=2),
         P("solve_banded", (2, 2), "ab5", "b") if single else None, ""),
        ("solve_banded(ab, b, lower_bandwidth=1, upper_bandwidth=1)",
         S("solve_banded", "AB_TRI", "B", lower_bandwidth=1, upper_bandwidth=1),
         P("solve_banded", (1, 1), "ab_tri", "b") if single else None, ""),
        ("solve_sylvester(a, a, c)", S("solve_sylvester", "A", "A", "C"), P("solve_sylvester", "a", "a", "c"), ""),
        ("kron(a, k)", S("kron", "A", "K3"),
         N("kron", "a", "k3") if single else ("numpy", Expr("a k3", _kron_batch)),
         "k는 3×3" if single else "k는 3×3, numpy는 브로드캐스트 곱을 reshape"),
        ("cross(v, w)", S("cross", "V3", "W3"), N("cross", "v3", "w3"), "행마다 앞 세 원소"),
        ("expm(a)", S("expm", "A"), P("expm", "a"), ""),
    ]


def long_rows():
    """(name, size, n, sd call, (lib, ref call), note), as baseline.py's long_rows."""
    band = "n=10^5"
    return [
        ("vecdot(v, v)", "(10^6,)", LONG_N, S("vecdot", "V", "V"), N("vecdot", "v", "v"), ""),
        ("vector_norm(v)", "(10^6,)", LONG_N, S("vector_norm", "V"), N("linalg.vector_norm", "v"), ""),
        ("vector_norm(v, ord=inf)", "(10^6,)", LONG_N, S("vector_norm", "V", ord=float("inf")),
         N("linalg.vector_norm", "v", ord=float("inf")), ""),
        ("solve_tridiagonal(dl, d, du, b)", band, BAND_N, S("solve_tridiagonal", "DL", "DD", "DU", "B"),
         P("solve_banded", (1, 1), "ab_tri", "b"), "scipy는 solve_banded((1, 1), ...)"),
        ("solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2)", band, BAND_N,
         S("solve_banded", "AB5", "B", lower_bandwidth=2, upper_bandwidth=2),
         P("solve_banded", (2, 2), "ab5", "b"), ""),
        ("cross(p, q)", "(10^6,3)", LONG_N, S("cross", "P3", "Q3"), N("cross", "p3", "q3"), ""),
    ]


# The representative rows of bench/linalg/final_table.py (PICK), less the slowest.
QUICK = {
    ("solve(a, b)", (3, 3)), ("solve(a, b)", (1000, 1000)), ("matrix_norm(a, ord=1)", (1000, 3, 3)),
    ("matrix_power(a, 3)", (100, 32, 32)), ("solve_triangular(t, b, upper=True)", (1000, 1000)),
    ("lu_solve(l, u, perm, b)", (1000, 1000)), ("qr(a)", (1000, 3, 3)), ("eigh(h)", (1000, 3, 3)),
    ("svd(a)", (100, 32, 32)), ("eig(a)", (100, 32, 32)),
    ("solve_banded(ab, b, lower_bandwidth=2, upper_bandwidth=2)", (1000, 3, 3)),
    ("expm(a)", (1000, 3, 3)), ("expm(a)", (300, 300)),
    ("vecdot(v, v)", "(10^6,)"), ("vector_norm(v)", "(10^6,)"),
}


def _keys(*calls):
    return sorted({k for c in calls if c is not None for k in c.keys if isinstance(k, str)})


def _ref(r):
    if r is None:
        return ()
    lib, call = r
    return (Ref(lib, call.make),)


def cases() -> list[Case]:
    out = []
    for shape in MATRIX_SHAPES:
        size = "(" + ",".join(map(str, shape)) + ")"
        for name, sdc, ref, note in matrix_rows(shape):
            keys = _keys(sdc, ref[1] if ref else None)
            quiet = name.startswith(("det(", "slogdet("))
            out.append(Case(
                area="linalg", group=KIND[shape], name=name, size=size,
                setup=lambda shape=shape, keys=keys, quiet=quiet: _matrix_data(shape, keys, quiet),
                sd=sdc.make, refs=_ref(ref), note=note,
                quick=(name, shape) in QUICK,
            ))
    for name, size, n, sdc, ref, note in long_rows():
        keys = _keys(sdc, ref[1])
        out.append(Case(
            area="linalg", group=LONG_GROUP, name=name, size=size, n=n,
            setup=lambda keys=keys: _long_data(keys),
            sd=sdc.make, refs=_ref(ref), note=note,
            quick=(name, size) in QUICK,
        ))
    return out
