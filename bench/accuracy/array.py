"""The array area: the float64 element-wise functions on vForce and pow, and the reductions.

Element-wise (tests/array/ref/<function>.npz, tools/array_ref.py): the error against Arb in
ULP and relative, numpy's own error at the same points, and the contract of
docs/api/array.md -- within 8 ULP of numpy (tests/array/test_vforce.py) -- as the largest ULP
distance to numpy over 8.

Reductions (sum, mean, var, std, prod, vecdot, matmul; tests/array/ref/reductions.npz, also
tools/array_ref.py): the references were computed once in exact arithmetic (Shewchuk's fsum
for sums and for dot products of exactly split products, rationals for the variance, Arb for the
product) and stored with their condition numbers. There is no documented bound, so they have
no contract: the measure is the relative error and the relative error over u times the
condition number (sum |d log f / d log x_i|: sum |x_i| / |sum x_i| for a sum), the size the
inputs' own rounding would give; the baseline comparison judges them.

Only sd's values (and numpy's, for the element-wise contract) are computed at measurement time.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .common import Custom, compare_libs, region

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "tests" / "array" / "ref"
U = 2.0**-53

ELEMENTWISE = (
    "exp", "log", "log2", "log10", "sin", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "pow", "cabs",
)  # fmt: skip
TINY = 2.0**-1022
NUMPY = {"pow": "power", "atan2": "arctan2"}
REDUCTIONS = ("sum", "mean", "var", "std", "prod", "vecdot", "matmul")


def _ordered(a):
    """The doubles as int64 in their order (-0 and +0 both 0)."""
    i = np.asarray(a, dtype=np.float64).view(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFFFFFFFFFF), i)


def ulp_distance(a, b):
    """The number of doubles between a and b (0 for two NaNs, inf for one)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    oa, ob = _ordered(np.where(na, 0.0, a)), _ordered(np.where(nb, 0.0, b))
    same_sign = (oa >= 0) == (ob >= 0)  # the int64 difference cannot overflow
    d = np.where(
        same_sign, np.abs(np.where(same_sign, oa - ob, 0)), np.abs(oa.astype(float)) + np.abs(ob.astype(float))
    )
    return np.where(na & nb, 0.0, np.where(na | nb, np.inf, d.astype(np.float64)))


R_ELEMENTWISE = {
    "exp": "exp(a)", "log": "log(a)", "log2": "log2(a)", "log10": "log10(a)", "sin": "sin(a)", "asin": "asin(a)",
    "acos": "acos(a)", "atan": "atan(a)", "atan2": "atan2(a, b)", "sinh": "sinh(a)", "cosh": "cosh(a)",
    "tanh": "tanh(a)", "asinh": "asinh(a)", "acosh": "acosh(a)", "atanh": "atanh(a)", "pow": "a^b",
    "cabs": "Mod(complex(real = a, imaginary = b))",
}  # fmt: skip


def _ew_values(name, args, lib):
    import stardust as sd

    with np.errstate(all="ignore"):
        if lib == "R":
            from . import rsession

            return rsession.evaluate(R_ELEMENTWISE[name], *args)
        if name == "cabs":
            zc = args[0] + 1j * args[1]
            return np.asarray(sd.abs(sd.asarray(zc))) if lib == "sd" else np.abs(zc)
        if lib == "sd":
            return np.asarray(getattr(sd, name)(*(sd.asarray(a) for a in args)))
        return getattr(np, NUMPY.get(name, name))(*args)


def _elementwise(name, lib="sd"):
    """sd's errors and contract (8 ULP of numpy; complex abs: 1.5 ULP of the exact modulus), or a
    comparator's (numpy, R) errors at the same points (its contract column only for complex abs)."""
    from tests.special import refdata as R

    z = np.load(REF / f"{name}.npz")
    keys = {"atan2": ("y", "x"), "pow": ("x", "y"), "cabs": ("re", "im")}.get(name, ("x",))
    args = [z[k] for k in keys]
    hi, lo = z["hi"], z["lo"]
    got = _ew_values(name, args, lib)
    keep = np.ones(hi.shape, bool) if lib == "sd" else np.isfinite(got) | ~np.isfinite(hi)
    with np.errstate(all="ignore"):
        ulp, rel = R.ulp_error(got, hi, lo), R.rel_error(got, hi, lo)
    if name == "cabs":
        contract, bound = ulp, 1.5
    elif lib == "sd":
        contract, bound = ulp_distance(got, _ew_values(name, args, "numpy")), 8.0
    else:
        contract, bound = None, None
    where = np.array([", ".join(f"{k}={v!r}" for k, v in zip(keys, p)) for p in zip(*(a.tolist() for a in args))])
    out = {}
    sub = np.abs(hi) < TINY  # results below the normal range: their ULP is the subnormal spacing
    for label, m0 in (("normal", ~sub), ("subnormal", sub)):
        m = m0 & keep
        if m0.any():
            out[label] = region(
                where[m], ulp=ulp[m], rel=rel[m], contract=None if contract is None else contract[m], bound=bound,
                excluded=None if lib == "sd" else int((m0 & ~keep).sum()),
            )  # fmt: skip
    return out


# ------------------------------------------------------------------------------ reductions
# The reductions' inputs and references are tests/array/ref/reductions.npz (tools/array_ref.py):
# per group the function, the inputs (or, for the large ones, kind, shape and seed to regenerate
# them with reduction_inputs below, checked against the stored SHA-256 of their bytes), the
# reference as hi + lo and its condition number. Nothing is computed here but sd's values.

STORE_MAX = 200_000  # inputs up to this many elements are stored; larger ones are regenerated


def reduction_inputs(kind, shape, seed):
    """The input of a reduction group (deterministic: numpy's PCG64 and its uniform, normal and
    permutation routines)."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    if kind == "uniform":
        x = rng.random(n)
    elif kind == "half_two":  # [0.5, 2]: a product that stays in range
        x = rng.uniform(0.5, 2.0, n)
    elif kind == "normal":
        x = rng.standard_normal(n)
    elif kind == "wide":  # signs and 40 decades
        x = np.where(rng.random(n) < 0.5, -1.0, 1.0) * 10.0 ** rng.uniform(-20, 20, n)
    elif kind == "offset":  # a large mean: the variance cancels
        x = 1e8 + rng.standard_normal(n)
    elif kind == "cancel":  # pairs x, -x with a small remainder
        h = rng.standard_normal(n // 2) * 1e6
        x = rng.permutation(np.r_[h, -h * (1 + 1e-12 * rng.standard_normal(n // 2))])
    else:
        raise ValueError(kind)
    return x.reshape(shape)


def _reduction_groups():
    z = np.load(REF / "reductions.npz")
    out = {}
    for k in z.files:
        g, key = k[1:].split("_", 1)
        out.setdefault(int(g), {})[key] = z[k]
    return [out[i] for i in sorted(out)]


def _operand(g, name):
    """A group's input `name` (x or y): stored, or regenerated and checked."""
    import hashlib

    if name in g:
        return g[name]
    x = reduction_inputs(str(g[f"{name}_kind"]), tuple(int(v) for v in g[f"{name}_shape"]), int(g[f"{name}_seed"]))
    if hashlib.sha256(x.tobytes()).hexdigest() != str(g[f"{name}_sha256"]):
        raise RuntimeError(f"array reductions: regenerated input {g['label']} {name} differs from the reference's")
    return x


def _red_values(func, g, lib):
    """A reduction group's value(s) from sd, numpy or R (None: no such function)."""
    c = int(g["correction"]) if "correction" in g else 0
    xn = _operand(g, "x")
    yn = _operand(g, "y") if func in ("vecdot", "matmul") else None
    if lib == "R":
        from . import rsession

        if func in ("var", "std") and c != 1:  # R's var and sd divide by n - 1 only
            return None
        n0 = xn.shape[0]
        expr = {
            "sum": "colSums(matrix(X, nrow = %d, byrow = TRUE))" % n0 if "axis" in g else "sum(X)",
            "mean": "mean(X)", "var": "var(X)", "std": "sd(X)", "prod": "prod(X)",
            "vecdot": "as.numeric(crossprod(X, Y))",
            "matmul": "as.numeric(t(matrix(X, nrow = %d, byrow = TRUE) %%*%% matrix(Y, nrow = %d, byrow = TRUE)))"
            % (n0, n0),
        }[func]  # fmt: skip
        data = {"X": xn} if yn is None else {"X": xn, "Y": yn}
        return rsession.evaluate_named(expr, data)
    if lib == "numpy":
        got = {
            "sum": lambda: np.sum(xn, axis=0) if "axis" in g else np.sum(xn), "mean": lambda: np.mean(xn),
            "var": lambda: np.var(xn, ddof=c), "std": lambda: np.std(xn, ddof=c), "prod": lambda: np.prod(xn),
            "vecdot": lambda: np.vecdot(xn, yn), "matmul": lambda: xn @ yn,
        }[func]()  # fmt: skip
        return np.atleast_1d(np.asarray(got, dtype=float)).ravel()
    import stardust as sd

    x = sd.asarray(xn)
    if func == "sum":
        got = sd.sum(x, axis=0) if "axis" in g else sd.sum(x)
    elif func == "mean":
        got = sd.mean(x)
    elif func in ("var", "std"):
        got = getattr(sd, func)(x, correction=c)
    elif func == "prod":
        got = sd.prod(x)
    elif func == "vecdot":
        got = sd.vecdot(x, sd.asarray(yn))
    else:
        got = x @ sd.asarray(yn)
    return np.atleast_1d(np.asarray(got, dtype=float)).ravel()


def _reduction(func, lib="sd"):
    from tests.special import refdata as R

    out = {}
    for g in _reduction_groups():
        if str(g["func"]) != func:
            continue
        got = _red_values(func, g, lib)
        if got is None:
            continue
        hi, lo, cond = g["hi"], g["lo"], g["cond"]
        with np.errstate(all="ignore"):
            rel = R.rel_error(got, hi, lo)
        where = [str(g["label"])] if hi.size == 1 else [f"{g['label']} [{i}]" for i in range(hi.size)]
        out[str(g["label"])] = region(where, rel=rel, rel_cond=rel / (U * cond))
    return out


def sweeps():
    out = []
    for name in ELEMENTWISE:
        fn = lambda lib="sd", name=name: _elementwise(name, lib)  # noqa: E731
        out.append(Custom("array", name, fn, compare_libs(fn, ("numpy", "R"))))
    for f in REDUCTIONS:
        fn = lambda lib="sd", f=f: _reduction(f, lib) or None  # noqa: E731
        out.append(Custom("array", f, fn, compare_libs(fn, ("numpy", "R"), seconds=120)))
    return out
