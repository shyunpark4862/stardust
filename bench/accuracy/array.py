"""The array area: the float64 element-wise functions on vForce and pow, and the reductions.

Element-wise (tests/array/ref/<function>.npz, tools/array_ref.py, read through
tests/array/refdata.py): the error against Arb in ULP (ulp_distance to the correctly rounded
value) and relative, numpy's own error at the same points, and the contract of
docs/api/array.md -- within ULP_BOUND of the correctly rounded value (tests/array/refdata.py,
tests/array/test_reference.py) -- as the largest ULP over the bound. The ULP distance to
numpy's value is recorded too (no contract: the two may differ in the last places).

Reductions (sum, mean, var, std, prod, vecdot, matmul; tests/array/ref/reductions.npz, also
tools/array_ref.py): the references were computed once in exact arithmetic (Shewchuk's fsum
for sums and for dot products of exactly split products, rationals for the variance, Arb for the
product) and stored with their condition numbers. There is no documented bound, so they have
no contract: the measure is the relative error and the relative error over u times the
condition number (sum |d log f / d log x_i|: sum |x_i| / |sum x_i| for a sum), the size the
inputs' own rounding would give; the baseline comparison judges them.

Only sd's values (and numpy's, for the distance to numpy) are computed at measurement time.
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region, ulp_distance

U = 2.0**-53

ELEMENTWISE = (
    "exp", "log", "log2", "log10", "sin", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "pow", "cabs",
)  # fmt: skip
TINY = 2.0**-1022
NUMPY = {"pow": "power", "atan2": "arctan2"}
REDUCTIONS = ("sum", "mean", "var", "std", "prod", "vecdot", "matmul")


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
    """The errors and contract (tests/array/refdata.py: ULP_BOUND of the correctly rounded value)
    of sd or a comparator (numpy, R) at the reference points; for sd also its ULP distance to
    numpy's values (recorded)."""
    from tests.array import refdata as A
    from tests.special import refdata as R

    inputs, hi, lo = A.elementwise(name)
    keys, args = list(inputs), list(inputs.values())
    got = _ew_values(name, args, lib)
    keep = np.ones(hi.shape, bool) if lib == "sd" else np.isfinite(got) | ~np.isfinite(hi)
    ulp, bound = A.elementwise_contract(name, got, hi)
    with np.errstate(all="ignore"):
        rel = R.rel_error(got, hi, lo)
    extra = {}
    if lib == "sd":
        npv = _ew_values(name, args, "numpy")
        d = ulp_distance(got, npv).astype(np.float64)
        extra["numpy_dist"] = np.where(np.isnan(got) == np.isnan(npv), d, np.inf)
    where = np.array([", ".join(f"{k}={v!r}" for k, v in zip(keys, p)) for p in zip(*(a.tolist() for a in args))])
    out = {}
    sub = np.abs(hi) < TINY  # results below the normal range: their ULP is the subnormal spacing
    for label, m0 in (("normal", ~sub), ("subnormal", sub)):
        m = m0 & keep
        if m0.any():
            out[label] = region(
                where[m], ulp=ulp[m], rel=rel[m], contract=ulp[m], bound=bound,
                excluded=None if lib == "sd" else int((m0 & ~keep).sum()), **{k: v[m] for k, v in extra.items()},
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
    from tests.array import refdata as A

    return A.reduction_groups()


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
