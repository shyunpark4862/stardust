"""Reference values for the float64 element-wise functions on Accelerate vForce and pow, with Arb.

    .venv/bin/python tools/array_ref.py              # every function, on 10 processes
    .venv/bin/python tools/array_ref.py exp atan2 --jobs=4

writes tests/array/ref/<function>.npz (x, [y,] hi, lo; cabs, the modulus of a complex: re, im),
read by bench/accuracy (python -m bench.accuracy, area array). The inputs come from seeded
generators: the ranges of tests/array/test_vforce.py (log-uniform over many decades, else
uniform) and, per function, the places where the error concentrates (near the zeros and the
poles, the ends of the domain, subnormal and huge arguments). Each value is the exact function
at the double input, settled to 2^-110 relative (absolute 2^-110 2^-1022 for results below the
normal range; tools/arbref.py), as a double-double hi + lo; a result beyond the double range is
+-inf.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np
from flint import arb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from arbref import DOUBLE_TINY, precision, rigorous, to_dd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # bench.accuracy.array: the reductions' inputs

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "array" / "ref"
N = 20_000  # random points per function, besides the special regions


def logu(rng, lo, hi, n, signed=False):
    x = np.exp(rng.uniform(math.log(lo), math.log(hi), n))
    return x * np.where(rng.random(n) < 0.5, -1.0, 1.0) if signed else x


def near(c, n, rng, ulps=2**20):
    """Doubles within a few ULP to 2^20 ULP of c, both sides."""
    k = np.exp(rng.uniform(0, math.log(ulps), n)).astype(np.int64) * np.where(rng.random(n) < 0.5, -1, 1)
    return np.array([c + int(i) * math.ulp(c) for i in k])


def inputs(name, rng):
    """(x,) or (x, y) for the function."""
    u = rng.uniform
    sub = logu(rng, 5e-324, 2.2e-308, 200)
    if name == "exp":
        x = np.r_[u(-745, 709.78, N), logu(rng, 1e-300, 1e-3, 2000, True), u(-745.2, -708, 1000)]
    elif name in ("log", "log2", "log10"):
        x = np.r_[logu(rng, 1e-300, 1e300, N), near(1.0, 2000, rng), sub, u(0.5, 2, 2000)]
    elif name == "sin":
        x = np.r_[u(-1e5, 1e5, N), logu(rng, 1e-300, 1, 2000, True), logu(rng, 1e5, 1e22, 1000, True)]
        x = np.r_[x, (np.arange(1, 1001) * math.pi)]  # near the zeros: the double nearest k pi
    elif name in ("asin", "acos"):
        x = np.r_[u(-1, 1, N), near(1.0, 1000, rng)[:1000], -near(1.0, 1000, rng), logu(rng, 1e-300, 1e-3, 1000, True)]
        x = np.clip(x, -1.0, 1.0)
    elif name == "atan":
        x = np.r_[logu(rng, 1e-300, 1e300, N, True), u(-4, 4, 2000)]
    elif name in ("sinh", "cosh"):
        x = np.r_[u(-710, 710, N), logu(rng, 1e-300, 1, 2000, True), u(700, 710.47, 500)]
    elif name == "tanh":
        x = np.r_[u(-30, 30, N), logu(rng, 1e-300, 1, 2000, True)]
    elif name == "asinh":
        x = np.r_[logu(rng, 1e-300, 1e300, N, True), u(-2, 2, 2000)]
    elif name == "acosh":
        x = np.r_[1.0 + logu(rng, 2.3e-16, 1e300, N), near(1.0, 2000, rng)]
        x = x[x >= 1.0]
    elif name == "atanh":
        x = np.r_[u(-1, 1, N), logu(rng, 1e-300, 1e-3, 1000, True)]
        x = np.r_[x, near(1.0, 1000, rng), -near(1.0, 1000, rng)]
        x = x[np.abs(x) < 1.0]
    elif name == "atan2":
        y, x = logu(rng, 1e-300, 1e300, N, True), logu(rng, 1e-300, 1e300, N, True)
        a, b = u(-1e3, 1e3, 5000), u(-1e3, 1e3, 5000)
        return np.r_[y, a], np.r_[x, b]
    elif name == "pow":
        x, y = logu(rng, 1e-30, 1e30, N), u(-60, 60, N)
        x = np.r_[x, u(0, 100, 5000), near(1.0, 2000, rng)]
        y = np.r_[y, u(-60, 60, 5000), logu(rng, 1, 1e15, 2000, True)]
        neg = -u(0.1, 10, 1000)  # negative base, integer exponent
        x, y = np.r_[x, neg], np.r_[y, rng.integers(-40, 41, 1000).astype(float)]
        keep = np.abs(y * np.log(np.abs(x))) < 709  # results in the double range (with subnormals)
        return x[keep], y[keep]
    elif name == "cabs":  # |re + i im|: random exponents e^+-5 (docs/design/array.md §5.3) and wide ones
        e = np.exp(u(-5, 5, (2, N)))
        s = np.where(rng.random((2, N)) < 0.5, -1.0, 1.0)
        re, im = s * e * u(0.5, 1, (2, N))
        w = logu(rng, 1e-300, 1e300, 5000, True), logu(rng, 1e-300, 1e300, 5000, True)
        t = u(-1e3, 1e3, 2000)
        return np.r_[re, w[0], t, t * (1 + 1e-9 * u(-1, 1, 2000))], np.r_[im, w[1], t[::-1], t]
    else:
        raise ValueError(name)
    return (x,)


ARB = {
    "exp": lambda x: x.exp(),
    "log": lambda x: x.log(),
    "log2": lambda x: x.log() / arb(2).log(),
    "log10": lambda x: x.log() / arb(10).log(),
    "sin": lambda x: x.sin(),
    "asin": lambda x: x.asin(),
    "acos": lambda x: x.acos(),
    "atan": lambda x: x.atan(),
    "sinh": lambda x: x.sinh(),
    "cosh": lambda x: x.cosh(),
    "tanh": lambda x: x.tanh(),
    "asinh": lambda x: x.asinh(),
    "acosh": lambda x: x.acosh(),
    "atanh": lambda x: x.atanh(),
    "atan2": lambda y, x: arb.atan2(y, x),
    "pow": None,
    "cabs": lambda re, im: (
        re * re + im * im
    ).sqrt(),  # _value: x^y = exp(y log x) for x > 0, x^k for a negative x and an integer k
}
FUNCTIONS = tuple(ARB)
EXACT_ONE = {"log", "log2", "log10", "acosh"}  # f(1) = 0 exactly: an exact ball, settled at once


def _value(name, args):
    if name in EXACT_ONE and args[0] == 1.0:
        return 0.0, 0.0
    if name in ("asin", "atan", "sinh", "tanh", "asinh", "atanh", "sin") and args[0] == 0.0:
        return args[0], 0.0
    if name == "pow" and args[1] == 0.0:
        return 1.0, 0.0
    if name == "pow":
        x, y = args
        fn = (lambda: arb(x) ** int(y)) if x < 0 else (lambda: (arb(y) * arb(x).log()).exp())
    else:
        fn = lambda: ARB[name](*(arb(a) for a in args))  # noqa: E731
    v = rigorous(fn, floor=DOUBLE_TINY)
    return to_dd(v)


def _work(job):
    name, chunk = job
    return [_value(name, a) for a in chunk]


def write(name, jobs):
    rng = np.random.default_rng(FUNCTIONS.index(name) + 1)
    cols = [np.asarray(c, dtype=np.float64) for c in inputs(name, rng)]
    pts = list(zip(*(c.tolist() for c in cols)))
    chunks = [pts[i : i + 500] for i in range(0, len(pts), 500)]
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=20) as pool:
        vals = [v for part in pool.map(_work, [(name, c) for c in chunks], chunksize=1) for v in part]
    hi = np.array([h for h, _ in vals])
    lo = np.array([lo for _, lo in vals])
    keys = ("x",) if len(cols) == 1 else {"atan2": ("y", "x"), "cabs": ("re", "im")}.get(name, ("x", "y"))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **dict(zip(keys, cols)), hi=hi, lo=lo)
    print(f"{name}: {len(hi)} points -> {(OUT / f'{name}.npz').relative_to(ROOT)}", flush=True)


# ------------------------------------------------------------------------------ reductions
# (function, kind of x, shape, seed, extra): y, where there is one, is a normal array of the same
# shape from seed + 1. The references are exact: Shewchuk's fsum (math.fsum: the exact sum
# correctly rounded) and a second fsum for its rounding error, giving hi + lo; dot products
# split every product exactly first (Veltkamp); the variance and its mean as rationals; the
# product in Arb at 4096 bits. Each group stores its condition number sum |d log f / d log x_i|.

REDUCTION_GROUPS = [
    ("sum", "uniform", (10**3,), 101, {}),
    ("sum", "uniform", (10**7,), 102, {}),
    ("sum", "normal", (10**5,), 103, {}),
    ("sum", "wide", (10**5,), 104, {}),
    ("sum", "cancel", (10**5,), 105, {}),
    ("sum", "normal", (1000, 1000), 106, {"axis": 0}),
    ("mean", "uniform", (10**6,), 201, {}),
    ("mean", "normal", (10**5,), 202, {}),
    ("mean", "offset", (10**5,), 203, {}),
    *[(f, k, (n,), 300 + i, {"correction": c}) for f in ("var", "std") for i, (k, n) in
      enumerate((("normal", 10**4), ("uniform", 10**5), ("offset", 10**4), ("wide", 10**3))) for c in (0, 1)],
    ("prod", "half_two", (1000,), 401, {}),
    ("prod", "normal", (200,), 402, {}),
    ("vecdot", "normal", (10**3,), 501, {}),
    ("vecdot", "normal", (10**6,), 502, {}),
    ("vecdot", "wide", (10**4,), 503, {}),
    ("matmul", "normal", (8, 8), 601, {}),
    ("matmul", "normal", (64, 64), 602, {}),
    ("matmul", "normal", (300, 300), 603, {}),
]  # fmt: skip


def _fsum_dd(x):
    hi = math.fsum(x)
    return (hi, 0.0) if not math.isfinite(hi) else (hi, math.fsum([*x, -hi]))


def _two_prod(a, b):
    """a * b = p + e exactly (Veltkamp's split; no overflow for these inputs)."""
    p = a * b
    c = 134217729.0
    ah = c * a - (c * a - a)
    bh = c * b - (c * b - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dot_dd(x, y):
    p, e = _two_prod(x, y)
    return _fsum_dd(np.r_[p, e].tolist())


def _frac_dd(q):
    from fractions import Fraction

    h = float(q)
    return h, float(q - Fraction(h))


def _reduction_ref(func, x, y, extra):
    """(hi, lo, cond) arrays."""
    from fractions import Fraction

    if func == "sum" and "axis" in extra:
        hl = [_fsum_dd(x[:, j].tolist()) for j in range(x.shape[1])]
        hi = np.array([h for h, _ in hl])
        return hi, np.array([lo for _, lo in hl]), np.abs(x).sum(axis=0) / np.abs(hi)
    if func == "sum":
        hi, lo = _fsum_dd(x.tolist())
        return np.array([hi]), np.array([lo]), np.array([np.abs(x).sum() / abs(hi)])
    if func == "mean":
        hi, lo = _fsum_dd(x.tolist())
        m = (Fraction(hi) + Fraction(lo)) / x.size  # hi + lo is the sum to 2^-106 relative
        h, low = _frac_dd(m)
        return np.array([h]), np.array([low]), np.array([np.abs(x).sum() / abs(hi)])
    if func in ("var", "std"):
        c, n = extra["correction"], x.size
        fx = [Fraction(v) for v in x.tolist()]
        m = sum(fx, Fraction(0)) / n
        dev = [v - m for v in fx]
        v = sum((d * d for d in dev), Fraction(0)) / (n - c)
        cond = float(sum(abs(2 * a * d) for a, d in zip(fx, dev)) / ((n - c) * v))  # sum |x_i dV/dx_i| / V
        if func == "var":
            h, low = _frac_dd(v)
            return np.array([h]), np.array([low]), np.array([cond])
        from flint import fmpq

        r = rigorous(lambda: arb(fmpq(v.numerator, v.denominator)).sqrt())
        h, low = to_dd(r)
        return np.array([h]), np.array([low]), np.array([cond / 2])
    if func == "prod":
        with precision(4096):
            pr = arb(1)
            for t in x.tolist():
                pr *= arb(t)
        h, low = to_dd(pr)
        return np.array([h]), np.array([low]), np.array([float(x.size)])
    if func == "vecdot":
        hi, lo = _dot_dd(x, y)
        return np.array([hi]), np.array([lo]), np.array([2 * np.abs(x * y).sum() / abs(hi)])
    # matmul
    n = x.shape[0]
    hl = [_dot_dd(x[i], y[:, j]) for i in range(n) for j in range(n)]
    hi = np.array([h for h, _ in hl])
    return hi, np.array([lo for _, lo in hl]), 2 * (np.abs(x) @ np.abs(y)).ravel() / np.abs(hi)


def _reduction_job(i):
    from bench.accuracy.array import reduction_inputs

    func, kind, shape, seed, extra = REDUCTION_GROUPS[i]
    x = reduction_inputs(kind, shape, seed)
    y = reduction_inputs("normal", shape, seed + 1) if func in ("vecdot", "matmul") else None
    return _reduction_ref(func, x, y, extra)


def write_reductions(jobs=10):
    import hashlib

    from bench.accuracy.array import STORE_MAX, reduction_inputs

    order = sorted(range(len(REDUCTION_GROUPS)), key=lambda i: -int(np.prod(REDUCTION_GROUPS[i][2])))
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:
        refs = dict(zip(order, pool.map(_reduction_job, order, chunksize=1)))
    arrays = {}
    for i, (func, kind, shape, seed, extra) in enumerate(REDUCTION_GROUPS):
        x = reduction_inputs(kind, shape, seed)
        y = reduction_inputs("normal", shape, seed + 1) if func in ("vecdot", "matmul") else None
        hi, lo, cond = refs[i]
        dims = "x".join(str(v) for v in shape)
        label = f"{kind} {dims}" + "".join(f" {k}={v}" for k, v in extra.items())
        g = {"func": func, "label": label, "hi": hi, "lo": lo, "cond": cond, **extra}
        for name, v, k, sd_ in (("x", x, kind, seed), ("y", y, "normal", seed + 1)):
            if v is None:
                continue
            if v.size <= STORE_MAX:
                g[name] = v
            else:  # regenerated at measurement time, checked against this digest
                g.update({f"{name}_kind": k, f"{name}_shape": np.array(shape), f"{name}_seed": sd_,
                          f"{name}_sha256": hashlib.sha256(v.tobytes()).hexdigest()})  # fmt: skip
        for k, v in g.items():
            arrays[f"g{i}_{k}"] = np.asarray(v)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "reductions.npz", **arrays)
    print(f"reductions: {len(REDUCTION_GROUPS)} groups -> {(OUT / 'reductions.npz').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("functions", nargs="*", help=", ".join((*FUNCTIONS, "reductions")))
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    bad = set(a.functions) - {*FUNCTIONS, "reductions"}
    if bad:
        ap.error(f"unknown functions {sorted(bad)}; choose from {', '.join(FUNCTIONS)}")
    for f in a.functions or (*FUNCTIONS, "reductions"):
        if f == "reductions":
            write_reductions(min(a.jobs, 10))
        else:
            write(f, min(a.jobs, 10))


if __name__ == "__main__":
    main()
