"""The whole array area against numpy: every public function, at three sizes
(3, 10^4, 10^6 elements), in several layouts (contiguous, strided, transposed,
broadcast) and dtypes, on 1 and 10 threads. numpy runs as it is (one thread,
except its BLAS).

Each case is first run once on both sides and the results compared (shape,
dtype kind, values with NaNs equal), so a fast row cannot hide a different
answer. Times are the best of 7 repeats of a loop of at least 20 ms, and the
best over ROUNDS full passes (the passes are spread over time, so a burst of
background work hurts one pass, not the minimum).

    .venv/bin/python bench/array/full_scan.py --rounds 3 --out bench/array/full_scan
    .venv/bin/python bench/array/full_scan.py --only "sort|argsort"     # a subset, by regex on the name

Writes OUT/full_scan.json (every round) and OUT/full_scan.md (the table).
"""

import argparse
import json
import math
import pathlib
import platform
import re
import subprocess
import time
import timeit

import numpy as np

import stardust as sd

N = 1_000_000
MID = 10_000


# ----------------------------------------------------------------- data


def make_data():
    r = np.random.default_rng(0)
    d = {}
    d["f"] = r.random(N) + 0.5  # positive: log, sqrt
    d["g"] = r.random(N) + 0.5
    d["fs"] = r.standard_normal(N)  # both signs
    d["u"] = r.random(N) * 1.98 - 0.99  # (-1, 1): asin, acos, atanh
    d["fh"] = r.random(N) + 1.5  # > 1: acosh
    d["fn"] = np.where(r.random(N) < 0.1, np.nan, d["fs"])  # 10% NaN
    d["i"] = r.integers(-1000, 1000, N)
    d["j"] = r.integers(1, 1000, N)
    d["sh"] = r.integers(0, 63, N)
    d["b"] = r.random(N) > 0.5
    d["b2"] = r.random(N) > 0.5
    d["c"] = d["f"] + 1j * r.random(N)
    d["cc"] = r.random(N) + 1j * r.random(N)
    d["f2"] = r.random((1000, 1000))
    d["g2"] = r.random((1000, 1000))
    d["i2"] = r.integers(-1000, 1000, (1000, 1000))
    d["v"] = r.random(1000)
    d["t3"] = r.random((100, 100, 100))
    d["big2"] = r.random(2 * N)  # strided views come from here
    d["m512"] = r.random((512, 512))
    d["m1k"] = r.random((1000, 1000))
    d["im"] = r.integers(-100, 100, (200, 200))
    d["bm"] = r.random((1000, 8, 8))
    d["obs"] = r.random((20, 100_000))
    d["iv"] = r.integers(0, 1000, N)
    d["ivw"] = r.integers(-(10**12), 10**12, N)
    d["idx"] = r.integers(0, N, 100_000)
    d["idxn"] = r.integers(0, N, N)
    d["rows"] = r.integers(0, 1000, 1000)
    d["idx2"] = r.integers(0, 1000, (1000, 1000))
    d["srt"] = np.sort(r.random(N))
    d["keys"] = r.random(100_000)
    d["q5"] = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    d["zh"] = np.where(r.random(N) < 0.5, 0.0, r.standard_normal(N)) * np.where(r.random(N) < 0.5, 1, -1)
    d["fm"] = r.random(MID) + 0.5
    d["gm"] = r.random(MID) + 0.5
    for nm, k in (("1e3", 1_000), ("1e5", 100_000), ("1e7", 10_000_000)):
        d["a" + nm] = r.random(k) + 0.5
        d["b" + nm] = r.random(k) + 0.5
        d["i" + nm] = r.integers(-1000, 1000, k)
        d["j" + nm] = r.integers(-1000, 1000, k)
    d["s3"] = np.array([1.0, 2.0, 3.0])
    d["t3s"] = np.array([4.0, 5.0, 6.0])
    d["i3"] = np.array([1, 2, 3])
    d["sm"] = np.arange(12.0).reshape(3, 4)
    return d


def namespaces(d):
    """The same names on both sides; numpy inputs (for conversions) as np_*."""
    npn = {"M": np, "np": np, "sd": sd, "N": N}
    sdn = {"M": sd, "np": np, "sd": sd, "N": N}
    for k, v in d.items():
        npn[k] = v.copy()
        sdn[k] = sd.asarray(v, copy=True)
    for ns, lib in ((npn, np), (sdn, sd)):
        ns["fst"] = ns["big2"][::2]  # strided, 1e6
        ns["ft"] = ns["f2"].T  # transposed
        ns["x"] = lib.asarray(d["f"], copy=True)  # mutated in place
        ns["gz"] = lib.zeros(N)
        ns["fw"] = lib.asarray(d["f"], copy=True)
    # conversions: numpy arrays given to both
    r = np.random.default_rng(1)
    m = r.random((2000, 2000))
    conv = {
        "np_f": d["f"],
        "np_fst": d["big2"][::2],
        "np_ft": d["f2"].T,
        "np_f32": d["f"].astype(np.float32),
        "np_f32t": m.astype(np.float32)[:1000, :1000].T,
        "np_f32r": m.astype(np.float32)[:1000, 1:1001],
        "np_be": d["f"].astype(">f8"),
        "np_be2": r.random((2000, 1000)).astype(">f8")[::2],
        "np_i32": d["i"].astype(np.int32),
        "np_list3": [1.0, 2.0, 3.0],
        "np_list": d["f"][:10_000].tolist(),
    }
    npn.update(conv)
    sdn.update(conv)
    return npn, sdn


# ---------------------------------------------------------------- cases
# (group, name, sd statement, numpy statement or None for the same text with M = np,
#  numpy statement for the result check only, when the timed one may legitimately
#  differ: an unstable sort's tie order, numpy's unsorted unique_values; "skip")


def cases():
    C = []
    add = lambda g, name, s, n=None, chk=None: C.append((g, name, s, n, chk))

    # creation
    for s in [
        "zeros(N)",
        "ones(N)",
        "empty(N)",
        "full(N, 2.5)",
        "zeros_like(f)",
        "ones_like(f)",
        "empty_like(f)",
        "full_like(f, 2.5)",
        "arange(N)",
        "linspace(0.0, 1.0, N)",
        "eye(1000)",
        "meshgrid(v, v)",
        "tril(f2)",
        "triu(f2)",
        "zeros((1000, 1000))",
    ]:
        add("creation", s, "M." + s)
    add("creation", "arange(N, float64)", "sd.arange(N, dtype=sd.float64)", "np.arange(N, dtype=np.float64)")
    add("creation", "asarray(copy=True)", "M.asarray(f, copy=True)")

    # conversion from numpy (sd.asarray(x) against np.asarray(x, dtype=float64): numpy keeps
    # the memory order, stardust makes C order)
    add("convert", "asarray numpy f64 (view)", "sd.asarray(np_f)", "np.asarray(np_f)")
    add("convert", "asarray numpy f64 strided", "sd.asarray(np_fst)", "np.asarray(np_fst)")
    add("convert", "asarray numpy f64 .T", "sd.asarray(np_ft)", "np.asarray(np_ft)")
    add("convert", "asarray numpy f32", "sd.asarray(np_f32)", "np.asarray(np_f32, dtype=np.float64)")
    add("convert", "asarray numpy f32 .T (np keeps F)", "sd.asarray(np_f32t)", "np.asarray(np_f32t, dtype=np.float64)")
    add(
        "convert",
        "asarray numpy f32 .T (np C order)",
        "sd.asarray(np_f32t)",
        "np.ascontiguousarray(np_f32t, dtype=np.float64)",
    )
    add("convert", "asarray numpy f32 rows m[:,1:1001]", "sd.asarray(np_f32r)", "np.asarray(np_f32r, dtype=np.float64)")
    add("convert", "asarray numpy >f8", "sd.asarray(np_be)", "np.asarray(np_be, dtype=np.float64)")
    add("convert", "asarray numpy >f8 2-D [::2]", "sd.asarray(np_be2)", "np.asarray(np_be2, dtype=np.float64)")
    add("convert", "asarray numpy i32", "sd.asarray(np_i32)", "np.asarray(np_i32, dtype=np.int64)")
    add("convert", "asarray list 3", "sd.asarray(np_list3)", "np.asarray(np_list3)")
    add("convert", "asarray list 1e4", "sd.asarray(np_list)", "np.asarray(np_list)")
    add("convert", "from_dlpack numpy", "sd.from_dlpack(np_f)", "np.from_dlpack(np_f)")
    add("convert", "numpy.asarray(sd array)", "np.asarray(f)", "np.asarray(f)")

    # manipulation
    for s in [
        "reshape(f, (1000, 1000))",
        "reshape(ft, (N,))",
        "permute_dims(t3, (2, 0, 1))",
        "matrix_transpose(f2)",
        "moveaxis(t3, 0, -1)",
        "squeeze(M.reshape(f, (1, N)), axis=0)",
        "expand_dims(f, axis=0)",
        "broadcast_to(v, (1000, 1000))",
        "broadcast_arrays(f2, v)",
        "concat([f, g])",
        "concat([f2, g2], axis=1)",
        "stack([f, g])",
        "flip(f)",
        "roll(f, 1)",
        "roll(f2, 1, axis=1)",
        "repeat(f, 2)",
        "tile(v, (1000,))",
        "unstack(f2)",
        "astype(f, M.int64)",
        "astype(i, M.float64)",
        "astype(f, M.complex128)",
        "astype(b, M.float64)",
        "astype(ft, M.float64)",
    ]:
        add("manip", s, "M." + s)
    add("manip", "ravel(ft) (copy)", "sd.reshape(ft, (-1,))", "np.reshape(ft, (-1,))")
    add("manip", "diff(f)", "M.diff(f)")

    # elementwise: binary float
    for op in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "remainder",
        "pow",
        "atan2",
        "maximum",
        "minimum",
        "copysign",
        "hypot",
        "logaddexp",
        "nextafter",
        "equal",
        "less",
        "greater_equal",
    ]:
        add("binary f64", op, f"M.{op}(f, g)")
    for op in ["logical_and", "logical_or", "logical_xor"]:
        add("binary bool", op, f"M.{op}(b, b2)")
    for op in [
        "add",
        "multiply",
        "floor_divide",
        "remainder",
        "maximum",
        "equal",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    ]:
        add("binary i64", op, f"M.{op}(i, j)")
    add("binary i64", "bitwise_left_shift", "M.bitwise_left_shift(i, sh)")
    add("binary i64", "bitwise_right_shift", "M.bitwise_right_shift(i, sh)")
    add("binary i64", "pow(i, 2)", "M.pow(i, 2)")
    for op in ["add", "multiply", "divide", "equal"]:
        add("binary c128", op, f"M.{op}(c, cc)")
    add("binary c128", "pow", "M.pow(c, cc)")
    add("binary mixed", "f * i (cast)", "f * i")
    add("binary mixed", "f * i + g", "f * i + g")
    add("binary mixed", "f + 1.0 (scalar)", "f + 1.0")
    add("binary mixed", "i + 1 (scalar)", "i + 1")
    add("binary mixed", "c * f", "c * f")
    add("binary layout", "f2 + v (broadcast)", "f2 + v")
    add("binary layout", "f2 + v[:, None] (broadcast col)", "f2 + M.reshape(v, (1000, 1))")
    add("binary layout", "ft + 1.0 (transposed)", "ft + 1.0")
    add("binary layout", "ft + f2 (mixed order)", "ft + f2")
    add("binary layout", "fst + 1.0 (strided)", "fst + 1.0")
    add("binary layout", "fst + f (strided + contiguous)", "fst + f")
    add("binary mid", "fm + gm (1e4)", "fm + gm")
    add("binary mid", "fm * 2.0 (1e4)", "fm * 2.0")

    # elementwise: unary
    unary_f = [
        "negative",
        "positive",
        "abs",
        "square",
        "sqrt",
        "sign",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "floor",
        "ceil",
        "trunc",
        "round",
        "isnan",
        "isinf",
        "isfinite",
        "signbit",
        "reciprocal",
    ]
    for op in unary_f:
        add("unary f64", op, f"M.{op}(f)")
    for op in ["asin", "acos", "atanh"]:
        add("unary f64", op, f"M.{op}(u)")
    add("unary f64", "acosh", "M.acosh(fh)")
    add("unary f64", "sign (both signs)", "M.sign(fs)")
    add("unary f64", "exp strided", "M.exp(fst)")
    add("unary f64", "sqrt transposed", "M.sqrt(ft)")
    add("unary f64", "exp (1e4)", "M.exp(fm)")
    for op in ["abs", "exp", "sqrt", "log", "sin", "conj", "real", "imag", "square", "isnan"]:
        add("unary c128", op, f"M.{op}(c)")
    for op in ["abs", "negative", "square", "sign", "bitwise_invert"]:
        add("unary i64", op, f"M.{op}(i)")
    add("unary bool", "logical_not", "M.logical_not(b)")
    add("select", "clip(f, 0.6, 1.2)", "M.clip(f, min=0.6, max=1.2)", "np.clip(f, 0.6, 1.2)")
    add("select", "clip(i, -10, 10)", "M.clip(i, min=-10, max=10)", "np.clip(i, -10, 10)")
    add("select", "where(b, f, g)", "M.where(b, f, g)")
    add("select", "where(b, f, 0.0)", "M.where(b, f, 0.0)")

    # reductions
    for op in ["sum", "prod", "min", "max", "mean", "var", "std", "argmin", "argmax"]:
        add("reduce f64", op, f"M.{op}(f)")
    for op in ["sum", "min", "max", "mean", "var", "argmax"]:
        add("reduce i64", op, f"M.{op}(i)")
    for op in ["sum", "prod", "mean", "var"]:
        add("reduce c128", op, f"M.{op}(c)")
    add("reduce bool", "any", "M.any(b)")
    add("reduce bool", "all", "M.all(b)")
    add("reduce bool", "sum(b)", "M.sum(b)")
    for op in ["sum", "mean", "max", "var", "argmax"]:
        add("reduce axis", f"{op}(f2, axis=0)", f"M.{op}(f2, axis=0)")
        add("reduce axis", f"{op}(f2, axis=1)", f"M.{op}(f2, axis=1)")
    add("reduce axis", "sum(t3, axis=(0, 2))", "M.sum(t3, axis=(0, 2))")
    add("reduce axis", "sum(ft) (transposed)", "M.sum(ft)")
    add("reduce axis", "sum(ft, axis=0)", "M.sum(ft, axis=0)")
    add("reduce axis", "sum(fst) (strided)", "M.sum(fst)")
    add("reduce axis", "sum(i2, axis=0)", "M.sum(i2, axis=0)")
    add("reduce mid", "sum(fm) (1e4)", "M.sum(fm)")
    add("reduce mid", "var(fm) (1e4)", "M.var(fm)")
    add("scan", "cumulative_sum(f)", "M.cumulative_sum(f)")
    add("scan", "cumulative_prod(g)", "M.cumulative_prod(g)")
    add("scan", "cumulative_sum(i)", "M.cumulative_sum(i)")
    add("scan", "cumulative_sum(f2, axis=0)", "M.cumulative_sum(f2, axis=0)")
    add("scan", "cumulative_sum(f2, axis=1)", "M.cumulative_sum(f2, axis=1)")
    add("scan", "cumulative_sum(c)", "M.cumulative_sum(c)")
    for op in ["nansum", "nanprod", "nanmin", "nanmax", "nanmean", "nanvar", "nanstd", "nanargmin", "nanargmax"]:
        add("nan", op, f"M.{op}(fn)")
    add("nan", "nancumulative_sum", "sd.nancumulative_sum(fn)", "np.nancumsum(fn)")
    add("nan", "nancumulative_prod", "sd.nancumulative_prod(fn)", "np.nancumprod(fn)")
    add("nan", "nansum(f2-like, axis=0)", "M.nansum(M.reshape(fn, (1000, 1000)), axis=0)")
    add("count", "count_nonzero(b)", "M.count_nonzero(b)")
    add("count", "count_nonzero(f)", "M.count_nonzero(fs)")
    add("count", "count_nonzero(f2, axis=0)", "M.count_nonzero(f2, axis=0)")

    # statistics
    add("stats", "quantile(f, q5)", "M.quantile(fs, q5)")
    add("stats", "quantile(f, 0.5)", "M.quantile(fs, 0.5)")
    add("stats", "median(f)", "M.median(fs)")
    add("stats", "median(f2, axis=0)", "M.median(f2, axis=0)")
    add("stats", "median(f2, axis=1)", "M.median(f2, axis=1)")
    add("stats", "nanquantile(fn, q5)", "M.nanquantile(fn, q5)")
    add("stats", "nanmedian(fn)", "M.nanmedian(fn)")
    add("stats", "cov(1000 x 1000)", "sd.cov(m1k, correction=1)", "np.cov(m1k)")
    add("stats", "cov(20 x 1e5)", "sd.cov(obs, correction=1)", "np.cov(obs)")
    add("stats", "corrcoef(1000 x 1000)", "M.corrcoef(m1k)")
    add("stats", "bincount(iv)", "M.bincount(iv)")
    add("stats", "bincount(iv, weights)", "M.bincount(iv, weights=f)")
    add("stats", "histogram(f, 100)", "M.histogram(fs, bins=100)")
    add("stats", "histogram(f, 100, range)", "M.histogram(fs, bins=100, range=(-3.0, 3.0))")

    # sorting and searching
    add("sort", "sort(f)", "M.sort(fs)")
    add("sort", "sort(f, stable=False)", "M.sort(fs, stable=False)")
    add("sort", "sort(half zeros)", "M.sort(zh)")
    add("sort", "sort(i)", "M.sort(i)")
    add("sort", "sort(i wide)", "M.sort(ivw)")
    add("sort", "sort(f2, axis=1)", "M.sort(f2, axis=1)")
    add("sort", "sort(f2, axis=0)", "M.sort(f2, axis=0)")
    add("sort", "sort(f, descending)", "sd.sort(fs, descending=True)", "np.sort(fs)[::-1].copy()")
    add("sort", "argsort(f)", "M.argsort(fs)")
    add("sort", "argsort(f, stable=False)", "M.argsort(fs, stable=False)")
    add("sort", "argsort(i)", "M.argsort(i)", None, "np.argsort(i, stable=True)")
    add("sort", "argsort(i, stable=False)", "M.argsort(i, stable=False)", None, "np.argsort(i, stable=True)")
    add("sort", "argsort(f2, axis=1)", "M.argsort(f2, axis=1)")
    add("sort", "sort(fm) (1e4)", "M.sort(fm)")
    add("sort", "argsort(fm) (1e4)", "M.argsort(fm)")
    add("unique", "unique_values(iv)", "M.unique_values(iv)", None, "np.unique(iv)")
    add("unique", "unique_counts(iv)", "M.unique_counts(iv)")
    add("unique", "unique_inverse(iv)", "M.unique_inverse(iv)")
    add("unique", "unique_all(iv)", "M.unique_all(iv)")
    add("unique", "unique_values(f)", "M.unique_values(fs)", None, "np.unique(fs)")
    add("search", "searchsorted(srt, keys)", "M.searchsorted(srt, keys)")
    add("search", "searchsorted(srt, f)", "M.searchsorted(srt, f)")
    add("search", "nonzero(b)", "M.nonzero(b)")
    add("search", "nonzero(f2 > 0.5)", "M.nonzero(f2 > 0.5)")
    add("search", "take(f, idx)", "M.take(f, idx)")
    add("search", "take(f2, rows, axis=0)", "M.take(f2, rows, axis=0)")
    add("search", "take_along_axis(f2, idx2, axis=1)", "M.take_along_axis(f2, idx2, axis=1)")

    # indexing (operators)
    add("index", "f[1000:2000] (view)", "f[1000:2000]")
    add("index", "f[::2] (view)", "f[::2]")
    add("index", "f2[5] (row view)", "f2[5]")
    add("index", "f2[:, 5] (column)", "f2[:, 5]")
    add("index", "f2[10, 20] (scalar)", "f2[10, 20]")
    add("index", "f[idx] (gather 1e5)", "f[idx]")
    add("index", "f[idxn] (gather 1e6)", "f[idxn]")
    add("index", "f[b] (mask)", "f[b]")
    add("index", "f2[rows] (fancy rows)", "f2[rows]")
    add("index", "f2[f2 > 0.5] (2-D mask)", "f2[f2 > 0.5]")
    add("index", "s3[1] (tiny scalar)", "s3[1]")
    add("assign", "gz[idx] = 1.0 (scatter 1e5)", "gz[idx] = 1.0")
    add("assign", "gz[idxn] = 1.0 (scatter 1e6)", "gz[idxn] = 1.0")
    add("assign", "fw[b] = 0.0 (mask)", "fw[b] = 0.0")
    add("assign", "fw[:] = 1.0 (fill)", "fw[:] = 1.0")
    add("assign", "fw[:] = g (copy)", "fw[:] = g")
    add("assign", "fw[::2] = 1.0 (strided fill)", "fw[::2] = 1.0")
    add("assign", "x += g (in place)", "x.__iadd__(g)", None, "skip")
    add("assign", "x[1:] += x[:-1] (overlap)", "x[1:] += x[:-1]")
    add("assign", "x *= 1.0000001", "x.__imul__(1.0000001)", None, "skip")

    # linear algebra in the array area
    add("matmul", "matmul 512", "M.matmul(m512, m512)")
    add("matmul", "matmul 1000", "M.matmul(m1k, m1k)")
    add("matmul", "matmul(f2, v) (gemv)", "M.matmul(f2, v)")
    add("matmul", "matmul(v, v) (dot)", "M.matmul(v, v)")
    add("matmul", "matmul batch 1000 x 8x8", "M.matmul(bm, bm)")
    add("matmul", "matmul int 200", "M.matmul(im, im)")
    add("matmul", "matmul(ft, f2) (transposed)", "M.matmul(ft, f2)")
    add("matmul", "vecdot(f2, g2)", "M.vecdot(f2, g2)")
    add("matmul", "tensordot(t3, t3, 2)", "M.tensordot(t3, t3, axes=2)")

    # the same operations across sizes (cache-resident to memory-bound; 1e7 is 80 MB)
    for nm in ("1e3", "1e5", "1e7"):
        add(f"size {nm}", f"a + b ({nm})", f"a{nm} + b{nm}")
        add(f"size {nm}", f"a * 2.0 ({nm})", f"a{nm} * 2.0")
        add(f"size {nm}", f"exp(a) ({nm})", f"M.exp(a{nm})")
        add(f"size {nm}", f"sin(a) ({nm})", f"M.sin(a{nm})")
        add(f"size {nm}", f"sum(a) ({nm})", f"M.sum(a{nm})")
        add(f"size {nm}", f"max(a) ({nm})", f"M.max(a{nm})")
        add(f"size {nm}", f"var(a) ({nm})", f"M.var(a{nm})")
        add(f"size {nm}", f"cumulative_sum(a) ({nm})", f"M.cumulative_sum(a{nm})")
        add(f"size {nm}", f"a < b ({nm})", f"a{nm} < b{nm}")
        add(f"size {nm}", f"a[a > 1.0] ({nm})", f"a{nm}[a{nm} > 1.0]")
        add(f"size {nm}", f"where(a > 1.0, a, b) ({nm})", f"M.where(a{nm} > 1.0, a{nm}, b{nm})")
        add(f"size {nm}", f"sort(a) ({nm})", f"M.sort(a{nm})")
        add(f"size {nm}", f"argsort(a) ({nm})", f"M.argsort(a{nm})")
        add(f"size {nm}", f"astype int ({nm})", f"M.astype(a{nm}, M.int64)")
        add(f"size {nm}", f"copy ({nm})", f"M.asarray(a{nm}, copy=True)")
        add(f"size {nm}", f"i + j ({nm})", f"i{nm} + j{nm}")
        add(f"size {nm}", f"sum(i) ({nm})", f"M.sum(i{nm})")
        add(f"size {nm}", f"sort(i) ({nm})", f"M.sort(i{nm})")
        add(f"size {nm}", f"median(a) ({nm})", f"M.median(a{nm})")
        add(f"size {nm}", f"unique_values(i) ({nm})", f"M.unique_values(i{nm})", None, f"np.unique(i{nm})")

    # per-call overhead (tiny arrays)
    add("tiny", "zeros(3)", "M.zeros(3)")
    add("tiny", "empty(3)", "M.empty(3)")
    add("tiny", "asarray(s3) (no-op)", "M.asarray(s3)")
    add("tiny", "s3 + t3s", "s3 + t3s")
    add("tiny", "s3 * 2.0", "s3 * 2.0")
    add("tiny", "i3 + 1", "i3 + 1")
    add("tiny", "sqrt(s3)", "M.sqrt(s3)")
    add("tiny", "exp(s3)", "M.exp(s3)")
    add("tiny", "s3 < t3s", "s3 < t3s")
    add("tiny", "sum(s3)", "M.sum(s3)")
    add("tiny", "max(s3)", "M.max(s3)")
    add("tiny", "mean(s3)", "M.mean(s3)")
    add("tiny", "sum(sm, axis=0)", "M.sum(sm, axis=0)")
    add("tiny", "sort(s3)", "M.sort(s3)")
    add("tiny", "reshape(sm, (4, 3))", "M.reshape(sm, (4, 3))")
    add("tiny", "sm.T", "sm.T")
    add("tiny", "sm[1]", "sm[1]")
    add("tiny", "sm[1:, ::2]", "sm[1:, ::2]")
    add("tiny", "concat([s3, t3s])", "M.concat([s3, t3s])")
    add("tiny", "where(s3 > 1, s3, t3s)", "M.where(s3 > 1.5, s3, t3s)")
    add("tiny", "matmul(sm.T, sm)", "M.matmul(sm.T, sm)")
    add("tiny", "result_type(f, i)", "M.result_type(f, i)")
    add("tiny", "can_cast(int64, float64)", "M.can_cast(M.int64, M.float64)")
    add("tiny", "finfo(float64)", "M.finfo(M.float64)", None, "skip")
    add("tiny", "s3.shape", "s3.shape")
    add("tiny", "len(s3)", "len(s3)")
    add("tiny", "float(s3[0])", "float(s3[0])")
    return C


# ------------------------------------------------------------- checking


def to_np(x):
    if isinstance(x, (tuple, list)) and not isinstance(x, str):
        return [to_np(e) for e in x]
    if hasattr(x, "_fields"):
        return [to_np(e) for e in x]
    if isinstance(x, sd.ndarray):
        return np.asarray(x)
    return x


def same(a, b):
    """'ok' or what differs: shape, kind, values (NaNs equal, a few ULP allowed)."""
    if a is None and b is None:
        return "ok"
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return "structure"
        for x, y in zip(a, b):
            r = same(x, to_np(y) if not isinstance(y, np.ndarray) else y)
            if r != "ok":
                return r
        return "ok"
    if hasattr(b, "_fields") or isinstance(b, tuple):
        return same(a if isinstance(a, list) else [a], list(b))
    if type(a).__name__ in ("finfo", "iinfo") or isinstance(a, (bool, type)) or not hasattr(np.asarray(a), "dtype"):
        return "ok"
    try:
        x, y = np.asarray(a), np.asarray(b)
    except Exception:
        return "ok"
    if x.dtype.kind in "OUS" or y.dtype.kind in "OUS":
        return "ok"
    if x.shape != y.shape:
        return f"shape {x.shape} vs {y.shape}"
    if x.dtype.kind != y.dtype.kind and not (x.dtype.kind in "iu" and y.dtype.kind in "iu"):
        return f"kind {x.dtype} vs {y.dtype}"
    if x.dtype.kind in "fc":
        if not np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True):
            return "values"
    elif not np.array_equal(x, y):
        return "values"
    return "ok"


# ------------------------------------------------------------- timing

MIN_LOOP = 0.02  # seconds per timed loop


def best(stmt, ns, repeat):
    t = timeit.Timer(stmt, globals=ns)
    n = 1
    while True:  # a loop of at least MIN_LOOP
        dt = t.timeit(n)
        if dt >= MIN_LOOP or n >= 1 << 22:
            break
        n *= max(2, min(10, int(1.2 * MIN_LOOP / max(dt, 1e-9)) + 1))
    return min(t.repeat(repeat=repeat, number=n)) / n


def run_case(case, npn, sdn, repeat, check):
    g, name, s, n, chk = case
    npstmt = n if n is not None else s
    row = {"group": g, "name": name}
    if check:
        try:
            is_expr = (
                "=" not in s.replace("==", "").replace("<=", "").replace(">=", "").replace("!=", "")
                or "(" in s.split("=")[0]
            )
            if chk == "skip":
                row["check"] = "ok"
            elif is_expr:
                a = to_np(eval(s, sdn))
                b = eval(chk or npstmt, npn)
                row["check"] = same(a, b)
            else:
                row["check"] = "stmt"
        except Exception as e:
            row["check"] = f"error {type(e).__name__}: {e}"[:120]
            return row
    try:
        row["np"] = best(npstmt, npn, repeat)
        sd.set_num_threads(1)
        row["sd1"] = best(s, sdn, repeat)
        sd.set_num_threads(10)
        row["sd10"] = best(s, sdn, repeat)
    except Exception as e:
        row["check"] = f"error {type(e).__name__}: {e}"[:120]
    return row


def fmt_t(t):
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "-"
    if t < 1e-6:
        return f"{t * 1e9:.0f} ns"
    if t < 1e-3:
        return f"{t * 1e6:.1f} µs"
    return f"{t * 1e3:.2f} ms"


def table(rows, env):
    out = [
        "# 배열 영역 전체 측정 (numpy 대비)",
        "",
        env,
        "",
        "배율 = numpy 시간 ÷ stardust 시간 (1보다 크면 stardust가 빠르다). numpy는 1스레드(BLAS만 여러 스레드).",
        "시간은 7번 반복 중 최솟값을 다시 라운드 사이 최솟값으로 잡았다. **굵은 배율**은 1스레드에서 numpy보다 느린 줄이다.",
        "",
    ]
    slow1 = [r for r in rows if r.get("sd1") and r["np"] / r["sd1"] < 1.0]
    slow10 = [r for r in rows if r.get("sd10") and r["np"] / r["sd10"] < 1.0]
    bad = [r for r in rows if r.get("check") not in ("ok", "stmt")]
    out += [
        f"- 케이스 {len(rows)}개. 1스레드에서 numpy보다 느린 것 {len(slow1)}개, 10스레드에서도 느린 것 {len(slow10)}개.",
        f"- 결과가 numpy와 다른 것: {len(bad)}개" + ("" if not bad else " (아래 '결과 확인' 열)"),
        "",
    ]
    if slow1:
        out += [
            "## 1스레드에서 numpy보다 느린 것 (배율 낮은 순)",
            "",
            "| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(slow1, key=lambda r: r["np"] / r["sd1"]):
            out.append(
                f"| {r['group']} | `{r['name']}` | {fmt_t(r['np'])} | {fmt_t(r['sd1'])} | {r['np'] / r['sd1']:.2f} | {fmt_t(r['sd10'])} | {r['np'] / r['sd10']:.2f} |"
            )
        out.append("")
    out += [
        "## 전체",
        "",
        "| 묶음 | 케이스 | numpy | sd 1T | 배율 1T | sd 10T | 배율 10T | 결과 확인 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if not r.get("sd1"):
            out.append(f"| {r['group']} | `{r['name']}` | - | - | - | - | - | {r.get('check')} |")
            continue
        q1, q10 = r["np"] / r["sd1"], r["np"] / r["sd10"]
        b1 = f"**{q1:.2f}**" if q1 < 1 else f"{q1:.2f}"
        out.append(
            f"| {r['group']} | `{r['name']}` | {fmt_t(r['np'])} | {fmt_t(r['sd1'])} | {b1} | {fmt_t(r['sd10'])} | {q10:.2f} | {r.get('check')} |"
        )
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="bench/array/full_scan")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allc = cases()
    if args.only:
        allc = [c for c in allc if re.search(args.only, c[1])]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
    ).stdout.strip()
    env = (
        f"환경: {platform.machine()} macOS, Python {platform.python_version()}, numpy {np.__version__}, "
        f"stardust {commit}{' + 작업 중 변경' if dirty else ''}, 원소 {N:,}개 기준, {args.rounds}라운드, "
        f"{time.strftime('%Y-%m-%d %H:%M')}"
    )
    print(env, flush=True)
    d = make_data()
    rounds = []
    t0 = time.time()
    for r in range(args.rounds):
        npn, sdn = namespaces(d)  # fresh (mutated arrays reset)
        rows = []
        for k, c in enumerate(allc):
            row = run_case(c, npn, sdn, args.repeat, check=(r == 0))
            rows.append(row)
            if k % 25 == 0:
                print(f"round {r + 1} case {k + 1}/{len(allc)} {c[1]}  ({time.time() - t0:.0f} s)", flush=True)
        rounds.append(rows)
        (out / "full_scan.json").write_text(json.dumps({"env": env, "rounds": rounds}, indent=1))
    merged = []
    for k in range(len(allc)):
        base = dict(rounds[0][k])
        for key in ("np", "sd1", "sd10"):
            vals = [rr[k][key] for rr in rounds if key in rr[k]]
            if vals:
                base[key] = min(vals)
        merged.append(base)
    (out / "full_scan.md").write_text(table(merged, env + f", 걸린 시간 {(time.time() - t0) / 60:.0f}분"))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
