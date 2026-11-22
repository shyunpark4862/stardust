"""Probes of sd.interpolate's foundations and of the libraries it is compared with
(docs/design/interpolate.md §3.2, §3.3, §5, §7.4). The records the tests read go to
tests/interpolate/data/; the kernel measurements, the elimination growth and the timings go to the
terminal and to build/logs/interpolate_probe/<stamp>.log, and docs/design/interpolate.md quotes them.

    .venv/bin/python tools/interpolate_probe.py                    # every section
    .venv/bin/python tools/interpolate_probe.py numpy scipy r      # only these sections

Sections (positional; default all):
  kernels  the four linear-kernel candidates of §9 question 1 and the branchless form of the chosen one
           (tools/interpolate_probe_kernels.cpp): the exact ends, a == b, monotonicity and range on 3000
           random intervals, and the time of a numpy.interp-like loop with each (and of the xp check of
           decision IP5)
  banded   the growth of elimination without pivoting on the spline systems of the real curves, in
           exact rationals: max (|L||U|)_ij / |A_ij| over the nonzeros (CubicSpline's natural, clamped
           and not-a-knot systems; make_interp_spline's collocation for k = 3 and 5, natural and clamped,
           the latter also with each derivative row after the interpolation row at its end) -- the plan
           of the proof of decision IP16 (§3.3, §3.4)
  numpy    numpy 2.5.4's interp at its edges (unsorted and repeated xp, non-finite fp, period, left and
           right, signed zeros, the scalar type) and the formula its wheel computes -> numpy.json
  scipy    scipy 1.18.1's interpolants at their edges (defaults, errors, special cases, the behaviours
           §2.9 departs from) and their values on the test data (curves, knots, grids, polynomials,
           RBF, roots, AAA) -> scipy.json
  r        R 4.6.2's stats::approx (rule 1 and 2) and splinefun (natural, periodic, fmm) on the real
           curves, values as hex floats -> r.json
  time     the scipy, numpy and R timings of §5 (best of three unless marked; one thread for the
           kernels that have no threads)

The C++ program is compiled into build/interpolate_probe/ with sd's flags (-O3 -mcpu=native
-ffp-contract=off) under the shared build lock (build/.agent-build-lock). Every child process runs
under a time limit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import warnings
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench import cli  # noqa: E402

BUILD = ROOT / "build" / "interpolate_probe"
DATA = ROOT / "tests" / "interpolate" / "data"
LOCK = ROOT / "build" / ".agent-build-lock"
FLAGS = ["-std=c++20", "-O3", "-mcpu=native", "-ffp-contract=off"]
SRC = ROOT / "tools" / "interpolate_probe_kernels.cpp"
SECTIONS = ("kernels", "banded", "numpy", "scipy", "r", "time")
CHILD_SECONDS = 1200
CUBIC_BC = ("natural", "clamped", "not-a-knot")
BSPLINE = (("k0", 0, None), ("k1", 1, None), ("k2", 2, None), ("k3", 3, None), ("k4", 4, None), ("k5", 5, None),
           ("k3_natural", 3, "natural"), ("k3_clamped", 3, "clamped"), ("k3_periodic", 3, "periodic"),
           ("k5_periodic", 5, "periodic"))  # fmt: skip


def S():
    from tests.interpolate import samples

    return samples


class Log:
    def __init__(self):
        self.path = cli.log_file("interpolate_probe")
        print(f"log: {self.path.relative_to(ROOT)}", flush=True)

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as f:
            f.write(text + "\n")


def load(arg):
    return f"load average {os.getloadavg()[0]:.1f} {arg}"


# ---------------------------------------------------------------- the C++ program


def build() -> pathlib.Path:
    exe = BUILD / "kernels"
    if exe.exists() and exe.stat().st_mtime >= SRC.stat().st_mtime:
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    try:
        subprocess.run(["clang++", *FLAGS, "-o", str(exe), str(SRC)], check=True, timeout=600)
    finally:
        LOCK.rmdir()
    return exe


def run(cmd) -> str:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=CHILD_SECONDS)
    out = p.stdout
    if p.returncode:
        out += f"(exit status {p.returncode})\n{p.stderr[-2000:]}"
    return out


def kernels(log):
    exe = build()
    log(f"## kernels ({load('at start')})")
    log(run([exe, "props"]))
    log(run([exe, "time"]))


# ---------------------------------------------------------------- elimination growth


def _growth(rows):
    """max over the nonzeros of (|L||U|)_ij / |A_ij| for elimination without pivoting (exact), and
    whether a zero pivot appeared."""
    n = len(rows)
    A = [list(r) for r in rows]
    L = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    U = [list(r) for r in A]
    for k in range(n):
        if U[k][k] == 0:
            return math.inf
        for i in range(k + 1, n):
            if U[i][k] == 0:
                continue
            m = U[i][k] / U[k][k]
            L[i][k] = m
            for j in range(k, n):
                U[i][j] -= m * U[k][j]
    worst = Fraction(0)
    for i in range(n):
        for j in range(n):
            if A[i][j] == 0:
                continue
            s = sum((abs(L[i][k]) * abs(U[k][j]) for k in range(min(i, j) + 1)), Fraction(0))
            worst = max(worst, s / abs(A[i][j]))
    return float(worst)


def _cubic_rows(X, bc):
    """CubicSpline's system (scipy's rows) as a dense rational matrix."""
    n = len(X)
    h = [X[i + 1] - X[i] for i in range(n - 1)]
    rows = [[Fraction(0)] * n for _ in range(n)]
    for i in range(1, n - 1):
        rows[i][i - 1], rows[i][i], rows[i][i + 1] = h[i], 2 * (h[i - 1] + h[i]), h[i - 1]
    if bc == "not-a-knot":
        rows[0][0], rows[0][1] = h[1], X[2] - X[0]
        rows[-1][-1], rows[-1][-2] = h[-2], X[-1] - X[-3]
    elif bc == "natural":
        rows[0][0], rows[0][1] = 2 * h[0], h[0]
        rows[-1][-1], rows[-1][-2] = 2 * h[-1], h[-1]
    else:
        rows[0][0] = Fraction(1)
        rows[-1][-1] = Fraction(1)
    return rows


def banded(log):
    import interpolate_ref as R

    M = S()
    log(f"## banded: growth max (|L||U|)_ij / |A_ij| of elimination without pivoting ({load('at start')})")
    worst = {}
    for cid in M.real_curve_ids():
        c = M.curve(cid)
        if c.n < 4:
            continue
        X = [Fraction(float(v)) for v in c.x]
        line = [f"  {cid:18s}"]
        for bc in CUBIC_BC:
            g = _growth(_cubic_rows(X, bc))
            worst[bc] = max(worst.get(bc, 0), g)
            line.append(f"{bc} {g:.4g}")
        for name, k, bc in (("k3", 3, None), ("k5", 5, None), ("k3_natural", 3, "natural"),
                            ("k3_clamped", 3, "clamped")):  # fmt: skip
            if c.n < k + 1:
                continue
            _, rows, _ = R.bspline_system(c.x, c.y, k, name)
            g = _growth(rows)
            worst[name] = max(worst.get(name, 0), g)
            line.append(f"{name} {g:.4g}")
            if bc:  # the derivative row after the interpolation row at each end
                g = _growth([rows[1], rows[0], *rows[2:-2], rows[-1], rows[-2]])
                worst[name + "_reordered"] = max(worst.get(name + "_reordered", 0), g)
                line.append(f"reordered {g:.4g}")
        log("  ".join(line))
    log("  largest: " + ", ".join(f"{k} {v:.4g}" for k, v in worst.items()))


# ---------------------------------------------------------------- records


def outcome(fn):
    """{"value": ...} or {"error": type, "msg": message}, and the warnings fn() raised."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            v = fn()
        except Exception as e:  # noqa: BLE001 -- recording what the library raises
            out = {"error": type(e).__name__, "msg": str(e)[:300]}
        else:
            out = {"value": plain(v)}
    if w:
        out["warnings"] = [f"{x.category.__name__}: {str(x.message)[:200]}" for x in w]
    return out


def plain(v):
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            return {"real": v.real.tolist(), "imag": v.imag.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
        if v.dtype == object:
            return {"object": [plain(x) for x in v.tolist()], "shape": list(v.shape)}
        return {"array": v.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, (np.floating,)):
        return {"np_float": float(v)}
    if isinstance(v, complex):
        return {"complex": [v.real, v.imag]}
    if isinstance(v, (np.complexfloating,)):
        return {"np_complex": [float(v.real), float(v.imag)]}
    if isinstance(v, (np.integer,)):
        return {"np_int": int(v)}
    if isinstance(v, float):
        return v
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def write_records(name, meta, records):
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in records.items()]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return f"{len(records)} records -> {(DATA / name).relative_to(ROOT)}"


def numpy_records(log):
    M = S()
    rec = {}
    interp = np.interp
    rec["unsorted"] = outcome(lambda: interp([0.5, 1.5, 2.5], [0, 2, 1, 3], [0, 10, 20, 30]))
    rec["repeated"] = outcome(lambda: interp([0.5, 1.0, 1.5, 2.0], [0, 1, 1, 2], [0, 1, 5, 6]))
    rec["repeated_end"] = outcome(lambda: interp([1.0, 1.5], [0, 1, 1], [0, 1, 5]))
    for name, fp in (("inf_1", [math.inf, 1.0]), ("1_inf", [1.0, math.inf]), ("inf_inf", [math.inf, math.inf]),
                     ("-inf_inf", [-math.inf, math.inf]), ("nan_1", [math.nan, 1.0]), ("1_nan", [1.0, math.nan]),
                     ("-inf_1", [-math.inf, 1.0])):  # fmt: skip
        rec[f"nonfinite/{name}"] = outcome(lambda fp=fp: interp([0.0, 0.25, 0.5, 1.0], [0.0, 1.0], fp))
    rec["nan_x"] = outcome(lambda: interp([math.nan, 0.5], [0.0, 1.0], [0.0, 1.0]))
    rec["nan_xp"] = outcome(lambda: interp([0.5, 1.5], [0.0, math.nan, 2.0], [0.0, 1.0, 2.0]))
    rec["left_right"] = outcome(lambda: interp([-1.0, 2.0], [0.0, 1.0], [3.0, 4.0], left=-7.0, right=9.0))
    rec["positional_left_right"] = outcome(lambda: interp([-1.0, 2.0], [0.0, 1.0], [3.0, 4.0], -7.0, 9.0))
    rec["one_point"] = outcome(lambda: interp([-1.0, 0.0, 1.0], [0.0], [5.0]))
    rec["empty"] = outcome(lambda: interp([0.5], [], []))
    rec["length_mismatch"] = outcome(lambda: interp([0.5], [0.0, 1.0], [1.0]))
    rec["xp_2d"] = outcome(lambda: interp([0.5], [[0.0, 1.0]], [[1.0, 2.0]]))
    rec["complex"] = outcome(lambda: interp([0.25, 2.0], [0.0, 1.0], [1 + 2j, 3 - 1j]))
    rec["complex_x"] = outcome(lambda: interp([0.25 + 1j], [0.0, 1.0], [1.0, 2.0]))
    rec["scalar_type"] = {"type": type(interp(0.5, [0.0, 1.0], [0.0, 1.0])).__name__}
    for name, kw in (("period0", {"period": 0}), ("period_nan", {"period": math.nan}),
                     ("period_inf", {"period": math.inf}), ("period_neg", {"period": -360.0}),
                     ("period_left", {"period": 360.0, "left": -1.0})):  # fmt: skip
        rec[name] = outcome(lambda kw=kw: interp([-180.0, -10.0, 5.0, 190.0, 370.0], [-170.0, 10.0, 170.0, 350.0],
                                                 [0.0, 1.0, 2.0, 3.0], **kw))  # fmt: skip
    rec["period_unsorted"] = outcome(lambda: interp([5.0, 175.0], [190.0, -10.0, 10.0], [1.0, 2.0, 3.0], period=360))
    zeros = []
    for a, b in ((-0.0, -0.0), (0.0, -0.0), (-0.0, 0.0), (0.0, 0.0)):
        v = interp([0.0, 0.5, 1.0], [0.0, 1.0], [a, b])
        zeros.append([math.copysign(1.0, float(x)) for x in v])
    rec["signed_zeros"] = {"ab": [[-0.0, -0.0], [0.0, -0.0], [-0.0, 0.0], [0.0, 0.0]], "signs": zeros}
    # the formula numpy's wheel computes: fma(s, x - xp[j], fp[j]) with s = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
    g = M.rng("numpy-formula")
    xp = np.sort(g.uniform(-10, 10, 2001))
    fp = g.standard_normal(2001) * 10.0 ** g.integers(-3, 4, 2001)
    x = g.uniform(xp[0], xp[-1], 40000)
    got = interp(x, xp, fp)
    j = np.searchsorted(xp, x, side="right") - 1
    s = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    fused = np.array([math.fma(si, xi - xa, fa) for si, xi, xa, fa in zip(s, x, xp[j], fp[j])])
    plain_ = s * (x - xp[j]) + fp[j]
    rec["formula"] = {
        "digest": M.digest(xp, fp, x),
        "points": int(x.size),
        "equal_fma": int(np.sum(got == fused)),
        "equal_unfused": int(np.sum(got == plain_)),
    }
    # the node on the right: the double just left of x1 on independent intervals [x0, x1] whose width is
    # about their position (so the step to that double is about an ulp of the width), against [a, b]
    x0 = g.uniform(0, 1, 20000)
    x1 = x0 + g.uniform(0.5, 2.0, 20000)
    a = g.standard_normal(20000)
    b = a + g.exponential(1.0, 20000) * np.where(g.random(20000) < 0.5, 1.0, -1.0)
    left = np.nextafter(x1, -np.inf)
    v = np.array([interp(q, [p0, p1], [fa, fb]) for q, p0, p1, fa, fb in zip(left, x0, x1, a, b)])
    over = int(np.sum((v > np.maximum(a, b)) | (v < np.minimum(a, b))))
    at_x1 = np.array([interp(p1, [p0, p1], [fa, fb]) for p0, p1, fa, fb in zip(x0, x1, a, b)])
    rec["right_node"] = {"digest": M.digest(x0, x1, a, b), "intervals": 20000, "out_of_range_left_of_x1": over,
                         "x1_exact": int(np.sum(at_x1 == b))}  # fmt: skip
    # values on the real curves (clamped outside)
    for cid in M.real_curve_ids():
        c = M.curve(cid)
        rec[f"interp/{cid}"] = {"digest": c.digest, "values": interp(c.xe, c.x, c.y).tolist()}
    meta = {"tool": "tools/interpolate_probe.py", "numpy": np.__version__}
    log(write_records("numpy.json", meta, rec))


def _scipy_make(si, method, c):
    if method in ("natural", "clamped", "not_a_knot", "periodic"):
        return si.CubicSpline(c.x, c.y, bc_type=method.replace("_", "-"))
    if method == "pchip":
        return si.PchipInterpolator(c.x, c.y)
    if method in ("akima", "makima"):
        return si.Akima1DInterpolator(c.x, c.y, method=method, extrapolate=True)
    _, k, bc = next(b for b in BSPLINE if b[0] == method)
    return si.make_interp_spline(c.x, c.y, k=k, bc_type=bc)


def scipy_records(log):
    import scipy
    import scipy.interpolate as si

    M = S()
    rec = {}
    x5, y5 = np.array([0.0, 1.0, 2.5, 3.0, 4.5]), np.array([1.0, -1.0, 0.5, 2.0, 0.0])
    # defaults of extrapolate
    rec["extrapolate/CubicSpline"] = outcome(lambda: si.CubicSpline(x5, y5).extrapolate)
    rec["extrapolate/CubicSpline_periodic"] = outcome(
        lambda: si.CubicSpline(x5, np.r_[y5[:-1], y5[0]], bc_type="periodic").extrapolate
    )
    rec["extrapolate/Pchip"] = outcome(lambda: si.PchipInterpolator(x5, y5).extrapolate)
    rec["extrapolate/Akima"] = outcome(lambda: si.Akima1DInterpolator(x5, y5).extrapolate)
    rec["extrapolate/Akima_outside"] = outcome(lambda: si.Akima1DInterpolator(x5, y5)([-1.0, 5.0]))
    rec["extrapolate/make_interp_spline"] = outcome(lambda: si.make_interp_spline(x5, y5).extrapolate)
    rec["extrapolate/make_interp_spline_periodic"] = outcome(
        lambda: si.make_interp_spline(x5, np.r_[y5[:-1], y5[0]], bc_type="periodic").extrapolate
    )
    rec["extrapolate/CubicHermiteSpline"] = outcome(lambda: si.CubicHermiteSpline(x5, y5, y5).extrapolate)
    # the input checks
    rec["unsorted/CubicSpline"] = outcome(lambda: si.CubicSpline([0.0, 2.0, 1.0], [1.0, 2.0, 3.0]))
    rec["unsorted/make_interp_spline"] = outcome(lambda: si.make_interp_spline([0.0, 2.0, 1.0, 3.0], [1, 2, 3, 4]))
    rec["duplicate_x/Pchip"] = outcome(lambda: si.PchipInterpolator([0.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    rec["decreasing/PPoly"] = outcome(lambda: si.PPoly(np.ones((2, 2)), [2.0, 1.0, 0.0])(1.5))
    for name, make in (("CubicSpline", lambda y: si.CubicSpline(x5, y)),
                       ("Pchip", lambda y: si.PchipInterpolator(x5, y)),
                       ("Akima", lambda y: si.Akima1DInterpolator(x5, y)),
                       ("make_interp_spline", lambda y: si.make_interp_spline(x5, y)),
                       ("CubicHermiteSpline", lambda y: si.CubicHermiteSpline(x5, y, np.ones(5))),
                       ("RBF", lambda y: si.RBFInterpolator(x5[:, None], y)(np.array([[1.5]]))),
                       ("PPoly", lambda y: si.PPoly(y[None, :4], x5)(np.array([0.5, 1.5]))),
                       ("BSpline", lambda y: si.BSpline(np.r_[0, 0, x5[1:-1], 5, 5], y, 1)(np.array([0.5, 1.5]))),
                       ("Barycentric", lambda y: si.BarycentricInterpolator(x5, y)(np.array([0.5]))),
                       ("Krogh", lambda y: si.KroghInterpolator(x5, y)(np.array([0.5]))),
                       ("interp1d", lambda y: si.interp1d(x5, y)(np.array([0.5])))):  # fmt: skip
        rec[f"nan_y/{name}"] = outcome(lambda make=make: make(np.r_[y5[:2], math.nan, y5[3:]]))
    rec["complex/Pchip"] = outcome(lambda: si.PchipInterpolator(x5, y5 + 1j))
    rec["complex/Akima"] = outcome(lambda: si.Akima1DInterpolator(x5, y5 + 1j))
    rec["complex/CubicSpline"] = outcome(lambda: si.CubicSpline(x5, y5 + 1j)(1.5))
    rec["complex/RGI_pchip"] = outcome(lambda: si.RegularGridInterpolator((x5,), y5 + 1j, method="pchip"))
    rec["complex_x/CubicSpline"] = outcome(lambda: si.CubicSpline(x5, y5)(1.5 + 1j))
    rec["scalar/CubicSpline"] = outcome(lambda: type(si.CubicSpline(x5, y5)(1.5)).__name__)
    rec["scalar/CubicSpline_ndim"] = outcome(lambda: si.CubicSpline(x5, y5)(1.5).ndim)
    # the special cases
    rec["n2/CubicSpline"] = outcome(lambda: si.CubicSpline([0.0, 2.0], [1.0, 5.0])([0.5, 3.0]))
    rec["n2/CubicSpline_clamped"] = outcome(lambda: si.CubicSpline([0.0, 2.0], [1.0, 5.0], bc_type="clamped")([0.5]))
    rec["n3/CubicSpline"] = outcome(lambda: si.CubicSpline([0.0, 1.0, 3.0], [1.0, 0.0, 4.0]).c)
    rec["n1/CubicSpline"] = outcome(lambda: si.CubicSpline([0.0], [1.0]))
    rec["periodic/tolerance_1e-16"] = outcome(
        lambda: si.CubicSpline([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 0.0, 1.0 + 2e-16], bc_type="periodic").c[0, 0]
    )
    rec["periodic/tolerance_1e-14"] = outcome(
        lambda: si.CubicSpline([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 0.0, 1.0 + 1e-14], bc_type="periodic").c[0, 0]
    )
    rec["periodic/make_interp_spline_1e-6"] = outcome(
        lambda: si.make_interp_spline(
            [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.0, 3.0, 1.0 + 1e-6], bc_type="periodic"
        ).c.tolist()
    )
    rec["periodic/one_end"] = outcome(lambda: si.CubicSpline(x5, y5, bc_type=("periodic", "natural")))
    rec["bc/tuple"] = outcome(lambda: si.CubicSpline(x5, y5, bc_type=((1, 0.5), (2, -1.0)))(2.0))
    rec["bc/mixed"] = outcome(lambda: si.CubicSpline(x5, y5, bc_type=("natural", "not-a-knot"))(2.0))
    rec["k5_natural"] = outcome(
        lambda: si.make_interp_spline(np.arange(8.0), np.arange(8.0) ** 2, k=5, bc_type="natural")
    )
    rec["k0/values"] = outcome(lambda: si.make_interp_spline(x5, y5, k=0)(np.r_[x5, 5.0, -1.0]))
    rec["k0/t"] = outcome(lambda: si.make_interp_spline(x5, y5, k=0).t)
    rec["nu_negative/BSpline"] = outcome(lambda: si.make_interp_spline(x5, y5)(1.0, nu=-1))
    rec["nu_negative/PPoly"] = outcome(lambda: si.CubicSpline(x5, y5)(1.0, nu=-1))
    rec["nu_large/PPoly"] = outcome(lambda: si.CubicSpline(x5, y5)(1.0, nu=5))
    rec["design_matrix"] = outcome(
        lambda: (lambda m: [m.nnz, int(m.count_nonzero()), type(m).__name__])(
            si.BSpline.design_matrix(np.array([0.0, 1.0, 2.5]), np.r_[0, 0, 0, 0, 1, 2, 3, 3, 3, 3.0], 3)
        )
    )
    # roots
    for pid in M.PIECES_IDS:
        P = M.pieces(pid)
        pp = si.PPoly(P.c, P.x)
        rec[pid] = {"digest": P.digest, **outcome(lambda pp=pp: pp.roots())}
        rec[f"roots_inside/{pid}"] = {"digest": P.digest, **outcome(lambda pp=pp: pp.roots(extrapolate=False))}
        rec[f"roots_nodisc/{pid}"] = {"digest": P.digest, **outcome(lambda pp=pp: pp.roots(discontinuity=False))}
    rec["roots/batch"] = outcome(lambda: si.CubicSpline(x5, np.column_stack([y5, -y5])).roots())
    # RGI
    gx, gy = np.arange(6.0), np.arange(6.0)
    q = np.add.outer(gx**2, gy**2)
    rec["rgi/cubic_quadratic"] = outcome(lambda: si.RegularGridInterpolator((gx, gy), q, method="cubic")([[3.0, 4.0]]))
    rec["rgi/nan_xi_bounds"] = outcome(lambda: si.RegularGridInterpolator((gx, gy), q)([[math.nan, 1.0]]))
    rec["rgi/nan_xi_fill"] = outcome(
        lambda: si.RegularGridInterpolator((gx, gy), q, bounds_error=False, fill_value=5.0)(
            [[math.nan, 1.0], [9.0, 1.0]]
        )
    )
    rec["rgi/nearest_tie"] = outcome(lambda: si.RegularGridInterpolator((gx,), gx * 10, method="nearest")([2.5, 3.5]))
    rec["rgi/nu_linear"] = outcome(lambda: si.RegularGridInterpolator((gx, gy), q)([[1.5, 2.5]], nu=(1, 0)))
    rec["rgi/nu_cubic"] = outcome(
        lambda: si.RegularGridInterpolator((gx, gy), q, method="cubic")([[1.5, 2.5]], nu=(1, 0))
    )
    rec["rgi/min_points_cubic"] = outcome(lambda: si.RegularGridInterpolator((gx[:3], gy), q[:3], method="cubic"))
    rec["rgi/one_point_linear"] = outcome(lambda: si.RegularGridInterpolator((gx[:1], gy), q[:1])([[0.0, 2.5]]))
    rec["rgi/complex_fill"] = outcome(lambda: si.RegularGridInterpolator((gx, gy), q, fill_value=1j))
    rec["rgi/tuple_xi"] = outcome(
        lambda: si.RegularGridInterpolator((gx, gy), q)((np.array([1.5, 2.0]), np.array([0.5, 4.0])))
    )
    rec["interpn/splinef2d"] = outcome(lambda: si.interpn((gx, gy), q, [[1.5, 2.5]], method="splinef2d"))
    g = M.rng("rgi-bits")
    xp = np.sort(g.uniform(0, 1, 101))
    fp = g.standard_normal(101)
    x = g.uniform(xp[0], xp[-1], 10000)
    lin = si.RegularGridInterpolator((xp,), fp)(x[:, None])
    slin = si.RegularGridInterpolator((xp,), fp, method="slinear")(x[:, None])
    npi = np.interp(x, xp, fp)
    k1 = si.make_interp_spline(xp, fp, k=1)(x)
    rec["bits/linear_paths"] = {
        "digest": M.digest(xp, fp, x), "points": int(x.size),
        "rgi_linear_eq_np_interp": int(np.sum(lin == npi)), "rgi_slinear_eq_linear": int(np.sum(slin == lin)),
        "k1_eq_np_interp": int(np.sum(k1 == npi)), "k1_eq_rgi_slinear": int(np.sum(k1 == slin)),
    }  # fmt: skip
    c = M.curve("random/50")
    rec["bits/k1_nodes"] = {"digest": c.digest, "nodes": c.n,
                            "not_exact": int(np.sum(si.make_interp_spline(c.x, c.y, k=1)(c.x) != c.y))}  # fmt: skip
    # RBF
    yy = M.scatter("scatter/duplicate2d")
    rec["rbf/duplicate"] = {"digest": yy.digest, **outcome(lambda: si.RBFInterpolator(yy.y, yy.d)(yy.xi[:3]))}
    yu = M.scatter("scatter/uniform2d")
    rec["rbf/negative_smoothing"] = outcome(lambda: si.RBFInterpolator(yu.y, yu.d, smoothing=-1.0)(yu.xi[:2]))
    rec["rbf/low_degree"] = outcome(lambda: si.RBFInterpolator(yu.y, yu.d, degree=0)(yu.xi[:2]))
    rec["rbf/degree_minus1"] = outcome(lambda: si.RBFInterpolator(yu.y, yu.d, degree=-1)(yu.xi[:2]))
    rec["rbf/no_epsilon"] = outcome(lambda: si.RBFInterpolator(yu.y, yu.d, kernel="gaussian"))
    rec["rbf/kernel_case"] = outcome(lambda: si.RBFInterpolator(yu.y, yu.d, kernel="Gaussian", epsilon=1.0)(yu.xi[:1]))
    rec["rbf/neighbors_gt_n"] = outcome(lambda: si.RBFInterpolator(yu.y, yu.d, neighbors=1000)(yu.xi[:2]))
    yc = M.scatter("scatter/collinear2d")
    rec["rbf/collinear"] = {"digest": yc.digest, **outcome(lambda: si.RBFInterpolator(yc.y, yc.d)(yc.xi[:2]))}
    # polynomial and rational
    pc = M.poly("poly/uniform30")
    rec["krogh/high_degree"] = outcome(lambda: si.KroghInterpolator(pc.xi, pc.yi)(pc.x[:3]))
    x40 = np.linspace(-1, 1, 40)
    rec["krogh/high_degree40"] = outcome(lambda: si.KroghInterpolator(x40, 1 / (1 + 25 * x40**2))([0.99]))
    rec["barycentric/rng"] = outcome(lambda: [si.BarycentricInterpolator(pc.xi, pc.yi, rng=s).wi.tolist()
                                              for s in (1, 2)])  # fmt: skip
    rec["barycentric/rng_values_equal"] = outcome(
        lambda: bool(
            np.array_equal(
                si.BarycentricInterpolator(pc.xi, pc.yi, rng=1)(pc.x),
                si.BarycentricInterpolator(pc.xi, pc.yi, rng=2)(pc.x),
            )
        )
    )
    rec["barycentric/node"] = outcome(lambda: si.BarycentricInterpolator(pc.xi, pc.yi)(pc.xi[:3]).tolist())
    rec["floater_hormann/attributes"] = outcome(
        lambda: sorted(a for a in vars(si.FloaterHormannInterpolator(pc.xi, pc.yi)) if not a.startswith("_"))
    )
    for aid in M.AAA_IDS:
        A = M.aaa(aid)
        r = outcome(lambda A=A: (lambda r: {"support_points": plain(r.support_points),
                                            "weights": plain(r.weights), "n_errors": len(r.errors),
                                            "errors_last": float(r.errors[-1])})(si.AAA(A.z, A.f)))  # fmt: skip
        rec[aid] = {"digest": A.digest, **r}
    rec["aaa/max_terms"] = outcome(lambda: si.AAA(M.aaa("aaa/abs").z, M.aaa("aaa/abs").f, max_terms=5).weights.size)
    rec["aaa/clean_up_return"] = outcome(
        lambda: si.AAA(M.aaa("aaa/abs").z, M.aaa("aaa/abs").f, rtol=0, max_terms=40, clean_up=False).clean_up()
    )
    rec["aaa/attributes"] = outcome(
        lambda: sorted(a for a in vars(si.AAA(M.aaa("aaa/tan").z, M.aaa("aaa/tan").f)) if not a.startswith("_"))
    )
    # values on the test data
    for cid in M.real_curve_ids():
        c = M.curve(cid)
        for meth in ("natural", "clamped", "not_a_knot", "periodic", "pchip", "akima", "makima") + tuple(
                b[0] for b in BSPLINE):  # fmt: skip
            if meth.endswith("periodic") and "periodic" not in c.tags:
                continue
            r = outcome(lambda meth=meth, c=c: _scipy_make(si, meth, c)(c.xe))
            rec[f"curve/{cid}/{meth}"] = {"digest": c.digest, **r}
        for name, k, bc in BSPLINE:
            if k >= 2:
                rec[f"knots/{cid}/{name}"] = {"digest": c.digest,
                                              **outcome(lambda k=k, bc=bc, c=c: si.make_interp_spline(
                                                  c.x, c.y, k=k, bc_type=bc).t)}  # fmt: skip
    for gid in M.GRID_IDS:
        G = M.grid(gid)
        for meth in ("linear", "nearest", "slinear", "cubic", "quintic", "pchip"):
            rec[f"{gid}/{meth}"] = {
                "digest": G.digest,
                **outcome(lambda G=G, meth=meth: si.RegularGridInterpolator(G.points, G.values, method=meth)(G.xi)),
            }
        rec[f"{gid}/cubic_legacy"] = {
            "digest": G.digest,
            **outcome(lambda G=G: si.RegularGridInterpolator(G.points, G.values, method="cubic_legacy")(G.xi)),
        }
    for pid in M.POLY_IDS:
        P = M.poly(pid)
        rec[f"{pid}/krogh"] = {"digest": P.digest, **outcome(lambda P=P: si.KroghInterpolator(P.xi, P.yi)(P.x))}
        if P.distinct:
            rec[f"{pid}/barycentric"] = {
                "digest": P.digest,
                **outcome(lambda P=P: si.BarycentricInterpolator(P.xi, P.yi, rng=0)(P.x)),
            }
            rec[f"{pid}/floater_hormann"] = {
                "digest": P.digest,
                **outcome(lambda P=P: si.FloaterHormannInterpolator(P.xi, P.yi)(P.x)),
            }
    for sid in M.SCATTER_IDS:
        sc = M.scatter(sid)
        rec[f"{sid}/thin_plate_spline"] = {
            "digest": sc.digest,
            **outcome(lambda sc=sc: si.RBFInterpolator(sc.y, sc.d)(sc.xi)),
        }
        rec[f"{sid}/nearest"] = {
            "digest": sc.digest,
            **outcome(lambda sc=sc: si.NearestNDInterpolator(sc.y, np.arange(sc.y.shape[0]))(sc.xi)),
        }
    meta = {"tool": "tools/interpolate_probe.py", "scipy": scipy.__version__, "numpy": np.__version__}
    log(write_records("scipy.json", meta, rec))


def r_records(log):
    M = S()
    rec = {}
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        script = ["out <- list()"]
        for k, cid in enumerate(M.real_curve_ids()):
            c = M.curve(cid)
            for name, arr in (("x", c.x), ("y", c.y), ("xe", c.xe)):
                (d / f"{name}{k}.txt").write_text("\n".join(v.hex() for v in arr.tolist()) + "\n")
            script.append(f'x <- as.numeric(readLines("{d / f"x{k}.txt"}"))')
            script.append(f'y <- as.numeric(readLines("{d / f"y{k}.txt"}"))')
            script.append(f'xe <- as.numeric(readLines("{d / f"xe{k}.txt"}"))')
            script.append(f'out[["{cid}|approx2"]] <- approx(x, y, xout = xe, rule = 2)$y')
            script.append(f'out[["{cid}|approx1"]] <- approx(x, y, xout = xe, rule = 1)$y')
            if c.n >= 2:
                script.append(f'out[["{cid}|natural"]] <- splinefun(x, y, method = "natural")(xe)')
                script.append(f'out[["{cid}|fmm"]] <- splinefun(x, y, method = "fmm")(xe)')
            if "periodic" in c.tags:
                script.append(f'out[["{cid}|periodic"]] <- splinefun(x, y, method = "periodic")(xe)')
        script.append(
            'for (k in names(out)) cat(k, "\\t", paste(sprintf("%a", out[[k]]), collapse = " "), "\\n", sep = "")'
        )
        (d / "run.R").write_text("\n".join(script))
        p = subprocess.run(["Rscript", str(d / "run.R")], capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, OMP_NUM_THREADS="1"))  # fmt: skip
        if p.returncode:
            log(f"R failed: {p.stderr[-2000:]}")
            return
        ver = subprocess.run(["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        key, _, vals = line.partition("\t")
        cid, _, meth = key.partition("|")
        values = [float.fromhex(v) if v not in ("NA", "NaN") else math.nan for v in vals.split()]
        rec[f"{meth}/{cid}"] = {"digest": M.curve(cid).digest, "values": values}
    log(write_records("r.json", {"tool": "tools/interpolate_probe.py", "R": ver.stdout.strip()}, rec))


# ---------------------------------------------------------------- timings


def best(fn, reps=3):
    out = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out * 1e3


def timings(log):
    import scipy.interpolate as si

    g = np.random.Generator(np.random.PCG64(20261123))
    log(f"## time ({load('at start')}); ms, best of 3 unless (1) -- one run")

    def row(what, ms, note=""):
        log(f"  {what:60s} {ms:10.3g}  {note}")

    for n in (10**3, 10**6):
        xp = np.arange(n) / (n - 1)
        fp = g.standard_normal(n)
        x = g.random(10**6)
        xs = np.sort(x)
        row(f"numpy interp n = {n:.0e}, 10^6 points", best(lambda: np.interp(x, xp, fp)),
            f"sorted {best(lambda: np.interp(xs, xp, fp)):.3g}")  # fmt: skip
    xp = np.arange(10**3) / 999
    fp = g.standard_normal(10**3)
    k1 = si.make_interp_spline(xp, fp, k=1)
    x5 = g.random(10**5)
    row("scipy k = 1 BSpline n = 1e3, 10^5 random points", best(lambda: k1(x5)))
    n = 10**6
    xb, yb = np.arange(n) / (n - 1), g.standard_normal(n)
    for bc in ("not-a-knot", "natural", "periodic"):
        yv = yb.copy()
        if bc == "periodic":
            yv[-1] = yv[0]
        row(f"scipy CubicSpline build n = 1e6, {bc}", best(lambda yv=yv, bc=bc: si.CubicSpline(xb, yv, bc_type=bc)))
    ym = g.standard_normal((1000, 1000))
    row("scipy CubicSpline build n = 1e3, 1000 columns", best(lambda: si.CubicSpline(xp, ym)))
    cs3 = si.CubicSpline(xp, fp)
    cs6 = si.CubicSpline(xb, yb)
    x6 = g.random(10**6)
    row(
        "scipy CubicSpline eval 10^6 points (n = 1e3 / 1e6)",
        best(lambda: cs3(x6)),
        f"n = 1e6 {best(lambda: cs6(x6)):.3g}",
    )
    row("scipy make_interp_spline k = 3 build n = 1e6", best(lambda: si.make_interp_spline(xb, yb)),
        f"k = 5, n = 1e5 {best(lambda: si.make_interp_spline(xb[:10**5] * 10, yb[:10**5], k=5)):.3g}")  # fmt: skip
    for nn in (10**3, 10**4):
        b = si.make_interp_spline(np.arange(nn) / (nn - 1), g.standard_normal(nn))
        row(f"scipy BSpline k = 3 eval 10^5 random points, n = {nn:.0e} (1)", best(lambda b=b: b(x5), reps=1))
    row("scipy PchipInterpolator build n = 1e6", best(lambda: si.PchipInterpolator(xb, yb)),
        f"eval 10^6 {best(lambda: si.PchipInterpolator(xb, yb)(x6)):.3g} (build included)")  # fmt: skip
    row("scipy Akima1DInterpolator build n = 1e6", best(lambda: si.Akima1DInterpolator(xb, yb)),
        f"makima {best(lambda: si.Akima1DInterpolator(xb, yb, method='makima')):.3g}")  # fmt: skip
    row("scipy derivative + antiderivative n = 1e6", best(lambda: (cs6.derivative(), cs6.antiderivative())),
        f"integrate {best(lambda: cs6.integrate(0, 1)):.3g}")  # fmt: skip
    row("scipy roots n = 1e3", best(lambda: cs3.roots()), f"n = 1e6 (1) {best(lambda: cs6.roots(), reps=1):.0f}")
    xc = np.cos(np.pi * np.arange(101) / 100)
    bary = si.BarycentricInterpolator(xc, np.exp(xc), rng=0)
    xr = g.uniform(-1, 1, 10**6)
    row("scipy BarycentricInterpolator eval 10^6, n = 101", best(lambda: bary(xr)),
        f"build {best(lambda: si.BarycentricInterpolator(xc, np.exp(xc), rng=0)):.3g}")  # fmt: skip
    z = np.linspace(-1, 1, 1000)
    si.AAA(z, np.abs(z))  # the first call pays a one-time cost (about 0.1 s)
    row("scipy AAA M = 1000, tan (1)", best(lambda: si.AAA(z, np.tan(np.pi * z / 2)), reps=1),
        f"|x| (1) {best(lambda: si.AAA(z, np.abs(z)), reps=1):.3g}")  # fmt: skip
    gx = np.arange(1000) / 999
    v2 = g.standard_normal((1000, 1000))
    q2 = g.random((10**6, 2))
    rgi = si.RegularGridInterpolator((gx, gx), v2)
    row("scipy RGI linear 2-D 1000^2, 10^6 points", best(lambda: rgi(q2)),
        f"nearest {best(lambda: rgi(q2, method='nearest')):.3g}")  # fmt: skip
    g3 = np.arange(100) / 99
    rgi3 = si.RegularGridInterpolator((g3, g3, g3), g.standard_normal((100, 100, 100)))
    q3 = g.random((10**6, 3))
    row("scipy RGI linear 3-D 100^3, 10^6 points", best(lambda: rgi3(q3)))
    vc = g.standard_normal((100, 100))
    row("scipy RGI cubic 2-D 100^2 build (1)", best(lambda: si.RegularGridInterpolator((g3, g3), vc, method="cubic"),
                                                    reps=1))  # fmt: skip
    rc = si.RegularGridInterpolator((g3, g3), vc, method="cubic")
    q5 = g.random((10**5, 2))
    row("scipy RGI cubic 2-D 100^2, 10^5 points (1)", best(lambda: rc(q5), reps=1))
    rp = si.RegularGridInterpolator((g3, g3), vc, method="pchip")
    row("scipy RGI pchip 2-D 100^2, 10^4 points (1)", best(lambda: rp(q5[: 10**4]), reps=1))
    for npts in (1000, 4000):
        yr = g.random((npts, 2))
        dr = g.standard_normal(npts)
        row(f"scipy RBF tps build n = {npts} (1)", best(lambda yr=yr, dr=dr: si.RBFInterpolator(yr, dr), reps=1))
    r1 = si.RBFInterpolator(g.random((1000, 2)), g.standard_normal(1000))
    row("scipy RBF tps eval 10^4 points, n = 1000", best(lambda: r1(q5[: 10**4])))
    rn = si.RBFInterpolator(g.random((10**5, 2)), g.standard_normal(10**5), neighbors=50)
    row("scipy RBF neighbors = 50, n = 1e5, 10^4 points (1)", best(lambda: rn(q5[: 10**4]), reps=1))
    nd = si.NearestNDInterpolator(g.random((10**5, 2)), g.standard_normal(10**5))
    row("scipy NearestNDInterpolator n = 1e5, 10^6 points (1)", best(lambda: nd(q2), reps=1))
    script = (
        "set.seed(1); x <- (0:999)/999; y <- rnorm(1000); xo <- runif(1e6);"
        "t <- c(); for (i in 1:3) t <- c(t, system.time(approx(x, y, xout = xo, rule = 2))[['elapsed']]);"
        "cat('approx', min(t) * 1000, '\\n');"
        "x6 <- (0:999999)/999999; y6 <- rnorm(1e6);"
        "t <- system.time(f <- splinefun(x6, y6, method = 'natural'))[['elapsed']];"
        "cat('splinefun_build', t * 1000, '\\n');"
        "f3 <- splinefun(x, y, method = 'natural'); t <- c();"
        "for (i in 1:3) t <- c(t, system.time(f3(xo))[['elapsed']]); cat('splinefun_eval', min(t) * 1000, '\\n')"
    )
    p = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=600)
    for line in p.stdout.splitlines():
        log(f"  R {line}")
    log(f"({load('at end')})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools/interpolate_probe.py", description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default all of {', '.join(SECTIONS)})")
    args = p.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)})")
    log = Log()
    table = {"kernels": kernels, "banded": banded, "numpy": numpy_records, "scipy": scipy_records, "r": r_records,
             "time": timings}  # fmt: skip
    for s in SECTIONS:
        if not args.sections or s in args.sections:
            table[s](log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
