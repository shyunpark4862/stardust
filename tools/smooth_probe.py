"""Probes of sd.smooth's foundations and of the libraries it is compared with (docs/design/smooth.md §2.9,
§3, §4, §5, §7.4). The records the tests read go to tests/smooth/data/; the timings go to the terminal and to
build/logs/smooth_probe/<stamp>.log, and docs/design/smooth.md quotes them.

    .venv/bin/python tools/smooth_probe.py                    # every section
    .venv/bin/python tools/smooth_probe.py proto scipy r      # only these sections

Sections (positional; default all):
  proto        float64 prototypes of the planned algorithms against the references of tests/smooth/ref
               (tools/smooth_ref.py must have run) -> prototype.json:
                 lowess      the one-pass closed form of the local line, its recentred form and windowed
                             power moments from prefix sums (the O(1)-per-point path §3.1 rejects), in
                             units of kappa u scale
                 loess       Householder QR of the weighted design against the moment matrix's Cholesky,
                             plain and with one correction from the residual (degree 2; one to three
                             variables), in the same units
                 spline      Givens QR of the stacked matrix, the normal equations' Cholesky and Reinsch's
                             form: fitted values and coefficients of the smoothing spline, n = 30 to 300
                 trace       the leverages from the band of (R'R)^-1 by the backward recurrence
                 kde         the cut of the direct sum (terms below u / n of the largest dropped) and the
                             linear-binning FFT path against its error bound
                 savgol      the Gram-polynomial recurrence in float64 and at 106 bits, and scipy's
                             Vandermonde least squares, against the exact coefficients; the orders up to
                             which the recurrence holds, per window
  kernels      the timing of the closed form and of QR against moments (tools/smooth_probe_kernels.cpp)
  scipy        scipy 1.18.1's smoothing splines, FITPACK replacements, gaussian_kde and Savitzky-Golay at
               their edges and on the test data -> scipy.json
  statsmodels  statsmodels 0.15.0's lowess, KDEUnivariate, KDEMultivariate, KernelReg and bandwidths
               -> statsmodels.json
  sklearn      scikit-learn 1.9.1's KernelDensity -> sklearn.json
  r            R 4.6.2's lowess, loess, smooth.spline, density, bw.*, ksmooth, KernSmooth, mgcv and MASS on
               the test data, values as hex floats -> r.json
  time         the comparators' timings of §5 at sizes up to 10^5 (best of three unless marked)

Records named rule/... are the behaviours the design departs from or relies on (§2.9). The C++ program is
compiled into build/smooth_probe/ with sd's flags (-O3 -mcpu=native -ffp-contract=off) under the shared build
lock (build/.agent-build-lock). Every child process runs under a time limit.
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

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench import cli  # noqa: E402

BUILD = ROOT / "build" / "smooth_probe"
DATA = ROOT / "tests" / "smooth" / "data"
LOCK = ROOT / "build" / ".agent-build-lock"
FLAGS = ["-std=c++20", "-O3", "-mcpu=native", "-ffp-contract=off"]
SRC = ROOT / "tools" / "smooth_probe_kernels.cpp"
SECTIONS = ("proto", "kernels", "scipy", "statsmodels", "sklearn", "r", "time")
CHILD_SECONDS = 1200
TOOL = "tools/smooth_probe.py"
U = 2.0**-53
TAU = 8.3  # the kernel's cut of the binned path, in bandwidths (§3.5)
# the half-width of a kernel of standard deviation 1 (R's scale): sklearn's and statsmodels' bandwidth of these
# kernels is the half-width of the support
HALF_WIDTH = {"epanechnikov": 5**0.5, "tophat": 3**0.5, "linear": 6**0.5}
SM_HALF_WIDTH = {"epa": 5**0.5, "uni": 3**0.5, "tri": 6**0.5, "biw": 7**0.5}


def S():
    from tests.smooth import samples

    return samples


def RD():
    from tests.smooth import refdata

    return refdata


class Log:
    def __init__(self):
        self.path = cli.log_file("smooth_probe")
        print(f"log: {self.path.relative_to(ROOT)}", flush=True)

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as f:
            f.write(text + "\n")


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
        out["warnings"] = sorted({f"{x.category.__name__}: {str(x.message)[:200]}" for x in w})
    return out


def plain(v):
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            return {"real": v.real.tolist(), "imag": v.imag.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
        return {"array": v.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, bool, int, str)) or v is None:
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


def best(fn, reps=3):
    out = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out * 1e3


# ---------------------------------------------------------------- prototypes (float64 against the references)


def tricube(d):
    a = np.abs(d)
    return np.where(a < 1.0, (1.0 - a**3) ** 3, 0.0)


def _window(xs, x0, k):
    return float(np.sort(np.abs(xs - x0))[k - 1])


def line_closed(xs, ys, ws, x0, h):
    d = (xs - x0) / h
    v = tricube(d) * ws
    keep = v > 0
    d, v, y = d[keep], v[keep], ys[keep]
    a, b, c, s0, s1 = v.sum(), (v * d).sum(), (v * d * d).sum(), (v * y).sum(), (v * d * y).sum()
    return (c * s0 - b * s1) / (a * c - b * b)


def line_recentred(xs, ys, ws, x0, h):
    d = (xs - x0) / h
    v = tricube(d) * ws
    keep = v > 0
    d, v, y = d[keep], v[keep], ys[keep]
    a = v.sum()
    e = d - (v * d).sum() / a
    return (v * y).sum() / a - ((v * d).sum() / a) * (v * e * y).sum() / (v * e * e).sum()


def line_prefix(ts, ys, ws, t0, h, prefix, prefix_y):
    """The same five moments from prefix sums of w t^p and w t^p y (t the data scaled to [0, 1]): the tricube
    weight is the polynomial 1 - 3|d|^3 + 3|d|^6 - |d|^9, so a window's moments are differences of prefix sums
    re-expanded about t0."""
    d = np.abs(ts - t0)
    inside = np.flatnonzero(d < h)
    lo, hi, mid = int(inside[0]), int(inside[-1]) + 1, int(np.searchsorted(ts, t0))
    mom = np.zeros(3)
    momy = np.zeros(2)
    for a, b, sgn in ((lo, max(lo, min(mid, hi)), -1.0), (max(lo, min(mid, hi)), hi, 1.0)):
        if b <= a:
            continue

        def central(pre, p, a=a, b=b):  # sum (t - t0)^p g over [a, b)
            return sum(math.comb(p, q) * (-t0) ** (p - q) * (pre[q][b] - pre[q][a]) for q in range(p + 1))

        for coef, q in ((1.0, 0), (-3.0, 3), (3.0, 6), (-1.0, 9)):
            for e in range(3):
                mom[e] += coef * sgn**q * central(prefix, q + e) / h ** (q + e)
            for e in range(2):
                momy[e] += coef * sgn**q * central(prefix_y, q + e) / h ** (q + e)
    a, b, c = mom
    return (c * momy[0] - b * momy[1]) / (a * c - b * b)


def proto_lowess(log, rec):
    import smooth_ref as R

    M = S()
    log("lowess: the local line at every data and evaluation point (frac 2/3), max error in units of kappa u scale")
    log(f"  {'set':14s} {'max kappa':>10s} {'closed form':>12s} {'recentred':>10s} {'prefix sums':>12s}")
    worst = {"closed": 0.0, "recentred": 0.0, "prefix": 0.0}
    for sid in (
        "uniform/40",
        "random/50",
        "random/200",
        "clustered/40",
        "narrow/30",
        "wide/30",
        "shifted/30",
        "outliers/40",
        "big/20",
        "weights/30",
    ):
        s = M.scatter(sid)
        order = np.argsort(s.x, kind="stable")
        xs, ys = s.x[order], s.y[order]
        ws = np.ones(s.n) if s.w is None else s.w[order]
        k = R.lowess_k(s.n, 2 / 3)
        Xq, Yq, Wq = [R.Q(v) for v in xs], [R.Q(v) for v in ys], [R.Q(v) for v in ws]
        span = xs[-1] - xs[0]
        ts = (xs - xs[0]) / span
        pre = [np.concatenate([[0.0], np.cumsum(ws * ts**p)]) for p in range(12)]
        prey = [np.concatenate([[0.0], np.cumsum(ws * ts**p * ys)]) for p in range(11)]
        units = {"closed": 0.0, "recentred": 0.0, "prefix": 0.0}
        kmax = 0.0
        for x0 in np.concatenate([xs, s.xe]):
            ex = R.local_1d(Xq, Yq, Wq, R.Q(x0), k, 1)
            if ex is None or ex[4] < 2:
                continue
            value, scale, _, kappa, _ = ex
            hi, lo = R.dd_of(value)
            bound = kappa * U * R.fl(scale)
            h = _window(xs, x0, k)
            got = {
                "closed": line_closed(xs, ys, ws, x0, h),
                "recentred": line_recentred(xs, ys, ws, x0, h),
                "prefix": line_prefix(ts, ys, ws, (x0 - xs[0]) / span, h / span, pre, prey),
            }
            for name, g in got.items():
                units[name] = max(units[name], abs((g - hi) - lo) / bound if bound > 0 else 0.0)
            kmax = max(kmax, kappa)
        log(f"  {sid:14s} {kmax:10.3g} {units['closed']:12.3g} {units['recentred']:10.3g} {units['prefix']:12.3g}")
        rec[f"lowess/{sid}"] = {"digest": s.digest, "max_kappa": kmax, "units": units}
        for name in worst:
            worst[name] = max(worst[name], units[name])
    rec["lowess/worst"] = worst
    log(f"  worst: {worst}")


def _design(U_, degree):
    p = U_.shape[1]
    cols = [np.ones(U_.shape[0])]
    if degree >= 1:
        cols += [U_[:, a] for a in range(p)]
    if degree >= 2:
        cols += [U_[:, a] * U_[:, b] for a in range(p) for b in range(a, p)]
    return np.stack(cols, axis=1)


def _fit_qr(Phi, v, y):
    s = np.sqrt(v)
    q, r = np.linalg.qr(Phi * s[:, None])
    return float(np.linalg.solve(r, q.T @ (s * y))[0])


def _fit_cholesky(Phi, v, y):
    Mm = Phi.T @ (Phi * v[:, None])
    L = np.linalg.cholesky(Mm)
    z = np.linalg.solve(L, Phi.T @ (v * y))
    return float(np.linalg.solve(L.T, z)[0])


def _fit_refined(Phi, v, y):
    """The moment matrix's Cholesky with one correction from the residual (corrected semi-normal equations)."""
    Mm = Phi.T @ (Phi * v[:, None])
    L = np.linalg.cholesky(Mm)

    def solve(rhs):
        return np.linalg.solve(L.T, np.linalg.solve(L, rhs))

    a = solve(Phi.T @ (v * y))
    return float((a + solve(Phi.T @ (v * (y - Phi @ a))))[0])


def proto_loess(log, rec):
    import smooth_ref as R

    M, D = S(), RD()
    log("loess: degree 2, span 0.75; max error in units of kappa u scale (kappa: the weighted design's condition)")
    log(f"  {'set':14s} {'max kappa':>10s} {'QR':>8s} {'Cholesky':>9s} {'refined':>8s}")
    ref1, refn = D.ref("local"), D.ref("local_nd")
    worst = {"qr": 0.0, "cholesky": 0.0, "refined": 0.0}
    for sid in (
        "uniform/40",
        "random/50",
        "clustered/40",
        "shifted/30",
        "narrow/30",
        "uniform2/60",
        "uniform3/60",
        "scales2/50",
        "lattice2/49",
    ):
        one = sid in M.scatter_ids()
        s = M.scatter(sid) if one else M.multi(sid)
        X = s.x[:, None] if one else s.X
        Xe = s.xe[:, None] if one else s.Xe
        r = ref1 if one else refn
        m = r.methods["loess_d2_s0.75"]
        q = R.loess_q(s.n, 0.75)
        units = {"qr": 0.0, "cholesky": 0.0, "refined": 0.0}
        kmax = 0.0
        for kind, pts in ((0, X), (1, Xe)):
            pos, idx = D.at(r, "loess_d2_s0.75", sid, kind)
            for at, i in zip(pos, idx):
                x0 = pts[i]
                d = np.sqrt(((X - x0) ** 2).sum(axis=1))
                h = np.sort(d)[q - 1]
                v = tricube(d / h)
                keep = v > 0
                Phi = _design((X[keep] - x0) / h, 2)
                bound = m.fields["kappa"][at] * U * m.fields["scale"][at]
                kmax = max(kmax, float(m.fields["kappa"][at]))
                for name, fit in (("qr", _fit_qr), ("cholesky", _fit_cholesky), ("refined", _fit_refined)):
                    try:
                        g = fit(Phi, v[keep], s.y[keep])
                    except np.linalg.LinAlgError:
                        g = math.inf
                    units[name] = max(units[name], abs((g - m.hi[at]) - m.lo[at]) / bound)
        log(f"  {sid:14s} {kmax:10.3g} {units['qr']:8.3g} {units['cholesky']:9.3g} {units['refined']:8.3g}")
        rec[f"loess/{sid}"] = {"digest": s.digest, "max_kappa": kmax, "units": units}
        for name in worst:
            worst[name] = max(worst[name], units[name])
    rec["loess/worst"] = worst
    log(f"  worst: {worst}")


def _cholesky_ratio(Phi, v):
    """(max L_ii / min L_ii)^2 of the moment matrix's Cholesky factor (inf when the factorization fails): the cheap
    condition estimate that picks the solver."""
    try:
        L = np.linalg.cholesky(Phi.T @ (Phi * v[:, None]))
    except np.linalg.LinAlgError:
        return math.inf
    d = np.abs(np.diag(L))
    return float((d.max() / d.min()) ** 2)


def proto_loess_conditioning(log, rec):
    """The three local solvers on windows of growing condition: 29 points in [0.5, 0.5 + eps] (a square of that side
    in two variables) and one at 1, evaluated at 0 with the whole window, degree 2; three draws per eps."""
    import smooth_ref as R

    log("loess solvers against the window's condition (degree 2, x0 outside a cluster of width eps); units of kappa u")
    log("scale, the worst of three draws; ratio = (max L_ii / min L_ii)^2 of the moment matrix's Cholesky factor")
    log(f"  {'p':>2s} {'eps':>7s} {'kappa':>9s} {'ratio':>9s} {'QR':>8s} {'Cholesky':>9s} {'refined':>8s}")
    g = S().rng("proto", "loess_conditioning")
    n = 30
    draws = []  # (ratio, the refined solve's error) of every draw
    for p in (1, 2):
        for e in range(1, 9):
            eps = 10.0**-e
            units = {"qr": 0.0, "cholesky": 0.0, "refined": 0.0}
            kmax = rmax = 0.0
            for _ in range(3):
                X = np.concatenate([0.5 + eps * g.random((n - 1, p)), np.ones((1, p))])
                y = np.sin(3.0 * X.sum(axis=1)) + 0.1 * g.standard_normal(n)
                Xq, Yq = [[R.Q(v) for v in row] for row in X], [R.Q(v) for v in y]
                if p == 1:
                    value, scale, _, kappa, _ = R.local_1d([r[0] for r in Xq], Yq, [R.Q(1.0)] * n, R.Q(0.0), n, 2)
                    hi, lo, scale = *R.dd_of(value), R.fl(scale)
                else:
                    hi, lo, scale, _, kappa, _ = R.local_nd(Xq, Yq, [R.Q(0.0)] * p, n, 2)
                d = np.sqrt((X**2).sum(axis=1))
                v = tricube(d / np.sort(d)[n - 1])
                keep = v > 0
                Phi = _design(X[keep] / np.sort(d)[n - 1], 2)
                ratio = _cholesky_ratio(Phi, v[keep])
                kmax, rmax = max(kmax, kappa), max(rmax, ratio)
                for name, fit in (("qr", _fit_qr), ("cholesky", _fit_cholesky), ("refined", _fit_refined)):
                    try:
                        got = fit(Phi, v[keep], y[keep])
                    except np.linalg.LinAlgError:
                        got = math.inf
                    err = abs((got - hi) - lo) / (kappa * U * scale)
                    units[name] = max(units[name], err)
                    if name == "refined":
                        draws.append((ratio, err))
            log(
                f"  {p:2d} {eps:7.0e} {kmax:9.3g} {rmax:9.3g} {units['qr']:8.3g} {units['cholesky']:9.3g} "
                f"{units['refined']:8.3g}"
            )
            rec[f"loess_conditioning/{p}/{eps:.0e}"] = {"max_kappa": kmax, "max_ratio": rmax, "units": units}
    first_bad = min((r for r, e in draws if e > 1.0), default=math.inf)
    good = max((r for r, e in draws if r < first_bad), default=0.0)
    rec["loess_conditioning/largest_good_ratio"] = good
    rec["loess_conditioning/smallest_bad_ratio"] = first_bad
    log(
        f"  every draw with ratio <= {good:.3g} has the refined solve within 1; the smallest ratio beyond 1: "
        f"{first_bad:.3g}"
    )


def givens_qr(A, b):
    """R (upper triangular) and Q'b of the least-squares problem by Givens rotations, row by row."""
    A, b = A.copy(), b.copy()
    rows, n = A.shape
    for j in range(n):
        for i in range(j + 1, rows):
            if A[i, j] == 0.0:
                continue
            r = math.hypot(A[j, j], A[i, j])
            c, s = A[j, j] / r, A[i, j] / r
            rj, ri = A[j, j:].copy(), A[i, j:].copy()
            A[j, j:], A[i, j:] = c * rj + s * ri, c * ri - s * rj
            b[j], b[i] = c * b[j] + s * b[i], c * b[i] - s * b[j]
    return np.triu(A[:n]), b[:n]


def band_inverse(Rm, bw):
    """The band (|i - j| <= bw) of (R'R)^-1 by the backward recurrence of §3.3."""
    n = Rm.shape[0]
    Z = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        top = min(i + bw, n - 1)
        for j in range(top, i - 1, -1):
            s = sum(Rm[i, l] * (Z[l, j] if l <= j else Z[j, l]) for l in range(i + 1, top + 1))
            Z[i, j] = (1.0 / Rm[i, i] ** 2 if i == j else 0.0) - s / Rm[i, i]
    return Z


def _spline_matrices(x, t):
    """B (n, nc), and G with G'G = the roughness matrix: G = chol(R)' T (T: the second derivatives at the distinct
    x, R: the Gram matrix of the piecewise-linear f'')."""
    from scipy.interpolate import BSpline

    nc = t.size - 4
    B = BSpline.design_matrix(x, t, 3).toarray()
    xd = np.unique(x)
    T = BSpline(t, np.eye(nc), 3)(xd, nu=2)
    h = np.diff(xd)
    Rm = np.zeros((xd.size, xd.size))
    for i, hi in enumerate(h):
        Rm[i, i] += hi / 3
        Rm[i + 1, i + 1] += hi / 3
        Rm[i, i + 1] = Rm[i + 1, i] = hi / 6
    return B, np.linalg.cholesky(Rm).T @ T


def reinsch(x, y, w, lam):
    n = x.size
    h = np.diff(x)
    Qm = np.zeros((n, n - 2))
    Rm = np.zeros((n - 2, n - 2))
    for j in range(n - 2):
        Qm[j, j], Qm[j + 1, j], Qm[j + 2, j] = 1 / h[j], -1 / h[j] - 1 / h[j + 1], 1 / h[j + 1]
        Rm[j, j] = (h[j] + h[j + 1]) / 3
        if j + 1 < n - 2:
            Rm[j, j + 1] = Rm[j + 1, j] = h[j + 1] / 6
    g = np.linalg.solve(Rm + lam * Qm.T @ (Qm / w[:, None]), Qm.T @ y)
    return y - lam * (Qm @ g) / w


def _float_fits(x, y, w, t, lam):
    B, G = _spline_matrices(x, t)
    sw = np.sqrt(w)
    Rm, qtb = givens_qr(
        np.vstack([B * sw[:, None], math.sqrt(lam) * G]), np.concatenate([sw * y, np.zeros(G.shape[0])])
    )
    c_qr = np.linalg.solve(Rm, qtb)
    A = B.T @ (B * w[:, None]) + lam * G.T @ G
    try:
        L = np.linalg.cholesky(A)
        c_ne = np.linalg.solve(L.T, np.linalg.solve(L, B.T @ (w * y)))
    except np.linalg.LinAlgError:
        c_ne = np.full(B.shape[1], np.inf)
    return B, Rm, c_qr, c_ne, reinsch(x, y, w, lam)


def _arb_reference(x, y, t, lam, prec=768):
    """The fitted values of the smoothing spline by an Arb solve of the exact normal equations (for n too large
    for the rational inverse)."""
    import smooth_ref as R
    from flint import arb, arb_mat, ctx, fmpq

    tq = [R.Q(v) for v in t]
    nc = len(tq) - 4
    om = R.roughness(tq)
    A = [[R.Q(lam) * om[i][j] for j in range(nc)] for i in range(nc)]
    rhs = [fmpq(0)] * nc
    rows = []
    for xv, yv in zip(x, y):
        l = R.span_of(tq, 3, R.Q(xv))
        b = R.basis(tq, 3, l, R.Q(xv))
        rows.append((l - 3, b))
        for a in range(4):
            rhs[l - 3 + a] += R.Q(yv) * b[a]
            for c in range(4):
                A[l - 3 + a][l - 3 + c] += b[a] * b[c]
    old = ctx.prec
    ctx.prec = prec
    try:
        sol = arb_mat(nc, nc, [R.arb_q(c) for row in A for c in row]).solve(arb_mat(nc, 1, [R.arb_q(c) for c in rhs]))
        fit = [sum((R.arb_q(b[a]) * sol[o + a, 0] for a in range(4)), arb(0)) for o, b in rows]
        assert all(f.rad() < 2.0**-200 * (abs(f.mid()) + 1) for f in fit)
        return np.array([float(f.mid()) for f in fit]), np.array([float(sol[i, 0].mid()) for i in range(nc)])
    finally:
        ctx.prec = old


def proto_spline(log, rec):
    M, D = S(), RD()
    rc, rf = D.ref("spline"), D.ref("spline_fit")
    log("smoothing spline: errors of float64 solvers; fitted values in units of u max|f|, coefficients of the QR in")
    log(
        "units of u kappa max|c| (kappa: the stacked matrix's condition), leverages (band of (R'R)^-1) in units of u "
        "kappa"
    )
    log(
        f"  {'set':13s} {'lam/range^3':>11s} {'kappa':>9s} {'QR fit':>8s} {'normal':>9s} {'Reinsch':>9s} "
        f"{'QR coef':>8s} "
        f"{'leverage':>9s} {'tr L':>8s}"
    )
    worst = {
        "qr_fit": 0.0,
        "qr_coef": 0.0,
        "leverage": 0.0,
        "leverage_kappa2": 0.0,
        "trace": 0.0,
        "normal_fit": 0.0,
        "reinsch_fit": 0.0,
    }
    for sid in (
        "uniform/40",
        "random/30",
        "random/50",
        "clustered/40",
        "shifted/30",
        "narrow/30",
        "wide/30",
        "outliers/40",
    ):
        s = M.scatter(sid)
        g = D.group_of(rc, sid)
        t = np.asarray(rc.group_inputs["t_ss"][g])
        w = np.ones(s.n)
        for c in ("1e-8", "1e-4", "1e-1", "1e2"):
            name = f"ss_{c}"
            lam = float(rc.group_inputs[f"lam_ss_{c}"][g])
            B, Rm, c_qr, c_ne, f_re = _float_fits(s.x, s.y, w, t, lam)
            pos, idx = D.at(rf, name, sid)
            fit = np.empty(s.n)
            fit[idx] = rf.methods[name].hi[pos]
            lev = np.empty(s.n)
            lev[idx] = rf.methods[name].fields["leverage"][pos]
            pos, idx = D.at(rc, name, sid)
            coef = np.empty(idx.size)
            coef[idx] = rc.methods[name].hi[pos]
            tm = rf.methods[f"trl_{name}"]
            gi = int(np.flatnonzero(tm.index == D.group_of(rf, sid))[0])
            kappa, trl = float(tm.fields["kappa"][gi]), float(tm.hi[gi])
            fs = U * np.abs(fit).max()
            Z = band_inverse(Rm, 3)
            lev_hat = np.array([B[i] @ (Z + np.triu(Z, 1).T) @ B[i] for i in range(s.n)])
            row = {
                "lam": lam,
                "kappa": kappa,
                "qr_fit": float(np.abs(B @ c_qr - fit).max() / fs),
                "normal_fit": float(np.abs(B @ c_ne - fit).max() / fs),
                "reinsch_fit": float(np.abs(f_re - fit).max() / fs),
                "qr_coef": float(np.abs(c_qr - coef).max() / (U * kappa * np.abs(coef).max())),
                "leverage": float(np.abs(lev_hat - lev).max() / (U * kappa)),
                "leverage_kappa2": float(np.abs(lev_hat - lev).max() / (U * kappa * kappa)),
                "trace": float(abs(lev_hat.sum() - trl) / (U * kappa)),
            }
            log(
                f"  {sid:13s} {c:>11s} {kappa:9.3g} {row['qr_fit']:8.3g} {row['normal_fit']:9.3g} "
                f"{row['reinsch_fit']:9.3g} "
                f"{row['qr_coef']:8.3g} {row['leverage']:9.3g} {row['trace']:8.3g}"
            )
            rec[f"spline/{sid}/{c}"] = {"digest": s.digest, **row}
            for k in worst:
                worst[k] = max(worst[k], row[k]) if math.isfinite(row[k]) else math.inf
    rec["spline/worst"] = worst
    log(f"  worst: {worst}")
    good = [
        r
        for k, r in rec.items()
        if k.startswith("spline/") and isinstance(r, dict) and "kappa" in r and r["kappa"] * U <= 2.0**-20
    ]
    rec["spline/worst_well_conditioned"] = {
        k: max(r[k] for r in good) for k in ("qr_fit", "qr_coef", "leverage", "trace")
    }
    log(f"  worst where u kappa <= 2^-20: {rec['spline/worst_well_conditioned']}")
    log("  growth with n (uniform grid on [0, 10] and uniform random x; the reference an Arb solve at 768 bits)")
    for n in (100, 300):
        for kind in ("grid", "random"):
            gen = M.rng("proto", "spline", n, kind)
            x = np.linspace(0.0, 10.0, n) if kind == "grid" else np.sort(10.0 * gen.random(n))
            y = np.sin(x) + 0.3 * gen.standard_normal(n)
            t = np.concatenate([[x[0]] * 3, x, [x[-1]] * 3])
            for c in (1e-8, 1e-2, 1e4):
                lam = c * float(x[-1] - x[0]) ** 3
                fit, coef = _arb_reference(x, y, t, lam)
                B, Rm, c_qr, c_ne, f_re = _float_fits(x, y, np.ones(n), t, lam)
                fs = U * np.abs(fit).max()
                row = {
                    "qr_fit": float(np.abs(B @ c_qr - fit).max() / fs),
                    "normal_fit": float(np.abs(B @ c_ne - fit).max() / fs),
                    "reinsch_fit": float(np.abs(f_re - fit).max() / fs),
                }
                log(
                    f"  n = {n} {kind:6s} lam/range^3 = {c:g}: QR {row['qr_fit']:.3g}, normal equations "
                    f"{row['normal_fit']:.3g}, Reinsch {row['reinsch_fit']:.3g} (units of u max|f|)"
                )
                rec[f"spline_growth/{kind}/{n}/{c:g}"] = row


def kde_cut(x, w, h, x0):
    """(pdf, logpdf) by the shifted and cut direct sum of §3.5."""
    D = ((x0 - x) / h) ** 2
    D0 = D.min()
    keep = D - D0 <= 2.0 * math.log(x.size / U)
    tot = float((w[keep] * np.exp(-0.5 * (D[keep] - D0))).sum())
    norm = h * math.sqrt(2 * math.pi)
    return math.exp(-0.5 * D0) * tot / norm, -0.5 * D0 + math.log(tot / norm), int(keep.sum())


def kde_binned(x, w, h, n=512, cut=3.0):
    """(grid, density, the grid step) by linear binning and an FFT convolution with the kernel cut at TAU h."""
    lo, hi = x.min() - cut * h, x.max() + cut * h
    delta = (hi - lo) / (n - 1)
    pos = (x - lo) / delta
    i = np.minimum(np.floor(pos).astype(int), n - 2)
    f = pos - i
    c = np.zeros(n)
    np.add.at(c, i, w * (1 - f))
    np.add.at(c, i + 1, w * f)
    L = int(math.floor(TAU * h / delta))
    ker = np.exp(-0.5 * (np.arange(-L, L + 1) * delta / h) ** 2) / (h * math.sqrt(2 * math.pi))
    size = n + 2 * L
    out = np.fft.irfft(np.fft.rfft(c, size) * np.fft.rfft(ker, size), size)[L : L + n]
    return lo + delta * np.arange(n), np.maximum(out, 0.0), delta


def proto_kde(log, rec):
    M, D = S(), RD()
    r = D.ref("kde")
    log("kde: the cut direct sum against the Arb sum of every point (pdf: relative error in u; logpdf: absolute error")
    log(
        "in u max(1, |logpdf|)); the draft constant is 8 + 2p + log2 n; D0 is the smallest squared distance in "
        "bandwidths"
    )
    worst = 0.0
    for sid in M.density_ids(tag="1d", exclude=("degenerate",)):
        s = M.density(sid)
        g = D.group_of(r, sid)
        h = float(r.group_inputs["bw"][g])
        w = np.full(s.n, 1.0 / s.n) if s.w is None else s.w / s.w.sum()
        pos, idx = D.at(r, "pdf", sid)
        rel = lg = tail = 0.0
        kept = s.n
        for at, i in zip(pos, idx):
            p, lp, nk = kde_cut(s.x, w, h, float(s.xe[i]))
            kept = min(kept, nk)
            hi = r.methods["pdf"].hi[at]
            if hi > 2.0**-1000:
                e = abs((p - hi) - r.methods["pdf"].lo[at]) / hi / U
                rel = max(rel, e)
                tail = max(tail, e / (8 + 2 + math.log2(s.n) + float((((s.xe[i] - s.x) / h) ** 2).min())))
            lhi = r.methods["logpdf"].hi[at]
            lg = max(lg, abs((lp - lhi) - r.methods["logpdf"].lo[at]) / (U * max(1.0, abs(lhi))))
        draft = 8 + 2 + math.log2(s.n)
        log(
            f"  {sid:15s} h = {h:.4g}: pdf {rel:.3g} u ({tail:.3g} of (draft + D0) u), logpdf {lg:.3g} u (draft "
            f"{draft:.1f}); "
            f"fewest points kept {kept} of {s.n}"
        )
        rec[f"kde_cut/{sid}"] = {
            "digest": s.digest,
            "bw": h,
            "pdf_units_of_u": rel,
            "pdf_units_of_draft_plus_D0": tail,
            "logpdf_units_of_u": lg,
            "draft_constant": draft,
            "fewest_kept": kept,
        }
        worst = max(worst, tail, lg / draft)
    rec["kde_cut/worst_over_contract"] = worst
    log(
        "  worst error over the contract (pdf: (8 + 2p + log2 n + D0) u; logpdf: (8 + 2p + log2 n) u max(1, "
        "|logpdf|)): "
        f"{worst:.3g}"
    )
    log(
        "kde: the binned FFT path against the direct sum at its 512 grid points; bound = delta^2 / (8 sqrt(2 pi) h^3) +"
    )
    log(f"phi({TAU}) / h (the FFT's own rounding is far below both)")
    worst = 0.0
    for sid in M.density_ids(tag="1d", exclude=("degenerate",)):
        s = M.density(sid)
        g = D.group_of(r, sid)
        w = np.full(s.n, 1.0 / s.n) if s.w is None else s.w / s.w.sum()
        for scale in (1.0, 0.25):
            h = scale * float(r.group_inputs["bw"][g])
            grid, dens, delta = kde_binned(s.x, w, h)
            direct = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - s.x[None, :]) / h) ** 2)).sum(axis=1) / (
                h * math.sqrt(2 * math.pi)
            )
            err = float(np.abs(dens - direct).max())
            bound = delta**2 / (8 * math.sqrt(2 * math.pi) * h**3) + math.exp(-0.5 * TAU**2) / (
                math.sqrt(2 * math.pi) * h
            )
            relerr = float((np.abs(dens - direct)[direct > 0] / direct[direct > 0]).max())
            log(
                f"  {sid:15s} h = {h:.4g}, delta = {delta:.4g}: error {err:.3g}, bound {bound:.3g} (ratio "
                f"{err / bound:.3g}); "
                f"largest relative error {relerr:.3g}"
            )
            rec[f"kde_binned/{sid}/{scale:g}"] = {
                "digest": s.digest,
                "bw": h,
                "delta": delta,
                "error": err,
                "bound": bound,
                "largest_relative_error": relerr,
            }
            worst = max(worst, err / bound)
    rec["kde_binned/worst_ratio"] = worst
    log(f"  worst error over the bound: {worst:.3g}")


class F106:
    """A float of 106 significant bits (round to nearest even) on exact rationals: the stand-in for double-double
    arithmetic (Arb's balls are not: a wide radius truncates the midpoint)."""

    __slots__ = ("q",)

    def __init__(self, v):
        from fractions import Fraction

        self.q = self._round(Fraction(v))

    @staticmethod
    def _round(q):
        from fractions import Fraction

        if q == 0:
            return q
        shift = 106 - (q.numerator.bit_length() - q.denominator.bit_length())
        scaled = q * (Fraction(2) ** shift)
        return Fraction(round(scaled)) / (Fraction(2) ** shift)

    def _new(self, q):
        out = object.__new__(F106)
        out.q = self._round(q)
        return out

    def __add__(self, o):
        return self._new(self.q + o.q)

    def __sub__(self, o):
        return self._new(self.q - o.q)

    def __mul__(self, o):
        return self._new(self.q * o.q)

    def __truediv__(self, o):
        return self._new(self.q / o.q)

    def __neg__(self):
        return self._new(-self.q)

    def __float__(self):
        return float(self.q)


def gram_coeffs(n, d, deriv, pos, number, every=False):
    """savgol coefficients in window order by the three-term recurrence of the discrete orthogonal polynomials
    on the window (Stieltjes), in the arithmetic of `number` (float or F106). every: the list for each order
    0 .. d (the recurrence gives them all in one pass)."""
    p = (2 * (n // 2) - 1) / 2 if (pos is None and n % 2 == 0) else (n // 2 if pos is None else pos)
    ts = [number(i) - number(p) for i in range(n)]
    zero, one = number(0), number(1)
    prev, cur = [zero] * n, [one] * n
    dprev = [zero] * (deriv + 1)  # derivatives of orders 0 .. deriv at t = 0
    dcur = [one] + [zero] * deriv
    norm_prev = one
    out = [zero] * n
    history = []
    for k in range(d + 1):
        norm = sum((c * c for c in cur), zero)
        if k >= deriv:
            f = dcur[deriv] / norm  # the nu-th derivative of sum c_j P_j at 0: no nu! (that is the monomial form's)
            out = [o + f * c for o, c in zip(out, cur)]
        history.append(out)
        if k == d:
            break
        alpha = sum((t * c * c for t, c in zip(ts, cur)), zero) / norm
        beta = norm / norm_prev if k > 0 else zero
        nxt = [(t - alpha) * c - beta * q for t, c, q in zip(ts, cur, prev)]
        dnxt = [
            (zero - alpha) * dcur[v] + (number(v) * dcur[v - 1] if v > 0 else zero) - beta * dprev[v]
            for v in range(deriv + 1)
        ]
        prev, cur, dprev, dcur, norm_prev = cur, nxt, dcur, dnxt, norm
    return history if every else out


def proto_savgol(log, rec):
    from scipy.signal import savgol_coeffs

    M, D = S(), RD()
    r = D.ref("savgol")
    log("savgol_coeffs: max |h - h*| in units of u max|h*| (the draft contract is 8)")
    log(
        f"  {'window, order, deriv, pos':26s} {'recurrence, float64':>20s} {'recurrence, 106 bits':>21s} "
        f"{'scipy (lstsq)':>14s}"
    )
    for case in M.SAVGOL:
        sid = M.savgol_id(case)
        pos, idx = D.at(r, "coeffs", sid)
        ref = np.empty(idx.size)
        ref[idx] = r.methods["coeffs"].hi[pos]
        scale = U * (np.abs(ref).max() or 1.0)
        f64 = np.array(gram_coeffs(*case, float))
        dd = (
            np.array([float(c) for c in gram_coeffs(*case, F106)])
            if case[0] * case[1] <= 13000
            else np.full(idx.size, np.nan)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n, d, deriv, p = case
            sc = savgol_coeffs(n, d, deriv=deriv, pos=p, use="dot")
        row = {
            "float64": float(np.abs(f64 - ref).max() / scale),
            "bits106": float(np.abs(dd - ref).max() / scale),
            "scipy": float(np.abs(sc - ref).max() / scale),
            "sum_scipy": float(sc.sum()),
            "sum_exact": float(ref.sum()),
        }
        log(f"  {sid:26s} {row['float64']:20.3g} {row['bits106']:21.3g} {row['scipy']:14.3g}")
        rec[f"savgol/{sid}"] = row


def proto_savgol_scan(log, rec):
    """Where the recurrence stays within the contract: per window, the largest polyorder up to which every lower
    order's coefficients (deriv 0; the centre and the edge position 0 that mode='interp' uses) are within 16 u
    max|h*|, in float64 and at 106 bits."""
    import smooth_ref as R

    log("savgol_coeffs: the largest polyorder d with every order <= d within 16 u max|h*| (deriv 0; centre, pos = 0)")
    log(
        f"  {'window':>6s} {'orders tried':>13s} {'float64 centre':>15s} {'float64 edge':>13s} "
        f"{'106 bits centre':>16s} "
        f"{'106 bits edge':>14s}"
    )
    for n in (5, 11, 25, 51, 101, 301):
        top = min(n - 1, 60 if n <= 101 else 40)
        good = {}
        for pos, tag in ((None, "centre"), (0, "edge")):
            runs = {
                "f": gram_coeffs(n, top, 0, pos, float, every=True),
                "d": gram_coeffs(n, top, 0, pos, F106, every=True),
            }
            ok = {"f": -1, "d": -1}
            alive = {"f": True, "d": True}
            for d in range(top + 1):
                ref = np.array([R.fl(c) for c in R.savgol_exact(n, d, 0, pos)])
                scale = 16 * U * np.abs(ref).max()
                for key in ("f", "d"):
                    got = np.array([float(c) for c in runs[key][d]])
                    alive[key] = alive[key] and bool(np.all(np.abs(got - ref) <= scale))
                    if alive[key]:
                        ok[key] = d
            good[tag] = ok
        log(
            f"  {n:6d} {top:13d} {good['centre']['f']:15d} {good['edge']['f']:13d} {good['centre']['d']:16d} "
            f"{good['edge']['d']:14d}"
        )
        rec[f"savgol_scan/{n}"] = {
            "orders_tried": top,
            "float64": {k: v["f"] for k, v in good.items()},
            "bits106": {k: v["d"] for k, v in good.items()},
        }


def proto(log):
    rec = {}
    for fn in (
        proto_lowess,
        proto_loess,
        proto_loess_conditioning,
        proto_spline,
        proto_kde,
        proto_savgol,
        proto_savgol_scan,
    ):
        fn(log, rec)
        log()
    import flint
    import scipy

    meta = {
        "tool": TOOL,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python-flint": flint.__version__,
        "note": "float64 prototypes (numpy) of the planned algorithms against tests/smooth/ref",
    }
    log(write_records("prototype.json", meta, rec))


# ---------------------------------------------------------------- the C++ program


def kernels(log):
    exe = BUILD / "kernels"
    if not (exe.exists() and exe.stat().st_mtime >= SRC.stat().st_mtime):
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
    p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=CHILD_SECONDS)
    log(f"kernels (load average {os.getloadavg()[0]:.1f}):")
    log(p.stdout if not p.returncode else f"(exit status {p.returncode})\n{p.stderr[-2000:]}")


# ---------------------------------------------------------------- scipy


def scipy_records(log):
    import scipy
    import scipy.interpolate as si
    from scipy.signal import savgol_coeffs, savgol_filter
    from scipy.stats import gaussian_kde

    M, D = S(), RD()
    rec = {}
    rc = D.ref("spline")
    # the values on the test data, at the references' tuning
    for g, sid in enumerate(rc.group_inputs["set"].tolist()):
        s = M.scatter(sid)
        distinct = np.unique(s.x).size == s.n
        order = np.argsort(s.x, kind="stable")
        for c in ("1e-8", "1e-4", "1e-1", "1e2"):
            lam = float(rc.group_inputs[f"lam_ss_{c}"][g])
            rec[f"ss/{sid}/{c}"] = {
                "digest": s.digest,
                "lam": lam,
                **outcome(
                    lambda: (
                        si.make_smoothing_spline(
                            s.x[order], s.y[order], w=None if s.w is None else s.w[order], lam=lam
                        ).c
                    )
                ),
            }
        t = np.asarray(rc.group_inputs["t_lsq"][g])
        if t.size and distinct:
            rec[f"lsq/{sid}"] = {
                "digest": s.digest,
                **outcome(
                    lambda: si.make_lsq_spline(s.x[order], s.y[order], t, k=3, w=None if s.w is None else s.w[order]).c
                ),
            }
        if distinct and s.w is None and s.n >= 20:
            sval = 0.09 * s.n
            rec[f"splrep/{sid}"] = {
                "digest": s.digest,
                "s": sval,
                **outcome(lambda: (lambda sp: {"t": sp.t, "c": sp.c})(si.make_splrep(s.x[order], s.y[order], s=sval))),
            }
            rec[f"generate_knots/{sid}"] = {
                "digest": s.digest,
                "s": sval,
                **outcome(lambda: [len(t_) for t_ in si.generate_knots(s.x[order], s.y[order], s=sval)]),
            }
            rec[f"gcv/{sid}"] = {
                "digest": s.digest,
                **outcome(lambda: si.make_smoothing_spline(s.x[order], s.y[order]).c),
            }
    for did in M.density_ids(exclude=("degenerate",)):
        s = M.density(did)
        dat = s.x[None, :] if s.p == 1 else s.x.T
        pts = s.xe[None, :] if s.p == 1 else s.xe.T
        for rule in ("scott", "silverman"):
            rec[f"kde/{did}/{rule}"] = {
                "digest": s.digest,
                **outcome(
                    lambda: (
                        lambda k: {
                            "factor": k.factor,
                            "covariance": k.covariance,
                            "neff": k.neff,
                            "pdf": k(pts),
                            "logpdf": k.logpdf(pts),
                        }
                    )(gaussian_kde(dat, bw_method=rule, weights=s.w))
                ),
            }
    for case in M.SAVGOL:
        n, d, deriv, p = case
        rec[f"savgol/{M.savgol_id(case)}"] = outcome(lambda: savgol_coeffs(n, d, deriv=deriv, pos=p, use="dot"))
    for sid, (n, d, deriv) in __import__("smooth_ref").savgol_filter_items():
        for mode in M.SAVGOL_MODES:
            rec[f"savgol_filter/{sid}|{n},{d},{deriv}|{mode}"] = {
                "digest": M.signal(sid).digest,
                **outcome(lambda: savgol_filter(M.signal(sid).x, n, d, deriv=deriv, mode=mode, cval=M.SAVGOL_CVAL)),
            }
    # the rules
    s = M.scatter("random/200")
    base = si.make_smoothing_spline(s.x, s.y)(s.x)
    rec["rule/gcv_depends_on_the_unit_of_x"] = {
        str(f): float(np.abs(si.make_smoothing_spline(s.x * f, s.y)(s.x * f) - base).max())
        for f in (0.01, 10.0, 100.0, 1000.0)
    }
    g = M.rng("probe", "illposed")
    xl = np.sort(10 * g.random(10**4))
    yl = np.sin(xl) + 0.3 * g.standard_normal(10**4)
    rec["rule/gcv_fails_on_10^4_random_x"] = outcome(lambda: si.make_smoothing_spline(xl, yl).c[:3])
    t = M.scatter("ties/30")
    rec["rule/smoothing_spline_repeated_x"] = outcome(lambda: si.make_smoothing_spline(t.x, t.y, lam=1.0).c)
    rec["rule/smoothing_spline_unsorted_x"] = outcome(
        lambda: si.make_smoothing_spline(M.scatter("unsorted/30").x, M.scatter("unsorted/30").y, lam=1.0).c
    )
    rec["rule/smoothing_spline_lam_inf"] = outcome(lambda: si.make_smoothing_spline(s.x, s.y, lam=math.inf).c[:3])
    rec["rule/smoothing_spline_4_points"] = outcome(lambda: si.make_smoothing_spline(s.x[:4], s.y[:4], lam=1.0).c)
    s50 = M.scatter("random/50")
    w2 = np.full(50, 2.0)
    f = si.make_smoothing_spline(s50.x, s50.y, lam=1.0)(s50.x)
    rec["rule/smoothing_spline_weights_are_w"] = {  # w = 2 everywhere equals lam / 2 when the objective is sum w r^2
        "w2_vs_half_lam": float(
            np.abs(
                si.make_smoothing_spline(s50.x, s50.y, w=w2, lam=1.0)(s50.x)
                - si.make_smoothing_spline(s50.x, s50.y, lam=0.5)(s50.x)
            ).max()
        ),
        "w2_vs_quarter_lam": float(
            np.abs(
                si.make_smoothing_spline(s50.x, s50.y, w=w2, lam=1.0)(s50.x)
                - si.make_smoothing_spline(s50.x, s50.y, lam=0.25)(s50.x)
            ).max()
        ),
        "w2_vs_plain": float(np.abs(si.make_smoothing_spline(s50.x, s50.y, w=w2, lam=1.0)(s50.x) - f).max()),
    }
    wv = 0.5 + M.rng("probe", "w").random(50)
    tk = np.concatenate([[s50.x[0]] * 4, s50.x[5:-5:5], [s50.x[-1]] * 4])
    fw = si.make_lsq_spline(s50.x, s50.y, tk, w=wv).c
    Bm = si.BSpline.design_matrix(s50.x, tk, 3).toarray()
    rec["rule/lsq_spline_weights_are_w_squared"] = {
        "vs_sum_w2_r2": float(np.abs(fw - np.linalg.lstsq(Bm * wv[:, None], wv * s50.y, rcond=None)[0]).max()),
        "vs_sum_w_r2": float(
            np.abs(fw - np.linalg.lstsq(Bm * np.sqrt(wv)[:, None], np.sqrt(wv) * s50.y, rcond=None)[0]).max()
        ),
    }
    rec["rule/splrep_weights_are_w_squared"] = {
        "s_reached_sum_w2_r2": float(((wv * (s50.y - si.make_splrep(s50.x, s50.y, w=wv, s=3.0)(s50.x))) ** 2).sum()),
        "s_reached_sum_w_r2": float((wv * (s50.y - si.make_splrep(s50.x, s50.y, w=wv, s=3.0)(s50.x)) ** 2).sum()),
        "s": 3.0,
    }
    differ = {}
    for sval in (30.0, 17.0, 15.0, 12.0, 8.0):
        old = si.splrep(s.x, s.y, s=sval)
        new = si.make_splrep(s.x, s.y, s=sval)
        differ[str(sval)] = {
            "knots_splrep": int(len(old[0])),
            "knots_make_splrep": int(new.t.size),
            "max_difference": float(np.abs(si.splev(s.x, old) - new(s.x)).max()),
        }
    rec["rule/splrep_and_make_splrep_differ"] = differ
    rec["rule/make_splrep_unreachable_s"] = outcome(lambda: si.make_splrep(s.x, s.y, s=1e-3, nest=12).t.size)
    rec["rule/make_splrep_s0_is_interpolation"] = float(
        np.abs(si.make_splrep(s50.x, s50.y, s=0).c - si.make_interp_spline(s50.x, s50.y).c).max()
    )
    d = M.density("normal/200")
    rec["rule/gaussian_kde_scalar_is_a_factor"] = {
        "bw_method": 0.5,
        "kernel_sd": float(np.sqrt(gaussian_kde(d.x, 0.5).covariance[0, 0])),
        "sample_sd": float(np.std(d.x, ddof=1)),
    }
    rec["rule/gaussian_kde_points_are_columns"] = outcome(lambda: gaussian_kde(M.density("normal2/60").x).d)
    rec["rule/gaussian_kde_degenerate"] = outcome(lambda: gaussian_kde(M.density("allsame/10").x)(0.0))
    rec["rule/gaussian_kde_far_tail"] = outcome(
        lambda: {"pdf": gaussian_kde(d.x)(np.array([200.0])), "logpdf": gaussian_kde(d.x).logpdf(np.array([200.0]))}
    )
    sig = M.signal("noisy/100").x
    rec["rule/savgol_complex_input"] = outcome(lambda: savgol_filter(M.signal("complex/40").x, 7, 3).dtype.name)
    nan = sig.copy()
    nan[2] = np.nan
    for mode in M.SAVGOL_MODES:
        rec[f"rule/savgol_nan/{mode}"] = outcome(lambda: np.flatnonzero(np.isnan(savgol_filter(nan, 7, 3, mode=mode))))
    rec["rule/savgol_integer_input"] = outcome(lambda: savgol_filter(np.arange(20), 5, 2).dtype.name)
    rec["rule/savgol_polyorder_too_large"] = outcome(lambda: savgol_coeffs(5, 5))
    rec["rule/savgol_window_longer_than_axis"] = outcome(lambda: savgol_filter(sig[:5], 7, 3))
    rec["rule/savgol_window_longer_than_axis_nearest"] = outcome(
        lambda: savgol_filter(sig[:5], 7, 3, mode="nearest").shape
    )
    log(write_records("scipy.json", {"tool": TOOL, "scipy": scipy.__version__, "numpy": np.__version__}, rec))


# ---------------------------------------------------------------- statsmodels


def statsmodels_records(log):
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.nonparametric import bandwidths
    from statsmodels.nonparametric.smoothers_lowess import lowess

    M, D = S(), RD()
    rec = {}
    for sid in M.scatter_ids():
        s = M.scatter(sid)
        if s.w is not None:
            continue
        rng_ = float(np.ptp(s.x))
        for frac, tag in ((0.3, "0.3"), (2 / 3, "0.67"), (1.0, "1")):
            for it in (0, 3):
                for delta, dtag in ((0.0, "0"), (0.01 * rng_, "r")):
                    rec[f"lowess/{sid}/f{tag}/it{it}/delta{dtag}"] = {
                        "digest": s.digest,
                        **outcome(lambda: lowess(s.y, s.x, frac=frac, it=it, delta=delta, return_sorted=False)),
                    }
            rec[f"lowess_xvals/{sid}/f{tag}"] = {
                "digest": s.digest,
                **outcome(lambda: lowess(s.y, s.x, frac=frac, it=0, xvals=s.xe)),
            }
    s = M.scatter("random/200")
    rec["rule/lowess_returns_a_sorted_table"] = outcome(lambda: lowess(s.y, s.x).shape)
    yn = s.y.copy()
    yn[3] = np.nan
    rec["rule/lowess_drops_nan"] = outcome(lambda: lowess(yn, s.x).shape)
    for n, frac in ((10, 0.25), (10, 0.3), (7, 3 / 7), (30, 0.1), (200, 0.005)):
        rec[f"rule/lowess_window/{n}/{frac:.6g}"] = {
            "int(frac n + 1e-10)": int(frac * n + 1e-10),
            "max(2, .)": max(2, int(frac * n + 1e-10)),
        }
    base = lowess(s.y, s.x, frac=0.1, it=0, return_sorted=False)
    rec["rule/lowess_depends_on_the_unit_of_x"] = {
        str(f): float(np.abs(lowess(s.y, s.x * f, frac=0.1, it=0, return_sorted=False) - base).max())
        for f in (1e-8, 1e-4, 1e4, 1e8)
    }
    t = M.scatter("allsame/8")
    rec["rule/lowess_all_x_equal"] = outcome(lambda: lowess(t.y, t.x, it=0, return_sorted=False))
    rec["rule/lowess_ties"] = outcome(
        lambda: lowess(M.scatter("ties/30").y, M.scatter("ties/30").x, frac=0.3, it=0, return_sorted=False)
    )
    # densities and bandwidths
    for did in M.density_ids(tag="1d"):
        d = M.density(did)
        rec[f"bw/{did}"] = {
            "digest": d.digest,
            "bw_scott": outcome(lambda: bandwidths.bw_scott(d.x)),
            "bw_silverman": outcome(lambda: bandwidths.bw_silverman(d.x)),
            "bw_normal_reference": outcome(lambda: bandwidths.bw_normal_reference(d.x)),
        }
    d = M.density("normal/200")
    rk = D.ref("kde")
    h = float(rk.group_inputs["bw"][D.group_of(rk, "normal/200")])

    def fft_density():
        k = sm.nonparametric.KDEUnivariate(d.x)
        k.fit(kernel="gau", bw=h, fft=True, gridsize=512, cut=3)
        return {"support": k.support, "density": k.density, "bw": k.bw}

    rec["kde_fft/normal/200"] = {"digest": d.digest, **outcome(fft_density)}
    for kern in ("gau", "epa", "uni", "tri", "biw", "cos"):

        def direct(kern=kern, bw=h):
            k = sm.nonparametric.KDEUnivariate(d.x)
            k.fit(kernel=kern, bw=bw, fft=False, gridsize=64, cut=3)
            return {"support": k.support, "density": k.density, "bw": k.bw}

        rec[f"kde_kernel/normal/200/{kern}"] = {"digest": d.digest, **outcome(direct)}
        if kern in SM_HALF_WIDTH:
            rec[f"kde_kernel/normal/200/{kern}/sd"] = {
                "digest": d.digest,
                "bandwidth": h * SM_HALF_WIDTH[kern],
                **outcome(lambda: direct(bw=h * SM_HALF_WIDTH[kern])),
            }
    rec["rule/kernel_bandwidth_is_the_half_width"] = {
        kern: outcome(
            lambda: float(
                sm.nonparametric.KDEUnivariate(np.zeros(1))
                .fit(kernel=kern, bw=1.0, fft=False, gridsize=3, cut=1)
                .density[1]
            )
        )
        for kern in SM_HALF_WIDTH
    }
    for bw in ("normal_reference", "cv_ml", "cv_ls"):
        rec[f"kde_multivariate/normal/200/{bw}"] = {
            "digest": d.digest,
            **outcome(lambda: sm.nonparametric.KDEMultivariate(d.x, "c", bw=bw).bw),
        }
    d2 = M.density("normal2/60")
    rec["kde_multivariate/normal2/60/normal_reference"] = {
        "digest": d2.digest,
        **outcome(lambda: sm.nonparametric.KDEMultivariate(d2.x, "cc").bw),
    }
    # kernel regression at the references' bandwidths
    rr = D.ref("kreg")
    for g, sid in enumerate(rr.group_inputs["set"].tolist()):
        one = sid in M.scatter_ids()
        s = M.scatter(sid) if one else M.multi(sid)
        if s.w is not None:
            continue
        X, Xe = (s.x, s.xe) if one else (s.X, s.Xe)
        p = 1 if one else s.p
        for tag in ("a", "b"):
            bw = np.atleast_1d(np.asarray(rr.group_inputs[f"bw_{tag}"][g]))
            for reg in ("lc", "ll"):
                rec[f"kernel_reg/{sid}/{reg}_{tag}"] = {
                    "digest": s.digest,
                    "bw": bw.tolist(),
                    **outcome(lambda: sm.nonparametric.KernelReg(s.y, X, "c" * p, reg_type=reg, bw=bw).fit(Xe)),
                }
    s = M.scatter("random/200")
    for bw in ("cv_ls", "aic"):
        for reg in ("ll", "lc"):
            rec[f"kernel_reg_bw/random/200/{reg}/{bw}"] = {
                "digest": s.digest,
                **outcome(lambda: sm.nonparametric.KernelReg(s.y, s.x, "c", reg_type=reg, bw=bw).bw),
            }
    far = np.array([float(s.x.min()) - 50 * float(np.ptp(s.x))])
    rec["rule/kernel_reg_far_point"] = outcome(
        lambda: sm.nonparametric.KernelReg(s.y, s.x, "c", reg_type="lc", bw=[0.5]).fit(far)
    )
    log(
        write_records(
            "statsmodels.json", {"tool": TOOL, "statsmodels": statsmodels.__version__, "numpy": np.__version__}, rec
        )
    )


# ---------------------------------------------------------------- scikit-learn


def sklearn_records(log):
    import sklearn
    from sklearn.neighbors import KernelDensity

    M, D = S(), RD()
    rec = {}
    rk = D.ref("kde")
    for did in M.density_ids(exclude=("degenerate",)):
        d = M.density(did)
        X = d.x[:, None] if d.p == 1 else d.x
        Xe = d.xe[:, None] if d.p == 1 else d.xe
        for rule in ("scott", "silverman"):
            rec[f"bandwidth/{did}/{rule}"] = {
                "digest": d.digest,
                **outcome(lambda: KernelDensity(bandwidth=rule).fit(X).bandwidth_),
            }
        if d.p == 1:
            h = float(rk.group_inputs["bw"][D.group_of(rk, did)])
            for kern in ("gaussian", "epanechnikov", "tophat", "linear", "cosine"):
                for scale, tag in ((1.0, ""), (HALF_WIDTH.get(kern, 1.0), "/sd")):
                    if tag and kern not in HALF_WIDTH:
                        continue
                    bw = h * scale
                    rec[f"kde/{did}/{kern}{tag}"] = {
                        "digest": d.digest,
                        "bandwidth": bw,
                        **outcome(
                            lambda: (
                                KernelDensity(bandwidth=bw, kernel=kern, rtol=0, atol=0)
                                .fit(X, sample_weight=d.w)
                                .score_samples(Xe)
                            )
                        ),
                    }
    # the scale of the bandwidth: the density at the centre of one point at bandwidth 1 (R's scale would give
    # 3 / (4 sqrt 5) = 0.3354 for the Epanechnikov kernel; the support's half-width gives 0.75)
    rec["rule/kernel_bandwidth_is_the_half_width"] = {
        kern: outcome(
            lambda: float(
                np.exp(KernelDensity(bandwidth=1.0, kernel=kern).fit(np.zeros((1, 1))).score_samples(np.zeros((1, 1))))[
                    0
                ]
            )
        )
        for kern in ("epanechnikov", "tophat", "linear", "cosine")
    }
    rec["rule/default_bandwidth"] = outcome(lambda: KernelDensity().bandwidth)
    log(
        write_records("sklearn.json", {"tool": TOOL, "scikit-learn": sklearn.__version__, "numpy": np.__version__}, rec)
    )


# ---------------------------------------------------------------- R


def r_records(log):
    M, D = S(), RD()
    rec, digests = {}, {}
    rc, rk, rr = D.ref("spline"), D.ref("kde"), D.ref("kreg")
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        script = [
            "suppressMessages({library(KernSmooth); library(mgcv); library(MASS)})",
            "out <- list()",
            "rd <- function(f) as.numeric(readLines(f))",
            "lw <- function(x, y, ...) { o <- order(x); r <- numeric(length(x)); r[o] <- lowess(x, y, ...)$y; r }",
            "ctl <- loess.control(surface = 'direct', statistics = 'exact')",
        ]

        def put(name, arr):
            (d / name).write_text("\n".join(float(v).hex() for v in np.ravel(arr).tolist()) + "\n")
            return f'rd("{d / name}")'

        for k, sid in enumerate(M.scatter_ids()):
            s = M.scatter(sid)
            if s.w is not None or s.n < 2:
                continue
            digests[sid] = s.digest
            script += [f"x <- {put(f'x{k}', s.x)}", f"y <- {put(f'y{k}', s.y)}", f"xe <- {put(f'xe{k}', s.xe)}"]
            for f, tag in ((0.3, "0.3"), (2 / 3, "0.67"), (1.0, "1")):
                for it in (0, 3):
                    script.append(
                        f'out[["lowess/{sid}/f{tag}/it{it}/delta0"]] <- lw(x, y, f = {f!r}, iter = {it}, delta = 0)'
                    )
            script.append(f'out[["lowess_default/{sid}"]] <- lw(x, y)')
            if s.n >= 20 and "subnormal" not in s.tags and "ties" not in s.tags:
                for name, deg, span in (
                    ("d0_s0.75", 0, 0.75),
                    ("d1_s0.75", 1, 0.75),
                    ("d2_s0.75", 2, 0.75),
                    ("d2_s0.5", 2, 0.5),
                    ("d1_s1.5", 1, 1.5),
                ):
                    for fam in ("gaussian", "symmetric"):
                        script.append(
                            f'out[["loess/{sid}/{name}/{fam}"]] <- tryCatch({{m <- loess(y ~ x, span = {span}, '
                            f"degree = {deg}, "
                            f'family = "{fam}", control = ctl); c(m$enp, m$one.delta, m$two.delta, m$s, m$trace.hat, '
                            f"fitted(m), predict(m, data.frame(x = xe)))}}, error = function(e) NaN)"
                        )
                script.append(
                    f'out[["loess_interpolate/{sid}"]] <- tryCatch(fitted(loess(y ~ x)), error = function(e) NaN)'
                )
            if sid in rc.group_inputs["set"].tolist() and np.unique(s.x).size == s.n:
                g = D.group_of(rc, sid)
                rng_ = float(np.ptp(s.x))
                for c in ("1e-8", "1e-4", "1e-1", "1e2"):
                    lam_r = float(rc.group_inputs[f"lam_ss_{c}"][g]) / rng_**3
                    script.append(
                        f'out[["smooth_spline/{sid}/{c}"]] <- tryCatch({{m <- smooth.spline(x, y, lambda = {lam_r!r}, '
                        f"all.knots = TRUE); c(m$df, m$lambda, m$cv.crit, predict(m, x)$y)}}, error = function(e) NaN)"
                    )
                script.append(
                    f'out[["smooth_spline_gcv/{sid}"]] <- tryCatch({{m <- smooth.spline(x, y, all.knots = TRUE); '
                    f"c(m$df, m$lambda, m$cv.crit, m$spar, predict(m, x)$y)}}, error = function(e) NaN)"
                )
            if sid in M.scatter_ids(tag="plain"):
                g = D.group_of(rr, sid) if sid in rr.group_inputs["set"].tolist() else None
                if g is not None:
                    for tag in ("a", "b"):
                        bw = float(np.ravel(rr.group_inputs[f"bw_{tag}"][g])[0])
                        script.append(
                            f'out[["ksmooth/{sid}/{tag}"]] <- ksmooth(x, y, "normal", bandwidth = {bw / 0.3706506!r}, '
                            f"x.points = xe)$y"
                        )
                script.append(f'out[["dpill/{sid}"]] <- tryCatch(dpill(x, y), error = function(e) NaN)')
        for k, mid in enumerate(M.multi_ids()):
            s = M.multi(mid)
            if s.p > 4:
                continue
            digests[mid] = s.digest
            script += [
                f"X <- matrix({put(f'mX{k}', s.X)}, ncol = {s.p}, byrow = TRUE)",
                f"y <- {put(f'mY{k}', s.y)}",
                "df <- data.frame(X); df$y <- y",
            ]
            form = "y ~ " + " + ".join(f"X{a + 1}" for a in range(s.p))
            for name, deg, span in (("d0_s0.5", 0, 0.5), ("d1_s0.75", 1, 0.75), ("d2_s0.75", 2, 0.75)):
                for norm in ("FALSE", "TRUE"):
                    script.append(
                        f'out[["loess/{mid}/{name}/normalize{norm}"]] <- '
                        f"tryCatch({{m <- loess({form}, df, span = {span}, "
                        f"degree = {deg}, normalize = {norm}, control = ctl); c(m$enp, m$one.delta, m$two.delta, m$s, "
                        f"m$trace.hat, fitted(m))}}, error = function(e) NaN)"
                    )
        for k, did in enumerate(M.density_ids(tag="1d")):
            s = M.density(did)
            digests[did] = s.digest
            script.append(f"x <- {put(f'd{k}', s.x)}")
            script.append(
                f'out[["bw/{did}"]] <- c(bw.nrd0(x), bw.nrd(x), tryCatch(bw.ucv(x), error = function(e) NaN), '
                "tryCatch(bw.bcv(x), error = function(e) NaN), tryCatch(bw.SJ(x, method = 'ste'), error = "
                "function(e) NaN), tryCatch(bw.SJ(x, method = 'dpi'), error = function(e) NaN), "
                "tryCatch(dpik(x), error = function(e) NaN), tryCatch(bandwidth.nrd(x), error = function(e) NaN))"
            )
            if "degenerate" in s.tags or s.w is not None:
                continue
            h = float(rk.group_inputs["bw"][D.group_of(rk, did)])
            for kern in ("gaussian", "epanechnikov", "rectangular", "triangular", "biweight", "cosine"):
                script.append(
                    f'out[["density/{did}/{kern}"]] <- {{m <- density(x, bw = {h!r}, kernel = "{kern}", n = 512, '
                    "cut = 3); "
                    "c(m$x[1], m$x[512], m$y)}"
                )
        d2 = M.density("normal2/60")
        digests["normal2/60"] = d2.digest
        script += [
            f"X <- matrix({put('kde2d', d2.x)}, ncol = 2, byrow = TRUE)",
            'out[["kde2d/normal2/60"]] <- {m <- kde2d(X[, 1], X[, 2], n = 8); c(bandwidth.nrd(X[, 1]), '
            "bandwidth.nrd(X[, 2]), m$x, m$y, m$z)}",
        ]
        s = M.scatter("random/200")
        script += [
            f"x <- {put('px', s.x)}",
            f"y <- {put('py', s.y)}",
            'out[["mgcv_ps/random/200"]] <- {m <- gam(y ~ s(x, bs = "ps", k = 20, m = c(2, 2)), method = "GCV.Cp"); '
            "c(m$sp, sum(m$edf), fitted(m))}",
            'out[["locpoly/random/200"]] <- {m <- locpoly(x, y, bandwidth = 0.5, gridsize = 401); '
            "c(m$x[1], m$x[401], m$y)}",
        ]
        script.append(
            'for (k in names(out)) cat(k, "\\t", paste(sprintf("%a", out[[k]]), collapse = " "), "\\n", sep = "")'
        )
        (d / "run.R").write_text("\n".join(script))
        p = subprocess.run(
            ["Rscript", str(d / "run.R")],
            capture_output=True,
            text=True,
            timeout=CHILD_SECONDS,
            env=dict(os.environ, OMP_NUM_THREADS="1"),
        )
        if p.returncode:
            log(f"R failed: {p.stderr[-3000:]}")
            return
        ver = subprocess.run(
            [
                "Rscript",
                "-e",
                'cat(R.version.string, "; KernSmooth", as.character(packageVersion('
                '"KernSmooth")), "; mgcv", as.character(packageVersion("mgcv")), "; MASS", '
                'as.character(packageVersion("MASS")))',
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    for line in p.stdout.splitlines():
        key, _, vals = line.partition("\t")
        values = [float.fromhex(v) if v not in ("NA", "NaN") else math.nan for v in vals.split()]
        sid = next((i for i in digests if f"/{i}" in key), None)
        rec[key] = {"digest": digests.get(sid), "values": values}
    log(
        write_records(
            "r.json",
            {
                "tool": TOOL,
                "R": ver.stdout.strip(),
                "fields": {
                    "lowess": "the fit at x in input order",
                    "loess": "enp, one.delta, two.delta, s, trace.hat, fitted[, predict(xe)]",
                    "smooth_spline": "df, lambda (R's unit), cv.crit, the fit at x (lambda = lam_ss / range^3, "
                    "all.knots = TRUE)",
                    "smooth_spline_gcv": "df, lambda, cv.crit, spar, the fit at x",
                    "bw": "nrd0, nrd, ucv, bcv, SJ ste, SJ dpi, KernSmooth dpik, MASS bandwidth.nrd",
                    "density": "x[1], x[512], y (bw = the references' bandwidth)",
                    "ksmooth": "the fit at xe (bandwidth = the references' bw / 0.3706506)",
                    "kde2d": "the two bandwidths, x, y, z",
                    "mgcv_ps": "sp, the sum of edf, fitted",
                    "locpoly": "x[1], x[401], y",
                },
            },
            rec,
        )
    )


# ---------------------------------------------------------------- timings


def timings(log):
    import scipy.interpolate as si
    import statsmodels.api as sm
    from scipy.signal import savgol_filter
    from scipy.stats import gaussian_kde
    from sklearn.neighbors import KernelDensity
    from statsmodels.nonparametric.smoothers_lowess import lowess

    log(f"timings in ms (best of three; one run when marked *), load average {os.getloadavg()[0]:.1f}")
    g = np.random.Generator(np.random.PCG64(20261123))

    def data(n):
        x = np.sort(10 * g.random(n))
        return x, np.sin(x) + 0.3 * g.standard_normal(n)

    x4, y4 = data(10**4)
    x5, y5 = data(10**5)
    log(f"  statsmodels lowess n=1e4 frac=2/3 it=3 delta=0 *: {best(lambda: lowess(y4, x4), 1):.0f}")
    log(f"  statsmodels lowess n=1e4 frac=0.1 *: {best(lambda: lowess(y4, x4, frac=0.1), 1):.0f}")
    log(f"  statsmodels lowess n=1e4 delta=1%: {best(lambda: lowess(y4, x4, delta=0.1)):.1f}")
    log(f"  statsmodels lowess n=1e5 delta=1%: {best(lambda: lowess(y5, x5, delta=0.1)):.0f}")
    for n, (x, y) in ((10**3, data(10**3)), (10**4, (x4, y4)), (10**5, (x5, y5))):
        log(f"  scipy make_smoothing_spline lam=1 n={n:g}: {best(lambda: si.make_smoothing_spline(x, y, lam=1.0)):.2f}")
        log(f"  scipy make_splrep s=0.09n n={n:g}: {best(lambda: si.make_splrep(x, y, s=0.09 * n)):.2f}")
        log(f"  scipy splrep s=0.09n n={n:g}: {best(lambda: si.splrep(x, y, s=0.09 * n)):.2f}")
    xg = np.linspace(0, 10, 1000)
    yg = np.sin(xg) + 0.3 * g.standard_normal(1000)
    log(f"  scipy make_smoothing_spline GCV n=1e3 (grid): {best(lambda: si.make_smoothing_spline(xg, yg)):.0f}")
    x500, y500 = data(500)
    log(
        "  statsmodels KernelReg ll cv_ls n=500 *: "
        f"{best(lambda: sm.nonparametric.KernelReg(y500, x500, 'c').bw, 1):.0f}"
    )
    grid = np.linspace(-4, 4, 512)
    for n in (10**3, 10**4, 10**5):
        z = g.standard_normal(n)
        log(f"  scipy gaussian_kde 512 points n={n:g}: {best(lambda: gaussian_kde(z)(grid)):.2f}")
        if n >= 10**4:
            log(f"  statsmodels KDEUnivariate fft n={n:g}: {best(lambda: sm.nonparametric.KDEUnivariate(z).fit()):.2f}")
            log(
                f"  sklearn KernelDensity 512 points n={n:g} *: "
                f"{best(lambda: KernelDensity(bandwidth=0.3).fit(z[:, None]).score_samples(grid[:, None]), 1):.0f}"
            )
    z4 = g.standard_normal(10**4)
    log(f"  scipy gaussian_kde n=m=1e4 *: {best(lambda: gaussian_kde(z4)(z4), 1):.0f}")
    for n in (10**4, 10**5, 10**6):
        v = g.standard_normal(n)
        log(f"  scipy savgol_filter (11, 3) n={n:g}: {best(lambda: savgol_filter(v, 11, 3)):.2f}")
        if n >= 10**5:
            log(f"  scipy savgol_filter (101, 3) n={n:g}: {best(lambda: savgol_filter(v, 101, 3)):.2f}")
    r = (
        "set.seed(1); t <- function(e) round(1000 * system.time(e)[['elapsed']], 1); "
        "suppressMessages(library(KernSmooth));"
        "x <- sort(runif(1e4, 0, 10)); y <- sin(x) + rnorm(1e4, 0, 0.3); z <- rnorm(1e5);"
        "cat('  R lowess n=1e4 delta=0:', t(lowess(x, y, delta = 0)), '\\n');"
        "cat('  R lowess n=1e4 default delta:', t(lowess(x, y)), '\\n');"
        "cat('  R loess n=1e4 degree 2 interpolate:', t(loess(y ~ x)), '\\n');"
        "cat('  R loess n=1e4 degree 2 direct:', t(loess(y ~ x, control = loess.control(surface = 'direct'))), '\\n');"
        "cat('  R smooth.spline GCV n=1e4:', t(smooth.spline(x, y)), '\\n');"
        "cat('  R smooth.spline GCV n=1e4 all.knots:', t(smooth.spline(x, y, all.knots = TRUE)), '\\n');"
        "cat('  R ksmooth normal n=1e4:', t(ksmooth(x, y, 'normal', 0.5, x.points = x)), '\\n');"
        "cat('  R locpoly n=1e4:', t(locpoly(x, y, bandwidth = 0.5)), '\\n');"
        "cat('  R density n=1e5:', t(density(z)), '\\n'); cat('  R bw.SJ n=1e5:', t(bw.SJ(z)), '\\n')"
    )
    p = subprocess.run(
        ["Rscript", "-e", r],
        capture_output=True,
        text=True,
        timeout=CHILD_SECONDS,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
    )
    log(p.stdout.rstrip() if not p.returncode else f"R failed: {p.stderr[-2000:]}")
    log("  not re-run (the draft's values stand): R loess n=1e5 (70 s), sizes of 1e6, statsmodels lowess n=1e5 delta=0")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default all of {', '.join(SECTIONS)})")
    args = p.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)})")
    log = Log()
    table = {
        "proto": proto,
        "kernels": kernels,
        "scipy": scipy_records,
        "statsmodels": statsmodels_records,
        "sklearn": sklearn_records,
        "r": r_records,
        "time": timings,
    }
    for s in SECTIONS:
        if not args.sections or s in args.sections:
            table[s](log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
