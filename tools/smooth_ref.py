"""Reference values for sd.smooth (docs/design/smooth.md §4, §7.2) on the test data of
tests/smooth/samples.py: exact rationals (python-flint's fmpq, fmpq_mat) and Arb.

    .venv/bin/python tools/smooth_ref.py                     # every family, on 3 processes
    .venv/bin/python tools/smooth_ref.py local savgol        # only these families
    .venv/bin/python tools/smooth_ref.py --list              # the families and their data

writes tests/smooth/ref/<file>.npz (layout: tests/reffile.py), read by tests/smooth/refdata.py. A point
has the inputs "set" (the id in samples.py), "kind" (0: a data point, a coefficient or an output sample;
1: an evaluation point of the set; 2: a far evaluation point), "index" (its number in the set) and
"digest" (the data's digest); a group is one set and holds the tuning the references were computed with.
The data are doubles taken as exact rationals; a reference is the true value of the estimator for those
doubles and that tuning, rounded once (a double-double hi + lo). The tuning (window size, bandwidth,
lambda, knots) is always given: no reference searches for one (§4).

    local       1-D local fits at every data point and evaluation point, exact rationals. The window of
                x0 is the k nearest points by exact distance and its radius h the k-th distance (h =
                span * the largest distance when span > 1); the weights are tricube (1 - (d / h)^3)^3
                times the set's weights (0 at d >= h); the fit is the weighted polynomial of the degree
                through the points of positive weight (the degree lowered to their distinct x minus 1;
                h = 0: the weighted mean of the points at x0). Methods lowess_f<frac> (degree 1, k =
                min(n, max(2, int(frac n + 1e-10))), statsmodels' rule) and loess_d<degree>_s<span>
                (k = floor(n span + 1e-5), R's rule; span > 1: every point, the radius sqrt(span) times the
                largest distance as R's code computes it (its documentation says span^(1/p)), in Arb).
                Fields: scale = sum |l_j| |y_j| over the hat row l
                (the contract's scale), leverage = l_ii (NaN off the data), kappa = a c / (a c - b^2) of
                the local line's moments (degree 1; the 2-norm condition of the weighted design
                otherwise), npos = the points of positive weight
    local_nd    the same for (n, p) data, Arb (the distances have square roots): loess_d<degree>_s<span>
                with Euclidean distances on the columns as given (normalize off). Windows whose design
                loses rank (exact rational rank) are dropped
    spline      one point per coefficient: ss_<c> the smoothing spline minimizing sum w r^2 + lam
                int f''^2 over the cubic splines with scipy's knots (every distinct x, the ends four
                times), lam = c (x_max - x_min)^3 as a double (group input lam_ss_<c>); ps_<lam> the
                P-spline (k = 3, second differences, the default n_segments, knots xb + j h on the left
                of xe and xe + j h from it); lsq the least-squares spline minimizing sum (w r)^2 on
                every fourth distinct x as interior knots. Exact normal equations (fmpq_mat). The knots
                are group inputs (t_ss, t_ps, t_lsq)
    spline_fit  one point per data point: the fitted value of each of those, with the field leverage =
                L_ii; over the groups trl_<m> = tr L, kappa_<m> (field of trl: the 2-norm condition of
                the stacked matrix, sqrt(cond(B'WB + lam P)), by numpy from the exact matrix and its
                exact inverse), and for unweighted sets of distinct x gcv_<m> = n RSS / (n - tr L)^2 and
                cv_<m> = sum ((y - f) / (1 - L_ii))^2 / n
    kreg        Nadaraya-Watson (nw_a, nw_b) and local linear (ll_a, ll_b) with a product Gaussian
                kernel, bandwidths a = range / 20 and b = range / 4 per column (1-D) or range / 5 and
                range / 2 (n-D) as doubles (group inputs bw_a, bw_b); Arb, every point summed; a far
                point (kind 2: 50 ranges below the data) where every double weight underflows. Field
                scale = sum |l_j| |y_j|
    kde         pdf, logpdf and (1-D) cdf of the Gaussian estimator with covariance H (group input H:
                1-D the square of R's nrd0 as a double, n-D scipy's Scott matrix as doubles); 1-D also
                pdf_<kernel> for epanechnikov, rectangular, triangular, biweight and cosine with the
                bandwidth the kernel's standard deviation (R's scale), and pdf_reflect for a bounded
                sample (f(x) + f(2 lo - x) + f(2 hi - x) inside the bounds, 0 outside). Arb, every point
                summed
                n-D also pdf_diag and logpdf_diag with the diagonal covariance diag(bw_diag^2) (group input
                bw_diag: the doubles sqrt(H_ll))
    bandwidth   one point per 1-D sample: scott and silverman (scipy's factor times the weighted sample
                standard deviation), nrd0 and nrd (R's rules, quantiles of type 7), and at h = the
                double nearest nrd the criteria cvls_value (int fhat^2 - 2/n sum fhat_-i(x_i)) and
                cvml_value (sum log fhat_-i(x_i)); for the unweighted samples of positive spread the roots
                and minimizers (bisection on exact rational h with Arb signs to 2^-50 relative; field width):
                sj_ste and sj_dpi (R's bw.SJ equations on its pair counts of nb = 1000 bins, from the doubles
                R's bw_den computes), isj (Botev's fixed point on the exact linear binning of n_grid = 256
                points over the doubles [min - range / 2, max + range / 2]; group input isj_n), cvls_min and
                cvml_min (the derivative's root next to the smallest criterion of 33 points of log h over
                nrd / 64 .. 64 nrd)
    savgol      one point per coefficient of samples.SAVGOL: the exact least-squares filter in
                window order (use='dot', delta = 1)
    savgol_filter  one point per output sample: the exact filtered value of the 1-D real signals for
                samples.SAVGOL_FILTERS and every mode (methods interp, mirror, nearest, constant, wrap;
                constant with cval = samples.SAVGOL_CVAL). Field scale = sum |h_j| |x_j| over the window

Settling: an exact rational is exact; an Arb value is settled to 2^-110 relative. Workers are forked,
replaced every 8 tasks (maxtasksperchild), each task under a time limit (SIGALRM). No mpmath (the
project's rule). The console output also goes to build/logs/smooth_ref/.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import signal
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arbref  # noqa: E402
from arbref import NotSettled  # noqa: E402
from flint import arb, arb_mat, fmpq, fmpq_mat  # noqa: E402

OUT = ROOT / "tests" / "smooth" / "ref"
TOOL = "tools/smooth_ref.py"
SECONDS = 900  # a task's time limit
NAN = math.nan

LOWESS = (("lowess_f0.3", 0.3), ("lowess_f0.67", 2 / 3), ("lowess_f1", 1.0))
LOESS = (
    ("loess_d0_s0.75", 0, 0.75),
    ("loess_d1_s0.75", 1, 0.75),
    ("loess_d2_s0.75", 2, 0.75),
    ("loess_d2_s0.5", 2, 0.5),
    ("loess_d1_s1.5", 1, 1.5),
)
LOESS_ND = (("loess_d0_s0.5", 0, 0.5), ("loess_d1_s0.75", 1, 0.75), ("loess_d2_s0.75", 2, 0.75))
SS_C = ("1e-8", "1e-4", "1e-1", "1e2")
PS_LAM = ("1e-2", "1", "1e3")
KERNELS = {"epanechnikov": 5, "rectangular": 3, "triangular": 6, "biweight": 7}  # a^2 of the support [-a, a]
KREG_1D = ("random/50", "clustered/40", "outliers/40", "weights/30", "ties/30", "line/20")
KREG_ND = ("uniform2/60", "uniform3/60", "scales2/50")


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def S():
    from tests.smooth import samples

    return samples


# ---------------------------------------------------------------- exact helpers


def Q(v) -> fmpq:
    return fmpq(*float(v).as_integer_ratio())


def fr(q: fmpq) -> Fraction:
    return Fraction(int(q.p), int(q.q))


def dd_of(q):
    """(hi, lo) of an exact rational: hi rounded to nearest (ties to even), lo the rest rounded;
    (inf, 0) past the double range."""
    q = fr(q) if isinstance(q, fmpq) else q
    try:
        hi = float(q)
    except OverflowError:
        return (math.inf if q > 0 else -math.inf), 0.0
    if math.isinf(hi):
        return hi, 0.0
    return hi, float(q - Fraction(hi))


def fl(q) -> float:
    return dd_of(q)[0]


def qabs(q):
    return -q if q < 0 else q


def arb_q(q: fmpq):
    return arb(q.p) / arb(q.q)


def settle(fn):
    return arbref.to_dd(arbref.rigorous(fn))


def arb_solve(M, b):
    try:
        return M.solve(b)
    except ZeroDivisionError as e:  # not certified at this precision
        raise NotSettled(str(e)) from None


# ---------------------------------------------------------------- local fits, 1-D (exact)


def lowess_k(n, frac):
    return min(n, max(2, int(frac * n + 1e-10)))


def loess_q(n, span):
    return min(n, int(math.floor(n * span + 1e-5)))


def local_1d(X, Y, W, x0, k, degree):
    """(value, scale, l_at_x0 or None, kappa, npos) of the exact local fit at x0, or None when no point has
    a positive weight. X, Y, W: lists of fmpq; l_at_x0 is a function i -> l_i for the leverage."""
    n = len(X)
    d = [qabs(x - x0) for x in X]
    if k < 1:
        return None
    h = sorted(d)[k - 1]
    if h == 0:
        idx = [j for j in range(n) if d[j] == 0 and W[j] > 0]
        v = {j: W[j] for j in idx}
        deg = 0
    else:
        v = {}
        for j in range(n):
            if d[j] < h and W[j] > 0:
                r = d[j] / h
                v[j] = (1 - r * r * r) ** 3 * W[j]
        idx = sorted(v)
        deg = min(degree, len({X[j] for j in idx}) - 1)
    if not idx:
        return None
    u = {j: (X[j] - x0) / h if h != 0 else fmpq(0) for j in idx}
    m = [sum((v[j] * u[j] ** e for j in idx), fmpq(0)) for e in range(2 * deg + 1)]
    M = fmpq_mat(deg + 1, deg + 1, [m[a + b] for a in range(deg + 1) for b in range(deg + 1)])
    e0 = fmpq_mat(deg + 1, 1, [1] + [0] * deg)
    a = M.solve(e0)
    coef = [a[i, 0] for i in range(deg + 1)]
    row = {j: v[j] * sum((coef[e] * u[j] ** e for e in range(deg + 1)), fmpq(0)) for j in idx}
    value = sum((row[j] * Y[j] for j in idx), fmpq(0))
    scale = sum((qabs(row[j]) * qabs(Y[j]) for j in idx), fmpq(0))
    if degree == 1 and deg == 1:
        kappa = fl(m[0] * m[2] / (m[0] * m[2] - m[1] * m[1]))
    elif deg == 0:
        kappa = 1.0
    else:  # sqrt(cond_2(M)) from the largest eigenvalues of M and of its exact inverse (both well conditioned)
        Mi = M.inv()
        kappa = math.sqrt(
            _eig_max([[M[a, b] for b in range(deg + 1)] for a in range(deg + 1)])
            * _eig_max([[Mi[a, b] for b in range(deg + 1)] for a in range(deg + 1)])
        )
    return value, scale, row, kappa, len(idx)


def local_1d_wide(X, Y, W, x0, degree, span):
    """local_1d for span > 1 as R computes it: R scales the largest squared distance by span, so the radius is
    sqrt(span) times the largest distance whatever the dimension (R's documentation says span^(1/p)). The radius
    is irrational: the weights, the fit and its scale in Arb; returns ((hi, lo), scale, row, kappa, npos)."""
    n = len(X)
    d2 = [(x - x0) ** 2 for x in X]
    h2 = max(d2) * Q(span)
    if h2 == 0:  # every point at x0: their weighted mean, exact
        r = local_1d(X, Y, W, x0, 1, 0)
        if r is None:
            return None
        value, scale, row, kappa, npos = r
        return dd_of(value), fl(scale), {j: fl(v) for j, v in row.items()}, kappa, npos
    idx = [j for j in range(n) if d2[j] < h2 and W[j] > 0]
    if not idx:
        return None
    deg = min(degree, len({X[j] for j in idx}) - 1)
    t = deg + 1

    def fit():
        h = arb_q(h2).sqrt()
        v, phi = [], []
        for j in idx:
            r = (arb_q(d2[j]) / arb_q(h2)).sqrt()
            v.append((1 - r**3) ** 3 * arb_q(W[j]))
            phi.append([(arb_q(X[j] - x0) / h) ** e for e in range(t)])
        M = arb_mat(
            t,
            t,
            [sum((v[i] * phi[i][a] * phi[i][b] for i in range(len(idx))), arb(0)) for a in range(t) for b in range(t)],
        )
        a = arb_solve(M, arb_mat(t, 1, [1] + [0] * (t - 1)))
        return M, [v[i] * sum((a[e, 0] * phi[i][e] for e in range(t)), arb(0)) for i in range(len(idx))]

    hi, lo = settle(lambda: sum((r * arb_q(Y[j]) for r, j in zip(fit()[1], idx)), arb(0)))
    with arbref.precision(256):
        M, row = fit()
        scale = float(sum((abs(r) * abs(arb_q(Y[j])) for r, j in zip(row, idx)), arb(0)).mid())
        Mi = M.inv()
        kappa = math.sqrt(
            _eig_max([[float(M[a, b].mid()) for b in range(t)] for a in range(t)])
            * _eig_max([[float(Mi[a, b].mid()) for b in range(t)] for a in range(t)])
        )
        if degree == 1 and deg == 1:  # the local line's kappa_1 = a c / (a c - b^2), as local_1d gives it
            a_, b_, c_ = (float(M[i, j].mid()) for i, j in ((0, 0), (0, 1), (1, 1)))
            kappa = a_ * c_ / (a_ * c_ - b_ * b_)
        elif deg == 0:
            kappa = 1.0
        rowd = {j: float(r.mid()) for r, j in zip(row, idx)}
    return (hi, lo), scale, rowd, kappa, len(idx)


def local_task(sid):
    s = S().scatter(sid)
    X, Y = [Q(v) for v in s.x], [Q(v) for v in s.y]
    W = [Q(v) for v in s.w] if s.w is not None else [fmpq(1)] * s.n
    settings = [(name, lowess_k(s.n, f), 1, None) for name, f in LOWESS]
    settings += [(name, loess_q(s.n, sp), deg, sp) for name, deg, sp in LOESS]
    points = []
    targets = [(0, i, X[i]) for i in range(s.n)] + [(1, i, Q(v)) for i, v in enumerate(s.xe)]
    for kind, i, x0 in targets:
        ms = {}
        for name, k, deg, span in settings:
            if span is not None and span > 1:
                r = local_1d_wide(X, Y, W, x0, deg, span)
                if r is None:
                    ms[name] = None
                    continue
                (hi, lo), scale, row, kappa, npos = r
                lev = row.get(i, 0.0) if kind == 0 else NAN
                ms[name] = (hi, lo, {"scale": scale, "leverage": lev, "kappa": kappa, "npos": float(npos)})
                continue
            r = local_1d(X, Y, W, x0, k, deg)
            if r is None:
                ms[name] = None
                continue
            value, scale, row, kappa, npos = r
            lev = fl(row.get(i, fmpq(0))) if kind == 0 else NAN
            ms[name] = (*dd_of(value), {"scale": fl(scale), "leverage": lev, "kappa": kappa, "npos": float(npos)})
        points.append((kind, i, ms))
    return {"set": sid, "digest": s.digest, "ginputs": {"n": s.n}, "points": points, "gmethods": {}}


# ---------------------------------------------------------------- local fits, n-D (Arb)


def monomials(p, degree):
    """The exponents of the local basis: 1, the p coordinates, then the products u_a u_b (a <= b)."""
    out = [()]
    if degree >= 1:
        out += [(a,) for a in range(p)]
    if degree >= 2:
        out += [(a, b) for a in range(p) for b in range(a, p)]
    return out


def local_nd(X, Y, x0, q, degree):
    n, p = len(X), len(x0)
    diff = [[X[j][a] - x0[a] for a in range(p)] for j in range(n)]
    d2 = [sum((c * c for c in row), fmpq(0)) for row in diff]
    h2 = sorted(d2)[q - 1]
    if h2 == 0:
        return None
    idx = [j for j in range(n) if d2[j] < h2]
    terms = monomials(p, degree)
    phi_q = [[math.prod((diff[j][a] for a in t), start=fmpq(1)) for t in terms] for j in idx]
    if fmpq_mat(len(idx), len(terms), [c for row in phi_q for c in row]).rank() < len(terms):
        return None

    def fit():
        h = arb_q(h2).sqrt()
        v, phi = [], []
        for j in idx:
            r = (arb_q(d2[j]) / arb_q(h2)).sqrt()
            v.append((1 - r**3) ** 3)
            phi.append([math.prod((arb_q(diff[j][a]) / h for a in t), start=arb(1)) for t in terms])
        t = len(terms)
        M = arb_mat(
            t,
            t,
            [sum((v[i] * phi[i][a] * phi[i][b] for i in range(len(idx))), arb(0)) for a in range(t) for b in range(t)],
        )
        a = arb_solve(M, arb_mat(t, 1, [1] + [0] * (t - 1)))
        row = [v[i] * sum((a[e, 0] * phi[i][e] for e in range(t)), arb(0)) for i in range(len(idx))]
        return M, row

    def value():
        _, row = fit()
        return sum((row[i] * arb_q(Y[j]) for i, j in enumerate(idx)), arb(0))

    hi, lo = settle(value)
    with arbref.precision(256):
        M, row = fit()
        scale = float(sum((abs(row[i]) * abs(arb_q(Y[j])) for i, j in enumerate(idx)), arb(0)).mid())
        t = len(terms)
        Mi = M.inv()
        kappa = math.sqrt(
            _eig_max([[float(M[a, b].mid()) for b in range(t)] for a in range(t)])
            * _eig_max([[float(Mi[a, b].mid()) for b in range(t)] for a in range(t)])
        )
        lev = {j: float(row[i].mid()) for i, j in enumerate(idx)}
    return hi, lo, scale, lev, kappa, len(idx)


def local_nd_task(mid):
    s = S().multi(mid)
    X = [[Q(v) for v in row] for row in s.X]
    Y = [Q(v) for v in s.y]
    targets = [(0, i, X[i]) for i in range(s.n)] + [(1, i, [Q(v) for v in row]) for i, row in enumerate(s.Xe)]
    points = []
    for kind, i, x0 in targets:
        ms = {}
        for name, deg, span in LOESS_ND:
            r = local_nd(X, Y, x0, loess_q(s.n, span), deg)
            if r is None:
                ms[name] = None
                continue
            hi, lo, scale, lev, kappa, npos = r
            ms[name] = (
                hi,
                lo,
                {
                    "scale": scale,
                    "leverage": lev.get(i, 0.0) if kind == 0 else NAN,
                    "kappa": kappa,
                    "npos": float(npos),
                },
            )
        points.append((kind, i, ms))
    return {"set": mid, "digest": s.digest, "ginputs": {"n": s.n, "p": s.p}, "points": points, "gmethods": {}}


# ---------------------------------------------------------------- splines (exact)


def span_of(t, k, v):
    """The knot interval of v: the largest l in [k, len(t) - k - 2] with t[l] <= v (the end pieces extended)."""
    lo, hi = k, len(t) - k - 2
    l = lo
    for j in range(lo, hi + 1):
        if t[j] <= v:
            l = j
    while l < hi and t[l] == t[l + 1]:
        l += 1
    return l


def basis(t, k, l, v, nu=0):
    """[B_{l-k+j, k}^{(nu)}(v)], j = 0 .. k, by the polynomial pieces of the interval [t_l, t_{l+1}]."""
    b = [fmpq(1)]
    for d in range(1, k + 1):
        nb = [fmpq(0)] * (d + 1)
        for j in range(d + 1):
            i = l - d + j
            if d <= k - nu:
                if j >= 1:
                    nb[j] += (v - t[i]) / (t[i + d] - t[i]) * b[j - 1]
                if j <= d - 1:
                    nb[j] += (t[i + d + 1] - v) / (t[i + d + 1] - t[i + 1]) * b[j]
            else:
                if j >= 1:
                    nb[j] += d * b[j - 1] / (t[i + d] - t[i])
                if j <= d - 1:
                    nb[j] -= d * b[j] / (t[i + d + 1] - t[i + 1])
        b = nb
    return b


def roughness(t):
    """Omega_ij = int B_i'' B_j'' over [t_3, t_{-4}] of the cubic B-splines on t, exact (each B'' is linear on a
    knot interval)."""
    nc = len(t) - 4
    om = [[fmpq(0)] * nc for _ in range(nc)]
    for l in range(3, nc):
        if t[l] == t[l + 1]:
            continue
        f0, f1 = basis(t, 3, l, t[l], 2), basis(t, 3, l, t[l + 1], 2)
        h = (t[l + 1] - t[l]) / 6
        for a in range(4):
            for b in range(4):
                om[l - 3 + a][l - 3 + b] += h * (2 * f0[a] * f0[b] + f0[a] * f1[b] + f1[a] * f0[b] + 2 * f1[a] * f1[b])
    return om


def difference_penalty(nc, order=2):
    D = np.diff(np.eye(nc, dtype=np.int64), n=order, axis=0)
    P = D.T @ D
    return [[fmpq(int(P[i, j])) for j in range(nc)] for i in range(nc)]


def pspline_knots(xb, xe, nseg, k=3):
    """The doubles of the uniform knots: xb + j h left of xe, xe + j h from it (h = (xe - xb) / nseg in double)."""
    h = (xe - xb) / nseg
    return np.array([xb + j * h for j in range(-k, nseg)] + [xe + j * h for j in range(0, k + 1)])


def _eig_max(A):
    """The largest eigenvalue of a symmetric positive definite matrix given exactly (fmpq) or as doubles: an
    absolute error of a few u times itself, whatever the matrix's condition."""
    return float(np.linalg.eigvalsh(np.array([[c if isinstance(c, float) else fl(c) for c in row] for row in A]))[-1])


def penalized(t, k, X, Y, W, P, lam):
    """The exact solution of min sum W r^2 + lam c'Pc: (coefficients, fitted, leverage, kappa) or None when the
    system is singular."""
    nc = len(t) - k - 1
    rows = []
    for x in X:
        l = span_of(t, k, x)
        rows.append((l - k, basis(t, k, l, x)))
    A = [[fmpq(0)] * nc for _ in range(nc)]
    rhs = [fmpq(0)] * nc
    for (o, b), y, w in zip(rows, Y, W):
        for a in range(k + 1):
            rhs[o + a] += w * y * b[a]
            for c in range(k + 1):
                A[o + a][o + c] += w * b[a] * b[c]
    if P is not None:
        for i in range(nc):
            for j in range(nc):
                A[i][j] += lam * P[i][j]
    Am = fmpq_mat(nc, nc, [c for row in A for c in row])
    try:
        inv = Am.inv()
    except ZeroDivisionError:
        return None
    coef = [sum((inv[i, j] * rhs[j] for j in range(nc)), fmpq(0)) for i in range(nc)]
    fitted = [sum((b[a] * coef[o + a] for a in range(k + 1)), fmpq(0)) for o, b in rows]
    lev = [
        w * sum((b[a] * inv[o + a, o + c] * b[c] for a in range(k + 1) for c in range(k + 1)), fmpq(0))
        for (o, b), w in zip(rows, W)
    ]
    kappa = math.sqrt(_eig_max(A) * _eig_max([[inv[i, j] for j in range(nc)] for i in range(nc)]))
    return coef, fitted, lev, kappa


def spline_sets():
    M = S()
    return [i for i in M.scatter_ids(exclude=("probe", "subnormal")) if np.unique(M.scatter(i).x).size >= 5]


def spline_task(sid):
    s = S().scatter(sid)
    X, Y = [Q(v) for v in s.x], [Q(v) for v in s.y]
    W = [Q(v) for v in s.w] if s.w is not None else [fmpq(1)] * s.n
    xd = np.unique(s.x)
    rng_ = float(xd[-1] - xd[0])
    plain = s.w is None and xd.size == s.n
    t_ss = np.concatenate([[xd[0]] * 3, xd, [xd[-1]] * 3])
    nseg = math.ceil(min(s.n / 4, 40))
    t_ps = pspline_knots(float(xd[0]), float(xd[-1]), nseg)
    t_lsq = np.concatenate([[xd[0]] * 4, xd[3:-3:4], [xd[-1]] * 4]) if xd.size >= 10 else np.zeros(0)
    gin = {"n": s.n, "t_ss": t_ss, "t_ps": t_ps, "t_lsq": t_lsq, "n_segments": nseg}
    jobs = []
    om = roughness([Q(v) for v in t_ss])
    for c in SS_C:
        lam = float(c) * rng_**3
        gin[f"lam_ss_{c}"] = lam
        jobs.append((f"ss_{c}", t_ss, W, om, Q(lam)))
    pen = difference_penalty(len(t_ps) - 4)
    for c in PS_LAM:
        jobs.append((f"ps_{c}", t_ps, W, pen, Q(float(c))))
    if t_lsq.size:
        jobs.append(("lsq", t_lsq, [w * w for w in W], None, fmpq(0)))
    coefs, fits, gm = {}, {}, {}
    for name, t, Wj, P, lam in jobs:
        r = penalized([Q(v) for v in t], 3, X, Y, Wj, P, lam)
        if r is None:
            continue
        coef, fitted, lev, kappa = r
        coefs[name] = [dd_of(c) for c in coef]
        fits[name] = [(*dd_of(f), {"leverage": fl(v)}) for f, v in zip(fitted, lev)]
        trl = sum(lev, fmpq(0))
        gm[f"trl_{name}"] = (*dd_of(trl), {"kappa": kappa})
        if plain and name != "lsq":
            n = fmpq(s.n)
            rss = sum(((y - f) ** 2 for y, f in zip(Y, fitted)), fmpq(0))
            if trl != n:
                gm[f"gcv_{name}"] = (*dd_of(n * rss / (n - trl) ** 2), {})
            if all(v != 1 for v in lev):
                gm[f"cv_{name}"] = (
                    *dd_of(sum((((y - f) / (1 - v)) ** 2 for y, f, v in zip(Y, fitted, lev)), fmpq(0)) / n),
                    {},
                )
    ncmax = max(len(v) for v in coefs.values())
    cpoints = [(0, i, {m: (*v[i], {}) for m, v in coefs.items() if i < len(v)}) for i in range(ncmax)]
    fpoints = [(0, i, {m: v[i] for m, v in fits.items()}) for i in range(s.n)]
    base = {"set": sid, "digest": s.digest}
    return (
        {**base, "ginputs": gin, "points": cpoints, "gmethods": {}},
        {**base, "ginputs": {"n": s.n}, "points": fpoints, "gmethods": gm},
    )


# ---------------------------------------------------------------- kernel regression (Arb)


def kreg_point(Xq, Yq, Wq, x0, bw, degree):
    """(hi, lo, scale) of the product-Gaussian kernel regression at x0 (lists of fmpq; bw per column)."""
    n, p = len(Xq), len(x0)
    z = [[(Xq[j][a] - x0[a]) / bw[a] for a in range(p)] for j in range(n)]
    e = [sum((c * c for c in row), fmpq(0)) / 2 for row in z]
    e0 = min(e)  # the common factor exp(-e0) cancels in the ratio: taken out so that no ball is 0 +- tiny

    def row():
        v = [arb_q(Wq[j]) * (-arb_q(e[j] - e0)).exp() for j in range(n)]
        if degree == 0:
            tot = sum(v, arb(0))
            return [c / tot for c in v]
        t = p + 1
        phi = [[arb(1)] + [arb_q(c) for c in z[j]] for j in range(n)]
        M = arb_mat(
            t, t, [sum((v[j] * phi[j][a] * phi[j][b] for j in range(n)), arb(0)) for a in range(t) for b in range(t)]
        )
        a = arb_solve(M, arb_mat(t, 1, [1] + [0] * p))
        return [v[j] * sum((a[i, 0] * phi[j][i] for i in range(t)), arb(0)) for j in range(n)]

    hi, lo = settle(lambda: sum((r * arb_q(y) for r, y in zip(row(), Yq)), arb(0)))
    scale = float(
        arbref.rigorous(lambda: sum((abs(r) * abs(arb_q(y)) for r, y in zip(row(), Yq)), arb(0)), target=60).mid()
    )
    return hi, lo, scale


def kreg_task(sid):
    M = S()
    if sid in KREG_1D:
        s = M.scatter(sid)
        X, Xe, w = s.x[:, None], s.xe[:, None], s.w
        fa, fb = 20.0, 4.0
    else:
        s = M.multi(sid)
        X, Xe, w = s.X, s.Xe, s.w
        fa, fb = 5.0, 2.0
    lo, hi = X.min(axis=0), X.max(axis=0)
    bws = {"a": (hi - lo) / fa, "b": (hi - lo) / fb}
    far = lo - 50.0 * (hi - lo)
    Xq = [[Q(v) for v in row] for row in X]
    Yq = [Q(v) for v in s.y]
    Wq = [Q(v) for v in w] if w is not None else [fmpq(1)] * len(Xq)
    points = []
    for kind, i, x0 in [(1, i, row) for i, row in enumerate(Xe)] + [(2, 0, far)]:
        ms = {}
        for tag, bw in bws.items():
            for name, deg in (("nw", 0), ("ll", 1)):
                try:
                    hi_, lo_, scale = kreg_point(Xq, Yq, Wq, [Q(v) for v in x0], [Q(v) for v in bw], deg)
                except NotSettled:
                    ms[f"{name}_{tag}"] = None
                    continue
                ms[f"{name}_{tag}"] = (hi_, lo_, {"scale": scale})
        points.append((kind, i, ms))
    gin = {"n": len(Xq), "bw_a": bws["a"], "bw_b": bws["b"], "far": far}
    return {"set": sid, "digest": s.digest, "ginputs": gin, "points": points, "gmethods": {}}


# ---------------------------------------------------------------- densities (Arb)


def quantile7(xs, p):
    """R's type 7 quantile of sorted exact xs."""
    h = (len(xs) - 1) * p
    i = int(h.p // h.q)
    return xs[i] if i + 1 >= len(xs) else xs[i] + (h - i) * (xs[i + 1] - xs[i])


def _norm_weights(w, n):
    if w is None:
        return [fmpq(1, n)] * n
    wq = [Q(v) for v in w]
    tot = sum(wq, fmpq(0))
    return [v / tot for v in wq]


def _var(xq, wq, unweighted_n=None):
    m = sum((w * x for w, x in zip(wq, xq)), fmpq(0))
    ss = sum((w * (x - m) ** 2 for w, x in zip(wq, xq)), fmpq(0))
    return ss / (1 - sum((w * w for w in wq), fmpq(0)))  # 1 / (n - 1) when the weights are 1 / n


def nrd_double(x, c):
    """R's bw.nrd0 (c = 0.9) and bw.nrd (c = 1.06) in double, as numpy computes them: the tuning of the references."""
    x = np.asarray(x, float)
    sd = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25]))) / 1.34
    lo = min(sd, iqr)
    if c == 0.9 and lo == 0:
        lo = sd or abs(float(x[0])) or 1.0
    return c * lo * x.size**-0.2


def scott_matrix(x, w):
    x = np.asarray(x, float)
    p = x.shape[1]
    wn = None if w is None else w / w.sum()
    neff = x.shape[0] if w is None else 1.0 / float((wn**2).sum())
    cov = np.atleast_2d(np.cov(x, rowvar=False, aweights=wn))
    H = neff ** (-2.0 / (p + 4)) * cov
    return (H + H.T) / 2


def kde_sets():
    return S().density_ids(exclude=("degenerate",))


def kde_task(did):
    s = S().density(did)
    one = s.p == 1
    x = s.x[:, None] if one else s.x
    xe = s.xe[:, None] if one else s.xe
    n, p = x.shape
    H = np.array([[nrd_double(s.x, 0.9) ** 2]]) if one else scott_matrix(x, s.w)
    Hq = fmpq_mat(p, p, [Q(v) for v in H.ravel()])
    Hinv, det = Hq.inv(), Hq.det()
    Xq = [[Q(v) for v in row] for row in x]
    wq = _norm_weights(s.w, n)
    hq = Q(math.sqrt(H[0, 0])) if one else None  # 1-D: the bandwidth is the double sqrt(H), H = that double squared
    if one:
        H = np.array([[float(hq) ** 2]])
        Hq = fmpq_mat(1, 1, [hq * hq])
        Hinv, det = Hq.inv(), Hq.det()

    def quad(x0):
        out = []
        for row in Xq:
            d = [x0[a] - row[a] for a in range(p)]
            out.append(sum((d[a] * Hinv[a, b] * d[b] for a in range(p) for b in range(p)), fmpq(0)))
        return out

    def gauss_sum(qs):
        q0 = min(qs)
        return q0, lambda: sum((arb_q(w) * (-arb_q(q - q0) / 2).exp() for w, q in zip(wq, qs)), arb(0))

    def norm():
        return (2 * arb.pi()) ** (-arb(p) / 2) / arb_q(det).sqrt()

    def pdf_at(x0):
        q0, tot = gauss_sum(quad(x0))
        return lambda: norm() * (-arb_q(q0) / 2).exp() * tot()

    def kernel_pdf(name, x0):
        zs = [(x0[0] - row[0]) / hq for row in Xq]

        def f():
            if name == "cosine":
                a = 1 / (arb(1) / 3 - 2 / arb.pi() ** 2).sqrt()
            else:
                a = arb(KERNELS[name]).sqrt()
            tot = arb(0)
            for w, z in zip(wq, zs):
                az = arb_q(qabs(z))
                if name == "cosine":
                    if az > a:
                        continue
                    if not az < a:
                        raise NotSettled(name)
                    k = (1 + (arb.pi() * az / a).cos()) / (2 * a)
                else:
                    if z * z >= KERNELS[name]:
                        continue
                    r = az / a
                    k = {
                        "epanechnikov": lambda: 3 * (1 - r * r) / (4 * a),
                        "rectangular": lambda: 1 / (2 * a),
                        "triangular": lambda: (1 - r) / a,
                        "biweight": lambda: 15 * (1 - r * r) ** 2 / (16 * a),
                    }[name]()
                tot += arb_q(w) * k
            return tot / arb_q(hq)

        return f

    points = []
    for i, row in enumerate(xe):
        x0 = [Q(v) for v in row]
        ms = {}
        f = pdf_at(x0)
        ms["pdf"] = (*settle(f), {})
        ms["logpdf"] = (*settle(lambda: f().log()), {})
        if one:
            zs = [(x0[0] - r[0]) / hq for r in Xq]
            ms["cdf"] = (
                *settle(
                    lambda: sum((arb_q(w) * (-arb_q(z) / arb(2).sqrt()).erfc() / 2 for w, z in zip(wq, zs)), arb(0))
                ),
                {},
            )
            for name in (*KERNELS, "cosine"):
                ms[f"pdf_{name}"] = (*settle(kernel_pdf(name, x0)), {})
            if s.bounds is not None:
                lo, hi = s.bounds
                if not (lo <= float(row[0]) <= hi):
                    ms["pdf_reflect"] = (0.0, 0.0, {})
                else:
                    parts = [f] + [pdf_at([2 * Q(b) - x0[0]]) for b in (lo, hi) if math.isfinite(b)]
                    ms["pdf_reflect"] = (*settle(lambda: sum((g() for g in parts), arb(0))), {})
        points.append((1, i, ms))
    bw_diag = np.sqrt(np.diag(H))
    if not one:
        dq = [Q(v) * Q(v) for v in bw_diag]  # the diagonal covariance of the doubles bw_diag
        detd = fmpq(1)
        for v in dq:
            detd *= v

        def pdf_diag(x0):
            qs = [sum(((x0[a] - row[a]) ** 2 / dq[a] for a in range(p)), fmpq(0)) for row in Xq]
            q0, tot = gauss_sum(qs)
            return lambda: (2 * arb.pi()) ** (-arb(p) / 2) / arb_q(detd).sqrt() * (-arb_q(q0) / 2).exp() * tot()

        for i, row in enumerate(xe):
            f = pdf_diag([Q(v) for v in row])
            points[i][2]["pdf_diag"] = (*settle(f), {})
            points[i][2]["logpdf_diag"] = (*settle(lambda: f().log()), {})
    gin = {"n": n, "p": p, "H": H, "bw": float(hq) if one else NAN, "bw_diag": bw_diag}
    return {"set": did, "digest": s.digest, "ginputs": gin, "points": points, "gmethods": {}}


def bandwidth_task(did):
    s = S().density(did)
    n = s.n
    xq = [Q(v) for v in s.x]
    wq = _norm_weights(s.w, n)
    var_w = _var(xq, wq)
    neff = 1 / sum((w * w for w in wq), fmpq(0))
    var = _var(xq, [fmpq(1, n)] * n)
    xs = sorted(xq)
    iqr = (quantile7(xs, fmpq(3, 4)) - quantile7(xs, fmpq(1, 4))) / Q(1.34)

    def rule(c):
        def f():
            sd = arb_q(var).sqrt()
            lo = arb_q(iqr) if (iqr * iqr < var) else sd
            if c == 0.9 and (var == 0 or iqr == 0):
                lo = sd if var != 0 else (arb_q(qabs(xq[0])) if xq[0] != 0 else arb(1))
            return arb_q(Q(c)) * lo * (-(arb(n).log()) / 5).exp()

        return f

    ms = {
        "scott": (*settle(lambda: (-(arb_q(neff).log()) / 5).exp() * arb_q(var_w).sqrt()), {}),
        "silverman": (*settle(lambda: (-((arb_q(neff) * 3 / 4).log()) / 5).exp() * arb_q(var_w).sqrt()), {}),
        "nrd0": (*settle(rule(0.9)), {}),
        "nrd": (*settle(rule(1.06)), {}),
    }
    h = nrd_double(s.x, 1.06)
    if s.w is None and h > 0:
        hq = Q(h)
        d2 = [[((xq[i] - xq[j]) / hq) ** 2 for j in range(n)] for i in range(n)]

        def cvls():
            c = 1 / (arb_q(hq) * (2 * arb.pi()).sqrt())
            a = sum(((-arb_q(d2[i][j]) / 4).exp() for i in range(n) for j in range(n)), arb(0)) * c / arb(2).sqrt()
            b = sum(((-arb_q(d2[i][j]) / 2).exp() for i in range(n) for j in range(n) if i != j), arb(0)) * c
            return a / (n * n) - 2 * b / (n * (n - 1))

        def cvml():
            c = 1 / (arb_q(hq) * (2 * arb.pi()).sqrt())
            return sum(
                (
                    (sum(((-arb_q(d2[i][j]) / 2).exp() for j in range(n) if j != i), arb(0)) * c / (n - 1)).log()
                    for i in range(n)
                ),
                arb(0),
            )

        ms["cvls_value"] = (*settle(cvls), {})
        ms["cvml_value"] = (*settle(cvml), {})
    isj_n = 256
    if s.w is None and h > 0:
        ms.update(bandwidth_roots(s.x, xq, var, isj_n))
    gin = {"n": n, "h": h, "isj_n": isj_n}
    return {"set": did, "digest": s.digest, "ginputs": gin, "points": [(0, 0, ms)], "gmethods": {}}


ROOT_BITS = 50


def bisect(sgn, lo, hi):
    """The root of a function whose sign sgn(h) (exact rational h) is -s at lo and s at hi: bisection to
    2^-ROOT_BITS relative, (hi, lo, {"width": relative width}) of the bracket's midpoint."""
    slo = sgn(lo)
    if sgn(hi) == slo or slo == 0:
        raise NotSettled("not bracketed")
    for _ in range(400):
        mid = (lo + hi) / 2
        sm = sgn(mid)
        if sm == 0:
            lo = hi = mid
            break
        if sm == slo:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= mid * fmpq(1, 2**ROOT_BITS):
            break
    mid = (lo + hi) / 2
    a, b = dd_of(mid)
    return a, b, {"width": float(fr((hi - lo) / mid))}


def bracket_change(sgn, grid):
    """(a, b): the first neighbours of grid (ascending) whose signs differ."""
    prev = sgn(grid[0])
    for a, b in zip(grid, grid[1:]):
        cur = sgn(b)
        if cur != prev:
            return a, b
        prev = cur
    raise NotSettled("no sign change")


def bandwidth_roots(x, xq, var, isj_n):
    """sj_ste, sj_dpi, isj, cvls_min, cvml_min of an unweighted 1-D sample (see the module's docstring)."""
    n = len(xq)
    out = {}
    xs = sorted(xq)
    # ---- Sheather-Jones: R's bw_den counts (n <= nb / 2) from the doubles R computes
    nb = 1000
    xd = np.asarray(x, float)
    dd = float((xd.max() - xd.min()) * 1.01) / nb
    assert n <= nb / 2
    bins = [int(v / dd) for v in xd]
    cnt = [0] * nb
    for i in range(1, n):
        for j in range(i):
            cnt[abs(bins[i] - bins[j])] += 1
    dq = Q(dd)
    used = [(i, c) for i, c in enumerate(cnt) if c]
    sq2pi = lambda: (2 * arb.pi()).sqrt()  # noqa: E731

    def phi(h, six):
        hh = arb_q(h) if isinstance(h, fmpq) else h
        tot = arb(0)
        for i, c in used:
            dl = (arb_q(i * dq) / hh) ** 2
            if not dl < 1500:
                continue
            poly = dl**3 - 15 * dl**2 + 45 * dl - 15 if six else dl**2 - 6 * dl + 3
            tot += (-dl / 2).exp() * poly * c
        tot = 2 * tot + (-15 * n if six else 3 * n)
        return tot / (arb(n) * (n - 1) * hh ** (7 if six else 5) * sq2pi())

    iqr = quantile7(xs, fmpq(3, 4)) - quantile7(xs, fmpq(1, 4))

    def scale():
        sdv = arb_q(var).sqrt()
        q = arb_q(iqr / Q(1.349))
        return q if q < sdv else sdv

    def consts():
        sc = scale()
        a = arb_q(Q(1.24)) * sc * arb(n) ** (-arb(1) / 7)
        b = arb_q(Q(1.23)) * sc * arb(n) ** (-arb(1) / 9)
        c1 = 1 / (2 * arb.pi().sqrt() * n)
        td = -phi(b, True)
        return sc, a, c1, td

    def dpi():
        sc, a, c1, td = consts()
        return (c1 / phi((arb_q(Q(2.394)) / (n * td)) ** (arb(1) / 7), False)) ** (arb(1) / 5)

    out["sj_dpi"] = (*settle(dpi), {})

    def fsd_sign(h):
        def f():
            sc, a, c1, td = consts()
            alph2 = arb_q(Q(1.357)) * (phi(a, False) / td) ** (arb(1) / 7)
            return (c1 / phi(alph2 * arb_q(h) ** (arb(5) / 7), False)) ** (arb(1) / 5) - arb_q(h)

        return arbref.sign(f)

    with arbref.precision(128):
        hmax = consts()[0] * arb_q(Q(1.144)) * arb(n) ** (-arb(1) / 5)
        grid = [
            fmpq(*float((hmax * arb(s).exp()).mid()).as_integer_ratio())
            for s in np.linspace(-np.log(64), np.log(64), 33)
        ]
    try:
        a, b = bracket_change(fsd_sign, grid)
        out["sj_ste"] = bisect(fsd_sign, a, b)
    except NotSettled:
        out["sj_ste"] = None
    # ---- ISJ on isj_n points (decision SM129)
    rng_ = float(xd.max() - xd.min())
    lo = float(xd.min() - rng_ / 2)
    hi = float(xd.max() + rng_ / 2)
    R = hi - lo
    delta = R / (isj_n - 1)
    lq, dlq = Q(lo), Q(delta)
    c = [fmpq(0)] * isj_n
    for v in xq:
        pos = (v - lq) / dlq
        k = int(pos.p // pos.q)
        if k >= isj_n - 1:
            k = isj_n - 2
        f = pos - k
        c[k] += (1 - f) / n
        c[k + 1] += f / n
    distinct = len(set(xs))

    def isj_sign(t):
        def f():
            pi = arb.pi()
            a2 = []
            for k in range(1, isj_n):
                ak = sum(
                    (arb_q(cj) * (pi * k * (2 * j + 1) / (2 * isj_n)).cos() for j, cj in enumerate(c) if cj), arb(0)
                )
                a2.append((arb(k) ** 2, ak * ak))
            tt = arb_q(t)

            def fs(s, time):
                return 2 * pi ** (2 * s) * sum((I**s * v * (-I * pi**2 * time).exp() for I, v in a2), arb(0))

            ff = fs(7, tt)
            for s_ in range(6, 1, -1):
                K0 = arb(1)
                for j in range(1, 2 * s_, 2):
                    K0 *= j
                K0 /= (2 * pi).sqrt()
                cst = (1 + arb(0.5) ** (s_ + arb(0.5))) / 3
                tm = (2 * cst * K0 / distinct / ff) ** (arb(2) / (3 + 2 * s_))
                ff = fs(s_, tm)
            return tt - (2 * distinct * pi.sqrt() * ff) ** (-arb(2) / 5)

        return arbref.sign(f)

    tgrid = [fmpq(1, 10) / fmpq(2**k) for k in range(50, -1, -1)]
    try:
        a, b = bracket_change(isj_sign, tgrid)
        th, tl, fl_ = bisect(isj_sign, a, b)
        t = Q(th) + Q(tl)
        out["isj"] = (*settle(lambda: arb_q(t).sqrt() * arb_q(Q(R))), fl_)
    except NotSettled:
        out["isj"] = None
    # ---- cross-validation minimizers
    href = nrd_double(x, 1.06)
    d2 = [[(xq[i] - xq[j]) ** 2 for j in range(n)] for i in range(n)]

    def crit(h, kind):
        hh = arb_q(h)
        if kind == "ls":
            c0 = 1 / (hh * (2 * arb.pi()).sqrt())
            A = sum(((-arb_q(d2[i][j]) / (4 * hh * hh)).exp() for i in range(n) for j in range(n)), arb(0))
            B = sum(((-arb_q(d2[i][j]) / (2 * hh * hh)).exp() for i in range(n) for j in range(n) if i != j), arb(0))
            return A * c0 / arb(2).sqrt() / (n * n) - 2 * B * c0 / (n * (n - 1))
        return (
            -sum(
                (
                    (sum(((-arb_q(d2[i][j]) / (2 * hh * hh)).exp() for j in range(n) if j != i), arb(0))).log()
                    for i in range(n)
                ),
                arb(0),
            )
            + n * hh.log()
        )

    def dcrit_sign(kind):
        def sg(h):
            def f():
                hh = arb_q(h)
                if kind == "ls":  # h^2 d cv / dh
                    a = sum(
                        (
                            (-arb_q(d2[i][j]) / (4 * hh * hh)).exp() * (arb_q(d2[i][j]) / (2 * hh * hh) - 1)
                            for i in range(n)
                            for j in range(n)
                        ),
                        arb(0),
                    ) / (2 * arb.pi().sqrt() * n * n)
                    b = (
                        sum(
                            (
                                (-arb_q(d2[i][j]) / (2 * hh * hh)).exp() * (arb_q(d2[i][j]) / (hh * hh) - 1)
                                for i in range(n)
                                for j in range(n)
                                if i != j
                            ),
                            arb(0),
                        )
                        * 2
                        / ((2 * arb.pi()).sqrt() * n * (n - 1))
                    )
                    return a - b
                tot = arb(0)  # h d(-L)/dh
                for i in range(n):
                    e = [(-arb_q(d2[i][j]) / (2 * hh * hh)).exp() for j in range(n) if j != i]
                    z = [arb_q(d2[i][j]) / (hh * hh) for j in range(n) if j != i]
                    tot += sum((a_ * b_ for a_, b_ in zip(e, z)), arb(0)) / sum(e, arb(0)) - 1
                return -tot

            return arbref.sign(f)

        return sg

    for kind in ("ls", "ml"):
        grid = [Q(href * math.exp(sv)) for sv in np.linspace(-np.log(64), np.log(64), 33)]
        with arbref.precision(160):
            vals = [crit(h, kind).mid() for h in grid]
        k = min(range(33), key=lambda i: (vals[i], -i))
        try:
            if not 0 < k < 32:
                raise NotSettled("the grid's end")
            out[f"cv{kind}_min"] = bisect(dcrit_sign(kind), grid[k - 1], grid[k + 1])
        except NotSettled:
            out[f"cv{kind}_min"] = None
    return out


# ---------------------------------------------------------------- Savitzky-Golay (exact)


def savgol_exact(n, d, deriv, pos):
    """The exact coefficients in window order (use='dot', delta = 1): row `deriv` of (A'A)^-1 A' times deriv!."""
    if deriv > d:
        return [fmpq(0)] * n
    p = fmpq(2 * (n // 2) - 1, 2) if (pos is None and n % 2 == 0) else fmpq(n // 2 if pos is None else pos)
    ts = [fmpq(i) - p for i in range(n)]
    A = fmpq_mat(n, d + 1, [t**j for t in ts for j in range(d + 1)])
    G = (A.transpose() * A).inv()
    e = fmpq_mat(1, d + 1, [G[deriv, j] for j in range(d + 1)]) * A.transpose()
    return [e[0, i] * math.factorial(deriv) for i in range(n)]


def savgol_task(case):
    M = S()
    h = savgol_exact(*case)
    points = [(0, i, {"coeffs": (*dd_of(c), {})}) for i, c in enumerate(h)]
    n, d, deriv, pos = case
    gin = {
        "window_length": n,
        "polyorder": d,
        "deriv": deriv,
        "pos": -1.0 if pos is None else float(pos),
        "norm_inf": fl(max(qabs(c) for c in h)),
    }
    return {
        "set": M.savgol_id(case),
        "digest": M.digest(np.array([n, d, deriv, -1 if pos is None else pos])),
        "ginputs": gin,
        "points": points,
        "gmethods": {},
    }


def savgol_filter_items():
    M = S()
    return [
        (sid, f)
        for sid in M.signal_ids(tag="real")
        if M.signal(sid).x.ndim == 1
        for f in M.SAVGOL_FILTERS
        if f[0] <= M.signal(sid).x.size
    ]


def _read(xq, i, mode, cval):
    """The sample at index i of the signal extended by scipy.ndimage's rule of the mode."""
    m = len(xq)
    if 0 <= i < m:
        return xq[i]
    if mode == "constant":
        return cval
    if mode == "nearest":
        return xq[0] if i < 0 else xq[-1]
    if mode == "wrap":
        return xq[i % m]
    if m == 1:
        return xq[0]
    i = i % (2 * m - 2)  # mirror: d c b | a b c d | c b a
    return xq[i] if i < m else xq[2 * m - 2 - i]


def savgol_filter_task(item):
    M = S()
    sid, (n, d, deriv) = item
    s = M.signal(sid)
    xq = [Q(v) for v in s.x]
    m = len(xq)
    centre = savgol_exact(n, d, deriv, None)
    half = n // 2
    first = half - 1 if n % 2 == 0 else half  # the window of output i starts at i - first (scipy's convolve1d origin)
    edge = {i: savgol_exact(n, d, deriv, i) for i in range(n)}
    cval = Q(M.SAVGOL_CVAL)
    points = []
    for i in range(m):
        ms = {}
        for mode in M.SAVGOL_MODES:
            if mode == "interp" and (i < half or i >= m - half):
                start = 0 if i < half else m - n
                h, win = edge[i - start], xq[start : start + n]
            else:
                h, win = centre, [_read(xq, i - first + j, mode, cval) for j in range(n)]
            ms[mode] = (
                *dd_of(sum((a * b for a, b in zip(h, win)), fmpq(0))),
                {"scale": fl(sum((qabs(a) * qabs(b) for a, b in zip(h, win)), fmpq(0)))},
            )
        points.append((0, i, ms))
    gin = {"signal": sid, "window_length": n, "polyorder": d, "deriv": deriv}
    return {"set": f"{sid}|{n},{d},{deriv}", "digest": s.digest, "ginputs": gin, "points": points, "gmethods": {}}


# ---------------------------------------------------------------- running and writing


def _items(fam):
    M = S()
    return {
        "local": lambda: list(M.scatter_ids()),
        "local_nd": lambda: list(M.multi_ids()),
        "spline": spline_sets,
        "kreg": lambda: list(KREG_1D + KREG_ND),
        "kde": kde_sets,
        "bandwidth": lambda: list(M.density_ids(tag="1d")),
        "savgol": lambda: list(M.SAVGOL),
        "savgol_filter": savgol_filter_items,
    }[fam]()


TASKS = {
    "local": local_task,
    "local_nd": local_nd_task,
    "spline": spline_task,
    "kreg": kreg_task,
    "kde": kde_task,
    "bandwidth": bandwidth_task,
    "savgol": savgol_task,
    "savgol_filter": savgol_filter_task,
}
# the files each task family writes
FILES = {
    "local": ("local",),
    "local_nd": ("local_nd",),
    "spline": ("spline", "spline_fit"),
    "kreg": ("kreg",),
    "kde": ("kde",),
    "bandwidth": ("bandwidth",),
    "savgol": ("savgol",),
    "savgol_filter": ("savgol_filter",),
}
FAMILIES = tuple(f for fs in FILES.values() for f in fs)
EXTRA = {
    "local": "exact rationals (fmpq, fmpq_mat)",
    "spline": "exact rationals (fmpq, fmpq_mat)",
    "savgol": "exact rationals (fmpq, fmpq_mat)",
    "savgol_filter": "exact rationals (fmpq, fmpq_mat)",
}


def _run(task):
    fam, k, item = task
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SECONDS)
    t0 = time.monotonic()
    try:
        res, err = TASKS[fam](item), None
    except (_Timeout, NotSettled) as e:
        res, err = None, type(e).__name__
    finally:
        signal.alarm(0)
    return fam, k, res, time.monotonic() - t0, err


def assemble(name, results):
    """Write one file from the tasks' results (in item order): one group per task."""
    from tests import reffile

    inputs = {"set": [], "kind": [], "index": [], "digest": []}
    group, gin, methods = [], {}, {}

    def put(m, over, at, entry):
        e = methods.setdefault(m, {"over": over, "index": [], "hi": [], "lo": [], "dropped": []})
        if entry is None:
            e["dropped"].append(at)
            return
        hi, lo, fields = entry
        e["index"].append(at)
        e["hi"].append(hi)
        e["lo"].append(lo)
        for k, v in fields.items():
            e.setdefault(k, []).append(v)

    for g, r in enumerate(results):
        for k, v in r["ginputs"].items():
            gin.setdefault(k, []).append(np.asarray(v))
        for kind, idx, ms in r["points"]:
            at = len(group)
            group.append(g)
            for key, v in (("set", r["set"]), ("kind", kind), ("index", idx), ("digest", r["digest"])):
                inputs[key].append(v)
            for m, entry in ms.items():
                put(m, "point", at, entry)
        for m, entry in r["gmethods"].items():
            put(m, "group", g, entry)
    for e in methods.values():
        for k in e:
            if k != "over":
                e[k] = np.asarray(e[k], dtype=np.int64 if k in ("index", "dropped") else np.float64)
    inputs = {
        "set": np.array(inputs["set"], dtype=str),
        "kind": np.array(inputs["kind"], np.int64),
        "index": np.array(inputs["index"], np.int64),
        "digest": np.array(inputs["digest"], dtype=str),
    }
    gin = {k: (np.stack(v) if len({a.shape for a in v}) == 1 else v) for k, v in gin.items()}
    gin = {"set": np.array([r["set"] for r in results], dtype=str), **gin}
    reffile.write(
        OUT / f"{name}.npz",
        **arbref.provenance(extra=EXTRA.get(name)),
        tool=TOOL,
        inputs=inputs,
        methods=methods,
        group=np.array(group, np.int64),
        ngroups=len(results),
        group_inputs=gin,
        n_requested=len(group),
    )
    ndrop = sum(len(e["dropped"]) for e in methods.values())
    print(
        f"{name}: {len(group)} points, {len(methods)} methods, {ndrop} dropped -> "
        f"{(OUT / f'{name}.npz').relative_to(ROOT)}",
        flush=True,
    )


def build(names, jobs):
    import resource

    start = time.monotonic()
    plans = {f: _items(f) for f in names}
    tasks = [(f, k, it) for f in names for k, it in enumerate(plans[f])]
    results = {f: {} for f in names}
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        for fam, k, res, secs, err in pool.imap_unordered(_run, tasks, chunksize=1):
            if res is None:
                print(f"  dropped {fam} {plans[fam][k]}: {err}", flush=True)
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]}: {secs:.0f} s", flush=True)
    for f in names:
        done = [results[f][k] for k in sorted(results[f])]
        if len(FILES[f]) == 1:
            assemble(f, done)
        else:
            for j, name in enumerate(FILES[f]):
                assemble(name, [r[j] for r in done])
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"{len(tasks)} tasks, {(time.monotonic() - start):.0f} s, largest worker RSS {rss:.2f} GB", flush=True)


def main(argv=None):
    from bench import cli

    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FILES))
    ap.add_argument("--list", action="store_true", help="list the families and their data")
    cli.add_jobs(ap, default=3)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FILES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FILES)})")
    names = [f for f in FILES if not a.names or f in a.names]
    if a.list:
        for f in names:
            items = _items(f)
            print(f"{'/'.join(FILES[f])} ({len(items)}): " + ", ".join(map(str, items)))
        return 0
    with arbref.logged("smooth_ref"):
        build(names, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
