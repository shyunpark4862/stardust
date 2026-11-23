"""Reference values for sd.interpolate (docs/design/interpolate.md §4, §7.2) on the test data of
tests/interpolate/samples.py: exact rationals (Fraction, python-flint's fmpq_mat) and Arb.

    .venv/bin/python tools/interpolate_ref.py                     # every family, on 3 processes
    .venv/bin/python tools/interpolate_ref.py slopes curves       # only these families
    .venv/bin/python tools/interpolate_ref.py --list              # the families and their data

writes tests/interpolate/ref/<family>.npz (layout: tests/reffile.py), read by tests/interpolate/refdata.py.
Every point names its data (input "set", the id in samples.py) and "digest" (the data's digest), so a
changed generator shows. The data are doubles taken as exact rationals; a reference is the true value
for those doubles, rounded once (a double-double hi + lo).

    linear      one point per (interval, x): the exact a + (x - x0) / (x1 - x0) (b - a) on the finite
                intervals of samples.lerps() and on every curve's evaluation points (the interval by the
                rule x[i] <= x < x[i+1], the end intervals extended outside)
    slopes      one point per node of every real curve: the exact node slopes of CubicSpline with
                bc_type natural, clamped, not-a-knot and periodic (the periodic curves), of
                PchipInterpolator and of Akima1DInterpolator (akima, makima) -- scipy 1.18.1's
                definitions, its special cases for 2 and 3 nodes and Akima's cutoff 1e-9 max(f12)
                evaluated exactly
    bspline     one point per coefficient: make_interp_spline's exact coefficients for k = 0 .. 5
                (default knots), k = 3 natural and clamped, k = 3 and 5 periodic (the periodic curves);
                per curve (group) the exact infinity norm of the inverse of each system's matrix
                (ainv_<method>, n <= 60) -- the scale of the contract of docs/design/interpolate.md §4
    curves      one point per evaluation point of every real curve: the exact value of each of those
                interpolants (built from the exact data and exact slopes or coefficients), the end
                pieces extended outside: the end-to-end column of the accuracy table
    polynomial  one point per evaluation point of samples.polys(): the interpolating polynomial's value
                and first two derivatives (Krogh's Hermite data included), the Lebesgue function
                sum |l_j(x)| and sum |l_j(x) y_j| (the bound of Higham's barycentric analysis), and
                the Floater-Hormann d = 3 rational (distinct nodes)
    grid        one point per xi of the grids: RegularGridInterpolator's linear, cubic, quintic and
                pchip values (exact tensor products; pchip folds the axes from the last, as scipy)
    rbf         one point per xi of the scattered sets: RBFInterpolator's value for each kernel (Arb:
                the kernels have square roots, logarithms and exponentials), epsilon 1 for the
                scale-invariant kernels and 3 for the others, the default degree; also smoothing 0.1
                and neighbors = 8 (the 8 nearest by exact distance, ties by index); sets whose system
                is singular (duplicate points, collinear points under degree 1) are left out
    roots       one point per distinct real root of every piece of samples.pieces(), in [x[i], x[i+1]]:
                the root (Arb isolation of the square-free factors, refined to 2^-110) with its piece
                and multiplicity
    scattered   one point per xi of samples.trisets() (the valid point sets of tests/spatial/geompoints.py
                with a smooth field; 64 points in the box and the first 8 data points) inside the convex
                hull: on sd.spatial.Delaunay's triangulation (the simplex find_simplex gives, checked exactly),
                the exact value of the linear interpolant (method linear) and sum_k lambda_k |f_k| (linear_scale),
                and for d = 2 the exact Clough-Tocher cubic of the gradients sd estimates (method clough_tocher,
                the gradients kept per set as the group input grad; sets whose gradients are not finite are left
                out) and its scale K (ct_scale; tests/interpolate/refdata.ct_exact)

Settling: an exact rational is exact; an Arb value is settled to 2^-110 relative. Workers are forked,
replaced every 8 tasks (maxtasksperchild), each task under a time limit (SIGALRM). No mpmath (the
project's rule). The console output also goes to build/logs/interpolate_ref/.
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

OUT = ROOT / "tests" / "interpolate" / "ref"
TOOL = "tools/interpolate_ref.py"
SECONDS = 600  # a task's time limit
AINV_MAX_N = 60
AKIMA_BREAK = 1e-9  # scipy's break_mult
RBF_KERNELS = ("linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric",
               "inverse_quadratic", "gaussian")  # fmt: skip
SCALE_INVARIANT = ("linear", "thin_plate_spline", "cubic", "quintic")
MIN_DEGREE = {"multiquadric": 0, "linear": 0, "thin_plate_spline": 1, "cubic": 1, "quintic": 2}
RBF_EPSILON = 3.0  # for the kernels that need one
NEIGHBORS = 8


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def S():
    from tests.interpolate import samples

    return samples


# ---------------------------------------------------------------- exact helpers


def fr(v) -> Fraction:
    return Fraction(float(v))


def dd_of(q: Fraction):
    """(hi, lo) of an exact rational: hi rounded to nearest (ties to even), lo the rest rounded;
    (inf, 0) past the double range."""
    try:
        hi = float(q)
    except OverflowError:
        return (math.inf if q > 0 else -math.inf), 0.0
    if math.isinf(hi):
        return hi, 0.0
    return hi, float(q - Fraction(hi))


def solve_dense(rows, rhs):
    """The exact solution of a square rational system (python-flint's fmpq_mat)."""
    from flint import fmpq, fmpq_mat

    A = fmpq_mat([[fmpq(v.numerator, v.denominator) for v in r] for r in rows])
    b = fmpq_mat([[fmpq(v.numerator, v.denominator)] for v in rhs])
    x = A.solve(b)
    return [Fraction(int(x[i, 0].p), int(x[i, 0].q)) for i in range(len(rhs))]


def inverse_norm(rows):
    """The exact infinity norm of the inverse of a square rational matrix."""
    from flint import fmpq, fmpq_mat

    A = fmpq_mat([[fmpq(v.numerator, v.denominator) for v in r] for r in rows])
    inv = A.inv()
    n = len(rows)
    best = Fraction(0)
    for i in range(n):
        s = sum((abs(Fraction(int(inv[i, j].p), int(inv[i, j].q))) for j in range(n)), Fraction(0))
        best = max(best, s)
    return best


def solve_tridiagonal(sub, diag, sup, rhs):
    """Exact elimination without pivoting (no rounding: the pivots are nonzero for these systems)."""
    n = len(diag)
    d, r = list(diag), list(rhs)
    for i in range(1, n):
        if d[i - 1] == 0:
            raise ZeroDivisionError("zero pivot")
        m = sub[i] / d[i - 1]
        d[i] -= m * sup[i - 1]
        r[i] -= m * r[i - 1]
    x = [Fraction(0)] * n
    x[-1] = r[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (r[i] - sup[i] * x[i + 1]) / d[i]
    return x


def interval(xs, v):
    """The interval i of v: xs[i] <= v < xs[i+1], the last one closed; clamped to the end intervals."""
    lo, hi = 0, len(xs) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if xs[mid] <= v:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sign(v):
    return (v > 0) - (v < 0)


# ---------------------------------------------------------------- the cubic family


def spline_slopes(X, Y, bc):
    """CubicSpline's node slopes for bc in 'natural', 'clamped', 'not-a-knot', 'periodic' (both ends)."""
    n = len(X)
    h = [X[i + 1] - X[i] for i in range(n - 1)]
    D = [(Y[i + 1] - Y[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        if bc == "clamped":
            return [Fraction(0), Fraction(0)]
        return [D[0], D[0]]  # not-a-knot and periodic become (1, slope); natural gives the line too
    if n == 3 and bc == "not-a-knot":  # the parabola through the three points
        c = (D[1] - D[0]) / (X[2] - X[0])
        return [D[0] - c * h[0], D[0] + c * h[0], D[0] + c * (h[0] + 2 * h[1])]
    if bc == "periodic":
        m = n - 1  # unknowns s_0 .. s_{n-2}, s_{n-1} = s_0
        rows = [[Fraction(0)] * m for _ in range(m)]
        rhs = []
        for i in range(m):
            hl, hr = h[i - 1] if i > 0 else h[n - 2], h[i]
            dl, dr = D[i - 1] if i > 0 else D[n - 2], D[i]
            rows[i][(i - 1) % m] += hr
            rows[i][i] += 2 * (hl + hr)
            rows[i][(i + 1) % m] += hl
            rhs.append(3 * (hr * dl + hl * dr))
        s = solve_dense(rows, rhs)
        return s + [s[0]]
    sub, diag, sup, rhs = [Fraction(0)] * n, [Fraction(0)] * n, [Fraction(0)] * n, [Fraction(0)] * n
    for i in range(1, n - 1):
        sub[i], diag[i], sup[i] = h[i], 2 * (h[i - 1] + h[i]), h[i - 1]
        rhs[i] = 3 * (h[i] * D[i - 1] + h[i - 1] * D[i])
    if bc == "not-a-knot":
        d = X[2] - X[0]
        diag[0], sup[0], rhs[0] = h[1], d, ((h[0] + 2 * d) * h[1] * D[0] + h[0] ** 2 * D[1]) / d
        d = X[-1] - X[-3]
        diag[-1], sub[-1], rhs[-1] = h[-2], d, (h[-1] ** 2 * D[-2] + (2 * d + h[-1]) * h[-2] * D[-1]) / d
    elif bc == "natural":
        diag[0], sup[0], rhs[0] = Fraction(2), Fraction(1), 3 * D[0]
        diag[-1], sub[-1], rhs[-1] = Fraction(2), Fraction(1), 3 * D[-1]
    else:  # clamped
        diag[0], sup[0], rhs[0] = Fraction(1), Fraction(0), Fraction(0)
        diag[-1], sub[-1], rhs[-1] = Fraction(1), Fraction(0), Fraction(0)
    return solve_tridiagonal(sub, diag, sup, rhs)


def _pchip_edge(h0, h1, m0, m1):
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if sign(d) != sign(m0):
        return Fraction(0)
    if sign(m0) != sign(m1) and abs(d) > 3 * abs(m0):
        return 3 * m0
    return d


def pchip_slopes(X, Y):
    """PchipInterpolator's slopes (scipy's _find_derivatives and _edge_case, exactly)."""
    n = len(X)
    h = [X[i + 1] - X[i] for i in range(n - 1)]
    m = [(Y[i + 1] - Y[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [m[0], m[0]]
    s = [Fraction(0)] * n
    for k in range(1, n - 1):
        if sign(m[k]) != sign(m[k - 1]) or m[k] == 0 or m[k - 1] == 0:
            continue
        w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
        s[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])
    s[0] = _pchip_edge(h[0], h[1], m[0], m[1])
    s[-1] = _pchip_edge(h[-1], h[-2], m[-1], m[-2])
    return s


def akima_slopes(X, Y, method):
    """Akima1DInterpolator's slopes (scipy's formulas, the cutoff 1e-9 max(f12) included)."""
    n = len(X)
    d = [(Y[i + 1] - Y[i]) / (X[i + 1] - X[i]) for i in range(n - 1)]
    if n == 2:
        return [d[0], d[0]]
    m = [None, None] + d + [None, None]
    m[1] = 2 * m[2] - m[3]
    m[0] = 2 * m[1] - m[2]
    m[-2] = 2 * m[-3] - m[-4]
    m[-1] = 2 * m[-2] - m[-3]
    dm = [abs(m[j + 1] - m[j]) for j in range(len(m) - 1)]
    if method == "makima":
        pm = [abs(m[j + 1] + m[j]) for j in range(len(m) - 1)]
        f1 = [dm[i + 2] + pm[i + 2] / 2 for i in range(n)]
        f2 = [dm[i] + pm[i] / 2 for i in range(n)]
    else:
        f1 = [dm[i + 2] for i in range(n)]
        f2 = [dm[i] for i in range(n)]
    f12 = [a + b for a, b in zip(f1, f2)]
    cut = fr(AKIMA_BREAK) * max(f12)
    out = []
    for i in range(n):
        if f12[i] > cut:
            out.append(m[i + 1] + (f2[i] / f12[i]) * (m[i + 2] - m[i + 1]))
        else:
            out.append((m[i + 3] + m[i]) / 2)
    return out


def hermite_value(X, Y, s, v):
    """The cubic Hermite interpolant with node slopes s at v (the end pieces extended)."""
    i = interval(X, v)
    h = X[i + 1] - X[i]
    D = (Y[i + 1] - Y[i]) / h
    c3 = (s[i] + s[i + 1] - 2 * D) / h**2
    c2 = (3 * D - 2 * s[i] - s[i + 1]) / h
    t = v - X[i]
    return ((c3 * t + c2) * t + s[i]) * t + Y[i]


# ---------------------------------------------------------------- B-splines


def not_a_knot_knots(x, k):
    """scipy's _not_a_knot (the even-k midpoints computed in doubles, as scipy does), as doubles."""
    if k % 2 == 1:
        k2, t = (k + 1) // 2, x.copy()
    else:
        k2, t = k // 2, (x[1:] + x[:-1]) / 2
    t = t[k2:-k2] if k2 else t
    return np.r_[(x[0],) * (k + 1), t, (x[-1],) * (k + 1)]


def periodic_knots(x, k):
    """scipy's _periodic_knots."""
    xc = np.copy(x)
    n = len(xc)
    if k % 2 == 0:
        dx = np.diff(xc)
        xc[1:-1] -= dx[:-1] / 2
    dx = np.diff(xc)
    t = np.zeros(n + 2 * k)
    t[k:-k] = xc
    for i in range(k):
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    return t


def knots_for(x, k, method):
    if k == 0:
        return np.r_[x, x[-1]]
    if k == 1:
        return np.r_[x[0], x, x[-1]]
    if method.endswith("periodic"):
        return periodic_knots(x, k)
    if method.endswith(("natural", "clamped")):
        return np.r_[(x[0],) * k, x, (x[-1],) * k]
    return not_a_knot_knots(x, k)


def knot_interval(T, k, nt, v):
    """l with T[l] <= v < T[l+1] in [k, nt - 1], clamped; v >= T[nt] is nt - 1 (scipy's rule: for k = 0
    that interval is empty and the value at the last node is c[-1] = y[-1])."""
    lo, hi = k, nt - 1
    if v >= T[nt]:
        return nt - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if T[mid] <= v:
            lo = mid
        else:
            hi = mid - 1
    return lo


def basis(T, k, l, v, nu=0):
    """{j: B_{j,k}^{(nu)}(v)} for j = l - k .. l, the degree-0 functions being [j == l] (so v outside
    [T[l], T[l+1]] gives the polynomial piece of interval l)."""
    memo = {}

    def B(j, kk, d):
        key = (j, kk, d)
        if key in memo:
            return memo[key]
        if kk == 0:
            r = Fraction(int(j == l)) if d == 0 else Fraction(0)
        elif d > 0:
            r = Fraction(0)
            den = T[j + kk] - T[j]
            if den:
                r += kk * B(j, kk - 1, d - 1) / den
            den = T[j + kk + 1] - T[j + 1]
            if den:
                r -= kk * B(j + 1, kk - 1, d - 1) / den
        else:
            r = Fraction(0)
            den = T[j + kk] - T[j]
            if den:
                r += (v - T[j]) / den * B(j, kk - 1, 0)
            den = T[j + kk + 1] - T[j + 1]
            if den:
                r += (T[j + kk + 1] - v) / den * B(j + 1, kk - 1, 0)
        memo[key] = r
        return r

    return {j: B(j, k, nu) for j in range(l - k, l + 1)}


def bspline_system(x, y, k, method):
    """(knots as doubles, rows, rhs) of make_interp_spline's system (k >= 2)."""
    t = knots_for(x, k, method)
    T = [fr(v) for v in t]
    X = [fr(v) for v in x]
    Y = [fr(v) for v in y]
    nt = len(t) - k - 1
    rows, rhs = [], []
    if method.endswith("periodic"):
        m = len(x) - 1
        for i in range(m):
            row = [Fraction(0)] * m
            for j, b in basis(T, k, knot_interval(T, k, nt, X[i]), X[i]).items():
                row[j % m] += b
            rows.append(row)
            rhs.append(Y[i])
        return t, rows, rhs
    derivs = {"natural": 2, "clamped": 1}.get(method.split("_")[-1])
    if derivs:
        rows.append([Fraction(0)] * nt)
        for j, b in basis(T, k, k, X[0], derivs).items():
            rows[-1][j] = b
        rhs.append(Fraction(0))
    for i in range(len(x)):
        row = [Fraction(0)] * nt
        for j, b in basis(T, k, knot_interval(T, k, nt, X[i]), X[i]).items():
            row[j] = b
        rows.append(row)
        rhs.append(Y[i])
    if derivs:
        rows.append([Fraction(0)] * nt)
        for j, b in basis(T, k, knot_interval(T, k, nt, X[-1]), X[-1], derivs).items():
            rows[-1][j] = b
        rhs.append(Fraction(0))
    return t, rows, rhs


def bspline_coefficients(x, y, k, method):
    """(knots, exact coefficients, the system's rows or None) of make_interp_spline."""
    if k in (0, 1):
        return knots_for(x, k, method), [fr(v) for v in y], None
    t, rows, rhs = bspline_system(x, y, k, method)
    c = solve_dense(rows, rhs)
    if method.endswith("periodic"):
        m = len(x) - 1
        c = [c[j % m] for j in range(len(t) - k - 1)]
    return t, c, rows


def bspline_value(t, k, c, v):
    T = [fr(a) for a in t]
    nt = len(t) - k - 1
    l = knot_interval(T, k, nt, v)
    return sum((c[j] * b for j, b in basis(T, k, l, v).items()), Fraction(0))


BSPLINE_METHODS = ("k0", "k1", "k2", "k3", "k4", "k5", "k3_natural", "k3_clamped", "k3_periodic", "k5_periodic")
SLOPE_METHODS = ("natural", "clamped", "not_a_knot", "periodic", "pchip", "akima", "makima")


def _k_of(method):
    return int(method[1])


def curve_task(cid):
    """slopes, B-spline coefficients and norms, and end-to-end values of one real curve."""
    M = S()
    c = M.curve(cid)
    x, y = c.x, c.y
    n = len(x)
    X, Y = [fr(v) for v in x], [fr(v) for v in y]
    periodic = "periodic" in c.tags
    slopes = {}
    for bc in ("natural", "clamped", "not-a-knot") + (("periodic",) if periodic else ()):
        slopes[bc.replace("-", "_")] = spline_slopes(X, Y, bc)
    slopes["pchip"] = pchip_slopes(X, Y)
    slopes["akima"] = akima_slopes(X, Y, "akima")
    slopes["makima"] = akima_slopes(X, Y, "makima")
    coefs, norms, knots = {}, {}, {}
    for meth in BSPLINE_METHODS:
        k = _k_of(meth)
        if meth.endswith("periodic") and not periodic:
            continue
        if n < (2 if meth.endswith(("natural", "clamped")) else k + 1):
            continue
        t, cc, rows = bspline_coefficients(x, y, k, meth)
        coefs[meth], knots[meth] = cc, t
        if rows is not None and len(rows) <= AINV_MAX_N:
            norms[meth] = inverse_norm(rows)
    values = {}
    for meth, s in slopes.items():
        values[meth] = [hermite_value(X, Y, s, fr(v)) for v in c.xe]
    for meth, cc in coefs.items():
        values[meth] = [bspline_value(knots[meth], _k_of(meth), cc, fr(v)) for v in c.xe]
    return {
        "slopes": {m: [dd_of(v) for v in s] for m, s in slopes.items()},
        "coefs": {m: [dd_of(v) for v in s] for m, s in coefs.items()},
        "knots": {m: np.asarray(t) for m, t in knots.items()},
        "norms": {m: dd_of(v) for m, v in norms.items()},
        "values": {m: [dd_of(v) for v in s] for m, s in values.items()},
    }


# ---------------------------------------------------------------- linear


def linear_items():
    M = S()
    items = [("lerp", L.id) for L in M.lerps() if M.lerp_finite(L)]
    items += [("curve", cid) for cid in M.real_curve_ids()]
    return items


def linear_task(item):
    M = S()
    kind, ident = item
    pts = []
    if kind == "lerp":
        L = next(L for L in M.lerps() if L.id == ident)
        x0, x1, a, b = fr(L.x0), fr(L.x1), fr(L.a), fr(L.b)
        for i, v in enumerate(L.x):
            y = a + (fr(v) - x0) / (x1 - x0) * (b - a)
            pts.append((f"lerp/{L.id}", i, L.digest, (L.x0, L.x1, L.a, L.b, float(v)), dd_of(y)))
        return pts
    c = M.curve(ident)
    X = [fr(v) for v in c.x]
    for i, v in enumerate(c.xe):
        j = interval(X, fr(v))
        x0, x1, a, b = X[j], X[j + 1], fr(c.y[j]), fr(c.y[j + 1])
        y = a + (fr(v) - x0) / (x1 - x0) * (b - a)
        pts.append((f"curve/{c.id}", i, c.digest, (c.x[j], c.x[j + 1], c.y[j], c.y[j + 1], float(v)), dd_of(y)))
    return pts


# ---------------------------------------------------------------- polynomial interpolation


def newton_poly(X, Y):
    """The interpolating polynomial (fmpq_poly) of nodes X with values Y; a repeated node carries the
    successive derivatives in Y (scipy's KroghInterpolator)."""
    from flint import fmpq, fmpq_poly

    n = len(X)
    first = {}
    for i in range(n):
        first.setdefault(X[i], i)
    table = [Y[first[X[i]]] for i in range(n)]
    coef = [table[0]]
    col = table
    for lev in range(1, n):
        new = []
        for i in range(n - lev):
            if X[i + lev] == X[i]:
                fact = math.factorial(lev)
                new.append(Y[first[X[i]] + lev] / fact)
            else:
                new.append((col[i + 1] - col[i]) / (X[i + lev] - X[i]))
        col = new
        coef.append(col[0])
    p = fmpq_poly([0])
    basis_ = fmpq_poly([1])
    for k in range(n):
        p += basis_ * fmpq(coef[k].numerator, coef[k].denominator)
        basis_ *= fmpq_poly([fmpq(-X[k].numerator, X[k].denominator), 1])
    return p


def poly_eval(p, v):
    from flint import fmpq

    r = p(fmpq(v.numerator, v.denominator))
    return Fraction(int(r.p), int(r.q))


def floater_hormann(X, Y, d, v):
    n = len(X) - 1
    w = []
    for k in range(n + 1):
        s = Fraction(0)
        for i in range(max(0, k - d), min(k, n - d) + 1):
            prod = Fraction(1)
            for j in range(i, i + d + 1):
                if j != k:
                    prod /= abs(X[k] - X[j])
            s += prod
        w.append((-1) ** ((k - d) % 2) * s)
    for k in range(n + 1):
        if v == X[k]:
            return Y[k]
    num = sum((w[k] * Y[k] / (v - X[k]) for k in range(n + 1)), Fraction(0))
    den = sum((w[k] / (v - X[k]) for k in range(n + 1)), Fraction(0))
    return num / den


def poly_task(pid):
    M = S()
    P = M.poly(pid)
    X, Y = [fr(v) for v in P.xi], [fr(v) for v in P.yi]
    p = newton_poly(X, Y)
    d1, d2 = p.derivative(), p.derivative().derivative()
    out = {"value": [], "d1": [], "d2": []}
    if P.distinct:
        out |= {"lebesgue": [], "abs_terms": [], "fh3": []}
        w = []
        for j in range(len(X)):
            prod = Fraction(1)
            for k in range(len(X)):
                if k != j:
                    prod *= X[j] - X[k]
            w.append(1 / prod)
    for v in P.x:
        V = fr(v)
        out["value"].append(dd_of(poly_eval(p, V)))
        out["d1"].append(dd_of(poly_eval(d1, V)))
        out["d2"].append(dd_of(poly_eval(d2, V)))
        if P.distinct:
            if V in X:
                ell = [Fraction(int(V == a)) for a in X]
            else:
                terms = [w[j] / (V - X[j]) for j in range(len(X))]
                tot = sum(terms, Fraction(0))
                ell = [t / tot for t in terms]
            out["lebesgue"].append(dd_of(sum((abs(e) for e in ell), Fraction(0))))
            out["abs_terms"].append(dd_of(sum((abs(e * yv) for e, yv in zip(ell, Y)), Fraction(0))))
            out["fh3"].append(dd_of(floater_hormann(X, Y, 3, V)))
    return out


# ---------------------------------------------------------------- grids


def _linear_1d(X, Yv, v):
    if len(X) == 1:
        return Yv[0]
    j = interval(X, v)
    return Yv[j] + (v - X[j]) / (X[j + 1] - X[j]) * (Yv[j + 1] - Yv[j])


def _pchip_1d(X, Yv, v):
    if len(X) == 2:
        return _linear_1d(X, Yv, v)
    return hermite_value(X, Yv, pchip_slopes(X, Yv), v)


def _spline_1d(k):
    def f(X, Yv, v):
        x = np.array([float(a) for a in X])
        t = not_a_knot_knots(x, k)
        T = [fr(a) for a in t]
        nt = len(t) - k - 1
        rows = []
        for xi in X:
            row = [Fraction(0)] * nt
            for j, b in basis(T, k, knot_interval(T, k, nt, xi), xi).items():
                row[j] = b
            rows.append(row)
        c = solve_dense(rows, Yv)
        return sum((c[j] * b for j, b in basis(T, k, knot_interval(T, k, nt, v), v).items()), Fraction(0))

    return f


GRID_METHODS = {"linear": (_linear_1d, 1), "cubic": (_spline_1d(3), 4), "quintic": (_spline_1d(5), 6),
                "pchip": (_pchip_1d, 4)}  # fmt: skip


def tensor_value(points, values, pt, f):
    """Fold the axes from the last to the first with the 1-D interpolant f (scipy's order)."""
    arr = values
    for ax in reversed(range(len(points))):
        X = points[ax]
        arr = np.apply_along_axis(lambda col, X=X, ax=ax: f(X, list(col), pt[ax]), ax, arr)
    return arr.item() if isinstance(arr, np.ndarray) else arr


def grid_items():
    M = S()
    return [g for g in M.GRID_IDS if not ({"nan", "batch"} & M.grid(g).tags)]


def grid_task(gid):
    M = S()
    G = M.grid(gid)
    pts, vals = [], G.values
    for ax, p in enumerate(G.points):  # scipy flips a decreasing axis
        if p.size > 1 and p[0] > p[-1]:
            p = p[::-1]
            vals = np.flip(vals, axis=ax)
        pts.append([fr(v) for v in p])
    V = np.vectorize(fr, otypes=[object])(vals)
    out = {}
    for meth, (f, need) in GRID_METHODS.items():
        if any(len(p) < need for p in pts):
            continue
        out[meth] = [dd_of(tensor_value(pts, V, [fr(v) for v in q], f)) for q in G.xi]
    return out


# ---------------------------------------------------------------- RBF (Arb)


def monomial_powers(d, degree):
    out = []
    for total in range(degree + 1):

        def rec(prefix, left, dim):
            if dim == d - 1:
                out.append(tuple(prefix + [left]))
                return
            for e in range(left, -1, -1):
                rec(prefix + [e], left - e, dim + 1)

        if d == 1:
            out.append((total,))
        else:
            rec([], total, 0)
    return out


def kernel_arb(name, r):
    from flint import arb

    if name == "linear":
        return -r
    if name == "thin_plate_spline":
        return arb(0) if r == 0 else r * r * r.log()
    if name == "cubic":
        return r * r * r
    if name == "quintic":
        return -(r**5)
    if name == "multiquadric":
        return -((1 + r * r).sqrt())
    if name == "inverse_multiquadric":
        return 1 / (1 + r * r).sqrt()
    if name == "inverse_quadratic":
        return 1 / (1 + r * r)
    return (-(r * r)).exp()


def _arb_of(q: Fraction):
    from flint import arb

    return arb(q.numerator) / arb(q.denominator)


def rbf_values(Yd, d, Xq, kernel, epsilon, degree, smoothing, prec):
    """The RBF interpolant of points Yd (rationals) at points Xq, at Arb precision prec."""
    from flint import arb, arb_mat

    with arbref.precision(prec):
        n, dim = len(Yd), len(Yd[0])
        powers = monomial_powers(dim, degree) if degree >= 0 else []
        q = len(powers)
        eps = _arb_of(fr(epsilon))

        def dist(a, b):
            s = sum(((u - v) ** 2 for u, v in zip(a, b)), Fraction(0))
            return arb(0) if s == 0 else _arb_of(s).sqrt()

        def mono(pt, pw):
            out = arb(1)
            for c, e in zip(pt, pw):
                out *= _arb_of(c) ** e
            return out

        A = [[arb(0)] * (n + q) for _ in range(n + q)]
        for i in range(n):
            for j in range(n):
                A[i][j] = kernel_arb(kernel, eps * dist(Yd[i], Yd[j]))
            A[i][i] += _arb_of(fr(smoothing))
            for k, pw in enumerate(powers):
                A[i][n + k] = A[n + k][i] = mono(Yd[i], pw)
        rhs = arb_mat([[_arb_of(v)] for v in d] + [[arb(0)] for _ in range(q)])
        coef = arb_mat(A).solve(rhs)
        out = []
        for x in Xq:
            v = arb(0)
            for j in range(n):
                v += coef[j, 0] * kernel_arb(kernel, eps * dist(x, Yd[j]))
            for k, pw in enumerate(powers):
                v += coef[n + k, 0] * mono(x, pw)
            out.append(v)
        return out


def settle_all(fn):
    p = arbref.START_PREC
    while p <= arbref.MAX_PREC:
        try:
            vals = fn(p)
        except ZeroDivisionError, ValueError:  # a singular matrix at this precision
            vals = None
        if vals is not None and all(arbref.settled(v) or (v.is_exact() and v == 0) for v in vals):
            return [arbref.to_dd(v) if not (v.is_exact() and v == 0) else (0.0, 0.0) for v in vals]
        p *= 2
    raise NotSettled(fn)


RBF_VARIANTS = tuple((k, {}) for k in RBF_KERNELS) + (("thin_plate_spline_s0.1", {"smoothing": 0.1}),
                                                       ("thin_plate_spline_k8", {"neighbors": NEIGHBORS}))  # fmt: skip


def rbf_items():
    M = S()
    out = []
    for sid in M.SCATTER_IDS:
        tags = M.scatter(sid).tags
        if "duplicate" in tags:
            continue
        for name, opt in RBF_VARIANTS:
            kernel = name.split("_s0.1")[0].split("_k8")[0]
            degree = max(MIN_DEGREE.get(kernel, -1), 0)
            if "collinear" in tags and degree >= 1 and "neighbors" not in opt:
                continue
            if "neighbors" in opt and sid not in ("scatter/uniform2d", "scatter/lattice2d"):
                continue
            out.append((sid, name))
    return out


def rbf_task(item):
    M = S()
    sid, name = item
    sc = M.scatter(sid)
    kernel = name.split("_s0.1")[0].split("_k8")[0]
    opt = dict(RBF_VARIANTS)[name]
    epsilon = 1.0 if kernel in SCALE_INVARIANT else RBF_EPSILON
    degree = max(MIN_DEGREE.get(kernel, -1), 0)
    smoothing = opt.get("smoothing", 0.0)
    Yd = [[fr(v) for v in row] for row in sc.y]
    d = [fr(v) for v in sc.d]
    Xq = [[fr(v) for v in row] for row in sc.xi]
    if "neighbors" not in opt:
        return settle_all(lambda p: rbf_values(Yd, d, Xq, kernel, epsilon, degree, smoothing, p))
    out = []
    for x in Xq:
        dist2 = [(sum(((u - v) ** 2 for u, v in zip(x, row)), Fraction(0)), j) for j, row in enumerate(Yd)]
        near = sorted(j for _, j in sorted(dist2)[: opt["neighbors"]])
        Ys, ds = [Yd[j] for j in near], [d[j] for j in near]
        out += settle_all(lambda p, Ys=Ys, ds=ds, x=x: rbf_values(Ys, ds, [x], kernel, epsilon, degree, 0, p))
    return out


# ---------------------------------------------------------------- roots


def roots_task(pid):
    """[(piece, multiplicity, (hi, lo))] of the distinct real roots of each piece in [0, h]."""
    from flint import fmpq, fmpq_poly

    M = S()
    P = M.pieces(pid)
    out = []
    for i in range(P.c.shape[1]):
        coeffs = [fr(v) for v in P.c[::-1, i]]  # lowest first
        if all(c == 0 for c in coeffs):
            continue  # a zero piece: scipy's [x[i], nan]; no reference
        p = fmpq_poly([fmpq(c.numerator, c.denominator) for c in coeffs])
        if p.degree() < 1:
            continue
        h = fr(P.x[i + 1]) - fr(P.x[i])
        _, factors = p.factor_squarefree()
        found = []
        for f, mult in factors:
            z = f.numer()
            for prec in (256, 512, 1024, 2048, 4096):
                with arbref.precision(prec):
                    roots = [r for r, _ in z.complex_roots() if r.imag == 0 and r.imag.is_exact()]
                    if all(arbref.settled(r.real) or r.real == 0 for r in roots):
                        break
            for r in roots:
                re = r.real
                # exact checks at the ends, Arb inside
                if re.contains(0) and poly_eval(f, Fraction(0)) == 0:
                    found.append((Fraction(0), mult))
                    continue
                if re.contains(_arb_of(h)) and poly_eval(f, h) == 0:
                    found.append((h, mult))
                    continue
                if re > 0 and re < _arb_of(h):
                    found.append((re, mult))
        for v, mult in found:
            x0 = fr(P.x[i])
            if isinstance(v, Fraction):
                hi_lo = dd_of(x0 + v)
            else:
                with arbref.precision(4096):
                    hi_lo = arbref.to_dd(_arb_of(x0) + v)
            out.append((i, mult, hi_lo))
    out.sort(key=lambda t: (t[0], t[2][0]))
    return out


# ---------------------------------------------------------------- scattered (IP9)


def scattered_task(sid):
    """Per point of the set: (index, simplex, {method: (hi, lo)}) and the set's gradients (n, 2) (None for d = 3 or
    gradients that are not finite)."""
    import stardust as sd
    from tests.interpolate import refdata as R
    from tests.spatial import exactgeom as X

    sc = S().tri_set(sid)
    y, f, xi = sc.y, sc.d, sc.xi
    d = y.shape[1]
    D = sd.spatial.Delaunay(sd.asarray(y))
    simp, nb = np.asarray(D.simplices), np.asarray(D.neighbors)
    ts = np.asarray(D.find_simplex(sd.asarray(xi)))
    grad = None
    if d == 2:
        g = np.asarray(sd.interpolate.CloughTocher2DInterpolator(y, f).grad)
        grad = g if np.all(np.isfinite(g)) else None
    out = []
    for i, (x, t) in enumerate(zip(xi, ts)):
        assert int(t) == X.find_simplex_reference(y, simp.tolist(), x), (sid, i)
        if t < 0:
            continue
        v, scale = R.scattered_linear_exact(y, simp[t], f, x)
        vals = {"linear": dd_of(v), "linear_scale": dd_of(scale)}
        if grad is not None:
            c, K, _ = R.ct_exact(y, simp, nb, int(t), f, grad, x)
            vals["clough_tocher"], vals["ct_scale"] = dd_of(c), dd_of(K)
        out.append((i, int(t), vals))
    return out, grad


# ---------------------------------------------------------------- driver


def _items(fam):
    M = S()
    return {
        "linear": linear_items,
        "curves": lambda: list(M.real_curve_ids()),
        "polynomial": lambda: list(M.POLY_IDS),
        "grid": grid_items,
        "rbf": rbf_items,
        "roots": lambda: list(M.PIECES_IDS),
        "scattered": lambda: list(M.tri_ids()),
    }[fam]()


TASKS = {"linear": linear_task, "curves": curve_task, "polynomial": poly_task, "grid": grid_task, "rbf": rbf_task,
         "roots": roots_task, "scattered": scattered_task}  # fmt: skip
# the files each task family writes
FILES = {"linear": ("linear",), "curves": ("slopes", "bspline", "curves"), "polynomial": ("polynomial",),
         "grid": ("grid",), "rbf": ("rbf",), "roots": ("roots",), "scattered": ("scattered",)}  # fmt: skip
FAMILIES = tuple(f for fs in FILES.values() for f in fs)


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


def _methods(per_point, n_points, dropped=()):
    """{method: {...}} from {method: {point: (hi, lo)}}."""
    out = {}
    for m, d in per_point.items():
        idx = sorted(d)
        out[m] = {"over": "point", "index": np.array(idx, np.int64), "hi": np.array([d[i][0] for i in idx]),
                  "lo": np.array([d[i][1] for i in idx]), "dropped": np.array(sorted(dropped), np.int64)}  # fmt: skip
    return out


def _write(name, inputs, methods, n_requested, **kw):
    from tests import reffile

    reffile.write(OUT / f"{name}.npz", **arbref.provenance(extra="exact rationals (Fraction, fmpq_mat)"), tool=TOOL,
                  inputs=inputs, methods=methods, n_requested=n_requested, **kw)  # fmt: skip
    print(f"{name}: {n_requested} points -> {(OUT / f'{name}.npz').relative_to(ROOT)}", flush=True)


def write_linear(items, results):
    sets, idx, dig, nums, vals = [], [], [], [], {}
    for k in sorted(results):
        for sid, i, d, inp, v in results[k]:
            vals[len(sets)] = v
            sets.append(sid), idx.append(i), dig.append(d), nums.append(inp)
    nums = np.array(nums).reshape(-1, 5)
    inputs = {"set": np.array(sets), "index": np.array(idx, np.int64), "digest": np.array(dig)}
    for j, name in enumerate(("x0", "x1", "a", "b", "x")):
        inputs[name] = nums[:, j].copy()
    _write("linear", inputs, _methods({"lerp": vals}, len(sets)), len(sets))


def write_curves(items, results):
    M = S()
    # slopes: one point per node
    sets, idx, dig, xs, per = [], [], [], [], {}
    for k in sorted(results):
        c = M.curve(items[k])
        for i in range(c.n):
            p = len(sets)
            for m, s in results[k]["slopes"].items():
                per.setdefault(m, {})[p] = s[i]
            sets.append(c.id), idx.append(i), dig.append(c.digest), xs.append(c.x[i])
    inputs = {"set": np.array(sets), "index": np.array(idx, np.int64), "digest": np.array(dig), "x": np.array(xs)}
    _write("slopes", inputs, _methods({m: per[m] for m in SLOPE_METHODS if m in per}, len(sets)), len(sets))
    # bspline: one point per coefficient, grouped by curve; group values the norms
    sets, idx, dig, group, per, norms, gsets = [], [], [], [], {}, {}, []
    knots = {m: [] for m in BSPLINE_METHODS}
    for g, k in enumerate(sorted(results)):
        c = M.curve(items[k])
        gsets.append(c.id)
        coefs = results[k]["coefs"]
        width = max(len(v) for v in coefs.values())
        for j in range(width):
            p = len(sets)
            for m, s in coefs.items():
                if j < len(s):
                    per.setdefault(m, {})[p] = s[j]
            sets.append(c.id), idx.append(j), dig.append(c.digest), group.append(g)
        for m in BSPLINE_METHODS:
            knots[m].append(results[k]["knots"].get(m, np.zeros(0)))
        for m, v in results[k]["norms"].items():
            norms.setdefault(m, {})[g] = v
    methods = _methods({m: per[m] for m in BSPLINE_METHODS if m in per}, len(sets))
    for m, d in norms.items():
        gi = sorted(d)
        methods[f"ainv_{m}"] = {"over": "group", "index": np.array(gi, np.int64), "hi": np.array([d[i][0] for i in gi]),
                                "lo": np.array([d[i][1] for i in gi]), "dropped": np.zeros(0, np.int64)}  # fmt: skip
    inputs = {"set": np.array(sets), "index": np.array(idx, np.int64), "digest": np.array(dig)}
    ginputs = {"set": np.array(gsets), **{f"t_{m}": knots[m] for m in BSPLINE_METHODS}}
    _write("bspline", inputs, methods, len(sets), group=np.array(group, np.int64), group_inputs=ginputs)
    # curves: one point per evaluation point
    sets, idx, dig, xs, per = [], [], [], [], {}
    for k in sorted(results):
        c = M.curve(items[k])
        for i, v in enumerate(c.xe):
            p = len(sets)
            for m, vals in results[k]["values"].items():
                per.setdefault(m, {})[p] = vals[i]
            sets.append(c.id), idx.append(i), dig.append(c.digest), xs.append(v)
    order = [m for m in SLOPE_METHODS + BSPLINE_METHODS if m in per]
    inputs = {"set": np.array(sets), "index": np.array(idx, np.int64), "digest": np.array(dig), "x": np.array(xs)}
    _write("curves", inputs, _methods({m: per[m] for m in order}, len(sets)), len(sets))


def _write_eval(name, items, results, getter, xs_of):
    M = S()
    sets, idx, dig, xs, per = [], [], [], [], {}
    for k in sorted(results):
        obj = getter(M, items[k])
        pts = xs_of(obj)
        res = results[k]
        for i in range(len(pts)):
            p = len(sets)
            for m, vals in res.items():
                if vals[i] is not None:
                    per.setdefault(m, {})[p] = vals[i]
            sets.append(obj.id), idx.append(i), dig.append(obj.digest), xs.append(pts[i])
    inputs = {"set": np.array(sets), "index": np.array(idx, np.int64), "digest": np.array(dig), "x": xs}
    _write(name, inputs, _methods(per, len(sets)), len(sets))


def write_polynomial(items, results):
    _write_eval("polynomial", items, results, lambda M, i: M.poly(i), lambda P: list(P.x))


def write_grid(items, results):
    _write_eval("grid", items, results, lambda M, i: M.grid(i), lambda G: list(G.xi))


def write_rbf(items, results):
    M = S()
    by_set = {}
    for k in sorted(results):
        sid, name = items[k]
        by_set.setdefault(sid, {})[name] = results[k]
    sids = list(by_set)
    _write_eval("rbf", sids, {i: by_set[s] for i, s in enumerate(sids)}, lambda M_, i: M.scatter(i),
                lambda sc: list(sc.xi))  # fmt: skip


def write_roots(items, results):
    M = S()
    sets, idx, dig, piece, mult, vals = [], [], [], [], [], {}
    for k in sorted(results):
        P = M.pieces(items[k])
        for j, (i, m, v) in enumerate(results[k]):
            vals[len(sets)] = v
            sets.append(P.id), idx.append(j), dig.append(P.digest), piece.append(i), mult.append(m)
    inputs = {"set": np.array(sets), "index": np.array(idx, np.int64), "digest": np.array(dig),
              "piece": np.array(piece, np.int64), "multiplicity": np.array(mult, np.int64)}  # fmt: skip
    _write("roots", inputs, _methods({"root": vals}, len(sets)), len(sets))


def write_scattered(items, results):
    M = S()
    sets, idx, dig, xs, simplex, group, per = [], [], [], [], [], [], {}
    gsets, grads = [], []
    for g, k in enumerate(sorted(results)):
        sc = M.tri_set(items[k])
        rows, grad = results[k]
        gsets.append(sc.id)
        grads.append(np.zeros((0, 2)) if grad is None else np.ascontiguousarray(grad))
        for i, t, vals in rows:
            p = len(sets)
            for m, v in vals.items():
                per.setdefault(m, {})[p] = v
            sets.append(sc.id), idx.append(i), dig.append(sc.digest), xs.append(sc.xi[i]), simplex.append(t)
            group.append(g)
    order = [m for m in ("linear", "linear_scale", "clough_tocher", "ct_scale") if m in per]
    inputs = {"set": np.array(sets), "index": np.array(idx, np.int64), "digest": np.array(dig), "x": xs,
              "simplex": np.array(simplex, np.int64)}  # fmt: skip
    _write("scattered", inputs, _methods({m: per[m] for m in order}, len(sets)), len(sets),
           group=np.array(group, np.int64), group_inputs={"set": np.array(gsets), "grad": grads})  # fmt: skip


WRITERS = {"linear": write_linear, "curves": write_curves, "polynomial": write_polynomial, "grid": write_grid,
           "rbf": write_rbf, "roots": write_roots, "scattered": write_scattered}  # fmt: skip


def build(names, jobs):
    import resource

    start = time.monotonic()
    plans = {f: _items(f) for f in names}
    tasks = [(f, k, it) for f in names for k, it in enumerate(plans[f])]
    results = {f: {} for f in names}
    dropped = {f: {} for f in names}
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        for fam, k, res, secs, err in pool.imap_unordered(_run, tasks, chunksize=1):
            if res is None:
                dropped[fam][k] = err
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]}: {secs:.0f} s", flush=True)
    for f in names:
        for k, err in sorted(dropped[f].items()):
            print(f"  dropped {f} {plans[f][k]}: {err}", flush=True)
        WRITERS[f](plans[f], results[f])
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"{len(tasks)} tasks, {(time.monotonic() - start):.0f} s, largest worker RSS {rss:.2f} GB", flush=True)


def main(argv=None):
    from bench import cli

    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their data")
    cli.add_jobs(ap, default=3)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    wanted = a.names or list(FAMILIES)
    names = [t for t, fs in FILES.items() if set(fs) & set(wanted)]
    if a.list:
        for f in names:
            items = _items(f)
            print(f"{'/'.join(FILES[f])} ({len(items)}): " + ", ".join(map(str, items)))
        return 0
    with arbref.logged("interpolate_ref"):
        build(names, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
