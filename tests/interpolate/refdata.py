"""sd.interpolate's references and their contract (docs/design/interpolate.md §4, §7), as the tests and the
accuracy measurement (bench/accuracy) will both judge them.

tests/interpolate/ref/<family>.npz (tools/interpolate_ref.py; layout: tests/reffile.py) holds one point per
evaluation point, node, coefficient or root, with the inputs "set" (the id in tests/interpolate/samples.py,
"lerp/<id>" and "curve/<id>" in the linear family), "index" (the point's number in its set) and
"digest" (the data's digest), the family's own inputs, and per method the reference hi + lo:

    linear      x0, x1, a, b, x; method lerp (the exact a + (x - x0) / (x1 - x0) (b - a))
    slopes      x (the node); methods natural, clamped, not_a_knot, periodic, pchip, akima, makima
    bspline     grouped by curve (group input "set" and the knots t_<method>); methods k0 .. k5,
                k3_natural, k3_clamped, k3_periodic, k5_periodic (the coefficients), and over the groups
                ainv_<method> (the exact infinity norm of the inverse of the system's matrix)
    curves      x; the slopes' and the B-splines' methods: the exact interpolant's value
    polynomial  x; methods value, d1, d2, lebesgue, abs_terms, fh3
    grid        x (the point (d,)); methods linear, cubic, quintic, pchip
    rbf         x (the point (d,)); one method per kernel, thin_plate_spline_s0.1 (smoothing 0.1) and
                thin_plate_spline_k8 (neighbors = 8)
    roots       piece, multiplicity; method root

tests/interpolate/data holds the other libraries' records (tools/interpolate_probe.py: scipy.json,
numpy.json, r.json), each with its own "_provenance".

The contract of the linear kernel (docs/design/interpolate.md §3.2 proves it, §4): with
t = (x - x0) / (x1 - x0) exact and y = a + t (b - a) exact, the computed value is within
gamma_6 (|a| + |b|) + 3 eta of y for 0 <= t <= 1 and within gamma_10 max(1, |t|) (|a| + |b|) + 3 eta outside
(decision IP40), eta = 2^-1075 (the products that underflow); at the nodes it is exact. 3 eta = 1.5 2^-1074 is
not a double; the bound uses the double above it, 2^-1073 (ETA3).

The cubic family's contracts (milestone IP2; docs/design/interpolate.md §3.3, §3.5 prove them, §4):

  spline slopes      |s^ - s| <= gamma_12 |A^-1| (|A| |s^| + |b|) componentwise (spline_bound): A and b the slopes'
                     system and right side of the exact data (|b| with the absolute values of its terms); natural and
                     clamped also within 150 u max|D| (D the divided differences), periodic within 150 u max|D|
  pchip slopes       |s^_i - s_i| <= 40 u max(|D| of the two intervals the formula reads)
  akima slopes       akima_bound: the formula's rounding with its conditioning, 3 E + 6 u M + |m2 - m1| (9 E / f12 +
                     gamma_6), E = 3 gamma_8 M, M the largest |m| of the node's window (2 E in the fallback branch)
  Hermite coefficients  hermite_bound: |c3^ - c3| <= gamma_8 (|s0| + |s1| + 2|D|) / h^2 + 2^-1074 (1 + 3/h + 2/h^2),
                     |c2^ - c2| <= gamma_7 (3|D| + 2|s0| + |s1|) / h + 2^-1074 (2 + 3/h) for the computed slopes
                     (the floor: each rounding of a subnormal result adds up to 2^-1075; decision IP51), c1, c0 exact
  evaluation         |p^ - p(x)| <= sum_e gamma_{2e+2} f_e |c_e| |s|^e + 2^-1074 sum_e |s|^e (ppoly_bound): e the degree
                     of a term after nu derivatives, f_e its factor, s = x - x[i] exact
  roots              see tests/interpolate/test_roots.py

The polynomial and rational interpolants' contracts (milestone IP4; docs/design/interpolate.md §3.6 proves them,
§4), N the number of nodes (support points), no weight or term underflowing:

  Barycentric        |p^ - p(x)| <= (gamma_{3N+1} sum|l_j(x) y_j| + gamma_{3N} L(x) |p(x)|) / (1 - gamma_{3N-1} L(x))
                     (barycentric_bound), p the exact interpolant of the data, L the Lebesgue function: the computed
                     weights' rounding included
  its derivatives    the nodal values D^k y (rational_nodal_bound: per order gamma_{6N} sum_{j != i} |D_ij| (|v_j| +
                     |v_i|) plus |D| times the previous order's bound), then the evaluation contract of the computed
                     nodal values plus sum_j |l_j(x)| times their bound (barycentric_derivative_bound)
  Krogh table        |c^_k - c_k| <= gamma_{3k+2} C~_k + E~_k (krogh_table_bound): C~ the table of the absolute values
                     (|a| + |b|) / |x_i - x_k| from |y| / s!, E~ the floor of the subnormal roundings
  Krogh evaluation   |p^(m)^ - p^(m)| <= m! (gamma_{2N+1} A~_m(|c^|) + 2^-1074 ((m + 1) A~_m(1) + 1)) (krogh_bound):
                     A~_m the m + 1 Horner passes on |c^| and |x - x_k|; end to end the table bound joins as A~_m
  a rational         |r^ - r| <= (gamma_a sum|lam_j f_j| + gamma_b |r| sum|lam_j|) / (|sum lam_j| - gamma_c sum|lam_j|)
                     (rational_bound), lam_j = w_j / (z - z_j) of the computed weights: (a, b, c) = (N + 3, N + 2,
                     N + 1) real (Floater-Hormann, a real AAA), (3N + 16, 2N + 16, 2N + 8) complex (AAA); the
                     Floater-Hormann weights are within gamma_{3d} of the exact ones, relatively, which adds 3d to
                     a, b and c end to end

The grids' contracts (milestone IP5; docs/design/interpolate.md §4, decisions IP79 and IP81), d the dimension:

  linear             the 1-D linear contract stacked axis by axis from the last to the first (grid_linear_exact): each
                     level linear_bound(|a^|, |b^|, t) plus (1 - t) and t times its inputs' bounds; inside the grid at
                     most d (gamma_13 M + 3 eta), M the largest |value| of the 2^d corners
  tensor spline      coefficients (tensor_collocation_bound): sum over the axes a of 2 gamma_{8k+8} |A_a^-1| |A_a| |R_a|
                     along a (R_a the coefficients after a's solve) carried through |A_b^-1| of the later axes b;
                     evaluation (tensor_bound): gamma_{d(6k+1)} sum_j |c_j| prod B~ + d (6k + 2) 2^-1074 (1 + 2^d
                     sum_j |c_j|), and end to end plus sum_j (coefficient bound)_j prod B~
"""

from __future__ import annotations

import json
import math
import pathlib

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("bspline", "curves", "grid", "linear", "polynomial", "rbf", "roots", "slopes")
U = 2.0**-53
ETA3 = 2.0**-1073  # 3 eta rounded up to a double (eta = 2^-1075 itself rounds to 0)


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def gamma(k: int) -> float:
    return k * U / (1 - k * U)


def linear_bound(a: float, b: float, t: float) -> float:
    """The linear kernel's contract: the absolute bound at t (the exact t of the point)."""
    s = abs(a) + abs(b)
    if 0.0 <= t <= 1.0:
        return gamma(6) * s + ETA3
    return gamma(10) * max(1.0, abs(t)) * s + ETA3


def linear_units(got: float, hi: float, lo: float, a: float, b: float, t: float) -> float:
    """|got - ref| in units of the linear contract (0 when both are exactly equal)."""
    err = abs((got - hi) - lo)
    if err == 0:
        return 0.0
    return err / linear_bound(a, b, t)


def isclose_dd(got: float, hi: float, lo: float) -> bool:
    """got is the reference rounded to a double (hi), or the reference is exactly a double."""
    return got == hi or (math.isnan(got) and math.isnan(hi))


# ---------------------------------------------------------------- the cubic family (IP2)


def spline_system(x, y, bc):
    """(A, |b|) of CubicSpline's slope system for the exact data, in float64 (the bound's scale): bc is
    (start, end), each 'not-a-knot', 'periodic' or (order, value); the periodic system has n - 1 unknowns."""
    import numpy as np

    n = len(x)
    h = np.diff(x)
    D = np.diff(y) / h
    b0, b1 = bc
    if n == 2:
        b0 = (1, D[0]) if b0 in ("not-a-knot", "periodic") else b0
        b1 = (1, D[0]) if b1 in ("not-a-knot", "periodic") else b1
    if b0 == "periodic":
        m = n - 1
        A = np.zeros((m, m))
        b = np.zeros(m)
        for i in range(m):
            hl, hr = h[i - 1] if i else h[n - 2], h[i]
            dl, dr = D[i - 1] if i else D[n - 2], D[i]
            A[i, (i - 1) % m] += hr
            A[i, i] += 2 * (hl + hr)
            A[i, (i + 1) % m] += hl
            b[i] = 3 * (hr * abs(dl) + hl * abs(dr))
        return A, b
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i], 2 * (h[i - 1] + h[i]), h[i - 1]
        b[i] = 3 * (h[i] * abs(D[i - 1]) + h[i - 1] * abs(D[i]))
    if n == 3 and b0 == "not-a-knot" and b1 == "not-a-knot":
        A[0, :2] = 1
        A[2, 1:] = 1
        b[0], b[2] = 2 * abs(D[0]), 2 * abs(D[1])
        return A, b
    if b0 == "not-a-knot":
        d = x[2] - x[0]
        A[0, 0], A[0, 1] = h[1], d
        b[0] = ((h[0] + 2 * d) * h[1] * abs(D[0]) + h[0] ** 2 * abs(D[1])) / d
    elif b0[0] == 1:
        A[0, 0], b[0] = 1.0, abs(b0[1])
    else:
        A[0, 0], A[0, 1] = 2 * h[0], h[0]
        b[0] = 0.5 * abs(b0[1]) * h[0] ** 2 + 3 * abs(y[1] - y[0])
    if b1 == "not-a-knot":
        d = x[-1] - x[-3]
        A[-1, -1], A[-1, -2] = h[-2], d
        b[-1] = (h[-1] ** 2 * abs(D[-2]) + (2 * d + h[-1]) * h[-2] * abs(D[-1])) / d
    elif b1[0] == 1:
        A[-1, -1], b[-1] = 1.0, abs(b1[1])
    else:
        A[-1, -1], A[-1, -2] = 2 * h[-1], h[-1]
        b[-1] = 0.5 * abs(b1[1]) * h[-1] ** 2 + 3 * abs(y[-1] - y[-2])
    return A, b


def spline_bound(x, y, bc, s_hat):
    """The componentwise bound of CubicSpline's slopes (one per node): gamma_12 |A^-1| (|A| |s^| + |b|)."""
    import numpy as np

    A, b = spline_system(np.asarray(x, float), np.asarray(y, float), bc)
    s = np.asarray(s_hat, float)[: A.shape[0]]
    Ainv = np.abs(np.linalg.inv(A))
    out = gamma(12) * (Ainv @ (np.abs(A) @ np.abs(s) + b))
    if bc[0] == "periodic" and len(x) > 2:
        out = np.append(out, out[0])
    return out


def pchip_bound(x, y):
    """40 u max(|D|) of the intervals each node's PCHIP slope reads (one per node)."""
    import numpy as np

    D = np.abs(np.diff(y) / np.diff(x))
    n = len(x)
    if n == 2:
        return np.full(2, 40 * U * D[0])
    out = np.empty(n)
    out[1:-1] = np.maximum(D[:-1], D[1:])
    out[0] = max(D[0], D[1])
    out[-1] = max(D[-1], D[-2])
    return 40 * U * out


def akima_bound(x, y, makima):
    """The rounding bound of each node's Akima slope with its conditioning (see the module docstring)."""
    import numpy as np

    n = len(x)
    D = np.diff(y) / np.diff(x)
    if n == 2:
        return np.full(2, gamma(3) * abs(D[0]))
    m = np.empty(n + 3)
    m[2:-2] = D
    m[1] = 2 * m[2] - m[3]
    m[0] = 2 * m[1] - m[2]
    m[-2] = 2 * m[-3] - m[-4]
    m[-1] = 2 * m[-2] - m[-3]
    dm = np.abs(np.diff(m))
    if makima:
        pm = np.abs(m[1:] + m[:-1])
        f1, f2 = dm[2:] + 0.5 * pm[2:], dm[:-2] + 0.5 * pm[:-2]
    else:
        f1, f2 = dm[2:], dm[:-2]
    f12 = f1 + f2
    cut = 1e-9 * np.max(f12)
    out = np.empty(n)
    for i in range(n):
        M = np.max(np.abs(m[i : i + 4]))
        E = 3 * gamma(8) * M
        if f12[i] > cut:
            out[i] = 3 * E + 6 * U * M + abs(m[i + 2] - m[i + 1]) * (9 * E / f12[i] + gamma(6))
        else:
            out[i] = 2 * E
    return 1.5 * out


def ppoly_bound(coeffs, s, nu=0):
    """The evaluation contract at the exact local s: coeffs highest degree first (one piece, one column)."""
    import math as _m

    k = len(coeffs) - 1
    total, eta = 0.0, 0.0
    a = abs(s)
    for j, c in enumerate(coeffs):
        d = k - j
        if d < nu:
            continue
        e = d - nu
        f = _m.perm(d, nu)
        total += gamma(2 * e + 2) * f * abs(c) * a**e
        eta += a**e
    return 1.01 * total + 2.0**-1074 * 1.01 * eta


def hermite_bound(h, s0, s1, d):
    """(bound of c3, bound of c2) of the Hermite coefficients built from the computed slopes s0, s1 and the
    divided difference d over a width h (all floats): the relative part and the floor of the subnormal roundings
    (the divided difference's, the sums', the quotients'; decision IP51)."""
    h, s0, s1, d = abs(h), abs(s0), abs(s1), abs(d)
    b3 = gamma(8) * (s0 + s1 + 2 * d) / h**2 + 2.0**-1074 * (1 + 3 / h + 2 / h**2)
    b2 = gamma(7) * (3 * d + 2 * s0 + s1) / h + 2.0**-1074 * (2 + 3 / h)
    return 1.01 * b3, 1.01 * b2


# ---------------------------------------------------------------- B-splines (IP3)
#
#   evaluation      |y^ - sum_j c_j B_j^(nu)(x)| <= gamma_{6k+1} sum_j |c_j| B~_j(x) + (6k + 2) 2^-1074
#                   (1 + sum_j |c_j|) (bspline_bound): c the computed coefficients, B~ the recursion of the basis
#                   (scipy's, with the weights as two quotients) run on the absolute values of its terms (B itself
#                   inside the base interval for nu = 0); the floor counts the subnormal roundings; a sum that
#                   overflows makes the bound infinite (large/1e300's high derivatives)
#   construction    |c^ - c| <= 2 gamma_{8k+8} |A^-1| |A| |c^| componentwise (collocation_bound): A the collocation
#                   matrix of the exact data (the periodic splines' folded one), the factor 2 the growth of the
#                   elimination without pivoting (docs/design/interpolate.md §3.4); k = 0, 1 copy y exactly

BSPLINE_METHODS = ("k0", "k1", "k2", "k3", "k4", "k5", "k3_natural", "k3_clamped", "k3_periodic", "k5_periodic")


def bspline_args(method):
    """(k, bc_type) of make_interp_spline for a method of bspline.npz."""
    k = int(method[1])
    for bc in ("natural", "clamped", "periodic"):
        if method.endswith(bc):
            return k, bc
    return k, None


def knot_interval(t, k, x):
    """The l in [k, n - 1] with t[l] <= x < t[l + 1] (x >= t[n]: n - 1; x < t[k]: k), n = len(t) - k - 1."""
    n = len(t) - k - 1
    count = sum(1 for v in t if v <= x)
    return min(max(count - 1, k), n - 1)


def _basis(t, k, l, x, nu, zero, absolute=False):
    """The recursion of csrc/interpolate/bspline.cpp (scipy's _deBoor_D): values of B_{l-k} .. B_l (or their
    derivatives of order nu) at x, in the arithmetic of the given numbers (Fractions: exact); absolute=True runs it
    on the absolute values of its terms."""
    h = [zero] * (k + 1)
    h[0] = zero + 1
    a = abs if absolute else (lambda v: v)
    for j in range(1, k - nu + 1):
        hh = h[:j]
        h[0] = zero
        for m in range(1, j + 1):
            xb, xa = t[l + m], t[l + m - j]
            if xb == xa:
                h[m] = zero
                continue
            d = xb - xa
            h[m - 1] = h[m - 1] + a(hh[m - 1]) * a((xb - x) / d)
            h[m] = a(hh[m - 1]) * a((x - xa) / d)
    for j in range(k - nu + 1, k + 1):
        hh = h[:j]
        h[0] = zero
        for m in range(1, j + 1):
            xb, xa = t[l + m], t[l + m - j]
            if xb == xa:
                h[m] = zero
                continue
            w = j * a(hh[m - 1]) / a(xb - xa)
            h[m - 1] = h[m - 1] + w if absolute else h[m - 1] - w
            h[m] = w
    return h


def bspline_exact(t, c, k, x, nu=0):
    """sum_j c_j B_j^(nu)(x) exactly (a Fraction) for float knots, coefficients and point (x not NaN)."""
    from fractions import Fraction

    if nu > k:
        return Fraction(0)
    T = [Fraction(float(v)) for v in t]
    X = Fraction(float(x))
    l = knot_interval(T, k, X)
    h = _basis(T, k, l, X, nu, Fraction(0))
    return sum((Fraction(float(c[l - k + r])) * h[r] for r in range(k + 1)), Fraction(0))


def bspline_bound(t, c, k, x, nu=0):
    """The evaluation contract at x for one column of computed coefficients c (floats)."""
    if nu > k:
        return 0.0
    t = [float(v) for v in t]
    l = knot_interval(t, k, float(x))
    hb = _basis(t, k, l, float(x), nu, 0.0, absolute=True)
    s = sum(abs(float(c[l - k + r])) * hb[r] for r in range(k + 1))
    cs = sum(abs(float(c[l - k + r])) for r in range(k + 1))
    return 1.01 * gamma(6 * k + 1) * s + 1.01 * (6 * k + 2) * 2.0**-1074 * (1 + cs)


def collocation_matrix(x, t, k, bc):
    """make_interp_spline's matrix for the exact data, in floats (each entry the exact value rounded once): the
    interpolation rows and, for bc 'natural' / 'clamped', the second / first derivative at both ends; 'periodic'
    the folded matrix of the n - 1 unknowns d (c_j = d_{j mod (n - 1)}). Rows in any order (the bound does not
    depend on it)."""
    from fractions import Fraction

    import numpy as np

    T = [Fraction(float(v)) for v in t]
    X = [Fraction(float(v)) for v in x]
    n, nt = len(X), len(T) - k - 1
    if bc == "periodic":
        m = n - 1
        A = np.zeros((m, m))
        for i in range(m):
            l = knot_interval(T, k, X[i])
            for r, b in enumerate(_basis(T, k, l, X[i], 0, Fraction(0))):
                A[i, (l - k + r) % m] += float(b)
        return A
    rows = []
    points = [(v, 0) for v in X]
    if bc in ("natural", "clamped"):
        nu = 2 if bc == "natural" else 1
        points += [(X[0], nu), (X[-1], nu)]
    for v, nu in points:
        row = np.zeros(nt)
        l = knot_interval(T, k, v)
        for r, b in enumerate(_basis(T, k, l, v, nu, Fraction(0))):
            row[l - k + r] = float(b)
        rows.append(row)
    return np.array(rows)


def collocation_bound(x, t, k, bc, c_hat):
    """The construction contract, one bound per coefficient: 2 gamma_{8k+8} |A^-1| |A| |c^|."""
    import numpy as np

    A = collocation_matrix(x, t, k, bc)
    c = np.abs(np.asarray(c_hat, float))
    m = A.shape[0]
    out = 2 * gamma(8 * k + 8) * (np.abs(np.linalg.inv(A)) @ (np.abs(A) @ c[:m]))
    if bc == "periodic":
        out = out[np.arange(len(c)) % m]
    return 1.01 * out


# ---------------------------------------------------------------- polynomial and rational interpolation (IP4)


def barycentric_bound(n: int, abs_terms: float, lebesgue: float, p: float) -> float:
    """The second barycentric formula's contract (weights computed, n nodes): docs/design/interpolate.md §3.6."""
    den = 1 - gamma(3 * n - 1) * lebesgue
    if den <= 0:
        return math.inf
    return (gamma(3 * n + 1) * abs_terms + gamma(3 * n) * lebesgue * abs(p)) / den


def rational_bound(n_terms: int, s_f: float, s: float, d: float, r: float, a: int, b: int, c: int) -> float:
    """(gamma_a s_f + gamma_b |r| s) / (|d| - gamma_c s): s_f = sum|lam_j f_j|, s = sum|lam_j|, d = sum lam_j."""
    den = abs(d) - gamma(c) * s
    if den <= 0:
        return math.inf
    return (gamma(a) * s_f + gamma(b) * abs(r) * s) / den


def lagrange_basis(X, v):
    """[l_j(v)] exactly (Fractions) for distinct nodes X (Fractions) and a point v (Fraction)."""
    from fractions import Fraction

    if v in X:
        return [Fraction(int(v == a)) for a in X]
    terms = []
    for j, a in enumerate(X):
        prod = Fraction(1)
        for k, b in enumerate(X):
            if k != j:
                prod *= a - b
        terms.append(1 / (prod * (v - a)))
    tot = sum(terms, Fraction(0))
    return [t / tot for t in terms]


def differentiation_matrix(X):
    """D_ij = (w_j / w_i) / (x_i - x_j), D_ii = -sum_{j != i} D_ij, exactly (Fractions)."""
    from fractions import Fraction

    n = len(X)
    w = []
    for j in range(n):
        prod = Fraction(1)
        for k in range(n):
            if k != j:
                prod *= X[j] - X[k]
        w.append(1 / prod)
    D = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j != i:
                D[i][j] = (w[j] / w[i]) / (X[i] - X[j])
        D[i][i] = -sum((D[i][j] for j in range(n) if j != i), Fraction(0))
    return D


def rational_nodal_bound(D, v_hat, prev_bound):
    """The bound of one more order's nodal values D v (v_hat the computed previous order, prev_bound its bound)."""
    n = len(D)
    out = []
    for i in range(n):
        off = sum(abs(D[i][j]) * (abs(v_hat[j]) + abs(v_hat[i])) for j in range(n) if j != i)
        prop = sum(abs(D[i][j]) * prev_bound[j] for j in range(n))
        out.append(gamma(6 * n) * float(off) + float(prop))
    return out


def krogh_table_exact(X, Y):
    """scipy's Hermite divided-difference table in exact arithmetic (Fractions), one column."""
    from fractions import Fraction

    n = len(X)
    c = [Fraction(0)] * n
    c[0] = Y[0]
    V = [Fraction(0)] * n
    for k in range(1, n):
        s = 0
        while s <= k and X[k - s] == X[k]:
            s += 1
        s -= 1
        V[0] = Y[k] / math.factorial(s)
        for i in range(k - s):
            V[i + 1] = ((c[i] if s == 0 else V[i + 1]) - V[i]) / (X[i] - X[k])
        c[k] = V[k - s]
    return c


def krogh_table_bound(X, Y):
    """The construction contract per coefficient: gamma_{3k+2} C~_k + E~_k (docs/design/interpolate.md §3.6)."""
    from fractions import Fraction

    n = len(X)
    eta = Fraction(1, 2**1074)
    ca, ce = [Fraction(0)] * n, [Fraction(0)] * n
    ca[0] = abs(Y[0])
    Va, Ve = [Fraction(0)] * n, [Fraction(0)] * n
    for k in range(1, n):
        s = 0
        while s <= k and X[k - s] == X[k]:
            s += 1
        s -= 1
        Va[0] = abs(Y[k]) / math.factorial(s)
        Ve[0] = eta if s else Fraction(0)
        for i in range(k - s):
            d = abs(X[i] - X[k])
            a, ae = (ca[i], ce[i]) if s == 0 else (Va[i + 1], Ve[i + 1])
            Va[i + 1] = (a + Va[i]) / d
            Ve[i + 1] = (ae + Ve[i]) / d + eta
        ca[k], ce[k] = Va[k - s], Ve[k - s]
    return [gamma(3 * k + 2) * float(ca[k]) * 1.01 + float(ce[k]) * 1.01 for k in range(n)]


def newton_passes(X, c, v, m):
    """The Horner passes of KroghInterpolator in exact arithmetic: p^(m)(v) / m! (X, c, v Fractions)."""
    b = list(c)
    n = len(b)
    s = [v - X[k] for k in range(n - 1)]
    for p in range(m + 1):
        for k in range(n - 2, p - 1, -1):
            b[k] = b[k] + s[k - p] * b[k + 1]
    return b[m] if m < n else 0


def krogh_bound(X, c_abs, v, m, table_bound=None):
    """The evaluation contract of the derivative of order m at v (exact X, v; c_abs = |c^|), plus the table's
    bound carried through the same passes when given (end to end)."""
    from fractions import Fraction

    n = len(X)
    if m >= n:
        return 0.0
    absx = [Fraction(abs(v - X[k])) for k in range(n)]
    Xa = [v - absx[k] for k in range(n)]  # centres whose distances to v are |v - x_k|
    a = newton_passes(Xa, [Fraction(abs(t)) for t in c_abs], v, m)
    ones = newton_passes(Xa, [Fraction(1)] * n, v, m)
    f = math.factorial(m)
    out = f * (gamma(2 * n + 1) * float(a) + 2.0**-1074 * ((m + 1) * float(ones) + 1))
    if table_bound is not None:
        out += f * float(newton_passes(Xa, [Fraction(t) for t in table_bound], v, m)) * (1 + gamma(2 * n + 1))
    return out * 1.01


# ---------------------------------------------------------------- RBF (IP6)
#
#   solve        |A^ c^ - [d; 0]| <= gamma_{4N} (|A^| + |L^| |D^| |L^|^T) |c^| componentwise (rbf_backward_bound;
#                Higham, Accuracy and Stability, Thm 11.3 for Bunch-Kaufman): A^ the computed system (N = n + q),
#                L^, D^ its sytrf factors (sd.linalg.ldl: the same factorization), the residual exact
#                (rbf_residual); the growth ||  |L^| |D^| |L^|^T ||_inf / ||A^||_inf goes with it (rbf_growth)
#   evaluation   |f^(x) - f(x)| <= sum_j |c_j| E_j (1 + gamma_N) + gamma_N sum_j |c_j| |phi_j|
#                + sum_k |b_k| |m_k| (gamma_{3 g_k} (1 + gamma_N) + gamma_N) (rbf_eval_bound): f the function of the
#                computed coefficients (c, b), shift and scale exactly, phi_j = phi(epsilon ||x - y_j||) exact, m_k the
#                exact monomials of degree g_k, E_j = 1.01 gamma_{m+5} |r phi'(r)| + gamma_4 |phi_j| + 8 2^-1074 the
#                kernel's rounding (r to gamma_{m+5} relatively: the distance kernel and the product with epsilon;
#                libm's log and exp within 1 ULP, the docs/design/spatial.md assumption L)

RBF_KERNELS = ("linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric",
               "inverse_quadratic", "gaussian")  # fmt: skip
RBF_SCALE_INVARIANT = ("linear", "thin_plate_spline", "cubic", "quintic")
RBF_MIN_DEGREE = {"multiquadric": 0, "linear": 0, "thin_plate_spline": 1, "cubic": 1, "quintic": 2}
RBF_EPSILON = 3.0  # tools/interpolate_ref.py's epsilon for the kernels that need one


def rbf_options(method):
    """(kernel, epsilon, smoothing, neighbors) of a method of the rbf family."""
    kernel = method.split("_s0.1")[0].split("_k8")[0]
    eps = 1.0 if kernel in RBF_SCALE_INVARIANT else RBF_EPSILON
    return kernel, eps, (0.1 if method.endswith("_s0.1") else 0.0), (8 if method.endswith("_k8") else None)


def exact_dot(a, b):
    """sum a_i b_i correctly rounded (the products split exactly by fma, then fsum)."""
    terms = []
    for x, y in zip(a, b):
        p = x * y
        terms += [p, math.fma(x, y, -p)]
    return math.fsum(terms)


def rbf_residual(A, c, d):
    """|A c - [d; 0]| exactly rounded, (N, S): A (N, N), c (N, S), d (n, S) float64 numpy arrays."""
    import numpy as np

    N, S = c.shape
    rhs = np.zeros((N, S))
    rhs[: d.shape[0]] = d
    out = np.empty((N, S))
    for i in range(N):
        for s in range(S):
            out[i, s] = abs(exact_dot(list(A[i]) + [-1.0], list(c[:, s]) + [rhs[i, s]]))
    return out


def rbf_backward_bound(A, c, L, D):
    """gamma_{4N} (|A| + |L| |D| |L|^T) |c|, (N, S), with 1% for the float64 products."""
    import numpy as np

    N = A.shape[0]
    M = np.abs(A) + np.abs(L) @ np.abs(D) @ np.abs(L).T
    return 1.01 * gamma(4 * N) * (M @ np.abs(c))


def rbf_growth(A, L, D):
    import numpy as np

    M = np.abs(L) @ np.abs(D) @ np.abs(L).T
    return float(np.max(M.sum(axis=1)) / np.max(np.abs(A).sum(axis=1)))


def kernel_slope(kernel, r):
    """An envelope of |r phi'(r)| (thin_plate_spline: r^2 (2 |log r| + 1), so no zero of phi' is trusted)."""
    if kernel == "linear":
        return r
    if kernel == "thin_plate_spline":
        return 0.0 if r == 0 else r * r * (2 * abs(math.log(r)) + 1)
    if kernel == "cubic":
        return 3 * r**3
    if kernel == "quintic":
        return 5 * r**5
    if kernel == "multiquadric":
        return r * r / math.sqrt(1 + r * r)
    if kernel == "inverse_multiquadric":
        return r * r / (1 + r * r) ** 1.5
    if kernel == "inverse_quadratic":
        return 2 * r * r / (1 + r * r) ** 2
    return 2 * r * r * math.exp(-r * r)


def kernel_arb(kernel, r):
    """phi(r) for an arb r."""
    from flint import arb

    if kernel == "linear":
        return -r
    if kernel == "thin_plate_spline":
        return arb(0) if r == 0 else r * r * r.log()
    if kernel == "cubic":
        return r * r * r
    if kernel == "quintic":
        return -(r**5)
    if kernel == "multiquadric":
        return -((1 + r * r).sqrt())
    if kernel == "inverse_multiquadric":
        return 1 / (1 + r * r).sqrt()
    if kernel == "inverse_quadratic":
        return 1 / (1 + r * r)
    return (-(r * r)).exp()


def rbf_exact(kernel, eps, y, x, coef, powers, shift, scale, prec=256):
    """(f(x) as an arb, [phi_j] and [m_k] as floats, [r_j] as floats) for one column of coefficients coef (N,): the
    function of the computed coefficients evaluated exactly (Arb at `prec` bits, the data exact rationals)."""
    from fractions import Fraction

    from flint import arb, ctx

    def a(q):
        q = Fraction(q)
        return arb(q.numerator) / arb(q.denominator)

    old = ctx.prec
    ctx.prec = prec
    try:
        n, m = len(y), len(x)
        f = arb(0)
        phis, rs = [], []
        e = a(eps)
        for j in range(n):
            s = sum(((Fraction(float(x[l])) - Fraction(float(y[j][l]))) ** 2 for l in range(m)), Fraction(0))
            r = e * a(s).sqrt() if s else arb(0)
            p = kernel_arb(kernel, r)
            f += a(float(coef[j])) * p
            phis.append(float(p.mid()))
            rs.append(float(r.mid()))
        monos = []
        xh = [(Fraction(float(x[l])) - Fraction(float(shift[l]))) / Fraction(float(scale[l])) for l in range(m)]
        for k, pw in enumerate(powers):
            v = Fraction(1)
            for l in range(m):
                v *= xh[l] ** int(pw[l])
            f += a(float(coef[n + k])) * a(v)
            monos.append(float(v))
        return f, phis, monos, rs
    finally:
        ctx.prec = old


def rbf_eval_bound(kernel, m, coef, phis, monos, rs, powers):
    """The evaluation contract's bound for one point and one column (coef (N,), the exact phi_j, m_k, r_j)."""
    n = len(phis)
    N = n + len(monos)
    gN = gamma(N)
    out = 0.0
    for j in range(n):
        c = abs(float(coef[j]))
        E = 1.01 * gamma(m + 5) * kernel_slope(kernel, rs[j]) + gamma(4) * abs(phis[j]) + 8 * 2.0**-1074
        out += c * E * (1 + gN) + gN * c * abs(phis[j])
    for k, mk in enumerate(monos):
        g = int(sum(int(p) for p in powers[k]))
        out += abs(float(coef[n + k])) * abs(mk) * (gamma(3 * g) * (1 + gN) + gN) + 2.0**-1074
    return 1.01 * out


# ---------------------------------------------------------------- grids (IP5)


def grid_linear_exact(points, values, x):
    """RegularGridInterpolator's linear value at x exactly (a Fraction) and the contract's bound for the computed one
    (docs/design/interpolate.md §4): the 1-D linear contract stacked axis by axis from the last to the first, each
    level |r^ - lerp(a^, b^)| <= linear_bound(|a^|, |b^|, t) plus (1 - t) and t times its inputs' bounds (|1 - t|, |t|
    outside). points: ascending float arrays; values: one column (n_0, ..., n_{d-1}); x: the point, not NaN."""
    from fractions import Fraction

    import numpy as np

    d = len(points)
    level = [(Fraction(float(v)), 0.0) for v in np.asarray(values, float).reshape(-1)]
    shape = [len(p) for p in points]
    for a in reversed(range(d)):
        p = [Fraction(float(v)) for v in points[a]]
        n = shape[a]
        rest = len(level) // n
        if n == 1:
            level = [level[r * n] for r in range(rest)]
            continue
        X = Fraction(float(x[a]))
        j = min(max(sum(1 for v in p if v <= X) - 1, 0), n - 2)
        t = (X - p[j]) / (p[j + 1] - p[j])
        tf = float(t)
        new = []
        for r in range(rest):
            (ea, ba), (eb, bb) = level[r * n + j], level[r * n + j + 1]
            ma, mb = abs(float(ea)) + ba, abs(float(eb)) + bb
            b = linear_bound(ma, mb, tf) + abs(1 - tf) * ba + abs(tf) * bb
            new.append((ea + t * (eb - ea), b * (1 + 4 * U)))
        level = new
    return level[0]


def _tensor_axis_basis(t, k, x, nu, exact):
    """(the first coefficient's index l - k, the k + 1 basis values) of one axis at x: Fractions when exact, else the
    recursion on absolute values in floats (B~)."""
    from fractions import Fraction

    if exact:
        T = [Fraction(float(v)) for v in t]
        X = Fraction(float(x))
        l = knot_interval(T, k, X)
        return l - k, _basis(T, k, l, X, nu, Fraction(0))
    t = [float(v) for v in t]
    l = knot_interval(t, k, float(x))
    return l - k, _basis(t, k, l, float(x), nu, 0.0, absolute=True)


def _tensor_terms(knots, k, x, nu, exact):
    """[(the coefficient's index tuple, the product of the axes' basis values)] around x."""
    import itertools

    axes = [_tensor_axis_basis(t, k, x[a], nu[a], exact) for a, t in enumerate(knots)]
    out = []
    for rs in itertools.product(range(k + 1), repeat=len(knots)):
        prod = 1
        for (_, h), r in zip(axes, rs):
            prod = prod * h[r]
        out.append((tuple(base + r for (base, _), r in zip(axes, rs)), prod))
    return out


def tensor_exact(knots, c, k, x, nu=None):
    """sum_j c_j prod_a B_{j_a}^(nu_a)(x_a) exactly (a Fraction) for float knots, coefficients (n_0, ..., n_{d-1})
    and point."""
    from fractions import Fraction

    nu = nu or [0] * len(knots)
    if any(v > k for v in nu):
        return Fraction(0)
    return sum((Fraction(float(c[j])) * b for j, b in _tensor_terms(knots, k, x, nu, True)), Fraction(0))


def tensor_bound(knots, c, k, x, nu=None, c_bound=None):
    """The evaluation contract of the tensor spline at x for computed coefficients c (docs/design/interpolate.md §4):
    gamma_{d(6k+1)} sum_j |c_j| prod B~ + d (6k + 2) 2^-1074 (1 + 2^d sum_j |c_j|); with c_bound (a bound
    per coefficient) also sum_j c_bound_j prod B~ (end to end)."""
    d = len(knots)
    nu = nu or [0] * d
    if any(v > k for v in nu):
        return 0.0
    s = floor = carried = 0.0
    for j, b in _tensor_terms(knots, k, x, nu, False):
        s += abs(float(c[j])) * b
        if c_bound is not None:
            carried += float(c_bound[j]) * b
    for j, _ in _tensor_terms(knots, k, x, [0] * d, False):
        floor += abs(float(c[j]))
    out = gamma(d * (6 * k + 1)) * s + d * (6 * k + 2) * 2.0**-1074 * (1 + 2**d * floor)
    return 1.01 * (out + carried * (1 + gamma(d * (6 * k + 1))))


def _along(M, R, axis):
    """M (n, n) applied along `axis` of R."""
    import numpy as np

    return np.moveaxis(np.tensordot(M, np.moveaxis(R, axis, 0), axes=(1, 0)), 0, axis)


def tensor_coefficients(points, knots, k, values):
    """The exact tensor coefficients (an object array of Fractions): the exact collocation solve along every axis."""
    from fractions import Fraction

    import flint
    import numpy as np

    R = np.vectorize(lambda v: Fraction(float(v)), otypes=[object])(np.asarray(values, float))
    for a, (x, t) in enumerate(zip(points, knots)):
        T = [Fraction(float(v)) for v in t]
        n = len(x)
        A = flint.fmpq_mat(n, n)
        for i, v in enumerate(x):
            X = Fraction(float(v))
            l = knot_interval(T, k, X)
            for r, b in enumerate(_basis(T, k, l, X, 0, Fraction(0))):
                A[i, l - k + r] = flint.fmpq(b.numerator, b.denominator)
        inv = A.inv()
        Minv = np.array([[Fraction(int(inv[i, j].p), int(inv[i, j].q)) for j in range(n)] for i in range(n)], object)
        R = _along(Minv, R, a)
    return R


def tensor_collocation_bound(points, knots, k, values):
    """The construction contract of the tensor coefficients, one bound per coefficient: the 1-D contract
    2 gamma_{8k+8} |A_a^-1| |A_a| |R_a| of every axis's solve (axis 0 first; R_a the coefficients after it) carried
    through the later axes' |A_b^-1| (b > a)."""
    import numpy as np

    R = np.asarray(values, float)
    mats = [collocation_matrix(x, t, k, None) for x, t in zip(points, knots)]
    invs = [np.linalg.inv(A) for A in mats]
    steps = []
    for a, inv in enumerate(invs):
        R = _along(inv, R, a)
        steps.append(2 * gamma(8 * k + 8) * _along(np.abs(inv) @ np.abs(mats[a]), np.abs(R), a))
    total = np.zeros_like(R)
    for a, e in enumerate(steps):
        for b in range(a + 1, len(points)):
            e = _along(np.abs(invs[b]), e, b)
        total = total + e
    return 1.01 * total
