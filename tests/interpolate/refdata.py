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
