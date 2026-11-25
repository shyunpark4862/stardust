"""Rounding accumulated by a long neutral recurrence: J_n(100) for n = 0..95 (n < x: neither solution dominates) from
exact J_0(100), J_1(100), (a) plain float64 c2 = (2n/x) c1 - c0, (b) the library's step fma(2n (1/x)_dd, c1, -c0)
(csrc/special/ops/bessel.hpp upward/coef), (c) double-double. Error / envelope sqrt(J^2 + Y^2) ~ sqrt(2/(pi x)),
in units of u. Output: longrec.dat. Also Airy Ai, Bi on [-10, 2.5] from stardust (airy.dat)."""
import math
import numpy as np
from fractions import Fraction as F
from dref import jv as jref, dec, D

U = 2.0 ** -53

def two_prod(a, b):
    p = a * b; e = float(F(a) * F(b) - F(p)); return p, e

def two_sum(a, b):
    s = a + b; bb = s - a; return s, (a - (s - bb)) + (b - bb)

def dd_add(a, b):
    s, e = two_sum(a[0], b[0]); e += a[1] + b[1]; return two_sum(s, e)

def dd_mul(a, b):
    p, e = two_prod(a[0], b[0]); e += a[0] * b[1] + a[1] * b[0]; return two_sum(p, e)

def main():
    x = 100.0; N = 96
    ex = [jref(n, x) for n in range(N + 1)]
    env = math.sqrt(2 / (math.pi * x))
    # (a) plain
    a = [float(ex[0]), float(ex[1])]
    for n in range(1, N): a.append(2 * n / x * a[n] - a[n - 1])
    # (b) library: 1/x as double-double, coef = fma(2n, xh, 2n xl), step fma(coef, c1, -c0)
    xr = F(1) / F(x); xh = float(xr); xl = float(xr - F(xh))
    def fma(p, q, r): return float(F(p) * F(q) + F(r))
    b = [float(ex[0]), float(ex[1])]
    for n in range(1, N):
        cf = fma(2.0 * n, xh, 2.0 * n * xl); b.append(fma(cf, b[n], -b[n - 1]))
    # (c) double-double
    def todd(d):
        h = float(d); return (h, float(d - D(repr(h))))
    c = [todd(ex[0]), todd(ex[1])]
    xd = (xh, xl)
    for n in range(1, N):
        cf = dd_mul((2.0 * n, 0.0), xd); t = dd_mul(cf, c[n]); c.append(dd_add(t, (-c[n - 1][0], -c[n - 1][1])))
    rows = []
    for n in range(N):
        e = lambda v: float(abs(dec(v) - ex[n])) / env / U
        rows.append((n, e(a[n]), e(b[n]), max(e(c[n][0]), 0.01)))
    with open("longrec.dat", "w") as f:
        f.write("n plain lib dd\n")
        for r in rows: f.write("%d %.4f %.4f %.4f\n" % r)
    for r in rows[::10]: print("n %2d plain %.1f lib %.1f dd %.2f (u of envelope)" % r)

def airy():
    import stardust as sd
    xs = np.linspace(-10, 2.5, 501)
    r = sd.special.airy(sd.asarray(xs))
    ai, aip, bi, bip = [np.asarray(v) for v in r]
    with open("airy.dat", "w") as f:
        f.write("x ai bi\n")
        for row in zip(xs, ai, bi): f.write("%.4f %.6e %.6e\n" % row)
    # Wronskian Ai Bi' - Ai' Bi = 1/pi
    w = (ai * bip - aip * bi) * math.pi - 1
    print("Airy Wronskian residual max %.2e on [-10, 2.5]" % abs(w).max())

if __name__ == "__main__":
    main(); airy()
