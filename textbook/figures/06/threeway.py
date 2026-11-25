"""Rounding in long Bessel recurrences, three arithmetics (x = 100, mu = 0.3, 400 steps):
  plain  c = fl(2v/x), y2 = fl(fl(c y1) - y0)                 (two roundings a step, c rounded)
  fast   c = fma(2v, (1/x)_hi, 2v (1/x)_lo), y2 = fma(c, y1, -y0)  (stardust's float64 path, nu < 500)
  dd     double-double throughout (stardust's general path, jy_temme)
J downward from exact J_400.3, J_401.3 to order 0.3; Y upward from exact Y_0.3, Y_1.3 to 400.3.
Reference: the same recurrences in 90-digit Decimal from exact seeds (Decimal series; Y by
(J_v cos v pi - J_-v) / sin v pi). Error relative to max(|y_v|, |y_v'|) of the pair, in u. Output: threeway.dat."""
import math
from fractions import Fraction as F
from decimal import Decimal as D
from dref import jv as jref, dec, PI, _sin
from longrec import dd_add, dd_mul

def cos_pi(v):
    return _sin(PI * (dec(v) + D("0.5")))

def yref(v, x):
    s = _sin(PI * dec(v)); c = cos_pi(v)
    return (jref(v, x) * c - jref(-dec(v), x)) / s

def fma(a, b, c): return float(F(a) * F(b) + F(c))

def run(x=100.0, mu=0.3, n=400):
    nu = mu + n  # exact double orders nu - k
    xr = F(1) / F(x); xh = float(xr); xl = float(xr - F(xh))
    X = dec(x)
    # references
    Jtop, Jtop1 = jref(dec(nu), x), jref(dec(nu) + 1, x)
    Jr = {n: Jtop, n + 1: Jtop1}
    for k in range(n, 0, -1):
        v = dec(nu) - (n - k)
        Jr[k - 1] = 2 * v / X * Jr[k] - Jr[k + 1]
    Y0, Y1 = yref(mu, x), yref(mu + 1, x)
    Yr = {0: Y0, 1: Y1}
    for k in range(1, n):
        v = dec(mu) + k if False else dec(nu) - (n - k)
        Yr[k + 1] = 2 * v / X * Yr[k] - Yr[k - 1]
    def todd(d):
        h = float(d); return (h, float(d - D(repr(h))))
    out = {}
    # J downward
    for mode in ("plain", "fast", "dd"):
        a0, a1 = (todd(Jtop), todd(Jtop1)) if mode == "dd" else (float(Jtop), float(Jtop1))
        errs = {}
        for k in range(n, 0, -1):
            v = nu - (n - k)
            if mode == "plain":
                c = 2 * v / x; a0, a1 = c * a0 - a1, a0
            elif mode == "fast":
                c = fma(2 * v, xh, 2 * v * xl); a0, a1 = fma(c, a0, -a1), a0
            else:
                c = dd_mul((2 * v, 0.0), (xh, xl)); t = dd_mul(c, a0); a0, a1 = dd_add(t, (-a1[0], -a1[1])), a0
            val = a0[0] if mode == "dd" else a0
            sc = max(abs(Jr[k - 1]), abs(Jr[k]))
            errs[k - 1] = float(abs(dec(val) - Jr[k - 1]) / sc) / 2 ** -53
        out["J_" + mode] = errs
    for mode in ("plain", "fast", "dd"):
        b0, b1 = (todd(Y0), todd(Y1)) if mode == "dd" else (float(Y0), float(Y1))
        errs = {}
        for k in range(1, n):
            v = nu - (n - k)
            if mode == "plain":
                c = 2 * v / x; b0, b1 = b1, c * b1 - b0
            elif mode == "fast":
                c = fma(2 * v, xh, 2 * v * xl); b0, b1 = b1, fma(c, b1, -b0)
            else:
                c = dd_mul((2 * v, 0.0), (xh, xl)); t = dd_mul(c, b1); b0, b1 = b1, dd_add(t, (-b0[0], -b0[1]))
            val = b1[0] if mode == "dd" else b1
            sc = max(abs(Yr[k + 1]), abs(Yr[k]))
            errs[k + 1] = float(abs(dec(val) - Yr[k + 1]) / sc) / 2 ** -53
        out["Y_" + mode] = errs
    with open("threeway.dat", "w") as f:
        f.write("k Jplain Jfast Jdd Yplain Yfast Ydd\n")
        for k in range(2, n):
            f.write("%d %s\n" % (k, " ".join("%.4f" % max(out[m][k], 0.01) for m in ("J_plain", "J_fast", "J_dd", "Y_plain", "Y_fast", "Y_dd"))))
    for m in out: print(m, "max %.1f u" % max(out[m].values()), "at the end %.1f" % (out[m][0] if m.startswith("J") else out[m][n]))

if __name__ == "__main__":
    run()
