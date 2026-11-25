"""Clenshaw's algorithm against Horner on the monomial form of the same polynomial.
p(t) = sum_{k<=40} c_k T_k(t), c_k = (+-)0.85^k (fixed random signs). Reference: exact rationals (Fraction).
Also stardust's i0e (Cephes chbevl on [0, 8]) against the Decimal series. Output: clenshaw.dat."""
import math
from fractions import Fraction as F
import numpy as np
from decimal import Decimal as D, getcontext

def cheb_to_mono(c):
    n = len(c); T = [[F(1)], [F(0), F(1)]]
    for k in range(2, n):
        a = [F(0)] + [2 * v for v in T[k - 1]]
        b = T[k - 2] + [F(0)] * (len(a) - len(T[k - 2]))
        T.append([x - y for x, y in zip(a, b)])
    m = [F(0)] * n
    for k in range(n):
        for i, v in enumerate(T[k]): m[i] += c[k] * v
    return m

def clenshaw(c, t):
    b1 = b2 = 0.0
    for k in range(len(c) - 1, 0, -1):
        b0 = 2 * t * b1 - b2 + c[k]; b2, b1 = b1, b0
    return c[0] + t * b1 - b2

def horner(m, t):
    s = 0.0
    for v in reversed(m): s = s * t + v
    return s

def exact(cF, tF):
    # exact Clenshaw in rationals
    b1 = b2 = F(0)
    for k in range(len(cF) - 1, 0, -1):
        b0 = 2 * tF * b1 - b2 + cF[k]; b2, b1 = b1, b0
    return cF[0] + tF * b1 - b2

def main():
    rng = np.random.default_rng(7)
    sg = rng.choice([-1, 1], 41)
    c = [float(s * 0.85 ** k) for k, s in enumerate(sg)]
    cF = [F(v) for v in c]
    mono = [float(v) for v in cheb_to_mono(cF)]
    print("largest monomial coefficient %.2e, sum |c_k| %.2f" % (max(abs(v) for v in mono), sum(abs(v) for v in c)))
    rows = []
    for t in np.linspace(-1, 1, 201):
        ex = exact(cF, F(float(t)))
        e1 = abs(F(clenshaw(c, float(t))) - ex); e2 = abs(F(horner(mono, float(t))) - ex)
        rows.append((t, max(float(e1), 1e-18), max(float(e2), 1e-18)))
    with open("clenshaw.dat", "w") as f:
        f.write("t clenshaw horner\n")
        for r in rows: f.write("%.4f %.4e %.4e\n" % r)
    a = np.array(rows)
    print("abs error: Clenshaw max %.2e (interior |t|<0.9: %.2e), Horner on monomials max %.2e" % (a[:, 1].max(), a[np.abs(a[:, 0]) < 0.9, 1].max(), a[:, 2].max()))

def i0e_check():
    import stardust as sd
    getcontext().prec = 60
    worst = {}
    for x in np.linspace(0.05, 30, 300):
        X = D(repr(float(x))); h = X / 2; t = D(1); s = D(0); k = 0
        while True:
            s += t; k += 1; t = t * h * h / (k * k)
            if t < D(10) ** -50 * s: break
        ex = s * (-X).exp()
        v = float(np.asarray(sd.special.i0e(sd.asarray(np.array([float(x)]))))[0])
        e = float(abs(D(repr(v)) - ex) / ex)
        key = "x<=8" if x <= 8 else "x>8"
        worst[key] = max(worst.get(key, 0), e)
    print("stardust i0e vs Decimal series: max rel", {k: "%.2e" % v for k, v in worst.items()})

if __name__ == "__main__":
    main(); i0e_check()
