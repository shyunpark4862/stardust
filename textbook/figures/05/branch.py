"""Branch cuts and the small parts of complex sqrt and log1p.
(1) signed zeros on the cuts: sd.sqrt, sd.log, sd.pow, sd.asin, sd.atanh, sd.special.loggamma at -x +- 0i.
(2) Im sqrt(1 + i y) relative error over y: sd.sqrt (the system's csqrt), Kahan's formula as special/cplx.hpp
    writes it (transcribed), the textbook formula sqrt((|z| - x)/2).
(3) Re log1p(z), z = t e^{i 3pi/4}... relative error over |z|: sd.log1p (numpy's hypot formula), cx::clog1p
    (transcribed: log1p(2a + a^2 + b^2)/2). Reference: Decimal.
Output: sqrt_small.dat, log1p_small.dat; the table to stdout."""
import math, cmath
import numpy as np
import stardust as sd
from dec import D, getcontext, atan2
from fractions import Fraction as F
getcontext().prec = 80

def dd(x): x = F(x); return D(x.numerator) / D(x.denominator)

def table():
    zs = [complex(-4, 0.0), complex(-4, -0.0)]
    def one(f, z): return complex(np.asarray(f(sd.asarray(np.array([z]))))[0])
    for name, f in [("sqrt", sd.sqrt), ("log", sd.log), ("arcsin", sd.asin), ("arctanh", sd.atanh)]:
        for z in ([complex(-4, 0.0), complex(-4, -0.0)] if name in ("sqrt", "log") else [complex(4, 0.0), complex(4, -0.0)] if name == "arcsin" else [complex(4, 0.0), complex(4, -0.0)]):
            print(name, z, one(f, z))
    for z in [complex(-2.5, 0.0), complex(-2.5, -0.0)]:
        print("loggamma", z, complex(np.asarray(sd.special.loggamma(sd.asarray(np.array([z]))))[0]))
    print("power (-8+0i)^(1/3)", complex(np.asarray(sd.pow(sd.asarray(np.array([complex(-8, 0.0), complex(-8, -0.0)])), sd.asarray(np.array([1 / 3 + 0j, 1 / 3 + 0j]))))[0]),
          complex(np.asarray(sd.pow(sd.asarray(np.array([complex(-8, 0.0), complex(-8, -0.0)])), sd.asarray(np.array([1 / 3 + 0j, 1 / 3 + 0j]))))[1]))

def kahan_sqrt(x, y):
    if x == 0 and y == 0: return 0.0, y
    t = math.sqrt(0.5 * (abs(x) + math.hypot(x, y)))
    if x >= 0: return t, y / (2 * t)
    return abs(y) / (2 * t), math.copysign(t, y)

def naive_sqrt(x, y):
    r = math.hypot(x, y)
    return math.sqrt((r + x) / 2), math.copysign(math.sqrt((r - x) / 2), y)

def ex_sqrt_im(x, y):
    X, Y = dd(x), dd(y); r = (X * X + Y * Y).sqrt()
    re = ((r + X) / 2).sqrt()
    return Y / (2 * re)

def sqrt_small():
    rows = []
    for y in np.geomspace(1e-300, 1.0, 151):
        ex = ex_sqrt_im(1.0, y)
        lib = complex(np.asarray(sd.sqrt(sd.asarray(np.array([complex(1.0, y)]))))[0]).imag
        k = kahan_sqrt(1.0, y)[1]; nv = naive_sqrt(1.0, y)[1]
        e = lambda v: max(float(abs(dd(v) - ex) / abs(ex)), 1e-18)
        rows.append((y, e(lib), e(k), e(nv)))
    with open("sqrt_small.dat", "w") as f:
        f.write("y lib kahan naive\n")
        for r in rows: f.write("%.4e %.4e %.4e %.4e\n" % r)
    a = np.array(rows)
    print("sqrt: lib max for y>=2^-511", a[a[:, 0] >= 2 ** -511, 1].max(), "below", a[a[:, 0] < 2 ** -511, 1].min(),
          "kahan max", a[:, 2].max(), "naive max for y<1e-9", a[a[:, 0] < 1e-9, 3].min())

def ex_log1p_re(a, b):
    A, B = dd(a), dd(b)
    w = 2 * A + A * A + B * B
    if abs(w) > D("1e-3"): return (1 + w).ln() / 2
    s, t, k = D(0), D(1), 1
    while True:  # log1p series
        t *= -w if k > 1 else w
        term = t / k if k > 1 else t
        if k == 1: t = w
        s += term if k == 1 else (t / k)
        if k > 1 and abs(t / k) < abs(s) * D(10) ** -40: break
        k += 1
    return s / 2

def cx_log1p_re(a, b):
    from fractions import Fraction
    v = float(F(a) * F(a + 2.0) + F(b * b))  # fma(a, a + 2, b*b)
    return 0.5 * math.log1p(v)

def log1p_small():
    rows = []
    th = 2.0  # angle 2 rad: Re z < 0, the real part of log(1+z) ~ -|z| 0.416
    for t in np.geomspace(1e-300, 0.5, 151):
        a, b = t * math.cos(th), t * math.sin(th)
        ex = ex_log1p_re(a, b)
        lib = complex(np.asarray(sd.log1p(sd.asarray(np.array([complex(a, b)]))))[0]).real
        cx = cx_log1p_re(a, b)
        e = lambda v: min(max(float(abs(dd(v) - ex) / abs(ex)), 1e-18), 10.0)
        rows.append((t, e(lib), e(cx)))
    with open("log1p_small.dat", "w") as f:
        f.write("t lib cx\n")
        for r in rows: f.write("%.4e %.4e %.4e\n" % r)
    a = np.array(rows)
    print("log1p: lib err at |z|=1e-8", a[np.argmin(abs(a[:, 0] - 1e-8)), 1], "cx max", a[:, 2].max())

if __name__ == "__main__":
    table(); sqrt_small(); log1p_small()
