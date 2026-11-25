"""Decimal references (no mpmath): Gamma for real arguments, Bessel J_nu(x) by its power series."""
from decimal import Decimal as D, getcontext
from fractions import Fraction as F
import math
getcontext().prec = 90

def _bern(n):
    A = [F(0)] * (n + 1); B = []
    for m in range(n + 1):
        A[m] = F(1, m + 1)
        for j in range(m, 0, -1): A[j - 1] = j * (A[j - 1] - A[j])
        B.append(A[0])
    return B
_B = _bern(60)

def _pi():
    def at_inv(x):
        x = D(x); x2 = x * x; t = 1 / x; s = t; k = 1; sg = -1
        while True:
            t = t / x2; term = t / (2 * k + 1)
            if term < D(10) ** -95: break
            s += sg * term; sg = -sg; k += 1
        return s
    return 16 * at_inv(5) - 4 * at_inv(239)
PI = _pi()

def dec(x):
    f = F(x); return D(f.numerator) / D(f.denominator)

def lgamma(x):
    """ln Gamma(x) for real x > 0 (Decimal)."""
    x = dec(x) if not isinstance(x, D) else x
    N = 60; w = x + N
    s = (w - D("0.5")) * w.ln() - w + (2 * PI).ln() / 2
    p = 1 / w; w2 = w * w
    for k in range(1, 28):
        c = _B[2 * k] / (2 * k * (2 * k - 1)); s += D(c.numerator) / D(c.denominator) * p; p /= w2
    prod = D(1)
    for j in range(N): prod *= (x + j)
    return s - prod.ln()

def gamma(x):
    x = dec(x) if not isinstance(x, D) else x
    if x > 0: return lgamma(x).exp()
    # reflection for negative non-integers
    import decimal
    s = _sin(PI * x)
    return PI / (s * gamma(1 - x))

def _sin(a):
    a = a % (2 * PI)
    x2 = a * a; s = a; t = a; k = 1
    while True:
        t = -t * x2 / ((2 * k) * (2 * k + 1)); s += t; k += 1
        if abs(t) < D(10) ** -95: break
    return s

def jv(nu, x):
    nu = dec(nu); x = dec(x)
    h = x / 2; q = -h * h
    g = gamma(nu + 1)
    t = h ** nu / g if nu != int(nu) or nu >= 0 else None
    s = D(0); k = 0
    while True:
        s += t; k += 1
        t = t * q / (k * (nu + k))
        if abs(t) < D(10) ** -80 * (abs(s) + D(10) ** -300) and k > 5: break
    return s
