"""Decimal references (no mpmath): pi, cos, sin, exp, ln, atan2 to about 60 digits."""
from decimal import Decimal as D, getcontext
from fractions import Fraction
getcontext().prec = 70

def _pi():
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
    def at_inv(x):
        x = D(x); x2 = x * x; t = 1 / x; s = t; k = 1; sgn = -1
        while True:
            t = t / x2; term = t / (2 * k + 1)
            if term < D(10) ** -75: break
            s += sgn * term; sgn = -sgn; k += 1
        return s
    return 16 * at_inv(5) - 4 * at_inv(239)
PI = _pi()

def dec(x):
    return D(Fraction(x).numerator) / D(Fraction(x).denominator) if not isinstance(x, D) else x

def cos_sin(a):
    """cos, sin of a Decimal angle (reduced mod 2 pi)."""
    a = a % (2 * PI)
    if a > PI: a -= 2 * PI
    x2 = a * a; c = D(1); s = a; tc = D(1); ts = a; k = 1
    while True:
        tc = -tc * x2 / ((2 * k - 1) * (2 * k)); ts = -ts * x2 / ((2 * k) * (2 * k + 1))
        c += tc; s += ts; k += 1
        if abs(tc) < D(10) ** -75 and abs(ts) < D(10) ** -75: break
    return c, s

def atan(x):
    # reduce: atan x = 2 atan(x / (1 + sqrt(1 + x^2))) until small
    x = D(x); m = 1
    while abs(x) > D("0.1"):
        x = x / (1 + (1 + x * x).sqrt()); m *= 2
    x2 = x * x; t = x; s = x; k = 1
    while True:
        t = -t * x2; term = t / (2 * k + 1); s += term; k += 1
        if abs(term) < D(10) ** -75: break
    return m * s

def atan2(y, x):
    y, x = D(y), D(x)
    if x > 0: return atan(y / x)
    if x < 0: return atan(y / x) + (PI if y >= 0 else -PI)
    return PI / 2 if y > 0 else -PI / 2

def ulp(x):
    import math
    return math.ulp(abs(x)) if x != 0 else math.ulp(0.0)
