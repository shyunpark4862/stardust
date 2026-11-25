"""The Hurwitz zeta direct sum's a += 1 (a = q + i) rounds when q is not an integer; the error of
a^-s is then about s times that of a. Corrected to first order with the exact low part of the sum
(two_sum: a + 1 = hi + lo exactly) as b (1 - s lo / a): maximum errors before and after, against
the exact reference of euler_maclaurin.py. For LIBRARY-NOTES.md (stardust's zeta.hpp)."""
from fractions import Fraction as F
import math
import random
import euler_maclaurin_ref as R  # zeta_ref, A, EPS


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def em(s, q, fix):
    sm1 = s - 1.0
    total = q ** -s
    a, lo, b, i = q, 0.0, 0.0, 0
    while i < 9 or a <= 9.0:
        i += 1
        a, e = two_sum(a, 1.0)
        lo += e
        b = a ** -s
        if fix:
            b *= 1.0 - s * lo / a
        total += b
        if abs(b / total) < R.EPS and abs(b * a / (sm1 * total)) < R.EPS:
            return total
    w = a
    total += b * w / sm1
    total -= 0.5 * b
    fac, k = 1.0, 0.0
    for j in range(12):
        fac *= s + k
        b /= w
        t = fac * b / R.A[j]
        total += t
        if abs(t / total) < R.EPS:
            break
        k += 1.0
        fac *= s + k
        b /= w
        k += 1.0
    return total


random.seed(7)
for s in (2, 3, 5, 10, 20):
    w0 = w1 = 0.0
    for _ in range(300):
        q = random.choice([random.uniform(0.01, 1.0), random.uniform(1, 100)])
        r = R.zeta_ref(s, q)
        u = F(math.ulp(float(r)))
        w0 = max(w0, abs(float((F(em(float(s), q, False)) - r) / u)))
        w1 = max(w1, abs(float((F(em(float(s), q, True)) - r) / u)))
    print(f"s={s}: now {w0:.2f} ULP, with the corrected a^-s {w1:.2f} ULP")
