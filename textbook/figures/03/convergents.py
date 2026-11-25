"""Errors of the convergents of sqrt(2) = [1; 2, 2, ...] and of Lambert's continued fraction for
tan(1) cut after n layers (Figure fig:convergents), in exact rationals. Writes convergents.dat."""
from fractions import Fraction as F
import math

ONE = 1 << 300
s2 = F(math.isqrt(2 * ONE * ONE), ONE)
tan1 = sum(F((-1) ** j, math.factorial(2 * j + 1)) for j in range(60)) / sum(F((-1) ** j, math.factorial(2 * j)) for j in range(60))


def tan_cf(k):  # 1/(1 - 1/(3 - ... - 1/(2k+1)))
    t = F(2 * k + 1)
    for j in range(k, 0, -1):
        t = (2 * j - 1) - F(1) / t
    return 1 / t


rows = []
A1, A0, B1, B0 = 1, 1, 0, 1
for n in range(12):
    if n:
        A1, A0 = A0, 2 * A0 + A1
        B1, B0 = B0, 2 * B0 + B1
    rows.append((n, float(abs(F(A0, B0) - s2)), float(abs((tan_cf(n) if n else F(1)) - tan1))))
with open("convergents.dat", "w") as f:
    f.write("n sqrt2 tan1\n" + "".join(f"{n} {a:.4e} {b:.4e}\n" for n, a, b in rows))
