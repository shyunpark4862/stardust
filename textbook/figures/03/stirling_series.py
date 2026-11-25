"""Stirling's series for log Gamma (Section 3.5) in exact rationals.

The correction sum_{j>=1} B_2j / (2j (2j-1) x^(2j-1)) cut after m terms, against the series cut at
its smallest term (for real x > 0 the error is below the first omitted term), for x in [7, 200]:
absolute truncation errors for m = 2..6 and for Cephes' fitted 5-coefficient polynomial A(1/x^2)/x
(stardust's lgamma_pos from x = 13; kLgamA). Writes stirling_err.dat."""
from fractions import Fraction as F
import math


def bernoulli(nmax):
    B = [F(1)]
    for n in range(1, nmax + 1):
        B.append(-sum(math.comb(n + 1, k) * B[k] for k in range(n)) / (n + 1))
    return B


Bn = bernoulli(160)
C = [Bn[2 * j] / (2 * j * (2 * j - 1)) for j in range(1, 80)]  # C[j-1] = B_2j / (2j(2j-1))
print("exact coefficients:", [float(c) for c in C[:6]])
KLGAMA = [8.11614167470508450300E-4, -5.95061904284301438324E-4, 7.93650340457716943945E-4,
          -2.77777777730099687205E-3, 8.33333333333331927722E-2]


def series(x, m):
    return sum(C[j] / x ** (2 * j + 1) for j in range(m))


def reference(x):
    s, j = F(0), 0
    while True:
        t = C[j] / x ** (2 * j + 1)
        nxt = C[j + 1] / x ** (2 * j + 3)
        s += t
        j += 1
        if abs(nxt) >= abs(t) or abs(nxt) < F(1, 2**150):
            return s


def cephes(x):  # exact rational evaluation of the fitted polynomial (truncation/fit error only)
    w = 1 / x
    p = F(0)
    for c in KLGAMA:
        p = p * w * w + F(c)
    return p * w


with open("stirling_err.dat", "w") as fh:
    fh.write("x m2 m3 m4 m5 m6 cephes ulp\n")
    for i in range(0, 120):
        xd = 7 * (200 / 7) ** (i / 119)
        x = F(xd)
        r = reference(x)
        row = [f"{xd:.5g}"] + [f"{float(abs(series(x, m) - r)):.4e}" for m in (2, 3, 4, 5, 6)]
        row.append(f"{float(abs(cephes(x) - r)):.4e}")
        row.append(f"{math.ulp(math.lgamma(xd)):.4e}")  # one ULP of log Gamma(x): the scale to compare with
        fh.write(" ".join(row) + "\n")
for xd in (7.0, 13.0, 30.0):
    x = F(xd)
    r = reference(x)
    print(xd, "m=5 %.2e" % float(abs(series(x, 5) - r)), "cephes %.2e" % float(abs(cephes(x) - r)),
          "ulp(lgamma) %.2e" % math.ulp(math.lgamma(xd)))
