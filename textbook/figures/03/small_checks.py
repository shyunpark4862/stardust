"""Numbers quoted in Chapter 3 that are checked here: Stirling at n = 10, the Stirling series at
n = 10 with 0, 1, 3 corrections, tan(1) by its continued fraction (Lambert) with k layers, and the
convergents of sqrt(2). Exact rationals; reference values to 100 digits by fixed-point integers."""
from fractions import Fraction as F
import math

ONE = 1 << 400


def atan_inv(n):
    s, t, k, sg = 0, ONE // n, 1, 1
    while t:
        s += sg * (t // k); t //= n * n; k += 2; sg = -sg
    return s


PI = F(4 * (4 * atan_inv(5) - atan_inv(239)), ONE)


def ln_fixed(x):  # log of a positive Fraction to ~400 bits: atanh series of (x-1)/(x+1) after scaling by 2^k
    k = 0
    while x > 2: x /= 2; k += 1
    while x < 1: x *= 2; k -= 1
    y = (x - 1) / (x + 1)
    s, t, j = F(0), y, 1
    while abs(t) > F(1, 2**420):
        s += t / j; t *= y * y; j += 2
    L2 = F(0); y2 = F(1, 3); t = y2; j = 1
    while abs(t) > F(1, 2**420):
        L2 += t / j; t *= y2 * y2; j += 2
    return 2 * s + k * 2 * L2


n = 10
st = math.sqrt(2 * math.pi * n) * (n / math.e) ** n
print("n!/stirling - 1 at n=10:", math.factorial(10) / st - 1)
lnfact = ln_fixed(F(math.factorial(10)))
base = (F(21, 2)) * ln_fixed(F(10)) - 10 + ln_fixed(2 * PI) / 2
corr = [F(1, 12), F(-1, 360), F(1, 1260)]
for m in (0, 1, 3):
    approx = base + sum(c / F(10) ** (2 * j + 1) for j, c in enumerate(corr[:m]))
    print(f"log 10! Stirling series with {m} corrections: error {float(abs(approx - lnfact)):.2e}")
# tan(1): x / (1 - x^2/(3 - x^2/(5 - ...)))
def tan_cf(k):
    t = F(2 * k + 1)
    for j in range(k, 0, -1):
        t = (2 * j - 1) - F(1) / t
    return 1 / t
# reference tan 1 = sin 1 / cos 1 by their Taylor series (60 terms: error far below 2^-200)
sinv, cosv = F(0), F(0)
for j in range(0, 60):
    term = F((-1) ** j, math.factorial(2 * j + 1)); sinv += term
    cosv += F((-1) ** j, math.factorial(2 * j))
tan1 = sinv / cosv
for k in range(1, 10):
    print(f"tan(1) CF with {k} layers below the top: error {float(abs(tan_cf(k) - tan1)):.2e}")
# sqrt(2) convergents
A1, A0, B1, B0 = 1, 1, 0, 1
conv = []
for _ in range(6):
    A1, A0 = A0, 2 * A0 + A1
    B1, B0 = B0, 2 * B0 + B1
    conv.append(f"{A0}/{B0}")
print("sqrt2 convergents:", ["1/1"] + conv)
