"""Binomial pmf in double precision (Section 3.3): the naive exp of lgamma differences against
the saddle-point form of Loader (stardust's stats/dist/discrete_basic.hpp, here in plain float64
without its double-double terms), both against the exact rational value (and the saddle form with D = n p - k formed exactly, as stardust does) C(n,k) p^k (1-p)^(n-k)
for the double p (Python Fractions). Writes binom_pmf.dat (n, relative errors at k = round(np)
and k = round(np + 2 sd))."""
from fractions import Fraction as F
import math

# Stirling's correction delta(z) = log Gamma(z+1) - [(z+1/2) log z - z + log sqrt(2 pi)]
#   = 1/(12z) - 1/(360z^3) + 1/(1260z^5) - 1/(1680z^7) + 1/(1188z^9) - ...   (z >= 13)
B = [F(1, 12), F(-1, 360), F(1, 1260), F(-1, 1680), F(1, 1188)]
BD = [float(b) for b in B]


def delta(z):
    w = 1.0 / z
    w2 = w * w
    s = BD[4]
    for c in reversed(BD[:4]):
        s = s * w2 + c
    return s * w


def mu(t):  # t - log(1 + t)
    if abs(t) <= 0.125:  # t^2 (1/2 - t/3 + t^2/4 - ...), 24 terms
        s = 0.0
        for j in range(25, 1, -1):
            s = s * t + (1.0 / j if j % 2 == 0 else -1.0 / j)
        return t * t * s
    return t - math.log1p(t)


def saddle(n, k, p, exact_d=False):
    m = n - k
    D = float(F(n) * F(p) - k) if exact_d else n * p - k  # stardust forms D = n p - k exactly (two_prod)
    t1, t2 = D / k, -D / m
    E = k * mu(t1) + m * mu(t2)
    S = delta(n) - delta(k) - delta(m)
    return math.sqrt(n / (2 * math.pi * k * m)) * math.exp(S - E)


def naive(n, k, p):
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p))


def exact(n, k, p):
    P = F(p)
    return math.comb(n, k) * P**k * (1 - P) ** (n - k)


p = 0.3
rows = []
for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
    sd = math.sqrt(n * p * (1 - p))
    out = [n]
    for k in (round(n * p), round(n * p + 2 * sd)):
        r = exact(n, k, p)
        for v in (naive(n, k, p), saddle(n, k, p), saddle(n, k, p, True)):
            out.append(abs(float((F(v) - r) / r)))
    rows.append(out)
    print(n, ["%.2e" % v for v in out[1:]])
with open("binom_pmf.dat", "w") as fh:
    fh.write("n naive_mean saddle_mean saddleD_mean naive_tail saddle_tail saddleD_tail\n")
    for r in rows:
        fh.write(" ".join(str(v) for v in r) + "\n")
