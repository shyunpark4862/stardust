"""Euler-Maclaurin for zeta(s, q) = sum_{k>=0} (k + q)^-s (Section 3.4), in exact rationals.

1. zeta(2): the first N - 1 terms summed directly, the tail from k = N by the integral, the
   endpoint term and m Bernoulli corrections; the error against pi^2/6 (pi to 300 bits, Machin)
   for N = 5, 10, 20 and m = 0..40 (Figure). The corrections first shrink, then grow (B_2j grows
   like (2j)!).
2. A double-precision version of stardust's euler_maclaurin (Cephes zeta.c: direct sum to
   N >= 9, a = q + N >= 9, then 12 corrections with the coefficients (2j)!/B_2j) against an exact
   rational reference (integer s: every term rational; 60 direct terms and corrections to 1e-60)
   at s = 2, 3, 5, 10 and q in (0, 100]: maximum error in ULP (Table).
Writes em_zeta2.dat."""
from fractions import Fraction as F
import math

BITS = 300
ONE = 1 << BITS


def arctan_inv(n):
    s, t, k, sign = 0, ONE // n, 1, 1
    while t:
        s += sign * (t // k)
        t //= n * n
        k += 2
        sign = -sign
    return s


PI = F(4 * (4 * arctan_inv(5) - arctan_inv(239)), ONE)


def bernoulli(nmax):
    B = [F(1)]
    for n in range(1, nmax + 1):
        B.append(-sum(math.comb(n + 1, k) * B[k] for k in range(n)) / (n + 1))
    return B


Bn = bernoulli(90)


def em_tail(s, a, m):
    """sum_{k>=0} (a + k)^-s ~ integral + f(a)/2 - sum_j B_2j/(2j)! f^(2j-1)(a), f(x) = x^-s."""
    t = a ** (1 - s) / (s - 1) + F(1, 2) * a ** (-s)
    # f^(2j-1)(a) = -s(s+1)...(s+2j-2) a^(-s-2j+1)
    for j in range(1, m + 1):
        rising = math.prod(range(s, s + 2 * j - 1))
        t += Bn[2 * j] / math.factorial(2 * j) * rising * a ** (-s - 2 * j + 1)
    return t


zeta2 = PI**2 / 6
with open("em_zeta2.dat", "w") as fh:
    fh.write("m N5 N10 N20\n")
    for m in range(0, 41):
        row = [m]
        for N in (5, 10, 20):
            approx = sum(F(1, k * k) for k in range(1, N)) + em_tail(2, F(N), m)
            row.append(f"{float(abs(approx - zeta2)):.4e}")
        fh.write(" ".join(map(str, row)) + "\n")
for N in (5, 10, 20):
    errs = [abs(sum(F(1, k * k) for k in range(1, N)) + em_tail(2, F(N), m) - zeta2) for m in range(41)]
    mbest = min(range(41), key=lambda m: errs[m])
    print(f"N={N}: m=0 {float(errs[0]):.2e}, m=1 {float(errs[1]):.2e}, m=3 {float(errs[3]):.2e}, m=5 {float(errs[5]):.2e}; best m={mbest} {float(errs[mbest]):.2e}")

# ---- 2. the Cephes-style double algorithm
A = [12.0, -720.0, 30240.0, -1209600.0, 47900160.0, -1.8924375803183791606e9, 7.47242496e10,
     -2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15,
     1.8152105401943546773e17, -7.1661652561756670113e18]
for j, a in enumerate(A, 1):  # (2j)!/B_2j
    assert abs(a - float(math.factorial(2 * j) / Bn[2 * j])) <= 1e-15 * abs(a), j
EPS = 2.0**-53


def em_double(s, q):
    sm1 = s - 1.0
    total = q ** -s
    a, b, i = q, 0.0, 0
    while i < 9 or a <= 9.0:
        i += 1
        a += 1.0
        b = a ** -s
        total += b
        if abs(b / total) < EPS and abs(b * a / (sm1 * total)) < EPS:
            return total
    w = a
    total += b * w / sm1
    total -= 0.5 * b
    fac, k = 1.0, 0.0
    for j in range(12):
        fac *= s + k
        b /= w
        t = fac * b / A[j]
        total += t
        if abs(t / total) < EPS:
            break
        k += 1.0
        fac *= s + k
        b /= w
        k += 1.0
    return total


def ref(s, q):
    Q = F(q)
    return sum((Q + k) ** (-s) for k in range(60)) + em_tail(s, Q + 60, 30)


import random

random.seed(7)
for s in (2, 3, 5, 10):
    worst = 0.0
    for _ in range(300):
        q = random.choice([random.uniform(0.01, 1.0), random.uniform(1, 100)])
        r = ref(s, q)
        e = abs(float((F(em_double(float(s), q)) - r) / F(math.ulp(float(r)))))
        worst = max(worst, e)
    print(f"s={s}: max error over 300 q in (0.01, 100]: {worst:.2f} ULP")
