"""Forward (Lentz) against backward evaluation of a continued fraction in double precision
(Section 3.6): the incomplete gamma function's fraction (stardust's incgamma cf_upper)

    x + 1 - a + a_1/(b_1 + a_2/(b_2 + ...)),  a_n = -n(n - a),  b_n = x + 2n + 1 - a,

for 1000 samples with a in [0.5, 60], x in [a + 1, 3a + 40] and 700 with a in [50, 3000],
x in [a + 1, a + 3 sqrt(a)] (more terms) (Python floats are IEEE doubles; math.fma
for the fused steps). Methods: modified Lentz (special/cf.hpp lentz, tol 2^-53), the backward
pass over the same number of terms + 4 (cf_backward), and cf_fast's backward P/Q form. The
reference is the backward evaluation in exact rationals over 3 n + 60 terms, checked against
n + 40 more terms (difference < 2^-90 relative). Writes cf_errors.dat (one line per sample) and
prints the distribution of ULP errors. Seed 8."""
from fractions import Fraction as F
import math
import random

TINY = 2.0**-1000
TOL = 2.0**-53


def terms(a, x):
    return (lambda n: (-n * (n - a), x + 2 * n + 1 - a)), x + 1 - a


def lentz(b0, term, tol=TOL, nmax=100000):
    f = TINY if b0 == 0 else b0
    c, d = f, 0.0
    for n in range(1, nmax + 1):
        a, b = term(n)
        d = b + a * d
        if d == 0:
            d = TINY
        c = b + a / c
        if c == 0:
            c = TINY
        d = 1.0 / d
        delta = c * d
        f *= delta
        if abs(delta - 1.0) <= tol:
            return f, n
    return f, nmax


def backward(b0, term, n):
    t = 0.0
    for k in range(n, 0, -1):
        a, b = term(k)
        t = a / (b + t)
    return b0 + t


def backward_pq(b0, term, n):
    P, Q = 0.0, 1.0
    for k in range(n, 0, -1):
        a, b = term(k)
        Pn = a * Q
        Q = math.fma(b, Q, P)
        P = Pn
        m = abs(Q)
        if m > 2.0**512 or 0 < m < 2.0**-512:  # exact rescaling by a power of two, as cf_fast
            e = math.frexp(m)[1] - 1
            P, Q = math.ldexp(P, -e), math.ldexp(Q, -e)
    return b0 + P / Q


def exact(a, x, n):
    A, X = F(a), F(x)
    t = F(0)
    for k in range(n, 0, -1):
        t = (-k * (k - A)) / (X + 2 * k + 1 - A + t)
    return X + 1 - A + t


random.seed(8)
rows = []
samples = [(random.uniform(0.5, 60), None) for _ in range(1000)] + [(random.uniform(50, 3000), "near") for _ in range(700)]
for a, kind in samples:
    x = random.uniform(a + 1, a + 3 * math.sqrt(a)) if kind else random.uniform(a + 1, 3 * a + 40)
    term, b0 = terms(a, x)
    fl, n = lentz(b0, term)
    fb = backward(b0, term, n + 4)
    fq = backward_pq(b0, term, n + 4)
    r = exact(a, x, 3 * n + 60)
    r2 = exact(a, x, 3 * n + 100)
    assert abs((r - r2) / r) < F(1, 2**90)
    ulp = F(math.ulp(float(r)))
    rows.append((a, x, n, float((F(fl) - r) / ulp), float((F(fb) - r) / ulp), float((F(fq) - r) / ulp)))
with open("cf_errors.dat", "w") as fh:
    fh.write("a x n lentz backward pq\n")
    for r in rows:
        fh.write("%.6g %.6g %d %.4f %.4f %.4f\n" % r)
for name, i in (("lentz", 3), ("backward", 4), ("P/Q", 5)):
    e = sorted(abs(r[i]) for r in rows)
    print(f"{name:9s} median {e[len(e)//2]:.2f}  95% {e[int(0.95*len(e))]:.2f}  max {e[-1]:.2f} ULP")
ns = [r[2] for r in rows]
print("terms: min", min(ns), "median", sorted(ns)[len(ns)//2], "max", max(ns))
# correlation of Lentz error with the number of terms
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 10000)):
    sel = [abs(r[3]) for r in rows if lo <= r[2] < hi]
    if sel:
        selb = [abs(r[4]) for r in rows if lo <= r[2] < hi]
        print(f"n in [{lo},{hi}): {len(sel)} samples, Lentz max {max(sel):.2f}, mean {sum(sel)/len(sel):.2f}; backward max {max(selb):.2f}")
