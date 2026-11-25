"""Measurements of stardust itself (a separate clone of the public repository, built with
`uv pip install .`; run this file with that environment's python): erfcx for x >= 8, the binomial
pmf, and the Hurwitz zeta function, against the exact references of erfc_asym.py, binom_pmf.py and
euler_maclaurin.py. Also checks that the Python transcriptions used for the figures give the same
bits as the library."""
from fractions import Fraction as F
import math
import random
import stardust as sd


# ---- references (the same as in erfc_asym.py, binom_pmf.py, euler_maclaurin.py)
BITS = 300
ONE = 1 << BITS


def arctan_inv(n):
    s, t, k, sg = 0, ONE // n, 1, 1
    while t:
        s += sg * (t // k); t //= n * n; k += 2; sg = -sg
    return s


PI_FIX = 4 * (4 * arctan_inv(5) - arctan_inv(239))
INV_SQRT_PI = F(math.isqrt(ONE * ONE * ONE // PI_FIX), ONE)


def erfcx_ref(x):
    X = F(x); u = 1 / (2 * X * X)
    s, t, k = F(0), F(1), 0
    while True:
        s += t
        nt = -t * (2 * k + 1) * u
        if abs(nt) >= abs(t) or abs(t) < F(1, 2**120):
            break
        t, k = nt, k + 1
    return s * INV_SQRT_PI / X


def erfcx_transcribed(x):
    u = 0.5 / (x * x); s = 1.0
    for k in range(17, 0, -1):
        s = 1.0 - (2 * k - 1) * u * s
    return (0.56418958354775628695 / x) * s


def binom_exact(n, k, p):
    P = F(p)
    return math.comb(n, k) * P**k * (1 - P) ** (n - k)


def bernoulli(nmax):
    Bn = [F(1)]
    for n in range(1, nmax + 1):
        Bn.append(-sum(math.comb(n + 1, k) * Bn[k] for k in range(n)) / (n + 1))
    return Bn


BN = bernoulli(70)


def zeta_ref(s, q):
    Q = F(q); a = Q + 60
    t = sum((Q + k) ** (-s) for k in range(60)) + a ** (1 - s) / (s - 1) + F(1, 2) * a ** (-s)
    for j in range(1, 31):
        t += BN[2 * j] / math.factorial(2 * j) * math.prod(range(s, s + 2 * j - 1)) * a ** (-s - 2 * j + 1)
    return t


def ulp_err(v, r):
    return float((F(v) - r) / F(math.ulp(float(r))))


xs = [8 * 1.05**i for i in range(150)]
lib = sd.special.erfcx(sd.asarray(xs)).tolist()
same = sum(1 for x, v in zip(xs, lib) if v == erfcx_transcribed(x))
worst = max(abs(ulp_err(v, erfcx_ref(x))) for x, v in zip(xs, lib))
print(f"erfcx: library max error {worst:.3f} ULP over {len(xs)} points; bit-identical to the transcription at {same}/{len(xs)}")

p = 0.3
for n in [100, 1000, 10000, 100000]:
    sdv = math.sqrt(n * p * (1 - p))
    out = []
    for k in (round(n * p), round(n * p + 2 * sdv)):
        v = sd.stats.Binomial(n=n, p=p).pmf(k)
        v = float(v)
        r = binom_exact(n, k, p)
        out.append(abs(float((F(v) - r) / r)))
    print(f"binomial n={n}: library relative error at mean {out[0]:.2e}, at mean+2sd {out[1]:.2e}")

random.seed(7)
for s in (2, 3, 5, 10):
    worst = 0.0
    for _ in range(300):
        q = random.choice([random.uniform(0.01, 1.0), random.uniform(1, 100)])
        v = float(sd.special.zeta(sd.asarray([float(s)]), sd.asarray([q])).tolist()[0])
        worst = max(worst, abs(ulp_err(v, zeta_ref(s, q))))
    print(f"zeta(s={s}, q): library max error {worst:.2f} ULP")
