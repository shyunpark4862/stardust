"""The asymptotic series of erfcx (Section 3.2), with exact rationals and fixed-point integers.

1. Term sizes t_k(x) = (2k-1)!! / (2x^2)^k for x = 2, 3, 4, 8 (Figure, Table): the smallest term
   and where it occurs; the number of terms below 2^-53 at x = 8.
2. The error of the double-precision nested evaluation used by stardust's erfcx for x >= 8
   (17 nested steps, then (1/sqrt(pi)) / x times the sum), against a reference with a certified
   error: the series cut at its smallest term or at the first term below 2^-120, in exact rationals
   (Theorem: the error is below the first omitted term, < 1e-27 relative for x >= 8), times 1/sqrt(pi) to 200 bits (Machin's pi, integer sqrt).
Writes erfc_terms.dat and erfcx_err.dat."""
from fractions import Fraction as F
import math

# ---- 1. term sizes
with open("erfc_terms.dat", "w") as f:
    f.write("k x2 x3 x4 x8\n")
    for k in range(0, 80):
        row = [k]
        for x in (2, 3, 4, 8):
            t = F(math.prod(range(1, 2 * k, 2)), (2 * x * x) ** k)
            row.append(f"{float(t):.6e}")
        f.write(" ".join(map(str, row)) + "\n")
for x in (2, 3, 4, 8):
    ts = [F(math.prod(range(1, 2 * k, 2)), (2 * x * x) ** k) for k in range(200)]
    kmin = min(range(200), key=lambda k: ts[k])
    first_below = next((k for k in range(200) if ts[k] < F(1, 2**53)), None)
    print(f"x={x}: smallest term k={kmin} size {float(ts[kmin]):.3e}; first term below 2^-53: k={first_below}")

# ---- 2. accuracy of the double evaluation
BITS = 220
ONE = 1 << BITS


def arctan_inv(n):  # arctan(1/n) in fixed point
    s, t, k, sign = 0, ONE // n, 1, 1
    n2 = n * n
    while t:
        s += sign * (t // k)
        t //= n2
        k += 2
        sign = -sign
    return s


PI = 4 * (4 * arctan_inv(5) - arctan_inv(239))
INV_SQRT_PI = math.isqrt(ONE * ONE * ONE // PI)  # (1/sqrt(pi)) * 2^BITS
assert abs(INV_SQRT_PI / ONE - 1 / math.sqrt(math.pi)) < 1e-15


TINY = F(1, 2**120)


def ref(x):  # x a double (exact rational), x >= 8
    X = F(x)
    u = 1 / (2 * X * X)
    s, t, k = F(0), F(1), 0
    while True:
        s += t
        nt = -t * (2 * k + 1) * u
        if abs(nt) >= abs(t) or abs(t) < TINY:  # stop at the smallest term or once below 2^-120
            break
        t, k = nt, k + 1
    return s * F(INV_SQRT_PI, ONE) / X


INV_SQRT_PI_D = 0.56418958354775628695  # the double of 1/sqrt(pi)


def stardust_like(x):
    u = 0.5 / (x * x)
    s = 1.0
    for k in range(17, 0, -1):
        s = 1.0 - (2 * k - 1) * u * s
    return (INV_SQRT_PI_D / x) * s


def ulp_err(v, r):
    return float((F(v) - r) / F(math.ulp(float(r))))


xs = [8 * 1.05**i for i in range(0, 150)]
worst = 0
with open("erfcx_err.dat", "w") as f:
    f.write("x ulp\n")
    for x in xs:
        e = ulp_err(stardust_like(x), ref(x))
        worst = max(worst, abs(e))
        f.write(f"{x:.6g} {e:.4f}\n")
print("max |error| over x in [8, 8*1.05^149] (ULP):", round(worst, 3))
