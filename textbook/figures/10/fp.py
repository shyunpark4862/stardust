"""10장 자료: 간격과 흔들림, 무작위 실수의 반올림 상대오차, 반올림 규칙의 치우침, 0.1+0.2 (정확한 유리수)."""
import math, random, sys
from fractions import Fraction as F
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
u = F(1, 2**53)
# 0.1 + 0.2
a, b = F(0.1), F(0.2)
print("0.1 =", a, float(a.numerator) / a.denominator)
s = a + b; lo = F(0.30000000000000004) - F(0.3)
print("exact 0.1+0.2 =", s, "decimal", f"{s.numerator/s.denominator:.25f}")
print("fl(0.3) =", repr(F(0.3)), " next =", repr(F(0.30000000000000004)), " midpoint?", s == (F(0.3) + F(0.30000000000000004)) / 2)
# 1. 상대 간격 ulp(x)/x
with open(f"{OUT}/spacing.dat", "w") as o:
    o.write("x rel\n")
    for i in range(0, 801):
        x = 2 ** (-2 + i * 6 / 800)
        e = math.floor(math.log2(x))
        o.write(f"{x:.6f} {2.0**(e-52)/x:.6e}\n")
# 2. 무작위 실수의 반올림 상대오차 (실수는 F로 정확히, 2^-120 해상도)
random.seed(4)
with open(f"{OUT}/relerr.dat", "w") as o:
    o.write("x rel\n")
    mx = 0
    for _ in range(3000):
        x = F(random.getrandbits(120), 2**120) * 8 + F(1, 4)   # [1/4, 8.25)
        r = F(float(x))
        rel = abs(r - x) / x / u
        mx = max(mx, rel)
        o.write(f"{float(x):.6f} {float(rel):.5f}\n")
print("max rel/u", float(mx))
# 3. 치우침: 정수 반나눗셈 합에서 반올림 규칙별로 누적 (Knuth 예)
random.seed(7)
def rnd(x, mode):
    q, r = divmod(x, 1)
    if r != F(1, 2): return q + (1 if r > F(1, 2) else 0)
    if mode == "even": return q + (q % 2)
    return q + 1  # half up
with open(f"{OUT}/bias.dat", "w") as o:
    o.write("k even up\n")
    se = su = F(0); te = 0
    for k in range(1, 2001):
        x = F(random.randint(0, 200), 2)  # halves
        se += rnd(x, "even") - x; su += rnd(x, "up") - x
        if k % 10 == 0: o.write(f"{k} {float(se)} {float(su)}\n")
    print("bias after 2000: even", float(se), "half-up", float(su))
