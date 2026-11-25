"""꼬리 확률의 수치 문제: Poisson(3)의 F(2)의 binary64 반올림, 1-Phi(x)의 소거(Decimal 참값)."""
import math
from decimal import Decimal as D, getcontext
from fractions import Fraction as Fr
getcontext().prec = 80
# Poisson(3), F(2) = e^-3 (1 + 3 + 9/2)
F2 = (-D(3)).exp() * D("8.5")
f = float(F2)
print("F(2) =", F2, " nearest double", repr(f), " double - exact =", D(f) - F2)
# 1 - Phi(x) by erfc in Decimal: use the continued fraction / series. erfc(z) = 1 - erf(z), erf series with high precision
def erf(z):
    z = D(z); s = D(0); term = z; k = 0
    while True:
        add = term / (2 * k + 1)
        s += add
        if abs(add) < D(10) ** -75: break
        k += 1
        term = -term * z * z / k
    return 2 / D(math.pi).__class__(str(math.pi)) * s if False else s
PI = D("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
def Phi(x):
    z = D(x) / D(2).sqrt()
    return (1 + 2 / PI.sqrt() * erf(z)) / 2
for x in (3, 5, 8):
    getcontext().prec = 120
    tail = 1 - Phi(x)
    p = float(Phi(x))
    naive = 1.0 - p
    print(f"x={x}: Phi(x) double {p!r}, 1-Phi naive {naive:.16e}, exact tail {float(tail):.16e}, rel err {abs(D(naive)-tail)/tail:.2e}, 1/tail {float(1/tail):.2e}")
print("log tail at 40 via asymptotic -x^2/2 - log(x sqrt(2pi)) =", -800 - math.log(40 * math.sqrt(2 * math.pi)))
