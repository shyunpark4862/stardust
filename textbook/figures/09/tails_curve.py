"""정규 위 꼬리를 x에 따라: ccdf와 1-cdf의 상대오차(따로 빌드한 stardust; 참값은 Mills 연분수의 십진 값)."""
import sys, stardust as sd
from decimal import Decimal as D, getcontext
getcontext().prec = 60
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
PI = D("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
def tail(x):
    x = D(x); s = D(0)
    for k in range(3000, 0, -1): s = k / (x + s)
    return (-(x * x) / 2).exp() / (2 * PI).sqrt() / (x + s)
N = sd.stats.Normal()
with open(f"{OUT}/tails_curve.dat", "w") as o:
    o.write("x ccdf naive\n")
    for i in range(5, 381, 5):
        x = i / 10; ex = tail(x)
        c = D(float(N.ccdf(x))); n = D(1.0 - float(N.cdf(x)))
        rc = max(float(abs(c - ex) / ex), 1e-18); rn = min(max(float(abs(n - ex) / ex), 1e-18), 1.0)
        o.write(f"{x} {rc:.4e} {rn:.4e}\n")
