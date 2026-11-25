"""따로 빌드한 stardust의 꼬리 확률·로그 확률·이산 분위수를 Decimal 참값과 견준다."""
import math, stardust as sd
from decimal import Decimal as D, getcontext
getcontext().prec = 120
PI = D("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
def erfc_tail(x):
    # 1 - Phi(x) for x > 0 by the continued fraction of Mills ratio (Laplace), run long enough
    x = D(x); t = D(0)
    for k in range(4000, 0, -1):
        t = k / (x + t)
    phi = (-(x * x) / 2).exp() / (2 * PI).sqrt()
    return phi / (x + t)
N = sd.stats.Normal()
for x in (3.0, 5.0, 8.0, 20.0, 37.0):
    ex = erfc_tail(x)
    c = float(N.ccdf(x)); one = 1.0 - float(N.cdf(x))
    print(f"x={x}: ccdf rel {float(abs(D(c)-ex)/ex):.1e}, 1-cdf rel {float(abs(D(one)-ex)/ex):.1e}")
for x in (40.0, 100.0):
    ex = erfc_tail(x).ln()
    lc = float(N.logccdf(x))
    print(f"logccdf({x}) = {lc!r}, exact {float(ex)!r}, rel {float(abs(D(lc)-ex)/abs(ex)):.1e}, ccdf = {float(N.ccdf(x))}")
P = sd.stats.Poisson(mu=3.0)
F2 = (-D(3)).exp() * D("8.5"); f = float(F2)
for p in (math.nextafter(f, 0), f, math.nextafter(f, 1), f * (1 + 1e-15)):
    print(f"Poisson(3).icdf({p!r}) = {float(P.icdf(p))}, exact quantile {'2' if D(p) <= F2 else '3'}")
print("Poisson cdf(2) =", repr(float(P.cdf(2.0))), "exact", F2)
