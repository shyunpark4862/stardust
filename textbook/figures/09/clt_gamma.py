"""지수 분포 n개의 합(모양 n의 감마)을 표준화한 분포 함수와 정규 분포의 거리.
F_n은 Decimal로 정확히(Poisson 합), Phi는 math.erfc. 참값에 mpmath를 쓰지 않는다."""
import math, sys
from decimal import Decimal as D, getcontext
getcontext().prec = 60
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
def Fn(n, x):
    y = D(n) + D(x) * D(n).sqrt()
    if y <= 0: return 0.0
    term = D(1); s = D(1)
    for k in range(1, n):
        term = term * y / k; s += term
    return float(1 - (-y).exp() * s)
Phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
phi = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
rho = 12 / math.e - 2
rows = []
for n in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000):
    # 최대 오차 위치를 촘촘한 격자로 찾고 주변을 다시 좁힌다
    best = (0, 0)
    for i in range(-400, 401):
        x = i / 100
        d = abs(Fn(n, x) - Phi(x))
        if d > best[0]: best = (d, x)
    lo = best[1] - 0.01
    for i in range(0, 201):
        x = lo + i * 1e-4
        d = abs(Fn(n, x) - Phi(x))
        if d > best[0]: best = (d, x)
    be = 0.4748 * rho / math.sqrt(n)
    edg = 2 * phi(0) / 6 / math.sqrt(n)
    rows.append((n, best[0], best[1], be, edg))
    print(n, f"{best[0]:.5f} at x={best[1]:.3f}  sqrt(n)*err={best[0]*math.sqrt(n):.4f}  BE={be:.4f} Edgeworth={edg:.5f}", flush=True)
with open(f"{OUT}/clt_gamma.dat", "w") as o:
    o.write("n err be edg\n")
    for n, e, x, be, edg in rows: o.write(f"{n} {e:.6e} {be:.6e} {edg:.6e}\n")
