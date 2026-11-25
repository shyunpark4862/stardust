"""조건부 기댓값의 그림 자료: 상관 0.7인 이변량 정규 표본 400개(따로 빌드한 stardust)와
x를 구간으로 나눈 구간 평균(조건부 기댓값의 어림)."""
import sys, math, numpy as np, stardust as sd
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = sd.random.default_rng(9)
n, rho = 400, 0.7
z1 = np.array(sd.stats.Normal().sample((n,), rng=rng).tolist())
z2 = np.array(sd.stats.Normal().sample((n,), rng=rng).tolist())
x, y = z1, rho * z1 + math.sqrt(1 - rho**2) * z2
with open(f"{OUT}/condexp_pts.dat", "w") as o:
    o.write("x y\n")
    for a, b in zip(x, y): o.write(f"{a:.4f} {b:.4f}\n")
edges = np.linspace(-2.5, 2.5, 11)
with open(f"{OUT}/condexp_bins.dat", "w") as o:
    o.write("x y\n")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 5: o.write(f"{(a+b)/2:.3f} {y[m].mean():.4f}\n")
print("corr", np.corrcoef(x, y)[0, 1])
