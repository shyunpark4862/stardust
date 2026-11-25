"""큰 수의 법칙과 그 실패: 따로 빌드한 stardust의 표본으로 누적 평균을 그린다(정규, 지수, Cauchy).
또 Glivenko--Cantelli: 정규 표본의 경험 분포 함수와 참 분포 함수의 최대 거리 D_n과 sqrt(n) D_n."""
import math, sys, numpy as np, stardust as sd
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = sd.random.default_rng(20261125)
n = 10**6
xs = {name: np.array(d.sample((n,), rng=rng).tolist()) for name, d in
      (("normal", sd.stats.Normal()), ("exp", sd.stats.Exponential()), ("cauchy", sd.stats.Cauchy()))}
idx = np.unique(np.round(np.logspace(0, 6, 400)).astype(int))
with open(f"{OUT}/lln.dat", "w") as o:
    o.write("n normal exp cauchy\n")
    cm = {k: np.cumsum(v) / np.arange(1, n + 1) for k, v in xs.items()}
    for k in idx:
        o.write(f"{k} {cm['normal'][k-1]:.6e} {cm['exp'][k-1]-1:.6e} {cm['cauchy'][k-1]:.6e}\n")
print("final means", {k: float(np.mean(v)) for k, v in xs.items()})
N = sd.stats.Normal()
with open(f"{OUT}/gc.dat", "w") as o:
    o.write("n D sqrtnD\n")
    for m in (10, 30, 100, 300, 1000, 3000, 10**4, 3*10**4, 10**5, 3*10**5, 10**6):
        x = np.sort(xs["normal"][:m])
        F = np.array(N.cdf(sd.asarray(x)).tolist())
        i = np.arange(1, m + 1)
        D = max(np.max(i / m - F), np.max(F - (i - 1) / m))
        o.write(f"{m} {D:.6e} {math.sqrt(m)*D:.6e}\n")
        print(m, D, math.sqrt(m) * D)
