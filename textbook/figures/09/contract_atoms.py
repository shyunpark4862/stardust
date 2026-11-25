"""Backward-error contract (T25) stressed at the atoms: p = cdf(k), ccdf(k) and +-1..3 ulps,
checked with exact Poisson cdfs (Decimal, 60 digits)."""
import math, numpy as np, stardust as sd
from decimal import Decimal as D, getcontext
getcontext().prec = 60
def exact_cdfs(mu, top):
    m = D(float(mu)); t = (-m).exp(); s = D(0); out = []
    for j in range(top):
        if j: t = t * m / j
        s += t; out.append(s)
    return out
def ok(k, p, ex):   # some p~ within 1e-14 min(p, 1-p) of p has exact quantile k
    pd = D(float(p)); w = D("1e-14") * min(pd, 1 - pd)
    Fk = ex[k] if k < len(ex) else D(1); Fkm = ex[k-1] if k > 0 else D(0)
    return Fkm < pd + w and Fk >= pd - w
tot = bad = 0; worst = []
for mu in (0.0025, 0.5, 3.0, 30.0, 300.0):
    P = sd.stats.Poisson(mu=mu); top = int(mu + 30*math.sqrt(mu) + 40); ex = exact_cdfs(mu, top + 50)
    ks = np.arange(top, dtype=float)
    c = np.array(P.cdf(sd.asarray(ks)).tolist()); cc = np.array(P.ccdf(sd.asarray(ks)).tolist())
    ps = []
    for k in range(top):
        for base in (c[k], 1.0 - cc[k] if cc[k] < 0.5 else c[k]):
            v = base
            for j in range(4):
                for s in (v, ):
                    ps.append(s)
                v = np.nextafter(v, 2.0)
            v = base
            for j in range(3):
                v = np.nextafter(v, -1.0); ps.append(v)
    ps = [float(p) for p in ps if 0 < p < 1]
    r = np.array(P.icdf(sd.asarray(np.array(ps))).tolist())
    for p, k in zip(ps, r):
        tot += 1
        if not ok(int(k), p, ex): bad += 1; worst.append((mu, p, k))
print("icdf near atoms:", tot, "points, violations", bad, worst[:5])
print("Poisson(0.0025).icdf(1-7.8e-16) =", sd.stats.Poisson(mu=0.0025).icdf(1 - 7.8e-16), " cdf(4) == p:", sd.stats.Poisson(mu=0.0025).cdf(4.0) == 1 - 7.8e-16, " ccdf(4) =", sd.stats.Poisson(mu=0.0025).ccdf(4.0))
