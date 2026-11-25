"""이산 분포의 항등식 시험(따로 빌드한 stardust): icdf(cdf(k)) == k, iccdf(ccdf(k)) == k,
그리고 무작위 p에서 icdf의 후방 오차 계약(정확한 누적 확률은 Fraction/Decimal로)."""
import math, numpy as np, stardust as sd
from decimal import Decimal as D, getcontext
getcontext().prec = 50
def pois_cdf(mu, k):  # exact-ish Decimal
    mu = D(repr(mu)); s = D(0); t = (-mu).exp()
    for j in range(0, k + 1):
        if j: t = t * mu / j
        s += t
    return s
rng = np.random.default_rng(5)
for mu in (0.5, 3.0, 30.0, 300.0):
    P = sd.stats.Poisson(mu=mu)
    top = int(mu + 12 * math.sqrt(mu) + 20)
    ks = np.arange(0, top)
    c = np.array(P.cdf(sd.asarray(ks.astype(float))).tolist()); cc = np.array(P.ccdf(sd.asarray(ks.astype(float))).tolist())
    i1 = np.array(P.icdf(sd.asarray(c)).tolist()); i2 = np.array(P.iccdf(sd.asarray(cc)).tolist())
    d1 = np.r_[True, c[1:] != c[:-1]] & (c <= 0.5); d2 = np.r_[True, cc[1:] != cc[:-1]] & (cc <= 0.5) & (cc > 0.0)
    e1 = int(np.sum((i1 != ks) & (c > 0.5) & (c < 1.0)))
    ok1 = int(np.sum((i1 == ks) & d1)); ok2 = int(np.sum((i2 == ks) & d2)); n1 = int(d1.sum()); n2 = int(d2.sum())
    # backward error on random p
    ps = [float(v) for v in rng.uniform(0, 1, 300)]; bad = 0
    ex = [pois_cdf(mu, k) for k in range(top + 5)]
    for p in ps:
        k = int(float(P.icdf(float(p))))
        lo = D(repr(p)) - D("1e-14") * D(repr(min(p, 1 - p))); hi = D(repr(p)) + D("1e-14") * D(repr(min(p, 1 - p)))
        # k is acceptable if it is the exact quantile of some p~ in [lo, hi]: F(k-1) < hi and F(k) >= lo
        Fk = ex[k]; Fkm = ex[k - 1] if k > 0 else D(0)
        if not (Fkm < hi and Fk >= lo): bad += 1
    print(f"Poisson({mu}): icdf(cdf(k))==k {ok1}/{n1}, iccdf(ccdf(k))==k {ok2}/{n2} (small side, distinct doubles), backward-error violations {bad}/300")
