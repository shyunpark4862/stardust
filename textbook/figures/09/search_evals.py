"""Predicate evaluations of disc::smallest (replicated in Python, same steps) from the two
starts: floor(mu) and the Cornish-Fisher start (normal_start), Poisson, uniform p."""
import math, numpy as np, stardust as sd
def smallest(lo, hi, guess, pred):
    cnt = [0]
    def P(k): cnt[0] += 1; return pred(k)
    g = math.floor(guess); g = max(g, lo); g = min(g, hi)
    if P(g):
        b = g; a = lo - 1; step = 1
        while b > lo:
            c = max(lo, b - step)
            if not P(c): a = c; break
            b = c; step *= 2
    else:
        a = g; b = hi; step = 1
        while True:
            c = a + step
            if c >= hi: break
            if P(c): b = c; break
            a = c; step *= 2
    while b - a > 1:
        m = math.floor(a + 0.5 * (b - a))
        if P(m): b = m
        else: a = m
    return b, cnt[0]
rng = np.random.default_rng(4)
nd = sd.special.ndtri
for mu in (10.0, 40.0, 63.9, 64.1, 1e4):
    Pd = sd.stats.Poisson(mu=mu); top = int(mu + 40*math.sqrt(mu) + 50)
    ks = np.arange(top, dtype=float); c = np.array(Pd.cdf(sd.asarray(ks)).tolist()); cc = np.array(Pd.ccdf(sd.asarray(ks)).tolist())
    sdv = math.sqrt(mu); ev = {"floor(mu)": [], "Cornish-Fisher": []}
    for p in rng.random(2000):
        pred = (lambda k: cc[int(k)] <= 1 - p) if p > 0.5 else (lambda k: c[int(k)] >= p)
        z = float(nd(float(p)))
        for name, g in (("floor(mu)", math.floor(mu)), ("Cornish-Fisher", math.floor(mu + sdv*(z + (1/sdv)*(z*z-1)/6)))):
            k, n = smallest(0, math.inf, g, pred); ev[name].append(n)
    print(f"mu={mu:g}: evaluations mean  floor(mu) {np.mean(ev['floor(mu)']):.1f}  Cornish-Fisher {np.mean(ev['Cornish-Fisher']):.1f}")
