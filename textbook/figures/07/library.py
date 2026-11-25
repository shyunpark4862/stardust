"""stardust's exact p-values (method='exact') against integer counts. No ties."""
import numpy as np
from fractions import Fraction as F
from math import comb, factorial, lcm
import stardust as sd
from exact import qbinom, subset_sums, mahonian, ks2_tail

rng = np.random.default_rng(11)

def rel(a, b): return abs(F(a) - b) / b

def mwu(c, d, shift=0.3, alt='less'):
    x = rng.normal(size=c) + shift; y = rng.normal(size=d)
    r = sd.htest.mannwhitneyu(sd.asarray(x), sd.asarray(y), method='exact', alternative=alt)
    U = sum(1 for a in x for b in y if a > b)
    P = qbinom(c + d, c); tot = comb(c + d, c)
    lo = F(sum(P[:U + 1]), tot); hi = F(sum(P[U:]), tot)
    p = float(r.pvalue)
    return min(rel(p, lo), rel(p, hi)), p, c, d

def wil(n, shift=0.2):
    d = rng.normal(size=n) + shift
    r = sd.htest.wilcoxon(sd.asarray(d), method='exact', alternative='less')
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    T = int(sum(rk for rk, v in zip(ranks, d) if v > 0))
    C = subset_sums(list(range(1, n + 1))); tot = 2 ** n
    lo = F(sum(C[:T + 1]), tot); hi = F(sum(C[T:]), tot)
    p = float(r.pvalue)
    return min(rel(p, lo), rel(p, hi)), p, n

def ken(n, rho=0.2):
    x = rng.normal(size=n); y = x * rho + rng.normal(size=n)
    r = sd.htest.kendalltau(sd.asarray(x), sd.asarray(y), method='exact', alternative='less')
    o = np.argsort(x); yy = y[o]
    K = sum(1 for i in range(n) for j in range(i + 1, n) if yy[i] > yy[j])
    M = mahonian(n); tot = factorial(n)
    lo = F(sum(M[:K + 1]), tot); hi = F(sum(M[K:]), tot)
    p = float(r.pvalue)
    return min(rel(p, lo), rel(p, hi)), p, n

def ks(m, n, shift=0.4):
    x = rng.normal(size=m) + shift; y = rng.normal(size=n)
    r = sd.htest.ks_2samp(sd.asarray(x), sd.asarray(y), method='exact')
    L = lcm(m, n); al, be = L // m, L // n
    # observed integer D*L
    z = sorted([(v, 0) for v in x] + [(v, 1) for v in y]); i = j = 0; h = 0
    for v, s in z:
        if s == 0: i += 1
        else: j += 1
        h = max(h, abs(i * al - j * be))
    cnt, tot = ks2_tail(m, n, h)
    ex = F(cnt, tot); p = float(r.pvalue)
    return rel(p, ex), p, m, n

if __name__ == "__main__":
    rows = []
    for c, d in ((5, 7), (10, 12), (20, 30), (30, 170), (60, 90)):
        e, p, *_ = mwu(c, d); rows.append(("mannwhitneyu", f"{c},{d}", p, e)); print("MWU", c, d, "p=%.3e rel %.1e" % (p, float(e)))
    for n in (10, 25, 60, 130):
        e, p, *_ = wil(n); rows.append(("wilcoxon", f"{n}", p, e)); print("Wilcoxon", n, "p=%.3e rel %.1e" % (p, float(e)))
    for n in (10, 30, 40, 80):
        e, p, *_ = ken(n); rows.append(("kendalltau", f"{n}", p, e)); print("Kendall", n, "p=%.3e rel %.1e" % (p, float(e)))
    for m, n in ((10, 10), (20, 30), (50, 70)):
        e, p, *_ = ks(m, n); rows.append(("ks_2samp", f"{m},{n}", p, e)); print("KS", m, n, "p=%.3e rel %.1e" % (p, float(e)))
    for c, d in ((20, 30), (60, 90)):
        e, p, *_ = mwu(c, d, 2.0, 'greater'); rows.append(("mannwhitneyu", f"{c},{d}*", p, e)); print("MWU tail", c, d, "p=%.3e rel %.1e" % (p, float(e)))
    for n in (25, 60):
        e, p, *_ = wil(n, -1.5); rows.append(("wilcoxon", f"{n}*", p, e)); print("Wilcoxon tail", n, "p=%.3e rel %.1e" % (p, float(e)))
    for n in (30, 80):
        e, p, *_ = ken(n, -3.0); rows.append(("kendalltau", f"{n}*", p, e)); print("Kendall tail", n, "p=%.3e rel %.1e" % (p, float(e)))
    for m, n in ((20, 30), (50, 70)):
        e, p, *_ = ks(m, n, 2.0); rows.append(("ks_2samp", f"{m},{n}*", p, e)); print("KS tail", m, n, "p=%.3e rel %.1e" % (p, float(e)))
    with open("library.dat", "w") as f:
        f.write("api size p rel\n")
        for a, s, p, e in rows: f.write("%s %s %.6e %.3e\n" % (a, s, p, float(e)))
