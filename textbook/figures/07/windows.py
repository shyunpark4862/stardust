"""Mahonian probabilities g_n(k) = M_n(k)/n! in binary64 by (a) the sliding window as a difference of running sums,
(b) the block window (prefix and suffix sums inside blocks of length j, no subtraction), against exact rationals;
the tail k <= b. Also the distributions for figures and the normal approximation's error.
Output: windows.dat, dists.dat (U for N=20, c=8; T+ for n=10; inversions for n=8), normal.dat."""
from fractions import Fraction as F
from math import comb, factorial, sqrt, erf
from exact import qbinom, subset_sums, mahonian

def step_diff(g, j, b):
    out = []; run = 0.0
    for k in range(b + 1):
        run += g[k] if k < len(g) else 0.0
        if 0 <= k - j < len(g): run -= g[k - j]
        out.append(run / j)
    return out

def step_block(g, j, b):
    G = [g[k] if k < len(g) else 0.0 for k in range(b + 1)]
    pre = [0.0] * (b + 1); suf = [0.0] * (b + 1)
    for s in range(0, b + 1, j):
        e = min(b, s + j - 1); acc = 0.0
        for k in range(s, e + 1): acc += G[k]; pre[k] = acc
        acc = 0.0
        for k in range(e, s - 1, -1): acc += G[k]; suf[k] = acc
    out = []
    for k in range(b + 1):
        l = k - j + 1
        w = suf[l] + pre[k] if (l > 0 and l % j != 0) else pre[k]
        out.append(w / j)
    return out

def run(n=60, b=400):
    ex = mahonian(n); tot = factorial(n)
    gd = [1.0]; gb = [1.0]
    for j in range(2, n + 1):
        gd = step_diff(gd, j, b); gb = step_block(gb, j, b)
    with open("windows.dat", "w") as f:
        f.write("k diff block\n")
        for k in range(b + 1):
            e = F(ex[k], tot)
            f.write("%d %.4e %.4e\n" % (k, max(float(abs(F(gd[k]) - e) / e), 1e-18), max(float(abs(F(gb[k]) - e) / e), 1e-18)))
    print("n=%d: diff-window rel err at k=0,10,100: %.1e %.1e %.1e; block %.1e %.1e %.1e" % ((n,) + tuple(float(abs(F(gd[k]) - F(ex[k], tot)) / F(ex[k], tot)) for k in (0, 10, 100)) + tuple(float(abs(F(gb[k]) - F(ex[k], tot)) / F(ex[k], tot)) for k in (0, 10, 100))))

def dists():
    U = qbinom(20, 8); T = subset_sums(list(range(1, 11))); M = mahonian(8)
    with open("dists_u.dat", "w") as f:
        f.write("k p\n")
        for k, v in enumerate(U): f.write("%d %.6e\n" % (k, v / comb(20, 8)))
    with open("dists_t.dat", "w") as f:
        f.write("k p\n")
        for k, v in enumerate(T): f.write("%d %.6e\n" % (k, v / 2 ** 10))
    with open("dists_k.dat", "w") as f:
        f.write("k p\n")
        for k, v in enumerate(M): f.write("%d %.6e\n" % (k, v / factorial(8)))

def normal():
    """P(U <= u) exact vs normal with continuity correction, N = 40, c = 20, and N = 200, c = 100."""
    rows = []
    for N, c in ((40, 20), (200, 100)):
        P = qbinom(N, c); tot = comb(N, c); d = N - c
        mu = c * d / 2; sd = sqrt(c * d * (N + 1) / 12)
        acc = 0
        for u in range(0, int(mu) + 1):
            acc += P[u]; ex = F(acc, tot)
            z = (u + 0.5 - mu) / sd; approx = 0.5 * (1 + erf(z / sqrt(2)))
            rows.append((N, u, float(ex), approx, abs(approx - float(ex)) / float(ex)))
    with open("normal.dat", "w") as f:
        f.write("N z exact approx rel\n")
        for N, u, ex, ap, r in rows:
            c = N // 2; mu = c * c / 2; sdv = sqrt(c * c * (N + 1) / 12)
            f.write("%d %.4f %.6e %.6e %.4e\n" % (N, (u + 0.5 - mu) / sdv, ex, ap, r))
    for N in (40, 200):
        sub = [r for r in rows if r[0] == N]
        for r in sub:
            if r[2] > 1e-6: print("N=%d first u with P>1e-6: exact %.3e normal %.3e rel %.2f" % (N, r[2], r[3], r[4])); break

if __name__ == "__main__":
    run(); dists(); normal()
