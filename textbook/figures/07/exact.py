"""Exact null distributions by integer counting (references for chapter 7)."""
from fractions import Fraction as F
from math import comb, factorial

def qbinom(N, c):
    """coefficients of [N, c]_q (Mann-Whitney U counts)"""
    P = [1]
    for j in range(1, c + 1):
        a = N - j + 1
        P = P + [0] * (a)              # room for degree growth
        for v in range(len(P) - 1, a - 1, -1): P[v] -= P[v - a]
        for v in range(j, len(P)): P[v] += P[v - j]
        while len(P) > 1 and P[-1] == 0: P.pop()
    return P

def subset_sums(scores):
    C = [1] + [0] * sum(scores)
    top = 0
    for a in scores:
        top += a
        for s in range(top, a - 1, -1): C[s] += C[s - a]
    return C

def mahonian(n):
    M = [1]
    for j in range(2, n + 1):
        new = [0] * (len(M) + j - 1)
        run = 0
        for k in range(len(new)):
            run += M[k] if k < len(M) else 0
            if k - j >= 0: run -= M[k - j]
            new[k] = run
        M = new
    return M

def ks2_tail(m, n, h):
    """#paths with max |i/m - j/n| >= h/L (L = lcm), two-sided, no ties; returns (count, total)"""
    from math import lcm
    L = lcm(m, n); al, be = L // m, L // n
    # count paths staying strictly inside |i al - j be| < h
    W = {0: 1}
    for p in range(m + n):
        W2 = {}
        for i, w in W.items():
            j = p - i
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii > m or jj > n: continue
                if abs(ii * al - jj * be) >= h: continue
                W2[ii] = W2.get(ii, 0) + w
        W = W2
    inside = W.get(m, 0)
    total = comb(m + n, m)
    return total - inside, total
