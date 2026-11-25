"""7장 보충 그림·수치: 초기하 pmf, 동점의 조건부 분포, Fisher--Yates 균등성(복제본으로)."""
import math, sys
from fractions import Fraction as Fr
from itertools import combinations
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# 1. 초기하 pmf (R=40, r=15, t=12)
R, r, t = 40, 15, 12
h = {k: Fr(math.comb(r, k) * math.comb(R - r, t - k), math.comb(R, t)) for k in range(0, min(r, t) + 1)}
mode = (t + 1) * (r + 1) // (R + 2)
with open(f"{OUT}/hypergeom.dat", "w") as f:
    f.write("k p\n")
    for k, p in h.items():
        f.write(f"{k} {float(p):.10e}\n")
print("hypergeom mode", mode, "argmax", max(h, key=h.get), "sum", sum(h.values()))

# 2. 동점이 있는 Mann--Whitney 순위합(두 배 중간 순위)의 조건부 분포
runs = [6, 1, 1, 1, 1, 1, 1]  # 같은 값 묶음의 크기, N = 12
c = 4
scores = []
b = 0
for tt in runs:
    scores += [2 * b + tt + 1] * tt
    b += tt
N = len(scores)
cnt = {}
for S in combinations(range(N), c):
    s = sum(scores[i] for i in S)
    cnt[s] = cnt.get(s, 0) + 1
tot = math.comb(N, c)
mean2 = Fr(c * sum(scores), N)  # 두 배 순위합의 평균
with open(f"{OUT}/ties.dat", "w") as f:
    f.write("s p m\n")
    for s in sorted(cnt):
        f.write(f"{s} {cnt[s]/tot:.10e} {float(2*mean2 - s)}\n")
sym = all(cnt.get(s, 0) == cnt.get(int(2 * mean2 - s), 0) for s in cnt) if mean2.denominator == 1 else False
print("ties N", N, "c", c, "mean2", mean2, "symmetric", sym, "support", min(cnt), max(cnt))
lo = sum(v for s, v in cnt.items() if s <= mean2 - 12); hi = sum(v for s, v in cnt.items() if s >= mean2 + 12)
print("P(S<=mean-10)", Fr(lo, tot), float(Fr(lo, tot)), "P(S>=mean+10)", Fr(hi, tot), float(Fr(hi, tot)))
