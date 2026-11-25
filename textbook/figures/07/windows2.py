"""The same comparison for n = 200 up to the middle (b = n(n-1)/4), where the windows are long against the
distribution's spread; probabilities below 1e-290 skipped (binary64 underflow; the library rescales by 2^600)."""
from fractions import Fraction as F
from math import factorial
from exact import mahonian
from windows import step_diff, step_block
n = 200; b = n * (n - 1) // 4
ex = mahonian(n); tot = factorial(n)
gd = [1.0]; gb = [1.0]
for j in range(2, n + 1):
    gd = step_diff(gd, j, b); gb = step_block(gb, j, b)
worst_d = worst_b = 0; rows = []
for k in range(0, b + 1, 25):
    e = F(ex[k], tot)
    if e < F(1, 10 ** 290): continue
    rd = float(abs(F(gd[k]) - e) / e); rb = float(abs(F(gb[k]) - e) / e)
    rows.append((k, rd, rb)); worst_d = max(worst_d, rd); worst_b = max(worst_b, rb)
with open("windows200.dat", "w") as f:
    f.write("k diff block\n")
    for r in rows: f.write("%d %.4e %.4e\n" % (r[0], max(r[1], 1e-18), max(r[2], 1e-18)))
print("n=200 up to the middle: diff max %.2e, block max %.2e" % (worst_d, worst_b))
