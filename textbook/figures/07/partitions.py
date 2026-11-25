"""Partition numbers p(s) by the product of 1/(1 - q^k) (one step-k prefix sum per factor, exact integers),
against the Hardy-Ramanujan leading term e^{pi sqrt(2s/3)} / (4 s sqrt 3). Output: partitions.dat."""
import math
S = 400
p = [1] + [0] * S
for k in range(1, S + 1):
    for s in range(k, S + 1): p[s] += p[s - k]
with open("partitions.dat", "w") as f:
    f.write("s p hr ratio\n")
    for s in range(1, S + 1):
        hr = math.exp(math.pi * math.sqrt(2 * s / 3)) / (4 * s * math.sqrt(3))
        f.write("%d %.6e %.6e %.6f\n" % (s, p[s], hr, hr / p[s]))
print("p(4) =", p[4], "p(100) =", p[100], "p(400) = %.4e" % p[400], "HR/p at 100: %.4f, 400: %.4f" % (math.exp(math.pi*math.sqrt(200/3))/(400*math.sqrt(3))/p[100], math.exp(math.pi*math.sqrt(800/3))/(1600*math.sqrt(3))/p[400]))
