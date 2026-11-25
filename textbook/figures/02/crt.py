"""Garner's reconstruction (Algorithm alg:crt) on random residues for several sets of pairwise
coprime moduli, checked against every congruence; also the worked example of Figure fig:crt-grid
(x = 2 mod 3, x = 1 mod 5 gives 11). Python ints, seed 6."""
import math
import random


def garner(m, a):
    v = []
    for j in range(len(m)):
        x = a[j]
        for i in range(j):
            x = (x - v[i]) % m[j] * pow(m[i], -1, m[j]) % m[j]
        v.append(x)
    return v


def value(m, v):
    x, w = 0, 1
    for mj, vj in zip(m, v):
        x += vj * w
        w *= mj
    return x


random.seed(6)
sets = [[3, 5], [7, 11, 13], [2**32 - 5, 2**31 - 1, 65521], [9, 10, 7, 11, 13, 17]]
for m in sets:
    assert all(math.gcd(p, q) == 1 for i, p in enumerate(m) for q in m[i + 1:])
    for _ in range(2000):
        a = [random.randrange(mj) for mj in m]
        v = garner(m, a)
        x = value(m, v)
        assert all(0 <= vj < mj for vj, mj in zip(v, m)) and all(x % mj == aj for mj, aj in zip(m, a))
print("example:", garner([3, 5], [2, 1]), value([3, 5], garner([3, 5], [2, 1])))
print("all checks passed")
