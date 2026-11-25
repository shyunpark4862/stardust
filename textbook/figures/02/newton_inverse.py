"""Correct low bits of the Newton iteration x <- x (2 - a x) for the inverse of an odd a mod 2^64.

Start x0 = a (3 correct bits, a*a = 1 mod 8) or x0 = (3a) xor 2 (5 correct bits). The number of
correct low bits is the number of trailing zeros of a*x - 1 (64 when exact). Reports the minimum
over 10^5 random odd a (Python random, seed 3) after each step, and the Philox multipliers.
"""
import random

M = 1 << 64
random.seed(3)


def tz(v):
    v %= M
    return 64 if v == 0 else (v & -v).bit_length() - 1


def run(start):
    worst = [64] * 6
    for _ in range(100000):
        a = random.getrandbits(64) | 1
        x = start(a) % M
        for k in range(6):
            worst[k] = min(worst[k], tz(a * x - 1))
            x = x * (2 - a * x) % M
    return worst


print("x0 = a       :", run(lambda a: a))
print("x0 = 3a xor 2:", run(lambda a: (3 * a) ^ 2))
for c in (0xD2E7470EE14C6C93, 0xCA5A826395121157, 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53):
    x = c
    for _ in range(5):
        x = x * (2 - c * x) % M
    assert c * x % M == 1 and x == pow(c, -1, M)
    print(hex(c), "->", hex(x))
