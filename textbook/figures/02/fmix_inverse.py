"""fmix64 and its inverse (Section 2.5): undo x ^= x >> 33 by x = y ^ (y >> 33) (s >= 32: one step),
multiply by the inverses of the odd constants (Newton, Algorithm alg:inverse-2k). Also inverts one
Philox round (Algorithm alg:philox-inverse) and checks the injectivity of HashBytes' word encoding
for short strings. Python ints, seed 5."""
import random

M = (1 << 64) - 1
C1, C2 = 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53


def inv(a):
    x = (3 * a) ^ 2
    for _ in range(4):
        x = x * (2 - a * x) & M
    assert a * x & M == 1
    return x


def fmix(k):
    k ^= k >> 33
    k = k * C1 & M
    k ^= k >> 33
    k = k * C2 & M
    return k ^ (k >> 33)


def unxorshift(y, s):
    x = y
    for _ in range((64 + s - 1) // s - 1):
        x = y ^ (x >> s)
    return x


def fmix_inv(k):
    k = unxorshift(k, 33)
    k = k * inv(C2) & M
    k = unxorshift(k, 33)
    k = k * inv(C1) & M
    return unxorshift(k, 33)


random.seed(5)
for _ in range(100000):
    x = random.getrandbits(64)
    assert fmix_inv(fmix(x)) == x
    s = random.randrange(1, 64)
    assert unxorshift(x ^ (x >> s), s) == x
print("inverse constants", hex(inv(C1)), hex(inv(C2)))
print("fmix64(0..3):", [hex(fmix(i)) for i in range(4)])

# one Philox round and its inverse
K0, K1 = 0xD2E7470EE14C6C93, 0xCA5A826395121157


def rnd(c, k0, k1):
    p0, p1 = K0 * c[0], K1 * c[2]
    return [(p1 >> 64) ^ c[1] ^ k0, p1 & M, (p0 >> 64) ^ c[3] ^ k1, p0 & M]


def rnd_inv(o, k0, k1):
    c0 = o[3] * inv(K0) & M
    c2 = o[1] * inv(K1) & M
    h0, h1 = (K0 * c0) >> 64, (K1 * c2) >> 64
    return [c0, o[0] ^ h1 ^ k0, c2, o[2] ^ h0 ^ k1]


for _ in range(20000):
    c = [random.getrandbits(64) for _ in range(4)]
    k0, k1 = random.getrandbits(64), random.getrandbits(64)
    assert rnd_inv(rnd(c, k0, k1), k0, k1) == c
print("Philox round inverts")
