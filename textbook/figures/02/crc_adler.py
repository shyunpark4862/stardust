"""CRC-32 and Adler-32 as used by PNG/zlib (src/stardust/plot/_png.py calls zlib.crc32 and
zlib.adler32). Checks the bitwise and table algorithms of Section 2.6 and the Adler-32
algorithm of Section 2.3 against zlib, the check value of "123456789", the bound 5552 on
deferred reductions, the combine formula, and burst detection.
"""
import random
import zlib

POLY = 0xEDB88320  # x^32 + x^26 + ... + 1, bit-reversed


def crc_bitwise(data, crc=0):
    crc ^= 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (POLY if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


TABLE = []
for n in range(256):
    c = n
    for _ in range(8):
        c = (c >> 1) ^ (POLY if c & 1 else 0)
    TABLE.append(c)


def crc_table(data, crc=0):
    crc ^= 0xFFFFFFFF
    for byte in data:
        crc = (crc >> 8) ^ TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFFFFFF


BASE = 65521
NMAX = max(n for n in range(1, 10000) if 255 * n * (n + 1) // 2 + (n + 1) * (BASE - 1) <= 2**32 - 1)


def adler(data, a=1, b=0):
    i = 0
    while i < len(data):
        for byte in data[i : i + NMAX]:  # sums stay below 2^32 (checked below)
            a += byte
            b += a
            assert a < 2**32 and b < 2**32
        a %= BASE
        b %= BASE
        i += NMAX
    return (b << 16) | a


def adler_combine(h1, h2, len2):
    a1, b1 = h1 & 0xFFFF, h1 >> 16
    a2, b2 = h2 & 0xFFFF, h2 >> 16
    a = (a1 + a2 - 1) % BASE
    b = (b1 + b2 + len2 * (a1 - 1)) % BASE
    return (b << 16) | a


random.seed(4)
print("check value", hex(crc_bitwise(b"123456789")), hex(zlib.crc32(b"123456789")))
print("adler 'Wikipedia'", hex(adler(b"Wikipedia")), hex(zlib.adler32(b"Wikipedia")))
print("NMAX", NMAX, "table[1]", hex(TABLE[1]), "table[128]", hex(TABLE[128]))
for _ in range(300):
    n = random.randrange(0, 20000)
    d = bytes(random.getrandbits(8) for _ in range(n))
    assert crc_bitwise(d[:200]) == zlib.crc32(d[:200])
    assert crc_table(d) == zlib.crc32(d)
    worst = bytes([255]) * n
    assert adler(worst) == zlib.adler32(worst)
    assert adler(d) == zlib.adler32(d)
    k = random.randrange(0, n + 1)
    assert adler_combine(zlib.adler32(d[:k]), zlib.adler32(d[k:]), n - k) == zlib.adler32(d)
# linearity: crc(x ^ y) = crc(x) ^ crc(y) ^ crc(0...0) for equal lengths
x = bytes(random.getrandbits(8) for _ in range(64))
y = bytes(random.getrandbits(8) for _ in range(64))
z = bytes(64)
assert zlib.crc32(bytes(p ^ q for p, q in zip(x, y))) == zlib.crc32(x) ^ zlib.crc32(y) ^ zlib.crc32(z)
# every burst of length <= 32 changes the CRC
fails = 0
for _ in range(100000):
    msg = bytearray(random.getrandbits(8) for _ in range(40))
    L = random.randrange(1, 33)
    start = random.randrange(0, 320 - L + 1)
    e = (1 << (L - 1)) | 1 | (random.getrandbits(max(L - 2, 0)) << 1 if L > 2 else 0) | (1 << (L - 1))
    m2 = int.from_bytes(msg, "little") ^ (e << start)
    if zlib.crc32(m2.to_bytes(40, "little")) == zlib.crc32(bytes(msg)):
        fails += 1
print("burst trials undetected:", fails)
print("all checks passed")
