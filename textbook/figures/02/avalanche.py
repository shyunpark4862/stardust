"""Avalanche matrices of 64-bit mixers (Section 2.5; csrc/frame/ops/hash.hpp Fmix64).

For f in {x*c1, one round x ^= x>>33; x *= c1, fmix64} and every input bit i and output bit j,
p_ij = P(bit j of f(x) != bit j of f(x ^ 2^i)) over N random x (numpy PCG64, seed 2). The
plotted value is |2 p_ij - 1|: 0 for an ideal mixer, 1 when bit j never or always changes.
Writes avalanche_{mul,round,fmix}.dat (pgfplots matrix plot rows: i j value).
"""
import numpy as np

N = 1 << 16
C1 = np.uint64(0xFF51AFD7ED558CCD)
C2 = np.uint64(0xC4CEB9FE1A85EC53)
S = np.uint64(33)


def mul(x):
    return x * C1


def rnd(x):
    x = x ^ (x >> S)
    return x * C1


def fmix(x):
    x = x ^ (x >> S)
    x = x * C1
    x = x ^ (x >> S)
    x = x * C2
    return x ^ (x >> S)


rng = np.random.default_rng(2)
x = rng.integers(0, 2**64, size=N, dtype=np.uint64)
bits = np.arange(64, dtype=np.uint64)
with np.errstate(over="ignore"):
    for name, f in (("mul", mul), ("round", rnd), ("fmix", fmix)):
        y = f(x)
        m = np.zeros((64, 64))
        for i in range(64):
            d = y ^ f(x ^ (np.uint64(1) << np.uint64(i)))
            m[i] = ((d[:, None] >> bits) & np.uint64(1)).mean(axis=0)
        v = np.abs(2 * m - 1)
        with open(f"avalanche_{name}.dat", "w") as fh:
            for i in range(64):
                for j in range(64):
                    fh.write(f"{i} {j} {v[i, j]:.4f}\n")
                fh.write("\n")
        print(name, "max", v.max().round(4), "mean", v.mean().round(4), "cells > 0.1:", int((v > 0.1).sum()))


# Grayscale images for the figure (pgfplots draws the axes with \addplot graphics): row 0 of the
# image is input bit 63 (the top of the axis), each cell 8x8 pixels, white 0 and black 1.
import struct
import zlib


def png_gray(path, img):
    h, w = len(img), len(img[0])
    raw = b"".join(b"\x00" + bytes(row) for row in img)

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(data)


S8 = 8
for name in ("mul", "round", "fmix"):
    v = np.loadtxt(f"avalanche_{name}.dat").reshape(64, 64, 3)[:, :, 2]  # v[i, j]
    img = []
    for i in range(63, -1, -1):
        row = [int(round(255 * (1 - v[i, j]))) for j in range(64) for _ in range(S8)]
        img.extend([row] * S8)
    png_gray(f"avalanche_{name}.png", img)
