"""Twiddle tables of stardust's FFT plans (_core.fft._twiddles(n): twiddle(k, n) for every k < n) against
Decimal references, every k, and the naive cos/sin of fl(2 pi k / n). Error |w_hat - w| / u (u = 2^-53).
Output: twiddle.dat (k/n bins: max naive, max library) for n = 3 * 2^16 + 1; summary to stdout."""
import math, sys
from multiprocessing import Pool
import numpy as np
from decimal import Decimal as D, getcontext
from fractions import Fraction as F

getcontext().prec = 40
U = 2.0 ** -53

def pi40():
    getcontext().prec = 50
    def at_inv(x):
        x = D(x); x2 = x * x; t = 1 / x; s = t; k = 1; sg = -1
        while True:
            t = t / x2; term = t / (2 * k + 1)
            if term < D(10) ** -48: break
            s += sg * term; sg = -sg; k += 1
        return s
    return 16 * at_inv(5) - 4 * at_inv(239)
PI = pi40()

def cs(num, den):
    """cos, sin of 2 pi num/den with the angle reduced to [-pi/4, pi/4] by exact octant symmetry."""
    getcontext().prec = 40
    r = F(num % den, den)  # turns
    o = math.floor(r * 8 + F(1, 2))  # nearest octant boundary... use quadrant reduction
    q = math.floor(r * 4 + F(1, 2)) % 4  # nearest quarter turn
    t = r - F(math.floor(r * 4 + F(1, 2)), 4)  # in [-1/8, 1/8]
    a = 2 * PI * D(t.numerator) / D(t.denominator)
    x2 = a * a; c = D(1); s = a; tc = D(1); ts = a; k = 1
    while True:
        tc = -tc * x2 / ((2 * k - 1) * (2 * k)); ts = -ts * x2 / ((2 * k) * (2 * k + 1))
        c += tc; s += ts; k += 1
        if abs(tc) < D(10) ** -38: break
    for _ in range(q): c, s = -s, c
    return c, s

def work(args):
    n, ks, tab = args
    out = []
    for k, (re, im) in zip(ks, tab):
        c, s = cs(k, n)
        e = float(abs(complex(float(D(re) - c), float(D(im) + s)))) / U  # w = c - i s
        th = 2 * math.pi * k / n
        nv = float(abs(complex(float(D(math.cos(th)) - c), float(D(-math.sin(th)) + s)))) / U
        out.append((k / n, nv, e))
    return out

def measure(n, procs=8, step=1):
    import stardust as sd
    from stardust import _core
    tab = np.asarray(_core.fft._twiddles(n))
    ks = list(range(0, n, step))
    chunks = [ks[i::procs] for i in range(procs)]
    with Pool(procs, maxtasksperchild=1) as p:
        res = p.map(work, [(n, c, [tuple(tab[k]) for k in c]) for c in chunks])
    rows = sorted(r for part in res for r in part)
    a = np.array(rows)
    return a

if __name__ == "__main__":
    for n in (1000, 4096, 65536, 3 * 2 ** 16 + 1, 2 ** 20):
        a = measure(n)
        print("n %8d  all k: library max %.3f u, naive max %.3f u" % (n, a[:, 2].max(), a[:, 1].max()), flush=True)
        if n == 3 * 2 ** 16 + 1:
            nb = 80
            with open("twiddle.dat", "w") as f:
                f.write("t naive lib\n")
                for b in range(nb):
                    m = (a[:, 0] >= b / nb) & (a[:, 0] < (b + 1) / nb)
                    f.write("%.6f %.4f %.4f\n" % ((b + 0.5) / nb, a[m, 1].max(), a[m, 2].max()))
