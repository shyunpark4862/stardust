"""The per-butterfly error bounds of the library's own complex FFT (docs/design/fft.md §4.2, decision
FT31; csrc/fft/cfft.hpp): a first-order propagation of every rounding of each butterfly exactly as the
C++ computes it, with the exact errors of its rounded constants.

Every intermediate value v (a real or an imaginary part) is tracked as two vectors over the 2r real
input parts x: its exact linear form L_v (v = L_v . x) and a bound E_v >= 0 on its accumulated error
(|v_hat - v| <= E_v . |x| to first order). A sum v = a + b rounds once: E_v = E_a + E_b + u |L_v|. A
product by a rounded constant c_hat = c + e: E_v = |c| E_a + |e| |L_a| + u |L_v|; a fused multiply-add
c a + b: E_v = |c| E_a + E_b + |e| |L_a| + u |L_v|. Exact operations (-i, negation) move the vectors.
With M the matrix of the outputs' E rows, ||y_hat - y|| <= ||M|x||| <= ||M||_2 ||x||, and the exact
butterfly has ||y|| = sqrt(r) ||x||, so eta_b = ||M||_2 / sqrt(r) bounds the butterfly's relative error.

A stage multiplies the butterfly's outputs by twiddles: the FMA complex product adds 2u (Jeannerod,
Kornerup, Louvet and Muller, Math. Comp. 86 (2017)) and the twiddle's own error mu (twiddle.hpp; the
proof takes mu = 1.5u, decision FT31): eta_stage = eta_b + (2u + mu)(1 + eta_b). The transform of
length n is within prod(1 + eta_stage) - 1 <= kappa L(n) u + O(u^2) whenever every stage has
eta_stage <= kappa log2(r) u, since sum log2 r = log2 n <= L(n). kappa = 8 covers the radices up to 11;
the radices 13 to 31 (milestone FT6, decision FT64) need up to 16.83u per log2 r (31), so a length with one
of them is held to kappa = 17.

The first-order bounds leave out terms of order u^2 (each rounding's (1 + d) products): with at most
30 roundings on any path they change a bound by less than 30 u relatively, 10^-14 of it.

Units: u = 1 (every bound is in units of u = 2^-53).
"""

from __future__ import annotations

import math
import pathlib
import re

import numpy as np

CFFT = pathlib.Path(__file__).parents[2] / "csrc" / "fft" / "cfft.hpp"
U = 1.0
MU = 1.5  # the twiddles' error bound (decision FT31)
CMUL = 2.0  # the FMA complex product
KAPPA = 8.0
KAPPA_WIDE = 17.0  # the radices 13 to 31 (decision FT64)
RADICES = (2, 4, 3, 5, 7, 11)
WIDE_RADICES = (13, 17, 19, 23, 29, 31)


class Val:
    __slots__ = ("L", "E")

    def __init__(self, L, E):
        self.L = np.asarray(L, float)
        self.E = np.asarray(E, float)


def _inputs(r):
    out = []
    for j in range(r):
        re, im = np.zeros(2 * r), np.zeros(2 * r)
        re[2 * j], im[2 * j + 1] = 1.0, 1.0
        out.append((Val(re, np.zeros(2 * r)), Val(im, np.zeros(2 * r))))
    return out


def _add(a, b, sign=1.0):
    L = a.L + sign * b.L
    return Val(L, a.E + b.E + U * np.abs(L))


def _mul(c, e, a):
    L = c * a.L
    return Val(L, abs(c) * a.E + abs(e) * np.abs(a.L) + U * np.abs(L))


def _fma(c, e, a, b):
    L = c * a.L + b.L
    return Val(L, abs(c) * a.E + b.E + abs(e) * np.abs(a.L) + U * np.abs(L))


def _cadd(x, y):
    return (_add(x[0], y[0]), _add(x[1], y[1]))


def _csub(x, y):
    return (_add(x[0], y[0], -1.0), _add(x[1], y[1], -1.0))


def _neg_i(x):
    return (x[1], Val(-x[0].L, x[0].E))


def constants():
    """{p: (c, s)} of cfft.hpp's OddConst: cos and sin of 2 pi m / p as the C++ has them."""
    text = CFFT.read_text()
    out = {}
    for p in (3, 5, 7, 11) + WIDE_RADICES:
        body = text[text.index(f"struct OddConst<{p}>") :]
        c = re.search(r"c\[\d+\] = \{([^}]*)\}", body).group(1)
        s = re.search(r"s\[\d+\] = \{([^}]*)\}", body).group(1)
        out[p] = (_parse(c), _parse(s))
    return out


def _parse(text):
    return [float.fromhex(v.strip()) if "x" in v else float(v) for v in text.split(",") if v.strip()]


def constant_errors(p, values, which):
    """|c_hat - c| in units of u for each constant (Arb at 300 bits) and whether each is the nearest double."""
    from flint import arb, ctx

    old = ctx.prec
    ctx.prec = 300
    try:
        errs, nearest = [], []
        for m, v in enumerate(values):
            exact = arb.cos_pi(arb(2 * m) / p) if which == "c" else arb.sin_pi(arb(2 * m) / p)
            d = abs(arb(v) - exact)
            errs.append(float((d / arb(2.0**-53)).upper()))
            ulp = math.ulp(v) if v != 0 else 2.0**-1074
            nearest.append(bool(d <= arb(ulp) / 2))
        return errs, nearest
    finally:
        ctx.prec = old


def butterfly(r):
    """The outputs (re, im Val pairs) of cfft.hpp's butterfly of radix r."""
    a = _inputs(r)
    if r == 2:
        return [_cadd(a[0], a[1]), _csub(a[0], a[1])]
    if r == 4:
        t0, t1 = _cadd(a[0], a[2]), _csub(a[0], a[2])
        t2, t3 = _cadd(a[1], a[3]), _neg_i(_csub(a[1], a[3]))
        return [_cadd(t0, t2), _cadd(t1, t3), _csub(t0, t2), _csub(t1, t3)]
    c, s = constants()[r]
    ce, _ = constant_errors(r, c, "c")
    se, _ = constant_errors(r, s, "s")
    h = (r - 1) // 2
    t = [None] + [_cadd(a[j], a[r - j]) for j in range(1, h + 1)]
    d = [None] + [_csub(a[j], a[r - j]) for j in range(1, h + 1)]
    y = [None] * r
    y0 = a[0]
    for j in range(1, h + 1):
        y0 = _cadd(y0, t[j])
    y[0] = y0
    for k in range(1, h + 1):
        A = a[0]
        for j in range(1, h + 1):
            q = (j * k) % r
            A = (_fma(c[q], ce[q], t[j][0], A[0]), _fma(c[q], ce[q], t[j][1], A[1]))
        q = k % r
        B = (_mul(s[q], se[q], d[1][0]), _mul(s[q], se[q], d[1][1]))
        for j in range(2, h + 1):
            q = (j * k) % r
            B = (_fma(s[q], se[q], d[j][0], B[0]), _fma(s[q], se[q], d[j][1], B[1]))
        y[k] = (_add(A[0], B[1]), _add(A[1], B[0], -1.0))  # A - i B
        y[r - k] = (_add(A[0], B[1], -1.0), _add(A[1], B[0]))  # A + i B
    return y


def eta_butterfly(r) -> float:
    ys = butterfly(r)
    M = np.array([v.E for y in ys for v in y])
    return float(np.linalg.norm(M, 2) / math.sqrt(r))


def eta_stage(r) -> float:
    eb = eta_butterfly(r)
    return eb + (CMUL + MU) * (1.0 + eb * 2.0**-53)


def table(radices=RADICES):
    """{r: (eta_b, eta_stage, eta_stage / log2 r)} in units of u."""
    return {r: (eta_butterfly(r), eta_stage(r), eta_stage(r) / math.log2(r)) for r in radices}


if __name__ == "__main__":
    for r, (eb, es, per) in table(RADICES + WIDE_RADICES).items():
        print(f"radix {r:2d}: eta_b {eb:7.3f} u  eta_stage {es:7.3f} u  per log2 r {per:6.3f} u")
