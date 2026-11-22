"""sd.fft's references and their contract (docs/design/fft.md §4, §7.2), as the tests and the accuracy
measurement (bench/accuracy) both judge them.

tests/fft/ref/<family>.npz (tools/fft_ref.py; layout: tests/reffile.py) holds one point per signal:
the inputs "signal" (the id in tests/fft/signals.py), "digest" (signals.digest of the input), "n"
(the transform length; nd: the shape's entry count) and "x" (the input itself, ragged), and per
method the reference as a double-double hi + lo (complex128 or float64 arrays, ragged):

    c2c        x complex (n); methods forward (sum_j x_j e^(-2 pi i jk/n)) and backward (the sign
               +, no 1/n), both unnormalised
    real       x float64 (n); method forward: the half spectrum k = 0 .. n // 2 of the forward DFT
    hermitian  x complex (n // 2 + 1); method c2r: the n real values of the inverse DFT, no 1/n,
               of the Hermitian extension of x with the imaginary parts of x_0 and (even n) x_(n/2)
               dropped: numpy's and scipy's irfft(x, n) * n
    r2r        x float64 (N); methods dct1 .. dct4, dst1 .. dst4: scipy's definitions at
               norm='backward' (dct1 for N >= 2)
    nd         x complex of a shape (inputs "shape"); methods forward, backward: the n-dimensional
               DFT over every axis, unnormalised

Arb settles every entry to 2^-120 ||y|| (the scale of reffile's target_bits: the transform's 2-norm),
since an entry of an FFT has no relative accuracy of its own (docs/design/fft.md §4.1); the stored
double-double adds its own rounding, 2^-106 of the entry. Both are far below the errors measured.
The signals of length above 2^14 are not stored: the accuracy measurement computes those few lines
with Arb where it runs.

The contract (decision FT27; constants fixed by milestone FT0, docs/design/fft.md §4.2), in units of
its bound (1 or less is within it): for a transform of length n with L(n) = ceil(log2 n),

    complex, real (rfft, ihfft)   ||y_hat - y|| <= (KAPPA L(n) + C0) u s sqrt(n) ||x|| + tau
    hermitian (irfft, hfft)       ||y_hat - y|| <= (KAPPA L(n) + C0) u s sqrt(n) ||x~||_H + tau
    n-dimensional                 L(n) -> sum of L(n_i), C0 -> C0 + (d - 1) C1
    dct, dst                      ||y_hat - y|| <= (KAPPA L(M) + C2) u s ||T|| ||x|| + tau
    Bluestein lengths             L(n) -> 2 L(m) + 2, the bound and tau times rho(n)

with s the normalisation factor, ||x~||_H the norm of the Hermitian extension, ||T|| the 2-norm of
the unnormalised real transform (norm_r2r), M = 2N + 2 (an upper bound on the inner FFT's length of
every type) and tau = KAPPA m 2^-1074 with m the transform's length (n; the entry count of an
n-dimensional one; 2N + 2 for dct and dst): the floor of gradual underflow. A product of subnormal
numbers is wrong by up to 2^-1075 in each part, so a stage adds an absolute error of norm up to
sqrt(2) 2^-1074 sqrt(n), and the later stages of radices r_i multiply it by sqrt(prod r_i); summed
over the stages that is below 3.5 sqrt(2) n 2^-1074 (and so below KAPPA n 2^-1074), which also
covers the final rounding of a subnormal result.

Bluestein lengths (decisions FT36, FT37; docs/design/fft.md §4.2): the complex transform of a length n
with a prime factor above 11 is a cyclic convolution of length m, the smallest 2^a 3^b 5^c >= 2n - 1
(bluestein_m), and rho(n) = max_k |DFT_m(b)_k| / sqrt(n) for the chirp kernel b_l = e^(pi i l^2 / n),
|l| < n (bluestein_rho: computed here independently of the library, in double precision, which moves
rho by far less than the bound's slack; the tests compare it with the plan's double-double value).
"""

from __future__ import annotations

import functools
import json
import math
import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("c2c", "real", "hermitian", "r2r", "nd")
U = 2.0**-53
TINY = 2.0**-1074
KAPPA = 8.0  # per log2 level of the factorisation (docs/design/fft.md §4.2)
C0 = 4.0  # normalisation, the real path's post-processing and the (de)interleaving
C1 = 2.0  # per further axis of an n-dimensional transform
C2 = 8.0  # the real transforms' pre- and post-processing around their inner FFT
R2R = ("dct1", "dct2", "dct3", "dct4", "dst1", "dst2", "dst3", "dst4")


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def levels(n: int) -> int:
    """L(n) = ceil(log2 n), L(1) = 0."""
    return (int(n) - 1).bit_length()


def norm2(a) -> float:
    """The 2-norm of a real or complex array, with a sum that neither overflows nor underflows."""
    a = np.abs(np.asarray(a, dtype=np.complex128)).ravel()
    if a.size == 0:
        return 0.0
    big = float(a.max())
    if big == 0.0 or not math.isfinite(big):
        return big
    e = math.frexp(big)[1]
    return math.ldexp(math.sqrt(math.fsum(math.ldexp(float(v), -e) ** 2 for v in a)), e)


def hermitian_norm(x, n: int) -> float:
    """||x~||_H: the 2-norm of the Hermitian extension (length n) of the half spectrum x (n // 2 + 1
    entries used, zero-padded; the imaginary parts of x_0 and, for even n, x_(n/2) dropped)."""
    m = n // 2 + 1
    z = np.zeros(m, np.complex128)
    z[: min(m, len(x))] = np.asarray(x)[:m]
    z[0] = z[0].real
    if n % 2 == 0:
        z[-1] = z[-1].real
    inner = z[1 : m - 1] if n % 2 == 0 else z[1:]
    parts = [abs(z[0])] + [math.sqrt(2) * abs(v) for v in inner]
    if n % 2 == 0 and m > 1:
        parts.append(abs(z[-1]))
    return norm2(np.array(parts))


def norm_r2r(method: str, n: int) -> float:
    """||T||_2 of the unnormalised real transform of length n (scipy's norm='backward' forward
    definitions): each is a diagonal scaling of an orthogonal matrix (dct1: an upper bound)."""
    if method == "dct1":
        return 2.0 * math.sqrt(n - 1)
    if method in ("dct2", "dst2"):
        return 2.0 * math.sqrt(n)
    if method == "dst1":
        return math.sqrt(2.0 * (n + 1))
    if method in ("dct3", "dct4", "dst3", "dst4"):
        return math.sqrt(2.0 * n)
    raise KeyError(method)


def is_smooth11(n: int) -> bool:
    """Whether every prime factor of n is at most 11: the lengths the mixed radix transforms directly."""
    n = int(n)
    for p in (2, 3, 5, 7, 11):
        while n % p == 0:
            n //= p
    return n == 1


def bluestein_m(n: int) -> int:
    """The inner length of Bluestein's transform of length n: the smallest 2^a 3^b 5^c >= 2n - 1."""
    target = 2 * int(n) - 1
    best = None
    p5 = 1
    while True:
        p35 = p5
        while True:
            v = p35
            while v < target:
                v *= 2
            best = v if best is None else min(best, v)
            if p35 >= target:
                break
            p35 *= 3
        if p5 >= target:
            break
        p5 *= 5
    return best


@functools.cache
def bluestein_rho(n: int) -> float:
    """rho(n) = max_k |DFT_m(b)_k| / sqrt(n), b the chirp kernel (see the module's docstring)."""
    n = int(n)
    m = bluestein_m(n)
    lsq = (np.arange(n, dtype=np.int64) ** 2) % (2 * n)
    w = np.exp(1j * np.pi * (lsq / n))
    b = np.zeros(m, np.complex128)
    b[:n] = w
    b[m - n + 1 :] = w[1:][::-1]
    return float(np.abs(np.fft.fft(b)).max() / math.sqrt(n))


def tau(m: int) -> float:
    """The underflow floor of a transform of length m."""
    return KAPPA * m * TINY


def bound(kind: str, n, xnorm: float, s: float = 1.0, method: str | None = None) -> float:
    """The contract's bound on ||y_hat - y|| (absolute) for a transform of the given kind:
    "c2c", "real", "hermitian" (xnorm: ||x~||_H), "nd" (n: the shape), or "r2r" (method: dct1 ..
    dst4), with input norm xnorm and normalisation factor s."""
    if kind == "nd":
        shape = tuple(int(v) for v in n)
        count = math.prod(shape)
        lv = sum(levels(v) for v in shape)
        c = C0 + (len(shape) - 1) * C1
        return (KAPPA * lv + c) * U * s * math.sqrt(count) * xnorm + tau(count)
    n = int(n)
    if kind == "c2c" and not is_smooth11(n):
        m, rho = bluestein_m(n), bluestein_rho(n)
        return rho * ((KAPPA * (2 * levels(m) + 2) + C0) * U * s * math.sqrt(n) * xnorm + tau(m))
    if kind in ("c2c", "real", "hermitian"):
        return (KAPPA * levels(n) + C0) * U * s * math.sqrt(n) * xnorm + tau(n)
    if kind == "r2r":
        return (KAPPA * levels(2 * n + 2) + C2) * U * s * norm_r2r(method, n) * xnorm + tau(2 * n + 2)
    raise KeyError(kind)


def error(got, hi, lo) -> float:
    """||got - (hi + lo)||: the double-double reference's error, its parts subtracted in order."""
    d = (np.asarray(got, dtype=np.complex128) - np.asarray(hi)) - np.asarray(lo)
    return norm2(d)


def units(err: float, b: float) -> float:
    """err in units of the bound b (0 when the error is exactly 0)."""
    if err == 0:
        return 0.0
    return math.inf if b == 0 else err / b


# Lines longer than the files hold (decision FT33): their reference computed where it is needed.
LONG = ("c/uniform/65536", "c/uniform/65537", "c/uniform/30021", "c/uniform/1048576")


def arb_dft(x, inverse: bool = False):
    """(hi, lo): the unnormalised DFT of the complex line x (sign -, or + with inverse) by Arb's acb.dft,
    each entry settled to 2^-120 of the output's 2-norm (as tools/fft_ref.py's files) and kept as a
    double-double. For the long lines of LONG (a few seconds at 2^20)."""
    from flint import acb, arb, ctx

    x = np.asarray(x, np.complex128)
    old = ctx.prec
    prec = 192
    try:
        while True:
            ctx.prec = prec
            v = [acb(float(z.real), float(-z.imag if inverse else z.imag)) for z in x]
            y = acb.dft(v) if v else []
            scale = norm2(np.array([complex(float(a.real.mid()), float(a.imag.mid())) for a in y]))
            limit = arb(scale) * arb(2) ** -120
            if all(a.real.rad() <= limit and a.imag.rad() <= limit for a in y) or prec >= 1536:
                break
            prec *= 2
        hi = np.empty(len(y), np.complex128)
        lo = np.empty(len(y), np.complex128)
        for k, a in enumerate(y):
            re, im = a.real.mid(), a.imag.mid()
            if inverse:
                im = -im
            h_re, h_im = float(re), float(im)
            hi[k] = complex(h_re, h_im)
            lo[k] = complex(float(re - arb(h_re)), float(im - arb(h_im)))
        return hi, lo
    finally:
        ctx.prec = old
