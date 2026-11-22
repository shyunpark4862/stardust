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
    n-dimensional                 L(n) -> sum of L(n_i), C0 -> C0 + (d - 1) C1 (bound_nd)
    dct, dst                      ||y_hat - y|| <= (KAPPA L(M) + C2) u s ||T|| ||x|| + tau
    a radix of 13 to 31           KAPPA -> KAPPA_WIDE (in the level term and tau; decision FT64)
    Bluestein lengths             L(n) -> 2 L(m) + 2, the bound and tau times rho(n)

with s the normalisation factor, ||x~||_H the norm of the Hermitian extension, ||T|| the 2-norm of
the unnormalised real transform (norm_r2r), M = 2N + 2 (an upper bound on the inner FFT's length of
every type) and tau = KAPPA m 2^-1074 with m the transform's length (n; the entry count of an
n-dimensional one; 2N + 2 for dct and dst): the floor of gradual underflow. A product of subnormal
numbers is wrong by up to 2^-1075 in each part, so a stage adds an absolute error of norm up to
sqrt(2) 2^-1074 sqrt(n), and the later stages of radices r_i multiply it by sqrt(prod r_i); summed
over the stages that is below 3.5 sqrt(2) n 2^-1074 (and so below KAPPA n 2^-1074), which also
covers the final rounding of a subnormal result.

Real and Hermitian transforms (milestone FT2, decision FT43; docs/design/fft.md §3.4, §4.2) run a complex
transform of the inner length real_inner(n): n / 2 for even n, n for odd n. When that length is
31-smooth the bound is the one above (with kappa of the inner length); when it is a Bluestein length h
the bound is Bluestein's with h's rho and inner length m, L(n) -> 2 L(m) + 2 (+ 1 for even n: the twiddle
pass), and tau(max(m, n)). The whole spectrum of fft of real input is held to the "real" bound (the twiddle
pass's error, mirrored, stays inside it; docs/design/fft.md §4.2).

n-dimensional transforms (milestone FT3, decision FT49; docs/design/fft.md §4.2): one pass per axis, each
the line transform of its length, so the bound adds the axes' level terms (a Bluestein axis or a real axis
with a Bluestein inner length its own term times its rho, as in one dimension) and takes C0 times the
largest rho plus C1 per further axis (bound_nd). ||y|| is s sqrt(N) ||x|| with N the product of the
lengths (rfftn: its half spectrum is part of the whole one); irfftn and hfftn use ||X||_W
(hermitian_norm_nd: the weights of ||.||_H along the last axis, the imaginary parts kept, since the
complex passes come before the real one drops them). The floor tau(N) grows by rho m / n for each
Bluestein axis (the product: an upper bound on the sum of the passes' floors). With one axis bound_nd is
bound's one-dimensional form.

DCT and DST (milestone FT4, decision FT53; docs/design/fft.md §3.5, §4.2): every type runs one inner complex
transform of length r2r_inner(method, n) -- n - 1 (DCT-I, the real FFT of 2 (n - 1)), n + 1 (DST-I), n / 2
or n (types 2 and 3 on the real FFT of n: the real plan for even n, the complex plan for odd n), n / 2 or
n (type 4: the complex FFT of n / 2 for even n, the odd algorithm's real FFT of n for odd n). When it is
31-smooth the bound is the one above (with its kappa); when it is a Bluestein length h the bound is
Bluestein's with h's rho and inner length m: L(M) -> 2 L(m) + 3 (the real pass's level, as the real
transforms' + 1), times rho, and tau(max(m, M)). With orthogonalize the transform is a diagonal sqrt 2
scaling of T whose norm stays below norm_r2r (the orthogonal matrix times sqrt(N'), N' the logical length);
r2r_reference gives the references of every norm and orthogonalize from the stored ones, exactly (Arb).

Bluestein lengths (decisions FT36, FT37, FT64; docs/design/fft.md §4.2): the complex transform of a length n
with a prime factor above 31 is a cyclic convolution of length m, the smallest 2^a 3^b 5^c >= 2n - 1
(bluestein_m), and rho(n) = max_k |DFT_m(b)_k| / sqrt(n) for the chirp kernel b_l = e^(pi i l^2 / n),
|l| < n (bluestein_rho: computed here independently of the library, in double precision, which moves
rho by far less than the bound's slack; the tests compare it with the plan's double-double value).
"""

from __future__ import annotations

import functools
import hashlib
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
KAPPA_WIDE = 17.0  # the same for a length whose largest prime factor is 13 to 31 (decision FT64)
WIDE_PRIMES = (13, 17, 19, 23, 29, 31)
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


def r2r_inner(method: str, n: int) -> int:
    """The length of the complex transform inside the DCT or DST `method` (dct1 .. dst4) of n points
    (csrc/fft/r2r.hpp): see the module's docstring."""
    n = int(n)
    t = int(method[-1])
    if t == 1:
        return n - 1 if method == "dct1" else n + 1
    return n // 2 if n % 2 == 0 else n


def r2r_logical(method: str, n: int) -> int:
    """N', the length the normalisation counts: 2 (n - 1) for dct1, 2 (n + 1) for dst1, 2n otherwise."""
    if method == "dct1":
        return 2 * (int(n) - 1)
    if method == "dst1":
        return 2 * (int(n) + 1)
    return 2 * int(n)


def is_smooth11(n: int) -> bool:
    """Whether every prime factor of n is at most 11: next_fast_len's set, the lengths whose level constant is
    KAPPA."""
    n = int(n)
    for p in (2, 3, 5, 7, 11):
        while n % p == 0:
            n //= p
    return n == 1


def is_mixed(n: int) -> bool:
    """Whether every prime factor of n is at most 31: the lengths the mixed radix transforms directly (decision
    FT64; the others go through Bluestein)."""
    n = int(n)
    for p in (2, 3, 5, 7, 11) + WIDE_PRIMES:
        while n % p == 0:
            n //= p
    return n == 1


def kappa(n: int) -> float:
    """The level constant of a mixed-radix length n: KAPPA when n is 11-smooth, KAPPA_WIDE when a radix of 13 to
    31 takes part (its stage needs up to 16.83u per log2 r, tests/fft/proof.py)."""
    return KAPPA if is_smooth11(n) else KAPPA_WIDE


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


def real_inner(n: int) -> int:
    """The length of the complex transform inside a real or Hermitian transform of length n (decision
    FT43): n / 2 for even n (the real plan), n for odd n (the complex plan and the Hermitian part)."""
    n = int(n)
    return n // 2 if n % 2 == 0 else n


def tau(m: int, k: float = KAPPA) -> float:
    """The underflow floor of a transform of length m (k: the level constant of its factorisation)."""
    return k * m * TINY


def axis_term(n: int, real: bool = False) -> tuple[float, float, float]:
    """(level term, rho, floor length) of one axis of length n: KAPPA L(n), 1 and n for an 11-smooth (inner)
    length (with a radix of 13 to 31: KAPPA_WIDE L(n) and n KAPPA_WIDE / KAPPA); for a Bluestein length its
    KAPPA (2 L(m) + 2) (a real axis: + KAPPA for an even n's twiddle pass), rho and m (a real axis: max(m, n)),
    as bound's one-dimensional forms."""
    n = int(n)
    h = real_inner(n) if real else n
    if is_mixed(h):
        return kappa(h) * levels(n), 1.0, n * kappa(h) / KAPPA
    m, rho = bluestein_m(h), bluestein_rho(h)
    extra = 1 if real and n % 2 == 0 else 0
    return KAPPA * (2 * levels(m) + 2 + extra), rho, (max(m, n) if real else m)


def bound_nd(lengths, xnorm: float, s: float = 1.0, real: bool = False) -> float:
    """The contract's bound on ||y_hat - y|| for an n-dimensional transform of the given lengths (in the
    order of the axes; with real, the last is the real axis of rfftn, irfftn, hfftn and ihfftn), input
    norm xnorm (irfftn: hermitian_norm_nd) and normalisation factor s (see the module's docstring)."""
    lengths = [int(v) for v in lengths]
    terms = [axis_term(v, real and i == len(lengths) - 1) for i, v in enumerate(lengths)]
    count = math.prod(lengths)
    lv = sum(rho * t for t, rho, _ in terms)
    c = max(rho for _, rho, _ in terms) * C0 + (len(lengths) - 1) * C1
    floor = tau(count) * math.prod(rho * m / v for (_, rho, m), v in zip(terms, lengths))
    return (lv + c) * U * s * math.sqrt(count) * xnorm + floor


def hermitian_norm_nd(x, n: int) -> float:
    """||X||_W of an n-dimensional half spectrum x (already cut or padded to n // 2 + 1 along the last axis):
    the 2-norm with the entries of the last axis weighted by 1 (k = 0 and, for even n, k = n / 2) or
    sqrt(2) (the others), imaginary parts kept. With one axis use hermitian_norm (they drop them)."""
    x = np.asarray(x, np.complex128)
    w = np.full(x.shape[-1], math.sqrt(2.0))
    w[0] = 1.0
    if n % 2 == 0 and x.shape[-1] == n // 2 + 1:
        w[-1] = 1.0
    return norm2(x * w)


def bound(kind: str, n, xnorm: float, s: float = 1.0, method: str | None = None) -> float:
    """The contract's bound on ||y_hat - y|| (absolute) for a transform of the given kind:
    "c2c", "real", "hermitian" (xnorm: ||x~||_H), "nd" (n: the shape), or "r2r" (method: dct1 ..
    dst4), with input norm xnorm and normalisation factor s."""
    if kind == "nd":
        return bound_nd(n, xnorm, s)
    n = int(n)
    if kind == "c2c" and not is_mixed(n):
        m, rho = bluestein_m(n), bluestein_rho(n)
        return rho * ((KAPPA * (2 * levels(m) + 2) + C0) * U * s * math.sqrt(n) * xnorm + tau(m))
    if kind in ("real", "hermitian") and not is_mixed(real_inner(n)):
        h = real_inner(n)
        m, rho = bluestein_m(h), bluestein_rho(h)
        extra = 1 if n % 2 == 0 else 0  # the twiddle pass of an even length: one more level
        return rho * ((KAPPA * (2 * levels(m) + 2 + extra) + C0) * U * s * math.sqrt(n) * xnorm + tau(max(m, n)))
    if kind in ("c2c", "real", "hermitian"):
        k = kappa(n if kind == "c2c" else real_inner(n))
        return (k * levels(n) + C0) * U * s * math.sqrt(n) * xnorm + tau(n, k)
    if kind == "r2r":
        M = 2 * n + 2
        h = r2r_inner(method, n)
        if is_mixed(h):
            k = kappa(h)
            return (k * levels(M) + C2) * U * s * norm_r2r(method, n) * xnorm + tau(M, k)
        m, rho = bluestein_m(h), bluestein_rho(h)
        return rho * ((KAPPA * (2 * levels(m) + 3) + C2) * U * s * norm_r2r(method, n) * xnorm + tau(max(m, M)))
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
# The same for the real path (milestone FT2): an even and an odd Bluestein length, 2^20.
LONG_REAL = ("r/uniform/65536", "r/uniform/65537", "r/uniform/1048576")
# The n-dimensional points with Arb computed where they are needed (milestone FT3): the stored nd family
# has the complex transform of 11-smooth shapes only. Complex shapes (fftn), real shapes (rfftn: the real
# parts of the nd signal) and half spectra (irfftn: (shape of the input, output length of the last axis)):
# first the mixed-radix forms (even and odd last lengths, axes and inner lengths of 13 to 31, kappa 17 since
# decision FT64), then Bluestein axes (a prime factor above 31: 37, 41, 43 and 74 = 2 37, as another axis,
# as the last axis and as a real axis's inner length).
ND_KINDS = ("uniform", "wide", "huge", "tiny")
ND_COMPLEX = ((13, 8), (3, 17, 4), (7, 19), (5, 11, 2),
              (37, 4), (3, 41), (74, 2), (2, 43, 3))  # fmt: skip
ND_REAL = ((6, 10), (5, 7), (3, 4, 9), (13, 8), (6, 17), (4, 26), (2, 3, 4, 5), (16, 24),
           (4, 37), (3, 74), (41, 6), (2, 43, 4))  # fmt: skip
ND_HERMITIAN = (((6, 6), 10), ((5, 4), 7), ((3, 4, 5), 9), ((13, 5), 8), ((6, 9), 17), ((4, 14), 26),
                ((2, 3, 4, 3), 5), ((16, 13), 24),
                ((4, 19), 37), ((3, 38), 74), ((43, 4), 6), ((2, 41, 3), 4))  # fmt: skip


def nd_points():
    """(family, signal id, x, s, hi, lo) of the n-dimensional points with Arb references: family "c2c"
    (fftn, unnormalised; also ifftn with norm='forward' via the backward reference), "real" (rfftn: the
    half spectrum) and "hermitian" (irfftn(x, s, norm='forward')). x is the input as the transform sees it
    (for hermitian: n // 2 + 1 entries along the last axis), s the transform's lengths."""
    from . import signals as S

    for kind in ND_KINDS:
        for shape in ND_COMPLEX:
            sig = S.get(S.sid("nd", kind, shape))
            hi, lo = arb_dftn(sig.x)
            yield "c2c", sig.id, sig.x, shape, hi, lo
            hi, lo = arb_dftn(sig.x, inverse=True)
            yield "c2c-backward", sig.id, sig.x, shape, hi, lo
        for shape in ND_REAL:
            sig = S.get(S.sid("nd", kind, shape))
            x = np.ascontiguousarray(sig.x.real)
            hi, lo = arb_dftn(x.astype(np.complex128))
            h = shape[-1] // 2 + 1
            yield "real", sig.id + "/real", x, shape, hi[..., :h], lo[..., :h]
        for shape, n in ND_HERMITIAN:
            sig = S.get(S.sid("nd", kind, shape))
            hi, lo = arb_irfftn(sig.x, n)
            yield "hermitian", sig.id + f"/n={n}", sig.x, shape[:-1] + (n,), hi, lo


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


def _acb_settled(vals, prec_limit_scale):
    from flint import arb

    limit = arb(prec_limit_scale) * arb(2) ** -120
    return all(a.real.rad() <= limit and a.imag.rad() <= limit for a in vals)


def _to_dd(vals, shape):
    """(hi, lo) complex128 arrays of a flat list of acb midpoints (double-double)."""
    from flint import arb

    hi = np.empty(len(vals), np.complex128)
    lo = np.empty(len(vals), np.complex128)
    for k, a in enumerate(vals):
        re, im = a.real.mid(), a.imag.mid()
        h_re, h_im = float(re), float(im)
        hi[k] = complex(h_re, h_im)
        lo[k] = complex(float(re - arb(h_re)), float(im - arb(h_im)))
    return hi.reshape(shape), lo.reshape(shape)


def _acb_dft_axis(a, axis: int, inverse: bool):
    """The unnormalised DFT (sign -, or + with inverse) of the object array a of acb along axis, exact
    (interval arithmetic)."""
    from flint import acb

    moved = np.moveaxis(a, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        v = [z.conjugate() for z in row] if inverse else list(row)
        y = acb.dft(v)
        out[i] = [z.conjugate() for z in y] if inverse else y
    return np.moveaxis(out.reshape(moved.shape), -1, axis)


def _acb_array(x):
    from flint import acb

    x = np.asarray(x, np.complex128)
    a = np.empty(x.shape, dtype=object)
    for idx in np.ndindex(x.shape):
        z = x[idx]
        a[idx] = acb(float(z.real), float(z.imag))
    return a


def _settle(compute):
    """compute() -> object array of acb at the working precision, repeated at doubled precision until every
    entry is settled to 2^-120 of the result's 2-norm (as arb_dft); (hi, lo) of the result."""
    from flint import ctx

    old = ctx.prec
    prec = 192
    try:
        while True:
            ctx.prec = prec
            y = compute()
            flat = list(y.ravel())
            scale = norm2(np.array([complex(float(a.real.mid()), float(a.imag.mid())) for a in flat]))
            if _acb_settled(flat, scale) or prec >= 1536:
                return _to_dd(flat, y.shape)
            prec *= 2
    finally:
        ctx.prec = old


def arb_dftn(x, inverse: bool = False):
    """(hi, lo): the unnormalised n-dimensional DFT over every axis of the complex array x (sign -, or +
    with inverse) by Arb, settled and kept as arb_dft does. For the small shapes of the n-dimensional
    tests and accuracy sweep (milestone FT3)."""

    def compute():
        a = _acb_array(x)
        for axis in range(a.ndim):
            a = _acb_dft_axis(a, axis, inverse)
        return a

    return _settle(compute)


def arb_irfftn(x, n: int):
    """(hi, lo) real parts: irfftn(x, s=(..., n), norm='forward') exactly (unnormalised): the inverse DFT
    over every axis but the last of the half spectrum x (already cut or padded to n // 2 + 1 along the
    last axis), then along the last the inverse DFT of the Hermitian extension of each line with the
    imaginary parts of its DC and (even n) Nyquist entries dropped -- numpy's and scipy's order, which
    decides what a non-Hermitian input loses."""
    x = np.asarray(x, np.complex128)
    h = n // 2 + 1
    assert x.shape[-1] == h

    def compute():
        from flint import acb

        a = _acb_array(x)
        for axis in range(a.ndim - 1):
            a = _acb_dft_axis(a, axis, True)
        flat = a.reshape(-1, h)
        out = np.empty((flat.shape[0], n), dtype=object)
        for i, row in enumerate(flat):
            e = [acb(0)] * n
            e[0] = acb(row[0].real)
            for j in range(1, h):
                z = row[j]
                if 2 * j == n:
                    z = acb(z.real)
                e[j] = z
                if j != n - j:
                    e[n - j] = z.conjugate()
            y = acb.dft([v.conjugate() for v in e])
            out[i] = [acb(v.real) for v in y]  # the conjugate's real part
        return out.reshape(x.shape[:-1] + (n,))

    hi, lo = _settle(compute)
    return hi.real, lo.real


# The DCT and DST lines longer than the files hold or with inner lengths the files lack (milestone FT4):
# a prime (Bluestein inner lengths for every type), 2 1009, and 2^16 and 65537 for the accuracy sweep.
LONG_R2R = ("r/uniform/1009", "r/uniform/2018", "r/uniform/65536", "r/uniform/65537")


def _r2r_extension(xs, method):
    """(the extension's acb entries, how to read y_k from its DFT d): the DFT of an even or odd extension
    (docs/design/fft.md §7.2; tools/fft_ref.py writes the stored references the same way)."""
    from flint import arb

    N = len(xs)
    zero = arb(0)
    if method == "dct1":
        v = xs + xs[-2:0:-1]
        return v, lambda d: [a.real for a in d[:N]]
    if method == "dst1":
        v = [zero] + xs + [zero] + [-a for a in xs[::-1]]
        return v, lambda d: [-a.imag for a in d[1 : N + 1]]
    if method in ("dct2", "dst2"):
        sign = 1 if method == "dct2" else -1
        v = [zero] * (4 * N)
        for i, a in enumerate(xs):
            v[2 * i + 1] = a
            v[4 * N - 2 * i - 1] = a if sign > 0 else -a
        if sign > 0:
            return v, lambda d: [a.real for a in d[:N]]
        return v, lambda d: [-a.imag for a in d[1 : N + 1]]
    if method == "dct3":
        v = [zero] * (4 * N)
        v[0] = xs[0]
        for i in range(1, N):
            v[i] = xs[i]
            v[4 * N - i] = xs[i]
        return v, lambda d: [d[2 * k + 1].real for k in range(N)]
    if method == "dst3":
        v = [zero] * (4 * N)
        for i in range(N):
            a = xs[i] / 2 if i == N - 1 else xs[i]
            v[i + 1] += a
            v[4 * N - i - 1] += -a
        return v, lambda d: [-d[2 * k + 1].imag for k in range(N)]
    sign = 1 if method == "dct4" else -1
    v = [zero] * (8 * N)
    for i, a in enumerate(xs):
        v[2 * i + 1] = a
        v[8 * N - 2 * i - 1] = a if sign > 0 else -a
    if sign > 0:
        return v, lambda d: [d[2 * k + 1].real for k in range(N)]
    return v, lambda d: [-d[2 * k + 1].imag for k in range(N)]


def arb_r2r(x, method: str):
    """(hi, lo) float64: the unnormalised DCT or DST `method` of the real line x (scipy's norm='backward'
    definitions) by Arb's acb.dft of its extension, settled and kept as arb_dft does. For the lines of
    LONG_R2R (a second or so at 2^16)."""
    from flint import acb, arb

    x = np.asarray(x, np.float64)

    def compute():
        xs = [arb(float(a)) for a in x]
        v, read = _r2r_extension(xs, method)
        y = read(acb.dft([acb(a) for a in v]))
        out = np.empty(len(y), dtype=object)
        for k, a in enumerate(y):
            out[k] = acb(a)
        return out

    hi, lo = _settle(compute)
    return hi.real, lo.real


def r2r_reference(method: str, x, hi, lo, s: float = 1.0, ortho: bool = False):
    """(hi, lo) float64 of s times the transform with orthogonalize's exact sqrt 2 when ortho, from the
    unnormalised reference hi + lo of `method` on x, in Arb: the transform is linear, so the sqrt 2 on an
    input entry adds (sqrt 2 - 1) x_j times the column T e_j, which is exact for the entries scipy scales
    (all ones for x_0 of dct1 and dct3, (-1)^k for x_(N-1) of dct1 and dst3), and the result's entry is
    divided by sqrt 2 (dct1: y_0 and y_(N-1), dct2: y_0, dst2: y_(N-1)). s is the normalisation factor
    (a double)."""
    from flint import arb, ctx

    n = len(x)
    old = ctx.prec
    ctx.prec = 256
    try:
        y = [arb(float(h)) + arb(float(l)) for h, l in zip(np.asarray(hi), np.asarray(lo))]
        if ortho:
            r2 = arb(2).sqrt()
            cols = []  # (input index, column T e_j as a function of k)
            if method in ("dct1", "dct3"):
                cols.append((0, lambda k: 1))
            if method == "dct1":
                cols.append((n - 1, lambda k: -1 if k % 2 else 1))
            if method == "dst3":
                cols.append((n - 1, lambda k: -1 if k % 2 else 1))
            for j, col in cols:
                add = (r2 - 1) * arb(float(x[j]))
                y = [v + add * col(k) for k, v in enumerate(y)]
            outs = {"dct1": (0, n - 1), "dct2": (0,), "dst2": (n - 1,)}.get(method, ())
            for k in set(outs):
                y[k] = y[k] / r2
        y = [v * arb(float(s)) for v in y]
        out_hi = np.array([float(v) for v in y])
        out_lo = np.array([float(v - arb(float(h))) for v, h in zip(y, out_hi)])
        return out_hi, out_lo
    finally:
        ctx.prec = old


# ---------------------------------------------------------------- FFTLog (milestone FT5)
#
# fht and ifht (docs/design/fft.md §3.9, §4.2; decision FT58): A = B_out P R(C X) with X the real DFT's half
# spectrum of v = B_in a, C the coefficients over n, R the unnormalised inverse real DFT of a Hermitian half
# spectrum, P the reversal. The contract:
#
#     ||A_hat - A|| <= (2 eps_F + C_FHT) u ||B_out||_inf M ||B_in a|| + (||B_out||_inf (M + 1) + 1) 2 tau_F
#
# with eps_F the real transform's level term in units of u (KAPPA L(n) + C0, or the Bluestein form of the inner
# length: rho (KAPPA (2 L(m) + 2 + [n even]) + C0)), tau_F its floor, and M = max_m |C_m| n (fht: max |U_m|,
# ifht: max 1 / |U_m|, over the effective coefficients). arb_fht computes A exactly from the double
# parameters (Arb's acb.gamma and acb.dft); fht_coefficients the coefficients themselves.
C_FHT = 8.0


def _fht_half(mu: float, q: float, sign: int):
    from fractions import Fraction

    return (Fraction(mu) + 1 + sign * Fraction(q)) / 2


def _is_pole(x) -> bool:
    return x.denominator == 1 and x <= 0


def _fht_acb_coefficients(n: int, dln: float, mu: float, offset: float, bias: float, inverse: bool):
    """[C_m n for m = 0 .. n // 2] as acb at the current precision: U_m (fht) or 1 / conj U_m (ifht), the real
    entries (m = 0, even n's n / 2) Re U_m or 1 / Re U_m, a singular entry 0 (see csrc/fft/fftlog.hpp)."""
    import math as _m

    from flint import acb, arb

    xp_f, xm_f = _fht_half(mu, bias, 1), _fht_half(mu, bias, -1)
    xp = arb(xp_f.numerator) / xp_f.denominator
    xm = arb(xm_f.numerator) / xm_f.denominator
    q, d, lnkr = arb(bias), arb(dln), arb(offset)
    ln2 = arb(2).log()
    out = []
    for m in range(n // 2 + 1):
        real = m == 0 or 2 * m == n
        if m == 0 and (_is_pole(xp_f) or _is_pole(xm_f)):
            if not (_is_pole(xp_f) and _is_pole(xm_f)):
                out.append(acb(0))
                continue
            j, k = int(-xp_f), int(-xm_f)
            u = arb(2) ** q * arb(_m.factorial(k)) / arb(_m.factorial(j)) * (-1) ** (j + k)
            out.append(acb(1 / u if inverse else u))
            continue
        y = arb.pi() * m / (n * d)
        u = (acb(2) ** acb(q, 2 * y)) * acb(xp, y).gamma() * acb(xm, -y).rgamma() * acb(0, -2 * y * lnkr).exp()
        if real:
            ur = u.real
            out.append(acb(1 / ur if inverse else ur))
        else:
            out.append(1 / u.conjugate() if inverse else u)
    del ln2
    return out


def fht_coefficients(n: int, dln: float, mu: float, offset: float = 0.0, bias: float = 0.0, inverse: bool = False):
    """(hi, lo) complex128: the plan's coefficients C_m = U_m / n (fht) or 1 / (n conj U_m) (ifht), m = 0 .. n // 2,
    exactly (Arb), each settled to 2^-120 of its own size."""
    from flint import acb, arb, ctx

    old = ctx.prec
    prec = 192
    try:
        while True:
            ctx.prec = prec
            c = [v / n for v in _fht_acb_coefficients(n, dln, mu, offset, bias, inverse)]
            sizes = [arb(abs(complex(float(v.real.mid()), float(v.imag.mid())))) for v in c]
            ok = all(max(v.real.rad(), v.imag.rad()) <= z * arb(2) ** -120 for v, z in zip(c, sizes))
            if ok or prec >= 1536:
                break
            prec *= 2
        hi, lo = _to_dd([acb(v) for v in c], (len(c),))
        return hi, lo
    finally:
        ctx.prec = old


def _fht_bias_exponents(n: int, dln: float, offset: float, bias: float, inverse: bool):
    """The exact exponents (acb-free arb) of the input and output biases at j = 0 .. n - 1 (lists)."""
    from flint import arb

    q, d, lnkr = arb(bias), arb(dln), arb(offset)
    jc = arb(n - 1) / 2
    ks = [arb(j) - jc for j in range(n)]
    if inverse:
        return [q * (k * d + lnkr) for k in ks], [q * k * d for k in ks]
    return [-q * k * d for k in ks], [-q * (k * d + lnkr) for k in ks]


def arb_fht(a, dln: float, mu: float, offset: float = 0.0, bias: float = 0.0, inverse: bool = False):
    """(hi, lo) float64: fht(a, dln, mu, offset=offset, bias=bias) (ifht with inverse) of the real line a exactly,
    from the double parameters (Arb), settled and kept as arb_dft does."""
    from flint import acb, arb

    a = np.asarray(a, np.float64)
    n = len(a)

    def compute():
        ein, eout = _fht_bias_exponents(n, dln, offset, bias, inverse)
        v = [acb(arb(float(x)) * e.exp()) for x, e in zip(a, ein)]
        X = acb.dft(v)
        c = _fht_acb_coefficients(n, dln, mu, offset, bias, inverse)
        W = [acb(0)] * n
        for k in range(n):
            ck = c[k] if k <= n // 2 else c[n - k].conjugate()
            W[k] = X[k] * ck
        y = acb.dft([w.conjugate() for w in W])  # n times the inverse DFT, conjugated
        out = np.empty(n, dtype=object)
        for j in range(n):
            out[j] = acb(y[n - 1 - j].real / n * eout[j].exp())
        return out

    hi, lo = _settle(compute)
    return hi.real, lo.real


def real_level(n: int) -> tuple[float, float]:
    """(eps_F / u, tau_F) of the real transform of length n: the contract's level term and floor (bound's "real"
    form)."""
    n = int(n)
    h = real_inner(n)
    if is_mixed(h):
        return kappa(h) * levels(n) + C0, tau(n, kappa(h))
    m, rho = bluestein_m(h), bluestein_rho(h)
    extra = 1 if n % 2 == 0 else 0
    return rho * (KAPPA * (2 * levels(m) + 2 + extra) + C0), rho * tau(max(m, n))


def fht_terms(a, dln: float, mu: float, offset: float = 0.0, bias: float = 0.0, inverse: bool = False):
    """(||B_out||_inf, M, ||B_in a||) of the contract, in double precision from the exact exponents and
    coefficients (rounding moves them by far less than the bound's slack)."""
    from flint import ctx

    a = np.asarray(a, np.float64)
    n = len(a)
    old = ctx.prec
    ctx.prec = 128
    try:
        ein, eout = _fht_bias_exponents(n, dln, offset, bias, inverse)
        bo = max(math.exp(float(e)) for e in eout)
        v = np.array([float(x) * math.exp(float(e)) for x, e in zip(a, ein)])
        M = max(abs(complex(float(c.real.mid()), float(c.imag.mid())))
                for c in _fht_acb_coefficients(n, dln, mu, offset, bias, inverse))  # fmt: skip
    finally:
        ctx.prec = old
    return bo, M, norm2(v)


def bound_fht(a, dln: float, mu: float, offset: float = 0.0, bias: float = 0.0, inverse: bool = False) -> float:
    """The contract's bound on ||A_hat - A|| for fht (ifht with inverse) of the real line a."""
    eps, tf = real_level(len(a))
    bo, M, vn = fht_terms(a, dln, mu, offset, bias, inverse)
    return (2 * eps + C_FHT) * U * bo * M * vn + (bo * (M + 1) + 1) * 2 * tf


# The parameter sets of the FFTLog points (tests/fft/test_fftlog.py and bench/accuracy/fft.py's fftlog sweep):
# (name, dln or None for the logspace(-4, 4, n) spacing, mu, offset, bias). unbiased: |U_m| = 1; biased: the
# biases and |U_m| != 1; the poles of Gamma at y = 0 (both: the finite limit; x+ alone: fht singular; x- alone:
# ifht singular); a large order; a small spacing (y up to 15708: the phases double precision loses).
FHT_PARAMS = (
    ("unbiased", 0.01, 0.5, 0.0, 0.0),
    ("logspace", None, 0.3, -6 * math.log(10), 0.0),
    ("reflection", 0.1, -0.7, 1.0, 0.4),
    ("negative-dln", -0.3, 1.5, 0.2, -0.6),
    ("poles-both", 0.05, -4.0, 0.0, 1.0),
    ("pole-plus", 0.05, -3.5, 0.0, 0.5),
    ("pole-minus", 0.05, -2.5, 0.0, 0.5),
    ("large-mu", 0.02, 50.5, 0.0, 0.0),
    ("small-dln", 1e-4, 0.0, 0.0, 0.0),
)
FHT_LENGTHS = (1, 2, 3, 16, 63, 64, 127, 1000, 1009, 2018, 4096)


def fht_dln(dln, n: int) -> float:
    return math.log(1e8) / (n - 1) if dln is None else dln


def fht_points(lengths=FHT_LENGTHS, values=("normal",)):
    """(id, a, dln, mu, offset, bias) of the FFTLog points: every length and parameter set (skipping those whose
    biases would overflow: |bias dln| n / 2 above 300), on standard normal values (seeded by the id), the
    physical r^(mu + 1) e^(-r^2 / 2) on logspace(-4, 4, n) ("gauss"), or the normal values times 2^-1060
    ("tiny", subnormal) or 2^1000 / n ("huge")."""
    for n in lengths:
        for name, dln0, mu, offset, bias in FHT_PARAMS:
            if dln0 is None and n < 2:
                continue
            dln = fht_dln(dln0, n)
            if abs(bias * dln) * n / 2 > 300:
                continue
            for value in values:
                if value in ("tiny", "huge") and bias != 0:
                    continue
                seed = int(hashlib.sha256(f"{name}/{value}/{n}".encode()).hexdigest()[:16], 16)
                g = np.random.Generator(np.random.PCG64(seed))
                if value == "gauss":
                    r = np.logspace(-4, 4, n)
                    a = r ** (mu + 1) * np.exp(-(r**2) / 2)
                else:
                    a = g.standard_normal(n)
                    if value == "tiny":
                        a = a * 2.0**-1060
                    elif value == "huge":
                        a = a * (2.0**1000 / n)
                yield f"{name}/{value}/{n}", a, dln, mu, offset, bias


def arb_fhtoffset(dln: float, mu: float, initial: float = 0.0, bias: float = 0.0) -> float:
    """scipy's fhtoffset exactly from the double parameters (Arb; the rounding of arg half to even), as the
    nearest double."""
    from flint import acb, arb, ctx

    old = ctx.prec
    ctx.prec = 256
    try:
        xp_f, xm_f = _fht_half(mu, bias, 1), _fht_half(mu, bias, -1)
        xp = arb(xp_f.numerator) / xp_f.denominator
        xm = arb(xm_f.numerator) / xm_f.denominator
        y = arb.pi() / (2 * arb(dln))
        zp, zm = acb(xp, y).lgamma(), acb(xm, y).lgamma()
        arg = (arb(2).log() - arb(initial)) / arb(dln) + (zp.imag + zm.imag) / arb.pi()
        mid = float(arg.mid())
        r = round(mid)  # Python's round: half to even
        f = arg - r
        if f > 0.5:
            f -= 1
        elif f < -0.5:
            f += 1
        return float(arb(initial) + f * arb(dln))
    finally:
        ctx.prec = old
