"""The test signals of sd.fft (docs/design/fft.md §7.1): made from formulas and fixed seeds, nothing
downloaded. The tests, the reference tool (tools/fft_ref.py), the probe of numpy, scipy, R and vDSP
(tools/fft_probe.py) and the measurements take their signals from here, by id.

A signal id is "<domain>/<kind>/<length>":

    domain   c   complex128 of length n (the input of fft, ifft)
             r   float64 of length n (rfft, the real path of fft, dct, dst)
             h   complex128 of length n // 2 + 1, the half spectrum of an output of length n (irfft,
                 hfft); its DC and (even n) Nyquist entries have a nonzero imaginary part the
                 transforms must drop (docs/design/fft.md §2.3)
             nd  complex128 of a shape written "AxBxC" (fftn)
    kind     uniform     real and imaginary parts uniform in [-1, 1)
             normal      standard normal parts
             impulse     one 1 at position n // 3, zeros elsewhere
             constant    every entry 1 (+ 1j): most of the spectrum is exactly 0
             sine        exp(2 pi i k j / n) (real: cos) at k = n // 4, an integer frequency
             sine_half   the same at k = n // 4 + 1/2 (between two bins)
             wide        magnitudes 10^U(-300, 300) with random signs: the norm contract does not
                         see the small entries
             huge        uniform scaled so that ||x||_1 is about 2^1020 (the edge of the overflow
                         domain of docs/design/fft.md §4.2)
             tiny        uniform times 2^-1060 (subnormal inputs and results)

Lengths, by group (LENGTHS): every n in 1..128, the edges of vDSP's length set and of 2^k, primes
and composites with a large prime factor, 7 2^k and 11-smooth lengths, 2^k and f 2^k (f = 3, 5, 15)
up to 2^20. `get(id)` builds any id (cached), so a group may name lengths nobody stores.

The non-finite cases (`nonfinite(n)`: NaN or inf in each position, real and complex) and the
batch layouts (`layouts()`: the transform axis first, in the middle and last, C and F order,
negative and gapped strides, an empty batch axis) are generators of their own.

    import tests.fft.signals as S
    s = S.get("c/uniform/1024")     # cached; S.ids(...) lists the ids of a family
    s.x, s.n, s.domain, s.kind
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math

import numpy as np

SEED = 20261123

KINDS = ("uniform", "normal", "impulse", "constant", "sine", "sine_half", "wide", "huge", "tiny")
DOMAINS = ("c", "r", "h", "nd")


def _pow2(lo, hi):
    return [1 << k for k in range(lo, hi + 1)]


# The length groups of docs/design/fft.md §7.1. "fast" lengths are vDSP's classes (2^k, f 2^k with
# f in {3, 5, 15}): the classes of decision FT22's measurement.
LENGTHS = {
    "all128": list(range(1, 129)),
    "edge": [23, 24, 25, 39, 40, 41, 47, 48, 49, 119, 120, 121, 239, 240, 241, 255, 257, 511, 513, 1023, 1025, 2047,
             2049, 4095, 4097],
    "prime": [127, 1009, 10007, 65537],
    "big_prime_factor": [2 * 1009, 3 * 10007, 2 * 127, 4 * 257],
    "smooth7": [7 * (1 << k) for k in range(3, 11)],
    "smooth11": [99, 500, 539, 1000, 2310, 4620, 9240],
    "pow2": _pow2(0, 20),
    "f3": [3 << k for k in range(3, 19)],
    "f5": [5 << k for k in range(3, 18)],
    "f15": [15 << k for k in range(3, 17)],
}  # fmt: skip

# The shapes of the n-dimensional signals (domain nd).
SHAPES = ((8, 8), (3, 5), (16, 24), (1, 7), (7, 1), (4, 6, 10), (64, 64), (2, 3, 4, 5), (128, 8))


@dataclasses.dataclass(frozen=True)
class Signal:
    id: str
    x: np.ndarray
    domain: str
    kind: str
    n: int  # the transform length (h: the output length; nd: the number of entries)
    note: str = ""


def hash_int(k) -> int:
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "little")


def rng(*key) -> np.random.Generator:
    """The generator of a key: the same numbers on every machine (PCG64 of SEED and the key)."""
    return np.random.Generator(np.random.PCG64([SEED, *[hash_int(k) for k in key]]))


def digest(*arrays) -> str:
    """SHA-256 of arrays as given (dtype, shape and bytes): the records and references name the
    signals they were made from."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(repr((str(a.dtype), a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- builders


def _values(kind: str, size: int, complex_: bool, key) -> np.ndarray:
    """`size` entries of a kind (complex128 or float64), from the generator of `key`."""
    g = rng(kind, *key)

    def parts(f):
        re = f()
        return re + 1j * f() if complex_ else re

    if kind == "uniform":
        return parts(lambda: g.uniform(-1.0, 1.0, size))
    if kind == "normal":
        return parts(lambda: g.standard_normal(size))
    if kind == "impulse":
        x = np.zeros(size, np.complex128 if complex_ else np.float64)
        x[size // 3] = 1.0
        return x
    if kind == "constant":
        return np.full(size, 1.0 + 1.0j if complex_ else 1.0)
    if kind in ("sine", "sine_half"):
        k = size // 4 + (0.5 if kind == "sine_half" else 0.0)
        ang = 2 * np.pi * k * np.arange(size) / size
        return np.exp(1j * ang) if complex_ else np.cos(ang)
    if kind == "wide":
        return parts(lambda: g.choice([-1.0, 1.0], size) * 10.0 ** g.uniform(-300, 300, size))
    if kind == "huge":
        scale = math.ldexp(1.0, 1020) / (2 * size)  # ||x||_1 <= 2 size * scale = 2^1020
        return parts(lambda: g.uniform(-1.0, 1.0, size)) * scale
    if kind == "tiny":
        return parts(lambda: g.uniform(-1.0, 1.0, size)) * math.ldexp(1.0, -1060)
    raise KeyError(kind)


def _half_spectrum(kind: str, n: int) -> np.ndarray:
    """n // 2 + 1 complex entries: the input of irfft(x, n=n). Built as a complex signal of that
    length, so DC and (even n) Nyquist keep an imaginary part (it must be dropped)."""
    m = n // 2 + 1
    x = _values(kind, m, True, ("h", n))
    if kind in ("impulse", "constant"):
        x = x + 0.5j  # a nonzero imaginary part everywhere, DC and Nyquist included
    return x


def _parse(sid: str):
    parts = sid.split("/")
    if len(parts) != 3 or parts[0] not in DOMAINS or parts[1] not in KINDS:
        raise KeyError(f"not a signal id: {sid!r}")
    domain, kind, size = parts
    if domain == "nd":
        shape = tuple(int(s) for s in size.split("x"))
        if not shape or min(shape) < 1:
            raise KeyError(sid)
        return domain, kind, shape
    n = int(size)
    if n < 1:
        raise KeyError(sid)
    return domain, kind, n


@functools.cache
def get(sid: str) -> Signal:
    domain, kind, size = _parse(sid)
    if domain == "nd":
        count = math.prod(size)
        x = _values(kind, count, True, ("nd", *size)).reshape(size)
        x.setflags(write=False)
        return Signal(sid, x, domain, kind, count)
    n = size
    if domain == "c":
        x = _values(kind, n, True, ("c", n))
    elif domain == "r":
        x = _values(kind, n, False, ("r", n))
    else:
        x = _half_spectrum(kind, n)
    x.setflags(write=False)
    return Signal(sid, x, domain, kind, n)


def sid(domain: str, kind: str, size) -> str:
    if domain == "nd":
        return f"nd/{kind}/" + "x".join(str(s) for s in size)
    return f"{domain}/{kind}/{size}"


def lengths(*groups: str) -> list[int]:
    """The sorted union of length groups."""
    out: set[int] = set()
    for g in groups:
        out.update(LENGTHS[g])
    return sorted(out)


# ---------------------------------------------------------------- non-finite values and layouts


def nonfinite(n: int):
    """(id, x) for every position j < n and value v in (nan, inf, -inf): a uniform signal with
    x[j] = v, real (float64) and complex (the real part, the imaginary part, both)."""
    base_r = _values("uniform", n, False, ("nonfinite", n))
    base_c = _values("uniform", n, True, ("nonfinite", n))
    for j in range(n):
        for name, v in (("nan", math.nan), ("inf", math.inf), ("-inf", -math.inf)):
            x = base_r.copy()
            x[j] = v
            yield f"r/{name}/{n}/{j}", x
            for part in ("re", "im", "both"):
                z = base_c.copy()
                re, im = z[j].real, z[j].imag
                if part in ("re", "both"):
                    re = v
                if part in ("im", "both"):
                    im = v
                z[j] = complex(re, im)
                yield f"c/{name}-{part}/{n}/{j}", z


def layouts(n: int = 12, batch: int = 5):
    """(id, array, axis) of the same lines in other layouts: a (batch, n) complex array with the
    transform axis last, first and in the middle of a 3-D array, in C and F order, with negative
    strides (batch axis, transform axis), gapped, and an empty batch axis. In "middle/C" the lines
    are batch[i, :, b] = lines[b] for both i; elsewhere a line holds the values of lines[b]. Every
    line is a uniform signal; tests check that a line's result is the same bytes in every layout."""
    lines = _values("uniform", batch * n, True, ("layouts", n, batch)).reshape(batch, n)
    yield "last/C", np.ascontiguousarray(lines), 1
    yield "last/F", np.asfortranarray(lines), 1
    yield "first/C", np.ascontiguousarray(lines.T), 0
    yield "middle/C", np.ascontiguousarray(np.broadcast_to(lines.T[None], (2, n, batch))), 1
    yield "last/negative-batch-stride", lines[::-1], 1
    yield "last/negative-axis-stride", np.ascontiguousarray(lines[:, ::-1])[:, ::-1], 1
    wide = np.zeros((batch, 3 * n), np.complex128)
    wide[:, ::3] = lines
    yield "last/gapped", wide[:, ::3], 1
    yield "last/empty-batch", np.zeros((0, n), np.complex128), 1
