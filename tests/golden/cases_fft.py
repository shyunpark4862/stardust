"""Golden cases for sd.fft (tools/golden.py has the contract).

The one-dimensional complex transforms and the helpers (milestone FT1, docs/design/fft.md §8): fft and
ifft on every length form (2^k, f 2^k, the other 11-smooth lengths with radices 7 and 11, Bluestein for
prime and composite lengths with a large prime factor) times norm, the edge values (NaN, inf, signed
zeros, huge and subnormal entries), cutting and zero-padding, every axis of 2-d and 3-d arrays in C, F
and strided layouts, batches large enough to split into many tasks at 3 and 10 threads, one long line,
the empty batch axis and length 1; fftfreq and rfftfreq (numpy's bits), fftshift and ifftshift on every
dtype and axes form, next_fast_len and prev_fast_len; the wrong calls (their errors are results).

Real input to fft (bool, int64, float64) is left out until milestone FT2 gives it the real path (decision
FT16; FT1 computes it as a complex input with zero imaginary parts, decision FT38), which changes its
bits. The four-step split of one long line comes with milestone FT6.

The inputs come from fixed literals and L.rng(seed); the largest result is 2^17 complex numbers.
"""

import math

import numpy as np

NAN, INF = math.nan, math.inf
NORMS = ("backward", "ortho", "forward")
# 2^k, f 2^k, other 11-smooth (7, 11, 9 = 3 3, 25 = 5 5, 77 = 7 11), Bluestein (primes, a large prime
# factor, a prime power)
LENGTHS = (1, 2, 3, 4, 5, 7, 8, 11, 16, 24, 40, 64, 120, 1024, 3072, 9, 25, 77, 1000, 2310, 13, 17, 127, 1009,
           2018, 169, 4097)  # fmt: skip


def _c(r, shape):
    return r.uniform(-1, 1, shape) + 1j * r.uniform(-1, 1, shape)


def cases(L):
    F = L.sd.fft
    a = L.arr
    r = L.rng(2311)

    # ---- every length form, both directions, every norm
    for n in LENGTHS:
        x = _c(r, n)
        for norm in NORMS:
            yield f"fft n={n} {norm}", lambda x=x, norm=norm: F.fft(a(x), norm=norm)
            yield f"ifft n={n} {norm}", lambda x=x, norm=norm: F.ifft(a(x), norm=norm)

    # ---- the edge values
    edge = np.array([1.0 + 2j, NAN + 0j, complex(0.0, INF), -0.0 - 0.0j, 1e300 + 1e300j, 5e-324 - 1e-310j,
                     complex(-INF, 1.0), 3.0 - 0.5j])  # fmt: skip
    for n in (8, 7, 13):
        e = edge[:n] if n <= 8 else np.concatenate([edge, edge[: n - 8]])
        yield f"fft edge n={n}", lambda e=e: F.fft(a(e))
        yield f"ifft edge n={n}", lambda e=e: F.ifft(a(e), norm="ortho")
    yield "fft huge", lambda: F.fft(a(_c(L.rng(2312), 64) * 2.0**1010))
    yield "fft subnormal", lambda: F.fft(a(_c(L.rng(2313), 127) * 2.0**-1060))
    yield "fft zeros", lambda: F.fft(a(np.zeros(12, np.complex128)))
    yield "fft length 1", lambda: F.fft(a(np.array([2.5 - 1j])))
    yield "ifft length 1 forward", lambda: F.ifft(a(np.array([complex(-0.0, NAN)])), norm="forward")

    # ---- cutting and padding, axes, layouts
    m = _c(r, (6, 10))
    for n in (1, 4, 10, 13, 16):
        yield f"fft cut/pad n={n} axis=-1", lambda n=n: F.fft(a(m), n=n)
        yield f"ifft cut/pad n={n} axis=0", lambda n=n: F.ifft(a(m), n=n, axis=0)
    t = _c(r, (5, 6, 7))
    for axis in (0, 1, 2, -1):
        yield f"fft 3-d axis={axis}", lambda axis=axis: F.fft(a(t), axis=axis)
        yield f"ifft 3-d axis={axis} F order", lambda axis=axis: F.ifft(a(np.asfortranarray(t)), axis=axis)
    yield "fft strided", lambda: F.fft(a(m[::-2, 1::3]), axis=0)
    yield "fft empty batch", lambda: F.fft(a(np.zeros((0, 4), np.complex128)))
    yield "fft empty batch padded", lambda: F.fft(a(np.zeros((3, 0), np.complex128)), n=5)

    # ---- batches that split into many tasks, and one long line
    big = _c(L.rng(2314), (300, 256))
    yield "fft batch 300x256 axis=-1", lambda: F.fft(a(big))
    yield "fft batch 300x256 axis=0", lambda: F.fft(a(big), axis=0)
    yield "ifft batch 64x1009 axis=-1", lambda: F.ifft(a(_c(L.rng(2315), (64, 1009))))
    yield "fft batch 40x1000 axis=0 ortho", lambda: F.fft(a(_c(L.rng(2316), (1000, 40))), axis=0, norm="ortho")
    yield "fft long 2^17", lambda: F.fft(a(_c(L.rng(2317), 1 << 17)))
    yield "ifft long 65537", lambda: F.ifft(a(_c(L.rng(2318), 65537)))

    # ---- helpers
    for n in (1, 2, 5, 8, 1001):
        for d in (1.0, 0.1, -0.5, 3):
            yield f"fftfreq n={n} d={d}", lambda n=n, d=d: F.fftfreq(n, d=d)
            yield f"rfftfreq n={n} d={d}", lambda n=n, d=d: F.rfftfreq(n, d=d)
    s2 = np.arange(12).reshape(3, 4)
    for name, x in (("int64", s2), ("float64", s2 * 0.5), ("complex128", s2 + 1j), ("bool", s2 % 3 == 0)):
        for axes in (None, 0, (1,), [0, 1], ()):
            yield f"fftshift {name} axes={axes}", lambda x=x, axes=axes: F.fftshift(a(x), axes=axes)
            yield f"ifftshift {name} axes={axes}", lambda x=x, axes=axes: F.ifftshift(a(x), axes=axes)
    yield "fftshift odd 1-d", lambda: F.fftshift(a(np.arange(7.0)))
    yield "next_fast_len 1..300", lambda: [F.next_fast_len(t) for t in range(1, 301)]
    yield "next_fast_len real 1..300", lambda: [F.next_fast_len(t, real=True) for t in range(1, 301)]
    yield "prev_fast_len 1..300", lambda: [F.prev_fast_len(t) for t in range(1, 301)]
    yield "prev_fast_len real 1..300", lambda: [F.prev_fast_len(t, real=True) for t in range(1, 301)]
    yield "fast lengths large", lambda: [F.next_fast_len(10**15 + 1), F.prev_fast_len(10**15 + 1, real=True),
                                         F.prev_fast_len(2**63 - 1)]  # fmt: skip

    # ---- wrong calls
    x4 = a(np.arange(4.0) + 0j)
    yield "fft list", lambda: F.fft([1.0, 2.0])
    yield "fft n=0", lambda: F.fft(x4, n=0)
    yield "fft n=True", lambda: F.fft(x4, n=True)
    yield "fft axis out of range", lambda: F.fft(x4, axis=1)
    yield "fft norm=None", lambda: F.fft(x4, norm=None)
    yield "ifft norm=bad", lambda: F.ifft(x4, norm="bad")
    yield "fft length 0", lambda: F.fft(a(np.zeros(0, np.complex128)))
    yield "fft positional n", lambda: F.fft(x4, 4)
    yield "fft workers", lambda: F.fft(x4, workers=2)
    yield "fftfreq n=0", lambda: F.fftfreq(0)
    yield "fftfreq d=0", lambda: F.fftfreq(4, d=0.0)
    yield "fftshift repeated axes", lambda: F.fftshift(a(s2), axes=(0, 0))
    yield "next_fast_len 0", lambda: F.next_fast_len(0)
    yield "next_fast_len True", lambda: F.next_fast_len(True)
    yield "next_fast_len beyond int64", lambda: F.next_fast_len(2**63 - 1)
