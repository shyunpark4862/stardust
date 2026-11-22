"""Golden cases for sd.fft (tools/golden.py has the contract).

The one-dimensional complex transforms and the helpers (milestone FT1, docs/design/fft.md §8): fft and
ifft on every length form (2^k, f 2^k, the other 11-smooth lengths with radices 7 and 11, Bluestein for
prime and composite lengths with a large prime factor) times norm, the edge values (NaN, inf, signed
zeros, huge and subnormal entries), cutting and zero-padding, every axis of 2-d and 3-d arrays in C, F
and strided layouts, batches large enough to split into many tasks at 3 and 10 threads, one long line,
the empty batch axis and length 1; fftfreq and rfftfreq (numpy's bits), fftshift and ifftshift on every
dtype and axes form, next_fast_len and prev_fast_len; the wrong calls (their errors are results).

The real and Hermitian transforms (milestone FT2): rfft, irfft, hfft and ihfft on every length form, even
and odd (an even length runs the complex plan of n / 2 and a twiddle pass, an odd one the complex plan of n
and the Hermitian part, decision FT43; Bluestein inner lengths both ways) times norm, the dropped parts of
a Hermitian input, the default lengths, cutting and padding, the edge values, the layouts and batches, a
long line, the input dtypes and the wrong calls; and real input to fft and ifft (bool, int64, float64),
which takes the real path since FT2 (decision FT16; FT1 left it out, decision FT38). The four-step split
of one long line comes with milestone FT6.

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

    # ---- the real and Hermitian transforms (FT2), and real input to fft and ifft
    x4 = a(np.arange(4.0) + 0j)
    for n in LENGTHS + (6, 10, 26, 34, 2046):
        x = r.uniform(-1, 1, n)
        h = _c(r, n // 2 + 1)
        for norm in NORMS:
            yield f"rfft n={n} {norm}", lambda x=x, norm=norm: F.rfft(a(x), norm=norm)
            yield f"ihfft n={n} {norm}", lambda x=x, norm=norm: F.ihfft(a(x), norm=norm)
            yield f"fft real n={n} {norm}", lambda x=x, norm=norm: F.fft(a(x), norm=norm)
            yield f"ifft real n={n} {norm}", lambda x=x, norm=norm: F.ifft(a(x), norm=norm)
            yield f"irfft n={n} {norm}", lambda h=h, n=n, norm=norm: F.irfft(a(h), n=n, norm=norm)
            yield f"hfft n={n} {norm}", lambda h=h, n=n, norm=norm: F.hfft(a(h), n=n, norm=norm)
    redge = np.array([1.0, NAN, INF, -0.0, 1e300, 5e-324, -INF, 3.0])
    for n in (8, 7, 13):
        e = redge[:n] if n <= 8 else np.concatenate([redge, redge[: n - 8]])
        yield f"rfft edge n={n}", lambda e=e: F.rfft(a(e))
        yield f"fft real edge n={n}", lambda e=e: F.fft(a(e))
        he = edge[: n // 2 + 1]
        yield f"irfft edge n={n}", lambda he=he, n=n: F.irfft(a(he), n=n)
        yield f"hfft edge n={n}", lambda he=he, n=n: F.hfft(a(he), n=n, norm="ortho")
    yield "rfft huge", lambda: F.rfft(a(L.rng(2319).uniform(-1, 1, 64) * 2.0**1012))
    yield "rfft subnormal", lambda: F.rfft(a(L.rng(2320).uniform(-1, 1, 127) * 2.0**-1060))
    yield "irfft subnormal", lambda: F.irfft(a(_c(L.rng(2321), 65) * 2.0**-1060))
    yield "rfft length 1", lambda: F.rfft(a(np.array([-2.5])))
    yield "irfft length 1 n=1", lambda: F.irfft(a(np.array([2.5 - 1j])), n=1)
    yield "rfft length 2", lambda: F.rfft(a(np.array([3.0, -1.25])))
    hm = _c(r, 9)
    yield "irfft default n", lambda: F.irfft(a(hm))
    yield "hfft default n", lambda: F.hfft(a(hm))
    for n in (1, 3, 8, 15, 16, 17, 30):
        yield f"irfft cut/pad n={n}", lambda n=n: F.irfft(a(hm), n=n)
        yield f"hfft cut/pad n={n} forward", lambda n=n: F.hfft(a(hm), n=n, norm="forward")
    mr = r.uniform(-1, 1, (6, 10))
    for n in (1, 4, 10, 13, 16):
        yield f"rfft cut/pad n={n} axis=-1", lambda n=n: F.rfft(a(mr), n=n)
        yield f"ihfft cut/pad n={n} axis=0", lambda n=n: F.ihfft(a(mr), n=n, axis=0)
        yield f"fft real cut/pad n={n} axis=0", lambda n=n: F.fft(a(mr), n=n, axis=0)
    tr = r.uniform(-1, 1, (5, 6, 7))
    for axis in (0, 1, 2, -1):
        yield f"rfft 3-d axis={axis}", lambda axis=axis: F.rfft(a(tr), axis=axis)
        yield f"irfft 3-d axis={axis} F order", lambda axis=axis: F.irfft(a(np.asfortranarray(t)), axis=axis)
        yield f"fft real 3-d axis={axis} F order", lambda axis=axis: F.fft(a(np.asfortranarray(tr)), axis=axis)
    yield "rfft strided", lambda: F.rfft(a(mr[::-2, 1::3]), axis=0)
    yield "rfft real part of complex", lambda: F.rfft(a(m.real))
    yield "irfft strided", lambda: F.irfft(a(m[::-2, 1::3]), axis=0)
    yield "rfft empty batch", lambda: F.rfft(a(np.zeros((0, 4))))
    yield "irfft empty batch padded", lambda: F.irfft(a(np.zeros((3, 0), np.complex128)), n=5)
    ints = np.array([[3, -1, 4, 1, -5, 9], [2, 6, -5, 3, 5, -8]])
    for name, x in (("int64", ints), ("bool", ints % 2 == 0), ("float64", ints * 0.5)):
        yield f"rfft {name}", lambda x=x: F.rfft(a(x))
        yield f"fft real {name}", lambda x=x: F.fft(a(x))
        yield f"ifft real {name} axis=0", lambda x=x: F.ifft(a(x), axis=0)
        yield f"irfft {name}", lambda x=x: F.irfft(a(x))
        yield f"hfft {name} n=7", lambda x=x: F.hfft(a(x), n=7)
    bigr = L.rng(2322).uniform(-1, 1, (300, 256))
    yield "rfft batch 300x256 axis=-1", lambda: F.rfft(a(bigr))
    yield "rfft batch 300x256 axis=0", lambda: F.rfft(a(bigr), axis=0)
    yield "fft real batch 300x256 axis=0", lambda: F.fft(a(bigr), axis=0)
    yield "irfft batch 64x1009 axis=-1", lambda: F.irfft(a(_c(L.rng(2323), (64, 505))), n=1009)
    yield "hfft batch 40x1000 axis=0 ortho", lambda: F.hfft(a(_c(L.rng(2324), (501, 40))), axis=0, norm="ortho")
    yield "rfft batch 500x26", lambda: F.rfft(a(L.rng(2325).uniform(-1, 1, (500, 26))))
    yield "rfft long 2^17", lambda: F.rfft(a(L.rng(2326).uniform(-1, 1, 1 << 17)))
    yield "fft real long 65537", lambda: F.fft(a(L.rng(2327).uniform(-1, 1, 65537)))
    yield "irfft long 2^17", lambda: F.irfft(a(_c(L.rng(2328), (1 << 16) + 1)))
    yield "hfft long 65537", lambda: F.hfft(a(_c(L.rng(2329), 32769)), n=65537)
    xr4 = a(np.arange(4.0))
    yield "rfft complex input", lambda: F.rfft(x4)
    yield "ihfft complex input", lambda: F.ihfft(x4)
    yield "irfft length 1 default n", lambda: F.irfft(a(np.array([1.0 + 0j])))
    yield "hfft n=0", lambda: F.hfft(x4, n=0)
    yield "rfft axis out of range", lambda: F.rfft(xr4, axis=1)
    yield "irfft norm=None", lambda: F.irfft(x4, norm=None)
    yield "rfft length 0", lambda: F.rfft(a(np.zeros(0)))
    yield "irfft positional n", lambda: F.irfft(x4, 4)
    yield "ihfft workers", lambda: F.ihfft(xr4, workers=2)

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
