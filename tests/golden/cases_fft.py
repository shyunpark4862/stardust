"""Golden cases for sd.fft (tools/golden.py has the contract).

The one-dimensional complex transforms and the helpers (milestone FT1, docs/design/fft.md §8): fft and
ifft on every length form (2^k, f 2^k, the other 11-smooth lengths with radices 7 and 11, the radices 13
to 31, Bluestein for prime and composite lengths with a prime factor above 31) times norm, the edge values
(NaN, inf, signed zeros, huge and subnormal entries), cutting and zero-padding, every axis of 2-d and 3-d arrays in C, F
and strided layouts, batches large enough to split into many tasks at 3 and 10 threads, one long line,
the empty batch axis and length 1; fftfreq and rfftfreq (numpy's bits), fftshift and ifftshift on every
dtype and axes form, next_fast_len and prev_fast_len; the wrong calls (their errors are results).

The real and Hermitian transforms (milestone FT2): rfft, irfft, hfft and ihfft on every length form, even
and odd (an even length runs the complex plan of n / 2 and a twiddle pass, an odd one the complex plan of n
and the Hermitian part, decision FT43; Bluestein inner lengths both ways) times norm, the dropped parts of
a Hermitian input, the default lengths, cutting and padding, the edge values, the layouts and batches, a
long line, the input dtypes and the wrong calls; and real input to fft and ifft (bool, int64, float64),
which takes the real path since FT2 (decision FT16; FT1 left it out, decision FT38). The long lines (2^15
and more) take the four-step split (milestone FT6, decision FT65).

The n-dimensional transforms (milestone FT3): fft2, ifft2, fftn, ifftn, rfft2, irfft2, rfftn, irfftn,
hfft2, ihfft2, hfftn and ihfftn on shapes with 2^k, other 11-smooth and Bluestein axes and even and odd
real last lengths, times norm; every form of s and axes (None, -1, cutting, padding, a subset, any order,
negative, a list, s without axes, an int); the parts of a non-Hermitian input irfftn drops; real, int64
and bool input; C, F and strided layouts; arrays large enough that every axis pass splits into many tasks
at 3 and 10 threads; the empty batch; the wrong calls.

The discrete cosine and sine transforms (milestone FT4): dct, idct, dst and idst of types 1-4 on even, odd
and Bluestein inner lengths (the real FFT of n, of 2 (n - 1) and 2 (n + 1), the complex FFT of n / 2, the odd
DCT-IV) times norm, orthogonalize both ways, cutting and padding, the edge values, complex, int64 and bool
input, the layouts and batches, a long line; dctn, idctn, dstn and idstn over the forms of s and axes; the
one-point lengths (DCT-I is an error, DST-I is 2x); the wrong calls.

FFTLog (milestone FT5): fht and ifht on even, odd and Bluestein inner lengths (1-4096) and the parameter sets of
tests/fft/refdata.FHT_PARAMS (unbiased, biased, negative spacing, the poles of Gamma, a large order, a small
spacing), the edge values, int64 and bool input, the layouts and batches (every line a task at 3 and 10
threads; each thread count runs in a process of its own, so the plans' coefficients are computed at each),
fhtoffset, the wrong calls.

The inputs come from fixed literals and L.rng(seed); the largest result is 2^17 complex numbers.
"""

import math

import numpy as np

NAN, INF = math.nan, math.inf
NORMS = ("backward", "ortho", "forward")
# 2^k, f 2^k, other 11-smooth (7, 11, 9 = 3 3, 25 = 5 5, 77 = 7 11), the radices 13 to 31 (13, 17, 169 = 13 13;
# decision FT64), Bluestein (127, 1009, 2018 = 2 1009, 4097 = 17 241)
LENGTHS = (1, 2, 3, 4, 5, 7, 8, 11, 16, 24, 40, 64, 120, 1024, 3072, 9, 25, 77, 1000, 2310, 13, 17, 127, 1009,
           2018, 169, 4097)  # fmt: skip


def _c(r, shape):
    return r.uniform(-1, 1, shape) + 1j * r.uniform(-1, 1, shape)


def _r2r_cases(L, F, a, r, redge, ints, t3, t3r, big3, t2r):
    """The discrete cosine and sine transforms (milestone FT4)."""
    rr = L.rng(2340)
    for n in (1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 24, 26, 34, 64, 100, 127, 1000, 1009, 2018):
        x = rr.uniform(-1, 1, n)
        for name in ("dct", "dst"):
            f = getattr(F, name)
            for t in (1, 2, 3, 4):
                for norm in NORMS:
                    yield f"{name} type={t} n={n} {norm}", lambda f=f, x=x, t=t, norm=norm: f(a(x), type=t, norm=norm)
    for n in (2, 5, 8, 13, 26, 127, 1000):
        x = rr.uniform(-1, 1, n)
        for name in ("idct", "idst"):
            f = getattr(F, name)
            for t in (1, 2, 3, 4):
                for norm in NORMS:
                    yield f"{name} type={t} n={n} {norm}", lambda f=f, x=x, t=t, norm=norm: f(a(x), type=t, norm=norm)
                for norm, o in (("backward", True), ("ortho", False), ("forward", True)):
                    yield (
                        f"{name} type={t} n={n} {norm} orthogonalize={o}",
                        (lambda f=f, x=x, t=t, norm=norm, o=o: f(a(x), type=t, norm=norm, orthogonalize=o)),
                    )
    for name in ("dct", "dst"):
        f = getattr(F, name)
        x = rr.uniform(-1, 1, 9)
        for t in (1, 2, 3, 4):
            yield f"{name} type={t} orthogonalize=True", lambda f=f, t=t: f(a(x), type=t, orthogonalize=True)
            for n in (2, 6, 9, 14):
                yield f"{name} type={t} cut/pad n={n}", lambda f=f, t=t, n=n: f(a(x), type=t, n=n)
            for e in (8, 7):
                yield f"{name} type={t} edge n={e}", lambda f=f, t=t, e=e: f(a(redge[:e]), type=t)
            yield f"{name} type={t} huge", lambda f=f, t=t: f(a(L.rng(2341).uniform(-1, 1, 64) * 2.0**1010), type=t)
            yield (
                f"{name} type={t} subnormal",
                lambda f=f, t=t: f(a(L.rng(2342).uniform(-1, 1, 127) * 2.0**-1060), type=t),
            )
            yield f"{name} type={t} complex", lambda f=f, t=t: f(a(_c(L.rng(2343), (3, 10))), type=t, axis=0)
            yield f"{name} type={t} int64", lambda f=f, t=t: f(a(ints), type=t)
            yield f"{name} type={t} bool", lambda f=f, t=t: f(a(ints % 3 == 0), type=t, norm="ortho")
            yield f"{name} type={t} zeros", lambda f=f, t=t: f(a(np.zeros(6)), type=t)
            yield f"{name} type={t} 3-d F order axis=1", lambda f=f, t=t: f(a(np.asfortranarray(t3r)), type=t, axis=1)
            yield f"{name} type={t} strided axis=0", lambda f=f, t=t: f(a(t3r[::-1, 1::2, ::3]), type=t, axis=0)
            yield f"{name} type={t} empty batch", lambda f=f, t=t: f(a(np.zeros((0, 5))), type=t)
    yield "dst type=1 one point", lambda: F.dst(a(np.array([2.0])), type=1)
    yield "dct type=2 one point", lambda: F.dct(a(np.array([2.0])), type=2, norm="ortho")
    yield "dct type=4 one point", lambda: F.dct(a(np.array([-3.0])), type=4)
    bigr = L.rng(2344).uniform(-1, 1, (300, 256))
    for t in (1, 2, 3, 4):
        yield f"dct type={t} batch 300x256 axis=-1", lambda t=t: F.dct(a(bigr), type=t)
        yield f"idst type={t} batch 300x256 axis=0", lambda t=t: F.idst(a(bigr), type=t, axis=0, norm="ortho")
        yield f"dst type={t} batch 64x1009", lambda t=t: F.dst(a(L.rng(2345).uniform(-1, 1, (64, 1009))), type=t)
    yield "dct type=2 long 2^17", lambda: F.dct(a(L.rng(2346).uniform(-1, 1, 1 << 17)))
    yield "idct type=4 long 65537", lambda: F.idct(a(L.rng(2347).uniform(-1, 1, 65537)), type=4)
    for s_, ax in ((None, None), ((5, 3), None), ((9, -1), [2, 0]), (None, (-1, -3)), ((8,), (1,)), (7, None),
                   ((3, 7, 12), (0, 1, 2))):  # fmt: skip
        tag = f"s={s_} axes={ax}"
        for t in (1, 2, 3, 4):
            yield f"dctn type={t} {tag}", lambda s_=s_, ax=ax, t=t: F.dctn(a(t3r), type=t, s=s_, axes=ax)
            yield (
                f"idctn type={t} {tag}",
                lambda s_=s_, ax=ax, t=t: F.idctn(a(t3r), type=t, s=s_, axes=ax, norm="ortho"),
            )
            yield (
                f"dstn type={t} {tag}",
                lambda s_=s_, ax=ax, t=t: F.dstn(a(t3r), type=t, s=s_, axes=ax, norm="forward"),
            )
            yield f"idstn type={t} {tag}", lambda s_=s_, ax=ax, t=t: F.idstn(a(t3r), type=t, s=s_, axes=ax)
    yield "dctn complex", lambda: F.dctn(a(t3), type=2)
    yield "idstn complex ortho", lambda: F.idstn(a(t3), type=3, norm="ortho", axes=(0, 2))
    yield "dctn int64", lambda: F.dctn(a(ints), type=1)
    yield "dstn bool", lambda: F.dstn(a(ints % 2 == 0), type=4)
    yield "dctn orthogonalize=True", lambda: F.dctn(a(t3r), type=3, orthogonalize=True)
    yield "dctn 48x64x40", lambda: F.dctn(a(big3.real), type=2)
    yield "idstn 48x64x40 axes=(0, 2)", lambda: F.idstn(a(big3.imag), type=4, axes=(0, 2), norm="ortho")
    yield "dctn empty batch", lambda: F.dctn(a(np.zeros((0, 4, 6))), axes=(1, 2))
    xr4 = a(np.arange(4.0))
    yield "dct type=1 one point", lambda: F.dct(a(np.array([2.0])), type=1)
    yield "idct type=1 n=1", lambda: F.idct(xr4, type=1, n=1)
    yield "dctn type=1 axis of one", lambda: F.dctn(a(np.ones((3, 1))), type=1)
    yield "dct type=5", lambda: F.dct(xr4, type=5)
    yield "dst type=0", lambda: F.dst(xr4, type=0)
    yield "dct type=2.0", lambda: F.dct(xr4, type=2.0)
    yield "dct type=True", lambda: F.dct(xr4, type=True)
    yield "dct type huge", lambda: F.dct(xr4, type=2**70)
    yield "dst orthogonalize=1", lambda: F.dst(xr4, orthogonalize=1)
    yield "idct n=0", lambda: F.idct(xr4, n=0)
    yield "idst norm=None", lambda: F.idst(xr4, norm=None)
    yield "dct length 0", lambda: F.dct(a(np.zeros(0)))
    yield "dct positional type", lambda: F.dct(xr4, 2)
    yield "dct list", lambda: F.dct([1.0, 2.0])
    yield "dct workers", lambda: F.dct(xr4, workers=2)
    yield "dctn repeated axes", lambda: F.dctn(t2r, axes=(0, 0))
    yield "dstn axes=()", lambda: F.dstn(t2r, axes=())
    yield "idctn s too long", lambda: F.idctn(t2r, s=(2, 2, 2))


def _fftlog_cases(L, F, a):
    """FFTLog (milestone FT5)."""
    rr = L.rng(2350)
    params = (("unbiased", 0.01, 0.5, 0.0, 0.0), ("logspace", None, 0.3, -6 * math.log(10), 0.0),
              ("reflection", 0.1, -0.7, 1.0, 0.4), ("negative-dln", -0.3, 1.5, 0.2, -0.6),
              ("poles-both", 0.05, -4.0, 0.0, 1.0), ("pole-plus", 0.05, -3.5, 0.0, 0.5),
              ("pole-minus", 0.05, -2.5, 0.0, 0.5), ("large-mu", 0.02, 50.5, 0.0, 0.0),
              ("small-dln", 1e-4, 0.0, 0.0, 0.0))  # fmt: skip
    for n in (1, 2, 3, 16, 63, 64, 127, 1000, 1009, 2018, 4096):
        x = rr.standard_normal(n)
        for name, dln, mu, offset, bias in params:
            if dln is None:
                if n < 2:
                    continue
                dln = math.log(1e8) / (n - 1)
            if abs(bias * dln) * n / 2 > 300:
                continue
            for fname in ("fht", "ifht"):
                f = getattr(F, fname)
                yield f"{fname} {name} n={n}", lambda f=f, x=x, d=dln, m=mu, o=offset, q=bias: f(
                    a(x), d, m, offset=o, bias=q)  # fmt: skip
    edge = np.array([1.0, NAN, INF, -0.0, 1e300, 5e-324, -INF, 3.0])
    yield "fht edge", lambda: F.fht(a(edge), 0.1, 0.5)
    yield "ifht edge biased", lambda: F.ifht(a(edge), 0.1, 0.5, bias=0.3)
    yield "fht zeros", lambda: F.fht(a(np.zeros(12)), 0.1, 0.5, bias=0.2)
    yield "fht subnormal", lambda: F.fht(a(rr.standard_normal(64) * 2.0**-1060), 0.05, 1.0)
    yield "fht int64", lambda: F.fht(a(np.arange(-5, 11)), 0.2, 0.0)
    yield "ifht bool", lambda: F.ifht(a(np.arange(15) % 3 == 0), 0.2, 2.0, offset=0.5)
    big = rr.standard_normal((300, 256))
    yield "fht batch 300x256", lambda: F.fht(a(big), 0.02, 0.5, offset=0.1)
    yield "ifht batch 300x256 biased", lambda: F.ifht(a(big), 0.02, 0.5, offset=0.1, bias=-0.3)
    yield "fht batch F order", lambda: F.fht(a(np.asfortranarray(big[:40, :63])), 0.05, 1.5, bias=0.2)
    yield "fht batch strided", lambda: F.fht(a(big[::3, ::2]), 0.05, 1.5)
    t3 = rr.standard_normal((4, 5, 1009))
    yield "fht 3-d 4x5x1009", lambda: F.fht(a(t3), 0.01, -0.5, offset=0.7, bias=0.1)
    yield "fht empty batch", lambda: F.fht(a(np.zeros((0, 8))), 0.1, 0.5)
    yield "fht long 2^16", lambda: F.fht(a(L.rng(2351).standard_normal(1 << 16)), 1e-3, 0.5, offset=0.25)
    for dln, mu, initial, bias in ((0.1, 2.0, 0.5, 0.0), (0.01, 0.0, -6 * math.log(10), 0.0), (-0.3, 1.5, 0.2, -0.6),
                                   (1e-4, 0.5, 3.0, 0.8), (2.5, -3.5, 0.0, 0.5), (0.05, 50.5, -1.0, 0.0)):  # fmt: skip
        name = f"fhtoffset dln={dln} mu={mu} initial={initial:.3g} bias={bias}"
        yield name, lambda d=dln, m=mu, i=initial, q=bias: F.fhtoffset(d, m, initial=i, bias=q)
    x8 = a(np.arange(8.0))
    yield "fht list", lambda: F.fht([1.0, 2.0], 0.1, 0.5)
    yield "fht complex", lambda: F.fht(a(np.ones(4) + 1j), 0.1, 0.5)
    yield "fht dln=0", lambda: F.fht(x8, 0.0, 0.5)
    yield "fht mu=nan", lambda: F.fht(x8, 0.1, NAN)
    yield "ifht bias=True", lambda: F.ifht(x8, 0.1, 0.5, bias=True)
    yield "fht length 0", lambda: F.fht(a(np.zeros((3, 0))), 0.1, 0.5)
    yield "fht positional offset", lambda: F.fht(x8, 0.1, 0.5, 0.0)
    yield "fht workers", lambda: F.fht(x8, 0.1, 0.5, workers=2)
    yield "fhtoffset dln=inf", lambda: F.fhtoffset(INF, 0.5)


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

    # ---- the n-dimensional transforms (FT3)
    shapes = ((8, 16), (6, 10), (5, 7), (13, 8), (4, 6, 9), (3, 17, 4), (2, 3, 4, 5))
    for shape in shapes:
        x = _c(r, shape)
        xr = r.uniform(-1, 1, shape)
        sh = "x".join(map(str, shape))
        for norm in NORMS:
            yield f"fftn {sh} {norm}", lambda x=x, norm=norm: F.fftn(a(x), norm=norm)
            yield f"ifftn {sh} {norm}", lambda x=x, norm=norm: F.ifftn(a(x), norm=norm)
            yield f"rfftn {sh} {norm}", lambda xr=xr, norm=norm: F.rfftn(a(xr), norm=norm)
            yield f"irfftn {sh} {norm}", lambda x=x, norm=norm: F.irfftn(a(x), norm=norm)
            yield f"hfftn {sh} {norm}", lambda x=x, norm=norm: F.hfftn(a(x), norm=norm)
            yield f"ihfftn {sh} {norm}", lambda xr=xr, norm=norm: F.ihfftn(a(xr), norm=norm)
            yield f"fftn real {sh} {norm}", lambda xr=xr, norm=norm: F.fftn(a(xr), norm=norm)
            yield f"ifftn real {sh} {norm}", lambda xr=xr, norm=norm: F.ifftn(a(xr), norm=norm)
    t3 = _c(r, (4, 6, 10))
    t3r = t3.real.copy()
    forms = ((None, None), ((5, 3), None), ((9, -1), [2, 0]), (None, (-1, -3)), ((8,), (1,)), ((3, 7, 12), None),
             ((4, 6, 11), (0, 1, 2)), (7, None), (None, 1), ((2, 2), (2, 1)))  # fmt: skip
    for s_, ax in forms:
        tag = f"s={s_} axes={ax}"
        yield f"fftn {tag}", lambda s_=s_, ax=ax: F.fftn(a(t3), s=s_, axes=ax)
        yield f"ifftn {tag}", lambda s_=s_, ax=ax: F.ifftn(a(t3), s=s_, axes=ax, norm="ortho")
        yield f"rfftn {tag}", lambda s_=s_, ax=ax: F.rfftn(a(t3r), s=s_, axes=ax)
        yield f"irfftn {tag}", lambda s_=s_, ax=ax: F.irfftn(a(t3), s=s_, axes=ax)
        yield f"hfftn {tag}", lambda s_=s_, ax=ax: F.hfftn(a(t3), s=s_, axes=ax, norm="forward")
        yield f"ihfftn {tag}", lambda s_=s_, ax=ax: F.ihfftn(a(t3r), s=s_, axes=ax)
    m2 = _c(r, (6, 10))
    for s_ in (None, (6, 10), (4, 13), (-1, 7), (9, 16)):
        yield f"fft2 s={s_}", lambda s_=s_: F.fft2(a(m2), s=s_)
        yield f"ifft2 s={s_}", lambda s_=s_: F.ifft2(a(m2), s=s_)
        yield f"rfft2 s={s_}", lambda s_=s_: F.rfft2(a(m2.real), s=s_)
        yield f"irfft2 s={s_}", lambda s_=s_: F.irfft2(a(m2), s=s_)
        yield f"hfft2 s={s_}", lambda s_=s_: F.hfft2(a(m2), s=s_, norm="ortho")
        yield f"ihfft2 s={s_}", lambda s_=s_: F.ihfft2(a(m2.real), s=s_)
    yield "fft2 axes=(0, 2)", lambda: F.fft2(a(t3), axes=(0, 2))
    yield "irfft2 axes=(2, 0)", lambda: F.irfft2(a(t3), axes=(2, 0))
    # a non-Hermitian input: irfftn drops numpy's and scipy's parts; the entries beyond n // 2 + 1 are never read
    nh = _c(r, (5, 9))
    nh2 = nh.copy()
    nh2[:, 7:] = NAN
    yield "irfftn non-Hermitian", lambda: F.irfftn(a(nh), s=(5, 12))
    yield "irfftn non-Hermitian NaN beyond", lambda: F.irfftn(a(nh2), s=(5, 12))
    yield "irfftn non-Hermitian odd", lambda: F.irfftn(a(nh), s=(6, 11))
    ndedge = edge[:6].reshape(2, 3)
    yield "fftn edge", lambda: F.fftn(a(ndedge))
    yield "irfftn edge", lambda: F.irfftn(a(ndedge))
    yield "fftn subnormal", lambda: F.fftn(a(_c(L.rng(2330), (13, 8)) * 2.0**-1060))
    yield "rfftn huge", lambda: F.rfftn(a(L.rng(2331).uniform(-1, 1, (8, 16)) * 2.0**1010))
    for name, x in (("int64", ints), ("bool", ints % 2 == 0), ("float64", ints * 0.5)):
        yield f"fftn {name}", lambda x=x: F.fftn(a(x))
        yield f"rfftn {name}", lambda x=x: F.rfftn(a(x))
        yield f"irfftn {name}", lambda x=x: F.irfftn(a(x))
        yield f"ihfft2 {name}", lambda x=x: F.ihfft2(a(x))
    yield "fftn F order", lambda: F.fftn(a(np.asfortranarray(t3)))
    yield "irfftn F order", lambda: F.irfftn(a(np.asfortranarray(t3)), axes=(0, 2))
    yield "rfftn strided", lambda: F.rfftn(a(t3r[::-1, 1::2, ::3]))
    yield "hfftn strided", lambda: F.hfftn(a(t3[::2, ::-1]))
    yield "fftn empty batch", lambda: F.fftn(a(np.zeros((0, 4, 6), np.complex128)), axes=(1, 2))
    yield "irfftn empty batch", lambda: F.irfftn(a(np.zeros((3, 0, 6), np.complex128)), axes=(0, 2))
    yield "fft2 padded from length 0", lambda: F.fft2(a(np.zeros((3, 0), np.complex128)), s=(3, 2))
    big3 = _c(L.rng(2332), (48, 64, 40))
    yield "fftn 48x64x40", lambda: F.fftn(a(big3))
    yield "ifftn 48x64x40 axes=(0, 2) s", lambda: F.ifftn(a(big3), s=(50, 40), axes=(0, 2))
    yield "rfftn 48x64x40", lambda: F.rfftn(a(big3.real))
    yield "irfftn 48x64x40", lambda: F.irfftn(a(big3))
    yield "hfftn 48x64x40 axes=(1, 0)", lambda: F.hfftn(a(big3), axes=(1, 0), norm="ortho")
    yield "fft2 34x1009", lambda: F.fft2(a(_c(L.rng(2333), (34, 1009))))
    yield "irfft2 256x257", lambda: F.irfft2(a(_c(L.rng(2334), (256, 257))))
    t2 = a(np.ones((4, 6)) + 0j)
    t2r = a(np.ones((4, 6)))
    yield "fftn s=(None, 4)", lambda: F.fftn(t2, s=(None, 4))
    yield "fftn s=(0, 4)", lambda: F.fftn(t2, s=(0, 4))
    yield "fftn s and axes lengths", lambda: F.fftn(t2, s=(4, 4, 4), axes=(0, 1))
    yield "fftn s too long", lambda: F.fftn(t2, s=(2, 2, 2))
    yield "fftn repeated axes", lambda: F.fftn(t2, axes=(1, 1))
    yield "fftn axes=()", lambda: F.fftn(t2, axes=())
    yield "fftn s=()", lambda: F.fftn(t2, s=())
    yield "fft2 1-d", lambda: F.fft2(x4)
    yield "rfftn complex input", lambda: F.rfftn(t2)
    yield "irfftn length 1 default", lambda: F.irfftn(a(np.ones((3, 1), np.complex128)))
    yield "ihfftn axis out of range", lambda: F.ihfftn(t2r, axes=(0, 2))
    yield "hfft2 norm=None", lambda: F.hfft2(t2, norm=None)
    yield "ifft2 positional s", lambda: F.ifft2(t2, (4, 4))
    yield "irfft2 workers", lambda: F.irfft2(t2, workers=2)

    # ---- the discrete cosine and sine transforms (FT4): their own scope, so no name above is rebound
    yield from _r2r_cases(L, F, a, r, redge, ints, t3, t3r, big3, t2r)

    # ---- FFTLog (FT5)
    yield from _fftlog_cases(L, F, a)

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
