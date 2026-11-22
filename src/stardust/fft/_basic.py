"""The transforms of sd.fft (docs/api/fft.md §2, §3; docs/design/fft.md §2.2-§2.4, §3.4, §3.7): fft and
ifft (milestone FT1), rfft, irfft, hfft and ihfft (milestone FT2), and the n-dimensional fft2, ifft2,
fftn, ifftn, rfft2, irfft2, rfftn, irfftn, hfft2, ihfft2, hfftn and ihfftn (milestone FT3).

The arguments are checked and the lines transformed by stardust._core.fft (csrc/fft/basic.cpp): every
line by the plan of its kind and length (complex: the mixed radix for 31-smooth lengths, Bluestein for the
others; real: the complex plan of n / 2 and one twiddle pass for even n, the complex plan of n and the
Hermitian part for odd n), so a line has the same bits in any batch, layout and thread count. Real input
to fft and ifft takes the real path (decision FT16): the first n // 2 + 1 entries of fft(x) are rfft(x)
bit for bit.

An n-dimensional transform is one pass per axis, from the last of `axes` to the first (decision FT25):
rfftn's real pass first, irfftn's last, as numpy and scipy, so a non-Hermitian input to irfftn loses
the same parts. `s` gives the lengths (-1: the input's), `axes` the axes (None: every axis, or with `s`
the last len(s) axes); the normalisation counts N = the product of the lengths.
"""

from .. import _core as _sd
from .._calls import function

_c2c = _sd.fft._c2c
_r2c = _sd.fft._r2c
_c2r = _sd.fft._c2r
_c2cn = _sd.fft._c2cn
_r2cn = _sd.fft._r2cn
_c2rn = _sd.fft._c2rn


@function
def fft(x, /, *, n=None, axis=-1, norm="backward"):
    """The discrete Fourier transform of x along `axis`: y_k = sum_j x_j e^(-2 pi i jk/n), times 1
    ('backward'), 1/sqrt(n) ('ortho') or 1/n ('forward'). n cuts or zero-pads the axis. complex128.
    Real input (bool, int64, float64) takes the real path: y_(n-k) = conj(y_k) exactly."""
    return _c2c(x, n, axis, norm, False)


@function
def ifft(x, /, *, n=None, axis=-1, norm="backward"):
    """The inverse discrete Fourier transform of x along `axis`: y_k = sum_j x_j e^(+2 pi i jk/n),
    times 1/n ('backward'), 1/sqrt(n) ('ortho') or 1 ('forward'). complex128."""
    return _c2c(x, n, axis, norm, True)


@function
def rfft(x, /, *, n=None, axis=-1, norm="backward"):
    """The first n // 2 + 1 entries of fft(x) for real x (bool, int64, float64) along `axis`; the
    imaginary parts of the DC and (even n) Nyquist entries are exactly +0. complex128."""
    return _r2c(x, n, axis, norm, "rfft")


@function
def irfft(x, /, *, n=None, axis=-1, norm="backward"):
    """The inverse of rfft: the n real values (default n = 2 (m - 1)) of the inverse transform of the
    Hermitian extension of x along `axis`; n // 2 + 1 entries of x are used, the imaginary parts of the
    DC and (even n) Nyquist entries dropped. Times 1/n ('backward'), 1/sqrt(n) or 1. float64."""
    return _c2r(x, n, axis, norm, "irfft")


@function
def hfft(x, /, *, n=None, axis=-1, norm="backward"):
    """The transform of a Hermitian signal given by its first half: irfft(conj(x), n) with the forward
    direction's factor (1 for 'backward'). Default n = 2 (m - 1). float64."""
    return _c2r(x, n, axis, norm, "hfft")


@function
def ihfft(x, /, *, n=None, axis=-1, norm="backward"):
    """The inverse of hfft: conj(rfft(x)) with the inverse direction's factor (1/n for 'backward').
    complex128."""
    return _r2c(x, n, axis, norm, "ihfft")


@function
def fft2(x, /, *, s=None, axes=(-2, -1), norm="backward"):
    """fftn over two axes (default the last two)."""
    return _c2cn(x, s, axes, norm, False, "fft2")


@function
def ifft2(x, /, *, s=None, axes=(-2, -1), norm="backward"):
    """ifftn over two axes (default the last two)."""
    return _c2cn(x, s, axes, norm, True, "ifft2")


@function
def fftn(x, /, *, s=None, axes=None, norm="backward"):
    """The n-dimensional discrete Fourier transform of x over `axes` (None: every axis): fft along each,
    from the last of `axes` to the first. s cuts or zero-pads (-1: the input's length); times 1, 1/sqrt(N)
    or 1/N with N the product of the lengths. complex128; real input takes the real path."""
    return _c2cn(x, s, axes, norm, False, "fftn")


@function
def ifftn(x, /, *, s=None, axes=None, norm="backward"):
    """The n-dimensional inverse discrete Fourier transform of x over `axes`: ifft along each, times 1/N
    ('backward'), 1/sqrt(N) or 1. complex128."""
    return _c2cn(x, s, axes, norm, True, "ifftn")


@function
def rfft2(x, /, *, s=None, axes=(-2, -1), norm="backward"):
    """rfftn over two axes (default the last two)."""
    return _r2cn(x, s, axes, norm, "rfft2")


@function
def irfft2(x, /, *, s=None, axes=(-2, -1), norm="backward"):
    """irfftn over two axes (default the last two)."""
    return _c2rn(x, s, axes, norm, "irfft2")


@function
def rfftn(x, /, *, s=None, axes=None, norm="backward"):
    """The n-dimensional transform of real x: rfft along the last of `axes` (s[-1] // 2 + 1 entries), then
    fft along the others. complex128."""
    return _r2cn(x, s, axes, norm, "rfftn")


@function
def irfftn(x, /, *, s=None, axes=None, norm="backward"):
    """The inverse of rfftn: ifft along all but the last of `axes`, then irfft along the last (default
    length 2 (m - 1); s[-1] // 2 + 1 entries used, the imaginary parts of its DC and Nyquist entries
    dropped). Times 1/N ('backward'), 1/sqrt(N) or 1. float64."""
    return _c2rn(x, s, axes, norm, "irfftn")


@function
def hfft2(x, /, *, s=None, axes=(-2, -1), norm="backward"):
    """hfftn over two axes (default the last two)."""
    return _c2rn(x, s, axes, norm, "hfft2")


@function
def ihfft2(x, /, *, s=None, axes=(-2, -1), norm="backward"):
    """ihfftn over two axes (default the last two)."""
    return _r2cn(x, s, axes, norm, "ihfft2")


@function
def hfftn(x, /, *, s=None, axes=None, norm="backward"):
    """The n-dimensional transform of a Hermitian signal given by its half: irfftn(conj(x)) with the
    forward direction's factor (1 for 'backward'; scipy's definition). float64."""
    return _c2rn(x, s, axes, norm, "hfftn")


@function
def ihfftn(x, /, *, s=None, axes=None, norm="backward"):
    """The inverse of hfftn: conj(rfftn(x)) with the inverse direction's factor (1/N for 'backward').
    complex128."""
    return _r2cn(x, s, axes, norm, "ihfftn")
