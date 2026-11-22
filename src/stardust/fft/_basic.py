"""The one-dimensional transforms of sd.fft (docs/api/fft.md §2, §3; docs/design/fft.md §2.2, §2.3, §3.4):
fft and ifft (milestone FT1), rfft, irfft, hfft and ihfft (milestone FT2).

The arguments are checked and the lines transformed by stardust._core.fft (csrc/fft/basic.cpp): every
line by the plan of its kind and length (complex: the mixed radix for 11-smooth lengths, Bluestein for the
others; real: the complex plan of n / 2 and one twiddle pass for even n, the complex plan of n and the
Hermitian part for odd n), so a line has the same bits in any batch, layout and thread count. Real input
to fft and ifft takes the real path (decision FT16): the first n // 2 + 1 entries of fft(x) are rfft(x)
bit for bit.
"""

from .. import _core as _sd
from .._calls import function

_c2c = _sd.fft._c2c
_r2c = _sd.fft._r2c
_c2r = _sd.fft._c2r


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
