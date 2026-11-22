"""fft and ifft of sd.fft (docs/api/fft.md §2; docs/design/fft.md §2.2; milestone FT1).

The arguments are checked and the lines transformed by stardust._core.fft (csrc/fft/basic.cpp): every
line by the plan of its length (the mixed radix for 11-smooth lengths, Bluestein for the others), so a
line has the same bits in any batch, layout and thread count.
"""

from .. import _core as _sd
from .._calls import function

_c2c = _sd.fft._c2c


@function
def fft(x, /, *, n=None, axis=-1, norm="backward"):
    """The discrete Fourier transform of x along `axis`: y_k = sum_j x_j e^(-2 pi i jk/n), times 1
    ('backward'), 1/sqrt(n) ('ortho') or 1/n ('forward'). n cuts or zero-pads the axis. complex128."""
    return _c2c(x, n, axis, norm, False)


@function
def ifft(x, /, *, n=None, axis=-1, norm="backward"):
    """The inverse discrete Fourier transform of x along `axis`: y_k = sum_j x_j e^(+2 pi i jk/n),
    times 1/n ('backward'), 1/sqrt(n) ('ortho') or 1 ('forward'). complex128."""
    return _c2c(x, n, axis, norm, True)
