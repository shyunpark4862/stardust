"""stardust.fft: discrete Fourier transforms, real and Hermitian transforms, discrete cosine and sine
transforms and their helpers (docs/api/fft.md, docs/design/fft.md).

Usage: ``import stardust as sd`` then ``sd.fft.fft(x)``, ``sd.fft.ifft(y, n=1024)`` and
``sd.fft.fftfreq(1024, d=0.1)``. The names are scipy.fft's (which include numpy.fft's), in one flat
namespace; inputs and options follow this library's rules (decisions FT1-FT68 in docs/design/fft.md
§1.1): x is an sd.ndarray, options are keyword-only, norm defaults to 'backward'.

The public names (35, docs/api/fft.md): the one-dimensional complex transforms (fft, ifft), the
one-dimensional real and Hermitian transforms (rfft, irfft, hfft, ihfft), the n-dimensional transforms
(fft2, ifft2, fftn, ifftn, rfft2, irfft2, rfftn, irfftn, hfft2, ihfft2, hfftn, ihfftn), the discrete cosine
and sine transforms (dct, idct, dst, idst, dctn, idctn, dstn, idstn), FFTLog (fht, ifht, fhtoffset) and the
helpers (fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len, prev_fast_len).
"""

from ._basic import (
    fft,
    fft2,
    fftn,
    hfft,
    hfft2,
    hfftn,
    ifft,
    ifft2,
    ifftn,
    ihfft,
    ihfft2,
    ihfftn,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfftn,
)
from ._fftlog import fht, fhtoffset, ifht
from ._helper import fftfreq, fftshift, ifftshift, next_fast_len, prev_fast_len, rfftfreq
from ._realtransforms import dct, dctn, dst, dstn, idct, idctn, idst, idstn

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "dct",
    "idct",
    "dst",
    "idst",
    "dctn",
    "idctn",
    "dstn",
    "idstn",
    "fht",
    "ifht",
    "fhtoffset",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
    "next_fast_len",
    "prev_fast_len",
]
