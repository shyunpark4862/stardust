"""stardust.fft: discrete Fourier transforms, real and Hermitian transforms, discrete cosine and sine
transforms and their helpers (docs/api/fft.md, docs/design/fft.md).

Usage: ``import stardust as sd`` then ``sd.fft.fft(x)``, ``sd.fft.ifft(y, n=1024)`` and
``sd.fft.fftfreq(1024, d=0.1)``. The names are scipy.fft's (which include numpy.fft's), in one flat
namespace; inputs and options follow this library's rules (decisions FT1-FT37 in docs/design/fft.md
§1.1): x is an sd.ndarray, options are keyword-only, norm defaults to 'backward'.

The public names come milestone by milestone (docs/design/fft.md §8): FT1 the one-dimensional complex
transforms (fft, ifft) and the helpers (fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len,
prev_fast_len).
"""

from ._basic import fft, ifft
from ._helper import fftfreq, fftshift, ifftshift, next_fast_len, prev_fast_len, rfftfreq

__all__ = ["fft", "ifft", "fftfreq", "rfftfreq", "fftshift", "ifftshift", "next_fast_len", "prev_fast_len"]
