"""The discrete cosine and sine transforms of sd.fft (docs/api/fft.md §4; docs/design/fft.md §2.6, §3.5):
dct, idct, dst, idst and the n-dimensional dctn, idctn, dstn, idstn, types 1-4 (milestone FT4).

The definitions are scipy.fft's. The arguments are checked and the lines transformed by
stardust._core.fft (csrc/fft/r2r.cpp, csrc/fft/r2r.hpp): every type is this library's own algorithm on
the real FFT (vDSP's DCT is single precision only), decided by the length alone, so a line has the same
bits in any batch, layout and thread count. DST-II, DST-III and DST-IV are the DCT of the same type with
exact sign changes and reversals, bit for bit. Complex input transforms its real and imaginary parts
separately (decision FT28). norm counts the logical length N' (2 (N - 1) for DCT-I, 2 (N + 1) for DST-I,
2N otherwise; the product over the axes for the n-dimensional ones); orthogonalize=None means
norm == 'ortho'.
"""

from .. import _core as _sd
from .._calls import function

_r2r = _sd.fft._r2r
_r2rn = _sd.fft._r2rn


@function
def dct(x, /, *, type=2, n=None, axis=-1, norm="backward", orthogonalize=None):
    """The discrete cosine transform of type 1-4 of x along `axis` (scipy's definitions). n cuts or
    zero-pads the axis (DCT-I needs n >= 2); times 1 ('backward'), 1/sqrt(N') ('ortho') or 1/N' ('forward').
    float64, or complex128 for complex input (its parts transformed separately)."""
    return _r2r(x, type, n, axis, norm, orthogonalize, False, False, "dct")


@function
def idct(x, /, *, type=2, n=None, axis=-1, norm="backward", orthogonalize=None):
    """The inverse of dct: the DCT of the inverse type (2 and 3 swapped) times 1/N' ('backward'),
    1/sqrt(N') ('ortho') or 1 ('forward')."""
    return _r2r(x, type, n, axis, norm, orthogonalize, True, False, "idct")


@function
def dst(x, /, *, type=2, n=None, axis=-1, norm="backward", orthogonalize=None):
    """The discrete sine transform of type 1-4 of x along `axis` (scipy's definitions); as dct."""
    return _r2r(x, type, n, axis, norm, orthogonalize, False, True, "dst")


@function
def idst(x, /, *, type=2, n=None, axis=-1, norm="backward", orthogonalize=None):
    """The inverse of dst: the DST of the inverse type (2 and 3 swapped) with the inverse direction's
    factor."""
    return _r2r(x, type, n, axis, norm, orthogonalize, True, True, "idst")


@function
def dctn(x, /, *, type=2, s=None, axes=None, norm="backward", orthogonalize=None):
    """The n-dimensional DCT of x over `axes` (None: every axis; with s, the last len(s) axes): dct along
    each, from the last of `axes` to the first, with one factor of N' = the product of the logical lengths."""
    return _r2rn(x, type, s, axes, norm, orthogonalize, False, False, "dctn")


@function
def idctn(x, /, *, type=2, s=None, axes=None, norm="backward", orthogonalize=None):
    """The n-dimensional inverse DCT of x over `axes`: idct along each."""
    return _r2rn(x, type, s, axes, norm, orthogonalize, True, False, "idctn")


@function
def dstn(x, /, *, type=2, s=None, axes=None, norm="backward", orthogonalize=None):
    """The n-dimensional DST of x over `axes`: dst along each."""
    return _r2rn(x, type, s, axes, norm, orthogonalize, False, True, "dstn")


@function
def idstn(x, /, *, type=2, s=None, axes=None, norm="backward", orthogonalize=None):
    """The n-dimensional inverse DST of x over `axes`: idst along each."""
    return _r2rn(x, type, s, axes, norm, orthogonalize, True, True, "idstn")
