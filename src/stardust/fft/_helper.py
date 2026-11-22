"""The helpers of sd.fft: fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len and prev_fast_len
(docs/api/fft.md §5; docs/design/fft.md §2.7; milestone FT1).

- fftfreq and rfftfreq (FT26) are numpy's formula, k * (1 / (n d)) with numpy's k, computed natively:
  numpy's bits.
- fftshift and ifftshift are sd.roll by n // 2 (and -(n // 2)) along each axis: the same bits.
- next_fast_len and prev_fast_len (FT18) are scipy's values: 11-smooth lengths (real=True: 5-smooth), the
  lengths the library transforms by its cheapest radices (the mixed radix also takes 13 to 31 since milestone
  FT6, decision FT64; a larger prime factor goes through Bluestein).
"""

from .. import _core as _sd
from .._calls import function
from . import _common as C

_core = _sd.fft


def _freq(name, n, d, real):
    n = C.integer(n, name, "n")
    if n < 1:
        raise ValueError(f"{name}(): 'n' must be at least 1, not {n}")
    d = C.finite_nonzero(C.real(d, name, "d"), name, "d")
    return _core._freq(n, 1.0 / (n * d), real)


@function
def fftfreq(n, /, *, d=1.0):
    """The sample frequencies of a transform of length n with sample spacing d: [0, 1, ..., n//2 - 1 or
    (n-1)//2, -(n//2), ..., -1] / (n d) (float64; numpy's bits)."""
    return _freq("fftfreq", n, d, False)


@function
def rfftfreq(n, /, *, d=1.0):
    """The sample frequencies of rfft's half spectrum: [0, 1, ..., n//2] / (n d) (float64; numpy's bits)."""
    return _freq("rfftfreq", n, d, True)


def _shift(name, x, axes, sign):
    x = C.array(x, name)
    ax = C.axes(axes, x.ndim, name)
    shape = x.shape
    shifts = tuple(sign * (shape[a] // 2) for a in ax)
    if not ax:
        return x.copy()
    return _sd.roll(x, shifts, axis=ax)


@function
def fftshift(x, /, *, axes=None):
    """x with the zero frequency moved to the middle: roll by n // 2 along each of `axes` (None: all)."""
    return _shift("fftshift", x, axes, 1)


@function
def ifftshift(x, /, *, axes=None):
    """fftshift's inverse: roll by -(n // 2) along each of `axes` (None: all)."""
    return _shift("ifftshift", x, axes, -1)


def _fast(name, target, real, prev):
    target = C.integer(target, name, "target")
    if real is not True and real is not False:
        raise TypeError(f"{name}(): 'real' must be True or False, not {C.type_name(real)}")
    if target < 1:
        raise ValueError(f"{name}(): 'target' must be at least 1, not {target}")
    v = _core._fast_len(target, real, prev)
    if v < 0:
        raise ValueError(f"{name}(): the next fast length after {target} is beyond int64")
    return v


@function
def next_fast_len(target, /, *, real=False):
    """The smallest fast length >= target: 11-smooth (every prime factor at most 11), or 5-smooth with
    real=True (scipy's values)."""
    return _fast("next_fast_len", target, real, False)


@function
def prev_fast_len(target, /, *, real=False):
    """The largest fast length <= target: 11-smooth, or 5-smooth with real=True (scipy's values)."""
    return _fast("prev_fast_len", target, real, True)
