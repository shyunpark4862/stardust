"""Savitzky-Golay filters: savgol_coeffs and savgol_filter (docs/api/smooth.md §6; docs/design/smooth.md §3.7,
§4.6; decisions SM24, SM25, SM32, SM70-SM76).

The meaning is scipy.signal's: the least-squares polynomial of degree polyorder over a window of window_length
equally spaced points, its deriv-th derivative at the position pos (the centre by default), as a linear filter.
The coefficients are the exact least-squares ones to 8 u max|h| (the Gram polynomials in double-double,
csrc/smooth/savgol.cpp); the filter runs along one axis of an n-D array of any layout, real or complex.
"""

import math
import numbers

from .. import _core as _sd
from ._common import _TAKES, contiguous, integer, real, type_name

_core = _sd.smooth
MODES = ("interp", "mirror", "nearest", "constant", "wrap")
_MODE_CODE = {m: i for i, m in enumerate(MODES)}


def _choice(v, allowed, func, arg):
    listed = ", ".join(f"'{a}'" for a in allowed[:-1]) + f" or '{allowed[-1]}'"
    if not isinstance(v, str):
        raise TypeError(f"{func}(): '{arg}' must be {listed}, not {type_name(v)}")
    if v not in allowed:
        raise ValueError(f"{func}(): '{arg}' must be {listed}, not {v!r}")
    return v


def _filter_args(window_length, polyorder, deriv, delta, func):
    N = integer(window_length, func, "window_length")
    d = integer(polyorder, func, "polyorder")
    nu = integer(deriv, func, "deriv")
    dl = real(delta, func, "delta")
    if N < 1:
        raise ValueError(f"{func}(): 'window_length' must be >= 1, not {N}")
    if d < 0:
        raise ValueError(f"{func}(): 'polyorder' must be >= 0, not {d}")
    if d >= N:
        raise ValueError(f"{func}(): 'polyorder' must be less than 'window_length' ({d} >= {N})")
    if nu < 0:
        raise ValueError(f"{func}(): 'deriv' must be >= 0, not {nu}")
    if dl == 0 or not math.isfinite(dl):
        raise ValueError(f"{func}(): 'delta' must be finite and nonzero, not {dl!r}")
    return N, d, nu, dl


def _check_work(N, d, nu, func):
    """MemoryError before the node values (16 N (d + 1) bytes) outgrow this machine's physical memory."""
    if nu > d:
        return
    nbytes = 16 * N * (d + 1)
    if nbytes > _sd._physical_memory():
        raise MemoryError(f"{func}(): cannot allocate {nbytes} bytes: more than this machine's physical memory")


def savgol_coeffs(window_length, polyorder, /, *, deriv=0, delta=1.0, pos=None, use="conv"):
    """The coefficients of a Savitzky-Golay filter: (window_length,) float64.

    Fits a polynomial of degree polyorder to window_length equally spaced points (spacing delta) by least squares
    and gives the weights that produce its deriv-th derivative at the point pos of the window (the centre when
    None; half a step off the middle for an even window). use='conv' orders them for a convolution (reversed),
    'dot' for a dot product with the window. Every coefficient is within 8 u max|h| of the exact rational value
    (window_length <= 1001); deriv > polyorder gives zeros.
    """
    func = "savgol_coeffs"
    N, d, nu, dl = _filter_args(window_length, polyorder, deriv, delta, func)
    if pos is None:
        p = -1
    else:
        p = integer(pos, func, "pos")
        if not 0 <= p < N:
            raise ValueError(f"{func}(): 'pos' must be >= 0 and < 'window_length', not {p}")
    _choice(use, ("conv", "dot"), func, "use")
    _check_work(N, d, nu, func)
    h = _core.savgol_coeffs(N, d, nu, dl, p)
    return contiguous(h[::-1]) if use == "conv" else h


def _signal(x, func):
    """x as a float64 or complex128 sd.ndarray of at least one dimension, C-contiguous."""
    if isinstance(x, numbers.Number) or (type(x).__module__ == "numpy" and getattr(x, "ndim", None) == 0):
        raise ValueError(f"{func}(): 'x' must have at least 1 dimension")
    if isinstance(x, _sd.ndarray):
        a = x
    elif type(x).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if not isinstance(x, _frame.Series):
            raise TypeError(f"{func}(): 'x' must be {_TAKES}, not {type_name(x)}")
        try:
            a = x.to_array(dtype=_sd.float64, na_value=math.nan)
        except TypeError, ValueError:
            raise TypeError(f"{func}(): 'x' must have a numeric column") from None
    else:
        if x is None or isinstance(x, (str, bytes)):
            raise TypeError(f"{func}(): 'x' must be {_TAKES}, not {type_name(x)}")
        try:
            a = _sd.asarray(x)
        except (TypeError, ValueError) as e:
            msg = str(e)
            if isinstance(e, ValueError):  # a ragged sequence, ...
                raise ValueError(f"{func}(): {msg.partition('(): ')[2] or msg}") from None
            raise TypeError(f"{func}(): 'x' must be {_TAKES}, not {type_name(x)}") from None
    if a.ndim == 0:
        raise ValueError(f"{func}(): 'x' must have at least 1 dimension")
    if a.dtype == _sd.string:
        raise TypeError(f"{func}(): 'x' must be numeric, not string")
    if a.dtype != _sd.complex128 and a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    return contiguous(a)


def savgol_filter(x, window_length, polyorder, /, *, deriv=0, delta=1.0, axis=-1, mode="interp", cval=0.0):
    """Applies a Savitzky-Golay filter along one axis of x: an array of x's shape, float64 (complex128 for complex
    x: the real and the imaginary parts filtered alike).

    Each output is the deriv-th derivative (spacing delta) of the degree-polyorder least-squares polynomial over
    the window_length points around it. mode says what the window reads past the ends: 'interp' (the default:
    the first and the last window_length // 2 outputs are the polynomial fitted to the first and the last window),
    'mirror', 'nearest', 'constant' (cval) or 'wrap', as scipy.ndimage extends an array. A NaN reaches every
    output whose window holds it.
    """
    func = "savgol_filter"
    N, d, nu, dl = _filter_args(window_length, polyorder, deriv, delta, func)
    ax = integer(axis, func, "axis")
    _choice(mode, MODES, func, "mode")
    cv = real(cval, func, "cval")
    a = _signal(x, func)
    if not -a.ndim <= ax < a.ndim:
        raise ValueError(f"{func}(): axis {ax} is out of bounds for an array of dimension {a.ndim}")
    ax %= a.ndim
    n = a.shape[ax]
    if mode == "interp" and N > n:
        raise ValueError(
            f"{func}(): 'window_length' must not exceed the length of the axis for mode 'interp' ({N} > {n})"
        )
    _check_work(N, d, nu, func)
    return _core.savgol_filter(a, N, d, nu, dl, ax, _MODE_CODE[mode], cv)
