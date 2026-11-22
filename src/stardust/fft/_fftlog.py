"""FFTLog of sd.fft (docs/api/fft.md §6; docs/design/fft.md §3.9; milestone FT5): fht, ifht and fhtoffset,
the fast Hankel transform of logarithmically spaced samples with scipy.fft's definitions.

The arguments are checked here in the order of the signature (the data, dln, mu, offset or initial, bias);
the transform runs in stardust._core.fft (csrc/fft/fftlog.hpp, fftlog.cpp): the coefficients
2^q Gamma((mu + 1 + q) / 2 + i y) / Gamma((mu + 1 - q) / 2 - i y) e^(2 i y (ln 2 - offset)) are formed in
double-double and rounded once (decision FT58), kept in a plan cache per (n, dln, mu, offset, bias,
direction), and every line along the last axis is biased, transformed by the real FFT, multiplied, transformed
back, reversed and biased again by the axis driver: a line has the same bits in any batch, layout and thread
count. A singular transform (a pole of one Gamma at y = 0) sets that coefficient to 0 as scipy does, without
scipy's warning (docs/api/fft.md §1.4).
"""

import math

from .. import _core as _sd
from .._calls import function
from . import _common as C

_core = _sd.fft
_REAL = ("bool", "int64", "float64")


def _data(a, name, arg):
    a = C.array(a, name, arg)
    kind = str(a.dtype)
    if kind not in _REAL:
        if kind == "complex128":
            raise TypeError(f"{name}(): '{arg}' must be real, not complex128")
        raise TypeError(f"{name}(): '{arg}' must be numeric, not a {kind} array")
    return a


def _finite(v, name, arg):
    v = C.real(v, name, arg)
    if not math.isfinite(v):
        raise ValueError(f"{name}(): '{arg}' must be finite, not {v!r}")
    return v


def _params(name, dln, mu, third, third_name, bias):
    dln = C.finite_nonzero(C.real(dln, name, "dln"), name, "dln")
    return dln, _finite(mu, name, "mu"), _finite(third, name, third_name), _finite(bias, name, "bias")


def _transform(name, arg, a, dln, mu, offset, bias, inverse):
    a = _data(a, name, arg)
    dln, mu, offset, bias = _params(name, dln, mu, offset, "offset", bias)
    if a.shape[-1] == 0:
        raise ValueError(f"{name}(): cannot transform an axis of length 0")
    return _core._fht(a, dln, mu, offset, bias, inverse)


@function
def fht(a, dln, mu, /, *, offset=0.0, bias=0.0):
    """The fast Hankel transform of the logarithmically spaced periodic sequence a (spacing dln, Bessel order
    mu) along its last axis, by FFTLog (scipy's definition): the output's spacing is dln and its centre
    log(k_c) = offset - log(r_c); bias q transforms a(r) (k r)^(-q) and biases the result back. float64."""
    return _transform("fht", "a", a, dln, mu, offset, bias, False)


@function
def ifht(A, dln, mu, /, *, offset=0.0, bias=0.0):
    """The inverse of fht with the same dln, mu, offset and bias (scipy's definition): the coefficients'
    reciprocals and the reciprocal biases. float64."""
    return _transform("ifht", "A", A, dln, mu, offset, bias, True)


@function
def fhtoffset(dln, mu, /, *, initial=0.0, bias=0.0):
    """The offset closest to `initial` that satisfies FFTLog's low-ringing condition for fht with spacing
    dln, order mu and bias (scipy's value: initial + (arg - round(arg)) dln, rounding half to even). float."""
    dln, mu, initial, bias = _params("fhtoffset", dln, mu, initial, "initial", bias)
    return _core._fhtoffset(dln, mu, initial, bias)
