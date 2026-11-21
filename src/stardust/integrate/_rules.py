"""fixed_quad and newton_cotes (docs/api/integrate.md §7; docs/design/integrate.md §3.7 and
decisions C42, C43, C52, C53; scipy 1.18.1's integrate/_quadrature.py).

fixed_quad is scipy's n-point Gauss-Legendre rule: func is called once at
(b - a) (x + 1) / 2.0 + a (scipy's expression) with the correctly rounded nodes x (._gauss), and
the weighted sum of its values is the library's pairwise sum (the sample rules' kernel).
newton_cotes computes the closed Newton-Cotes weights and error coefficient exactly in rationals
and rounds each once.
"""

import functools
import math
import numbers
import operator
from fractions import Fraction

from .. import _core as _sd
from ..optimize._common import _callable, _check_int, _real, _type_name
from . import _gauss
from ._samples import WEIGHTED, as_data

_k = _sd.integrate


def _limit(v, func, arg):
    x = _real(v, func, arg)
    if not math.isfinite(x):
        raise ValueError(f"{func}(): '{arg}' must be finite (Gaussian quadrature needs a finite interval)")
    return x


def _shape_text(shape):
    return "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"


def _values(v, n, func):
    """func's value at the n points: a float64 sd.ndarray of shape (n,) or (..., n)."""
    if isinstance(v, _sd.ndarray):
        a = v
    else:
        if isinstance(v, (str, bytes, bool)) or v is None:
            raise TypeError(f"{func}(): 'func' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}")
        if isinstance(v, numbers.Number) or getattr(v, "ndim", None) == 0:
            a = None
        else:
            try:
                a = _sd.asarray(v)
            except OverflowError:
                raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
            except TypeError, ValueError:
                raise TypeError(
                    f"{func}(): 'func' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
                ) from None
    if a is not None:
        if a.dtype == _sd.complex128:
            raise TypeError(f"{func}(): 'func' must return real values, not a complex128 array")
        if a.dtype == _sd.string:
            raise TypeError(f"{func}(): 'func' must return real values, not a string array")
        if a.dtype == _sd.bool:
            raise TypeError(f"{func}(): 'func' must return real values, not a bool array")
    if a is None or a.shape[-1] != n:
        got = "a scalar" if a is None else _shape_text(a.shape)
        raise ValueError(
            f"{func}(): 'func' must return shape ({n},) or (..., {n}) for its argument of shape ({n},), not {got}"
        )
    return a if a.dtype == _sd.float64 else a.astype(_sd.float64)


def fixed_quad(func, a, b, /, *, n=5):
    """The integral of func from a to b by n-point Gauss-Legendre quadrature
    (scipy.integrate.fixed_quad; docs/api/integrate.md §7): one value (C43), or an array for a
    func with values of shape (..., n)."""
    name = "fixed_quad"
    _callable(func, name, "func")
    a = _limit(a, name, "a")
    b = _limit(b, name, "b")
    n = _check_int(n, name, "n", 1)
    x, w = _gauss.legendre(n)
    y = (b - a) * (x + 1) / 2.0 + a
    v = _values(func(y), n, name)
    weights = w if v.ndim == 1 else _sd.broadcast_to(w, tuple(v.shape))
    s = _k.samples(name, WEIGHTED, v, weights, None, v.ndim - 1, None)
    return (b - a) / 2.0 * s


# ---- newton_cotes


def _positions(rn, equal, func):
    """(N, positions as Fractions, equally spaced) of rn: an int N (positions 0 ... N) or the
    relative positions of the N + 1 samples (from 0 to N)."""
    if isinstance(rn, numbers.Integral) and not isinstance(rn, bool):
        n = operator.index(rn)
        if n < 1:
            raise ValueError(f"{func}(): 'rn' must be >= 1 (the number of intervals)")
        return n, [Fraction(i) for i in range(n + 1)], True
    if isinstance(rn, bool) or type(rn).__name__ == "bool":
        raise TypeError(f"{func}(): 'rn' must be an int or a sequence of numbers, not {_type_name(rn)}")
    if isinstance(rn, numbers.Number):
        raise TypeError(f"{func}(): 'rn' must be an int or a sequence of numbers, not {_type_name(rn)}")
    r = as_data(rn, func, "rn", complex_ok=False)
    if r.ndim != 1:
        raise ValueError(f"{func}(): 'rn' must be 1-dimensional, not shape {_shape_text(r.shape)}")
    n = r.shape[0] - 1
    if n < 1:
        raise ValueError(f"{func}(): 'rn' must hold at least 2 positions")
    if equal:  # scipy: only the length counts
        return n, [Fraction(i) for i in range(n + 1)], True
    vals = r.tolist()
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{func}(): the sample positions must be finite")
    if vals[0] != 0 or vals[-1] != n:
        raise ValueError(f"{func}(): the sample positions must start at 0 and end at N = {n}")
    pos = [Fraction(v) for v in vals]
    if len(set(pos)) != len(pos):
        raise ValueError(f"{func}(): the sample positions must be distinct")
    return n, pos, all(pos[i + 1] - pos[i] == 1 for i in range(n))


def _weights_exact(n, pos):
    """a_i = the integral over [0, N] of the Lagrange basis polynomial of node i, exactly."""
    # W(s) = prod (s - r_j), coefficients from the constant term up
    coef = [Fraction(1)]
    for r in pos:
        nxt = [Fraction(0)] * (len(coef) + 1)
        for k, c in enumerate(coef):
            nxt[k + 1] += c
            nxt[k] -= r * c
        coef = nxt
    big = Fraction(n)
    powers = [big ** (k + 1) / (k + 1) for k in range(n + 1)]  # integral of s^k over [0, N]
    out = []
    for i, r in enumerate(pos):
        # Q(s) = W(s) / (s - r) by synthetic division (degree n), highest coefficient first
        q = [Fraction(0)] * (n + 1)
        acc = Fraction(0)
        for k in range(n + 1, 0, -1):
            acc = coef[k] + acc * r
            q[k - 1] = acc
        den = Fraction(1)
        for j, s in enumerate(pos):
            if j != i:
                den *= r - s
        out.append(sum(q[k] * powers[k] for k in range(n + 1)) / den)
    return out


@functools.lru_cache(maxsize=64)
def _newton_cotes_exact(n, pos, equally):
    """(the weights, B) as rounded doubles; cached, since the exact rationals cost far more than
    a call (0.2 ms for N = 8)."""
    a = _weights_exact(n, pos)
    p = n + 2 if (n % 2 == 0 and equally) else n + 1  # scipy's power of the error term
    big = Fraction(n)
    b = (big ** (p + 1) / (p + 1) - sum(ai * r**p for ai, r in zip(a, pos))) / math.factorial(p)
    return tuple(float(v) for v in a), float(b)


def newton_cotes(rn, /, *, equal=False):
    """The closed Newton-Cotes weights an and error coefficient B of rn (an int N, or the relative
    positions of N + 1 samples from 0 to N), each the exact value rounded once
    (scipy.integrate.newton_cotes; docs/api/integrate.md §7)."""
    func = "newton_cotes"
    if equal is not True and equal is not False:
        raise TypeError(f"{func}(): 'equal' must be True or False, not {_type_name(equal)}")
    n, pos, equally = _positions(rn, equal, func)
    a, b = _newton_cotes_exact(n, tuple(pos), equally)
    return _sd.asarray(list(a)), b
