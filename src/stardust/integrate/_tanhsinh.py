"""tanhsinh and nsum (docs/api/integrate.md §6; docs/design/integrate.md §3.5, §3.6 and decisions
C4, C84-C93; scipy 1.18.1's scipy.integrate.tanhsinh and nsum).

Both are of the elementwise family (C4): the limits (and nsum's step) and args broadcast
together, each element is integrated or summed on its own, the first call is f on all elements
as (k,) (scipy's eim._initialize: tanhsinh at each interval's midpoint, nsum at a), then f gets
(k, p) arrays of the elements in progress with args as (k, 1) (C26). The steps and evaluation
points are scipy's to the bit (C5, C84, C87): the tanh-sinh table of scipy's formula with libm, numpy's
pairwise sums and scipy's logsumexp. The iterations are the native kernels
stardust._core.integrate.tanhsinh_solve and nsum_solve (csrc/integrate/tanhsinh.cpp).
"""

import math

from .. import _core as _sd
from ..differentiate._derivative import _values
from ..optimize._common import _callable, _check_bool, _check_int, _real
from ..optimize._elementwise import _args, _inputs, _shaped
from ._result import IntegrateResult

_k = _sd.integrate

_EPS = 2.0**-52
_MAX_LEVEL = 16  # the largest max_level (C88)
_MAX_TERMS = 2**24  # the largest max_terms (C89)


def _tolerance(v, default, log, func, name):
    """atol or rtol: a finite number >= 0; with log=True its log, a real number below +inf (-inf is
    a tolerance of 0)."""
    if v is None:
        return default
    x = _real(v, func, name)
    if log:
        if math.isnan(x) or x == math.inf:
            raise ValueError(
                f"{func}(): with log=True, '{name}' must be a real number below inf (the log of a tolerance)"
            )
    elif not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): '{name}' must be a finite number >= 0")
    return x


def _fields(shape, status, nfev, **values):
    out = {k: _shaped(v, shape) for k, v in values.items()}
    out.update(converged=_shaped(status == 0, shape), status=_shaped(status, shape), nfev=_shaped(nfev, shape))
    return out


def tanhsinh(f, a, b, /, *, args=(), log=False, max_level=None, min_level=2, atol=None, rtol=None):
    """The integral of the elementwise f from a to b by tanh-sinh quadrature, each element on its
    own (scipy.integrate.tanhsinh; docs/api/integrate.md §6)."""
    func = "tanhsinh"
    _callable(f, func, "f")
    log = _check_bool(log, func, "log")
    max_level = 10 if max_level is None else _check_int(max_level, func, "max_level", 0)
    if max_level > _MAX_LEVEL:
        raise ValueError(f"{func}(): 'max_level' must be at most {_MAX_LEVEL}")
    min_level = _check_int(min_level, func, "min_level", 0)
    atol = _tolerance(atol, -math.inf if log else 0.0, log, func, "atol")
    rtol = _tolerance(rtol, 0.75 * math.log(_EPS) if log else _EPS**0.75, log, func, "rtol")
    (av, bv), extra, shape = _inputs(func, [("a", a), ("b", b)], _args(args, func))
    if log and bool(_sd.any(bv < av)):
        raise ValueError(
            f"{func}(): with log=True, 'b' must not be less than 'a' (the log of the negative integral is complex)"
        )
    integral, error, status, nfev, level = _k.tanhsinh_solve(
        func, f, _values(func), extra, av, bv, log, max_level, min(min_level, max_level), atol, rtol
    )
    return IntegrateResult._make(
        batch=shape, **_fields(shape, status, nfev, integral=integral, error=error), max_level=_shaped(level, shape)
    )


def _n_steps(max_terms):
    """scipy's candidates for the first term below the tolerance: a + n step for n = 2^0, ...,
    2^(floor(log2 max_terms) - 1) and max_terms."""
    top = math.floor(math.log2(max_terms)) if max_terms else 0
    return _sd.asarray([float(2**i) for i in range(top)] + [float(max_terms)], dtype=_sd.float64)


def nsum(f, a, b, /, *, step=1, args=(), log=False, max_terms=1048576, atol=None, rtol=None):
    """The sum of f(a + k step) over a <= a + k step <= b (b may be infinite) for the smooth,
    positive, unimodal elementwise f, each element on its own: directly up to max_terms terms,
    by the integral test past them (scipy.integrate.nsum; docs/api/integrate.md §6)."""
    func = "nsum"
    _callable(f, func, "f")
    log = _check_bool(log, func, "log")
    max_terms = _check_int(max_terms, func, "max_terms", 0)
    if max_terms > _MAX_TERMS:
        raise ValueError(f"{func}(): 'max_terms' must be at most {_MAX_TERMS} (2**24)")
    atol = _tolerance(atol, -math.inf if log else 0.0, log, func, "atol")
    rtol = _tolerance(rtol, 0.5 * math.log(_EPS) if log else _EPS**0.5, log, func, "rtol")
    (av, bv, sv), extra, shape = _inputs(func, [("a", a), ("b", b), ("step", step)], _args(args, func))
    s, error, status, nfev = _k.nsum_solve(
        func, f, _values(func), extra, av, bv, sv, log, _n_steps(max_terms), max_terms, atol, rtol
    )
    return IntegrateResult._make(batch=shape, **_fields(shape, status, nfev, sum=s, error=error))
