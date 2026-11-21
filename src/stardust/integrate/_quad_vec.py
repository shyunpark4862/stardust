"""quad_vec (docs/api/integrate.md §4; docs/design/integrate.md §3.3 and decisions C5, C10, C94-C107;
scipy 1.18.1's scipy.integrate.quad_vec).

The arguments are checked here; the integration is the native stardust._core.integrate.quad_vec_solve
(csrc/integrate/adaptive.cpp): scipy's Python algorithm with the same evaluation points and the
same steps (numpy's and Python's arithmetic, Python's heap, scipy's cache of the intervals'
integrals). f is called at one Python float a time and gives a number or an array of one shape;
the result unpacks into (integral, error) like scipy's tuple (C10).
"""

import math
import numbers

from .. import _core as _sd
from ..differentiate._derivative import _real_array
from ..optimize._common import _callable, _check_choice, _check_int, _real, _type_name
from ._quadpack import _limit_value, _points
from ._result import IntegrateResult

_k = _sd.integrate
_NORMS = ("2", "max")
_RULES = ("gk21", "gk15", "trapezoid")
_RULE_CODE = {None: 0, "gk21": 1, "gk15": 2, "trapezoid": 3}

# scipy's texts for its status codes (CONVERGED, NOT_CONVERGED, ROUNDING_ERROR, NOT_A_NUMBER)
MESSAGES = {
    0: "Target precision reached.",
    1: "Target precision not reached.",
    2: "Target precision could not be reached due to rounding error.",
    3: "Non-finite values encountered.",
}


def _value_check(func):
    """check(v) for the native kernel: a value of f that is not a Python float, an sd.ndarray of
    float64 or a float64 buffer, as a float or a float64 sd.ndarray of one dimension or more."""

    def check(v):
        if isinstance(v, (str, bytes, bool)) or v is None:
            raise TypeError(f"{func}(): 'f' must return a real number or an array of them, not {_type_name(v)}")
        if isinstance(v, numbers.Real):
            return float(v)
        if isinstance(v, numbers.Complex):
            raise TypeError(f"{func}(): 'f' must return real values, not {_type_name(v)}")
        a = _real_array(v, func, "f")
        if a is None:  # a 0-d array
            kind = getattr(getattr(v, "dtype", None), "kind", "f")
            if kind == "c":
                raise TypeError(f"{func}(): 'f' must return real values, not a complex value")
            if kind == "b":
                raise TypeError(f"{func}(): 'f' must return real values, not a bool value")
            return float(v)
        if a.ndim == 0:
            return float(a)
        return a if a.dtype == _sd.float64 else a.astype(_sd.float64)

    return check


def _tolerance(v, func, name):
    x = v if type(v) is float else _real(v, func, name)
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): '{name}' must be a finite number >= 0")
    return x


def quad_vec(f, a, b, /, *, atol=1e-200, rtol=1e-08, norm="2", limit=10000, points=None, quadrature=None):
    """The integral of the number- or array-valued f on [a, b] by scipy's adaptive Gauss-Kronrod
    (or trapezoid) rule (docs/api/integrate.md §4); the result unpacks into (integral, error)."""
    fn = "quad_vec"
    _callable(f, fn, "f")
    a = _limit_value(a, fn, "a")
    b = _limit_value(b, fn, "b")
    atol = _tolerance(atol, fn, "atol")
    rtol = _tolerance(rtol, fn, "rtol")
    norm = _check_choice(norm, fn, "norm", _NORMS)
    limit = _check_int(limit, fn, "limit", 1)
    pts = _points(points, fn)
    if quadrature is not None:
        _check_choice(quadrature, fn, "quadrature", _RULES)
    integral, error, status, nfev, ends, values, errors = _k.quad_vec_solve(
        fn,
        f,
        _value_check(fn),
        a,
        b,
        atol,
        rtol,
        _NORMS.index(norm),
        limit,
        pts or [],
        pts is not None,
        _RULE_CODE[quadrature],
    )
    return IntegrateResult._quad(
        {
            "integral": integral,
            "error": error,
            "converged": status == 0,
            "status": status,
            "message": MESSAGES[status],
            "nfev": nfev,
            "intervals": ends,  # read-only arrays made by the native code
            "interval_integrals": values,
            "interval_errors": errors,
            "n_intervals": int(ends.shape[0]),
        }
    )
