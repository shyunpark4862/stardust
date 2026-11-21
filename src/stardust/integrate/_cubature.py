"""cubature (docs/api/integrate.md §5; docs/design/integrate.md §3.4 and decisions C5, C94-C107;
scipy 1.18.1's scipy.integrate.cubature with its rules in scipy/integrate/_rules).

The arguments are checked here; the integration is the native stardust._core.integrate.cubature_solve
(csrc/integrate/adaptive.cpp): scipy's algorithm with the same evaluation points and steps. f is
called with the nodes of a region as an (n_points, ndim) float64 sd.ndarray and gives
(n_points, ...). The result does not unpack (C21); converged and an integer status take the
place of scipy's strings 'converged' and 'not_converged' (C94).
"""

import math
import numbers

from .. import _core as _sd
from ..differentiate._derivative import _real_array, _shape_text
from ..optimize._common import _callable, _check_int, _real, _type_name
from ._quad_vec import MESSAGES
from ._result import IntegrateResult

_k = _sd.integrate
_RULES = ("gk21", "gk15", "genz-malik")
_ALIASES = {"gauss-kronrod": "gk21"}  # scipy's other name, not taken (no aliases, §1.1)


def _limits(v, func, arg):
    """a or b: a 1-D sequence of real numbers (+-inf allowed, NaN not) as a list of floats."""
    if isinstance(v, (str, bytes, dict)) or isinstance(v, numbers.Number) or not hasattr(v, "__len__"):
        raise TypeError(f"{func}(): '{arg}' must be a 1-D sequence of numbers, not {_type_name(v)}")
    if isinstance(v, _sd.ndarray) or hasattr(v, "tolist"):
        if getattr(v, "ndim", 1) != 1:
            raise ValueError(f"{func}(): '{arg}' must be 1-dimensional")
        v = v.tolist()
    out = []
    for t in v:
        if isinstance(t, bool) or not isinstance(t, numbers.Real):
            raise TypeError(f"{func}(): the values of '{arg}' must be real numbers, not {_type_name(t)}")
        t = _real(t, func, arg)
        if math.isnan(t):
            raise ValueError(f"{func}(): the values of '{arg}' must not be NaN")
        out.append(t)
    return out


def _tolerance(v, func, name):
    x = _real(v, func, name)
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): '{name}' must be a finite number >= 0")
    return x


def _values(func):
    """check(v, (n, ndim)) for the native kernel: f's value at n points as a float64 sd.ndarray
    of shape (n,) or (n, ...), or the error."""

    def check(v, xshape):
        n, d = xshape
        a = _real_array(v, func, "f")
        if a is None or a.ndim == 0 or a.shape[0] != n:
            got = "a scalar" if a is None or a.ndim == 0 else _shape_text(a.shape)
            raise ValueError(
                f"{func}(): 'f' must return shape ({n},) or ({n}, ...) for its argument of shape ({n}, {d}), not {got}"
            )
        return a if a.dtype == _sd.float64 else a.astype(_sd.float64)

    return check


def cubature(f, a, b, /, *, rule="gk21", rtol=1e-08, atol=0.0, max_subdivisions=10000, points=None):
    """The integral of the array-valued f over the box [a, b] by scipy's adaptive cubature
    (docs/api/integrate.md §5): rule 'gk21', 'gk15' (Gauss-Kronrod products) or 'genz-malik'."""
    fn = "cubature"
    _callable(f, fn, "f")
    lo = _limits(a, fn, "a")
    hi = _limits(b, fn, "b")
    if not lo or not hi:
        raise ValueError(f"{fn}(): 'a' and 'b' must not be empty")
    if len(lo) != len(hi):
        raise ValueError(f"{fn}(): 'a' and 'b' must have the same length, not {len(lo)} and {len(hi)}")
    d = len(lo)
    for i, (x, y) in enumerate(zip(lo, hi)):
        if x == y and math.isinf(x):
            raise ValueError(f"{fn}(): 'a' and 'b' must not be the same infinity (variable {i})")
    if not isinstance(rule, str):
        raise TypeError(f"{fn}(): 'rule' must be 'gk21', 'gk15' or 'genz-malik', not {_type_name(rule)}")
    if rule not in _RULES:
        hint = f" (scipy's {rule!r} is {_ALIASES[rule]!r})" if rule in _ALIASES else ""
        raise ValueError(f"{fn}(): 'rule' must be 'gk21', 'gk15' or 'genz-malik', not {rule!r}{hint}")
    if rule == "genz-malik" and d < 2:
        raise ValueError(f"{fn}(): rule='genz-malik' needs ndim >= 2 (it is defined from 2 variables), not {d}")
    rtol = _tolerance(rtol, fn, "rtol")
    atol = _tolerance(atol, fn, "atol")
    max_subdivisions = _check_int(max_subdivisions, fn, "max_subdivisions", 1)
    pts = []
    if points is not None:
        if isinstance(points, (str, bytes, dict)) or not hasattr(points, "__len__"):
            raise TypeError(f"{fn}(): 'points' must be a sequence of points or None, not {_type_name(points)}")
        for p in points:
            q = _limits(p, fn, "points")
            if len(q) != d:
                raise ValueError(f"{fn}(): each of 'points' must have ndim = {d} values, not {len(q)}")
            pts.append(q)
    integral, error, status, subdivisions, nfev, bounds, values, errors = _k.cubature_solve(
        fn, f, _values(fn), lo, hi, _RULES.index(rule), rtol, atol, max_subdivisions, pts
    )
    return IntegrateResult._make(
        integral=integral,
        error=error,
        converged=status == 0,
        status=status,
        message=MESSAGES[status],
        nfev=nfev,
        subdivisions=subdivisions,
        regions=bounds,
        region_integrals=values,
        region_errors=errors,
        n_regions=int(bounds.shape[0]),
    )
