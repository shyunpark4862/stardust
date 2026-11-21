"""quad, dblquad, tplquad and nquad (docs/api/integrate.md §2, §3; docs/design/integrate.md §3.1,
§3.2, decisions C5, C10, C11, C55-C66).

The arguments are checked here; the integration is the native stardust._core.integrate.quad_solve
(one quad: scipy 1.18.1's dispatch around QUADPACK's QAGS, QAGI and QAGP, the points and results
scipy's bit for bit) and nquad_solve (the nested quads of nquad, dblquad and tplquad, every level
in C++ with the user's functions called back). Each returns an IntegrateResult that unpacks into
(integral, error) like scipy's tuples (C10).

The weights of quad (weight, wvar, max_p1, limlst: QAWO, QAWF, QAWS, QAWC) come with milestone
C5 (docs/design/integrate.md §8); until then weight= raises NotImplementedError.
"""

import math
import numbers

from .. import _core as _sd
from ..optimize._common import _callable, _check_bool, _check_int, _real, _scalar_check, _type_name
from ._result import IntegrateResult

_Q = _sd.integrate
_TOL = 1.49e-8  # scipy's epsabs and epsrel
_LIMIT = 50
_EPS = 2.0**-52
_INT_MAX = 2**31 - 1  # QUADPACK's limit is a C int
_WEIGHTS = ("cos", "sin", "alg", "alg-loga", "alg-logb", "alg-log", "cauchy")
_OPT_KEYS = ("atol", "rtol", "limit", "points", "weight", "wvar")
_SCIPY_KEYS = {"epsabs": "atol", "epsrel": "rtol"}  # nquad's opts in scipy's names

# QUADPACK's ier as scipy 1.18.1's text (its "\n  " breaks kept, O86, C58); 0 is QUADPACK's own
# description of a normal end (scipy has no text for it, C59).
_MESSAGES = {
    0: "Normal and reliable termination of the routine. It is assumed that the requested accuracy has been achieved.",
    2: "The occurrence of roundoff error is detected, which prevents \n  the requested tolerance from being achieved.  "
    "The error may be \n  underestimated.",
    3: "Extremely bad integrand behavior occurs at some points of the\n  integration interval.",
    4: "The algorithm does not converge.  Roundoff error is detected\n  in the extrapolation table.  It is assumed "
    "that the requested tolerance\n  cannot be achieved, and that the returned result (if full_output = 1) is \n  "
    "the best which can be obtained.",
    5: "The integral is probably divergent, or slowly convergent.",
    7: "Abnormal termination of the routine.  The estimates for result\n  and error are less reliable.  It is assumed "
    "that the requested accuracy\n  has not been achieved.",
}


def _message(ier, limit):
    if ier == 0:
        return _MESSAGES[0]
    if ier == 1:
        return (
            f"The maximum number of subdivisions ({limit}) has been achieved.\n  If increasing the limit yields no "
            f"improvement it is advised to analyze \n  the integrand in order to determine the difficulties.  If the "
            f"position of a \n  local difficulty can be determined (singularity, discontinuity) one will \n  probably "
            f"gain from splitting up the interval and calling the integrator \n  on the subranges.  Perhaps a "
            f"special-purpose integrator should be used."
        )
    return _MESSAGES.get(ier, "Unknown error.")


_CHECK = {fn: _scalar_check(fn) for fn in ("quad", "dblquad", "tplquad", "nquad")}
_DEFAULT_OPTS = (_TOL, _TOL, _LIMIT, None)

# ---- the arguments


def _limit_value(v, func, arg):
    """An integration limit: a real number, +-inf allowed, NaN not."""
    x = _real(v, func, arg)
    if math.isnan(x):
        raise ValueError(f"{func}(): '{arg}' must be a number or +-inf, not NaN")
    return x


def _points(v, func, arg="points"):
    """quad's break points: None or a sequence of real numbers (not NaN), as a list of floats."""
    if v is None:
        return None
    if isinstance(v, (str, bytes, dict)) or isinstance(v, numbers.Number) or not hasattr(v, "__len__"):
        raise TypeError(f"{func}(): '{arg}' must be a sequence of numbers or None, not {_type_name(v)}")
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


def _tolerances(atol, rtol, func):
    atol = _real(atol, func, "atol")
    rtol = _real(rtol, func, "rtol")
    for name, v in (("atol", atol), ("rtol", rtol)):
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{func}(): '{name}' must be a finite number >= 0")
    # QUADPACK's ier = 6 for the tolerances (scipy's ValueError, its condition)
    if atol <= 0 and rtol < max(50 * _EPS, 5e-29):
        raise ValueError(f"{func}(): if 'atol' is 0, 'rtol' must be at least 50 * machine epsilon (1.11e-14)")
    return atol, rtol


def _limit_count(v, func, arg="limit"):
    v = _check_int(v, func, arg, 1)
    if v > _INT_MAX:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _weight(weight, wvar, func):
    """weight and wvar: milestone C5 (until then a weight is NotImplementedError)."""
    if weight is not None:
        if not isinstance(weight, str):
            raise TypeError(f"{func}(): 'weight' must be a str or None, not {_type_name(weight)}")
        if weight not in _WEIGHTS:
            raise ValueError(f"{func}(): 'weight' must be one of {', '.join(map(repr, _WEIGHTS))}, not {weight!r}")
        raise NotImplementedError(f"{func}(): 'weight' (QAWO, QAWF, QAWS, QAWC) comes with milestone C5")
    if wvar is not None:
        raise ValueError(f"{func}(): 'wvar' is only used with 'weight'")


def _opts(func, atol, rtol, limit, points):
    """One quad's options as the native tuple (atol, rtol, limit, points or None)."""
    atol, rtol = _tolerances(atol, rtol, func)
    return (atol, rtol, _limit_count(limit, func), _points(points, func))


def _complex_check(func):
    """check(value, what) for complex_func=True: a value that is not a Python float or complex,
    as a complex (a real or complex number, a 0-d numpy array), or the error."""
    real = _scalar_check(func)

    def check(v, what):
        if isinstance(v, numbers.Complex) and not isinstance(v, bool):
            return complex(v)
        kind = getattr(getattr(v, "dtype", None), "kind", "")
        if tuple(getattr(v, "shape", (1,))) == () and kind == "c":
            return complex(v)
        return complex(real(v, what))

    return check


# ---- the results


def _quad_result(r, limit):
    value, err, neval, ier, last, (ends, values, errors) = r
    return IntegrateResult._quad(
        {
            "integral": value,
            "error": err,
            "converged": ier == 0,
            "status": ier,
            "message": _message(ier, limit),
            "nfev": neval,
            "intervals": ends,  # read-only arrays made by the native code
            "interval_integrals": values,
            "interval_errors": errors,
            "n_intervals": last,
        }
    )


# ---- quad


def quad(
    func,
    a,
    b,
    /,
    *,
    atol=1.49e-08,
    rtol=1.49e-08,
    limit=50,
    points=None,
    weight=None,
    wvar=None,
    max_p1=50,
    limlst=50,
    complex_func=False,
):
    """The integral of func on [a, b] by QUADPACK (docs/api/integrate.md §2): QAGS on a finite
    range, QAGI on an infinite one, QAGP with break points; the result unpacks into
    (integral, error)."""
    fn = "quad"
    if not callable(func):
        _callable(func, fn, "func")
    if type(a) is not float or a != a:  # the checks' fast path: a float that is not NaN
        a = _limit_value(a, fn, "a")
    if type(b) is not float or b != b:
        b = _limit_value(b, fn, "b")
    if atol == _TOL and rtol == _TOL and limit == _LIMIT and points is None and type(limit) is int:
        opts = (
            (float(atol), float(rtol), _LIMIT, None)
            if type(atol) is not float or type(rtol) is not float
            else _DEFAULT_OPTS
        )
    else:
        opts = _opts(fn, atol, rtol, limit, points)
    if not (max_p1 == 50 and limlst == 50 and type(max_p1) is int and type(limlst) is int):
        _check_int(max_p1, fn, "max_p1", 1)
        _check_int(limlst, fn, "limlst", 3)
    if weight is not None or wvar is not None:
        _weight(weight, wvar, fn)
    if max_p1 != 50:
        raise ValueError(f"{fn}(): 'max_p1' is only used with weight='cos' or 'sin'")
    if limlst != 50:
        raise ValueError(f"{fn}(): 'limlst' is only used with weight='cos' or 'sin' on an infinite range")
    if complex_func is False:
        return _quad_result(_Q.quad_solve(fn, func, _CHECK[fn], a, b, opts, 0), opts[2])
    _check_bool(complex_func, fn, "complex_func")
    check = _complex_check(fn)
    re = _quad_result(_Q.quad_solve(fn, func, check, a, b, opts, 1), opts[2])
    im = _quad_result(_Q.quad_solve(fn, func, check, a, b, opts, 2), opts[2])
    bad = re if re.status != 0 else im
    return IntegrateResult._quad(
        {
            "integral": complex(re.integral, im.integral),
            "error": complex(re.error, im.error),
            "converged": re.converged and im.converged,
            "status": bad.status,
            "message": bad.message,
            "nfev": re.nfev + im.nfev,
            "real": re,
            "imag": im,
        }
    )


# ---- nquad, dblquad, tplquad


def _pair(v, func, what):
    """A range (lo, hi) of real numbers (+-inf allowed, NaN not), as a tuple of floats."""
    if isinstance(v, (str, bytes, dict)) or not hasattr(v, "__len__") or not hasattr(v, "__getitem__"):
        raise TypeError(f"{func}(): {what} must be a pair (lo, hi) of numbers, not {_type_name(v)}")
    if len(v) != 2:
        raise ValueError(f"{func}(): {what} must be a pair (lo, hi) of numbers, not {len(v)} values")
    out = []
    for t in (v[0], v[1]):
        if isinstance(t, bool) or not isinstance(t, numbers.Real):
            if hasattr(t, "shape") and tuple(t.shape) == () and getattr(t.dtype, "kind", "") in "iuf":
                t = float(t)
            else:
                raise TypeError(f"{func}(): {what} must be a pair (lo, hi) of real numbers, not {_type_name(t)}")
        t = _real(t, func, what.strip("'"))
        if math.isnan(t):
            raise ValueError(f"{func}(): {what} must not be NaN")
        out.append(t)
    return (out[0], out[1])


def _bound(v, func, name):
    """dblquad's and tplquad's gfun ... rfun at a point: a real number (not NaN)."""
    v = v if type(v) is float else _CHECK[func](v, name)
    if math.isnan(v):
        raise ValueError(f"{func}(): '{name}' returned NaN")
    return v


def _opts_dict(d, func, what):
    """An nquad level's options (a dict with quad's option names) as the native tuple."""
    if not isinstance(d, dict):
        raise TypeError(f"{func}(): {what} must be a dict, not {_type_name(d)}")
    for k in d:
        if k not in _OPT_KEYS:
            hint = f" (scipy's {k!r} is {_SCIPY_KEYS[k]!r})" if k in _SCIPY_KEYS else ""
            raise ValueError(f"{func}(): {what} has no option {k!r}{hint}; the options are {', '.join(_OPT_KEYS)}")
    _weight(d.get("weight"), d.get("wvar"), func)
    return _opts(func, d.get("atol", _TOL), d.get("rtol", _TOL), d.get("limit", _LIMIT), d.get("points"))


def _solve(fn, func, levels, range_check, opts_check):
    value, err, nfev, ier, limit = _Q.nquad_solve(fn, func, _CHECK[fn], levels, range_check, opts_check)
    return IntegrateResult._quad(
        {
            "integral": value,
            "error": err,
            "converged": ier == 0,
            "status": ier,
            "message": _message(ier, limit),
            "nfev": nfev,
        }
    )


def nquad(func, ranges, /, *, opts=None):
    """The integral of func(x0, x1, ..., xn) over ranges (innermost variable first) by nested quads
    (docs/api/integrate.md §3); the result unpacks into (integral, error)."""
    fn = "nquad"
    _callable(func, fn, "func")
    if isinstance(ranges, (str, bytes, dict)) or not hasattr(ranges, "__len__"):
        raise TypeError(f"{fn}(): 'ranges' must be a sequence of ranges, not {_type_name(ranges)}")
    ranges = list(ranges)
    n = len(ranges)
    if n == 0:
        raise ValueError(f"{fn}(): 'ranges' must have at least one range")
    shared = opts is None or isinstance(opts, dict)
    if opts is None:
        opts = [{}] * n
    elif isinstance(opts, dict):
        opts = [opts] * n
    elif isinstance(opts, (str, bytes)) or not hasattr(opts, "__len__"):
        raise TypeError(
            f"{fn}(): 'opts' must be a dict, a sequence of dicts or callables, or None, not {_type_name(opts)}"
        )
    else:
        opts = list(opts)
        if len(opts) != n:
            raise ValueError(f"{fn}(): 'opts' must have one entry per range ({n}), not {len(opts)}")
    levels = []
    for i, (rng, opt) in enumerate(zip(ranges, opts)):
        what = f"'ranges[{i}]'"
        r = rng if callable(rng) else _pair(rng, fn, what)
        o = opt if callable(opt) else _opts_dict(opt, fn, "'opts'" if shared else f"'opts[{i}]'")
        levels.append((r, o))

    def range_check(i, v):
        return _pair(v, fn, f"the range of 'ranges[{i}]'")

    def opts_check(i, v):
        return _opts_dict(v, fn, f"the options of 'opts[{i}]'")

    return _solve(fn, func, levels, range_check, opts_check)


def _fixed(v, func, name):
    if callable(v):
        return v
    return _bound(v, func, name)


def dblquad(func, a, b, gfun, hfun, /, *, atol=1.49e-08, rtol=1.49e-08):
    """The integral of func(y, x) for y from gfun(x) to hfun(x), x from a to b (docs/api/integrate.md
    §3); the result unpacks into (integral, error)."""
    fn = "dblquad"
    _callable(func, fn, "func")
    a = _limit_value(a, fn, "a")
    b = _limit_value(b, fn, "b")
    g = _fixed(gfun, fn, "gfun")
    h = _fixed(hfun, fn, "hfun")
    o = _opts(fn, atol, rtol, _LIMIT, None)
    if callable(g) or callable(h):

        def inner(x):  # scipy's temp_ranges: gfun, then hfun
            return (g(x) if callable(g) else g, h(x) if callable(h) else h)

        def range_check(i, v):
            return (_bound(v[0], fn, "gfun"), _bound(v[1], fn, "hfun"))

    else:
        inner = (g, h)
        range_check = None
    return _solve(fn, func, [(inner, o), ((a, b), o)], range_check or _no_check, _no_check)


def tplquad(func, a, b, gfun, hfun, qfun, rfun, /, *, atol=1.49e-08, rtol=1.49e-08):
    """The integral of func(z, y, x) for z from qfun(x, y) to rfun(x, y), y from gfun(x) to hfun(x),
    x from a to b (docs/api/integrate.md §3); the result unpacks into (integral, error)."""
    fn = "tplquad"
    _callable(func, fn, "func")
    a = _limit_value(a, fn, "a")
    b = _limit_value(b, fn, "b")
    g = _fixed(gfun, fn, "gfun")
    h = _fixed(hfun, fn, "hfun")
    q = _fixed(qfun, fn, "qfun")
    r = _fixed(rfun, fn, "rfun")
    o = _opts(fn, atol, rtol, _LIMIT, None)
    names = (("qfun", "rfun"), ("gfun", "hfun"))
    if callable(q) or callable(r):

        def r0(y, x):  # scipy's ranges0: qfun(x, y), then rfun(x, y)
            return (q(x, y) if callable(q) else q, r(x, y) if callable(r) else r)

    else:
        r0 = (q, r)
    if callable(g) or callable(h):

        def r1(x):  # scipy's ranges1
            return (g(x) if callable(g) else g, h(x) if callable(h) else h)

    else:
        r1 = (g, h)

    def range_check(i, v):
        lo, hi = names[i]
        return (_bound(v[0], fn, lo), _bound(v[1], fn, hi))

    return _solve(fn, func, [(r0, o), (r1, o), ((a, b), o)], range_check, _no_check)


def _no_check(i, v):  # a level without a function: never called
    raise AssertionError((i, v))
