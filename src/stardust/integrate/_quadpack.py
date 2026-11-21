"""quad, dblquad, tplquad and nquad (docs/api/integrate.md §2, §3; docs/design/integrate.md §3.1,
§3.2, decisions C5, C10, C11, C55-C66 and C71-C83).

The arguments are checked here; the integration is the native stardust._core.integrate.quad_solve
(one quad: scipy 1.18.1's dispatch around QUADPACK's QAGS, QAGI and QAGP and, with a weight,
QAWO, QAWF, QAWS and QAWC, the points and results scipy's bit for bit) and nquad_solve (the
nested quads of nquad, dblquad and tplquad, every level in C++ with the user's functions called
back). Each returns an IntegrateResult that unpacks into (integral, error) like scipy's tuples
(C10). What depends on the range (a weight on an infinite range, QAWF's tolerances, the Cauchy
point at an end) is checked by the native code, after the empty range's shortcut.
"""

import math
import numbers

from .. import _core as _sd
from ..optimize._common import (
    _callable,
    _check_bool,
    _check_int,
    _real,
    _real_pair,
    _scalar_check,
    _type_name,
    _values_text,
)
from ._result import IntegrateResult

_Q = _sd.integrate
_TOL = 1.49e-8  # scipy's epsabs and epsrel
_LIMIT = 50
_EPS = 2.0**-52
_INT_MAX = 2**31 - 1  # QUADPACK's limit is a C int
_WEIGHTS = ("cos", "sin", "alg", "alg-loga", "alg-logb", "alg-log", "cauchy")
_WEIGHT_CODE = {w: i + 1 for i, w in enumerate(_WEIGHTS)}  # the native codes (0: no weight)
_MAXP1 = _LIMLST = 50
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


# QAWF's texts for ier 1, 4 and 7 (scipy's, for 'cos' and 'sin' on an infinite range)
_QAWF_MESSAGES = {
    1: "The maximum number of cycles allowed has been achieved., e.e.\n  of subintervals (a+(k-1)c, a+kc) where c = "
    "(2*int(abs(omega)+1))\n  *pi/abs(omega), for k = 1, 2, ..., lst.  One can allow more cycles by increasing the "
    "value of limlst.  Look at info['ierlst'] with full_output=1.",
    4: "The extrapolation table constructed for convergence acceleration\n  of the series formed by the integral "
    "contributions over the cycles, \n  does not converge to within the requested accuracy.  Look at \n  "
    "info['ierlst'] with full_output=1.",
    7: "Bad integrand behavior occurs within one or more of the cycles.\n  Location and type of the difficulty "
    "involved can be determined from \n  the vector info['ierlist'] obtained with full_output=1.",
}


# a normal end (QUADPACK's 0) with an integral or error that is not finite: not converged, status
# -3 (tanhsinh's and nsum's number) with quad_vec's text (C119, [scipy differs]: scipy's ier 0)
_NONFINITE = -3
_MESSAGES[_NONFINITE] = "Non-finite values encountered."


def _status(ier, value, err):
    """QUADPACK's ier, or _NONFINITE for ier 0 with a value or error that is not finite."""
    if ier == 0 and not (math.isfinite(value) and math.isfinite(err)):
        return _NONFINITE
    return ier


def _message(ier, limit, qawf=False):
    if ier == 0 or ier == _NONFINITE:
        return _MESSAGES[ier]
    if qawf and ier in _QAWF_MESSAGES:
        return _QAWF_MESSAGES[ier]
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
_DEFAULT_OPTS = (_TOL, _TOL, _LIMIT, None, 0, None, _MAXP1, _LIMLST)

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
    """weight and wvar as the native (code, wvar): code 0 without a weight; wvar the frequency
    omega (cos, sin) or the point c (cauchy) as a float, (alpha, beta) for the 'alg' weights."""
    if weight is None:
        if wvar is not None:
            raise ValueError(f"{func}(): 'wvar' is only used with 'weight'")
        return 0, None
    if not isinstance(weight, str):
        raise TypeError(f"{func}(): 'weight' must be a str or None, not {_type_name(weight)}")
    if weight not in _WEIGHTS:
        raise ValueError(f"{func}(): 'weight' must be one of {', '.join(map(repr, _WEIGHTS))}, not {weight!r}")
    if weight.startswith("alg"):
        if type(wvar) is tuple and len(wvar) == 2 and type(wvar[0]) is float and type(wvar[1]) is float:
            alfa, beta = wvar  # the checks' fast path (finite below)
            if not (math.isfinite(alfa) and math.isfinite(beta)):
                raise ValueError(f"{func}(): the values of 'wvar' must be finite")
        else:
            alfa, beta = _real_pair(wvar, func, "wvar", "pair (alpha, beta) of real numbers")
        if not (alfa > -1 and beta > -1):
            raise ValueError(f"{func}(): 'wvar' (alpha, beta) must both be > -1 for weight={weight!r}")
        return _WEIGHT_CODE[weight], (alfa, beta)
    w = wvar if type(wvar) is float else _real(wvar, func, "wvar")
    if not math.isfinite(w):
        raise ValueError(f"{func}(): 'wvar' must be finite for weight={weight!r}")
    return _WEIGHT_CODE[weight], w


def _opts(func, atol, rtol, limit, points, weight=None, wvar=None, max_p1=_MAXP1, limlst=_LIMLST):
    """One quad's options as the native tuple (atol, rtol, limit, points or None, weight code,
    wvar, max_p1, limlst)."""
    # the fast paths compare only Python ints and floats (an array's == is not a bool)
    if not (type(atol) is float and type(rtol) is float and atol == _TOL and rtol == _TOL):
        atol, rtol = _tolerances(atol, rtol, func)
    if not (type(limit) is int and 1 <= limit <= _INT_MAX):
        limit = _limit_count(limit, func)
    points = None if points is None else _points(points, func)
    if not (type(max_p1) is int and max_p1 == _MAXP1):
        max_p1 = _limit_count(max_p1, func, "max_p1")
    if not (type(limlst) is int and limlst == _LIMLST):
        limlst = _check_int(limlst, func, "limlst", 3)
        if limlst > _INT_MAX:
            raise OverflowError(f"{func}(): 'limlst' is out of range")
    code, wv = _weight(weight, wvar, func)
    if code and points is not None:
        raise ValueError(f"{func}(): 'points' is not used with 'weight'")
    if code and weight.startswith("alg") and limit < 2:
        raise ValueError(f"{func}(): 'limit' must be >= 2 with weight={weight!r}")
    if max_p1 != _MAXP1 and weight not in ("cos", "sin"):
        raise ValueError(f"{func}(): 'max_p1' is only used with weight='cos' or 'sin'")
    if limlst != _LIMLST and weight not in ("cos", "sin"):
        raise ValueError(f"{func}(): 'limlst' is only used with weight='cos' or 'sin' on an infinite range")
    return (atol, rtol, limit, points, code, wv, max_p1, limlst)


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
    value, err, neval, ier, last, arrays, qawf = r
    ier = _status(ier, value, err)
    if qawf:  # QAWF: the cycles (scipy's lst, rslst, erlst, ierlst)
        values, errors, iers = arrays
        return IntegrateResult._quad(
            {
                "integral": value,
                "error": err,
                "converged": ier == 0,
                "status": ier,
                "message": _message(ier, limit, True),
                "nfev": neval,
                "cycle_integrals": values,  # read-only arrays made by the native code
                "cycle_errors": errors,
                "cycle_status": iers,
                "n_cycles": last,
            }
        )
    ends, values, errors = arrays
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
    range, QAGI on an infinite one, QAGP with break points, QAWO, QAWF, QAWS or QAWC with a
    weight; the result unpacks into (integral, error)."""
    fn = "quad"
    if not callable(func):
        _callable(func, fn, "func")
    if type(a) is not float or a != a:  # the checks' fast path: a float that is not NaN
        a = _limit_value(a, fn, "a")
    if type(b) is not float or b != b:
        b = _limit_value(b, fn, "b")
    if (  # the defaults (the types first: an array's == is not a bool)
        type(atol) in (int, float)
        and type(rtol) in (int, float)
        and type(limit) is int
        and type(max_p1) is int
        and type(limlst) is int
        and atol == _TOL
        and rtol == _TOL
        and limit == _LIMIT
        and points is None
        and weight is None
        and wvar is None
        and max_p1 == _MAXP1
        and limlst == _LIMLST
    ):
        opts = _DEFAULT_OPTS
    else:
        opts = _opts(fn, atol, rtol, limit, points, weight, wvar, max_p1, limlst)
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
        raise ValueError(f"{func}(): {what} must be a pair (lo, hi) of numbers, not {_values_text(len(v))}")
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
    return _opts(
        func,
        d.get("atol", _TOL),
        d.get("rtol", _TOL),
        d.get("limit", _LIMIT),
        d.get("points"),
        d.get("weight"),
        d.get("wvar"),
    )


def _solve(fn, func, levels, range_check, opts_check):
    value, err, nfev, ier, limit, qawf = _Q.nquad_solve(fn, func, _CHECK[fn], levels, range_check, opts_check)
    ier = _status(ier, value, err)
    return IntegrateResult._quad(
        {
            "integral": value,
            "error": err,
            "converged": ier == 0,
            "status": ier,
            "message": _message(ier, limit, qawf),
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
        return _opts_dict(v, fn, f"the result of 'opts[{i}]'")

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
