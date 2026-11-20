"""Scalar roots and minima: root_scalar and minimize_scalar (docs/api/optimize.md §4, §5,
docs/design/optimize.md §3.2, §3.3).

The iterations are the native kernels stardust._core.optimize.root_solve and min_solve, each the
same steps as scipy 1.18.1's method (O9): brentq and bisect as scipy's C code, toms748 as its
TOMS748Solver, newton, halley and secant as its newton(), brent as its Brent class with bracket(),
golden and bounded (fminbound) as its _minimize_scalar_golden and _minimize_scalar_bounded. The
user's function gets a Python float and returns a real number (a value that is not a float is
checked here: _scalar_check).
"""

import math

from .. import _core as _sd
from ._common import (
    _EPS,
    _callable,
    _check_int,
    _check_method,
    _check_options,
    _check_tol,
    _finite_real,
    _real_pair,
    _scalar_check,
)
from ._result import OptimizeResult

_k = _sd.optimize

_ROOT_METHODS = ("brentq", "toms748", "bisect", "newton", "halley", "secant")
_ROOT_CODE = {"brentq": 0, "toms748": 1, "bisect": 2, "newton": 3, "halley": 4, "secant": 5}
_BRACKETING = ("brentq", "toms748", "bisect")
# the kernel's codes (csrc/optimize/scalar.cpp, RootStatus, RootReason)
_R_SIGN, _R_CONV, _R_NAN = -1, -2, -3
_DERIV_ZERO, _SECANT_TOL = 1, 2

_MIN_METHODS = ("brent", "golden", "bounded")
_MIN_CODE = {"brent": 0, "golden": 1, "bounded": 2}
_M_BAD_F = -1  # a three-point bracket without f(xb) below both ends (MinStatus)
# scipy 1.18.1's messages: _minimize_scalar_bounded's flags, and the same numbers for brent and
# golden (scipy gives them no status): their messages, _status_message['nan'], BracketError's
# and bracket()'s iteration limit (a RuntimeError in scipy)
_BOUNDED_MESSAGES = {
    0: "Solution found.",
    1: "Maximum number of function calls reached.",
    2: "NaN result encountered.",
}
_BRACKET_MESSAGES = {
    1: "\nMaximum number of iterations exceeded",
    2: "NaN result encountered.",
    3: "The algorithm terminated without finding a valid bracket. Consider trying different initial points.",
    4: "No valid bracket was found before the iteration limit was reached. Consider trying different "
    "initial points or increasing `maxiter`.",
}


def _not_given(func, method, **given):
    """ValueError for the first argument given that method does not take."""
    for name, v in given.items():
        if v is not None:
            raise ValueError(f"{func}(): method={method!r} takes no {name!r}")


def root_scalar(
    f,
    /,
    *,
    method=None,
    bracket=None,
    fprime=None,
    fprime2=None,
    x0=None,
    x1=None,
    xtol=None,
    rtol=None,
    max_iter=None,
    options=None,
):
    """A root of f(x) = 0: in a bracket (a, b) with f(a), f(b) of opposite signs ('brentq',
    'toms748', 'bisect') or from x0 ('newton', 'halley', 'secant') (docs/api/optimize.md §4)."""
    func = "root_scalar"
    _callable(f, func, "f")
    if fprime is not None:
        _callable(fprime, func, "fprime")
    if fprime2 is not None:
        _callable(fprime2, func, "fprime2")
    if method is None:  # scipy's choice
        if bracket is not None:
            method = "brentq"
        elif x0 is not None:
            if fprime is not None:
                method = "halley" if fprime2 is not None else "newton"
            elif x1 is not None:
                method = "secant"
            else:
                method = "newton"
        else:
            raise ValueError(f"{func}(): give 'bracket' or 'x0'")
    else:
        _check_method(method, func, _ROOT_METHODS)

    a = b = xs = 0.0
    x1v = None
    if method in _BRACKETING:
        if bracket is None:
            raise ValueError(f"{func}(): 'bracket' is required when method={method!r}")
        _not_given(func, method, x0=x0, x1=x1, fprime=fprime, fprime2=fprime2)
        a, b = _real_pair(bracket, func, "bracket", "pair (a, b)")
        if method == "toms748" and not a < b:
            raise ValueError(f"{func}(): method='toms748' needs a < b in 'bracket', not ({a!r}, {b!r})")
        xtol = 2e-12 if xtol is None else _check_tol(xtol, func, "xtol")
        rmin = _EPS if method == "toms748" else 4 * _EPS
        rtol = 4 * _EPS if rtol is None else _check_tol(rtol, func, "rtol")
        if rtol < rmin:
            what = "the machine epsilon" if method == "toms748" else "4 times the machine epsilon"
            raise ValueError(f"{func}(): 'rtol' must be >= {rmin:.2g} ({what}) when method={method!r}")
        max_iter = 100 if max_iter is None else _check_int(max_iter, func, "max_iter", 1 if method == "toms748" else 0)
    else:
        if x0 is None:
            raise ValueError(f"{func}(): 'x0' is required when method={method!r}")
        _not_given(func, method, bracket=bracket)
        xs = _finite_real(x0, func, "x0")
        if method == "secant":
            _not_given(func, method, fprime=fprime, fprime2=fprime2)
            if x1 is not None:
                x1v = _finite_real(x1, func, "x1")
                if x1v == xs:
                    raise ValueError(f"{func}(): 'x1' must differ from 'x0'")
        else:
            _not_given(func, method, x1=x1)
            if method == "newton" and fprime2 is not None:
                raise ValueError(f"{func}(): method='newton' takes no 'fprime2' (method='halley' takes it)")
            if method == "halley":
                for name, v in (("fprime", fprime), ("fprime2", fprime2)):
                    if v is None:
                        raise ValueError(f"{func}(): {name!r} is required when method='halley'")
        xtol = 1.48e-8 if xtol is None else _check_tol(xtol, func, "xtol")
        rtol = 0.0 if rtol is None else _check_tol(rtol, func, "rtol")
        max_iter = 50 if max_iter is None else _check_int(max_iter, func, "max_iter", 1)
    opts = _check_options(options, func, method, ("k",) if method == "toms748" else ())
    k = 1
    if "k" in opts:
        k = _check_int(opts["k"], func, "k", 1)
        if k > 100:
            raise ValueError(f"{func}(): 'k' must be <= 100")

    x, status, n_iter, nfev, njev, nhev, fa, fb, reason, aux = _k.root_solve(
        func,
        f,
        fprime,
        fprime2,
        _scalar_check(func),
        _ROOT_CODE[method],
        a,
        b,
        xs,
        x1v,
        xtol,
        rtol,
        max_iter,
        k,
    )
    if status == _R_SIGN:  # O32
        raise ValueError(
            f"{func}(): 'bracket' does not enclose a root: f(a) and f(b) have the same sign "
            f"(f({a!r}) = {fa!r}, f({b!r}) = {fb!r})"
        )
    if status == 0:
        message = "converged"
    elif status == _R_NAN and method == "toms748" and math.isinf(fa):  # O45 (scipy raises ValueError)
        message = f"The function value at x={x} is {fa}; solver cannot continue."
    elif status == _R_NAN:
        message = f"The function value at x={x} is NaN; solver cannot continue."
    elif reason == _DERIV_ZERO:
        message = "Derivative was zero."
    elif reason == _SECANT_TOL:
        message = f"Tolerance of {aux} reached."
    else:
        message = "convergence error"
    fields = dict(x=x, converged=status == 0, status=status, message=message, n_iter=n_iter, nfev=nfev, method=method)
    if method in ("newton", "halley"):
        fields["njev"] = njev
    if method == "halley":
        fields["nhev"] = nhev
    return OptimizeResult._make(**fields)


def minimize_scalar(fun, /, *, bracket=None, bounds=None, method=None, tol=None, options=None):
    """A local minimum of fun(x): from a bracket ('brent', 'golden') or in bounds ('bounded')
    (docs/api/optimize.md §5)."""
    func = "minimize_scalar"
    _callable(fun, func, "fun")
    if method is None:
        method = "brent" if bounds is None else "bounded"
    else:
        _check_method(method, func, _MIN_METHODS)
    br, lo, hi = [], 0.0, 0.0
    if method == "bounded":
        if bracket is not None:
            raise ValueError(f"{func}(): method='bounded' takes no 'bracket' (it searches 'bounds')")
        if bounds is None:
            raise ValueError(f"{func}(): 'bounds' is required when method='bounded'")
        lo, hi = _real_pair(bounds, func, "bounds", "pair (lower, upper)")
        if lo > hi:
            raise ValueError(f"{func}(): the lower bound must not be greater than the upper bound")
        key, xtol, max_iter = "xatol", 1e-5, 500
    else:
        if bounds is not None:
            raise ValueError(f"{func}(): method={method!r} takes no 'bounds' (method='bounded' does)")
        if bracket is not None:
            br = _real_pair(bracket, func, "bracket", "pair (xa, xb) or a triple (xa, xb, xc)", (2, 3))
            if len(br) == 3:
                xa, xb, xc = br
                if xa > xc:
                    xa, xc = xc, xa
                if not (xa < xb < xc):
                    raise ValueError(f"{func}(): 'bracket' must have xa < xb < xc (or xc < xb < xa), not {tuple(br)!r}")
                br = [xa, xb, xc]
        key = "xtol"
        xtol, max_iter = (1.48e-8, 500) if method == "brent" else (math.sqrt(_EPS), 5000)
    opts = _check_options(options, func, method, (key, "max_iter"))
    if tol is not None:
        tol = _check_tol(tol, func, "tol")
        if key in opts:
            raise ValueError(f"{func}(): give 'tol' or options[{key!r}], not both")
        opts[key] = tol
    if key in opts:
        xtol = _check_tol(opts[key], func, key)
    if "max_iter" in opts:
        max_iter = _check_int(opts["max_iter"], func, "max_iter", 0)

    x, f, status, n_iter, nfev, fa, fb, fc = _k.min_solve(
        func, fun, _scalar_check(func), _MIN_CODE[method], br, lo, hi, xtol, max_iter
    )
    if status == _M_BAD_F:
        raise ValueError(
            f"{func}(): 'bracket' must have f(xb) < f(xa) and f(xb) < f(xc), "
            f"not f(xa) = {fa!r}, f(xb) = {fb!r}, f(xc) = {fc!r}"
        )
    if method == "bounded":
        message = _BOUNDED_MESSAGES[status]
    elif status == 0:
        message = (
            "\nOptimization terminated successfully;\n"
            "The returned value satisfies the termination criteria\n"
            f"(using xtol = {xtol} )"
        )
    else:
        message = _BRACKET_MESSAGES[status]
    return OptimizeResult._make(
        x=x,
        fun=f,
        converged=status == 0,
        status=status,
        message=message,
        n_iter=n_iter,
        nfev=nfev,
        method=method,
    )
