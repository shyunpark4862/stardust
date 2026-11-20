"""Systems of equations: root (docs/api/optimize.md §8, docs/design/optimize.md §3.6).

method='hybr' is MINPACK's hybrj (a Jacobian) or hybrd (forward differences), Powell's hybrid
method, in the native kernel stardust._core.optimize.hybr_solve; 'lm' is least_squares' lmder
kernel (lm_solve) with m = n, lmdif's differences without a Jacobian. Both take scipy 1.18.1's
steps, options, status and message (O9, O17, O80). The user's fun and jac get a new float64
sd.ndarray x; what they return is taken as it is when it is a float64 sd.ndarray of the right
shape, else checked and converted here.
"""

import math
import numbers

from .. import _core as _sd
from ._common import (
    _callable,
    _check_int,
    _check_method,
    _check_options,
    _check_tol,
    _is_number,
    _not_a_pair,
    _real,
    _real_vector,
    _show_x,
    _type_name,
    _values_text,
    _x0,
)
from ._lsq import _returned
from ._result import OptimizeResult

_k = _sd.optimize

_METHODS = ("hybr", "lm")
_XTOL = 1.49012e-08  # scipy's default xtol (and ftol for 'lm', O80)
_OPTIONS = {
    "hybr": ("xtol", "max_nfev", "band", "eps", "factor", "diag"),
    "lm": ("xtol", "ftol", "gtol", "max_nfev", "eps", "factor", "diag"),
}
# hybr_solve's jac modes
_J_CALLABLE, _J_WITH_FUN, _J_DIFF = 0, 1, 2


def _hybr_message(status, xtol, max_nfev):
    """scipy 1.18.1's _root_hybr messages (O17)."""
    return {
        1: "The solution converged.",
        2: f"The number of calls to function has reached maxfev = {max_nfev}.",
        3: f"xtol={xtol:f} is too small, no further improvement in the approximate\n solution is possible.",
        4: "The iteration is not making good progress, as measured by the \n improvement from the last five "
        "Jacobian evaluations.",
        5: "The iteration is not making good progress, as measured by the \n improvement from the last ten iterations.",
    }[status]


def _lm_message(status, ftol, xtol, gtol, max_nfev):
    """scipy 1.18.1's leastsq messages (O17)."""
    return {
        1: f"Both actual and predicted relative reductions in the sum of squares\n  are at most {ftol:f}",
        2: f"The relative error between two consecutive iterates is at most {xtol:f}",
        3: f"Both actual and predicted relative reductions in the sum of squares\n  are at most {ftol:f} and the "
        f"relative error between two consecutive iterates is at \n  most {xtol:f}",
        4: f"The cosine of the angle between func(x) and any column of the\n  Jacobian is at most {gtol:f} in "
        "absolute value",
        5: f"Number of calls to function has reached maxfev = {max_nfev}.",
        6: f"ftol={ftol:f} is too small, no further reduction in the sum of squares\n  is possible.",
        7: f"xtol={xtol:f} is too small, no further improvement in the approximate\n  solution is possible.",
        8: f"gtol={gtol:f} is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision.",
    }[status]


def _checks(func, n):
    """check(value, what) for the kernels: F ('fun') as a float64 sd.ndarray (n,), a Jacobian
    ('jac') as one (n, n), fun's pair (F, J) with jac=True ('pair'), or the error."""

    def values(v):
        a = _returned(v, func, "fun")
        if a.ndim != 1:
            raise ValueError(f"{func}(): 'fun' must return a 1-dimensional array, not shape {tuple(a.shape)}")
        if a.shape[0] != n:
            raise ValueError(f"{func}(): 'fun' must return {_values_text(n)} (one per variable), not {a.shape[0]}")
        return a

    def jacobian(v):
        if n == 1 and isinstance(v, numbers.Real) and not isinstance(v, bool):
            v = [[float(v)]]  # a number for the (1, 1) Jacobian, as minimize's (1, 1) Hessian
        elif n == 1 and getattr(v, "ndim", None) == 0 and getattr(getattr(v, "dtype", None), "kind", "") in "iuf":
            v = [[float(v)]]
        a = _returned(v, func, "jac")
        if n == 1 and a.ndim == 1 and a.shape[0] == 1:
            a = _sd.reshape(a, (1, 1))
        if a.ndim != 2 or a.shape[0] != n or a.shape[1] != n:
            raise ValueError(f"{func}(): 'jac' must return shape ({n}, {n}), not {tuple(a.shape)}")
        return a

    def check(v, what):
        if what == "fun":
            return values(v)
        if what == "jac":
            return jacobian(v)
        # a pair (F, J); with n >= 2 its J is not a number, so that F alone as a list of two
        # numbers is not read as a pair
        if isinstance(v, (tuple, list)) and len(v) == 2 and (n == 1 or not _is_number(v[1])):
            return (values(v[0]), jacobian(v[1]))
        raise TypeError(f"{func}(): 'fun' must return a pair (F, J) when jac=True, not {_not_a_pair(v)}")

    return check


def _memo_pair(fun, check, x0, pair):
    """lmder's fun and jac from fun returning (F, J), memoized by x as scipy's MemoizeJac (fun is
    called again only at another point than its last call)."""
    state = {"x": list(x0), "F": pair[0], "J": pair[1]}

    def at(x):
        xl = x.tolist()
        if xl != state["x"]:
            state["F"], state["J"] = check(fun(x), "pair")
            state["x"] = xl
        return state

    return (lambda x: at(x)["F"]), (lambda x: at(x)["J"])


def _factor(opts, func):
    v = opts.get("factor")
    if v is None:
        return 100.0
    x = _real(v, func, "factor")
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"{func}(): 'factor' must be finite and > 0")
    return x


def _band(v, func, n):
    if isinstance(v, (str, bytes)) or not hasattr(v, "__len__") or not hasattr(v, "__getitem__"):
        raise TypeError(f"{func}(): 'band' must be a pair (ml, mu), not {_type_name(v)}")
    if len(v) != 2:
        raise ValueError(f"{func}(): 'band' must be a pair (ml, mu), not {_values_text(len(v))}")
    ml, mu = (_check_int(t, func, "band", 0) for t in (v[0], v[1]))
    return min(ml, n - 1), min(mu, n - 1)


def root(fun, x0, /, *, method="hybr", jac=None, tol=None, options=None):
    """A root of the system fun(x) = 0 (n equations in n variables) from x0: method='hybr'
    (MINPACK's hybrid Powell method) or 'lm' (MINPACK's Levenberg-Marquardt)
    (docs/api/optimize.md §8)."""
    func = "root"
    _callable(fun, func, "fun")
    x = _x0(x0, func)
    n = x.shape[0]
    _check_method(method, func, _METHODS)
    if jac is None:
        mode = _J_DIFF
    elif jac is True:
        mode = _J_WITH_FUN
    elif callable(jac):
        mode = _J_CALLABLE
    elif jac is False:
        raise ValueError(f"{func}(): 'jac' must be callable, True or None, not False")
    else:
        raise TypeError(f"{func}(): 'jac' must be callable, True or None, not {_type_name(jac)}")
    # root's only cap is max_nfev (scipy's 'lm' maxiter caps the calls too)
    opts = _check_options(
        options, func, method, _OPTIONS[method], {"maxiter": " (the cap on function evaluations is 'max_nfev')"}
    )
    if tol is not None:
        tol = _check_tol(tol, func, "tol")
        if opts.get("xtol") is not None:
            raise ValueError(f"{func}(): give 'tol' or options['xtol'], not both")
        opts["xtol"] = tol
    xtol = _XTOL if opts.get("xtol") is None else _check_tol(opts["xtol"], func, "xtol")
    if mode != _J_DIFF:
        for key in ("eps", "band"):
            if opts.get(key) is not None:
                raise ValueError(f"{func}(): {key!r} is only used when 'jac' is None")
    eps = 0.0 if opts.get("eps") is None else _check_tol(opts["eps"], func, "eps")
    factor = _factor(opts, func)
    diag = [] if opts.get("diag") is None else _real_vector(opts["diag"], func, "diag", n, positive=True)
    if opts.get("max_nfev") is None:
        max_nfev = 100 * (n + 1) if mode != _J_DIFF else 200 * (n + 1)  # scipy's maxfev = 0
    else:
        max_nfev = _check_int(opts["max_nfev"], func, "max_nfev", 1)

    if method == "lm":  # every option is checked before the user's function is called
        ftol = _XTOL if opts.get("ftol") is None else _check_tol(opts["ftol"], func, "ftol")
        gtol = 0.0 if opts.get("gtol") is None else _check_tol(opts["gtol"], func, "gtol")
    check = _checks(func, n)
    xl = [float(v) for v in x.tolist()]
    f0 = check(fun(_sd.asarray(xl)), "fun" if mode != _J_WITH_FUN else "pair")
    if not bool(_sd.all(_sd.isfinite(f0[0] if mode == _J_WITH_FUN else f0))):
        raise ValueError(f"{func}(): 'fun' is not finite at 'x0'")

    if method == "hybr":
        ml, mu = (n - 1, n - 1) if opts.get("band") is None else _band(opts["band"], func, n)
        xs, f, status, nfev, njev, n_iter = _k.hybr_solve(
            func,
            fun,
            jac if mode == _J_CALLABLE else None,
            check,
            xl,
            f0,
            mode,
            xtol,
            max_nfev,
            ml,
            mu,
            eps,
            factor,
            diag,
        )
        if status < 0:
            what = "the differences of 'fun' are" if mode == _J_DIFF else "'jac' is"
            raise ValueError(f"{func}(): {what} not finite at x = {_show_x(xs)}")
        message = _hybr_message(status, xtol, max_nfev)
        converged = status == 1
    else:
        f_call, j_call = fun, (jac if mode == _J_CALLABLE else None)
        if mode == _J_WITH_FUN:
            f_call, j_call = _memo_pair(fun, check, xl, f0)
            f0 = f0[0]
        xs, f, status, nfev, njev, n_iter = _k.lm_solve(
            func, f_call, j_call, check, xl, f0, ftol, xtol, gtol, max_nfev, eps, factor, diag
        )
        if status < 0:
            what = "the differences of 'fun' are" if mode == _J_DIFF else "'jac' is"
            raise ValueError(f"{func}(): {what} not finite at x = {_show_x(xs)}")
        message = _lm_message(status, ftol, xtol, gtol, max_nfev)
        converged = 1 <= status <= 4
    return OptimizeResult._make(
        x=_sd.asarray(xs),
        fun=f,
        converged=converged and bool(_sd.all(_sd.isfinite(f))),
        status=status,
        message=message,
        n_iter=n_iter,
        nfev=nfev,
        njev=njev,
        method=method,
    )
