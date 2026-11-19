"""Multivariate minimization: minimize (docs/api/optimize.md §7, docs/design/optimize.md §3.5).

The iterations are the native kernels stardust._core.optimize.nelder_mead_solve, bfgs_solve,
lbfgsb_solve, newton_cg_solve and trust_solve, each the same steps as scipy 1.18.1's method
(O9): _minimize_neldermead, _minimize_bfgs with its Wolfe line searches, _minimize_lbfgsb with
scipy's C translation of L-BFGS-B 3.0, _minimize_newtoncg, and _minimize_trust_region with the
Steihaug-Toint (trust-ncg) and Moré-Sorensen (trust-exact) subproblems. The gradient is the
user's jac, fun's second value (jac=True) or scipy's differences, with scipy's ScalarFunction memo
(csrc/optimize/objective.hpp); the Hessian the user's hess or hessp, or scipy's differences of
the gradient (csrc/optimize/hessian.hpp). The user's functions get a new float64 sd.ndarray x;
what they return is checked here when it is not a Python float (fun) or a float64 sd.ndarray of
the shape asked for ((n,) for a gradient or a Hessian product, (n, n) for a Hessian).
"""

import math
import numbers

from .. import _core as _sd
from .. import linalg as _linalg
from ._common import (
    _EPS,
    _as_array,
    _bounds,
    _callable,
    _check_bool,
    _check_int,
    _check_method,
    _check_options,
    _check_tol,
    _real,
    _real_vector,
    _scalar_check,
    _type_name,
    _x0,
)
from ._result import OptimizeResult, _Lazy

_k = _sd.optimize

_METHODS = ("nelder-mead", "bfgs", "l-bfgs-b", "newton-cg", "trust-ncg", "trust-exact")
_NEWTON = ("newton-cg", "trust-ncg", "trust-exact")  # the methods that take hess, hessp (O4)
_TRUST_OPTIONS = ("initial_trust_radius", "max_trust_radius", "eta", "gtol", "max_iter")
_OPTIONS = {
    "nelder-mead": ("max_iter", "max_nfev", "xatol", "fatol", "adaptive", "initial_simplex"),
    "bfgs": ("gtol", "norm", "eps", "max_iter", "finite_diff_rel_step", "xrtol", "c1", "c2", "hess_inv0"),
    "l-bfgs-b": ("maxcor", "ftol", "gtol", "eps", "max_nfev", "max_iter", "max_ls", "finite_diff_rel_step"),
    "newton-cg": ("xtol", "eps", "max_iter", "c1", "c2"),
    "trust-ncg": _TRUST_OPTIONS,
    "trust-exact": _TRUST_OPTIONS + ("subproblem_max_iter",),
}
# tol sets these options (scipy's minimize)
_TOL_KEYS = {
    "nelder-mead": ("xatol", "fatol"),
    "bfgs": ("gtol",),
    "l-bfgs-b": ("ftol", "gtol"),
    "newton-cg": ("xtol",),
    "trust-ncg": ("gtol",),
    "trust-exact": ("gtol",),
}

# the kernels' gradient modes (csrc/optimize/objective.hpp, GradMode)
_G_CALLABLE, _G_2POINT, _G_3POINT, _G_ABSOLUTE, _G_WITH_FUN = 0, 1, 2, 3, 4
# and Hessian modes (csrc/optimize/hessian.hpp, HessMode)
_H_NONE, _H_CALLABLE, _H_2POINT, _H_3POINT = 0, 1, 2, 3

_SQRT_EPS = math.sqrt(_EPS)  # scipy's _epsilon, BFGS's eps

# scipy 1.18.1's _status_message
_SUCCESS = "Optimization terminated successfully."
_MAXFEV = "Maximum number of function evaluations has been exceeded."
_MAXITER = "Maximum number of iterations has been exceeded."
_PR_LOSS = "Desired error not necessarily achieved due to precision loss."
_NAN = "NaN result encountered."
_NM_MESSAGES = {0: _SUCCESS, 1: _MAXFEV, 2: _MAXITER}
_BFGS_MESSAGES = {0: _SUCCESS, 1: _MAXITER, 2: _PR_LOSS, 3: _NAN}
# Newton-CG's (status, message) by the kernel's status: scipy's terminate() calls, 3 twice
_NCG_RESULTS = {
    0: (0, _SUCCESS),
    1: (1, "Warning: " + _MAXITER),
    2: (2, "Warning: " + _PR_LOSS),
    3: (3, "Warning: CG iterations didn't converge. The Hessian is not positive definite."),
    4: (3, _NAN),
}
# _minimize_trust_region's status_messages
_TRUST_MESSAGES = {
    0: _SUCCESS,
    1: _MAXITER,
    2: "A bad approximation caused failure to predict improvement.",
    3: "A linalg error occurred, such as a non-psd Hessian.",
}
# L-BFGS-B's task and its message (scipy's status_messages and task_messages)
_LB_TASKS = {4: "CONVERGENCE", 5: "STOP", 8: "ABNORMAL"}
_LB_MESSAGES = {
    0: "",
    401: "NORM OF PROJECTED GRADIENT <= PGTOL",
    402: "RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    502: "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    504: "TOTAL NO. OF ITERATIONS REACHED LIMIT",
}
_FIXED = "All independent variables were fixed by bounds."


def _checks(func, n):
    """check(value, what) for the kernels: what fun returned ('fun') as a float, a gradient
    ('jac') or a Hessian product ('hessp') as a float64 sd.ndarray (n,), a Hessian ('hess') as a
    float64 sd.ndarray (n, n), fun's pair (f, g) with jac=True ('pair'), or the error."""
    real = _scalar_check(func)

    def check(v, what):
        if what == "fun":
            return real(v, "fun")
        if what == "pair":
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return (v[0], v[1])
            got = f"{len(v)} values" if isinstance(v, (tuple, list)) else _type_name(v)
            raise TypeError(f"{func}(): 'fun' must return a pair (f, g) when jac=True, not {got}")
        if what == "hess":
            return _hessian(v, func, n)
        return _gradient(v, func, n, what)

    return check


def _not_dense(v, func, name):
    """scipy's sparse matrices and LinearOperator are not taken (O63)."""
    if type(v).__module__.startswith("scipy.sparse"):
        raise TypeError(f"{func}(): {name!r} must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}")


def _gradient(v, func, n, name="jac"):
    if isinstance(v, bool) or v is None or isinstance(v, (str, bytes)):
        raise TypeError(f"{func}(): {name!r} must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}")
    _not_dense(v, func, name)
    scalar = isinstance(v, numbers.Real) or (hasattr(v, "ndim") and getattr(v, "ndim", None) == 0)
    if scalar and not isinstance(v, complex) and getattr(getattr(v, "dtype", None), "kind", "f") in "iuf":
        if n == 1:
            return _sd.asarray([float(v)])
        raise ValueError(f"{func}(): {name!r} must return {n} values (one per variable), not a scalar")
    try:
        a = _as_array(v, func, name)
    except TypeError as e:
        raise TypeError(str(e).replace(f"{name!r} must be", f"{name!r} must return")) from None
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{func}(): {name!r} must return {n} values (one per variable), not shape {tuple(a.shape)}")
    return a


def _hessian(v, func, n):
    """A returned Hessian as a float64 sd.ndarray (n, n); a number when n = 1 (O63)."""
    if isinstance(v, bool) or v is None or isinstance(v, (str, bytes)):
        raise TypeError(f"{func}(): 'hess' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}")
    _not_dense(v, func, "hess")
    scalar = isinstance(v, numbers.Real) or (hasattr(v, "ndim") and getattr(v, "ndim", None) == 0)
    if scalar and not isinstance(v, complex) and getattr(getattr(v, "dtype", None), "kind", "f") in "iuf":
        if n == 1:
            return _sd.asarray([[float(v)]])
        raise ValueError(f"{func}(): 'hess' must return shape ({n}, {n}), not a scalar")
    try:
        a = _as_array(v, func, "hess")
    except TypeError as e:
        raise TypeError(str(e).replace("'hess' must be", "'hess' must return")) from None
    except ValueError:
        raise ValueError(f"{func}(): 'hess' must return shape ({n}, {n}), not a scalar") from None
    if tuple(a.shape) != (n, n):
        raise ValueError(f"{func}(): 'hess' must return shape ({n}, {n}), not {tuple(a.shape)}")
    return a


def _cap(opts, key, default, func, lo):
    return default if opts.get(key) is None else _check_int(opts[key], func, key, lo)


def _steps(opts, key, default, func, n):
    v = opts.get(key)
    if v is None:
        return default
    return _real_vector(v, func, key, n, positive=True)


def _jac_mode(jac, func, method):
    if jac is None:
        return _G_ABSOLUTE
    if jac is True:
        return _G_WITH_FUN
    if callable(jac):
        return _G_CALLABLE
    if isinstance(jac, str):
        if jac == "2-point":
            return _G_2POINT
        if jac == "3-point":
            return _G_3POINT
        raise ValueError(f"{func}(): 'jac' must be callable, True, '2-point', '3-point' or None, not {jac!r}")
    if jac is False:
        raise ValueError(f"{func}(): 'jac' must be callable, True, '2-point', '3-point' or None, not False")
    raise TypeError(f"{func}(): 'jac' must be callable, True, '2-point', '3-point' or None, not {_type_name(jac)}")


def _difference_steps(opts, mode, func, n, eps_default):
    """steps for the kernel: eps (jac None) or finite_diff_rel_step (jac '2-point', '3-point');
    each only where it is used (as diff_step in least_squares, O29)."""
    if mode == _G_ABSOLUTE:
        if opts.get("finite_diff_rel_step") is not None:
            raise ValueError(f"{func}(): 'finite_diff_rel_step' is only used when 'jac' is '2-point' or '3-point'")
        return _steps(opts, "eps", [eps_default] * n, func, n)
    if opts.get("eps") is not None:
        raise ValueError(f"{func}(): 'eps' is only used when 'jac' is None")
    if mode in (_G_2POINT, _G_3POINT):
        return _steps(opts, "finite_diff_rel_step", [], func, n)
    if opts.get("finite_diff_rel_step") is not None:
        raise ValueError(f"{func}(): 'finite_diff_rel_step' is only used when 'jac' is '2-point' or '3-point'")
    return []


def _fun_value(fun, x, func):
    return _scalar_check(func)(fun(x), "fun")


def minimize(fun, x0, /, *, method=None, jac=None, hess=None, hessp=None, bounds=None, tol=None, options=None):
    """A local minimum of fun(x) from x0: method='nelder-mead' (the simplex method), 'bfgs',
    'l-bfgs-b' (bounds allowed), or with the gradient and the Hessian (hess or hessp)
    'newton-cg', 'trust-ncg' and 'trust-exact'; None picks 'bfgs', or 'l-bfgs-b' with bounds
    (docs/api/optimize.md §7)."""
    func = "minimize"
    _callable(fun, func, "fun")
    x = _x0(x0, func)
    n = x.shape[0]
    if method is None:
        method = "bfgs" if bounds is None else "l-bfgs-b"  # O50
    else:
        _check_method(method, func, _METHODS)
    if method not in _NEWTON:
        for name, v in (("hess", hess), ("hessp", hessp)):
            if v is not None:
                raise ValueError(f"{func}(): method={method!r} takes no {name!r}")
    if method == "nelder-mead" and jac is not None:
        raise ValueError(f"{func}(): method='nelder-mead' takes no 'jac'")
    if method in ("bfgs",) + _NEWTON and bounds is not None:  # O66
        raise ValueError(f"{func}(): method={method!r} takes no bounds (method='l-bfgs-b' and method='nelder-mead' do)")
    mode = _jac_mode(jac, func, method) if method != "nelder-mead" else None
    if method in _NEWTON:
        _needs_jac(jac, mode, func, method)  # O65
        hmode = _hess_mode(hess, hessp, func, method)  # O63, O64
    bd = None if bounds is None else _bounds(bounds, func, n, allow_equal=True)  # O51
    xl = [float(v) for v in x.tolist()]
    if bd is not None and not all(a <= t <= b for t, a, b in zip(xl, *bd)):
        raise ValueError(f"{func}(): 'x0' must be within 'bounds'")  # O52
    opts = _check_options(options, func, method, _OPTIONS[method])
    if tol is not None:
        tol = _check_tol(tol, func, "tol")
        for key in _TOL_KEYS[method]:
            if opts.get(key) is not None:
                raise ValueError(f"{func}(): give 'tol' or options[{key!r}], not both")
            opts[key] = tol
    if method == "nelder-mead":
        return _nelder_mead(fun, xl, bd, opts, func)
    g = jac if mode == _G_CALLABLE else None
    if method == "bfgs":
        return _bfgs(fun, g, mode, xl, opts, func)
    if method == "l-bfgs-b":
        return _lbfgsb(fun, g, mode, xl, bd, opts, func)
    h = hess if hmode == _H_CALLABLE else None
    if method == "newton-cg":
        return _newton_cg(fun, g, h, hessp, mode, hmode, xl, opts, func)
    return _trust(fun, g, h, hessp, mode, hmode, method, xl, opts, func)


def _nelder_mead(fun, xl, bd, opts, func):
    n = len(xl)
    max_iter = None if opts.get("max_iter") is None else _check_int(opts["max_iter"], func, "max_iter", 0)
    max_nfev = None if opts.get("max_nfev") is None else _check_int(opts["max_nfev"], func, "max_nfev", 1)
    # scipy: 200 n each when neither is given; one given leaves the other uncapped
    if max_iter is None and max_nfev is None:
        max_iter = max_nfev = 200 * n
    elif max_iter is None:
        max_iter = -1
    elif max_nfev is None:
        max_nfev = -1
    xatol = _check_tol(opts.get("xatol", 1e-4), func, "xatol")
    fatol = _check_tol(opts.get("fatol", 1e-4), func, "fatol")
    adaptive = _check_bool(opts.get("adaptive", False), func, "adaptive")
    sim0 = []
    if opts.get("initial_simplex") is not None:
        s = _as_array(opts["initial_simplex"], func, "initial_simplex")
        if s.ndim != 2 or s.shape[0] != s.shape[1] + 1:
            raise ValueError(f"{func}(): 'initial_simplex' must have shape (N + 1, N), not {tuple(s.shape)}")
        if s.shape[1] != n:
            raise ValueError(f"{func}(): 'initial_simplex' must have {n} columns like 'x0', not {s.shape[1]}")
        if not bool(_sd.all(_sd.isfinite(s))):
            raise ValueError(f"{func}(): 'initial_simplex' must be finite")
        sim0 = [float(v) for v in _sd.reshape(s, (-1,)).tolist()]
    lb, ub = bd if bd is not None else ([-math.inf] * n, [math.inf] * n)
    x, f, n_iter, nfev, status, sim, fsim = _k.nelder_mead_solve(
        func, fun, _checks(func, n), xl, sim0, lb, ub, bd is not None, adaptive, xatol, fatol, max_iter, max_nfev
    )
    return OptimizeResult(
        x=x,
        fun=f,
        final_simplex=(sim, fsim),
        converged=status == 0 and math.isfinite(f),
        status=status,
        message=_NM_MESSAGES[status],
        n_iter=n_iter,
        nfev=nfev,
        method="nelder-mead",
    )


def _hess_inv0(v, func, n):
    h = _as_array(v, func, "hess_inv0")
    if h.ndim != 2 or h.shape[0] != n or h.shape[1] != n:
        raise ValueError(f"{func}(): 'hess_inv0' must have shape ({n}, {n}), not {tuple(h.shape)}")
    if not bool(_sd.all(_sd.isfinite(h))):
        raise ValueError(f"{func}(): 'hess_inv0' must be finite")
    rows = h.tolist()
    if any(rows[i][j] != rows[j][i] for i in range(n) for j in range(i)):
        raise ValueError(f"{func}(): 'hess_inv0' must be symmetric positive definite")
    try:
        _linalg.cholesky(h)
    except _linalg.LinAlgError:
        raise ValueError(f"{func}(): 'hess_inv0' must be symmetric positive definite") from None
    return [float(t) for r in rows for t in r]


def _c1_c2(opts, func):
    c1 = _real(opts.get("c1", 1e-4), func, "c1")
    c2 = _real(opts.get("c2", 0.9), func, "c2")
    if not (0 < c1 < c2 < 1):
        raise ValueError(f"{func}(): 'c1' and 'c2' must satisfy 0 < c1 < c2 < 1")
    return c1, c2


def _bfgs(fun, jac, mode, xl, opts, func):
    n = len(xl)
    gtol = _check_tol(opts.get("gtol", 1e-5), func, "gtol")
    norm = _real(opts.get("norm", math.inf), func, "norm")
    if math.isnan(norm) or norm == 0:
        raise ValueError(f"{func}(): 'norm' must be a nonzero number (inf for the largest |g_i|)")
    steps = _difference_steps(opts, mode, func, n, _SQRT_EPS)
    max_iter = _cap(opts, "max_iter", 200 * n, func, 0)
    xrtol = _check_tol(opts.get("xrtol", 0.0), func, "xrtol")
    c1, c2 = _c1_c2(opts, func)
    h0 = [] if opts.get("hess_inv0") is None else _hess_inv0(opts["hess_inv0"], func, n)
    x, f, g, H, status, n_iter, nfev, njev = _k.bfgs_solve(
        func, fun, jac, _checks(func, n), mode, xl, h0, gtol, norm, steps, max_iter, xrtol, c1, c2
    )
    return OptimizeResult(
        x=x,
        fun=f,
        jac=g,
        hess_inv=H,
        converged=status == 0 and math.isfinite(f),
        status=status,
        message=_BFGS_MESSAGES[status],
        n_iter=n_iter,
        nfev=nfev,
        njev=njev,
        method="bfgs",
    )


def _lbfgs_inverse_hessian(S, Y):
    """scipy's LbfgsInvHessProduct(S, Y).todense(): the two-loop recursion applied to the
    identity, the pairs in the order given."""
    nc, n = S.shape
    Q = _sd.eye(n)
    if nc == 0:
        return Q
    s = [S[i] for i in range(nc)]
    y = [Y[i] for i in range(nc)]
    rho = [1.0 / float(_linalg.vecdot(s[i], y[i])) for i in range(nc)]
    alpha = [None] * nc
    for i in range(nc - 1, -1, -1):
        alpha[i] = rho[i] * (s[i] @ Q)
        Q = Q - _sd.reshape(y[i], (n, 1)) * _sd.reshape(alpha[i], (1, n))
    R = Q
    for i in range(nc):
        beta = rho[i] * (y[i] @ R)
        R = R + _sd.reshape(s[i], (n, 1)) * _sd.reshape(alpha[i] - beta, (1, n))
    return R


def _lbfgsb(fun, jac, mode, xl, bd, opts, func):
    n = len(xl)
    m = _cap(opts, "maxcor", 10, func, 1)
    ftol = _check_tol(opts.get("ftol", 2.220446049250313e-09), func, "ftol")
    gtol = _check_tol(opts.get("gtol", 1e-5), func, "gtol")
    steps = _difference_steps(opts, mode, func, n, 1e-8)
    max_nfev = _cap(opts, "max_nfev", 15000, func, 1)
    max_iter = _cap(opts, "max_iter", 15000, func, 0)
    max_ls = _cap(opts, "max_ls", 20, func, 1)
    lb, ub = bd if bd is not None else ([-math.inf] * n, [math.inf] * n)

    fixed = [a == b for a, b in zip(lb, ub)]
    if bd is not None and all(fixed):  # scipy's _optimize_result_for_equal_bounds
        x = _sd.asarray(lb)
        f = _fun_value(fun, x.copy(), func)
        if not math.isfinite(f):
            raise ValueError(f"{func}(): 'fun' is not finite at 'x0'")
        return OptimizeResult(
            x=x, fun=f, converged=True, status=0, message=_FIXED, n_iter=0, nfev=1, njev=0, method="l-bfgs-b"
        )
    removed = any(fixed) and mode != _G_CALLABLE and mode != _G_WITH_FUN  # scipy removes them for differences
    user = fun
    if removed:
        free = [i for i in range(n) if not fixed[i]]
        base = list(lb)

        def user(xr):
            full = list(base)
            for i, v in zip(free, xr.tolist()):
                full[i] = v
            return fun(_sd.asarray(full))

        xl = [xl[i] for i in free]
        lb = [lb[i] for i in free]
        ub = [ub[i] for i in free]
        if len(steps) == n:
            steps = [steps[i] for i in free]
    k = len(xl)
    x, f, g, task, task_msg, n_iter, nfev, njev, S, Y, nc = _k.lbfgsb_solve(
        func, user, jac, _checks(func, k), mode, xl, lb, ub, m, ftol / _EPS, gtol, steps, max_nfev, max_iter, max_ls
    )
    if task == 4:
        status = 0
    elif nfev > max_nfev or n_iter >= max_iter:
        status = 1
    else:
        status = 2
    message = _LB_TASKS[task] + ": " + _LB_MESSAGES[task_msg]
    if removed:
        xs, gs, full_x, full_g = x.tolist(), g.tolist(), list(base), [math.nan] * n
        for i, v, gv in zip(free, xs, gs):
            full_x[i] = v
            full_g[i] = gv
        x, g = _sd.asarray(full_x), _sd.asarray(full_g)
        hess_inv = None  # O58: no hess_inv field (the removed variables have none)
    else:
        hess_inv = _Lazy(lambda: _lbfgs_inverse_hessian(S[:nc], Y[:nc]))  # O55: computed when first read
    fields = {} if hess_inv is None else {"hess_inv": hess_inv}
    return OptimizeResult(
        x=x,
        fun=f,
        jac=g,
        **fields,
        converged=status == 0 and math.isfinite(f),
        status=status,
        message=message,
        n_iter=n_iter,
        nfev=nfev,
        njev=njev,
        method="l-bfgs-b",
    )


# ---- Newton-CG, trust-ncg and trust-exact (O4)


def _needs_jac(jac, mode, func, method):
    """O65: a callable jac or jac=True (scipy turns the difference forms into None and raises)."""
    if jac is None:
        raise ValueError(
            f"{func}(): 'jac' is required when method={method!r} (a callable, or True when 'fun' returns (f, g))"
        )
    if mode not in (_G_CALLABLE, _G_WITH_FUN):
        raise ValueError(f"{func}(): method={method!r} needs 'jac' as a callable or True, not {jac!r}")


def _hess_mode(hess, hessp, func, method):
    """The Hessian's form for the kernels (O63, O64): hess callable, '2-point' or '3-point'
    (products by differences of the gradient), or hessp alone."""
    hmode = _H_NONE
    if hess is not None:
        if callable(hess):
            hmode = _H_CALLABLE
        elif isinstance(hess, str) and hess in ("2-point", "3-point"):
            hmode = _H_2POINT if hess == "2-point" else _H_3POINT
        elif isinstance(hess, str):
            raise ValueError(f"{func}(): 'hess' must be callable, '2-point', '3-point' or None, not {hess!r}")
        else:
            raise TypeError(f"{func}(): 'hess' must be callable, '2-point', '3-point' or None, not {_type_name(hess)}")
    if hessp is not None:
        if not callable(hessp):
            raise TypeError(f"{func}(): 'hessp' must be callable or None, not {_type_name(hessp)}")
        if method == "trust-exact":
            raise ValueError(f"{func}(): method='trust-exact' takes no 'hessp' (it needs the matrix 'hess')")
        if hmode in (_H_2POINT, _H_3POINT):
            raise ValueError(f"{func}(): give 'hessp' or hess={hess!r}, not both")
        if method == "newton-cg" and hmode == _H_CALLABLE:  # O74: hessp would be unused (O62)
            raise ValueError(f"{func}(): method='newton-cg' takes 'hess' or 'hessp', not both")
    if method == "trust-exact" and hmode != _H_CALLABLE:
        if hess is None:
            raise ValueError(f"{func}(): 'hess' is required when method='trust-exact' (a callable giving the matrix)")
        raise ValueError(f"{func}(): method='trust-exact' needs 'hess' as a callable (the matrix), not {hess!r}")
    if method == "trust-ncg" and hess is None and hessp is None:
        raise ValueError(f"{func}(): method='trust-ncg' needs 'hess' or 'hessp'")
    return hmode


def _not_finite(status, xs, func):
    what = {-1: "'jac' is", -2: "'hess' is", -3: "the trust-region subproblem is"}[status]
    raise ValueError(f"{func}(): {what} not finite at x = {xs}")


def _newton_cg(fun, jac, hess, hessp, mode, hmode, xl, opts, func):
    n = len(xl)
    xtol = _check_tol(opts.get("xtol", 1e-5), func, "xtol")
    if opts.get("eps") is not None and (hmode != _H_NONE or hessp is not None):
        raise ValueError(f"{func}(): 'eps' is only used when method='newton-cg' has no 'hess' and no 'hessp'")
    eps = _steps(opts, "eps", [_SQRT_EPS] * n, func, n)
    max_iter = _cap(opts, "max_iter", 200 * n, func, 0)
    c1, c2 = _c1_c2(opts, func)
    x, f, g, kstatus, n_iter, nfev, njev, nhev = _k.newton_cg_solve(
        func, fun, jac, hess, hessp, _checks(func, n), mode, hmode, xl, xtol, eps, max_iter, c1, c2
    )
    status, message = _NCG_RESULTS[kstatus]
    return OptimizeResult(
        x=x,
        fun=f,
        jac=g,
        converged=status == 0 and math.isfinite(f),
        status=status,
        message=message,
        n_iter=n_iter,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
        method="newton-cg",
    )


def _trust(fun, jac, hess, hessp, mode, hmode, method, xl, opts, func):
    n = len(xl)
    r0 = _real(opts.get("initial_trust_radius", 1.0), func, "initial_trust_radius")
    rmax = _real(opts.get("max_trust_radius", 1000.0), func, "max_trust_radius")
    eta = _real(opts.get("eta", 0.15), func, "eta")
    if not (0 <= eta < 0.25):
        raise ValueError(f"{func}(): 'eta' must be >= 0 and < 0.25")
    if not rmax > 0:
        raise ValueError(f"{func}(): 'max_trust_radius' must be > 0")
    if not (math.isfinite(r0) and r0 > 0):
        raise ValueError(f"{func}(): 'initial_trust_radius' must be finite and > 0")
    if not r0 < rmax:
        raise ValueError(f"{func}(): 'initial_trust_radius' must be less than 'max_trust_radius'")
    gtol = _check_tol(opts.get("gtol", 1e-4), func, "gtol")
    max_iter = _cap(opts, "max_iter", 200 * n, func, 0)
    sub = _cap(opts, "subproblem_max_iter", 25, func, 1)
    x, f, g, H, status, n_iter, nfev, njev, nhev = _k.trust_solve(
        func,
        fun,
        jac,
        hess,
        hessp,
        _checks(func, n),
        mode,
        hmode,
        method == "trust-exact",
        xl,
        r0,
        rmax,
        eta,
        gtol,
        max_iter,
        sub,
    )
    if status < 0:
        _not_finite(status, x.tolist(), func)
    fields = {} if H is None else {"hess": H}
    return OptimizeResult(
        x=x,
        fun=f,
        jac=g,
        **fields,
        converged=status == 0 and math.isfinite(f),
        status=status,
        message=_TRUST_MESSAGES[status],
        n_iter=n_iter,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
        method=method,
    )
