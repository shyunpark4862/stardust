"""Nonlinear least squares: least_squares (docs/api/optimize.md §3, docs/design/optimize.md §3.4).

The iterations are the native kernel stardust._core.optimize.lsq_solve, which sd.regression.nls
runs too: 'trf' is scipy 1.18.1's trust region reflective method (the 'exact' subproblem,
x_scale 1, no loss), 'lm' MINPACK's lmder; difference Jacobians are scipy's approx_derivative
'2-point' and '3-point' (with diff_step and the bounds). The user's fun and jac get a new
float64 sd.ndarray x; what they return is taken as it is when it is a float64 sd.ndarray of
the right shape, else checked and converted here (_conform).
"""

import math

from .. import _core as _sd
from .. import linalg as _linalg
from ._common import _EPS, _bounds, _callable, _check_int, _check_method, _check_tol, _real_vector, _type_name, _x0
from ._result import OptimizeResult

_k = _sd.optimize

_METHODS = ("trf", "lm")
_JACS = ("2-point", "3-point")
_DIFF = {"2-point": 1, "3-point": 2}

# the kernel's stopping codes (csrc/optimize/lsq.cpp, enum Status): scipy's status, and two of
# its own that are errors here
_ST_JAC, _ST_START = -1, -2
# scipy 1.18.1's least_squares TERMINATION_MESSAGES
_MESSAGES = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}


def _conform(func, n):
    """check(value, what) for the kernel: what fun ('fun') or jac ('jac') returned as a float64
    sd.ndarray of shape (m,) or (m, n), or the error."""
    state = {}

    def check(v, what):
        a = _returned(v, func, what)
        if what == "fun":
            if a.ndim != 1:
                raise ValueError(f"{func}(): 'fun' must return a 1-dimensional array, not shape {tuple(a.shape)}")
            m = state.get("m")
            if m is not None and a.shape[0] != m:
                raise ValueError(f"{func}(): 'fun' must return {m} values as at 'x0', not {a.shape[0]}")
            return a
        m = state["m"]
        if n == 1 and a.ndim == 1 and a.shape[0] == m:
            a = _sd.reshape(a, (m, 1))
        if a.ndim != 2 or a.shape[0] != m or a.shape[1] != n:
            raise ValueError(f"{func}(): 'jac' must return shape ({m}, {n}), not {tuple(a.shape)}")
        return a

    return check, state


def _returned(v, func, what):
    if isinstance(v, _sd.ndarray):
        a = v
    else:
        if isinstance(v, (str, bytes)) or v is None or isinstance(v, bool):
            raise TypeError(
                f"{func}(): '{what}' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
            )
        if (isinstance(v, (int, float)) and not isinstance(v, bool)) or (hasattr(v, "ndim") and v.ndim == 0):
            raise ValueError(
                f"{func}(): '{what}' must return a 1-dimensional array, not a scalar"
                if what == "fun"
                else f"{func}(): 'jac' must return a 2-dimensional array, not a scalar"
            )
        try:
            a = _sd.asarray(v)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
        except TypeError, ValueError:
            raise TypeError(
                f"{func}(): '{what}' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
            ) from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{what}' must return real values, not a complex128 array")
    if a.dtype == _sd.string:
        raise TypeError(f"{func}(): '{what}' must return real values, not a string array")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    return a


def _strictly_feasible(x, lb, ub, rstep=1e-10):
    """scipy's make_strictly_feasible (least_squares moves trf's x0 so)."""
    out = list(x)
    for i, (v, a, b) in enumerate(zip(x, lb, ub)):
        lo = math.isfinite(a) and v - a <= min(b - v, rstep * max(1.0, abs(a)))
        hi = math.isfinite(b) and b - v <= min(v - a, rstep * max(1.0, abs(b)))
        if hi:
            out[i] = b - rstep * max(1.0, abs(b))
        elif lo:
            out[i] = a + rstep * max(1.0, abs(a))
        if out[i] < a or out[i] > b:
            out[i] = 0.5 * (a + b)
    return out


def _active_mask(x, lb, ub, rtol):
    """scipy's find_active_constraints: -1 at a lower bound, 1 at an upper bound (within
    rtol max(1, |bound|) and nearer to it than to the other), else 0."""
    out = []
    for v, a, b in zip(x, lb, ub):
        if rtol == 0:
            out.append(1 if v >= b else (-1 if v <= a else 0))
            continue
        lo_d, up_d = v - a, b - v
        k = 0
        if math.isfinite(a) and lo_d <= min(up_d, rtol * max(1.0, abs(a))):
            k = -1
        if math.isfinite(b) and up_d <= min(lo_d, rtol * max(1.0, abs(b))):
            k = 1
        out.append(k)
    return out


def least_squares(
    fun,
    x0,
    /,
    *,
    jac="2-point",
    bounds=None,
    method="trf",
    ftol=1e-08,
    xtol=1e-08,
    gtol=1e-08,
    diff_step=None,
    max_nfev=None,
):
    """Minimizes 1/2 sum f_i(x)^2 from x0: method='trf' (scipy's trust region reflective
    method, bounds allowed) or 'lm' (MINPACK's Levenberg-Marquardt) (docs/api/optimize.md §3)."""
    func = "least_squares"
    _callable(fun, func, "fun")
    if callable(jac):
        diff = 0
    elif isinstance(jac, str):
        if jac not in _JACS:
            raise ValueError(f"{func}(): 'jac' must be '2-point', '3-point' or callable, not {jac!r}")
        diff = _DIFF[jac]
    else:
        raise TypeError(f"{func}(): 'jac' must be '2-point', '3-point' or callable, not {_type_name(jac)}")
    x = _x0(x0, func)
    n = x.shape[0]
    _check_method(method, func, _METHODS)
    lb, ub = _bounds(bounds, func, n)
    bounded = any(math.isfinite(a) or math.isfinite(b) for a, b in zip(lb, ub))
    if method == "lm" and bounded:
        raise ValueError(f"{func}(): method='lm' takes no bounds (method='trf' does)")
    xl = [float(v) for v in x.tolist()]
    if not all(a <= t <= b for t, a, b in zip(xl, lb, ub)):
        raise ValueError(f"{func}(): 'x0' must be within 'bounds'")
    ftol = _check_tol(ftol, func, "ftol")
    xtol = _check_tol(xtol, func, "xtol")
    gtol = _check_tol(gtol, func, "gtol")
    if ftol < _EPS and xtol < _EPS and gtol < _EPS:
        raise ValueError(
            f"{func}(): at least one of 'ftol', 'xtol' and 'gtol' must be >= 2.2e-16 (the machine epsilon)"
        )
    steps = []
    if diff_step is not None:
        if diff == 0:
            raise ValueError(f"{func}(): 'diff_step' is only used when 'jac' is '2-point' or '3-point'")
        steps = _real_vector(diff_step, func, "diff_step", n, positive=True)
    if max_nfev is None:
        # O25: 100 n iterations of one trial each, a difference Jacobian costing k n calls
        max_nfev = 100 * n if diff == 0 else 100 * n * (diff * n + 1)
    else:
        max_nfev = _check_int(max_nfev, func, "max_nfev", 1)

    if method == "trf":
        xl = _strictly_feasible(xl, lb, ub)
    check, state = _conform(func, n)
    f0 = check(fun(_sd.asarray(xl)), "fun")
    m = f0.shape[0]
    if m == 0:
        raise ValueError(f"{func}(): 'fun' must return at least 1 value")
    state["m"] = m
    if not bool(_sd.all(_sd.isfinite(f0))):
        raise ValueError(f"{func}(): 'fun' is not finite at 'x0'")
    if method == "lm" and m < n:
        raise ValueError(f"{func}(): method='lm' needs at least as many residuals as variables, not {m} < {n}")
    xs, status, nfev, njev, n_iter, f, J = _k.lsq_solve(
        func,
        fun,
        jac if diff == 0 else None,
        check,
        xl,
        lb,
        ub,
        1 if method == "trf" else 2,  # 'lm' in scipy's build of lmder (nls runs 0)
        ftol,
        xtol,
        gtol,
        max_nfev,
        m,
        diff,
        steps,
        True,
        f0,
        True,
    )
    if status == _ST_JAC:
        what = "'jac' is" if diff == 0 else "the differences of 'fun' are"
        raise ValueError(f"{func}(): {what} not finite at x = {xs}")
    x = _sd.asarray(xs)
    cost = 0.5 * float(_linalg.vecdot(f, f))
    grad = _sd.matmul(f, J)
    if method == "trf":
        g = grad.tolist()
        v = []
        for t, gi, a, b in zip(xs, g, lb, ub):  # scipy's CL_scaling_vector
            vi = 1.0
            if gi < 0 and math.isfinite(b):
                vi = b - t
            if gi > 0 and math.isfinite(a):
                vi = t - a
            v.append(vi)
        optimality = max(abs(gi * vi) for gi, vi in zip(g, v))
        active = _active_mask(xs, lb, ub, xtol)
    else:
        optimality = float(_sd.max(_sd.abs(grad)))
        active = [0] * n
    # converged: the stopping test passed at a finite cost (O20, regression G89)
    converged = status > 0 and math.isfinite(cost)
    return OptimizeResult(
        x=x,
        cost=cost,
        fun=f,
        jac=J,
        grad=grad,
        optimality=optimality,
        active_mask=_sd.asarray(active, dtype=_sd.int64),
        converged=converged,
        status=status,
        message=_MESSAGES[status],
        n_iter=n_iter,
        nfev=nfev,
        njev=njev,
        method=method,
    )
