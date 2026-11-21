"""derivative, jacobian and hessian (docs/api/differentiate.md §2, §3; docs/design/integrate.md
§3.8 and decisions C26-C38; scipy 1.18.1's scipy.differentiate).

The method and the evaluation points are scipy's (C5): a difference formula of order 2n repeated
with the step divided by step_factor until two estimates agree. The weights are the exact ones
correctly rounded (C28, ._weights); the central sum adds the differences of the pairs of nodes
(C38). The iterations are the native kernels
stardust._core.differentiate.derivative_solve, jacobian_solve and hessian_solve
(csrc/differentiate/derivative.cpp).

derivative is of the elementwise family (C4): the inputs and args broadcast together, the first
call is f(x) on all elements as (k,), then f gets (k, p) arrays of the elements in progress
(p = 2n points in the first iteration, 2 after it; C26) with args as (k, 1). jacobian and hessian
take a vectorized f: (m, N) in, (k, N) or (N,) out (C27).
"""

import math
import numbers

from .. import _core as _sd
from ..optimize._common import (
    _as_array,
    _callable,
    _check_int,
    _check_tol,
    _real,
    _real_vector,
    _type_name,
)
from ..optimize._elementwise import _args, _inputs, _shaped
from ._result import DifferentiateResult
from ._weights import central_weights, one_sided_weights

_k = _sd.differentiate

_TINY = 2.2250738585072014e-308  # the smallest normal double: scipy's default atol
_EPS = 2.0**-52
_RTOL = _EPS**0.5  # scipy's default rtol, sqrt(eps)
_RTOL_MIN = 100 * _EPS  # hessian: the smallest rtol other than 0 (C29)
# the highest order (C118): every even order up to it keeps the contract (atol + rtol |df|) at
# the eleven elementary functions' nine points against Arb, order 20 does not (log at 1e4,
# 1.02 of the bound) and higher ones less and less (sqrt at 1e4 64 times from order 32; from
# order 60 on wrong values converge, as exp'(1) = 0 at 160): the inner nodes h / c^(n-1) of the
# first iteration fall towards rounding, and the exact weights cost O(n^2) big rationals
_ORDER_MAX = 18


def _real_array(v, func, what):
    """A value f returned as an sd.ndarray in its own dtype: the errors of the elementwise
    family's check (stardust.optimize._elementwise._values)."""
    if isinstance(v, _sd.ndarray):
        a = v
    else:
        if isinstance(v, (str, bytes, bool)) or v is None:
            raise TypeError(
                f"{func}(): '{what}' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
            )
        if isinstance(v, numbers.Number) or getattr(v, "ndim", None) == 0:
            return None  # a scalar: the caller names the shape it wanted
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
    if a.dtype == _sd.bool:  # a comparison returned where numbers were meant
        raise TypeError(f"{func}(): '{what}' must return real values, not a bool array")
    return a


def _shape_text(shape):
    return "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"


def _values(func):
    """check(value, shape) for derivative's kernel: what f returned, as a float64 sd.ndarray of
    its argument's shape, (k,) (an int) or (k, p) (a tuple), or the error (optimize's form)."""

    def check(v, shape):
        want = (shape,) if isinstance(shape, int) else tuple(shape)
        a = _real_array(v, func, "f")
        if a is None:
            raise ValueError(f"{func}(): 'f' must return shape {_shape_text(want)} like its argument, not a scalar")
        if tuple(a.shape) != want:
            raise ValueError(
                f"{func}(): 'f' must return shape {_shape_text(want)} like its argument, not {_shape_text(a.shape)}"
            )
        return a if a.dtype == _sd.float64 else a.astype(_sd.float64)

    return check


def _vectorized_values(func):
    """check(value, (m, N), expect) for jacobian's and hessian's kernels: f's value at an
    argument of shape (m, N) as a float64 sd.ndarray (N,) or (k, N); expect is -1 (either: the
    first call of jacobian), -2 (N,) only (hessian), else k."""

    def check(v, xshape, expect):
        m, n = xshape
        a = _real_array(v, func, "f")
        got = "a scalar" if a is None else _shape_text(a.shape)
        ok = a is not None and (
            (a.ndim == 1 and a.shape[0] == n and expect in (-1, -2))
            or (a.ndim == 2 and a.shape[1] == n and (expect == -1 or a.shape[0] == expect))
        )
        if not ok:
            if expect == -1:
                want = f"({n},) or (k, {n})"
            elif expect == -2:
                want = f"({n},)"
            else:
                want = f"({expect}, {n})"
            raise ValueError(f"{func}(): 'f' must return shape {want} for its argument of shape ({m}, {n}), not {got}")
        return a if a.dtype == _sd.float64 else a.astype(_sd.float64)

    return check


def _tolerances(atol, rtol, func):
    atol = _TINY if atol is None else _check_tol(atol, func, "atol")
    rtol = _RTOL if rtol is None else _check_tol(rtol, func, "rtol")
    return atol, rtol


def _step_factor(v, func):
    c = _real(v, func, "step_factor")
    if not (math.isfinite(c) and c > 0 and c != 1):
        raise ValueError(f"{func}(): 'step_factor' must be a finite number > 0 other than 1")  # C31
    return c


def _stencil(order, step_factor, func):
    """(n, c, central weights): n = (order + 1) // 2 terms and the step factor, checked."""
    order = _check_int(order, func, "order", 1)
    if order > _ORDER_MAX:
        raise ValueError(f"{func}(): 'order' must be at most {_ORDER_MAX}")
    c = _step_factor(step_factor, func)
    n = (order + 1) // 2
    return n, c, _sd.asarray(central_weights(n, c), dtype=_sd.float64)


def _one_sided(n, c, d):
    """The one-sided weights when an element of the step directions d takes a side (d != 0 and
    not NaN), else an empty array: the kernels read them only for such elements."""
    if bool(_sd.any(_sd.abs(d) > 0)):
        return _sd.asarray(one_sided_weights(n, c), dtype=_sd.float64)
    return _sd.zeros(0)


def _fields(shape, df, err, status, nit, nfev):
    return dict(
        df=_shaped(df, shape),
        error=_shaped(err, shape),
        converged=_shaped(status == 0, shape),
        status=_shaped(status, shape),
        n_iter=_shaped(nit, shape),
        nfev=_shaped(nfev, shape),
    )


def derivative(
    f, x, /, *, args=(), atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0
):
    """The derivative of the elementwise f at each x by repeated high-order differences
    (scipy.differentiate.derivative; docs/api/differentiate.md §2)."""
    func = "derivative"
    _callable(f, func, "f")
    atol, rtol = _tolerances(atol, rtol, func)
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    n, c, wc = _stencil(order, step_factor, func)
    named = [("x", x), ("initial_step", initial_step), ("step_direction", step_direction)]
    (xs, h0, d), extra, shape = _inputs(func, named, _args(args, func))
    wr = _one_sided(n, c, d)
    df, err, status, nit, nfev = _k.derivative_solve(
        func, f, _values(func), extra, xs, h0, d, wc, wr, n, c, atol, rtol, max_iter
    )
    return DifferentiateResult._make(batch=shape, **_fields(shape, df, err, status, nit, nfev), x=_shaped(xs, shape))


def _point(x, func):
    """jacobian's and hessian's x: a 1-dimensional, non-empty float64 sd.ndarray (C27)."""
    a = _as_array(x, func, "x")
    if a.ndim != 1:
        raise ValueError(f"{func}(): 'x' must be 1-dimensional, not shape {tuple(a.shape)}")
    if a.shape[0] == 0:
        raise ValueError(f"{func}(): 'x' is empty (no variables)")
    return a


def _per_variable(v, func, arg, m):
    return _sd.asarray(_real_vector(v, func, arg, m, like="'x'"), dtype=_sd.float64)


def jacobian(
    f, x, /, *, atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0
):
    """The Jacobian (k, m) of the vectorized f: R^m -> R^k at x (the gradient (m,) when f is
    scalar), each entry by derivative's method (scipy.differentiate.jacobian;
    docs/api/differentiate.md §3)."""
    func = "jacobian"
    _callable(f, func, "f")
    atol, rtol = _tolerances(atol, rtol, func)
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    n, c, wc = _stencil(order, step_factor, func)
    x0 = _point(x, func)
    m = x0.shape[0]
    h0 = _per_variable(initial_step, func, "initial_step", m)
    d = _per_variable(step_direction, func, "step_direction", m)
    wr = _one_sided(n, c, d)
    df, err, status, nit, nfev, k, scalar = _k.jacobian_solve(
        func, f, _vectorized_values(func), x0, h0, d, wc, wr, n, c, atol, rtol, max_iter
    )
    shape = (m,) if scalar else (k, m)
    return DifferentiateResult._make(batch=shape, **_fields(shape, df, err, status, nit, nfev))


def hessian(f, x, /, *, atol=None, rtol=None, max_iter=10, order=8, initial_step=0.5, step_factor=2.0):
    """The Hessian (m, m) of the vectorized scalar f at x: the jacobian of the gradient that an
    inner jacobian computes with rtol / 100 (scipy.differentiate.hessian;
    docs/api/differentiate.md §3)."""
    func = "hessian"
    _callable(f, func, "f")
    atol, rtol = _tolerances(atol, rtol, func)
    if 0 < rtol < _RTOL_MIN:  # scipy warns and raises it to 100 eps (C29)
        raise ValueError(
            f"{func}(): 'rtol' must be 0 or >= {_RTOL_MIN!r} (100 times the machine epsilon: the inner "
            f"jacobian's is rtol / 100), not {rtol!r}"
        )
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    n, c, wc = _stencil(order, step_factor, func)
    x0 = _point(x, func)
    m = x0.shape[0]
    h0 = _per_variable(initial_step, func, "initial_step", m)
    wr = _sd.zeros(0)  # always central
    ddf, err, status, _nit, nfev, _, _ = _k.hessian_solve(
        func, f, _vectorized_values(func), x0, h0, wc, wr, n, c, atol, rtol, max_iter
    )
    shape = (m, m)
    fields = _fields(shape, ddf, err, status, _nit, nfev)
    del fields["n_iter"], fields["df"]  # scipy's hessian has neither (docs/api/differentiate.md §3)
    return DifferentiateResult._make(batch=shape, ddf=_shaped(ddf, shape), **fields)
