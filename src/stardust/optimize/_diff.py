"""Numerical differentiation: approx_fprime, check_grad (docs/api/optimize.md §2,
docs/design/optimize.md §3.1). Their default is the accurate 5-point difference Jacobian they
share with sd.regression.nls (regression G70, G88), which lives in the differentiate area
(differentiate/_differences.py; docs/design/integrate.md §6.1, decisions C3, C15); with epsilon,
scipy's forward differences. The loop around the differences (the checks of f's values, the
errors) stays here.
"""

import math
import numbers

from .. import _core as _sd
from .. import linalg as _linalg
from ..differentiate._differences import Differences
from ._common import _callable, _check_choice, _real, _real_vector, _type_name, _values_text, _x0

# ---- the public functions


def _f_values(v, func, what, m=None):
    """What a user function returned: (values as a 1-D float64 sd.ndarray, scalar). A real
    number (numpy's 0-dimensional arrays too) is a scalar, of 1 value; else 1-dimensional, of
    m values when m is given (the count at the first point)."""
    if isinstance(v, _sd.ndarray):
        a = v
    elif isinstance(v, numbers.Number) or (hasattr(v, "ndim") and v.ndim == 0):
        if isinstance(v, bool):
            raise TypeError(f"{func}(): '{what}' must return a real number or a 1-dimensional array, not bool")
        if isinstance(v, complex) or (hasattr(v, "dtype") and getattr(v.dtype, "kind", "") == "c"):
            raise TypeError(f"{func}(): '{what}' must return real values, not complex")
        try:
            t = float(v)
        except TypeError, ValueError:
            raise TypeError(
                f"{func}(): '{what}' must return a real number or a 1-dimensional array, not {_type_name(v)}"
            ) from None
        a = _sd.asarray([t])
        if m is not None and m != 1:
            raise ValueError(f"{func}(): '{what}' must return {_values_text(m)} as at the first point, not a scalar")
        return a, True
    else:
        if isinstance(v, (str, bytes)) or v is None:
            raise TypeError(
                f"{func}(): '{what}' must return a real number or a 1-dimensional array, not {_type_name(v)}"
            )
        try:
            a = _sd.asarray(v)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
        except TypeError, ValueError:
            raise TypeError(
                f"{func}(): '{what}' must return a real number or a 1-dimensional array, not {_type_name(v)}"
            ) from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{what}' must return real values, not a complex128 array")
    if a.dtype == _sd.string:
        raise TypeError(f"{func}(): '{what}' must return real values, not a string array")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    if a.ndim != 1:
        raise ValueError(
            f"{func}(): '{what}' must return a real number or a 1-dimensional array, not shape {tuple(a.shape)}"
        )
    if m is not None and a.shape[0] != m:
        raise ValueError(
            f"{func}(): '{what}' must return {_values_text(m)} as at the first point, not shape {tuple(a.shape)}"
        )
    return a, False


def _derivative(func, what, f, x, epsilon):
    """(J (m, n), scalar): approx_fprime's work for the public function func, whose messages
    name f as `what`."""
    n = x.shape[0]
    eps = (
        None
        if epsilon is None
        else _real_vector(
            epsilon, func, "epsilon", n, positive=True, like="'xk'" if func == "approx_fprime" else "'x0'"
        )
    )
    xl = [float(v) for v in x.tolist()]
    f0, scalar = _f_values(f(x.copy()), func, what)
    m = 1 if scalar else f0.shape[0]
    if m == 0:  # least_squares' message (O28)
        raise ValueError(f"{func}(): '{what}' must return at least 1 value")

    # f is called with a new float64 sd.ndarray each time; a float (m = 1) or a float64 array of
    # m values is read as it is, anything else goes through _f_values
    D = Differences(f, n, [-math.inf] * n, [math.inf] * n, array_x=True, check=lambda v: _f_values(v, func, what, m)[0])
    if eps is not None:
        # scipy's approx_fprime: (f(x + h_j e_j) - f(x)) / ((x_j + h_j) - x_j) with h_j = eps_j, or
        # sqrt(eps) sign(x_j) max(1, |x_j|) (sqrt(eps) = 2^-26) where x_j + eps_j rounds to x_j
        # (approx_derivative's abs_step)
        return D.forward(xl, f0, eps), scalar
    if not bool(_sd.all(_sd.isfinite(f0))):
        raise ValueError(f"{func}(): '{what}' is not finite at the point")
    J, bad = D.jacobian(xl, f0)
    if J is None:
        raise ValueError(f"{func}(): '{what}' is not finite near the point (the differences of variable {bad})")
    return J, scalar


def approx_fprime(xk, f, /, *, epsilon=None):
    """The gradient (f returns a real number: (n,)) or the Jacobian (f returns m values:
    (m, n)) of f at xk by differences: with epsilon=None the accurate 5-point differences
    with a step chosen per variable (about 1e-11 - 1e-13 relative); with epsilon (a number or
    n numbers) scipy's forward differences (docs/api/optimize.md §2)."""
    func = "approx_fprime"
    x = _x0(xk, func, "xk")
    _callable(f, func, "f")
    J, scalar = _derivative(func, "f", f, x, epsilon)
    return J[0] if scalar else J


def check_grad(func, grad, x0, /, *, epsilon=None, direction="all", rng=None):
    """The 2-norm of grad(x0) - approx_fprime(x0, func, epsilon=epsilon), as scipy's; with
    direction='random' |grad(x0) . v - d/dw func(x0 + w v)| at w = 0 along one direction v of
    standard normal entries drawn from rng (docs/api/optimize.md §2)."""
    fn = "check_grad"
    _callable(func, fn, "func")
    _callable(grad, fn, "grad")
    x = _x0(x0, fn, "x0")
    n = x.shape[0]
    _check_choice(direction, fn, "direction", ("all", "random"))
    if direction == "all":
        if rng is not None:
            raise ValueError(f"{fn}(): 'rng' is only used when direction='random'")
    else:
        from .. import random as _random

        if not isinstance(rng, _random.Generator):
            raise TypeError(
                f"{fn}(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not {_type_name(rng)}"
            )
        if epsilon is not None:
            _real(epsilon, fn, "epsilon")  # one step, along the direction

    def value(t):
        v = func(t)
        if type(v) is float:
            return v
        a, scalar = _f_values(v, fn, "func")
        if not scalar:
            raise ValueError(f"{fn}(): 'func' must return a real number, not shape {tuple(a.shape)}")
        return float(a[0])

    from ._minimize import _gradient  # minimize's jac rules: a number when n = 1, no bool array

    g0 = _gradient(grad(x.copy()), fn, n, "grad")
    if direction == "all":
        J, _ = _derivative(fn, "func", value, x, epsilon)
        return float(_linalg.vector_norm(g0 - J[0]))
    from ..stats import Normal

    v = Normal().sample((n,), rng=rng)
    analytic = float(_linalg.vecdot(g0, v))
    J, _ = _derivative(fn, "func", lambda w: value(x + float(w[0]) * v), _sd.asarray([0.0]), epsilon)
    return abs(analytic - float(J[0][0]))
