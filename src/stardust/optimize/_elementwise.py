"""The batched scalar functions: find_root, bracket_root, find_minimum, bracket_minimum
(docs/api/optimize.md §6, docs/design/optimize.md §3.2, §3.3; scipy 1.18.1's
optimize.elementwise, O5).

Each element is solved on its own, the same steps as scipy's (Chandrupatla's methods and scipy's
bracket searches; O9). The user's function is called once per iteration with the elements still
in progress, as a new float64 sd.ndarray of shape (k,), and returns k values (O38); its extra
arguments args (O39, the one exception to O12) broadcast with the inputs and reach it reduced to
the same elements. The inputs
broadcast together; their shape is the result's batch shape, and every field of the result has
that shape (Python numbers when all inputs are numbers). The iterations are the native kernels
stardust._core.optimize.find_root_solve etc. (csrc/optimize/batch.cpp).
"""

import math
import numbers

from .. import _core as _sd
from ._common import _EPS, _as_array, _callable, _check_int, _check_tol, _real, _type_name, _values_text
from ._result import OptimizeResult

_k = _sd.optimize

_TINY = 2.2250738585072014e-308  # the smallest normal double
# scipy's default max_iter of find_root: log2(max) - log2(tiny), enough to halve any bracket to
# the smallest normal number
_ROOT_MAX_ITER = 2046


def _values(func):
    """check(value, k) for the kernels: what f returned, as a float64 sd.ndarray of shape (k,), or
    the error."""

    def check(v, k):
        if isinstance(v, _sd.ndarray):
            a = v
        else:
            if isinstance(v, (str, bytes, bool)) or v is None:
                raise TypeError(
                    f"{func}(): 'f' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
                )
            if isinstance(v, numbers.Number) or getattr(v, "ndim", None) == 0:
                raise ValueError(f"{func}(): 'f' must return shape ({k},) like its argument, not a scalar")
            try:
                a = _sd.asarray(v)
            except OverflowError:
                raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
            except TypeError, ValueError:
                raise TypeError(
                    f"{func}(): 'f' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
                ) from None
        if a.dtype == _sd.complex128:
            raise TypeError(f"{func}(): 'f' must return real values, not a complex128 array")
        if a.dtype == _sd.string:
            raise TypeError(f"{func}(): 'f' must return real values, not a string array")
        if a.dtype == _sd.bool:  # a comparison returned where numbers were meant
            raise TypeError(f"{func}(): 'f' must return real values, not a bool array")
        if a.ndim != 1 or a.shape[0] != k:
            raise ValueError(f"{func}(): 'f' must return shape ({k},) like its argument, not {tuple(a.shape)}")
        if a.dtype != _sd.float64:
            a = a.astype(_sd.float64)
        return a

    return check


def _broadcast_shape(func, shapes):
    out = []
    for s in shapes:
        s = tuple(s)
        n = max(len(out), len(s))
        a = [1] * (n - len(out)) + out
        b = [1] * (n - len(s)) + list(s)
        merged = []
        for p, q in zip(a, b):
            if p != q and p != 1 and q != 1:
                listed = ", ".join(str(tuple(t)) for t in shapes)
                raise ValueError(f"{func}(): the inputs must broadcast together, not shapes {listed}")
            merged.append(q if p == 1 else p)
        out = merged
    return tuple(out)


def _args(args, func):
    """The user's extra arguments (O39): a tuple of numbers or arrays, each as an sd.ndarray of its
    own dtype (a number as one value) and whether it was a number."""
    if not isinstance(args, tuple):
        raise TypeError(f"{func}(): 'args' must be a tuple, not {_type_name(args)}")
    out = []
    for j, v in enumerate(args):
        name = f"args[{j}]"
        number = isinstance(v, numbers.Number) or (getattr(v, "ndim", None) == 0 and not isinstance(v, _sd.ndarray))
        if number and hasattr(v, "item"):
            v = v.item()
        if isinstance(v, _sd.ndarray):
            a = v
        elif isinstance(v, (str, bytes)) or v is None:
            raise TypeError(f"{func}(): '{name}' must be a number or an array, not {_type_name(v)}")
        else:
            try:
                a = _sd.asarray([v] if number else v)
            except OverflowError:
                raise OverflowError(f"{func}(): '{name}': Python int too large to convert to int64") from None
            except TypeError, ValueError:
                raise TypeError(f"{func}(): '{name}' must be a number or an array, not {_type_name(v)}") from None
        if a.dtype == _sd.string:
            raise TypeError(f"{func}(): '{name}' must be a number or an array, not a string array")
        out.append((a, number))
    return out


def _inputs(func, named, args=()):
    """The inputs (name, value) as float64 sd.ndarrays of n elements (1-dimensional), the user's
    args (from _args) as 1-dimensional sd.ndarrays of n elements of their own dtype, and the
    broadcast shape of them all (() when all are numbers)."""
    vals = []
    for name, v in named:
        if isinstance(v, numbers.Real) and not isinstance(v, bool):
            vals.append((name, _real(v, func, name)))
        elif getattr(v, "ndim", None) == 0 and not isinstance(v, _sd.ndarray):  # a 0-d numpy array
            if getattr(v.dtype, "kind", "") == "c":
                raise TypeError(f"{func}(): '{name}' must be real, not complex")
            vals.append((name, _real(v.item() if hasattr(v, "item") else v, func, name)))
        elif isinstance(v, bool):
            raise TypeError(f"{func}(): '{name}' must be a real number or an array, not bool")
        else:
            vals.append((name, _as_array(v, func, name)))
    shapes = [() if isinstance(v, float) else v.shape for _, v in vals]
    shape = _broadcast_shape(func, shapes + [() if number else a.shape for a, number in args])
    n = math.prod(shape)
    out = []
    for _, v in vals:
        if isinstance(v, float):
            out.append(_sd.full((n,), v, dtype=_sd.float64))
        else:
            out.append(_sd.reshape(_sd.broadcast_to(v, shape), (n,)))
    extra = tuple(
        _sd.broadcast_to(a, (n,)) if number else _sd.reshape(_sd.broadcast_to(a, shape), (n,)) for a, number in args
    )
    return out, extra, shape


def _init(init, func, names):
    """init: a tuple or list (or an array split along its first axis) of len(names) values."""
    want = len(names)
    form = "pair" if want == 2 else "triple"
    listed = "(" + ", ".join(names) + ")"
    if isinstance(init, (tuple, list)):
        parts = list(init)
    elif isinstance(init, _sd.ndarray) or getattr(init, "ndim", 0) >= 1:
        parts = [init[i] for i in range(len(init))]
    else:
        raise TypeError(f"{func}(): 'init' must be a {form} {listed}, not {_type_name(init)}")
    if len(parts) != want:
        raise ValueError(f"{func}(): 'init' must be a {form} {listed}, not {_values_text(len(parts))}")
    return parts


def _tol(v, default, func, name):
    return default if v is None else _check_tol(v, func, name)


def _shaped(a, shape):
    """A result array (n,) in the batch shape: a Python number when shape is ()."""
    if shape == ():
        return a[0]
    return _sd.reshape(a, shape)


def _result(shape, status, nit, nfev, **fields):
    out = {
        k: (tuple(_shaped(a, shape) for a in v) if isinstance(v, tuple) else _shaped(v, shape))
        for k, v in fields.items()
    }
    return OptimizeResult._make(
        batch=shape,
        **out,
        converged=_shaped(status == 0, shape),
        status=_shaped(status, shape),
        n_iter=_shaped(nit, shape),
        nfev=_shaped(nfev, shape),
    )


def find_root(f, init, /, *, args=(), xatol=None, xrtol=None, fatol=None, frtol=None, max_iter=None):
    """The root of f in each bracket init = (xl, xr), f(xl) and f(xr) of opposite signs, by
    Chandrupatla's method (scipy.optimize.elementwise.find_root; docs/api/optimize.md §6)."""
    func = "find_root"
    _callable(f, func, "f")
    xl, xr = _init(init, func, ("xl", "xr"))
    (xl, xr), extra, shape = _inputs(func, [("init", xl), ("init", xr)], _args(args, func))
    xatol = _tol(xatol, 4 * _TINY, func, "xatol")
    xrtol = _tol(xrtol, 4 * _EPS, func, "xrtol")
    fatol = _tol(fatol, _TINY, func, "fatol")
    frtol = _tol(frtol, 0.0, func, "frtol")
    max_iter = _ROOT_MAX_ITER if max_iter is None else _check_int(max_iter, func, "max_iter", 0)
    x, fx, bl, br, fl, fr, status, nit, nfev = _k.find_root_solve(
        func, f, _values(func), extra, xl, xr, xatol, xrtol, fatol, frtol, max_iter
    )
    return _result(shape, status, nit, nfev, x=x, fun=fx, bracket=(bl, br), f_bracket=(fl, fr))


def find_minimum(f, init, /, *, args=(), xatol=None, xrtol=None, fatol=None, frtol=None, max_iter=100):
    """The minimum of f in each bracket init = (x1, x2, x3), x1 < x2 < x3 and f(x2) below f(x1)
    and f(x3), by Chandrupatla's method (scipy.optimize.elementwise.find_minimum;
    docs/api/optimize.md §6)."""
    func = "find_minimum"
    _callable(f, func, "f")
    parts = _init(init, func, ("x1", "x2", "x3"))
    (x1, x2, x3), extra, shape = _inputs(func, [("init", p) for p in parts], _args(args, func))
    xatol = _tol(xatol, _TINY, func, "xatol")
    xrtol = _tol(xrtol, math.sqrt(_EPS), func, "xrtol")
    fatol = _tol(fatol, _TINY, func, "fatol")
    frtol = _tol(frtol, _TINY, func, "frtol")
    max_iter = _check_int(max_iter, func, "max_iter", 0)
    x, fx, bl, bm, br, fl, fm, fr, status, nit, nfev = _k.find_minimum_solve(
        func, f, _values(func), extra, x1, x2, x3, xatol, xrtol, fatol, frtol, max_iter
    )
    return _result(shape, status, nit, nfev, x=x, fun=fx, bracket=(bl, bm, br), f_bracket=(fl, fm, fr))


def _check_factor(a, func):
    if not bool(_sd.all(a > 1)):
        raise ValueError(f"{func}(): 'factor' must be > 1")


def bracket_root(f, xl0, /, xr0=None, *, xmin=None, xmax=None, factor=None, args=(), max_iter=1000):
    """A bracket (xl, xr) of a root of f for each element: the interval (xl0, xr0) grown by
    'factor' in both directions until f changes sign (scipy.optimize.elementwise.bracket_root;
    docs/api/optimize.md §6)."""
    func = "bracket_root"
    _callable(f, func, "f")
    named = [
        ("xl0", xl0),
        ("xr0", math.nan if xr0 is None else xr0),
        ("xmin", -math.inf if xmin is None else xmin),
        ("xmax", math.inf if xmax is None else xmax),
        ("factor", 2.0 if factor is None else factor),
    ]
    (xl, xr, lo, hi, fac), extra, shape = _inputs(func, named, _args(args, func))
    _check_factor(fac, func)
    max_iter = _check_int(max_iter, func, "max_iter", 0)
    bl, br, fl, fr, status, nit, nfev = _k.bracket_root_solve(
        func, f, _values(func), extra, xl, xr, xr0 is not None, lo, hi, fac, max_iter
    )
    return _result(shape, status, nit, nfev, bracket=(bl, br), f_bracket=(fl, fr))


def bracket_minimum(f, xm0, /, *, xl0=None, xr0=None, xmin=None, xmax=None, factor=None, args=(), max_iter=1000):
    """A bracket (xl, xm, xr) of a minimum of f for each element, f(xm) below f(xl) and f(xr):
    from (xl0, xm0, xr0) downhill by steps growing by 'factor' (scipy.optimize.elementwise.
    bracket_minimum; docs/api/optimize.md §6)."""
    func = "bracket_minimum"
    _callable(f, func, "f")
    named = [
        ("xm0", xm0),
        ("xl0", math.nan if xl0 is None else xl0),
        ("xr0", math.nan if xr0 is None else xr0),
        ("xmin", -math.inf if xmin is None else xmin),
        ("xmax", math.inf if xmax is None else xmax),
        ("factor", 2.0 if factor is None else factor),
    ]
    (xm, xl, xr, lo, hi, fac), extra, shape = _inputs(func, named, _args(args, func))
    _check_factor(fac, func)
    max_iter = _check_int(max_iter, func, "max_iter", 0)
    bl, bm, br, fl, fm, fr, status, nit, nfev = _k.bracket_minimum_solve(
        func, f, _values(func), extra, xm, xl, xl0 is not None, xr, xr0 is not None, lo, hi, fac, max_iter
    )
    return _result(shape, status, nit, nfev, bracket=(bl, bm, br), f_bracket=(fl, fm, fr))
