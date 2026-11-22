"""RegularGridInterpolator and interpn of sd.interpolate (docs/api/interpolate.md §6; docs/design/interpolate.md §2.7,
§3.7; milestone IP5).

A grid is one strictly monotone 1-D array of nodes per axis and the values (n_0, ..., n_{d-1}, ...batch); a decreasing
axis is flipped at construction (scipy's rule: `grid` and `values` are then ascending). The six methods
(csrc/interpolate/grid.cpp):

  linear, slinear   the linear kernel axis by axis from the last to the first: exact at the nodes and, in 1-D, the
                    bits of sd.interp and of make_interp_spline(k=1) (slinear is the same function as linear; with
                    `nu` it is the k = 1 tensor B-spline)
  nearest           per axis the left node when t <= 0.5, else the right one
  cubic, quintic    the tensor product of not-a-knot B-splines: the coefficients are the 1-D collocation solve along
                    each axis in turn (exact, no iterative solver; decision IP23), computed once per degree
  pchip             per point the 1-D PCHIP along each axis on the window of four nodes around it: scipy's function

Outside the grid: bounds_error=True raises, else fill_value (None extends the end pieces). A point with a NaN
coordinate is NaN in every case (decision IP24).
"""

import math
import numbers

from .. import _core as _sd
from .._calls import function, method
from . import _common as C
from ._bspline import _KNOTS_NOT_A_KNOT

_core = _sd.interpolate

METHODS = ("linear", "nearest", "slinear", "cubic", "quintic", "pchip")
_DEGREE = {"slinear": 1, "cubic": 3, "quintic": 5, "pchip": 3}  # each axis needs degree + 1 nodes (scipy)
_SPLINES = ("slinear", "cubic", "quintic")  # the methods that take nu (scipy)
_RULE_FILL, _RULE_EXTEND, _RULE_CHECK = 0, 1, 2


def _readonly(a):
    a._set_readonly()
    return a


def _is_bool(v):
    return isinstance(v, bool) or (type(v).__module__ == "numpy" and type(v).__name__ == "bool")


def _axes(points, name):
    """The axes: (read-only ascending C-contiguous float64 copies, the indices of the decreasing ones)."""
    if not isinstance(points, (tuple, list)):
        raise TypeError(f"{name}(): 'points' must be a tuple or a list of 1-D arrays, not {C.type_name(points)}")
    if not points:
        raise ValueError(f"{name}(): 'points' must have an axis or more")
    grid, flipped = [], []
    for i, p in enumerate(points):
        a = C.array(p, name, "points")
        if a.ndim != 1:
            raise ValueError(f"{name}(): the points in dimension {i} must be 1-D, not {a.ndim}-D")
        if a.shape[0] < 1:
            raise ValueError(f"{name}(): the points in dimension {i} must not be empty")
        a = C.contiguous(a.copy())
        code = _core.check_x(a, True)
        if code == -1:
            raise ValueError(f"{name}(): the points in dimension {i} must be finite")
        if code == -2:
            raise ValueError(
                f"{name}(): the points in dimension {i} must be strictly increasing or strictly decreasing"
            )
        if code == 1:
            a = C.contiguous(_sd.flip(a, axis=(0,)).copy())
            flipped.append(i)
        grid.append(_readonly(a))
    return tuple(grid), tuple(flipped)


def _check_counts(grid, meth, name):
    k = _DEGREE.get(meth)
    if k is None:
        return
    for i, p in enumerate(grid):
        if p.shape[0] <= k:
            raise ValueError(
                f"{name}(): method '{meth}' needs at least {k + 1} points per dimension, not {p.shape[0]} in "
                f"dimension {i}"
            )


def _fill(v, complex_values, name):
    if v is None:
        return None
    if not _is_bool(v) and isinstance(v, numbers.Real):
        return float(v)
    if not _is_bool(v) and isinstance(v, numbers.Complex):
        if not complex_values:
            raise ValueError(f"{name}(): 'fill_value' must be None or a real number for real values, not complex")
        return complex(v)
    raise TypeError(f"{name}(): 'fill_value' must be None or a number, not {C.type_name(v)}")


def _broadcast(shapes, name):
    nd = max((len(s) for s in shapes), default=0)
    out = []
    for k in range(nd):
        dims = {s[len(s) - nd + k] for s in shapes if len(s) - nd + k >= 0} - {1}
        if len(dims) > 1:
            raise ValueError(f"{name}(): the coordinate arrays of 'xi' do not broadcast to one shape")
        out.append(dims.pop() if dims else 1)
    return tuple(out)


def _coordinate(v, name):
    if isinstance(v, numbers.Number) or C._is_numpy_scalar(v):
        if isinstance(v, numbers.Complex) and not isinstance(v, numbers.Real):
            raise TypeError(f"{name}(): 'xi' must be real, not {C.type_name(v)}")
        return float(v)
    return C.array(v, name, "xi")


def _points(xi, d, name):
    """xi as (a C-contiguous (m, d) float64 array, the points' shape): a (..., d) array, a 1-D array of d coordinates
    per point (scipy's reshape(-1, d)), or a tuple of d coordinate arrays (broadcast, scipy's rule)."""
    sd = _sd
    if isinstance(xi, tuple) and len(xi) == 1:
        xi = xi[0]  # scipy: a 1-tuple is its element
    if isinstance(xi, tuple):
        parts = [_coordinate(v, name) for v in xi]
        if len(parts) != d:
            raise ValueError(f"{name}(): 'xi' has {len(parts)} coordinate arrays, but the grid has dimension {d}")
        lead = _broadcast([() if isinstance(p, float) else tuple(p.shape) for p in parts], name)
        m = math.prod(lead)
        cols = []
        for p in parts:
            if isinstance(p, float):
                cols.append(sd.full((m,), p))
            else:
                cols.append(sd.reshape(sd.broadcast_to(p, lead), (m,)))
        flat = C.contiguous(sd.stack(cols, axis=1)) if m else sd.zeros((0, d))
        return flat, lead
    if isinstance(xi, numbers.Number) or C._is_numpy_scalar(xi):
        raise ValueError(f"{name}(): 'xi' must have at least 1 dimension")
    a = C.array(xi, name, "xi")
    if a.ndim == 1:
        if a.shape[0] % d:
            raise ValueError(f"{name}(): 'xi' of {a.shape[0]} coordinates is not a whole number of {d}-D points")
        a = sd.reshape(a, (a.shape[0] // d, d))
    if a.shape[-1] != d:
        raise ValueError(f"{name}(): the points in 'xi' have dimension {a.shape[-1]}, but the grid has dimension {d}")
    lead = tuple(a.shape[:-1])
    return C.contiguous(sd.reshape(a, (math.prod(lead), d))), lead


class RegularGridInterpolator:
    """Interpolation on a regular grid in d dimensions: the nodes of each axis and the values at every node."""

    __slots__ = ("_grid", "_values", "_v2", "_batch", "_method", "_bounds_error", "_fill_value", "_coefs", "_finite")

    @method("RegularGridInterpolator")
    def __init__(self, points, values, /, *, method="linear", bounds_error=True, fill_value=math.nan):
        self._build(points, values, method, bounds_error, fill_value, "RegularGridInterpolator")

    def _build(self, points, values, meth, bounds_error, fill_value, name):
        sd = _sd
        meth = C.choice(meth, name, "method", METHODS)
        if not _is_bool(bounds_error):
            raise TypeError(f"{name}(): 'bounds_error' must be True or False, not {C.type_name(bounds_error)}")
        grid, flipped = _axes(points, name)
        _check_counts(grid, meth, name)
        vv = C.array(values, name, "values", complex_ok=True)
        d = len(grid)
        if vv.ndim < d:
            raise ValueError(f"{name}(): there are {d} point arrays, but 'values' has {vv.ndim} dimensions")
        for i, p in enumerate(grid):
            if vv.shape[i] != p.shape[0]:
                raise ValueError(f"{name}(): there are {p.shape[0]} points and {vv.shape[i]} values in dimension {i}")
        fill = _fill(fill_value, vv.dtype == sd.complex128, name)
        if flipped:
            vv = sd.flip(vv, axis=flipped)
        vv = _readonly(C.contiguous(vv.copy()))
        self._grid = grid
        self._values = vv
        self._batch = tuple(vv.shape[d:])
        rows = math.prod(vv.shape[:d])
        self._v2 = C.contiguous(sd.reshape(vv, (rows, math.prod(self._batch))))
        self._method = meth
        self._bounds_error = bool(bounds_error)
        self._fill_value = fill
        self._coefs = {}
        self._finite = None
        self._ready(meth, name)

    def _ready(self, meth, name):
        """The checks of a method on these values (and the spline coefficients it needs): the global solves and pchip
        take finite values, pchip real ones (scipy's errors)."""
        _check_counts(self._grid, meth, name)
        if meth == "pchip" and self._v2.dtype == _sd.complex128:
            raise ValueError(f"{name}(): method 'pchip' takes real values, not complex128")
        if meth in ("cubic", "quintic", "pchip"):
            if self._finite is None:
                self._finite = bool(_core.all_finite(self._v2))
            if not self._finite:
                raise ValueError(f"{name}(): 'values' must be finite for method '{meth}'")
        if meth in ("cubic", "quintic"):
            self._spline(_DEGREE[meth], name)

    def _spline(self, k, name):
        """(the knots of every axis, the coefficients (rows, K)) of the degree-k tensor spline."""
        got = self._coefs.get(k)
        if got is None:
            knots = [_core.bspline_knots(p, k, _KNOTS_NOT_A_KNOT) for p in self._grid]
            if k == 1:
                c = self._v2
            else:
                c, status = _core.grid_spline_coefs(list(self._grid), knots, k, self._v2)
                if status < 0:
                    from ..linalg import LinAlgError

                    raise LinAlgError(f"{name}(): the collocation matrix of an axis is singular")
            got = self._coefs[k] = (knots, _readonly(c))
        return got

    # ---- attributes ----

    @property
    def grid(self):
        """The nodes of every axis, ascending (a decreasing axis is flipped), read-only."""
        return self._grid

    @property
    def values(self):
        """The values (n_0, ..., n_{d-1}, ...batch), flipped with the decreasing axes, read-only."""
        return self._values

    @property
    def method(self):
        """The default method of the calls."""
        return self._method

    @property
    def bounds_error(self):
        """True: a point outside the grid is a ValueError."""
        return self._bounds_error

    @property
    def fill_value(self):
        """The value outside the grid when bounds_error is False (None: the end pieces extended)."""
        return self._fill_value

    def __repr__(self):
        shape = "x".join(str(p.shape[0]) for p in self._grid)
        return f"{type(self).__name__}(grid={shape}, method={self._method!r})"

    # ---- evaluation ----

    def __call__(self, xi, /, *, method=None, nu=None):
        """The interpolated values at the points xi: (...xi's points, ...batch)."""
        return self._evaluate(xi, method, nu, type(self).__name__)

    def _evaluate(self, xi, meth, nu, name):
        sd = _sd
        meth = self._method if meth is None else C.choice(meth, name, "method", METHODS)
        d = len(self._grid)
        orders = None
        if nu is not None:
            if meth not in _SPLINES:
                raise ValueError(f"{name}(): 'nu' is for the methods 'slinear', 'cubic' and 'quintic', not '{meth}'")
            if isinstance(nu, (str, bytes)) or not hasattr(nu, "__iter__"):
                raise TypeError(f"{name}(): 'nu' must be a sequence of {d} ints, not {C.type_name(nu)}")
            orders = [C.order(v, name, "nu") for v in nu]
            if len(orders) != d:
                raise ValueError(f"{name}(): 'nu' must have one order per dimension ({d}), not {len(orders)}")
        if meth != self._method:
            self._ready(meth, name)
        flat, lead = _points(xi, d, name)
        if self._bounds_error:
            rule = _RULE_CHECK
        else:
            rule = _RULE_EXTEND if self._fill_value is None else _RULE_FILL
        f = 0j if self._fill_value is None else complex(self._fill_value)
        grid = list(self._grid)
        if meth in ("linear", "nearest") or (meth == "slinear" and not any(orders or ())):
            out, bad = _core.grid_linear(grid, self._v2, flat, meth == "nearest", rule, f.real, f.imag)
        elif meth == "pchip":
            out, bad = _core.grid_pchip(grid, self._v2, flat, rule, f.real)
        else:
            k = _DEGREE[meth]
            knots, c = self._spline(k, name)
            nua = sd.asarray(orders or [0] * d, dtype=sd.int64)
            out, bad = _core.grid_spline_eval(knots, k, c, flat, nua, rule, f.real, f.imag)
        if bad >= 0:
            raise ValueError(f"{name}(): one of the requested xi is out of bounds in dimension {bad}")
        shape = lead + self._batch
        if not shape:
            return out.reshape((out.size,))[0]
        return sd.reshape(out, shape)


@function
def interpn(points, values, xi, /, *, method="linear", bounds_error=True, fill_value=math.nan):
    """Interpolation on a regular grid at the points xi: RegularGridInterpolator(points, values, ...)(xi)."""
    name = "interpn"
    obj = RegularGridInterpolator.__new__(RegularGridInterpolator)
    obj._build(points, values, method, bounds_error, fill_value, name)
    return obj._evaluate(xi, None, None, name)
