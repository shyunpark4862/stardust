"""The interpolants on sd.spatial's Delaunay triangulation (docs/api/interpolate.md §7; docs/design/interpolate.md
§2.8, §3.8; docs/design/spatial_geometry.md §2.6, §3.8, §3.9; milestone IP9): LinearNDInterpolator (d = 2, 3),
CloughTocher2DInterpolator (d = 2) and griddata.

An object holds its Delaunay (built from the points, or the one given as `points`), the values as an (n, S) array with
the rest of their shape, and fill_value; a CloughTocher2DInterpolator also its gradients (n, ..., 2), estimated once
(csrc/interpolate/scattered.cpp: scipy's global Gauss-Seidel method, one thread per column, so the same bits for any
thread count). A call locates each point exactly (Delaunay.find_simplex's smallest closed simplex, -1 outside the
convex hull) and evaluates it in the same pass: the barycentric weights from the value kernels, or the Clough-Tocher
cubic. Outside the hull the value is fill_value; a point with a NaN coordinate is NaN.

Equal data points (the Delaunay keeps the smallest index) must have the same values to the bit (ValueError
otherwise; scipy keeps one of them silently). rescale is scipy's (the mean and the range of each axis, as
NearestNDInterpolator). Pickling keeps the slots (decision IP110): the Delaunay pickles as its points and is rebuilt
into the same triangulation, the gradients are kept as they are.
"""

import math
import numbers

from .. import _core as _sd
from .._calls import call, function, method
from . import _common as C
from ._rbf import NearestNDInterpolator, _finish, _readonly, _values, point_set, rescaling, xi_points

_core = _sd.interpolate

METHODS = ("nearest", "linear", "cubic")


def _fill(v, complex_values, name):
    if isinstance(v, bool) or type(v).__name__ == "bool" or not isinstance(v, numbers.Complex):
        raise TypeError(f"{name}(): 'fill_value' must be a number, not {C.type_name(v)}")
    if isinstance(v, numbers.Real):
        return float(v)
    if not complex_values:
        raise ValueError(f"{name}(): 'fill_value' must be a real number for real values, not complex")
    return complex(v)


def _flag(v, name, arg):
    if v is not True and v is not False:
        raise TypeError(f"{name}(): '{arg}' must be True or False, not {C.type_name(v)}")
    return v


def _triangulate(points, rescale, dims, name):
    """(Delaunay, its points as given (n, d) read-only, offset, scale): `points` an (n, d) set (or a tuple of d
    coordinate arrays) or an sd.spatial.Delaunay; d in dims."""
    from ..spatial import Delaunay

    if isinstance(points, Delaunay):
        if rescale:
            raise ValueError(f"{name}(): 'rescale' must be False when 'points' is a Delaunay")
        tri = points
        pts = tri.points
    else:
        a = C.array(point_set(points, name, "points"), name, "points")
        if a.ndim != 2:
            raise ValueError(f"{name}(): 'points' must have 2 dimensions (points, coordinates), not {a.ndim}")
        tri = None
        pts = _readonly(C.contiguous(a.copy()))
    d = pts.shape[1]
    if d not in dims:
        cols = " or ".join(str(k) for k in dims)
        raise ValueError(f"{name}(): 'points' must have {cols} columns, not {d}")
    offset = scale = None
    if tri is None:
        C.finite(pts, name, "points")
        base = pts
        if rescale:
            offset, scale, base = rescaling(pts)
        try:
            tri = Delaunay(base)
        except (ValueError, MemoryError) as e:
            raise type(e)(f"{name}(): {str(e).partition('(): ')[2]}") from None
    return tri, pts, offset, scale


class _Scattered:
    """What LinearNDInterpolator and CloughTocher2DInterpolator share: the triangulation, the values, fill_value,
    the call's points."""

    __slots__ = ("_tri", "_points", "_v", "_rest", "_fill", "_offset", "_scale")

    def __reduce__(self):  # sd.ndarray does not pickle: the slots, arrays as bytes (the Delaunay rebuilds itself)
        return C.reduce(self)

    def _setup(self, points, values, fill_value, rescale, dims, name):
        _flag(rescale, name, "rescale")
        tri, pts, offset, scale = _triangulate(points, rescale, dims, name)
        v, rest = _values(values, pts.shape[0], name, "values")
        fill = _fill(fill_value, v.dtype == _sd.complex128, name)
        i, j = _core.scattered_duplicate(tri._tri, v)
        if j >= 0:
            raise ValueError(f"{name}(): data points {i} and {j} are equal and their values differ")
        self._tri, self._points, self._v, self._rest = tri, pts, v, rest
        self._fill, self._offset, self._scale = fill, offset, scale

    def _xi(self, xi, name):
        X, batch = xi_points(xi, self._points.shape[1], name)
        if self._offset is not None and X.shape[0]:
            X = C.contiguous((X - self._offset) / self._scale)
        return X, batch

    # ---- attributes ----

    @property
    def points(self):
        """The data points (n, d) float64, read-only (as given: before any rescale)."""
        return self._points

    @property
    def values(self):
        """The values (n, ...) float64 or complex128, read-only."""
        return _sd.reshape(self._v, (self._v.shape[0],) + self._rest)

    @property
    def tri(self):
        """The sd.spatial.Delaunay the interpolant is built on (of the rescaled points when rescale=True)."""
        return self._tri

    @property
    def fill_value(self):
        """The value outside the convex hull of the points (a float, or a complex for complex values)."""
        return self._fill


def _fill_parts(fill):
    f = complex(fill)
    return f.real, f.imag


class LinearNDInterpolator(_Scattered):
    """The piecewise linear interpolant on the Delaunay triangulation of the points (d = 2, 3)."""

    __slots__ = ()

    @method("LinearNDInterpolator")
    def __init__(self, points, values, /, *, fill_value=math.nan, rescale=False):
        self._setup(points, values, fill_value, rescale, (2, 3), "LinearNDInterpolator")

    def __repr__(self):
        n, d = self._points.shape
        return f"LinearNDInterpolator(points={n}, dimensions={d})"

    @call
    def __call__(self, xi, /):
        """The interpolated values at xi ((..., d), or a tuple of d coordinate arrays): fill_value outside the convex
        hull, NaN at a point with a NaN coordinate."""
        return self._evaluate(xi, "LinearNDInterpolator")

    def _evaluate(self, xi, name):
        X, batch = self._xi(xi, name)
        out = _core.linear_nd(self._tri._tri, X, self._v, *_fill_parts(self._fill))
        return _finish(out, batch, self._rest)


class CloughTocher2DInterpolator(_Scattered):
    """The piecewise cubic, C1 interpolant of Clough and Tocher on the Delaunay triangulation of 2-D points, with the
    gradients of scipy's global estimate."""

    __slots__ = ("_grad", "_converged", "_niter")

    @method("CloughTocher2DInterpolator")
    def __init__(self, points, values, /, *, fill_value=math.nan, tol=1e-6, maxiter=400, rescale=False):
        self._build(points, values, fill_value, tol, maxiter, rescale, "CloughTocher2DInterpolator")

    def _build(self, points, values, fill_value, tol, maxiter, rescale, name):
        t = C.real(tol, name, "tol")
        if not (math.isfinite(t) and t >= 0):
            raise ValueError(f"{name}(): 'tol' must be a finite number >= 0, not {t!r}")
        it = C.integer(maxiter, name, "maxiter")
        if it < 1:
            raise ValueError(f"{name}(): 'maxiter' must be >= 1, not {it}")
        self._setup(points, values, fill_value, rescale, (2,), name)
        C.finite(self._v, name, "values")
        indptr, indices = self._tri.vertex_neighbor_vertices  # made once and kept by the Delaunay
        grad, sweeps, converged = _core.ct_gradients(self._tri._tri, self._v, t, it, indptr, indices)
        self._grad = _readonly(grad)
        self._niter, self._converged = int(sweeps), bool(converged)

    @property
    def grad(self):
        """The estimated gradients (n, ..., 2) at the data points, of the values' dtype (in the rescaled coordinates
        when rescale=True), read-only."""
        return _sd.reshape(self._grad, (self._grad.shape[0],) + self._rest + (2,))

    @property
    def converged(self):
        """True when every column's sweeps stopped by tol (False: they stopped at maxiter)."""
        return self._converged

    @property
    def niter(self):
        """The Gauss-Seidel sweeps of the gradient estimate (the most any column took)."""
        return self._niter

    def __repr__(self):
        return f"CloughTocher2DInterpolator(points={self._points.shape[0]}, dimensions=2)"

    @call
    def __call__(self, xi, /):
        """The interpolated values at xi ((..., 2), or a tuple of 2 coordinate arrays): fill_value outside the convex
        hull, NaN at a point with a NaN coordinate."""
        return self._evaluate(xi, "CloughTocher2DInterpolator")

    def _evaluate(self, xi, name):
        X, batch = self._xi(xi, name)
        out = _core.ct_eval(self._tri._tri, X, self._v, self._grad, *_fill_parts(self._fill))
        return _finish(out, batch, self._rest)


@function
def griddata(points, values, xi, /, *, method="linear", fill_value=math.nan, rescale=False):
    """Interpolation of scattered data at xi: NearestNDInterpolator (any dimension; fill_value unused),
    LinearNDInterpolator (2-D, 3-D) or CloughTocher2DInterpolator (2-D) of (points, values), called once."""
    name = "griddata"
    meth = C.choice(method, name, "method", METHODS)
    from ..spatial import Delaunay

    if meth == "nearest":
        obj = NearestNDInterpolator.__new__(NearestNDInterpolator)
        pts = points.points if isinstance(points, Delaunay) else _one_column(points, name)
        obj._build(pts, values, rescale, name)
        if pts.shape[1] == 1 and not isinstance(xi, tuple):  # 1-D data: a 1-D xi is its points (scipy's form)
            a = C.array(xi, name, "xi")
            if a.ndim == 1:
                return obj._evaluate(_sd.reshape(a, (a.shape[0], 1)), name)
        return obj._evaluate(xi, name)

    if not isinstance(points, Delaunay):
        a = _one_column(points, name)
        d = a.shape[1] if getattr(a, "ndim", 0) == 2 else None
        if d == 1:
            raise ValueError(f"{name}(): 1-D points: use sd.interp (method 'linear') or CubicSpline (method 'cubic')")
        if meth == "cubic" and d == 3:
            raise ValueError(f"{name}(): method 'cubic' is for 2-D points, not 3-D")
        points = a
    if meth == "linear":
        obj = LinearNDInterpolator.__new__(LinearNDInterpolator)
        obj._setup(points, values, fill_value, rescale, (2, 3), name)
    else:
        obj = CloughTocher2DInterpolator.__new__(CloughTocher2DInterpolator)
        obj._build(points, values, fill_value, 1e-6, 400, rescale, name)
    return obj._evaluate(xi, name)


def _one_column(points, name):
    """griddata's points: a 1-D array is n points of one coordinate (scipy's form for 1-D data)."""
    from ..spatial import Delaunay

    if isinstance(points, Delaunay):
        return points
    a = C.array(point_set(points, name, "points"), name, "points")
    if a.ndim == 1:
        a = _sd.reshape(a, (a.shape[0], 1))
    return a
