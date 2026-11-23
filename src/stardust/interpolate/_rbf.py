"""The scattered-data interpolants of sd.interpolate (docs/api/interpolate.md §7; docs/design/interpolate.md §2.8,
§3.8; milestone IP6): RBFInterpolator and NearestNDInterpolator.

An RBFInterpolator holds its data points y (n, m), the values as an (n, S) array with the rest of their shape,
the smoothing (n,), the kernel, epsilon and the monomial powers of the polynomial tail (scipy's order). Without
neighbors (or with neighbors >= n, the same system: decision IP95) it solves the saddle-point system once
(csrc/interpolate/rbf.cpp: sytrf, then ldlt_solve) and keeps the coefficients with the tail's shift and scale; with
neighbors=k it keeps an sd.spatial.KDTree and, at each call, solves one local system per distinct set of k
nearest data points (ties by index, so the sets do not depend on the run or the thread count).

Errors instead of scipy's warnings and silent values (decisions IP25, IP29, IP90-IP94): a negative smoothing, a
degree below the kernel's minimum (-1 included), an epsilon that is not finite and positive are ValueError; two
equal data points with zero smoothing, a polynomial tail without full column rank and a singular system are
LinAlgError. A NearestNDInterpolator returns the value of the nearest data point (KDTree.query(k=1): ties to the
smaller index), rescale as scipy's. A point that is not finite gives NaN where no neighbourhood exists (the local
systems, NearestND); the global interpolant computes its formula there as scipy does.
"""

import math

from .. import _core as _sd
from .._calls import call, method
from . import _common as C

_core = _sd.interpolate

KERNELS = ("linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric",
           "inverse_quadratic", "gaussian")  # fmt: skip
SCALE_INVARIANT = ("linear", "thin_plate_spline", "cubic", "quintic")
MIN_DEGREE = {"multiquadric": 0, "linear": 0, "thin_plate_spline": 1, "cubic": 1, "quintic": 2}


def _readonly(a):
    a._set_readonly()
    return a


def monomial_powers(m, degree):
    """scipy's monomial exponents of degree <= `degree` in m variables, as an (q, m) int64 array (q = 0 for -1)."""
    out = []
    for total in range(degree + 1):

        def rec(start, left, row):
            if left == 0:
                out.append(list(row))
                return
            for var in range(start, m):  # combinations_with_replacement(range(m), total), in its order
                row[var] += 1
                rec(var, left - 1, row)
                row[var] -= 1

        rec(0, total, [0] * m)
    a = _sd.asarray(out, dtype=_sd.int64) if out else _sd.zeros((0, m), dtype=_sd.int64)
    return C.contiguous(_sd.reshape(a, (len(out), m)))


def _coords(x, name, arg):
    """Points (n, m): a C-contiguous finite float64 copy, at least one point and one coordinate."""
    a = C.array(x, name, arg)
    if a.ndim != 2:
        raise ValueError(f"{name}(): '{arg}' must have 2 dimensions (points, coordinates), not {a.ndim}")
    if a.shape[0] < 1:
        raise ValueError(f"{name}(): '{arg}' must have at least 1 point")
    if a.shape[1] < 1:
        raise ValueError(f"{name}(): '{arg}' must have at least 1 coordinate")
    a = C.contiguous(a.copy())
    C.finite(a, name, arg)
    return _readonly(a)


def _values(v, n, name, arg):
    """(values (n, S) C-contiguous copy, the rest of their shape): float64 or complex128, first axis n."""
    a = C.array(v, name, arg, complex_ok=True)
    if a.shape[0] != n:
        raise ValueError(f"{name}(): '{arg}' must have one row per data point ({n}), not {a.shape[0]}")
    rest = tuple(a.shape[1:])
    return _readonly(C.contiguous(_sd.reshape(a, (n, math.prod(rest))).copy())), rest


def _eval_points(x, m, name, arg="x"):
    """Evaluation points (..., m) as (a C-contiguous (Q, m) float64 array, the batch shape)."""
    if isinstance(x, (int, float, complex)) or C._is_numpy_scalar(x):
        raise ValueError(f"{name}(): '{arg}' must have at least 1 dimension (points of {m} coordinates)")
    a = C.array(x, name, arg)
    if a.shape[-1] != m:
        raise ValueError(
            f"{name}(): '{arg}' must have {m} entries on its last axis (the data points' dimension), not {a.shape[-1]}"
        )
    batch = tuple(a.shape[:-1])
    return C.contiguous(_sd.reshape(a, (math.prod(batch), m))), batch


def _finish(out, batch, rest):
    """(Q, S) values as batch + rest; a Python scalar for one point without a batch and scalar values."""
    if not batch and not rest:
        return out.reshape((out.size,))[0]
    return _sd.reshape(out, batch + rest)


def _finite_rows(X):
    """(the rows of X that are finite as a C-contiguous array, their mask, or None when all are)."""
    sd = _sd
    ok = sd.all(sd.isfinite(X), axis=1)
    if bool(sd.all(ok)):
        return X, None
    return C.contiguous(X[ok]), ok


def _scatter(out_f, ok, q, dtype):
    """(Q, S) results with NaN at the rows outside the mask (None: out_f itself)."""
    if ok is None:
        return out_f
    out = _sd.full((q, out_f.shape[1]), math.nan, dtype=dtype)
    out[ok] = out_f
    return out


# ---------------------------------------------------------------- RBFInterpolator


class RBFInterpolator:
    """The radial basis function interpolant of values d at the scattered points y."""

    __slots__ = ("_y", "_d", "_rest", "_smoothing", "_kernel", "_kid", "_epsilon", "_degree", "_powers", "_k",
                 "_tree", "_coeffs", "_shift", "_scale")  # fmt: skip

    def __reduce__(self):  # sd.ndarray does not pickle: the slots, arrays as bytes
        return C.reduce(self)

    @method("RBFInterpolator")
    def __init__(self, y, d, /, *, neighbors=None, smoothing=0.0, kernel="thin_plate_spline", epsilon=None,
                 degree=None):  # fmt: skip
        name = "RBFInterpolator"
        sd = _sd
        yy = _coords(y, name, "y")
        n, m = yy.shape
        dd, rest = _values(d, n, name, "d")
        C.finite(dd, name, "d")
        kern = C.choice(kernel, name, "kernel", KERNELS)
        sm = _smoothing(smoothing, n, name)
        if epsilon is None:
            if kern not in SCALE_INVARIANT:
                raise ValueError(
                    f"{name}(): 'epsilon' must be given for kernel '{kern}' (the default 1 is only for the "
                    "scale-invariant linear, thin_plate_spline, cubic and quintic)"
                )
            eps = 1.0
        else:
            eps = C.real(epsilon, name, "epsilon")
            if not (math.isfinite(eps) and eps > 0):
                raise ValueError(f"{name}(): 'epsilon' must be finite and positive, not {eps!r}")
        lowest = MIN_DEGREE.get(kern, -1)
        if degree is None:
            deg = max(lowest, 0)
        else:
            deg = C.integer(degree, name, "degree")
            if deg < lowest:
                if lowest == -1:
                    raise ValueError(f"{name}(): 'degree' must be at least -1, not {deg}")
                raise ValueError(
                    f"{name}(): 'degree' must be at least {lowest} for kernel '{kern}' (the system may not be "
                    f"solvable below it), not {deg}"
                )
        k = None
        if neighbors is not None:
            k = C.integer(neighbors, name, "neighbors")
            if k < 1:
                raise ValueError(f"{name}(): 'neighbors' must be at least 1, not {k}")
            k = min(k, n)
        nobs = n if k is None else k
        q = math.comb(deg + m, m) if deg >= 0 else 0
        if q > nobs:
            what = "data points" if k is None else "neighbors"
            raise ValueError(
                f"{name}(): at least {q} {what} are required for 'degree' {deg} in {m} dimensions, not {nobs}"
            )
        i, j = _core.rbf_duplicate(yy, sm)
        if j >= 0:
            from ..linalg import LinAlgError

            raise LinAlgError(
                f"{name}(): data points {i} and {j} are equal and their smoothing is 0: the system is singular"
            )
        self._y, self._d, self._rest, self._smoothing = yy, dd, rest, sm
        self._kernel, self._kid, self._epsilon, self._degree = kern, KERNELS.index(kern), eps, deg
        self._powers = _readonly(monomial_powers(m, deg))
        self._k, self._tree, self._coeffs, self._shift, self._scale = k, None, None, None, None
        N = nobs + q  # the system's order: global, or each neighbourhood's
        if 8 * N * N > sd._physical_memory():
            raise MemoryError(
                f"{name}(): cannot allocate the {N} x {N} system: more than this machine's physical memory"
            )
        if k is None or k >= n:  # one system for every point (neighbors >= n: the same set everywhere)
            coef, shift, scale, status, rank = _core.rbf_fit(yy, dd, sm, self._kid, eps, self._powers)
            _check_status(status, rank, q, deg, name, "the data points")
            self._coeffs, self._shift, self._scale = _readonly(coef), _readonly(shift), _readonly(scale)
        else:
            from ..spatial import KDTree

            self._tree = KDTree(yy)

    def __repr__(self):
        n, m = self._y.shape
        nb = "" if self._k is None else f", neighbors={self._k}"
        return f"RBFInterpolator(points={n}, dimensions={m}, kernel='{self._kernel}'{nb})"

    @call
    def __call__(self, x, /):
        """The interpolant's value at the points x (..., m)."""
        name = "RBFInterpolator"
        m = self._y.shape[1]
        X, batch = _eval_points(x, m, name)
        if self._coeffs is not None:
            out = _core.rbf_eval(X, self._y, self._coeffs, self._kid, self._epsilon, self._powers, self._shift,
                                 self._scale)  # fmt: skip
            return _finish(out, batch, self._rest)
        Xf, ok = _finite_rows(X)
        if Xf.shape[0] == 0:
            out_f = _sd.empty((0, self._d.shape[1]), dtype=self._d.dtype)
        else:
            idx = self._tree.query(Xf, k=self._k)[1]
            idx = C.contiguous(_sd.reshape(idx, (Xf.shape[0], self._k)))
            out_f, status, point, rank = _core.rbf_local(
                Xf, self._y, self._d, self._smoothing, self._kid, self._epsilon, self._powers, idx
            )
            if status:
                where = point if ok is None else int(_sd.nonzero(ok)[0][point])
                q = self._powers.shape[0]
                at = f"the {self._k} neighbors of point {where} of 'x'"
                _check_status(status, rank, q, self._degree, name, at)
        return _finish(_scatter(out_f, ok, X.shape[0], self._d.dtype), batch, self._rest)


def _smoothing(v, n, name):
    """The smoothing as an (n,) C-contiguous float64 array: a number or an (n,) array, finite and 0 or more."""
    sd = _sd
    if isinstance(v, bool) or type(v).__name__ == "bool":
        raise TypeError(f"{name}(): 'smoothing' must be a number or an array, not {C.type_name(v)}")
    if isinstance(v, (int, float)) or C._is_numpy_scalar(v):
        s = C.real(v, name, "smoothing")
        if not (math.isfinite(s) and s >= 0):
            raise ValueError(f"{name}(): 'smoothing' must be finite and >= 0, not {s!r}")
        return _readonly(sd.full((n,), s))
    a = C.array(v, name, "smoothing")
    if tuple(a.shape) != (n,):
        raise ValueError(f"{name}(): 'smoothing' must be a number or an array of shape ({n},), not {tuple(a.shape)}")
    a = C.contiguous(a.copy())
    if not (_core.all_finite(a) and bool(sd.all(a >= 0))):
        raise ValueError(f"{name}(): 'smoothing' must be finite and >= 0")
    return _readonly(a)


def _check_status(status, rank, q, degree, name, at):
    """LinAlgError for a failed solve of the system of the points `at` (rbf.cpp's status: 1 the tail's rank, 2 an
    exact zero pivot, 3 a coefficient not finite)."""
    if status == 0:
        return
    from ..linalg import LinAlgError

    if status == 1:
        raise LinAlgError(
            f"{name}(): the monomials of degree {degree} at {at} do not have full column rank ({max(rank, 0)}/{q}): "
            "the system is singular"
        )
    raise LinAlgError(f"{name}(): the system of {at} is singular")


# ---------------------------------------------------------------- NearestNDInterpolator


class NearestNDInterpolator:
    """The value of the nearest data point (ties to the smaller index)."""

    __slots__ = ("_x", "_y", "_rest", "_tree", "_offset", "_scale")

    def __reduce__(self):  # sd.ndarray does not pickle: the slots, arrays as bytes
        return C.reduce(self)

    @method("NearestNDInterpolator")
    def __init__(self, x, y, /, *, rescale=False):
        self._build(x, y, rescale, "NearestNDInterpolator")

    def _build(self, x, y, rescale, name):
        pts = _coords(point_set(x, name, "x"), name, "x")
        n = pts.shape[0]
        self._y, self._rest = _values(y, n, name, "y")
        if rescale is not True and rescale is not False:
            raise TypeError(f"{name}(): 'rescale' must be True or False, not {C.type_name(rescale)}")
        self._x = pts
        self._offset = self._scale = None
        if rescale:
            self._offset, self._scale, pts = rescaling(pts)
        from ..spatial import KDTree

        self._tree = KDTree(pts)

    def __repr__(self):
        n, m = self._x.shape
        return f"NearestNDInterpolator(points={n}, dimensions={m})"

    @call
    def __call__(self, xi, /):
        """The value of the nearest data point at each point of xi ((..., m), or a tuple of m coordinate arrays)."""
        return self._evaluate(xi, "NearestNDInterpolator")

    def _evaluate(self, xi, name):
        sd = _sd
        X, batch = xi_points(xi, self._x.shape[1], name)
        Xf, ok = _finite_rows(X)
        if self._offset is not None and Xf.shape[0]:
            Xf = C.contiguous((Xf - self._offset) / self._scale)
        if Xf.shape[0] == 0:
            out_f = sd.empty((0, self._y.shape[1]), dtype=self._y.dtype)
        else:
            idx = sd.reshape(self._tree.query(Xf, k=1)[1], (Xf.shape[0],))
            out_f = C.contiguous(self._y[idx])
        return _finish(_scatter(out_f, ok, X.shape[0], self._y.dtype), batch, self._rest)


# ---------------------------------------------------------------- shared with the triangulation's interpolants


def point_set(x, name, arg):
    """The data points as given, or a tuple of m 1-D coordinate arrays of one length stacked as (n, m) columns (scipy's
    form for NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator and griddata)."""
    if not isinstance(x, tuple):
        return x
    sd = _sd
    cols = [C.array(v, name, arg) for v in x]
    if not cols or any(c.ndim != 1 for c in cols) or len({c.shape[0] for c in cols}) != 1:
        raise ValueError(f"{name}(): the coordinate arrays of a tuple '{arg}' must be 1-D and of one length")
    return C.contiguous(sd.stack(cols, axis=1))


def rescaling(pts):
    """scipy's rescale: (offset, scale, (pts - offset) / scale) with the mean and the peak-to-peak range of each axis
    (a zero range is 1); the mean is the array's pairwise sum."""
    sd = _sd
    off = sd.mean(pts, axis=0)
    span = sd.max(pts, axis=0) - sd.min(pts, axis=0)
    offset = _readonly(C.contiguous(off))
    scale = _readonly(C.contiguous(sd.where(span > 0, span, 1.0)))
    return offset, scale, C.contiguous((pts - offset) / scale)


def xi_points(xi, m, name):
    """Evaluation points (..., m) or a tuple of m coordinate arrays or numbers (broadcast) as (a C-contiguous (Q, m)
    float64 array, the batch shape); a tuple of numbers is one point (batch ())."""
    sd = _sd
    if not isinstance(xi, tuple):
        return _eval_points(xi, m, name, "xi")
    if len(xi) != m:
        raise ValueError(f"{name}(): a tuple 'xi' must have {m} coordinate arrays, not {len(xi)}")
    parts = [C.array(v, name, "xi") if not isinstance(v, (int, float)) else sd.asarray([float(v)])
             for v in xi]  # fmt: skip
    scalar = all(isinstance(v, (int, float)) or C._is_numpy_scalar(v) for v in xi)
    try:
        parts = sd.broadcast_arrays(*parts)
    except ValueError:
        raise ValueError(f"{name}(): the arrays of a tuple 'xi' must broadcast together") from None
    stacked = sd.stack([C.contiguous(p) for p in parts], axis=-1)
    X, batch = _eval_points(stacked, m, name, "xi")
    return X, () if scalar else batch
