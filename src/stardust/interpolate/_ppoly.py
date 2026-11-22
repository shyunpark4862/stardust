"""PPoly and the cubic family of sd.interpolate (docs/api/interpolate.md §3; docs/design/interpolate.md §2.4,
§3.3, §3.5; milestone IP2).

A PPoly holds read-only coefficients c (k + 1, m, ...), highest degree first, piece i a polynomial in the local
variable s = x - x[i], and read-only breakpoints x (m + 1,), strictly increasing or decreasing (decisions IP12,
IP43). CubicHermiteSpline, CubicSpline, PchipInterpolator and Akima1DInterpolator are its subclasses: each
finds the node slopes its own way (csrc/interpolate/cubic.cpp) and every one builds the coefficients with the
same Hermite routine (IP15, IP42), so equal slopes give equal coefficients, bit for bit. A coefficient that is
not finite is a ValueError naming the interval (IP36).

Every call is one native call on C-contiguous data: evaluation, the antiderivative's constants, the integral
and the roots are in csrc/interpolate/ppoly.cpp; the derivative is a product of sd arrays here.
"""

import math
import numbers

from .. import _core as _sd
from .._calls import function, method
from . import _common as C

_core = _sd.interpolate

_KINDS = {"not-a-knot": 0, "periodic": 3}
_BC_NAMES = ("not-a-knot", "clamped", "natural", "periodic")


def _readonly(a):
    a._set_readonly()
    return a


def _poch(a, n):
    """a (a + 1) ... (a + n - 1) as a float (the rising factorial of the derivative's factors)."""
    v = 1
    for t in range(n):
        v *= a + t
    return float(v)


def _scalar_or(a, shape):
    """The reshaped result, a Python scalar when its shape is ()."""
    if shape == ():
        return a.reshape((a.size,))[0]
    return _sd.reshape(a, shape)


class PPoly:
    """A piecewise polynomial: coefficients c (k + 1, m, ...) in the local power basis of the breakpoints x."""

    __slots__ = ("_c", "_x", "_extrapolate", "_axis", "_ascending")

    @method("PPoly")
    def __init__(self, c, x, /, *, extrapolate=None, axis=0):
        name = "PPoly"
        sd = _sd
        ext = C.extrapolate(extrapolate, name)
        cc = C.array(c, name, "c", complex_ok=True)
        if cc.ndim < 2:
            raise ValueError(f"{name}(): 'c' must have at least 2 dimensions, not {cc.ndim}")
        ax = C.axis_index(axis, cc.ndim - 1, name)
        if ax != 0:  # c (..., k + 1, m, ...) with k + 1 at axis: the two axes to the front (scipy)
            cc = sd.moveaxis(cc, ax + 1, 0)
            cc = sd.moveaxis(cc, ax + 1, 0)
        xx, asc = C.nodes(x, name, decreasing_ok=True)
        if cc.shape[0] == 0:
            raise ValueError(f"{name}(): 'c' must have at least one coefficient per piece")
        if cc.shape[1] != xx.shape[0] - 1:
            raise ValueError(
                f"{name}(): 'c' must have one piece per interval of 'x' ({xx.shape[0] - 1}) along axis 1, not "
                f"{cc.shape[1]}"
            )
        self._init(C.contiguous(cc.copy()), xx, True if ext is None else ext, ax, asc)

    def _init(self, c, x, extrapolate, axis, ascending):
        self._c = _readonly(c)
        self._x = x
        self._extrapolate = extrapolate
        self._axis = axis
        self._ascending = ascending

    @classmethod
    def _make(cls, c, x, extrapolate, axis, ascending):
        """An object of this class from checked parts (scipy's construct_fast, kept private: IP12)."""
        self = cls.__new__(cls)
        PPoly._init(self, c, x, extrapolate, axis, ascending)
        return self

    # ---- attributes ----

    @property
    def c(self):
        """The coefficients (k + 1, m, ...), highest degree first, read-only."""
        return self._c

    @property
    def x(self):
        """The breakpoints (m + 1,), read-only."""
        return self._x

    @property
    def extrapolate(self):
        """True, False or 'periodic' (docs/api/interpolate.md §1.3)."""
        return self._extrapolate

    @property
    def axis(self):
        """The axis of the values that the evaluation points go to."""
        return self._axis

    def __repr__(self):
        k, m = self._c.shape[0] - 1, self._c.shape[1]
        return f"{type(self).__name__}(degree={k}, pieces={m}, extrapolate={self._extrapolate!r})"

    def _c3(self):
        c = self._c
        return _sd.reshape(c, (c.shape[0], c.shape[1], math.prod(c.shape[2:])))

    def _batch(self):
        return tuple(self._c.shape[2:])

    def _ext(self, extrapolate, name):
        e = C.extrapolate(extrapolate, name)
        return self._extrapolate if e is None else e

    # ---- evaluation ----

    def __call__(self, x, /, *, nu=0, extrapolate=None):
        """The value (or the derivative of order nu) at x: x's shape at `axis` of the values' shape."""
        name = type(self).__name__
        k = C.order(nu, name)
        ext = self._ext(extrapolate, name)
        flat, shape = C.points(x, name)
        out = _core.ppoly_eval(self._c3(), self._x, flat, k, C.RULE[ext])
        batch = self._batch()
        if not shape and not batch:
            return out.reshape((out.size,))[0]
        out = _sd.reshape(out, shape + batch)
        nx, ax = len(shape), self._axis
        if ax and nx:
            order = list(range(nx, nx + ax)) + list(range(nx)) + list(range(nx + ax, nx + len(batch)))
            out = _sd.permute_dims(out, tuple(order))
        return out

    # ---- derivatives and antiderivatives ----

    def derivative(self, *, nu=1):
        """The derivative of order nu as a piecewise polynomial of the same class (nu < 0: the antiderivative)."""
        k = C.integer(nu, "derivative", "nu")
        if k < 0:
            return self.antiderivative(nu=-k)
        sd = _sd
        c = self._c
        k1 = c.shape[0]
        if k == 0:
            c2 = c
        elif k >= k1:
            c2 = sd.zeros((1,) + tuple(c.shape[1:]), dtype=c.dtype)
        else:
            f = sd.asarray([_poch(a, k) for a in range(k1 - k, 0, -1)])
            c2 = C.contiguous(c[: k1 - k] * sd.reshape(f, (k1 - k,) + (1,) * (c.ndim - 1)))
        return type(self)._make(_readonly(c2) if c2 is not c else c, self._x, self._extrapolate, self._axis,
                                self._ascending)  # fmt: skip

    def antiderivative(self, *, nu=1):
        """The antiderivative of order nu, zero at x[0] and continuous (nu <= 0: the derivative); a 'periodic'
        object's antiderivative does not extrapolate (scipy)."""
        k = C.integer(nu, "antiderivative", "nu")
        if k <= 0:
            return self.derivative(nu=-k)
        c3 = _core.ppoly_antiderivative(self._c3(), self._x, k)
        c = _sd.reshape(c3, (c3.shape[0], c3.shape[1]) + self._batch())
        ext = False if self._extrapolate == "periodic" else self._extrapolate
        return type(self)._make(_readonly(c), self._x, ext, self._axis, self._ascending)

    # ---- integral ----

    def integrate(self, a, b, /, *, extrapolate=None):
        """The definite integral from a to b (a > b changes the sign), one value per column of the batch."""
        name = "integrate"
        lo, hi = C.real(a, name, "a"), C.real(b, name, "b")
        ext = self._ext(extrapolate, name)
        sign = 1.0
        if hi < lo:
            lo, hi, sign = hi, lo, -1.0
        c3, x = self._c3(), self._x
        if ext == "periodic" and lo == lo and hi == hi:
            # whole periods and the rest, as scipy (on [min(x), max(x)], for either order of the breakpoints)
            x0, x1 = float(x[0]), float(x[x.shape[0] - 1])
            xs, xe = min(x0, x1), max(x0, x1)
            period = xe - xs
            n_periods, left = divmod(hi - lo, period)
            if n_periods > 0:
                total = _core.ppoly_integrate(c3, x, xs, xe, False) * n_periods
            else:
                total = _sd.zeros((c3.shape[2],), dtype=c3.dtype)
            lo = xs + (lo - xs) % period
            hi = lo + left
            if hi <= xe:
                total = total + _core.ppoly_integrate(c3, x, lo, hi, False)
            else:
                total = total + _core.ppoly_integrate(c3, x, lo, xe, False)
                total = total + _core.ppoly_integrate(c3, x, xs, xs + left + lo - xe, False)
            out = total
        else:
            out = _core.ppoly_integrate(c3, x, lo, hi, bool(ext))
        if sign < 0:
            out = -out
        return _scalar_or(out, self._batch())

    # ---- roots ----

    def solve(self, y=0.0, /, *, discontinuity=True, extrapolate=None):
        """The real x where the function equals y, sorted (a piece equal to y gives [x[i], nan]); a batch
        gives a nested list of the batch's shape."""
        return self._solve(y, discontinuity, extrapolate, "solve")

    def roots(self, *, discontinuity=True, extrapolate=None):
        """The real roots: solve(0.0)."""
        return self._solve(0.0, discontinuity, extrapolate, "roots")

    def _solve(self, y, discontinuity, extrapolate, name):
        yv = C.real(y, name, "y")
        if not math.isfinite(yv):
            raise ValueError(f"{name}(): 'y' must be finite")
        if discontinuity is not True and discontinuity is not False:
            raise TypeError(f"{name}(): 'discontinuity' must be True or False, not {C.type_name(discontinuity)}")
        ext = self._ext(extrapolate, name)
        if self._c.dtype == _sd.complex128:
            raise ValueError(f"{name}(): root finding is only for real-valued polynomials")
        found = _core.ppoly_roots(self._c3(), self._x, yv, discontinuity, bool(ext))
        batch = self._batch()
        if not batch:
            return found[0]
        return _nest(found, batch)

    # ---- from a B-spline ----

    @classmethod
    def from_spline(cls, tck, /, *, extrapolate=None):
        """The B-spline tck (a BSpline or a (t, c, k) tuple) as a piecewise polynomial of the same function: the
        breakpoints are the distinct knots of the base interval [t[k], t[n]], so the two agree under every outside
        rule (decision IP54); piece j's coefficients are the derivatives at its left breakpoint over d!."""
        name = "from_spline"
        from ._bspline import BSpline

        ext = C.extrapolate(extrapolate, name)
        if isinstance(tck, BSpline):
            spl = tck
        elif isinstance(tck, (tuple, list)) and len(tck) == 3:
            try:
                spl = BSpline(*tck)
            except (TypeError, ValueError) as e:
                msg = str(e)
                if msg.startswith("BSpline(): "):
                    raise type(e)(name + "(): " + msg[len("BSpline(): ") :]) from None
                raise
        else:
            raise TypeError(f"{name}(): 'tck' must be a BSpline or a (t, c, k) tuple, not {C.type_name(tck)}")
        if ext is None:
            ext = spl.extrapolate if isinstance(tck, BSpline) else True
        cp, x = _core.bspline_to_ppoly(spl.t, spl._c2(), spl.k)
        x._set_readonly()
        c = _readonly(_sd.reshape(cp, (cp.shape[0], cp.shape[1]) + spl._batch()))
        return cls._make(c, x, ext, spl.axis, True)

    # ---- extend ----

    def extend(self, c, x, /):
        """Adds pieces on the right (or on the left): c (k' + 1, m', ...) over the breakpoints x (m',) that continue
        self.x; the object changes (scipy's one modifying method, IP12)."""
        name = "extend"
        sd = _sd
        cc = C.array(c, name, "c", complex_ok=True)
        xx = C.array(x, name, "x")
        if cc.ndim < 2:
            raise ValueError(f"{name}(): 'c' must have at least 2 dimensions, not {cc.ndim}")
        if xx.ndim != 1:
            raise ValueError(f"{name}(): 'x' must be 1-D, not {xx.ndim}-D")
        if xx.shape[0] != cc.shape[1]:
            raise ValueError(f"{name}(): 'x' must have one point per piece of 'c' ({cc.shape[1]}), not {xx.shape[0]}")
        if cc.ndim != self._c.ndim or tuple(cc.shape[2:]) != self._batch():
            raise ValueError(f"{name}(): 'c' must have the shape (k + 1, m) + {self._batch()}, not {tuple(cc.shape)}")
        if cc.size == 0:
            return
        if not _core.all_finite(C.contiguous(xx)):
            raise ValueError(f"{name}(): 'x' must be finite")
        old = self._x
        n0 = old.shape[0]
        first, last = float(old[0]), float(old[n0 - 1])
        xs = [float(v) for v in xx.tolist()]
        up = self._ascending
        step_ok = all((b > a) if up else (b < a) for a, b in zip(xs, xs[1:]))
        if not step_ok:
            raise ValueError(f"{name}(): 'x' must be strictly {'increasing' if up else 'decreasing'}, as self.x")
        if (xs[0] > last) if up else (xs[0] < last):
            action = "append"
        elif (xs[-1] < first) if up else (xs[-1] > first):
            action = "prepend"
        else:
            raise ValueError(f"{name}(): 'x' must lie beyond either end of self.x, without touching it")
        k2 = max(cc.shape[0], self._c.shape[0])
        dtype = sd.complex128 if sd.complex128 in (cc.dtype, self._c.dtype) else sd.float64
        m0, m1 = self._c.shape[1], cc.shape[1]
        c2 = sd.zeros((k2, m0 + m1) + self._batch(), dtype=dtype)
        if action == "append":
            c2[k2 - self._c.shape[0] :, :m0] = self._c
            c2[k2 - cc.shape[0] :, m0:] = cc
            x2 = sd.concat([old, xx])
        else:
            c2[k2 - cc.shape[0] :, :m1] = cc
            c2[k2 - self._c.shape[0] :, m1:] = self._c
            x2 = sd.concat([xx, old])
        x2 = C.contiguous(x2.copy())
        x2._set_readonly()
        self._c = _readonly(c2)
        self._x = x2


def _nest(found, batch):
    """The flat list of per-column root arrays as nested lists of the batch's shape."""
    if len(batch) == 1:
        return list(found)
    step = math.prod(batch[1:])
    return [_nest(found[i * step : (i + 1) * step], batch[1:]) for i in range(batch[0])]


# ---------------------------------------------------------------- the cubic family


def _prepare(x, y, axis, name, *, complex_ok=True):
    """(nodes, values (n, K) C-contiguous, the values' shape without the axis, the axis) of the cubic family."""
    sd = _sd
    xx, _ = C.nodes(x, name)
    yy = C.array(y, name, "y", complex_ok=complex_ok)
    ax = C.axis_index(axis, yy.ndim, name)
    n = xx.shape[0]
    if yy.shape[ax] != n:
        raise ValueError(f"{name}(): 'y' must have as many points as 'x' ({n}) along axis {ax}, not {yy.shape[ax]}")
    if ax:
        yy = sd.moveaxis(yy, ax, 0)
    rest = tuple(yy.shape[1:])
    y2 = C.contiguous(sd.reshape(yy, (n, math.prod(rest))))
    C.finite(y2, name, "y")
    return xx, y2, rest, ax


def _coefficients(cls_name, x, y2, s2, rest):
    c, bad = _core.hermite(x, y2, s2)
    if bad >= 0:
        raise ValueError(
            f"{cls_name}(): the coefficients overflow on interval {bad}, [x[{bad}], x[{bad + 1}]] = "
            f"[{float(x[bad])!r}, {float(x[bad + 1])!r}]"
        )
    return _readonly(_sd.reshape(c, (4, c.shape[1]) + rest))


class CubicHermiteSpline(PPoly):
    """The piecewise cubic through (x, y) with the derivatives dydx at the nodes."""

    __slots__ = ()

    @method("CubicHermiteSpline")
    def __init__(self, x, y, dydx, /, *, axis=0, extrapolate=None):
        name = "CubicHermiteSpline"
        sd = _sd
        ext = C.extrapolate(extrapolate, name)
        xx, y2, rest, ax = _prepare(x, y, axis, name)
        d = C.array(dydx, name, "dydx", complex_ok=True)
        yshape = rest[:ax] + (xx.shape[0],) + rest[ax:]
        if tuple(d.shape) != yshape:
            raise ValueError(f"{name}(): 'dydx' must have the shape of 'y' {yshape}, not {tuple(d.shape)}")
        if ax:
            d = sd.moveaxis(d, ax, 0)
        d2 = C.contiguous(sd.reshape(d, (xx.shape[0], y2.shape[1])))
        C.finite(d2, name, "dydx")
        if d2.dtype != y2.dtype:
            y2, d2 = y2.astype(sd.complex128), d2.astype(sd.complex128)
        c = _coefficients(name, xx, y2, d2, rest)
        self._init(c, xx, True if ext is None else ext, ax, True)


def _bc(v, name, rest, dtype):
    """One end's condition: (kind, value or None, complex?)."""
    sd = _sd
    if isinstance(v, str):
        if v == "clamped":
            return 1, sd.zeros((math.prod(rest),), dtype=dtype), False
        if v == "natural":
            return 2, sd.zeros((math.prod(rest),), dtype=dtype), False
        if v in _KINDS:
            return _KINDS[v], None, False
        raise ValueError(
            f"{name}(): 'bc_type' must be 'not-a-knot', 'clamped', 'natural', 'periodic' or (order, value) pairs, "
            f"not {v!r}"
        )
    if not isinstance(v, (tuple, list)) or len(v) != 2:
        raise ValueError(f"{name}(): a boundary condition must be a name or an (order, value) pair")
    order, value = v
    k = C.integer(order, name, "bc_type")
    if k not in (1, 2):
        raise ValueError(f"{name}(): the derivative order of a boundary condition must be 1 or 2, not {k}")
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        arr = sd.asarray([value]) if not rest else None
        if arr is None:
            raise ValueError(f"{name}(): a boundary value must have the shape {rest}, not ()")
    else:
        arr = C.array(value, name, "bc_type", complex_ok=True)
        if tuple(arr.shape) != rest:
            raise ValueError(f"{name}(): a boundary value must have the shape {rest}, not {tuple(arr.shape)}")
        arr = sd.reshape(arr, (math.prod(rest),))
    if arr.dtype != sd.complex128 and arr.dtype != sd.float64:
        arr = arr.astype(sd.float64)
    C.finite(arr, name, "bc_type")
    return k, arr, arr.dtype == sd.complex128


def _periodic_ok(y2):
    """numpy's allclose(y[0], y[-1], rtol=1e-15, atol=1e-15) over the columns."""
    first, last = y2[0], y2[y2.shape[0] - 1]
    diff = _sd.abs(first - last)
    return bool(_sd.all(diff <= 1e-15 + 1e-15 * _sd.abs(last)))


class CubicSpline(PPoly):
    """The cubic spline through (x, y), twice continuously differentiable, with the end conditions bc_type."""

    __slots__ = ()

    @method("CubicSpline")
    def __init__(self, x, y, /, *, axis=0, bc_type="not-a-knot", extrapolate=None):
        name = "CubicSpline"
        sd = _sd
        ext = C.extrapolate(extrapolate, name)
        xx, y2, rest, ax = _prepare(x, y, axis, name)
        if isinstance(bc_type, str):
            if bc_type == "periodic" and not _periodic_ok(y2):
                raise ValueError(
                    f"{name}(): 'y' must have equal first and last values along axis {ax} for bc_type='periodic' "
                    "(within rtol = atol = 1e-15)"
                )
            ends = (bc_type, bc_type)
        elif isinstance(bc_type, (tuple, list)) and len(bc_type) == 2:
            if any(isinstance(b, str) and b == "periodic" for b in bc_type):
                raise ValueError(f"{name}(): bc_type='periodic' applies to both ends at once")
            ends = tuple(bc_type)
        elif isinstance(bc_type, (tuple, list)):
            raise ValueError(f"{name}(): 'bc_type' must be one condition or a pair of them, not {len(bc_type)}")
        else:
            raise TypeError(f"{name}(): 'bc_type' must be a str or a pair of conditions, not {C.type_name(bc_type)}")
        conds = [_bc(b, name, rest, y2.dtype) for b in ends]
        if any(cpx for _, _, cpx in conds) and y2.dtype != sd.complex128:
            y2 = y2.astype(sd.complex128)
        vals = []
        for _, v, _ in conds:
            if v is not None and v.dtype != y2.dtype:
                v = v.astype(y2.dtype)
            vals.append(None if v is None else C.contiguous(v))
        s, bad = _core.spline_slopes(xx, y2, conds[0][0], vals[0], conds[1][0], vals[1])
        if bad >= 0:
            raise ValueError(
                f"{name}(): the system for the slopes is singular in floating point (pivot {bad}): the spacings at "
                "that end differ by a factor of about 2^50 or more"
            )
        c = _coefficients(name, xx, y2, s, rest)
        if ext is None:
            ext = "periodic" if conds[0][0] == 3 else True
        self._init(c, xx, ext, ax, True)


class PchipInterpolator(PPoly):
    """The shape-preserving piecewise cubic (Fritsch-Butland slopes): monotone on monotone data."""

    __slots__ = ()

    @method("PchipInterpolator")
    def __init__(self, x, y, /, *, axis=0, extrapolate=None):
        name = "PchipInterpolator"
        ext = C.extrapolate(extrapolate, name)
        xx, y2, rest, ax = _prepare(x, y, axis, name, complex_ok=True)
        if y2.dtype == _sd.complex128:
            raise ValueError(f"{name}(): 'y' must be real, not complex128")
        s = _core.pchip_slopes(xx, y2)
        self._init(_coefficients(name, xx, y2, s, rest), xx, True if ext is None else ext, ax, True)


class Akima1DInterpolator(PPoly):
    """Akima's piecewise cubic (method='akima', 1970) or the modified Akima (method='makima')."""

    __slots__ = ()

    @method("Akima1DInterpolator")
    def __init__(self, x, y, /, *, axis=0, method="akima", extrapolate=None):
        name = "Akima1DInterpolator"
        meth = C.choice(method, name, "method", ("akima", "makima"))
        ext = C.extrapolate(extrapolate, name)
        xx, y2, rest, ax = _prepare(x, y, axis, name, complex_ok=True)
        if y2.dtype == _sd.complex128:
            raise ValueError(f"{name}(): 'y' must be real, not complex128")
        s = _core.akima_slopes(xx, y2, meth == "makima")
        self._init(_coefficients(name, xx, y2, s, rest), xx, False if ext is None else ext, ax, True)


@function
def pchip_interpolate(xi, yi, x, /, *, der=0, axis=0):
    """PchipInterpolator(xi, yi, axis=axis)(x, nu=der); a sequence `der` gives the list of those values."""
    name = "pchip_interpolate"
    if isinstance(der, (list, tuple)):
        orders = [C.order(d, name, "der") for d in der]
    else:
        orders = None
        k = C.order(der, name, "der")
    try:
        p = PchipInterpolator(xi, yi, axis=axis)
    except (TypeError, ValueError) as e:
        msg = str(e)
        if msg.startswith("PchipInterpolator(): "):
            raise type(e)(name + "(): " + msg[len("PchipInterpolator(): ") :]) from None
        raise
    try:
        if orders is None:
            return p(x, nu=k)
        return [p(x, nu=d) for d in orders]
    except (TypeError, ValueError) as e:
        msg = str(e)
        if msg.startswith("PchipInterpolator(): "):
            raise type(e)(name + "(): " + msg[len("PchipInterpolator(): ") :]) from None
        raise
