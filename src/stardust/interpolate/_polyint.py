"""The polynomial interpolants of sd.interpolate (docs/api/interpolate.md §5; docs/design/interpolate.md §2.6,
§3.6; milestone IP4): BarycentricInterpolator, KroghInterpolator and their function forms.

A BarycentricInterpolator holds its nodes xi (distinct, finite, any order), the weights wi
(csrc/interpolate/polyint.cpp: the products counted with separate exponents, in the order of the input, no rng;
decision IP21) and the values as an (n, K) array with the axis and the rest of their shape. The derivative of
order k is the barycentric interpolant of D^k y (scipy's recursion), the nodal values cached per order.
A KroghInterpolator holds the Newton coefficients of scipy's Hermite divided-difference table (a repeated node
carries the next derivative); its value is Horner on the Newton form and its derivatives repeated Horner passes.
Neither warns (decision IP22). The values propagate as IEEE does (no finiteness check): a NaN value gives NaN
where it counts.

Results: x's shape at `axis` of the values' shape, as the piecewise polynomials (docs/api/interpolate.md §1.2);
a scalar x without a batch gives a Python scalar. A derivative of order n or more (n nodes) is 0 (NaN at a NaN
point: decision IP52's rule, IP64).
"""

import math

from .. import _core as _sd
from .._calls import call, function, method, methods
from . import _common as C

_core = _sd.interpolate


def _readonly(a):
    a._set_readonly()
    return a


def _finish(out, shape, rest, axis):
    """(q, K) values as x's shape at `axis` of the values' shape; a Python scalar for shape () without a batch."""
    if not shape and not rest:
        return out.reshape((out.size,))[0]
    out = _sd.reshape(out, shape + rest)
    nx = len(shape)
    if axis and nx:
        order = list(range(nx, nx + axis)) + list(range(nx)) + list(range(nx + axis, nx + len(rest)))
        out = _sd.permute_dims(out, tuple(order))
    return out


def _finish_many(out, shape, rest, axis):
    """(count, q, K) values as (count,) + the shape _finish gives."""
    count = out.shape[0]
    out = _sd.reshape(out, (count,) + shape + rest)
    nx = len(shape)
    if axis and nx:
        order = [0] + list(range(nx + 1, nx + 1 + axis)) + list(range(1, nx + 1))
        order += list(range(nx + 1 + axis, nx + 1 + len(rest)))
        out = _sd.permute_dims(out, tuple(order))
    return out


def _nodes(x, name, arg="xi"):
    """1-D finite nodes, at least one: a C-contiguous float64 copy."""
    a = C.array(x, name, arg)
    if a.ndim != 1:
        raise ValueError(f"{name}(): '{arg}' must be 1-D, not {a.ndim}-D")
    if a.shape[0] < 1:
        raise ValueError(f"{name}(): '{arg}' must have at least 1 point")
    a = C.contiguous(a.copy())
    if not _core.all_finite(a):
        raise ValueError(f"{name}(): '{arg}' must be finite")
    return a


def _values(y, n, axis, name, arg="yi"):
    """(values (n, K) C-contiguous, the rest of their shape, the axis in [0, ndim))."""
    sd = _sd
    yy = C.array(y, name, arg, complex_ok=True)
    ax = C.axis_index(axis, yy.ndim, name)
    if yy.shape[ax] != n:
        raise ValueError(
            f"{name}(): '{arg}' must have as many points as 'xi' ({n}) along axis {ax}, not {yy.shape[ax]}"
        )
    if ax:
        yy = sd.moveaxis(yy, ax, 0)
    rest = tuple(yy.shape[1:])
    return _readonly(C.contiguous(sd.reshape(yy, (n, math.prod(rest))).copy())), rest, ax


def _beyond(flat, ncol, dtype):
    """A derivative of order n or more (n nodes) at the points: (q, ncol) zeros, NaN at the NaN points."""
    sd = _sd
    q = flat.shape[0]
    nan = sd.reshape(sd.isnan(flat), (q, 1))
    out = sd.broadcast_to(sd.where(nan, math.nan, 0.0), (q, ncol)).astype(dtype)
    return C.contiguous(out)


def _renamed(e, old, new):
    msg = str(e)
    if msg.startswith(old + "(): "):
        return type(e)(new + "(): " + msg[len(old) + 4 :])
    return e


# ---------------------------------------------------------------- BarycentricInterpolator


@methods
class BarycentricInterpolator:
    """The interpolating polynomial through (xi, yi) in the second barycentric form."""

    __slots__ = ("_xi", "_wi", "_y", "_rest", "_axis", "_nodal")

    def __reduce__(self):  # sd.ndarray does not pickle: the slots, arrays as bytes
        return C.reduce(self)

    @method("BarycentricInterpolator")
    def __init__(self, xi, yi=None, /, *, axis=0, wi=None):
        name = "BarycentricInterpolator"
        x = _nodes(xi, name)
        n = x.shape[0]
        if wi is None:
            w = _weights(x, name)
        else:
            w = C.array(wi, name, "wi")
            if tuple(w.shape) != (n,):
                raise ValueError(f"{name}(): 'wi' must have one weight per node ({n},), not {tuple(w.shape)}")
            _distinct(x, name)
            w = _readonly(C.contiguous(w.copy()))
        self._xi = _readonly(x)
        self._wi = w
        self._nodal = []
        if yi is None:
            self._y, self._rest = None, ()
            self._axis = C.integer(axis, name, "axis")
        else:
            self._y, self._rest, self._axis = _values(yi, n, axis, name)

    # ---- attributes ----

    @property
    def xi(self):
        """The nodes (n,), read-only."""
        return self._xi

    @property
    def wi(self):
        """The barycentric weights (n,), read-only: a common factor of the true ones, the largest in [0.5, 1)."""
        return self._wi

    @property
    def yi(self):
        """The values in their own shape (the nodes along `axis`), read-only; None before set_yi."""
        if self._y is None:
            return None
        y = _sd.reshape(self._y, (self._y.shape[0],) + self._rest)
        if self._axis:
            y = _sd.moveaxis(y, 0, self._axis)
        return _readonly(C.contiguous(y.copy()))

    def __repr__(self):
        return f"BarycentricInterpolator(nodes={self._xi.shape[0]})"

    # ---- evaluation ----

    def _need_values(self, name):
        if self._y is None:
            raise ValueError(f"{name}(): no values: give them with set_yi()")

    def _eval(self, v, flat):
        return _core.bary_eval(self._xi, self._wi, v, flat, False)

    @call
    def __call__(self, x, /):
        """The polynomial's value at x."""
        name = "BarycentricInterpolator"
        self._need_values(name)
        flat, shape = C.points(x, name)
        return _finish(self._eval(self._y, flat), shape, self._rest, self._axis)

    def _derivative_values(self, k, flat):
        """(q, K) values of the derivative of order k at the points."""
        n = self._xi.shape[0]
        if k == 0:
            return self._eval(self._y, flat)
        if k >= n:
            return _beyond(flat, self._y.shape[1], self._y.dtype)
        while len(self._nodal) < k:
            prev = self._y if not self._nodal else self._nodal[-1]
            self._nodal.append(_readonly(_core.bary_node_derivative(self._xi, self._wi, prev)))
        return self._eval(self._nodal[k - 1], flat)

    def derivative(self, x, /, *, der=1):
        """The derivative of order der at x (der = 0: the value)."""
        name = "derivative"
        k = C.order(der, name, "der")
        self._need_values(name)
        flat, shape = C.points(x, name)
        return _finish(self._derivative_values(k, flat), shape, self._rest, self._axis)

    def derivatives(self, x, /, *, der=None):
        """The derivatives of orders 0 .. der - 1 at x (der None: n of them), stacked on a first axis."""
        name = "derivatives"
        count = self._xi.shape[0] if der is None else C.order(der, name, "der")
        self._need_values(name)
        flat, shape = C.points(x, name)
        rows = [self._derivative_values(k, flat) for k in range(count)]
        q, K = flat.shape[0], self._y.shape[1]
        out = _sd.stack(rows, axis=0) if rows else _sd.zeros((0, q, K), dtype=self._y.dtype)
        return _finish_many(out, shape, self._rest, self._axis)

    # ---- changing the object (scipy's add_xi and set_yi: decision IP12) ----

    def add_xi(self, xi, yi=None, /):
        """Adds nodes (and their values, when the object has values); the weights are those of all the nodes, as a
        new object's (decision IP62)."""
        name = "add_xi"
        sd = _sd
        new = _nodes(xi, name)
        if yi is not None:
            if self._y is None:
                raise ValueError(f"{name}(): the object has no values to add to: give them with set_yi() first")
            yy = C.array(yi, name, "yi", complex_ok=True)
            want = self._rest[: self._axis] + (new.shape[0],) + self._rest[self._axis :]
            if tuple(yy.shape) != want:
                raise ValueError(f"{name}(): 'yi' must have the shape {want}, not {tuple(yy.shape)}")
            if self._axis:
                yy = sd.moveaxis(yy, self._axis, 0)
            y2 = sd.reshape(yy, (new.shape[0], self._y.shape[1]))
            if y2.dtype != self._y.dtype:
                y2 = y2.astype(sd.complex128)
            y = sd.concat([self._y.astype(y2.dtype), y2], axis=0)
        elif self._y is not None:
            raise ValueError(f"{name}(): 'yi' is needed: the object has values")
        else:
            y = None
        x = C.contiguous(sd.concat([self._xi, new], axis=0))
        w = _weights(x, name)
        self._xi = _readonly(x)
        self._wi = w
        if y is not None:
            self._y = _readonly(C.contiguous(y))
        self._nodal = []

    def set_yi(self, yi, /, *, axis=None):
        """Replaces the values (axis None: the object's axis)."""
        name = "set_yi"
        if yi is None:
            self._y, self._rest = None, ()
        else:
            ax = self._axis if axis is None else axis
            self._y, self._rest, self._axis = _values(yi, self._xi.shape[0], ax, name)
        self._nodal = []


def _weights(x, name):
    w, dup = _core.bary_weights(x)
    if dup >= 0:
        raise ValueError(f"{name}(): 'xi' must be distinct: xi[{dup}] = {float(x[dup])!r} repeats an earlier node")
    return _readonly(w)


def _distinct(x, name):
    sd = _sd
    s = sd.sort(x)
    if s.shape[0] > 1 and bool(sd.any(s[1:] == s[:-1])):
        raise ValueError(f"{name}(): 'xi' must be distinct")


# ---------------------------------------------------------------- KroghInterpolator


@methods
class KroghInterpolator:
    """The (Hermite) interpolating polynomial in Newton form: a repeated node carries the next derivative."""

    __slots__ = ("_xi", "_c", "_rest", "_axis")

    def __reduce__(self):  # sd.ndarray does not pickle: the slots, arrays as bytes
        return C.reduce(self)

    @method("KroghInterpolator")
    def __init__(self, xi, yi, /, *, axis=0):
        name = "KroghInterpolator"
        x = _nodes(xi, name)
        y, rest, ax = _values(yi, x.shape[0], axis, name)
        c, bad = _core.krogh_table(x, y)
        if bad >= 0:
            raise ValueError(f"{name}(): a repeated value of 'xi' must follow its equals: xi[{bad}] = "
                             f"{float(x[bad])!r} repeats an earlier node that is not next to it")  # fmt: skip
        self._xi = _readonly(x)
        self._c = _readonly(c)
        self._rest = rest
        self._axis = ax

    def __repr__(self):
        return f"KroghInterpolator(nodes={self._xi.shape[0]})"

    def _orders(self, flat, first, count):
        return _core.krogh_eval(self._xi, self._c, flat, first, count)

    @call
    def __call__(self, x, /):
        """The polynomial's value at x."""
        flat, shape = C.points(x, "KroghInterpolator")
        return _finish(self._orders(flat, 0, 1)[0], shape, self._rest, self._axis)

    def derivative(self, x, /, *, der=1):
        """The derivative of order der at x (der = 0: the value)."""
        name = "derivative"
        k = C.order(der, name, "der")
        flat, shape = C.points(x, name)
        return _finish(self._orders(flat, k, 1)[0], shape, self._rest, self._axis)

    def derivatives(self, x, /, *, der=None):
        """The derivatives of orders 0 .. der - 1 at x (der None: n of them), stacked on a first axis."""
        name = "derivatives"
        count = self._xi.shape[0] if der is None else C.order(der, name, "der")
        flat, shape = C.points(x, name)
        return _finish_many(self._orders(flat, 0, count), shape, self._rest, self._axis)


# ---------------------------------------------------------------- function forms


def _orders_of(der, name):
    if isinstance(der, (list, tuple)):
        return [C.order(d, name, "der") for d in der]
    return C.order(der, name, "der")


def _call(cls, name, xi, yi, x, der, axis):
    orders = _orders_of(der, name)
    try:
        p = cls(xi, yi, axis=axis)
    except (TypeError, ValueError) as e:
        raise _renamed(e, cls.__name__, name) from None
    try:
        if isinstance(orders, list):
            return [p.derivative(x, der=d) for d in orders]
        return p(x) if orders == 0 else p.derivative(x, der=orders)
    except (TypeError, ValueError) as e:
        e2 = _renamed(_renamed(e, cls.__name__, name), "derivative", name)
        raise e2 from None


@function
def barycentric_interpolate(xi, yi, x, /, *, axis=0, der=0):
    """BarycentricInterpolator(xi, yi, axis=axis) at x: its value (der = 0), its derivative of order der, or the
    list of those for a sequence der."""
    return _call(BarycentricInterpolator, "barycentric_interpolate", xi, yi, x, der, axis)


@function
def krogh_interpolate(xi, yi, x, /, *, der=0, axis=0):
    """KroghInterpolator(xi, yi, axis=axis) at x: its value (der = 0), its derivative of order der, or the list of
    those for a sequence der."""
    return _call(KroghInterpolator, "krogh_interpolate", xi, yi, x, der, axis)
