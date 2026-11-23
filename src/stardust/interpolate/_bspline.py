"""BSpline and make_interp_spline of sd.interpolate (docs/api/interpolate.md §4; docs/design/interpolate.md §2.5,
§3.4; milestone IP3).

A BSpline holds read-only knots t (n + k + 1, nondecreasing, finite), read-only coefficients c (rows, ...) with the
interpolation axis first (rows >= n; only the first n count, scipy's convention) and the degree k. Evaluation is de
Boor's recursion with the interval found by a count of the knots (csrc/interpolate/bspline.cpp: O(log n) per point in
any order); k = 1 evaluates through the linear path of sd.interp (csrc/interpolate/linear.cpp), so that a k = 1
spline and sd.interp give the same bits. make_interp_spline solves the collocation system without pivoting, the
derivative conditions right after their end's interpolation row (decision IP35).
"""

import math

from .. import _core as _sd
from .._calls import call, function, method, methods
from . import _common as C

_core = _sd.interpolate

_KNOTS_NOT_A_KNOT, _KNOTS_AUGMENTED, _KNOTS_PERIODIC = 0, 1, 2


def _readonly(a):
    a._set_readonly()
    return a


def _scalar_or(a, shape):
    if shape == ():
        return a.reshape((a.size,))[0]
    return _sd.reshape(a, shape)


def _place(out, shape, batch, axis):
    """The evaluation result (q, ncol) reshaped to x's shape at `axis` of the batch's shape (scipy's layout)."""
    if not shape and not batch:
        return out.reshape((out.size,))[0]
    out = _sd.reshape(out, shape + batch)
    nx = len(shape)
    if axis and nx:
        order = list(range(nx, nx + axis)) + list(range(nx)) + list(range(nx + axis, nx + len(batch)))
        out = _sd.permute_dims(out, tuple(order))
    return out


def _knots(t, name, k):
    """The knots: a read-only C-contiguous 1-D float64 copy, finite and nondecreasing, at least 2k + 2 of them, with
    t[k] < t[n] (scipy's checks and order)."""
    tt = C.array(t, name, "t")
    if tt.ndim != 1:
        raise ValueError(f"{name}(): 't' must be 1-D, not {tt.ndim}-D")
    N = tt.shape[0]
    if N < 2 * k + 2:
        raise ValueError(f"{name}(): 't' must have at least 2k + 2 = {2 * k + 2} knots for degree {k}, not {N}")
    tt = C.contiguous(tt.copy())
    code = _core.check_knots(tt)
    if code == -1:
        raise ValueError(f"{name}(): 't' must be finite")
    if code == -2:
        raise ValueError(f"{name}(): 't' must be nondecreasing")
    n = N - k - 1
    if not float(tt[k]) < float(tt[n]):
        raise ValueError(f"{name}(): the base interval [t[k], t[n]] = [t[{k}], t[{n}]] must not be empty")
    return _readonly(tt)


def _degree(k, name):
    kk = C.integer(k, name, "k")
    if kk < 0:
        raise ValueError(f"{name}(): 'k' must be >= 0, not {kk}")
    return kk


@methods
class BSpline:
    """A spline of degree k in the B-spline basis of the knots t: sum_j c[j] B_{j,k,t}(x)."""

    __slots__ = ("_t", "_c", "_k", "_extrapolate", "_axis", "_linear")

    def __reduce__(self):  # sd.ndarray does not pickle: the slots, arrays as bytes
        return C.reduce(self)

    @method("BSpline")
    def __init__(self, t, c, k, /, *, extrapolate=True, axis=0):
        name = "BSpline"
        kk = _degree(k, name)
        ext = C.extrapolate(extrapolate, name)
        tt = _knots(t, name, kk)
        cc = C.array(c, name, "c", complex_ok=True)
        ax = C.axis_index(axis, cc.ndim, name)
        if ax:
            cc = _sd.moveaxis(cc, ax, 0)
        n = tt.shape[0] - kk - 1
        if cc.shape[0] < n:
            raise ValueError(
                f"{name}(): 'c' must have at least n = len(t) - k - 1 = {n} coefficients along axis {ax}, not "
                f"{cc.shape[0]}"
            )
        self._init(tt, _readonly(C.contiguous(cc.copy())), kk, True if ext is None else ext, ax)

    def _init(self, t, c, k, extrapolate, axis):
        self._t = t
        self._c = c
        self._k = k
        self._extrapolate = extrapolate
        self._axis = axis
        self._linear = None

    @classmethod
    def _make(cls, t, c, k, extrapolate, axis):
        """An object of this class from checked parts (scipy's construct_fast, kept private: IP12)."""
        self = cls.__new__(cls)
        BSpline._init(self, _readonly(t), _readonly(c), k, extrapolate, axis)
        return self

    # ---- attributes ----

    @property
    def t(self):
        """The knots (n + k + 1,), read-only."""
        return self._t

    @property
    def c(self):
        """The coefficients (rows, ...), the interpolation axis first, read-only."""
        return self._c

    @property
    def k(self):
        """The degree."""
        return self._k

    @property
    def extrapolate(self):
        """True, False or 'periodic' (docs/api/interpolate.md §1.3)."""
        return self._extrapolate

    @property
    def axis(self):
        """The axis of the values that the evaluation points go to."""
        return self._axis

    @property
    def tck(self):
        """(t, c, k)."""
        return self._t, self._c, self._k

    def __repr__(self):
        return f"{type(self).__name__}(degree={self._k}, knots={self._t.shape[0]}, extrapolate={self._extrapolate!r})"

    def _n(self):
        return self._t.shape[0] - self._k - 1

    def _batch(self):
        return tuple(self._c.shape[1:])

    def _c2(self):
        c = self._c
        return C.contiguous(_sd.reshape(c, (c.shape[0], math.prod(c.shape[1:]))))

    def _ext(self, extrapolate, name):
        e = C.extrapolate(extrapolate, name)
        return self._extrapolate if e is None else e

    def _linear_parts(self):
        """k = 1: (the nodes t[1 .. n], the values c[:n]) when the nodes strictly increase (the linear path of
        sd.interp), else False."""
        if self._linear is None:
            n = self._n()
            xp = C.contiguous(self._t[1 : n + 1].copy())
            self._linear = (xp, C.contiguous(self._c2()[:n])) if _core.check_x(xp, False) == 0 else False
        return self._linear

    # ---- evaluation ----

    @call
    def __call__(self, x, /, *, nu=0, extrapolate=None):
        """The value (or the derivative of order nu) at x: x's shape at `axis` of the coefficients' batch shape."""
        name = type(self).__name__
        k = C.order(nu, name)
        ext = self._ext(extrapolate, name)
        flat, shape = C.points(x, name)
        out = self._eval(flat, k, C.RULE[ext])
        return _place(out, shape, self._batch(), self._axis)

    def _eval(self, flat, nu, rule):
        if self._k == 1 and nu == 0:
            lin = self._linear_parts()
            if lin:
                return _core.linear(flat, lin[0], lin[1], rule)
        return _core.bspline_eval(self._t, self._c2(), self._k, flat, nu, rule)

    # ---- derivatives and antiderivatives ----

    def derivative(self, *, nu=1):
        """The derivative of order nu as a BSpline of degree k - nu (nu < 0: the antiderivative); nu > k is an error
        (scipy)."""
        name = "derivative"
        k = C.integer(nu, name, "nu")
        if k < 0:
            return self.antiderivative(nu=-k)
        if k > self._k:
            raise ValueError(f"{name}(): 'nu' must be at most the degree k = {self._k}, not {k}")
        got = _core.bspline_derivative(self._t, self._c2(), self._k, k)
        if got is None:
            raise ValueError(f"{name}(): the spline has repeated internal knots and is not differentiable {k} times")
        t2, c2 = got
        return type(self)._make(t2, _sd.reshape(c2, (c2.shape[0],) + self._batch()), self._k - k, self._extrapolate,
                                self._axis)  # fmt: skip

    def antiderivative(self, *, nu=1):
        """The antiderivative of order nu as a BSpline of degree k + nu (nu <= 0: the derivative); a 'periodic'
        spline's antiderivative does not extrapolate (scipy)."""
        k = C.integer(nu, "antiderivative", "nu")
        if k <= 0:
            return self.derivative(nu=-k)
        t2, c2 = _core.bspline_antiderivative(self._t, self._c2(), self._k, k)
        ext = False if self._extrapolate == "periodic" else self._extrapolate
        return type(self)._make(t2, _sd.reshape(c2, (c2.shape[0],) + self._batch()), self._k + k, ext, self._axis)

    # ---- integral ----

    def integrate(self, a, b, /, *, extrapolate=None):
        """The definite integral from a to b (a > b changes the sign): extrapolate=False integrates over the part of
        [a, b] inside the base interval [t[k], t[n]], 'periodic' wraps whole periods and the rest (scipy)."""
        name = "integrate"
        lo, hi = C.real(a, name, "a"), C.real(b, name, "b")
        ext = self._ext(extrapolate, name)
        batch = self._batch()
        ncol = math.prod(batch)
        if lo != lo or hi != hi:  # a NaN end (IP49)
            nan = _sd.full((ncol,), math.nan, dtype=self._c.dtype)
            return _scalar_or(nan, batch)
        sign = 1.0
        if hi < lo:
            lo, hi, sign = hi, lo, -1.0
        n = self._n()
        ts, te = float(self._t[self._k]), float(self._t[n])
        ta, ca = _core.bspline_antiderivative(self._t, self._c2(), self._k, 1)
        ka = self._k + 1

        def F(points):  # the antiderivative's values, rows per point
            return _core.bspline_eval(ta, ca, ka, _sd.asarray(points), 0, 1)

        if ext == "periodic":
            period = te - ts
            n_periods, left = divmod(hi - lo, period)
            if n_periods > 0:
                v = F([ts, te])
                total = (v[1] - v[0]) * n_periods
            else:
                total = _sd.zeros((ncol,), dtype=self._c.dtype)
            lo = ts + (lo - ts) % period
            hi = lo + left
            if hi <= te:
                v = F([lo, hi])
                total = total + (v[1] - v[0])
            else:
                v = F([lo, te, ts, ts + hi - te])
                total = total + (v[1] - v[0])
                total = total + (v[3] - v[2])
            out = total
        else:
            if ext is False:
                lo, hi = min(max(lo, ts), te), min(max(hi, ts), te)
            v = F([lo, hi])
            out = v[1] - v[0]
        if sign < 0:
            out = -out
        return _scalar_or(out, batch)

    # ---- construction from other forms ----

    @classmethod
    @method("basis_element")
    def basis_element(cls, t, /, *, extrapolate=True):
        """The single B-spline B(x | t[0], ..., t[k+1]) of degree k = len(t) - 2: k more knots at each end (t[0] - 1
        and t[-1] + 1) and the coefficient 1 at position k (scipy)."""
        name = "basis_element"
        ext = C.extrapolate(extrapolate, name)
        tt = C.array(t, name, "t")
        if tt.ndim != 1:
            raise ValueError(f"{name}(): 't' must be 1-D, not {tt.ndim}-D")
        if tt.shape[0] < 2:
            raise ValueError(f"{name}(): 't' must have at least 2 knots, not {tt.shape[0]}")
        k = tt.shape[0] - 2
        first, last = float(tt[0]), float(tt[tt.shape[0] - 1])
        full = _sd.concat([_sd.full((k,), first - 1.0), tt, _sd.full((k,), last + 1.0)])
        tk = _knots(full, name, k)
        c = _sd.zeros((tk.shape[0],))
        c[k] = 1.0
        return cls._make(tk, c, k, True if ext is None else ext, 0)

    @classmethod
    @method("design_matrix")
    def design_matrix(cls, x, t, k, /, *, extrapolate=False):
        """The (len(x), n) sd.sparse.csr_array of the basis functions at x: row i holds the k + 1 functions of x[i]'s
        interval, zeros included (scipy). extrapolate=False refuses points outside [t[k], t[n]]."""
        name = "design_matrix"
        kk = _degree(k, name)
        ext = C.extrapolate(extrapolate, name, none_ok=False)
        xx = C.array(x, name, "x")
        if xx.ndim != 1:
            raise ValueError(f"{name}(): 'x' must be 1-D, not {xx.ndim}-D")
        xx = C.contiguous(xx)
        if not _core.all_finite(xx):
            raise ValueError(f"{name}(): 'x' must be finite")
        tt = _knots(t, name, kk)
        n = tt.shape[0] - kk - 1
        ts, te = float(tt[kk]), float(tt[n])
        if ext == "periodic":
            xx = ts + _sd.remainder(xx - ts, te - ts)
            xx = C.contiguous(xx)
        elif ext is False and xx.size and (float(_sd.min(xx)) < ts or float(_sd.max(xx)) > te):
            raise ValueError(
                f"{name}(): 'x' must lie in the base interval [t[k], t[n]] = [{ts!r}, {te!r}] when extrapolate=False"
            )
        data, indices, indptr = _core.bspline_basis(tt, kk, xx, ext is not False)
        from .. import sparse

        return sparse.csr_array((data, indices, indptr), shape=(xx.shape[0], n))

    def insert_knot(self, x, /, *, m=1):
        """A new BSpline of the same function with the knot x inserted m times (Boehm's algorithm, scipy's insert)."""
        name = "insert_knot"
        xv = C.real(x, name, "x")
        times = C.integer(m, name, "m")
        n = self._n()
        ts, te = float(self._t[self._k]), float(self._t[n])
        if not ts <= xv <= te:
            raise ValueError(f"{name}(): 'x' must lie in the base interval [t[k], t[n]] = [{ts!r}, {te!r}], not {xv!r}")
        if times <= 0:
            raise ValueError(f"{name}(): 'm' must be positive, not {times}")
        got = _core.bspline_insert(self._t, self._c2(), self._k, xv, times, self._extrapolate == "periodic")
        if got is None:
            raise ValueError(
                f"{name}(): a periodic spline needs k = {self._k} internal knots on one side of the new knot"
            )
        t2, c2 = got
        return type(self)._make(t2, _sd.reshape(c2, (c2.shape[0],) + self._batch()), self._k, self._extrapolate,
                                self._axis)  # fmt: skip

    @classmethod
    @method("from_power_basis")
    def from_power_basis(cls, pp, /, *, bc_type="not-a-knot"):
        """The CubicSpline pp in the B-spline basis of the knots its boundary condition bc_type implies (scipy: only
        CubicSpline objects)."""
        name = "from_power_basis"
        from ._ppoly import CubicSpline

        if not isinstance(pp, CubicSpline):
            raise TypeError(f"{name}(): 'pp' must be a CubicSpline, not {C.type_name(pp)}")
        bc = C.choice(bc_type, name, "bc_type", ("not-a-knot", "natural", "clamped", "periodic"))
        x = pp.x
        k = pp.c.shape[0] - 1
        kind = {"not-a-knot": _KNOTS_NOT_A_KNOT, "periodic": _KNOTS_PERIODIC}.get(bc, _KNOTS_AUGMENTED)
        t = _core.bspline_knots(x, k, kind)
        if t.shape[0] < 2 * k + 2:
            raise ValueError(f"{name}(): too few breakpoints ({x.shape[0]}) for bc_type={bc!r}")
        c = _core.bspline_from_power(x, pp._c3(), t, k)
        return cls._make(t, _sd.reshape(c, (c.shape[0],) + pp._batch()), k, pp.extrapolate, pp.axis)


# ---------------------------------------------------------------- make_interp_spline


def _derivative_spec(v, name, rest, dtype):
    """One end's derivative conditions: [(order, value (ncol,))] (None: none)."""
    sd = _sd
    if v is None:
        return []
    ncol = math.prod(rest)
    if isinstance(v, str):
        if v == "clamped":
            return [(1, sd.zeros((ncol,), dtype=dtype))]
        if v == "natural":
            return [(2, sd.zeros((ncol,), dtype=dtype))]
        raise ValueError(
            f"{name}(): 'bc_type' must be None, 'not-a-knot', 'natural', 'clamped', 'periodic' or a pair of "
            f"end conditions, not {v!r}"
        )
    if not isinstance(v, (list, tuple)):
        raise TypeError(
            f"{name}(): an end condition must be None, 'natural', 'clamped' or a list of (order, value) pairs, "
            f"not {C.type_name(v)}"
        )
    out = []
    for pair in v:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name}(): an end condition must be a list of (order, value) pairs")
        order, value = pair
        k = C.integer(order, name, "bc_type")
        if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
            if rest:
                raise ValueError(f"{name}(): a boundary value must have the shape {rest}, not ()")
            arr = sd.asarray([value])
        else:
            arr = C.array(value, name, "bc_type", complex_ok=True)
            if tuple(arr.shape) != rest:
                raise ValueError(f"{name}(): a boundary value must have the shape {rest}, not {tuple(arr.shape)}")
            arr = sd.reshape(arr, (ncol,))
        if arr.dtype != sd.complex128 and arr.dtype != sd.float64:
            arr = arr.astype(sd.float64)
        C.finite(C.contiguous(arr), name, "bc_type")
        out.append((k, arr))
    return out


def _periodic_ok(y2):
    """numpy's allclose(y[0], y[-1], rtol=1e-15, atol=1e-15) over the columns (CubicSpline's test, IP28)."""
    first, last = y2[0], y2[y2.shape[0] - 1]
    return bool(_sd.all(_sd.abs(first - last) <= 1e-15 + 1e-15 * _sd.abs(last)))


def _stack(rows, ncol, dtype):
    sd = _sd
    if not rows:
        return sd.zeros((0, ncol), dtype=dtype)
    return C.contiguous(sd.stack([r if r.dtype == dtype else r.astype(dtype) for r in rows]))


@function
def make_interp_spline(x, y, /, *, k=3, t=None, bc_type=None, axis=0):
    """The interpolating B-spline of degree k through (x, y), with the knots t (scipy's rules by default) and the end
    conditions bc_type."""
    name = "make_interp_spline"
    sd = _sd
    kk = _degree(k, name)
    periodic = isinstance(bc_type, str) and bc_type == "periodic"
    if bc_type is None or (isinstance(bc_type, str) and bc_type in ("not-a-knot", "periodic")):
        ends = (None, None)
    elif isinstance(bc_type, str):
        ends = (bc_type, bc_type)
    elif isinstance(bc_type, (list, tuple)):
        if len(bc_type) != 2:
            raise ValueError(
                f"{name}(): 'bc_type' must be one condition or a pair of end conditions, not {len(bc_type)}"
            )
        ends = tuple(bc_type)
    else:
        raise TypeError(
            f"{name}(): 'bc_type' must be None, a str or a pair of end conditions, not {C.type_name(bc_type)}"
        )
    xx, _ = C.nodes(x, name)
    yy = C.array(y, name, "y", complex_ok=True)
    ax = C.axis_index(axis, yy.ndim, name)
    n = xx.shape[0]
    if yy.shape[ax] != n:
        raise ValueError(f"{name}(): 'y' must have as many points as 'x' ({n}) along axis {ax}, not {yy.shape[ax]}")
    if ax:
        yy = sd.moveaxis(yy, ax, 0)
    rest = tuple(yy.shape[1:])
    ncol = math.prod(rest)
    y2 = C.contiguous(sd.reshape(yy, (n, ncol)))
    C.finite(y2, name, "y")
    left = _derivative_spec(ends[0], name, rest, y2.dtype)
    right = _derivative_spec(ends[1], name, rest, y2.dtype)
    if periodic and ncol and not _periodic_ok(y2):
        raise ValueError(
            f"{name}(): 'y' must have equal first and last values along axis {ax} for bc_type='periodic' "
            "(within rtol = atol = 1e-15)"
        )
    has_derivs = ends != (None, None)
    if kk == 0:
        if t is not None or has_derivs:
            raise ValueError(f"{name}(): k=0 takes neither 't' nor end conditions in 'bc_type'")
        tt = sd.concat([xx, xx[n - 1 : n]])
        return BSpline._make(C.contiguous(tt), _sd.reshape(y2.copy(), (n,) + rest), 0, True, ax)
    if kk == 1 and t is None:
        if has_derivs:
            raise ValueError(f"{name}(): k=1 takes no end conditions in 'bc_type' (unless 't' is given)")
        tt = sd.concat([xx[0:1], xx, xx[n - 1 : n]])
        return BSpline._make(C.contiguous(tt), _sd.reshape(y2.copy(), (n,) + rest), 1, True, ax)
    if periodic and t is not None:
        raise ValueError(f"{name}(): bc_type='periodic' makes its own knots: 't' must be None")
    if t is None:
        kind = _KNOTS_PERIODIC if periodic else (_KNOTS_AUGMENTED if has_derivs else _KNOTS_NOT_A_KNOT)
        tt = _core.bspline_knots(xx, kk, kind)
    else:
        tt = C.array(t, name, "t")
        if tt.ndim != 1:
            raise ValueError(f"{name}(): 't' must be 1-D, not {tt.ndim}-D")
        tt = C.contiguous(tt.copy())
        code = _core.check_knots(tt)
        if code == -1:
            raise ValueError(f"{name}(): 't' must be finite")
        if code == -2:
            raise ValueError(f"{name}(): 't' must be nondecreasing")
    N = tt.shape[0]
    if N < n + kk + 1:
        raise ValueError(f"{name}(): 't' must have at least len(x) + k + 1 = {n + kk + 1} knots, not {N}")
    if float(xx[0]) < float(tt[kk]) or float(xx[n - 1]) > float(tt[N - kk]):
        raise ValueError(f"{name}(): 'x' must lie within the knots: t[k] <= x[0] and x[-1] <= t[-k]")
    nt = N - kk - 1
    if periodic:
        if not ncol:
            c = sd.zeros((nt, 0), dtype=y2.dtype)
        else:
            c, status = _core.bspline_periodic(xx, y2, tt, kk)
            if status < 0:
                _singular(name)
        return BSpline._make(tt, _sd.reshape(c, (nt,) + rest), kk, "periodic", ax)
    for side, conds in (("x[0]", left), ("x[-1]", right)):
        for order, _ in conds:
            if not 0 <= order <= kk:
                raise ValueError(f"{name}(): a derivative order at {side} must be in [0, k] = [0, {kk}], not {order}")
    if nt - n != len(left) + len(right):
        raise ValueError(
            f"{name}(): the knots need {nt - n} derivative condition{'' if nt - n == 1 else 's'} at the ends, not "
            f"{len(left)} + {len(right)}"
        )
    if not ncol:
        return BSpline._make(tt, sd.zeros((nt,) + rest, dtype=y2.dtype), kk, True, ax)
    if any(v.dtype == sd.complex128 for _, v in left + right) and y2.dtype != sd.complex128:
        y2 = y2.astype(sd.complex128)
    left = sorted(left, key=lambda p: p[0])  # in row order (IP35): ascending away from x[0] ...
    right = sorted(right, key=lambda p: -p[0])  # ... and descending toward x[-1]
    # elimination without pivoting where docs/design/interpolate.md §3.4 proves it stable (decision IP53): no
    # derivative condition, the knots given or not (a totally nonnegative matrix, growth 1), and with
    # make_interp_spline's own knots, at each end, the orders 1 .. r (growth 1) or k = 3's second derivative alone
    # (growth below 2); any other set of orders (or knots given with end conditions) pivots
    plain = not left and not right
    stable = plain or (t is None and all(_prefix_or_natural(sorted(p[0] for p in side), kk) for side in (left, right)))
    lo = sd.asarray([p[0] for p in left], dtype=sd.int64)
    ro = sd.asarray([p[0] for p in right], dtype=sd.int64)
    c, status = _core.bspline_interp(xx, y2, tt, kk, lo, _stack([p[1] for p in left], ncol, y2.dtype), ro,
                                     _stack([p[1] for p in right], ncol, y2.dtype), not stable)  # fmt: skip
    if status == -2:
        raise ValueError(
            f"{name}(): the derivatives of the end conditions overflow: the knot spacing at an end is too small "
            f"([x[0], x[1]] = [{float(xx[0])!r}, {float(xx[1])!r}], [x[-2], x[-1]] = [{float(xx[n - 2])!r}, "
            f"{float(xx[n - 1])!r}])"
        )
    if status < 0:
        _singular(name)
    return BSpline._make(tt, _sd.reshape(c, (nt,) + rest), kk, True, ax)


def _prefix_or_natural(orders, k):
    return orders == list(range(1, len(orders) + 1)) or (k == 3 and orders == [2])


def _singular(name):
    from ..linalg import LinAlgError

    raise LinAlgError(f"{name}(): the collocation matrix is singular (the knots and the points fail the "
                      "Schoenberg-Whitney conditions, or the end conditions repeat)")  # fmt: skip
