"""FITPACK's smoothing splines: make_splrep, make_splprep and generate_knots (docs/api/smooth.md §3,
docs/design/smooth.md §2.3, §3.3; decisions SM12, SM61-SM66).

The spline of degree k with sum (w_i (y_i - g(x_i)))^2 <= s whose k-th derivative jumps least, scipy 1.18.1's
_fitpack_repro rules: the knots grow from none where the residuals are largest until the sum is within s
(generate_knots), then the weight of the jumps is the root of fp(p) = s (rational steps, then bisection where
scipy warns). s = 0 is sd.interpolate.make_interp_spline, the same bits. The C++ core is
csrc/smooth/fitpack.cpp.
"""

import math

from .. import _core as _sd
from .._calls import function
from ..interpolate import BSpline, make_interp_spline
from ._common import array, choice, contiguous, integer, real, vector

_core = _sd.smooth
_TOL = 0.001
_BC = ("not-a-knot", "periodic")


def _linalg_error(msg):
    from ..linalg import LinAlgError

    return LinAlgError(msg)


def _bc(bc_type, func):
    if bc_type is None:
        return False
    return choice(bc_type, func, "bc_type", _BC) == "periodic"


def _s(s, func):
    v = real(s, func, "s")
    if not (math.isfinite(v) and v >= 0):
        raise ValueError(f"{func}(): 's' must be a finite number >= 0, not {v!r}")
    return v


def _k(k, func):
    kk = integer(k, func, "k")
    if not 1 <= kk <= 5:
        raise ValueError(f"{func}(): 'k' must be in [1, 5] (FITPACK's degrees), not {kk}")
    return kk


def _nest(nest, func, k):
    if nest is None:
        return None
    v = integer(nest, func, "nest")
    if v < 2 * (k + 1):
        raise ValueError(f"{func}(): 'nest' must be at least 2(k + 1) = {2 * (k + 1)}, not {v}")
    return v


def _finite(a, func, arg):
    if bool(_sd.any(_sd.isnan(a))):
        raise ValueError(f"{func}(): '{arg}' contains NaN")
    if bool(_sd.any(_sd.isinf(a))):
        raise ValueError(f"{func}(): '{arg}' contains infinity")


def _inputs(func, x, y2, w, k, xb, xe, periodic, xname="x"):
    """The checked FITPACK data: (x, y2 (m, d), w (m,), xb, xe)."""
    m = x.shape[0]
    _finite(x, func, xname)
    _finite(y2, func, "y" if xname == "x" else "x")
    if m and bool(_sd.any(x[1:] < x[:-1])):
        raise ValueError(f"{func}(): '{xname}' must be nondecreasing")
    if m <= k:
        raise ValueError(f"{func}(): {m} point{'s' * (m != 1)} for degree {k}: at least k + 1 = {k + 1} are needed")
    if w is None:
        wv = _sd.ones((m,))
    else:
        wv = vector(w, func, "w")
        if wv.shape[0] != m:
            raise ValueError(f"{func}(): 'w' must have one value per point ({m}), not {wv.shape[0]}")
        _finite(wv, func, "w")
        if bool(_sd.any(wv < 0)):
            raise ValueError(f"{func}(): 'w' must be >= 0")
        if not bool(_sd.any(wv > 0)):
            raise ValueError(f"{func}(): at least one weight must be positive")
    lo, hi = float(x[0]), float(x[m - 1])
    b0 = lo if xb is None else real(xb, func, "xb" if xname == "x" else "ub")
    b1 = hi if xe is None else real(xe, func, "xe" if xname == "x" else "ue")
    if not (math.isfinite(b0) and math.isfinite(b1) and b0 <= lo and hi <= b1 and b0 < b1):
        raise ValueError(f"{func}(): the interval [{b0!r}, {b1!r}] must contain the data [{lo!r}, {hi!r}] and not be "
                         "empty")  # fmt: skip
    if periodic:
        first, last = y2[0], y2[m - 1]
        if not bool(_sd.all(_sd.abs(first - last) <= 1e-15 + 1e-5 * _sd.abs(last))):
            raise ValueError(f"{func}(): the first and last points must match for bc_type='periodic'")
    return x, y2, wv, b0, b1


def _fpcheck(func, x, t, k, periodic):
    """scipy's fpcheck (FITPACK's fpchec and fpchep): the knots t fit the data x."""
    m, n = x.shape[0], t.shape[0]
    nk1 = n - k - 1
    xs, ts = x.tolist(), t.tolist()
    if periodic:
        if not k + 1 <= nk1 <= m + k - 1:
            raise ValueError(f"{func}(): the knots need k + 1 <= len(t) - k - 1 <= m + k - 1 (m = {m}, len(t) = {n})")
    elif not k + 1 <= nk1 <= m:
        raise ValueError(f"{func}(): the knots need k + 1 <= len(t) - k - 1 <= m (m = {m}, len(t) = {n})")
    if any(ts[i] > ts[i + 1] for i in range(k + 1)) or any(ts[i] < ts[i - 1] for i in range(nk1, n)):
        raise ValueError(f"{func}(): the k + 1 knots at each end must be ordered")
    if any(ts[i] <= ts[i - 1] for i in range(k + 1, n - k)):
        raise ValueError(f"{func}(): the interior knots must be distinct and increasing")
    if xs[0] < ts[k] or xs[m - 1] > ts[n - k - 1]:
        raise ValueError(f"{func}(): the data must lie within [t[k], t[-k-1]]")
    mesg = f"{func}(): the Schoenberg-Whitney conditions fail for the knots t and the data x"
    if periodic:
        per = ts[n - k - 1] - ts[k]
        for shift in range(1, m):
            ok = True
            for j in range(k, nk1):
                tj, tl = ts[j], ts[j + k + 1]
                found = False
                for i in range(shift, shift + m):
                    idx = i if i < m else i - m
                    xi = xs[idx] + (0.0 if i < m else per)
                    if tj < xi < tl:
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                return
        raise ValueError(mesg)
    if xs[0] >= ts[k + 1] or xs[m - 1] <= ts[n - k - 2]:
        raise ValueError(mesg)
    l = k + 1
    for j in range(1, n - k - 3 + 1):
        tj = ts[j]
        l += 1
        tl = ts[l]
        i = next((q for q in range(m) if xs[q] > tj), 0)
        if i >= m - 1 or xs[i] >= tl:
            raise ValueError(mesg)


def _interp_knots(x, k, periodic, loop=False):
    """scipy's _not_a_knot or _periodic_knots (generate_knots(s=0) takes the periodic ones only for k > 1)."""
    return _sd.interpolate.bspline_knots(x, k, 2 if periodic and (loop or k > 1) else 0)


def _knot_list(func, x, y2, w, xb, xe, k, s, nest, periodic):
    m = x.shape[0]
    if nest is None:
        nest = max(m + 2 * k, 2 * k + 3) if periodic else max(m + k + 1, 2 * k + 3)
    try:
        knots, interp = _core.fitpack_knots(x, y2, w, xb, xe, k, s, nest, periodic)
    except ValueError:
        raise ValueError(f"{func}(): the weighted sum of squared residuals is NaN (s too small for the data's "
                         "scale)") from None  # fmt: skip
    knots = list(knots)
    if interp:
        knots.append(_interp_knots(x, k, periodic, loop=True))
    return knots


def _fit(func, x, y2, w, xb, xe, k, s, t, periodic):
    """The coefficients (nc, d) of the smoothing spline on the knots t."""
    c, fp, code = _core.fitpack_fit(x, y2, w, xb, xe, k, s, t, periodic)
    if code == 1:
        raise ValueError(f"{func}(): s = {s!r} cannot be reached with these knots: the least-squares spline on them "
                         f"has sum (w r)^2 = {fp!r} (more knots: a larger 'nest', or other 't')")  # fmt: skip
    if code == 2:  # the least-squares polynomial (the constant when periodic) is within s
        return _polynomial(x, y2, w, xb, xe, k, t, periodic)
    if code == 3:
        raise ValueError(f"{func}(): the weighted sum of squared residuals is NaN")
    if code == 4:
        raise _linalg_error(f"{func}(): the least-squares system on the knots is singular")
    if code == 5:
        raise ValueError(f"{func}(): the root of fp(p) = s was not found within 120 steps")
    return c


def _polynomial(x, y2, w, xb, xe, k, t, periodic):
    """The least-squares polynomial of degree k (periodic: the constant) as coefficients on the knots t."""
    nc = t.shape[0] - k - 1
    if periodic:
        m = x.shape[0]
        per = xe - xb
        tc = _sd.asarray([float(x[0]) - (k - i) * per for i in range(k + 1)]
                         + [float(x[m - 1]) + i * per for i in range(k + 1)])  # fmt: skip
        c0, _, _ = _core.fitpack_lsq(x, y2, w, tc, k, True)
        return contiguous(_sd.broadcast_to(c0[0:1], (nc, y2.shape[1])))
    tc = _sd.asarray([xb] * (k + 1) + [xe] * (k + 1))
    c0, _, _ = _core.fitpack_lsq(x, y2, w, tc, k, False)
    sp = BSpline._make(contiguous(tc), contiguous(c0), k, True, 0)
    for v in t[k + 1 : t.shape[0] - k - 1].tolist():
        sp = sp.insert_knot(v)
    return contiguous(sp.c)


def _knots_given(t, func, x, k, periodic):
    tt = array(t, func, "t")
    if tt.ndim != 1:
        raise ValueError(f"{func}(): 't' must be 1-D, not {tt.ndim}-D")
    tt = contiguous(tt.copy())
    _finite(tt, func, "t")
    _fpcheck(func, x, tt, k, periodic)
    return tt


@function
def generate_knots(x, y, /, *, w=None, xb=None, xe=None, k=3, s=0.0, nest=None, bc_type=None):
    """The knot vectors FITPACK's knot placement goes through (scipy's generate_knots, as a list): from no
    interior knot, knots are added where the residuals of the least-squares spline are largest until
    sum (w r)^2 <= s, the knots reach `nest` or the interpolation's knots. y may be (m, d) (a curve)."""
    func = "generate_knots"
    periodic = _bc(bc_type, func)
    kk = _k(k, func)
    sv = _s(s, func)
    xv = vector(x, func, "x")
    if sv == 0:
        if nest is not None or w is not None:
            raise ValueError(f"{func}(): s = 0 is interpolation: 'nest' and 'w' are not used")
        _finite(xv, func, "x")
        return [_interp_knots(xv, kk, periodic)]
    nst = _nest(nest, func, kk)
    ya = array(y, func, "y")
    if ya.ndim not in (1, 2):
        raise ValueError(f"{func}(): 'y' must be 1-D or 2-D, not {ya.ndim}-D")
    if ya.shape[0] != xv.shape[0]:
        raise ValueError(f"{func}(): 'y' must have as many points as 'x' ({xv.shape[0]}), not {ya.shape[0]}")
    y2 = contiguous(_sd.reshape(ya, (ya.shape[0], -1)) if ya.ndim == 2 else _sd.reshape(ya, (ya.shape[0], 1)))
    xv, y2, wv, b0, b1 = _inputs(func, xv, y2, w, kk, xb, xe, periodic)
    return _knot_list(func, xv, y2, wv, b0, b1, kk, sv, nst, periodic)


@function
def make_splrep(x, y, /, *, w=None, xb=None, xe=None, k=3, s=0.0, t=None, nest=None, bc_type=None):
    """The smoothing spline of FITPACK (scipy's make_splrep) as an sd.interpolate.BSpline: the degree-k spline with
    sum (w_i (y_i - g(x_i)))^2 <= s whose k-th derivative jumps least, on the knots of generate_knots (or t).
    s = 0 is make_interp_spline(x, y, k=k) (bc_type='periodic': its periodic spline). A ValueError when s cannot
    be reached (the knots reached `nest`)."""
    func = "make_splrep"
    periodic = _bc(bc_type, func)
    kk = _k(k, func)
    sv = _s(s, func)
    if sv == 0:
        if t is not None or w is not None or nest is not None:
            raise ValueError(f"{func}(): s = 0 is interpolation: 't', 'w' and 'nest' are not used")
        return make_interp_spline(x, y, k=kk, bc_type="periodic" if periodic else None)
    nst = _nest(nest, func, kk)
    if nst is not None and t is not None:
        raise ValueError(f"{func}(): give 't' or 'nest', not both")
    xv = vector(x, func, "x")
    yv = vector(y, func, "y")
    if yv.shape[0] != xv.shape[0]:
        raise ValueError(f"{func}(): 'x' and 'y' must have the same length, not {xv.shape[0]} and {yv.shape[0]}")
    y2 = contiguous(_sd.reshape(yv, (yv.shape[0], 1)))
    xv, y2, wv, b0, b1 = _inputs(func, xv, y2, w, kk, xb, xe, periodic)
    if t is None:
        tt = contiguous(_knot_list(func, xv, y2, wv, b0, b1, kk, sv, nst, periodic)[-1])
    else:
        tt = _knots_given(t, func, xv, kk, periodic)
    c = _fit(func, xv, y2, wv, b0, b1, kk, sv, tt, periodic)
    nc = tt.shape[0] - kk - 1
    return BSpline._make(tt, contiguous(_sd.reshape(c, (nc,))), kk, "periodic" if periodic else True, 0)


@function
def make_splprep(x, /, *, w=None, u=None, ub=None, ue=None, k=3, s=0.0, t=None, nest=None, bc_type=None):
    """The smoothing parametric curve of FITPACK (scipy's make_splprep): x is (d, m) (or a list of d 1-D arrays),
    the curve g(u) with sum_j w_j^2 |g(u_j) - x_j|^2 <= s whose k-th derivative jumps least. u defaults to the
    normalized cumulative chord length. Returns (BSpline with axis=1, u)."""
    func = "make_splprep"
    periodic = _bc(bc_type, func)
    kk = _k(k, func)
    sv = _s(s, func)
    if isinstance(x, (list, tuple)):
        rows = [vector(r, func, "x") for r in x]
        if not rows:
            raise ValueError(f"{func}(): 'x' must have at least one coordinate")
        if len({r.shape[0] for r in rows}) != 1:
            raise ValueError(f"{func}(): the coordinates of 'x' must have the same length")
        X = _sd.stack(rows, axis=1)
    else:
        xa = array(x, func, "x")
        if xa.ndim != 2:
            raise ValueError(f"{func}(): 'x' must be 2-D (d, m), not {xa.ndim}-D")
        X = _sd.permute_dims(xa, (1, 0))
    X = contiguous(X)  # (m, d)
    _finite(X, func, "x")
    m = X.shape[0]
    if u is None:
        if m < 2:
            raise ValueError(f"{func}(): at least 2 points are needed")
        uv = _core.fitpack_chord(X)  # scipy's sums in scipy's order: the same parameters
        if not float(uv[m - 1]) == 1.0:
            raise ValueError(f"{func}(): the points are all equal: no chord length to parametrize by")
    else:
        uv = vector(u, func, "u")
        if uv.shape[0] != m:
            raise ValueError(f"{func}(): 'u' must have one value per point ({m}), not {uv.shape[0]}")
    if sv == 0:
        if t is not None or w is not None or nest is not None:
            raise ValueError(f"{func}(): s = 0 is interpolation: 't', 'w' and 'nest' are not used")
        sp = make_interp_spline(uv, _sd.permute_dims(X, (1, 0)), k=kk, bc_type="periodic" if periodic else None,
                                axis=1)  # fmt: skip
        return sp, uv
    nst = _nest(nest, func, kk)
    if nst is not None and t is not None:
        raise ValueError(f"{func}(): give 't' or 'nest', not both")
    uv, X, wv, b0, b1 = _inputs(func, uv, X, w, kk, ub, ue, periodic, xname="u")
    if t is None:
        tt = contiguous(_knot_list(func, uv, X, wv, b0, b1, kk, sv, nst, periodic)[-1])
    else:
        tt = _knots_given(t, func, uv, kk, periodic)
    c = _fit(func, uv, X, wv, b0, b1, kk, sv, tt, periodic)
    return BSpline._make(tt, contiguous(c), kk, "periodic" if periodic else True, 1), uv


__all__ = ["generate_knots", "make_splprep", "make_splrep"]
