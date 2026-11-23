"""Smoothing splines, P-splines and least-squares splines (docs/api/smooth.md §3, docs/design/smooth.md §2.3,
§3.3; decisions SM11, SM13, SM14, SM30, SM50-SM69).

smoothing_spline(x, y) minimizes sum_i w_i (y_i - f(x_i))^2 + lam int f''^2 over the cubic splines with a knot at
every distinct x (the natural cubic spline), pspline(x, y) the same with the d-th differences of the coefficients
of a B-spline on uniform knots as the penalty (Eilers and Marx), make_lsq_spline(x, y, t) the weighted least
squares on given knots. The C++ core (csrc/smooth/spline.cpp, csrc/smooth/bandqr.hpp) merges equal x, solves the
stacked rows by banded Givens QR and gets the leverage from the band of (R'R)^-1; lam = 0 of the smoothing spline
is sd.interpolate.make_interp_spline(bc_type='natural') itself, so the two give the same bits.
"""

import math

from .. import _core as _sd
from .._calls import function
from ..interpolate import BSpline, make_interp_spline
from ._common import alpha as _alpha
from ._common import array, choice, contiguous, integer, points, policy, real, vector
from ._results import SmootherResult

_core = _sd.smooth

_KIND_SMOOTHING, _KIND_PSPLINE, _KIND_LSQ = 0, 1, 2
_MODE_GIVEN, _MODE_INF, _MODE_ZERO, _MODE_CHOOSE, _MODE_DF = 0, 1, 2, 3, 4
_CRITERIA = ("gcv", "cv")


def _linalg_error(msg):
    from ..linalg import LinAlgError

    return LinAlgError(msg)


# ---------------------------------------------------------------- arguments


def _axis(axis, ndim, func):
    ax = integer(axis, func, "axis")
    if not -ndim <= ax < ndim:
        raise ValueError(f"{func}(): axis {ax} is out of bounds for an array of dimension {ndim}")
    return ax % ndim


def _data(func, x, y, w, axis, nan_policy, wname):
    """The checked data: (x (n,), y2 (n, ncol), w (n,) or None, rest, ax, keep, has_nan, n_in).

    nan_policy None (the scipy-named makers): a NaN is a ValueError. keep: the points kept by 'omit' when it
    dropped some; has_nan: 'propagate' met a NaN."""
    pol = None if nan_policy is None else policy(nan_policy, func)
    xv = vector(x, func, "x")
    ya = array(y, func, "y")
    if ya.ndim == 0:
        raise ValueError(f"{func}(): 'y' must have at least 1 dimension")
    ax = _axis(axis, ya.ndim, func)
    n = xv.shape[0]
    if ya.shape[ax] != n:
        raise ValueError(f"{func}(): 'y' must have as many points as 'x' ({n}) along axis {ax}, not {ya.shape[ax]}")
    if ax:
        ya = _sd.moveaxis(ya, ax, 0)
    rest = tuple(ya.shape[1:])
    ncol = math.prod(rest)
    y2 = contiguous(_sd.reshape(ya, (n, ncol)))
    wv = None
    if w is not None:
        wv = vector(w, func, wname)
        if wv.shape[0] != n:
            raise ValueError(f"{func}(): '{wname}' must have one value per point ({n}), not {wv.shape[0]}")
    if n == 0:
        raise ValueError(f"{func}(): 'x' and 'y' must have at least 1 point")
    for arg, a in (("x", xv), ("y", y2)) + (((wname, wv),) if wv is not None else ()):
        if bool(_sd.any(_sd.isinf(a))):
            raise ValueError(f"{func}(): '{arg}' contains infinity")
    if wv is not None and bool(_sd.any(wv < 0)):
        raise ValueError(f"{func}(): '{wname}' must be >= 0")
    bad = _sd.isnan(xv)
    if ncol:
        bad = _sd.logical_or(bad, _sd.any(_sd.isnan(y2), axis=1))
    if wv is not None:
        bad = _sd.logical_or(bad, _sd.isnan(wv))
    keep, has_nan = None, False
    if bool(_sd.any(bad)):
        if pol is None:
            raise ValueError(f"{func}(): the input contains NaN")
        if pol == "raise":
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
        if pol == "propagate":
            has_nan = True
        else:
            keep = _sd.logical_not(bad)
            xv, y2 = contiguous(xv[keep]), contiguous(y2[keep])
            wv = contiguous(wv[keep]) if wv is not None else None
            if xv.shape[0] == 0:
                raise ValueError(f"{func}(): no point is left without NaN (nan_policy='omit')")
    if not has_nan and wv is not None and not bool(_sd.any(wv > 0)):
        raise ValueError(f"{func}(): at least one weight must be positive")
    if not has_nan and xv.shape[0] and not math.isfinite(float(_sd.max(xv)) - float(_sd.min(xv))):
        raise ValueError(f"{func}(): the range of 'x' overflows")
    return xv, y2, wv, rest, ax, keep, has_nan, n


def _lam(lam, func):
    if lam is None:
        return None
    v = real(lam, func, "lam")
    if math.isnan(v) or v < 0:
        raise ValueError(f"{func}(): 'lam' must be >= 0 (inf allowed), not {v!r}")
    return v


def _df(df, func):
    if df is None:
        return None
    v = real(df, func, "df")
    if not math.isfinite(v):
        raise ValueError(f"{func}(): 'df' must be finite, not {v!r}")
    return v


def _ones(n):
    return _sd.ones((n,))


def _distinct_weighted(xv, wv):
    xs = xv if wv is None else contiguous(xv[wv > 0])
    if xs.shape[0] == 0:
        return 0
    s = _sd.sort(xs)
    return 1 + int(_sd.count_nonzero(s[1:] != s[:-1]))


# ---------------------------------------------------------------- layout


def _place(out, shape, rest, axis):
    """Evaluation values (q, ncol) at the points' shape placed at `axis` of the batch shape (BSpline's layout)."""
    if not shape and not rest:
        return out.reshape((out.size,))[0]
    out = _sd.reshape(out, shape + rest)
    nx = len(shape)
    if axis and nx:
        order = list(range(nx, nx + axis)) + list(range(nx)) + list(range(nx + axis, nx + len(rest)))
        out = _sd.permute_dims(out, tuple(order))
    return out


def _columns(v, rest):
    """A per-column statistic (ncol,) as a float (1-D y) or an array of the batch's shape."""
    if not rest:
        return float(v.reshape((v.size,))[0])
    return _sd.reshape(v, rest)


def _scatter_rows(values, keep, n_in):
    """The (n, ncol) rows of the kept points at their places among n_in rows of NaN (nan_policy='omit')."""
    if keep is None:
        return values
    out = _sd.full((n_in,) + tuple(values.shape[1:]), math.nan)
    out[keep] = values
    return out


def _restore(y2, rest, ax):
    """(n, ncol) back to y's layout."""
    out = _sd.reshape(y2, (y2.shape[0],) + rest)
    return _sd.moveaxis(out, 0, ax) if ax else out


def _spline_of(t, c, k, rest, ax):
    nc = t.shape[0] - k - 1
    return BSpline._make(t, contiguous(_sd.reshape(c, (nc,) + rest)), k, True, ax)


# ---------------------------------------------------------------- results


class _PenalizedResult(SmootherResult):
    """The shared part of SmoothingSplineResult and PSplineResult."""

    def __init__(self, hat, rest, spline, **fields):
        super().__init__(**fields)
        object.__setattr__(self, "_hat", hat)  # (t, k, state) or None after a NaN (propagate)
        object.__setattr__(self, "_rest", rest)
        object.__setattr__(self, "_spline", spline)  # a BSpline is callable: not a lazy field of the base

    @property
    def spline(self):
        """The fitted sd.interpolate.BSpline."""
        return object.__getattribute__(self, "_spline")

    def predict(self, x_new, /):
        """The fitted spline at x_new: spline(x_new), the same bits."""
        return self.spline(x_new)

    def predict_interval(self, x_new, /, *, alpha=0.05):
        """(..., 2): the pointwise 1 - alpha interval of the mean, value -+ t scale ||l||, with ||l||^2 =
        sum_j l_j^2 / w_j of the hat row at the point and t the Student t quantile with nobs - df degrees of
        freedom. The bias is not in it (docs/api/smooth.md §1.3)."""
        a = _alpha(alpha, "predict_interval")
        pts, shape = points(x_new, "predict_interval")
        sp = self.spline
        rest = object.__getattribute__(self, "_rest")
        flat = BSpline._make(sp.t, contiguous(_sd.reshape(sp.c, (sp.c.shape[0], math.prod(rest)))), sp.k, True, 0)
        values = flat(pts)
        values = _sd.reshape(values, (pts.shape[0], math.prod(rest)))
        hat = object.__getattribute__(self, "_hat")
        if hat is None:
            norm2 = _sd.full((pts.shape[0],), math.nan)
        else:
            t, k, state = hat
            norm2 = _core.spline_hat(t, k, state, pts)
        dof = self.nobs - self.df
        if dof > 0 and math.isfinite(dof):
            from ..stats import StudentT

            q = float(StudentT(df=dof).iccdf(a / 2))
        else:
            q = math.nan
        scale = self.scale
        if isinstance(scale, _sd.ndarray):
            sc = _sd.reshape(scale, (1, math.prod(rest)))
        else:
            sc = _sd.full((1, 1), scale)
        half = q * sc * _sd.reshape(_sd.sqrt(norm2), (pts.shape[0], 1))
        lo = _place(contiguous(values - half), shape, rest, sp.axis)
        hi = _place(contiguous(values + half), shape, rest, sp.axis)
        if not isinstance(lo, _sd.ndarray):  # a number and 1-D y: (2,)
            return _sd.asarray([lo, hi])
        return _sd.stack([lo, hi], axis=-1)

    def summary(self):
        if isinstance(self.scale, _sd.ndarray):  # a batch: the box shows the first column's scale
            from .. import _summary

            left = [
                ("Method:", self.METHOD),
                ("No. Observations:", _summary.count(self.nobs)),
                ("Df (effective):", _summary.df(self.df)),
                ("Scale:", _summary.compact(self.scale)),
            ]
            right = [(label, self._cell(field)) for label, field in self.TUNING]
            return _summary.render(self.TITLE, header=(left, right))
        return super().summary()


class SmoothingSplineResult(_PenalizedResult):
    """The result of smoothing_spline(): fittedvalues, resid, nobs, df, leverage, scale, spline, lam, gcv, cv,
    criterion, status, converged; predict(x_new) is spline(x_new). Not a public constructor
    (docs/api/smooth.md §7)."""

    TITLE = "Smoothing Spline Results"
    METHOD = "Smoothing spline"
    TUNING = (("Lambda:", "lam"), ("Criterion:", "criterion"), ("GCV:", "gcv"), ("CV:", "cv"),
              ("Converged:", "converged"))  # fmt: skip
    REPR = ("nobs", "df", "lam")


class PSplineResult(_PenalizedResult):
    """The result of pspline(): the fields of SmoothingSplineResult and n_segments, k, diff_order. Not a public
    constructor (docs/api/smooth.md §7)."""

    TITLE = "P-Spline Results"
    METHOD = "P-spline"
    TUNING = (("Lambda:", "lam"), ("Segments:", "n_segments"), ("Degree:", "k"), ("Penalty order:", "diff_order"),
              ("Criterion:", "criterion"), ("GCV:", "gcv"), ("CV:", "cv"), ("Converged:", "converged"))  # fmt: skip
    REPR = ("nobs", "df", "lam", "n_segments")


# ---------------------------------------------------------------- the fits


def _code_error(func, code, kappa=None):
    if code == 5:
        return _linalg_error(f"{func}(): the system is singular to working precision (estimated condition number "
                             f"{kappa:.3g}; a fit takes up to 2^43, and a lam search or df also needs the "
                             "leverages to keep 10 bits): some values of 'x' are far closer together than the "
                             "others (merge them)")  # fmt: skip
    if code == 1:
        return ValueError(f"{func}(): 'x' must have at least 5 distinct values")
    if code == 2:
        return ValueError(f"{func}(): at least 2 distinct values of 'x' must have a positive weight")
    if code == 3:
        return ValueError(f"{func}(): the spacing of 'x' is too small: the penalty's rows overflow")
    return _linalg_error(f"{func}(): the penalized least-squares system is singular")


def _penalized(func, kind, xv, y2, wv, k, d, t, lam, df, criterion, stats):
    """The C++ fit with the mode of (lam, df, criterion): its dict."""
    if lam is not None and df is not None:
        raise ValueError(f"{func}(): give 'lam' or 'df', not both")
    crit = _CRITERIA.index(criterion)
    if df is not None:
        mode, value, target = _MODE_DF, 0.0, df
    elif lam is None:
        mode, value, target = _MODE_CHOOSE, 0.0, 0.0
    elif lam == math.inf:
        mode, value, target = _MODE_INF, 0.0, 0.0
    elif lam == 0:
        mode, value, target = _MODE_ZERO, 0.0, 0.0
    else:
        mode, value, target = _MODE_GIVEN, lam, 0.0
    w = wv if wv is not None else _ones(xv.shape[0])
    return _core.spline_fit(xv, w, y2, kind, k, d, t, mode, value, crit, target, stats)


def _natural_interpolant(func, xv, y2, wv, rest, ax, stats):
    """lam = 0 of the smoothing spline: make_interp_spline(bc_type='natural') through the merged points of
    positive weight; (spline, fields)."""
    w = wv if wv is not None else _ones(xv.shape[0])
    r = _core.spline_fit(xv, w, y2, _KIND_SMOOTHING, 3, 2, None, _MODE_INF, 0.0, 0, 0.0, True)
    if r["code"] != 0:
        raise _code_error(func, r["code"], r.get("kappa"))
    xg, wg, yg = r["xg"], r["wg"], r["yg"]
    pos = wg > 0
    xp, wp, yp = contiguous(xg[pos]), contiguous(wg[pos]), contiguous(yg[pos])
    sp = make_interp_spline(xp, _restore(yp, rest, 0), k=3, bc_type="natural", axis=0)
    spline = BSpline._make(sp.t, sp.c, 3, True, ax)
    if not stats:
        return spline, None
    # the leverage of a point is w_i / W_g (the group's mean), tr L the number of groups
    group_w = _core.spline_merge_weights(xv, w)
    lev = _sd.where(w > 0, w / group_w, 0.0)
    state = _core.natural_hat(sp.t, xp, wp)
    return spline, {"leverage": lev, "trl": float(xp.shape[0]), "hat": (sp.t, 3, state), "nobs": r["nobs"]}


def _finish(cls, func, xv, y2, wv, rest, ax, keep, n_in, spline, lev, trl, nobs, extra, hat):
    """The result object from the fitted spline: the fitted values are spline(x)."""
    fitted2 = _sd.reshape(BSpline._make(spline.t, contiguous(_sd.reshape(spline.c, (spline.c.shape[0], y2.shape[1]))),
                                        spline.k, True, 0)(xv), (xv.shape[0], y2.shape[1]))  # fmt: skip
    resid2 = y2 - fitted2
    w = wv if wv is not None else _ones(xv.shape[0])
    rss = _sd.sum(w[:, None] * resid2 * resid2, axis=0) if y2.shape[1] else _sd.zeros((0,))
    dof = nobs - trl
    scale = _sd.sqrt(rss / dof) if dof > 0 else _sd.full((y2.shape[1],), math.nan)
    lev_out = lev
    if keep is not None:
        lev_out = _sd.full((n_in,), math.nan)
        lev_out[keep] = lev
    return cls(
        hat,
        rest,
        spline,
        fittedvalues=_restore(_scatter_rows(fitted2, keep, n_in), rest, ax),
        resid=_restore(_scatter_rows(resid2, keep, n_in), rest, ax),
        nobs=nobs,
        df=trl,
        leverage=lev_out,
        scale=_columns(scale, rest),
        **extra,
    )


def _nan_result(cls, n_in, y2, rest, ax, k, nobs, extra):
    nan2 = _sd.full((n_in, y2.shape[1]), math.nan)
    t = _sd.asarray([0.0] * (k + 1) + [1.0] * (k + 1))
    spline = _spline_of(t, _sd.full((k + 1, y2.shape[1]), math.nan), k, rest, ax)
    return cls(
        None,
        rest,
        spline,
        fittedvalues=_restore(nan2, rest, ax),
        resid=_restore(nan2.copy(), rest, ax),
        nobs=nobs,
        df=math.nan,
        leverage=_sd.full((n_in,), math.nan),
        scale=_columns(_sd.full((y2.shape[1],), math.nan), rest),
        **extra,
    )


def _criteria_of_interpolant(xv, y2, wv, lev, trl, nobs, spline):
    """GCV and CV of the interpolating spline (lam = 0), from their definitions."""
    w = wv if wv is not None else _ones(xv.shape[0])
    flat = BSpline._make(spline.t, contiguous(_sd.reshape(spline.c, (spline.c.shape[0], y2.shape[1]))), 3, True, 0)
    r = y2 - _sd.reshape(flat(xv), y2.shape)
    rss = float(_sd.sum(w[:, None] * r * r))
    den = nobs - trl
    gcv = nobs * rss / (den * den) if den != 0 else math.nan
    one = 1.0 - lev
    pos = w > 0
    if bool(_sd.any(_sd.logical_and(pos, one == 0))):
        cv = math.nan
    else:
        q = _sd.where(pos, w[:, None] * (r / _sd.where(one == 0, 1.0, one)[:, None]) ** 2, 0.0)
        cv = float(_sd.sum(q)) / nobs
    return gcv, cv


@function
def smoothing_spline(x, y, /, *, lam=None, df=None, criterion="gcv", weights=None, axis=0, nan_policy="propagate"):
    """The cubic smoothing spline of y on x: the minimizer of sum_i w_i (y_i - f(x_i))^2 + lam int f''(x)^2 dx,
    a natural cubic spline with a knot at every distinct x (scipy's knots and scipy's lam). lam=None chooses lam
    by `criterion` ('gcv' or 'cv') on a grid of log(lam) free of the units of x, then Brent's method; df= finds
    the lam with tr L = df. lam = 0 is the natural interpolating spline, lam = inf the weighted least-squares
    line. Equal x are merged. Returns a SmoothingSplineResult whose arrays are in the input order."""
    func = "smoothing_spline"
    lam = _lam(lam, func)
    df = _df(df, func)
    criterion = choice(criterion, func, "criterion", _CRITERIA)
    if lam is not None and df is not None:
        raise ValueError(f"{func}(): give 'lam' or 'df', not both")
    xv, y2, wv, rest, ax, keep, has_nan, n_in = _data(func, x, y, weights, axis, nan_policy, "weights")
    extra_nan = {"lam": math.nan, "gcv": math.nan, "cv": math.nan, "criterion": criterion, "status": 0,
                 "converged": False}  # fmt: skip
    if has_nan:
        return _nan_result(SmoothingSplineResult, n_in, y2, rest, ax, 3, xv.shape[0], extra_nan)
    if df is not None:
        nd = _distinct_weighted(xv, wv)
        if not 2 < df <= nd:
            raise ValueError(f"{func}(): 'df' must be in (2, {nd}] (the distinct x of positive weight), not {df!r}")
        if df == nd:  # tr L = the number of points: the interpolating spline
            lam, df = 0.0, None
    if lam == 0:
        spline, f = _natural_interpolant(func, xv, y2, wv, rest, ax, True)
        gcv, cv = _criteria_of_interpolant(xv, y2, wv, f["leverage"], f["trl"], f["nobs"], spline)
        extra = {"lam": 0.0, "gcv": gcv, "cv": cv, "criterion": criterion, "status": 0, "converged": True}
        return _finish(SmoothingSplineResult, func, xv, y2, wv, rest, ax, keep, n_in, spline, f["leverage"], f["trl"],
                       f["nobs"], extra, f["hat"])  # fmt: skip
    r = _penalized(func, _KIND_SMOOTHING, xv, y2, wv, 3, 2, None, lam, df, criterion, True)
    if r["code"] != 0:
        raise _code_error(func, r["code"], r.get("kappa"))
    spline = _spline_of(r["t"], r["c"], 3, rest, ax)
    converged = bool(r["converged"]) if lam is None else True
    status = int(r["status"]) if lam is None else 0
    extra = {"lam": float(r["lam"]), "gcv": float(r["gcv"]), "cv": float(r["cv"]), "criterion": criterion,
             "status": status, "converged": converged}  # fmt: skip
    return _finish(SmoothingSplineResult, func, xv, y2, wv, rest, ax, keep, n_in, spline, r["leverage"],
                   float(r["trl"]), int(r["nobs"]), extra, (r["t"], 3, r["hat"]))  # fmt: skip


@function
def make_smoothing_spline(x, y, /, *, w=None, lam=None, axis=0):
    """The cubic smoothing spline (scipy's make_smoothing_spline) as an sd.interpolate.BSpline: the minimizer of
    sum_i w_i (y_i - f(x_i))^2 + lam int f''^2, lam=None chosen by GCV (free of the units of x). The knots are
    every distinct x with the ends four times."""
    func = "make_smoothing_spline"
    lam = _lam(lam, func)
    xv, y2, wv, rest, ax, _, _, _ = _data(func, x, y, w, axis, None, "w")
    if lam == 0:
        spline, _ = _natural_interpolant(func, xv, y2, wv, rest, ax, False)
        return spline
    r = _penalized(func, _KIND_SMOOTHING, xv, y2, wv, 3, 2, None, lam, None, "gcv", False)
    if r["code"] != 0:
        raise _code_error(func, r["code"], r.get("kappa"))
    return _spline_of(r["t"], r["c"], 3, rest, ax)


def _segments(v, n, func):
    if v is None:
        return max(1, math.ceil(min(n / 4, 40)))
    s = integer(v, func, "n_segments")
    if s < 1:
        raise ValueError(f"{func}(): 'n_segments' must be >= 1, not {s}")
    return s


def _bound(v, func, arg, default):
    if v is None:
        return default
    b = real(v, func, arg)
    if not math.isfinite(b):
        raise ValueError(f"{func}(): '{arg}' must be finite, not {b!r}")
    return b


@function
def pspline(x, y, /, *, n_segments=None, k=3, diff_order=2, lam=None, df=None, criterion="gcv", xb=None, xe=None,
            weights=None, axis=0, nan_policy="propagate"):  # fmt: skip
    """The P-spline of y on x (Eilers and Marx): the degree-k B-spline on n_segments uniform intervals of
    [xb, xe] minimizing sum_i w_i (y_i - f(x_i))^2 + lam ||D c||^2, D the diff_order-th differences of the
    coefficients. lam, df and criterion as in smoothing_spline; lam = inf is the least squares over the
    coefficients that are polynomials of degree diff_order - 1 in their index (polynomials of that degree in x).
    Returns a PSplineResult."""
    func = "pspline"
    kk = integer(k, func, "k")
    if kk < 0:
        raise ValueError(f"{func}(): 'k' must be >= 0, not {kk}")
    d = integer(diff_order, func, "diff_order")
    if not 1 <= d <= kk + 1:
        raise ValueError(f"{func}(): 'diff_order' must be in [1, k + 1] = [1, {kk + 1}], not {d}")
    lam = _lam(lam, func)
    df = _df(df, func)
    criterion = choice(criterion, func, "criterion", _CRITERIA)
    if lam is not None and df is not None:
        raise ValueError(f"{func}(): give 'lam' or 'df', not both")
    xv, y2, wv, rest, ax, keep, has_nan, n_in = _data(func, x, y, weights, axis, nan_policy, "weights")
    n = xv.shape[0]
    nseg = _segments(n_segments, n, func)
    extra_nan = {"lam": math.nan, "gcv": math.nan, "cv": math.nan, "criterion": criterion, "status": 0,
                 "converged": False, "n_segments": nseg, "k": kk, "diff_order": d}  # fmt: skip
    if has_nan:
        return _nan_result(PSplineResult, n_in, y2, rest, ax, kk, n, extra_nan)
    lo, hi = float(_sd.min(xv)), float(_sd.max(xv))
    b0 = _bound(xb, func, "xb", lo)
    b1 = _bound(xe, func, "xe", hi)
    if not b0 < b1:
        raise ValueError(f"{func}(): the interval [xb, xe] = [{b0!r}, {b1!r}] must not be empty")
    if lo < b0 or hi > b1:
        raise ValueError(f"{func}(): 'x' must lie within [xb, xe] = [{b0!r}, {b1!r}]")
    t = _core.pspline_knots(b0, b1, nseg, kk)
    if not bool(_sd.all(_sd.isfinite(t))):
        raise ValueError(f"{func}(): the knots overflow")
    nc = nseg + kk
    if df is not None:
        top = min(nc, _distinct_weighted(xv, wv))
        if not d < df <= top:
            raise ValueError(f"{func}(): 'df' must be in ({d}, {top}], not {df!r}")
        if df == top == nc:  # every coefficient free: the least-squares spline
            lam, df = 0.0, None
    r = _penalized(func, _KIND_PSPLINE, xv, y2, wv, kk, d, t, lam, df, criterion, True)
    if r["code"] != 0:
        if r["code"] == 4 and lam == 0:
            raise _linalg_error(f"{func}(): lam=0 is the least-squares spline and its system is singular (an "
                                "interval without data: the Schoenberg-Whitney conditions fail)")  # fmt: skip
        raise _code_error(func, r["code"], r.get("kappa"))
    spline = _spline_of(r["t"], r["c"], kk, rest, ax)
    converged = bool(r["converged"]) if lam is None else True
    status = int(r["status"]) if lam is None else 0
    extra = {"lam": float(r["lam"]), "gcv": float(r["gcv"]), "cv": float(r["cv"]), "criterion": criterion,
             "status": status, "converged": converged, "n_segments": nseg, "k": kk, "diff_order": d}  # fmt: skip
    return _finish(PSplineResult, func, xv, y2, wv, rest, ax, keep, n_in, spline, r["leverage"], float(r["trl"]),
                   int(r["nobs"]), extra, (r["t"], kk, r["hat"]))  # fmt: skip


def _knots_arg(t, func, k):
    tt = array(t, func, "t")
    if tt.ndim != 1:
        raise ValueError(f"{func}(): 't' must be 1-D, not {tt.ndim}-D")
    tt = contiguous(tt.copy())
    N = tt.shape[0]
    if N < 2 * k + 2:
        raise ValueError(f"{func}(): 't' must have at least 2k + 2 = {2 * k + 2} knots for degree {k}, not {N}")
    if not bool(_sd.all(_sd.isfinite(tt))):
        raise ValueError(f"{func}(): 't' must be finite")
    if bool(_sd.any(tt[1:] < tt[:-1])):
        raise ValueError(f"{func}(): 't' must be nondecreasing")
    return tt


@function
def make_lsq_spline(x, y, t, /, *, k=3, w=None, axis=0):
    """The weighted least-squares spline of degree k on the knots t (scipy's make_lsq_spline): the minimizer of
    sum_i (w_i (y_i - g(x_i)))^2 (FITPACK's weights, 1 / sigma). A LinAlgError when the Schoenberg-Whitney
    conditions fail (some B-spline has no data point of positive weight)."""
    func = "make_lsq_spline"
    kk = integer(k, func, "k")
    if kk < 0:
        raise ValueError(f"{func}(): 'k' must be >= 0, not {kk}")
    tt = _knots_arg(t, func, kk)
    xv, y2, wv, rest, ax, _, _, _ = _data(func, x, y, w, axis, None, "w")
    N = tt.shape[0]
    lo, hi = float(tt[kk]), float(tt[N - kk - 1] if kk == 0 else tt[N - kk])
    if float(_sd.min(xv)) < lo or float(_sd.max(xv)) > hi:
        raise ValueError(f"{func}(): 'x' must lie within the knots: t[k] <= x <= t[-k]")
    w = wv if wv is not None else _ones(xv.shape[0])
    order = _sd.argsort(xv, stable=True)
    xs, ws = contiguous(xv[order]), contiguous(w[order])
    bad = _core.schoenberg_whitney(xs, ws, tt, kk)
    if bad >= 0:
        raise _linalg_error(f"{func}(): the Schoenberg-Whitney conditions fail: no data point of positive weight in "
                            f"the support of B-spline {bad} (the knots [{float(tt[bad])!r}, "
                            f"{float(tt[bad + kk + 1])!r}])")  # fmt: skip
    r = _core.spline_fit(xv, w, y2, _KIND_LSQ, kk, 0, tt, _MODE_ZERO, 0.0, 0, 0.0, False)
    if r["code"] == 5:
        raise _code_error(func, 5, r.get("kappa"))
    if r["code"] != 0:
        raise _linalg_error(f"{func}(): the least-squares system is singular")
    return _spline_of(tt, r["c"], kk, rest, ax)


__all__ = ["make_lsq_spline", "make_smoothing_spline", "smoothing_spline", "SmoothingSplineResult", "pspline",
           "PSplineResult"]  # fmt: skip
