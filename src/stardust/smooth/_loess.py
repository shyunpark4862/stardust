"""Local polynomial regression: loess (docs/api/smooth.md §2, docs/design/smooth.md §2.2, §3.2; decisions SM10, SM16,
SM28, SM29, SM95-SM111).

loess(X, y) fits, at every data point, a weighted polynomial of degree 0, 1 or 2 in the p variables from its
q = floor(n span + 1e-5) nearest points with tricube weights (R's loess computed directly, surface='direct'), and
with family='symmetric' reweights the points by R's bisquare of the residuals for `iterations` fits. The C++ core
(csrc/smooth/loess.cpp; one variable and degree 1 on lowess's kernel csrc/smooth/local.hpp) scales the columns,
sorts the points by (coordinates, y, weight, index), so the result is a function of the data and not of their order,
and computes every point with the same kernel whatever the thread count.
"""

import math
import numbers

from .. import _calls
from .. import _core as _sd
from .._calls import function
from ._common import alpha as _alpha
from ._common import choice, contiguous, integer, points, real, type_name
from ._kreg import _matrix, _regression_data
from ._results import SmootherResult

_core = _sd.smooth
_FAMILIES = ("gaussian", "symmetric")
TWO_DELTA_MAX = 4096  # delta2 needs the dense smoother matrix: computed up to this many points (decision SM7)


def _scalar_or(a, shape):
    if shape == ():
        return a.reshape((a.size,))[0]
    return _sd.reshape(a, shape)


def _terms(p, degree):
    return 1 if degree == 0 else p + 1 if degree == 1 else (p + 1) * (p + 2) // 2


class LoessResult(SmootherResult):
    """The result of loess(): fittedvalues, resid, nobs, df, leverage, scale, span, q, degree, family, enp,
    one_delta, two_delta, robust_weights, n_rank_deficient; predict(X_new) and predict_interval(X_new, alpha=0.05)
    evaluate the same local polynomial at new points with the last robust weights. Not a public constructor
    (docs/api/smooth.md §7)."""

    _TITLE = "Loess Results"
    _METHOD = "LOESS"
    _TUNING = (
        ("Span:", "span"),
        ("Window points:", "q"),
        ("Degree:", "degree"),
        ("Family:", "family"),
        ("Equivalent parameters:", "enp"),
        ("Rank-deficient fits:", "n_rank_deficient"),
    )
    _REPR = ("nobs", "df", "span", "degree")

    def __init__(self, data, p, one_d, **fields):
        super().__init__(**fields)
        object.__setattr__(self, "_data", data)  # (Xs, ys, ws, rho, divisor, widen) sorted, or None (propagate)
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "_one_d", one_d)

    def _points(self, X_new, func):
        """(an (m, p) C-contiguous float64 array, the result's shape)."""
        p = object.__getattribute__(self, "_p")
        if object.__getattribute__(self, "_one_d"):
            pts, shape = points(X_new, func, "x_new")
            return contiguous(_sd.reshape(pts, (pts.shape[0], 1))), shape
        if isinstance(X_new, numbers.Number):
            raise ValueError(f"{func}(): 'X_new' must have {p} columns, not a number")
        a = _matrix(X_new, func, "X_new")
        if a.ndim == 1:
            if a.shape[0] != p:
                raise ValueError(f"{func}(): 'X_new' must have {p} values (one point) or be (m, {p}), not {a.shape[0]}")
            return contiguous(_sd.reshape(a, (1, p))), ()
        if a.ndim != 2 or a.shape[1] != p:
            raise ValueError(f"{func}(): 'X_new' must be (m, {p}), not {tuple(a.shape)}")
        return contiguous(a), (a.shape[0],)

    def _evaluate(self, X_new, func, norms):
        q, shape = self._points(X_new, func)
        data = object.__getattribute__(self, "_data")
        if data is None:
            nan = _sd.full((q.shape[0],), math.nan)
            return nan, (nan.copy() if norms else None), shape
        Xs, ys, ws, rho, divisor, widen = data
        values, norm2 = _core.loess_predict(Xs, ys, ws, rho, self.degree, self.q, widen, divisor, q, norms, -1)
        return values, norm2, shape

    def predict(self, X_new, /):
        """The local polynomial at new points with the last robust weights: one variable given as 1-D takes any
        shape (a number gives a float), several variables (m, p) or one point (p,). NaN where no point of the
        window has a positive weight, and at points with a NaN."""
        values, _, shape = self._evaluate(X_new, "predict", False)
        return _scalar_or(values, shape)

    def predict_interval(self, X_new, /, *, alpha=0.05):
        """(..., 2): the pointwise 1 - alpha interval of the mean, value -+ t scale ||l||, with ||l||^2 =
        sum l_j^2 / w_j of the hat row at the point and t the Student t quantile with one_delta^2 / two_delta
        degrees of freedom (R's), the normal quantile when two_delta is not computed (more than 4096 points). The
        bias is not in it; the robust weights are taken as fixed (docs/api/smooth.md §1.3)."""
        a = _alpha(alpha, "predict_interval")
        values, norm2, shape = self._evaluate(X_new, "predict_interval", True)
        d1, d2 = self.one_delta, self.two_delta
        if math.isnan(d1):
            q = math.nan
        elif math.isnan(d2) and self.nobs > TWO_DELTA_MAX:
            from ..stats import Normal

            q = float(Normal().iccdf(a / 2))
        else:
            dof = d1 * d1 / d2 if d2 > 0 else math.nan
            if dof > 0 and math.isfinite(dof):
                from ..stats import StudentT

                q = float(StudentT(df=dof).iccdf(a / 2))
            else:
                q = math.nan
        half = q * self.scale * _sd.sqrt(norm2)
        out = _sd.stack([values - half, values + half], axis=-1)
        return _sd.reshape(out, shape + (2,))


def _span(v):
    s = real(v, "loess", "span")
    if not (s > 0 and math.isfinite(s)):
        raise ValueError("loess(): 'span' must be a finite number > 0")
    return s


def _back(values, keep, n):
    if keep is None:
        return values
    out = _sd.full((n,), math.nan)
    out[keep] = values
    return out


@function
def loess(
    X, y, /, *, span=0.75, degree=2, family="gaussian", iterations=4, normalize=True, weights=None,
    nan_policy="propagate"
):  # fmt: skip
    """Local polynomial regression (Cleveland, Grosse and Shyu's loess; R's estimator, computed directly at every
    point) of y on X.

    The window of a point x0 is its q = floor(n span + 1e-5) nearest points (Euclidean distance) and its radius h
    the q-th distance; span > 1 takes every point and the radius sqrt(span) times the farthest distance (R's
    computation). The weights are tricube (1 - (d / h)^3)^3 times the robust weights and `weights`; the fit is the
    weighted polynomial of the degree (0, 1 or 2) in the p variables, its value at x0. family='symmetric' fits
    `iterations` times, reweighting by R's bisquare of the residuals over six times their median absolute value.
    normalize=True divides each of several variables by its 10% trimmed standard deviation (R's). X is (n,) or
    (n, p). Returns a LoessResult whose arrays are in the input order."""
    span = _span(span)
    degree = integer(degree, "loess", "degree")
    if degree not in (0, 1, 2):
        raise ValueError("loess(): 'degree' must be 0, 1 or 2")
    family = choice(family, "loess", "family", _FAMILIES)
    iterations = integer(iterations, "loess", "iterations")
    if iterations < 1:
        raise ValueError("loess(): 'iterations' must be >= 1")
    if not isinstance(normalize, bool):
        raise TypeError(f"loess(): 'normalize' must be a bool, not {type_name(normalize)}")
    Xa, yv, w, keep, has_nan, one_d = _regression_data("loess", X, y, weights, nan_policy)
    n_in = keep.shape[0] if keep is not None else Xa.shape[0]
    n, p = Xa.shape
    q = min(n, int(math.floor(n * span + 1e-5)))
    t = _terms(p, degree)
    if q < t:
        raise ValueError(
            f"loess(): 'span' is too small: the window has {q} points, fewer than the {t} terms of the local "
            f"polynomial of degree {degree} in {p} variable{'s' if p > 1 else ''}"
        )
    widen = math.sqrt(span) if span > 1 else 1.0
    fits = iterations if family == "symmetric" else 1
    nobs = n if w is None else int(_sd.count_nonzero(w > 0))
    common = {"span": span, "q": q, "degree": degree, "family": family}
    if has_nan:
        nan = _sd.full((n_in,), math.nan)
        return LoessResult(
            None,
            p,
            one_d,
            fittedvalues=nan,
            resid=nan.copy(),
            nobs=n,
            df=math.nan,
            leverage=nan.copy(),
            scale=math.nan,
            **common,
            enp=math.nan,
            one_delta=math.nan,
            two_delta=math.nan,
            robust_weights=nan.copy(),
            n_rank_deficient=0,
        )
    ranges = [float(r) for r in (_sd.max(Xa, axis=0) - _sd.min(Xa, axis=0)).tolist()]
    top = max(ranges)
    if not math.isfinite(top) or (top > 0 and not math.isfinite(top * math.sqrt(sum((r / top) ** 2 for r in ranges)))):
        raise ValueError("loess(): the range of 'X' overflows")
    out = _core.loess_fit(Xa, yv, w, degree, q, widen, fits, normalize, -1)
    if isinstance(out, int):
        raise ValueError(
            "loess(): a variable has no spread in its middle 80% (normalize=True divides by the 10% trimmed "
            "standard deviation); give normalize=False or drop the variable"
        )
    fitted, leverage, rho, df, rss, delta1, deficient, Xs, ys, ws, rhos, divisor = out
    scale = math.sqrt(rss / delta1) if delta1 > 0 else math.nan
    enp = delta1 + 2.0 * df - nobs
    data = (Xs, ys, ws, rhos, divisor, widen)

    def two_delta():
        if nobs > TWO_DELTA_MAX:
            return math.nan
        return float(_core.loess_two_delta(Xs, ys, ws, rhos, degree, q, widen, -1))

    resid = yv - fitted
    return LoessResult(
        data,
        p,
        one_d,
        fittedvalues=_back(fitted, keep, n_in),
        resid=_back(resid, keep, n_in),
        nobs=nobs,
        df=df,
        leverage=_back(leverage, keep, n_in),
        scale=scale,
        **common,
        enp=enp,
        one_delta=delta1,
        two_delta=two_delta,
        robust_weights=_back(rho, keep, n_in),
        n_rank_deficient=int(deficient),
    )


_calls.methods(LoessResult)
