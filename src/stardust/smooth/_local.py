"""Scatterplot smoothing: lowess (docs/api/smooth.md §2, docs/design/smooth.md §2.2, §3.1; decisions SM8, SM15,
SM27, SM36-SM49).

lowess(x, y) fits a weighted local line at every data point from its k = min(n, max(2, int(frac n + 1e-10)))
nearest neighbours with tricube weights, then reweights the points it robust iterations with bisquare weights
of the residuals (statsmodels' estimator, the reference library). The C++ core (csrc/smooth/local.cpp, the
kernel csrc/smooth/local.hpp) sorts the data by (x, y, weight, index), so the result is a function of the data
and not of their order, and computes every point with the same kernel whatever the thread count.
"""

import math

from .. import _core as _sd
from .._calls import function
from ._common import alpha as _alpha
from ._common import integer, points, real, scatter, scatter_back
from ._results import SmootherResult

_core = _sd.smooth


def _scalar_or(a, shape):
    """The reshaped result, a Python float when its shape is ()."""
    if shape == ():
        return a.reshape((a.size,))[0]
    return _sd.reshape(a, shape)


class LowessResult(SmootherResult):
    """The result of lowess(): fittedvalues, resid, nobs, df, leverage, scale, frac, k, it, delta, robust_weights;
    predict(x_new) and predict_interval(x_new, alpha=0.05) evaluate the same local line at new points with the
    last robust weights. Not a public constructor (docs/api/smooth.md §7)."""

    TITLE = "Lowess Results"
    METHOD = "LOWESS"
    TUNING = (("Frac:", "frac"), ("Window points:", "k"), ("Robust iterations:", "it"), ("Delta:", "delta"))
    REPR = ("nobs", "df", "frac", "it")

    def __init__(self, data, **fields):
        super().__init__(**fields)
        object.__setattr__(self, "_data", data)  # (xs, ys, ws, rho) sorted, or None after a NaN (propagate)

    def _evaluate(self, x_new, func, norms):
        pts, shape = points(x_new, func)
        data = object.__getattribute__(self, "_data")
        if data is None:
            nan = _sd.full((pts.shape[0],), math.nan)
            return nan, (nan if norms else None), shape
        xs, ys, ws, rho = data
        values, norm2 = _core.lowess_predict(xs, ys, ws, rho, self.k, pts, norms)
        return values, norm2, shape

    def predict(self, x_new, /):
        """The local line at the points x_new (any shape; a number gives a float), from the k nearest data points
        with the last robust weights (statsmodels' xvals). NaN where fewer than two points have a positive
        weight, and at NaN points."""
        values, _, shape = self._evaluate(x_new, "predict", False)
        return _scalar_or(values, shape)

    def predict_interval(self, x_new, /, *, alpha=0.05):
        """(..., 2): the pointwise 1 - alpha interval of the mean, value -+ t scale ||l||, with ||l||^2 =
        sum l_j^2 / w_j of the hat row at the point and t the Student t quantile with nobs - df degrees of freedom.
        The bias is not in it; the robust weights are taken as fixed (docs/api/smooth.md §1.3)."""
        a = _alpha(alpha, "predict_interval")
        values, norm2, shape = self._evaluate(x_new, "predict_interval", True)
        dof = self.nobs - self.df
        if dof > 0 and math.isfinite(dof):
            from ..stats import StudentT

            q = float(StudentT(df=dof).iccdf(a / 2))
        else:
            q = math.nan
        half = q * self.scale * _sd.sqrt(norm2)
        out = _sd.stack([values - half, values + half], axis=-1)
        return _sd.reshape(out, shape + (2,))


def _frac(v):
    f = real(v, "lowess", "frac")
    if not 0.0 <= f <= 1.0:
        raise ValueError("lowess(): 'frac' must be in [0, 1]")
    return f


@function
def lowess(x, y, /, *, frac=2 / 3, it=3, delta=0.0, weights=None, nan_policy="propagate"):
    """Locally weighted scatterplot smoothing (Cleveland's lowess, statsmodels' estimator) of y on x.

    The window of a point x0 is its k = min(n, max(2, int(frac n + 1e-10))) nearest points and its radius h the
    k-th distance; the weights are tricube (1 - (|x_j - x0| / h)^3)^3 times the robust weights and `weights`;
    the fit is the weighted local line. `it` robust iterations follow (it + 1 fits): bisquare weights of the
    residuals over six times their median absolute value. delta > 0 fits only points more than delta beyond
    the last fitted one and interpolates linearly in between (0: every point). Returns a LowessResult whose
    arrays are in the input order."""
    frac = _frac(frac)
    it = integer(it, "lowess", "it")
    if it < 0:
        raise ValueError("lowess(): 'it' must be >= 0")
    delta = real(delta, "lowess", "delta")
    if not (math.isfinite(delta) and delta >= 0):
        raise ValueError("lowess(): 'delta' must be a finite number >= 0")
    xv, yv, wv, keep, has_nan = scatter("lowess", x, y, weights, nan_policy)
    n_in = keep.shape[0] if keep is not None else xv.shape[0]
    n = xv.shape[0]
    k = min(n, max(2, int(frac * n + 1e-10)))
    if has_nan:
        nan = _sd.full((n_in,), math.nan)
        return LowessResult(
            None,
            fittedvalues=nan,
            resid=nan.copy(),
            nobs=n,
            df=math.nan,
            leverage=nan.copy(),
            scale=math.nan,
            frac=frac,
            k=k,
            it=it,
            delta=delta,
            robust_weights=nan.copy(),
        )
    if not math.isfinite(float(_sd.max(xv)) - float(_sd.min(xv))):
        raise ValueError("lowess(): the range of 'x' overflows")
    fitted, leverage, rho, df, rss, xs, ys, ws, rhos = _core.lowess_fit(xv, yv, wv, k, it, delta)
    nobs = n if wv is None else int(_sd.count_nonzero(wv > 0))
    dof = nobs - df
    scale = math.sqrt(rss / dof) if dof > 0 else math.nan
    resid = yv - fitted
    return LowessResult(
        (xs, ys, ws, rhos),
        fittedvalues=scatter_back(fitted, keep, n_in),
        resid=scatter_back(resid, keep, n_in),
        nobs=nobs,
        df=df,
        leverage=scatter_back(leverage, keep, n_in),
        scale=scale,
        frac=frac,
        k=k,
        it=it,
        delta=delta,
        robust_weights=scatter_back(rho, keep, n_in),
    )
