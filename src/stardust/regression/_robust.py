"""Robust regression: rlm (M and MM estimation), theilslopes, siegelslopes
(docs/api/regression.md §5, docs/design/regression.md §3.6).

The fits are the native kernels stardust._core.regression.rlm_fit (IRLS through the least
squares core of the linear models; MM's fast-S start; Huber's H1 and robustbase's
.vcov.avar1 covariances) and stardust._core.regression.slopes (the pairwise slopes' exact
medians). This module checks and converts the inputs, draws MM's subsample keys from the
user's Generator and builds the results.
"""

import math
import numbers

from .. import _calls
from .. import _core as _sd
from .. import _summary as _S
from .. import random as _random
from .. import special as _special
from .._results import SiegelslopesResult, TheilslopesResult
from ._glm import _broadcast_shape, _check_int, _check_positive
from ._linear import (
    _aliased_note,
    _as_real,
    _batch_summary,
    _check_alpha,
    _check_batches,
    _check_bool,
    _check_choice,
    _check_policy,
    _check_rows,
    _check_rtol,
    _coef_table,
    _count,
    _expand,
    _frame_columns,
    _has_nan,
    _names,
    _prepare,
    _read_y,
    _RegressionResult,
    _ro,
    _rows_message,
    _type_name,
    _y_units,
    _y_unscale,
)

_k = _sd.regression

_METHODS = ("M", "MM")
_NORMS = ("huber", "bisquare", "hampel")
_NORM_CODE = {"huber": 0, "bisquare": 1, "hampel": 2}
_SCALES = ("mad", "huber")
# statsmodels' class names, for summary()
_NORM_NAME = {"huber": "HuberT", "bisquare": "TukeyBiweight", "hampel": "Hampel"}
# the tuning constants' defaults (G54): statsmodels' HuberT(), TukeyBiweight(), Hampel() (95%
# efficiency at the normal for Huber and bisquare, 0.990 for Hampel (2, 4, 8)); MM's M step:
# robustbase's lmrob.control() tuning.psi
_C_DEFAULT = {"huber": 1.345, "bisquare": 4.685, "hampel": (2.0, 4.0, 8.0)}
_C_MM = 4.685061
_NRESAMPLE = 500  # lmrob.control()'s nResample

# the kernel's per-regression scalars (csrc/regression/robust.cpp, enum RStat)
_R_NOBS, _R_RANK, _R_SCALE, _R_NITER, _R_CONV, _R_ISNAN, _R_NGROUPS, _R_LAGS = range(8)


class RLMResult(_RegressionResult):
    """The result of sd.regression.rlm: statsmodels' names (G10). The intercept is params[0]."""

    __slots__ = ("_method", "_norm", "_scale_est")
    _I_NOBS, _I_RANK, _I_ISNAN, _I_NGROUPS, _I_LAGS = _R_NOBS, _R_RANK, _R_ISNAN, _R_NGROUPS, _R_LAGS

    def __init__(self, raw, *, method, norm, scale_est, **kw):
        super().__init__(raw, **kw)
        self._method = method
        self._norm = norm
        self._scale_est = scale_est

    # -- the fit's configuration: read-only (G98)
    @property
    def method(self):
        return self._method

    @property
    def norm(self):
        return self._norm

    @property
    def resid(self):
        return self._raw[4]

    @property
    def weights(self):
        """The final IRLS weights psi(r_i/s) / (r_i/s) (statsmodels' weights, lmrob's rweights)."""
        return self._raw[7]

    @property
    def scale(self):
        """The scale s of the residuals: M, the MAD or Huber's proposal 2 of the final
        residuals; MM, the S estimate's scale (lmrob's scale)."""
        return self._stat(_R_SCALE)

    @property
    def df_model(self):
        return self._cached(
            "df_model", lambda: self._nan_where(self._stat(_R_RANK) - (1.0 if self._intercept else 0.0))
        )

    @property
    def df_resid(self):
        return self._cached("df_resid", lambda: self._nan_where(self.nobs - self._stat(_R_RANK)))

    def _cov_s(self):
        """The covariance in the fit's units; all of it NaN with no degree of freedom left
        (n = rank: 0/0, G95)."""

        def make():
            c = self._raw[1]
            df = self.df_resid
            if self._batch:
                return _sd.where(_expand(df == 0.0, 2), math.nan, c)
            return c * math.nan if df == 0 else c

        return self._cached("cov_s", make)

    def _perfect(self):
        """A scale of exactly 0 (an exact fit of the rows that decide it) with df_resid > 0:
        t and p NaN (D-3)."""
        z = self._stat(_R_SCALE) == 0.0
        df = self.df_resid
        if self._batch:
            return _sd.logical_and(z, df > 0.0)
        return bool(z) and df > 0

    @property
    def converged(self):
        v = self._raw[6][..., _R_CONV]
        return v == 1.0 if self._batch else float(v) == 1.0

    @property
    def n_iter(self):
        v = self._raw[6][..., _R_NITER]
        if self._batch:
            return _sd.where(_sd.isnan(v), 0.0, v).astype(_sd.int64)
        v = float(v)
        return 0 if math.isnan(v) else int(v)

    def predict(self, X_new, /):
        """X_new @ params (the intercept added), summed in double-double. Aliased columns take
        no part; a NaN in a column of the model gives NaN."""
        return self._xbeta(self._design(X_new, "predict"), "predict")

    def predict_interval(self, X_new, /, *, alpha=0.05):
        """(..., m, 2): the 1 - alpha confidence interval of the fitted value (the mean's
        location) at each row of X_new, x params -+ t se, se^2 = x cov_params() x^T (G103). No
        interval for a new observation: the robust fit has no error distribution to draw it
        from (the scale is a robust spread, not a standard deviation)."""
        return self._interval(X_new, "predict_interval", alpha)

    def summary(self, *, alpha=0.05):
        """A text table like statsmodels' RLM summary: the fit, then the coefficients with their
        standard errors, t, p and 1 - alpha intervals; a batch as ols's (G193)."""
        alpha = _check_alpha(alpha, "summary")
        if self._batch:
            return _batch(self)
        return _summary(self, alpha)

    def __repr__(self):
        name = type(self).__name__
        if self._batch:
            return f"{name}(batch shape {tuple(self._batch)}, {self.params.shape[-1]} parameters)"
        fields = [("method", self.method), ("norm", self.norm), ("nobs", float(self.nobs)), ("rank", int(self.rank))]
        return _S.one_line(name, fields + [("scale", float(self.scale)), ("params", self.params)])


def _summary(r, alpha):
    left = [
        ("Model:", "RLM"),
        ("Method:", "IRLS" if r.method == "M" else "MM"),
        ("Norm:", _NORM_NAME[r.norm]),
        ("Scale Est.:", r._scale_est),
        ("Covariance Type:", r.cov_type),
        ("No. Iterations:", str(r.n_iter)),
    ]
    right = [
        ("No. Observations:", _count(r.nobs)),
        ("Df Residuals:", _count(r.df_resid)),
        ("Df Model:", _count(r.df_model)),
        ("Scale:", f"{r.scale:#8.5g}".strip()),
    ]
    names = [str(v) for v in r.param_names]
    table = _coef_table(r, alpha)
    notes = []
    if r.method == "M":
        notes.append("Standard errors are Huber's H1 (statsmodels' default); t uses df_resid degrees of freedom.")
    else:
        notes.append(
            "MM estimate: S start (500 elemental subsets, bisquare, 50% breakdown), bisquare M step with "
            "the S scale; standard errors are robustbase's .vcov.avar1; t uses df_resid degrees of freedom."
        )
    if not r._isnan():
        notes += _aliased_note(r, names)
        if not r.converged:
            notes.append(f"The IRLS did not converge in {r.n_iter} iterations.")
    return _S.render("Robust linear Model Regression Results", header=(left, right), tables=[table], notes=notes)


def _batch(r):
    left = [
        ("Model:", "RLM"),
        ("Method:", "IRLS" if r.method == "M" else "MM"),
        ("Norm:", _NORM_NAME[r.norm]),
        ("Scale Est.:", r._scale_est),
        ("Covariance Type:", r.cov_type),
    ]
    stats = [("nobs", _S.count), ("scale", _S.forg), ("n_iter", str), ("converged", _S.flag)]
    return _batch_summary(r, "Robust linear Model Regression Results", left, stats)


# ---- the fit


def _check_rng(rng, func):
    if not isinstance(rng, _random.Generator):
        raise TypeError(
            f"{func}(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not {_type_name(rng)}"
        )
    return rng


def _check_c(c, func, norm, method):
    """The norm's tuning constants as the kernel's (c0, c1, c2): Huber (t, 0, 0), bisquare
    (0, 0, c), Hampel (a, b, c)."""
    if c is None:
        c = _C_MM if method == "MM" else _C_DEFAULT[norm]
    if norm == "hampel":
        msg = f"{func}(): 'c' must be three numbers 0 < a <= b < c for the hampel norm"
        if isinstance(c, (str, bytes)) or not hasattr(c, "__len__"):
            raise TypeError(f"{msg}, not {_type_name(c)}")
        vals = list(c.tolist() if isinstance(c, _sd.ndarray) else c)
        if len(vals) != 3:
            raise ValueError(msg)
        for v in vals:
            if isinstance(v, bool) or not isinstance(v, numbers.Real):
                raise TypeError(f"{msg}, not {_type_name(v)}")
        a, b, cc = (float(v) for v in vals)
        if not (all(math.isfinite(v) for v in (a, b, cc)) and 0 < a <= b < cc):
            raise ValueError(msg)
        return a, b, cc
    msg = f"{func}(): 'c' must be None or a finite number > 0 for the {norm} norm"
    if isinstance(c, bool) or not isinstance(c, numbers.Real):
        raise TypeError(f"{msg}, not {_type_name(c)}")
    try:
        v = float(c)
    except OverflowError:
        raise OverflowError(f"{func}(): 'c' is out of range") from None
    if not (math.isfinite(v) and v > 0):
        raise ValueError(msg)
    return (v, 0.0, 0.0) if norm == "huber" else (0.0, 0.0, v)


@_calls.function
def rlm(
    y,
    X,
    /,
    *,
    method="M",
    norm=None,
    c=None,
    scale_est=None,
    intercept=True,
    max_iter=50,
    tol=1e-8,
    rng=None,
    rtol=None,
    nan_policy="propagate",
):
    """Robust linear model of y (..., n) on X (..., n, k) or (n,), an intercept column first
    (docs/api/regression.md §5). method='M': IRLS from least squares with the MAD (or
    Huber's proposal 2) scale updated each iteration, statsmodels' RLM; method='MM': the
    S estimate from elemental subsets drawn with `rng`, then a bisquare M step, robustbase's
    lmrob."""
    func = "rlm"
    _check_choice(method, func, "method", _METHODS)
    if norm is None:
        norm = "huber" if method == "M" else "bisquare"
    _check_choice(norm, func, "norm", _NORMS)
    if method == "MM" and norm != "bisquare":
        raise ValueError(f"{func}(): method='MM' takes the bisquare norm, not {norm!r}")
    if scale_est is None:
        scale_est = "mad" if method == "M" else None
    elif method == "MM":
        raise ValueError(f"{func}(): 'scale_est' is only used with method='M' (MM's scale is the S estimate's)")
    else:
        _check_choice(scale_est, func, "scale_est", _SCALES)
    cc = _check_c(c, func, norm, method)
    _check_bool(intercept, func, "intercept")
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_positive(tol, func, "tol")
    rt = _check_rtol(rtol, func)
    _check_policy(nan_policy, func)
    if method == "MM":
        _check_rng(rng, func)
    elif rng is not None:
        raise ValueError(f"{func}(): 'rng' is only used with method='MM'")

    columns = _frame_columns(X)
    yy, XX, _, names, _, _, _ = _prepare(func, y, X, None, nan_policy, intercept=intercept)
    # y far outside [2^-256, 2^256] is fitted in power-of-two units (exact; the M and S
    # estimates are equivariant under y's scale)
    yy, fy = _y_units(yy)
    keys = None
    if method == "MM":
        shape = _broadcast_shape(func, tuple(yy.shape[:-1]), tuple(XX.shape[:-2]))
        # two 64-bit words per regression, in the batch's C order (docs/api/regression.md §5)
        keys = rng.integers(-(2**63), 2**63 - 1, endpoint=True, shape=shape + (2,))
    raw = _k.rlm_fit(
        func,
        yy,
        XX,
        keys,
        intercept,
        rt,
        _METHODS.index(method),
        _NORM_CODE[norm],
        cc[0],
        cc[1],
        cc[2],
        _SCALES.index(scale_est) if method == "M" else 2,
        max_iter,
        tol,
        _NRESAMPLE,
    )
    extra = None
    if fy is not None:
        raw = list(raw)
        extra = _y_unscale(raw, fy, (_R_SCALE,))
    est = scale_est if method == "M" else "S"
    if est == "huber":
        est = "Huber"
    return RLMResult(
        raw,
        extra=extra,
        method=method,
        norm=norm,
        scale_est=est,
        param_names=_names(names, XX.shape[-1], intercept),
        intercept=intercept,
        cov_type="H1" if method == "M" else "avar1",
        batch=tuple(raw[0].shape[:-1]),
        columns=columns,
    )


# ---- Theil-Sen and Siegel (G56): scipy.stats' signatures and results


def _slopes_input(func, y, x, nan_policy):
    yy = _read_y(y, func)
    if yy.ndim == 0:
        raise ValueError(f"{func}(): 'y' must have at least 1 dimension")
    n = yy.shape[-1]
    if x is None:
        xx = _sd.arange(n, dtype=_sd.float64)
    else:
        xx, _ = _as_real(x, func, "x")
        if xx.ndim == 0:
            raise ValueError(f"{func}(): 'x' must have at least 1 dimension")
        if xx.shape[-1] != n:
            raise ValueError(_rows_message(func, "x", n, xx.shape[-1], yy.shape, "values"))
    _check_batches(func, [("y", yy.shape[:-1]), ("x", xx.shape[:-1])])
    batch = yy.ndim > 1 or xx.ndim > 1
    if nan_policy == "raise":
        if _has_nan(yy, xx):
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
    elif nan_policy == "omit":
        if batch:
            raise ValueError(
                f"{func}(): nan_policy='omit' takes one regression, not a batch "
                "(the regressions would keep different points)"
            )
        _check_rows(func, [("y", yy), ("x", xx)])  # inf stays an error in a row with a NaN (G24)
        bad = _sd.logical_or(_sd.isnan(yy), _sd.isnan(xx))
        if bool(_sd.any(bad)):
            keep = _sd.logical_not(bad)
            yy, xx = yy[keep], xx[keep]
    return yy, xx, batch


def _out(v, batch, cls, info):
    if batch:
        r = cls(*(_ro(v[..., i]) for i in range(v.shape[-1])))
    else:
        r = cls(*(float(a) for a in v.tolist()))
    r._info = info  # what summary() shows beyond the fields (G192)
    return r


def _slopes_info(yy, method, alpha=None):
    d = {"method": method, "nobs": yy.shape[-1]}
    if alpha is not None:
        d["alpha"] = alpha
    return d


def _too_few(yy, xx, batch, cls, info):
    """Fewer than 2 points: no slope; every field NaN (G96's empty-input rule; scipy 1.18.1
    gives NaN with a warning), or None when there are enough."""
    if yy.shape[-1] >= 2:
        return None
    nf = len(cls._fields)
    if not batch:
        r = cls(*([math.nan] * nf))
    else:
        shape = _broadcast_shape("", tuple(yy.shape[:-1]), tuple(xx.shape[:-1]))
        r = cls(*(_ro(_sd.full(shape, math.nan)) for _ in range(nf)))
    r._info = info
    return r


@_calls.function
def theilslopes(y, x=None, /, *, alpha=0.05, method="separate", nan_policy="propagate"):
    """scipy.stats.theilslopes: the median of the pairwise slopes, the intercept
    median(y) - slope median(x) ('separate') or median(y - slope x) ('joint'), and Sen's
    (1968) confidence bounds on the slope at significance level alpha, the 1 - alpha interval
    (as conf_int's alpha; folded as scipy's, so 0.05 and 0.95 both mean 95%). x=None is
    arange(len(y))."""
    func = "theilslopes"
    _check_choice(method, func, "method", ("joint", "separate"))
    a = _check_alpha(alpha, func)
    _check_policy(nan_policy, func)
    yy, xx, batch = _slopes_input(func, y, x, nan_policy)
    level = 1.0 - a if a > 0.5 else a
    info = _slopes_info(yy, method, level)
    few = _too_few(yy, xx, batch, TheilslopesResult, info)
    if few is not None:
        return few
    a = level
    z = float(_special.ndtri(a / 2.0))
    v = _k.slopes(func, yy, xx, 0, method == "joint", z)
    return _out(v, batch, TheilslopesResult, info)


@_calls.function
def siegelslopes(y, x=None, /, *, method="hierarchical", nan_policy="propagate"):
    """scipy.stats.siegelslopes: the repeated median of the slopes (per point the median of
    its slopes to the others, then their median), and the intercept median(y - slope x)
    ('hierarchical') or the repeated median of the pairwise intercepts ('separate'). x=None
    is arange(len(y))."""
    func = "siegelslopes"
    _check_choice(method, func, "method", ("hierarchical", "separate"))
    _check_policy(nan_policy, func)
    yy, xx, batch = _slopes_input(func, y, x, nan_policy)
    info = _slopes_info(yy, method)
    few = _too_few(yy, xx, batch, SiegelslopesResult, info)
    if few is not None:
        return few
    v = _k.slopes(func, yy, xx, 1, method == "separate", 0.0)
    return _out(v, batch, SiegelslopesResult, info)


# the methods' call-shape errors in the library's form (R89)
_calls.methods(RLMResult)
