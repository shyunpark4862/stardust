"""Linear models: ols, wls, gls, linregress (docs/api/regression.md §2, docs/design/regression.md §2-§3).

The fit itself is the native kernel stardust._core.regression.fit (QR, aliased columns,
refinement on the augmented system, sums of squares in double-double, covariances). This
module checks and converts the inputs (frame columns, NaN policy, GLS whitening) and builds
the result object, whose inference fields are computed on first use from the kernel's
output with sd.stats' distributions. Every field is an sd.ndarray, or a Python number for a
scalar of one regression; a batch of regressions puts the batch shape in front.
"""

import math
import numbers
import operator
import textwrap

from .. import _core as _sd
from .. import _results
from .. import frame as _frame
from .. import linalg as _linalg
from ..stats import ChiSquared as _ChiSquared
from ..stats import F as _F
from ..stats import StudentT as _StudentT

_k = _sd.regression

# Corrections of the augmented system after the QR solution (G15, docs/design/regression.md §3.1).
_REFINE_STEPS = 3  # at most; the kernel stops when a further step cannot matter (docs/design/regression.md G15)

_COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster")
_POLICIES = ("propagate", "omit", "raise")

# the kernel's per-regression scalars (csrc/regression/linear.cpp, enum Stat)
_NOBS, _RANK, _SSR, _CTSS, _UTSS, _ESSC, _ESSU, _SUMLOGW, _WALDF, _WALDQ, _ISNAN, _NGROUPS, _LAGS = range(13)

_LOG_2PI = math.log(2 * math.pi)


def _type_name(v):
    return type(v).__name__


# ---- argument checks (the message forms of API.md §2.1 and of sd.stats / sd.linalg)

def _check_bool(v, func, arg):
    if v is not True and v is not False:
        raise TypeError(f"{func}(): '{arg}' must be True or False, not {_type_name(v)}")
    return v


def _check_choice(v, func, arg, choices):
    listed = ", ".join(f"'{c}'" for c in choices[:-1]) + f" or '{choices[-1]}'"
    if not isinstance(v, str):
        raise TypeError(f"{func}(): '{arg}' must be {listed}, not {_type_name(v)}")
    if v not in choices:
        raise ValueError(f"{func}(): '{arg}' must be {listed}, not {v!r}")
    return v


def _check_policy(v, func):
    # sd.stats' message for the same option (stats/_descriptive.py)
    if v not in _POLICIES:
        raise ValueError(f"{func}(): 'nan_policy' must be one of 'propagate', 'omit', 'raise', not {v!r}")
    return v


def _check_rtol(v, func):
    """None -> -1 (the kernel's default, max(nobs, k) * eps); else a finite real >= 0."""
    if v is None:
        return -1.0
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): 'rtol' must be None, an int or a float, not {_type_name(v)}")
    try:
        x = float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): 'rtol' is out of range") from None
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): 'rtol' must be a finite number >= 0")
    return x


def _check_maxlags(v, func, cov_type):
    """None -> -1 (the kernel's default, floor(4 (nobs/100)^(2/9))); else an int >= 0, and only
    with cov_type='HAC'."""
    if v is None:
        return -1
    if cov_type != "HAC":
        raise ValueError(f"{func}(): 'maxlags' is only used when cov_type='HAC'")
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{func}(): 'maxlags' must be None or an int, not {_type_name(v)}")
    v = operator.index(v)
    if v < 0:
        raise ValueError(f"{func}(): 'maxlags' must be >= 0")
    if v >= 2**63:
        raise OverflowError(f"{func}(): 'maxlags' is out of range")
    return v


_GROUPS_TYPES = "an sd.ndarray, a sequence of labels or a frame Series"


def _groups(groups, func, n, cov_type):
    """(codes, gtot, missing) for cov_type='cluster': int64 codes 0 .. gtot-1 per row ((..., n)
    for an sd.ndarray with batch axes, else (n,)), -1 where the label is missing (NaN, None or a
    frame NA); `missing` is the bool mask of those rows, or None. Labels: integers, bools,
    floats (NaN missing), strings (sequences and frame Series); only equality matters."""
    if groups is None:
        if cov_type == "cluster":
            raise ValueError(f"{func}(): 'groups' is required when cov_type='cluster'")
        return None, 0, None
    if cov_type != "cluster":
        raise ValueError(f"{func}(): 'groups' is only used when cov_type='cluster'")
    if isinstance(groups, _frame.Series):
        if len(groups) != n:
            raise ValueError(f"{func}(): 'groups' must have {n} values like 'y', not {len(groups)}")
        col, _, gtot = _frame._core.factorize_rows([groups._col])
        codes = col.to_array_view(False).astype(_sd.int64)
        miss = groups.isna().to_array()
        if bool(_sd.any(miss)):
            codes = _sd.where(miss, -1, codes)
            return codes, gtot, miss
        return codes, gtot, None
    if isinstance(groups, _frame.DataFrame) or isinstance(groups, (str, bytes, numbers.Number)):
        raise TypeError(f"{func}(): 'groups' must be {_GROUPS_TYPES}, not {_type_name(groups)}")
    a = groups if isinstance(groups, _sd.ndarray) else None
    if a is None:
        try:
            labels = list(groups)
        except TypeError:
            raise TypeError(f"{func}(): 'groups' must be {_GROUPS_TYPES}, not {_type_name(groups)}") from None
        try:
            a = _sd.asarray(groups)  # numbers (a batch of them too)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to int64") from None
        except (TypeError, ValueError):
            a = None
        if a is None:
            # strings (or mixed labels): numbered in order of first appearance
            if len(labels) != n:
                raise ValueError(f"{func}(): 'groups' must have {n} values like 'y', not {len(labels)}")
            seen = {}
            out = []
            for v in labels:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    out.append(-1)
                    continue
                try:
                    out.append(seen.setdefault(v, len(seen)))
                except TypeError:
                    raise TypeError(f"{func}(): the labels in 'groups' must be strings or numbers, "
                                    f"not {_type_name(v)}") from None
            codes = _sd.asarray(out, dtype=_sd.int64)
            miss = codes < 0
            return codes, len(seen), (miss if bool(_sd.any(miss)) else None)
    if a.ndim == 0:
        raise ValueError(f"{func}(): 'groups' must have at least 1 dimension")
    if a.shape[-1] != n:
        raise ValueError(f"{func}(): 'groups' must have {n} values like 'y', not {a.shape[-1]}")
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): 'groups' must be real, not a complex128 array")
    miss = None
    if a.dtype == _sd.float64:
        nan = _sd.isnan(a)
        if bool(_sd.any(nan)):
            miss = nan
            a = _sd.where(nan, 0.0, a)
    elif a.dtype != _sd.int64:
        a = a.astype(_sd.int64)
    u = _sd.unique_inverse(a)
    codes = _sd.reshape(u.inverse_indices, a.shape).astype(_sd.int64)
    if miss is not None:
        codes = _sd.where(miss, -1, codes)
    return codes, int(u.values.shape[0]), miss


def _check_alpha(v, func):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): 'alpha' must be an int or a float, not {_type_name(v)}")
    x = float(v)
    if not 0 < x < 1:
        raise ValueError(f"{func}(): 'alpha' must be between 0 and 1 (exclusive)")
    return x


def _as_real(x, func, arg):
    """(x as a float64 sd.ndarray, its column names or None). Takes sd.ndarrays, sequences of
    numbers and frame Series/DataFrames (missing values become NaN)."""
    names = None
    if isinstance(x, (_frame.Series, _frame.DataFrame)):
        if isinstance(x, _frame.Series):
            names = [str(x.name)] if x.name is not None else None
        else:
            names = [str(c) for c in x.columns]
        try:
            a = x.to_array(dtype=_sd.float64, na_value=math.nan)
        except TypeError:
            raise TypeError(f"{func}(): '{arg}' must have numeric columns") from None
        return a, names
    if isinstance(x, _sd.ndarray):
        a = x
    elif isinstance(x, numbers.Number):
        if isinstance(x, complex):
            raise TypeError(f"{func}(): '{arg}' must be real, not complex")
        raise ValueError(f"{func}(): '{arg}' must have at least 1 dimension")
    else:
        try:
            a = _sd.asarray(x)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
        except (TypeError, ValueError):
            raise TypeError(f"{func}(): '{arg}' must be an sd.ndarray, a sequence of numbers or a frame "
                            f"Series or DataFrame, not {_type_name(x)}") from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not a complex128 array")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    return a, names


def _has_nan(*arrays):
    return any(a is not None and a.size and bool(_sd.any(_sd.isnan(a))) for a in arrays)


def _expand(v, nd):
    """A per-regression array (batch shape) with `nd` trailing axes of length 1."""
    if not isinstance(v, _sd.ndarray):
        return v
    return _sd.reshape(v, tuple(v.shape) + (1,) * nd)


# ---- the result object

class _LinearResult:
    """The result of ols, wls or gls: statsmodels' names (G10). The intercept is params[0]."""

    __slots__ = ("_raw", "_cache", "param_names", "cov_type", "_intercept", "_batch", "_extra", "_fitargs")

    def __init__(self, raw, *, param_names, intercept, cov_type, batch, extra=None, fitargs=None):
        self._raw = raw          # params, cov, R, fitted, resid, kept, stats
        self._cache = {}
        self.param_names = param_names
        self.cov_type = cov_type
        self._intercept = intercept
        self._batch = batch      # the batch shape, () for one regression
        self._extra = extra or {}
        # what the kernel was given (func, y, X, w, u, rtol, group codes, their count): the
        # diagnostics refactor or refit the same whitened problem
        self._fitargs = fitargs

    # -- helpers
    def _cached(self, name, make):
        c = self._cache
        if name not in c:
            c[name] = make()
        return c[name]

    def _stat(self, i):
        v = self._raw[6][..., i]
        return float(v) if not self._batch else v

    def _scalar(self, v):
        """A per-regression value: a Python float for one regression."""
        if self._batch or not isinstance(v, _sd.ndarray):
            return v
        return float(v)

    def _per(self, v, nd=1):
        """A per-regression value made to broadcast against arrays with `nd` more axes."""
        return _expand(v, nd) if self._batch else v

    @property
    def _robust(self):
        return self.cov_type != "nonrobust"

    # -- fit
    @property
    def params(self):
        return self._raw[0]

    @property
    def fittedvalues(self):
        return self._raw[3]

    @property
    def resid(self):
        return self._raw[4]

    @property
    def nobs(self):
        return self._stat(_NOBS)

    @property
    def rank(self):
        r = self._raw[6][..., _RANK]
        return int(float(r)) if not self._batch else r.astype(_sd.int64)

    def _isnan(self):
        """True for a regression whose data had NaN (nan_policy='propagate')."""
        f = self._raw[6][..., _ISNAN]
        return f != 0 if self._batch else float(f) != 0

    def _nan_where(self, v):
        f = self._isnan()
        if self._batch:
            return _sd.where(f, math.nan, v)
        return math.nan if f else v

    @property
    def df_model(self):
        return self._cached("df_model", lambda: self._nan_where(self._stat(_RANK) - (1.0 if self._intercept else 0.0)))

    @property
    def df_resid(self):
        return self._cached("df_resid", lambda: self.nobs - self._stat(_RANK))

    @property
    def ssr(self):
        return self._stat(_SSR)

    @property
    def centered_tss(self):
        return self._stat(_CTSS)

    @property
    def uncentered_tss(self):
        return self._stat(_UTSS)

    def _tss(self):
        return self.centered_tss if self._intercept else self.uncentered_tss

    def _ess_direct(self):
        e = self._stat(_ESSC) if self._intercept else self._stat(_ESSU)
        # the intercept alone explains nothing: ESS is 0 exactly (the summed value is the
        # rounding of params[0] to a double); statsmodels gives 0 too
        if not self._intercept:
            return e
        if self._batch:
            return _sd.where(self._raw[6][..., _RANK] == 1.0, 0.0, e)
        return 0.0 if self._stat(_RANK) == 1.0 else e

    @property
    def ess(self):
        """The explained sum of squares: summed directly where it is the smaller part of the
        total (R^2 < 1/2, where tss - ssr would cancel), else tss - ssr."""
        def make():
            e, s, t = self._ess_direct(), self.ssr, self._tss()
            if self._batch:
                return _sd.where(e < s, e, t - s)
            return e if e < s else t - s
        return self._cached("ess", make)

    def _one_minus_r2(self):
        e, s, t = self._ess_direct(), self.ssr, self._tss()
        if self._batch:
            return _sd.where(e < s, 1.0 - e / t, s / t)
        return 1.0 - _div(e, t) if e < s else _div(s, t)

    @property
    def rsquared(self):
        def make():
            e, s, t = self._ess_direct(), self.ssr, self._tss()
            if self._batch:
                return _sd.where(e < s, e / t, 1.0 - s / t)
            return _div(e, t) if e < s else 1.0 - _div(s, t)
        return self._cached("rsquared", make)

    @property
    def rsquared_adj(self):
        def make():
            c = 1.0 if self._intercept else 0.0
            return 1.0 - _div(self.nobs - c, self.df_resid) * self._one_minus_r2()
        return self._cached("rsquared_adj", make)

    @property
    def scale(self):
        return self._cached("scale", lambda: _div(self.ssr, self.df_resid))

    @property
    def llf(self):
        def make():
            n = self.nobs
            v = -n / 2 * (_LOG_2PI + _log(_div(self.ssr, n)) + 1.0) + self._stat(_SUMLOGW) / 2
            if "logdet" in self._extra:
                v = v - self._extra["logdet"] / 2
            return v
        return self._cached("llf", make)

    @property
    def aic(self):
        return self._cached("aic", lambda: -2.0 * self.llf + 2.0 * self._stat(_RANK))

    @property
    def bic(self):
        return self._cached("bic", lambda: -2.0 * self.llf + _log(self.nobs) * self._stat(_RANK))

    @property
    def condition_number(self):
        return self._cached("condition_number", lambda: _k.cond(self._raw[2], self._raw[5]))

    # -- inference
    def cov_params(self):
        """The covariance of the parameters (k x k); rows and columns of aliased parameters are NaN."""
        def make():
            c = self._raw[1]
            if self._robust:
                return c
            return self._per(self.scale, 2) * c
        return self._cached("cov_params", make)

    @property
    def bse(self):
        return self._cached("bse", lambda: _sd.sqrt(_linalg.diagonal(self.cov_params())))

    @property
    def tvalues(self):
        return self._cached("tvalues", lambda: self.params / self.bse)

    def _df_inference(self):
        """The degrees of freedom of the t and F reference distributions: df_resid, or the
        number of clusters - 1 under cov_type='cluster' (Stata; statsmodels' df_resid_inference)."""
        if self.cov_type == "cluster":
            return self._cached("df_inference", lambda: self._nan_where(self._stat(_NGROUPS) - 1.0))
        return self.df_resid

    @property
    def df_resid_inference(self):
        """The degrees of freedom of the t (and the F's denominator) used for p-values and
        intervals: df_resid, or G - 1 for a cluster covariance (G27; statsmodels' name)."""
        return self._df_inference()

    def _dist(self, nd=1):
        """The reference distribution of the t statistics: Student t with df_resid, under a
        robust covariance too (user decision 2026-11-18: as R's coeftest and Stata; statsmodels'
        default is the normal); G - 1 degrees of freedom for cluster (G27)."""
        return _StudentT(df=self._per(self._df_inference(), nd))

    @property
    def pvalues(self):
        return self._cached("pvalues", lambda: 2.0 * self._dist().ccdf(_sd.abs(self.tvalues)))

    def conf_int(self, alpha=0.05):
        """(..., k, 2): the lower and upper bounds of the 1 - alpha confidence intervals."""
        alpha = _check_alpha(alpha, "conf_int")
        q = self._dist().iccdf(alpha / 2)
        half = q * self.bse
        return _sd.stack([self.params - half, self.params + half], axis=-1)

    @property
    def fvalue(self):
        def make():
            if self._robust:
                return self._stat(_WALDF)
            dfm = self.df_model
            f = _div(_div(self.ess, dfm), self.scale)
            # no parameter but the intercept: no F test (statsmodels: NaN)
            if self._batch:
                return _sd.where(dfm == 0.0, math.nan, f)
            return math.nan if dfm == 0 else f
        return self._cached("fvalue", make)

    @property
    def f_pvalue(self):
        def make():
            dfn = self._stat(_WALDQ) if self._robust else self.df_model
            return self._scalar(_F(dfn=_arg(dfn), dfd=_arg(self._df_inference())).ccdf(_arg(self.fvalue)))
        return self._cached("f_pvalue", make)

    # -- prediction
    def _beff(self):
        """params with 0 for the aliased columns (they take no part in predictions, as in R)."""
        def make():
            keep = _sd.logical_or(self._raw[5], self._per(self._isnan(), 1))
            return _sd.where(keep, self.params, 0.0)
        return self._cached("beff", make)

    def _design(self, X_new, func):
        Xn, _ = _as_real(X_new, func, "X_new")
        p = self.params.shape[-1] - (1 if self._intercept else 0)
        if Xn.ndim == 0:
            raise ValueError(f"{func}(): 'X_new' must have at least 1 dimension")
        if Xn.ndim == 1:
            if p != 1:
                raise ValueError(f"{func}(): 'X_new' must have {p} columns like the fit's 'X', not 1-dimensional")
            Xn = _sd.reshape(Xn, (Xn.shape[0], 1))
        if Xn.shape[-1] != p:
            raise ValueError(f"{func}(): 'X_new' must have {p} columns like the fit's 'X', not {Xn.shape[-1]}")
        return Xn

    def predict(self, X_new, /):
        """X_new @ params (the intercept added), summed in double-double. Aliased columns take
        no part."""
        Xn = self._design(X_new, "predict")
        return _k.apply("predict", Xn, self._beff(), self._intercept, None)[0]

    def predict_interval(self, X_new, /, *, alpha=0.05, kind="mean"):
        """(..., m, 2): the bounds of the 1 - alpha interval for the mean at each row of X_new
        (kind='mean') or for a new observation there (kind='observation', variance scale + the
        mean's)."""
        func = "predict_interval"
        alpha = _check_alpha(alpha, func)
        _check_choice(kind, func, "kind", ("mean", "observation"))
        Xn = self._design(X_new, func)
        mean = _k.apply(func, Xn, self._beff(), self._intercept, None)[0]
        if self._intercept:
            Xf = _sd.concat([_sd.ones(tuple(Xn.shape[:-1]) + (1,)), Xn], axis=-1)
        else:
            Xf = Xn
        kept = self._raw[5]
        both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
        C = _sd.where(both, self.cov_params(), 0.0)
        var = _sd.sum(_sd.matmul(Xf, C) * Xf, axis=-1)
        if self._batch:
            var = _sd.where(_expand(self._isnan(), 1), math.nan, var)
        elif self._isnan():
            var = var * math.nan
        if kind == "observation":
            var = var + self._per(self.scale, 1)
        q = self._dist().iccdf(alpha / 2)
        half = q * _sd.sqrt(var)
        return _sd.stack([mean - half, mean + half], axis=-1)

    # -- diagnostics (G17, G28, G29). All are of the whitened problem the kernel fits (the model's
    # sqrt(w) y on sqrt(w) X for WLS, L^-1 y on L^-1 X for GLS), as the robust covariances are;
    # rows with weight 0 are out of the fit: leverage, studentized residuals and Cook's distance
    # are 0 there (the weighted formulas) and the sums leave them out.

    def _wresid(self):
        """The whitened residuals e~: resid (OLS), sqrt(w) resid (WLS), L^-1 resid (GLS)."""
        def make():
            func, y, X, w, u, tol, codes, gtot = self._fitargs
            if u is not None:
                D = X
                if self._intercept:
                    uc = _sd.broadcast_to(_sd.expand_dims(u, axis=-1), tuple(X.shape[:-1]) + (1,))
                    D = _sd.concat([uc, X], axis=-1)
                e = _k.apply("resid", D, self._beff(), False, y)[1]
                return _sd.broadcast_to(e, tuple(self.resid.shape)) if e.shape != self.resid.shape else e
            if w is not None:
                return _sd.sqrt(w) * self.resid
            return self.resid
        return self._cached("wresid", make)

    @property
    def leverage(self):
        """The hat values h_t (..., n): the diagonal of the hat matrix, ||q_t||^2 from the thin Q
        of the fit's QR (no X^T X is formed). 0 for rows with weight 0."""
        def make():
            func, y, X, w, u, tol, codes, gtot = self._fitargs
            return _k.fit(func, y, X, w, u, self._intercept, tol, 0, 0, codes, gtot, -1, True)
        return self._cached("leverage", make)

    @property
    def resid_studentized_internal(self):
        """e~_t / (s sqrt(1 - h_t)), s^2 = scale (statsmodels' resid_studentized_internal, R's rstandard)."""
        def make():
            return self._wresid() / _sd.sqrt(self._per(self.scale) * (1.0 - self.leverage))
        return self._cached("resid_studentized_internal", make)

    @property
    def resid_studentized_external(self):
        """e~_t / (s_(t) sqrt(1 - h_t)) with s_(t) the residual scale without row t (statsmodels'
        resid_studentized_external, R's rstudent): r sqrt((df - 1) / (df - r^2)), r the internal one."""
        def make():
            r = self.resid_studentized_internal
            df = self._per(self.df_resid)
            return r * _sd.sqrt((df - 1.0) / (df - r * r))
        return self._cached("resid_studentized_external", make)

    @property
    def cooks_distance(self):
        """Cook's distance r_t^2 / rank * h_t / (1 - h_t), r the internal studentized residual
        (R's cooks.distance; statsmodels' cooks_distance[0])."""
        def make():
            r, h = self.resid_studentized_internal, self.leverage
            return r * r / self._per(self._stat(_RANK)) * (h / (1.0 - h))
        return self._cached("cooks_distance", make)

    def _resid_stat(self, i):
        """The whitened residuals' sums over the fit's rows (the kernel's resid_stats, double-
        double): 0 Durbin-Watson's numerator, 1-3 the central sums of powers 2, 3, 4."""
        def make():
            w = self._fitargs[3]
            return _k.resid_stats(self._wresid(), w)
        v = self._cached("resid_stats", make)[..., i]
        return float(v) if not self._batch else v

    @property
    def durbin_watson(self):
        """sum (e~_t - e~_{t-1})^2 / sum e~_t^2 over the fit's rows in order (sums in double-double)."""
        return self._cached("durbin_watson", lambda: _div(self._resid_stat(0), self.ssr))

    def jarque_bera(self):
        """JarqueBeraResult(statistic, pvalue): n/6 (S^2 + (K - 3)^2 / 4) with the skewness S and
        kurtosis K of the whitened residuals (central moments over n, summed in double-double),
        against chi-square with 2 degrees of freedom (statsmodels' jarque_bera of wresid)."""
        def make():
            n = self.nobs
            mu2 = _div(self._resid_stat(1), n)
            skew = _div(_div(self._resid_stat(2), n), mu2 * _sqrt(mu2))
            kurt = _div(_div(self._resid_stat(3), n), mu2 * mu2)
            ek = kurt - 3.0
            jb = n / 6.0 * (skew * skew + ek * ek / 4.0)
            p = self._scalar(_ChiSquared(df=2.0).ccdf(_arg(jb)))
            return _results.JarqueBeraResult(jb, p)
        return self._cached("jarque_bera", make)

    def breusch_pagan(self, *, studentize=True):
        """BreuschPaganResult(lm, lm_pvalue, fvalue, f_pvalue): the auxiliary least squares of
        e~^2 on a constant and the whitened design's columns (statsmodels' het_breuschpagan on
        wresid and the model's design with a constant). studentize=True (Koenker, the default;
        statsmodels and R's lmtest): lm = n R^2; False (Breusch and Pagan): lm = ESS / (2
        sigma^4), sigma^2 = ssr / n. lm against chi-square with the auxiliary df_model, the F
        test is the auxiliary regression's."""
        _check_bool(studentize, "breusch_pagan", "studentize")

        def aux():
            func, y, X, w, u, tol, codes, gtot = self._fitargs
            e = self._wresid()
            wa = None
            if u is not None:
                Z = X
                if self._intercept:
                    uc = _sd.broadcast_to(_sd.expand_dims(u, axis=-1), tuple(X.shape[:-1]) + (1,))
                    Z = _sd.concat([uc, X], axis=-1)
            elif w is not None:
                sw = _sd.expand_dims(_sd.sqrt(w), axis=-1)
                Z = sw * X
                if self._intercept:
                    Z = _sd.concat([_sd.broadcast_to(sw, tuple(Z.shape[:-1]) + (1,)), Z], axis=-1)
                wa = _sd.where(w > 0, 1.0, 0.0)  # the rows out of the fit stay out
            else:
                Z = X
            raw = _k.fit("breusch_pagan", e * e, Z, wa, None, True, tol, 0, _REFINE_STEPS)
            return OLSResult(raw, param_names=None, intercept=True, cov_type="nonrobust", batch=_batch_shape(raw))

        def make():
            a = self._cached("bp_aux", aux)
            if studentize:
                lm = a.nobs * a.rsquared
            else:
                s2 = _div(self.ssr, self.nobs)
                lm = _div(a.ess, 2.0 * s2 * s2)
            df = a.df_model
            if self._batch:
                df = _sd.where(df > 0, df, math.nan)
            elif not df > 0:
                df = math.nan
            p = self._scalar(_ChiSquared(df=_arg(df)).ccdf(_arg(lm)))
            return _results.BreuschPaganResult(lm, p, a.fvalue, a.f_pvalue)
        return self._cached(("breusch_pagan", studentize), make)

    # -- summary (G13)
    def summary(self, *, alpha=0.05):
        """A text table like statsmodels' OLS summary: the fit, the coefficients with their
        standard errors, t, p and 1 - alpha intervals, Durbin-Watson, Jarque-Bera and the
        condition number. One regression only."""
        alpha = _check_alpha(alpha, "summary")
        if self._batch:
            raise ValueError(f"summary(): takes one regression, not a batch of shape {tuple(self._batch)}")
        return _summary(self, alpha)

    def __repr__(self):
        name = type(self).__name__
        if self._batch:
            return f"{name}(batch shape {tuple(self._batch)}, {self.params.shape[-1]} parameters)"
        return f"{name}(nobs={self.nobs:g}, rank={self.rank}, rsquared={self.rsquared:.6g}, params={self.params})"


class OLSResult(_LinearResult):
    """The result of sd.regression.ols."""
    __slots__ = ()


class WLSResult(_LinearResult):
    """The result of sd.regression.wls."""
    __slots__ = ()


class GLSResult(_LinearResult):
    """The result of sd.regression.gls."""
    __slots__ = ()


def _div(a, b):
    """a / b with IEEE results for a zero divisor (Python floats raise)."""
    if isinstance(a, _sd.ndarray) or isinstance(b, _sd.ndarray):
        return a / b
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _log(x):
    if isinstance(x, _sd.ndarray):
        return _sd.log(x)
    if math.isnan(x) or x < 0:
        return math.nan
    return math.log(x) if x > 0 else -math.inf


def _sqrt(x):
    if isinstance(x, _sd.ndarray):
        return _sd.sqrt(x)
    return math.sqrt(x) if x >= 0 else math.nan


# ---- summary (G13): statsmodels' layout and number formats

_WIDTH = 78
_COV_NOTES = {
    "nonrobust": "Standard errors assume that the covariance matrix of the errors is correctly specified.",
    "HC0": "Standard errors are heteroscedasticity robust (HC0).",
    "HC1": "Standard errors are heteroscedasticity robust (HC1).",
    "HC2": "Standard errors are heteroscedasticity robust (HC2).",
    "HC3": "Standard errors are heteroscedasticity robust (HC3).",
}


def _forg(x, prec=3):
    """statsmodels' forg: fixed with prec decimals, or prec significant digits outside [1e-4, 1e4)."""
    if math.isfinite(x) and (abs(x) >= 1e4 or abs(x) < 1e-4) and x != 0:
        return f"{x:.{prec}g}"
    return f"{x:.{prec}f}"


def _count(v):
    return "nan" if math.isnan(v) else str(int(v))


def _pairs(left, right):
    """Two columns of (label, value) cells, each 38 wide, values right-aligned."""
    half = (_WIDTH - 2) // 2
    rows = []
    for i in range(max(len(left), len(right))):
        cells = []
        for side in (left, right):
            if i < len(side):
                label, value = side[i]
                cells.append(label + value.rjust(half - len(label)))
            else:
                cells.append(" " * half)
        rows.append((cells[0] + "  " + cells[1]).rstrip())
    return rows


def _quantile_label(q):
    """0.025, 0.975, 0.05, 0.95 ...: the interval's quantiles as statsmodels prints them."""
    return f"{q:.3f}" if abs(q * 1000 - round(q * 1000)) < 1e-9 else f"{q:g}"


def _summary(r, alpha=0.05):
    name = type(r).__name__[:3]
    stars = "=" * _WIDTH
    lines = [f"{name} Regression Results".center(_WIDTH).rstrip(), stars]
    left = [("Model:", name), ("Method:", "Least Squares"), ("No. Observations:", _count(r.nobs)),
            ("Df Residuals:", _count(r.df_resid)), ("Df Model:", _count(r.df_model)), ("Covariance Type:", r.cov_type)]
    right = [("R-squared:", f"{r.rsquared:#.3f}"), ("Adj. R-squared:", f"{r.rsquared_adj:#.3f}"),
             ("F-statistic:", f"{r.fvalue:#.4g}"), ("Prob (F-statistic):", f"{r.f_pvalue:#.3g}"),
             ("Log-Likelihood:", f"{r.llf:#.5g}"), ("AIC:", f"{r.aic:#.4g}"), ("BIC:", f"{r.bic:#.4g}")]
    lines += _pairs(left, right)
    lines.append(stars)
    names = [str(v) for v in r.param_names]
    ci = r.conf_int(alpha)
    params, bse, tv, pv = (r.params, r.bse, r.tvalues, r.pvalues)
    stat = "t"
    wname = max([len(v) for v in names] + [9]) + 1
    cols = ["coef", "std err", stat, f"P>|{stat}|", "[" + _quantile_label(alpha / 2), _quantile_label(1 - alpha / 2) + "]"]
    head = " " * wname + "".join(c.rjust(11) for c in cols)
    width = max(_WIDTH, len(head))
    lines[1] = "=" * width
    lines[-1] = "=" * width
    lines.append(head)
    lines.append("-" * width)
    for j, nm in enumerate(names):
        vals = [_forg(float(params[j]), 4), _forg(float(bse[j])), _forg(float(tv[j])), f"{float(pv[j]):#.3f}",
                _forg(float(ci[j, 0])), _forg(float(ci[j, 1]))]
        lines.append(nm.ljust(wname) + "".join(v.rjust(11) for v in vals))
    lines.append("=" * width)
    jb = r.jarque_bera()
    lines += _pairs([("Durbin-Watson:", f"{r.durbin_watson:#.3f}"), ("Prob(JB):", f"{jb.pvalue:#.3g}")],
                    [("Jarque-Bera (JB):", f"{jb.statistic:#.3f}"), ("Cond. No.", f"{r.condition_number:#.3g}")])
    lines.append("=" * width)
    notes = []
    ct = r.cov_type
    if ct == "HAC":
        notes.append(f"Standard errors are heteroscedasticity and autocorrelation robust (HAC) using "
                     f"{_count(r._stat(_LAGS))} lags (Bartlett kernel) and without small sample correction.")
    elif ct == "cluster":
        g = r._stat(_NGROUPS)
        notes.append(f"Standard errors are robust to cluster correlation (cluster) with {_count(g)} groups; "
                     f"t and F use {_count(g - 1)} degrees of freedom.")
    else:
        notes.append(_COV_NOTES[ct])
    aliased = [nm for nm, k in zip(names, r._raw[5].tolist()) if not k] if not r._isnan() else []
    if aliased:
        notes.append("Aliased (linearly dependent on earlier columns), not estimated: " + ", ".join(aliased) + ".")
    lines.append("")
    lines.append("Notes:")
    for i, t in enumerate(notes):
        lines += textwrap.wrap(f"[{i + 1}] {t}", width, subsequent_indent="    ")
    return "\n".join(lines)


def _arg(v):
    """A distribution parameter or input: an sd.ndarray or a Python float."""
    return v if isinstance(v, _sd.ndarray) else float(v)


# ---- the fits

def _prepare(func, y, X, weights, nan_policy, extra_rows=None, groups=None, cov_type="nonrobust"):
    """Converted and checked y, X (..., n, p), weights and cluster groups; NaN rows (and rows
    with a missing group label) dropped for 'omit'. Returns (y, X, w, names, batch_shape,
    keep_rows or None, (codes, gtot))."""
    y, yn = _as_real(y, func, "y")
    if isinstance(y, _sd.ndarray) and y.ndim == 0:
        raise ValueError(f"{func}(): 'y' must have at least 1 dimension")
    X, names = _as_real(X, func, "X")
    if X.ndim == 0:
        raise ValueError(f"{func}(): 'X' must have at least 1 dimension")
    if X.ndim == 1:
        X = _sd.reshape(X, (X.shape[0], 1))
    n = y.shape[-1]
    if X.shape[-2] != n:
        raise ValueError(f"{func}(): 'X' must have {n} rows like 'y', not {X.shape[-2]}")
    w = None
    if weights is not None:
        w, _ = _as_real(weights, func, "weights")
        if w.ndim == 0 or w.shape[-1] != n:
            got = "a scalar" if w.ndim == 0 else str(w.shape[-1])
            raise ValueError(f"{func}(): 'weights' must have {n} values like 'y', not {got}")
    codes, gtot, gmiss = _groups(groups, func, n, cov_type)
    batch = y.ndim > 1 or X.ndim > 2 or (w is not None and w.ndim > 1) or (codes is not None and codes.ndim > 1)
    keep = None
    if nan_policy == "raise":
        if _has_nan(y, X, w) or gmiss is not None:
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
    elif nan_policy == "omit":
        if batch:
            raise ValueError(f"{func}(): nan_policy='omit' takes one regression, not a batch "
                             "(the regressions would keep different rows)")
        bad = _sd.logical_or(_sd.isnan(y), _sd.any(_sd.isnan(X), axis=1))
        if w is not None:
            bad = _sd.logical_or(bad, _sd.isnan(w))
        if extra_rows is not None:
            bad = _sd.logical_or(bad, extra_rows)
        if gmiss is not None:
            bad = _sd.logical_or(bad, gmiss)
        if bool(_sd.any(bad)):
            keep = _sd.logical_not(bad)
            y, X = y[keep], X[keep]
            if w is not None:
                w = w[keep]
            if codes is not None:
                codes = codes[keep]
    return y, X, w, names, batch, keep, (codes, gtot)


def _names(names, p, intercept):
    if names is None or len(names) != p:
        names = [f"x{j + 1}" for j in range(p)]
    return (["const"] if intercept else []) + list(names)


def _batch_shape(raw):
    return tuple(raw[0].shape[:-1])


def _check_options(func, intercept, cov_type, maxlags, rtol, nan_policy):
    _check_bool(intercept, func, "intercept")
    _check_choice(cov_type, func, "cov_type", _COV_TYPES)
    lags = _check_maxlags(maxlags, func, cov_type)
    tol = _check_rtol(rtol, func)
    _check_policy(nan_policy, func)
    return lags, tol


def _linear(func, cls, y, X, weights, intercept, cov_type, maxlags, groups, rtol, nan_policy):
    lags, tol = _check_options(func, intercept, cov_type, maxlags, rtol, nan_policy)
    y, X, w, names, _, _, (codes, gtot) = _prepare(func, y, X, weights, nan_policy, groups=groups, cov_type=cov_type)
    raw = _k.fit(func, y, X, w, None, intercept, tol, _COV_TYPES.index(cov_type), _REFINE_STEPS, codes, gtot, lags,
                 False)
    return cls(raw, param_names=_names(names, X.shape[-1], intercept), intercept=intercept, cov_type=cov_type,
               batch=_batch_shape(raw), fitargs=(func, y, X, w, None, tol, codes, gtot))


def ols(y, X, /, *, intercept=True, cov_type="nonrobust", maxlags=None, groups=None, rtol=None,
        nan_policy="propagate"):
    """Ordinary least squares of y (..., n) on X (..., n, k) or (n,), an intercept column first."""
    return _linear("ols", OLSResult, y, X, None, intercept, cov_type, maxlags, groups, rtol, nan_policy)


def wls(y, X, /, *, weights, intercept=True, cov_type="nonrobust", maxlags=None, groups=None, rtol=None,
        nan_policy="propagate"):
    """Weighted least squares: minimizes sum w_i (y_i - x_i b)^2. Rows with weight 0 take no
    part in the fit, nobs or the degrees of freedom (R's lm); their fitted values and
    residuals are computed."""
    if weights is None:
        raise TypeError("wls(): 'weights' must be an sd.ndarray, a sequence of numbers or a frame Series, not NoneType")
    return _linear("wls", WLSResult, y, X, weights, intercept, cov_type, maxlags, groups, rtol, nan_policy)


def gls(y, X, /, *, sigma, intercept=True, cov_type="nonrobust", maxlags=None, groups=None, rtol=None,
        nan_policy="propagate"):
    """Generalized least squares with the known error covariance sigma (..., n, n): the fit of
    L^-1 y on L^-1 X for sigma = L L^T (Cholesky; its lower triangle is read). Fitted values
    and residuals are on the original scale."""
    func = "gls"
    lags, tol = _check_options(func, intercept, cov_type, maxlags, rtol, nan_policy)
    if sigma is None:
        raise TypeError("gls(): 'sigma' must be an sd.ndarray or a sequence of numbers, not NoneType")
    S, _ = _as_real(sigma, func, "sigma")
    yy, _ = _as_real(y, func, "y")
    n = yy.shape[-1] if yy.ndim else 0
    if S.ndim < 2 or S.shape[-1] != n or S.shape[-2] != n:
        got = "x".join(str(v) for v in S.shape[-2:]) if S.ndim >= 2 else f"{S.ndim}-dimensional"
        raise ValueError(f"gls(): 'sigma' must be {n}x{n} like the rows of 'y', not {got}")
    if nan_policy == "raise" and _has_nan(S):
        raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
    if nan_policy == "omit" and S.ndim > 2:
        raise ValueError(f"{func}(): nan_policy='omit' takes one regression, not a batch "
                         "(the regressions would keep different rows)")
    nan_rows = None
    if nan_policy == "omit" and _has_nan(S):
        # observation i has NaN when sigma's row or column i has one
        nanS = _sd.isnan(S)
        nan_rows = _sd.logical_or(_sd.any(nanS, axis=1), _sd.any(nanS, axis=0))
    y, X, _, names, _, keep, (codes, gtot) = _prepare(func, yy, X, None, nan_policy, extra_rows=nan_rows,
                                                      groups=groups, cov_type=cov_type)
    if keep is not None:
        idx = _sd.nonzero(keep)[0]
        S = _sd.take(_sd.take(S, idx, axis=0), idx, axis=1)
    m = y.shape[-1]
    # NaN under 'propagate' (sd.linalg takes none): whiten with the NaN replaced (sigma by I),
    # then give the kernel a NaN y for those regressions, so all their results are NaN
    bad = None
    if nan_policy == "propagate" and _has_nan(y, X, S):
        bad_s = _sd.any(_sd.isnan(S), axis=(-2, -1), keepdims=True)
        S = _sd.where(bad_s, _sd.eye(m), S)
        bad = _sd.logical_or(_sd.logical_or(_sd.any(_sd.isnan(y), axis=-1, keepdims=True),
                                             _sd.any(_sd.isnan(X), axis=(-2, -1), keepdims=True)[..., 0]),
                             bad_s[..., 0])
        y = _sd.where(_sd.isnan(y), 0.0, y)
        X = _sd.where(_sd.isnan(X), 0.0, X)
    if S.size and bool(_sd.any(_sd.isinf(S))):
        raise ValueError("gls(): 'sigma' contains infinity")
    try:
        L = _linalg.cholesky(S)
    except _linalg.LinAlgError as e:
        msg = str(e)
        at = msg[msg.find(" (at batch index"):] if " (at batch index" in msg else ""
        raise _linalg.LinAlgError(f"gls(): 'sigma' is not positive definite{at}") from None
    yt = _linalg.solve_triangular(L, _sd.expand_dims(y, axis=-1), upper=False)[..., 0]
    if bad is not None:
        yt = _sd.where(bad, math.nan, yt)
    Xt = _linalg.solve_triangular(L, X, upper=False)
    ones = _linalg.solve_triangular(L, _sd.ones((m, 1)), upper=False)[..., 0]
    raw = list(_k.fit(func, yt, Xt, None, ones, intercept, tol, _COV_TYPES.index(cov_type), _REFINE_STEPS, codes,
                      gtot, lags, False))
    res = GLSResult(raw, param_names=_names(names, X.shape[-1], intercept), intercept=intercept, cov_type=cov_type,
                    batch=_batch_shape(raw), fitargs=(func, yt, Xt, None, ones, tol, codes, gtot))
    # fitted values and residuals on the original scale (statsmodels' fittedvalues and resid)
    fitted, resid = _k.apply(func, X, res._beff(), intercept, y)
    if raw[3].shape != fitted.shape:  # sigma's batch axes added to y's
        fitted, resid = _sd.broadcast_to(fitted, raw[3].shape), _sd.broadcast_to(resid, raw[3].shape)
    raw[3], raw[4] = fitted, resid
    logdet = 2.0 * _sd.sum(_sd.log(_linalg.diagonal(L)), axis=-1)
    if res._batch:
        logdet = _sd.broadcast_to(logdet, res._batch) if isinstance(logdet, _sd.ndarray) else logdet
    else:
        logdet = float(logdet)
    res._extra["logdet"] = logdet
    return res


def linregress(x, y, /):
    """scipy.stats.linregress: the least squares line y = intercept + slope x, with the
    correlation, the two-sided p-value of slope = 0 (Student t, n - 2 degrees of freedom) and
    the two standard errors. A batch takes x and y (..., n)."""
    func = "linregress"
    xa, _ = _as_real(x, func, "x")
    ya, _ = _as_real(y, func, "y")
    if xa.ndim == 0 or ya.ndim == 0:
        raise ValueError(f"{func}(): 'x' and 'y' must have at least 1 dimension")
    if xa.shape[-1] != ya.shape[-1]:
        raise ValueError(f"{func}(): 'x' must have {ya.shape[-1]} values like 'y', not {xa.shape[-1]}")
    X = _sd.expand_dims(xa, axis=-1)
    # only an exactly constant x is degenerate here (scipy): rtol 0
    raw = _k.fit(func, ya, X, None, None, True, 0.0, 0, _REFINE_STEPS)
    r = OLSResult(raw, param_names=["const", "x1"], intercept=True, cov_type="nonrobust", batch=_batch_shape(raw))
    nan = math.nan
    if not r._batch:
        b0, b1 = float(r.params[0]), float(r.params[1])
        se0, se1 = float(r.bse[0]), float(r.bse[1])
        rvalue = min(1.0, max(-1.0, math.copysign(math.sqrt(r.rsquared), b1))) if r.rsquared >= 0 else nan
        pvalue = 2.0 * _StudentT(df=r.df_resid).ccdf(abs(_div(b1, se1)))
        # scipy's degenerate cases: x or y constant gives r = NaN, and with it p and both
        # standard errors; a constant x gives no line at all
        xconst = float(_sd.max(xa)) == float(_sd.min(xa))
        yconst = float(_sd.max(ya)) == float(_sd.min(ya))
        if xconst:
            b0 = b1 = nan
        if xconst or yconst:
            rvalue = pvalue = se0 = se1 = nan
        return _results.LinregressResult(b1, b0, rvalue, pvalue, se1, se0)
    b0, b1 = r.params[..., 0], r.params[..., 1]
    se0, se1 = r.bse[..., 0], r.bse[..., 1]
    rvalue = _sd.clip(_sd.copysign(_sd.sqrt(r.rsquared), b1), min=-1.0, max=1.0)
    pvalue = 2.0 * _StudentT(df=r.df_resid).ccdf(_sd.abs(b1 / se1))
    xconst = _sd.broadcast_to(_sd.max(xa, axis=-1) == _sd.min(xa, axis=-1), r._batch)
    yconst = _sd.broadcast_to(_sd.max(ya, axis=-1) == _sd.min(ya, axis=-1), r._batch)
    degen = _sd.logical_or(xconst, yconst)
    return _results.LinregressResult(_sd.where(xconst, nan, b1), _sd.where(xconst, nan, b0),
                                     _sd.where(degen, nan, rvalue), _sd.where(degen, nan, pvalue),
                                     _sd.where(degen, nan, se1), _sd.where(degen, nan, se0))
