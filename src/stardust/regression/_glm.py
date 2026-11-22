"""Generalized linear models: glm (docs/api/regression.md §3, docs/design/regression.md §3.4).

The fit is the native kernel stardust._core.regression.glm_fit: IRLS whose every iteration is
one weighted least squares fit through the linear models' core (the same QR and aliased
columns), a final refined solve with the covariance, separation, and the deviance and
log-likelihood without cancellation. This module checks and converts the inputs (family and
link names, the weights, trials, offset and exposure, NaN policy) and builds the result,
whose inference fields come from the kernel's output and sd.stats' distributions.
"""

import math
import numbers
import operator

from .. import _calls
from .. import _core as _sd
from .. import _summary as _S
from .. import special as _special
from ..optimize._common import _values_text
from ..stats import ChiSquared as _ChiSquared
from ..stats import Normal as _Normal
from ..stats import StudentT as _StudentT
from ._linear import (
    _COV_NOTES,
    _COV_TYPES,
    _aliased_note,
    _arg,
    _as_real,
    _as_real_owned,
    _batch_summary,
    _check_alpha,
    _check_batches,
    _check_bool,
    _check_choice,
    _check_maxlags,
    _check_params,
    _check_policy,
    _check_rows,
    _check_rtol,
    _coef_table,
    _count,
    _div,
    _expand,
    _frame_columns,
    _groups,
    _has_nan,
    _log,
    _names,
    _pow2,
    _read_y_owned,
    _RegressionResult,
    _rows_message,
)

_k = _sd.regression

_FAMILIES = ("gaussian", "binomial", "poisson", "gamma")
# the links of each family (G31), the canonical one first
_FAMILY_LINKS = {
    "gaussian": ("identity", "log"),
    "binomial": ("logit", "probit", "cloglog", "log"),
    "poisson": ("log", "identity", "sqrt"),
    "gamma": ("inverse", "log", "identity"),
}
# the kernel's codes (csrc/regression/glm.cpp, enum Family and enum Link)
_FAMILY_CODE = {"gaussian": 0, "binomial": 1, "poisson": 2, "gamma": 3}
_LINK_CODE = {"identity": 0, "log": 1, "logit": 2, "probit": 3, "cloglog": 4, "sqrt": 5, "inverse": 6}
# summary(): statsmodels' class names
_FAMILY_NAME = {"gaussian": "Gaussian", "binomial": "Binomial", "poisson": "Poisson", "gamma": "Gamma"}
_LINK_NAME = {
    "identity": "Identity",
    "log": "Log",
    "logit": "Logit",
    "probit": "Probit",
    "cloglog": "CLogLog",
    "sqrt": "Sqrt",
    "inverse": "InversePower",
}

# the kernel's per-regression scalars (csrc/regression/glm.cpp, enum GStat)
(
    _G_NOBS,
    _G_RANK,
    _G_DEV,
    _G_NULLDEV,
    _G_PEARSON,
    _G_SCALE,
    _G_LLF,
    _G_LLNULL,
    _G_NITER,
    _G_CONV,
    _G_SEP,
    _G_ISNAN,
    _G_NGROUPS,
    _G_LAGS,
) = range(14)


def _inverse_link(link, eta):
    """mu = g^-1(eta) through sd.special and the array functions."""
    if link == "identity":
        return eta
    if link == "log":
        return _sd.exp(eta)
    if link == "logit":
        return _special.expit(eta)
    if link == "probit":
        return _special.ndtr(eta)
    if link == "cloglog":
        return -_sd.expm1(-_sd.exp(eta))
    if link == "sqrt":
        return eta * eta
    return 1.0 / eta


class GLMResult(_RegressionResult):
    """The result of sd.regression.glm: statsmodels' names (G10). The intercept is params[0]."""

    __slots__ = ("_family", "_link", "_scale_fixed")
    _I_NOBS, _I_RANK, _I_ISNAN, _I_NGROUPS, _I_LAGS = _G_NOBS, _G_RANK, _G_ISNAN, _G_NGROUPS, _G_LAGS

    def __init__(self, raw, *, family, link, scale_fixed, **kw):
        super().__init__(raw, **kw)
        self._family = family
        self._link = link
        self._scale_fixed = scale_fixed  # binomial and poisson by default, or a given value

    # -- the fit's configuration: read-only (G98)
    @property
    def family(self):
        return self._family

    @property
    def link(self):
        return self._link

    # -- fit
    @property
    def linear_predictor(self):
        """eta = X params + offset (+ log exposure): the fitted values on the link's scale."""
        return self._raw[4]

    @property
    def resid_response(self):
        """y - mu (on the scale of y: proportions for a binomial with trials)."""
        return self._raw[7]

    @property
    def resid_pearson(self):
        """(y - mu) sqrt(m / V(mu)), m = var_weights x trials."""
        return self._raw[8]

    @property
    def resid_deviance(self):
        """sign(y - mu) sqrt(m d(y, mu)): the square roots of the deviance's terms."""
        return self._raw[9]

    @property
    def resid_working(self):
        """(y - mu) g'(mu): the IRLS's working residuals."""
        return self._raw[10]

    # the kernel's sums are in the fit's units when it fitted y / fy (gaussian, gamma; see
    # glm()): the gaussian's squares scale by fy^2, the log-likelihoods shift by -nobs log fy
    def _out2(self, v):
        f2 = self._extra.get("dev2")
        return v if f2 is None else v * self._scalar(f2)

    def _outll(self, v):
        sh = self._extra.get("llshift")
        return v if sh is None else v + self._scalar(sh)

    @property
    def deviance(self):
        return self._out2(self._stat(_G_DEV))

    @property
    def null_deviance(self):
        return self._out2(self._stat(_G_NULLDEV))

    @property
    def pearson_chi2(self):
        return self._out2(self._stat(_G_PEARSON))

    @property
    def scale(self):
        return self._out2(self._stat(_G_SCALE))

    @property
    def llf(self):
        return self._outll(self._stat(_G_LLF))

    @property
    def llnull(self):
        return self._outll(self._stat(_G_LLNULL))

    @property
    def df_model(self):
        return self._cached(
            "df_model", lambda: self._nan_where(self._stat(_G_RANK) - (1.0 if self._intercept else 0.0))
        )

    @property
    def df_resid(self):
        return self._cached("df_resid", lambda: self._nan_where(self.nobs - self._stat(_G_RANK)))

    def _cov_s(self):
        """The covariance in the fit's units; all of it NaN when an estimated scale has no
        degree of freedom (0/0, G95)."""

        def make():
            c = self._raw[1]
            if not self._robust:
                c = self._per(self._stat(_G_SCALE), 2) * c
            if self._scale_fixed:
                return c
            df = self.df_resid
            if self._batch:
                return _sd.where(_expand(df == 0.0, 2), math.nan, c)
            return c * math.nan if df == 0 else c

        return self._cached("cov_s", make)

    def _perfect(self):
        """An estimated scale of exactly 0 (a perfect fit) with df_resid > 0: t and p NaN (D-3)."""
        if self._scale_fixed:
            return False
        z = self._stat(_G_SCALE) == 0.0
        df = self.df_resid
        if self._batch:
            return _sd.logical_and(z, df > 0.0)
        return bool(z) and df > 0

    @property
    def aic(self):
        return self._cached("aic", lambda: -2.0 * self.llf + 2.0 * self._stat(_G_RANK))

    @property
    def bic(self):
        """-2 llf + log(nobs) rank (statsmodels' bic_llf, its bic)."""
        return self._cached("bic", lambda: -2.0 * self.llf + _log(self.nobs) * self._stat(_G_RANK))

    @property
    def llr(self):
        """The likelihood ratio statistic 2 (llf - llnull) against the null model."""
        return self._cached("llr", lambda: 2.0 * (self.llf - self.llnull))

    def pseudo_rsquared(self, *, kind="cs"):
        """The pseudo R-squared (statsmodels' name and default, G100): kind='cs' (Cox and Snell,
        the summary's) 1 - exp(2 (llnull - llf) / nobs), or 'mcf' (McFadden) 1 - llf / llnull.
        NaN where it is 0/0 (a perfect fit whose null model fits too: llf = llnull = +inf; or
        llnull 0 for McFadden) and, for McFadden, wherever llf is infinite (G112)."""
        _check_choice(kind, "pseudo_rsquared", "kind", ("cs", "mcf"))

        def make():
            if kind == "cs":
                v = -_expm1(-_div(self.llr, self.nobs))
            else:
                # an infinite llf (a zero scale, G97) has no ratio: NaN (G112); Cox-Snell
                # keeps its limit through exp
                llf, lln = self.llf, self.llnull
                v = 1.0 - _div(llf, lln)
                if isinstance(v, _sd.ndarray):
                    v = _sd.where(_sd.logical_or(_sd.isinf(llf), lln == 0.0), math.nan, v)
                elif math.isinf(llf) or lln == 0.0:
                    v = math.nan
            return self._nan_where(v)

        return self._cached(("pseudo_rsquared", kind), make)

    @property
    def llr_pvalue(self):
        """llr against chi-square with df_model degrees of freedom."""

        def make():
            df = self.df_model
            if self._batch:
                df = _sd.where(df > 0, df, math.nan)
            elif not df > 0:
                df = math.nan
            return self._scalar(_ChiSquared(df=_arg(df)).ccdf(_arg(self.llr)))

        return self._cached("llr_pvalue", make)

    @property
    def converged(self):
        v = self._raw[6][..., _G_CONV]
        return v == 1.0 if self._batch else float(v) == 1.0

    @property
    def n_iter(self):
        v = self._raw[6][..., _G_NITER]
        if self._batch:
            return _sd.where(_sd.isnan(v), 0.0, v).astype(_sd.int64)
        v = float(v)
        return 0 if math.isnan(v) else int(v)

    @property
    def separation(self):
        v = self._raw[6][..., _G_SEP]
        return v == 1.0 if self._batch else float(v) == 1.0

    # -- inference (G38, G39): the normal for a fixed scale; t(df_resid) for an estimated one.
    # Robust HC0-HC3 and HAC keep the normal for the fixed-scale families (binomial, poisson
    # without an estimated scale: R's coeftest, Stata) and t(df_resid) for the others; cluster
    # is t(G - 1) (G27)
    def _normal(self):
        if self.cov_type == "cluster":
            return False
        if self._robust:
            return self._scale_fixed and self.family in ("binomial", "poisson")
        return self._scale_fixed

    def _df_inference(self):
        if self._normal():
            return self._nan_where(math.inf) if not self._batch else _sd.where(self._isnan(), math.nan, math.inf)
        return super()._df_inference()

    def _dist(self, nd=1):
        if self._normal():
            return _Normal()
        return _StudentT(df=self._per(self._df_inference(), nd))

    def _cov_factor(self):
        if self._robust:
            return None

        def make():
            from ._tests import _unit_triangle

            R = self._raw[2]
            fx = self._xfactor()
            if fx is not None:  # back to the fit's units (powers of two, exact)
                R = R / _sd.expand_dims(fx, axis=-2)
            s = self._stat(_G_SCALE)
            return _unit_triangle(R, self._raw[5]), _sd.sqrt(s) if isinstance(s, _sd.ndarray) else math.sqrt(
                s
            ) if s >= 0 else math.nan

        return self._cached("cov_factor", make)

    # -- influence (G7a decision 14): the linear models' measures of the final IRLS's weighted
    # least squares, with the Pearson residuals
    def _influence(self):
        from . import _diagnostics

        return self._cached("influence", lambda: _diagnostics.glm_influence(self))

    @property
    def leverage(self):
        """The hat values (..., n) of sqrt(W) X, W the final IRLS weights (the expected
        information's, times the frequency weights; R's hatvalues of a glm). 0 for the rows out
        of the fit."""
        from . import _diagnostics

        if "influence" in self._cache:
            return self._influence()[0]
        return self._cached("leverage", lambda: _diagnostics.glm_leverage(self))

    @property
    def cooks_distance(self):
        """e_i^2 h_i / (scale rank (1 - h_i)^2), e the Pearson residual (R's cooks.distance of a
        glm, statsmodels' GLMInfluence.cooks_distance[0])."""
        return self._influence()[1]

    @property
    def dffits(self):
        """e_i sqrt(h_i) / (s_(i) (1 - h_i)), e the Pearson residual, s_(i) the scale without row
        i for a family whose scale is estimated (R's dffits of a glm), sqrt(scale) for a fixed one."""
        return self._influence()[2]

    @property
    def dfbetas(self):
        """(..., n, k): the one-step change of the parameters when row i is deleted, (X~^T X~)^-1
        x~_i e_i / (1 - h_i), over s_(i) sqrt([(X~^T X~)^-1]_jj) (dffits' s_(i)); NaN for the
        aliased parameters."""
        return self._influence()[3]

    # -- nested models (G7a decision 8)
    def compare_f_test(self, small, /):
        """CompareFTestResult of this GLM against the nested `small` one: R's anova.glm F,
        ((D_s - D_b) / (r_b - r_s)) / scale_b against F(r_b - r_s, df_resid_b). Only for an
        estimated scale; with a fixed one the deviance's test is compare_lr_test()."""
        from . import _tests

        return _tests.compare_f_test(self, small)

    def compare_lr_test(self, small, /):
        """CompareLRTestResult of this GLM against the nested `small` one: -2 (llf_s - llf_b)
        against chi-square(r_b - r_s), each model's llf (statsmodels)."""
        from . import _tests

        return _tests.compare_lr_test(self, small)

    # -- prediction
    def _offset(self, offset, exposure, func, m):
        o = None
        for name, v in (("offset", offset), ("exposure", exposure)):
            if v is None:
                continue
            a, _ = _as_real(v, func, name, per_row=f"{_values_text(m)} like the rows of 'X_new'")
            if a.shape[-1] != m:
                raise ValueError(
                    f"{func}(): '{name}' must have {_values_text(m)} like the rows of 'X_new', not {a.shape[-1]}"
                )
            if a.size and bool(_sd.any(_sd.isinf(a))):  # as X_new's (G91)
                raise ValueError(f"{func}(): '{name}' contains infinity")
            if name == "exposure":
                if self.link != "log":
                    raise ValueError(f"{func}(): 'exposure' is only used with the log link")
                if bool(_sd.any(a <= 0)):
                    raise ValueError(f"{func}(): 'exposure' must be > 0")
                a = _sd.log(a)
            o = a if o is None else o + a
        return o

    def predict(self, X_new, /, *, offset=None, exposure=None, which="mean"):
        """The mean g^-1(eta) (which='mean') or the linear predictor eta = X_new params + offset
        (+ log exposure) (which='linear') at the rows of X_new. Aliased columns take no part."""
        func = "predict"
        _check_choice(which, func, "which", ("mean", "linear"))
        Xn = self._design(X_new, func)
        eta = self._xbeta(Xn, func)
        o = self._offset(offset, exposure, func, Xn.shape[-2])
        if o is not None:
            eta = eta + o
        return eta if which == "linear" else _inverse_link(self.link, eta)

    def predict_interval(self, X_new, /, *, offset=None, exposure=None, alpha=0.05, which="mean"):
        """(..., m, 2): the 1 - alpha interval for the linear predictor at each row of X_new,
        eta -+ q se(eta) (q the inference distribution's quantile), mapped through the inverse
        link for which='mean' (lower, upper in order)."""
        func = "predict_interval"
        alpha = _check_alpha(alpha, func)
        _check_choice(which, func, "which", ("mean", "linear"))
        Xn = self._design(X_new, func)
        eta = self._xbeta(Xn, func)
        o = self._offset(offset, exposure, func, Xn.shape[-2])
        if o is not None:
            eta = eta + o
        var = self._mean_var_s(Xn)
        q = self._dist().iccdf(alpha / 2)
        half = q * _sd.sqrt(var) * self._per(self._unit(), 1)
        lo, hi = eta - half, eta + half
        if which == "mean":
            lo, hi = _inverse_link(self.link, lo), _inverse_link(self.link, hi)
            lo, hi = _sd.minimum(lo, hi), _sd.maximum(lo, hi)  # the inverse link decreases
        return _sd.stack([lo, hi], axis=-1)

    # -- summary
    def summary(self, *, alpha=0.05):
        """A text table like statsmodels' GLM summary: the fit, then the coefficients with their
        standard errors, z (or t), p and 1 - alpha intervals; a batch as ols's (G193)."""
        alpha = _check_alpha(alpha, "summary")
        if self._batch:
            return _batch(self)
        return _summary(self, alpha)

    def __repr__(self):
        name = type(self).__name__
        if self._batch:
            return f"{name}(batch shape {tuple(self._batch)}, {self.params.shape[-1]} parameters)"
        fields = [("family", self.family), ("link", self.link), ("nobs", float(self.nobs)), ("rank", int(self.rank))]
        return _S.one_line(name, fields + [("deviance", float(self.deviance)), ("params", self.params)])


def _summary(r, alpha):
    nan = r._isnan()
    prsq = r.pseudo_rsquared()  # Cox-Snell: 1 - exp((llnull - llf) 2/n)
    left = [
        ("Model:", "GLM"),
        ("Model Family:", _FAMILY_NAME[r.family]),
        ("Link Function:", _LINK_NAME[r.link]),
        ("Method:", "IRLS"),
        ("No. Iterations:", str(r.n_iter)),
        ("Covariance Type:", r.cov_type),
    ]
    right = [
        ("No. Observations:", _count(r.nobs)),
        ("Df Residuals:", _count(r.df_resid)),
        ("Df Model:", _count(r.df_model)),
        ("Scale:", f"{r.scale:#8.5g}".strip()),
        ("Log-Likelihood:", f"{r.llf:#8.5g}".strip()),
        ("Deviance:", f"{r.deviance:#8.5g}".strip()),
        ("Pearson chi2:", f"{r.pearson_chi2:#6.3g}".strip()),
        ("Pseudo R-squ. (CS):", f"{prsq:#6.4g}".strip()),
    ]
    names = [str(v) for v in r.param_names]
    table = _coef_table(r, alpha, "z" if r._normal() else "t")
    notes = []
    ct = r.cov_type
    if ct == "HAC":
        notes.append(
            f"Standard errors are heteroscedasticity and autocorrelation robust (HAC) using "
            f"{_count(r._stat(_G_LAGS))} lags (Bartlett kernel) and without small sample correction."
        )
    elif ct == "cluster":
        g = r._stat(_G_NGROUPS)
        notes.append(
            f"Standard errors are robust to cluster correlation (cluster) with {_count(g)} groups; "
            f"t uses {_count(g - 1)} degrees of freedom."
        )
    elif ct == "nonrobust":
        notes.append(
            "Standard errors are from the expected (Fisher) information"
            + ("." if r._scale_fixed else " and the estimated scale.")
        )
    else:
        notes.append(_COV_NOTES[ct])
    if not nan:
        notes += _aliased_note(r, names)
        if r.separation:
            notes.append(
                "Separation: the maximum likelihood estimate does not exist (some fitted values are at the "
                "boundary); the coefficients and their inference are NaN, the fit is its limit."
            )
        if not r.converged:
            notes.append(f"The IRLS did not converge in {r.n_iter} iterations.")
    return _S.render("Generalized Linear Model Regression Results", header=(left, right), tables=[table], notes=notes)


def _batch(r):
    left = [
        ("Model:", "GLM"),
        ("Model Family:", _FAMILY_NAME[r.family]),
        ("Link Function:", _LINK_NAME[r.link]),
        ("Method:", "IRLS"),
        ("Covariance Type:", r.cov_type),
    ]
    stats = [
        ("nobs", _S.count),
        ("deviance", _S.forg),
        ("llf", _S.forg),
        ("n_iter", str),
        ("converged", _S.flag),
    ]
    return _batch_summary(r, "Generalized Linear Model Regression Results", left, stats)


# ---- the fit


def _check_int(v, func, arg, lo):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {_sd._type_name(v)}")
    v = operator.index(v)
    if v < lo:
        raise ValueError(f"{func}(): '{arg}' must be >= {lo}")
    if v >= 2**31:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _check_positive(v, func, arg):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {_sd._type_name(v)}")
    try:
        x = float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"{func}(): '{arg}' must be a finite number > 0")
    return x


def _check_scale(v, func):
    """(mode, value): None (the family's default), 'X2' (Pearson chi2 / df), 'dev' (deviance /
    df) or a fixed number > 0."""
    if v is None:
        return 0, 0.0
    if isinstance(v, str):
        if v == "X2":
            return 1, 0.0
        if v == "dev":
            return 2, 0.0
        raise ValueError(f"{func}(): 'scale' must be None, 'X2', 'dev' or a number > 0, not {v!r}")
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): 'scale' must be None, 'X2', 'dev' or a number > 0, not {_sd._type_name(v)}")
    return 3, _check_positive(v, func, "scale")


def _expm1(x):
    if isinstance(x, _sd.ndarray):
        return _sd.expm1(x)
    return math.expm1(x) if not math.isnan(x) else math.nan


def _broadcast_shape(func, a, b):
    """The broadcast of two batch shapes (the per-row arrays' batch axes)."""
    nd = max(len(a), len(b))
    a, b = (1,) * (nd - len(a)) + a, (1,) * (nd - len(b)) + b
    out = []
    for x, z in zip(a, b):
        if x != z and x != 1 and z != 1:
            raise ValueError(f"{func}(): the batch shapes {a} and {b} of the per-row arrays do not broadcast")
        out.append(z if x == 1 else x)
    return tuple(out)


@_calls.function
def glm(
    y,
    X,
    /,
    *,
    family,
    link=None,
    var_weights=None,
    freq_weights=None,
    trials=None,
    offset=None,
    exposure=None,
    intercept=True,
    scale=None,
    cov_type="nonrobust",
    maxlags=None,
    groups=None,
    max_iter=100,
    tol=1e-8,
    rtol=None,
    nan_policy="propagate",
):
    """Generalized linear model of y (..., n) on X (..., n, k) or (n,), an intercept column
    first, fitted by IRLS (docs/api/regression.md §3)."""
    func = "glm"
    _check_choice(family, func, "family", _FAMILIES)
    links = _FAMILY_LINKS[family]
    if link is None:
        link = links[0]
    elif not isinstance(link, str) or link not in links:
        listed = ", ".join(f"'{c}'" for c in links[:-1]) + f" or '{links[-1]}'"
        what = repr(link) if isinstance(link, str) else _sd._type_name(link)
        err = ValueError if isinstance(link, str) else TypeError
        raise err(f"{func}(): 'link' must be None, {listed} for the {family} family, not {what}")
    _check_bool(intercept, func, "intercept")
    _check_choice(cov_type, func, "cov_type", _COV_TYPES)
    lags = _check_maxlags(maxlags, func, cov_type)
    rt = _check_rtol(rtol, func)
    _check_policy(nan_policy, func)
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_positive(tol, func, "tol")
    smode, svalue = _check_scale(scale, func)
    if trials is not None and family != "binomial":
        raise ValueError(f"{func}(): 'trials' is only used with the binomial family")
    if exposure is not None and link != "log":
        raise ValueError(f"{func}(): 'exposure' is only used with the log link")

    yy, own_y = _read_y_owned(y, func)
    if yy.ndim == 0:
        raise ValueError(f"{func}(): 'y' must have at least 1 dimension")
    columns = _frame_columns(X)
    XX, names, own_x = _as_real_owned(X, func, "X")
    if XX.ndim == 0:
        raise ValueError(f"{func}(): 'X' must have at least 1 dimension")
    if XX.ndim == 1:
        XX = _sd.reshape(XX, (XX.shape[0], 1))
    n = yy.shape[-1]
    if XX.shape[-2] != n:
        raise ValueError(_rows_message(func, "X", n, XX.shape[-2], yy.shape))
    _check_params(func, XX.shape[-1], intercept)
    rows = {}
    for name, v in (
        ("var_weights", var_weights),
        ("freq_weights", freq_weights),
        ("trials", trials),
        ("offset", offset),
        ("exposure", exposure),
    ):
        if v is None:
            continue
        a, _ = _as_real(v, func, name, per_row=f"{n} value{'' if n == 1 else 's'} like 'y'")
        if a.shape[-1] != n:
            raise ValueError(_rows_message(func, name, n, a.shape[-1], yy.shape, "values"))
        rows[name] = a
    if "exposure" in rows:
        e = rows["exposure"]
        if bool(_sd.any(e <= 0)):
            raise ValueError(f"{func}(): 'exposure' must be > 0")
        if bool(_sd.any(_sd.isinf(e))):
            raise ValueError(f"{func}(): 'exposure' contains infinity")
    codes, gtot, gmiss = _groups(groups, func, n, cov_type)
    named = [("y", yy.shape[:-1]), ("X", XX.shape[:-2])] + [(k, a.shape[:-1]) for k, a in rows.items()]
    if codes is not None:
        named.append(("groups", codes.shape[:-1]))
    _check_batches(func, named)
    batch = (
        yy.ndim > 1 or XX.ndim > 2 or any(a.ndim > 1 for a in rows.values()) or (codes is not None and codes.ndim > 1)
    )
    if nan_policy == "raise":
        if _has_nan(yy, XX, *rows.values()) or gmiss is not None:
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
    elif nan_policy == "omit":
        if batch:
            raise ValueError(
                f"{func}(): nan_policy='omit' takes one regression, not a batch "
                "(the regressions would keep different rows)"
            )
        # inf and invalid weights stay errors in a row that also has a NaN (G24)
        _check_rows(func, [("y", yy), ("X", XX)] + list(rows.items()), nonneg=("var_weights", "freq_weights"))
        if "trials" in rows and bool(_sd.any(rows["trials"] <= 0)):
            raise ValueError(f"{func}(): 'trials' must be finite and > 0")
        bad = _sd.logical_or(_sd.isnan(yy), _sd.any(_sd.isnan(XX), axis=1))
        for a in rows.values():
            bad = _sd.logical_or(bad, _sd.isnan(a))
        if gmiss is not None:
            bad = _sd.logical_or(bad, gmiss)
        if bool(_sd.any(bad)):
            keep = _sd.logical_not(bad)
            yy, XX = yy[keep], XX[keep]
            own_y = own_x = True  # the selection is new memory
            rows = {k: a[keep] for k, a in rows.items()}
            if codes is not None:
                codes = codes[keep]
            n = yy.shape[-1]
    off = rows.get("offset")
    if "exposure" in rows:
        le = _sd.log(rows["exposure"])
        off = le if off is None else off + le
    parts = [rows.get("var_weights"), rows.get("freq_weights"), rows.get("trials"), off]
    shape = ()
    for a in parts:
        if a is not None:
            shape = _broadcast_shape(func, shape, tuple(a.shape[:-1]))
    fills = (1.0, 1.0, 1.0, 0.0)
    aux = _sd.stack(
        [_sd.broadcast_to(a if a is not None else _sd.full((n,), fills[i]), shape + (n,)) for i, a in enumerate(parts)],
        axis=-2,
    )
    # data far outside [2^-256, 2^256] is fitted in power-of-two units (exact), as the linear
    # models: the columns for every family (beta_j fx_j), y for the families whose model is
    # equivariant under its scale (gaussian, gamma: mu / fy; the log link through the offset
    # -log fy, which leaves beta). Data inside the range is fitted as given.
    fy = fx = None
    n = yy.shape[-1]
    if n:
        if family in ("gaussian", "gamma"):
            fy = _pow2(_sd.max(_sd.abs(yy), axis=-1))
        if XX.shape[-1]:
            fx = _pow2(_sd.max(_sd.abs(XX), axis=-2))
    fy1 = None
    if fy is not None:
        fy1 = _sd.expand_dims(fy, axis=-1) if isinstance(fy, _sd.ndarray) else fy
        yy = yy / fy1
        own_y = True
        if link == "log":
            # mu / fy = exp(eta - log fy): the offset carries the factor
            if isinstance(fy1, _sd.ndarray):
                lf = _sd.log(fy1)
                off = _sd.broadcast_to(0.0 - lf, tuple(lf.shape[:-1]) + (n,)) if off is None else off - lf
            else:
                lf = math.log(fy1)
                off = _sd.full((n,), -lf) if off is None else off - lf
            parts[3] = off
            shape = _broadcast_shape(func, shape, tuple(off.shape[:-1]))
            aux = _sd.stack(
                [
                    _sd.broadcast_to(a if a is not None else _sd.full((n,), fills[i]), shape + (n,))
                    for i, a in enumerate(parts)
                ],
                axis=-2,
            )
    XK = XX if fx is None else XX / _sd.expand_dims(fx, axis=-2)
    raw = _k.glm_fit(
        func,
        yy,
        XK,
        aux,
        codes,
        gtot,
        _FAMILY_CODE[family],
        _LINK_CODE[link],
        parts[0] is not None,
        parts[1] is not None,
        parts[2] is not None,
        off is not None,
        intercept,
        rt,
        _COV_TYPES.index(cov_type),
        lags,
        max_iter,
        tol,
        smode,
        svalue,
    )
    extra = _glm_unscale(list(raw), fy, fx, family, link, intercept) if (fy is not None or fx is not None) else None
    if extra is not None:
        raw, extra = extra
    fixed = smode == 3 or (smode == 0 and family in ("binomial", "poisson"))
    # what the comparisons, the analysis of variance and the influence measures read later:
    # arrays of the result's own (G86)
    fitargs = {
        "y": yy if own_y else yy.copy(),
        "X": XK if (fx is not None or own_x) else XK.copy(),
        "aux": aux,
        "has": (parts[0] is not None, parts[1] is not None, parts[2] is not None, off is not None),
        "family": family,
        "link": link,
        "family_code": _FAMILY_CODE[family],
        "link_code": _LINK_CODE[link],
        "rtol": rt,
        "max_iter": max_iter,
        "tol": tol,
        "smode": smode,
        "svalue": svalue,
    }
    return GLMResult(
        raw,
        extra=extra,
        family=family,
        link=link,
        scale_fixed=fixed,
        param_names=_names(names, XX.shape[-1], intercept),
        intercept=intercept,
        cov_type=cov_type,
        batch=tuple(raw[0].shape[:-1]),
        columns=columns,
        fitargs=fitargs,
    )


def _glm_unscale(raw, fy, fx, family, link, intercept):
    """raw (glm_fit's list) from the fit's units to the data's: the kernel fitted y / fy (fy
    None: y as given) on the columns X_j / fx_j (fx None: as given). The parameters become
    beta' u / fx_j with u the unit of the linear predictor (fy for the identity link, 1 / fy
    for the inverse, 1 for the log link, whose factor the offset carried); mu, the response
    residuals and, for the gaussian, the Pearson and deviance residuals scale by fy; the
    covariance and the stats stay (the result scales them as it reads them). Returns (raw,
    extra)."""
    params = raw[0]
    batch = tuple(params.shape[:-1])
    k = params.shape[-1]
    if fx is None:
        fxk = _sd.ones(batch + (k,))
    else:
        fxk = _sd.concat([_sd.ones(tuple(fx.shape[:-1]) + (1,)), fx], axis=-1) if intercept else fx
        fxk = _sd.broadcast_to(fxk, batch + (k,))
    fyv = 1.0 if fy is None else fy
    if isinstance(fyv, _sd.ndarray):
        fyb = _sd.broadcast_to(fyv, batch)
    else:
        fyb = _sd.full(batch, fyv) if batch else fyv
    u = 1.0 if link == "log" else (1.0 / fyb if link == "inverse" else fyb)
    if fy is None:
        u = _sd.full(batch, 1.0) if batch else 1.0
    extra = {"fy": u, "fx": fxk, "beff_params_s": params}
    u1 = _expand(u, 1) if isinstance(u, _sd.ndarray) else u
    raw[0] = params * u1 / fxk
    raw[2] = raw[2] * _sd.expand_dims(fxk, axis=-2)
    if fy is not None:
        f1 = _expand(fyb, 1) if isinstance(fyb, _sd.ndarray) else fyb
        raw[3] = raw[3] * f1  # mu
        if link == "log":
            raw[4] = raw[4] + (_sd.log(f1) if isinstance(f1, _sd.ndarray) else math.log(f1))
        else:
            raw[4] = raw[4] * u1  # eta
        raw[7] = raw[7] * f1  # y - mu
        if family == "gaussian":
            raw[8] = raw[8] * f1  # (y - mu) sqrt(m)
            raw[9] = raw[9] * f1
            extra["dev2"] = fyb * fyb
        # (y - mu) g'(mu): fy for the identity link, 1 / fy for the inverse, 1 for the log
        if link == "identity":
            raw[10] = raw[10] * f1
        elif link == "inverse":
            raw[10] = raw[10] / f1
        # -nobs log fy: the gaussian's log sigma^2 and the gamma's -log y
        nobs = raw[6][..., _G_NOBS]
        lf = _sd.log(fyb) if isinstance(fyb, _sd.ndarray) else math.log(fyb)
        extra["llshift"] = (0.0 - nobs) * lf if batch else -float(nobs) * lf
    return raw, extra


# the methods' call-shape errors in the library's form (R89)
_calls.methods(GLMResult)
