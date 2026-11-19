"""Quantile regression: quantreg (docs/api/regression.md §6, docs/design/regression.md §3.6).

The fit is the native kernel stardust._core.regression.quantreg_fit (the Frisch-Newton
interior point method, the crossover to a basis, simplex pivots to a certified optimal
vertex, the vertex rule of non-unique solutions, R's summary.rq covariances). This module
checks and converts the inputs, computes quantreg's bandwidths and builds the result.
"""

import math
import textwrap

from .. import _core as _sd
from .. import special as _special
from ._glm import _check_int, _check_positive
from ._linear import (
    _WIDTH,
    _call_errors,
    _check_alpha,
    _check_bool,
    _check_choice,
    _check_policy,
    _check_pred_batch,
    _check_rtol,
    _count,
    _div,
    _forg,
    _frame_columns,
    _fresh,
    _names,
    _new_design,
    _number_or_1d,
    _pairs,
    _prepare,
    _quantile_label,
    _RegressionResult,
)

_k = _sd.regression

_COV_TYPES = ("iid", "nid", "ker")
_BANDWIDTHS = ("hall-sheather", "bofinger")
_BW_ALPHA = 0.05  # bandwidth.rq's alpha (the Hall-Sheather rule's interval level)

# the kernel's per-(regression, quantile) scalars (csrc/regression/quantile.cpp, enum QStat)
(_Q_NOBS, _Q_RANK, _Q_OBJ, _Q_NULL, _Q_NITER, _Q_NPIV, _Q_CONV, _Q_NONUNIQUE, _Q_ISNAN, _Q_BW, _Q_NGROUPS, _Q_LAGS) = (
    range(12)
)

_COV_NOTES = {
    "iid": "Standard errors assume iid errors (Koenker-Bassett: the sparsity from a median regression on the "
    "residuals' order statistics around q, as R's summary.rq se='iid').",
    "nid": "Standard errors are the Hendricks-Koenker sandwich (the sparsity from the fits at q +- bandwidth, as "
    "R's summary.rq se='nid').",
    "ker": "Standard errors are Powell's kernel sandwich (R's summary.rq se='ker').",
}


class QuantRegResult(_RegressionResult):
    """The result of sd.regression.quantreg: statsmodels' names (G10). The intercept is params[0]."""

    __slots__ = ("_q", "_bw_rule", "_qaxis")
    _I_NOBS, _I_RANK, _I_ISNAN, _I_NGROUPS, _I_LAGS = _Q_NOBS, _Q_RANK, _Q_ISNAN, _Q_NGROUPS, _Q_LAGS

    def __init__(self, raw, *, q, bw_rule, qaxis, **kw):
        super().__init__(raw, **kw)
        self._q = q
        self._bw_rule = bw_rule
        self._qaxis = qaxis

    @property
    def q(self):
        """The quantile(s) as given: a number, or the 1-D array (a new copy at each access,
        G113; read-only, G98)."""
        return _fresh(self._q)

    @property
    def resid(self):
        return self._raw[4]

    @property
    def df_model(self):
        return self._cached(
            "df_model", lambda: self._nan_where(self._stat(_Q_RANK) - (1.0 if self._intercept else 0.0))
        )

    @property
    def df_resid(self):
        return self._cached("df_resid", lambda: self._nan_where(self.nobs - self._stat(_Q_RANK)))

    @property
    def objective(self):
        """sum rho_q(resid_i), rho_q(r) = r (q - [r < 0]) (summed in double-double)."""
        return self._stat(_Q_OBJ)

    @property
    def prsquared(self):
        """Koenker and Machado's (1999) R1: 1 - objective / the objective of the intercept alone
        at its optimum (the type-1 sample quantile of y); without an intercept, of the zero
        model (sum rho_q(y_i))."""
        return self._cached("prsquared", lambda: 1.0 - _div(self.objective, self._stat(_Q_NULL)))

    @property
    def bandwidth(self):
        """The bandwidth h of quantreg's bandwidth.rq at q and nobs (in units of q); for 'nid'
        and 'ker' halved until q - h >= 0 and q + h <= 1, as summary.rq."""
        return self._stat(_Q_BW)

    def _flag(self, i):
        v = self._raw[6][..., i]
        return v == 1.0 if self._batch else float(v) == 1.0

    @property
    def converged(self):
        """True when the returned vertex carries its optimality certificate (every basis row's
        dual variable in [q - 1, q])."""
        return self._flag(_Q_CONV)

    @property
    def nonunique(self):
        """True when the minimizers form more than a point (the objective's minimum is unique;
        params is the vertex of docs/api/regression.md §6's rule)."""
        return self._flag(_Q_NONUNIQUE)

    def _count_stat(self, i):
        v = self._raw[6][..., i]
        if self._batch:
            return _sd.where(_sd.isnan(v), 0.0, v).astype(_sd.int64)
        v = float(v)
        return 0 if math.isnan(v) else int(v)

    @property
    def n_iter(self):
        """The interior point method's iterations (0 for a quantile started from the previous
        one's basis)."""
        return self._count_stat(_Q_NITER)

    @property
    def n_pivots(self):
        """The simplex pivots from the interior point method's basis (or the previous
        quantile's) to the returned vertex."""
        return self._count_stat(_Q_NPIV)

    def _shape_X(self, Xn):
        if not self._qaxis:
            return Xn
        return _sd.reshape(Xn, tuple(Xn.shape[:-2]) + (1,) + tuple(Xn.shape[-2:]))

    def _design(self, X_new, func):
        p = self.params.shape[-1] - (1 if self._intercept else 0)
        Xn = _new_design(X_new, func, p, self._columns)
        # X_new's batch against the regressions' (the quantile axis is the result's own)
        _check_pred_batch(func, Xn, self._batch[:-1] if self._qaxis else self._batch)
        return Xn

    def predict(self, X_new, /):
        """X_new @ params (the intercept added), summed in double-double (..., [L,] m). Aliased
        columns take no part; a NaN in a column of the model gives NaN."""
        return self._xbeta(self._design(X_new, "predict"), "predict")

    def predict_interval(self, X_new, /, *, alpha=0.05):
        """(..., [L,] m, 2): the 1 - alpha confidence interval of the fitted q-quantile at each
        row of X_new, x params -+ t(df_resid) se, se^2 = x cov_params() x^T (G103). No interval
        for a new observation: that is the conditional distribution's spread, which a single
        quantile does not describe (fit q/2 and 1 - q/2 for one)."""
        return self._interval(X_new, "predict_interval", alpha)

    def summary(self, *, alpha=0.05):
        """A text table like statsmodels' QuantReg summary: the fit, then the coefficients with
        their standard errors, t, p and 1 - alpha intervals. One regression and one quantile only."""
        alpha = _check_alpha(alpha, "summary")
        if self._batch:
            raise ValueError(f"summary(): takes one regression at one quantile, not {self._shape_text()}")
        return _summary(self, alpha)

    def _shape_text(self):
        """The leading axes of the fields: the batch's, then the quantile axis of an array q."""
        batch = tuple(self._batch[:-1]) if self._qaxis else tuple(self._batch)
        parts = [f"batch shape {batch}"] if batch else []
        if self._qaxis:
            parts.append(f"{self._batch[-1]} quantiles")
        return " and ".join(parts)

    def __repr__(self):
        name = type(self).__name__
        if self._batch:
            return f"{name}({self._shape_text()}, {self.params.shape[-1]} parameters)"
        return (
            f"{name}(q={self.q:g}, nobs={self.nobs:g}, rank={self.rank}, objective={self.objective:.6g}, "
            f"params={self.params})"
        )


def _summary(r, alpha):
    stars = "=" * _WIDTH
    lines = ["QuantReg Regression Results".center(_WIDTH).rstrip(), stars]
    left = [
        ("Model:", "QuantReg"),
        ("Method:", "Frisch-Newton"),
        ("Quantile:", f"{r.q:g}"),
        ("Covariance Type:", r.cov_type),
        ("Bandwidth rule:", r._bw_rule),
    ]
    right = [
        ("Pseudo R-squared:", f"{r.prsquared:#8.4g}".strip()),
        ("Bandwidth:", f"{r.bandwidth:#8.4g}".strip()),
        ("No. Observations:", _count(r.nobs)),
        ("Df Residuals:", _count(r.df_resid)),
        ("Df Model:", _count(r.df_model)),
    ]
    lines += _pairs(left, right)
    lines.append(stars)
    names = [str(v) for v in r.param_names]
    ci = r.conf_int(alpha=alpha)
    params, bse, tv, pv = (r.params, r.bse, r.tvalues, r.pvalues)
    wname = max([len(v) for v in names] + [9]) + 1
    cols = ["coef", "std err", "t", "P>|t|", "[" + _quantile_label(alpha / 2), _quantile_label(1 - alpha / 2) + "]"]
    head = " " * wname + "".join(c.rjust(11) for c in cols)
    width = max(_WIDTH, len(head))
    lines[1] = "=" * width
    lines[-1] = "=" * width
    lines.append(head)
    lines.append("-" * width)
    for j, nm in enumerate(names):
        vals = [
            _forg(float(params[j]), 4),
            _forg(float(bse[j])),
            _forg(float(tv[j])),
            f"{float(pv[j]):#.3f}",
            _forg(float(ci[j, 0])),
            _forg(float(ci[j, 1])),
        ]
        lines.append(nm.ljust(wname) + "".join(v.rjust(11) for v in vals))
    lines.append("=" * width)
    notes = [_COV_NOTES[r.cov_type] + " t uses df_resid degrees of freedom."]
    if not r._isnan():
        if r.nonunique:
            notes.append(
                "The solution is not unique: params is the lexicographically smallest minimizer "
                f"(objective {r.objective:.6g})."
            )
        aliased = [nm for nm, k in zip(names, r._raw[5].tolist()) if not k]
        if aliased:
            notes.append("Aliased (linearly dependent on earlier columns), not estimated: " + ", ".join(aliased) + ".")
        if not r.converged:
            notes.append("The simplex did not reach a certified optimum.")
    lines.append("")
    lines.append("Notes:")
    for i, t in enumerate(notes):
        lines += textwrap.wrap(f"[{i + 1}] {t}", width, subsequent_indent="    ")
    return "\n".join(lines)


# ---- the fit


def _normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bandwidth(q, n, hall_sheather):
    """quantreg's bandwidth.rq(q, n, hs, alpha=0.05)."""
    x0 = float(_special.ndtri(q))
    f0 = _normal_pdf(x0)
    if hall_sheather:
        z = float(_special.ndtri(1.0 - _BW_ALPHA / 2.0))
        return n ** (-1.0 / 3.0) * z ** (2.0 / 3.0) * ((1.5 * f0 * f0) / (2.0 * x0 * x0 + 1.0)) ** (1.0 / 3.0)
    return n**-0.2 * ((4.5 * f0**4) / (2.0 * x0 * x0 + 1.0) ** 2) ** 0.2


def _check_q(q, func):
    """(list of floats, the value to keep in the result (a copy of an array), whether q is an
    array). The forms and messages of ridge's alpha (_number_or_1d)."""
    a, qaxis = _number_or_1d(q, func, "q")
    vals = [float(v) for v in a.tolist()]
    if not all(0.0 < v < 1.0 for v in vals):
        raise ValueError(f"{func}(): 'q' must be a number or a 1-dimensional array of numbers strictly between 0 and 1")
    return vals, (a.copy() if qaxis else vals[0]), qaxis


def quantreg(
    y,
    X,
    /,
    *,
    q=0.5,
    intercept=True,
    cov_type="nid",
    bandwidth="hall-sheather",
    max_iter=100,
    tol=1e-05,
    rtol=None,
    nan_policy="propagate",
):
    """Linear quantile regression of y (..., n) on X (..., n, k) or (n,), an intercept column
    first: minimizes sum rho_q(y_i - x_i b), rho_q(r) = r (q - [r < 0]) (Koenker and Bassett
    1978). q is a number or a 1-dimensional array (a quantile axis after the batch axes)
    (docs/api/regression.md §6)."""
    func = "quantreg"
    qs, qkeep, qaxis = _check_q(q, func)
    _check_bool(intercept, func, "intercept")
    _check_choice(cov_type, func, "cov_type", _COV_TYPES)
    _check_choice(bandwidth, func, "bandwidth", _BANDWIDTHS)
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_positive(tol, func, "tol")
    rt = _check_rtol(rtol, func)
    _check_policy(nan_policy, func)

    columns = _frame_columns(X)
    yy, XX, _, names, _, _, _ = _prepare(func, y, X, None, nan_policy, intercept=intercept)
    nobs = yy.shape[-1]
    hs = bandwidth == "hall-sheather"
    hraw, hhalf, kerc = [], [], []
    for v in qs:
        h = _bandwidth(v, nobs, hs) if nobs > 0 else math.nan
        hraw.append(h)
        while v - h < 0 or v + h > 1:
            h /= 2.0
        hhalf.append(h)
        kerc.append(float(_special.ndtri(v + h)) - float(_special.ndtri(v - h)) if math.isfinite(h) else math.nan)
    raw = _k.quantreg_fit(
        func, yy, XX, qs, intercept, rt, _COV_TYPES.index(cov_type), hraw, hhalf, kerc, max_iter, tol, qaxis
    )
    return QuantRegResult(
        raw,
        q=qkeep,
        bw_rule=bandwidth,
        qaxis=qaxis,
        param_names=_names(names, XX.shape[-1], intercept),
        intercept=intercept,
        cov_type=cov_type,
        batch=tuple(raw[0].shape[:-1]),
        columns=columns,
    )


# the methods' call-shape errors in the library's form (R89)
_call_errors(QuantRegResult)
