"""Nonlinear least squares: nls (docs/api/regression.md §7, docs/design/regression.md §3.7).

The iterations are sd.optimize's least squares kernel stardust._core.optimize.lsq_solve
(MINPACK's Levenberg-Marquardt without bounds, scipy's trust region reflective method with
them; MINPACK's forward differences), which calls back into this module for the residuals
sqrt(w) (f(x, *theta) - y) of the rows used and for a user Jacobian. After a converged
iteration, Gauss-Newton steps with an accurate Jacobian (the user's, or sd.optimize's 5-point
differences, optimize/_diff.py) solved by the linear models' QR kernel (with its refinement)
continue while the parameters move by more than rounding (G73). The covariance
sigma^2 (J^T W J)^-1 and the aliased parameters (G9, G21) come from the same kernel at the
solution.
"""

import inspect
import math
import numbers
import textwrap

from .. import _core as _sd
from .. import frame as _frame
from .. import linalg as _linalg
from ..optimize._diff import Differences as _Differences
from ._glm import _check_int
from ._linear import (
    _call_errors,
    _RegressionResult,
    _as_real,
    _check_alpha,
    _check_choice,
    _check_policy,
    _check_rows,
    _check_rtol,
    _count,
    _div0,
    _frame_columns,
    _read_y,
    _real,
    _rows_message,
    _div,
    _forg,
    _log,
    _pairs,
    _quantile_label,
    _type_name,
    _WIDTH,
)

_k = _sd.regression

_METHODS = ("lm", "trf")
_EPS = 2.0**-52

# the result's scalars (NLSResult._raw[6])
(
    _S_NOBS,
    _S_RANK,
    _S_SSR,
    _S_ISNAN,
    _S_NGROUPS,
    _S_LAGS,
    _S_NFEV,
    _S_NJEV,
    _S_NITER,
    _S_STATUS,
    _S_NPOLISH,
    _S_SUMLOGW,
) = range(12)
_NSTAT = 12
_LOG_2PI = math.log(2 * math.pi)

# the kernel's stopping codes (csrc/optimize/lsq.cpp, enum Status)
_ST_MAXNFEV, _ST_GTOL, _ST_FTOL, _ST_XTOL, _ST_BOTH, _ST_JAC, _ST_START = 0, 1, 2, 3, 4, -1, -2
_STATUS_TEXT = {
    _ST_MAXNFEV: "the iteration reached max_nfev",
    _ST_GTOL: "the iteration stopped on the gradient test (gtol)",
    _ST_FTOL: "the iteration stopped on the relative reduction of the sum of squares (ftol)",
    _ST_XTOL: "the iteration stopped on the relative step (xtol)",
    _ST_BOTH: "the iteration stopped on both ftol and xtol",
}

# Gauss-Newton steps after the iteration (G73): at most this many; they stop when a step is
# below 2^-50 of the parameters (componentwise), or the ratio of two steps predicts the rest
# below that, or a step does not shrink (or shrinks by less than half below 2^-40: the
# rounding floor)
_POLISH_STEPS = 10
_POLISH_TOL = 2.0**-50
_POLISH_FLOOR = 2.0**-40
# rtol's default (G21's aliased columns) for a difference Jacobian: its columns are accurate to
# about 1e-12 - 1e-11 relative, so an exactly dependent column shows |R_jj| of that size; the
# default with a user jac is the linear models' max(nobs, p) eps
_FD_RTOL = 2.0**-30


def _param_names(f, p):
    """f's parameter names after x when it has exactly p plain positional ones, else b1 .. bp."""
    try:
        sig = inspect.signature(f)
    except TypeError, ValueError:
        sig = None
    if sig is not None:
        ps = list(sig.parameters.values())
        kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        if len(ps) == p + 1 and all(q.kind in kinds for q in ps):
            return [q.name for q in ps[1:]]
    return [f"b{j + 1}" for j in range(p)]


class NLSResult(_RegressionResult):
    """The result of sd.regression.nls: the names of the other result objects (G10, R's
    nls for the model). params are in f's order of the parameters."""

    __slots__ = ("_method", "_f", "_xshape", "_jac_kind", "_active", "_model", "_polish_cut")
    _I_NOBS, _I_RANK, _I_ISNAN, _I_NGROUPS, _I_LAGS = _S_NOBS, _S_RANK, _S_ISNAN, _S_NGROUPS, _S_LAGS

    def __init__(self, raw, *, method, f, xshape, jac_kind, active, model=None, polish_cut=False, **kw):
        super().__init__(raw, intercept=False, cov_type="nonrobust", batch=(), **kw)
        self._method = method
        self._f = f
        self._xshape = xshape  # x's shape after the first axis
        self._jac_kind = jac_kind  # 'jac' or 'differences'
        self._active = active  # parameters at a bound (trf), by index
        self._model = model  # the fit's _Model (jac, bounds, the differences' steps)
        self._polish_cut = polish_cut  # max_nfev stopped the polish before rounding (G93)

    @property
    def method(self):
        """'lm' or 'trf' (read-only, G98)."""
        return self._method

    @property
    def resid(self):
        """y - f(x, *params) for every row (zero weight rows too)."""
        return self._raw[4]

    @property
    def ssr(self):
        """sum w_i (y_i - f(x_i, *params))^2 over the rows used."""
        return self._stat(_S_SSR)

    @property
    def scale(self):
        """sigma^2 = ssr / df_resid (R's nls: sigma(fit)^2); NaN for df_resid = 0 (0/0, G95)."""
        return self._cached("scale", lambda: _div0(self.ssr, self.df_resid))

    @property
    def tvalues(self):
        """params / bse; NaN for a perfect fit (ssr exactly 0 with df_resid > 0: bse is 0, and
        theta / 0 would show a significance the fit cannot have, as the slopes of rounding size
        a model like a + b x keeps for a constant y; G109)."""

        def make():
            t = self.params / self.bse
            if self.ssr == 0.0 and self.df_resid > 0:
                return t * math.nan
            return t

        return self._cached("tvalues", make)

    @property
    def llf(self):
        """The Gaussian log-likelihood at its maximum over sigma, as ols / wls:
        -nobs/2 (log 2 pi + log(ssr / nobs) + 1) + 1/2 sum log w_i (the rows used; R's
        logLik.nls)."""

        def make():
            n = self.nobs
            return -n / 2 * (_LOG_2PI + _log(_div(self.ssr, n)) + 1.0) + self._stat(_S_SUMLOGW) / 2

        return self._cached("llf", make)

    @property
    def aic(self):
        """-2 llf + 2 rank (the scale is not counted, as ols; R's AIC counts it: 2 more)."""
        return self._cached("aic", lambda: -2.0 * self.llf + 2.0 * self._stat(_S_RANK))

    @property
    def bic(self):
        """-2 llf + log(nobs) rank (R's BIC: log(nobs) more)."""
        return self._cached("bic", lambda: -2.0 * self.llf + _log(self.nobs) * self._stat(_S_RANK))

    @property
    def df_model(self):
        """The number of estimated (not aliased) parameters, rank."""
        return self._cached("df_model", lambda: self._nan_where(self._stat(_S_RANK)))

    @property
    def df_resid(self):
        return self._cached("df_resid", lambda: self._nan_where(self.nobs - self._stat(_S_RANK)))

    def _count_stat(self, i):
        v = float(self._raw[6][i])
        return 0 if math.isnan(v) else int(v)

    @property
    def converged(self):
        """True when the iteration stopped on one of its tests (ftol, xtol or gtol) at a point
        whose sum of squares is finite and the polish finished; False when it reached max_nfev
        (the last iterate is returned, G32; the polish's calls count against it too, G93) or
        when ssr overflows (G89: the tests can pass far from any solution when the Jacobian's
        scaling is extreme; scipy reports success there)."""
        return self._count_stat(_S_STATUS) > 0 and math.isfinite(self.ssr) and not self._polish_cut

    @property
    def n_iter(self):
        """The iterations of the trust-region method: the Jacobians it evaluated (p0's included)."""
        return self._count_stat(_S_NITER)

    @property
    def nfev(self):
        """Calls of f: the iteration's (its forward differences included), the Gauss-Newton
        steps' and the accurate Jacobian's."""
        return self._count_stat(_S_NFEV)

    @property
    def njev(self):
        """Jacobians evaluated: calls of jac, or difference Jacobians."""
        return self._count_stat(_S_NJEV)

    def _xnew(self, x_new, func):
        """x_new like the fit's x: the same columns (a DataFrame's names and order when both
        are DataFrames, G90; else by position), no inf (G91)."""
        if self._columns is not None and isinstance(x_new, _frame.DataFrame):
            got = [str(c) for c in x_new.columns]
            if got != self._columns:
                what = "in the fit's order" if sorted(got) == sorted(self._columns) else "like the fit's 'x'"
                raise ValueError(f"{func}(): 'x_new' must have the columns {self._columns} {what}, not {got}")
        xn, _ = _as_real(x_new, func, "x_new")
        if xn.ndim == 0:
            raise ValueError(f"{func}(): 'x_new' must have at least 1 dimension")
        if tuple(xn.shape[1:]) != self._xshape:
            want = "1-dimensional" if not self._xshape else f"{self._xshape[0]} columns"
            got = "1-dimensional" if xn.ndim == 1 else f"shape {tuple(xn.shape)}"
            raise ValueError(f"{func}(): 'x_new' must be {want} like the fit's 'x', not {got}")
        if xn.size and bool(_sd.any(_sd.isinf(xn))):
            raise ValueError(f"{func}(): 'x_new' contains infinity")
        return xn

    def predict(self, x_new, /):
        """f(x_new, *params): x_new is (m,) or (m, d) like the fit's x."""
        func = "predict"
        xn = self._xnew(x_new, func)
        m = xn.shape[0]
        if self._isnan():
            return _sd.full((m,), math.nan)
        return _values(self._f(xn, *[float(v) for v in self.params.tolist()]), m, func, "f", "'x_new'")

    def predict_interval(self, x_new, /, *, alpha=0.05):
        """(m, 2): the 1 - alpha confidence interval of the fitted value f(x_new, params) by the
        delta method, f -+ t(df_resid) se, se^2 = J cov_params() J^T with J = df/dtheta at the
        rows of x_new and the solution (the user's jac, or the fit's 5-point differences with
        the steps chosen at the solution) (G103). No interval for a new observation: a nonlinear
        fit gives no error distribution for one (the interval of f is the delta method's
        linearization alone)."""
        func = "predict_interval"
        alpha = _check_alpha(alpha, func)
        xn = self._xnew(x_new, func)
        m = xn.shape[0]
        if self._isnan() or self._model is None:
            return _sd.full((m, 2), math.nan)
        theta = [float(v) for v in self.params.tolist()]
        mean = _values(self._f(xn, *theta), m, func, "f", "'x_new'")
        # a row with NaN (x_new or its f) has a NaN interval; the Jacobian is formed on the
        # other rows only (its differences would find the model "not finite" there)
        ok = _sd.isfinite(mean)
        if xn.ndim == 2:
            ok = _sd.logical_and(ok, _sd.all(_sd.isfinite(xn), axis=1))
        else:
            ok = _sd.logical_and(ok, _sd.isfinite(xn))
        var = _sd.full((m,), math.nan)
        if bool(_sd.any(ok)):
            all_ok = bool(_sd.all(ok))
            xs = xn if all_ok else xn[ok]
            ms = mean if all_ok else mean[ok]
            fm = self._model
            M = _Model(func, fm.f, fm.jac, xs, _sd.zeros((xs.shape[0],)), None, None, fm.p, fm.lb, fm.ub)
            M.diff.steps = list(fm.diff.steps)
            J = M.accurate_jac(theta, ms)
            v = _sd.sum(_sd.matmul(J, self.cov_params()) * J, axis=-1)
            if all_ok:
                var = v
            else:  # back to every row, NaN where it was left out
                out = [math.nan] * m
                for i, vi in zip(_sd.nonzero(ok)[0].tolist(), v.tolist()):
                    out[i] = vi
                var = _sd.asarray(out)
        q = self._dist().iccdf(alpha / 2)
        half = q * _sd.sqrt(var)
        return _sd.stack([mean - half, mean + half], axis=-1)

    def summary(self, *, alpha=0.05):
        """A text table: the fit, then the parameters with their standard errors, t, p and
        1 - alpha intervals."""
        alpha = _check_alpha(alpha, "summary")
        return _summary(self, alpha)

    def __repr__(self):
        return (
            f"NLSResult(method={self.method!r}, nobs={self.nobs:g}, ssr={self.ssr:.6g}, "
            f"converged={self.converged}, params={self.params})"
        )


def _summary(r, alpha):
    stars = "=" * _WIDTH
    lines = ["Nonlinear Least Squares Results".center(_WIDTH).rstrip(), stars]
    method = "Levenberg-Marquardt" if r.method == "lm" else "Trust Region Reflective"
    left = [
        ("Model:", "NLS"),
        ("Method:", method),
        ("No. Observations:", _count(r.nobs)),
        ("Df Residuals:", _count(r.df_resid)),
        ("Df Model:", _count(r.df_model)),
    ]
    right = [
        ("Residual Std. Error:", f"{math.sqrt(r.scale) if r.scale >= 0 else math.nan:#.4g}"),
        ("Sum of Squares:", f"{r.ssr:#.6g}"),
        ("Converged:", str(r.converged)),
        ("No. Iterations:", str(r.n_iter)),
        ("Function Evals:", str(r.nfev)),
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
    jac = "the user's jac" if r._jac_kind == "jac" else "5-point differences"
    notes = [
        f"Standard errors are sigma^2 (J^T W J)^-1 with J from {jac} at the solution; t uses df_resid "
        "degrees of freedom."
    ]
    if not r._isnan():
        notes.append(_STATUS_TEXT.get(r._count_stat(_S_STATUS), "").capitalize() + ".")
        if r._count_stat(_S_STATUS) > 0 and not math.isfinite(r.ssr):
            notes.append("The sum of squares overflows there: not converged.")
        aliased = [nm for nm, k in zip(names, r._raw[5].tolist()) if not k]
        if aliased:
            notes.append(
                "Aliased (the Jacobian's column is linearly dependent on earlier ones), no inference: "
                + ", ".join(aliased)
                + "."
            )
        if r._active:
            notes.append("At a bound: " + ", ".join(names[j] for j in r._active) + ".")
    lines.append("")
    lines.append("Notes:")
    for i, t in enumerate(notes):
        lines += textwrap.wrap(f"[{i + 1}] {t}", width, subsequent_indent="    ")
    return "\n".join(lines)


# ---- inputs


def _values(v, n, func, what, rows="'x'"):
    """What f (or jac with p columns) returned, as a float64 sd.ndarray of shape (n,)."""
    if not isinstance(v, _sd.ndarray):
        if isinstance(v, numbers.Number) and not isinstance(v, complex):
            raise ValueError(f"{func}(): '{what}' must return {n} values (one per row of {rows}), not a scalar")
        if isinstance(v, (str, bytes)) or v is None:
            raise TypeError(
                f"{func}(): '{what}' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
            )
        try:
            v = _sd.asarray(v)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
        except TypeError, ValueError:
            raise TypeError(
                f"{func}(): '{what}' must return an sd.ndarray or a sequence of numbers, not {_type_name(v)}"
            ) from None
    if v.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{what}' must return real values, not a complex128 array")
    if v.dtype != _sd.float64:
        v = v.astype(_sd.float64)
    if v.ndim != 1 or v.shape[0] != n:
        got = "a scalar" if v.ndim == 0 else f"shape {tuple(v.shape)}"
        raise ValueError(f"{func}(): '{what}' must return {n} values (one per row of {rows}), not {got}")
    return v


def _check_tol(v, func, arg):
    x = _real(v, func, arg)
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): '{arg}' must be a finite number >= 0")
    return x


def _real_list(v, func, arg, p=None):
    """A number (repeated p times) or a 1-dimensional sequence of numbers as a list of floats."""
    if isinstance(v, bool):
        raise TypeError(f"{func}(): '{arg}' must be a number or a sequence of numbers, not bool")
    if isinstance(v, numbers.Real):
        if p is None:
            raise TypeError(f"{func}(): '{arg}' must be a sequence of numbers, not {_type_name(v)}")
        return [float(v)] * p
    if isinstance(v, (str, bytes)) or v is None:
        raise TypeError(f"{func}(): '{arg}' must be a sequence of numbers, not {_type_name(v)}")
    try:
        a, _ = _as_real(v, func, arg)
    except TypeError:
        raise TypeError(f"{func}(): '{arg}' must be a sequence of numbers, not {_type_name(v)}") from None
    if a.ndim != 1:
        raise ValueError(f"{func}(): '{arg}' must be 1-dimensional")
    out = [float(t) for t in a.tolist()]
    if p is not None and len(out) != p:
        raise ValueError(f"{func}(): '{arg}' must have {p} values like 'p0', not {len(out)}")
    return out


def _bounds(bounds, func, p):
    if bounds is None:
        return [-math.inf] * p, [math.inf] * p
    if isinstance(bounds, (str, bytes)) or not hasattr(bounds, "__len__"):
        raise TypeError(f"{func}(): 'bounds' must be None or a pair (lower, upper), not {_type_name(bounds)}")
    if len(bounds) != 2:  # the right kind, the wrong length
        raise ValueError(f"{func}(): 'bounds' must be a pair (lower, upper), not {len(bounds)} values")
    lb = _real_list(bounds[0], func, "bounds", p)
    ub = _real_list(bounds[1], func, "bounds", p)
    if any(math.isnan(v) for v in lb + ub):
        raise ValueError(f"{func}(): 'bounds' contains NaN")
    if not all(a < b for a, b in zip(lb, ub)):
        raise ValueError(f"{func}(): each lower bound must be less than its upper bound")
    return lb, ub


def _strictly_feasible(x, lb, ub, rstep=1e-10):
    """scipy's make_strictly_feasible (least_squares moves trf's x0 so)."""
    out = list(x)
    for i, (v, a, b) in enumerate(zip(x, lb, ub)):
        lo = math.isfinite(a) and v - a <= min(b - v, rstep * max(1.0, abs(a)))
        hi = math.isfinite(b) and b - v <= min(v - a, rstep * max(1.0, abs(b)))
        if hi:
            out[i] = b - rstep * max(1.0, abs(b))
        elif lo:
            out[i] = a + rstep * max(1.0, abs(a))
        if out[i] < a or out[i] > b:
            out[i] = 0.5 * (a + b)
    return out


# ---- the model


class _Model:
    """Calls of f and jac on the full x, the residuals of the rows used, and the accurate
    Jacobian (counted)."""

    def __init__(self, func, f, jac, x, y, mask, sw, p, lb, ub):
        self.func, self.f, self.jac, self.x, self.p = func, f, jac, x, p
        self.n = y.shape[0]
        self.mask, self.sw = mask, sw
        self.yu = y[mask] if mask is not None else y
        self.lb, self.ub = lb, ub
        self.nfev = 0
        self.njev = 0
        # the accurate differences of f on the rows used (their steps, chosen once, are kept)
        self.diff = _Differences(lambda theta: self._rows(self.values(theta)), p, lb, ub, sw)

    def _rows(self, v):
        return v[self.mask] if self.mask is not None else v

    def values(self, theta):
        v = _values(self.f(self.x, *theta), self.n, self.func, "f")
        self.nfev += 1
        return v

    def step_cost(self):
        """Calls of f a polishing step takes: the new point and its accurate Jacobian (4 per
        parameter with differences)."""
        return 1 + (0 if self.jac is not None else 4 * self.p)

    def resid_of(self, v):
        """sqrt(w) (f - y) on the rows used."""
        r = self._rows(v) - self.yu
        return r * self.sw if self.sw is not None else r

    def residuals(self, theta):  # the kernel's callback
        return self.resid_of(self.values(theta))

    def user_jac(self, theta):
        """sqrt(w) jac(x, *theta) on the rows used: (m, p)."""
        J = self.jac(self.x, *theta)
        if isinstance(J, numbers.Number) and not isinstance(J, complex):  # as f's (_values)
            raise ValueError(f"{self.func}(): 'jac' must return shape ({self.n}, {self.p}), not a scalar")
        if not isinstance(J, _sd.ndarray):
            try:
                J = _sd.asarray(J)
            except TypeError, ValueError:
                raise TypeError(
                    f"{self.func}(): 'jac' must return an sd.ndarray or a nested sequence of numbers, "
                    f"not {_type_name(J)}"
                ) from None
        if J.dtype == _sd.complex128:
            raise TypeError(f"{self.func}(): 'jac' must return real values, not a complex128 array")
        if J.dtype != _sd.float64:
            J = J.astype(_sd.float64)
        if J.ndim == 1 and self.p == 1 and J.shape[0] == self.n:
            J = _sd.reshape(J, (self.n, 1))
        if J.ndim != 2 or J.shape[0] != self.n or J.shape[1] != self.p:
            raise ValueError(f"{self.func}(): 'jac' must return shape ({self.n}, {self.p}), not {tuple(J.shape)}")
        J = self._rows(J)
        if self.sw is not None:
            J = J * _sd.reshape(self.sw, (self.sw.shape[0], 1))
        return J

    def accurate_jac(self, theta, v0):
        """sqrt(w) df/dtheta on the rows used, (m, p): the user's jac, or the accurate 5-point
        differences of sd.optimize (with the steps chosen at the first call; a column at its
        rounding level is 0, G88)."""
        self.njev += 1
        if self.jac is not None:
            return self.user_jac(theta)
        J, bad = self.diff.jacobian(theta, self._rows(v0), zero_noise=True)
        if J is None:
            raise ValueError(
                f"{self.func}(): the model is not finite near the parameters {list(theta)} "
                "(the differences for the Jacobian)"
            )
        return J

    def second(self, theta, v0, c, idx):
        """S = sum_i c_i d^2 f_i / dtheta^2 (c on the rows used) over the parameters idx (0
        elsewhere), by central differences with the Jacobian's steps; None when a point is
        outside the bounds or not finite."""
        return self.diff.second(theta, self._rows(v0), c, idx)


def _ssr(r):
    return float(_linalg.vecdot(r, r)) if r.shape[0] else 0.0


def _rel_size(delta, theta):
    """max_j |delta_j| / |theta_j| (|theta|_inf for a zero theta_j)."""
    big = max((abs(t) for t in theta), default=0.0)
    out = 0.0
    for d, t in zip(delta, theta):
        s = abs(t) if t != 0 else (big if big > 0 else 1.0)
        out = max(out, abs(d) / s)
    return out


def _gn_step(func, r, J, rt, free):
    """The Gauss-Newton step argmin ||r + J d|| over the free parameters (the linear models'
    QR kernel with its refinement; 0 for the others and for aliased columns), and the free
    columns' (J^T J)^-1 (None when a column is aliased)."""
    cols = [j for j in range(len(free)) if free[j]]
    Jf = J if len(cols) == len(free) else _sd.take(J, _sd.asarray(cols, dtype=_sd.int64), axis=1)
    raw = _k.fit(func, -r, Jf, None, None, False, rt, 0, 3)
    d = [0.0] * len(free)
    for j, v in zip(cols, raw[0].tolist()):
        d[j] = v if math.isfinite(v) else 0.0
    C = raw[1].tolist() if all(raw[5].tolist()) else None
    return d, cols, C


def _newton_step(d, cols, C, S):
    """Newton's step from the Gauss-Newton one: (J^T J + S) dn = J^T J d, that is
    (I + (J^T J)^-1 S) dn = d on the free columns. None when singular."""
    q = len(cols)
    A = [
        [(1.0 if a == b else 0.0) + sum(C[a][c] * S[cols[c]][cols[b]] for c in range(q)) for b in range(q)]
        for a in range(q)
    ]
    try:
        z = _linalg.solve(_sd.asarray(A), _sd.asarray([d[j] for j in cols])).tolist()
    except _linalg.LinAlgError:
        return None
    out = list(d)
    for a, j in enumerate(cols):
        out[j] = z[a]
    return out if all(math.isfinite(v) for v in out) else None


def _newton_pays(M, size, q, left):
    """Whether to take Newton's steps: the Gauss-Newton steps still needed at the ratio q
    (the remaining distance size q / (1 - q) down to 2^-50) are more than the `left` ones
    allowed, or cost more calls of f than S (2 p (p - 1) calls, 2 p^2 with a user jac) and two
    steps (a step: one call and a Jacobian, 4 p calls of differences)."""
    if q <= 0:
        return False
    p = M.p
    rest = size * q / (1 - q) if q < 1 else math.inf
    need = (
        math.log(rest / _POLISH_TOL) / math.log(1 / q) if q < 1 and rest > _POLISH_TOL else (0.0 if q < 1 else math.inf)
    )
    step = 2.0 if M.jac is not None else 4.0 * p + 1
    cost_s = 2.0 * p * p if M.jac is not None else 2.0 * p * (p - 1)
    return need > left or need * step > cost_s + 2 * step


def _in_box(theta, lb, ub):
    return all(a <= t <= b for t, a, b in zip(theta, lb, ub))


def _polish(M, func, theta, v, r, J, rt, free, budget=math.inf):
    """Gauss-Newton steps from the converged iterate while they shrink (G73); Newton's steps
    with the residuals' curvature term S from second differences (as the GLM's G44) when the
    Gauss-Newton steps still needed at the last two steps' ratio cost more than S
    (_newton_pays), or once when a step stops shrinking. budget: the calls of f left under
    max_nfev (the polish's count against it, G93). Returns (theta, v, r, J, steps, cut): J is
    the Jacobian at the returned theta (to 2^-45); cut when the budget stopped a step that was
    still needed."""
    prev = None
    steps = 0
    S = None
    tried = False
    start = M.nfev

    def afford(calls):
        return M.nfev - start + calls <= budget

    for _ in range(_POLISH_STEPS):
        if not afford(M.step_cost()):
            return theta, v, r, J, steps, True
        d, cols, C = _gn_step(func, r, J, rt, free)
        if S is not None:
            dn = _newton_step(d, cols, C, S) if C is not None else None
            if dn is None:
                S = None
            else:
                d = dn
        size = _rel_size(d, theta)
        if not math.isfinite(size):
            break
        if prev is not None and size >= prev:
            # not contracting: Newton's steps from here, once, unless at the rounding floor
            if S is None and not tried and C is not None and prev > _POLISH_FLOOR and afford(2 * M.p * M.p):
                tried = True
                S = M.second(theta, v, r * M.sw if M.sw is not None else r, cols)
                prev = None
                if S is not None:
                    continue
            break
        tn = [t + s for t, s in zip(theta, d)]
        ok = _in_box(tn, M.lb, M.ub)
        if ok:
            vn = M.values(tn)
            rn = M.resid_of(vn)
            ok = bool(_sd.all(_sd.isfinite(rn))) and (_ssr(rn) <= _ssr(r) * (1 + 2.0**-4) or size <= 2.0**-30)
        if not ok:
            if S is not None:  # Newton's step failed: back to Gauss-Newton
                S = None
                prev = None
                continue
            break
        theta, v, r = tn, vn, rn
        steps += 1
        done = size <= _POLISH_TOL
        q = size / prev if prev is not None else 0.0
        if prev is not None and not done:
            # the predicted rest is below 2^-50, or the steps stopped shrinking at rounding level
            done = (q < 1 and size * q / (1 - q) <= _POLISH_TOL) or (q > 0.5 and size <= _POLISH_FLOOR)
        if done and size <= 2.0**-45:
            break
        J = M.accurate_jac(theta, v)
        if done:
            break
        prev = size
        if (
            S is None
            and not tried
            and C is not None
            and _newton_pays(M, size, q, _POLISH_STEPS - steps)
            and afford(2 * M.p * M.p)
        ):
            tried = True
            c = r * M.sw if M.sw is not None else r
            S = M.second(theta, v, c, cols)
            prev = None
    return theta, v, r, J, steps, False


def _active(theta, lb, ub, g, rtol):
    """Parameters on a bound at the solution: within scipy's find_active_constraints
    distance (rtol = xtol) and with the gradient of the sum of squares pointing outwards."""
    out = []
    for j, (t, a, b) in enumerate(zip(theta, lb, ub)):
        if math.isfinite(a) and t - a <= min(b - t, rtol * max(1.0, abs(a))) and g[j] > 0:
            out.append((j, a))
        elif math.isfinite(b) and b - t <= min(t - a, rtol * max(1.0, abs(b))) and g[j] < 0:
            out.append((j, b))
    return out


def _nan_result(f, n, p, names, method, xshape, jac_kind, nobs=math.nan, columns=None):
    """NaN in the data (nobs NaN) or no row in the fit (nobs 0, G96)."""
    nan = math.nan
    stats = [nan] * _NSTAT
    stats[_S_NOBS] = nobs
    stats[_S_RANK] = 0.0
    stats[_S_ISNAN] = 1.0
    stats[_S_NFEV] = stats[_S_NJEV] = stats[_S_NITER] = stats[_S_NPOLISH] = 0.0
    stats[_S_STATUS] = 0.0
    raw = [
        _sd.full((p,), nan),
        _sd.full((p, p), nan),
        _sd.full((p, p), nan),
        _sd.full((n,), nan),
        _sd.full((n,), nan),
        _sd.ones((p,), dtype=_sd.bool),
        _sd.asarray(stats),
    ]
    return NLSResult(
        raw, method=method, f=f, xshape=xshape, jac_kind=jac_kind, active=[], param_names=names, columns=columns
    )


def nls(
    f,
    x,
    y,
    /,
    *,
    p0,
    jac=None,
    bounds=None,
    method=None,
    weights=None,
    ftol=1e-08,
    xtol=1e-08,
    gtol=1e-08,
    max_nfev=None,
    rtol=None,
    nan_policy="propagate",
):
    """Nonlinear least squares: minimizes sum w_i (y_i - f(x_i, theta))^2 over theta from p0.
    f(x, *theta) gets x (an sd.ndarray (n,) or (n, d)) and the parameters as floats and
    returns n values (scipy's curve_fit form, G68). Without bounds the Levenberg-Marquardt
    method (MINPACK lmder), with them the trust region reflective method (G69)
    (docs/api/regression.md §7)."""
    func = "nls"
    if not callable(f):
        raise TypeError(f"{func}(): 'f' must be callable, not {_type_name(f)}")
    if jac is not None and not callable(jac):
        raise TypeError(f"{func}(): 'jac' must be None or callable, not {_type_name(jac)}")
    theta0 = _real_list(p0, func, "p0")
    p = len(theta0)
    if p == 0:
        raise ValueError(f"{func}(): the model has no parameters ('p0' is empty)")
    if not all(math.isfinite(t) for t in theta0):
        raise ValueError(f"{func}(): 'p0' must be finite")
    lb, ub = _bounds(bounds, func, p)
    bounded = any(math.isfinite(a) or math.isfinite(b) for a, b in zip(lb, ub))
    if method is None:
        method = "trf" if bounded else "lm"
    _check_choice(method, func, "method", _METHODS)
    if method == "lm" and bounded:
        raise ValueError(f"{func}(): method='lm' takes no bounds (method='trf' does)")
    if not _in_box(theta0, lb, ub):
        raise ValueError(f"{func}(): 'p0' must be within 'bounds'")
    ftol = _check_tol(ftol, func, "ftol")
    xtol = _check_tol(xtol, func, "xtol")
    gtol = _check_tol(gtol, func, "gtol")
    if ftol < _EPS and xtol < _EPS and gtol < _EPS:
        raise ValueError(
            f"{func}(): at least one of 'ftol', 'xtol' and 'gtol' must be >= 2.2e-16 (the machine epsilon)"
        )
    if max_nfev is None:
        max_nfev = 100 * p if jac is not None else 100 * p * (p + 1)
    else:
        max_nfev = _check_int(max_nfev, func, "max_nfev", 1)
    rt = _check_rtol(rtol, func)
    if rt < 0 and jac is None:
        rt = _FD_RTOL
    _check_policy(nan_policy, func)

    yy = _read_y(y, func)
    if yy.ndim != 1:
        raise ValueError(f"{func}(): 'y' must be 1-dimensional (nls fits one regression, not a batch)")
    columns = _frame_columns(x)
    xx, _ = _as_real(x, func, "x")
    if xx.ndim not in (1, 2):
        raise ValueError(f"{func}(): 'x' must be 1-dimensional (n,) or 2-dimensional (n, d)")
    n = yy.shape[0]
    if xx.shape[0] != n:
        raise ValueError(_rows_message(func, "x", n, xx.shape[0], yy.shape))
    w = None
    if weights is not None:
        # wls's forms and messages for the same mistakes
        w, _ = _as_real(weights, func, "weights", per_row=f"{n} value{'' if n == 1 else 's'} like 'y'")
        if w.ndim != 1:
            raise ValueError(f"{func}(): 'weights' must be 1-dimensional (nls fits one regression, not a batch)")
        if w.shape[0] != n:
            raise ValueError(_rows_message(func, "weights", n, w.shape[0], yy.shape, "values"))
    # inf and negative weights are errors whatever the NaN policy, also in a row with a NaN
    _check_rows(func, [("y", yy), ("x", xx), ("weights", w)], nonneg=("weights",))
    names = _param_names(f, p)
    xshape = tuple(xx.shape[1:])
    jac_kind = "jac" if jac is not None else "differences"
    bad = _sd.isnan(yy)
    if xx.ndim == 2:
        bad = _sd.logical_or(bad, _sd.any(_sd.isnan(xx), axis=1))
    else:
        bad = _sd.logical_or(bad, _sd.isnan(xx))
    if w is not None:
        bad = _sd.logical_or(bad, _sd.isnan(w))
    if n and bool(_sd.any(bad)):
        if nan_policy == "raise":
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
        if nan_policy == "propagate":
            return _nan_result(f, n, p, names, method, xshape, jac_kind, columns=columns)
        keep = _sd.logical_not(bad)
        yy, xx = yy[keep], xx[keep]
        if w is not None:
            w = w[keep]
        n = yy.shape[0]
    mask = sw = None
    if w is not None:
        pos = w > 0
        if not bool(_sd.all(pos)):
            mask = pos
        sw = _sd.sqrt(w[mask] if mask is not None else w)
    nobs = int(_sd.sum(mask)) if mask is not None else n
    if nobs == 0:  # no row in the fit: f is not called (G96)
        return _nan_result(f, n, p, names, method, xshape, jac_kind, nobs=0.0, columns=columns)
    if nobs < p:
        raise ValueError(f"{func}(): 'p0' has {p} parameters, more than the {nobs} observations used")

    M = _Model(func, f, jac, xx, yy, mask, sw, p, lb, ub)
    x0 = _strictly_feasible(theta0, lb, ub) if method == "trf" else theta0
    kjac = M.user_jac if jac is not None else None
    xs, status, nfev, njev, n_iter, _, _ = _sd.optimize.lsq_solve(
        func,
        M.residuals,
        kjac,
        None,
        x0,
        lb,
        ub,
        _METHODS.index(method),
        ftol,
        xtol,
        gtol,
        max_nfev,
        nobs,
        0,
        [],
        False,
        None,
        False,
    )
    if status == _ST_START:
        raise ValueError(f"{func}(): 'f' is not finite at 'p0' (rows used)")
    if status == _ST_JAC:
        what = "'jac' is" if jac is not None else "the model's forward differences are"
        raise ValueError(f"{func}(): {what} not finite at the parameters {xs}")
    M.nfev, M.njev = 0, 0
    theta = list(xs)
    v = M.values(theta)
    r = M.resid_of(v)
    J = M.accurate_jac(theta, v)
    free = [True] * p
    active = []
    if method == "trf" and bounded:
        g = (_sd.matmul(r, J)).tolist()
        act = _active(theta, lb, ub, g, xtol)
        if act:
            tn = list(theta)
            for j, b in act:
                tn[j] = b
            vn = M.values(tn)
            rn = M.resid_of(vn)
            if bool(_sd.all(_sd.isfinite(rn))) and _ssr(rn) <= _ssr(r) * (1 + 2.0**-4):
                theta, v, r = tn, vn, rn
                J = M.accurate_jac(theta, v)
            for j, _b in act:
                free[j] = False
            active = [j for j, _b in act]
    npolish = 0
    cut = False
    if status > 0 and any(free):
        # the polish's calls of f count against max_nfev (the solution's own Jacobian, needed
        # for the covariance, is outside it) (G93)
        budget = max_nfev - nfev
        theta, v, r, J, npolish, cut = _polish(M, func, theta, v, r, J, rt, free, budget)
    # the covariance and the aliased parameters from the QR of J at the solution
    raw = _k.fit(func, -r, J, None, None, False, rt, 0, 3)
    stats = [math.nan] * _NSTAT
    stats[_S_NOBS] = float(nobs)
    stats[_S_RANK] = float(raw[6][1])
    stats[_S_SSR] = _ssr(r)
    stats[_S_ISNAN] = 0.0
    stats[_S_NFEV] = float(nfev + M.nfev)
    stats[_S_NJEV] = float(njev + M.njev)
    stats[_S_NITER] = float(n_iter)
    stats[_S_STATUS] = float(status)
    stats[_S_NPOLISH] = float(npolish)
    stats[_S_SUMLOGW] = float(_sd.sum(_sd.log(w[mask] if mask is not None else w))) if w is not None else 0.0
    out = [_sd.asarray(theta), raw[1], raw[2], v, yy - v, raw[5], _sd.asarray(stats)]
    return NLSResult(
        out,
        method=method,
        f=f,
        xshape=xshape,
        jac_kind=jac_kind,
        active=active,
        param_names=names,
        model=M,
        polish_cut=cut,
        columns=columns,
    )


# the methods' call-shape errors in the library's form (R89)
_call_errors(NLSResult)
