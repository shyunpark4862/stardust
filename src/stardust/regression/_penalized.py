"""Penalized regression: ridge, lasso, elastic_net, their cross-validation and the penalized
GLM (docs/api/regression.md §4, docs/design/regression.md §3.5).

The fits are the native kernel stardust._core.regression.penalized_path: ridge by one SVD for
every alpha, the lasso and the elastic net by coordinate descent with the KKT polish, the
penalized GLM by IRLS with coordinate descent inside. This module checks and converts the
inputs, draws the cross-validation folds from an sd.random Generator and builds the results,
which are prediction-focused (G48): no standard errors or p-values.

An `alpha` that is a 1-D array is a path: solved largest first with warm starts, and every
per-fit field gains an axis of the alphas' length just before its own axes (params
(..., L, k), the scalars (..., L)).
"""

import math
import numbers
import operator

from .. import _core as _sd
from .. import random as _random
from .. import special as _special
from . import _glm as _glm_mod
from ._glm import _broadcast_shape
from ._linear import _arg, _as_real, _check_bool, _check_choice, _check_policy, _div, _names, _prepare

_k = _sd.regression

# the kernel's per-alpha scalars (csrc/regression/penalized.cpp, enum PStat)
(_NOBS, _SUMW, _NITER, _GAP, _CONV, _DF, _ISNAN, _RSS, _TSS, _TEST_SSE, _TEST_SW, _POLISHED, _GCV) = range(13)
# the kernel's kinds and families
_ENET, _RIDGE, _GLMNET = 0, 1, 2
_FAMILY_CODE = {"gaussian": 0, "binomial": 1, "poisson": 2}
_FAMILIES = ("binomial", "poisson")
# ridge's default alpha grid: alpha_max as if l1_ratio were 1e-3 (glmnet's rule for alpha = 0;
# scikit-learn has no automatic grid for l1_ratio = 0)
_RIDGE_GRID_RATIO = 1e-3
# scikit-learn's floor of alpha_max (np.finfo(float).resolution)
_ALPHA_FLOOR = 1e-15


def _type_name(v):
    return type(v).__name__


# ---- argument checks

def _check_alpha_value(v, func, arg="alpha"):
    """A finite real >= 0 as a float."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be a number or a 1-dimensional array of numbers, not {_type_name(v)}")
    try:
        x = float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"{func}(): '{arg}' must be finite and >= 0")
    return x


def _check_alpha(v, func, arg="alpha"):
    """(alphas as a 1-D float64 sd.ndarray, is_path, the echo): a number, or a 1-D array or
    sequence of them (a path)."""
    if isinstance(v, numbers.Number) and not isinstance(v, bool):
        x = _check_alpha_value(v, func, arg)
        return _sd.asarray([x]), False, x
    if isinstance(v, (str, bytes, bool)) or v is None:
        raise TypeError(f"{func}(): '{arg}' must be a number or a 1-dimensional array of numbers, not {_type_name(v)}")
    try:
        a = v if isinstance(v, _sd.ndarray) else _sd.asarray(v)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"{func}(): '{arg}' must be a number or a 1-dimensional array of numbers, "
                        f"not {_type_name(v)}") from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not a complex128 array")
    if a.ndim != 1:
        raise ValueError(f"{func}(): '{arg}' must be a number or a 1-dimensional array, not {a.ndim}-dimensional")
    if a.shape[0] == 0:
        raise ValueError(f"{func}(): '{arg}' must not be empty")
    a = a.astype(_sd.float64)
    if not bool(_sd.all(_sd.isfinite(a))) or bool(_sd.any(a < 0)):
        raise ValueError(f"{func}(): '{arg}' must be finite and >= 0")
    return a, True, a


def _check_ratio(v, func):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): 'l1_ratio' must be an int or a float, not {_type_name(v)}")
    x = float(v)
    if not 0 <= x <= 1:
        raise ValueError(f"{func}(): 'l1_ratio' must be between 0 and 1 (inclusive)")
    return x


def _check_int(v, func, arg, lo):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {_type_name(v)}")
    v = operator.index(v)
    if v < lo:
        raise ValueError(f"{func}(): '{arg}' must be >= {lo}")
    if v >= 2**31:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _check_tol(v, func):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): 'tol' must be an int or a float, not {_type_name(v)}")
    x = float(v)
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"{func}(): 'tol' must be a finite number > 0")
    return x


def _check_eps(v, func):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): 'eps' must be an int or a float, not {_type_name(v)}")
    x = float(v)
    if not 0 < x <= 1:
        raise ValueError(f"{func}(): 'eps' must be > 0 and <= 1")
    return x


def _check_rng(rng, func):
    if not isinstance(rng, _random.Generator):
        raise TypeError(f"{func}(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), "
                        f"not {_type_name(rng)}")
    return rng


# ---- results

def _scalar_or(v):
    return float(v) if not isinstance(v, _sd.ndarray) else v


class _PenalizedResult:
    """What the penalized fits share: the kernel's params (..., L, k) and per-alpha scalars,
    the alpha axis kept for a path and dropped for one alpha. The intercept is params[0]."""

    __slots__ = ("_P", "_S", "_cache", "param_names", "alpha", "l1_ratio", "standardize", "_intercept", "_batch",
                 "_path", "_fit")

    def __init__(self, P, S, *, alpha, l1_ratio, path, param_names, intercept, standardize, fit):
        self._P = P            # params (batch..., L, k)
        self._S = S            # stats (batch..., L, nstat)
        self._cache = {}
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.standardize = standardize
        self.param_names = param_names
        self._intercept = intercept
        self._path = path
        self._batch = tuple(P.shape[:-2])
        self._fit = fit        # (y, X, weights or None) of the fit, after nan_policy='omit'

    def _cached(self, name, make):
        c = self._cache
        if name not in c:
            c[name] = make()
        return c[name]

    def _stat(self, i):
        """A per-alpha scalar: (batch..., L) for a path, the batch shape (a float for one
        regression) for one alpha."""
        v = self._S[..., i]
        if self._path:
            return v
        return v[..., 0]

    @property
    def params(self):
        return self._cached("params", lambda: self._P if self._path else self._P[..., 0, :])

    @property
    def nobs(self):
        """The rows in the fit (weight > 0)."""
        return self._stat(_NOBS)

    @property
    def df_model(self):
        """The effective degrees of freedom of the coefficients (the intercept not counted):
        tr Z_A (Z_A^T W Z_A + n alpha (1 - l1_ratio) I)^-1 Z_A^T W over the nonzero
        coefficients A (standardized coordinates under standardize); ridge's sum s^2/(s^2 +
        alpha), the lasso's number of nonzero coefficients."""
        return self._stat(_DF)

    def _shape_X(self, Xn):
        """X (..., m, p) made to broadcast against params (..., L, k) for a path."""
        if not self._path:
            return Xn
        return _sd.reshape(Xn, tuple(Xn.shape[:-2]) + (1,) + tuple(Xn.shape[-2:]))

    def _design(self, X_new, func):
        Xn, _ = _as_real(X_new, func, "X_new")
        p = self._P.shape[-1] - (1 if self._intercept else 0)
        if Xn.ndim == 0:
            raise ValueError(f"{func}(): 'X_new' must have at least 1 dimension")
        if Xn.ndim == 1:
            if p != 1:
                raise ValueError(f"{func}(): 'X_new' must have {p} columns like the fit's 'X', not 1-dimensional")
            Xn = _sd.reshape(Xn, (Xn.shape[0], 1))
        if Xn.shape[-1] != p:
            raise ValueError(f"{func}(): 'X_new' must have {p} columns like the fit's 'X', not {Xn.shape[-1]}")
        return Xn

    def _linear(self, Xn, func):
        """X params (the intercept added), summed in double-double."""
        return _k.apply(func, self._shape_X(Xn), self.params, self._intercept, None)[0]

    def _fitted_resid(self):
        def make():
            y, X, _ = self._fit
            yy = _sd.reshape(y, tuple(y.shape[:-1]) + (1, y.shape[-1])) if self._path else y
            return _k.apply("fittedvalues", self._shape_X(X), self.params, self._intercept, yy)
        return self._cached("fitted_resid", make)

    def __repr__(self):
        name = type(self).__name__
        shape = self._batch + ((self._P.shape[-2],) if self._path else ())
        if shape:
            return f"{name}(shape {shape}, {self._P.shape[-1]} parameters)"
        return f"{name}(alpha={self.alpha:g}, nobs={self.nobs:g}, df_model={self.df_model:.6g}, params={self.params})"


class _LinearPenalized(_PenalizedResult):
    __slots__ = ()

    @property
    def fittedvalues(self):
        return self._fitted_resid()[0]

    @property
    def resid(self):
        return self._fitted_resid()[1]

    @property
    def ssr(self):
        """sum w (y - fitted)^2 over the fit's rows (from resid, whose terms are in double-double)."""
        def make():
            r = self.resid
            w = self._fit[2]
            rr = r * r
            if w is not None:
                ww = _sd.reshape(w, tuple(w.shape[:-1]) + (1, w.shape[-1])) if self._path else w
                rr = rr * ww
            return _scalar_or(_sd.sum(rr, axis=-1))
        return self._cached("ssr", make)

    @property
    def rsquared(self):
        """1 - ssr / tss, tss about the weighted mean of y (uncentered without the intercept, as ols)."""
        return self._cached("rsquared", lambda: 1.0 - _div(self.ssr, self._stat(_TSS)))

    def predict(self, X_new, /):
        """X_new params, the intercept added (..., [L,] m)."""
        return self._linear(self._design(X_new, "predict"), "predict")


class _Iterative:
    """converged, n_iter, dual_gap (the coordinate descent fits)."""
    __slots__ = ()

    @property
    def converged(self):
        v = self._stat(_CONV)
        return v == 1.0 if isinstance(v, _sd.ndarray) else v == 1.0

    @property
    def n_iter(self):
        v = self._stat(_NITER)
        if isinstance(v, _sd.ndarray):
            return _sd.where(_sd.isnan(v), 0.0, v).astype(_sd.int64)
        return 0 if math.isnan(v) else int(v)

    @property
    def dual_gap(self):
        """The duality gap at the returned coefficients, in the objective's units (scikit-learn's
        formulation; the penalized GLM's is that of its last quadratic (IRLS) model)."""
        return self._stat(_GAP)


class RidgeResult(_LinearPenalized):
    """The result of sd.regression.ridge."""
    __slots__ = ()


class LassoResult(_Iterative, _LinearPenalized):
    """The result of sd.regression.lasso."""
    __slots__ = ()


class ElasticNetResult(_Iterative, _LinearPenalized):
    """The result of sd.regression.elastic_net."""
    __slots__ = ()


class _CV:
    """The cross-validation's fields of ridge_cv, lasso_cv, elastic_net_cv: the grid, the curve
    and its standard errors, the two choices."""
    __slots__ = ()

    def _cv(self, name):
        return self._cv_fields[name]

    @property
    def alphas(self):
        """The grid (..., L), largest first."""
        return self._cv("alphas")

    @property
    def cv_error(self):
        """The mean over the folds of each fold's weighted mean squared error (..., L); for
        ridge_cv(cv='gcv') the generalized cross-validation criterion."""
        return self._cv("cv_error")

    @property
    def cv_se(self):
        """The standard error of cv_error: the folds' standard deviation / sqrt(folds) (NaN for GCV)."""
        return self._cv("cv_se")

    @property
    def alpha_min(self):
        """The alpha of the smallest cv_error (the largest such alpha on a tie)."""
        return self._cv("alpha_min")

    @property
    def alpha_1se(self):
        """The largest alpha whose cv_error is within one cv_se of the minimum (NaN for GCV)."""
        return self._cv("alpha_1se")


class RidgeCVResult(_CV, RidgeResult):
    """The result of sd.regression.ridge_cv: the refit at the chosen alpha and the curve."""
    __slots__ = ("_cv_fields",)


class LassoCVResult(_CV, LassoResult):
    """The result of sd.regression.lasso_cv: the refit at the chosen alpha and the curve."""
    __slots__ = ("_cv_fields",)


class ElasticNetCVResult(_CV, ElasticNetResult):
    """The result of sd.regression.elastic_net_cv: the refit at the chosen alpha and the curve."""
    __slots__ = ("_cv_fields",)


class PenalizedGLMResult(_Iterative, _PenalizedResult):
    """The result of sd.regression.penalized_glm."""
    __slots__ = ("family", "_sep", "_glm0")

    def __init__(self, *a, family, sep, glm0, **kw):
        super().__init__(*a, **kw)
        self.family = family
        self._sep = sep      # (batch..., L) 1.0 where alpha = 0 and glm found separation
        self._glm0 = glm0    # (mask of the alphas that are 0, glm's result) or None

    def _mean(self, eta):
        return _special.expit(eta) if self.family == "binomial" else _sd.exp(eta)

    @property
    def separation(self):
        """True where alpha = 0 and the maximum likelihood estimate does not exist (glm's
        certificates, G33, G50): the coefficients are NaN, the fit is its limit. Always False
        for alpha > 0, whose solution is finite."""
        v = self._sep if self._path else self._sep[..., 0]
        return v == 1.0

    @property
    def linear_predictor(self):
        """eta = X params; at alpha = 0 glm's (+-inf at the boundary under separation)."""
        def make():
            eta = self._fitted_resid()[0]
            if self._glm0 is None:
                return eta
            mask, g = self._glm0
            e0 = g.linear_predictor
            if not self._path:
                return e0
            return _sd.where(_sd.expand_dims(mask, axis=-1), _sd.expand_dims(e0, axis=-2), eta)
        return self._cached("linear_predictor", make)

    @property
    def fittedvalues(self):
        """The fitted means mu = g^-1(eta)."""
        return self._cached("fittedvalues", lambda: self._mean(self.linear_predictor))

    @property
    def resid_response(self):
        """y - mu."""
        def make():
            y = self._fit[0]
            yy = _sd.reshape(y, tuple(y.shape[:-1]) + (1, y.shape[-1])) if self._path else y
            return yy - self.fittedvalues
        return self._cached("resid_response", make)

    @property
    def deviance(self):
        return self._stat(_RSS)

    @property
    def null_deviance(self):
        """The deviance of the intercept-only model (eta = 0 without the intercept)."""
        return self._stat(_TSS)

    @property
    def deviance_ratio(self):
        """1 - deviance / null_deviance (glmnet's dev.ratio)."""
        return self._cached("deviance_ratio", lambda: 1.0 - _div(self.deviance, self.null_deviance))

    def predict(self, X_new, /, *, which="mean"):
        """The mean g^-1(eta) (which='mean') or eta (which='linear') at the rows of X_new."""
        func = "predict"
        _check_choice(which, func, "which", ("mean", "linear"))
        eta = self._linear(self._design(X_new, func), func)
        return eta if which == "linear" else self._mean(eta)


# ---- the fits

def _aux(w, tw, n):
    """The kernel's (..., 2, n) block of the weights and test weights, or None."""
    if w is None and tw is None:
        return None
    if tw is None:
        return _sd.stack([w, _sd.zeros(tuple(w.shape))], axis=-2)
    if w is None:
        w = _sd.ones(tuple(tw.shape))
    shape = tuple(_sd.broadcast_arrays(w, tw)[0].shape)
    return _sd.stack([_sd.broadcast_to(w, shape), _sd.broadcast_to(tw, shape)], axis=-2)


def _run(func, kind, y, X, w, alphas, l1_ratio, family, intercept, standardize, max_iter, tol, tw=None):
    aux = _aux(w, tw, y.shape[-1])
    return _k.penalized_path(func, kind, y, X, aux, w is not None, tw is not None, alphas, l1_ratio,
                             _FAMILY_CODE[family], intercept, standardize, max_iter, tol)


def _common(func, y, X, weights, intercept, standardize, nan_policy):
    _check_bool(intercept, func, "intercept")
    _check_bool(standardize, func, "standardize")
    _check_policy(nan_policy, func)
    y, X, w, names, _, _, _ = _prepare(func, y, X, weights, nan_policy)
    return y, X, w, _names(names, X.shape[-1], intercept)


def _fit(func, cls, kind, y, X, alpha, l1_ratio, weights, intercept, standardize, max_iter, tol, nan_policy,
         family="gaussian", **extra):
    y, X, w, names = _common(func, y, X, weights, intercept, standardize, nan_policy)
    alphas, path, echo = _check_alpha(alpha, func)
    P, S = _run(func, kind, y, X, w, alphas, l1_ratio, family, intercept, standardize, max_iter, tol)
    if kind == _GLMNET:
        extra["family"] = family
    return cls(P, S, alpha=echo, l1_ratio=l1_ratio, path=path, param_names=names, intercept=intercept,
               standardize=standardize, fit=(y, X, w), **extra)


def ridge(y, X, /, *, alpha, weights=None, intercept=True, standardize=False, nan_policy="propagate"):
    """Ridge regression: minimizes (1/(2 sw)) sum w (y - b0 - X beta)^2 + alpha/2 ||S beta||^2
    (scikit-learn's ElasticNet objective at l1_ratio = 0; its Ridge's alpha is sw times this
    one). Solved by one SVD for every alpha."""
    return _fit("ridge", RidgeResult, _RIDGE, y, X, alpha, 0.0, weights, intercept, standardize, 1, 1.0, nan_policy)


def lasso(y, X, /, *, alpha, weights=None, intercept=True, standardize=False, max_iter=1000, tol=1e-4,
          nan_policy="propagate"):
    """The lasso: minimizes (1/(2 sw)) sum w (y - b0 - X beta)^2 + alpha ||S beta||_1 by
    coordinate descent (scikit-learn's Lasso objective and tolerance)."""
    func = "lasso"
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_tol(tol, func)
    return _fit(func, LassoResult, _ENET, y, X, alpha, 1.0, weights, intercept, standardize, max_iter, tol, nan_policy)


def elastic_net(y, X, /, *, alpha, l1_ratio=0.5, weights=None, intercept=True, standardize=False, max_iter=1000,
                tol=1e-4, nan_policy="propagate"):
    """The elastic net: minimizes (1/(2 sw)) sum w (y - b0 - X beta)^2 + alpha (l1_ratio
    ||S beta||_1 + (1 - l1_ratio)/2 ||S beta||^2) by coordinate descent (scikit-learn's
    ElasticNet objective and tolerance)."""
    func = "elastic_net"
    l1_ratio = _check_ratio(l1_ratio, func)
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_tol(tol, func)
    return _fit(func, ElasticNetResult, _ENET, y, X, alpha, l1_ratio, weights, intercept, standardize, max_iter, tol,
                nan_policy)


def penalized_glm(y, X, /, *, family, alpha, l1_ratio=0.5, weights=None, intercept=True, standardize=False,
                   max_iter=100, tol=1e-8, nan_policy="propagate"):
    """The penalized GLM (binomial with the logit link, poisson with the log link): minimizes
    deviance / (2 sw) + alpha (l1_ratio ||S beta||_1 + (1 - l1_ratio)/2 ||S beta||^2), the
    deviance weighted by w (glmnet's objective). IRLS outside, coordinate descent inside."""
    func = "penalized_glm"
    _check_choice(family, func, "family", _FAMILIES)
    l1_ratio = _check_ratio(l1_ratio, func)
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_tol(tol, func)
    y, X, w, names = _common(func, y, X, weights, intercept, standardize, nan_policy)
    alphas, path, echo = _check_alpha(alpha, func)
    P, S = _run(func, _GLMNET, y, X, w, alphas, l1_ratio, family, intercept, standardize, max_iter, tol)
    L = alphas.shape[0]
    zero = [v == 0.0 for v in alphas.tolist()]
    sep = _sd.zeros(tuple(S.shape[:-1]))
    glm0 = None
    if any(zero):
        # no penalty: glm's fit and its rules (G50): separation by glm's certificates (G33),
        # aliased columns (G9), the same weights' meaning (var_weights: the deviance's weights)
        g = _glm_mod.glm(y, X, family=family, var_weights=w, intercept=intercept, max_iter=max_iter, tol=tol)
        mask = _sd.asarray(zero)
        glm0 = (mask, g)

        def per(v):  # a per-regression value as (batch..., 1) against (batch..., L)
            return _sd.expand_dims(v, axis=-1) if isinstance(v, _sd.ndarray) else v

        gsep = g.separation
        gsep = gsep.astype(_sd.float64) if isinstance(gsep, _sd.ndarray) else float(gsep)
        P = _sd.where(_sd.expand_dims(mask, axis=-1), _sd.expand_dims(g.params, axis=-2), P)
        cols = [S[..., j] for j in range(S.shape[-1])]
        n_iter = g.n_iter.astype(_sd.float64) if isinstance(g.n_iter, _sd.ndarray) else float(g.n_iter)
        conv = g.converged.astype(_sd.float64) if isinstance(g.converged, _sd.ndarray) else float(g.converged)
        for j, v in ((_NITER, n_iter), (_CONV, conv), (_RSS, g.deviance), (_TSS, g.null_deviance),
                     (_DF, g.df_model), (_POLISHED, 0.0)):
            cols[j] = _sd.where(mask, per(v), cols[j])
        cols[_GAP] = _sd.where(_sd.logical_and(mask, per(gsep) == 1.0), math.nan, cols[_GAP])
        S = _sd.stack(cols, axis=-1)
        sep = _sd.where(mask, per(gsep), sep)
    return PenalizedGLMResult(P, S, alpha=echo, l1_ratio=l1_ratio, path=path, param_names=names, intercept=intercept,
                              standardize=standardize, fit=(y, X, w), family=family, sep=sep, glm0=glm0)



# ---- cross-validation (G46)

def _grid(func, y, X, w, l1_ratio, intercept, standardize, alphas, eps, ratio):
    """(grid (..., L) largest first): scikit-learn's default (alpha_max eps^(i/(L-1)), alpha_max
    the smallest alpha whose solution is 0 at l1_ratio `ratio`), or the given values sorted."""
    if isinstance(alphas, numbers.Integral) and not isinstance(alphas, bool):
        L = _check_int(alphas, func, "alphas", 1)
        amax = _k.alpha_max(func, y, X, w, ratio, 0, intercept, standardize)
        if isinstance(amax, _sd.ndarray):
            amax = _sd.where(amax <= _ALPHA_FLOOR, _ALPHA_FLOOR, amax)
        elif amax <= _ALPHA_FLOOR:
            amax = _ALPHA_FLOOR
        ratios = _sd.asarray([1.0] + [math.pow(eps, i / (L - 1)) for i in range(1, L - 1)] + ([eps] if L > 1 else []))
        if isinstance(amax, _sd.ndarray):
            return _sd.expand_dims(amax, axis=-1) * ratios
        return amax * ratios
    a, _, _ = _check_alpha(alphas, func, "alphas")
    return _sd.flip(_sd.sort(a))


def _cv(func, cls, kind, y, X, *, rng, cv, alphas, eps, select, l1_ratio, weights, intercept, standardize, max_iter,
        tol, nan_policy):
    y, X, w, names = _common(func, y, X, weights, intercept, standardize, nan_policy)
    gcv = cv == "gcv" and kind == _RIDGE
    if not gcv:
        if kind == _RIDGE and isinstance(cv, str):
            raise ValueError(f"{func}(): 'cv' must be an int or 'gcv', not {cv!r}")
        if kind == _RIDGE and not (isinstance(cv, numbers.Integral) and not isinstance(cv, bool)):
            raise TypeError(f"{func}(): 'cv' must be an int or 'gcv', not {_type_name(cv)}")
        cv = _check_int(cv, func, "cv", 2)
        _check_rng(rng, func)
    elif rng is not None:
        _check_rng(rng, func)
    eps = _check_eps(eps, func)
    _check_choice(select, func, "select", ("min", "1se"))
    if gcv and select == "1se":
        raise ValueError(f"{func}(): select='1se' needs folds (cv='gcv' has no standard errors)")
    ratio = _RIDGE_GRID_RATIO if kind == _RIDGE else l1_ratio
    grid = _grid(func, y, X, w, l1_ratio, intercept, standardize, alphas, eps, ratio)
    n = y.shape[-1]
    nan = math.nan
    if gcv:
        Pg, Sg = _run(func, kind, y, X, w, grid, l1_ratio, "gaussian", intercept, standardize, max_iter, tol)
        err = Sg[..., _GCV]
        se = err * nan
    else:
        if cv > n:
            raise ValueError(f"{func}(): 'cv' must be <= the number of rows ({n}), not {cv}")
        labels = _sd.asarray([i % cv for i in range(n)], dtype=_sd.int64)
        fold = _sd.take(labels, rng.permutation(n))
        batch = _broadcast_shape(func, _broadcast_shape(func, tuple(y.shape[:-1]), tuple(X.shape[:-2])),
                                 tuple(w.shape[:-1]) if w is not None else ())
        base = w if w is not None else _sd.ones((n,))
        base = _sd.broadcast_to(base, batch + (n,))
        ks = _sd.reshape(_sd.arange(cv, dtype=_sd.int64), (cv,) + (1,) * len(batch) + (1,))
        test = _sd.where(fold == ks, 1.0, 0.0)  # (cv, 1..., n)
        tw = test * base
        trw = (1.0 - test) * base
        _, S = _run(func, kind, y, X, trw, grid, l1_ratio, "gaussian", intercept, standardize, max_iter, tol, tw=tw)
        mse = S[..., _TEST_SSE] / S[..., _TEST_SW]  # (cv, batch..., L)
        err = _sd.mean(mse, axis=0)
        se = _sd.std(mse, axis=0, correction=1) / math.sqrt(cv)
    # the choices (largest alpha first in the grid, so the first minimum is the largest alpha)
    if not isinstance(err, _sd.ndarray) or err.ndim == 1:
        e = err.tolist()
        g = grid.tolist()
        s = se.tolist()
        pick = None
        if any(math.isnan(v) for v in e):
            amin = a1se = nan
        else:
            imin = min(range(len(e)), key=lambda i: (e[i], i))
            pick = imin
            amin = g[imin]
            a1se = nan if gcv else g[min(i for i in range(len(e)) if e[i] <= e[imin] + s[imin])]
        bad = math.isnan(amin)
    else:
        imin = _sd.expand_dims(_sd.argmin(err, axis=-1), axis=-1)
        pick = imin
        emin = _sd.take_along_axis(err, imin, axis=-1)
        amin = _sd.take_along_axis(grid if grid.ndim == err.ndim else _sd.broadcast_to(grid, tuple(err.shape)),
                                   imin, axis=-1)[..., 0]
        if gcv:
            a1se = amin * nan
        else:
            smin = _sd.take_along_axis(se, imin, axis=-1)
            ok = err <= emin + smin
            first = _sd.expand_dims(_sd.argmax(ok, axis=-1), axis=-1)
            gb = grid if grid.ndim == err.ndim else _sd.broadcast_to(grid, tuple(err.shape))
            a1se = _sd.take_along_axis(gb, first, axis=-1)[..., 0]
        anynan = _sd.any(_sd.isnan(err), axis=-1)
        amin = _sd.where(anynan, nan, amin)
        a1se = _sd.where(anynan, nan, a1se)
        bad = None
    chosen = amin if select == "min" else a1se
    if isinstance(chosen, _sd.ndarray):
        refit_alphas = _sd.where(_sd.isnan(chosen), 0.0, _sd.expand_dims(chosen, axis=-1))
    else:
        refit_alphas = _sd.asarray([0.0 if bad else chosen])
    if gcv and pick is not None:
        # the grid's fits are the refit (each alpha from the same SVD, as ridge at that alpha)
        if isinstance(pick, int):
            P, S = Pg[pick:pick + 1], Sg[pick:pick + 1]
        else:
            P = _sd.take_along_axis(Pg, _sd.expand_dims(pick, axis=-1), axis=-2)
            S = _sd.take_along_axis(Sg, _sd.expand_dims(pick, axis=-1), axis=-2)
    else:
        P, S = _run(func, kind, y, X, w, refit_alphas, l1_ratio, "gaussian", intercept, standardize, max_iter, tol)
    if isinstance(chosen, _sd.ndarray):
        nanrow = _sd.reshape(_sd.isnan(chosen), tuple(chosen.shape) + (1, 1))
        P = _sd.where(nanrow, nan, P)
        S = _sd.where(nanrow, nan, S)
    elif bad:
        P = P * nan
        S = S * nan
    r = cls(P, S, alpha=chosen, l1_ratio=l1_ratio, path=False, param_names=names, intercept=intercept,
            standardize=standardize, fit=(y, X, w))
    r._cv_fields = {"alphas": grid, "cv_error": err, "cv_se": se, "alpha_min": amin, "alpha_1se": a1se}
    return r


def ridge_cv(y, X, /, *, rng=None, cv=5, alphas=100, eps=1e-3, select="min", weights=None, intercept=True,
             standardize=False, nan_policy="propagate"):
    """ridge with alpha chosen by k-fold cross-validation (folds from `rng`) or, cv='gcv', by
    generalized cross-validation; refit on all the data at the chosen alpha."""
    return _cv("ridge_cv", RidgeCVResult, _RIDGE, y, X, rng=rng, cv=cv, alphas=alphas, eps=eps, select=select,
               l1_ratio=0.0, weights=weights, intercept=intercept, standardize=standardize, max_iter=1, tol=1.0,
               nan_policy=nan_policy)


def lasso_cv(y, X, /, *, rng, cv=5, alphas=100, eps=1e-3, select="min", weights=None, intercept=True,
             standardize=False, max_iter=1000, tol=1e-4, nan_policy="propagate"):
    """lasso with alpha chosen by k-fold cross-validation (folds from `rng`); refit on all the
    data at the chosen alpha."""
    func = "lasso_cv"
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_tol(tol, func)
    return _cv(func, LassoCVResult, _ENET, y, X, rng=rng, cv=cv, alphas=alphas, eps=eps, select=select, l1_ratio=1.0,
               weights=weights, intercept=intercept, standardize=standardize, max_iter=max_iter, tol=tol,
               nan_policy=nan_policy)


def elastic_net_cv(y, X, /, *, rng, l1_ratio=0.5, cv=5, alphas=100, eps=1e-3, select="min", weights=None,
                   intercept=True, standardize=False, max_iter=1000, tol=1e-4, nan_policy="propagate"):
    """elastic_net with alpha chosen by k-fold cross-validation (folds from `rng`); refit on all
    the data at the chosen alpha."""
    func = "elastic_net_cv"
    l1_ratio = _check_ratio(l1_ratio, func)
    if l1_ratio == 0 and isinstance(alphas, numbers.Integral) and not isinstance(alphas, bool):
        raise ValueError(f"{func}(): the default grid needs l1_ratio > 0 (give 'alphas' for l1_ratio = 0)")
    max_iter = _check_int(max_iter, func, "max_iter", 1)
    tol = _check_tol(tol, func)
    return _cv(func, ElasticNetCVResult, _ENET, y, X, rng=rng, cv=cv, alphas=alphas, eps=eps, select=select,
               l1_ratio=l1_ratio, weights=weights, intercept=intercept, standardize=standardize, max_iter=max_iter,
               tol=tol, nan_policy=nan_policy)
