"""Linear models: ols, wls, gls, linregress (docs/api/regression.md §2, docs/design/regression.md §2-§3).

The fit itself is the native kernel stardust._core.regression.fit (QR, aliased columns,
refinement on the augmented system, sums of squares in double-double, covariances). This
module checks and converts the inputs (frame columns, NaN policy, GLS whitening) and builds
the result object, whose inference fields are computed on first use from the kernel's
output with sd.stats' distributions. Every field is an sd.ndarray, or a Python number for a
scalar of one regression; a batch of regressions puts the batch shape in front.
"""

import functools
import inspect
import math
import numbers
import operator
import re
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
    return _sd._type_name(v)  # (numpy's types as numpy.bool: API.md §2.1)


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
    # the string-option form, as sd.stats' same option (stats/_descriptive.py)
    return _check_choice(v, func, "nan_policy", tuple(_POLICIES))


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
            raise ValueError(_rows_message(func, "groups", n, len(groups), (), "values"))
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
                raise ValueError(_rows_message(func, "groups", n, len(labels), (), "values"))
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
        raise ValueError(_rows_message(func, "groups", n, a.shape[-1], (), "values"))
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


def _real(v, func, arg):
    """v (an int or a float; numpy's too) as a float: TypeError for anything else, OverflowError
    'out of range' for an int beyond the doubles (API.md §2)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {_type_name(v)}")
    try:
        return float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None


def _check_alpha(v, func):
    x = _real(v, func, "alpha")
    if not 0 < x < 1:
        raise ValueError(f"{func}(): 'alpha' must be between 0 and 1 (exclusive)")
    return x


def _number_or_1d(v, func, arg, scalar="a number"):
    """(values as a 1-D float64 sd.ndarray, is_array): a number, or a 1-D array or sequence of
    numbers (a path of alphas, several quantiles). The same mistake gives the same message for
    every such argument (ridge's alpha, quantreg's q): a non-number (bools, bool arrays, strings,
    None) is a TypeError, a 0-, 2- or more-dimensional or empty array a ValueError, an int beyond
    the doubles an OverflowError. The values' range is the caller's to check."""
    what = f"{func}(): '{arg}' must be {scalar} or a 1-dimensional array of numbers"
    if isinstance(v, bool) or isinstance(v, (str, bytes)) or v is None:
        raise TypeError(f"{what}, not {_type_name(v)}")
    if isinstance(v, numbers.Number):
        if not isinstance(v, numbers.Real):
            raise TypeError(f"{what}, not {_type_name(v)}")
        return _sd.asarray([_real(v, func, arg)]), False
    if getattr(v, "ndim", None) == 0:  # a 0-d (numpy) array: sd has none
        raise ValueError(f"{func}(): '{arg}' must be {scalar} or a 1-dimensional array, not 0-dimensional")
    try:
        a = v if isinstance(v, _sd.ndarray) else _sd.asarray(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None
    except (TypeError, ValueError):
        raise TypeError(f"{what}, not {_seq_name(v)}") from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not a complex128 array")
    if a.dtype == _sd.bool:
        raise TypeError(f"{what}, not a bool array")
    if a.ndim != 1:
        raise ValueError(f"{func}(): '{arg}' must be {scalar} or a 1-dimensional array, not {a.ndim}-dimensional")
    if a.shape[0] == 0:
        raise ValueError(f"{func}(): '{arg}' must not be empty")
    return a.astype(_sd.float64), True


def _as_real(x, func, arg, per_row=None):
    """(x as a float64 sd.ndarray, its column names or None). Takes sd.ndarrays, sequences of
    numbers and frame Series/DataFrames (missing values become NaN). per_row: for an argument
    with a value per row, what it must have (e.g. "5 values like 'y'"), said for a scalar."""
    names = None
    if isinstance(x, (_frame.Series, _frame.DataFrame)):
        if isinstance(x, _frame.Series):
            names = [str(x.name)] if x.name is not None else None
        else:
            names = [str(c) for c in x.columns]
        try:
            a = x.to_array(dtype=_sd.float64, na_value=math.nan)
        except (TypeError, ValueError):  # a string column is a ValueError of to_array
            raise TypeError(f"{func}(): '{arg}' must have numeric columns") from None
        return a, names
    if isinstance(x, _sd.ndarray):
        a = x
    elif isinstance(x, numbers.Number):
        if isinstance(x, complex):
            raise TypeError(f"{func}(): '{arg}' must be real, not complex")
        if per_row is not None:
            raise ValueError(f"{func}(): '{arg}' must have {per_row}, not a scalar")
        raise ValueError(f"{func}(): '{arg}' must have at least 1 dimension")
    else:
        try:
            a = _sd.asarray(x)
        except OverflowError:
            raise OverflowError(f"{func}(): Python int too large to convert to float64") from None
        except (TypeError, ValueError):
            raise TypeError(f"{func}(): '{arg}' must be an sd.ndarray, a sequence of numbers or a frame "
                            f"Series or DataFrame, not {_seq_name(x)}") from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not a complex128 array")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    return a, names


def _seq_name(x):
    """The type name of a rejected value; for a list or tuple also the first element type that
    is not a number (`list of str`): the function takes lists, just not of those."""
    name = _type_name(x)
    if isinstance(x, (list, tuple)):
        # depth first in order: a reversed stack, so each element is pushed and popped once
        stack = [x]
        while stack:
            v = stack.pop()
            if isinstance(v, (list, tuple)):
                stack.extend(reversed(v))
            elif v is None or isinstance(v, bool) or not isinstance(v, numbers.Number):
                return f"{name} of {_type_name(v)}"
    return name


def _frame_columns(X):
    """The column names of X when it is a frame DataFrame (predictions from a DataFrame are then
    checked against them, docs/api/regression.md §1), else None."""
    return [str(c) for c in X.columns] if isinstance(X, _frame.DataFrame) else None


def _new_design(X_new, func, p, columns):
    """X_new as the (..., m, p) design of a prediction from a fit with p columns (columns: the
    fit's DataFrame column names, or None). A DataFrame X_new of a DataFrame fit must have the
    same columns in the same order (scikit-learn's rule); otherwise the columns are taken by
    position. inf is an error as in the fit's inputs (G24); NaN gives NaN."""
    if columns is not None and isinstance(X_new, _frame.DataFrame):
        got = [str(c) for c in X_new.columns]
        if got != columns:
            what = "in the fit's order" if sorted(got) == sorted(columns) else "like the fit's 'X'"
            raise ValueError(f"{func}(): 'X_new' must have the columns {columns} {what}, not {got}")
    Xn, _ = _as_real(X_new, func, "X_new")
    cols = f"{p} column" + ("" if p == 1 else "s")
    if Xn.ndim == 0:
        raise ValueError(f"{func}(): 'X_new' must have at least 1 dimension")
    if Xn.ndim == 1:
        if p != 1:
            raise ValueError(f"{func}(): 'X_new' must have {cols} like the fit's 'X', not 1-dimensional")
        Xn = _sd.reshape(Xn, (Xn.shape[0], 1))
    if Xn.shape[-1] != p:
        raise ValueError(f"{func}(): 'X_new' must have {cols} like the fit's 'X', not {Xn.shape[-1]}")
    if Xn.size and bool(_sd.any(_sd.isinf(Xn))):
        raise ValueError(f"{func}(): 'X_new' contains infinity")
    return Xn


def _check_batches(func, named):
    """named: (argument, its batch shape) pairs; ValueError naming them when they do not
    broadcast (the kernels' own check names no argument)."""
    out = []
    for _, shape in named:
        shape = tuple(shape)
        if len(shape) > len(out):
            out = [1] * (len(shape) - len(out)) + out
        lead = len(out) - len(shape)
        for d, s in enumerate(shape):
            r = out[lead + d]
            if s == r or s == 1:
                continue
            if r == 1:
                out[lead + d] = s
                continue
            listed = ", ".join(f"'{a}' {tuple(sh)}" for a, sh in named)
            raise ValueError(f"{func}(): the batch shapes of {listed} do not broadcast")
    return tuple(out)


def _has_nan(*arrays):
    return any(a is not None and a.size and bool(_sd.any(_sd.isnan(a))) for a in arrays)


def _expand(v, nd):
    """A per-regression array (batch shape) with `nd` trailing axes of length 1."""
    if not isinstance(v, _sd.ndarray):
        return v
    return _sd.reshape(v, tuple(v.shape) + (1,) * nd)


# ---- the result object

class _RegressionResult:
    """What every regression result shares (the linear models' and the GLM's): the kernel's
    raw output (params, cov, R, ., ., kept, stats, ...), the cached fields, the coefficients'
    inference and the design checks of predictions. statsmodels' names (G10); the intercept
    is params[0]."""

    __slots__ = ("_raw", "_cache", "_param_names", "_cov_type", "_intercept", "_batch", "_extra", "_fitargs", "_columns")
    # the positions of the shared scalars in the kernel's stats
    _I_NOBS, _I_RANK, _I_ISNAN, _I_NGROUPS, _I_LAGS = _NOBS, _RANK, _ISNAN, _NGROUPS, _LAGS

    def __init__(self, raw, *, param_names, intercept, cov_type, batch, extra=None, fitargs=None, columns=None):
        # exact zero coefficients are +0 (G84): the solves' -0 is a sign of rounding, not data
        if not isinstance(raw, list):
            raw = list(raw)
        raw[0] = raw[0] + 0.0
        self._raw = raw          # params, cov, R, fitted, resid, kept, stats
        self._columns = columns  # the fit's DataFrame column names, or None
        self._cache = {}
        self._param_names = param_names
        self._cov_type = cov_type
        self._intercept = intercept
        self._batch = batch      # the batch shape, () for one regression
        self._extra = extra or {}
        # what the kernel was given (func, y, X, w, u, rtol, group codes, their count): the
        # diagnostics refactor or refit the same whitened problem
        self._fitargs = fitargs

    # -- the fit's configuration: read-only (G98)
    @property
    def param_names(self):
        """The coefficients' names ('const' for the intercept, then the columns'): a new list
        at each access (G113)."""
        return _fresh(self._param_names)

    @property
    def cov_type(self):
        """The covariance's kind."""
        return self._cov_type

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
    def nobs(self):
        return self._stat(self._I_NOBS)

    @property
    def rank(self):
        r = self._raw[6][..., self._I_RANK]
        return int(float(r)) if not self._batch else r.astype(_sd.int64)

    def _isnan(self):
        """True for a regression whose data had NaN (nan_policy='propagate')."""
        f = self._raw[6][..., self._I_ISNAN]
        return f != 0 if self._batch else float(f) != 0

    def _nan_where(self, v):
        f = self._isnan()
        if self._batch:
            return _sd.where(f, math.nan, v)
        return math.nan if f else v

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
            return self._cached("df_inference", lambda: self._nan_where(self._stat(self._I_NGROUPS) - 1.0))
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

    def conf_int(self, *, alpha=0.05):
        """(..., k, 2): the lower and upper bounds of the 1 - alpha confidence intervals
        (alpha keyword-only, G99)."""
        alpha = _check_alpha(alpha, "conf_int")
        q = self._dist().iccdf(alpha / 2)
        half = q * self.bse
        return _sd.stack([self.params - half, self.params + half], axis=-1)

    # -- prediction
    def _beff(self):
        """params with 0 for the aliased columns: the fit's own (finite) design times it."""
        def make():
            keep = _sd.logical_or(self._raw[5], self._per(self._isnan(), 1))
            return _sd.where(keep, self.params, 0.0)
        return self._cached("beff", make)

    def _pred_args(self):
        """(params, mask) of a prediction (the kernel's apply): mask is 1.0 for the columns in
        the model, whatever their coefficient (a NaN in X_new there gives NaN, also under a
        coefficient 0), and 0.0 for the aliased ones, which take no part (R's predict.lm); every
        column of a NaN regression, whose NaN params give NaN."""
        def make():
            keep = _sd.logical_or(self._raw[5], self._per(self._isnan(), 1))
            return self.params, _sd.where(keep, 1.0, 0.0)
        return self._cached("pred_args", make)

    def _shape_X(self, Xn):
        """X_new made to broadcast against params (the quantile axis adds an axis)."""
        return Xn

    def _xbeta(self, Xn, func):
        """X_new params (the intercept added), summed in double-double."""
        b, mask = self._pred_args()
        return _k.apply(func, self._shape_X(Xn), b, self._intercept, None, mask)[0]

    def _design(self, X_new, func):
        p = self.params.shape[-1] - (1 if self._intercept else 0)
        Xn = _new_design(X_new, func, p, self._columns)
        _check_pred_batch(func, Xn, self._batch)
        return Xn

    # the internal units of the covariance (the linear models' power-of-two scaling) and the
    # factor that brings a standard error of a prediction back
    def _cov_s(self):
        return self.cov_params()

    def _xfactor(self):
        """None, or the columns' factors (..., k) the internal covariance is in."""
        return None

    def _unit(self):
        return 1.0

    def _mean_var_s(self, Xn):
        """The variance of x params at the rows of X_new (..., [L,] m), internal units: the
        aliased columns take no part (as in the prediction), a NaN in a column in the model
        gives NaN."""
        Xs = self._shape_X(Xn)
        Xf = _sd.concat([_sd.ones(tuple(Xs.shape[:-1]) + (1,)), Xs], axis=-1) if self._intercept else Xs
        fx = self._xfactor()
        if fx is not None:
            Xf = Xf / _sd.expand_dims(fx, axis=-2)
        _, mask = self._pred_args()
        Xf = _sd.where(_sd.expand_dims(mask, axis=-2) != 0.0, Xf, 0.0)
        kept = self._raw[5]
        both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
        C = _sd.where(both, self._cov_s(), 0.0)
        var = _sd.sum(_sd.matmul(Xf, C) * Xf, axis=-1)
        bad = self._per(self._isnan(), 1)
        if isinstance(bad, _sd.ndarray):
            return _sd.where(bad, math.nan, var)
        return var * math.nan if bad else var

    def _interval(self, X_new, func, alpha, var_extra=None):
        """(..., [L,] m, 2): the 1 - alpha interval mean -+ q sqrt(var) at the rows of X_new."""
        alpha = _check_alpha(alpha, func)
        Xn = self._design(X_new, func)
        mean = self._xbeta(Xn, func)
        var = self._mean_var_s(Xn)
        if var_extra is not None:
            var = var + var_extra
        q = self._dist().iccdf(alpha / 2)
        half = q * _sd.sqrt(var) * self._per(self._unit(), 1)
        return _sd.stack([mean - half, mean + half], axis=-1)


_TAKES = re.compile(r"takes (from (\d+) to )?(\d+) positional arguments? but (\d+)( positional arguments?)?(.*) "
                    r"(was|were) given")


def _call_message(name, rest):
    """Python's call-shape error of a method (its numbers count self) in the library's form:
    "conf_int(): takes 0 positional arguments but 1 was given" (the form of sd.stats'
    constructors, T36)."""
    m = _TAKES.match(rest)
    if m:
        lo, hi, got = (int(m.group(2)) - 1 if m.group(2) else None), int(m.group(3)) - 1, int(m.group(4)) - 1
        span = f"from {lo} to {hi}" if lo is not None else f"{hi}"
        noun = "argument" if hi == 1 and lo is None else "arguments"
        was = "was" if got == 1 else "were"
        tail = m.group(6) or ""
        if m.group(5):
            tail = f" positional argument{'' if got == 1 else 's'}{tail}"
        rest = f"takes {span} positional {noun} but {got}{tail} {was} given"
    elif rest.startswith("got an unexpected"):
        rest = rest[len("got an "):]
    return f"{name}(): {rest}"


def _call_errors(cls):
    """Wraps the public methods defined in cls so that a wrong call shape (positional for a
    keyword-only option, a missing or an unexpected argument) says the method's name, as
    every other error of the library: "summary(): takes 0 positional arguments but 1 was
    given", not "_LinearResult.summary() takes 1 ...". Errors from inside the method pass."""
    for name, fn in list(vars(cls).items()):
        if name.startswith("_") or not inspect.isfunction(fn) or getattr(fn, "_call_errors", False):
            continue

        def make(fn=fn, name=name):
            head = f"{fn.__qualname__}() "

            @functools.wraps(fn)
            def method(self, *args, **kwargs):
                try:
                    return fn(self, *args, **kwargs)
                except TypeError as e:
                    msg = str(e)
                    # only the call's own binding error (no frame of fn beneath this one)
                    if not msg.startswith(head) or e.__traceback__.tb_next is not None:
                        raise
                    raise TypeError(_call_message(name, msg[len(head):])) from None
            method._call_errors = True
            return method
        setattr(cls, name, make())
    return cls


def _fresh(v):
    """A configuration field as the result gives it (G113): a new list or array at each
    access, so that changing what was returned does not change the result."""
    if isinstance(v, list):
        return list(v)
    if isinstance(v, _sd.ndarray):
        return v.copy()
    return v


def _check_pred_batch(func, Xn, batch):
    """X_new's batch axes must broadcast against the fit's (the kernel's message names no argument)."""
    xb = tuple(Xn.shape[:-2])
    if not xb or not batch:
        return
    a, b = (1,) * (len(batch) - len(xb)) + xb, (1,) * (len(xb) - len(batch)) + tuple(batch)
    if any(u != v and u != 1 and v != 1 for u, v in zip(a, b)):
        raise ValueError(f"{func}(): the batch shapes of 'X_new' {xb} and the fit {tuple(batch)} do not broadcast")


class _LinearResult(_RegressionResult):
    """The result of ols, wls or gls: statsmodels' names (G10). The intercept is params[0].

    The kernel may have fitted y / fy on the columns X_j / fx_j (powers of two, exact; only for
    data far outside [2^-256, 2^256], whose squares and covariances would leave the doubles):
    params, R, fitted values and residuals are given back on the data's scale at the fit, the
    sums of squares (stats) and the covariance stay in the fit's units, and every derived field
    is formed from them and scaled once at the end, so that the inference is scale-equivariant
    (decision 2026-11-19)."""

    __slots__ = ()

    # -- the internal units (fy per regression, fx per column; 1 and None without scaling)
    def _fy(self):
        return self._extra.get("fy", 1.0)

    def _unit(self):
        return self._fy()

    def _xfactor(self):
        return self._extra.get("fx")

    def _out2(self, v):
        """A sum of squares from the fit's units to the data's."""
        fy = self._fy()
        if isinstance(fy, float) and fy == 1.0:
            return v
        return v * self._scalar(fy * fy)

    @property
    def resid(self):
        return self._raw[4]

    @property
    def df_model(self):
        return self._cached("df_model", lambda: self._nan_where(self._stat(_RANK) - (1.0 if self._intercept else 0.0)))

    @property
    def df_resid(self):
        return self._cached("df_resid", lambda: self._nan_where(self.nobs - self._stat(_RANK)))

    def _ssr_s(self):
        return self._stat(_SSR)

    @property
    def ssr(self):
        return self._out2(self._ssr_s())

    @property
    def centered_tss(self):
        return self._out2(self._stat(_CTSS))

    @property
    def uncentered_tss(self):
        return self._out2(self._stat(_UTSS))

    def _tss(self):
        return self._stat(_CTSS) if self._intercept else self._stat(_UTSS)

    def _ess_direct(self):
        e = self._stat(_ESSC) if self._intercept else self._stat(_ESSU)
        # the intercept alone explains nothing: ESS is 0 exactly (the summed value is the
        # rounding of params[0] to a double); statsmodels gives 0 too
        if not self._intercept:
            return e
        if self._batch:
            return _sd.where(self._raw[6][..., _RANK] == 1.0, 0.0, e)
        return 0.0 if self._stat(_RANK) == 1.0 else e

    def _ess_s(self):
        """The explained sum of squares in the fit's units: summed directly where it is the
        smaller part of the total (R^2 < 1/2, where tss - ssr would cancel), else tss - ssr."""
        def make():
            e, s, t = self._ess_direct(), self._ssr_s(), self._tss()
            # a constant y (TSS exactly 0) leaves nothing to explain: ESS 0 (G85)
            if self._batch:
                return _sd.where(t == 0.0, 0.0, _sd.where(e < s, e, t - s))
            return 0.0 if t == 0.0 else (e if e < s else t - s)
        return self._cached("ess_s", make)

    @property
    def ess(self):
        return self._out2(self._ess_s())

    def _no_tss(self, v):
        """v, NaN where TSS is exactly 0 (a constant y: R^2 and the F test are 0/0, G85)."""
        t = self._tss()
        if self._batch:
            return _sd.where(t == 0.0, math.nan, v)
        return math.nan if t == 0.0 else v

    def _one_minus_r2(self):
        e, s, t = self._ess_direct(), self._ssr_s(), self._tss()
        if self._batch:
            return self._no_tss(_sd.where(e < s, 1.0 - e / t, s / t))
        return self._no_tss(1.0 - _div(e, t) if e < s else _div(s, t))

    @property
    def rsquared(self):
        def make():
            e, s, t = self._ess_direct(), self._ssr_s(), self._tss()
            if self._batch:
                return self._no_tss(_sd.where(e < s, e / t, 1.0 - s / t))
            return self._no_tss(_div(e, t) if e < s else 1.0 - _div(s, t))
        return self._cached("rsquared", make)

    @property
    def rsquared_adj(self):
        def make():
            c = 1.0 if self._intercept else 0.0
            return 1.0 - _div(self.nobs - c, self.df_resid) * self._one_minus_r2()
        return self._cached("rsquared_adj", make)

    def _scale_s(self):
        """ssr / df_resid in the fit's units; NaN for df_resid = 0 (0/0: the fit uses every
        degree of freedom, G95)."""
        return self._cached("scale_s", lambda: _div0(self._ssr_s(), self.df_resid))

    @property
    def scale(self):
        return self._cached("scale", lambda: self._out2(self._scale_s()))

    @property
    def llf(self):
        def make():
            n = self.nobs
            ls = _log(_div(self._ssr_s(), n))
            fy = self._fy()
            if not (isinstance(fy, float) and fy == 1.0):
                ls = ls + 2.0 * _log(self._scalar(fy))
            v = -n / 2 * (_LOG_2PI + ls + 1.0) + self._stat(_SUMLOGW) / 2
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
    def fvalue(self):
        def make():
            if self._robust:
                return self._no_tss(self._stat(_WALDF))
            dfm = self.df_model
            f = _div(_div(self._ess_s(), dfm), self._scale_s())
            # no parameter but the intercept: no F test (statsmodels: NaN)
            if self._batch:
                return self._no_tss(_sd.where(dfm == 0.0, math.nan, f))
            return self._no_tss(math.nan if dfm == 0 else f)
        return self._cached("fvalue", make)

    @property
    def f_pvalue(self):
        def make():
            dfn = self._stat(_WALDQ) if self._robust else self.df_model
            return self._scalar(_F(dfn=_arg(dfn), dfd=_arg(self._df_inference())).ccdf(_arg(self.fvalue)))
        return self._cached("f_pvalue", make)

    # -- the covariance (in the fit's units, then scaled once)
    def _cov_s(self):
        def make():
            c = self._raw[1]
            if self._robust:
                # no degree of freedom left: 0/0 like the classical one (G95)
                df = self.df_resid
                if self._batch:
                    return _sd.where(_expand(df == 0.0, 2), math.nan, c)
                return c * math.nan if df == 0 else c
            return self._per(self._scale_s(), 2) * c
        return self._cached("cov_s", make)

    def _colfactor(self):
        """fy / fx per parameter (..., k): a parameter in the data's units over the fit's; None
        without scaling."""
        fx = self._xfactor()
        if fx is None:
            return None
        return self._per(self._fy(), 1) / fx

    def cov_params(self):
        """The covariance of the parameters (k x k); rows and columns of aliased parameters are NaN."""
        def make():
            g = self._colfactor()
            c = self._cov_s()
            if g is None:
                return c
            return c * (_sd.expand_dims(g, axis=-1) * _sd.expand_dims(g, axis=-2))
        return self._cached("cov_params", make)

    @property
    def bse(self):
        def make():
            b = _sd.sqrt(_linalg.diagonal(self._cov_s()))
            g = self._colfactor()
            return b if g is None else b * g
        return self._cached("bse", make)

    @property
    def tvalues(self):
        def make():
            if "beff_params_s" not in self._extra:
                return self.params / self.bse
            # the fit's own units, beta' / sqrt(C'): neither can leave the doubles there, while a
            # parameter or its standard error in the data's units may overflow or underflow
            return (self._extra["beff_params_s"] + 0.0) / _sd.sqrt(_linalg.diagonal(self._cov_s()))
        return self._cached("tvalues", make)

    def predict(self, X_new, /):
        """X_new @ params (the intercept added), summed in double-double. Aliased columns take
        no part; a NaN in a column of the model gives NaN."""
        return self._xbeta(self._design(X_new, "predict"), "predict")

    def predict_interval(self, X_new, /, *, alpha=0.05, kind="mean"):
        """(..., m, 2): the bounds of the 1 - alpha interval for the mean at each row of X_new
        (kind='mean') or for a new observation there (kind='observation', variance scale + the
        mean's)."""
        func = "predict_interval"
        _check_choice(kind, func, "kind", ("mean", "observation"))
        extra = self._per(self._scale_s(), 1) if kind == "observation" else None
        return self._interval(X_new, func, alpha, extra)

    # -- diagnostics (G17, G28, G29). All are of the whitened problem the kernel fits (the model's
    # sqrt(w) y on sqrt(w) X for WLS, L^-1 y on L^-1 X for GLS), as the robust covariances are;
    # rows with weight 0 are out of the fit: leverage, studentized residuals and Cook's distance
    # are 0 there (the weighted formulas) and the sums leave them out. They are formed in the
    # fit's units (the residuals over fy) and are ratios, free of them.

    def _wresid(self):
        """The whitened residuals e~ in the fit's units: resid (OLS), sqrt(w) resid (WLS), L^-1
        resid (GLS); 0 for a perfect fit (ssr 0: residuals within rounding, G95)."""
        def make():
            func, y, X, w, u, tol, codes, gtot = self._fitargs
            r = self._extra.get("resid_s", self.resid)
            if u is not None:
                D = X
                if self._intercept:
                    uc = _sd.broadcast_to(_sd.expand_dims(u, axis=-1), tuple(X.shape[:-1]) + (1,))
                    D = _sd.concat([uc, X], axis=-1)
                b = self._beff()
                if "beff_params_s" in self._extra:  # the fit's units, as y and X here
                    keep = _sd.logical_or(self._raw[5], self._per(self._isnan(), 1))
                    b = _sd.where(keep, self._extra["beff_params_s"], 0.0)
                e = _k.apply("resid", D, b, False, y)[1]
                e = _sd.broadcast_to(e, tuple(self.resid.shape)) if e.shape != self.resid.shape else e
            elif w is not None:
                e = _sd.sqrt(w) * r
            else:
                e = r
            zero = self._ssr_s() == 0.0
            if self._batch:
                return _sd.where(_expand(zero, 1), 0.0, e)
            return _sd.zeros(tuple(e.shape)) if zero else e
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
            return self._wresid() / _sd.sqrt(self._per(self._scale_s()) * (1.0 - self.leverage))
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
        return self._cached("durbin_watson", lambda: _div(self._resid_stat(0), self._ssr_s()))

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
                s2 = _div(self._ssr_s(), self.nobs)  # the fit's units, as the auxiliary's
                lm = _div(a._ess_s(), 2.0 * s2 * s2)
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


_call_errors(_RegressionResult)
_call_errors(_LinearResult)


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


def _div0(a, b):
    """a / b, NaN where b is 0 (no degree of freedom left: the estimate is 0/0 whatever a)."""
    if isinstance(b, _sd.ndarray):
        return _sd.where(b == 0.0, math.nan, a / b)
    if b == 0:
        return math.nan
    return _div(a, b)


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
    ci = r.conf_int(alpha=alpha)
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

def _rows_message(func, arg, n, got, yshape, unit="rows"):
    """"'X' must have 5 rows like 'y', not 4" (1 row, 1 value), and for a y of shape (..., m, 1)
    (one row per regression: a column vector read under the batch rules) what to give
    instead (decision 2026-11-19, G105)."""
    noun = unit[:-1] if n == 1 else unit
    msg = f"{func}(): '{arg}' must have {n} {noun} like 'y', not {got}"
    if n == 1 and len(yshape) >= 2 and got != 1:
        msg += (f" ('y' of shape {tuple(yshape)} is a batch of regressions with 1 row each; give a "
                "1-dimensional 'y' for one regression)")
    return msg


def _read_y(y, func):
    """y as an sd.ndarray: a one-column frame DataFrame is read as that column (n,) (G105)."""
    if isinstance(y, _frame.DataFrame) and len(list(y.columns)) == 1:
        a, _ = _as_real(y, func, "y")
        return _sd.reshape(a, (a.shape[0],))
    a, _ = _as_real(y, func, "y")
    return a


def _check_rows(func, named, nonneg=()):
    """inf in the per-row arrays (named: (argument, array or None)) is an error whatever the NaN
    policy (G24), and so are negative weights (`nonneg`): checked before nan_policy='omit' drops
    rows, so that a row with a NaN cannot hide them."""
    for arg, a in named:
        if a is None or not a.size:
            continue
        if bool(_sd.any(_sd.isinf(a))):
            raise ValueError(f"{func}(): '{arg}' contains infinity")
        if arg in nonneg and bool(_sd.any(a < 0)):
            raise ValueError(f"{func}(): '{arg}' must be >= 0")


def _prepare(func, y, X, weights, nan_policy, extra_rows=None, groups=None, cov_type="nonrobust", intercept=None):
    """Converted and checked y, X (..., n, p), weights and cluster groups; NaN rows (and rows
    with a missing group label) dropped for 'omit'. Returns (y, X, w, names, batch_shape,
    keep_rows or None, (codes, gtot))."""
    y = _read_y(y, func)
    if y.ndim == 0:
        raise ValueError(f"{func}(): 'y' must have at least 1 dimension")
    X, names = _as_real(X, func, "X")
    if X.ndim == 0:
        raise ValueError(f"{func}(): 'X' must have at least 1 dimension")
    if X.ndim == 1:
        X = _sd.reshape(X, (X.shape[0], 1))
    n = y.shape[-1]
    if X.shape[-2] != n:
        raise ValueError(_rows_message(func, "X", n, X.shape[-2], y.shape))
    if intercept is not None:
        _check_params(func, X.shape[-1], intercept)
    w = None
    if weights is not None:
        w, _ = _as_real(weights, func, "weights", per_row=f"{n} value{'' if n == 1 else 's'} like 'y'")
        if w.shape[-1] != n:
            raise ValueError(_rows_message(func, "weights", n, w.shape[-1], y.shape, "values"))
    codes, gtot, gmiss = _groups(groups, func, n, cov_type)
    named = [("y", y.shape[:-1]), ("X", X.shape[:-2])]
    if w is not None:
        named.append(("weights", w.shape[:-1]))
    if codes is not None:
        named.append(("groups", codes.shape[:-1]))
    _check_batches(func, named)
    batch = y.ndim > 1 or X.ndim > 2 or (w is not None and w.ndim > 1) or (codes is not None and codes.ndim > 1)
    keep = None
    if nan_policy == "raise":
        if _has_nan(y, X, w) or gmiss is not None:
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
    elif nan_policy == "omit":
        if batch:
            raise ValueError(f"{func}(): nan_policy='omit' takes one regression, not a batch "
                             "(the regressions would keep different rows)")
        _check_rows(func, [("y", y), ("X", X), ("weights", w)], nonneg=("weights",))
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


def _check_params(func, p, intercept):
    """A model must have a parameter (G101): X's columns or the intercept."""
    if p == 0 and not intercept:
        raise ValueError(f"{func}(): the model has no parameters (no columns in 'X' and intercept=False)")


def _own(a):
    """A copy of a (None stays None) that a result can keep for its lazy fields."""
    return None if a is None else a.copy()


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


def _exact_constant(raw, y, w, intercept, cov_type, original_scale=True):
    """G92: a regression with an intercept whose y is exactly constant over the fit's rows
    (weight > 0) has the exact solution (y0, 0, ..., 0) (aliased columns stay NaN), whatever
    the design and, for GLS, the whitening (the intercept's column is in the model, so y0 times
    it reproduces y exactly): params, the fit's sums (SSR, TSS and ESS 0), a robust covariance
    (0: the scores vanish; a cluster covariance of one cluster stays NaN, G108) and,
    original_scale, the fitted values (y0) and residuals (y - y0) are set to it in raw (a list,
    changed in place). The QR leaves rounding there (a slope of 6e-32). Returns the mask of
    those regressions (batch shape), or None."""
    if not intercept or y.shape[-1] == 0:
        return None
    robust = cov_type != "nonrobust"
    params, cov, kept, st = raw[0], raw[1], raw[5], raw[6]
    batch = tuple(params.shape[:-1])
    if w is None:
        hi, lo = _sd.max(y, axis=-1), _sd.min(y, axis=-1)
    else:
        used = w > 0
        hi = _sd.max(_sd.where(used, y, -math.inf), axis=-1)
        lo = _sd.min(_sd.where(used, y, math.inf), axis=-1)
    # one cluster: no covariance at all (the single-cluster NaN wins over the zero scores)
    zero_cov = robust and cov_type != "cluster"
    if not batch:
        hi, lo = float(hi), float(lo)
        const = hi == lo and math.isfinite(hi) and float(st[_ISNAN]) == 0 and bool(kept[0])
        if not const:
            return None
        k = params.shape[-1]
        target = [hi] + [0.0 if bool(kept[j]) else math.nan for j in range(1, k)]
        raw[0] = _sd.asarray(target)
        if robust and (zero_cov or float(st[_NGROUPS]) >= 2):
            both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
            raw[1] = _sd.where(both, 0.0, cov)
        s = st.tolist()
        s[_SSR] = s[_CTSS] = s[_ESSC] = 0.0
        raw[6] = _sd.asarray(s)
        if original_scale:
            raw[3] = _sd.full(tuple(raw[3].shape), hi)
            raw[4] = y - hi
        return True
    # a 1-D y (the batch from X or w alone) gives Python floats
    hi = _sd.broadcast_to(hi, batch) if isinstance(hi, _sd.ndarray) else _sd.full(batch, float(hi))
    lo = _sd.broadcast_to(lo, batch) if isinstance(lo, _sd.ndarray) else _sd.full(batch, float(lo))
    const = _sd.logical_and(_sd.logical_and(hi == lo, _sd.isfinite(hi)),
                            _sd.logical_and(st[..., _ISNAN] == 0.0, kept[..., 0]))
    if not bool(_sd.any(const)):
        return None
    k = params.shape[-1]
    c1 = _sd.expand_dims(const, axis=-1)
    first = _sd.arange(k) == 0
    target = _sd.where(first, _sd.expand_dims(hi, axis=-1), _sd.where(kept, 0.0, math.nan))
    raw[0] = _sd.where(c1, target, params)
    if robust:
        both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
        zc = c1 if zero_cov else _sd.logical_and(c1, _sd.expand_dims(st[..., _NGROUPS] >= 2.0, axis=-1))
        raw[1] = _sd.where(_sd.logical_and(_sd.expand_dims(zc, axis=-1), both), 0.0, cov)
    zero_cols = _sd.logical_or(_sd.logical_or(_sd.arange(st.shape[-1]) == _SSR, _sd.arange(st.shape[-1]) == _CTSS),
                               _sd.arange(st.shape[-1]) == _ESSC)
    raw[6] = _sd.where(_sd.logical_and(c1, zero_cols), 0.0, st)
    if original_scale:
        h1 = _sd.expand_dims(hi, axis=-1)
        raw[3] = _sd.where(c1, _sd.broadcast_to(h1, tuple(raw[3].shape)), raw[3])
        raw[4] = _sd.where(c1, _sd.broadcast_to(y - h1, tuple(raw[4].shape)), raw[4])
    return const


# Data whose magnitude leaves [2^-256, 2^256] is fitted in power-of-two units (exact): their
# squares (sums of squares) and the covariance's (R^T R)^-1 would leave the doubles otherwise.
_SAFE_HI, _SAFE_LO = 2.0 ** 256, 2.0 ** -256


def _pow2(m):
    """Power-of-two factors for the maxima m (a float or an array): 2^e with m = f 2^e, f in
    [1/2, 1), where m is finite and outside [2^-256, 2^256]; 1 elsewhere (0, NaN, inf: the
    kernel's own rules). None when every one is inside."""
    if isinstance(m, _sd.ndarray):
        if m.size == 0:
            return None
        out = bool(_sd.any(m > _SAFE_HI)) or bool(_sd.any(_sd.logical_and(m > 0.0, m < _SAFE_LO)))
    else:
        out = m > _SAFE_HI or 0.0 < m < _SAFE_LO
    if not out:
        return None

    def f(v):
        if isinstance(v, list):
            return [f(u) for u in v]
        if math.isfinite(v) and v > 0.0 and (v > _SAFE_HI or v < _SAFE_LO):
            return math.ldexp(1.0, math.frexp(v)[1])
        return 1.0
    return _sd.asarray(f(m.tolist())) if isinstance(m, _sd.ndarray) else f(m)


def _fit_kernel(func, y, X, w, u, intercept, tol, cov, codes, gtot, lags):
    """The kernel's fit, in power-of-two units where the data's magnitude needs them. Returns
    (raw with params, R, fitted values and residuals on the data's scale, extra, y and X as
    fitted)."""
    n = y.shape[-1]
    fy = fx = None
    if n:
        fy = _pow2(_sd.max(_sd.abs(y), axis=-1))
        if X.shape[-1]:
            fx = _pow2(_sd.max(_sd.abs(X), axis=-2))
    if fy is not None:
        y = y / (_sd.expand_dims(fy, axis=-1) if isinstance(fy, _sd.ndarray) else fy)
    if fx is not None:
        X = X / _sd.expand_dims(fx, axis=-2)
    raw = list(_k.fit(func, y, X, w, u, intercept, tol, cov, _REFINE_STEPS, codes, gtot, lags, False))
    extra = {}
    if fy is None and fx is None:
        return raw, extra, y, X
    k = raw[0].shape[-1]
    fyv = 1.0 if fy is None else fy
    batch = tuple(raw[0].shape[:-1])
    if fx is None:
        fxk = _sd.ones(batch + (k,))
    else:
        fxk = _sd.concat([_sd.ones(tuple(fx.shape[:-1]) + (1,)), fx], axis=-1) if intercept else fx
        fxk = _sd.broadcast_to(fxk, batch + (k,))
    fyb = _sd.broadcast_to(fyv, batch) if isinstance(fyv, _sd.ndarray) else (_sd.full(batch, fyv) if batch else fyv)
    extra["fy"] = fyb
    extra["fx"] = fxk
    return raw, extra, y, X


def _unscale(raw, extra):
    """raw's params, R, fitted values and residuals from the fit's units to the data's (the
    covariance and the sums of squares stay; the result scales them as it reads them)."""
    if not extra:
        return
    fy, fxk = extra["fy"], extra["fx"]
    fy1 = _expand(fy, 1) if isinstance(fy, _sd.ndarray) else fy
    extra["beff_params_s"] = raw[0]
    extra["resid_s"] = raw[4]
    raw[0] = raw[0] * fy1 / fxk
    raw[2] = raw[2] * _sd.expand_dims(fxk, axis=-2)
    raw[3] = raw[3] * fy1
    raw[4] = raw[4] * fy1


def _linear(func, cls, y, X, weights, intercept, cov_type, maxlags, groups, rtol, nan_policy):
    lags, tol = _check_options(func, intercept, cov_type, maxlags, rtol, nan_policy)
    columns = _frame_columns(X)
    y, X, w, names, _, _, (codes, gtot) = _prepare(func, y, X, weights, nan_policy, groups=groups, cov_type=cov_type,
                                                   intercept=intercept)
    # the diagnostics read y, X and w later: copies (G86), so that changing the caller's
    # arrays (or the numpy arrays they view) does not change the result
    y, X, w = _own(y), _own(X), _own(w)
    raw, extra, ys, Xs = _fit_kernel(func, y, X, w, None, intercept, tol, _COV_TYPES.index(cov_type), codes, gtot,
                                     lags)
    _exact_constant(raw, ys, w, intercept, cov_type)
    _unscale(raw, extra)
    return cls(raw, param_names=_names(names, X.shape[-1], intercept), intercept=intercept, cov_type=cov_type,
               batch=_batch_shape(raw), extra=extra, fitargs=(func, ys, Xs, w, None, tol, codes, gtot),
               columns=columns)


def ols(y, X, /, *, intercept=True, cov_type="nonrobust", maxlags=None, groups=None, rtol=None,
        nan_policy="propagate"):
    """Ordinary least squares of y (..., n) on X (..., n, k) or (n,), an intercept column first."""
    return _linear("ols", OLSResult, y, X, None, intercept, cov_type, maxlags, groups, rtol, nan_policy)


def wls(y, X, /, *, weights, intercept=True, cov_type="nonrobust", maxlags=None, groups=None, rtol=None,
        nan_policy="propagate"):
    """Weighted least squares: minimizes sum w_i (y_i - x_i b)^2. Rows with weight 0 take no
    part in the fit, nobs or the degrees of freedom (R's lm); their fitted values and
    residuals are computed."""
    if weights is None:  # required: the message of any other non-array (_as_real's)
        _as_real(weights, "wls", "weights")
    return _linear("wls", WLSResult, y, X, weights, intercept, cov_type, maxlags, groups, rtol, nan_policy)


def gls(y, X, /, *, sigma, intercept=True, cov_type="nonrobust", maxlags=None, groups=None, rtol=None,
        nan_policy="propagate"):
    """Generalized least squares with the known error covariance sigma (..., n, n): the fit of
    L^-1 y on L^-1 X for sigma = L L^T (Cholesky; its lower triangle is read). Fitted values
    and residuals are on the original scale."""
    func = "gls"
    lags, tol = _check_options(func, intercept, cov_type, maxlags, rtol, nan_policy)
    S, _ = _as_real(sigma, func, "sigma")  # None included: the message of any other non-array
    yy = _read_y(y, func)
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
    columns = _frame_columns(X)
    y, X, _, names, _, keep, (codes, gtot) = _prepare(func, yy, X, None, nan_policy, extra_rows=nan_rows,
                                                      groups=groups, cov_type=cov_type, intercept=intercept)
    # inf before the whitening, which would turn it into NaN or another column's inf
    _check_rows(func, [("y", y), ("X", X)])
    if keep is not None:
        idx = _sd.nonzero(keep)[0]
        S = _sd.take(_sd.take(S, idx, axis=0), idx, axis=1)
    _check_batches(func, [("y", y.shape[:-1]), ("X", X.shape[:-2]), ("sigma", S.shape[:-2])])
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
    raw, extra, ys, Xs = _fit_kernel(func, yt, Xt, None, ones, intercept, tol, _COV_TYPES.index(cov_type), codes, gtot,
                                     lags)
    # the whitened problem's constant: y itself (L^-1 y = y0 L^-1 1 holds exactly only in exact
    # arithmetic); the original-scale fit below then gives fitted y0, residuals y - y0 exactly
    yc = y
    if extra:
        fy = extra["fy"]
        yc = y / (_sd.expand_dims(fy, axis=-1) if isinstance(fy, _sd.ndarray) else fy)
    const = _exact_constant(raw, yc, None, intercept, cov_type, original_scale=False)
    _unscale(raw, extra)
    res = GLSResult(raw, param_names=_names(names, X.shape[-1], intercept), intercept=intercept, cov_type=cov_type,
                    batch=_batch_shape(raw), extra=extra, fitargs=(func, ys, Xs, None, ones, tol, codes, gtot),
                    columns=columns)
    # fitted values and residuals on the original scale (statsmodels' fittedvalues and resid)
    fitted, resid = _k.apply(func, X, res._beff(), intercept, y)
    if raw[3].shape != fitted.shape:  # sigma's batch axes added to y's
        fitted, resid = _sd.broadcast_to(fitted, raw[3].shape), _sd.broadcast_to(resid, raw[3].shape)
    raw[3], raw[4] = fitted, resid
    if const is not None:  # the whitened residuals L^-1 (y - y0 1) are exactly 0 there
        wr = res._wresid()
        res._cache["wresid"] = (_sd.where(_sd.expand_dims(const, axis=-1), 0.0, wr) if res._batch
                                else _sd.zeros(tuple(wr.shape)))
    logdet = 2.0 * _sd.sum(_sd.log(_linalg.diagonal(L)), axis=-1)
    if res._batch:
        logdet = _sd.broadcast_to(logdet, res._batch) if isinstance(logdet, _sd.ndarray) else logdet
    else:
        logdet = float(logdet)
    res._extra["logdet"] = logdet
    return res


def _constant(v, batch):
    """Whether each regression's values in v (..., n) are all equal, broadcast to the batch
    shape: a 1-D v (the other argument carries the batch) gives one answer for all."""
    if v.ndim == 1:
        return _sd.full(batch, float(_sd.max(v)) == float(_sd.min(v)))
    return _sd.broadcast_to(_sd.max(v, axis=-1) == _sd.min(v, axis=-1), batch)


def linregress(x, y, /, *, nan_policy="propagate"):
    """scipy.stats.linregress: the least squares line y = intercept + slope x, with the
    correlation, the two-sided p-value of slope = 0 (Student t, n - 2 degrees of freedom) and
    the two standard errors. A batch takes x and y (..., n). nan_policy: theilslopes' rules
    (G102)."""
    func = "linregress"
    _check_policy(nan_policy, func)
    xa, _ = _as_real(x, func, "x")
    ya = _read_y(y, func)
    if xa.ndim == 0 or ya.ndim == 0:
        raise ValueError(f"{func}(): 'x' and 'y' must have at least 1 dimension")
    if xa.shape[-1] != ya.shape[-1]:
        raise ValueError(_rows_message(func, "x", ya.shape[-1], xa.shape[-1], ya.shape, "values"))
    _check_batches(func, [("x", xa.shape[:-1]), ("y", ya.shape[:-1])])
    _check_rows(func, [("x", xa), ("y", ya)])
    batch = xa.ndim > 1 or ya.ndim > 1
    if nan_policy == "raise":
        if _has_nan(xa, ya):
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
    elif nan_policy == "omit":
        if batch:
            raise ValueError(f"{func}(): nan_policy='omit' takes one regression, not a batch "
                             "(the regressions would keep different points)")
        bad = _sd.logical_or(_sd.isnan(xa), _sd.isnan(ya))
        if bool(_sd.any(bad)):
            keep = _sd.logical_not(bad)
            xa, ya = xa[keep], ya[keep]
    if ya.shape[-1] < 2:
        raise ValueError(f"{func}(): 'x' and 'y' must have length at least 2")
    X = _sd.expand_dims(xa, axis=-1)
    # only an exactly constant x is degenerate here (scipy): rtol 0
    raw, extra, _, _ = _fit_kernel(func, ya, X, None, None, True, 0.0, 0, None, 0, -1)
    _unscale(raw, extra)
    r = OLSResult(raw, param_names=["const", "x1"], intercept=True, cov_type="nonrobust", batch=_batch_shape(raw),
                  extra=extra)
    nan = math.nan
    if not r._batch:
        b0, b1 = float(r.params[0]), float(r.params[1])
        se0, se1 = float(r.bse[0]), float(r.bse[1])
        rsq = r.rsquared
        t1 = float(r.tvalues[1])
        # the slope's sign from the fitted units' t (the slope itself may underflow to 0)
        sgn = t1 if math.isfinite(t1) and t1 != 0 else b1
        rvalue = min(1.0, max(-1.0, math.copysign(math.sqrt(rsq), sgn))) if rsq >= 0 else nan
        pvalue = 2.0 * _StudentT(df=r.df_resid).ccdf(abs(t1))
        # scipy's degenerate cases: x or y constant gives r = NaN, and with it p and both
        # standard errors; a constant x gives no line at all. A NaN regression stays NaN.
        if not r._isnan():
            xconst = float(_sd.max(xa)) == float(_sd.min(xa))
            yconst = float(_sd.max(ya)) == float(_sd.min(ya))
            if yconst and not xconst:
                # the exact line (scipy's): slope +0 and the constant, not the solve's rounding
                b1, b0 = 0.0, float(ya[0])
            if xconst:
                b0 = b1 = nan
            if xconst or yconst:
                rvalue = pvalue = se0 = se1 = nan
        return _results.LinregressResult(b1 + 0.0, b0 + 0.0, rvalue, pvalue, se1, se0)
    b0, b1 = r.params[..., 0], r.params[..., 1]
    se0, se1 = r.bse[..., 0], r.bse[..., 1]
    t1 = r.tvalues[..., 1]
    sgn = _sd.where(_sd.logical_and(_sd.isfinite(t1), t1 != 0.0), t1, b1)
    rvalue = _sd.clip(_sd.copysign(_sd.sqrt(r.rsquared), sgn), min=-1.0, max=1.0)
    pvalue = 2.0 * _StudentT(df=r.df_resid).ccdf(_sd.abs(t1))
    ok = _sd.logical_not(r._isnan())
    xconst = _sd.logical_and(_constant(xa, r._batch), ok)
    yconst = _sd.logical_and(_constant(ya, r._batch), ok)
    degen = _sd.logical_or(xconst, yconst)
    # a constant y: the exact line, slope +0 and the constant (scipy's)
    b1 = _sd.where(yconst, 0.0, b1)
    y0 = _sd.full(r._batch, float(ya[0])) if ya.ndim == 1 else _sd.broadcast_to(ya[..., 0], r._batch)
    b0 = _sd.where(yconst, y0, b0)
    return _results.LinregressResult(_sd.where(xconst, nan, b1) + 0.0, _sd.where(xconst, nan, b0) + 0.0,
                                     _sd.where(degen, nan, rvalue), _sd.where(degen, nan, pvalue),
                                     _sd.where(degen, nan, se1), _sd.where(degen, nan, se0))
