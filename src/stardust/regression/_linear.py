"""Linear models: ols, wls, gls (docs/api/regression.md §2, docs/design/regression.md §2-§3).

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

from .. import _calls, _results
from .. import _core as _sd
from .. import _summary as _S
from .. import frame as _frame
from .. import linalg as _linalg
from ..htest._gof import JarqueBeraResult, _jb_of_sums
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
        except TypeError, ValueError:
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
                    raise TypeError(
                        f"{func}(): the labels in 'groups' must be strings or numbers, not {_type_name(v)}"
                    ) from None
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
    except TypeError, ValueError:
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
    a, names, _ = _as_real_owned(x, func, arg, per_row)
    return a, names


def _as_real_owned(x, func, arg, per_row=None):
    """_as_real's (array, names) and whether the array is new memory of its own (a list's
    conversion, a dtype conversion, a DataFrame's to_array), which no caller's object can
    change later; False for the caller's sd.ndarray, a view of a numpy array (asarray shares
    a buffer's memory) or a Series' view of its column."""
    names = None
    if isinstance(x, (_frame.Series, _frame.DataFrame)):
        if isinstance(x, _frame.Series):
            names = [str(x.name)] if x.name is not None else None
        else:
            names = [str(c) for c in x.columns]
        try:
            a = x.to_array(dtype=_sd.float64, na_value=math.nan)
        except TypeError, ValueError:  # a string column is a ValueError of to_array
            raise TypeError(f"{func}(): '{arg}' must have numeric columns") from None
        return a, names, isinstance(x, _frame.DataFrame)
    owned = False
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
        except ValueError as e:
            # a shape the arrays do not take (a ragged list, a 0-d numpy array): asarray's own
            # words under the function's name (CX-11)
            msg = str(e)
            head = "asarray(): "
            raise ValueError(f"{func}(): {msg[len(head) :] if msg.startswith(head) else msg}") from None
        except TypeError:
            raise TypeError(
                f"{func}(): '{arg}' must be an sd.ndarray, a sequence of numbers or a frame "
                f"Series or DataFrame, not {_seq_name(x)}"
            ) from None
        owned = isinstance(x, (list, tuple))
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not a complex128 array")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
        owned = True
    return a, names, owned


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


def _ro(v):
    """v as a result keeps and gives it (D-4, API.md §2): an sd.ndarray is made read-only in
    place (no copy; views made from it later are read-only too), so that changing a returned
    field cannot change the fields computed from it later; a tuple's arrays too (a cached pair,
    a result namedtuple). Other values pass."""
    if isinstance(v, _sd.ndarray):
        v._set_readonly()
    elif isinstance(v, tuple):
        for e in v:
            _ro(e)
    return v


# ---- the result object


class _RegressionResult:
    """What every regression result shares (the linear models' and the GLM's): the kernel's
    raw output (params, cov, R, ., ., kept, stats, ...), the cached fields, the coefficients'
    inference and the design checks of predictions. statsmodels' names (G10); the intercept
    is params[0]."""

    __slots__ = (
        "_raw",
        "_cache",
        "_param_names",
        "_cov_type",
        "_intercept",
        "_batch",
        "_extra",
        "_fitargs",
        "_columns",
    )
    # the positions of the shared scalars in the kernel's stats
    _I_NOBS, _I_RANK, _I_ISNAN, _I_NGROUPS, _I_LAGS = _NOBS, _RANK, _ISNAN, _NGROUPS, _LAGS

    def __init__(self, raw, *, param_names, intercept, cov_type, batch, extra=None, fitargs=None, columns=None):
        # exact zero coefficients are +0 (G84): the solves' -0 is a sign of rounding, not data
        if not isinstance(raw, list):
            raw = list(raw)
        raw[0] = raw[0] + 0.0
        for a in raw:
            _ro(a)  # the result's arrays are read-only views (D-4)
        self._raw = raw  # params, cov, R, fitted, resid, kept, stats
        self._columns = columns  # the fit's DataFrame column names, or None
        self._cache = {}
        self._param_names = param_names
        self._cov_type = cov_type
        self._intercept = intercept
        self._batch = batch  # the batch shape, () for one regression
        self._extra = extra or {}
        # what the kernel was given (func, y, X, w, u, rtol, group codes, their count): the
        # diagnostics refactor or refit the same whitened problem
        self._fitargs = fitargs

    # -- the fit's configuration: read-only (G98)
    @property
    def param_names(self):
        """The coefficients' names ('const' for the intercept, then the columns'), as a new list
        at each access: changing the list returned does not change the result."""
        return _fresh(self._param_names)

    @property
    def cov_type(self):
        """The covariance's kind."""
        return self._cov_type

    # -- helpers
    def _cached(self, name, make):
        c = self._cache
        if name not in c:
            c[name] = _ro(make())
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

    # -- the internal units. A fit may run on y / fy and the columns X_j / fx_j (powers of two,
    # exact; only for data whose magnitude would leave the doubles in the fit's sums or
    # covariance, docs/design/regression.md §3.1): params are given back on the data's scale,
    # the covariance stays in the fit's units (_cov_s) and the inference fields are formed from
    # it and scaled once at the end, so that they are scale-equivariant. extra["fy"] is the
    # unit of a fitted value (per regression), extra["fx"] the columns' factors (..., k) and
    # extra["beff_params_s"] the parameters in the fit's units.
    def _fy(self):
        return self._extra.get("fy", 1.0)

    def _unit(self):
        return self._fy()

    def _xfactor(self):
        """None, or the columns' factors (..., k) the internal covariance is in."""
        return self._extra.get("fx")

    def _cov_s(self):
        """The covariance in the fit's units (the scale applied to a classical one)."""

        def make():
            c = self._raw[1]
            if self._robust:
                return c
            return self._per(self.scale, 2) * c

        return self._cached("cov_s", make)

    def _colfactor(self):
        """fy / fx per parameter (..., k): a parameter in the data's units over the fit's; None
        without scaling."""
        fx = self._xfactor()
        if fx is None:
            return None
        return self._per(self._fy(), 1) / fx

    def _perfect(self):
        """Where the fit is perfect with degrees of freedom left (an estimated scale of exactly
        0, df_resid > 0): False here; the families with a scale say."""
        return False

    # -- inference
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
        """params / bse; NaN for a perfect fit (an estimated scale of 0 with df_resid > 0: bse
        is 0, and a coefficient of rounding size over 0 would show a significance the fit cannot
        have; D-3, as nls's G109)."""

        def make():
            if "beff_params_s" not in self._extra:
                t = self.params / self.bse
            else:
                # the fit's own units, beta' / sqrt(C'): neither can leave the doubles there,
                # while a parameter or its standard error in the data's units may overflow
                t = (self._extra["beff_params_s"] + 0.0) / _sd.sqrt(_linalg.diagonal(self._cov_s()))
            bad = self._perfect()
            if isinstance(bad, _sd.ndarray):
                return _sd.where(_expand(bad, 1), math.nan, t)
            return t * math.nan if bad else t

        return self._cached("tvalues", make)

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
        if "beff_params_s" not in self._extra:
            half = q * self.bse
            return _sd.stack([self.params - half, self.params + half], axis=-1)
        # the ends in the fit's own units, beta' -+ q sqrt(C'), then scaled as the parameter is:
        # each end keeps its sign and may overflow to +-inf, never inf - inf = NaN
        b = self._extra["beff_params_s"]
        half = q * _sd.sqrt(_linalg.diagonal(self._cov_s()))
        fy = self._fy()
        fy1 = _expand(fy, 1) if isinstance(fy, _sd.ndarray) else fy
        fxk = self._xfactor()
        return _sd.stack([(b - half) * fy1 / fxk, (b + half) * fy1 / fxk], axis=-1)

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

    def _cov_factor(self):
        """(U, s) with the covariance in the fit's units s^2 (U^T U)^-1 on the kept columns (U
        the identity on the aliased ones), for an accurate quadratic form (G7a); None where the
        covariance is not of that form (robust ones, rlm, quantreg, nls)."""
        return None

    # -- linear hypotheses (G7a, docs/api/regression.md §9.1)
    def t_test(self, R, /, q=None):
        """TTestResult of each row of R params = q (R (m, k) or one row (k,), q (m,) or a
        number, default 0): effect = R params, its sd, tvalue, the two-sided pvalue with the
        coefficients' reference distribution, df, and conf_int(*, alpha=0.05) of effect. A row
        with a nonzero weight on an aliased parameter is NaN."""
        from . import _tests

        return _tests.t_test(self, R, q)

    def wald_test(self, R, /, q=None):
        """WaldTestResult of the joint hypothesis R params = q: F = W / m against F(m,
        df_resid_inference), or W against chi-square(m) for a GLM whose scale is fixed. Rows
        that are linear combinations of earlier ones are dropped (m is the rank); a row on an
        aliased parameter makes the test NaN."""
        from . import _tests

        return _tests.wald_test(self, R, q)

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


def _fresh(v):
    """A configuration field as the result gives it: a list is copied at each access, so that
    changing the list returned does not change the result. Any other value is returned as it
    is (an array field is a read-only view, which cannot be changed)."""
    if isinstance(v, list):
        return list(v)
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

    def _perfect(self):
        """A perfect fit (ssr 0, G95) with df_resid > 0: t and p are NaN there (D-3)."""
        z = self._ssr_s() == 0.0
        df = self.df_resid
        if self._batch:
            return _sd.logical_and(z, df > 0.0)
        return bool(z) and df > 0

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
            e = self._wresid_raw()
            zero = self._ssr_s() == 0.0
            if self._batch:
                return _sd.where(_expand(zero, 1), 0.0, e)
            return _sd.zeros(tuple(e.shape)) if zero else e

        return self._cached("wresid", make)

    def _wresid_raw(self):
        """The whitened residuals in the fit's units as computed (a GLS fit of a constant y:
        exactly 0, its whitened residuals L^-1 (y - y0 1))."""

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
            const = self._extra.get("const")
            if const is not None:
                if self._batch:
                    e = _sd.where(_sd.expand_dims(const, axis=-1), 0.0, e)
                else:
                    e = _sd.zeros(tuple(e.shape))
            return e

        return self._cached("wresid_raw", make)

    @property
    def wresid(self):
        """The whitened residuals (..., n) on the data's scale (statsmodels' wresid): resid (OLS),
        sqrt(w) resid (WLS; 0 at the rows with weight 0), L^-1 resid (GLS). The residuals the
        diagnostics use, e.g. for sd.htest's normality tests."""

        def make():
            e = self._wresid_raw()
            fy = self._fy()
            if isinstance(fy, float) and fy == 1.0:
                return e
            return e * (_expand(fy, 1) if isinstance(fy, _sd.ndarray) else fy)

        return self._cached("wresid_public", make)

    def _cov_factor(self):
        if self._robust:
            return None

        def make():
            from ._tests import _unit_triangle

            R = self._raw[2]
            fx = self._xfactor()
            if fx is not None:
                R = R / _sd.expand_dims(fx, axis=-2)
            return _unit_triangle(R, self._raw[5]), _sqrt(self._scale_s())

        return self._cached("cov_factor", make)

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
        double): 0 Durbin-Watson's numerator, 1-3 the central sums of powers 2, 3, 4, 4 the mean."""

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
        """JarqueBeraResult (sd.htest's class, G165), fields statistic and pvalue, unpacking as
        (statistic, pvalue): n/6 (S^2 + (K - 3)^2 / 4) with the skewness S and
        kurtosis K of the whitened residuals (central moments over n, summed in double-double),
        against chi-square with 2 degrees of freedom (statsmodels' jarque_bera of wresid). The
        sums and the statistic are sd.htest's Jarque-Bera kernel (H17): the bits of
        sd.htest.jarque_bera on the same residuals; NaN where they are constant by sd.stats skew's
        rule (H20)."""

        def make():
            sums = (self._resid_stat(i) for i in (1, 2, 3, 4))
            jb, p = _jb_of_sums(self.nobs, *sums)
            info = {} if self._batch else {"nobs": self.nobs}  # summary() (G191)
            return JarqueBeraResult._make(batch=self._batch, info=info, statistic=jb, pvalue=p)

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
            r = _results.BreuschPaganResult(lm, p, a.fvalue, a.f_pvalue)
            r._info = {"pairs": [("Studentized:", str(studentize))]}  # summary() (G191)
            if not self._batch:
                r._info["nobs"] = self.nobs
            return r

        return self._cached(("breusch_pagan", studentize), make)

    # -- diagnostics added in G7a (docs/api/regression.md §9.2)
    @property
    def vif(self):
        """The variance inflation factors (..., k) at the places of params: 1 / (1 - R_j^2), R_j^2
        the centered R^2 of column j on the intercept and the other columns (the whitened
        design; an intercept is put in for the auxiliary regressions of a model without one).
        NaN at the intercept and the aliased columns."""
        from . import _diagnostics

        return self._cached("vif", lambda: _diagnostics.vif(self))

    @property
    def dffits(self):
        """t_i sqrt(h_i / (1 - h_i)) (..., n), t the external studentized residual (R's dffits,
        statsmodels' dffits[0]); 0 for the rows with weight 0."""

        def make():
            h = self.leverage
            return self.resid_studentized_external * _sd.sqrt(h / (1.0 - h))

        return self._cached("dffits", make)

    @property
    def dfbetas(self):
        """(..., n, k): (beta - beta_(i))_j / (s_(i) sqrt([(X^T X)^-1]_jj)) for each row i and
        parameter j, the intercept included (R's dfbetas, statsmodels' dfbetas), from the fit's
        thin Q (no refit). NaN for the aliased parameters, 0 for the rows with weight 0."""
        from . import _diagnostics

        return self._cached("dfbetas", lambda: _diagnostics.dfbetas(self))

    def white_test(self, *, cross_terms=True):
        """WhiteTestResult(lm, lm_pvalue, fvalue, f_pvalue): e~^2 on a constant, the whitened
        design's columns, their squares and (cross_terms) their cross products."""
        from . import _diagnostics

        return self._cached(("white_test", cross_terms), lambda: _diagnostics.white_test(self, cross_terms))

    def goldfeld_quandt(self, *, split=0.5, drop=0, order_by=None, alternative="greater"):
        """GoldfeldQuandtResult(fvalue, pvalue, alternative): the fit's rows (ordered by
        order_by) split into the first `split` rows (or that fraction) and the rows after `drop`
        more; F = (SSR_2 / df_2) / (SSR_1 / df_1) against F(df_2, df_1). ols and wls only."""
        from . import _diagnostics

        return _diagnostics.goldfeld_quandt(self, split, drop, order_by, alternative)

    def reset_test(self, *, power=(2, 3), kind="fitted"):
        """ResetTestResult(fvalue, pvalue, df_num, df_denom): Ramsey's RESET, the F test of the
        powers of the fitted values (kind='fitted') or of each column (kind='regressor') added
        to the model (a robust covariance: the Wald F with the extended model's)."""
        from . import _diagnostics

        return _diagnostics.reset_test(self, power, kind)

    # -- nested models (G7a, docs/api/regression.md §9.1)
    def compare_f_test(self, small, /):
        """CompareFTestResult(fvalue, pvalue, df_num, df_denom) of this model against the
        nested `small` one fitted to the same data: F = ((SSR_s - SSR_b) / (r_b - r_s)) / (SSR_b
        / df_resid_b)."""
        from . import _tests

        return _tests.compare_f_test(self, small)

    def compare_lr_test(self, small, /):
        """CompareLRTestResult(lr, pvalue, df) of this model against the nested `small` one:
        lr = -2 (llf_s - llf_b) against chi-square(r_b - r_s)."""
        from . import _tests

        return _tests.compare_lr_test(self, small)

    # -- summary (G13)
    def summary(self, *, alpha=0.05):
        """A text table like statsmodels' OLS summary: the fit, the coefficients with their
        standard errors, t, p and 1 - alpha intervals, Durbin-Watson, Jarque-Bera and the
        condition number. A batch: a table of the fit statistics and one of the coefficients,
        one row per regression (G193)."""
        alpha = _check_alpha(alpha, "summary")
        if self._batch:
            return _batch(self)
        return _summary(self, alpha)

    def __repr__(self):
        name = type(self).__name__
        if self._batch:
            return f"{name}(batch shape {tuple(self._batch)}, {self.params.shape[-1]} parameters)"
        return f"{name}(nobs={self.nobs:g}, rank={self.rank}, rsquared={self.rsquared:.6g}, params={self.params})"


_calls.methods(_RegressionResult)
_calls.methods(_LinearResult)


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


# ---- summary (G13): statsmodels' layout and number formats, drawn by stardust._summary (G189)

_WIDTH = _S.WIDTH
_forg = _S.forg
_count = _S.count
_pairs = _S.pairs
_quantile_label = _S.quantile_label
_COV_NOTES = {
    "nonrobust": "Standard errors assume that the covariance matrix of the errors is correctly specified.",
    "HC0": "Standard errors are heteroscedasticity robust (HC0).",
    "HC1": "Standard errors are heteroscedasticity robust (HC1).",
    "HC2": "Standard errors are heteroscedasticity robust (HC2).",
    "HC3": "Standard errors are heteroscedasticity robust (HC3).",
}


def _coef_table(r, alpha, stat="t"):
    """The coefficient table of a model's summary: coef, std err, t (or z), P>|t| and the
    1 - alpha interval per parameter."""
    names = [str(v) for v in r.param_names]
    ci = r.conf_int(alpha=alpha)
    params, bse, tv, pv = (r.params, r.bse, r.tvalues, r.pvalues)
    rows = []
    for j, nm in enumerate(names):
        cells = [
            _S.forg(params[j], 4),
            _S.forg(bse[j]),
            _S.forg(tv[j]),
            _S.pvalue(pv[j]),
            _S.forg(ci[j, 0]),
            _S.forg(ci[j, 1]),
        ]
        rows.append((nm, cells))
    return _S.Table(["coef", "std err", stat, f"P>|{stat}|"] + _S.interval_columns(alpha), rows)


def _aliased_note(r, names, what="linearly dependent on earlier columns", tail="not estimated"):
    aliased = [nm for nm, k in zip(names, r._raw[5].tolist()) if not k]
    if aliased:
        return [f"Aliased ({what}), {tail}: " + ", ".join(aliased) + "."]
    return []


def _batch_summary(r, title, left, stats, right=None):
    """The summary of a batch of regressions (G193): the fit's common description, then a
    table of the per-regression statistics `stats` ((field name, formatter) or (name,
    formatter, value of lane i)) and one of the coefficients, one row per regression (a long
    batch cut to its first and last rows)."""
    shape = tuple(r._batch)
    n = _S.lanes(shape)
    if right is None:
        right = [("Batch shape:", _S.shape_text(shape))]
    right = right + [("Df Model:", _S.count(_first(r.df_model)))]
    cols = [(st[2] if len(st) > 2 else _S.flat(getattr(r, st[0]), n), st[1]) for st in stats]
    t1 = _S.Table([st[0] for st in stats], _S.lane_rows(shape, cols))
    names = [str(v) for v in r.param_names]
    P = _sd.reshape(r.params, (n, len(names)))
    rows = []
    for i in _S.shown(n):
        if i is None:
            rows.append(None)
            continue
        row = P[i].tolist()
        rows.append((_S.lane_label(i, shape), [_S.forg(v, 4) for v in row]))
    t2 = _S.Table(names, rows, stub_head="params")
    return _S.render(title, header=(left, right), tables=[t1, t2])


def _first(v):
    """A field the regressions of a batch share (df_model): its first value."""
    if isinstance(v, _sd.ndarray):
        return _sd.reshape(v, (v.size,))[0] if v.size else math.nan
    return v


def _summary(r, alpha=0.05):
    name = type(r).__name__[:3]
    left = [
        ("Model:", name),
        ("Method:", "Least Squares"),
        ("No. Observations:", _count(r.nobs)),
        ("Df Residuals:", _count(r.df_resid)),
        ("Df Model:", _count(r.df_model)),
        ("Covariance Type:", r.cov_type),
    ]
    right = [
        ("R-squared:", f"{r.rsquared:#.3f}"),
        ("Adj. R-squared:", f"{r.rsquared_adj:#.3f}"),
        ("F-statistic:", f"{r.fvalue:#.4g}"),
        ("Prob (F-statistic):", f"{r.f_pvalue:#.3g}"),
        ("Log-Likelihood:", f"{r.llf:#.5g}"),
        ("AIC:", f"{r.aic:#.4g}"),
        ("BIC:", f"{r.bic:#.4g}"),
    ]
    names = [str(v) for v in r.param_names]
    jb = r.jarque_bera()
    footer = (
        [("Durbin-Watson:", f"{r.durbin_watson:#.3f}"), ("Prob(JB):", f"{jb.pvalue:#.3g}")],
        [("Jarque-Bera (JB):", f"{jb.statistic:#.3f}"), ("Cond. No.", f"{r.condition_number:#.3g}")],
    )
    notes = []
    ct = r.cov_type
    if ct == "HAC":
        notes.append(
            f"Standard errors are heteroscedasticity and autocorrelation robust (HAC) using "
            f"{_count(r._stat(_LAGS))} lags (Bartlett kernel) and without small sample correction."
        )
    elif ct == "cluster":
        g = r._stat(_NGROUPS)
        notes.append(
            f"Standard errors are robust to cluster correlation (cluster) with {_count(g)} groups; "
            f"t and F use {_count(g - 1)} degrees of freedom."
        )
    else:
        notes.append(_COV_NOTES[ct])
    if not r._isnan():
        notes += _aliased_note(r, names)
    return _S.render(
        f"{name} Regression Results",
        header=(left, right),
        tables=[_coef_table(r, alpha)],
        footer=footer,
        notes=notes,
    )


def _batch(r):
    name = type(r).__name__[:3]
    left = [("Model:", name), ("Method:", "Least Squares"), ("Covariance Type:", r.cov_type)]
    stats = [
        ("nobs", _S.count),
        ("rsquared", _S.forg),
        ("fvalue", _S.forg),
        ("f_pvalue", _S.pvalue),
        ("llf", _S.forg),
    ]
    return _batch_summary(r, f"{name} Regression Results", left, stats)


def _arg(v):
    """A distribution parameter or input: an sd.ndarray or a Python float."""
    return v if isinstance(v, _sd.ndarray) else float(v)


# ---- the fits


def _rows_message(func, arg, n, got, yshape, unit="rows"):
    """ "'X' must have 5 rows like 'y', not 4" (1 row, 1 value), and for a y of shape (..., m, 1)
    (one row per regression: a column vector read under the batch rules) what to give
    instead (decision 2026-11-19, G105)."""
    noun = unit[:-1] if n == 1 else unit
    msg = f"{func}(): '{arg}' must have {n} {noun} like 'y', not {got}"
    if n == 1 and len(yshape) >= 2 and got != 1:
        msg += (
            f" ('y' of shape {tuple(yshape)} is a batch of regressions with 1 row each; give a "
            "1-dimensional 'y' for one regression)"
        )
    return msg


def _read_y(y, func):
    """y as an sd.ndarray: a one-column frame DataFrame is read as that column (n,) (G105)."""
    return _read_y_owned(y, func)[0]


def _read_y_owned(y, func):
    """(_read_y's array, whether it is new memory of its own: _as_real_owned)."""
    a, _, owned = _as_real_owned(y, func, "y")
    if isinstance(y, _frame.DataFrame) and len(list(y.columns)) == 1:
        a = _sd.reshape(a, (a.shape[0],))
    return a, owned


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


def _prepare(
    func, y, X, weights, nan_policy, extra_rows=None, groups=None, cov_type="nonrobust", intercept=None, own=False
):
    """Converted and checked y, X (..., n, p), weights and cluster groups; NaN rows (and rows
    with a missing group label) dropped for 'omit'. Returns (y, X, w, names, batch_shape,
    keep_rows or None, (codes, gtot)). own: y, X and w are arrays of the result's own (G86), so
    that changing the caller's arrays (or the numpy arrays they view) later changes nothing: a
    copy of those the conversion did not already make new."""
    y, own_y = _read_y_owned(y, func)
    if y.ndim == 0:
        raise ValueError(f"{func}(): 'y' must have at least 1 dimension")
    X, names, own_x = _as_real_owned(X, func, "X")
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
    own_w = True
    if weights is not None:
        w, _, own_w = _as_real_owned(weights, func, "weights", per_row=f"{n} value{'' if n == 1 else 's'} like 'y'")
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
            raise ValueError(
                f"{func}(): nan_policy='omit' takes one regression, not a batch "
                "(the regressions would keep different rows)"
            )
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
            own_y = own_x = own_w = True  # the selection is new memory
            if w is not None:
                w = w[keep]
            if codes is not None:
                codes = codes[keep]
    if own:
        y = y if own_y else y.copy()
        X = X if own_x else X.copy()
        w = w if (w is None or own_w) else w.copy()
    return y, X, w, names, batch, keep, (codes, gtot)


def _check_params(func, p, intercept):
    """A model must have a parameter (G101): X's columns or the intercept."""
    if p == 0 and not intercept:
        raise ValueError(f"{func}(): the model has no parameters (no columns in 'X' and intercept=False)")


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
    const = _sd.logical_and(
        _sd.logical_and(hi == lo, _sd.isfinite(hi)), _sd.logical_and(st[..., _ISNAN] == 0.0, kept[..., 0])
    )
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
    zero_cols = _sd.logical_or(
        _sd.logical_or(_sd.arange(st.shape[-1]) == _SSR, _sd.arange(st.shape[-1]) == _CTSS),
        _sd.arange(st.shape[-1]) == _ESSC,
    )
    raw[6] = _sd.where(_sd.logical_and(c1, zero_cols), 0.0, st)
    if original_scale:
        h1 = _sd.expand_dims(hi, axis=-1)
        raw[3] = _sd.where(c1, _sd.broadcast_to(h1, tuple(raw[3].shape)), raw[3])
        raw[4] = _sd.where(c1, _sd.broadcast_to(y - h1, tuple(raw[4].shape)), raw[4])
    return const


# Data whose magnitude leaves [2^-256, 2^256] is fitted in power-of-two units (exact): their
# squares (sums of squares) and the covariance's (R^T R)^-1 would leave the doubles otherwise.
_SAFE_HI, _SAFE_LO = 2.0**256, 2.0**-256


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


def _y_units(y):
    """(y / fy, fy): fy the power-of-two factor of each regression's y (_pow2 of max |y_i|),
    None (and y as given) when every y is inside [2^-256, 2^256]. For the fits that are
    equivariant under y's scale alone (rlm, quantreg)."""
    if y.shape[-1] == 0:
        return y, None
    fy = _pow2(_sd.max(_sd.abs(y), axis=-1))
    if fy is None:
        return y, None
    return y / (_sd.expand_dims(fy, axis=-1) if isinstance(fy, _sd.ndarray) else fy), fy


def _y_unscale(raw, fy, stat_cols, lead=0):
    """raw (a kernel's list: params, cov, R, fitted, resid, kept, stats, ...) of a fit of y / fy
    back to y's units: params, fitted values, residuals and the stats in `stat_cols` times fy;
    the covariance stays in the fit's units (the result scales it as it reads it). lead: axes
    between the batch and the fields' own (quantreg's quantile axis). Returns extra."""
    params = raw[0]
    batch = tuple(params.shape[:-1])
    if isinstance(fy, _sd.ndarray):
        fyl = _sd.reshape(fy, tuple(fy.shape) + (1,) * lead)
        fyb = _sd.broadcast_to(fyl, batch)
    else:
        fyb = _sd.full(batch, fy) if batch else fy
    f1 = _expand(fyb, 1) if isinstance(fyb, _sd.ndarray) else fyb
    extra = {"fy": fyb, "fx": _sd.ones(batch + (params.shape[-1],)), "beff_params_s": params}
    raw[0] = params * f1
    raw[3] = raw[3] * f1
    raw[4] = raw[4] * f1
    st = raw[6]
    cols = _sd.arange(st.shape[-1])
    sel = cols == stat_cols[0]
    for c in stat_cols[1:]:
        sel = _sd.logical_or(sel, cols == c)
    raw[6] = _sd.where(sel, st * f1, st)
    return extra


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
    # the diagnostics read y, X and w later: arrays of the result's own (G86), so that changing
    # the caller's arrays (or the numpy arrays they view) does not change the result
    prepared = _prepare(
        func, y, X, weights, nan_policy, groups=groups, cov_type=cov_type, intercept=intercept, own=True
    )
    y, X, w, names, _, keep, (codes, gtot) = prepared
    raw, extra, ys, Xs = _fit_kernel(func, y, X, w, None, intercept, tol, _COV_TYPES.index(cov_type), codes, gtot, lags)
    _exact_constant(raw, ys, w, intercept, cov_type)
    _unscale(raw, extra)
    extra["maxlags"] = lags
    if keep is not None:
        extra["keep"] = keep
    return cls(
        raw,
        param_names=_names(names, X.shape[-1], intercept),
        intercept=intercept,
        cov_type=cov_type,
        batch=_batch_shape(raw),
        extra=extra,
        fitargs=(func, ys, Xs, w, None, tol, codes, gtot),
        columns=columns,
    )


@_calls.function
def ols(y, X, /, *, intercept=True, cov_type="nonrobust", maxlags=None, groups=None, rtol=None, nan_policy="propagate"):
    """Ordinary least squares of y (..., n) on X (..., n, k) or (n,), an intercept column first."""
    return _linear("ols", OLSResult, y, X, None, intercept, cov_type, maxlags, groups, rtol, nan_policy)


@_calls.function
def wls(
    y,
    X,
    /,
    *,
    weights,
    intercept=True,
    cov_type="nonrobust",
    maxlags=None,
    groups=None,
    rtol=None,
    nan_policy="propagate",
):
    """Weighted least squares: minimizes sum w_i (y_i - x_i b)^2. Rows with weight 0 take no
    part in the fit, nobs or the degrees of freedom (R's lm); their fitted values and
    residuals are computed."""
    if weights is None:  # required: the message of any other non-array (_as_real's)
        _as_real(weights, "wls", "weights")
    return _linear("wls", WLSResult, y, X, weights, intercept, cov_type, maxlags, groups, rtol, nan_policy)


@_calls.function
def gls(
    y,
    X,
    /,
    *,
    sigma,
    intercept=True,
    cov_type="nonrobust",
    maxlags=None,
    groups=None,
    rtol=None,
    nan_policy="propagate",
):
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
        raise ValueError(
            f"{func}(): nan_policy='omit' takes one regression, not a batch (the regressions would keep different rows)"
        )
    nan_rows = None
    if nan_policy == "omit" and _has_nan(S):
        # observation i has NaN when sigma's row or column i has one
        nanS = _sd.isnan(S)
        nan_rows = _sd.logical_or(_sd.any(nanS, axis=1), _sd.any(nanS, axis=0))
    columns = _frame_columns(X)
    y, X, _, names, _, keep, (codes, gtot) = _prepare(
        func, yy, X, None, nan_policy, extra_rows=nan_rows, groups=groups, cov_type=cov_type, intercept=intercept
    )
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
        bad = _sd.logical_or(
            _sd.logical_or(
                _sd.any(_sd.isnan(y), axis=-1, keepdims=True),
                _sd.any(_sd.isnan(X), axis=(-2, -1), keepdims=True)[..., 0],
            ),
            bad_s[..., 0],
        )
        y = _sd.where(_sd.isnan(y), 0.0, y)
        X = _sd.where(_sd.isnan(X), 0.0, X)
    if S.size and bool(_sd.any(_sd.isinf(S))):
        raise ValueError("gls(): 'sigma' contains infinity")
    try:
        L = _linalg.cholesky(S)
    except _linalg.LinAlgError as e:
        msg = str(e)
        at = msg[msg.find(" (at batch index") :] if " (at batch index" in msg else ""
        raise _linalg.LinAlgError(f"gls(): 'sigma' is not positive definite{at}") from None
    yt = _linalg.solve_triangular(L, _sd.expand_dims(y, axis=-1), upper=False)[..., 0]
    if bad is not None:
        yt = _sd.where(bad, math.nan, yt)
    Xt = _linalg.solve_triangular(L, X, upper=False)
    ones = _linalg.solve_triangular(L, _sd.ones((m, 1)), upper=False)[..., 0]
    raw, extra, ys, Xs = _fit_kernel(
        func, yt, Xt, None, ones, intercept, tol, _COV_TYPES.index(cov_type), codes, gtot, lags
    )
    # the whitened problem's constant: y itself (L^-1 y = y0 L^-1 1 holds exactly only in exact
    # arithmetic); the original-scale fit below then gives fitted y0, residuals y - y0 exactly
    yc = y
    if extra:
        fy = extra["fy"]
        yc = y / (_sd.expand_dims(fy, axis=-1) if isinstance(fy, _sd.ndarray) else fy)
    const = _exact_constant(raw, yc, None, intercept, cov_type, original_scale=False)
    _unscale(raw, extra)
    res = GLSResult(
        raw,
        param_names=_names(names, X.shape[-1], intercept),
        intercept=intercept,
        cov_type=cov_type,
        batch=_batch_shape(raw),
        extra=extra,
        fitargs=(func, ys, Xs, None, ones, tol, codes, gtot),
        columns=columns,
    )
    # fitted values and residuals on the original scale (statsmodels' fittedvalues and resid)
    fitted, resid = _k.apply(func, X, res._beff(), intercept, y)
    if raw[3].shape != fitted.shape:  # sigma's batch axes added to y's
        fitted, resid = _sd.broadcast_to(fitted, raw[3].shape), _sd.broadcast_to(resid, raw[3].shape)
    raw[3], raw[4] = _ro(fitted), _ro(resid)
    if const is not None:  # the whitened residuals L^-1 (y - y0 1) are exactly 0 there
        res._extra["const"] = const
    res._extra["L"] = L
    res._extra["X0"] = X.copy()
    res._extra["maxlags"] = lags
    if keep is not None:
        res._extra["keep"] = keep
    logdet = 2.0 * _sd.sum(_sd.log(_linalg.diagonal(L)), axis=-1)
    if res._batch:
        logdet = _sd.broadcast_to(logdet, res._batch) if isinstance(logdet, _sd.ndarray) else logdet
    else:
        logdet = float(logdet)
    res._extra["logdet"] = logdet
    return res
