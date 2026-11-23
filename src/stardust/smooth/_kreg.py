"""Kernel regression: kernel_regression and bw_dpill (docs/api/smooth.md §4, docs/design/smooth.md §2.4, §3.4;
decisions SM17, SM35, SM80-SM94).

kernel_regression(X, y) fits the Nadaraya-Watson (degree=0) or local linear (degree=1) estimator with a product
Gaussian kernel whose standard deviations are bw, at every data point, and chooses bw by leave-one-out cross
validation ('cv_ls', statsmodels' default), the corrected AIC ('aicc') or the direct plug-in rule ('dpill', one
variable). The C++ core (csrc/smooth/kreg.cpp on the Gaussian sums of csrc/smooth/gauss_sum.hpp) sorts the points,
takes the common factor exp(-D0 / 2) out of every sum (a far point still has the ratio of its sums), and computes
every point with the same kernel whatever the thread count.
"""

import math
import numbers

from .. import _core as _sd
from .._calls import function
from ._common import alpha as _alpha
from ._common import array, choice, contiguous, integer, points, policy, real, type_name, vector
from ._results import SmootherResult

_core = _sd.smooth
_CRITERIA = {"cv_ls": 1, "aicc": 2}
_RULES = ("cv_ls", "aicc", "dpill")


def _scalar_or(a, shape):
    if shape == ():
        return a.reshape((a.size,))[0]
    return _sd.reshape(a, shape)


class KernelRegressionResult(SmootherResult):
    """The result of kernel_regression(): fittedvalues, resid, nobs, df, leverage, scale, bw, degree, kernel,
    criterion, criterion_value, status, converged; predict(X_new), predict_interval(X_new, alpha=0.05) and
    gradient(X_new) evaluate the same estimator at new points. Not a public constructor (docs/api/smooth.md §7)."""

    TITLE = "Kernel Regression Results"
    METHOD = "Kernel regression"
    TUNING = (
        ("Degree:", "degree"),
        ("Kernel:", "kernel"),
        ("Bandwidth:", "bw"),
        ("Bandwidth rule:", "criterion"),
        ("Criterion value:", "criterion_value"),
        ("Converged:", "converged"),
    )
    REPR = ("nobs", "df", "bw", "degree")

    def __init__(self, data, p, one_d, **fields):
        super().__init__(**fields)
        object.__setattr__(self, "_data", data)  # (Xs, ys, ws, bw) sorted, or None after a NaN (propagate)
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "_one_d", one_d)

    def _cell(self, field):
        if getattr(self, field) is None:
            return "None"
        return super()._cell(field)

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

    def _evaluate(self, X_new, func, what):
        q, shape = self._points(X_new, func)
        data = object.__getattribute__(self, "_data")
        m = q.shape[0]
        if data is None:
            if what == 2:
                return _sd.full((m, object.__getattribute__(self, "_p")), math.nan), shape
            nan = _sd.full((m,), math.nan)
            return ((nan, nan.copy()) if what == 1 else nan), shape
        Xs, ys, ws, bw = data
        return _core.kreg_predict(Xs, ys, ws, bw, self.degree, q, what), shape

    def predict(self, X_new, /):
        """The estimator at new points: 1-D data take any shape (a number gives a float), n-D data (m, p) or one
        point (p,). NaN at points with a NaN."""
        values, shape = self._evaluate(X_new, "predict", 0)
        return _scalar_or(values, shape)

    def predict_interval(self, X_new, /, *, alpha=0.05):
        """(..., 2): the pointwise 1 - alpha interval of the mean, value -+ t scale ||l||, with ||l||^2 =
        sum l_j^2 / w_j of the hat row at the point and t the Student t quantile with nobs - df degrees of freedom.
        The bias is not in it (docs/api/smooth.md §1.3)."""
        a = _alpha(alpha, "predict_interval")
        (values, norm2), shape = self._evaluate(X_new, "predict_interval", 1)
        dof = self.nobs - self.df
        if dof > 0 and math.isfinite(dof):
            from ..stats import StudentT

            q = float(StudentT(df=dof).iccdf(a / 2))
        else:
            q = math.nan
        half = q * self.scale * _sd.sqrt(norm2)
        out = _sd.stack([values - half, values + half], axis=-1)
        return _sd.reshape(out, shape + (2,))

    def gradient(self, X_new, /):
        """The local line's slope in the units of x at new points (degree=1): (m, p), or (p,) for one point; 1-D
        data give the shape of x_new (a number gives a float). statsmodels' fit()[1]."""
        if self.degree == 0:
            raise ValueError("gradient(): the local constant (degree=0) has no gradient; fit with degree=1")
        g, shape = self._evaluate(X_new, "gradient", 2)
        p = object.__getattribute__(self, "_p")
        if object.__getattribute__(self, "_one_d"):
            return _scalar_or(_sd.reshape(g, (g.shape[0],)), shape)
        if shape == ():
            return _sd.reshape(g, (p,))
        return g


def _matrix(X, func, arg):
    """X as a float64 sd.ndarray (a frame DataFrame's columns are the variables)."""
    if type(X).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if isinstance(X, _frame.DataFrame):
            try:
                return X.to_array(dtype=_sd.float64, na_value=math.nan)
            except TypeError, ValueError:
                raise TypeError(f"{func}(): '{arg}' must have numeric columns") from None
    return array(X, func, arg)


def _bandwidth(bw, p):
    """(rule or None, (p,) array or None)."""
    if isinstance(bw, str):
        return choice(bw, "kernel_regression", "bw", _RULES), None
    if isinstance(bw, bool) or bw is None:
        raise TypeError(
            f"kernel_regression(): 'bw' must be a number > 0, {p} numbers > 0, 'cv_ls', 'aicc' or 'dpill', "
            f"not {type_name(bw)}"
        )
    if isinstance(bw, numbers.Real) or (type(bw).__module__ == "numpy" and getattr(bw, "ndim", None) == 0):
        h = real(bw, "kernel_regression", "bw")
        if not (h > 0 and math.isfinite(h)):
            raise ValueError("kernel_regression(): 'bw' must be a finite number > 0")
        return None, _sd.full((p,), h)
    a = array(bw, "kernel_regression", "bw")
    if a.ndim != 1 or a.shape[0] != p:
        raise ValueError(f"kernel_regression(): 'bw' must be a number or have {p} values, not shape {tuple(a.shape)}")
    if not bool(_sd.all(_sd.logical_and(a > 0, _sd.isfinite(a)))):
        raise ValueError("kernel_regression(): 'bw' must be finite numbers > 0")
    return None, contiguous(a.copy())


def _regression_data(func, X, y, weights, nan_policy):
    """(X (n, p), y, w or None, keep or None, has_nan, one_d) checked as the area's data (docs/api/smooth.md §1.2)."""
    nan_policy = policy(nan_policy, func)
    if isinstance(X, numbers.Number):
        raise ValueError(f"{func}(): 'X' must be 1-D or 2-D, not 0-D")
    Xa = _matrix(X, func, "X")
    if Xa.ndim not in (1, 2):
        raise ValueError(f"{func}(): 'X' must be 1-D or 2-D, not {Xa.ndim}-D")
    one_d = Xa.ndim == 1
    n = Xa.shape[0]
    Xa = contiguous(_sd.reshape(Xa, (n, 1)) if one_d else Xa)
    if Xa.shape[1] == 0:
        raise ValueError(f"{func}(): 'X' must have at least 1 column")
    yv = vector(y, func, "y")
    if yv.shape[0] != n:
        raise ValueError(f"{func}(): 'X' and 'y' must have the same number of points, not {n} and {yv.shape[0]}")
    w = None
    if weights is not None:
        w = vector(weights, func, "weights")
        if w.shape[0] != n:
            raise ValueError(f"{func}(): 'weights' must have one value per point ({n}), not {w.shape[0]}")
    if n == 0:
        raise ValueError(f"{func}(): 'X' and 'y' must have at least 1 point")
    for arg, a in [("X", Xa), ("y", yv)] + ([("weights", w)] if w is not None else []):
        if bool(_sd.any(_sd.isinf(a))):
            raise ValueError(f"{func}(): '{arg}' contains infinity")
    if w is not None and bool(_sd.any(w < 0)):
        raise ValueError(f"{func}(): 'weights' must be >= 0")
    bad = _sd.logical_or(_sd.any(_sd.isnan(Xa), axis=1), _sd.isnan(yv))
    if w is not None:
        bad = _sd.logical_or(bad, _sd.isnan(w))
    keep = None
    if bool(_sd.any(bad)):
        if nan_policy == "raise":
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
        if nan_policy == "propagate":
            return Xa, yv, w, None, True, one_d
        keep = _sd.logical_not(bad)
        Xa, yv = contiguous(Xa[keep]), contiguous(yv[keep])
        w = contiguous(w[keep]) if w is not None else None
        if yv.shape[0] == 0:
            raise ValueError(f"{func}(): no point is left without NaN (nan_policy='omit')")
    if w is not None and not bool(_sd.any(w > 0)):
        raise ValueError(f"{func}(): at least one weight must be positive")
    return Xa, yv, w, keep, False, one_d


def _back(values, keep, n):
    if keep is None:
        return values
    out = _sd.full((n,), math.nan)
    out[keep] = values
    return out


def _dpill(x, y, func):
    h, code = _core.bw_dpill(x, y)
    if code == 1:
        raise ValueError(
            f"{func}(): the direct plug-in bandwidth needs at least 6 points after trimming 1% at each end"
        )
    if code == 2:
        raise ValueError(f"{func}(): the direct plug-in bandwidth needs x of positive spread")
    if code == 3:
        raise ValueError(f"{func}(): the direct plug-in bandwidth's quartic block fits need 5 distinct x per block")
    if code == 4:
        raise ValueError(
            f"{func}(): the direct plug-in bandwidth cannot be estimated from these data (a curvature or variance "
            f"estimate is 0 or not finite); give 'bw' as a number or use 'cv_ls'"
        )
    return h


@function
def kernel_regression(X, y, /, *, degree=1, bw="cv_ls", kernel="gaussian", weights=None, nan_policy="propagate"):
    """Kernel regression of y on X with a product Gaussian kernel (statsmodels' KernelReg for continuous variables).

    degree=0 is the Nadaraya-Watson (local constant) estimator, degree=1 the local linear one. bw: the kernel's
    standard deviation per variable (a number for every variable, or p numbers), or the rule that chooses it:
    'cv_ls' (leave-one-out least-squares cross validation), 'aicc' (Hurvich, Simonoff and Tsai's corrected AIC) or
    'dpill' (Ruppert, Sheather and Wand's direct plug-in for one variable and degree=1, without binning). The
    searches are fixed procedures (one variable: 33 points of log h then Brent; several: Nelder-Mead), from
    statsmodels' starting point 1.06 sigma n^(-1/(4+p)). X is (n,) or (n, p). Returns a KernelRegressionResult whose
    arrays are in the input order."""
    degree = integer(degree, "kernel_regression", "degree")
    if degree not in (0, 1):
        raise ValueError("kernel_regression(): 'degree' must be 0 or 1")
    if not isinstance(kernel, str):
        raise TypeError(f"kernel_regression(): 'kernel' must be 'gaussian', not {type_name(kernel)}")
    if kernel != "gaussian":
        raise ValueError(f"kernel_regression(): 'kernel' must be 'gaussian', not {kernel!r}")
    Xa, yv, w, keep, has_nan, one_d = _regression_data("kernel_regression", X, y, weights, nan_policy)
    n_in = keep.shape[0] if keep is not None else Xa.shape[0]
    p = Xa.shape[1]
    rule, h = _bandwidth(bw, p)
    if rule == "dpill":
        if p != 1 or degree != 1:
            raise ValueError("kernel_regression(): bw='dpill' is for one variable and degree=1")
        if weights is not None:
            raise ValueError("kernel_regression(): bw='dpill' takes no weights")
    nobs = Xa.shape[0] if w is None else int(_sd.count_nonzero(w > 0))
    if has_nan:
        nan = _sd.full((n_in,), math.nan)
        bwv = (math.nan if p == 1 else _sd.full((p,), math.nan)) if h is None else (float(h[0]) if p == 1 else h)
        return KernelRegressionResult(
            None,
            p,
            one_d,
            fittedvalues=nan,
            resid=nan.copy(),
            nobs=Xa.shape[0],
            df=math.nan,
            leverage=nan.copy(),
            scale=math.nan,
            bw=bwv,
            degree=degree,
            kernel=kernel,
            criterion=rule,
            criterion_value=math.nan,
            status=0,
            converged=rule is None,
        )
    criterion = 0
    if rule == "dpill":
        h = _sd.full((1,), _dpill(contiguous(_sd.reshape(Xa, (Xa.shape[0],))), yv, "kernel_regression"))
    elif rule is not None:
        if nobs < 2:
            raise ValueError(f"kernel_regression(): bw='{rule}' needs at least 2 points of positive weight")
        criterion = _CRITERIA[rule]
    fitted, leverage, df, rss, cv, aicc, bwo, status, converged, nfev, Xs, ys, ws = _core.kreg_fit(
        Xa, yv, w, degree, h, criterion
    )
    if status == -4:
        raise ValueError(f"kernel_regression(): bw='{rule}' needs every variable of X to have a positive spread")
    if status == -3:
        raise ValueError(f"kernel_regression(): the criterion '{rule}' is not finite at any bandwidth of the search")
    dof = nobs - df
    scale = math.sqrt(rss / dof) if dof > 0 else math.nan
    if criterion == 0:
        cval, status, converged = math.nan, 0, True
    else:
        cval = cv if criterion == 1 else aicc
    resid = yv - fitted
    return KernelRegressionResult(
        (Xs, ys, ws, bwo),
        p,
        one_d,
        fittedvalues=_back(fitted, keep, n_in),
        resid=_back(resid, keep, n_in),
        nobs=nobs,
        df=df,
        leverage=_back(leverage, keep, n_in),
        scale=scale,
        bw=float(bwo[0]) if p == 1 else bwo,
        degree=degree,
        kernel=kernel,
        criterion=rule,
        criterion_value=cval,
        status=int(status),
        converged=bool(converged),
    )


def _criterion(X, y, bw, *, degree=1, criterion="cv_ls", weights=None):
    """The criterion's value at the bandwidth bw (tests and measurements; not public)."""
    Xa, yv, w, _, _, _ = _regression_data("kernel_regression", X, y, weights, "raise")
    _, h = _bandwidth(bw, Xa.shape[1])
    return _core.kreg_criterion(Xa, yv, w, degree, h, _CRITERIA[criterion])


@function
def bw_dpill(x, y, /):
    """Ruppert, Sheather and Wand's direct plug-in bandwidth of the local linear estimator (KernSmooth's dpill,
    computed without its binning on 401 grid points): the standard deviation of the Gaussian kernel. x and y are
    1-D of one length, finite (NaN and infinity are a ValueError)."""
    xv = vector(x, "bw_dpill", "x")
    yv = vector(y, "bw_dpill", "y")
    if xv.shape[0] != yv.shape[0]:
        raise ValueError(f"bw_dpill(): 'x' and 'y' must have the same length, not {xv.shape[0]} and {yv.shape[0]}")
    for arg, a in (("x", xv), ("y", yv)):
        if not bool(_sd.all(_sd.isfinite(a))):
            raise ValueError(f"bw_dpill(): '{arg}' must be finite (NaN or infinity found)")
    return float(_dpill(xv, yv, "bw_dpill"))
