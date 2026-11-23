"""The bandwidth rules and selectors of the kernel density estimate: bw_scott, bw_silverman, bw_nrd0, bw_nrd, bw_sj,
bw_isj, bw_cv_ls, bw_cv_ml (docs/api/smooth.md §5, docs/design/smooth.md §3.6; decisions SM19, SM127-SM132).

Every function takes the data only and returns the kernel's standard deviation (a float, or (p,) for the functions
that take several variables). The closed forms (scott, silverman: scipy's factors; nrd0, nrd: R's rules) are
computed here; Sheather-Jones (R's bw.SJ), ISJ (Botev, Grotowski and Kroese) and the cross-validation searches are
the C++ core's (csrc/smooth/bandwidth.cpp).
"""

import math

from .. import _core as _sd
from .._calls import function
from ._common import choice, contiguous, integer, type_name

_core = _sd.smooth


def _linalg_diag(a):
    from .. import linalg

    return contiguous(linalg.diagonal(a))


def _rows(data, func, arg="data"):
    """(an (n, p) C-contiguous float64 array, one_d) of finite data (NaN or infinity: ValueError)."""
    from ._kreg import _matrix

    if data is None or isinstance(data, (str, bytes, int, float, bool)):
        raise TypeError(
            f"{func}(): '{arg}' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame "
            f"Series or DataFrame, not {type_name(data)}"
        )
    a = _matrix(data, func, arg)
    if a.ndim not in (1, 2):
        raise ValueError(f"{func}(): '{arg}' must be 1-D or 2-D, not {a.ndim}-D")
    one_d = a.ndim == 1
    n = a.shape[0]
    a = contiguous(_sd.reshape(a, (n, 1)) if one_d else a)
    if a.shape[1] == 0:
        raise ValueError(f"{func}(): '{arg}' must have at least 1 column")
    if not bool(_sd.all(_sd.isfinite(a))):
        raise ValueError(f"{func}(): '{arg}' must be finite (NaN or infinity found)")
    return a, one_d


def _vector(x, func):
    a, one_d = _rows(x, func, "x")
    if not one_d:
        raise ValueError(f"{func}(): 'x' must be 1-D, not 2-D")
    if a.shape[0] < 2:
        raise ValueError(f"{func}(): 'x' must have at least 2 points")
    return contiguous(_sd.reshape(a, (a.shape[0],)))


def _weights(weights, n, func):
    if weights is None:
        return None
    from ._common import vector

    w = vector(weights, func, "weights")
    if w.shape[0] != n:
        raise ValueError(f"{func}(): 'weights' must have one value per point ({n}), not {w.shape[0]}")
    if not bool(_sd.all(_sd.isfinite(w))):
        raise ValueError(f"{func}(): 'weights' must be finite (NaN or infinity found)")
    if bool(_sd.any(w < 0)):
        raise ValueError(f"{func}(): 'weights' must be >= 0")
    if not bool(_sd.any(w > 0)):
        raise ValueError(f"{func}(): at least one weight must be positive")
    return w


def _moments(X, w):
    """(sorted points, weights, sum w, w_max / w_min, mean, covariance, neff) of the points of positive weight."""
    Xs, ws, sw, wr = _core.kde_prepare(X, w)
    mean, cov, neff = _core.kde_moments(Xs, ws, sw)
    return Xs, ws, sw, wr, mean, cov, neff


def _factor(rule, neff, p):
    if rule == "scott":
        return neff ** (-1.0 / (p + 4))
    return (neff * (p + 2) / 4.0) ** (-1.0 / (p + 4))


def _rule_sd(func, data, weights, rule):
    X, one_d = _rows(data, func)
    w = _weights(weights, X.shape[0], func)
    if X.shape[0] < 1:
        raise ValueError(f"{func}(): 'data' must have at least 1 point")
    p = X.shape[1]
    *_, cov, neff = _moments(X, w)
    f = _factor(rule, neff, p)
    if one_d:
        return f * math.sqrt(float(cov[0, 0])) if float(cov[0, 0]) >= 0 else math.nan
    return f * _sd.sqrt(_linalg_diag(cov))


@function
def bw_scott(data, /, *, weights=None):
    """Scott's rule as scipy's gaussian_kde has it: n_eff^(-1/(p + 4)) times the weighted sample standard deviation
    (divisor 1 - sum wn^2, n - 1 without weights) of each variable, n_eff = (sum w)^2 / sum w^2. A float for 1-D
    data, (p,) for (n, p) data."""
    return _rule_sd("bw_scott", data, weights, "scott")


@function
def bw_silverman(data, /, *, weights=None):
    """Silverman's rule as scipy's gaussian_kde has it: (n_eff (p + 2) / 4)^(-1/(p + 4)) times the weighted sample
    standard deviation of each variable. A float for 1-D data, (p,) for (n, p) data."""
    return _rule_sd("bw_silverman", data, weights, "silverman")


def _iqr_sd(x):
    q = _sd.quantile(x, _sd.asarray([0.25, 0.75]))
    return float(q[1]) - float(q[0]), float(_sd.std(x, correction=1))


def _nrd0(x):
    iqr, hi = _iqr_sd(x)
    lo = min(hi, iqr / 1.34)
    if not lo:
        lo = hi or abs(float(x[0])) or 1.0
    return 0.9 * lo * x.shape[0] ** -0.2


def _nrd(x):
    iqr, sdv = _iqr_sd(x)
    return 1.06 * min(sdv, iqr / 1.34) * x.shape[0] ** -0.2


@function
def bw_nrd0(x, /):
    """R's bw.nrd0: 0.9 min(sd, IQR / 1.34) n^(-1/5) (sd with divisor n - 1, the quartiles linear: R's type 7); when
    that minimum is 0, sd, then |x_1|, then 1 stand in for it (R's fallbacks). x is 1-D with at least 2 points."""
    return _nrd0(_vector(x, "bw_nrd0"))


@function
def bw_nrd(x, /):
    """R's bw.nrd: 1.06 min(sd, IQR / 1.34) n^(-1/5) (0 when the quartiles coincide). x is 1-D with at least 2
    points."""
    return _nrd(_vector(x, "bw_nrd"))


@function
def bw_sj(x, /, *, method="ste", nb=1000):
    """Sheather and Jones' bandwidth as R's bw.SJ: the pair distances counted in nb bins (binned from n > nb / 2 on,
    as R), method 'ste' solves the equation h = (c1 / S(alpha2 h^(5/7)))^(1/5) (to 1e-13 relative; R's uniroot stops
    at 0.1 lower), 'dpi' is the direct plug-in. The root searched is the smallest sign change of 33 points of log h
    over hmax / 64 .. 64 hmax (hmax = 1.144 scale n^(-1/5), R's interval)."""
    method = choice(method, "bw_sj", "method", ("ste", "dpi"))
    nbins = integer(nb, "bw_sj", "nb")
    if nbins < 1:
        raise ValueError("bw_sj(): 'nb' must be >= 1")
    xv = _vector(x, "bw_sj")
    h, code = _core.bw_sj(xv, 0 if method == "ste" else 1, nbins)
    if code == 2:
        raise ValueError("bw_sj(): the scale min(sd, IQR / 1.349) of 'x' is 0")
    if code in (3, 4):
        raise ValueError(
            "bw_sj(): the sample is too sparse to estimate the density's derivatives (R: 'sample is too "
            "sparse'); use bw_nrd0 or bw_isj"
        )
    if code == 5:
        raise ValueError("bw_sj(): the equation has no root between hmax / 64 and 64 hmax; use method='dpi'")
    return h


@function
def bw_isj(x, /, *, n_grid=16384):
    """Botev, Grotowski and Kroese's improved Sheather-Jones bandwidth: the points binned linearly on n_grid points
    over [min - range / 2, max + range / 2], the DCT-II of the bins and the smallest root t in (0, 0.1] of their
    fixed-point equation (seven stages, N the number of distinct points); the bandwidth is sqrt(t) times the grid's
    width. ValueError when the equation has no root there (few or discrete data): use bw_sj or bw_nrd0."""
    N = integer(n_grid, "bw_isj", "n_grid")
    if N < 2:
        raise ValueError("bw_isj(): 'n_grid' must be >= 2")
    xv = _vector(x, "bw_isj")
    h, code = _core.bw_isj(xv, N)
    if code == 1:
        raise ValueError("bw_isj(): 'x' must have a positive spread")
    if code == 5:
        raise ValueError(
            "bw_isj(): the fixed-point equation has no root in (0, 0.1] (too few or too discrete data); "
            "use bw_sj or bw_nrd0"
        )
    return h


def _href(X):
    """The centre of the cross-validation searches per variable: bw_nrd's value, or 1.06 sd n^(-1/5) when the
    quartiles coincide (decision SM131)."""
    n, p = X.shape
    out = []
    for l in range(p):
        col = contiguous(X[:, l])
        h = _nrd(col)
        if not h > 0:
            h = 1.06 * float(_sd.std(col, correction=1)) * n**-0.2
        out.append(h)
    return out


def _cv(func, data, kind, nb):
    X, one_d = _rows(data, func)
    n, p = X.shape
    if n < 3:
        raise ValueError(f"{func}(): 'data' must have at least 3 points")
    nbins = 0
    if nb is not None:
        nbins = integer(nb, func, "nb")
        if nbins < 1:
            raise ValueError(f"{func}(): 'nb' must be >= 1")
        if p != 1:
            raise ValueError(f"{func}(): 'nb' (R's binned criterion) is for 1-D data")
    href = _href(X)
    if not all(h > 0 and math.isfinite(h) for h in href):
        raise ValueError(f"{func}(): every variable of 'data' must have a positive spread")
    h, status, value, _ = _core.bw_cv(X, kind, nbins, _sd.asarray(href))
    if h is None:
        raise ValueError(f"{func}(): the criterion is not finite at any bandwidth of the search")
    return (float(h[0]) if one_d else h), status


@function
def bw_cv_ls(data, /, *, nb=None):
    """The least-squares cross-validation bandwidth: the h (per variable, a product Gaussian kernel) minimizing
    int f^2 - 2/n sum f_-i(x_i) (statsmodels' cv_ls), computed from every pair (the proven cut only); nb (1-D) uses
    R's bw.ucv instead: the pair distances counted in nb bins and R's formula. The search spans bw_nrd / 64 ..
    64 bw_nrd per variable (1-D: 33 points of log h, then the derivative's root; several: Nelder-Mead). A float for
    1-D data, (p,) for (n, p)."""
    return _cv("bw_cv_ls", data, 1, nb)[0]


@function
def bw_cv_ml(data, /):
    """The likelihood cross-validation bandwidth: the h (per variable) maximizing sum log f_-i(x_i) (statsmodels'
    cv_ml), with the search of bw_cv_ls. A float for 1-D data, (p,) for (n, p)."""
    return _cv("bw_cv_ml", data, 2, None)[0]


def _cv_value(data, h, *, kind="ls", nb=None):
    """The criterion at h (tests and measurements; not public): cv_ls (or R's ucv with nb) or sum log f_-i."""
    X, _ = _rows(data, "bw_cv_ls")
    p = X.shape[1]
    hv = _sd.asarray([float(h)] * p) if isinstance(h, (int, float)) else contiguous(_sd.asarray(h).astype(_sd.float64))
    return _core.bw_cv_value(X, 1 if kind == "ls" else 2, 0 if nb is None else int(nb), hv)
