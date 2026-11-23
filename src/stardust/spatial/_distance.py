"""pdist, cdist, squareform and the fourteen distance functions of sd.spatial (docs/api/spatial.md §2,
docs/design/spatial.md §2.2, §3.1; milestone SPT1).

The arguments are checked here; the pairs are computed by stardust._core.spatial (csrc/spatial/
distance.cpp), every pair by the one kernel of its metric, so a pair has the same bits in pdist, cdist
and the distance functions.
"""

import math

from .. import _core as _sd
from .. import linalg as _linalg
from .._calls import function
from . import _common as C

_core = _sd.spatial


# ---- the metric and its options ---------------------------------------------------------------


def _plan(name, metric, p, w, V, VI, n, stack):
    """(kernel metric, p, w, aux) of a call on data of n columns; stack() gives the rows the default V
    and VI are computed from (pdist's X, cdist's XA over XB)."""
    sd = _sd
    pv = 2.0
    if metric == "minkowski":
        pv = 2.0 if p is None else C.p_value(p, name)
    wv = None if w is None else C.vector(w, name, "w", n)
    aux = None
    if metric == "haversine" and n != 2:
        raise ValueError(f"{name}(): metric='haversine' needs 2 columns (latitude and longitude in radians), not {n}")
    if metric == "seuclidean":
        if V is not None:
            aux = C.vector(V, name, "V", n, positive=True)
        else:
            rows = stack()
            aux = sd.var(rows, axis=0, correction=1)
            # A constant column has no variance: refused as a given V with a zero is. Its computed variance
            # may be a rounding residue of the mean instead of 0 (0.1 three times: 2.9e-34), so the column
            # itself is looked at (one pass; NaN is not equal to itself, so it still propagates).
            constant = rows.shape[0] >= 2 and bool(sd.any(sd.all(rows == rows[:1], axis=0)))
            if constant or bool(sd.any(aux == 0)):
                raise ValueError(
                    f"{name}(): the variance of a column of the data is 0, so the default 'V' is not positive; give 'V'"
                )
    elif metric == "mahalanobis":
        if VI is not None:
            aux = C.matrix_vi(VI, name, n)
        else:
            aux = _default_vi(stack(), name)
    return C.kernel_metric(metric, pv), pv, wv, aux


def _default_vi(X, name):
    """The inverse of the data's covariance (scipy's default VI): more observations than columns. Data with
    NaN or inf give a NaN VI, so every distance is NaN (the values propagate, SPT12; scipy gives NaN too)."""
    sd = _sd
    m, n = X.shape
    if m <= n:
        raise ValueError(
            f"{name}(): the default 'VI' needs more observations than columns, not {m} observations of {n} "
            "columns; give 'VI'"
        )
    cov = sd.cov(X, rowvar=False, correction=1)
    if not isinstance(cov, sd.ndarray):  # one column: a scalar
        cov = sd.asarray([[float(cov)]])
    if not bool(sd.all(sd.isfinite(cov))):  # a NaN or inf in the data: every distance NaN (SPT12, as scipy)
        return sd.full((n, n), math.nan)
    try:
        return _linalg.inv(cov)
    except _linalg.LinAlgError:
        raise _linalg.LinAlgError(
            f"{name}(): the covariance of the data is singular, so the default 'VI' does not exist; give 'VI'"
        ) from None


def _data2(x, name, arg, keep):
    a = C.data(x, name, arg, keep=keep)
    if a.ndim != 2:
        raise ValueError(f"{name}(): '{arg}' must have 2 dimensions, not {a.ndim}")
    return a


# ---- pdist, cdist -----------------------------------------------------------------------------


@function
def pdist(X, /, *, metric="euclidean", p=None, w=None, V=None, VI=None):
    """The distances between the rows of X, pairs i < j row by row (m (m - 1) / 2,)."""
    name = "pdist"
    mt = C.metric(metric, name)
    C.applies(name, mt, p, w, V, VI)
    X = _data2(X, name, "X", mt in C.COUNTING)
    m, n = X.shape
    pairs = m * (m - 1) // 2 if m >= 2 else 0
    km, pv, wv, aux = _plan(name, mt, p, w, V, VI, n, lambda: X if X.dtype == _sd.float64 else X.astype(_sd.float64))
    C.check_alloc(8 * pairs, name)
    return _core._pdist(X, km, pv, wv, aux)


@function
def cdist(XA, XB, /, *, metric="euclidean", p=None, w=None, V=None, VI=None):
    """The distances between every row of XA and every row of XB (mA, mB)."""
    name = "cdist"
    mt = C.metric(metric, name)
    C.applies(name, mt, p, w, V, VI)
    keep = mt in C.COUNTING
    A = _data2(XA, name, "XA", keep)
    B = _data2(XB, name, "XB", keep)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"{name}(): 'XA' and 'XB' must have the same number of columns, not {A.shape[1]} and {B.shape[1]}"
        )
    if keep:
        A, B = C.common_dtype(A, B)
    n = A.shape[1]
    km, pv, wv, aux = _plan(name, mt, p, w, V, VI, n, lambda: _sd.concat([A, B], axis=0))
    C.check_alloc(8 * A.shape[0] * B.shape[0], name)
    return _core._cdist(A, B, km, pv, wv, aux)


# ---- the distance functions -------------------------------------------------------------------


def _paired(name, metric, u, v, *, p=2.0, w=None, aux=None):
    """The distance of u and v along their last axis, the leading axes broadcast: a Python float for two
    1-D inputs, else an array of the batch shape."""
    sd = _sd
    keep = metric in C.COUNTING
    U = C.data(u, name, "u", keep=keep)
    Vv = C.data(v, name, "v", keep=keep)
    n = U.shape[-1]
    if Vv.shape[-1] != n:
        raise ValueError(
            f"{name}(): 'u' and 'v' must have the same length on their last axis, not {n} and {Vv.shape[-1]}"
        )
    if metric == "haversine" and n != 2:
        raise ValueError(
            f"{name}(): 'u' and 'v' must have 2 entries on their last axis (latitude and longitude in radians), not {n}"
        )
    if keep:
        U, Vv = C.common_dtype(U, Vv)
    wv = None if w is None else C.vector(w, name, "w", n)
    if metric == "seuclidean":
        aux = C.vector(aux, name, "V", n, positive=True)
    elif metric == "mahalanobis":
        aux = C.matrix_vi(aux, name, n)
    if U.ndim == 1 and Vv.ndim == 1:
        out = _core._paired(sd.reshape(U, (1, n)), sd.reshape(Vv, (1, n)), metric, p, wv, aux)
        return out[0]
    batch = C.broadcast(U.shape[:-1], Vv.shape[:-1], name, U.shape, Vv.shape)
    P = math.prod(batch)
    full = batch + (n,)
    U2 = sd.reshape(U if tuple(U.shape) == full else sd.broadcast_to(U, full), (P, n))
    V2 = sd.reshape(Vv if tuple(Vv.shape) == full else sd.broadcast_to(Vv, full), (P, n))
    C.check_alloc(8 * P, name)
    return sd.reshape(_core._paired(U2, V2, metric, p, wv, aux), batch)


@function
def braycurtis(u, v, /, *, w=None):
    """sum w |u - v| / sum w |u + v| along the last axis."""
    return _paired("braycurtis", "braycurtis", u, v, w=w)


@function
def canberra(u, v, /, *, w=None):
    """sum w |u - v| / (|u| + |v|) along the last axis (a 0/0 term is 0)."""
    return _paired("canberra", "canberra", u, v, w=w)


@function
def chebyshev(u, v, /, *, w=None):
    """max |u - v| along the last axis (over w > 0 when weighted); NaN propagates."""
    return _paired("chebyshev", "chebyshev", u, v, w=w)


@function
def cityblock(u, v, /, *, w=None):
    """sum w |u - v| along the last axis."""
    return _paired("cityblock", "cityblock", u, v, w=w)


@function
def correlation(u, v, /, *, w=None, centered=True):
    """The cosine distance of u and v after subtracting their (weighted) means (centered=False: none)."""
    C.flag(centered, "correlation", "centered")
    return _paired("correlation", "correlation" if centered else "cosine", u, v, w=w)


@function
def cosine(u, v, /, *, w=None):
    """1 - u.v / (|u| |v|) along the last axis, in [0, 2]."""
    return _paired("cosine", "cosine", u, v, w=w)


@function
def euclidean(u, v, /, *, w=None):
    """sqrt(sum w (u - v)^2) along the last axis."""
    return _paired("euclidean", "euclidean", u, v, w=w)


@function
def hamming(u, v, /, *, w=None):
    """The (weighted) share of positions where u and v differ, comparing bool, int64 and float64 as they are."""
    return _paired("hamming", "hamming", u, v, w=w)


@function
def haversine(u, v, /):
    """The central angle (radians, on the unit sphere) between points given as (latitude, longitude) in
    radians on the last axis."""
    return _paired("haversine", "haversine", u, v)


@function
def jaccard(u, v, /, *, w=None):
    """The (weighted) share of the positions where u or v is nonzero at which exactly one of them is."""
    return _paired("jaccard", "jaccard", u, v, w=w)


@function
def mahalanobis(u, v, VI, /):
    """sqrt((u - v)^T VI (u - v)) along the last axis (NaN where the form is negative)."""
    return _paired("mahalanobis", "mahalanobis", u, v, aux=VI)


@function
def minkowski(u, v, /, *, p=2.0, w=None):
    """(sum w |u - v|^p)^(1/p) along the last axis; p = 1, 2, inf are cityblock, euclidean, chebyshev."""
    pv = C.p_value(p, "minkowski")
    return _paired("minkowski", C.kernel_metric("minkowski", pv), u, v, p=pv, w=w)


@function
def seuclidean(u, v, V, /):
    """sqrt(sum (u - v)^2 / V) along the last axis."""
    return _paired("seuclidean", "seuclidean", u, v, aux=V)


@function
def sqeuclidean(u, v, /, *, w=None):
    """sum w (u - v)^2 along the last axis."""
    return _paired("sqeuclidean", "sqeuclidean", u, v, w=w)


# ---- squareform -------------------------------------------------------------------------------

_FORCE = ("no", "tomatrix", "tovector")


@function
def squareform(X, /, *, force="no", checks=True):
    """A condensed distance vector to the square matrix, or a square distance matrix to its condensed
    vector (scipy's squareform; the dtype kept)."""
    name = "squareform"
    sd = _sd
    a = C.data(X, name, "X", keep=True)
    force = C._choice(force, name, "force", _FORCE)
    C.flag(checks, name, "checks")
    if a.ndim not in (1, 2):
        raise ValueError(f"{name}(): 'X' must have 1 or 2 dimensions, not {a.ndim}")
    if a.ndim == 1:
        if force == "tovector":
            raise ValueError(f"{name}(): force='tovector' needs a square matrix, not a 1-D 'X'")
        k = a.shape[0]
        d = (1 + math.isqrt(1 + 8 * k)) // 2
        if d * (d - 1) // 2 != k:
            raise ValueError(f"{name}(): the length of 'X' ({k}) is not d (d - 1) / 2 for any d")
        C.check_alloc((1 if a.dtype == sd.bool else 8) * d * d, name)
        return _core._squareform_tomatrix(a, d)
    if force == "tomatrix":
        raise ValueError(f"{name}(): force='tomatrix' needs a condensed vector, not a 2-D 'X'")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name}(): 'X' must be square, not of shape {tuple(a.shape)}")
    code, out = _core._squareform_tovector(a, checks)
    if code == 1:
        raise ValueError(f"{name}(): 'X' must be symmetric")
    if code == 2:
        raise ValueError(f"{name}(): 'X' has NaN off the diagonal, so it is not symmetric")
    if code == 3:
        raise ValueError(f"{name}(): 'X' must have a zero diagonal")
    return out
