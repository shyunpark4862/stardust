"""kneighbors and radius_neighbors of sd.spatial (docs/api/spatial.md §3, docs/design/spatial.md §2.3, §3.2;
milestone SPT2).

The arguments are checked here as cdist checks them (the metric and its options, the data, the default V
and VI from XA over XB), and the data must be finite (SPT12). The search is stardust._core.spatial's
(csrc/spatial/neighbors.cpp): every distance it returns or compares is cdist's value of the pair, so the
answer is fixed by the definition whatever the path (direct, or the GEMM pre-selection of §3.2 for
euclidean and sqeuclidean), the thread count or Accelerate's thread mode.
"""

import math

from .. import _core as _sd
from .._calls import function
from .._results import QueryResult
from . import _common as C
from ._distance import _data2, _plan
from ._radius import RadiusResult

_core = _sd.spatial


def _sides(name, XA, XB, metric, p, w, V, VI):
    """(A, B, kernel metric, p, w, aux) as cdist's, with finite data."""
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
    C.finite(A, name, "XA")
    C.finite(B, name, "XB")
    km, pv, wv, aux = _plan(name, mt, p, w, V, VI, A.shape[1], lambda: _sd.concat([A, B], axis=0))
    return A, B, km, pv, wv, aux


def _ranks(k, name):
    """(the largest rank, the ranks as a list or None for 1 .. k, squeeze: k is the int 1)."""
    kk = C.k_ranks(k, name)  # an int, or the ranks as a list
    if isinstance(kk, list):
        return max(kk), kk, False
    return kk, None, kk == 1


def _knn(name, XA, XB, k, metric, p, w, V, VI, distance_upper_bound, path=0):
    sd = _sd
    kmax, ranks, squeeze = _ranks(k, name)
    ub = C.upper_bound(distance_upper_bound, name)
    A, B, km, pv, wv, aux = _sides(name, XA, XB, metric, p, w, V, VI)
    ma, mb = A.shape[0], B.shape[0]
    width = kmax if ranks is None else len(ranks)
    C.check_alloc(16 * ma * width, name)
    kc = max(1, min(kmax, mb))  # ranks past the data points are the fill (inf, mB)
    d, i = _core._kneighbors(A, B, km, pv, wv, aux, kc, ub, path)
    if ranks is None:
        if kmax > kc:
            d = sd.concat([d, sd.full((ma, kmax - kc), math.inf)], axis=1)
            i = sd.concat([i, sd.full((ma, kmax - kc), mb, dtype=sd.int64)], axis=1)
    else:
        if max(ranks) > kc:
            d = sd.concat([d, sd.full((ma, 1), math.inf)], axis=1)
            i = sd.concat([i, sd.full((ma, 1), mb, dtype=sd.int64)], axis=1)
        cols = sd.asarray([min(r - 1, kc) for r in ranks], dtype=sd.int64)
        d, i = d[:, cols], i[:, cols]
    if squeeze:
        d, i = d[:, 0].copy(), i[:, 0].copy()
    d._set_readonly()
    i._set_readonly()
    return QueryResult(d, i)


def _radius(name, XA, XB, r, metric, p, w, V, VI, return_length, path=0):
    C.flag(return_length, name, "return_length")
    A, B, km, pv, wv, aux = _sides(name, XA, XB, metric, p, w, V, VI)
    rr = C.radius(r, name, A.shape[0])
    out = _core._radius_neighbors(A, B, km, pv, wv, aux, rr, return_length, path)
    if return_length:
        return out
    indptr, indices, distances = out
    return RadiusResult(indptr, indices, distances, (A.shape[0], B.shape[0]))


@function
def kneighbors(XA, XB, /, *, k=1, metric="euclidean", p=None, w=None, V=None, VI=None, distance_upper_bound=math.inf):
    """The k nearest rows of XB to every row of XA: QueryResult(distances, indices), each row in (distance,
    index) order, the distances cdist's values; slots without a neighbour closer than distance_upper_bound
    are (inf, len(XB)). k = 1 drops the last axis."""
    return _knn("kneighbors", XA, XB, k, metric, p, w, V, VI, distance_upper_bound)


@function
def radius_neighbors(XA, XB, r, /, *, metric="euclidean", p=None, w=None, V=None, VI=None, return_length=False):
    """The rows of XB within distance r (boundary included) of every row of XA: a RadiusResult (CSR layout,
    indices increasing within a row), or with return_length the counts (int64, one per row of XA)."""
    return _radius("radius_neighbors", XA, XB, r, metric, p, w, V, VI, return_length)
