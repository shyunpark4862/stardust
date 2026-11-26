"""k-medoids (docs/api/cluster.md §3; docs/design/cluster.md §2.3, §3.4, §4.3; decisions CL12, CL26 and those of
milestone CL2 in §1.1): R cluster's PAM (BUILD and SWAP, pam(variant = "original")) and CLARA.

The medoids are points of the data; the objective is the mean distance of the points to their nearest medoid (R's
pam objective). Ties go to the smaller key (CL7: the points' lexicographic order; for precomputed distances the
input index), so the same points in another order give the same medoids. SWAP finds R's best swap with FastPAM1's
reuse of the distances (csrc/cluster/kmedoids.cpp). CLARA (R's clara) runs PAM on samples drawn with rng (R's own
sampler cannot be matched) and keeps the best over all points.
"""

from .. import _calls
from .. import _core as _sd
from .._calls import function
from ._common import (
    check_alloc,
    check_rng,
    choice,
    contiguous,
    distances,
    integer,
    key,
    labels_back,
    no_rng,
    points,
    positive_int,
)
from ._common import (
    metric as _metric,
)
from ._common import (
    rows as _rows,
)
from ._results import ClusterResult

_core = _sd.cluster
METHODS = ("pam", "clara")
_ON_THE_FLY = {"euclidean": 0, "cityblock": 1}


class KMedoidsResult(ClusterResult):
    """The result of kmedoids (docs/api/cluster.md §3).

    Fields: labels ((n,) int64, first-seen numbering; -1 for the rows nan_policy='omit' left out), n_clusters (k),
    nobs, omitted, medoids ((k,) int64: the input rows of the medoids, medoids[j] that of label j), medoid_points
    ((k, p) observations; None for precomputed distances), objective (the mean distance to the nearest medoid after
    SWAP, R's objective["swap"]), build_objective (after BUILD; CLARA: the kept sample's BUILD medoids over all
    points), n_swaps, converged (False when max_iter swaps stopped SWAP), method and metric.
    """

    _TITLE = "K-Medoids Results"
    _METHOD = "k-medoids ({method})"
    _OBJECTIVE = ("Mean distance:", "objective")
    _TUNING = (("Metric:", "metric"), ("BUILD objective:", "build_objective"), ("Swaps:", "n_swaps"))
    _COLUMNS = (("Medoid", "medoids"),)
    _REPR = ("method", "nobs", "n_clusters", "objective")

    def predict(self, X_new, /):
        """The nearest medoid of each new point ((m,) int64; the medoid of smaller key on a tie). Observations only:
        a fit on precomputed distances has no points to measure new ones against (ValueError)."""
        func = "predict"
        if self.medoid_points is None:
            raise ValueError(f"{func}(): a fit on precomputed distances cannot place new points")
        from .. import spatial as _spatial

        Xn = points(X_new, func, "X_new")
        p = self.medoid_points.shape[1]
        if Xn.shape[1] != p:
            raise ValueError(f"{func}(): 'X_new' must have {p} columns (the fit's), not {Xn.shape[1]}")
        if not bool(_sd.all(_sd.isfinite(Xn))):
            raise ValueError(f"{func}(): 'X_new' must be finite (no NaN or infinity)")
        M = contiguous(self.medoid_points[self._by_key])
        lab = _sd.argmin(_spatial.cdist(Xn, M, metric=self.metric), axis=1)
        return contiguous(self._by_key[lab])


_calls.methods(KMedoidsResult)


@function
def kmedoids(
    X,
    k,
    /,
    *,
    method="pam",
    metric="euclidean",
    rng=None,
    max_iter=1000,
    samples=5,
    sample_size=None,
    nan_policy="raise",
):
    """k-medoids clustering: k data points (medoids) that make the mean distance of every point to its nearest
    medoid small, by R cluster's PAM (method='pam': BUILD, then the best swap of a medoid and a point while one
    lowers the objective, at most max_iter swaps) or CLARA (method='clara': PAM on `samples` random samples of
    sample_size points, default min(n, 40 + 2k), each holding the best medoids so far; the best over all points
    kept; rng required).

    X holds observations ((n, p)) with metric one of sd.spatial's names, or distances with metric='precomputed' (an
    (n, n) matrix or a condensed vector). Ties go to the point of smaller key (its place in the points'
    lexicographic order), so the same points in another order give the same medoids. nan_policy 'raise' (the
    default) or 'omit' (observations only).
    """
    func = "kmedoids"
    method = choice(method, func, "method", METHODS)
    metric = _metric(metric, func)
    if method == "pam":
        no_rng(rng, func, "method='pam'")
    else:
        rng = check_rng(rng, func)
    max_iter = integer(max_iter, func, "max_iter")
    if max_iter < 0:
        raise ValueError(f"{func}(): 'max_iter' must be >= 0, not {max_iter}")
    samples = positive_int(samples, func, "samples")
    omitted = None
    if metric == "precomputed":
        D, n = distances(X, func)
        n_input = n
        order = _sd.arange(n, dtype=_sd.int64)
        Xk = None
        Xpts = None
        code = 0
    else:
        Xpts = points(X, func)
        n_input = Xpts.shape[0]
        Xpts, _, omitted = _rows(Xpts, func, nan_policy)
        n = Xpts.shape[0]
        order = _core.lex_order(Xpts)
        Xr = contiguous(Xpts[order])
        if metric in _ON_THE_FLY:
            Xk, D, code = Xr, None, _ON_THE_FLY[metric]
        else:
            from .. import spatial as _spatial

            check_alloc(8 * (n * (n - 1) // 2), func)
            Xk, D, code = None, contiguous(_spatial.pdist(Xr, metric=metric)), 0
    k = positive_int(k, func, "k")
    if k > n:
        raise ValueError(f"{func}(): 'k' ({k}) is larger than the number of points ({n})")
    if method == "clara":
        if k >= n:
            raise ValueError(f"{func}(): 'k' ({k}) must be smaller than the number of points ({n}) with method='clara'")
        size = min(n, 40 + 2 * k) if sample_size is None else positive_int(sample_size, func, "sample_size")
        if not k < size <= n:
            raise ValueError(f"{func}(): 'sample_size' must be between k + 1 ({k + 1}) and n ({n}), not {size}")
        k0, k1 = key(rng)
    else:
        if sample_size is not None:
            positive_int(sample_size, func, "sample_size")
        size, k0, k1 = n, 0, 0
    lab_key, med_key, obj, build, swaps, converged = _core.kmedoids(
        Xk, D, k, 0 if method == "pam" else 1, code, max_iter, samples, size, k0, k1
    )
    labels, number = _core.first_seen(lab_key, order, k)
    perm = _sd.empty((k,), dtype=_sd.int64)
    perm[number] = _sd.arange(k, dtype=_sd.int64)  # perm[new] = old slot
    keys = contiguous(med_key[perm])  # each label's medoid key
    rows = contiguous(order[keys])  # its row among the kept rows
    if omitted is not None:
        kept = _sd.nonzero(_sd.logical_not(omitted))[0]
        medoids = contiguous(kept[rows])
    else:
        medoids = rows
    by_key = _sd.argsort(keys)  # the labels in their medoids' key order (predict's tie rule)
    om = omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)
    return KMedoidsResult(
        labels=labels_back(labels, omitted),
        n_clusters=k,
        nobs=n,
        omitted=om,
        medoids=medoids,
        medoid_points=contiguous(Xpts[rows]) if Xpts is not None else None,
        objective=float(obj),
        build_objective=float(build),
        n_swaps=int(swaps),
        converged=bool(converged),
        method=method,
        metric=metric,
        _by_key=contiguous(by_key.astype(_sd.int64)),
    )
