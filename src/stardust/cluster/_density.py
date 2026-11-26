"""Density clustering (docs/api/cluster.md §4; docs/design/cluster.md §2.4, §3.6, §4.4; decisions CL28-CL30 and
those of milestone CL3 in §1.1): DBSCAN, HDBSCAN and OPTICS, in the meaning and with the defaults of scikit-learn
1.9.1.

Every point is numbered by its key (CL7: its rank in the points' lexicographic order; for precomputed distances the
input index) and every tie goes to the smaller key, so the same points in another order give the same partition
(scikit-learn's DBSCAN border points, HDBSCAN trees at equal mutual reachabilities and OPTICS orders at equal
reachabilities follow the input order instead). The distances are sd.spatial's (the bits of cdist and pdist):
'euclidean', 'cityblock' and 'chebyshev' observations through a KD tree or a scan of every pair (the same bits,
chosen by shape), any other metric through sd.spatial.pdist's condensed matrix, and metric='precomputed' distances
as given (csrc/cluster/density.cpp).
"""

import collections
import math

from .. import _calls
from .. import _core as _sd
from .._calls import function
from ._common import (
    check_alloc,
    choice,
    contiguous,
    distances,
    integer,
    labels_back,
    points,
    policy,
    positive_int,
    real,
    type_name,
)
from ._common import metric as _metric
from ._common import rows as _rows
from ._common import weights as _weights
from ._results import ClusterResult

_core = _sd.cluster
_ON_THE_FLY = {"euclidean": 0, "cityblock": 1, "chebyshev": 2}
_SELECTION = ("eom", "leaf")
_EXTRACTION = ("xi", "dbscan")

CondensedTree = collections.namedtuple("CondensedTree", ["parent", "child", "lambda_val", "child_size"])
CondensedTree.__doc__ = """HDBSCANResult.condensed_tree: one row per edge of the condensed tree. parent is a cluster
(the root n, the others n + 1, ...), child a cluster or a point (0 .. n - 1, the fitted rows' numbers), lambda_val
1 / the height where the child leaves the parent (inf at height 0) and child_size the child's points."""


def _bool(v, func, arg):
    if not isinstance(v, bool):
        raise TypeError(f"{func}(): '{arg}' must be a bool, not {type_name(v)}")
    return v


def _data(X, func, metric, nan_policy, w=None):
    """The data of a density method: (Xk key-ordered observations or None, D key-ordered condensed distances or
    None, metric code (0, 1, 2 for the on-the-fly kernels, -1 for a matrix), order (key -> fitted row), the fitted
    rows' observations (input order) or None, weights in key order or None, omitted mask or None, n_input)."""
    metric = _metric(metric, func)
    nan_policy = policy(nan_policy, func)
    if metric == "precomputed":
        D, n = distances(X, func)
        wk = None
        if w is not None:
            wk = _weights(w, func, n)
            if bool(_sd.any(_sd.isnan(wk))):
                raise ValueError(f"{func}(): 'weights' contains NaN")
            if not bool(_sd.any(wk > 0)):
                raise ValueError(f"{func}(): at least one weight must be positive")
        return None, D, -1, _sd.arange(n, dtype=_sd.int64), None, wk, None, n
    Xp = points(X, func)
    n_input = Xp.shape[0]
    w = _weights(w, func, n_input)
    Xp, w, omitted = _rows(Xp, func, nan_policy, w)
    n = Xp.shape[0]
    order = _core.lex_order(Xp)
    Xr = contiguous(Xp[order])
    wk = contiguous(w[order]) if w is not None else None
    if metric in _ON_THE_FLY:
        return Xr, None, _ON_THE_FLY[metric], order, Xp, wk, omitted, n_input
    from .. import spatial as _spatial

    check_alloc(8 * (n * (n - 1) // 2), func)
    D = contiguous(_spatial.pdist(Xr, metric=metric)) if n > 1 else _sd.empty((0,), dtype=_sd.float64)
    return None, D, -1, order, Xp, wk, omitted, n_input


def _by_input(v_key, order, n):
    """A per-key array at the fitted rows' places ((n,), v[order[r]] = v_key[r])."""
    out = _sd.empty((n,), dtype=v_key.dtype)
    out[order] = v_key
    return out


def _back(v, omitted, fill):
    """A per-fitted-row array at the input's places, `fill` at the rows nan_policy='omit' left out."""
    if omitted is None:
        return v
    out = _sd.full((omitted.shape[0],), fill, dtype=v.dtype)
    out[_sd.logical_not(omitted)] = v
    return out


def _omitted(omitted, n_input):
    return omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)


def _fraction_or_int(v, func, arg, n, low):
    """scikit-learn's OPTICS sizes: an int >= low (at most n), or a float in (0, 1] taken as a fraction of the n
    points (at least 2)."""
    kind = type(v).__name__
    numpy = type(v).__module__ == "numpy"
    if isinstance(v, float) or (numpy and kind.startswith("float")):
        f = float(v)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"{func}(): a float '{arg}' must be in (0, 1] (a fraction of the points), not {f!r}")
        return max(2, int(f * n))
    if isinstance(v, bool) or (numpy and kind == "bool") or not hasattr(v, "__index__"):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float in (0, 1], not {type_name(v)}")
    k = integer(v, func, arg)
    if k < low:
        raise ValueError(f"{func}(): '{arg}' must be >= {low}, not {k}")
    if k > n:
        raise ValueError(f"{func}(): '{arg}' ({k}) must be at most the number of points ({n})")
    return k


# ---- DBSCAN ----------------------------------------------------------------------------------------------------------


class DBSCANResult(ClusterResult):
    """The result of dbscan (docs/api/cluster.md §4).

    Fields: labels ((n,) int64, first-seen numbering; -1 for noise and for the rows nan_policy='omit' left out),
    n_clusters, nobs, omitted, core_mask ((n,) bool: the core points), eps, min_samples and metric.
    """

    _TITLE = "DBSCAN Results"
    _METHOD = "DBSCAN"
    _TUNING = (("Eps:", "eps"), ("Min samples:", "min_samples"), ("Metric:", "metric"))
    _REPR = ("nobs", "n_clusters", "eps", "min_samples")


_calls.methods(DBSCANResult)


@function
def dbscan(X, /, *, eps=0.5, min_samples=5, metric="euclidean", weights=None, nan_policy="raise"):
    """DBSCAN (Ester et al. 1996; scikit-learn's DBSCAN): a point is core when the points within eps of it (d <= eps,
    itself counted; with weights, their sum) number at least min_samples; the clusters are the core points joined
    by pairs within eps; a border point (not core, within eps of a core point) joins the cluster of its nearest core
    point (the smaller key on a tie; scikit-learn: the first cluster to reach it), every other point is noise (-1).

    X holds observations ((n, p)) with metric one of sd.spatial's names, or distances with metric='precomputed'.
    weights (one per point, >= 0) count in the core test only. nan_policy 'raise' (the default) or 'omit'.
    """
    func = "dbscan"
    eps = real(eps, func, "eps")
    if not (eps > 0 and math.isfinite(eps)):
        raise ValueError(f"{func}(): 'eps' must be finite and > 0, not {eps!r}")
    min_samples = positive_int(min_samples, func, "min_samples")
    Xk, D, code, order, _, wk, omitted, n_input = _data(X, func, metric, nan_policy, weights)
    n = order.shape[0]
    lab_key, core_key = _core.dbscan(Xk, D, eps, float(min_samples), wk, code, 0)
    k = int(_sd.max(lab_key)) + 1 if n else 0
    labels, _ = _core.first_seen(lab_key, order, max(k, 0))
    core = _by_input(core_key, order, n)
    return DBSCANResult(
        labels=labels_back(labels, omitted),
        n_clusters=k,
        nobs=n,
        omitted=_omitted(omitted, n_input),
        core_mask=_back(core, omitted, False),
        eps=eps,
        min_samples=min_samples,
        metric=_metric(metric, func),
    )


# ---- HDBSCAN ---------------------------------------------------------------------------------------------------------


class HDBSCANResult(ClusterResult):
    """The result of hdbscan (docs/api/cluster.md §4).

    Fields: labels ((n,) int64, first-seen numbering; -1 for noise and the rows nan_policy='omit' left out),
    n_clusters, nobs, omitted, probabilities ((n,): the strength of each point's membership, 0 for noise; NaN for
    the omitted rows), outlier_scores ((n,): GLOSH, 1 - lambda / lambda_max of the cluster the point leaves),
    single_linkage (the single linkage tree of the mutual reachability: scipy's Z over the fitted rows),
    condensed_tree (a CondensedTree namedtuple), persistence ((n_clusters,)), centroids and medoids ((n_clusters,
    p): the probability-weighted mean and the point of smallest probability-weighted distance sum, computed when
    first read; None for precomputed distances), min_cluster_size, min_samples, cluster_selection_method and
    metric.
    """

    _TITLE = "HDBSCAN Results"
    _METHOD = "HDBSCAN ({cluster_selection_method})"
    _TUNING = (
        ("Min cluster size:", "min_cluster_size"),
        ("Min samples:", "min_samples"),
        ("Metric:", "metric"),
    )
    _COLUMNS = (("Persistence", "persistence"),)
    _REPR = ("nobs", "n_clusters", "min_cluster_size", "min_samples")

    def dbscan_clustering(self, cut_distance, /, *, min_cluster_size=5):
        """The flat clustering of the single linkage tree cut at cut_distance (scikit-learn's dbscan_clustering,
        DBSCAN* at eps = cut_distance): the merges below cut_distance joined, a component of fewer than
        min_cluster_size points noise (-1). (n,) int64, first-seen numbering."""
        func = "dbscan_clustering"
        cut = real(cut_distance, func, "cut_distance")
        if math.isnan(cut):
            raise ValueError(f"{func}(): 'cut_distance' must not be NaN")
        mcs = positive_int(min_cluster_size, func, "min_cluster_size")
        lab = _core.hdbscan_cut(self.single_linkage, cut, mcs)
        return labels_back(lab, self.omitted if bool(_sd.any(self.omitted)) else None)


_calls.methods(HDBSCANResult)


@function
def hdbscan(
    X,
    /,
    *,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    max_cluster_size=None,
    metric="euclidean",
    alpha=1.0,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    nan_policy="raise",
):
    """HDBSCAN (Campello et al. 2013; scikit-learn's HDBSCAN): the core distance of a point is the distance to its
    min_samples-th nearest point (itself counted; min_samples None is min_cluster_size), the mutual reachability of
    two points max(core_a, core_b, d / alpha); the single linkage tree of its minimum spanning tree is condensed
    with min_cluster_size (the merges of one height taken together) and the clusters are chosen by stability:
    'eom' (excess of mass) or 'leaf', with cluster_selection_epsilon, max_cluster_size and allow_single_cluster as
    in scikit-learn. Noise is -1.

    X holds observations ((n, p)) with metric one of sd.spatial's names, or distances with metric='precomputed'.
    nan_policy 'raise' (the default) or 'omit'.
    """
    func = "hdbscan"
    mcs = integer(min_cluster_size, func, "min_cluster_size")
    if mcs < 2:
        raise ValueError(f"{func}(): 'min_cluster_size' must be >= 2, not {mcs}")
    ms = mcs if min_samples is None else positive_int(min_samples, func, "min_samples")
    cse = real(cluster_selection_epsilon, func, "cluster_selection_epsilon")
    if not (cse >= 0 and math.isfinite(cse)):
        raise ValueError(f"{func}(): 'cluster_selection_epsilon' must be finite and >= 0, not {cse!r}")
    max_size = -1 if max_cluster_size is None else positive_int(max_cluster_size, func, "max_cluster_size")
    alpha = real(alpha, func, "alpha")
    if not (alpha > 0 and math.isfinite(alpha)):
        raise ValueError(f"{func}(): 'alpha' must be finite and > 0, not {alpha!r}")
    method = choice(cluster_selection_method, func, "cluster_selection_method", _SELECTION)
    single = _bool(allow_single_cluster, func, "allow_single_cluster")
    Xk, D, code, order, Xp, _, omitted, n_input = _data(X, func, metric, nan_policy)
    n = order.shape[0]
    if n < 2:
        raise ValueError(f"{func}(): 'X' must have at least 2 points")
    if ms > n:
        raise ValueError(f"{func}(): 'min_samples' ({ms}) must be at most the number of points ({n})")
    lab_key, prob_key, out_key, Z, cond, stability, selected, persistence = _core.hdbscan(
        Xk, D, order, ms, alpha, code, 0, mcs, _SELECTION.index(method), single, cse, max_size
    )
    k = int(selected.shape[0])
    labels, number = _core.first_seen(lab_key, order, k)
    pers = _sd.empty((k,), dtype=_sd.float64)
    pers[number] = persistence
    metric_name = _metric(metric, func)
    lazy = {}
    if Xp is not None:
        lazy = _centres(Xk if Xk is not None else contiguous(Xp[order]), lab_key, prob_key, k, number, metric_name)
    return HDBSCANResult(
        labels=labels_back(labels, omitted),
        n_clusters=k,
        nobs=n,
        omitted=_omitted(omitted, n_input),
        probabilities=_back(_by_input(prob_key, order, n), omitted, math.nan),
        outlier_scores=_back(_by_input(out_key, order, n), omitted, math.nan),
        single_linkage=Z,
        condensed_tree=CondensedTree(*cond),
        persistence=pers,
        centroids=lazy.get("centroids"),
        medoids=lazy.get("medoids"),
        min_cluster_size=mcs,
        min_samples=ms,
        cluster_selection_method=method,
        metric=metric_name,
        _stability=stability,
    )


def _centres(Xr, lab_key, prob_key, k, number, metric):
    """The lazy centroids and medoids (scikit-learn's store_centers): key-ordered data, per-key labels (slots) and
    probabilities; the slots' new numbers `number`."""
    cache = {}

    def compute():
        if "c" not in cache:
            code = _ON_THE_FLY.get(metric, 3)  # 3: the centroids alone
            C, med = _core.hdbscan_centres(Xr, lab_key, prob_key, k, code)
            if metric not in _ON_THE_FLY:  # the medoids through sd.spatial.pdist, cluster by cluster
                from .. import spatial as _spatial

                med = _sd.empty((k,), dtype=_sd.int64)
                for j in range(k):
                    members = contiguous(_sd.nonzero(lab_key == j)[0])
                    pts = contiguous(Xr[members])
                    check_alloc(8 * (len(members) * (len(members) - 1) // 2), "medoids")
                    Dj = contiguous(_spatial.pdist(pts, metric=metric)) if len(members) > 1 else _sd.empty((0,))
                    med[j] = int(members[_core.weighted_medoid(Dj, contiguous(prob_key[members]))])
            cache["c"] = (C, med)
        return cache["c"]

    def centroids():
        C, _ = compute()
        out = _sd.empty(tuple(C.shape), dtype=_sd.float64)
        out[number] = C
        return out

    def medoids():
        _, med = compute()
        M = contiguous(Xr[med]) if k else _sd.empty((0, Xr.shape[1]), dtype=_sd.float64)
        out = _sd.empty(tuple(M.shape), dtype=_sd.float64)
        out[number] = M
        return out

    return {"centroids": centroids, "medoids": medoids}


# ---- OPTICS ----------------------------------------------------------------------------------------------------------


class OPTICSResult(ClusterResult):
    """The result of optics and of its extract_xi and extract_dbscan (docs/api/cluster.md §4).

    Fields: labels ((n,) int64, first-seen numbering; -1 for noise and the rows nan_policy='omit' left out; the
    clusters of cluster_method), n_clusters, nobs, omitted, ordering ((nobs,) the input rows in the cluster order),
    reachability, core_distances ((n,) per input row; inf where undefined, NaN for the omitted rows), predecessor
    ((n,) the input row each point was reached from, -1 for the starts and the omitted rows), cluster_hierarchy
    ((m, 2) int64: the xi clusters as [start, end] positions in the ordering, smaller nested clusters first; None
    for cluster_method='dbscan'), min_samples, max_eps, cluster_method, eps, xi and metric.
    """

    _TITLE = "OPTICS Results"
    _METHOD = "OPTICS ({cluster_method})"
    _TUNING = (("Min samples:", "min_samples"), ("Max eps:", "max_eps"), ("Metric:", "metric"))
    _REPR = ("nobs", "n_clusters", "cluster_method", "min_samples")

    def extract_xi(self, *, xi=0.05, min_samples=None, min_cluster_size=None, predecessor_correction=True):
        """The clusters of the same ordering by the xi-steep method (scikit-learn's cluster_optics_xi): a new
        OPTICSResult with cluster_method='xi'. min_samples None is the fit's; min_cluster_size None is
        min_samples."""
        func = "extract_xi"
        g = self._graph
        n = int(g[0].shape[0])
        ms = self.min_samples if min_samples is None else _fraction_or_int(min_samples, func, "min_samples", n, 2)
        return _extract(self, "xi", func, xi=xi, ms=ms, mcs=min_cluster_size, correction=predecessor_correction)

    def extract_dbscan(self, eps, /):
        """The clusters of the same ordering at eps (scikit-learn's cluster_optics_dbscan; eps <= max_eps): a new
        OPTICSResult with cluster_method='dbscan'."""
        return _extract(self, "dbscan", "extract_dbscan", eps=eps)


_calls.methods(OPTICSResult)


def _xi_value(xi, func):
    xi = real(xi, func, "xi")
    if not 0.0 <= xi <= 1.0:
        raise ValueError(f"{func}(): 'xi' must be in [0, 1], not {xi!r}")
    return xi


def _eps_value(eps, func, max_eps):
    eps = real(eps, func, "eps")
    if not eps >= 0:
        raise ValueError(f"{func}(): 'eps' must be >= 0, not {eps!r}")
    if eps > max_eps:
        raise ValueError(f"{func}(): 'eps' ({eps!r}) must be at most 'max_eps' ({max_eps!r})")
    return eps


def _labels_of(graph, method, *, xi=None, ms=None, mcs=None, correction=True, eps=None):
    """(labels of the fitted rows (first-seen), K, cluster_hierarchy or None) of one extraction."""
    ordering, reach, core, pred = graph
    n = int(ordering.shape[0])
    if method == "xi":
        lab, clusters = _core.optics_xi(
            contiguous(reach[ordering]), contiguous(pred[ordering]), ordering, ms, mcs, xi, correction
        )
    else:
        lab, clusters = _core.optics_dbscan(reach, core, ordering, eps), None
    k = int(_sd.max(lab)) + 1 if n else 0
    labels, _ = _core.first_seen(lab, _sd.arange(n, dtype=_sd.int64), max(k, 0))
    return labels, k, clusters


def _result(graph, omitted, n_input, labels, k, clusters, fields):
    ordering, reach, core, pred = graph
    n = int(ordering.shape[0])
    if omitted is not None:
        kept = contiguous(_sd.nonzero(_sd.logical_not(omitted))[0])
        to_input = lambda v: contiguous(kept[v])  # noqa: E731
        pred_in = _sd.where(pred >= 0, kept[_sd.maximum(pred, 0)], -1)
    else:
        to_input = lambda v: v  # noqa: E731
        pred_in = pred
    return OPTICSResult(
        labels=labels_back(labels, omitted),
        n_clusters=k,
        nobs=n,
        omitted=_omitted(omitted, n_input),
        ordering=to_input(ordering),
        reachability=_back(reach, omitted, math.nan),
        core_distances=_back(core, omitted, math.nan),
        predecessor=_back(contiguous(pred_in.astype(_sd.int64)), omitted, -1),
        cluster_hierarchy=clusters,
        _graph=graph,
        **fields,
    )


def _extract(res, method, func, *, xi=0.05, ms=None, mcs=None, correction=True, eps=None):
    g = res._graph
    n = int(g[0].shape[0])
    fields = {k: getattr(res, k) for k in ("min_samples", "max_eps", "metric")}
    om = res.omitted if bool(_sd.any(res.omitted)) else None
    if method == "xi":
        xi = _xi_value(xi, func)
        mcs = ms if mcs is None else _fraction_or_int(mcs, func, "min_cluster_size", n, 2)
        correction = _bool(correction, func, "predecessor_correction")
        labels, k, clusters = _labels_of(g, "xi", xi=xi, ms=ms, mcs=mcs, correction=correction)
        fields.update(cluster_method="xi", eps=None, xi=xi)
    else:
        eps = _eps_value(eps, func, res.max_eps)
        labels, k, clusters = _labels_of(g, "dbscan", eps=eps)
        fields.update(cluster_method="dbscan", eps=eps, xi=None)
    return _result(g, om, res.omitted.shape[0], labels, k, clusters, fields)


@function
def optics(
    X,
    /,
    *,
    min_samples=5,
    max_eps=math.inf,
    metric="euclidean",
    cluster_method="xi",
    eps=None,
    xi=0.05,
    predecessor_correction=True,
    min_cluster_size=None,
    nan_policy="raise",
):
    """OPTICS (Ankerst et al. 1999; scikit-learn's OPTICS): the points in the order that always takes next the
    unprocessed point of smallest reachability (the smaller key on a tie), each core point (core distance, the
    distance to its min_samples-th nearest point with itself counted, at most max_eps) lowering the reachability
    of the points within max_eps to max(core distance, d). The clusters are extracted by cluster_method 'xi'
    (the xi-steep method with xi, predecessor_correction and min_cluster_size; scikit-learn's
    cluster_optics_xi) or 'dbscan' (at eps, default max_eps; cluster_optics_dbscan). min_samples and
    min_cluster_size are an int (>= 2) or a float in (0, 1], a fraction of the points (at least 2).

    X holds observations ((n, p)) with metric one of sd.spatial's names, or distances with metric='precomputed'.
    nan_policy 'raise' (the default) or 'omit'.
    """
    func = "optics"
    max_eps = real(max_eps, func, "max_eps")
    if not max_eps >= 0:
        raise ValueError(f"{func}(): 'max_eps' must be >= 0, not {max_eps!r}")
    method = choice(cluster_method, func, "cluster_method", _EXTRACTION)
    xi = _xi_value(xi, func)
    correction = _bool(predecessor_correction, func, "predecessor_correction")
    if eps is not None:
        eps = real(eps, func, "eps")
        if not eps >= 0:
            raise ValueError(f"{func}(): 'eps' must be >= 0, not {eps!r}")
    Xk, D, code, order, _, _, omitted, n_input = _data(X, func, metric, nan_policy)
    n = order.shape[0]
    ms = _fraction_or_int(min_samples, func, "min_samples", n, 2)
    mcs = ms if min_cluster_size is None else _fraction_or_int(min_cluster_size, func, "min_cluster_size", n, 2)
    ord_k, reach_k, core_k, pred_k = _core.optics(Xk, D, ms, max_eps, code, 0)
    pred_rows = _sd.where(pred_k >= 0, order[_sd.maximum(pred_k, 0)], -1).astype(_sd.int64)
    graph = (
        contiguous(order[ord_k]),
        _by_input(reach_k, order, n),
        _by_input(core_k, order, n),
        _by_input(contiguous(pred_rows), order, n),
    )
    fields = {"min_samples": ms, "max_eps": max_eps, "metric": _metric(metric, func)}
    if method == "xi":
        labels, k, clusters = _labels_of(graph, "xi", xi=xi, ms=ms, mcs=mcs, correction=correction)
        fields.update(cluster_method="xi", eps=None, xi=xi)
    else:
        e = max_eps if eps is None else eps
        if e > max_eps:
            raise ValueError(f"{func}(): 'eps' ({e!r}) must be at most 'max_eps' ({max_eps!r})")
        labels, k, clusters = _labels_of(graph, "dbscan", eps=e)
        fields.update(cluster_method="dbscan", eps=e, xi=None)
    return _result(graph, omitted, n_input, labels, k, clusters, fields)
