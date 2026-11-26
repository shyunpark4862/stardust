"""k-means and its kin (docs/api/cluster.md §3; docs/design/cluster.md §2.3, §3.1, §3.3, §4.2; decisions CL5-CL11,
CL23-CL25, CL41-CL43, CL48, CL49 and those of milestone CL2 in §1.1).

kmeans is Lloyd's iteration on the squared Euclidean distance in the difference form sum (x - c)^2 (sd.spatial's
sqeuclidean of each pair; scikit-learn's expanded form loses digits far from the origin), with scikit-learn's
starts (greedy k-means++, 'random', an array), stop (no label changed, or the centres moved by at most tol times
the mean of the columns' variances), relocation of empty clusters and default values. The kernels
(csrc/cluster/kmeans.cpp) take the rows in key order (CL7: the lexicographic order of the points), so the same
points in another order give the same partition, and every path (Lloyd, Hamerly's bounds, the expanded form's
proven filter) gives the same bits on any number of threads. The random numbers come from two 64-bit words of rng
(CL41). Labels are numbered by first appearance (CL6) and the centres follow them.

minibatch_kmeans is scikit-learn's MiniBatchKMeans, bisecting_kmeans its BisectingKMeans; both return the same kind
of result (bisecting's with the tree of its splits).
"""

import math
import numbers

from .. import _calls
from .. import _core as _sd
from .._calls import function
from ._common import (
    check_rng,
    choice,
    contiguous,
    integer,
    key,
    labels_back,
    new_points,
    points,
    positive_int,
    real,
    type_name,
    weights,
)
from ._common import (
    rows as _rows,
)
from ._results import ClusterResult

_core = _sd.cluster
INITS = ("k-means++", "random")
STRATEGIES = ("biggest_inertia", "largest_cluster")


# ---- arguments ---------------------------------------------------------------------------------------------------


def _k(v, func, n):
    k = positive_int(v, func, "k")
    if k > n:
        raise ValueError(f"{func}(): 'k' ({k}) is larger than the number of points ({n})")
    return k


def _init(v, func, k, p, names=INITS):
    """('k-means++' | 'random', None) or ('array', the (k, p) float64 centres); names: the init names allowed."""
    if isinstance(v, str):
        return choice(v, func, "init", names), None
    if v is None or isinstance(v, (bool, numbers.Number)):
        listed = ", ".join(f"'{c}'" for c in names)
        raise TypeError(f"{func}(): 'init' must be {listed} or a (k, p) array, not {type_name(v)}")
    from ._common import _array

    a = _array(v, func, "init")
    if a.ndim == 1 and p == 1:
        a = _sd.reshape(a, (a.shape[0], 1))
    if a.ndim != 2 or tuple(a.shape) != (k, p):
        raise ValueError(f"{func}(): an array 'init' must have shape (k, p) = ({k}, {p}), not {tuple(a.shape)}")
    if bool(_sd.any(_sd.isnan(a))):
        raise ValueError(f"{func}(): 'init' contains NaN")
    if bool(_sd.any(_sd.isinf(a))):
        raise ValueError(f"{func}(): 'init' contains infinity")
    return "array", contiguous(a)


def _n_init(v, func, init, auto):
    if isinstance(v, str):
        if v != "auto":
            raise ValueError(f"{func}(): 'n_init' must be 'auto' or an int >= 1, not {v!r}")
        return 1 if init != "random" else auto
    n = positive_int(v, func, "n_init")
    if init == "array" and n != 1:
        raise ValueError(f"{func}(): 'n_init' must be 1 with an array 'init' (one start), not {n}")
    return n


def _tol(v, func):
    t = real(v, func, "tol")
    if not (t >= 0 and math.isfinite(t)):
        raise ValueError(f"{func}(): 'tol' must be finite and >= 0, not {t!r}")
    return t


def _distinct_check(X, order, w, k, func):
    d = _core.kmeans_distinct(X, order, w)
    if k > d:
        what = "distinct points of positive weight" if w is not None else "distinct points"
        raise ValueError(f"{func}(): 'k' ({k}) is larger than the number of {what} ({d})")


def _prepare(X, k, func, weights_, nan_policy, init):
    """(X, w, omitted, n_input, order, k, init kind, init array) after the common checks."""
    X = points(X, func)
    n_input = X.shape[0]
    w = weights(weights_, func, n_input)
    X, w, omitted = _rows(X, func, nan_policy, w)
    n, p = X.shape
    k = _k(k, func, n)
    kind, C0 = _init(init, func, k, p)
    order = _core.lex_order(X)
    _distinct_check(X, order, w, k, func)
    return X, w, omitted, n_input, order, k, kind, C0


_INIT_CODE = {"k-means++": 0, "random": 1, "array": 2}


# ---- the results -------------------------------------------------------------------------------------------------


class KMeansResult(ClusterResult):
    """The result of kmeans and minibatch_kmeans (docs/api/cluster.md §3): the partition and its centres.

    Fields: labels ((n,) int64, first-seen numbering; -1 for the rows nan_policy='omit' left out), n_clusters (k),
    nobs, omitted, centers ((k, p): centers[j] is the centre of label j), inertia (= objective: the weighted sum
    of squared distances to the own centre), cluster_inertia ((k,): each cluster's share), n_iter, converged,
    n_init (the starts run) and best_init (the start kept). minibatch_kmeans' results also have n_steps.
    """

    _TITLE = "K-Means Results"
    _METHOD = "{method}"
    _OBJECTIVE = ("Inertia:", "inertia")
    _TUNING = (("Init:", "init"), ("No. Starts:", "n_init"), ("Best start:", "best_init"))
    _COLUMNS = (("Inertia", "cluster_inertia"),)
    _REPR = ("nobs", "n_clusters", "inertia", "n_iter")

    def predict(self, X_new, /):
        """The nearest centre of each new point ((m,) int64): the smallest squared distance, and on a tie the
        centre the fit made first, so the fit's own points get the fit's labels."""
        func = "predict"
        Xn = self._rows_of(X_new, func)
        lab, _ = _core.kmeans_predict(Xn, self._fit_centers, self._fit_scaled, self._fit_scale, 0)
        return _sd.take(self._fit_numbers, lab)

    def transform(self, X_new, /):
        """The Euclidean distance of each new point to each centre ((m, k), sd.spatial.cdist)."""
        from .. import spatial as _spatial

        return _spatial.cdist(self._rows_of(X_new, "transform"), self.centers)

    def _rows_of(self, X_new, func):
        Xn = new_points(X_new, func, self.centers.shape[1])
        return Xn


_calls.methods(KMeansResult)


def _result(cls, out, order, k, omitted, n_input, method, init, n_init, **extra):
    """The result from the core's ((labels in key order, centres, scaled centres, e, inertia, per-cluster inertia),
    n_iter, converged, best_init, ...): labels to the input's order and first-seen numbers (CL6)."""
    (lab_key, C, Cs, e, inertia, per), n_iter, converged, best_init = out[:4]
    labels, number = _core.first_seen(lab_key, order, k)
    perm = _sd.empty((k,), dtype=_sd.int64)
    perm[number] = _sd.arange(k, dtype=_sd.int64)  # perm[new] = old
    centers = contiguous(C[perm])
    om = omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)
    return cls(
        labels=labels_back(labels, omitted),
        n_clusters=k,
        nobs=int(order.shape[0]),
        omitted=om,
        centers=centers,
        inertia=float(inertia),
        objective=float(inertia),
        cluster_inertia=contiguous(per[perm]),
        n_iter=int(n_iter),
        converged=bool(converged),
        n_init=int(n_init),
        best_init=best_init if isinstance(best_init, _sd.ndarray) else int(best_init),
        method=method,
        init=init,
        _fit_centers=C,
        _fit_scaled=Cs,
        _fit_scale=int(e),
        _fit_numbers=number,
        **extra,
    )


# ---- kmeans ----------------------------------------------------------------------------------------------------


@function
def kmeans(
    X, k, /, *, rng, init="k-means++", n_init="auto", max_iter=300, tol=0.0001, weights=None, nan_policy="raise"
):
    """k-means clustering (Lloyd's iteration): k centres and the partition of the points by their nearest centre.

    X holds the points ((n, p), a 1-D X is (n, 1)); k the number of clusters (1 .. the number of distinct points).
    rng (an sd.random.Generator, required) draws the starts: two 64-bit words a call. init 'k-means++'
    (scikit-learn's greedy version), 'random' (k points) or a (k, p) array; n_init the starts ('auto': 1 for
    'k-means++' and an array, 10 for 'random'), the best being the smallest (inertia, start). A start iterates
    until no label changes or the centres move by at most tol times the mean of the columns' variances (their
    squared moves summed), at most max_iter times. weights: the points' weights (scikit-learn's sample_weight).
    nan_policy 'raise' (the default) or 'omit' (rows with a NaN left out, label -1).

    The squared distance is sum (x - c)^2 (scikit-learn's ||x||^2 - 2 x.c + ||c||^2 loses digits far from the
    origin); a tie goes to the smaller centre number; an empty cluster takes the point farthest from its centre.
    The same points in another order give the same partition, and any thread count the same bits.
    """
    func = "kmeans"
    X, w, omitted, n_input, order, k, kind, C0 = _prepare(X, k, func, weights, nan_policy, init)
    n_init = _n_init(n_init, func, kind, 10)
    max_iter = positive_int(max_iter, func, "max_iter")
    tol = _tol(tol, func)
    rng = check_rng(rng, func)
    k0, k1 = key(rng)
    out = _core.kmeans(X, order, w, k, _INIT_CODE[kind], C0, n_init, max_iter, tol, k0, k1, 0)
    return _result(KMeansResult, out, order, k, omitted, n_input, "k-means (Lloyd)", kind, n_init)


# ---- mini-batch k-means ---------------------------------------------------------------------------------------


@function
def minibatch_kmeans(
    X,
    k,
    /,
    *,
    rng,
    init="k-means++",
    n_init="auto",
    batch_size=1024,
    max_iter=100,
    max_no_improvement=10,
    tol=0.0,
    init_size=None,
    reassignment_ratio=0.01,
    weights=None,
    nan_policy="raise",
):
    """Mini-batch k-means (scikit-learn's MiniBatchKMeans): k centres moved by random batches of the points.

    Each start draws its centres from init_size random points ('auto' n_init: 1 for 'k-means++' and an array, 3
    for 'random'), the start of smallest inertia on init_size other random points is kept; then each step draws
    batch_size points (with probability proportional to the weights) and moves each centre to the mean of all the
    points it has taken so far. Centres that took fewer than reassignment_ratio times the most points move to
    random points of the batch now and then. The steps stop after max_iter passes over the data, when the centres
    move by at most tol times the mean of the columns' variances, or after max_no_improvement steps (None: never)
    without a smaller smoothed batch inertia. The result's labels and inertia are those of every point at the final
    centres; n_steps the steps taken, n_iter the passes (ceil(n_steps batch_size / n)).
    """
    func = "minibatch_kmeans"
    X, w, omitted, n_input, order, k, kind, C0 = _prepare(X, k, func, weights, nan_policy, init)
    n = X.shape[0]
    n_init = _n_init(n_init, func, kind, 3)
    batch = min(positive_int(batch_size, func, "batch_size"), n)
    max_iter = positive_int(max_iter, func, "max_iter")
    if max_no_improvement is None:
        mni = -1
    else:
        mni = integer(max_no_improvement, func, "max_no_improvement")
        if mni < 0:
            raise ValueError(f"{func}(): 'max_no_improvement' must be >= 0 or None, not {mni}")
    tol = _tol(tol, func)
    if init_size is None:
        isz = 3 * batch if 3 * batch >= k else 3 * k
    else:
        isz = positive_int(init_size, func, "init_size")
        if isz < k:
            raise ValueError(f"{func}(): 'init_size' ({isz}) must be at least 'k' ({k})")
    isz = min(isz, n)
    ratio = real(reassignment_ratio, func, "reassignment_ratio")
    if not (ratio >= 0 and math.isfinite(ratio)):
        raise ValueError(f"{func}(): 'reassignment_ratio' must be finite and >= 0, not {ratio!r}")
    rng = check_rng(rng, func)
    k0, k1 = key(rng)
    out = _core.minibatch_kmeans(
        X, order, w, k, _INIT_CODE[kind], C0, n_init, batch, max_iter, mni, tol, isz, ratio, k0, k1, 0
    )
    return _result(
        KMeansResult, out, order, k, omitted, n_input, "mini-batch k-means", kind, n_init, n_steps=int(out[4])
    )


# ---- bisecting k-means ---------------------------------------------------------------------------------------


class BisectingKMeansResult(KMeansResult):
    """The result of bisecting_kmeans (docs/api/cluster.md §3): KMeansResult's fields (n_iter the iterations of all
    the splits, converged whether all converged, best_init each split's start kept) and tree, the splits as a
    linkage matrix ((k - 1, 4): the clusters 0 .. k - 1 by label are its leaves, each row two clusters joined (a
    split undone) at the inertia of the cluster that was split, carried up so the heights never decrease, and the
    number of leaves below; dendrogram(tree) draws it).
    predict goes down the tree (the nearer of the two centres at each split), as scikit-learn's."""

    _TITLE = "Bisecting K-Means Results"
    _TUNING = (("Init:", "init"), ("No. Starts:", "n_init"), ("Strategy:", "strategy"))

    def predict(self, X_new, /):
        """The leaf of each new point ((m,) int64): from the root, the child whose centre is nearer at each split
        (the left on a tie), as scikit-learn's BisectingKMeans.predict."""
        Xn = self._rows_of(X_new, "predict")
        lab = _core.bisecting_predict(Xn, self._node_centers, self._children, self._leaf)
        return _sd.take(self._fit_numbers, lab)


_calls.methods(BisectingKMeansResult)


def _tree(children, heights, leaf, number, k):
    """The linkage matrix of the splits: rows in (height carried up, later split first) order."""
    ns = k - 1
    ch = children.tolist()
    h = heights.tolist()
    lf = leaf.tolist()
    num = number.tolist()
    split_of = {}
    for v, x in enumerate(lf):
        if x <= -2:
            split_of[v] = -2 - x
    up = list(h)
    for s in range(ns - 1, -1, -1):  # children split later
        for c in ch[s]:
            if c in split_of:
                up[s] = max(up[s], up[split_of[c]])
    rows = sorted(range(ns), key=lambda s: (up[s], -s))
    pos = {s: i for i, s in enumerate(rows)}

    def node_id(v):
        return k + pos[split_of[v]] if v in split_of else num[lf[v]]

    leaves = {}

    def count(v):  # the clusters (leaves) below node v
        if v not in split_of:
            return 1
        if v not in leaves:
            a, b = ch[split_of[v]]
            leaves[v] = count(a) + count(b)
        return leaves[v]

    Z = []
    for s in rows:
        a, b = sorted(node_id(c) for c in ch[s])
        Z.append([float(a), float(b), up[s], float(count(ch[s][0]) + count(ch[s][1]))])
    return _sd.asarray(Z, dtype=_sd.float64) if Z else _sd.empty((0, 4), dtype=_sd.float64)


@function
def bisecting_kmeans(
    X,
    k,
    /,
    *,
    rng,
    init="random",
    n_init=1,
    max_iter=300,
    tol=0.0001,
    strategy="biggest_inertia",
    weights=None,
    nan_policy="raise",
):
    """Bisecting k-means (scikit-learn's BisectingKMeans): k - 1 times the cluster of largest inertia
    (strategy='biggest_inertia') or of most points ('largest_cluster') is split in two by kmeans (init 'random' or
    'k-means++', n_init starts, max_iter, tol relative to the whole data's variances). The result's tree holds the
    splits; predict goes down it. A cluster of fewer than two distinct points is never split.
    """
    func = "bisecting_kmeans"
    if not isinstance(init, str):
        raise TypeError(f"{func}(): 'init' must be 'k-means++' or 'random', not {type_name(init)}")
    X, w, omitted, n_input, order, k, kind, C0 = _prepare(X, k, func, weights, nan_policy, init)
    n_init = positive_int(n_init, func, "n_init")
    max_iter = positive_int(max_iter, func, "max_iter")
    tol = _tol(tol, func)
    strategy = choice(strategy, func, "strategy", STRATEGIES)
    rng = check_rng(rng, func)
    k0, k1 = key(rng)
    out = _core.bisecting_kmeans(
        X, order, w, k, _INIT_CODE[kind], n_init, max_iter, tol, strategy == "largest_cluster", k0, k1, 0
    )
    head, n_iter, converged, best_init, node_c, children, heights, leaf, sizes = out
    res_number = _core.first_seen(head[0], order, k)[1]
    tree = _tree(children, heights, leaf, res_number, k)
    return _result(
        BisectingKMeansResult,
        (head, n_iter, converged, best_init),
        order,
        k,
        omitted,
        n_input,
        "bisecting k-means",
        kind,
        n_init,
        strategy=strategy,
        tree=tree,
        _node_centers=node_c,
        _children=children,
        _leaf=leaf,
    )
