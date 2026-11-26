"""Hierarchical clustering (docs/api/cluster.md §2; docs/design/cluster.md §2.2, §3.2, §4.1; decisions CL4, CL12,
CL15-CL22, CL43, CL45-CL47, CL50-CL57).

Two faces (CL4): scipy's array tools (linkage -> Z, fcluster, cut_tree, cophenet, leaves_list, dendrogram, ...)
and the result object of agglomerative and divisive (HierarchyResult: .linkage is the same Z, .cut(k=, height=)).

The tree is the primitive definition (CL17): n - 1 times, the pair of clusters of smallest (linkage value, key)
merges, the key the pair's two smallest point keys (CL7: the rank of the points' lexicographic order; for
precomputed distances the input index). So the same points in another order give the same tree. The algorithms
(csrc/cluster/hierarchy.cpp): the minimum spanning tree for single (Prim, or dual-tree Boruvka by shape),
the nearest-neighbour chain for complete, average, weighted and ward with the generic algorithm for the inputs
whose scans meet an exact tie (CL46), the generic algorithm for centroid and median; ward, centroid and median
of observations need no distance matrix (CL18). The heights are the Lance-Williams recurrence on the final tree
(CL45), the same bits on every path; Z's rows are in the primitive order of the final heights.

Z is scipy's (n - 1, 4) float64: row i merges clusters Z[i, 0] < Z[i, 1] (the points 0 .. n - 1, row i's
cluster n + i) at height Z[i, 2] into a cluster of Z[i, 3] points.
"""

import collections
import math
import numbers

from .. import _calls, _summary
from .. import _core as _sd
from .._calls import function
from ._common import (
    _array,
    check_alloc,
    choice,
    contiguous,
    distances,
    integer,
    labels_back,
    metric_columns,
    points,
    policy,
    real,
    type_name,
)
from ._common import (
    metric as _metric,
)
from ._common import (
    rows as _rows,
)
from ._results import ClusterResult

_core = _sd.cluster
METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")
_CODE = {m: i for i, m in enumerate(METHODS)}
_CENTROID = ("centroid", "median", "ward")
_MATRIX_FREE = ("single", "centroid", "median", "ward")
CRITERIA = ("inconsistent", "distance", "maxclust", "monocrit", "maxclust_monocrit")
TRUNCATE = ("lastp", "level")
ORIENTATIONS = ("top", "left", "bottom", "right")

DendrogramResult = collections.namedtuple(
    "DendrogramResult", ["icoord", "dcoord", "ivl", "leaves", "color_list", "leaves_color_list"]
)
DendrogramResult.__doc__ = """dendrogram()'s layout (scipy's dendrogram(no_plot=True) as a namedtuple): icoord and
dcoord ((m, 4) float64: the x and y of each link's four corners, leaves at x = 5, 15, 25, ...), ivl (the leaves'
labels), leaves ((k,) int64: the leaves' cluster numbers), color_list ((m,) int64: each link's colour group, -1
above color_threshold) and leaves_color_list ((k,) int64: each leaf's link's group)."""
CophenetResult = collections.namedtuple("CophenetResult", ["correlation", "distances"])
CophenetResult.__doc__ = """cophenet_correlation()'s result: correlation (the Pearson correlation of the cophenetic
distances and y) and distances (the condensed cophenetic distances)."""


# ---- inputs ---------------------------------------------------------------------------------------------------


def _method(v, func):
    return choice(v, func, "method", METHODS)


def _bool(v, func, arg):
    if not isinstance(v, bool):
        raise TypeError(f"{func}(): '{arg}' must be a bool, not {type_name(v)}")
    return v


def _check_squared(method, metric, func):
    if method in _CENTROID and metric not in ("euclidean", "precomputed"):
        raise ValueError(f"{func}(): method={method!r} requires metric='euclidean'")


def _pdist(X, metric, func):
    from .. import spatial as _spatial

    n = X.shape[0]
    metric_columns(metric, X.shape[1], func)
    check_alloc(16 * (n * (n - 1) // 2), func)  # the distances and the algorithm's working copy
    return contiguous(_spatial.pdist(X, metric=metric))


def _tree(X, D, order, method, path=0):
    """Z of observations X in key order (or None) and/or condensed distances D in key order (or None)."""
    Z, _ = _core.linkage(X, D, order, _CODE[method], path)
    return Z


def _fit(y, method, metric, optimal_ordering, func, path=0, condensed_1d=True, nan_policy=None):
    """(Z, nobs, omitted) of the data y: observations (2-D, or 1-D for agglomerative) or distances."""
    method = _method(method, func)
    metric = _metric(metric, func)
    optimal_ordering = _bool(optimal_ordering, func, "optimal_ordering")
    _check_squared(method, metric, func)
    a = _array(y, func, "y" if condensed_1d else "X")
    omitted = None
    if metric == "precomputed" or (condensed_1d and a.ndim == 1):
        if metric not in ("euclidean", "precomputed"):
            raise ValueError(f"{func}(): a 1-D 'y' holds condensed distances; 'metric' does not apply, not {metric!r}")
        D, n = distances(a, func, "y" if condensed_1d else "X")
        order = _sd.arange(n, dtype=_sd.int64)
        check_alloc(8 * D.shape[0], func)  # the algorithm's working copy
        Z = _tree(None, D, order, method, path)
        if optimal_ordering:
            Z = _olo(Z, D, order, n, func)
        return Z, n, omitted
    arg = "y" if condensed_1d else "X"
    X = points(a, func, arg)
    X, _, omitted = _rows(X, func, nan_policy, None, arg)
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"{func}(): '{arg}' must have at least 2 points, not {n}")
    order = _core.lex_order(X)
    Xr = contiguous(X[order])
    if method in _MATRIX_FREE and metric == "euclidean" and path != 1:
        Z = _tree(Xr, None, order, method, path)
    else:
        D = _pdist(Xr, metric, func)
        Z = _tree(Xr if (method in _CENTROID or path == 1) else None, D, order, method, path)
    if optimal_ordering:
        D = _pdist(X, metric, func)
        key = _sd.empty((n,), dtype=_sd.int64)
        key[order] = _sd.arange(n, dtype=_sd.int64)
        Z = _olo(Z, D, key, n, func)
    return Z, n, omitted


def _olo(Z, D, key, n, func):
    check_alloc(8 * n * n, func)
    return _core.optimal_leaf_ordering(Z, D, key)


class _Rows:
    """A checked linkage matrix: n, the rows as lists, and the array."""

    __slots__ = ("Z", "n", "a", "b", "h", "c")

    def __init__(self, Z, n, a, b, h, c):
        self.Z, self.n, self.a, self.b, self.h, self.c = Z, n, a, b, h, c

    def monotone(self):
        h = self.h
        return all(h[i] <= h[i + 1] for i in range(len(h) - 1))


def _why_invalid(Z):
    """(the reason Z is not a valid linkage matrix, or None, and the checked _Rows)."""
    if isinstance(Z, _sd.ndarray):
        a = Z
    else:
        try:
            a = _sd.asarray(Z)
        except TypeError, ValueError:
            return "it is not an array of numbers", None
    if a.dtype == _sd.complex128 or a.dtype == _sd.string or a.dtype == _sd.bool:
        return f"its dtype is {a.dtype}", None
    if a.dtype != _sd.float64:
        try:
            a = a.astype(_sd.float64)
        except TypeError, ValueError:
            return f"its dtype is {a.dtype}", None
    if a.ndim != 2 or a.shape[1] != 4:
        return f"its shape is {tuple(a.shape)}, not (n - 1, 4)", None
    m = a.shape[0]
    if m < 1:
        return "it has no row (at least 2 observations are needed)", None
    n = m + 1
    rows = a.tolist()
    seen = [False] * (2 * n - 1)
    counts = [1] * n + [0] * m
    A, B, H, Cn = [], [], [], []
    for i, (x, y, h, c) in enumerate(rows):
        for v in (x, y):
            if not (v == v and float(v).is_integer()):
                return f"row {i} has a cluster number that is not an integer ({v!r})", None
            v = int(v)
            if not 0 <= v < n + i:
                return f"row {i} uses cluster {v} before it is formed", None
            if seen[v]:
                return f"row {i} uses cluster {v} a second time", None
            seen[v] = True
        if not (h == h and math.isfinite(h)):
            return f"row {i} has a height that is not finite ({h!r})", None
        if h < 0:
            return f"row {i} has a negative height ({h!r})", None
        x, y = int(x), int(y)
        if c != counts[x] + counts[y]:
            return f"row {i} has the count {c!r}, not {counts[x] + counts[y]}", None
        counts[n + i] = counts[x] + counts[y]
        A.append(x)
        B.append(y)
        H.append(h)
        Cn.append(int(c))
    return None, _Rows(contiguous(a), n, A, B, H, Cn)


def _linkage(Z, func, arg="Z"):
    why, r = _why_invalid(Z)
    if why is not None:
        raise ValueError(f"{func}(): '{arg}' is not a valid linkage matrix: {why}")
    return r


def _first_seen(parent_of, n):
    """Labels 0 .. K - 1 by first appearance in the points' order from a find function's roots."""
    out = [0] * n
    seen = {}
    for i in range(n):
        out[i] = seen.setdefault(parent_of(i), len(seen))
    return out


def _partition(r, merges):
    """The first-seen labels after the first `merges` rows of r (a list)."""
    n = r.n
    parent = list(range(2 * n - 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(merges):
        parent[find(r.a[i])] = n + i
        parent[find(r.b[i])] = n + i
    return _first_seen(find, n)


def _merges_below(r, height, func, why="its heights decrease somewhere"):
    """The number of rows of a monotone tree at or below height (ValueError for an inverted tree)."""
    if not r.monotone():
        raise ValueError(f"{func}(): cannot cut a non-monotonic tree by height ({why})")
    import bisect

    return bisect.bisect_right(r.h, height)


def _k(v, func, arg, n):
    k = integer(v, func, arg)
    if not 1 <= k <= n:
        raise ValueError(f"{func}(): '{arg}' must be between 1 and the number of points ({n}), not {k}")
    return k


def _height(v, func, arg):
    h = real(v, func, arg)
    if math.isnan(h):
        raise ValueError(f"{func}(): '{arg}' must not be NaN")
    return h


def _int_array(values):
    return _sd.asarray(values, dtype=_sd.int64) if values else _sd.empty((0,), dtype=_sd.int64)


# ---- linkage and the result -------------------------------------------------------------------------------------


@function
def linkage(y, /, *, method="single", metric="euclidean", optimal_ordering=False):
    """Hierarchical (agglomerative) clustering: the linkage matrix Z ((n - 1, 4) float64), scipy's form.

    y is a condensed distance vector (1-D, length n (n - 1) / 2) or observations ((n, p), one point a row; with
    metric='precomputed' an (n, n) distance matrix). method is one of 'single' (the default, as scipy),
    'complete', 'average', 'weighted', 'centroid', 'median' and 'ward' with scipy's heights; 'ward', 'centroid'
    and 'median' of observations need metric='euclidean' (given distances are taken to be Euclidean). Ties go
    to the pair of smallest point keys (the points sorted by their coordinates), so the same points in another
    order give the same tree. optimal_ordering=True flips merges to the leaf order of smallest sum of
    neighbouring distances (Bar-Joseph et al. 2001), as optimal_leaf_ordering.
    """
    func = "linkage"
    Z, _, _ = _fit(y, method, metric, optimal_ordering, func)
    return Z


def _coefficient(r):
    """R's agglomerative (agnes) and divisive (diana) coefficient: the mean of 1 - h(i) / h_top over the points,
    h(i) the height of the merge that first joins point i, h_top the last merge's height."""
    n = r.n
    first = [0.0] * n
    for i in range(n - 1):
        for v in (r.a[i], r.b[i]):
            if v < n:
                first[v] = r.h[i]
    top = r.h[-1]
    if top == 0:
        return math.nan
    return math.fsum(1.0 - v / top for v in first) / n


class HierarchyResult(ClusterResult):
    """The result of agglomerative and divisive (docs/api/cluster.md §2): the tree as scipy's linkage matrix and
    the cuts of it.

    Fields: linkage (Z, (n - 1, 4)), method ('single' .. 'ward', or 'diana'), metric, heights (Z's heights),
    order (the leaves' order, leaves_list(Z)), coefficient (R's agglomerative or divisive coefficient), nobs (the
    points in the tree) and omitted (the rows nan_policy='omit' left out). Z numbers the points of the tree:
    the input's rows without the omitted ones. A tree has no one partition: cut() gives them.
    """

    _TITLE = "Hierarchical Clustering Results"
    _REPR = ("method", "nobs", "metric", "coefficient")

    def cut(self, *, k=None, height=None):
        """The partition into k clusters (the last k - 1 merges undone, R's cutree(k=)) or at a height (the
        merges at or below it; a tree whose heights decrease somewhere cannot be cut so: ValueError): (n,)
        int64 labels 0 .. K - 1 in first-seen order, -1 for the rows left out. Exactly one of k and height."""
        func = "cut"
        if (k is None) == (height is None):
            raise TypeError(f"{func}(): give exactly one of 'k' and 'height'")
        r = _linkage(self.linkage, func)
        if k is not None:
            merges = r.n - _k(k, func, "k", r.n)
        else:
            merges = _merges_below(r, _height(height, func, "height"), func, f"method={self.method!r}")
        labels = _sd.asarray(_partition(r, merges), dtype=_sd.int64)
        om = self.omitted
        return labels_back(labels, om if bool(_sd.any(om)) else None)

    def cophenet(self):
        """The cophenetic distances: the condensed vector of the heights at which each pair first joins."""
        return cophenet(self.linkage)

    def dendrogram(
        self,
        *,
        p=30,
        truncate_mode=None,
        color_threshold=None,
        orientation="top",
        labels=None,
        count_sort=False,
        distance_sort=False,
        show_leaf_counts=True,
    ):
        """The dendrogram's layout without a plot: dendrogram(self.linkage, ...)."""
        return _dendrogram(
            self.linkage,
            p,
            truncate_mode,
            color_threshold,
            orientation,
            labels,
            count_sort,
            distance_sort,
            show_leaf_counts,
            "dendrogram",
        )

    def summary(self):
        """The regression box: the method, the points, the metric and the coefficient; a table of the last
        merges (the clusters before each, its height and the merged cluster's size)."""
        r = _linkage(self.linkage, "summary")
        om = int(_sd.count_nonzero(self.omitted))
        name = "divisive (DIANA)" if self.method == "diana" else f"agglomerative ({self.method})"
        left = [("Method:", name), ("No. Observations:", _summary.count(self.nobs))]
        if om:
            left.append(("Omitted rows (NaN):", str(om)))
        left.append(("Monotone:", _summary.flag(r.monotone())))
        right = [
            ("Metric:", self.metric),
            ("Coefficient:", _summary.stat(self.coefficient)),
            ("Top height:", _summary.stat(r.h[-1])),
        ]
        rows = []
        last = list(range(r.n - 2, max(-1, r.n - 12), -1))
        for i in last:
            rows.append((str(r.n - i), [_summary.forg(r.h[i], 4), str(r.c[i])]))
        table = _summary.Table(["Height", "Size"], rows, stub_head="clusters")
        return _summary.render(self._TITLE, header=(left, right), tables=(table,))


_calls.methods(HierarchyResult)


def _result(Z, method, metric, nobs, omitted, n_input):
    om = omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)
    heights = contiguous(Z[:, 2])
    holder = {}

    def rows():
        if "r" not in holder:
            holder["r"] = _linkage(Z, "HierarchyResult")
        return holder["r"]

    return HierarchyResult(
        linkage=Z,
        method=method,
        metric=metric,
        heights=heights,
        order=lambda: leaves_list(Z),
        coefficient=lambda: _coefficient(rows()),
        nobs=nobs,
        omitted=om,
    )


@function
def agglomerative(X, /, *, method="single", metric="euclidean", optimal_ordering=False, nan_policy="raise"):
    """Hierarchical (agglomerative) clustering as a result object: HierarchyResult with .linkage (the Z of
    linkage), .cut(k=, height=), .cophenet(), .dendrogram() and R's agglomerative coefficient.

    X holds observations ((n, p), a 1-D X is (n, 1)), or distances with metric='precomputed' (an (n, n) matrix
    or a condensed vector). method, metric and optimal_ordering as linkage; nan_policy 'raise' (the default) or
    'omit' (the rows with a NaN left out of the tree: cut() labels them -1).
    """
    func = "agglomerative"
    policy(nan_policy, func)
    Z, nobs, omitted = _fit(X, method, metric, optimal_ordering, func, condensed_1d=False, nan_policy=nan_policy)
    return _result(Z, method, metric, nobs, omitted, nobs if omitted is None else omitted.shape[0])


@function
def divisive(X, /, *, metric="euclidean", nan_policy="raise"):
    """Divisive hierarchical clustering, R's DIANA (cluster::diana): HierarchyResult with method='diana'.

    From one cluster of all points, every cluster of two or more points is split: the point of largest mean
    distance to the others starts a splinter group, and while some point is on average strictly closer to the
    splinter group than to the rest, the one where that difference is largest joins it (ties: the smaller key).
    The split's height is the cluster's diameter; .coefficient is R's divisive coefficient (dc). X as in
    agglomerative.
    """
    func = "divisive"
    metric = _metric(metric, func)
    policy(nan_policy, func)
    a = _array(X, func, "X")
    if metric == "precomputed":
        D, n = distances(a, func, "X")
        order = _sd.arange(n, dtype=_sd.int64)
        omitted, n_input = None, n
    else:
        P = points(a, func, "X")
        n_input = P.shape[0]
        P, _, omitted = _rows(P, func, nan_policy, None, "X")
        n = P.shape[0]
        if n < 2:
            raise ValueError(f"{func}(): 'X' must have at least 2 points, not {n}")
        order = _core.lex_order(P)
        D = _pdist(contiguous(P[order]), metric, func)
    Z = _core.diana(D, order)
    return _result(Z, "diana", metric, n, omitted, n_input)


# ---- cutting -------------------------------------------------------------------------------------------------


def _monocrit_labels(r, mc, t):
    """scipy's flat clusters of a monotone criterion: every maximal node with mc <= t is a cluster, the others'
    points singletons; numbered 1, 2, ... in scipy's traversal (left subtree, right subtree, then the node's own
    point children)."""
    n = r.n
    T = [0] * n
    visited = [False] * (2 * n - 1)
    stack = [2 * n - 2]
    nc = 0
    leader = -1
    while stack:
        root = stack[-1] - n
        lc, rc = r.a[root], r.b[root]
        if leader == -1 and mc[root] <= t:
            leader = root
            nc += 1
        if lc >= n and not visited[lc]:
            visited[lc] = True
            stack.append(lc)
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            stack.append(rc)
            continue
        if lc < n:
            if leader == -1:
                nc += 1
            T[lc] = nc
        if rc < n:
            if leader == -1:
                nc += 1
            T[rc] = nc
        if leader == root:
            leader = -1
        stack.pop()
    return T


def _count_clusters(r, mc, t):
    n = r.n
    count = 0  # the maximal nodes with mc <= t and the points outside them
    stack = [2 * n - 2]
    while stack:
        v = stack.pop()
        if v < n:
            count += 1
            continue
        if mc[v - n] <= t:
            count += 1
            continue
        stack.append(r.a[v - n])
        stack.append(r.b[v - n])
    return count


def _max_below(r, values):
    """values carried up the tree: out[i] = max(values[i], out of the row's merged children)."""
    n = r.n
    out = list(values)
    for i in range(n - 1):
        for c in (r.a[i], r.b[i]):
            if c >= n and out[c - n] > out[i]:
                out[i] = out[c - n]
    return out


def _maxclust(r, mc, k):
    """scipy's maxclust: the smallest threshold among mc's values that leaves at most k clusters (fewer where
    nodes share a value); k >= n gives the points 1 .. n."""
    n = r.n
    if k >= n:
        return list(range(1, n + 1))
    vals = sorted(set(mc))
    lo, hi = 0, len(vals) - 1  # vals[hi] (the root's) leaves one cluster
    while lo < hi:
        mid = (lo + hi) // 2
        if _count_clusters(r, mc, vals[mid]) <= k:
            hi = mid
        else:
            lo = mid + 1
    return _monocrit_labels(r, mc, vals[lo])


def _check_monocrit(r, mc, func):
    n = r.n
    for i in range(n - 1):
        for c in (r.a[i], r.b[i]):
            if c >= n and mc[c - n] > mc[i]:
                raise ValueError(
                    f"{func}(): 'monocrit' must not decrease from a cluster to its parent (row {c - n} > row {i})"
                )


def _row_values(v, r, func, arg, cols=None):
    a = _array(v, func, arg)
    want = (r.n - 1,) if cols is None else (r.n - 1, cols)
    if tuple(a.shape) != want:
        raise ValueError(f"{func}(): '{arg}' must have the shape {want}, not {tuple(a.shape)}")
    if bool(_sd.any(_sd.isnan(a))):
        raise ValueError(f"{func}(): '{arg}' contains NaN")
    return contiguous(a)


def _inconsistency(R, r, func):
    R = _row_values(R, r, func, "R", 4)
    rows = R.tolist()
    for i, (_, sd_, cnt, _) in enumerate(rows):
        if sd_ < 0 or cnt < 1 or not float(cnt).is_integer():
            raise ValueError(f"{func}(): 'R' is not an inconsistency matrix (row {i})")
    return rows


@function
def fcluster(Z, t, /, *, criterion="inconsistent", depth=2, R=None, monocrit=None):
    """Flat clusters of a tree (scipy's fcluster): (n,) int64 labels from 1, numbered as scipy numbers them.

    criterion 'inconsistent' (the default: every cluster's nodes have an inconsistency coefficient <= t, from R
    or inconsistent(Z, d=depth)), 'distance' (cophenetic distances <= t; a tree whose heights decrease somewhere
    gives ValueError), 'maxclust' (at most t clusters: the smallest height that leaves no more, so fewer where
    merges share a height; a non-monotone tree is cut into exactly t clusters by merge order), 'monocrit'
    (monocrit[i] <= t, monocrit not decreasing up the tree) and 'maxclust_monocrit' (at most t clusters by
    monocrit). t is a real number, an int >= 1 for the maxclust criteria.
    """
    func = "fcluster"
    r = _linkage(Z, func)
    choice(criterion, func, "criterion", CRITERIA)
    n = r.n
    if criterion in ("maxclust", "maxclust_monocrit"):
        k = integer(t, func, "t")
        if k < 1:
            raise ValueError(f"{func}(): 't' must be >= 1 for criterion={criterion!r}, not {k}")
    else:
        tt = _height(t, func, "t")
    if criterion != "inconsistent" and R is not None:
        raise ValueError(f"{func}(): 'R' is used only with criterion='inconsistent'")
    if criterion not in ("monocrit", "maxclust_monocrit") and monocrit is not None:
        raise ValueError(f"{func}(): 'monocrit' is used only with the monocrit criteria")
    if criterion == "inconsistent":
        if R is None:
            d = integer(depth, func, "depth")
            if d < 0:
                raise ValueError(f"{func}(): 'depth' must be >= 0, not {d}")
            rows = inconsistent(Z, d=d).tolist()
        else:
            rows = _inconsistency(R, r, func)
        T = _monocrit_labels(r, _max_below(r, [row[3] for row in rows]), tt)
    elif criterion == "distance":
        if not r.monotone():
            raise ValueError(f"{func}(): cannot cut a non-monotonic tree by height (criterion='distance')")
        T = _monocrit_labels(r, _max_below(r, r.h), tt)
    elif criterion == "maxclust":
        if r.monotone():
            T = _maxclust(r, _max_below(r, r.h), k)
        else:
            T = list(range(1, n + 1)) if k >= n else _monocrit_labels(r, list(range(n - 1)), n - 1 - k)
    else:
        if monocrit is None:
            raise ValueError(f"{func}(): criterion={criterion!r} needs 'monocrit'")
        mc = _row_values(monocrit, r, func, "monocrit").tolist()
        _check_monocrit(r, mc, func)
        T = _monocrit_labels(r, mc, tt) if criterion == "monocrit" else _maxclust(r, mc, k)
    return _sd.asarray(T, dtype=_sd.int64)


def _values(v, func, arg, kind):
    """(the values as a list, given as a scalar)."""
    if isinstance(v, (numbers.Number, str)) or (type(v).__module__ == "numpy" and getattr(v, "ndim", None) == 0):
        return [kind(v, func, arg)], True
    try:
        a = v if isinstance(v, _sd.ndarray) else _sd.asarray(v)
    except TypeError, ValueError:
        raise TypeError(f"{func}(): '{arg}' must be a number or a 1-D sequence, not {type_name(v)}") from None
    if a.ndim != 1:
        raise ValueError(f"{func}(): '{arg}' must be a number or a 1-D sequence, not {a.ndim}-D")
    return [kind(x, func, arg) for x in a.tolist()], False


@function
def cut_tree(Z, /, *, n_clusters=None, height=None):
    """The partitions of a tree at given numbers of clusters or heights (scipy's cut_tree): (n, m) int64, a
    column per value (m = 1 for one number), labels 0 .. K - 1 in first-seen order. n_clusters undoes the last
    k - 1 merges (R's cutree(k=)); height keeps the merges at or below it (a non-monotone tree: ValueError);
    neither gives the n columns after 0, 1, ..., n - 1 merges."""
    func = "cut_tree"
    r = _linkage(Z, func)
    n = r.n
    if n_clusters is not None and height is not None:
        raise TypeError(f"{func}(): give at most one of 'n_clusters' and 'height'")
    if n_clusters is not None:
        ks, _ = _values(n_clusters, func, "n_clusters", lambda v, f, a: _k(v, f, a, n))
        merges = [n - k for k in ks]
    elif height is not None:
        hs, _ = _values(height, func, "height", _height)
        merges = [_merges_below(r, h, func) for h in hs]
    else:
        merges = list(range(n))
    cols = [_partition(r, m) for m in merges]
    out = _sd.empty((n, len(cols)), dtype=_sd.int64)
    for j, col in enumerate(cols):
        out[:, j] = _sd.asarray(col, dtype=_sd.int64)
    return out


# ---- the other scipy tools --------------------------------------------------------------------------------------


@function
def cophenet(Z, /):
    """The cophenetic distances of a tree: the condensed vector (n (n - 1) / 2,) of the height at which each pair
    of points first joins (scipy's cophenet(Z))."""
    r = _linkage(Z, "cophenet")
    check_alloc(8 * (r.n * (r.n - 1) // 2), "cophenet")
    return _core.cophenet(r.Z)


@function
def cophenet_correlation(Z, y, /):
    """The cophenetic correlation: CophenetResult(correlation, distances), the Pearson correlation of the
    cophenetic distances with the condensed distances y (scipy's cophenet(Z, Y), which returns a tuple)."""
    func = "cophenet_correlation"
    r = _linkage(Z, func)
    a = _array(y, func, "y")
    if a.ndim != 1:
        raise ValueError(f"{func}(): 'y' must be a condensed distance vector (1-D), not {a.ndim}-D")
    if a.shape[0] != r.n * (r.n - 1) // 2:
        raise ValueError(f"{func}(): 'y' must have n (n - 1) / 2 = {r.n * (r.n - 1) // 2} values, not {a.shape[0]}")
    if bool(_sd.any(_sd.isnan(a))):
        raise ValueError(f"{func}(): 'y' contains NaN")
    if bool(_sd.any(_sd.isinf(a))):
        raise ValueError(f"{func}(): 'y' contains infinity")
    c = cophenet(Z)
    dc = c - _sd.mean(c)
    dy = a - _sd.mean(a)
    num = float(_sd.sum(dc * dy))
    den = math.sqrt(float(_sd.sum(dc * dc)) * float(_sd.sum(dy * dy)))
    return CophenetResult(num / den if den > 0 else math.nan, c)


@function
def leaves_list(Z, /):
    """The leaves' order of the tree (scipy's leaves_list): (n,) int64, the points left to right, every merge's
    first cluster (Z[i, 0]) on the left."""
    r = _linkage(Z, "leaves_list")
    n = r.n
    out = []
    stack = [2 * n - 2]
    while stack:
        v = stack.pop()
        if v < n:
            out.append(v)
        else:
            stack.append(r.b[v - n])
            stack.append(r.a[v - n])
    return _sd.asarray(out, dtype=_sd.int64)


@function
def optimal_leaf_ordering(Z, y, /, *, metric="euclidean"):
    """The tree with its merges flipped to the leaf order of smallest sum of distances between neighbouring leaves
    (Bar-Joseph, Gifford and Jaakkola 2001; scipy's optimal_leaf_ordering): a new Z. y is a condensed distance
    vector or observations (with metric). The root's first cluster stays on the left; among orders of equal sum
    the smaller point numbers come first. O(n^3) time, O(n^2) memory."""
    func = "optimal_leaf_ordering"
    r = _linkage(Z, func)
    metric = _metric(metric, func)
    a = _array(y, func, "y")
    if a.ndim == 1:
        if metric not in ("euclidean", "precomputed"):
            raise ValueError(f"{func}(): a 1-D 'y' holds condensed distances; 'metric' does not apply, not {metric!r}")
        D, n = distances(a, func, "y")
    elif metric == "precomputed":
        D, n = distances(a, func, "y")
    else:
        X = points(a, func, "y")
        _rows(X, func, None, None, "y")
        n = X.shape[0]
        D = _pdist(X, metric, func)
    if n != r.n:
        raise ValueError(f"{func}(): 'y' has {n} points but 'Z' has {r.n}")
    return _olo(r.Z, D, _sd.arange(n, dtype=_sd.int64), n, func)


@function
def inconsistent(Z, /, *, d=2):
    """The inconsistency statistics of every merge (scipy's inconsistent): (n - 1, 4) float64 rows (the mean and
    the standard deviation (n - 1 in the denominator; 0 for one height) of the heights of the merge and the
    merges up to d - 1 levels below it, their count, and the coefficient (height - mean) / deviation, 0 when the
    deviation is 0). d = 0 counts the merge alone, as d = 1."""
    func = "inconsistent"
    r = _linkage(Z, func)
    depth = integer(d, func, "d")
    if depth < 0:
        raise ValueError(f"{func}(): 'd' must be >= 0, not {depth}")
    n = r.n
    out = []
    for i in range(n - 1):
        hs = []
        stack = [(i, 1)]
        while stack:
            j, lev = stack.pop()
            hs.append(r.h[j])
            if lev < depth:
                for c in (r.a[j], r.b[j]):
                    if c >= n:
                        stack.append((c - n, lev + 1))
        k = len(hs)
        mean = math.fsum(hs) / k
        if k > 1:
            var = math.fsum((h - mean) ** 2 for h in hs) / (k - 1)
            sdv = math.sqrt(var)
        else:
            sdv = 0.0
        coef = (r.h[i] - mean) / sdv if sdv > 0 else 0.0
        out.append([mean, sdv, float(k), coef])
    return _sd.asarray(out, dtype=_sd.float64)


@function
def maxdists(Z, /):
    """The largest height in each merge's subtree (scipy's maxdists): (n - 1,) float64."""
    r = _linkage(Z, "maxdists")
    return _sd.asarray(_max_below(r, r.h), dtype=_sd.float64)


@function
def maxinconsts(Z, R, /):
    """The largest inconsistency coefficient in each merge's subtree (scipy's maxinconsts), R from
    inconsistent: (n - 1,) float64."""
    func = "maxinconsts"
    r = _linkage(Z, func)
    rows = _inconsistency(R, r, func)
    return _sd.asarray(_max_below(r, [row[3] for row in rows]), dtype=_sd.float64)


@function
def maxRstat(Z, R, i, /):  # noqa: N802  (scipy's name)
    """The largest R[:, i] in each merge's subtree (scipy's maxRstat), i in 0 .. 3: (n - 1,) float64."""
    func = "maxRstat"
    r = _linkage(Z, func)
    rows = _inconsistency(R, r, func)
    col = integer(i, func, "i")
    if not 0 <= col <= 3:
        raise ValueError(f"{func}(): 'i' must be 0, 1, 2 or 3, not {col}")
    return _sd.asarray(_max_below(r, [row[col] for row in rows]), dtype=_sd.float64)


@function
def leaders(Z, T, /):
    """The leaders of flat clusters (scipy's leaders): (L, M), L[j] the largest cluster of the tree (a point
    number below n, else n + row) whose points are exactly flat cluster M[j]; (k,) int64 each, in scipy's order.
    T holds one flat label per point (fcluster's); a flat cluster that is not one subtree gives ValueError."""
    func = "leaders"
    r = _linkage(Z, func)
    a = _array(T, func, "T")
    if a.ndim != 1 or a.shape[0] != r.n:
        raise ValueError(
            f"{func}(): 'T' must have one label per point ({r.n}), not {a.shape[0] if a.ndim == 1 else a.size}"
        )
    labels = a.tolist()
    if any(not float(v).is_integer() for v in labels):
        raise ValueError(f"{func}(): 'T' must hold integer labels")
    labels = [int(v) for v in labels]
    n = r.n
    fid = [0] * (n - 1)
    L, M = [], []
    seen = set()

    def lead(node, c):
        if c in seen:
            raise ValueError(f"{func}(): flat cluster {c} of 'T' is not one subtree of 'Z'")
        seen.add(c)
        L.append(node)
        M.append(c)

    # scipy's order: a traversal from the root (left subtree, right subtree, then the node itself)
    visited = [False] * (2 * n - 1)
    stack = [2 * n - 2]
    while stack:
        i = stack[-1] - n
        lc, rc = r.a[i], r.b[i]
        if lc >= n and not visited[lc]:
            visited[lc] = True
            stack.append(lc)
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            stack.append(rc)
            continue
        fl = labels[lc] if lc < n else fid[lc - n]
        fr = labels[rc] if rc < n else fid[rc - n]
        if fl is not None and fl == fr:
            fid[i] = fl
        else:
            fid[i] = None
            if fl is not None:
                lead(lc, fl)
            if fr is not None:
                lead(rc, fr)
        stack.pop()
    if fid[n - 2] is not None:
        lead(2 * n - 2, fid[n - 2])
    return _int_array(L), _int_array(M)


@function
def is_monotonic(Z, /):
    """Whether the heights never decrease along the rows (scipy's is_monotonic)."""
    r = _linkage(Z, "is_monotonic")
    return r.monotone()


@function
def is_valid_linkage(Z, /):
    """Whether Z is a valid linkage matrix: (n - 1, 4) numbers, n >= 2, every row merging two clusters formed
    before it (integers), each cluster merged once, finite heights >= 0, and counts the sums of the merged
    clusters' (scipy's is_valid_linkage without warning, throw and name; it does not check integers, counts
    or finiteness)."""
    why, _ = _why_invalid(Z)
    return why is None


@function
def num_obs_linkage(Z, /):
    """The number of points of a tree: Z's rows + 1."""
    return _linkage(Z, "num_obs_linkage").n


@function
def is_isomorphic(T1, T2, /):
    """Whether two flat clusterings are the same partition up to the labels' names (scipy's is_isomorphic)."""
    func = "is_isomorphic"
    out = []
    for arg, T in (("T1", T1), ("T2", T2)):
        if isinstance(T, (list, tuple)) and not any(isinstance(v, (list, tuple)) for v in T):
            out.append(list(T))  # labels of any hashable kind (sklearn's and scipy's: numbers, strings)
            continue
        if isinstance(T, _sd.ndarray):
            a = T
        else:
            try:
                a = _sd.asarray(T)
            except TypeError, ValueError:
                raise TypeError(f"{func}(): '{arg}' must be a 1-D array of labels, not {type_name(T)}") from None
        if a.ndim != 1:
            raise ValueError(f"{func}(): '{arg}' must be 1-D, not {a.ndim}-D")
        out.append(a.tolist())
    a, b = out
    if len(a) != len(b):
        raise ValueError(f"{func}(): 'T1' and 'T2' must have the same length, not {len(a)} and {len(b)}")
    ab, ba = {}, {}
    for x, y in zip(a, b):
        if ab.setdefault(x, y) != y or ba.setdefault(y, x) != x:
            return False
    return True


# ---- dendrogram -------------------------------------------------------------------------------------------------


def _dendrogram(
    Z, p, truncate_mode, color_threshold, orientation, labels, count_sort, distance_sort, show_leaf_counts, func
):
    r = _linkage(Z, func)
    n = r.n
    p = integer(p, func, "p")
    if truncate_mode is not None:
        choice(truncate_mode, func, "truncate_mode", TRUNCATE)
    choice(orientation, func, "orientation", ORIENTATIONS)
    for arg, v in (("count_sort", count_sort), ("distance_sort", distance_sort)):
        if not (isinstance(v, bool) or v in ("ascending", "descending")):
            raise ValueError(f"{func}(): '{arg}' must be False, True, 'ascending' or 'descending', not {v!r}")
    _bool(show_leaf_counts, func, "show_leaf_counts")
    if labels is not None:
        labels = list(labels.tolist() if isinstance(labels, _sd.ndarray) else labels)
        if len(labels) != n:
            raise ValueError(f"{func}(): 'labels' must have one label per point ({n}), not {len(labels)}")
    if color_threshold is None:
        ct = 0.7 * max(r.h)
    else:
        ct = _height(color_threshold, func, "color_threshold")
    if truncate_mode == "lastp" and (p > n or p == 0):
        p = n
    if truncate_mode == "level" and p <= 0:
        p = math.inf
    icoord, dcoord, colors, ivl, leaves, leaf_colors = [], [], [], [], [], []
    state = {"color": 0, "below": False}

    def count(i):
        return 1 if i < n else r.c[i - n]

    def dist(i):
        return 0.0 if i < n else r.h[i - n]

    def leaf(i, level):
        """(x, width, height, maxdist) when node i is drawn as a leaf, else None."""
        if truncate_mode == "lastp" and 2 * n - p > i >= n:
            leaves.append(i)
            ivl.append(f"({r.c[i - n]})" if show_leaf_counts else "")
            return dist(i)
        if truncate_mode == "level" and i > n and level > p:
            leaves.append(i)
            ivl.append(f"({r.c[i - n]})" if show_leaf_counts else "")
            return dist(i)
        if i < n:
            leaves.append(i)
            ivl.append(labels[i] if labels is not None else str(i))
            return 0.0
        return None

    def order(i):
        aa, ab = r.a[i - n], r.b[i - n]
        if count_sort in ("ascending", True):
            return (ab, aa) if count(aa) > count(ab) else (aa, ab)
        if count_sort == "descending":
            return (aa, ab) if count(aa) > count(ab) else (ab, aa)
        if distance_sort in ("ascending", True):
            return (ab, aa) if dist(aa) > dist(ab) else (aa, ab)
        if distance_sort == "descending":
            return (aa, ab) if dist(aa) > dist(ab) else (ab, aa)
        return aa, ab

    # an explicit stack for scipy's recursion: frames [i, iv, level, stage, ua, ub, left result, colour]
    results = []
    stack = [[2 * n - 2, 0.0, 0, 0, 0, 0, None, None, None]]
    while stack:
        f = stack[-1]
        i, iv, level, stage = f[0], f[1], f[2], f[3]
        if stage == 0:
            md = leaf(i, level)
            if md is not None:
                results.append((iv + 5.0, 10.0, 0.0, md, True))
                stack.pop()
                continue
            f[4], f[5] = order(i)
            f[3] = 1
            stack.append([f[4], iv, level + 1, 0, 0, 0, None, None, None])
            continue
        if stage == 1:
            f[6] = results.pop()
            h = r.h[i - n]
            if h >= ct or ct <= 0:
                c = -1
                if state["below"]:
                    state["color"] += 1
                state["below"] = False
            else:
                state["below"] = True
                c = state["color"]
            f[7] = c
            f[3] = 2
            stack.append([f[5], iv + f[6][1], level + 1, 0, 0, 0, None, None, None])
            continue
        right = results.pop()
        left = f[6]
        h = r.h[i - n]
        icoord.append([left[0], left[0], right[0], right[0]])
        dcoord.append([left[2], h, h, right[2]])
        colors.append(f[7])
        for side in (left, right):
            if side[4]:
                leaf_colors.append((side[0], f[7]))
        stack.pop()
        results.append(((left[0] + right[0]) / 2.0, left[1] + right[1], h, max(left[3], right[3], h), False))
    by_x = dict(leaf_colors)
    leaf_color_list = [by_x.get(5.0 + 10.0 * j, -1) for j in range(len(leaves))]

    def arr(rows, cols):
        if not rows:
            return _sd.empty((0, cols), dtype=_sd.float64)
        return _sd.asarray(rows, dtype=_sd.float64)

    return DendrogramResult(
        arr(icoord, 4), arr(dcoord, 4), ivl, _int_array(leaves), _int_array(colors), _int_array(leaf_color_list)
    )


@function
def dendrogram(
    Z,
    /,
    *,
    p=30,
    truncate_mode=None,
    color_threshold=None,
    orientation="top",
    labels=None,
    count_sort=False,
    distance_sort=False,
    show_leaf_counts=True,
):
    """The layout of a tree's dendrogram without a plot (scipy's dendrogram(no_plot=True)): DendrogramResult(icoord,
    dcoord, ivl, leaves, color_list, leaves_color_list).

    truncate_mode None (the whole tree), 'lastp' (the last p merges: the clusters below them become leaves,
    labelled with their counts when show_leaf_counts) or 'level' (no more than p levels below the root);
    count_sort and distance_sort order each merge's two sides (False, True or 'ascending', 'descending');
    labels name the points. The links below color_threshold (default 0.7 times the largest height) are coloured
    by group: color_list holds group numbers 0, 1, ... in the order scipy's colours cycle, -1 for the links at
    or above the threshold (scipy gives matplotlib colour names). orientation is checked but changes nothing
    here (it is the plot's).
    """
    return _dendrogram(
        Z,
        p,
        truncate_mode,
        color_threshold,
        orientation,
        labels,
        count_sort,
        distance_sort,
        show_leaf_counts,
        "dendrogram",
    )
