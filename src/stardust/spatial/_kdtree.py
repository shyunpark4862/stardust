"""KDTree of sd.spatial (docs/api/spatial.md §4, docs/design/spatial.md §2.4, §3.3, §3.4; milestone SPT3).

A Python class holding the native tree (stardust._core.spatial._KDTree, csrc/spatial/kdtree.cpp), a data
structure like the sparse arrays and frame's Series (ARCHITECTURE §3). The data are always copied: the
tree keeps its own copy in leaf order, and `data` is a read-only copy in input order, made from it on first
use (decision SPT76: one copy of the data unless `data` is read). The build is
deterministic (decision SPT50: median splits under the total order (coordinate, index)), so a pickled
tree is rebuilt from its data and leafsize into the same tree.

query(x, k=, p=, distance_upper_bound=) returns the k nearest data points of each query point, sorted by
(distance, index), every distance the pair kernel of cdist and minkowski for that pair (SPT2): the result
is the brute force's with the same kernel, whatever the thread count.

The radius, pair and count queries (milestone SPT4; csrc/spatial/query.cpp, dual.cpp) keep that rule:
every distance they return and every boundary test d <= r is the same kernel's value, so
query_ball_point and query_ball_tree are radius_neighbors' rows, query_pairs pdist's pairs, and
count_neighbors the count of cdist's entries, whatever the path or the thread count.
"""

import math

from .. import _core as _sd
from .._calls import method
from .._results import QueryResult
from . import _common as C
from ._radius import RadiusResult

_core = _sd.spatial


def _readonly(a):
    a._set_readonly()
    return a


def _leafsize(v):
    if not C._int(v):
        raise TypeError(f"KDTree(): 'leafsize' must be an int, not {C.type_name(v)}")
    v = int(v)
    if v < 1:
        raise ValueError(f"KDTree(): 'leafsize' must be at least 1, not {v}")
    if v > C._INT64_MAX:
        raise OverflowError("KDTree(): 'leafsize' is out of range")
    return v


def _brute_pays(n, m, k):
    """Whether kneighbors' brute force (the GEMM pre-selection) beats the tree for p = 2 (decisions SPT21,
    SPT75, measured in SPT6, docs/design/spatial.md §5): the tree's work per query grows about as 2^(5m/6)
    and falls a little with k, the brute force's as n."""
    return n > 0 and m * 5 / 6 + 0.25 * math.log2(k) + 7 - math.log2(n) >= 0.5


def _search(tree, X, ranks, p, ub, kmax):
    """The k-nearest search of the rows of X (Q, m), (distances, indices) of the ranks. The path is the
    tree, or for p = 2 and a shape where kneighbors' brute force is faster (high m for the number of points)
    that brute force over the data in input order (SPT21, SPT75): the same kernel and the same order (d, j),
    so the same result (docs/design/spatial.md §1.3)."""
    n = tree.n
    if not (p == 2.0 and _brute_pays(n, tree.m, kmax)):
        return tree._tree.query(X, ranks, p, ub)
    sd = _sd
    C.finite(X, "query", "x")
    kc = min(kmax, n)
    d, i = _core._kneighbors(X, tree.data, "euclidean", 2.0, None, None, kc, ub, 0)
    q = X.shape[0]
    if kmax > kc:  # ranks past the data points are the fill (inf, n)
        d = sd.concat([d, sd.full((q, 1), math.inf)], axis=1)
        i = sd.concat([i, sd.full((q, 1), n, dtype=sd.int64)], axis=1)
    cols = sd.minimum(ranks - 1, kc)
    return d[:, cols], i[:, cols]


class KDTree:
    """A KD tree over the rows of `data` (n, m) for exact nearest-neighbour queries."""

    __slots__ = ("_tree", "_data", "_mins", "_maxes")

    @method("KDTree")
    def __init__(self, data, /, *, leafsize=10):
        name = "KDTree"
        ls = _leafsize(leafsize)
        a = C.data(data, name, "data")
        if a.ndim != 2:
            raise ValueError(f"{name}(): 'data' must have 2 dimensions, not {a.ndim}")
        self._tree = _core._KDTree(a, ls)  # a copy in leaf order; checks that the data are finite
        self._data = None  # the input-order copy, made from the tree's on first use (SPT76)
        self._mins = _readonly(self._tree.mins())
        self._maxes = _readonly(self._tree.maxes())

    # ---- attributes ----

    @property
    def data(self):
        """The data (n, m) float64 in input order, read-only (a copy: changing the input afterwards changes
        nothing here)."""
        if self._data is None:
            self._data = _readonly(self._tree.data())
        return self._data

    @property
    def n(self):
        return self._tree.n

    @property
    def m(self):
        return self._tree.m

    @property
    def leafsize(self):
        return self._tree.leafsize

    @property
    def maxes(self):
        """The largest coordinate of the data on each axis (m,), read-only (zeros when n = 0)."""
        return self._maxes

    @property
    def mins(self):
        """The smallest coordinate of the data on each axis (m,), read-only (zeros when n = 0)."""
        return self._mins

    @property
    def size(self):
        """The number of nodes."""
        return self._tree.size

    def __repr__(self):
        return f"KDTree(n={self.n}, m={self.m}, leafsize={self.leafsize})"

    def __reduce__(self):  # rebuilt from the data and leafsize: the same tree (SPT50, SPT56)
        return (_rebuild, (memoryview(self.data).tobytes(), self.n, self.m, self.leafsize))

    # ---- queries ----

    @method("query")
    def query(self, x, /, *, k=1, p=2.0, distance_upper_bound=math.inf):
        """The k nearest data points of each point of x (..., m), as QueryResult(distances, indices)."""
        name = "query"
        sd = _sd
        kk = C.k_ranks(k, name)
        pv = C.tree_p(p, name)
        ub = C.upper_bound(distance_upper_bound, name)
        X = C.data(x, name, "x")
        m = self.m
        if X.shape[-1] != m:
            raise ValueError(f"{name}(): 'x' must have {m} entries on its last axis (the tree's m), not {X.shape[-1]}")
        batch = tuple(X.shape[:-1])
        q = math.prod(batch)
        ncol = kk if isinstance(kk, int) else len(kk)
        C.check_alloc(16 * q * ncol, name)
        ranks = sd.arange(1, kk + 1, dtype=sd.int64) if isinstance(kk, int) else sd.asarray(kk, dtype=sd.int64)
        d, i = _search(self, sd.reshape(X, (q, m)), ranks, pv, ub, kk if isinstance(kk, int) else max(kk))
        if kk == 1:  # an int k = 1: the last axis dropped
            if not batch:
                return QueryResult(float(d[0, 0]), int(i[0, 0]))
            shape = batch
        else:
            shape = batch + (ncol,)
        return QueryResult(_readonly(sd.reshape(d, shape)), _readonly(sd.reshape(i, shape)))

    def _points(self, x, name):
        """x (..., m) as (a C-contiguous (Q, m) float64 array, the batch shape), finite."""
        sd = _sd
        X = C.data(x, name, "x")
        m = self.m
        if X.shape[-1] != m:
            raise ValueError(f"{name}(): 'x' must have {m} entries on its last axis (the tree's m), not {X.shape[-1]}")
        C.finite(X, name, "x")
        batch = tuple(X.shape[:-1])
        X2 = sd.reshape(X, (math.prod(batch), m))
        if not X2._flags()["c_contiguous"]:
            X2 = X2.copy()
        return X2, batch

    @method("query_ball_point")
    def query_ball_point(self, x, r, /, *, p=2.0, return_length=False):
        """The data points within distance r (boundary included) of each point of x (..., m): a RadiusResult
        whose rows are the points of x in C order (indices increasing within a row), or with return_length
        the counts (int64 of x's batch shape; an int for one point)."""
        name = "query_ball_point"
        pv = C.tree_p(p, name)
        C.flag(return_length, name, "return_length")
        X, batch = self._points(x, name)
        rr = C.ball_radius(r, name, batch)
        out = _core._ball_point(self._tree, X, rr, pv, return_length)
        if return_length:
            return int(out[0]) if not batch else _sd.reshape(out, batch)
        indptr, indices, distances = out
        return RadiusResult(indptr, indices, distances, (X.shape[0], self.n))

    @method("query_ball_tree")
    def query_ball_tree(self, other, r, /, *, p=2.0):
        """For each point of this tree, the points of other within distance r (boundary included): a
        RadiusResult (self.n, other.n), indices increasing within a row."""
        name = "query_ball_tree"
        other = C.tree_other(other, name, self.m)
        rv = C.scalar_radius(r, name)
        pv = C.tree_p(p, name)
        indptr, indices, distances = _core._ball_tree(self._tree, other._tree, rv, pv)
        return RadiusResult(indptr, indices, distances, (self.n, other.n))

    @method("query_pairs")
    def query_pairs(self, r, /, *, p=2.0):
        """Every pair (i, j), i < j, of data points within distance r (boundary included): a (k, 2) int64
        array sorted by (i, j)."""
        name = "query_pairs"
        rv = C.scalar_radius(r, name)
        pv = C.tree_p(p, name)
        return _core._pairs(self._tree, rv, pv)

    @method("count_neighbors")
    def count_neighbors(self, other, r, /, *, p=2.0, weights=None, cumulative=True):
        """The number of pairs (x1 of self, x2 of other) within distance r (boundary included), or with
        weights the sum of the products of their weights; r one number (an int or a float) or a 1-D array
        (an array of counts). cumulative=False: the pairs with r[i - 1] < d <= r[i] (r nondecreasing)."""
        name = "count_neighbors"
        sd = _sd
        other = C.tree_other(other, name, self.m)
        scalar, radii = _radii(r, name)
        pv = C.tree_p(p, name)
        wa, wb = _weights(self, other, weights, name)
        C.flag(cumulative, name, "cumulative")
        if not cumulative and any(radii[k] > radii[k + 1] for k in range(len(radii) - 1)):
            raise ValueError(f"{name}(): 'r' must be nondecreasing when cumulative=False")
        order = sorted(range(len(radii)), key=radii.__getitem__)  # stable
        srt = sd.asarray([radii[k] for k in order], dtype=sd.float64) if radii else sd.zeros(0)
        idx = sd.asarray(order, dtype=sd.int64) if radii else sd.zeros(0, dtype=sd.int64)
        out = _core._count(self._tree, other._tree, srt, idx, pv, wa, wb, cumulative)
        if scalar:
            return float(out[0]) if out.dtype == sd.float64 else int(out[0])
        return out

    @method("sparse_distance_matrix")
    def sparse_distance_matrix(self, other, max_distance, /, *, p=2.0):
        """The distances of the pairs (x1 of self, x2 of other) within max_distance (boundary included) as an
        sd.sparse.csr_array (self.n, other.n); a zero distance is a stored zero (query_ball_tree(other,
        max_distance).tocsr())."""
        from .. import sparse

        name = "sparse_distance_matrix"
        other = C.tree_other(other, name, self.m)
        rv = C.scalar_radius(max_distance, name, "max_distance")
        pv = C.tree_p(p, name)
        indptr, indices, distances = _core._ball_tree(self._tree, other._tree, rv, pv)
        return sparse.csr_array((distances, indices, indptr), shape=(self.n, other.n))


def _radii(r, name):
    """count_neighbors' r: (scalar, the radii as a list of floats). One number (also numpy's scalars and 0-d
    arrays) or a 1-D array of nonnegative numbers or inf; NaN and negatives are ValueErrors (scipy counts
    every pair for NaN and counts pairs for a negative r)."""
    if isinstance(r, (int, float)) or type(r).__module__ == "numpy" and getattr(r, "ndim", 1) == 0:
        return True, [C.scalar_radius(r, name)]
    try:
        a = C.data(r, name, "r")
    except TypeError:
        raise TypeError(f"{name}(): 'r' must be a number or an array of numbers, not {C.type_name(r)}") from None
    if a.ndim != 1:
        raise ValueError(f"{name}(): 'r' must be a number or a 1-D array, not a {a.ndim}-D array")
    vals = a.tolist()
    if not all(v >= 0.0 for v in vals):  # NaN too
        raise ValueError(f"{name}(): every entry of 'r' must be a nonnegative number or inf")
    return False, vals


def _weights(tree, other, weights, name):
    """count_neighbors' weights as (self's, other's), each None or a finite (n,) float64 array: None, a tuple
    (self's, other's) either of which may be None, or one array when other is this tree (scipy's forms)."""
    if weights is None:
        return None, None
    if isinstance(weights, tuple):
        if len(weights) != 2:
            raise ValueError(
                f"{name}(): 'weights' as a tuple must have 2 entries (self's and other's), not {len(weights)}"
            )
        wa, wb = weights
    else:
        if other is not tree:
            raise ValueError(f"{name}(): one array of 'weights' is for 'other' being this tree; give a tuple "
                             "(self's, other's) for two trees")  # fmt: skip
        wa = wb = weights
    return _weight_vector(wa, name, "this tree", tree.n), _weight_vector(wb, name, "'other'", other.n)


def _weight_vector(w, name, side, n):
    if w is None:
        return None
    a = C.data(w, name, "weights")
    if a.ndim != 1:
        raise ValueError(f"{name}(): the weights of {side} must have 1 dimension, not {a.ndim}")
    if a.shape[0] != n:
        raise ValueError(f"{name}(): the weights of {side} must have {n} entries (one per point), not {a.shape[0]}")
    C.finite(a, name, "weights")
    return a


def _rebuild(raw, n, m, leafsize):
    """A pickled KDTree: its data (bytes of n x m float64) and leafsize."""
    sd = _sd
    flat = sd.asarray(memoryview(raw).cast("d")) if n * m else sd.zeros(0)
    return KDTree(sd.reshape(flat, (n, m)), leafsize=leafsize)
