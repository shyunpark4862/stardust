"""KDTree of sd.spatial (docs/api/spatial.md §4, docs/design/spatial.md §2.4, §3.3, §3.4; milestone SPT3).

A Python class holding the native tree (stardust._core.spatial._KDTree, csrc/spatial/kdtree.cpp), a data
structure like the sparse arrays and frame's Series (ARCHITECTURE §3). The data are always copied: the
tree keeps its own copy in leaf order, and `data` is a read-only copy in input order. The build is
deterministic (decision SPT50: median splits under the total order (coordinate, index)), so a pickled
tree is rebuilt from its data and leafsize into the same tree.

query(x, k=, p=, distance_upper_bound=) returns the k nearest data points of each query point, sorted by
(distance, index), every distance the pair kernel of cdist and minkowski for that pair (SPT2): the result
is the brute force's with the same kernel, whatever the thread count.
"""

import math

from .. import _core as _sd
from .._calls import method
from .._results import QueryResult
from . import _common as C

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


def _search(tree, X, ranks, p, ub):
    """The k-nearest search of the rows of X (Q, m). The path is the tree; decision SPT21 lets a query of
    a shape where kneighbors' brute force is faster (high m, few points) take that path instead, with the
    same result (docs/design/spatial.md §1.3). SPT6 measures the threshold and adds the switch here."""
    return tree._tree.query(X, ranks, p, ub)


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
        own = _readonly(a.copy())  # always a copy: changing the input afterwards changes nothing here
        self._tree = _core._KDTree(own, ls)  # checks that the data are finite
        self._data = own
        self._mins = _readonly(self._tree.mins())
        self._maxes = _readonly(self._tree.maxes())

    # ---- attributes ----

    @property
    def data(self):
        """The data (n, m) float64 in input order, read-only."""
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
        return (_rebuild, (memoryview(self._data).tobytes(), self.n, self.m, self.leafsize))

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
        d, i = _search(self, sd.reshape(X, (q, m)), ranks, pv, ub)
        if kk == 1:  # an int k = 1: the last axis dropped
            if not batch:
                return QueryResult(float(d[0, 0]), int(i[0, 0]))
            shape = batch
        else:
            shape = batch + (ncol,)
        return QueryResult(_readonly(sd.reshape(d, shape)), _readonly(sd.reshape(i, shape)))


def _rebuild(raw, n, m, leafsize):
    """A pickled KDTree: its data (bytes of n x m float64) and leafsize."""
    sd = _sd
    flat = sd.asarray(memoryview(raw).cast("d")) if n * m else sd.zeros(0)
    return KDTree(sd.reshape(flat, (n, m)), leafsize=leafsize)
