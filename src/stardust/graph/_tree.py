"""The minimum spanning tree of sd.graph (docs/api/graph.md §4, docs/design/graph.md §2.4, §3.4; milestone GR3).

The argument is checked here; Kruskal's algorithm is stardust._core.graph's (csrc/graph/tree.cpp). The edges
of the undirected reading (the smaller weight of two directions) are compared by the key (w, min(i, j),
max(i, j)), a total order, so the minimum spanning forest is unique (GR17) and every chosen edge keeps the
place of the stored entry that gave its weight.
"""

from .. import _core as _sd
from .._calls import function
from . import _common as C

_core = _sd.graph


@function
def minimum_spanning_tree(csgraph, /):
    """The minimum spanning forest of the undirected reading of csgraph (a tree per connected component):
    an (n, n) float64 csr_array whose stored entries are the chosen edges, each at the place of the stored
    entry that gave its weight (the upper triangle when both directions weigh the same). Unique: edges of
    equal weight are compared by (min(i, j), max(i, j)). Negative weights are allowed; the diagonal is
    ignored."""
    name = "minimum_spanning_tree"
    n, indptr, indices, data = C.graph(csgraph, name, values=True)
    nonfinite, _ = _core._scan(data)
    if nonfinite:
        raise ValueError(f"{name}(): stored edge weights must be finite")
    return C.csr(n, *_core._mst(indptr, indices, data, n))
