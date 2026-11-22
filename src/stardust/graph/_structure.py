"""The articulation points, bridges, biconnected components and core numbers of sd.graph (docs/api/graph.md §6,
docs/design/graph.md §2.6, §3.7; milestone GR5).

The four read the graph undirected: the union of the two directions' patterns (a stored 0 or NaN is an edge;
a dense matrix by the graph rule), the diagonal left out. Each answer is a set or a partition, unique, and
comes sorted: the articulation points ascending, the bridges and the edges as rows (i, j) with i < j in
lexicographic order, the biconnected components numbered by the order of their smallest edge (decision
GR18). The searches are stardust._core.graph's (csrc/graph/structure.cpp): one iterative depth-first search
with Hopcroft and Tarjan's low points, and Batagelj and Zaversnik's peeling for the core numbers.
"""

from .. import _core as _sd
from .._calls import function
from .._results import BiconnectedComponentsResult
from . import _common as C

_core = _sd.graph


@function
def articulation_points(csgraph, /):
    """The nodes whose removal increases the number of connected components of the undirected graph
    (int64, ascending)."""
    n, indptr, indices, _ = C.graph(csgraph, "articulation_points")
    return _core._biconnected(indptr, indices, n, 0)


@function
def bridges(csgraph, /):
    """The edges whose removal increases the number of connected components of the undirected graph:
    (k, 2) int64, rows (i, j) with i < j in lexicographic order."""
    n, indptr, indices, _ = C.graph(csgraph, "bridges")
    return _core._biconnected(indptr, indices, n, 1)


@function
def biconnected_components(csgraph, /):
    """The edges of the undirected graph split into biconnected components:
    BiconnectedComponentsResult(n_components, edges, labels), edges every edge as a row (i, j) with i < j in
    lexicographic order, labels each edge's component, numbered by the order of the components' smallest
    edges. A component's nodes are those of edges[labels == c]."""
    n, indptr, indices, _ = C.graph(csgraph, "biconnected_components")
    count, edges, labels = _core._biconnected(indptr, indices, n, 2)
    return BiconnectedComponentsResult(int(count), C.readonly(edges), C.readonly(labels))


@function
def core_number(csgraph, /):
    """The core number of every node of the undirected graph (int64 (n,)): the largest k such that the node
    belongs to a subgraph whose nodes all have degree k or more. The k-core is core_number(G) >= k."""
    n, indptr, indices, _ = C.graph(csgraph, "core_number")
    return _core._core_number(indptr, indices, n)
