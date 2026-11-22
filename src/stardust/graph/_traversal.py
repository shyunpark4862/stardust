"""The traversals, connected components and topological sort of sd.graph (docs/api/graph.md §2,
docs/design/graph.md §2.2, §3.3; milestone GR1).

The arguments are checked here; the searches are stardust._core.graph's (csrc/graph/traversal.cpp). The
answers are fixed by definitions, not by an algorithm's path: the neighbours of a node are visited in
ascending order (undirected: the union of its out- and in-neighbours; GR13), a component is numbered by
the order of its smallest node (GR12), and the topological order is the lexicographically smallest one.
"""

from .. import _core as _sd
from .._calls import function
from .._results import ConnectedComponentsResult, TraversalResult
from ..stats._descriptive import _choice
from . import _common as C

_core = _sd.graph


def _order(name, csgraph, i_start, directed, return_predecessors, depth):
    directed = C.flag(directed, name, "directed")
    return_predecessors = C.flag(return_predecessors, name, "return_predecessors")
    n, indptr, indices, _ = C.graph(csgraph, name)
    s = C.node(i_start, n, name, "i_start")
    order, pred = _core._traverse(indptr, indices, None, n, s, directed, depth, False)
    if return_predecessors:
        return TraversalResult(C.readonly(order), C.readonly(pred))
    return order


def _tree(name, csgraph, i_start, directed, depth):
    directed = C.flag(directed, name, "directed")
    n, indptr, indices, data = C.graph(csgraph, name, values=True)
    s = C.node(i_start, n, name, "i_start")
    tp, ti, td = _core._traverse(indptr, indices, data, n, s, directed, depth, True)
    return C.csr(n, tp, ti, td)


@function
def breadth_first_order(csgraph, i_start, /, *, directed=True, return_predecessors=True):
    """The nodes reached from i_start, breadth first (neighbours in ascending order), as
    TraversalResult(node_array, predecessors) (int64; -9999 for i_start and the nodes not reached), or
    node_array alone with return_predecessors=False."""
    return _order("breadth_first_order", csgraph, i_start, directed, return_predecessors, False)


@function
def depth_first_order(csgraph, i_start, /, *, directed=True, return_predecessors=True):
    """The nodes reached from i_start in the order of the recursive depth-first search (the smallest
    neighbour not yet found first), as breadth_first_order's result."""
    return _order("depth_first_order", csgraph, i_start, directed, return_predecessors, True)


@function
def breadth_first_tree(csgraph, i_start, /, *, directed=True):
    """The breadth-first search tree from i_start: an (n, n) float64 csr_array, an edge parent -> child of
    the original weight (undirected: the stored direction, the smaller of two) per node reached."""
    return _tree("breadth_first_tree", csgraph, i_start, directed, False)


@function
def depth_first_tree(csgraph, i_start, /, *, directed=True):
    """The depth-first search tree from i_start, as breadth_first_tree's."""
    return _tree("depth_first_tree", csgraph, i_start, directed, True)


@function
def connected_components(csgraph, /, *, directed=True, connection="weak", return_labels=True):
    """The connected components (weak or strong; directed=False: of the undirected reading) as
    ConnectedComponentsResult(n_components, labels): each component numbered by the order of its smallest
    node (int64 labels), or n_components alone with return_labels=False."""
    name = "connected_components"
    directed = C.flag(directed, name, "directed")
    connection = _choice(connection, name, "connection", ("weak", "strong"))
    return_labels = C.flag(return_labels, name, "return_labels")
    n, indptr, indices, _ = C.graph(csgraph, name)
    count, labels = _core._components(indptr, indices, n, directed and connection == "strong")
    if return_labels:
        return ConnectedComponentsResult(int(count), C.readonly(labels))
    return int(count)


@function
def topological_sort(csgraph, /):
    """The lexicographically smallest order of the nodes in which every edge i -> j has i before j (int64
    (n,)); ValueError when the graph has a cycle (a self loop too)."""
    name = "topological_sort"
    n, indptr, indices, _ = C.graph(csgraph, name)
    order = _core._toposort(indptr, indices, n)
    if order is None:
        raise ValueError(f"{name}(): the graph has a cycle")
    return order
