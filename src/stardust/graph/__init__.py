"""stardust.graph: algorithms on graphs given as adjacency matrices (docs/api/graph.md,
docs/design/graph.md).

Usage: ``import stardust as sd`` then ``sd.graph.dijkstra(G, indices=0)``,
``sd.graph.connected_components(G)`` and ``sd.graph.minimum_spanning_tree(G)``. A graph is a square
adjacency matrix (an sd.sparse array or a dense array; nodes 0 .. n-1) and there is no Graph class.
The names are scipy.sparse.csgraph's (with scipy.optimize's linear_sum_assignment, and networkx's
meaning for the algorithms scipy lacks), in one flat namespace; inputs and options follow this
library's rules (decisions GR1-GR28 in docs/design/graph.md §1.1).

The public names come milestone by milestone (docs/design/graph.md §8): GR1 the traversals, connected
components, topological sort and the conversion tools.
"""

from ._convert import csgraph_from_dense, csgraph_to_dense
from ._traversal import (
    breadth_first_order,
    breadth_first_tree,
    connected_components,
    depth_first_order,
    depth_first_tree,
    topological_sort,
)

__all__ = [
    "connected_components",
    "breadth_first_order",
    "depth_first_order",
    "breadth_first_tree",
    "depth_first_tree",
    "topological_sort",
    "csgraph_from_dense",
    "csgraph_to_dense",
]
