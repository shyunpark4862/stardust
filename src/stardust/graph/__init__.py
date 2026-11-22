"""stardust.graph: algorithms on graphs given as adjacency matrices (docs/api/graph.md,
docs/design/graph.md).

Usage: ``import stardust as sd`` then ``sd.graph.dijkstra(G, indices=0)``,
``sd.graph.connected_components(G)`` and ``sd.graph.minimum_spanning_tree(G)``. A graph is a square
adjacency matrix (an sd.sparse array or a dense array; nodes 0 .. n-1) and there is no Graph class.
The names are scipy.sparse.csgraph's (with scipy.optimize's linear_sum_assignment, and networkx's
meaning for the algorithms scipy lacks), in one flat namespace; inputs and options follow this
library's rules (decisions GR1-GR68 in docs/design/graph.md §1.1).

The public names: the traversals, connected components, topological sort and the conversion tools;
the shortest paths; the minimum spanning tree, the maximum flow, the matchings and the assignment; the
reverse Cuthill-McKee ordering, the structural rank and the laplacian; the articulation points,
bridges, biconnected components, core numbers, PageRank and betweenness centrality.
"""

from ._centrality import betweenness_centrality, pagerank
from ._common import NegativeCycleError
from ._convert import csgraph_from_dense, csgraph_to_dense
from ._flow import maximum_flow
from ._matching import linear_sum_assignment, maximum_bipartite_matching, min_weight_full_bipartite_matching
from ._ordering import laplacian, reverse_cuthill_mckee, structural_rank
from ._paths import (
    bellman_ford,
    construct_dist_matrix,
    dijkstra,
    floyd_warshall,
    johnson,
    reconstruct_path,
    shortest_path,
    yen,
)
from ._structure import articulation_points, biconnected_components, bridges, core_number
from ._traversal import (
    breadth_first_order,
    breadth_first_tree,
    connected_components,
    depth_first_order,
    depth_first_tree,
    topological_sort,
)
from ._tree import minimum_spanning_tree

__all__ = [
    "connected_components",
    "breadth_first_order",
    "depth_first_order",
    "breadth_first_tree",
    "depth_first_tree",
    "topological_sort",
    "csgraph_from_dense",
    "csgraph_to_dense",
    "shortest_path",
    "dijkstra",
    "bellman_ford",
    "johnson",
    "floyd_warshall",
    "yen",
    "construct_dist_matrix",
    "reconstruct_path",
    "NegativeCycleError",
    "minimum_spanning_tree",
    "maximum_flow",
    "maximum_bipartite_matching",
    "min_weight_full_bipartite_matching",
    "linear_sum_assignment",
    "reverse_cuthill_mckee",
    "structural_rank",
    "laplacian",
    "articulation_points",
    "bridges",
    "biconnected_components",
    "core_number",
    "pagerank",
    "betweenness_centrality",
]
