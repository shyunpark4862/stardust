// The graph area: the submodule stardust._core.graph (docs/design/graph.md §6).
#include "bind/graph_bind.hpp"

#include "graph/graph_mod.hpp"

namespace nb = nanobind;

namespace sd {

void bind_graph(nb::module_& core) {
    nb::module_ m = core.def_submodule("graph", "stardust graph native core");
    graph::bind_input(m);
    graph::bind_traversal(m);
    graph::bind_paths(m);
    graph::bind_yen(m);
    graph::bind_tree(m);
    graph::bind_flow(m);
    graph::bind_matching(m);
    graph::bind_ordering(m);
    graph::bind_structure(m);
    graph::bind_centrality(m);
}

}  // namespace sd
