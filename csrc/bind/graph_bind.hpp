// The graph area: the submodule stardust._core.graph (docs/design/graph.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_graph(nanobind::module_& core);

}  // namespace sd
