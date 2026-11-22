// The graph area's native functions (docs/design/graph.md §6): one bind_* per source file.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::graph {

void bind_input(nanobind::module_& m);
void bind_traversal(nanobind::module_& m);

}  // namespace sd::graph
