// The spatial area's native functions (docs/design/spatial.md §6): one bind_* per source file.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::spatial {

void bind_distance(nanobind::module_& m);
void bind_kdtree(nanobind::module_& m);
void bind_neighbors(nanobind::module_& m);
void bind_query(nanobind::module_& m);
void bind_dual(nanobind::module_& m);

}  // namespace sd::spatial
