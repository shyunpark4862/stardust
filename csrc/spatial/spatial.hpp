// The spatial area's native functions (docs/design/spatial.md §6): one bind_* per source file.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::spatial {

void bind_distance(nanobind::module_& m);

}  // namespace sd::spatial
