// The interpolate area's native core: one bind_* per file, called by bind/interpolate_bind.cpp
// (docs/design/interpolate.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::interpolate {

void bind_linear(nanobind::module_& m);
void bind_cubic(nanobind::module_& m);
void bind_ppoly(nanobind::module_& m);

}  // namespace sd::interpolate
