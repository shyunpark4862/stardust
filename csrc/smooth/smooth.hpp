// The smooth area's native core: one bind_* per file, called by bind/smooth_bind.cpp (docs/design/smooth.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::smooth {

void bind_local(nanobind::module_& m);
void bind_loess(nanobind::module_& m);
void bind_savgol(nanobind::module_& m);
void bind_spline(nanobind::module_& m);
void bind_fitpack(nanobind::module_& m);
void bind_kreg(nanobind::module_& m);
void bind_kde(nanobind::module_& m);
void bind_bandwidth(nanobind::module_& m);

}  // namespace sd::smooth
