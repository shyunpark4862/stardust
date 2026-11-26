// The cluster area's native core: one bind_* per file, called by bind/cluster_bind.cpp (docs/design/cluster.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::cluster {

void bind_prototype(nanobind::module_& m);  // milestone CL0's measurements (prototype.cpp); CL1 and CL2 remove it

}  // namespace sd::cluster
