// The cluster area's native core: one bind_* per file, called by bind/cluster_bind.cpp (docs/design/cluster.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::cluster {

void bind_hierarchy(nanobind::module_& m);  // hierarchy.cpp: linkage, DIANA, cophenet, optimal leaf ordering
void bind_prototype(nanobind::module_& m);  // milestone CL0's k-means measurements (prototype.cpp); CL2 removes it

}  // namespace sd::cluster
