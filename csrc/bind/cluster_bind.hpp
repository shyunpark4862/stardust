// The cluster area: the submodule stardust._core.cluster (docs/design/cluster.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_cluster(nanobind::module_& core);

}  // namespace sd
