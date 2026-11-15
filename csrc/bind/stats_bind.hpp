// The statistics area's bindings: the submodule stardust._core.stats.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

// Call after the array bindings, like bind_special.
void bind_stats(nanobind::module_& core);

}  // namespace sd
