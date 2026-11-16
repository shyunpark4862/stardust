// The statistics area's bindings: the submodule stardust._core.stats.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

// Call after the array bindings, like bind_special.
void bind_stats(nanobind::module_& core);

// The samples, <distribution>_sample (bind/stats_sample.cpp); bind_stats calls it.
void bind_stats_sample(nanobind::module_& stats);

}  // namespace sd
