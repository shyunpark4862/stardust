// The statistics area's bindings: the submodule stardust._core.stats.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

// Call after the array bindings, like bind_special.
void bind_stats(nanobind::module_& core);

// The samples, <distribution>_sample (bind/stats_sample.cpp); bind_stats calls it.
void bind_stats_sample(nanobind::module_& stats);

// The descriptive statistics' lane kernels, desc_* (bind/stats_desc.cpp).
void bind_stats_desc(nanobind::module_& stats);

// The multivariate distributions' kernels, mv_*, multinomial_*, dirichlet_* (bind/stats_mv.cpp).
void bind_stats_mv(nanobind::module_& stats);

// The multivariate normal and t cdfs, mv_cdf (bind/stats_mvcdf.cpp).
void bind_stats_mvcdf(nanobind::module_& stats);

}  // namespace sd
