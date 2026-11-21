// Multiple comparisons and corrections (milestone H8, docs/design/htest.md §3.9): tukey_hsd and
// dunnett (the studentized range and the one-factor multivariate t, htest/studentized_range.hpp
// and htest/dunnett.hpp), the corrections of multipletests and false_discovery_control (one
// kernel, htest/multitest.hpp) and combine_pvalues, as kernels of stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_multicomp(nanobind::module_& m);

}  // namespace sd::htest
