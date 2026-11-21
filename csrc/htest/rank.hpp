// The rank tests (milestone H3, docs/design/htest.md §3.4): mannwhitneyu, ranksums, wilcoxon,
// kruskal, friedmanchisquare, brunnermunzel, ansari, mood, median_test and page_trend_test, as
// lane kernels of stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_rank(nanobind::module_& m);

}  // namespace sd::htest
