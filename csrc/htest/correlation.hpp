// The correlation tests (milestone H5, docs/design/htest.md §3.6): pearsonr, spearmanrho and
// spearmanr, pointbiserialr, kendalltau, weightedtau, somersd and chatterjeexi, as lane kernels
// of stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_correlation(nanobind::module_& m);

}  // namespace sd::htest
