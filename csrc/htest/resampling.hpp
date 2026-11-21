// Resampling and Monte Carlo (milestone H9, docs/design/htest.md §3.10): the resamples of
// permutation_test and bootstrap (exact enumeration or sd.random's Philox streams), the p-value
// counts, the bootstrap's quantiles, the Baumgartner-Weiss-Schindler statistic, the null tables
// of chi2_contingency and fisher_exact, multiscale_graphcorr and goodness_of_fit's fits, as
// kernels of stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_resampling(nanobind::module_& m);

}  // namespace sd::htest
