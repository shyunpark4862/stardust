// The proportions and the binomial (milestone H7, docs/design/htest.md §3.8): binomtest and
// its intervals, quantile_test, poisson_means_test (the E-test), statsmodels' proportions_ztest,
// proportions_chisquare (with a null value), proportion_confint and the tests and intervals of
// two independent proportions, as kernels of stardust._core.htest.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::htest {

void bind_proportion(nanobind::module_& m);

}  // namespace sd::htest
