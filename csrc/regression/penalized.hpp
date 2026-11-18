// The penalized regressions' numerical core: ridge by one SVD for every alpha, lasso and
// elastic net by coordinate descent with an active-set (KKT) polish, and the penalized
// binomial and poisson GLMs (outer IRLS, inner coordinate descent) (docs/design/regression.md §3.5).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::regression {

void bind_penalized(nanobind::module_& m);

}  // namespace sd::regression
