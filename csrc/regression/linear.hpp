// The linear models' numerical core: QR least squares with iterative refinement
// on the augmented system, rank deficiency, covariances (docs/design/regression.md §3).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::regression {

void bind_linear(nanobind::module_& m);

}  // namespace sd::regression
