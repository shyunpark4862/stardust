// The generalized linear models' numerical core: IRLS through the least squares core
// (regression/lsq.hpp), deviance and likelihood without cancellation, separation
// (docs/design/regression.md §3.4).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::regression {

void bind_glm(nanobind::module_& m);

}  // namespace sd::regression
