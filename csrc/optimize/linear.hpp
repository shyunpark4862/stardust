// Linear least squares with bounds (docs/design/optimize.md §3.4): lsq_linear's 'trf' and
// 'bvls' and nnls (linear.cpp).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::optimize {

void bind_linear(nanobind::module_& m);

}  // namespace sd::optimize
