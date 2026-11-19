// Scalar roots and minima (docs/design/optimize.md §3.2, §3.3): root_scalar's and
// minimize_scalar's iterations (scalar.cpp) and the batched find_root, bracket_root,
// find_minimum and bracket_minimum (batch.cpp), on functions computed by Python callbacks.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::optimize {

void bind_scalar(nanobind::module_& m);
void bind_batch(nanobind::module_& m);

}  // namespace sd::optimize
