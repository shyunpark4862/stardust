// Nonlinear least squares' iterations (docs/design/regression.md §3.7): MINPACK's
// Levenberg-Marquardt (lmder) and scipy's trust region reflective method with bounds, on a
// residual vector computed by a Python callback.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::regression {

void bind_nls(nanobind::module_& m);

}  // namespace sd::regression
