// Nonlinear least squares' iterations (docs/design/optimize.md §3.4): MINPACK's
// Levenberg-Marquardt (lmder) and scipy's trust region reflective method, on a residual
// vector computed by a Python callback. sd.optimize.least_squares and sd.regression.nls
// both run them.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::optimize {

void bind_lsq(nanobind::module_& m);

}  // namespace sd::optimize
