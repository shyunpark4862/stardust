// Robust regression's numerical core (docs/design/regression.md §3.6): M and MM
// estimation by IRLS through the least squares core (regression/lsq.hpp), the fast-S
// start of MM, and the Theil-Sen and Siegel slopes with exact medians.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::regression {

void bind_robust(nanobind::module_& m);

}  // namespace sd::regression
