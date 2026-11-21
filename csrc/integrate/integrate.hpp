// sd.integrate's native parts (docs/design/integrate.md §3.7, §6.3): the sample rules
// (samples.cpp) and the Gauss-Legendre nodes and weights (gauss.cpp), bound in the submodule
// stardust._core.integrate.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::integrate {

void bind_samples(nanobind::module_& m);
void bind_gauss(nanobind::module_& m);

}  // namespace sd::integrate
