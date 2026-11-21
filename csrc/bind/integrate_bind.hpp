// The numerical integration area: the submodule stardust._core.integrate
// (docs/design/integrate.md §6.3).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_integrate(nanobind::module_& core);

}  // namespace sd
