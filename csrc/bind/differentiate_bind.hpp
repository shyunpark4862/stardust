// The numerical differentiation area: the submodule stardust._core.differentiate
// (docs/design/integrate.md §6.3).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_differentiate(nanobind::module_& core);

}  // namespace sd
