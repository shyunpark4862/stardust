// The interpolate area: the submodule stardust._core.interpolate (docs/design/interpolate.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_interpolate(nanobind::module_& core);

}  // namespace sd
