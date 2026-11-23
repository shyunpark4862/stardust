// The smooth area: the submodule stardust._core.smooth (docs/design/smooth.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_smooth(nanobind::module_& core);

}  // namespace sd
