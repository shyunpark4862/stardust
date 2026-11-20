// The hypothesis tests' area: the submodule stardust._core.htest (docs/design/htest.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_htest(nanobind::module_& core);

}  // namespace sd
