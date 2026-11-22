// The plot area: the submodule stardust._core.plot (docs/design/plot.md §3.5, §7).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_plot(nanobind::module_& core);

}  // namespace sd
