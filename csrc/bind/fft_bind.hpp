// The fft area: the submodule stardust._core.fft (docs/design/fft.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_fft(nanobind::module_& core);

}  // namespace sd
