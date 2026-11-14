// sd.random's bindings: the submodule stardust._core.random (docs/design/random.md).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

// Call after the array bindings: the kernels return the ndarray type registered there.
void bind_random(nanobind::module_& core);

}  // namespace sd
