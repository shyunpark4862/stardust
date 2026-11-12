// The special-function area's bindings: the submodule stardust._core.special.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

// Call after the array bindings: the functions take and return the ndarray
// type and the dtype objects registered there.
void bind_special(nanobind::module_& core);

}  // namespace sd
