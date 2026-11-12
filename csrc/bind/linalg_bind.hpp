// The linalg area's bindings: the submodule stardust._core.linalg.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

// Call after the array bindings: linalg's functions take and return the
// ndarray type and the dtype objects registered there.
void bind_linalg(nanobind::module_& core);

}  // namespace sd
