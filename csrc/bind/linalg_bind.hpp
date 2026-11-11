// The linalg area's bindings: the submodule stardust._core.linalg.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

// Call after the array bindings: linalg looks up the ndarray type, asarray,
// get_num_threads and the dtype objects on `core`.
void bind_linalg(nanobind::module_& core);

}  // namespace sd
