// The sparse area: the submodule stardust._core.sparse (docs/design/sparse.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

void bind_sparse(nanobind::module_& core);

}  // namespace sd
