// Direct solves of sd.sparse through Accelerate's LU (docs/design/sparse.md §3.6): the scaffold
// of milestone SP0, a private entry point of stardust._core.sparse that the wrapper's tests call.
#pragma once

#include <nanobind/nanobind.h>

namespace sd::sparse {

void bind_direct(nanobind::module_& m);

}  // namespace sd::sparse
