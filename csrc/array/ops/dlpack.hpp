// DLPack: from_dlpack and ndarray.__dlpack__ / __dlpack_device__
// (docs/api/array.md §3.5).
#pragma once

#include "array/ndarray.hpp"

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_dlpack(nb::module_& m, nb::class_<NDArray>& cls);

}  // namespace sd
