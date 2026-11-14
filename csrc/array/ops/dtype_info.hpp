// can_cast, result_type, isdtype, finfo, iinfo (docs/api/array.md §2.1) and
// tensordot (docs/api/array.md §9).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_dtype_info(nb::module_& m);

}  // namespace sd
