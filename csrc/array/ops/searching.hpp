// take, take_along_axis, nonzero, count_nonzero, searchsorted
// (docs/api/array.md §5.4, §8.4).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_searching(nb::module_& m);

}  // namespace sd
