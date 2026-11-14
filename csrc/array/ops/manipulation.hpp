// flip, roll, repeat, tile, unstack, meshgrid, tril, triu, diff
// (docs/api/array.md §4.4).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_manipulation(nb::module_& m);

}  // namespace sd
