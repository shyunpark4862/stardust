// interp: numpy's piecewise linear interpolation (docs/api/array.md §8.5; docs/design/interpolate.md §2.2).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_interp(nb::module_& m);

}  // namespace sd
