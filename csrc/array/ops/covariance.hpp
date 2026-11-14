// cov and corrcoef (docs/api/array.md §7.11).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_covariance(nb::module_& m);

}  // namespace sd
