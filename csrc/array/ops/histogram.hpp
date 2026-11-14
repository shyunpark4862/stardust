// bincount and histogram (docs/api/array.md §7.12).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_histogram(nb::module_& m);

}  // namespace sd
