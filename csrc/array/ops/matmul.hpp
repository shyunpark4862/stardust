// matmul and the @ operator (docs/api/array.md §9).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

nb::object matmul(nb::handle x1, nb::handle x2);

void bind_matmul(nb::module_& m);

}  // namespace sd
