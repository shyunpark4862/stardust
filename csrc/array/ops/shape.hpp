// Shape manipulation (docs/api/array.md §4) and the real/imag views.
//
// Almost everything here is a *view*: a new NDArray over the same Buffer with
// different shape/strides/data. No element is touched -- that is why
// transposing a 1 GB array costs the same as transposing a 2x2 one. Only
// reshape (when the strides cannot express the new shape), concat and stack
// copy data.
#pragma once

#include "array/ndarray.hpp"

#include <nanobind/nanobind.h>

#include <optional>
#include <span>

namespace sd {

namespace nb = nanobind;

// reshape in C (row-major) index order: a view when the strides allow it,
// otherwise a copy, as `copy` (docs/design/array.md §2.8) permits.
nb::object reshape(nb::handle x, std::span<const std::int64_t> shape, std::optional<bool> copy);
nb::object reshape(nb::handle x, nb::handle shape, nb::handle copy);

nb::object matrix_transpose(nb::handle x);  // .mT: swap the last two axes
nb::object transpose_2d(nb::handle x);      // .T: 2-D only (docs/design/array.md §2.8)

void bind_shape(nb::module_& m);

}  // namespace sd
