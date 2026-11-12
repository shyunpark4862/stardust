// Moving data between Python objects and Columns (docs/api/frame.md §2,
// docs/design/frame.md §3.4).
#pragma once

#include "frame/column.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <optional>

namespace sd::frame {

namespace nb = nanobind;

// A column from a Python sequence. `na` is the sd.frame.NA singleton; it and None
// become NA. Without `dtype` the dtype is inferred (docs/api/frame.md §2.2);
// with it, each value must convert to it without loss (docs/api/frame.md §2.3).
Column from_list(nb::handle seq, nb::handle na, std::optional<DType> dtype);

// A column from a 1-D numeric or bool numpy array, always copied and widened
// to int64/float64 (docs/api/frame.md §2.4). String and object arrays go through
// from_list instead (the Python layer calls tolist() first).
Column from_numpy(const nb::ndarray<nb::ro, nb::ndim<1>, nb::device::cpu>& arr);

// The values as Python objects; NA rows become `na`.
nb::list to_list(const Column& col, nb::handle na);

// A read-only numpy view of a bool/int64/float64 column's values, sharing the
// Buffer (NA rows show their zero fill). The array keeps the Buffer alive.
nb::object to_numpy_view(const Column& col);

// A fresh numpy bool array: True where the row has a value.
nb::object validity_mask(const Column& col);

}  // namespace sd::frame
