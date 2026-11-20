// The frame area and the array area meet here, and only here (DESIGN.md §4.3,
// ARCHITECTURE.md §2): Column <-> sd.ndarray. (The sums both areas share --
// the pairwise tree and the exact int sum -- are in base/.)
#pragma once

#include "frame/column.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <vector>

namespace sd::frame {

namespace nb = nanobind;

// A 1-D sd.ndarray over the values of `col` (bool, int64 or float64, no NA),
// sharing its Buffer: no copy. Read-only unless `writeable` -- only for a
// column nobody else holds (just made by a conversion or an NA fill). A frame
// write while the view is alive copies first (Copy-on-Write sees the extra
// Buffer reference, docs/design/frame.md §5.2).
nb::object to_array_view(const Column& col, bool writeable);

// A Column holding a copy of the 1-D sd.ndarray `arr` (any strides). An array
// is always copied: it can change through another, writable view, which the
// frame's Copy-on-Write could not see. complex128 is a TypeError (the frame
// has no complex dtype).
Column from_array(nb::handle arr);

// A new (n, k) sd.ndarray in F order from k columns of one dtype without NA:
// each column is one memcpy, and sd.linalg takes F-order matrices as they are.
nb::object columns_to_array(const std::vector<Column>& cols, std::int64_t n);

}  // namespace sd::frame
