// Indexing (docs/api/array.md §5).
//
// Basic indices (int, slice, `...`, newaxis) never copy: they only move the
// data pointer and rewrite shape/strides. x[2:8:3] is "start 2 elements in,
// take 2, stride x3".
// Advanced indices (integer arrays, boolean masks) select arbitrary
// positions, so the result is a copy: gather for x[idx], scatter for
// x[idx] = v. numpy's semantics, except that advanced indices separated by a
// basic one are an IndexError.
#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>

namespace sd {

namespace nb = nanobind;

// x[key]: a view (basic), a copy (advanced), or a Python scalar when every
// axis is indexed by an int (docs/design/array.md §2.2).
nb::object getitem(nb::handle x, nb::handle key);

// x[key] = value: value (array, nested list or Python scalar) is broadcast to
// the selected shape and converted upward only (docs/design/array.md §2.5).
// With repeated advanced indices the last write wins.
void setitem(nb::handle x, nb::handle key, nb::handle value);

// x[i] for 0 <= i < len(x), without parsing a key: what iteration uses.
nb::object item_along_axis0(nb::handle x, std::int64_t i);

}  // namespace sd
