// Array creation (docs/api/array.md §3). Python-facing signatures take raw handles
// so that argument rules (API.md §1, docs/api/array.md §1.1-§1.3, §1.5) are
// enforced in one place.
#pragma once

#include "array/ndarray.hpp"

#include <nanobind/nanobind.h>

#include <optional>

namespace sd {

namespace nb = nanobind;

nb::object asarray(nb::handle obj, nb::handle dtype, nb::handle copy);

// An exact numpy.ndarray of one of the four dtypes (native byte order, aligned,
// a bool one holding only 0 and 1) as a view of its memory, read from numpy's C
// struct; an invalid object for anything else, and when dtype= differs or copy is
// True (asarray.cpp). asarray and from_dlpack try it first.
nb::object share_numpy_array(nb::handle obj, std::optional<DType> dtype, std::optional<bool> copy);

nb::object zeros(nb::handle shape, nb::handle dtype);
nb::object ones(nb::handle shape, nb::handle dtype);
nb::object empty(nb::handle shape, nb::handle dtype);
nb::object full(nb::handle shape, nb::handle fill_value, nb::handle dtype);

nb::object zeros_like(nb::handle x, nb::handle dtype);
nb::object ones_like(nb::handle x, nb::handle dtype);
nb::object empty_like(nb::handle x, nb::handle dtype);
nb::object full_like(nb::handle x, nb::handle fill_value, nb::handle dtype);

nb::object arange(nb::handle start, nb::handle stop, nb::handle step, nb::handle dtype);
nb::object linspace(nb::handle start, nb::handle stop, nb::handle num, nb::handle dtype, nb::handle endpoint);
nb::object eye(nb::handle n_rows, nb::handle n_cols, nb::handle k, nb::handle dtype);

void bind_creation(nb::module_& m);

}  // namespace sd
