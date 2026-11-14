// Reductions (docs/api/array.md §7): sum, prod, min, max, mean, var, std, argmin,
// argmax, any, all, cumulative_sum, cumulative_prod, and their nan* forms.
//
// Results: a reduction over every axis gives a Python scalar
// (docs/design/array.md §2.2) unless keepdims=True. Summation order is not
// numpy's (docs/design/array.md §6.9) but is fixed: the same data in the same
// layout always gives the same bits.
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

nb::object sum(nb::handle x, nb::handle axis, nb::handle dtype, nb::handle keepdims);
nb::object prod(nb::handle x, nb::handle axis, nb::handle dtype, nb::handle keepdims);
nb::object min(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object max(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object mean(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object var(nb::handle x, nb::handle axis, nb::handle correction, nb::handle keepdims);
nb::object stddev(nb::handle x, nb::handle axis, nb::handle correction, nb::handle keepdims);  // Python: std
nb::object argmin(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object argmax(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object any(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object all(nb::handle x, nb::handle axis, nb::handle keepdims);

// NaN-skipping forms (docs/api/array.md §7.9).
nb::object nansum(nb::handle x, nb::handle axis, nb::handle dtype, nb::handle keepdims);
nb::object nanprod(nb::handle x, nb::handle axis, nb::handle dtype, nb::handle keepdims);
nb::object nanmin(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object nanmax(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object nanmean(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object nanvar(nb::handle x, nb::handle axis, nb::handle correction, nb::handle keepdims);
nb::object nanstd(nb::handle x, nb::handle axis, nb::handle correction, nb::handle keepdims);
nb::object nanargmin(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object nanargmax(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object count_nonzero(nb::handle x, nb::handle axis, nb::handle keepdims);  // docs/api/array.md §8.4

void bind_reductions(nb::module_& m);

}  // namespace sd
