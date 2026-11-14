// quantile, median and their NaN-skipping forms nanquantile, nanmedian
// (docs/api/array.md §7.10).
//
// The values are numpy's bits: the same positions in the sorted lane and the
// same interpolation formula (numpy's _quantile and _lerp), evaluated in the
// same order. The result is always float64 (docs/design/array.md §6.10).
#pragma once

#include <nanobind/nanobind.h>

namespace sd {

namespace nb = nanobind;

void bind_quantile(nb::module_& m);

}  // namespace sd
