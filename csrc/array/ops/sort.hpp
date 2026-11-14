// sort / argsort and unique_values / unique_counts / unique_inverse /
// unique_all (docs/api/array.md §8).
#pragma once

#include <nanobind/nanobind.h>

#include <cmath>
#include <cstdint>

namespace sd {

namespace nb = nanobind;

void bind_sort(nb::module_& m);

// Sorts a[0, n) ascending in the total order where -0.0 < +0.0; `a` holds
// no NaN. With `parallel`, a long array is sorted in pieces on the pool
// (docs/design/array.md §8.5) -- the same result either way, since the order
// has no ties between distinguishable values. quantile uses it
// (docs/design/array.md §6.10).
void sort_float64_total(double* a, std::int64_t n, bool parallel);

// The total order above, for selection (nth_element) of the same values.
struct TotalLess {
    bool operator()(double a, double b) const { return a < b || (a == b && std::signbit(a) && !std::signbit(b)); }
};

}  // namespace sd
