// logsumexp, softmax, log_softmax (docs/api/special.md §6; S5): reductions built
// from the array's max and pairwise sum (special/reduce.cpp).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::special {

namespace nb = nanobind;

nb::object logsumexp(nb::handle x, nb::handle axis, nb::handle keepdims);
nb::object softmax(nb::handle x, nb::handle axis);
nb::object log_softmax(nb::handle x, nb::handle axis);

}  // namespace sd::special
