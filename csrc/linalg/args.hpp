// Parsing of the non-array arguments (docs/api/linalg.md §1.1).
#pragma once

#include "linalg/types.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace sd::linalg {

namespace nb = nanobind;

// A Python int; bool is rejected (as in the array area, docs/design/array.md §2.8).
// TypeError otherwise.
std::int64_t parse_int(nb::handle o, const char* func, const char* what);

// `dtype=`: None -> nullopt, else one of the four sd dtype objects (API.md §1).
std::optional<DType> parse_dtype(nb::handle o, const char* func);

// `axis=` of reductions: None -> every axis; an int or a tuple of ints, each
// in [-ndim, ndim), no repeats. Returns sorted axes in [0, ndim).
std::vector<int> parse_axes(nb::handle o, int ndim, const char* func);

// `rtol=` (docs/api/linalg.md §1.1): None -> max(m, n) * eps; else a finite real number >= 0.
double parse_rtol(nb::handle o, std::int64_t m, std::int64_t n, const char* func);

// One axis in [-ndim, ndim) -> [0, ndim); ValueError otherwise.
int normalize_axis(std::int64_t axis, int ndim, const char* func);

}  // namespace sd::linalg
