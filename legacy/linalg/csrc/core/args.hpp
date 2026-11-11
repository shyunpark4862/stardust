// Parsing of the non-array arguments (API.md §0.1, §0.4).
#pragma once

#include "core/types.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace sdla {

namespace nb = nanobind;

// A Python int; bool is rejected (as in stardust-numpy). TypeError otherwise.
std::int64_t parse_int(nb::handle o, const char* func, const char* what);

// `dtype=`: None -> nullopt, else one of the four sdnp dtype objects (API.md §0.1).
std::optional<DType> parse_dtype(nb::handle o, const char* func);

// `axis=` of reductions: None -> every axis; an int or a tuple of ints, each
// in [-ndim, ndim), no repeats. Returns sorted axes in [0, ndim).
std::vector<int> parse_axes(nb::handle o, int ndim, const char* func);

// `rtol=` (API.md §0.4): None -> max(m, n) * eps; else a finite real number >= 0.
double parse_rtol(nb::handle o, std::int64_t m, std::int64_t n, const char* func);

// One axis in [-ndim, ndim) -> [0, ndim); ValueError otherwise.
int normalize_axis(std::int64_t axis, int ndim, const char* func);

}  // namespace sdla
