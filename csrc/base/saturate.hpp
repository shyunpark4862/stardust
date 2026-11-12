// float64 -> int64 for an explicit conversion, the one rule of every area
// (API.md §1, docs/design/array.md §2.4): truncate toward zero, saturate
// out-of-range values, nan -> 0. That is exactly what the ARM `FCVTZS`
// instruction does, so it is called directly. (A plain C++ cast is undefined
// behaviour for nan, inf and out-of-range values; the three branches that
// used to spell the rule out kept the conversion from vectorizing: 1.8x
// slower than numpy.) The compiler turns a loop of these into the vector FCVTZS.
#pragma once

#include <arm_neon.h>

#include <cstdint>

namespace sd {

inline std::int64_t saturate_to_int64(double x) { return vcvtd_s64_f64(x); }

}  // namespace sd
