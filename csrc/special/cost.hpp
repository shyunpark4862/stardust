// Per-element cost of each special function, for the parallel split
// (docs/design/special.md §3.3).
//
// The array's iterator goes parallel when the work reaches a threshold and
// cuts it into tasks of a fixed size, both counted in units of one add per
// element (array/parallel_limits.hpp). Special functions cost from a few to
// hundreds of adds per element, so each passes its weight -- its cost in
// adds -- and an expensive function splits at a smaller size, into tasks of
// fewer elements, the same way a lane kernel does (docs/design/array.md §4.5).
//
// The costs are serial times per element on this machine (Apple M4 Pro, one
// thread, inputs spread over the function's guaranteed region), measured by
// bench/special/cost.py. A factor of two either way changes only where the
// split starts, never a result.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sd::special {

// add on contiguous float64, per element (bench/special/cost.py).
inline constexpr double kAddNs = 0.19;  // measured 0.185 (2026-11-13)

// The weight NdIter::run takes: the cost in adds, at least 1.
constexpr std::int64_t weight_of(double ns) {
    const double w = ns / kAddNs;
    return w < 1.0 ? 1 : static_cast<std::int64_t>(w + 0.5);
}

namespace cost {
// The engine's test functions (special/ops/testing.hpp).
inline constexpr double kFma3 = 0.24;
inline constexpr double kSum4 = 0.30;
inline constexpr double kPowers4 = 0.45;
inline constexpr double kCfTan = 16;
// S1 (special/ops/logistic.hpp), bench/special/s1/cost.md
inline constexpr double kExpit = 1.3;
inline constexpr double kLogit = 3.5;
inline constexpr double kLogExpit = 14.5;
inline constexpr double kXlogy = 3.5;
inline constexpr double kXlog1py = 4.8;
inline constexpr double kEntr = 2.9;
inline constexpr double kRelEntr = 10.8;
}  // namespace cost

}  // namespace sd::special
