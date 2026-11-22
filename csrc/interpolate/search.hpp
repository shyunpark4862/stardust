// sd.interpolate's interval search and outside rule (docs/design/interpolate.md §2.3, §3.1).
//
// The interval of a point among strictly increasing breakpoints x[0 .. n) is the right-open
// x[i] <= v < x[i + 1], the last one closed; outside [x[0], x[n - 1]] it is the end interval (extended) or
// the point is NaN, and the periodic rule first wraps the point into the domain. The count of breakpoints at
// or below the point is base/interval.hpp's, the one the array's sd.interp uses: at a breakpoint every
// interval rule gives the same value (the kernels are exact at the ends), so all paths agree.
#pragma once

#include "array/kernels/scalar_ops.hpp"
#include "base/interval.hpp"

#include <algorithm>
#include <cstdint>

namespace sd::interpolate {

// extrapolate= (docs/design/interpolate.md §2.3): False, True, 'periodic'
enum class Extrapolate : int { Nan = 0, Extend = 1, Periodic = 2 };

// v wrapped into [x0, x1] for the periodic rule: x0 + (v - x0) % (x1 - x0), with numpy's remainder (scipy's
// PPoly and BSpline compute the same expression on numpy arrays)
inline double wrap_periodic(double v, double x0, double x1) { return x0 + ops::Remainder::f(v - x0, x1 - x0); }

// The interval of a point from its count c of breakpoints at or below it (n >= 2): c - 1 clamped to the
// first and the last interval.
inline std::int64_t interval_of(std::int64_t c, std::int64_t n) { return std::clamp<std::int64_t>(c - 1, 0, n - 2); }

}  // namespace sd::interpolate
