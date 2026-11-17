// Halley's method for the inverses of S4 (docs/design/special.md §5.5).
#pragma once

#include "special/common.hpp"

#include <algorithm>
#include <cmath>

namespace sd::special {

// Halley's method for F(x) = t, F increasing (inc) or decreasing in x in (lo, hi),
// from x0. `eval(x)` returns {F(x), x F'(x), F''(x) / F'(x)} (x F' rather than F':
// it stays finite where F' overflows, x subnormal). While F is more than a
// factor 2 from t the step is Newton's on log F (in log x when F increases). Every x seen
// narrows a bracket [lo, hi]; a step that leaves it is replaced by bisection
// (geometric while the bracket spans more than a factor 4). Stops when the
// residual is down to rounding (|F - t| <= 2^-50 F; the step from there is
// final) or the step is below 2^-54 x. The result is accurate to the forward
// error times F / (x F') (docs/api/special.md §3). At most 100 steps: NaN if they do not settle.
template <class Eval>
inline double halley(double x, double t, bool inc, Eval&& eval, double lo = 0.0, double hi = kInf) {
    double step = kInf;  // the last step taken, relative
    for (int it = 0; it < 100; ++it) {
        const auto [F, xfp, h] = eval(x);
        const double f = F - t;
        if (f == 0.0) return x;
        if ((f > 0.0) == inc) hi = x;
        else lo = x;
        double xn = kNaN;
        if (xfp != 0.0 && std::isfinite(xfp) && F > 0.0) {
            if (F > 2.0 * t || F < 0.5 * t) {
                // Far from t a linear step moves x by about F / F'. Newton on log F
                // instead: in log x for an increasing F (a lower tail is a power
                // of x, F ~ x^a), in x for a decreasing one (an upper tail is an
                // exponential, F ~ e^-x).
                const double g = std::log(F / t);
                xn = inc ? x * std::exp(-g * (F / xfp)) : x - x * (g * (F / xfp));
            } else {
                const double u = x * (f / xfp);
                const double den = 1.0 - 0.5 * u * h;
                xn = x - (den > 0.5 && den < 2.0 ? u / den : u);
            }
        }
        // Converged before the bracket test: a final step below the resolution of
        // x gives xn == x, which is also the bracket's end just set.
        if (std::isfinite(xn) && (std::fabs(f) <= 0x1p-50 * F || std::fabs(xn - x) <= 0x1p-54 * std::fabs(xn)))
            return xn;
        if (!(xn > lo && xn < hi)) {
            if (std::isinf(hi)) xn = std::max(2.0 * x, 1.0);
            else if (lo > 0.0 && hi > 4.0 * lo) xn = std::sqrt(lo) * std::sqrt(hi);
            else if (lo == 0.0 && hi > 1e-300) xn = hi * 0x1p-20;
            else xn = 0.5 * (lo + hi);
        }
        if (xn == x || (std::isfinite(hi) && hi - lo <= 0x1p-53 * hi)) return xn;
        step = std::fabs(xn - x) / std::fabs(xn);
        x = xn;
    }
    // 100 steps without the tests above passing: the iterate if it only moves in its last
    // bits (a cycle between neighbouring doubles), else no value
    if (step <= 0x1p-48) return x;
    // (a bracket that has fallen below the normal doubles: the root underflows, 0 by the
    // inverses' convention)
    return hi <= 0x1p-1022 ? 0.0 : kNaN;
}

}  // namespace sd::special
