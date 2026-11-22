// Halley's method for the inverses of S4 (incgamma.hpp, incbeta.hpp), special/halley.hpp's
// iteration with one more way to end: a step that leaves the next one below rounding.
//
// halley() ends at a residual down to rounding (|F - t| <= 2^-50 F) or a step below 2^-54 x,
// which an evaluation at the converged point shows: from a start good to 1e-3 that is three
// evaluations of F (1e-3 -> 1e-9 -> 1e-27, then the one that sees it), from Boost's 10-digit
// start two. Here a Halley step d (not a log step, not a bisection) is taken as the last when
// the error it leaves, |C| d^3 with Halley's constant C = (h/2)^2 - F3/(6 F1) = h^2/12 - h1/6
// (h = F2/F1, h1 = h' and F3/F1 = h^2 + h1, Fk the k-th derivative), is below 2^-56 x with a
// factor 16 of margin, d is below 2^-16 x and small against the derivatives' own scale
// (d (|h| + sqrt|h1|) <= 2^-10, where the cubic term is the error): one evaluation fewer.
// eval(x) returns {F, x F1, h, h1}; the incomplete functions' h and h1 are closed forms of the
// parameters (h = (a - 1)/x - 1, h1 = -(a - 1)/x^2 for P(a, x)).
#pragma once

#include "special/common.hpp"

#include <algorithm>
#include <cmath>

namespace sd::special {

template <class Eval>
inline double halley_fast(double x, double t, bool inc, Eval&& eval, double lo = 0.0, double hi = kInf) {
    double step = kInf;
    for (int it = 0; it < 100; ++it) {
        const auto [F, xfp, h, h1] = eval(x);
        const double f = F - t;
        if (f == 0.0) return x;
        if ((f > 0.0) == inc) hi = x;
        else lo = x;
        double xn = kNaN;
        bool halley_step = false;
        if (xfp != 0.0 && std::isfinite(xfp) && F > 0.0) {
            if (F > 2.0 * t || F < 0.5 * t) {
                const double g = std::log(F / t);
                xn = inc ? x * std::exp(-g * (F / xfp)) : x - x * (g * (F / xfp));
            } else {
                const double u = x * (f / xfp);
                const double den = 1.0 - 0.5 * u * h;
                xn = x - (den > 0.5 && den < 2.0 ? u / den : u);
                halley_step = true;
            }
        }
        if (std::isfinite(xn)) {
            const double d = std::fabs(xn - x), ax = std::fabs(xn);
            if (std::fabs(f) <= 0x1p-50 * F || d <= 0x1p-54 * ax) return xn;
            const double c = h * h / 12.0 + std::fabs(h1) / 6.0;  // >= |C|
            if (halley_step && xn > lo && xn < hi && d <= 0x1p-16 * ax &&
                d * (std::fabs(h) + std::sqrt(std::fabs(h1))) <= 0x1p-10 && 16.0 * c * d * d * d <= 0x1p-56 * ax)
                return xn;
        }
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
    // (as halley(): a cycle in the last bits is the root; a bracket below the normals, 0)
    if (step <= 0x1p-48) return x;
    return hi <= 0x1p-1022 ? 0.0 : kNaN;
}

}  // namespace sd::special
