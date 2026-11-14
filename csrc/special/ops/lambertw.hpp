// The Lambert W function, branches k = 0 and k = -1, real arguments
// (docs/api/special.md §11, docs/design/special.md §5.11; S7): W e^W = x.
//
//   - Near the branch point -1/e: the series in p = +-sqrt(2 (e x + 1))
//     (coeffs/lambertw_branch.hpp), with e x + 1 from e as a double-double and
//     an fma. In float64, e x + 1 cancels down to its last bits (scipy loses
//     2e7 ULP there); the condition number 1/|1 + W| of W is large there, but
//     the input is exact, so W keeps its relative accuracy.
//   - Elsewhere: Halley's iteration on w e^w = x from an initial value (the
//     branch series, log(1 + x), or the asymptotic L1 - L2 + L2/L1), to
//     2^-53 relative.
// The float64 nearest -1/e (-0.36787944117144233, 8e-18 beyond -1/e) is taken
// as the branch point itself: W = -1 on both branches (S7 user decision, as
// Boost). Below it, and x > 0 on k = -1, there is no real value: NaN.
#pragma once

#include "special/coeffs/lambertw_branch.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace lambertw_detail {

inline constexpr double kEHi = 2.718281828459045, kELo = 1.4456468917292502e-16;  // e = hi + lo
inline constexpr double kMinusInvE = -0.36787944117144233;  // the float64 nearest -1/e

// 2 (e x + 1), exact to rounding for x near -1/e: e_hi x + 1 by fma, then + e_lo x.
inline double two_ex_plus_one(double x) { return 2.0 * (std::fma(kEHi, x, 1.0) + kELo * x); }

// Halley on f(w) = w e^w - x from w.
inline double halley(double w, double x) {
    for (int it = 0; it < 64; ++it) {
        const double ew = std::exp(w);
        const double f = w * ew - x;
        const double wp1 = w + 1.0;
        const double d = ew * wp1 - (w + 2.0) * f / (2.0 * wp1);
        const double wn = w - f / d;
        if (std::fabs(wn - w) <= 0x1p-53 * std::fabs(wn) || !std::isfinite(wn)) return wn;
        w = wn;
    }
    return w;
}

inline double w0(double x) {
    if (std::isnan(x)) return kNaN;
    if (x == 0.0) return x;  // keeps -0.0
    if (x == kInf) return kInf;
    if (x == kMinusInvE) return -1.0;
    if (x < kMinusInvE) return kNaN;
    const double q = two_ex_plus_one(x);
    if (q <= 0.25) return horner(std::sqrt(q), kLambertwBranch);  // |p| <= 1/2
    if (std::fabs(x) < 0x1p-27) return x - x * x;  // W = x - x^2 + 3/2 x^3: the cube is below 2^-53 x
    double w;
    if (x < 3.0) w = std::log1p(x);  // W(x) ~ log(1 + x) to within 0.4 on [-0.35, 3]
    else {
        const double L1 = std::log(x), L2 = std::log(L1);
        w = L1 - L2 + L2 / L1;
    }
    if (x < 0.0) w = horner(std::sqrt(q), kLambertwBranch);  // still a good start up to |p| = 1.4
    return halley(w, x);
}

inline double wm1(double x) {
    if (std::isnan(x)) return kNaN;
    if (x == 0.0) return -kInf;
    if (x == kMinusInvE) return -1.0;
    if (x < kMinusInvE || x > 0.0) return kNaN;
    const double q = two_ex_plus_one(x);
    const double p = -std::sqrt(q);
    if (q <= 0.25) return horner(p, kLambertwBranch);
    double w;
    if (x > -0.25) {
        const double L1 = std::log(-x), L2 = std::log(-L1);
        w = L1 - L2 + L2 / L1;
    } else {
        w = horner(p, kLambertwBranch);
    }
    return halley(w, x);
}

}  // namespace lambertw_detail

struct Lambertw0 : Real {
    static constexpr const char* name = "lambertw";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kLambertw;
    static double f(double x) { return lambertw_detail::w0(x); }
};
struct LambertwM1 : Real {
    static constexpr const char* name = "lambertw";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kLambertwM1;
    static double f(double x) { return lambertw_detail::wm1(x); }
};

}  // namespace sd::special
