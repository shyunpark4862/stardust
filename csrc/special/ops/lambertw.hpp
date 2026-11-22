// The Lambert W function, branches k = 0 and k = -1, real arguments
// (docs/api/special.md §11, docs/design/special.md §5.11; S7): W e^W = x.
//
//   - Near the branch point -1/e: the series in p = +-sqrt(2 (e x + 1))
//     (coeffs/lambertw_branch.hpp), with e x + 1 from e as a double-double and
//     an fma. In float64, e x + 1 cancels down to its last bits (scipy loses
//     2e7 ULP there); the condition number 1/|1 + W| of W is large there, but
//     the input is exact, so W keeps its relative accuracy.
//   - |x + 1/e| >= 0.05: Boost (special/boost.hpp; w0_routed, wm1_routed).
//   - Between the two, and where Boost's result is not normal: Halley's
//     iteration on w e^w = x from an initial value (the branch series,
//     log(1 + x), or the asymptotic L1 - L2 + L2/L1), to 2^-53 relative.
// The float64 nearest -1/e (-0.36787944117144233, 8e-18 beyond -1/e) is taken
// as the branch point itself: W = -1 on both branches (S7 user decision, as
// Boost). Below it, and x > 0 on k = -1, there is no real value: NaN.
#pragma once

#include "special/boost.hpp"
#include "special/coeffs/lambertw_branch.hpp"
#include "special/cplx.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"

#include <array>
#include <cmath>
#include <complex>

namespace sd::special {

namespace lambertw_detail {

inline constexpr double kEHi = 2.718281828459045, kELo = 1.4456468917292502e-16;  // e = hi + lo
inline constexpr double kMinusInvE = -0.36787944117144233;  // the float64 nearest -1/e

// 2 (e x + 1), exact to rounding for x near -1/e: e_hi x + 1 by fma, then + e_lo x.
inline double two_ex_plus_one(double x) { return 2.0 * (std::fma(kEHi, x, 1.0) + kELo * x); }

// Halley on f(w) = w e^w - x from w.
// (The loops here stop after a step below kDone of the iterate: Halley's error after a step s is
// ~K s^3, far below an ULP for s < 2^-26, so that step's result is the value -- before, they
// waited for a step below 2^-53, which the rounding of f near the root need never give: 11% of
// complex z in [-1, 3] x [-2, 2] cycled between neighbouring doubles for all 64 steps, 1.6 us.
// After 64 steps the iterate is the value if its last step was in its last bits, NaN otherwise.)
inline constexpr double kDone = 0x1p-26;
inline constexpr double kSettled = 0x1p-48;
inline double halley(double w, double x) {
    double step = kInf;
    for (int it = 0; it < 64; ++it) {
        const double ew = std::exp(w);
        const double f = w * ew - x;
        const double wp1 = w + 1.0;
        const double d = ew * wp1 - (w + 2.0) * f / (2.0 * wp1);
        const double wn = w - f / d;
        if (std::fabs(wn - w) <= kDone * std::fabs(wn) || !std::isfinite(wn)) return wn;
        step = std::fabs(wn - w) / std::fabs(wn);
        w = wn;
    }
    return step <= kSettled ? w : kNaN;
}

// W_-1 for a tiny negative x (a subnormal's w e^w has only a few bits: Halley on it does not
// settle): Halley on g(w) = w + log(-w) - L, L = log(-x), with g' = 1 + 1/w, g'' = -1/w^2.
inline constexpr double kLogFormAbove = -0x1p-1000;
inline double halley_log(double w, double L) {
    double step = kInf;
    for (int it = 0; it < 64; ++it) {
        const double iw = 1.0 / w;
        const double g = w + std::log(-w) - L;
        const double g1 = 1.0 + iw;
        const double wn = w - g / (g1 + g * iw * iw / (2.0 * g1));
        if (std::fabs(wn - w) <= kDone * std::fabs(wn) || !(wn < 0.0)) return wn < 0.0 ? wn : kNaN;
        step = std::fabs(wn - w) / std::fabs(wn);
        w = wn;
    }
    return step <= kSettled ? w : kNaN;
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
    if (x > kLogFormAbove) {
        const double L = std::log(-x), L2 = std::log(-L);
        return halley_log(L - L2 + L2 / L, L);
    }
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

// The real branches as the public functions compute them: Boost (special/boost.hpp)
// away from the branch point, ours near it.
// (ours out of line: the Boost call alone in the hot path, a sixth less time)
[[gnu::noinline]] inline double w0_ours(double x) { return w0(x); }
[[gnu::noinline]] inline double wm1_ours(double x) { return wm1(x); }
inline double w0_routed(double x) {
    if (x != 0.0 && std::isfinite(x) && std::fabs(x - kMinusInvE) >= 0.05) [[likely]] {
        const double r = bst::w0(x);
        if (bst::normal(r)) [[likely]]
            return r;
    }
    return w0_ours(x);
}
inline double wm1_routed(double x) {
    if (x <= kLogFormAbove && std::fabs(x - kMinusInvE) >= 0.05) [[likely]] {
        const double r = bst::wm1(x);
        if (bst::normal(r)) [[likely]]
            return r;
    }
    return wm1_ours(x);
}

// ---- complex argument, any branch k (S8) ---------------------------------------
using cplx = std::complex<double>;
using cx::cexp;
using cx::clog;
using cx::csqrt;
using cx::div;
using cx::kTwoPi;
using cx::mul;
using cx::norm1;

inline cplx chorner(cplx p) {
    cplx r = kLambertwBranch[0];
    for (std::size_t i = 1; i < kLambertwBranch.size(); ++i) r = mul(r, p) + kLambertwBranch[i];
    return r;
}

// Halley on w e^w - z.
inline cplx challey(cplx w, cplx z) {
    double step = kInf;
    for (int it = 0; it < 64; ++it) {
        const cplx ew = cexp(w);
        const cplx f = mul(w, ew) - z;
        const cplx wp1 = w + 1.0;
        const cplx d = mul(ew, wp1) - div(mul(w + 2.0, f), 2.0 * wp1);
        const cplx wn = w - div(f, d);
        if (norm1(wn - w) <= lambertw_detail::kDone * norm1(wn) || !std::isfinite(wn.real()) ||
            !std::isfinite(wn.imag()))
            return wn;
        step = norm1(wn - w) / norm1(wn);
        w = wn;
    }
    return step <= lambertw_detail::kSettled ? w : cplx{kNaN, kNaN};
}
// For |z| beyond 1e150 or below 1e-150 (k != 0), where w e^w would overflow or
// underflow: Halley on w + log w - L, L = log z + 2 pi i k (|W| > 300 there, away
// from where the logarithm's branch could differ).
inline cplx challey_log(cplx w, cplx L) {
    double step = kInf;
    for (int it = 0; it < 64; ++it) {
        const cplx iw = div(1.0, w);
        const cplx g = w + clog(w) - L;
        const cplx g1 = 1.0 + iw, g2 = -mul(iw, iw);
        const cplx wn = w - div(g, g1 - div(mul(g, g2), 2.0 * g1));
        if (norm1(wn - w) <= lambertw_detail::kDone * norm1(wn)) return wn;
        step = norm1(wn - w) / norm1(wn);
        w = wn;
    }
    return step <= lambertw_detail::kSettled ? w : cplx{kNaN, kNaN};
}

inline cplx cw(cplx z, double k) {
    const double x = z.real(), y = z.imag();
    if (std::isnan(x) || std::isnan(y)) return {kNaN, std::isnan(y) ? kNaN : 0.0};
    if (x == 0.0 && y == 0.0) return k == 0.0 ? z : cplx{-kInf, 0.0};
    if (std::isinf(x) || std::isinf(y)) return {kInf, std::atan2(y, x) + kTwoPi * k};  // log z + 2 pi i k
    const bool below = std::signbit(y);
    // the real branches on the real axis (exactly real): W_0 on [-1/e, inf), W_-1 from above
    // and W_1 from below on [-1/e, 0)
    if (y == 0.0 && x > kMinusInvE) {
        if (k == 0.0) return {w0_routed(x), y};
        if (x < 0.0 && ((k == -1.0 && !below) || (k == 1.0 && below))) return {wm1_routed(x), y};
    }
    // Next to the real branches' side of the cut (below it for k = 1, above for k = -1, on
    // (-1/e, 0)): the real value and its slope, W(x + iy) = W(x) + i y W / (x (1 + W)), exact to
    // (y/x)^2. (W itself is next to the negative axis there, where the iterations' log w would
    // change its branch from one step to the next.)
    if (x < 0.0 && x > kMinusInvE + 0.05 && std::fabs(y) <= 0x1p-30 * -x &&
        ((k == -1.0 && !below) || (k == 1.0 && below))) {
        const double wr = wm1_routed(x);
        return {wr, y * (wr / (x * (1.0 + wr)))};
    }
    const cplx L = clog(z) + cplx{0.0, kTwoPi * k};
    const double r = std::hypot(x, y);
    if (k == 0.0 && r < 1e-9) return z - mul(z, z);  // the cube is below 2^-53 relative
    if (r > 1e150 || (k != 0.0 && r < 1e-150)) {
        const cplx L2 = clog(L);
        return challey_log(L - L2 + div(L2, L), L);
    }
    // e z + 1 with e as a double-double: exact to rounding near -1/e
    const cplx ez1{std::fma(kEHi, x, 1.0) + kELo * x, kEHi * y + kELo * y};
    const bool nearbp = std::hypot(ez1.real(), ez1.imag()) < 0.3 * kEHi;  // |z + 1/e| < 0.3
    const bool branch = k == 0.0 || (k == -1.0 && !below) || (k == 1.0 && below);
    cplx w;
    if (nearbp && branch) {
        const cplx p = csqrt(2.0 * ez1);
        const cplx pk = k == 0.0 ? p : -p;  // k = -1 above, k = 1 below: the other sign
        if (norm1(2.0 * ez1) <= 0.25) return chorner(pk);  // |p| <= 1/2 (as the real case)
        w = chorner(pk);
    } else if (k == 0.0 && r < 3.0 && x > -0.5) {
        w = clog(1.0 + z);  // W(z) ~ log(1 + z) near 0
    } else {
        const cplx L2 = clog(L);
        w = L - L2 + div(L2, L);
    }
    return challey(w, z);
}

}  // namespace lambertw_detail

struct Lambertw0 : Real {
    static constexpr const char* name = "lambertw";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kLambertw;
    static double f(double x) { return lambertw_detail::w0_routed(x); }
};
struct LambertwM1 : Real {
    static constexpr const char* name = "lambertw";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kLambertwM1;
    static double f(double x) { return lambertw_detail::wm1_routed(x); }
};
// A complex argument, any branch: the second operand is k (an integer, passed as complex).
struct LambertwComplex {
    static constexpr const char* name = "lambertw";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"z", "k"};
    static constexpr double cost_ns = cost::kLambertwC;
    template <typename T> static constexpr bool ok = ops::is_cplx<T>;
    static std::complex<double> f(std::complex<double> z, std::complex<double> k) {
        return lambertw_detail::cw(z, k.real());
    }
};

}  // namespace sd::special
