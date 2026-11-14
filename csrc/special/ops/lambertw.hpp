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

#include "special/boost.hpp"
#include "special/coeffs/lambertw_branch.hpp"
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

// The real branches as the public functions compute them: Boost (special/boost.hpp)
// away from the branch point, ours near it.
inline double w0_routed(double x) {
    if (x != 0.0 && std::isfinite(x) && std::fabs(x - kMinusInvE) >= 0.05) {
        const double r = bst::w0(x);
        if (bst::normal(r)) return r;
    }
    return w0(x);
}
inline double wm1_routed(double x) {
    if (x < 0.0 && std::fabs(x - kMinusInvE) >= 0.05) {
        const double r = bst::wm1(x);
        if (bst::normal(r)) return r;
    }
    return wm1(x);
}

// ---- complex argument, any branch k (S8) ---------------------------------------
using cplx = std::complex<double>;
inline constexpr double kTwoPi = 6.283185307179586477;

inline cplx cmul(cplx a, cplx b) { return ops::cmul(a, b); }
inline cplx cdiv(cplx a, cplx b) { return ops::cdiv(a, b); }
inline double norm1(cplx a) { return std::fabs(a.real()) + std::fabs(a.imag()); }
inline cplx clog(cplx z) { return {std::log(std::hypot(z.real(), z.imag())), std::atan2(z.imag(), z.real())}; }
inline cplx cexp(cplx z) {
    const double m = std::exp(z.real());
    return {m * std::cos(z.imag()), m * std::sin(z.imag())};
}
// The principal square root (Kahan): the sign of a zero imaginary part picks the side of the cut.
inline cplx csqrt(cplx z) {
    const double x = z.real(), y = z.imag();
    if (x == 0.0 && y == 0.0) return {0.0, y};
    const double t = std::sqrt(0.5 * (std::fabs(x) + std::hypot(x, y)));
    if (x >= 0.0) return {t, y / (2.0 * t)};
    return {std::fabs(y) / (2.0 * t), std::copysign(t, y)};
}
inline cplx chorner(cplx p) {
    cplx r = kLambertwBranch[0];
    for (std::size_t i = 1; i < kLambertwBranch.size(); ++i) r = cmul(r, p) + kLambertwBranch[i];
    return r;
}

// Halley on w e^w - z.
inline cplx challey(cplx w, cplx z) {
    for (int it = 0; it < 64; ++it) {
        const cplx ew = cexp(w);
        const cplx f = cmul(w, ew) - z;
        const cplx wp1 = w + 1.0;
        const cplx d = cmul(ew, wp1) - cdiv(cmul(w + 2.0, f), 2.0 * wp1);
        const cplx wn = w - cdiv(f, d);
        if (norm1(wn - w) <= 0x1p-53 * norm1(wn) || !std::isfinite(wn.real()) || !std::isfinite(wn.imag())) return wn;
        w = wn;
    }
    return w;
}
// For |z| beyond 1e150 or below 1e-150 (k != 0), where w e^w would overflow or
// underflow: Halley on w + log w - L, L = log z + 2 pi i k (|W| > 300 there, away
// from where the logarithm's branch could differ).
inline cplx challey_log(cplx w, cplx L) {
    for (int it = 0; it < 64; ++it) {
        const cplx iw = cdiv(1.0, w);
        const cplx g = w + clog(w) - L;
        const cplx g1 = 1.0 + iw, g2 = -cmul(iw, iw);
        const cplx wn = w - cdiv(g, g1 - cdiv(cmul(g, g2), 2.0 * g1));
        if (norm1(wn - w) <= 0x1p-53 * norm1(wn)) return wn;
        w = wn;
    }
    return w;
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
    const cplx L = clog(z) + cplx{0.0, kTwoPi * k};
    const double r = std::hypot(x, y);
    if (k == 0.0 && r < 1e-9) return z - cmul(z, z);  // the cube is below 2^-53 relative
    if (r > 1e150 || (k != 0.0 && r < 1e-150)) {
        const cplx L2 = clog(L);
        return challey_log(L - L2 + cdiv(L2, L), L);
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
        w = L - L2 + cdiv(L2, L);
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
