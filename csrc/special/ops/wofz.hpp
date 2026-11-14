// The Faddeeva function w(z) = exp(-z^2) erfc(-iz), and erf, erfc, erfcx of
// a complex argument through it (docs/api/special.md §12, docs/design/special.md
// §5.13; S8).
//
// w in the upper half plane, Re z >= 0 (Re z < 0 by w(-conj z) = conj w(z)):
//   - 0 <= Re z <= 7, 0 <= Im z <= 4: the Taylor series about the nearest point
//     z0 of a 0.1 grid. The table (coeffs/wofz_grid.hpp) holds w(z0) as a
//     double-double; the coefficients follow from it by the recurrence of
//     w' = -2zw + 2i/sqrt(pi):  a_{n+1} = -(2 z0 a_n + 2 a_{n-1}) / (n + 1).
//     |z - z0| <= 0.0707, so |2 z0 (z - z0)| <= 1.14 and the terms fall like
//     1.14^n / n!: about 20 of them.
//   - elsewhere: the Laplace continued fraction
//         w(z) = (i/sqrt(pi)) / (z - (1/2) / (z - 1 / (z - (3/2) / (z - ...)))),
//     with the number of terms of Poppe and Wijers as fitted by S. G. Johnson
//     (Faddeeva package, MIT), plus a margin. Near the real axis the fraction
//     misses the part exp(-x^2), which beyond Re z = 7 is below 1e-21 of |w|.
// The lower half plane: w(z) = 2 exp(-z^2) - w(-z). Zeros of w lie there, so the
// contract there is the envelope |error| <= 1e-14 (2|exp(-z^2)| + |w(-z)|)
// (S8 user decision).
//
// exp(-z^2): -z^2 = (y - x)(y + x) - 2ixy formed exactly as double-doubles, the
// phase's two parts through libm's sin and cos (exact argument reduction).
// scipy (Johnson's code) rounds z^2 and loses up to 450 ULP there.
#pragma once

#include "special/coeffs/wofz_grid.hpp"
#include "special/cplx.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"
#include "special/ops/erf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>

namespace sd::special {

using cplx = std::complex<double>;

namespace wofz_detail {

inline constexpr double kInvSqrtPi = 0.56418958354775628695;  // 1/sqrt(pi)
inline constexpr double kTwoInvSqrtPi = 1.1283791670955125739;  // 2/sqrt(pi)

using cx::div;
using cx::mul;
using cx::norm1;

// exp(-z^2) as m 2^e (|m| < 1); the phase is NaN once 2xy overflows.
struct CScaled {
    cplx m;
    std::int64_t e;
};
inline cplx value(CScaled s) {
    if (s.m == cplx{0.0, 0.0}) return s.m;
    const int e = static_cast<int>(std::clamp<std::int64_t>(s.e, -2200, 2200));
    return {std::ldexp(s.m.real(), e), std::ldexp(s.m.imag(), e)};
}
inline CScaled exp_negsq(double x, double y) {
    const double rd = (y - x) * (y + x);  // y^2 - x^2, to decide the range first
    if (rd < -1e4) return {{0.0, 0.0}, 0};  // below every subnormal whatever the phase
    const dd::DD p = dd::two_prod(x, y);  // NaN low part once xy overflows: the phase is NaN
    const double th = -2.0 * p.hi, tl = -2.0 * p.lo;  // the phase -2xy = th + tl
    double c = std::cos(th), s = std::sin(th);
    if (tl != 0.0) {
        const double cl = std::cos(tl), sl = std::sin(tl);
        const double c2 = c * cl - s * sl;
        s = s * cl + c * sl;
        c = c2;
    }
    if (rd > 1e4) return {{c, s}, 1000000};  // beyond every double: value() saturates (0 stays 0)
    const dd::DD re = dd::two_sum(y, -x) * dd::two_sum(y, x);  // |y^2 - x^2| <= 1e4: no overflow
    const Scaled mag = exp_scaled(re.hi);
    const double m = std::fma(mag.m, re.lo, mag.m);  // e^(hi + lo) = e^hi (1 + lo), |lo| < 1e-12
    return {{m * c, m * s}, mag.e};
}

// The Taylor series about the grid point; 0 <= x <= 7, 0 <= y <= 4.
inline cplx w_grid(double x, double y) {
    const int i = static_cast<int>(std::nearbyint(x * 10.0));
    const int j = static_cast<int>(std::nearbyint(y * 10.0));
    const double x0 = i * kWofzStep, y0 = j * kWofzStep;
    const cplx h{x - x0, y - y0};  // exact (Sterbenz, or x0 = 0)
    const auto& g = kWofzGrid[static_cast<std::size_t>(i * kWofzNy + j)];
    const cplx z0{x0, y0}, a0{g[0], g[2]};
    cplx prev = a0;
    cplx a = cplx{0.0, kTwoInvSqrtPi} - 2.0 * mul(z0, a0);  // a_1
    cplx hn = h, sum = mul(a, h);
    const double tiny = 0x1p-56 * norm1(a0);
    for (int n = 1; n < 60; ++n) {
        const cplx next = -(2.0 * mul(z0, a) + 2.0 * prev) / static_cast<double>(n + 1);
        prev = a;
        a = next;
        hn = mul(hn, h);
        const cplx t = mul(a, hn);
        sum += t;
        if (norm1(t) <= tiny) break;
    }
    return {g[0] + (g[1] + sum.real()), g[2] + (g[3] + sum.imag())};
}

// Terms of the continued fraction beyond Johnson's fit (measured in S8: 76 ULP without, 2.8 with 4).
inline constexpr int kWofzCfMargin = 4;

// The continued fraction; x >= 0, y >= 0, outside the grid.
inline cplx w_cf(double x, double y) {
    const cplx z{x, y}, iinvsqrtpi{0.0, kInvSqrtPi};
    if (x + y > 3e8) return div(iinvsqrtpi, z);  // the next term, 1/(2z^2), is below 2^-56 relative
    const int nu = static_cast<int>(3.9 + 11.398 / (0.08254 * x + 0.1421 * y + 0.2023)) + kWofzCfMargin;
    cplx t = z;
    for (int k = nu - 1; k >= 1; --k) t = z - cx::rdiv(0.5 * k, t);
    return div(iinvsqrtpi, t);
}

inline cplx w_upper(double x, double y) {  // x >= 0, y >= 0, finite
    if (x <= 7.0 && y <= 4.0) return w_grid(x, y);
    return w_cf(x, y);
}

inline cplx wofz(cplx z) {
    const double x = z.real(), y = z.imag();
    if (x == 0.0) return {Erfcx::f(y), x};  // w(iy) = erfcx(y), real (as scipy, the sign of Re z kept)
    if (std::isnan(x) || std::isnan(y)) return {kNaN, kNaN};
    if (!(y < 0.0)) {
        if (std::isinf(x) || std::isinf(y)) return {0.0, std::isinf(y) ? 0.0 : std::copysign(0.0, x)};
        if (x > 0.0) return w_upper(x, y);
        return std::conj(w_upper(-x, y));
    }
    if (std::isinf(y)) return {kNaN, kNaN};  // |w| grows without bound and the phase turns
    if (std::isinf(x)) return {0.0, std::copysign(0.0, x)};
    // the lower half plane: 2 exp(-z^2) - w(-z)
    const cplx e = value(exp_negsq(x, y));
    const cplx wm = x > 0.0 ? std::conj(w_upper(x, -y)) : w_upper(-x, -y);  // w(-z), -z = (-x, -y)
    return 2.0 * e - wm;
}

// erfcx(z) = w(iz)
inline cplx erfcx(cplx z) { return wofz({-z.imag(), z.real()}); }

// erfc(z) = exp(-z^2) w(iz) for Re z >= 0 (iz in the upper half plane: relative accuracy);
// 2 - erfc(-z) for Re z < 0, where the zeros are (the envelope 2 + |erfc(-z)|). As m 2^e.
inline CScaled erfc_right(double x, double y) {  // x >= 0
    const CScaled e = exp_negsq(x, y);
    const cplx w = wofz({-y, x});
    const cplx m = mul(e.m, w);
    return {m, e.e};
}
inline cplx erf(cplx z);
inline cplx erfc(cplx z) {
    const double x = z.real(), y = z.imag();
    if (y == 0.0) {  // the real axis: the real erfc, Im exactly -y (as scipy)
        double r;
        Erfc::chunk(&x, &r, 1);
        return {r, -y};
    }
    if (x == 0.0) return {1.0, -erf(z).imag()};  // erfc(iy) = 1 - i erfi(y)
    if (std::isnan(x) || std::isnan(y)) return {kNaN, kNaN};
    if (std::isinf(x) && !std::isinf(y)) return {x > 0 ? 0.0 : 2.0, 0.0};
    if (!std::signbit(x)) return value(erfc_right(x, y));
    const cplx r = value(erfc_right(-x, -y));
    return {2.0 - r.real(), -r.imag()};
}

// erf: the Taylor series 2/sqrt(pi) sum (-1)^n z^(2n+1) / (n! (2n+1)) for |z| < 1/2;
// otherwise 1 - erfc(z) for Re z >= 0 and -erf(-z) for Re z < 0. The zeros of erf
// lie beyond |z| = 2.3; from |z| = 2 the contract is the envelope 1 + |erfc(+-z)|.
inline cplx erf_taylor(cplx z) {
    const cplx z2 = mul(z, z);
    cplx t = z, s = z;
    for (int n = 1; n < 40; ++n) {
        t = -mul(t, z2) / static_cast<double>(n);
        const cplx u = t / static_cast<double>(2 * n + 1);
        s += u;
        if (norm1(u) <= 0x1p-56 * norm1(s)) break;
    }
    return kTwoInvSqrtPi * s;
}
inline cplx erf(cplx z) {
    const double x = z.real(), y = z.imag();
    if (y == 0.0) {  // the real axis: the real erf, Im exactly y (as scipy)
        double r;
        Erf::chunk(&x, &r, 1);
        return {r, y};
    }
    if (x == 0.0) {  // the imaginary axis: erf(iy) = i erfi(y), Re exactly x
        if (std::isnan(y) || std::isinf(y)) return {x, y};
        return {x, erf(cplx{0x1p-1074, y}).imag()};  // the general path; its Re is dropped
    }
    if (std::isnan(x) || std::isnan(y)) return {kNaN, kNaN};
    if (std::isinf(x) && !std::isinf(y)) return {std::copysign(1.0, x), 0.0};
    if (x * x + y * y < 0.25) return erf_taylor(z);
    if (std::signbit(x)) {
        const cplx r = erf(-z);
        return {-r.real(), -r.imag()};
    }
    const CScaled c = erfc_right(x, y);
    if (c.m == cplx{0.0, 0.0}) return {1.0, 0.0};
    if (std::isnan(c.m.real()) && (y - x) * (y + x) <= 0.0)  // 2xy overflowed: |z| > 1e154, so
        return {1.0, 0.0};                                    // |erfc| <= 1/(sqrt(pi)|z|): erf = 1
    if (c.e < -60) {  // erfc below 2^-60 of 1: erf = 1 (the phase may be unknown there)
        const double im = -value(c).imag();
        return {1.0, std::isnan(im) ? 0.0 : im};
    }
    const cplx r = value(c);
    return {1.0 - r.real(), -r.imag()};
}

}  // namespace wofz_detail

struct Wofz {
    static constexpr const char* name = "wofz";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kWofz;
    template <typename T> static constexpr bool ok = ops::is_cplx<T>;
    static cplx f(cplx z) { return wofz_detail::wofz(z); }
};

// erf, erfc, erfcx of a complex argument (the real ones are in ops/erf.hpp).
struct ErfComplex {
    static constexpr const char* name = "erf";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kErfC;
    template <typename T> static constexpr bool ok = ops::is_cplx<T>;
    static cplx f(cplx z) { return wofz_detail::erf(z); }
};
struct ErfcComplex {
    static constexpr const char* name = "erfc";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kErfcC;
    template <typename T> static constexpr bool ok = ops::is_cplx<T>;
    static cplx f(cplx z) { return wofz_detail::erfc(z); }
};
struct ErfcxComplex {
    static constexpr const char* name = "erfcx";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kErfcxC;
    template <typename T> static constexpr bool ok = ops::is_cplx<T>;
    static cplx f(cplx z) { return wofz_detail::erfcx(z); }
};

}  // namespace sd::special
