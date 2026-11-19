// The order-0 and order-1 Bessel functions (docs/api/special.md §8.1,
// docs/design/special.md §5.7; S6), Cephes (coeffs/cephes_bessel.hpp):
//
//     i0 i1 i0e i1e   Chebyshev series for e^-x I(x) on [0, 8] and for
//                     e^-x sqrt(x) I(x) in 32/x - 2 beyond
//     k0 k1 k0e k1e   K + log(x/2) I (x <= 2) and e^x sqrt(x) K in 8/x - 2
//     j0 j1 y0 y1     rationals up to 5 (J's first zeros factored out),
//                     Hankel's amplitude and phase rationals beyond
//
// Changed from Cephes:
//   - The Hankel phase x - pi/4 (3pi/4) is not rounded: cos(x - pi/4) =
//     (cos x + sin x) / sqrt 2 with libm's exactly reduced sin x, cos x.
//     Rounding x - 3pi/4 costs ulp(x)/2 of phase: scipy's y1 loses 932 ULP of
//     the envelope at x = 3828.
//   - e^x and e^-x as m 2^e: i0, i1 finite up to 713.98 (scipy inf from
//     713), k0, k1 subnormal tails kept.
//   - log(x/2) as log x - log 2: x/2 underflows at x = 5e-324 (scipy's
//     k0(5e-324) = inf; the value is 744.4).
//   - Infinite arguments give their limits (scipy NaN): i0(+-inf) = inf,
//     i1(+-inf) = +-inf, j0, j1, y0, y1 -> 0.
#pragma once

#include "special/boost.hpp"

#include "special/coeffs/cephes_bessel.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/gamma_core.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace bessel01_detail {

inline constexpr double kSqrtPi = 1.77245385090551602730;
inline constexpr double kTwoOverPi = 0.63661977236758134308;  // 2/pi

// Cephes chbevl: sum'_k c_k T_k(x/2), c[0] the highest.
template <std::size_t N>
inline double chbevl(double x, const std::array<double, N>& c) {
    double b0 = c[0], b1 = 0.0, b2 = 0.0;
    for (std::size_t i = 1; i < N; ++i) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c[i];
    }
    return 0.5 * (b0 - b2);
}

// polevl / p1evl (Cephes): the denominators with an implicit leading 1.
template <std::size_t N>
inline double p1evl(double x, const std::array<double, N>& c) {
    return horner1(x, c);
}

inline double log_half(double x) { return std::log(x) - kLn2Hi; }  // log(x / 2) without x / 2

// e^-x I0(x) for x >= 0, finite.
inline double i0e(double x) {
    if (x <= 8.0) return chbevl(0.5 * x - 2.0, cephes::kI0A);
    return chbevl(32.0 / x - 2.0, cephes::kI0B) / std::sqrt(x);
}
// e^-x I1(x) for x >= 0, finite.
inline double i1e(double x) {
    if (x <= 8.0) return chbevl(0.5 * x - 2.0, cephes::kI1A) * x;
    return chbevl(32.0 / x - 2.0, cephes::kI1B) / std::sqrt(x);
}
inline double i0(double x) { return x <= 8.0 ? std::exp(x) * i0e(x) : value(exp_scaled(x) * scaled(i0e(x))); }
inline double i1(double x) { return x <= 8.0 ? std::exp(x) * i1e(x) : value(exp_scaled(x) * scaled(i1e(x))); }

// e^x K(x) for x > 2, finite.
inline double k0e_large(double x) { return chbevl(8.0 / x - 2.0, cephes::kK0B) / std::sqrt(x); }
inline double k1e_large(double x) { return chbevl(8.0 / x - 2.0, cephes::kK1B) / std::sqrt(x); }
// K(x) for 0 < x <= 2.
inline double k0_small(double x) { return chbevl(x * x - 2.0, cephes::kK0A) - log_half(x) * i0(x); }
inline double k1_small(double x) { return log_half(x) * i1(x) + chbevl(x * x - 2.0, cephes::kK1A) / x; }

// Hankel's P and Q for x > 5 (z = 25 / x^2), and sin x, cos x.
struct Hankel {
    double p, q, w, s, c;  // w = 5 / x
};
inline Hankel hankel0(double x) {
    const double w = 5.0 / x, z = w * w;
    return {
        horner(z, cephes::kJ0Pp) / horner(z, cephes::kJ0Pq), horner(z, cephes::kJ0Qp) / p1evl(z, cephes::kJ0Qq), w,
        std::sin(x), std::cos(x)
    };
}
inline Hankel hankel1(double x) {
    const double w = 5.0 / x, z = w * w;
    return {
        horner(z, cephes::kJ1Pp) / horner(z, cephes::kJ1Pq), horner(z, cephes::kJ1Qp) / p1evl(z, cephes::kJ1Qq), w,
        std::sin(x), std::cos(x)
    };
}
// sqrt(2/pi)/sqrt(x) times p cos(chi) - w q sin(chi), with chi = x - pi/4 or x - 3pi/4 written out:
//   x - pi/4:  cos = (c + s)/sqrt2, sin = (s - c)/sqrt2   ->  1/sqrt(pi x) [p (c + s) - w q (s - c)]
//   x - 3pi/4: cos = (s - c)/sqrt2, sin = -(s + c)/sqrt2
inline double j0_large(double x) {
    const Hankel h = hankel0(x);
    return (h.p * (h.c + h.s) - h.w * h.q * (h.s - h.c)) / (kSqrtPi * std::sqrt(x));
}
inline double y0_large(double x) {
    const Hankel h = hankel0(x);
    return (h.p * (h.s - h.c) + h.w * h.q * (h.c + h.s)) / (kSqrtPi * std::sqrt(x));
}
inline double j1_large(double x) {
    const Hankel h = hankel1(x);
    return (h.p * (h.s - h.c) + h.w * h.q * (h.s + h.c)) / (kSqrtPi * std::sqrt(x));
}
inline double y1_large(double x) {
    const Hankel h = hankel1(x);
    return (-h.p * (h.s + h.c) + h.w * h.q * (h.s - h.c)) / (kSqrtPi * std::sqrt(x));
}

// J0, J1 for 0 <= x <= 5 (Cephes, zeros factored).
inline double j0_small(double x) {
    const double z = x * x;
    if (x < 1.0e-5) return 1.0 - z / 4.0;
    const double p = (z - cephes::kJ0Dr1) * (z - cephes::kJ0Dr2);
    return p * horner(z, cephes::kJ0Rp) / p1evl(z, cephes::kJ0Rq);
}
inline double j1_small(double x) {
    const double z = x * x;
    const double w = horner(z, cephes::kJ1Rp) / p1evl(z, cephes::kJ1Rq);
    // (next to the subnormals x last: w x would round there, 1e-13 of the result)
    if (x < 0x1p-900) return (w * (z - cephes::kJ1Z1) * (z - cephes::kJ1Z2)) * x;
    return w * x * (z - cephes::kJ1Z1) * (z - cephes::kJ1Z2);
}

inline double j0(double x) {
    x = std::fabs(x);
    if (std::isinf(x)) return 0.0;
    return x <= 5.0 ? j0_small(x) : j0_large(x);
}
inline double j1(double x) {
    const double a = std::fabs(x);
    double r;
    if (std::isinf(a)) r = 0.0;
    else r = a <= 5.0 ? j1_small(a) : j1_large(a);
    return std::signbit(x) ? -r : r;
}

}  // namespace bessel01_detail

#define SD_BESSEL01_OP(Struct, Name, Cost)                     \
    struct Struct : Real {                                     \
        static constexpr const char* name = Name;              \
        static constexpr int nin = 1, nout = 1;                \
        static constexpr std::array<const char*, 1> args{"x"}; \
        static constexpr double cost_ns = cost::Cost;          \
        static double f(double x);                             \
    };

SD_BESSEL01_OP(I0, "i0", kI0)
SD_BESSEL01_OP(I0e, "i0e", kI0e)
SD_BESSEL01_OP(I1, "i1", kI1)
SD_BESSEL01_OP(I1e, "i1e", kI1e)
SD_BESSEL01_OP(K0, "k0", kK0)
SD_BESSEL01_OP(K0e, "k0e", kK0e)
SD_BESSEL01_OP(K1, "k1", kK1)
SD_BESSEL01_OP(K1e, "k1e", kK1e)
SD_BESSEL01_OP(J0, "j0", kJ0)
SD_BESSEL01_OP(J1, "j1", kJ1)
SD_BESSEL01_OP(Y0, "y0", kY0)
SD_BESSEL01_OP(Y1, "y1", kY1)
#undef SD_BESSEL01_OP

// i0, i1, k0, k1: Boost (special/boost.hpp; 2-4.5x faster, as accurate), ours where its
// result overflows or leaves the normal range (the true values of i0, i1 near 713, k1(5e-324)).
inline double I0::f(double x) {
    if (std::isnan(x)) return kNaN;
    x = std::fabs(x);
    if (std::isinf(x)) return kInf;
    const double r = bst::i(0.0, x);
    return bst::normal(r) ? r : bessel01_detail::i0(x);
}
// The scaled forms: Boost's value times e^-+x for 0 < x < 700 (both factors in range there,
// one more rounding; bench/special/boost/routing.md), ours elsewhere.
inline double boost_scaled(double r, double x, double sign) {
    if (!bst::normal(r)) return kNaN;
    const double s = r * std::exp(sign * x);
    return bst::normal(s) ? s : kNaN;
}
inline double I0e::f(double x) {
    if (std::isnan(x)) return kNaN;
    x = std::fabs(x);
    if (x > 0.0 && x < 700.0) {
        const double r = boost_scaled(bst::i(0.0, x), x, -1.0);
        if (!std::isnan(r)) return r;
    }
    return std::isinf(x) ? 0.0 : bessel01_detail::i0e(x);
}
inline double I1::f(double x) {
    if (std::isnan(x)) return kNaN;
    const double a = std::fabs(x);
    double r = std::isinf(a) ? kInf : bst::i(1.0, a);
    if (!std::isinf(a) && !bst::normal(r)) r = bessel01_detail::i1(a);
    return x < 0.0 ? -r : r;  // scipy: i1(-0.0) = +0.0
}
inline double I1e::f(double x) {
    if (std::isnan(x)) return kNaN;
    const double a = std::fabs(x);
    double r = kNaN;
    if (a > 0.0 && a < 700.0) r = boost_scaled(bst::i(1.0, a), a, -1.0);
    if (std::isnan(r)) r = std::isinf(a) ? 0.0 : bessel01_detail::i1e(a);
    return x < 0.0 ? -r : r;
}
inline double K0::f(double x) {
    using namespace bessel01_detail;
    if (std::isnan(x) || x < 0.0) return kNaN;
    if (x == 0.0) return kInf;
    if (std::isinf(x)) return 0.0;
    const double r = bst::k(0.0, x);
    if (bst::normal(r)) return r;
    return x <= 2.0 ? k0_small(x) : value(exp_scaled(-x) * scaled(k0e_large(x)));
}
inline double K0e::f(double x) {
    using namespace bessel01_detail;
    if (std::isnan(x) || x < 0.0) return kNaN;
    if (x == 0.0) return kInf;
    if (std::isinf(x)) return 0.0;
    if (x < 700.0) {
        const double r = boost_scaled(bst::k(0.0, x), x, 1.0);
        if (!std::isnan(r)) return r;
    }
    return x <= 2.0 ? k0_small(x) * std::exp(x) : k0e_large(x);
}
inline double K1::f(double x) {
    using namespace bessel01_detail;
    if (std::isnan(x) || x < 0.0) return kNaN;
    if (x == 0.0) return kInf;
    if (std::isinf(x)) return 0.0;
    const double r = bst::k(1.0, x);
    if (bst::normal(r)) return r;
    return x <= 2.0 ? k1_small(x) : value(exp_scaled(-x) * scaled(k1e_large(x)));
}
inline double K1e::f(double x) {
    using namespace bessel01_detail;
    if (std::isnan(x) || x < 0.0) return kNaN;
    if (x == 0.0) return kInf;
    if (std::isinf(x)) return 0.0;
    if (x < 700.0) {
        const double r = boost_scaled(bst::k(1.0, x), x, 1.0);
        if (!std::isnan(r)) return r;
    }
    return x <= 2.0 ? k1_small(x) * std::exp(x) : k1e_large(x);
}
inline double J0::f(double x) { return std::isnan(x) ? kNaN : bessel01_detail::j0(x); }
inline double J1::f(double x) { return std::isnan(x) ? kNaN : bessel01_detail::j1(x); }
inline double Y0::f(double x) {
    using namespace bessel01_detail;
    if (std::isnan(x) || x < 0.0) return kNaN;
    if (x == 0.0) return -kInf;
    if (std::isinf(x)) return 0.0;
    if (x > 5.0) return y0_large(x);
    const double z = x * x;
    return horner(z, cephes::kJ0Yp) / p1evl(z, cephes::kJ0Yq) + kTwoOverPi * std::log(x) * j0_small(x);
}
inline double Y1::f(double x) {
    using namespace bessel01_detail;
    if (std::isnan(x) || x < 0.0) return kNaN;
    if (x == 0.0) return -kInf;
    if (std::isinf(x)) return 0.0;
    if (x > 5.0) return y1_large(x);
    const double z = x * x;
    return x * (horner(z, cephes::kJ1Yp) / p1evl(z, cephes::kJ1Yq)) +
           kTwoOverPi * (j1_small(x) * std::log(x) - 1.0 / x);
}

}  // namespace sd::special
