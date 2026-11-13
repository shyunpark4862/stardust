// The Riemann and Hurwitz zeta functions (docs/api/special.md §7,
// docs/design/special.md §5.6; S3). Hurwitz: Cephes zeta.c (Euler-Maclaurin:
// the first terms summed, then the tail as an integral and Bernoulli
// corrections). Riemann: the same with q = 1 for x >= 0, the functional
// equation for x < 0.
#pragma once

#include "special/coeffs/cephes_gamma.hpp"
#include "special/cost.hpp"
#include "special/gamma_core.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace zeta_detail {

// sum_k (k + q)^-s by Euler-Maclaurin (Cephes zeta.c), for s != 1. sm1 is
// s - 1 given exactly by the caller: the integral of the tail is
// (q + N)^(1-s) / (s - 1), and near s = 1 a rounded s - 1 would be the whole
// error. Cephes stops the direct sum once a term is negligible; for s < 1
// (Riemann only) the terms grow, and the sum runs to N = 9 before the
// corrections continue it analytically.
inline double euler_maclaurin(double s, double sm1, double q) {
    double sum = std::pow(q, -s);
    double a = q, b = 0.0;
    int i = 0;
    while (i < 9 || a <= 9.0) {
        ++i;
        a += 1.0;
        b = std::pow(a, -s);
        sum += b;
        if (std::fabs(b / sum) < kEps && std::fabs(b * a / (sm1 * sum)) < kEps) return sum;
    }
    const double w = a;
    sum += b * w / sm1;
    sum -= 0.5 * b;
    double fac = 1.0, k = 0.0;
    for (int j = 0; j < 12; ++j) {
        fac *= s + k;
        b /= w;
        const double t = fac * b / cephes::kZetaA[static_cast<std::size_t>(j)];
        sum += t;
        if (std::fabs(t / sum) < kEps) break;
        k += 1.0;
        fac *= s + k;
        b /= w;
        k += 1.0;
    }
    return sum;
}

// Hurwitz zeta(s, q), s >= 1 (s = 1: inf), with Cephes' conventions: NaN for
// s < 1; inf at q = 0, -1, -2, ...; NaN for q < 0 when s is not an integer
// (q^-s is complex); q > 1e8 by the first terms of the asymptotic series.
// s = inf is the limit (S3 user decision: zeta(inf, q > 1) = 0; scipy NaN).
inline double hurwitz(double s, double sm1, double q) {
    if (std::isnan(s) || std::isnan(q)) return kNaN;
    if (sm1 == 0.0) return kInf;  // s - 1, not s: 1 + 1e-323 rounds to 1
    if (sm1 < 0.0) return kNaN;
    if (q <= 0.0) {
        if (q == std::floor(q)) return kInf;
        if (s != std::floor(s)) return kNaN;
    }
    if (s == kInf) {
        if (q > 1.0) return 0.0;
        if (q == 1.0) return 1.0;
        return kInf;
    }
    if (q > 1e8) {  // 1/(s-1) + 1/(2q) + s/(12 q^2), without overflowing q^2
        const double sq = s / q;
        return (1.0 / sm1 + (6.0 + sq) / q / 12.0) * std::pow(q, -sm1);
    }
    return euler_maclaurin(s, sm1, q);
}

// zeta(s, q) as m 2^e, for q > 0 and s > 1 where q^-s may underflow
// (polygamma's n! zeta(n + 1, x) for large n and x): every term is taken
// relative to the first, q^-s, from pow_scaled, so the sum is an ordinary
// number and q^-s is applied once, scaled.
inline Scaled hurwitz_scaled(double s, double sm1, double q) {
    if (q <= 0.0 || q > 1e8 || s * std::log2(q) < 900.0) return scaled(hurwitz(s, sm1, q));
    const Scaled first = pow_scaled(q, -s);
    auto rel = [&](double a) { return value(pow_scaled(a, -s) / first); };  // (a/q)^-s, <= 1
    double sum = 1.0, a = q, b = 0.0;
    int i = 0;
    while (i < 9) {
        ++i;
        a += 1.0;
        b = rel(a);
        sum += b;
        if (std::fabs(b / sum) < kEps && std::fabs(b * a / (sm1 * sum)) < kEps) return first * scaled(sum);
    }
    const double w = a;
    sum += b * w / sm1;
    sum -= 0.5 * b;
    double fac = 1.0, k = 0.0;
    for (int j = 0; j < 12; ++j) {
        fac *= s + k;
        b /= w;
        const double t = fac * b / cephes::kZetaA[static_cast<std::size_t>(j)];
        sum += t;
        if (std::fabs(t / sum) < kEps) break;
        k += 1.0;
        fac *= s + k;
        b /= w;
        k += 1.0;
    }
    return first * scaled(sum);
}

}  // namespace zeta_detail

// Riemann zeta(x):
//     x > 1, 0 <= x < 1:  Euler-Maclaurin with q = 1
//     x < 0:  zeta(x) = (2 pi)^x / pi  sinpi(x/2)  Gamma(1 - x)  zeta(1 - x)
//             with y = -x exact: Gamma(1 - x) = y Gamma(y), zeta(1 - x) from
//             s - 1 = y; (2 pi)^x as exp of x log(2 pi) carried in two parts;
//             the product scaled, so it overflows only where zeta does
//             (beyond x = -260). The trivial zeros (even negative integers) are 0.
struct Zeta : Real {
    static constexpr const char* name = "zeta";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kZeta;
    static double f(double x) {
        if (std::isnan(x)) return x;
        if (x == 1.0) return kInf;
        if (x == kInf) return 1.0;
        if (x == -kInf) return kNaN;
        if (std::fabs(x) < 0x1p-30) return -0.5 - kLogSqrt2Pi * x;  // zeta(x) = -1/2 - log(2 pi) x / 2 + O(x^2)
        if (x >= 0.0) return zeta_detail::euler_maclaurin(x, x - 1.0, 1.0);
        const double s = sinpi(0.5 * x);
        if (s == 0.0) return 0.0;
        const double y = -x;
        if (y > 400.0) return std::copysign(kInf, s);
        const double p = x * kLog2PiHi;
        const double perr = std::fma(x, kLog2PiHi, -p) + x * kLog2PiLo;
        Scaled r = exp_scaled(p) * scaled(1.0 + perr);
        r = r * scaled(s * zeta_detail::hurwitz(1.0 + y, y, 1.0) / kPi);
        r = r * scaled(y) * gamma_scaled(y);
        return value(r);
    }
};

// Hurwitz zeta(x, q) (zeta's two-argument form, docs/api/special.md §7).
struct Hurwitz : Real {
    static constexpr const char* name = "zeta";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"x", "q"};
    static constexpr double cost_ns = cost::kHurwitz;
    static double f(double x, double q) { return zeta_detail::hurwitz(x, x - 1.0, q); }
};

}  // namespace sd::special
