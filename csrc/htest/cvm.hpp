// The null distribution of the one-sample Cramér-von Mises statistic (docs/design/htest.md §3.3;
// H14: inside sd.htest): Csörgő and Faraway's (1996) finite-sample form
//
//     F_n(x) = V(x) (1 + 1/(12 n)) + psi1'(x) / n     for 1/(12 n) < x < n/3,
//
// 0 below and 1 above, V the limit distribution (their eq. 1.2) and psi1' their psi1 (eq. 1.10)
// without its V(x)/12 (scipy's _psi1_mod). Both are series of modified Bessel functions K of
// orders 1/4, 3/4 and 5/4 (special/ops/bessel.hpp's kve, the factor e^-z of each term taken
// with it: e^-z K(z) = kve(z) e^-2z), summed until a term is below 2^-60 of the sum (scipy
// stops at an absolute 1e-7). The p-value is max(0, 1 - F_n(x)), as scipy's: its error is about
// 1e-16 absolute, and the approximation itself goes below 0 in the far tail for small n.
#pragma once

#include "special/ops/bessel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sd::htest::cvm {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr int kMaxTerms = 10000;

// e^-z K_nu(z)
inline double ek(double nu, double z) { return special::Kve::f(nu, z) * std::exp(-2.0 * z); }

// V(x), the limit distribution (eq. 1.2, 1.3).
inline double cdf_inf(double x) {
    if (!(x > 0.0)) return x == x ? 0.0 : kNaN;
    double tot = 0.0;
    for (int k = 0; k < kMaxTerms; ++k) {
        const double y = 4.0 * k + 1.0;
        const double q = y * y / (16.0 * x);
        const double u = std::exp(std::lgamma(k + 0.5) - std::lgamma(k + 1.0)) / (std::pow(kPi, 1.5) * std::sqrt(x));
        const double t = u * std::sqrt(y) * ek(0.25, q);
        tot += t;
        if (t <= 0x1p-60 * tot) return tot;
    }
    return kNaN;
}

namespace detail {

inline double ed2(double y) {
    const double z = y * y / 4.0;
    return std::pow(y / 2.0, 1.5) * (ek(0.25, z) + ek(0.75, z)) / std::sqrt(kPi);
}
inline double ed3(double y) {
    const double z = y * y / 4.0;
    return std::pow(y / 2.0, 2.5) * (2.0 * ek(0.25, z) + 3.0 * ek(0.75, z) - ek(1.25, z)) / std::sqrt(kPi);
}

}  // namespace detail

// psi1'(x) = psi1(x) - V(x)/12 (eq. 1.10; scipy's _psi1_mod).
inline double psi1_mod(double x) {
    using detail::ed2, detail::ed3;
    const double sx = 2.0 * std::sqrt(x), y1 = std::pow(x, 0.75), y2 = std::pow(x, 1.25);
    double tot = 0.0;
    for (int k = 0; k < kMaxTerms; ++k) {
        const double kd = k;
        const double m = 2.0 * kd + 1.0;
        // Gamma(k + 1/2) / k! and Gamma(k + 3/2) / k!
        const double g1 = std::exp(std::lgamma(kd + 0.5) - std::lgamma(kd + 1.0)), g3 = (kd + 0.5) * g1;
        const double e1 = m * g1 * ed2((4.0 * kd + 3.0) / sx) / (9.0 * y1);
        const double e2 = g1 * ed3((4.0 * kd + 1.0) / sx) / (72.0 * y2);
        const double e3 = 2.0 * (m + 2.0) * g3 * ed3((4.0 * kd + 5.0) / sx) / (12.0 * y2);
        const double e4 = 7.0 * m * g1 * ed2((4.0 * kd + 1.0) / sx) / (144.0 * y1);
        const double e5 = 7.0 * m * g1 * ed2((4.0 * kd + 5.0) / sx) / (144.0 * y1);
        const double z = -(e1 + e2 + e3 + e4 + e5) / kPi;
        tot += z;
        if (k > 0 && std::fabs(z) <= 0x1p-60 * std::fabs(tot)) return tot;
    }
    return kNaN;
}

// The p-value P(W >= x) of the statistic of a sample of n by Csörgő and Faraway's form, at
// least 0.
inline double sf(double x, double n) {
    if (x != x || !(n > 0.0)) return kNaN;
    if (x <= 1.0 / (12.0 * n)) return 1.0;
    if (x >= n / 3.0) return 0.0;
    const double cdf = cdf_inf(x) * (1.0 + 1.0 / (12.0 * n)) + psi1_mod(x) / n;
    return std::max(0.0, 1.0 - cdf);
}

}  // namespace sd::htest::cvm
