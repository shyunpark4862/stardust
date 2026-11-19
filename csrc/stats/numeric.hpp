// Shared numerics of the distributions (docs/design/stats.md §3).
//
// The recurring problem is the standardized variable z = (x - mu) / sigma:
// rounded to float64 it carries a relative error of an ULP, and a density
// like exp(-z^2 / 2) multiplies that by z^2 (z = 30: 900 ULP). So z is kept
// as a double-double from the exact inputs, exponents are double-doubles,
// and a cdf evaluated at the rounded z.hi is corrected to first order by
// f(z.hi) z.lo (the second-order term is ~z.lo^2: nothing).
#pragma once

#include "special/common.hpp"
#include "special/coeffs/cephes_gamma.hpp"
#include "special/common.hpp"
#include "special/dd.hpp"
#include "stats/coeffs/zeta_minus_one.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace sd::stats {

namespace num {

namespace dd = special::dd;

using dd::DD;

inline constexpr DD kLogSqrt2Pi = {0.9189385332046728, -3.8782941580672414e-17};  // log sqrt(2 pi)
inline constexpr DD kSqrt2Pi = {2.5066282746310007, -1.8328579980459167e-16};
inline constexpr DD kNormalEntropy = {1.4189385332046727, 7.223936088184323e-17};  // (1 + log 2 pi) / 2
inline constexpr DD kPi = {3.141592653589793, 1.2246467991473532e-16};
inline constexpr DD kInvPi = {0.3183098861837907, -1.9678676675182486e-17};
inline constexpr DD kLn2 = {0.6931471805599453, 2.3190468138462996e-17};
inline constexpr DD kLog4Pi = {2.5310242469692907, 5.664688743963382e-17};
inline constexpr DD kLogPi = {1.1447298858494002, 1.0265951162707826e-17};
inline constexpr DD kOnePlusLn2 = {1.6931471805599454, -8.783183432405266e-17};
inline constexpr double kInvSqrt2Pi = 0.3989422804014327;

// The last value of a parameter-only function, per thread: a distribution method
// called elementwise with scalar parameters (the common case) recomputes the same
// log-gammas and prefix factors for every element otherwise (t's pdf: 300 ns of which
// the parameters' part was most). Same bits either way.
struct Memo {
    double a = kNaNd, b = kNaNd;
    DD v{0.0, 0.0};
    template <class F>
    DD get(double x, double y, F&& f) {
        if (x == a && y == b) return v;
        v = f();
        a = x, b = y;
        return v;
    }
    static constexpr double kNaNd = std::numeric_limits<double>::quiet_NaN();
};

// exp(e) for e = hi + lo, one rounding: exp(hi) (1 + lo). Into the subnormals
// exp(hi) rounds once more, which the subnormal (absolute) rule allows.
// A low part that is not below 1 (hi past 2^53, where exp(hi) is 0 or inf already) or NaN (hi
// infinite) is not applied: a select, no branch.
inline double exp_dd(DD e) {
    const double lo = std::fabs(e.lo) < 1.0 ? e.lo : 0.0;
    return std::exp(e.hi) * (1.0 + lo);
}

// d / s as a double-double: q = d.hi / s, then the remainder d - q s exactly by fma. (For |d|
// below 2^-969 the remainder, 2^-53 of d, is rounded to a multiple of 2^-1074: the quotient
// then holds d only to 2^-1074 / |d|. A caller that multiplies the result's relative error
// handles such an operand itself: weibull_detail::log_ratio.)
inline DD div(DD d, double s) {
    const double q = d.hi / s;
    if (!std::isfinite(q) || q == 0.0) return {q, 0.0};
    const double r = std::fma(-q, s, d.hi) + d.lo;  // d - q s (exact product, one rounding of the sum)
    return dd::quick_two_sum(q, r / s);
}

// z = (x - mu) / sigma as a double-double, for finite x, mu and sigma > 0.
// x - mu may overflow while z does not (sigma > 1): then (x/2 - mu/2) / (sigma/2).
inline DD standardize(double x, double mu, double sigma) {
    const DD d = dd::two_sum(x, -mu);
    if (std::isfinite(d.hi)) return div(d, sigma);
    return div(dd::two_sum(0.5 * x, -0.5 * mu), 0.5 * sigma);  // the halves are exact (sigma > 2^-1021 or z overflows)
}

// x / s as a double-double (x finite, s > 0).
inline DD ratio(double x, double s) { return div(DD{x, 0.0}, s); }

// log(v) as a double-double for v > 0 finite (subnormal too): special's dd::log_pos.
inline DD log_dd(double v) { return dd::log_pos(v); }

// log(a b) as a double-double for a, b > 0 finite, the product unrounded.
inline DD log_prod(double a, double b) {
    const DD p = dd::two_prod(a, b);
    if (std::isfinite(p.hi) && p.hi >= 0x1p-1000) return dd::log(p);
    return log_dd(a) + log_dd(b);
}

// e^d - 1 for |d| < 1 in double-double (Taylor).
inline DD expm1_dd(DD d) {
    DD term = d, sum = d;
    for (int n = 2; n < 80; ++n) {
        term = term * d / static_cast<double>(n);
        sum = sum + term;
        if (std::fabs(term.hi) <= std::fabs(sum.hi) * 0x1p-108) break;
    }
    return sum;
}

// e^d as a double-double to ~1e-30: d = n log 2 + f, |f| <= 0.35, Taylor for e^f.
inline DD exp_full(DD d) {
    const double n = std::nearbyint(d.hi / kLn2.hi);
    const DD f = d - kLn2 * n;
    DD term{1.0, 0.0}, sum{1.0, 0.0};
    for (int k = 1; k < 60; ++k) {
        term = term * f / static_cast<double>(k);
        sum = sum + term;
        if (std::fabs(term.hi) <= 0x1p-110) break;
    }
    return dd::ldexp(sum, static_cast<int>(n));
}

// log Gamma(z) for z > 0 a double-double, to ~1e-19 absolute: special's dd::loggamma
// (the shift to z + n >= 30 and Stirling's form, the large terms in double-double).
inline DD log_gamma(DD z) { return dd::loggamma(z); }

inline constexpr DD kEulerGamma = {0.5772156649015329, -4.942915152430645e-18};

// log Gamma(1 + x) as a double-double: the series -euler x + sum_j>=2 (-1)^j zeta(j) x^j / j
// for |x| <= 1/2 (no 1 + x to round for a tiny x), else log_gamma of the exact 1 + x.
inline DD lgamma1p(double x) {
    if (std::fabs(x - 1.0) <= 0.25) {
        // about its zero at x = 1: log(1 + d) + log Gamma(1 + d), d = x - 1 (exact), each to
        // its own last bit (log_gamma's absolute 1e-19 is no part of a result near 0)
        const DD d{x - 1.0, 0.0};
        return (d - dd::log1pmx_neg(d)) + lgamma1p(d.hi);
    }
    if (std::fabs(x) > 0.5) return num::log_gamma(dd::two_sum(1.0, x));
    DD sum = kEulerGamma * -x, p{x, 0.0};
    for (int j = 2; j < 200; ++j) {
        p = p * x;
        DD zeta{1.0, 0.0};
        if (j <= kZetaMinusOneMax) zeta = zeta + DD{kZetaMinusOne[j - 2][0], kZetaMinusOne[j - 2][1]};
        DD term = zeta * p / static_cast<double>(j);
        if (j % 2 == 1) term = -term;
        sum = sum + term;
        if (std::fabs(term.hi) <= std::fabs(sum.hi) * 0x1p-108) break;
    }
    return sum;
}

// A root x from an inverse outside the guaranteed parameters is returned only if the forward
// log tail lf at it meets the target lt: within the tail's change over 8 ULP of x (its slope
// |x| f / F, from log_xf = log(|x| f(x))) and 1e-12 max(1, |lt|). NaN otherwise (and for a
// NaN lf). The slope is taken over the larger of the tail and the target: a tail far below
// the target (-inf for one that underflowed) must not widen its own tolerance.
inline double accept_root(double x, double lf, double log_xf, double lt) {
    const double tol = 0x1p-50 * std::exp(log_xf - std::max(lf, lt)) + 1e-12 * std::max(1.0, std::fabs(lt));
    return std::fabs(lf - lt) <= tol ? x : std::numeric_limits<double>::quiet_NaN();
}

// E = e^D - 1 in double-double.
inline DD expm1_full(DD d) {
    if (std::fabs(d.hi) < 0.5) return expm1_dd(d);
    return exp_full(d) - 1.0;
}


// ---- the entropies' log-gamma/digamma pieces ---------------------------------
//
// The entropies of the gamma, beta, t and F distributions are sums of
// log Gamma and (z - k) psi(z) terms of size z log z whose total is O(log z):
// for large parameters every bit of the terms is lost. They are regrouped into
//     G(z)  =  log Gamma(z) - (z - 1) psi(z) + z - log(z)/2 - log(2 pi)/2
//     Gc(z) = -log Gamma(z) + (z - 2) psi(z) - z + 3 log(z)/2 + log(2 pi)/2
// which are O(1) (with Stirling's series for log Gamma and psi:
// G = 1/2 - 1/(2z) + S + (z - 1) T, Gc = -1/2 + 1/z - S - (z - 2) T,
// S = sum B_2k z^(1-2k) / (2k (2k-1)), T = sum B_2k z^-2k / (2k)), plus the
// logs, which the callers sum in double-double. Below z = 12 the pieces come
// from log_gamma (double-double) and psi directly: no term is large there.

inline constexpr std::array<double, 8> kBernoulli2k = {1.0 / 6.0,  -1.0 / 30.0,     1.0 / 42.0, -1.0 / 30.0,
                                                       5.0 / 66.0, -691.0 / 2730.0, 7.0 / 6.0,  -3617.0 / 510.0};

// S(z), z T(z) and T(z) for z >= 12.
struct Stirling {
    double s, zt, t;
};
inline Stirling stirling(double z) {
    const double r = 1.0 / z, r2 = r * r;
    double s = 0.0, zt = 0.0, t = 0.0, p = r;  // p = z^(1-2k)
    for (int k = 1; k <= 8; ++k) {
        const double b = kBernoulli2k[k - 1];
        s += b * p / (2.0 * k * (2.0 * k - 1.0));
        zt += b * p / (2.0 * k);
        t += b * p * r / (2.0 * k);
        p *= r2;
    }
    return {s, zt, t};
}

inline double entropy_g(double z, double (*digamma)(double)) {
    if (z >= 12.0) {
        const Stirling st = stirling(z);
        return 0.5 - 0.5 / z + st.s + st.zt - st.t;
    }
    const DD v = log_gamma(DD{z, 0.0}) + z - log_dd(z) * 0.5 - kLogSqrt2Pi;
    return (v.hi + v.lo) - (z - 1.0) * digamma(z);
}

inline double entropy_gc(DD z, double (*digamma)(double)) {
    if (z.hi >= 12.0) {
        const Stirling st = stirling(z.hi);
        return -0.5 + 1.0 / z.hi - st.s - st.zt + 2.0 * st.t;
    }
    const DD v = -log_gamma(z) - z + dd::log(z) * 1.5 + kLogSqrt2Pi;
    return (v.hi + v.lo) + (z.hi - 2.0) * digamma(z.hi);
}

// psi(z + h) - psi(z) for z >= 12, h > 0 (by the asymptotic series, the large
// logs as log1p(h / z) and each power difference as z^-2k expm1(-2k log1p(h/z))).
inline double digamma_diff_asym(double z, double h) {
    const double l1 = std::log1p(h / z);
    double d = l1 - 0.5 / (z + h) + 0.5 / z;
    double p = 1.0;
    const double r2 = 1.0 / (z * z);
    for (int k = 1; k <= 8; ++k) {
        p *= r2;
        d -= kBernoulli2k[k - 1] / (2.0 * k) * p * std::expm1(-2.0 * k * l1);
    }
    return d;
}

}  // namespace num

}  // namespace sd::stats
