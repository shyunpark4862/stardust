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
inline double exp_dd(DD e) { return std::exp(e.hi) * (1.0 + e.lo); }

// d / s as a double-double: q = d.hi / s, then the remainder d - q s exactly by fma.
inline DD div(DD d, double s) {
    const double q = d.hi / s;
    if (!std::isfinite(q) || q == 0.0) return {q, 0.0};
    const double r = std::fma(-q, s, d.hi) + d.lo;  // d - q s (exact product, one rounding of the sum)
    return dd::quick_two_sum(q, r / s);
}

// z = (x - mu) / sigma as a double-double, for finite x, mu and sigma > 0.
// x - mu may overflow while z does not (sigma > 1): then x / sigma - mu / sigma.
inline DD standardize(double x, double mu, double sigma) {
    const DD d = dd::two_sum(x, -mu);
    if (std::isfinite(d.hi)) return div(d, sigma);
    return div(dd::two_sum(x / sigma, -mu / sigma), 1.0);
}

// x / s as a double-double (x finite, s > 0).
inline DD ratio(double x, double s) { return div(DD{x, 0.0}, s); }

// log(v) as a double-double for v > 0 finite (subnormal too).
inline DD log_dd(double v) {
    if (v >= 0x1p-1000) return dd::log(v);
    return dd::log(std::ldexp(v, 200)) - kLn2 * 200.0;
}

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

// log Gamma(z) for z > 0 a double-double, to ~1e-19 absolute: shifted to
// w = z + n >= 30 (log Gamma(z) = log Gamma(w) - log(z (z+1) ... (z+n-1)), the
// product and its log in double-double), then Stirling,
//     log Gamma(w) = w log w - w - log(w)/2 + log sqrt(2 pi) + S(w),
// with every large term in double-double and S(w) ~ 1/(12 w) (Cephes' series,
// special/coeffs/cephes_gamma.hpp) in double: its own error is ~1e-19.
inline DD log_gamma(DD z) {
    DD w = z, prod{1.0, 0.0};
    while (w.hi < 30.0) {
        prod = prod * w;
        w = w + 1.0;
    }
    const DD lw = dd::log(w);
    const double r = 1.0 / w.hi;
    const double series = special::horner(r * r, special::cephes::kLgamA) * r;
    DD l = w * lw - w - lw * 0.5 + kLogSqrt2Pi + series;
    if (prod.hi != 1.0 || prod.lo != 0.0) l = l - dd::log(prod);
    return l;
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

inline constexpr std::array<double, 8> kBernoulli2k = {1.0 / 6.0,  -1.0 / 30.0, 1.0 / 42.0,      -1.0 / 30.0,
                                                       5.0 / 66.0, -691.0 / 2730.0, 7.0 / 6.0, -3617.0 / 510.0};

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
