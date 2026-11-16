// The Weibull distribution with shape c and scale s (scipy's weibull_min; T18):
// P(X > x) = exp(-t), t = (x/s)^c. t = exp(c log(x/s)) with the exponent a
// double-double (exp multiplies its rounding by |c log(x/s)|), and the
// inverses x = s exp(log(t) / c) the same way.
//
// The moments are m_k = s^k Gamma(1 + k/c). As c grows X concentrates
// (X ~ s (1 + log E / c)) and the central moments cancel: the variance
// Gamma(1 + 2x) - Gamma(1 + x)^2 (x = 1/c) is O(x^2) out of terms of size 1.
// So they are written through
//     D_k = log Gamma(1 + k x) - k log Gamma(1 + x)
//         = sum_{j>=2} (-1)^j zeta(j) / j ((k x)^j - k x^j)       (k x < 1)
// in which the linear terms cancel exactly, and with E_k = e^(D_k) - 1:
//     mu_2 / m_1^2 = E_2,   mu_3 / m_1^3 = E_3 - 3 E_2,   mu_4 / m_1^4 = E_4 - 4 E_3 + 6 E_2,
// all in double-double (the skewness crosses 0 near c = 3.6, the excess
// kurtosis near c = 2.25 and 5.77). Where k x is too large for the series,
// D_k comes from gammaln directly.
#pragma once

#include "special/ops/gamma.hpp"
#include "stats/coeffs/zeta_minus_one.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace weibull_detail {

using num::DD;
namespace dd = num::dd;

// log(x / s) as a double-double, for x, s > 0 finite.
inline DD log_ratio(double x, double s) {
    const DD y = num::ratio(x, s);
    if (std::isfinite(y.hi) && y.hi >= 0x1p-1000) return dd::log(y);
    return num::log_dd(x) - num::log_dd(s);
}

// D_k by the series, x = 1/c a double-double, k x <= 0.85.
inline DD lgamma_diff(int k, DD x) {
    const DD kx = x * static_cast<double>(k);
    DD pk = kx * kx, p1 = x * x, sum{0.0, 0.0};
    for (int j = 2; j < 4000; ++j) {
        DD zeta{1.0, 0.0};
        if (j <= kZetaMinusOneMax) zeta = zeta + DD{kZetaMinusOne[j - 2][0], kZetaMinusOne[j - 2][1]};
        DD term = (zeta / static_cast<double>(j)) * (pk - p1 * static_cast<double>(k));
        if (j % 2 == 1) term = -term;
        sum = sum + term;
        if (std::fabs(term.hi) <= std::fabs(sum.hi) * 0x1p-108) break;
        pk = pk * kx;
        p1 = p1 * x;
    }
    return sum;
}

inline constexpr double kSeriesMax = 0.85;  // the largest k x for the series

// D_k = log Gamma(1 + k x) - k log Gamma(1 + x): the series while |k x| is small, else
// the Stirling forms in double-double (no cancellation left that matters: D_k is then of
// the size of its terms). x may be negative (GEV's c; 1 + k x > 0 is the caller's).
inline DD lgamma_ratio_x(int k, DD x) {
    if (std::fabs(k * x.hi) <= kSeriesMax) return lgamma_diff(k, x);
    return num::log_gamma(x * static_cast<double>(k) + 1.0) - num::log_gamma(x + 1.0) * static_cast<double>(k);
}
// The Weibull's D_k, x = 1/c.
inline DD lgamma_ratio(int k, double c) { return lgamma_ratio_x(k, num::div(DD{1.0, 0.0}, c)); }

// The skewness of W with E[W^k] = Gamma(1 + k x): (E_3 - 3 E_2) / E_2^(3/2); past
// D_3 = 600 scaled by e^-D_3 (the E's overflow).
inline double skewness_x(DD x) {
    const DD d2 = lgamma_ratio_x(2, x), d3 = lgamma_ratio_x(3, x);
    if (d3.hi > 600.0) {
        const double a = d2.hi, b = d3.hi;
        const double n = -std::expm1(-b) - 3.0 * (std::exp(a - b) - std::exp(-b));
        const double den = std::pow(-std::expm1(-a), 1.5);
        return std::exp(b - 1.5 * a) * n / den;
    }
    const DD e2 = num::expm1_full(d2), e3 = num::expm1_full(d3);
    const DD n = e3 - e2 * 3.0;
    return (n / (e2 * std::sqrt(e2.hi))).hi * (1.0 - 0.5 * e2.lo / e2.hi);
}
// Its excess kurtosis: (E_4 - 4 E_3 + 6 E_2) / E_2^2 - 3; past D_4 = 600 scaled by e^-D_4.
inline double kurtosis_x(DD x) {
    const DD d2 = lgamma_ratio_x(2, x), d3 = lgamma_ratio_x(3, x), d4 = lgamma_ratio_x(4, x);
    if (d4.hi > 600.0) {
        const double a = d2.hi, b = d3.hi, e = d4.hi;
        const double em = std::exp(-e);
        const double n = (1.0 - em) - 4.0 * (std::exp(b - e) - em) + 6.0 * (std::exp(a - e) - em) -
                         3.0 * std::exp(2.0 * a - e) * std::pow(-std::expm1(-a), 2.0);
        const double den = std::exp(2.0 * a - e) * std::pow(-std::expm1(-a), 2.0);
        return n / den - 3.0;
    }
    const DD e2 = num::expm1_full(d2), e3 = num::expm1_full(d3), e4 = num::expm1_full(d4);
    const DD den = e2 * e2;
    const DD n = e4 - e3 * 4.0 + e2 * 6.0 - den * 3.0;
    const DD r = n / den;
    return r.hi + r.lo;
}

}  // namespace weibull_detail

struct Weibull {
    static constexpr char name[] = "weibull";
    static constexpr std::array<const char*, 2> params{"c", "scale"};

    static bool valid(double c, double s) { return c > 0.0 && std::isfinite(c) && s > 0.0 && std::isfinite(s); }

    // t = (x/s)^c as a double-double, and E = c log(x/s). exp(E.hi) is rounded, and e^-t
    // multiplies that (relative) rounding by t: past t = 4 (more than ~4 ULP) the
    // residual E - log(exp(E.hi)) is folded into the low part.
    struct T {
        num::DD t, e;
    };
    static T t_of(double x, double c, double s) {
        const num::DD e = weibull_detail::log_ratio(x, s) * c;
        const double th = std::exp(e.hi);
        if (th > 4.0 && std::isfinite(th)) {
            const num::DD r = e - num::log_dd(th);
            return {num::DD{th, th * (r.hi + r.lo)}, e};
        }
        return {num::DD{th, th * e.lo}, e};
    }

    // log c - log x + E - t: the log density at x > 0 finite.
    // log c - log x + E - t = (log c - log s) + (c - 1) log y - t, y = x/s, with
    // (c - 1) log y as E - log y (c - 1 is not exact below 1/2) and log c - log s
    // remembered per parameters (num::Memo): one log per element.
    static num::DD log_density(double x, double c, double s) {
        const num::DD ly = weibull_detail::log_ratio(x, s);
        const num::DD e = ly * c;
        const double th = std::exp(e.hi);
        if (std::isinf(th)) return {-kInf, 0.0};
        num::DD t{th, th * e.lo};
        if (th > 4.0) {
            const num::DD r = e - num::log_dd(th);
            t = num::DD{th, th * (r.hi + r.lo)};
        }
        thread_local num::Memo memo;
        const num::DD k = memo.get(c, s, [&] { return num::log_dd(c) - num::log_dd(s); });
        return k + e - ly - t;
    }
    static double pdf(double x, double c, double s) {
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return c < 1.0 ? kInf : (c == 1.0 ? 1.0 / s : 0.0);
        const num::DD l = log_density(x, c, s);
        return std::isinf(l.hi) ? 0.0 : num::exp_dd(l);
    }
    static double logpdf(double x, double c, double s) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return c < 1.0 ? kInf : (c == 1.0 ? -std::log(s) : -kInf);
        const num::DD l = log_density(x, c, s);
        return l.hi + l.lo;
    }
    static double cdf(double x, double c, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return 1.0;
        const num::DD t = t_of(x, c, s).t;
        return -std::expm1(-t.hi) + std::exp(-t.hi) * t.lo;
    }
    static double ccdf(double x, double c, double s) {
        if (x <= 0.0) return 1.0;
        if (std::isinf(x)) return 0.0;
        const num::DD t = t_of(x, c, s).t;
        return std::isinf(t.hi) ? 0.0 : num::exp_dd(-t);
    }
    static double logcdf(double x, double c, double s) {
        if (x <= 0.0) return -kInf;
        if (std::isinf(x)) return 0.0;
        const T v = t_of(x, c, s);
        if (v.t.hi < 0x1p-60) {  // log(1 - e^-t) = log t - t/2 + ...: E to the last bit
            const num::DD l = v.e - v.t * 0.5;
            return l.hi + l.lo;
        }
        if (v.t.hi < num::kLn2.hi) return std::log(-std::expm1(-v.t.hi) + std::exp(-v.t.hi) * v.t.lo);
        return std::isinf(v.t.hi) ? 0.0 : std::log1p(-num::exp_dd(-v.t));
    }
    static double logccdf(double x, double c, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return -kInf;
        const num::DD t = t_of(x, c, s).t;
        return -(t.hi + t.lo);
    }

    // s exp(log(t) / c) from log t (a double-double); s goes into the exponent when the
    // power alone would leave the normal range.
    static double from_log_t(num::DD lt, double c, double s) {
        if (std::isinf(lt.hi)) return lt.hi > 0.0 ? kInf : 0.0;
        const num::DD e = num::div(lt, c);
        if (std::fabs(e.hi) < 700.0) return s * num::exp_dd(e);
        return num::exp_dd(e + num::log_dd(s));
    }
    static double icdf(double p, double c, double s) {  // t = -log(1 - p)
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return 0.0;
        if (p == 1.0) return kInf;
        if (p > 0.5) return from_log_t(weibull_detail::dd::log(-num::log_dd(1.0 - p)), c, s);
        if (p < 0x1p-60) return from_log_t(num::log_dd(p) + 0.5 * p, c, s);  // t = p + p^2/2 + ...
        return from_log_t(num::log_dd(-std::log1p(-p)), c, s);
    }
    static double iccdf(double p, double c, double s) {  // t = -log p
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return kInf;
        if (p == 1.0) return 0.0;
        if (p > 0.5) return icdf(1.0 - p, c, s);
        return from_log_t(weibull_detail::dd::log(-num::log_dd(p)), c, s);
    }
    static double ilogcdf(double lp, double c, double s) {  // t = -log(1 - e^lp)
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return kInf;
        if (lp == -kInf) return 0.0;
        if (lp < -40.0) return from_log_t(num::DD{lp, 0.5 * std::exp(lp)}, c, s);  // log t = lp + p/2 + ...
        if (lp < -num::kLn2.hi) return from_log_t(num::log_dd(-std::log1p(-std::exp(lp))), c, s);
        return from_log_t(weibull_detail::dd::log(-num::log_dd(-std::expm1(lp))), c, s);
    }
    static double ilogccdf(double lp, double c, double s) {  // t = -lp
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return 0.0;
        if (lp == -kInf) return kInf;
        return from_log_t(num::log_dd(-lp), c, s);
    }

    // Gamma(1 + 1/c), 1/c a double-double: Gamma(1 + x.hi) (1 + psi(1 + x.hi) x.lo). The
    // rounding of 1/c alone would cost psi(1 + x) x ULP (300 at c = 0.013).
    static double gamma1(double c) {
        const num::DD x = num::div(num::DD{1.0, 0.0}, c);
        const double g = special::Gamma::f(1.0 + x.hi);
        const num::DD a = num::dd::two_sum(1.0, x.hi);  // 1 + x.hi, its rounding too
        return g + g * (special::Digamma::f(a.hi) * (a.lo + x.lo));
    }
    static double mean(double c, double s) {
        const double g = gamma1(c);
        if (std::isfinite(g)) return s * g;
        const num::DD l = num::log_gamma(num::div(num::DD{1.0, 0.0}, c) + 1.0) + num::log_dd(s);
        return num::exp_dd(l);
    }
    // s^2 Gamma(1 + 1/c)^2 E_2, through logs when the factors overflow.
    static double variance(double c, double s) {
        using namespace weibull_detail;
        const num::DD d2 = lgamma_ratio(2, c);
        const double m1 = s * gamma1(c);
        if (std::isfinite(m1 * m1) && d2.hi < 600.0) return m1 * m1 * num::expm1_full(d2).hi;
        // log var = 2 log s + log Gamma(1 + 2/c) + log(1 - e^-D_2)
        const num::DD l = num::log_gamma(num::div(num::DD{2.0, 0.0}, c) + 1.0) + num::log_dd(s) * 2.0;
        return num::exp_dd(l + std::log1p(-std::exp(-d2.hi)));
    }
    static double standard_deviation(double c, double s) {
        using namespace weibull_detail;
        const num::DD d2 = lgamma_ratio(2, c);
        const double m1 = s * gamma1(c);
        if (std::isfinite(m1) && d2.hi < 600.0) return m1 * std::sqrt(num::expm1_full(d2).hi);
        const num::DD l = num::log_gamma(num::div(num::DD{2.0, 0.0}, c) + 1.0) + num::log_dd(s) * 2.0;
        return num::exp_dd((l + std::log1p(-std::exp(-d2.hi))) * 0.5);
    }
    static double skewness(double c, double) { return weibull_detail::skewness_x(num::div(num::DD{1.0, 0.0}, c)); }
    static double kurtosis(double c, double) { return weibull_detail::kurtosis_x(num::div(num::DD{1.0, 0.0}, c)); }
    static double median(double c, double s) {  // s (log 2)^(1/c)
        constexpr num::DD kLogLn2 = {-0.36651292058166435, 2.0606571710351486e-17};
        return s * num::exp_dd(num::div(kLogLn2, c));
    }
    static double mode(double c, double s) {  // s ((c - 1)/c)^(1/c) for c > 1, else 0
        if (c <= 1.0) return 0.0;
        const num::DD l = num::log_dd(c - 1.0) - num::log_dd(c);  // c - 1 exact
        return s * num::exp_dd(num::div(l, c));
    }
    static double entropy(double c, double s) {  // euler (1 - 1/c) + log(s / c) + 1
        constexpr num::DD kEuler = {0.5772156649015329, -4.942915152430645e-18};
        const num::DD x = num::div(num::DD{1.0, 0.0}, c);
        const num::DD h = kEuler - kEuler * x + num::log_dd(s) - num::log_dd(c) + 1.0;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
