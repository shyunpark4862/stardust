// The gamma distribution with shape a and scale s (T18), and the chi-square
// distribution with df degrees of freedom (= gamma(df/2, scale 2)).
//
// The cdfs are sd.special's regularized incomplete gamma P(a, y), Q(a, y) at
// y = x / s, which is a double-double (stats/numeric.hpp): the rounded y.hi
// goes to the incomplete gamma and y.lo enters to first order through
// y P'(y) = y^a e^-y / Gamma(a) (the condition number of P in y reaches a).
// Below the normal range the log tails come from the unrounded pieces
// (igam_detail::log_tail: the prefix's exponent in double-double times the
// series or continued fraction), as the inverses' deep-tail iteration does.
#pragma once

#include "special/ops/gamma.hpp"
#include "special/ops/incgamma.hpp"
#include "special/prefix.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace gamma_detail {

using num::DD;
namespace dd = num::dd;

// log of the standard density y^(a-1) e^-y / Gamma(a) at y > 0 finite (a double-double).
// For a >= 13 through the prefix's exponent E = y - a - a log(y/a) (Temme's form,
// special/prefix.hpp): log(a D / y) = log a - log y - E - log(sqrt(2 pi a) Gamma*(a));
// E at y.hi in double-double, y.lo to first order (dE/dy = 1 - a/y).
inline DD log_density_std(double a, DD y) {
    if (a >= 13.0) {
        DD e = special::gamma_exponent(a, y.hi);
        e = e + (1.0 - a / y.hi) * y.lo;
        thread_local num::Memo memo;  // log a - log(sqrt(2 pi a) Gamma*(a))
        const DD c = memo.get(a, 0.0, [&] {
            return num::log_dd(a) - std::log(std::sqrt(2.0 * special::kPi * a) * special::gamma_star(a));
        });
        return c - dd::log(y) - e;
    }
    // (a - 1) log y - y - log Gamma(a); log Gamma(a) = log Gamma(1 + a) - log a (no
    // large terms to round for a tiny a). (a - 1) log y as a log y - log y: a - 1 is not
    // exact below a = 1/2, and log y reaches 745 (39 ULP).
    const DD ly = y.hi >= 0x1p-1000 ? dd::log(y) : num::log_dd(y.hi);
    thread_local num::Memo memo;
    const DD lg = memo.get(a, 0.0, [&] { return num::log_gamma(DD{1.0, 0.0} + a) - num::log_dd(a); });
    return ly * a - ly - y - lg;
}

// P(a, y) (upper: Q) at y = hi + lo > 0 finite.
inline double tail(double a, DD y, bool upper) {
    const special::igam_detail::PQ r = special::igam_detail::igam(a, y.hi);
    const double d = r.xderiv * (y.lo / y.hi);  // y P'(y) dy / y
    return upper ? r.q - d : r.p + d;
}

// log P(a, y) (upper: log Q). The smaller side: log of its value, or below the
// normal range log_tail's unrounded pieces; the larger: log1p(-smaller).
inline double log_tail(double a, DD y, bool upper) {
    const special::igam_detail::PQ r = special::igam_detail::igam(a, y.hi);
    const double d = r.xderiv * (y.lo / y.hi);
    const double p = r.p + d, q = r.q - d;
    const bool lower_small = p <= q;
    const double small = lower_small ? p : q, large = lower_small ? q : p;
    if (upper == !lower_small) {  // the small side is asked for
        if (small >= 0x1p-1000) return std::log(small);
        const auto [l, dl] = special::igam_detail::log_tail(a, y.hi, lower_small, DD{0.0, 0.0});
        return l + dl * y.lo;
    }
    (void)large;
    return std::log1p(-small);
}

// s y for the y with log P(a, y) = lt (lower) or log Q(a, y) = lt (upper), lt a
// double-double far below the normal range: Newton on log F from the unrounded pieces
// (igam_detail::log_tail), in log y for P (y tiny) and in y for Q (y large), from the
// tails' leading terms. A small lower root comes from the series
// P = y^a / Gamma(a + 1) (1 - a y / (a + 1) + O(y^2)):
// log y = (lt + log Gamma(a + 1)) / a + y / (a + 1), exact once y < 1e-9, in
// double-double with s folded into the exponent (y may be subnormal while s y is not).
inline double deep_inverse(double a, DD lt, bool lower, double s) {
    using special::igam_detail::log_tail;
    double y;
    if (lower) {
        const DD ly = (lt + num::log_gamma(DD{1.0, 0.0} + a)) / a;
        const double y0 = std::exp(ly.hi);
        if (y0 < 1e-9) return num::exp_dd(ly + y0 / (a + 1.0) + num::log_dd(s));
        y = std::exp(ly.hi);
    } else {  // Q ~ y^(a-1) e^-y / Gamma(a): y = -L + (a - 1) log y
        const double L = lt.hi + special::Gammaln::f(a);
        y = std::max(1.0, -L);
        for (int i = 0; i < 8; ++i) y = std::max(1.0, -L + (a - 1.0) * std::log(y));
    }
    if (!std::isfinite(y)) return s * y;
    for (int it = 0; it < 100; ++it) {
        const auto [g, dlf] = log_tail(a, y, lower, lt);
        double yn = lower ? y * std::exp(-g / (y * dlf)) : y - g / dlf;
        if (!(yn > 0.0) || !std::isfinite(yn)) yn = lower ? 0.5 * y : 2.0 * y;
        if (std::fabs(yn - y) <= 0x1p-54 * yn) return s * yn;
        y = yn;
    }
    return s * y;
}

// Below this a probability goes to the deep inverses (from its log in double-double).
inline constexpr double kDeepP = 0x1p-1000;

// s y with P(a, y) = p, given p and lt = log p: sd.special's gammaincinv while p and
// the root are normal numbers, else the deep inverse (a root below 1e-250 loses bits
// as a float64 before s scales it).
inline double lower_inverse(double a, double s, double p, DD lt) {
    if (p >= kDeepP) {
        const double est = (lt.hi + special::Gammaln::f(a + 1.0)) / a;  // log of the leading-term root
        if (est > -40.0) return s * special::Gammaincinv::f(a, p);
    }
    return deep_inverse(a, lt, true, s);
}
inline double upper_inverse(double a, double s, double q, DD lt) {
    if (q >= kDeepP) return s * special::Gammainccinv::f(a, q);
    return deep_inverse(a, lt, false, s);
}

// The differential entropy of gamma(a, 1): a + log Gamma(a) + (1 - a) psi(a)
// = G(a) + log(a)/2 + log(2 pi)/2 (stats/numeric.hpp: G is O(1); the terms grow
// like a log a).
inline DD entropy_std(double a) {
    return num::log_dd(a) * 0.5 + num::kLogSqrt2Pi + num::entropy_g(a, special::Digamma::f);
}

}  // namespace gamma_detail

struct Gamma {
    static constexpr char name[] = "gamma";
    static constexpr std::array<const char*, 2> params{"a", "scale"};

    static bool valid(double a, double s) { return a > 0.0 && std::isfinite(a) && s > 0.0 && std::isfinite(s); }

    // log density of X at x > 0 finite.
    static num::DD log_density(double x, double a, double s) {
        thread_local num::Memo memo;
        return gamma_detail::log_density_std(a, num::ratio(x, s)) - memo.get(s, 0.0, [&] { return num::log_dd(s); });
    }
    static double pdf(double x, double a, double s) {
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return a < 1.0 ? kInf : (a == 1.0 ? 1.0 / s : 0.0);
        return num::exp_dd(log_density(x, a, s));
    }
    static double logpdf(double x, double a, double s) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return a < 1.0 ? kInf : (a == 1.0 ? -std::log(s) : -kInf);
        const num::DD l = log_density(x, a, s);
        return l.hi + l.lo;
    }
    static double cdf(double x, double a, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return 1.0;
        return gamma_detail::tail(a, num::ratio(x, s), false);
    }
    static double ccdf(double x, double a, double s) {
        if (x <= 0.0) return 1.0;
        if (std::isinf(x)) return 0.0;
        return gamma_detail::tail(a, num::ratio(x, s), true);
    }
    static double logcdf(double x, double a, double s) {
        if (x <= 0.0) return -kInf;
        if (std::isinf(x)) return 0.0;
        return gamma_detail::log_tail(a, num::ratio(x, s), false);
    }
    static double logccdf(double x, double a, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return -kInf;
        return gamma_detail::log_tail(a, num::ratio(x, s), true);
    }

    // The side solved is the smaller of p and 1 - p (1 - p exact for p >= 1/2).
    static double icdf(double p, double a, double s) {
        if (!(p > 0.0 && p < 1.0)) return special::Gammaincinv::f(a, p) * s;  // ends and NaN
        if (p > 0.5) return gamma_detail::upper_inverse(a, s, 1.0 - p, num::log_dd(1.0 - p));
        return gamma_detail::lower_inverse(a, s, p, num::log_dd(p));
    }
    static double iccdf(double p, double a, double s) {
        if (!(p > 0.0 && p < 1.0)) return special::Gammainccinv::f(a, p) * s;
        if (p > 0.5) return gamma_detail::lower_inverse(a, s, 1.0 - p, num::log_dd(1.0 - p));
        return gamma_detail::upper_inverse(a, s, p, num::log_dd(p));
    }
    // From log p: the side with the smaller probability, e^lp (lp exact) or 1 - e^lp.
    static double from_log(double lp, double a, double s, bool upper) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return upper ? 0.0 : kInf;
        if (lp == -kInf) return upper ? kInf : 0.0;
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return upper ? gamma_detail::lower_inverse(a, s, q, num::log_dd(q))
                         : gamma_detail::upper_inverse(a, s, q, num::log_dd(q));
        }
        const double p = std::exp(lp);
        return upper ? gamma_detail::upper_inverse(a, s, p, num::DD{lp, 0.0})
                     : gamma_detail::lower_inverse(a, s, p, num::DD{lp, 0.0});
    }
    static double ilogcdf(double lp, double a, double s) { return from_log(lp, a, s, false); }
    static double ilogccdf(double lp, double a, double s) { return from_log(lp, a, s, true); }

    static double mean(double a, double s) { return a * s; }
    static double variance(double a, double s) { return a * s * s; }
    static double standard_deviation(double a, double s) { return std::sqrt(a) * s; }
    static double skewness(double a, double) { return 2.0 / std::sqrt(a); }
    static double kurtosis(double a, double) { return 6.0 / a; }
    static double mode(double a, double s) { return a > 1.0 ? (a - 1.0) * s : 0.0; }  // a - 1 exact
    static double entropy(double a, double s) {
        const num::DD h = gamma_detail::entropy_std(a) + num::log_dd(s);
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

// chi-square(df) = gamma(df/2, 2): y = x/2 is exact.
struct ChiSquared {
    static constexpr char name[] = "chisquared";
    static constexpr std::array<const char*, 1> params{"df"};

    static bool valid(double df) { return df > 0.0 && std::isfinite(df); }

    static double pdf(double x, double df) { return Gamma::pdf(x, 0.5 * df, 2.0); }
    static double logpdf(double x, double df) { return Gamma::logpdf(x, 0.5 * df, 2.0); }
    static double cdf(double x, double df) { return Gamma::cdf(x, 0.5 * df, 2.0); }
    static double ccdf(double x, double df) { return Gamma::ccdf(x, 0.5 * df, 2.0); }
    static double logcdf(double x, double df) { return Gamma::logcdf(x, 0.5 * df, 2.0); }
    static double logccdf(double x, double df) { return Gamma::logccdf(x, 0.5 * df, 2.0); }
    static double icdf(double p, double df) { return Gamma::icdf(p, 0.5 * df, 2.0); }
    static double iccdf(double p, double df) { return Gamma::iccdf(p, 0.5 * df, 2.0); }
    static double ilogcdf(double lp, double df) { return Gamma::ilogcdf(lp, 0.5 * df, 2.0); }
    static double ilogccdf(double lp, double df) { return Gamma::ilogccdf(lp, 0.5 * df, 2.0); }

    static double mean(double df) { return df; }
    static double variance(double df) { return 2.0 * df; }
    static double standard_deviation(double df) { return std::sqrt(2.0 * df); }
    static double skewness(double df) { return std::sqrt(8.0 / df); }
    static double kurtosis(double df) { return 12.0 / df; }
    static double median(double df) { return Gamma::icdf(0.5, 0.5 * df, 2.0); }
    static double mode(double df) { return df > 2.0 ? df - 2.0 : 0.0; }
    static double entropy(double df) { return Gamma::entropy(0.5 * df, 2.0); }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double) { return kInf; }
};

}  // namespace sd::stats
