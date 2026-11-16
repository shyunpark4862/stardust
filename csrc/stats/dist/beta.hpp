// The beta distribution on [0, 1] with shapes a, b, and the incomplete-beta
// pieces the t and F distributions share.
//
// The density is the incomplete beta's prefix x^a y^b / B(a, b) (special/prefix.hpp:
// its exponent in double-double, no over- or underflow) divided by x y; the cdfs
// are sd.special's I_x(a, b) and 1 - I_x(a, b) (ibeta, TOMS 708). Below the
// normal range the log of the small tail is the log of the prefix (from its
// exponent) plus the log of the series or continued fraction that multiplies it
// (BPSER, BFRAC), so logcdf keeps going where the tail itself underflows.
#pragma once

#include "special/ops/gamma.hpp"
#include "special/ops/incbeta.hpp"
#include "special/prefix.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace beta_detail {

using num::DD;
namespace dd = num::dd;

// log of the prefix x^a y^b / B(a, b), x + y = 1 (double-doubles).
inline DD log_prefix(double a, double b, DD x, DD y) {
    thread_local num::Memo memo;
    const DD f = memo.get(a, b, [&] { return DD{std::log(special::beta_factor(a, b)), 0.0}; });
    return -special::beta_exponent(a, b, x, y) + f;
}

// log I_x(a, b) for a deep lower tail (x well below the mean): the log prefix plus
// log(I / prefix), the factor by BFRAC when a, b > 1 (TOMS 708's choice away from the
// mean) and by BPSER otherwise, both given a unit prefix; BGRAT (in logs) for a > 15,
// b <= 1 and x within 1e-3 of 1, where BPSER's series in x needs ~40 / (1 - x) terms and
// stops at its cap (the negative binomial's ccdf at k = 3.4e7, p = 1.2e-4 was 7e-10 off).
// Above that BPSER stays: measured on the t's deep tails (y ~ 2e-3), it is 2e-16 to
// BGRAT's 4e-16.
inline double log_lower_deep(double a, double b, DD x, DD y) {
    if (b <= 1.0 && a > 15.0 && y.hi < 1e-3) return special::ibeta_detail::log_bgrat(a, b, x, y);
    const special::Scaled unit = special::scaled(1.0);
    const double ratio = (a > 1.0 && b > 1.0) ? special::ibeta_detail::bfrac(a, b, x, y, unit)
                                              : special::ibeta_detail::bpser(a, b, x, y, unit);
    const DD l = log_prefix(a, b, x, y);
    return l.hi + (l.lo + std::log(ratio));
}

// Below this a tail is "deep": its log comes from the pieces, not from its value.
inline constexpr double kDeep = 0x1p-960;

// log I_x(a, b) (upper: log(1 - I_x(a, b))).
inline double log_tail(double a, double b, DD x, DD y, bool upper) {
    const special::ibeta_detail::IB r = special::ibeta_detail::ibeta(a, b, x, y);
    const bool lower_small = r.i <= r.ic;
    const double small = lower_small ? r.i : r.ic;
    if (upper != lower_small) {  // the small side is asked for
        if (small >= kDeep) return std::log(small);
        return lower_small ? log_lower_deep(a, b, x, y) : log_lower_deep(b, a, y, x);
    }
    return std::log1p(-small);
}

// log B(a, b) in double-double (log-gammas: the start and the leading term below).
inline DD log_beta(double a, double b) {
    return num::log_gamma(DD{a, 0.0}) + num::log_gamma(DD{b, 0.0}) - num::log_gamma(dd::two_sum(a, b));
}

// log x for the x with log I_x(a, b) = lt (a double-double), deep below the normal
// range. For a small root, the series I = x^a / (a B(a, b)) (1 + c x + O(x^2)),
// c = a (1 - b) / (a + 1): log x = (lt + log a + log B) / a - c x / a, exact once
// c x < 1e-9 (the next term is below 2^-60). Otherwise Newton in log x on log I from
// the pieces, d log I / d log x = x f / I = prefix / (y I).
inline DD deep_log_inverse(double a, double b, DD lt) {
    const DD lx0 = (lt + num::log_dd(a) + log_beta(a, b)) / a;
    const double c = a * (1.0 - b) / (a + 1.0);
    const double x0 = std::exp(lx0.hi);
    if (std::fabs(c) * x0 < 1e-9) return lx0 - c * x0 / a;
    double lx = lx0.hi, step = 0.0;
    for (int it = 0; it < 100; ++it) {
        const double x = std::exp(lx);
        const DD xd{x, 0.0}, yd = dd::two_sum(1.0, -x);
        const double li = log_lower_deep(a, b, xd, yd);
        const DD lp = log_prefix(a, b, xd, yd);
        const double slope = std::exp(lp.hi - std::log(yd.hi) - li);
        const DD g = DD{li, 0.0} - lt;
        step = (g.hi + g.lo) / slope;
        lx -= step;
        if (std::fabs(step) <= 0x1p-60 * std::max(1.0, std::fabs(lx))) break;
    }
    return DD{lx, 0.0};
}

// Below this a probability goes to the deep inverse (from its log in double-double).
inline constexpr double kDeepP = 0x1p-1000;

// x with I_x(a, b) = p (p <= 1/2 the small side, lt = log p): sd.special's betaincinv
// while p and the root are normal numbers, else the deep inverse. Returns log x too
// when it came from the deep side (lx.hi = NaN otherwise).
struct Root {
    double x;
    DD lx;
};
inline Root lower_root(double a, double b, double p, DD lt) {
    if (p >= kDeepP) {
        const double est = (lt.hi + std::log(a) + special::Betaln::f(a, b)) / a;  // log of the leading-term root
        if (est > -40.0) return {special::Betaincinv::f(a, b, p), DD{kNaN, 0.0}};
    }
    const DD lx = deep_log_inverse(a, b, lt);
    return {num::exp_dd(lx), lx};
}

}  // namespace beta_detail

struct Beta {
    static constexpr char name[] = "beta";
    static constexpr std::array<const char*, 2> params{"a", "b"};

    static bool valid(double a, double b) { return a > 0.0 && std::isfinite(a) && b > 0.0 && std::isfinite(b); }

    // log density at 0 < x < 1: log prefix - log x - log(1 - x).
    static num::DD log_density(double x, double a, double b) {
        const num::DD xd{x, 0.0}, yd = num::dd::two_sum(1.0, -x);
        return beta_detail::log_prefix(a, b, xd, yd) - num::log_dd(x) - num::dd::log(yd);
    }
    // The density at an end: x = 0 is (a < 1: inf, a = 1: b, a > 1: 0), x = 1 mirrored.
    static double end_density(double s, double other) { return s < 1.0 ? kInf : (s == 1.0 ? other : 0.0); }
    static double pdf(double x, double a, double b) {
        if (x < 0.0 || x > 1.0) return 0.0;
        if (x == 0.0) return end_density(a, b);
        if (x == 1.0) return end_density(b, a);
        return num::exp_dd(log_density(x, a, b));
    }
    static double logpdf(double x, double a, double b) {
        if (x < 0.0 || x > 1.0) return -kInf;
        if (x == 0.0) return std::log(end_density(a, b));
        if (x == 1.0) return std::log(end_density(b, a));
        const num::DD l = log_density(x, a, b);
        return l.hi + l.lo;
    }
    static double tail(double x, double a, double b, bool upper) {
        if (x <= 0.0) return upper ? 1.0 : 0.0;
        if (x >= 1.0) return upper ? 0.0 : 1.0;
        const special::ibeta_detail::IB r = special::ibeta_detail::ibeta(a, b, x);
        return upper ? r.ic : r.i;
    }
    static double cdf(double x, double a, double b) { return tail(x, a, b, false); }
    static double ccdf(double x, double a, double b) { return tail(x, a, b, true); }
    static double log_tail(double x, double a, double b, bool upper) {
        if (x <= 0.0) return upper ? 0.0 : -kInf;
        if (x >= 1.0) return upper ? -kInf : 0.0;
        return beta_detail::log_tail(a, b, num::DD{x, 0.0}, num::dd::two_sum(1.0, -x), upper);
    }
    static double logcdf(double x, double a, double b) { return log_tail(x, a, b, false); }
    static double logccdf(double x, double a, double b) { return log_tail(x, a, b, true); }

    // x with I_x(a, b) = t (upper: 1 - I_x(a, b) = t) for a normal t, lt = log t: sd.special's
    // inverses, which choose the side by the root (x or 1 - x, whichever is small), unless
    // t is tiny or the root is (beta_detail::lower_root: the deep inverse on the log side).
    // For the upper side a tiny t means 1 - x tiny: solved as the lower side of (b, a).
    static double solve(double t, num::DD lt, double a, double b, bool upper) {
        if (!upper) return beta_detail::lower_root(a, b, t, lt).x;
        if (t >= beta_detail::kDeepP) {
            const double est = (lt.hi + std::log(b) + special::Betaln::f(a, b)) / b;  // log of a tiny 1 - x
            if (est > -40.0) return special::Betainccinv::f(a, b, t);
        }
        return 1.0 - beta_detail::lower_root(b, a, t, lt).x;
    }
    static double icdf(double p, double a, double b) {
        if (!(p > 0.0 && p < 1.0)) return special::Betaincinv::f(a, b, p);
        return solve(p, num::log_dd(p), a, b, false);
    }
    static double iccdf(double p, double a, double b) {
        if (!(p > 0.0 && p < 1.0)) return special::Betainccinv::f(a, b, p);
        return solve(p, num::log_dd(p), a, b, true);
    }
    // From log p: e^lp (lp exact) for the side asked, or, above log 1/2, 1 - e^lp for the
    // other side (the probability there is the small one).
    static double from_log(double lp, double a, double b, bool upper) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return upper ? 0.0 : 1.0;
        if (lp == -kInf) return upper ? 1.0 : 0.0;
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return solve(q, num::log_dd(q), a, b, !upper);
        }
        return solve(std::exp(lp), num::DD{lp, 0.0}, a, b, upper);
    }
    static double ilogcdf(double lp, double a, double b) { return from_log(lp, a, b, false); }
    static double ilogccdf(double lp, double a, double b) { return from_log(lp, a, b, true); }

    static double mean(double a, double b) { return a / (a + b); }
    static double variance(double a, double b) {
        const double c = a + b;
        return (a / c) * (b / c) / (c + 1.0);
    }
    static double skewness(double a, double b) {
        const double c = a + b;
        return 2.0 * (b - a) * std::sqrt(c + 1.0) / ((c + 2.0) * std::sqrt(a * b));
    }
    // 6 ((a - b)^2 (c + 1) - a b (c + 2)) / (a b (c + 2) (c + 3)): the numerator crosses 0,
    // so it is formed in double-double.
    static double kurtosis(double a, double b) {
        const num::DD c = num::dd::two_sum(a, b);
        const num::DD d = num::dd::two_sum(a, -b);
        const num::DD ab = num::dd::two_prod(a, b);
        const num::DD n = d * d * (c + 1.0) - ab * (c + 2.0);
        const num::DD r = n * 6.0 / (ab * (c + 2.0) * (c + 3.0));
        return r.hi + r.lo;
    }
    // The single largest point (T21): (a - 1)/(a + b - 2) inside; an end where only that
    // end is largest; NaN for two ends (a, b < 1) or the uniform a = b = 1.
    static double mode(double a, double b) {
        if (a > 1.0 && b > 1.0) return (a - 1.0) / (a + b - 2.0);
        if (a <= 1.0 && b <= 1.0) {
            if (a == 1.0 && b < 1.0) return 1.0;
            if (b == 1.0 && a < 1.0) return 0.0;
            return kNaN;
        }
        return a <= 1.0 ? 0.0 : 1.0;
    }
    // log B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (c - 2) psi(c)
    // = G(a) + G(b) + Gc(c) + log(a)/2 + log(b)/2 - 3 log(c)/2 + log(2 pi)/2 (numeric.hpp).
    static double entropy(double a, double b) {
        const num::DD c = num::dd::two_sum(a, b);
        const num::DD logs = (num::log_dd(a) + num::log_dd(b) - num::dd::log(c) * 3.0) * 0.5 + num::kLogSqrt2Pi;
        const double g = num::entropy_g(a, special::Digamma::f) + num::entropy_g(b, special::Digamma::f) +
                         num::entropy_gc(c, special::Digamma::f);
        const num::DD h = logs + g;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return 1.0; }
};

}  // namespace sd::stats
