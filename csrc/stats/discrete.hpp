// What the discrete distributions share (docs/design/stats.md T25, §3.2):
//
//   * the quantile search: the smallest k with P(X <= k) >= p (or P(X > k) <= p),
//     comparing on the side whose probability is the small one (1 - p is exact for
//     p >= 1/2) and on the log side below the normal range, by galloping from a
//     guess and bisecting -- O(log distance) evaluations, the answer exact up to
//     the cdf's own last bits;
//   * tail sums for the distributions with no closed-form cdf (hypergeometric,
//     beta-binomial): the terms relative to the first by their ratio recurrence,
//     resynchronized every 8 terms from the log pmf (so rounding does not pile up
//     over thousands of terms), log of the sum kept apart from the log pmf so the log
//     tails go below the normal range;
//   * the entropy -sum pmf log pmf over the support, from the mode outwards, compensated.
#pragma once

#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sd::stats::disc {

// Below this a probability is compared on the log side.
inline constexpr double kTinyP = 0x1p-1000;

// The smallest integer k in [lo, hi] with pred(k), pred monotone (false, then true)
// and true at hi (hi may be inf).
template <class P>
double smallest(double lo, double hi, double guess, P&& pred) {
    double g = std::floor(guess);
    if (!(g >= lo)) g = lo;
    if (g > hi) g = hi;
    double a, b;  // pred(a) false (a = lo - 1 counts as false), pred(b) true
    if (pred(g)) {
        b = g;
        a = lo - 1.0;
        for (double step = 1.0; b > lo; step *= 2.0) {
            const double c = std::max(lo, b - step);
            if (!pred(c)) {
                a = c;
                break;
            }
            b = c;
        }
    } else {
        a = g;
        b = hi;
        for (double step = 1.0;; step *= 2.0) {
            const double c = a + step;
            if (c >= hi) break;
            if (!(c < 0x1p1023)) {
                // The last doubling would pass 2^1023: the largest double decides whether the
                // quantile is still finite (Zipf a = 2.00008 at q = 1.2e-308: 4.9e307).
                const double top = std::numeric_limits<double>::max();
                if (!pred(top)) return kInf;  // the quantile is beyond float64 (an infinite support)
                b = top;
                break;
            }
            if (pred(c)) {
                b = c;
                break;
            }
            a = c;
        }
    }
    while (b - a > 1.0) {
        const double m = std::floor(a + 0.5 * (b - a));
        if (m <= a || m >= b) break;  // past 2^53 no float64 lies between: b is the answer rounded
        if (pred(m)) b = m;
        else a = m;
    }
    return b;
}

// a <= b for double-doubles (normalized: |lo| <= ULP(hi) / 2).
inline bool dd_le(num::DD a, num::DD b) { return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo); }

// The four inverses of a discrete distribution from its cdf, ccdf and their logs at
// integers (F: struct with cdf(k), ccdf(k), logcdf(k), logccdf(k), lo, hi, guess; with
// logccdf_dd(k), the log of a deep upper tail is compared as a double-double: a large
// quantile there moves by ULP(log q) / (k |d log ccdf / dk|), 1e-13 / (a - 1) for Zipf).
template <class F>
double icdf(double p, const F& f) {
    if (!(p >= 0.0 && p <= 1.0)) return kNaN;
    if (p == 0.0) return f.lo;
    if (p == 1.0) return f.hi;
    if (p > 0.5) {
        const double q = 1.0 - p;  // exact
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.ccdf(k) <= q; });
    }
    if (p >= kTinyP) return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.cdf(k) >= p; });
    const double lp = std::log(p);
    return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logcdf(k) >= lp; });
}
template <class F>
double iccdf(double q, const F& f) {
    if (!(q >= 0.0 && q <= 1.0)) return kNaN;
    if (q == 0.0) return f.hi;
    if (q == 1.0) return f.lo;
    if (q > 0.5) {
        const double p = 1.0 - q;
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.cdf(k) >= p; });
    }
    if (q >= kTinyP) return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.ccdf(k) <= q; });
    if constexpr (requires { f.logccdf_dd(0.0); }) {  // log q ~ -745: compare below its ULP
        const num::DD lq = num::log_dd(q);
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return dd_le(f.logccdf_dd(k), lq); });
    }
    const double lq = std::log(q);
    return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logccdf(k) <= lq; });
}
template <class F>
double ilogcdf(double lp, const F& f) {
    if (!(lp <= 0.0)) return kNaN;
    if (lp == 0.0) return f.hi;
    if (lp == -kInf) return f.lo;
    if (lp <= -num::kLn2.hi) return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logcdf(k) >= lp; });
    return iccdf(-std::expm1(lp), f);  // P(X > k) <= 1 - e^lp, the small side
}
template <class F>
double ilogccdf(double lq, const F& f) {
    if (!(lq <= 0.0)) return kNaN;
    if (lq == 0.0) return f.lo;
    if (lq == -kInf) return f.hi;
    if (lq <= -num::kLn2.hi) {
        if constexpr (requires { f.logccdf_dd(0.0); })
            return smallest(f.lo, f.hi, f.guess, [&](double k) { return dd_le(f.logccdf_dd(k), num::DD{lq, 0.0}); });
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logccdf(k) <= lq; });
    }
    return icdf(-std::expm1(lq), f);
}

// log sum_{j} pmf(j) / pmf(k) over j = k, k + dir, k + 2 dir, ... up to `end` (inclusive),
// the terms decreasing away from k: t_{j+dir} = t_j ratio(j) (ratio(j) = pmf(j + dir) / pmf(j)),
// resynchronized every 8 terms as exp(logpmf(j) - logpmf(k)) (logpmf: a double-double).
template <class Ratio, class LogPmf>
double log_tail_factor(double k, double dir, double end, Ratio&& ratio, LogPmf&& logpmf) {
    const num::DD l0 = logpmf(k);
    double sum = 1.0, t = 1.0, j = k;
    for (int i = 1; i < 100000000; ++i) {
        if ((dir > 0.0 && j >= end) || (dir < 0.0 && j <= end)) break;
        t *= ratio(j);
        j += dir;
        if (i % 8 == 0) {
            const num::DD d = logpmf(j) - l0;
            t = std::exp(d.hi + d.lo);
        }
        sum += t;
        if (t <= sum * 0x1p-60) break;
    }
    return std::log(sum);
}

// A sum compensated as Neumaier's (thousands of terms, as for the entropy of a spread pmf,
// would otherwise lose ~sqrt(n) ULP).
struct KahanSum {
    double s = 0.0, c = 0.0;
    void add(double x) {
        const double t = s + x;
        c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    double value() const { return s + c; }
};

// The entropy -sum pmf log pmf, from the mode outwards until the rest is negligible, each
// term from the double-double log pmf, the sum compensated. The rest past a term t whose
// ratio to the one before is r < 1 is about t r / (1 - r) (the ratios keep falling past
// the peak); the walk stops when that is below 2^-60 of the sum. A stop at t below 2^-60
// of the sum alone left 5e-13 of a negative binomial with sd 2.3e5 (8.6 sd out, the rest
// ~3e4 terms of that size).
template <class LogPmf>
double entropy_sum(double mode, double lo, double hi, LogPmf&& logpmf) {
    KahanSum h;
    auto term = [&](double k) {
        const num::DD l = logpmf(k);
        const double lv = l.hi + l.lo;
        return lv == -kInf ? 0.0 : -num::exp_dd(l) * lv;
    };
    auto done = [&](double t, double prev) {
        const double r = std::fabs(prev) > 0.0 ? std::fabs(t) / std::fabs(prev) : 0.0;
        return r < 1.0 && std::fabs(t) * r <= (1.0 - r) * std::fabs(h.s) * 0x1p-60;
    };
    double m = std::clamp(std::floor(mode), lo, hi);
    const double tm = term(m);
    h.add(tm);
    double prev = tm;
    for (double k = m + 1.0; k <= hi; k += 1.0) {
        const double t = term(k);
        h.add(t);
        if (k > mode + 1.0 && done(t, prev)) break;
        prev = t;
    }
    prev = tm;
    for (double k = m - 1.0; k >= lo; k -= 1.0) {
        const double t = term(k);
        h.add(t);
        if (k < mode - 1.0 && done(t, prev)) break;
        prev = t;
    }
    return h.value();
}

// A closed-form estimate of the quantile as the search's start (the search then settles
// it in two or three evaluations); anything not finite or past 2^1023 keeps the default.
template <class F>
F with_guess(F f, double g) {
    if (g >= f.lo && g < 0x1p1023) f.guess = std::min(std::floor(g), f.hi);
    return f;
}

// k is an integer (finite, no fractional part).
inline bool is_int(double k) { return std::isfinite(k) && std::floor(k) == k; }

}  // namespace sd::stats::disc
