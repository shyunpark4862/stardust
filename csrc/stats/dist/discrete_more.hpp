// The discrete uniform, hypergeometric, beta-binomial, logarithmic series and Zipf
// distributions (docs/api/stats.md §2.1, T25).
//
//   * hypergeometric: up to a total of 120 the pmf and the cdfs as ratios of exact
//     128-bit counts; beyond, the pmf as dbinom(k; K, p) dbinom(n - k; N - K, p) /
//     dbinom(n; N, p) (any p; the identity cancels it), each a binomial pmf from its
//     prefix, in logs; skewness in closed form, the kurtosis's bracket in exact integers;
//   * beta-binomial: log-gammas in double-double; skewness in closed form, the excess
//     kurtosis as a polynomial without the "- 3", in double-double;
//   * both: cdfs as tail sums (stats/discrete.hpp) of the side below 1/2, from its
//     largest term outwards by the ratio recurrence in double-double, as double-double logs;
//     a long smooth run of terms by disc::smooth_tail_sum instead (disc::tail_sum chooses);
//     over arrays the hypergeometric's pmf near the mean by the binomials' fast path;
//   * logarithmic series: the tail sum_{j>k} p^j / j directly or, for p >= 0.9, by
//     Euler-Maclaurin with e^x E_1(x) for the integral; log(L / p) apart, so the log pmf
//     at k = 1 keeps a tiny p;
//   * Zipf: Riemann and Hurwitz zeta (sd.special), log zeta(a) as log1p(zeta(a) - 1), the
//     cdf past k = 1000 as a sum plus a Hurwitz difference by Euler-Maclaurin (no
//     cancellation near a = 1), the central moments as finite sums plus Hurwitz tails.
#pragma once

#include "special/cf.hpp"
#include "special/ops/expint.hpp"
#include "special/ops/gamma.hpp"
#include "special/ops/zeta.hpp"
#include "stats/chunk.hpp"
#include "stats/discrete.hpp"
#include "stats/dist/discrete_basic.hpp"
#include "stats/dist/two_point.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace sd::stats {

// ---------------------------------------------------------------- discrete uniform

struct DiscreteUniform {
    static constexpr char name[] = "discreteuniform";
    static constexpr std::array<const char*, 2> params{"low", "high"};

    // low and high integers with 0 < high - low < inf. The width in range also says that both
    // ends are finite, and a width past the float64 range has no pmf (1 / width): NaN.
    static bool valid(double lo, double hi) {
        const double w = hi - lo;
        return std::floor(lo) == lo && std::floor(hi) == hi && w > 0.0 && w < kInf;
    }

    static double pmf(double k, double lo, double hi) {
        return disc::is_int(k) && k >= lo && k < hi ? 1.0 / (hi - lo) : 0.0;
    }
    static double logpmf(double k, double lo, double hi) {
        return disc::is_int(k) && k >= lo && k < hi ? -std::log(hi - lo) : -kInf;
    }
    static double tail(double x, double lo, double hi, bool upper) {
        const double k = std::floor(x);
        if (k < lo) return upper ? 1.0 : 0.0;
        if (k >= hi - 1.0) return upper ? 0.0 : 1.0;
        return (upper ? hi - 1.0 - k : k - lo + 1.0) / (hi - lo);
    }
    static double cdf(double x, double lo, double hi) { return tail(x, lo, hi, false); }
    static double ccdf(double x, double lo, double hi) { return tail(x, lo, hi, true); }
    static double log_tail(double x, double lo, double hi, bool upper) {
        const double v = tail(x, lo, hi, upper);
        return v <= 0.5 ? std::log(v) : std::log1p(-tail(x, lo, hi, !upper));
    }
    static double logcdf(double x, double lo, double hi) { return log_tail(x, lo, hi, false); }
    static double logccdf(double x, double lo, double hi) { return log_tail(x, lo, hi, true); }

    struct Fns {
        double a, b, lo, hi, guess;
        double cdf(double k) const { return DiscreteUniform::cdf(k, a, b); }
        double ccdf(double k) const { return DiscreteUniform::ccdf(k, a, b); }
        double logcdf(double k) const { return DiscreteUniform::logcdf(k, a, b); }
        double logccdf(double k) const { return DiscreteUniform::logccdf(k, a, b); }
    };
    static Fns fns(double lo, double hi) { return {lo, hi, lo, hi - 1.0, std::floor(0.5 * lo + 0.5 * hi)}; }
    // The search starts at the closed form lo + ceil(p m) - 1 (m = hi - lo).
    static Fns at(double lo, double hi, double p) {
        return disc::with_guess(fns(lo, hi), lo + std::ceil(p * (hi - lo)) - 1.0);
    }
    static double icdf(double t, double lo, double hi) {
        return disc::settle<DiscreteUniform>(disc::Inv::icdf, t, disc::icdf(t, at(lo, hi, t)), nullptr, lo, hi);
    }
    static double iccdf(double t, double lo, double hi) {
        return disc::settle<DiscreteUniform>(disc::Inv::iccdf, t, disc::iccdf(t, at(lo, hi, 1.0 - t)), nullptr, lo, hi);
    }
    static double ilogcdf(double t, double lo, double hi) {
        return disc::settle<DiscreteUniform>(
            disc::Inv::ilogcdf, t, disc::ilogcdf(t, at(lo, hi, std::exp(t))), nullptr, lo, hi
        );
    }
    static double ilogccdf(double t, double lo, double hi) {
        return disc::settle<DiscreteUniform>(
            disc::Inv::ilogccdf, t, disc::ilogccdf(t, at(lo, hi, -std::expm1(t))), nullptr, lo, hi
        );
    }
    SD_DISC_MEMO_SCALAR(DiscreteUniform, 2, logcdf, true)  // a log each; the rest is arithmetic
    SD_DISC_MEMO_SCALAR(DiscreteUniform, 2, logccdf, true)

    // (low + high - 1) / 2, the sum of the halves exact as a double-double (past 2^53
    // high - 1 is not a float64, and low + high may cancel).
    static double mean(double lo, double hi) {
        const num::DD s = num::dd::two_sum(0.5 * lo, 0.5 * hi);
        return (s.hi - 0.5) + s.lo;
    }
    static double variance(double lo, double hi) {  // (m^2 - 1) / 12, m = hi - lo
        const double m = hi - lo;
        return (m - 1.0) * (m + 1.0) / 12.0;
    }
    static double skewness(double lo, double hi) { return hi - lo == 1.0 ? kNaN : 0.0; }
    static double kurtosis(double lo, double hi) {  // -6 (m^2 + 1) / (5 (m^2 - 1))
        const double m = hi - lo;
        if (m == 1.0) return kNaN;
        return -6.0 * (m * m + 1.0) / (5.0 * (m - 1.0) * (m + 1.0));
    }
    static double mode(double lo, double hi) { return hi - lo == 1.0 ? lo : kNaN; }  // every point (T21)
    static double entropy(double lo, double hi) { return std::log(hi - lo); }
    static double support_lo(double lo, double) { return lo; }
    static double support_hi(double, double hi) { return hi - 1.0; }
};

// ---------------------------------------------------------------- shared: sums over a support

namespace disc_sum {

using num::DD;

// The central moments m2, m3, m4 (sum (k - mu)^j pmf(k)) by summing outwards from the mode
// until the rest is negligible: past a term t of m4 with ratio r < 1 to the one before, the
// rest is about t r / (1 - r) (disc::entropy_run's rule). At most disc::kMaxMomentTerms terms:
// the moments are NaN if the sums have not ended by then, or the mode is past 2^52 (the walk
// needs its float64 neighbours).
struct Central {
    double m2, m3, m4;
};
template <class LogPmf>
Central central(double mu, double mode, double lo, double hi, LogPmf&& logpmf) {
    constexpr Central kNone{kNaN, kNaN, kNaN};
    Central c{0.0, 0.0, 0.0};
    auto add = [&](double k) {
        const num::DD l = logpmf(k);
        const num::DD wd = num::exp_full(l);  // the pmf to a rounding (its error is the moments')
        const double w = wd.hi + wd.lo, d = k - mu, d2 = d * d;
        c.m2 += w * d2;
        c.m3 += w * d2 * d;
        c.m4 += w * d2 * d2;
        return w * d2 * d2;
    };
    auto done = [&](double t, double prev) {
        const double r = prev > 0.0 ? t / prev : 0.0;
        return r < 1.0 && t * r <= (1.0 - r) * c.m4 * 0x1p-60;
    };
    const double m = std::clamp(std::floor(mode), lo, hi);
    if (!(std::fabs(m) < disc::kMaxWalkStart)) return kNone;
    int left = disc::kMaxMomentTerms;
    const double tm = add(m);
    double prev = tm;
    for (double k = m + 1.0; k <= hi; k += 1.0) {
        if (--left < 0) return kNone;
        const double t = add(k);
        if (k > mode + 1.0 && done(t, prev)) break;
        prev = t;
    }
    prev = tm;
    for (double k = m - 1.0; k >= lo; k -= 1.0) {
        if (--left < 0) return kNone;
        const double t = add(k);
        if (k < mode - 1.0 && done(t, prev)) break;
        prev = t;
    }
    return c;
}

// m3 / m2^(3/2) and m4 / m2^2 - 3; factor by factor where the denominator would leave the
// normal range (a variance below 2^-500: a distribution all but at one point).
inline double skewness_of(double m2, double m3) {
    const double d = m2 * std::sqrt(m2);
    return d >= 0x1p-1000 ? m3 / d : m3 / m2 / std::sqrt(m2);
}
inline double kurtosis_of(double m2, double m4) {
    const double d = m2 * m2;
    return d >= 0x1p-1000 ? m4 / d - 3.0 : m4 / m2 / m2 - 3.0;
}

// log P(X <= k) (upper: log P(X > k)) for a unimodal pmf with its peak at `mode`, as a
// double-double (exponentiated deep in a tail, a log of size 600 rounded to float64 would
// cost 70 ULP), lo <= k < hi: the side summed from its largest term outwards (the peak
// when the side holds it), so every run of terms falls and stops once negligible.
// factor(j, dir, end, ratio): log sum_i pmf(i) / pmf(j) over i = j, j + dir, ... up to end
// (disc::log_tail_factor, or a distribution's own for its long tails).
template <class LogPmf, class Up, class Down, class Factor>
DD log_side(
    double k, bool upper, double mode, double lo, double hi, LogPmf&& logpmf, Up&& up, Down&& down, Factor&& factor
) {
    const double m = std::clamp(std::floor(mode), lo, hi);
    if (!upper) {
        if (k <= m) return logpmf(k) + factor(k, -1.0, lo, down);
        const DD lm = logpmf(m), d = logpmf(m + 1.0) - lm;  // the peak m < k: [lo, m] and (m, k]
        const double a = factor(m, -1.0, lo, down);
        const double b = factor(m + 1.0, 1.0, k, up);
        return lm + std::log(std::exp(a) + std::exp(d.hi + d.lo + b));
    }
    const double s = k + 1.0;
    if (s >= m) return logpmf(s) + factor(s, 1.0, hi, up);
    const DD lm = logpmf(m), d = logpmf(m - 1.0) - lm;  // the peak m > s: [m, hi] and [s, m)
    const double a = factor(m, 1.0, hi, up);
    const double b = factor(m - 1.0, -1.0, s, down);
    return lm + std::log(std::exp(a) + std::exp(d.hi + d.lo + b));
}
template <class LogPmf, class Up, class Down>
DD log_side(double k, bool upper, double mode, double lo, double hi, LogPmf&& logpmf, Up&& up, Down&& down) {
    return log_side(k, upper, mode, lo, hi, logpmf, up, down, [](double j, double dir, double end, auto&& ratio) {
        return disc::log_tail_factor(j, dir, end, ratio);
    });
}

// P(X <= k) (upper: P(X > k)) or its log from side(upper) -> the log of a side summed
// directly: the side asked for when it is below 1/2, else 1 - the other side (then below
// 1/2: no cancellation either way). `small` guesses whether the side asked for is the
// small one (k below the mean for the lower side), so usually one sum suffices.
//
// A side may be NaN (a sum past its bound). The answer then comes from the other side where
// that settles it: a side below 1/2 gives both its own value and the complement's; a side
// above 1/2 gives its own probability (a sum of positive terms) but not its log, which near
// 0 would carry the absolute error of the log pmf it starts from. Anything else is NaN.
template <class Side>
double prob(bool upper, bool log, bool small, Side&& side) {
    constexpr double half = -num::kLn2.hi;
    auto direct = [&](DD l) { return log ? l.hi + l.lo : num::exp_dd(l); };
    auto other = [&](DD o) { return log ? std::log1p(-num::exp_dd(o)) : -std::expm1(o.hi + o.lo); };
    if (small) {
        const DD l = side(upper);
        if (l.hi < half) return direct(l);
        const DD o = side(!upper);
        if (o.hi < half || (l.hi == l.hi && o.hi == o.hi)) return other(o);
        return log ? kNaN : direct(l);  // the other side is NaN (or this one is)
    }
    const DD o = side(!upper);
    if (o.hi < half) return other(o);
    const DD l = side(upper);
    return o.hi == o.hi || !log || l.hi < half ? direct(l) : kNaN;
}

}  // namespace disc_sum

// ---------------------------------------------------------------- hypergeometric

namespace hyper_detail {

using U = unsigned __int128;

// C(n, k) exactly for n <= 120 (the running product stays below C(120, 60) 120 < 2^128).
inline U choose(double nd, double kd) {
    if (kd < 0.0 || kd > nd) return 0;
    const int n = static_cast<int>(nd);
    const int k = static_cast<int>(std::min(kd, nd - kd));
    U r = 1;
    for (int i = 1; i <= k; ++i) r = r * static_cast<U>(n - k + i) / static_cast<U>(i);
    return r;
}

// a / d for integers 0 <= a <= d, or its log: log1p(-(d - a) / d) above one half (the log
// near 0 relative to its own size).
inline double ratio(U a, U d, bool log) {
    if (!log) return static_cast<double>(a) / static_cast<double>(d);
    if (a == 0) return -kInf;
    if (2 * a > d) return std::log1p(-static_cast<double>(d - a) / static_cast<double>(d));
    return std::log(static_cast<double>(a) / static_cast<double>(d));
}

}  // namespace hyper_detail

struct Hypergeometric {
    // Up to this total the pmf and the cdfs are ratios of exact integers (counts of the
    // draws): correctly rounded up to 1.5 ULP, so the ties of the quantiles (cdf exactly
    // 1/2 and the like) come out exact.
    static constexpr double kExactN = 120.0;
    // A run of more terms than this that count is disc::smooth_tail_sum's (disc::tail_sum): where
    // the sum by terms (3.9 ns a term) and the smooth sum (~73 us) cost the same, 1.85e4 terms
    // (measured; both within 2e-15 of Arb there).
    static constexpr double kSmoothFrom = 0x1p14;

    static constexpr char name[] = "hypergeometric";
    static constexpr std::array<const char*, 3> params{"total", "successes", "draws"};

    static bool valid(double N, double K, double n) {
        return disc::is_int(N) && disc::is_int(K) && disc::is_int(n) && N >= 0.0 && K >= 0.0 && K <= N && n >= 0.0 &&
               n <= N;
    }
    static double lo_of(double N, double K, double n) { return std::max(0.0, n - (N - K)); }
    static double hi_of(double, double K, double n) { return std::min(n, K); }

    // log pmf at an integer k inside the support: dbinom(k; K, p) dbinom(n - k; N - K, p) /
    // dbinom(n; N, p) with p = n / N (any p would do: the identity holds for the rounded one).
    static num::DD log_pmf(double k, double N, double K, double n) {
        if (lo_of(N, K, n) == hi_of(N, K, n)) return {0.0, 0.0};  // one point
        const double p = n / N;
        const num::DD q = num::dd::two_sum(1.0, -p);
        return binom_detail::log_pmf(k, K, p, q) + binom_detail::log_pmf(n - k, N - K, p, q) -
               binom_detail::log_pmf(n, N, p, q);
    }
    // Past a total of disc::kMaxCount the counts K - k, n - k, N - K - n + k and their
    // neighbours are not all float64s: the pmf (and the cdfs, the quantiles, the entropy)
    // inside the support is NaN.
    static double logpmf(double k, double N, double K, double n) {
        if (!disc::is_int(k) || k < lo_of(N, K, n) || k > hi_of(N, K, n)) return -kInf;
        if (N > disc::kMaxCount) return kNaN;
        if (N <= kExactN) return exact_pmf(k, N, K, n, true);
        const num::DD l = log_pmf(k, N, K, n);
        return l.hi + l.lo;
    }
    static double pmf(double k, double N, double K, double n) {
        if (!disc::is_int(k) || k < lo_of(N, K, n) || k > hi_of(N, K, n)) return 0.0;
        if (N > disc::kMaxCount) return kNaN;
        if (N <= kExactN) return exact_pmf(k, N, K, n, false);
        return num::exp_dd(log_pmf(k, N, K, n));
    }
    static double exact_pmf(double k, double N, double K, double n, bool log) {
        using hyper_detail::choose;
        return hyper_detail::ratio(choose(K, k) * choose(N - K, n - k), choose(N, n), log);
    }
    static double mode_point(double N, double K, double n) { return std::floor((n + 1.0) * (K + 1.0) / (N + 2.0)); }
    // P(X <= x) (upper: P(X > x)) or its log.
    static double tail(double x, double N, double K, double n, bool upper, bool log) {
        const double k = std::floor(x), lo = lo_of(N, K, n), hi = hi_of(N, K, n);
        if (k < lo) return upper ? (log ? 0.0 : 1.0) : (log ? -kInf : 0.0);
        if (k >= hi) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
        if (N > disc::kMaxCount) return kNaN;  // no pmf to sum (logpmf)
        if (N <= kExactN) {
            using hyper_detail::choose;
            hyper_detail::U s = 0;
            for (double j = lo; j <= k; j += 1.0) s += choose(K, j) * choose(N - K, n - j);
            const hyper_detail::U d = choose(N, n);
            return hyper_detail::ratio(upper ? d - s : s, d, log);
        }
        // pmf(j + 1) / pmf(j) and pmf(j - 1) / pmf(j) as their exact factors (the counts)
        using num::DD;
        auto up = [&](double j) {
            return disc::Ratio4{DD{K - j, 0.0}, DD{n - j, 0.0}, DD{j + 1.0, 0.0}, DD{N - K - n + j + 1.0, 0.0}};
        };
        auto down = [&](double j) {
            return disc::Ratio4{DD{j, 0.0}, DD{N - K - n + j, 0.0}, DD{K - j + 1.0, 0.0}, DD{n - j + 1.0, 0.0}};
        };
        auto lp = [&](double j) { return log_pmf(j, N, K, n); };
        // (the pmf is log-concave)
        auto factor = [&](double j, double dir, double end, auto&& ratio) {
            return std::log(disc::tail_sum(j, dir, end, lo, hi, lp, ratio, true, kSmoothFrom));
        };
        auto side = [&](bool u) { return disc_sum::log_side(k, u, mode_point(N, K, n), lo, hi, lp, up, down, factor); };
        return disc_sum::prob(upper, log, (k < mean(N, K, n)) != upper, side);
    }
    static double logcdf(double x, double N, double K, double n) { return tail(x, N, K, n, false, true); }
    static double logccdf(double x, double N, double K, double n) { return tail(x, N, K, n, true, true); }
    static double cdf(double x, double N, double K, double n) { return tail(x, N, K, n, false, false); }
    static double ccdf(double x, double N, double K, double n) { return tail(x, N, K, n, true, false); }

    struct Fns {
        double N, K, n, lo, hi, guess;
        double cdf(double k) const { return Hypergeometric::cdf(k, N, K, n); }
        double ccdf(double k) const { return Hypergeometric::ccdf(k, N, K, n); }
        double logcdf(double k) const { return Hypergeometric::logcdf(k, N, K, n); }
        double logccdf(double k) const { return Hypergeometric::logccdf(k, N, K, n); }
    };
    static Fns fns(double N, double K, double n) {
        return {N, K, n, lo_of(N, K, n), hi_of(N, K, n), std::floor(n * K / N)};
    }
    static auto searcher(disc::Inv which, double t, disc::InvMemo* m, double N, double K, double n) {
        auto f = disc::Memoized(fns(N, K, n), m);
        const double sd = std::sqrt(variance(N, K, n));
        if (sd >= disc::kWideSd) disc::normal_start(f, which, t, mean(N, K, n), sd, skewness(N, K, n));
        return f;
    }
    SD_DISC_INVERSES(Hypergeometric, 3)
    // the value methods behind their memos (stats/discrete.hpp)
    // pmf and logpmf by the binomials' saddle-point fast path (discrete_basic.hpp, saddle::):
    // the three binomial pmfs of log_pmf, each as X_i = xh + xl and a factor,
    //     X = X_1 + X_2 - X_3 (two_sum),   r = K (N - K) n (N - n) / (2 pi N k (K - k) (n - k) (N - K - n + k)),
    // the pmf sqrt(r) e^X. Every count is >= 13 and every t within 1/8, or the element goes to
    // the scalar definition (as do totals up to kExactN, whose ratios of integers are exact).
    //   pmf: X within 0.5u (E_1 + E_2 + E_3) + 3.3u; ck::exp 2.5u; 1 + xl and two products
    //        3u; r nine products and quotients and the constant, 9.5u, sqrt: 5.75u. Together
    //        0.5 E + 14.6 units: the fast path keeps to E <= 2.5 (2.2 sd around the mean).
    //   logpmf = X + log(r) / 2 for r <= 1/2 (so log(r) / 2 <= -0.34 and X <= 0.03 add without
    //        cancelling to speak of): 0.5u E + 3.3u, (9.5u + 4u |log r|) / 2, the sums 2u |v|, with
    //        |v| >= E - 0.03 and >= |log r| / 2 - 0.03: at most 6.6 |v| + 8.1 units, 14.7 of
    //        max(1, |v|) at any E.
    struct FastParts {
        saddle::Parts q;
        bool ok;
    };
    static saddle::Parts third(double N, double n) { return saddle::binomial(n, N, n / N); }
    static saddle::Parts fast_parts(double k, double N, double K, double n, const saddle::Parts& c) {
        const double p = n / N;
        const saddle::Parts a = saddle::binomial(k, K, p), b = saddle::binomial(n - k, N - K, p);
        const num::DD s1 = num::dd::two_sum(a.xh, b.xh), s2 = num::dd::two_sum(s1.hi, -c.xh);
        const double xl = (s1.lo + s2.lo) + (a.xl + b.xl - c.xl);
        const double r =
            (K * (N - K)) * (n * (N - n)) / (2.0 * special::kPi * N * ((k * (K - k)) * ((n - k) * (N - K - n + k))));
        return {s2.hi, xl, r, a.e + b.e + c.e, a.ok && b.ok && c.ok && N > kExactN && r <= 0.5};
    }
    template <class F>
    static double scalar_at(const Chunk<3>& c, int i, F&& f) {
        return valid(c.p[0][i], c.p[1][i], c.p[2][i]) && c.x[i] == c.x[i] ? f(c.x[i], c.p[0][i], c.p[1][i], c.p[2][i])
                                                                          : kNaN;
    }
    template <class Run, class F>
    static void fast_kernel(const Chunk<3>& c, double* out, Run&& run, F&& scalar) {
        const bool same = c.same[0] && c.same[2];
        saddle::Parts t = third(c.p[0][0], c.p[2][0]);
        run(
            c.n, out,
            [&](int i) {
                if (!same) t = third(c.p[0][i], c.p[2][i]);
                return fast_parts(c.x[i], c.p[0][i], c.p[1][i], c.p[2][i], t);
            },
            [&](int i) { return scalar_at(c, i, scalar); }
        );
    }
    static void pmf_kernel(const Chunk<3>& c, double* out) {
        fast_kernel(
            c, out,
            [](int n, double* o, auto&& parts, auto&& precise) { saddle::pmf_kernel(n, o, parts, precise, 2.5); },
            &Hypergeometric::pmf
        );
    }
    static void logpmf_kernel(const Chunk<3>& c, double* out) {
        fast_kernel(
            c, out, [](int n, double* o, auto&& parts, auto&& precise) { saddle::logpmf_kernel(n, o, parts, precise); },
            &Hypergeometric::logpmf
        );
    }
    SD_DISC_MEMO_KERNEL(Hypergeometric, 3, pmf, false)
    SD_DISC_MEMO_KERNEL(Hypergeometric, 3, logpmf, false)
    SD_DISC_MEMO_SCALAR(Hypergeometric, 3, cdf, true)
    SD_DISC_MEMO_SCALAR(Hypergeometric, 3, ccdf, true)
    SD_DISC_MEMO_SCALAR(Hypergeometric, 3, logcdf, true)
    SD_DISC_MEMO_SCALAR(Hypergeometric, 3, logccdf, true)

    static double mean(double N, double K, double n) { return N == 0.0 ? 0.0 : n * (K / N); }
    static double variance(double N, double K, double n) {
        if (N <= 1.0) return 0.0;
        return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1.0));
    }
    static disc_sum::Central central(double N, double K, double n) {
        return disc_sum::central(mean(N, K, n), mode_point(N, K, n), lo_of(N, K, n), hi_of(N, K, n), [&](double j) {
            return log_pmf(j, N, K, n);
        });
    }
    // (N - 2K) (N - 2n) sqrt(N - 1) / ((N - 2) sqrt(n K (N - K) (N - n))): exactly 0 when
    // N = 2K or N = 2n (a sum of the central moments would leave 1e-16).
    static double skewness(double N, double K, double n) {
        if (lo_of(N, K, n) == hi_of(N, K, n)) return kNaN;
        if (N - 2.0 * K == 0.0 || N - 2.0 * n == 0.0) return 0.0;
        return (N - 2.0 * K) * (N - 2.0 * n) * std::sqrt(N - 1.0) /
               ((N - 2.0) * std::sqrt(n * K) * std::sqrt((N - K) * (N - n)));
    }
    // [(N - 1) N^2 (N (N + 1) - 6 K (N - K) - 6 n (N - n)) + 6 n K (N - K) (N - n) (5N - 6)]
    // / (n K (N - K) (N - n) (N - 2) (N - 3)): the bracket cancels (often to 0 exactly), so
    // it is formed in exact 128-bit integers, which hold it while N <= 2^24 (its terms
    // < 2^124); for N <= 3 the central moments as sums (at most 4 points). Past 2^24 it is
    // NaN: a sum of central moments loses what m4 / m2^2 - 3 cancels (the kurtosis is ~1 / sd^2).
    static double kurtosis(double N, double K, double n) {
        if (lo_of(N, K, n) == hi_of(N, K, n)) return kNaN;
        if (N > 0x1p24) return kNaN;
        if (N > 3.0) {
            using I = __int128;
            const I iN = static_cast<I>(N), iK = static_cast<I>(K), in = static_cast<I>(n);
            const I a = iK * (iN - iK), b = in * (iN - in);
            const I br = (iN - 1) * iN * iN * (iN * (iN + 1) - 6 * a - 6 * b) + 6 * a * b * (5 * iN - 6);
            return static_cast<double>(br) / (static_cast<double>(a) * static_cast<double>(b)) /
                   ((N - 2.0) * (N - 3.0));
        }
        const disc_sum::Central c = central(N, K, n);
        return c.m4 / (c.m2 * c.m2) - 3.0;
    }
    // floor((n + 1)(K + 1) / (N + 2)); two modes when that ratio is an integer (T25).
    // Past N = disc::kMaxCount (the limit of the pmf, kept for the mode): NaN.
    static double mode(double N, double K, double n) {
        if (lo_of(N, K, n) == hi_of(N, K, n)) return lo_of(N, K, n);
        if (N > disc::kMaxCount) return kNaN;
        const __int128 a = static_cast<__int128>(n + 1.0) * static_cast<__int128>(K + 1.0);
        const __int128 d = static_cast<__int128>(N + 2.0);
        if (a % d == 0) return kNaN;
        return static_cast<double>(a / d);
    }
    static double entropy(double N, double K, double n) {
        if (lo_of(N, K, n) == hi_of(N, K, n)) return 0.0;  // one point
        if (N > disc::kMaxCount) return kNaN;               // no pmf to sum (logpmf)
        if (!(variance(N, K, n) <= disc::kMaxEntropyVariance)) return kNaN;  // a sum past its bound
        return disc::entropy_sum(
            mode_point(N, K, n), lo_of(N, K, n), hi_of(N, K, n), [&](double j) { return log_pmf(j, N, K, n); },
            // pmf(j + 1) / pmf(j) = (K - j) (n - j) / ((j + 1) (N - K - n + j + 1)), the products
            // of two counts exact as double-doubles; one step down, its reciprocal at j - 1
            [&](double j, double dir) {
                using num::dd::two_prod;
                return dir > 0.0 ? two_prod(K - j, n - j) / two_prod(j + 1.0, N - K - n + j + 1.0)
                                 : two_prod(j, N - K - n + j) / two_prod(K - j + 1.0, n - j + 1.0);
            }
        );
    }
    static double support_lo(double N, double K, double n) { return lo_of(N, K, n); }
    static double support_hi(double N, double K, double n) { return hi_of(N, K, n); }
};

// ---------------------------------------------------------------- beta-binomial

namespace betabinom_detail {

using num::DD;

// log B(x, y) for x, y > 0 double-doubles.
inline DD log_beta(DD x, DD y) { return num::log_gamma(x) + num::log_gamma(y) - num::log_gamma(x + y); }

}  // namespace betabinom_detail

struct BetaBinomial {
    static constexpr char name[] = "betabinomial";
    static constexpr std::array<const char*, 3> params{"n", "a", "b"};

    static bool valid(double n, double a, double b) {
        return n >= 0.0 && disc::is_int(n) && a > 0.0 && std::isfinite(a) && b > 0.0 && std::isfinite(b);
    }
    // A shape below 2^-1021: its limit at 0 (user decision 2026-11-19, T50), the point mass at
    // 0 (a), at n (b), or both atoms with P(n) = a / (a + b) (stats/dist/two_point.hpp; the
    // moments and the entropy are those of the atoms, past has_moments' shape range).
    static bool at_limit(double n, double a, double b) {
        return valid(n, a, b) && (a < two_point::kMinShape || b < two_point::kMinShape);
    }
    template <class M>
    static double limit_in(double x, double n, double a, double b) {
        return two_point::in<M>(x, two_point::of_shapes(a, b, 0.0, n, true));
    }
    template <class M>
    static double limit_par(double n, double a, double b) {
        return two_point::par<M>(two_point::of_shapes(a, b, 0.0, n, true));
    }

    // Whether there is a pmf: n + a + b up to 2^35. The log pmf below is a difference of
    // log-gammas of that size, each to ~2^-90 of itself, so its error grows in proportion to
    // n + a + b: against Arb 1.2e-15 at 2^34 and 2.3e-15 at 2^35 (3.7e-15 with both shapes
    // there), doubling with each further power of 2. Beyond, the pmf, and with it the cdfs,
    // the quantiles and the entropy, are NaN rather than that.
    static bool has_pmf(double n, double a, double b) { return n + a + b <= 0x1p35; }
    // A run of more terms than this is disc::smooth_tail_sum's where it applies: where the sum by
    // terms (5.4 ns a term: two of the factors are double-double sums) and the smooth sum (45-90 us)
    // cost the same, 1.1e4-1.3e4 terms (measured; both within 5e-16 of Arb there).
    static constexpr double kSmoothFrom = 0x1.8p13;
    // Whether the moments' closed forms stay in the normal range: n, a, b and 1 / a, 1 / b up
    // to 2^150 (the kurtosis multiplies five of them). Outside, the moments are NaN.
    static bool has_moments(double n, double a, double b) {
        return n <= 0x1p150 && a >= 0x1p-150 && a <= 0x1p150 && b >= 0x1p-150 && b <= 0x1p150;
    }

    // log C(n, k) B(k + a, n - k + b) / B(a, b)
    //   = log Gamma(k + a) + log Gamma(n - k + b) - log Gamma(k + 1) - log Gamma(n - k + 1) + C,
    //   C = log Gamma(n + 1) - log Gamma(n + a + b) - log B(a, b),
    // C of the parameters alone (kept per thread for the last n, a, b): four log-gammas an
    // element (they were six, two of them of n + a + b and n + 2).
    static num::DD log_pmf(double k, double n, double a, double b) {
        using num::DD;
        using num::log_gamma;
        if (!has_pmf(n, a, b)) return {kNaN, 0.0};
        struct Last {
            double n = kNaN, a = kNaN, b = kNaN;
            DD c{0.0, 0.0};
        };
        thread_local Last last;
        if (!(last.n == n && last.a == a && last.b == b)) {
            last.c = log_gamma(DD{n + 1.0, 0.0}) - log_gamma(num::dd::two_sum(a, b) + n) -
                     betabinom_detail::log_beta(DD{a, 0.0}, DD{b, 0.0});
            last.n = n, last.a = a, last.b = b;
        }
        return log_gamma(num::dd::two_sum(k, a)) + log_gamma(num::dd::two_sum(n - k, b)) - log_gamma(DD{k + 1.0, 0.0}) -
               log_gamma(DD{n - k + 1.0, 0.0}) + last.c;
    }
    static double logpmf(double k, double n, double a, double b) {
        if (!disc::is_int(k) || k < 0.0 || k > n) return -kInf;
        const num::DD l = log_pmf(k, n, a, b);
        return l.hi + l.lo;
    }
    static double pmf(double k, double n, double a, double b) {
        if (!disc::is_int(k) || k < 0.0 || k > n) return 0.0;
        return num::exp_dd(log_pmf(k, n, a, b));
    }
    // The point where pmf(k + 1) / pmf(k) crosses 1: pmf(k + 1) >= pmf(k) exactly when
    // k (2 - a - b) >= n (1 - a) + b - 1 (the ratio's quadratic terms cancel).
    static double turn(double n, double a, double b) { return (n * (a - 1.0) + 1.0 - b) / (a + b - 2.0); }
    // The side split for the tail sums: the mode for a unimodal pmf (a + b > 2), else the
    // trough between the two ends (or the end the pmf runs to).
    static double split(double n, double a, double b) {
        if (a + b == 2.0) return a >= 1.0 ? n : 0.0;
        return std::clamp(std::floor(turn(n, a, b)) + 1.0, 0.0, n);
    }
    // P(X <= x) (upper: P(X > x)) or its log; not unimodal (a + b <= 2: U-shaped or
    // monotone), each side summed in full (the terms may rise again toward the end).
    static double tail(double x, double n, double a, double b, bool upper, bool log) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? (log ? 0.0 : 1.0) : (log ? -kInf : 0.0);
        if (k >= n) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
        if (!has_pmf(n, a, b)) return kNaN;  // no pmf to sum
        auto lp = [&](double j) { return log_pmf(j, n, a, b); };
        // the ratios as exact factors (every count is exact: n is below 2^35)
        // (log-concave for a, b >= 1; else the ratios may rise outwards, a power of the distance
        // to an end)
        auto factor = [&](double j, double dir, double end, auto&& ratio) {
            return std::log(disc::tail_sum(j, dir, end, 0.0, n, lp, ratio, a >= 1.0 && b >= 1.0, kSmoothFrom));
        };
        auto run = [&](auto&& up, auto&& down) {
            auto side = [&](bool u) -> num::DD {
                if (a + b > 2.0) return disc_sum::log_side(k, u, split(n, a, b), 0.0, n, lp, up, down, factor);
                if (!u) return lp(k) + full_sum(k, -1.0, 0.0, down, lp, n);
                return lp(k + 1.0) + full_sum(k + 1.0, 1.0, n, up, lp, n);
            };
            return disc_sum::prob(upper, log, (k < mean(n, a, b)) != upper, side);
        };
        using num::DD;
        using num::dd::two_sum;
        return run(
            [&](double j) {
                return disc::Ratio4{DD{n - j, 0.0}, two_sum(j, a), DD{j + 1.0, 0.0}, two_sum(n - j - 1.0, b)};
            },
            [&](double j) {
                return disc::Ratio4{DD{j, 0.0}, two_sum(n - j, b), DD{n - j + 1.0, 0.0}, two_sum(j - 1.0, a)};
            }
        );
    }
    // log sum_j pmf(j) / pmf(k) over every j from k to `end`: the ratio recurrence in
    // double-double (disc::tail_sum_dd); where a term underflows below 1e-200, the rounded
    // ratios resynchronized from the log pmf every 8 terms. NaN for more than
    // disc::kMaxTailTerms terms. The log as
    // a double-double: toward an end where the pmf rises (small shapes) the sum is far above
    // its first term, and its log rounded to float64 would cost |log| ULP of the probability.
    template <class R, class L>
    static num::DD full_sum(double k, double dir, double end, R&& ratio, L&& lp, double n) {
        constexpr num::DD kNone{kNaN, 0.0};
        if constexpr (std::is_same_v<std::invoke_result_t<R&, double>, disc::Ratio4>) {
            if (std::fabs(end - k) > kSmoothFrom) {  // a long run: the smooth sum, where it applies
                const double sum = disc::smooth_tail_sum(k, dir, end, 0.0, n, lp, ratio, true);
                if (sum == sum) return num::log_dd(sum);
            }
            const double sum = disc::tail_sum_dd(k, dir, end, ratio, true);
            if (!(sum < kInf)) return kNone;
            if (sum > 0.0) return num::log_dd(sum);
            // a term underflowed: the rounded ratios, resynchronized from the log pmf
            auto rounded = [&](double j) {
                const disc::Ratio4 r = ratio(j);
                return (r.a1.hi * r.a2.hi) / (r.b1.hi * r.b2.hi);
            };
            return full_sum(k, dir, end, rounded, lp, n);
        } else {
            // every term is needed: not formed for more than the bound, or past 2^53, where
            // the next integer is not a float64 and j would not move
            if (!(std::fabs(end - k) <= disc::kMaxTailTerms && std::fabs(k) < 0x1p53 && std::fabs(end) < 0x1p53))
                return kNone;
            const num::DD l0 = lp(k);
            double sum = 1.0, t = 1.0, j = k;
            for (int i = 1; (dir > 0.0 ? j < end : j > end); ++i) {
                t *= ratio(j);
                j += dir;
                if (i % 8 == 0) {
                    const num::DD d = lp(j) - l0;
                    t = std::exp(d.hi + d.lo);
                }
                sum += t;
            }
            return sum < kInf ? num::log_dd(sum) : kNone;
        }
    }
    static double logcdf(double x, double n, double a, double b) { return tail(x, n, a, b, false, true); }
    static double logccdf(double x, double n, double a, double b) { return tail(x, n, a, b, true, true); }
    static double cdf(double x, double n, double a, double b) { return tail(x, n, a, b, false, false); }
    static double ccdf(double x, double n, double a, double b) { return tail(x, n, a, b, true, false); }

    struct Fns {
        double n, a, b, lo = 0.0, hi, guess;
        double cdf(double k) const { return BetaBinomial::cdf(k, n, a, b); }
        double ccdf(double k) const { return BetaBinomial::ccdf(k, n, a, b); }
        double logcdf(double k) const { return BetaBinomial::logcdf(k, n, a, b); }
        double logccdf(double k) const { return BetaBinomial::logccdf(k, n, a, b); }
    };
    static Fns fns(double n, double a, double b) { return {n, a, b, 0.0, n, std::floor(n * a / (a + b))}; }
    static auto searcher(disc::Inv which, double t, disc::InvMemo* m, double n, double a, double b) {
        auto f = disc::Memoized(fns(n, a, b), m);
        const double sd = std::sqrt(variance_of(n, a, b));
        // a wide beta-binomial is n times a beta(a, b) variable smeared by a binomial: the
        // search starts at n times the beta quantile (sd.special's inverses)
        if (sd >= disc::kWideSd) {
            const bool log = which == disc::Inv::ilogcdf || which == disc::Inv::ilogccdf;
            const bool upper = which == disc::Inv::iccdf || which == disc::Inv::ilogccdf;
            const double pr = log ? std::exp(t) : t;
            const double g =
                std::floor(n * (upper ? special::Betainccinv::f(a, b, pr) : special::Betaincinv::f(a, b, pr)));
            if (g >= 0.0 && g <= n) f.guess = g;
        }
        return f;
    }
    SD_DISC_INVERSES(BetaBinomial, 3)
    // the value methods behind their memos (stats/discrete.hpp)
    SD_DISC_MEMO_SCALAR(BetaBinomial, 3, pmf, false)
    SD_DISC_MEMO_SCALAR(BetaBinomial, 3, logpmf, false)
    SD_DISC_MEMO_SCALAR(BetaBinomial, 3, cdf, true)
    SD_DISC_MEMO_SCALAR(BetaBinomial, 3, ccdf, true)
    SD_DISC_MEMO_SCALAR(BetaBinomial, 3, logcdf, true)
    SD_DISC_MEMO_SCALAR(BetaBinomial, 3, logccdf, true)

    static double mean(double n, double a, double b) { return has_moments(n, a, b) ? n * (a / (a + b)) : kNaN; }
    static double variance_of(double n, double a, double b) {
        const double c = a + b;
        return n * (a / c) * (b / c) * ((c + n) / (c + 1.0));
    }
    static double variance(double n, double a, double b) { return has_moments(n, a, b) ? variance_of(n, a, b) : kNaN; }
    // (a + b + 2n) (b - a) / (a + b + 2) sqrt((1 + a + b) / (n a b (n + a + b))): exactly 0
    // for a = b.
    static double skewness(double n, double a, double b) {
        if (n == 0.0 || !has_moments(n, a, b)) return kNaN;
        const double c = a + b;
        return (c + 2.0 * n) * (b - a) / (c + 2.0) * std::sqrt((1.0 + c) / (n * a)) / std::sqrt(b * (n + c));
    }
    // The excess kurtosis without the "- 3" (which cancels where it crosses 0): with s = a + b,
    // d = b - a,
    //     [d^2 A / 4 - s^2 B / 2] / (a b n (s + 2) (s + 3) (s + n)),
    //     A = 36 n^2 + 6 s^3 + s^2 (30 n + 6) + s (30 n^2 + 36 n),
    //     B = 6 n^2 + 3 n s^2 + s^3 + 3 n^2 s + 6 n s + 2 s + 3 s^2,
    // (s^2 - d^2 = 4ab put into the textbook form), the bracket in double-double from the
    // exact s and d, so only its own cancellation near the zero is left (2^-106 of it).
    static double kurtosis(double n, double a, double b) {
        if (n == 0.0 || !has_moments(n, a, b)) return kNaN;
        using num::DD;
        const DD s = num::dd::two_sum(a, b), d = num::dd::two_sum(b, -a), s2 = s * s, s3 = s2 * s;
        const DD nd{n, 0.0}, n2 = num::dd::two_prod(n, n);
        const DD A = s3 * 6.0 + s2 * (nd * 30.0 + 6.0) + s * (n2 * 30.0 + nd * 36.0) + n2 * 36.0;
        const DD B = s2 * (nd * 3.0 + 3.0) + s3 + s * (n2 * 3.0 + nd * 6.0 + 2.0) + n2 * 6.0;
        const DD br = d * d * A * 0.25 - s2 * B * 0.5;
        const double den = a * b * n * ((s.hi + 2.0) * (s.hi + 3.0) * (s.hi + n));
        return br.hi / den;
    }
    // The single largest point (T21); NaN for a tie (the crossing at an integer inside, equal
    // ends of a U-shaped pmf, or the uniform a = b = 1).
    //   a + b > 2: from the ratio's crossing (increasing while k <= t), NaN where the crossing
    //       is past the float64 range;
    //   a + b <= 2: the pmf falls to a trough and rises, or runs one way: the larger end, and
    //       pmf(n) / pmf(0) = prod_{i < n} (a + i) / (b + i) is above 1 exactly when a > b.
    static double mode(double n, double a, double b) {
        if (n == 0.0) return 0.0;
        if (a + b <= 2.0) return a == b ? kNaN : (a > b ? n : 0.0);
        const double t = turn(n, a, b);
        if (!std::isfinite(t)) return kNaN;
        if (t < 0.0) return 0.0;
        if (t >= n) return n;
        if (std::floor(t) == t) return kNaN;  // pmf(t + 1) = pmf(t)
        return std::floor(t) + 1.0;
    }
    static double entropy(double n, double a, double b) {
        if (n == 0.0) return 0.0;  // the point mass at 0 (T50: exactly 0, the sum's log pmf(0) is not)
        if (!has_pmf(n, a, b)) return kNaN;  // no pmf to sum
        auto lp = [&](double j) { return log_pmf(j, n, a, b); };
        // pmf(j + 1) / pmf(j) = (n - j) (j + a) / ((j + 1) (n - j - 1 + b)); one step down, its
        // reciprocal at j - 1
        auto step = [&](double j, double dir) {
            using num::dd::two_sum;
            return dir > 0.0 ? (two_sum(j, a) * (n - j)) / (two_sum(n - j - 1.0, b) * (j + 1.0))
                             : (two_sum(n - j, b) * j) / (two_sum(j - 1.0, a) * (n - j + 1.0));
        };
        if (a + b > 2.0) {
            if (!(variance_of(n, a, b) <= disc::kMaxEntropyVariance)) return kNaN;  // a sum past its bound
            return disc::entropy_sum(split(n, a, b), 0.0, n, lp, step);
        }
        // not unimodal: the pmf falls from each end to a trough, or all the way from one end
        // (a + b = 2: from n for a > 1, from 0 otherwise); every point is summed
        const double trough = a + b == 2.0 ? (a > 1.0 ? 0.0 : n) : split(n, a, b);
        return disc::entropy_sum_ends(trough, 0.0, n, lp, step);
    }
    static double support_lo(double, double, double) { return 0.0; }
    static double support_hi(double n, double, double) { return n; }
};

// ---------------------------------------------------------------- logarithmic series

namespace logser_detail {

using num::DD;

// L = -log(1 - p) (> 0), 1 - p exact as a double-double.
inline double big_l(double p) {
    const DD l = -num::dd::log(num::dd::two_sum(1.0, -p));
    return l.hi + l.lo;
}

// L / p - 1 = sum_{j >= 1} p^j / (j + 1): the series for p < 1/64 (L - p would underflow
// for a tiny p), else (L - p) / p with L - p = -p - log(1 - p) without cancellation.
inline double l_over_p_m1(double p) {
    if (p < 0x1p-6) {
        double s = 0.0, t = 1.0;
        for (int j = 1; j < 40; ++j) {
            t *= p;
            const double term = t / (j + 1);
            s += term;
            if (term <= s * 0x1p-60) break;
        }
        return s;
    }
    const DD m = num::dd::log1pmx_neg(DD{-p, 0.0});
    return (m.hi + m.lo) / p;
}

inline constexpr int kExpE1Terms = 1000;  // the most terms of exp_e1's continued fraction

// e^x E_1(x), x > 0: the continued fraction 1 / (x + 1 - 1 / (x + 3 - 4 / (x + 5 - ...)))
// for x > 1 (sd.special's E_n fraction, which is this scaled value before e^-x), else
// e^x times the series.
inline double exp_e1(double x) {
    if (x > 1.0) {
        // the fraction's error after n terms falls like e^(-4 sqrt(n x)): 2^-53 by n = 85 / x
        const auto r = special::lentz_backward(
            x + 1.0,
            [&](int k) {
                const double kd = k;
                return std::pair{-kd * kd, x + 1.0 + 2.0 * kd};
            },
            kExpE1Terms
        );
        return r.converged ? 1.0 / r.value : kNaN;
    }
    return std::exp(x) * special::Expn::f(1.0, x);
}

// B_2j / (2j)!, j = 1..10.
// clang-format off
inline constexpr double kBernoulliOverFactorial[10] = {
    1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0, -1.0 / 1209600.0, 1.0 / 47900160.0,
    -691.0 / 1307674368000.0, 1.0 / 74724249600.0, -3617.0 / 10670622842880000.0,
    43867.0 / 5109094217170944000.0, -174611.0 / 802857662698291200000.0};
// clang-format on

// Phi(p, 1, v) = sum_{m >= 0} p^m / (v + m) for lw = -log p <= 0.1 by Euler-Maclaurin: the
// first terms directly until V = v + M >= 20, then with f(x) = e^(-lw x) / (V + x)
//     sum_{m >= 0} f(m) = int_0^inf f + f(0) / 2 - sum_j B_2j / (2j)! f^(2j-1)(0),
//     int_0^inf f = e^(V lw) E_1(V lw),
//     f^(n)(0) = (-1)^n n! / V sum_{i <= n} lw^(n-i) / (n-i)! / V^i  (terms of one sign),
// the corrections falling like (h / 2 pi)^2j (2j)!, h = max(lw, 1 / V) <= 0.1: ten reach
// 2^-60 (the 2F1 transformation cancels like e^(v w) here, the direct series needs
// 42 / lw terms).
inline double phi_em(double p, double lw, double v) {
    double s = 0.0, t = 1.0, V = v;
    while (V < 20.0) {
        s += t / V;
        t *= p;
        V += 1.0;
    }
    double tail = exp_e1(V * lw) + 0.5 / V;
    const double r = 1.0 / V;
    double nfact = 1.0;  // n!
    for (int j = 1; j <= 10; ++j) {
        const int n = 2 * j - 1;
        nfact *= (j == 1 ? 1.0 : double(n) * (n - 1));
        double poly = 0.0, c = 1.0;  // sum_{q <= n} lw^q / q! r^(n-q); c = lw^q / q!
        for (int q = 0; q <= n; ++q) {
            poly += c * std::pow(r, n - q);
            c *= lw / (q + 1);
        }
        const double d = -nfact * r * poly;  // f^(n)(0), n odd
        const double corr = kBernoulliOverFactorial[j - 1] * d;
        tail -= corr;
        if (std::fabs(corr) <= tail * 0x1p-60) break;
    }
    return s + t * tail;
}

// log T(k) for T(k) = sum_{j > k} p^j / j = p^(k+1) Phi(p, 1, k + 1): log p^(k+1) plus the log
// of Phi, by its series (terms p^m / (v + m), 42 / lw of them) for p < 0.9 (lw > 0.1) and
// by Euler-Maclaurin (phi_em) above.
inline DD log_tail_sum(double k, double p) {
    const double v = k + 1.0;
    const double lw = -std::log(p);
    if (v >= disc::kFarCount && disc::overflows(lw, v)) return {-kInf, 0.0};  // p^(k+1) is below any float64's log
    double phi;
    if (lw <= 0.1) {
        phi = phi_em(p, lw, v);
    } else {
        // p^m <= e^(-m / 10) is below 2^-60 of the first term from m = 416
        double s = 0.0, t = 1.0;
        bool ended = false;
        for (int m = 0; m < 512 && !ended; ++m) {
            const double term = t / (v + m);
            s += term;
            ended = term <= s * 0x1p-60;
            t *= p;
        }
        phi = ended ? s : kNaN;
    }
    return num::log_dd(p) * v + std::log(phi);
}

}  // namespace logser_detail

struct LogSeries {
    static constexpr char name[] = "logseries";
    static constexpr std::array<const char*, 1> params{"p"};

    static bool valid(double p) { return p > 0.0 && p < 1.0; }

    // log(-p^k / (k log(1 - p))) = (k - 1) log p - log k - log(L / p): log L would cancel
    // against log p at k = 1 (the log pmf there is -p/2 for a small p).
    static double log_l_over_p(double p) {
        thread_local num::Memo memo;
        return memo.get(p, 0.0, [&] { return num::DD{std::log1p(logser_detail::l_over_p_m1(p)), 0.0}; }).hi;
    }
    static num::DD log_pmf(double k, double p) {
        const num::DD lp = num::log_dd(p);
        if (k >= disc::kFarCount && disc::overflows(lp.hi, k - 1.0)) return {-kInf, 0.0};
        return lp * (k - 1.0) - num::log_dd(k) - log_l_over_p(p);
    }
    static double logpmf(double k, double p) {
        if (!disc::is_int(k) || k < 1.0) return -kInf;
        const num::DD l = log_pmf(k, p);
        return l.hi + l.lo;
    }
    static double pmf(double k, double p) {
        if (!disc::is_int(k) || k < 1.0) return 0.0;
        return num::exp_dd(log_pmf(k, p));
    }
    // log P(X > k) = log T(k) - log L; P(X <= k) directly as a sum for small k, else 1 - that.
    // log P(X > k) as a double-double: exponentiated, a log of size 460 rounded to float64
    // would cost 50 ULP.
    static num::DD log_ccdf_dd(double k, double p) {
        const num::DD t = logser_detail::log_tail_sum(k, p);
        if (t.hi == -kInf) return t;
        return t - num::log_dd(p) - log_l_over_p(p);
    }
    static double logccdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (std::isinf(k)) return -kInf;
        const num::DD l = log_ccdf_dd(k, p);
        return l.hi + l.lo;
    }
    static double ccdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 1.0;
        if (std::isinf(k)) return 0.0;
        return num::exp_dd(log_ccdf_dd(k, p));
    }
    static double cdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (std::isinf(k)) return 1.0;
        const double c = ccdf(x, p);
        if (c <= 0.5 || k > 1000.0) return 1.0 - c;
        double s = 0.0;  // sum_{j <= k} pmf(j), k <= 1000
        for (double j = 1.0; j <= k; j += 1.0) s += pmf(j, p);
        return s;
    }
    static double logcdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return -kInf;
        if (std::isinf(k)) return 0.0;
        const double c = ccdf(x, p);
        return c <= 0.5 ? std::log1p(-c) : std::log(cdf(x, p));
    }

    struct Fns {
        double p, lo = 1.0, hi = kInf, guess = 1.0;
        double cdf(double k) const { return LogSeries::cdf(k, p); }
        double ccdf(double k) const { return LogSeries::ccdf(k, p); }
        double logcdf(double k) const { return LogSeries::logcdf(k, p); }
        double logccdf(double k) const { return LogSeries::logccdf(k, p); }
    };
    static Fns fns(double p) { return {p}; }
    static auto searcher(disc::Inv, double, disc::InvMemo* m, double p) { return disc::Memoized(fns(p), m); }
    SD_DISC_INVERSES(LogSeries, 1)
    // the value methods behind their memos (stats/discrete.hpp)
    // pmf = exp((k - 1) log p - log(L / p)) / k, the exponent in double-double with its two
    // logs taken once per chunk for a scalar p (by the functions the per-element path calls):
    // ck::exp 2.5u, the factor 1 + lo, the product and the division 3u: 5.5 units (no
    // log k: the scalar definition's third log). logpmf = exponent - log k with vForce's
    // log: both parts are negative, so the 4u of log k and the two sums' 2u are relative to
    // the result: 6 units.
    static void pmf_kernel(const Chunk<1>& c, double* out) { exponent_kernel(c, out, false); }
    static void logpmf_kernel(const Chunk<1>& c, double* out) { exponent_kernel(c, out, true); }
    static void exponent_kernel(const Chunk<1>& c, double* out, bool log) {
        double eh[kChunkSize], el[kChunkSize], w[kChunkSize];
        bool fast[kChunkSize];
        const bool same = c.same[0];
        const bool ok0 = c.p[0][0] > 0.0 && c.p[0][0] < 1.0;
        num::DD lp = num::log_dd(same && ok0 ? c.p[0][0] : 0.5);
        double llp = same && ok0 ? log_l_over_p(c.p[0][0]) : 0.0;
        for (int i = 0; i < c.n; ++i) {
            const double k = c.x[i], p = c.p[0][i];
            fast[i] = p > 0.0 && p < 1.0 && k >= 1.0 && std::isfinite(k) && std::floor(k) == k;
            eh[i] = el[i] = 0.0;
            w[i] = 1.0;
            if (!fast[i]) continue;
            if (!same) lp = num::log_dd(p), llp = log_l_over_p(p);
            const num::DD e = lp * (k - 1.0) - llp;
            // below: the scalar definition (results into the subnormals, k up to 2^53 times smaller)
            fast[i] = e.hi > -650.0 && k < 0x1p53;
            if (fast[i]) eh[i] = e.hi, el[i] = e.lo, w[i] = k;
        }
        if (log) {
            ck::log(w, w, c.n);
            for (int i = 0; i < c.n; ++i) out[i] = (eh[i] - w[i]) + el[i];
        } else {
            double y[kChunkSize];
            ck::exp(y, eh, c.n);
            for (int i = 0; i < c.n; ++i) out[i] = y[i] * (1.0 + el[i]) / w[i];
        }
        ck::fallback(c.n, fast, out, [&](int i) {
            if (!valid(c.p[0][i]) || c.x[i] != c.x[i]) return kNaN;
            return log ? LogSeries::logpmf(c.x[i], c.p[0][i]) : LogSeries::pmf(c.x[i], c.p[0][i]);
        });
    }
    SD_DISC_MEMO_KERNEL(LogSeries, 1, pmf, false)
    SD_DISC_MEMO_KERNEL(LogSeries, 1, logpmf, false)
    SD_DISC_MEMO_SCALAR(LogSeries, 1, cdf, true)
    SD_DISC_MEMO_SCALAR(LogSeries, 1, ccdf, true)
    SD_DISC_MEMO_SCALAR(LogSeries, 1, logcdf, true)
    SD_DISC_MEMO_SCALAR(LogSeries, 1, logccdf, true)

    // The raw moments E[X^r] = p A_r / (L w^r), A = 1, 1, 1 + p, 1 + 4p + p^2 (w = 1 - p).
    static double mean(double p) { return p / (logser_detail::big_l(p) * (1.0 - p)); }
    // p (L - p) / (w^2 L^2), L - p = mu(-p) = -p - log(1 - p) without cancellation.
    static double variance(double p) {
        const double L = logser_detail::big_l(p), w = 1.0 - p;
        const num::DD lmp = num::dd::log1pmx_neg(num::DD{-p, 0.0});
        return p * (lmp.hi + lmp.lo) / (w * w * L * L);
    }
    // For p <= 1/2 the central moments as sums (the raw moments cancel there: X is mostly
    // 1); above, from the raw moments (L is large: no cancellation).
    static disc_sum::Central central(double p) {
        if (p <= 0.5) return disc_sum::central(mean(p), 1.0, 1.0, kInf, [&](double j) { return log_pmf(j, p); });
        const double L = logser_detail::big_l(p), w = 1.0 - p;
        const double e1 = p / (L * w), e2 = p / (L * w * w), e3 = p * (1.0 + p) / (L * w * w * w),
                     e4 = p * (1.0 + p * (4.0 + p)) / (L * w * w * w * w);
        const double m2 = e2 - e1 * e1, m3 = e3 - 3.0 * e1 * e2 + 2.0 * e1 * e1 * e1,
                     m4 = e4 - 4.0 * e1 * e3 + 6.0 * e1 * e1 * e2 - 3.0 * e1 * e1 * e1 * e1;
        return {m2, m3, m4};
    }
    static double skewness(double p) {
        const disc_sum::Central c = central(p);
        return disc_sum::skewness_of(c.m2, c.m3);
    }
    static double kurtosis(double p) {
        const disc_sum::Central c = central(p);
        return disc_sum::kurtosis_of(c.m2, c.m4);
    }
    static double mode(double) { return 1.0; }
    // H = w E[X] + E[log X] + log L with w = -log p, E[X] = p / ((1 - p) L) and
    // E[log X] = T / L, T = sum_k p^k log(k) / k.
    //
    // Above w = 2^-10 the sum over the support from k = 1 (terms falling like p^k: at most
    // 5e4 of them). Below, T = -d/ds Li_s(e^-w) at s = 1 from the polylogarithm's expansion at
    // w = 0, Li_s(e^-w) = Gamma(1 - s) w^(s-1) + sum_{n >= 0} zeta(s - n) (-w)^n / n! (|w| < 2 pi),
    // whose two poles at s = 1 cancel:
    //     T = log(w)^2 / 2 + gamma log(w) + C - sum_{n >= 1} zeta'(1 - n) (-w)^n / n!,
    //     C = gamma^2 / 2 + pi^2 / 12 + gamma_1 (gamma_1: the first Stieltjes constant),
    // the terms of one size and sign pattern log(w)^2 / 2 >> gamma |log w| (no cancellation to
    // speak of), the series cut after n = 5: from n = 6 on |zeta'(1 - n)| / n! < (2 pi)^-n
    // (8e-7 at n = 6), so the rest is below 2 (w / 2 pi)^6 < 2^-75. The direct sum would need
    // 42 / w terms.
    static double entropy(double p) {
        const num::DD lp = num::log_dd(p);
        const double w = -(lp.hi + lp.lo);
        if (w > 0x1p-10) {
            return disc::entropy_sum(
                1.0, 1.0, kInf, [&](double j) { return log_pmf(j, p); },
                // pmf(k + 1) / pmf(k) = p k / (k + 1)
                [&](double k, double dir) {
                    return dir > 0.0 ? num::dd::two_prod(p, k) / (k + 1.0)
                                     : num::DD{k, 0.0} / (num::dd::two_prod(p, k - 1.0));
                }
            );
        }
        constexpr double kEuler = 0.5772156649015329, kC = 0.9162401498442958;
        // zeta'(0), -zeta'(-1) / 2!, zeta'(-2) / 3!, -zeta'(-3) / 4!, zeta'(-4) / 5!
        constexpr double c1 = -0.9189385332046727, c2 = 0.08271057185022546, c3 = -0.005074742843065545,
                         c4 = -0.00022410734824059588, c5 = 6.65317620855718690e-05;
        // log w and L as double-doubles: T / L is most of H, and a float64 log w would leave
        // its rounding twice over in log(w)^2
        const num::DD l = num::log_dd(w), L = -num::dd::log(num::dd::two_sum(1.0, -p));
        const num::DD t = l * l * 0.5 + l * kEuler + (kC + w * (c1 + w * (c2 + w * (c3 + w * (c4 + w * c5)))));
        const num::DD h = t / L + num::dd::log(L) + w * p / ((1.0 - p) * L.hi);
        return h.hi + h.lo;
    }
    static double support_lo(double) { return 1.0; }
    static double support_hi(double) { return kInf; }
};

// ---------------------------------------------------------------- Zipf

namespace zipf_detail {

using num::DD;

inline double zeta(double a) {
    thread_local num::Memo memo;
    return memo.get(a, 0.0, [&] { return DD{special::Zeta::f(a), 0.0}; }).hi;
}

// log zeta(a): log1p(zeta(a, 2)) while zeta(a) - 1 < 1/2 (log zeta is ~2^-a there), else
// the log of zeta(a) in double-double.
inline DD log_zeta(double a) {
    const double zm1 = special::Hurwitz::f(a, 2.0);
    return zm1 < 0.5 ? DD{std::log1p(zm1), 0.0} : num::log_dd(zeta(a));
}

// log zeta(a, q) for q >= 1: the log of sd.special's Hurwitz zeta while it is a normal
// number; past that -a log q + log sum_j (1 + j/q)^-a (the terms fall fast: a/q is large
// whenever zeta(a, q) underflows at a moderate q), or for a large q the Euler-Maclaurin
// form q^(1-a)/(a-1) (1 + (a-1)/(2q) + (a-1) a / (12 q^2) - ...).
inline DD log_hurwitz(double a, double q) {
    const double z = special::Hurwitz::f(a, q);
    if (z >= 0x1p-1000) return num::log_dd(z);
    if (a / q >= 1.0) {
        // The terms pass 2^-60 at j = q (2^(60 / a) - 1) <= a (2^(60 / a) - 1), which falls in
        // a, and zeta(a, q) < 2^-1000 with q <= a needs a >= 140: at most 49 terms.
        double s = 0.0;
        bool ended = false;
        for (int j = 0; j < 128 && !ended; ++j) {
            const double t = std::exp(-a * std::log1p(j / q));
            s += t;
            ended = t <= s * 0x1p-60;
        }
        return ended ? num::log_dd(q) * -a + std::log(s) : num::DD{kNaN, 0.0};
    }
    const double r = 1.0 / q;
    const double c = 1.0 + (a - 1.0) * r * (0.5 + a * r * (1.0 / 12.0 - (a + 1.0) * (a + 2.0) * r * r / 720.0));
    return num::log_dd(q) * (1.0 - a) + (std::log(c) - std::log(a - 1.0));
}

// sum_{j <= k} j^-a for k > 1000 where zeta(a) - zeta(a, k + 1) cancels (a near 1): the
// terms to 1000, then zeta(a, M) - zeta(a, K) (M = 1001, K = k + 1) by the Euler-Maclaurin
// form of each, zeta(a, q) = q^(1-a) / (a-1) + q^-a / 2 + sum_j B_2j / (2j)! (a)_(2j-1) q^(-a-2j+1),
// the leading difference as M^(1-a) (1 - (K/M)^(1-a)) / (a - 1) with expm1 (no
// cancellation), the rest small.
inline double partial_sum(double a, double k) {
    constexpr double M = 1001.0;
    const double K = k + 1.0;
    double s = 0.0;
    for (double j = M - 1.0; j >= 1.0; j -= 1.0) s += std::pow(j, -a);
    const double am1 = a - 1.0;
    const double lead = std::exp(-am1 * std::log(M)) * -std::expm1(-am1 * std::log(K / M)) / am1;
    auto pw = [&](double q, double e) { return std::exp(e * std::log(q)); };
    double rest = 0.5 * (pw(M, -a) - pw(K, -a));
    constexpr double c[3] = {1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0};
    double poch = a;  // (a)_(2j-1)
    for (int j = 1; j <= 3; ++j) {
        const double e = -a - 2.0 * j + 1.0;
        rest += c[j - 1] * poch * (pw(M, e) - pw(K, e));
        poch *= (a + 2.0 * j - 1.0) * (a + 2.0 * j);
    }
    return s + (lead + rest);
}

// sum_{k >= 1} (k - mu)^m k^-a / zeta(a) for m = 2, 3, 4: the terms up to K, then the tail
// sum_{k > K} (k - mu)^m k^-a = sum_j C(m, j) (-mu)^(m-j) zeta(a - j, K + 1) (mu / K small:
// no cancellation to speak of). Requires a > m + 1 for the m-th.
inline double central(int m, double a, double mu) {
    constexpr double K = 1000.0;
    disc::KahanSum sum;
    for (double k = 1.0; k <= K; k += 1.0) sum.add(std::pow(k - mu, m) * std::pow(k, -a));
    const double s = sum.value();
    double tail = 0.0, c = 1.0;  // C(m, j), j from m down
    for (int j = m; j >= 0; --j) {
        tail += c * std::pow(-mu, m - j) * special::Hurwitz::f(a - j, K + 1.0);
        c = c * j / (m - j + 1);
    }
    return (s + tail) / zeta(a);
}

}  // namespace zipf_detail

struct Zipf {
    static constexpr char name[] = "zipf";
    static constexpr std::array<const char*, 1> params{"a"};

    static bool valid(double a) { return a > 1.0 && std::isfinite(a); }

    static num::DD log_pmf(double k, double a) {
        thread_local num::Memo memo;
        const num::DD lz = memo.get(a, 0.0, [&] { return zipf_detail::log_zeta(a); });
        return num::log_dd(k) * -a - lz;
    }
    static double logpmf(double k, double a) {
        if (!disc::is_int(k) || k < 1.0) return -kInf;
        const num::DD l = log_pmf(k, a);
        return l.hi + l.lo;
    }
    static double pmf(double k, double a) {
        if (!disc::is_int(k) || k < 1.0) return 0.0;
        return num::exp_dd(log_pmf(k, a));
    }
    // P(X > k) = zeta(a, k + 1) / zeta(a); P(X <= k) as the sum for k <= 1000, else
    // (zeta(a) - zeta(a, k + 1)) / zeta(a) (which then cancels at most a little).
    static double logccdf(double x, double a) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (std::isinf(k)) return -kInf;
        const num::DD l = log_ccdf_dd(k, a);
        return l.hi > -num::kLn2.hi ? std::log1p(-cdf(x, a)) : l.hi + l.lo;  // near 0: from the small side
    }
    static num::DD log_ccdf_dd(double k, double a) {
        thread_local num::Memo memo;
        const num::DD lz = memo.get(a, 0.0, [&] { return zipf_detail::log_zeta(a); });
        return zipf_detail::log_hurwitz(a, k + 1.0) - lz;
    }
    static double ccdf(double x, double a) {
        const double k = std::floor(x);
        if (k < 1.0) return 1.0;
        if (std::isinf(k)) return 0.0;
        const double z = special::Hurwitz::f(a, k + 1.0);
        return z >= 0x1p-1000 ? z / zipf_detail::zeta(a) : num::exp_dd(log_ccdf_dd(k, a));
    }
    static double cdf(double x, double a) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (std::isinf(k)) return 1.0;
        if (k <= 1000.0) {
            double s = 0.0;
            for (double j = k; j >= 1.0; j -= 1.0) s += std::pow(j, -a);  // small terms first
            return s / zipf_detail::zeta(a);
        }
        const double z = zipf_detail::zeta(a), h = special::Hurwitz::f(a, k + 1.0);
        if (h <= 0.25 * z) return (z - h) / z;  // cancels at most 4/3
        return zipf_detail::partial_sum(a, k) / z;
    }
    static double logcdf(double x, double a) {
        const double c = ccdf(x, a);
        return c <= 0.5 ? std::log1p(-c) : std::log(cdf(x, a));
    }

    struct Fns {
        double a, lo = 1.0, hi = kInf, guess = 1.0;
        double cdf(double k) const { return Zipf::cdf(k, a); }
        double ccdf(double k) const { return Zipf::ccdf(k, a); }
        double logcdf(double k) const { return Zipf::logcdf(k, a); }
        double logccdf(double k) const { return Zipf::logccdf(k, a); }
        num::DD logccdf_dd(double k) const { return Zipf::log_ccdf_dd(k, a); }  // k >= 1, an integer
    };
    static Fns fns(double a) { return {a}; }
    static auto searcher(disc::Inv, double, disc::InvMemo* m, double a) { return disc::Memoized(fns(a), m); }
    SD_DISC_INVERSES(Zipf, 1)
    // the value methods behind their memos (stats/discrete.hpp)
    SD_DISC_MEMO_SCALAR(Zipf, 1, pmf, false)
    SD_DISC_MEMO_SCALAR(Zipf, 1, logpmf, false)
    SD_DISC_MEMO_SCALAR(Zipf, 1, cdf, true)
    SD_DISC_MEMO_SCALAR(Zipf, 1, ccdf, true)
    SD_DISC_MEMO_SCALAR(Zipf, 1, logcdf, true)
    SD_DISC_MEMO_SCALAR(Zipf, 1, logccdf, true)

    // zeta(a - 1) / zeta(a) for a > 2 (else +inf); the central moments by zipf_detail::central;
    // a ratio is +inf where the variance is finite and its moment diverges, NaN where the
    // variance is not finite (the rule of docs/api/stats.md §2).
    static double mean(double a) { return a > 2.0 ? special::Zeta::f(a - 1.0) / zipf_detail::zeta(a) : kInf; }
    static double variance(double a) {
        if (!(a > 3.0)) return a > 2.0 ? kInf : kNaN;
        return zipf_detail::central(2, a, mean(a));
    }
    static double skewness(double a) {
        if (!(a > 4.0)) return a > 3.0 ? kInf : kNaN;
        const double mu = mean(a), m2 = zipf_detail::central(2, a, mu);
        return disc_sum::skewness_of(m2, zipf_detail::central(3, a, mu));
    }
    static double kurtosis(double a) {
        if (!(a > 5.0)) return a > 3.0 ? kInf : kNaN;
        const double mu = mean(a), m2 = zipf_detail::central(2, a, mu);
        return disc_sum::kurtosis_of(m2, zipf_detail::central(4, a, mu));
    }
    static double mode(double) { return 1.0; }
    // log zeta(a) + a S / zeta(a), S = sum log k k^-a: the terms to K and the Euler-Maclaurin
    // tail with f(x) = log x x^-a: int_K^inf f - f(K)/2 - f'(K)/12 + f'''(K)/720.
    static double entropy(double a) {
        constexpr double K = 10000.0;
        disc::KahanSum sum;
        for (double k = K - 1.0; k >= 2.0; k -= 1.0) sum.add(std::log(k) * std::pow(k, -a));  // small terms first
        double s = sum.value();
        const double lk = std::log(K), pk = std::pow(K, -a);
        const double integral = K * pk * (lk / (a - 1.0) + 1.0 / ((a - 1.0) * (a - 1.0)));
        const double f = lk * pk;
        const double f1 = pk / K * (1.0 - a * lk);
        const double f3 =
            pk / (K * K * K) * (a * (a + 1.0) + (a + 2.0) * (2.0 * a + 1.0) - a * (a + 1.0) * (a + 2.0) * lk);
        s += integral + 0.5 * f - f1 / 12.0 + f3 / 720.0;  // sum_{k >= K} = int + f(K)/2 - f'(K)/12 + f'''(K)/720
        const double z = zipf_detail::zeta(a);
        return std::log(z) + a * s / z;
    }
    static double support_lo(double) { return 1.0; }
    static double support_hi(double) { return kInf; }
};

}  // namespace sd::stats
