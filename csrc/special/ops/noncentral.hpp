// The noncentral distributions (docs/api/special.md §14; T2): chi-square
// (chndtr, chndtrc, chndtrix), Student t (nctdtr, nctdtrc, nctdtrit) and F
// (ncfdtr, ncfdtrc, ncfdtri). x comes last in every signature (T2 decision;
// scipy's chndtr takes x first). The *c functions are the upper tails, summed
// directly (scipy has none).
//
// Special values follow the true value or limit (T1/T2 decision): nc = 0 is
// the central function exactly; an infinite parameter is the limit
// distribution; the differences from scipy are in docs/design/special.md §2.4.

#pragma once

#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/cf.hpp"
#include "special/ops/distr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace sd::special {

namespace nc_detail {

// ---- Poisson mixtures (chi-square, F, and the noncentral t's series)
//
// S = sum_j w_j G(s0 + j), w_j = D(j + h, mu) = mu^(j+h) e^-mu / Gamma(j+h+1) (h = 0:
// the Poisson weights; h = 1/2: the t's second family), G a family of incomplete
// functions in the shape s with T(s) = |G(s) - G(s+1)| (the prefix): a lower family
// (P, I_x: G(s) = G(s+1) + T(s)) or an upper one (Q, 1 - I_x: G(s+1) = G(s) + T(s)).
// Every term is positive, so both tails keep their relative accuracy.
//
// The recurrence for G is stable only in the direction that adds T, so one walk
// covers every term: from a far end J on the other side (direct values of w, G and
// T there, the only ones), across the largest terms, to where the terms fade (see
// mixture). w and T follow their ratios in double-double, so rounding cannot drift.
// The Boost.Math versions (what scipy 1.18 calls) were measured first and lose the
// far tails (zeros, up to 8.8e15 ULP): not used (docs/design/special.md §5.16).
//
// In the bulk (few terms, values inside a double's range) the same sums are taken from
// j = 0 instead, without the far end's incomplete function (`lite` below); the probability
// functions try that first, the wide tails (sd.stats' log forms and inverses) never do.

inline constexpr double kTol = 0x1p-56;

// Every loop here ends within a fixed number of steps: a sum takes at most kMaxTerms terms
// (its walk twice that), and one that has not ended by then gives NaN, never the part
// summed so far. Inside the guaranteed region (nc <= 1e4, |nc| <= 100 for the t) the
// longest sum is the walk across the terms' bell, 9 widths out and 18 back. Counted over the
// reference points of the region (tests/special/ref, tests/stats/ref; 2026-11-19): the
// G kernels' series at most 2421 terms, the walk at most 2300 steps out and 2701 back for
// the probabilities and 3805 out and 4645 back for the wide sums (sd.stats' log forms and
// inverses). A wide tail whose peak sat at j* = 1e8 would take 9 sqrt(j* / 2) = 6.4e4 out
// and twice that back, inside the bound.
inline constexpr int kMaxTerms = 1 << 18;
inline constexpr Scaled kNotSummed{kNaN, 0};
// A probability from a sum: rounding may pass 1 by an ULP (NaN stays NaN: std::min would
// return its first argument for it).
inline double at_most_one(double v) { return v > 1.0 ? 1.0 : v; }

// ---- wide m 2^e values (the tails as Scaled, for sd.stats' log forms and inverses: their
// targets reach e^-1e10, where exp_scaled, which clamps at |x| = 1e5, would not reach)

// e^x as m 2^e for any x: 2^n e^r, n an int64, r = x - n log 2 (fma, log 2 in two parts);
// inf or 0 from |x| = 2^62 (n would pass an int64).
inline Scaled exp_wide(double x) {
    if (std::fabs(x) <= 1e5) return exp_scaled(x);
    if (!(x == x)) return {kNaN, 0};
    if (!(std::fabs(x) < 0x1p62)) return x > 0.0 ? Scaled{kInf, 0} : Scaled{0.0, 0};
    const double n = std::nearbyint(x / kLn2Hi);
    double r = std::fma(-n, kLn2Hi, x);
    r = std::fma(-n, kLn2Lo, r);
    Scaled s = scaled(std::exp(r));
    s.e += static_cast<std::int64_t>(n);
    return s;
}
// (the low part as 1 + lo while it is small; from |hi| = 2^53 it need not be, and 1 + lo
// could even be negative. Past 2^62 the high part alone decides, 0 or inf: e^lo there may
// be the other of the two, and their product NaN.)
inline Scaled exp_wide(dd::DD x) {
    if (std::fabs(x.lo) <= 0.5) return exp_wide(x.hi) * scaled(1.0 + x.lo);
    if (!(std::fabs(x.hi) < 0x1p62)) return exp_wide(x.hi);
    return exp_wide(x.hi) * exp_wide(x.lo);
}

// gamma_prefix and beta_prefix past their exponent's clamp (|E| > 1e5)
inline Scaled gamma_prefix_w(double a, dd::DD E) {
    if (std::fabs(E.hi) < 9e4 || !(E.hi < 1e300)) return gamma_prefix(a, E);
    return exp_wide(-E) / scaled(std::sqrt(2.0 * kPi * a) * gamma_star(a));
}
inline Scaled beta_prefix_w(double a, double b, dd::DD E) {
    if (std::fabs(E.hi) < 9e4 || !(E.hi < 1e300)) return beta_prefix(a, b, E);
    if (std::fabs(E.lo) <= 0.5) return exp_wide(-E.hi) * scaled((1.0 - E.lo) * beta_factor(a, b));
    return exp_wide(-E) * scaled(beta_factor(a, b));
}

// a + b for a, b >= 0 (or of opposite signs with |a + b| not far below them), as m 2^e
inline Scaled add_s(Scaled a, Scaled b) {
    if (a.m == 0.0) return b;
    if (b.m == 0.0) return a;
    const std::int64_t e = std::max(a.e, b.e);
    auto at = [&](Scaled v) { return std::ldexp(v.m, static_cast<int>(std::max<std::int64_t>(v.e - e, -2200))); };
    Scaled r = scaled(at(a) + at(b));
    r.e += e;
    return r;
}
inline Scaled times(Scaled a, double c) { return a * scaled(c); }
// a probability: rounding may pass 1 (or, after a difference, 0) by an ULP
inline Scaled clamp01(Scaled s) {
    if (!(s.m > 0.0)) return s.m < 0.0 ? scaled(0.0) : s;
    return (s.e > 1 || (s.e == 1 && s.m > 0.5)) ? scaled(1.0) : s;
}
// 1 - s for 0 <= s <= 1, where s is the small side (the far-tail forms below compute one side
// and hand the other out as its complement). A complement below 1/64 has lost its digits (it
// is what a tiny shape makes of the "large" side: ncfdtr(1e7, 1e-300, 2, 0.1) was 2.2e-16 for
// 3.4e-298): no value.
inline Scaled one_minus(Scaled s) {
    const double c = 1.0 - value(s);
    return c >= 0x1p-6 ? scaled(c) : Scaled{kNaN, 0};
}

// log Gamma(z), z > 0, as a double-double: Stirling's form (z - 1/2) log z - z + log sqrt(2 pi)
// + log Gamma*(z), Gamma* by gamma_star (relative 1 ULP, so its log is exact to ~1e-16
// absolute, while the large terms keep double-double accuracy: no cancellation between
// two log Gammas of 1e5 rounds the difference).
inline dd::DD lgamma_dd(dd::DD z) {
    constexpr dd::DD kLogSqrt2Pi{0.9189385332046728, -3.8782941580672414e-17};
    return (z - 0.5) * dd::log(z) - z + kLogSqrt2Pi + std::log(gamma_star(z.hi));
}

// D(a, mu) as m 2^e; D(0, mu) = e^-mu. mu = hi + lo (the t's nc^2 / 2 is not a
// double; rounding it moved its tails by nc^2 eps |d log F / d mu|, 89 ULP at nc = 17):
// d log D / d mu = a / mu - 1.
inline Scaled prefix_s(double a, double mu, double mu_lo = 0.0) {
    const Scaled d = a == 0.0 ? exp_wide(-mu) : gamma_prefix_w(a, gamma_exponent(a, mu));
    return mu_lo == 0.0 ? d : d * scaled(1.0 + mu_lo * (a / mu - 1.0));
}

// The shape s0 + j is a double-double (two_sum): rounded to a double it would
// move G by s eps |d log G / ds|, which is s eps |log(y/s)| in the tails (250 ULP
// at s = 800). G and T are evaluated at s.hi and moved by s.lo to first order: T
// by its log-derivative (digamma), G by a forward difference over 2^-20 s (the
// correction is ~eps of G, so the difference's own error, ~2^-20, is nothing).
// That takes G nearly linear over the step: far out at a large shape (the wide tails:
// s ~ 1e7, where G grows by e^170 over 2^-20 s) it is not, and the step is cut by 2^10
// until G changes by less than a factor of two over it (2^-40 s is always such a step
// for s <= 1e8 here: |d log G / ds| is below 700); else NaN.
template <class F>
inline Scaled at_shape(const F& g, dd::DD s) {
    const Scaled v = g(s.hi);
    if (s.lo == 0.0 || v.m == 0.0) return v;
    for (int cut = -20; cut >= -40; cut -= 10) {
        const double up = s.hi + std::ldexp(s.hi, cut);
        const double q = value(g(up) / v);
        if (q >= 0.5 && q <= 2.0) return v * scaled(1.0 + s.lo * ((q - 1.0) / (up - s.hi)));
    }
    return kNotSummed;
}

// sum_k r_0 r_1 ... r_(k-1) (k >= 0) for ratios r_k < 1 eventually, as m 2^e: the
// terms may climb far above the double range first (1 - I_x(s, b) at a large s).
// At most kMaxTerms terms (a value below kTinyG at a shape s <= 1e8 needs 9 sqrt(s) of the
// gamma series and 39 / log(1 / x), x^s < 2^-1000, of the beta one); not ended by then: NaN.
template <class R>
inline Scaled ratio_series(const R& ratio) {
    double sum = 1.0, term = 1.0;
    std::int64_t e = 0;
    for (int k = 0; k < kMaxTerms; ++k) {
        term *= ratio(k);
        sum += term;
        if (sum > 0x1p500) sum = std::ldexp(sum, -500), term = std::ldexp(term, -500), e += 500;
        if (!(term > sum * 0x1p-56)) {  // faded (a NaN term ends it too: the sum is NaN with it)
            Scaled r = scaled(sum);
            r.e += e;
            return r;
        }
    }
    return kNotSummed;
}

// Below this a G value is taken from the prefix times its series or fraction, so
// that it keeps its exponent (a result in the subnormals keeps its absolute accuracy).
inline constexpr double kTinyG = 0x1p-1000;

// 1 / d to double-double accuracy: one division and a Newton correction.
inline dd::DD recip(dd::DD d) {
    const double r = 1.0 / d.hi;
    const double e = std::fma(-r, d.hi, 1.0) - r * d.lo;
    return dd::quick_two_sum(r, r * e);
}


// P(s, y) or Q(s, y) along s; T(s) = D(s, y).
struct GammaFamily {
    double y;
    bool upper;
    dd::DD inv_y = recip(dd::DD{y, 0.0});
    Scaled G0(double s) const {
        if (s == 0.0) return scaled(upper ? 0.0 : 1.0);
        const igam_detail::PQ r = igam_detail::igam(s, y);
        const double g = upper ? r.q : r.p;
        if (g >= kTinyG) return scaled(g);
        const Scaled d = gamma_prefix_w(s, gamma_exponent(s, y));
        if (!upper) return d * ratio_series([&](int k) { return y / (s + k + 1.0); });  // P = D sum_k y^k / (s+1)_k
        // Q = D s / tail (igam_detail::log_tail's fraction): at most 1e5 terms (it converges in
        // about sqrt(y) + 40 for y > s, and a Q this small has y far above s); not converged: NaN
        const CfResult tail = lentz_backward(
            y + 1.0 - s,
            [&](int n) {
                const double k = n;
                return std::pair{-k * (k - s), y + 2.0 * k + 1.0 - s};
            },
            100000);
        if (!tail.converged) return kNotSummed;
        return d * scaled(s / tail.value);
    }
    Scaled G(dd::DD s) const { return at_shape([&](double v) { return G0(v); }, s); }
    Scaled T(dd::DD s) const {
        const Scaled t = prefix_s(s.hi, y);
        return s.lo == 0.0 ? t : t * scaled(1.0 + s.lo * (std::log(y) - Digamma::f(s.hi + 1.0)));
    }
    double ratio(dd::DD s) const {  // T(s+1) / T(s) = y / (s + 1)
        const dd::DD d = s + 1.0;
        return y / d.hi * (1.0 - d.lo / d.hi);
    }
    dd::DD up_dd(dd::DD s) const { return recip(s + 1.0) * y; }  // T(s+1) / T(s)
    dd::DD down_dd(dd::DD s) const { return (s + 1.0) * inv_y; }  // T(s) / T(s+1)
    // T(s) / T(s+1) as r 2^e: at y below 2^-900, (s+1)/y would overflow (y scaled by 2^-ilogb(y), exactly)
    dd::DD down_dd_e(dd::DD s, int& e) const {
        if (!(y < 0x1p-900)) {
            e = 0;
            return down_dd(s);
        }
        const int ey = std::ilogb(y);
        e = -ey;
        return (s + 1.0) * recip(dd::DD{std::ldexp(y, -ey), 0.0});
    }
    double gbound(dd::DD s) const { return std::min(1.0, ratio(s)); }  // >= P(s+1) / P(s)
    // log of a bound on G(s): P(s) <= T(s) / (1 - y/(s+1)) (the series' ratios are at most
    // y/(s+1)); Q(s) <= T(s-1) y / (y - s + 1) (below); else 1
    bool deep(double s) const { return !(y > 0x1p-20 * (s + 1.0) && y < 0x1p20 * (s + 1.0)); }
    // (rough logs, for the e^-2000 test: lgamma. From a shape of 2^40 its three terms round
    // by more than the test's margin, a 2^-52 |log y| each: the prefix's exponent there.)
    double log_gbound(double s) const {
        auto log_d = [&](double a) {
            if (!(a > 0.0)) return -y;
            if (a > 0x1p40) return log_abs(prefix_s(a, y));
            return a * std::log(y) - y - std::lgamma(a + 1.0);
        };
        if (!upper) return s + 1.0 > y ? log_d(s) - std::log1p(-y / (s + 1.0)) : 0.0;
        return s >= 1.0 && y > s - 1.0 ? log_d(s - 1.0) + std::log(y / (y - s + 1.0)) : 0.0;
    }
    // >= Q(s-1) / Q(s): Gamma(s, y) <= y^(s-1) e^-y y / (y - s + 1) for s >= 1, y > s - 1
    // (t^(s-1) <= y^(s-1) e^((s-1)(t-y)/y)), so Q(s-1) / Q(s) = 1 - T(s-1) / Q(s) <= (s-1) / y.
    double gbound_down(dd::DD s) const { return s.hi >= 1.0 && y > s.hi - 1.0 ? std::min(1.0, (s.hi - 1.0) / y) : 1.0; }
};

// I_x(s, b) or 1 - I_x(s, b) along s; T(s) = x^s y^b / (s B(s, b)).
struct BetaFamily {
    double b;
    dd::DD x, y;  // y = 1 - x
    bool upper;
    dd::DD inv_x = recip(x);
    Scaled G0(double s) const {
        const ibeta_detail::IB r = ibeta_detail::ibeta(s, b, x, y);
        const double g = upper ? r.ic : r.i;
        if (g >= kTinyG) return scaled(g);
        if (!upper) {  // I_x(s, b) = sum_k T(s + k)
            const Scaled t = beta_prefix_w(s, b, beta_exponent(s, b, x, y)) / scaled(s);
            return t * ratio_series([&](int k) { return x.hi * (s + k + b) / (s + k + 1.0); });
        }
        // 1 - I_x(s, b) = I_y(b, s) = sum_k y^c x^s / (c B(c, s)), c = b + k
        const Scaled t = beta_prefix_w(b, s, beta_exponent(b, s, y, x)) / scaled(b);
        return t * ratio_series([&](int k) { return y.hi * (b + k + s) / (b + k + 1.0); });
    }
    Scaled G(dd::DD s) const { return at_shape([&](double v) { return G0(v); }, s); }
    Scaled T(dd::DD s) const {
        const Scaled t = beta_prefix_w(s.hi, b, beta_exponent(s.hi, b, x, y)) / scaled(s.hi);
        if (s.lo == 0.0) return t;
        return t * scaled(1.0 + s.lo * (std::log(x.hi) - 1.0 / s.hi - Digamma::f(s.hi) + Digamma::f(s.hi + b)));
    }
    double ratio(dd::DD s) const {  // T(s+1) / T(s) = x (s + b) / (s + 1)
        const dd::DD n = s + b, d = s + 1.0;
        return x.hi * (n.hi / d.hi) * (1.0 + (n.lo / n.hi - d.lo / d.hi) + x.lo / x.hi);
    }
    dd::DD up_dd(dd::DD s) const { return x * (s + b) * recip(s + 1.0); }  // T(s+1) / T(s)
    dd::DD down_dd(dd::DD s) const { return (s + 1.0) * recip(s + b) * inv_x; }  // T(s) / T(s+1)
    dd::DD down_dd_e(dd::DD s, int& e) const {  // x is at least 2^-1000 here (smaller: the tiny paths)
        e = 0;
        return down_dd(s);
    }
    double gbound(dd::DD s) const {  // >= I(s+1) / I(s): the T ratio, times e^((1-b)/(s+b)) for b < 1
        const double r = ratio(s) * (b < 1.0 ? std::exp((1.0 - b) / (s.hi + b)) : 1.0);
        return std::min(1.0, r);
    }
    double gbound_down(dd::DD) const { return 1.0; }  // >= (1 - I(s-1)) / (1 - I(s))
    // log of a bound on G(s): I_x(s, b) <= T(s) / (1 - r) for the ratio bound r < 1 (gbound's),
    // 1 - I_x(s, b) = I_y(b, s) likewise with the roles turned; else 1
    bool deep(double) const { return x.hi < 0x1p-20 || y.hi < 0x1p-20; }
    // (rough logs, for the e^-2000 test: lgamma. From a shape of 2^40 the three lgammas
    // cancel to nothing, 2^-52 b log b: the prefix's exponent there.)
    double log_gbound(double s) const {
        if (s > 0x1p40 || b > 0x1p40) {
            if (!upper) {
                const double r = x.hi * (s + b) / (s + 1.0) * (b < 1.0 ? std::exp((1.0 - b) / (s + b)) : 1.0);
                return r < 1.0 ? log_abs(beta_prefix_w(s, b, beta_exponent(s, b, x, y)) / scaled(s)) - std::log1p(-r) : 0.0;
            }
            const double r = y.hi * (b + s) / (b + 1.0) * (s < 1.0 ? std::exp((1.0 - s) / (b + s)) : 1.0);
            return r < 1.0 ? log_abs(beta_prefix_w(b, s, beta_exponent(b, s, y, x)) / scaled(b)) - std::log1p(-r) : 0.0;
        }
        auto log_t = [&](double c, double d, double lx, double ly) {  // log(x^c y^d / (c B(c, d)))
            return c * lx + d * ly - std::log(c) - (std::lgamma(c) + std::lgamma(d) - std::lgamma(c + d));
        };
        const double lx = std::log(x.hi), ly = std::log(y.hi);
        if (!upper) {
            const double r = x.hi * (s + b) / (s + 1.0) * (b < 1.0 ? std::exp((1.0 - b) / (s + b)) : 1.0);
            return r < 1.0 ? log_t(s, b, lx, ly) - std::log1p(-r) : 0.0;
        }
        const double r = y.hi * (b + s) / (b + 1.0) * (s < 1.0 ? std::exp((1.0 - s) / (b + s)) : 1.0);
        return r < 1.0 ? log_t(b, s, ly, lx) - std::log1p(-r) : 0.0;
    }
};

using dd::Sum;  // the compensated sum (special/dd.hpp)

// The sum. j* (the largest term, roughly) splits the terms into the side where
// the recurrence for G adds T (stable) and the side where it would subtract. The
// walk starts on the second side at J, far enough out that a rigorous bound on the
// terms from j* to J (the weights' ratio times a bound on G's ratio) has fallen
// below kTol of the term at j*, and comes back across j* to where the terms fade:
// one direct evaluation of w, G and T, at J. The terms are summed in the frame 2^k0
// of the first term, w kept in [2^-500, 2^500] by moving powers of two between w
// and G (their product unchanged), so terms far below the double range keep their bits.
// As m 2^e: `wide` (sd.stats' log forms) also sums the tails below e^-2000 that the
// probability (mixture below) rounds to 0.
//
// Bounded: the terms are indexed by doubles, exact integers below 2^52 (a weights' mode
// beyond that: NaN); the search for j* is at most 52 doublings and 53 halvings; the way out
// to J takes at most kMaxTerms steps and the walk back, over it and as far beyond j*, at
// most twice that. A sum that needs more (the weights alone span 18 sqrt(mu) terms: mu
// beyond 2e8 or so) is NaN.
template <class Fam>
Scaled mixture_walk(double mu, double h, double s0, const Fam& f, double mu_lo, bool wide) {
    const bool lower = !f.upper;
    auto shape = [&](double j) { return dd::two_sum(s0, j); };  // s0 + j exactly
    auto term_ratio = [&](double j) {  // (term j+1) / (term j), with G's ratio taken as T's (deep tails)
        return mu / (j + 1.0 + h) * (lower ? f.ratio(shape(j)) : (s0 + j > 0 ? f.ratio(shape(j - 1.0)) : 1.0));
    };
    // j*: the weights' mode, moved to where term_ratio crosses 1 if that is further
    // into the tail (below the mode for a lower family, above it for an upper one).
    const double jmode = std::max(0.0, std::floor(mu - h));
    if (!(jmode < 0x1p52)) return kNotSummed;
    // The crossing matters only on the tail side of the mode: below it for a lower family
    // (term_ratio(jmode) <= 1), above it for an upper one (> 1). Else j* is the mode itself
    // (one ratio instead of a bisection: most inputs).
    double jstar = jmode;
    const bool past = term_ratio(jmode) <= 1.0;
    if (lower ? (past && jmode > 0.0) : !past) {
        double lo = lower ? 0.0 : jmode, hi = lower ? jmode : std::ceil(std::max(4.0 * mu, jmode + 64.0));  // integers
        if (!lower) {
            while (term_ratio(hi) > 1.0 && hi < 0x1p52) hi *= 2.0;  // (at most 52 times)
            // The terms still climb at hi >= 2^52 (an upper tail at x ~ 1e100). With hi >= 4 mu
            // that takes T(s + 1) / T(s) > 4 at s = s0 + hi: G there is a tail four times and
            // more beyond its distribution's mean, below e^-s, and every G before it is
            // smaller; the terms after it are below their weights, whose sum from hi on is
            // below e^(-hi (log 4 - 1)). So the sum is under e^(-1e15): 0 as a probability.
            // Its own value: not summed.
            if (term_ratio(hi) > 1.0) return wide ? kNotSummed : scaled(0.0);
        }
        if (lower && term_ratio(0.0) <= 1.0) hi = 0.0;
        while (hi - lo > 1.0) {  // (integers below 2^53: the gap halves, at most 53 times)
            const double mid = std::floor(0.5 * (lo + hi));
            (term_ratio(mid) > 1.0 ? lo : hi) = mid;
        }
        jstar = hi;
    }
    // A sum far below the double range: its largest term, ~ the one at j*, under e^-2000
    // (a margin no misplacement of j* can bridge). Also keeps the walk off j* ~ 2^52 (a
    // tail at x ~ 1e100) and off exponents beyond Scaled's exp (-1.4e5).
    // (Rough logs suffice for that margin: lgamma, no prefix.)
    const double a = jstar + h;
    // Only where it can happen: j* off the mode, or an argument 2^20 away from the shape.
    if ((jstar != jmode || f.deep(jstar + s0)) &&
        (a > 0.0 ? a * std::log(mu) - mu - std::lgamma(a + 1.0) : -mu) + f.log_gbound(jstar + s0) < -2000.0) {
        if (!wide) return scaled(0.0);
        if (jstar > 1e8) return {kNaN, 0};  // a walk of 1e8 terms (a tail at x ~ 1e100): not taken
    }

    // J: past j* on the unstable side (up for a lower family, down for an upper one)
    // until the bound times the bounded tail beyond, prod / (1 - r), is below kTol.
    // (At most kMaxTerms steps: the walk back would take as many.)
    double J = jstar, prod = 1.0;
    bool far = false;
    for (int n = 0; n < kMaxTerms; ++n) {
        double r;
        if (lower) {
            r = mu / (J + 1.0 + h) * f.gbound(shape(J));
            J += 1.0;
        } else {
            if (J == 0.0) {
                far = true;
                break;
            }
            r = (J - 1.0 + h + 1.0) / mu * f.gbound_down(shape(J));  // w_(J-1) / w_J = (J + h) / mu
            J -= 1.0;
        }
        prod *= r;
        if (r < 1.0 && prod <= kTol * (1.0 - r)) {
            far = true;
            break;
        }
        if (prod > 0x1p500) prod = 0x1p500;  // the climb to the weights' mode: only its fall matters
    }
    if (!far) return kNotSummed;

    const Scaled wJ = prefix_s(J + h, mu, mu_lo), gJ = f.G(shape(J));
    const Scaled tJ = f.T(shape(lower ? J - 1.0 : J));  // the T the first step adds
    // G and T both exactly 0 (e^-E with E beyond 1e300): every term of the walk is 0, and
    // nothing would ever fall to end it
    if (gJ.m == 0.0 && tJ.m == 0.0) return scaled(0.0);
    // (a start that is NaN, G's kernels not having ended, makes every term NaN)
    if (wJ.m != wJ.m || gJ.m != gJ.m || tJ.m != tJ.m) return kNotSummed;
    std::int64_t ef = gJ.m != 0.0 ? gJ.e : tJ.e;
    // T past G by 2^900 (a lower family at y ~ 1e-307, wide): T sets the frame, else it
    // would overflow in G's (G is then nothing beside it)
    if (tJ.m != 0.0 && tJ.e - ef > 900) ef = tJ.e;
    const std::int64_t k0 = -(wJ.e + ef);
    auto frame = [](Scaled v, std::int64_t k) {
        return std::ldexp(v.m, static_cast<int>(std::clamp<std::int64_t>(v.e + k, -2200, 2200)));
    };
    // w and T follow their ratios in double-double (exact to ~2^-100 a step: no drift
    // over the hundreds of steps from J, and nothing to recompute); G and the sum are
    // compensated sums of doubles. w is stored times 2^kw, G and T times 2^kg, the terms,
    // their sum and the last term times 2^(kw + kg). Whichever of w and G leaves
    // [2^-500, 2^500] is brought back to about 1 by its own exponent, and the sum and the
    // last term follow by the step's whole move at once (`moved`): a step of 1 / mu or of
    // (s + 1) / y can be 2^1000 and more (mu or y near the smallest doubles), and moved in
    // two parts they would overflow between the parts.
    std::int64_t kw = -wJ.e, kg = k0 - kw;
    dd::DD w{frame(wJ, kw), 0.0}, t{frame(tJ, kg), 0.0};
    // mu as mm 2^em with mm in [1, 2) where 1 / mu, or w times it, could leave the doubles
    int em = 0;
    dd::DD mud{mu, mu_lo};
    if (!(mu >= 0x1p-400 && mu <= 0x1p400) && std::isfinite(mu)) {
        em = std::ilogb(mu);
        mud = dd::ldexp(mud, -em);
    }
    const dd::DD inv_mu = recip(mud);
    const int wstep = lower ? em : -em;  // w's exponent moves by this each step (0 unless mu is scaled)
    Sum g;
    g.add(frame(gJ, kg));
    Sum S;
    double prev = w.hi * g.total(), j = J;
    S.add(prev);
    const double dir = lower ? -1.0 : 1.0;
    auto shift = [](dd::DD v, int e) { return dd::DD{std::ldexp(v.hi, e), std::ldexp(v.lo, e)}; };
    bool ended = false;  // by the terms (their fading or j = 0), not by the count
    for (int n = 0; n < 2 * kMaxTerms; ++n) {  // (back over the way out, and as far beyond j*)
        if (lower && j == 0.0) {
            ended = true;
            break;
        }
        const double jn = j + dir;
        g.add(t);
        w = lower ? w * (j + h) * inv_mu : w * mud * recip(dd::DD{j + 1.0 + h, 0.0});
        j = jn;
        double gv = g.total();
        int moved = wstep;  // the terms' frame kw + kg against the last step's
        kw += wstep;
        if (!(std::fabs(w.hi) <= 0x1p500 && std::fabs(w.hi) >= 0x1p-500) && w.hi != 0.0) {  // w back near 1
            const int e = std::ilogb(w.hi);
            kw -= e, moved -= e;
            w = shift(w, -e);
        }
        if (!(std::fabs(gv) <= 0x1p500 && std::fabs(gv) >= 0x1p-500) && gv != 0.0) {  // G back near 1, T with it
            const int e = std::ilogb(gv);
            kg -= e, moved -= e;
            t = shift(t, -e);
            g.scale(-e);
            gv = g.total();
        }
        if (moved != 0) {
            // The sum would pass 2^1000 in the new frame: the terms have fallen 2^-500 below
            // it (they are at most 2^501 there), which only happens past their peak. Nothing
            // further counts. (kw back, so that kw + kg is still the sum's frame.)
            if (S.hi != 0.0 && std::ilogb(S.hi) + moved > 1000) {
                kw -= moved;
                ended = true;
                break;
            }
            prev = std::ldexp(prev, moved);
            S.scale(moved);
        }
        const double term = w.hi * gv + w.lo * gv;
        S.add(term);
        int er = 0;  // T's step as rt 2^er: (s+1)/y passes the double range at y ~ 1e-308
        const dd::DD rt = lower ? f.down_dd_e(shape(j - 1.0), er) : f.up_dd(shape(j - 1.0));
        t = t * rt;
        // past j* and fading: the terms beyond fall at least geometrically (log-concave);
        // or past j* with the terms at 0 (all of them below the frame's range)
        if ((lower ? j < jstar : j > jstar) &&
            ((term < prev && term <= kTol * S.total() * (1.0 - term / prev)) || (term == 0.0 && prev == 0.0))) {
            ended = true;
            break;
        }
        prev = term;
        if (er != 0) {  // G's frame moves by 2^er (the stored t is t rt), and the terms' with it
            kg -= er;
            prev = std::ldexp(prev, -er);
            g.scale(-er), S.scale(-er);
        }
    }
    if (!ended) return kNotSummed;
    Scaled r = scaled(S.total());
    r.e -= kw + kg;
    return r;
}
// mu = 0 (a noncentrality whose half, or half square, is below the doubles) has the
// weights D(j + h, 0): 1 at j + h = 0 and 0 elsewhere. (Apart from the walk, whose code it
// would otherwise change: with this test inside, the lower gamma walk is half as slow again.)
template <class Fam>
Scaled mixture_s(double mu, double h, double s0, const Fam& f, double mu_lo = 0.0, bool wide = false) {
    if (mu == 0.0) return h == 0.0 ? f.G(dd::DD{s0, 0.0}) : scaled(0.0);
    return mixture_walk(mu, h, s0, f, mu_lo, wide);
}
template <class Fam>
double mixture(double mu, double h, double s0, const Fam& f, double mu_lo = 0.0) {
    return at_most_one(value(mixture_s(mu, h, s0, f, mu_lo)));
}

// The upper family's sum when its small argument y is below 2^-1000 (it may not
// even be a double: t ~ 1e250, f ~ 1e300), from I_y(b, s) = y^b Gamma(b+s) /
// (Gamma(b+1) Gamma(s)) (1 + O(y)):
//     sum_j D(j + h, mu) I_y(b, s0 + j) = y^b sum_j t_j,  t_j = D(j + h, mu) Gamma(b + s_j) / (Gamma(b+1) Gamma(s_j)),
// s_j = s0 + j. The terms by their ratio t_(j+1) / t_j = mu (s_j + b) / ((j + 1 + h) s_j)
// (falling in j) in double-double, out from the largest t_j*, which alone is direct: D by
// prefix_s, the log Gamma ratio by lgamma_dd. (lgamma differences in double cancel: at
// s ~ 5e3 they kept 1e-12 of the result.) As m 2^e: y^b reaches e^-1e10.
// Bounded as the walk: j* an exact integer below 2^52 (52 doublings, 53 halvings), at most
// kMaxTerms terms each way (they span 9 sqrt(j*) or so); else NaN. NaN too where a shape
// passes 2^50: the three log Gammas of the largest term then cancel past their bits.
inline Scaled tiny_upper_s(double mu, double h, double s0, double b, dd::DD logy, double mu_lo = 0.0) {
    const dd::DD mud{mu, mu_lo};
    auto ratio = [&](double j) {  // t_(j+1) / t_j
        const dd::DD sj = dd::two_sum(s0, j);
        return mud * (sj + b) * recip(sj * (j + 1.0 + h));
    };
    // j*: the first j with ratio(j) <= 1 (the ratio falls in j)
    double lo = -1.0, hi = std::max(1.0, std::floor(mu - h));
    while (ratio(hi).hi > 1.0 && hi < 0x1p52) lo = hi, hi *= 2.0;
    if (!(hi < 0x1p52)) return kNotSummed;
    while (hi - lo > 1.0) {
        const double mid = std::floor(0.5 * (lo + hi));
        (ratio(mid).hi > 1.0 ? lo : hi) = mid;
    }
    const double jstar = std::max(0.0, hi);
    if (!(s0 + jstar <= 0x1p50 && b <= 0x1p50)) return kNotSummed;
    // (the O(y) left out of I_y(b, s) is y s: nothing at a shape of the region, 1e5 times
    // 2^-1000; a shape large enough to bring it to 2^-60 is not summed this way)
    if (!(logy.hi + std::log(s0 + jstar + 64.0) < -42.0)) return kNotSummed;
    const dd::DD ss = dd::two_sum(s0, jstar);
    const dd::DD lratio = lgamma_dd(ss + b) - lgamma_dd(ss) - lgamma_dd(dd::DD{b, 0.0} + 1.0);
    const Scaled top = prefix_s(jstar + h, mu, mu_lo) * exp_wide(logy * b + lratio);
    Sum S;
    S.add(1.0);
    dd::DD u{1.0, 0.0};
    bool ended = false;
    for (int n = 0; n < kMaxTerms; ++n) {  // up
        u = u * ratio(jstar + n);
        S.add(u.hi, u.lo);
        if (!(u.hi > 0x1p-60 * S.total())) {
            ended = true;
            break;
        }
    }
    if (!ended) return kNotSummed;
    u = dd::DD{1.0, 0.0};
    ended = jstar == 0.0;
    for (int n = 1; n <= kMaxTerms && !ended; ++n) {  // down: t_j = t_(j+1) / ratio(j)
        const double j = jstar - n;
        u = u * recip(ratio(j));
        S.add(u.hi, u.lo);
        ended = j == 0.0 || !(u.hi > 0x1p-60 * S.total());
    }
    if (!ended) return kNotSummed;
    return top * scaled(S.total());
}

// ---- the bulk: the same sums from j = 0, without an incomplete function per walk
//
// With t_k = T(s0 + k) / T(s0) and w_j = D(j + h, mu) / D(h, mu) (t_0 = w_0 = 1), from
// G(s) = sum_(k >= 0) T(s + k) (lower) and G(s + j) = G(s) + sum_(i < j) T(s + i) (upper):
//
//     lower:  sum_j D(j + h, mu) G(s0 + j) = T(s0) D(h, mu) sum_k t_k W_k,   W_k = sum_(j <= k) w_j
//     upper:  sum_j D(j + h, mu) G(s0 + j) = G(s0) sum_j D(j + h, mu) + T(s0) D(h, mu) sum_(j >= 1) w_j C_j,
//                                                                          C_j = sum_(i < j) t_i
//
// Every term is positive, so both keep their relative accuracy at any size of the result;
// what bounds their use is the number of terms (the lower sum runs to where t_k fades,
// k ~ y - s0 for the gamma family; the upper one to where w_j fades, j ~ mu) and the range
// of a double (t_k and w_j climb to e^y and e^mu). Past kLiteTerms terms, or outside
// kLiteArg, the walk from the far end (mixture_s) takes over.
//
// The recurrences run in "double + error" arithmetic (Lite: h + l, l a few ULP of h, not
// renormalized; a product keeps the exact fma remainder and the cross terms, dropping only
// l l, relative 2^-100): a term after k steps is exact to k 2^-100 or so, against 2 k ULP
// in plain double (measured: 1e-14 at 30 terms). The sums are compensated the same way.
// So the error of a sum is its truncation (below 2^-58, a bound on the geometric tail)
// and the two direct values T(s0) and D(h, mu) (and G(s0) for the upper sum), each one
// evaluation of the kernels the far-end walk also starts from.
namespace lite {

struct Lite {
    double h, l;
};
inline Lite mul(Lite a, Lite b) {
    const double p = a.h * b.h;
    return {p, std::fma(a.h, b.h, -p) + (a.h * b.l + a.l * b.h)};
}
// n / d: the quotient through one reciprocal (so not correctly rounded: no matter, its
// remainder is taken exactly), then the remainder's share.
inline Lite quot(Lite n, Lite d) {
    const double inv = 1.0 / d.h, q = n.h * inv;
    return {q, ((std::fma(-q, d.h, n.h) + n.l) - q * d.l) * inv};
}
inline Lite sum2(double a, double b) {  // a + b exactly
    const double v = a + b, bb = v - a;
    return {v, (a - (v - bb)) + (b - bb)};
}
inline void bump(Lite& s) {  // s += 1
    const double v = s.h + 1.0, bb = v - s.h;
    s.l += (s.h - (v - bb)) + (1.0 - bb);
    s.h = v;
}
inline void add(Lite& s, Lite t) {  // s += t (two_sum of the high parts)
    const double v = s.h + t.h, bb = v - s.h;
    s.l += ((s.h - (v - bb)) + (t.h - bb)) + t.l;
    s.h = v;
}
inline double value(Lite a) { return a.h + a.l; }

inline constexpr int kLiteTerms = 400;
inline constexpr double kLiteArg = 340.0;  // e^(2 * 340) is a double
inline constexpr double kLiteTol = 0x1p-58;

// t_(k+1) / t_k of the gamma family at shape s0, for k = 0, 1, ... in turn: y / (s0 + k + 1),
// falling in k.
struct GammaRatio {
    Lite y, d;  // d = s0 + k + 1
    GammaRatio(double s0, double y_) : y{y_, 0.0}, d(sum2(s0, 1.0)) {}
    Lite next() {
        const Lite r = quot(y, d);
        bump(d);
        return r;
    }
    double later(double r) const { return r; }  // a bound on the ratios after one of r
    double dweight() const { return 1.0; }      // density(s0 + k + 1) / T(s0 + k), up to the family's factor
};
// of the beta family: x (s0 + k + b) / (s0 + k + 1); falls to x for b > 1, climbs to x for b < 1.
struct BetaRatio {
    Lite x, n, d;  // n = s0 + k + b, d = s0 + k + 1
    BetaRatio(double s0, double b, dd::DD x_) : x{x_.hi, x_.lo}, n(sum2(s0, b)), d(sum2(s0, 1.0)) {}
    Lite next() {
        const Lite r = quot(mul(x, n), d);
        bump(n), bump(d);
        return r;
    }
    double later(double r) const { return std::max(r, x.h); }
    double dweight() const { return n.h; }
};
// w_(j+1) / w_j = mu / (j + 1 + h), for j = 0, 1, ... in turn.
struct WeightRatio {
    Lite mu;
    double d;  // j + 1 + h (exact)
    WeightRatio(dd::DD mu_, double h) : mu{mu_.hi, mu_.lo}, d(1.0 + h) {}
    Lite next() {
        const Lite r = quot(mu, Lite{d, 0.0});
        d += 1.0;
        return r;
    }
};

// sum_k t_k W_k; wtotal >= sum_j w_j. False: not ended within kLiteTerms. dens (the inverses'
// Newton steps: a derivative, so plain double): sum_k w_(k+1) t_k dweight_k, the mixture's
// density up to the family's factors.
template <class TR>
inline bool lower_sum(TR tr, dd::DD mu, double h, double wtotal, double& out, double& dens) {
    WeightRatio wr(mu, h);
    Lite t{1.0, 0.0}, w{1.0, 0.0}, W{1.0, 0.0}, S{0.0, 0.0};
    dens = 0.0;
    for (int k = 0; k < kLiteTerms; ++k) {
        add(S, mul(t, W));
        const double tk = t.h * tr.dweight();
        const Lite r = tr.next();
        t = mul(t, r);
        // the terms beyond: at most wtotal t_(k+1) (1 + q + q^2 + ...), q a bound on the later ratios
        const double q = tr.later(r.h);
        if (q < 1.0 && t.h * wtotal <= kLiteTol * S.h * (1.0 - q)) {
            out = value(S);
            dens += w.h * wr.next().h * tk;  // this k's term (the rest: below wtotal t_(k+1) / (1 - q) too)
            return true;
        }
        w = mul(w, wr.next());
        add(W, w);
        dens += w.h * tk;
    }
    return false;
}

// sum_(j >= 1) w_j C_j; ctotal >= sum_i t_i (may be inf). dens as lower_sum's.
template <class TR>
inline bool upper_sum(TR tr, dd::DD mu, double h, double ctotal, double& out, double& dens) {
    WeightRatio wr(mu, h);
    Lite t{1.0, 0.0}, w{1.0, 0.0}, C{0.0, 0.0}, S{0.0, 0.0};
    dens = 0.0;
    for (int j = 0; j < kLiteTerms; ++j) {
        add(C, t);  // C_(j+1)
        const double tk = t.h * tr.dweight();
        const Lite r = tr.next();
        t = mul(t, r);  // t_(j+1)
        w = mul(w, wr.next());  // w_(j+1)
        add(S, mul(w, C));
        dens += w.h * tk;
        // the terms beyond: at most cmax w_(j+1) (rho + rho^2 + ...), rho = w_(j+2) / w_(j+1) (falling),
        // cmax a bound on every C: ctotal, or C_(j+1) + t_(j+1) / (1 - q) once t falls
        const double rho = mu.hi / (j + 2.0 + h), q = tr.later(r.h);
        if (rho < 1.0) {
            const double cmax = q < 1.0 ? std::min(ctotal, C.h + t.h / (1.0 - q)) : ctotal;
            if (w.h * rho * cmax <= kLiteTol * S.h * (1.0 - rho)) {
                out = value(S);
                return true;
            }
        }
    }
    return false;
}

}  // namespace lite

// The first terms in the bulk, T(s0) e^-mu (the weights' e^-mu with it: one exponential):
// the powers by pow, and the factor of the parameters alone (e^-mu / Gamma(a + 1),
// e^-mu / (a B(a, b))) as e^l, l in double-double from dd::loggamma, kept per thread for the
// last parameters (an array's are usually one value; a pure function of the key, so a hit and
// a miss, a scalar parameter and an array of it, give the same bits). In units of 2^-53 (one
// rounded operation 1, a libm call 2): each pow or exp 2, the factor 4 (exp, 1 + l.lo, their
// product), the first-order correction for the arguments' low parts 1, each product 1:
// 10 for the gamma family, 12 for the beta one (a^2 2^-106 is left of the low parts: a <=
// 2^20). Where a power leaves [2^-400, 2^400]: the prefixes (Temme's form,
// special/prefix.hpp), as the far-end walk.
struct Memo3 {
    double a = kNaN, b = kNaN, mu = kNaN, mu_lo = kNaN, v = 0.0;
    bool hit(double a_, double b_, dd::DD m) const { return a == a_ && b == b_ && mu == m.hi && mu_lo == m.lo; }
};
inline constexpr double kLitePow = 0x1p-400;
inline bool lite_normal(double v) { return v >= kLitePow && v <= 1.0 / kLitePow; }
// D(a, y) e^-mu = y^a e^-y e^-mu / Gamma(a + 1), a > 0
inline double gamma_first(double a, double y, dd::DD mu) {
    const double py = std::pow(y, a), ey = std::exp(-y);
    if (lite_normal(py) && ey >= kLitePow && a <= 0x1p20) {
        thread_local Memo3 m;
        if (!m.hit(a, 0.0, mu)) {
            const dd::DD l = -dd::loggamma(dd::DD{a, 0.0} + 1.0) - mu;
            m = {a, 0.0, mu.hi, mu.lo, std::exp(l.hi) * (1.0 + l.lo)};
        }
        if (lite_normal(m.v)) return py * ey * m.v;
    }
    return value(gamma_prefix(a, y)) * std::exp(-mu.hi);
}
// x^a y^b e^-mu / (a B(a, b)), x + y = 1 as double-doubles: by the powers (false outside their range)
inline bool beta_first_pow(double a, double b, dd::DD x, dd::DD y, dd::DD mu, double& out) {
    const double px = std::pow(x.hi, a), py = std::pow(y.hi, b);
    if (!(px >= kLitePow && py >= kLitePow && a <= 0x1p20 && b <= 0x1p20)) return false;
    thread_local Memo3 m;
    if (!m.hit(a, b, mu)) {
        const dd::DD l = dd::loggamma(dd::two_sum(a, b)) - dd::loggamma(dd::DD{a, 0.0} + 1.0) -
                         dd::loggamma(dd::DD{b, 0.0}) - mu;
        m = {a, b, mu.hi, mu.lo, std::exp(l.hi) * (1.0 + l.lo)};
    }
    if (!lite_normal(m.v)) return false;
    out = px * py * (1.0 + (a * (x.lo / x.hi) + b * (y.lo / y.hi))) * m.v;
    return true;
}
inline double beta_first(double a, double b, dd::DD x, dd::DD y, dd::DD mu) {
    double out;
    if (beta_first_pow(a, b, x, y, mu, out)) return out;
    return value(beta_prefix_w(a, b, beta_exponent(a, b, x, y)) / scaled(a)) * (std::exp(-mu.hi) * (1.0 - mu.lo));
}

// Rough term counts, to choose between the sums (and not to start one that cannot end):
// the lower sum runs to the peak of t_k and down its far side, the upper one over the
// weights.
inline double lite_terms_weights(double mu) { return mu + 9.0 * std::sqrt(mu) + 12.0; }
inline double lite_terms_gamma(double s0, double y) { return std::max(y - s0, 0.0) + 9.0 * std::sqrt(y) + 12.0; }
inline double lite_terms_beta(double s0, double b, double x, double y) {
    // t_k ~ k^(b-1) x^k past its peak k* = (x (s0 + b) - s0 - 1) / y
    const double peak = std::max((x * (s0 + b) - s0 - 1.0) / y, 0.0), lx = -std::log1p(-y);
    return peak + (40.0 + std::max(b - 1.0, 0.0) * std::log1p(40.0 / lx)) / lx + 9.0 * std::sqrt(peak / y);
}

// The choice, for both tails: `lower(s)` and `upper(s)` run the sums (false: not ended) and
// give the tail; nl and nu their rough term counts; z is the point's distance from the
// middle of the distribution in standard deviations (roughly: only a guide to which sum to
// try, so that one that cannot be used is seldom run). The asked tail from its own sum, or
// where cheaper as 1 - the other: of a lower sum at most 1/2 (its absolute error is then
// inside the asked tail's relative one; the lower sum has no incomplete function in it), of
// an upper sum at most 1/4 (a third of its error, which includes its kernel's: P, Q and the
// incomplete beta were measured to 2-3e-15 at small shapes).
template <class L, class U>
inline bool lite_pick(bool upper, double mu, double nl, double nu, double direct, double z, const L& lower,
                      const U& upper_sum_of, double& out) {
    const bool below = z < 0.0, above = z > 0.75;  // (a normal's upper tail at 0.75 is 0.23)
    // The far-end walk costs about one incomplete function and the prefixes (60 terms' worth)
    // and four times a term for each of its own (the weights' width, both sides): a lower
    // sum longer than that is not started (measured: 20, 60 degrees of freedom at nc = 25
    // took 1.7 us for 350 terms against the walk's 1.1).
    const double walk = 60.0 + 4.0 * (9.0 * std::sqrt(mu) + 12.0);
    const bool l_ok = nl <= std::min<double>(lite::kLiteTerms, walk), u_ok = nu <= lite::kLiteTerms;
    const bool l_cheaper = nl <= nu + direct;  // the upper sum needs one incomplete function (`direct` terms' worth)
    double v;
    if (!upper) {
        if (l_ok && (l_cheaper || !u_ok)) return lower(out);
        if (u_ok && above && upper_sum_of(v) && v <= 0.25) {
            out = 1.0 - v;
            return true;
        }
        return l_ok && lower(out);
    }
    if (u_ok && !(below && l_cheaper && l_ok)) return upper_sum_of(out);
    if (l_ok && lower(v) && v <= 0.5) {
        out = 1.0 - v;
        return true;
    }
    return u_ok && upper_sum_of(out);
}

// chndtr (upper: chndtrc) by the sums above; false where they do not apply. pdf (when asked:
// the inverses' Newton steps, to 1e-8 or so): the density, sum_j D(j, mu) D(a + j - 1, y) / 2.
// Error of the lower sum, in units of 2^-53: the first term 10 (gamma_first), the sum's
// rounding and the product 2: 12; the upper one adds Q(a, y) (igam: the far-end walk's own
// kernel) and one addition.
inline bool ncx2_lite(double df, double nc, double x, bool upper, double& out, double* pdf = nullptr) {
    const double mu = 0.5 * nc, a = 0.5 * df, y = 0.5 * x;
    if (!(mu <= lite::kLiteArg && y <= lite::kLiteArg && y >= 0x1p-500 && a <= 1e6)) return false;
    const lite::GammaRatio tr(a, y);
    const double tw = a == 0.0 ? std::exp(-y) * std::exp(-mu) : gamma_first(a, y, dd::DD{mu, 0.0});  // T(a) e^-mu
    if (!(tw >= 0x1p-900)) return false;
    double dens = 0.0;
    auto lower = [&](double& v) {
        double s;
        if (!lite::lower_sum(tr, dd::DD{mu, 0.0}, 0.0, std::exp(mu) * (1.0 + 0x1p-40), s, dens)) return false;
        v = at_most_one(tw * s);
        return v >= 0x1p-1000;
    };
    auto upper_of = [&](double& v) {
        double s;
        const double q0 = a == 0.0 ? 0.0 : igam_detail::igam(a, y).q;
        if (!(q0 >= kTinyG || a == 0.0) || !lite::upper_sum(tr, dd::DD{mu, 0.0}, 0.0, std::exp(-mu) / tw, s, dens)) return false;
        v = at_most_one(q0 + tw * s);
        return v >= 0x1p-1000;
    };
    if (!lite_pick(upper, mu, lite_terms_gamma(a, y), lite_terms_weights(mu), 30.0, (x - (df + nc)) / std::sqrt(2.0 * (df + 2.0 * nc)),
                   lower, upper_of, out)) return false;
    if (pdf) *pdf = 0.5 * tw * (a / y + dens);
    return true;
}

// ncfdtr (upper: ncfdtrc) by the sums above (12 units for the first term, 2 for the sum and
// the product: 14; the upper one adds 1 - I_w(a, b), fdtrc's kernel). pdf: the density in x.
inline bool ncf_lite(double dfn, double dfd, double nc, double x, bool upper, double& out, double* pdf = nullptr) {
    const double mu = 0.5 * nc, a = 0.5 * dfn, b = 0.5 * dfd;
    if (!(mu <= lite::kLiteArg && a <= 1e6 && b <= 1e6)) return false;
    const distr_detail::Arg g = distr_detail::f_arg(dfn, dfd, x);
    if (!(g.x.hi >= 0x1p-500 && g.y.hi >= 0x1p-500)) return false;
    const double nl = lite_terms_beta(a, b, g.x.hi, g.y.hi), nu = lite_terms_weights(mu);
    if (!(nl <= lite::kLiteTerms || nu <= lite::kLiteTerms)) return false;
    const lite::BetaRatio tr(a, b, g.x);
    const double tw = beta_first(a, b, g.x, g.y, dd::DD{mu, 0.0});  // T(a) e^-mu
    if (!(tw >= 0x1p-900)) return false;
    double dens = 0.0;
    auto lower = [&](double& v) {
        double s;
        if (!lite::lower_sum(tr, dd::DD{mu, 0.0}, 0.0, std::exp(mu) * (1.0 + 0x1p-40), s, dens)) return false;
        v = at_most_one(tw * s);
        return v >= 0x1p-1000;
    };
    auto upper_of = [&](double& v) {
        double s;
        // 1 - I_w(a, b): the central F's upper tail (fdtrc's kernels: for small shapes more accurate than
        // ibeta alone, which was 3.6e-15 off at a = 0.04, b = 0.7)
        const double q0 = distr_detail::f_value(dfn, dfd, x, true);
        if (!(q0 >= kTinyG) || !lite::upper_sum(tr, dd::DD{mu, 0.0}, 0.0, std::exp(-mu) / tw, s, dens)) return false;
        v = at_most_one(q0 + tw * s);
        return v >= 0x1p-1000;
    };
    // log x as a normal variable: the numerator's chi-square over dfn has mean m = 1 + nc / dfn
    // and relative variance vn = 2 (dfn + 2 nc) / (dfn + nc)^2, so its log is about
    // log m - vn / 2 with variance vn; the denominator's log has mean about -1 / dfd and
    // variance trigamma(dfd / 2) ~ 2 / dfd + 2 / dfd^2. (Judged by the numerator alone, a
    // third of the points of F(3, 5, nc = 300) called "above" have an upper tail over 1/4:
    // a whole upper sum would run for nothing before the far-end walk.)
    const double m = dfn + nc, vn = 2.0 * (m + nc) / (m * m);
    const double z = (std::log(x * dfn / m) + 0.5 * vn - 1.0 / dfd) / std::sqrt(vn + 2.0 * (dfd + 1.0) / (dfd * dfd));
    if (!lite_pick(upper, mu, nl, nu, 28.0, z, lower, upper_of, out)) return false;
    // sum_j D(j, mu) beta(w; a + j, b) dw/dx, beta(w; s, b) = T(s - 1) (s - 1 + b) / v, dw/dx = w v / x
    if (pdf) *pdf = tw * (a + g.x.hi * dens) / x;
    return true;
}

// The side of a mixture that is near 1 is summed across the whole bell of the weights, about
// 19 sqrt(mu) terms, so it runs out of steps first (chi-square from nc ~ 1.47e9, where the
// smaller side still has some 0.6e9 to go): a sum that did not end (NaN) is 1 minus the other
// side when that one is at most 1/2 (its absolute error eps is a relative one of the sum).
// sum(upper, wide) is the mixture; for a wide sum whose other side is not summed either (a
// tail far below e^-2000, its terms' peak out of reach), the other side's probability, 0
// there, still says that this one is 1.
template <class Sum>
inline Scaled or_complement(const Sum& sum, bool upper, bool wide) {
    const Scaled v = sum(upper, wide);
    if (v.m == v.m) return v;
    Scaled o = sum(!upper, wide);
    if (o.m != o.m && wide) o = sum(!upper, false);
    return o.m == o.m && value(o) <= 0.5 ? one_minus(o) : v;
}

// chndtr (upper: chndtrc) for 0 < df, nc, x finite (df = 0 too), as m 2^e (wide: see mixture_s).
inline Scaled ncx2_s(double df, double nc, double x, bool upper, bool wide = false) {
    return or_complement(
        [&](bool u, bool w) { return mixture_s(0.5 * nc, 0.0, 0.5 * df, GammaFamily{0.5 * x, u}, 0.0, w); }, upper,
        wide);
}
inline double ncx2(double df, double nc, double x, bool upper) {
    double out;
    if (ncx2_lite(df, nc, x, upper, out)) return out;
    return at_most_one(value(ncx2_s(df, nc, x, upper)));
}

// ncfdtr (upper: ncfdtrc) for finite positive dfn, dfd, nc and 0 < x < inf, as m 2^e.
inline Scaled ncf_s(double dfn, double dfd, double nc, double x, bool upper, bool wide = false) {
    const distr_detail::Arg g = distr_detail::f_arg(dfn, dfd, x);
    if (g.y.hi < distr_detail::kTinyArg) {  // v = dfd / (dfn x + dfd) below 2^-1000: the upper tail from v^b
        const dd::DD logv = dd::log(dfd) - dd::log(dfn) - dd::log(x);
        const Scaled up = clamp01(tiny_upper_s(0.5 * nc, 0.0, 0.5 * dfn, 0.5 * dfd, logv));
        return upper ? up : one_minus(up);
    }
    if (g.x.hi < distr_detail::kTinyArg) {
        // w below 2^-1000: only the j = 0 term counts (the next carries a factor w), fdtr's own series;
        // wide, from I_w(a, b) = w^a / (a B(a, b)) (1 + O(w)) in logs (fdtr's value may underflow).
        // (The factor is mu w (a + b) / a, with w = dfn x / dfd here: for a dfd so large that it
        // is not below 2^-60, the term counts and the sum is not this one.)
        if (!(0.5 * nc * (x * (dfn / dfd)) * (0.5 * dfn + 0.5 * dfd + 1.0) <= 0x1p-60 * std::min(1.0, 0.5 * dfn))) return kNotSummed;
        Scaled lower = scaled(std::exp(-0.5 * nc) * Fdtr::f(dfn, dfd, x));
        if (wide && !(value(lower) >= 0x1p-1000)) {
            const double a = 0.5 * dfn, b = 0.5 * dfd;
            // (log B from three log Gammas: beyond a shape of 2^40 they cancel past its bits)
            if (a > 0x1p40 || b > 0x1p40) return kNotSummed;
            const dd::DD logw = dd::log(dfn) + dd::log(x) - dd::log(dfd);  // dfn x / dfd below 2^-1000 of 1
            const dd::DD lbeta = lgamma_dd(dd::DD{a, 0.0}) + lgamma_dd(dd::DD{b, 0.0}) - lgamma_dd(dd::two_sum(a, b));
            lower = exp_wide(logw * a - dd::log(a) - lbeta - 0.5 * nc);
        }
        return upper ? one_minus(lower) : lower;
    }
    return or_complement(
        [&](bool u, bool w) { return mixture_s(0.5 * nc, 0.0, 0.5 * dfn, BetaFamily{0.5 * dfd, g.x, g.y, u}, 0.0, w); },
        upper, wide);
}
inline double ncf(double dfn, double dfd, double nc, double x, bool upper) {
    double out;
    if (ncf_lite(dfn, dfd, nc, x, upper, out)) return out;
    return at_most_one(value(ncf_s(dfn, dfd, nc, x, upper)));
}

// The most nodes on a side of the noncentral t's integrals (nct_opposite_s, and sd.stats'
// density): their map v - v* = c A sinh(u / A) is finite while u / A = k kStep / kA < 709,
// k < 22688 (in the region the integrand has faded by k = 200: it falls as e^(df (v - v*))
// at least). An integrand still there at the last node: NaN.
inline constexpr int kNctNodes = 22688;

// e^u - 1 - u, accurate for small u.
inline double em1mx(double u) {
    if (std::fabs(u) > 0.5) return std::expm1(u) - u;
    double term = 0.5 * u * u, sum = term;
    for (int k = 3; k < 30; ++k) {
        term *= u / k;
        sum += term;
        if (std::fabs(term) <= 0x1p-56 * std::fabs(sum)) break;
    }
    return sum;
}

// P(T > t) for the noncentral t with df, nc = -a < 0, t > 0: the tail away from the
// noncentrality, where the series' two families cancel. As an integral over
// S = sqrt(V / df), s = e^v:
//     Q = int Phi(-(a + t s)) f_S(s) ds = int erfcx(z / sqrt2) / 2 e^E(v) dv,  z = a + t e^v,
//     E(v) = -z^2/2 + log(df/pi)/2 - log Gamma*(df/2) - (df/2)(e^2v - 1 - 2v),
// every factor positive. E is split at its stationary point v* (closed form) as
// E* (double-double, so e^E* keeps its last bit whatever its size) plus a remainder
// written without cancellation; the integral is the trapezoid rule in u over
// v - v* = c A sinh(u / A), c = min(sigma, 0.4), A = 8 (sigma = 1/sqrt(-E''(v*)); see
// the loop for why the step is capped and the map is linear over the peak).
inline Scaled nct_opposite_s(double df, double a, double t) {
    // (t^2 + df) s^2 + a t s - df = 0; with t taken out where a t + root passes the doubles
    // (t next to the largest of them: the far tail of a small df is no 0 there)
    const double wide = std::sqrt(a * a + 4.0 * df);
    const double den = a * t + std::hypot(t * wide, 2.0 * df);
    const double sstar = std::isfinite(den) ? 2.0 * df / den : 2.0 * df / t / (a + std::hypot(wide, 2.0 * df / t));
    const double p = t * sstar;
    const dd::DD zdd = dd::two_prod(t, sstar) + a;
    const double z = zdd.hi;
    const dd::DD s2 = dd::two_prod(sstar, sstar);
    // E* and E'(v*) (~0 up to rounding: kept, so the remainder below stays exact)
    const dd::DD hstar = s2 - 1.0 - dd::log(sstar) * 2.0;
    const dd::DD Estar = -(zdd * zdd) * 0.5 + (dd::log(df) - dd::log(kPi)) * 0.5 - hstar * (0.5 * df) -
                         std::log(gamma_star(0.5 * df));
    const double c1 = (-(zdd * dd::DD{p, 0.0}) - (s2 - 1.0) * df).hi;
    const double kappa = p * p + z * p + 2.0 * df * sstar * sstar;
    const double sigma = 1.0 / std::sqrt(kappa);
    auto integrand = [&](double delta) {
        const double em = em1mx(delta), q = p * std::expm1(delta);
        const double dE = c1 * delta - z * p * em - 0.5 * q * q - 0.5 * df * (sstar * sstar) * em1mx(2.0 * delta);
        return 0.5 * Erfcx::f((a + p * std::exp(delta)) * 0.70710678118654752440) * std::exp(dE);
    };
    // v - v* = c A sinh(u / A): linear over the peak with the step c kStep, exponentially
    // wider in the tails (the left one falls off only as e^(df (v - v*)), slowly for
    // small df). c = min(sigma, 0.4): besides the peak's width, the step must resolve
    // the e^(2v) in E, whose strip of analyticity is pi/4 wide whatever sigma.
    constexpr double kStep = 0.25, kA = 8.0;  // measured: 1e-17 on the reference cases (kA = 4: 2e-13)
    const double c = std::min(sigma, 0.4);
    // The grid is laid about s* as the peak: E'(v*) = c1 is 0 up to the rounding of s*, and
    // the peak then lies c1 / kappa from it. Within its width sigma the linear part of the map
    // (steps of sigma / 4) covers it; further off (|a| or t beyond 1e15 or so, where s* in a
    // double misses the peak by more than its width) the nodes would step over it: NaN.
    if (!(std::fabs(c1) <= std::sqrt(kappa))) return kNotSummed;
    Sum I;
    I.add(integrand(0.0));
    for (int dir = -1; dir <= 1; dir += 2) {
        bool faded = false;
        for (int k = 1; k < kNctNodes; ++k) {
            const double u = dir * k * kStep / kA;
            const double v = integrand(c * kA * std::sinh(u)) * std::cosh(u);
            I.add(v);
            if (!(v > 0x1p-60 * I.total()) && k > 8) {  // (a NaN node ends it: the sum is NaN)
                faded = true;
                break;
            }
        }
        if (!faded) return kNotSummed;
    }
    const double integral = I.total() * kStep * c;
    return exp_wide(Estar.hi) * scaled((1.0 + Estar.lo) * integral);
}
inline double nct_opposite(double df, double a, double t) { return value(nct_opposite_s(df, a, t)); }

// Phi(-z) as m 2^e (wide: Phi(-100) ~ e^-5000): erfcx(z / sqrt 2) / 2 e^(-z^2/2), z^2 exact.
inline Scaled ndtr_neg_s(double z) {
    const double v = distr_detail::one<Ndtr>(-z);
    if (v >= 0x1p-1000 || !(z > 0.0)) return scaled(v);
    return scaled(0.5 * Erfcx::f(z * 0.70710678118654752440)) * exp_wide(dd::two_prod(z, z) * -0.5);
}

// P(T <= t) (upper: P(T > t)) for finite df > 0, nc != 0, t finite, as m 2^e.
inline Scaled nct_s(double df, double nc, double t, bool upper, bool wide = false) {
    if (t < 0.0) return nct_s(df, -nc, -t, !upper, wide);  // T(nc) = -T(-nc)
    if (t == 0.0) return upper ? ndtr_neg_s(-nc) : ndtr_neg_s(nc);
    if (upper && nc < 0.0) return nct_opposite_s(df, -nc, t);
    // The series (t > 0), x = t^2 / (df + t^2), b = df/2, mu = nc^2/2:
    //   lower = Phi(-nc) + (1/2) sum_j [D(j, mu) I_x(j + 1/2, b) + sign(nc) D(j + 1/2, mu) I_x(j + 1, b)]
    //   upper = (1/2) sum_j [D(j, mu) (1 - I_x(j + 1/2, b)) + D(j + 1/2, mu) (1 - I_x(j + 1, b))]   (nc > 0)
    // (lower with nc < 0 is >= Phi(-nc) >= 1/2: its cancellation costs nothing relative.)
    const distr_detail::Arg g = distr_detail::t_arg(df, t);  // g.x = df / (df + t^2), g.y = t^2 / (df + t^2)
    const dd::DD mu = dd::two_prod(nc, nc) * 0.5;
    if (g.x.hi < distr_detail::kTinyArg && nc > 0.0) {
        // df / (df + t^2) below 2^-1000: the far tail on the noncentrality's side from y^b
        const dd::DD logy = dd::log(df) - dd::log(t) * 2.0;
        const Scaled up = clamp01(times(
            add_s(tiny_upper_s(mu.hi, 0.0, 0.5, 0.5 * df, logy, mu.lo), tiny_upper_s(mu.hi, 0.5, 1.0, 0.5 * df, logy, mu.lo)),
            0.5));
        return upper ? up : one_minus(up);
    }
    if (g.x.hi < distr_detail::kTinyArg) return upper ? nct_opposite_s(df, -nc, t) : one_minus(nct_opposite_s(df, -nc, t));
    // t^2 / (df + t^2) below 2^-1000: t S, S = sqrt(V / df), is t to 1e-148 (and t itself below
    // 1e-148 unless df is past 1e296), so T <= t is Z + nc <= t
    if (g.y.hi < distr_detail::kTinyArg) return upper ? ndtr_neg_s(t - nc) : ndtr_neg_s(nc - t);
    const double b = 0.5 * df;
    const BetaFamily fam{b, g.y, g.x, upper};
    const Scaled sp = mixture_s(mu.hi, 0.0, 0.5, fam, mu.lo, wide);
    const Scaled sq = mixture_s(mu.hi, 0.5, 1.0, fam, mu.lo, wide);
    if (upper) return clamp01(times(add_s(sp, sq), 0.5));
    return clamp01(add_s(ndtr_neg_s(nc), times(add_s(sp, times(sq, std::copysign(1.0, nc))), 0.5)));
}

// nctdtr (upper: nctdtrc) by the sums of the bulk (lite), the two families of nct_s's series
// each from j = 0; false where they do not apply (and for the tail away from the
// noncentrality, the integral's). pdf: the density (for the inverses' Newton steps).
//
// The first terms: T(1/2) e^-mu by beta_first (12 units); T(1) D(1/2, mu) = b x y^b |nc|
// sqrt(2/pi) e^-mu in closed form (pow and exp 2 each, the low parts' correction 1, six
// products: 11). With the sums' rounding, the products with them and the two additions:
// 16 and 15 at most, each on its own positive part (Phi(-nc), the third, is ndtr's).
inline bool nct_lite(double df, double nc, double t, bool upper, double& out, double* pdf = nullptr) {
    if (t < 0.0) return nct_lite(df, -nc, -t, !upper, out, pdf);  // T(nc) = -T(-nc): the same density
    if (t == 0.0 || (upper && nc < 0.0)) return false;
    const dd::DD mu = dd::two_prod(nc, nc) * 0.5;
    const double b = 0.5 * df;
    if (!(mu.hi <= lite::kLiteArg && mu.hi >= 0x1p-500 && b <= 1e6)) return false;
    const distr_detail::Arg g = distr_detail::t_arg(df, t);  // g.y = t^2 / (df + t^2): the families' argument
    if (!(g.x.hi >= 0x1p-500 && g.y.hi >= 0x1p-500)) return false;
    const double nl = lite_terms_beta(0.5, b, g.y.hi, g.x.hi), nu = lite_terms_weights(mu.hi);
    if (!(nl <= lite::kLiteTerms || nu <= lite::kLiteTerms)) return false;
    const double tp = beta_first(0.5, b, g.y, g.x, mu);  // T(1/2) D(0, mu)
    const double pw = std::pow(g.x.hi, b), em = std::exp(-mu.hi);
    if (!(pw >= kLitePow && em >= kLitePow)) return false;
    const double corr = 1.0 + ((g.y.lo / g.y.hi + b * (g.x.lo / g.x.hi)) - mu.lo);
    const double ybw = pw * corr * em;                                         // y^b e^-mu
    const double tq = std::fabs(nc) * 0.79788456080286535588 * b * g.y.hi * ybw;  // T(1) D(1/2, mu)
    if (!(tp >= 0x1p-900 && tq >= 0x1p-900)) return false;
    const lite::BetaRatio rp(0.5, b, g.y), rq(1.0, b, g.y);
    const double sign = nc < 0.0 ? -1.0 : 1.0;
    double dp = 0.0, dq = 0.0;
    auto lower = [&](double& v) {
        double sp, sq;
        // the weights' totals over their first: sum_j D(j, mu) = 1, sum_j D(j + 1/2, mu) = erf(sqrt(mu)) <= 1
        const double wq = std::fabs(nc) * 0.79788456080286535588 * em;
        if (!lite::lower_sum(rp, mu, 0.0, (1.0 + 0x1p-40) / em, sp, dp) || !lite::lower_sum(rq, mu, 0.5, 1.0 / wq, sq, dq))
            return false;
        v = at_most_one(value(ndtr_neg_s(nc)) + 0.5 * (tp * sp + sign * (tq * sq)));
        return v >= 0x1p-1000;
    };
    auto upper_of = [&](double& v) {  // nc > 0
        if (nc < 0.0) return false;
        double sp, sq;
        // 1 - I_x(1/2, b) = P(|T0| > t) of the central t (stdtr's kernel); 1 - I_x(1, b) = y^b
        const double qp = 2.0 * distr_detail::t_tail(df, t), qq = pw * (1.0 + b * (g.x.lo / g.x.hi));
        if (!(qp >= kTinyG)) return false;
        const double wq = std::fabs(nc) * 0.79788456080286535588 * em;
        if (!lite::upper_sum(rp, mu, 0.0, em / tp, sp, dp) || !lite::upper_sum(rq, mu, 0.5, wq / tq, sq, dq)) return false;
        const double eq = std::erf(std::fabs(nc) * 0.70710678118654752440);  // sum_j D(j + 1/2, mu)
        v = at_most_one(0.5 * ((qp + tp * sp) + (qq * eq + tq * sq)));
        return v >= 0x1p-1000;
    };
    // T > t is Z + nc > t S with S = sqrt(V / df), mean about 1 - 1 / (4 df) and variance
    // about 1 / (2 df): Z + nc - t S as a normal variable (the width at t, not at nc: for a
    // small df the upper tail is far heavier than the width at nc tells)
    double z = (t * (df > 1.0 ? 1.0 - 0.25 / df : 0.75) - nc) / std::sqrt(1.0 + 0.5 * t * t / df);
    if (df < 2.0 && nc > 0.0) {
        // A small df: S is no normal variable, and the upper tail is S below about
        // sqrt(nc^2 + 1) / t, P(V < y) ~ (y / 2)^(df/2) / Gamma(df/2 + 1) at y = df (nc^2 + 1) / t^2
        // (the Gamma factor is within 13% of 1 here): "above" where that is under 0.2.
        const bool far = 0.5 * df * std::log(0.5 * df * (nc * nc + 1.0) / (t * t)) <= -1.6094379124341003;
        z = far ? 1.0 : std::min(z, 0.75);
    }
    if (!lite_pick(upper, mu.hi, nl, nu, 14.0, z, lower, upper_of, out)) return false;
    // d/dt of the lower series: I_x(s, b)' = T(s - 1) (s - 1 + b) / y dx/dt, dx/dt = 2 x y / t
    if (pdf) *pdf = (tp * (0.5 + g.y.hi * dp) + sign * (tq * (1.0 + g.y.hi * dq))) / t;
    return true;
}
inline double nct(double df, double nc, double t, bool upper) {
    double out;
    if (nct_lite(df, nc, t, upper, out)) return out;
    return value(nct_s(df, nc, t, upper));
}

// A subnormal p has no inverse here: the forward functions return it with one or a few bits,
// and the root found for those is anywhere in a wide range of x (chndtrix(14.38, 2, 5e-324)
// was 9% off, nctdtrit(2, 2, 5e-324) 16%). NaN (the central inverses, which solve in logs,
// serve nc = 0).
inline constexpr double kMinInverseP = 0x1p-1022;

// ---- inverses
//
// The root of g = 0, g the log of the smaller tail minus the log of its target
// (1 - p is exact for p >= 1/2), so a target of 1e-300 is solved as easily as
// 0.3: bracketed from a start by growing steps, narrowed by bisection on a log
// scale (x > 0) or the asinh scale (t real) to a factor of two, then Brent's
// zeroin in x itself to the last bit.

// Brent's zeroin on [a, b] with g(a) < 0 < g(b) (finite), to |dx| <= 2 eps |x|.
// At most 200 steps: the bracket comes in a factor of two wide, 53 bisections from its
// end, and a step that does not halve the one before last is replaced by a bisection, so
// the count stays within a few times that (measured over the guaranteed region: at most
// 16). A NaN from g (a sum it could not finish), or no end by then: NaN.
template <class G>
inline double zeroin(const G& g, double a, double fa, double b, double fb) {
    double c = a, fc = fa, d = b - a, e = d;
    for (int it = 0; it < 200; ++it) {
        if ((fb > 0.0) == (fc > 0.0)) {
            c = a, fc = fa;
            d = e = b - a;
        }
        if (std::fabs(fc) < std::fabs(fb)) {
            a = b, b = c, c = a;
            fa = fb, fb = fc, fc = fa;
        }
        const double tol = 2.0 * 0x1p-53 * std::fabs(b) + 0x1p-1074;
        const double m = 0.5 * (c - b);
        if (std::fabs(m) <= tol || fb == 0.0) return b;
        if (std::fabs(e) >= tol && std::fabs(fa) > std::fabs(fb)) {
            const double s = fb / fa;
            double pp, q;
            if (a == c) {
                pp = 2.0 * m * s, q = 1.0 - s;
            } else {
                const double qq = fa / fc, r = fb / fc;
                pp = s * (2.0 * m * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (pp > 0.0) q = -q;
            else pp = -pp;
            if (2.0 * pp < std::min(3.0 * m * q - std::fabs(tol * q), std::fabs(e * q))) {
                e = d, d = pp / q;
            } else {
                d = m, e = m;
            }
        } else {
            d = m, e = m;
        }
        a = b, fa = fb;
        b += std::fabs(d) > tol ? d : (m > 0.0 ? tol : -tol);
        fb = g(b);
        if (fb != fb) return kNaN;
        // Converging superlinearly: after a step below 3e-9 |b| the secant through the last
        // two points is exact to the last bit (its error ~ the product of the last two
        // errors, < 1e-17), so it ends here instead of closing the bracket (two evaluations
        // fewer). Taken only when it shows that regime: it lies in the bracket as it now
        // stands ([b, c] or [a, b] by fb's sign) and corrects b by under 1% of the last
        // step. Where g is flat or noisy at the root the secant moves by about a whole step,
        // fails that test, and the bracket is closed as before.
        if (std::fabs(b - a) <= 3e-9 * std::fabs(b) && fb != fa && std::isfinite(fb)) {
            const double x = b - fb * (b - a) / (fb - fa);
            const double other = (fb > 0.0) == (fc > 0.0) ? a : c;  // the bracket's far end
            if ((x - b) * (x - other) <= 0.0 && std::fabs(x - b) <= 0.01 * std::fabs(b - a)) return x;
        }
    }
    return kNaN;
}

// g increasing in x; `to_u` / `from_u` the bracketing scale (log, asinh), `step` its first step.
// A NaN from g (a sum it could not finish) is the answer: NaN, at once.
//
// Bounded: the walk away from x0 doubles its step eight times (to 32 on the scale) and both
// scales span under 1460 from end to end of the doubles, so it is at the last finite value
// within 8 + 46 steps (64 more at most where a step is below the spacing of the doubles:
// each of those doubles it): at most kBracketSteps. The bisection halves a bracket at most
// that wide down to 1/2, or to neighbouring doubles: under 64 halvings. Then zeroin.
inline constexpr int kBracketSteps = 256;
template <class G, class To, class From>
inline double solve(const G& g, double x0, const To& to_u, const From& from_u, double step, double lowest,
                    double highest) {
    double glo = g(x0);
    if (glo == 0.0) return x0;
    if (glo != glo) return kNaN;
    double lo = x0, hi = x0, ghi = glo;
    const bool up = glo < 0.0;
    double u = to_u(x0);
    if (!std::isfinite(u)) return kNaN;
    // walk away from x0 until the sign changes; past the last finite value (the largest
    // double, the smallest subnormal) the root is beyond the double range
    const double top = highest > 0.0 ? std::numeric_limits<double>::max() : 0.0;
    const double bottom = lowest < 0.0 ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::denorm_min();
    bool bracketed = false;
    for (int k = 0; k < kBracketSteps; ++k) {
        u += up ? step : -step;
        double x = from_u(u);
        if (up) {
            if (!(x < top)) x = top;
            if (x <= hi) {  // no progress: at the top already, or a step below the subnormals' spacing
                if (hi == top) return highest;
                if (k < 64) step *= 2.0;
                continue;
            }
            lo = hi, glo = ghi, hi = x, ghi = g(x);
            if (ghi != ghi) return kNaN;
            if (ghi >= 0.0) {
                bracketed = true;
                break;
            }
        } else {
            if (!(x > bottom)) x = bottom;
            if (x >= lo) {
                if (lo == bottom) return lowest;
                if (k < 64) step *= 2.0;
                continue;
            }
            hi = lo, ghi = glo, lo = x, glo = g(x);
            if (glo != glo) return kNaN;
            if (glo <= 0.0) {
                bracketed = true;
                break;
            }
        }
        if (k < 8) step *= 2.0;
    }
    if (!bracketed) return kNaN;
    if (glo == 0.0) return lo;
    if (ghi == 0.0) return hi;
    // bisect on the bracketing scale while the bracket is wide
    for (int k = 0; k < 200; ++k) {
        const double ulo = to_u(lo), uhi = to_u(hi);
        if (uhi - ulo <= 0.5 && std::isfinite(glo) && std::isfinite(ghi)) break;
        const double mid = from_u(0.5 * (ulo + uhi));
        if (!(mid > lo && mid < hi)) break;
        const double gm = g(mid);
        if (gm == 0.0) return mid;
        if (gm != gm) return kNaN;
        if (gm < 0.0) lo = mid, glo = gm;
        else hi = mid, ghi = gm;
    }
    if (!std::isfinite(glo) || !std::isfinite(ghi)) return std::isfinite(glo) ? lo : hi;
    return zeroin(g, lo, glo, hi, ghi);
}

// The x > 0 with F(x) = p (Fc = 1 - F), 0 < p < 1.
template <class L, class U>
inline double invert_positive(const L& F, const U& Fc, double p, double x0) {
    const bool low = p <= 0.5;
    const double target = low ? p : 1.0 - p;
    // log(F / target), not log F - log target: at F ~ 1e-238 each log alone carries
    // 546 eps of absolute error, 1e-13 of F (the root moved by 190 ULP)
    auto g = [&](double x) { return low ? std::log(F(x) / target) : std::log(target / Fc(x)); };
    if (!(x0 > 0.0 && x0 < kInf)) x0 = 1.0;
    return solve(g, x0, [](double x) { return std::log(x); }, [](double u) { return std::exp(u); }, 0.125, 0.0, kInf);
}

// The real t with F(t) = p.
template <class L, class U>
inline double invert_real(const L& F, const U& Fc, double p, double t0) {
    const bool low = p <= 0.5;
    const double target = low ? p : 1.0 - p;
    // log(F / target), not log F - log target: at F ~ 1e-238 each log alone carries
    // 546 eps of absolute error, 1e-13 of F (the root moved by 190 ULP)
    auto g = [&](double t) { return low ? std::log(F(t) / target) : std::log(target / Fc(t)); };
    if (!std::isfinite(t0)) t0 = 0.0;
    return solve(g, t0, [](double t) { return std::asinh(t); }, [](double u) { return std::sinh(u); }, 0.125, -kInf,
                 kInf);
}

// Starting points: Patnaik's approximation, chi-square(k, nc) ~ c chi-square(k'),
// c = (k + 2 nc) / (k + nc), k' = (k + nc)^2 / (k + 2 nc).
inline double ncx2_start(double df, double nc, double p) {
    const double c = (df + 2.0 * nc) / (df + nc), k = (df + nc) / c;
    return c * 2.0 * (p <= 0.5 ? Gammaincinv::f(0.5 * k, p) : Gammainccinv::f(0.5 * k, 1.0 - p));
}

}  // namespace nc_detail

// ---------------------------------------------------------- chi-square

// chndtr(df, nc, x) = sum_j Poisson(j; nc/2) P(df/2 + j, x/2). df = 0 is
// allowed (a point mass e^(-nc/2) at 0 plus the rest), as chdtr's df = 0.
inline bool ncx2_edge(double df, double nc, double x, bool upper, double& out) {
    if (std::isnan(df) || std::isnan(nc) || std::isnan(x) || df < 0.0 || nc < 0.0 || x < 0.0) {
        out = kNaN;
    } else if (nc == 0.0) {
        out = upper ? Chdtrc::f(df, x) : Chdtr::f(df, x);
    } else if (std::isinf(x)) {
        out = (std::isinf(df) || std::isinf(nc)) ? kNaN : (upper ? 0.0 : 1.0);
    } else if (std::isinf(df) || std::isinf(nc)) {
        out = upper ? 1.0 : 0.0;  // the distribution runs off to infinity
    } else if (x == 0.0) {
        // P(X <= 0): the point mass e^(-nc/2) for df = 0 (chdtr(0, 0) is NaN, as gammainc(0, 0)), else 0
        out = df == 0.0 ? kNaN : (upper ? 1.0 : 0.0);
    } else if (x < 0x1p-1021) {
        // x/2 below the normals: only the j = 0 term counts (the next carries a factor x/2),
        // e^-mu P; the upper tail as Q + P (1 - e^-mu), both positive (1 - e^-mu P would
        // lose all of it where P is 1 to rounding: df = 0, where it is the point mass's
        // complement 1 - e^-mu)
        const double p = Chdtr::f(df, x);
        out = upper ? Chdtrc::f(df, x) - p * std::expm1(-0.5 * nc) : std::exp(-0.5 * nc) * p;
    } else {
        return false;
    }
    return true;
}

struct Chndtr : Real {
    static constexpr const char* name = "chndtr";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"df", "nc", "x"};
    static constexpr double cost_ns = cost::kChndtr;
    static double f(double df, double nc, double x) {
        double out;
        if (ncx2_edge(df, nc, x, false, out)) return out;
        return nc_detail::ncx2(df, nc, x, false);
    }
};

struct Chndtrc : Real {
    static constexpr const char* name = "chndtrc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"df", "nc", "x"};
    static constexpr double cost_ns = cost::kChndtrc;
    static double f(double df, double nc, double x) {
        double out;
        if (ncx2_edge(df, nc, x, true, out)) return out;
        return nc_detail::ncx2(df, nc, x, true);
    }
};

// chndtrix(df, nc, p): the x with chndtr(df, nc, x) = p.
struct Chndtrix : Real {
    static constexpr const char* name = "chndtrix";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"df", "nc", "p"};
    static constexpr double cost_ns = cost::kChndtrix;
    static double f(double df, double nc, double p) {
        if (std::isnan(df) || std::isnan(nc) || std::isnan(p) || df < 0.0 || nc < 0.0 || p < 0.0 || p > 1.0)
            return kNaN;
        if (nc == 0.0) return 2.0 * Gammaincinv::f(0.5 * df, p);
        if (p == 0.0) return 0.0;
        if (p == 1.0 || std::isinf(df) || std::isinf(nc)) return kInf;
        if (p < nc_detail::kMinInverseP) return kNaN;
        if (df == 0.0 && p <= std::exp(-0.5 * nc)) return 0.0;  // inside the point mass at 0
        return nc_detail::invert_positive([&](double x) { return Chndtr::f(df, nc, x); },
                                          [&](double x) { return Chndtrc::f(df, nc, x); }, p,
                                          nc_detail::ncx2_start(df, nc, p));
    }
};

// ---------------------------------------------------------- Student t

// nctdtr(df, nc, t) = P(T <= t), T = (Z + nc) / sqrt(V / df). df = inf is
// the normal distribution shifted by nc.
// A noncentrality whose square is below the doubles (the series' mu = nc^2 / 2): the central
// distribution, to 1e-150 of the value (d log P / d nc is of order 1 at any t;
// nctdtr(0.5, -1e-300, -1e300) was 10% off, nctdtr(1e-300, 5e-324, t) 0.46 for 1/2).
inline constexpr double kCentralNc = 0x1p-500;
inline bool nct_edge(double df, double nc, double t, bool upper, double& out) {
    if (std::isnan(df) || std::isnan(nc) || std::isnan(t) || !(df > 0.0)) {
        out = kNaN;
    } else if (std::fabs(nc) < kCentralNc) {
        out = upper ? Stdtr::f(df, -t) : Stdtr::f(df, t);
    } else if (std::isinf(t)) {
        out = std::isinf(nc) && (nc > 0) == (t > 0) ? kNaN : ((t > 0) != upper ? 1.0 : 0.0);
    } else if (std::isinf(nc)) {
        out = (nc > 0) != upper ? 0.0 : 1.0;
    } else if (std::isinf(df)) {
        out = distr_detail::one<Ndtr>(upper ? nc - t : t - nc);
    } else {
        return false;
    }
    return true;
}

struct Nctdtr : Real {
    static constexpr const char* name = "nctdtr";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"df", "nc", "t"};
    static constexpr double cost_ns = cost::kNctdtr;
    static double f(double df, double nc, double t) {
        double out;
        if (nct_edge(df, nc, t, false, out)) return out;
        return nc_detail::nct(df, nc, t, false);
    }
};

struct Nctdtrc : Real {
    static constexpr const char* name = "nctdtrc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"df", "nc", "t"};
    static constexpr double cost_ns = cost::kNctdtrc;
    static double f(double df, double nc, double t) {
        double out;
        if (nct_edge(df, nc, t, true, out)) return out;
        return nc_detail::nct(df, nc, t, true);
    }
};

struct Nctdtrit : Real {
    static constexpr const char* name = "nctdtrit";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"df", "nc", "p"};
    static constexpr double cost_ns = cost::kNctdtrit;
    static double f(double df, double nc, double p) {
        if (std::isnan(df) || std::isnan(nc) || std::isnan(p) || !(df > 0.0) || p < 0.0 || p > 1.0) return kNaN;
        if (std::fabs(nc) < kCentralNc) return Stdtrit::f(df, p);
        if (p == 0.0) return -kInf;
        if (p == 1.0) return kInf;
        if (std::isinf(nc)) return nc;
        if (p < nc_detail::kMinInverseP) return kNaN;
        if (std::isinf(df)) return nc + distr_detail::one<Ndtri>(p);
        // start: P(T <= t) ~ Phi((t (1 - 1/(4 df)) - nc) / sqrt(1 + t^2 / (2 df))), solved for t
        const double z = distr_detail::one<Ndtri>(p), A = 1.0 - 0.25 / df, qa = A * A - z * z / (2.0 * df);
        double t0 = nc + z;
        if (qa > 0.0) {
            const double disc = A * A * nc * nc - qa * (nc * nc - z * z);
            if (disc >= 0.0) t0 = (A * nc + std::copysign(std::sqrt(disc), z)) / qa;
        }
        return nc_detail::invert_real([&](double t) { return Nctdtr::f(df, nc, t); },
                                      [&](double t) { return Nctdtrc::f(df, nc, t); }, p, t0);
    }
};

// ------------------------------------------------------------------ F

// ncfdtr(dfn, dfd, nc, f) = sum_j Poisson(j; nc/2) I_w(dfn/2 + j, dfd/2),
// w = dfn f / (dfn f + dfd). dfd = inf: dfn F -> noncentral chi-square(dfn, nc);
// dfn = inf: the numerator chi-square / dfn -> 1, as the central F.
inline bool ncf_edge(double dfn, double dfd, double nc, double x, bool upper, double& out) {
    if (std::isnan(dfn) || std::isnan(dfd) || std::isnan(nc) || std::isnan(x) || !(dfn > 0.0) || !(dfd > 0.0) ||
        nc < 0.0 || x < 0.0) {
        out = kNaN;
    } else if (nc == 0.0) {
        out = upper ? Fdtrc::f(dfn, dfd, x) : Fdtr::f(dfn, dfd, x);
    } else if (std::isinf(x)) {
        out = std::isinf(nc) ? kNaN : (upper ? 0.0 : 1.0);
    } else if (x == 0.0) {
        out = upper ? 1.0 : 0.0;
    } else if (std::isinf(nc)) {
        out = upper ? 1.0 : 0.0;
    } else if (std::isinf(dfd)) {
        out = upper ? Chndtrc::f(dfn, nc, dfn * x) : Chndtr::f(dfn, nc, dfn * x);
    } else if (std::isinf(dfn)) {
        out = upper ? Fdtrc::f(dfn, dfd, x) : Fdtr::f(dfn, dfd, x);
    } else {
        return false;
    }
    return true;
}

struct Ncfdtr : Real {
    static constexpr const char* name = "ncfdtr";
    static constexpr int nin = 4, nout = 1;
    static constexpr std::array<const char*, 4> args{"dfn", "dfd", "nc", "f"};
    static constexpr double cost_ns = cost::kNcfdtr;
    static double f(double dfn, double dfd, double nc, double x) {
        double out;
        if (ncf_edge(dfn, dfd, nc, x, false, out)) return out;
        return nc_detail::ncf(dfn, dfd, nc, x, false);
    }
};

struct Ncfdtrc : Real {
    static constexpr const char* name = "ncfdtrc";
    static constexpr int nin = 4, nout = 1;
    static constexpr std::array<const char*, 4> args{"dfn", "dfd", "nc", "f"};
    static constexpr double cost_ns = cost::kNcfdtrc;
    static double f(double dfn, double dfd, double nc, double x) {
        double out;
        if (ncf_edge(dfn, dfd, nc, x, true, out)) return out;
        return nc_detail::ncf(dfn, dfd, nc, x, true);
    }
};

struct Ncfdtri : Real {
    static constexpr const char* name = "ncfdtri";
    static constexpr int nin = 4, nout = 1;
    static constexpr std::array<const char*, 4> args{"dfn", "dfd", "nc", "p"};
    static constexpr double cost_ns = cost::kNcfdtri;
    static double f(double dfn, double dfd, double nc, double p) {
        if (std::isnan(dfn) || std::isnan(dfd) || std::isnan(nc) || std::isnan(p) || !(dfn > 0.0) || !(dfd > 0.0) ||
            nc < 0.0 || p < 0.0 || p > 1.0)
            return kNaN;
        if (nc == 0.0) return Fdtri::f(dfn, dfd, p);
        if (p == 0.0) return 0.0;
        if (p == 1.0 || std::isinf(nc)) return kInf;
        if (p < nc_detail::kMinInverseP) return kNaN;
        if (std::isinf(dfd)) return Chndtrix::f(dfn, nc, p) / dfn;
        if (std::isinf(dfn)) return Fdtri::f(dfn, dfd, p);
        // start: dfn F ~ c chi-square(k') over chi-square(dfd) / dfd (Patnaik), a central F
        const double c = (dfn + 2.0 * nc) / (dfn + nc), k = (dfn + nc) / c;
        const double x0 = c * k / dfn * Fdtri::f(k, dfd, p);
        return nc_detail::invert_positive([&](double x) { return Ncfdtr::f(dfn, dfd, nc, x); },
                                          [&](double x) { return Ncfdtrc::f(dfn, dfd, nc, x); }, p, x0);
    }
};

namespace nc_detail {

// ---- the tails as m 2^e over the whole domain (sd.stats' log forms and inverses, whose
// values and targets reach far below the double range): the probability's own edges, then
// the wide sums (wide = false: the probability's sums, 0 for a tail below e^-2000 instead
// of its value). NaN: a sum that was not finished (kMaxTerms), or, wide, a tail below
// e^-2000 whose terms peak beyond the walk's reach (the chi-square's upper tail at
// nc x > 4e16).

inline Scaled chndtr_tail(double df, double nc, double x, bool upper, bool wide = true) {
    if (df > 0.0 && std::isfinite(df) && nc > 0.0 && std::isfinite(nc) && x > 0.0 && x < 0x1p-1021) {
        // x/2 below the normals (x/2 itself would round): the j = 0 term, P(a, y) = y^a / Gamma(a + 1) (1 + O(y)),
        // e^(-nc/2) times that in logs
        const double a = 0.5 * df;
        const Scaled lower = exp_wide((dd::log(x) - dd::kLn2) * a - lgamma_dd(dd::DD{a, 0.0} + 1.0) - 0.5 * nc);
        return upper ? one_minus(lower) : lower;
    }
    double out;
    if (ncx2_edge(df, nc, x, upper, out)) return scaled(out);
    return clamp01(ncx2_s(df, nc, x, upper, wide));
}
inline Scaled nctdtr_tail(double df, double nc, double t, bool upper, bool wide = true) {
    double out;
    if (nct_edge(df, nc, t, upper, out)) return scaled(out);
    return nct_s(df, nc, t, upper, wide);
}
inline Scaled ncfdtr_tail(double dfn, double dfd, double nc, double x, bool upper, bool wide = true) {
    double out;
    if (ncf_edge(dfn, dfd, nc, x, upper, out)) return scaled(out);
    return clamp01(ncf_s(dfn, dfd, nc, x, upper, wide));
}

}  // namespace nc_detail

}  // namespace sd::special
