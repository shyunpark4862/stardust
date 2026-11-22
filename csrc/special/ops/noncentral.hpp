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
#include "special/ops/expint.hpp"

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
// mixture). w and T follow their ratios in "double + error" arithmetic (lite::Lite, ~2^-100 a
// step), so rounding cannot drift.
// The Boost.Math versions (what scipy 1.18 calls) were measured first and lose the
// far tails (zeros, up to 8.8e15 ULP): not used (docs/design/special.md §5.16).
//
// In the bulk (few terms, values inside a double's range) the same sums are taken from
// j = 0 instead, without the far end's incomplete function (`lite` below); the probability
// functions try that first, the wide tails (sd.stats' log forms and inverses) never do.

inline constexpr double kTol = 0x1p-56;
// The least beta shape (half a degrees of freedom) the bulk's sums (lite) and plain_walk take:
// above 0. The half of the smallest df, 5e-324, is 0 (dfd: halvable_dfn moves only dfn), a
// shape b = 0 that first_factor takes for the gamma family's mark (ncfdtr(5e-324, 5e-324, 1,
// 0.25) was e^-1/2 with ncfdtrc 1, no longer their complements): mixture_walk alone, as before.
inline constexpr double kMinShape = 0x1p-1074;

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


// "Double + error" arithmetic (Lite: h + l, l a few ULP of h, not renormalized; a product keeps
// the exact fma remainder and the cross terms, dropping only l l, relative 2^-100): the walk
// below and the sums of the bulk (lite::lower_sum, upper_sum) follow their ratios in it.
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

// s += d for d = +-1 (two_sum of the high parts)
inline void step(Lite& s, double d) {
    const double v = s.h + d, bb = v - s.h;
    s.l += (s.h - (v - bb)) + (d - bb);
    s.h = v;
}
inline Lite of(dd::DD v) { return {v.hi, v.lo}; }
// a c for an exact double c
inline Lite mul_d(Lite a, double c) {
    const double p = a.h * c;
    return {p, std::fma(a.h, c, -p) + a.l * c};
}

}  // namespace lite

// P(s, y) or Q(s, y) along s; T(s) = D(s, y).
struct GammaFamily {
    static constexpr bool kBeta = false;
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
        const CfResult tail = cf_fast(
            y + 1.0 - s,
            [&](int n) {
                const double k = n;
                return std::pair{-k * (k - s), y + 2.0 * k + 1.0 - s};
            },
            100000
        );
        if (!tail.converged) return kNotSummed;
        return d * scaled(s / tail.value);
    }
    Scaled G(dd::DD s) const {
        return at_shape([&](double v) { return G0(v); }, s);
    }
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
    // plain_walk's T ratios, from s (exact as hi + lo) in turn: up, T(s+1) / T(s) = y / (s + 1),
    // then s + 1; down, T(s-1) / T(s) = s / y, then s - 1. The shape's low part enters each
    // ratio to first order: s0 + j rounded is off by the same amount at every j of a binade,
    // and that error would add up along the walk (100 ULP and more over a few hundred steps).
    // For the same reason each ratio is rounded once from its exact value (a quotient through
    // its remainder, a product by fma): a correction below half an ULP added after a rounding
    // is lost every time, the same way (measured: d / mu as d (1/mu).hi + d (1/mu).lo drifted by
    // -0.45 ULP a step).
    //
    // Each ratio is taken at s + k or s - k directly (k the step's count, the shape exact by
    // two_sum), with no state carried from step to step: the walk computes a block of them ahead,
    // independent of each other, so that a division's latency (~18 cycles here) is not on the
    // terms' chain (measured: 3x the step's time when it was).
    struct PlainSteps {
        double ah, al;  // s + 1 (up) or s (down), exact as hi + lo
        dd::DD iy;
        double y;
        // T(s'+1) / T(s') at s' = s + k: y / a, the quotient corrected by its exact remainder (and
        // a's low part); with it the weights' ratio rw = mu / dj (+ mu_lo) the same way, one
        // division for both
        double up(double k, double mu, double mu_lo, double dj, double& rw) const {
            const dd::DD a = dd::two_sum(ah, k);
            const double alo = a.lo + al;
            const double q = 1.0 / (a.hi * dj), qa = dj * q, qj = a.hi * q;
            const double r = y * qa, e = std::fma(-r, a.hi, y) - r * alo;
            const double v = mu * qj;
            rw = v + (std::fma(-v, dj, mu) + mu_lo) * qj;
            return r + e * qa;
        }
        // T(s'-1) / T(s') at s' = s - k: a / y, one rounding of the exact product and the low parts
        double down(double k) const {
            const dd::DD a = dd::two_sum(ah, -k);
            const double alo = a.lo + al;
            return std::fma(a.hi, iy.hi, a.hi * iy.lo + alo * iy.hi);
        }
    };
    PlainSteps plain_steps(dd::DD s, bool up) const {
        const dd::DD a = dd::two_sum(s.hi, up ? 1.0 : 0.0);
        return {a.hi, a.lo + s.lo, inv_y, y};
    }
    // The walk's way out (mixture_walk): mu / q times gbound at shape s (q = j + 1 + h), and
    // wr times gbound_down, each with a relative margin of 2^-40 over its rounding, in one
    // division at most.
    double out_up(double mu, double q, double s) const {
        const double d = s + 1.0;
        return (y < d ? (mu * y) / (q * d) : mu / q) * (1.0 + 0x1p-40);
    }
    double out_down(double wr, double s) const {
        return s >= 1.0 && y > s - 1.0 ? wr * std::min(1.0, (s - 1.0) * inv_y.hi * (1.0 + 0x1p-40)) : wr;
    }
    // out_up and out_down from the walk's own ratios (plain_walk): rw times gbound (rt, T's
    // ratio up, is that bound: out_up_simple), or rw times gbound_down, with the same margin
    static constexpr bool out_up_simple() { return true; }
    static constexpr bool plain_shapes_ok() { return true; }
    // plain_walk's density sum_j w_j T(s_j) s_j times this is mixture_walk's (T(s - 1) = T(s) s / y)
    double dens_factor() const { return inv_y.hi; }
    double out_down_r(double rw, double s) const {
        return (s >= 1.0 && y > s - 1.0 ? rw * std::min(1.0, (s - 1.0) * inv_y.hi) : rw) * (1.0 + 0x1p-40);
    }
    // The walk's T ratios in turn from shape s: lower, T(s) / T(s+1) for s, s - 1, ... (times
    // 2^e, constant: y below 2^-900 scaled into [1, 2)); upper, T(s+1) / T(s) for s, s + 1, ...
    struct Steps {
        lite::Lite n, c;  // s + 1; 1/y (lower) or y
        bool lower;
        lite::Lite next() {
            if (lower) {
                const lite::Lite r = lite::mul(n, c);
                lite::step(n, -1.0);
                return r;
            }
            const lite::Lite r = lite::quot(c, n);
            lite::step(n, 1.0);
            return r;
        }
    };
    Steps steps(bool lower, dd::DD s, int& e) const {
        lite::Lite n = lite::sum2(s.hi, 1.0);
        n.l += s.lo;
        e = 0;
        if (!lower) return {n, {y, 0.0}, false};
        if (!(y < 0x1p-900)) return {n, lite::of(inv_y), true};
        const int ey = std::ilogb(y);
        e = -ey;
        return {n, lite::of(recip(dd::DD{std::ldexp(y, -ey), 0.0})), true};
    }
};

// I_x(s, b) or 1 - I_x(s, b) along s; T(s) = x^s y^b / (s B(s, b)).
struct BetaFamily {
    static constexpr bool kBeta = true;
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
    Scaled G(dd::DD s) const {
        return at_shape([&](double v) { return G0(v); }, s);
    }
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
    // as GammaFamily's: for b < 1 the factor e^u, u = (1 - b) / (s + b), by 1 / (1 - u) >= e^u
    // where u <= 1/2
    double out_up(double mu, double q, double s) const {
        double num = x.hi * (s + b), den = s + 1.0;
        if (b < 1.0) {
            const double v = s + (2.0 * b - 1.0);  // (s + b) (1 - u)
            if (!(v >= 0.5 * (s + b))) return mu / q * gbound(dd::DD{s, 0.0}) * (1.0 + 0x1p-40);
            num *= s + b, den *= v;
        }
        return (num < den ? (mu * num) / (q * den) : mu / q) * (1.0 + 0x1p-40);
    }
    double out_down(double wr, double) const { return wr; }
    // (plain_walk, as GammaFamily's; b < 1 takes out_up's own factor)
    bool out_up_simple() const { return b >= 1.0; }
    bool plain_shapes_ok() const { return b >= kMinShape; }
    static constexpr double dens_factor() { return 1.0; }
    double out_down_r(double rw, double) const { return rw * (1.0 + 0x1p-40); }
    // as GammaFamily's: up, T(s+1) / T(s) = x (s + b) / (s + 1); down, T(s-1) / T(s) =
    // s / (x (s - 1 + b)): c n / d either way, n and d exact as hi + lo at the step. The quotient
    // of the exact product c n.hi through its remainder, the low parts (c's, n's, d's) to first
    // order. (Not as c + c (b - 1) / d: down, that cancels by b / s.)
    struct PlainSteps {
        double nh, nl, dh, dl;  // up: s + b, s + 1; down: s, s - 1 + b
        double c, crel;  // x (up) or 1/x (down), and its low part's share
        // c n / d at the shifted counters, for q ~ 1 / d.hi
        double ratio(const dd::DD& n, double nlo, const dd::DD& d, double dlo, double q) const {
            const double p = c * n.hi, pl = std::fma(c, n.hi, -p) + (c * nlo + p * crel);
            const double r = p * q;
            return r + ((std::fma(-r, d.hi, p) + pl) - r * dlo) * q;
        }
        double up(double i, double mu, double mu_lo, double dj, double& rw) const {  // (rw as GammaFamily's)
            const dd::DD n = dd::two_sum(nh, i), d = dd::two_sum(dh, i);
            const double q = 1.0 / (d.hi * dj), qd = dj * q, qj = d.hi * q;
            const double v = mu * qj;
            rw = v + (std::fma(-v, dj, mu) + mu_lo) * qj;
            return ratio(n, n.lo + nl, d, d.lo + dl, qd);
        }
        double down(double i) const {
            const dd::DD n = dd::two_sum(nh, -i), d = dd::two_sum(dh, -i);
            return ratio(n, n.lo + nl, d, d.lo + dl, 1.0 / d.hi);
        }
    };
    PlainSteps plain_steps(dd::DD s, bool up) const {
        // s - 1 + b: b - 1 exactly first (two_sum), then the shape
        const dd::DD bm1 = dd::two_sum(b, -1.0);
        const dd::DD n = dd::two_sum(s.hi, up ? b : 0.0);
        const dd::DD d = up ? dd::two_sum(s.hi, 1.0) : dd::two_sum(s.hi, bm1.hi);
        const double nlo = n.lo + s.lo, dlo = d.lo + s.lo + (up ? 0.0 : bm1.lo);
        if (up) return {n.hi, nlo, d.hi, dlo, x.hi, x.lo / x.hi};
        return {n.hi, nlo, d.hi, dlo, inv_x.hi, inv_x.lo / inv_x.hi};
    }
    // as GammaFamily's: lower, T(s) / T(s+1) = (s + 1) / (x (s + b)); upper, x (s + b) / (s + 1)
    struct Steps {
        lite::Lite n, d, c;  // s + 1, s + b; 1/x (lower) or x
        bool lower;
        lite::Lite next() {
            if (lower) {
                // (d = s + b falls to s0 + b, which may be far below the 1s the steps took off it:
                // its high part then is 0 or rounding and the value in the low part, hence the
                // renormalization; ncfdtr(1e-100, 1e-100, 3000, 0.25) was NaN)
                const lite::Lite r = lite::quot(lite::mul(n, c), lite::sum2(d.h, d.l));
                lite::step(n, -1.0), lite::step(d, -1.0);
                return r;
            }
            const lite::Lite r = lite::quot(lite::mul(c, d), n);
            lite::step(n, 1.0), lite::step(d, 1.0);
            return r;
        }
    };
    Steps steps(bool lower, dd::DD s, int& e) const {
        lite::Lite n = lite::sum2(s.hi, 1.0), d = lite::sum2(s.hi, b);
        n.l += s.lo, d.l += s.lo;
        e = 0;
        return {n, d, lower ? lite::of(inv_x) : lite::of(x), lower};
    }
    // log of a bound on G(s): I_x(s, b) <= T(s) / (1 - r) for the ratio bound r < 1 (gbound's),
    // 1 - I_x(s, b) = I_y(b, s) likewise with the roles turned; else 1
    bool deep(double) const { return x.hi < 0x1p-20 || y.hi < 0x1p-20; }
    // (rough logs, for the e^-2000 test: lgamma. From a shape of 2^40 the three lgammas
    // cancel to nothing, 2^-52 b log b: the prefix's exponent there.)
    double log_gbound(double s) const {
        if (s > 0x1p40 || b > 0x1p40) {
            if (!upper) {
                const double r = x.hi * (s + b) / (s + 1.0) * (b < 1.0 ? std::exp((1.0 - b) / (s + b)) : 1.0);
                return r < 1.0 ? log_abs(beta_prefix_w(s, b, beta_exponent(s, b, x, y)) / scaled(s)) - std::log1p(-r)
                               : 0.0;
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
//
// walk_jstar finds j* (and the early ends: false, with the sum's value in `early`); mixture_walk
// and plain_walk below share it.
template <class Fam>
bool walk_jstar(double mu, double h, double s0, const Fam& f, bool wide, double& jstar, Scaled& early) {
    const bool lower = !f.upper;
    auto shape = [&](double j) { return dd::two_sum(s0, j); };  // s0 + j exactly
    auto term_ratio = [&](double j) {  // (term j+1) / (term j), with G's ratio taken as T's (deep tails)
        return mu / (j + 1.0 + h) * (lower ? f.ratio(shape(j)) : (s0 + j > 0 ? f.ratio(shape(j - 1.0)) : 1.0));
    };
    // j*: the weights' mode, moved to where term_ratio crosses 1 if that is further
    // into the tail (below the mode for a lower family, above it for an upper one).
    const double jmode = std::max(0.0, std::floor(mu - h));
    early = kNotSummed;
    if (!(jmode < 0x1p52)) return false;
    // The crossing matters only on the tail side of the mode: below it for a lower family
    // (term_ratio(jmode) <= 1), above it for an upper one (> 1). Else j* is the mode itself
    // (one ratio instead of a bisection: most inputs).
    jstar = jmode;
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
            if (term_ratio(hi) > 1.0) {
                if (!wide) early = scaled(0.0);
                return false;
            }
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
        if (!wide) {
            early = scaled(0.0);
            return false;
        }
        if (jstar > 1e8) return false;  // a walk of 1e8 terms (a tail at x ~ 1e100): not taken
    }
    return true;
}

template <class Fam>
Scaled mixture_walk(double mu, double h, double s0, const Fam& f, double mu_lo, bool wide, Scaled* dens = nullptr) {
    const bool lower = !f.upper;
    if (dens) *dens = {kNaN, 0};  // (the early ends below: no density)
    auto shape = [&](double j) { return dd::two_sum(s0, j); };  // s0 + j exactly
    double jstar;
    Scaled early;
    if (!walk_jstar(mu, h, s0, f, wide, jstar, early)) return early;

    // J: past j* on the unstable side (up for a lower family, down for an upper one)
    // until the bound times the bounded tail beyond, prod / (1 - r), is below kTol.
    // (At most kMaxTerms steps: the walk back would take as many.)
    double J = jstar, prod = 1.0;
    bool far = false;
    const double inv_mu_out = (1.0 / mu) * (1.0 + 0x1p-40);
    for (int n = 0; n < kMaxTerms; ++n) {
        double r;
        if (lower) {
            r = f.out_up(mu, J + 1.0 + h, s0 + J);
            J += 1.0;
        } else {
            if (J == 0.0) {
                far = true;
                break;
            }
            r = f.out_down((J + h) * inv_mu_out, s0 + J);  // w_(J-1) / w_J = (J + h) / mu
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
    // w and T follow their ratios in "double + error" arithmetic (lite::Lite: exact to ~2^-100 a
    // step, no drift over the hundreds of steps from J, and nothing to recompute); G and the sum are
    // compensated sums of doubles. w is stored times 2^kw, G and T times 2^kg, the terms,
    // their sum and the last term times 2^(kw + kg). Whichever of w and G leaves
    // [2^-500, 2^500] is brought back to about 1 by its own exponent, and the sum and the
    // last term follow by the step's whole move at once (`moved`): a step of 1 / mu or of
    // (s + 1) / y can be 2^1000 and more (mu or y near the smallest doubles), and moved in
    // two parts they would overflow between the parts.
    std::int64_t kw = -wJ.e, kg = k0 - kw;
    lite::Lite w{frame(wJ, kw), 0.0}, t{frame(tJ, kg), 0.0};
    // mu as mm 2^em with mm in [1, 2) where 1 / mu, or w times it, could leave the doubles
    int em = 0;
    dd::DD mud{mu, mu_lo};
    if (!(mu >= 0x1p-400 && mu <= 0x1p400) && std::isfinite(mu)) {
        em = std::ilogb(mu);
        mud = dd::ldexp(mud, -em);
    }
    const lite::Lite inv_mu = lite::of(recip(mud)), mul_mu = lite::of(mud);
    const int wstep = lower ? em : -em;  // w's exponent moves by this each step (0 unless mu is scaled)
    // T's steps as rt 2^er: (s+1)/y passes the double range at y ~ 1e-308
    int er = 0;
    auto tsteps = f.steps(lower, shape(lower ? J - 2.0 : J), er);
    Sum g;
    g.add(frame(gJ, kg));
    Sum S;
    double prev = w.h * g.total(), j = J;
    S.add(prev);
    // The density (dens): sum_j w_j G'(s0 + j), up to the family's sign and factor (the
    // argument's derivative, plain double: the inverses' Newton steps): T(s - 1) = T(s) s / y
    // for the gamma family, T(s) s for the beta one (s = s0 + j). The far end's own term from
    // T at J - 1 (lower) or J (upper), the one at hand.
    const bool want = dens != nullptr;
    double D = 0.0;
    if (want) {
        const double sJ = s0 + J;
        if constexpr (Fam::kBeta) D = w.h * t.h * (lower ? f.ratio(shape(J - 1.0)) * sJ : sJ);
        else D = w.h * t.h * (lower ? 1.0 : sJ * f.inv_y.hi);
    }
    const double dir = lower ? -1.0 : 1.0;
    auto shift = [](lite::Lite v, int e) { return lite::Lite{std::ldexp(v.h, e), std::ldexp(v.l, e)}; };
    bool ended = false;  // by the terms (their fading or j = 0), not by the count
    for (int n = 0; n < 2 * kMaxTerms; ++n) {  // (back over the way out, and as far beyond j*)
        if (lower && j == 0.0) {
            ended = true;
            break;
        }
        // w's ratio: (j + h) / mu down, mu / (j + 1 + h) up
        const lite::Lite rw = lower ? lite::mul_d(inv_mu, j + h) : lite::quot(mul_mu, lite::Lite{j + 1.0 + h, 0.0});
        const lite::Lite rt = tsteps.next();
        g.add(t.h, t.l);
        w = lite::mul(w, rw);
        j += dir;
        double gv = g.total();
        int moved = wstep;  // the terms' frame kw + kg against the last step's
        kw += wstep;
        if (!(std::fabs(w.h) <= 0x1p500 && std::fabs(w.h) >= 0x1p-500) && w.h != 0.0) {  // w back near 1
            const int e = std::ilogb(w.h);
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
            D = std::ldexp(D, moved);
        }
        const double term = (w.h + w.l) * gv;
        S.add(term);
        // (the T at s0 + j for the beta family's density is t before the step down, after the
        // step up; the gamma family's T(s - 1) the other way round)
        if (want && Fam::kBeta == lower) D += w.h * t.h * (Fam::kBeta ? s0 + j : 1.0);
        t = lite::mul(t, rt);
        if (want && Fam::kBeta != lower) D += w.h * t.h * (Fam::kBeta ? s0 + j : 1.0);
        // past j* and fading: the terms beyond fall at least geometrically (log-concave);
        // or past j* with the terms at 0 (all of them below the frame's range)
        if ((lower ? j < jstar : j > jstar) &&
            ((term < prev && term <= kTol * S.hi && term <= kTol * S.total() * (1.0 - term / prev)) ||
             (term == 0.0 && prev == 0.0))) {
            ended = true;
            break;
        }
        prev = term;
        if (er != 0) {  // G's frame moves by 2^er (the stored t is t rt), and the terms' with it
            kg -= er;
            prev = std::ldexp(prev, -er);
            g.scale(-er), S.scale(-er);
            D = std::ldexp(D, -er);
        }
    }
    if (!ended) return kNotSummed;
    Scaled r = scaled(S.total());
    r.e -= kw + kg;
    if (want) {
        *dens = scaled(D);
        dens->e -= kw + kg;
    }
    return r;
}
// ---- the walk in plain double (the probabilities)
//
// mixture_walk's sum reordered so that no recurrence runs far before it reaches the largest
// terms: w and T start from their direct values at j* (the largest term) and follow their
// ratios outward from there, both ways (a product of ratios in either direction: each step's
// rounding stays a relative ULP of the term, about sqrt(n) ULP n steps out, where the terms
// have faded). G's recurrence is stable only toward j* (it adds T there), so on the far side
// (j* to J) the terms are summed in an order that needs no G but the far end's (summation by
// parts over that side):
//     lower (J > j*):  sum_(j = j*..J) w_j G_j = G_J W_J + sum_(i = j*..J-1) T_i W_i,      W_i = sum_(j = j*..i) w_j
//     upper (J < j*):  sum_(j = J..j*) w_j G_j = G_J W'_J + sum_(i = J..j*-1) T_i W'_(i+1),  W'_i = sum_(j = i..j*) w_j
// (G_j = G_J + sum_(i = j..J-1) T_i, or G_J + sum_(i = J..j-1) T_i), all terms positive. J comes
// from the same rigorous bound as mixture_walk's, taken in the same loop; G at j* is G_J plus
// the T's of the way out, and from j* the near side follows G's own stable recurrence to where
// the terms fade. Three direct values (w and T at j*, G at J) and about 18 sqrt(mu) steps of a
// few plain operations (mixture_walk: 27 sqrt(mu) steps in double + error arithmetic).
//
// Frames: w times 2^kw, T times 2^kt, G times 2^kg (each about 1 at j*: T may lie far below G,
// a tail near 1 whose terms barely move it). Where w climbs past 2^600 on the far side (j* in a
// tail, the weights' mode beyond it) 2^600 moves from w to T, the products unchanged; the sum
// comes out as m 2^e (a probability far below the doubles keeps its exponent). False where a
// frame would not hold or the walk does not end: mixture_walk then.
//
// dens (the inverses' Newton steps): sum_j w_j T(s_j) s_j times the family's dens_factor
// (1 / y for the gamma family), mixture_walk's density.
//
// Its error grows with the walk's length (the terms near j* are a sum over about one width,
// sqrt(mu), of independently rounded ratios: measured 10-20 ULP at p999 for mu ~ 5000, 1000 ULP
// at mu = 1e9): kPlainMu bounds it, mixture_walk beyond.
inline constexpr double kPlainMu = 2e4;
// (the shapes kMinShape and up: kMinShape's own declaration says why)
template <class Fam>
bool plain_walk(double mu, double h, double s0, const Fam& f, double mu_lo, Scaled& out, Scaled* dens = nullptr) {
    const bool lower = !f.upper;
    if (!(mu <= kPlainMu && s0 > 0.0 && f.plain_shapes_ok())) return false;
    double jstar;
    Scaled early;
    if (!walk_jstar(mu, h, s0, f, false, jstar, early)) return false;
    const dd::DD sstar = dd::two_sum(s0, jstar);
    const Scaled ws = prefix_s(jstar + h, mu, mu_lo), ts = f.T(sstar);
    if (!(ws.m > 0.0 && ts.m > 0.0 && std::isfinite(ws.m) && std::isfinite(ts.m))) return false;
    const std::int64_t kw0 = -ws.e, kt0 = -ts.e;
    std::int64_t kw = kw0, kt = kt0;
    // the weights' ratios: mu / (j + 1 + h) up, (j + h) / mu down; mu's low part (the t's
    // nc^2 / 2) in the remainder, 1 / mu as a double-double
    const dd::DD inv_mu = recip(dd::DD{mu, mu_lo});
    // (each rounded once from its exact value, as the T ratios: see GammaFamily::PlainSteps; the
    // ratio up comes with T's, from the same division)
    auto w_down = [&](double j) {  // w_(j-1) / w_j
        const double d = j + h;
        return std::fma(d, inv_mu.hi, d * inv_mu.lo);
    };
    const bool want = dens != nullptr;
    // the far side: S2 = sum T_i W_i (frame kt + kw), Tsum (kt), Wc (kw); D (kt + kw)
    double t = ts.m, w = ws.m, Wc = lower ? 0.0 : w, Tsum = 0.0, S2 = 0.0, J = jstar, prod = 1.0;
    double D = want && !lower ? w * t * (s0 + jstar) : 0.0;  // (the upper side's loop starts past j*)
    const auto far_steps = f.plain_steps(sstar, lower);
    // the ratios in blocks of kBlock ahead (each independent of the others: see PlainSteps), then
    // the block's steps
    constexpr int kBlock = 8;
    double rts[kBlock], rws[kBlock], rbs[kBlock];
    // (out_up's own bound where the ratio is not one, a beta family with b < 1: in a pass of its
    // own, out of the block's loop, which it slows by 2x where it is not even taken)
    const bool simple = f.out_up_simple();
    bool far = false;
    for (int n = 0; n < kMaxTerms && !far;) {
        for (int k = 0; k < kBlock; ++k) {
            const double i = n + k;
            if (lower) {
                const double Jk = jstar + i;
                rts[k] = far_steps.up(i, mu, mu_lo, Jk + 1.0 + h, rws[k]);
                rbs[k] = rws[k] * std::min(1.0, rts[k]) * (1.0 + 0x1p-40);
            } else {  // (past J = 0 unused: the loop ends there first)
                const double Jk = jstar - i;
                rws[k] = w_down(Jk);
                rts[k] = far_steps.down(i);
                rbs[k] = f.out_down_r(rws[k], s0 + Jk);
            }
        }
        if (lower && !simple) [[unlikely]] {
            for (int k = 0; k < kBlock; ++k) rbs[k] = f.out_up(mu, jstar + (n + k) + 1.0 + h, s0 + (jstar + (n + k)));
        }
        for (int k = 0; k < kBlock; ++k, ++n) {
            const double r = rbs[k];
            if (lower) {  // index J: T_J W_J; then on to J + 1
                Wc += w;
                S2 += t * Wc;
                Tsum += t;
                if (want) D += w * t * (s0 + J);
                t *= rts[k];
                w *= rws[k];
                J += 1.0;
            } else {  // on to J - 1: T_(J-1) W'_J, then W'_(J-1)
                if (J == 0.0) {
                    far = true;
                    break;
                }
                t *= rts[k];
                w *= rws[k];
                J -= 1.0;
                S2 += t * Wc;
                Tsum += t;
                Wc += w;
                if (want) D += w * t * (s0 + J);
            }
            if (Wc > 0x1p600) {  // the weights climbing (toward their mode): 2^600 from w to T
                Wc *= 0x1p-600, w *= 0x1p-600, t *= 0x1p600, Tsum *= 0x1p600;
                kw -= 600, kt += 600;
            }
            prod *= r;
            if (r < 1.0 && prod <= kTol * (1.0 - r)) {
                far = true;
                break;
            }
            if (prod > 0x1p500) prod = 0x1p500;
        }
    }
    if (!far || !std::isfinite(S2) || !std::isfinite(Tsum)) return false;
    // the far end: G_J W_J (lower: W_J still needs w_J, and its density term)
    if (lower) {
        Wc += w;
        if (want) D += w * t * (s0 + J);
    }
    const Scaled gJ = f.G(dd::two_sum(s0, J));
    if (!(gJ.m == gJ.m && std::isfinite(gJ.m))) return false;
    auto absolute = [](double v, std::int64_t k) {  // v 2^-k as m 2^e
        Scaled r = scaled(v);
        r.e -= k;
        return r;
    };
    const Scaled gstar = add_s(gJ, absolute(Tsum, kt));  // G at j*
    Scaled sum = add_s(gJ * absolute(Wc, kw), absolute(S2, kw + kt));
    if (!(gstar.m > 0.0)) return false;
    // the near side from j*: w and T from their direct values (frames kw0, kt0), G in its own
    // frame kg; T enters G times 2^(kg - kt0) (<= about 1: T_j <= G_j)
    const std::int64_t kg = -gstar.e;
    if (!(kg - kt0 <= 4)) return false;
    // T below 2^-900 of G (a tail near 1): its additions are nothing (and a subnormal factor would
    // make every step a slow one, 3x); a T that climbs to 2^800 in its frame would count again:
    // mixture_walk then
    const bool t_out = kg - kt0 < -900;
    const double tg = t_out ? 0.0 : std::ldexp(1.0, static_cast<int>(kg - kt0));
    double G = gstar.m;
    t = ts.m, w = ws.m;
    // the sum so far in the near side's frame kw0 + kg (it holds the term at j*, about 1 there)
    const Scaled s_far = sum;
    double S = value(Scaled{s_far.m, s_far.e + kw0 + kg});
    if (!std::isfinite(S)) return false;
    const auto near_steps = f.plain_steps(sstar, !lower);
    double j = jstar, prev = w * G;
    bool ended = false;
    for (int n = 0; n < 2 * kMaxTerms && !ended;) {
        for (int k = 0; k < kBlock; ++k) {
            const double i = n + k;
            if (lower) {  // (past j = 0 unused)
                rws[k] = w_down(jstar - i);
                rts[k] = near_steps.down(i);
            } else {
                rts[k] = near_steps.up(i, mu, mu_lo, jstar + i + 1.0 + h, rws[k]);
            }
        }
        for (int k = 0; k < kBlock; ++k, ++n) {
            if (lower) {  // down: G_(j-1) = G_j + T_(j-1)
                if (j == 0.0) {
                    ended = true;
                    break;
                }
                t *= rts[k];
                w *= rws[k];
                j -= 1.0;
                G += t * tg;
            } else {  // up: G_(j+1) = G_j + T_j
                w *= rws[k];
                j += 1.0;
                G += t * tg;
                t *= rts[k];  // T_j for the next step (and the density)
            }
            if (t_out && t > 0x1p800) return false;
            const double term = w * G;
            S += term;
            if (want) D += w * t * (s0 + j);
            // (the quotient only once the term is below kTol S: a division a step otherwise)
            if ((term < prev && term <= kTol * S && term <= kTol * S * (1.0 - term / prev)) ||
                (term == 0.0 && prev == 0.0)) {
                ended = true;
                break;
            }
            prev = term;
        }
    }
    if (!ended || !std::isfinite(S)) return false;
    out = absolute(S, kw0 + kg);
    if (want) *dens = absolute(D, kw0 + kt0) * scaled(f.dens_factor());
    return true;
}

// mu = 0 (a noncentrality whose half, or half square, is below the doubles) has the
// weights D(j + h, 0): 1 at j + h = 0 and 0 elsewhere. (Apart from the walk, whose code it
// would otherwise change: with this test inside, the lower gamma walk is half as slow again.)
template <class Fam>
Scaled mixture_s(
    double mu, double h, double s0, const Fam& f, double mu_lo = 0.0, bool wide = false, Scaled* dens = nullptr
) {
    if (mu == 0.0) {
        if (dens) *dens = {kNaN, 0};  // (not summed: the inverses' bracketing then)
        return h == 0.0 ? f.G(dd::DD{s0, 0.0}) : scaled(0.0);
    }
    Scaled v;
    if (!wide && plain_walk(mu, h, s0, f, mu_lo, v, dens)) return v;
    return mixture_walk(mu, h, s0, f, mu_lo, wide, dens);
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
    double dweight() const { return 1.0; }  // density(s0 + k + 1) / T(s0 + k), up to the family's factor
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
// e^-mu / (a B(a, b))) by first_factor (below), kept per thread for the
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
// The memos' values: e^(-mu) / Gamma(a + 1) and e^(-mu) Gamma(a + b) / (Gamma(a + 1) Gamma(b)),
// for shapes inside ibeta_detail's direct range by its Gamma = e^S F split (Gamma(z) below 13
// by gamma_small, 1 ULP each, Stirling's S as a double-double above: the shape-only factor
// in ~40 ns), else by dd::loggamma (three of them were 120 of a nc < 1 ncfdtr's 330 ns).
inline double first_factor(double a, double b, dd::DD mu) {
    using namespace ibeta_detail;
    const bool direct =
        a >= kDirectShapeLo && a <= kDirectShapeHi && (b == 0.0 || (b >= kDirectShapeLo && b <= kDirectShapeHi));
    if (direct) {
        const GammaSplit ga1 = gamma_split(dd::two_sum(a, 1.0));
        if (b == 0.0) {
            const dd::DD l = -ga1.s - mu;
            return std::exp(l.hi) * (1.0 + l.lo) / ga1.f;
        }
        const GammaSplit gb = gamma_split(dd::DD{b, 0.0}), gc = gamma_split(dd::two_sum(a, b));
        const dd::DD l = gc.s - ga1.s - gb.s - mu;
        return std::exp(l.hi) * (1.0 + l.lo) * (gc.f / (ga1.f * gb.f));
    }
    const dd::DD l = b == 0.0 ? -dd::loggamma(dd::DD{a, 0.0} + 1.0) - mu
                              : dd::loggamma(dd::two_sum(a, b)) - dd::loggamma(dd::DD{a, 0.0} + 1.0) -
                                    dd::loggamma(dd::DD{b, 0.0}) - mu;
    return std::exp(l.hi) * (1.0 + l.lo);
}
// D(a, y) e^-mu = y^a e^-y e^-mu / Gamma(a + 1), a > 0
inline double gamma_first(double a, double y, dd::DD mu) {
    const double py = std::pow(y, a), ey = std::exp(-y);
    if (lite_normal(py) && ey >= kLitePow && a <= 0x1p20) {
        thread_local Memo3 m;
        if (!m.hit(a, 0.0, mu)) m = {a, 0.0, mu.hi, mu.lo, first_factor(a, 0.0, mu)};
        if (lite_normal(m.v)) return py * ey * m.v;
    }
    return value(gamma_prefix(a, y)) * std::exp(-mu.hi);
}
// x^a y^b e^-mu / (a B(a, b)), x + y = 1 as double-doubles: by the powers (false outside their range)
inline bool beta_first_pow(double a, double b, dd::DD x, dd::DD y, dd::DD mu, double& out) {
    const double px = std::pow(x.hi, a), py = std::pow(y.hi, b);
    if (!(px >= kLitePow && py >= kLitePow && a <= 0x1p20 && b <= 0x1p20)) return false;
    thread_local Memo3 m;
    if (!m.hit(a, b, mu)) m = {a, b, mu.hi, mu.lo, first_factor(a, b, mu)};
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
inline bool lite_pick(
    bool upper,
    double mu,
    double nl,
    double nu,
    double direct,
    double z,
    const L& lower,
    const U& upper_sum_of,
    double& out,
    double walk_fixed = 70.0
) {
    const bool below = z < 0.0, above = z > 0.75;  // (a normal's upper tail at 0.75 is 0.23)
    // The walk (plain_walk) costs about one incomplete function and the prefixes (walk_fixed
    // terms' worth: 70 for the gamma family, more for the beta one and the t's two families) and
    // under a term for each of its own steps, 18 sqrt(mu) (the weights' width, both sides): a sum
    // longer than that is not started (measured over the reference points: the walk was the
    // faster from mu ~ 100 for the gamma family, ~ 20-100 for the beta one and the t).
    const double walk = walk_fixed + 15.0 * std::sqrt(mu);
    const bool l_ok = nl <= std::min<double>(lite::kLiteTerms, walk),
               u_ok = nu + direct <= std::min<double>(lite::kLiteTerms, walk);
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

// The point's distance from the middle of its distribution in standard deviations, roughly
// (a guide to which tail is the small one: lite_pick's choice of sum, and the probabilities'
// choice of the side to sum first, small_side below).
inline double ncx2_z(double df, double nc, double x) { return (x - (df + nc)) / std::sqrt(2.0 * (df + 2.0 * nc)); }
// log x as a normal variable: the numerator's chi-square over dfn has mean m = 1 + nc / dfn
// and relative variance vn = 2 (dfn + 2 nc) / (dfn + nc)^2, so its log is about
// log m - vn / 2 with variance vn; the denominator's log has mean about -1 / dfd and
// variance trigamma(dfd / 2) ~ 2 / dfd + 2 / dfd^2. (Judged by the numerator alone, a
// third of the points of F(3, 5, nc = 300) called "above" have an upper tail over 1/4:
// a whole upper sum would run for nothing before the far-end walk.)
inline double ncf_z(double dfn, double dfd, double nc, double x) {
    const double m = dfn + nc, vn = 2.0 * (m + nc) / (m * m);
    return (std::log(x * dfn / m) + 0.5 * vn - 1.0 / dfd) / std::sqrt(vn + 2.0 * (dfd + 1.0) / (dfd * dfd));
}
// T > t is Z + nc > t S with S = sqrt(V / df), mean about 1 - 1 / (4 df) and variance
// about 1 / (2 df): Z + nc - t S as a normal variable (the width at t, not at nc: for a
// small df the upper tail is far heavier than the width at nc tells). t >= 0.
inline double nct_z(double df, double nc, double t) {
    double z = (t * (df > 1.0 ? 1.0 - 0.25 / df : 0.75) - nc) / std::sqrt(1.0 + 0.5 * t * t / df);
    if (df < 2.0 && nc > 0.0) {
        // A small df: S is no normal variable, and the upper tail is S below about
        // sqrt(nc^2 + 1) / t, P(V < y) ~ (y / 2)^(df/2) / Gamma(df/2 + 1) at y = df (nc^2 + 1) / t^2
        // (the Gamma factor is within 13% of 1 here): "above" where that is under 0.2.
        const bool far = 0.5 * df * std::log(0.5 * df * (nc * nc + 1.0) / (t * t)) <= -1.6094379124341003;
        z = far ? 1.0 : std::min(z, 0.75);
    }
    return z;
}
// The probabilities past the bulk's sums: where the asked tail is the large one by z (past
// 1.5 standard deviations: the other is 0.07 for a normal), the other is summed first and its
// complement taken while it is at most 1/4 (absolute error u there, a relative u of the asked
// tail). The large side's walk spans the weights' whole bell (its G is near 1 at every term);
// the small one's terms are those of its own peak, or none at all (below e^-2000: walk_jstar's
// early end). Measured: nctdtrc(14861, 94.6, 1.74) = 1 took 25 us, its other side 0.1.
inline constexpr double kSideZ = 1.5;
inline constexpr double kPairAbove = 1.0 - 0x1p-20;
//
// The same rule, after the fact, makes the pair consistent near 1: where the asked side came
// out above kPairAbove (z did not foresee it, or the bulk's sums took it directly), the other is
// evaluated and the asked side is 1 minus it. So a tail below 2^-20 and its complement add to 1
// up to the subtraction's rounding (two sums near 1 each a few ULP off gave cdf + ccdf =
// 1 - 5.6e-16 at ncfdtr(1e-100, 1e-100, 3000, 10)); in the bulk the two sums agree to their own
// errors, ~1e-15. (From 3/4 on, the second evaluation cost 25-40% over the reference mix.)
template <class Tail>
inline double small_side(double z, bool upper, const Tail& tail) {
    if (upper ? z < -kSideZ : z > kSideZ) {
        const double o = tail(!upper);
        if (o <= 0.25) return 1.0 - o;
    }
    const double v = tail(upper);
    if (v > kPairAbove && v < 1.0) {  // (1 itself: the other side is below the sums' own error, ~1e-15)
        const double o = tail(!upper);
        if (o <= 0.25) return 1.0 - o;
    }
    return v;
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
        if (!(q0 >= kTinyG || a == 0.0) || !lite::upper_sum(tr, dd::DD{mu, 0.0}, 0.0, std::exp(-mu) / tw, s, dens))
            return false;
        v = at_most_one(q0 + tw * s);
        return v >= 0x1p-1000;
    };
    if (!lite_pick(
            upper, mu, lite_terms_gamma(a, y), lite_terms_weights(mu), 30.0, ncx2_z(df, nc, x), lower, upper_of, out
        ))
        return false;
    if (pdf) *pdf = 0.5 * tw * (a / y + dens);
    return true;
}

// ncfdtr (upper: ncfdtrc) by the sums above (12 units for the first term, 2 for the sum and
// the product: 14; the upper one adds 1 - I_w(a, b), fdtrc's kernel). pdf: the density in x.
inline bool ncf_lite(double dfn, double dfd, double nc, double x, bool upper, double& out, double* pdf = nullptr) {
    const double mu = 0.5 * nc, a = 0.5 * dfn, b = 0.5 * dfd;
    if (!(mu <= lite::kLiteArg && a <= 1e6 && b <= 1e6 && a >= kMinShape && b >= kMinShape)) return false;
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
    if (!lite_pick(upper, mu, nl, nu, 28.0, ncf_z(dfn, dfd, nc, x), lower, upper_of, out, 110.0)) return false;
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
        [&](bool u, bool w) { return mixture_s(0.5 * nc, 0.0, 0.5 * df, GammaFamily{0.5 * x, u}, 0.0, w); }, upper, wide
    );
}
// (one side: the bulk's sums, else the walk; ncx2 pairs the two)
inline double ncx2_one(double df, double nc, double x, bool upper) {
    double out;
    if (ncx2_lite(df, nc, x, upper, out)) return out;
    return at_most_one(value(ncx2_s(df, nc, x, upper)));
}
inline double ncx2(double df, double nc, double x, bool upper) {
    return small_side(ncx2_z(df, nc, x), upper, [&](bool u) { return ncx2_one(df, nc, x, u); });
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
        // (The terms' ratio t_1 / t_0 = mu I_w(a + 1, b) / I_w(a, b) = mu w (a + b) / (a + 1) to
        // first order, with w = dfn x / dfd here; it is taken against the lower tail and, for
        // the upper one 1 - e^-mu ~ mu, against mu: max(1, mu) w (a + b) / (a + 1) <= 2^-60, or
        // the sum is not this one. It was (a + b + 1) / min(1, a): a dfn below ~2^-997 never
        // passed, NaN, while 1e-300 had its value; 2026-11-19.)
        if (!(std::max(1.0, 0.5 * nc) * (x * (dfn / dfd)) * ((0.5 * dfn + 0.5 * dfd) / (0.5 * dfn + 1.0)) <= 0x1p-60))
            return kNotSummed;
        Scaled lower = scaled(std::exp(-0.5 * nc) * Fdtr::f(dfn, dfd, x));
        if (wide && !(value(lower) >= 0x1p-1000)) {
            const double a = 0.5 * dfn, b = 0.5 * dfd;
            // (log B from three log Gammas: beyond a shape of 2^40 they cancel past its bits)
            if (a > 0x1p40 || b > 0x1p40) return kNotSummed;
            const dd::DD logw = dd::log(dfn) + dd::log(x) - dd::log(dfd);  // dfn x / dfd below 2^-1000 of 1
            const dd::DD lbeta = lgamma_dd(dd::DD{a, 0.0}) + lgamma_dd(dd::DD{b, 0.0}) - lgamma_dd(dd::two_sum(a, b));
            lower = exp_wide(logw * a - dd::log(a) - lbeta - 0.5 * nc);
        }
        // the upper tail as its own sum, e^-mu (1 - I_w(a, b)) + (1 - e^-mu) (the j >= 1 terms' 1 - I
        // are 1 to O(w)): as 1 - lower it lost its digits below 1/64 (ncfdtrc(1e-10, 1, 1e-300,
        // 1e-300) was NaN for 3.6e-8)
        if (upper && !wide) return scaled(std::exp(-0.5 * nc) * Fdtrc::f(dfn, dfd, x) - std::expm1(-0.5 * nc));
        return upper ? one_minus(lower) : lower;
    }
    return or_complement(
        [&](bool u, bool w) { return mixture_s(0.5 * nc, 0.0, 0.5 * dfn, BetaFamily{0.5 * dfd, g.x, g.y, u}, 0.0, w); },
        upper, wide
    );
}
inline double ncf_one(double dfn, double dfd, double nc, double x, bool upper) {
    double out;
    if (ncf_lite(dfn, dfd, nc, x, upper, out)) return out;
    return at_most_one(value(ncf_s(dfn, dfd, nc, x, upper)));
}
// A dfd below 2^-999 (b = dfd/2 below 2^-1000, or not even a double: 0.5 * 5e-324 is 0): every
// term past j = 0 is I_w(a + j, b) <= b (-log(1 - w)) <= 2^-1000 * 2100 (w < 1 by a factor
// dfd / (dfn x) of 1e-923 at most), so the sum is e^-mu I_w(a, b), the central F's cdf, to
// 2^-988 absolute (relative wherever the value is above 2^-930), and its complement
// e^-mu (1 - I_w(a, b)) + (1 - e^-mu); the pair adds to 1. (The sums took b as 0 or rounded:
// ncfdtr(1e-320, 1e-320, 1e-300, 1e-300) was 0 with ncfdtrc 0.5, for 0.5 each.)
inline constexpr double kTinyShape = 0x1p-1000;
inline double ncf_tiny_dfd(double dfn, double dfd, double nc, double x, bool upper) {
    const double e = std::exp(-0.5 * nc);
    return upper ? e * Fdtrc::f(dfn, dfd, x) - std::expm1(-0.5 * nc) : e * Fdtr::f(dfn, dfd, x);
}
// A dfn below 2^-71 (a = dfn/2 below 2^-72): a in the terms past j = 0 moves each by a relative
// a |d log I_w(s, b) / ds| <= a |log w| <= 2100 a < 2^-61, so the sum is the j = 0 term, e^-mu times the central F's tail, plus sum_(j >= 1) D(j, mu)
// I_w(j, b) -- a mixture from shape 1 with weights D(j + 1, mu) (h = 1) -- or for w below
// 2^-1000 just the j = 0 term (the rest is O(w)). The pair adds to 1 again (the sums lost
// ncfdtrc(5e-324, 1, 1e-300, 0.25), NaN against ncfdtr's 1).
inline constexpr double kTinyDfnShape = 0x1p-72;
inline bool ncf_tiny_dfn(double dfn, double dfd, double nc, double x, bool upper, double& out) {
    const distr_detail::Arg g = distr_detail::f_arg(dfn, dfd, x);
    if (!(g.y.hi >= distr_detail::kTinyArg)) return false;
    const double mu = 0.5 * nc, e = std::exp(-mu);
    const double first = upper ? e * Fdtrc::f(dfn, dfd, x) : e * Fdtr::f(dfn, dfd, x);
    if (g.x.hi < distr_detail::kTinyArg) {
        out = upper ? first - std::expm1(-mu) : first;
        return true;
    }
    // both sides, the asked one as 1 minus the other where that is the small one (as small_side)
    auto side = [&](bool u) {
        const double f0 = u ? e * Fdtrc::f(dfn, dfd, x) : e * Fdtr::f(dfn, dfd, x);
        return at_most_one(f0 + value(mixture_s(mu, 1.0, 1.0, BetaFamily{0.5 * dfd, g.x, g.y, u})));
    };
    (void)first;
    const double o = side(!upper);
    out = o <= 0.25 ? 1.0 - o : side(upper);
    return out == out;
}
inline double ncf(double dfn, double dfd, double nc, double x, bool upper) {
    if (0.5 * dfd < kTinyShape) return ncf_tiny_dfd(dfn, dfd, nc, x, upper);
    if (double out; 0.5 * dfn < kTinyDfnShape && ncf_tiny_dfn(dfn, dfd, nc, x, upper, out)) return out;
    return small_side(ncf_z(dfn, dfd, nc, x), upper, [&](bool u) { return ncf_one(dfn, dfd, nc, x, u); });
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
// dens (the inverses' Newton steps): -dQ/dt = int s phi(z) f_S(s) ds, the same integral with
// e^E(v) s / sqrt(2 pi) for its integrand (Phi(-z) = erfcx(z / sqrt 2) / 2 e^(-z^2/2)).
inline Scaled nct_opposite_s(double df, double a, double t, Scaled* dens = nullptr) {
    if (dens) *dens = {kNaN, 0};
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
    const dd::DD Estar =
        -(zdd * zdd) * 0.5 + (dd::log(df) - dd::log(kPi)) * 0.5 - hstar * (0.5 * df) - std::log(gamma_star(0.5 * df));
    const double c1 = (-(zdd * dd::DD{p, 0.0}) - (s2 - 1.0) * df).hi;
    const double kappa = p * p + z * p + 2.0 * df * sstar * sstar;
    const double sigma = 1.0 / std::sqrt(kappa);
    const bool want = dens != nullptr;
    double dsum = 0.0;  // the density's integral (plain double: a derivative)
    // (one exponential of delta for all three of its functions: e^d - 1 - d and e^2d - 1 - 2d
    // from xm = e^d - 1 off the series' range, where em1mx forms them the same way)
    auto integrand = [&](double delta, double jac) {
        double xm, ed, em, em2;
        if (std::fabs(delta) > 0.5) {
            ed = std::exp(delta), xm = ed - 1.0;
            em = xm - delta, em2 = xm * (xm + 2.0) - 2.0 * delta;
        } else {
            xm = std::expm1(delta), ed = xm + 1.0;
            em = em1mx(delta), em2 = em1mx(2.0 * delta);
        }
        const double q = p * xm;
        const double dE = c1 * delta - z * p * em - 0.5 * q * q - 0.5 * df * (sstar * sstar) * em2;
        const double e = std::exp(dE);
        if (want) dsum += jac * (sstar * ed * e);
        return 0.5 * Erfcx::f((a + p * ed) * 0.70710678118654752440) * e;
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
    I.add(integrand(0.0, 1.0));
    // The nodes in blocks of kBlock, each evaluated apart from the others (their chains of four
    // exponentials and erfcx overlap: a node at a time took 3x as long), then added in order.
    constexpr int kBlock = 8;
    for (int dir = -1; dir <= 1; dir += 2) {
        bool faded = false;
        for (int k0 = 1; k0 < kNctNodes && !faded; k0 += kBlock) {
            double vs[kBlock];
            for (int i = 0; i < kBlock; ++i) {
                const double u = dir * (k0 + i) * kStep / kA;
                // sinh u and cosh u from one exponential: expm1 near 0 (no cancellation in sinh),
                // exp beyond (e^u itself: from e^u - 1 it would be lost below e^-37)
                double eu, e1;
                if (std::fabs(u) > 0.5) e1 = std::exp(u), eu = e1 - 1.0;
                else eu = std::expm1(u), e1 = eu + 1.0;
                const double sh = 0.5 * (eu + eu / e1), ch = 0.5 * (e1 + 1.0 / e1);
                vs[i] = integrand(c * kA * sh, ch) * ch;
            }
            for (int i = 0; i < kBlock; ++i) {
                I.add(vs[i]);
                if (!(vs[i] > 0x1p-60 * I.total()) && k0 + i > 8) {  // (a NaN node ends it: the sum is NaN)
                    faded = true;
                    break;
                }
            }
        }
        if (!faded) return kNotSummed;
    }
    const double integral = I.total() * kStep * c;
    if (want) *dens = exp_wide(Estar.hi) * scaled((1.0 + Estar.lo) * (dsum * kStep * c * prefix_detail::kInvSqrt2Pi));
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
            add_s(
                tiny_upper_s(mu.hi, 0.0, 0.5, 0.5 * df, logy, mu.lo),
                tiny_upper_s(mu.hi, 0.5, 1.0, 0.5 * df, logy, mu.lo)
            ),
            0.5
        ));
        return upper ? up : one_minus(up);
    }
    if (g.x.hi < distr_detail::kTinyArg)
        return upper ? nct_opposite_s(df, -nc, t) : one_minus(nct_opposite_s(df, -nc, t));
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
    if (t == 0.0) return false;
    const dd::DD mu = dd::two_prod(nc, nc) * 0.5;
    const double b = 0.5 * df;
    if (!(mu.hi <= lite::kLiteArg && mu.hi >= 0x1p-500 && b <= 1e6 && b >= kMinShape)) return false;
    const distr_detail::Arg g = distr_detail::t_arg(df, t);  // g.y = t^2 / (df + t^2): the families' argument
    if (!(g.x.hi >= 0x1p-500 && g.y.hi >= 0x1p-500)) return false;
    const double nl = lite_terms_beta(0.5, b, g.y.hi, g.x.hi), nu = lite_terms_weights(mu.hi);
    if (!(nl <= lite::kLiteTerms || nu <= lite::kLiteTerms)) return false;
    const double tp = beta_first(0.5, b, g.y, g.x, mu);  // T(1/2) D(0, mu)
    const double pw = std::pow(g.x.hi, b), em = std::exp(-mu.hi);
    if (!(pw >= kLitePow && em >= kLitePow)) return false;
    const double corr = 1.0 + ((g.y.lo / g.y.hi + b * (g.x.lo / g.x.hi)) - mu.lo);
    const double ybw = pw * corr * em;  // y^b e^-mu
    const double tq = std::fabs(nc) * 0.79788456080286535588 * b * g.y.hi * ybw;  // T(1) D(1/2, mu)
    if (!(tp >= 0x1p-900 && tq >= 0x1p-900)) return false;
    const lite::BetaRatio rp(0.5, b, g.y), rq(1.0, b, g.y);
    const double sign = nc < 0.0 ? -1.0 : 1.0;
    double dp = 0.0, dq = 0.0;
    auto lower = [&](double& v) {
        double sp, sq;
        // the weights' totals over their first: sum_j D(j, mu) = 1, sum_j D(j + 1/2, mu) = erf(sqrt(mu)) <= 1
        const double wq = std::fabs(nc) * 0.79788456080286535588 * em;
        if (!lite::lower_sum(rp, mu, 0.0, (1.0 + 0x1p-40) / em, sp, dp) ||
            !lite::lower_sum(rq, mu, 0.5, 1.0 / wq, sq, dq))
            return false;
        v = at_most_one(value(ndtr_neg_s(nc)) + 0.5 * (tp * sp + sign * (tq * sq)));
        return v >= 0x1p-1000;
    };
    // the two families' upper sums: cp = sum_j D(j, mu) (1 - I_x(j + 1/2, b)), cq = sum_j D(j + 1/2, mu)
    // (1 - I_x(j + 1, b)); the upper tail is (cp + cq) / 2 for nc > 0, (cp - cq) / 2 for nc < 0
    auto upper_parts = [&](double& cp, double& cq) {
        double sp, sq;
        // 1 - I_x(1/2, b) = P(|T0| > t) of the central t (stdtr's kernel); 1 - I_x(1, b) = y^b
        const double qp = 2.0 * distr_detail::t_tail(df, t), qq = pw * (1.0 + b * (g.x.lo / g.x.hi));
        if (!(qp >= kTinyG)) return false;
        const double wq = std::fabs(nc) * 0.79788456080286535588 * em;
        if (!lite::upper_sum(rp, mu, 0.0, em / tp, sp, dp) || !lite::upper_sum(rq, mu, 0.5, wq / tq, sq, dq))
            return false;
        const double eq = std::erf(std::fabs(nc) * 0.70710678118654752440);  // sum_j D(j + 1/2, mu)
        cp = qp + tp * sp, cq = qq * eq + tq * sq;
        return true;
    };
    auto upper_of = [&](double& v) {  // nc > 0
        double cp, cq;
        if (nc < 0.0 || !upper_parts(cp, cq)) return false;
        v = at_most_one(0.5 * (cp + cq));
        return v >= 0x1p-1000;
    };
    if (upper && nc < 0.0) {
        // The tail away from the noncentrality, (cp - cq) / 2: where cq is at most cp / 4 the
        // difference loses under a bit (its relative error (cp + cq) / (cp - cq) <= 5/3 times the
        // parts'), and this is far cheaper than nct_opposite's integral (whose tail at a small
        // df needs a thousand nodes and more). cq / cp grows with |nc| (the second family's
        // weights are |nc| sqrt(2/pi) e^-mu (1 + ...)) and with df and t: the integral there.
        // (cp <= 1 and cq >= y^b erf(|nc| / sqrt 2), its first part: past 1/4 no sum is run)
        const double cq_low = pw * std::erf(std::fabs(nc) * 0.70710678118654752440);
        double cp, cq;
        if (!(nu <= lite::kLiteTerms) || !(cq_low <= 0.25) || !upper_parts(cp, cq) || !(cq <= 0.25 * cp)) return false;
        out = 0.5 * (cp - cq);
        if (!(out >= 0x1p-1000)) return false;
        if (pdf) *pdf = (tp * (0.5 + g.y.hi * dp) + sign * (tq * (1.0 + g.y.hi * dq))) / t;
        return true;
    }
    if (!lite_pick(upper, mu.hi, nl, nu, 14.0, nct_z(df, nc, t), lower, upper_of, out, 150.0)) return false;
    // d/dt of the lower series: I_x(s, b)' = T(s - 1) (s - 1 + b) / y dx/dt, dx/dt = 2 x y / t
    if (pdf) *pdf = (tp * (0.5 + g.y.hi * dp) + sign * (tq * (1.0 + g.y.hi * dq))) / t;
    return true;
}
inline double nct_one(double df, double nc, double t, bool upper) {
    double out;
    if (nct_lite(df, nc, t, upper, out)) return out;
    return value(nct_s(df, nc, t, upper));
}
// A tiny df: T = (Z + nc) / S with S = sqrt(V / df) above any s only with probability
// P(V > df s^2) ~ (df / 2) log(2 / (df s^2)), below 1083 df for every s > 1 / 1.8e308 a t can make
// count (|t| s ~ 1): T <= t is Z <= -nc to that, Phi(-nc) whatever t, and the upper tail Phi(nc),
// taken where 1083 df is below 2^-60 of the smaller tail (or both are below 2^-1000). (The opposite tail's integral cannot fade at such a df: nctdtrc(2^-1022, -1,
// 1e-300) was 0.137 for Phi(-1) = 0.159, against nctdtr's 0.841.)
inline double nct(double df, double nc, double t, bool upper) {
    if (df < 0x1p-71) {
        // (the bound 1083 df on the difference, against 2^-60 of the smaller of Phi(-nc), Phi(nc), or
        // both below 2^-1000)
        const double small = value(ndtr_neg_s(std::fabs(nc)));
        if (1083.0 * df <= 0x1p-60 * small || (small < 0x1p-1000 && df < 0x1p-999))
            return value(ndtr_neg_s(upper ? -nc : nc));
    }
    const double z = t >= 0.0 ? nct_z(df, nc, t) : -nct_z(df, -nc, -t);  // T(nc) = -T(-nc)
    // (z not trusted where the other side is the tail away from the noncentrality, P(T > t) for
    // t > 0 > nc or P(T < t) for t < 0 < nc: its integral costs more than this side's series;
    // the pairing after the fact still applies)
    const bool opposite_other = upper ? (t < 0.0 && nc > 0.0) : (t > 0.0 && nc < 0.0);
    // The tail away from the noncentrality itself (P(T > t), t > 0 > nc, or P(T < t), t < 0 < nc):
    // 1 minus the other side (a series, positive terms) while that is at most 7/8 (7 ULP at most
    // from the subtraction), else its own form. The integral's step was set for df ~ 1 and up; at
    // a small df its tail fades so slowly that it lost 1700 ULP (nctdtr(0.001, 1, -10)), and the two
    // sides of a pair disagreed by 3.8e-14.
    const bool opposite = upper ? (t > 0.0 && nc < 0.0) : (t < 0.0 && nc > 0.0);
    if (opposite) {
        const double o = nct_one(df, nc, t, !upper);
        if (o <= 0.875) return 1.0 - o;
        return nct_one(df, nc, t, upper);
    }
    return small_side(opposite_other ? 0.0 : z, upper, [&](bool u) { return nct_one(df, nc, t, u); });
}

// The tail (upper: the upper one) and the density at a point, for the inverses' Newton steps
// (newton_root): the sums of the bulk where they apply, else the walk with its density, in the
// forward function's own order (so the root is the forward function's). False where neither
// gives both: a tail the forward function takes elsewhere (its edges, the tiny arguments, the
// noncentral t's opposite tail), a sum that did not end, a tail below 2^-1000. The density is
// handed over as the tail's length tail / density, from the m 2^e values (the density
// underflows before the tail in a heavy one: the noncentral t's, which falls as |t|^-df).
inline bool tail_len(double tail, Scaled v, Scaled d, double factor, double& len) {
    len = value(v / d) * factor;
    return tail >= 0x1p-1000 && len > 0.0 && std::isfinite(len);
}
inline bool tail_len(double tail, double pdf, double& len) {
    len = tail / pdf;
    return pdf > 0.0 && len > 0.0 && std::isfinite(len);
}
inline bool ncx2_tail_len(double df, double nc, double x, bool upper, double& tail, double& len) {
    double pdf;
    if (ncx2_lite(df, nc, x, upper, tail, &pdf)) return tail_len(tail, pdf, len);
    Scaled d;
    const Scaled v = mixture_s(0.5 * nc, 0.0, 0.5 * df, GammaFamily{0.5 * x, upper}, 0.0, false, &d);
    tail = at_most_one(value(v));
    return tail_len(tail, v, d, 2.0, len);  // d/dx P(s, x/2) = T(s - 1) / 2
}
inline bool ncf_tail_len(double dfn, double dfd, double nc, double x, bool upper, double& tail, double& len) {
    double pdf;
    if (ncf_lite(dfn, dfd, nc, x, upper, tail, &pdf)) return tail_len(tail, pdf, len);
    const distr_detail::Arg g = distr_detail::f_arg(dfn, dfd, x);
    if (!(g.x.hi >= distr_detail::kTinyArg && g.y.hi >= distr_detail::kTinyArg)) return false;
    Scaled d;
    const Scaled v = mixture_s(0.5 * nc, 0.0, 0.5 * dfn, BetaFamily{0.5 * dfd, g.x, g.y, upper}, 0.0, false, &d);
    tail = at_most_one(value(v));
    return tail_len(tail, v, d, x, len);  // d/dx I_w(s, b) = T(s) s / (w v) dw/dx, dw/dx = w v / x
}
inline bool nct_tail_len(double df, double nc, double t, bool upper, double& tail, double& len) {
    double pdf;
    if (nct_lite(df, nc, t, upper, tail, &pdf)) return tail_len(tail, pdf, len);
    if (t < 0.0) nc = -nc, t = -t, upper = !upper;  // T(nc) = -T(-nc): the same density
    if (!(t > 0.0)) return false;
    if (upper && nc < 0.0) {  // the tail away from the noncentrality: its integral
        Scaled d;
        const Scaled v = nct_opposite_s(df, -nc, t, &d);
        tail = value(v);
        return tail_len(tail, v, d, 1.0, len);
    }
    const distr_detail::Arg g = distr_detail::t_arg(df, t);
    if (!(g.x.hi >= distr_detail::kTinyArg && g.y.hi >= distr_detail::kTinyArg)) return false;
    const dd::DD mu = dd::two_prod(nc, nc) * 0.5;
    if (!(mu.hi > 0.0)) return false;
    const BetaFamily fam{0.5 * df, g.y, g.x, upper};
    Scaled dp, dq;
    const Scaled sp = mixture_s(mu.hi, 0.0, 0.5, fam, mu.lo, false, &dp);
    const Scaled sq = mixture_s(mu.hi, 0.5, 1.0, fam, mu.lo, false, &dq);
    const double sign = std::copysign(1.0, nc);  // (+1 on the upper side: nc > 0 there)
    const Scaled v = upper ? clamp01(times(add_s(sp, sq), 0.5))
                           : clamp01(add_s(ndtr_neg_s(nc), times(add_s(sp, times(sq, sign)), 0.5)));
    tail = value(v);
    // (1/2) (D_p + sign D_q) / (X Y) dX/dt, X = t^2 / (df + t^2), dX/dt = 2 X Y / t
    return tail_len(tail, v, add_s(dp, times(dq, sign)), t, len);
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
inline double solve(
    const G& g, double x0, const To& to_u, const From& from_u, double step, double lowest, double highest
) {
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
    const double bottom =
        lowest < 0.0 ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::denorm_min();
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
    const double r =
        solve(g, x0, [](double x) { return std::log(x); }, [](double u) { return std::exp(u); }, 0.125, 0.0, kInf);
    if (r == r) return r;
    // no bracket inside the doubles: a root beyond the largest one is inf (the tail there is
    // still short of the target: ncfdtri(5e-324, 5, 2, 1/2), 2026-11-19), below the smallest 0
    constexpr double kMax = std::numeric_limits<double>::max(), kMin = std::numeric_limits<double>::denorm_min();
    if (g(kMax) < 0.0) return kInf;
    if (g(kMin) > 0.0) return 0.0;
    return r;
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
    return solve(
        g, t0, [](double t) { return std::asinh(t); }, [](double u) { return std::sinh(u); }, 0.125, -kInf, kInf
    );
}

// The root of F(x) = p by Newton's method on log(tail / target), the smaller tail against its
// own target as invert_positive (in log x for a positive variable, else in asinh x: a tail that
// falls as a power of |x| is a line there, the noncentral t's for a small df): eval(x, upper,
// tail, len) gives the tail and its length tail / f, f the density (a derivative: 1e-8 is enough), by the
// sums of the bulk or the walk (the forward functions' own, so the root is theirs). The
// root's error is the tail's, times tail / (x f): invert_positive's, without its bracketing
// and zeroin (5 to 12 evaluations, against 3 or 4 here). A step d ends it when the next
// error, K d^2 + 2^-30 d (K = d / d_prev^2 measured from the last two steps, both in the
// quadratic regime: d_prev below 2^-5 of the scale max(|x|, tail / f) and d below d_prev / 8;
// 2^-30 for the density's own error), is below 2^-53 of that scale, or d itself is below
// 2^-50 of it. False (the bracketing solver then): a sum does not apply at an iterate, or
// no end in 24 steps.
template <class Eval>
inline bool newton_root(const Eval& eval, double p, double x0, bool positive, double& out) {
    const bool upper = p > 0.5;
    const double target = upper ? 1.0 - p : p;
    if (!(target >= 0x1p-900 && std::isfinite(x0)) || (positive && !(x0 > 0.0))) return false;
    // x moved by du on its scale: x e^du, or sinh(asinh x + du), for a short step as x cosh du +
    // sqrt(1 + x^2) sinh du (no rounding of asinh x; a long one would cancel there)
    auto move = [&](double x, double du) {
        if (positive) return x + x * std::expm1(du);
        if (std::fabs(du) > 0.5) return std::sinh(std::asinh(x) + du);
        return x + (x * (std::cosh(du) - 1.0) + std::hypot(1.0, x) * std::sinh(du));
    };
    // Every iterate narrows a bracket [ulo, uhi] on the scale, and a step past its end is
    // replaced by a bisection there. A step is at most `cap` on the scale, doubled while the
    // steps keep being cut in one direction (a start far from a root in a heavy tail: a t with
    // P(T <= t) = 1e-28 at df = 0.14 is -3e193, 445 on the asinh scale from a start at -1).
    double x = x0, prev = 0.0, ulo = -kInf, uhi = kInf, cap = 2.0, last_cut = 0.0;
    for (int it = 0; it < 24; ++it) {
        double tail, len;
        if (!eval(x, upper, tail, len) || !(len > 0.0 && tail > 0.0 && std::isfinite(len))) return false;
        double g = std::log(tail / target);  // made increasing in x
        if (upper) g = -g;
        const double u = positive ? std::log(x) : std::asinh(x), jac = positive ? x : std::hypot(1.0, x);
        if (g > 0.0) uhi = u;
        else ulo = u;
        double du = -g * len / jac;  // Newton's step on the scale
        const double scale = std::max(std::fabs(x), len);
        double xn = move(x, du), ad = std::fabs(xn - x);
        if (std::isfinite(xn) && (!positive || xn > 0.0) && std::fabs(du) <= 0x1p-10 &&
            (ad <= 0x1p-50 * scale || (prev > 0.0 && prev <= 0x1p-5 * scale && ad <= 0x1p-3 * prev &&
                                       ad * (ad * ad / (prev * prev) + 0x1p-30) <= 0x1p-53 * scale))) {
            out = xn;
            return true;
        }
        bool newton = true;
        if (std::fabs(du) > cap) {
            du = std::copysign(cap, du);
            newton = false;
            if (last_cut == du) cap *= 2.0;
            last_cut = du;
        } else {
            last_cut = 0.0;
        }
        // (a step below the resolution of u lands on the bracket's end just set: no bisection)
        if (!((u + du > ulo && u + du < uhi) || std::fabs(du) <= 0x1p-40 * (1.0 + std::fabs(u)))) {
            if (!(std::isfinite(ulo) && std::isfinite(uhi))) return false;
            du = 0.5 * (ulo + uhi) - u;
            newton = false;
        }
        xn = move(x, du);
        if (!std::isfinite(xn) || (positive && !(xn > 0.0)) || xn == x) return false;
        prev = newton ? std::fabs(xn - x) : 0.0;  // (the quadratic estimate only from two Newton steps)
        x = xn;
    }
    return false;
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
    } else if (df > 0.0 && df < 0x1p-960) {
        // a = df/2 below 2^-961 (gammainc's tiny shapes: the upper tail was NaN for df up to
        // ~1e-305 at small x, chndtrc(1e-307, 1, 1e-300); 2026-11-19): from the df = 0 law,
        // whose shapes j >= 1 are off from a + j by a relative a |log y| < 2^-950, and its
        // j = 0 term, a point mass at 0 there: here e^-mu Q(a, y) with Q(a, y) = a E1(y) to
        // the same order (Gamma(a) = 1/a, Gamma(a, y) = E1(y) + O(a)). The upper tail adds it
        // to df = 0's (it counts when mu is tiny too: chndtrc(2^-1022, 1e-300, 1e-300)); the
        // lower one, 1 to rounding beside it, is df = 0's.
        double o;
        if (!ncx2_edge(0.0, nc, x, upper, o)) o = nc_detail::ncx2(0.0, nc, x, upper);
        if (upper) {
            const double y = 0.5 * x;
            // (a = df/2 formed after the scaling by 2^1000, so that a subnormal df's half keeps its
            // bits: 0.5 * df first rounded it, chndtrc(5e-324, 5e-324, 1e-300) was 0; 2026-11-19)
            const double j0 = std::exp(-0.5 * nc) * ((df * 0x1p1000 * 0.5) * expint_detail::en(1.0, y)) * 0x1p-1000;
            o = o + j0;
        }
        out = o;
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

// A tail of the probability below e^-1000 by Chernoff's bound from the moment generating
// function (1 - 2s)^(-df/2) e^(nc s / (1 - 2s)): P(X <= x) <= e^(x/2) 2^(-df/2) e^(-nc/4)
// (s = -1/2), P(X >= x) <= e^(-x/4) 2^(df/2) e^(nc/2) (s = 1/4). 0 and 1 without the walk,
// which could not reach the terms' peak there (chndtr(1, 1e16, 1) was NaN for 0). (Not for
// sd.stats' log forms: they want the tail's value, chndtr_tail.)
inline bool ncx2_obvious(double df, double nc, double x, bool upper, double& out) {
    if (0.5 * x - kLn2Hi * 0.5 * df - 0.25 * nc < -1000.0) {
        out = upper ? 1.0 : 0.0;
    } else if (-0.25 * x + kLn2Hi * 0.5 * df + 0.5 * nc < -1000.0) {
        out = upper ? 0.0 : 1.0;
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
        if (ncx2_edge(df, nc, x, false, out) || ncx2_obvious(df, nc, x, false, out)) return out;
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
        if (ncx2_edge(df, nc, x, true, out) || ncx2_obvious(df, nc, x, true, out)) return out;
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
        const double x0 = nc_detail::ncx2_start(df, nc, p);
        double r;
        // (x below 2^-1021 or a tiny df: the forward function's own edges, not the sums)
        auto eval = [&](double x, bool u, double& t, double& f) {
            double o;
            return x >= 0x1p-1000 && df >= 0x1p-960 && !ncx2_obvious(df, nc, x, u, o) &&
                   nc_detail::ncx2_tail_len(df, nc, x, u, t, f);
        };
        if (nc_detail::newton_root(eval, p, x0, true, r)) return r;
        return nc_detail::invert_positive(
            [&](double x) { return Chndtr::f(df, nc, x); }, [&](double x) { return Chndtrc::f(df, nc, x); }, p, x0
        );
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
        // A small df: far out the tails fall as |t|^-df, past every normal-type start (P(T <= t) =
        // 1e-34 at df = 0.22 is t = -6e146): the central quantile's, where it is farther out and the
        // noncentrality leans the same way (the tail at least as heavy as the central one's).
        if (df <= 2.0 && std::min(p, 1.0 - p) < 1e-8 && (p < 0.5 ? nc <= 0.0 : nc >= 0.0)) {
            const double tc = Stdtrit::f(df, p);
            t0 = p < 0.5 ? std::min(t0, tc) : std::max(t0, tc);
        }
        double r;
        auto eval = [&](double t, bool u, double& v, double& f) { return nc_detail::nct_tail_len(df, nc, t, u, v, f); };
        if (nc_detail::newton_root(eval, p, t0, false, r)) return r;
        return nc_detail::invert_real(
            [&](double t) { return Nctdtr::f(df, nc, t); }, [&](double t) { return Nctdtrc::f(df, nc, t); }, p, t0
        );
    }
};

// ------------------------------------------------------------------ F

namespace nc_detail {
// The smallest subnormal dfn has no half (0.5 * 5e-324 rounds to 0, and the mixture's first
// beta shape a = 0 gave NaN for f ~ 1e23-1e173, where w is not tiny; 2026-11-19): the next
// one, 1e-323. The values move by the j = 0 term's 1 - I_w(a, b) ~ a |log w| alone, below
// 2^-1063 absolute.
// Not where dfd itself is tiny (ncf's tiny-dfd form, the central F's cdf, takes dfn as it is:
// moved, ncfdtr(5e-324, 5e-324, nc, f) was e^-mu / 3 for e^-mu / 2, and apart from fdtr at nc = 0).
inline double halvable_dfn(double dfn, double dfd) {
    return dfn == 0x1p-1074 && !(0.5 * dfd < kTinyShape) ? 0x1p-1073 : dfn;
}
}  // namespace nc_detail

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
    static constexpr std::array<const char*, 4> args{"dfn", "dfd", "nc", "x"};
    static constexpr double cost_ns = cost::kNcfdtr;
    static double f(double dfn, double dfd, double nc, double x) {
        dfn = nc_detail::halvable_dfn(dfn, dfd);
        double out;
        if (ncf_edge(dfn, dfd, nc, x, false, out)) return out;
        return nc_detail::ncf(dfn, dfd, nc, x, false);
    }
};

struct Ncfdtrc : Real {
    static constexpr const char* name = "ncfdtrc";
    static constexpr int nin = 4, nout = 1;
    static constexpr std::array<const char*, 4> args{"dfn", "dfd", "nc", "x"};
    static constexpr double cost_ns = cost::kNcfdtrc;
    static double f(double dfn, double dfd, double nc, double x) {
        dfn = nc_detail::halvable_dfn(dfn, dfd);
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
        dfn = nc_detail::halvable_dfn(dfn, dfd);
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
        double r;
        auto eval = [&](double x, bool u, double& v, double& f) {
            return nc_detail::ncf_tail_len(dfn, dfd, nc, x, u, v, f);
        };
        if (nc_detail::newton_root(eval, p, x0, true, r)) return r;
        return nc_detail::invert_positive(
            [&](double x) { return Ncfdtr::f(dfn, dfd, nc, x); }, [&](double x) { return Ncfdtrc::f(dfn, dfd, nc, x); },
            p, x0
        );
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
    dfn = nc_detail::halvable_dfn(dfn, dfd);
    double out;
    if (ncf_edge(dfn, dfd, nc, x, upper, out)) return scaled(out);
    return clamp01(ncf_s(dfn, dfd, nc, x, upper, wide));
}

}  // namespace nc_detail

}  // namespace sd::special
