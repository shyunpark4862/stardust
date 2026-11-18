// The distributions' own sampling methods (T7, docs/design/stats.md §4), where they beat
// inversion (sample.hpp; measured in bench/stats/t7/). Every method is exact in
// distribution (rejection or composition, no approximation); the acceptance tests of the
// discrete rejections use the double-double log pmfs of the distributions, so they stay
// exact where lgamma differences of size 1e16 would not (huge mu or n); a plain-double
// value decides first where its error margin leaves no doubt (quick_accept).
//
// Every loop here ends in a bounded or geometrically distributed number of rounds, whatever
// the parameters (docs/design/stats.md §1.3): each round of a rejection draws fresh words and
// is accepted with at least the stated probability, by a test that involves no parameter-
// dependent quantity that could be NaN or inf (so a NaN in a later test can only reject):
//     std_normal_rest    wedges >= 0.5 a round; the tail's 2 E2 > (E1 / R)^2 >= 0.93
//     std_exp_rest       wedges >= 0.5 a round (the tail is direct)
//     gamma_mt(_loop)    the squeeze u < 1 - 0.0331 x^4 alone: >= 0.92 (d >= 2/3, any size)
//     poisson PTRS       the squeeze us >= 0.07, V <= vr alone: >= 0.34 (mu >= 10)
//     binomial BTRS      the squeeze alone: >= 0.26 (n p >= 10, p <= 1/2)
//     beta, Joehnk       Gamma(a+1) Gamma(b+1) / Gamma(a+b+1) >= 1/2 (a, b < 1)
//     log series, Kemp   a round repeats only when V = 0 or q rounds to 1: < 2^-50
//     Zipf, Devroye      (1 - 2^(1-a)) zeta(a) >= log 2 = 0.69
//     Lemire's integers  >= 1/2
// The inversions step a pmf recurrence until it underflows (Poisson mu < 10 and binomial
// n p < 10: at most ~230 steps, 1024 by its bound) or within 2^20 of the mode (hypergeometric).
//
// A specialization Custom<D> has either
//     static double draw(Words& w, params...)
// or a per-parameter-set state, computed once for a call whose parameters are scalars:
//     static State prep(params...);  static double draw(Words& w, const State& s)
// and optionally kWork, the pool's work estimate per element, and kExtra or extra(state)
// (the method draws extra words: the kernel computes four elements' first extra blocks
// together, 4.5 ns an element where one block alone takes 12).
#pragma once

#include "stats/dist/beta.hpp"
#include "stats/dist/cauchy.hpp"
#include "stats/dist/discrete_basic.hpp"
#include "stats/dist/discrete_more.hpp"
#include "stats/dist/exponential.hpp"
#include "stats/dist/extreme.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/laplace.hpp"
#include "stats/dist/lifetime.hpp"
#include "stats/dist/lognormal.hpp"
#include "stats/dist/noncentral.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/dist/uniform.hpp"
#include "stats/dist/weibull.hpp"
#include "stats/sample.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sd::stats::smp {

// ---- the standard normal: Marsaglia and Tsang's ziggurat (2000), 256 layers, as numpy's
// random_standard_normal: the low 8 bits of a word pick the layer, the next one the sign,
// 52 bits the abscissa; the base layer's tail by Marsaglia's exponential method.

struct ZigNormal {
    static constexpr int kN = 256;
    static constexpr double kR = 3.6541528853610088;      // the base layer's right end
    static constexpr double kV = 4.92867323399e-3;        // the common layer area
    std::uint64_t ki[kN];
    double wi[kN], fi[kN];
    ZigNormal() {
        const double m1 = 0x1p52;
        double dn = kR, tn = dn;
        const double q = kV / std::exp(-0.5 * dn * dn);
        ki[0] = static_cast<std::uint64_t>((dn / q) * m1);
        ki[1] = 0;
        wi[0] = q / m1;
        wi[kN - 1] = dn / m1;
        fi[0] = 1.0;
        fi[kN - 1] = std::exp(-0.5 * dn * dn);
        for (int i = kN - 2; i >= 1; --i) {
            dn = std::sqrt(-2.0 * std::log(kV / dn + std::exp(-0.5 * dn * dn)));
            ki[i + 1] = static_cast<std::uint64_t>((dn / tn) * m1);
            tn = dn;
            fi[i] = std::exp(-0.5 * dn * dn);
            wi[i] = dn / m1;
        }
    }
};
inline const ZigNormal kZigNormal;  // built at load: no guard on every draw
inline const ZigNormal& zig_normal() { return kZigNormal; }

// Outside the layer's rectangle (~1%), out of line: the common path stays small enough to
// inline, with the element's words in registers (5.6 -> 2 ns a normal).
[[gnu::noinline]] inline double std_normal_rest(Words& w, int idx, bool neg, double x) {
    const ZigNormal& z = zig_normal();
    for (;;) {
        if (idx == 0) {  // the tail beyond R
            for (;;) {
                const double xx = -std::log(w.u()) / ZigNormal::kR;
                const double yy = -std::log(w.u());
                if (yy + yy > xx * xx) return neg ? -(ZigNormal::kR + xx) : ZigNormal::kR + xx;
            }
        }
        if ((z.fi[idx - 1] - z.fi[idx]) * w.u01() + z.fi[idx] < std::exp(-0.5 * x * x)) return x;
        std::uint64_t r = w.next();
        idx = static_cast<int>(r & 0xff);
        r >>= 8;
        neg = r & 1;
        const std::uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
        x = std::bit_cast<double>(std::bit_cast<std::uint64_t>(static_cast<double>(rabs) * z.wi[idx]) |
                                  (static_cast<std::uint64_t>(neg) << 63));
        if (rabs < z.ki[idx]) return x;
    }
}

inline double std_normal(Words& w) {
    const ZigNormal& z = zig_normal();
    std::uint64_t r = w.next();
    const int idx = static_cast<int>(r & 0xff);
    r >>= 8;
    const bool neg = r & 1;
    const std::uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
    // the sign by its bit (a branch on a random bit mispredicts half the time: 3 ns)
    const double x = std::bit_cast<double>(std::bit_cast<std::uint64_t>(static_cast<double>(rabs) * z.wi[idx]) |
                                           (static_cast<std::uint64_t>(neg) << 63));
    if (rabs < z.ki[idx]) [[likely]] return x;  // inside the layer's rectangle (~99%)
    return std_normal_rest(w, idx, neg, x);
}

// ---- the standard exponential: the ziggurat of Marsaglia and Tsang (2000), 256 layers, as
// numpy's random_standard_exponential (53 bits of the word's top 61 for the abscissa, the
// low 8 of those for the layer); the base layer's tail beyond R is R + its own exponential.

struct ZigExp {
    static constexpr int kN = 256;
    static constexpr double kR = 7.69711747013104972;
    static constexpr double kV = 3.949659822581572e-3;
    std::uint64_t ke[kN];
    double we[kN], fe[kN];
    ZigExp() {
        const double m2 = 0x1p53;
        double de = kR, te = de;
        const double q = kV / std::exp(-de);
        ke[0] = static_cast<std::uint64_t>((de / q) * m2);
        ke[1] = 0;
        we[0] = q / m2;
        we[kN - 1] = de / m2;
        fe[0] = 1.0;
        fe[kN - 1] = std::exp(-de);
        for (int i = kN - 2; i >= 1; --i) {
            de = -std::log(kV / de + std::exp(-de));
            ke[i + 1] = static_cast<std::uint64_t>((de / te) * m2);
            te = de;
            fe[i] = std::exp(-de);
            we[i] = de / m2;
        }
    }
};
inline const ZigExp kZigExp;

[[gnu::noinline]] inline double std_exp_rest(Words& w, int idx, double x) {  // as std_normal_rest
    const ZigExp& z = kZigExp;
    for (;;) {
        if (idx == 0) return ZigExp::kR - std::log(w.u());
        if ((z.fe[idx - 1] - z.fe[idx]) * w.u01() + z.fe[idx] < std::exp(-x)) return x;
        std::uint64_t r = w.next() >> 3;
        idx = static_cast<int>(r & 0xff);
        r >>= 8;
        x = static_cast<double>(r) * z.we[idx];
        if (r < z.ke[idx]) return x;
    }
}

inline double std_exp(Words& w) {
    const ZigExp& z = kZigExp;
    std::uint64_t r = w.next() >> 3;
    const int idx = static_cast<int>(r & 0xff);
    r >>= 8;
    const double x = static_cast<double>(r) * z.we[idx];
    if (r < z.ke[idx]) [[likely]] return x;  // ~99%
    return std_exp_rest(w, idx, x);
}

// ---- gamma: Marsaglia and Tsang (2000) for a >= 1; for a < 1, G(a) = G(a + 1) U^(1/a)
// (Stuart), U^(1/a) = exp(-E / a) from a standard exponential, and in logs for tiny shapes so
// the sample need not underflow before its use in a ratio.

struct GammaPrep {
    double a, d, c;  // d = a' - 1/3, c = 1 / sqrt(9 d), a' = max(a, a + 1 for a < 1)
    bool small;      // a < 1
};
inline GammaPrep gamma_prep(double a) {
    const bool small = a < 1.0;
    const double a1 = small ? a + 1.0 : a;
    const double d = a1 - 1.0 / 3.0;
    return {a, d, 1.0 / std::sqrt(9.0 * d), small};
}
[[gnu::noinline]] inline double gamma_mt_loop(Words& w, double d, double c) {
    for (;;) {
        double x, v;
        do {
            x = std_normal(w);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = w.u01();
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
        if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
    }
}
// the first round inline (accepted ~96%: no call, the words in registers), then the loop
inline double gamma_mt(Words& w, double d, double c) {
    const double x = std_normal(w), v1 = 1.0 + c * x;
    if (v1 > 0.0) [[likely]] {
        const double v = v1 * v1 * v1, u = w.u01(), x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) [[likely]] return d * v;
        if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
    }
    return gamma_mt_loop(w, d, c);
}
// A shape below 1: G(a) = G(a + 1) U^(1/a), and U^(1/a) = exp(-E / a) with E a standard
// exponential (the ziggurat: no log). Shapes from kDirectShape up cannot underflow before
// the last digit of the tail (E / a > 700 has probability e^-43): the product directly.
inline constexpr double kDirectShape = 0.0625;
// log G(a)
inline double log_gamma_draw(Words& w, const GammaPrep& g) {
    const double l = std::log(gamma_mt(w, g.d, g.c));
    return g.small ? l - std_exp(w) / g.a : l;
}
[[gnu::noinline]] inline double gamma_draw_small(Words& w, const GammaPrep& g) {  // a < 1
    if (g.a >= kDirectShape) {
        const double x = gamma_mt(w, g.d, g.c);
        return x * std::exp(-std_exp(w) / g.a);
    }
    return std::exp(log_gamma_draw(w, g));
}
// log G(a1) - log G(a2) of two draws (the first's words first). Subnormal shapes put both
// logs at -inf (E / a overflows): the difference is then +-inf by the larger E / a, compared
// after scaling by the smaller shape (the finite parts cannot matter at that size).
inline double log_gamma_diff(Words& w, const GammaPrep& g1, const GammaPrep& g2) {
    const double m1 = std::log(gamma_mt(w, g1.d, g1.c)), e1 = g1.small ? std_exp(w) : 0.0;
    const double m2 = std::log(gamma_mt(w, g2.d, g2.c)), e2 = g2.small ? std_exp(w) : 0.0;
    const double l1 = g1.small ? m1 - e1 / g1.a : m1, l2 = g2.small ? m2 - e2 / g2.a : m2;
    if (!(std::isinf(l1) && std::isinf(l2))) [[likely]] return l1 - l2;
    const double m = std::min(g1.a, g2.a), s1 = e1 * (m / g1.a), s2 = e2 * (m / g2.a);
    const double inf = std::numeric_limits<double>::infinity();
    return s1 > s2 ? -inf : (s1 < s2 ? inf : 0.0);
}
// log(x / y) for positive finite x, y: of the quotient where it is a normal double (the
// usual case, one log), else the difference of the logs (x / y would overflow or lose bits)
inline double log_ratio(double x, double y) {
    const double r = x / y;
    return r >= 0x1p-1000 && r <= 0x1p1000 ? std::log(r) : std::log(x) - std::log(y);
}
inline double gamma_draw(Words& w, const GammaPrep& g) {
    if (!g.small) [[likely]] return gamma_mt(w, g.d, g.c);
    return gamma_draw_small(w, g);
}

// log k! for an integer k >= 0 in plain double, for the discrete rejections' first test: a
// table below 16, then Stirling's series to 1 / (1260 k^5) (the next term, 1 / (1680 k^7), is
// below 2.3e-12 from 16 on).
inline constexpr double kLogFactSmall[16] = {0.0, 0.0, 0.693147180559945, 1.791759469228055, 3.1780538303479444, 4.787491742782047, 6.579251212010101, 8.525161361065415, 10.60460290274525, 12.801827480081467, 15.104412573075514, 17.502307845873887, 19.987214495661885, 22.55216385312342, 25.191221182738683, 27.89927138384089};
inline double log_fact(double k) {
    if (k < 16.0) return kLogFactSmall[static_cast<int>(k)];
    const double r = 1.0 / k, r2 = r * r;
    return (k + 0.5) * std::log(k) - k + 0.9189385332046728 + r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0)));
}
// The acceptance test v <= l of a discrete rejection, l a log pmf (ratio) whose plain-double
// value ld is within 1e-11 + 2^-48 mag of it (mag the sum of the magnitudes of the terms
// that made ld: 32 roundings' worth, and the series' truncation): +1 accept, -1 reject, 0
// too close to tell (the caller then takes the double-double log pmf, as every draw did
// before: 150 ns, 14% of the rounds). The decision is the double-double one either way.
inline int quick_accept(double v, double ld, double mag) {
    const double margin = 1e-11 + 0x1p-48 * mag;
    if (v <= ld - margin) return 1;
    if (v > ld + margin) return -1;
    return 0;
}

// ---- Poisson: inversion from 0 for mu < 10 (one uniform, ~mu steps), PTRS (Hoermann,
// "The transformed rejection method for generating Poisson random variables", 1993) above.

struct PoissonPrep {
    double mu;
    bool ptrs;
    double e0;                                   // e^-mu (inversion)
    double b, a, invalpha, vr, log_ia, log_mu;   // PTRS
};
inline PoissonPrep poisson_prep(double mu) {
    PoissonPrep s{};
    s.mu = mu;
    s.ptrs = mu >= 10.0;
    if (!s.ptrs) {
        s.e0 = std::exp(-mu);
        return s;
    }
    const double slam = std::sqrt(mu);
    s.b = 0.931 + 2.53 * slam;
    s.a = -0.059 + 0.02483 * s.b;
    s.invalpha = 1.1239 + 1.1328 / (s.b - 3.4);
    s.vr = 0.9277 - 3.6224 / (s.b - 2.0);
    s.log_ia = std::log(s.invalpha);
    s.log_mu = std::log(mu);
    return s;
}
inline double poisson_draw(Words& w, const PoissonPrep& s) {
    if (s.mu == 0.0) return 0.0;
    if (!s.ptrs) {
        const double u = w.u01();
        double k = 0.0, p = s.e0, f = p;
        while (u >= f && p > 0.0) {
            k += 1.0;
            p *= s.mu / k;
            f += p;
        }
        return k;
    }
    for (;;) {
        const double U = w.u01() - 0.5, V = w.u01();
        const double us = 0.5 - std::fabs(U);
        const double k = std::floor((2.0 * s.a / us + s.b) * U + s.mu + 0.43);
        if (us >= 0.07 && V <= s.vr) return k;
        if (k < 0.0 || (us < 0.013 && V > us)) continue;
        // log pmf(k) = k log mu - mu - log k!, in plain double first
        const double lhs = std::log(V) + s.log_ia - std::log(s.a / (us * us) + s.b);
        const double klm = k * s.log_mu, lf = log_fact(k);
        const int quick = quick_accept(lhs, (klm - s.mu) - lf, std::fabs(klm) + s.mu + lf + 2.0 * k + 1.0);
        if (quick > 0) return k;
        if (quick < 0) continue;
        const num::DD l = Poisson::log_pmf_dd(k, s.mu);
        if (lhs <= l.hi + l.lo) return k;
    }
}

// ---- binomial: p (or q = 1 - p, the smaller of the two: n - X) by inversion from 0 when
// n p < 10, else BTRS (Hoermann 1993), acceptance on the exact log pmf ratio.

struct BinomPrep {
    double n, p;    // p <= 1/2 after the flip
    num::DD q;      // 1 - p
    bool flip;      // the sample is n - X
    bool btrs;
    double qn;      // q^n (inversion)
    double b, a, c, vr, alpha, m;  // BTRS
    double lfm, lpq;               // log m! + log (n - m)!, log(p / q): the plain first test
};
inline BinomPrep binom_prep(double n, double p, num::DD q) {
    BinomPrep s{};
    s.flip = p > 0.5;
    if (s.flip) {
        // the smaller side: q's value (q.hi + q.lo rounds to it) and p's complement exactly
        const num::DD pp = q;
        q = num::dd::two_sum(1.0, -pp.hi) - pp.lo;
        p = pp.hi + pp.lo;
    }
    s.n = n, s.p = p, s.q = q;
    s.btrs = n * p >= 10.0;
    if (!s.btrs) {
        s.qn = std::exp(n * std::log1p(-p));  // q^n, n p < 10
        return s;
    }
    const double spq = std::sqrt(n * p * q.hi);
    s.b = 1.15 + 2.53 * spq;
    s.a = -0.0873 + 0.0248 * s.b + 0.01 * p;
    s.c = n * p + 0.5;
    s.vr = 0.92 - 4.2 / s.b;
    s.alpha = (2.83 + 5.1 / s.b) * spq;
    s.m = std::floor((n + 1.0) * p);
    s.lfm = log_fact(s.m) + log_fact(n - s.m);
    s.lpq = std::log(p) - std::log1p(-p);
    return s;
}
// The inversion's p / q (binom_odds) is formed here (Given = false), or passed in by a caller
// that draws many times from one BinomPrep and computes it once (a quotient a draw
// otherwise, 4% at n = 20).
inline double binom_odds(const BinomPrep& s) { return s.p / s.q.hi; }
template <bool Given>
inline double binom_draw_with(Words& w, const BinomPrep& s, double odds) {
    if (s.n == 0.0 || s.p == 0.0) return s.flip ? s.n : 0.0;
    double x;
    if (!s.btrs) {
        // inversion: P(X = k + 1) / P(X = k) = (n - k) p / ((k + 1) q)
        const double r = Given ? odds : binom_odds(s);
        double u = w.u01(), px = s.qn, k = 0.0;
        // to min(n, 1024), not n: for n p < 10 the pmf is below the smallest double from
        // k ~ 230 on, and where rounding leaves u above the whole sum (~1e-16 of the draws) a
        // walk to n would take 2^53 steps, or never end once k + 1 == k.
        // The bound is a local, not a field of BinomPrep (the store costs a caller that
        // prepares per draw, the multinomial, 7%), and the loop is `while (u >= px &&
        // k < last)` with the bound tested first and the walk's own test on the way back: two
        // plain branches (as one condition the bound is compiled into the exit test, which
        // every draw mispredicts once: 3% of a draw at n = 20).
        const double last = s.n <= 1024.0 ? s.n : 1024.0;
        if (u >= px) {
            do {
                if (!(k < last)) break;
                u -= px;
                px *= (s.n - k) * r / (k + 1.0);
                k += 1.0;
            } while (u >= px);
        }
        x = k;
    } else {
        for (;;) {
            const double u = w.u01() - 0.5;
            double v = w.u01();
            const double us = 0.5 - std::fabs(u);
            const double k = std::floor((2.0 * s.a / us + s.b) * u + s.c);
            if (k < 0.0 || k > s.n) continue;
            if (us >= 0.07 && v <= s.vr) {
                x = k;
                break;
            }
            v = std::log(v * s.alpha / (s.a / (us * us) + s.b));
            // log pmf(k) - log pmf(m) = log m! (n-m)! - log k! (n-k)! + (k - m) log(p / q), in
            // plain double first
            const double lfk = log_fact(k) + log_fact(s.n - k), t = (k - s.m) * s.lpq;
            const int quick = quick_accept(v, (s.lfm - lfk) + t, s.lfm + lfk + std::fabs(t) + 4.0 * s.n + 1.0);
            if (quick > 0) {
                x = k;
                break;
            }
            if (quick < 0) continue;
            const num::DD d = binom_detail::log_pmf(k, s.n, s.p, s.q) - binom_detail::log_pmf(s.m, s.n, s.p, s.q);
            if (v <= d.hi + d.lo) {
                x = k;
                break;
            }
        }
    }
    return s.flip ? s.n - x : x;
}
inline double binom_draw(Words& w, const BinomPrep& s) { return binom_draw_with<false>(w, s, 0.0); }
inline double binom_draw(Words& w, const BinomPrep& s, double odds) { return binom_draw_with<true>(w, s, odds); }

// ---- inversion outward from the mode for a unimodal pmf (hypergeometric): the support's
// points in the fixed order of their pmf from the mode outwards (the larger neighbour
// first), their pmfs by the ratio recurrence, until the running sum passes u: exact
// inversion over that order, ~ (standard deviation) steps.
// The caller bounds the walk through lo and hi (Custom<Hypergeometric>: the support cut at
// kModeReach from the mode, 16 standard deviations at least): every step moves kl or kh
// toward one of them.
inline constexpr double kModeReach = 1048576.0;  // 2^20
// nl and nh are the pmfs of the mode's two neighbours (first_down, first_up below): they do not
// depend on u, so the caller computes them once for all its draws (two quotients a draw).
template <class Down>
double first_down(double mode, double lo, double pmode, Down&& down) { return mode > lo ? pmode * down(mode) : 0.0; }
template <class Up>
double first_up(double mode, double hi, double pmode, Up&& up) { return mode < hi ? pmode * up(mode) : 0.0; }
template <class Up, class Down>
double from_mode(double u, double mode, double lo, double hi, double pmode, double nl, double nh, Up&& up, Down&& down) {
    double s = pmode;
    if (u < s) return mode;
    double kl = mode, kh = mode, pl = pmode, ph = pmode;
    for (;;) {  // at most hi - lo steps
        if (nl <= 0.0 && nh <= 0.0) return mode;  // rounding left u past the sum: the mode
        if (nh >= nl) {
            kh += 1.0, ph = nh, s += ph;
            if (u < s) return kh;
            nh = kh < hi ? ph * up(kh) : 0.0;
        } else {
            kl -= 1.0, pl = nl, s += pl;
            if (u < s) return kl;
            nl = kl > lo ? pl * down(kl) : 0.0;
        }
    }
}

// ---- the specializations ---------------------------------------------------------------

template <>
struct Custom<Normal> {
    static double draw(Words& w, double mu, double sigma) { return std::fma(sigma, std_normal(w), mu); }
};
template <>
struct Custom<LogNormal> {
    static double draw(Words& w, double mu, double sigma) { return std::exp(std::fma(sigma, std_normal(w), mu)); }
};
template <>
struct Custom<Exponential> {
    static double draw(Words& w, double scale) { return scale * std_exp(w); }
};
template <>
struct Custom<Uniform> {
    // a + (b - a) U, U in [0, 1); b - a may overflow for a = -b near DBL_MAX: then a/2 + b/2 form
    static double draw(Words& w, double a, double b) {
        const double u = w.u01(), d = b - a;
        const double x = std::isfinite(d) ? std::fma(d, u, a) : 2.0 * std::fma(0.5 * b - 0.5 * a, u, 0.5 * a);
        return x < b ? x : a;  // rounding up to b: [a, b) as sd.random's random()
    }
};
template <>
struct Custom<Laplace> {
    static double draw(Words& w, double mu, double sigma) {
        const double u = w.u();  // (0, 1): +-E by the half (selects, not branches: the half is random)
        const bool lo = u < 0.5;
        const double l = std::log(2.0 * (lo ? u : 1.0 - u));
        return mu + sigma * (lo ? l : -l);
    }
};
template <>
struct Custom<Cauchy> {
    static double draw(Words& w, double mu, double sigma) {
        const double u = w.u();
        return mu + sigma * std::tan(special::kPi * (u - 0.5));
    }
};
template <>
struct Custom<Logistic> {
    static double draw(Words& w, double mu, double sigma) {
        const double u = w.u();
        return mu + sigma * std::log(u / (1.0 - u));  // 1 - u exact
    }
};
template <>
struct Custom<Gumbel> {
    static double draw(Words& w, double mu, double sigma) { return mu - sigma * std::log(std_exp(w)); }
};
template <>
struct Custom<GEV> {
    // T = (1 - c z)^(1/c) is a standard exponential: z = (1 - E^c) / c = -expm1(c log E) / c
    // (Gumbel's -log E at c = 0)
    static double draw(Words& w, double c, double mu, double sigma) {
        const double le = std::log(std_exp(w));
        if (c == 0.0) return mu - sigma * le;
        return mu - sigma * (std::expm1(c * le) / c);
    }
};
template <>
struct Custom<Weibull> {
    static double draw(Words& w, double c, double scale) { return scale * std::pow(std_exp(w), 1.0 / c); }
};
template <>
struct Custom<Pareto> {
    static double draw(Words& w, double b, double scale) { return scale * std::exp(std_exp(w) / b); }
};
template <>
struct Custom<Rayleigh> {
    static double draw(Words& w, double scale) { return scale * std::sqrt(2.0 * std_exp(w)); }
};

template <>
struct Custom<Gamma> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double scale;
    };
    static State prep(double a, double scale) { return {gamma_prep(a), scale}; }
    static double draw(Words& w, const State& s) {
        if (!s.g.small) [[likely]] return s.scale * gamma_mt(w, s.g.d, s.g.c);
        if (s.g.a >= kDirectShape) return s.scale * gamma_draw_small(w, s.g);
        return std::exp(log_gamma_draw(w, s.g) + std::log(s.scale));
    }
};
template <>
struct Custom<ChiSquared> {
    static constexpr bool kExtra = true;  // draws extra words
    static GammaPrep prep(double df) { return gamma_prep(0.5 * df); }
    static double draw(Words& w, const GammaPrep& g) { return 2.0 * gamma_draw(w, g); }
};
template <>
struct Custom<InverseGamma> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double log_scale;
    };
    static State prep(double a, double scale) { return {gamma_prep(a), std::log(scale)}; }
    static double draw(Words& w, const State& s) {
        if (!s.g.small) return std::exp(s.log_scale) / gamma_mt(w, s.g.d, s.g.c);
        return std::exp(s.log_scale - log_gamma_draw(w, s.g));
    }
};
// X / (X + Y) from log X, log Y (no underflow for tiny shapes); NaN where both logs are -inf
// (subnormal shapes: the ratio is then not known)
inline double beta_of_logs(double lx, double ly) {
    const double d = ly - lx;
    return d > 0.0 ? std::exp(-d) / (1.0 + std::exp(-d)) : 1.0 / (1.0 + std::exp(d));
}
template <>
struct Custom<Beta> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep ga, gb;
    };
    static State prep(double a, double b) { return {gamma_prep(a), gamma_prep(b)}; }
    static double draw(Words& w, const State& s) {
        if (!s.ga.small && !s.gb.small) {  // shapes >= 1: X / (X + Y) directly
            const double x = gamma_mt(w, s.ga.d, s.ga.c);
            return x / (x + gamma_mt(w, s.gb.d, s.gb.c));
        }
        if (s.ga.small && s.gb.small) {
            // both below 1: Joehnk (1964), X = U^(1/a), Y = V^(1/b) given X + Y <= 1 (accepted
            // ~79% at a = b = 1/2), the powers as exp(-E / a) of standard exponentials (the
            // ziggurat: no logs); tiny shapes underflow X and Y: in logs
            if (s.ga.a >= kDirectShape && s.gb.a >= kDirectShape) {
                for (;;) {
                    const double x = std::exp(-std_exp(w) / s.ga.a), y = std::exp(-std_exp(w) / s.gb.a);
                    const double t = x + y;
                    if (t <= 1.0 && t > 0.0) return x / t;
                }
            }
            for (;;) {
                const double ex = std_exp(w), ey = std_exp(w);
                const double lx = -ex / s.ga.a, ly = -ey / s.gb.a;
                if (std::isinf(lx) && std::isinf(ly)) {
                    // subnormal shapes: both logs overflow (X + Y <= 1 holds); X / (X + Y) is 1
                    // or 0 by the larger log, compared after scaling by the smaller shape
                    const double m = std::min(s.ga.a, s.gb.a);
                    const double sx = ex * (m / s.ga.a), sy = ey * (m / s.gb.a);
                    return sx < sy ? 1.0 : (sx > sy ? 0.0 : 0.5);
                }
                const double hi = std::max(lx, ly), lo = std::min(lx, ly);
                if (hi + std::log1p(std::exp(lo - hi)) <= 0.0) return beta_of_logs(lx, ly);
            }
        }
        const double lx = log_gamma_draw(w, s.ga);
        return beta_of_logs(lx, log_gamma_draw(w, s.gb));
    }
};
// A degree of freedom below 2^-1021 has no exact half (the subnormals and the smallest
// normal binade): the t's sample is not drawn (its half has no gamma to draw from: NaN, the
// same boundary as the value methods' StudentT::halvable, docs/api/stats.md §2.2, §2.5);
// df = inf is the normal's.
inline constexpr double kMinSampleDf = 0x1p-1021;

// Z sqrt(df / V), V = 2 G(df / 2): Z exp((log df - log 2 - log G) / 2)
template <>
struct Custom<StudentT> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double l;  // log(df / 2)
        int kind;  // 0 the t, 1 the normal (df = inf), 2 NaN (df < 2^-1021)
    };
    static State prep(double df) {
        const int kind = df == kInf ? 1 : (df < kMinSampleDf ? 2 : 0);
        return {gamma_prep(kind == 0 ? 0.5 * df : 1.0), std::log(0.5 * df), kind};
    }
    static double draw(Words& w, const State& s) {
        const double z = std_normal(w);
        if (s.kind != 0) return s.kind == 1 ? z : std::numeric_limits<double>::quiet_NaN();
        if (!s.g.small) return z * std::sqrt(s.g.a / gamma_mt(w, s.g.d, s.g.c));
        return z * std::exp(0.5 * (s.l - log_gamma_draw(w, s.g)));
    }
};
// (V1 / dfn) / (V2 / dfd) = G1 / G2 * dfd / dfn; one infinite df: G1 / (dfn / 2) (dfd = inf),
// (dfd / 2) / G2 (dfn = inf)
template <>
struct Custom<F> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g1, g2;
        double l;  // log(dfd / dfn)
        int lim;   // 0, 1 dfd = inf, 2 dfn = inf
    };
    static State prep(double dfn, double dfd) {
        const int lim = dfd == kInf ? 1 : (dfn == kInf ? 2 : 0);
        return {gamma_prep(lim == 2 ? 1.0 : 0.5 * dfn), gamma_prep(lim == 1 ? 1.0 : 0.5 * dfd),
                lim == 0 ? log_ratio(dfd, dfn) : 0.0, lim};
    }
    static double draw(Words& w, const State& s) {
        if (s.lim == 1) return gamma_draw(w, s.g1) / s.g1.a;
        if (s.lim == 2) return s.g2.a / gamma_draw(w, s.g2);
        if (!s.g1.small && !s.g2.small) {
            const double g1 = gamma_mt(w, s.g1.d, s.g1.c);
            return g1 / gamma_mt(w, s.g2.d, s.g2.c) * (s.g2.a / s.g1.a);
        }
        return std::exp(log_gamma_diff(w, s.g1, s.g2) + s.l);
    }
};
// df > 1: chi^2(df - 1) + (Z + sqrt(nc))^2 (the noncentrality on one normal coordinate: a
// gamma and a normal); df = 1: the square alone; df < 1: 2 G(df/2 + N), N ~ Poisson(nc/2).
// nc = 0: the central chi-square's own draw (its bits, as every other method).
template <>
struct Custom<NoncentralChiSquared> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        double h, df, snc;  // df / 2, df, sqrt(nc)
        PoissonPrep pn;
        GammaPrep g0;  // df / 2 (df < 1, N = 0)
        GammaPrep g1;  // (df - 1) / 2 (df > 1)
    };
    static State prep(double df, double nc) {
        return {0.5 * df, df, std::sqrt(nc), poisson_prep(0.5 * nc), gamma_prep(0.5 * df),
                gamma_prep(df > 1.0 ? 0.5 * (df - 1.0) : 1.0)};
    }
    static double draw(Words& w, const State& s) {
        if (s.snc == 0.0) return Custom<ChiSquared>::draw(w, s.g0);
        if (s.df >= 1.0) {
            const double z = std_normal(w) + s.snc;
            return s.df == 1.0 ? z * z : 2.0 * gamma_draw(w, s.g1) + z * z;
        }
        const double n = poisson_draw(w, s.pn);
        return 2.0 * gamma_draw(w, n == 0.0 ? s.g0 : gamma_prep(s.h + n));
    }
    static double log_half(Words& w, const State& s) {  // log(X / 2)
        if (s.df >= 1.0) return std::log(0.5 * draw(w, s));
        const double n = poisson_draw(w, s.pn);
        return log_gamma_draw(w, n == 0.0 ? s.g0 : gamma_prep(s.h + n));
    }
};
// (Z + nc) sqrt(df / V); df = inf: Z + nc; df < 2^-1021: NaN (as the t)
template <>
struct Custom<NoncentralT> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double l, nc;
        int kind;  // as Custom<StudentT>
    };
    static State prep(double df, double nc) {
        const int kind = df == kInf ? 1 : (df < kMinSampleDf ? 2 : 0);
        return {gamma_prep(kind == 0 ? 0.5 * df : 1.0), std::log(0.5 * df), nc, kind};
    }
    static double draw(Words& w, const State& s) {
        const double z = std_normal(w) + s.nc;
        if (s.kind != 0) return s.kind == 1 ? z : std::numeric_limits<double>::quiet_NaN();
        if (!s.g.small) return z * std::sqrt(s.g.a / gamma_mt(w, s.g.d, s.g.c));
        return z * std::exp(0.5 * (s.l - log_gamma_draw(w, s.g)));
    }
};
// (X1 / dfn) / (V2 / dfd), X1 noncentral chi-square(dfn, nc); dfd = inf: X1 / dfn; dfn = inf:
// (dfd / 2) / G2; either degree of freedom below 2^-1021 (no exact half): NaN, as the value
// methods (user decision 2026-11-19, docs/api/stats.md §2.5); nc = 0: the central F's own
// draw (its bits).
template <>
struct Custom<NoncentralF> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        Custom<NoncentralChiSquared>::State x1;
        GammaPrep g2;
        double l, r;  // log(dfd / dfn), dfd / dfn
        double dfn;
        int lim;  // 0, 1 dfd = inf, 2 dfn = inf, 3 NaN, 4 nc = 0 (the central F)
        Custom<F>::State central;
    };
    static State prep(double dfn, double dfd, double nc) {
        const int lim = dfd == kInf ? 1 : (dfn == kInf ? 2 : (dfn < kMinSampleDf || dfd < kMinSampleDf ? 3 : 0));
        const int kind = nc == 0.0 && lim != 3 ? 4 : lim;
        return {Custom<NoncentralChiSquared>::prep(lim == 2 ? 1.0 : dfn, nc), gamma_prep(lim == 1 ? 1.0 : 0.5 * dfd),
                lim == 0 ? log_ratio(dfd, dfn) : 0.0, dfd / dfn, dfn, kind,
                kind == 4 ? Custom<F>::prep(dfn, dfd) : Custom<F>::State{}};
    }
    static double draw(Words& w, const State& s) {
        if (s.lim == 4) return Custom<F>::draw(w, s.central);
        if (s.lim == 3) return std::numeric_limits<double>::quiet_NaN();
        if (s.lim == 1) return Custom<NoncentralChiSquared>::draw(w, s.x1) / s.dfn;
        if (s.lim == 2) return s.g2.a / gamma_draw(w, s.g2);
        if (s.x1.df >= 1.0 && !s.g2.small) {  // no tiny values: the ratio directly
            const double x1 = Custom<NoncentralChiSquared>::draw(w, s.x1);
            return 0.5 * x1 / gamma_mt(w, s.g2.d, s.g2.c) * s.r;
        }
        const double l1 = Custom<NoncentralChiSquared>::log_half(w, s.x1);
        return std::exp(l1 - log_gamma_draw(w, s.g2) + s.l);
    }
};

// ---- discrete

template <>
struct Custom<Bernoulli> {
    static double draw(Words& w, double p) { return w.u01() < p ? 1.0 : 0.0; }
};
template <>
struct Custom<DiscreteUniform> {
    static double draw(Words& w, double lo, double hi) {
        // more integers than one word indexes (or a width that overflowed): NaN
        if (!(hi - lo < 0x1p64)) return std::numeric_limits<double>::quiet_NaN();
        const auto r = static_cast<std::uint64_t>(hi - lo);  // [low, high): integers below 2^53 apart
        return lo + static_cast<double>(random::bounded(w.next(), r, [&](std::uint64_t) { return w.next(); }));
    }
};
template <>
struct Custom<Geometric> {
    // trials to the first success (k >= 1): ceil(log U / log(1 - p)), P(K > k) = (1 - p)^k
    static double prep(double p) { return p == 1.0 ? 0.0 : 1.0 / std::log1p(-p); }
    static double draw(Words& w, double inv) {
        if (inv == 0.0) return 1.0;
        return std::max(1.0, std::ceil(std::log(w.u()) * inv));
    }
};
template <>
struct Custom<Poisson> {
    static constexpr bool kExtra = true;  // draws extra words
    static PoissonPrep prep(double mu) { return poisson_prep(mu); }
    static bool extra(const PoissonPrep& s) { return s.ptrs; }  // inversion: one word
    static double draw(Words& w, const PoissonPrep& s) { return poisson_draw(w, s); }
};
template <>
struct Custom<Binomial> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        BinomPrep b;
        double odds;  // binom_odds(b)
    };
    static State prep(double n, double p) {
        if (p == 1.0) {
            BinomPrep s{};
            s.n = n, s.p = 0.0, s.flip = true;
            return {s, 0.0};
        }
        const BinomPrep s = binom_prep(n, p, num::dd::two_sum(1.0, -p));
        return {s, binom_odds(s)};
    }
    static double draw(Words& w, const State& s) { return binom_draw(w, s.b, s.odds); }
    static bool extra(const State& s) { return s.b.btrs; }
};
// Poisson(G(n) q / p)
template <>
struct Custom<NegativeBinomial> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double lr;  // log(q / p)
        bool one;   // p = 1
        double r;   // q / p
    };
    static State prep(double n, double p) {  // (n = 0 or p = 1: the point mass at 0)
        const double lr = std::log1p(-p) - std::log(p);
        return {gamma_prep(n == 0.0 ? 1.0 : n), lr, p == 1.0 || n == 0.0, std::exp(lr)};
    }
    static double draw(Words& w, const State& s) {
        if (s.one) return 0.0;
        // n >= 1 and a ratio in range: the product directly (no log and exp)
        const bool direct = !s.g.small && s.r >= 0x1p-500 && s.r <= 0x1p500;
        const double lam = direct ? gamma_mt(w, s.g.d, s.g.c) * s.r : std::exp(log_gamma_draw(w, s.g) + s.lr);
        if (!std::isfinite(lam)) return lam;
        return poisson_draw(w, poisson_prep(lam));
    }
};
// Binomial(n, B), B ~ Beta(a, b): p and q = 1 - p both from the gamma logs
template <>
struct Custom<BetaBinomial> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        double n;
        GammaPrep ga, gb;
    };
    static State prep(double n, double a, double b) { return {n, gamma_prep(a), gamma_prep(b)}; }
    static double draw(Words& w, const State& s) {
        if (s.n == 0.0) return 0.0;
        const double d = log_gamma_diff(w, s.ga, s.gb);  // log X - log Y
        const double p = beta_of_logs(0.0, -d), q = beta_of_logs(0.0, d);
        if (p == 0.0) return 0.0;
        if (q == 0.0) return s.n;
        // p and q each to a few ULP; the flip in binom_prep takes the smaller one as given
        BinomPrep bp = p <= q ? binom_prep(s.n, p, num::DD{q, 0.0}) : binom_prep(s.n, q, num::DD{p, 0.0});
        const double x = binom_draw(w, bp);
        return p <= q ? x : s.n - x;
    }
};
template <>
struct Custom<Hypergeometric> {
    struct State {
        double N, K, n, lo, hi, mode, pmode;  // lo, hi: the walk's ends
        double nl, nh;                        // the pmfs next to the mode (from_mode)
    };
    // P(k+1)/P(k) = (K-k)(n-k) / ((k+1)(N-K-n+k+1)) and P(k-1)/P(k)
    static double up(double k, double N, double K, double n) { return (K - k) * (n - k) / ((k + 1.0) * (N - K - n + k + 1.0)); }
    static double down(double k, double N, double K, double n) { return k * (N - K - n + k) / ((K - k + 1.0) * (n - k + 1.0)); }
    // The walk from the mode reaches the quantile of u within |z| <= 8.3 standard deviations
    // (u is a multiple of 2^-53): it is cut at kModeReach = 16 kMaxSd from the mode (2^21
    // steps at most), and a standard deviation above kMaxSd, or a mode without a pmf (the
    // distribution's own NaN beyond 2^53), is answered NaN without walking (in the
    // guaranteed region, total <= 3000, sd <= 14).
    static constexpr double kMaxSd = 65536.0;
    static State prep(double N, double K, double n) {
        const double lo = Hypergeometric::lo_of(N, K, n), hi = Hypergeometric::hi_of(N, K, n);
        const double m = std::clamp(Hypergeometric::mode_point(N, K, n), lo, hi);
        const double f = K / N, var = n * f * (1.0 - f) * ((N - n) / std::max(N - 1.0, 1.0));
        const double pm = Hypergeometric::pmf(m, N, K, n);
        const bool nan = !(var <= kMaxSd * kMaxSd) || !(pm > 0.0);
        // The answers without a walk cost a draw nothing: an infinite pmf at the "mode" makes
        // the walk's first test (u below the mode's pmf) return it, the one point or NaN.
        constexpr double inf = std::numeric_limits<double>::infinity();
        if (lo == hi) return {N, K, n, lo, hi, lo, inf, 0.0, 0.0};
        if (nan) return {N, K, n, lo, hi, std::numeric_limits<double>::quiet_NaN(), inf, 0.0, 0.0};
        const double wl = std::max(lo, m - kModeReach), wh = std::min(hi, m + kModeReach);
        return {N, K, n, wl, wh, m, pm, first_down(m, wl, pm, [&](double k) { return down(k, N, K, n); }),
                first_up(m, wh, pm, [&](double k) { return up(k, N, K, n); })};
    }
    static double draw(Words& w, const State& s) {
        const double N = s.N, K = s.K, n = s.n;
        return from_mode(w.u01(), s.mode, s.lo, s.hi, s.pmode, s.nl, s.nh, [&](double k) { return up(k, N, K, n); },
                         [&](double k) { return down(k, N, K, n); });
    }
};
// Kemp's LK algorithm (1981; numpy's random_logseries)
template <>
struct Custom<LogSeries> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        double p, r;  // r = log(1 - p)
    };
    static State prep(double p) { return {p, std::log1p(-p)}; }
    static double draw(Words& w, const State& s) {
        for (;;) {
            const double V = w.u01();
            if (V >= s.p) return 1.0;
            const double U = w.u01();
            const double q = -std::expm1(s.r * U);
            if (V <= q * q) {
                const double k = std::floor(1.0 + std::log(V) / std::log(q));
                if (k < 1.0 || V == 0.0) continue;
                return k;
            }
            return V >= q ? 1.0 : 2.0;
        }
    }
};
// Devroye's rejection (Non-Uniform Random Variate Generation, 1986, X.6; numpy's random_zipf),
// T - 1 and b - 1 by expm1 so the test keeps its bits for huge X; inf beyond float64 (T25)
template <>
struct Custom<Zipf> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        double am1, inv, b, bm1;
    };
    static State prep(double a) {
        const double am1 = a - 1.0;
        return {am1, 1.0 / am1, std::exp2(am1), std::expm1(am1 * special::kLn2)};
    }
    static double draw(Words& w, const State& s) {
        // 2^(a-1) beyond float64 (a > 1025): P(X > 1) < 2^-1024, and the test below would
        // compare inf / inf forever
        if (!(s.b < std::numeric_limits<double>::infinity())) return s.b != s.b ? s.b : 1.0;
        for (;;) {
            const double U = 1.0 - w.u01(), V = w.u01();  // U in (0, 1]
            const double X = std::floor(std::exp(-std::log(U) * s.inv));
            if (std::isinf(X)) {  // beyond float64: the test's limit, X (T - 1) -> a - 1, T -> 1
                if (V * s.am1 / s.bm1 <= 1.0 / s.b) return X;
                continue;
            }
            if (X < 1.0) continue;
            const double tm1 = std::expm1(s.am1 * std::log1p(1.0 / X));
            if (V * X * tm1 / s.bm1 <= (1.0 + tm1) / s.b) return X;
        }
    }
};

}  // namespace sd::stats::smp
