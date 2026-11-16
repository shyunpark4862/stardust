// The distributions' own sampling methods (T7, docs/design/stats.md §4), where they beat
// inversion (sample.hpp; measured in bench/stats/t7/). Every method is exact in
// distribution (rejection or composition, no approximation); the acceptance tests of the
// discrete rejections use the double-double log pmfs of the distributions, so they stay
// exact where lgamma differences of size 1e16 would not (huge mu or n).
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

inline double std_normal(Words& w) {
    const ZigNormal& z = zig_normal();
    for (;;) {
        std::uint64_t r = w.next();
        const int idx = static_cast<int>(r & 0xff);
        r >>= 8;
        const bool neg = r & 1;
        const std::uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
        // the sign by its bit (a branch on a random bit mispredicts half the time: 3 ns)
        const double x = std::bit_cast<double>(std::bit_cast<std::uint64_t>(static_cast<double>(rabs) * z.wi[idx]) |
                                               (static_cast<std::uint64_t>(neg) << 63));
        if (rabs < z.ki[idx]) return x;  // inside the layer's rectangle (~99%)
        if (idx == 0) {                  // the tail beyond R
            for (;;) {
                const double xx = -std::log(w.u()) / ZigNormal::kR;
                const double yy = -std::log(w.u());
                if (yy + yy > xx * xx) return neg ? -(ZigNormal::kR + xx) : ZigNormal::kR + xx;
            }
        }
        if ((z.fi[idx - 1] - z.fi[idx]) * w.u01() + z.fi[idx] < std::exp(-0.5 * x * x)) return x;
    }
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

inline double std_exp(Words& w) {
    const ZigExp& z = kZigExp;
    for (;;) {
        std::uint64_t r = w.next() >> 3;
        const int idx = static_cast<int>(r & 0xff);
        r >>= 8;
        const double x = static_cast<double>(r) * z.we[idx];
        if (r < z.ke[idx]) return x;  // ~99%
        if (idx == 0) return ZigExp::kR - std::log(w.u());
        if ((z.fe[idx - 1] - z.fe[idx]) * w.u01() + z.fe[idx] < std::exp(-x)) return x;
    }
}

// ---- gamma: Marsaglia and Tsang (2000) for a >= 1; for a < 1, G(a) = G(a + 1) U^(1/a)
// (Stuart), in logs so a tiny a's sample need not underflow before its use in a ratio.

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
inline double gamma_mt(Words& w, double d, double c) {
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
// log G(a)
inline double log_gamma_draw(Words& w, const GammaPrep& g) {
    const double l = std::log(gamma_mt(w, g.d, g.c));
    return g.small ? l + std::log(w.u()) / g.a : l;
}
inline double gamma_draw(Words& w, const GammaPrep& g) {
    return g.small ? std::exp(log_gamma_draw(w, g)) : gamma_mt(w, g.d, g.c);
}

// ---- Poisson: inversion from 0 for mu < 10 (one uniform, ~mu steps), PTRS (Hoermann,
// "The transformed rejection method for generating Poisson random variables", 1993) above.

struct PoissonPrep {
    double mu;
    bool ptrs;
    double e0;                                   // e^-mu (inversion)
    double b, a, invalpha, vr, log_ia;           // PTRS
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
        const num::DD l = Poisson::log_pmf_dd(k, s.mu);
        if (std::log(V) + s.log_ia - std::log(s.a / (us * us) + s.b) <= l.hi + l.lo) return k;
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
    num::DD lm;                    // log pmf(m)
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
        s.qn = std::exp((num::dd::log(q) * n).hi);
        return s;
    }
    const double spq = std::sqrt(n * p * q.hi);
    s.b = 1.15 + 2.53 * spq;
    s.a = -0.0873 + 0.0248 * s.b + 0.01 * p;
    s.c = n * p + 0.5;
    s.vr = 0.92 - 4.2 / s.b;
    s.alpha = (2.83 + 5.1 / s.b) * spq;
    s.m = std::floor((n + 1.0) * p);
    s.lm = binom_detail::log_pmf(s.m, n, p, q);
    return s;
}
inline double binom_draw(Words& w, const BinomPrep& s) {
    if (s.n == 0.0 || s.p == 0.0) return s.flip ? s.n : 0.0;
    double x;
    if (!s.btrs) {
        // inversion: P(X = k + 1) / P(X = k) = (n - k) p / ((k + 1) q)
        const double r = s.p / s.q.hi;
        double u = w.u01(), px = s.qn, k = 0.0;
        while (u >= px && k < s.n) {
            u -= px;
            px *= (s.n - k) * r / (k + 1.0);
            k += 1.0;
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
            const num::DD d = binom_detail::log_pmf(k, s.n, s.p, s.q) - s.lm;
            if (v <= d.hi + d.lo) {
                x = k;
                break;
            }
        }
    }
    return s.flip ? s.n - x : x;
}

// ---- inversion outward from the mode for a unimodal pmf (hypergeometric): the support's
// points in the fixed order of their pmf from the mode outwards (the larger neighbour
// first), their pmfs by the ratio recurrence, until the running sum passes u: exact
// inversion over that order, ~ (standard deviation) steps.
template <class Up, class Down>
double from_mode(double u, double mode, double lo, double hi, double pmode, Up&& up, Down&& down) {
    double s = pmode;
    if (u < s) return mode;
    double kl = mode, kh = mode, pl = pmode, ph = pmode;
    double nl = kl > lo ? pl * down(kl) : 0.0, nh = kh < hi ? ph * up(kh) : 0.0;
    for (;;) {
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
        if (!s.g.small) return s.scale * gamma_mt(w, s.g.d, s.g.c);
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
// X / (X + Y) from log X, log Y (no underflow for tiny shapes)
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
            // both below 1: Joehnk (1964), X = U^(1/a), Y = V^(1/b) given X + Y <= 1, in logs
            // (tiny shapes underflow X and Y; accepted ~79% at a = b = 1/2)
            for (;;) {
                const double lx = std::log(w.u()) / s.ga.a, ly = std::log(w.u()) / s.gb.a;
                const double hi = std::max(lx, ly), lo = std::min(lx, ly);
                if (hi + std::log1p(std::exp(lo - hi)) <= 0.0) return beta_of_logs(lx, ly);
            }
        }
        const double lx = log_gamma_draw(w, s.ga);
        return beta_of_logs(lx, log_gamma_draw(w, s.gb));
    }
};
// Z sqrt(df / V), V = 2 G(df / 2): Z exp((log df - log 2 - log G) / 2)
template <>
struct Custom<StudentT> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double l;  // log(df / 2)
    };
    static State prep(double df) { return {gamma_prep(0.5 * df), std::log(0.5 * df)}; }
    static double draw(Words& w, const State& s) {
        const double z = std_normal(w);
        if (!s.g.small) return z * std::sqrt(s.g.a / gamma_mt(w, s.g.d, s.g.c));
        return z * std::exp(0.5 * (s.l - log_gamma_draw(w, s.g)));
    }
};
// (V1 / dfn) / (V2 / dfd) = G1 / G2 * dfd / dfn
template <>
struct Custom<F> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g1, g2;
        double l;  // log(dfd / dfn)
    };
    static State prep(double dfn, double dfd) { return {gamma_prep(0.5 * dfn), gamma_prep(0.5 * dfd), std::log(dfd / dfn)}; }
    static double draw(Words& w, const State& s) {
        if (!s.g1.small && !s.g2.small) {
            const double g1 = gamma_mt(w, s.g1.d, s.g1.c);
            return g1 / gamma_mt(w, s.g2.d, s.g2.c) * (s.g2.a / s.g1.a);
        }
        const double l1 = log_gamma_draw(w, s.g1);
        return std::exp(l1 - log_gamma_draw(w, s.g2) + s.l);
    }
};
// 2 G(df/2 + N), N ~ Poisson(nc/2)
template <>
struct Custom<NoncentralChiSquared> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        double h;
        PoissonPrep pn;
        GammaPrep g0;  // nc = 0
    };
    static State prep(double df, double nc) { return {0.5 * df, poisson_prep(0.5 * nc), gamma_prep(0.5 * df)}; }
    static double log_half(Words& w, const State& s) {  // log of the G
        const double n = poisson_draw(w, s.pn);
        return log_gamma_draw(w, n == 0.0 ? s.g0 : gamma_prep(s.h + n));
    }
    static double draw(Words& w, const State& s) {
        const double n = poisson_draw(w, s.pn);
        return 2.0 * gamma_draw(w, n == 0.0 ? s.g0 : gamma_prep(s.h + n));
    }
};
// (Z + nc) sqrt(df / V)
template <>
struct Custom<NoncentralT> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double l, nc;
    };
    static State prep(double df, double nc) { return {gamma_prep(0.5 * df), std::log(0.5 * df), nc}; }
    static double draw(Words& w, const State& s) {
        const double z = std_normal(w) + s.nc;
        if (!s.g.small) return z * std::sqrt(s.g.a / gamma_mt(w, s.g.d, s.g.c));
        return z * std::exp(0.5 * (s.l - log_gamma_draw(w, s.g)));
    }
};
// (X1 / dfn) / (V2 / dfd), X1 noncentral chi-square(dfn, nc)
template <>
struct Custom<NoncentralF> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        Custom<NoncentralChiSquared>::State x1;
        GammaPrep g2;
        double l;
    };
    static State prep(double dfn, double dfd, double nc) {
        return {Custom<NoncentralChiSquared>::prep(dfn, nc), gamma_prep(0.5 * dfd), std::log(dfd / dfn)};
    }
    static double draw(Words& w, const State& s) {
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
    static BinomPrep prep(double n, double p) {
        if (p == 1.0) {
            BinomPrep s{};
            s.n = n, s.p = 0.0, s.flip = true;
            return s;
        }
        return binom_prep(n, p, num::dd::two_sum(1.0, -p));
    }
    static double draw(Words& w, const BinomPrep& s) { return binom_draw(w, s); }
    static bool extra(const BinomPrep& s) { return s.btrs; }
};
// Poisson(G(n) q / p)
template <>
struct Custom<NegativeBinomial> {
    static constexpr bool kExtra = true;  // draws extra words
    struct State {
        GammaPrep g;
        double lr;  // log(q / p)
        bool one;   // p = 1
    };
    static State prep(double n, double p) {
        return {gamma_prep(n), std::log1p(-p) - std::log(p), p == 1.0};
    }
    static double draw(Words& w, const State& s) {
        if (s.one) return 0.0;
        const double lam = std::exp(log_gamma_draw(w, s.g) + s.lr);
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
        const double lx = log_gamma_draw(w, s.ga), ly = log_gamma_draw(w, s.gb);
        const double p = beta_of_logs(lx, ly), q = beta_of_logs(ly, lx);
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
        double N, K, n, lo, hi, mode, pmode;
    };
    static State prep(double N, double K, double n) {
        const double lo = Hypergeometric::lo_of(N, K, n), hi = Hypergeometric::hi_of(N, K, n);
        const double m = std::clamp(Hypergeometric::mode_point(N, K, n), lo, hi);
        return {N, K, n, lo, hi, m, Hypergeometric::pmf(m, N, K, n)};
    }
    static double draw(Words& w, const State& s) {
        if (s.lo == s.hi) return s.lo;
        const double N = s.N, K = s.K, n = s.n;
        // P(k+1)/P(k) = (K-k)(n-k) / ((k+1)(N-K-n+k+1))
        auto up = [&](double k) { return (K - k) * (n - k) / ((k + 1.0) * (N - K - n + k + 1.0)); };
        auto down = [&](double k) { return k * (N - K - n + k) / ((K - k + 1.0) * (n - k + 1.0)); };
        return from_mode(w.u01(), s.mode, s.lo, s.hi, s.pmode, up, down);
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
