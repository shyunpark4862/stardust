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

inline constexpr double kTol = 0x1p-56;

// D(a, mu) as m 2^e; D(0, mu) = e^-mu. mu = hi + lo (the t's nc^2 / 2 is not a
// double; rounding it moved its tails by nc^2 eps |d log F / d mu|, 89 ULP at nc = 17):
// d log D / d mu = a / mu - 1.
inline Scaled prefix_s(double a, double mu, double mu_lo = 0.0) {
    const Scaled d = a == 0.0 ? exp_scaled(-mu) : gamma_prefix(a, gamma_exponent(a, mu));
    return mu_lo == 0.0 ? d : d * scaled(1.0 + mu_lo * (a / mu - 1.0));
}

// The shape s0 + j is a double-double (two_sum): rounded to a double it would
// move G by s eps |d log G / ds|, which is s eps |log(y/s)| in the tails (250 ULP
// at s = 800). G and T are evaluated at s.hi and moved by s.lo to first order: T
// by its log-derivative (digamma), G by a forward difference over 2^-20 s (the
// correction is ~eps of G, so the difference's own error, ~2^-20, is nothing).
template <class F>
inline Scaled at_shape(const F& g, dd::DD s) {
    const Scaled v = g(s.hi);
    if (s.lo == 0.0 || v.m == 0.0) return v;
    const double up = s.hi + std::ldexp(s.hi, -20);
    const double dlog = (value(g(up) / v) - 1.0) / (up - s.hi);
    return v * scaled(1.0 + s.lo * dlog);
}

// sum_k r_0 r_1 ... r_(k-1) (k >= 0) for ratios r_k < 1 eventually, as m 2^e: the
// terms may climb far above the double range first (1 - I_x(s, b) at a large s).
template <class R>
inline Scaled ratio_series(const R& ratio) {
    double sum = 1.0, term = 1.0;
    std::int64_t e = 0;
    for (int k = 0; k < 10000000; ++k) {
        term *= ratio(k);
        sum += term;
        if (sum > 0x1p500) sum = std::ldexp(sum, -500), term = std::ldexp(term, -500), e += 500;
        if (term <= sum * 0x1p-56) break;
    }
    Scaled r = scaled(sum);
    r.e += e;
    return r;
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
        const Scaled d = gamma_prefix(s, gamma_exponent(s, y));
        if (!upper) return d * ratio_series([&](int k) { return y / (s + k + 1.0); });  // P = D sum_k y^k / (s+1)_k
        const double tail = lentz_backward(  // Q = D s / tail (igam_detail::log_tail's fraction)
                                y + 1.0 - s,
                                [&](int n) {
                                    const double k = n;
                                    return std::pair{-k * (k - s), y + 2.0 * k + 1.0 - s};
                                },
                                100000)
                                .value;
        return d * scaled(s / tail);
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
    double gbound(dd::DD s) const { return std::min(1.0, ratio(s)); }  // >= P(s+1) / P(s)
    // log of a bound on G(s): P(s) <= T(s) / (1 - y/(s+1)) (the series' ratios are at most
    // y/(s+1)); Q(s) <= T(s-1) y / (y - s + 1) (below); else 1
    bool deep(double s) const { return !(y > 0x1p-20 * (s + 1.0) && y < 0x1p20 * (s + 1.0)); }
    double log_gbound(double s) const {  // rough logs (for the e^-2000 test)
        auto log_d = [&](double a) { return a > 0.0 ? a * std::log(y) - y - std::lgamma(a + 1.0) : -y; };
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
            const Scaled t = beta_prefix(s, b, beta_exponent(s, b, x, y)) / scaled(s);
            return t * ratio_series([&](int k) { return x.hi * (s + k + b) / (s + k + 1.0); });
        }
        // 1 - I_x(s, b) = I_y(b, s) = sum_k y^c x^s / (c B(c, s)), c = b + k
        const Scaled t = beta_prefix(b, s, beta_exponent(b, s, y, x)) / scaled(b);
        return t * ratio_series([&](int k) { return y.hi * (b + k + s) / (b + k + 1.0); });
    }
    Scaled G(dd::DD s) const { return at_shape([&](double v) { return G0(v); }, s); }
    Scaled T(dd::DD s) const {
        const Scaled t = beta_prefix(s.hi, b, beta_exponent(s.hi, b, x, y)) / scaled(s.hi);
        if (s.lo == 0.0) return t;
        return t * scaled(1.0 + s.lo * (std::log(x.hi) - 1.0 / s.hi - Digamma::f(s.hi) + Digamma::f(s.hi + b)));
    }
    double ratio(dd::DD s) const {  // T(s+1) / T(s) = x (s + b) / (s + 1)
        const dd::DD n = s + b, d = s + 1.0;
        return x.hi * (n.hi / d.hi) * (1.0 + (n.lo / n.hi - d.lo / d.hi) + x.lo / x.hi);
    }
    dd::DD up_dd(dd::DD s) const { return x * (s + b) * recip(s + 1.0); }  // T(s+1) / T(s)
    dd::DD down_dd(dd::DD s) const { return (s + 1.0) * recip(s + b) * inv_x; }  // T(s) / T(s+1)
    double gbound(dd::DD s) const {  // >= I(s+1) / I(s): the T ratio, times e^((1-b)/(s+b)) for b < 1
        const double r = ratio(s) * (b < 1.0 ? std::exp((1.0 - b) / (s.hi + b)) : 1.0);
        return std::min(1.0, r);
    }
    double gbound_down(dd::DD) const { return 1.0; }  // >= (1 - I(s-1)) / (1 - I(s))
    // log of a bound on G(s): I_x(s, b) <= T(s) / (1 - r) for the ratio bound r < 1 (gbound's),
    // 1 - I_x(s, b) = I_y(b, s) likewise with the roles turned; else 1
    bool deep(double) const { return x.hi < 0x1p-20 || y.hi < 0x1p-20; }
    double log_gbound(double s) const {  // rough logs (for the e^-2000 test)
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

// Neumaier's compensated sum.
struct Sum {
    double s = 0.0, c = 0.0;
    void add(double v) {
        const double t = s + v;
        c += std::fabs(s) >= std::fabs(v) ? (s - t) + v : (v - t) + s;
        s = t;
    }
    double value() const { return s + c; }
    void scale(int e) { s = std::ldexp(s, e), c = std::ldexp(c, e); }
};

// The sum. j* (the largest term, roughly) splits the terms into the side where
// the recurrence for G adds T (stable) and the side where it would subtract. The
// walk starts on the second side at J, far enough out that a rigorous bound on the
// terms from j* to J (the weights' ratio times a bound on G's ratio) has fallen
// below kTol of the term at j*, and comes back across j* to where the terms fade:
// one direct evaluation of w, G and T, at J. The terms are summed in the frame 2^k0
// of the first term, w kept in [2^-500, 2^500] by moving powers of two between w
// and G (their product unchanged), so terms far below the double range keep their bits.
template <class Fam>
double mixture(double mu, double h, double s0, const Fam& f, double mu_lo = 0.0) {
    const bool lower = !f.upper;
    auto shape = [&](double j) { return dd::two_sum(s0, j); };  // s0 + j exactly
    auto term_ratio = [&](double j) {  // (term j+1) / (term j), with G's ratio taken as T's (deep tails)
        return mu / (j + 1.0 + h) * (lower ? f.ratio(shape(j)) : (s0 + j > 0 ? f.ratio(shape(j - 1.0)) : 1.0));
    };
    // j*: the weights' mode, moved to where term_ratio crosses 1 if that is further
    // into the tail (below the mode for a lower family, above it for an upper one).
    const double jmode = std::max(0.0, std::floor(mu - h));
    // The crossing matters only on the tail side of the mode: below it for a lower family
    // (term_ratio(jmode) <= 1), above it for an upper one (> 1). Else j* is the mode itself
    // (one ratio instead of a bisection: most inputs).
    double jstar = jmode;
    const bool past = term_ratio(jmode) <= 1.0;
    if (lower ? (past && jmode > 0.0) : !past) {
        double lo = lower ? 0.0 : jmode, hi = lower ? jmode : std::ceil(std::max(4.0 * mu, jmode + 64.0));  // integers
        if (!lower)
            while (term_ratio(hi) > 1.0 && hi < 0x1p52) hi *= 2.0;
        if (lower && term_ratio(0.0) <= 1.0) hi = 0.0;
        while (hi - lo > 1.0) {
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
        (a > 0.0 ? a * std::log(mu) - mu - std::lgamma(a + 1.0) : -mu) + f.log_gbound(jstar + s0) < -2000.0)
        return 0.0;

    // J: past j* on the unstable side (up for a lower family, down for an upper one)
    // until the bound times the bounded tail beyond, prod / (1 - r), is below kTol.
    double J = jstar, prod = 1.0;
    for (;;) {
        double r;
        if (lower) {
            r = mu / (J + 1.0 + h) * f.gbound(shape(J));
            J += 1.0;
        } else {
            if (J == 0.0) break;
            r = (J - 1.0 + h + 1.0) / mu * f.gbound_down(shape(J));  // w_(J-1) / w_J = (J + h) / mu
            J -= 1.0;
        }
        prod *= r;
        if (r < 1.0 && prod <= kTol * (1.0 - r)) break;
        if (prod > 0x1p500) prod = 0x1p500;  // the climb to the weights' mode: only its fall matters
        if (std::fabs(J - jstar) > 1e9) break;  // (never in range; a guard)
    }

    const Scaled wJ = prefix_s(J + h, mu, mu_lo), gJ = f.G(shape(J));
    const Scaled tJ = f.T(shape(lower ? J - 1.0 : J));  // the T the first step adds
    const std::int64_t k0 = -(wJ.e + (gJ.m != 0.0 ? gJ.e : tJ.e));
    auto frame = [](Scaled v, std::int64_t k) {
        return std::ldexp(v.m, static_cast<int>(std::clamp<std::int64_t>(v.e + k, -2200, 2200)));
    };
    // w and T follow their ratios in double-double (exact to ~2^-100 a step: no drift
    // over the hundreds of steps from J, and nothing to recompute); G and the sum are
    // compensated sums of doubles.
    std::int64_t kw = -wJ.e, kg = k0 - kw, k = k0;  // w in 2^kw, G and T in 2^kg, terms in 2^k
    dd::DD w{frame(wJ, kw), 0.0}, t{frame(tJ, kg), 0.0};
    const dd::DD mud{mu, mu_lo}, inv_mu = recip(mud);
    Sum g;
    g.add(frame(gJ, kg));
    Sum S;
    double prev = w.hi * g.value(), j = J;
    S.add(prev);
    const double dir = lower ? -1.0 : 1.0;
    auto shift = [](dd::DD v, int e) { return dd::DD{std::ldexp(v.hi, e), std::ldexp(v.lo, e)}; };
    for (long n = 1; n < 100000000; ++n) {
        if (lower && j == 0.0) break;
        const double jn = j + dir;
        g.add(t.hi);
        g.add(t.lo);
        w = lower ? w * (j + h) * inv_mu : w * mud * recip(dd::DD{j + 1.0 + h, 0.0});
        j = jn;
        double gv = g.value();
        if (!(std::fabs(w.hi) <= 0x1p500 && std::fabs(w.hi) >= 0x1p-500) && w.hi != 0.0) {  // w back near 1, the term unchanged
            const int e = std::ilogb(w.hi);
            kw -= e, kg += e;
            w = shift(w, -e), t = shift(t, e);
            g.scale(e);
            gv = g.value();
        }
        if (!(std::fabs(gv) <= 0x1p500 && std::fabs(gv) >= 0x1p-500) && gv != 0.0) {  // the frame follows the terms
            const int e = std::ilogb(gv);
            kg -= e, k -= e;
            t = shift(t, -e), prev = std::ldexp(prev, -e);
            g.scale(-e), S.scale(-e);
            gv = g.value();
        }
        const double term = w.hi * gv + w.lo * gv;
        S.add(term);
        t = lower ? t * f.down_dd(shape(j - 1.0)) : t * f.up_dd(shape(j - 1.0));
        // past j* and fading: the terms beyond fall at least geometrically (log-concave)
        if ((lower ? j < jstar : j > jstar) && term < prev && term <= kTol * S.value() * (1.0 - term / prev)) break;
        prev = term;
    }
    return std::min(1.0, value(Scaled{S.value(), -k}));  // a probability (rounding may pass 1 by an ULP)
}

// The upper family's sum when its small argument y is below 2^-1000 (it may not
// even be a double: t ~ 1e250, f ~ 1e300), from I_y(b, s) = y^b Gamma(b+s) /
// (Gamma(b+1) Gamma(s)) (1 + O(y)):
//     sum_j D(j + h, mu) I_y(b, s0 + j) = y^b / Gamma(b+1) sum_j D(j + h, mu) Gamma(b + s0 + j) / Gamma(s0 + j),
// the terms by lgamma in logs (the tails of the heavy-tailed laws at astronomic x:
// outside the measured range).
inline double tiny_upper(double mu, double h, double s0, double b, dd::DD logy) {
    auto lterm = [&](double j) {
        const double a = j + h;
        const double ld = a > 0.0 ? a * std::log(mu) - mu - std::lgamma(a + 1.0) : -mu;
        return ld + std::lgamma(b + s0 + j) - std::lgamma(s0 + j);
    };
    double peak = -kInf, sum = 0.0;  // sum of e^(lterm - peak), peak rescaled as it rises
    for (double j = 0.0; j < 1e8; j += 1.0) {
        const double l = lterm(j);
        if (l > peak) {
            sum = sum * std::exp(peak - l) + 1.0;
            peak = l;
        } else {
            sum += std::exp(l - peak);
            if (j > mu && l < peak - 45.0) break;  // past the weights' mode and the peak: e^-45
        }
        if (mu == 0.0 && h == 0.0) break;  // only j = 0
    }
    // b log y in double-double: it reaches -1e3, whose rounding alone was 4e-14 of the result
    return distr_detail::exp_dd(logy * b + (peak - std::lgamma(b + 1.0))) * sum;
}

// chndtr (upper: chndtrc) for 0 < df, nc, x finite (df = 0 too).
inline double ncx2(double df, double nc, double x, bool upper) {
    return mixture(0.5 * nc, 0.0, 0.5 * df, GammaFamily{0.5 * x, upper});
}

// ncfdtr (upper: ncfdtrc) for finite positive dfn, dfd, nc and 0 < x < inf.
inline double ncf(double dfn, double dfd, double nc, double x, bool upper) {
    const distr_detail::Arg g = distr_detail::f_arg(dfn, dfd, x);
    if (g.y.hi < distr_detail::kTinyArg) {  // v = dfd / (dfn x + dfd) below 2^-1000: the upper tail from v^b
        const dd::DD logv = dd::log(dfd) - dd::log(dfn) - dd::log(x);
        const double up = std::min(1.0, tiny_upper(0.5 * nc, 0.0, 0.5 * dfn, 0.5 * dfd, logv));
        return upper ? up : 1.0 - up;
    }
    if (g.x.hi < distr_detail::kTinyArg) {
        // w below 2^-1000: only the j = 0 term counts (the next carries a factor w), fdtr's own series
        const double lower = std::exp(-0.5 * nc) * Fdtr::f(dfn, dfd, x);
        return upper ? 1.0 - lower : lower;
    }
    return mixture(0.5 * nc, 0.0, 0.5 * dfn, BetaFamily{0.5 * dfd, g.x, g.y, upper});
}

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
// written without cancellation; the integral over v = v* + sigma sinh(u) is the
// trapezoid rule in u (double-exponential decay on both sides).
inline double nct_opposite(double df, double a, double t) {
    const double root = std::hypot(t * std::sqrt(a * a + 4.0 * df), 2.0 * df);
    const double sstar = 2.0 * df / (a * t + root);  // (t^2 + df) s^2 + a t s - df = 0
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
    Sum I;
    I.add(integrand(0.0));
    for (int dir = -1; dir <= 1; dir += 2) {
        for (int k = 1; k < 100000; ++k) {
            const double u = dir * k * kStep / kA;
            const double v = integrand(c * kA * std::sinh(u)) * std::cosh(u);
            I.add(v);
            if (v <= 0x1p-60 * I.value() && k > 8) break;
        }
    }
    const double integral = I.value() * kStep * c;
    return value(exp_scaled(Estar.hi) * scaled((1.0 + Estar.lo) * integral));
}

// P(T <= t) (upper: P(T > t)) for finite df > 0, nc != 0, t finite.
inline double nct(double df, double nc, double t, bool upper) {
    if (t < 0.0) return nct(df, -nc, -t, !upper);  // T(nc) = -T(-nc)
    if (t == 0.0) return distr_detail::one<Ndtr>(upper ? nc : -nc);
    if (upper && nc < 0.0) return nct_opposite(df, -nc, t);
    // The series (t > 0), x = t^2 / (df + t^2), b = df/2, mu = nc^2/2:
    //   lower = Phi(-nc) + (1/2) sum_j [D(j, mu) I_x(j + 1/2, b) + sign(nc) D(j + 1/2, mu) I_x(j + 1, b)]
    //   upper = (1/2) sum_j [D(j, mu) (1 - I_x(j + 1/2, b)) + D(j + 1/2, mu) (1 - I_x(j + 1, b))]   (nc > 0)
    // (lower with nc < 0 is >= Phi(-nc) >= 1/2: its cancellation costs nothing relative.)
    const distr_detail::Arg g = distr_detail::t_arg(df, t);  // g.x = df / (df + t^2), g.y = t^2 / (df + t^2)
    if (g.x.hi < distr_detail::kTinyArg && nc > 0.0) {
        // df / (df + t^2) below 2^-1000: the far tail on the noncentrality's side from y^b
        const dd::DD logy = dd::log(df) - dd::log(t) * 2.0;
        const double mu = 0.5 * nc * nc;
        const double up = std::min(1.0, 0.5 * (tiny_upper(mu, 0.0, 0.5, 0.5 * df, logy) + tiny_upper(mu, 0.5, 1.0, 0.5 * df, logy)));
        return upper ? up : 1.0 - up;
    }
    if (g.x.hi < distr_detail::kTinyArg) return upper ? nct_opposite(df, -nc, t) : 1.0 - nct_opposite(df, -nc, t);
    // t^2 / (df + t^2) below 2^-1000: the step at t = 0 is all (the rest is O(t), t < 1e-150)
    if (g.y.hi < distr_detail::kTinyArg) return distr_detail::one<Ndtr>(upper ? nc : -nc);
    const dd::DD mu = dd::two_prod(nc, nc) * 0.5;
    const double b = 0.5 * df;
    const BetaFamily fam{b, g.y, g.x, upper};
    const double sp = mixture(mu.hi, 0.0, 0.5, fam, mu.lo);
    const double sq = mixture(mu.hi, 0.5, 1.0, fam, mu.lo);
    if (upper) return std::min(1.0, 0.5 * (sp + sq));
    return std::clamp(distr_detail::one<Ndtr>(-nc) + 0.5 * (sp + std::copysign(sq, nc)), 0.0, 1.0);
}


// ---- inverses
//
// The root of g = 0, g the log of the smaller tail minus the log of its target
// (1 - p is exact for p >= 1/2), so a target of 1e-300 is solved as easily as
// 0.3: bracketed from a start by growing steps, narrowed by bisection on a log
// scale (x > 0) or the asinh scale (t real) to a factor of two, then Brent's
// zeroin in x itself to the last bit.

// Brent's zeroin on [a, b] with g(a) < 0 < g(b) (finite), to |dx| <= 2 eps |x|.
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
        // Converging superlinearly: after a step below 3e-9 |b| the secant through the last
        // two points is exact to the last bit (its error ~ the product of the last two,
        // < 1e-17), so it ends here instead of closing the bracket (two evaluations fewer).
        if (std::fabs(b - a) <= 3e-9 * std::fabs(b) && fb != fa && std::isfinite(fb)) {
            const double x = b - fb * (b - a) / (fb - fa);
            if ((x - b) * (x - c) <= 0.0 || std::fabs(x - b) <= 1e-8 * std::fabs(b)) return x;
        }
    }
    return b;
}

// g increasing in x; `to_u` / `from_u` the bracketing scale (log, asinh), `step` its first step.
template <class G, class To, class From>
inline double solve(const G& g, double x0, const To& to_u, const From& from_u, double step, double lowest,
                    double highest) {
    double glo = g(x0);
    if (glo == 0.0) return x0;
    double lo = x0, hi = x0, ghi = glo;
    const bool up = glo < 0.0;
    double u = to_u(x0);
    // walk away from x0 until the sign changes; past the last finite value (the largest
    // double, the smallest subnormal) the root is beyond the double range
    const double top = highest > 0.0 ? std::numeric_limits<double>::max() : 0.0;
    const double bottom = lowest < 0.0 ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::denorm_min();
    for (int k = 0;; ++k) {
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
            if (ghi >= 0.0) break;
        } else {
            if (!(x > bottom)) x = bottom;
            if (x >= lo) {
                if (lo == bottom) return lowest;
                if (k < 64) step *= 2.0;
                continue;
            }
            hi = lo, ghi = glo, lo = x, glo = g(x);
            if (glo <= 0.0) break;
        }
        if (k < 8) step *= 2.0;
    }
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
        // x/2 below the normals: only the j = 0 term counts (the next carries a factor x/2)
        const double lower = std::exp(-0.5 * nc) * Chdtr::f(df, x);
        out = upper ? 1.0 - lower : lower;
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
        if (df == 0.0 && p <= std::exp(-0.5 * nc)) return 0.0;  // inside the point mass at 0
        return nc_detail::invert_positive([&](double x) { return Chndtr::f(df, nc, x); },
                                          [&](double x) { return Chndtrc::f(df, nc, x); }, p,
                                          nc_detail::ncx2_start(df, nc, p));
    }
};

// ---------------------------------------------------------- Student t

// nctdtr(df, nc, t) = P(T <= t), T = (Z + nc) / sqrt(V / df). df = inf is
// the normal distribution shifted by nc.
inline bool nct_edge(double df, double nc, double t, bool upper, double& out) {
    if (std::isnan(df) || std::isnan(nc) || std::isnan(t) || !(df > 0.0)) {
        out = kNaN;
    } else if (nc == 0.0) {
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
        if (nc == 0.0) return Stdtrit::f(df, p);
        if (p == 0.0) return -kInf;
        if (p == 1.0) return kInf;
        if (std::isinf(nc)) return nc;
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
        if (std::isinf(dfd)) return Chndtrix::f(dfn, nc, p) / dfn;
        if (std::isinf(dfn)) return Fdtri::f(dfn, dfd, p);
        // start: dfn F ~ c chi-square(k') over chi-square(dfd) / dfd (Patnaik), a central F
        const double c = (dfn + 2.0 * nc) / (dfn + nc), k = (dfn + nc) / c;
        const double x0 = c * k / dfn * Fdtri::f(k, dfd, p);
        return nc_detail::invert_positive([&](double x) { return Ncfdtr::f(dfn, dfd, nc, x); },
                                          [&](double x) { return Ncfdtrc::f(dfn, dfd, nc, x); }, p, x0);
    }
};

}  // namespace sd::special
