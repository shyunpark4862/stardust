// The distribution functions of scipy.special built on the incomplete gamma
// and beta functions (docs/api/special.md §13; T1): Student t (stdtr,
// stdtrit), chi-square (chdtr, chdtrc, chdtri), F (fdtr, fdtrc, fdtri),
// Poisson (pdtr, pdtrc), binomial (bdtr, bdtrc), negative binomial (nbdtr,
// nbdtrc), and ndtri_exp.
//
// Each is a composition, arranged so that the composition loses no accuracy:
//   * the arguments of betainc (x = df / (df + t^2) and its complement
//     t^2 / (df + t^2)) are formed in double-double from the exact inputs and
//     rounded once;
//   * the smaller of a probability and its complement is computed directly
//     (betaincc with the complement argument, never 1 - betainc), so the
//     tails keep their relative accuracy;
//   * 1 - p is taken only where it is exact (p >= 1/2, Sterbenz).
// Special values follow the true value or limit (T1 decision), with the
// differences from scipy in docs/design/special.md §2.4.

#pragma once

#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/ops/erf.hpp"
#include "special/ops/gamma.hpp"
#include "special/ops/incbeta.hpp"
#include "special/ops/incgamma.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace distr_detail {

// A chunk-only Op (ndtr, ndtri, log_ndtr) on one value.
template <typename Op>
inline double one(double x) {
    double o;
    Op::chunk(&x, &o, 1);
    return o;
}

// exp(e) for e = hi + lo: one rounding (the result's relative error ~1 ULP
// whatever |e|; exp(hi) alone would carry hi's rounding times |e|). Past the
// doubles exp(hi) is 0 or inf as it stands: lo is then ulp(hi)/2, thousands
// or more, and 1 + lo could be negative and flip the sign of the result.
inline double exp_dd(dd::DD e) {
    const double e0 = std::exp(e.hi);
    if (e0 == 0.0 || std::isinf(e0)) return e0;
    return e0 * (1.0 + e.lo);
}

// s B(s, c) = Gamma(s + 1) Gamma(c) / Gamma(s + c) for s, c > 0. As a product it is lost
// where B(s, c) overflows: a shape below about 5.6e-309 (subnormal; s = 0 for the half of the
// smallest double). B(s, c) = (s + c) / (s c) (1 + O(s + c)) there for the smaller one, so the
// product is 1 + s / c (stdtr(1e-310, 1) was 1 for 0.5).
inline double shape_beta(double s, double c) {
    const double B = Beta::f(s, c);
    if (std::isfinite(B)) return s * B;
    return 1.0 + s / c;
}

// log(v c) as hi + lo for v > 0 (subnormal too: scaled by 2^64 first) and c > 0 normal.
inline dd::DD log_product(double v, double c) {
    if (v >= 0x1p-960) return dd::log(dd::two_prod(v, c));
    // (the logs apart: the product itself underflows for a small c -- fdtri(3, 1e16, 5e-324)
    // was 14 for 2.3e-216)
    return dd::log_pos(v) + dd::log_pos(c);
}

// An argument of the incomplete beta function and its complement, both
// double-double: x + y = 1 to ~2^-100. Passed to ibeta_detail::ibeta as they
// are, never rounded: with a large parameter, I_x is sensitive to x (a
// condition number ~ a x), and rounding x to double cost up to 300 ULP in the
// tails (T1 measurement). The incomplete-beta core takes the pair directly.
struct Arg {
    dd::DD x, y;
};

// Below this an argument is treated by the small-argument series instead:
// x^a / (a B(a, b)) (1 + O(x)), the O(x) far below the last bit.
inline constexpr double kTinyArg = 0x1p-1000;

// x = df / (df + t^2), y = t^2 / (df + t^2), t != 0 finite, df > 0 finite.
inline Arg t_arg(double df, double t) {
    const double at = std::fabs(t);
    if (std::ilogb(at) > 500 || df > 1e300) {
        if (std::ilogb(df) + 64 < 2 * std::ilogb(at)) {
            // t^2 dwarfs df: x = df / t^2 to the last bit (the next term, x^2, is below it).
            const double x = (df / at) / at;
            return {{x, 0.0}, dd::two_sum(1.0, -x)};
        }
        // Both large: the same ratios of copies scaled by 2^-512 and 2^-1024.
        return t_arg(std::ldexp(df, -1024), std::ldexp(at, -512));
    }
    const dd::DD z = dd::two_prod(at, at);
    const dd::DD s = z + df;
    return {dd::DD{df, 0.0} / s, z / s};
}

// P(T <= -|t|) for Student's t (df > 0 finite, t != 0 finite).
inline double t_tail(double df, double t) {
    const Arg g = t_arg(df, t);
    const double a = 0.5 * df;
    if (g.y.hi == 0.0) return 0.5;  // t^2 below df's last bit
    if (g.x.hi < kTinyArg) {
        // I_x(a, 1/2) ~ x^a / (a B(a, 1/2)), x^a = (sqrt(df) / |t|)^df: x itself underflows
        // (stdtr(1, -1e200) = 3.2e-201 while x = 1e-400).
        // (the ratio itself underflows for a tiny df with a huge t, where the power is near 1:
        // from the logs there -- stdtr(1e-300, -1.7e308) was 0 for 0.5)
        const double r = std::sqrt(df) / std::fabs(t);
        // (the exponent as hi + lo: it reaches -700, and its rounding would enter that many times)
        const double xa =
            r >= 0x1p-1000 ? std::pow(r, df) : exp_dd((dd::log_pos(df) * 0.5 - dd::log_pos(std::fabs(t))) * df);
        return 0.5 * xa / shape_beta(a, 0.5);
    }
    if (a <= 6.0) {
        // Small a: the condition number of I_x in x is at most a x <= 6, so rounding
        // x (or y) costs <= 3 ULP, and Boost's betainc (special/boost.hpp, S10) is
        // the faster path. The smaller argument goes in.
        return 0.5 * (g.x.hi <= g.y.hi ? Betainc::f(a, 0.5, g.x.hi) : Betaincc::f(0.5, a, g.y.hi));
    }
    return 0.5 * ibeta_detail::ibeta(a, 0.5, g.x, g.y).i;
}

// The F argument w = dfn x / (dfn x + dfd) and v = dfd / (dfn x + dfd) = 1 - w.
inline Arg f_arg(double dfn, double dfd, double x) {
    // (near the largest doubles dfn x + dfd overflows at a moderate x: w is the same for both
    // scaled by 2^-64 -- fdtrc(1.8e308, 1.8e308, 1e-10) was 0 for 1)
    if (dfn > 0x1p960 || dfd > 0x1p960) dfn *= 0x1p-64, dfd *= 0x1p-64;
    // (and next to the smallest doubles dfn x underflows while w does not: both scaled up
    // -- fdtr(1e-10, 1e-300, 5e-324) was 3e-8 off, its w = 5e-34 taken for 0)
    if (dfn * x < 0x1p-900) {
        const double m = std::max(dfn, dfd), sc = m < 0x1p-100 ? 0x1p1000 : (m < 0x1p400 ? 0x1p600 : 1.0);
        dfn *= sc, dfd *= sc;
    }
    const dd::DD u = dd::two_prod(dfn, x);
    const dd::DD s = u + dfd;
    if (!std::isfinite(u.hi) || !std::isfinite(s.hi) || !std::isfinite(u.lo)) {
        // dfn x overflows: v = dfd / (dfn x), w = 1 - v.
        const double v = (dfd / x) / dfn;
        return {dd::two_sum(1.0, -v), {v, 0.0}};
    }
    return {u / s, dd::DD{dfd, 0.0} / s};
}

// fdtr (upper: fdtrc) for finite positive dfn, dfd and 0 < x < inf.
// (a, b the shapes: dfn / 2, dfd / 2, or what stands for one of them, f_value below)
inline double f_value_shapes(double a, double b, double dfn, double dfd, double x, bool upper) {
    const Arg g = f_arg(dfn, dfd, x);
    // A tiny argument, which may itself be below the doubles (w = 0 at x = 1e-320,
    // dfn / dfd = 1e-5, where fdtr is 0.99 for dfn = 2e-5): from its log, by
    // ibeta_detail::ibeta_tiny, which gives both sides without subtracting (1 - i from
    // i = w^a / (a B) was 1.0e-14 off at dfn = 2.7e-4, x = 3.7e-299, i = 0.91). The other
    // shape has to be below 2^900 for it; beyond, the F distribution is its chi-square
    // limit (ibeta_detail::gamma_limit: fdtr(2, 1e308, 1) = 1 - 1/e, where w = 1e-308).
    if (g.x.hi < kTinyArg) {  // I_w(a, b), w ~ (dfn / dfd) x; b w = a x
        const ibeta_detail::IB r =
            b < 0x1p900 ? ibeta_detail::ibeta_tiny(a, b, dd::log_pos(dfn) + dd::log_pos(x) - dd::log_pos(dfd), false)
                        : ibeta_detail::gamma_limit(a, dd::two_prod(a, x), dd::log_pos(a) + dd::log_pos(x));
        return upper ? r.ic : r.i;
    }
    if (g.y.hi < kTinyArg) {  // 1 - I_w = I_v(b, a), v ~ dfd / (dfn x); a v = b / x
        const ibeta_detail::IB r =
            a < 0x1p900 ? ibeta_detail::ibeta_tiny(b, a, dd::log_pos(dfd) - dd::log_pos(dfn) - dd::log_pos(x), false)
                        : ibeta_detail::gamma_limit(b, dd::DD{b, 0.0} / x, dd::log_pos(b) - dd::log_pos(x));
        return upper ? r.i : r.ic;
    }
    if (a <= 6.0 && b <= 6.0) {  // as in t_tail: rounding costs <= 3 ULP, Boost is faster
        if (g.x.hi <= g.y.hi) return upper ? Betaincc::f(a, b, g.x.hi) : Betainc::f(a, b, g.x.hi);
        return upper ? Betainc::f(b, a, g.y.hi) : Betaincc::f(b, a, g.y.hi);
    }
    const ibeta_detail::IB r = ibeta_detail::ibeta(a, b, g.x, g.y);
    return upper ? r.ic : r.i;
}
inline double f_value(double dfn, double dfd, double x, bool upper) {
    const double a = 0.5 * dfn, b = 0.5 * dfd;
    if (a == 0.0 || b == 0.0) [[unlikely]] {
        // dfn or dfd is the smallest double: its half is not one (fdtr(1e-300, 5e-324, x) was
        // 0 for 4.9e-24). Both below 2^-699: I_w(a, b) = b / (a + b) = dfd / (dfn + dfd). Else
        // the side that vanishes with that shape is linear in it (1 - I_w(a, b) ~ a C for
        // a -> 0, I_w(a, b) ~ b C for b -> 0): half its value at the double, the other side 1.
        if (std::max(dfn, dfd) < 0x1p-699) return (upper ? dfn : dfd) / (dfn + dfd);
        if (a == 0.0) return upper ? 0.5 * f_value_shapes(dfn, b, dfn, dfd, x, true) : 1.0;
        return upper ? 1.0 : 0.5 * f_value_shapes(a, dfd, dfn, dfd, x, false);
    }
    return f_value_shapes(a, b, dfn, dfd, x, upper);
}

// |t| with P(|T| > |t|) = q2 for df >= 1: Hill's start (CACM algorithm 396, ~6 digits)
// and Halley steps on the tail itself, F(t) = t_tail(df, -t), F' = -f,
// F''/F' = -(df + 1) t / (df + t^2). A step below 4e-6 |t| leaves ~1e-17 (cubic), so
// most inputs take one tail evaluation; betaincinv, the general way, takes four or five
// (1100 -> 300 ns). False if it does not settle (the caller's general way).
inline bool t_quantile(double df, double q2, double& out) {
    double t;
    if (df == 1.0) {
        t = 1.0 / std::tan(0.5 * kPi * q2);
    } else if (df == 2.0) {
        t = std::sqrt(2.0 / (q2 * (2.0 - q2)) - 2.0);
    } else {
        const double a = 1.0 / (df - 0.5), b = 48.0 / (a * a);
        double c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
        const double d = ((94.5 / (b + c) - 3.0) / b + 1.0) * std::sqrt(a * 0.5 * kPi) * df;
        double y = std::pow(d * q2, 2.0 / df);
        if (y > 0.05 + a) {
            const double x = one<Ndtri>(0.5 * q2);
            y = x * x;
            if (df < 5.0) c += 0.3 * (df - 4.5) * (x + 0.6);
            c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c;
            y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
            y = std::expm1(a * y * y);
        } else {
            y = ((1.0 / (((df + 6.0) / (df * y) - 0.089 * d - 0.822) * (df + 2.0) * 3.0) + 0.5 / (df + 4.0)) * y -
                 1.0) *
                    (df + 1.0) / (df + 2.0) +
                1.0 / y;
        }
        t = std::sqrt(df * y);
    }
    if (!(t > 0.0) || !std::isfinite(t)) return false;
    const double q = 0.5 * q2;
    const double norm = 1.0 / (std::sqrt(df) * Beta::f(0.5 * df, 0.5));  // the density at 0
    for (int it = 0; it < 8; ++it) {
        const double tail = t_tail(df, t);
        const double w = df / (df + t * t);  // (1 + t^2/df)^-1
        const double dens = norm * std::pow(w, 0.5 * (df + 1.0));
        if (!(dens > 0.0) || !(tail > 0.0)) return false;
        const double r = (tail - q) / -dens;  // Newton's step
        const double step = r / (1.0 + 0.5 * r * (df + 1.0) * t / (df + t * t));
        const double tn = t - step;
        if (!(tn > 0.0) || !std::isfinite(tn)) return false;
        const bool done = std::fabs(step) <= 4e-6 * tn;
        t = tn;
        if (done) {
            out = t;
            return true;
        }
    }
    return false;
}

}  // namespace distr_detail

// ---------------------------------------------------------------- Student t

// stdtr(df, t) = P(T <= t). df = inf is the normal distribution.
struct Stdtr : Real {
    static constexpr const char* name = "stdtr";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"df", "t"};
    static constexpr double cost_ns = cost::kBetainc;
    static double f(double df, double t) {
        if (std::isnan(df) || std::isnan(t) || !(df > 0.0)) return kNaN;
        if (std::isinf(t)) return t > 0 ? 1.0 : 0.0;
        if (std::isinf(df)) return distr_detail::one<Ndtr>(t);
        if (t == 0.0) return 0.5;
        // df past 2^54 (1 + t^2)^2: the normal limit to 2^-56 (the tail's argument df /
        // (df + t^2) rounds to 1 there: stdtr(1.8e308, 1e-10) was 0.5)
        if (df > 0x1p54 * (1.0 + t * t) * (1.0 + t * t)) return distr_detail::one<Ndtr>(t);
        const double tail = distr_detail::t_tail(df, t);
        return t < 0 ? tail : 1.0 - tail;
    }
};

// stdtrit(df, p): the t with stdtr(df, t) = p.
struct Stdtrit : Real {
    static constexpr const char* name = "stdtrit";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"df", "p"};
    static constexpr double cost_ns = cost::kBetainc + cost::kBetaincinv;  // the branch test, then the inverse
    static double f(double df, double p) {
        if (std::isnan(df) || std::isnan(p) || !(df > 0.0) || p < 0.0 || p > 1.0) return kNaN;
        if (p == 0.0) return -kInf;
        if (p == 1.0) return kInf;
        if (p == 0.5) return 0.0;
        if (std::isinf(df)) return distr_detail::one<Ndtri>(p);
        const bool lower = p < 0.5;
        // |t| = sqrt(df) (q2 a B(a, 1/2))^(-1/df) (the far tail below) is past the doubles for
        // every p != 1/2 when df < 2^-1000: -log(q2 a B) >= 2^-54 against 1/df > 2^1000. (For
        // df = 5e-324, df / 2 is not a double: betaincinv's shape would be 0.)
        if (df < 0x1p-1000) return lower ? -kInf : kInf;
        const double q2 = 2.0 * (lower ? p : 1.0 - p);  // the two-sided tail, exact
        if (df >= 1.0 && q2 > 1e-12) {  // (farther out the tail and density lose t^2 against df: the general way)
            double t;
            if (distr_detail::t_quantile(df, q2, t)) return lower ? -t : t;
        }
        const double a = 0.5 * df;
        // Solve for the smaller of x = df / (df + t^2) and y = 1 - x: x < 1/2
        // exactly when q2 = I_x(a, 1/2) is below I_{1/2}(a, 1/2) (choosing by
        // q2 < 1/2 instead lost 1 - y's digits for small df: 4e-9 at df = 0.06).
        double t;
        if (q2 < Betainc::f(a, 0.5, 0.5)) {  // a far tail: x is small
            const double x = Betaincinv::f(a, 0.5, q2);
            if (x < 0x1p-60) {
                // The series q2 = x^a / (a B(a, 1/2)) (1 + O(x)) is exact to the last bit
                // here, and t = sqrt(df / x) = sqrt(df) (q2 a B)^(-1/(2a)) needs no root
                // (near the subnormals betaincinv's x had lost digits: 168 ULP).
                // log t = log(df)/2 - log(q2 a B) / df (2a = df exactly), in double-double: a
                // pow with the rounded exponent -1/df cost |log(q2 a B)| / df ULP (191 ULP).
                const dd::DD logt =
                    dd::log(df) * 0.5 - distr_detail::log_product(q2, distr_detail::shape_beta(a, 0.5)) / df;
                t = distr_detail::exp_dd(logt);
            } else {
                t = std::sqrt(df) * (std::sqrt(1.0 - x) / std::sqrt(x));  // sqrt(df (1 - x) / x), no overflow
            }
        } else {  // nearer the centre: y is small, from the complement
            const double y = Betainccinv::f(0.5, a, q2);
            t = std::sqrt(df) * std::sqrt(y / (1.0 - y));
        }
        return lower ? -t : t;
    }
};

// ---------------------------------------------------------------- chi-square

// chdtr(df, x) = P(df/2, x/2); chdtrc its complement; chdtri inverts chdtrc
// (scipy's convention). Halving is exact but for subnormal x, where x/2 may
// round (or vanish: chdtr(0.18, 5e-324) was 0 for 8e-30, T2): there
// P(a, y) = y^a / Gamma(a + 1) to the last bit (the series' next term is y), with
// log y = log(x/2) in double-double.
// v = 0 is the point mass at 0: 1 for every x > 0, subnormal too (x/2 = 0 made it
// gammainc(0, 0) = NaN).
namespace distr_detail {
inline bool chi2_subnormal(double v, double x) { return x > 0.0 && x < 0x1p-1021 && v >= 0.0 && std::isfinite(v); }
inline double chi2_tiny(double v, double x) {
    if (v == 0.0) return 1.0;
    const double a = 0.5 * v;
    const dd::DD e = log_product(x, 0.5) * a;
    if (!(e.hi > -1e4)) return 0.0;  // (a huge v: the product is -inf, or NaN as a double-double)
    return exp_dd(e) * Rgamma::f(a + 1.0);
}
// The complement there, 1 - y^a / Gamma(a + 1) = -expm1(a log y - log Gamma(a + 1)): for a
// small a the lower value is within a rounding of 1 (chdtrc(1e-100, 5e-324) was 0 for
// 3.7e-98). Below the normal doubles a = v / 2 would round: the exponent is v (log y +
// euler) / 2 there (log Gamma(1 + a) = -euler a).
inline double chi2_tiny_upper(double v, double x) {
    if (v == 0.0) return 0.0;
    const double a = 0.5 * v, ly = log_product(x, 0.5).hi;
    const double e = a < 0x1p-1000 ? v * (0.5 * (ly + kEuler)) : a * ly - igam_detail::lgamma1p(a);
    if (a < 1.0 && e > -0.5) return -std::expm1(e);
    return 1.0 - chi2_tiny(v, x);
}
}  // namespace distr_detail

struct Chdtr : Real {
    static constexpr const char* name = "chdtr";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"df", "x"};
    static constexpr double cost_ns = cost::kGammainc;
    static double f(double df, double x) {
        if (distr_detail::chi2_subnormal(df, x)) return distr_detail::chi2_tiny(df, x);
        // (no point mass at 0 for df > 0: chdtr(5e-324, 0) was gammainc(0, 0) = NaN, df / 2 = 0)
        if (x == 0.0 && df > 0.0) return 0.0;
        return Gammainc::f(0.5 * df, 0.5 * x);
    }
};

struct Chdtrc : Real {
    static constexpr const char* name = "chdtrc";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"df", "x"};
    static constexpr double cost_ns = cost::kGammaincc;
    static double f(double df, double x) {
        if (distr_detail::chi2_subnormal(df, x)) return distr_detail::chi2_tiny_upper(df, x);
        // df subnormal with an odd last bit: its half is not a double, and Q(a, x) is linear in
        // a -> 0 (to a relative O(a)): half of Q(df, x/2) (chdtrc(5e-324, 1e-300) was 0 for
        // 1.7e-321; chdtrc(1.5e-323, 1e-300) 6.8e-321 for 5.2e-321, the half rounded up, 2026-11-19)
        // At x = 0 or inf (or a NaN or negative x) the value is that of any positive shape, df's
        // own (chdtrc(5e-324, 0) was 0.5 for 1).
        if (df > 0.0 && df < 0x1p-1021 && 2.0 * (0.5 * df) != df) [[unlikely]]
            return x > 0.0 && std::isfinite(x) ? 0.5 * Gammaincc::f(df, 0.5 * x) : Gammaincc::f(df, x);
        return Gammaincc::f(0.5 * df, 0.5 * x);
    }
};

struct Chdtri : Real {
    static constexpr const char* name = "chdtri";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"df", "q"};
    static constexpr double cost_ns = cost::kGammainccinv;
    static double f(double df, double q) {
        // df subnormal with an odd last bit, as in chdtrc: Q(df/2, y) = Q(df, y) / 2 to a relative
        // O(df), so the root of Q(df/2, y) = q is that of Q(df, y) = 2 q. For q > 1/2 it is below
        // every double: Q(df, y) ~ df E1(y) there (chdtri(5e-324, q) was NaN, df / 2 = 0).
        if (df > 0.0 && df < 0x1p-1021 && 2.0 * (0.5 * df) != df && q >= 0.0 && q <= 1.0) [[unlikely]]
            return q <= 0.5 ? 2.0 * Gammainccinv::f(df, 2.0 * q) : 0.0;
        return 2.0 * Gammainccinv::f(0.5 * df, q);
    }
};

// ------------------------------------------------------------------------- F

// fdtr(dfn, dfd, x) = I_w(dfn/2, dfd/2), w = dfn x / (dfn x + dfd). An infinite
// dfd is the limit dfn F -> chi-square(dfn); an infinite dfn, dfd / F ->
// chi-square(dfd); both, the point mass at 1 (fdtr(inf, inf, 1) = 1).
inline bool f_edge(double dfn, double dfd, double x, bool upper, double& out) {
    if (std::isnan(dfn) || std::isnan(dfd) || std::isnan(x) || !(dfn > 0.0) || !(dfd > 0.0) || x < 0.0) {
        out = kNaN;
    } else if (x == 0.0) {
        out = upper ? 1.0 : 0.0;
    } else if (std::isinf(x)) {
        out = upper ? 0.0 : 1.0;
    } else if (std::isinf(dfn) && std::isinf(dfd)) {
        out = (x < 1.0) != upper ? 0.0 : 1.0;
    } else if (std::isinf(dfd)) {
        out = upper ? Gammaincc::f(0.5 * dfn, 0.5 * dfn * x) : Gammainc::f(0.5 * dfn, 0.5 * dfn * x);
    } else if (std::isinf(dfn)) {
        out = upper ? Gammainc::f(0.5 * dfd, 0.5 * dfd / x) : Gammaincc::f(0.5 * dfd, 0.5 * dfd / x);
    } else {
        return false;
    }
    return true;
}

struct Fdtr : Real {
    static constexpr const char* name = "fdtr";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"dfn", "dfd", "x"};
    static constexpr double cost_ns = cost::kBetainc;
    static double f(double dfn, double dfd, double x) {
        double out;
        if (f_edge(dfn, dfd, x, false, out)) return out;
        return distr_detail::f_value(dfn, dfd, x, false);
    }
};

struct Fdtrc : Real {
    static constexpr const char* name = "fdtrc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"dfn", "dfd", "x"};
    static constexpr double cost_ns = cost::kBetaincc;
    static double f(double dfn, double dfd, double x) {
        double out;
        if (f_edge(dfn, dfd, x, true, out)) return out;
        return distr_detail::f_value(dfn, dfd, x, true);
    }
};

// fdtri(dfn, dfd, p): the x with fdtr(dfn, dfd, x) = p.
struct Fdtri : Real {
    static constexpr const char* name = "fdtri";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"dfn", "dfd", "p"};
    static constexpr double cost_ns = cost::kBetainc + cost::kBetaincinv;
    static double f(double dfn, double dfd, double p) {
        if (std::isnan(dfn) || std::isnan(dfd) || std::isnan(p) || !(dfn > 0.0) || !(dfd > 0.0) || p < 0.0 || p > 1.0)
            return kNaN;
        if (p == 0.0) return 0.0;
        if (p == 1.0) return kInf;
        if (std::isinf(dfn) && std::isinf(dfd)) return 1.0;
        if (std::isinf(dfd)) return 2.0 * Gammaincinv::f(0.5 * dfn, p) / dfn;
        if (std::isinf(dfn)) return dfd / (2.0 * Gammainccinv::f(0.5 * dfd, p));
        const double a = 0.5 * dfn, b = 0.5 * dfd;
        if (a == 0.0 || b == 0.0) [[unlikely]] {
            // dfn or dfd is the smallest double, its half not one (f_value's cases; fdtri(5e-324,
            // 1, 0.5) was NaN). Both below 2^-699: fdtr is dfd / (dfn + dfd) at every 0 < x < inf,
            // point masses at 0 and inf (betaincinv's both-small root w / (1 - w) = 1 at the
            // tie). A vanishing a: fdtr = 1 - a C(x) is above every p < 1, so x = 0. A vanishing
            // b: fdtr = b C(x), linear in b, so fdtr with b = dfd / 2 equals p where the one with
            // b = dfd equals 2 p (for p > 1/2 the root is past the doubles).
            if (std::max(dfn, dfd) < 0x1p-699) {
                const double m = dfd / (dfn + dfd);
                return p < m ? 0.0 : (p > m ? kInf : dfd / dfn);
            }
            if (a == 0.0) return 0.0;
            if (p > 0.5) return kInf;
            const double v = Betainccinv::f(dfd, a, 2.0 * p);
            return ((1.0 - v) / v) * (dfd / dfn);
        }
        // One shape far beyond the other (the incomplete beta's gamma limit, ibeta_detail::
        // limit_shapes): w = z / b with P(a, z) = p, or v = z / a with Q(b, z) = p, and x =
        // (b/a) w / (1 - w) = (b/a) (1 - v) / v without the quotient z / b itself (subnormal
        // for b near the largest doubles).
        if (ibeta_detail::limit_shapes(a, b)) {
            const double z = Gammaincinv::f(a, p);
            return (z / a) / (1.0 - z / b);
        }
        if (ibeta_detail::limit_shapes(b, a)) {
            const double z = Gammainccinv::f(b, p);
            return (b / z) * (1.0 - z / a);
        }
        // Solve for the smaller of w and v = 1 - w (choosing by p <= 1/2 lost
        // all of a tiny w: fdtri(0.019, 262, 0.70) gave 0 for 1.2e-14).
        if (p < Betainc::f(a, b, 0.5)) {
            const double w = Betaincinv::f(a, b, p);
            // p = w^a / (a B) (1 + O((b + 1) w)): x = (dfd/dfn) w from logs in double-double
            if (w < 0x1p-60 && w * b < 0x1p-56) {
                const dd::DD l = distr_detail::log_product(p, distr_detail::shape_beta(a, b));
                // (past 2^1000 the quotient by a small a is below every double, and as a
                // double-double it would be NaN: fdtri(1e-310, 1, 0.5) was NaN for 0)
                if (!(l.hi / a > -0x1p1000)) return 0.0;
                return distr_detail::exp_dd(l / a + dd::log(dfd) - dd::log(dfn));
            }
            return (w / (1.0 - w)) * (dfd / dfn);
        }
        const double v = Betainccinv::f(b, a, p);  // 1 - I_v(b, a) = p, i.e. I_w(a, b) = p
        // 1 - p = v^b / (b B) (1 + O((a + 1) v)), 1 - p exact here: x = (dfd/dfn) / v
        if (v < 0x1p-60 && v * a < 0x1p-56 && p >= 0.5) {
            const dd::DD l = distr_detail::log_product(1.0 - p, distr_detail::shape_beta(b, a));
            if (!(l.hi / b > -0x1p1000)) return kInf;  // (as above: fdtri(1, 1e-310, 0.5))
            return distr_detail::exp_dd(dd::log(dfd) - dd::log(dfn) - l / b);
        }
        return ((1.0 - v) / v) * (dfd / dfn);
    }
};

// ------------------------------------------------------------------- Poisson

// pdtr(k, m) = P(X <= k), X ~ Poisson(m): Q(floor(k) + 1, m). k < 0 gives 0
// (pdtrc: 1), the true value (scipy: NaN).
inline bool count_edge(double& k, bool upper, double& out) {
    if (std::isnan(k)) {
        out = kNaN;
        return true;
    }
    k = std::floor(k);
    if (k < 0.0) {
        out = upper ? 1.0 : 0.0;
        return true;
    }
    return false;
}

struct Pdtr : Real {
    static constexpr const char* name = "pdtr";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"k", "m"};
    static constexpr double cost_ns = cost::kGammaincc;
    static double f(double k, double m) {
        double out;
        if (std::isnan(m) || m < 0.0) return kNaN;
        if (count_edge(k, false, out)) return out;
        return Gammaincc::f(k + 1.0, m);
    }
};

struct Pdtrc : Real {
    static constexpr const char* name = "pdtrc";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"k", "m"};
    static constexpr double cost_ns = cost::kGammainc;
    static double f(double k, double m) {
        double out;
        if (std::isnan(m) || m < 0.0) return kNaN;
        if (count_edge(k, true, out)) return out;
        return Gammainc::f(k + 1.0, m);
    }
};

// ------------------------------------------------------------------ binomial

// bdtr(k, n, p) = P(X <= k), X ~ Binomial(n, p): 1 - I_p(k + 1, n - k), as
// betaincc at p itself (never 1 - p). n must be a non-negative integer
// (scipy floors it); k >= n gives 1.
inline bool binom_edge(double& k, double n, double p, bool upper, double& out) {
    if (std::isnan(n) || std::isnan(p) || !(n >= 0.0) || std::isinf(n) || n != std::floor(n) || p < 0.0 || p > 1.0) {
        out = kNaN;
        return true;
    }
    if (count_edge(k, upper, out)) return true;
    if (k >= n) {
        out = upper ? 0.0 : 1.0;
        return true;
    }
    return false;
}

struct Bdtr : Real {
    static constexpr const char* name = "bdtr";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"k", "n", "p"};
    static constexpr double cost_ns = cost::kBetaincc;
    static double f(double k, double n, double p) {
        double out;
        if (binom_edge(k, n, p, false, out)) return out;
        return Betaincc::f(k + 1.0, n - k, p);
    }
};

struct Bdtrc : Real {
    static constexpr const char* name = "bdtrc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"k", "n", "p"};
    static constexpr double cost_ns = cost::kBetainc;
    static double f(double k, double n, double p) {
        double out;
        if (binom_edge(k, n, p, true, out)) return out;
        return Betainc::f(k + 1.0, n - k, p);
    }
};

// --------------------------------------------------------- negative binomial

// nbdtr(k, n, p) = P(X <= k), X the failures before the n-th success with
// success probability p: I_p(n, k + 1). n is any real > 0 (scipy floors it);
// betainc's own rules give n = 0 (no failures: 1) and n = inf (0).
struct Nbdtr : Real {
    static constexpr const char* name = "nbdtr";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"k", "n", "p"};
    static constexpr double cost_ns = cost::kBetainc;
    static double f(double k, double n, double p) {
        double out;
        if (std::isnan(n) || std::isnan(p) || n < 0.0 || p < 0.0 || p > 1.0) return kNaN;
        if (count_edge(k, false, out)) return out;
        return Betainc::f(n, k + 1.0, p);
    }
};

struct Nbdtrc : Real {
    static constexpr const char* name = "nbdtrc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"k", "n", "p"};
    static constexpr double cost_ns = cost::kBetaincc;
    static double f(double k, double n, double p) {
        double out;
        if (std::isnan(n) || std::isnan(p) || n < 0.0 || p < 0.0 || p > 1.0) return kNaN;
        if (count_edge(k, true, out)) return out;
        return Betaincc::f(n, k + 1.0, p);
    }
};

// ------------------------------------------------------------------ ndtri_exp

// ndtri_exp(y) = ndtri(exp(y)), for y <= 0:
//   y > log(1/2):  -ndtri(-expm1(y))  (1 - p from expm1, without cancellation)
//   y > -700:      ndtri(exp(y))      (exp(y) normal; x is insensitive to p's rounding there)
//   below:         Newton on log_ndtr(x) = y from the asymptote
//                  x^2 = -2y - log(2 pi) - 2 log(-x)
struct NdtriExp : Real {
    static constexpr const char* name = "ndtri_exp";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"y"};
    static constexpr double cost_ns = cost::kNdtri + 12;
    static double f(double y) {
        using distr_detail::one;
        if (std::isnan(y) || y > 0.0) return kNaN;
        if (y == 0.0) return kInf;
        if (std::isinf(y)) return -kInf;
        constexpr double kLogHalf = -0.69314718055994530942;
        if (y > kLogHalf) return -one<Ndtri>(-std::expm1(y));
        if (y > -700.0) return one<Ndtri>(std::exp(y));
        constexpr double kLog2Pi = 1.83787706640934548356;
        constexpr double kSqrt2 = 1.41421356237309504880;
        // The asymptote log ndtr(x) = -x^2/2 - log(-x) - log(2 pi)/2 + log S,
        // S = 1 - 1/x^2 + 3/x^4 - ..., as a fixed point for x^2.
        // Past y = -2^1000, -2y would overflow (from y = -8.99e307): the same fixed point for
        // x^2 / 2, x = -sqrt(2) sqrt(-y - log(2 pi)/2 - log(-x) + log S).
        const bool huge = y < -0x1p1000;
        auto asymptote = [&](double x) {
            const double r = 1.0 / (x * x);
            if (huge) return -kSqrt2 * std::sqrt(-y - 0.5 * kLog2Pi - std::log(-x) + std::log1p(r * (-1.0 + 3.0 * r)));
            return -std::sqrt(-2.0 * y - kLog2Pi - 2.0 * std::log(-x) + 2.0 * std::log1p(r * (-1.0 + 3.0 * r)));
        };
        double x = huge ? -kSqrt2 * std::sqrt(-y - 0.5 * kLog2Pi) : -std::sqrt(-2.0 * y - kLog2Pi);
        for (int i = 0; i < 5; ++i) x = asymptote(x);
        // For y < -1e8 (|x| > 1.4e4) the series' next term (15/x^6) is far below
        // the last bit, so the fixed point is the answer. Newton's slope
        // phi/ndtr subtracts numbers of size x^2/2: past |x| ~ 1e6 that
        // cancellation ruins it (y = -1e17 gave 2e13 ULP).
        if (y < -1e8) return x;
        for (int i = 0; i < 8; ++i) {
            const double l = one<LogNdtr>(x);
            // d/dx log ndtr(x) = phi(x) / ndtr(x) = exp(-x^2/2 - log(2 pi)/2 - log ndtr(x))
            const double slope = std::exp(-(0.5 * x) * x - 0.5 * kLog2Pi - l);
            const double step = (l - y) / slope;
            x -= step;
            if (std::fabs(step) <= 1e-17 * std::fabs(x)) break;
        }
        return x;
    }
};

}  // namespace sd::special
