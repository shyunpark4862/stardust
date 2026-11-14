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
// whatever |e|; exp(hi) alone would carry hi's rounding times |e|).
inline double exp_dd(dd::DD e) { return std::exp(e.hi) * (1.0 + e.lo); }

// log(v c) as hi + lo for v > 0 (subnormal too: scaled by 2^64 first) and c > 0 normal.
inline dd::DD log_product(double v, double c) {
    if (v >= 0x1p-960) return dd::log(dd::two_prod(v, c));
    return dd::log(dd::two_prod(std::ldexp(v, 64), c)) - dd::kLn2 * 64.0;
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
        return 0.5 * std::pow(std::sqrt(df) / std::fabs(t), df) / (a * Beta::f(a, 0.5));
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
inline double f_value(double dfn, double dfd, double x, bool upper) {
    const Arg g = f_arg(dfn, dfd, x);
    const double a = 0.5 * dfn, b = 0.5 * dfd;
    if (g.x.hi < kTinyArg) {  // I_w ~ w^a / (a B), w ~ (dfn / dfd) x
        const double i = std::pow(dfn / dfd, a) * std::pow(x, a) / (a * Beta::f(a, b));
        return upper ? 1.0 - i : i;
    }
    if (g.y.hi < kTinyArg) {  // 1 - I_w = I_v(b, a) ~ v^b / (b B), v ~ dfd / (dfn x)
        const double ic = std::pow(dfd / dfn, b) * std::pow(x, -b) / (b * Beta::f(a, b));
        return upper ? ic : 1.0 - ic;
    }
    if (a <= 6.0 && b <= 6.0) {  // as in t_tail: rounding costs <= 3 ULP, Boost is faster
        if (g.x.hi <= g.y.hi) return upper ? Betaincc::f(a, b, g.x.hi) : Betainc::f(a, b, g.x.hi);
        return upper ? Betainc::f(b, a, g.y.hi) : Betaincc::f(b, a, g.y.hi);
    }
    const ibeta_detail::IB r = ibeta_detail::ibeta(a, b, g.x, g.y);
    return upper ? r.ic : r.i;
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
        const double q2 = 2.0 * (lower ? p : 1.0 - p);  // the two-sided tail, exact
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
                const dd::DD logt = dd::log(df) * 0.5 - distr_detail::log_product(q2, a * Beta::f(a, 0.5)) / df;
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

// chdtr(v, x) = P(v/2, x/2); chdtrc its complement; chdtri inverts chdtrc
// (scipy's convention). Halving is exact but for subnormal x, where x/2 may
// round (or vanish: chdtr(0.18, 5e-324) was 0 for 8e-30, T2): there
// P(a, y) = y^a / Gamma(a + 1) to the last bit (the series' next term is y), with
// log y = log(x/2) in double-double.
namespace distr_detail {
inline bool chi2_subnormal(double v, double x) { return x > 0.0 && x < 0x1p-1021 && v > 0.0 && std::isfinite(v); }
inline double chi2_tiny(double v, double x) {
    const double a = 0.5 * v;
    return exp_dd(log_product(x, 0.5) * a) * Rgamma::f(a + 1.0);
}
}  // namespace distr_detail

struct Chdtr : Real {
    static constexpr const char* name = "chdtr";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"v", "x"};
    static constexpr double cost_ns = cost::kGammainc;
    static double f(double v, double x) {
        if (distr_detail::chi2_subnormal(v, x)) return distr_detail::chi2_tiny(v, x);
        return Gammainc::f(0.5 * v, 0.5 * x);
    }
};

struct Chdtrc : Real {
    static constexpr const char* name = "chdtrc";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"v", "x"};
    static constexpr double cost_ns = cost::kGammaincc;
    static double f(double v, double x) {
        if (distr_detail::chi2_subnormal(v, x)) return 1.0 - distr_detail::chi2_tiny(v, x);
        return Gammaincc::f(0.5 * v, 0.5 * x);
    }
};

struct Chdtri : Real {
    static constexpr const char* name = "chdtri";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"v", "p"};
    static constexpr double cost_ns = cost::kGammainccinv;
    static double f(double v, double p) { return 2.0 * Gammainccinv::f(0.5 * v, p); }
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
        // Solve for the smaller of w and v = 1 - w (choosing by p <= 1/2 lost
        // all of a tiny w: fdtri(0.019, 262, 0.70) gave 0 for 1.2e-14).
        if (p < Betainc::f(a, b, 0.5)) {
            const double w = Betaincinv::f(a, b, p);
            if (w < 0x1p-60)  // p = w^a / (a B) (1 + O(w)): x = (dfd/dfn) w from logs in double-double
                return distr_detail::exp_dd(distr_detail::log_product(p, a * Beta::f(a, b)) / a + dd::log(dfd) -
                                            dd::log(dfn));
            return (w / (1.0 - w)) * (dfd / dfn);
        }
        const double v = Betainccinv::f(b, a, p);  // 1 - I_v(b, a) = p, i.e. I_w(a, b) = p
        if (v < 0x1p-60 && p >= 0.5)  // 1 - p = v^b / (b B) (1 + O(v)), 1 - p exact here: x = (dfd/dfn) / v
            return distr_detail::exp_dd(dd::log(dfd) - dd::log(dfn) -
                                        distr_detail::log_product(1.0 - p, b * Beta::f(a, b)) / b);
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
        // The asymptote log ndtr(x) = -x^2/2 - log(-x) - log(2 pi)/2 + log S,
        // S = 1 - 1/x^2 + 3/x^4 - ..., as a fixed point for x^2.
        auto asymptote = [&](double x) {
            const double r = 1.0 / (x * x);
            return -std::sqrt(-2.0 * y - kLog2Pi - 2.0 * std::log(-x) + 2.0 * std::log1p(r * (-1.0 + 3.0 * r)));
        };
        double x = -std::sqrt(-2.0 * y - kLog2Pi);
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
