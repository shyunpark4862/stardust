// Student's t (df) and the F distribution (dfn, dfd): incomplete betas at the
// arguments sd.special forms in double-double from the exact inputs
// (distr_detail::t_arg, f_arg):
//     t:  x = df / (df + t^2),            I_x(df/2, 1/2) = 2 P(T <= -|t|)
//     F:  w = dfn x / (dfn x + dfd),      I_w(dfn/2, dfd/2) = P(F <= x)
// The densities are the prefix at those arguments (stats/dist/beta.hpp) over
// |t| and x; the cdfs are sd.special's stdtr, fdtr, fdtrc; the deep log tails
// and the deep inverses go through beta_detail as for the beta distribution.
#pragma once

#include "special/ops/distr.hpp"
#include "special/ops/gamma.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/lifetime.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace t_detail {

using num::DD;
namespace dd = num::dd;

// L(a) = log Gamma(a + 1/2) - log Gamma(a) - log(a)/2 for a >= 12, without the large
// terms: with u = 1/(2a), Stirling's forms give
//     L = (a log1p(u) - 1/2) + S(a + 1/2) - S(a),
// a log1p(u) - 1/2 = (2 a u - 1)/2 - a mu(u) (mu(u) = u - log1p(u), special/dd.hpp)
// and each power difference of S as a^(1-2k) expm1((1-2k) log1p(u)).
inline double big_l(double a) {
    const double u = 0.5 / a;
    const DD mu = dd::log1pmx_neg(DD{u, 0.0});
    const double l1 = std::log1p(u);
    double ds = 0.0, p = 1.0 / a;  // a^(1-2k)
    for (int k = 1; k <= 8; ++k) {
        ds += num::kBernoulli2k[k - 1] / (2.0 * k * (2.0 * k - 1.0)) * p * std::expm1((1.0 - 2.0 * k) * l1);
        p /= a * a;
    }
    const DD r = DD{0.5 * std::fma(2.0 * a, u, -1.0), 0.0} - mu * a;
    return r.hi + r.lo + ds;
}

inline constexpr double kBig = 12.0;  // from here the asymptotic forms

// log B(a, 1/2) as a double-double: log sqrt(pi) - L(a) - log(a)/2 for a >= 12, else
// the double-double log-gammas.
inline DD log_beta_half(double a) {
    if (a >= kBig) return num::kLogPi * 0.5 - num::log_dd(a) * 0.5 - big_l(a);
    return num::log_gamma(DD{a, 0.0}) + num::kLogPi * 0.5 - num::log_gamma(dd::two_sum(a, 0.5));
}

// log(1 + t^2/df) as a double-double, any finite t.
inline DD log1p_t2(double t, double df) {
    const double at = std::fabs(t);
    if (at == 0.0) return {0.0, 0.0};  // (ilogb(0) is INT_MIN: the difference below overflowed for df >= 4)
    if (std::ilogb(at) > 500 || std::ilogb(at) - std::ilogb(df) / 2 > 480) {
        // t^2 dwarfs df: 2 log|t| - log df + log1p(df / t^2)
        const double r = (df / at) / at;
        return num::log_dd(at) * 2.0 - num::log_dd(df) + std::log1p(r);
    }
    const DD v = dd::two_prod(t, t) / df;
    if (v.hi < 0.5) return v - dd::log1pmx_neg(v);  // log1p(v) = v - mu(v)
    return dd::log(v + 1.0);
}

}  // namespace t_detail

struct StudentT {
    static constexpr char name[] = "studentt";
    static constexpr std::array<const char*, 1> params{"df"};

    static bool valid(double df) { return df > 0.0 && std::isfinite(df); }
    // df = inf: the standard normal, every method (as special.stdtr and MultivariateT)
    static bool at_limit(double df) { return df == kInf; }
    template <class M>
    static double limit_in(double x, double) { return method_of<Normal, M>(x, 0.0, 1.0); }
    template <class M>
    static double limit_par(double) { return method_of<Normal, M>(0.0, 1.0); }

    // log density: log f(0) - (a + 1/2) log(1 + t^2/df), log f(0) = -(log df / 2 + log B(a, 1/2)).
    static num::DD log_density(double t, double df) {
        const double a = 0.5 * df;
        thread_local num::Memo memo;
        const num::DD l0 = memo.get(df, 0.0, [&] { return -(num::log_dd(df) * 0.5 + t_detail::log_beta_half(a)); });
        return l0 - t_detail::log1p_t2(t, df) * num::dd::two_sum(a, 0.5);
    }
    // The double-double definitions (the kernel's fallback: the tails, huge df).
    static double pdf_dd(double t, double df) {
        if (std::isinf(t)) return 0.0;
        return num::exp_dd(log_density(t, df));
    }
    static double logpdf_dd(double t, double df) {
        if (std::isinf(t)) return -kInf;
        const num::DD l = log_density(t, df);
        return l.hi + l.lo;
    }
    // The chunk kernel of pdf and logpdf (dens::kernel, stats/dist/gamma.hpp):
    //     l = l0 - (a + 1/2) log(1 + v),   v = t^2 / df
    // with 1 + v a double-double from exact pieces (two_prod, the division's remainder) and
    // l0 the scalar definition's double-double.
    static void density_chunk_raw(const Chunk<1>& c, double* out, bool log) {
        const double* df = c.p[0];
        thread_local dens::Const cst;
        dens::kernel<false>(
            c.n, log, out,
            [&](int i) {
                dens::In in = dens::kBlank;
                const double a = 0.5 * df[i];
                in.c = cst.get(df[i], 0.0, [&] {
                    if (!(df[i] > 0.0 && df[i] <= gamma_detail::kPlainMax)) return num::DD{kNaN, 0.0};
                    return -(num::log_dd(df[i]) * 0.5 + t_detail::log_beta_half(a));
                });
                const num::DD q = num::dd::two_prod(c.x[i], c.x[i]);
                const double vh = q.hi / df[i];
                const double vl = (std::fma(-vh, df[i], q.hi) + q.lo) / df[i];
                const num::DD s = num::dd::two_sum(1.0, vh);
                in.x1 = num::dd::quick_two_sum(s.hi, s.lo + vl);
                in.k1 = -num::dd::two_sum(a, 0.5);
                // t^2 exact to the last bit (not subnormal) or zero, and v in range
                in.ok = (q.hi >= 0x1p-960 || q.hi == 0.0) && vh <= 0x1p1000 && in.c.hi == in.c.hi;
                return in;
            },
            [&](int i) {
                if (!(c.x[i] == c.x[i] && valid(df[i]))) return kNaN;
                return log ? logpdf_dd(c.x[i], df[i]) : pdf_dd(c.x[i], df[i]);
            });
    }
    // A df below 2^-1021 has no exact half (the beta and gamma shapes are df / 2): NaN, as the
    // cdfs. Checked once for broadcast parameters.
    static void density_chunk(const Chunk<1>& c, double* out, bool log) {
        density_chunk_raw(c, out, log);
        if (c.same[0]) {
            if (!(c.p[0][0] < 0x1p-1021)) return;
            for (int i = 0; i < c.n; ++i) out[i] = kNaN;
            return;
        }
        for (int i = 0; i < c.n; ++i) out[i] = (c.p[0][i] < 0x1p-1021) ? kNaN : out[i];
    }
    static void pdf_chunk(const Chunk<1>& c, double* out) { density_chunk(c, out, false); }
    static void logpdf_chunk(const Chunk<1>& c, double* out) { density_chunk(c, out, true); }
    // The scalar forms: the kernel on one element (every caller gets the same bits).
    static double density_one(double t, double df, bool log) {
        double out;
        density_chunk(Chunk<1>{1, &t, {&df}, {true}}, &out, log);
        return out;
    }
    static double pdf(double t, double df) { return density_one(t, df, false); }
    static double logpdf(double t, double df) { return density_one(t, df, true); }
    // The beta shape is df / 2: below 2^-1021 a df has no exact half (the smallest subnormals),
    // and the methods through the beta functions are NaN.
    static bool halvable(double df) { return df >= 0x1p-1021; }
    // Where the beta argument x = df / (df + t^2) is below the doubles' normal range (a huge
    // |t|, or a tiny df) stdtr's value is 0 or 1, though the tail may be anything up to 1/2
    // (df = 1e-50, t = -1e300: 0.5): there the smaller side from its log (logcdf's leading
    // term).
    // (For |t| <= 2^400 and df >= 2^-160 the quotient is at least 2^-960: two comparisons
    // answer the usual call, without the divisions.)
    static bool far(double t, double df) {
        if (std::fabs(t) <= 0x1p400 && df >= 0x1p-160) [[likely]] return false;
        return df / t / t < 0x1p-960;
    }
    static double cdf(double t, double df) {
        if (!halvable(df)) return kNaN;
        if (std::isfinite(t) && far(t, df)) {
            const double v = num::exp_dd(far_log_lower(-std::fabs(t), df));
            return t < 0.0 ? v : 1.0 - v;
        }
        return special::Stdtr::f(df, t);
    }
    static double ccdf(double t, double df) { return cdf(-t, df); }
    // log P(T <= t) for t < 0 with x = 1 / (1 + t^2 / df) far below the range:
    // I_x(a, 1/2) / 2 = x^a / (2 a B(a, 1/2)) (the next term is O(x)).
    static num::DD far_log_lower(double t, double df) {
        const double a = 0.5 * df;
        const num::DD lx = -t_detail::log1p_t2(t, df);
        return lx * a - num::log_dd(a) - t_detail::log_beta_half(a) - num::kLn2;
    }
    // log P(T <= t): the small side (t < 0) from its value, from the beta pieces below the
    // normal range, and from the leading term (I_x(a, 1/2) = x^a / (a B), x = 1/(1 + t^2/df))
    // when x itself is below 1e-270.
    static double logcdf(double t, double df) {
        if (!halvable(df)) return kNaN;
        if (std::isinf(t)) return t > 0.0 ? 0.0 : -kInf;
        if (t > 0.0 && far(t, df)) return std::log1p(-num::exp_dd(far_log_lower(-t, df)));
        if (t >= 0.0) return std::log1p(-special::Stdtr::f(df, -t));
        const double v = special::Stdtr::f(df, t);
        if (v >= beta_detail::kDeep) return std::log(v);
        const double a = 0.5 * df;
        const num::DD lx = -t_detail::log1p_t2(t, df);
        if (lx.hi < -620.0) {
            const num::DD l = lx * a - num::log_dd(a) - t_detail::log_beta_half(a) - num::kLn2;
            return l.hi + l.lo;
        }
        const special::distr_detail::Arg g = special::distr_detail::t_arg(df, t);
        return beta_detail::log_lower_deep(a, 0.5, g.x, g.y) - num::kLn2.hi;
    }
    static double logccdf(double t, double df) { return logcdf(-t, df); }

    // The t <= 0 with P(T <= t) = p (p <= 1/2, lt = log p): stdtrit, or for a tiny p (or a
    // tiny x = df/(df + t^2)) I_x(a, 1/2) = 2p solved on the log side, t = -sqrt(df (1-x)/x).
    static double lower_t(double p, num::DD lt, double df) {
        const double t = lower_t_raw(p, lt, df);
        // outside the measured df (tests/stats/ref: 1e-2 to 1e6) a root is returned only if
        // the forward log tail reproduces the target (num::accept_root)
        if ((df >= 1e-3 && df <= 1e6) || t != t) return t;
        if (!(t <= 0.0)) return kNaN;
        if (std::isinf(t)) {  // a root beyond the doubles: the tail at the last one is still above the target
            return logcdf(-std::numeric_limits<double>::max(), df) >= lt.hi - 1e-9 * std::max(1.0, std::fabs(lt.hi)) ? t : kNaN;
        }
        if (t == 0.0) return kNaN;  // (p < 1/2 here)
        const num::DD ld = log_density(t, df);
        return num::accept_root(t, logcdf(t, df), ld.hi + std::log(-t), lt.hi);
    }
    static double lower_t_raw(double p, num::DD lt, double df) {
        const double a = 0.5 * df;
        if (p >= beta_detail::kDeepP) {
            thread_local dens::Const cst;  // log(2 a B(a, 1/2)), once per df
            const double k = cst.get(df, 0.0, [&] {
                return num::DD{num::kLn2.hi + std::log(a) + special::Betaln::f(a, 0.5), 0.0};
            }).hi;
            const double est = (lt.hi + k) / a;
            if (est > -40.0) return special::Stdtrit::f(df, p);
        }
        // log x and log(1 - x) both from the solve (near x = 1, the centre, refined in
        // log(1 - x) itself: beta_detail::deep_log_inverse_both)
        const beta_detail::LogPair r = beta_detail::deep_log_inverse_both(a, 0.5, lt + num::kLn2);
        return -num::exp_dd((num::log_dd(df) - r.lx + r.ly) * 0.5);
    }
    static double icdf(double p, double df) {
        if (!halvable(df)) return kNaN;
        if (!(p > 0.0 && p < 1.0) || p == 0.5) return special::Stdtrit::f(df, p);
        if (p > 0.5) return -lower_t(1.0 - p, num::log_dd(1.0 - p), df);
        return lower_t(p, num::log_dd(p), df);
    }
    static double iccdf(double p, double df) { return -icdf(p, df); }
    static double ilogcdf(double lp, double df) {
        if (!(lp <= 0.0) || !halvable(df)) return kNaN;
        if (lp == 0.0) return kInf;
        if (lp == -kInf) return -kInf;
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return -lower_t(q, num::log_dd(q), df);
        }
        return lower_t(std::exp(lp), num::DD{lp, 0.0}, df);
    }
    static double ilogccdf(double lp, double df) { return -ilogcdf(lp, df); }

    static double mean(double df) { return df > 1.0 ? 0.0 : kNaN; }
    static double variance(double df) { return df > 2.0 ? df / (df - 2.0) : (df > 1.0 ? kInf : kNaN); }
    static double standard_deviation(double df) {
        return df > 2.0 ? std::sqrt(df / (df - 2.0)) : (df > 1.0 ? kInf : kNaN);
    }
    static double skewness(double df) { return df > 3.0 ? 0.0 : kNaN; }
    static double kurtosis(double df) { return df > 4.0 ? 6.0 / (df - 4.0) : (df > 2.0 ? kInf : kNaN); }
    // (a df below 2^-1021 has no exact half: NaN as the pdf, the cdfs and the entropy)
    static double median(double df) { return df < 0x1p-1021 ? kNaN : 0.0; }
    static double mode(double) { return 0.0; }
    // (a + 1/2)(psi(a + 1/2) - psi(a)) + log B(a, 1/2) + log(df)/2, a = df/2; from a = 12
    // = (a + 1/2) dpsi + log(2 pi)/2 - L(a) (t_detail::big_l: the log-gammas' large terms
    // cancelled by hand, the digamma difference by its series).
    static double entropy(double df) {
        if (!halvable(df)) return kNaN;  // (2^-1022 <= df < 2^-1021 gave 4.5e307)
        const double a = 0.5 * df;
        if (a >= t_detail::kBig) {
            const num::DD h = num::kLogSqrt2Pi + ((a + 0.5) * num::digamma_diff_asym(a, 0.5) - t_detail::big_l(a));
            return h.hi + h.lo;
        }
        const double dpsi = special::Digamma::f(a + 0.5) - special::Digamma::f(a);
        const num::DD h = t_detail::log_beta_half(a) + num::log_dd(df) * 0.5 + (a + 0.5) * dpsi;
        return h.hi + h.lo;
    }
    static double support_lo(double) { return -kInf; }
    static double support_hi(double) { return kInf; }
};

struct F {
    static constexpr char name[] = "f";
    static constexpr std::array<const char*, 2> params{"dfn", "dfd"};

    static bool valid(double dfn, double dfd) {
        return dfn > 0.0 && std::isfinite(dfn) && dfd > 0.0 && std::isfinite(dfd);
    }
    // One infinite degree of freedom (the other finite and valid), as special.fdtr:
    // dfd = inf is chi-square(dfn) / dfn, dfn = inf is dfd / chi-square(dfd), the inverse gamma
    // (dfd/2, dfd/2). Both infinite (a point mass at 1) stays NaN.
    static bool at_limit(double dfn, double dfd) {
        return (dfd == kInf && dfn > 0.0 && std::isfinite(dfn)) || (dfn == kInf && dfd > 0.0 && std::isfinite(dfd));
    }
    template <class M>
    static double limit_in(double x, double dfn, double dfd) {
        if (dfd == kInf) return scaled_of<ChiSquared, M>(dfn, x, dfn);
        return method_of<InverseGamma, M>(x, 0.5 * dfd, 0.5 * dfd);
    }
    template <class M>
    static double limit_par(double dfn, double dfd) {
        if (dfd == kInf) return scaled_of<ChiSquared, M>(dfn, dfn);
        return method_of<InverseGamma, M>(0.5 * dfd, 0.5 * dfd);
    }

    // log B(p, q) where q may be far beyond p: log Gamma(p) - p log q once (p / q) max(1, p) is
    // below 2^-56 (the next term is p (p - 1) / (2 q)); the log-gammas while both are below
    // 1e15 (their cancellation leaves 1e-32 of q log q); NaN where neither applies.
    static num::DD log_beta_far(double p, double q) {
        if (p <= q * 0x1p-56 && p * p <= q * 0x1p-56) return num::log_gamma(num::DD{1.0, 0.0} + p) - num::log_dd(p) - num::log_dd(q) * p;
        if (p <= 1e15 && q <= 1e15) return beta_detail::log_beta(p, q);
        return {kNaN, 0.0};
    }
    static num::DD log_density(double x, double dfn, double dfd) {
        const special::distr_detail::Arg g = special::distr_detail::f_arg(dfn, dfd, x);
        const bool tiny_w = g.x.hi < special::distr_detail::kTinyArg, tiny_v = g.y.hi < special::distr_detail::kTinyArg;
        if (tiny_w || tiny_v) {
            // The beta argument w = u / (1 + u), u = dfn x / dfd, or v = 1 - w is below the
            // normal range (a rounded subnormal, or 0): the density from the logs, with
            // log(1 + u) = u for a tiny u and log u + 1/u for a huge one:
            //   tiny u:  a log u - (a + b) u   - log x - log B(a, b)
            //   huge u: -b log u - (a + b) / u - log x - log B(a, b)
            const double a = 0.5 * dfn, b = 0.5 * dfd;
            const num::DD lx = num::log_dd(x);
            const num::DD lu = num::log_dd(dfn) + lx - num::log_dd(dfd);
            const num::DD ls = num::log_dd(a + b);
            const num::DD lb = tiny_w ? (b >= a ? log_beta_far(a, b) : log_beta_far(b, a)) : (a >= b ? log_beta_far(b, a) : log_beta_far(a, b));
            if (tiny_w) return lu * a - num::exp_dd(ls + lu) - lx - lb;
            return -(lu * b) - num::exp_dd(ls - lu) - lx - lb;
        }
        return beta_detail::log_prefix(0.5 * dfn, 0.5 * dfd, g.x, g.y) - lx_of(x);
    }
    static num::DD lx_of(double x) { return num::log_dd(x); }
    // The density at 0: dfn < 2 inf, dfn = 2 one, dfn > 2 zero.
    static double density0(double dfn) { return dfn < 2.0 ? kInf : (dfn == 2.0 ? 1.0 : 0.0); }
    // The double-double definitions (the kernel's fallback: the tails, huge shapes, the ends).
    static double pdf_dd(double x, double dfn, double dfd) {
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return density0(dfn);
        return num::exp_dd(log_density(x, dfn, dfd));
    }
    static double logpdf_dd(double x, double dfn, double dfd) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return std::log(density0(dfn));
        const num::DD l = log_density(x, dfn, dfd);
        return l.hi + l.lo;
    }
    // The chunk kernel of pdf and logpdf (dens::kernel, stats/dist/gamma.hpp). With a = dfn/2,
    // b = dfd/2, u = dfn x / dfd (the prefix w^a v^b / B at w = u / (1 + u), over x):
    //     l = (a - 1) log x - (a + b) log(1 + u) + a log(dfn / dfd) - log B(a, b),
    // 1 + u a double-double from exact pieces.
    static void density_chunk_raw(const Chunk<2>& c, double* out, bool log) {
        const double *dfn = c.p[0], *dfd = c.p[1];
        thread_local dens::Const cst;
        dens::kernel<true>(
            c.n, log, out,
            [&](int i) {
                dens::In in = dens::kBlank;
                const double a = 0.5 * dfn[i], b = 0.5 * dfd[i];
                in.c = cst.get(dfn[i], dfd[i], [&] {
                    if (!valid(dfn[i], dfd[i])) return num::DD{kNaN, 0.0};
                    return (num::log_dd(dfn[i]) - num::log_dd(dfd[i])) * a - beta_detail::log_beta_dd(a, b);
                });
                const num::DD p = num::dd::two_prod(dfn[i], c.x[i]);
                const double uh = p.hi / dfd[i];
                const double ul = (std::fma(-uh, dfd[i], p.hi) + p.lo) / dfd[i];
                const num::DD s = num::dd::two_sum(1.0, uh);
                in.k1 = num::dd::two_sum(a, -1.0), in.x1 = {c.x[i], 0.0};
                in.k2 = -num::dd::two_sum(a, b), in.x2 = num::dd::quick_two_sum(s.hi, s.lo + ul);
                // false for a NaN, x <= 0, inf; dfn x exact to the last bit
                in.ok = c.x[i] >= 0x1p-1000 && p.hi >= 0x1p-960 && uh <= 0x1p1000 && in.c.hi == in.c.hi;
                return in;
            },
            [&](int i) {
                if (!(c.x[i] == c.x[i] && valid(dfn[i], dfd[i]))) return kNaN;
                return log ? logpdf_dd(c.x[i], dfn[i], dfd[i]) : pdf_dd(c.x[i], dfn[i], dfd[i]);
            });
    }
    // A df below 2^-1021 has no exact half (the beta and gamma shapes are df / 2): NaN, as the
    // cdfs. Checked once for broadcast parameters.
    static void density_chunk(const Chunk<2>& c, double* out, bool log) {
        density_chunk_raw(c, out, log);
        if (c.same[0] && c.same[1]) {
            if (!(c.p[0][0] < 0x1p-1021 || c.p[1][0] < 0x1p-1021)) return;
            for (int i = 0; i < c.n; ++i) out[i] = kNaN;
            return;
        }
        for (int i = 0; i < c.n; ++i) out[i] = (c.p[0][i] < 0x1p-1021 || c.p[1][i] < 0x1p-1021) ? kNaN : out[i];
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, false); }
    static void logpdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, true); }
    // The scalar forms: the kernel on one element (every caller gets the same bits).
    static double density_one(double x, double dfn, double dfd, bool log) {
        double out;
        density_chunk(Chunk<2>{1, &x, {&dfn, &dfd}, {true, true}}, &out, log);
        return out;
    }
    static double pdf(double x, double dfn, double dfd) { return density_one(x, dfn, dfd, false); }
    static double logpdf(double x, double dfn, double dfd) { return density_one(x, dfn, dfd, true); }
    // The beta shapes are dfn / 2 and dfd / 2: below 2^-1021 they are not exact halves, and the
    // methods through the beta functions are NaN.
    // Both past 2^101 (the beta shapes past 2^100: Beta::degenerate) are NaN too.
    static bool halvable(double dfn, double dfd) {
        return dfn >= 0x1p-1021 && dfd >= 0x1p-1021 && !(dfn > 0x1p101 && dfd > 0x1p101);
    }
    static double cdf(double x, double dfn, double dfd) {
        if (!halvable(dfn, dfd)) return kNaN;
        return x <= 0.0 ? 0.0 : special::Fdtr::f(dfn, dfd, x);
    }
    static double ccdf(double x, double dfn, double dfd) {
        if (!halvable(dfn, dfd)) return kNaN;
        return x <= 0.0 ? 1.0 : special::Fdtrc::f(dfn, dfd, x);
    }
    // For dfn, dfd <= 12 the smaller side's value is fdtr's or fdtrc's (Boost's betainc there,
    // as cdf and ccdf: special/ops/distr.hpp), the side x < 1 suggests first; its log, or log1p
    // of minus it. A deep tail and larger shapes: the double-double ibeta and the pieces.
    static double log_tail(double x, double dfn, double dfd, bool upper) {
        if (!halvable(dfn, dfd)) return kNaN;
        if (x <= 0.0) return upper ? 0.0 : -kInf;
        if (std::isinf(x)) return upper ? -kInf : 0.0;
        if (dfn <= 12.0 && dfd <= 12.0) {
            bool lower_small = x < 1.0;  // (the beta argument below its mean)
            double small = lower_small ? special::Fdtr::f(dfn, dfd, x) : special::Fdtrc::f(dfn, dfd, x);
            if (small > 0.5) {
                lower_small = !lower_small;
                small = lower_small ? special::Fdtr::f(dfn, dfd, x) : special::Fdtrc::f(dfn, dfd, x);
            }
            if (small >= beta_detail::kDeep) return upper != lower_small ? std::log(small) : std::log1p(-small);
        }
        const special::distr_detail::Arg g = special::distr_detail::f_arg(dfn, dfd, x);
        // A tiny beta argument (w ~ (dfn / dfd) x, or v = 1 - w ~ dfd / (dfn x)): from its log.
        // f_arg's double-double is a rounded subnormal there, or 0 (F(10, 3).logcdf(1e-320)
        // was 6.7e-8 off through it).
        const double a = 0.5 * dfn, b = 0.5 * dfd;
        // The other shape from 2^900 up: its product with the argument is not small, and the
        // distribution is its chi-square limit (special's gamma_limit; b w = a x, a v = b / x).
        if (g.x.hi < special::distr_detail::kTinyArg) {
            if (b < 0x1p900) return tiny_log_tail(a, b, num::log_dd(dfn) + num::log_dd(x) - num::log_dd(dfd), upper);
            return limit_log_tail(a, num::dd::two_prod(a, x), num::log_dd(a) + num::log_dd(x), upper);
        }
        if (g.y.hi < special::distr_detail::kTinyArg) {
            if (a < 0x1p900) return tiny_log_tail(b, a, num::log_dd(dfd) - num::log_dd(dfn) - num::log_dd(x), !upper);
            return limit_log_tail(b, num::DD{b, 0.0} / x, num::log_dd(b) - num::log_dd(x), !upper);
        }
        return beta_detail::log_tail(a, b, g.x, g.y, upper);
    }
    // log P(s, z) (upper: log Q(s, z)), z = hi + lo with the log lz: the gamma distribution's
    // own log tail while z is a normal number; below, the pair from special's gamma_limit and,
    // for a lower tail below the normal range, P = z^s / Gamma(s + 1) (1 + O(z)) in logs.
    static double limit_log_tail(double s, num::DD z, num::DD lz, bool upper) {
        if (z.hi >= 0x1p-1000) return gamma_detail::dist_log_tail(s, z, upper);
        const special::ibeta_detail::IB r = special::ibeta_detail::gamma_limit(s, z, lz);
        if (upper) return r.ic < r.i ? std::log(r.ic) : std::log1p(-r.i);
        if (r.i >= beta_detail::kDeep) return r.i <= r.ic ? std::log(r.i) : std::log1p(-r.ic);
        const num::DD l = lz * s - num::log_gamma(num::DD{s, 0.0} + 1.0);
        return l.hi + l.lo;
    }
    // log I_w(a, b) (upper: log(1 - I_w(a, b))) for w = e^lw below 2^-1000: the pair from
    // special's ibeta_tiny; a lower tail below the normal range from the leading term in
    // logs, I = w^a / (a B(a, b)) (the next term, of the order of b w, is nothing).
    static double tiny_log_tail(double a, double b, num::DD lw, bool upper) {
        const special::ibeta_detail::IB r = special::ibeta_detail::ibeta_tiny(a, b, lw, false);
        if (upper) return r.ic < r.i ? std::log(r.ic) : std::log1p(-r.i);
        if (r.i >= beta_detail::kDeep) return r.i <= r.ic ? std::log(r.i) : std::log1p(-r.ic);
        // (log B by its far form when one shape dwarfs the other: the log-gammas cancel)
        const num::DD l = lw * a - num::log_dd(a) - (a <= b ? log_beta_far(a, b) : log_beta_far(b, a));
        return l.hi + l.lo;
    }
    static double logcdf(double x, double dfn, double dfd) { return log_tail(x, dfn, dfd, false); }
    static double logccdf(double x, double dfn, double dfd) { return log_tail(x, dfn, dfd, true); }

    // x from w (the cdf's beta argument) and v = 1 - w: (dfd / dfn) w / v.
    static double from_wv(double w, double v, double dfn, double dfd) { return (w / v) * (dfd / dfn); }

    // x with P(F <= x) = t (lower) or P(F > x) = t, t the smaller side, lt = log t: fdtri
    // and the complement solve while t and the root are normal numbers; else the deep
    // inverse on the beta side, w (lower) or v = 1 - w (upper) tiny, x from logs.
    static double solve(double t, num::DD lt, double dfn, double dfd, bool lower) {
        const double x = solve_raw(t, lt, dfn, dfd, lower);
        // outside the measured shapes (tests/stats/ref: 0.1 to 1e5) a root is returned only if
        // the forward log tail reproduces the target (num::accept_root)
        if ((dfn >= 1e-3 && dfd >= 1e-3 && dfn <= 1e5 && dfd <= 1e5) || x != x) return x;
        if (!(x >= 0.0)) return kNaN;
        if (x == 0.0 || std::isinf(x)) {
            // a root beyond the doubles: the tail at the last double inside is already past the
            // target (the lower tail grows with x)
            const bool low = x == 0.0;
            const double lf = log_tail(low ? 0x1p-1074 : std::numeric_limits<double>::max(), dfn, dfd, !lower);
            const double slack = 1e-9 * std::max(1.0, std::fabs(lt.hi));
            return ((low == lower) ? lf >= lt.hi - slack : lf <= lt.hi + slack) ? x : kNaN;
        }
        const num::DD ld = log_density(x, dfn, dfd);
        return num::accept_root(x, log_tail(x, dfn, dfd, !lower), ld.hi + std::log(x), lt.hi);
    }
    static double solve_raw(double t, num::DD lt, double dfn, double dfd, bool lower) {
        const double a = 0.5 * dfn, b = 0.5 * dfd;
        if (beta_detail::fast_shapes(a, b) && t >= 0x1p-200) {
            // small shapes: the beta argument w (or v = 1 - w, whichever is small) by Halley's
            // method on Boost's betainc (beta_detail::fast_root), x = (dfd / dfn) w / v
            beta_detail::XY r;
            if (beta_detail::fast_root(a, b, t, !lower, r) && r.x >= 0x1p-60 && r.y >= 0x1p-60)
                return from_wv(r.x, r.y, dfn, dfd);
        }
        const double sa = lower ? a : b, sb = lower ? b : a;  // the beta whose argument is small
        if (t >= beta_detail::kDeepP) {
            const double est = (lt.hi + std::log(sa) + special::Betaln::f(sa, sb)) / sa;
            if (est > -40.0) return lower ? special::Fdtri::f(dfn, dfd, t) : upper_solve(t, dfn, dfd);
        }
        // log w or log v, and log(1 - it) (refined there near 1: deep_log_inverse_both)
        const beta_detail::LogPair pr = beta_detail::deep_log_inverse_both(sa, sb, lt);
        const num::DD lr = pr.lx, lo = pr.ly;
        const num::DD ratio = num::log_dd(dfd) - num::log_dd(dfn);
        return num::exp_dd(lower ? lr - lo + ratio : lo - lr + ratio);
    }
    // P(F > x) = I_v(b, a) = q (q <= 1/2, normal): the smaller of v and w, as fdtri does.
    static double upper_solve(double q, double dfn, double dfd) {
        const double a = 0.5 * dfn, b = 0.5 * dfd;
        if (q < special::Betainc::f(b, a, 0.5)) {  // v < 1/2
            const double v = special::Betaincinv::f(b, a, q);
            if (v < 0x1p-60)  // q = v^b / (b B) (1 + O(v)): x = (dfd/dfn) / v from logs
                return num::exp_dd(num::log_dd(dfd) - num::log_dd(dfn) -
                                   special::distr_detail::log_product(q, b * special::Beta::f(a, b)) / b);
            return from_wv(1.0 - v, v, dfn, dfd);
        }
        const double w = special::Betainccinv::f(a, b, q);  // I_w(a, b) = 1 - q
        return from_wv(w, 1.0 - w, dfn, dfd);
    }
    static double icdf(double p, double dfn, double dfd) {
        if (!halvable(dfn, dfd)) return kNaN;
        if (!(p > 0.0 && p < 1.0)) return special::Fdtri::f(dfn, dfd, p);
        if (p > 0.5) return solve(1.0 - p, num::log_dd(1.0 - p), dfn, dfd, false);
        return solve(p, num::log_dd(p), dfn, dfd, true);
    }
    static double iccdf(double q, double dfn, double dfd) {
        if (!(q >= 0.0 && q <= 1.0) || !halvable(dfn, dfd)) return kNaN;
        if (q == 0.0) return kInf;
        if (q == 1.0) return 0.0;
        if (q > 0.5) return solve(1.0 - q, num::log_dd(1.0 - q), dfn, dfd, true);
        return solve(q, num::log_dd(q), dfn, dfd, false);
    }
    static double from_log(double lp, double dfn, double dfd, bool upper) {
        if (!(lp <= 0.0) || !halvable(dfn, dfd)) return kNaN;
        if (lp == 0.0) return upper ? 0.0 : kInf;
        if (lp == -kInf) return upper ? kInf : 0.0;
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return solve(q, num::log_dd(q), dfn, dfd, upper);
        }
        return solve(std::exp(lp), num::DD{lp, 0.0}, dfn, dfd, !upper);
    }
    static double ilogcdf(double lp, double dfn, double dfd) { return from_log(lp, dfn, dfd, false); }
    static double ilogccdf(double lp, double dfn, double dfd) { return from_log(lp, dfn, dfd, true); }

    // The moments: finite past dfd = 2, 4, 6, 8 in turn. Below, the mean is +inf (X >= 0);
    // a higher ratio is +inf where the variance is finite and its own moment diverges,
    // NaN where the variance is not finite.
    static double mean(double, double dfd) { return dfd > 2.0 ? dfd / (dfd - 2.0) : kInf; }
    static double variance(double dfn, double dfd) {
        if (!(dfd > 4.0)) return dfd > 2.0 ? kInf : kNaN;
        const double r = dfd / (dfd - 2.0);  // 2 dfd^2 (dfn + dfd - 2) / (dfn (dfd - 2)^2 (dfd - 4))
        const double den = dfn * (dfd - 4.0);
        if (!(den >= 0x1p-900 && den <= 0x1p900)) return 2.0 * r * r * (((dfn + dfd - 2.0) / dfn) / (dfd - 4.0));  // the product leaves the range
        return 2.0 * r * r * ((dfn + dfd - 2.0) / den);
    }
    static double standard_deviation(double dfn, double dfd) {
        const double v = variance(dfn, dfd);
        if (v != v || (v >= 0x1p-900 && v <= 0x1p900)) return std::sqrt(v);
        if (!(dfd > 4.0)) return v;  // inf
        // the variance leaves the range where its root may not: by factors
        return 1.4142135623730951 * (dfd / (dfd - 2.0)) * (std::sqrt((dfn + dfd - 2.0) / (dfd - 4.0)) / std::sqrt(dfn));
    }
    // (2 dfn + dfd - 2) sqrt(8 (dfd - 4)) / ((dfd - 6) sqrt(dfn (dfn + dfd - 2)))
    static double skewness(double dfn, double dfd) {
        if (!(dfd > 6.0)) return dfd > 4.0 ? kInf : kNaN;
        const double prod = dfn * (dfn + dfd - 2.0);
        if (prod >= 0x1p-900 && prod <= 0x1p900 && dfd <= 0x1p900)
            return (2.0 * dfn + dfd - 2.0) * std::sqrt(8.0 * (dfd - 4.0)) / ((dfd - 6.0) * std::sqrt(prod));
        // A product or a sum leaves the range where the value does not (a subnormal dfn, a
        // huge one): by factors, the sums on copies scaled by k = 2^-16 where they could overflow,
        //     [(2 dfn + dfd - 2) / sqrt(dfn + dfd - 2)]  [sqrt(dfd - 4) / (dfd - 6)]  / sqrt(dfn)
        const bool big = dfn > 0x1p1000 || dfd > 0x1p1000;
        const double k = big ? 0x1p-16 : 1.0, n = dfn * k, d = dfd * k;
        const double lead = (2.0 * n + d - 2.0 * k) / std::sqrt(n + d - 2.0 * k) * (big ? 0x1p8 : 1.0);
        return 2.8284271247461903 * (lead * (std::sqrt(dfd - 4.0) / (dfd - 6.0))) / std::sqrt(dfn);
    }
    static double kurtosis(double dfn, double dfd) {
        // the fourth moment diverges for dfd <= 8; the variance is finite from dfd > 4 (inf, as
        // the t's kurtosis for 2 < df <= 4), NaN below
        if (!(dfd > 8.0)) return dfd > 4.0 ? kInf : kNaN;
        // 12 (dfn (5 dfd - 22)(dfn + dfd - 2) + (dfd - 4)(dfd - 2)^2) / (dfn (dfd - 6)(dfd - 8)(dfn + dfd - 2)),
        // as sums of ratios (no products of the parameters to overflow); every term is positive.
        if (dfd <= 0x1p900 && dfn <= 0x1p900 && dfn >= 0x1p-900) {
            const double s = dfn + dfd - 2.0;
            const double t1 = (5.0 * dfd - 22.0) / (dfd - 6.0) / (dfd - 8.0);
            const double t2 = ((dfd - 4.0) / (dfd - 6.0)) * ((dfd - 2.0) / (dfd - 8.0)) * ((dfd - 2.0) / dfn) / s;
            return 12.0 * (t1 + t2);
        }
        // Next to the ends of the doubles 5 dfd, dfn + dfd and (dfd - 2) / dfn leave the range
        // where the value does not: (dfd - 2) / (dfn + dfd - 2) = 1 / (1 + dfn / (dfd - 2)) first
        const double t1 = (5.0 - 22.0 / dfd) * (dfd / (dfd - 6.0)) / (dfd - 8.0);
        const double share = 1.0 / (1.0 + dfn / (dfd - 2.0));
        const double t2 = ((dfd - 4.0) / (dfd - 6.0)) * ((dfd - 2.0) / (dfd - 8.0)) * share / dfn;
        return 12.0 * (t1 + t2);
    }
    static double mode(double dfn, double dfd) { return dfn > 2.0 ? ((dfn - 2.0) / dfn) * (dfd / (dfd + 2.0)) : 0.0; }
    static double median(double dfn, double dfd) {
        // fdtri inside the measured shapes; outside, the checked inverse
        if (dfn >= 1e-3 && dfd >= 1e-3 && dfn <= 1e5 && dfd <= 1e5) return special::Fdtri::f(dfn, dfd, 0.5);
        return icdf(0.5, dfn, dfd);
    }
    // log(b/a) + log B(a, b) + (1 - a) psi(a) - (1 + b) psi(b) + (a + b) psi(a + b)
    // = G(a) + G(b) + Gc(c) - log(a)/2 + 3 (log b - log c)/2 + log(2 pi)/2 + 2 (psi(c) - psi(b)).
    static double entropy(double dfn, double dfd) {
        // below 1e-2 the digammas' poles (-2/dfn, -2/dfd, which cancel in the sum) take the
        // digits: 3.5e-8 off at 1e-10, half the value at 1e-300. NaN.
        if (!(dfn >= 1e-2 && dfd >= 1e-2)) return kNaN;
        const double a = 0.5 * dfn, b = 0.5 * dfd;
        const num::DD c = num::dd::two_sum(a, b);
        const double dpsi = b >= 12.0 ? num::digamma_diff_asym(b, a) : special::Digamma::f(c.hi) - special::Digamma::f(b);
        const num::DD logs = -num::log_dd(a) * 0.5 + (num::log_dd(b) - num::dd::log(c)) * 1.5 + num::kLogSqrt2Pi;
        const double g = num::entropy_g(a, special::Digamma::f) + num::entropy_g(b, special::Digamma::f) +
                         num::entropy_gc(c, special::Digamma::f) + 2.0 * dpsi;
        const num::DD h = logs + g;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
