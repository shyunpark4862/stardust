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

    // log density: log f(0) - (a + 1/2) log(1 + t^2/df), log f(0) = -(log df / 2 + log B(a, 1/2)).
    static num::DD log_density(double t, double df) {
        const double a = 0.5 * df;
        thread_local num::Memo memo;
        const num::DD l0 = memo.get(df, 0.0, [&] { return -(num::log_dd(df) * 0.5 + t_detail::log_beta_half(a)); });
        return l0 - t_detail::log1p_t2(t, df) * num::dd::two_sum(a, 0.5);
    }
    static double pdf(double t, double df) {
        if (std::isinf(t)) return 0.0;
        return num::exp_dd(log_density(t, df));
    }
    static double logpdf(double t, double df) {
        if (std::isinf(t)) return -kInf;
        const num::DD l = log_density(t, df);
        return l.hi + l.lo;
    }
    static double cdf(double t, double df) { return special::Stdtr::f(df, t); }
    static double ccdf(double t, double df) { return special::Stdtr::f(df, -t); }
    // log P(T <= t): the small side (t < 0) from its value, from the beta pieces below the
    // normal range, and from the leading term (I_x(a, 1/2) = x^a / (a B), x = 1/(1 + t^2/df))
    // when x itself is below 1e-270.
    static double logcdf(double t, double df) {
        if (std::isinf(t)) return t > 0.0 ? 0.0 : -kInf;
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
        const double a = 0.5 * df;
        if (p >= beta_detail::kDeepP) {
            const double est = (lt.hi + num::kLn2.hi + std::log(a) + special::Betaln::f(a, 0.5)) / a;
            if (est > -40.0) return special::Stdtrit::f(df, p);
        }
        const num::DD lx = beta_detail::deep_log_inverse(a, 0.5, lt + num::kLn2);
        const double x = std::exp(lx.hi);
        // log(1 - x), 1 - x = -expm1(log x): x rounded and subtracted from 1 lost digits near
        // the centre (x ~ 1)
        const num::DD ly = x < 0x1p-60 ? num::DD{-x, 0.0} : num::log_dd(-std::expm1(lx.hi));
        return -num::exp_dd((num::log_dd(df) - lx + ly) * 0.5);
    }
    static double icdf(double p, double df) {
        if (!(p > 0.0 && p < 1.0) || p == 0.5) return special::Stdtrit::f(df, p);
        if (p > 0.5) return -lower_t(1.0 - p, num::log_dd(1.0 - p), df);
        return lower_t(p, num::log_dd(p), df);
    }
    static double iccdf(double p, double df) { return -icdf(p, df); }
    static double ilogcdf(double lp, double df) {
        if (!(lp <= 0.0)) return kNaN;
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
    static double median(double) { return 0.0; }
    static double mode(double) { return 0.0; }
    // (a + 1/2)(psi(a + 1/2) - psi(a)) + log B(a, 1/2) + log(df)/2, a = df/2; from a = 12
    // = (a + 1/2) dpsi + log(2 pi)/2 - L(a) (t_detail::big_l: the log-gammas' large terms
    // cancelled by hand, the digamma difference by its series).
    static double entropy(double df) {
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

    static num::DD log_density(double x, double dfn, double dfd) {
        const special::distr_detail::Arg g = special::distr_detail::f_arg(dfn, dfd, x);
        return beta_detail::log_prefix(0.5 * dfn, 0.5 * dfd, g.x, g.y) - num::log_dd(x);
    }
    // The density at 0: dfn < 2 inf, dfn = 2 one, dfn > 2 zero.
    static double density0(double dfn) { return dfn < 2.0 ? kInf : (dfn == 2.0 ? 1.0 : 0.0); }
    static double pdf(double x, double dfn, double dfd) {
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return density0(dfn);
        return num::exp_dd(log_density(x, dfn, dfd));
    }
    static double logpdf(double x, double dfn, double dfd) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return std::log(density0(dfn));
        const num::DD l = log_density(x, dfn, dfd);
        return l.hi + l.lo;
    }
    static double cdf(double x, double dfn, double dfd) { return x <= 0.0 ? 0.0 : special::Fdtr::f(dfn, dfd, x); }
    static double ccdf(double x, double dfn, double dfd) { return x <= 0.0 ? 1.0 : special::Fdtrc::f(dfn, dfd, x); }
    static double log_tail(double x, double dfn, double dfd, bool upper) {
        if (x <= 0.0) return upper ? 0.0 : -kInf;
        if (std::isinf(x)) return upper ? -kInf : 0.0;
        const special::distr_detail::Arg g = special::distr_detail::f_arg(dfn, dfd, x);
        return beta_detail::log_tail(0.5 * dfn, 0.5 * dfd, g.x, g.y, upper);
    }
    static double logcdf(double x, double dfn, double dfd) { return log_tail(x, dfn, dfd, false); }
    static double logccdf(double x, double dfn, double dfd) { return log_tail(x, dfn, dfd, true); }

    // x from w (the cdf's beta argument) and v = 1 - w: (dfd / dfn) w / v.
    static double from_wv(double w, double v, double dfn, double dfd) { return (w / v) * (dfd / dfn); }

    // x with P(F <= x) = t (lower) or P(F > x) = t, t the smaller side, lt = log t: fdtri
    // and the complement solve while t and the root are normal numbers; else the deep
    // inverse on the beta side, w (lower) or v = 1 - w (upper) tiny, x from logs.
    static double solve(double t, num::DD lt, double dfn, double dfd, bool lower) {
        const double a = 0.5 * dfn, b = 0.5 * dfd;
        const double sa = lower ? a : b, sb = lower ? b : a;  // the beta whose argument is small
        if (t >= beta_detail::kDeepP) {
            const double est = (lt.hi + std::log(sa) + special::Betaln::f(sa, sb)) / sa;
            if (est > -40.0) return lower ? special::Fdtri::f(dfn, dfd, t) : upper_solve(t, dfn, dfd);
        }
        const num::DD lr = beta_detail::deep_log_inverse(sa, sb, lt);  // log w or log v
        const double r = std::exp(lr.hi);
        const num::DD lo = r < 0x1p-60 ? num::DD{-r, 0.0} : num::log_dd(-std::expm1(lr.hi));  // log(1 - r)
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
        if (!(p > 0.0 && p < 1.0)) return special::Fdtri::f(dfn, dfd, p);
        if (p > 0.5) return solve(1.0 - p, num::log_dd(1.0 - p), dfn, dfd, false);
        return solve(p, num::log_dd(p), dfn, dfd, true);
    }
    static double iccdf(double q, double dfn, double dfd) {
        if (!(q >= 0.0 && q <= 1.0)) return kNaN;
        if (q == 0.0) return kInf;
        if (q == 1.0) return 0.0;
        if (q > 0.5) return solve(1.0 - q, num::log_dd(1.0 - q), dfn, dfd, true);
        return solve(q, num::log_dd(q), dfn, dfd, false);
    }
    static double from_log(double lp, double dfn, double dfd, bool upper) {
        if (!(lp <= 0.0)) return kNaN;
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
        return 2.0 * r * r * ((dfn + dfd - 2.0) / (dfn * (dfd - 4.0)));
    }
    static double standard_deviation(double dfn, double dfd) {
        const double v = variance(dfn, dfd);
        return std::sqrt(v);
    }
    static double skewness(double dfn, double dfd) {
        if (!(dfd > 6.0)) return dfd > 4.0 ? kInf : kNaN;
        return (2.0 * dfn + dfd - 2.0) * std::sqrt(8.0 * (dfd - 4.0)) / ((dfd - 6.0) * std::sqrt(dfn * (dfn + dfd - 2.0)));
    }
    static double kurtosis(double dfn, double dfd) {
        if (!(dfd > 8.0)) return dfd > 6.0 ? kInf : kNaN;
        // 12 (dfn (5 dfd - 22)(dfn + dfd - 2) + (dfd - 4)(dfd - 2)^2) / (dfn (dfd - 6)(dfd - 8)(dfn + dfd - 2)),
        // as sums of ratios (no products of the parameters to overflow); every term is positive.
        const double s = dfn + dfd - 2.0;
        const double t1 = (5.0 * dfd - 22.0) / (dfd - 6.0) / (dfd - 8.0);
        const double t2 = ((dfd - 4.0) / (dfd - 6.0)) * ((dfd - 2.0) / (dfd - 8.0)) * ((dfd - 2.0) / dfn) / s;
        return 12.0 * (t1 + t2);
    }
    static double mode(double dfn, double dfd) { return dfn > 2.0 ? ((dfn - 2.0) / dfn) * (dfd / (dfd + 2.0)) : 0.0; }
    static double median(double dfn, double dfd) { return special::Fdtri::f(dfn, dfd, 0.5); }
    // log(b/a) + log B(a, b) + (1 - a) psi(a) - (1 + b) psi(b) + (a + b) psi(a + b)
    // = G(a) + G(b) + Gc(c) - log(a)/2 + 3 (log b - log c)/2 + log(2 pi)/2 + 2 (psi(c) - psi(b)).
    static double entropy(double dfn, double dfd) {
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
