// The logistic, Gumbel (maximum) and generalized extreme value distributions
// (docs/api/stats.md §2.1, docs/design/stats.md T26).
//
//   * logistic: P(X <= x) = 1 / (1 + e^-z), z = (x - mu) / sigma a double-double
//     (stats/numeric.hpp); the small side e^w / (1 + e^w), w = -|z|.
//   * Gumbel and GEV: P(X <= x) = exp(-t), t = e^E. Gumbel E = -z; GEV (scipy's c,
//     c = -xi) E = log(1 - c z) / c = -z - mu(u) / c, u = -c z, mu(u) = u - log(1 + u)
//     (num::dd::log1pmx_neg), which is O(c z^2) and so continuous through c = 0. t from
//     E as the Weibull's t (exp's rounding folded back in past t = 4). The inverses go
//     back through lt = log t: Gumbel z = -lt, GEV z = -expm1(c lt) / c.
//   * GEV moments: Z = (1 - W) / c with W = Y^c, Y ~ Exp(1), E[W^k] = Gamma(1 + k c): the
//     Weibull's D_k machinery with x = c (weibull_detail::skewness_x, kurtosis_x).
#pragma once

#include "special/ops/gamma.hpp"
#include "stats/coeffs/zeta_minus_one.hpp"
#include "stats/dist/weibull.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace extreme_detail {

using num::DD;
namespace dd = num::dd;

inline constexpr DD kEuler = {0.5772156649015329, -4.942915152430645e-18};
inline constexpr DD kPi2Over6 = {1.6449340668482264, 3.040672350398476e-17};
inline constexpr DD kLogLn2 = {-0.36651292058166435, 2.0606571710351486e-17};  // log(log 2)

// t = e^E as a double-double: exp(E.hi) (1 + E.lo), with the residual of exp's rounding
// folded in past t = 4 (e^-t multiplies t's relative error by t).
inline DD exp_t(DD e) {
    const double th = std::exp(e.hi);
    if (th > 4.0 && std::isfinite(th)) {
        const DD r = e - num::log_dd(th);
        return {th, th * (r.hi + r.lo)};
    }
    return {th, th * e.lo};
}

// P(X <= x) = e^-t (lower) or P(X > x) = 1 - e^-t, or their logs, from E = log t.
inline double tail(DD e, bool upper, bool log) {
    const DD t = exp_t(e);
    if (!upper) {
        if (log) return -(t.hi + t.lo);
        return std::isinf(t.hi) ? 0.0 : num::exp_dd(-t);
    }
    if (!log) return -std::expm1(-t.hi) + std::exp(-t.hi) * t.lo;
    if (t.hi < 0x1p-60) {  // log(1 - e^-t) = log t - t/2 + ...: E to the last bit
        const DD l = e - t * 0.5;
        return l.hi + l.lo;
    }
    if (t.hi < num::kLn2.hi) return std::log(-std::expm1(-t.hi) + std::exp(-t.hi) * t.lo);
    return std::isinf(t.hi) ? 0.0 : std::log1p(-num::exp_dd(-t));
}

// log(-log p) for 0 < p < 1 and log(-log(1 - q)) for 0 < q < 1, as double-doubles.
inline DD lt_neglog1m(double q) {
    if (q > 0.5) return dd::log(-num::log_dd(1.0 - q));  // 1 - q exact
    if (q < 0x1p-60) return num::log_dd(q) + 0.5 * q;     // -log(1 - q) = q + q^2/2 + ...
    return num::log_dd(-std::log1p(-q));
}
inline DD lt_neglog(double p) { return p > 0.5 ? lt_neglog1m(1.0 - p) : dd::log(-num::log_dd(p)); }
// log(-log(1 - e^lq)) for lq < 0.
inline DD lt_neglog1m_exp(double lq) {
    if (lq < -40.0) return {lq, 0.5 * std::exp(lq)};  // lq + q/2 + ...
    if (lq < -num::kLn2.hi) return num::log_dd(-std::log1p(-std::exp(lq)));
    return dd::log(-num::log_dd(-std::expm1(lq)));
}

// mu + sigma z for a double-double z.
inline double affine(double mu, double sigma, DD z) {
    if (std::isinf(z.hi)) return z.hi;
    return std::fma(sigma, z.hi, mu) + sigma * z.lo;
}

// log Gamma(1 + x) as a double-double: the series -euler x + sum_j>=2 (-1)^j zeta(j) x^j / j
// for |x| <= 1/2 (no 1 + x to round for a tiny x), else log_gamma of the exact 1 + x.
inline DD lgamma1p(double x) {
    if (std::fabs(x) > 0.5) return num::log_gamma(dd::two_sum(1.0, x));
    DD sum = kEuler * -x, p{x, 0.0};
    for (int j = 2; j < 200; ++j) {
        p = p * x;
        DD zeta{1.0, 0.0};
        if (j <= kZetaMinusOneMax) zeta = zeta + DD{kZetaMinusOne[j - 2][0], kZetaMinusOne[j - 2][1]};
        DD term = zeta * p / static_cast<double>(j);
        if (j % 2 == 1) term = -term;
        sum = sum + term;
        if (std::fabs(term.hi) <= std::fabs(sum.hi) * 0x1p-108) break;
    }
    return sum;
}

// e^d - 1 for any double-double d (-inf -> -1, +inf -> inf), to ~2^-77 relative (dd::log's
// accuracy; the GEV inverse needs a few bits past a double): libm's expm1 of hi, then one
// Newton step on log(1 + m) = d with log(1 + m) in double-double (1 + m exact). Below 2^-20
// the Taylor series to d^6 (1 + m would hold a tiny m only to 2^-106 absolute). The Taylor
// series in double-double everywhere (num::expm1_full) cost 240 ns an element.
inline DD expm1_any(DD d) {
    const double h = d.hi;
    if (h < -750.0) return {-1.0, 0.0};
    if (h > 710.0) return {kInf, 0.0};
    if (std::fabs(h) < 0x1p-20) {
        DD term = d, sum = d;
        for (int n = 2; n <= 6; ++n) {
            term = term * d / static_cast<double>(n);
            sum = sum + term;
        }
        return sum;
    }
    if (h < -38.0) return {-1.0, std::exp(h)};  // e^h below half an ULP of 1
    const double m = std::expm1(h);
    if (!std::isfinite(m)) return {m, 0.0};
    const DD one = dd::two_sum(1.0, m);
    return DD{m, 0.0} - (dd::log(one) - d) * one;
}

}  // namespace extreme_detail

struct Logistic {
    static constexpr char name[] = "logistic";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    // w = -|z| and e^w: the density is e^w / (sigma (1 + e^w)^2).
    static num::DD neg_abs(num::DD z) { return z.hi > 0.0 ? -z : z; }
    static num::DD log_density(double x, double mu, double sigma) {
        const num::DD w = neg_abs(num::standardize(x, mu, sigma));
        thread_local num::Memo memo;
        return w - 2.0 * std::log1p(num::exp_dd(w)) - memo.get(sigma, 0.0, [&] { return num::log_dd(sigma); });
    }
    static double pdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return 0.0;
        return num::exp_dd(log_density(x, mu, sigma));
    }
    static double logpdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return -kInf;
        const num::DD l = log_density(x, mu, sigma);
        return l.hi + l.lo;
    }
    // P(X <= x) (upper: P(X > x)) or its log: the small side e^w / (1 + e^w), the large
    // 1 / (1 + e^w), w = -|z|.
    static double tail(double x, double mu, double sigma, bool upper, bool log) {
        if (std::isinf(x)) {
            const bool zero = (x > 0.0) == upper;
            return log ? (zero ? -kInf : 0.0) : (zero ? 0.0 : 1.0);
        }
        num::DD z = num::standardize(x, mu, sigma);
        if (upper) z = -z;
        const bool small = z.hi < 0.0;
        const num::DD w = small ? z : -z;
        const double e = num::exp_dd(w);
        if (small) {
            if (log) {
                const num::DD l = w - std::log1p(e);
                return l.hi + l.lo;
            }
            return e / (1.0 + e);
        }
        return log ? -std::log1p(e) : 1.0 / (1.0 + e);
    }
    static double cdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false, false); }
    static double ccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true, false); }
    static double logcdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false, true); }
    static double logccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true, true); }

    // z = log(p / (1 - p)) for 0 < p <= 1/2: log p - log1p(-p) below 1/4, else
    // -log1p((1 - 2p) / p) (1 - 2p exact).
    static num::DD z_small(double p) {
        if (p <= 0.25) return num::log_dd(p) - std::log1p(-p);
        return {-std::log1p((1.0 - 2.0 * p) / p), 0.0};
    }
    static num::DD z_of(double p) { return p <= 0.5 ? z_small(p) : -z_small(1.0 - p); }  // 1 - p exact
    // the same from lp = log p
    static num::DD z_of_log(double lp) {
        if (lp <= -1.3862943611198906) return num::DD{lp, 0.0} - std::log1p(-std::exp(lp));  // p <= 1/4
        const double q = -std::expm1(lp);
        if (q <= 0.25) return -z_small(q);
        return num::DD{lp, 0.0} - std::log(q);
    }
    static double icdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? -kInf : kInf;
        return extreme_detail::affine(mu, sigma, z_of(p));
    }
    static double iccdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? kInf : -kInf;
        return extreme_detail::affine(mu, sigma, -z_of(p));
    }
    static double ilogcdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? kInf : -kInf;
        return extreme_detail::affine(mu, sigma, z_of_log(lp));
    }
    static double ilogccdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? -kInf : kInf;
        return extreme_detail::affine(mu, sigma, -z_of_log(lp));
    }

    static double mean(double mu, double) { return mu; }
    static double variance(double, double sigma) { return sigma * sigma * 3.289868133696453; }  // pi^2 / 3
    static double standard_deviation(double, double sigma) { return sigma * 1.8137993642342178; }  // pi / sqrt 3
    static double skewness(double, double) { return 0.0; }
    static double kurtosis(double, double) { return 1.2; }
    static double median(double mu, double) { return mu; }
    static double mode(double mu, double) { return mu; }
    static double entropy(double, double sigma) {  // log sigma + 2
        const num::DD h = num::log_dd(sigma) + 2.0;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return -kInf; }
    static double support_hi(double, double) { return kInf; }
};

struct Gumbel {
    static constexpr char name[] = "gumbel";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    // E - t - log sigma, E = -z, t = e^E.
    static num::DD log_density(double x, double mu, double sigma) {
        const num::DD e = -num::standardize(x, mu, sigma);
        const num::DD t = extreme_detail::exp_t(e);
        if (std::isinf(t.hi)) return {-kInf, 0.0};
        thread_local num::Memo memo;
        return e - t - memo.get(sigma, 0.0, [&] { return num::log_dd(sigma); });
    }
    static double pdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return 0.0;
        const num::DD l = log_density(x, mu, sigma);
        return std::isinf(l.hi) ? 0.0 : num::exp_dd(l);
    }
    static double logpdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return -kInf;
        const num::DD l = log_density(x, mu, sigma);
        return l.hi + l.lo;
    }
    static double tail(double x, double mu, double sigma, bool upper, bool log) {
        if (std::isinf(x)) {
            const bool zero = (x > 0.0) == upper;
            return log ? (zero ? -kInf : 0.0) : (zero ? 0.0 : 1.0);
        }
        return extreme_detail::tail(-num::standardize(x, mu, sigma), upper, log);
    }
    static double cdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false, false); }
    static double ccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true, false); }
    static double logcdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false, true); }
    static double logccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true, true); }

    // x = mu - sigma lt, lt = log t.
    static double from_lt(num::DD lt, double mu, double sigma) { return extreme_detail::affine(mu, sigma, -lt); }
    static double icdf(double p, double mu, double sigma) {  // t = -log p
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? -kInf : kInf;
        return from_lt(extreme_detail::lt_neglog(p), mu, sigma);
    }
    static double iccdf(double p, double mu, double sigma) {  // t = -log(1 - p)
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? kInf : -kInf;
        return from_lt(extreme_detail::lt_neglog1m(p), mu, sigma);
    }
    static double ilogcdf(double lp, double mu, double sigma) {  // t = -lp
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? kInf : -kInf;
        return from_lt(num::log_dd(-lp), mu, sigma);
    }
    static double ilogccdf(double lp, double mu, double sigma) {  // t = -log(1 - e^lp)
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? -kInf : kInf;
        return from_lt(extreme_detail::lt_neglog1m_exp(lp), mu, sigma);
    }

    static double mean(double mu, double sigma) { return extreme_detail::affine(mu, sigma, extreme_detail::kEuler); }
    static double variance(double, double sigma) { return sigma * sigma * 1.6449340668482264; }   // pi^2 / 6
    static double standard_deviation(double, double sigma) { return sigma * 1.282549830161864; }  // pi / sqrt 6
    static double skewness(double, double) { return 1.1395470994046486; }  // 12 sqrt 6 zeta(3) / pi^3
    static double kurtosis(double, double) { return 2.4; }
    static double median(double mu, double sigma) { return from_lt(extreme_detail::kLogLn2, mu, sigma); }
    static double mode(double mu, double) { return mu; }
    static double entropy(double, double sigma) {  // log sigma + euler + 1
        const num::DD h = num::log_dd(sigma) + extreme_detail::kEuler + 1.0;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return -kInf; }
    static double support_hi(double, double) { return kInf; }
};

// GEV with scipy's shape c (= -xi; T26): P(X <= x) = exp(-(1 - c z)^(1/c)), the Gumbel at
// c = 0; c > 0 bounds the support above at mu + sigma/c, c < 0 below.
struct GEV {
    static constexpr char name[] = "gev";
    static constexpr std::array<const char*, 3> params{"c", "mu", "sigma"};

    static bool valid(double c, double mu, double sigma) {
        return std::isfinite(c) && std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma);
    }

    // E = log t and log y (y = 1 - c z) at x, or `inside` false when y <= 0 (`edge` when
    // y is exactly 0).
    struct At {
        num::DD e, ly;
        bool inside, edge;
    };
    static At at(double x, double c, double mu, double sigma) {
        const num::DD z = num::standardize(x, mu, sigma);
        if (c == 0.0) return {-z, {0.0, 0.0}, true, false};
        const num::DD u = -(z * c);  // y = 1 + u
        if (u.hi < -1.0 || (u.hi == -1.0 && u.lo <= 0.0)) return {{0.0, 0.0}, {0.0, 0.0}, false, u.hi == -1.0 && u.lo == 0.0};
        num::DD ly;  // log(1 + u): u - mu(u) only for a small u (it cancels for a large one)
        if (u.hi > 1e300) ly = num::log_dd(u.hi);
        else if (std::fabs(u.hi) < 0.5) ly = u - num::dd::log1pmx_neg(u);
        else ly = num::dd::log(num::DD{1.0, 0.0} + u);
        return {num::div(ly, c), ly, true, false};
    }
    static num::DD log_density(const At& a, double sigma) {
        const num::DD t = extreme_detail::exp_t(a.e);
        if (std::isinf(t.hi)) return {-kInf, 0.0};
        thread_local num::Memo memo;
        return a.e - a.ly - t - memo.get(sigma, 0.0, [&] { return num::log_dd(sigma); });
    }
    // the density at the end of the support (y = 0): 0, 1/sigma or inf as c <, =, > 1 above;
    // 0 at the lower end (c < 0).
    static double edge_pdf(double c, double sigma) { return c < 1.0 ? 0.0 : (c == 1.0 ? 1.0 / sigma : kInf); }
    static double pdf(double x, double c, double mu, double sigma) {
        if (std::isinf(x)) return 0.0;
        const At a = at(x, c, mu, sigma);
        if (!a.inside) return a.edge && c > 0.0 ? edge_pdf(c, sigma) : 0.0;
        const num::DD l = log_density(a, sigma);
        return std::isinf(l.hi) ? 0.0 : num::exp_dd(l);
    }
    static double logpdf(double x, double c, double mu, double sigma) {
        if (std::isinf(x)) return -kInf;
        const At a = at(x, c, mu, sigma);
        if (!a.inside) return a.edge && c > 0.0 ? std::log(edge_pdf(c, sigma)) : -kInf;
        const num::DD l = log_density(a, sigma);
        return l.hi + l.lo;
    }
    static double tail(double x, double c, double mu, double sigma, bool upper, bool log) {
        const At a = std::isinf(x) ? At{} : at(x, c, mu, sigma);
        if (std::isinf(x) || !a.inside) {
            // beyond an end: above it (x > 0 side, or c > 0) the cdf is 1, below it 0
            const bool above = std::isinf(x) ? x > 0.0 : c > 0.0;
            const bool zero = above == upper;
            return log ? (zero ? -kInf : 0.0) : (zero ? 0.0 : 1.0);
        }
        return extreme_detail::tail(a.e, upper, log);
    }
    static double cdf(double x, double c, double mu, double sigma) { return tail(x, c, mu, sigma, false, false); }
    static double ccdf(double x, double c, double mu, double sigma) { return tail(x, c, mu, sigma, true, false); }
    static double logcdf(double x, double c, double mu, double sigma) { return tail(x, c, mu, sigma, false, true); }
    static double logccdf(double x, double c, double mu, double sigma) { return tail(x, c, mu, sigma, true, true); }

    // z = -expm1(c lt) / c (-lt at c = 0), x = mu + sigma z.
    static num::DD z_of(num::DD lt, double c) {
        if (c == 0.0) return -lt;
        if (std::isinf(lt.hi)) {  // t = 0 or inf: an end of the support
            const bool to_end = (lt.hi < 0.0) == (c > 0.0);
            return to_end ? num::div(num::DD{1.0, 0.0}, c) : num::DD{c > 0.0 ? -kInf : kInf, 0.0};
        }
        const num::DD m = extreme_detail::expm1_any(lt * c);
        if (std::isinf(m.hi)) return {c > 0.0 ? -kInf : kInf, 0.0};
        return -num::div(m, c);
    }
    static double from_lt(num::DD lt, double c, double mu, double sigma) {
        const num::DD cl = lt * c;
        if (c != 0.0 && std::isfinite(cl.hi) && cl.hi > 700.0) {
            // z = -(e^cl - 1) / c ~ -e^cl / c overflows while sigma z may not: through the logs
            const num::DD l = cl + num::log_dd(sigma) - num::log_dd(std::fabs(c));
            const double v = num::exp_dd(l);
            return c > 0.0 ? mu - v : mu + v;
        }
        return extreme_detail::affine(mu, sigma, z_of(lt, c));
    }
    static double icdf(double p, double c, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return support_lo(c, mu, sigma);
        if (p == 1.0) return support_hi(c, mu, sigma);
        return from_lt(extreme_detail::lt_neglog(p), c, mu, sigma);
    }
    static double iccdf(double p, double c, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return support_hi(c, mu, sigma);
        if (p == 1.0) return support_lo(c, mu, sigma);
        return from_lt(extreme_detail::lt_neglog1m(p), c, mu, sigma);
    }
    static double ilogcdf(double lp, double c, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return support_hi(c, mu, sigma);
        if (lp == -kInf) return support_lo(c, mu, sigma);
        return from_lt(num::log_dd(-lp), c, mu, sigma);
    }
    static double ilogccdf(double lp, double c, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return support_lo(c, mu, sigma);
        if (lp == -kInf) return support_hi(c, mu, sigma);
        return from_lt(extreme_detail::lt_neglog1m_exp(lp), c, mu, sigma);
    }

    // E[Z] = (1 - Gamma(1 + c)) / c = -expm1(log Gamma(1 + c)) / c (euler at c = 0), finite
    // for c > -1 (else +inf).
    static double mean(double c, double mu, double sigma) {
        if (c == 0.0) return extreme_detail::affine(mu, sigma, extreme_detail::kEuler);
        if (!(c > -1.0)) return kInf;
        const num::DD m = -num::div(extreme_detail::expm1_any(extreme_detail::lgamma1p(c)), c);
        return extreme_detail::affine(mu, sigma, m);
    }
    // sigma^2 Gamma(1 + c)^2 E_2 / c^2 (pi^2/6 at c = 0); +inf for -1 < c <= -1/2, NaN below.
    static num::DD var_std(double c) {
        if (c == 0.0) return extreme_detail::kPi2Over6;
        const num::DD e2 = num::expm1_full(weibull_detail::lgamma_ratio_x(2, num::DD{c, 0.0}));
        return num::exp_full(extreme_detail::lgamma1p(c) * 2.0) * num::div(num::div(e2, c), c);
    }
    static double variance(double c, double, double sigma) {
        if (!(2.0 * c > -1.0)) return c > -1.0 ? kInf : kNaN;
        const num::DD v = var_std(c);
        return sigma * sigma * (v.hi + v.lo);
    }
    static double standard_deviation(double c, double, double sigma) {
        if (!(2.0 * c > -1.0)) return c > -1.0 ? kInf : kNaN;
        const num::DD v = var_std(c);
        return sigma * std::sqrt(v.hi + v.lo);
    }
    // skewness(Z) = -sign(c) skewness(W); finite for c > -1/3, +inf while the variance is
    // finite, NaN beyond (the rule of docs/api/stats.md §2).
    static double skewness(double c, double, double) {
        if (c == 0.0) return 1.1395470994046486;
        if (!(std::fma(3.0, c, 1.0) > 0.0)) return 2.0 * c > -1.0 ? kInf : kNaN;
        const double s = weibull_detail::skewness_x(num::DD{c, 0.0});
        return c > 0.0 ? -s : s;
    }
    // The excess kurtosis: its numerator E_4 - 4 E_3 + 6 E_2 - 3 E_2^2 cancels its c^2 terms
    // (O(c^4) out of O(c^2): 1e-14 left of double-double at c = 1e-9), so below |c| = 1e-4 the
    // Taylor series about c = 0 (coefficients from mpmath's taylor of the exact Gamma form at
    // 120 digits; radius 1/4, the next term below 1e-22 there).
    static double kurtosis(double c, double, double) {
        if (!(4.0 * c > -1.0)) return 2.0 * c > -1.0 ? kInf : kNaN;
        if (std::fabs(c) < 1e-4) {
            constexpr double k[7] = {2.4, -28.917688340799625, 152.61603300554222, -666.448329815961,
                                     2877.87676765926, -12120.830749353447, 50326.42880216044};
            double r = k[6];
            for (int i = 5; i >= 0; --i) r = r * c + k[i];
            return r;
        }
        return weibull_detail::kurtosis_x(num::DD{c, 0.0});
    }
    static double median(double c, double mu, double sigma) { return from_lt(extreme_detail::kLogLn2, c, mu, sigma); }
    // the density's peak at t = 1 - c (log t = log1p(-c)); for c >= 1 the upper end.
    static double mode(double c, double mu, double sigma) {
        if (c >= 1.0) return support_hi(c, mu, sigma);
        const num::DD l = num::DD{-c, 0.0} - num::dd::log1pmx_neg(num::DD{-c, 0.0});  // log(1 - c)
        return from_lt(l, c, mu, sigma);
    }
    static double entropy(double c, double, double sigma) {  // log sigma + euler (1 - c) + 1
        const num::DD h = num::log_dd(sigma) + extreme_detail::kEuler - extreme_detail::kEuler * c + 1.0;
        return h.hi + h.lo;
    }
    // mu + sigma / c at the bounded end.
    static double end(double c, double mu, double sigma) {
        return extreme_detail::affine(mu, sigma, num::div(num::DD{1.0, 0.0}, c));
    }
    static double support_lo(double c, double mu, double sigma) { return c < 0.0 ? end(c, mu, sigma) : -kInf; }
    static double support_hi(double c, double mu, double sigma) { return c > 0.0 ? end(c, mu, sigma) : kInf; }
};

}  // namespace sd::stats
