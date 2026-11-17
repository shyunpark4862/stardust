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
#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
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
    if (th == 0.0 || std::isinf(th)) return {th, 0.0};
    if (th > 4.0) {
        const DD r = e - num::log_dd(th);
        return {th, th * (r.hi + r.lo)};
    }
    return {th, th * e.lo};
}

// P(X <= x) = e^-t (lower) or P(X > x) = 1 - e^-t, or their logs, from E = log t.
inline double tail(DD e, bool upper, bool log) {
    if (std::isinf(e.hi)) {  // E overflowed: t = 0 or inf
        const bool zero = upper ? e.hi < 0.0 : e.hi > 0.0;
        return log ? (zero ? -kInf : 0.0) : (zero ? 0.0 : 1.0);
    }
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
    const double h = std::fma(sigma, z.hi, mu);
    return std::isinf(h) ? h : h + sigma * z.lo;
}

using num::lgamma1p;

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
    if (h < -38.0) return {-1.0, num::exp_dd(d)};  // e^d below half an ULP of 1
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
        if (std::isinf(w.hi)) return {-kInf, 0.0};  // z overflowed: the density is 0
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
        if (std::isinf(w.hi)) return small ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);  // z overflowed
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

    // ---- chunk kernels: e = e^-|z| by vForce (relative error kVfExp + 1 = 3.5), z the scalar
    // definition's double-double (ckd::standardize; NaN where it must take over). zh, zl:
    // -|z|. small[i]: z < 0 (for the upper tail: z > 0).
    static void e_chunk(const Chunk<2>& c, bool upper, double* wh, double* wl, double* e, bool* small) {
        ckd::standardize(c.x, c.p[0], c.p[1], wh, wl, c.n);
        for (int i = 0; i < c.n; ++i) {
            const bool neg = upper ? wh[i] > 0.0 : wh[i] < 0.0;
            const bool flip = wh[i] > 0.0;  // to -|z|
            small[i] = neg;
            wh[i] = flip ? -wh[i] : wh[i];
            wl[i] = flip ? -wl[i] : wl[i];
        }
        ckd::exp_dd(e, wh, wl, c.n);
    }
    // e / (sigma (1 + e)^2) in double: e's error reaches it times (1 - e) / (1 + e) <= 1, the
    // sum counts twice, the square and two quotients once each: 3.5 + 5.
    static void pdf_chunk(const Chunk<2>& c, double* out) {
        double wh[kChunkSize], wl[kChunkSize];
        bool fast[kChunkSize];
        e_chunk(c, false, wh, wl, out, fast);
        for (int i = 0; i < c.n; ++i) {
            const double e = out[i], q = 1.0 + e;
            out[i] = (e / (q * q)) / c.p[1][i];
            fast[i] = e >= 0x1p-1000 && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;  // false for NaN
        }
        ck::fallback(c.n, fast, out, [&](int i) { return pdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // w - 2 L - log sigma with L = log(1 + e) (ckd::log1p; absolute error kLog1p L + 3.5 e /
    // (1 + e) <= 6.4) and the rest in double-double: 12.8 absolute and the last rounding, inside
    // logpdf's mixed bound (T20) for any value.
    static void logpdf_chunk(const Chunk<2>& c, double* out) {
        double wh[kChunkSize], wl[kChunkSize], kh[kChunkSize], kl[kChunkSize];
        bool fast[kChunkSize];
        e_chunk(c, false, wh, wl, out, fast);
        ckd::log1p(out, out, c.n);
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[1][i]); });
        for (int i = 0; i < c.n; ++i) {
            const num::DD l = num::DD{wh[i], wl[i]} - 2.0 * out[i] - num::DD{kh[i], kl[i]};
            out[i] = l.hi + l.lo;
            fast[i] = wh[i] == wh[i];
        }
        ck::fallback(c.n, fast, out, [&](int i) { return logpdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // The small side e / (1 + e): 3.5 / (1 + e) + 2; the large 1 / (1 + e): 3.5 e / (1 + e) + 2.
    // Their logs: w - L (absolute 6.4 over |value| >= log 2, and the rounding: 10.3) and -L
    // (e's 3.5 times e / ((1 + e) L) <= 1, plus kLog1p: 10.2).
    static void tail_chunk(const Chunk<2>& c, double* out, bool upper, bool log) {
        double wh[kChunkSize], wl[kChunkSize], e[kChunkSize];
        bool fast[kChunkSize], small[kChunkSize];
        e_chunk(c, upper, wh, wl, e, small);
        if (log) ckd::log1p(out, e, c.n);
        for (int i = 0; i < c.n; ++i) {
            fast[i] = wh[i] == wh[i] && e[i] >= 0x1p-1000;
            if (log) {
                const num::DD l = num::DD{wh[i], wl[i]} - out[i];
                out[i] = small[i] ? l.hi + l.lo : -out[i];
            } else {
                out[i] = (small[i] ? e[i] : 1.0) / (1.0 + e[i]);
            }
        }
        ck::fallback(c.n, fast, out, [&](int i) { return tail(c.x[i], c.p[0][i], c.p[1][i], upper, log); });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false, false); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true, false); }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false, true); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true, true); }
    // mu + sigma z with z = log(p / (1 - p)) by vForce's log: z's absolute error is 2 (the
    // difference and the quotient) + kVfLog |z|, x's that times sigma. Taken where it is inside
    // the budget relative to x, and for p in [1/4, 3/4] always: there |z| <= 1.1 (6.4 units)
    // and the quantile's condition number is kappa = sigma / ((1 - p) |x|) >= (4/3) sigma / |x|
    // (for iccdf sigma / (p |x|)), so the error is at most 4.8 of the contract's max(1, kappa)
    // (docs/design/stats.md §2.3).
    static void quantile_chunk(const Chunk<2>& c, double* out, bool upper) {
        double z[kChunkSize];
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) z[i] = c.x[i] / (1.0 - c.x[i]);
        ck::log(z, z, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i], mu = c.p[0][i], sigma = c.p[1][i];
            const double zi = upper ? -z[i] : z[i];
            out[i] = std::fma(sigma, zi, mu);
            const bool central = p >= 0.25 && p <= 0.75;
            fast[i] = p >= 0x1p-1000 && p < 1.0 &&
                      (central || (2.0 + ck::kVfLog * std::fabs(zi)) * sigma <= (kFastBudget - 1.0) * std::fabs(out[i]));
        }
        ck::fallback(c.n, fast, out, [&](int i) {
            return upper ? iccdf(c.x[i], c.p[0][i], c.p[1][i]) : icdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void icdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, false); }
    static void iccdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, true); }

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
        if (std::isinf(e.hi)) return {-kInf, 0.0};  // z overflowed: the density is 0 on both sides
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

    // ---- chunk kernels (weibull_detail::t_chunk, tail_chunk): E = -z is the scalar
    // definition's double-double (eu = 0), t = e^E plain (grade 0) or folded (grade 1).
    static void tail_kernel(const Chunk<2>& c, double* out, bool* fast, weibull_detail::Tail kind, int grade) {
        double eh[kChunkSize], el[kChunkSize], tau[kChunkSize], th[kChunkSize], tl[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], eh, el, c.n);
        for (int i = 0; i < c.n; ++i) eh[i] = -eh[i], el[i] = -el[i], tau[i] = 0.0;
        weibull_detail::t_chunk(eh, el, tau, grade > 0, th, tl, tau, c.n);
        weibull_detail::tail_chunk(kind, th, tl, tau, out, eh, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = eh[i] <= kFastBudget;  // false for NaN
    }
    static void tail_run(const Chunk<2>& c, double* out, weibull_detail::Tail kind, bool upper, bool log) {
        bool fast[kChunkSize];
        tail_kernel(c, out, fast, kind, 0);
        ckd::retry(c, out, fast, [&](const Chunk<2>& s, double* o, bool* f) { tail_kernel(s, o, f, kind, 1); });
        ck::fallback(c.n, fast, out, [&](int i) { return tail(c.x[i], c.p[0][i], c.p[1][i], upper, log); });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kExp, false, false); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kOneExp, true, false); }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogExp, false, true); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogOne, true, true); }
    // E - t - log sigma in the scalar definition's double-doubles: absolute error t tau and the
    // last bit; logpdf against its mixed bound (T20), pdf with exp's kVfExp + 1 more. The pdf's
    // grade 0 is t e^-t / sigma in double instead: t's error (tau = kVfExp + 1) reaches it
    // times |1 - t|, then exp and two operations: tau |1 - t| + kVfExp + 2.
    static void density_kernel(const Chunk<2>& c, double* out, bool* fast, bool log, int grade) {
        double eh[kChunkSize], el[kChunkSize], tau[kChunkSize], th[kChunkSize], tl[kChunkSize], kh[kChunkSize],
            kl[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], eh, el, c.n);
        for (int i = 0; i < c.n; ++i) eh[i] = -eh[i], el[i] = -el[i], tau[i] = 0.0;
        weibull_detail::t_chunk(eh, el, tau, grade > 0, th, tl, tau, c.n);
        if (grade == 0 && !log) {
            for (int i = 0; i < c.n; ++i) el[i] = -th[i];
            ck::exp(el, el, c.n);
            for (int i = 0; i < c.n; ++i) {
                const double t = th[i] + tl[i], m = t * el[i];
                out[i] = m / c.p[1][i];
                fast[i] = tau[i] * std::fabs(1.0 - t) + ck::kVfExp + 2.0 <= kFastBudget && m >= 0x1p-900 &&
                          out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;  // false for NaN
            }
            return;
        }
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[1][i]); });
        for (int i = 0; i < c.n; ++i) {
            const num::DD l = num::DD{eh[i], el[i]} - num::DD{th[i], tl[i]} - num::DD{kh[i], kl[i]};
            const double abs = th[i] * tau[i] + 1.0;
            if (log) {
                out[i] = l.hi + l.lo;
                fast[i] = out[i] == out[i] && abs <= kFastBudget * std::max(1.0, std::fabs(out[i]));
            } else {
                eh[i] = l.hi, el[i] = l.lo;
                tau[i] = abs + ck::kVfExp + 1.0;
            }
        }
        if (log) return;
        ckd::exp_dd(out, eh, el, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = tau[i] <= kFastBudget && out[i] >= 0x1p-1000;
    }
    static void density_run(const Chunk<2>& c, double* out, bool log) {
        bool fast[kChunkSize];
        density_kernel(c, out, fast, log, 0);
        ckd::retry(c, out, fast, [&](const Chunk<2>& s, double* o, bool* f) { density_kernel(s, o, f, log, 1); });
        ck::fallback(c.n, fast, out, [&](int i) {
            return log ? logpdf(c.x[i], c.p[0][i], c.p[1][i]) : pdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) { density_run(c, out, false); }
    static void logpdf_chunk(const Chunk<2>& c, double* out) { density_run(c, out, true); }
    // x = mu - sigma log t in double from t with relative error tu: log t's absolute error is
    // e = tu + kVfLog |log t|, x's that times sigma. Taken where it is inside the budget
    // relative to x, and for p in [0.01, 0.99] where e r is, r = t for icdf (t = -log p, tu =
    // kVfLog) and (1 - p) t / p for iccdf (t = -log(1 - p), tu = 7.5): the quantile's
    // condition number is kappa = sigma / (r |x|), so e r is the error in units of the
    // contract's max(1, kappa) (docs/design/stats.md §2.3; with kappa < 1, sigma / |x| < r).
    static void from_t_chunk(const Chunk<2>& c, double* t, double tu, bool upper, double* out, bool* fast) {
        double lt[kChunkSize];
        ck::log(lt, t, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i], mu = c.p[0][i], sigma = c.p[1][i];
            out[i] = std::fma(-sigma, lt[i], mu);
            const double e = tu + ck::kVfLog * std::fabs(lt[i]);
            const double r = upper ? (1.0 - p) * t[i] / p : t[i];
            const bool central = p >= 0.01 && p <= 0.99 && e * r <= kFastBudget - 1.0;
            fast[i] = fast[i] && (central || e * sigma <= (kFastBudget - 1.0) * std::fabs(out[i]));
        }
    }
    static void icdf_chunk(const Chunk<2>& c, double* out) {
        double t[kChunkSize];
        bool fast[kChunkSize];
        ck::log(t, c.x, c.n);
        for (int i = 0; i < c.n; ++i) {
            t[i] = -t[i];
            fast[i] = c.x[i] >= 0x1p-1000 && c.x[i] < 1.0;
        }
        from_t_chunk(c, t, ck::kVfLog, false, out, fast);
        ck::fallback(c.n, fast, out, [&](int i) { return icdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // t = -log(1 - p) as Weibull::icdf_chunk's (tu = 7.5).
    static void iccdf_chunk(const Chunk<2>& c, double* out) {
        double t[kChunkSize];
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) t[i] = 1.0 - c.x[i];
        ck::log(t, t, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i];
            t[i] = p < 0.25 ? -std::log1p(-p) : -t[i];
            fast[i] = p >= 0x1p-1000 && p < 1.0;
        }
        from_t_chunk(c, t, 7.5, true, out, fast);
        ck::fallback(c.n, fast, out, [&](int i) { return iccdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }

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
    // log y at x next to the bounded end, where y = 1 - c z cancels: y = n / sigma with
    // n = sigma - c (x - mu) summed exactly. x - mu = d (two_sum) and its products with c
    // (two_prod) are exact, and the five terms are summed by distillation (passes of two_sum
    // leave the sum in the last element and its error in the ones before). For a small sigma the
    // products' low parts would be subnormal: sigma and the smaller of c and d are scaled by
    // 2^600 first (c d is next to sigma, so that factor is below 2^-200). When the leading
    // terms cancel to nothing, n is the last product c d.lo alone, taken through its logs (it
    // may be below the doubles). `sign`: of n (0 at the end itself); NaN where a term leaves
    // the double range.
    struct Edge {
        num::DD ly;
        double sign;
    };
    static Edge edge_log_y(double x, double c, double mu, double sigma) {
        num::DD d = num::dd::two_sum(x, -mu);
        double cc = c, ss = sigma;
        if (sigma < 0x1p-400) {
            ss = sigma * 0x1p600;
            if (std::fabs(c) < std::fabs(d.hi)) cc = c * 0x1p600;
            else d = num::DD{d.hi * 0x1p600, d.lo * 0x1p600};
        }
        const num::DD p = num::dd::two_prod(cc, d.hi);
        if (!std::isfinite(p.hi)) return {{0.0, 0.0}, kNaN};
        double v[3] = {-p.lo, -p.hi, ss};
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < 2; ++i) {
                const num::DD s = num::dd::two_sum(v[i], v[i + 1]);
                v[i] = s.lo, v[i + 1] = s.hi;
            }
        }
        const num::DD lead = num::dd::two_sum(v[2], v[1] + v[0]);
        const num::DD ls = num::log_dd(ss);
        if (lead.hi == 0.0) {  // n = -cc d.lo
            if (d.lo == 0.0) return {{0.0, 0.0}, 0.0};
            if ((cc > 0.0) == (d.lo > 0.0)) return {{0.0, 0.0}, -1.0};
            return {num::log_dd(std::fabs(cc)) + num::log_dd(std::fabs(d.lo)) - ls, 1.0};
        }
        // all five terms: the product with d.lo may be larger than n itself, so it enters exactly too
        const num::DD q = num::dd::two_prod(cc, d.lo);
        double w[5] = {-q.lo, -q.hi, -p.lo, -p.hi, ss};
        for (int pass = 0; pass < 4; ++pass) {
            for (int i = 0; i < 4; ++i) {
                const num::DD t = num::dd::two_sum(w[i], w[i + 1]);
                w[i] = t.lo, w[i + 1] = t.hi;
            }
        }
        const num::DD n = num::dd::two_sum(w[4], w[3] + (w[2] + (w[1] + w[0])));
        if (!(n.hi > 0.0)) return {{0.0, 0.0}, n.hi == 0.0 ? 0.0 : -1.0};
        return {num::dd::log_pos(n) - ls, 1.0};
    }
    static At at(double x, double c, double mu, double sigma) {
        const num::DD z = num::standardize(x, mu, sigma);
        if (c == 0.0) return {-z, {0.0, 0.0}, true, false};
        num::DD u;  // -c z: y = 1 + u
        const double uz = std::fabs(z.hi * c);
        if (uz >= 0x1p-900 && uz <= 0x1p1000 && std::fabs(z.hi) >= 0x1p-960 && std::isfinite(z.hi)) {
            u = -(z * c);
        } else {
            // z or c z leaves the double range (or loses bits below it): the mantissas of
            // x - mu, sigma and c apart from their exponents, u = us 2^k
            num::DD d = num::dd::two_sum(x, -mu);
            int k = 0, ed = 0, es = 0, ec = 0;
            if (!std::isfinite(d.hi)) d = num::dd::two_sum(0.5 * x, -0.5 * mu), k = 1;
            if (d.hi == 0.0) return {{0.0, 0.0}, {0.0, 0.0}, true, false};  // x = mu: t = 1
            std::frexp(d.hi, &ed);
            const double ss = std::frexp(sigma, &es), cs = std::frexp(c, &ec);
            // (the scaled low part may round where d.hi is below 2^-969: it is then below
            // 2^-1074, an error of 2^-105 of d at most)
            const num::DD us = -(num::div(num::DD{std::ldexp(d.hi, -ed), std::ldexp(d.lo, -ed)}, ss) * cs);
            k += ed - es + ec;  // |us| in (1/8, 2)
            if (k > 1010) {  // |u| beyond the range: outside the support, or log y = log u
                if (us.hi < 0.0) return {{0.0, 0.0}, {0.0, 0.0}, false, false};
                const num::DD ly = num::dd::log(us) + num::kLn2 * static_cast<double>(k);
                return {num::div(ly, c), ly, true, false};
            }
            u = k < -1000 ? num::DD{std::ldexp(us.hi, k), 0.0} : num::dd::ldexp(us, k);
        }
        if (std::fabs(1.0 + u.hi) < 0x1p-20) {
            // next to the bounded end y = 1 + u cancels: y = (sigma - c (x - mu)) / sigma with
            // the numerator summed exactly (where its terms are in the double range)
            const Edge g = edge_log_y(x, c, mu, sigma);
            if (g.sign == g.sign) {
                if (!(g.sign > 0.0)) return {{0.0, 0.0}, {0.0, 0.0}, false, g.sign == 0.0};
                return {num::div(g.ly, c), g.ly, true, false};
            }
        }
        if (u.hi < -1.0 || (u.hi == -1.0 && u.lo <= 0.0)) return {{0.0, 0.0}, {0.0, 0.0}, false, u.hi == -1.0 && u.lo == 0.0};
        if (std::fabs(u.hi) < 0x1p-60) {
            // log(1 + u) / c = -z (1 - u/2 + ...): the Gumbel's E to first order in u (the
            // quotient by a tiny c would divide a rounded u)
            if (std::isinf(z.hi)) return {-z, {u.hi, 0.0}, true, false};
            return {z * (0.5 * u.hi) - z, {u.hi, 0.0}, true, false};
        }
        num::DD ly;  // log(1 + u): u - mu(u) only for a small u (it cancels for a large one)
        if (u.hi > 1e300) ly = num::log_dd(u.hi);
        else if (std::fabs(u.hi) < 0.5) ly = u - num::dd::log1pmx_neg(u);
        else ly = num::dd::log(num::DD{1.0, 0.0} + u);
        return {num::div(ly, c), ly, true, false};
    }
    static num::DD log_density(const At& a, double sigma) {
        if (std::isinf(a.e.hi)) return {-kInf, 0.0};  // t = 0 or inf beyond the double range
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

    // ---- chunk kernels (weibull_detail::tail_chunk for the four tails) ------------------
    //
    // y = 1 - c z for the chunk as the scalar definition's double-double (1 + u, u = -(z c));
    // hi = NaN where z is not ckd::standardize's. u is good to 2^-103 of itself, so toward
    // the bounded end of the support (u near -1) y is only good to 2^-103 / y: kYErr / y in
    // kUlp, which the grades below carry into their bounds (the scalar definition has the
    // same limit: 5e-15 at y = 1e-17, bench/stats/t6/accuracy.md).
    static constexpr double kYErr = 0x1p-50;
    static void y_chunk(const Chunk<3>& c, double* yh, double* yl) {
        ckd::standardize(c.x, c.p[1], c.p[2], yh, yl, c.n);
        for (int i = 0; i < c.n; ++i) {
            const num::DD u = -(num::DD{yh[i], yl[i]} * c.p[0][i]);
            const num::DD y = num::dd::two_sum(1.0, u.hi) + u.lo;
            yh[i] = y.hi, yl[i] = y.lo;
        }
    }
    // Grade 0: t = y^(1/c) by vForce's pow (ckd::pow_dd). The rounded 1/c moves t by |log t|
    // units, and |log t| = |log y| / |c| <= |y - 1| / (min(1, y) |c|); y's own error by
    // kYErr / (y |c|): tau = kPowDd + both.
    // NaN (th and tau) outside the support's interior, for |c| < 2^-30 (c = 0 too: grade 2
    // has the Gumbel form) and where y or t leaves [2^-999, 2^999].
    static void t_pow_chunk(const Chunk<3>& c, double* yh, double* yl, double* th, double* tl, double* tau) {
        y_chunk(c, yh, yl);
        for (int i = 0; i < c.n; ++i) tl[i] = 1.0 / c.p[0][i];
        ckd::pow_dd(th, yh, yl, tl, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double ac = std::fabs(c.p[0][i]);
            const bool ok = ac >= 0x1p-30 && yh[i] >= 0x1p-999 && yh[i] <= 0x1p999 && th[i] >= 0x1p-999 && th[i] <= 0x1p999;
            tau[i] = ok ? ckd::kPowDd + (std::fabs(yh[i] - 1.0) / std::min(1.0, yh[i]) + kYErr / yh[i]) / ac : kNaN;
            th[i] = ok ? th[i] : kNaN;
            tl[i] = 0.0;
        }
    }
    // Grade 2: E = log(y) / c and log y in the scalar definition's double-doubles (c = 0 is
    // replaced afterwards: gumbel_part),
    // ckd::log_dd for its two forms of log(1 + u): that needs |u| >= 2^-30 (1 + u keeps u to
    // 2^-106 absolute, 2^-76 of log y), else NaN. E's absolute error: eu = 2 kLq |E| +
    // kYErr / (y |c|).
    static void log_t_chunk(const Chunk<3>& c, double* eh, double* el, double* eu, double* lh, double* ll) {
        double zh[kChunkSize], zl[kChunkSize];
        ckd::standardize(c.x, c.p[1], c.p[2], zh, zl, c.n);
        for (int i = 0; i < c.n; ++i) {
            const num::DD u = -(num::DD{zh[i], zl[i]} * c.p[0][i]);
            const num::DD y = num::dd::two_sum(1.0, u.hi) + u.lo;
            eh[i] = std::fabs(u.hi) >= 0x1p-30 ? y.hi : kNaN, el[i] = y.lo;
            eu[i] = kYErr / y.hi;
        }
        ckd::log_dd(eh, el, lh, ll, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double cc = c.p[0][i];
            if (cc == 0.0) {
                eh[i] = -zh[i], el[i] = -zl[i], eu[i] = 0.0;
                lh[i] = ll[i] = 0.0;
            } else {
                const num::DD e = num::div(num::DD{lh[i], ll[i]}, cc);
                eh[i] = e.hi, el[i] = e.lo;
                eu[i] = 2.0 * ckd::kLq * std::fabs(e.hi) + eu[i] / std::fabs(cc);
            }
        }
    }
    // The elements with c = 0 are the Gumbel distribution's, to the bit (T26): its kernel
    // `gumbel(sub-chunk, out)` on them, gathered, whatever the kernels here gave. Returns
    // true when that was the whole chunk (a broadcast c = 0: nothing else to compute).
    template <class G>
    static bool gumbel_all(const Chunk<3>& c, double* out, G&& gumbel) {
        if (!(c.same[0] && c.p[0][0] == 0.0)) return false;
        gumbel(Chunk<2>{c.n, c.x, {c.p[1], c.p[2]}, {c.same[1], c.same[2]}}, out);
        return true;
    }
    template <class G>
    static void gumbel_part(const Chunk<3>& c, double* out, bool* fast, G&& gumbel) {
        if (c.same[0]) return;  // c != 0 throughout (gumbel_all took c = 0)
        int idx[kChunkSize];
        int m = 0;
        for (int i = 0; i < c.n; ++i) {
            idx[m] = i;
            m += c.p[0][i] == 0.0 ? 1 : 0;
        }
        if (m == 0) return;
        double xb[kChunkSize], mb[kChunkSize], sb[kChunkSize], ob[kChunkSize];
        for (int j = 0; j < m; ++j) xb[j] = c.x[idx[j]], mb[j] = c.p[1][idx[j]], sb[j] = c.p[2][idx[j]];
        gumbel(Chunk<2>{m, xb, {mb, sb}, {false, false}}, ob);
        for (int j = 0; j < m; ++j) out[idx[j]] = ob[j], fast[idx[j]] = true;
    }
    static void tail_kernel(const Chunk<3>& c, double* out, bool* fast, weibull_detail::Tail kind, int grade) {
        double eh[kChunkSize], el[kChunkSize], tau[kChunkSize], th[kChunkSize], tl[kChunkSize];
        if (grade == 0) {
            t_pow_chunk(c, eh, el, th, tl, tau);
        } else {
            log_t_chunk(c, eh, el, tau, th, tl);
            weibull_detail::t_chunk(eh, el, tau, true, th, tl, tau, c.n);
        }
        weibull_detail::tail_chunk(kind, th, tl, tau, out, eh, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = eh[i] <= kFastBudget;  // false for NaN
    }
    static void tail_run(const Chunk<3>& c, double* out, weibull_detail::Tail kind, bool upper, bool log) {
        const auto gumbel = [&](const Chunk<2>& g, double* o) { Gumbel::tail_run(g, o, kind, upper, log); };
        if (gumbel_all(c, out, gumbel)) return;
        bool fast[kChunkSize];
        tail_kernel(c, out, fast, kind, 0);
        ckd::retry(c, out, fast, [&](const Chunk<3>& s, double* o, bool* f) { tail_kernel(s, o, f, kind, 2); });
        gumbel_part(c, out, fast, gumbel);
        ck::fallback(c.n, fast, out, [&](int i) { return tail(c.x[i], c.p[0][i], c.p[1][i], c.p[2][i], upper, log); });
    }
    static void cdf_chunk(const Chunk<3>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kExp, false, false); }
    static void ccdf_chunk(const Chunk<3>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kOneExp, true, false); }
    static void logcdf_chunk(const Chunk<3>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogExp, false, true); }
    static void logccdf_chunk(const Chunk<3>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogOne, true, true); }
    // The density.
    //   0: pdf = t e^-t / (y sigma) in double (y^(1/c - 1) = t / y): t's error reaches it times
    //      |1 - t|, y's own is kYErr / y, then exp and three operations: tau |1 - t| +
    //      kYErr / y + kVfExp + 3; logpdf its log by vForce (kVfLog |value| more, against the
    //      mixed bound of T20).
    //   2: E - log y - t - log sigma in the scalar definition's double-doubles, t folded:
    //      absolute error t tau, E's (at most tau: tau contains eu), log y's (kLq |log y|, and
    //      y's own kYErr / y <= |c| eu) and the last bit; pdf adds kVfExp + 1.
    static void density_kernel(const Chunk<3>& c, double* out, bool* fast, bool log, int grade) {
        double eh[kChunkSize], el[kChunkSize], tau[kChunkSize], th[kChunkSize], tl[kChunkSize], lh[kChunkSize],
            ll[kChunkSize];
        if (grade == 0) {
            t_pow_chunk(c, eh, el, th, tl, tau);  // eh: y
            for (int i = 0; i < c.n; ++i) el[i] = -th[i];
            ck::exp(el, el, c.n);
            for (int i = 0; i < c.n; ++i) {
                const double m = th[i] * el[i];
                out[i] = (m / eh[i]) / c.p[2][i];
                tau[i] = tau[i] * std::fabs(1.0 - th[i]) + kYErr / eh[i] + ck::kVfExp + 3.0;
                fast[i] = m >= 0x1p-900 && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;
            }
            if (log) ck::log(out, out, c.n);
            for (int i = 0; i < c.n; ++i) {
                const double v = std::fabs(out[i]);
                fast[i] = fast[i] && (log ? tau[i] + ck::kVfLog * v <= kFastBudget * std::max(1.0, v) : tau[i] <= kFastBudget);
            }
            return;
        }
        log_t_chunk(c, eh, el, tau, lh, ll);
        weibull_detail::t_chunk(eh, el, tau, true, th, tl, tau, c.n);
        double* kh = out;  // log sigma's high part, until the result replaces it
        double kl[kChunkSize];
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[2][i]); });
        for (int i = 0; i < c.n; ++i) {
            const num::DD l = num::DD{eh[i], el[i]} - num::DD{lh[i], ll[i]} - num::DD{th[i], tl[i]} - num::DD{kh[i], kl[i]};
            const double abs = th[i] * tau[i] + tau[i] * (1.0 + std::fabs(c.p[0][i])) + ckd::kLq * std::fabs(lh[i]) + 1.0;
            if (log) {
                out[i] = l.hi + l.lo;
                fast[i] = out[i] == out[i] && abs <= kFastBudget * std::max(1.0, std::fabs(out[i]));
            } else {
                eh[i] = l.hi, el[i] = l.lo;
                tau[i] = abs + ck::kVfExp + 1.0;
            }
        }
        if (log) return;
        ckd::exp_dd(out, eh, el, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = tau[i] <= kFastBudget && out[i] >= 0x1p-1000;
    }
    static void density_run(const Chunk<3>& c, double* out, bool log) {
        const auto gumbel = [&](const Chunk<2>& g, double* o) { Gumbel::density_run(g, o, log); };
        if (gumbel_all(c, out, gumbel)) return;
        bool fast[kChunkSize];
        density_kernel(c, out, fast, log, 0);
        ckd::retry(c, out, fast, [&](const Chunk<3>& s, double* o, bool* f) { density_kernel(s, o, f, log, 2); });
        gumbel_part(c, out, fast, gumbel);
        ck::fallback(c.n, fast, out, [&](int i) {
            return log ? logpdf(c.x[i], c.p[0][i], c.p[1][i], c.p[2][i]) : pdf(c.x[i], c.p[0][i], c.p[1][i], c.p[2][i]);
        });
    }
    static void pdf_chunk(const Chunk<3>& c, double* out) { density_run(c, out, false); }
    static void logpdf_chunk(const Chunk<3>& c, double* out) { density_run(c, out, true); }

    // z = -expm1(c lt) / c (-lt at c = 0), x = mu + sigma z.
    static num::DD z_of(num::DD lt, double c) {
        if (c == 0.0) return -lt;
        if (std::isinf(lt.hi)) {  // t = 0 or inf: an end of the support
            const bool to_end = (lt.hi < 0.0) == (c > 0.0);
            return to_end ? num::div(num::DD{1.0, 0.0}, c) : num::DD{c > 0.0 ? -kInf : kInf, 0.0};
        }
        const double a = lt.hi * c;
        // -expm1(a) / c = -lt (1 + a/2 + ...): to first order for a tiny a (which may be
        // subnormal: the quotient by c would divide a rounded product)
        if (std::fabs(a) < 0x1p-60) return -lt - lt * (0.5 * a);
        if (a > 710.0) return {c > 0.0 ? -kInf : kInf, 0.0};
        if (a < -750.0) return num::div(num::DD{1.0, 0.0}, c);
        const num::DD m = extreme_detail::expm1_any(lt * c);
        if (std::isinf(m.hi)) return {c > 0.0 ? -kInf : kInf, 0.0};
        return -num::div(m, c);
    }
    static double from_lt(num::DD lt, double c, double mu, double sigma) {
        const double a = lt.hi * c;
        if (c != 0.0 && (std::isinf(lt.hi) || a < -750.0)) {
            // t = 0 or inf, or e^(c lt) below half an ULP of 1: the bounded end, or beyond the range
            if (a < 0.0) return end(c, mu, sigma);
            return c > 0.0 ? -kInf : kInf;
        }
        const num::DD cl = std::fabs(a) <= 1e300 ? lt * c : num::DD{a, 0.0};
        if (c != 0.0 && a < -38.0 && std::isfinite(sigma / c)) {
            // e^(c lt) is below half an ULP of 1: x = (mu + sigma / c) - (sigma / c) e^(c lt), the
            // small term through the logs (alone it may be subnormal where its product is not)
            const num::DD r = num::div(num::DD{sigma, 0.0}, c);
            const double w = num::exp_dd(cl + num::log_dd(sigma) - num::log_dd(std::fabs(c)));
            const num::DD e = r + mu;
            const double v = e.hi + (e.lo - (c > 0.0 ? w : -w));
            return v == v ? v : r.hi + mu;
        }
        if (c != 0.0 && std::isfinite(cl.hi) && cl.hi > 700.0) {
            // z = -(e^cl - 1) / c ~ -e^cl / c overflows while sigma z may not: through the logs
            const num::DD l = cl + num::log_dd(sigma) - num::log_dd(std::fabs(c));
            const double v = num::exp_dd(l);
            return c > 0.0 ? mu - v : mu + v;
        }
        return extreme_detail::affine(mu, sigma, z_of(lt, c));
    }
    // x = mu + sigma z, z = -expm1(a) / c, a = c log t, in double from t with relative error
    // tu: log t by vForce (absolute tu + kVfLog |log t|), a (|a| more for the product), m =
    // expm1(a) by libm (a's absolute error times e^a, and 2 |m|), the quotient and the fma:
    //     dz = (tu + (kVfLog + 1) |log t|) e^a + 2 |m| / |c| + |z|     (absolute, in kUlp)
    // and x is off by sigma dz. Taken where that is inside the budget relative to x, and for
    // p in [0.01, 0.99] where dz times r e^-a is: the quantile's condition number is kappa =
    // sigma e^a / (r |x|) with r = t (icdf, t = -log p) or (1 - p) t / p (iccdf, t = -log(1 -
    // p)), so that is the error in units of the contract's max(1, kappa) (§2.3; r e^-a <
    // sigma / |x| when kappa < 1). c = 0 is the Gumbel's kernel (gumbel_all, gumbel_part).
    static void quantile_chunk(const Chunk<3>& c, double* out, bool upper) {
        const auto gumbel = [&](const Chunk<2>& g, double* o) { upper ? Gumbel::iccdf_chunk(g, o) : Gumbel::icdf_chunk(g, o); };
        if (gumbel_all(c, out, gumbel)) return;
        double t[kChunkSize];
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) t[i] = upper ? 1.0 - c.x[i] : c.x[i];
        ck::log(t, t, c.n);
        for (int i = 0; i < c.n; ++i) t[i] = upper && c.x[i] < 0.25 ? -std::log1p(-c.x[i]) : -t[i];
        ck::log(out, t, c.n);
        const double tu = upper ? 7.5 : ck::kVfLog;  // as Gumbel's
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i], cc = c.p[0][i], mu = c.p[1][i], sigma = c.p[2][i];
            const double lt = out[i], a = cc * lt;
            const double m = std::expm1(a), ea = 1.0 + m;
            const double z = -m / cc;
            out[i] = std::fma(sigma, z, mu);
            const double dz = (tu + (ck::kVfLog + 1.0) * std::fabs(lt)) * ea + 2.0 * std::fabs(m / cc) + std::fabs(z);
            const double r = upper ? (1.0 - p) * t[i] / p : t[i];
            const bool central = p >= 0.01 && p <= 0.99 && dz * r <= (kFastBudget - 1.0) * ea;
            // (a product c log t in the subnormals has lost bits that the quotient by c would magnify)
            fast[i] = p >= 0x1p-1000 && p < 1.0 && std::fabs(a) < 700.0 && std::fabs(a) >= 0x1p-1021 &&
                      (central || sigma * dz <= (kFastBudget - 1.0) * std::fabs(out[i]));  // false for NaN, c = 0
        }
        ckd::retry(c, out, fast, [&](const Chunk<3>& s, double* o, bool* f) { quantile_dd_kernel(s, o, f, upper); });
        gumbel_part(c, out, fast, gumbel);
        ck::fallback(c.n, fast, out, [&](int i) {
            return upper ? iccdf(c.x[i], c.p[0][i], c.p[1][i], c.p[2][i]) : icdf(c.x[i], c.p[0][i], c.p[1][i], c.p[2][i]);
        });
    }
    // The quantile's second grade, for a cancelling mu + sigma z outside the first one's reach:
    // the scalar definition's double-doubles with ckd::log_dd for dd::log: log t = log(-log q),
    // q = p or 1 - p (exact as a double-double); a = c log t; m = expm1(a) by
    // extreme_detail::expm1_any's Newton step on libm's value (m - (log(1 + m) - a) (1 + m),
    // its form for 2^-20 <= |a| <= 38; outside: the scalar definition); z = -m / c;
    // mu + sigma z. The three logs leave a off by 3 kLq |a| and m by 4 kLq |a| e^a (absolute),
    // so x by dx = 4 kLq |a| e^a sigma / |c|, plus its last two roundings: taken where dx is
    // inside the budget less 2, relative to x.
    static void quantile_dd_kernel(const Chunk<3>& c, double* out, bool* fast, bool upper) {
        using num::DD;
        double th[kChunkSize], tl[kChunkSize], lh[kChunkSize], ll[kChunkSize], ah[kChunkSize], al[kChunkSize],
            m0[kChunkSize];
        for (int i = 0; i < c.n; ++i) {
            const DD q = upper ? num::dd::two_sum(1.0, -c.x[i]) : DD{c.x[i], 0.0};
            th[i] = q.hi, tl[i] = q.lo;
        }
        ckd::log_dd(th, tl, lh, ll, c.n);
        for (int i = 0; i < c.n; ++i) th[i] = -lh[i], tl[i] = -ll[i];  // t
        ckd::log_dd(th, tl, lh, ll, c.n);                               // log t
        for (int i = 0; i < c.n; ++i) {
            const DD a = DD{lh[i], ll[i]} * c.p[0][i];
            const bool ok = std::fabs(a.hi) >= 0x1p-20 && std::fabs(a.hi) <= 38.0;  // false for NaN
            ah[i] = ok ? a.hi : kNaN, al[i] = a.lo;
            m0[i] = std::expm1(ah[i]);
            const DD one = num::dd::two_sum(1.0, m0[i]);
            th[i] = one.hi, tl[i] = one.lo;
        }
        ckd::log_dd(th, tl, lh, ll, c.n);  // log(1 + m0)
        for (int i = 0; i < c.n; ++i) {
            const double cc = c.p[0][i], mu = c.p[1][i], sigma = c.p[2][i];
            const DD m = DD{m0[i], 0.0} - (DD{lh[i], ll[i]} - DD{ah[i], al[i]}) * DD{th[i], tl[i]};
            const DD z = -num::div(m, cc);
            out[i] = std::fma(sigma, z.hi, mu) + sigma * z.lo;
            const double dx = 4.0 * ckd::kLq * std::fabs(ah[i]) * (1.0 + m0[i]) * sigma / std::fabs(cc);
            fast[i] = c.x[i] > 0.0 && c.x[i] < 1.0 && dx <= (kFastBudget - 2.0) * std::fabs(out[i]);  // false for NaN
        }
    }
    static void icdf_chunk(const Chunk<3>& c, double* out) { quantile_chunk(c, out, false); }
    static void iccdf_chunk(const Chunk<3>& c, double* out) { quantile_chunk(c, out, true); }
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
        if (std::fabs(c) < 0x1p-60) {  // euler - (euler^2 / 2 + pi^2 / 12) c: log Gamma(1 + c) is subnormal for a tiny c
            return extreme_detail::affine(mu, sigma, extreme_detail::kEuler + (-0.9890559953279725 * c));
        }
        if (c > 1e17) return -kInf;  // -sigma Gamma(1 + c) / c beyond the range for every sigma
        const num::DD m = -num::div(extreme_detail::expm1_any(extreme_detail::lgamma1p(c)), c);
        return extreme_detail::affine(mu, sigma, m);
    }
    // sigma^2 Gamma(1 + c)^2 E_2 / c^2 (pi^2/6 at c = 0); +inf for -1 < c <= -1/2, NaN below.
    static num::DD var_std(double c) {
        if (c == 0.0) return extreme_detail::kPi2Over6;
        // pi^2/6 - 2 (zeta(3) + euler pi^2/6) c: E_2 / c^2 underflows for a tiny c
        if (std::fabs(c) < 0x1p-60) return extreme_detail::kPi2Over6 + (-4.303077228549152 * c);
        const num::DD e2 = num::expm1_full(weibull_detail::lgamma_ratio_x(2, num::DD{c, 0.0}));
        return num::exp_full(extreme_detail::lgamma1p(c) * 2.0) * num::div(num::div(e2, c), c);
    }
    // log of the variance for c > 20, where Gamma(1 + 2c) leaves the range: 2 log sigma +
    // log Gamma(1 + 2c) + log(1 - e^-D_2) - 2 log c; +inf past c = 1e17 (for every sigma).
    static num::DD log_var_large(double c, double sigma) {
        if (c > 1e17) return {kInf, 0.0};
        const num::DD d2 = weibull_detail::lgamma_ratio_x(2, num::DD{c, 0.0});
        return num::log_gamma(num::dd::two_prod(2.0, c) + 1.0) + (num::log_dd(sigma) - num::log_dd(c)) * 2.0 +
               std::log(-std::expm1(-d2.hi));
    }
    static double variance(double c, double, double sigma) {
        if (!(2.0 * c > -1.0)) return c > -1.0 ? kInf : kNaN;
        if (c > 20.0) return num::exp_dd(log_var_large(c, sigma));
        const num::DD v = var_std(c);
        const double w = v.hi + v.lo, s2 = sigma * sigma;
        // sigma^2 alone may leave the range where the product does not
        return (s2 >= 0x1p-900 && std::isfinite(s2)) ? s2 * w : sigma * (sigma * w);
    }
    static double standard_deviation(double c, double, double sigma) {
        if (!(2.0 * c > -1.0)) return c > -1.0 ? kInf : kNaN;
        if (c > 1e17) return kInf;
        if (c > 20.0) return num::exp_dd(log_var_large(c, sigma) * 0.5);
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
    // The excess kurtosis of W (weibull_detail::kurtosis_x, with its series about c = 0).
    static double kurtosis(double c, double, double) {
        if (!(4.0 * c > -1.0)) return 2.0 * c > -1.0 ? kInf : kNaN;
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
        const num::DD r = num::div(num::DD{sigma, 0.0}, c);  // (1 / c may overflow where sigma / c does not)
        if (std::isinf(r.hi)) return r.hi;
        const num::DD e = r + mu;
        const double v = e.hi + e.lo;
        return v == v ? v : r.hi + mu;  // the sum overflowed (its low part is NaN): +-inf
    }
    static double support_lo(double c, double mu, double sigma) { return c < 0.0 ? end(c, mu, sigma) : -kInf; }
    static double support_hi(double c, double mu, double sigma) { return c > 0.0 ? end(c, mu, sigma) : kInf; }
};

}  // namespace sd::stats
