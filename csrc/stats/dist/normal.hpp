// The normal distribution N(mu, sigma^2) (docs/api/stats.md §2.1): sd.special's
// ndtr, log_ndtr, ndtri and ndtri_exp at the standardized z, which is a
// double-double (stats/numeric.hpp): the cdfs at z.hi corrected by
// phi(z.hi) z.lo, the density's exponent z^2 / 2 + log(sigma sqrt(2 pi)) in
// double-double, exponentiated once.
#pragma once

#include "special/ops/distr.hpp"
#include "special/ops/erf.hpp"
#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace normal_detail {

using num::DD;

inline double ndtr(double z) { return special::distr_detail::one<special::Ndtr>(z); }
inline double log_ndtr(double z) { return special::distr_detail::one<special::LogNdtr>(z); }
inline double ndtri(double p) { return special::distr_detail::one<special::Ndtri>(p); }

// phi(z) to a few ULP (it only scales the first-order correction).
inline double phi(double z) { return num::kInvSqrt2Pi * std::exp(-0.5 * z * z); }

// Past this |z| the density is 0 and the exponent beyond any double-double care.
inline constexpr double kHugeZ = 1e10;

// z^2 / 2 + log(sigma sqrt(2 pi)): minus the log density, a double-double.
inline DD neg_log_density(DD z, double sigma) {
    thread_local num::Memo memo;
    const DD c = memo.get(sigma, 0.0, [&] { return num::log_dd(sigma) + num::kLogSqrt2Pi; });
    return (z * z) * 0.5 + c;
}

// P(Z <= z) (upper: P(Z > z)) at a double-double z, the smaller side by ndtr directly.
inline double tail_z(DD z, bool upper) {
    if (upper) z = -z;
    if (std::isinf(z.hi)) return z.hi > 0.0 ? 1.0 : 0.0;
    const double v = ndtr(z.hi);
    return z.lo == 0.0 ? v : v + phi(z.hi) * z.lo;
}

// log P(Z <= z) (upper: log P(Z > z)); d log ndtr(z) / dz = phi(z) / ndtr(z).
inline double log_tail_z(DD z, bool upper) {
    if (upper) z = -z;
    if (std::isinf(z.hi)) return z.hi > 0.0 ? 0.0 : -kInf;
    const double v = log_ndtr(z.hi);
    if (z.lo == 0.0 || std::fabs(z.hi) > kHugeZ) return v;
    return v + std::exp(-0.5 * z.hi * z.hi - num::kLogSqrt2Pi.hi - v) * z.lo;
}

// z with P(Z <= z) = p (upper: P(Z > z) = p) as a double-double. ndtri's z
// carries a few ULP of relative error, which mu + sigma z multiplies by
// |sigma z / x| when it cancels and exp(mu + sigma z) by |sigma z| (the
// log-normal); the conditioning in p is only ~1/|z| in the tails. So one Newton
// step on the log of the smaller side q follows: dz = (log q - log ndtr(z)) /
// (phi / ndtr)(z), which leaves an error ~ULP / |z| (log q and log ndtr(z) are
// each within an ULP of their size, and the slope is ~|z|).
// log ndtr(z) as a double-double for z <= -1: -z^2/2 exactly, plus
// log(erfcx(-z / sqrt 2) / 2) (of size log|z|, to an ULP). log_ndtr's own value
// is within a few ULP of |log ndtr| (~700 in the tail), too coarse for the step.
inline DD log_ndtr_dd(double z) {
    return num::dd::two_prod(z, z) * -0.5 + std::log(0.5 * special::Erfcx::f(-z * special::kSqrtHalf));
}

// The smaller side q <= 1/2 comes with lq = log q (a double-double); q = 0 when
// only lq is known.
inline DD quantile_small(double q, DD lq, bool flip) {
    const double z = q >= 0x1p-1000 ? ndtri(q) : special::NdtriExp::f(lq.hi);  // <= 0
    if (std::isinf(z)) return {flip ? -z : z, 0.0};
    // |z| beyond 2^500 (log q below -2^1000): the asymptote's fixed point is the answer (its
    // next term is 15 / z^6 below the last bit), and z^2 / 2 in the step would overflow
    if (lq.hi < -0x1p1000) return {flip ? -z : z, 0.0};
    DD lz;
    if (z <= -1.0) lz = log_ndtr_dd(z);
    else lz = {log_ndtr(z), 0.0};
    // phi / ndtr: for z <= -1 as 2 / (sqrt(2 pi) erfcx(-z / sqrt 2)), the two exponents -z^2/2
    // cancelled by hand (their difference in floating point is lost for large |z|)
    const double slope = z <= -1.0 ? 2.0 * num::kInvSqrt2Pi / special::Erfcx::f(-z * special::kSqrtHalf)
                                   : std::exp(-0.5 * z * z - num::kLogSqrt2Pi.hi - lz.hi);
    const DD d = lq - lz;
    const DD zz = num::dd::quick_two_sum(z, (d.hi + d.lo) / slope);
    return flip ? -zz : zz;
}

// The z for icdf (upper: iccdf) at p in [0, 1].
inline DD quantile_z(double p, bool upper) {
    if (p == 1.0) return {upper ? -kInf : kInf, 0.0};
    if (p == 0.0) return {upper ? kInf : -kInf, 0.0};
    if (p > 0.5) return quantile_small(1.0 - p, num::log_dd(1.0 - p), !upper);
    return quantile_small(p, num::log_dd(p), upper);
}

// The z for ilogcdf (upper: ilogccdf) at lp <= 0.
inline DD quantile_z_log(double lp, bool upper) {
    if (lp > -num::kLn2.hi) {
        const double q = -std::expm1(lp);  // the other side
        if (q == 0.0) return {upper ? -kInf : kInf, 0.0};
        return quantile_small(q, num::log_dd(q), !upper);
    }
    return quantile_small(0.0, DD{lp, 0.0}, upper);
}

// The quantile z (icdf, upper: iccdf) without the Newton step when it would not
// matter: ndtri's few ULP reach x = mu + sigma z multiplied by |sigma z / x|, so with
// that factor at most 4 (no cancellation; at most ~22 ULP) z.hi alone is enough.
inline bool needs_step(double mu, double sigma, double z) {
    return std::fabs(sigma * z) > 4.0 * std::fabs(std::fma(sigma, z, mu)) || std::fabs(z) > 30.0;
}

// mu + sigma z in double-double, rounded once.
inline double locate(double mu, double sigma, DD z) {
    if (std::isinf(z.hi)) return z.hi * sigma;
    if (!std::isfinite(sigma * z.hi)) {
        // sigma z overflows; mu + sigma z may not (opposite signs): half of each, doubled
        const double h = 0.5 * sigma;
        if (!std::isfinite(h * z.hi)) return std::copysign(kInf, z.hi);
        const DD x = num::dd::two_prod(h, z.hi) + h * z.lo + 0.5 * mu;
        return 2.0 * (x.hi + x.lo);
    }
    const DD x = num::dd::two_prod(sigma, z.hi) + sigma * z.lo + mu;
    const double r = x.hi + x.lo;
    return r == r ? r : sigma * z.hi + mu;  // the sum overflowed (its low part is NaN): +-inf
}

// ---- chunk kernels on a standardized z given as hi/lo arrays (the normal's own z, and the
// log-normal's) ----------------------------------------------------------------------------

// tail_z for the chunk: special's ndtr kernel at z.hi (the scalar definition calls the same
// kernel with n = 1) plus phi(z.hi) z.lo, phi through vForce's exp. Only the correction
// differs from the scalar definition, by kVfExp + 1 of itself, and it is at most
// |z| phi / Phi * 2^-53 <= (z^2 + 1) 2^-53 of the result: the two agree to the last bit or
// differ in it (bound: 1). z is used as given (negate it for the upper tail); out is NaN
// where z.hi is.
inline void tail_z_chunk(const double* zh, const double* zl, double* out, int n) {
    double e[kChunkSize];
    special::Ndtr::chunk(zh, out, n);
    for (int i = 0; i < n; ++i) e[i] = -0.5 * zh[i] * zh[i];
    ck::exp(e, e, n);
    for (int i = 0; i < n; ++i) out[i] += (num::kInvSqrt2Pi * e[i]) * zl[i];
}

// log_tail_z for the chunk. With t = ndtr(-|z|) (the kernel; relative error e_t):
//   z >= 0: log1p(-t), the scalar definition's own form (the same bits);
//   z < 0:  log t by vForce instead of -z^2/2 + log(erfcx(-z / sqrt 2) / 2): e_t / |log t| +
//           kVfLog with |log t| >= log 2. ndtr's e_t is largest where 1/2 + erf/2 cancels
//           (-1.41 < z < 0): |erf| / (2 t) times erf's error (Cephes' rational, 3 units)
//           plus 1, which over |log t| is at most 7 (at z = -1.41); beyond, erfc's few
//           units over |log t| > 2.5. In all at most 11; measured against the scalar
//           definition 4.4 (bench/stats/opt/a). ok[i] is false where t is below 2^-1000
//           (the scalar form's region) or z.hi is NaN.
// Then the first-order correction (phi / Phi)(z.hi) z.lo as the scalar definition's, its
// exp by vForce (a few units of a term that is itself 2^-53 (z^2 + 1) of the result).
inline void log_tail_z_chunk(const double* zh, const double* zl, double* out, bool* ok, int n) {
    double t[kChunkSize], e[kChunkSize];
    for (int i = 0; i < n; ++i) e[i] = -std::fabs(zh[i]);
    special::Ndtr::chunk(e, t, n);
    ck::log(out, t, n);
    for (int i = 0; i < n; ++i) {
        if (zh[i] >= 0.0) out[i] = std::log1p(-t[i]);
        ok[i] = t[i] >= 0x1p-1000;  // false for NaN
        e[i] = -0.5 * zh[i] * zh[i] - num::kLogSqrt2Pi.hi - out[i];
    }
    ck::exp(e, e, n);
    for (int i = 0; i < n; ++i) out[i] += e[i] * zl[i];
}

}  // namespace normal_detail

struct Normal {
    static constexpr char name[] = "normal";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    static double pdf(double x, double mu, double sigma) {
        using namespace normal_detail;
        if (std::isinf(x)) return 0.0;
        const DD z = num::standardize(x, mu, sigma);
        if (std::fabs(z.hi) > kHugeZ) return 0.0;
        return num::exp_dd(-neg_log_density(z, sigma));
    }
    static double logpdf(double x, double mu, double sigma) {
        using namespace normal_detail;
        if (std::isinf(x)) return -kInf;
        const DD z = num::standardize(x, mu, sigma);
        if (std::fabs(z.hi) > kHugeZ) return -0.5 * z.hi * z.hi - std::log(sigma) - num::kLogSqrt2Pi.hi;
        const DD e = neg_log_density(z, sigma);
        return -(e.hi + e.lo);
    }

    // -(z^2 / 2 + log(sigma sqrt(2 pi))) for the chunk as hi/lo arrays: the scalar definition's
    // double-double arithmetic (ckd::standardize, the same products and sums), so the exponent
    // has the same bits. hi = NaN where the scalar definition takes another form.
    static void exponent_chunk(const Chunk<2>& c, double* eh, double* el) {
        using namespace normal_detail;
        double kh[kChunkSize], kl[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], eh, el, c.n);
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[1][i]) + num::kLogSqrt2Pi; });
        for (int i = 0; i < c.n; ++i) {
            const DD z{eh[i], el[i]};
            const DD e = -((z * z) * 0.5 + DD{kh[i], kl[i]});
            const bool ok = std::fabs(z.hi) <= kHugeZ;  // false for NaN
            eh[i] = ok ? e.hi : kNaN, el[i] = e.lo;
        }
    }
    // Grade 0, in double: z = (x - mu) / sigma (2 units), z^2 (5), the exponent -z^2 / 2
    // (2.5 z^2 absolute), exp by vForce and the product with 1 / (sigma sqrt(2 pi)) (itself 1.5):
    // 2.5 z^2 + kVfExp + 3.5, inside the budget up to |z| = 2.
    // Grade 1, for the rest: exp of the double-double exponent by vForce, the scalar definition
    // with libm's exp replaced: relative error kVfExp + 1 whatever z (the exponent is exact to
    // 2^-100). Results below 2^-1000 (where the product with 1 + lo may round a second time
    // into the subnormals) are left to it.
    static void pdf_kernel(const Chunk<2>& c, double* out, bool* fast, int grade) {
        double eh[kChunkSize], el[kChunkSize];
        if (grade == 0) {
            for (int i = 0; i < c.n; ++i) {
                const double z = (c.x[i] - c.p[0][i]) / c.p[1][i];
                el[i] = z * z;
                eh[i] = -0.5 * el[i];
            }
            ck::exp(out, eh, c.n);
            for (int i = 0; i < c.n; ++i) {
                const double k = num::kInvSqrt2Pi / c.p[1][i];
                out[i] *= k;
                fast[i] = 2.5 * el[i] + ck::kVfExp + 3.5 <= kFastBudget && k >= 0x1p-1000 && k <= 0x1p1000;  // false for NaN
            }
            return;
        }
        exponent_chunk(c, eh, el);
        ckd::exp_dd(out, eh, el, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = out[i] >= 0x1p-1000;
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) {
        bool fast[kChunkSize];
        pdf_kernel(c, out, fast, 0);
        ckd::retry(c, out, fast, [&](const Chunk<2>& s, double* o, bool* f) { pdf_kernel(s, o, f, 1); });
        ck::fallback(c.n, fast, out, [&](int i) { return pdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // The same bits as the scalar definition (no elementary function per element).
    static void logpdf_chunk(const Chunk<2>& c, double* out) {
        double eh[kChunkSize], el[kChunkSize];
        bool fast[kChunkSize];
        exponent_chunk(c, eh, el);
        for (int i = 0; i < c.n; ++i) {
            out[i] = eh[i] + el[i];
            fast[i] = eh[i] == eh[i];
        }
        ck::fallback(c.n, fast, out, [&](int i) { return logpdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }

    static double tail(double x, double mu, double sigma, bool upper) {
        if (std::isinf(x)) return (x > 0.0) == upper ? 0.0 : 1.0;
        return normal_detail::tail_z(num::standardize(x, mu, sigma), upper);
    }
    static double cdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false); }
    static double ccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true); }
    static double log_tail(double x, double mu, double sigma, bool upper) {
        if (std::isinf(x)) return (x > 0.0) == upper ? -kInf : 0.0;
        return normal_detail::log_tail_z(num::standardize(x, mu, sigma), upper);
    }
    static double logcdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, false); }
    static double logccdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, true); }

    static void tail_chunk(const Chunk<2>& c, double* out, bool upper) {
        double zh[kChunkSize], zl[kChunkSize];
        bool fast[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], zh, zl, c.n);
        if (upper) {
            for (int i = 0; i < c.n; ++i) zh[i] = -zh[i], zl[i] = -zl[i];
        }
        normal_detail::tail_z_chunk(zh, zl, out, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = zh[i] == zh[i];
        ck::fallback(c.n, fast, out, [&](int i) { return tail(c.x[i], c.p[0][i], c.p[1][i], upper); });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true); }
    static void log_tail_chunk(const Chunk<2>& c, double* out, bool upper) {
        double zh[kChunkSize], zl[kChunkSize];
        bool fast[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], zh, zl, c.n);
        if (upper) {
            for (int i = 0; i < c.n; ++i) zh[i] = -zh[i], zl[i] = -zl[i];
        }
        normal_detail::log_tail_z_chunk(zh, zl, out, fast, c.n);
        ck::fallback(c.n, fast, out, [&](int i) { return log_tail(c.x[i], c.p[0][i], c.p[1][i], upper); });
    }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { log_tail_chunk(c, out, false); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { log_tail_chunk(c, out, true); }

    static double icdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p > 0.0 && p < 1.0) {
            const double z = normal_detail::ndtri(p);
            if (!normal_detail::needs_step(mu, sigma, z)) return std::fma(sigma, z, mu);
        }
        return normal_detail::locate(mu, sigma, normal_detail::quantile_z(p, false));
    }
    static double iccdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p > 0.0 && p < 1.0) {
            const double z = -normal_detail::ndtri(p);
            if (!normal_detail::needs_step(mu, sigma, z)) return std::fma(sigma, z, mu);
        }
        return normal_detail::locate(mu, sigma, normal_detail::quantile_z(p, true));
    }
    // The scalar definition's first branch for the chunk (special's ndtri kernel, then
    // mu + sigma z by fma where no Newton step is needed): the same bits.
    //
    // Where mu + sigma z cancels and the smaller side q = min(p, 1 - p) has z0 = ndtri(q) in
    // [-1, 0], the Newton step is taken in the chunk in linear form: z = z0 + (q - t) / phi(z0)
    // with t = ndtr(z0) (the kernel; q - t is exact, the two being within a few ULP, and
    // log(q / t) = (q - t) / t to 2^-100). The scalar definition steps on the logs (log q -
    // log_ndtr(z0)) instead; both leave t's relative error e_t (at most 7.5 units on [-1, 0]:
    // 3 of erf's times |erf| / (2 t) <= 2.15, plus 1) times t / phi in z, that is e_t times
    // q sigma / phi in x, and the quantile's condition number is kappa = p sigma / (phi |x|)
    // >= q sigma / (phi |x|): within e_t + 2 of the contract's max(1, kappa) (docs/design/
    // stats.md §2.3). The rest (p = 0, 1, outside [0, 1], cancellation with z0 < -1, |z| > 30)
    // goes to the scalar definition.
    static void quantile_chunk(const Chunk<2>& c, double* out, bool upper) {
        double z[kChunkSize], q[kChunkSize], t[kChunkSize];
        int idx[kChunkSize];
        bool fast[kChunkSize];
        special::Ndtri::chunk(c.x, z, c.n);
        int m = 0;
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i], mu = c.p[0][i], sigma = c.p[1][i];
            const double zi = upper ? -z[i] : z[i];
            fast[i] = p > 0.0 && p < 1.0 && !normal_detail::needs_step(mu, sigma, zi);
            out[i] = std::fma(sigma, zi, mu);
            const bool newton = !fast[i] && p > 0.0 && p < 1.0 && std::fabs(z[i]) <= 1.0;
            idx[m] = i;
            m += newton ? 1 : 0;
        }
        if (m > 0) {
            for (int j = 0; j < m; ++j) {
                const double p = c.x[idx[j]];
                q[j] = p > 0.5 ? 1.0 - p : p;
                z[j] = -std::fabs(z[idx[j]]);  // z0 = ndtri(q): ndtri is odd about 1/2 to the last bit
            }
            special::Ndtr::chunk(z, t, m);
            for (int j = 0; j < m; ++j) t[j] = q[j] - t[j];
            for (int j = 0; j < m; ++j) q[j] = -0.5 * z[j] * z[j];
            ck::exp(q, q, m);
            for (int j = 0; j < m; ++j) {
                const int i = idx[j];
                const double mu = c.p[0][i], sigma = c.p[1][i];
                // z0 + dz on the lower side; the other side by symmetry
                const bool flip = (c.x[i] > 0.5) != upper;
                const double dz = t[j] / (num::kInvSqrt2Pi * q[j]);
                const double zh = flip ? -z[j] : z[j], zl = flip ? -dz : dz;
                const num::DD x = num::dd::two_prod(sigma, zh) + sigma * zl + mu;
                out[i] = x.hi + x.lo;
                fast[i] = true;
            }
        }
        ck::fallback(c.n, fast, out, [&](int i) {
            return upper ? iccdf(c.x[i], c.p[0][i], c.p[1][i]) : icdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void icdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, false); }
    static void iccdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, true); }
    static double ilogcdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return normal_detail::locate(mu, sigma, normal_detail::quantile_z_log(lp, false));
    }
    static double ilogccdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return normal_detail::locate(mu, sigma, normal_detail::quantile_z_log(lp, true));
    }

    static double mean(double mu, double) { return mu; }
    static double variance(double, double sigma) { return sigma * sigma; }
    static double standard_deviation(double, double sigma) { return sigma; }
    static double skewness(double, double) { return 0.0; }
    static double kurtosis(double, double) { return 0.0; }
    static double median(double mu, double) { return mu; }
    static double mode(double mu, double) { return mu; }
    // (1 + log 2 pi) / 2 + log sigma, in double-double: it crosses 0 at sigma = 0.242.
    static double entropy(double, double sigma) {
        const num::DD h = num::kNormalEntropy + num::log_dd(sigma);
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return -kInf; }
    static double support_hi(double, double) { return kInf; }

    template <class M>
    static constexpr double cost() {
        using namespace method;
        if constexpr (std::is_same_v<M, Pdf> || std::is_same_v<M, Logpdf>) return 4.0;  // the chunk kernels'
        else if constexpr (std::is_same_v<M, Cdf> || std::is_same_v<M, Ccdf>) return 7.0;
        else if constexpr (std::is_same_v<M, Logcdf> || std::is_same_v<M, Logccdf>) return 15.0;
        else if constexpr (std::is_same_v<M, Icdf> || std::is_same_v<M, Iccdf>) return 9.0;
        else if constexpr (std::is_same_v<M, Ilogcdf> || std::is_same_v<M, Ilogccdf>) return 70.0;
        else return 2.0;
    }
};

}  // namespace sd::stats
