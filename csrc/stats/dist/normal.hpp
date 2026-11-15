// The normal distribution N(mu, sigma^2) (docs/api/stats.md §2.1): sd.special's
// ndtr, log_ndtr, ndtri and ndtri_exp at the standardized z, which is a
// double-double (stats/numeric.hpp): the cdfs at z.hi corrected by
// phi(z.hi) z.lo, the density's exponent z^2 / 2 + log(sigma sqrt(2 pi)) in
// double-double, exponentiated once.
#pragma once

#include "special/ops/distr.hpp"
#include "special/ops/erf.hpp"
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
    DD lz;
    if (z <= -1.0) lz = log_ndtr_dd(z);
    else lz = {log_ndtr(z), 0.0};
    const double slope = std::exp(-0.5 * z * z - num::kLogSqrt2Pi.hi - lz.hi);  // phi / ndtr
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
    const DD x = num::dd::two_prod(sigma, z.hi) + sigma * z.lo + mu;
    return x.hi + x.lo;
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
        if constexpr (std::is_same_v<M, Pdf> || std::is_same_v<M, Logpdf>) return 25.0;
        else if constexpr (std::is_same_v<M, Cdf> || std::is_same_v<M, Ccdf>) return 20.0;
        else if constexpr (std::is_same_v<M, Logcdf> || std::is_same_v<M, Logccdf>) return 35.0;
        else if constexpr (std::is_same_v<M, Icdf> || std::is_same_v<M, Iccdf>) return 20.0;
        else if constexpr (std::is_same_v<M, Ilogcdf> || std::is_same_v<M, Ilogccdf>) return 30.0;
        else return 2.0;
    }
};

}  // namespace sd::stats
