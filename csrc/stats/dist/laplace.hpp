// The Laplace distribution with location mu and scale sigma (T18):
// density exp(-|x - mu| / sigma) / (2 sigma). z is a double-double
// (stats/numeric.hpp), as for the normal distribution.
#pragma once

#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

struct Laplace {
    static constexpr char name[] = "laplace";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    static num::DD abs(num::DD z) { return z.hi < 0.0 ? -z : z; }
    // -(|z| + log 2 sigma)
    static num::DD log_density(num::DD z, double sigma) {
        if (std::isinf(z.hi)) return {-kInf, 0.0};  // (x - mu) / sigma overflowed: the density is 0
        thread_local num::Memo memo;  // log 2 sigma, per sigma
        return -(abs(z) + memo.get(sigma, 0.0, [&] { return num::log_dd(sigma) + num::kLn2; }));
    }

    static double pdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return 0.0;
        return num::exp_dd(log_density(num::standardize(x, mu, sigma), sigma));
    }
    static double logpdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return -kInf;
        const num::DD l = log_density(num::standardize(x, mu, sigma), sigma);
        return l.hi + l.lo;
    }
    // -(|z| + log 2 sigma) for the chunk, the scalar definition's double-double arithmetic
    // (the same bits); hi = NaN where ckd::standardize leaves z to it.
    static void exponent_chunk(const Chunk<2>& c, double* eh, double* el) {
        double kh[kChunkSize], kl[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], eh, el, c.n);
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[1][i]) + num::kLn2; });
        for (int i = 0; i < c.n; ++i) {
            const num::DD e = -(abs(num::DD{eh[i], el[i]}) + num::DD{kh[i], kl[i]});
            eh[i] = e.hi, el[i] = e.lo;
        }
    }
    // exp of it by vForce: relative error kVfExp + 1 (the scalar definition with libm's exp).
    static void pdf_chunk(const Chunk<2>& c, double* out) {
        double eh[kChunkSize], el[kChunkSize];
        bool fast[kChunkSize];
        exponent_chunk(c, eh, el);
        ckd::exp_dd(out, eh, el, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = out[i] >= 0x1p-1000;  // false for NaN
        ck::fallback(c.n, fast, out, [&](int i) { return pdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // The same bits as the scalar definition.
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
    // e = exp(-|z|) for the chunk by vForce (relative error kVfExp + 1 = 3.5), z the scalar
    // definition's double-double; zh: z.hi (NaN where the scalar definition must take over),
    // negated for the upper tail.
    static void small_side_chunk(const Chunk<2>& c, bool upper, double* zh, double* zl, double* e) {
        ckd::standardize(c.x, c.p[0], c.p[1], zh, zl, c.n);
        double wh[kChunkSize];
        for (int i = 0; i < c.n; ++i) {
            if (upper) zh[i] = -zh[i], zl[i] = -zl[i];
            const bool neg = zh[i] < 0.0;
            wh[i] = neg ? zh[i] : -zh[i];
            zl[i] = neg ? zl[i] : -zl[i];  // the low part of -|z|
        }
        ckd::exp_dd(e, wh, zl, c.n);
    }
    // e / 2 or 1 - e / 2: relative error at most 3.5 (the large side: e / 2 <= 1 of it, and
    // the difference's rounding).
    static void tail_chunk(const Chunk<2>& c, double* out, bool upper) {
        double zh[kChunkSize], zl[kChunkSize], e[kChunkSize];
        bool fast[kChunkSize];
        small_side_chunk(c, upper, zh, zl, e);
        for (int i = 0; i < c.n; ++i) {
            out[i] = zh[i] < 0.0 ? 0.5 * e[i] : 1.0 - 0.5 * e[i];
            fast[i] = zh[i] == zh[i] && e[i] >= 0x1p-1000;
        }
        ck::fallback(c.n, fast, out, [&](int i) { return tail(c.x[i], c.p[0][i], c.p[1][i], upper); });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true); }
    // The small side z - log 2 in double-double (the scalar definition's bits); the large side
    // log(1 - e/2) (ckd::log1p): e's 3.5 units times 1 / (1 - e/2) <= 2, plus kLog1p: 13.7.
    static void log_tail_chunk(const Chunk<2>& c, double* out, bool upper) {
        double zh[kChunkSize], zl[kChunkSize], e[kChunkSize];
        bool fast[kChunkSize];
        small_side_chunk(c, upper, zh, zl, e);
        for (int i = 0; i < c.n; ++i) e[i] *= -0.5;
        ckd::log1p(e, e, c.n);
        for (int i = 0; i < c.n; ++i) {
            fast[i] = zh[i] == zh[i];
            // zl is the low part of -|z|: of z itself on the small side
            const num::DD l = num::DD{zh[i], zl[i]} - num::kLn2;
            out[i] = zh[i] < 0.0 ? l.hi + l.lo : e[i];
        }
        ck::fallback(c.n, fast, out, [&](int i) { return log_tail(c.x[i], c.p[0][i], c.p[1][i], upper); });
    }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { log_tail_chunk(c, out, false); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { log_tail_chunk(c, out, true); }

    // P(X <= x) (upper: P(X > x)): e^z / 2 on the small side, 1 - e^-z / 2 on the other.
    static double tail(double x, double mu, double sigma, bool upper) {
        if (std::isinf(x)) return (x > 0.0) == upper ? 0.0 : 1.0;
        num::DD z = num::standardize(x, mu, sigma);
        if (upper) z = -z;
        if (z.hi < 0.0) return 0.5 * num::exp_dd(z);
        return 1.0 - 0.5 * num::exp_dd(-z);
    }
    static double cdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false); }
    static double ccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true); }
    static double log_tail(double x, double mu, double sigma, bool upper) {
        if (std::isinf(x)) return (x > 0.0) == upper ? -kInf : 0.0;
        num::DD z = num::standardize(x, mu, sigma);
        if (upper) z = -z;
        if (std::isinf(z.hi)) return z.hi < 0.0 ? -kInf : 0.0;  // z overflowed
        if (z.hi < 0.0) {
            const num::DD l = z - num::kLn2;
            return l.hi + l.lo;
        }
        return std::log1p(-0.5 * num::exp_dd(-z));
    }
    static double logcdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, false); }
    static double logccdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, true); }

    // The standardized z with P(Z <= z) = p, from the side where p is the small one.
    static double z_of(double p) {
        if (p <= 0.5) return std::log(2.0 * p);
        return -std::log(2.0 * (1.0 - p));
    }
    static double icdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        return std::fma(sigma, z_of(p), mu);
    }
    static double iccdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        return std::fma(sigma, -z_of(p), mu);
    }
    // mu + sigma z, z = +-log(2 min(p, 1 - p)) by vForce's log: z differs from the scalar
    // definition's by kVfLog |z| units, x by kVfLog |sigma z / x|. Taken where that is inside the
    // budget, and for p in [1/4, 3/4] always: there |z| <= log 2 and the quantile's condition
    // number in p is at least sigma / |x| (dz/dp = 1/min(p, 1 - p) >= 1), so the difference
    // is below kVfLog log 2 of the contract's max(1, kappa) (docs/design/stats.md §2.3).
    static void quantile_chunk(const Chunk<2>& c, double* out, bool upper) {
        double z[kChunkSize];
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) z[i] = 2.0 * (c.x[i] <= 0.5 ? c.x[i] : 1.0 - c.x[i]);
        ck::log(z, z, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i], mu = c.p[0][i], sigma = c.p[1][i];
            const double zi = (p <= 0.5) != upper ? z[i] : -z[i];
            out[i] = std::fma(sigma, zi, mu);
            const bool central = p >= 0.25 && p <= 0.75;
            fast[i] = p > 0.0 && p < 1.0 &&
                      (central || ck::kVfLog * std::fabs(sigma * zi) <= (kFastBudget - 1.0) * std::fabs(out[i]));
        }
        ck::fallback(c.n, fast, out, [&](int i) {
            return upper ? iccdf(c.x[i], c.p[0][i], c.p[1][i]) : icdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void icdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, false); }
    static void iccdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, true); }
    // z with log P(Z <= z) = lp: lp + log 2 below log 1/2, else -log(2 (1 - e^lp)).
    static double z_of_log(double lp) {
        if (lp == -kInf) return -kInf;
        if (lp < -num::kLn2.hi) {
            const num::DD z = num::kLn2 + lp;
            return z.hi + z.lo;
        }
        return -(std::log(-std::expm1(lp)) + num::kLn2.hi);
    }
    static double ilogcdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return std::fma(sigma, z_of_log(lp), mu);
    }
    static double ilogccdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return std::fma(sigma, -z_of_log(lp), mu);
    }

    static double mean(double mu, double) { return mu; }
    static double variance(double, double sigma) { return 2.0 * sigma * sigma; }
    static double standard_deviation(double, double sigma) { return 1.4142135623730951 * sigma; }
    static double skewness(double, double) { return 0.0; }
    static double kurtosis(double, double) { return 3.0; }
    static double median(double mu, double) { return mu; }
    static double mode(double mu, double) { return mu; }
    static double entropy(double, double sigma) {  // 1 + log 2 sigma
        const num::DD h = num::kOnePlusLn2 + num::log_dd(sigma);
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return -kInf; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
