// The Cauchy distribution with location mu and scale sigma (T18):
// density 1 / (pi sigma (1 + z^2)). The tails are taken from the side where
// they are small: P(Z <= z) = atan(-1/z) / pi for z < -1 (no 1/2 - ... to
// cancel), the inverse from tan(pi p) on the matching side.
#pragma once

#include "special/gamma_core.hpp"
#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

struct Cauchy {
    static constexpr char name[] = "cauchy";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    // The tails are powers of z = (x - mu) / sigma, so they have values where z itself is
    // beyond the double range (sigma / |x - mu| is a double down to 5e-324 / 3.6e308):
    // sigma / |x - mu| and log|z| from x - mu directly (by halves where it overflows).
    static double inv_abs_z(double x, double mu, double sigma) {
        const double d = x - mu;
        return std::isfinite(d) ? sigma / std::fabs(d) : (0.5 * sigma) / std::fabs(0.5 * x - 0.5 * mu);
    }
    static num::DD log_abs_z(double x, double mu, double sigma) {
        const num::DD d = num::dd::two_sum(x, -mu);
        if (std::isfinite(d.hi)) return num::dd::log_pos(d.hi < 0.0 ? -d : d) - num::log_dd(sigma);
        return num::log_dd(std::fabs(0.5 * x - 0.5 * mu)) + num::kLn2 - num::log_dd(sigma);
    }
    static double pdf(double x, double mu, double sigma) {
        using special::scaled;
        if (std::isinf(x)) return 0.0;
        const double z = std::fabs(num::standardize(x, mu, sigma).hi);
        if (std::isinf(z)) {  // sigma / (pi (x - mu)^2)
            const double d = x - mu;
            const special::Scaled sd =
                std::isfinite(d) ? scaled(std::fabs(d)) : scaled(std::fabs(0.5 * x - 0.5 * mu)) * scaled(2.0);
            return special::value(scaled(num::kInvPi.hi) * scaled(sigma) / (sd * sd));
        }
        // 1 / (pi sigma t), t = 1 + z^2, every factor as m 2^e: no over- or underflow on the way
        const special::Scaled t = z < 1e150 ? scaled(1.0 + z * z) : scaled(z) * scaled(z);
        return special::value(scaled(num::kInvPi.hi) / (scaled(sigma) * t));
    }
    static double logpdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return -kInf;
        const num::DD z = num::standardize(x, mu, sigma);
        num::DD lt;  // log(1 + z^2)
        if (std::fabs(z.hi) < 1e150) lt = num::dd::log(z * z + 1.0);
        else lt = log_abs_z(x, mu, sigma) * 2.0;
        const num::DD l = -(lt + num::log_dd(sigma) + num::kLogPi);
        return l.hi + l.lo;
    }
    // P(Z <= z) for the standardized z.
    static double tail_z(double z) {
        if (z < -1.0) return std::atan(-1.0 / z) * num::kInvPi.hi;
        return 0.5 + std::atan(z) * num::kInvPi.hi;
    }
    static double cdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return x > 0.0 ? 1.0 : 0.0;
        return tail(x, mu, sigma, false);
    }
    static double ccdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return x > 0.0 ? 0.0 : 1.0;
        return tail(x, mu, sigma, true);
    }
    static double tail(double x, double mu, double sigma, bool upper) {
        const double z = upper ? -num::standardize(x, mu, sigma).hi : num::standardize(x, mu, sigma).hi;
        if (std::isinf(z)) {  // atan(1/|z|) / pi = sigma / (pi |x - mu|)
            const double v = inv_abs_z(x, mu, sigma) * num::kInvPi.hi;
            return z < 0.0 ? v : 1.0 - v;
        }
        return tail_z(z);
    }
    // Past |z| = 1e150 the small side is 1 / (pi |z|) to the last bit and may be subnormal:
    // its log from log|z|, the large side's as -1 / (pi |z|).
    static double log_tail(double x, double mu, double sigma, bool upper) {
        const double z = upper ? -num::standardize(x, mu, sigma).hi : num::standardize(x, mu, sigma).hi;
        if (std::fabs(z) > 1e150) {
            if (z > 0.0) return -(inv_abs_z(x, mu, sigma) * num::kInvPi.hi);
            const num::DD l = -(log_abs_z(x, mu, sigma) + num::kLogPi);
            return l.hi + l.lo;
        }
        return log_side(z);
    }
    static double log_side(double z) {  // log P(Z <= z)
        const double v = tail_z(z);
        return v <= 0.5 ? std::log(v) : std::log1p(-tail_z(-z));
    }
    static double logcdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return x > 0.0 ? 0.0 : -kInf;
        return log_tail(x, mu, sigma, false);
    }
    static double logccdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return x > 0.0 ? -kInf : 0.0;
        return log_tail(x, mu, sigma, true);
    }

    // ---- chunk kernels: z = (x - mu) / sigma is ckd::standardize's high part (the scalar
    // definition's z.hi; NaN where it must take over).
    //
    // 1 / (pi sigma (1 + z^2)) in double where nothing leaves the normal range (the scalar
    // definition carries every factor as m 2^e): z^2 (2 for z's 1/2 ULP, 1 for the product),
    // the sum, the product with sigma and the quotient: at most 6.
    static void pdf_chunk(const Chunk<2>& c, double* out) {
        double zh[kChunkSize], zl[kChunkSize];
        bool fast[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], zh, zl, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double z = zh[i], sigma = c.p[1][i];
            const double d = sigma * (1.0 + z * z);
            out[i] = num::kInvPi.hi / d;
            fast[i] =
                std::fabs(z) < 1e150 && sigma >= 0x1p-900 && d <= 0x1p1000 && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;
        }
        ck::fallback(c.n, fast, out, [&](int i) { return pdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // -(log(1 + z^2) + log sigma + log pi), the first log by vForce (1 + z^2 carries 4 units,
    // absolute in its log; kVfLog |log| more), the rest in double-double: 4 + kVfLog |log(1 +
    // z^2)| and the last rounding, against logpdf's mixed bound (T20).
    static void logpdf_chunk(const Chunk<2>& c, double* out) {
        double zh[kChunkSize], zl[kChunkSize], kh[kChunkSize], kl[kChunkSize];
        bool fast[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], zh, zl, c.n);
        for (int i = 0; i < c.n; ++i) zl[i] = 1.0 + zh[i] * zh[i];
        ck::log(zl, zl, c.n);
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[1][i]) + num::kLogPi; });
        for (int i = 0; i < c.n; ++i) {
            const num::DD l = -(num::DD{kh[i], kl[i]} + zl[i]);
            out[i] = l.hi + l.lo;
            const double v = std::fabs(out[i]);
            fast[i] = std::fabs(zh[i]) < 1e150 &&
                      4.0 + ck::kVfLog * zl[i] + v <= kFastBudget * std::max(1.0, v);  // false for NaN
        }
        ck::fallback(c.n, fast, out, [&](int i) { return logpdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // s = P(Z <= -|z|) for the chunk by vForce's atan (ckd::kAtan): atan(1 / |z|) / pi beyond
    // |z| = 1 (the quotient, atan, the product: kAtan + 2), 1/2 - atan(|z|) / pi inside (the
    // difference is at least 1/4 and the subtrahend at most 1/4: kAtan + 2); z's own rounding
    // (1 unit) reaches s times at most 1: at most kAtan + 3 = 7.
    static void small_side_chunk(const double* zh, double* s, int n) {
        for (int i = 0; i < n; ++i) {
            const double a = std::fabs(zh[i]);
            s[i] = a > 1.0 ? 1.0 / a : a;
        }
        ckd::atan(s, s, n);
        for (int i = 0; i < n; ++i) {
            const double v = s[i] * num::kInvPi.hi;
            s[i] = std::fabs(zh[i]) > 1.0 ? v : 0.5 - v;
        }
    }
    // The small side s, the large 1 - s (s <= 1/2: its error counts at most once); their logs
    // log s by vForce (7 / log 2 over |log s|, kVfLog: at most 14.1) and log1p(-s) by libm (s's
    // 7 times s / ((1 - s) |log1p(-s)|) <= 1.45, plus 2: 12.2).
    static void tail_chunk(const Chunk<2>& c, double* out, bool upper, bool log) {
        double zh[kChunkSize], zl[kChunkSize], s[kChunkSize];
        bool fast[kChunkSize];
        ckd::standardize(c.x, c.p[0], c.p[1], zh, zl, c.n);
        small_side_chunk(zh, s, c.n);
        if (log) ck::log(out, s, c.n);
        for (int i = 0; i < c.n; ++i) {
            const bool small = upper ? zh[i] > 0.0 : zh[i] < 0.0;
            fast[i] = s[i] >= 0x1p-1000;  // false for NaN
            if (log) out[i] = small ? out[i] : std::log1p(-s[i]);
            else out[i] = small ? s[i] : 1.0 - s[i];
        }
        ck::fallback(c.n, fast, out, [&](int i) {
            const double x = c.x[i], mu = c.p[0][i], sigma = c.p[1][i];
            if (log) return upper ? logccdf(x, mu, sigma) : logcdf(x, mu, sigma);
            return upper ? ccdf(x, mu, sigma) : cdf(x, mu, sigma);
        });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false, false); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true, false); }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, false, true); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { tail_chunk(c, out, true, true); }

    // z with P(Z <= z) = p: -1/tan(pi p) in the lower quarter, tan(pi (p - 1/2)) in the
    // middle (p - 1/2 exact), 1/tan(pi (1 - p)) in the upper quarter.
    static double z_of(double p) {
        if (p == 0.0) return -kInf;
        if (p == 1.0) return kInf;
        if (p < 0.25) return -1.0 / std::tan(num::kPi.hi * p);
        if (p > 0.75) return 1.0 / std::tan(num::kPi.hi * (1.0 - p));
        return std::tan(num::kPi.hi * (p - 0.5));
    }
    // mu + sigma z(p) (upper: mu - sigma z(p)); for a tiny p z = -1/(pi p), taken as
    // (sigma / pi) / p (pi p is subnormal or z overflows while sigma z does not).
    // sigma / (pi q) for a tiny q: the quotient first (sigma / pi alone rounds for a subnormal
    // sigma), unless it overflows where the result does not.
    static double far(double sigma, double q) {
        const double r = sigma / q;
        return std::isinf(r) ? (sigma * num::kInvPi.hi) / q : r * num::kInvPi.hi;
    }
    static double locate(double p, double mu, double sigma, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        const double sign = upper ? -1.0 : 1.0;
        if (p > 0.0 && p < 1e-300) return mu - sign * far(sigma, p);
        if (p < 1.0 && 1.0 - p < 1e-300) return mu + sign * far(sigma, 1.0 - p);
        return std::fma(sigma, sign * z_of(p), mu);
    }
    static double icdf(double p, double mu, double sigma) { return locate(p, mu, sigma, false); }
    static double iccdf(double p, double mu, double sigma) { return locate(p, mu, sigma, true); }
    // x with log P(X <= x) = lp (upper: log P(X > x) = lp).
    static double from_log(double lp, double mu, double sigma, bool upper) {
        if (!(lp <= 0.0)) return kNaN;
        const double sign = upper ? -1.0 : 1.0;
        if (lp == -kInf) return -sign * kInf;
        if (lp < -700.0) {
            // z = -1/(pi p) to the last bit: -e^(-lp) / pi, the product with sigma in the exponent
            const num::DD e = num::DD{-lp, 0.0} + num::log_dd(sigma) - num::kLogPi;
            return mu - sign * num::exp_dd(e);
        }
        const double z = lp < -num::kLn2.hi ? z_of(std::exp(lp)) : -z_of(-std::expm1(lp));
        return std::fma(sigma, sign * z, mu);
    }
    static double ilogcdf(double lp, double mu, double sigma) { return from_log(lp, mu, sigma, false); }
    static double ilogccdf(double lp, double mu, double sigma) { return from_log(lp, mu, sigma, true); }

    // No moments: the mean is undefined, and so is every moment built on it (NaN).
    static double mean(double, double) { return kNaN; }
    static double variance(double, double) { return kNaN; }
    static double standard_deviation(double, double) { return kNaN; }
    static double skewness(double, double) { return kNaN; }
    static double kurtosis(double, double) { return kNaN; }
    static double median(double mu, double) { return mu; }
    static double mode(double mu, double) { return mu; }
    static double entropy(double, double sigma) {  // log(4 pi sigma)
        const num::DD h = num::kLog4Pi + num::log_dd(sigma);
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return -kInf; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
