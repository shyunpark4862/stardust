// The Cauchy distribution with location mu and scale sigma (T18):
// density 1 / (pi sigma (1 + z^2)). The tails are taken from the side where
// they are small: P(Z <= z) = atan(-1/z) / pi for z < -1 (no 1/2 - ... to
// cancel), the inverse from tan(pi p) on the matching side.
#pragma once

#include "special/gamma_core.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

struct Cauchy {
    static constexpr char name[] = "cauchy";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    static double pdf(double x, double mu, double sigma) {
        using special::scaled;
        if (std::isinf(x)) return 0.0;
        const double z = std::fabs(num::standardize(x, mu, sigma).hi);
        // 1 / (pi sigma t), t = 1 + z^2, every factor as m 2^e: no over- or underflow on the way
        const special::Scaled t = z < 1e150 ? scaled(1.0 + z * z) : scaled(z) * scaled(z);
        return special::value(scaled(num::kInvPi.hi) / (scaled(sigma) * t));
    }
    static double logpdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return -kInf;
        const num::DD z = num::standardize(x, mu, sigma);
        num::DD lt;  // log(1 + z^2)
        if (std::fabs(z.hi) < 1e150) lt = num::dd::log(z * z + 1.0);
        else lt = num::log_dd(std::fabs(z.hi)) * 2.0;
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
        return tail_z(num::standardize(x, mu, sigma).hi);
    }
    static double ccdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return x > 0.0 ? 0.0 : 1.0;
        return tail_z(-num::standardize(x, mu, sigma).hi);
    }
    static double log_side(double z) {  // log P(Z <= z)
        const double v = tail_z(z);
        return v <= 0.5 ? std::log(v) : std::log1p(-tail_z(-z));
    }
    static double logcdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return x > 0.0 ? 0.0 : -kInf;
        return log_side(num::standardize(x, mu, sigma).hi);
    }
    static double logccdf(double x, double mu, double sigma) {
        if (std::isinf(x)) return x > 0.0 ? -kInf : 0.0;
        return log_side(-num::standardize(x, mu, sigma).hi);
    }

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
    static double locate(double p, double mu, double sigma, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        const double sign = upper ? -1.0 : 1.0;
        if (p > 0.0 && p < 1e-300) return mu - sign * ((sigma * num::kInvPi.hi) / p);
        if (p < 1.0 && 1.0 - p < 1e-300) return mu + sign * ((sigma * num::kInvPi.hi) / (1.0 - p));
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
