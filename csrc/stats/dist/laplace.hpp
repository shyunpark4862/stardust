// The Laplace distribution with location mu and scale sigma (T18):
// density exp(-|x - mu| / sigma) / (2 sigma). z is a double-double
// (stats/numeric.hpp), as for the normal distribution.
#pragma once

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
