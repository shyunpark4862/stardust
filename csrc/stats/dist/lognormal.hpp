// The log-normal distribution: log X ~ N(mu, sigma^2) (mu and sigma of the
// underlying normal; T18). z = (log x - mu) / sigma with log x a double-double
// (stats/numeric.hpp): the tails multiply log x's rounding by |z| |log x| / sigma.
// The inverses exponentiate mu + sigma z in double-double: exp(E) multiplies E's
// rounding by |E| (up to 709).
#pragma once

#include "special/ops/distr.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

struct LogNormal {
    static constexpr char name[] = "lognormal";
    static constexpr std::array<const char*, 2> params{"mu", "sigma"};

    static bool valid(double mu, double sigma) { return std::isfinite(mu) && sigma > 0.0 && std::isfinite(sigma); }

    // z for finite x > 0.
    static num::DD z_of(double x, double mu, double sigma) { return num::div(num::log_dd(x) + (-mu), sigma); }

    // -(z^2 / 2 + log(sigma sqrt(2 pi)) + log x)
    static num::DD log_density(double x, double mu, double sigma) {
        const num::DD lx = num::log_dd(x);  // once, for z and the 1/x
        return -(normal_detail::neg_log_density(num::div(lx + (-mu), sigma), sigma) + lx);
    }
    static double pdf(double x, double mu, double sigma) {
        if (x <= 0.0 || std::isinf(x)) return 0.0;
        return num::exp_dd(log_density(x, mu, sigma));
    }
    static double logpdf(double x, double mu, double sigma) {
        if (x <= 0.0 || std::isinf(x)) return -kInf;
        const num::DD l = log_density(x, mu, sigma);
        return l.hi + l.lo;
    }
    static double tail(double x, double mu, double sigma, bool upper) {
        if (x <= 0.0) return upper ? 1.0 : 0.0;
        if (std::isinf(x)) return upper ? 0.0 : 1.0;
        return normal_detail::tail_z(z_of(x, mu, sigma), upper);
    }
    static double cdf(double x, double mu, double sigma) { return tail(x, mu, sigma, false); }
    static double ccdf(double x, double mu, double sigma) { return tail(x, mu, sigma, true); }
    static double log_tail(double x, double mu, double sigma, bool upper) {
        if (x <= 0.0) return upper ? 0.0 : -kInf;
        if (std::isinf(x)) return upper ? -kInf : 0.0;
        return normal_detail::log_tail_z(z_of(x, mu, sigma), upper);
    }
    static double logcdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, false); }
    static double logccdf(double x, double mu, double sigma) { return log_tail(x, mu, sigma, true); }

    // exp(mu + sigma z), z and the exponent double-doubles (normal_detail::quantile_z).
    static double from_z(num::DD z, double mu, double sigma) {
        if (std::isinf(z.hi)) return z.hi > 0.0 ? kInf : 0.0;
        return num::exp_dd(num::dd::two_prod(sigma, z.hi) + sigma * z.lo + mu);
    }
    // Without the Newton step while sigma |z| <= 4: ndtri's few ULP in z reach x multiplied
    // by sigma |z| (exp of mu + sigma z), at most ~22 ULP then.
    static double icdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p > 0.0 && p < 1.0) {
            const double z = normal_detail::ndtri(p);
            if (sigma * std::fabs(z) <= 4.0) return from_z(num::DD{z, 0.0}, mu, sigma);
        }
        return from_z(normal_detail::quantile_z(p, false), mu, sigma);
    }
    static double iccdf(double p, double mu, double sigma) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p > 0.0 && p < 1.0) {
            const double z = -normal_detail::ndtri(p);
            if (sigma * std::fabs(z) <= 4.0) return from_z(num::DD{z, 0.0}, mu, sigma);
        }
        return from_z(normal_detail::quantile_z(p, true), mu, sigma);
    }
    static double ilogcdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return from_z(normal_detail::quantile_z_log(lp, false), mu, sigma);
    }
    static double ilogccdf(double lp, double mu, double sigma) {
        if (!(lp <= 0.0)) return kNaN;
        return from_z(normal_detail::quantile_z_log(lp, true), mu, sigma);
    }

    // t = e^(sigma^2) - 1 with sigma^2 = s2.hi + s2.lo: expm1(hi) + e^hi lo.
    static double expm1_dd(num::DD s2) { return std::expm1(s2.hi) + std::exp(s2.hi) * s2.lo; }

    static double mean(double mu, double sigma) {  // exp(mu + sigma^2 / 2)
        return num::exp_dd(num::dd::two_prod(sigma, sigma) * 0.5 + mu);
    }
    // (e^(sigma^2) - 1) e^(2 mu + sigma^2); past sigma^2 = 700 the first factor is e^(sigma^2).
    static double variance(double mu, double sigma) {
        const num::DD s2 = num::dd::two_prod(sigma, sigma);
        if (s2.hi > 700.0) return num::exp_dd(s2 * 2.0 + 2.0 * mu);
        return expm1_dd(s2) * num::exp_dd(s2 + 2.0 * mu);
    }
    static double standard_deviation(double mu, double sigma) {
        const num::DD s2 = num::dd::two_prod(sigma, sigma);
        if (s2.hi > 700.0) return num::exp_dd(s2 + mu);
        return std::sqrt(expm1_dd(s2)) * num::exp_dd(s2 * 0.5 + mu);
    }
    static double skewness(double, double sigma) {  // (t + 3) sqrt(t)
        const double t = expm1_dd(num::dd::two_prod(sigma, sigma));
        return (t + 3.0) * std::sqrt(t);
    }
    // e^(4s) + 2 e^(3s) + 3 e^(2s) - 6 = 16 t + 15 t^2 + 6 t^3 + t^4: all terms positive.
    static double kurtosis(double, double sigma) {
        const double t = expm1_dd(num::dd::two_prod(sigma, sigma));
        return t * (16.0 + t * (15.0 + t * (6.0 + t)));
    }
    static double median(double mu, double) { return std::exp(mu); }
    static double mode(double mu, double sigma) { return num::exp_dd(-num::dd::two_prod(sigma, sigma) + mu); }
    static double entropy(double mu, double sigma) {  // mu + 1/2 + log(sigma sqrt(2 pi))
        const num::DD h = num::kNormalEntropy + num::log_dd(sigma) + mu;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
