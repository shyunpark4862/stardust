// The exponential distribution with scale s (rate 1/s; T18): density
// exp(-x/s)/s on [0, inf). y = x / s is a double-double (stats/numeric.hpp):
// exp(-y) multiplies y's rounding by y (745 ULP at the bottom).
#pragma once

#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

struct Exponential {
    static constexpr char name[] = "exponential";
    static constexpr std::array<const char*, 1> params{"scale"};

    static bool valid(double s) { return s > 0.0 && std::isfinite(s); }

    // -(y + log s): the log density at y = x / s >= 0.
    static num::DD log_density(num::DD y, double s) { return -(y + num::log_dd(s)); }

    static double pdf(double x, double s) {
        if (x < 0.0 || std::isinf(x)) return 0.0;
        const num::DD y = num::ratio(x, s);
        if (y.hi < 700.0 && s >= 0x1p-900) return num::exp_dd(-y) / s;
        return num::exp_dd(log_density(y, s));
    }
    static double logpdf(double x, double s) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        const num::DD l = log_density(num::ratio(x, s), s);
        return l.hi + l.lo;
    }
    static double cdf(double x, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return 1.0;
        const num::DD y = num::ratio(x, s);
        return -std::expm1(-y.hi) + std::exp(-y.hi) * y.lo;
    }
    static double ccdf(double x, double s) {
        if (x <= 0.0) return 1.0;
        if (std::isinf(x)) return 0.0;
        return num::exp_dd(-num::ratio(x, s));
    }
    static double logcdf(double x, double s) {
        if (x <= 0.0) return -kInf;
        if (std::isinf(x)) return 0.0;
        const num::DD y = num::ratio(x, s);
        if (y.hi < num::kLn2.hi) return std::log(cdf(x, s));
        return std::log1p(-num::exp_dd(-y));
    }
    static double logccdf(double x, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return -kInf;
        const num::DD y = num::ratio(x, s);
        return -(y.hi + y.lo);
    }

    static double icdf(double p, double s) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        return -s * std::log1p(-p);
    }
    static double iccdf(double p, double s) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        return -s * std::log(p);
    }
    static double ilogcdf(double lp, double s) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == -kInf) return 0.0;
        if (lp < -700.0) return num::exp_dd(num::DD{lp, 0.0} + num::log_dd(s));  // s e^lp: -log1p(-p) = p
        if (lp < -num::kLn2.hi) return -s * std::log1p(-std::exp(lp));
        return -s * std::log(-std::expm1(lp));
    }
    static double ilogccdf(double lp, double s) {
        if (!(lp <= 0.0)) return kNaN;
        return -s * lp;
    }

    static double mean(double s) { return s; }
    static double variance(double s) { return s * s; }
    static double standard_deviation(double s) { return s; }
    static double skewness(double) { return 2.0; }
    static double kurtosis(double) { return 6.0; }
    static double median(double s) { return s * num::kLn2.hi + s * num::kLn2.lo; }
    static double mode(double) { return 0.0; }
    static double entropy(double s) {  // 1 + log s, crossing 0 at s = 1/e
        const num::DD h = num::log_dd(s) + 1.0;
        return h.hi + h.lo;
    }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double) { return kInf; }
};

}  // namespace sd::stats
