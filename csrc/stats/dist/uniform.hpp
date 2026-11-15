// The uniform distribution on [a, b] (scipy's Uniform(a, b)). The width b - a
// may overflow for finite a < b: then everything is computed on the halves
// a/2, b/2 (exact) and scaled back.
#pragma once

#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

struct Uniform {
    static constexpr char name[] = "uniform";
    static constexpr std::array<const char*, 2> params{"a", "b"};

    static bool valid(double a, double b) { return std::isfinite(a) && std::isfinite(b) && a < b; }

    // b - a, or half of it with half = true when it overflows.
    static double width(double a, double b, bool& half) {
        const double w = b - a;
        half = std::isinf(w);
        return half ? 0.5 * b - 0.5 * a : w;
    }

    static double pdf(double x, double a, double b) {
        if (x < a || x > b) return 0.0;
        bool half;
        const double w = width(a, b, half);
        return half ? 0.5 / w : 1.0 / w;
    }
    static double logpdf(double x, double a, double b) {
        if (x < a || x > b) return -kInf;
        bool half;
        const double w = width(a, b, half);
        return half ? -(std::log(w) + num::kLn2.hi) : -std::log(w);
    }
    // P(X <= x) (upper: P(X > x)).
    static double tail(double x, double a, double b, bool upper) {
        if (x <= a) return upper ? 1.0 : 0.0;
        if (x >= b) return upper ? 0.0 : 1.0;
        bool half;
        const double w = width(a, b, half);
        const double d = upper ? (half ? 0.5 * b - 0.5 * x : b - x) : (half ? 0.5 * x - 0.5 * a : x - a);
        return d / w;
    }
    static double cdf(double x, double a, double b) { return tail(x, a, b, false); }
    static double ccdf(double x, double a, double b) { return tail(x, a, b, true); }
    static double log_tail(double x, double a, double b, bool upper) {
        const double v = tail(x, a, b, upper);
        return v <= 0.5 ? std::log(v) : std::log1p(-tail(x, a, b, !upper));
    }
    static double logcdf(double x, double a, double b) { return log_tail(x, a, b, false); }
    static double logccdf(double x, double a, double b) { return log_tail(x, a, b, true); }

    // a + p (b - a) for p <= 1/2, b - q (b - a) above (q = 1 - p exact), one rounding.
    static double from_lower(double p, double a, double b) {
        bool half;
        const double w = width(a, b, half);
        return half ? 2.0 * std::fma(p, w, 0.5 * a) : std::fma(p, w, a);
    }
    static double from_upper(double q, double a, double b) {
        bool half;
        const double w = width(a, b, half);
        return half ? 2.0 * std::fma(-q, w, 0.5 * b) : std::fma(-q, w, b);
    }
    static double icdf(double p, double a, double b) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return a;
        if (p == 1.0) return b;
        return p <= 0.5 ? from_lower(p, a, b) : from_upper(1.0 - p, a, b);
    }
    static double iccdf(double p, double a, double b) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return b;
        if (p == 1.0) return a;
        return p <= 0.5 ? from_upper(p, a, b) : from_lower(1.0 - p, a, b);
    }
    static double ilogcdf(double lp, double a, double b) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp < -num::kLn2.hi) return lp == -kInf ? a : from_lower(std::exp(lp), a, b);
        return lp == 0.0 ? b : from_upper(-std::expm1(lp), a, b);
    }
    static double ilogccdf(double lp, double a, double b) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp < -num::kLn2.hi) return lp == -kInf ? b : from_upper(std::exp(lp), a, b);
        return lp == 0.0 ? a : from_lower(-std::expm1(lp), a, b);
    }

    static double mean(double a, double b) {
        const double s = a + b;
        return std::isinf(s) ? 0.5 * a + 0.5 * b : 0.5 * s;
    }
    static double variance(double a, double b) {
        bool half;
        const double w = width(a, b, half);
        if (half) return kInf;  // (b - a)^2 / 12 > 1e600
        const double w2 = w * w;
        return std::isinf(w2) ? (w / 12.0) * w : w2 / 12.0;
    }
    static double standard_deviation(double a, double b) {
        bool half;
        const double w = width(a, b, half);
        constexpr double kInvSqrt12 = 0.28867513459481287;  // 1 / sqrt(12)
        constexpr double kInvSqrt3 = 0.57735026918962573;   // 2 / sqrt(12), for the half width
        return half ? w * kInvSqrt3 : w * kInvSqrt12;
    }
    static double skewness(double, double) { return 0.0; }
    static double kurtosis(double, double) { return -1.2; }
    static double median(double a, double b) { return mean(a, b); }
    static double mode(double, double) { return kNaN; }  // every point of [a, b] (T21)
    static double entropy(double a, double b) {
        bool half;
        const double w = width(a, b, half);
        return half ? std::log(w) + num::kLn2.hi : std::log(w);
    }
    static double support_lo(double a, double) { return a; }
    static double support_hi(double, double b) { return b; }
};

}  // namespace sd::stats
