// The exponential distribution with scale s (rate 1/s; T18): density
// exp(-x/s)/s on [0, inf). y = x / s is a double-double (stats/numeric.hpp):
// exp(-y) multiplies y's rounding by y (745 ULP at the bottom).
#pragma once

#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
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
        if (std::isinf(y.hi)) return 0.0;  // x / s overflows
        if (y.hi < 700.0 && s >= 0x1p-900) return num::exp_dd(-y) / s;
        return num::exp_dd(log_density(y, s));
    }
    // The scalar definition's first branch with vForce's exp: exp(-y) (1 - y.lo) / s, y = x / s
    // the same double-double (ckd::ratio): relative error kVfExp + 2 whatever y.
    static void pdf_chunk(const Chunk<1>& c, double* out) {
        double yh[kChunkSize], yl[kChunkSize];
        bool fast[kChunkSize];
        ckd::ratio(c.x, c.p[0], yh, yl, c.n);
        for (int i = 0; i < c.n; ++i) {
            fast[i] = c.x[i] >= 0.0 && yh[i] < 700.0 && c.p[0][i] >= 0x1p-900;  // false for NaN
            yh[i] = -yh[i], yl[i] = -yl[i];
        }
        ckd::exp_dd(out, yh, yl, c.n);
        for (int i = 0; i < c.n; ++i) out[i] /= c.p[0][i];
        ck::fallback(c.n, fast, out, [&](int i) { return pdf(c.x[i], c.p[0][i]); });
    }
    static double logpdf(double x, double s) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        const num::DD y = num::ratio(x, s);
        if (std::isinf(y.hi)) return -kInf;  // x / s overflows
        const num::DD l = log_density(y, s);
        return l.hi + l.lo;
    }
    // -x/s - log s in double: the division (1 unit of y) and vForce's log of s (kVfLog |log s|,
    // from a one-element call when s is a broadcast scalar: the same bits as in an array of
    // scales) and the sum (1 unit of the result): at most y + kVfLog |log s| + |value|.
    // The mixed bound max(1, |value|) of logpdf (T20) covers a cancellation (y near -log s,
    // s < 1).
    static void logpdf_chunk(const Chunk<1>& c, double* out) {
        double ls[kChunkSize];
        bool fast[kChunkSize];
        if (c.same[0]) {
            double l;
            ck::log(&l, c.p[0], 1);
            for (int i = 0; i < c.n; ++i) ls[i] = l;
        } else {
            ck::log(ls, c.p[0], c.n);
        }
        for (int i = 0; i < c.n; ++i) {
            const double y = c.x[i] / c.p[0][i];
            out[i] = -y - ls[i];
            const double mag = y + ck::kVfLog * std::fabs(ls[i]);
            const double v = std::fabs(out[i]);
            fast[i] = c.x[i] >= 0.0 && mag + v <= kFastBudget * std::max(1.0, v);  // false for NaN
        }
        ck::fallback(c.n, fast, out, [&](int i) { return logpdf(c.x[i], c.p[0][i]); });
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
    // 1 - e^-y (ckd::one_minus_exp) with y = x/s in double: the quotient's rounding (1 unit)
    // reaches the result times y e^-y / (1 - e^-y) <= 1, which is all the scalar definition's
    // correction e^-y y.lo restores: 1 + kOneMinusExp = 9.5. Every x: 0 below the support, 1 at
    // +inf and where x/s overflows.
    static void cdf_chunk(const Chunk<1>& c, double* out) {
        for (int i = 0; i < c.n; ++i) out[i] = c.x[i] > 0.0 ? c.x[i] / c.p[0][i] : 0.0;
        ckd::one_minus_exp(out, out, c.n);
    }
    // exp(-y) (1 - y.lo) by vForce, y the same double-double: relative error kVfExp + 1.
    static void ccdf_chunk(const Chunk<1>& c, double* out) {
        double yh[kChunkSize], yl[kChunkSize];
        bool fast[kChunkSize];
        ckd::ratio(c.x, c.p[0], yh, yl, c.n);
        for (int i = 0; i < c.n; ++i) yh[i] = -yh[i], yl[i] = -yl[i];
        ckd::exp_dd(out, yh, yl, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = c.x[i] > 0.0 && out[i] >= 0x1p-1000;  // false for NaN
        ck::fallback(c.n, fast, out, [&](int i) { return ccdf(c.x[i], c.p[0][i]); });
    }
    // With y = x/s the scalar definition's double-double, e = e^-y by vForce (kVfExp + 1) and
    // m = 1 - e^-y.hi (ckd::one_minus_exp):
    //   y < log 2:  log(m + (1 - m) y.lo) by vForce: m's bound kTanh (1 - m/2) + 2 (tanh's error
    //               reaches m times 1 / (1 + T) = 1 - m/2) and the sum, over |log m| >= log 2,
    //               plus kVfLog: at most 15.4 (at m = 1/2);
    //   y >= log 2: log(1 - e) (ckd::log1p): (kVfExp + 1) / (1 - e) + kLog1p <= 13.7.
    // x/s = inf gives log(1 - 0) = -0 as the scalar definition's log1p(-0).
    static void logcdf_chunk(const Chunk<1>& c, double* out) {
        double yh[kChunkSize], yl[kChunkSize], e[kChunkSize], m[kChunkSize];
        bool fast[kChunkSize];
        ckd::ratio(c.x, c.p[0], yh, yl, c.n);
        for (int i = 0; i < c.n; ++i) e[i] = -yh[i];
        ck::exp(e, e, c.n);
        ckd::one_minus_exp(m, yh, c.n);
        for (int i = 0; i < c.n; ++i) {
            m[i] += (1.0 - m[i]) * yl[i];
            e[i] = -std::min(e[i] * (1.0 - yl[i]), 0.5);
        }
        ck::log(m, m, c.n);
        ckd::log1p(e, e, c.n);
        for (int i = 0; i < c.n; ++i) {
            out[i] = yh[i] < num::kLn2.hi ? m[i] : e[i];
            fast[i] = c.x[i] > 0.0 && yh[i] >= 0x1p-1000;  // false for NaN
        }
        ck::fallback(c.n, fast, out, [&](int i) { return logcdf(c.x[i], c.p[0][i]); });
    }
    static double logcdf(double x, double s) {
        if (x <= 0.0) return -kInf;
        if (std::isinf(x)) return 0.0;
        const num::DD y = num::ratio(x, s);
        if (y.hi < 0x1p-60) {
            // log(1 - e^-y) = log y - y/2 + ...: log x - log s (the quotient may be subnormal)
            const num::DD l = num::log_dd(x) - num::log_dd(s);
            return l.hi + l.lo;
        }
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
    // -s log p with vForce's log: relative error kVfLog + 1. Every p (NaN outside [0, 1]).
    static void iccdf_chunk(const Chunk<1>& c, double* out) {
        ck::log(out, c.x, c.n);
        for (int i = 0; i < c.n; ++i) out[i] = c.x[i] <= 1.0 ? -c.p[0][i] * out[i] : kNaN;
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
