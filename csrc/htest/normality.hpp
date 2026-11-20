// The sample moments of the normality tests (docs/design/htest.md §3.3; decisions H17, H20):
// the mean and the central sums of powers 2, 3 and 4 of a sample, and Jarque-Bera's statistic
// and p-value from them. sd.regression's r.jarque_bera() calls the same two functions on its
// whitened residuals (regression/linear.cpp resid_stats), so the two areas give the same bits
// for the same values (H17).
//
// The sums are double-doubles (special/dd.hpp): the sample's sum gives the mean m, then
// d = x - m (a double-double), and the sums of d^2, d^3, d^4, each rounded once at the end.
// The rows go in fixed chunks of kSumChunk added in order (in parallel for a long sample), so
// the bits do not depend on the thread count.
#pragma once

#include "parallel/pool.hpp"
#include "special/dd.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sd::htest {

// The central sums of one sample of m values (no NaN expected: a NaN makes every sum NaN).
struct CentralSums {
    special::dd::DD mean{std::numeric_limits<double>::quiet_NaN(), 0.0};
    special::dd::DD s2{std::numeric_limits<double>::quiet_NaN(), 0.0};
    special::dd::DD s3{std::numeric_limits<double>::quiet_NaN(), 0.0};
    special::dd::DD s4{std::numeric_limits<double>::quiet_NaN(), 0.0};
};

inline constexpr std::int64_t kSumChunk = 4096;

// body(c) for the chunks c of m values: in parallel for 2 chunks or more and m >= 2^16
// (parallel_for runs a nested call serially).
template <class Body>
void sum_chunks(std::int64_t m, Body&& body) {
    const std::int64_t nc = (m + kSumChunk - 1) / kSumChunk;
    if (nc >= 2 && m >= (1 << 16)) {
        parallel::parallel_for(nc, [&](std::int64_t c) { body(c); });
    } else {
        for (std::int64_t c = 0; c < nc; ++c) body(c);
    }
}

// The central sums of x[0..m). part: scratch of 8 doubles per chunk (8 * ceil(m / kSumChunk)).
inline CentralSums central_sums(const double* x, std::int64_t m, double* part) {
    using special::dd::DD;
    using special::dd::operator+;
    using special::dd::operator-;
    using special::dd::operator*;
    CentralSums out;
    if (m <= 0) return out;
    const std::int64_t nc = (m + kSumChunk - 1) / kSumChunk;
    auto total = [&](int q) {
        DD acc{0.0, 0.0};
        for (std::int64_t c = 0; c < nc; ++c) acc = acc + DD{part[c * 8 + 2 * q], part[c * 8 + 2 * q + 1]};
        return acc;
    };
    sum_chunks(m, [&](std::int64_t c) {
        const std::int64_t t0 = c * kSumChunk, t1 = std::min(m, t0 + kSumChunk);
        DD s1{0.0, 0.0};
        for (std::int64_t t = t0; t < t1; ++t) s1 = s1 + DD{x[t], 0.0};
        part[c * 8 + 0] = s1.hi, part[c * 8 + 1] = s1.lo;
    });
    const DD mean = special::dd::operator/(total(0), DD{static_cast<double>(m), 0.0});
    sum_chunks(m, [&](std::int64_t c) {
        const std::int64_t t0 = c * kSumChunk, t1 = std::min(m, t0 + kSumChunk);
        DD a2{0.0, 0.0}, a3{0.0, 0.0}, a4{0.0, 0.0};
        for (std::int64_t t = t0; t < t1; ++t) {
            const DD d = DD{x[t], 0.0} - mean;
            const DD d2 = d * d;
            a2 = a2 + d2;
            a3 = a3 + d2 * d;
            a4 = a4 + d2 * d2;
        }
        part[c * 8 + 0] = a2.hi, part[c * 8 + 1] = a2.lo, part[c * 8 + 2] = a3.hi, part[c * 8 + 3] = a3.lo;
        part[c * 8 + 4] = a4.hi, part[c * 8 + 5] = a4.lo;
    });
    out.mean = mean;
    out.s2 = total(0);
    out.s3 = total(1);
    out.s4 = total(2);
    return out;
}

// sd.stats skew's zero-variance rule (H20): the central second moment m2 at most (eps mean)^2,
// eps = 2^-52, is a constant sample.
inline bool zero_variance(double m2, double mean) {
    const double e = 0x1p-52 * mean;
    return m2 <= e * e;
}

// Jarque-Bera from n and the rounded sums: n/6 (S^2 + (K - 3)^2 / 4), S = m3 / m2^(3/2),
// K = m4 / m2^2 with the moments over n, against ChiSquared(df=2).ccdf (H17). NaN for a
// constant sample (H20), and wherever the sums are NaN.
struct JarqueBera {
    double stat, p;
};
inline JarqueBera jarque_bera_of(double n, double s2, double s3, double s4, double mean) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const double mu2 = s2 / n;
    if (zero_variance(mu2, mean)) return {kNaN, kNaN};
    const double skew = (s3 / n) / (mu2 * std::sqrt(mu2));
    const double kurt = (s4 / n) / (mu2 * mu2);
    const double ek = kurt - 3.0;
    const double jb = n / 6.0 * (skew * skew + ek * ek / 4.0);
    return {jb, stats::method_of<stats::ChiSquared, stats::method::Ccdf>(jb, 2.0)};
}

}  // namespace sd::htest
