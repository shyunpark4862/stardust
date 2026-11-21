// The sample moments of the normality tests (docs/design/htest.md §3.3; decisions H17, H20):
// the mean and the central sums of powers 2, 3 and 4 of a sample, and Jarque-Bera's statistic
// and p-value from them. sd.regression's r.jarque_bera() calls the same two functions on its
// whitened residuals (regression/linear.cpp resid_stats), so the two areas give the same bits
// for the same values (H17).
//
// The sums are double-doubles (special/dd.hpp): the sample's sum gives the mean m, then
// d = x - m (a double-double), and the sums of d^2, d^3, d^4, each rounded once at the end.
// A sample whose largest |x| is past 2^120 or below 2^-120 is summed in a unit of 2^e, e that
// value's exponent (H160; H1's unit, htest/moments.hpp, with the bounds of a fourth power): d^4
// then neither overflows nor leaves the normal doubles for any finite sample, and every
// statistic formed from the sums (a ratio of matching powers, and H20's rule) is the same. Other
// samples are summed as they are (their bits unchanged).
// The rows go in fixed chunks of kSumChunk added in order (in parallel for a long sample), so
// the bits do not depend on the thread count.
#pragma once

#include "htest/vdd.hpp"
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
    int e = 0;  // the sums (and the mean) are those of x 2^-e
};

// The unit's exponent of x[0..m): that of the largest |x| when it is past 2^120 or below 2^-120
// (x 2^-e then has its largest |value| in [1, 2)), else 0 (also for 0 and non-finite values).
inline int scale_exponent(const double* x, std::int64_t m) {
    double big = 0.0;
    for (std::int64_t t = 0; t < m; ++t) big = std::max(big, std::fabs(x[t]));
    if (!(big > 0.0) || !std::isfinite(big)) return 0;
    const int e = std::ilogb(big);
    return (e > 120 || e < -120) ? e : 0;
}

inline constexpr std::int64_t kSumChunk = 4096;

// body(c) for the chunks c of m values: in parallel for 2 chunks or more and m >= 2^16
// (parallel_for runs a nested call serially; sd.regression's resid_stats uses it too).
template <class Body>
void sum_chunks(std::int64_t m, Body&& body) {
    const std::int64_t nc = (m + kSumChunk - 1) / kSumChunk;
    if (nc >= 2 && m >= (1 << 16)) {
        parallel::parallel_for(nc, [&](std::int64_t c) { body(c); });
    } else {
        for (std::int64_t c = 0; c < nc; ++c) body(c);
    }
}

// body(c) for the pairs of chunks (2c, 2c + 1) of m values: in parallel for 2 chunks or more and
// m >= 2^16 (parallel_for runs a nested call serially). A pair is ~0.1 ms, worth waking the
// sleeping workers, which the pool does only for a job of more than 16 tasks (htest/lanes.hpp
// for_pieces_woken): fewer pairs are padded with empty tasks.
template <class Body>
void sum_chunk_pairs(std::int64_t m, Body&& body) {
    const std::int64_t nc = (m + kSumChunk - 1) / kSumChunk, np = (nc + 1) / 2;
    if (nc >= 2 && m >= (1 << 16)) {
        parallel::parallel_for(std::max<std::int64_t>(np, 17), [&](std::int64_t c) {
            if (c < np) body(c);
        });
    } else {
        for (std::int64_t c = 0; c < np; ++c) body(c);
    }
}

// The central sums of x[0..m). part: scratch of 8 doubles per chunk (8 * ceil(m / kSumChunk)).
// Each chunk's sums are one chain of double-double additions in order; two full chunks run side
// by side in the two lanes of NEON registers (htest/vdd.hpp: the same operations lane by lane, so
// each chunk's bits are the scalar chain's).
inline CentralSums central_sums(const double* x, std::int64_t m, double* part) {
    using special::dd::DD;
    using special::dd::operator+;
    using special::dd::operator-;
    using special::dd::operator*;
    CentralSums out;
    if (m <= 0) return out;
    const int e = scale_exponent(x, m);
    const double sc = std::ldexp(1.0, -std::clamp(e, -1020, 1020));
    const double sc2 = std::ldexp(1.0, -e + std::clamp(e, -1020, 1020));  // (|e| > 1020: two factors)
    const float64x2_t vsc = vdupq_n_f64(sc), vsc2 = vdupq_n_f64(sc2);
    auto at = [&](std::int64_t t) { return x[t] * sc * sc2; };
    const std::int64_t nc = (m + kSumChunk - 1) / kSumChunk;
    auto total = [&](int q) {
        DD acc{0.0, 0.0};
        for (std::int64_t c = 0; c < nc; ++c) acc = acc + DD{part[c * 8 + 2 * q], part[c * 8 + 2 * q + 1]};
        return acc;
    };
    auto full = [&](std::int64_t c) { return c < nc && (c + 1) * kSumChunk <= m; };
    // element t of chunks c and c + 1
    auto pair_at = [&](std::int64_t c, std::int64_t t) {
        return vmulq_f64(vmulq_f64(float64x2_t{x[c * kSumChunk + t], x[(c + 1) * kSumChunk + t]}, vsc), vsc2);
    };
    auto put = [&](std::int64_t c, int q, vdd::D2 v) {
        part[c * 8 + 2 * q] = vgetq_lane_f64(v.hi, 0), part[c * 8 + 2 * q + 1] = vgetq_lane_f64(v.lo, 0);
        part[(c + 1) * 8 + 2 * q] = vgetq_lane_f64(v.hi, 1), part[(c + 1) * 8 + 2 * q + 1] = vgetq_lane_f64(v.lo, 1);
    };
    const float64x2_t zero = vdupq_n_f64(0.0);
    auto sum1 = [&](std::int64_t c) {
        const std::int64_t t0 = c * kSumChunk, t1 = std::min(m, t0 + kSumChunk);
        DD s1{0.0, 0.0};
        for (std::int64_t t = t0; t < t1; ++t) s1 = s1 + DD{at(t), 0.0};
        part[c * 8 + 0] = s1.hi, part[c * 8 + 1] = s1.lo;
    };
    sum_chunk_pairs(m, [&](std::int64_t p) {
        const std::int64_t c = 2 * p;
        if (full(c) && full(c + 1)) {
            vdd::D2 s1{zero, zero};
            for (std::int64_t t = 0; t < kSumChunk; ++t) s1 = vdd::add(s1, vdd::D2{pair_at(c, t), zero});
            put(c, 0, s1);
            return;
        }
        sum1(c);
        if (c + 1 < nc) sum1(c + 1);
    });
    const DD mean = special::dd::operator/(total(0), DD{static_cast<double>(m), 0.0});
    auto sums234 = [&](std::int64_t c) {
        const std::int64_t t0 = c * kSumChunk, t1 = std::min(m, t0 + kSumChunk);
        DD a2{0.0, 0.0}, a3{0.0, 0.0}, a4{0.0, 0.0};
        for (std::int64_t t = t0; t < t1; ++t) {
            const DD d = DD{at(t), 0.0} - mean;
            const DD d2 = d * d;
            a2 = a2 + d2;
            a3 = a3 + d2 * d;
            a4 = a4 + d2 * d2;
        }
        part[c * 8 + 0] = a2.hi, part[c * 8 + 1] = a2.lo, part[c * 8 + 2] = a3.hi, part[c * 8 + 3] = a3.lo;
        part[c * 8 + 4] = a4.hi, part[c * 8 + 5] = a4.lo;
    };
    const vdd::D2 nmean{vdupq_n_f64(-mean.hi), vdupq_n_f64(-mean.lo)};
    sum_chunk_pairs(m, [&](std::int64_t p) {
        const std::int64_t c = 2 * p;
        if (full(c) && full(c + 1)) {
            vdd::D2 a2{zero, zero}, a3{zero, zero}, a4{zero, zero};
            for (std::int64_t t = 0; t < kSumChunk; ++t) {
                const vdd::D2 d = vdd::add(vdd::D2{pair_at(c, t), zero}, nmean);
                const vdd::D2 d2 = vdd::mul(d, d);
                a2 = vdd::add(a2, d2);
                a3 = vdd::add(a3, vdd::mul(d2, d));
                a4 = vdd::add(a4, vdd::mul(d2, d2));
            }
            put(c, 0, a2), put(c, 1, a3), put(c, 2, a4);
            return;
        }
        sums234(c);
        if (c + 1 < nc) sums234(c + 1);
    });
    out.e = e;
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
