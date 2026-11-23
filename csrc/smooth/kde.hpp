// What the kernel density estimate (smooth/kde.cpp) and the bandwidth selectors (smooth/bandwidth.cpp) share
// (docs/design/smooth.md §3.5, §3.6; decisions SM117-SM122): the kernels, the normalising constant kept as a
// mantissa and a power of two, the value exp(a) m 2^-E with one rounding even when it is subnormal, the linear
// binning in a fixed order and the FFT lengths.
#pragma once

#include "array/kernels/vmath.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::smooth::kde {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kSqrt2Pi = 0x1.40d931ff62705p+1;  // fl(sqrt(2 pi))
inline constexpr double kLog2e = 0x1.71547652b82fep+0;
inline constexpr double kLn2Hi = 0x1.62e42fee00000p-1;  // 21 trailing zero bits: k kLn2Hi is exact for |k| < 2^21
inline constexpr double kLn2Lo = 0x1.a39ef35793c76p-33;
inline constexpr double kTau = 8.3;  // the binned path's Gaussian cut, in standard deviations (decision SM31)

enum Kernel : int { kGauss = 0, kEpa = 1, kRect = 2, kTri = 3, kBiw = 4, kCos = 5 };

// The half width a of a compact kernel of standard deviation 1: sqrt 5, sqrt 3, sqrt 6, sqrt 7 and
// 1 / sqrt(1/3 - 2 / pi^2) (R's density).
inline double half_width(int k) {
    switch (k) {
        case kEpa: return std::sqrt(5.0);
        case kRect: return std::sqrt(3.0);
        case kTri: return std::sqrt(6.0);
        case kBiw: return std::sqrt(7.0);
        case kCos: return 1.0 / std::sqrt(1.0 / 3.0 - 2.0 / (M_PI * M_PI));
        default: return kInf;
    }
}

// The compact kernel of standard deviation 1 at z (|z| < a; 0 outside), from r = |z| / a without the
// cancellation of 1 - r^2 (decision SM119).
inline double kernel_at(int k, double z, double a) {
    const double r = std::fabs(z) / a;
    if (!(r < 1.0)) return 0.0;
    const double om = 1.0 - r, op = 1.0 + r;
    switch (k) {
        case kEpa: return 0.75 * (om * op) / a;
        case kRect: return 0.5 / a;
        case kTri: return om / a;
        case kBiw: {
            const double q = om * op;
            return 0.9375 * (q * q) / a;
        }
        case kCos: {
            const double c = std::cos(M_PI_2 * r);
            return (c * c) / a;
        }
        default: return kNaN;
    }
}

// The kernel's distribution function at z (standard deviation 1): closed forms of r = z / a clipped to [-1, 1],
// each written without cancellation at the end it vanishes at.
inline double kernel_cdf(int k, double z, double a) {
    if (k == kGauss) return 0.5 * std::erfc(-z * M_SQRT1_2);
    if (!(z > -a)) return 0.0;
    if (!(z < a)) return 1.0;
    const double r = z / a, op = 1.0 + r, om = 1.0 - r;
    switch (k) {
        case kEpa: return 0.25 * (op * op) * (2.0 - r);
        case kRect: return 0.5 * op;
        case kTri: return r < 0 ? 0.5 * (op * op) : 1.0 - 0.5 * (om * om);
        case kBiw: return (op * op * op) * ((8.0 - 9.0 * r) + 3.0 * (r * r)) * 0.0625;
        case kCos: return 0.5 * (op + std::sin(M_PI * r) / M_PI);
        default: return kNaN;
    }
}

// A positive finite number as m 2^e with m in [0.5, 1): the normalising constants (Sum w (2 pi)^(p/2) det A) are
// products kept this way, so a bandwidth near the ends of the double range neither overflows nor underflows.
struct Scaled {
    double m = 0.5;
    int e = 1;  // 1 = 0.5 * 2^1
    void mul(double v) {
        int ev;
        const double mv = std::frexp(v, &ev);
        int er;
        m = std::frexp(m * mv, &er);
        e += ev + er;
    }
};

// exp(a) q 2^-E for a <= 0 (or any a), q > 0 finite: a = k ln 2 + r (Cody and Waite), exp(r) q rounded once more
// and scaled by 2^(k - E) with scalbn's single rounding (so a subnormal value is within its unit).
inline double exp_scaled(double a, double q, int E) {
    if (a != a || q != q) return kNaN;
    if (!(q > 0)) return 0.0;
    const double kd = std::nearbyint(a * kLog2e);
    if (kd - E < -1200.0) return 0.0;
    if (kd - E > 1200.0) return kInf;
    const double r = (a - kd * kLn2Hi) - kd * kLn2Lo;
    return std::scalbn(vmath::exp1(r) * q, static_cast<int>(kd) - E);
}

// a + log q - E ln 2 for q > 0: the integer part exactly, the rest added once.
inline double log_scaled(double a, double q, int E) {
    if (a != a || q != q) return kNaN;
    if (!(q > 0)) return -kInf;
    const double e = static_cast<double>(-E);
    return (e * kLn2Hi + (a + vmath::log1(q))) + e * kLn2Lo;
}

// The smallest length >= m whose prime factors are 2, 3, 5 and 7 (even when `even`): the FFT lengths of the
// binned convolution (the fft area's plans are fastest on them; the choice depends on m alone).
inline std::int64_t good_length(std::int64_t m, bool even) {
    std::int64_t best = std::numeric_limits<std::int64_t>::max();
    for (std::int64_t a = 1; a < 2 * m + 2; a *= 2)
        for (std::int64_t b = a; b < 2 * m + 2; b *= 3)
            for (std::int64_t c = b; c < 2 * m + 2; c *= 5)
                for (std::int64_t d = c; d < 2 * m + 2; d *= 7)
                    if (d >= m && d < best && (!even || d % 2 == 0)) best = d;
    return best;
}

// Linear binning of sorted points x (weights w, null: 1) onto G grid points g_k = lo + (k - E) delta (decision
// SM120): a point at pos = (x - lo) / delta + E in [0, G - 1] gives w (1 - f) to floor(pos) and w f to the next
// one (f = pos - floor(pos)); points outside are left out. A grid point's sum adds its contributions in the order
// of the points, so tasks over ranges of grid points give the bits of one pass whatever the ranges: the ranges
// split the sorted points into equal runs (the grid has fewer points than one task's worth), and the positions
// of the points are computed in chunks. c: G doubles, overwritten.
inline constexpr std::int64_t kBelow = std::numeric_limits<std::int64_t>::min();
inline constexpr std::int64_t kAbove = std::numeric_limits<std::int64_t>::max();

inline void linbin(
    const double* x, const double* w, std::int64_t n, double lo, double delta, std::int64_t E, std::int64_t G, double* c
) {
    std::vector<std::int64_t> bin(n);
    std::vector<double> a(n), b(n);
    const double top = static_cast<double>(G - 1);
    const auto place = [&](std::int64_t j0, std::int64_t j1) {
        for (std::int64_t j = j0; j < j1; ++j) {
            const double pos = (x[j] - lo) / delta + static_cast<double>(E);
            const double wj = w ? w[j] : 1.0;
            if (!(pos >= 0.0 && pos <= top)) {  // left out: below the grid first, above it last (x is sorted)
                bin[j] = pos < 0.0 ? kBelow : kAbove;
                a[j] = b[j] = 0.0;
                continue;
            }
            double fl = std::floor(pos);
            if (fl >= top) fl = top - 1.0;
            const double f = pos - fl;
            bin[j] = static_cast<std::int64_t>(fl);
            a[j] = wj * (1.0 - f);
            b[j] = wj * f;
        }
    };
    // bin is nondecreasing (pos is monotone in x; the points left out sit at either end)
    const auto range_task = [&](std::int64_t g0, std::int64_t g1) {
        for (std::int64_t g = g0; g < g1; ++g) c[g] = 0.0;
        const auto first = std::lower_bound(bin.begin(), bin.end(), g0 - 1);
        for (std::int64_t j = first - bin.begin(); j < n; ++j) {
            const std::int64_t i = bin[j];
            if (i >= g1) break;
            if (i >= g0) c[i] += a[j];
            if (i + 1 >= g0 && i + 1 < g1) c[i + 1] += b[j];
        }
    };
    const std::int64_t work = 8 * n + G;
    if (work >= kParallelThreshold && n >= 4096 && parallel::available()) {
        const std::int64_t T = std::min<std::int64_t>(64, 2 * std::max(1, parallel::num_threads()));
        parallel::parallel_for(T, [&](std::int64_t t) { place(t * n / T, (t + 1) * n / T); }, work_ns(8 * n / T));
        std::vector<std::int64_t> cut(T + 1);
        cut[0] = 0, cut[T] = G;
        for (std::int64_t t = 1; t < T; ++t)
            cut[t] = std::max(cut[t - 1], std::clamp<std::int64_t>(bin[t * n / T], 0, G));
        parallel::parallel_for(
            T, [&](std::int64_t t) { range_task(cut[t], std::max(cut[t], cut[t + 1])); }, work_ns(n / T + G / T)
        );
    } else {
        place(0, n);
        range_task(0, G);
    }
}

}  // namespace sd::smooth::kde
