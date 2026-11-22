// Double-double building blocks of the chunk kernels (stats/chunk.hpp, docs/design/stats.md
// §3.4): the standardized variable, a ratio and a logarithm as hi/lo arrays, and the second
// tier of a fast path.
//
// The precise (scalar) definitions keep z = (x - mu) / sigma, x / s and log x as double-
// doubles (stats/numeric.hpp) because a tail multiplies their rounding. The loops here do the
// same arithmetic for a whole chunk without branches, so a fast path can keep the exponent's
// low part and only replace libm's exp/log by vForce's; its error bound is then vForce's
// own (ck::kVfExp, ck::kVfLog), not that times the condition of the formula.
//
// Conventions: every routine is elementwise (out[i] from in[i] only), so results do not
// depend on position or chunk length. An element a routine cannot handle (non-finite,
// overflowing, out of the reduction's range) gets hi = NaN: any bound computed from it is
// NaN, `bound <= kFastBudget` is false, and the element goes to the precise definition.
// Bounds are in units of kUlp = 2^-53: one correctly rounded operation costs 1 (half an ULP),
// libm's functions are taken as 2 (1 ULP) and vForce's as the constants below.
#pragma once

#include "special/dd.hpp"
#include "stats/chunk.hpp"
#include "stats/engine.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace sd::stats::ckd {

// ---- more of vForce, with error bounds ------------------------------------------------------
//
// Relative error in kUlp, measured against double-double references on 4e6 points or more each
// (bench/stats/opt/a/vforce_error.cpp at commit 4151d2a, which also measures ck::kVfExp and ck::kVfLog) and
// rounded up:
//                measured   bound
//     pow          2.08      2.5     (results in the normal range; bases near 1 with exponents
//                                     to 1e6, bases to e^+-600, exponents of both signs)
//     tanh         5.27      6.5     (arguments in [2^-60, 20]; 2.5e8 points: the largest
//                                     errors are below x = 0.35, 3.0 from there, 2.0 from 3)
//     atanh        3.77      4.5     (|x| <= 1/3)
//     atan         <= 4      4       (within 1 ULP of libm's on 4e6 points, |x| in [1e-300, 1e300])
// All four are on accelerate/vforce.hpp's list of position-independent functions. pow is
// tighter than exp of a log (kVfExp + kVfLog |log|): it carries the logarithm's low bits.
inline constexpr double kPow = 2.5;
inline constexpr double kTanh = 6.5;
inline constexpr double kAtanh = 4.5;
inline constexpr double kAtan = 4.0;

inline void tanh(double* out, const double* x, int n) { vvtanh(out, x, &n); }
inline void atanh(double* out, const double* x, int n) { vvatanh(out, x, &n); }
inline void atan(double* out, const double* x, int n) { vvatan(out, x, &n); }

// out = 1 - e^-t for t >= 0 (t = inf: 1): 2 T / (1 + T), T = tanh(t / 2). T's error reaches
// the result times 1 / (1 + T) <= 1, then a sum and a quotient: kOneMinusExp = kTanh + 2.
// (libm's expm1 costs 5 ns an element on mixed arguments, this 1.5.) out may be t.
inline constexpr double kOneMinusExp = kTanh + 2.0;
inline void one_minus_exp(double* out, const double* t, int n) {
    for (int i = 0; i < n; ++i) out[i] = 0.5 * t[i];
    tanh(out, out, n);
    for (int i = 0; i < n; ++i) out[i] = 2.0 * out[i] / (1.0 + out[i]);
}

// out = log(1 + x) for -1/2 <= x <= 1: 2 atanh(x / (2 + x)). The argument |s| <= 1/3 carries
// 2 units (a sum and a quotient), atanh multiplies them by s / ((1 - s^2) atanh s) <= 1.09
// and adds its own: kLog1p = 2.2 + kAtanh. An error of x itself reaches the result times
// x / ((1 + x) log(1 + x)), at most 1 for x >= 0 and 1 / (1 + x) for x < 0. out may be x.
inline constexpr double kLog1p = 2.2 + kAtanh;
inline void log1p(double* out, const double* x, int n) {
    for (int i = 0; i < n; ++i) out[i] = x[i] / (2.0 + x[i]);
    atanh(out, out, n);
    for (int i = 0; i < n; ++i) out[i] *= 2.0;
}

// (zh, zl) = (x - mu) / sigma: num::standardize's arithmetic (two_sum, the quotient and the
// exact remainder by fma), bit for bit, wherever x - mu is finite, at least 2^-960 in size (or
// 0: below that the remainder is not a double) and the quotient neither overflows nor
// underflows to 0; hi = NaN elsewhere (num::standardize has other forms there). Relative
// error of the pair ~2^-104.
inline void standardize(const double* x, const double* mu, const double* sigma, double* zh, double* zl, int n) {
    for (int i = 0; i < n; ++i) {
        const double a = x[i], b = -mu[i], s = sigma[i];
        const double d = a + b;
        const double bb = d - a;
        const double dl = (a - (d - bb)) + (b - bb);
        const double q = d / s;
        const double r = std::fma(-q, s, d) + dl;
        const double rl = r / s;
        const double h = q + rl;
        // (a quotient below 2^-960 has a low part in the subnormals: left to num::standardize's callers)
        const bool ok = std::isfinite(q) && (std::fabs(d) >= 0x1p-960 ? std::fabs(q) >= 0x1p-960 : d == 0.0);
        zh[i] = ok ? h : kNaN;
        zl[i] = ok ? rl - (h - q) : 0.0;
    }
}

// (yh, yl) = x / s: num::ratio's arithmetic, bit for bit, for |x| >= 2^-960 and a finite
// nonzero quotient (and for x = 0); hi = NaN elsewhere.
inline void ratio(const double* x, const double* s, double* yh, double* yl, int n) {
    for (int i = 0; i < n; ++i) {
        const double q = x[i] / s[i];
        const double rl = std::fma(-q, s[i], x[i]) / s[i];
        const double h = q + rl;
        const bool ok = std::isfinite(q) && (std::fabs(x[i]) >= 0x1p-960 ? std::fabs(q) >= 0x1p-960 : x[i] == 0.0);
        yh[i] = ok ? h : kNaN;
        yl[i] = ok ? rl - (h - q) : 0.0;
    }
}

// The relative error of log_dd below, in kUlp: 2^-59. Measured against special::dd::log
// (2^-77) over 6e6 points (the whole range [2^-999, 2^999], the neighbourhood of 1 down to
// |x - 1| = 2^-53, random low parts): at most 2^-68 (bench/stats/opt/a/log_dd.cpp at commit 4151d2a).
inline constexpr double kLq = 0x1p-6;

// (oh, ol) = log(xh + xl) for a normalised double-double in [2^-999, 2^999]; hi = NaN outside
// (0, negative, subnormal, huge, inf, NaN), where the caller's precise path takes over.
//
// special::dd's reduction (dd.hpp `reduce`): x = 2^k c (1 + r), c = j/128 the table point
// nearest the mantissa in [3/4, 3/2), r = (m - c)/c as hi + lo (m - c exact, the table's
// rounded 1/c corrected by one fma), |r| <= 2^-7.58. Then
//     log(1 + r) = r - r^2/2 + r^3 (1/3 - r/4 + ... + r^6/9),
// r^2 with its rounding (fma) and the cubic tail in double: the tail is below 2^-16 of r, so
// its few ULP are 2^-68 of the result, and the series is cut at r^10/10 < 2^-79. k log 2,
// log c (kLogTable) and r are summed by two_sum with every low part gathered in one double
// (they are 2^-53 of the terms). dd::log carries r^2 and r^3 as double-doubles and normalises
// each sum: 7 ns an element against 3.5 here, and 2^-77 against 2^-68.
inline void log_dd_one(double xh, double xl, double& oh, double& ol) {
    namespace dd = special::dd;
    std::uint64_t bits;
    std::memcpy(&bits, &xh, sizeof bits);
    const int e = static_cast<int>(bits >> 52);  // the sign and the biased exponent
    const bool ok = e > 23 && e < 2023;
    int k = e - 1023;
    bits = (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m;
    std::memcpy(&m, &bits, sizeof m);  // in [1, 2)
    const bool up = m >= 1.5;
    m = up ? 0.5 * m : m;
    k += up ? 1 : 0;
    const std::uint64_t sbits = static_cast<std::uint64_t>((1023 - k) & 0x7ff) << 52;
    double scale;
    std::memcpy(&scale, &sbits, sizeof scale);  // 2^-k
    const int j = static_cast<int>(m * 128.0 + 0.5);  // 96 .. 192
    const dd::LogPoint& c = dd::kLogTable[static_cast<std::size_t>(j - 96)];
    const double cv = j * 0x1p-7;
    const double d = m - cv;  // exact
    const double q0 = d * c.inv;
    const double q1 = (std::fma(-q0, cv, d) + xl * scale) * c.inv;
    const double rh = q0 + q1;  // r, normalised (|q0| >= |q1|, or q0 = 0)
    const double rl = q1 - (rh - q0);
    const double r2 = rh * rh;
    const double r2l = std::fma(rh, rh, -r2) + 2.0 * rh * rl;
    const double poly =
        1.0 / 3 + rh * (-0.25 + rh * (0.2 + rh * (-1.0 / 6 + rh * (1.0 / 7 + rh * (-0.125 + rh * (1.0 / 9))))));
    const double kd = static_cast<double>(k);
    const double kh = kd * dd::kLn2.hi;
    const double kl = std::fma(kd, dd::kLn2.hi, -kh) + kd * dd::kLn2.lo;
    const double s1 = kh + c.hi;
    const double b1 = s1 - kh;
    const double e1 = (kh - (s1 - b1)) + (c.hi - b1);
    const double s2 = s1 + rh;
    const double b2 = s2 - s1;
    const double e2 = (s1 - (s2 - b2)) + (rh - b2);
    const double h2 = -0.5 * r2;
    const double s3 = s2 + h2;
    const double b3 = s3 - s2;
    const double e3 = (s2 - (s3 - b3)) + (h2 - b3);
    const double lo = ((kl + c.lo) + (e1 + e2)) + ((e3 + (rl - 0.5 * r2l)) + (r2 * rh) * poly);
    const double h = s3 + lo;
    oh = ok ? h : kNaN;
    ol = ok ? lo - (h - s3) : 0.0;
}

#if defined(__aarch64__)
// Two elements as NEON lanes: the same operations in the same order as log_dd_one (IEEE
// arithmetic and fma per lane), so the same bits.
inline void log_dd_two(const double* xh, const double* xl, double* oh, double* ol) {
    namespace dd = special::dd;
    const auto f = [](double v) { return vdupq_n_f64(v); };
    const uint64x2_t bits0 = vreinterpretq_u64_f64(vld1q_f64(xh));
    const uint64x2_t e = vshrq_n_u64(bits0, 52);
    const uint64x2_t ok = vandq_u64(vcgtq_u64(e, vdupq_n_u64(23)), vcltq_u64(e, vdupq_n_u64(2023)));
    const uint64x2_t mb =
        vorrq_u64(vandq_u64(bits0, vdupq_n_u64(0x000fffffffffffffULL)), vdupq_n_u64(0x3ff0000000000000ULL));
    float64x2_t m = vreinterpretq_f64_u64(mb);
    const uint64x2_t up = vcgeq_f64(m, f(1.5));
    m = vbslq_f64(up, vmulq_f64(f(0.5), m), m);
    // k = e - 1023 + up (up is all ones = -1 where set)
    const int64x2_t k = vsubq_s64(vsubq_s64(vreinterpretq_s64_u64(e), vdupq_n_s64(1023)), vreinterpretq_s64_u64(up));
    const uint64x2_t sb =
        vshlq_n_u64(vandq_u64(vreinterpretq_u64_s64(vsubq_s64(vdupq_n_s64(1023), k)), vdupq_n_u64(0x7ff)), 52);
    const float64x2_t scale = vreinterpretq_f64_u64(sb);
    const int64x2_t j = vcvtq_s64_f64(vaddq_f64(vmulq_f64(m, f(128.0)), f(0.5)));  // truncates: 96 .. 192
    const dd::LogPoint& c0 = dd::kLogTable[static_cast<std::size_t>(vgetq_lane_s64(j, 0) - 96)];
    const dd::LogPoint& c1 = dd::kLogTable[static_cast<std::size_t>(vgetq_lane_s64(j, 1) - 96)];
    const float64x2_t chi = vcombine_f64(vld1_f64(&c0.hi), vld1_f64(&c1.hi));
    const float64x2_t clo = vcombine_f64(vld1_f64(&c0.lo), vld1_f64(&c1.lo));
    const float64x2_t cinv = vcombine_f64(vld1_f64(&c0.inv), vld1_f64(&c1.inv));
    const float64x2_t cv = vmulq_f64(vcvtq_f64_s64(j), f(0x1p-7));
    const float64x2_t d = vsubq_f64(m, cv);
    const float64x2_t q0 = vmulq_f64(d, cinv);
    const float64x2_t q1 = vmulq_f64(vaddq_f64(vfmsq_f64(d, q0, cv), vmulq_f64(vld1q_f64(xl), scale)), cinv);
    const float64x2_t rh = vaddq_f64(q0, q1);
    const float64x2_t rl = vsubq_f64(q1, vsubq_f64(rh, q0));
    const float64x2_t r2 = vmulq_f64(rh, rh);
    const float64x2_t r2l = vaddq_f64(vfmaq_f64(vnegq_f64(r2), rh, rh), vmulq_f64(vmulq_f64(f(2.0), rh), rl));
    float64x2_t poly = vmulq_f64(rh, f(1.0 / 9));
    poly = vmulq_f64(rh, vaddq_f64(f(-0.125), poly));
    poly = vmulq_f64(rh, vaddq_f64(f(1.0 / 7), poly));
    poly = vmulq_f64(rh, vaddq_f64(f(-1.0 / 6), poly));
    poly = vmulq_f64(rh, vaddq_f64(f(0.2), poly));
    poly = vmulq_f64(rh, vaddq_f64(f(-0.25), poly));
    poly = vaddq_f64(f(1.0 / 3), poly);
    const float64x2_t kd = vcvtq_f64_s64(k);
    const float64x2_t kh = vmulq_f64(kd, f(dd::kLn2.hi));
    const float64x2_t kl = vaddq_f64(vfmaq_f64(vnegq_f64(kh), kd, f(dd::kLn2.hi)), vmulq_f64(kd, f(dd::kLn2.lo)));
    const float64x2_t s1 = vaddq_f64(kh, chi);
    const float64x2_t b1 = vsubq_f64(s1, kh);
    const float64x2_t e1 = vaddq_f64(vsubq_f64(kh, vsubq_f64(s1, b1)), vsubq_f64(chi, b1));
    const float64x2_t s2 = vaddq_f64(s1, rh);
    const float64x2_t b2 = vsubq_f64(s2, s1);
    const float64x2_t e2 = vaddq_f64(vsubq_f64(s1, vsubq_f64(s2, b2)), vsubq_f64(rh, b2));
    const float64x2_t h2 = vmulq_f64(f(-0.5), r2);
    const float64x2_t s3 = vaddq_f64(s2, h2);
    const float64x2_t b3 = vsubq_f64(s3, s2);
    const float64x2_t e3 = vaddq_f64(vsubq_f64(s2, vsubq_f64(s3, b3)), vsubq_f64(h2, b3));
    const float64x2_t lo = vaddq_f64(
        vaddq_f64(vaddq_f64(kl, clo), vaddq_f64(e1, e2)),
        vaddq_f64(vaddq_f64(e3, vsubq_f64(rl, vmulq_f64(f(0.5), r2l))), vmulq_f64(vmulq_f64(r2, rh), poly))
    );
    const float64x2_t h = vaddq_f64(s3, lo);
    vst1q_f64(oh, vbslq_f64(ok, h, f(kNaN)));
    vst1q_f64(ol, vbslq_f64(ok, vsubq_f64(lo, vsubq_f64(h, s3)), f(0.0)));
}
#endif

inline void log_dd(const double* xh, const double* xl, double* oh, double* ol, int n) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 2 <= n; i += 2) log_dd_two(xh + i, xl + i, oh + i, ol + i);
#endif
    for (; i < n; ++i) log_dd_one(xh[i], xl[i], oh[i], ol[i]);
}

// out = (yh + yl)^e for a double-double base and |e| <= 2^30: vForce's pow of the high part
// times e^w = 1 + w + w^2/2, w = e yl / yh (|w| <= 2^-23: the cubic term is below 2^-71, and
// w^2/2 alone reaches 2^-47, 64 units): relative error kPowDd = kPow + 2 (the sum with 1 and
// the final product; the roundings inside w are 2^-53 of a term below 2^-23). NaN in gives
// NaN.
inline constexpr double kPowDd = kPow + 2.0;
inline void pow_dd(double* out, const double* yh, const double* yl, const double* e, int n) {
    ck::pow(out, yh, e, n);
    for (int i = 0; i < n; ++i) {
        const double w = e[i] * (yl[i] / yh[i]);
        out[i] *= 1.0 + w * (1.0 + 0.5 * w);
    }
}

// out = exp(eh + el), |el| << 1: ck::exp of the high part times (1 + el), relative error
// kVfExp + 1 (num::exp_dd with libm's exp replaced).
inline void exp_dd(double* out, const double* eh, const double* el, int n) {
    ck::exp(out, eh, n);
    for (int i = 0; i < n; ++i) out[i] *= 1.0 + el[i];
}

// A parameter-only constant as hi/lo arrays: computed once when every parameter is a
// broadcast scalar, else per element. f(i) returns a num::DD-like {hi, lo}.
template <std::size_t NP, class F>
inline void constant(const Chunk<NP>& c, double* kh, double* kl, F&& f) {
    bool same = true;
    for (bool s : c.same) same = same && s;
    if (same) {
        const auto v = f(0);
        for (int i = 0; i < c.n; ++i) kh[i] = v.hi, kl[i] = v.lo;
    } else {
        for (int i = 0; i < c.n; ++i) {
            const auto v = f(i);
            kh[i] = v.hi, kl[i] = v.lo;
        }
    }
}

// The second tier of a fast path: `kernel(sub, out, fast)` is run again on the elements with
// !fast[i], gathered into a contiguous sub-chunk, and its results taken where it reports
// fast. For a kernel that is cheap but narrow first (plain double, vForce's log) and a wider
// one behind it (double-double log), before the precise definition. Each element's result
// comes from the first tier that accepts it, whatever its neighbours, so f(x)[i] == f(x[i]).
template <std::size_t NP, class K>
inline void retry(const Chunk<NP>& c, double* out, bool* fast, K&& kernel) {
    int idx[kChunkSize];
    int m = 0;
    for (int i = 0; i < c.n; ++i) {
        idx[m] = i;
        m += fast[i] ? 0 : 1;
    }
    if (m == 0) return;
    double xb[kChunkSize], ob[kChunkSize], pb[NP][kChunkSize];
    bool fb[kChunkSize];
    Chunk<NP> sub{m, xb, {}, c.same};
    for (int j = 0; j < m; ++j) xb[j] = c.x[idx[j]];
    for (std::size_t k = 0; k < NP; ++k) {
        if (c.same[k]) {
            sub.p[k] = c.p[k];  // all equal: any m of them
        } else {
            for (int j = 0; j < m; ++j) pb[k][j] = c.p[k][idx[j]];
            sub.p[k] = pb[k];
        }
    }
    kernel(sub, ob, fb);
    for (int j = 0; j < m; ++j) {
        if (fb[j]) out[idx[j]] = ob[j], fast[idx[j]] = true;
    }
}

}  // namespace sd::stats::ckd
