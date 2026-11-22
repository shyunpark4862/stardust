// The terms of the Cressie-Read power divergence statistics (docs/design/htest.md §3.3), shared
// by power_divergence (htest/gof.cpp) and the contingency tables' chi-squared tests
// (median_test's, htest/rank.cpp; chi2_contingency's, htest/contingency.cpp). A zero count
// with lambda <= -1 contributes +inf (the limit, power_divergence's rule) everywhere.
//
// With delta = (O - E)/E and a = lambda + 1, every Cressie-Read statistic is
// 2 sum E g_a(delta) + (2/a) sum (O - E), with
//     g_a(delta) = ((1 + delta)^a - 1 - a delta) / (a (a - 1)) >= 0
// (g_1 = (1 + delta) log(1 + delta) - delta: the G test; the limit a = 0 is the modified one,
// 2 sum E log(E/O) = 2 sum E g_0 - 2 sum (O - E)). The terms of the first sum are never negative,
// so a statistic near 0 (O close to E) keeps its digits.
#pragma once

#include "htest/moments.hpp"
#include "special/dd.hpp"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace sd::htest::div {

// atanh's series past its first term: S(v) = sum_{k >= 0} v^k / (2k + 3), v <= 1/9: 18 terms
// (the next is below 1e-18 of the sum; every v takes all of them, as the NEON blocks below do)
// (Top + 1 coefficients, Top odd: the even and odd ones as two Horner chains in v^2, half the
// latency of one chain; FMA steps, as the NEON form below)
template <int Top>
inline double atanh_tail_n(double v) {
    static_assert(Top % 2 == 1);
    const double v2 = v * v;
    double e = 1.0 / (2.0 * (Top - 1) + 3.0), o = 1.0 / (2.0 * Top + 3.0);
    for (int k = Top - 3; k >= 0; k -= 2) {
        e = std::fma(e, v2, 1.0 / (2.0 * k + 3.0));
        o = std::fma(o, v2, 1.0 / (2.0 * (k + 1) + 3.0));
    }
    return e + v * o;
}
inline double atanh_tail(double v) { return atanh_tail_n<17>(v); }

// g_1 through a double-double mu(t) = t - log1p(t) (special/dd.hpp): ~1 ulp. g_1 takes it for
// delta in (1, 4] (the direct form there loses up to three bits), g_a with a < 1 everywhere (g_1
// meets a term of the other sign there, psi < 0).
inline double g1_dd(double t) {
    if (t == -1.0) return 1.0;
    if (!std::isfinite(t)) return t;
    const DD mu = special::dd::log1pmx_neg(DD{t, 0.0});
    const DD v = dd::two_prod(t, t) - mu - mu * t;  // (1 + t) mu = mu + t mu
    return v.hi + v.lo;
}

// g_1(delta) = (1 + delta) log1p(delta) - delta >= 0. For delta in [-1/2, 1], with u = delta /
// (2 + delta) in [-1/3, 1/3] (log1p(delta) = 2 atanh u, 1 + delta = (1 + u) / (1 - u)):
//     g_1 = 2 u^2 (1 + u (1 + u) S(u^2)) / (1 - u),
// positive terms without the cancellation of the direct form near 0 (the bracket stays in
// [0.92, 1.05]). u's rounding (it would count twice in u^2) is taken back to first order (below).
// A few ulps whatever delta (measured: 4 against the double-double form); (1, 4] by g1_dd, and
// elsewhere the direct form loses at most two bits.
inline double g1(double t) {
    if (t == -1.0) return 1.0;
    if (!std::isfinite(t)) return t;
    if (t >= -0.5 && t <= 1.0) {
        // 1 / (1 - u) = (2 + delta) / 2 (no quotient, the exact u's); the rounded u's in u^2 taken
        // back to first order: u + r / (2 + delta) = delta / (2 + delta) with r from the exact
        // remainders moves g by g 2 r / delta, and 2 + delta's low part by g lo / (2 + delta). A
        // correction ~1e-16 of g needs few bits: the reciprocals are NEON's estimates (2^-8).
        const DD den = dd::two_sum(2.0, t);
        const double u = t / den.hi;
        const double r = std::fma(-u, den.hi, t) - u * den.lo;
        const double v = u * u;
        const double g = v * (1.0 + u * (1.0 + u) * atanh_tail(v)) * den.hi;
        if (!(std::fabs(t) >= 0x1p-500)) return g;
        return g + g * (2.0 * r * vrecped_f64(t) + den.lo * vrecped_f64(den.hi));
    }
    if (t > 1.0 && t <= 4.0) return g1_dd(t);
    return (1.0 + t) * std::log1p(t) - t;
}

// 1/k! for k = 0..19
inline constexpr double kInvFact[20] = {
    1.0,
    1.0,
    1.0 / 2,
    1.0 / 6,
    1.0 / 24,
    1.0 / 120,
    1.0 / 720,
    1.0 / 5040,
    1.0 / 40320,
    1.0 / 362880,
    1.0 / 3628800,
    1.0 / 39916800,
    1.0 / 479001600,
    1.0 / 6227020800.0,
    1.0 / 87178291200.0,
    1.0 / 1307674368000.0,
    1.0 / 20922789888000.0,
    1.0 / 355687428096000.0,
    1.0 / 6402373705728000.0,
    1.0 / 121645100408832000.0
};

// psi(y) = (e^y - 1 - y) / y = sum_{k >= 2} y^(k-1) / k!: for |y| < 1/2 by Horner to k = 19 (the
// next term is below 1e-23 of the sum), else from expm1
inline double psi(double y) {
    if (std::fabs(y) < 0.5) {  // (the even and odd terms as two chains in y^2)
        const double y2 = y * y;
        double e = kInvFact[18], o = kInvFact[19];
        for (int k = 16; k >= 2; k -= 2) e = std::fma(e, y2, kInvFact[k]), o = std::fma(o, y2, kInvFact[k + 1]);
        return (e + y * o) * y;
    }
    return (std::expm1(y) - y) / y;
}

// g_a(delta) for delta > -1 (O > 0), a != 1
inline double ga(double a, double t) {
    if (t == -1.0) return a > 0.0 ? 1.0 / a : std::numeric_limits<double>::infinity();
    if (a < 0.5) {
        // g_a(t) = (1 + t) g_(1-a)(-t / (1 + t))
        return (1.0 + t) * ga(1.0 - a, -t / (1.0 + t));
    }
    if (a == 1.0) return g1(t);
    // (a >= 1: g_1 and the psi term have one sign, the form below keeps the digits for any t)
    if (a < 1.0 && std::fabs(t) <= 0x1p-4) {
        // sum_{k >= 2} c_k t^k, c_2 = 1/2, c_(k+1) = c_k (a - k) / (k + 1): for a in [1/2, 1)
        // and |t| <= 1/16 the term k is below 16^-(k-2) of the first, so 24 terms reach 1e-26 of
        // it; Horner from the last
        // (the coefficients of the last a, per thread: a test calls with one a for every cell)
        thread_local double last = std::numeric_limits<double>::quiet_NaN(), c[25];
        if (!(a == last)) {
            c[2] = 0.5;
            for (int k = 2; k < 24; ++k) c[k + 1] = c[k] * (a - k) / (k + 1.0);
            last = a;
        }
        double s = c[24];
        for (int k = 23; k >= 2; --k) s = s * t + c[k];
        return s * t * t;
    }
    const double L = std::log1p(t);
    return ((a < 1.0 ? g1_dd(t) : g1(t)) + (1.0 + t) * L * psi((a - 1.0) * L)) / a;
}

// ---- blocks of terms ---------------------------------------------------------------------
//
// g_1 and g_a of an array of delta, two at a time in NEON where the scalar forms above take
// their series (delta in [-1/2, 1], not tiny; for g_a, a >= 1 and |(a - 1) log1p delta| < 1/2): the
// same operations lane by lane, log1p from vForce (Accelerate's vvlog1p, the array of the block;
// within an ulp). Any other element is the scalar form's. kGBlock elements at most.
inline constexpr int kGBlock = 256;

namespace gvec {
using V = float64x2_t;
inline V dup(double x) { return vdupq_n_f64(x); }
// atanh_tail_n<17> lane by lane
inline V atanh_tail(V v) {
    const V v2 = vmulq_f64(v, v);
    V e = dup(1.0 / (2.0 * 16 + 3.0)), o = dup(1.0 / (2.0 * 17 + 3.0));
    for (int k = 14; k >= 0; k -= 2) {
        e = vfmaq_f64(dup(1.0 / (2.0 * k + 3.0)), e, v2);
        o = vfmaq_f64(dup(1.0 / (2.0 * (k + 1) + 3.0)), o, v2);
    }
    return vaddq_f64(e, vmulq_f64(v, o));
}
// g1's series branch lane by lane, and the lanes it holds for
inline V g1(V t, uint64x2_t& ok) {
    const V two = dup(2.0);
    const V s = vaddq_f64(two, t), bb = vsubq_f64(s, two);
    const V lo = vaddq_f64(vsubq_f64(two, vsubq_f64(s, bb)), vsubq_f64(t, bb));
    const V u = vdivq_f64(t, s);
    const V r = vsubq_f64(vfmaq_f64(t, vnegq_f64(u), s), vmulq_f64(u, lo));
    const V v = vmulq_f64(u, u);
    const V br = vaddq_f64(dup(1.0), vmulq_f64(vmulq_f64(u, vaddq_f64(dup(1.0), u)), atanh_tail(v)));
    const V g = vmulq_f64(vmulq_f64(v, br), s);
    const V c = vaddq_f64(vmulq_f64(vmulq_f64(two, r), vrecpeq_f64(t)), vmulq_f64(lo, vrecpeq_f64(s)));
    ok = vandq_u64(vandq_u64(vcgeq_f64(t, dup(-0.5)), vcleq_f64(t, dup(1.0))), vcgeq_f64(vabsq_f64(t), dup(0x1p-500)));
    return vaddq_f64(g, vmulq_f64(g, c));
}
// psi's series lane by lane (|y| < 1/2), and those lanes
inline V psi(V y, uint64x2_t& ok) {
    const V y2 = vmulq_f64(y, y);
    V e = dup(kInvFact[18]), o = dup(kInvFact[19]);
    for (int k = 16; k >= 2; k -= 2) {
        e = vfmaq_f64(dup(kInvFact[k]), e, y2);
        o = vfmaq_f64(dup(kInvFact[k + 1]), o, y2);
    }
    ok = vcltq_f64(vabsq_f64(y), dup(0.5));
    return vmulq_f64(vaddq_f64(e, vmulq_f64(y, o)), y);
}
}  // namespace gvec

// g[i] = g_1(t[i]) for i < n: the vector pass over pairs (no branch: independent pairs overlap),
// then the scalar form where a lane is outside the series
inline void g1_block(const double* t, double* g, int n) {
    std::uint64_t okm[kGBlock / 2];
    int i = 0;
    for (; i + 4 <= n; i += 4) {  // (two pairs in one body: their chains interleave)
        uint64x2_t ok0, ok1;
        const gvec::V r0 = gvec::g1(vld1q_f64(t + i), ok0), r1 = gvec::g1(vld1q_f64(t + i + 2), ok1);
        vst1q_f64(g + i, r0), vst1q_f64(g + i + 2, r1);
        okm[i / 2] = (vgetq_lane_u64(ok0, 0) & 1u) | (vgetq_lane_u64(ok0, 1) & 2u);
        okm[i / 2 + 1] = (vgetq_lane_u64(ok1, 0) & 1u) | (vgetq_lane_u64(ok1, 1) & 2u);
    }
    for (; i + 2 <= n; i += 2) {
        uint64x2_t ok;
        vst1q_f64(g + i, gvec::g1(vld1q_f64(t + i), ok));
        okm[i / 2] = (vgetq_lane_u64(ok, 0) & 1u) | (vgetq_lane_u64(ok, 1) & 2u);
    }
    for (int j = 0; j + 2 <= n; j += 2) {
        if (okm[j / 2] == 3u) continue;
        if (!(okm[j / 2] & 1u)) g[j] = g1(t[j]);
        if (!(okm[j / 2] & 2u)) g[j + 1] = g1(t[j + 1]);
    }
    for (; i < n; ++i) g[i] = g1(t[i]);
}

// g[i] = g_a(t[i]) for i < n (a != 1)
inline void ga_block(double a, const double* t, double* g, int n) {
    if (!(a >= 1.0)) {
        for (int i = 0; i < n; ++i) g[i] = ga(a, t[i]);
        return;
    }
    alignas(16) double L[kGBlock];
    vvlog1p(L, t, &n);
    const gvec::V am1 = gvec::dup(a - 1.0), va = gvec::dup(a), one = gvec::dup(1.0);
    std::uint64_t okm[kGBlock / 2];
    auto pair = [&](int j) {
        const gvec::V tv = vld1q_f64(t + j), lv = vld1q_f64(L + j);
        uint64x2_t ok1, ok2;
        const gvec::V g1v = gvec::g1(tv, ok1);
        const gvec::V pv = gvec::psi(vmulq_f64(am1, lv), ok2);
        vst1q_f64(g + j, vdivq_f64(vaddq_f64(g1v, vmulq_f64(vmulq_f64(vaddq_f64(one, tv), lv), pv)), va));
        const uint64x2_t ok = vandq_u64(ok1, ok2);
        okm[j / 2] = (vgetq_lane_u64(ok, 0) & 1u) | (vgetq_lane_u64(ok, 1) & 2u);
    };
    int i = 0;
    for (; i + 4 <= n; i += 4) pair(i), pair(i + 2);  // (two pairs in one body: their chains interleave)
    for (; i + 2 <= n; i += 2) pair(i);
    for (int j = 0; j + 2 <= n; j += 2) {
        if (okm[j / 2] == 3u) continue;
        if (!(okm[j / 2] & 1u)) g[j] = ga(a, t[j]);
        if (!(okm[j / 2] & 2u)) g[j + 1] = ga(a, t[j + 1]);
    }
    for (; i < n; ++i) g[i] = ga(a, t[i]);
}

// scipy's term for the pairs a rearranged form does not cover (E <= 0 or O < 0)
inline double xlogy(double x, double y) { return (x == 0.0 && y == y) ? 0.0 : x * std::log(y); }
inline double scipy_term(double lam, double o, double e) {
    if (lam == 1.0) return (o - e) * (o - e) / e;
    if (lam == 0.0) return 2.0 * xlogy(o, o / e);
    if (lam == -1.0) return 2.0 * xlogy(e, e / o);
    return o * (std::pow(o / e, lam) - 1.0) / (0.5 * lam * (lam + 1.0));
}

// The statistic of cells with expected counts E (double-doubles, > 0) and observed counts
// E + d (d the exact difference O - E as a double-double, O >= 0): the rearranged sum above.
struct Divergence {
    double lam;
    Acc pos, lin;
    explicit Divergence(double lambda) : lam(lambda) {}
    void add(DD E, DD d) {
        if (lam == 1.0) {
            pos.add((d * d) / E);
            return;
        }
        const DD q = d / E;
        const double t = q.hi + q.lo;
        const double aa = lam + 1.0;
        const double g = lam == 0.0 ? g1(t) : (lam == -1.0 ? ga(0.0, t) : ga(aa, t));
        // a zero count with lambda <= -1 contributes its limit +inf (power_divergence's rule)
        if (std::isfinite(g)) pos.add(E * (2.0 * g));
        else pos.add(2.0 * E.hi * g);
        lin.add(d);
    }
    double value() const {
        const double coef = lam == -1.0 ? -2.0 : (lam == 1.0 ? 0.0 : 2.0 / (lam + 1.0));
        Acc tot = pos;
        tot.add(lin.value() * coef);
        return to_double(tot.value());
    }
};

}  // namespace sd::htest::div
