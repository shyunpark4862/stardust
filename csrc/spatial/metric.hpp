// The pair kernels of sd.spatial (docs/design/spatial.md §3.1; decisions SPT22, SPT26, SPT27, SPT29-SPT33).
//
// One kernel decides the distance of a pair: pdist, cdist and the distance functions compute every pair
// (x, y) in the same order, so the same pair has the same bits wherever it is computed (SPT2). The order
// is the tile order (SPT26): a pair's n terms in blocks of kBlock = 32, each block a sequential chain
// from 0 (t = op(t, term)), the block sums added one after another (s = t_1, s = s + t_2, ...).
//
//   tile<Op, R, W>     R rows of x against a packed panel of 8 rows of y: the R x 8 pairs side by side
//                      in NEON registers (two pairs a register), each pair's own sequential chain, so a
//                      lane holds exactly what pair_scalar<Op, W> computes for that pair. R = 4 for the
//                      ops with one accumulator, 2 for those with two (16 accumulator registers).
//   pair_scalar<Op, W> one pair, the same order (the distance functions, the careful paths' first pass).
//
// An Op has K accumulators and a step on doubles and on float64x2_t built from the same IEEE operations
// (sub, abs, fma, add, div, max, compares), so a lane and the scalar give the same bits (a NaN's payload
// aside). W: weighted (w[k] given), else w is 1.
//
// The metrics and their finish (per pair, from the sums):
//   euclidean    Sq (W: WSq, fma(w z, z, t)); sqrt. seuclidean is WSq with w = 1/V (SPT32).
//                Careful path (SPT27) when s is inf or below kTiny: z scaled by the power of two of
//                max |z_i|, the same order, sqrt, scaled back (exact powers of two).
//   sqeuclidean  the same sum; careful path below kTiny (the true value may still be subnormal).
//   cityblock    Abs: t + |z| (W: fma(w, |z|, t)).
//   chebyshev    Max: max |z| with NaN propagated (FMAX); W: over w_i > 0 only.
//   canberra     |z| / (|x| + |y|), a zero denominator gives 0 (both are 0); W: fma(w, q, t).
//   braycurtis   two sums, |z| and |x + y| (W: both weighted); their quotient (0/0 NaN, x/0 inf).
//   hamming      sum of w [x != y] (w = 1: an exact count); divided by n or sum w.
//   jaccard      sums of w [(x != 0) xor (y != 0)] and w [(x != 0) or (y != 0)]; a zero denominator 0.
//   cosine       rows normalized once (x / ||x||_w, the norm by the euclidean kernel with its careful
//                path), then the WSq kernel and d = s / 2 clipped to 2 (SPT29: ||x^ - y^||^2 / 2, no
//                cancellation of 1 - c for nearly parallel rows). correlation centers the rows first
//                (weighted mean; a constant row gets its value as the mean exactly, so it is NaN).
//   minkowski    sum of |z|^p (std::pow, p = 3 (|z| |z|) |z|; W: fma(w, |z|^p, t), and w z for w = 0 so a NaN or inf
//                there still propagates), then the root (SPT30: pow(s, 1/p) and, when 1/p is not a
//                power of two, one Newton step against pow(r, p)). Careful path: |z_i| / max |z| (a
//                power of two would overflow or underflow |z'|^p for large p), times max |z|.
//   mahalanobis  t_i = sum_j VI_ij z_j (tile order), q = sum_i z_i t_i (tile order), sqrt (q < 0: NaN);
//                careful path as euclidean on z.
//   haversine    SPT31, SPT81, SPT88: h = sin^2(dphi/2) + cos phi1 cos phi2 sin^2(dlam/2) and g = 1 - h
//                written as cos^2(dphi/2) cos^2(dlam/2) + sin^2(S/2) sin^2(dlam/2), S = phi1 + phi2: both
//                sums of nonnegative terms; d = 2 asin(sqrt(h)) for h <= 1/2, else pi - 2 asin(sqrt(g)), so
//                near the antipode g keeps its digits (sklearn's 2 asin(sqrt(h)) loses them there); dlam
//                exact by TwoSum (the dateline).
#pragma once

#include "base/sincos.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sd::spatial {

using std::int64_t;

inline constexpr int64_t kBlock = 32;  // the tile order's block of terms (SPT26)
inline constexpr int kPanel = 8;       // y rows of a packed panel
// The careful paths start below this sum (2^-968): above it the squares (or powers) that underflowed
// add at most n 2^-1075, relatively below 2^-107 for n < 2^40.
inline constexpr double kTiny = 0x1p-968;
inline constexpr double kHuge = std::numeric_limits<double>::max();

// ---- ops -----------------------------------------------------------------------------------

inline float64x2_t f2_zero() { return vdupq_n_f64(0.0); }

// t + term for the sums, max for Max (one block's chain, and the blocks' combination)
struct SumCombine {
    static float64x2_t combine(float64x2_t s, float64x2_t t) { return vaddq_f64(s, t); }
    static double combine(double s, double t) { return s + t; }
};

inline double max1(double a, double b) {  // FMAX of one lane: a NaN propagates, as vmaxq_f64
    return vget_lane_f64(vmax_f64(vdup_n_f64(a), vdup_n_f64(b)), 0);
}

struct SqOp : SumCombine {  // euclidean, sqeuclidean (unweighted)
    static constexpr int K = 1;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t) {
        const float64x2_t z = vsubq_f64(x, y);
        t[0] = vfmaq_f64(t[0], z, z);
    }
    static void step(double* t, double x, double y, double) {
        const double z = x - y;
        t[0] = std::fma(z, z, t[0]);
    }
};

struct WSqOp : SumCombine {  // weighted euclidean and sqeuclidean, seuclidean (w = 1/V), cosine
    static constexpr int K = 1;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t w) {
        const float64x2_t z = vsubq_f64(x, y);
        t[0] = vfmaq_f64(t[0], vmulq_f64(w, z), z);
    }
    static void step(double* t, double x, double y, double w) {
        const double z = x - y;
        t[0] = std::fma(w * z, z, t[0]);
    }
};

struct AbsOp : SumCombine {  // cityblock
    static constexpr int K = 1;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t) {
        t[0] = vaddq_f64(t[0], vabsq_f64(vsubq_f64(x, y)));
    }
    static void step(double* t, double x, double y, double) { t[0] = t[0] + std::fabs(x - y); }
};

struct WAbsOp : SumCombine {
    static constexpr int K = 1;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t w) {
        t[0] = vfmaq_f64(t[0], w, vabsq_f64(vsubq_f64(x, y)));
    }
    static void step(double* t, double x, double y, double w) { t[0] = std::fma(w, std::fabs(x - y), t[0]); }
};

struct MaxOp {  // chebyshev
    static constexpr int K = 1;
    static float64x2_t combine(float64x2_t s, float64x2_t t) { return vmaxq_f64(s, t); }
    static double combine(double s, double t) { return max1(s, t); }
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t) {
        t[0] = vmaxq_f64(t[0], vabsq_f64(vsubq_f64(x, y)));
    }
    static void step(double* t, double x, double y, double) { t[0] = max1(t[0], std::fabs(x - y)); }
};

struct WMaxOp : MaxOp {  // chebyshev over w_i > 0
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t w) {
        const float64x2_t a = vbslq_f64(vcgtzq_f64(w), vabsq_f64(vsubq_f64(x, y)), f2_zero());
        t[0] = vmaxq_f64(t[0], a);
    }
    static void step(double* t, double x, double y, double w) { t[0] = max1(t[0], w > 0 ? std::fabs(x - y) : 0.0); }
};

inline float64x2_t canberra_term(float64x2_t x, float64x2_t y) {
    const float64x2_t den = vaddq_f64(vabsq_f64(x), vabsq_f64(y));
    const float64x2_t q = vdivq_f64(vabsq_f64(vsubq_f64(x, y)), den);
    return vbslq_f64(vceqzq_f64(den), f2_zero(), q);
}
inline double canberra_term(double x, double y) {
    const double den = std::fabs(x) + std::fabs(y);
    return den == 0 ? 0.0 : std::fabs(x - y) / den;
}

struct CanberraOp : SumCombine {
    static constexpr int K = 1;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t) {
        t[0] = vaddq_f64(t[0], canberra_term(x, y));
    }
    static void step(double* t, double x, double y, double) { t[0] = t[0] + canberra_term(x, y); }
};

struct WCanberraOp : SumCombine {
    static constexpr int K = 1;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t w) {
        t[0] = vfmaq_f64(t[0], w, canberra_term(x, y));
    }
    static void step(double* t, double x, double y, double w) { t[0] = std::fma(w, canberra_term(x, y), t[0]); }
};

struct BrayOp : SumCombine {  // braycurtis: sum |x - y|, sum |x + y|
    static constexpr int K = 2;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t) {
        t[0] = vaddq_f64(t[0], vabsq_f64(vsubq_f64(x, y)));
        t[1] = vaddq_f64(t[1], vabsq_f64(vaddq_f64(x, y)));
    }
    static void step(double* t, double x, double y, double) {
        t[0] = t[0] + std::fabs(x - y);
        t[1] = t[1] + std::fabs(x + y);
    }
};

struct WBrayOp : SumCombine {
    static constexpr int K = 2;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t w) {
        t[0] = vfmaq_f64(t[0], w, vabsq_f64(vsubq_f64(x, y)));
        t[1] = vfmaq_f64(t[1], w, vabsq_f64(vaddq_f64(x, y)));
    }
    static void step(double* t, double x, double y, double w) {
        t[0] = std::fma(w, std::fabs(x - y), t[0]);
        t[1] = std::fma(w, std::fabs(x + y), t[1]);
    }
};

// The counting ops take w = 1 unweighted: a sum of ones below 2^53 is the exact count.
struct HammingOp : SumCombine {  // sum w [x != y] (NaN != NaN)
    static constexpr int K = 1;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t w) {
        t[0] = vaddq_f64(t[0], vbslq_f64(vceqq_f64(x, y), f2_zero(), w));
    }
    template <class T>
    static void step(double* t, T x, T y, double w) {
        t[0] = t[0] + (x != y ? w : 0.0);
    }
};

struct JaccardOp : SumCombine {  // sum w [nz(x) xor nz(y)], sum w [nz(x) or nz(y)]; NaN is nonzero
    static constexpr int K = 2;
    static void step(float64x2_t* t, float64x2_t x, float64x2_t y, float64x2_t w) {
        const uint64x2_t zx = vceqzq_f64(x), zy = vceqzq_f64(y);
        const uint64x2_t wb = vreinterpretq_u64_f64(w);
        t[0] = vaddq_f64(t[0], vreinterpretq_f64_u64(vandq_u64(veorq_u64(zx, zy), wb)));
        t[1] = vaddq_f64(t[1], vreinterpretq_f64_u64(vbicq_u64(wb, vandq_u64(zx, zy))));
    }
    template <class T>
    static void step(double* t, T x, T y, double w) {
        const bool nx = x != T(0), ny = y != T(0);
        t[0] = t[0] + (nx != ny ? w : 0.0);
        t[1] = t[1] + (nx || ny ? w : 0.0);
    }
};

// ---- the two kernels -------------------------------------------------------------------------

// Packs rows j0 .. j0 + 8 of y (rows of n, leading dimension ld) into a panel: panel[k * 8 + c] =
// y[j0 + c][k]; rows past `rows` are zero.
inline void pack_panel(const double* y, int64_t ld, int64_t rows, int64_t n, double* panel) {
    for (int64_t k = 0; k < n; ++k)
        for (int c = 0; c < kPanel; ++c) panel[k * kPanel + c] = c < rows ? y[c * ld + k] : 0.0;
}

// The sums of the R x 8 pairs of rows x[r] and a packed panel, in the tile order: out[(k * R + r) * 8
// + c] is accumulator k of pair (r, c). W: weighted (w[0 .. n)), else w = 1.
template <class Op, int R, bool W>
inline void tile(const double* const* x, const double* panel, const double* w, int64_t n, double* out) {
    constexpr int K = Op::K;
    if (n == 0) {
        for (int i = 0; i < K * R * kPanel; ++i) out[i] = 0.0;
        return;
    }
    for (int64_t b = 0; b < n; b += kBlock) {
        float64x2_t t[R][4][K];
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < 4; ++c)
                for (int k = 0; k < K; ++k) t[r][c][k] = f2_zero();
        const int64_t e = std::min(n, b + kBlock);
        for (int64_t i = b; i < e; ++i) {
            float64x2_t yv[4];
            for (int c = 0; c < 4; ++c) yv[c] = vld1q_f64(panel + i * kPanel + 2 * c);
            const float64x2_t wv = W ? vld1q_dup_f64(w + i) : vdupq_n_f64(1.0);
            for (int r = 0; r < R; ++r) {
                const float64x2_t xv = vld1q_dup_f64(x[r] + i);
                for (int c = 0; c < 4; ++c) Op::step(t[r][c], xv, yv[c], wv);
            }
        }
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < 4; ++c) {
                    double* o = out + (k * R + r) * kPanel + 2 * c;
                    vst1q_f64(o, b == 0 ? t[r][c][k] : Op::combine(vld1q_f64(o), t[r][c][k]));
                }
    }
}

// G full blocks of one pair from term b side by side (G independent chains, each block's own sequential
// chain from 0), their sums combined into s in block order (s = t when `first`).
template <class Op, bool W, int G, class T>
inline void pair_blocks(const T* x, const T* y, const double* w, int64_t b, bool first, double* s) {
    constexpr int K = Op::K;
    double t[G][K];
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < K; ++k) t[g][k] = 0.0;
    for (int64_t i = 0; i < kBlock; ++i)
        for (int g = 0; g < G; ++g) {
            const int64_t q = b + g * kBlock + i;
            Op::step(t[g], x[q], y[q], W ? w[q] : 1.0);
        }
    for (int g = 0; g < G; ++g)
        for (int k = 0; k < K; ++k) s[k] = first && g == 0 ? t[g][k] : Op::combine(s[k], t[g][k]);
}

// The sums of a pair longer than one block (pair_scalar): the blocks are independent chains, so they run
// side by side (up to 8 at a time, SPT6) and are combined in block order: the same operations on the same
// values as one block after another, so the same bits, without waiting on one chain of n fma.
template <class Op, bool W, class T>
[[gnu::noinline]] void pair_scalar_long(const T* x, const T* y, const double* w, int64_t n, double* s) {
    constexpr int K = Op::K;
    for (int k = 0; k < K; ++k) s[k] = 0.0;
    int64_t b = 0;
    for (; b + 8 * kBlock <= n; b += 8 * kBlock) pair_blocks<Op, W, 8>(x, y, w, b, b == 0, s);
    if (b + 4 * kBlock <= n) {
        pair_blocks<Op, W, 4>(x, y, w, b, b == 0, s);
        b += 4 * kBlock;
    }
    if (b + 2 * kBlock <= n) {
        pair_blocks<Op, W, 2>(x, y, w, b, b == 0, s);
        b += 2 * kBlock;
    }
    for (; b < n; b += kBlock) {
        double t[K];
        for (int k = 0; k < K; ++k) t[k] = 0.0;
        const int64_t e = std::min(n, b + kBlock);
        for (int64_t i = b; i < e; ++i) Op::step(t, x[i], y[i], W ? w[i] : 1.0);
        for (int k = 0; k < K; ++k) s[k] = b == 0 ? t[k] : Op::combine(s[k], t[k]);
    }
}

// The same sums for one pair: s[0 .. K). A pair of one block (the tree's points, short rows) is its chain
// alone, inlined where it is called; a longer one is pair_scalar_long.
template <class Op, bool W, class T = double>
[[gnu::always_inline]] inline void pair_scalar(const T* x, const T* y, const double* w, int64_t n, double* s) {
    constexpr int K = Op::K;
    if (n > kBlock) {
        pair_scalar_long<Op, W, T>(x, y, w, n, s);
        return;
    }
    double t[K];
    for (int k = 0; k < K; ++k) t[k] = 0.0;
    for (int64_t i = 0; i < n; ++i) Op::step(t, x[i], y[i], W ? w[i] : 1.0);
    for (int k = 0; k < K; ++k) s[k] = t[k];
}

// ---- careful paths ---------------------------------------------------------------------------

// z_i = x_i - y_i (y null: z_i = x_i), z_i 2^e exactly (no underflow: up to 1074 by two factors)
inline double scaled(double z, int e) {
    if (e > 1000) return (z * 0x1p600) * std::ldexp(1.0, e - 600);
    if (e < -1000) return (z * 0x1p-600) * std::ldexp(1.0, e + 600);
    return z * std::ldexp(1.0, e);
}

inline double zdiff(const double* x, const double* y, int64_t i) { return y ? x[i] - y[i] : x[i] - 0.0; }

// max |z_i| (NaN propagated)
inline double max_abs(const double* x, const double* y, int64_t n) {
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) m = max1(m, std::fabs(zdiff(x, y, i)));
    return m;
}

// sum (W: w_i) (z_i 2^e)^2 in the tile order
template <bool W>
inline double sq_scaled(const double* x, const double* y, const double* w, int64_t n, int e) {
    double s = 0.0;
    for (int64_t b = 0; b < n; b += kBlock) {
        double t = 0.0;
        const int64_t end = std::min(n, b + kBlock);
        for (int64_t i = b; i < end; ++i) {
            const double z = scaled(zdiff(x, y, i), e);
            t = W ? std::fma(w[i] * z, z, t) : std::fma(z, z, t);
        }
        s = b == 0 ? t : s + t;
    }
    return s;
}

// Whether a sum of squares needs the careful path: inf, or below kTiny (not NaN).
inline bool needs_care(double s) { return !(s >= kTiny && s <= kHuge) && s == s; }

// The euclidean distance from its sum of squares s (W: weighted), with the careful path.
template <bool W>
inline double euclid_finish(double s, const double* x, const double* y, const double* w, int64_t n) {
    if (!needs_care(s)) return std::sqrt(s);
    const double m = max_abs(x, y, n);
    if (m == 0.0 || !(m <= kHuge)) return W ? std::sqrt(s) : m;  // all zero; inf (a NaN made s NaN)
    const int k = std::ilogb(m);
    return std::ldexp(std::sqrt(sq_scaled<W>(x, y, w, n, -k)), k);
}

// The squared euclidean distance from s, with the careful path for a tiny sum.
template <bool W>
inline double sq_finish(double s, const double* x, const double* y, const double* w, int64_t n) {
    if (!(s < kTiny)) return s;  // NaN, inf (the true value overflows) or big enough
    const double m = max_abs(x, y, n);
    if (m == 0.0) return s;
    const int k = std::ilogb(m);
    return std::ldexp(sq_scaled<W>(x, y, w, n, -k), 2 * k);
}

// ---- minkowski -------------------------------------------------------------------------------

struct Root {  // s^(1/p) (SPT30)
    double p, inv_p;
    bool exact;  // 1/p is a power of two: pow(s, 1/p) is the root up to pow's own error
    explicit Root(double p_) : p(p_), inv_p(1.0 / p_) {
        int e;
        exact = std::frexp(p_, &e) == 0.5;
    }
    double operator()(double s) const {
        const double r = std::pow(s, inv_p);
        if (exact || !(r > 0.0 && r <= kHuge)) return r;
        const double t = std::pow(r, p);  // s (1 + p eps + pow's error), eps the root's error
        if (!(t <= kHuge)) return r;
        return r - r * (((t - s) / s) / p);
    }
};

// |z|^p: std::pow, or for p = 3 the products (a a) a (SPT80: a relative error of at most (1 + u)^2 - 1 <=
// gamma_2, the contract's c_pow; pow costs about five times the two products).
inline double power(double a, double p, bool cube) { return cube ? (a * a) * a : std::pow(a, p); }

template <bool W>
inline double minkowski_sum(const double* x, const double* y, const double* w, int64_t n, double p, double scale) {
    const bool cube = p == 3.0;
    double s = 0.0;
    for (int64_t b = 0; b < n; b += kBlock) {
        double t = 0.0;
        const int64_t end = std::min(n, b + kBlock);
        for (int64_t i = b; i < end; ++i) {
            const double z = x[i] - y[i];
            const double a = scale == 1.0 ? std::fabs(z) : std::fabs(z) / scale;
            if constexpr (W) t = w[i] > 0 ? std::fma(w[i], power(a, p, cube), t) : (scale == 1.0 ? t + w[i] * z : t);
            else t = t + power(a, p, cube);
        }
        s = b == 0 ? t : s + t;
    }
    return s;
}

template <bool W>
inline double minkowski_pair(const double* x, const double* y, const double* w, int64_t n, const Root& root) {
    const double s = minkowski_sum<W>(x, y, w, n, root.p, 1.0);
    if (!needs_care(s)) return root(s);
    double m = 0.0;  // max |z_i| (over w_i > 0)
    for (int64_t i = 0; i < n; ++i)
        if (!W || w[i] > 0) m = max1(m, std::fabs(x[i] - y[i]));
    if (m == 0.0 || !(m <= kHuge)) return m;
    return m * root(minkowski_sum<W>(x, y, w, n, root.p, m));
}

// ---- mahalanobis -----------------------------------------------------------------------------

inline double dot_tile(const double* a, const double* z, int64_t n) {
    double s = 0.0;
    for (int64_t b = 0; b < n; b += kBlock) {
        double t = 0.0;
        const int64_t end = std::min(n, b + kBlock);
        for (int64_t j = b; j < end; ++j) t = std::fma(a[j], z[j], t);
        s = b == 0 ? t : s + t;
    }
    return s;
}

// z^T VI z of z (n), in the tile order: t_i = dot(VI row i, z), q = sum z_i t_i
inline double quad_form(const double* vi, const double* z, int64_t n) {
    double s = 0.0;
    for (int64_t b = 0; b < n; b += kBlock) {
        double t = 0.0;
        const int64_t end = std::min(n, b + kBlock);
        for (int64_t i = b; i < end; ++i) t = std::fma(z[i], dot_tile(vi + i * n, z, n), t);
        s = b == 0 ? t : s + t;
    }
    return s;
}

// zbuf: n doubles of scratch
inline double mahalanobis_pair(const double* x, const double* y, const double* vi, int64_t n, double* zbuf) {
    for (int64_t i = 0; i < n; ++i) zbuf[i] = x[i] - y[i];
    const double q = quad_form(vi, zbuf, n);
    if (!(std::fabs(q) < kTiny || std::fabs(q) > kHuge) || q != q) return std::sqrt(q);
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) m = max1(m, std::fabs(zbuf[i]));
    if (m == 0.0 || !(m <= kHuge)) return std::sqrt(q);
    const int k = std::ilogb(m);
    for (int64_t i = 0; i < n; ++i) zbuf[i] = scaled(zbuf[i], -k);
    return std::ldexp(std::sqrt(quad_form(vi, zbuf, n)), k);
}

// Up to kLanes pairs (x[l], y[l]), l < cnt, side by side: lane l runs mahalanobis_pair's first pass for
// pair l (t_i = dot(VI row i, z) and q = sum z_i t_i in the tile order, each fma the same operation on
// the same values), so q has the same bits. out[l] = sqrt(q), except for the lanes whose q needs the
// careful path, which are finished by mahalanobis_pair itself. scratch: 2 kLanes n doubles. (SPT6: the
// pairs fill the NEON lanes; one pair alone is a chain of dependent fma.)
inline constexpr int kLanes = 8;
inline void mahalanobis_lanes(
    const double* const* x, const double* const* y, int cnt, const double* vi, int64_t n, double* scratch, double* out
) {
    double* Z = scratch;              // Z[i * 8 + l] = z_l[i]
    double* Tv = scratch + kLanes * n;  // Tv[i * 8 + l] = t_i of pair l
    for (int64_t i = 0; i < n; ++i)
        for (int l = 0; l < kLanes; ++l) Z[i * kLanes + l] = l < cnt ? x[l][i] - y[l][i] : 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double* a = vi + i * n;
        float64x2_t s[4];
        for (int64_t b = 0; b < n; b += kBlock) {
            float64x2_t t[4] = {f2_zero(), f2_zero(), f2_zero(), f2_zero()};
            const int64_t e = std::min(n, b + kBlock);
            for (int64_t j = b; j < e; ++j) {
                const float64x2_t av = vdupq_n_f64(a[j]);
                for (int c = 0; c < 4; ++c) t[c] = vfmaq_f64(t[c], av, vld1q_f64(Z + j * kLanes + 2 * c));
            }
            for (int c = 0; c < 4; ++c) s[c] = b == 0 ? t[c] : vaddq_f64(s[c], t[c]);
        }
        for (int c = 0; c < 4; ++c) vst1q_f64(Tv + i * kLanes + 2 * c, n ? s[c] : f2_zero());
    }
    float64x2_t q[4] = {f2_zero(), f2_zero(), f2_zero(), f2_zero()};
    for (int64_t b = 0; b < n; b += kBlock) {
        float64x2_t t[4] = {f2_zero(), f2_zero(), f2_zero(), f2_zero()};
        const int64_t e = std::min(n, b + kBlock);
        for (int64_t i = b; i < e; ++i)
            for (int c = 0; c < 4; ++c)
                t[c] = vfmaq_f64(t[c], vld1q_f64(Z + i * kLanes + 2 * c), vld1q_f64(Tv + i * kLanes + 2 * c));
        for (int c = 0; c < 4; ++c) q[c] = b == 0 ? t[c] : vaddq_f64(q[c], t[c]);
    }
    double qs[kLanes];
    for (int c = 0; c < 4; ++c) vst1q_f64(qs + 2 * c, q[c]);
    for (int l = 0; l < cnt; ++l) {
        const double v = qs[l];
        if (!(std::fabs(v) < kTiny || std::fabs(v) > kHuge) || v != v) out[l] = std::sqrt(v);
        else out[l] = mahalanobis_pair(x[l], y[l], vi, n, scratch);  // the careful path (scratch is free now)
    }
}

// ---- haversine -------------------------------------------------------------------------------

// (phi1, lam1), (phi2, lam2) in radians; c1 = cos phi1, c2 = cos phi2 (precomputed per row). SPT81: d = 2
// asin(sqrt(h)) while h <= 1/2 (d <= pi/2, where asin's condition number is at most 1.3), else d = pi - 2
// asin(sqrt(g)) with g the cancellation-free 1 - h above (g <= 1/2 there too, the antipode's digits kept):
// one asin for SPT31's atan2, and g with its sine only for the far half.
// SPT88: the longitude difference is dl + e exactly (TwoSum: dl = fl(lam2 - lam1), e its rounding error), and
// sin((dl + e)/2), cos((dl + e)/2) are taken to first order in e/2 (one fma each; the second-order terms are
// below e^2/8). Without it a pair across the dateline (dl near 2 pi, d small) carried the absolute error
// u |dl| / 2 of dl into sin(dl/2): a relative error u pi / d in d (1e-10 for d = 3e-6). A sum h below kTiny
// is recomputed from the sines scaled by 2^600 (exact), so the squares of a tiny distance do not underflow.
inline double haversine_pair(double f1, double l1, double c1, double f2, double l2, double c2) {
    const SinCos a = sincos((f2 - f1) * 0.5);
    const double dl = l2 - l1, lv = dl + l1, bv = dl - lv;
    const double he = ((l2 - lv) + (-l1 - bv)) * 0.5;  // e / 2
    const SinCos b = sincos(dl * 0.5);
    const double bs = std::fma(he, b.c, b.s);
    const double cc = c1 * c2;
    const double h = a.s * a.s + cc * (bs * bs);
    if (h <= 0.5) {
        if (h >= kTiny) return 2.0 * std::asin(std::sqrt(h));
        const double x = a.s * 0x1p600, y = bs * 0x1p600;  // the careful path (h = 0 stays 0)
        return 2.0 * std::asin(std::sqrt(x * x + cc * (y * y)) * 0x1p-600);
    }
    const double bc = std::fma(-he, b.s, b.c);
    const double ss = std::sin((f1 + f2) * 0.5);
    const double ac = a.c * bc, sd_ = ss * bs;
    const double g = ac * ac + sd_ * sd_;
    return 0x1.921fb54442d18p+1 - 2.0 * std::asin(std::sqrt(g));
}

// ---- rows prepared once ----------------------------------------------------------------------

// The norm ||x||_w of a row by the euclidean kernel (x - 0), careful path included.
template <bool W>
inline double row_norm(const double* x, const double* w, int64_t n) {
    double s = 0.0;
    for (int64_t b = 0; b < n; b += kBlock) {
        double t = 0.0;
        const int64_t end = std::min(n, b + kBlock);
        for (int64_t i = b; i < end; ++i) {
            const double z = x[i] - 0.0;
            t = W ? std::fma(w[i] * z, z, t) : std::fma(z, z, t);
        }
        s = b == 0 ? t : s + t;
    }
    return euclid_finish<W>(s, x, nullptr, w, n);
}

// out = x / ||x||_w (cosine), after centering by the (weighted) mean when `center` (correlation).
// sumw: the sum of the weights (W) in the tile order. The row is first scaled by the power of two of its
// largest entry (exact, and neither distance depends on the scale), so the mean and the norm of a tiny or
// huge row keep their digits (the mean of subnormal entries, a subnormal or overflowing norm would not);
// the centered row is scaled again before its norm. For a row in the normal range the scalings change no
// bit of the result.
template <bool W>
inline void prepare_row(const double* x, const double* w, int64_t n, bool center, double sumw, double* out) {
    auto rescale = [&](const double* src) {
        const double m = max_abs(src, nullptr, n);
        const int e = (m > 0.0 && m <= kHuge) ? -std::ilogb(m) : 0;
        for (int64_t i = 0; i < n; ++i) out[i] = e ? scaled(src[i], e) : src[i];
    };
    rescale(x);
    if (center) {
        double mu = std::numeric_limits<double>::quiet_NaN();
        int64_t first = -1;  // the first entry that counts (w_i > 0)
        for (int64_t i = 0; i < n && first < 0; ++i)
            if (!W || w[i] > 0) first = i;
        bool constant = first >= 0;
        for (int64_t i = first + 1; constant && i < n; ++i)
            if ((!W || w[i] > 0) && !(out[i] == out[first])) constant = false;
        if (constant) {
            mu = out[first];
        } else if (n > 0) {
            double s = 0.0;
            for (int64_t b = 0; b < n; b += kBlock) {
                double t = 0.0;
                const int64_t end = std::min(n, b + kBlock);
                for (int64_t i = b; i < end; ++i) t = W ? std::fma(w[i], out[i], t) : t + out[i];
                s = b == 0 ? t : s + t;
            }
            mu = s / (W ? sumw : static_cast<double>(n));
        }
        for (int64_t i = 0; i < n; ++i) out[i] = out[i] - mu;
        rescale(out);
    }
    const double nrm = row_norm<W>(out, w, n);
    for (int64_t i = 0; i < n; ++i) out[i] = out[i] / nrm;
}

// The sum of n weights in the tile order (hamming's denominator, correlation's mean).
inline double weight_sum(const double* w, int64_t n) {
    double s = 0.0;
    for (int64_t b = 0; b < n; b += kBlock) {
        double t = 0.0;
        const int64_t end = std::min(n, b + kBlock);
        for (int64_t i = b; i < end; ++i) t = t + w[i];
        s = b == 0 ? t : s + t;
    }
    return s;
}

// A bool row packed into 64-bit words (bit j of word k is x[64 k + j]).
inline void pack_bits(const std::uint8_t* x, int64_t n, std::uint64_t* out) {
    const int64_t words = (n + 63) / 64;
    for (int64_t k = 0; k < words; ++k) {
        std::uint64_t v = 0;
        const int64_t e = std::min<int64_t>(64, n - 64 * k);
        for (int64_t j = 0; j < e; ++j) v |= static_cast<std::uint64_t>(x[64 * k + j] != 0) << j;
        out[k] = v;
    }
}

}  // namespace sd::spatial
