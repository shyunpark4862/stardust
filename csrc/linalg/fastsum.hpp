// Fast sums for the dot products and norms of float64 and complex128 data read in place
// (vecdot, vector_norm, matrix_norm; docs/design/linalg.md §8.2).
//
// A long sum is cut into chunks of kChunk terms -- a partition fixed by the length only,
// so the result is the same for any thread count -- and the chunks' partial sums are
// added along a balanced tree over the chunks. Long sums run their chunks on the pool.
// Within a chunk:
//   - dot products and sums of squares: the pairwise tree of 16-lane fused leaves when short
//     (the bits of before), Accelerate's ddot / zdotc when long (the AMX unit: 3-9x the
//     NEON loop from ~1000 terms);
//   - sums of magnitudes: the pairwise sum's 16-lane leaves in place when short, Accelerate's
//     dasum when long;
//   - largest / smallest magnitude: a 16-lane NEON pass (exact, so the order is free).
// The rounding error of a sum is that of a multi-accumulator sum within a chunk and of a
// pairwise sum across chunks. Accelerate's ddot / dasum add into about 256 accumulators
// (measured: 1 + n terms of 1e-16 loses n / 256 of them), so a lane of a 16384-term chunk
// adds 64 terms in order, against 32 in a 512-term leaf of the 16-lane pairwise sum this
// replaced: the same order of error, and on random data no measurable difference. Longer
// chunks were up to 15% faster in cache but let the error grow with the chunk (65536: 255
// terms lost of the 65535 above).
#pragma once

#include <arm_neon.h>

#include "linalg/batch.hpp"
#include "linalg/pairwise.hpp"
#include "linalg/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::linalg::fast {

inline constexpr std::int64_t kChunk = 1 << 14;        // terms per chunk (the fixed partition; see above)
inline constexpr std::int64_t kParallelMin = 1 << 17;  // shorter sums run on the calling thread
inline constexpr std::int64_t kReleaseMin = 1 << 15;   // from here a serial sum releases the GIL
inline constexpr std::int64_t kBlasDotMin = 1024;      // ddot / dasum above this many terms
inline constexpr std::int64_t kBlasSqMin = 2048;       // ddot(x, x) above this: the one-load NEON leaf wins below
inline constexpr std::int64_t kBlasCdotMin = 256;      // zdotc above this many complex terms (one leaf)

namespace detail {

// part(lo) for chunk lo, combined along the balanced tree over chunks [lo, hi).
template <class T, class Get, class Combine>
T tree(std::int64_t lo, std::int64_t hi, const Get& get, const Combine& combine) {
    if (hi - lo == 1) return get(lo);
    const std::int64_t mid = lo + (hi - lo) / 2;
    return combine(tree<T>(lo, mid, get, combine), tree<T>(mid, hi, get, combine));
}

}  // namespace detail

// The reduction of n terms: part(begin, len) reduces one chunk (len <= kChunk), combine
// joins two partial results. The chunks and the tree depend on n only: the same bits on
// any thread count. `part` is called from several threads at once on long inputs.
template <class T, class Part, class Combine>
T chunked(std::int64_t n, const Part& part, const Combine& combine) {
    if (n <= kChunk) return part(std::int64_t(0), n);
    const std::int64_t nc = (n + kChunk - 1) / kChunk;
    auto chunk = [&](std::int64_t c) { return part(c * kChunk, std::min(kChunk, n - c * kChunk)); };
    if (n >= kParallelMin && linalg::detail::begin_parallel()) {
        std::vector<T> parts(static_cast<std::size_t>(nc));
        linalg::detail::run_tasks(nc, [&](std::int64_t c) { parts[static_cast<std::size_t>(c)] = chunk(c); });
        return detail::tree<T>(0, nc, [&](std::int64_t c) { return parts[static_cast<std::size_t>(c)]; }, combine);
    }
    const parallel::Unlocked unlocked(n >= kReleaseMin);
    return detail::tree<T>(0, nc, chunk, combine);
}

template <class T>
T chunked_sum(std::int64_t n, const auto& part) {
    return chunked<T>(n, part, [](T a, T b) { return a + b; });
}

// ---- kernels (one chunk) ------------------------------------------------------------

// sum x[i]^2, contiguous: one load per term, 16 lanes, fused.
inline double sq_leaf(const double* x, std::int64_t n) {
    std::int64_t i = 0;
    double r = 0.;
    if (n >= 16) {
        float64x2_t acc[8];
        for (int j = 0; j < 8; ++j) {
            const float64x2_t v = vld1q_f64(x + 2 * j);
            acc[j] = vmulq_f64(v, v);
        }
        for (i = 16; i + 16 <= n; i += 16)
            for (int j = 0; j < 8; ++j) {
                const float64x2_t v = vld1q_f64(x + i + 2 * j);
                acc[j] = vfmaq_f64(acc[j], v, v);
            }
        const float64x2_t s = vaddq_f64(
            vaddq_f64(vaddq_f64(acc[0], acc[1]), vaddq_f64(acc[2], acc[3])),
            vaddq_f64(vaddq_f64(acc[4], acc[5]), vaddq_f64(acc[6], acc[7]))
        );
        r = vgetq_lane_f64(s, 0) + vgetq_lane_f64(s, 1);
    }
    for (; i < n; ++i) r = std::fma(x[i], x[i], r);
    return r;
}

// NaN-propagating max / min of two doubles (as FMAX / FMIN).
template <bool Max>
inline double pick_scalar(double a, double b) {
    if (a != a || b != b) return a + b;
    return Max ? (b > a ? b : a) : (b < a ? b : a);
}

// The largest (max) or smallest |x[i]|, contiguous; NaN if any x[i] is NaN (FMAX / FMIN
// propagate it). n >= 1. Exact, so lanes and chunks combine in any order.
template <bool Max>
inline double extreme_abs(const double* x, std::int64_t n) {
    auto pick = [](float64x2_t a, float64x2_t b) { return Max ? vmaxq_f64(a, b) : vminq_f64(a, b); };
    std::int64_t i = 0;
    double r = std::fabs(x[0]);
    if (n >= 16) {
        float64x2_t m[8];
        for (int j = 0; j < 8; ++j) m[j] = vabsq_f64(vld1q_f64(x + 2 * j));
        for (i = 16; i + 16 <= n; i += 16)
            for (int j = 0; j < 8; ++j) m[j] = pick(m[j], vabsq_f64(vld1q_f64(x + i + 2 * j)));
        const float64x2_t s = pick(pick(pick(m[0], m[1]), pick(m[2], m[3])), pick(pick(m[4], m[5]), pick(m[6], m[7])));
        r = Max ? vmaxvq_f64(s) : vminvq_f64(s);
    }
    for (; i < n; ++i) r = pick_scalar<Max>(r, std::fabs(x[i]));
    return r;
}

// sum x[i] y[i] over n terms with element strides incx, incy (> 0).
// Below the BLAS sizes (n <= kBlasDotMin, kBlasSqMin, kBlasCdotMin): the pairwise tree of
// 16-lane fused leaves (base/pairwise.hpp, linalg/pairwise.hpp) that every length used
// before, so short sums keep their bits; strided terms are gathered into a leaf buffer first.
inline double dot_chunk(std::int64_t n, const double* x, std::int64_t incx, const double* y, std::int64_t incy) {
    using linalg::detail::fma_dot_leaf;
    if (incx == 1 && incy == 1) {
        if (x == y) {
            if (n <= kPairwiseBlock) return sq_leaf(x, n);
            if (n <= kBlasSqMin)
                return pairwise::tree(0, n, [x](std::int64_t b, std::int64_t m) { return sq_leaf(x + b, m); });
            return cblas_ddot(n, x, 1, x, 1);
        }
        if (n <= kPairwiseBlock) return fma_dot_leaf(x, y, n);
        if (n <= kBlasDotMin)
            return pairwise::tree(0, n, [x, y](std::int64_t b, std::int64_t m) {
                return fma_dot_leaf(x + b, y + b, m);
            });
    } else if (n <= kBlasDotMin) {
        return pairwise::tree(0, n, [&](std::int64_t b, std::int64_t m) {
            alignas(16) double a[kPairwiseBlock], c[kPairwiseBlock];
            for (std::int64_t i = 0; i < m; ++i) {
                a[i] = x[(b + i) * incx];
                c[i] = y[(b + i) * incy];
            }
            return fma_dot_leaf(a, c, m);
        });
    }
    return cblas_ddot(n, x, incx, y, incy);
}

// sum conj(x[i]) y[i] over n complex terms with element strides incx, incy (> 0).
inline cplx cdot_chunk(std::int64_t n, const cplx* x, std::int64_t incx, const cplx* y, std::int64_t incy) {
    if (n <= kBlasCdotMin) {
        if (incx == 1 && incy == 1) return linalg::detail::fma_cdot_leaf(x, y, n);
        alignas(16) double ra[kPairwiseBlock], rc[kPairwiseBlock];  // not cplx[]: its constructor would zero it
        cplx* a = reinterpret_cast<cplx*>(ra);
        cplx* c = reinterpret_cast<cplx*>(rc);
        for (std::int64_t i = 0; i < n; ++i) {
            a[i] = x[i * incx];
            c[i] = y[i * incy];
        }
        return linalg::detail::fma_cdot_leaf(a, c, n);
    }
    cplx r;
    cblas_zdotc_sub(n, x, incx, y, incy, &r);
    return r;
}

// sum |x[i]|; up to kBlasDotMin terms the pairwise sum of before (sd.sum's bits for the same terms).
inline double abs_chunk(std::int64_t n, const double* x, std::int64_t incx) {
    if (n <= kBlasDotMin) {
        auto leaf = [x, incx](std::int64_t b, std::int64_t m) {
            const double* p = x + b * incx;
            if (incx == 1) return pairwise::leaf(m, [p](std::int64_t i) { return std::fabs(p[i]); });
            return pairwise::leaf(m, [p, incx](std::int64_t i) { return std::fabs(p[i * incx]); });
        };
        return n <= kPairwiseBlock ? leaf(0, n) : pairwise::tree(0, n, leaf);
    }
    return cblas_dasum(n, x, incx);
}

// ---- whole sums -----------------------------------------------------------------------

// Each whole sum takes a single chunk inline (the common short case: no closure, no
// call), and the chunked driver otherwise.
inline double dot(std::int64_t n, const double* x, std::int64_t incx, const double* y, std::int64_t incy) {
    if (n <= kChunk) return 0. + dot_chunk(n, x, incx, y, incy);
    return 0. + chunked_sum<double>(n, [&](std::int64_t b, std::int64_t len) {
               return dot_chunk(len, x + b * incx, incx, y + b * incy, incy);
           });
}

inline cplx cdot(std::int64_t n, const cplx* x, std::int64_t incx, const cplx* y, std::int64_t incy) {
    const cplx s =
        n <= kChunk ? cdot_chunk(n, x, incx, y, incy) : chunked_sum<cplx>(n, [&](std::int64_t b, std::int64_t len) {
            return cdot_chunk(len, x + b * incx, incx, y + b * incy, incy);
        });
    return {0. + s.real(), 0. + s.imag()};
}

inline double sum_squares(std::int64_t n, const double* x, std::int64_t incx) { return dot(n, x, incx, x, incx); }

inline double sum_abs(std::int64_t n, const double* x, std::int64_t incx) {
    if (n <= kChunk) return 0. + abs_chunk(n, x, incx);
    return 0. +
           chunked_sum<double>(n, [&](std::int64_t b, std::int64_t len) { return abs_chunk(len, x + b * incx, incx); });
}

// max (Max) or min |x[i]| of n >= 1 contiguous doubles, NaN propagating.
template <bool Max>
inline double extreme(std::int64_t n, const double* x) {
    if (n <= kChunk) return extreme_abs<Max>(x, n);
    return chunked<double>(
        n, [&](std::int64_t b, std::int64_t len) { return extreme_abs<Max>(x + b, len); },
        [](double a, double b) { return pick_scalar<Max>(a, b); }
    );
}

// ---- line sums (matrix_norm and cond, ord +-1 and +-inf) ---------------------------------

// out[j] = sum_i |p[i * es + j]| for j < w (w <= kStrip, len >= 1): lines side by side in
// memory (the columns of a C-order matrix for ord=1, the rows of an F-order one for inf),
// summed a row of the strip at a time, so the adds vectorize across the lines. A leaf of
// at most kStripLeaf rows keeps 16 lines in 8 registers and adds its rows in order (one
// load per term, no store per row: 1.4x the plain row loop); above, pairwise over the rows.
inline constexpr std::int64_t kStrip = 128;
inline constexpr std::int64_t kStripLeaf = 32;

inline void strip_leaf(const double* p, std::int64_t es, std::int64_t len, std::int64_t w, double* out) {
    std::int64_t j = 0;
    for (; j + 16 <= w; j += 16) {
        float64x2_t acc[8];
        for (int k = 0; k < 8; ++k) acc[k] = vabsq_f64(vld1q_f64(p + j + 2 * k));
        for (std::int64_t i = 1; i < len; ++i) {
            const double* row = p + i * es + j;
            for (int k = 0; k < 8; ++k) acc[k] = vaddq_f64(acc[k], vabsq_f64(vld1q_f64(row + 2 * k)));
        }
        for (int k = 0; k < 8; ++k) vst1q_f64(out + j + 2 * k, acc[k]);
    }
    for (; j + 2 <= w; j += 2) {
        float64x2_t acc = vabsq_f64(vld1q_f64(p + j));
        for (std::int64_t i = 1; i < len; ++i) acc = vaddq_f64(acc, vabsq_f64(vld1q_f64(p + i * es + j)));
        vst1q_f64(out + j, acc);
    }
    for (; j < w; ++j) {
        double acc = std::fabs(p[j]);
        for (std::int64_t i = 1; i < len; ++i) acc += std::fabs(p[i * es + j]);
        out[j] = acc;
    }
}

inline void strip_abs_sums(const double* p, std::int64_t es, std::int64_t len, int w, double* out) {
    if (len <= kStripLeaf) {
        strip_leaf(p, es, len, w, out);
        return;
    }
    const std::int64_t h = len / 2;
    strip_abs_sums(p, es, h, w, out);
    double tmp[kStrip];
    strip_abs_sums(p + h * es, es, len - h, w, tmp);
    for (int j = 0; j < w; ++j) out[j] += tmp[j];
}

// The largest (max) or smallest sum of |x| along `lines` >= 1 lines of `len` float64
// values: element i of line l at p[l * ls + i * es] (element strides), with either es == 1
// (each line a fast sum) or ls == 1 and es > 0 (the lines in strips). NaN if a sum is NaN.
// One large matrix runs its lines or strips on the pool; each sum is computed as in the
// serial loop and max/min do not depend on the order, so the bits do not depend on it.
inline double line_abs_extreme(
    const double* p, std::int64_t lines, std::int64_t len, std::int64_t ls, std::int64_t es, bool max
) {
    double best = max ? 0.0 : std::numeric_limits<double>::infinity();
    bool nan = false;
    auto take = [&](double sum) {
        if (sum != sum) nan = true;
        else best = max ? (sum > best ? sum : best) : (sum < best ? sum : best);
    };
    const bool runs = es == 1;
    auto strip = [&](std::int64_t l0, std::int64_t w, double* out) {
        if (len == 0) std::fill(out, out + w, 0.0);
        else strip_abs_sums(p + l0, es, len, static_cast<int>(w), out);
    };
    const bool large = lines * len >= kReleaseMin;
    if (large && lines > 1 && linalg::detail::begin_parallel()) {
        std::vector<double> sums(static_cast<std::size_t>(lines));
        if (runs) {
            const std::int64_t per = std::max<std::int64_t>(1, 4096 / std::max<std::int64_t>(len, 1));
            linalg::detail::run_tasks((lines + per - 1) / per, [&](std::int64_t t) {
                for (std::int64_t l = t * per; l < std::min(lines, (t + 1) * per); ++l)
                    sums[static_cast<std::size_t>(l)] = sum_abs(len, p + l * ls, 1);
            });
        } else {
            // narrower strips than the serial loop's, so that there are tasks for every thread
            constexpr std::int64_t w = 32;
            linalg::detail::run_tasks((lines + w - 1) / w, [&](std::int64_t t) {
                strip(t * w, std::min(w, lines - t * w), sums.data() + t * w);
            });
        }
        for (double v : sums) take(v);
    } else {
        const parallel::Unlocked unlocked(large);
        if (runs) {
            for (std::int64_t l = 0; l < lines; ++l) take(sum_abs(len, p + l * ls, 1));
        } else {
            double out[kStrip];
            for (std::int64_t l0 = 0; l0 < lines; l0 += kStrip) {
                const std::int64_t w = std::min(kStrip, lines - l0);
                strip(l0, w, out);
                for (std::int64_t j = 0; j < w; ++j) take(out[j]);
            }
        }
    }
    return nan ? std::numeric_limits<double>::quiet_NaN() : best;
}

// The 2-norm from the plain sum of squares s of `count` terms, or a negative number when
// s cannot be trusted: inf (an overflow, or inf among the terms), NaN, or so small that
// subnormal squares could matter (below count * 2^-1000, where the squares' absolute
// errors, at most 2^-1075 each, stay under 2^-75 of s). The caller then takes the
// careful path (vnorm.hpp: NaN, inf, or the pass rescaled by the largest magnitude).
inline double norm_from_squares(double s, std::int64_t count) {
    if (!(s <= std::numeric_limits<double>::max()) || s < static_cast<double>(count) * 0x1p-1000) return -1.0;
    return std::sqrt(s);
}

}  // namespace sd::linalg::fast
