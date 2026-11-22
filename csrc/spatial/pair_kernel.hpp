// Prototype pair kernels for the squared Euclidean distance (docs/design/spatial.md §3.1, decision
// SPT22): the two candidate summation orders, each as a one-pair kernel and as the batched kernel
// a cdist-like loop would run. Milestone SPT0 timed them (tools/spatial_probe.py, section
// kernels); SPT1 builds metric.hpp from the order the measurement chose. Nothing here is bound to
// Python yet.
//
// A distance is d = sqrt(s), s = sum z_i^2 with z_i = fl(x_i - y_i) formed in a register. The two
// orders differ only in how the n terms of s are added:
//
//   NormOrder   sd.linalg.vector_norm(x - y)'s order (linalg/fastsum.hpp: dot_chunk(x, x)):
//               n < 16: fma(z_i, z_i, r) one after another from r = 0;
//               16 <= n <= 512: sq_leaf -- 16 lanes (8 NEON registers), the first 16 squares
//               rounded (vmulq), later full groups of 16 fused (vfmaq), the lanes folded
//               ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)) then lane 0 + lane 1, a tail of n % 16 terms
//               fused one by one; 512 < n <= 2048: base/pairwise.hpp's tree over such leaves;
//               above: Accelerate's cblas_ddot(z, z) on a buffer (chunks of 16384 on a balanced
//               tree). Then 0.0 + s. A pair occupies 8 accumulators, so the batched kernel holds
//               one x row against two y rows (16 accumulators); below 16 terms it is TileOrder's
//               kernel (the same sequential fma chain per pair).
//   TileOrder   each pair's own order, fit for a register tile of 4 x rows by 8 y rows (16
//               accumulators, two pairs per register): the terms in blocks of kTileBlock, each
//               block a sequential fma chain from 0, the block sums added one after another.
//               The batched kernel and the one-pair kernel give the same bits; the error bound
//               is that of a sum of depth kTileBlock + ceil(n / kTileBlock).
//
// The prototype's sq_leaf is a copy of linalg::fast::sq_leaf with z formed on the fly (the
// probe checks its distances against sd.linalg.vector_norm bit for bit); SPT1 shares the one
// definition instead. The careful path of a norm that overflows or underflows (vnorm.hpp) is not
// part of the prototype: the timing data never reach it.
#pragma once

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include "base/pairwise.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::spatial::proto {

inline constexpr std::int64_t kLeaf = pairwise::kBlock;  // 512: the leaf of the 16-lane sums
inline constexpr std::int64_t kBlasSqMin = 2048;         // linalg::fast::kBlasSqMin
inline constexpr std::int64_t kChunk = 1 << 14;          // linalg::fast::kChunk
inline constexpr std::int64_t kTileBlock = 32;           // TileOrder's block of terms

// ---- NormOrder ----------------------------------------------------------------------------

// sq_leaf of z = x - y over n <= kLeaf terms, for P = 1 or 2 pairs sharing x (y[p] the rows).
template <int P>
inline void norm_leaf(const double* x, const double* const* y, std::int64_t n, double* out) {
    std::int64_t i = 0;
    double r[P];
    for (int p = 0; p < P; ++p) r[p] = 0.;
    if (n >= 16) {
        float64x2_t acc[P][8];
        for (int j = 0; j < 8; ++j) {
            const float64x2_t xv = vld1q_f64(x + 2 * j);
            for (int p = 0; p < P; ++p) {
                const float64x2_t z = vsubq_f64(xv, vld1q_f64(y[p] + 2 * j));
                acc[p][j] = vmulq_f64(z, z);
            }
        }
        for (i = 16; i + 16 <= n; i += 16)
            for (int j = 0; j < 8; ++j) {
                const float64x2_t xv = vld1q_f64(x + i + 2 * j);
                for (int p = 0; p < P; ++p) {
                    const float64x2_t z = vsubq_f64(xv, vld1q_f64(y[p] + i + 2 * j));
                    acc[p][j] = vfmaq_f64(acc[p][j], z, z);
                }
            }
        for (int p = 0; p < P; ++p) {
            const float64x2_t s = vaddq_f64(
                vaddq_f64(vaddq_f64(acc[p][0], acc[p][1]), vaddq_f64(acc[p][2], acc[p][3])),
                vaddq_f64(vaddq_f64(acc[p][4], acc[p][5]), vaddq_f64(acc[p][6], acc[p][7]))
            );
            r[p] = vgetq_lane_f64(s, 0) + vgetq_lane_f64(s, 1);
        }
    }
    for (; i < n; ++i)
        for (int p = 0; p < P; ++p) {
            const double z = x[i] - y[p][i];
            r[p] = std::fma(z, z, r[p]);
        }
    for (int p = 0; p < P; ++p) out[p] = r[p];
}

// base/pairwise.hpp's tree over norm_leaf (begin .. begin + n), per pair.
template <int P>
inline void norm_tree(const double* x, const double* const* y, std::int64_t begin, std::int64_t n, double* out) {
    if (n <= kLeaf) {
        const double* yb[P];
        for (int p = 0; p < P; ++p) yb[p] = y[p] + begin;
        norm_leaf<P>(x + begin, yb, n, out);
        return;
    }
    const std::int64_t n2 = pairwise::split(n);
    double a[P], b[P];
    norm_tree<P>(x, y, begin, n2, a);
    norm_tree<P>(x, y, begin + n2, n - n2, b);
    for (int p = 0; p < P; ++p) out[p] = a[p] + b[p];
}

// The balanced tree over chunks of a BLAS-length sum (linalg::fast::chunked, serial).
inline double norm_blas(const double* z, std::int64_t n) {
    auto chunk = [&](std::int64_t c) {
        const std::int64_t b = c * kChunk, len = std::min(kChunk, n - b);
        return cblas_ddot(len, z + b, 1, z + b, 1);
    };
    if (n <= kChunk) return chunk(0);
    auto tree = [&](auto&& self, std::int64_t lo, std::int64_t hi) -> double {
        if (hi - lo == 1) return chunk(lo);
        const std::int64_t mid = lo + (hi - lo) / 2;
        return self(self, lo, mid) + self(self, mid, hi);
    };
    return tree(tree, 0, (n + kChunk - 1) / kChunk);
}

// s of P pairs sharing x in NormOrder (0.0 + the sum). zbuf: n doubles, used above kBlasSqMin.
template <int P>
inline void norm_order(const double* x, const double* const* y, std::int64_t n, double* out, double* zbuf) {
    if (n <= kBlasSqMin) {
        norm_tree<P>(x, y, 0, n, out);
    } else {
        for (int p = 0; p < P; ++p) {
            for (std::int64_t i = 0; i < n; ++i) zbuf[i] = x[i] - y[p][i];
            out[p] = norm_blas(zbuf, n);
        }
    }
    for (int p = 0; p < P; ++p) out[p] = 0. + out[p];
}

// ---- TileOrder ----------------------------------------------------------------------------

// s of one pair in TileOrder.
inline double tile_order(const double* x, const double* y, std::int64_t n) {
    double s = 0.;
    for (std::int64_t b = 0; b < n; b += kTileBlock) {
        double t = 0.;
        const std::int64_t e = std::min(n, b + kTileBlock);
        for (std::int64_t k = b; k < e; ++k) {
            const double z = x[k] - y[k];
            t = std::fma(z, z, t);
        }
        s = b == 0 ? t : s + t;
    }
    return s;
}

// Packs rows j0 .. j0 + 8 of y (row length n, leading dimension ldy) into a panel: panel[k * 8 + c]
// is y[j0 + c][k]. Missing rows (past `rows`) are zero.
inline void pack_panel(const double* y, std::int64_t ldy, std::int64_t rows, std::int64_t n, double* panel) {
    for (std::int64_t k = 0; k < n; ++k)
        for (int c = 0; c < 8; ++c) panel[k * 8 + c] = c < rows ? y[c * ldy + k] : 0.;
}

// s of the 4 x 8 pairs of rows x[r] (r < 4) and a packed panel, in TileOrder: out[r * 8 + c].
// Below 16 terms this is also NormOrder (one block: the sequential chain from 0).
inline void tile_4x8(const double* const* x, const double* panel, std::int64_t n, double* out) {
    for (std::int64_t b = 0; b < n; b += kTileBlock) {
        float64x2_t t[4][4];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) t[r][c] = vdupq_n_f64(0.);
        const std::int64_t e = std::min(n, b + kTileBlock);
        for (std::int64_t k = b; k < e; ++k) {
            float64x2_t yv[4];
            for (int c = 0; c < 4; ++c) yv[c] = vld1q_f64(panel + k * 8 + 2 * c);
            for (int r = 0; r < 4; ++r) {
                const float64x2_t xv = vld1q_dup_f64(x[r] + k);
                for (int c = 0; c < 4; ++c) {
                    const float64x2_t z = vsubq_f64(xv, yv[c]);
                    t[r][c] = vfmaq_f64(t[r][c], z, z);
                }
            }
        }
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
                double* o = out + r * 8 + 2 * c;
                vst1q_f64(o, b == 0 ? t[r][c] : vaddq_f64(vld1q_f64(o), t[r][c]));
            }
    }
}

}  // namespace sd::spatial::proto
