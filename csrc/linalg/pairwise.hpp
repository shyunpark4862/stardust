// Pairwise summation for every floating-point sum in linalg: numpy's
// DOUBLE_pairwise_sum and CDOUBLE_pairwise_sum algorithm with base/pairwise.hpp's
// wider leaf (the tree, the leaf sums, the parallel plan, defined once there), so
// linalg's sums have the array area's bits, not numpy's (docs/design/linalg.md §8.2).
// A sum is 0.0 + tree, as numpy's add.reduce does.
//
// What is linalg's own is how terms reach a leaf: fill(begin, len, buf)
// writes terms begin .. begin+len-1 (at most one leaf) to buf, in order. A
// leaf is filled first and then summed, so the summing loop is the same for
// every source layout and vectorizes.
//
// Sums of products (vecdot, the squares of the 2-norm) have leaves of their own:
// the products fused into the sum (vfmaq, one rounding per term) on 16 lanes, read
// straight from contiguous float64 input or from two filled buffers otherwise. The
// leaf is a function of its terms' values only, so every layout gives the same bits,
// and the tree is the same as every other sum's (docs/design/linalg.md §8.2).
#pragma once

#include <arm_neon.h>

#include "base/pairwise.hpp"
#include "linalg/batch.hpp"
#include "linalg/types.hpp"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace sd::linalg {

inline constexpr std::int64_t kPairwiseBlock = pairwise::kBlock;  // 512 doubles (numpy's PW_BLOCKSIZE is 128)

namespace detail {

inline double pairwise_leaf(const double* a, std::int64_t n) {
    return pairwise::leaf(n, [a](std::int64_t i) { return a[i]; });
}

inline cplx pairwise_leaf_complex(const cplx* a, std::int64_t n) {
    return pairwise::leaf_complex(n, [a](std::int64_t i) { return a[i]; });
}

template <class Fill>
double pairwise(std::int64_t begin, std::int64_t n, Fill& fill) {
    return pairwise::tree(begin, n, [&fill](std::int64_t b, std::int64_t m) {
        double buf[kPairwiseBlock];
        fill(b, m, buf);
        return pairwise_leaf(buf, m);
    });
}

template <class Fill>
cplx pairwise_complex(std::int64_t begin, std::int64_t n, Fill& fill) {
    return pairwise::tree_complex(begin, n, [&fill](std::int64_t b, std::int64_t m) {
        alignas(16) double raw[kPairwiseBlock];  // not cplx[]: its constructor would zero it
        cplx* buf = reinterpret_cast<cplx*>(raw);
        fill(b, m, buf);
        return pairwise_leaf_complex(buf, m);
    });
}

// sum a[i] * b[i] over i < n (n <= kPairwiseBlock): 16 lanes (8 NEON registers), each
// product fused into its lane (vfmaq), the lanes added pairwise; a tail of fewer than 16
// terms, and a leaf shorter than 16, fused one by one.
inline double fma_dot_leaf(const double* a, const double* b, std::int64_t n) {
    std::int64_t i = 0;
    double r = 0.;
    if (n >= 16) {
        float64x2_t acc[8];
        for (int j = 0; j < 8; ++j) acc[j] = vmulq_f64(vld1q_f64(a + 2 * j), vld1q_f64(b + 2 * j));
        for (i = 16; i + 16 <= n; i += 16)
            for (int j = 0; j < 8; ++j) acc[j] = vfmaq_f64(acc[j], vld1q_f64(a + i + 2 * j), vld1q_f64(b + i + 2 * j));
        const float64x2_t s = vaddq_f64(
            vaddq_f64(vaddq_f64(acc[0], acc[1]), vaddq_f64(acc[2], acc[3])),
            vaddq_f64(vaddq_f64(acc[4], acc[5]), vaddq_f64(acc[6], acc[7]))
        );
        r = vgetq_lane_f64(s, 0) + vgetq_lane_f64(s, 1);
    }
    for (; i < n; ++i) r = std::fma(a[i], b[i], r);
    return r;
}

// sum conj(x[i]) * y[i] over i < n (n <= kPairwiseBlock / 2), the same way: 4 complex
// lanes, each a pair of registers -- x * y gives (xr yr, xi yi), whose sum is the real
// part, and x * swap(y) gives (xr yi, xi yr), whose difference is the imaginary part.
inline cplx fma_cdot_leaf(const cplx* x, const cplx* y, std::int64_t n) {
    const double* xp = reinterpret_cast<const double*>(x);
    const double* yp = reinterpret_cast<const double*>(y);
    std::int64_t i = 0;
    double re = 0., im = 0.;
    if (n >= 4) {
        float64x2_t accr[4], acci[4];
        for (int j = 0; j < 4; ++j) {
            const float64x2_t xv = vld1q_f64(xp + 2 * j), yv = vld1q_f64(yp + 2 * j);
            accr[j] = vmulq_f64(xv, yv);
            acci[j] = vmulq_f64(xv, vextq_f64(yv, yv, 1));
        }
        for (i = 4; i + 4 <= n; i += 4)
            for (int j = 0; j < 4; ++j) {
                const float64x2_t xv = vld1q_f64(xp + 2 * (i + j)), yv = vld1q_f64(yp + 2 * (i + j));
                accr[j] = vfmaq_f64(accr[j], xv, yv);
                acci[j] = vfmaq_f64(acci[j], xv, vextq_f64(yv, yv, 1));
            }
        const float64x2_t sr = vaddq_f64(vaddq_f64(accr[0], accr[1]), vaddq_f64(accr[2], accr[3]));
        const float64x2_t si = vaddq_f64(vaddq_f64(acci[0], acci[1]), vaddq_f64(acci[2], acci[3]));
        re = vgetq_lane_f64(sr, 0) + vgetq_lane_f64(sr, 1);
        im = vgetq_lane_f64(si, 0) - vgetq_lane_f64(si, 1);
    }
    for (; i < n; ++i) {
        const double xr = xp[2 * i], xi = xp[2 * i + 1], yr = yp[2 * i], yi = yp[2 * i + 1];
        re = std::fma(xi, yi, std::fma(xr, yr, re));
        im = std::fma(-xi, yr, std::fma(xr, yi, im));
    }
    return {re, im};
}

}  // namespace detail

// The sum of n terms (double or cplx), 0 + tree as numpy's add.reduce.
template <class T, class Fill>
T pairwise_sum(std::int64_t n, Fill&& fill) {
    // A single leaf inline: short sums (a 3x3 matrix's columns, a short
    // diagonal) skip the recursive tree call. Measured: matrix_norm(ord=1) of
    // (1000, 3, 3) is 13 us with this path and 17 us without, at one thread.
    if constexpr (std::is_same_v<T, cplx>) {
        cplx s;
        if (n <= kPairwiseBlock / 2) {
            alignas(16) double raw[kPairwiseBlock];  // not cplx[]: its constructor would zero it
            cplx* buf = reinterpret_cast<cplx*>(raw);
            fill(0, n, buf);
            s = detail::pairwise_leaf_complex(buf, n);
        } else {
            s = detail::pairwise_complex(0, n, fill);
        }
        return {0. + s.real(), 0. + s.imag()};
    } else {
        if (n <= kPairwiseBlock) {
            double buf[kPairwiseBlock];
            fill(0, n, buf);
            return 0. + detail::pairwise_leaf(buf, n);
        }
        return 0. + detail::pairwise(0, n, fill);
    }
}

// ---- one long sum on many threads, with the same bits (docs/design/linalg.md §8.3) --
// The subtrees of at most kSumLeaf terms are summed in parallel and combined
// along the very same tree (pairwise::plan, pairwise::combine): bit for bit
// the serial sum, for any thread count. The constants are the array area's.
inline constexpr std::int64_t kSumLeaf = 1 << 12;          // terms per task (the array area's kParallelTask)
inline constexpr std::int64_t kSumParallelMin = 1 << 15;   // shorter sums stay serial (its kParallelThreshold)

// The leaves to compute: subtrees of at most kSumLeaf terms, left to right; or
// the whole range as one leaf when the sum is short or no pool is available.
inline pairwise::Leaves pairwise_plan(std::int64_t n, bool complex) {
    if (n < kSumParallelMin || !detail::begin_parallel()) return {{0, n}};
    return pairwise::plan(n, kSumLeaf, complex);
}

// Runs leaf(t) for every leaf of a plan: on the pool when there are several.
template <class Leaf>
void pairwise_run(const pairwise::Leaves& plan, Leaf&& leaf) {
    if (plan.size() == 1) {
        leaf(std::int64_t(0));
        return;
    }
    detail::run_tasks(static_cast<std::int64_t>(plan.size()), leaf);
}

// The leaves' partial sums combined along the tree they came from.
template <class T>
T pairwise_combine(std::int64_t n, bool /*complex: from T*/, const std::vector<T>& parts) {
    if (parts.size() == 1) return parts[0];
    return pairwise::combine(n, kSumLeaf, parts);
}

// pairwise_sum on the pool for long sums. `fill` is called from several
// threads at once, so it must not write anything shared.
template <class T, class Fill>
T pairwise_sum_mt(std::int64_t n, Fill&& fill) {
    constexpr bool complex = std::is_same_v<T, cplx>;
    if (n < kSumParallelMin) return pairwise_sum<T>(n, fill);  // short: no plan, no allocation
    const auto plan = pairwise_plan(n, complex);
    if (plan.size() == 1) {  // no pool: the serial sum, without the GIL like the split one
        const parallel::Unlocked unlocked(true);
        return pairwise_sum<T>(n, fill);
    }
    std::vector<T> parts(plan.size());
    pairwise_run(plan, [&](std::int64_t t) {
        auto f = fill;  // a copy per task
        if constexpr (complex) parts[t] = detail::pairwise_complex(plan[t].first, plan[t].second, f);
        else parts[t] = detail::pairwise(plan[t].first, plan[t].second, f);
    });
    const T s = pairwise_combine(n, complex, parts);
    if constexpr (complex) return {0. + s.real(), 0. + s.imag()};
    else return 0. + s;
}

// ---- trees of leaf sums (the product sums above) ------------------------------
// leaf_sum(begin, m) is the sum of terms begin .. begin+m-1, one leaf (at most
// kPairwiseBlock doubles: kPairwiseBlock / 2 complex). The tree, the parallel plan and
// the combination are pairwise_sum_mt's, so long sums run on the pool with the bits of
// the serial sum. `leaf_sum` is called from several threads at once.
template <class T, class LeafSum>
T pairwise_tree_mt(std::int64_t n, LeafSum&& leaf_sum) {
    constexpr bool complex = std::is_same_v<T, cplx>;
    auto tree = [&](std::int64_t b, std::int64_t m) -> T {
        if constexpr (complex) return pairwise::tree_complex(b, m, leaf_sum);
        else return pairwise::tree(b, m, leaf_sum);
    };
    T s;
    if (n < kSumParallelMin) {
        s = tree(0, n);
    } else {
        const auto plan = pairwise_plan(n, complex);
        if (plan.size() == 1) {  // no pool: the serial sum, without the GIL like the split one
            const parallel::Unlocked unlocked(true);
            s = tree(0, n);
        } else {
            std::vector<T> parts(plan.size());
            pairwise_run(plan, [&](std::int64_t t) { parts[t] = tree(plan[t].first, plan[t].second); });
            s = pairwise_combine(n, complex, parts);
        }
    }
    if constexpr (complex) return {0. + s.real(), 0. + s.imag()};
    else return 0. + s;
}

}  // namespace sd::linalg
