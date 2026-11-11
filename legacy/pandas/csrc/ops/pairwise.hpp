// numpy's pairwise float summation and IEEE 754-2019 min/max, shared by the
// reductions of M5 and the grouped ones of M6 so both give the same bits
// (API.md §15, §16).
#pragma once

#include "parallel/pool.hpp"

#include <arm_neon.h>

#include <cstdint>
#include <vector>

namespace sd {

// numpy's DOUBLE_pairwise_sum, as in stardust-numpy: eight accumulators in
// blocks of 128, the halves of a longer run added recursively. Eight
// independent chains are faster than one, and the rounding error grows like
// log n instead of n.
inline constexpr std::int64_t kPairwiseBlock = 128;  // numpy's PW_BLOCKSIZE, in doubles

template <class Get>
double pairwise_sum(std::int64_t begin, std::int64_t n, const Get& get) {
    if (n < 8) {
        double res = 0.;
        for (std::int64_t i = 0; i < n; ++i) res += get(begin + i);
        return res;
    }
    if (n <= kPairwiseBlock) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = get(begin + j);
        std::int64_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += get(begin + i + j);
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += get(begin + i);
        return res;
    }
    std::int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(begin, n2, get) + pairwise_sum(begin + n2, n - n2, get);
}

// The pairwise tree in parallel (M11): the subtrees of at most
// kPairwiseLeaf values are the tasks, each summed by pairwise_sum itself;
// then the nodes above them are added in the same shape. Every node is the
// one the serial recursion computes, so the bits are the same as
// pairwise_sum(0, n) for any thread count (API.md §15, as in stardust-numpy's
// pairwise_parallel). The tree depends on n alone.
inline constexpr std::int64_t kPairwiseLeaf = 1 << 14;

namespace detail {

// The leaves of the tree over [begin, begin + n), left to right.
inline void pairwise_leaves(std::int64_t begin, std::int64_t n, std::vector<std::int64_t>& starts,
                            std::vector<std::int64_t>& sizes) {
    if (n <= kPairwiseLeaf) {
        starts.push_back(begin);
        sizes.push_back(n);
        return;
    }
    std::int64_t n2 = n / 2;  // the split of pairwise_sum, for n > kPairwiseBlock
    n2 -= n2 % 8;
    pairwise_leaves(begin, n2, starts, sizes);
    pairwise_leaves(begin + n2, n - n2, starts, sizes);
}

// Adds the leaf sums back up in the shape of the tree; `next` walks the leaves.
inline double pairwise_combine(std::int64_t n, const std::vector<double>& leaf, std::size_t& next) {
    if (n <= kPairwiseLeaf) return leaf[next++];
    std::int64_t n2 = n / 2;
    n2 -= n2 % 8;
    const double left = pairwise_combine(n2, leaf, next);
    return left + pairwise_combine(n - n2, leaf, next);
}

}  // namespace detail

// pairwise_sum(0, n, get), bit for bit, with the leaves in parallel.
// `cost` as in parallel::for_chunks.
template <class Get>
double pairwise_sum_parallel(std::int64_t n, const Get& get, std::int64_t cost = 1) {
    if (n <= kPairwiseLeaf || n * cost < parallel::kThreshold) return pairwise_sum(0, n, get);
    std::vector<std::int64_t> starts, sizes;
    detail::pairwise_leaves(0, n, starts, sizes);
    std::vector<double> leaf(starts.size());
    parallel::parallel_for(static_cast<std::int64_t>(leaf.size()), [&](std::int64_t t) {
        const auto k = static_cast<std::size_t>(t);
        leaf[k] = pairwise_sum(starts[k], sizes[k], get);
    });
    std::size_t next = 0;
    return detail::pairwise_combine(n, leaf, next);
}

// numpy's add.reduce starts from the identity: 0.0 + pairwise(all). That
// start is why a sum of -0.0s is +0.0, as in numpy.
template <class Get>
double float_sum(std::int64_t n, const Get& get, std::int64_t cost = 1) {
    return 0.0 + pairwise_sum_parallel(n, get, cost);
}

// IEEE 754-2019 minimum/maximum, which is what AArch64's FMIN/FMAX do: NaN
// propagates and -0.0 < 0.0 (as numpy). One lane of the vector instruction
// does the scalar case, so scalar and vector paths agree bit for bit.
template <bool Max>
double ieee_minmax(double a, double b) {
    const float64x2_t x = vdupq_n_f64(a), y = vdupq_n_f64(b);
    return vgetq_lane_f64(Max ? vmaxq_f64(x, y) : vminq_f64(x, y), 0);
}

}  // namespace sd
