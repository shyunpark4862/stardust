// numpy's pairwise float summation and IEEE 754-2019 min/max, shared by the
// reductions and the grouped ones so both give the same bits
// (docs/api/frame.md §6.5, §7.5). The sum itself -- the tree, the leaf sums, the parallel plan -- is
// the one definition of every area (base/pairwise.hpp), so a frame sum has
// the array area's bits.
#pragma once

#include "base/pairwise.hpp"
#include "frame/parallel.hpp"

#include <arm_neon.h>

#include <cstdint>
#include <vector>

namespace sd::frame {

// get(begin) + ... + get(begin + n - 1) along numpy's tree.
template <class Get>
double pairwise_sum(std::int64_t begin, std::int64_t n, const Get& get) {
    return pairwise::tree(begin, n, [&get](std::int64_t b, std::int64_t m) {
        return pairwise::leaf(m, [&get, b](std::int64_t i) { return get(b + i); });
    });
}

// The pairwise tree in parallel: the subtrees of at most kPairwiseLeaf
// values are the tasks (pairwise::plan), each summed by pairwise_sum itself;
// then the nodes above them are added in the same shape (pairwise::combine).
// Every node is the one the serial recursion computes, so the bits are the
// same as pairwise_sum(0, n) for any thread count (docs/design/frame.md §7.4). The tree
// depends on n alone.
inline constexpr std::int64_t kPairwiseLeaf = 1 << 14;

// pairwise_sum(0, n, get), bit for bit, with the leaves in parallel.
// `cost` as in for_chunks.
template <class Get>
double pairwise_sum_parallel(std::int64_t n, const Get& get, std::int64_t cost = 1) {
    if (n <= kPairwiseLeaf || n * cost < kParallelThreshold) return pairwise_sum(0, n, get);
    const pairwise::Leaves leaves = pairwise::plan(n, kPairwiseLeaf, /*complex=*/false);
    std::vector<double> part(leaves.size());
    run_unlocked(static_cast<std::int64_t>(part.size()), [&](std::int64_t t) {
        const auto k = static_cast<std::size_t>(t);
        part[k] = pairwise_sum(leaves[k].first, leaves[k].second, get);
    });
    return pairwise::combine(n, kPairwiseLeaf, part);
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

}  // namespace sd::frame
