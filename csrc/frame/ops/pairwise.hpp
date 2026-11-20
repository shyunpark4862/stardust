// Pairwise float summation and IEEE 754-2019 min/max, shared by the
// reductions and the grouped ones so both give the same bits
// (docs/api/frame.md §6.5, §7.5). The sum itself -- the tree, the leaf sums
// (with one fma per square for var), the parallel plan -- is the one
// definition of every area (base/pairwise.hpp), so a frame sum or var has the
// array area's bits.
#pragma once

#include "base/pairwise.hpp"
#include "frame/parallel.hpp"

#include <arm_neon.h>

#include <cstdint>
#include <vector>

namespace sd::frame {

// get(begin) + ... + get(begin + n - 1) along the pairwise tree.
template <class Get>
double pairwise_sum(std::int64_t begin, std::int64_t n, const Get& get) {
    return pairwise::tree(begin, n, [&get](std::int64_t b, std::int64_t m) {
        return pairwise::leaf(m, [&get, b](std::int64_t i) { return get(b + i); });
    });
}

// dev(begin)^2 + ... + dev(begin + n - 1)^2 along the same tree, each square
// added by one fma (pairwise::leaf_squares): the second pass of var.
template <class Dev>
double pairwise_squares(std::int64_t begin, std::int64_t n, const Dev& dev) {
    return pairwise::tree(begin, n, [&dev](std::int64_t b, std::int64_t m) {
        return pairwise::leaf_squares(m, [&dev, b](std::int64_t i) { return dev(b + i); });
    });
}

// The pairwise tree in parallel: the subtrees of at most kPairwiseLeaf
// values are the tasks (pairwise::plan), each summed by pairwise_sum itself;
// then the nodes above them are added in the same shape (pairwise::combine).
// Every node is the one the serial recursion computes, so the bits are the
// same as pairwise_sum(0, n) for any thread count (docs/design/frame.md §7.4). The tree
// depends on n alone.
inline constexpr std::int64_t kPairwiseLeaf = 1 << 14;

// tree(0, n), bit for bit, with the subtrees in parallel; tree(b, m) is
// pairwise_sum or pairwise_squares. `cost` as in for_chunks.
template <class Tree>
double pairwise_parallel(std::int64_t n, const Tree& tree, std::int64_t cost) {
    if (n <= kPairwiseLeaf || n * cost < kParallelThreshold) return tree(0, n);
    const pairwise::Leaves leaves = pairwise::plan(n, kPairwiseLeaf, /*complex=*/false);
    std::vector<double> part(leaves.size());
    run_unlocked(static_cast<std::int64_t>(part.size()), [&](std::int64_t t) {
        const auto k = static_cast<std::size_t>(t);
        part[k] = tree(leaves[k].first, leaves[k].second);
    });
    return pairwise::combine(n, kPairwiseLeaf, part);
}

// A sum starts from the identity: 0.0 + pairwise(all), as the array area's
// (and numpy's add.reduce). That start is why a sum of -0.0s is +0.0.
template <class Get>
double float_sum(std::int64_t n, const Get& get, std::int64_t cost = 1) {
    return 0.0 + pairwise_parallel(n, [&get](std::int64_t b, std::int64_t m) { return pairwise_sum(b, m, get); }, cost);
}

// The sum of dev(i)^2 for var: 0.0 + the squares' tree, as the array area's var.
template <class Dev>
double squares_sum(std::int64_t n, const Dev& dev, std::int64_t cost = 1) {
    return 0.0 +
           pairwise_parallel(n, [&dev](std::int64_t b, std::int64_t m) { return pairwise_squares(b, m, dev); }, cost);
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
