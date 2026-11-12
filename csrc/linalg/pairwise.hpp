// Pairwise summation for every floating-point sum in linalg: numpy's
// DOUBLE_pairwise_sum and CDOUBLE_pairwise_sum as defined once in
// base/pairwise.hpp (the tree, the leaf sums, the parallel plan), so linalg's
// sums have the array area's bits (docs/design/linalg.md §8.2). A sum is
// 0.0 + tree, as numpy's add.reduce does.
//
// What is linalg's own is how terms reach a leaf: fill(begin, len, buf)
// writes terms begin .. begin+len-1 (at most one leaf) to buf, in order. A
// leaf is filled first and then summed, so the summing loop is the same for
// every source layout and vectorizes.
#pragma once

#include "base/pairwise.hpp"
#include "linalg/batch.hpp"
#include "linalg/types.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace sd::linalg {

inline constexpr std::int64_t kPairwiseBlock = pairwise::kBlock;  // numpy's PW_BLOCKSIZE, in doubles

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
    if (plan.size() == 1) return pairwise_sum<T>(n, fill);
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

}  // namespace sd::linalg
