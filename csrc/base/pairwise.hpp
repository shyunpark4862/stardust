// Pairwise float summation, defined once for every area: the shape of the
// tree, the leaf sums, and how a long sum is cut into leaves for the pool and
// put back together. What these functions do decides the bits of every float
// sum in stardust; the areas only choose how the terms reach a leaf (array: an
// accessor, docs/design/array.md §6.2; linalg: a buffer filled per leaf,
// docs/design/linalg.md §8.3).
//
// The tree depends on the length only. Runs of at most kBlock doubles (256
// complex numbers) are summed with kLanes = 16 independent accumulators (8 per
// part for complex) -- eight NEON registers, which the compiler fills from the
// 16 chains -- and longer runs are split in two at a multiple of 16 doubles,
// recursively. The rounding error grows like log n instead of n. This is
// numpy's algorithm (DOUBLE_pairwise_sum) with a wider leaf (numpy: 8
// accumulators, blocks of 128), so the last bits can differ from numpy's: a
// long sum ran 1.2-1.6x faster on one thread (docs/design/array.md §6.2).
#pragma once

#include "base/dtype.hpp"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace sd::pairwise {

inline constexpr std::int64_t kBlock = 512;  // in doubles
inline constexpr int kLanes = 16;            // accumulators of a real leaf (8 per part for complex)

// Where a run of n terms splits: half of it, rounded down to a multiple of 16
// doubles. Complex terms count as two doubles, so the cut is at
// (n - n % 16) / 2 complex numbers.
inline std::int64_t split(std::int64_t n) {
    const std::int64_t m = n / 2;
    return m - m % kLanes;
}
inline std::int64_t split_complex(std::int64_t n) { return (n - n % kLanes) / 2; }

// The 16 accumulators of a leaf, added in a fixed tree: adjacent pairs, then
// pairs of pairs.
[[gnu::always_inline]] inline double fold_lanes(double* r) {
    for (int w = 1; w < kLanes; w *= 2)
        for (int j = 0; j < kLanes; j += 2 * w) r[j] += r[j + w];
    return r[0];
}

// A leaf: get(0) + ... + get(n - 1) for n <= kBlock.
template <typename Get>
[[gnu::always_inline]] inline double leaf(std::int64_t n, const Get& get) {
    if (n < kLanes) {
        double res = 0.;
        for (std::int64_t i = 0; i < n; ++i) res += get(i);
        return res;
    }
    double r[kLanes];
    for (int j = 0; j < kLanes; ++j) r[j] = get(j);
    std::int64_t i = kLanes;
    for (; i + kLanes <= n; i += kLanes)
        for (int j = 0; j < kLanes; ++j) r[j] += get(i + j);  // 16 independent chains
    double res = fold_lanes(r);
    for (; i < n; ++i) res += get(i);
    return res;
}

// A leaf of squares, dev(0)^2 + ... + dev(n - 1)^2, each square added with one
// rounding: fma(d, d, acc) (the second pass of var, docs/design/array.md §6.8).
template <typename Dev>
[[gnu::always_inline]] inline double leaf_squares(std::int64_t n, const Dev& dev) {
    if (n < kLanes) {
        double res = 0.;
        for (std::int64_t i = 0; i < n; ++i) {
            const double d = dev(i);
            res = std::fma(d, d, res);
        }
        return res;
    }
    double r[kLanes];
    for (int j = 0; j < kLanes; ++j) {
        const double d = dev(j);
        r[j] = d * d;
    }
    std::int64_t i = kLanes;
    for (; i + kLanes <= n; i += kLanes)
        for (int j = 0; j < kLanes; ++j) {
            const double d = dev(i + j);
            r[j] = std::fma(d, d, r[j]);
        }
    double res = fold_lanes(r);
    for (; i < n; ++i) {
        const double d = dev(i);
        res = std::fma(d, d, res);
    }
    return res;
}

// A complex leaf (n <= kBlock / 2): real and imaginary parts side by side,
// 8 accumulators each.
template <typename Get>
[[gnu::always_inline]] inline complex128 leaf_complex(std::int64_t n, const Get& get) {
    constexpr int kPart = kLanes / 2;
    if (n < kPart) {
        double rr = 0., ri = 0.;
        for (std::int64_t i = 0; i < n; ++i) {
            const complex128 z = get(i);
            rr += z.real();
            ri += z.imag();
        }
        return {rr, ri};
    }
    double re[kPart], im[kPart];
    for (int j = 0; j < kPart; ++j) {
        const complex128 z = get(j);
        re[j] = z.real();
        im[j] = z.imag();
    }
    std::int64_t i = kPart;
    for (; i + kPart <= n; i += kPart)
        for (int j = 0; j < kPart; ++j) {
            const complex128 z = get(i + j);
            re[j] += z.real();
            im[j] += z.imag();
        }
    for (int w = 1; w < kPart; w *= 2)
        for (int j = 0; j < kPart; j += 2 * w) {
            re[j] += re[j + w];
            im[j] += im[j + w];
        }
    double rr = re[0], ri = im[0];
    for (; i < n; ++i) {
        const complex128 z = get(i);
        rr += z.real();
        ri += z.imag();
    }
    return {rr, ri};
}

inline complex128 add(complex128 a, complex128 b) { return {a.real() + b.real(), a.imag() + b.imag()}; }

// The tree over terms [begin, begin + n): leaf_sum(b, m) sums one leaf.
template <typename LeafSum>
double tree(std::int64_t begin, std::int64_t n, const LeafSum& leaf_sum) {
    if (n <= kBlock) return leaf_sum(begin, n);
    const std::int64_t n2 = split(n);
    return tree(begin, n2, leaf_sum) + tree(begin + n2, n - n2, leaf_sum);
}

template <typename LeafSum>
complex128 tree_complex(std::int64_t begin, std::int64_t n, const LeafSum& leaf_sum) {
    if (n <= kBlock / 2) return leaf_sum(begin, n);
    const std::int64_t n2 = split_complex(n);
    const complex128 a = tree_complex(begin, n2, leaf_sum), b = tree_complex(begin + n2, n - n2, leaf_sum);
    return add(a, b);
}

// ---- one long sum on many threads, with the same bits ----------------------
// The subtrees of at most `task` terms are independent: plan() lists them left
// to right, the caller sums them in any order (on the pool), and combine()
// adds the partial sums along the very same tree. The result is bit for bit
// the serial tree's, for any thread count.

using Leaves = std::vector<std::pair<std::int64_t, std::int64_t>>;  // (begin, n)

inline Leaves plan(std::int64_t n, std::int64_t task, bool complex) {
    Leaves leaves;
    auto collect = [&](auto&& self, std::int64_t b, std::int64_t m) -> void {
        if (m <= task) {
            leaves.emplace_back(b, m);
            return;
        }
        const std::int64_t m2 = complex ? split_complex(m) : split(m);
        self(self, b, m2);
        self(self, b + m2, m - m2);
    };
    collect(collect, 0, n);
    return leaves;
}

template <typename T>
T combine(std::int64_t n, std::int64_t task, const std::vector<T>& parts) {
    constexpr bool complex = std::is_same_v<T, complex128>;
    std::size_t next = 0;
    auto rec = [&](auto&& self, std::int64_t m) -> T {
        if (m <= task) return parts[next++];
        const std::int64_t m2 = complex ? split_complex(m) : split(m);
        const T left = self(self, m2);
        const T right = self(self, m - m2);
        if constexpr (complex) return add(left, right);
        else return left + right;
    };
    return rec(rec, n);
}

}  // namespace sd::pairwise
