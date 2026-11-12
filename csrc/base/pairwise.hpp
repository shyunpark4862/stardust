// numpy's pairwise float summation (DOUBLE_pairwise_sum / CDOUBLE_pairwise_sum),
// defined once for every area: the shape of the tree, the leaf sums, and how a
// long sum is cut into leaves for the pool and put back together. What these
// functions do decides the bits of every float sum in stardust; the areas only
// choose how the terms reach a leaf (array: an accessor, docs/design/array.md
// §6.2; linalg: a buffer filled per leaf, docs/design/linalg.md §8.3).
//
// The tree depends on the length only. Runs of at most kBlock doubles (64
// complex numbers) are summed with 8 independent accumulators (4 per part for
// complex), longer runs are split in two at a multiple of 8 doubles,
// recursively. The rounding error grows like log n instead of n, and the 8
// chains keep the FP units busy.
#pragma once

#include "base/dtype.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace sd::pairwise {

inline constexpr std::int64_t kBlock = 128;  // numpy's PW_BLOCKSIZE, in doubles

// Where a run of n terms splits: half of it, rounded down to a multiple of 8
// doubles. numpy counts doubles, so for complex terms the cut is at
// ((2n)/2 - ((2n)/2) % 8) doubles, i.e. (n - n % 8) / 2 complex numbers.
inline std::int64_t split(std::int64_t n) {
    const std::int64_t m = n / 2;
    return m - m % 8;
}
inline std::int64_t split_complex(std::int64_t n) { return (n - n % 8) / 2; }

// A leaf: get(0) + ... + get(n - 1) for n <= kBlock.
template <typename Get>
[[gnu::always_inline]] inline double leaf(std::int64_t n, const Get& get) {
    if (n < 8) {
        double res = 0.;
        for (std::int64_t i = 0; i < n; ++i) res += get(i);
        return res;
    }
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = get(j);
    std::int64_t i = 8;
    for (; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; ++j) r[j] += get(i + j);  // 8 independent chains
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += get(i);
    return res;
}

// A complex leaf (n <= kBlock / 2): real and imaginary parts side by side,
// 4 accumulators each.
template <typename Get>
[[gnu::always_inline]] inline complex128 leaf_complex(std::int64_t n, const Get& get) {
    if (n < 4) {
        double rr = 0., ri = 0.;
        for (std::int64_t i = 0; i < n; ++i) {
            const complex128 z = get(i);
            rr += z.real();
            ri += z.imag();
        }
        return {rr, ri};
    }
    double re[4], im[4];
    for (int j = 0; j < 4; ++j) {
        const complex128 z = get(j);
        re[j] = z.real();
        im[j] = z.imag();
    }
    std::int64_t i = 4;
    for (; i < n - (n % 4); i += 4)
        for (int j = 0; j < 4; ++j) {
            const complex128 z = get(i + j);
            re[j] += z.real();
            im[j] += z.imag();
        }
    double rr = (re[0] + re[1]) + (re[2] + re[3]);
    double ri = (im[0] + im[1]) + (im[2] + im[3]);
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
