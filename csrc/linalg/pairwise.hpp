// Pairwise summation: numpy's (and stardust-numpy's) DOUBLE_pairwise_sum and
// CDOUBLE_pairwise_sum, for every floating-point sum here (M8, user decision,
// DESIGN.md §2.17).
//
// The tree depends on the length only: runs of at most 128 doubles are summed
// with 8 independent accumulators (4 complex ones), and longer runs are split
// in two at a multiple of 8, recursively. The rounding error grows like log n
// instead of n, and 8 chains run in parallel on the hardware. A sum is
// 0.0 + tree, as numpy's add.reduce does, so the same terms give numpy's bits.
//
// The terms come from fill(begin, len, buf): it writes terms begin .. begin+len-1
// (at most one leaf) to buf, in order. A leaf is filled first and then summed,
// so the summing loop is the same for every source layout and vectorizes.
#pragma once

#include "linalg/batch.hpp"
#include "linalg/types.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace sd::linalg {

inline constexpr std::int64_t kPairwiseBlock = 128;  // numpy's PW_BLOCKSIZE, in doubles

namespace detail {

inline double pairwise_leaf(const double* a, std::int64_t n) {
    if (n < 8) {
        double res = 0.;
        for (std::int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    std::int64_t i = 8;
    for (; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; ++j) r[j] += a[i + j];  // 8 independent chains
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
}

template <class Fill>
double pairwise(std::int64_t begin, std::int64_t n, Fill& fill) {
    if (n <= kPairwiseBlock) {
        double buf[kPairwiseBlock];
        fill(begin, n, buf);
        return pairwise_leaf(buf, n);
    }
    std::int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(begin, n2, fill) + pairwise(begin + n2, n - n2, fill);
}

// The complex leaf: numpy counts doubles, so a leaf holds 64 complex numbers
// and the real and imaginary parts have 4 accumulators each.
inline cplx pairwise_leaf_complex(const cplx* a, std::int64_t n) {
    if (n < 4) {
        double rr = 0., ri = 0.;
        for (std::int64_t i = 0; i < n; ++i) {
            rr += a[i].real();
            ri += a[i].imag();
        }
        return {rr, ri};
    }
    double re[4], im[4];
    for (int j = 0; j < 4; ++j) {
        re[j] = a[j].real();
        im[j] = a[j].imag();
    }
    std::int64_t i = 4;
    for (; i < n - (n % 4); i += 4)
        for (int j = 0; j < 4; ++j) {
            re[j] += a[i + j].real();
            im[j] += a[i + j].imag();
        }
    double rr = (re[0] + re[1]) + (re[2] + re[3]);
    double ri = (im[0] + im[1]) + (im[2] + im[3]);
    for (; i < n; ++i) {
        rr += a[i].real();
        ri += a[i].imag();
    }
    return {rr, ri};
}

template <class Fill>
cplx pairwise_complex(std::int64_t begin, std::int64_t n, Fill& fill) {
    if (n <= kPairwiseBlock / 2) {
        alignas(16) double raw[kPairwiseBlock];  // not cplx[]: its constructor would zero it
        cplx* buf = reinterpret_cast<cplx*>(raw);
        fill(begin, n, buf);
        return pairwise_leaf_complex(buf, n);
    }
    std::int64_t n2 = n;  // numpy: n2 = (2n)/2 - ((2n)/2) % 8 doubles
    n2 -= n2 % 8;
    n2 /= 2;
    const cplx a = pairwise_complex(begin, n2, fill), b = pairwise_complex(begin + n2, n - n2, fill);
    return {a.real() + b.real(), a.imag() + b.imag()};
}

}  // namespace detail

// The sum of n terms (double or cplx), 0 + tree as numpy's add.reduce.
template <class T, class Fill>
T pairwise_sum(std::int64_t n, Fill&& fill) {
    // A single leaf inline (short sums: no recursive call), else the tree.
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


// ---- one long sum on many threads, with the same bits (M9, user decision) --
// As stardust-numpy does it (its reduce.hpp, M8): the tree depends on n only,
// so its subtrees of at most kSumLeaf terms are independent. They are summed
// in parallel and then combined along the very same tree: bit for bit the
// serial sum, for any thread count.

inline constexpr std::int64_t kSumLeaf = 1 << 12;          // terms per task (stardust-numpy's kTaskWork)
inline constexpr std::int64_t kSumParallelMin = 1 << 15;   // shorter sums stay serial (its kThreshold)

namespace detail {

inline std::int64_t split_at(std::int64_t m, bool complex) {
    if (complex) {
        std::int64_t m2 = m;
        m2 -= m2 % 8;
        return m2 / 2;
    }
    const std::int64_t m2 = m / 2;
    return m2 - m2 % 8;
}

}  // namespace detail

// The leaves to compute: subtrees of at most kSumLeaf terms, left to right; or
// the whole range as one leaf when the sum is short or no pool is available.
inline std::vector<std::pair<std::int64_t, std::int64_t>> pairwise_plan(std::int64_t n, bool complex) {
    std::vector<std::pair<std::int64_t, std::int64_t>> leaves;
    if (n < kSumParallelMin || !detail::begin_parallel()) {
        leaves.emplace_back(0, n);
        return leaves;
    }
    auto collect = [&](auto&& self, std::int64_t b, std::int64_t m) -> void {
        if (m <= kSumLeaf) {
            leaves.emplace_back(b, m);
            return;
        }
        const std::int64_t m2 = detail::split_at(m, complex);
        self(self, b, m2);
        self(self, b + m2, m - m2);
    };
    collect(collect, 0, n);
    return leaves;
}

// Runs leaf(t) for every leaf of a plan: on the pool when there are several.
template <class Leaf>
void pairwise_run(const std::vector<std::pair<std::int64_t, std::int64_t>>& plan, Leaf&& leaf) {
    if (plan.size() == 1) {
        leaf(std::int64_t(0));
        return;
    }
    detail::run_tasks(static_cast<std::int64_t>(plan.size()), leaf);
}

// The leaves' partial sums combined along the tree they came from.
template <class T>
T pairwise_combine(std::int64_t n, bool complex, const std::vector<T>& parts) {
    if (parts.size() == 1) return parts[0];
    std::size_t next = 0;
    auto combine = [&](auto&& self, std::int64_t m) -> T {
        if (m <= kSumLeaf) return parts[next++];
        const std::int64_t m2 = detail::split_at(m, complex);
        const T left = self(self, m2);
        const T right = self(self, m - m2);
        if constexpr (std::is_same_v<T, cplx>) return {left.real() + right.real(), left.imag() + right.imag()};
        else return left + right;
    };
    return combine(combine, n);
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
