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

#include "core/types.hpp"

#include <cstdint>

namespace sdla {

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

}  // namespace sdla
