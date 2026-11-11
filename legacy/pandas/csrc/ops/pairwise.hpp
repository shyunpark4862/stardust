// numpy's pairwise float summation and IEEE 754-2019 min/max, shared by the
// reductions of M5 and the grouped ones of M6 so both give the same bits
// (API.md §15, §16).
#pragma once

#include <arm_neon.h>

#include <cstdint>

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

// numpy's add.reduce starts from the identity: 0.0 + pairwise(all). That
// start is why a sum of -0.0s is +0.0, as in numpy.
template <class Get>
double float_sum(std::int64_t n, const Get& get) {
    return 0.0 + pairwise_sum(0, n, get);
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
