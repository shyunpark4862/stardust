// MINPACK's arithmetic as scipy 1.18.1 computes it (docs/design/optimize.md §3.4, §3.6): scipy's
// _minpack module is a C translation of MINPACK built with clang's default contraction
// (-ffp-contract=on), read from its machine code. hybrd.cpp (root's 'hybr') and the fused
// lmder of lsq.cpp (least_squares' and root's 'lm') share these pieces.
//
// The fused multiply-adds are written out (std::fma), so the code computes the same whatever
// contraction the including file asks for.
#pragma once

#include <arm_neon.h>

#include <bit>
#include <cmath>
#include <cstdint>

namespace sd::optimize::minpack {

using lint = std::int64_t;

// scipy's enorm constants: 2^-64.5 and 2^63.5, as double
inline const double kRdwarf = std::bit_cast<double>(std::uint64_t{0x3be6a09e667f3bccULL});
inline const double kRgiant = std::bit_cast<double>(std::uint64_t{0x43e6a09e667f3bcdULL});

// t^2 rounded on its own (MINPACK's t**2; scipy's pow(t, 2)), never fused into a sum
inline double sq(double t) { return t * t; }

// MINPACK's enorm: the Euclidean norm with separate sums for the small, the intermediate and the
// large components (no overflow or destructive underflow).
inline double enorm(lint n, const double* x) {
    double s1 = 0, s2 = 0, s3 = 0, x1max = 0, x3max = 0;
    const double agiant = kRgiant / static_cast<double>(n);
    for (lint i = 0; i < n; ++i) {
        const double xabs = std::abs(x[i]);
        // (the squares of the ratios are rounded before they are used: scipy's pow(t, 2))
        if (xabs > kRdwarf && xabs < agiant) {
            s2 = std::fma(xabs, xabs, s2);
        } else if (xabs > kRdwarf) {
            if (xabs > x1max) {
                const double t2 = sq(x1max / xabs);
                s1 = std::fma(s1, t2, 1.0);
                x1max = xabs;
            } else {
                s1 = s1 + sq(xabs / x1max);
            }
        } else {
            if (xabs > x3max) {
                const double t2 = sq(x3max / xabs);
                s3 = std::fma(s3, t2, 1.0);
                x3max = xabs;
            } else if (xabs != 0) {
                s3 = s3 + sq(xabs / x3max);
            }
        }
    }
    if (s1 != 0) return x1max * std::sqrt(s1 + (s2 / x1max) / x1max);
    if (s2 != 0) {
        if (s2 >= x3max) return std::sqrt(s2 * std::fma(x3max / s2, x3max * s3, 1.0));
        return std::sqrt(x3max * std::fma(x3max, s3, s2 / x3max));
    }
    return x3max * std::sqrt(s3);
}

// A running sum of products as scipy's build computes it where clang vectorized the loop
// (qrfac's inner product, lmder's gradient test): the products of each full block of 8 are
// rounded and added in order; the rest are fused.
inline double dot8(lint len, const double* a, const double* b) {
    double sum = 0;
    lint i = 0;
    for (; i + 8 <= len; i += 8)
        for (lint k = i; k < i + 8; ++k) {
            const double p = a[k] * b[k];
            sum = sum + p;
        }
    for (; i < len; ++i) sum = std::fma(a[i], b[i], sum);
    return sum;
}

// The running sums s_g = sum_i v[i] c_g[i] of G columns at once over [0, len): the products
// rounded on their own and added in order up to split, fused after it (dot8 is split =
// len / 8 * 8; a fused sum is split = 0; an unfused one split = len). Each sum takes its own
// column's order: the G independent chains (and the products of two rows at a time) only serve
// the processor's parallelism, so the bits are those of one column at a time.
template <int G>
inline void column_sums(lint len, lint split, const double* v, double* const* c, double* s) {
    for (int g = 0; g < G; ++g) s[g] = 0;
    lint i = 0;
    for (; i + 2 <= split; i += 2) {
        const float64x2_t x = vld1q_f64(v + i);
        for (int g = 0; g < G; ++g) {
            const float64x2_t p = vmulq_f64(x, vld1q_f64(c[g] + i));
            s[g] = s[g] + vgetq_lane_f64(p, 0);
            s[g] = s[g] + vgetq_lane_f64(p, 1);
        }
    }
    for (; i < split; ++i)
        for (int g = 0; g < G; ++g) {
            const double p = v[i] * c[g][i];
            s[g] = s[g] + p;
        }
    for (; i < len; ++i)
        for (int g = 0; g < G; ++g) s[g] = std::fma(v[i], c[g][i], s[g]);
}

// c_g[i] = fma(t_g, v[i], c_g[i]) over [0, len) for G columns in one pass over v
template <int G>
inline void column_axpy(lint len, const double* v, double* const* c, const double* t) {
    lint i = 0;
    for (; i + 2 <= len; i += 2) {
        const float64x2_t x = vld1q_f64(v + i);
        for (int g = 0; g < G; ++g) vst1q_f64(c[g] + i, vfmaq_n_f64(vld1q_f64(c[g] + i), x, t[g]));
    }
    for (; i < len; ++i)
        for (int g = 0; g < G; ++g) c[g][i] = std::fma(t[g], v[i], c[g][i]);
}

// One Householder step on the columns [k0, k1) of a (leading dimension lda; a points at the step's
// first row of column 0): each column c gets c -= (sum_i v[i] c[i] / v[0]) v over len rows, the
// sum as column_sums with split (qrfac: dot8; qform: fused). G columns at a time.
inline void reflect_columns(lint len, lint split, const double* v, double* a, lint lda, lint k0, lint k1) {
    constexpr int G = 8;
    double* c[G];
    double s[G], t[G];
    lint k = k0;
    for (; k + G <= k1; k += G) {
        for (int g = 0; g < G; ++g) c[g] = a + (k + g) * lda;
        column_sums<G>(len, split, v, c, s);
        for (int g = 0; g < G; ++g) t[g] = -(s[g] / v[0]);
        column_axpy<G>(len, v, c, t);
    }
    for (; k < k1; ++k) {
        c[0] = a + k * lda;
        column_sums<1>(len, split, v, c, s);
        t[0] = -(s[0] / v[0]);
        column_axpy<1>(len, v, c, t);
    }
}

}  // namespace sd::optimize::minpack
