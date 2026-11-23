// The Gaussian sums of sd.smooth (docs/design/smooth.md §3.4, §3.5; decisions SM80-SM83): the kernel regression's
// weights (smooth/kreg.cpp) and the kernel density's sums (milestone SM5) share it.
//
// A sum over data points x_j with prior weights w_j > 0 at a query point x0 is taken in bandwidth units: the squared
// distance D_j = sum_l z_l^2, z_l = ((x_jl - x0_l) sc_l) inv_l (inv_l = 1 / (h_l sc_l), sc_l = 2^1000 when
// h_l < 2^-900 so that 1 / h does not overflow; the products by powers of two are exact), and every term is
// e_j = w_j exp(-(D_j - D0) / 2) with D0 the smallest D_j: the common factor exp(-D0 / 2) is taken out, so a query
// point far from every data point (every double weight underflows) still has the ratio of its sums, and the
// logarithm of a sum is -D0 / 2 + log(sum e).
//
//   exponent   D_j - D0 is computed from the nearest point c as sum_l d_l (2 z_cl + d_l), d_l = ((x_jl - x_cl) sc_l)
//              inv_l (a difference of data coordinates), not as D_j - D0: the subtraction of two rounded D loses
//              u D0 absolutely, which far from the data (D0 = 10^6) moved the weights by 10^-10 and the local
//              constant by 7000 times its contract (decision SM81)
//
//   cut        terms with D_j - D0 > T, T = 2 (ln(n / u) + ln(w_max / w_min)), are left out: each is below
//              u / n times the largest term (that of D0, at least w_min), and fewer than n of them sum below
//              u times the kept sum (§4.7's proof). A sum whose terms may cancel (a local line's slope) takes
//              kUnderflow instead: beyond it every term is exactly 0, so the cut changes no bit
//   1-D        data sorted ascending: the nearest point by a binary search, the kept points one contiguous range
//              (D_j is monotone in |x_j - x0| after rounding), found by binary searches on either side
//   n-D        D_j of every point, the kept points in index order (milestone SM5 adds a tree's candidates: the
//              same set in the same order gives the same bits)
//   exp        the array area's NEON exp (array/kernels/vmath.hpp): a pair of lanes and the scalar twin give the
//              same bits, so a term never depends on its position
//   sums       sum and dot add in blocks of 64 with four accumulators (j mod 4) and add the blocks pairwise: the
//              order is a function of the length alone
#pragma once

#include "array/kernels/vmath.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sd::smooth::gauss {

inline constexpr double kUnit = 0x1p-53;
inline constexpr double kUnderflow = 1492.0;  // exp(-746) rounds to 0: terms beyond it are exactly 0
inline constexpr double kTinyBandwidth = 0x1p-900;
inline constexpr double kBandwidthScale = 0x1p1000;

// The cut T for n points of positive weight whose weights span the ratio w_max / w_min (1 without weights).
inline double cut_threshold(std::int64_t n, double wratio) {
    return 2.0 * (std::log(static_cast<double>(std::max<std::int64_t>(n, 1)) / kUnit) + std::log(wratio));
}

// One axis' scaling: z = ((x - x0) sc) inv.
struct Axis {
    double sc = 1.0, inv = 1.0;
};

inline Axis axis_of(double h) {
    Axis a;
    a.sc = h < kTinyBandwidth ? kBandwidthScale : 1.0;
    a.inv = 1.0 / (h * a.sc);
    return a;
}

inline double z_of(double x, double x0, Axis a) { return ((x - x0) * a.sc) * a.inv; }

inline double sq_of(double x, double x0, Axis a) {
    const double z = z_of(x, x0, a);
    return z * z;
}

// ---- sums in a fixed order ------------------------------------------------------------------------------------

namespace detail {
inline constexpr std::int64_t kBlock = 64;

template <class F>
inline double block_sum(std::int64_t lo, std::int64_t hi, const F& term) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::int64_t j = lo;
    for (; j + 4 <= hi; j += 4) { s0 += term(j), s1 += term(j + 1), s2 += term(j + 2), s3 += term(j + 3); }
    for (; j < hi; ++j) s0 += term(j);
    return (s0 + s1) + (s2 + s3);
}

template <class F>
inline double tree_sum(std::int64_t lo, std::int64_t hi, const F& term) {
    if (hi - lo <= kBlock) return block_sum(lo, hi, term);
    const std::int64_t nb = (hi - lo + kBlock - 1) / kBlock;
    const std::int64_t mid = lo + (nb / 2) * kBlock;
    return tree_sum(lo, mid, term) + tree_sum(mid, hi, term);
}

template <int K, class F>
inline void block_sums(std::int64_t lo, std::int64_t hi, const F& term, double* out) {
    double acc[4][K] = {}, v[K];
    std::int64_t j = lo;
    for (; j + 4 <= hi; j += 4) {
        for (int r = 0; r < 4; ++r) {
            term(j + r, v);
            for (int k = 0; k < K; ++k) acc[r][k] += v[k];
        }
    }
    for (; j < hi; ++j) {
        term(j, v);
        for (int k = 0; k < K; ++k) acc[0][k] += v[k];
    }
    for (int k = 0; k < K; ++k) out[k] = (acc[0][k] + acc[1][k]) + (acc[2][k] + acc[3][k]);
}

template <int K, class F>
inline void tree_sums(std::int64_t lo, std::int64_t hi, const F& term, double* out) {
    if (hi - lo <= kBlock) {
        block_sums<K>(lo, hi, term, out);
        return;
    }
    const std::int64_t nb = (hi - lo + kBlock - 1) / kBlock;
    const std::int64_t mid = lo + (nb / 2) * kBlock;
    double a[K], b[K];
    tree_sums<K>(lo, mid, term, a);
    tree_sums<K>(mid, hi, term, b);
    for (int k = 0; k < K; ++k) out[k] = a[k] + b[k];
}
}  // namespace detail

// K sums at once: term(j, v) sets the K terms v[k] of j; out[k] gets their sums in the order of sum_of (each of the K
// sums has the bits sum_of would give it).
template <int K, class F>
inline void sums_of(std::int64_t n, const F& term, double* out) {
    if (n <= 0) {
        for (int k = 0; k < K; ++k) out[k] = 0.0;
        return;
    }
    detail::tree_sums<K>(0, n, term, out);
}

// sum of term(j) over [0, n), in the order set by n alone.
template <class F>
inline double sum_of(std::int64_t n, const F& term) {
    return n <= 0 ? 0.0 : detail::tree_sum(0, n, term);
}

inline double sum(const double* v, std::int64_t n) {
    return sum_of(n, [v](std::int64_t j) { return v[j]; });
}

inline double dot(const double* a, const double* b, std::int64_t n) {
    return sum_of(n, [a, b](std::int64_t j) { return a[j] * b[j]; });
}

// ---- exp of the terms -----------------------------------------------------------------------------------------

// out[j] = w[j] exp(-(D[j] - d0) / 2) (w null: 1), j in [0, n). D[j] - d0 >= 0 is the caller's.
inline void terms(const double* D, const double* w, double d0, std::int64_t n, double* out) {
    const float64x2_t vd0 = vdupq_n_f64(d0), half = vdupq_n_f64(-0.5);
    std::int64_t j = 0;
    for (; j + 2 <= n; j += 2) {
        const float64x2_t x = vmulq_f64(vsubq_f64(vld1q_f64(D + j), vd0), half);
        float64x2_t e = vdupq_n_f64(0.0);
        if (vgetq_lane_f64(x, 1) >= -vmath::kExpMax && vgetq_lane_f64(x, 0) >= -vmath::kExpMax) {
            e = vmath::exp_lanes(x);
        } else {
            e = vsetq_lane_f64(vmath::exp1(vgetq_lane_f64(x, 0)), e, 0);
            e = vsetq_lane_f64(vmath::exp1(vgetq_lane_f64(x, 1)), e, 1);
        }
        if (w) e = vmulq_f64(e, vld1q_f64(w + j));
        vst1q_f64(out + j, e);
    }
    for (; j < n; ++j) {
        const double e = vmath::exp1((D[j] - d0) * -0.5);
        out[j] = w ? e * w[j] : e;
    }
}

// ---- 1-D: sorted data -----------------------------------------------------------------------------------------

// A sorted index nearest x0 (ties: the left one), never `skip`; -1 when there is none.
inline std::int64_t nearest_1d(const double* x, std::int64_t n, double x0, std::int64_t skip = -1) {
    std::int64_t r = std::lower_bound(x, x + n, x0) - x;
    std::int64_t l = r - 1;
    if (l == skip) --l;
    if (r == skip) ++r;
    if (l < 0) return r < n ? r : -1;
    if (r >= n) return l;
    return (x0 - x[l] <= x[r] - x0) ? l : r;
}

// D_j - D0 of a point x relative to the nearest point xc whose z (relative to x0) is zc.
inline double excess_1d(double x, double xc, double zc, Axis a) {
    const double d = ((x - xc) * a.sc) * a.inv;
    return d * ((zc + zc) + d);
}

// The points [lo, hi) around the nearest point c with D_j - D0 <= T (contiguous: D_j - D0 grows with |x_j - x0| on
// either side of x0).
inline void range_1d(
    const double* x, std::int64_t n, double x0, Axis a, double T, std::int64_t c, std::int64_t& lo, std::int64_t& hi
) {
    const double zc = z_of(x[c], x0, a);
    const auto inside = [&](std::int64_t j) { return excess_1d(x[j], x[c], zc, a) <= T; };
    std::int64_t l = 0, r = c;  // first inside index in [0, c]
    while (l < r) {
        const std::int64_t m = l + (r - l) / 2;
        if (inside(m)) r = m;
        else l = m + 1;
    }
    lo = l;
    l = c + 1, r = n;  // first outside index in [c + 1, n)
    while (l < r) {
        const std::int64_t m = l + (r - l) / 2;
        if (inside(m)) l = m + 1;
        else r = m;
    }
    hi = l;
}

// ---- n-D: rows of p coordinates -------------------------------------------------------------------------------

// D[j] of every row j of X (n x p, row-major) at x0; +inf for `skip`.
inline void sqdist_nd(
    const double* X, std::int64_t n, std::int64_t p, const double* x0, const Axis* ax, std::int64_t skip, double* D
) {
    for (std::int64_t j = 0; j < n; ++j) {
        const double* row = X + j * p;
        double s = 0;
        for (std::int64_t l = 0; l < p; ++l) {
            const double z = z_of(row[l], x0[l], ax[l]);
            s += z * z;
        }
        D[j] = s;
    }
    if (skip >= 0) D[skip] = std::numeric_limits<double>::infinity();
}

// D_j - D0 of the row xj relative to the nearest row xc, whose z (relative to x0) are zc.
inline double excess_nd(const double* xj, const double* xc, const double* zc, const Axis* ax, std::int64_t p) {
    double s = 0;
    for (std::int64_t l = 0; l < p; ++l) {
        const double d = ((xj[l] - xc[l]) * ax[l].sc) * ax[l].inv;
        s += d * ((zc[l] + zc[l]) + d);
    }
    return s;
}

}  // namespace sd::smooth::gauss
