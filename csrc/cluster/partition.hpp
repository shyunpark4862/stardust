// What the partitioning kernels share (kmeans.cpp, kmedoids.cpp, fuzzy.cpp; docs/design/cluster.md §3.1, §3.3-§3.5,
// §4.2; decisions CL10, CL42, CL43, CL49).
//
// - The squared distance of k-means, fuzzy c-means and their bounds (CL10): spatial's sqeuclidean of a pair, the
//   difference form sum (x_d - c_d)^2 in the tile order with its careful finish below 2^-968 (sqd; the bits of
//   sd.spatial.cdist(metric='sqeuclidean')), on the data scaled by CL43's power of two. Its error (§4.2):
//   |sqd - d^2| <= gamma(p + 3) d^2 + 2^-1074.
// - Bounds on true distances from computed squares (CL42): with gamma = 2 (p + 4) u, dist_up(s) >= d and
//   dist_down(s) <= d for every pair whose sqd is s (§4.2 proves it); a bound may skip a centre only when it clears
//   the skip factor (1 + 2 gamma)(1 + 4u) and kMinSkip.
// - Rows of the arguments: C-contiguous float64 arrays, as the Python layer passes them.
#pragma once

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "cluster/common.hpp"
#include "spatial/metric.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace sd::cluster {

namespace nb = nanobind;

inline constexpr double kInfD = std::numeric_limits<double>::infinity();
inline constexpr double kTinyAbs = 0x1p-1073;  // covers sqd's absolute error near underflow (2^-1074)
inline constexpr double kMinSkip = 0x1p-480;   // a lower bound below it never skips (its square nears underflow)

// spatial's sqeuclidean of one pair (the bits of cdist(metric='sqeuclidean'))
inline double sqd(const double* x, const double* c, int64_t p) {
    double s;
    spatial::pair_scalar<spatial::SqOp, false>(x, c, nullptr, p, &s);
    return spatial::sq_finish<false>(s, x, c, nullptr, p);
}

// sqd of N pairs (x[r], c[r]) side by side: N independent chains, each in pair_scalar's order (blocks of 32 terms,
// each its own chain from 0, the block sums added in block order), so each has sqd's bits. Eight chains keep the
// four FP pipes busy through fma's latency (four chains wait on it).
template <int N>
inline void sqdN(const double* const* x, const double* const* c, int64_t p, double* out) {
    double s[N];
    for (int64_t b = 0; b < p; b += spatial::kBlock) {
        double t[N];
        for (int r = 0; r < N; ++r) t[r] = 0.0;
        const int64_t e = std::min(p, b + spatial::kBlock);
        for (int64_t d = b; d < e; ++d)
            for (int r = 0; r < N; ++r) {
                const double z = x[r][d] - c[r][d];
                t[r] = std::fma(z, z, t[r]);
            }
        for (int r = 0; r < N; ++r) s[r] = b == 0 ? t[r] : s[r] + t[r];
    }
    if (p == 0)
        for (int r = 0; r < N; ++r) s[r] = 0.0;
    for (int r = 0; r < N; ++r) out[r] = spatial::sq_finish<false>(s[r], x[r], c[r], nullptr, p);
}

inline void sqd4(const double* const* x, const double* const* c, int64_t p, double* out) { sqdN<4>(x, c, p, out); }

// sqd of rows i0 .. i1 of X to the centres C[lab[i]] (lab null: to the one centre C): out[i - i0]
inline void sqd_rows(
    const double* X, int64_t p, int64_t i0, int64_t i1, const double* C, const int64_t* lab, double* out
) {
    for (int64_t i = i0; i < i1; i += 8) {
        const int R = static_cast<int>(std::min<int64_t>(8, i1 - i));
        const double* xr[8];
        const double* cr[8];
        for (int r = 0; r < 8; ++r) {
            const int64_t ii = i + std::min(r, R - 1);
            xr[r] = X + ii * p;
            cr[r] = lab ? C + lab[ii] * p : C;
        }
        double v[8];
        sqdN<8>(xr, cr, p, v);
        for (int r = 0; r < R; ++r) out[i - i0 + r] = v[r];
    }
}

// The next double above (below) a value >= 0 by its bits (std::nextafter's result there, without the call);
// infinity stays, and below 0 down gives 0.
inline double next_up(double v) {
    if (!(v < kInfD)) return v;
    if (v < 0.0) return std::nextafter(v, kInfD);
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    ++b;
    std::memcpy(&v, &b, sizeof b);
    return v;
}
inline double next_down(double v) {
    if (!(v > 0.0)) return 0.0;
    if (!(v < kInfD)) return std::numeric_limits<double>::max();
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    --b;
    std::memcpy(&v, &b, sizeof b);
    return v;
}
inline double up2(double v) { return next_up(next_up(v)); }
inline double down2(double v) { return next_down(next_down(v)); }

struct Margin {
    double g;     // gamma = 2 (p + 4) u
    double skip;  // (1 + 2 gamma)(1 + 4u), rounded up
    explicit Margin(int64_t p)
        : g(2.0 * static_cast<double>(p + 4) * kU), skip(up2((1.0 + 2.0 * g) * (1.0 + 4.0 * kU))) {}
    // an upper bound on the true distance of a pair whose sqd is s
    double up(double s) const { return up2(std::sqrt(s + kTinyAbs) * (1.0 + g)); }
    // a lower bound on it
    double down(double s) const { return down2(std::sqrt(std::max(s - kTinyAbs, 0.0)) * (1.0 - g)); }
    // whether a lower bound lo on every other centre's distance proves the own centre (upper bound hi) the
    // nearest by sqd with room for the tie rule: lo > hi (1 + 2 gamma)(1 + 4u) and lo > kMinSkip
    bool clear(double lo, double hi) const { return lo > hi * skip && lo > kMinSkip; }
};

// CL43: the exponent for data whose largest |x| is maxabs, and a scaled copy when it is not 0
inline double max_abs_of(const double* X, int64_t m) {  // eight lanes (a max is exact in any order; NaN skipped)
    float64x2_t a = vdupq_n_f64(0.0), b = a, c = a, d = a;
    int64_t i = 0;
    for (; i + 8 <= m; i += 8) {
        a = vmaxnmq_f64(a, vabsq_f64(vld1q_f64(X + i))), b = vmaxnmq_f64(b, vabsq_f64(vld1q_f64(X + i + 2)));
        c = vmaxnmq_f64(c, vabsq_f64(vld1q_f64(X + i + 4))), d = vmaxnmq_f64(d, vabsq_f64(vld1q_f64(X + i + 6)));
    }
    double mx = vmaxnmvq_f64(vmaxnmq_f64(vmaxnmq_f64(a, b), vmaxnmq_f64(c, d)));
    for (; i < m; ++i) mx = std::max(mx, std::fabs(X[i]));
    return mx;
}

inline std::vector<double> scaled_copy(const double* X, int64_t m, int e) {
    std::vector<double> out(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) out[static_cast<size_t>(i)] = spatial::scaled(X[i], e);
    return out;
}

// ---- the partitions other kernels run (the gap statistic, metrics.cpp) ----------------------------------------

// The labels (key order, the centres' numbers) of sd.cluster.kmeans(X, k) with its defaults (k-means++, one start,
// max_iter 300, tol 1e-4) on rows X already in key order, keyed by the call's two words (kmeans.cpp).
std::vector<int64_t> kmeans_labels(const double* X, int64_t n, int64_t p, int64_t k, random::Key key);
// The labels (key order, medoid slots) of sd.cluster.kmedoids(X, k) (PAM, euclidean, max_iter 1000) on rows in key
// order (kmedoids.cpp).
std::vector<int64_t> pam_labels(const double* X, int64_t n, int64_t p, int64_t k);

// ---- arguments ---------------------------------------------------------------------------------------------------

inline const double* matrix_arg(nb::handle o, const char* what, int64_t& n, int64_t& p) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Float64 || !a.c_contiguous || a.ndim != 2) throw nb::value_error("needs a C (n, p) float64");
    n = a.shape[0];
    p = a.shape[1];
    return reinterpret_cast<const double*>(a.data);
}

inline const double* vector_arg(nb::handle o, const char* what, int64_t& m) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Float64 || !a.c_contiguous || a.ndim != 1) throw nb::value_error("needs a C 1-D float64");
    m = a.shape[0];
    return reinterpret_cast<const double*>(a.data);
}

inline const int64_t* index_arg(nb::handle o, const char* what, int64_t& m) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Int64 || !a.c_contiguous || a.ndim != 1) throw nb::value_error("needs a C 1-D int64");
    m = a.shape[0];
    return reinterpret_cast<const int64_t*>(a.data);
}

inline NDArray float_array(const std::vector<double>& v, int64_t rows, int64_t cols = -1) {
    const int64_t s2[2] = {rows, cols};
    const int64_t s1[1] = {rows};
    NDArray a = cols < 0 ? new_array(DType::Float64, s1, false) : new_array(DType::Float64, s2, false);
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(a.data));
    return a;
}

inline NDArray int_array(const std::vector<int64_t>& v, int64_t rows, int64_t cols = -1) {
    const int64_t s2[2] = {rows, cols};
    const int64_t s1[1] = {rows};
    NDArray a = cols < 0 ? new_array(DType::Int64, s1, false) : new_array(DType::Int64, s2, false);
    std::copy(v.begin(), v.end(), reinterpret_cast<int64_t*>(a.data));
    return a;
}

}  // namespace sd::cluster
