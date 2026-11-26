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

// sqd of four pairs (x[r], c[r]) side by side: four independent chains, each in pair_scalar's order (blocks of 32
// terms, each its own chain from 0, the block sums added in block order), so each has sqd's bits.
inline void sqd4(const double* const* x, const double* const* c, int64_t p, double* out) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int64_t b = 0; b < p; b += spatial::kBlock) {
        double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
        const int64_t e = std::min(p, b + spatial::kBlock);
        for (int64_t d = b; d < e; ++d) {
            const double z0 = x[0][d] - c[0][d], z1 = x[1][d] - c[1][d], z2 = x[2][d] - c[2][d], z3 = x[3][d] - c[3][d];
            t0 = std::fma(z0, z0, t0), t1 = std::fma(z1, z1, t1), t2 = std::fma(z2, z2, t2), t3 = std::fma(z3, z3, t3);
        }
        if (b == 0) s0 = t0, s1 = t1, s2 = t2, s3 = t3;
        else s0 = s0 + t0, s1 = s1 + t1, s2 = s2 + t2, s3 = s3 + t3;
    }
    const double t[4] = {s0, s1, s2, s3};
    for (int r = 0; r < 4; ++r) out[r] = spatial::sq_finish<false>(t[r], x[r], c[r], nullptr, p);
}

// sqd of rows i0 .. i1 of X to the centres C[lab[i]] (lab null: to the one centre C): out[i - i0]
inline void sqd_rows(
    const double* X, int64_t p, int64_t i0, int64_t i1, const double* C, const int64_t* lab, double* out
) {
    for (int64_t i = i0; i < i1; i += 4) {
        const int R = static_cast<int>(std::min<int64_t>(4, i1 - i));
        const double* xr[4];
        const double* cr[4];
        for (int r = 0; r < 4; ++r) {
            const int64_t ii = i + std::min(r, R - 1);
            xr[r] = X + ii * p;
            cr[r] = lab ? C + lab[ii] * p : C;
        }
        double v[4];
        sqd4(xr, cr, p, v);
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
inline double max_abs_of(const double* X, int64_t m) {
    double mx = 0.0;
    for (int64_t i = 0; i < m; ++i) mx = std::max(mx, std::fabs(X[i]));
    return mx;
}

inline std::vector<double> scaled_copy(const double* X, int64_t m, int e) {
    std::vector<double> out(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) out[static_cast<size_t>(i)] = spatial::scaled(X[i], e);
    return out;
}

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
