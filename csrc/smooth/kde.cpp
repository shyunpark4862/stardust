// Kernel density estimation (docs/design/smooth.md §3.5, §3.8, §4.5; decisions SM18-SM22, SM26, SM31,
// SM115-SM126): the sorted data, the weighted moments of the rules, pdf / logpdf / cdf by the direct sum with the
// proven cut, the reflection at bounds, and the binned FFT density.
//
// _core.smooth.kde_prepare(X, w) -> (Xs, ws, sw, wratio)
//     X (n, p) C-contiguous float64 finite, w (n,) >= 0 with a positive value or None. The points of positive
//     weight sorted by (coordinates, weight, index) (decision SM84's rule), their weights (None without), the sum
//     of the weights in that order (pairwise) and w_max / w_min (1 without weights).
// _core.smooth.kde_moments(Xs, ws, sw) -> (mean (p,), cov (p, p), neff)
//     with wn = w / sw: mean = sum wn x, cov = sum wn (x - m)(x - m)^T / (1 - sum wn^2) (numpy's cov with
//     aweights: scipy's gaussian_kde; unweighted the divisor n - 1), neff = 1 / sum wn^2 (n unweighted); pairwise
//     sums in the sorted order.
// _core.smooth.kde_eval(Xs, ws, sw, wratio, kernel, bw, H, Xq, what, lo, hi) -> (m,)
//     the estimator with bw (p,) the kernel's standard deviations (diagonal: the Axis scaling of
//     smooth/gauss_sum.hpp) or, bw None, H (p, p) its covariance; what 0 pdf, 1 logpdf, 2 cdf (1-D); lo, hi the
//     reflection bounds (1-D; -inf, inf for none). A query with a NaN gives NaN. ValueError when H is not
//     positive definite.
// _core.smooth.kde_grid(Xs, ws, sw, kernel, bw, H, lo, hi, n, blo, bhi, max_bytes) -> density or None
//     the binned FFT density (linear binning, the kernel sampled on the grid and cut at kTau standard deviations,
//     a real FFT convolution) at the grid points lo_l + i (hi_l - lo_l) / (n_l - 1), C order (axis 0 slowest),
//     clipped below at 0; None when the work arrays would pass max_bytes. blo, bhi: 1-D reflection bounds.
// _core.smooth.kde_tree_mode(mode) -> the previous mode: 0 the shape's rule, 1 always the scan, 2 the tree
//     whenever it can (tests: the two paths have the same bits)
//
// A query point is one call of the same function whatever task computes it, and the sums run in the sorted order
// with the order of smooth/gauss_sum.hpp, so every value has the same bits for any thread count (decision SM3);
// the tree and the scan keep the same points in the same order (smooth/gauss_tree.hpp), so they have the same bits.

#include "smooth/gauss_sum.hpp"
#include "smooth/gauss_tree.hpp"
#include "smooth/kde.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "fft/plan.hpp"
#include "parallel/pool.hpp"
#include "smooth/smooth.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace sd::smooth {

namespace gauss {
std::atomic<int> g_tree_mode{0};
}

namespace nb = nanobind;

namespace {

using gauss::Axis;
using kde::kInf;
using kde::kNaN;

// The tree path of the direct sum (decision SM140; the rule decides the speed only, both paths have the same bits):
// several variables, the Gaussian kernel, at least kTreeMinPoints points and kTreeMinQueries queries, and a cut
// radius sqrt(T) at most 1/16 of the whitened data's extent on every axis (the kept points are then a small part of
// the data; at Scott's bandwidth they are most of it and the tree was 1.7-4 times slower than the scan).
constexpr std::int64_t kTreeMinPoints = 4096;
constexpr std::int64_t kTreeMinQueries = 16;
constexpr double kTreeExtent = 16.0;

// ---- argument helpers ------------------------------------------------------------------------------------------

const NDArray& vec_arg(nb::handle o, const char* what, std::int64_t n = -1) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 1-D float64 array").c_str());
    if (n >= 0 && a.shape[0] != n) throw nb::value_error(("'" + std::string(what) + "' has the wrong length").c_str());
    return a;
}

const NDArray& mat_arg(nb::handle o, const char* what) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[1] < 1)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 2-D float64 array").c_str());
    return a;
}

const double* dptr(const NDArray& a) { return reinterpret_cast<const double*>(a.data); }
double* dout(NDArray& a) { return reinterpret_cast<double*>(a.data); }

NDArray out1(std::int64_t n) {
    const std::int64_t s[1] = {n};
    return new_array(DType::Float64, s);
}

NDArray out2(std::int64_t n, std::int64_t p) {
    const std::int64_t s[2] = {n, p};
    return new_array(DType::Float64, s);
}

// ---- the model -------------------------------------------------------------------------------------------------

// The estimator over sorted points: the kernel's shape (diagonal through the Axis scaling, or a full covariance H
// through L = chol(s^2 H)^-1 with s a power of two that brings H's diagonal near 1), the normalising constants
// knorm = (2 pi)^(p/2) det A (A = chol H; a compact kernel: h) and norm = sum w knorm kept as m 2^e, and the cut
// T of the direct sum.
struct Model {
    std::int64_t n = 0, p = 1;
    const double* X = nullptr;
    const double* w = nullptr;
    double sw = 1, T = 0;
    int kernel = kde::kGauss;
    double a = kInf;  // a compact kernel's half width (standard deviation 1)
    double h1 = 1;    // 1-D: the bandwidth
    bool full = false;
    std::vector<Axis> ax;    // the diagonal
    std::vector<double> L;   // full: p x p lower triangular, row-major
    double s = 1;            // full: the scale of the differences
    double grow = 1;         // full: the largest absolute row sum of L
    kde::Scaled norm, knorm;
    // the tree (decisions SM115, SM116)
    std::vector<double> Xt;  // p >= 2: the columns (p x n) of the scan
    std::vector<double> mu;
    double xmax = 0;  // full: the largest |(x - mu) s|
    gauss::TreePath tree;
    bool use_tree = false;
};

// Cholesky of the p x p symmetric H (row-major) into the lower A; false unless positive definite.
bool cholesky(const double* H, std::int64_t p, std::vector<double>& A) {
    A.assign(p * p, 0.0);
    for (std::int64_t i = 0; i < p; ++i) {
        for (std::int64_t j = 0; j <= i; ++j) {
            double s = H[i * p + j];
            for (std::int64_t k = 0; k < j; ++k) s -= A[i * p + k] * A[j * p + k];
            if (i == j) {
                if (!(s > 0) || !std::isfinite(s)) return false;
                A[i * p + i] = std::sqrt(s);
            } else {
                A[i * p + j] = s / A[j * p + j];
            }
        }
    }
    return true;
}

// The model of bw (p,) or H (p, p); false when H is not positive definite.
bool make_model(
    Model& M,
    const double* X,
    const double* w,
    std::int64_t n,
    std::int64_t p,
    double sw,
    double wratio,
    int kernel,
    const double* bw,
    const double* H
) {
    M.n = n, M.p = p, M.X = X, M.w = w, M.sw = sw, M.kernel = kernel;
    M.T = gauss::cut_threshold(n, wratio);
    M.a = kde::half_width(kernel);
    M.knorm = kde::Scaled{};
    if (kernel == kde::kGauss)
        for (std::int64_t l = 0; l < p; ++l) M.knorm.mul(kde::kSqrt2Pi);
    if (bw) {
        M.full = false;
        M.ax.resize(p);
        for (std::int64_t l = 0; l < p; ++l) {
            M.ax[l] = gauss::axis_of(bw[l]);
            M.knorm.mul(bw[l]);
        }
        M.h1 = bw[0];
    } else {
        M.full = true;
        double dmax = 0;
        for (std::int64_t l = 0; l < p; ++l) dmax = std::max(dmax, H[l * p + l]);
        if (!(dmax > 0) || !std::isfinite(dmax)) return false;
        int e;
        std::frexp(std::sqrt(dmax), &e);
        M.s = std::ldexp(1.0, -e);  // s^2 H has its largest diagonal entry in [1/4, 1)
        std::vector<double> Hs(p * p), A;
        for (std::int64_t i = 0; i < p * p; ++i) Hs[i] = (H[i] * M.s) * M.s;
        if (!cholesky(Hs.data(), p, A)) return false;
        M.L.assign(p * p, 0.0);  // L = A_s^-1, column by column
        for (std::int64_t c = 0; c < p; ++c) {
            for (std::int64_t i = c; i < p; ++i) {
                double v = i == c ? 1.0 : 0.0;
                for (std::int64_t k = c; k < i; ++k) v -= A[i * p + k] * M.L[k * p + c];
                M.L[i * p + c] = v / A[i * p + i];
            }
        }
        M.grow = 0;
        for (std::int64_t i = 0; i < p; ++i) {
            double r = 0;
            for (std::int64_t k = 0; k <= i; ++k) r += std::fabs(M.L[i * p + k]);
            M.grow = std::max(M.grow, r);
        }
        for (std::int64_t l = 0; l < p; ++l) M.knorm.mul(A[l * p + l]);
        M.knorm.e += e * static_cast<int>(p);  // det A = det A_s / s^p = det A_s 2^(e p)
    }
    M.norm = M.knorm;
    M.norm.mul(sw);
    if (p >= 2) {
        M.Xt.resize(n * p);
        for (std::int64_t j = 0; j < n; ++j)
            for (std::int64_t l = 0; l < p; ++l) M.Xt[l * n + j] = X[j * p + l];
    }
    return true;
}

// The exact squared distance in bandwidth units of the point j and the query x0.
inline double sqdist(const Model& M, std::int64_t j, const double* x0) {
    const double* row = M.X + j * M.p;
    double s = 0;
    if (!M.full) {
        for (std::int64_t l = 0; l < M.p; ++l) {
            const double z = gauss::z_of(row[l], x0[l], M.ax[l]);
            s += z * z;
        }
        return s;
    }
    double dsmall[16];
    std::vector<double> dbig(M.p > 16 ? M.p : 0);
    double* d = M.p > 16 ? dbig.data() : dsmall;
    for (std::int64_t l = 0; l < M.p; ++l) d[l] = (row[l] - x0[l]) * M.s;
    for (std::int64_t i = 0; i < M.p; ++i) {
        double z = 0;
        for (std::int64_t k = 0; k <= i; ++k) z += M.L[i * M.p + k] * d[k];
        s += z * z;
    }
    return s;
}

// sqdist of every point (the same operations in the same order for each point, so the same bits as sqdist; the
// loops over the points vectorize): the columns of Xt (p x n).
// p = 2, 3 in registers, two points at a time: the operations of sqdist (a sum that starts at 0 adds its first
// term exactly, and a square is +0 whatever the sign of a zero, so z_0 = L_00 d_0 and D = z_0^2 + z_1^2 + ... give
// the same bits).
template <int P>
void sqdist_small(const Model& M, const double* x0, double* D) {
    const std::int64_t n = M.n;
    const double* Xt = M.Xt.data();
    float64x2_t t[P], a[P], b[P], L[P][P];
    for (int l = 0; l < P; ++l) {
        t[l] = vdupq_n_f64(x0[l]);
        a[l] = vdupq_n_f64(M.full ? M.s : M.ax[l].sc);
        b[l] = vdupq_n_f64(M.full ? 1.0 : M.ax[l].inv);
        for (int q = 0; q < P; ++q) L[l][q] = vdupq_n_f64(M.full && q <= l ? M.L[l * P + q] : 0.0);
    }
    std::int64_t k = 0;
    const bool full = M.full;
    for (; k + 2 <= n; k += 2) {
        float64x2_t d[P];
        for (int l = 0; l < P; ++l) d[l] = vmulq_f64(vsubq_f64(vld1q_f64(Xt + l * n + k), t[l]), a[l]);
        float64x2_t acc;
        if (!full) {
            float64x2_t z = vmulq_f64(d[0], b[0]);
            acc = vmulq_f64(z, z);
            for (int l = 1; l < P; ++l) {
                z = vmulq_f64(d[l], b[l]);
                acc = vaddq_f64(acc, vmulq_f64(z, z));
            }
        } else {
            float64x2_t z = vmulq_f64(L[0][0], d[0]);
            acc = vmulq_f64(z, z);
            for (int i = 1; i < P; ++i) {
                z = vmulq_f64(L[i][0], d[0]);
                for (int q = 1; q <= i; ++q) z = vaddq_f64(z, vmulq_f64(L[i][q], d[q]));
                acc = vaddq_f64(acc, vmulq_f64(z, z));
            }
        }
        vst1q_f64(D + k, acc);
    }
    for (; k < n; ++k) D[k] = sqdist(M, k, x0);
}

void sqdist_all(const Model& M, const double* x0, double* D) {
    const std::int64_t n = M.n, p = M.p;
    if (p == 2) return sqdist_small<2>(M, x0, D);
    if (p == 3) return sqdist_small<3>(M, x0, D);
    const double* Xt = M.Xt.data();
    constexpr std::int64_t B = 256;
    std::vector<double> dbuf(M.full ? p * B : 0);
    for (std::int64_t j0 = 0; j0 < n; j0 += B) {
        const std::int64_t len = std::min(B, n - j0);
        double* Dj = D + j0;
        for (std::int64_t k = 0; k < len; ++k) Dj[k] = 0.0;
        if (!M.full) {
            for (std::int64_t l = 0; l < p; ++l) {
                const double* col = Xt + l * n + j0;
                const double t = x0[l], sc = M.ax[l].sc, inv = M.ax[l].inv;
                for (std::int64_t k = 0; k < len; ++k) {
                    const double z = ((col[k] - t) * sc) * inv;
                    Dj[k] += z * z;
                }
            }
            continue;
        }
        for (std::int64_t l = 0; l < p; ++l) {
            const double* col = Xt + l * n + j0;
            const double t = x0[l], s = M.s;
            for (std::int64_t k = 0; k < len; ++k) dbuf[l * B + k] = (col[k] - t) * s;
        }
        for (std::int64_t i = 0; i < p; ++i) {
            const double* Li = M.L.data() + i * p;
            double z[B];
            for (std::int64_t k = 0; k < len; ++k) z[k] = 0.0;
            for (std::int64_t q = 0; q <= i; ++q) {
                const double c = Li[q];
                const double* dq = dbuf.data() + q * B;
                for (std::int64_t k = 0; k < len; ++k) z[k] += c * dq[k];
            }
            for (std::int64_t k = 0; k < len; ++k) Dj[k] += z[k] * z[k];
        }
    }
}

// The whitened coordinates about mu that the tree holds (and of a query point).
inline void whiten(const Model& M, const double* x, double* W) {
    if (!M.full) {
        for (std::int64_t l = 0; l < M.p; ++l) W[l] = gauss::z_of(x[l], M.mu[l], M.ax[l]);
        return;
    }
    double dsmall[16];
    std::vector<double> dbig(M.p > 16 ? M.p : 0);
    double* d = M.p > 16 ? dbig.data() : dsmall;
    for (std::int64_t l = 0; l < M.p; ++l) d[l] = (x[l] - M.mu[l]) * M.s;
    for (std::int64_t i = 0; i < M.p; ++i) {
        double z = 0;
        for (std::int64_t k = 0; k <= i; ++k) z += M.L[i * M.p + k] * d[k];
        W[i] = z;
    }
}

// The tree's coordinate bound for a query (decision SM115): 2^-44 times the largest magnitude a whitened coordinate,
// a whitened difference or its rounding can carry (the few roundings of each are below 8 (p + 2) u of it).
inline double tree_eps(const Model& M, const double* x0, const double* W0) {
    if (!M.full) {
        double q = 0;
        for (std::int64_t l = 0; l < M.p; ++l) q = std::max(q, std::fabs(W0[l]));
        return 0x1p-44 * (M.tree.wmax() + q) + 0x1p-1000;
    }
    double q = 0;
    for (std::int64_t l = 0; l < M.p; ++l) q = std::max(q, std::fabs((x0[l] - M.mu[l]) * M.s));
    return 0x1p-44 * static_cast<double>(M.p) * M.grow * (M.xmax + q) + 0x1p-1000;
}

// Builds the tree when the shape asks for it (decision SM116) or the tests force it.
void attach_tree(Model& M, std::int64_t queries) {
    const int mode = gauss::g_tree_mode.load();
    M.use_tree = false;
    if (M.p < 2 || M.kernel != kde::kGauss || M.p > 16 || mode == 1) return;
    if (mode != 2 && (M.n < kTreeMinPoints || queries < kTreeMinQueries)) return;
    const std::int64_t n = M.n, p = M.p;
    M.mu.assign(p, 0.0);
    for (std::int64_t l = 0; l < p; ++l)
        M.mu[l] = gauss::sum_of(n, [&](std::int64_t j) { return M.X[j * p + l]; }) / static_cast<double>(n);
    std::vector<double> W(n * p), lo(p, kInf), hi(p, -kInf);
    M.xmax = 0;
    for (std::int64_t j = 0; j < n; ++j) {
        double* w = W.data() + j * p;
        whiten(M, M.X + j * p, w);
        for (std::int64_t l = 0; l < p; ++l) lo[l] = std::min(lo[l], w[l]), hi[l] = std::max(hi[l], w[l]);
        if (M.full)
            for (std::int64_t l = 0; l < p; ++l) M.xmax = std::max(M.xmax, std::fabs((M.X[j * p + l] - M.mu[l]) * M.s));
    }
    if (mode != 2) {
        const double r = std::sqrt(M.T);
        for (std::int64_t l = 0; l < p; ++l)
            if (!(kTreeExtent * r <= hi[l] - lo[l])) return;
    }
    M.use_tree = M.tree.build(W.data(), n, p);
}

// ---- the direct sum at one query point ----------------------------------------------------------------------------

struct Scratch {
    std::vector<double> D, Dk, wk, e, W0;
    std::vector<std::int64_t> idx;
    gauss::TreeScratch ts;
};

// The sum at x0 as (a, S): the estimator is exp(a) S / norm (a = -D0 / 2 for the Gaussian kernel, 0 otherwise);
// S = 0 (a = -inf) when no point is near enough (or every distance overflows). x0 has no NaN.
struct Part {
    double a = -kInf, S = 0, alo = 0;  // the exponent a + alo (alo: the low part of an exact -D0 / 2)
};

// Knuth's two-sum of a - b: s + e is the exact difference.
inline void two_diff(double a, double b, double& s, double& e) {
    s = a - b;
    const double bb = s - a;
    e = (a - (s - bb)) - (b + bb);
}

Part part_1d(const Model& M, double t, Scratch& S) {
    const std::int64_t n = M.n;
    const Axis ax = M.ax[0];
    if (M.kernel != kde::kGauss) {
        // the compact kernel: the points with |z| < a, a contiguous range of the sorted data
        const auto zl = [&](std::int64_t j) { return gauss::z_of(M.X[j], t, ax); };
        std::int64_t l = 0, r = n;
        while (l < r) {  // first z > -a
            const std::int64_t m = l + (r - l) / 2;
            if (zl(m) > -M.a) r = m;
            else l = m + 1;
        }
        const std::int64_t lo = l;
        r = n;
        while (l < r) {  // first z >= a
            const std::int64_t m = l + (r - l) / 2;
            if (zl(m) < M.a) l = m + 1;
            else r = m;
        }
        const std::int64_t hi = l;
        const double s = gauss::sum_of(hi - lo, [&](std::int64_t k) {
            const std::int64_t j = lo + k;
            const double v = kde::kernel_at(M.kernel, zl(j), M.a);
            return M.w ? v * M.w[j] : v;
        });
        return {0.0, s};
    }
    const std::int64_t c = gauss::nearest_1d(M.X, n, t);
    if (c < 0) return {};
    const double zc0 = gauss::z_of(M.X[c], t, ax);
    if (!(zc0 * zc0 < kInf)) return {};
    std::int64_t lo, hi;
    gauss::range_1d(M.X, n, t, ax, M.T, c, lo, hi);
    // The exponents from exact differences (decision SM117): x - t = s + e exactly (scaled by sc, exact), the
    // nearest point's z = (x_c - t) / h to twice the precision and D0 = z^2 as a pair, and every term's
    // D_j - D0 = (x_j - x_c)(x_j + x_c - 2t) / h^2 from the exact differences: one rounding of the difference and
    // of the sum each, so a term's exponent errs by a few u of D_j - D0 itself, not of D0.
    const double hs = M.h1 * ax.sc;
    double ch, cl;
    two_diff(M.X[c], t, ch, cl);
    ch *= ax.sc, cl *= ax.sc;
    const double zh = ch / hs;
    const double zl = (std::fma(-zh, hs, ch) + cl) / hs;
    const double d0 = zh * zh;
    const double d0lo = std::fma(zh, zh, -d0) + 2.0 * zh * zl;
    const std::int64_t m = hi - lo;
    S.D.resize(m), S.e.resize(m);
    const double* x = M.X + lo;
    const double inv = ax.inv, sc = ax.sc;
    for (std::int64_t k = 0; k < m; ++k) {
        const double s0 = x[k] - t;
        const double bb = s0 - x[k];
        const double e0 = ((x[k] - (s0 - bb)) - (t + bb)) * sc;
        const double s1 = s0 * sc;
        const double d = ((s1 - ch) + (e0 - cl)) * inv;
        const double e = ((s1 + ch) + (e0 + cl)) * inv;
        S.D[k] = d * e;
    }
    gauss::terms(S.D.data(), M.w ? M.w + lo : nullptr, 0.0, m, S.e.data());
    return {-0.5 * d0, gauss::sum(S.e.data(), m), -0.5 * d0lo};
}

// The smallest of D[0, n) (no NaN).
double min_of(const double* D, std::int64_t n) {
    float64x2_t m = vdupq_n_f64(kInf), m1 = m, m2 = m, m3 = m;
    std::int64_t k = 0;
    for (; k + 8 <= n; k += 8) {
        m = vminq_f64(m, vld1q_f64(D + k)), m1 = vminq_f64(m1, vld1q_f64(D + k + 2));
        m2 = vminq_f64(m2, vld1q_f64(D + k + 4)), m3 = vminq_f64(m3, vld1q_f64(D + k + 6));
    }
    m = vminq_f64(vminq_f64(m, m1), vminq_f64(m2, m3));
    for (; k + 2 <= n; k += 2) m = vminq_f64(m, vld1q_f64(D + k));
    double r = std::min(vgetq_lane_f64(m, 0), vgetq_lane_f64(m, 1));
    for (; k < n; ++k) r = D[k] < r ? D[k] : r;
    return r;
}

// e[k] = w[k] exp(-(D[k] - d0) / 2) when D[k] - d0 <= T, else +0 (w null: 1): gauss::terms' operations for a kept
// term (the exponent clipped at -700 for the cut ones, whose value is then dropped), the lanes and the scalar twin
// giving the same bits.
void masked_terms(const double* D, const double* w, double d0, double T, std::int64_t n, double* e) {
    const float64x2_t vd0 = vdupq_n_f64(d0), half = vdupq_n_f64(-0.5), vtop = vdupq_n_f64(d0 + 1400.0),
                      vT = vdupq_n_f64(T), zero = vdupq_n_f64(0.0);
    std::int64_t k = 0;
    for (; k + 2 <= n; k += 2) {
        const float64x2_t d = vld1q_f64(D + k);
        const float64x2_t x = vmulq_f64(vsubq_f64(vminq_f64(d, vtop), vd0), half);
        float64x2_t v = vmath::exp_lanes(x);
        if (w) v = vmulq_f64(v, vld1q_f64(w + k));
        const uint64x2_t keep = vcleq_f64(vsubq_f64(d, vd0), vT);
        vst1q_f64(e + k, vbslq_f64(keep, v, zero));
    }
    for (; k < n; ++k) {
        if (!(D[k] - d0 <= T)) {
            e[k] = 0.0;
            continue;
        }
        const double v = vmath::exp1((D[k] - d0) * -0.5);
        e[k] = w ? v * w[k] : v;
    }
}

// gauss::sum of the n terms that are val[k] at the indices idx[k] (ascending, k in [a, b)) and +0 elsewhere, over
// the index range [lo, hi): the blocks of 64 and their pairwise tree of smooth/gauss_sum.hpp, the four accumulators
// by the position in the block, visiting only the blocks that hold an index (adding +0 changes no bit).
double sparse_sum(
    const std::int64_t* idx, const double* val, std::int64_t a, std::int64_t b, std::int64_t lo, std::int64_t hi
) {
    if (a == b || hi <= lo) return 0.0;
    if (hi - lo <= 64) {
        double acc[4] = {0, 0, 0, 0};
        const std::int64_t full = lo + ((hi - lo) / 4) * 4;
        for (std::int64_t k = a; k < b; ++k) {
            const std::int64_t j = idx[k];
            acc[j < full ? (j - lo) % 4 : 0] += val[k];
        }
        return (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    const std::int64_t nb = (hi - lo + 63) / 64;
    const std::int64_t mid = lo + (nb / 2) * 64;
    const std::int64_t m = std::lower_bound(idx + a, idx + b, mid) - idx;
    return sparse_sum(idx, val, a, m, lo, mid) + sparse_sum(idx, val, m, b, mid, hi);
}

Part part_nd(const Model& M, const double* x0, Scratch& S) {
    const std::int64_t n = M.n, p = M.p;
    bool tree = false;
    if (M.use_tree) {
        S.W0.resize(p);
        whiten(M, x0, S.W0.data());
        tree = M.tree.candidates(
            S.W0.data(), M.T, tree_eps(M, x0, S.W0.data()), -1, [&](std::int64_t j) { return sqdist(M, j, x0); }, S.ts
        );
    }
    const std::int64_t cnt = tree ? static_cast<std::int64_t>(S.ts.cand.size()) : n;
    const std::int64_t* cand = tree ? S.ts.cand.data() : nullptr;
    S.D.resize(cnt);
    if (tree) {
        for (std::int64_t k = 0; k < cnt; ++k) S.D[k] = sqdist(M, cand[k], x0);
    } else {
        sqdist_all(M, x0, S.D.data());
    }
    const double d0 = min_of(S.D.data(), cnt);
    if (!(d0 < kInf)) return {};
    // The sum by the points' indices in the order of smooth/gauss_sum.hpp with the cut terms +0 (decision SM118):
    // adding +0 changes no bit, so the dense sum of every term and the sparse sum of the kept ones have the same
    // bits, and the tree's candidates give the scan's sum. Which of the two runs (a choice of speed only) depends on
    // how many terms are kept: the scan takes the dense sum from 1/16 of them on (decision SM139; the sparse sum's
    // gathering cost more than the exp it saves up to there).
    std::int64_t kept = 0;
    {
        const double T = M.T;
        const double* D = S.D.data();
        const float64x2_t vd0 = vdupq_n_f64(d0), vT = vdupq_n_f64(T);
        uint64x2_t c0 = vdupq_n_u64(0), c1 = c0;
        std::int64_t k = 0;
        for (; k + 4 <= cnt; k += 4) {  // a true lane is all ones: subtracting it counts one
            c0 = vsubq_u64(c0, vcleq_f64(vsubq_f64(vld1q_f64(D + k), vd0), vT));
            c1 = vsubq_u64(c1, vcleq_f64(vsubq_f64(vld1q_f64(D + k + 2), vd0), vT));
        }
        const uint64x2_t c = vaddq_u64(c0, c1);
        kept = static_cast<std::int64_t>(vgetq_lane_u64(c, 0) + vgetq_lane_u64(c, 1));
        for (; k < cnt; ++k) kept += D[k] - d0 <= T ? 1 : 0;
    }
    if (!tree && 16 * kept >= cnt) {
        S.e.resize(cnt);
        masked_terms(S.D.data(), M.w, d0, M.T, cnt, S.e.data());
        return {-0.5 * d0, gauss::sum(S.e.data(), n)};
    }
    S.idx.resize(kept), S.Dk.resize(kept), S.e.resize(kept);
    if (M.w) S.wk.resize(kept);
    std::int64_t m = 0;
    for (std::int64_t k = 0; k < cnt; ++k) {
        if (!(S.D[k] - d0 <= M.T)) continue;
        const std::int64_t j = cand ? cand[k] : k;
        S.idx[m] = j, S.Dk[m] = S.D[k];
        if (M.w) S.wk[m] = M.w[j];
        ++m;
    }
    masked_terms(S.Dk.data(), M.w ? S.wk.data() : nullptr, d0, M.T, kept, S.e.data());
    return {-0.5 * d0, sparse_sum(S.idx.data(), S.e.data(), 0, kept, 0, n)};
}

Part part_at(const Model& M, const double* x0, Scratch& S) {
    return M.p == 1 ? part_1d(M, x0[0], S) : part_nd(M, x0, S);
}

// ---- the distribution function (1-D) --------------------------------------------------------------------------

// Prefix sums of the weights in a fixed pairwise order: the sum of [0, k) adds the aligned blocks of the binary
// decomposition of k left to right (error O(log n) u).
struct Prefix {
    std::vector<std::vector<double>> lv;
    void build(const double* w, std::int64_t n) {
        lv.clear();
        lv.emplace_back(w ? std::vector<double>(w, w + n) : std::vector<double>(n, 1.0));
        while (lv.back().size() > 1) {
            const auto& b = lv.back();
            std::vector<double> u((b.size() + 1) / 2);
            for (std::size_t i = 0; i < u.size(); ++i) u[i] = 2 * i + 1 < b.size() ? b[2 * i] + b[2 * i + 1] : b[2 * i];
            lv.push_back(std::move(u));
        }
    }
    double sum(std::int64_t k) const {
        double acc = 0;
        std::int64_t pos = 0;
        for (std::int64_t lev = static_cast<std::int64_t>(lv.size()) - 1; lev >= 0; --lev) {
            const std::int64_t size = std::int64_t{1} << lev;
            if (pos + size <= k) {
                acc += lv[lev][pos >> lev];
                pos += size;
            }
        }
        return acc;
    }
};

// F(t) of the unbounded estimator: (sum of the weights whose kernel is entirely left of t + sum w G((t - x) / h))
// / sum w, G the kernel's distribution function (0.5 erfc(-z / sqrt 2) for the Gaussian, which rounds to 1 from
// z = 8.5 on and to 0 below z = -38.5).
double cdf_1d(const Model& M, const Prefix& P, double t) {
    if (t != t) return kNaN;
    const std::int64_t n = M.n;
    const Axis ax = M.ax[0];
    const double left = M.kernel == kde::kGauss ? 8.5 : M.a, right = M.kernel == kde::kGauss ? -38.5 : -M.a;
    const auto u = [&](std::int64_t j) { return gauss::z_of(t, M.X[j], ax); };  // decreasing in j
    std::int64_t l = 0, r = n;
    while (l < r) {  // first u < left
        const std::int64_t m = l + (r - l) / 2;
        if (u(m) < left) r = m;
        else l = m + 1;
    }
    const std::int64_t k1 = l;
    r = n;
    while (l < r) {  // first u <= right
        const std::int64_t m = l + (r - l) / 2;
        if (u(m) <= right) r = m;
        else l = m + 1;
    }
    const std::int64_t k2 = l;
    const double mid = gauss::sum_of(k2 - k1, [&](std::int64_t k) {
        const std::int64_t j = k1 + k;
        const double g = kde::kernel_cdf(M.kernel, u(j), M.a);
        return M.w ? g * M.w[j] : g;
    });
    return (P.sum(k1) + mid) / M.sw;
}

// ---- one query point: pdf, logpdf, cdf with the reflection (1-D) ----------------------------------------------

struct Query {
    int what = 0;
    double lo = -kInf, hi = kInf;  // reflection bounds
};

double value_at(const Model& M, const Prefix& P, const Query& q, const double* x0, Scratch& S) {
    for (std::int64_t l = 0; l < M.p; ++l)
        if (x0[l] != x0[l]) return kNaN;
    const bool bounded = M.p == 1 && (q.lo > -kInf || q.hi < kInf);
    if (q.what == 2) {
        const double t = x0[0];
        if (!bounded) return cdf_1d(M, P, t);
        if (t < q.lo) return 0.0;
        const double tt = std::min(t, q.hi);
        double v = cdf_1d(M, P, tt);
        if (q.lo > -kInf) v -= cdf_1d(M, P, 2.0 * q.lo - tt);
        if (q.hi < kInf) v += cdf_1d(M, P, 2.0 * q.hi - q.lo) - cdf_1d(M, P, 2.0 * q.hi - tt);
        return v;
    }
    Part parts[3];
    int np = 0;
    if (bounded) {
        const double t = x0[0];
        if (t < q.lo || t > q.hi) return q.what == 1 ? -kInf : 0.0;
        parts[np++] = part_at(M, x0, S);
        if (q.lo > -kInf) {
            const double r = 2.0 * q.lo - t;
            parts[np++] = part_at(M, &r, S);
        }
        if (q.hi < kInf) {
            const double r = 2.0 * q.hi - t;
            parts[np++] = part_at(M, &r, S);
        }
    } else {
        parts[np++] = part_at(M, x0, S);
    }
    if (q.what == 0) {
        double v = 0;
        for (int i = 0; i < np; ++i)
            v += kde::exp_scaled(parts[i].a, (parts[i].S / M.norm.m) * (1.0 + parts[i].alo), M.norm.e);
        return v;
    }
    double lg[3], best = -kInf;
    for (int i = 0; i < np; ++i) {
        lg[i] = kde::log_scaled(parts[i].a, parts[i].S / M.norm.m, M.norm.e) + parts[i].alo;
        best = std::max(best, lg[i]);
    }
    if (np == 1 || !(best > -kInf) || !(best < kInf)) return np == 1 ? lg[0] : best;
    double s = 0;
    for (int i = 0; i < np; ++i)
        if (lg[i] != best) s += vmath::exp1(lg[i] - best);
    return best + std::log1p(s);
}

// ---- the binned FFT density ----------------------------------------------------------------------------------------

// An n-dimensional real FFT (p <= 3) on the fft area's plans: the real transform along the last axis, the complex
// one along the others; every line is transformed alone, in a fixed order.
struct Fft {
    std::int64_t p = 1;
    std::int64_t N[3] = {1, 1, 1};
    std::shared_ptr<const fft::Plan> real, cplx[2];

    std::int64_t half() const { return N[p - 1] / 2 + 1; }
    std::int64_t reals() const {
        std::int64_t t = 1;
        for (std::int64_t l = 0; l < p; ++l) t *= N[l];
        return t;
    }
    std::int64_t spectrum() const {
        std::int64_t t = half();
        for (std::int64_t l = 0; l + 1 < p; ++l) t *= N[l];
        return t;
    }
    std::int64_t work() const {
        std::int64_t w = real->work_doubles();
        for (std::int64_t l = 0; l + 1 < p; ++l) w = std::max(w, cplx[l]->work_doubles());
        return w;
    }

    // The complex transforms along the axes before the last (C of spectrum() complex values).
    void along(
        double* C, bool inverse, std::vector<double>& buf, std::vector<double>& out, std::vector<double>& wk
    ) const {
        std::int64_t dims[3];
        for (std::int64_t l = 0; l + 1 < p; ++l) dims[l] = N[l];
        dims[p - 1] = half();
        for (std::int64_t ax = 0; ax + 1 < p; ++ax) {
            std::int64_t stride = 1;
            for (std::int64_t l = ax + 1; l < p; ++l) stride *= dims[l];
            const std::int64_t len = dims[ax], outer = spectrum() / (len * stride);
            buf.resize(2 * len), out.resize(2 * len);
            for (std::int64_t o = 0; o < outer; ++o) {
                for (std::int64_t in = 0; in < stride; ++in) {
                    double* base = C + 2 * (o * len * stride + in);
                    for (std::int64_t k = 0; k < len; ++k)
                        buf[2 * k] = base[2 * k * stride], buf[2 * k + 1] = base[2 * k * stride + 1];
                    cplx[ax]->execute(buf.data(), out.data(), wk.data(), inverse);
                    for (std::int64_t k = 0; k < len; ++k)
                        base[2 * k * stride] = out[2 * k], base[2 * k * stride + 1] = out[2 * k + 1];
                }
            }
        }
    }

    void forward(const double* r, double* C) const {
        std::vector<double> wk(work()), buf, out;
        const std::int64_t nl = N[p - 1], lines = reals() / nl;
        for (std::int64_t i = 0; i < lines; ++i) real->r2c(r + i * nl, C + 2 * i * half(), wk.data());
        along(C, false, buf, out, wk);
    }

    void inverse(double* C, double* r) const {
        std::vector<double> wk(work()), buf, out, z(N[p - 1]);
        along(C, true, buf, out, wk);
        const std::int64_t nl = N[p - 1], lines = reals() / nl;
        for (std::int64_t i = 0; i < lines; ++i) real->c2r<false>(C + 2 * i * half(), r + i * nl, z.data(), wk.data());
    }
};

// The grid of the binned density (decisions SM120, SM121): steps delta_l, the kernel's reach in steps L_l (kTau
// standard deviations, or the compact kernel's half width), the padding E_l = L_l + 1 on either side of the
// requested points (a point farther out is beyond the kernel's cut of every output), the padded length G_l and the
// FFT length N_l >= G_l (no wrap reaches an output: the kernel spans 2 L_l + 1 <= G_l).
struct Grid {
    std::int64_t p = 1;
    double lo[3], delta[3];
    std::int64_t n[3], L[3], E[3], G[3];
};

nb::object kde_grid(
    nb::handle xs_obj,
    nb::handle ws_obj,
    double sw,
    int kernel,
    nb::handle bw_obj,
    nb::handle H_obj,
    nb::handle lo_obj,
    nb::handle hi_obj,
    nb::handle n_obj,
    double blo,
    double bhi,
    double max_bytes
) {
    const NDArray& Xa = mat_arg(xs_obj, "Xs");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    if (p > 3) throw nb::value_error("the binned density takes at most 3 variables");
    const NDArray* wa = ws_obj.is_none() ? nullptr : &vec_arg(ws_obj, "ws", n);
    const double* bw = bw_obj.is_none() ? nullptr : dptr(vec_arg(bw_obj, "bw", p));
    const double* H = nullptr;
    if (!bw) {
        const NDArray& Ha = mat_arg(H_obj, "H");
        if (Ha.shape[0] != p || Ha.shape[1] != p) throw nb::value_error("'H' must be (p, p)");
        H = dptr(Ha);
    }
    const double* glo = dptr(vec_arg(lo_obj, "lo", p));
    const double* ghi = dptr(vec_arg(hi_obj, "hi", p));
    const double* gn = dptr(vec_arg(n_obj, "n", p));
    Model M;
    if (!make_model(M, dptr(Xa), wa ? dptr(*wa) : nullptr, n, p, sw, 1.0, kernel, bw, H))
        throw nb::value_error("'H' must be symmetric positive definite");
    Grid g;
    g.p = p;
    double total = 1;
    Fft F;
    F.p = p;
    for (std::int64_t l = 0; l < p; ++l) {
        g.n[l] = static_cast<std::int64_t>(gn[l]);
        g.lo[l] = glo[l];
        g.delta[l] = (ghi[l] - glo[l]) / static_cast<double>(g.n[l] - 1);
        if (!(g.delta[l] > 0) || !std::isfinite(g.delta[l]))
            throw nb::value_error("the grid must have a positive step");
        const double sd = bw ? bw[l] : std::sqrt(H[l * p + l]);
        const double reach = kernel == kde::kGauss ? kde::kTau * sd : M.a * sd;
        const double Ld = std::floor(reach / g.delta[l]);
        if (!(Ld < 0x1p40)) return nb::none();
        g.L[l] = static_cast<std::int64_t>(Ld);
        g.E[l] = g.L[l] + 1;
        g.G[l] = g.n[l] + 2 * g.E[l];
        F.N[l] = kde::good_length(g.G[l], l == p - 1);
        total *= static_cast<double>(F.N[l]);
    }
    if (!(8.0 * total * 5.0 <= max_bytes)) return nb::none();
    for (std::int64_t l = 0; l + 1 < p; ++l) F.cplx[l] = fft::get_plan(fft::Kind::Complex, F.N[l]);
    F.real = fft::get_plan(fft::Kind::Real, F.N[p - 1]);
    std::int64_t outn = 1;
    for (std::int64_t l = 0; l < p; ++l) outn *= g.n[l];
    NDArray out = out1(outn);
    {
        const parallel::Unlocked unlocked(true);
        const std::int64_t R = F.reals(), C = F.spectrum();
        std::vector<double> r(R, 0.0), k(R, 0.0), cs(2 * C), ks(2 * C);
        std::int64_t st[3];  // strides of the padded real array
        st[p - 1] = 1;
        for (std::int64_t l = p - 2; l >= 0; --l) st[l] = st[l + 1] * F.N[l + 1];
        const double* X = M.X;
        const double* w = M.w;
        // the binned weights
        if (p == 1) {
            kde::linbin(X, w, n, g.lo[0], g.delta[0], g.E[0], g.G[0], r.data());
            for (const double b : {blo, bhi}) {
                if (!std::isfinite(b)) continue;
                std::vector<double> xr(n), wr(w ? n : 0), tmp(g.G[0]);
                for (std::int64_t j = 0; j < n; ++j) {
                    xr[j] = 2.0 * b - X[n - 1 - j];  // ascending
                    if (w) wr[j] = w[n - 1 - j];
                }
                kde::linbin(xr.data(), w ? wr.data() : nullptr, n, g.lo[0], g.delta[0], g.E[0], g.G[0], tmp.data());
                for (std::int64_t i = 0; i < g.G[0]; ++i) r[i] += tmp[i];
            }
        } else {
            for (std::int64_t j = 0; j < n; ++j) {
                std::int64_t i0[3];
                double f[3];
                bool in = true;
                for (std::int64_t l = 0; l < p && in; ++l) {
                    const double pos = (X[j * p + l] - g.lo[l]) / g.delta[l] + static_cast<double>(g.E[l]);
                    const double top = static_cast<double>(g.G[l] - 1);
                    if (!(pos >= 0.0 && pos <= top)) {
                        in = false;
                        break;
                    }
                    double fl = std::floor(pos);
                    if (fl >= top) fl = top - 1.0;
                    i0[l] = static_cast<std::int64_t>(fl), f[l] = pos - fl;
                }
                if (!in) continue;
                const double wj = w ? w[j] : 1.0;
                for (std::int64_t corner = 0; corner < (std::int64_t{1} << p); ++corner) {
                    double v = wj;
                    std::int64_t at = 0;
                    for (std::int64_t l = 0; l < p; ++l) {
                        const bool up = (corner >> (p - 1 - l)) & 1;
                        v *= up ? f[l] : 1.0 - f[l];
                        at += (i0[l] + (up ? 1 : 0)) * st[l];
                    }
                    r[at] += v;
                }
            }
        }
        // the kernel on the offsets [-L, L], wrapped
        std::int64_t off[3] = {0, 0, 0};
        std::int64_t cnt = 1;
        for (std::int64_t l = 0; l < p; ++l) cnt *= 2 * g.L[l] + 1;
        const double tau2 = kde::kTau * kde::kTau;
        for (std::int64_t c = 0; c < cnt; ++c) {
            std::int64_t rem = c;
            for (std::int64_t l = p - 1; l >= 0; --l) {
                off[l] = rem % (2 * g.L[l] + 1) - g.L[l];
                rem /= 2 * g.L[l] + 1;
            }
            double val;
            if (kernel != kde::kGauss) {
                const double z = gauss::z_of(static_cast<double>(off[0]) * g.delta[0], 0.0, M.ax[0]);
                const double kv = kde::kernel_at(kernel, z, M.a);
                val = kv > 0 ? kde::exp_scaled(0.0, kv / M.knorm.m, M.knorm.e) : 0.0;
            } else {
                double pt[3], zero[3] = {0, 0, 0};
                for (std::int64_t l = 0; l < p; ++l) pt[l] = static_cast<double>(off[l]) * g.delta[l];
                double q;
                if (!M.full) {
                    q = 0;
                    for (std::int64_t l = 0; l < p; ++l) {
                        const double z = gauss::z_of(pt[l], 0.0, M.ax[l]);
                        q += z * z;
                    }
                } else {
                    const Model& MM = M;
                    double d[3];
                    for (std::int64_t l = 0; l < p; ++l) d[l] = (pt[l] - zero[l]) * MM.s;
                    q = 0;
                    for (std::int64_t i = 0; i < p; ++i) {
                        double z = 0;
                        for (std::int64_t kk = 0; kk <= i; ++kk) z += MM.L[i * p + kk] * d[kk];
                        q += z * z;
                    }
                }
                val = q <= tau2 ? kde::exp_scaled(-0.5 * q, 1.0 / M.knorm.m, M.knorm.e) : 0.0;
            }
            std::int64_t at = 0;
            for (std::int64_t l = 0; l < p; ++l) at += ((off[l] + F.N[l]) % F.N[l]) * st[l];
            k[at] = val;
        }
        F.forward(r.data(), cs.data());
        F.forward(k.data(), ks.data());
        for (std::int64_t i = 0; i < C; ++i) {
            const double ar = cs[2 * i], ai = cs[2 * i + 1], br = ks[2 * i], bi = ks[2 * i + 1];
            cs[2 * i] = ar * br - ai * bi;
            cs[2 * i + 1] = ar * bi + ai * br;
        }
        F.inverse(cs.data(), r.data());
        const double scale = 1.0 / (static_cast<double>(R) * sw);
        double* o = dout(out);
        for (std::int64_t i = 0; i < outn; ++i) {
            std::int64_t rem = i, at = 0;
            for (std::int64_t l = p - 1; l >= 0; --l) {
                const std::int64_t il = rem % g.n[l];
                rem /= g.n[l];
                at += (il + g.E[l]) * st[l];
            }
            double v = r[at] * scale;
            if (!(v > 0)) v = 0.0;
            if (p == 1 && (std::isfinite(blo) || std::isfinite(bhi))) {
                const double x = g.lo[0] + static_cast<double>(i) * g.delta[0];
                if (x < blo || x > bhi) v = 0.0;
            }
            o[i] = v;
        }
    }
    return wrap(std::move(out));
}

// ---- bindings ----------------------------------------------------------------------------------------------------

// The order of the points by (coordinates, weight, index) (decision SM84's rule; -0 equals +0, as a comparison
// sees them): a least-significant-digit radix sort of 11-bit digits of the doubles' order-preserving keys, the
// weight first, then the coordinates from the last to the first; every pass is stable, so equal keys keep the
// index order. O(n) per key instead of a comparison sort's O(n log n) (decision SM125).
void radix_order(const double* X, const double* w, std::int64_t p, std::vector<std::int64_t>& ord) {
    const std::int64_t m = static_cast<std::int64_t>(ord.size());
    if (m < 2) return;
    std::vector<std::int64_t> tmp(m);
    std::vector<std::uint64_t> key(m), ktmp(m);
    const auto key_of = [](double v) {
        if (v == 0.0) v = 0.0;  // -0 as +0
        std::uint64_t b;
        std::memcpy(&b, &v, 8);
        return (b >> 63) ? ~b : (b | (std::uint64_t{1} << 63));
    };
    // From kRadixParallel points on, every step runs on contiguous chunks of the points in parallel: a pass counts
    // the digits of each chunk, gives chunk c of digit d the places after those of digit d in chunks before it, and
    // scatters every chunk in order, so the result is the serial pass's (the stable order is unique), and only the
    // speed depends on the number of chunks.
    constexpr std::int64_t kRadixParallel = std::int64_t{1} << 16;
    constexpr std::uint64_t mask = (std::uint64_t{1} << 11) - 1;
    const bool par = m >= kRadixParallel && parallel::available();
    const std::int64_t C = par ? std::min<std::int64_t>(64, std::max(1, parallel::num_threads())) : 1;
    const auto chunk_lo = [&](std::int64_t c) { return c * m / C; };
    const auto chunks = [&](const auto& fn) {
        if (C == 1) {
            fn(0, m, 0);
            return;
        }
        parallel::parallel_for(C, [&](std::int64_t c) { fn(chunk_lo(c), chunk_lo(c + 1), c); }, work_ns(4 * m / C));
    };
    std::vector<std::int64_t> cnt(static_cast<std::size_t>(C * 2049));
    std::vector<std::uint64_t> ors(C), ands(C);
    const auto pass_key = [&](auto value) {
        chunks([&](std::int64_t lo, std::int64_t hi, std::int64_t c) {
            std::uint64_t o = 0, a = ~std::uint64_t{0};
            for (std::int64_t k = lo; k < hi; ++k) {
                key[k] = key_of(value(ord[k]));
                o |= key[k], a &= key[k];
            }
            ors[c] = o, ands[c] = a;
        });
        std::uint64_t all_or = 0, all_and = ~std::uint64_t{0};
        for (std::int64_t c = 0; c < C; ++c) all_or |= ors[c], all_and &= ands[c];
        const std::uint64_t varying = all_or ^ all_and;  // digits where every key agrees need no pass
        for (int shift = 0; shift < 64; shift += 11) {
            if (((varying >> shift) & mask) == 0) continue;
            std::fill(cnt.begin(), cnt.end(), 0);
            chunks([&](std::int64_t lo, std::int64_t hi, std::int64_t c) {
                std::int64_t* cc = cnt.data() + c * 2049;
                for (std::int64_t k = lo; k < hi; ++k) ++cc[((key[k] >> shift) & mask) + 1];
            });
            // the start of digit d in chunk c: the counts of the smaller digits, then those of d in earlier chunks
            std::int64_t run = 0;
            for (std::int64_t d = 0; d < 2048; ++d) {
                for (std::int64_t c = 0; c < C; ++c) {
                    const std::int64_t v = cnt[c * 2049 + d + 1];
                    cnt[c * 2049 + d] = run;
                    run += v;
                }
            }
            chunks([&](std::int64_t lo, std::int64_t hi, std::int64_t c) {
                std::int64_t* cc = cnt.data() + c * 2049;
                for (std::int64_t k = lo; k < hi; ++k) {
                    const std::int64_t at = cc[(key[k] >> shift) & mask]++;
                    tmp[at] = ord[k];
                    ktmp[at] = key[k];
                }
            });
            ord.swap(tmp);
            key.swap(ktmp);
        }
    };
    if (w) pass_key([&](std::int64_t j) { return w[j]; });
    for (std::int64_t l = p - 1; l >= 0; --l) pass_key([&](std::int64_t j) { return X[j * p + l]; });
}

nb::object kde_prepare(nb::handle X_obj, nb::handle w_obj) {
    const NDArray& Xa = mat_arg(X_obj, "X");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray* wa = w_obj.is_none() ? nullptr : &vec_arg(w_obj, "weights", n);
    const double* X = dptr(Xa);
    const double* w = wa ? dptr(*wa) : nullptr;
    std::vector<std::int64_t> ord;
    ord.reserve(n);
    for (std::int64_t j = 0; j < n; ++j)
        if (!w || w[j] > 0) ord.push_back(j);
    {
        const parallel::Unlocked unlocked(n >= kParallelThreshold);
        radix_order(X, w, p, ord);
    }
    const std::int64_t m = static_cast<std::int64_t>(ord.size());
    NDArray xs = out2(m, p);
    NDArray ws = w ? out1(m) : NDArray{};
    double* xo = dout(xs);
    double* wo = w ? dout(ws) : nullptr;
    {
        const parallel::Unlocked unlocked(m * p >= kParallelThreshold);
        const auto gather = [&](std::int64_t k0, std::int64_t k1) {
            for (std::int64_t k = k0; k < k1; ++k) {
                for (std::int64_t l = 0; l < p; ++l) xo[k * p + l] = X[ord[k] * p + l];
                if (wo) wo[k] = w[ord[k]];
            }
        };
        if (m * (p + 1) >= 4 * kParallelThreshold && parallel::available()) {
            const std::int64_t T = std::min<std::int64_t>(64, std::max(1, parallel::num_threads()));
            parallel::parallel_for(T, [&](std::int64_t t) { gather(t * m / T, (t + 1) * m / T); }, work_ns(4 * m / T));
        } else {
            gather(0, m);
        }
    }
    double sw = static_cast<double>(m), wratio = 1.0;
    if (w) {
        sw = gauss::sum(wo, m);
        const auto [mn, mx] = std::minmax_element(wo, wo + m);
        wratio = *mx / *mn;
    }
    nb::object wobj = w ? wrap(std::move(ws)) : nb::none();
    return nb::make_tuple(wrap(std::move(xs)), wobj, sw, wratio);
}

nb::object kde_moments(nb::handle xs_obj, nb::handle ws_obj, double sw) {
    const NDArray& Xa = mat_arg(xs_obj, "Xs");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray* wa = ws_obj.is_none() ? nullptr : &vec_arg(ws_obj, "ws", n);
    const double* X = dptr(Xa);
    const double* w = wa ? dptr(*wa) : nullptr;
    NDArray mean = out1(p), cov = out2(p, p);
    double neff;
    {
        const parallel::Unlocked unlocked(n * p * p >= kParallelThreshold);
        std::vector<double> wn(n);
        for (std::int64_t j = 0; j < n; ++j) wn[j] = w ? w[j] / sw : 1.0 / static_cast<double>(n);
        double* mo = dout(mean);
        for (std::int64_t l = 0; l < p; ++l) {
            mo[l] = w ? gauss::sum_of(n, [&](std::int64_t j) { return wn[j] * X[j * p + l]; })
                      : gauss::sum_of(n, [&](std::int64_t j) { return X[j * p + l]; }) / static_cast<double>(n);
        }
        double den;
        if (w) {
            const double q = gauss::sum_of(n, [&](std::int64_t j) { return wn[j] * wn[j]; });
            neff = 1.0 / q;
            den = 1.0 - q;
        } else {
            neff = static_cast<double>(n);
            den = static_cast<double>(n - 1);
        }
        double* co = dout(cov);
        for (std::int64_t a = 0; a < p; ++a) {
            for (std::int64_t b = a; b < p; ++b) {
                const double s = gauss::sum_of(n, [&](std::int64_t j) {
                    const double v = (X[j * p + a] - mo[a]) * (X[j * p + b] - mo[b]);
                    return w ? wn[j] * v : v;
                });
                co[a * p + b] = co[b * p + a] = s / den;
            }
        }
    }
    return nb::make_tuple(wrap(std::move(mean)), wrap(std::move(cov)), neff);
}

nb::object kde_eval(
    nb::handle xs_obj,
    nb::handle ws_obj,
    double sw,
    double wratio,
    int kernel,
    nb::handle bw_obj,
    nb::handle H_obj,
    nb::handle q_obj,
    int what,
    double lo,
    double hi
) {
    const NDArray& Xa = mat_arg(xs_obj, "Xs");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray* wa = ws_obj.is_none() ? nullptr : &vec_arg(ws_obj, "ws", n);
    const double* bw = bw_obj.is_none() ? nullptr : dptr(vec_arg(bw_obj, "bw", p));
    const double* H = nullptr;
    if (!bw) {
        const NDArray& Ha = mat_arg(H_obj, "H");
        if (Ha.shape[0] != p || Ha.shape[1] != p) throw nb::value_error("'H' must be (p, p)");
        H = dptr(Ha);
    }
    const NDArray& qa = mat_arg(q_obj, "X_new");
    if (qa.shape[1] != p) throw nb::value_error("'X_new' has the wrong number of columns");
    if (what < 0 || what > 2 || (what == 2 && p != 1)) throw nb::value_error("'what' must be 0, 1 or 2 (1-D)");
    if (kernel < 0 || kernel > 5 || (kernel != kde::kGauss && p != 1)) throw nb::value_error("bad kernel");
    const std::int64_t m = qa.shape[0];
    NDArray out = out1(m);
    Model M;
    if (!make_model(M, dptr(Xa), wa ? dptr(*wa) : nullptr, n, p, sw, wratio, kernel, bw, H))
        throw nb::value_error("'H' must be symmetric positive definite");
    {
        const parallel::Unlocked unlocked(m * n >= kParallelThreshold);
        attach_tree(M, m);
        Prefix P;
        if (what == 2) P.build(M.w, n);
        Query q;
        q.what = what, q.lo = lo, q.hi = hi;
        const double* xq = dptr(qa);
        double* o = dout(out);
        const std::int64_t cost = std::max<std::int64_t>(1, n * p);  // the scan's work per query (the shape alone)
        const std::int64_t work = m * cost;
        if (work >= kParallelThreshold && m > 1 && parallel::available()) {
            // about 2^17 of work per task, but at most 8 tasks per thread: a task sets up its own scratch
            const std::int64_t threads = std::max(1, parallel::num_threads());
            const std::int64_t per = std::max<std::int64_t>(
                std::max<std::int64_t>(1, (std::int64_t{1} << 17) / cost), (m + 8 * threads - 1) / (8 * threads)
            );
            const std::int64_t tasks = (m + per - 1) / per;
            parallel::parallel_for(
                tasks,
                [&](std::int64_t t) {
                    Scratch S;
                    const std::int64_t end = std::min(m, (t + 1) * per);
                    for (std::int64_t i = t * per; i < end; ++i) o[i] = value_at(M, P, q, xq + i * p, S);
                },
                work_ns(per * cost)
            );
        } else {
            Scratch S;
            for (std::int64_t i = 0; i < m; ++i) o[i] = value_at(M, P, q, xq + i * p, S);
        }
    }
    return wrap(std::move(out));
}

int kde_tree_mode(int mode) {
    if (mode < 0 || mode > 2) throw nb::value_error("'mode' must be 0, 1 or 2");
    return gauss::g_tree_mode.exchange(mode);
}

}  // namespace

void bind_kde(nb::module_& m) {
    m.def("kde_prepare", named("kde_prepare", &kde_prepare), nb::arg("X").none(), nb::arg("w").none());
    m.def(
        "kde_moments", named("kde_moments", &kde_moments), nb::arg("Xs").none(), nb::arg("ws").none(), nb::arg("sw")
    );  // fmt: skip
    m.def(
        "kde_eval", named("kde_eval", &kde_eval), nb::arg("Xs").none(), nb::arg("ws").none(), nb::arg("sw"),
        nb::arg("wratio"), nb::arg("kernel"), nb::arg("bw").none(), nb::arg("H").none(), nb::arg("X_new").none(),
        nb::arg("what"), nb::arg("lo"), nb::arg("hi")
    );  // fmt: skip
    m.def(
        "kde_grid", named("kde_grid", &kde_grid), nb::arg("Xs").none(), nb::arg("ws").none(), nb::arg("sw"),
        nb::arg("kernel"), nb::arg("bw").none(), nb::arg("H").none(), nb::arg("lo").none(), nb::arg("hi").none(),
        nb::arg("n").none(), nb::arg("blo"), nb::arg("bhi"), nb::arg("max_bytes")
    );  // fmt: skip
    m.def("kde_tree_mode", &kde_tree_mode, nb::arg("mode"));
}

}  // namespace sd::smooth
