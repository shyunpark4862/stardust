// Gaussian mixtures (docs/design/cluster.md §2.5, §3.7, §3.10, §4.5; decisions CL32-CL35, CL41, CL43 and milestone
// CL4's CL75-CL81 in §1.1): scikit-learn's GaussianMixture EM with its four covariance types, on the rows in key order
// (CL7) scaled by CL43's power of two.
//
//   factors      a covariance S's precision factor U is upper triangular with U U^T = S^-1 (scikit-learn's
//                precisions_cholesky_): the Cholesky factor L of S (fixed order, the lower triangle read), then
//                U = (L^-1)^T by forward substitution. A given precision P is factored directly as U U^T = P (the
//                Cholesky factorization run from the last row up; the upper triangle read). diag and spherical:
//                U = 1 / sqrt(S). A pivot that is not > 0 is an ill-defined covariance.
//   E step       the log density of component j at x in the difference form (sd's quadratic forms, CL10):
//                z = x - mu_j, y = U_j^T z (y_e = sum_{d <= e} U_de z_d, d ascending), q = sum_e y_e^2, and
//                l_j = -0.5 (p log(2 pi) + q) + sum_d log U_dd (scikit-learn's form of the rest), w_j = l_j + log
//                pi_j; m = max_j w_j, e_j = exp(w_j - m), s = sum_j e_j, log p(x) = m + log s and the
//                responsibilities r_j = e_j / s. scikit-learn computes q as ||x U||^2 - ... by products (its diag and
//                spherical forms expand the square), which loses digits far from the means.
//   M step       nk_j = sum_i r_ij + 10 eps (scikit-learn's guard), the means sum_i r_ij x_i / nk_j, then the
//                covariances by a second pass about those means (CL75: sum_i r_ij z z^T / nk_j + reg_covar on the
//                diagonal; tied: summed over the components, over sum_j nk_j; diag: the diagonal; spherical: the
//                mean of the diagonal, reg_covar included), weights nk_j / sum nk (nk_j / n at the start).
//   sums         CL48's chunk sums: every sum over the points is per chunk of 4096 rows in key order, point by
//                point, the chunks' totals met by the pairwise tree in chunk order (each output's tree is fixed by
//                n alone, so the tasks can split by chunk and by component and the bits do not depend on threads).
//   EM           per start: the initial parameters, then until |lower bound - previous| < tol (at most max_iter
//                times): E step (the lower bound is the mean of log p(x)), M step, factors. The best start has the
//                largest lower bound (the first on a tie); its labels are the components of largest w_j at its
//                final parameters (the first on a tie, as predict), its log-likelihood the sum of log p(x) there.
//   kernels      the E step takes eight points a block (their rows as columns), z = x - mu by NEON pairs and the
//                quadratic form by tiles of four columns e x eight points (quad_full); the second moments of full and
//                tied keep a tile of four rows x eight columns in registers through a batch of 128 terms (Moments).
//                Every element's operations are those of the scalar definition in its order, so the bits do not
//                depend on a point's block or the tiling.
//
// _core.cluster:
//   gmm_fit(X, order, k, type, init, init_data, w0, m0, P0, n_init, max_iter, tol, reg, key0, key1)
//       -> (labels in key order (n,), weights, means, covariances, precisions_cholesky (scikit-learn's shapes,
//           the data's units), lower_bound, loglik, n_iter, converged, best_init, e, means, U, log det U, log
//           weights (on the fit's grid: predict's)) or (None, component, start) for an ill-defined covariance
//           (component -1 - j: the given precision of component j is not positive definite)
//       type 0 full, 1 tied, 2 diag, 3 spherical; init 0 labels init_data (n_init, n) one-hot (kmeans), 1 rows
//       init_data (n_init, k) (k-means++'s), 2 'random', 3 'random_from_data', 4 none (w0, m0, P0 all given)
//   gmm_estimate(X, means, U, logdet, logw, type, e) -> (labels, responsibilities (m, k), log p(x)) or None when a
//       row overflows on the fit's grid
//   gmm_precisions(covariances, type, k, p) -> (U, log det U) or None (the factor fails; the tests' entry)
//   gmm_mstep(X, resp, type, reg) -> (weights, means, covariances)   the M step at given responsibilities (tests)
//   gmm_sample(N, weights, perm, means, U, type, e, key0, key1) -> (points (N, p), labels (N,))
//   gmm_subkey(key0, key1, r) -> (key0', key1')   a sub-call's two words at start r of the stream (CL41)
#include "array/kernels/vmath.hpp"
#include "cluster/cluster.hpp"
#include "cluster/partition.hpp"
#include "parallel/pool.hpp"
#include "stats/sample_dist.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <optional>
#include <vector>

namespace sd::cluster {

namespace {

enum Type : int { kFull = 0, kTied = 1, kDiag = 2, kSph = 3 };
constexpr int64_t kTask = 512;         // rows of an E-step task
constexpr double kGuard = 10 * 0x1p-52;  // scikit-learn's 10 eps added to nk

struct Ill {  // a covariance whose factor failed: its component (0 for tied); -1 - j for a given precision
    int64_t comp;
};

int64_t cov_size(int type, int64_t k, int64_t p) {
    switch (type) {
        case kFull: return k * p * p;
        case kTied: return p * p;
        case kDiag: return k * p;
        default: return k;
    }
}

// ---- small dense factors (fixed order) ----------------------------------------------------------------------

// L (lower, row-major p x p) with L L^T = A (A's lower triangle read); false when a pivot is not > 0
bool chol_lower(const double* A, int64_t p, double* L) {
    std::fill(L, L + p * p, 0.0);
    for (int64_t j = 0; j < p; ++j) {
        double s = A[j * p + j];
        for (int64_t t = 0; t < j; ++t) s = std::fma(-L[j * p + t], L[j * p + t], s);
        if (!(s > 0.0) || !std::isfinite(s)) return false;
        const double ljj = std::sqrt(s);
        L[j * p + j] = ljj;
        for (int64_t i = j + 1; i < p; ++i) {
            double v = A[i * p + j];
            for (int64_t t = 0; t < j; ++t) v = std::fma(-L[i * p + t], L[j * p + t], v);
            L[i * p + j] = v / ljj;
        }
    }
    return true;
}

// U = (L^-1)^T (upper, row-major): column c of L^-1 by forward substitution
void inverse_transpose_lower(const double* L, int64_t p, double* U) {
    std::fill(U, U + p * p, 0.0);
    for (int64_t c = 0; c < p; ++c) {
        U[c * p + c] = 1.0 / L[c * p + c];  // (L^-1)_cc
        for (int64_t i = c + 1; i < p; ++i) {
            double s = 0.0;
            for (int64_t t = c; t < i; ++t) s = std::fma(L[i * p + t], U[c * p + t], s);  // U[c][t] = (L^-1)_tc
            U[c * p + i] = -s / L[i * p + i];
        }
    }
}

// U (upper, row-major) with U U^T = P (P's upper triangle read), the factorization from the last row up; false when
// a pivot is not > 0
bool chol_upper_of_precision(const double* P, int64_t p, double* U) {
    std::fill(U, U + p * p, 0.0);
    for (int64_t j = p - 1; j >= 0; --j) {
        double s = P[j * p + j];
        for (int64_t t = j + 1; t < p; ++t) s = std::fma(-U[j * p + t], U[j * p + t], s);
        if (!(s > 0.0) || !std::isfinite(s)) return false;
        const double ujj = std::sqrt(s);
        U[j * p + j] = ujj;
        for (int64_t i = j - 1; i >= 0; --i) {
            double v = P[i * p + j];
            for (int64_t t = j + 1; t < p; ++t) v = std::fma(-U[i * p + t], U[j * p + t], v);
            U[i * p + j] = v / ujj;
        }
    }
    return true;
}

// L = (U^-1)^T (lower, row-major) of an upper U: column c of U^-1 by back substitution
void inverse_transpose_upper(const double* U, int64_t p, double* L) {
    std::fill(L, L + p * p, 0.0);
    for (int64_t c = 0; c < p; ++c) {
        L[c * p + c] = 1.0 / U[c * p + c];
        for (int64_t i = c - 1; i >= 0; --i) {
            double s = 0.0;
            for (int64_t t = i + 1; t <= c; ++t) s = std::fma(U[i * p + t], L[c * p + t], s);  // L[c][t] = (U^-1)_tc
            L[c * p + i] = -s / U[i * p + i];
        }
    }
}

// ---- the model -----------------------------------------------------------------------------------------------

struct Model {
    int type = kFull;
    int64_t k = 0, p = 0;
    std::vector<double> w, mu, cov, U, logdet, logw;

    Model() = default;
    Model(int t, int64_t kk, int64_t pp) : type(t), k(kk), p(pp) {
        w.assign(static_cast<size_t>(k), 0.0);
        mu.assign(static_cast<size_t>(k * p), 0.0);
        cov.assign(static_cast<size_t>(cov_size(t, k, p)), 0.0);
        U.assign(cov.size(), 0.0);
        logdet.assign(static_cast<size_t>(k), 0.0);
        logw.assign(static_cast<size_t>(k), 0.0);
    }
    const double* Uof(int64_t j) const {
        switch (type) {
            case kFull: return U.data() + j * p * p;
            case kTied: return U.data();
            case kDiag: return U.data() + j * p;
            default: return U.data() + j;
        }
    }
    void set_logw() {
        for (int64_t j = 0; j < k; ++j) logw[static_cast<size_t>(j)] = std::log(w[static_cast<size_t>(j)]);
    }
    // log det U per component (scikit-learn's _compute_log_det_cholesky)
    void set_logdet() {
        for (int64_t j = 0; j < k; ++j) {
            const double* u = Uof(j);
            double s = 0.0;
            if (type == kFull || type == kTied)
                for (int64_t d = 0; d < p; ++d) s += std::log(u[d * p + d]);
            else if (type == kDiag)
                for (int64_t d = 0; d < p; ++d) s += std::log(u[d]);
            else s = static_cast<double>(p) * std::log(u[0]);
            logdet[static_cast<size_t>(j)] = s;
        }
    }
    // the factors of the covariances (throws Ill)
    void factor() {
        std::vector<double> L(static_cast<size_t>(p * p));
        if (type == kFull || type == kTied) {
            const int64_t m = type == kFull ? k : 1;
            for (int64_t j = 0; j < m; ++j) {
                if (!chol_lower(cov.data() + j * p * p, p, L.data())) throw Ill{j};
                inverse_transpose_lower(L.data(), p, U.data() + j * p * p);
            }
        } else {
            for (size_t q = 0; q < cov.size(); ++q) {
                if (!(cov[q] > 0.0) || !std::isfinite(cov[q]))
                    throw Ill{static_cast<int64_t>(type == kDiag ? q / p : q)};
                U[q] = 1.0 / std::sqrt(cov[q]);
            }
        }
        set_logdet();
    }
    // the factors from precisions P (scikit-learn's shapes; throws Ill), the covariances as (U U^T)^-1
    void factor_precisions(const double* P) {
        std::vector<double> L(static_cast<size_t>(p * p));
        if (type == kFull || type == kTied) {
            const int64_t m = type == kFull ? k : 1;
            for (int64_t j = 0; j < m; ++j) {
                double* u = U.data() + j * p * p;
                if (!chol_upper_of_precision(P + j * p * p, p, u)) throw Ill{-1 - j};  // a given precision
                inverse_transpose_upper(u, p, L.data());  // S = L L^T
                double* c = cov.data() + j * p * p;
                for (int64_t a = 0; a < p; ++a)
                    for (int64_t b = 0; b <= a; ++b) {
                        double s = 0.0;
                        for (int64_t t = 0; t <= b; ++t) s = std::fma(L[a * p + t], L[b * p + t], s);
                        c[a * p + b] = c[b * p + a] = s;
                    }
            }
        } else {
            for (size_t q = 0; q < cov.size(); ++q) {
                if (!(P[q] > 0.0) || !std::isfinite(P[q]))
                    throw Ill{-1 - static_cast<int64_t>(type == kDiag ? q / p : q)};
                U[q] = std::sqrt(P[q]);
                cov[q] = 1.0 / P[q];
            }
        }
        set_logdet();
    }
};

// ---- E step ----------------------------------------------------------------------------------------------------

// The factors as the E step reads them: full and tied blocks with rows padded to a multiple of 4 (zeros), so a tile
// of four columns never reads past a row.
struct Padded {
    int64_t pw = 0;  // the padded row width
    std::vector<double> U;
    explicit Padded(const Model& M) {
        if (M.type != kFull && M.type != kTied) return;
        const int64_t p = M.p, m = M.type == kFull ? M.k : 1;
        pw = (p + 3) / 4 * 4;
        U.assign(static_cast<size_t>(m * p * pw), 0.0);
        for (int64_t j = 0; j < m; ++j)
            for (int64_t d = 0; d < p; ++d)
                std::copy(M.U.data() + (j * p + d) * p, M.U.data() + (j * p + d + 1) * p, U.data() + (j * p + d) * pw);
    }
};

constexpr int kLanes = 8;  // points of a block: four NEON pairs side by side

// q = ||U^T z||^2 of eight points (z: p x 8, lane l the point l): y_e = sum_{d <= e} U_de z_d by fma in d order, then
// q = sum_e y_e^2 by fma in e order; a lane's operations are those of its point alone (no lane depends on another),
// so a point's bits do not depend on its block. A tile holds four columns e of the eight points; its zeros (d > e,
// and the padding) add exact zeros to y and are left out of q.
inline void quad_full(const double* Up, int64_t pw, int64_t p, const double* z, double* q) {
    float64x2_t q0 = vdupq_n_f64(0.0), q1 = q0, q2 = q0, q3 = q0;
    for (int64_t e0 = 0; e0 < p; e0 += 4) {
        float64x2_t a[4][4];
        for (int t = 0; t < 4; ++t)
            for (int v = 0; v < 4; ++v) a[t][v] = vdupq_n_f64(0.0);
        const int64_t dmax = std::min(p, e0 + 4);
        for (int64_t d = 0; d < dmax; ++d) {
            const double* zd = z + d * kLanes;
            const float64x2_t z0 = vld1q_f64(zd), z1 = vld1q_f64(zd + 2), z2 = vld1q_f64(zd + 4),
                              z3 = vld1q_f64(zd + 6);
            const double* u = Up + d * pw + e0;
            const float64x2_t u01 = vld1q_f64(u), u23 = vld1q_f64(u + 2);
            a[0][0] = vfmaq_laneq_f64(a[0][0], z0, u01, 0), a[0][1] = vfmaq_laneq_f64(a[0][1], z1, u01, 0);
            a[0][2] = vfmaq_laneq_f64(a[0][2], z2, u01, 0), a[0][3] = vfmaq_laneq_f64(a[0][3], z3, u01, 0);
            a[1][0] = vfmaq_laneq_f64(a[1][0], z0, u01, 1), a[1][1] = vfmaq_laneq_f64(a[1][1], z1, u01, 1);
            a[1][2] = vfmaq_laneq_f64(a[1][2], z2, u01, 1), a[1][3] = vfmaq_laneq_f64(a[1][3], z3, u01, 1);
            a[2][0] = vfmaq_laneq_f64(a[2][0], z0, u23, 0), a[2][1] = vfmaq_laneq_f64(a[2][1], z1, u23, 0);
            a[2][2] = vfmaq_laneq_f64(a[2][2], z2, u23, 0), a[2][3] = vfmaq_laneq_f64(a[2][3], z3, u23, 0);
            a[3][0] = vfmaq_laneq_f64(a[3][0], z0, u23, 1), a[3][1] = vfmaq_laneq_f64(a[3][1], z1, u23, 1);
            a[3][2] = vfmaq_laneq_f64(a[3][2], z2, u23, 1), a[3][3] = vfmaq_laneq_f64(a[3][3], z3, u23, 1);
        }
        const int tmax = static_cast<int>(std::min<int64_t>(4, p - e0));
        for (int t = 0; t < tmax; ++t) {
            q0 = vfmaq_f64(q0, a[t][0], a[t][0]), q1 = vfmaq_f64(q1, a[t][1], a[t][1]);
            q2 = vfmaq_f64(q2, a[t][2], a[t][2]), q3 = vfmaq_f64(q3, a[t][3], a[t][3]);
        }
    }
    vst1q_f64(q, q0), vst1q_f64(q + 2, q1), vst1q_f64(q + 4, q2), vst1q_f64(q + 6, q3);
}

// q of eight points for diag (u: p scales) and spherical (one scale): y_d = z_d u_d, q = sum_d y_d^2 by fma
inline void quad_scaled(const double* u, bool one, int64_t p, const double* z, double* q) {
    float64x2_t q0 = vdupq_n_f64(0.0), q1 = q0, q2 = q0, q3 = q0;
    for (int64_t d = 0; d < p; ++d) {
        const double* zd = z + d * kLanes;
        const float64x2_t s = vdupq_n_f64(one ? u[0] : u[d]);
        const float64x2_t y0 = vmulq_f64(vld1q_f64(zd), s), y1 = vmulq_f64(vld1q_f64(zd + 2), s);
        const float64x2_t y2 = vmulq_f64(vld1q_f64(zd + 4), s), y3 = vmulq_f64(vld1q_f64(zd + 6), s);
        q0 = vfmaq_f64(q0, y0, y0), q1 = vfmaq_f64(q1, y1, y1), q2 = vfmaq_f64(q2, y2, y2), q3 = vfmaq_f64(q3, y3, y3);
    }
    vst1q_f64(q, q0), vst1q_f64(q + 2, q1), vst1q_f64(q + 4, q2), vst1q_f64(q + 6, q3);
}

// the weighted log densities W (rows i0 .. i1 of X, row-major (i1 - i0) x k): blocks of eight points (the last one
// repeats its last row)
void weighted_logprob(const Model& M, const Padded& P, const double* X, int64_t i0, int64_t i1, double* W) {
    const int64_t k = M.k, p = M.p;
    const double c = static_cast<double>(p) * std::log(2.0 * std::numbers::pi);
    alignas(16) double q[kLanes];
    std::vector<double> xt(static_cast<size_t>(p * kLanes)), zb(static_cast<size_t>(p * kLanes));
    for (int64_t b = i0; b < i1; b += kLanes) {
        const int R = static_cast<int>(std::min<int64_t>(kLanes, i1 - b));
        for (int l = 0; l < kLanes; ++l) {  // the block's rows as columns (lane l: row b + l)
            const double* x = X + (b + std::min(l, R - 1)) * p;
            for (int64_t d = 0; d < p; ++d) xt[static_cast<size_t>(d * kLanes + l)] = x[d];
        }
        for (int64_t j = 0; j < k; ++j) {
            const double* m = M.mu.data() + j * p;
            for (int64_t d = 0; d < p; ++d) {  // z = x - mu_j, lane by lane
                const float64x2_t md = vdupq_n_f64(m[d]);
                const double* xs = xt.data() + d * kLanes;
                double* zs = zb.data() + d * kLanes;
                vst1q_f64(zs, vsubq_f64(vld1q_f64(xs), md)), vst1q_f64(zs + 2, vsubq_f64(vld1q_f64(xs + 2), md));
                vst1q_f64(zs + 4, vsubq_f64(vld1q_f64(xs + 4), md)),
                    vst1q_f64(zs + 6, vsubq_f64(vld1q_f64(xs + 6), md));
            }
            if (M.type == kFull || M.type == kTied)
                quad_full(P.U.data() + (M.type == kFull ? j : 0) * p * P.pw, P.pw, p, zb.data(), q);
            else quad_scaled(M.Uof(j), M.type == kSph, p, zb.data(), q);
            const double ld = M.logdet[static_cast<size_t>(j)], lw = M.logw[static_cast<size_t>(j)];
            for (int l = 0; l < R; ++l) {
                const double v = -0.5 * (c + q[l]) + ld;
                W[(b - i0 + l) * k + j] = v + lw;
            }
        }
    }
}

// from the weighted log densities of rows (m x k): log p(x) and, when resp is given, the responsibilities
void normalize(const double* W, int64_t m, int64_t k, double* lpn, double* resp) {
    std::vector<double> a(static_cast<size_t>(m * k)), e(static_cast<size_t>(m * k));
    std::vector<double> mx(static_cast<size_t>(m));
    for (int64_t r = 0; r < m; ++r) {
        double v = -kInfD;
        for (int64_t j = 0; j < k; ++j) v = std::max(v, W[r * k + j]);
        mx[static_cast<size_t>(r)] = v;
        const double shift = std::isfinite(v) ? v : 0.0;
        for (int64_t j = 0; j < k; ++j) a[static_cast<size_t>(r * k + j)] = W[r * k + j] - shift;
    }
    vmath::exp(e.data(), a.data(), m * k);
    for (int64_t r = 0; r < m; ++r) {
        const double* er = e.data() + r * k;
        double s = 0.0;
        for (int64_t j = 0; j < k; ++j) s += er[j];
        const double v = mx[static_cast<size_t>(r)];
        if (!(v > -kInfD)) {  // every density underflowed: no responsibility
            lpn[r] = -kInfD;
            if (resp) std::fill(resp + r * k, resp + (r + 1) * k, 0.0);
            continue;
        }
        lpn[r] = v + std::log(s);
        if (resp)
            for (int64_t j = 0; j < k; ++j) resp[r * k + j] = er[j] / s;
    }
}

// The E step on all rows: log p(x) (n,), the responsibilities (n x k) when resp, the labels (largest w_j, the first
// on a tie) when lab
void estep(const Model& M, const double* X, int64_t n, double* lpn, double* resp, int64_t* lab = nullptr) {
    const int64_t k = M.k, p = M.p;
    const Padded P(M);
    const int64_t nt = (n + kTask - 1) / kTask;
    auto body = [&](int64_t t) {
        const int64_t i0 = t * kTask, i1 = std::min(n, (t + 1) * kTask);
        std::vector<double> W(static_cast<size_t>((i1 - i0) * k));
        weighted_logprob(M, P, X, i0, i1, W.data());
        if (lab)
            for (int64_t i = i0; i < i1; ++i) {
                const double* w = W.data() + (i - i0) * k;
                int64_t best = 0;
                for (int64_t j = 1; j < k; ++j)
                    if (w[j] > w[best]) best = j;
                lab[i] = best;
            }
        normalize(W.data(), i1 - i0, k, lpn + i0, resp ? resp + i0 * k : nullptr);
    };
    const int64_t work = M.type == kFull || M.type == kTied ? p * (p + 3) / 2 + 30 : 2 * p + 30;
    if (nt > 1 && n * k * work >= kSplitWork) parallel::parallel_for(nt, body, kTask * k * work);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
}

// ---- chunk sums (CL48) -----------------------------------------------------------------------------------------

// The pairwise tree of CL48 over leaves computed by tasks: leaf(c, g, out) writes group g's `width` doubles of chunk
// c (groups split the outputs: their trees are the same, so the bits are those of one pairwise_chunks), and the
// chunks meet in chunk order. total: groups x width.
template <class Leaf>
void chunk_sums(int64_t n, int64_t groups, int64_t width, Leaf&& leaf, double* total, int64_t leaf_ns) {
    const int64_t nc = chunks_of(n);
    std::vector<double> leaves(static_cast<size_t>(nc * groups * width));
    auto body = [&](int64_t t) {
        const int64_t c = t / groups, g = t % groups;
        leaf(c, g, leaves.data() + (c * groups + g) * width);
    };
    const int64_t nt = nc * groups;
    if (nt > 1 && leaf_ns * nt >= kSplitWork) parallel::parallel_for(nt, body, leaf_ns);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    for (int64_t g = 0; g < groups; ++g) {
        PairwiseStack st(width);
        for (int64_t c = 0; c < nc; ++c) st.push(leaves.data() + (c * groups + g) * width);
        st.finish(total + g * width);
    }
}

// ---- M step ----------------------------------------------------------------------------------------------------

// s[d] = fma(a, x[d], s[d]) for d < p, two lanes at a time (each element its own fma)
inline void axpy(double a, const double* x, double* s, int64_t p) {
    const float64x2_t av = vdupq_n_f64(a);
    int64_t d = 0;
    for (; d + 4 <= p; d += 4) {
        vst1q_f64(s + d, vfmaq_f64(vld1q_f64(s + d), av, vld1q_f64(x + d)));
        vst1q_f64(s + d + 2, vfmaq_f64(vld1q_f64(s + d + 2), av, vld1q_f64(x + d + 2)));
    }
    for (; d < p; ++d) s[d] = std::fma(a, x[d], s[d]);
}

// The centred second moments sum_t r_t z_t z_t^T (z_t = x_t - mu_t) of a sequence of terms, each upper element
// acc_ab = fma(r z_a, z_b, acc_ab) term by term in order (the order CL48 fixes within a chunk). The terms are taken
// in batches; a tile of four rows a and eight columns b stays in registers through a batch, so the order of every
// element's sum is the terms' order whatever the tiling (the elements below the diagonal a tile also computes are
// dropped).
struct Moments {
    static constexpr int64_t kBatch = 128;
    int64_t p, pw, pr, cnt = 0;
    std::vector<double> A, Z, T;  // A: pr x pw; Z, T: kBatch x pw (z and r z, zero padded)
    explicit Moments(int64_t pp) : p(pp), pw((pp + 7) / 8 * 8), pr((pp + 3) / 4 * 4) {
        A.assign(static_cast<size_t>(pr * pw), 0.0);
        Z.assign(static_cast<size_t>(kBatch * pw), 0.0);
        T.assign(static_cast<size_t>(kBatch * pw), 0.0);
    }
    void add(const double* x, const double* m, double r) {
        double* z = Z.data() + cnt * pw;
        double* t = T.data() + cnt * pw;
        for (int64_t d = 0; d < p; ++d) z[d] = x[d] - m[d], t[d] = r * z[d];
        if (++cnt == kBatch) flush();
    }
    void flush() {
        for (int64_t a0 = 0; a0 < p; a0 += 4)
            for (int64_t c0 = a0 / 8 * 8; c0 < p; c0 += 8) {
                float64x2_t acc[4][4];
                for (int r = 0; r < 4; ++r)
                    for (int v = 0; v < 4; ++v) acc[r][v] = vld1q_f64(A.data() + (a0 + r) * pw + c0 + 2 * v);
                for (int64_t q = 0; q < cnt; ++q) {
                    const double* z = Z.data() + q * pw + c0;
                    const double* t = T.data() + q * pw + a0;
                    const float64x2_t z0 = vld1q_f64(z), z1 = vld1q_f64(z + 2), z2 = vld1q_f64(z + 4),
                                      z3 = vld1q_f64(z + 6);
                    const float64x2_t t01 = vld1q_f64(t), t23 = vld1q_f64(t + 2);
                    acc[0][0] = vfmaq_laneq_f64(acc[0][0], z0, t01, 0),
                    acc[0][1] = vfmaq_laneq_f64(acc[0][1], z1, t01, 0);
                    acc[0][2] = vfmaq_laneq_f64(acc[0][2], z2, t01, 0),
                    acc[0][3] = vfmaq_laneq_f64(acc[0][3], z3, t01, 0);
                    acc[1][0] = vfmaq_laneq_f64(acc[1][0], z0, t01, 1),
                    acc[1][1] = vfmaq_laneq_f64(acc[1][1], z1, t01, 1);
                    acc[1][2] = vfmaq_laneq_f64(acc[1][2], z2, t01, 1),
                    acc[1][3] = vfmaq_laneq_f64(acc[1][3], z3, t01, 1);
                    acc[2][0] = vfmaq_laneq_f64(acc[2][0], z0, t23, 0),
                    acc[2][1] = vfmaq_laneq_f64(acc[2][1], z1, t23, 0);
                    acc[2][2] = vfmaq_laneq_f64(acc[2][2], z2, t23, 0),
                    acc[2][3] = vfmaq_laneq_f64(acc[2][3], z3, t23, 0);
                    acc[3][0] = vfmaq_laneq_f64(acc[3][0], z0, t23, 1),
                    acc[3][1] = vfmaq_laneq_f64(acc[3][1], z1, t23, 1);
                    acc[3][2] = vfmaq_laneq_f64(acc[3][2], z2, t23, 1),
                    acc[3][3] = vfmaq_laneq_f64(acc[3][3], z3, t23, 1);
                }
                for (int r = 0; r < 4; ++r)
                    for (int v = 0; v < 4; ++v) vst1q_f64(A.data() + (a0 + r) * pw + c0 + 2 * v, acc[r][v]);
            }
        cnt = 0;
    }
    // the upper triangle (a <= b, row by row) into out
    void finish(double* out) {
        if (cnt) flush();
        for (int64_t a = 0; a < p; ++a)
            for (int64_t b = a; b < p; ++b) *out++ = A[static_cast<size_t>(a * pw + b)];
    }
};

// The M step from responsibilities R (n x k, rows of X): nk, means, covariances into M (weights nk / sum nk, or
// nk / n when by_n); lpn given: its sum (the lower bound's numerator) into *lpn_sum.
void mstep(
    Model& M,
    const double* X,
    int64_t n,
    const double* R,
    double reg,
    bool by_n,
    const double* lpn = nullptr,
    double* lpn_sum = nullptr
) {
    const int64_t k = M.k, p = M.p;
    // pass 1: [sum lpn, nk_j, sum r x] per chunk
    const int64_t w1 = 1 + k + k * p;
    std::vector<double> t1(static_cast<size_t>(w1));
    chunk_sums(
        n, 1, w1,
        [&](int64_t c, int64_t, double* out) {
            std::fill(out, out + w1, 0.0);
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) {
                if (lpn) out[0] += lpn[i];
                const double* r = R + i * k;
                const double* x = X + i * p;
                for (int64_t j = 0; j < k; ++j) {
                    const double rj = r[j];
                    out[1 + j] += rj;
                    axpy(rj, x, out + 1 + k + j * p, p);
                }
            }
        },
        t1.data(), kChunk * k * (p + 2)
    );
    if (lpn_sum) *lpn_sum = t1[0];
    std::vector<double> nk(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
        nk[static_cast<size_t>(j)] = t1[static_cast<size_t>(1 + j)] + kGuard;
        for (int64_t d = 0; d < p; ++d)
            M.mu[static_cast<size_t>(j * p + d)] =
                t1[static_cast<size_t>(1 + k + j * p + d)] / nk[static_cast<size_t>(j)];
    }
    double nsum = 0.0;
    for (int64_t j = 0; j < k; ++j) nsum += nk[static_cast<size_t>(j)];
    const double denom = by_n ? static_cast<double>(n) : nsum;
    for (int64_t j = 0; j < k; ++j) M.w[static_cast<size_t>(j)] = nk[static_cast<size_t>(j)] / denom;
    // pass 2: the centred second moments
    const bool tri = M.type == kFull || M.type == kTied;
    const int64_t wd = tri ? p * (p + 1) / 2 : p;  // per component (tied: one block)
    const int64_t groups = M.type == kTied ? 1 : k;
    std::vector<double> t2(static_cast<size_t>(groups * wd));
    chunk_sums(
        n, groups, wd,
        [&](int64_t c, int64_t g, double* out) {
            std::fill(out, out + wd, 0.0);
            const int64_t j0 = M.type == kTied ? 0 : g, j1 = M.type == kTied ? k : g + 1;
            const int64_t i1 = std::min(n, (c + 1) * kChunk);
            if (tri) {
                Moments S(p);
                for (int64_t i = c * kChunk; i < i1; ++i)
                    for (int64_t j = j0; j < j1; ++j) {
                        const double r = R[i * k + j];
                        if (r != 0.0) S.add(X + i * p, M.mu.data() + j * p, r);  // r = 0 adds +0 exactly
                    }
                S.finish(out);
            } else {
                std::vector<double> z(static_cast<size_t>(p));
                for (int64_t i = c * kChunk; i < i1; ++i) {
                    const double* x = X + i * p;
                    const double r = R[i * k + g];
                    if (r == 0.0) continue;
                    const double* m = M.mu.data() + g * p;
                    for (int64_t d = 0; d < p; ++d) {
                        const double zd = x[d] - m[d];
                        out[d] = std::fma(r * zd, zd, out[d]);
                    }
                }
            }
        },
        t2.data(), kChunk * (M.type == kTied ? k : 1) * (wd + 2 * p)
    );
    if (tri) {
        const int64_t m = M.type == kFull ? k : 1;
        for (int64_t j = 0; j < m; ++j) {
            const double den = M.type == kFull ? nk[static_cast<size_t>(j)] : nsum;
            const double* s = t2.data() + j * wd;
            double* cv = M.cov.data() + j * p * p;
            for (int64_t a = 0; a < p; ++a)
                for (int64_t b = a; b < p; ++b, ++s) {
                    double v = *s / den;
                    if (a == b) v = v + reg;
                    cv[a * p + b] = cv[b * p + a] = v;
                }
        }
    } else if (M.type == kDiag) {
        for (int64_t j = 0; j < k; ++j)
            for (int64_t d = 0; d < p; ++d)
                M.cov[static_cast<size_t>(j * p + d)] =
                    t2[static_cast<size_t>(j * p + d)] / nk[static_cast<size_t>(j)] + reg;
    } else {
        for (int64_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (int64_t d = 0; d < p; ++d) s += t2[static_cast<size_t>(j * p + d)] / nk[static_cast<size_t>(j)] + reg;
            M.cov[static_cast<size_t>(j)] = s / static_cast<double>(p);
        }
    }
}

// the sum of v[0 .. n) by CL48's chunk sums
double chunk_total(const double* v, int64_t n) {
    double t = 0.0;
    chunk_sums(
        n, 1, 1,
        [&](int64_t c, int64_t, double* out) {
            double s = 0.0;
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) s += v[i];
            out[0] = s;
        },
        &t, kChunk
    );
    return t;
}

// ---- starts (CL41, CL76) -----------------------------------------------------------------------------------------

// (m + 1/2) 2^-53 from a word: in (0, 1)
inline double open_unit(std::uint64_t w) { return (static_cast<double>(w >> 11) + 0.5) * 0x1p-53; }

// The initial responsibilities of start s (n x k): init 0 one-hot of labels, 1 one-hot rows, 2 'random' (uniforms
// u_ij from element s 2^40 + i k + j, each row divided by its sum), 3 'random_from_data' (k rows drawn uniformly
// without replacement: the k largest uniforms of element s 2^40 + i, the j-th largest component j's)
void initial_resp(
    int init, int64_t s, const int64_t* init_data, int64_t n, int64_t k, random::Key key, std::vector<double>& R
) {
    R.assign(static_cast<size_t>(n * k), 0.0);
    if (init == 0) {
        const int64_t* lab = init_data + s * n;
        for (int64_t i = 0; i < n; ++i) R[static_cast<size_t>(i * k + lab[i])] = 1.0;
    } else if (init == 1) {
        const int64_t* rows = init_data + s * k;
        for (int64_t j = 0; j < k; ++j) R[static_cast<size_t>(rows[j] * k + j)] = 1.0;
    } else if (init == 2) {
        const int64_t nt = (n + kTask - 1) / kTask;
        auto body = [&](int64_t t) {
            random::Stream st(key);
            for (int64_t i = t * kTask; i < std::min(n, (t + 1) * kTask); ++i) {
                double* r = R.data() + i * k;
                double sum = 0.0;
                for (int64_t j = 0; j < k; ++j) {
                    r[j] = open_unit(
                        st.first(element(static_cast<std::uint64_t>(s), static_cast<std::uint64_t>(i * k + j)))
                    );
                    sum += r[j];
                }
                for (int64_t j = 0; j < k; ++j) r[j] /= sum;
            }
        };
        if (nt > 1 && n * k >= kSplitWork / 8) parallel::parallel_for(nt, body, kTask * k * 8);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    } else {
        random::Stream st(key);
        std::vector<double> u(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            u[static_cast<size_t>(i)] =
                open_unit(st.first(element(static_cast<std::uint64_t>(s), static_cast<std::uint64_t>(i))));
        std::vector<int64_t> idx(static_cast<size_t>(n));
        std::iota(idx.begin(), idx.end(), int64_t{0});
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t a, int64_t b) {
            const double x = u[static_cast<size_t>(a)], y = u[static_cast<size_t>(b)];
            return x != y ? x > y : a < b;
        });
        for (int64_t j = 0; j < k; ++j) R[static_cast<size_t>(idx[static_cast<size_t>(j)] * k + j)] = 1.0;
    }
}

struct Run {
    Model M;
    double lb = -kInfD;
    int64_t n_iter = 0;
    bool converged = false;
};

// ---- arguments -----------------------------------------------------------------------------------------------

// an array of any rank (C, float64): its data and element count
const double* any_float(nb::handle o, const char* what, int64_t& count) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Float64 || !a.c_contiguous) throw nb::value_error("needs a C float64 array");
    count = 1;
    for (int d = 0; d < a.ndim; ++d) count *= a.shape[d];
    return reinterpret_cast<const double*>(a.data);
}

NDArray shaped(int type, int64_t k, int64_t p, const std::vector<double>& v) {
    switch (type) {
        case kFull: {
            const int64_t s[3] = {k, p, p};
            NDArray a = new_array(DType::Float64, s, false);
            std::copy(v.begin(), v.end(), reinterpret_cast<double*>(a.data));
            return a;
        }
        case kTied: return float_array(v, p, p);
        case kDiag: return float_array(v, k, p);
        default: return float_array(v, k);
    }
}

// ---- gmm_fit -------------------------------------------------------------------------------------------------

nb::object fit_py(
    nb::handle Xo,
    nb::handle ordo,
    int64_t k,
    int type,
    int init,
    nb::handle init_o,
    nb::handle w0o,
    nb::handle m0o,
    nb::handle P0o,
    int64_t n_init,
    int64_t max_iter,
    double tol,
    double reg,
    int64_t key0,
    int64_t key1
) {
    int64_t n, p, no;
    const double* X0 = matrix_arg(Xo, "X", n, p);
    const int64_t* order = index_arg(ordo, "order", no);
    if (no != n || k < 1 || k > n || n_init < 1 || max_iter < 0 || type < 0 || type > 3) throw nb::value_error("gmm");
    const int64_t* init_data = nullptr;
    if (init == 0 || init == 1) {
        int64_t rows, cols;
        NDArray& a = require_ndarray(init_o, "init_data");
        if (a.dtype != DType::Int64 || !a.c_contiguous || a.ndim != 2) throw nb::value_error("init_data");
        rows = a.shape[0], cols = a.shape[1];
        if (rows != n_init || cols != (init == 0 ? n : k)) throw nb::value_error("init_data shape");
        init_data = reinterpret_cast<const int64_t*>(a.data);
    }
    const double *w0 = nullptr, *m0 = nullptr, *P0 = nullptr;
    int64_t cnt;
    if (!w0o.is_none()) w0 = vector_arg(w0o, "w0", cnt);
    if (!m0o.is_none()) {
        int64_t kk, pp;
        m0 = matrix_arg(m0o, "m0", kk, pp);
        if (kk != k || pp != p) throw nb::value_error("m0");
    }
    if (!P0o.is_none()) {
        P0 = any_float(P0o, "P0", cnt);
        if (cnt != cov_size(type, k, p)) throw nb::value_error("P0");
    }
    if (init == 4 && !(w0 && m0 && P0)) throw nb::value_error("init");
    Run best;
    int64_t best_init = 0;
    std::vector<int64_t> lab(static_cast<size_t>(n));
    double loglik = 0.0;
    int e = 0;
    std::optional<std::pair<int64_t, int64_t>> ill;
    {
        const parallel::Unlocked unlocked(n * p >= kSplitWork);
        // CL43 (CL77): the grid of the data, the means given and sqrt(reg_covar)
        double mx = max_abs_of(X0, n * p);
        if (m0) mx = std::max(mx, max_abs_of(m0, k * p));
        if (reg > 0.0) mx = std::max(mx, std::sqrt(reg));
        e = scale_exponent(mx);
        std::vector<double> Xs(static_cast<size_t>(n * p));
        for (int64_t r = 0; r < n; ++r)
            for (int64_t d = 0; d < p; ++d) {
                const double v = X0[order[r] * p + d];
                Xs[static_cast<size_t>(r * p + d)] = e ? spatial::scaled(v, e) : v;
            }
        const double* X = Xs.data();
        const double regs = e ? std::ldexp(reg, 2 * e) : reg;
        std::vector<double> P0s;
        if (P0) {
            P0s.assign(P0, P0 + cov_size(type, k, p));
            if (e)
                for (auto& v : P0s) v = std::ldexp(v, -2 * e);
        }
        const random::Key key = key_of(key0, key1);
        std::vector<double> R(static_cast<size_t>(n * k)), lpn(static_cast<size_t>(n));
        bool have = false;
        int64_t s = 0;
        try {
            for (s = 0; s < n_init; ++s) {
                Run run;
                run.M = Model(type, k, p);
                Model& M = run.M;
                if (init != 4) {
                    initial_resp(init, s, init_data, n, k, key, R);
                    mstep(M, X, n, R.data(), regs, true);
                }
                if (w0) std::copy(w0, w0 + k, M.w.begin());
                if (m0)
                    for (int64_t q = 0; q < k * p; ++q)
                        M.mu[static_cast<size_t>(q)] = e ? spatial::scaled(m0[q], e) : m0[q];
                if (P0) M.factor_precisions(P0s.data());
                else M.factor();
                M.set_logw();
                double lb = -kInfD;
                for (int64_t it = 1; it <= max_iter; ++it) {
                    const double prev = lb;
                    estep(M, X, n, lpn.data(), R.data());
                    double total = 0.0;
                    mstep(M, X, n, R.data(), regs, false, lpn.data(), &total);
                    lb = total / static_cast<double>(n);
                    M.factor();
                    M.set_logw();
                    run.n_iter = it;
                    if (std::fabs(lb - prev) < tol) {
                        run.converged = true;
                        break;
                    }
                }
                run.lb = lb;
                if (!have || run.lb > best.lb) {
                    best = std::move(run);
                    best_init = s;
                    have = true;
                }
            }
            estep(best.M, X, n, lpn.data(), nullptr, lab.data());
            loglik = chunk_total(lpn.data(), n);
        } catch (const Ill& bad) { ill = std::make_pair(bad.comp, s); }
    }
    if (ill) return nb::make_tuple(nb::none(), ill->first, ill->second);
    const Model& M = best.M;
    // the data's units (CL43): log p(x 2^e) = log p(x) - p e log 2
    const double shift = static_cast<double>(p) * static_cast<double>(e) * std::numbers::ln2;
    std::vector<double> mu = M.mu, cv = M.cov, U = M.U;
    if (e) {
        for (auto& v : mu) v = spatial::scaled(v, -e);
        for (auto& v : cv) v = std::ldexp(v, -2 * e);
        for (auto& v : U) v = spatial::scaled(v, e);
    }
    const double lb = e ? best.lb + shift : best.lb;
    const double ll = e ? loglik + static_cast<double>(n) * shift : loglik;
    return nb::make_tuple(
        wrap(int_array(lab, n)), wrap(float_array(M.w, k)), wrap(float_array(mu, k, p)), wrap(shaped(type, k, p, cv)),
        wrap(shaped(type, k, p, U)), lb, ll, best.n_iter, best.converged, best_init, e, wrap(float_array(M.mu, k, p)),
        wrap(shaped(type, k, p, M.U)), wrap(float_array(M.logdet, k)), wrap(float_array(M.logw, k))
    );
}

// the model of a fit on its grid from the arrays gmm_fit returned
Model model_of(nb::handle mo, nb::handle Uo, nb::handle ldo, nb::handle lwo, int type) {
    int64_t k, p, cnt;
    const double* mu = matrix_arg(mo, "means", k, p);
    Model M(type, k, p);
    std::copy(mu, mu + k * p, M.mu.begin());
    const double* U = any_float(Uo, "U", cnt);
    if (cnt != cov_size(type, k, p)) throw nb::value_error("U");
    std::copy(U, U + cnt, M.U.begin());
    const double* ld = vector_arg(ldo, "logdet", cnt);
    if (cnt != k) throw nb::value_error("logdet");
    std::copy(ld, ld + k, M.logdet.begin());
    const double* lw = vector_arg(lwo, "logw", cnt);
    if (cnt != k) throw nb::value_error("logw");
    std::copy(lw, lw + k, M.logw.begin());
    return M;
}

nb::object estimate_py(nb::handle Xo, nb::handle mo, nb::handle Uo, nb::handle ldo, nb::handle lwo, int type, int e) {
    const Model M = model_of(mo, Uo, ldo, lwo, type);
    int64_t m, p;
    const double* X0 = matrix_arg(Xo, "X", m, p);
    if (p != M.p) throw nb::value_error("X");
    const int64_t k = M.k;
    std::vector<int64_t> lab(static_cast<size_t>(m));
    std::vector<double> resp(static_cast<size_t>(m * k)), lpn(static_cast<size_t>(m));
    bool overflow = false;
    {
        const parallel::Unlocked unlocked(m * p * k >= kSplitWork);
        std::vector<double> Xs;
        const double* X = X0;
        if (e) {
            Xs = scaled_copy(X0, m * p, e);
            for (double v : Xs) overflow = overflow || !std::isfinite(v);
            X = Xs.data();
        }
        if (!overflow) {
            estep(M, X, m, lpn.data(), resp.data(), lab.data());
            if (e) {
                const double shift = static_cast<double>(p) * static_cast<double>(e) * std::numbers::ln2;
                for (auto& v : lpn) v += shift;
            }
        }
    }
    if (overflow) return nb::none();
    return nb::make_tuple(wrap(int_array(lab, m)), wrap(float_array(resp, m, k)), wrap(float_array(lpn, m)));
}

// (U, log det U) of covariances in scikit-learn's shapes (k from the array; tied: k = 1), or None
nb::object precisions_py(nb::handle co, int type, int64_t k, int64_t p) {
    int64_t cnt;
    const double* C = any_float(co, "covariances", cnt);
    if (cnt != cov_size(type, k, p)) throw nb::value_error("covariances");
    Model M(type, k, p);
    std::copy(C, C + cnt, M.cov.begin());
    try {
        M.factor();
    } catch (const Ill&) { return nb::none(); }
    return nb::make_tuple(wrap(shaped(type, k, p, M.U)), wrap(float_array(M.logdet, k)));
}

nb::object mstep_py(nb::handle Xo, nb::handle Ro, int type, double reg) {
    int64_t n, p, nr, k;
    const double* X = matrix_arg(Xo, "X", n, p);
    const double* R = matrix_arg(Ro, "resp", nr, k);
    if (nr != n) throw nb::value_error("resp");
    Model M(type, k, p);
    {
        const parallel::Unlocked unlocked(n * p * k >= kSplitWork);
        mstep(M, X, n, R, reg, false);
    }
    return nb::make_tuple(wrap(float_array(M.w, k)), wrap(float_array(M.mu, k, p)), wrap(shaped(type, k, p, M.cov)));
}

// Samples (CL80): sample t is element t of the call's stream: its first word picks the component (the first label l
// whose running sum of the weights in label order exceeds u times the total), the next words p standard normals z
// (sd.stats' ziggurat, stats/sample_dist.hpp), and the point is mu + L z with L = (U^-1)^T (on the fit's grid, then
// 2^-e).
nb::object sample_py(
    int64_t N,
    nb::handle wo,
    nb::handle permo,
    nb::handle mo,
    nb::handle Uo,
    int type,
    int e,
    int64_t key0,
    int64_t key1
) {
    int64_t k, p, cnt, kp;
    const double* w = vector_arg(wo, "weights", k);
    const int64_t* perm = index_arg(permo, "perm", kp);
    const double* mu = matrix_arg(mo, "means", cnt, p);
    if (kp != k || cnt != k || N < 0) throw nb::value_error("sample");
    const double* U = any_float(Uo, "U", cnt);
    if (cnt != cov_size(type, k, p)) throw nb::value_error("U");
    std::vector<double> X(static_cast<size_t>(N * p));
    std::vector<int64_t> lab(static_cast<size_t>(N));
    {
        const parallel::Unlocked unlocked(N * p >= kSplitWork);
        // L per component (full: k blocks; tied: one; diag: 1 / U; spherical: 1 / U)
        const int64_t nL = type == kFull ? k : 1;
        std::vector<double> L;
        if (type == kFull || type == kTied) {
            L.resize(static_cast<size_t>(nL * p * p));
            for (int64_t j = 0; j < nL; ++j) inverse_transpose_upper(U + j * p * p, p, L.data() + j * p * p);
        } else {
            L.resize(static_cast<size_t>(cnt));
            for (int64_t q = 0; q < cnt; ++q) L[static_cast<size_t>(q)] = 1.0 / U[q];
        }
        std::vector<double> cum(static_cast<size_t>(k));
        double s = 0.0;
        for (int64_t l = 0; l < k; ++l) cum[static_cast<size_t>(l)] = s = s + w[l];
        const random::Key key = key_of(key0, key1);
        constexpr int64_t kPer = 1024;
        const int64_t nt = (N + kPer - 1) / kPer;
        auto body = [&](int64_t t) {
            random::Stream st(key);
            std::vector<double> z(static_cast<size_t>(p));
            for (int64_t i = t * kPer; i < std::min(N, (t + 1) * kPer); ++i) {
                stats::smp::Words wd(st, static_cast<std::uint64_t>(i));
                const double target = wd.u01() * s;
                int64_t l = std::upper_bound(cum.begin(), cum.end(), target) - cum.begin();
                if (l >= k) {  // rounding: the last label of positive weight
                    l = k - 1;
                    while (l > 0 && !(w[l] > 0.0)) --l;
                }
                const int64_t j = perm[l];
                for (int64_t d = 0; d < p; ++d) z[static_cast<size_t>(d)] = stats::smp::std_normal(wd);
                double* x = X.data() + i * p;
                const double* m = mu + j * p;
                if (type == kFull || type == kTied) {
                    const double* Lj = L.data() + (type == kFull ? j : 0) * p * p;
                    for (int64_t a = 0; a < p; ++a) {
                        double v = 0.0;
                        for (int64_t b = 0; b <= a; ++b) v = std::fma(Lj[a * p + b], z[static_cast<size_t>(b)], v);
                        x[a] = m[a] + v;
                    }
                } else {
                    for (int64_t a = 0; a < p; ++a) {
                        const double sc = type == kDiag ? L[static_cast<size_t>(j * p + a)] : L[static_cast<size_t>(j)];
                        x[a] = std::fma(sc, z[static_cast<size_t>(a)], m[a]);
                    }
                }
                if (e)
                    for (int64_t a = 0; a < p; ++a) x[a] = spatial::scaled(x[a], -e);
                lab[static_cast<size_t>(i)] = l;
            }
        };
        if (nt > 1 && N * p >= kSplitWork / 8) parallel::parallel_for(nt, body, kPer * p * 8);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    }
    return nb::make_tuple(wrap(float_array(X, N, p)), wrap(int_array(lab, N)));
}

// a sub-call's key (CL41): the first words of elements r 2^40 and r 2^40 + 1, as the int64 words key() hands out
nb::object subkey_py(int64_t key0, int64_t key1, int64_t r) {
    random::Stream st(key_of(key0, key1));
    const std::uint64_t a = st.first(element(static_cast<std::uint64_t>(r), 0));
    const std::uint64_t b = st.first(element(static_cast<std::uint64_t>(r), 1));
    return nb::make_tuple(static_cast<int64_t>(a), static_cast<int64_t>(b));
}

}  // namespace

void bind_mixture(nb::module_& m) {
    m.def(
        "gmm_fit", &fit_py, nb::arg("X"), nb::arg("order"), nb::arg("k"), nb::arg("type"), nb::arg("init"),
        nb::arg("init_data").none(), nb::arg("w0").none(), nb::arg("m0").none(), nb::arg("P0").none(),
        nb::arg("n_init"), nb::arg("max_iter"), nb::arg("tol"), nb::arg("reg"), nb::arg("key0"), nb::arg("key1")
    );
    m.def(
        "gmm_estimate", &estimate_py, nb::arg("X"), nb::arg("means"), nb::arg("U"), nb::arg("logdet"), nb::arg("logw"),
        nb::arg("type"), nb::arg("e")
    );
    m.def("gmm_precisions", &precisions_py, nb::arg("covariances"), nb::arg("type"), nb::arg("k"), nb::arg("p"));
    m.def("gmm_mstep", &mstep_py, nb::arg("X"), nb::arg("resp"), nb::arg("type"), nb::arg("reg"));
    m.def(
        "gmm_sample", &sample_py, nb::arg("N"), nb::arg("weights"), nb::arg("perm"), nb::arg("means"), nb::arg("U"),
        nb::arg("type"), nb::arg("e"), nb::arg("key0"), nb::arg("key1")
    );
    m.def("gmm_subkey", &subkey_py, nb::arg("key0"), nb::arg("key1"), nb::arg("r"));
}

}  // namespace sd::cluster
