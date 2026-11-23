// Kernel regression (docs/design/smooth.md §3.4, §3.8; decisions SM17, SM80-SM94): the Nadaraya-Watson and local
// linear fits with a product Gaussian kernel, their leverages and leave-one-out values, the CV and AICc criteria, the
// bandwidth searches and the direct plug-in bandwidth (dpill), on the Gaussian sums of smooth/gauss_sum.hpp.
//
// _core.smooth.kreg_fit(X, y, w, degree, bw, criterion) -> (fitted, leverage, df, rss, cv, aicc, bw, status,
//     converged, nfev, Xs, ys, ws)
//     X (n, p) C-contiguous float64 and y (n,) finite, w (n,) >= 0 with a positive value or None, degree 0 or 1;
//     bw (p,) positive, or None with criterion 1 (cv_ls) or 2 (aicc) to choose it. The points of positive weight
//     are sorted (1-D by (x, y, w, index), n-D by the coordinates, y, w and the index: decision SM84), so the result
//     is a function of the data and not of their order. fitted and leverage come back in the input order (a point
//     of weight 0 has the fit at its x and leverage 0); df = tr L and rss = sum w r^2 over the points of positive
//     weight; cv = sum w (y - yhat_-i)^2 / nobs and aicc = log(rss / nobs) + (1 + df / nobs) / (1 - (df + 2) / nobs)
//     at the bandwidth used; status 0 (inside), -1 / 1 (the search ended at the small / large end), 2 (the
//     evaluation cap); nfev the criterion evaluations; Xs, ys, ws (None without weights) the sorted points of
//     positive weight that a result keeps for predict. status -3: no bandwidth has a finite criterion (the
//     caller's ValueError); -4: a spread of 0 (the caller's ValueError).
// _core.smooth.kreg_predict(Xs, ys, ws, bw, degree, Xq, what) -> values (q,) and, what 1, sum l_j^2 / w_j (q,);
//     what 2: the gradient (q, p) of the local line (degree 1)
// _core.smooth.kreg_criterion(X, y, w, degree, bw, criterion) -> the criterion's value at bw (cv 1, aicc 2)
// _core.smooth.bw_dpill(x, y) -> (h, code): code 0, or 1 too few points, 2 x of no spread, 3 a block fit without
//     full rank, 4 an estimate that is not a positive number (h NaN)
//
// Every evaluation point is one call of local_fit, whatever task computes it, and the criteria add their per-point
// terms pairwise in sorted order: the results have the same bits for every thread count (decision SM3).

#include "smooth/gauss_sum.hpp"
#include "smooth/gauss_tree.hpp"
#include "smooth/search.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "smooth/smooth.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::smooth {

namespace nb = nanobind;

namespace {

using gauss::Axis;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kU = 0x1p-53;

// ---- Householder QR with column pivoting -----------------------------------------------------------------------

// The m x k matrix A (column-major, leading dimension m) and the right-hand side b (m) are overwritten. R (k x k
// row-major; its first `rank` rows), perm (R's column j is A's column perm[j]) and qtb (Q^T b, first rank entries).
// A column whose remaining norm is at most tol times the first pivot's ends the factorization (the rank).
struct QR {
    std::int64_t rank = 0, k = 0;
    std::vector<double> R, qtb;
    std::vector<std::int64_t> perm;
};

void qr_pivoted(double* A, double* b, std::int64_t m, std::int64_t k, double tol, QR& out) {
    out.k = k;
    out.R.assign(k * k, 0.0);
    out.qtb.assign(k, 0.0);
    out.perm.resize(k);
    std::iota(out.perm.begin(), out.perm.end(), std::int64_t{0});
    out.rank = 0;
    double first = 0;
    const std::int64_t steps = std::min(m, k);
    for (std::int64_t s = 0; s < steps; ++s) {
        std::int64_t piv = s;
        double best = -1;
        for (std::int64_t c = s; c < k; ++c) {
            const double* col = A + c * m;
            double nrm = 0;
            for (std::int64_t j = s; j < m; ++j) nrm += col[j] * col[j];
            if (nrm > best) best = nrm, piv = c;
        }
        best = std::sqrt(best);
        if (s == 0) first = best;
        if (!(best > tol * first) || !(best > 0)) break;
        if (piv != s) {
            std::swap_ranges(A + s * m, A + s * m + m, A + piv * m);
            std::swap(out.perm[s], out.perm[piv]);
        }
        double* v = A + s * m;
        const double alpha = v[s] > 0 ? -best : best;
        const double v0 = v[s] - alpha;
        // H = I - tau u u^T with u = (1, v[s+1:] / v0), tau = -v0 / alpha
        const double tau = -v0 / alpha;
        for (std::int64_t j = s + 1; j < m; ++j) v[j] /= v0;
        const auto apply = [&](double* col) {
            double d = col[s];
            for (std::int64_t j = s + 1; j < m; ++j) d += v[j] * col[j];
            d *= tau;
            col[s] -= d;
            for (std::int64_t j = s + 1; j < m; ++j) col[j] -= d * v[j];
        };
        for (std::int64_t c = s + 1; c < k; ++c) apply(A + c * m);
        apply(b);
        v[s] = alpha;
        out.rank = s + 1;
    }
    for (std::int64_t r = 0; r < out.rank; ++r) {
        for (std::int64_t c = r; c < k; ++c) out.R[r * k + c] = A[c * m + r];
        out.qtb[r] = b[r];
    }
}

// x = R11^-1 rhs (rank r, back substitution in place).
void back_solve(const QR& q, double* x) {
    const std::int64_t r = q.rank, k = q.k;
    for (std::int64_t i = r - 1; i >= 0; --i) {
        double s = x[i];
        for (std::int64_t c = i + 1; c < r; ++c) s -= q.R[i * k + c] * x[c];
        x[i] = s / q.R[i * k + i];
    }
}

// x = R11^-T rhs (forward substitution in place).
void forward_solve(const QR& q, double* x) {
    const std::int64_t r = q.rank, k = q.k;
    for (std::int64_t i = 0; i < r; ++i) {
        double s = x[i];
        for (std::int64_t c = 0; c < i; ++c) s -= q.R[c * k + i] * x[c];
        x[i] = s / q.R[i * k + i];
    }
}

// ---- the data and one local fit --------------------------------------------------------------------------------

struct KData {
    std::int64_t n = 0, p = 1;
    const double* X = nullptr;  // n x p row-major; p = 1: sorted ascending
    const double* y = nullptr;
    const double* w = nullptr;  // null: every weight 1; else every weight > 0
    double T = 0;               // the cut of the local constant (gauss::cut_threshold)
    // the tree path of several variables (smooth/gauss_tree.hpp; decisions SM115, SM116): built over
    // ((x - mu) sc) inv of the bandwidth in use; null: every point is scanned
    const gauss::TreePath* tree = nullptr;
    const double* mu = nullptr;
};

struct Scratch {
    std::vector<double> D, e, t, yk, wk, A, b, tmp;
    std::vector<double> e2, t2, w2;  // the second tier of a one-variable local line (decision SM86)
    std::vector<std::int64_t> idx, idx2, ord;
    std::vector<double> w0;
    gauss::TreeScratch ts;
};

// The candidates of the query x0 from the tree (smooth/gauss_tree.hpp): false when the caller scans.
bool tree_candidates(const KData& K, const Axis* ax, const double* x0, double T, std::int64_t skip, Scratch& S) {
    if (!K.tree) return false;
    const std::int64_t p = K.p;
    S.w0.resize(p);
    double q = 0;
    for (std::int64_t l = 0; l < p; ++l) {
        S.w0[l] = gauss::z_of(x0[l], K.mu[l], ax[l]);
        q = std::max(q, std::fabs(S.w0[l]));
    }
    const double eps = 0x1p-44 * (K.tree->wmax() + q) + 0x1p-1000;
    const auto D = [&](std::int64_t j) {
        const double* row = K.X + j * p;
        double s = 0;
        for (std::int64_t l = 0; l < p; ++l) {
            const double z = gauss::z_of(row[l], x0[l], ax[l]);
            s += z * z;
        }
        return s;
    };
    return K.tree->candidates(S.w0.data(), T, eps, skip, D, S.ts);
}

// One local fit: the kept points (scratch e, t, yk, wk; m of them), their sums and, degree 1, the centred slope.
struct Local {
    bool ok = false;
    bool tier = false;  // p = 1: the slope comes from the second tier (every other x's weight underflows)
    std::int64_t m = 0, m2 = 0, rank = 0;
    double d0 = 0, a = 0, ybar = kNaN, value = kNaN, et2 = 0;  // et2: sum e2 t2 of the second tier
    std::vector<double> xc, tbar, beta, t0;  // the centre (a nearest point), mean of t, slope in t units, t of x0
    QR qr;                                   // p >= 2; p = 1 keeps sqrt(c') in qr.R[0]
};

// t of the point x (p coordinates) relative to the centre, per axis.
inline void t_of(const double* x, const Local& L, const Axis* ax, std::int64_t p, double* t) {
    for (std::int64_t l = 0; l < p; ++l) t[l] = gauss::z_of(x[l], L.xc[l], ax[l]);
}

void local_fit(const KData& K, const Axis* ax, int degree, const double* x0, std::int64_t skip, Scratch& S, Local& L) {
    const std::int64_t n = K.n, p = K.p;
    L.ok = false, L.tier = false, L.rank = 0, L.m = 0, L.m2 = 0, L.value = kNaN, L.ybar = kNaN;
    L.qr.rank = 0;
    for (std::int64_t l = 0; l < p; ++l)
        if (!(x0[l] == x0[l])) return;
    const double T = degree == 0 ? K.T : gauss::kUnderflow;
    std::int64_t c;
    double d0;
    L.xc.resize(p), L.tbar.assign(p, 0.0), L.beta.assign(p, 0.0), L.t0.resize(p);
    if (p == 1) {
        c = gauss::nearest_1d(K.X, n, x0[0], skip);
        if (c < 0) return;
        d0 = gauss::sq_of(K.X[c], x0[0], ax[0]);
        if (!(d0 < kInf)) return;
        std::int64_t lo, hi;
        gauss::range_1d(K.X, n, x0[0], ax[0], T, c, lo, hi);
        const std::int64_t m = hi - lo;
        const double zc = gauss::z_of(K.X[c], x0[0], ax[0]);
        S.D.resize(m), S.e.resize(m), S.yk.resize(m), S.wk.resize(m), S.idx.resize(m);
        for (std::int64_t j = 0; j < m; ++j) {
            S.D[j] = j + lo == skip ? kInf : gauss::excess_1d(K.X[j + lo], K.X[c], zc, ax[0]);
            S.yk[j] = K.y[j + lo];
            S.wk[j] = K.w ? K.w[j + lo] : 1.0;
            S.idx[j] = j + lo;
        }
        L.m = m;
    } else {
        // the tree's candidates (ascending) or every point: the same kept set in the same order
        const bool tree = tree_candidates(K, ax, x0, T, skip, S);
        const std::int64_t cnt = tree ? static_cast<std::int64_t>(S.ts.cand.size()) : n;
        const std::int64_t* cand = tree ? S.ts.cand.data() : nullptr;
        S.D.resize(cnt);
        if (tree) {
            for (std::int64_t k = 0; k < cnt; ++k) {
                const double* row = K.X + cand[k] * p;
                double s = 0;
                for (std::int64_t l = 0; l < p; ++l) {
                    const double z = gauss::z_of(row[l], x0[l], ax[l]);
                    s += z * z;
                }
                S.D[k] = s;
            }
        } else {
            gauss::sqdist_nd(K.X, n, p, x0, ax, skip, S.D.data());
        }
        c = -1, d0 = kInf;
        for (std::int64_t k = 0; k < cnt; ++k)
            if (S.D[k] < d0) d0 = S.D[k], c = cand ? cand[k] : k;
        if (c < 0) return;
        std::int64_t m = 0;
        S.idx.resize(cnt), S.yk.resize(cnt), S.wk.resize(cnt), S.e.resize(cnt);
        S.tmp.resize(p);
        double* zc = S.tmp.data();
        for (std::int64_t l = 0; l < p; ++l) zc[l] = gauss::z_of(K.X[c * p + l], x0[l], ax[l]);
        const double* xcr = K.X + c * p;
        for (std::int64_t k = 0; k < cnt; ++k) {
            const std::int64_t j = cand ? cand[k] : k;
            if (!(S.D[k] < kInf)) continue;  // skip, or a distance that overflows
            const double ex = gauss::excess_nd(K.X + j * p, xcr, zc, ax, p);
            if (!(ex <= T)) continue;
            S.D[m] = ex, S.idx[m] = j, S.yk[m] = K.y[j], S.wk[m] = K.w ? K.w[j] : 1.0;
            ++m;
        }
        L.m = m;
    }
    const std::int64_t m = L.m;
    S.e.resize(m);
    gauss::terms(S.D.data(), K.w ? S.wk.data() : nullptr, 0.0, m, S.e.data());
    for (std::int64_t l = 0; l < p; ++l) L.xc[l] = K.X[c * p + l];
    L.ok = true, L.d0 = d0;
    const double* e = S.e.data();
    const double* yk = S.yk.data();
    if (degree == 0 || p > 1) {
        double s2[2];
        gauss::sums_of<2>(m, [&](std::int64_t j, double* v) { v[0] = e[j], v[1] = e[j] * yk[j]; }, s2);
        L.a = s2[0];
        L.ybar = s2[1] / L.a;
        L.value = L.ybar;
        if (degree == 0) return;
    }
    t_of(x0, L, ax, p, L.t0.data());
    S.t.resize(m * p);  // column-major: t[l * m + j]
    for (std::int64_t j = 0; j < m; ++j) {
        const double* row = K.X + S.idx[j] * p;
        for (std::int64_t l = 0; l < p; ++l) S.t[l * m + j] = gauss::z_of(row[l], L.xc[l], ax[l]);
    }
    if (p == 1) {
        const double* t = S.t.data();
        double s3[3];
        gauss::sums_of<3>(
            m, [&](std::int64_t j, double* v) { v[0] = e[j], v[1] = e[j] * t[j], v[2] = e[j] * yk[j]; }, s3
        );
        L.a = s3[0];
        const double tb = s3[1] / L.a, yb = s3[2] / L.a;
        L.tbar[0] = tb, L.ybar = yb, L.value = yb;
        double s2[2];
        gauss::sums_of<2>(
            m,
            [&](std::int64_t j, double* v) {
                const double d = t[j] - tb, ed = e[j] * d;
                v[0] = ed * d, v[1] = ed * (yk[j] - yb);
            },
            s2
        );
        double cc = s2[0], s1 = s2[1];
        if (!(cc > 0)) {
            // Every positive weight at one x, xg: the limit of the local line when the other x's weights underflow
            // is the line through the weighted mean there with the slope of the next tier, weighted relative to the
            // nearest point c2 of another x (decision SM86). Without another x: the local mean.
            const double xg = K.X[c];
            const std::int64_t g0 = std::lower_bound(K.X, K.X + n, xg) - K.X;
            const std::int64_t g1 = std::upper_bound(K.X, K.X + n, xg) - K.X;
            std::int64_t lft = g0 - 1, rgt = g1;
            if (lft == skip) --lft;
            if (rgt == skip) ++rgt;
            const double x00 = x0[0];
            std::int64_t c2 = -1;
            if (lft >= 0 && rgt < n) c2 = (x00 - K.X[lft] <= K.X[rgt] - x00) ? lft : rgt;
            else if (lft >= 0) c2 = lft;
            else if (rgt < n) c2 = rgt;
            if (c2 < 0) return;
            std::int64_t lo2, hi2;
            gauss::range_1d(K.X, n, x00, ax[0], gauss::kUnderflow, c2, lo2, hi2);
            const double z2 = gauss::z_of(K.X[c2], x00, ax[0]);
            S.D.clear(), S.t2.clear(), S.w2.clear(), S.idx2.clear(), S.tmp.clear();
            for (std::int64_t j = lo2; j < hi2; ++j) {
                if (j == skip || (j >= g0 && j < g1)) continue;
                S.D.push_back(gauss::excess_1d(K.X[j], K.X[c2], z2, ax[0]));
                S.t2.push_back(gauss::z_of(K.X[j], xg, ax[0]));
                S.w2.push_back(K.w ? K.w[j] : 1.0);
                S.tmp.push_back(K.y[j]);
                S.idx2.push_back(j);
            }
            const std::int64_t m2 = static_cast<std::int64_t>(S.idx2.size());
            S.e2.resize(m2);
            gauss::terms(S.D.data(), K.w ? S.w2.data() : nullptr, 0.0, m2, S.e2.data());
            const double *e2 = S.e2.data(), *t2 = S.t2.data(), *y2 = S.tmp.data();
            double q2[3];
            gauss::sums_of<3>(
                m2,
                [&](std::int64_t j, double* v) {
                    const double et = e2[j] * t2[j];
                    v[0] = et * t2[j], v[1] = et * (y2[j] - yb), v[2] = et;
                },
                q2
            );
            if (!(q2[0] > 0)) return;
            L.tier = true, L.m2 = m2, L.et2 = q2[2];
            cc = q2[0], s1 = q2[1];
        }
        L.rank = 1;
        L.qr.k = 1, L.qr.rank = 1;
        L.qr.R.assign(1, std::sqrt(cc));
        L.beta[0] = s1 / cc;
        L.value = yb + (L.t0[0] - tb) * L.beta[0];
        return;
    }
    // p >= 2: the weighted design [1, t] (t from the centre, not from the mean: subtracting a rounded mean would
    // bury the directions that only light rows carry) by Householder QR with column pivoting, the rows by decreasing
    // weight (graded rows); a column is dropped only when its remaining norm is exactly 0 (decision SM86)
    const std::int64_t k = p + 1;
    S.ord.resize(m);
    std::iota(S.ord.begin(), S.ord.end(), std::int64_t{0});
    std::stable_sort(S.ord.begin(), S.ord.end(), [e](std::int64_t i, std::int64_t j) { return e[i] > e[j]; });
    S.A.resize(m * k), S.b.resize(m);
    for (std::int64_t r = 0; r < m; ++r) {
        const std::int64_t j = S.ord[r];
        const double sq = std::sqrt(e[j]);
        S.A[r] = sq;
        for (std::int64_t l = 0; l < p; ++l) S.A[(l + 1) * m + r] = sq * S.t[l * m + j];
        S.b[r] = sq * yk[j];
    }
    qr_pivoted(S.A.data(), S.b.data(), m, k, 0.0, L.qr);
    L.rank = L.qr.rank - 1;  // slope directions kept (the constant column is never 0)
    S.tmp.assign(L.qr.qtb.begin(), L.qr.qtb.begin() + L.qr.rank);
    back_solve(L.qr, S.tmp.data());
    std::vector<double> full(k, 0.0);
    for (std::int64_t i = 0; i < L.qr.rank; ++i) full[L.qr.perm[i]] = S.tmp[i];
    double v = full[0];
    for (std::int64_t l = 0; l < p; ++l) L.beta[l] = full[l + 1], v += L.t0[l] * full[l + 1];
    L.value = v;
}

// p >= 2: q = M^+ phi (phi = (1, t), M = R^T R over the kept columns), into q (p + 1).
void solve_normal(const Local& L, const double* t, std::int64_t p, Scratch& S, double* q, bool twice) {
    const std::int64_t r = L.qr.rank;
    S.tmp.resize(r);
    for (std::int64_t i = 0; i < r; ++i) {
        const std::int64_t c = L.qr.perm[i];
        S.tmp[i] = c == 0 ? 1.0 : t[c - 1];
    }
    forward_solve(L.qr, S.tmp.data());
    if (!twice) {
        for (std::int64_t i = 0; i < r; ++i) q[i] = S.tmp[i];
        return;
    }
    back_solve(L.qr, S.tmp.data());
    for (std::int64_t i = 0; i <= p; ++i) q[i] = 0.0;
    for (std::int64_t i = 0; i < r; ++i) q[L.qr.perm[i]] = S.tmp[i];
}

// phi^T M^-1 phi for phi = (1, t) in the fit's units (h' of decision SM85): p = 1 in the centred form
// 1 / a + ((t - tbar) / sqrt(c'))^2, p >= 2 |R^-T P^T phi|^2.
double hprime(const Local& L, const double* t, std::int64_t p, Scratch& S) {
    if (p == 1) {
        double h = 1.0 / L.a;
        if (L.rank == 0) return h;
        const double d = (t[0] - L.tbar[0]) / L.qr.R[0];
        return h + d * d;
    }
    std::vector<double> q(p + 1);
    solve_normal(L, t, p, S, q.data(), false);
    double s = 0;
    for (std::int64_t i = 0; i < L.qr.rank; ++i) s += q[i] * q[i];
    return s;
}

// The hat row of the fit at its point x0: l_j = e_j (1 / a + (t_j - tbar)^T q), q = C^+ (t0 - tbar); returns
// sum l_j^2 / w_j over the kept points (unweighted: sum l_j^2) and, self >= 0, l of that data point in *lself.
double hat_norm2(const Local& L, std::int64_t p, Scratch& S, std::int64_t self = -1, double* lself = nullptr) {
    if (!L.ok) return kNaN;
    const std::int64_t m = L.m;
    std::vector<double> q(p + 1, 0.0);
    if (p == 1) {
        if (L.rank > 0) {
            const double r = L.qr.R[0];
            q[0] = (L.t0[0] - L.tbar[0]) / (r * r);
        }
    } else if (L.qr.rank > 0) {
        solve_normal(L, L.t0.data(), p, S, q.data(), true);
    }
    // p = 1: l_j = e_j (1 / a + (t_j - tbar) q); the second tier moves the first tier's mean by -t0 b, so its rows
    // are e_j / a (1 - q sum e2 t2). p >= 2: l_j = e_j (q_0 + t_j^T q_1:)
    const double inva = L.tier ? (1.0 - q[0] * L.et2) / L.a : 1.0 / L.a;
    const auto row = [&](std::int64_t j) {
        if (p > 1 && L.qr.rank > 0) {
            double s = q[0];
            for (std::int64_t l = 0; l < p; ++l) s += S.t[l * m + j] * q[l + 1];
            return S.e[j] * s;
        }
        double s = inva;
        if (p == 1 && L.rank > 0) s += (S.t[j] - L.tbar[0]) * q[0];
        return S.e[j] * s;
    };
    if (lself) {
        *lself = 0;
        for (std::int64_t j = 0; j < m; ++j)
            if (S.idx[j] == self) *lself = row(j);
        for (std::int64_t j = 0; j < L.m2; ++j)
            if (S.idx2[j] == self) *lself = S.e2[j] * S.t2[j] * q[0];
    }
    double s = gauss::sum_of(m, [&](std::int64_t j) {
        const double l = row(j);
        return l * l / S.wk[j];
    });
    if (L.tier)  // the second tier's rows l_j = e2_j t_j q (their t - tbar is t: tbar = 0 at the first tier's x)
        s += gauss::sum_of(L.m2, [&](std::int64_t j) {
            const double l = S.e2[j] * S.t2[j] * q[0];
            return l * l / S.w2[j];
        });
    return s;
}

// ---- the fit at every data point --------------------------------------------------------------------------------

struct PointFit {
    double fitted = kNaN, leverage = kNaN, cv = kNaN, rss = kNaN;
};

// The fit at data point i of K (decision SM85): the leave-one-out fit yhat_-i first, then the full fit from it by
// the rank-one update of the point's own term: L_ii = w h' / (E + w h'), 1 - L_ii = E / (E + w h'), with E =
// exp(-D0' / 2) the factor the leave-one-out sums took out; yhat = (1 - L_ii) yhat_-i + L_ii y_i and y - yhat =
// (1 - L_ii)(y - yhat_-i), neither with a cancellation. A leave-one-out line without full rank (the point brings
// the rank) takes the full fit directly.
PointFit fit_point(const KData& K, const Axis* ax, int degree, std::int64_t i, Scratch& S, Local& Lx, Local& Lf) {
    const std::int64_t p = K.p;
    const double* xi = K.X + i * p;
    const double yi = K.y[i], wi = K.w ? K.w[i] : 1.0;
    PointFit out;
    local_fit(K, ax, degree, xi, i, S, Lx);
    if (!Lx.ok) {  // the only point
        out.fitted = yi, out.leverage = 1.0, out.cv = kNaN, out.rss = 0.0;
        return out;
    }
    const double loo = Lx.value;
    if (degree == 1 && (Lx.rank < p || Lx.tier)) {
        local_fit(K, ax, degree, xi, -1, S, Lf);
        double lself = 0;
        std::vector<double> ti(p);
        t_of(xi, Lf, ax, p, ti.data());
        const double hp = hprime(Lf, ti.data(), p, S);
        // e_i = w_i: the point is at D = 0, the smallest
        lself = wi * hp;
        out.fitted = Lf.value, out.leverage = std::min(lself, 1.0);
        const double r = yi - Lf.value;
        out.rss = wi * r * r;
        out.cv = wi * (yi - loo) * (yi - loo);
        return out;
    }
    double hp;
    if (degree == 0) {
        hp = 1.0 / Lx.a;
    } else {
        std::vector<double> ti(p);
        t_of(xi, Lx, ax, p, ti.data());
        hp = hprime(Lx, ti.data(), p, S);
    }
    const double E = vmath::exp1(-0.5 * Lx.d0);
    const double wh = wi * hp, den = E + wh;
    const double lev = wh / den, om = E / den;
    const double d = yi - loo;
    out.fitted = om * loo + lev * yi;
    out.leverage = lev;
    out.cv = wi * d * d;
    const double r = om * d;
    out.rss = wi * r * r;
    return out;
}

// Runs fn(i, scratch) for i in [0, count) in tasks split by size alone (each point about `per_point` work).
template <class F>
void for_points(std::int64_t count, std::int64_t per_point, const F& fn) {
    if (count == 0) return;
    const std::int64_t cost = std::max<std::int64_t>(1, per_point);
    const std::int64_t work = count * cost;
    if (work >= kParallelThreshold && count > 1 && parallel::available()) {
        const std::int64_t per = std::max<std::int64_t>(1, (std::int64_t{1} << 17) / cost);
        const std::int64_t tasks = (count + per - 1) / per;
        parallel::parallel_for(
            tasks,
            [&](std::int64_t t) {
                Scratch S;
                Local A, B;
                const std::int64_t end = std::min(count, (t + 1) * per);
                for (std::int64_t i = t * per; i < end; ++i) fn(i, S, A, B);
            },
            work_ns(per * cost)
        );
    } else {
        Scratch S;
        Local A, B;
        for (std::int64_t i = 0; i < count; ++i) fn(i, S, A, B);
    }
}

std::int64_t point_cost(const KData& K, int degree) { return K.n * (8 + 4 * K.p * (degree + 1)); }

struct Fit {
    std::vector<double> fitted, leverage, cv, rss;  // sorted order of K
    double df = kNaN, rss_sum = kNaN, cv_value = kNaN, aicc = kNaN;
};

// The tree path (decision SM116): the local constant of several variables from kTreeMinPoints points on (the shape
// alone decides; the local line's sums keep every point that does not underflow, so the tree would cut little).
constexpr std::int64_t kTreeMinPoints = 4096;

struct TreeHold {
    gauss::TreePath tree;
    std::vector<double> mu;
};

void attach_tree(KData& K, const std::vector<Axis>& ax, int degree, TreeHold& H) {
    const int mode = gauss::g_tree_mode.load();
    if (K.p < 2 || mode == 1) return;
    if (mode != 2 && (degree != 0 || K.n < kTreeMinPoints)) return;
    const std::int64_t n = K.n, p = K.p;
    H.mu.assign(p, 0.0);
    for (std::int64_t l = 0; l < p; ++l)
        H.mu[l] = gauss::sum_of(n, [&](std::int64_t j) { return K.X[j * p + l]; }) / static_cast<double>(n);
    std::vector<double> W(n * p);
    for (std::int64_t j = 0; j < n; ++j)
        for (std::int64_t l = 0; l < p; ++l) W[j * p + l] = gauss::z_of(K.X[j * p + l], H.mu[l], ax[l]);
    if (H.tree.build(W.data(), n, p)) K.tree = &H.tree, K.mu = H.mu.data();
}

void fit_all(const KData& K0, const std::vector<double>& bw, int degree, Fit& F) {
    const std::int64_t n = K0.n;
    std::vector<Axis> ax(K0.p);
    for (std::int64_t l = 0; l < K0.p; ++l) ax[l] = gauss::axis_of(bw[l]);
    KData K = K0;
    TreeHold th;
    attach_tree(K, ax, degree, th);
    F.fitted.resize(n), F.leverage.resize(n), F.cv.resize(n), F.rss.resize(n);
    for_points(n, point_cost(K, degree), [&](std::int64_t i, Scratch& S, Local& A, Local& B) {
        const PointFit f = fit_point(K, ax.data(), degree, i, S, A, B);
        F.fitted[i] = f.fitted, F.leverage[i] = f.leverage, F.cv[i] = f.cv, F.rss[i] = f.rss;
    });
    const double nn = static_cast<double>(n);
    F.df = gauss::sum(F.leverage.data(), n);
    F.rss_sum = gauss::sum(F.rss.data(), n);
    F.cv_value = gauss::sum(F.cv.data(), n) / nn;
    const double den = 1.0 - (F.df + 2.0) / nn;
    F.aicc = den > 0 ? std::log(F.rss_sum / nn) + (1.0 + F.df / nn) / den : kInf;
}

double criterion_of(const Fit& F, int criterion) { return criterion == 1 ? F.cv_value : F.aicc; }

// ---- the bandwidth searches (decisions SM87, SM88) ----------------------------------------------------------------

constexpr double kLog64 = 4.1588830833596715;  // log 64: the search spans h_ref / 64 .. 64 h_ref
constexpr std::int64_t kGrid = 33;

struct Choice {
    std::vector<double> bw;
    std::int64_t status = 0, nfev = 0;
    bool converged = true;
};

// The objective of the searches: log CV or AICc (scale free, so that the tolerances need no units); NaN is +inf.
double objective(const KData& K, const std::vector<double>& href, const double* s, int degree, int criterion) {
    std::vector<double> bw(K.p);
    for (std::int64_t l = 0; l < K.p; ++l) bw[l] = href[l] * vmath::exp1(s[l]);
    Fit F;
    fit_all(K, bw, degree, F);
    double v = criterion_of(F, criterion);
    if (criterion == 1) v = std::log(v);
    return v == v ? v : kInf;
}

Choice search_1d(const KData& K, const std::vector<double>& href, int degree, int criterion) {
    Choice ch;
    double g[kGrid], s[kGrid];
    const double step = 2.0 * kLog64 / (kGrid - 1);
    double best = kInf;
    std::int64_t kb = -1;
    for (std::int64_t k = 0; k < kGrid; ++k) {
        s[k] = -kLog64 + static_cast<double>(k) * step;
        g[k] = objective(K, href, &s[k], degree, criterion);
        if (g[k] <= best && g[k] < kInf) best = g[k], kb = k;  // ties: the larger bandwidth
    }
    ch.nfev = kGrid;
    if (kb < 0) {
        ch.status = -3;
        return ch;
    }
    double sb = s[kb];
    if (kb == 0 || kb == kGrid - 1) {
        ch.status = kb == 0 ? -1 : 1;
        ch.converged = false;
    } else {
        const ScalarResult r = fminbound(
            [&](double v) { return objective(K, href, &v, degree, criterion); }, s[kb - 1], s[kb + 1], 1e-6, 60
        );
        ch.nfev += r.nfev;
        if (r.f < best) sb = r.x;
        if (!r.converged) ch.status = 2, ch.converged = false;
    }
    ch.bw = {href[0] * vmath::exp1(sb)};
    return ch;
}

// Nelder-Mead on s = log(h / h_ref) in the box [-log 64, log 64]^p (scipy's coefficients 1, 2, 1/2, 1/2; points
// clipped into the box): the simplex x0 = 0 and 0.5 e_k, stopping when every vertex is within 1e-5 of the best in
// every coordinate and every value within 1e-9 of the best, or after 200 p evaluations.
Choice search_nd(const KData& K, const std::vector<double>& href, int degree, int criterion) {
    const std::int64_t N = K.p;
    const std::int64_t maxfev = 200 * N;
    Choice ch;
    std::vector<std::vector<double>> sim(N + 1, std::vector<double>(N, 0.0));
    std::vector<double> fs(N + 1);
    const auto clip = [](std::vector<double>& v) {
        for (double& x : v) x = std::min(kLog64, std::max(-kLog64, x));
    };
    const auto f = [&](const std::vector<double>& v) {
        ++ch.nfev;
        return objective(K, href, v.data(), degree, criterion);
    };
    for (std::int64_t k = 0; k < N; ++k) sim[k + 1][k] = 0.5;
    for (std::int64_t k = 0; k <= N; ++k) fs[k] = f(sim[k]);
    std::vector<std::int64_t> ord(N + 1);
    const auto sort = [&]() {
        std::iota(ord.begin(), ord.end(), std::int64_t{0});
        std::stable_sort(ord.begin(), ord.end(), [&](std::int64_t a, std::int64_t b) { return fs[a] < fs[b]; });
        std::vector<std::vector<double>> s2(N + 1);
        std::vector<double> f2(N + 1);
        for (std::int64_t k = 0; k <= N; ++k) s2[k] = sim[ord[k]], f2[k] = fs[ord[k]];
        sim.swap(s2), fs.swap(f2);
    };
    sort();
    bool converged = false;
    std::vector<double> xbar(N), xr(N), xe(N), xc(N);
    while (ch.nfev < maxfev) {
        double dx = 0, df = 0;
        for (std::int64_t k = 1; k <= N; ++k) {
            for (std::int64_t l = 0; l < N; ++l) dx = std::max(dx, std::fabs(sim[k][l] - sim[0][l]));
            df = std::max(df, fs[k] == fs[0] ? 0.0 : std::fabs(fs[k] - fs[0]));
        }
        if (dx <= 1e-5 && df <= 1e-9) {
            converged = true;
            break;
        }
        for (std::int64_t l = 0; l < N; ++l) {
            double s = 0;
            for (std::int64_t k = 0; k < N; ++k) s += sim[k][l];
            xbar[l] = s / static_cast<double>(N);
        }
        for (std::int64_t l = 0; l < N; ++l) xr[l] = 2.0 * xbar[l] - sim[N][l];
        clip(xr);
        const double fr = f(xr);
        bool shrink = false;
        if (fr < fs[0]) {
            for (std::int64_t l = 0; l < N; ++l) xe[l] = 3.0 * xbar[l] - 2.0 * sim[N][l];
            clip(xe);
            const double fe = f(xe);
            if (fe < fr) sim[N] = xe, fs[N] = fe;
            else sim[N] = xr, fs[N] = fr;
        } else if (fr < fs[N - 1]) {
            sim[N] = xr, fs[N] = fr;
        } else if (fr < fs[N]) {
            for (std::int64_t l = 0; l < N; ++l) xc[l] = 1.5 * xbar[l] - 0.5 * sim[N][l];
            clip(xc);
            const double fc = f(xc);
            if (fc <= fr) sim[N] = xc, fs[N] = fc;
            else shrink = true;
        } else {
            for (std::int64_t l = 0; l < N; ++l) xc[l] = 0.5 * xbar[l] + 0.5 * sim[N][l];
            clip(xc);
            const double fc = f(xc);
            if (fc < fs[N]) sim[N] = xc, fs[N] = fc;
            else shrink = true;
        }
        if (shrink) {
            for (std::int64_t k = 1; k <= N; ++k) {
                for (std::int64_t l = 0; l < N; ++l) sim[k][l] = sim[0][l] + 0.5 * (sim[k][l] - sim[0][l]);
                clip(sim[k]);
                fs[k] = f(sim[k]);
            }
        }
        sort();
    }
    if (!(fs[0] < kInf)) {
        ch.status = -3;
        return ch;
    }
    ch.converged = converged;
    ch.status = converged ? 0 : 2;
    for (std::int64_t l = 0; l < N; ++l) {
        if (sim[0][l] <= -kLog64) ch.status = -1, ch.converged = false;
        else if (sim[0][l] >= kLog64 && ch.status != -1) ch.status = 1, ch.converged = false;
    }
    ch.bw.resize(N);
    for (std::int64_t l = 0; l < N; ++l) ch.bw[l] = href[l] * vmath::exp1(sim[0][l]);
    return ch;
}

// h_ref per variable: 1.06 sigma nobs^(-1/(4 + p)) with sigma the weighted standard deviation (divisor sum w), as
// statsmodels' starting point (decision SM87). Empty when a variable has no spread.
std::vector<double> reference_bandwidth(const KData& K) {
    const std::int64_t n = K.n, p = K.p;
    const double sw = K.w ? gauss::sum(K.w, n) : static_cast<double>(n);
    const double f = 1.06 * std::pow(static_cast<double>(n), -1.0 / (4.0 + static_cast<double>(p)));
    std::vector<double> h(p);
    for (std::int64_t l = 0; l < p; ++l) {
        const double mean =
            gauss::sum_of(n, [&](std::int64_t j) { return (K.w ? K.w[j] : 1.0) * K.X[j * p + l]; }) / sw;
        const double var = gauss::sum_of(
                               n,
                               [&](std::int64_t j) {
                                   const double d = K.X[j * p + l] - mean;
                                   return (K.w ? K.w[j] : 1.0) * d * d;
                               }
                           ) /
                           sw;
        const double s = std::sqrt(var);
        if (!(s > 0) || !std::isfinite(s)) return {};
        h[l] = f * s;
    }
    return h;
}

// ---- dpill (decision SM90) ---------------------------------------------------------------------------------------

// A local polynomial of degree `deg` at x0 with the Gaussian kernel of standard deviation g on sorted x (unit
// weights): the coefficient of t^r, t = (x - x0) / g (NaN without full rank).
double local_poly_coef(
    const double* x, const double* y, std::int64_t n, double x0, double g, int deg, int r, Scratch& S, QR& q
) {
    const Axis a = gauss::axis_of(g);
    const std::int64_t c = gauss::nearest_1d(x, n, x0);
    std::int64_t lo, hi;
    gauss::range_1d(x, n, x0, a, gauss::kUnderflow, c, lo, hi);
    const std::int64_t m = hi - lo, k = deg + 1;
    const double zc = gauss::z_of(x[c], x0, a);
    S.D.resize(m), S.e.resize(m);
    for (std::int64_t j = 0; j < m; ++j) S.D[j] = gauss::excess_1d(x[j + lo], x[c], zc, a);
    gauss::terms(S.D.data(), nullptr, 0.0, m, S.e.data());
    S.A.resize(m * k), S.b.resize(m);
    for (std::int64_t j = 0; j < m; ++j) {
        const double sq = std::sqrt(S.e[j]), t = gauss::z_of(x[j + lo], x0, a);
        double v = sq;
        for (std::int64_t l = 0; l < k; ++l) S.A[l * m + j] = v, v *= t;
        S.b[j] = sq * y[j + lo];
    }
    qr_pivoted(S.A.data(), S.b.data(), m, k, static_cast<double>(std::max(m, k)) * kU, q);
    if (q.rank < k) return kNaN;
    S.tmp.assign(q.qtb.begin(), q.qtb.end());
    back_solve(q, S.tmp.data());
    for (std::int64_t i = 0; i < k; ++i)
        if (q.perm[i] == r) return S.tmp[i];
    return kNaN;
}

// The quartic fits on N blocks of the sorted points (R's blkest and cp): RSS and, want, sum m'' m'''' over the points.
// The quartic is fitted in s = (x - mid) / half of the block (the same fit, without the powers of x); false when a
// block has no full rank.
bool block_fits(
    const double* x, const double* y, std::int64_t n, std::int64_t N, double& rss, double* th24, Scratch& S, QR& q
) {
    const std::int64_t idiv = n / N;
    rss = 0;
    double t24 = 0;
    for (std::int64_t b = 0; b < N; ++b) {
        const std::int64_t lo = b * idiv, hi = b == N - 1 ? n : (b + 1) * idiv, m = hi - lo;
        const double mid = 0.5 * (x[lo] + x[hi - 1]), half = 0.5 * (x[hi - 1] - x[lo]);
        if (!(half > 0) || m < 5) return false;
        S.A.resize(m * 5), S.b.resize(m);
        for (std::int64_t j = 0; j < m; ++j) {
            const double s = (x[lo + j] - mid) / half;
            double v = 1.0;
            for (int l = 0; l < 5; ++l) S.A[l * m + j] = v, v *= s;
            S.b[j] = y[lo + j];
        }
        qr_pivoted(S.A.data(), S.b.data(), m, 5, static_cast<double>(m) * kU, q);
        if (q.rank < 5) return false;
        double cf[5], co[5];
        for (int i = 0; i < 5; ++i) co[i] = q.qtb[i];
        back_solve(q, co);
        for (int i = 0; i < 5; ++i) cf[q.perm[i]] = co[i];
        const double h2 = half * half;
        for (std::int64_t j = 0; j < m; ++j) {
            const double s = (x[lo + j] - mid) / half;
            const double fit = cf[0] + s * (cf[1] + s * (cf[2] + s * (cf[3] + s * cf[4])));
            const double r = y[lo + j] - fit;
            rss += r * r;
            const double ddm = (2.0 * cf[2] + s * (6.0 * cf[3] + s * 12.0 * cf[4])) / h2;
            const double d4 = 24.0 * cf[4] / (h2 * h2);
            t24 += ddm * d4;
        }
    }
    if (th24) *th24 = t24;
    return true;
}

// KernSmooth's dpill computed without binning: trim 1% of the sorted points at each end, choose the number of
// quartic blocks by Mallows' C_p, estimate sigma^2 and theta_24 from the blocks, theta_22 from a local cubic's
// second derivative (bandwidth gamseh) at the points inside the middle 90% of the range, sigma^2 again from a local
// line (bandwidth lamseh) as RSS / (n - 2 tr S + tr S^T S), and combine.
std::pair<double, int> dpill(const double* x0, const double* y0, std::int64_t n0) {
    std::vector<std::int64_t> ord(n0);
    std::iota(ord.begin(), ord.end(), std::int64_t{0});
    std::sort(ord.begin(), ord.end(), [&](std::int64_t a, std::int64_t b) {
        if (x0[a] != x0[b]) return x0[a] < x0[b];
        if (y0[a] != y0[b]) return y0[a] < y0[b];
        return a < b;
    });
    const std::int64_t cut = static_cast<std::int64_t>(std::floor(0.01 * static_cast<double>(n0)));
    const std::int64_t n = n0 - 2 * cut;
    if (n < 6) return {kNaN, 1};
    std::vector<double> x(n), y(n);
    for (std::int64_t j = 0; j < n; ++j) x[j] = x0[ord[cut + j]], y[j] = y0[ord[cut + j]];
    const double a = x[0], b = x[n - 1], range = b - a;
    if (!(range > 0)) return {kNaN, 2};
    const double nn = static_cast<double>(n);
    Scratch S;
    QR q;
    const std::int64_t nmax = std::max<std::int64_t>(std::min<std::int64_t>(n / 20, 5), 1);
    std::vector<double> rss(nmax + 1);
    for (std::int64_t N = 1; N <= nmax; ++N)
        if (!block_fits(x.data(), y.data(), n, N, rss[N], nullptr, S, q)) return {kNaN, 3};
    std::int64_t nval = 1;
    double cpbest = kInf;
    for (std::int64_t N = 1; N <= nmax; ++N) {
        const double cp =
            (rss[N] / rss[nmax]) * (nn - 5.0 * static_cast<double>(nmax)) - (nn - 10.0 * static_cast<double>(N));
        if (cp < cpbest) cpbest = cp, nval = N;
    }
    double r, th24;
    block_fits(x.data(), y.data(), n, nval, r, &th24, S, q);
    const double sigsq = r / (nn - 5.0 * static_cast<double>(nval));
    th24 /= nn;
    const double sqpi = std::sqrt(M_PI);
    double gam = sigsq * range / (std::fabs(th24) * nn);
    if (th24 < 0) gam = std::pow(3.0 * gam / (8.0 * sqpi), 1.0 / 7.0);
    else if (th24 > 0) gam = std::pow(15.0 * gam / (16.0 * sqpi), 1.0 / 7.0);
    else return {kNaN, 4};
    if (!(gam > 0) || !std::isfinite(gam)) return {kNaN, 4};
    const double glo = a + 0.05 * range, ghi = b - 0.05 * range;
    std::vector<double> dd(n, 0.0);
    std::int64_t count = 0;
    for (std::int64_t i = 0; i < n; ++i) count += x[i] >= glo && x[i] <= ghi;
    std::vector<std::int64_t> pts;
    pts.reserve(count);
    for (std::int64_t i = 0; i < n; ++i)
        if (x[i] >= glo && x[i] <= ghi) pts.push_back(i);
    for_points(count, n * 40, [&](std::int64_t k, Scratch& Sk, Local&, Local&) {
        QR qk;
        const std::int64_t i = pts[k];
        const double c2 = local_poly_coef(x.data(), y.data(), n, x[i], gam, 3, 2, Sk, qk);
        dd[k] = 2.0 * c2 / (gam * gam);
    });
    const double th22 = gauss::sum_of(count, [&](std::int64_t k) { return dd[k] * dd[k]; }) / nn;
    if (!(th22 > 0) || !std::isfinite(th22)) return {kNaN, 4};
    double c3k = 0.5 + 2.0 * std::sqrt(2.0) - (4.0 / 3.0) * std::sqrt(3.0);
    c3k = std::pow(4.0 * c3k / std::sqrt(2.0 * M_PI), 1.0 / 9.0);
    const double lam = c3k * std::pow(sigsq * sigsq * range / ((th22 * nn) * (th22 * nn)), 1.0 / 9.0);
    if (!(lam > 0) || !std::isfinite(lam)) return {kNaN, 4};
    KData K;
    K.n = n, K.p = 1, K.X = x.data(), K.y = y.data(), K.w = nullptr, K.T = gauss::kUnderflow;
    const Axis ax = gauss::axis_of(lam);
    std::vector<double> res2(n), sii(n), rown(n);
    for_points(n, n * 16, [&](std::int64_t i, Scratch& Sk, Local& L, Local&) {
        local_fit(K, &ax, 1, &x[i], -1, Sk, L);
        double li = 0;
        rown[i] = hat_norm2(L, 1, Sk, i, &li);
        sii[i] = li;
        const double d = y[i] - L.value;
        res2[i] = d * d;
    });
    const double sn = gauss::sum(res2.data(), n);
    const double sd = nn - 2.0 * gauss::sum(sii.data(), n) + gauss::sum(rown.data(), n);
    const double sig2 = sn / sd;
    const double h = std::pow(sig2 * range / (2.0 * sqpi * th22 * nn), 0.2);
    if (!(h > 0) || !std::isfinite(h)) return {kNaN, 4};
    return {h, 0};
}

// ---- bindings ----------------------------------------------------------------------------------------------------

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

// The sorted points of positive weight (decision SM84) and the original index of each.
struct Sorted {
    std::vector<double> X, y, w;
    std::vector<std::int64_t> orig;
    bool weighted = false;
};

void sort_points(const double* X, const double* y, const double* w, std::int64_t n, std::int64_t p, Sorted& s) {
    std::vector<std::int64_t> ord;
    ord.reserve(n);
    for (std::int64_t j = 0; j < n; ++j)
        if (!w || w[j] > 0) ord.push_back(j);
    std::sort(ord.begin(), ord.end(), [&](std::int64_t a, std::int64_t b) {
        for (std::int64_t l = 0; l < p; ++l)
            if (X[a * p + l] != X[b * p + l]) return X[a * p + l] < X[b * p + l];
        if (y[a] != y[b]) return y[a] < y[b];
        if (w && w[a] != w[b]) return w[a] < w[b];
        return a < b;
    });
    const std::int64_t m = static_cast<std::int64_t>(ord.size());
    s.X.resize(m * p), s.y.resize(m), s.orig = ord, s.weighted = w != nullptr;
    if (w) s.w.resize(m);
    for (std::int64_t k = 0; k < m; ++k) {
        for (std::int64_t l = 0; l < p; ++l) s.X[k * p + l] = X[ord[k] * p + l];
        s.y[k] = y[ord[k]];
        if (w) s.w[k] = w[ord[k]];
    }
}

KData data_of(const Sorted& s, std::int64_t p) {
    KData K;
    K.p = p;
    K.n = static_cast<std::int64_t>(s.y.size());
    K.X = s.X.data(), K.y = s.y.data(), K.w = s.weighted ? s.w.data() : nullptr;
    double wr = 1.0;
    if (K.w) {
        const auto [mn, mx] = std::minmax_element(s.w.begin(), s.w.end());
        wr = *mx / *mn;
    }
    K.T = gauss::cut_threshold(K.n, wr);
    return K;
}

nb::object kreg_fit(
    nb::handle X_obj, nb::handle y_obj, nb::handle w_obj, int degree, nb::handle bw_obj, int criterion
) {
    const NDArray& Xa = mat_arg(X_obj, "X");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray& ya = vec_arg(y_obj, "y", n);
    const NDArray* wa = w_obj.is_none() ? nullptr : &vec_arg(w_obj, "weights", n);
    if (degree != 0 && degree != 1) throw nb::value_error("'degree' must be 0 or 1");
    std::vector<double> bw;
    if (!bw_obj.is_none()) {
        const NDArray& ba = vec_arg(bw_obj, "bw", p);
        bw.assign(dptr(ba), dptr(ba) + p);
    } else if (criterion != 1 && criterion != 2) {
        throw nb::value_error("'criterion' must be 1 or 2");
    }
    const double *X = dptr(Xa), *y = dptr(ya), *w = wa ? dptr(*wa) : nullptr;
    NDArray fit_o = out1(n), lev_o = out1(n), bw_o = out1(p);
    Sorted s;
    Fit F;
    Choice ch;
    std::vector<double> zero_fit;
    std::vector<std::int64_t> zeros;
    {
        const parallel::Unlocked unlocked(n * n >= kParallelThreshold);
        sort_points(X, y, w, n, p, s);
        const KData K = data_of(s, p);
        if (bw.empty()) {
            const std::vector<double> href = reference_bandwidth(K);
            if (href.empty()) {
                ch.status = -4;
            } else {
                ch = p == 1 ? search_1d(K, href, degree, criterion) : search_nd(K, href, degree, criterion);
            }
            bw = ch.bw;
        }
        if (!bw.empty()) {
            fit_all(K, bw, degree, F);
            for (std::int64_t j = 0; j < n; ++j)
                if (w && !(w[j] > 0)) zeros.push_back(j);
            zero_fit.resize(zeros.size());
            std::vector<Axis> ax(p);
            for (std::int64_t l = 0; l < p; ++l) ax[l] = gauss::axis_of(bw[l]);
            for_points(
                static_cast<std::int64_t>(zeros.size()), point_cost(K, degree),
                [&](std::int64_t k, Scratch& S, Local& L, Local&) {
                    local_fit(K, ax.data(), degree, X + zeros[k] * p, -1, S, L);
                    zero_fit[k] = L.value;
                }
            );
            double *fo = dout(fit_o), *lo = dout(lev_o);
            for (std::size_t k = 0; k < s.orig.size(); ++k) fo[s.orig[k]] = F.fitted[k], lo[s.orig[k]] = F.leverage[k];
            for (std::size_t k = 0; k < zeros.size(); ++k) fo[zeros[k]] = zero_fit[k], lo[zeros[k]] = 0.0;
            for (std::int64_t l = 0; l < p; ++l) dout(bw_o)[l] = bw[l];
        }
    }
    if (bw.empty()) {
        return nb::make_tuple(
            nb::none(), nb::none(), kNaN, kNaN, kNaN, kNaN, nb::none(), ch.status, false, ch.nfev, nb::none(),
            nb::none(), nb::none()
        );
    }
    const std::int64_t m = static_cast<std::int64_t>(s.y.size());
    NDArray xs_o = out2(m, p), ys_o = out1(m);
    NDArray ws_o = w ? out1(m) : NDArray{};
    std::copy(s.X.begin(), s.X.end(), dout(xs_o));
    std::copy(s.y.begin(), s.y.end(), dout(ys_o));
    if (w) std::copy(s.w.begin(), s.w.end(), dout(ws_o));
    nb::object ws_obj = w ? wrap(std::move(ws_o)) : nb::none();
    return nb::make_tuple(
        wrap(std::move(fit_o)), wrap(std::move(lev_o)), F.df, F.rss_sum, F.cv_value, F.aicc, wrap(std::move(bw_o)),
        ch.status, ch.converged, ch.nfev, wrap(std::move(xs_o)), wrap(std::move(ys_o)), ws_obj
    );
}

nb::object kreg_predict(
    nb::handle xs_obj, nb::handle ys_obj, nb::handle ws_obj, nb::handle bw_obj, int degree, nb::handle q_obj, int what
) {
    const NDArray& Xa = mat_arg(xs_obj, "Xs");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray& ya = vec_arg(ys_obj, "ys", n);
    const NDArray* wa = ws_obj.is_none() ? nullptr : &vec_arg(ws_obj, "ws", n);
    const NDArray& ba = vec_arg(bw_obj, "bw", p);
    const NDArray& qa = mat_arg(q_obj, "X_new");
    if (qa.shape[1] != p) throw nb::value_error("'X_new' has the wrong number of columns");
    if (degree != 0 && degree != 1) throw nb::value_error("'degree' must be 0 or 1");
    const std::int64_t m = qa.shape[0];
    NDArray val_o = what == 2 ? out2(m, p) : out1(m);
    NDArray nrm_o = what == 1 ? out1(m) : NDArray{};
    {
        const parallel::Unlocked unlocked(m * n >= kParallelThreshold);
        Sorted s;
        s.X.assign(dptr(Xa), dptr(Xa) + n * p);
        s.y.assign(dptr(ya), dptr(ya) + n);
        s.weighted = wa != nullptr;
        if (wa) s.w.assign(dptr(*wa), dptr(*wa) + n);
        KData K = data_of(s, p);
        std::vector<Axis> ax(p);
        std::vector<double> h(dptr(ba), dptr(ba) + p);
        for (std::int64_t l = 0; l < p; ++l) ax[l] = gauss::axis_of(h[l]);
        TreeHold th;
        attach_tree(K, ax, degree, th);
        const double* q = dptr(qa);
        double* vo = dout(val_o);
        double* no = what == 1 ? dout(nrm_o) : nullptr;
        for_points(m, point_cost(K, degree), [&](std::int64_t i, Scratch& S, Local& L, Local&) {
            local_fit(K, ax.data(), degree, q + i * p, -1, S, L);
            if (what == 2) {
                for (std::int64_t l = 0; l < p; ++l) vo[i * p + l] = L.ok ? L.beta[l] / h[l] : kNaN;
                return;
            }
            vo[i] = L.value;
            if (no) no[i] = hat_norm2(L, p, S);
        });
    }
    if (what == 1) return nb::make_tuple(wrap(std::move(val_o)), wrap(std::move(nrm_o)));
    return wrap(std::move(val_o));
}

double kreg_criterion(
    nb::handle X_obj, nb::handle y_obj, nb::handle w_obj, int degree, nb::handle bw_obj, int criterion
) {
    const NDArray& Xa = mat_arg(X_obj, "X");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray& ya = vec_arg(y_obj, "y", n);
    const NDArray* wa = w_obj.is_none() ? nullptr : &vec_arg(w_obj, "weights", n);
    const NDArray& ba = vec_arg(bw_obj, "bw", p);
    std::vector<double> bw(dptr(ba), dptr(ba) + p);
    const parallel::Unlocked unlocked(n * n >= kParallelThreshold);
    Sorted s;
    sort_points(dptr(Xa), dptr(ya), wa ? dptr(*wa) : nullptr, n, p, s);
    const KData K = data_of(s, p);
    Fit F;
    fit_all(K, bw, degree, F);
    return criterion_of(F, criterion);
}

nb::object bw_dpill_core(nb::handle x_obj, nb::handle y_obj) {
    const NDArray& xa = vec_arg(x_obj, "x");
    const std::int64_t n = xa.shape[0];
    const NDArray& ya = vec_arg(y_obj, "y", n);
    std::pair<double, int> r;
    {
        const parallel::Unlocked unlocked(n * n >= kParallelThreshold);
        r = dpill(dptr(xa), dptr(ya), n);
    }
    return nb::make_tuple(r.first, r.second);
}

}  // namespace

void bind_kreg(nb::module_& m) {
    m.def(
        "kreg_fit", named("kreg_fit", &kreg_fit), nb::arg("X").none(), nb::arg("y").none(), nb::arg("w").none(),
        nb::arg("degree"), nb::arg("bw").none(), nb::arg("criterion")
    );  // fmt: skip
    m.def(
        "kreg_predict", named("kreg_predict", &kreg_predict), nb::arg("Xs").none(), nb::arg("ys").none(),
        nb::arg("ws").none(), nb::arg("bw").none(), nb::arg("degree"), nb::arg("X_new").none(), nb::arg("what")
    );  // fmt: skip
    m.def(
        "kreg_criterion", named("kreg_criterion", &kreg_criterion), nb::arg("X").none(), nb::arg("y").none(),
        nb::arg("w").none(), nb::arg("degree"), nb::arg("bw").none(), nb::arg("criterion")
    );  // fmt: skip
    m.def("bw_dpill", named("bw_dpill", &bw_dpill_core), nb::arg("x").none(), nb::arg("y").none());
}

}  // namespace sd::smooth
