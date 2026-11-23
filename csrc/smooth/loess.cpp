// loess (docs/design/smooth.md §3.2, §3.8; decisions SM28, SM29, SM95-SM111): R's local polynomial regression
// computed directly at every point (R's surface = 'direct'), in one or several variables, degree 0, 1 or 2, with
// R's symmetric (bisquare) iterations and the statistics trace L, delta1 and delta2.
//
// _core.smooth.loess_fit(X, y, w, degree, q, widen, iterations, normalize, path)
//     -> (fitted, leverage, robust_weights, df, rss, delta1, n_rank_deficient, Xs, ys, ws, rho, divisor)
//     X (n, p), y (n,), w (n,) or None: finite float64, C-contiguous, in the caller's order (NaN already dropped);
//     q = min(n, floor(n span + 1e-5)) >= the local polynomial's terms, widen = sqrt(span) when span > 1 (else 1),
//     iterations >= 1 fits (gaussian: 1), normalize (p >= 2: every column divided by R's 10% trimmed standard
//     deviation), path (-1: by shape; 0: scan every distance; 1: the KD tree's k nearest; tests only). The data are
//     scaled, then sorted by (coordinates, y, w, index) (decision SM96), so the result does not depend on the input
//     order. fitted, leverage and robust_weights come back in the input order; df = sum of the leverages, rss =
//     sum w r^2 and delta1 = sum over the rows of (I - L) of their squared norms, over the points of positive
//     weight in sorted order, all of the last fit's linear smoother; Xs, ys, ws (None without weights), rho (the
//     last robust weights) are the sorted data a result keeps, divisor the p column scales (ones when not
//     normalized). Error codes instead of a tuple: the int 1 (a column's trimmed spread is zero or not finite).
// _core.smooth.loess_predict(Xs, ys, ws, rho, degree, q, widen, divisor, X_new, norms, path) -> (values, norms or None)
//     the local polynomial at each row of X_new (m, p) (divided by divisor) with the prior weights rho w; NaN where
//     no point of the window has a positive weight (or X_new has a NaN); norms: sum l_j^2 / w_j of each hat row.
// _core.smooth.loess_two_delta(Xs, ys, ws, rho, degree, q, widen, path) -> delta2
//     tr(((I - L)(I - L)^T)^2) over the points of positive weight (n <= 4096 is the caller's rule): the dense rows of
//     I - L, then the squared Frobenius norm of their Gram matrix by a fixed blocking (decision SM103).
//
// The local fit (decision SM98): the window is the q nearest points (p = 1: the sorted window of smooth/local.hpp;
// p >= 2: the spatial area's Euclidean kernel distances, scanned or from the KD tree, the same set), its radius h the
// q-th distance (span > 1: the farthest times sqrt(span), SM29), the weights tricube (1 - (d / h)^3)^3 times the
// prior weights, the basis the monomials of u = (x - x0) / h up to the degree (1, u_a, then u_a u_b for a <= b). p = 1
// and degree 1 is lowess's closed form (fit_line: the same bits as lowess at it = 0). Otherwise: degree 0 is the
// weighted mean; degree 1, 2 the moment matrix's Cholesky with one correction from the residual (the corrected
// semi-normal equations), and Householder QR of the weighted design when the Cholesky fails or its diagonal ratio
// squared exceeds 2^32 (SM28). Rank (SM99): p = 1 lowers the degree to the window's distinct x minus 1 (exact); p >= 2
// drops a column whose remaining norm in the QR is at most max(m, t) eps times its norm (the regression area's rule),
// and such a point counts in n_rank_deficient. One positive weight: that point's y (l = 1); none: a data point keeps
// its y (leverage 1), a new point is NaN.
//
// Every point is one call of fit_point whatever task computes it, and the tasks are cut by size alone: the results
// have the same bits for every thread count (decision SM3). The robust step (R's lowesw) is serial and O(n).

#include "smooth/local.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "smooth/smooth.hpp"
#include "spatial/kdtree.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace sd::smooth {

namespace nb = nanobind;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 0x1p-52;
constexpr double kCholRatio2 = 0x1p32;  // (max L_ii / min L_ii)^2 above which the window goes to QR (SM28)
constexpr std::int64_t kBlock = 256;  // the moments' inner block (two-level sums)

std::int64_t terms_of(std::int64_t p, int degree) {
    if (degree == 0) return 1;
    if (degree == 1) return p + 1;
    return (p + 1) * (p + 2) / 2;
}

// ---- the problem and one task's scratch ----------------------------------------------------------------------

struct Problem {
    std::int64_t n = 0, p = 1, q = 1;
    int degree = 2;
    double widen = 1.0;
    const double* X = nullptr;  // n x p, sorted (p = 1: ascending)
    const double* y = nullptr;
    const double* w = nullptr;  // the user's weights (sorted) or null
    const double* pw = nullptr;  // prior weights w rho >= 0
    const spatial::KdTree* tree = nullptr;  // p >= 2 and the k-nearest path
};

struct Scratch {
    std::vector<double> dist, sel;
    std::vector<std::pair<double, std::int64_t>> knn;
    std::vector<std::int64_t> idx;  // the window's points of positive weight (sorted index, ascending)
    std::vector<double> v, yv, cols, r, lrow, A, sv, tmp, mt;
    std::vector<double> M, L, b, c, s, g;
    std::unique_ptr<spatial::KdSearch<spatial::KdEuclid>> search;
};

// One point's fit: the value and, when asked, the hat row lrow[k] at the window points idx[k] (scratch).
struct Fit {
    double value = kNaN;
    bool deficient = false;
    bool none = false;  // no point of positive weight: a data point keeps its y
    bool nan = false;  // a NaN point, or a distance overflowed
    // with the hat row asked for: l at the data point fitted, sum l_j^2 and sum l_j^2 / w_j (w: the user's weights)
    double lself = 0, ss = 0, ssw = 0;
    bool stats = false;  // the statistics are set (no row kept)
};

// ---- sums in a fixed order ------------------------------------------------------------------------------------

// sum_k a[k] b[k] in two lanes over pairs, blocks of kBlock added to the total (a function of m alone).
inline double dot(const double* a, const double* b, std::int64_t m) {
    float64x2_t tot = vdupq_n_f64(0.0);
    std::int64_t k = 0;
    while (k + 2 <= m) {
        const std::int64_t end = std::min(m & ~std::int64_t{1}, k + kBlock);
        float64x2_t acc = vdupq_n_f64(0.0);
        for (; k + 2 <= end; k += 2) acc = vfmaq_f64(acc, vld1q_f64(a + k), vld1q_f64(b + k));
        tot = vaddq_f64(tot, acc);
    }
    double s = vgetq_lane_f64(tot, 0) + vgetq_lane_f64(tot, 1);
    if (k < m) s = std::fma(a[k], b[k], s);
    return s;
}

// The moment matrix M (t x t, both triangles) and b = sum v phi y of the window: cols holds the t basis columns
// (column-major, m each), v the weights, y the values. Two lanes over pairs of points, blocks of kBlock.
template <int T>
void moments_t(const double* cols, const double* v, const double* y, std::int64_t m, double* M, double* b) {
    constexpr int NM = T * (T + 1) / 2;
    float64x2_t tm[NM], tb[T];
    for (int e = 0; e < NM; ++e) tm[e] = vdupq_n_f64(0.0);
    for (int e = 0; e < T; ++e) tb[e] = vdupq_n_f64(0.0);
    std::int64_t k = 0;
    const std::int64_t m2 = m & ~std::int64_t{1};
    while (k < m2) {
        const std::int64_t end = std::min(m2, k + kBlock);
        float64x2_t am[NM], ab[T];
        for (int e = 0; e < NM; ++e) am[e] = vdupq_n_f64(0.0);
        for (int e = 0; e < T; ++e) ab[e] = vdupq_n_f64(0.0);
        for (; k < end; k += 2) {
            const float64x2_t vv = vld1q_f64(v + k), yy = vld1q_f64(y + k);
            float64x2_t ph[T];
            for (int a = 0; a < T; ++a) ph[a] = vld1q_f64(cols + a * m + k);
            int e = 0;
            for (int a = 0; a < T; ++a) {
                const float64x2_t va = vmulq_f64(vv, ph[a]);
                ab[a] = vfmaq_f64(ab[a], va, yy);
                for (int c = a; c < T; ++c) am[e] = vfmaq_f64(am[e], va, ph[c]), ++e;
            }
        }
        for (int e = 0; e < NM; ++e) tm[e] = vaddq_f64(tm[e], am[e]);
        for (int e = 0; e < T; ++e) tb[e] = vaddq_f64(tb[e], ab[e]);
    }
    int e = 0;
    for (int a = 0; a < T; ++a) {
        b[a] = vgetq_lane_f64(tb[a], 0) + vgetq_lane_f64(tb[a], 1);
        for (int c = a; c < T; ++c, ++e) M[a * T + c] = vgetq_lane_f64(tm[e], 0) + vgetq_lane_f64(tm[e], 1);
    }
    if (k < m) {
        e = 0;
        for (int a = 0; a < T; ++a) {
            const double va = v[k] * cols[a * m + k];
            b[a] = std::fma(va, y[k], b[a]);
            for (int c = a; c < T; ++c, ++e) M[a * T + c] = std::fma(va, cols[c * m + k], M[a * T + c]);
        }
    }
    for (int a = 0; a < T; ++a)
        for (int c = 0; c < a; ++c) M[a * T + c] = M[c * T + a];
}

// The same for any t (the scalar lanes of the template, in memory).
void moments_any(
    const double* cols,
    const double* v,
    const double* y,
    std::int64_t m,
    std::int64_t t,
    double* M,
    double* b,
    std::vector<double>& tmp
) {
    const std::int64_t nm = t * (t + 1) / 2;
    tmp.assign(static_cast<std::size_t>(4 * (nm + t)), 0.0);
    double *tm = tmp.data(), *tb = tm + 2 * nm, *am = tb + 2 * t, *ab = am + 2 * nm;
    std::int64_t k = 0;
    const std::int64_t m2 = m & ~std::int64_t{1};
    while (k < m2) {
        const std::int64_t end = std::min(m2, k + kBlock);
        std::fill(am, am + 2 * nm + 2 * t, 0.0);
        for (; k < end; k += 2) {
            for (int l = 0; l < 2; ++l) {
                const std::int64_t j = k + l;
                std::int64_t e = 0;
                for (std::int64_t a = 0; a < t; ++a) {
                    const double va = v[j] * cols[a * m + j];
                    ab[2 * a + l] = std::fma(va, y[j], ab[2 * a + l]);
                    for (std::int64_t c = a; c < t; ++c, ++e)
                        am[2 * e + l] = std::fma(va, cols[c * m + j], am[2 * e + l]);
                }
            }
        }
        for (std::int64_t e = 0; e < 2 * nm; ++e) tm[e] += am[e];
        for (std::int64_t e = 0; e < 2 * t; ++e) tb[e] += ab[e];
    }
    std::int64_t e = 0;
    for (std::int64_t a = 0; a < t; ++a) {
        b[a] = tb[2 * a] + tb[2 * a + 1];
        for (std::int64_t c = a; c < t; ++c, ++e) M[a * t + c] = tm[2 * e] + tm[2 * e + 1];
    }
    if (k < m) {
        e = 0;
        for (std::int64_t a = 0; a < t; ++a) {
            const double va = v[k] * cols[a * m + k];
            b[a] = std::fma(va, y[k], b[a]);
            for (std::int64_t c = a; c < t; ++c, ++e) M[a * t + c] = std::fma(va, cols[c * m + k], M[a * t + c]);
        }
    }
    for (std::int64_t a = 0; a < t; ++a)
        for (std::int64_t c = 0; c < a; ++c) M[a * t + c] = M[c * t + a];
}

void moments(
    const double* cols,
    const double* v,
    const double* y,
    std::int64_t m,
    std::int64_t t,
    double* M,
    double* b,
    std::vector<double>& tmp
) {
    switch (t) {
        case 2: return moments_t<2>(cols, v, y, m, M, b);
        case 3: return moments_t<3>(cols, v, y, m, M, b);
        case 4: return moments_t<4>(cols, v, y, m, M, b);
        case 6: return moments_t<6>(cols, v, y, m, M, b);
        default: return moments_any(cols, v, y, m, t, M, b, tmp);
    }
}

// out[k] = phi_k^T c for every window point k: c_0 phi_k0, then fma(c_a, phi_ka, .) for a = 1, 2, ... (column by
// column, so the loops run over k in NEON lanes; every element takes the same operations in the same order).
inline void phi_dots(const double* cols, std::int64_t m, std::int64_t t, const double* c, double* out) {
    for (std::int64_t k = 0; k < m; ++k) out[k] = c[0] * cols[k];
    for (std::int64_t a = 1; a < t; ++a) {
        const double ca = c[a];
        const double* col = cols + a * m;
        for (std::int64_t k = 0; k < m; ++k) out[k] = std::fma(ca, col[k], out[k]);
    }
}

// ---- Cholesky with one correction, and QR ----------------------------------------------------------------------

// L (lower, t x t) of M; false when a pivot is not positive or the diagonal ratio squared exceeds 2^32 (SM28).
bool cholesky(const double* M, std::int64_t t, double* L) {
    double dmax = 0, dmin = kInf;
    for (std::int64_t i = 0; i < t; ++i) {
        for (std::int64_t j = 0; j <= i; ++j) {
            double s = M[i * t + j];
            for (std::int64_t k = 0; k < j; ++k) s -= L[i * t + k] * L[j * t + k];
            if (i == j) {
                if (!(s > 0) || !std::isfinite(s)) return false;
                L[i * t + i] = std::sqrt(s);
                dmax = std::max(dmax, L[i * t + i]), dmin = std::min(dmin, L[i * t + i]);
            } else {
                L[i * t + j] = s / L[j * t + j];
            }
        }
    }
    const double ratio = dmax / dmin;
    return ratio * ratio <= kCholRatio2;
}

// x = M^-1 x by L L^T.
void chol_solve(const double* L, std::int64_t t, double* x) {
    for (std::int64_t i = 0; i < t; ++i) {
        double s = x[i];
        for (std::int64_t k = 0; k < i; ++k) s -= L[i * t + k] * x[k];
        x[i] = s / L[i * t + i];
    }
    for (std::int64_t i = t - 1; i >= 0; --i) {
        double s = x[i];
        for (std::int64_t k = i + 1; k < t; ++k) s -= L[k * t + i] * x[k];
        x[i] = s / L[i * t + i];
    }
}

// The local least squares of the window (m points of weight v > 0, the t columns cols, values yv) by the moment
// Cholesky and one correction (value = the intercept), or Householder QR (value = l^T y with the hat row from Q; the
// QR path always fills lrow). With `hat`, lrow[k] = l_k. tol: the QR's aliasing rule (0: only an exactly zero
// column is dropped; the reflectors were applied to each later column as they were made, so a dropped column takes
// no part). Returns whether a column was dropped.
bool solve_window(Scratch& S, std::int64_t m, std::int64_t t, double tol, bool hat, double& value) {
    const double *cols = S.cols.data(), *v = S.v.data(), *yv = S.yv.data();
    S.M.resize(static_cast<std::size_t>(t * t));
    S.L.resize(static_cast<std::size_t>(t * t));
    S.b.resize(static_cast<std::size_t>(t));
    S.c.resize(static_cast<std::size_t>(t));
    S.s.resize(static_cast<std::size_t>(t));
    S.g.resize(static_cast<std::size_t>(t));
    moments(cols, v, yv, m, t, S.M.data(), S.b.data(), S.mt);
    if (cholesky(S.M.data(), t, S.L.data())) {
        double *c = S.c.data(), *s = S.s.data();
        std::copy(S.b.begin(), S.b.end(), c);
        chol_solve(S.L.data(), t, c);
        // one correction: s = sum v phi (y - phi^T c), c += M^-1 s
        S.r.resize(static_cast<std::size_t>(m));
        double* r = S.r.data();
        phi_dots(cols, m, t, c, r);
        for (std::int64_t k = 0; k < m; ++k) r[k] = v[k] * (yv[k] - r[k]);
        for (std::int64_t a = 0; a < t; ++a) s[a] = dot(cols + a * m, r, m);
        chol_solve(S.L.data(), t, s);
        for (std::int64_t a = 0; a < t; ++a) c[a] += s[a];
        value = c[0];
        if (hat) {
            double* g = S.g.data();
            std::fill(g, g + t, 0.0);
            g[0] = 1.0;
            chol_solve(S.L.data(), t, g);
            S.lrow.resize(static_cast<std::size_t>(m));
            phi_dots(cols, m, t, g, S.lrow.data());
            for (std::int64_t k = 0; k < m; ++k) S.lrow[k] *= v[k];
        }
        return false;
    }
    // Householder QR of A = diag(sqrt v) [cols], unpivoted; a column whose remaining norm is at most tol times its
    // own norm is dropped (it receives no reflector and gives none)
    S.A.resize(static_cast<std::size_t>(m * t));
    S.sv.resize(static_cast<std::size_t>(m));
    double *A = S.A.data(), *sv = S.sv.data();
    for (std::int64_t k = 0; k < m; ++k) sv[k] = std::sqrt(v[k]);
    for (std::int64_t a = 0; a < t; ++a)
        for (std::int64_t k = 0; k < m; ++k) A[a * m + k] = sv[k] * cols[a * m + k];
    std::vector<std::int64_t> kept;
    std::vector<double> tau;
    kept.reserve(static_cast<std::size_t>(t));
    std::int64_t row = 0;
    for (std::int64_t a = 0; a < t; ++a) {
        double* col = A + a * m;
        double n0 = 0, nr = 0;
        for (std::int64_t k = 0; k < m; ++k) n0 += col[k] * col[k];
        for (std::int64_t k = row; k < m; ++k) nr += col[k] * col[k];
        n0 = std::sqrt(n0), nr = std::sqrt(nr);
        if (row >= m || !(nr > 0) || !(nr > tol * n0)) continue;
        const double alpha = col[row] > 0 ? -nr : nr;
        const double v0 = col[row] - alpha;
        const double ta = -v0 / alpha;
        for (std::int64_t k = row + 1; k < m; ++k) col[k] /= v0;
        const auto apply = [&](double* x) {
            double d = x[row];
            for (std::int64_t k = row + 1; k < m; ++k) d += col[k] * x[k];
            d *= ta;
            x[row] -= d;
            for (std::int64_t k = row + 1; k < m; ++k) x[k] -= d * col[k];
        };
        for (std::int64_t a2 = a + 1; a2 < t; ++a2) apply(A + a2 * m);
        col[row] = alpha;
        kept.push_back(a);
        tau.push_back(ta);
        ++row;
    }
    const std::int64_t rk = static_cast<std::int64_t>(kept.size());
    // R (rk x rk): R[i][jj] = A[kept[jj] m + i], i <= jj
    const auto R = [&](std::int64_t i, std::int64_t jj) { return A[kept[jj] * m + i]; };
    // the hat row l = W^(1/2) Q z with z = R^-T e0 (the minimum-norm problem Phi^T l = e0: no R^-1, so no second
    // factor kappa), and the value l^T y; the column of ones is never dropped (it comes first and v > 0)
    double* z = S.g.data();
    for (std::int64_t i = 0; i < rk; ++i) {
        double s = i == 0 ? 1.0 : 0.0;
        for (std::int64_t c2 = 0; c2 < i; ++c2) s -= R(c2, i) * z[c2];
        z[i] = s / R(i, i);
    }
    S.lrow.assign(static_cast<std::size_t>(m), 0.0);
    double* l = S.lrow.data();
    for (std::int64_t i = 0; i < rk; ++i) l[i] = z[i];
    for (std::int64_t i = rk - 1; i >= 0; --i) {  // Q = H_0 H_1 ... H_(rk-1)
        const double* col = A + kept[i] * m;
        double d = l[i];
        for (std::int64_t k = i + 1; k < m; ++k) d += col[k] * l[k];
        d *= tau[i];
        l[i] -= d;
        for (std::int64_t k = i + 1; k < m; ++k) l[k] -= d * col[k];
    }
    for (std::int64_t k = 0; k < m; ++k) l[k] *= sv[k];
    value = dot(l, yv, m);
    return rk < t;
}

// ---- the window ------------------------------------------------------------------------------------------------

struct Window {
    double h = 0, sc = 1, hs = 0, inv = 0;
    bool nan = false;
};

// The tricube factor of an inside point at distance dist (the operations of local.hpp's tricube_of).
inline double tricube_dist(double dist, double sc, double hs, double inv) {
    const double ds = dist * sc;
    const double ad = ds * inv;
    const double t = ((hs - ds) * inv) * ((1.0 + ad) + ad * ad);
    return t * t * t;
}

// The weights v_j = tricube pw_j (0 outside the radius h) and the coordinates d_j = ((x_j - x0) sc) inv of the sorted
// points [lo, hi), two lanes at a time: the operations of local.hpp's tricube_of, so the bits of window_weight.
void window_weights_1d(
    const double* x,
    const double* pw,
    std::int64_t lo,
    std::int64_t hi,
    double x0,
    double h,
    double sc,
    double inv,
    const Edge& E,
    double* v,
    double* d
) {
    const float64x2_t vx0 = vdupq_n_f64(x0), vh = vdupq_n_f64(h), vsc = vdupq_n_f64(sc), vinv = vdupq_n_f64(inv);
    const float64x2_t vxl = vdupq_n_f64(E.xl), vxr = vdupq_n_f64(E.xr), vgl = vdupq_n_f64(E.gl);
    const float64x2_t vgr = vdupq_n_f64(E.gr), one = vdupq_n_f64(1.0), zero = vdupq_n_f64(0.0);
    std::int64_t j = lo;
    for (; j + 2 <= hi; j += 2) {
        const float64x2_t xx = vld1q_f64(x + j);
        const float64x2_t diff = vsubq_f64(xx, vx0);
        const float64x2_t ad = vmulq_f64(vabsq_f64(vmulq_f64(diff, vsc)), vinv);
        const float64x2_t right = vaddq_f64(vmulq_f64(vsubq_f64(vxr, xx), vsc), vgr);
        const float64x2_t left = vaddq_f64(vmulq_f64(vsubq_f64(xx, vxl), vsc), vgl);
        const float64x2_t gap = vmaxq_f64(vbslq_f64(vcgezq_f64(diff), right, left), zero);
        const float64x2_t t = vmulq_f64(vmulq_f64(gap, vinv), vaddq_f64(vaddq_f64(one, ad), vmulq_f64(ad, ad)));
        float64x2_t w = vmulq_f64(vmulq_f64(vmulq_f64(t, t), t), vld1q_f64(pw + j));
        w = vbslq_f64(vcltq_f64(vabsq_f64(diff), vh), w, zero);
        vst1q_f64(v + (j - lo), w);
        vst1q_f64(d + (j - lo), vmulq_f64(vmulq_f64(diff, vsc), vinv));
    }
    for (; j < hi; ++j) {
        const bool in = std::fabs(x[j] - x0) < h;
        v[j - lo] = in ? tricube_of(x[j], x0, sc, inv, E) * pw[j] : 0.0;
        d[j - lo] = ((x[j] - x0) * sc) * inv;
    }
}

// p = 1: the q nearest of the sorted x (the window of smooth/local.hpp), the points of positive weight within the
// radius into S.idx and S.v; the coordinates d = (x - x0) / h into S.tmp (aligned with idx). h = 0: the points at x0
// with v = pw.
Window gather_1d(const Problem& P, double x0, Scratch& S) {
    Window W;
    S.idx.clear(), S.v.clear(), S.tmp.clear();
    const double* x = P.X;
    const std::int64_t n = P.n, k = P.q;
    const std::int64_t l = window_left(x, n, k, x0), r = l + k;
    const double hfar = std::max(x0 - x[l], x[r - 1] - x0);
    const double h = P.widen != 1.0 ? hfar * P.widen : hfar;
    W.h = h;
    if (!(h < kInf)) {
        W.nan = true;
        return W;
    }
    if (h == 0) {
        std::int64_t t0 = l, t1 = r;
        while (t0 > 0 && x[t0 - 1] == x0) --t0;
        while (t1 < n && x[t1] == x0) ++t1;
        for (std::int64_t j = t0; j < t1; ++j)
            if (P.pw[j] > 0) S.idx.push_back(j), S.v.push_back(P.pw[j]), S.tmp.push_back(0.0);
        return W;
    }
    W.sc = h < kTinyRadius ? kRadiusScale : 1.0;
    W.hs = P.widen != 1.0 ? (hfar * W.sc) * P.widen : h * W.sc;  // as fit_line: widened after the scaling
    W.inv = 1.0 / W.hs;
    const Edge E = window_edge(x[l], x[r - 1], x0, W.sc, W.hs, P.widen == 1.0);
    const std::size_t len = static_cast<std::size_t>(r - l);
    S.idx.resize(len), S.v.resize(len), S.tmp.resize(len);
    window_weights_1d(x, P.pw, l, r, x0, h, W.sc, W.inv, E, S.v.data(), S.tmp.data());
    std::size_t m = 0;
    for (std::size_t k = 0; k < len; ++k) {  // keep the points of positive weight
        const double vk = S.v[k];
        S.idx[m] = l + static_cast<std::int64_t>(k), S.v[m] = vk, S.tmp[m] = S.tmp[k];
        m += vk > 0;
    }
    S.idx.resize(m), S.v.resize(m), S.tmp.resize(m);
    return W;
}

// dist[j] = the spatial area's Euclidean kernel distance from x0 to row j of X (n x p): for p = 2 and 3 two rows at a
// time in NEON lanes with the kernel's own chain (z = x0_a - X_ja, t = fma(z, z, t) from 0 in axis order, then the
// square root where the sum is in the plain range, else the kernel's careful finish), so the kernel's bits.
void all_distances(
    const spatial::KdEuclid& eu, const double* x0, const double* X, std::int64_t n, std::int64_t p, double* dist
) {
    std::int64_t j = 0;
    if (p == 2 || p == 3) {
        const float64x2_t zero = vdupq_n_f64(0.0);
        for (; j + 2 <= n; j += 2) {
            float64x2_t t = zero;
            if (p == 2) {
                const float64x2x2_t r = vld2q_f64(X + j * 2);
                const float64x2_t z0 = vsubq_f64(vdupq_n_f64(x0[0]), r.val[0]);
                const float64x2_t z1 = vsubq_f64(vdupq_n_f64(x0[1]), r.val[1]);
                t = vfmaq_f64(vfmaq_f64(t, z0, z0), z1, z1);
            } else {
                const float64x2x3_t r = vld3q_f64(X + j * 3);
                const float64x2_t z0 = vsubq_f64(vdupq_n_f64(x0[0]), r.val[0]);
                const float64x2_t z1 = vsubq_f64(vdupq_n_f64(x0[1]), r.val[1]);
                const float64x2_t z2 = vsubq_f64(vdupq_n_f64(x0[2]), r.val[2]);
                t = vfmaq_f64(vfmaq_f64(vfmaq_f64(t, z0, z0), z1, z1), z2, z2);
            }
            const double s0 = vgetq_lane_f64(t, 0), s1 = vgetq_lane_f64(t, 1);
            if (spatial::KdEuclid::plain(s0) && spatial::KdEuclid::plain(s1)) {
                vst1q_f64(dist + j, vsqrtq_f64(t));
            } else {
                dist[j] = eu.finish(s0, x0, X + j * p);
                dist[j + 1] = eu.finish(s1, x0, X + (j + 1) * p);
            }
        }
    }
    for (; j < n; ++j) dist[j] = spatial::kd_distance(eu, x0, X + j * p);
}

// The q-th smallest of dist[0, n) (1 <= q <= n, no NaN): an order statistic is one value, so any exact selection
// gives the same bits. Above kSelectDirect points a sorted sample of kSample distances brackets the rank q (about
// three standard deviations of the sample quantile's rank on either side), one branch-free pass counts the points
// below the bracket and gathers those inside it, and nth_element runs on the bracket alone (a few percent of the
// points); when the bracket misses the rank, nth_element on every distance.
constexpr std::int64_t kSelectDirect = 2048;
constexpr std::int64_t kSample = 512;

double select_kth(const double* dist, std::int64_t n, std::int64_t q, std::vector<double>& sel) {
    if (n <= kSelectDirect) {
        sel.assign(dist, dist + n);
        std::nth_element(sel.begin(), sel.begin() + (q - 1), sel.end());
        return sel[static_cast<std::size_t>(q - 1)];
    }
    double sample[kSample];
    for (std::int64_t i = 0; i < kSample; ++i) sample[i] = dist[(i * n) / kSample + (n / kSample) / 2];
    std::sort(sample, sample + kSample);
    const double frac = static_cast<double>(q) / static_cast<double>(n);
    const double sd = std::sqrt(frac * (1.0 - frac) * static_cast<double>(kSample)) + 1.0;
    const double at = frac * static_cast<double>(kSample);
    const std::int64_t ilo = static_cast<std::int64_t>(std::floor(at - 3.0 * sd - 1.0));
    const std::int64_t ihi = static_cast<std::int64_t>(std::ceil(at + 3.0 * sd));
    const double lo = ilo < 0 ? -kInf : sample[ilo], hi = ihi >= kSample ? kInf : sample[ihi];
    sel.resize(static_cast<std::size_t>(n));
    double* out = sel.data();
    std::int64_t below = 0, inside = 0;
    for (std::int64_t j = 0; j < n; ++j) {
        const double d = dist[j];
        out[inside] = d;
        inside += (d >= lo) & (d <= hi);
        below += d < lo;
    }
    if (below < q && q <= below + inside) {
        const std::int64_t r = q - below - 1;
        std::nth_element(out, out + r, out + inside);
        return out[r];
    }
    sel.assign(dist, dist + n);
    std::nth_element(sel.begin(), sel.begin() + (q - 1), sel.end());
    return sel[static_cast<std::size_t>(q - 1)];
}

// p >= 2: the q nearest by the spatial area's Euclidean kernel (scanned, or the KD tree's k nearest: the same radius
// and the same set), the points of positive weight with distance < h into S.idx (ascending) and S.v.
Window gather_nd(const Problem& P, const spatial::KdEuclid& eu, const double* x0, Scratch& S) {
    Window W;
    S.idx.clear(), S.v.clear();
    const std::int64_t n = P.n, p = P.p, q = P.q;
    for (std::int64_t a = 0; a < p; ++a)
        if (!(x0[a] == x0[a])) {
            W.nan = true;
            return W;
        }
    double h = kNaN, hfar = kNaN;
    S.dist.resize(static_cast<std::size_t>(n));
    bool scan = true;
    if (P.tree && P.widen == 1.0) {
        S.search->run(x0, kInf, S.knn);
        h = S.knn[static_cast<std::size_t>(q - 1)].first;
        // h = 0: more than q points may sit at x0; the scan finds them all
        scan = h == 0;
        if (!scan) {
            for (const auto& e : S.knn) S.dist[static_cast<std::size_t>(e.second)] = e.first;
            S.idx.clear();
            for (const auto& e : S.knn)
                if (e.first < h) S.idx.push_back(e.second);
            std::sort(S.idx.begin(), S.idx.end());
        }
    }
    if (scan) {
        all_distances(eu, x0, P.X, n, p, S.dist.data());
        if (P.widen != 1.0) {
            hfar = *std::max_element(S.dist.begin(), S.dist.end());
            h = hfar * P.widen;
        } else {
            h = select_kth(S.dist.data(), n, q, S.sel);
        }
        S.idx.resize(static_cast<std::size_t>(n));
        std::int64_t cnt = 0;
        for (std::int64_t j = 0; j < n; ++j) {
            S.idx[cnt] = j;
            cnt += S.dist[j] < h;
        }
        S.idx.resize(static_cast<std::size_t>(cnt));
    }
    W.h = h;
    if (!(h < kInf)) {
        W.nan = true;
        return W;
    }
    if (h == 0) {  // the points at x0
        S.idx.clear();
        for (std::int64_t j = 0; j < n; ++j)
            if (S.dist[j] == 0 && P.pw[j] > 0) S.idx.push_back(j), S.v.push_back(P.pw[j]);
        return W;
    }
    W.sc = h < kTinyRadius ? kRadiusScale : 1.0;
    W.hs = P.widen != 1.0 ? (hfar * W.sc) * P.widen : h * W.sc;
    W.inv = 1.0 / W.hs;
    std::size_t out = 0;
    S.v.resize(S.idx.size());
    for (std::size_t e = 0; e < S.idx.size(); ++e) {
        const std::int64_t j = S.idx[e];
        if (!(P.pw[j] > 0)) continue;
        const double vj = tricube_dist(S.dist[j], W.sc, W.hs, W.inv) * P.pw[j];
        if (!(vj > 0)) continue;
        S.v[out] = vj;
        S.idx[out++] = j;
    }
    S.idx.resize(out), S.v.resize(out);
    return W;
}

// ---- one point -------------------------------------------------------------------------------------------------

enum Want { kValue = 0, kStats = 1, kRows = 2 };  // the value; its hat row's statistics; the hat row itself

// The weighted mean of the window as the fit (degree 0, or a lowered degree): the value, two lanes, and per `want`
// the hat row v_k / a (kRows) or its statistics (kStats: sum v^2 / a^2 and its weighted form).
double window_mean(const Problem& P, Scratch& S, int want, std::int64_t self, Fit& F) {
    const std::int64_t m = static_cast<std::int64_t>(S.idx.size());
    const double *v = S.v.data(), *yv = S.yv.data();
    float64x2_t va = vdupq_n_f64(0.0), vs = va, vq = va;
    std::int64_t k = 0;
    for (; k + 2 <= m; k += 2) {
        const float64x2_t vk = vld1q_f64(v + k);
        va = vaddq_f64(va, vk), vs = vfmaq_f64(vs, vk, vld1q_f64(yv + k)), vq = vfmaq_f64(vq, vk, vk);
    }
    double a = vgetq_lane_f64(va, 0) + vgetq_lane_f64(va, 1), s0 = vgetq_lane_f64(vs, 0) + vgetq_lane_f64(vs, 1);
    double q2 = vgetq_lane_f64(vq, 0) + vgetq_lane_f64(vq, 1);
    if (k < m) a += v[k], s0 = std::fma(v[k], yv[k], s0), q2 = std::fma(v[k], v[k], q2);
    if (want == kRows) {
        S.lrow.resize(static_cast<std::size_t>(m));
        for (std::int64_t j = 0; j < m; ++j) S.lrow[j] = v[j] / a;
    } else if (want == kStats) {
        F.ss = q2 / (a * a);
        if (P.w) {
            double sw = 0;
            for (std::int64_t j = 0; j < m; ++j) sw += (v[j] / a) * (v[j] / a) / P.w[S.idx[j]];
            F.ssw = sw;
        } else {
            F.ssw = F.ss;
        }
        F.lself = 0;
        if (self >= 0) {
            const auto it = std::lower_bound(S.idx.begin(), S.idx.end(), self);
            if (it != S.idx.end() && *it == self) F.lself = v[it - S.idx.begin()] / a;
        }
        F.stats = true;
    }
    return s0 / a;
}

// F.lself, F.ss, F.ss_w of the hat row S.idx / S.lrow (idx ascending): two lanes, a function of the row alone.
void row_stats(const Problem& P, const Scratch& S, std::int64_t self, Fit& F) {
    const std::int64_t m = static_cast<std::int64_t>(S.idx.size());
    const double* l = S.lrow.data();
    F.lself = 0;
    if (self >= 0) {
        const auto it = std::lower_bound(S.idx.begin(), S.idx.end(), self);
        if (it != S.idx.end() && *it == self) F.lself = l[it - S.idx.begin()];
    }
    float64x2_t a = vdupq_n_f64(0.0);
    std::int64_t k = 0;
    for (; k + 2 <= m; k += 2) {
        const float64x2_t lk = vld1q_f64(l + k);
        a = vfmaq_f64(a, lk, lk);
    }
    double ss = vgetq_lane_f64(a, 0) + vgetq_lane_f64(a, 1);
    if (k < m) ss = std::fma(l[k], l[k], ss);
    F.ss = ss;
    if (!P.w) {
        F.ssw = ss;
        return;
    }
    double sw = 0;
    for (std::int64_t j = 0; j < m; ++j) sw += l[j] * l[j] / P.w[S.idx[j]];
    F.ssw = sw;
}

// The hat row of lowess's line F at the points [F.lo, F.hi) into S.idx / S.lrow (l_j = v_j (P + Q d_j); the weights
// of window_weights_1d, so hat_at's bits).
void line_row(const Problem& P, const LineFit& L, Scratch& S) {
    const std::int64_t len = L.hi - L.lo;
    S.idx.resize(static_cast<std::size_t>(len)), S.lrow.resize(static_cast<std::size_t>(len));
    S.tmp.resize(static_cast<std::size_t>(len));
    if (L.h == 0) {  // the weighted mean of the points at x0
        const Sorted1D D{P.X, P.y, P.pw, P.n};
        for (std::int64_t k = 0; k < len; ++k) S.idx[k] = L.lo + k, S.lrow[k] = hat_at(D, L, L.lo + k);
        return;
    }
    window_weights_1d(P.X, P.pw, L.lo, L.hi, L.x0, L.h, L.sc, L.inv, L.edge, S.lrow.data(), S.tmp.data());
    const double Pc = L.P, Qc = L.kind == LineKind::Mean ? 0.0 : L.Q;
    for (std::int64_t k = 0; k < len; ++k) {
        S.idx[k] = L.lo + k;
        const double v = S.lrow[k];
        S.lrow[k] = v == 0 ? 0.0 : (L.kind == LineKind::Mean ? v * Pc : v * (Pc + Qc * S.tmp[k]));
    }
}

// sum l_j^2 (and sum l_j^2 / w_j) of lowess's line L over [L.lo, L.hi), two lanes (the weights of window_weights_1d).
template <bool Weighted>
void line_sums(const Problem& P, const LineFit& L, Fit& F) {
    const double* x = P.X;
    const float64x2_t vx0 = vdupq_n_f64(L.x0), vh = vdupq_n_f64(L.h), vsc = vdupq_n_f64(L.sc);
    const float64x2_t vinv = vdupq_n_f64(L.inv), vxl = vdupq_n_f64(L.edge.xl), vxr = vdupq_n_f64(L.edge.xr);
    const float64x2_t vgl = vdupq_n_f64(L.edge.gl), vgr = vdupq_n_f64(L.edge.gr), one = vdupq_n_f64(1.0);
    const float64x2_t zero = vdupq_n_f64(0.0), vP = vdupq_n_f64(L.P), vQ = vdupq_n_f64(L.Q);
    float64x2_t ss = zero, sw = zero;
    std::int64_t j = L.lo;
    for (; j + 2 <= L.hi; j += 2) {
        const float64x2_t xx = vld1q_f64(x + j);
        const float64x2_t diff = vsubq_f64(xx, vx0);
        const float64x2_t ad = vmulq_f64(vabsq_f64(vmulq_f64(diff, vsc)), vinv);
        const float64x2_t right = vaddq_f64(vmulq_f64(vsubq_f64(vxr, xx), vsc), vgr);
        const float64x2_t left = vaddq_f64(vmulq_f64(vsubq_f64(xx, vxl), vsc), vgl);
        const float64x2_t gap = vmaxq_f64(vbslq_f64(vcgezq_f64(diff), right, left), zero);
        const float64x2_t t = vmulq_f64(vmulq_f64(gap, vinv), vaddq_f64(vaddq_f64(one, ad), vmulq_f64(ad, ad)));
        float64x2_t v = vmulq_f64(vmulq_f64(vmulq_f64(t, t), t), vld1q_f64(P.pw + j));
        v = vbslq_f64(vcltq_f64(vabsq_f64(diff), vh), v, zero);
        const float64x2_t d = vmulq_f64(vmulq_f64(diff, vsc), vinv);
        const float64x2_t l = vmulq_f64(v, vaddq_f64(vP, vmulq_f64(vQ, d)));
        ss = vfmaq_f64(ss, l, l);
        if constexpr (Weighted) {
            const float64x2_t wj = vld1q_f64(P.w + j);
            const float64x2_t q = vdivq_f64(vmulq_f64(l, l), wj);
            sw = vaddq_f64(sw, vbslq_f64(vcgtq_f64(v, zero), q, zero));
        }
    }
    double s1 = vgetq_lane_f64(ss, 0) + vgetq_lane_f64(ss, 1), s2 = vgetq_lane_f64(sw, 0) + vgetq_lane_f64(sw, 1);
    for (; j < L.hi; ++j) {
        const bool in = std::fabs(x[j] - L.x0) < L.h;
        const double v = in ? tricube_of(x[j], L.x0, L.sc, L.inv, L.edge) * P.pw[j] : 0.0;
        const double l = v * (L.P + L.Q * (((x[j] - L.x0) * L.sc) * L.inv));
        s1 = std::fma(l, l, s1);
        if (Weighted && v > 0) s2 += l * l / P.w[j];
    }
    F.ss = s1, F.ssw = Weighted ? s2 : s1;
}

// The local fit at x0 (p values), self the sorted index of the data point fitted or -1. want: kValue, kStats (the
// row's statistics into F) or kRows (the hat row into S.idx / S.lrow).
Fit fit_point_row(
    const Problem& P, const spatial::KdEuclid& eu, const double* x0, std::int64_t self, int want, Scratch& S
) {
    Fit F;
    const bool hat = want != kValue;
    const std::int64_t p = P.p;
    if (p == 1 && P.degree == 1) {  // lowess's closed form (SM27, SM46)
        const Sorted1D D{P.X, P.y, P.pw, P.n};
        const LineFit L = fit_line(D, P.q, x0[0], self, P.widen);
        if (L.kind == LineKind::Nan) {
            F.nan = true;
            return F;
        }
        if (L.kind == LineKind::Few) {  // fewer than two positive weights: one gives its y
            gather_1d(P, x0[0], S);
            if (S.idx.empty()) {
                F.none = true;
                return F;
            }
            F.value = P.y[S.idx[0]], F.deficient = true;
            if (hat) S.lrow.assign(1, 1.0);
            return F;
        }
        F.value = L.value, F.deficient = L.kind == LineKind::Mean;
        if (want == kRows) line_row(P, L, S);
        if (want == kStats) {
            if (L.h == 0) {
                line_row(P, L, S);
            } else {
                if (P.w) line_sums<true>(P, L, F);
                else line_sums<false>(P, L, F);
                F.stats = true;
            }
        }
        if (hat && self >= 0) F.lself = hat_at(D, L, self);  // lowess's leverage, bit for bit
        return F;
    }
    const Window W = p == 1 ? gather_1d(P, x0[0], S) : gather_nd(P, eu, x0, S);
    if (W.nan) {
        F.nan = true;
        return F;
    }
    const std::int64_t m = static_cast<std::int64_t>(S.idx.size());
    if (m == 0) {
        F.none = true;
        return F;
    }
    const std::int64_t t = terms_of(p, P.degree);
    if (m == 1) {
        F.value = P.y[S.idx[0]], F.deficient = t > 1;
        if (hat) S.lrow.assign(1, 1.0);
        return F;
    }
    S.yv.resize(static_cast<std::size_t>(m));
    for (std::int64_t k = 0; k < m; ++k) S.yv[k] = P.y[S.idx[k]];
    if (W.h == 0 || P.degree == 0) {
        F.value = window_mean(P, S, want, self, F), F.deficient = t > 1;
        return F;
    }
    int deg = P.degree;
    if (p == 1) {  // the degree the window's distinct x carry (exact)
        std::int64_t distinct = 1;
        for (std::int64_t k = 1; k < m && distinct <= deg; ++k) distinct += P.X[S.idx[k]] != P.X[S.idx[k - 1]];
        deg = static_cast<int>(std::min<std::int64_t>(deg, distinct - 1));
        F.deficient = deg < P.degree;
        if (deg == 0) {
            F.value = window_mean(P, S, want, self, F);
            return F;
        }
    }
    const std::int64_t te = terms_of(p, deg);
    S.cols.resize(static_cast<std::size_t>(m * te));
    double* cols = S.cols.data();
    std::fill(cols, cols + m, 1.0);
    if (p == 1) {
        for (std::int64_t k = 0; k < m; ++k) {
            const double d = S.tmp[k];
            cols[m + k] = d;
            if (deg == 2) cols[2 * m + k] = d * d;
        }
    } else {
        for (std::int64_t k = 0; k < m; ++k) {
            const double* xj = P.X + S.idx[k] * p;
            for (std::int64_t a = 0; a < p; ++a) cols[(1 + a) * m + k] = ((xj[a] - x0[a]) * W.sc) * W.inv;
        }
        if (deg == 2) {
            std::int64_t e = 1 + p;
            for (std::int64_t a = 0; a < p; ++a)
                for (std::int64_t c = a; c < p; ++c, ++e) {
                    const double *ua = cols + (1 + a) * m, *uc = cols + (1 + c) * m;
                    double* out = cols + e * m;
                    for (std::int64_t k = 0; k < m; ++k) out[k] = ua[k] * uc[k];
                }
        }
    }
    const double tol = p == 1 ? 0.0 : static_cast<double>(std::max(m, te)) * kEps;
    double value = kNaN;
    const bool dropped = solve_window(S, m, te, tol, hat, value);
    F.value = value;
    F.deficient = F.deficient || dropped;
    return F;
}

// fit_point_row and, for kStats and kRows, the row's statistics (the line's leverage stays hat_at's).
Fit fit_point(
    const Problem& P, const spatial::KdEuclid& eu, const double* x0, std::int64_t self, int want, Scratch& S
) {
    Fit F = fit_point_row(P, eu, x0, self, want, S);
    if (want != kValue && !F.nan && !F.none && !F.stats) {
        const double keep = F.lself;
        const bool line = P.p == 1 && P.degree == 1 && S.idx.size() > 1;
        row_stats(P, S, self, F);
        if (line && self >= 0) F.lself = keep;
    }
    return F;
}

// ---- tasks ----------------------------------------------------------------------------------------------------

// The work of one point: its window's points times the basis's weight.
std::int64_t point_cost(const Problem& P) {
    const std::int64_t t = terms_of(P.p, P.degree);
    return std::max<std::int64_t>(1, P.q * (4 + t + t * (t + 1) / 2) + (P.p > 1 ? P.n * (P.p + 2) : 0));
}

// Runs fn(scratch, i) for i in [0, count), split into tasks by size alone, each task with its own scratch.
template <class Fn>
void for_points(const Problem& P, std::int64_t count, const Fn& fn) {
    if (count == 0) return;
    const std::int64_t cost = point_cost(P);
    const std::int64_t work = count * cost;
    const auto run = [&](std::int64_t begin, std::int64_t end) {
        Scratch S;
        fn(S, begin, end);
    };
    if (work >= kParallelThreshold && count > 1 && parallel::available()) {
        // about 2^17 of work a task, and at most 256 tasks: a task's scratch (the window's columns) is made once
        const std::int64_t per = std::max({std::int64_t{1}, (std::int64_t{1} << 17) / cost, (count + 255) / 256});
        const std::int64_t tasks = (count + per - 1) / per;
        parallel::parallel_for(
            tasks, [&](std::int64_t t) { run(t * per, std::min(count, (t + 1) * per)); },
            work_ns(std::min(count, per) * cost)
        );
    } else {
        const parallel::Unlocked unlocked(work >= kParallelThreshold);
        run(0, count);
    }
}

// The spatial pieces a call needs (p >= 2): the kernel, the tree's cut and, on the k-nearest path, the tree.
struct Spatial {
    spatial::KdEuclid eu;
    spatial::KdCut cut;
    spatial::KdTree tree;
    bool use_tree = false;
    explicit Spatial(std::int64_t p) : eu{p}, cut(p, false) {}
};

// path: -1 by shape (the tree when 32 q <= n; decision SM140), 0 scan, 1 tree.
void setup_spatial(Problem& P, Spatial& sp, int path) {
    if (P.p < 2 || P.widen != 1.0) return;
    sp.use_tree = path == 1 || (path < 0 && 32 * P.q <= P.n);
    if (!sp.use_tree) return;
    sp.tree = spatial::kd_build(P.X, P.n, P.p, P.p, 10);
    P.tree = &sp.tree;
}

void prepare(const Problem& P, const Spatial& sp, Scratch& S) {
    if (P.tree && !S.search)
        S.search = std::make_unique<spatial::KdSearch<spatial::KdEuclid>>(*P.tree, sp.eu, sp.cut, P.q);
}

// ---- R's robust weights ---------------------------------------------------------------------------------------

// R's lowesw on the residuals of the points of positive weight: cmad = 6 median |r| (3 times the two middle values'
// sum for an even count), weights 1 for |r| <= 0.001 cmad, 0 for |r| > 0.999 cmad, else (1 - (r / cmad)^2)^2; every
// weight 1 when cmad is below the smallest normal number (decision SM100).
void lowesw(const double* r, const double* w, std::int64_t n, double* rho, std::vector<double>& buf) {
    buf.clear();
    for (std::int64_t j = 0; j < n; ++j)
        if (!w || w[j] > 0) buf.push_back(std::fabs(r[j]));
    const std::size_t c = buf.size(), mid = c / 2;
    std::nth_element(buf.begin(), buf.begin() + mid, buf.end());
    double cmad;
    if (c % 2 == 0) {
        const double lower = *std::max_element(buf.begin(), buf.begin() + mid);
        cmad = 3.0 * (buf[mid] + lower);
    } else {
        cmad = 6.0 * buf[mid];
    }
    if (cmad < std::numeric_limits<double>::min()) {
        std::fill(rho, rho + n, 1.0);
        return;
    }
    const double hi = cmad * 0.999, lo = cmad * 0.001;
    for (std::int64_t j = 0; j < n; ++j) {
        const double a = std::fabs(r[j]);
        if (hi < a) rho[j] = 0.0;
        else if (lo < a) {
            const double u = a / cmad;
            const double t = 1.0 - u * u;
            rho[j] = t * t;
        } else rho[j] = 1.0;
    }
}

// ---- arguments ------------------------------------------------------------------------------------------------

const NDArray& vec_arg(nb::handle o, const char* what, std::int64_t n = -1) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 1-D float64 array").c_str());
    if (n >= 0 && a.shape[0] != n) throw nb::value_error(("'" + std::string(what) + "' has the wrong length").c_str());
    return a;
}

const NDArray& mat_arg(nb::handle o, const char* what, std::int64_t p = -1) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[1] < 1)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 2-D float64 array").c_str());
    if (p >= 0 && a.shape[1] != p)
        throw nb::value_error(("'" + std::string(what) + "' has the wrong number of columns").c_str());
    return a;
}

const double* dptr(const NDArray& a) { return reinterpret_cast<const double*>(a.data); }

NDArray vec_out(std::int64_t n) {
    const std::int64_t shape[1] = {n};
    return new_array(DType::Float64, shape);
}

NDArray mat_out(std::int64_t n, std::int64_t p) {
    const std::int64_t shape[2] = {n, p};
    return new_array(DType::Float64, shape);
}

double* optr(NDArray& a) { return reinterpret_cast<double*>(a.data); }

void check_tuning(std::int64_t n, std::int64_t p, int degree, std::int64_t q, double widen) {
    if (degree < 0 || degree > 2) throw nb::value_error("'degree' must be 0, 1 or 2");
    if (n < 1 || q < terms_of(p, degree) || q > n) throw nb::value_error("'q' must be in [terms, n]");
    if (!(widen >= 1.0) || !std::isfinite(widen)) throw nb::value_error("'widen' must be a finite number >= 1");
}

// The stored problem of a result: (Xs, ys, ws, rho) sorted.
struct Stored {
    const NDArray* X = nullptr;
    std::int64_t n = 0, p = 0;
    const double *x = nullptr, *y = nullptr, *w = nullptr, *rho = nullptr;
};

Stored stored(nb::handle xs, nb::handle ys, nb::handle ws, nb::handle rho) {
    Stored s;
    const NDArray& Xa = mat_arg(xs, "Xs");
    s.n = Xa.shape[0], s.p = Xa.shape[1];
    s.x = dptr(Xa);
    s.y = dptr(vec_arg(ys, "ys", s.n));
    s.w = ws.is_none() ? nullptr : dptr(vec_arg(ws, "ws", s.n));
    s.rho = dptr(vec_arg(rho, "rho", s.n));
    return s;
}

// ---- the bindings ---------------------------------------------------------------------------------------------

nb::object loess_fit(
    nb::handle X_obj,
    nb::handle y_obj,
    nb::handle w_obj,
    int degree,
    std::int64_t q,
    double widen,
    std::int64_t iterations,
    bool normalize,
    int path
) {
    const NDArray& Xa = mat_arg(X_obj, "X");
    const std::int64_t n = Xa.shape[0], p = Xa.shape[1];
    const NDArray& ya = vec_arg(y_obj, "y", n);
    const NDArray* wa = w_obj.is_none() ? nullptr : &vec_arg(w_obj, "weights", n);
    check_tuning(n, p, degree, q, widen);
    if (iterations < 1) throw nb::value_error("'iterations' must be >= 1");
    const double *X = dptr(Xa), *y = dptr(ya), *w = wa ? dptr(*wa) : nullptr;

    NDArray fitted_o = vec_out(n), lev_o = vec_out(n), rho_o = vec_out(n);
    NDArray xs_o = mat_out(n, p), ys_o = vec_out(n), rhos_o = vec_out(n), div_o = vec_out(p);
    NDArray ws_o = wa ? vec_out(n) : NDArray{};
    double df = 0, rss = 0, delta1 = 0;
    std::int64_t deficient = 0;
    bool flat = false;
    {
        const parallel::Unlocked unlocked(n * p >= 4096);
        // R's normalization (p >= 2): every column over its 10% trimmed standard deviation
        double* div = optr(div_o);
        std::fill(div, div + p, 1.0);
        if (normalize && p > 1) {
            const std::int64_t trim = static_cast<std::int64_t>(std::ceil(0.1 * static_cast<double>(n)));
            const std::int64_t cnt = n - 2 * trim;
            std::vector<double> col(static_cast<std::size_t>(n));
            for (std::int64_t a = 0; a < p && !flat; ++a) {
                for (std::int64_t j = 0; j < n; ++j) col[j] = X[j * p + a];
                std::sort(col.begin(), col.end());
                double mean = 0, ss = 0;
                for (std::int64_t j = trim; j < n - trim; ++j) mean += col[j];
                mean /= static_cast<double>(cnt);
                for (std::int64_t j = trim; j < n - trim; ++j) ss += (col[j] - mean) * (col[j] - mean);
                const double sdv = cnt >= 2 ? std::sqrt(ss / static_cast<double>(cnt - 1)) : kNaN;
                if (!(sdv > 0) || !std::isfinite(sdv)) flat = true;
                div[a] = sdv;
            }
        }
        if (!flat) {
            std::vector<double> Xn(static_cast<std::size_t>(n * p));
            for (std::int64_t j = 0; j < n; ++j)
                for (std::int64_t a = 0; a < p; ++a) Xn[j * p + a] = X[j * p + a] / div[a];
            std::vector<std::int64_t> order(static_cast<std::size_t>(n));
            std::iota(order.begin(), order.end(), std::int64_t{0});
            std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
                for (std::int64_t c = 0; c < p; ++c)
                    if (Xn[a * p + c] != Xn[b * p + c]) return Xn[a * p + c] < Xn[b * p + c];
                if (y[a] != y[b]) return y[a] < y[b];
                if (w && w[a] != w[b]) return w[a] < w[b];
                return a < b;
            });
            double *xs = optr(xs_o), *ys = optr(ys_o), *rho = optr(rhos_o);
            double* ws = wa ? optr(ws_o) : nullptr;
            for (std::int64_t j = 0; j < n; ++j) {
                const std::int64_t o = order[j];
                for (std::int64_t a = 0; a < p; ++a) xs[j * p + a] = Xn[o * p + a];
                ys[j] = y[o], rho[j] = 1.0;
                if (ws) ws[j] = w[o];
            }
            std::vector<double> pw(static_cast<std::size_t>(n)), fit(static_cast<std::size_t>(n)),
                resid(static_cast<std::size_t>(n)), lev(static_cast<std::size_t>(n)), row1(static_cast<std::size_t>(n)),
                buf;
            std::vector<char> def(static_cast<std::size_t>(n));
            Problem P;
            P.n = n, P.p = p, P.q = q, P.degree = degree, P.widen = widen;
            P.X = xs, P.y = ys, P.w = ws, P.pw = pw.data();
            Spatial sp(p);
            setup_spatial(P, sp, path);
            for (std::int64_t step = 1; step <= iterations; ++step) {
                for (std::int64_t j = 0; j < n; ++j) pw[j] = ws ? ws[j] * rho[j] : rho[j];
                const bool last = step == iterations;
                for_points(P, n, [&](Scratch& S, std::int64_t begin, std::int64_t end) {
                    prepare(P, sp, S);
                    for (std::int64_t i = begin; i < end; ++i) {
                        const Fit F = fit_point(P, sp.eu, xs + i * p, i, last ? kStats : kValue, S);
                        fit[i] = F.none ? ys[i] : F.value;
                        def[i] = F.deficient || F.none;
                        if (!last) continue;
                        if (F.none) {
                            lev[i] = 1.0, row1[i] = 0.0;
                            continue;
                        }
                        const double li = F.lself;
                        lev[i] = li, row1[i] = (1.0 - li) * (1.0 - li) + (F.ss - li * li);
                    }
                });
                if (last) break;
                for (std::int64_t j = 0; j < n; ++j) resid[j] = ys[j] - fit[j];
                lowesw(resid.data(), ws, n, rho, buf);
            }
            double *fo = optr(fitted_o), *lo = optr(lev_o), *ro = optr(rho_o);
            for (std::int64_t j = 0; j < n; ++j) {
                const bool in = !ws || ws[j] > 0;
                const double l = in ? lev[j] : 0.0;
                const double r = ys[j] - fit[j];
                if (in) {
                    df += l;
                    rss += ws ? ws[j] * r * r : r * r;
                    delta1 += row1[j];
                }
                deficient += def[j];
                const std::int64_t o = order[j];
                fo[o] = fit[j], lo[o] = l, ro[o] = rho[j];
            }
        }
    }
    if (flat) return nb::int_(1);
    nb::object ws_obj = wa ? wrap(std::move(ws_o)) : nb::none();
    return nb::make_tuple(
        wrap(std::move(fitted_o)), wrap(std::move(lev_o)), wrap(std::move(rho_o)), df, rss, delta1, deficient,
        wrap(std::move(xs_o)), wrap(std::move(ys_o)), ws_obj, wrap(std::move(rhos_o)), wrap(std::move(div_o))
    );
}

nb::object loess_predict(
    nb::handle xs_obj,
    nb::handle ys_obj,
    nb::handle ws_obj,
    nb::handle rho_obj,
    int degree,
    std::int64_t q,
    double widen,
    nb::handle div_obj,
    nb::handle q_obj,
    bool norms,
    int path
) {
    const Stored D = stored(xs_obj, ys_obj, ws_obj, rho_obj);
    const std::int64_t n = D.n, p = D.p;
    check_tuning(n, p, degree, q, widen);
    const double* div = dptr(vec_arg(div_obj, "divisor", p));
    const NDArray& qa = mat_arg(q_obj, "X_new", p);
    const std::int64_t m = qa.shape[0];
    NDArray val_o = vec_out(m);
    NDArray norm_o = norms ? vec_out(m) : NDArray{};
    {
        const parallel::Unlocked unlocked(n * p >= 4096 || m * q >= kParallelThreshold);
        const double* qx = dptr(qa);
        std::vector<double> pw(static_cast<std::size_t>(n)), Q(static_cast<std::size_t>(m * p));
        for (std::int64_t j = 0; j < n; ++j) pw[j] = D.w ? D.w[j] * D.rho[j] : D.rho[j];
        for (std::int64_t i = 0; i < m; ++i)
            for (std::int64_t a = 0; a < p; ++a) Q[i * p + a] = qx[i * p + a] / div[a];
        Problem P;
        P.n = n, P.p = p, P.q = q, P.degree = degree, P.widen = widen;
        P.X = D.x, P.y = D.y, P.w = D.w, P.pw = pw.data();
        Spatial sp(p);
        setup_spatial(P, sp, path);
        double* vo = optr(val_o);
        double* no = norms ? optr(norm_o) : nullptr;
        for_points(P, m, [&](Scratch& S, std::int64_t begin, std::int64_t end) {
            prepare(P, sp, S);
            for (std::int64_t i = begin; i < end; ++i) {
                const Fit F = fit_point(P, sp.eu, Q.data() + i * p, -1, norms ? kStats : kValue, S);
                const bool ok = !F.nan && !F.none;
                vo[i] = ok ? F.value : kNaN;
                if (!no) continue;
                if (!ok) {
                    no[i] = kNaN;
                    continue;
                }
                no[i] = F.ssw;
            }
        });
    }
    nb::object n_obj = norms ? wrap(std::move(norm_o)) : nb::none();
    return nb::make_tuple(wrap(std::move(val_o)), n_obj);
}

// sum over the 4 x 4 block of rows (I, J) of the squared dot products, with the symmetric weights: the rows are
// those of the dense (I - L) (nr x nr, row-major), over the columns [k0, k1).
double gram_block(
    const double* A,
    std::int64_t nr,
    std::int64_t i0,
    std::int64_t ni,
    std::int64_t j0,
    std::int64_t nj,
    std::int64_t k0,
    std::int64_t k1
) {
    float64x2_t acc[4][4];
    for (auto& rowa : acc)
        for (auto& e : rowa) e = vdupq_n_f64(0.0);
    const double* a[4];
    const double* b[4];
    for (int r = 0; r < 4; ++r) a[r] = A + (i0 + std::min<std::int64_t>(r, ni - 1)) * nr;
    for (int r = 0; r < 4; ++r) b[r] = A + (j0 + std::min<std::int64_t>(r, nj - 1)) * nr;
    std::int64_t k = k0;
    for (; k + 2 <= k1; k += 2) {
        const float64x2_t a0 = vld1q_f64(a[0] + k), a1 = vld1q_f64(a[1] + k), a2 = vld1q_f64(a[2] + k),
                          a3 = vld1q_f64(a[3] + k);
        for (int c = 0; c < 4; ++c) {
            const float64x2_t bc = vld1q_f64(b[c] + k);
            acc[0][c] = vfmaq_f64(acc[0][c], a0, bc);
            acc[1][c] = vfmaq_f64(acc[1][c], a1, bc);
            acc[2][c] = vfmaq_f64(acc[2][c], a2, bc);
            acc[3][c] = vfmaq_f64(acc[3][c], a3, bc);
        }
    }
    double s = 0;
    for (std::int64_t r = 0; r < ni; ++r)
        for (std::int64_t c = 0; c < nj; ++c) {
            const std::int64_t i = i0 + r, j = j0 + c;
            if (j < i) continue;
            double d = vgetq_lane_f64(acc[r][c], 0) + vgetq_lane_f64(acc[r][c], 1);
            if (k < k1) d = std::fma(a[r][k], b[c][k], d);
            s += (j == i ? 1.0 : 2.0) * (d * d);
        }
    return s;
}

double loess_two_delta(
    nb::handle xs_obj,
    nb::handle ys_obj,
    nb::handle ws_obj,
    nb::handle rho_obj,
    int degree,
    std::int64_t q,
    double widen,
    int path
) {
    const Stored D = stored(xs_obj, ys_obj, ws_obj, rho_obj);
    const std::int64_t n = D.n, p = D.p;
    check_tuning(n, p, degree, q, widen);
    double out = 0;
    {
        const parallel::Unlocked unlocked(true);
        std::vector<double> pw(static_cast<std::size_t>(n));
        for (std::int64_t j = 0; j < n; ++j) pw[j] = D.w ? D.w[j] * D.rho[j] : D.rho[j];
        // the points of positive weight, renumbered
        std::vector<std::int64_t> pos(static_cast<std::size_t>(n), -1), pts;
        for (std::int64_t j = 0; j < n; ++j)
            if (!D.w || D.w[j] > 0) pos[j] = static_cast<std::int64_t>(pts.size()), pts.push_back(j);
        const std::int64_t nr = static_cast<std::int64_t>(pts.size());
        std::vector<double> A(static_cast<std::size_t>(nr * nr), 0.0);
        std::vector<std::int64_t> lo(static_cast<std::size_t>(nr)), hi(static_cast<std::size_t>(nr));
        Problem P;
        P.n = n, P.p = p, P.q = q, P.degree = degree, P.widen = widen;
        P.X = D.x, P.y = D.y, P.w = D.w, P.pw = pw.data();
        Spatial sp(p);
        setup_spatial(P, sp, path);
        for_points(P, nr, [&](Scratch& S, std::int64_t begin, std::int64_t end) {
            prepare(P, sp, S);
            for (std::int64_t r = begin; r < end; ++r) {
                const std::int64_t i = pts[r];
                const Fit F = fit_point(P, sp.eu, D.x + i * p, i, kRows, S);
                double* row = A.data() + r * nr;
                std::int64_t a = r, b = r;
                if (!F.none && !F.nan)
                    for (std::size_t k = 0; k < S.idx.size(); ++k) {
                        const std::int64_t c = pos[S.idx[k]];
                        if (c < 0) continue;
                        row[c] -= S.lrow[k];
                        a = std::min(a, c), b = std::max(b, c);
                    }
                else row[r] -= 1.0;  // the data point keeps its y: l_ii = 1
                row[r] += 1.0;
                lo[r] = a, hi[r] = b + 1;
            }
        });
        // the Gram matrix of the rows, 4 x 4 blocks, the upper triangle; a task per block row, summed in order
        const std::int64_t nb4 = (nr + 3) / 4;
        std::vector<double> part(static_cast<std::size_t>(nb4), 0.0);
        const auto block_row = [&](std::int64_t I) {
            const std::int64_t i0 = 4 * I, ni = std::min<std::int64_t>(4, nr - i0);
            std::int64_t alo = lo[i0], ahi = hi[i0];
            for (std::int64_t r = 1; r < ni; ++r) alo = std::min(alo, lo[i0 + r]), ahi = std::max(ahi, hi[i0 + r]);
            double s = 0;
            for (std::int64_t J = I; J < nb4; ++J) {
                const std::int64_t j0 = 4 * J, nj = std::min<std::int64_t>(4, nr - j0);
                std::int64_t blo = lo[j0], bhi = hi[j0];
                for (std::int64_t r = 1; r < nj; ++r) blo = std::min(blo, lo[j0 + r]), bhi = std::max(bhi, hi[j0 + r]);
                const std::int64_t k0 = std::max(alo, blo), k1 = std::min(ahi, bhi);
                if (k0 >= k1) continue;
                s += gram_block(A.data(), nr, i0, ni, j0, nj, k0, k1);
            }
            part[I] = s;
        };
        const std::int64_t work = nr * nr * std::max<std::int64_t>(1, nr / 8);
        if (work >= kParallelThreshold && nb4 > 1 && parallel::available()) {
            parallel::parallel_for(nb4, block_row, work_ns(work / nb4));
        } else {
            for (std::int64_t I = 0; I < nb4; ++I) block_row(I);
        }
        for (std::int64_t I = 0; I < nb4; ++I) out += part[I];
    }
    return out;
}

}  // namespace

void bind_loess(nb::module_& m) {
    m.def(
        "loess_fit", named("loess_fit", &loess_fit), nb::arg("X").none(), nb::arg("y").none(), nb::arg("w").none(),
        nb::arg("degree"), nb::arg("q"), nb::arg("widen"), nb::arg("iterations"), nb::arg("normalize"),
        nb::arg("path")
    );  // fmt: skip
    m.def(
        "loess_predict", named("loess_predict", &loess_predict), nb::arg("Xs").none(), nb::arg("ys").none(),
        nb::arg("ws").none(), nb::arg("rho").none(), nb::arg("degree"), nb::arg("q"), nb::arg("widen"),
        nb::arg("divisor").none(), nb::arg("X_new").none(), nb::arg("norms"), nb::arg("path")
    );  // fmt: skip
    m.def(
        "loess_two_delta", named("loess_two_delta", &loess_two_delta), nb::arg("Xs").none(), nb::arg("ys").none(),
        nb::arg("ws").none(), nb::arg("rho").none(), nb::arg("degree"), nb::arg("q"), nb::arg("widen"),
        nb::arg("path")
    );  // fmt: skip
}

}  // namespace sd::smooth
