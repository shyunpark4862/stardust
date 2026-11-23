// The penalized splines of sd.smooth (docs/design/smooth.md §3.3; decisions SM30, SM50-SM60, SM67-SM69): the
// smoothing spline (make_smoothing_spline, smoothing_spline), the P-spline (pspline) and the least-squares spline
// (make_lsq_spline), all as one banded least squares (smooth/bandqr.hpp).
//
//   data       x is sorted (stable) and equal x are merged into one point with the summed weight W_g and the
//              weighted mean of y (the penalized problem is exactly the same; no tolerance). The rows are
//              sqrt(W_g) b_g, b_g the k + 1 B-splines at x_g (interpolate/deboor.hpp)
//   penalty    smoothing spline: the knots are every distinct x with the ends four times (n_g + 2 coefficients),
//              and int f''^2 over an interval [x_j, x_j+1] of length h, where f'' is linear from m_0 to m_1, is
//              (h / 4)(m_0 + m_1)^2 + (h / 12)(m_0 - m_1)^2: two rows per interval, each over the interval's four
//              coefficients, so the stacked matrix is banded with width 4 and needs no factorization of the
//              Gram matrix of f'' (SM50). The minimizer over these n_g + 2 coefficients is the natural spline
//              (the minimizer over all functions lies in the space). P-spline: the rows of the d-th difference
//              matrix. Each penalty row is scaled by sqrt(lam)
//   solve      the data rows are triangularized once (R_d), then for each lam the rows of R_d and the penalty
//              rows go through a fresh Givens QR in the order of their first column (R_d's row j, then the
//              penalty rows starting at column j): O(n_c bw^2) per lam, the data's O(n bw^2) once (SM51)
//   leverage   L_ii = w_i b_g' Sigma b_g with the band of Sigma = (R'R)^-1 (the selected inverse); tr L = sum_g
//              W_g b_g' Sigma b_g. Below lam_0 the smoothing spline's 1 - L_g also comes from Reinsch's form and
//              each merged point keeps the smaller of the two as the one computed directly (SM53)
//   guards     the condition of R with unit columns above 2^43 is a LinAlgError (SM58); a lam of the search is a
//              candidate only where tr(Sigma M'M) = n_c holds within 2^-10 max(n - tr L, 1) as computed
//   lam = inf  the weighted least squares on the null space of the penalty (the lines for the smoothing spline,
//              the coefficient sequences that are polynomials of degree d - 1 in the index for the P-spline),
//              by Householder QR of the small (n_g x d) matrix
//   search     rho = log(lam / lam_0), lam_0 = sum w (x_max - x_min)^3 2^-35 for the smoothing spline and
//              tr(B'WB) / tr(D'D) for the P-spline (free of the units of x, y and w; SM54), rho in
//              [-32, 20]: the criterion at the 53 integers of the interval (in parallel), then fminbound around
//              the smallest (|d rho| <= 1e-8, 60 values); the end is the answer when the smallest is at an end
//              (status -1 the low end, 1 the high end). df: brentq on tr L(rho) - df (|tr L - df| <= 1e-10 df,
//              100 values)
//
// Private bindings (the Python layer checks the arguments):
//   spline_fit(x, w, y, kind, k, d, t, mode, lam, crit, target, stats) -> dict
//       kind 0 smoothing spline (k = 3, the knots from x), 1 P-spline (t given), 2 least squares (t given, no
//       penalty; w are FITPACK's weights, applied as w_i^2); mode 0 lam given (> 0), 1 lam = inf, 2 lam = 0,
//       3 choose by crit (0 gcv, 1 cv), 4 df = target. stats: also the leverage, the criteria and the hat state.
//       The dict: code (0, or an error: 1 too few distinct x, 2 too few weighted x, 3 a penalty row overflows,
//       4 singular, 5 too ill-conditioned: SM58), t, c (n_c, ncol), lam, trl, gcv, cv, rss (ncol), nobs, status,
//       converged, lam0, kappa, leverage (n,), fitted (n, ncol: the internal fit, for the criteria), xg, wg, yg (the
//       merged data), hat (the state)
//   spline_hat(t, k, hat, xnew) -> (m,)  sum_j l_j(x0)^2 / w_j, the hat row's weighted squared norm
//   natural_hat(t, xg, wg) -> hat       the hat state of the natural interpolating spline (lam = 0)
//   spline_merge_weights(x, w) -> (n,)  the summed weight of the points with each point's x
//   pspline_knots(xb, xe, nseg, k) -> t
//   schoenberg_whitney(x, w, t, k) -> -1, or the first basis function no data point of positive weight
//       supports (x sorted)

#include "smooth/smooth.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "base/buffer.hpp"
#include "base/interval.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "interpolate/deboor.hpp"
#include "parallel/pool.hpp"
#include "smooth/bandqr.hpp"
#include "smooth/search.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::smooth {

namespace nb = nanobind;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kRhoLo = -32.0, kRhoHi = 20.0;
constexpr std::int64_t kGrid = 53;
// The largest estimated condition number of the stacked matrix a fit accepts: u kappa <= 2^-10, so that the
// coefficients keep 10 bits (SM58); above it the fit is a LinAlgError and a lam of the search is not a candidate.
constexpr double kMaxKappa = 0x1p43;
// A lam is a candidate of the search (and the df root acceptable) only where the computed leverages sum with the
// penalty rows' to n_c within 2^-10 (n - tr L) (at least 2^-10): tr(Sigma M'M) = n_c exactly, and its defect is the
// error the band of Sigma carries into tr L (SM58).
constexpr double kMaxDefect = 0x1p-10;

enum Kind : int { kSmoothing = 0, kPSpline = 1, kLeastSquares = 2 };
enum Mode : int { kGiven = 0, kInfinite = 1, kZero = 2, kChoose = 3, kDf = 4 };
enum Code : int { kOk = 0, kFewDistinct = 1, kFewWeighted = 2, kOverflow = 3, kSingular = 4, kIllConditioned = 5 };
enum HatMode : int { kHatBand = 0, kHatNull = 1 };

const NDArray& f64_arg(nb::handle o, const char* what, int ndim) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != ndim || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error((std::string("'") + what + "' must be a C-contiguous float64 array").c_str());
    return a;
}

const double* data_of(const NDArray& a) { return reinterpret_cast<const double*>(a.data); }

NDArray new_f64(std::int64_t rows, std::int64_t cols = -1) {
    if (cols < 0) {
        const std::int64_t shape[1] = {rows};
        return new_array(DType::Float64, shape);
    }
    const std::int64_t shape[2] = {rows, cols};
    return new_array(DType::Float64, shape);
}

NDArray vec_array(const std::vector<double>& v) {
    NDArray a = new_f64(static_cast<std::int64_t>(v.size()));
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(a.data));
    return a;
}

NDArray mat_array(const std::vector<double>& v, std::int64_t rows, std::int64_t cols) {
    NDArray a = new_f64(rows, cols);
    std::copy(v.begin(), v.begin() + rows * cols, reinterpret_cast<double*>(a.data));
    return a;
}

// The basis interval l (k <= l <= nb - 1) of x for knots t with nb basis functions.
inline std::int64_t interval_of(const double* t, std::int64_t N, std::int64_t k, std::int64_t nb, double x) {
    std::int64_t c;
    count_le_lanes<1>(t, N, &x, &c);
    return interpolate::knot_interval(c, k, nb);
}

// ------------------------------------------------------------------------------------------------ the problem

struct Problem {
    int kind = kSmoothing;
    std::int64_t n = 0, ncol = 0, k = 3, d = 2;
    // the merged data, sorted
    std::int64_t ng = 0, npos = 0;
    std::vector<double> xg, wg, yg;  // ng, ng, ng x ncol
    std::vector<std::int64_t> group;  // n: the group of each original point
    const double* x = nullptr;
    const double* w = nullptr;
    const double* y = nullptr;
    std::vector<double> within;  // ncol: sum_i w_i (y_i - ybar_g)^2
    // the basis
    std::vector<double> t;
    std::int64_t nc = 0, bw = 0, kd = 0;  // coefficients, the merged band width, the data rows' width k + 1
    std::vector<std::int64_t> off;  // ng
    std::vector<double> bas;  // ng x kd
    // the penalty rows, unscaled
    std::int64_t np = 0, pw = 0;
    std::vector<std::int64_t> poff;
    std::vector<double> prow;  // np x pw
    // the null space of the penalty (lam = inf)
    std::int64_t dn = 0;
    std::vector<double> N;  // nc x dn
    double lam0 = kNaN;
    // the merged points of positive weight and their spacings (Reinsch's form)
    std::vector<std::int64_t> pos;
    std::vector<double> hpos;
    // Reinsch's form without lam (reinsch_parts): A = RA + lam QQ by band entry (mc x 3), and the products
    // q(r, c1) q(r, c2) of each row r of Q over its columns (m x 3 x 3)
    std::vector<double> RA, QQ, QP;
    // the data rows' triangle
    BandQR data;
};

int merge(Problem& P, double wpow) {
    const std::int64_t n = P.n, ncol = P.ncol;
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) { return P.x[a] < P.x[b]; });
    P.group.assign(static_cast<std::size_t>(n), 0);
    P.xg.clear();
    P.wg.clear();
    P.yg.clear();
    P.npos = 0;
    std::int64_t g = -1;
    double prev = kNaN;
    std::vector<double> sums;
    for (std::int64_t r = 0; r < n; ++r) {
        const std::int64_t i = order[r];
        const double xi = P.x[i];
        double wi = P.w[i];
        if (wpow == 2.0) wi = wi * wi;
        if (r == 0 || xi != prev) {
            ++g;
            P.xg.push_back(xi);
            P.wg.push_back(0.0);
            P.yg.resize(static_cast<std::size_t>((g + 1) * ncol), 0.0);
            prev = xi;
        }
        P.group[i] = g;
        P.wg[g] += wi;
        if (wi > 0) ++P.npos;
        for (std::int64_t q = 0; q < ncol; ++q) P.yg[g * ncol + q] += wi * P.y[i * ncol + q];
    }
    P.ng = g + 1;
    for (std::int64_t j = 0; j < P.ng; ++j)
        for (std::int64_t q = 0; q < ncol; ++q) P.yg[j * ncol + q] = P.wg[j] > 0 ? P.yg[j * ncol + q] / P.wg[j] : 0.0;
    P.within.assign(static_cast<std::size_t>(ncol), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        double wi = P.w[i];
        if (wpow == 2.0) wi = wi * wi;
        if (!(wi > 0)) continue;
        const std::int64_t gi = P.group[i];
        for (std::int64_t q = 0; q < ncol; ++q) {
            const double r = P.y[i * ncol + q] - P.yg[gi * ncol + q];
            P.within[q] += wi * r * r;
        }
    }
    P.pos.clear();
    for (std::int64_t j = 0; j < P.ng; ++j)
        if (P.wg[j] > 0) P.pos.push_back(j);
    P.hpos.assign(P.pos.size() > 0 ? P.pos.size() - 1 : 0, 0.0);
    for (std::size_t j = 0; j + 1 < P.pos.size(); ++j) P.hpos[j] = P.xg[P.pos[j + 1]] - P.xg[P.pos[j]];
    return static_cast<int>(P.pos.size());
}

// The basis of every merged point: off[g] = l - k, bas[g] = B_{l-k} .. B_l at x_g.
void basis(Problem& P) {
    const std::int64_t N = static_cast<std::int64_t>(P.t.size()), k = P.k;
    P.nc = N - k - 1;
    P.kd = k + 1;
    P.off.assign(static_cast<std::size_t>(P.ng), 0);
    P.bas.assign(static_cast<std::size_t>(P.ng * P.kd), 0.0);
    std::vector<double> hh(static_cast<std::size_t>(k + 1));
    std::int64_t l = k;
    for (std::int64_t g = 0; g < P.ng; ++g) {
        const double xv = P.xg[g];
        while (l < P.nc - 1 && P.t[l + 1] <= xv) ++l;
        P.off[g] = l - k;
        basis_at(P.t.data(), k, l, xv, P.bas.data() + g * P.kd, hh.data());
    }
}

// The smoothing spline's penalty rows (two per interval) and null space (the lines, in Greville abscissae).
bool smoothing_penalty(Problem& P) {
    const std::int64_t ng = P.ng;
    P.pw = 4;
    P.np = 2 * (ng - 1);
    P.poff.assign(static_cast<std::size_t>(P.np), 0);
    P.prow.assign(static_cast<std::size_t>(P.np * P.pw), 0.0);
    double e0[4], e1[4], hh[4];
    for (std::int64_t j = 0; j + 1 < ng; ++j) {
        const std::int64_t l = j + 3;
        interpolate::basis_values(P.t.data(), 3, l, P.xg[j], 2, e0, hh);
        interpolate::basis_values(P.t.data(), 3, l, P.xg[j + 1], 2, e1, hh);
        const double h = P.xg[j + 1] - P.xg[j];
        const double a = std::sqrt(h / 4.0), b = std::sqrt(h / 12.0);
        double* r0 = P.prow.data() + (2 * j) * 4;
        double* r1 = r0 + 4;
        for (int i = 0; i < 4; ++i) {
            r0[i] = a * (e0[i] + e1[i]);
            r1[i] = b * (e0[i] - e1[i]);
            if (!std::isfinite(r0[i]) || !std::isfinite(r1[i])) return false;
        }
        P.poff[2 * j] = P.poff[2 * j + 1] = j;
    }
    P.dn = 2;
    P.N.assign(static_cast<std::size_t>(P.nc * 2), 0.0);
    const double mid = 0.5 * (P.xg[0] + P.xg[ng - 1]), half = 0.5 * (P.xg[ng - 1] - P.xg[0]);
    for (std::int64_t i = 0; i < P.nc; ++i) {
        const double xi = (P.t[i + 1] + P.t[i + 2] + P.t[i + 3]) / 3.0;
        P.N[i * 2] = 1.0;
        P.N[i * 2 + 1] = (xi - mid) / half;
    }
    return true;
}

// The P-spline's difference rows and null space (index polynomials of degree < d).
void difference_penalty(Problem& P) {
    const std::int64_t d = P.d, nc = P.nc;
    P.pw = d + 1;
    P.np = std::max<std::int64_t>(0, nc - d);
    P.poff.assign(static_cast<std::size_t>(P.np), 0);
    P.prow.assign(static_cast<std::size_t>(P.np * P.pw), 0.0);
    std::vector<double> coef(static_cast<std::size_t>(d + 1));
    for (std::int64_t a = 0; a <= d; ++a) {  // (-1)^(d - a) C(d, a)
        double c = 1.0;
        for (std::int64_t i = 0; i < a; ++i) c = c * static_cast<double>(d - i) / static_cast<double>(i + 1);
        coef[a] = ((d - a) % 2 ? -1.0 : 1.0) * c;
    }
    for (std::int64_t r = 0; r < P.np; ++r) {
        P.poff[r] = r;
        for (std::int64_t a = 0; a <= d; ++a) P.prow[r * P.pw + a] = coef[a];
    }
    P.dn = d;
    P.N.assign(static_cast<std::size_t>(nc * d), 0.0);
    const double ic = 0.5 * static_cast<double>(nc - 1), sc = std::max(ic, 1.0);
    for (std::int64_t i = 0; i < nc; ++i) {
        const double u = (static_cast<double>(i) - ic) / sc;
        double v = 1.0;
        for (std::int64_t j = 0; j < d; ++j) {
            P.N[i * d + j] = v;
            v *= u;
        }
    }
}

void factor_data(Problem& P) {
    P.bw = std::max(P.kd, P.pw);
    P.data.reset(P.nc, P.kd, 0, P.ncol);
    std::vector<double> h(static_cast<std::size_t>(P.kd)), y(static_cast<std::size_t>(P.ncol)), hb;
    double tb = 0.0, tg = 0.0;
    for (std::int64_t g = 0; g < P.ng; ++g) {
        const double W = P.wg[g];
        if (!(W > 0)) continue;
        const double s = std::sqrt(W);
        for (std::int64_t a = 0; a < P.kd; ++a) {
            h[a] = s * P.bas[g * P.kd + a];
            tb += W * P.bas[g * P.kd + a] * P.bas[g * P.kd + a];
        }
        for (std::int64_t q = 0; q < P.ncol; ++q) y[q] = s * P.yg[g * P.ncol + q];
        P.data.add(P.off[g], h.data(), hb.data(), y.data());
    }
    for (double v : P.prow) tg += v * v;
    P.lam0 = tg > 0 ? tb / tg : kNaN;
    if (P.kind == kSmoothing) {  // sum w (x_max - x_min)^3 2^-35 (SM54)
        double sw = 0.0;
        for (std::int64_t g = 0; g < P.ng; ++g) sw += std::max(P.wg[g], 0.0);
        const double r = (P.xg[P.ng - 1] - P.xg[0]) * 0x1p-12;
        P.lam0 = sw * (r * r * r) * 2.0;
    }
}

// The condition estimate of R with its columns scaled to unit 2-norm: Givens QR is invariant under a scaling of the
// columns (the rotations and the rounding of each entry scale with it), so the condition that bounds its errors is
// that of the best-scaled matrix, and unit columns are within a factor sqrt(n_c) of the best (van der Sluis). Knots
// close together give columns of very different norms, which alone would make the unscaled estimate large (SM58).
double scaled_condition(const std::vector<double>& R, std::int64_t nc, std::int64_t bw) {
    std::vector<double> norm(static_cast<std::size_t>(nc), 0.0), Rs(R);
    for (std::int64_t j = 0; j < nc; ++j) {
        double big = 0.0;
        for (std::int64_t i = std::max<std::int64_t>(0, j - bw + 1); i <= j; ++i)
            big = std::max(big, std::abs(R[i * bw + (j - i)]));
        double s = 0.0;
        if (big > 0)
            for (std::int64_t i = std::max<std::int64_t>(0, j - bw + 1); i <= j; ++i) {
                const double v = R[i * bw + (j - i)] / big;
                s += v * v;
            }
        norm[j] = big * std::sqrt(s);
    }
    for (std::int64_t i = 0; i < nc; ++i)
        for (std::int64_t d = 0; d < bw && i + d < nc; ++d)
            if (norm[i + d] > 0) Rs[i * bw + d] = R[i * bw + d] / norm[i + d];
    return condition_estimate(Rs.data(), nc, bw);
}

// ------------------------------------------------------------------------------------------------ one fit

struct Fit {
    BandQR qr;
    std::vector<double> c, S, f, q, rss;  // coefficients, Sigma's band, fitted (ng x ncol), b' Sigma b, RSS
    std::vector<double> Lg, Cg;  // the merged points' leverage W_g q_g and 1 - W_g q_g
    std::vector<double> RN;  // lam = inf: the small triangle (dn x dn)
    double trl = kNaN, gcv = kNaN, cv = kNaN, dfres = kNaN;
    double kappa = kNaN;  // the estimated condition number of the stacked matrix (columns scaled)
    double defect = kNaN;  // |tr(Sigma M'M) - n_c| as computed
    bool singular = false, illcond = false, trusted = true;  // trusted: the leverage keeps 10 bits
    std::vector<double> C, U, SU;  // scratch of Reinsch's form (a Fit is reused across the fits of a search)
};


// sum_j b_j x[off + j] over the columns: the fitted values of the merged points.
void fitted(const Problem& P, Fit& F) {
    F.f.resize(static_cast<std::size_t>(P.ng * P.ncol));  // every entry is written
    for (std::int64_t g = 0; g < P.ng; ++g) {
        const double* b = P.bas.data() + g * P.kd;
        const double* c = F.c.data() + P.off[g] * P.ncol;
        for (std::int64_t q = 0; q < P.ncol; ++q) {
            double v = b[0] * c[q];
            for (std::int64_t a = 1; a < P.kd; ++a) v += b[a] * c[a * P.ncol + q];
            F.f[g * P.ncol + q] = v;
        }
    }
    F.rss = P.within;
    for (std::int64_t g = 0; g < P.ng; ++g) {
        const double W = P.wg[g];
        if (!(W > 0)) continue;
        for (std::int64_t q = 0; q < P.ncol; ++q) {
            const double r = P.yg[g * P.ncol + q] - F.f[g * P.ncol + q];
            F.rss[q] += W * r * r;
        }
    }
}

// b' Sigma b over the K columns o .. o + K - 1 from the band of Sigma (S, width bw): sum_a b_a (S_aa b_a +
// sum_{e > a} 2 S_ae b_e), in that order.
template <int K>
inline double quad_band(const double* S, std::int64_t bw, std::int64_t o, const double* b) {
    double v = 0.0;
    for (int a = 0; a < K; ++a) {
        double row = S[(o + a) * bw] * b[a];
        for (int e = a + 1; e < K; ++e) row += 2.0 * S[(o + a) * bw + (e - a)] * b[e];
        v += b[a] * row;
    }
    return v;
}

inline double quad_band(const double* S, std::int64_t bw, std::int64_t o, const double* b, std::int64_t K) {
    switch (K) {
        case 3: return quad_band<3>(S, bw, o, b);
        case 4: return quad_band<4>(S, bw, o, b);
        case 5: return quad_band<5>(S, bw, o, b);
        default: break;
    }
    double v = 0.0;
    for (std::int64_t a = 0; a < K; ++a) {
        double row = S[(o + a) * bw] * b[a];
        for (std::int64_t e = a + 1; e < K; ++e) row += 2.0 * S[(o + a) * bw + (e - a)] * b[e];
        v += b[a] * row;
    }
    return v;
}

// q_g = b_g' Sigma b_g from the band of Sigma, and L_g = W_g q_g.
void leverage_band(const Problem& P, Fit& F, std::int64_t bw) {
    F.q.resize(static_cast<std::size_t>(P.ng));  // every entry is written
    F.Lg.resize(static_cast<std::size_t>(P.ng));
    F.Cg.resize(static_cast<std::size_t>(P.ng));
    for (std::int64_t g = 0; g < P.ng; ++g) {
        const double v = quad_band(F.S.data(), bw, P.off[g], P.bas.data() + g * P.kd, P.kd);
        F.q[g] = v;
        F.Lg[g] = P.wg[g] * v;
        F.Cg[g] = 1.0 - F.Lg[g];
    }
}

// The smoothing spline's 1 - L_g from Reinsch's form (Green and Silverman 1994, §2.3): I - L = lam W^-1 Q A^-1 Q'
// with A = R + lam Q'W^-1 Q over the merged points of positive weight, Q (m x (m - 2)) the second divided
// differences and R the Gram matrix of the hat functions (tridiagonal). No cancellation where L_g is near 1
// (small lam), where the stacked QR's L_g loses its digits. A is pentadiagonal: its band Cholesky and the
// band of A^-1 (smooth/bandqr.hpp's selected inverse). False when the Cholesky fails.
// The parts of Reinsch's form that do not depend on lam (the same operations as computed inside each fit).
void reinsch_parts(Problem& P) {
    const std::int64_t m = static_cast<std::int64_t>(P.pos.size());
    if (m <= 2) return;
    const std::int64_t mc = m - 2;
    // Q's entries by row: q(r, r) = 1 / h_r, q(r, r - 1) = -(1 / h_{r-1} + 1 / h_r), q(r, r - 2) = 1 / h_{r-1}
    const auto qv = [&](std::int64_t r, std::int64_t c) {
        if (c == r) return 1.0 / P.hpos[r];
        if (c == r - 1) return -(1.0 / P.hpos[r - 1] + 1.0 / P.hpos[r]);
        return 1.0 / P.hpos[r - 1];
    };
    P.RA.assign(static_cast<std::size_t>(mc * 3), 0.0);
    P.QQ.assign(static_cast<std::size_t>(mc * 3), 0.0);
    for (std::int64_t c = 0; c < mc; ++c) {
        for (std::int64_t d = 0; d < 3 && c + d < mc; ++d) {
            P.RA[c * 3 + d] = d == 0 ? (P.hpos[c] + P.hpos[c + 1]) / 3.0 : (d == 1 ? P.hpos[c + 1] / 6.0 : 0.0);
            double qq = 0.0;  // sum_r Q_rc Q_r,c+d / W_r over the rows r in [c + d, c + 2]
            for (std::int64_t r = c + d; r <= c + 2; ++r) qq += qv(r, c) * qv(r, c + d) / P.wg[P.pos[r]];
            P.QQ[c * 3 + d] = qq;
        }
    }
    P.QP.assign(static_cast<std::size_t>(m * 9), 0.0);
    for (std::int64_t r = 0; r < m; ++r) {
        const std::int64_t c0 = std::max<std::int64_t>(0, r - 2), c1 = std::min(r, mc - 1);
        for (std::int64_t a = c0; a <= c1; ++a)
            for (std::int64_t b = c0; b <= c1; ++b) P.QP[r * 9 + (a - c0) * 3 + (b - c0)] = qv(r, a) * qv(r, b);
    }
}

bool reinsch_complement(
    const Problem& P, double lam, std::vector<double>& C, std::vector<double>& U, std::vector<double>& S
) {
    const std::int64_t m = static_cast<std::int64_t>(P.pos.size());
    C.resize(static_cast<std::size_t>(P.ng));  // the caller reads the entries of P.pos only
    if (m <= 2) {
        for (std::int64_t g : P.pos) C[g] = 0.0;
        return true;
    }
    const std::int64_t mc = m - 2;
    U.resize(static_cast<std::size_t>(mc * 3));
    for (std::int64_t c = 0; c < mc; ++c)
        for (std::int64_t d = 0; d < 3; ++d) U[c * 3 + d] = c + d < mc ? P.RA[c * 3 + d] + lam * P.QQ[c * 3 + d] : 0.0;
    for (std::int64_t c = 0; c < mc; ++c) {  // U'U = A in place, U upper with bandwidth 3
        for (std::int64_t d = 0; d < 3 && c + d < mc; ++d) {
            double v = U[c * 3 + d];
            for (std::int64_t l = std::max<std::int64_t>(0, c + d - 2); l < c; ++l)
                if (c - l < 3) v -= U[l * 3 + (c - l)] * U[l * 3 + (c + d - l)];
            if (d == 0) {
                if (!(v > 0.0 && std::isfinite(v))) return false;
                U[c * 3] = std::sqrt(v);
            } else {
                U[c * 3 + d] = v / U[c * 3];
            }
        }
    }
    S.resize(static_cast<std::size_t>(mc * 3));
    selected_inverse(U.data(), mc, 3, S.data());
    const auto sig = [&](std::int64_t a, std::int64_t b) { return a <= b ? S[a * 3 + (b - a)] : S[b * 3 + (a - b)]; };
    for (std::int64_t r = 0; r < m; ++r) {
        double v = 0.0;
        const std::int64_t c0 = std::max<std::int64_t>(0, r - 2), ce = std::min(r, mc - 1);
        const double* qp = P.QP.data() + r * 9;
        for (std::int64_t c1 = c0; c1 <= ce; ++c1)
            for (std::int64_t c2 = c0; c2 <= ce; ++c2) v += qp[(c1 - c0) * 3 + (c2 - c0)] * sig(c1, c2);
        const std::int64_t g = P.pos[r];
        C[g] = lam / P.wg[g] * v;
        if (!std::isfinite(C[g])) return false;
    }
    return true;
}

// tr L and n - tr L from L_g and 1 - L_g (zero-weight points are not in n).
void traces(const Problem& P, Fit& F) {
    double tr = 0.0, res = static_cast<double>(P.npos - static_cast<std::int64_t>(P.pos.size()));
    for (std::int64_t g : P.pos) {
        tr += F.Lg[g];
        res += F.Cg[g];
    }
    F.trl = tr;
    F.dfres = res;
}

void criteria(const Problem& P, Fit& F, bool want_cv) {
    const double n = static_cast<double>(P.npos);
    double rss = 0.0;
    for (double v : F.rss) rss += v;
    const double den = F.dfres;
    F.gcv = n * rss / (den * den);
    F.cv = kNaN;
    if (!want_cv) return;
    double acc = 0.0;
    for (std::int64_t i = 0; i < P.n; ++i) {
        const double wi = P.kind == kLeastSquares ? P.w[i] * P.w[i] : P.w[i];
        if (!(wi > 0)) continue;
        const std::int64_t g = P.group[i];
        const double share = wi / P.wg[g];
        const double one = (1.0 - share) + share * F.Cg[g];  // 1 - w_i q_g without cancellation
        for (std::int64_t q = 0; q < P.ncol; ++q) {
            const double r = (P.y[i * P.ncol + q] - F.f[g * P.ncol + q]) / one;
            acc += wi * r * r;
        }
    }
    F.cv = acc / n;
}

// The penalized fit at a finite lam > 0 (lam = 0: the data rows alone).
// kappa: also the condition estimate (the final fit's guard; the search's candidates are judged by the defect).
void fit_lambda(const Problem& P, double lam, Fit& F, bool stats, bool want_cv, bool kappa = true) {
    const std::int64_t nc = P.nc, bw = P.bw, ncol = P.ncol;
    F.qr.reset(nc, bw, 0, ncol);
    std::vector<double> h(static_cast<std::size_t>(bw)), y(static_cast<std::size_t>(ncol));
    double* hb = nullptr;
    const double sl = std::sqrt(lam);
    const bool pw4 = P.pw == 4 && bw == 4;  // the smoothing spline's rows fill h
    std::int64_t p = 0;
    const auto penalty_rows = [&](std::int64_t upto) {  // the penalty rows starting at columns <= upto
        if (!(lam > 0)) return;
        for (; p < P.np && P.poff[p] <= upto; ++p) {
            if (pw4) {
                const double* g = P.prow.data() + p * 4;
                h[0] = sl * g[0], h[1] = sl * g[1], h[2] = sl * g[2], h[3] = sl * g[3];
            } else {
                std::fill(h.begin(), h.end(), 0.0);
                for (std::int64_t a = 0; a < P.pw; ++a) h[a] = sl * P.prow[p * P.pw + a];
            }
            std::fill(y.begin(), y.end(), 0.0);
            F.qr.add(P.poff[p], h.data(), hb, y.data());
        }
    };
    for (std::int64_t j = 0; j < nc; ++j) {
        std::fill(h.begin(), h.end(), 0.0);
        const double* r = P.data.R.data() + j * P.kd;
        bool any = false;
        for (std::int64_t a = 0; a < P.kd && j + a < nc; ++a) {
            h[a] = r[a];
            any = any || r[a] != 0.0;
        }
        for (std::int64_t q = 0; q < ncol; ++q) y[q] = P.data.Z[j * ncol + q];
        if (any) F.qr.add(j, h.data(), hb, y.data());
        penalty_rows(j);
    }
    F.singular = F.qr.first_bad_diagonal() >= 0;
    if (F.singular) {
        F.c.assign(static_cast<std::size_t>(nc * ncol), kNaN);
        return;
    }
    F.c.resize(static_cast<std::size_t>(nc * ncol));  // the back substitution writes every entry
    if (kappa) {
        F.kappa = scaled_condition(F.qr.R, nc, bw);
        F.illcond = !(F.kappa <= kMaxKappa);
    }
    F.qr.solve(F.c.data());
    fitted(P, F);
    if (!stats) return;
    F.S.resize(static_cast<std::size_t>(nc * bw));  // selected_inverse writes every entry
    selected_inverse(F.qr.R.data(), nc, bw, F.S.data());
    leverage_band(P, F, bw);
    {  // tr(Sigma M'M) = n_c: the defect of the computed band of Sigma decides whether the leverage is trusted (SM58)
        double tr = 0.0;
        for (std::int64_t g : P.pos) tr += F.Lg[g];
        if (lam > 0) {
            double tp = 0.0;
            for (std::int64_t p = 0; p < P.np; ++p)
                tp += quad_band(F.S.data(), bw, P.poff[p], P.prow.data() + p * P.pw, P.pw);
            tr += lam * tp;
        }
        F.defect = std::abs(tr - static_cast<double>(nc));
    }
    // below lam_0 the smoothing spline's L_g near 1 come from Reinsch's form: each merged point takes the smaller
    // of L_g (the QR's) and 1 - L_g (Reinsch's) as the one computed directly (SM53)
    if (P.kind == kSmoothing && lam > 0 && lam < P.lam0) {
        std::vector<double>& C = F.C;
        if (reinsch_complement(P, lam, C, F.U, F.SU)) {
            for (std::int64_t g : P.pos) {
                if (C[g] < F.Lg[g]) {
                    F.Cg[g] = C[g];
                    F.Lg[g] = 1.0 - C[g];
                }
            }
        }
    }
    traces(P, F);
    F.trusted = F.defect <= kMaxDefect * std::max(F.dfres, 1.0);
    criteria(P, F, want_cv);
}

// lam = inf: the weighted least squares on the null space N by Householder QR of X = diag(sqrt W) B N.
void fit_infinite(const Problem& P, Fit& F, bool want_cv) {
    const std::int64_t ng = P.ng, dn = P.dn, ncol = P.ncol, nc = P.nc;
    std::vector<double> X(static_cast<std::size_t>(ng * dn), 0.0), Y(static_cast<std::size_t>(ng * ncol), 0.0);
    for (std::int64_t g = 0; g < ng; ++g) {
        const double s = std::sqrt(std::max(P.wg[g], 0.0));
        const double* b = P.bas.data() + g * P.kd;
        for (std::int64_t j = 0; j < dn; ++j) {
            double v = 0.0;
            for (std::int64_t a = 0; a < P.kd; ++a) v += b[a] * P.N[(P.off[g] + a) * dn + j];
            X[g * dn + j] = s * v;
        }
        for (std::int64_t q = 0; q < ncol; ++q) Y[g * ncol + q] = s * P.yg[g * ncol + q];
    }
    // Householder on the columns of X (row-major, ng x dn), applied to Y
    std::vector<double> v(static_cast<std::size_t>(ng));
    F.RN.assign(static_cast<std::size_t>(dn * dn), 0.0);
    F.singular = false;
    for (std::int64_t j = 0; j < dn; ++j) {
        double scale = 0.0;
        for (std::int64_t i = j; i < ng; ++i) scale = std::max(scale, std::abs(X[i * dn + j]));
        if (scale == 0.0) {
            F.singular = true;
            break;
        }
        double nrm = 0.0;
        for (std::int64_t i = j; i < ng; ++i) {
            const double u = X[i * dn + j] / scale;
            nrm += u * u;
        }
        nrm = scale * std::sqrt(nrm);
        const double alpha = X[j * dn + j] > 0 ? -nrm : nrm;
        for (std::int64_t i = j; i < ng; ++i) v[i] = X[i * dn + j];
        v[j] -= alpha;
        double vv = 0.0;
        for (std::int64_t i = j; i < ng; ++i) vv += v[i] * v[i];
        auto reflect = [&](double* col, std::int64_t stride) {
            double s = 0.0;
            for (std::int64_t i = j; i < ng; ++i) s += v[i] * col[i * stride];
            const double f = 2.0 * s / vv;
            for (std::int64_t i = j; i < ng; ++i) col[i * stride] -= f * v[i];
        };
        for (std::int64_t jj = j + 1; jj < dn; ++jj) reflect(X.data() + jj, dn);
        for (std::int64_t q = 0; q < ncol; ++q) reflect(Y.data() + q, ncol);
        X[j * dn + j] = alpha;
        for (std::int64_t jj = j; jj < dn; ++jj) F.RN[j * dn + jj] = X[j * dn + jj];
    }
    F.c.assign(static_cast<std::size_t>(nc * ncol), kNaN);
    if (!F.singular) {
        double big = 0.0;
        for (std::int64_t j = 0; j < dn; ++j) big = std::max(big, std::abs(F.RN[j * dn + j]));
        for (std::int64_t j = 0; j < dn; ++j)
            if (!(std::abs(F.RN[j * dn + j]) > big * 1e-14)) F.singular = true;
    }
    if (F.singular) return;
    std::vector<double> a(static_cast<std::size_t>(dn * ncol));
    for (std::int64_t q = 0; q < ncol; ++q)
        for (std::int64_t j = dn - 1; j >= 0; --j) {
            double s = Y[j * ncol + q];
            for (std::int64_t jj = j + 1; jj < dn; ++jj) s -= F.RN[j * dn + jj] * a[jj * ncol + q];
            a[j * ncol + q] = s / F.RN[j * dn + j];
        }
    for (std::int64_t i = 0; i < nc; ++i)
        for (std::int64_t q = 0; q < ncol; ++q) {
            double s = 0.0;
            for (std::int64_t j = 0; j < dn; ++j) s += P.N[i * dn + j] * a[j * ncol + q];
            F.c[i * ncol + q] = s;
        }
    fitted(P, F);
    F.q.assign(static_cast<std::size_t>(ng), 0.0);
    std::vector<double> z(static_cast<std::size_t>(dn));
    double tr = 0.0;
    for (std::int64_t g = 0; g < ng; ++g) {
        const double* b = P.bas.data() + g * P.kd;
        for (std::int64_t j = 0; j < dn; ++j) {
            double s = 0.0;
            for (std::int64_t e = 0; e < P.kd; ++e) s += b[e] * P.N[(P.off[g] + e) * dn + j];
            z[j] = s;
        }
        double nrm = 0.0;
        for (std::int64_t j = 0; j < dn; ++j) {  // R' u = z
            double s = z[j];
            for (std::int64_t i = 0; i < j; ++i) s -= F.RN[i * dn + j] * z[i];
            z[j] = s / F.RN[j * dn + j];
            nrm += z[j] * z[j];
        }
        F.q[g] = nrm;
        tr += P.wg[g] * nrm;
    }
    F.Lg.assign(static_cast<std::size_t>(ng), 0.0);
    F.Cg.assign(static_cast<std::size_t>(ng), 1.0);
    for (std::int64_t g = 0; g < ng; ++g) {
        F.Lg[g] = P.wg[g] * F.q[g];
        F.Cg[g] = 1.0 - F.Lg[g];
    }
    traces(P, F);
    (void)tr;
    criteria(P, F, want_cv);
}

// ------------------------------------------------------------------------------------------------ the search

struct Choice {
    double rho = kNaN, lam = kNaN;
    int status = 0;
    bool converged = true;
};

double criterion_of(const Fit& F, int crit) {
    const double v = crit == 1 ? F.cv : F.gcv;
    return std::isnan(v) ? kInf : v;
}

Choice choose(const Problem& P, int crit) {
    std::vector<double> vals(static_cast<std::size_t>(kGrid), kInf);
    // A task fits a run of grid points in one Fit (its arrays reused: fresh ones cost a page fault per page); every
    // point is the same computation whichever task has it.
    const auto run = [&](std::int64_t j0, std::int64_t j1) {
        Fit F;
        for (std::int64_t j = j0; j < j1; ++j) {
            fit_lambda(P, P.lam0 * std::exp(kRhoLo + static_cast<double>(j)), F, true, crit == 1, false);
            if (!F.singular && F.trusted) vals[j] = criterion_of(F, crit);
        }
    };
    const std::int64_t per = (P.ng + P.nc) * P.bw * P.bw * (2 + P.ncol);
    if (per * kGrid >= kParallelThreshold && parallel::available()) {
        // at most a quarter of the physical memory in the tasks' Fits at once (docs/design/smooth.md §3.9): fewer
        // tasks run fewer at a time, and the values do not depend on the split
        const double fit_bytes =
            8.0 * static_cast<double>(2 * P.nc * P.bw + 2 * P.nc * P.ncol + P.ng * (P.ncol + 5) + 6 * P.ng);
        const double budget = static_cast<double>(physical_memory()) / 4.0;
        const std::int64_t fit_cap = std::max<std::int64_t>(1, static_cast<std::int64_t>(budget / fit_bytes));
        const std::int64_t tasks = std::min({kGrid, 2 * std::int64_t{parallel::num_threads()}, fit_cap});
        parallel::parallel_for(
            tasks, [&](std::int64_t t) { run(t * kGrid / tasks, (t + 1) * kGrid / tasks); },
            work_ns(per * kGrid / tasks)
        );
    } else {
        const parallel::Unlocked unlocked(per * kGrid >= kParallelThreshold);
        run(0, kGrid);
    }
    std::int64_t best = 0;
    for (std::int64_t j = 1; j < kGrid; ++j)
        if (vals[j] <= vals[best]) best = j;  // a tie goes to the larger lam
    Choice out;
    if (best == 0 || best == kGrid - 1) {
        out.rho = kRhoLo + static_cast<double>(best);
        out.status = best == 0 ? -1 : 1;
        out.converged = false;
    } else {
        const parallel::Unlocked unlocked(per >= kParallelThreshold);
        Fit F;
        const auto f = [&](double rho) {
            fit_lambda(P, P.lam0 * std::exp(rho), F, true, crit == 1, false);
            return F.singular || !F.trusted ? kInf : criterion_of(F, crit);
        };
        const double a = kRhoLo + static_cast<double>(best - 1), b = kRhoLo + static_cast<double>(best + 1);
        const ScalarResult r = fminbound(f, a, b, 1e-8, 60);
        out.rho = r.f <= vals[best] ? r.x : kRhoLo + static_cast<double>(best);
        out.converged = true;
        out.status = 0;
    }
    out.lam = P.lam0 * std::exp(out.rho);
    return out;
}

Choice choose_df(const Problem& P, double df) {
    const std::int64_t per = (P.ng + P.nc) * P.bw * P.bw * (2 + P.ncol);
    const parallel::Unlocked unlocked(per >= kParallelThreshold);
    Fit F;
    const auto f = [&](double rho) {
        fit_lambda(P, P.lam0 * std::exp(rho), F, true, false, false);
        return F.singular ? kNaN : F.trl - df;
    };
    Choice out;
    const double flo = f(kRhoLo), fhi = f(kRhoHi);
    if (!(flo >= 0)) {
        out.rho = kRhoLo;
        out.status = -1;
        out.converged = false;
    } else if (!(fhi <= 0)) {
        out.rho = kRhoHi;
        out.status = 1;
        out.converged = false;
    } else {
        const ScalarResult r = brentq(f, kRhoLo, kRhoHi, flo, fhi, 4e-16, 1e-10 * df, 100);
        out.rho = r.x;
        out.converged = r.converged;
        out.status = 0;
    }
    out.lam = P.lam0 * std::exp(out.rho);
    return out;
}

// ------------------------------------------------------------------------------------------------ bindings

nb::dict spline_fit(
    nb::handle xo,
    nb::handle wo,
    nb::handle yo,
    int kind,
    std::int64_t k,
    std::int64_t d,
    nb::handle to,
    int mode,
    double lam,
    int crit,
    double target,
    bool stats
) {
    const NDArray& xa = f64_arg(xo, "x", 1);
    const NDArray& wa = f64_arg(wo, "w", 1);
    const NDArray& ya = f64_arg(yo, "y", 2);
    Problem P;
    P.kind = kind;
    P.n = xa.shape[0];
    P.ncol = ya.shape[1];
    P.k = k;
    P.d = d;
    P.x = data_of(xa);
    P.w = data_of(wa);
    P.y = data_of(ya);
    if (kind != kSmoothing) {
        const NDArray& ta = f64_arg(to, "t", 1);
        P.t.assign(data_of(ta), data_of(ta) + ta.shape[0]);
    }
    nb::dict out;
    int code = kOk;
    Choice ch;
    Fit F;
    bool infinite = false;
    {
        const parallel::Unlocked unlocked(P.n * (P.ncol + 4) >= kParallelThreshold);
        const int gpos = merge(P, kind == kLeastSquares ? 2.0 : 1.0);
        if (kind == kSmoothing) {
            if (P.ng < 5) code = kFewDistinct;
            else if (gpos < 2) code = kFewWeighted;
            if (code == kOk) {
                P.t.clear();
                for (int i = 0; i < 3; ++i) P.t.push_back(P.xg[0]);
                P.t.insert(P.t.end(), P.xg.begin(), P.xg.end());
                for (int i = 0; i < 3; ++i) P.t.push_back(P.xg[P.ng - 1]);
            }
        }
        if (code == kOk) {
            basis(P);
            if (kind == kSmoothing) {
                if (!smoothing_penalty(P)) code = kOverflow;
            } else if (kind == kPSpline) {
                difference_penalty(P);
            } else {
                P.pw = 0;
                P.np = 0;
            }
        }
        if (code == kOk) {
            factor_data(P);
            if (kind == kSmoothing) reinsch_parts(P);
            if ((mode == kChoose || mode == kDf) && !(P.lam0 > 0 && std::isfinite(P.lam0))) code = kOverflow;
        }
        if (code == kOk) {
            if (mode == kChoose) {
                ch = choose(P, crit);
            } else if (mode == kDf) {
                ch = choose_df(P, target);
            } else {
                ch.lam = mode == kInfinite ? kInf : (mode == kZero ? 0.0 : lam);
                ch.rho = std::log(ch.lam / P.lam0);
            }
            if (ch.lam == kInf) {
                infinite = true;
                fit_infinite(P, F, crit == 1 || stats);
            } else {
                fit_lambda(P, ch.lam, F, stats || mode == kChoose || mode == kDf, true);
            }
            if (F.singular) code = kSingular;
            else if (F.illcond || ((mode == kChoose || mode == kDf) && !infinite && !F.trusted)) code = kIllConditioned;
        }
    }
    out["code"] = code;
    if (code != kOk && code != kSingular && code != kIllConditioned) return out;
    out["t"] = wrap(vec_array(P.t));
    out["c"] = wrap(mat_array(F.c, P.nc, P.ncol));
    out["lam"] = ch.lam;
    out["status"] = ch.status;
    out["converged"] = ch.converged;
    out["nobs"] = P.npos;
    out["trl"] = F.trl;
    out["gcv"] = F.gcv;
    out["cv"] = F.cv;
    out["lam0"] = P.lam0;
    out["kappa"] = F.kappa;
    if (code != kOk) return out;
    out["rss"] = wrap(vec_array(F.rss));
    if (!stats) return out;
    NDArray lev = new_f64(P.n), fit = new_f64(P.n, P.ncol);
    auto* lp = reinterpret_cast<double*>(lev.data);
    auto* fp = reinterpret_cast<double*>(fit.data);
    for (std::int64_t i = 0; i < P.n; ++i) {
        const double wi = kind == kLeastSquares ? P.w[i] * P.w[i] : P.w[i];
        const std::int64_t g = P.group[i];
        lp[i] = F.Lg.empty() ? kNaN : (wi > 0 ? wi / P.wg[g] * F.Lg[g] : 0.0);
        for (std::int64_t q = 0; q < P.ncol; ++q) fp[i * P.ncol + q] = F.f[g * P.ncol + q];
    }
    out["leverage"] = wrap(std::move(lev));
    out["fitted"] = wrap(std::move(fit));
    out["xg"] = wrap(vec_array(P.xg));
    out["wg"] = wrap(vec_array(P.wg));
    out["yg"] = wrap(mat_array(P.yg, P.ng, P.ncol));
    nb::list hat;
    if (infinite) {
        hat.append(static_cast<int>(kHatNull));
        hat.append(wrap(mat_array(F.RN, P.dn, P.dn)));
        hat.append(wrap(mat_array(P.N, P.nc, P.dn)));
    } else {
        hat.append(static_cast<int>(kHatBand));
        hat.append(wrap(mat_array(F.qr.R, P.nc, P.bw)));
        hat.append(wrap(vec_array(P.wg)));
    }
    hat.append(wrap(vec_array(P.xg)));
    out["hat"] = nb::tuple(hat);
    return out;
}

// The interpolating natural spline's hat state through merged points xg with weights wg (lam = 0): the
// triangle of the square system [f''(x_0) = 0; B; f''(x_-1) = 0] (rows unweighted) and the factors 1 / W_g.
nb::tuple natural_hat(nb::handle to, nb::handle xo, nb::handle wo) {
    const NDArray& ta = f64_arg(to, "t", 1);
    const NDArray& xa = f64_arg(xo, "x", 1);
    const NDArray& wa = f64_arg(wo, "w", 1);
    const double* t = data_of(ta);
    const std::int64_t N = ta.shape[0], nc = N - 4, ng = xa.shape[0];
    BandQR Q;
    Q.reset(nc, 4, 0, 0);
    double h[4], hh[4], y[1];
    interpolate::basis_values(t, 3, 3, t[3], 2, h, hh);
    Q.add(0, h, nullptr, y);
    std::int64_t l = 3;
    for (std::int64_t g = 0; g < ng; ++g) {
        const double xv = data_of(xa)[g];
        while (l < nc - 1 && t[l + 1] <= xv) ++l;
        basis_at(t, 3, l, xv, h, hh);
        Q.add(l - 3, h, nullptr, y);
    }
    interpolate::basis_values(t, 3, nc - 1, t[nc], 2, h, hh);
    Q.add(nc - 4, h, nullptr, y);
    std::vector<double> fac(static_cast<std::size_t>(ng));
    for (std::int64_t g = 0; g < ng; ++g) fac[g] = 1.0 / data_of(wa)[g];
    nb::list hat;
    hat.append(static_cast<int>(kHatBand));
    hat.append(wrap(mat_array(Q.R, nc, 4)));
    hat.append(wrap(vec_array(fac)));
    hat.append(wrap(vec_array(std::vector<double>(data_of(xa), data_of(xa) + ng))));
    return nb::tuple(hat);
}

nb::object spline_hat(nb::handle to, std::int64_t k, nb::handle hat, nb::handle xo) {
    const NDArray& ta = f64_arg(to, "t", 1);
    const NDArray& xn = f64_arg(xo, "x", 1);
    const nb::tuple st = nb::borrow<nb::tuple>(hat);
    const int mode = nb::cast<int>(st[0]);
    const double* t = data_of(ta);
    const std::int64_t N = ta.shape[0], nc = N - k - 1, m = xn.shape[0];
    NDArray out = new_f64(m);
    auto* op = reinterpret_cast<double*>(out.data);
    const double* xs = data_of(xn);
    if (mode == kHatNull) {
        const NDArray& RN = f64_arg(st[1], "hat", 2);
        const NDArray& Na = f64_arg(st[2], "hat", 2);
        const std::int64_t dn = RN.shape[0];
        const parallel::Unlocked unlocked(m * dn * (k + 1) >= kParallelThreshold);
        std::vector<double> b(static_cast<std::size_t>(k + 1)), hh(b.size()), z(static_cast<std::size_t>(dn));
        for (std::int64_t i = 0; i < m; ++i) {
            if (std::isnan(xs[i])) {
                op[i] = kNaN;
                continue;
            }
            const std::int64_t l = interval_of(t, N, k, nc, xs[i]);
            basis_at(t, k, l, xs[i], b.data(), hh.data());
            double nrm = 0.0;
            for (std::int64_t j = 0; j < dn; ++j) {
                double s = 0.0;
                for (std::int64_t e = 0; e <= k; ++e) s += b[e] * data_of(Na)[(l - k + e) * dn + j];
                for (std::int64_t jj = 0; jj < j; ++jj) s -= data_of(RN)[jj * dn + j] * z[jj];
                z[j] = s / data_of(RN)[j * dn + j];
                nrm += z[j] * z[j];
            }
            op[i] = nrm;
        }
        return wrap(std::move(out));
    }
    const NDArray& Ra = f64_arg(st[1], "hat", 2);
    const NDArray& fa = f64_arg(st[2], "hat", 1);
    const NDArray& xg = f64_arg(st[3], "hat", 1);
    const std::int64_t bw = Ra.shape[1], ng = xg.shape[0];
    const double* R = data_of(Ra);
    const double* fac = data_of(fa);
    const double* xgp = data_of(xg);
    std::vector<std::int64_t> goff(static_cast<std::size_t>(ng));
    std::vector<double> gb(static_cast<std::size_t>(ng * (k + 1)));
    {
        std::vector<double> hh(static_cast<std::size_t>(k + 1));
        for (std::int64_t g = 0; g < ng; ++g) {
            const std::int64_t l = interval_of(t, N, k, nc, xgp[g]);
            goff[g] = l - k;
            basis_at(t, k, l, xgp[g], gb.data() + g * (k + 1), hh.data());
        }
    }
    const auto point = [&](std::int64_t i, std::vector<double>& v, std::vector<double>& b, std::vector<double>& hh) {
        if (std::isnan(xs[i])) {
            op[i] = kNaN;
            return;
        }
        const std::int64_t l = interval_of(t, N, k, nc, xs[i]);
        basis_at(t, k, l, xs[i], b.data(), hh.data());
        std::fill(v.begin(), v.end(), 0.0);
        for (std::int64_t e = 0; e <= k; ++e) v[l - k + e] = b[e];
        normal_solve(R, nc, bw, l - k, v.data());
        double s = 0.0;
        for (std::int64_t g = 0; g < ng; ++g) {
            const double* bg = gb.data() + g * (k + 1);
            double u = 0.0;
            for (std::int64_t e = 0; e <= k; ++e) u += bg[e] * v[goff[g] + e];
            s += fac[g] * u * u;
        }
        op[i] = s;
    };
    const std::int64_t per = nc * bw + ng * (k + 1);
    const std::int64_t grain = std::max<std::int64_t>(1, kParallelTask / std::max<std::int64_t>(per, 1));
    const std::int64_t tasks = (m + grain - 1) / grain;
    const auto run = [&](std::int64_t task) {
        std::vector<double> v(static_cast<std::size_t>(nc)), b(static_cast<std::size_t>(k + 1)), hh(b.size());
        for (std::int64_t i = task * grain; i < std::min(m, (task + 1) * grain); ++i) point(i, v, b, hh);
    };
    if (m * per >= kParallelThreshold && tasks > 1 && parallel::available()) {
        parallel::parallel_for(tasks, run, work_ns(grain * per));
    } else {
        const parallel::Unlocked unlocked(m * per >= kParallelThreshold);
        for (std::int64_t task = 0; task < tasks; ++task) run(task);
    }
    return wrap(std::move(out));
}

// W_g of each point: the summed weight of the points with its x.
nb::object spline_merge_weights(nb::handle xo, nb::handle wo) {
    const NDArray& xa = f64_arg(xo, "x", 1);
    const NDArray& wa = f64_arg(wo, "w", 1);
    const double* x = data_of(xa);
    const double* w = data_of(wa);
    const std::int64_t n = xa.shape[0];
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) { return x[a] < x[b]; });
    NDArray out = new_f64(n);
    auto* op = reinterpret_cast<double*>(out.data);
    for (std::int64_t r = 0; r < n;) {
        std::int64_t e = r;
        double s = 0.0;
        while (e < n && x[order[e]] == x[order[r]]) s += w[order[e++]];
        for (std::int64_t i = r; i < e; ++i) op[order[i]] = s;
        r = e;
    }
    return wrap(std::move(out));
}

nb::object pspline_knots(double xb, double xe, std::int64_t nseg, std::int64_t k) {
    const double h = (xe - xb) / static_cast<double>(nseg);
    std::vector<double> t;
    for (std::int64_t j = -k; j < nseg; ++j) t.push_back(xb + static_cast<double>(j) * h);
    for (std::int64_t j = 0; j <= k; ++j) t.push_back(xe + static_cast<double>(j) * h);
    return wrap(vec_array(t));
}

// Schoenberg-Whitney: every B_j (j < nb) has a data point of positive weight where it is nonzero, the points
// taken in increasing x without repetition (x sorted, w the weights). -1, or the first j that fails.
std::int64_t schoenberg_whitney(nb::handle xo, nb::handle wo, nb::handle to, std::int64_t k) {
    const NDArray& xa = f64_arg(xo, "x", 1);
    const NDArray& wa = f64_arg(wo, "w", 1);
    const NDArray& ta = f64_arg(to, "t", 1);
    const double* x = data_of(xa);
    const double* w = data_of(wa);
    const double* t = data_of(ta);
    const std::int64_t m = xa.shape[0], N = ta.shape[0], nb = N - k - 1;
    std::int64_t i = 0;
    double last = -kInf;
    for (std::int64_t j = 0; j < nb; ++j) {
        // B_j is nonzero on (t_j, t_{j+k+1}), and at t_j when t_j = t_{j+k} (the left end), at t_{j+k+1} when it
        // is the base interval's right end t_nb with t_{j+1} = t_{j+k+1}
        const double lo = t[j], hi = t[j + k + 1];
        const bool left_closed = t[j] == t[j + k] && t[j + k] < t[j + k + 1];
        const bool right_closed = j == nb - 1;
        while (i < m) {
            const double xi = x[i];
            if (!(w[i] > 0) || !(xi > last) || xi < lo || (xi == lo && !left_closed)) {
                ++i;
                continue;
            }
            break;
        }
        if (i >= m) return j;
        const double xi = x[i];
        if (!(xi < hi || (xi == hi && right_closed))) return j;
        last = xi;
        ++i;
    }
    return -1;
}

}  // namespace

void bind_spline(nb::module_& m) {
    m.def(
        "spline_fit", &spline_fit, nb::arg("x").none(), nb::arg("w").none(), nb::arg("y").none(), nb::arg("kind"),
        nb::arg("k"), nb::arg("d"), nb::arg("t").none(), nb::arg("mode"), nb::arg("lam"), nb::arg("crit"),
        nb::arg("target"), nb::arg("stats")
    );
    m.def("natural_hat", &natural_hat, nb::arg("t").none(), nb::arg("x").none(), nb::arg("w").none());
    m.def("spline_hat", &spline_hat, nb::arg("t").none(), nb::arg("k"), nb::arg("hat").none(), nb::arg("x").none());
    m.def("spline_merge_weights", &spline_merge_weights, nb::arg("x").none(), nb::arg("w").none());
    m.def("pspline_knots", &pspline_knots, nb::arg("xb"), nb::arg("xe"), nb::arg("nseg"), nb::arg("k"));
    m.def(
        "schoenberg_whitney", &schoenberg_whitney, nb::arg("x").none(), nb::arg("w").none(), nb::arg("t").none(),
        nb::arg("k")
    );
}

}  // namespace sd::smooth
