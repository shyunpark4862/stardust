// The penalized regressions' numerical core (docs/design/regression.md §3.5).
//
// The objective (G45, scikit-learn's): (1/(2 sw)) sum w_i (y_i - b0 - x_i beta)^2
// + alpha (rho ||S beta||_1 + (1 - rho)/2 ||S beta||^2), sw = sum w (the sample weights of
// scikit-learn, which rescales them to sum to n), S = diag(s_j) the columns' population
// standard deviations under standardize (glmnet's), else I. The intercept is outside the
// penalty: the columns and y are centered by their weighted means (none without the intercept).
// In the centered, scaled coordinates z_ij = (x_ij - c_j)/s_j, b_j = s_j beta_j, with the
// normalized weights omega = w / sw, the problem is 1/2 sum omega (v - z b)^2 + l1 ||b||_1
// + l2/2 ||b||^2, v = y - c_y, l1 = alpha rho, l2 = alpha (1 - rho).
//
// One regression per batch element (linalg's run_batch); a path of alphas runs inside one
// element, largest alpha first, each solution starting from the previous one (warm start).
// Cross-validation folds are batch elements too: the training rows by their weights, the test
// rows by a second set of weights whose squared errors are summed for every alpha.
//   - The rows with w > 0 are the fit's (G20). One pass gives the weighted means (and the
//     constant columns, which leave the fit); the Gram of sqrt(w) [(x - c), y - c_y, 1] by
//     dsyrk then gives the exactly centered Gram (G - S S^T / sw, S the last column: the
//     rounding of the means does not stay) and the scales.
//   - ridge (rho = 0 by the closed form): the QR of the centered whitened scaled design (TSQR
//     over fixed blocks of rows when tall: the blocks' QR on the pool, then the QR of their
//     stacked R), the SVD of R; b(alpha) = V diag(s/(s^2 + alpha)) U^T v~ for every alpha, the
//     residual sum of squares, the effective degrees of freedom and GCV in closed form. No
//     X^T X. When the solution's condition (s_1^2 + alpha)/(s_min^2 + alpha) exceeds
//     kRefineCond k, up to kPolishSteps refinements with residuals from the data in
//     double-double (G51: the fixed point is the exact problem's solution, as G1's OLS).
//   - lasso and elastic net: cyclic coordinate descent in column order, covariance updates
//     (the Gram) up to kCovMaxCols columns, naive updates of the residual above; active-set
//     cycling (sweeps over the nonzero coefficients until they settle, then a full sweep).
//     Converged when the duality gap (scikit-learn's formulation, written without
//     cancellation) is <= tol x sum omega v^2 (scikit-learn's tol ||y||^2 in its scaling), the
//     gap computed after a full sweep whose largest change is <= tol x the largest
//     coefficient. The KKT polish: with the active set A and its signs, the stationarity
//     conditions are the linear system (Z_A^T Omega Z_A + l2 I) b_A = Z_A^T Omega v - l1 sign_A,
//     solved by Cholesky of the Gram block (refined with double-double residuals from the data,
//     G51, when its estimated condition number exceeds kRefineCond |A|); it is taken when the signs hold and
//     every inactive column has |z_j^T Omega r| <= l1. It is tried after the gap test passes
//     and, during the descent, once per sign pattern that two consecutive sweeps leave
//     unchanged. The polished solution depends only on A and the signs, so a path and a single
//     fit give the same bits when they agree on them.
//   - penalized GLM (binomial logit, poisson log): IRLS outside (the quadratic approximation
//     at the current coefficients, from one working-weight Gram pass per iteration), coordinate
//     descent inside on the same penalty, step halving on the penalized objective
//     F = deviance / (2 sw) + penalty, converged when |F - F_old| / (|F| + 0.1) < tol; then
//     Newton's method on the active set with fixed signs and the same KKT test. Without an L1
//     term (l1_ratio = 0) Newton's method runs from the start.
// Row passes are cut into chunks fixed by the sizes alone and their partial sums are added in
// order: the bits do not depend on the thread count or on running inside a batch.
#include "regression/penalized.hpp"

#include "regression/lsq.hpp"

#include "bind/errors.hpp"
#include "linalg/ops/common.hpp"
#include "parallel/pool.hpp"
#include "special/dd.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace sd::regression {

namespace nb = nanobind;
using linalg::Batch;
using linalg::Failure;
using linalg::Input;
using linalg::lint;
using linalg::Out;
using linalg::Scratch;
using DD = special::dd::DD;
using special::dd::operator+;
using special::dd::operator-;
using special::dd::operator*;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kU = 0x1p-53;

// The per-alpha scalars, in this order (stardust/regression/_penalized.py reads them).
// pRss: sum w r^2 (GLM: the deviance); pTss: sum w (y - ybar)^2, or sum w y^2 without the
// intercept (GLM: the null deviance); pTestSse, pTestSw: the squared error (GLM: deviance)
// and the weight sum over the rows of the test weights (cross-validation); pGcv: ridge's
// generalized cross-validation; pPolished: 1 when the KKT polish was taken.
enum PStat : int {
    pNobs = 0, pSumW, pNIter, pGap, pConv, pDf, pIsNaN, pRss, pTss, pTestSse, pTestSw, pPolished, pGcv, pNStat
};

// Failure codes of one regression.
constexpr int eInfY = 1, eInfX = 2, eInfW = 3, eNegW = 4, eBadY = 5;

enum Family : int { fGaussian = 0, fBinomial = 1, fPoisson = 2 };

// Covariance (Gram) updates up to this many columns, naive (residual) updates above.
constexpr lint kCovMaxCols = 1024;
// The penalized GLM's inner coordinate descent: covariance updates up to this many columns
// (the Gram is formed again at each IRLS iteration), its relative duality gap, its sweeps.
constexpr lint kGlmCovMaxCols = 1024;
constexpr double kInnerTol = 1e-9;
constexpr int kInnerMaxSweeps = 1000;
// The KKT polish refines its Gram solve with residuals from the data when the active Gram
// block's estimated condition number (LAPACK's 1-norm estimate from the Cholesky factor, which
// can exceed the 2-norm one by the active count a) exceeds this times a (the Gram solve's
// error is about kappa u).
constexpr double kRefineCond = 64;
// Refinement steps of the KKT polish, Newton steps of the GLM's polish.
constexpr int kPolishSteps = 3;
// A failed KKT polish retries after coordinate descent to a tolerance kRetryTol times smaller.
constexpr int kPolishRetries = 2;
constexpr double kRetryTol = 1e-6;
constexpr int kNewtonSteps = 30;
// The KKT test of an inactive column: |gradient| <= l1 (1 + kKktRel) + kKktAbs sqrt(d_j v'v).
constexpr double kKktRel = 1e-9;
constexpr double kKktAbs = 64 * kU;

inline double rd(const std::byte* p) {
    double v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    e = std::fma(a, b, -p);
}

// Rows per chunk of a row pass, and the work (rows x columns) from which it goes to the pool.
constexpr lint kRowChunk = 4096;
constexpr double kParMin = 65536;

template <class F>
void for_chunks(lint nch, double work, F&& body) {
    if (nch >= 2 && work >= kParMin) {
        parallel::parallel_for(nch, [&](std::int64_t c) { body(static_cast<lint>(c)); });
    } else {
        for (lint c = 0; c < nch; ++c) body(c);
    }
}

// out[0..len) = sum over the chunks of kRowChunk rows of body(t0, t1, part) (each part zeroed
// first), added in chunk order: the same bits for any thread count.
template <class F>
void chunk_sum(lint m, lint len, double* out, double cols, F&& body) {
    const lint nch = std::max<lint>(1, (m + kRowChunk - 1) / kRowChunk);
    std::vector<double> parts(static_cast<std::size_t>(nch * len), 0.0);
    for_chunks(nch, double(m) * cols, [&](lint c) {
        body(c * kRowChunk, std::min(m, (c + 1) * kRowChunk), parts.data() + c * len);
    });
    std::fill(out, out + len, 0.0);
    for (lint c = 0; c < nch; ++c)
        for (lint q = 0; q < len; ++q) out[q] += parts[static_cast<std::size_t>(c * len + q)];
}

// G (q x q, column-major, full) = B^T B over the m rows, B packed by pack(t0, t1, buf) as a
// (t1 - t0) x q row-major block (rows contiguous). Fixed chunks of rows; slot s accumulates the chunks s,
// s + S, ... in order by dsyrk and the slots are added in order (S fixed by q and m).
template <class Pack>
void gram(lint m, lint q, double* G, Pack&& pack) {
    std::fill(G, G + q * q, 0.0);
    if (q == 0 || m == 0) return;
    const lint rows = std::clamp<lint>((1 << 20) / q, 256, 4096);
    const lint nch = (m + rows - 1) / rows;
    const lint S = std::min<lint>(nch, std::clamp<lint>((1 << 22) / (q * q), 1, 8));
    std::vector<double> slots(static_cast<std::size_t>(S * q * q), 0.0);
    auto run = [&](lint s) {
        std::vector<double> buf(static_cast<std::size_t>(rows * q));
        double* slot = slots.data() + s * q * q;
        for (lint c = s; c < nch; c += S) {
            const lint t0 = c * rows, t1 = std::min(m, t0 + rows), len = t1 - t0;
            pack(t0, t1, buf.data());
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, static_cast<int>(q), static_cast<int>(len), 1.0,
                        buf.data(), static_cast<int>(q), 1.0, slot, static_cast<int>(q));
        }
    };
    for_chunks(S, double(m) * double(q) * double(q) / 16.0, run);
    for (lint s = 0; s < S; ++s)
        for (lint j = 0; j < q; ++j)
            for (lint i = 0; i <= j; ++i) G[j * q + i] += slots[static_cast<std::size_t>(s * q * q + j * q + i)];
    for (lint j = 0; j < q; ++j)
        for (lint i = j + 1; i < q; ++i) G[j * q + i] = G[i * q + j];
}

// ---- one regression's data

struct Src {
    lint n, p;
    const std::byte* y;
    std::int64_t ys;
    const std::byte* X;
    std::int64_t xrs, xcs;
    const std::byte* w;   // null: unit weights
    std::int64_t ws;
    const std::byte* tw;  // null: no test rows
    std::int64_t tws;
    double x(lint i, lint j) const { return rd(X + i * xrs + j * xcs); }
    double yv(lint i) const { return rd(y + i * ys); }
    double wv(lint i) const { return w ? rd(w + i * ws) : 1.0; }
    double twv(lint i) const { return rd(tw + i * tws); }
};

// Inf is an error, NaN makes the results NaN (returns -1), negative weights are an error,
// the GLM's y must lie in the family's range.
int check(const Src& s, int fam) {
    bool nan = false;
    for (lint i = 0; i < s.n; ++i) {
        const double v = s.yv(i);
        if (std::isnan(v)) nan = true;
        else if (std::isinf(v)) return eInfY;
    }
    lsq::Data d{};
    d.X = s.X;
    d.xrs = s.xrs;
    d.xcs = s.xcs;
    if (const int f = lsq::scan_x(d, s.n, s.p, nan)) return f;
    if (s.w)
        for (lint i = 0; i < s.n; ++i) {
            const double v = s.wv(i);
            if (std::isnan(v)) nan = true;
            else if (std::isinf(v)) return eInfW;
            else if (v < 0) return eNegW;
        }
    if (nan) return -1;
    if (fam != fGaussian)
        for (lint i = 0; i < s.n; ++i) {
            const double v = s.yv(i);
            if (fam == fBinomial ? !(v == 0.0 || v == 1.0) : !(v >= 0.0)) return eBadY;
        }
    return 0;
}

// The fit's rows, weights, centering and scales.
struct Prep {
    lint m = 0;                  // the rows with w > 0
    std::vector<lint> idx;       // their indices
    std::vector<double> wt;      // their weights
    double sw = 0;               // the sum of the weights
    std::vector<double> cx, sc;  // per column: centering constant (0 without the intercept), scale
    std::vector<lint> cols;      // the columns in the fit (not constant), in order
    double cy = 0;               // y's centering constant (0 without the intercept)
};

// Rows, weights and the weighted means (double sums over fixed chunks; the Gram pass's
// centering correction makes them exact to rounding). A constant column (or, without the
// intercept, a zero one) is out of the fit: its coefficient is 0.
void prep_rows(const Src& s, bool intercept, Prep& P) {
    P.idx.clear();
    P.wt.clear();
    for (lint i = 0; i < s.n; ++i) {
        const double w = s.wv(i);
        if (w > 0) {
            P.idx.push_back(i);
            P.wt.push_back(w);
        }
    }
    P.m = static_cast<lint>(P.idx.size());
    DD sw{0.0, 0.0};
    for (const double w : P.wt) sw = sw + DD{w, 0.0};
    P.sw = sw.hi;
    const lint p = s.p, m = P.m;
    if (m == 0) return;
    // per column: sum w x, whether every value equals the first row's; y's sum
    const lint len = 3 * p + 1;
    std::vector<double> acc(static_cast<std::size_t>(len));
    const lint nch = std::max<lint>(1, (m + kRowChunk - 1) / kRowChunk);
    std::vector<double> parts(static_cast<std::size_t>(nch * len));
    for_chunks(nch, double(m) * p, [&](lint c) {
        double* q = parts.data() + c * len;
        const lint i0 = P.idx[0];
        for (lint j = 0; j < p; ++j) q[j] = 0.0, q[p + j] = s.x(i0, j), q[2 * p + j] = 1.0;
        q[3 * p] = 0.0;
        for (lint t = c * kRowChunk; t < std::min(m, (c + 1) * kRowChunk); ++t) {
            const lint i = P.idx[t];
            const double w = P.wt[t];
            const std::byte* row = s.X + i * s.xrs;
            for (lint j = 0; j < p; ++j) {
                const double x = rd(row + j * s.xcs);
                q[j] += w * x;
                if (x != q[p + j]) q[2 * p + j] = 0.0;
            }
            q[3 * p] += w * s.yv(i);
        }
    });
    for (lint j = 0; j < p; ++j) acc[j] = 0.0, acc[p + j] = parts[p + j], acc[2 * p + j] = 1.0;
    acc[3 * p] = 0.0;
    for (lint c = 0; c < nch; ++c) {
        const double* q = parts.data() + c * len;
        for (lint j = 0; j < p; ++j) {
            acc[j] += q[j];
            if (q[2 * p + j] == 0.0) acc[2 * p + j] = 0.0;
        }
        acc[3 * p] += q[3 * p];
    }
    P.cx.assign(static_cast<std::size_t>(p), 0.0);
    P.sc.assign(static_cast<std::size_t>(p), 1.0);
    P.cols.clear();
    for (lint j = 0; j < p; ++j) {
        const bool constant = acc[2 * p + j] == 1.0;
        if (intercept) {
            P.cx[j] = constant ? acc[p + j] : acc[j] / P.sw;
            if (!constant) P.cols.push_back(j);
        } else if (!(constant && acc[p + j] == 0.0)) {
            P.cols.push_back(j);
        }
    }
    P.cy = intercept ? acc[3 * p] / P.sw : 0.0;
}

// The centered, whitened and scaled design of the fit's columns: pack rows [t0, t1) of
// sqrt(f_t) (x - c)/s (and y - cy in column pc when `with_y`, a column sqrt(f_t) in column
// pc + 1 when `with_one`) into buf (column-major, (t1 - t0) rows).
struct Packer {
    const Src& s;
    const Prep& P;
    const double* f;      // per fit row: the whitening weight (w or w / sw ...)
    const double* yv;     // per input row: the response (null: s.y)
    double cy;
    bool with_y, with_one;
    bool rowmajor = false;  // rows contiguous (len x q row-major: the Gram's layout)
    void operator()(lint t0, lint t1, double* buf) const {
        const lint pc = static_cast<lint>(P.cols.size());
        std::vector<double> cc(static_cast<std::size_t>(pc)), is(static_cast<std::size_t>(pc));
        std::vector<std::int64_t> off(static_cast<std::size_t>(pc));
        for (lint q = 0; q < pc; ++q) {
            const lint j = P.cols[q];
            cc[q] = P.cx[j];
            is[q] = 1.0 / P.sc[j];
            off[q] = j * s.xcs;
        }
        const lint yc = pc, oc = pc + (with_y ? 1 : 0), q_all = oc + (with_one ? 1 : 0);
        // element (t, q) at out[q * cs]: cs = len column-major, 1 row-major (row stride q_all)
        const lint cs = rowmajor ? 1 : t1 - t0, rs = rowmajor ? q_all : 1;
        for (lint t = t0; t < t1; ++t) {
            const lint i = P.idx[t];
            const double r = std::sqrt(f[t]);
            const std::byte* row = s.X + i * s.xrs;
            double* out = buf + (t - t0) * rs;
            for (lint q = 0; q < pc; ++q) out[q * cs] = r * ((rd(row + off[q]) - cc[q]) * is[q]);
            if (with_y) out[yc * cs] = r * ((yv ? yv[i] : s.yv(i)) - cy);
            if (with_one) out[oc * cs] = r;
        }
    }
};

// ---- coordinate descent on 1/2 sum omega (v - Z b)^2 + l1 ||b||_1 + l2/2 ||b||^2

struct Solver {
    lint pc = 0;
    bool cov = true;
    // covariance: G = Z^T Omega Z, c = Z^T Omega v; naive: Z (m x pc, sqrt(omega) folded in), rv
    std::vector<double> G, c;
    lint m = 0;
    std::vector<double> Z, rv;
    double vv = 0;  // sum omega v^2
    double R2 = 0;  // sum omega r^2 at the last gap()
    std::vector<double> d, b, Gb, r, xta;

    void init_state() {
        d.assign(static_cast<std::size_t>(pc), 0.0);
        if (cov) {
            for (lint j = 0; j < pc; ++j) d[j] = G[j * pc + j];
        } else {
            for (lint j = 0; j < pc; ++j) d[j] = cblas_ddot(static_cast<int>(m), Z.data() + j * m, 1, Z.data() + j * m, 1);
        }
        if (b.size() != static_cast<std::size_t>(pc)) b.assign(static_cast<std::size_t>(pc), 0.0);
        xta.assign(static_cast<std::size_t>(pc), 0.0);
        refresh();
    }
    // Gb = G b (covariance) or r = v - Z b (naive) from scratch
    void refresh() {
        if (cov) {
            Gb.assign(static_cast<std::size_t>(pc), 0.0);
            for (lint k = 0; k < pc; ++k)
                if (b[k] != 0.0) cblas_daxpy(static_cast<int>(pc), b[k], G.data() + k * pc, 1, Gb.data(), 1);
        } else {
            r = rv;
            for (lint k = 0; k < pc; ++k)
                if (b[k] != 0.0) cblas_daxpy(static_cast<int>(m), -b[k], Z.data() + k * m, 1, r.data(), 1);
        }
    }
    // the sweep over `list`; dmax, wmax: the largest change and coefficient
    void sweep(const lint* list, lint cnt, double l1, double l2, double& dmax, double& wmax) {
        dmax = 0.0;
        wmax = 0.0;
        for (lint q = 0; q < cnt; ++q) {
            const lint j = list[q];
            const double dj = d[j];
            if (!(dj > 0.0)) continue;
            const double bj = b[j];
            const double g = cov ? c[j] - Gb[j] : cblas_ddot(static_cast<int>(m), Z.data() + j * m, 1, r.data(), 1);
            const double tmp = g + dj * bj;
            const double a = std::fabs(tmp) - l1;
            const double nb = a > 0.0 ? std::copysign(a, tmp) / (dj + l2) : 0.0;
            if (nb != bj) {
                const double dl = nb - bj;
                if (cov) cblas_daxpy(static_cast<int>(pc), dl, G.data() + j * pc, 1, Gb.data(), 1);
                else cblas_daxpy(static_cast<int>(m), -dl, Z.data() + j * m, 1, r.data(), 1);
                b[j] = nb;
            }
            dmax = std::max(dmax, std::fabs(nb - bj));
            wmax = std::max(wmax, std::fabs(nb));
        }
    }
    // scikit-learn's duality gap (formulation A with an L1 term, B for ridge, the gradient's
    // squared norm for least squares) at b; xta = Z^T Omega r on return (then minus l2 b)
    double gap(double l1, double l2) {
        refresh();
        double R2, D;
        if (cov) {
            double bc = 0.0, bGb = 0.0;
            for (lint j = 0; j < pc; ++j) bc += b[j] * c[j], bGb += b[j] * Gb[j];
            R2 = std::max(0.0, vv - 2.0 * bc + bGb);
            for (lint j = 0; j < pc; ++j) xta[j] = c[j] - Gb[j];
        } else {
            R2 = cblas_ddot(static_cast<int>(m), r.data(), 1, r.data(), 1);
            if (pc > 0)
                cblas_dgemv(CblasColMajor, CblasTrans, static_cast<int>(m), static_cast<int>(pc), 1.0, Z.data(),
                            static_cast<int>(m), r.data(), 1, 0.0, xta.data(), 1);
        }
        D = 0.0;
        for (lint j = 0; j < pc; ++j) D += b[j] * xta[j];
        this->R2 = R2;
        return gap_from(l1, l2, R2, D);
    }
    // The gap from R2 = sum omega r^2, D = sum omega r (v - r) (the fitted part; b^T xta plus
    // the intercept's term) and xta = Z^T Omega r. Written so that no terms of the size of
    // v^T Omega v cancel: at the scale s of the dual point, primal - dual =
    // R2 (1 - s)^2 / 2 + l2 ||b||^2 (1 + s^2) / 2 + l1 ||b||_1 - s D (formulation A), and
    // ||xta - l2 b||^2 / (2 l2) + b^T xta - D (B).
    double gap_from(double l1, double l2, double R2, double D) {
        double l2n = 0.0, l1n = 0.0, bx = 0.0;
        for (lint j = 0; j < pc; ++j) l2n += b[j] * b[j], l1n += std::fabs(b[j]), bx += b[j] * xta[j];
        if (l1 > 0.0) {
            double dn = 0.0;
            for (lint j = 0; j < pc; ++j) {
                xta[j] -= l2 * b[j];
                if (d[j] > 0.0) dn = std::max(dn, std::fabs(xta[j]));
            }
            const double s = dn > l1 ? l1 / dn : 1.0;
            return 0.5 * R2 * (1.0 - s) * (1.0 - s) + 0.5 * l2 * l2n * (1.0 + s * s) + l1 * l1n - s * D;
        }
        if (l2 > 0.0) {
            double q = 0.0;
            for (lint j = 0; j < pc; ++j) q += (xta[j] - l2 * b[j]) * (xta[j] - l2 * b[j]);
            return q / (2.0 * l2) + bx - D;
        }
        double dn2 = 0.0;
        for (lint j = 0; j < pc; ++j) dn2 += xta[j] * xta[j];
        return dn2;
    }
};

struct CDResult {
    int n_iter = 0;
    double gap = kNaN;
    bool converged = false;
    bool polished = false;  // the last iterate is the KKT polish's (polish_fn succeeded)
};

// Cyclic coordinate descent from s.b: full sweeps, and between them sweeps over the nonzero
// coefficients until their largest change is <= dtol x the largest coefficient. The gap is
// tested at the start and after a full sweep that small (scikit-learn's rule). With polish_fn
// (the KKT polish, which replaces s.b on success), a full sweep that leaves the signs of every
// coefficient as the previous one did (the active set has settled) also tries the polish, once
// per sign pattern: when it succeeds and its gap passes, that is the solution (coordinate
// descent converges only linearly on an ill-conditioned active set; the polish solves it).
CDResult cd_solve(Solver& s, double l1, double l2, int max_iter, double gap_tol, double dtol,
                  const std::function<bool()>* polish_fn = nullptr) {
    CDResult out;
    out.gap = s.gap(l1, l2);
    if (out.gap <= gap_tol) {
        out.converged = true;
        return out;
    }
    std::vector<lint> all(static_cast<std::size_t>(s.pc)), act;
    std::iota(all.begin(), all.end(), lint(0));
    std::vector<std::int8_t> pat, prev, tried;
    auto pattern = [&](std::vector<std::int8_t>& v) {
        v.resize(static_cast<std::size_t>(s.pc));
        for (lint j = 0; j < s.pc; ++j) v[j] = static_cast<std::int8_t>((s.b[j] > 0) - (s.b[j] < 0));
    };
    // after a sweep: the polish when the signs settled (true: converged on it)
    auto settle = [&]() {
        if (!polish_fn) return false;
        pattern(pat);
        bool done = false;
        if (pat == prev && pat != tried) {
            tried = pat;
            if ((*polish_fn)()) {
                out.gap = s.gap(l1, l2);
                done = out.gap <= gap_tol;
            }
        }
        prev.swap(pat);
        return done;
    };
    int it = 0;
    while (true) {
        double dmax, wmax;
        s.sweep(all.data(), s.pc, l1, l2, dmax, wmax);
        ++it;
        if (settle()) {
            out.converged = out.polished = true;
            break;
        }
        if (wmax == 0.0 || dmax <= dtol * wmax || it >= max_iter) {
            out.gap = s.gap(l1, l2);
            if (out.gap <= gap_tol) {
                out.converged = true;
                break;
            }
            if (it >= max_iter) break;
        }
        bool done = false;
        while (it < max_iter) {
            act.clear();
            for (lint j = 0; j < s.pc; ++j)
                if (s.b[j] != 0.0) act.push_back(j);
            if (act.empty()) break;
            s.sweep(act.data(), static_cast<lint>(act.size()), l1, l2, dmax, wmax);
            ++it;
            if (settle()) {
                done = true;
                break;
            }
            if (wmax == 0.0 || dmax <= dtol * wmax) break;
        }
        if (done) {
            out.converged = out.polished = true;
            break;
        }
        if (it >= max_iter) {
            out.gap = s.gap(l1, l2);
            out.converged = out.gap <= gap_tol;
            break;
        }
    }
    out.n_iter = it;
    return out;
}

// H = Z_A^T Omega Z_A + l2 I (a x a, column-major) for the active columns A of the solver.
void active_gram(const Solver& s, const std::vector<lint>& A, double l2, std::vector<double>& H) {
    const lint a = static_cast<lint>(A.size());
    H.assign(static_cast<std::size_t>(a * a), 0.0);
    if (s.cov) {
        for (lint q = 0; q < a; ++q)
            for (lint u = 0; u < a; ++u) H[q * a + u] = s.G[A[q] * s.pc + A[u]];
    } else {
        std::vector<double> ZA(static_cast<std::size_t>(s.m * a));
        for (lint q = 0; q < a; ++q) std::copy_n(s.Z.data() + A[q] * s.m, s.m, ZA.data() + q * s.m);
        cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, static_cast<int>(a), static_cast<int>(s.m), 1.0, ZA.data(),
                    static_cast<int>(s.m), 0.0, H.data(), static_cast<int>(a));
        for (lint q = 0; q < a; ++q)
            for (lint u = q + 1; u < a; ++u) H[u * a + q] = H[q * a + u];  // H[col u, row q] from lower
    }
    for (lint q = 0; q < a; ++q) H[q * a + q] += l2;
}

// The effective degrees of freedom tr(Z_A (Z_A^T Omega Z_A + l2 I)^-1 Z_A^T Omega) =
// a - l2 tr(H^-1) (the number of nonzero coefficients for the lasso).
double effective_df(std::vector<double> H, lint a, double l2) {
    if (a == 0) return 0.0;
    if (l2 == 0.0) return double(a);
    if (linalg::lapack::potrf('U', a, H.data(), a) != 0) return kNaN;
    if (linalg::lapack::potri_upper(a, H.data(), a) != 0) return kNaN;
    double tr = 0.0;
    for (lint q = 0; q < a; ++q) tr += H[q * a + q];
    return double(a) - l2 * tr;
}

// ---- the linear model's row passes in the centered coordinates

// The refinement's residual pass (G51: ridge, and the lasso's and elastic net's KKT polish):
// over the fit's rows, out = [sum w r, sum w z_{A_q} r (a values), sum w r^2, sum w r (y - cy)]
// with r_t = (y - cy) - g0 - sum_q z_q bA_q in double-double (G1's Dot2): y - cy and x - c exact (two_sum), z = (x - c)/s
// and every product and sum in double-double, rounded once at the end. The refinement's fixed
// point is then the solution of the exact problem (to the final rounding), not of the design
// rounded to doubles: its accuracy no longer degrades with the solution's condition number.
void residual_pass_dd(const Src& s, const Prep& P, const std::vector<lint>& Acols, const double* bA, double g0,
                      std::vector<double>& out) {
    const lint a = static_cast<lint>(Acols.size());
    const lint len = a + 3;
    out.assign(static_cast<std::size_t>(len), 0.0);
    std::vector<double> cA(static_cast<std::size_t>(a)), sA(static_cast<std::size_t>(a));
    std::vector<std::int64_t> off(static_cast<std::size_t>(a));
    for (lint q = 0; q < a; ++q) cA[q] = P.cx[Acols[q]], sA[q] = P.sc[Acols[q]], off[q] = Acols[q] * s.xcs;
    const lint nch = std::max<lint>(1, (P.m + kRowChunk - 1) / kRowChunk);
    std::vector<double> parts(static_cast<std::size_t>(nch * len * 2), 0.0);
    for_chunks(nch, double(P.m) * double(a + 1) * 8.0, [&](lint c) {
        std::vector<DD> acc(static_cast<std::size_t>(len), DD{0.0, 0.0}), z(static_cast<std::size_t>(a));
        for (lint t = c * kRowChunk; t < std::min(P.m, (c + 1) * kRowChunk); ++t) {
            const lint i = P.idx[t];
            const std::byte* row = s.X + i * s.xrs;
            const DD v = special::dd::two_sum(s.yv(i), -P.cy);
            DD r = v - DD{g0, 0.0};
            for (lint q = 0; q < a; ++q) {
                const DD u = special::dd::two_sum(rd(row + off[q]), -cA[q]);
                z[q] = sA[q] == 1.0 ? u : special::dd::operator/(u, sA[q]);
                r = r - z[q] * bA[q];
            }
            const DD wr = r * P.wt[t];
            acc[0] = acc[0] + wr;
            for (lint q = 0; q < a; ++q) acc[1 + q] = acc[1 + q] + wr * z[q];
            acc[a + 1] = acc[a + 1] + wr * r;
            acc[a + 2] = acc[a + 2] + wr * v;
        }
        double* pt = parts.data() + c * len * 2;
        for (lint q = 0; q < len; ++q) pt[2 * q] = acc[q].hi, pt[2 * q + 1] = acc[q].lo;
    });
    for (lint q = 0; q < len; ++q) {
        DD tot{0.0, 0.0};
        for (lint c = 0; c < nch; ++c) tot = tot + DD{parts[(c * len + q) * 2], parts[(c * len + q) * 2 + 1]};
        out[q] = tot.hi + tot.lo;
    }
}

struct PolishOut {
    bool ok = false;
    bool refined = false;    // refinement passes over the data were made
    double g0 = 0.0;         // the intercept in the centered coordinates (refined)
    lint a = 0;              // the active columns
    std::vector<double> H;   // Z_A^T Omega Z_A + l2 I
};

// The KKT polish of a converged iterate s.b (see the file comment): the system on the active
// set solved by Cholesky of its Gram block, refined with double-double residuals from the
// data (G51) when the estimated condition number of that block exceeds refine_cond times its
// size (the Gram solve's error is about kappa u). On success s.b holds the polished coefficients.
PolishOut polish(Solver& s, const Src& src, const Prep& P, bool intercept, double l1, double l2, double refine_cond) {
    PolishOut o;
    std::vector<lint> A;
    std::vector<double> sg;
    for (lint j = 0; j < s.pc; ++j)
        // without an L1 term every column is in the system and the signs are free
        if (s.b[j] != 0.0 || (l1 == 0.0 && s.d[j] > 0.0)) A.push_back(j), sg.push_back(s.b[j] > 0 ? 1.0 : -1.0);
    const lint a = static_cast<lint>(A.size());
    o.a = a;
    active_gram(s, A, l2, o.H);
    std::vector<double> L = o.H, bA(static_cast<std::size_t>(a));
    double g0 = 0.0;
    if (a > 0) {
        if (linalg::lapack::potrf('L', a, L.data(), a) != 0) return o;
        // c_A - l1 sign
        if (s.cov) {
            for (lint q = 0; q < a; ++q) bA[q] = s.c[A[q]] - l1 * sg[q];
        } else {
            for (lint q = 0; q < a; ++q)
                bA[q] = cblas_ddot(static_cast<int>(s.m), s.Z.data() + A[q] * s.m, 1, s.rv.data(), 1) - l1 * sg[q];
        }
        linalg::lapack::potrs('L', a, 1, L.data(), a, bA.data(), a);
        const double rc = linalg::lapack::trcon_1('L', a, L.data(), a);
        const double kappa = rc > 0.0 ? 1.0 / (rc * rc) : INFINITY;
        if (kappa > refine_cond * double(a)) {
            o.refined = true;
            std::vector<lint> Acols(static_cast<std::size_t>(a));
            for (lint q = 0; q < a; ++q) Acols[q] = P.cols[A[q]];
            std::vector<double> sums, gA(static_cast<std::size_t>(a));
            double prev = INFINITY;
            for (int step = 0; step < kPolishSteps; ++step) {
                residual_pass_dd(src, P, Acols, bA.data(), g0, sums);
                const double dg0 = intercept ? sums[0] / P.sw : 0.0;
                double dmax = 0.0, bmax = 0.0;
                for (lint q = 0; q < a; ++q) gA[q] = sums[1 + q] / P.sw - l2 * bA[q] - l1 * sg[q];
                linalg::lapack::potrs('L', a, 1, L.data(), a, gA.data(), a);
                for (lint q = 0; q < a; ++q) {
                    bA[q] += gA[q];
                    dmax = std::max(dmax, std::fabs(gA[q]));
                    bmax = std::max(bmax, std::fabs(bA[q]));
                }
                g0 += dg0;
                if (dmax <= 4 * kU * bmax || dmax >= 0.5 * prev) break;
                prev = dmax;
            }
        }
    }
    for (lint q = 0; q < a; ++q)
        if (l1 > 0.0 && !(bA[q] * sg[q] > 0.0)) return o;
    // the inactive columns' KKT conditions |z_j^T Omega r| <= l1
    std::vector<std::uint8_t> in(static_cast<std::size_t>(s.pc), 0);
    for (lint q = 0; q < a; ++q) in[A[q]] = 1;
    std::vector<double> grad(static_cast<std::size_t>(s.pc), 0.0);
    if (s.cov) {
        for (lint j = 0; j < s.pc; ++j) {
            if (in[j]) continue;
            double g = s.c[j];
            for (lint q = 0; q < a; ++q) g -= s.G[A[q] * s.pc + j] * bA[q];
            grad[j] = g;
        }
    } else {
        std::vector<double> r = s.rv;
        for (lint q = 0; q < a; ++q) cblas_daxpy(static_cast<int>(s.m), -bA[q], s.Z.data() + A[q] * s.m, 1, r.data(), 1);
        if (s.pc > 0)
            cblas_dgemv(CblasColMajor, CblasTrans, static_cast<int>(s.m), static_cast<int>(s.pc), 1.0, s.Z.data(),
                        static_cast<int>(s.m), r.data(), 1, 0.0, grad.data(), 1);
    }
    for (lint j = 0; j < s.pc; ++j) {
        if (in[j] || !(s.d[j] > 0.0)) continue;
        if (!(std::fabs(grad[j]) <= l1 * (1.0 + kKktRel) + kKktAbs * std::sqrt(s.d[j] * s.vv))) return o;
    }
    std::fill(s.b.begin(), s.b.end(), 0.0);
    for (lint q = 0; q < a; ++q) s.b[A[q]] = bA[q];
    o.ok = true;
    o.g0 = g0;
    return o;
}

// beta_j = b_j / s_j and the intercept cy + g0 - sum c_j beta_j (Dot2) into params (k).
void write_params(const Prep& P, lint p, bool intercept, const std::vector<double>& b, double g0, double* params) {
    double* beta = params + (intercept ? 1 : 0);
    std::fill(beta, beta + p, 0.0);
    for (std::size_t q = 0; q < P.cols.size(); ++q) beta[P.cols[q]] = b[q] / P.sc[P.cols[q]];
    if (!intercept) return;
    double S = P.cy, E = g0;
    for (lint j = 0; j < p; ++j) {
        if (beta[j] == 0.0) continue;
        double hp, ep, q;
        two_prod(-P.cx[j], beta[j], hp, ep);
        two_sum(S, hp, S, q);
        E += q + ep;
    }
    params[0] = S + E;
}

void fill_nan(lint L, lint k, double* params, double* stats, double nobs) {
    std::fill(params, params + L * k, kNaN);
    for (lint l = 0; l < L; ++l) {
        double* st = stats + l * pNStat;
        std::fill(st, st + pNStat, kNaN);
        st[pIsNaN] = 1.0;
        st[pNobs] = nobs;
        st[pPolished] = 0.0;
    }
}

// The squared error over the test rows (tw > 0) of every alpha's params (L x k), into
// stats[pTestSse], stats[pTestSw]: chunks of rows by dgemm, added in order.
void test_errors(const Src& s, bool intercept, lint L, const double* params, double* stats, int fam = fGaussian) {
    if (!s.tw) return;
    const lint n = s.n, p = s.p, k = p + (intercept ? 1 : 0);
    std::vector<lint> rows;
    for (lint i = 0; i < n; ++i)
        if (s.twv(i) > 0) rows.push_back(i);
    const lint nt = static_cast<lint>(rows.size());
    std::vector<double> B(static_cast<std::size_t>(p * L));
    for (lint l = 0; l < L; ++l)
        for (lint j = 0; j < p; ++j) B[l * p + j] = params[l * k + (intercept ? 1 : 0) + j];
    const lint len = L + 1;
    std::vector<double> tot(static_cast<std::size_t>(len));
    chunk_sum(nt, len, tot.data(), double(p) * double(L), [&](lint t0, lint t1, double* part) {
        const lint cnt = t1 - t0;
        std::vector<double> Xc(static_cast<std::size_t>(cnt * std::max<lint>(p, 1))), Pm(static_cast<std::size_t>(cnt * L));
        for (lint t = t0; t < t1; ++t)
            for (lint j = 0; j < p; ++j) Xc[j * cnt + (t - t0)] = s.x(rows[t], j);
        if (p > 0)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(cnt), static_cast<int>(L),
                        static_cast<int>(p), 1.0, Xc.data(), static_cast<int>(cnt), B.data(), static_cast<int>(p), 0.0,
                        Pm.data(), static_cast<int>(cnt));
        else
            std::fill(Pm.begin(), Pm.end(), 0.0);
        for (lint t = t0; t < t1; ++t) {
            const lint i = rows[t];
            const double tw = s.twv(i), y = s.yv(i);
            part[L] += tw;
            for (lint l = 0; l < L; ++l) {
                const double eta = Pm[l * cnt + (t - t0)] + (intercept ? params[l * k] : 0.0);
                double e;
                if (fam == fGaussian) {
                    const double r = y - eta;
                    e = r * r;
                } else if (fam == fBinomial) {
                    // -2 log likelihood of y in {0, 1}: 2 log(1 + e^{-+eta})
                    const double z = y == 1.0 ? -eta : eta;
                    e = 2.0 * (z > 0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z)));
                } else {
                    const double mu = std::exp(eta);
                    e = y == 0.0 ? 2.0 * mu : 2.0 * (y * std::log(y / mu) - (y - mu));
                }
                part[l] += tw * e;
            }
        }
    });
    for (lint l = 0; l < L; ++l) {
        stats[l * pNStat + pTestSse] = tot[l];
        stats[l * pNStat + pTestSw] = tot[L];
    }
}

// The alphas' order: largest first (ties in input order).
std::vector<lint> path_order(const double* alphas, std::int64_t as, lint L) {
    std::vector<lint> ord(static_cast<std::size_t>(L));
    std::iota(ord.begin(), ord.end(), lint(0));
    std::stable_sort(ord.begin(), ord.end(),
                     [&](lint a, lint b) { return rd(reinterpret_cast<const std::byte*>(alphas) + a * as) >
                                                  rd(reinterpret_cast<const std::byte*>(alphas) + b * as); });
    return ord;
}

struct Opts {
    double l1_ratio;
    bool intercept, standardize;
    int max_iter;
    double tol;
    bool polish;
    int family;  // GLM
    double refine_cond;  // the KKT polish refines with the data above this condition number
};

// The centered Gram (with y and a column of sqrt(w)) of the fit's columns, the exact
// centering correction, the scales; fills the covariance solver (or, naive, the packed Z).
void build(const Src& src, Prep& P, const Opts& op, Solver& s) {
    const lint m = P.m, pc0 = static_cast<lint>(P.cols.size());
    std::vector<double> f(P.wt);
    const bool cov = pc0 <= kCovMaxCols;
    const lint q = pc0 + 2;
    // with unit scales: the Gram of sqrt(w) [(x - c), y - cy, 1] (covariance updates), or (naive
    // updates) only its diagonal and the sums of sqrt(w) (x - c), sqrt(w) (y - cy): one pass
    std::vector<double> Gq(static_cast<std::size_t>(cov ? q * q : 2 * q));
    if (cov) {
        gram(m, q, Gq.data(), Packer{src, P, f.data(), nullptr, P.cy, true, true, true});
    } else {
        chunk_sum(m, 2 * q, Gq.data(), double(pc0), [&](lint t0, lint t1, double* part) {
            for (lint t = t0; t < t1; ++t) {
                const lint i = P.idx[t];
                const double w = P.wt[t];
                for (lint a = 0; a < pc0; ++a) {
                    const double u = src.x(i, P.cols[a]) - P.cx[P.cols[a]];
                    part[a] += w * u * u;
                    part[q + a] += w * u;
                }
                const double u = src.yv(i) - P.cy;
                part[pc0] += w * u * u;
                part[q + pc0] += w * u;
            }
        });
    }
    const double sw = P.sw;
    // the exact centering: Gram of sqrt(w) (u - mean) = G - S1 S1^T / sw, S1 = the last column
    const lint yi = pc0, oi = pc0 + 1;
    std::vector<double> S1(static_cast<std::size_t>(pc0 + 1));
    for (lint a = 0; a <= pc0; ++a) S1[a] = op.intercept ? (cov ? Gq[oi * q + a] : Gq[q + a]) : 0.0;
    auto Gc = [&](lint a, lint b) {
        const double g = cov ? Gq[b * q + a] : (a == b ? Gq[a] : kNaN);
        return g - S1[a] * S1[b] / sw;
    };
    if (op.intercept) {
        for (lint a = 0; a < pc0; ++a) P.cx[P.cols[a]] += S1[a] / sw;
        P.cy += S1[yi] / sw;
    }
    // scales; a column whose centered norm is 0 leaves the fit
    std::vector<lint> keep;
    std::vector<double> scale;
    for (lint a = 0; a < pc0; ++a) {
        const double var = Gc(a, a) / sw;
        if (!(var > 0.0)) continue;
        keep.push_back(a);
        scale.push_back(op.standardize ? std::sqrt(var) : 1.0);
    }
    const lint pc = static_cast<lint>(keep.size());
    std::vector<lint> cols(static_cast<std::size_t>(pc));
    for (lint a = 0; a < pc; ++a) {
        cols[a] = P.cols[keep[a]];
        P.sc[cols[a]] = scale[a];
    }
    s.pc = pc;
    s.vv = std::max(0.0, Gc(yi, yi)) / sw;
    s.cov = cov;
    if (s.cov) {
        s.G.assign(static_cast<std::size_t>(pc * pc), 0.0);
        s.c.assign(static_cast<std::size_t>(pc), 0.0);
        for (lint a = 0; a < pc; ++a) {
            for (lint b = 0; b < pc; ++b) s.G[b * pc + a] = Gc(keep[a], keep[b]) / (sw * scale[a] * scale[b]);
            s.c[a] = Gc(keep[a], yi) / (sw * scale[a]);
        }
    }
    P.cols = cols;
    if (!s.cov) {
        // the packed design sqrt(omega) z (m x pc) and sqrt(omega) v
        for (lint t = 0; t < m; ++t) f[t] = P.wt[t] / sw;
        s.m = m;
        s.Z.assign(static_cast<std::size_t>(m * pc), 0.0);
        s.rv.assign(static_cast<std::size_t>(m), 0.0);
        const lint nch = std::max<lint>(1, (m + kRowChunk - 1) / kRowChunk);
        for_chunks(nch, double(m) * pc, [&](lint c) {
            const lint t0 = c * kRowChunk, t1 = std::min(m, t0 + kRowChunk);
            for (lint t = t0; t < t1; ++t) {
                const lint i = P.idx[t];
                const double r = std::sqrt(f[t]);
                for (lint a = 0; a < pc; ++a) s.Z[a * m + t] = r * ((src.x(i, cols[a]) - P.cx[cols[a]]) / P.sc[cols[a]]);
                s.rv[t] = r * (src.yv(i) - P.cy);
            }
        });
    }
    s.b.assign(static_cast<std::size_t>(pc), 0.0);
    s.init_state();
}

// One regression's elastic net path.
int enet_one(const Opts& op, const Src& src, const double* alphas, std::int64_t as, lint L, double* params,
             double* stats) {
    const lint p = src.p, k = p + (op.intercept ? 1 : 0);
    const int chk = check(src, fGaussian);
    if (chk > 0) return chk;
    if (chk < 0) {
        fill_nan(L, k, params, stats, kNaN);
        return 0;
    }
    Prep P;
    prep_rows(src, op.intercept, P);
    if (P.m == 0) {
        fill_nan(L, k, params, stats, 0.0);
        return 0;
    }
    Solver s;
    build(src, P, op, s);
    const double rho = op.l1_ratio;
    const double tss = s.vv * P.sw;
    for (const lint l : path_order(alphas, as, L)) {
        const double alpha = rd(reinterpret_cast<const std::byte*>(alphas) + l * as);
        const double l1 = alpha * rho, l2 = alpha * (1.0 - rho);
        PolishOut po;
        const std::function<bool()> try_polish = [&]() {
            po = polish(s, src, P, op.intercept, l1, l2, op.refine_cond);
            return po.ok;
        };
        CDResult cd = cd_solve(s, l1, l2, op.max_iter, op.tol * s.vv, op.tol, op.polish ? &try_polish : nullptr);
        if (!cd.polished) po = PolishOut{};
        if (cd.converged && op.polish && !cd.polished) {
            po = polish(s, src, P, op.intercept, l1, l2, op.refine_cond);
            // a failed polish means the iterate's active set or signs are not the solution's
            // yet (the gap test passed first): tighter coordinate descent, then again
            double t = op.tol;
            for (int retry = 0; retry < kPolishRetries && !po.ok && cd.n_iter < op.max_iter; ++retry) {
                t *= kRetryTol;
                const CDResult more = cd_solve(s, l1, l2, op.max_iter - cd.n_iter, t * s.vv, t);
                cd.n_iter += more.n_iter;
                cd.gap = more.gap;
                if (!more.converged) break;
                po = polish(s, src, P, op.intercept, l1, l2, op.refine_cond);
            }
        }
        double* st = stats + l * pNStat;
        std::fill(st, st + pNStat, kNaN);
        st[pNobs] = double(P.m);
        st[pSumW] = P.sw;
        st[pNIter] = cd.n_iter;
        st[pConv] = cd.converged ? 1.0 : 0.0;
        st[pIsNaN] = 0.0;
        st[pTss] = tss;
        st[pPolished] = po.ok ? (po.refined ? 2.0 : 1.0) : 0.0;
        const double g0 = po.ok ? po.g0 : 0.0;
        // the gap and the residual sum of squares at the returned coefficients (the Gram's, or
        // the packed design's, residual)
        st[pGap] = po.ok ? s.gap(l1, l2) : cd.gap;
        st[pRss] = s.R2 * P.sw;
        if (po.ok) {
            st[pDf] = effective_df(po.H, po.a, l2);
        } else {
            std::vector<lint> A;
            for (lint j = 0; j < s.pc; ++j)
                if (s.b[j] != 0.0) A.push_back(j);
            std::vector<double> H;
            active_gram(s, A, l2, H);
            st[pDf] = effective_df(H, static_cast<lint>(A.size()), l2);
        }
        write_params(P, p, op.intercept, s.b, g0, params + l * k);
    }
    test_errors(src, op.intercept, L, params, stats);
    return 0;
}

// ---- ridge: one SVD for every alpha

// The QR of the m x pc column-major A (destroyed) with the vector v (destroyed): R (pc x pc
// upper, column-major), d1 = (Q^T v)[0, pc) and acc += the squares of (Q^T v)[pc, m). Tall
// matrices (at least 4 blocks of max(4096, 4 pc) rows) go by TSQR: each block's QR (on the
// pool), then the QR of the blocks' stacked R (and Q^T v) the same way. The blocks are fixed
// by the sizes, so the bits do not depend on the thread count.
void tsqr(lint m, lint pc, double* A, double* v, double* R, double* d1, DD& acc) {
    const lint rows = std::max<lint>(4096, 4 * pc);
    const lint nb = (m + rows - 1) / rows;
    auto direct = [&](lint mm, double* a, lint lda, double* vv, double* r, double* d, DD& sq, Scratch& sc) {
        std::vector<double> tau(static_cast<std::size_t>(pc));
        linalg::lapack::geqrf(mm, pc, a, lda, tau.data(), sc);
        linalg::lapack::ormqr('T', mm, 1, std::min(mm, pc), a, lda, tau.data(), vv, mm, sc);
        for (lint j = 0; j < pc; ++j)
            for (lint i = 0; i < pc; ++i) r[j * pc + i] = i <= j && i < mm ? a[j * lda + i] : 0.0;
        for (lint i = 0; i < pc; ++i) d[i] = i < mm ? vv[i] : 0.0;
        for (lint t = pc; t < mm; ++t) sq = sq + DD{vv[t], 0.0} * DD{vv[t], 0.0};
    };
    if (nb < 4) {
        Scratch sc;
        direct(m, A, m, v, R, d1, acc, sc);
        return;
    }
    // each block's rows copied out (column-major, its own leading dimension)
    std::vector<double> Rs(static_cast<std::size_t>(nb * pc * pc)), ds(static_cast<std::size_t>(nb * pc)),
        sq(static_cast<std::size_t>(2 * nb));
    for_chunks(nb, double(m) * pc * pc, [&](lint b) {
        const lint t0 = b * rows, t1 = std::min(m, t0 + rows), len = t1 - t0;
        std::vector<double> blk(static_cast<std::size_t>(len * pc)), vb(v + t0, v + t1);
        for (lint j = 0; j < pc; ++j) std::copy_n(A + j * m + t0, len, blk.data() + j * len);
        DD s2{0.0, 0.0};
        Scratch sc;
        direct(len, blk.data(), len, vb.data(), Rs.data() + b * pc * pc, ds.data() + b * pc, s2, sc);
        sq[2 * b] = s2.hi, sq[2 * b + 1] = s2.lo;
    });
    for (lint b = 0; b < nb; ++b) acc = acc + DD{sq[2 * b], sq[2 * b + 1]};
    // the stacked R's (nb pc x pc) and their Q^T v's
    const lint ms = nb * pc;
    std::vector<double> S(static_cast<std::size_t>(ms * pc)), w(static_cast<std::size_t>(ms));
    for (lint b = 0; b < nb; ++b) {
        for (lint j = 0; j < pc; ++j)
            for (lint i = 0; i < pc; ++i) S[j * ms + b * pc + i] = Rs[b * pc * pc + j * pc + i];
        std::copy_n(ds.data() + b * pc, pc, w.data() + b * pc);
    }
    tsqr(ms, pc, S.data(), w.data(), R, d1, acc);
}


int ridge_one(const Opts& op, const Src& src, const double* alphas, std::int64_t as, lint L, double* params,
              double* stats, Scratch& sc) {
    const lint p = src.p, k = p + (op.intercept ? 1 : 0);
    const int chk = check(src, fGaussian);
    if (chk > 0) return chk;
    if (chk < 0) {
        fill_nan(L, k, params, stats, kNaN);
        return 0;
    }
    Prep P;
    prep_rows(src, op.intercept, P);
    if (P.m == 0) {
        fill_nan(L, k, params, stats, 0.0);
        return 0;
    }
    // the centering correction and the scales (one pass of sums per column), then the packed design
    {
        const lint pc0 = static_cast<lint>(P.cols.size());
        const lint len = 2 * pc0 + 2;
        std::vector<double> sums(static_cast<std::size_t>(len));
        chunk_sum(P.m, len, sums.data(), double(pc0), [&](lint t0, lint t1, double* part) {
            for (lint t = t0; t < t1; ++t) {
                const lint i = P.idx[t];
                const double w = P.wt[t];
                for (lint a = 0; a < pc0; ++a) {
                    const double u = src.x(i, P.cols[a]) - P.cx[P.cols[a]];
                    part[a] += w * u;
                    part[pc0 + a] += w * u * u;
                }
                const double u = src.yv(i) - P.cy;
                part[2 * pc0] += w * u;
                part[2 * pc0 + 1] += w * u * u;
            }
        });
        std::vector<lint> cols;
        for (lint a = 0; a < pc0; ++a) {
            const lint j = P.cols[a];
            const double s1 = op.intercept ? sums[a] : 0.0;
            const double var = (sums[pc0 + a] - s1 * s1 / P.sw) / P.sw;
            if (op.intercept) P.cx[j] += s1 / P.sw;
            if (!(var > 0.0)) continue;
            cols.push_back(j);
            P.sc[j] = op.standardize ? std::sqrt(var) : 1.0;
        }
        if (op.intercept) P.cy += sums[2 * pc0] / P.sw;
        P.cols = cols;
    }
    const lint m = P.m, pc = static_cast<lint>(P.cols.size());
    std::vector<double> f(static_cast<std::size_t>(m));
    for (lint t = 0; t < m; ++t) f[t] = P.wt[t] / P.sw;
    std::vector<double> Z(static_cast<std::size_t>(m * std::max<lint>(pc, 1) + m));
    {
        const Packer pk{src, P, f.data(), nullptr, P.cy, true, false};
        const lint nch = std::max<lint>(1, (m + kRowChunk - 1) / kRowChunk);
        // pack into a temporary per chunk and copy: Packer writes (t1 - t0) rows per column
        for_chunks(nch, double(m) * pc, [&](lint c) {
            const lint t0 = c * kRowChunk, t1 = std::min(m, t0 + kRowChunk), len = t1 - t0;
            std::vector<double> buf(static_cast<std::size_t>(len * (pc + 1)));
            pk(t0, t1, buf.data());
            for (lint a = 0; a <= pc; ++a) std::copy_n(buf.data() + a * len, len, Z.data() + a * m + t0);
        });
    }
    double* rv = Z.data() + pc * m;
    // rss_perp: the part of v outside the columns' span; e = U^T v; the singular values and V^T
    const lint r = std::min(m, pc);
    std::vector<double> sv(static_cast<std::size_t>(std::max<lint>(r, 1))), e(static_cast<std::size_t>(std::max<lint>(r, 1))),
        Vt(static_cast<std::size_t>(std::max<lint>(r * pc, 1)));
    double rss_perp = 0.0;
    double vv = 0.0;
    for (lint t = 0; t < m; ++t) vv += rv[t] * rv[t];
    if (pc == 0) {
        rss_perp = vv;
    } else if (m >= pc) {
        DD acc{0.0, 0.0};
        std::vector<double> R(static_cast<std::size_t>(pc * pc), 0.0), U(static_cast<std::size_t>(pc * pc)),
            d1(static_cast<std::size_t>(pc));
        tsqr(m, pc, Z.data(), rv, R.data(), d1.data(), acc);
        rss_perp = acc.hi;
        if (linalg::lapack::gesdd('A', pc, pc, R.data(), pc, sv.data(), U.data(), pc, Vt.data(), pc, sc) != 0)
            return -2;
        for (lint i = 0; i < pc; ++i) e[i] = cblas_ddot(static_cast<int>(pc), U.data() + i * pc, 1, d1.data(), 1);
    } else {
        std::vector<double> U(static_cast<std::size_t>(m * m));
        std::vector<double> v0(rv, rv + m);
        if (linalg::lapack::gesdd('S', m, pc, Z.data(), m, sv.data(), U.data(), m, Vt.data(), m, sc) != 0) return -2;
        for (lint i = 0; i < m; ++i) e[i] = cblas_ddot(static_cast<int>(m), U.data() + i * m, 1, v0.data(), 1);
        std::vector<double> rp(v0);
        for (lint i = 0; i < m; ++i) cblas_daxpy(static_cast<int>(m), -e[i], U.data() + i * m, 1, rp.data(), 1);
        for (lint t = 0; t < m; ++t) rss_perp += rp[t] * rp[t];
    }
    const lint ldvt = m >= pc ? pc : m;
    // singular values at rounding level are zero (the minimum-norm solution at alpha = 0)
    const double cut = r > 0 ? double(std::max(m, pc)) * 2 * kU * sv[0] : 0.0;
    const double nobs = double(m), c0 = op.intercept ? 1.0 : 0.0;
    std::vector<double> b(static_cast<std::size_t>(pc));
    for (lint l = 0; l < L; ++l) {
        const double alpha = rd(reinterpret_cast<const std::byte*>(alphas) + l * as);
        std::fill(b.begin(), b.end(), 0.0);
        double rss = rss_perp, df = 0.0;
        for (lint i = 0; i < r; ++i) {
            const double s_ = sv[i] > cut ? sv[i] : 0.0;
            const double den = s_ * s_ + alpha;
            const double fi = s_ > 0.0 ? s_ / den : 0.0;
            const double shrink = s_ > 0.0 ? alpha / den : 1.0;
            rss += shrink * shrink * e[i] * e[i];
            df += s_ > 0.0 ? s_ * s_ / den : 0.0;
            const double coef = fi * e[i];
            if (coef != 0.0)
                for (lint j = 0; j < pc; ++j) b[j] += coef * Vt[j * ldvt + i];
        }
        // the solution's condition (s_1^2 + alpha)/(s_min^2 + alpha): refined with double-double
        // residuals from the data (G51) above kRefineCond pc (the SVD solution's error is about that times u)
        double smin = 0.0;
        for (lint i = 0; i < r; ++i)
            if (sv[i] > cut) smin = sv[i];
        if (r < pc) smin = 0.0;
        const double kap = r > 0 ? (sv[0] * sv[0] + alpha) / (smin * smin + alpha) : 1.0;
        double g0 = 0.0;
        bool refined = false;
        if (pc > 0 && kap > kRefineCond * double(pc) && op.polish) {
            refined = true;
            std::vector<double> sums, g(static_cast<std::size_t>(pc)), t(static_cast<std::size_t>(r));
            double prev = INFINITY;
            for (int step = 0; step < kPolishSteps; ++step) {
                residual_pass_dd(src, P, P.cols, b.data(), g0, sums);
                for (lint j = 0; j < pc; ++j) g[j] = sums[1 + j] / P.sw - alpha * b[j];
                // (Z^T Omega Z + alpha I)^-1 g: V diag(1/(s^2 + alpha)) V^T g, and g's part
                // outside V's span over alpha
                std::fill(t.begin(), t.end(), 0.0);
                for (lint i = 0; i < r; ++i)
                    for (lint j = 0; j < pc; ++j) t[i] += Vt[j * ldvt + i] * g[j];
                std::vector<double> dlt(static_cast<std::size_t>(pc), 0.0), outside(g);
                for (lint i = 0; i < r; ++i) {
                    const double s_ = sv[i] > cut ? sv[i] : 0.0;
                    const double den = s_ * s_ + alpha;
                    const double f_ = den > 0.0 ? t[i] / den : 0.0;
                    for (lint j = 0; j < pc; ++j) {
                        dlt[j] += f_ * Vt[j * ldvt + i];
                        outside[j] -= t[i] * Vt[j * ldvt + i];
                    }
                }
                double dmax = 0.0, bmax = 0.0;
                for (lint j = 0; j < pc; ++j) {
                    if (alpha > 0.0 && r < pc) dlt[j] += outside[j] / alpha;
                    b[j] += dlt[j];
                    dmax = std::max(dmax, std::fabs(dlt[j]));
                    bmax = std::max(bmax, std::fabs(b[j]));
                }
                if (op.intercept) g0 += sums[0] / P.sw;
                if (dmax <= 4 * kU * bmax || dmax >= 0.5 * prev) break;
                prev = dmax;
            }
        }
        double* st = stats + l * pNStat;
        std::fill(st, st + pNStat, kNaN);
        st[pNobs] = nobs;
        st[pSumW] = P.sw;
        st[pNIter] = 0.0;
        st[pConv] = 1.0;
        st[pIsNaN] = 0.0;
        st[pRss] = rss * P.sw;
        st[pTss] = vv * P.sw;
        st[pDf] = df;
        st[pPolished] = refined ? 2.0 : 0.0;
        const double den = 1.0 - (df + c0) / nobs;
        st[pGcv] = rss / (den * den);
        write_params(P, p, op.intercept, b, g0, params + l * k);
    }
    test_errors(src, op.intercept, L, params, stats);
    return 0;
}

// ---- the penalized GLM (binomial logit, poisson log)

// (1 + r) log1p(r) - r = x log x - (x - 1), x = 1 + r = y / mu, without cancellation (the
// series of regression/glm.cpp's deviance).
inline double mu_fn(double r, double x) {
    if (r >= -0.5 && r <= 1.0) {
        const double s = r / (2.0 + r), s2 = s * s;
        double t = 0.0;
        for (int j = 17; j >= 0; --j) t = 1.0 / (2 * j + 3) + s2 * t;
        return r * s - 2.0 * s * s2 * t;
    }
    return r > 1.0 ? r - std::log1p(r) : r - std::log(x);
}
inline double phi_fn(double r, double x) {
    if (r >= -0.5 && r <= 1.0) return r * r - (1.0 + r) * mu_fn(r, x);
    return x * std::log(x) - r;
}
// log(1 + e^z) without overflow
inline double softplus(double z) { return z > 0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z)); }

// mu and the variance (= mu' for the canonical links) at eta
inline void link(int fam, double eta, double& mu, double& var) {
    if (fam == fBinomial) {
        if (eta >= 0) {
            const double e = std::exp(-eta);
            mu = 1.0 / (1.0 + e);
            var = mu * (e / (1.0 + e));
        } else {
            const double e = std::exp(eta);
            mu = e / (1.0 + e);
            var = mu * (1.0 / (1.0 + e));
        }
    } else {
        mu = std::exp(eta);
        var = mu;
    }
}
// the unit deviance at eta
inline double unit_dev(int fam, double y, double eta, double mu) {
    if (fam == fBinomial) return 2.0 * softplus(y == 1.0 ? -eta : eta);
    if (y == 0.0) return 2.0 * mu;
    return 2.0 * mu * phi_fn((y - mu) / mu, y / mu);
}

struct GlmState {
    std::vector<double> eta, mu, var;  // per fit row
    double dev = 0;
    bool valid = true;
};

// eta (Dot2 over the fit's columns), mu, var and the deviance (double-double over fixed chunks)
void glm_eval(const Src& src, const Prep& P, int fam, const double* beta, double b0, GlmState& g) {
    const lint m = P.m;
    g.eta.resize(static_cast<std::size_t>(m));
    g.mu.resize(static_cast<std::size_t>(m));
    g.var.resize(static_cast<std::size_t>(m));
    std::vector<lint> nz;
    for (const lint j : P.cols)
        if (beta[j] != 0.0) nz.push_back(j);
    const lint nch = std::max<lint>(1, (m + kRowChunk - 1) / kRowChunk);
    std::vector<double> parts(static_cast<std::size_t>(2 * nch));
    std::vector<std::uint8_t> bad(static_cast<std::size_t>(nch), 0);
    for_chunks(nch, double(m) * double(nz.size() + 4), [&](lint c) {
        DD acc{0.0, 0.0};
        for (lint t = c * kRowChunk; t < std::min(m, (c + 1) * kRowChunk); ++t) {
            const lint i = P.idx[t];
            double S = b0, E = 0.0;
            for (const lint j : nz) {
                double hp, ep, q;
                two_prod(src.x(i, j), beta[j], hp, ep);
                two_sum(S, hp, S, q);
                E += q + ep;
            }
            const double eta = S + E;
            double mu, var;
            link(fam, eta, mu, var);
            g.eta[t] = eta, g.mu[t] = mu, g.var[t] = var;
            const double dv = unit_dev(fam, src.yv(i), eta, mu);
            if (!std::isfinite(dv) || !std::isfinite(eta)) bad[c] = 1;
            acc = acc + DD{P.wt[t] * dv, 0.0};
        }
        parts[2 * c] = acc.hi, parts[2 * c + 1] = acc.lo;
    });
    DD dev{0.0, 0.0};
    bool valid = true;
    for (lint c = 0; c < nch; ++c) {
        dev = dev + DD{parts[2 * c], parts[2 * c + 1]};
        if (bad[c]) valid = false;
    }
    g.dev = dev.hi;
    g.valid = valid && std::isfinite(dev.hi);
}

double penalty(const Prep& P, const double* beta, double l1, double l2) {
    double a = 0.0, q = 0.0;
    for (const lint j : P.cols) {
        const double b = beta[j] * P.sc[j];
        a += std::fabs(b), q += b * b;
    }
    return l1 * a + 0.5 * l2 * q;
}

// The working weights' (W = w var) means of the fit's columns and of z = eta + (y - mu)/var;
// returns sum W.
double working_means(const Src& src, const Prep& P, const GlmState& g, bool intercept, std::vector<double>& cw,
                     double& zbar) {
    const lint pc = static_cast<lint>(P.cols.size());
    std::vector<double> sums(static_cast<std::size_t>(pc + 2));
    chunk_sum(P.m, pc + 2, sums.data(), double(pc), [&](lint t0, lint t1, double* part) {
        for (lint t = t0; t < t1; ++t) {
            const lint i = P.idx[t];
            const double W = P.wt[t] * g.var[t];
            if (!(W > 0.0)) continue;
            for (lint a = 0; a < pc; ++a) part[a] += W * src.x(i, P.cols[a]);
            part[pc] += W * (g.eta[t] + (src.yv(i) - g.mu[t]) / g.var[t]);
            part[pc + 1] += W;
        }
    });
    const double sW = sums[pc + 1];
    cw.assign(static_cast<std::size_t>(src.p), 0.0);
    zbar = 0.0;
    if (intercept && sW > 0.0) {
        for (lint a = 0; a < pc; ++a) cw[P.cols[a]] = sums[a] / sW;
        zbar = sums[pc] / sW;
    }
    return sW;
}

// Fills the solver with the IRLS quadratic model at g: Z = sqrt(W/sW) (x - cw)/s, v = z - zbar.
void glm_subproblem(const Src& src, const Prep& P, const GlmState& g, const std::vector<double>& cw, double zbar,
                    double sW, Solver& s) {
    const lint m = P.m, pc = static_cast<lint>(P.cols.size());
    std::vector<double> f(static_cast<std::size_t>(m)), z(static_cast<std::size_t>(src.n), 0.0);
    for (lint t = 0; t < m; ++t) {
        const double W = P.wt[t] * g.var[t];
        f[t] = W > 0.0 ? W / sW : 0.0;
        z[P.idx[t]] = W > 0.0 ? g.eta[t] + (src.yv(P.idx[t]) - g.mu[t]) / g.var[t] : 0.0;
    }
    Prep Q = P;
    Q.cx = cw;
    s.pc = pc;
    s.cov = pc <= kGlmCovMaxCols;
    if (s.cov) {
        const lint q = pc + 1;
        std::vector<double> Gq(static_cast<std::size_t>(q * q));
        gram(m, q, Gq.data(), Packer{src, Q, f.data(), z.data(), zbar, true, false, true});
        s.G.assign(static_cast<std::size_t>(pc * pc), 0.0);
        s.c.assign(static_cast<std::size_t>(pc), 0.0);
        for (lint a = 0; a < pc; ++a) {
            for (lint b = 0; b < pc; ++b) s.G[b * pc + a] = Gq[b * q + a];
            s.c[a] = Gq[pc * q + a];
        }
        s.vv = Gq[pc * q + pc];
    } else {
        s.m = m;
        s.Z.assign(static_cast<std::size_t>(m * pc), 0.0);
        s.rv.assign(static_cast<std::size_t>(m), 0.0);
        const Packer pk{src, Q, f.data(), z.data(), zbar, true, false};
        const lint nch = std::max<lint>(1, (m + kRowChunk - 1) / kRowChunk);
        for_chunks(nch, double(m) * pc, [&](lint c) {
            const lint t0 = c * kRowChunk, t1 = std::min(m, t0 + kRowChunk), len = t1 - t0;
            std::vector<double> buf(static_cast<std::size_t>(len * (pc + 1)));
            pk(t0, t1, buf.data());
            for (lint a = 0; a < pc; ++a) std::copy_n(buf.data() + a * len, len, s.Z.data() + a * m + t0);
            std::copy_n(buf.data() + pc * len, len, s.rv.data() + t0);
        });
        double vv = 0.0;
        for (lint t = 0; t < m; ++t) vv += s.rv[t] * s.rv[t];
        s.vv = vv;
    }
    s.init_state();
}

// The working-weight Gram at g of B = [sqrt(W) xhat_A (na columns), sqrt(W), sqrt(W) eta, u]:
// xhat = (x - P.cx)/P.sc (centered by the prior weights' means), W = w var (var = mu' for the
// canonical links), u = sqrt(w/var)(y - mu) (0 where var is 0). Every sum an IRLS step and a
// Newton step need comes from it in one pass: the W-centered Gram of the columns
// G_ab - S_a S_b / sW (S_a = sum W xhat_a), the gradient sum w (xhat - chat)(y - mu) =
// G_au - S_a G_1u / sW, sum w (y - mu) = G_1u, and the working response z = eta + (y - mu)/var
// through sqrt(W) z = sqrt(W) eta + u.
struct WG {
    lint na = 0, q = 0;
    std::vector<double> G;
    double at(lint i, lint j) const { return G[static_cast<std::size_t>(j * q + i)]; }
    lint one() const { return na; }
    lint eta() const { return na + 1; }
    lint u() const { return na + 2; }
    double sW() const { return at(one(), one()); }
    double S(lint a) const { return at(a, one()); }
    double Hc(lint a, lint b) const { return at(a, b) - S(a) * S(b) / sW(); }
    double gc(lint a) const { return at(a, u()) - S(a) * at(one(), u()) / sW(); }
    double g0() const { return at(one(), u()); }
    double zbar() const { return (at(one(), eta()) + g0()) / sW(); }
};

void wgram(const Src& src, const Prep& P, const GlmState& g, const std::vector<lint>& cols, WG& out) {
    const lint na = static_cast<lint>(cols.size()), q = na + 3;
    out.na = na;
    out.q = q;
    out.G.assign(static_cast<std::size_t>(q * q), 0.0);
    std::vector<double> cc(static_cast<std::size_t>(na)), is(static_cast<std::size_t>(na));
    std::vector<std::int64_t> off(static_cast<std::size_t>(na));
    for (lint a = 0; a < na; ++a) cc[a] = P.cx[cols[a]], is[a] = 1.0 / P.sc[cols[a]], off[a] = cols[a] * src.xcs;
    gram(P.m, q, out.G.data(), [&](lint t0, lint t1, double* buf) {
        for (lint t = t0; t < t1; ++t) {
            const lint i = P.idx[t];
            const double var = g.var[t];
            const double W = P.wt[t] * var;
            const double r = W > 0.0 ? std::sqrt(W) : 0.0;
            const std::byte* row = src.X + i * src.xrs;
            double* o = buf + (t - t0) * q;
            for (lint a = 0; a < na; ++a) o[a] = r * ((rd(row + off[a]) - cc[a]) * is[a]);
            o[na] = r;
            o[na + 1] = r * g.eta[t];
            o[na + 2] = r > 0.0 ? std::sqrt(P.wt[t] / var) * (src.yv(i) - g.mu[t]) : 0.0;
        }
    });
}

int glm_one(const Opts& op, const Src& src, const double* alphas, std::int64_t as, lint L, double* params,
            double* stats) {
    const lint p = src.p, k = p + (op.intercept ? 1 : 0);
    const int fam = op.family;
    const int chk = check(src, fam);
    if (chk > 0) return chk;
    if (chk < 0) {
        fill_nan(L, k, params, stats, kNaN);
        return 0;
    }
    Prep P;
    prep_rows(src, op.intercept, P);
    const lint m = P.m;
    if (m == 0) {
        fill_nan(L, k, params, stats, 0.0);
        return 0;
    }
    // scales (population sd with the prior weights, glmnet's) and the fit's columns
    {
        const lint pc0 = static_cast<lint>(P.cols.size());
        std::vector<double> sums(static_cast<std::size_t>(2 * pc0));
        chunk_sum(m, 2 * pc0, sums.data(), double(pc0), [&](lint t0, lint t1, double* part) {
            for (lint t = t0; t < t1; ++t) {
                const lint i = P.idx[t];
                for (lint a = 0; a < pc0; ++a) {
                    const double u = src.x(i, P.cols[a]) - P.cx[P.cols[a]];
                    part[a] += P.wt[t] * u;
                    part[pc0 + a] += P.wt[t] * u * u;
                }
            }
        });
        std::vector<lint> cols;
        for (lint a = 0; a < pc0; ++a) {
            const lint j = P.cols[a];
            const double s1 = op.intercept ? sums[a] : 0.0;
            const double var = (sums[pc0 + a] - s1 * s1 / P.sw) / P.sw;
            if (op.intercept) P.cx[j] += s1 / P.sw;
            if (!(var > 0.0)) continue;
            cols.push_back(j);
            P.sc[j] = op.standardize ? std::sqrt(var) : 1.0;
        }
        P.cols = cols;
    }
    const lint pc = static_cast<lint>(P.cols.size());
    // the null model: the weighted mean of y (intercept) or eta = 0
    DD sy{0.0, 0.0};
    for (lint t = 0; t < m; ++t) sy = sy + DD{P.wt[t] * src.yv(P.idx[t]), 0.0};
    const double ybar = sy.hi / P.sw;
    double eta0 = 0.0;
    if (op.intercept) {
        if (!(ybar > 0.0) || (fam == fBinomial && !(ybar < 1.0))) {
            // all y at the boundary: no finite fit
            fill_nan(L, k, params, stats, double(m));
            for (lint l = 0; l < L; ++l) stats[l * pNStat + pIsNaN] = 0.0, stats[l * pNStat + pConv] = 0.0;
            return 0;
        }
        eta0 = fam == fBinomial ? std::log(ybar) - std::log1p(-ybar) : std::log(ybar);
    }
    std::vector<double> beta(static_cast<std::size_t>(p), 0.0), beta_old;
    double b0 = eta0;
    GlmState g, gn;
    glm_eval(src, P, fam, beta.data(), b0, g);
    const double null_dev = g.dev;
    Solver s;
    WG wg;
    std::vector<double> cw(static_cast<std::size_t>(p), 0.0);
    // the W-means of the columns (cw) from a working Gram over all the fit's columns
    auto means_from = [&](const WG& w, const std::vector<lint>& cols) {
        std::fill(cw.begin(), cw.end(), 0.0);
        if (!op.intercept) return;
        for (lint a = 0; a < w.na; ++a) cw[cols[a]] = P.cx[cols[a]] + P.sc[cols[a]] * (w.S(a) / w.sW());
    };
    for (const lint l : path_order(alphas, as, L)) {
        const double alpha = rd(reinterpret_cast<const std::byte*>(alphas) + l * as);
        const double l1 = alpha * op.l1_ratio, l2 = alpha * (1.0 - op.l1_ratio);
        double* st = stats + l * pNStat;
        std::fill(st, st + pNStat, kNaN);
        st[pNobs] = double(m);
        st[pSumW] = P.sw;
        st[pIsNaN] = 0.0;
        st[pTss] = null_dev;
        glm_eval(src, P, fam, beta.data(), b0, g);
        auto objective = [&](const GlmState& gs, const std::vector<double>& b) {
            return gs.dev / (2.0 * P.sw) + penalty(P, b.data(), l1, l2);
        };
        // IRLS from (beta, b0): at most `limit` iterations, the inner coordinate descent to the
        // relative gap `inner`; returns whether the objective settled (used: the iterations)
        auto irls = [&](double inner, int limit, int& used) -> bool {
            double F = objective(g, beta);
            bool conv = false;
            int it = 0;
            while (it < limit) {
                ++it;
                double sW, zbar;
                if (pc <= kGlmCovMaxCols) {
                    wgram(src, P, g, P.cols, wg);
                    sW = wg.sW();
                    if (!(sW > 0.0)) break;
                    means_from(wg, P.cols);
                    zbar = op.intercept ? wg.zbar() : 0.0;
                    // the quadratic model in the W-centered (no intercept: uncentered) columns
                    s.pc = pc;
                    s.cov = true;
                    s.G.assign(static_cast<std::size_t>(pc * pc), 0.0);
                    s.c.assign(static_cast<std::size_t>(pc), 0.0);
                    const double ze = wg.at(wg.eta(), wg.eta()) + 2.0 * wg.at(wg.eta(), wg.u()) + wg.at(wg.u(), wg.u());
                    for (lint u_ = 0; u_ < pc; ++u_) {
                        for (lint v_ = 0; v_ < pc; ++v_)
                            s.G[v_ * pc + u_] = (op.intercept ? wg.Hc(u_, v_) : wg.at(u_, v_)) / sW;
                        const double cz = wg.at(u_, wg.eta()) + wg.at(u_, wg.u());
                        s.c[u_] = (op.intercept ? cz - wg.S(u_) * zbar : cz) / sW;
                    }
                    s.vv = std::max(0.0, op.intercept ? ze / sW - zbar * zbar : ze / sW);
                    s.init_state();
                } else {
                    sW = working_means(src, P, g, op.intercept, cw, zbar);
                    if (!(sW > 0.0)) break;
                    glm_subproblem(src, P, g, cw, zbar, sW, s);
                }
                const double scl = P.sw / sW;
                for (lint a_ = 0; a_ < pc; ++a_) s.b[a_] = beta[P.cols[a_]] * P.sc[P.cols[a_]];
                s.refresh();
                cd_solve(s, l1 * scl, l2 * scl, kInnerMaxSweeps, inner * s.vv, inner);
                beta_old = beta;
                const double b0_old = b0;
                std::fill(beta.begin(), beta.end(), 0.0);
                double nb0 = zbar;
                for (lint a_ = 0; a_ < pc; ++a_) {
                    const lint j = P.cols[a_];
                    beta[j] = s.b[a_] / P.sc[j];
                    nb0 -= cw[j] * beta[j];
                }
                b0 = op.intercept ? nb0 : 0.0;
                glm_eval(src, P, fam, beta.data(), b0, gn);
                double Fn = objective(gn, beta);
                // step halving on a rise of the objective (or an invalid one)
                for (int h = 0; h < 64 && (!gn.valid || (Fn - F) / (std::fabs(Fn) + 0.1) >= op.tol); ++h) {
                    for (lint j = 0; j < p; ++j) beta[j] = 0.5 * (beta[j] + beta_old[j]);
                    b0 = 0.5 * (b0 + b0_old);
                    glm_eval(src, P, fam, beta.data(), b0, gn);
                    Fn = objective(gn, beta);
                }
                if (!gn.valid) {  // no valid point along the step: the last iterate stands
                    beta = beta_old;
                    b0 = b0_old;
                    break;
                }
                const bool settled = std::fabs(Fn - F) / (std::fabs(Fn) + 0.1) < op.tol;
                std::swap(g, gn);
                F = Fn;
                if (settled) {
                    conv = true;
                    break;
                }
            }
            used = it;
            return conv;
        };
        // the polish: Newton's method on the active set with the signs fixed (every column
        // without an L1 term); true when it settled and the KKT conditions hold
        int newton_steps = 0;
        auto try_polish = [&]() -> bool {
            std::vector<lint> A, Acols;
            std::vector<double> sg;
            for (lint a_ = 0; a_ < pc; ++a_)
                if (beta[P.cols[a_]] != 0.0 || l1 == 0.0)
                    A.push_back(a_), Acols.push_back(P.cols[a_]), sg.push_back(beta[P.cols[a_]] > 0 ? 1.0 : -1.0);
            const lint na = static_cast<lint>(A.size());
            std::vector<double> nbeta = beta, ob, H, gA(static_cast<std::size_t>(na));
            double nb0 = b0, prev = INFINITY;
            GlmState gp = g;
            WG w;
            bool settled = false;
            double Fp = objective(gp, nbeta);
            newton_steps = 0;
            for (int step = 0; step < kNewtonSteps; ++step) {
                ++newton_steps;
                wgram(src, P, gp, Acols, w);
                const double sW = w.sW();
                if (!(sW > 0.0)) return false;
                H.assign(static_cast<std::size_t>(na * na), 0.0);
                for (lint q = 0; q < na; ++q) {
                    const lint j = Acols[q];
                    gA[q] = (op.intercept ? w.gc(q) : w.at(q, w.u())) / P.sw - l2 * nbeta[j] * P.sc[j] - l1 * sg[q];
                    for (lint r_ = 0; r_ < na; ++r_) H[r_ * na + q] = (op.intercept ? w.Hc(q, r_) : w.at(q, r_)) / P.sw;
                    H[q * na + q] += l2;
                }
                if (na > 0) {
                    if (linalg::lapack::potrf('L', na, H.data(), na) != 0) return false;
                    linalg::lapack::potrs('L', na, 1, H.data(), na, gA.data(), na);
                }
                // the step in (gamma', b), gamma' = b0 + sum cw beta (the intercept of the
                // W-centered columns, whose Hessian block is diagonal)
                std::vector<double> cwA(static_cast<std::size_t>(na), 0.0);
                if (op.intercept)
                    for (lint q = 0; q < na; ++q) cwA[q] = P.cx[Acols[q]] + P.sc[Acols[q]] * (w.S(q) / sW);
                const double dg = op.intercept ? w.g0() / sW : 0.0;
                double gam = nb0;
                for (lint q = 0; q < na; ++q) gam += cwA[q] * nbeta[Acols[q]];
                ob = nbeta;
                const double ob0 = nb0;
                double dmax = 0.0, bmax = 0.0;
                for (lint q = 0; q < na; ++q) {
                    const lint j = Acols[q];
                    const double bj = nbeta[j] * P.sc[j] + gA[q];
                    nbeta[j] = bj / P.sc[j];
                    dmax = std::max(dmax, std::fabs(gA[q]));
                    bmax = std::max(bmax, std::fabs(bj));
                }
                gam += dg;
                nb0 = gam;
                for (lint q = 0; q < na; ++q) nb0 -= cwA[q] * nbeta[Acols[q]];
                if (!op.intercept) nb0 = 0.0;
                glm_eval(src, P, fam, nbeta.data(), nb0, gp);
                double Fn = objective(gp, nbeta);
                // step halving on an invalid point or a rise of the objective (far from the
                // minimum; the IRLS's rule)
                for (int h = 0; h < 64 && (!gp.valid || (Fn - Fp) / (std::fabs(Fn) + 0.1) >= op.tol); ++h) {
                    for (const lint j : Acols) nbeta[j] = 0.5 * (nbeta[j] + ob[j]);
                    nb0 = 0.5 * (nb0 + ob0);
                    glm_eval(src, P, fam, nbeta.data(), nb0, gp);
                    Fn = objective(gp, nbeta);
                }
                if (!gp.valid) return false;
                Fp = Fn;
                const double step_size = std::max(dmax, std::fabs(dg));
                const double scale = std::max(bmax, std::fabs(gam));
                if (step_size <= 4 * kU * scale) {
                    settled = true;
                    break;
                }
                // the steps stopped shrinking: at the rounding floor when they are that small
                if (step >= 2 && step_size >= 0.5 * prev) {
                    settled = step_size <= 0x1p-26 * scale;
                    break;
                }
                prev = step_size;
            }
            if (!settled) return false;
            for (lint q = 0; q < na; ++q)
                if (l1 > 0.0 && !(nbeta[Acols[q]] * sg[q] > 0.0)) return false;
            if (na < pc) {
                // the inactive columns' KKT conditions at the polished point
                wgram(src, P, gp, P.cols, w);
                std::vector<std::uint8_t> in(static_cast<std::size_t>(pc), 0);
                for (const lint a_ : A) in[a_] = 1;
                for (lint a_ = 0; a_ < pc; ++a_) {
                    if (in[a_]) continue;
                    const double gj = (op.intercept ? w.gc(a_) : w.at(a_, w.u())) / P.sw;
                    const double dj = std::max(0.0, op.intercept ? w.Hc(a_, a_) : w.at(a_, a_)) / P.sw;
                    if (!(std::fabs(gj) <= l1 * (1.0 + kKktRel) + kKktAbs * std::sqrt(dj * gp.dev / P.sw))) return false;
                }
            }
            beta = nbeta;
            b0 = nb0;
            std::swap(g, gp);
            return true;
        };
        int it = 0;
        bool conv = false, polished = false;
        // without an L1 term the objective is smooth: Newton's method from the start (IRLS with
        // the exact inner solve), the coordinate descent only when it fails
        if (l1 == 0.0 && op.polish) {
            polished = conv = try_polish();
            it = newton_steps;
        }
        if (!polished) {
            int used = 0;
            conv = irls(kInnerTol, op.max_iter - it, used);
            it += used;
        }
        if (conv && op.polish && !polished) {
            polished = try_polish();
            // a failed polish: the iterate's active set is not the solution's yet; IRLS again
            // with the inner coordinate descent to a smaller gap
            double t = kInnerTol;
            for (int retry = 0; retry < kPolishRetries && !polished && it < op.max_iter; ++retry) {
                t *= kRetryTol;
                int more = 0;
                const bool c2 = irls(t, op.max_iter - it, more);
                it += more;
                if (!c2) {
                    conv = false;
                    break;
                }
                polished = try_polish();
            }
        }
        st[pNIter] = it;
        st[pConv] = conv ? 1.0 : 0.0;
        st[pPolished] = polished ? 1.0 : 0.0;
        // the final values: deviance, the quadratic model's duality gap, the effective df
        st[pRss] = g.dev;
        wgram(src, P, g, P.cols, wg);
        const double sW = wg.sW();
        if (sW > 0.0) {
            const double scl = P.sw / sW;
            means_from(wg, P.cols);
            const double zbar = op.intercept ? wg.zbar() : 0.0;
            s.pc = pc;
            s.d.assign(static_cast<std::size_t>(pc), 1.0);
            s.b.assign(static_cast<std::size_t>(pc), 0.0);
            s.xta.assign(static_cast<std::size_t>(pc), 0.0);
            std::vector<lint> A;
            for (lint a_ = 0; a_ < pc; ++a_) {
                s.b[a_] = beta[P.cols[a_]] * P.sc[P.cols[a_]];
                s.xta[a_] = (op.intercept ? wg.gc(a_) : wg.at(a_, wg.u())) / sW;
                if (s.b[a_] != 0.0) A.push_back(a_);
            }
            double gam = b0 - zbar;
            for (const lint j : P.cols) gam += cw[j] * beta[j];
            double D = (op.intercept ? gam : 0.0) * wg.g0() / sW;
            for (lint a_ = 0; a_ < pc; ++a_) D += s.b[a_] * s.xta[a_];
            st[pGap] = s.gap_from(l1 * scl, l2 * scl, wg.at(wg.u(), wg.u()) / sW, D) / scl;
            const lint na = static_cast<lint>(A.size());
            if (l2 == 0.0 || na == 0) {
                st[pDf] = double(na);
            } else {
                std::vector<double> H(static_cast<std::size_t>(na * na));
                for (lint q = 0; q < na; ++q) {
                    for (lint r_ = 0; r_ < na; ++r_)
                        H[r_ * na + q] = (op.intercept ? wg.Hc(A[q], A[r_]) : wg.at(A[q], A[r_])) / P.sw;
                    H[q * na + q] += l2;
                }
                st[pDf] = effective_df(H, na, l2);
            }
        }
        double* pr = params + l * k;
        if (op.intercept) pr[0] = b0;
        std::copy(beta.begin(), beta.end(), pr + (op.intercept ? 1 : 0));
    }
    test_errors(src, op.intercept, L, params, stats, fam);
    return 0;
}

// ---- alpha_max: the smallest alpha whose solution is 0

int alpha_max_one(const Opts& op, const Src& src, double* out) {
    const int chk = check(src, op.family);
    if (chk > 0) return chk;
    if (chk < 0) {
        *out = kNaN;
        return 0;
    }
    Prep P;
    prep_rows(src, op.intercept, P);
    if (P.m == 0) {
        *out = kNaN;
        return 0;
    }
    const lint pc0 = static_cast<lint>(P.cols.size());
    // without the intercept the GLM's null model is eta = 0: mu0 = 1/2 or 1
    const double mu0 = op.intercept || op.family == fGaussian ? 0.0 : (op.family == fBinomial ? 0.5 : 1.0);
    const lint len = 3 * pc0 + 2;
    std::vector<double> sums(static_cast<std::size_t>(len));
    chunk_sum(P.m, len, sums.data(), double(pc0), [&](lint t0, lint t1, double* part) {
        for (lint t = t0; t < t1; ++t) {
            const lint i = P.idx[t];
            const double w = P.wt[t];
            const double v = src.yv(i) - P.cy - mu0;
            for (lint a = 0; a < pc0; ++a) {
                const double u = src.x(i, P.cols[a]) - P.cx[P.cols[a]];
                part[a] += w * u;
                part[pc0 + a] += w * u * u;
                part[2 * pc0 + a] += w * u * v;
            }
            part[3 * pc0] += w * v;
            part[3 * pc0 + 1] += w;
        }
    });
    double best = 0.0;
    const double sw = P.sw;
    const double vbar = op.intercept ? sums[3 * pc0] / sw : 0.0;
    for (lint a = 0; a < pc0; ++a) {
        const double s1 = op.intercept ? sums[a] : 0.0;
        const double var = (sums[pc0 + a] - s1 * s1 / sw) / sw;
        if (!(var > 0.0)) continue;
        // sum w (x - xbar)(v - vbar) / sw, the scale under standardize
        const double cxv = (sums[2 * pc0 + a] - s1 * vbar) / sw;
        const double scl = op.standardize ? std::sqrt(var) : 1.0;
        best = std::max(best, std::fabs(cxv) / scl);
    }
    *out = best / op.l1_ratio;
    return 0;
}

// ---- bindings

[[noreturn]] void raise_failure(const Batch& b, const Failure& f, const char* func, int fam) {
    std::string what;
    switch (f.code) {
        case eInfY: what = "'y' contains infinity"; break;
        case eInfX: what = "'X' contains infinity"; break;
        case eInfW: what = "'weights' contains infinity"; break;
        case eNegW: what = "'weights' must be >= 0"; break;
        case eBadY:
            what = fam == fBinomial ? "'y' must be 0 or 1 for the binomial family" : "'y' must be >= 0 for the poisson family";
            break;
        default: what = "the SVD did not converge"; break;
    }
    throw nb::value_error(linalg::failure_message(b, f.index, (std::string(func) + "(): " + what).c_str()).c_str());
}

std::vector<std::int64_t> with(std::vector<std::int64_t> shape, std::initializer_list<std::int64_t> tail) {
    shape.insert(shape.end(), tail);
    return shape;
}

// kind: 0 elastic net (coordinate descent), 1 ridge (SVD), 2 penalized GLM.
// path(func, kind, y (..., n), X (..., n, p), aux (..., 2, n) | None [weights, test weights],
//      has_w, has_t, alphas (..., L), l1_ratio, family, intercept, standardize, max_iter, tol, polish)
// -> (params (..., L, k), stats (..., L, S)). The test weights (cross-validation) mark the
// rows whose error is summed into stats[pTestSse].
nb::object path(const std::string& func, int kind, nb::handle yo, nb::handle xo, nb::handle auxo, bool has_w,
                bool has_t, nb::handle ao, double l1_ratio, int family, bool intercept, bool standardize, int max_iter,
                double tol, bool polish, double refine) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X"), A(ao, fn, "alpha");
        const bool has_aux = !auxo.is_none();
        std::optional<Input> aux;
        if (has_aux) aux.emplace(auxo, fn, "weights");
        const linalg::BatchOperand none{};
        Batch b({{y.batch_shape(1), y.batch_strides(1)},
                 {X.batch_shape(2), X.batch_strides(2)},
                 has_aux ? linalg::BatchOperand{aux->batch_shape(2), aux->batch_strides(2)} : none,
                 {A.batch_shape(1), A.batch_strides(1)}},
                fn);
        const lint n = X.shape(-2), p = X.shape(-1), k = p + (intercept ? 1 : 0), L = A.shape(-1);
        Opts op{l1_ratio, intercept, standardize, max_iter, tol, polish, family, refine};
        Out po = linalg::alloc_c(with(b.shape(), {L, k}), DType::Float64);
        Out so = linalg::alloc_c(with(b.shape(), {L, pNStat}), DType::Float64);
        const double nk = double(n) * double(std::max<lint>(p, 1));
        const double element_ns = 5000.0 + nk * (2.0 * double(std::min<lint>(p, kCovMaxCols)) + 20.0 * double(L)) *
                                               (kind == 2 ? 10.0 : 1.0);
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            const std::byte* ab = has_aux ? aux->data() + off[2] : nullptr;
            const std::int64_t rs = has_aux ? aux->stride(-2) : 0, cs = has_aux ? aux->stride(-1) : 0;
            Src s{n, p, y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2), X.stride(-1),
                  has_w ? ab : nullptr, cs, has_t ? ab + rs : nullptr, cs};
            const double* al = reinterpret_cast<const double*>(A.data() + off[3]);
            const std::int64_t as = A.stride(-1);
            double* pr = po.as<double>() + i * L * k;
            double* st = so.as<double>() + i * L * pNStat;
            int r;
            if (kind == 0) r = enet_one(op, s, al, as, L, pr, st);
            else if (kind == 1) r = ridge_one(op, s, al, as, L, pr, st, sc);
            else r = glm_one(op, s, al, as, L, pr, st);
            return r < 0 ? 99 : r;
        });
        if (f) raise_failure(b, f, fn, family);
        return nb::make_tuple(linalg::to_array(po), linalg::to_array(so));
    });
}

// alpha_max(func, y, X, w | None, l1_ratio, family, intercept, standardize) -> (...)
nb::object alpha_max(const std::string& func, nb::handle yo, nb::handle xo, nb::handle wo, double l1_ratio, int family,
                     bool intercept, bool standardize) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X");
        const bool has_w = !wo.is_none();
        std::optional<Input> w;
        if (has_w) w.emplace(wo, fn, "weights");
        const linalg::BatchOperand none{};
        Batch b({{y.batch_shape(1), y.batch_strides(1)},
                 {X.batch_shape(2), X.batch_strides(2)},
                 has_w ? linalg::BatchOperand{w->batch_shape(1), w->batch_strides(1)} : none},
                fn);
        const lint n = X.shape(-2), p = X.shape(-1);
        Opts op{l1_ratio, intercept, standardize, 0, 0.0, false, family, 0.0};
        Out oo = linalg::alloc_c(b.shape(), DType::Float64);
        const Failure f = linalg::run_batch(b, 2000.0 + 4.0 * double(n) * double(std::max<lint>(p, 1)),
                                            [&](std::int64_t i, const std::int64_t* off, Scratch&) {
            Src s{n, p, y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2), X.stride(-1),
                  has_w ? w->data() + off[2] : nullptr, has_w ? w->stride(-1) : 0, nullptr, 0};
            return alpha_max_one(op, s, oo.as<double>() + i);
        });
        if (f) raise_failure(b, f, fn, family);
        return b.ndim() == 0 ? nb::cast(oo.as<double>()[0]) : linalg::to_array(oo);
    });
}

}  // namespace

void bind_penalized(nb::module_& m) {
    m.def("penalized_path", &path, nb::arg("func"), nb::arg("kind"), nb::arg("y").none(), nb::arg("X").none(),
          nb::arg("aux").none(), nb::arg("has_w"), nb::arg("has_t"), nb::arg("alphas").none(), nb::arg("l1_ratio"),
          nb::arg("family"),
          nb::arg("intercept"), nb::arg("standardize"), nb::arg("max_iter"), nb::arg("tol"), nb::arg("polish") = true,
          nb::arg("refine") = kRefineCond);
    m.def("alpha_max", &alpha_max, nb::arg("func"), nb::arg("y").none(), nb::arg("X").none(), nb::arg("w").none(),
          nb::arg("l1_ratio"), nb::arg("family"), nb::arg("intercept"), nb::arg("standardize"));
}

}  // namespace sd::regression
