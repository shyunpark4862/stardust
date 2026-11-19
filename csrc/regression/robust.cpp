// Robust regression's numerical core (docs/design/regression.md §3.6).
//
// One regression per batch element, through linalg's run_batch (one task per regression,
// the same bits for any thread count):
//   1. Checks (inf is an error, NaN gives an all-NaN result) and the least squares start:
//      the full core of the linear models (regression/lsq.hpp, QR with refinement), which
//      also decides the aliased columns (G9, G21) and gives (X^T X)^-1 for Huber's H1.
//   2. M (G53, G55): IRLS. Each iteration is one weighted least squares fit of y with the
//      weights w(r_i/s) = psi(r_i/s)/(r_i/s) of the last residuals, by the normal
//      equations (Cholesky, as the GLM's inner iterations, G40) or the QR when they cannot;
//      then the scale of the new residuals: MAD about 0 (statsmodels' mad(center=0), the
//      median of |r|/Phi^-1(3/4)) or Huber's proposal 2 (statsmodels' HuberScale(d=2.5),
//      its equation sum chi(r/s) = (n - rank) E chi(Z) solved exactly). Converged when
//      the objective sum rho(r/s) settles, |D - D_old|/(|D| + 0.1) < tol (the GLM's form);
//      then the iterations go on by the QR with refinement while the coefficients still
//      move by more than rounding, and the last iteration's least squares is always the
//      refined QR's.
//   3. MM (G53): the fast-S estimator of Salibian-Barrera and Yohai (2006) as robustbase's
//      lmrob (its defaults: 500 elemental subsets, one local improvement step each, the two
//      best refined to convergence, bisquare chi with c = 1.54764 and b = 1/2, the scale
//      equation sum chi(r/s) = (n - p) b), then the M step: bisquare psi (c = 4.685061
//      by default) by the IRLS of 2 with the S scale fixed. The subsets come from
//      Philox4x64-10 keyed by two words the Python layer draws from the user's Generator;
//      candidate j draws from the elements j * 2^32 + t of that stream, so every candidate
//      is computed on its own (in parallel for one large regression) and the result does
//      not depend on the thread count. Beyond 2000 rows (G58), robustbase's large-n fast-S:
//      the search in five groups of 400 drawn rows, one step on the merged rows, then all
//      the rows (fast_s below).
//   4. Covariances (G57): M, Huber's H1 (statsmodels' default cov='H1'):
//      K^2 [sum psi^2/(n - rank)] s^2 / mean(psi')^2 (X^T X)^-1, K = 1 + rank/n var(psi')/
//      mean(psi')^2. MM, robustbase's .vcov.avar1 (Croux, Dhaene and Hoorelbeke 2003; the
//      default of lmrob.control for MM), negative eigenvalues fixed as its "posdefify".
//
// Row sums are cut into chunks of rows fixed by n alone and added in order; medians are
// selections. Nothing depends on the thread count.
#include "regression/robust.hpp"

#include "regression/lsq.hpp"
#include "regression/dot2.hpp"

#include "bind/errors.hpp"
#include "linalg/ops/common.hpp"
#include "parallel/pool.hpp"
#include "random/philox.hpp"
#include "special/dd.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd::regression {

namespace nb = nanobind;
using namespace lsq;
using linalg::Batch;
using linalg::Failure;
using linalg::Input;
using linalg::Out;
using DD = special::dd::DD;
using special::dd::operator+;

void bind_slopes(nb::module_& m);  // regression/slopes.cpp

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 0x1p-52;

// The Python layer maps the names (stardust/regression/_robust.py).
enum Norm : int { nHuber = 0, nBisquare, nHampel };
enum Method : int { mM = 0, mMM };
enum ScaleEst : int { sMad = 0, sHuber, sFixed };
// internal: the S estimator's M-scale (MM's S step, G82)
constexpr int sMScale = 3;

// The per-regression scalars, in this order (stardust/regression/_robust.py reads them).
// rNGroups and rLags are NaN (the shared result code asks for them).
enum RStat : int { rNobs = 0, rRank, rScale, rNIter, rConverged, rIsNaN, rNGroups, rLags, rNStat };

constexpr int eInfY = 1, eInfX = 2;

// statsmodels' Gaussian.ppf(3/4) (scipy's norm.ppf(0.75)): mad() divides by it
constexpr double kMadC = 0.6744897501960817;
// statsmodels' HuberScale() default d (Huber's proposal 2)
constexpr double kHuberD = 2.5;
// robustbase 0.99.7 lmrob.control(): tuning.chi (bisquare, 50% breakdown with bb = 1/2),
// best.r.s, k.max, refine.tol, solve.tol, mts; and its MAD's constant
constexpr double kSChi = 1.54764;
constexpr double kSB = 0.5;
constexpr int kBestR = 2;
constexpr int kKMax = 200;
constexpr double kRefineTol = 1e-7;
constexpr double kSolveTol = 1e-7;
constexpr int kMts = 1000;
constexpr double kRbMad = 1.4826;
constexpr int kRefineSteps = 3;  // the refined QR solves (G15)

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

// Row passes: chunks of rows fixed by n alone, partial sums added in order (the same bits
// for any thread count), on the pool from kRowParallelMin rows (serially inside a task).
constexpr lint kRowChunk = 4096;
constexpr lint kRowParallelMin = 1 << 16;

inline lint row_chunks(lint n) { return std::max<lint>(1, (n + kRowChunk - 1) / kRowChunk); }

template <class F>
void for_row_chunks(lint n, F&& body) {
    const lint nc = (n + kRowChunk - 1) / kRowChunk;
    auto one = [&](lint c) { body(c, c * kRowChunk, std::min(n, (c + 1) * kRowChunk)); };
    if (nc >= 2 && n >= kRowParallelMin) {
        parallel::parallel_for(nc, [&](std::int64_t c) { one(static_cast<lint>(c)); });
    } else {
        for (lint c = 0; c < nc; ++c) one(c);
    }
}

// A sum over rows in double-double: f(i) summed per chunk, the chunks added in order.
template <class F>
DD row_sum(lint n, F&& f) {
    const lint nc = row_chunks(n);
    std::vector<DD> part(static_cast<std::size_t>(nc), DD{0.0, 0.0});
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        DD a{0.0, 0.0};
        for (lint i = i0; i < i1; ++i) a = a + DD{f(i), 0.0};
        part[static_cast<std::size_t>(c)] = a;
    });
    DD s{0.0, 0.0};
    for (const DD& p : part) s = s + p;
    return s;
}

// W plain sums over the rows: body(i0, i1, part) adds rows [i0, i1) into part[0 .. W)
// (zeroed), per fixed chunk; the chunks added in order.
template <class F>
std::vector<double> chunk_sums(lint n, lint W, F&& body) {
    const lint nc = row_chunks(n);
    std::vector<double> part(static_cast<std::size_t>(nc * W), 0.0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) { body(i0, i1, part.data() + c * W); });
    std::vector<double> t(static_cast<std::size_t>(W), 0.0);
    for (lint c = 0; c < nc; ++c)
        for (lint q = 0; q < W; ++q) t[q] += part[c * W + q];
    return t;
}

// ---- the norms (G54): psi, the IRLS weight psi(z)/z, rho and psi' -------------------------
// statsmodels' HuberT(t), TukeyBiweight(c), Hampel(a, b, c); 1 - u^2 as (1 - u)(1 + u)
struct NormF {
    int kind;
    double a, b, c;  // Huber: a = t; bisquare: c; Hampel: a, b, c
    double psi(double z) const {
        const double az = std::fabs(z);
        switch (kind) {
            case nHuber: return az <= a ? z : std::copysign(a, z);
            case nBisquare: {
                if (!(az <= c)) return 0.0;
                const double u = z / c, t = (1.0 - u) * (1.0 + u);
                return z * t * t;
            }
            default:
                if (az <= a) return z;
                if (az <= b) return std::copysign(a, z);
                if (az <= c) return std::copysign(a * (c - az) / (c - b), z);
                return 0.0;
        }
    }
    double weight(double z) const {
        const double az = std::fabs(z);
        switch (kind) {
            case nHuber: return az <= a ? 1.0 : a / az;
            case nBisquare: {
                if (!(az <= c)) return 0.0;
                const double u = z / c, t = (1.0 - u) * (1.0 + u);
                return t * t;
            }
            default:
                if (az <= a) return 1.0;
                if (az <= b) return a / az;
                if (az <= c) return a * (c - az) / (az * (c - b));
                return 0.0;
        }
    }
    double rho(double z) const {
        const double az = std::fabs(z);
        switch (kind) {
            case nHuber: return az <= a ? 0.5 * z * z : az * a - 0.5 * a * a;
            case nBisquare: {
                const double f = c * c / 6.0;
                if (!(az <= c)) return f;
                const double u = z / c, t = (1.0 - u) * (1.0 + u);
                return f * (1.0 - t * t * t);
            }
            default: {
                const double top = a * (b + c - a) * 0.5;
                if (az <= a) return 0.5 * z * z;
                if (az <= b) return a * az - 0.5 * a * a;
                if (az <= c) return top - a * (c - az) * (c - az) / (c - b) * 0.5;
                return top;
            }
        }
    }
    double dpsi(double z) const {
        const double az = std::fabs(z);
        switch (kind) {
            case nHuber: return az <= a ? 1.0 : 0.0;
            case nBisquare: {
                if (!(az <= c)) return 0.0;
                const double u = z / c, t = (1.0 - u) * (1.0 + u);
                return t * (t - 4.0 * u * u);
            }
            default:
                if (az <= a) return 1.0;
                if (az <= b) return 0.0;
                if (az <= c) return -a / (c - b);
                return 0.0;
        }
    }
};

// The S estimator's chi: robustbase's normalized bisquare rho~(u) = 1 - (1 - (u/c)^2)^3 (1
// for |u| > c), its derivative and the IRLS weight (1 - (u/c)^2)^2.
inline double chi_s(double u, double c) {
    if (!(std::fabs(u) <= c)) return 1.0;
    const double v = u / c, t = (1.0 - v) * (1.0 + v);
    return 1.0 - t * t * t;
}
inline double dchi_s(double u, double c) {
    if (!(std::fabs(u) <= c)) return 0.0;
    const double v = u / c, t = (1.0 - v) * (1.0 + v);
    return 6.0 * u * t * t / (c * c);
}
inline double wchi_s(double u, double c) {
    if (!(std::fabs(u) <= c)) return 0.0;
    const double v = u / c, t = (1.0 - v) * (1.0 + v);
    return t * t;
}

// z = r / s; a zero scale (an exact fit of most rows) gives 0 for a zero residual and
// +-inf otherwise
inline double std_resid(double r, double s) {
    if (s > 0.0) return r / s;
    return r == 0.0 ? 0.0 : std::copysign(kInf, r);
}

// ---- the design --------------------------------------------------------------------------
struct Design {
    lint n, p, k;
    bool intercept;
    double rtol;
    const std::byte* X;
    std::int64_t xrs, xcs;
    double at(lint i, lint c) const {
        return (intercept && c == 0) ? 1.0 : rd(X + i * xrs + (c - (intercept ? 1 : 0)) * xcs);
    }
};

const double kOneValue = 1.0;

// The columns of D with a coefficient (NaN marks a column without one), in order.
void design_cols(const Design& D, const double* beta, std::vector<dot2::ColRef>& cols, std::vector<double>& b) {
    cols.clear();
    b.clear();
    for (lint c = 0; c < D.k; ++c) {
        if (std::isnan(beta[c])) continue;
        if (D.intercept && c == 0) cols.push_back({reinterpret_cast<const std::byte*>(&kOneValue), 0});
        else cols.push_back({D.X + (c - (D.intercept ? 1 : 0)) * D.xcs, D.xrs});
        b.push_back(beta[c]);
    }
}

// r = y - X beta and fit = X beta for every row, each in double-double rounded once (Dot2;
// NaN marks a column without a coefficient). fit may be null.
void resid_dd(const Design& D, const double* y, const double* beta, double* r, double* fit) {
    std::vector<dot2::ColRef> cols;
    std::vector<double> b;
    design_cols(D, beta, cols, b);
    const lint nc = static_cast<lint>(cols.size());
    for_row_chunks(D.n, [&](lint, lint i0, lint i1) {
        constexpr lint kBlk = 256;
        double P[kBlk], S[kBlk];
        for (lint a = i0; a < i1; a += kBlk) {
            const lint nt = std::min(i1 - a, kBlk);
            dot2::dot2_rows(cols.data(), b.data(), nc, nt, [&](lint t) { return a + t; }, P, S);
            for (lint t = 0; t < nt; ++t) {
                const lint i = a + t;
                if (fit) fit[i] = P[t] + S[t];
                double hh, e;
                two_sum(y[i], -P[t], hh, e);
                r[i] = hh + (e - S[t]);
            }
        }
    });
}

// One weighted least squares solve through the least squares core (QR, G21's aliased
// columns with these weights, `steps` refinements): beta (NaN for aliased columns), kept,
// fit = X beta (Dot2). Returns the rank.
lint qr_wls(const Design& D, const double* y, const double* w, double* beta, std::uint8_t* kept, double* fit, int steps,
            Scratch& sc) {
    Problem pb{D.n, D.p, D.k, D.intercept, true, false, false, D.rtol, kNonrobust, steps, -1, 0, false};
    pb.lean = true;
    Data d{reinterpret_cast<const std::byte*>(y), 8, D.X, D.xrs, D.xcs, reinterpret_cast<const std::byte*>(w), 8,
           nullptr, 0, nullptr, 0};
    double st[kNStat];
    Dest o{beta, nullptr, nullptr, fit, nullptr, kept, st, nullptr};
    fit_one(pb, d, o, sc);
    return static_cast<lint>(st[kRank]);
}

// The normal equations X_K^T W X_K b = X_K^T W y over the columns K of `keep` (the GLM's
// inner solve, G40, regression/glm.cpp: chunked dsyrk/dgemv slots added in order, Cholesky,
// two triangular solves). False (the caller takes the QR) when the factorization fails,
// its condition estimate says the normal equations lose more than 1e-4, or G21's rule
// would drop a column. fit = X beta in plain double.
constexpr double kCholMaxErr = 1e-4;
constexpr double kCholMinWork = 512;

bool chol_wls(const Design& D, const double* y, const double* w, const std::uint8_t* keep, double* beta,
              std::uint8_t* kept, double* fit) {
    const lint n = D.n, k = D.k;
    if (double(n) * double(k) < kCholMinWork) return false;
    std::vector<lint> K;
    for (lint j = 0; j < k; ++j)
        if (keep[j]) K.push_back(j);
    const lint kk = static_cast<lint>(K.size());
    if (kk == 0) return false;
    const lint nc = row_chunks(n);
    const std::size_t slot = static_cast<std::size_t>(kk * kk + kk);
    std::vector<double> part(static_cast<std::size_t>(nc) * slot, 0.0);
    std::vector<std::uint8_t> any(static_cast<std::size_t>(nc), 0);
    std::vector<lint> cnt(static_cast<std::size_t>(nc), 0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        std::vector<double> A(static_cast<std::size_t>((i1 - i0) * kk)), v(static_cast<std::size_t>(i1 - i0));
        lint len = 0;
        for (lint i = i0; i < i1; ++i) {
            if (!(w[i] > 0.0)) continue;
            const double s = std::sqrt(w[i]);
            v[len] = s * y[i];
            double* a = A.data() + len * kk;
            for (lint jj = 0; jj < kk; ++jj) a[jj] = s * D.at(i, K[jj]);
            ++len;
        }
        cnt[static_cast<std::size_t>(c)] = len;
        if (len == 0) return;
        any[static_cast<std::size_t>(c)] = 1;
        double* G = part.data() + static_cast<std::size_t>(c) * slot;
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, kk, len, 1.0, A.data(), kk, 0.0, G, kk);
        cblas_dgemv(CblasColMajor, CblasNoTrans, kk, len, 1.0, A.data(), kk, v.data(), 1, 0.0, G + kk * kk, 1);
    });
    std::vector<double> G(slot, 0.0);
    lint m = 0;
    for (lint c = 0; c < nc; ++c) {
        m += cnt[static_cast<std::size_t>(c)];
        if (!any[static_cast<std::size_t>(c)]) continue;
        const double* q = part.data() + static_cast<std::size_t>(c) * slot;
        for (std::size_t e = 0; e < slot; ++e) G[e] += q[e];
    }
    double* U = G.data();
    double* h = G.data() + kk * kk;
    std::vector<double> nrm(static_cast<std::size_t>(kk));
    for (lint jj = 0; jj < kk; ++jj) nrm[jj] = std::sqrt(U[jj * kk + jj]);
    if (linalg::lapack::potrf('U', kk, U, kk) > 0) return false;
    const double rc = linalg::lapack::trcon_1('U', kk, U, kk);
    if (!(rc > 0.0) || 0x1p-53 / (rc * rc) > kCholMaxErr) return false;
    const double tol = D.rtol >= 0 ? D.rtol : static_cast<double>(std::max(m, kk)) * kEps;
    for (lint jj = 0; jj < kk; ++jj)
        if (!(std::fabs(U[jj * kk + jj]) > 2.0 * tol * nrm[jj])) return false;
    linalg::lapack::trtrs('U', 'T', 'N', kk, 1, U, kk, h, kk);
    linalg::lapack::trtrs('U', 'N', 'N', kk, 1, U, kk, h, kk);
    for (lint jj = 0; jj < kk; ++jj)
        if (!std::isfinite(h[jj])) return false;
    for (lint j = 0; j < k; ++j) beta[j] = kNaN, kept[j] = 0;
    for (lint jj = 0; jj < kk; ++jj) beta[K[jj]] = h[jj], kept[K[jj]] = 1;
    // fit in plain double, four rows at a time (each row's sum over the columns in order)
    std::vector<dot2::ColRef> cols;
    std::vector<double> hb;
    design_cols(D, beta, cols, hb);
    for_row_chunks(n, [&](lint, lint i0, lint i1) {
        dot2::plain_rows(cols.data(), hb.data(), kk, i1 - i0, [&](lint t) { return i0 + t; }, fit + i0);
    });
    return true;
}

// ---- scales ------------------------------------------------------------------------------

// statsmodels' mad(r, center=0): the median of |r_i| / Phi^-1(3/4) (the two middle values'
// mean for an even n, numpy's median), by selection. buf: n doubles.
double mad0(const double* r, lint n, double* buf) {
    if (n == 0) return kNaN;
    for (lint i = 0; i < n; ++i) buf[i] = std::fabs(r[i]) / kMadC;
    const lint mid = n / 2;
    std::nth_element(buf, buf + mid, buf + n);
    const double hi = buf[mid];
    if (n % 2) return hi;
    const double lo = *std::max_element(buf, buf + mid);
    return (lo + hi) / 2;
}

// Huber's proposal 2 (statsmodels' HuberScale): the s with sum chi(r_i/s) = T, chi(z) =
// min(z^2, d^2)/2, T = n h = n ((n - rank)/n) E chi(Z). In t = 1/s^2 the sum
// f(t) = sum min(r_i^2 t, d^2)/2 is concave, increasing and piecewise linear, so Newton's
// steps from t = 0 increase to the root and stop at it: each solves f = T on the current
// split of the rows, t = (T - n_out d^2/2) / (sum_in r_i^2 / 2). Sums in double-double.
double huber_scale(const double* r, lint n, double T) {
    if (!(T > 0.0)) return 0.0;  // n <= rank: the fit is exact, its scale 0 (as mscale)
    const double d2 = kHuberD * kHuberD;
    double t = 0.0;
    for (int it = 0; it < 1000; ++it) {
        const lint nc = row_chunks(n);
        std::vector<DD> part(static_cast<std::size_t>(nc), DD{0.0, 0.0});
        std::vector<lint> outs(static_cast<std::size_t>(nc), 0);
        for_row_chunks(n, [&](lint c, lint i0, lint i1) {
            DD a{0.0, 0.0};
            lint o = 0;
            for (lint i = i0; i < i1; ++i) {
                const double q = r[i] * r[i];
                if (q * t < d2) {
                    double hp, ep;
                    two_prod(r[i], r[i], hp, ep);
                    a = a + DD{hp, ep};
                } else {
                    ++o;
                }
            }
            part[static_cast<std::size_t>(c)] = a;
            outs[static_cast<std::size_t>(c)] = o;
        });
        DD A{0.0, 0.0};
        lint nout = 0;
        for (lint c = 0; c < nc; ++c) A = A + part[static_cast<std::size_t>(c)], nout += outs[static_cast<std::size_t>(c)];
        const double num = T - static_cast<double>(nout) * d2 * 0.5;
        const double den = 0.5 * (A.hi + A.lo);
        // the rows inside are all exactly 0: f stays n_out d^2/2 for every larger t, so below
        // T there is no root and the scale is 0 (an exact fit of the rows inside)
        if (!(den > 0.0)) return (num > 0.0 || !(t > 0.0)) ? 0.0 : 1.0 / std::sqrt(t);
        const double tn = num / den;
        if (!(tn > t)) break;
        t = tn;
    }
    return t > 0.0 ? 1.0 / std::sqrt(t) : kInf;
}

// The S estimator's M-scale: the s > 0 with sum rho~(r_i / s) = T (rho~ the normalized
// bisquare chi, T = (n - p) b), 0 when at most T residuals are nonzero (robustbase's
// "exact fit"). g(s) = sum rho~(r_i/s) - T decreases from #nonzero - T > 0 (s -> 0) to -T:
// safeguarded Newton from s0 inside the bracket the signs give (halving or doubling when a
// step leaves it), to rounding (exact) or a relative 1e-10 (robustbase's scale.tol, for the
// candidates). One pass gives g and g'.
double mscale(const double* r, lint n, double c, double T, double s0, bool exact) {
    lint nz = 0;
    double amax = 0.0;
    for (lint i = 0; i < n; ++i)
        if (r[i] != 0.0) ++nz, amax = std::max(amax, std::fabs(r[i]));
    if (!(static_cast<double>(nz) > T) || !(T > 0.0)) return 0.0;  // T <= 0: n <= p, an exact fit
    const lint nc = row_chunks(n);
    std::vector<DD> pg(static_cast<std::size_t>(nc)), pd(static_cast<std::size_t>(nc));
    auto eval = [&](double s, double& g, double& dg) {
        for_row_chunks(n, [&](lint ch, lint i0, lint i1) {
            DD a{0.0, 0.0}, b{0.0, 0.0};
            for (lint i = i0; i < i1; ++i) {
                const double u = r[i] / s;
                if (!(std::fabs(u) <= c)) {
                    a = a + DD{1.0, 0.0};
                    continue;
                }
                const double v = u / c, t = (1.0 - v) * (1.0 + v);
                a = a + DD{1.0 - t * t * t, 0.0};
                b = b + DD{6.0 * u * u * t * t / (c * c), 0.0};
            }
            pg[static_cast<std::size_t>(ch)] = a, pd[static_cast<std::size_t>(ch)] = b;
        });
        DD A{0.0, 0.0}, B{0.0, 0.0};
        for (lint ch = 0; ch < nc; ++ch) A = A + pg[static_cast<std::size_t>(ch)], B = B + pd[static_cast<std::size_t>(ch)];
        g = (A.hi - T) + A.lo;
        dg = -(B.hi + B.lo) / s;  // d/ds sum rho~(r/s) = -sum rho~'(u) u / s
    };
    if (!(s0 > 0.0 && std::isfinite(s0))) s0 = amax / c;
    const double rel = exact ? 2.0 * kEps : 1e-10;
    double lo = 0.0, hi = kInf, s = s0;
    for (int it = 0; it < 300; ++it) {
        double g, dg;
        eval(s, g, dg);
        if (g > 0.0) lo = s;
        else if (g < 0.0) hi = s;
        else return s;
        double sn = dg < 0.0 ? s - g / dg : kNaN;
        if (std::fabs(sn - s) <= rel * s) return sn;  // Newton's step is below the tolerance
        if (!(sn > lo && sn < hi)) {
            // outside the bracket: bisect it, or double / halve towards the unknown end
            if (std::isinf(hi)) sn = 2.0 * s;
            else if (lo > 0.0) sn = 0.5 * (lo + hi);
            else sn = 0.5 * s;
        }
        if (std::isfinite(hi) && lo > 0.0 && hi - lo <= rel * hi) return 0.5 * (lo + hi);
        s = sn;
    }
    return s;
}

// ---- IRLS --------------------------------------------------------------------------------

struct Buffers {
    std::vector<double> w, fit, buf, beta_new;
    std::vector<std::uint8_t> kept_new;
    Buffers(lint n, lint k)
        : w(static_cast<std::size_t>(std::max<lint>(n, 1))), fit(w.size()), buf(w.size()),
          beta_new(static_cast<std::size_t>(std::max<lint>(k, 1))), kept_new(beta_new.size()) {}
};

struct ScaleRule {
    int mode;  // sMad, sHuber, sFixed, sMScale
    double T;  // Huber's proposal 2: the target n h; the M-scale: (n - p) b
};

double new_scale(const ScaleRule& sr, const double* r, lint n, double s, Buffers& B) {
    switch (sr.mode) {
        case sMad: return mad0(r, n, B.buf.data());
        case sHuber: return huber_scale(r, n, sr.T);
        case sMScale: return mscale(r, n, kSChi, sr.T, s, true);
        default: return s;
    }
}

// ---- exact fits to rounding --------------------------------------------------------------
// A residual is 0 to rounding when |r_i| <= (k + 16) u (|y_i| + ||x_i||_1 ||beta||_inf),
// quantreg's snapping tolerance (regression/quantile.cpp): rounding beta alone moves a row
// on the fitted plane by about that much. When the rows that decide the scale are 0 to
// rounding, the scale rule would give 0 for the exact fit: MAD, more than half the rows;
// Huber's proposal 2, (n - zeros) d^2/2 <= T (no root, as huber_scale); the M-scale, at most
// T rows not 0 (as mscale). The iterations then stop as at a scale of exactly 0
// (docs/api/regression.md §5): the residuals of the rows on the fit are rounding noise, and
// over a scale of the same size their psi, the weights and Newton's model are noise too (an
// IRLS or Newton step from there can land anywhere, with a smaller |F|).
struct ZeroTol {
    double ymax = 0.0, x1max = 0.0;  // max |y_i| and max ||x_i||_1, for a quick bound first
};

ZeroTol zero_tol(const Design& D, const double* y) {
    ZeroTol z;
    for (lint i = 0; i < D.n; ++i) {
        double l1 = 0.0;
        for (lint c = 0; c < D.k; ++c) l1 += std::fabs(D.at(i, c));
        z.ymax = std::max(z.ymax, std::fabs(y[i]));
        z.x1max = std::max(z.x1max, l1);
    }
    return z;
}

inline double beta_inf(const double* beta, lint k) {  // NaN marks a column without a coefficient
    double m = 0.0;
    for (lint c = 0; c < k; ++c)
        if (!std::isnan(beta[c])) m = std::max(m, std::fabs(beta[c]));
    return m;
}

inline double zero_tol_row(const Design& D, const double* y, lint i, double bn) {
    double l1 = 0.0;
    for (lint c = 0; c < D.k; ++c) l1 += std::fabs(D.at(i, c));
    return (static_cast<double>(D.k) + 16.0) * (0.5 * kEps) * (std::fabs(y[i]) + l1 * bn);
}

// The rows 0 to rounding a zero scale needs (more than n: the rule has no exact fit).
lint zero_need(int mode, double T, lint n) {
    switch (mode) {
        case sMad: return n / 2 + 1;
        case sHuber: return std::max<lint>(1, n - static_cast<lint>(std::floor(2.0 * T / (kHuberD * kHuberD))));
        case sMScale: return std::max<lint>(1, n - static_cast<lint>(std::floor(T)));
        default: return n + 1;
    }
}

// Whether at least `need` residuals are 0 to rounding (the quick bound with the largest
// tolerance first, so a fit far from exact costs one pass over r).
bool zero_scale(const Design& D, const ZeroTol& zt, const double* y, const double* beta, const double* r, lint need) {
    const lint n = D.n;
    if (need > n) return false;
    const double bn = beta_inf(beta, D.k);
    const double emax = (static_cast<double>(D.k) + 16.0) * (0.5 * kEps) * (zt.ymax + zt.x1max * bn);
    lint c0 = 0;
    for (lint i = 0; i < n; ++i) c0 += std::fabs(r[i]) <= emax;
    if (c0 < need) return false;
    lint c = 0;
    for (lint i = 0; i < n; ++i)
        if (std::fabs(r[i]) <= emax && std::fabs(r[i]) <= zero_tol_row(D, y, i, bn)) ++c;
    return c >= need;
}

// The exact fit the iterations tend to as the scale goes to 0: the least squares fit of the
// rows 0 to rounding (weight 1, the others 0: the weights an exact fit reports), by the
// refined QR. beta, kept and r (Dot2) become its; unchanged when it would lose a column.
void exact_fit(const Design& D, const double* y, double* beta, std::uint8_t* kept, double* r, Buffers& B, Scratch& sc) {
    const lint n = D.n, k = D.k;
    const double bn = beta_inf(beta, k);
    lint had = 0;
    for (lint c = 0; c < k; ++c) had += kept[c] != 0;
    for (lint i = 0; i < n; ++i) B.w[i] = std::fabs(r[i]) <= zero_tol_row(D, y, i, bn) ? 1.0 : 0.0;
    if (qr_wls(D, y, B.w.data(), B.beta_new.data(), B.kept_new.data(), B.fit.data(), kRefineSteps, sc) < had) return;
    std::copy(B.beta_new.begin(), B.beta_new.begin() + k, beta);
    std::copy(B.kept_new.begin(), B.kept_new.begin() + k, kept);
    resid_dd(D, y, beta, r, nullptr);
}

// Newton's step is taken only when the new scale stays within this factor of the old: its
// model linearizes s(beta), and |F| = |sum psi(r_i/s) x_i| also falls as s grows (psi(z) ~ z
// for small z), so a step that blows the scale up would pass the |F| test.
constexpr double kNewtonScaleFactor = 2.0;
inline bool scale_kept(double sn, double s) {
    return sn > 0.0 && std::isfinite(sn) && sn <= kNewtonScaleFactor * s && s <= kNewtonScaleFactor * sn;
}

inline bool settled(DD a, DD b, double tol) {  // |b - a|/(|b| + 0.1) < tol
    const double d = (b.hi - a.hi) + (b.lo - a.lo);
    return std::fabs(d) / (std::fabs(b.hi) + 0.1) < tol;
}

struct IrlsOut {
    int iters = 0;
    bool converged = false;
};

// The gradient of the scale with respect to beta at residuals r (scale s), for Newton's
// step: MAD, s = |r_m| / C (the mean of two such terms for an even n), so ds = -sign(r_m)
// x_m / C; Huber's proposal 2, s^2 = sum_in r_i^2 / (2T - n_out d^2) on the rows inside
// |r| < d s; the S estimator's M-scale (sum rho~(r_i/s) = T, differentiated: sum rho~'(z_i)
// (x_i + z_i ds) = 0), ds = -sum psi(z_i) x_i / sum psi(z_i) z_i (psi proportional to rho~';
// 0 at the fixed point, where the first sum is F); a fixed scale, 0. Over the columns K.
void scale_grad(const Design& D, const ScaleRule& sr, const double* r, double s, const std::vector<lint>& K,
                std::vector<double>& g) {
    const lint n = D.n, kk = static_cast<lint>(K.size());
    g.assign(static_cast<std::size_t>(kk), 0.0);
    if (sr.mode == sFixed) return;
    if (sr.mode == sMScale) {
        const NormF nf{nBisquare, 0.0, 0.0, kSChi};
        const std::vector<double> acc = chunk_sums(n, kk + 1, [&](lint i0, lint i1, double* P) {
            for (lint i = i0; i < i1; ++i) {
                const double z = r[i] / s, q = nf.psi(z);
                if (q == 0.0) continue;
                P[kk] += q * z;
                for (lint a = 0; a < kk; ++a) P[a] += q * D.at(i, K[a]);
            }
        });
        const double den = acc[kk];
        if (den > 0.0)
            for (lint a = 0; a < kk; ++a) g[a] = -acc[a] / den;
        return;
    }
    if (sr.mode == sMad) {
        // the rows of the middle |r| / C (s itself for an odd n; the two around it for an even n)
        std::vector<std::pair<double, lint>> v(static_cast<std::size_t>(n));
        for (lint i = 0; i < n; ++i) v[i] = {std::fabs(r[i]) / kMadC, i};
        const lint mid = n / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        std::vector<lint> rows{v[mid].second};
        if (n % 2 == 0) rows.push_back(std::max_element(v.begin(), v.begin() + mid)->second);
        const double f = 1.0 / (kMadC * static_cast<double>(rows.size()));
        for (lint i : rows) {
            const double sg = r[i] > 0.0 ? 1.0 : (r[i] < 0.0 ? -1.0 : 0.0);
            for (lint a = 0; a < kk; ++a) g[a] -= sg * D.at(i, K[a]) * f;
        }
        return;
    }
    // Huber's proposal 2
    const double d2 = kHuberD * kHuberD, t = 1.0 / (s * s);
    lint nout = 0;
    for (lint i = 0; i < n; ++i) nout += !(r[i] * r[i] * t < d2);
    const std::vector<double> acc = chunk_sums(n, kk, [&](lint i0, lint i1, double* P) {
        for (lint i = i0; i < i1; ++i)
            if (r[i] * r[i] * t < d2)
                for (lint a = 0; a < kk; ++a) P[a] += r[i] * D.at(i, K[a]);
    });
    const double den = 2.0 * sr.T - static_cast<double>(nout) * d2;
    for (lint a = 0; a < kk; ++a) g[a] = -acc[a] / (s * den);
}

// The estimating equations F = sum psi(r_i/s) x_i over the columns K (double-double per
// column, chunks added in order) and their 2-norm.
double estimating(const Design& D, const NormF& nf, const double* r, double s, const std::vector<lint>& K,
                  std::vector<double>& F) {
    const lint n = D.n, kk = static_cast<lint>(K.size());
    const lint nc = row_chunks(n);
    std::vector<DD> part(static_cast<std::size_t>(nc * kk), DD{0.0, 0.0});
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        DD* P = part.data() + c * kk;
        for (lint i = i0; i < i1; ++i) {
            const double q = nf.psi(r[i] / s);
            if (q == 0.0) continue;
            for (lint a = 0; a < kk; ++a) {
                double hp, ep;
                two_prod(q, D.at(i, K[a]), hp, ep);
                P[a] = P[a] + DD{hp, ep};
            }
        }
    });
    F.assign(static_cast<std::size_t>(kk), 0.0);
    double nrm = 0.0;
    for (lint a = 0; a < kk; ++a) {
        DD t{0.0, 0.0};
        for (lint c = 0; c < nc; ++c) t = t + part[c * kk + a];
        F[a] = t.hi + t.lo;
        nrm += F[a] * F[a];
    }
    return std::sqrt(nrm);
}

// Newton's step for F(beta) = sum psi(r_i/s(beta)) x_i = 0 with the scale's dependence on
// beta: F' = -(1/s) [X^T Psi' X + (X^T Psi' z) (grad s)^T], so beta_new = beta + s M^-1 F.
// False when M is singular.
bool newton_step(const Design& D, const NormF& nf, const ScaleRule& sr, const double* beta, const double* r, double s,
                 const std::vector<lint>& K, const std::vector<double>& F, double* beta_new) {
    const lint n = D.n, kk = static_cast<lint>(K.size());
    if (kk == 0) return false;
    const lint nc = row_chunks(n);
    const std::size_t slot = static_cast<std::size_t>(kk * kk + kk);
    std::vector<double> part(static_cast<std::size_t>(nc) * slot, 0.0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        double* P = part.data() + static_cast<std::size_t>(c) * slot;
        std::vector<double> x(static_cast<std::size_t>(kk));
        for (lint i = i0; i < i1; ++i) {
            const double z = r[i] / s, dp = nf.dpsi(z);
            if (dp == 0.0) continue;
            for (lint a = 0; a < kk; ++a) x[a] = D.at(i, K[a]);
            for (lint a = 0; a < kk; ++a) {
                const double da = dp * x[a];
                for (lint b = 0; b < kk; ++b) P[a * kk + b] += da * x[b];
                P[kk * kk + a] += da * z;
            }
        }
    });
    std::vector<double> M(static_cast<std::size_t>(kk * kk), 0.0), u(static_cast<std::size_t>(kk), 0.0), g;
    for (lint c = 0; c < nc; ++c) {
        const double* P = part.data() + static_cast<std::size_t>(c) * slot;
        for (lint e = 0; e < kk * kk; ++e) M[e] += P[e];
        for (lint a = 0; a < kk; ++a) u[a] += P[kk * kk + a];
    }
    scale_grad(D, sr, r, s, K, g);
    for (lint a = 0; a < kk; ++a)
        for (lint b = 0; b < kk; ++b) M[a * kk + b] += u[a] * g[b];
    // M is row-major: solve M^T in LAPACK's column-major view, i.e. transpose
    std::vector<double> Mc(M.size());
    for (lint a = 0; a < kk; ++a)
        for (lint b = 0; b < kk; ++b) Mc[b * kk + a] = M[a * kk + b];
    std::vector<lint> piv(static_cast<std::size_t>(kk));
    std::vector<double> d(static_cast<std::size_t>(kk));
    for (lint a = 0; a < kk; ++a) d[a] = s * F[a];
    if (linalg::lapack::getrf(kk, kk, Mc.data(), kk, piv.data()) != 0) return false;
    linalg::lapack::getrs('N', kk, 1, Mc.data(), kk, piv.data(), d.data(), kk);
    for (lint c = 0; c < D.k; ++c) beta_new[c] = kNaN;
    for (lint a = 0; a < kk; ++a) {
        if (!std::isfinite(d[a])) return false;
        beta_new[K[a]] = beta[K[a]] + d[a];
    }
    return true;
}

// G44's test for the IRLS (regression/glm.cpp): steps not below half the one before three
// times in a row (a rate of at least 1/2, dozens of iterations to rounding) switch to
// Newton's steps.
constexpr double kSlowRatio = 0.5;
constexpr int kSlowRuns = 3;

// IRLS from beta (r its residuals, s their scale) for at most max_iter iterations (least
// squares solves or Newton's steps). Past the objective's convergence, and earlier when the
// IRLS is seen to be slow, the iterations are Newton's steps for the estimating equations
// (the scale's dependence on beta included), each taken when it lowers |F| and keeps the
// scale within kNewtonScaleFactor (else the refined weighted least squares). A scale whose
// rows are 0 to rounding (zero_scale) is set to 0: the iterations stop there as at an exact
// fit, converged. On return beta, kept, r (Dot2) and s are the last iterate's.
IrlsOut irls(const Design& D, const double* y, const NormF& nf, const ScaleRule& sr, int max_iter, double tol,
             double* beta, std::uint8_t* kept, double* r, double& s, Buffers& B, Scratch& sc) {
    IrlsOut out;
    const lint n = D.n, k = D.k;
    DD dold{kInf, 0.0};
    bool conv = false, refined = true, newton_mode = false;
    double prevstep = kInf;
    int slow_runs = 0;
    int it = 0;
    std::vector<double> F, Fn, rn(static_cast<std::size_t>(std::max<lint>(n, 1)));
    const lint need = zero_need(sr.mode, sr.T, n);
    const ZeroTol zt = need <= n ? zero_tol(D, y) : ZeroTol{};
    bool snapped = false;
    const auto at_zero = [&] {
        snapped = snapped || (s > 0.0 && std::isfinite(s) && zero_scale(D, zt, y, beta, r, need));
        return snapped;
    };
    bool polished = false;  // the polish reached rounding (not cut by max_iter)
    while (it < max_iter) {
        if (at_zero()) s = 0.0;
        if (!(s > 0.0) || !std::isfinite(s)) {
            if (s == 0.0) conv = polished = true;
            break;
        }
        bool took_newton = false;
        if (conv || newton_mode) {
            std::vector<lint> K;
            for (lint c = 0; c < k; ++c)
                if (kept[c]) K.push_back(c);
            const double f0 = estimating(D, nf, r, s, K, F);
            if (newton_step(D, nf, sr, beta, r, s, K, F, B.beta_new.data())) {
                resid_dd(D, y, B.beta_new.data(), rn.data(), nullptr);
                const double sn = new_scale(sr, rn.data(), n, s, B);
                if (scale_kept(sn, s) && estimating(D, nf, rn.data(), sn, K, Fn) < f0) {
                    std::copy(rn.begin(), rn.begin() + n, r);
                    std::copy(kept, kept + k, B.kept_new.begin());
                    s = sn;
                    took_newton = true;
                }
            }
        }
        if (!took_newton) {
            for_row_chunks(n, [&](lint, lint i0, lint i1) {
                for (lint i = i0; i < i1; ++i) B.w[i] = nf.weight(r[i] / s);
            });
            const bool fine = conv || newton_mode;
            const bool by_chol = !fine && chol_wls(D, y, B.w.data(), kept, B.beta_new.data(), B.kept_new.data(), B.fit.data());
            if (!by_chol) qr_wls(D, y, B.w.data(), B.beta_new.data(), B.kept_new.data(), B.fit.data(), fine ? kRefineSteps : 0, sc);
            if (by_chol) {
                for (lint i = 0; i < n; ++i) r[i] = y[i] - B.fit[i];
            } else {
                resid_dd(D, y, B.beta_new.data(), r, nullptr);
            }
            s = new_scale(sr, r, n, s, B);
            refined = fine;
        } else {
            refined = true;
        }
        ++it;
        double step = 0.0, bn = 0.0;
        for (lint c = 0; c < k; ++c) {
            if (std::isnan(B.beta_new[c])) continue;
            bn = std::max(bn, std::fabs(B.beta_new[c]));
            if (!std::isnan(beta[c])) step = std::max(step, std::fabs(B.beta_new[c] - beta[c]));
            else step = kInf;
        }
        std::copy(B.beta_new.begin(), B.beta_new.begin() + k, beta);
        std::copy(B.kept_new.begin(), B.kept_new.begin() + k, kept);
        const DD dev = (s > 0.0) ? row_sum(n, [&](lint i) { return nf.rho(r[i] / s); }) : DD{0.0, 0.0};
        if (!conv) {
            if (settled(dold, dev, tol)) conv = true;
            dold = dev;
            if (!newton_mode) {
                if (step < kInf && prevstep < kInf && prevstep > 0.0 && step >= kSlowRatio * prevstep) {
                    if (++slow_runs >= kSlowRuns) newton_mode = true;
                } else {
                    slow_runs = 0;
                }
            }
            prevstep = (conv || newton_mode) ? kInf : step;
            if (!conv) continue;
            if (!newton_mode) continue;  // the first polishing step follows
        }
        // past the objective's convergence: go on while the coefficients still move by more
        // than rounding (the GLM's rule): stop when this step is below it, when the remaining
        // ones predicted from the last two steps' ratio are, or when they stop contracting
        const double lim = 0x1p-50 * bn;
        if (step <= lim) {
            polished = true;
            break;
        }
        if (prevstep < kInf) {
            const double q = step / prevstep;
            if ((q < 1.0 && step * q / (1.0 - q) <= lim) || q >= 1.0) {
                polished = true;
                break;
            }
        }
        prevstep = step;
    }
    if (at_zero()) s = 0.0, conv = polished = true;  // max_iter (or the polishing's end) reached it
    if (snapped) exact_fit(D, y, beta, kept, r, B, sc);
    out.iters = it;
    // converged: the objective's test passed and the polish finished; max_iter cutting the
    // polish short is not converged (decision 1, G93)
    out.converged = conv && polished;
    if (!refined && it > 0 && s > 0.0) {
        // the last iteration's least squares by the refined QR (its weights are in B.w)
        qr_wls(D, y, B.w.data(), beta, kept, B.fit.data(), kRefineSteps, sc);
        resid_dd(D, y, beta, r, nullptr);
        s = new_scale(sr, r, n, s, B);
    }
    return out;
}

// ---- fast-S (MM's start) ----------------------------------------------------------------

struct SOut {
    std::vector<double> beta;  // p (the packed design's columns)
    double scale = kNaN;
    bool converged = false;
    bool ok = false;
};

// The candidates' local improvement and the refinement steps: one scale step
// s <- s sqrt(sum rho~(r/s) / T) (robustbase's refine_fast_s), the bisquare chi weights at
// r/s, one weighted least squares fit. r: the residuals at beta (plain double).
bool s_step(const Design& Dk, const double* y, double* beta, double* r, double& s, double T, Buffers& B, Scratch& sc) {
    const lint n = Dk.n, p = Dk.k;
    const DD v = row_sum(n, [&](lint i) { return chi_s(r[i] / s, kSChi); });
    s = s * std::sqrt((v.hi + v.lo) / T);
    if (!(s > 0.0) || !std::isfinite(s)) return false;
    for (lint i = 0; i < n; ++i) B.w[i] = wchi_s(r[i] / s, kSChi);
    std::vector<std::uint8_t> all(static_cast<std::size_t>(p), 1);
    if (!chol_wls(Dk, y, B.w.data(), all.data(), B.beta_new.data(), B.kept_new.data(), B.fit.data())) {
        const lint rk = qr_wls(Dk, y, B.w.data(), B.beta_new.data(), B.kept_new.data(), B.fit.data(), 0, sc);
        if (rk < p) return false;
    }
    std::copy(B.beta_new.begin(), B.beta_new.begin() + p, beta);
    for (lint i = 0; i < n; ++i) r[i] = y[i] - B.fit[i];
    return true;
}

// r = y - X beta in plain double over the packed rows (the candidates' residuals)
void resid_plain(const double* Xr, lint n, lint p, const double* y, const double* beta, double* r) {
    for_row_chunks(n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            double acc = 0.0;
            for (lint c = 0; c < p; ++c) acc += Xr[i * p + c] * beta[c];
            r[i] = y[i] - acc;
        }
    });
}

// Candidate j: an elemental subset (rows drawn in random order, each kept only when it is
// linearly independent of those kept: its component orthogonal to them above solve.tol of
// its norm, the effect of robustbase's "nonsingular" subsampling), the exact solution
// through it and one local improvement step (k.fast.s = 1) from robustbase's initial scale
// 1.4826 median |r|. beta: the improved coefficients, s0 the step's scale. False when no
// subset or step is found.
bool candidate(const Design& Dk, const double* Xr, const double* y, random::Key key, std::int64_t j, double T,
               double* beta, double& s0, Buffers& B, std::vector<double>& r, Scratch& sc) {
    const lint n = Dk.n, p = Dk.k;
    random::Stream st(key);
    std::vector<lint> rows;
    std::vector<double> Q(static_cast<std::size_t>(p * p)), v(static_cast<std::size_t>(p));
    for (int t = 0; t < kMts && static_cast<lint>(rows.size()) < p; ++t) {
        const std::uint64_t e = (static_cast<std::uint64_t>(j) << 32) | static_cast<std::uint64_t>(t);
        const lint i = static_cast<lint>(
            random::bounded(st.first(e), static_cast<std::uint64_t>(n), [&](std::uint64_t q) { return st.extra(e, q); }));
        if (std::find(rows.begin(), rows.end(), i) != rows.end()) continue;
        double nx = 0.0;
        for (lint c = 0; c < p; ++c) v[c] = Xr[i * p + c], nx += v[c] * v[c];
        nx = std::sqrt(nx);
        if (!(nx > 0.0)) continue;
        const lint have = static_cast<lint>(rows.size());
        for (int pass = 0; pass < 2; ++pass)
            for (lint q = 0; q < have; ++q) {
                double dq = 0.0;
                for (lint c = 0; c < p; ++c) dq += Q[q * p + c] * v[c];
                for (lint c = 0; c < p; ++c) v[c] -= dq * Q[q * p + c];
            }
        double nv = 0.0;
        for (lint c = 0; c < p; ++c) nv += v[c] * v[c];
        nv = std::sqrt(nv);
        if (!(nv > kSolveTol * nx)) continue;
        for (lint c = 0; c < p; ++c) Q[have * p + c] = v[c] / nv;
        rows.push_back(i);
    }
    if (static_cast<lint>(rows.size()) < p) return false;
    std::vector<double> A(static_cast<std::size_t>(p * p));
    std::vector<lint> piv(static_cast<std::size_t>(p));
    for (lint a = 0; a < p; ++a) {
        for (lint c = 0; c < p; ++c) A[c * p + a] = Xr[rows[a] * p + c];
        beta[a] = y[rows[a]];
    }
    if (linalg::lapack::getrf(p, p, A.data(), p, piv.data()) != 0) return false;
    linalg::lapack::getrs('N', p, 1, A.data(), p, piv.data(), beta, p);
    for (lint c = 0; c < p; ++c)
        if (!std::isfinite(beta[c])) return false;
    resid_plain(Xr, n, p, y, beta, r.data());
    double* b = B.buf.data();
    for (lint i = 0; i < n; ++i) b[i] = std::fabs(r[i]);
    const lint mid = n / 2;
    std::nth_element(b, b + mid, b + n);
    s0 = kRbMad * (n % 2 ? b[mid] : 0.5 * (b[mid] + *std::max_element(b, b + mid)));
    if (s0 > 0.0 && !s_step(Dk, y, beta, r.data(), s0, T, B, sc)) return false;
    return true;
}

// A kept candidate: its coefficients and scale.
struct Cand {
    std::vector<double> beta;
    double scale;
};

// The candidates jbase .. jbase + nres - 1 on the design (Dk, Xr, y; T = (n - p) b) and
// robustbase's selection of the kBestR best among them.
std::vector<Cand> best_candidates(const Design& Dk, const double* Xr, const double* y, random::Key key,
                                  std::int64_t jbase, int nres, double T) {
    const lint n = Dk.n, p = Dk.k;
    // 1. the candidates, each on its own (in parallel for a large regression)
    std::vector<double> cb(static_cast<std::size_t>(nres * p), kNaN), c0(static_cast<std::size_t>(nres), kNaN);
    std::vector<std::uint8_t> ok(static_cast<std::size_t>(nres), 0);
    constexpr std::int64_t kPerTask = 4;
    const std::int64_t ntasks = (nres + kPerTask - 1) / kPerTask;
    auto task = [&](std::int64_t t) {
        Buffers B(n, p);
        std::vector<double> r(static_cast<std::size_t>(std::max<lint>(n, 1)));
        Scratch sc;
        for (std::int64_t j = t * kPerTask; j < std::min<std::int64_t>(nres, (t + 1) * kPerTask); ++j) {
            double s0 = kNaN;
            ok[static_cast<std::size_t>(j)] =
                candidate(Dk, Xr, y, key, jbase + j, T, cb.data() + j * p, s0, B, r, sc) ? 1 : 0;
            c0[static_cast<std::size_t>(j)] = s0;
        }
    };
    if (ntasks >= 2 && double(n) * double(p) * double(nres) >= 1e5)
        parallel::parallel_for(ntasks, task);
    else
        for (std::int64_t t = 0; t < ntasks; ++t) task(t);
    // 2. robustbase's selection, in candidate order: a candidate whose residuals give
    // sum rho~(r / worst) < T has a smaller scale than the worst kept one; only then is its
    // scale found, and it replaces the worst
    std::vector<double> r(static_cast<std::size_t>(std::max<lint>(n, 1)));
    std::vector<Cand> best;
    for (std::int64_t j = 0; j < nres; ++j) {
        if (!ok[static_cast<std::size_t>(j)]) continue;
        resid_plain(Xr, n, p, y, cb.data() + j * p, r.data());
        double worst = kInf;
        std::size_t wi = 0;
        if (static_cast<int>(best.size()) == kBestR)
            for (std::size_t q = 0; q < best.size(); ++q)
                if (best[q].scale > worst || q == 0) worst = best[q].scale, wi = q;
        if (std::isfinite(worst)) {
            if (worst == 0.0) continue;
            const DD v = row_sum(n, [&](lint i) { return chi_s(r[i] / worst, kSChi); });
            if (!((v.hi - T) + v.lo < 0.0)) continue;
        }
        const double sj = mscale(r.data(), n, kSChi, T, c0[static_cast<std::size_t>(j)], false);
        if (!std::isfinite(sj)) continue;
        Cand c{std::vector<double>(cb.begin() + j * p, cb.begin() + (j + 1) * p), sj};
        if (static_cast<int>(best.size()) < kBestR) best.push_back(std::move(c));
        else best[wi] = std::move(c);
    }
    return best;
}

// robustbase's large-n fast-S (lmrob.control(): fast.s.large.n 2000, groups 5, n.group 400;
// Salibian-Barrera and Yohai 2006, section 5): beyond kLargeN rows, kGroups x kNGroup rows
// drawn without replacement are cut into kGroups groups; each group gets the small-n search
// (nResample candidates, the best kBestR); the kGroups kBestR candidates take one more local
// step on the merged rows from their group's scale (k.fast.s), and the best kBestR by the
// merged rows' scale go on to the refinement on all the rows.
constexpr lint kLargeN = 2000;
constexpr lint kGroups = 5;
constexpr lint kNGroup = 400;

SOut fast_s(const Design& Dk, const double* Xr, const double* y, random::Key key, int nres, bool large, Scratch& sc0) {
    const lint n = Dk.n, p = Dk.k;
    SOut out;
    const double T = (static_cast<double>(n) - static_cast<double>(p)) * kSB;
    std::vector<double> r(static_cast<std::size_t>(std::max<lint>(n, 1)));
    std::vector<Cand> best;
    if (large && n > kLargeN && kNGroup > p) {
        // the merged rows: a partial Fisher-Yates shuffle from the stream's elements 2^63 + t
        // (the candidates' elements stay below 2^63)
        const lint m = kGroups * kNGroup;
        random::Stream st(key);
        std::unordered_map<lint, lint> moved;
        std::vector<lint> rows(static_cast<std::size_t>(m));
        for (lint t = 0; t < m; ++t) {
            const std::uint64_t e = (std::uint64_t(1) << 63) | static_cast<std::uint64_t>(t);
            const lint u = t + static_cast<lint>(random::bounded(st.first(e), static_cast<std::uint64_t>(n - t),
                                                                 [&](std::uint64_t q) { return st.extra(e, q); }));
            const auto at = [&](lint i) {
                const auto it = moved.find(i);
                return it == moved.end() ? i : it->second;
            };
            const lint vu = at(u), vt = at(t);
            moved[u] = vt;
            moved[t] = vu;
            rows[static_cast<std::size_t>(t)] = vu;
        }
        std::vector<double> Xm(static_cast<std::size_t>(m * p)), ym(static_cast<std::size_t>(m));
        for (lint t = 0; t < m; ++t) {
            const lint i = rows[static_cast<std::size_t>(t)];
            std::copy(Xr + i * p, Xr + (i + 1) * p, Xm.begin() + t * p);
            ym[static_cast<std::size_t>(t)] = y[i];
        }
        std::vector<Cand> pool;
        for (lint g = 0; g < kGroups; ++g) {
            const double* Xg = Xm.data() + g * kNGroup * p;
            Design Dg{kNGroup, p, p, false, Dk.rtol, reinterpret_cast<const std::byte*>(Xg), static_cast<std::int64_t>(p * 8),
                      8};
            const double Tg = (static_cast<double>(kNGroup) - static_cast<double>(p)) * kSB;
            std::vector<Cand> bg = best_candidates(Dg, Xg, ym.data() + g * kNGroup, key, g * nres, nres, Tg);
            for (auto& c : bg) pool.push_back(std::move(c));
        }
        // one local step on the merged rows from each group's scale; the best kBestR by the
        // merged rows' M-scale (the earlier on a tie)
        Design Dm{m, p, p, false, Dk.rtol, reinterpret_cast<const std::byte*>(Xm.data()), static_cast<std::int64_t>(p * 8),
                  8};
        const double Tm = (static_cast<double>(m) - static_cast<double>(p)) * kSB;
        Buffers Bm(m, p);
        std::vector<double> rm(static_cast<std::size_t>(m));
        for (auto& c : pool) {
            resid_plain(Xm.data(), m, p, ym.data(), c.beta.data(), rm.data());
            double s = c.scale;
            if (s > 0.0 && !s_step(Dm, ym.data(), c.beta.data(), rm.data(), s, Tm, Bm, sc0)) continue;
            const double sm = mscale(rm.data(), m, kSChi, Tm, s, false);
            if (!std::isfinite(sm)) continue;
            std::size_t wi = 0;
            if (static_cast<int>(best.size()) == kBestR) {
                for (std::size_t q = 1; q < best.size(); ++q)
                    if (best[q].scale > best[wi].scale) wi = q;
                if (!(sm < best[wi].scale)) continue;
                best[wi] = Cand{c.beta, sm};
            } else {
                best.push_back(Cand{c.beta, sm});
            }
        }
    } else {
        best = best_candidates(Dk, Xr, y, key, 0, nres, T);
    }
    if (best.empty()) return out;
    // 3. the kept ones refined to convergence (|d beta| < refine.tol max(refine.tol, |beta|)),
    // in the order kept; the smallest final scale wins (the earlier on a tie). A scale whose
    // rows are 0 to rounding is 0 (zero_scale), here and in 4.
    const lint needS = zero_need(sMScale, T, n);
    const ZeroTol ztS = zero_tol(Dk, y);
    Buffers B(n, p);
    double bestscale = kInf;
    std::vector<double> bb(static_cast<std::size_t>(p)), cur(static_cast<std::size_t>(p)), prev(cur.size());
    bool conv_best = false;
    for (std::size_t q = 0; q < best.size(); ++q) {
        std::copy(best[q].beta.begin(), best[q].beta.end(), cur.begin());
        double s = best[q].scale;
        resid_plain(Xr, n, p, y, cur.data(), r.data());
        bool conv = false;
        int it = 0;
        while (it < kKMax && s > 0.0) {
            std::copy(cur.begin(), cur.end(), prev.begin());
            if (!s_step(Dk, y, cur.data(), r.data(), s, T, B, sc0)) break;
            ++it;
            if (zero_scale(Dk, ztS, y, cur.data(), r.data(), needS)) {
                s = 0.0;  // step 4 finds it again from the chosen one's residuals
                break;
            }
            double d2 = 0.0, b2 = 0.0;
            for (lint c = 0; c < p; ++c) d2 += (cur[c] - prev[c]) * (cur[c] - prev[c]), b2 += prev[c] * prev[c];
            if (std::sqrt(d2) < kRefineTol * std::max(kRefineTol, std::sqrt(b2))) {
                conv = true;
                break;
            }
        }
        if (s == 0.0) conv = true;
        if (s < bestscale) bestscale = s, bb = cur, conv_best = conv;
    }
    if (!std::isfinite(bestscale)) return out;
    // 4. the chosen one polished to the S estimator's fixed point (beta the weighted fit at
    // the chi weights of its residuals over their exact M-scale, i.e. the root of the
    // estimating equations F = sum psi(r_i/s(beta)) x_i, psi ~ rho~'): each iteration tries
    // Newton's step for F (the M step's, with the M-scale's dependence on beta), taken when it
    // lowers |F| and keeps the scale (scale_kept; G82); else the weighted fit by the normal
    // equations. The fixed point is the
    // same; the weighted fit alone converges linearly (rate 0.8-0.9). The last solve is the
    // refined QR's.
    std::vector<std::uint8_t> kept(static_cast<std::size_t>(p), 1);
    resid_dd(Dk, y, bb.data(), r.data(), nullptr);
    double s = mscale(r.data(), n, kSChi, T, bestscale, true);
    double prevstep = kInf;
    const NormF nfS{nBisquare, 0.0, 0.0, kSChi};
    const ScaleRule srS{sMScale, T};
    std::vector<lint> Kall(static_cast<std::size_t>(p));
    std::iota(Kall.begin(), Kall.end(), lint(0));
    std::vector<double> F, Fn, rn(static_cast<std::size_t>(std::max<lint>(n, 1)));
    bool snapped = false;
    bool fixed_point = !(s > 0.0);  // the polish reached its end (not kKMax)
    for (int it = 0; it < kKMax && s > 0.0; ++it) {
        if (zero_scale(Dk, ztS, y, bb.data(), r.data(), needS)) {
            s = 0.0, snapped = true, fixed_point = true;
            break;
        }
        bool took_newton = false;
        const double f0 = estimating(Dk, nfS, r.data(), s, Kall, F);
        if (newton_step(Dk, nfS, srS, bb.data(), r.data(), s, Kall, F, B.beta_new.data())) {
            resid_dd(Dk, y, B.beta_new.data(), rn.data(), nullptr);
            const double sn = mscale(rn.data(), n, kSChi, T, s, true);
            if (scale_kept(sn, s) && estimating(Dk, nfS, rn.data(), sn, Kall, Fn) < f0) {
                std::copy(rn.begin(), rn.begin() + n, r.begin());
                s = sn;
                took_newton = true;
            }
        }
        if (!took_newton) {
            for (lint i = 0; i < n; ++i) B.w[i] = wchi_s(r[i] / s, kSChi);
            if (!chol_wls(Dk, y, B.w.data(), kept.data(), B.beta_new.data(), B.kept_new.data(), B.fit.data()))
                if (qr_wls(Dk, y, B.w.data(), B.beta_new.data(), B.kept_new.data(), B.fit.data(), 0, sc0) < p) break;
        }
        double step = 0.0, bn = 0.0;
        for (lint c = 0; c < p; ++c)
            step = std::max(step, std::fabs(B.beta_new[c] - bb[c])), bn = std::max(bn, std::fabs(B.beta_new[c]));
        std::copy(B.beta_new.begin(), B.beta_new.begin() + p, bb.begin());
        if (!took_newton) {
            for (lint i = 0; i < n; ++i) r[i] = y[i] - B.fit[i];
            s = mscale(r.data(), n, kSChi, T, s, true);
        }
        const double lim = 0x1p-46 * bn;  // the normal equations' floor; the refined solve follows
        if (step <= lim) {
            fixed_point = true;
            break;
        }
        if (prevstep < kInf) {
            const double q = step / prevstep;
            if (q >= 1.0 || step * q / (1.0 - q) <= lim) {
                fixed_point = true;
                break;
            }
        }
        prevstep = step;
    }
    if (s > 0.0) {
        for (lint i = 0; i < n; ++i) B.w[i] = wchi_s(r[i] / s, kSChi);
        if (qr_wls(Dk, y, B.w.data(), B.beta_new.data(), B.kept_new.data(), B.fit.data(), kRefineSteps, sc0) == p)
            std::copy(B.beta_new.begin(), B.beta_new.begin() + p, bb.begin());
        resid_dd(Dk, y, bb.data(), r.data(), nullptr);
        s = mscale(r.data(), n, kSChi, T, s, true);
        if (s > 0.0 && zero_scale(Dk, ztS, y, bb.data(), r.data(), needS)) s = 0.0, snapped = true;
    }
    if (snapped) exact_fit(Dk, y, bb.data(), kept.data(), r.data(), B, sc0);
    out.beta = bb;
    out.scale = s;
    // the chosen candidate's refinement (robustbase's 1e-7) and the polish to the fixed point
    // both finished (decision 1, G93)
    out.converged = conv_best && fixed_point;
    out.ok = true;
    return out;
}

// ---- covariances -------------------------------------------------------------------------

// Sums over rows of sum_i f_i x_i x_i^T (k x k, row-major, columns K) and sum_i g_i x_i (k)
// in chunks of rows added in order.
// With Rk (kk x kk upper triangular, row-major: the least squares start's R of the columns
// K), each row is first mapped to q = Rk^-T x (the thin Q's row), so the sums are those of
// Q^T diag(f) Q, whose condition does not carry kappa(X)^2.
void gram_sums(const Design& D, const std::vector<lint>& K, const std::vector<double>& f, const std::vector<double>* g,
               std::vector<double>& G, std::vector<double>& v, const double* Rk = nullptr) {
    const lint kk = static_cast<lint>(K.size()), n = D.n;
    const lint nc = row_chunks(n);
    const std::size_t slot = static_cast<std::size_t>(kk * kk + kk);
    std::vector<double> part(static_cast<std::size_t>(nc) * slot, 0.0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        double* P = part.data() + static_cast<std::size_t>(c) * slot;
        std::vector<double> x(static_cast<std::size_t>(kk));
        for (lint i = i0; i < i1; ++i) {
            for (lint a = 0; a < kk; ++a) x[a] = D.at(i, K[a]);
            if (Rk)
                for (lint a = 0; a < kk; ++a) {  // forward substitution with Rk^T
                    double t = x[a];
                    for (lint b = 0; b < a; ++b) t -= Rk[b * kk + a] * x[b];
                    x[a] = t / Rk[a * kk + a];
                }
            const double fi = f[i];
            for (lint a = 0; a < kk; ++a)
                for (lint b = 0; b < kk; ++b) P[a * kk + b] += fi * x[a] * x[b];
            if (g)
                for (lint a = 0; a < kk; ++a) P[kk * kk + a] += (*g)[i] * x[a];
        }
    });
    G.assign(static_cast<std::size_t>(kk * kk), 0.0);
    v.assign(static_cast<std::size_t>(kk), 0.0);
    for (lint c = 0; c < nc; ++c) {
        const double* P = part.data() + static_cast<std::size_t>(c) * slot;
        for (lint e = 0; e < kk * kk; ++e) G[e] += P[e];
        for (lint a = 0; a < kk; ++a) v[a] += P[kk * kk + a];
    }
}

// robustbase's .vcov.avar1 for MM (lmrob.control()'s default cov): r the final residuals,
// r0 the S residuals, s the S scale; V (kk x kk row-major) over the columns K.
// Rk: the least squares start's R over K (row-major), or null: the sums are then formed in
// the coordinates of the thin Q and mapped back, V = Rk^-1 V_Q Rk^-T (accuracy kappa(X) u
// rather than kappa(X)^2 u).
void avar1(const Design& D, const std::vector<lint>& K, const double* r, const double* r0, double s, double cpsi,
           const double* Rk, double* V, Scratch& sc) {
    const lint n = D.n, kk = static_cast<lint>(K.size());
    if (kk == 0) return;  // V stays as the caller filled it (NaN)
    NormF bis{nBisquare, 0.0, 0.0, cpsi};
    std::vector<double> f1(static_cast<std::size_t>(n)), f2(f1.size()), g1(f1.size()), g2(f1.size());
    for (lint i = 0; i < n; ++i) {
        const double z = r[i] / s, z0 = r0[i] / s;
        const double dp = bis.dpsi(z), ps = bis.psi(z), ch = chi_s(z0, kSChi);
        f1[i] = dp;
        g1[i] = dp * z;
        f2[i] = ps * ps;
        g2[i] = ps * ch;
    }
    const DD m0 = row_sum(n, [&](lint i) { return dchi_s(r0[i] / s, kSChi) * (r0[i] / s); });
    const DD m4 = row_sum(n, [&](lint i) {
        const double ch = chi_s(r0[i] / s, kSChi);
        return ch * ch - kSB * kSB;
    });
    const double dn = static_cast<double>(n);
    const double mw0 = (m0.hi + m0.lo) / dn, mu4 = (m4.hi + m4.lo) / dn;
    std::vector<double> G1, v1, G2, v2;
    gram_sums(D, K, f1, &g1, G1, v1, Rk);
    gram_sums(D, K, f2, &g2, G2, v2, Rk);
    // A = s G1^-1
    std::vector<double> A(static_cast<std::size_t>(kk * kk), 0.0), L(G1);
    std::vector<lint> piv(static_cast<std::size_t>(kk));
    for (lint a = 0; a < kk; ++a) A[a * kk + a] = s;
    if (linalg::lapack::getrf(kk, kk, L.data(), kk, piv.data()) != 0) {
        std::fill(V, V + kk * kk, kNaN);
        return;
    }
    linalg::lapack::getrs('N', kk, kk, L.data(), kk, piv.data(), A.data(), kk);  // symmetric: layout-free
    std::vector<double> a(static_cast<std::size_t>(kk), 0.0), Av2(a.size(), 0.0), tmp(A.size(), 0.0);
    for (lint i = 0; i < kk; ++i)
        for (lint j = 0; j < kk; ++j) a[i] += A[i * kk + j] * v1[j], Av2[i] += A[i * kk + j] * v2[j];
    for (lint i = 0; i < kk; ++i) a[i] /= mw0;
    // u1 / n = A G2 A
    for (lint i = 0; i < kk; ++i)
        for (lint j = 0; j < kk; ++j) {
            double acc = 0.0;
            for (lint l = 0; l < kk; ++l) acc += A[i * kk + l] * G2[l * kk + j];
            tmp[i * kk + j] = acc;
        }
    for (lint i = 0; i < kk; ++i)
        for (lint j = 0; j < kk; ++j) {
            double acc = 0.0;
            for (lint l = 0; l < kk; ++l) acc += tmp[i * kk + l] * A[l * kk + j];
            // u2 = a (A^T v2)^T, u3 = (A v2) a^T, u4 = mean(chi^2 - b^2) a a^T
            V[i * kk + j] = acc + (-(a[i] * Av2[j]) - Av2[i] * a[j] + mu4 * a[i] * a[j]) / dn;
        }
    if (Rk) {  // back to the coefficients: V = Rk^-1 V_Q Rk^-T (two triangular solves)
        for (int pass = 0; pass < 2; ++pass) {
            for (lint col = 0; col < kk; ++col)  // V := Rk^-1 V, column by column
                for (lint a = kk - 1; a >= 0; --a) {
                    double t = V[a * kk + col];
                    for (lint b = a + 1; b < kk; ++b) t -= Rk[a * kk + b] * V[b * kk + col];
                    V[a * kk + col] = t / Rk[a * kk + a];
                }
            for (lint a = 0; a < kk; ++a)  // transpose: the second pass gives (Rk^-1 (Rk^-1 V_Q)^T)^T
                for (lint b = 0; b < a; ++b) std::swap(V[a * kk + b], V[b * kk + a]);
        }
    }
    for (lint i = 0; i < kk; ++i)  // symmetric to rounding: make it exactly so
        for (lint j = 0; j < i; ++j) {
            const double m = 0.5 * (V[i * kk + j] + V[j * kk + i]);
            V[i * kk + j] = V[j * kk + i] = m;
        }
    // robustbase's posdefify: negative eigenvalues set to 0, the diagonal rescaled back
    std::vector<double> E(V, V + kk * kk), ev(static_cast<std::size_t>(kk));
    if (linalg::lapack::syevd('V', kk, E.data(), kk, ev.data(), sc) != 0) return;
    bool neg = false;
    for (double e : ev) neg = neg || e < 0.0;
    if (!neg) return;
    std::vector<double> od(static_cast<std::size_t>(kk));
    for (lint i = 0; i < kk; ++i) od[i] = V[i * kk + i];
    for (lint i = 0; i < kk; ++i)
        for (lint j = 0; j < kk; ++j) {
            double acc = 0.0;
            for (lint l = 0; l < kk; ++l)  // E column-major: eigenvector l is E[l*kk + .]
                if (ev[l] > 0.0) acc += E[l * kk + i] * ev[l] * E[l * kk + j];
            V[i * kk + j] = acc;
        }
    std::vector<double> dsc(static_cast<std::size_t>(kk));
    for (lint i = 0; i < kk; ++i) dsc[i] = std::sqrt(std::max(0.0, od[i]) / V[i * kk + i]);
    for (lint i = 0; i < kk; ++i)
        for (lint j = 0; j < kk; ++j) V[i * kk + j] *= dsc[i] * dsc[j];
}

// ---- the regression ----------------------------------------------------------------------

struct Opts {
    int method, norm, scale_est, max_iter, nres;
    double c0, c1, c2, tol, rtol;
    bool intercept;
    bool large_n = true;  // G58; false (only the benchmarks) keeps the small-n search for every n
};

struct Outs {
    double *params, *cov, *R, *fitted, *resid, *stats, *weights, *init;
    std::uint8_t* kept;
};

void fill_nan(lint n, lint k, const Outs& o) {
    std::fill(o.params, o.params + k, kNaN);
    std::fill(o.cov, o.cov + k * k, kNaN);
    std::fill(o.R, o.R + k * k, kNaN);
    std::fill(o.fitted, o.fitted + n, kNaN);
    std::fill(o.resid, o.resid + n, kNaN);
    std::fill(o.weights, o.weights + n, kNaN);
    std::fill(o.init, o.init + k, kNaN);
    std::fill(o.kept, o.kept + k, std::uint8_t(0));
    std::fill(o.stats, o.stats + rNStat, kNaN);
    o.stats[rRank] = 0;
    o.stats[rIsNaN] = 1;
}

int rlm_one(const Opts& op, lint n, lint p, const std::byte* yb, std::int64_t ys, const std::byte* Xb, std::int64_t xrs,
            std::int64_t xcs, const std::byte* keyb, std::int64_t keys, const Outs& o, Scratch& sc) {
    const lint k = p + (op.intercept ? 1 : 0);
    // ---- 1. checks and the least squares start
    bool nan = false;
    std::vector<double> y(static_cast<std::size_t>(std::max<lint>(n, 1)));
    for (lint i = 0; i < n; ++i) {
        y[i] = rd(yb + i * ys);
        if (std::isnan(y[i])) nan = true;
        else if (std::isinf(y[i])) return eInfY;
    }
    Data d0{reinterpret_cast<const std::byte*>(y.data()), 8, Xb, xrs, xcs, nullptr, 0, nullptr, 0, nullptr, 0};
    if (scan_x(d0, n, p, nan)) return eInfX;
    if (nan) {
        fill_nan(n, k, o);
        return 0;
    }
    if (n == 0) {  // no row: a NaN result with nobs 0 (G96)
        fill_nan(n, k, o);
        o.stats[rNobs] = 0.0;
        o.stats[rNIter] = 0.0;
        o.stats[rConverged] = 0.0;
        return 0;
    }
    Design D{n, p, k, op.intercept, op.rtol, Xb, xrs, xcs};
    std::vector<double> cov0(static_cast<std::size_t>(std::max<lint>(k * k, 1))), r(static_cast<std::size_t>(std::max<lint>(n, 1)));
    std::vector<double> beta(static_cast<std::size_t>(std::max<lint>(k, 1)));
    std::vector<std::uint8_t> kept(beta.size());
    double st0[kNStat];
    {
        Problem pb{n, p, k, op.intercept, false, false, false, op.rtol, kNonrobust, kRefineSteps, -1, 0, false};
        pb.checked = true;  // y and X were scanned above
        Dest od{beta.data(), cov0.data(), o.R, o.fitted, r.data(), kept.data(), st0, nullptr};
        fit_one(pb, d0, od, sc);
    }
    const lint rank = static_cast<lint>(st0[kRank]);
    std::vector<lint> K;
    for (lint c = 0; c < k; ++c)
        if (kept[c]) K.push_back(c);
    const double dn = static_cast<double>(n), df = dn - static_cast<double>(rank);
    std::fill(o.stats, o.stats + rNStat, kNaN);
    o.stats[rNobs] = dn;
    o.stats[rIsNaN] = 0;
    std::fill(o.init, o.init + k, kNaN);
    resid_dd(D, y.data(), beta.data(), r.data(), nullptr);
    Buffers B(n, k);
    double s = 0.0;
    IrlsOut io;
    NormF nf{op.norm, op.c0, op.c1, op.c2};
    std::vector<double> r0;  // MM: the S residuals
    // an exact start: y constant with the intercept (the exact solution (y0, 0, ..., 0), G94;
    // the least squares leave rounding there, a slope of 1e-33 and a t of -inf), or no
    // degree of freedom left (n = rank: the least squares fit is exact). The perfect fit's
    // rule (§5): scale 0, no iteration, the start's coefficients (MM's S coefficients)
    bool exact = rank > 0 && n == rank;
    if (op.intercept && kept[0] && rank > 0) {
        bool cy = true;
        for (lint i = 1; i < n && cy; ++i) cy = y[i] == y[0];
        if (cy) {
            exact = true;
            beta[0] = y[0];
            for (lint c = 1; c < k; ++c) beta[c] = kept[c] ? 0.0 : kNaN;
            resid_dd(D, y.data(), beta.data(), r.data(), nullptr);
        }
    }
    if (exact || rank == 0) {
        if (exact) {
            s = 0.0;
        } else if (op.method == mM) {  // no column: the residuals' scale, nothing to iterate
            const double ech = kHuberD * kHuberD + (1.0 - kHuberD * kHuberD) * (0.5 * std::erfc(-kHuberD / std::sqrt(2.0))) -
                               0.5 - kHuberD / std::sqrt(2.0 * M_PI) * std::exp(-0.5 * kHuberD * kHuberD);
            s = op.scale_est == sHuber ? huber_scale(r.data(), n, df * ech) : mad0(r.data(), n, B.buf.data());
        } else {  // MM without a column: the S scale of y itself
            s = mscale(r.data(), n, kSChi, dn * kSB, kRbMad * mad0(r.data(), n, B.buf.data()) * kMadC, true);
        }
        io.iters = 0;
        io.converged = true;
        if (op.method == mMM) {
            r0 = r;
            for (lint c = 0; c < k; ++c) o.init[c] = beta[c];
        }
    } else if (op.method == mM) {
        // ---- 2. M
        ScaleRule sr{op.scale_est, 0.0};
        if (op.scale_est == sHuber) {
            const double dd = kHuberD;
            const double ech = dd * dd + (1.0 - dd * dd) * (0.5 * std::erfc(-dd / std::sqrt(2.0))) - 0.5 -
                               dd / std::sqrt(2.0 * M_PI) * std::exp(-0.5 * dd * dd);
            sr.T = dn * (df / dn * ech);
        }
        s = sr.mode == sMad ? mad0(r.data(), n, B.buf.data()) : huber_scale(r.data(), n, sr.T);
        io = irls(D, y.data(), nf, sr, op.max_iter, op.tol, beta.data(), kept.data(), r.data(), s, B, sc);
    } else {
        // ---- 3. MM: fast-S on the kept columns (packed row-major), then the M step
        const lint pk = rank;
        std::vector<double> Xr(static_cast<std::size_t>(std::max<lint>(n * pk, 1)));
        for (lint i = 0; i < n; ++i)
            for (lint a = 0; a < pk; ++a) Xr[i * pk + a] = D.at(i, K[a]);
        Design Dk{n, pk, pk, false, op.rtol, reinterpret_cast<const std::byte*>(Xr.data()),
                  static_cast<std::int64_t>(pk * 8), 8};
        std::uint64_t k0, k1;
        std::memcpy(&k0, keyb, 8);
        std::memcpy(&k1, keyb + keys, 8);
        SOut so = fast_s(Dk, Xr.data(), y.data(), random::Key{k0, k1}, op.nres, op.large_n, sc);
        if (!so.ok) {
            fill_nan(n, k, o);
            o.stats[rIsNaN] = 0;
            o.stats[rNobs] = dn;
            o.stats[rRank] = static_cast<double>(rank);
            o.stats[rConverged] = 0;
            o.stats[rNIter] = 0;
            return 0;
        }
        const double sS = so.scale;
        std::vector<double> bk(so.beta), rk(static_cast<std::size_t>(std::max<lint>(n, 1)));
        std::vector<std::uint8_t> kk(static_cast<std::size_t>(std::max<lint>(pk, 1)), 1);
        resid_dd(Dk, y.data(), bk.data(), rk.data(), nullptr);
        r0 = rk;
        for (lint a = 0; a < pk; ++a) o.init[K[a]] = bk[a];
        Buffers Bk(n, pk);
        s = sS;
        ScaleRule sr{sFixed, 0.0};
        io = irls(Dk, y.data(), nf, sr, op.max_iter, op.tol, bk.data(), kk.data(), rk.data(), s, Bk, sc);
        std::fill(beta.begin(), beta.end(), kNaN);
        std::fill(kept.begin(), kept.end(), std::uint8_t(0));
        for (lint a = 0; a < pk; ++a)
            if (kk[a]) beta[K[a]] = bk[a], kept[K[a]] = 1;
        // the S estimate's refinement to its fixed point is part of the fit: one cut short
        // (its iteration limit) leaves MM not converged (decision 1, G93)
        io.converged = io.converged && so.converged;
        s = sS;
    }
    // ---- outputs: the fit (Dot2), the weights, the covariance
    resid_dd(D, y.data(), beta.data(), o.resid, o.fitted);
    lint rk = 0;
    for (lint c = 0; c < k; ++c) rk += kept[c] != 0;
    for (lint c = 0; c < k; ++c) o.params[c] = beta[c], o.kept[c] = kept[c];
    std::vector<double> z(static_cast<std::size_t>(std::max<lint>(n, 1)));
    const double bn = beta_inf(beta.data(), k);
    for (lint i = 0; i < n; ++i) {
        // an exact fit (s = 0): the rows whose residual is 0 to rounding are on the fit
        const bool on = s == 0.0 && std::fabs(o.resid[i]) <= zero_tol_row(D, y.data(), i, bn);
        z[i] = on ? 0.0 : std_resid(o.resid[i], s);
        o.weights[i] = nf.weight(z[i]);
    }
    std::fill(o.cov, o.cov + k * k, kNaN);
    if (s == 0.0) {
        // an exact fit of most rows: the covariance's limit 0 (statsmodels takes the scaled
        // residuals as 0)
        for (lint a = 0; a < k; ++a)
            for (lint b = 0; b < k; ++b)
                if (kept[a] && kept[b]) o.cov[a * k + b] = 0.0;
    } else if (op.method == mM) {
        // Huber's H1 (statsmodels' RLMResults.bcov_scaled, cov='H1')
        const DD sd1 = row_sum(n, [&](lint i) { return nf.dpsi(z[i]); });
        const double m = (sd1.hi + sd1.lo) / dn;
        const DD sv = row_sum(n, [&](lint i) {
            const double e = nf.dpsi(z[i]) - m;
            return e * e;
        });
        const double var = (sv.hi + sv.lo) / dn;
        const DD ss = row_sum(n, [&](lint i) {
            const double q = nf.psi(z[i]);
            return q * q;
        });
        const double kf = 1.0 + static_cast<double>(rk) / dn * var / (m * m);
        const double f = kf * kf * ((ss.hi + ss.lo) / df * s * s) / (m * m);
        for (lint a = 0; a < k; ++a)
            for (lint b = 0; b < k; ++b)
                if (kept[a] && kept[b]) o.cov[a * k + b] = f * cov0[a * k + b];
    } else {
        std::vector<lint> Kf;
        for (lint c = 0; c < k; ++c)
            if (kept[c]) Kf.push_back(c);
        const lint kf = static_cast<lint>(Kf.size());
        std::vector<double> V(static_cast<std::size_t>(std::max<lint>(kf * kf, 1)));
        // the start's R when the M step kept the start's columns (always but for weights that
        // leave a column without rows)
        std::vector<double> Rk;
        if (Kf == K) {
            Rk.resize(static_cast<std::size_t>(kf * kf));
            for (lint a = 0; a < kf; ++a)
                for (lint b = 0; b < kf; ++b) Rk[a * kf + b] = b >= a ? o.R[K[a] * k + K[b]] : 0.0;
            for (lint a = 0; a < kf; ++a)
                if (!(Rk[a * kf + a] != 0.0 && std::isfinite(Rk[a * kf + a]))) Rk.clear();
        }
        avar1(D, Kf, o.resid, r0.data(), s, op.c2, Rk.empty() ? nullptr : Rk.data(), V.data(), sc);
        for (lint a = 0; a < kf; ++a)
            for (lint b = 0; b < kf; ++b) o.cov[Kf[a] * k + Kf[b]] = V[a * kf + b];
    }
    o.stats[rRank] = static_cast<double>(rk);
    o.stats[rScale] = s;
    o.stats[rNIter] = io.iters;
    o.stats[rConverged] = io.converged ? 1.0 : 0.0;
    return 0;
}

std::vector<std::int64_t> with(std::vector<std::int64_t> shape, std::initializer_list<std::int64_t> tail) {
    shape.insert(shape.end(), tail);
    return shape;
}

// rlm_fit(func, y (..., n), X (..., n, p), keys (..., 2) int64 | None, intercept, rtol, method, norm,
//         c0, c1, c2, scale_est, max_iter, tol, nresample)
// -> (params, cov, R, fitted, resid, kept, stats, weights, init_params). cov is the final
// covariance (H1 or avar1); R is the least squares start's; init_params the S estimate (MM).
nb::object rlm_fit(const std::string& func, nb::handle yo, nb::handle xo, nb::handle ko, bool intercept, double rtol,
                   int method, int norm, double c0, double c1, double c2, int scale_est, int max_iter, double tol,
                   int nres, bool large_n) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X");
        const bool has_k = !ko.is_none();
        std::optional<Input> key;
        if (has_k) key.emplace(ko, fn, "keys");
        if (method == mMM && !has_k) throw nb::value_error("rlm_fit(): MM needs keys");
        const linalg::BatchOperand none{};
        Batch b({{y.batch_shape(1), y.batch_strides(1)},
                 {X.batch_shape(2), X.batch_strides(2)},
                 has_k ? linalg::BatchOperand{key->batch_shape(1), key->batch_strides(1)} : none},
                fn);
        const lint n = X.shape(-2), p = X.shape(-1), k = p + (intercept ? 1 : 0);
        Opts op{method, norm, scale_est, max_iter, nres, c0, c1, c2, tol, rtol, intercept, large_n};
        Out po = linalg::alloc_c(with(b.shape(), {k}), DType::Float64);
        Out co = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out ro = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out fo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out eo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out kpo = linalg::alloc_c(with(b.shape(), {k}), DType::Bool);
        Out so = linalg::alloc_c(with(b.shape(), {rNStat}), DType::Float64);
        Out wo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out io = linalg::alloc_c(with(b.shape(), {k}), DType::Float64);
        const double nk = double(n) * double(std::max<lint>(k, 1));
        double element_ns = 2000.0 + nk * (double(k) + 20.0) * 20.0;
        if (method == mMM) element_ns += double(nres) * nk * 30.0;
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            Outs o{po.as<double>() + i * k, co.as<double>() + i * k * k, ro.as<double>() + i * k * k,
                   fo.as<double>() + i * n, eo.as<double>() + i * n, so.as<double>() + i * rNStat,
                   wo.as<double>() + i * n, io.as<double>() + i * k, kpo.as<std::uint8_t>() + i * k};
            return rlm_one(op, n, p, y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2), X.stride(-1),
                           has_k ? key->data() + off[2] : nullptr, has_k ? key->stride(-1) : 0, o, sc);
        });
        if (f) {
            const char* what = f.code == eInfY ? "'y' contains infinity" : "'X' contains infinity";
            throw nb::value_error(linalg::failure_message(b, f.index, (std::string(fn) + "(): " + what).c_str()).c_str());
        }
        return nb::make_tuple(linalg::to_array(po), linalg::to_array(co), linalg::to_array(ro), linalg::to_array(fo),
                              linalg::to_array(eo), linalg::to_array(kpo), linalg::to_array(so), linalg::to_array(wo),
                              linalg::to_array(io));
    });
}

}  // namespace

void bind_robust(nb::module_& m) {
    m.def("rlm_fit", &rlm_fit, nb::arg("func"), nb::arg("y").none(), nb::arg("X").none(), nb::arg("keys").none(),
          nb::arg("intercept"), nb::arg("rtol"), nb::arg("method"), nb::arg("norm"), nb::arg("c0"), nb::arg("c1"),
          nb::arg("c2"), nb::arg("scale_est"), nb::arg("max_iter"), nb::arg("tol"), nb::arg("nresample"),
          nb::arg("large_n") = true);
    bind_slopes(m);
}

}  // namespace sd::regression
