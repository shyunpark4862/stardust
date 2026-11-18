// The least squares core of the regression area, shared by the linear models
// (regression/linear.cpp) and the IRLS of the generalized linear models
// (regression/glm.cpp): one weighted least squares fit with the QR, the aliased
// columns, the refinement and the covariances of docs/design/regression.md §3.1-§3.3.
#pragma once

#include "linalg/ops/common.hpp"

#include <cstddef>
#include <cstdint>

namespace sd::regression::lsq {

using linalg::lint;
using linalg::Scratch;

// cov_type codes (the Python layer maps the names)
enum Cov : int { kNonrobust = 0, kHC0 = 1, kHC1 = 2, kHC2 = 3, kHC3 = 4, kHAC = 5, kCluster = 6 };

// The per-regression scalars, in this order (stardust/regression/_linear.py reads them).
// kNGroups: the clusters among the fit's rows (cluster), kLags: the lags used (HAC); NaN
// for the other covariances.
enum Stat : int {
    kNobs = 0, kRank, kSsr, kCtss, kUtss, kEssC, kEssU, kSumLogW, kWaldF, kWaldQ, kIsNaN, kNGroups, kLags,
    kNStat
};

// Failure codes of one regression.
constexpr int kInfY = 1, kInfX = 2, kInfW = 3, kNegW = 4, kInfU = 5;

struct Problem {
    lint n, p, k;   // rows, columns of X, columns of the design (p + intercept)
    bool intercept, has_w, has_u, has_g;
    double rtol;    // < 0: max(nobs, k) * eps
    int cov;
    int steps;
    std::int64_t maxlags;  // HAC; < 0: floor(4 (nobs / 100)^(2/9))
    std::int64_t gtot;     // cluster: the codes are 0 .. gtot - 1 (-1: missing)
    bool hat_only;
    // An inner solve of the IRLS (regression/glm.cpp): the inputs are known finite, and only
    // params, kept, the fitted values of every row, stats[kNobs], stats[kRank] and (when Dest
    // has it) R are written (no sums of squares or covariance).
    bool lean = false;
    // The GLM's final solve: no Wald F (the GLM has none), the robust options of Data.
    bool glm = false;
};

// The inputs of one regression (byte pointers and strides).
struct Data {
    const std::byte* y;
    std::int64_t ys;
    const std::byte* X;
    std::int64_t xrs, xcs;
    const std::byte* w;
    std::int64_t ws;
    const std::byte* u;
    std::int64_t us;
    const std::byte* g;  // int64 cluster codes
    std::int64_t gs;
    // GLM robust covariances (Problem::glm), per row i of the input, contiguous, or null:
    // smult: a factor on the whitened residual of the HC and HAC scores (1/sqrt(freq weight));
    // hdiff: the observed information's weight is w_i - hdiff_i (the bread becomes the
    // observed information's inverse, R^-1 (I - Q^T diag(hdiff/w) Q)^-1 R^-T).
    const double* smult = nullptr;
    const double* hdiff = nullptr;
    double nobs_eff = -1.0;  // the n of HC1's and the cluster's factors (< 0: the rows used)
    // GLM Newton step (with hdiff, G42): when given (k values in the design's column order,
    // NaN for a column without one), the solve is Newton's from these coefficients instead
    // of the least squares (Fisher's) one, started from these coefficients; refined on the
    // augmented system below with the same QR
    // (the correction's normal-equations block is R^T (I - T) R, T = Q^T diag(hdiff/w) Q).
    const double* newton_b0 = nullptr;
    // ... and the linear predictor (offset removed) at which hdiff, the weights and y were
    // formed (per row of the input): the system is X^T W r + X^T C (X beta - eta0) = 0, the
    // Newton step about the state as it was evaluated (with X b0 in place of eta0 its rounding
    // would enter the score through C, amplified by kappa^2)
    const double* newton_eta0 = nullptr;
};

// Outputs of one regression.
struct Dest {
    double* params;  // k
    double* cov;     // k x k, row-major
    double* R;       // k x k, row-major (kept rows/columns in their places)
    double* fitted;  // n
    double* resid;   // n
    std::uint8_t* kept;  // k
    double* stats;   // kNStat
    double* hat;     // n (hat_only)
    // GLM (with Data::hdiff): T = Q^T diag(hdiff/w) Q from the fit's thin Q (k x k row-major,
    // kept rows/columns in their places, 0 elsewhere), for Newton's step R^-1 (I - T)^-1 R d
    // (regression/glm.cpp); costs one orgqr
    double* T = nullptr;
    // set to 1 when the Newton step of Data::newton_b0 was taken (0: I - T is not positive
    // definite, the least squares solution stands)
    std::uint8_t* newton = nullptr;
};

// Scans X (n x p) for NaN (sets nan) and infinity (returns kInfX).
int scan_x(const Data& d, lint n, lint p, bool& nan);

// One regression. Returns 0 or a failure code.
int fit_one(const Problem& pb, const Data& d, const Dest& o, Scratch& sc);

}  // namespace sd::regression::lsq
