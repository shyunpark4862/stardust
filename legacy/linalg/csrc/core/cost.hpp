// The estimated serial cost of one batch element, in ns, for deciding whether
// a batch is split over the thread pool and into how many tasks (M9,
// DESIGN.md §2.18; user decision: fixed constants per family).
//
// One model for every family: ns ~ a + c * w, where w is the matrix's m * n
// (n * n when square) or, for the streaming functions, the elements one batch
// element touches. Fitted to the serial times of n = 2..256
// (experiments/m9_threshold/out/percost.txt): within a factor of ~2 there,
// which is all a split decision needs. For large n the model underestimates
// (the work grows like n^3), but then any batch of two already splits.
#pragma once

namespace sdla {

struct Family {
    double a, c;
};

inline constexpr Family kLU{70, 3.0};         // solve, inv, det, slogdet, lu, cholesky, ldl, matrix_power
inline constexpr Family kTriangular{30, 0.5};  // solve_triangular, cholesky_solve, lu_solve
inline constexpr Family kQR{180, 10};          // qr, qr_pivoted, hessenberg
inline constexpr Family kEigh{250, 45};        // eigh, eigvalsh (and the generalized ones)
inline constexpr Family kSVD{1000, 80};        // svd, svdvals, matrix_rank, pinv, cond, lstsq, matrix_norm 2/-2/nuc
inline constexpr Family kEig{700, 200};        // eig, eigvals, schur
inline constexpr Family kSylvester{500, 370};  // solve_sylvester, w = (m^2 + n^2) / 2
inline constexpr Family kBanded{20, 3};        // solve_tridiagonal, solve_banded, w = n * (bandwidths + 1)
inline constexpr Family kStream{12, 0.3};      // vecdot, norms, trace: w = elements summed (per output)
inline constexpr Family kCopy{2, 0.5};         // diagonal, outer: w = elements copied

inline double cost(Family f, double w) { return f.a + f.c * w; }

// gbsv has a fixed part of about 1 us per call that does not get faster on
// more threads (it slowed down 2-3x when split, for any small band). Split a
// gbsv batch only when n * (kl + ku + 1) reaches this, where the per-band work
// wins (experiments/m9_threshold/out/band.txt: crossover at n = 64 for kl = ku = 2,
// n = 48 for kl = ku = 5).
inline constexpr double kBandSplitMin = 400;

}  // namespace sdla
