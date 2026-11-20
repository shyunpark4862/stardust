// The estimated serial cost of one batch element, in ns, for deciding whether
// a batch is split over the thread pool and into how many tasks: fixed
// constants per family (docs/design/linalg.md §7.1).
//
// One model for every family: ns ~ a + c * w, where w is the matrix's m * n
// (n * n when square) or, for the streaming functions, the elements one batch
// element touches. Fitted to the serial times of n = 2..256
// (docs/design/linalg.md §7.1): within a factor of ~2 there,
// which is all a split decision needs. For large n the model underestimates
// (the work grows like n^3), but then any batch of two already splits.
#pragma once

namespace sd::linalg {

struct Family {
    double a, c;
};

inline constexpr Family kLU{70, 3.0};  // solve, inv, det, slogdet, lu, cholesky, ldl, matrix_power, cond (other ords)
inline constexpr Family kTriangular{30, 0.5};  // solve_triangular, cholesky_solve, lu_solve
inline constexpr Family kQR{180, 10};  // qr, qr_pivoted, hessenberg
inline constexpr Family kEigh{250, 45};  // eigh, eigvalsh (and the generalized ones)
inline constexpr Family kSVD{1000, 80};  // svd, svdvals, matrix_rank, pinv, cond +-2, lstsq, matrix_norm 2/-2/nuc
inline constexpr Family kEig{700, 200};  // eig, eigvals, schur
inline constexpr Family kSylvester{500, 370};  // solve_sylvester, w = (m^2 + n^2) / 2
inline constexpr Family kBanded{20, 3};  // solve_tridiagonal, solve_banded, w = n * (bandwidths + 1)
inline constexpr Family kStream{12, 0.3};  // vecdot, norms, trace: w = elements summed (per output)
inline constexpr Family kCopy{2, 0.5};  // diagonal, outer, kron, cross: w = elements copied
inline constexpr Family kExpm{600, 15};  // expm (docs/design/linalg.md §7.1)

inline double cost(Family f, double w) { return f.a + f.c * w; }

// The right-hand sides of the solve family (solve, lstsq, the triangular, banded and
// tridiagonal solves) add about this per element of the n x k right side (packing and
// substitution; 0.9-1.6 ns at n = 3-4, docs/design/linalg.md §7.1), which the matrix
// term alone misses for a small matrix with many right sides.
inline constexpr double kRhsNs = 1.0;
inline double rhs_cost(double n, double k) { return kRhsNs * n * k; }

// gbsv has a fixed part of about 1 us per call that does not get faster on
// more threads (it slowed down 1.1-2.7x when split, for any small band). Split a
// gbsv batch only when n * (kl + ku + 1) reaches this, where the per-band work
// wins (docs/design/linalg.md §7.4: the crossover lies between n = 64 and 96
// for kl = ku = 2, between n = 32 and 48 for kl = ku = 5).
inline constexpr double kBandSplitMin = 400;

// Accelerate runs two paths of small LAPACK calls through the AMX unit that
// the cores share, so they do not scale and get slower beyond a few threads
// (docs/design/linalg.md §7.5):
//   - getrf for n = 32-46: those sizes factor with getf2 instead
//     (lapack.hpp lu_in_place: 2.1-4.4x on a batch at ten threads, 1.2-1.6x on one);
//   - the triangular solves (trsm, inside getrs/trtrs/potrs) with k >= 8
//     right-hand sides for n = 24-40 (best 3 threads to n = 34, 5 to 40; k <= 7
//     and n <= 23 scale).
// lu_task_cap(n, k) is the cap on tasks for a batch that solves k right-hand
// sides of an n x n system; 0 = no cap.
inline bool kLuTaskCapOn = true;  // a variable only for the experiment (_core._set_lu_task_cap)
inline int lu_task_cap(std::int64_t n, std::int64_t k) {
    if (!kLuTaskCapOn) return 0;
    if (k >= 8 && n >= 24 && n <= 34) return 3;
    if (k >= 8 && n >= 35 && n <= 40) return 5;
    return 0;
}

}  // namespace sd::linalg
