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

namespace sd::linalg {

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
inline constexpr Family kCopy{2, 0.5};         // diagonal, outer, kron, cross: w = elements copied
inline constexpr Family kExpm{600, 15};        // expm (M10, experiments/m10_expm/out/percost.txt)

inline double cost(Family f, double w) { return f.a + f.c * w; }

// gbsv has a fixed part of about 1 us per call that does not get faster on
// more threads (it slowed down 1.1-2.7x when split, for any small band). Split a
// gbsv batch only when n * (kl + ku + 1) reaches this, where the per-band work
// wins (experiments/m9_threshold/out/band.txt: the crossover lies between n = 64
// and 96 for kl = ku = 2, between n = 32 and 48 for kl = ku = 5).
inline constexpr double kBandSplitMin = 400;

// Accelerate runs two paths of small LAPACK calls through the AMX unit that
// the cores share, so they do not scale and get slower beyond a few threads
// (after M11: experiments/opt_impl/out/lu_scaling.txt, lu_scaling_k.txt;
// experiments/opt_analysis/out/lapack_scaling.txt):
//   - getrf for n = 32-40 (best 5 threads; n <= 31 and n >= 48 scale); capped
//     only to n = 35: beyond, lu's own unpacking still scales and the cap cost
//     it 5-12% (out/cap_check.txt);
//   - the triangular solves (trsm, inside getrs/trtrs/potrs) with k >= 8
//     right-hand sides for n = 24-40 (best 3 threads to n = 34, 5 to 40; k <= 7
//     and n <= 23 scale).
// lu_task_cap(n, k) is the cap on tasks for a batch that factors with getrf
// (`factors`) and/or solves k right-hand sides; 0 = no cap.
inline bool kLuTaskCapOn = true;  // a variable only for the experiment (_core._set_lu_task_cap)
inline int lu_task_cap(std::int64_t n, std::int64_t k, bool factors) {
    if (!kLuTaskCapOn) return 0;
    int cap = 0;
    auto tighten = [&](int c) { cap = cap == 0 ? c : (c < cap ? c : cap); };
    if (factors && n >= 32 && n <= 35) tighten(5);
    if (k >= 8 && n >= 24 && n <= 34) tighten(3);
    if (k >= 8 && n >= 35 && n <= 40) tighten(5);
    return cap;
}

}  // namespace sd::linalg
