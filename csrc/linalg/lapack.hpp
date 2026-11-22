// Thin typed wrappers over Accelerate LAPACK (docs/design/linalg.md §1.2): one
// template per routine, dispatching to the d (double) or z (complex) version.
// They return LAPACK's `info`; a negative info is a bug in our arguments.
#pragma once

#include "linalg/batch.hpp"
#include "linalg/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sd::linalg::lapack {

inline lint checked(lint info, const char* routine) {
    if (info < 0) throw std::logic_error(std::string(routine) + ": illegal argument " + std::to_string(-info));
    return info;
}

// LU with partial pivoting: A = P L U, in place. info > 0: U(info, info) is exactly zero.
template <class T>
lint getrf(lint m, lint n, T* a, lint lda, lint* ipiv) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dgetrf_(&m, &n, a, &lda, ipiv, &info);
    else zgetrf_(&m, &n, a, &lda, ipiv, &info);
    return checked(info, "getrf");
}

// The same factorization, unblocked (getf2): the factors getrf would give up to
// rounding, but not through Accelerate's blocked path (docs/design/linalg.md §7.5).
template <class T>
lint getf2(lint m, lint n, T* a, lint lda, lint* ipiv) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dgetf2_(&m, &n, a, &lda, ipiv, &info);
    else zgetf2_(&m, &n, a, &lda, ipiv, &info);
    return checked(info, "getf2");
}

// The sizes factored by getf2 instead of getrf: both sides of the matrix in
// [kGetf2Min, kGetf2Max] (docs/design/linalg.md §7.5). There Accelerate's getrf takes a
// blocked path that runs serialized on the AMX unit: getf2 was 2.1-4.4x faster on a
// batch at ten threads (1.2-1.6x on one), and slower from n = 48 on.
inline constexpr lint kGetf2Min = 32, kGetf2Max = 46;
inline bool use_getf2(lint m, lint n) { return m >= kGetf2Min && m <= kGetf2Max && n >= kGetf2Min && n <= kGetf2Max; }

// LU with partial pivoting by the routine the shape selects (getrf or getf2).
template <class T>
lint lu_in_place(lint m, lint n, T* a, lint lda, lint* ipiv) {
    return use_getf2(m, n) ? getf2(m, n, a, lda, ipiv) : getrf(m, n, a, lda, ipiv);
}

// Solves A X = B (trans 'N') with the factors from getrf; B is overwritten by X.
template <class T>
lint getrs(char trans, lint n, lint nrhs, const T* a, lint lda, const lint* ipiv, T* b, lint ldb) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    else zgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return checked(info, "getrs");
}

// Cholesky A = L L^H (uplo 'L') in place; reads only that triangle. info > 0: not positive definite.
template <class T>
lint potrf(char uplo, lint n, T* a, lint lda) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dpotrf_(&uplo, &n, a, &lda, &info);
    else zpotrf_(&uplo, &n, a, &lda, &info);
    return checked(info, "potrf");
}

// Solves A X = B with A = L L^H (or U^H U) from potrf; B is overwritten. No singularity check.
template <class T>
lint potrs(char uplo, lint n, lint nrhs, const T* a, lint lda, T* b, lint ldb) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    else zpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return checked(info, "potrs");
}

// Solves a triangular system; reads only triangle `uplo` (and the diagonal unless diag 'U').
// info > 0: a zero on the diagonal.
template <class T>
lint trtrs(char uplo, char trans, char diag, lint n, lint nrhs, const T* a, lint lda, T* b, lint ldb) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    else ztrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return checked(info, "trtrs");
}

// Solves op(A) X = B, A triangular n x n, B n x k column-major (B overwritten), with no
// singularity check (the callers look at the diagonal first). A is column-major, or
// row-major (`row_major`: a C-order input read in place, docs/design/linalg.md §5.5).
// Up to kTrsvMaxK columns go through BLAS trsv one column at a time: Accelerate's trsm (and
// so trtrs, getrs, potrs) was 3-4x slower than trsv for one column from n = 100 on, 1.7x
// for two, and faster from four. More columns: trtrs, on the transposed view for a
// row-major A (op 'N' <-> 'T'); a row-major A with op 'C' and more columns has no such
// view: callers pack it (row_major_ok).
inline constexpr lint kTrsvMaxK = 2;

inline bool row_major_ok(bool complex, char trans, lint k) { return !(complex && trans == 'C' && k > kTrsvMaxK); }

template <class T>
void tri_solve(bool row_major, char uplo, char trans, char diag, lint n, lint k, const T* a, lint lda, T* b, lint ldb) {
    if (n == 0 || k == 0) return;
    if (k <= kTrsvMaxK) {
        const auto lay = row_major ? CblasRowMajor : CblasColMajor;
        const auto ul = uplo == 'U' ? CblasUpper : CblasLower;
        const auto tr = trans == 'N' ? CblasNoTrans : (trans == 'T' ? CblasTrans : CblasConjTrans);
        const auto dg = diag == 'U' ? CblasUnit : CblasNonUnit;
        for (lint j = 0; j < k; ++j) {
            if constexpr (std::is_same_v<T, double>) cblas_dtrsv(lay, ul, tr, dg, n, a, lda, b + j * ldb, 1);
            else cblas_ztrsv(lay, ul, tr, dg, n, a, lda, b + j * ldb, 1);
        }
        return;
    }
    if (row_major) {  // A^T column-major: the other triangle, the other op
        if (trans == 'C' && std::is_same_v<T, cplx>) throw std::logic_error("tri_solve: row-major A with op 'C'");
        trtrs(uplo == 'U' ? 'L' : 'U', trans == 'N' ? 'T' : 'N', diag, n, k, a, lda, b, ldb);
        return;
    }
    trtrs(uplo, trans, diag, n, k, a, lda, b, ldb);
}

// getrs('N') with the factors of getrf: the row swaps, then L (unit) and U by tri_solve.
template <class T>
void lu_solve_factored(lint n, lint k, const T* lu, const lint* ipiv, T* b, lint ldb) {
    if (n == 0 || k == 0) return;
    if (k > kTrsvMaxK) {
        getrs('N', n, k, lu, n, ipiv, b, ldb);
        return;
    }
    for (lint j = 0; j < k; ++j) {
        T* c = b + j * ldb;
        for (lint i = 0; i < n; ++i)
            if (ipiv[i] - 1 != i) std::swap(c[i], c[ipiv[i] - 1]);
    }
    tri_solve(false, 'L', 'N', 'U', n, k, lu, n, b, ldb);
    tri_solve(false, 'U', 'N', 'N', n, k, lu, n, b, ldb);
}

// The reciprocal 1-norm condition number of a real triangular matrix (an estimate, Hager's
// method: within a small factor of the true value). Reads only triangle `uplo`. The
// workspaces come from the wrappers' slots of `s` (docs/design/linalg.md §6.3).
inline double trcon_1(char uplo, lint n, const double* a, lint lda, Scratch& s) {
    if (n == 0) return 1.0;
    double* work = s.get<double>(static_cast<std::size_t>(3 * n), Scratch::kWork);
    lint* iwork = s.get_ints(static_cast<std::size_t>(n), Scratch::kIWork);
    char norm = '1', diag = 'N';
    double rcond = 0.0;
    lint info = 0;
    dtrcon_(&norm, &uplo, &diag, &n, a, &lda, &rcond, work, iwork, &info);
    checked(info, "trcon");
    return rcond;
}
// The same for a caller without a Scratch (one of its own, freed on return).
inline double trcon_1(char uplo, lint n, const double* a, lint lda) {
    Scratch s;
    return trcon_1(uplo, n, a, lda, s);
}

// Solves a tridiagonal system (partial pivoting) in one call: dl, d, du are
// overwritten by the factorization, b by the solution. info > 0: an exact zero pivot.
template <class T>
lint gtsv(lint n, lint nrhs, T* dl, T* d, T* du, T* b, lint ldb) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dgtsv_(&n, &nrhs, dl, d, du, b, &ldb, &info);
    else zgtsv_(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return checked(info, "gtsv");
}

// Solves a band system (gbtrf + gbtrs in one call). ab holds the band in rows
// kl .. 2*kl+ku (ldab >= 2*kl+ku+1); rows 0 .. kl-1 are room for the fill-in.
// info > 0: an exact zero pivot.
template <class T>
lint gbsv(lint n, lint kl, lint ku, lint nrhs, T* ab, lint ldab, lint* ipiv, T* b, lint ldb) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dgbsv_(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    else zgbsv_(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    return checked(info, "gbsv");
}

// ---- Routines with a workspace. Each asks LAPACK for the size (lwork = -1)
// once per Scratch (Scratch::sizes: once per batch when it runs serially,
// once per task when run_batch splits it; docs/design/linalg.md §6.3) and
// takes the memory from the scratch slots reserved for it.

inline lint work_size(double q) { return std::max<lint>(1, static_cast<lint>(q)); }
inline lint work_size(cplx q) { return std::max<lint>(1, static_cast<lint>(q.real())); }

// Routine ids for Scratch::sizes keys.
enum WorkId : lint {
    kGeqrf = 1,
    kGeqp3,
    kOrgqr,
    kGelsd,
    kSyevd,
    kSygvd,
    kSygv,
    kGesdd,
    kGesvd,
    kGeev,
    kGees,
    kGehrd,
    kOrghr,
    kSytrf,
    kTrsyl3,
    kOrmqr
};

template <class T>
constexpr lint esize() {
    return static_cast<lint>(sizeof(T));
}

// QR: A = Q R with Householder reflectors below R's diagonal and their scalars in tau (min(m, n)).
template <class T>
lint geqrf(lint m, lint n, T* a, lint lda, T* tau, Scratch& s) {
    lint info = 0;
    lint lwork = s.sizes({kGeqrf, m, n, lda, 0, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>) dgeqrf_(&m, &n, a, &lda, tau, &q, &q_lwork, &info);
                      else zgeqrf_(&m, &n, a, &lda, tau, &q, &q_lwork, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dgeqrf_(&m, &n, a, &lda, tau, w, &lwork, &info);
    else zgeqrf_(&m, &n, a, &lda, tau, w, &lwork, &info);
    return checked(info, "geqrf");
}

// QR with column pivoting: A P = Q R; jpvt (n, 1-based) must be 0 on entry (all columns free).
template <class T>
lint geqp3(lint m, lint n, T* a, lint lda, lint* jpvt, T* tau, Scratch& s) {
    lint info = 0;
    double* rw = nullptr;
    if constexpr (!std::is_same_v<T, double>) rw = s.get<double>(static_cast<std::size_t>(2 * n), Scratch::kRWork);
    lint lwork = s.sizes({kGeqp3, m, n, lda, 0, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>) dgeqp3_(&m, &n, a, &lda, jpvt, tau, &q, &q_lwork, &info);
                      else zgeqp3_(&m, &n, a, &lda, jpvt, tau, &q, &q_lwork, rw, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dgeqp3_(&m, &n, a, &lda, jpvt, tau, w, &lwork, &info);
    else zgeqp3_(&m, &n, a, &lda, jpvt, tau, w, &lwork, rw, &info);
    return checked(info, "geqp3");
}

// Forms the m x n matrix Q with orthonormal columns from k reflectors (m >= n >= k), in place.
template <class T>
lint orgqr(lint m, lint n, lint k, T* a, lint lda, const T* tau, Scratch& s) {
    lint info = 0;
    lint lwork = s.sizes({kOrgqr, m, n, k, lda, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>) dorgqr_(&m, &n, &k, a, &lda, tau, &q, &q_lwork, &info);
                      else zungqr_(&m, &n, &k, a, &lda, tau, &q, &q_lwork, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dorgqr_(&m, &n, &k, a, &lda, tau, w, &lwork, &info);
    else zungqr_(&m, &n, &k, a, &lda, tau, w, &lwork, &info);
    return checked(info, "orgqr");
}

// Applies Q (trans 'N') or Q^T (trans 'T') from k reflectors (geqrf's, in a) to the
// m x n matrix c from the left, in place. Real only (sd.regression, docs/design/regression.md §3.1).
inline lint ormqr(
    char trans, lint m, lint n, lint k, const double* a, lint lda, const double* tau, double* c, lint ldc, Scratch& s
) {
    lint info = 0;
    const char side = 'L';
    double* av = const_cast<double*>(a);  // dormqr's prototype is not const; it does not write a
    lint lwork = s.sizes({kOrmqr, m, n, k, trans, esize<double>()}, [&] {
                      lint q_lwork = -1;
                      double q = 0;
                      dormqr_(&side, &trans, &m, &n, &k, av, &lda, tau, c, &ldc, &q, &q_lwork, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    double* w = s.get<double>(static_cast<std::size_t>(lwork), Scratch::kWork);
    dormqr_(&side, &trans, &m, &n, &k, av, &lda, tau, c, &ldc, w, &lwork, &info);
    return checked(info, "ormqr");
}

// (U^T U)^-1 from the upper triangular U, in place (upper triangle). info > 0: U has a zero on its diagonal.
inline lint potri_upper(lint n, double* a, lint lda) {
    lint info = 0;
    const char uplo = 'U';
    dpotri_(&uplo, &n, a, &lda, &info);
    return checked(info, "potri");
}

// One elementary reflector H = I - tau v v^T with H [alpha; x] = [beta; 0] (dlarfg):
// alpha becomes beta, x becomes v(2:n) (v(1) = 1).
inline void larfg(lint n, double* alpha, double* x, double* tau) {
    const lint one = 1;
    dlarfg_(&n, alpha, x, &one, tau);
}

// C = H C for H = I - tau v v^T (dlarf, side 'L'); c is m x n, work has n doubles.
inline void larf_left(lint m, lint n, const double* v, double tau, double* c, lint ldc, double* work) {
    const char side = 'L';
    const lint one = 1;
    dlarf_(&side, &m, &n, v, &one, &tau, c, &ldc, work);
}

// Minimum-norm least squares by SVD (divide and conquer). b (ldb >= max(m, n))
// holds the solution in its first n rows on exit; sv gets the min(m, n)
// singular values, descending. info > 0: the SVD did not converge.
template <class T>
lint gelsd(
    lint m, lint n, lint nrhs, T* a, lint lda, T* b, lint ldb, double* sv, double rcond, lint& rank, Scratch& s
) {
    lint info = 0;
    const WorkSizes z = s.sizes({kGelsd, m, n, nrhs, ldb, esize<T>()}, [&] {
        lint q_lwork = -1, iq = 0;
        T q{};
        double rq = 0;
        if constexpr (std::is_same_v<T, double>)
            dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, &q, &q_lwork, &iq, &info);
        else zgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, &q, &q_lwork, &rq, &iq, &info);
        return WorkSizes{work_size(q), work_size(rq), std::max<lint>(1, iq)};
    });
    lint lwork = z.work;
    T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    lint* iw = s.get_ints(static_cast<std::size_t>(z.iwork), Scratch::kIWork);
    if constexpr (std::is_same_v<T, double>) {
        dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, w, &lwork, iw, &info);
    } else {
        double* rw = s.get<double>(static_cast<std::size_t>(z.rwork), Scratch::kRWork);
        zgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, w, &lwork, rw, iw, &info);
    }
    return checked(info, "gelsd");
}

// Hermitian eigenproblem by divide and conquer (heevd/syevd), lower triangle.
// jobz 'V': eigenvectors overwrite a. w (n) ascending. info > 0: no convergence.
template <class T>
lint syevd(char jobz, lint n, T* a, lint lda, double* w, Scratch& s) {
    lint info = 0;
    const char uplo = 'L';
    const WorkSizes z = s.sizes({kSyevd, n, lda, jobz, 0, esize<T>()}, [&] {
        lint q_lwork = -1, q_liwork = -1, q_lrwork = -1, iq = 0;
        T q{};
        double rq = 0;
        if constexpr (std::is_same_v<T, double>)
            dsyevd_(&jobz, &uplo, &n, a, &lda, w, &q, &q_lwork, &iq, &q_liwork, &info);
        else zheevd_(&jobz, &uplo, &n, a, &lda, w, &q, &q_lwork, &rq, &q_lrwork, &iq, &q_liwork, &info);
        return WorkSizes{work_size(q), work_size(rq), std::max<lint>(1, iq)};
    });
    lint lwork = z.work, lrwork = z.rwork, liwork = z.iwork;
    T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
    if constexpr (std::is_same_v<T, double>) {
        dsyevd_(&jobz, &uplo, &n, a, &lda, w, wk, &lwork, iw, &liwork, &info);
    } else {
        double* rw = s.get<double>(static_cast<std::size_t>(lrwork), Scratch::kRWork);
        zheevd_(&jobz, &uplo, &n, a, &lda, w, wk, &lwork, rw, &lrwork, iw, &liwork, &info);
    }
    return checked(info, "syevd");
}

// Generalized a v = w b v (itype 1), lower triangles, b Hermitian positive definite.
// divide: hegvd/sygvd (used with vectors); otherwise hegv/sygv (used for values only).
// info in 1..n: no convergence; info > n: b is not positive definite.
template <class T>
lint sygv(bool divide, char jobz, lint n, T* a, lint lda, T* b, lint ldb, double* w, Scratch& s) {
    lint info = 0, itype = 1;
    const char uplo = 'L';
    double* hegv_rw = nullptr;  // zhegv's fixed real workspace
    if constexpr (!std::is_same_v<T, double>)
        if (!divide) hegv_rw = s.get<double>(static_cast<std::size_t>(std::max<lint>(1, 3 * n - 2)), Scratch::kRWork);
    const WorkSizes z = s.sizes({divide ? kSygvd : kSygv, n, lda, ldb, jobz, esize<T>()}, [&] {
        lint q_lwork = -1, q_liwork = -1, q_lrwork = -1, iq = 0;
        T q{};
        double rq = 0;
        if constexpr (std::is_same_v<T, double>) {
            if (divide) dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &q_lwork, &iq, &q_liwork, &info);
            else dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &q_lwork, &info);
        } else {
            if (divide)
                zhegvd_(
                    &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &q_lwork, &rq, &q_lrwork, &iq, &q_liwork, &info
                );
            else zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &q_lwork, hegv_rw, &info);
        }
        return WorkSizes{work_size(q), work_size(rq), std::max<lint>(1, iq)};
    });
    lint lwork = z.work, lrwork = z.rwork, liwork = z.iwork;
    T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) {
        if (divide) {
            lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
            dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, iw, &liwork, &info);
        } else {
            dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, &info);
        }
    } else {
        if (divide) {
            double* rw = s.get<double>(static_cast<std::size_t>(lrwork), Scratch::kRWork);
            lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
            zhegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, rw, &lrwork, iw, &liwork, &info);
        } else {
            zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, hegv_rw, &info);
        }
    }
    return checked(info, "sygv");
}

// SVD by divide and conquer: jobz 'S' (reduced U, Vh) or 'A' (complete). a is destroyed.
// info > 0: no convergence.
template <class T>
lint gesdd(char jobz, lint m, lint n, T* a, lint lda, double* sv, T* u, lint ldu, T* vt, lint ldvt, Scratch& s) {
    lint info = 0;
    const lint mn = std::min(m, n), mx = std::max(m, n);
    lint* iw = s.get_ints(static_cast<std::size_t>(std::max<lint>(1, 8 * mn)), Scratch::kIWork);
    double* rw = nullptr;
    if constexpr (!std::is_same_v<T, double>) {
        // zgesdd does not report its real workspace; LAPACK 3.7+'s documented size.
        const lint lrwork =
            jobz == 'N' ? std::max<lint>(1, 7 * mn) : std::max<lint>(1, mn * std::max(5 * mn + 7, 2 * mx + 2 * mn + 1));
        rw = s.get<double>(static_cast<std::size_t>(lrwork), Scratch::kRWork);
    }
    lint lwork = s.sizes({kGesdd, m, n, lda, jobz, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>)
                          dgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, &q, &q_lwork, iw, &info);
                      else zgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, &q, &q_lwork, rw, iw, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>)
        dgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, wk, &lwork, iw, &info);
    else zgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, wk, &lwork, rw, iw, &info);
    return checked(info, "gesdd");
}

// Singular values only, by QR iteration (gesvd with jobu = jobvt = 'N'). a is destroyed.
template <class T>
lint gesvd_values(lint m, lint n, T* a, lint lda, double* sv, Scratch& s) {
    lint info = 0, one = 1;
    const char no = 'N';
    T dummy{};
    double* rw = nullptr;
    if constexpr (!std::is_same_v<T, double>)
        rw = s.get<double>(static_cast<std::size_t>(std::max<lint>(1, 5 * std::min(m, n))), Scratch::kRWork);
    lint lwork = s.sizes({kGesvd, m, n, lda, 0, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>)
                          dgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, &q, &q_lwork, &info);
                      else zgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, &q, &q_lwork, rw, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>)
        dgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, wk, &lwork, &info);
    else zgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, wk, &lwork, rw, &info);
    return checked(info, "gesvd");
}

// General eigenproblem (geev): eigenvalues, and right eigenvectors if jobvr 'V'.
// Real: wr/wi and vr (complex pairs as two real columns); complex: w and vr.
// `wr` holds n values for double (and `wi` n more); for cplx `w` is cplx*.
// info > 0: the QR algorithm did not converge.
inline lint geev(char jobvr, lint n, double* a, lint lda, double* wr, double* wi, double* vr, lint ldvr, Scratch& s) {
    lint info = 0, one = 1;
    const char no = 'N';
    double dummy = 0;
    lint lwork = s.sizes({kGeev, n, lda, ldvr, jobvr, esize<double>()}, [&] {
                      lint q_lwork = -1;
                      double q = 0;
                      dgeev_(&no, &jobvr, &n, a, &lda, wr, wi, &dummy, &one, vr, &ldvr, &q, &q_lwork, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    double* wk = s.get<double>(static_cast<std::size_t>(lwork), Scratch::kWork);
    dgeev_(&no, &jobvr, &n, a, &lda, wr, wi, &dummy, &one, vr, &ldvr, wk, &lwork, &info);
    return checked(info, "geev");
}
inline lint geev(char jobvr, lint n, cplx* a, lint lda, cplx* w, cplx* vr, lint ldvr, Scratch& s) {
    lint info = 0, one = 1;
    const char no = 'N';
    cplx dummy{};
    double* rw = s.get<double>(static_cast<std::size_t>(std::max<lint>(1, 2 * n)), Scratch::kRWork);
    lint lwork = s.sizes({kGeev, n, lda, ldvr, jobvr, esize<cplx>()}, [&] {
                      lint q_lwork = -1;
                      cplx q{};
                      zgeev_(&no, &jobvr, &n, a, &lda, w, &dummy, &one, vr, &ldvr, &q, &q_lwork, rw, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    cplx* wk = s.get<cplx>(static_cast<std::size_t>(lwork), Scratch::kWork);
    zgeev_(&no, &jobvr, &n, a, &lda, w, &dummy, &one, vr, &ldvr, wk, &lwork, rw, &info);
    return checked(info, "geev");
}

// Schur form A = Z T Z^H (gees, no sorting): T overwrites a, Z goes to vs.
// Real input gives the real (quasi-triangular) Schur form. info > 0: no convergence.
template <class T>
lint gees(lint n, T* a, lint lda, T* vs, lint ldvs, Scratch& s) {
    lint info = 0, sdim = 0;
    const char jobvs = 'V', sort = 'N';
    __LAPACK_bool* bwork = nullptr;  // not referenced when sort = 'N'
    if constexpr (std::is_same_v<T, double>) {
        // wr and wi, each starting on a 16-byte boundary (common.hpp: aligned16)
        const lint half = (n + 1) & ~lint(1);
        double* wr = s.get<double>(static_cast<std::size_t>(2 * half), Scratch::kRWork);
        double* wi = wr + half;
        lint lwork =
            s.sizes({kGees, n, lda, ldvs, 0, esize<T>()}, [&] {
                 lint q_lwork = -1;
                 T q{};
                 dgees_(&jobvs, &sort, nullptr, &n, a, &lda, &sdim, wr, wi, vs, &ldvs, &q, &q_lwork, bwork, &info);
                 return WorkSizes{work_size(q)};
             }).work;
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        dgees_(&jobvs, &sort, nullptr, &n, a, &lda, &sdim, wr, wi, vs, &ldvs, wk, &lwork, bwork, &info);
    } else {
        // w (n complex) and rwork (n real) share the real-workspace slot
        double* rw = s.get<double>(static_cast<std::size_t>(3 * n), Scratch::kRWork);
        cplx* w = reinterpret_cast<cplx*>(rw);
        lint lwork =
            s.sizes({kGees, n, lda, ldvs, 0, esize<T>()}, [&] {
                 lint q_lwork = -1;
                 T q{};
                 zgees_(
                     &jobvs, &sort, nullptr, &n, a, &lda, &sdim, w, vs, &ldvs, &q, &q_lwork, rw + 2 * n, bwork, &info
                 );
                 return WorkSizes{work_size(q)};
             }).work;
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        zgees_(&jobvs, &sort, nullptr, &n, a, &lda, &sdim, w, vs, &ldvs, wk, &lwork, rw + 2 * n, bwork, &info);
    }
    return checked(info, "gees");
}

// Hessenberg reduction A = Q H Q^H (gehrd, ilo = 1, ihi = n): reflectors below the subdiagonal.
template <class T>
lint gehrd(lint n, T* a, lint lda, T* tau, Scratch& s) {
    lint info = 0, ilo = 1, ihi = n;
    lint lwork = s.sizes({kGehrd, n, lda, 0, 0, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>)
                          dgehrd_(&n, &ilo, &ihi, a, &lda, tau, &q, &q_lwork, &info);
                      else zgehrd_(&n, &ilo, &ihi, a, &lda, tau, &q, &q_lwork, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dgehrd_(&n, &ilo, &ihi, a, &lda, tau, wk, &lwork, &info);
    else zgehrd_(&n, &ilo, &ihi, a, &lda, tau, wk, &lwork, &info);
    return checked(info, "gehrd");
}

// Forms Q from gehrd's reflectors, in place (orghr/unghr).
template <class T>
lint orghr(lint n, T* a, lint lda, const T* tau, Scratch& s) {
    lint info = 0, ilo = 1, ihi = n;
    lint lwork = s.sizes({kOrghr, n, lda, 0, 0, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>)
                          dorghr_(&n, &ilo, &ihi, a, &lda, tau, &q, &q_lwork, &info);
                      else zunghr_(&n, &ilo, &ihi, a, &lda, tau, &q, &q_lwork, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dorghr_(&n, &ilo, &ihi, a, &lda, tau, wk, &lwork, &info);
    else zunghr_(&n, &ilo, &ihi, a, &lda, tau, wk, &lwork, &info);
    return checked(info, "orghr");
}

// Bunch-Kaufman A = L D L^T (sytrf) or L D L^H (hetrf), lower triangle, in place.
// info > 0: D has an exact zero on its diagonal (the factorization is still complete).
template <class T>
lint sytrf(lint n, T* a, lint lda, lint* ipiv, Scratch& s) {
    lint info = 0;
    const char uplo = 'L';
    lint lwork = s.sizes({kSytrf, n, lda, 0, 0, esize<T>()}, [&] {
                      lint q_lwork = -1;
                      T q{};
                      if constexpr (std::is_same_v<T, double>) dsytrf_(&uplo, &n, a, &lda, ipiv, &q, &q_lwork, &info);
                      else zhetrf_(&uplo, &n, a, &lda, ipiv, &q, &q_lwork, &info);
                      return WorkSizes{work_size(q)};
                  }).work;
    T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dsytrf_(&uplo, &n, a, &lda, ipiv, wk, &lwork, &info);
    else zhetrf_(&uplo, &n, a, &lda, ipiv, wk, &lwork, &info);
    return checked(info, "sytrf");
}

// Solves A X + X B = scale C for upper (quasi-)triangular A (m x m) and B (n x n),
// the Schur forms: the blocked level-3 trsyl3 (LAPACK 3.10), chosen over trsyl
// by measurement (docs/design/linalg.md §10.7). C is overwritten
// by X; scale <= 1 avoids overflow. info = 1: A and -B have common or close
// eigenvalues, and perturbed values were used.
template <class T>
lint trsyl3(lint m, lint n, const T* a, lint lda, const T* b, lint ldb, T* c, lint ldc, double& scale, Scratch& s) {
    lint info = 0, isgn = 1;
    const char no = 'N';
    // The query returns IWORK's length in iwork[0] and SWORK's rows and columns
    // in swork[0], swork[1]; kept as (work: rows, rwork: cols, iwork: length).
    const WorkSizes z = s.sizes({kTrsyl3, m, n, lda, ldb, esize<T>()}, [&] {
        lint q_liwork = -1, q_ldswork = -1, iq = 0;
        double sq[2] = {0, 0};
        if constexpr (std::is_same_v<T, double>)
            dtrsyl3_(&no, &no, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, &iq, &q_liwork, sq, &q_ldswork, &info);
        else ztrsyl3_(&no, &no, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, sq, &q_ldswork, &info);
        checked(info, "trsyl3");
        return WorkSizes{
            std::max<lint>(2, static_cast<lint>(sq[0])), std::max<lint>(1, static_cast<lint>(sq[1])),
            std::max<lint>(1, iq)
        };
    });
    lint ldswork = z.work, liwork = z.iwork;
    double* sw = s.get<double>(static_cast<std::size_t>(z.work * z.rwork), Scratch::kRWork);
    if constexpr (std::is_same_v<T, double>) {
        lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
        dtrsyl3_(&no, &no, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, iw, &liwork, sw, &ldswork, &info);
    } else {
        ztrsyl3_(&no, &no, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, sw, &ldswork, &info);
    }
    return checked(info, "trsyl3");
}

// C = op(A) op(B), column-major; op is 'N', 'T' or 'C' (conjugate transpose).
template <class T>
void gemm(char ta, char tb, lint m, lint n, lint k, const T* a, lint lda, const T* b, lint ldb, T* c, lint ldc) {
    auto op = [](char t) { return t == 'N' ? CblasNoTrans : (t == 'T' ? CblasTrans : CblasConjTrans); };
    if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasColMajor, op(ta), op(tb), m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    } else {
        const T one(1.0), zero(0.0);
        cblas_zgemm(CblasColMajor, op(ta), op(tb), m, n, k, &one, a, lda, b, ldb, &zero, c, ldc);
    }
}

// C = A B for tiny n x n float64 column-major matrices (n <= kTinyGemm): each column of C
// an fma chain over the inner index. BLAS gemm's call cost was most of the time of a 3 x 3
// product (matrix_power, expm of small matrices).
inline constexpr lint kTinyGemm = 8;

inline void tiny_gemm(lint n, const double* a, const double* b, double* c) {
    for (lint j = 0; j < n; ++j) {
        double* cj = c + j * n;
        const double* bj = b + j * n;
        for (lint i = 0; i < n; ++i) cj[i] = a[i] * bj[0];
        for (lint p = 1; p < n; ++p) {
            const double bp = bj[p];
            const double* ap = a + p * n;
            for (lint i = 0; i < n; ++i) cj[i] = std::fma(ap[i], bp, cj[i]);
        }
    }
}

// C = A B for n x n column-major matrices (BLAS gemm, or tiny_gemm for tiny float64 ones;
// C must not alias A or B).
template <class T>
void gemm_square(lint n, const T* a, const T* b, T* c) {
    if constexpr (std::is_same_v<T, double>) {
        if (n <= kTinyGemm) {
            tiny_gemm(n, a, b, c);
            return;
        }
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
    } else {
        const T one(1.0), zero(0.0);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, &one, a, n, b, n, &zero, c, n);
    }
}

// y = op(A) x for an n x n column-major A; op 'N', 'T' or 'C' (conjugate transpose).
template <class T>
void gemv(char op, lint n, const T* a, const T* x, T* y) {
    const auto o = op == 'N' ? CblasNoTrans : (op == 'T' ? CblasTrans : CblasConjTrans);
    if constexpr (std::is_same_v<T, double>) {
        cblas_dgemv(CblasColMajor, o, n, n, 1.0, a, n, x, 1, 0.0, y, 1);
    } else {
        const T one(1.0), zero(0.0);
        cblas_zgemv(CblasColMajor, o, n, n, &one, a, n, x, 1, &zero, y, 1);
    }
}

// One step of the 1-norm estimator (Hager 1984, Higham 1988), reverse
// communication: on return kase = 1 asks for x <- A x, kase = 2 for
// x <- A^H x, kase = 0 means `est` is final. Deterministic (no random start).
// isgn (n ints) is used by the real version only.
template <class T>
void lacn2(lint n, T* v, T* x, lint* isgn, double& est, lint& kase, lint* isave) {
    if constexpr (std::is_same_v<T, double>) dlacn2_(&n, v, x, isgn, &est, &kase, isave);
    else zlacn2_(&n, v, x, &est, &kase, isave);
}

// ---- The small dense problems inside ARPACK (sd.sparse eigsh, eigs; docs/design/sparse.md §3.8).
// Plain calls, no workspace query: the caller passes LAPACK's documented minimum workspace.

// All eigenvalues (ascending, in d) and eigenvectors (z, ldz) of the symmetric tridiagonal matrix
// with diagonal d and off-diagonal e (dsteqr, compz 'I'); work has max(1, 2n - 2) doubles.
inline lint steqr(lint n, double* d, double* e, double* z, lint ldz, double* work) {
    lint info = 0;
    const char compz = 'I';
    dsteqr_(&compz, &n, d, e, z, &ldz, work, &info);
    return checked(info, "steqr");
}

// The Schur form of an upper Hessenberg matrix by the double-shift QR algorithm (dlahqr, zlahqr):
// wantt, wantz, rows iloz..ihiz of z updated. Real: eigenvalues in wr, wi; complex: in w.
inline lint lahqr(
    bool wantt,
    bool wantz,
    lint n,
    lint ilo,
    lint ihi,
    double* h,
    lint ldh,
    double* wr,
    double* wi,
    lint iloz,
    lint ihiz,
    double* z,
    lint ldz
) {
    lint info = 0;
    const __LAPACK_bool t = wantt, zz = wantz;
    dlahqr_(&t, &zz, &n, &ilo, &ihi, h, &ldh, wr, wi, &iloz, &ihiz, z, &ldz, &info);
    return checked(info, "lahqr");
}
inline lint lahqr(
    bool wantt,
    bool wantz,
    lint n,
    lint ilo,
    lint ihi,
    cplx* h,
    lint ldh,
    cplx* w,
    lint iloz,
    lint ihiz,
    cplx* z,
    lint ldz
) {
    lint info = 0;
    const __LAPACK_bool t = wantt, zz = wantz;
    zlahqr_(&t, &zz, &n, &ilo, &ihi, h, &ldh, w, &iloz, &ihiz, z, &ldz, &info);
    return checked(info, "lahqr");
}

// Right eigenvectors of a (quasi-)triangular matrix (dtrevc, ztrevc; side 'R'): howmny 'A' (all),
// 'B' (back-transformed by vr) or 'S' (the selected ones). Real work: 3n doubles; complex work:
// 2n, rwork: n.
inline lint trevc(
    char howmny,
    __LAPACK_bool* select,
    lint n,
    const double* t,
    lint ldt,
    double* vr,
    lint ldvr,
    lint mm,
    lint& m,
    double* work
) {
    lint info = 0, ldvl = 1;
    const char side = 'R';
    double vl = 0;
    dtrevc_(&side, &howmny, select, &n, t, &ldt, &vl, &ldvl, vr, &ldvr, &mm, &m, work, &info);
    return checked(info, "trevc");
}
inline lint trevc(
    char howmny,
    __LAPACK_bool* select,
    lint n,
    cplx* t,
    lint ldt,
    cplx* vr,
    lint ldvr,
    lint mm,
    lint& m,
    cplx* work,
    double* rwork
) {
    lint info = 0, ldvl = 1;
    const char side = 'R';
    cplx vl{};
    ztrevc_(&side, &howmny, select, &n, t, &ldt, &vl, &ldvl, vr, &ldvr, &mm, &m, work, rwork, &info);
    return checked(info, "trevc");
}

// Reorders a (quasi-)triangular Schur form so the selected eigenvalues lead (dtrsen, ztrsen; job
// 'N', compq 'V'); m returns their number. Real work: n doubles; complex work: n.
inline lint trsen(
    const __LAPACK_bool* select,
    lint n,
    double* t,
    lint ldt,
    double* q,
    lint ldq,
    double* wr,
    double* wi,
    lint& m,
    double* work
) {
    lint info = 0, liwork = 1, iwork = 0;
    const char job = 'N', compq = 'V';
    double s = 0, sep = 0;
    dtrsen_(&job, &compq, select, &n, t, &ldt, q, &ldq, wr, wi, &m, &s, &sep, work, &n, &iwork, &liwork, &info);
    return checked(info, "trsen");
}
inline lint trsen(
    const __LAPACK_bool* select, lint n, cplx* t, lint ldt, cplx* q, lint ldq, cplx* w, lint& m, cplx* work
) {
    lint info = 0;
    const char job = 'N', compq = 'V';
    double s = 0, sep = 0;
    ztrsen_(&job, &compq, select, &n, t, &ldt, q, &ldq, w, &m, &s, &sep, work, &n, &info);
    return checked(info, "trsen");
}

// The unblocked QR factorization (dgeqr2, zgeqr2); work has n entries.
template <class T>
lint geqr2(lint m, lint n, T* a, lint lda, T* tau, T* work) {
    lint info = 0;
    if constexpr (std::is_same_v<T, double>) dgeqr2_(&m, &n, a, &lda, tau, work, &info);
    else zgeqr2_(&m, &n, a, &lda, tau, work, &info);
    return checked(info, "geqr2");
}

// The 1-norm of an upper Hessenberg matrix (dlanhs, zlanhs '1').
template <class T>
double lanhs1(lint n, const T* a, lint lda) {
    const char norm = '1';
    double work = 0;
    if constexpr (std::is_same_v<T, double>) return dlanhs_(&norm, &n, a, &lda, &work);
    else return zlanhs_(&norm, &n, a, &lda, &work);
}

// C = H C (side 'L') or C H (side 'R') for H = I - tau v v^T (dlarf); work has n ('L') or m
// ('R') doubles.
inline void larf(char side, lint m, lint n, const double* v, double tau, double* c, lint ldc, double* work) {
    const lint one = 1;
    dlarf_(&side, &m, &n, v, &one, &tau, c, &ldc, work);
}

}  // namespace sd::linalg::lapack
