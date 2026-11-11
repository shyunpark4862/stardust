// Thin typed wrappers over Accelerate LAPACK (ARCHITECTURE.md §3.5): one
// template per routine, dispatching to the d (double) or z (complex) version.
// They return LAPACK's `info`; a negative info is a bug in our arguments.
#pragma once

#include "core/batch.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sdla::lapack {

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

// ---- Routines with a workspace. Each asks LAPACK for the size (lwork = -1)
// and takes the memory from the scratch slots reserved for it; M1-M7 query on
// every call (ARCHITECTURE.md §3.4), M8 measures doing it once per batch.

inline lint work_size(double q) { return std::max<lint>(1, static_cast<lint>(q)); }
inline lint work_size(cplx q) { return std::max<lint>(1, static_cast<lint>(q.real())); }

// QR: A = Q R with Householder reflectors below R's diagonal and their scalars in tau (min(m, n)).
template <class T>
lint geqrf(lint m, lint n, T* a, lint lda, T* tau, Scratch& s) {
    lint info = 0, lwork = -1;
    T q{};
    if constexpr (std::is_same_v<T, double>) dgeqrf_(&m, &n, a, &lda, tau, &q, &lwork, &info);
    else zgeqrf_(&m, &n, a, &lda, tau, &q, &lwork, &info);
    lwork = work_size(q);
    T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dgeqrf_(&m, &n, a, &lda, tau, w, &lwork, &info);
    else zgeqrf_(&m, &n, a, &lda, tau, w, &lwork, &info);
    return checked(info, "geqrf");
}

// QR with column pivoting: A P = Q R; jpvt (n, 1-based) must be 0 on entry (all columns free).
template <class T>
lint geqp3(lint m, lint n, T* a, lint lda, lint* jpvt, T* tau, Scratch& s) {
    lint info = 0, lwork = -1;
    T q{};
    if constexpr (std::is_same_v<T, double>) {
        dgeqp3_(&m, &n, a, &lda, jpvt, tau, &q, &lwork, &info);
        lwork = work_size(q);
        T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        dgeqp3_(&m, &n, a, &lda, jpvt, tau, w, &lwork, &info);
    } else {
        double* rw = s.get<double>(static_cast<std::size_t>(2 * n), Scratch::kRWork);
        zgeqp3_(&m, &n, a, &lda, jpvt, tau, &q, &lwork, rw, &info);
        lwork = work_size(q);
        T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        zgeqp3_(&m, &n, a, &lda, jpvt, tau, w, &lwork, rw, &info);
    }
    return checked(info, "geqp3");
}

// Forms the m x n matrix Q with orthonormal columns from k reflectors (m >= n >= k), in place.
template <class T>
lint orgqr(lint m, lint n, lint k, T* a, lint lda, const T* tau, Scratch& s) {
    lint info = 0, lwork = -1;
    T q{};
    if constexpr (std::is_same_v<T, double>) dorgqr_(&m, &n, &k, a, &lda, tau, &q, &lwork, &info);
    else zungqr_(&m, &n, &k, a, &lda, tau, &q, &lwork, &info);
    lwork = work_size(q);
    T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
    if constexpr (std::is_same_v<T, double>) dorgqr_(&m, &n, &k, a, &lda, tau, w, &lwork, &info);
    else zungqr_(&m, &n, &k, a, &lda, tau, w, &lwork, &info);
    return checked(info, "orgqr");
}

// Minimum-norm least squares by SVD (divide and conquer). b (ldb >= max(m, n))
// holds the solution in its first n rows on exit; sv gets the min(m, n)
// singular values, descending. info > 0: the SVD did not converge.
template <class T>
lint gelsd(lint m, lint n, lint nrhs, T* a, lint lda, T* b, lint ldb, double* sv, double rcond, lint& rank,
           Scratch& s) {
    lint info = 0, lwork = -1, iq = 0;
    T q{};
    if constexpr (std::is_same_v<T, double>) {
        dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, &q, &lwork, &iq, &info);
        lwork = work_size(q);
        T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        lint* iw = s.get_ints(static_cast<std::size_t>(std::max<lint>(1, iq)), Scratch::kIWork);
        dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, w, &lwork, iw, &info);
    } else {
        double rq = 0;
        zgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, &q, &lwork, &rq, &iq, &info);
        lwork = work_size(q);
        T* w = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        double* rw = s.get<double>(static_cast<std::size_t>(work_size(rq)), Scratch::kRWork);
        lint* iw = s.get_ints(static_cast<std::size_t>(std::max<lint>(1, iq)), Scratch::kIWork);
        zgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, sv, &rcond, &rank, w, &lwork, rw, iw, &info);
    }
    return checked(info, "gelsd");
}

// Hermitian eigenproblem by divide and conquer (heevd/syevd), lower triangle.
// jobz 'V': eigenvectors overwrite a. w (n) ascending. info > 0: no convergence.
template <class T>
lint syevd(char jobz, lint n, T* a, lint lda, double* w, Scratch& s) {
    lint info = 0, lwork = -1, liwork = -1, iq = 0;
    T q{};
    const char uplo = 'L';
    if constexpr (std::is_same_v<T, double>) {
        dsyevd_(&jobz, &uplo, &n, a, &lda, w, &q, &lwork, &iq, &liwork, &info);
        lwork = work_size(q);
        liwork = std::max<lint>(1, iq);
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
        dsyevd_(&jobz, &uplo, &n, a, &lda, w, wk, &lwork, iw, &liwork, &info);
    } else {
        double rq = 0;
        lint lrwork = -1;
        zheevd_(&jobz, &uplo, &n, a, &lda, w, &q, &lwork, &rq, &lrwork, &iq, &liwork, &info);
        lwork = work_size(q);
        lrwork = work_size(rq);
        liwork = std::max<lint>(1, iq);
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        double* rw = s.get<double>(static_cast<std::size_t>(lrwork), Scratch::kRWork);
        lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
        zheevd_(&jobz, &uplo, &n, a, &lda, w, wk, &lwork, rw, &lrwork, iw, &liwork, &info);
    }
    return checked(info, "syevd");
}

// Generalized a v = w b v (itype 1), lower triangles, b Hermitian positive definite.
// divide: hegvd/sygvd (used with vectors); otherwise hegv/sygv (used for values only).
// info in 1..n: no convergence; info > n: b is not positive definite.
template <class T>
lint sygv(bool divide, char jobz, lint n, T* a, lint lda, T* b, lint ldb, double* w, Scratch& s) {
    lint info = 0, lwork = -1, itype = 1;
    T q{};
    const char uplo = 'L';
    if constexpr (std::is_same_v<T, double>) {
        if (divide) {
            lint liwork = -1, iq = 0;
            dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &lwork, &iq, &liwork, &info);
            lwork = work_size(q);
            liwork = std::max<lint>(1, iq);
            T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
            lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
            dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, iw, &liwork, &info);
        } else {
            dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &lwork, &info);
            lwork = work_size(q);
            T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
            dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, &info);
        }
    } else {
        if (divide) {
            lint liwork = -1, iq = 0, lrwork = -1;
            double rq = 0;
            zhegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &lwork, &rq, &lrwork, &iq, &liwork, &info);
            lwork = work_size(q);
            lrwork = work_size(rq);
            liwork = std::max<lint>(1, iq);
            T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
            double* rw = s.get<double>(static_cast<std::size_t>(lrwork), Scratch::kRWork);
            lint* iw = s.get_ints(static_cast<std::size_t>(liwork), Scratch::kIWork);
            zhegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, rw, &lrwork, iw, &liwork, &info);
        } else {
            double* rw = s.get<double>(static_cast<std::size_t>(std::max<lint>(1, 3 * n - 2)), Scratch::kRWork);
            zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &q, &lwork, rw, &info);
            lwork = work_size(q);
            T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
            zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, wk, &lwork, rw, &info);
        }
    }
    return checked(info, "sygv");
}

// SVD by divide and conquer: jobz 'S' (reduced U, Vh) or 'A' (complete). a is destroyed.
// info > 0: no convergence.
template <class T>
lint gesdd(char jobz, lint m, lint n, T* a, lint lda, double* sv, T* u, lint ldu, T* vt, lint ldvt, Scratch& s) {
    lint info = 0, lwork = -1;
    const lint mn = std::min(m, n), mx = std::max(m, n);
    lint* iw = s.get_ints(static_cast<std::size_t>(std::max<lint>(1, 8 * mn)), Scratch::kIWork);
    T q{};
    if constexpr (std::is_same_v<T, double>) {
        dgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, &q, &lwork, iw, &info);
        lwork = work_size(q);
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        dgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, wk, &lwork, iw, &info);
    } else {
        // zgesdd does not report its real workspace; LAPACK 3.7+'s documented size.
        const lint lrwork = jobz == 'N' ? std::max<lint>(1, 7 * mn)
                                        : std::max<lint>(1, mn * std::max(5 * mn + 7, 2 * mx + 2 * mn + 1));
        double* rw = s.get<double>(static_cast<std::size_t>(lrwork), Scratch::kRWork);
        zgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, &q, &lwork, rw, iw, &info);
        lwork = work_size(q);
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        zgesdd_(&jobz, &m, &n, a, &lda, sv, u, &ldu, vt, &ldvt, wk, &lwork, rw, iw, &info);
    }
    return checked(info, "gesdd");
}

// Singular values only, by QR iteration (gesvd with jobu = jobvt = 'N'). a is destroyed.
template <class T>
lint gesvd_values(lint m, lint n, T* a, lint lda, double* sv, Scratch& s) {
    lint info = 0, lwork = -1, one = 1;
    const char no = 'N';
    T q{}, dummy{};
    if constexpr (std::is_same_v<T, double>) {
        dgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, &q, &lwork, &info);
        lwork = work_size(q);
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        dgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, wk, &lwork, &info);
    } else {
        double* rw = s.get<double>(static_cast<std::size_t>(std::max<lint>(1, 5 * std::min(m, n))), Scratch::kRWork);
        zgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, &q, &lwork, rw, &info);
        lwork = work_size(q);
        T* wk = s.get<T>(static_cast<std::size_t>(lwork), Scratch::kWork);
        zgesvd_(&no, &no, &m, &n, a, &lda, sv, &dummy, &one, &dummy, &one, wk, &lwork, rw, &info);
    }
    return checked(info, "gesvd");
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

// C = A B for n x n column-major matrices (BLAS gemm; C must not alias A or B).
template <class T>
void gemm_square(lint n, const T* a, const T* b, T* c) {
    if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
    } else {
        const T one(1.0), zero(0.0);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, &one, a, n, b, n, &zero, c, n);
    }
}

}  // namespace sdla::lapack
