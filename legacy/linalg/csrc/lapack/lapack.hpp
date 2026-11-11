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
