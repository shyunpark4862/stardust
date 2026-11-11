// Thin typed wrappers over Accelerate LAPACK (ARCHITECTURE.md §3.5): one
// template per routine, dispatching to the d (double) or z (complex) version.
// They return LAPACK's `info`; a negative info is a bug in our arguments.
#pragma once

#include "core/types.hpp"

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
