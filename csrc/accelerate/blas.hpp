// The BLAS routines the array area calls (docs/design/array.md §9.2): thin,
// type-generic wrappers over Accelerate's CBLAS, one template per routine for
// float64 and complex128 (the LAPACK side, bound to linalg's types, is linalg/lapack.hpp).
//
// All are row-major with alpha = 1 and beta = 0, and unconjugated (matmul
// never conjugates) -- except syrk_upper, cov's A A^H. Integers are Accelerate's 64-bit __LAPACK_int (ILP64, set
// in CMakeLists.txt).
#pragma once

#include <Accelerate/Accelerate.h>

#include <complex>
#include <type_traits>

namespace sd::blas {

using Int = __LAPACK_int;
using Trans = CBLAS_TRANSPOSE;

// *out = sum x[i] * y[i] over k terms.
template <class T>
void dot(Int k, const T* x, Int incx, const T* y, Int incy, T* out) {
    if constexpr (std::is_same_v<T, std::complex<double>>) cblas_zdotu_sub(k, x, incx, y, incy, out);
    else *out = cblas_ddot(k, x, incx, y, incy);
}

// y = op(A) x, A stored rows x cols with leading dimension lda; y has unit stride.
template <class T>
void gemv(Trans trans, Int rows, Int cols, const T* a, Int lda, const T* x, Int incx, T* y) {
    const T one(1), zero(0);
    if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zgemv(CblasRowMajor, trans, rows, cols, &one, a, lda, x, incx, &zero, y, 1);
    else cblas_dgemv(CblasRowMajor, trans, rows, cols, one, a, lda, x, incx, zero, y, 1);
}

// C (m x n, leading dimension ldc) = op(A) op(B), op(A) m x k, op(B) k x n.
template <class T>
void gemm(Trans ta, Trans tb, Int m, Int n, Int k, const T* a, Int lda, const T* b, Int ldb, T* c, Int ldc) {
    const T one(1), zero(0);
    if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zgemm(CblasRowMajor, ta, tb, m, n, k, &one, a, lda, b, ldb, &zero, c, ldc);
    else cblas_dgemm(CblasRowMajor, ta, tb, m, n, k, one, a, lda, b, ldb, zero, c, ldc);
}

// The upper triangle of C (n x n, leading dimension ldc) = A A^T for float64
// (dsyrk) or A A^H for complex128 (zherk, whose diagonal is exactly real);
// A is n x k row-major. Half the multiply-adds of gemm (cov, docs/design/array.md §6.12).
template <class T>
void syrk_upper(Int n, Int k, const T* a, Int lda, T* c, Int ldc) {
    if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, n, k, 1.0, a, lda, 0.0, c, ldc);
    else cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, k, 1.0, a, lda, 0.0, c, ldc);
}

}  // namespace sd::blas
