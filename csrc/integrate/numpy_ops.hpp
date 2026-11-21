// numpy's arithmetic where scipy's Python integrators use it (docs/design/integrate.md §3.3-§3.5,
// decisions C87, C97): the add reduction's pairwise sum and the BLAS dot product of
// np.linalg.norm, so that a sum or a norm that decides a stopping test is numpy's to the bit.
//
//   - pairwise / np_sum: DOUBLE_pairwise_sum (8 accumulators, blocks of 128) and np.sum along a
//     contiguous row, which starts from the reduction's identity 0;
//   - norm2: np.linalg.norm of a float64 vector, sqrt(x.dot(x)): the product of the one element,
//     else 0.0 + Accelerate's cblas_ddot. From 897 elements Accelerate's ddot depends on where the
//     vector is in memory (its address modulo 1024 and more; never below that size for 16-byte
//     aligned vectors, and never between page-aligned ones: measured 2026-11-22), so the vector is
//     copied to a page-aligned buffer first: the result is deterministic and numpy's whenever
//     numpy's array is page-aligned too (numpy's own result moves with its allocations there).
#pragma once

#include "linalg/types.hpp"  // Accelerate's CBLAS (ILP64)

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>

namespace sd::integrate::numpy {

// DOUBLE_pairwise_sum over n doubles
inline double pairwise(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + static_cast<std::size_t>(j)];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

// np.sum along a contiguous row: the reduction starts from add's identity 0
inline double np_sum(const double* a, std::size_t n) { return 0.0 + pairwise(a, n); }

// A page-aligned scratch vector (grows, never shrinks).
class PageBuffer {
public:
    PageBuffer() = default;
    PageBuffer(const PageBuffer&) = delete;
    PageBuffer& operator=(const PageBuffer&) = delete;
    ~PageBuffer() { std::free(p_); }
    double* get(std::size_t n) {
        if (n > cap_) {
            void* q = nullptr;
            const std::size_t bytes = (n * sizeof(double) + 4095) / 4096 * 4096;
            if (posix_memalign(&q, 4096, bytes) != 0) throw std::bad_alloc();
            std::free(p_);
            p_ = static_cast<double*>(q);
            cap_ = bytes / sizeof(double);
        }
        return p_;
    }

private:
    double* p_ = nullptr;
    std::size_t cap_ = 0;
};

// np.linalg.norm(x) of a float64 vector of n >= 1 elements already in buf (a PageBuffer's memory)
inline double norm2_inplace(const double* buf, std::size_t n) {
    if (n == 1) return std::sqrt(buf[0] * buf[0]);
    return std::sqrt(0.0 + cblas_ddot(static_cast<__LAPACK_int>(n), buf, 1, buf, 1));
}

// np.amax(abs(x)): NaN when any element is
inline double maxabs(const double* x, std::size_t n) {
    double m = std::abs(x[0]);
    for (std::size_t i = 1; i < n && !std::isnan(m); ++i) {
        const double v = std::abs(x[i]);
        if (std::isnan(v) || v > m) m = v;
    }
    return m;
}

}  // namespace sd::integrate::numpy
