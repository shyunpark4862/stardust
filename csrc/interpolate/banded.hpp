// The tridiagonal solve of sd.interpolate's splines, without pivoting (docs/design/interpolate.md §3.3;
// decision IP16): one factorization A = LU of the system's matrix, shared by every column of a batch, then a
// forward and a back substitution per column. The columns of a row are contiguous (the right side is (n, K)
// row-major), so a substitution step runs over a row's columns at once.
//
// The systems that use it (CubicSpline's slopes, cubic.cpp) have nonnegative entries and positive pivots in
// exact arithmetic, so L >= 0 and U >= 0 and |L||U| = |LU| = |A|: the componentwise backward error of the
// solve is a few units of roundoff times |A| itself (the proof, row by row, is docs/design/interpolate.md §3.3).
// factor() reports the first pivot that is not positive (or not finite): in floating point that happens
// only for a not-a-knot end whose two last spacings differ by more than about 2^50, where the matrix of the
// rounded entries is itself singular to working precision.
//
//   l[i] = a[i] / u[i - 1]          i >= 1 (a: the subdiagonal, l overwrites it)
//   u[i] = b[i] - l[i] c[i - 1]     (b: the diagonal, u overwrites it; c: the superdiagonal)
//   w[i] = 1 / u[i]
//   forward  r[i] -= l[i] r[i - 1]
//   back     r[n - 1] *= w[n - 1];  r[i] = (r[i] - c[i] r[i + 1]) w[i]
//
// The factorization's chain needs a division per row; the back substitution multiplies by the reciprocals,
// which the factorization takes off its chain: a long column is bound by the latency of these chains, and a
// division in both was 1.2x the time (one more rounding per row, counted in §3.3's proof). One rounding per
// operation: the build compiles with -ffp-contract=off (CMakeLists.txt).
#pragma once

#include <cmath>
#include <cstdint>

namespace sd::interpolate {

// Factors in place: a[1 .. n) becomes l, b[0 .. n) becomes u, and w[0 .. n) gets 1 / u. Returns -1, or the
// first row whose pivot is not a positive finite number.
inline std::int64_t tridiagonal_factor(double* a, double* b, const double* c, double* w, std::int64_t n) {
    if (!(b[0] > 0.0 && std::isfinite(b[0]))) return 0;
    w[0] = 1.0 / b[0];
    for (std::int64_t i = 1; i < n; ++i) {
        a[i] = a[i] / b[i - 1];
        b[i] = b[i] - a[i] * c[i - 1];
        if (!(b[i] > 0.0 && std::isfinite(b[i]))) return i;
        w[i] = 1.0 / b[i];
    }
    return -1;
}

// Solves LU X = R in place for the columns [c0, c1) of R (n, K) row-major (w: the pivots' reciprocals).
inline void tridiagonal_solve(
    const double* l,
    const double* w,
    const double* c,
    std::int64_t n,
    double* R,
    std::int64_t K,
    std::int64_t c0,
    std::int64_t c1
) {
    for (std::int64_t i = 1; i < n; ++i) {
        double* r = R + i * K;
        const double* p = r - K;
        const double li = l[i];
        for (std::int64_t j = c0; j < c1; ++j) r[j] = r[j] - li * p[j];
    }
    {
        double* r = R + (n - 1) * K;
        const double wn = w[n - 1];
        for (std::int64_t j = c0; j < c1; ++j) r[j] = r[j] * wn;
    }
    for (std::int64_t i = n - 2; i >= 0; --i) {
        double* r = R + i * K;
        const double* q = r + K;
        const double ci = c[i], wi = w[i];
        for (std::int64_t j = c0; j < c1; ++j) r[j] = (r[j] - ci * q[j]) * wi;
    }
}

}  // namespace sd::interpolate
