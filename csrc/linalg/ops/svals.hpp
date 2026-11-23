// Singular values only, shared by svdvals, matrix_rank, cond, matrix_norm
// (2, -2, 'nuc') and lstsq (rtol below eps): one routine, so they all see the
// same numbers (docs/design/linalg.md §10.5). normal_rank and add_shifted
// below are pinv's and lstsq's.
#pragma once

#include "linalg/ops/common.hpp"

#include <algorithm>
#include <vector>

namespace sd::linalg {

// The min(m, n) singular values (descending) of the m x n matrix of `x` at byte
// offset `off`, written to `out`. Returns 0, kNonFinite or kNoConvergence.
// Values-only mode (gesvd with no vectors; measured fastest for values,
// docs/design/linalg.md §10.1).
template <class T>
int singular_values(const Input& x, std::int64_t off, lint m, lint n, Scratch& s, double* out) {
    const lint lda = std::max<lint>(m, 1);
    T* a = s.get<T>(static_cast<std::size_t>(lda * n), 3);
    if (!pack(x.dtype(), x.data() + off, m, n, x.stride(-2), x.stride(-1), a, lda, true)) return kNonFinite;
    if (m == 0 || n == 0) return 0;
    return lapack::gesvd_values(m, n, a, lda, out, s) > 0 ? kNoConvergence : 0;
}

// Products with V S^-1 U^H (pinv, lstsq's SVD solve) where a kept singular value has no
// double reciprocal: below 2^-1022 1/sigma overflows to inf, and inf times an exact zero of a
// singular vector is NaN where the exact entry is 0 (pinv(diag(1, 5e-324)) was
// [[nan, nan], [nan, inf]]). Those terms are formed with sigma 2^600 instead (exact: a power
// of two; 1/(sigma 2^600) is at most 2^474), summed apart, and scaled back by 2^600 at the
// end: an exact zero stays zero, an entry beyond the doubles is +-inf as rounding the exact
// value gives, and the other terms keep their bits.
inline constexpr double kTinySingular = 0x1p-1022, kSingularShift = 0x1p600;

// How many of the first `rank` singular values (descending) have a double reciprocal.
inline lint normal_rank(const double* sv, lint rank) {
    lint r = 0;
    while (r < rank && sv[r] >= kTinySingular) ++r;
    return r;
}

// out (n x k, column-major) += shift * part, skipping exact zeros of `part` (so a -0.0 of
// `out` keeps its sign and a 0 term adds nothing).
template <class T>
void add_shifted(T* out, const T* part, lint count) {
    for (lint i = 0; i < count; ++i)
        if (part[i] != T(0)) out[i] += part[i] * kSingularShift;
}

}  // namespace sd::linalg
