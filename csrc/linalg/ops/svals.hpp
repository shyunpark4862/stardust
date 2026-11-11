// Singular values only, shared by svdvals, matrix_rank, cond and matrix_norm
// (2, -2, 'nuc'): one routine, so they all see the same numbers
// (docs/design/linalg.md §10.5).
#pragma once

#include "linalg/ops/common.hpp"

#include <algorithm>

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

}  // namespace sd::linalg
