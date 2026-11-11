// Pieces shared by the function implementations.
#pragma once

#include "core/batch.hpp"
#include "core/errors.hpp"
#include "core/input.hpp"
#include "core/pack.hpp"
#include "core/types.hpp"
#include "lapack/lapack.hpp"

#include <nanobind/nanobind.h>

#include <string>
#include <type_traits>

namespace sdla {

namespace nb = nanobind;

// Runs f(std::type_identity<T>) with T = cplx or double.
template <class F>
nb::object with_type(bool complex, F&& f) {
    if (complex) return f(std::type_identity<cplx>{});
    return f(std::type_identity<double>{});
}

// Failure codes of batch bodies (DESIGN.md §2.4, §2.10, §2.12). Within one
// batch element the arguments are checked in order; a matrix's singularity
// right after that matrix.
constexpr int kNonFinite = 1;   // NaN or inf in the first matrix argument
constexpr int kSingular = 2;    // exact zero pivot / zero on a triangular diagonal
constexpr int kNonFiniteB = 3;  // NaN or inf in the right-hand side
constexpr int kNotPD = 4;       // Cholesky: not positive definite
constexpr int kNonFiniteU = 5;  // lu_solve: NaN or inf in 'u'
constexpr int kBadPerm = 6;     // lu_solve: 'perm' is not a permutation
constexpr int kNoConvergence = 7;     // the SVD did not converge
constexpr int kNoConvergenceEig = 8;  // an eigenvalue algorithm did not converge

// Turns a batch failure into the Python exception. `arg` names the first
// matrix argument; the right-hand side is always 'b'.
[[noreturn]] inline void raise(const Batch& b, const Failure& f, const char* func, const char* arg) {
    auto msg = [&](const std::string& what) { return failure_message(b, f.index, (std::string(func) + "(): " + what).c_str()); };
    switch (f.code) {
        case kNonFinite: throw nb::value_error(msg(std::string("'") + arg + "' contains NaN or infinity").c_str());
        case kNonFiniteB: throw nb::value_error(msg("'b' contains NaN or infinity").c_str());
        case kNonFiniteU: throw nb::value_error(msg("'u' contains NaN or infinity").c_str());
        case kBadPerm: throw nb::value_error(msg("'perm' is not a permutation of 0..n-1").c_str());
        case kNotPD: throw LinAlgError(failure_message(b, f.index, "matrix is not positive definite"));
        case kNoConvergence: throw LinAlgError(failure_message(b, f.index, "SVD did not converge"));
        case kNoConvergenceEig:
            throw LinAlgError(failure_message(b, f.index, "eigenvalue algorithm did not converge"));
        default: throw LinAlgError(failure_message(b, f.index, "singular matrix"));
    }
}

// The right-hand side of a solve: a 1-D b is one vector (M,), anything else a
// stack of matrices (..., M, K) (API.md §0.2). Checks M against the matrix size n.
struct Rhs {
    bool vec;
    lint k;            // number of columns (1 for a vector)
    int core;          // 1 or 2 trailing core axes
    std::int64_t rs, cs;  // byte strides of rows and columns
};

inline Rhs rhs_of(const Input& y, lint n, const char* func, const char* aname, bool square = true) {
    const bool vec = y.ndim() == 1;
    const lint rows = vec ? y.shape(0) : y.shape(-2);
    if (rows != n) {
        const std::string what = square ? "is " + std::to_string(n) + "x" + std::to_string(n)
                                        : "has " + std::to_string(n) + " rows";
        throw nb::value_error((std::string(func) + "(): '" + aname + "' " + what + " but 'b' has " +
                               std::to_string(rows) + " rows")
                                  .c_str());
    }
    return {vec, vec ? 1 : y.shape(-1), vec ? 1 : 2, vec ? y.stride(0) : y.stride(-2), vec ? 0 : y.stride(-1)};
}

// Packs the square matrix of `x` at byte offset `off` into scratch and factors
// it with getrf. Returns 0, kNonFinite or kSingular; `a` and `ipiv` point into scratch.
template <class T>
int lu_factor(const Input& x, std::int64_t off, lint n, Scratch& s, T*& a, lint*& ipiv) {
    a = s.get<T>(static_cast<std::size_t>(n * n));
    ipiv = s.get_ints(static_cast<std::size_t>(n));
    if (!pack(x.dtype(), x.data() + off, n, n, x.stride(-2), x.stride(-1), a, n, true)) return kNonFinite;
    if (n == 0) return 0;
    return lapack::getrf(n, n, a, n, ipiv) > 0 ? kSingular : 0;
}

// The inverse from an LU factorization, written to `r` (n x n, column-major):
// getrs against the identity (DESIGN.md §2.10), so inv(a) == solve(a, I) bit for bit.
template <class T>
void lu_inverse(const T* lu, const lint* ipiv, lint n, T* r) {
    if (n == 0) return;
    for (lint k = 0; k < n * n; ++k) r[k] = T(0);
    for (lint k = 0; k < n; ++k) r[k * n + k] = T(1);
    lapack::getrs('N', n, n, lu, n, ipiv, r, n);
}

}  // namespace sdla
