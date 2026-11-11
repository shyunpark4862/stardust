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

// Failure codes of batch bodies (DESIGN.md §2.4, §2.10).
constexpr int kNonFinite = 1;   // NaN or inf in the first matrix argument
constexpr int kSingular = 2;    // exact zero pivot
constexpr int kNonFiniteB = 3;  // NaN or inf in the second argument

// Turns a batch failure into the Python exception.
[[noreturn]] inline void raise(const Batch& b, const Failure& f, const char* func, const char* arg,
                               const char* arg_b = "b") {
    if (f.code == kNonFinite || f.code == kNonFiniteB) {
        const std::string msg =
            std::string(func) + "(): '" + (f.code == kNonFinite ? arg : arg_b) + "' contains NaN or infinity";
        throw nb::value_error(failure_message(b, f.index, msg.c_str()).c_str());
    }
    throw LinAlgError(failure_message(b, f.index, "singular matrix"));
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
