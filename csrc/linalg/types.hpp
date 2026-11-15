// Basic types shared by the whole core.
#pragma once

#include "base/saturate.hpp"  // saturate_to_int64: float -> int64, the rule of every area
#include "base/dtype.hpp"

#include <Accelerate/Accelerate.h>  // ILP64 + new LAPACK via compile definitions (docs/design/linalg.md §1.2)

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace sd::linalg {

using cplx = std::complex<double>;
using lint = __LAPACK_int;  // 64-bit LAPACK integer
static_assert(sizeof(lint) == 8, "ACCELERATE_LAPACK_ILP64 must be defined");

// DType and itemsize are the library's (base/dtype.hpp): an input's dtype is
// its NDArray's.

// What a computing function works in for a given input: bool and int64 become
// float64 (docs/design/linalg.md §2.1); the result dtype depends on input dtypes
// only (docs/api/linalg.md §1.5).
inline bool computes_complex(DType d) { return d == DType::Complex128; }

template <class T>
constexpr DType dtype_of() {
    if constexpr (std::is_same_v<T, cplx>) return DType::Complex128;
    else if constexpr (std::is_same_v<T, double>) return DType::Float64;
    else if constexpr (std::is_same_v<T, std::int64_t>) return DType::Int64;
    else return DType::Bool;
}

// The C++ type that stores each dtype in memory.
template <class F>
decltype(auto) visit_storage(DType d, F&& f) {
    switch (d) {
        case DType::Bool: return f(std::type_identity<std::uint8_t>{});
        case DType::Int64: return f(std::type_identity<std::int64_t>{});
        case DType::Float64: return f(std::type_identity<double>{});
        default: return f(std::type_identity<cplx>{});
    }
}

template <class S>
inline S load(const std::byte* p) {
    S v;
    std::memcpy(&v, p, sizeof(S));
    return v;
}

// Converts a stored value S to the computing type T (double, cplx or int64).
// complex -> real never happens: callers reject it before (docs/design/array.md §2.4).
template <class T, class S>
inline T convert(S v) {
    if constexpr (std::is_same_v<S, std::uint8_t>) return T(v != 0 ? 1 : 0);
    else if constexpr (std::is_same_v<T, std::int64_t>) {
        if constexpr (std::is_same_v<S, double>) return saturate_to_int64(v);  // the rule of every area (base/saturate.hpp)
        else return static_cast<std::int64_t>(v);
    } else if constexpr (std::is_same_v<S, std::int64_t>) return T(static_cast<double>(v));
    else return T(v);  // double -> double/complex, complex -> complex
}

// Products of single elements, written exactly as the array area (and numpy)
// compute them, so elementwise results agree bit for bit (cmul in
// array/kernels/scalar_ops.hpp, docs/design/array.md §5.3): numpy is built with
// FMA contraction, which turns a*b - c*d into fma(a, b, -(c*d)). std::complex's
// operator* does not.
inline double mul(double a, double b) { return a * b; }
inline cplx mul(cplx a, cplx b) {
    const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    return {std::fma(ar, br, -(ai * bi)), std::fma(ar, bi, ai * br)};
}

}  // namespace sd::linalg
