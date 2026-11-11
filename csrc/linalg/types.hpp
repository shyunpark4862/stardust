// Basic types shared by the whole core.
#pragma once

#include <Accelerate/Accelerate.h>  // ILP64 + new LAPACK via compile definitions (DESIGN.md §1.5)

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

// The four stardust-numpy dtypes, read from the buffer format of an input.
enum class DType : std::uint8_t { Bool, Int64, Float64, Complex128 };

inline std::int64_t itemsize(DType d) {
    switch (d) {
        case DType::Bool: return 1;
        case DType::Complex128: return 16;
        default: return 8;
    }
}

// What a computing function works in for a given input: bool and int64 become
// float64 (DESIGN.md §1.4); the result dtype depends on input dtypes only (API §0.3).
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

// float -> int64 as stardust-numpy converts explicitly (its DESIGN.md S3):
// toward zero, saturating at the int64 range, NaN -> 0.
inline std::int64_t saturate_to_int(double v) {
    if (std::isnan(v)) return 0;
    if (v >= 9223372036854775808.0) return std::numeric_limits<std::int64_t>::max();
    if (v < -9223372036854775808.0) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(v);
}

// Converts a stored value S to the computing type T (double, cplx or int64).
// complex -> real never happens: callers reject it before (S4).
template <class T, class S>
inline T convert(S v) {
    if constexpr (std::is_same_v<S, std::uint8_t>) return T(v != 0 ? 1 : 0);
    else if constexpr (std::is_same_v<T, std::int64_t>) {
        if constexpr (std::is_same_v<S, double>) return saturate_to_int(v);
        else return static_cast<std::int64_t>(v);
    } else if constexpr (std::is_same_v<S, std::int64_t>) return T(static_cast<double>(v));
    else return T(v);  // double -> double/complex, complex -> complex
}

// Products of single elements, written exactly as stardust-numpy (and numpy)
// compute them, so elementwise results agree bit for bit (stardust-numpy
// scalar_ops.hpp cmul): numpy is built with FMA contraction, which turns
// a*b - c*d into fma(a, b, -(c*d)). std::complex's operator* does not.
inline double mul(double a, double b) { return a * b; }
inline cplx mul(cplx a, cplx b) {
    const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    return {std::fma(ar, br, -(ai * bi)), std::fma(ar, bi, ai * br)};
}

}  // namespace sd::linalg
