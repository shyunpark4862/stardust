// The element types of every area (DESIGN.md §4.2): the array area uses the
// first four, whose order is also their promotion order (docs/design/array.md
// §2.1); a frame column uses Bool, Int64, Float64 and String.
#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

namespace sd {

// The array dtypes come first, ordered by "kind": promotion of two of them
// is simply the larger one. String (UTF-8, frame columns only) has no fixed
// item size and takes no part in promotion.
enum class DType : std::uint8_t { Bool, Int64, Float64, Complex128, String };

// The array area's dtypes: DType values [0, kNumArrayDTypes).
inline constexpr int kNumArrayDTypes = 4;

constexpr bool is_array_dtype(DType d) { return static_cast<int>(d) < kNumArrayDTypes; }

using complex128 = std::complex<double>;

// Array dtypes only.
constexpr DType promote(DType a, DType b) { return a > b ? a : b; }

constexpr std::int64_t itemsize(DType d) {
    switch (d) {
        case DType::Bool: return 1;
        case DType::Int64: return 8;
        case DType::Float64: return 8;
        case DType::Complex128: return 16;
        case DType::String: return 0;  // the bytes live elsewhere (offsets + UTF-8)
    }
    __builtin_unreachable();
}

constexpr const char* dtype_name(DType d) {
    switch (d) {
        case DType::Bool: return "bool";
        case DType::Int64: return "int64";
        case DType::Float64: return "float64";
        case DType::Complex128: return "complex128";
        case DType::String: return "string";
    }
    __builtin_unreachable();
}

// C++ storage type of each dtype.
template <typename T> inline constexpr DType dtype_of = [] {
    if constexpr (std::is_same_v<T, bool>) return DType::Bool;
    else if constexpr (std::is_same_v<T, std::int64_t>) return DType::Int64;
    else if constexpr (std::is_same_v<T, double>) return DType::Float64;
    else if constexpr (std::is_same_v<T, complex128>) return DType::Complex128;
    else static_assert(!sizeof(T), "not a stardust storage type");
}();

// Runtime array dtype -> compile-time type. Calls f(std::type_identity<T>{})
// where T is the storage type of `d` (an array dtype: callers check first); this is how one templated function body is
// instantiated once per dtype and selected at runtime.
template <typename F>
decltype(auto) visit_dtype(DType d, F&& f) {
    switch (d) {
        case DType::Bool: return f(std::type_identity<bool>{});
        case DType::Int64: return f(std::type_identity<std::int64_t>{});
        case DType::Float64: return f(std::type_identity<double>{});
        case DType::Complex128: return f(std::type_identity<complex128>{});
        case DType::String: break;  // not an array dtype
    }
    __builtin_unreachable();
}

}  // namespace sd
