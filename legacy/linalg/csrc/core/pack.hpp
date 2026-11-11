// Packing: one strided input matrix -> an F-order work buffer, in one pass
// that also converts the dtype and checks finiteness (ARCHITECTURE.md §3.2).
#pragma once

#include "core/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace sdla {

namespace detail {

template <class S>
inline S load(const std::byte* p) {
    S v;
    std::memcpy(&v, p, sizeof(S));
    return v;
}

template <class T, class S>
inline T convert(S v) {
    if constexpr (std::is_same_v<S, std::uint8_t>) return T(v != 0 ? 1.0 : 0.0);
    else if constexpr (std::is_same_v<S, std::int64_t>) return T(static_cast<double>(v));
    else return T(v);  // double -> double/complex, complex -> complex
}

inline bool finite(double v) { return std::isfinite(v); }
inline bool finite(cplx v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

// The loop for one source type S. Only float sources can hold NaN or inf.
template <class T, class S>
bool pack_loop(const std::byte* p, std::int64_t m, std::int64_t n, std::int64_t rs, std::int64_t cs, T* dst,
               std::int64_t ld, bool check) {
    bool ok = true;
    for (std::int64_t j = 0; j < n; ++j) {
        const std::byte* col = p + j * cs;
        T* out = dst + j * ld;
        for (std::int64_t i = 0; i < m; ++i) {
            const S v = load<S>(col + i * rs);
            if constexpr (std::is_same_v<S, double> || std::is_same_v<S, cplx>) {
                if (check) ok &= finite(v);
            }
            out[i] = convert<T>(v);
        }
    }
    return ok;
}

}  // namespace detail

// Copies the m x n matrix at `p` (row stride `rs`, column stride `cs`, in
// bytes, any sign, 0 allowed) into `dst` in column-major order with leading
// dimension `ld`, converting `src` to T. With `check`, returns false if any
// element is NaN or inf (the whole matrix is still copied).
//
// T is double or cplx; a complex source never meets a double destination,
// because the destination type comes from promoting all inputs (API §0.3).
template <class T>
bool pack(DType src, const std::byte* p, std::int64_t m, std::int64_t n, std::int64_t rs, std::int64_t cs, T* dst,
          std::int64_t ld, bool check) {
    switch (src) {
        case DType::Bool: return detail::pack_loop<T, std::uint8_t>(p, m, n, rs, cs, dst, ld, check);
        case DType::Int64: return detail::pack_loop<T, std::int64_t>(p, m, n, rs, cs, dst, ld, check);
        case DType::Float64: return detail::pack_loop<T, double>(p, m, n, rs, cs, dst, ld, check);
        case DType::Complex128:
            if constexpr (std::is_same_v<T, cplx>) return detail::pack_loop<T, cplx>(p, m, n, rs, cs, dst, ld, check);
            break;
    }
    return false;  // unreachable: complex into double
}

}  // namespace sdla
