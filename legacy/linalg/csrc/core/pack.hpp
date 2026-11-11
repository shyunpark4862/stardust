// Packing: one strided input matrix -> an F-order work buffer, in one pass
// that also converts the dtype and checks finiteness (ARCHITECTURE.md §3.2).
#pragma once

#include "core/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sdla {

inline bool finite(double v) { return std::isfinite(v); }
inline bool finite(cplx v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

namespace detail {

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
    return visit_storage(src, [&]<class S>(std::type_identity<S>) {
        if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) return false;  // unreachable
        else return detail::pack_loop<T, S>(p, m, n, rs, cs, dst, ld, check);
    });
}

// Like pack, for an n x n matrix of which only one triangle is read: lower
// (i >= j) or upper (i <= j), with or without the diagonal. Only those
// elements are copied and checked (DESIGN.md §2.12); the rest of `dst` is
// left as it was, since the LAPACK routine that follows does not read it.
template <class T>
bool pack_triangle(DType src, const std::byte* p, std::int64_t n, std::int64_t rs, std::int64_t cs, T* dst,
                   std::int64_t ld, bool lower, bool with_diag, bool check) {
    return visit_storage(src, [&]<class S>(std::type_identity<S>) {
        if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) {
            return false;  // unreachable
        } else {
            bool ok = true;
            for (std::int64_t j = 0; j < n; ++j) {
                const std::int64_t i0 = lower ? (with_diag ? j : j + 1) : 0;
                const std::int64_t i1 = lower ? n : (with_diag ? j + 1 : j);
                const std::byte* col = p + j * cs;
                for (std::int64_t i = i0; i < i1; ++i) {
                    const S v = load<S>(col + i * rs);
                    if constexpr (std::is_same_v<S, double> || std::is_same_v<S, cplx>) {
                        if (check) ok &= finite(v);
                    }
                    dst[j * ld + i] = convert<T>(v);
                }
            }
            return ok;
        }
    });
}

// True if an n x n column-major matrix has an exact zero on its diagonal.
template <class T>
bool zero_on_diagonal(const T* a, std::int64_t n, std::int64_t ld) {
    for (std::int64_t i = 0; i < n; ++i)
        if (a[i * ld + i] == T(0)) return true;
    return false;
}

}  // namespace sdla
