// Cast kernels: one inner loop per (from, to) dtype pair, 4 x 4 = 16 loops,
// generated from a single template and looked up in a table at runtime.
//
// Contiguous runs take an indexed loop the compiler can vectorize
// (docs/design/array.md §5.1); anything else the general strided loop.
#pragma once

#include "array/convert.hpp"
#include "array/iter.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <type_traits>

namespace sd {

// args[0] = source, args[1] = destination.
template <typename From, typename To>
void cast_loop(std::byte* const* args, std::int64_t n, const std::int64_t* strides, void*) {
    if (strides[0] == sizeof(From) && strides[1] == sizeof(To)) {  // contiguous
        if constexpr (std::is_same_v<From, To>) {
            // A plain copy: memcpy uses the widest loads/stores the CPU has
            // and is tuned per chip by the OS (docs/design/array.md §5.1).
            std::memcpy(args[1], args[0], static_cast<size_t>(n) * sizeof(To));
            return;
        }
        const From* src = reinterpret_cast<const From*>(args[0]);
        To* dst = reinterpret_cast<To*>(args[1]);
        std::int64_t i = 0;
        if constexpr (std::is_same_v<From, double> && std::is_same_v<To, std::int64_t>) {
            // float -> int64 (docs/design/array.md §2.4) is the vector FCVTZS, two
            // lanes at a time.
            for (; i + 4 <= n; i += 4) {
                vst1q_s64(dst + i, vcvtq_s64_f64(vld1q_f64(src + i)));
                vst1q_s64(dst + i + 2, vcvtq_s64_f64(vld1q_f64(src + i + 2)));
            }
        }
        for (; i < n; ++i) dst[i] = cast_value<To>(src[i]);
        return;
    }
    if (strides[0] == 0) {
        // A broadcast scalar (x[...] = 1.0, full, the fill of zeros_like and so on):
        // convert once, then a plain store loop. Reading the source in every step
        // kept the loop scalar -- the store may alias it (10^6 elements 248 us
        // against numpy's 63).
        const To v = cast_value<To>(*reinterpret_cast<const From*>(args[0]));
        if (strides[1] == sizeof(To)) {
            To* dst = reinterpret_cast<To*>(args[1]);
            std::fill(dst, dst + n, v);
        } else {
            std::byte* dst = args[1];
            for (std::int64_t i = 0; i < n; ++i, dst += strides[1]) std::memcpy(dst, &v, sizeof(To));
        }
        return;
    }
    const std::byte* src = args[0];
    std::byte* dst = args[1];
    for (std::int64_t i = 0; i < n; ++i, src += strides[0], dst += strides[1])
        *reinterpret_cast<To*>(dst) = cast_value<To>(*reinterpret_cast<const From*>(src));
}

namespace detail {
template <typename From>
inline constexpr std::array<Loop, kNumArrayDTypes> cast_row = {
    cast_loop<From, bool>, cast_loop<From, std::int64_t>, cast_loop<From, double>, cast_loop<From, complex128>};
inline constexpr std::array<std::array<Loop, kNumArrayDTypes>, kNumArrayDTypes> cast_table = {
    cast_row<bool>, cast_row<std::int64_t>, cast_row<double>, cast_row<complex128>};
}  // namespace detail

// complex -> real entries exist in the table but are unreachable:
// check_castable runs before any kernel is chosen.
inline Loop cast_kernel(DType from, DType to) {
    return detail::cast_table[static_cast<int>(from)][static_cast<int>(to)];
}

}  // namespace sd
