// Packing: one strided input matrix -> an F-order work buffer, in one pass
// that also converts the dtype and checks finiteness (docs/design/linalg.md §5).
#pragma once

#include "linalg/batch.hpp"
#include "linalg/types.hpp"

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace sd::linalg {

inline bool finite(double v) { return std::isfinite(v); }
inline bool finite(cplx v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

namespace detail {

// True if no element of v[0..n) is NaN or inf: the exponent field is all ones
// exactly for those. Integer operations only, so the loop vectorizes (a
// floating-point test would be an ordered reduction under -ffp-contract=off).
inline bool all_finite(const double* v, std::int64_t n) {
    std::uint64_t bad = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint64_t u;
        std::memcpy(&u, v + i, sizeof u);
        bad |= static_cast<std::uint64_t>((u & 0x7ff0000000000000ull) == 0x7ff0000000000000ull);
    }
    return bad == 0;
}

// One contiguous run of n elements: src -> dst, converting, and checking
// finiteness when `check` (float sources only). Same-type runs copy and check
// in one pass, on the bits (vectorizes).
template <class T, class S>
bool copy_run(const std::byte* src, std::int64_t n, T* dst, bool check) {
    if constexpr (std::is_same_v<S, T>) {
        const std::int64_t nd = n * static_cast<std::int64_t>(sizeof(T) / sizeof(double));
        double* d = reinterpret_cast<double*>(dst);
        std::uint64_t bad = 0;
        for (std::int64_t i = 0; i < nd; ++i) {
            std::uint64_t u;
            std::memcpy(&u, src + i * 8, sizeof u);
            bad |= static_cast<std::uint64_t>((u & 0x7ff0000000000000ull) == 0x7ff0000000000000ull);
            std::memcpy(d + i, &u, sizeof u);
        }
        return !check || bad == 0;
    } else {
        bool ok = true;
        for (std::int64_t i = 0; i < n; ++i) {
            const S v = load<S>(src + i * static_cast<std::int64_t>(sizeof(S)));
            if constexpr (std::is_same_v<S, double> || std::is_same_v<S, cplx>) {
                if (check) ok &= finite(v);
            }
            dst[i] = convert<T>(v);
        }
        return ok;
    }
}

// The loop for one source type S, for destination columns [j0, j1). Only
// float sources can hold NaN or inf. Fast paths (docs/design/linalg.md §5.3):
// columns contiguous in the source (F-order: a copy per column) and rows
// contiguous (C-order: a transpose in 32 x 32 tiles, so both the reads and the
// writes stay within a few cache lines). The result is the same for every
// path: element (i, j) goes to dst[j * ld + i].
template <class T, class S>
bool pack_loop(const std::byte* p, std::int64_t m, std::int64_t j0, std::int64_t j1, std::int64_t rs, std::int64_t cs,
               T* dst, std::int64_t ld, bool check) {
    constexpr std::int64_t sz = sizeof(S);
    bool ok = true;
    if (rs == sz) {
        for (std::int64_t j = j0; j < j1; ++j) ok &= copy_run<T, S>(p + j * cs, m, dst + j * ld, check);
        return ok;
    }
    if (cs == sz && m > 1) {
        constexpr std::int64_t tile = 32;
        for (std::int64_t i0 = 0; i0 < m; i0 += tile) {
            const std::int64_t i1 = std::min(m, i0 + tile);
            for (std::int64_t jt = j0; jt < j1; jt += tile) {
                const std::int64_t jt1 = std::min(j1, jt + tile);
                for (std::int64_t j = jt; j < jt1; ++j)
                    for (std::int64_t i = i0; i < i1; ++i) dst[j * ld + i] = convert<T>(load<S>(p + i * rs + j * sz));
            }
        }
        if constexpr (std::is_same_v<S, double> || std::is_same_v<S, cplx>) {
            if (check)
                for (std::int64_t j = j0; j < j1; ++j)
                    ok &= all_finite(reinterpret_cast<const double*>(dst + j * ld),
                                     m * static_cast<std::int64_t>(sizeof(T) / sizeof(double)));
        }
        return ok;
    }
    for (std::int64_t j = j0; j < j1; ++j) {
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

// From this many elements on, one C-order matrix is transposed into its work
// buffer on the thread pool, in blocks of destination columns.
// Measured (docs/design/linalg.md §5.6, solve_triangular of a C-order
// matrix): 1.25x at 256^2, 1.4x at 362^2, 1.5x at 1000^2, 2x at 2000^2; but
// inv lost 7-11% at 181^2-300^2 (and broke even at 362^2), so the line is at
// 2^17 = 131072 elements (362^2 = 131044 is just below it; square matrices
// reach it from 363^2). A variable only so that measurements can move it
// (_core._set_pack_parallel_min).
inline std::int64_t kPackParallelMin = 1 << 17;

// Runs block(j0, j1) -> bool (finite) over blocks of 32 destination columns
// on the pool. Every block writes only its own columns, so the result is the
// serial one. Everything for large matrices is out of line (pack_large,
// pack_triangle_large): with the split inside the small-matrix path, small
// calls got 3-10 ns slower each (solve_tridiagonal (1000,3,3): 36 -> 49 us on
// one thread). Now the only addition there is one size test before the
// dtype dispatch: +1-8% over before, at most 3 ns a call
// (docs/design/linalg.md §5.6).
template <class Block>
[[gnu::noinline]] bool on_pool_by_columns(std::int64_t n, const Block& block) {
    constexpr std::int64_t kCols = 32;
    if (n <= kCols || !begin_parallel()) return block(std::int64_t(0), n);
    const std::int64_t ntasks = (n + kCols - 1) / kCols;
    std::vector<unsigned char> ok(static_cast<std::size_t>(ntasks), 1);
    run_tasks(ntasks, [&](std::int64_t t) {
        ok[static_cast<std::size_t>(t)] = block(t * kCols, std::min(n, (t + 1) * kCols));
    });
    return std::all_of(ok.begin(), ok.end(), [](unsigned char v) { return v != 0; });
}

// pack() for a large matrix, out of line: a C-order source is transposed on
// the pool in blocks of destination columns; everything else as in pack().
// Only the transpose gained from the pool (strided inputs got slower,
// docs/design/linalg.md §5.6).
template <class T>
[[gnu::noinline]] bool pack_large(DType src, const std::byte* p, std::int64_t m, std::int64_t n, std::int64_t rs,
                                  std::int64_t cs, T* dst, std::int64_t ld, bool check) {
    return visit_storage(src, [&]<class S>(std::type_identity<S>) {
        if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) {
            return false;  // unreachable
        } else {
            if (cs == std::int64_t(sizeof(S)) && rs != cs && m > 1)
                return on_pool_by_columns(n, [&](std::int64_t j0, std::int64_t j1) {
                    return pack_loop<T, S>(p, m, j0, j1, rs, cs, dst, ld, check);
                });
            return pack_loop<T, S>(p, m, 0, n, rs, cs, dst, ld, check);
        }
    });
}

}  // namespace detail

// Copies the m x n matrix at `p` (row stride `rs`, column stride `cs`, in
// bytes, any sign, 0 allowed) into `dst` in column-major order with leading
// dimension `ld`, converting `src` to T. With `check`, returns false if any
// element is NaN or inf (the whole matrix is still copied).
//
// T is double or cplx; a complex source never meets a double destination,
// because the destination type comes from promoting all inputs (docs/api/linalg.md §1.5).
template <class T>
bool pack(DType src, const std::byte* p, std::int64_t m, std::int64_t n, std::int64_t rs, std::int64_t cs, T* dst,
          std::int64_t ld, bool check) {
    // A large matrix goes to the out-of-line entry before anything else, so the
    // small-matrix path below stays as it was (the test inside the dispatch
    // cost small calls ~2.5 ns each, docs/design/linalg.md §5.6).
    if (m * n >= detail::kPackParallelMin) [[unlikely]]
        return detail::pack_large(src, p, m, n, rs, cs, dst, ld, check);
    return visit_storage(src, [&]<class S>(std::type_identity<S>) {
        if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) return false;  // unreachable
        else return detail::pack_loop<T, S>(p, m, 0, n, rs, cs, dst, ld, check);
    });
}

namespace detail {

// pack_triangle's loop for one source type S, destination columns [j0, j1).
template <class T, class S>
bool pack_triangle_loop(const std::byte* p, std::int64_t n, std::int64_t j0, std::int64_t j1, std::int64_t rs,
                        std::int64_t cs, T* dst, std::int64_t ld, bool lower, bool with_diag, bool check) {
    bool ok = true;
    if (rs == static_cast<std::int64_t>(sizeof(S))) {  // columns contiguous: a run per column (docs/design/linalg.md §5.3)
        for (std::int64_t j = j0; j < j1; ++j) {
            const std::int64_t i0 = lower ? (with_diag ? j : j + 1) : 0;
            const std::int64_t i1 = lower ? n : (with_diag ? j + 1 : j);
            if (i1 > i0)
                ok &= detail::copy_run<T, S>(p + j * cs + i0 * rs, i1 - i0, dst + j * ld + i0, check);
        }
        return ok;
    }
    if (cs == static_cast<std::int64_t>(sizeof(S)) && n > 1) {
        // rows contiguous (C-order): a transpose in 32 x 32 tiles, then the
        // finiteness check on the written runs
        constexpr std::int64_t tile = 32;
        for (std::int64_t r0 = 0; r0 < n; r0 += tile) {
            const std::int64_t r1 = std::min(n, r0 + tile);
            for (std::int64_t jt = j0; jt < j1; jt += tile) {
                const std::int64_t jt1 = std::min(j1, jt + tile);
                for (std::int64_t j = jt; j < jt1; ++j) {
                    // this column's rows of the triangle, clipped to the tile
                    const std::int64_t a = std::max(r0, lower ? (with_diag ? j : j + 1) : 0);
                    const std::int64_t e = std::min(r1, lower ? n : (with_diag ? j + 1 : j));
                    for (std::int64_t i = a; i < e; ++i)
                        dst[j * ld + i] = convert<T>(load<S>(p + i * rs + j * cs));
                }
            }
        }
        if constexpr (std::is_same_v<S, double> || std::is_same_v<S, cplx>) {
            if (check)
                for (std::int64_t j = j0; j < j1; ++j) {
                    const std::int64_t i0 = lower ? (with_diag ? j : j + 1) : 0;
                    const std::int64_t i1 = lower ? n : (with_diag ? j + 1 : j);
                    if (i1 > i0)
                        ok &= detail::all_finite(reinterpret_cast<const double*>(dst + j * ld + i0),
                                                 (i1 - i0) * static_cast<std::int64_t>(sizeof(T) / sizeof(double)));
                }
        }
        return ok;
    }
    for (std::int64_t j = j0; j < j1; ++j) {
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

// pack_triangle() for a large matrix, out of line (as pack_large).
template <class T>
[[gnu::noinline]] bool pack_triangle_large(DType src, const std::byte* p, std::int64_t n, std::int64_t rs,
                                           std::int64_t cs, T* dst, std::int64_t ld, bool lower, bool with_diag,
                                           bool check) {
    return visit_storage(src, [&]<class S>(std::type_identity<S>) {
        if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) {
            return false;  // unreachable
        } else {
            if (cs == std::int64_t(sizeof(S)) && rs != cs && n > 1)
                return on_pool_by_columns(n, [&](std::int64_t j0, std::int64_t j1) {
                    return pack_triangle_loop<T, S>(p, n, j0, j1, rs, cs, dst, ld, lower, with_diag, check);
                });
            return pack_triangle_loop<T, S>(p, n, 0, n, rs, cs, dst, ld, lower, with_diag, check);
        }
    });
}

}  // namespace detail

// Like pack, for an n x n matrix of which only one triangle is read: lower
// (i >= j) or upper (i <= j), with or without the diagonal. Only those
// elements are copied and checked (docs/design/linalg.md §5.4); the rest of
// `dst` is left as it was, since the LAPACK routine that follows does not read it.
template <class T>
bool pack_triangle(DType src, const std::byte* p, std::int64_t n, std::int64_t rs, std::int64_t cs, T* dst,
                   std::int64_t ld, bool lower, bool with_diag, bool check) {
    if (n * n >= detail::kPackParallelMin) [[unlikely]]  // as in pack()
        return detail::pack_triangle_large(src, p, n, rs, cs, dst, ld, lower, with_diag, check);
    return visit_storage(src, [&]<class S>(std::type_identity<S>) {
        if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) return false;  // unreachable
        else return detail::pack_triangle_loop<T, S>(p, n, 0, n, rs, cs, dst, ld, lower, with_diag, check);
    });
}

// For an operand LAPACK reads in place (docs/design/linalg.md §5.5): true if
// the triangle that pack_triangle would read has no NaN or inf. a is
// column-major T with leading dimension ld.
template <class T>
bool triangle_finite(const T* a, std::int64_t n, std::int64_t ld, bool lower, bool with_diag) {
    bool ok = true;
    for (std::int64_t j = 0; j < n; ++j) {
        const std::int64_t i0 = lower ? (with_diag ? j : j + 1) : 0;
        const std::int64_t i1 = lower ? n : (with_diag ? j + 1 : j);
        if (i1 > i0)
            ok &= detail::all_finite(reinterpret_cast<const double*>(a + j * ld + i0),
                                     (i1 - i0) * static_cast<std::int64_t>(sizeof(T) / sizeof(double)));
    }
    return ok;
}

// True if an n x n column-major matrix has an exact zero on its diagonal.
template <class T>
bool zero_on_diagonal(const T* a, std::int64_t n, std::int64_t ld) {
    for (std::int64_t i = 0; i < n; ++i)
        if (a[i * ld + i] == T(0)) return true;
    return false;
}

}  // namespace sd::linalg
