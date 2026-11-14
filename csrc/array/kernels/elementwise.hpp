// Inner loops for the operations in scalar_ops.hpp, and the per-operation
// loop tables (docs/design/array.md §4.1: table[op][dtype]).
//
// One template per arity turns Op::f into a 1-D strided loop with the Loop
// kernel signature. Each loop first checks for the common layouts --
// everything contiguous, or one operand a broadcast scalar (stride 0) -- and
// runs those as plain indexed loops, which the compiler can vectorize; the
// fully general strided loop is the fallback (docs/design/array.md §5.1).
#pragma once

#include "array/iter.hpp"
#include "array/kernels/neon.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/kernels/vforce.hpp"

#include <array>
#include <type_traits>
#include <utility>

namespace sd {

template <typename T>
inline T load_as(const std::byte* p) { return *reinterpret_cast<const T*>(p); }

template <typename Op, typename T>
using unary_result_t = decltype(Op::f(std::declval<T>()));
template <typename Op, typename T>
using binary_result_t = decltype(Op::f(std::declval<T>(), std::declval<T>()));

// In place (out is exactly an input: x += y, sd.sin(x, out=x)): the plain indexed
// loop below gets a runtime overlap check from the compiler, which sees out == x as an
// overlap and falls back to scalar code (add in place: 260 us for 10^6 against 185 into
// another array). Read a block, compute, then write it: correct for out == x (each
// element is read before its own place is written, and blocks do not interact), and
// free of the check. Partial overlaps never reach the loops: they are copied first
// (docs/design/array.md §3.5).
inline constexpr int kInPlaceBlock = 16;
template <typename F, typename R, typename... P>
inline void in_place_blocks(R* o, std::int64_t n, F&& f, const P*... in) {
    std::int64_t i = 0;
    for (; i + kInPlaceBlock <= n; i += kInPlaceBlock) {
        R v[kInPlaceBlock];
        for (int k = 0; k < kInPlaceBlock; ++k) v[k] = f(in[i + k]...);
        for (int k = 0; k < kInPlaceBlock; ++k) o[i + k] = v[k];
    }
    for (; i < n; ++i) o[i] = f(in[i]...);
}

// args: x, out
template <typename Op, typename T>
void unary_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    using R = unary_result_t<Op, T>;
    if constexpr (ops::is_float<T> && vforce_unary<Op> != nullptr) {  // Accelerate vForce, any layout
        vforce_unary_run(vforce_unary<Op>, args[0], s[0], args[1], s[1], n);
        return;
    }
    if (s[0] == sizeof(T) && s[1] == sizeof(R)) {
        // Contiguous: indexed pointers instead of byte strides. Same function,
        // but now the compiler sees a[i] -> o[i] and can use vector registers.
        const T* x = reinterpret_cast<const T*>(args[0]);
        R* o = reinterpret_cast<R*>(args[1]);
        if constexpr (std::is_same_v<R, T>) {
            if (static_cast<const void*>(o) == static_cast<const void*>(x)) {
                in_place_blocks(o, n, [](T v) { return Op::f(v); }, x);
                return;
            }
        }
        for (std::int64_t i = 0; i < n; ++i) o[i] = Op::f(x[i]);
        return;
    }
    const std::byte* x = args[0];
    std::byte* o = args[1];
    for (std::int64_t i = 0; i < n; ++i, x += s[0], o += s[1])
        *reinterpret_cast<R*>(o) = Op::f(load_as<T>(x));
}

// args: a, b, out
template <typename Op, typename T>
void binary_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    using R = binary_result_t<Op, T>;
    if constexpr (ops::is_float<T> && vforce_binary<Op> != nullptr) {  // pow, atan2 through vForce
        vforce_binary_run(vforce_binary<Op>, args[0], s[0], args[1], s[1], args[2], s[2], n);
        return;
    }
    if (s[2] == sizeof(R)) {
        const T* a = reinterpret_cast<const T*>(args[0]);
        const T* b = reinterpret_cast<const T*>(args[1]);
        R* o = reinterpret_cast<R*>(args[2]);
        if (s[0] == sizeof(T) && s[1] == sizeof(T)) {  // array op array
            if constexpr (neon_binary<Op, T>::available) {
                neon_binary<Op, T>::run(a, b, o, n);  // hand-written NEON (docs/design/array.md §5.2)
            } else {
                bool same = false;
                if constexpr (std::is_same_v<R, T>)
                    same = static_cast<const void*>(o) == static_cast<const void*>(a) ||
                           static_cast<const void*>(o) == static_cast<const void*>(b);
                if (same) in_place_blocks(o, n, [](T u, T v) { return Op::f(u, v); }, a, b);
                else for (std::int64_t i = 0; i < n; ++i) o[i] = Op::f(a[i], b[i]);
            }
            return;
        }
        if (s[0] == sizeof(T) && s[1] == 0) {  // array op scalar (x + 1.0)
            if constexpr (requires { neon_binary<Op, T>::run_scalar_b; }) {
                neon_binary<Op, T>::run_scalar_b(a, b, o, n);
            } else {
                const T bv = *b;
                if constexpr (std::is_same_v<R, T>) {
                    if (static_cast<const void*>(o) == static_cast<const void*>(a)) {  // x += 1.0
                        in_place_blocks(o, n, [bv](T u) { return Op::f(u, bv); }, a);
                        return;
                    }
                }
                for (std::int64_t i = 0; i < n; ++i) o[i] = Op::f(a[i], bv);
            }
            return;
        }
        if (s[0] == 0 && s[1] == sizeof(T)) {  // scalar op array (1.0 - x)
            if constexpr (requires { neon_binary<Op, T>::run_scalar_a; }) {
                neon_binary<Op, T>::run_scalar_a(a, b, o, n);
            } else {
                const T av = *a;
                for (std::int64_t i = 0; i < n; ++i) o[i] = Op::f(av, b[i]);
            }
            return;
        }
    }
    const std::byte* a = args[0];
    const std::byte* b = args[1];
    std::byte* o = args[2];
    for (std::int64_t i = 0; i < n; ++i, a += s[0], b += s[1], o += s[2])
        *reinterpret_cast<R*>(o) = Op::f(load_as<T>(a), load_as<T>(b));
}

// args: condition (bool), x1, x2, out
template <typename T>
void where_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* c = args[0];
    const std::byte* a = args[1];
    const std::byte* b = args[2];
    std::byte* o = args[3];
    for (std::int64_t i = 0; i < n; ++i, c += s[0], a += s[1], b += s[2], o += s[3])
        *reinterpret_cast<T*>(o) = load_as<bool>(c) ? load_as<T>(a) : load_as<T>(b);
}

// clip = minimum(maximum(x, lo), hi), fused into one pass, with the same NaN
// and signed-zero rules as maximum/minimum. args: x, lo, hi, out
template <typename T>
void clip_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    constexpr std::int64_t t = sizeof(T);
    const bool bounds_ok = (s[1] == 0 || s[1] == t) && (s[2] == 0 || s[2] == t);
    if (s[0] == t && s[3] == t && bounds_ok) {  // contiguous x and out; each bound contiguous or a scalar
        const T* xp = reinterpret_cast<const T*>(args[0]);
        const T* lo = reinterpret_cast<const T*>(args[1]);
        const T* hi = reinterpret_cast<const T*>(args[2]);
        T* op = reinterpret_cast<T*>(args[3]);
        if constexpr (ops::is_float<T>) {
            neon_clip(xp, lo, s[1] / t, hi, s[2] / t, op, n);  // FMAX then FMIN: the same rules
        } else {
            const std::int64_t sl = s[1] / t, sh = s[2] / t;
            for (std::int64_t i = 0; i < n; ++i) op[i] = ops::Minimum::f(ops::Maximum::f(xp[i], lo[i * sl]), hi[i * sh]);
        }
        return;
    }
    const std::byte* x = args[0];
    const std::byte* lo = args[1];
    const std::byte* hi = args[2];
    std::byte* o = args[3];
    for (std::int64_t i = 0; i < n; ++i, x += s[0], lo += s[1], hi += s[2], o += s[3])
        *reinterpret_cast<T*>(o) =
            ops::Minimum::f(ops::Maximum::f(load_as<T>(x), load_as<T>(lo)), load_as<T>(hi));
}

// ---- n inputs, m outputs (docs/design/special.md §3.1) ---------------------
// For operations with more operands than binary has, or more than one
// result: Op says how many with `nin` and `nout`, and f takes nin values.
// With nout == 1 it returns the result; otherwise a std::array of nout
// results (airy: Ai, Ai', Bi, Bi'). Used by sd.special.
//
// The functions this serves cost tens to hundreds of nanoseconds per
// element, so the loop keeps to two layouts: everything contiguous (indexed
// pointers, as in unary_loop) and the general strided loop. A broadcast
// scalar parameter (betainc(a, b, x) with scalar a, b) takes the strided
// loop; its stride-0 loads cost nothing next to the function.

namespace detail {

template <typename Op, typename T, typename Seq> struct NaryResult;
template <typename Op, typename T, std::size_t... I>
struct NaryResult<Op, T, std::index_sequence<I...>> {
    using type = decltype(Op::f(((void)I, std::declval<T>())...));
};

template <typename R> struct NaryOut { using type = R; static constexpr int n = 1; };
template <typename R, std::size_t N> struct NaryOut<std::array<R, N>> { using type = R; static constexpr int n = N; };

}  // namespace detail

template <typename Op, typename T>
using nary_result_t = typename detail::NaryResult<Op, T, std::make_index_sequence<Op::nin>>::type;

// args: the nin inputs, then the nout outputs
template <typename Op, typename T>
void nary_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    constexpr int nin = Op::nin;
    using Ret = nary_result_t<Op, T>;
    using R = typename detail::NaryOut<Ret>::type;
    constexpr int nout = detail::NaryOut<Ret>::n;
    static_assert(nout == Op::nout, "Op::f returns nout results");
    static_assert(nin + nout <= kMaxOperands);
    bool contiguous = true;
    for (int k = 0; k < nin; ++k) contiguous = contiguous && s[k] == sizeof(T);
    for (int j = 0; j < nout; ++j) contiguous = contiguous && s[nin + j] == sizeof(R);
    auto store = [&](auto&& out_at, const Ret& r) {
        if constexpr (nout == 1) out_at(0) = r;
        else for (int j = 0; j < nout; ++j) out_at(j) = r[static_cast<std::size_t>(j)];
    };
    if (contiguous) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            const T* const x[] = {reinterpret_cast<const T*>(args[I])...};
            R* o[nout];
            for (int j = 0; j < nout; ++j) o[j] = reinterpret_cast<R*>(args[nin + j]);
            for (std::int64_t i = 0; i < n; ++i) store([&](int j) -> R& { return o[j][i]; }, Op::f(x[I][i]...));
        }(std::make_index_sequence<nin>{});
        return;
    }
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        std::array<std::byte*, nin + nout> p;
        for (int k = 0; k < nin + nout; ++k) p[static_cast<std::size_t>(k)] = args[k];
        for (std::int64_t i = 0; i < n; ++i) {
            store([&](int j) -> R& { return *reinterpret_cast<R*>(p[static_cast<std::size_t>(nin + j)]); },
                  Op::f(load_as<T>(p[I])...));
            for (int k = 0; k < nin + nout; ++k) p[static_cast<std::size_t>(k)] += s[k];
        }
    }(std::make_index_sequence<nin>{});
}

// One table row: the loop for each computation dtype (null where Op has
// none), and the dtype that loop writes.
struct LoopEntry {
    Loop loop = nullptr;
    DType out = DType::Bool;
};
using LoopTable = std::array<LoopEntry, kNumArrayDTypes>;

namespace detail {

template <typename Op, typename T, int Arity>
constexpr LoopEntry entry() {
    if constexpr (!Op::template ok<T>) {
        return {};
    } else if constexpr (Arity == 1) {
        return {&unary_loop<Op, T>, dtype_of<unary_result_t<Op, T>>};
    } else {
        return {&binary_loop<Op, T>, dtype_of<binary_result_t<Op, T>>};
    }
}

template <typename Op, typename T>
constexpr LoopEntry nary_entry() {
    if constexpr (!Op::template ok<T>) {
        return {};
    } else {
        using R = typename NaryOut<nary_result_t<Op, T>>::type;
        return {&nary_loop<Op, T>, dtype_of<R>};  // every output has this dtype
    }
}

template <typename Op, int Arity>
constexpr LoopTable make_table() {
    return {entry<Op, bool, Arity>(), entry<Op, std::int64_t, Arity>(), entry<Op, double, Arity>(),
            entry<Op, complex128, Arity>()};
}

}  // namespace detail

template <typename Op> inline constexpr LoopTable unary_table = detail::make_table<Op, 1>();
template <typename Op> inline constexpr LoopTable binary_table = detail::make_table<Op, 2>();
template <typename Op>
inline constexpr LoopTable nary_table = {detail::nary_entry<Op, bool>(), detail::nary_entry<Op, std::int64_t>(),
                                         detail::nary_entry<Op, double>(), detail::nary_entry<Op, complex128>()};

}  // namespace sd
