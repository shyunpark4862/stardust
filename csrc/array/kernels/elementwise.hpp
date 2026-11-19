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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace sd {

template <typename T>
inline T load_as(const std::byte* p) {
    return *reinterpret_cast<const T*>(p);
}

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
        if constexpr ((std::is_same_v<Op, ops::IsInf> || std::is_same_v<Op, ops::IsFinite>) &&
                      std::is_same_v<T, double>) {
            // One absolute compare per vector (FACGE |x| >= inf, FACGT inf > |x|; NaN
            // false either way), narrowed to bytes. The loop is bound by its
            // instruction count, not memory: masking the sign first cost a second
            // instruction per vector (10^6: 88 us, isnan's one-compare loop 70).
            const float64x2_t inf = vdupq_n_f64(std::numeric_limits<double>::infinity());
            auto test = [&](const double* p) {
                const float64x2_t v = vld1q_f64(p);
                if constexpr (std::is_same_v<Op, ops::IsInf>) return vcageq_f64(v, inf);
                else return vcagtq_f64(inf, v);
            };
            auto* ob = reinterpret_cast<std::uint8_t*>(o);
            std::int64_t i = 0;
            for (; i + 16 <= n; i += 16) {
                uint32x4_t w[4];
                for (int k = 0; k < 4; ++k)
                    w[k] = vuzp1q_u32(
                        vreinterpretq_u32_u64(test(x + i + 4 * k)), vreinterpretq_u32_u64(test(x + i + 4 * k + 2))
                    );
                const uint16x8_t h0 = vuzp1q_u16(vreinterpretq_u16_u32(w[0]), vreinterpretq_u16_u32(w[1]));
                const uint16x8_t h1 = vuzp1q_u16(vreinterpretq_u16_u32(w[2]), vreinterpretq_u16_u32(w[3]));
                const uint8x16_t b8 = vuzp1q_u8(vreinterpretq_u8_u16(h0), vreinterpretq_u8_u16(h1));
                vst1q_u8(ob + i, vandq_u8(b8, vdupq_n_u8(1)));
            }
            for (; i < n; ++i) o[i] = Op::f(x[i]);
            return;
        }
        if constexpr (std::is_same_v<Op, ops::Sign> && std::is_same_v<T, std::int64_t>) {
            // (a < 0 mask) - (a > 0 mask): three instructions per vector. With the
            // data in cache this loop is bound by its instruction count (the
            // shift-and-OR form: 125 us against numpy's 97 in the full scan).
            std::int64_t i = 0;
            for (; i + 4 <= n; i += 4) {
                const int64x2_t v0 = vld1q_s64(x + i), v1 = vld1q_s64(x + i + 2);
                vst1q_s64(
                    o + i, vsubq_s64(vreinterpretq_s64_u64(vcltzq_s64(v0)), vreinterpretq_s64_u64(vcgtzq_s64(v0)))
                );
                vst1q_s64(
                    o + i + 2, vsubq_s64(vreinterpretq_s64_u64(vcltzq_s64(v1)), vreinterpretq_s64_u64(vcgtzq_s64(v1)))
                );
            }
            for (; i < n; ++i) o[i] = Op::f(x[i]);
            return;
        }
        if constexpr (std::is_same_v<Op, ops::Square> && std::is_same_v<T, complex128>) {
            // Two complex values at a time, real and imaginary parts split by the
            // load: the same fused operations as Square::f, so the same bits (the
            // complex loop did not vectorize: 10^6 311 us against numpy's 266).
            const double* xp = reinterpret_cast<const double*>(x);
            double* op = reinterpret_cast<double*>(o);
            std::int64_t i = 0;
            for (; i + 2 <= n; i += 2) {
                const float64x2x2_t v = vld2q_f64(xp + 2 * i);
                float64x2x2_t w;
                w.val[0] = vfmaq_f64(vnegq_f64(vmulq_f64(v.val[1], v.val[1])), v.val[0], v.val[0]);
                w.val[1] = vfmaq_f64(vmulq_f64(v.val[1], v.val[0]), v.val[0], v.val[1]);
                vst2q_f64(op + 2 * i, w);
            }
            for (; i < n; ++i) o[i] = Op::f(x[i]);
            return;
        }
        for (std::int64_t i = 0; i < n; ++i) o[i] = Op::f(x[i]);
        return;
    }
    const std::byte* x = args[0];
    std::byte* o = args[1];
    for (std::int64_t i = 0; i < n; ++i, x += s[0], o += s[1]) *reinterpret_cast<R*>(o) = Op::f(load_as<T>(x));
}

// x ** e for int64 x and one exponent e >= 0 (x ** 2, the usual call): the
// squaring steps of int_pow taken for a block of elements at a time, so the
// loop over the exponent's bits is shared and the per-element work has no
// branches (int_pow per element: 10^6 x ** 2 686 us against numpy's 261). The
// products wrap exactly as int_pow's. A negative e goes element by element.
inline void int_pow_scalar(const std::int64_t* a, std::int64_t e, std::int64_t* o, std::int64_t n) {
    if (e < 0) {
        for (std::int64_t i = 0; i < n; ++i) o[i] = ops::int_pow(a[i], e);
        return;
    }
    if (e == 2) {
        for (std::int64_t i = 0; i < n; ++i) o[i] = ops::wrap_mul(a[i], a[i]);
        return;
    }
    constexpr std::int64_t kBlock = 256;
    for (std::int64_t i0 = 0; i0 < n; i0 += kBlock) {
        const std::int64_t m = std::min(kBlock, n - i0);
        std::int64_t base[kBlock], res[kBlock];
        for (std::int64_t k = 0; k < m; ++k) base[k] = a[i0 + k], res[k] = 1;
        for (std::int64_t x = e; x;) {
            if (x & 1)
                for (std::int64_t k = 0; k < m; ++k) res[k] = ops::wrap_mul(res[k], base[k]);
            x >>= 1;
            if (x)
                for (std::int64_t k = 0; k < m; ++k) base[k] = ops::wrap_mul(base[k], base[k]);
        }
        std::copy(res, res + m, o + i0);
    }
}

// The exponent loop of pow_float_run for one exact kind (a broadcast scalar exponent).
template <ops::PowKind K>
inline void pow_exact_run(const std::byte* a, std::int64_t sa, std::byte* o, std::int64_t so, std::int64_t n) {
    constexpr std::int64_t d = sizeof(double);
    auto f = [](double u) { return ops::pow_exact(K, u); };
    if (sa == d && so == d) {
        const double* x = reinterpret_cast<const double*>(a);
        double* y = reinterpret_cast<double*>(o);
        if (static_cast<const void*>(x) == static_cast<const void*>(y)) {  // x **= e
            in_place_blocks(y, n, f, x);
        } else {
            for (std::int64_t i = 0; i < n; ++i) y[i] = f(x[i]);
        }
        return;
    }
    for (std::int64_t i = 0; i < n; ++i) *reinterpret_cast<double*>(o + i * so) = f(load_as<double>(a + i * sa));
}

// float64 x ** y: libm's pow (numpy's: the same bits as numpy and the frame; more accurate than
// vForce's vvpow, docs/design/array.md §5.4) element by element, except the exponents 2, 0, 1,
// -1 and 0.5, which have exact forms (ops::pow_exact: x * x, 1, x, 1 / x, sqrt with C99 pow's
// special values; numpy's fast paths). A broadcast scalar exponent of those takes its own
// loop; otherwise each element is ops::Pow::f, so the bits do not depend on the exponent's
// stride (array, broadcast or scalar).
inline void pow_float_run(
    const std::byte* a,
    std::int64_t sa,
    const std::byte* b,
    std::int64_t sb,
    std::byte* o,
    std::int64_t so,
    std::int64_t n
) {
    if (sb == 0) {
        switch (ops::pow_kind(load_as<double>(b))) {
            case ops::PowKind::kSquare: return pow_exact_run<ops::PowKind::kSquare>(a, sa, o, so, n);
            case ops::PowKind::kOne: return pow_exact_run<ops::PowKind::kOne>(a, sa, o, so, n);
            case ops::PowKind::kSame: return pow_exact_run<ops::PowKind::kSame>(a, sa, o, so, n);
            case ops::PowKind::kRecip: return pow_exact_run<ops::PowKind::kRecip>(a, sa, o, so, n);
            case ops::PowKind::kSqrt: return pow_exact_run<ops::PowKind::kSqrt>(a, sa, o, so, n);
            default: break;
        }
    }
    for (std::int64_t i = 0; i < n; ++i)
        *reinterpret_cast<double*>(o + i * so) = ops::Pow::f(load_as<double>(a + i * sa), load_as<double>(b + i * sb));
}

// args: a, b, out
template <typename Op, typename T>
void binary_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    using R = binary_result_t<Op, T>;
    if constexpr (std::is_same_v<Op, ops::Pow> && ops::is_float<T>) {
        pow_float_run(args[0], s[0], args[1], s[1], args[2], s[2], n);
        return;
    }
    if constexpr (ops::is_float<T> && vforce_binary<Op> != nullptr) {  // atan2 through vForce
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
                else
                    for (std::int64_t i = 0; i < n; ++i) o[i] = Op::f(a[i], b[i]);
            }
            return;
        }
        if (s[0] == sizeof(T) && s[1] == 0) {  // array op scalar (x + 1.0)
            if constexpr (requires { neon_binary<Op, T>::run_scalar_b; }) {
                neon_binary<Op, T>::run_scalar_b(a, b, o, n);
            } else if constexpr (std::is_same_v<Op, ops::Pow> && std::is_same_v<T, std::int64_t>) {
                int_pow_scalar(a, *b, o, n);
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
    constexpr std::int64_t t = sizeof(T);
    if constexpr (t == 8) {
        if (s[0] == 1 && s[3] == t && (s[1] == 0 || s[1] == t) && (s[2] == 0 || s[2] == t)) {
            // Contiguous condition and output, each value contiguous or a scalar: a
            // select on the raw words through a mask, branch-free and vectorized (a
            // random condition defeats branch prediction; the byte-strided loop took
            // 290 us for 10^6 where(b, f, g), numpy 276).
            const auto* cp = reinterpret_cast<const std::uint8_t*>(args[0]);
            const auto* ap = reinterpret_cast<const std::uint64_t*>(args[1]);
            const auto* bp = reinterpret_cast<const std::uint64_t*>(args[2]);
            auto* op = reinterpret_cast<std::uint64_t*>(args[3]);
            const std::int64_t sa = s[1] / t, sb = s[2] / t;  // 1 or 0 (a scalar)
            auto sel = [](std::uint8_t c, std::uint64_t x, std::uint64_t y) {
                const std::uint64_t m = 0 - static_cast<std::uint64_t>(c);
                return (x & m) | (y & ~m);
            };
            const std::uint64_t x0 = ap[0], y0 = bp[0];
            if (sa && sb)
                for (std::int64_t i = 0; i < n; ++i) op[i] = sel(cp[i], ap[i], bp[i]);
            else if (sa)
                for (std::int64_t i = 0; i < n; ++i) op[i] = sel(cp[i], ap[i], y0);
            else if (sb)
                for (std::int64_t i = 0; i < n; ++i) op[i] = sel(cp[i], x0, bp[i]);
            else
                for (std::int64_t i = 0; i < n; ++i) op[i] = sel(cp[i], x0, y0);
            return;
        }
    }
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
            std::int64_t i = 0;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                if (sl == 0 && sh == 0) {
                    // Scalar bounds (the usual call): compare-selects, two lanes at a
                    // time (NEON has no 64-bit integer min/max; the indexed scalar
                    // loop took 376 us for 10^6 against numpy's 135).
                    const int64x2_t l = vdupq_n_s64(*lo), h = vdupq_n_s64(*hi);
                    for (; i + 4 <= n; i += 4) {
                        int64x2_t v0 = vld1q_s64(xp + i), v1 = vld1q_s64(xp + i + 2);
                        v0 = vbslq_s64(vcltq_s64(v0, l), l, v0);
                        v1 = vbslq_s64(vcltq_s64(v1, l), l, v1);
                        v0 = vbslq_s64(vcgtq_s64(v0, h), h, v0);
                        v1 = vbslq_s64(vcgtq_s64(v1, h), h, v1);
                        vst1q_s64(op + i, v0);
                        vst1q_s64(op + i + 2, v1);
                    }
                }
            }
            for (; i < n; ++i) op[i] = ops::Minimum::f(ops::Maximum::f(xp[i], lo[i * sl]), hi[i * sh]);
        }
        return;
    }
    const std::byte* x = args[0];
    const std::byte* lo = args[1];
    const std::byte* hi = args[2];
    std::byte* o = args[3];
    for (std::int64_t i = 0; i < n; ++i, x += s[0], lo += s[1], hi += s[2], o += s[3])
        *reinterpret_cast<T*>(o) = ops::Minimum::f(ops::Maximum::f(load_as<T>(x), load_as<T>(lo)), load_as<T>(hi));
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

template <typename R> struct NaryOut {
    using type = R;
    static constexpr int n = 1;
};
template <typename R, std::size_t N> struct NaryOut<std::array<R, N>> {
    using type = R;
    static constexpr int n = N;
};

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
        else
            for (int j = 0; j < nout; ++j) out_at(j) = r[static_cast<std::size_t>(j)];
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
            store(
                [&](int j) -> R& { return *reinterpret_cast<R*>(p[static_cast<std::size_t>(nin + j)]); },
                Op::f(load_as<T>(p[I])...)
            );
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
    return {
        entry<Op, bool, Arity>(), entry<Op, std::int64_t, Arity>(), entry<Op, double, Arity>(),
        entry<Op, complex128, Arity>()
    };
}

}  // namespace detail

template <typename Op> inline constexpr LoopTable unary_table = detail::make_table<Op, 1>();
template <typename Op> inline constexpr LoopTable binary_table = detail::make_table<Op, 2>();
template <typename Op>
inline constexpr LoopTable nary_table = {
    detail::nary_entry<Op, bool>(), detail::nary_entry<Op, std::int64_t>(), detail::nary_entry<Op, double>(),
    detail::nary_entry<Op, complex128>()
};

}  // namespace sd
