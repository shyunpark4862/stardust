// Reduction loops (docs/design/array.md §6).
//
// A reduction runs on the nd iterator with two operands, the input and an
// output that has extent 1 on every reduced axis, broadcast back to the input
// shape (stride 0 on those axes). The kernel then does  out = f(out, x).
// What the iterator hands it decides the strategy:
//   * out stride 0  -> inner-axis reduction: fold a run into one accumulator
//   * out stride != 0 -> outer-axis reduction: add a whole row into the
//     result row, element by element (vectorizes along the kept axis)
// The binary operations are the elementwise ones (ops::Add, ops::Maximum, ...).
//
// A float/complex sum folds each run with numpy's pairwise summation
// (8 accumulators in blocks of 128, halves added recursively): faster, since
// 8 independent chains replace one, and more accurate, since the rounding
// error grows like log n instead of n. max/min fold with NEON FMAX/FMIN
// (docs/design/array.md §6.2, §6.3).
#pragma once

#include "array/kernels/elementwise.hpp"
#include "base/pairwise.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace sd {

// ---- pairwise summation (numpy's DOUBLE/CDOUBLE_pairwise_sum) --------------
// The tree and the leaf sums are base/pairwise.hpp, shared by every area; here
// the terms reach a leaf through an accessor (docs/design/array.md §6.2).

inline constexpr std::int64_t kPairwiseBlock = pairwise::kBlock;  // numpy's PW_BLOCKSIZE, in doubles

// Sum of get(begin), ..., get(begin + n - 1).
template <typename Get>
double pairwise_sum(std::int64_t begin, std::int64_t n, const Get& get) {
    return pairwise::tree(begin, n, [&](std::int64_t b, std::int64_t m) {
        return pairwise::leaf(m, [&](std::int64_t i) { return get(b + i); });
    });
}

// The complex version: real and imaginary parts summed side by side. numpy
// counts doubles, so a block holds 64 complex numbers and 4 accumulators.
template <typename Get>
complex128 pairwise_sum_complex(std::int64_t begin, std::int64_t n, const Get& get) {
    return pairwise::tree_complex(begin, n, [&](std::int64_t b, std::int64_t m) {
        return pairwise::leaf_complex(m, [&](std::int64_t i) { return get(b + i); });
    });
}

// ---- the same sums on many threads, with the same bits ---------------------
// The subtrees of at most kLeaf elements are summed in parallel, then combined
// along the very same tree (pairwise::plan, pairwise::combine). The result is
// bit-for-bit the serial one (and numpy's), for any thread count
// (docs/design/array.md §6.2).

inline constexpr std::int64_t kLeaf = kParallelTask;

template <typename T, typename SumLeaf>
T pairwise_parallel(std::int64_t n, std::int64_t leaf, const SumLeaf& sum_leaf) {
    const pairwise::Leaves leaves = pairwise::plan(n, leaf, std::is_same_v<T, complex128>);
    std::vector<T> part(leaves.size());
    parallel::parallel_for(static_cast<std::int64_t>(leaves.size()), [&](std::int64_t t) {
        part[t] = sum_leaf(leaves[t].first, leaves[t].second);
    });
    return pairwise::combine(n, leaf, part);
}

template <typename Get>
double pairwise_sum_mt(std::int64_t n, const Get& get) {
    if (n < kParallelThreshold || !parallel::available()) return pairwise_sum(0, n, get);
    return pairwise_parallel<double>(n, kLeaf, [&](std::int64_t b, std::int64_t m) { return pairwise_sum(b, m, get); });
}

template <typename Get>
complex128 pairwise_sum_complex_mt(std::int64_t n, const Get& get) {
    if (n < kParallelThreshold || !parallel::available()) return pairwise_sum_complex(0, n, get);
    return pairwise_parallel<complex128>(n, kLeaf / 2, [&](std::int64_t b, std::int64_t m) {
        return pairwise_sum_complex(b, m, get);
    });
}

// ---- float and complex products in fixed blocks ---------------------------
// A product folded left to right is one chain of multiplies, 4 cycles each
// (prod of 1M float64: 0.78 ms, as numpy). Instead a run is cut into pieces
// of kLeaf elements; within a piece, eight accumulators take every eighth
// element and are combined as ((a0*a1)*(a2*a3))*((a4*a5)*(a6*a7)); the pieces
// are then multiplied left to right. The structure depends on n only, the
// pieces may run in parallel, and the bits are the same for any thread count
// -- but not numpy's: the order of the multiplies changed
// (docs/design/array.md §6.6). It can also over- or underflow at different points
// than a left-to-right product of the same numbers.

template <typename T, typename Get>
T product_piece(std::int64_t b, std::int64_t m, const Get& get) {
    if (m < 8) {
        if (m == 0) return T(1);
        T r = get(b);
        for (std::int64_t i = 1; i < m; ++i) r = ops::Multiply::f(r, get(b + i));
        return r;
    }
    T a[8];
    for (int j = 0; j < 8; ++j) a[j] = get(b + j);
    std::int64_t i = 8;
    for (; i + 8 <= m; i += 8)
        for (int j = 0; j < 8; ++j) a[j] = ops::Multiply::f(a[j], get(b + i + j));
    for (int j = 0; i < m; ++i, ++j) a[j] = ops::Multiply::f(a[j], get(b + i));
    for (int w = 1; w < 8; w *= 2)
        for (int j = 0; j < 8; j += 2 * w) a[j] = ops::Multiply::f(a[j], a[j + w]);
    return a[0];
}

template <typename T, typename Get>
T product_mt(std::int64_t n, const Get& get) {
    const std::int64_t pieces = std::max<std::int64_t>((n + kLeaf - 1) / kLeaf, 1);
    auto piece = [&](std::int64_t t) { return product_piece<T>(t * kLeaf, std::min(kLeaf, n - t * kLeaf), get); };
    T r = piece(0);
    if (n < kParallelThreshold || !parallel::available()) {
        for (std::int64_t t = 1; t < pieces; ++t) r = ops::Multiply::f(r, piece(t));
        return r;
    }
    auto part = std::make_unique_for_overwrite<T[]>(static_cast<size_t>(pieces));
    parallel::parallel_for(pieces - 1, [&](std::int64_t t) { part[t + 1] = piece(t + 1); });
    for (std::int64_t t = 1; t < pieces; ++t) r = ops::Multiply::f(r, part[t]);
    return r;
}

// Folds that do not depend on the order at all -- wrapping integer + and *,
// max/min, and/or -- are cut into fixed pieces, folded in parallel, and the
// piece results folded in order.
template <typename Op, typename T>
inline constexpr bool order_free =
    std::is_same_v<Op, ops::Maximum> || std::is_same_v<Op, ops::Minimum> || std::is_same_v<Op, ops::NanMaximum> ||
    std::is_same_v<Op, ops::NanMinimum> || std::is_same_v<Op, ops::LogicalOr> || std::is_same_v<Op, ops::LogicalAnd> ||
    ((std::is_same_v<Op, ops::Add> || std::is_same_v<Op, ops::Multiply>) && (ops::is_int<T> || ops::is_bool<T>));

// ---- max/min of a contiguous run with NEON ---------------------------------

template <bool Max, bool SkipNan = false>
double neon_fold(const double* x, std::int64_t n, double acc) {
    // FMAXNM/FMINNM skip a NaN operand (nanmax/nanmin); FMAX/FMIN propagate it.
    auto pick = [](float64x2_t a, float64x2_t b) {
        if constexpr (SkipNan) return Max ? vmaxnmq_f64(a, b) : vminnmq_f64(a, b);
        else return Max ? vmaxq_f64(a, b) : vminq_f64(a, b);
    };
    auto one = [](double a, double b) {
        if constexpr (SkipNan) return Max ? ops::NanMaximum::f(a, b) : ops::NanMinimum::f(a, b);
        else return Max ? ops::fmax_nan(a, b) : ops::fmin_nan(a, b);
    };
    std::int64_t i = 0;
    if (n >= 8) {
        // Four independent registers (8 lanes), like the 8 sum accumulators:
        // FMAX's latency is hidden behind the other three.
        float64x2_t v[4];
        for (int j = 0; j < 4; ++j) v[j] = vld1q_f64(x + 2 * j);
        for (i = 8; i + 8 <= n; i += 8)
            for (int j = 0; j < 4; ++j) {
                const float64x2_t y = vld1q_f64(x + i + 2 * j);
                v[j] = pick(v[j], y);
            }
        const float64x2_t m = pick(pick(v[0], v[1]), pick(v[2], v[3]));
        double lanes;  // across the two lanes
        if constexpr (SkipNan) lanes = Max ? vmaxnmvq_f64(m) : vminnmvq_f64(m);
        else lanes = Max ? vmaxvq_f64(m) : vminvq_f64(m);
        acc = one(acc, lanes);
    }
    for (; i < n; ++i) acc = one(acc, x[i]);
    return acc;
}

// int64 max/min: NEON has no 64-bit integer max/min instruction, so each
// step is a compare and a select, ~5 cycles of latency. The plain loop was
// one dependency chain (1.6x slower than numpy, docs/design/array.md §6.3); four
// registers still left it latency-bound (no change, measured), eight
// independent registers (16 lanes) keep enough steps in flight.
template <bool Max>
std::int64_t neon_fold_int(const std::int64_t* x, std::int64_t n, std::int64_t acc) {
    constexpr int kRegs = 8;
    std::int64_t i = 0;
    if (n >= 2 * kRegs) {
        int64x2_t v[kRegs];
        for (int j = 0; j < kRegs; ++j) v[j] = vld1q_s64(x + 2 * j);
        auto pick = [](int64x2_t a, int64x2_t b) {
            return Max ? vbslq_s64(vcgtq_s64(a, b), a, b) : vbslq_s64(vcgtq_s64(a, b), b, a);
        };
        for (i = 2 * kRegs; i + 2 * kRegs <= n; i += 2 * kRegs)
            for (int j = 0; j < kRegs; ++j) v[j] = pick(v[j], vld1q_s64(x + i + 2 * j));
        for (int w = kRegs / 2; w >= 1; w /= 2)
            for (int j = 0; j < w; ++j) v[j] = pick(v[j], v[j + w]);
        for (const std::int64_t lane : {vgetq_lane_s64(v[0], 0), vgetq_lane_s64(v[0], 1)})
            acc = Max ? std::max(acc, lane) : std::min(acc, lane);
    }
    for (; i < n; ++i) acc = Max ? std::max(acc, x[i]) : std::min(acc, x[i]);
    return acc;
}

// The nan* reductions read x through this: a NaN (complex: either part)
// counts as `fill`, the identity of the fold (0 for sums, 1 for products).
// Folding x so is folding a copy of x with its NaNs replaced -- the same
// order, the same bits -- without making the copy (docs/design/array.md §6.11).
template <typename T>
inline T nan_as(T v, T fill) {
    if constexpr (ops::is_cplx<T>) return (std::isnan(v.real()) || std::isnan(v.imag())) ? fill : v;
    else if constexpr (ops::is_float<T>) return v != v ? fill : v;
    else return v;
}

// args: x, out (read and written). SkipNan: x's NaNs read as the identity
// (Add and Multiply of float64 / complex128 only).
template <typename Op, typename T, bool SkipNan = false>
void reduce_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    std::byte* o = args[1];
    [[maybe_unused]] const T identity = std::is_same_v<Op, ops::Multiply> ? T(1) : T(0);
    auto value = [&](T v) {
        if constexpr (SkipNan) return nan_as(v, identity);
        else return v;
    };
    if (s[1] == 0) {  // inner-axis: fold the run into one value
        T acc = load_as<T>(o);
        auto at = [&](std::int64_t i) { return value(load_as<T>(x + i * s[0])); };
        if constexpr (ops::is_bool<T> && (std::is_same_v<Op, ops::LogicalOr> || std::is_same_v<Op, ops::LogicalAnd>)) {
            // any/all stop as soon as the answer is known: any at the first
            // True, all at the first False. Checked once per block of 4096,
            // so the block itself stays a branch-free (vectorized) fold. A
            // later run of the same output returns at once (docs/design/array.md §6.4).
            constexpr bool decided = std::is_same_v<Op, ops::LogicalOr>;
            for (std::int64_t off = 0; off < n && acc != decided; off += 4096) {
                const std::int64_t m = std::min<std::int64_t>(4096, n - off);
                bool part = !decided;
                if (s[0] == 1) {
                    // Folded as bytes: with `bool` the compiler turned the OR
                    // into a find-the-first-true loop, one element per branch
                    // (0.9 ns/element, measured); on uint8_t it is a plain vector OR.
                    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(x) + off;
                    std::uint8_t bits = decided ? 0 : 1;
                    for (std::int64_t i = 0; i < m; ++i) bits = decided ? (bits | p[i]) : (bits & p[i]);
                    part = bits != 0;
                } else {
                    for (std::int64_t i = 0; i < m; ++i) part = Op::f(part, at(off + i));
                }
                acc = Op::f(acc, part);
            }
            *reinterpret_cast<T*>(o) = acc;
            return;
        }
        if constexpr (order_free<Op, T>) {
            if (n >= kParallelThreshold && parallel::available()) {
                // A long run with one output (a full reduction): pieces in parallel.
                const std::int64_t pieces = (n + kParallelTask - 1) / kParallelTask;
                auto part = std::make_unique_for_overwrite<T[]>(static_cast<size_t>(pieces));  // not vector<bool>
                parallel::parallel_for(pieces, [&](std::int64_t t) {
                    const std::int64_t begin = t * kParallelTask;
                    const std::int64_t m = std::min(kParallelTask, n - begin);
                    const std::byte* first = x + begin * s[0];
                    part[t] = load_as<T>(first);  // each piece folds onto its own first element
                    std::byte* piece[] = {const_cast<std::byte*>(first + s[0]), reinterpret_cast<std::byte*>(&part[t])};
                    const std::int64_t piece_strides[] = {s[0], 0};
                    reduce_loop<Op, T>(piece, m - 1, piece_strides, nullptr);  // serial: we are inside a task
                });
                for (std::int64_t t = 0; t < pieces; ++t) acc = Op::f(acc, part[t]);
                *reinterpret_cast<T*>(o) = acc;
                return;
            }
        }
        if constexpr (std::is_same_v<Op, ops::Add> && ops::is_float<T> && SkipNan) {
            // A typed contiguous read lets the NaN select vectorize (2x on 1e7).
            if (s[0] == sizeof(T)) {
                const T* xp = reinterpret_cast<const T*>(x);
                acc = acc + pairwise_sum_mt(n, [xp](std::int64_t i) {
                          const T v = xp[i];
                          return v != v ? T(0) : v;
                      });
            } else {
                acc = acc + pairwise_sum_mt(n, at);
            }
        } else if constexpr (std::is_same_v<Op, ops::Add> && ops::is_float<T>) {
            acc = acc + pairwise_sum_mt(n, at);
        } else if constexpr (std::is_same_v<Op, ops::Add> && ops::is_cplx<T>) {
            const complex128 p = pairwise_sum_complex_mt(n, at);
            acc = {acc.real() + p.real(), acc.imag() + p.imag()};
        } else if constexpr (std::is_same_v<Op, ops::Multiply> && (ops::is_float<T> || ops::is_cplx<T>)) {
            if (s[0] == sizeof(T)) {
                const T* xp = reinterpret_cast<const T*>(x);
                acc = Op::f(acc, product_mt<T>(n, [&](std::int64_t i) { return value(xp[i]); }));
            } else {
                acc = Op::f(acc, product_mt<T>(n, at));
            }
        } else if constexpr ((std::is_same_v<Op, ops::Maximum> || std::is_same_v<Op, ops::Minimum>) &&
                             ops::is_float<T>) {
            if (s[0] == sizeof(T))
                acc = neon_fold<std::is_same_v<Op, ops::Maximum>>(reinterpret_cast<const T*>(x), n, acc);
            else
                for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, at(i));
        } else if constexpr (std::is_same_v<Op, ops::NanMaximum> || std::is_same_v<Op, ops::NanMinimum>) {
            if (s[0] == sizeof(T))
                acc = neon_fold<std::is_same_v<Op, ops::NanMaximum>, true>(reinterpret_cast<const T*>(x), n, acc);
            else
                for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, at(i));
        } else if constexpr ((std::is_same_v<Op, ops::Maximum> || std::is_same_v<Op, ops::Minimum>) && ops::is_int<T>) {
            if (s[0] == sizeof(T)) {
                acc = neon_fold_int<std::is_same_v<Op, ops::Maximum>>(reinterpret_cast<const T*>(x), n, acc);
            } else {
                for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, at(i));
            }
        } else if (s[0] == sizeof(T)) {
            const T* xp = reinterpret_cast<const T*>(x);  // indexed: integer folds vectorize
            for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, xp[i]);
        } else {
            for (std::int64_t i = 0; i < n; ++i) acc = Op::f(acc, at(i));
        }
        *reinterpret_cast<T*>(o) = acc;
        return;
    }
    if (s[0] == sizeof(T) && s[1] == sizeof(T)) {  // outer-axis, contiguous rows: o = f(o, x)
        T* op = reinterpret_cast<T*>(o);
        const T* xp = reinterpret_cast<const T*>(x);
        if constexpr (neon_binary<Op, T>::available && !SkipNan) neon_binary<Op, T>::run(op, xp, op, n);
        else
            for (std::int64_t i = 0; i < n; ++i) op[i] = Op::f(op[i], value(xp[i]));
        return;
    }
    for (std::int64_t i = 0; i < n; ++i, x += s[0], o += s[1])
        *reinterpret_cast<T*>(o) = Op::f(load_as<T>(o), value(load_as<T>(x)));
}

// Counting folds into int64 (order-free): Kind 0 counts the non-NaN
// elements (nanmean, nanvar), Kind 1 the nonzero ones (count_nonzero; NaN
// is nonzero). A long run of one output is counted in parallel pieces.
// args: x, out (int64)
template <typename T, int Kind>
void count_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    std::byte* o = args[1];
    auto hit = [&](std::int64_t i) -> std::int64_t {
        const T v = load_as<T>(x + i * s[0]);
        if constexpr (Kind == 0) {
            if constexpr (ops::is_cplx<T>) return !(std::isnan(v.real()) || std::isnan(v.imag()));
            else if constexpr (ops::is_float<T>) return v == v;
            else return 1;
        } else {
            if constexpr (ops::is_cplx<T>) return v.real() != 0 || v.imag() != 0;
            else return v != T(0);
        }
    };
    auto run = [&](std::int64_t b, std::int64_t e) {
        std::int64_t acc = 0;
        if constexpr (std::is_same_v<T, bool> && Kind == 1) {
            if (s[0] == 1) {
                // Bytes: sixteen at a time into byte counters (a nonzero byte's test
                // mask is -1), widened every 255 steps before they can wrap. (Each byte
                // added into an int64: 10^6 took 47 us against numpy's 23.)
                const auto* p = reinterpret_cast<const std::uint8_t*>(x);
                std::int64_t i = b;
                while (i + 16 <= e) {
                    uint8x16_t c = vdupq_n_u8(0);
                    const std::int64_t stop = std::min(e - 15, i + 255 * 16);
                    for (; i < stop; i += 16) {
                        const uint8x16_t v = vld1q_u8(p + i);
                        c = vsubq_u8(c, vtstq_u8(v, v));
                    }
                    acc += vaddlvq_u8(c);
                }
                for (; i < e; ++i) acc += p[i] != 0;
                return acc;
            }
        }
        if (s[0] == sizeof(T)) {  // typed and contiguous: vectorizes
            const T* xp = reinterpret_cast<const T*>(x);
            for (std::int64_t i = b; i < e; ++i) {
                const T v = xp[i];
                if constexpr (Kind == 0) {
                    if constexpr (ops::is_cplx<T>) acc += !(v.real() != v.real() || v.imag() != v.imag());
                    else if constexpr (ops::is_float<T>) acc += v == v;
                    else acc += 1;
                } else {
                    if constexpr (ops::is_cplx<T>) acc += v.real() != 0 || v.imag() != 0;
                    else acc += v != T(0);
                }
            }
            return acc;
        }
        for (std::int64_t i = b; i < e; ++i) acc += hit(i);
        return acc;
    };
    if (s[1] == 0) {
        std::int64_t acc = 0;
        if (n >= kParallelThreshold && parallel::available()) {
            const std::int64_t pieces = (n + kParallelTask - 1) / kParallelTask;
            std::vector<std::int64_t> part(pieces);
            parallel::parallel_for(pieces, [&](std::int64_t t) {
                part[t] = run(t * kParallelTask, std::min(n, (t + 1) * kParallelTask));
            });
            for (std::int64_t v : part) acc += v;
        } else {
            acc = run(0, n);
        }
        *reinterpret_cast<std::int64_t*>(o) += acc;
        return;
    }
    for (std::int64_t i = 0; i < n; ++i) *reinterpret_cast<std::int64_t*>(o + i * s[1]) += hit(i);
}

// Sum of squared deviations, the second pass of var: out += |x - m|^2.
// With SkipNan (nanvar) a NaN element adds +0.0 instead, which leaves every
// partial sum of the pairwise tree unchanged.
// args: x, m (the mean, broadcast), out (float64)
template <typename T, bool SkipNan = false>
void sq_dev_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    const std::byte* x = args[0];
    const std::byte* m = args[1];
    std::byte* o = args[2];
    auto sq = [&](std::int64_t i) {
        const T v = load_as<T>(x + i * s[0]);
        const T d = v - load_as<T>(m + i * s[1]);
        double sq;
        if constexpr (ops::is_cplx<T>) sq = d.real() * d.real() + d.imag() * d.imag();
        else sq = d * d;
        if constexpr (SkipNan) {
            // A select, not a branch: NaNs scattered at random defeat branch prediction.
            bool nan;
            if constexpr (ops::is_cplx<T>) nan = v.real() != v.real() || v.imag() != v.imag();
            else nan = v != v;
            return nan ? 0.0 : sq;
        }
        return sq;
    };
    if (s[2] == 0) {  // inner-axis: pairwise, like sum
        *reinterpret_cast<double*>(o) += pairwise_sum_mt(n, sq);
        return;
    }
    for (std::int64_t i = 0; i < n; ++i) *reinterpret_cast<double*>(o + i * s[2]) += sq(i);
}

// ---- argmin / argmax over one lane (index order, so ties keep the first) ----

// float64, contiguous: two lanes, each keeping its best value and where it
// was. Strict > keeps the first index within a lane; between the lanes a tie
// goes to the smaller index. NaNs are only noted on the way (any NaN wins,
// the first one) and then located by a scalar rescan, as they are rare.
// SkipNan (nanargmin/nanargmax): a NaN reads as +inf (min) or -inf (max),
// numpy's replacement, so it never wins over a number and the first of
// equal values still wins.
template <bool Max, bool SkipNan = false>
std::int64_t neon_arg_extreme(const double* x, std::int64_t n) {
    const double far = Max ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    const float64x2_t farv = vdupq_n_f64(far);
    auto load = [&](const double* p) {
        const float64x2_t v = vld1q_f64(p);
        if constexpr (SkipNan) return vbslq_f64(vceqq_f64(v, v), v, farv);
        else return v;
    };
    float64x2_t best = load(x);
    int64x2_t at = {0, 1}, idx = {0, 1};
    const int64x2_t two = vdupq_n_s64(2);
    uint64x2_t ordered = vceqq_f64(best, best);  // all-ones while a lane has seen no NaN (x == x)
    std::int64_t i = 2;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t v = load(x + i);
        idx = vaddq_s64(idx, two);
        const uint64x2_t better = Max ? vcgtq_f64(v, best) : vcltq_f64(v, best);
        best = vbslq_f64(better, v, best);
        at = vbslq_s64(better, idx, at);
        if constexpr (!SkipNan) ordered = vandq_u64(ordered, vceqq_f64(v, v));
    }
    if (!SkipNan && vminvq_u32(vreinterpretq_u32_u64(ordered)) == 0)  // some lane met a NaN
        for (std::int64_t k = 0;; ++k)
            if (std::isnan(x[k])) return k;
    double b0 = vgetq_lane_f64(best, 0), b1 = vgetq_lane_f64(best, 1);
    std::int64_t a0 = vgetq_lane_s64(at, 0), a1 = vgetq_lane_s64(at, 1);
    double b = b0;
    std::int64_t a = a0;
    if (Max ? (b1 > b0 || (b1 == b0 && a1 < a0)) : (b1 < b0 || (b1 == b0 && a1 < a0))) {
        b = b1;
        a = a1;
    }
    for (; i < n; ++i) {  // the odd last element
        double v = x[i];
        if constexpr (SkipNan) {
            if (v != v) v = far;
        } else {
            if (std::isnan(v)) return i;
        }
        if (Max ? v > b : v < b) {
            b = v;
            a = i;
        }
    }
    return a;
}

// int64, contiguous: two passes over the data -- the extreme value with four
// independent compare-select chains, then the first index holding it, four
// compares at a time (stopping there; ties keep the first index by
// construction). The one-pass index tracking was a chain of three dependent
// selects per vector (10^6 elements: scalar 493 us, one pass 222, numpy 125).
template <bool Max>
std::int64_t neon_arg_extreme_i64(const std::int64_t* x, std::int64_t n) {
    auto pick = [](int64x2_t a, int64x2_t b) { return vbslq_s64(Max ? vcgtq_s64(a, b) : vcltq_s64(a, b), a, b); };
    int64x2_t m0 = vld1q_s64(x), m1 = vld1q_s64(x + 2), m2 = m0, m3 = m1;
    std::int64_t i = 4;
    for (; i + 8 <= n; i += 8) {
        m0 = pick(vld1q_s64(x + i), m0);
        m1 = pick(vld1q_s64(x + i + 2), m1);
        m2 = pick(vld1q_s64(x + i + 4), m2);
        m3 = pick(vld1q_s64(x + i + 6), m3);
    }
    const int64x2_t m = pick(pick(m0, m1), pick(m2, m3));
    std::int64_t best = vgetq_lane_s64(m, 0);
    const std::int64_t b1 = vgetq_lane_s64(m, 1);
    if (Max ? b1 > best : b1 < best) best = b1;
    for (; i < n; ++i)
        if (Max ? x[i] > best : x[i] < best) best = x[i];
    const int64x2_t bv = vdupq_n_s64(best);
    std::int64_t k = 0;
    for (; k + 4 <= n; k += 4) {
        const uint64x2_t e = vorrq_u64(vceqq_s64(vld1q_s64(x + k), bv), vceqq_s64(vld1q_s64(x + k + 2), bv));
        if (vmaxvq_u32(vreinterpretq_u32_u64(e))) break;
    }
    for (;; ++k)
        if (x[k] == best) return k;
}

template <typename T, bool Max, bool SkipNan = false>
std::int64_t arg_extreme(const std::byte* p, std::int64_t n, std::int64_t stride) {
    if constexpr (ops::is_float<T>) {
        if (stride == sizeof(double) && n >= 2)
            return neon_arg_extreme<Max, SkipNan>(reinterpret_cast<const double*>(p), n);
    }
    if constexpr (std::is_same_v<T, std::int64_t>) {
        if (stride == sizeof(std::int64_t) && n >= 8)
            return neon_arg_extreme_i64<Max>(reinterpret_cast<const std::int64_t*>(p), n);
    }
    [[maybe_unused]] const double far =
        Max ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    auto load = [&](const std::byte* q) {
        T v = load_as<T>(q);
        if constexpr (SkipNan && ops::is_float<T>) {
            if (v != v) v = far;
        }
        return v;
    };
    T best = load(p);
    std::int64_t at = 0;
    if constexpr (ops::is_float<T> && !SkipNan) {
        if (std::isnan(best)) return 0;  // the first NaN wins outright
    }
    p += stride;
    for (std::int64_t i = 1; i < n; ++i, p += stride) {
        const T v = load(p);
        if constexpr (ops::is_float<T> && !SkipNan) {
            if (std::isnan(v)) return i;
        }
        if (Max ? v > best : v < best) {
            best = v;
            at = i;
        }
    }
    return at;
}

// ---- cumulative sum / product along one lane ----
// The first output is the first input itself (not identity + x), so -0.0
// survives, as in numpy.

// SkipNan (nancumulative_*): x's NaNs read as the identity, as in reduce_loop.
template <typename Op, typename T, bool SkipNan = false>
inline T scan_value(T v) {
    if constexpr (SkipNan) return nan_as(v, std::is_same_v<Op, ops::Multiply> ? T(1) : T(0));
    else return v;
}

template <typename Op, typename T, bool SkipNan = false>
void scan_serial(const std::byte* x, std::int64_t xs, std::byte* o, std::int64_t os, std::int64_t n) {
    if (n == 0) return;
    T acc = scan_value<Op, T, SkipNan>(load_as<T>(x));
    *reinterpret_cast<T*>(o) = acc;
    for (std::int64_t i = 1; i < n; ++i) {
        x += xs;
        o += os;
        acc = Op::f(acc, scan_value<Op, T, SkipNan>(load_as<T>(x)));
        *reinterpret_cast<T*>(o) = acc;
    }
}

// Long lanes are scanned in blocks of kScanBlock elements (docs/design/array.md §6.7):
//   1. each block is scanned on its own, from its own first element;
//   2. carry[b] = the end of everything before block b, from the block ends;
//   3. every element of block b >= 1 becomes carry[b] (op) its local value.
// Step 1 is one dependency chain per block (an add or multiply's latency per
// element, 0.5 ns); four blocks interleaved in one loop keep four chains in
// flight, and blocks run in parallel on the pool. The blocks depend on n
// only, so the result is the same for every thread count; for floats it is
// not numpy's bits. Integer scans wrap and give the same bits either way:
// they only take this path when it can run in parallel, since step 3 is an
// extra pass.
inline constexpr std::int64_t kScanBlock = 4096;
inline constexpr std::int64_t kScanGroup = 4;  // blocks interleaved in one loop (and one task)

// Contig: both strides are sizeof(T), known to the compiler (a runtime
// stride made the loop 2.3x slower, measured).
template <typename Op, typename T, bool Contig, bool SkipNan = false>
void scan_group(
    const std::byte* x,
    std::int64_t xs,
    std::byte* o,
    std::int64_t os,
    std::int64_t first,
    std::int64_t blocks,
    std::int64_t n
) {
    if constexpr (Contig) xs = os = sizeof(T);
    std::int64_t b = first;
    if (blocks == kScanGroup && (first + kScanGroup) * kScanBlock <= n) {  // four full blocks side by side
        // Eight elements of every block are loaded, scanned, then stored
        // together. One store per block in turn ran 5.7x slower when `o` was
        // freshly allocated memory (first-touch page faults; 1.7 ms against
        // 0.3 ms for 1M doubles, docs/design/array.md §6.7).
        const std::byte* xp[kScanGroup];
        std::byte* op[kScanGroup];
        T acc[kScanGroup];
        for (int j = 0; j < kScanGroup; ++j) {
            xp[j] = x + (first + j) * kScanBlock * xs;
            op[j] = o + (first + j) * kScanBlock * os;
        }
        // Typed accesses when contiguous: through load_as (a memcpy) the
        // compiler kept v on the stack, three memory operations per element.
        auto ld = [&](int j, std::int64_t i) {
            if constexpr (Contig) return scan_value<Op, T, SkipNan>(reinterpret_cast<const T*>(xp[j])[i]);
            else return scan_value<Op, T, SkipNan>(load_as<T>(xp[j] + i * xs));
        };
        auto st = [&](int j, std::int64_t i, T val) {
            if constexpr (Contig) reinterpret_cast<T*>(op[j])[i] = val;
            else *reinterpret_cast<T*>(op[j] + i * os) = val;
        };
        // First: each block starts from its own first element. A compile-time
        // flag, so every loop below has constant bounds and v stays in
        // registers (a runtime start index kept it on the stack, 2x slower).
        auto chunk = [&]<bool First>(std::int64_t i) {
            T v[kScanGroup][8];
            for (int j = 0; j < kScanGroup; ++j)
                for (int k = 0; k < 8; ++k) v[j][k] = ld(j, i + k);
            if constexpr (First)
                for (int j = 0; j < kScanGroup; ++j) acc[j] = v[j][0];
            for (int k = First ? 1 : 0; k < 8; ++k)
                for (int j = 0; j < kScanGroup; ++j) v[j][k] = acc[j] = Op::f(acc[j], v[j][k]);
            for (int j = 0; j < kScanGroup; ++j)
                for (int k = 0; k < 8; ++k) st(j, i + k, v[j][k]);
        };
        chunk.template operator()<true>(0);
        for (std::int64_t i = 8; i < kScanBlock; i += 8) chunk.template operator()<false>(i);
        return;
    }
    for (; b < first + blocks; ++b) {
        const std::int64_t begin = b * kScanBlock;
        scan_serial<Op, T, SkipNan>(x + begin * xs, xs, o + begin * os, os, std::min(kScanBlock, n - begin));
    }
}

template <typename Op, typename T, bool SkipNan = false>
void scan_lane(const std::byte* x, std::int64_t xs, std::byte* o, std::int64_t os, std::int64_t n) {
    const bool blocked =
        (ops::is_float<T> || ops::is_cplx<T>) ? n >= 2 * kScanBlock : n >= kParallelThreshold && parallel::available();
    if (!blocked) {
        scan_serial<Op, T, SkipNan>(x, xs, o, os, n);
        return;
    }
    const std::int64_t blocks = (n + kScanBlock - 1) / kScanBlock;
    const std::int64_t groups = (blocks + kScanGroup - 1) / kScanGroup;
    const bool contig = xs == sizeof(T) && os == sizeof(T);
    parallel::parallel_for(groups, [&](std::int64_t g) {
        const std::int64_t first = g * kScanGroup, count = std::min(kScanGroup, blocks - first);
        if (contig) scan_group<Op, T, true, SkipNan>(x, xs, o, os, first, count, n);
        else scan_group<Op, T, false, SkipNan>(x, xs, o, os, first, count, n);
    });
    auto at = [&](std::int64_t i) -> T& { return *reinterpret_cast<T*>(o + i * os); };
    auto carry = std::make_unique_for_overwrite<T[]>(static_cast<size_t>(blocks));
    carry[1] = at(kScanBlock - 1);
    for (std::int64_t b = 2; b < blocks; ++b) carry[b] = Op::f(carry[b - 1], at(b * kScanBlock - 1));
    parallel::parallel_for(groups, [&](std::int64_t g) {
        for (std::int64_t b = std::max<std::int64_t>(g * kScanGroup, 1); b < std::min(blocks, (g + 1) * kScanGroup);
             ++b) {
            const T c = carry[b];
            const std::int64_t end = std::min(n, (b + 1) * kScanBlock);
            if (os == sizeof(T)) {
                T* p = reinterpret_cast<T*>(o);
                for (std::int64_t i = b * kScanBlock; i < end; ++i) p[i] = Op::f(c, p[i]);
            } else {
                for (std::int64_t i = b * kScanBlock; i < end; ++i) at(i) = Op::f(c, at(i));
            }
        }
    });
}

}  // namespace sd
