// The arithmetic of the sparse products and sums (docs/design/sparse.md §2.6): one term at a time,
// and how the partial sums of a long line are put together.
//
// A sum over one output element takes its terms in ascending index order and adds them one by one
// from zero. A product term is fma(a, x, acc) for float64 (scipy's csr_matvec, which clang
// contracts to an fma), the wrapping integer product and sum for int64, and acc + a*x with numpy's
// complex product (ops::cmul) for complex128 (measured: scipy's complex csr_matvec is that, bit
// for bit). The sparse value is always the left factor, whatever the format, so the bits do not
// depend on the storage. A line of more than kLine terms is cut into pieces of kLine terms, each
// summed in order, and the piece sums are added along base/pairwise.hpp's tree, which depends on
// the number of pieces only.
#pragma once

#include "array/kernels/scalar_ops.hpp"
#include "base/dtype.hpp"
#include "base/pairwise.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sd::sparse {

// L of docs/design/sparse.md §2.6.
inline constexpr std::int64_t kLine = 4096;

template <class T>
inline T madd(T a, T x, T acc) {
    if constexpr (std::is_same_v<T, double>) return std::fma(a, x, acc);
    else if constexpr (std::is_same_v<T, std::int64_t>) return ops::wrap_add(acc, ops::wrap_mul(a, x));
    else {
        const complex128 p = ops::cmul(a, x);
        return {acc.real() + p.real(), acc.imag() + p.imag()};
    }
}

template <class T>
inline T plus(T acc, T a) {
    if constexpr (std::is_same_v<T, std::int64_t>) return ops::wrap_add(acc, a);
    else if constexpr (std::is_same_v<T, complex128>) return {acc.real() + a.real(), acc.imag() + a.imag()};
    else return acc + a;
}

// The sum of the piece sums parts[0], parts[s], parts[2 s], ... (n of them) along the pairwise tree
// (integers: in order; their sum does not depend on the order).
template <class T>
T tree_sum(const T* parts, std::int64_t n, std::int64_t s = 1) {
    if constexpr (std::is_same_v<T, double>) {
        return pairwise::tree(0, n, [&](std::int64_t b, std::int64_t m) {
            return pairwise::leaf(m, [&](std::int64_t i) { return parts[(b + i) * s]; });
        });
    } else if constexpr (std::is_same_v<T, complex128>) {
        return pairwise::tree_complex(0, n, [&](std::int64_t b, std::int64_t m) {
            return pairwise::leaf_complex(m, [&](std::int64_t i) { return parts[(b + i) * s]; });
        });
    } else {
        T acc{};
        for (std::int64_t i = 0; i < n; ++i) acc = plus(acc, parts[i * s]);
        return acc;
    }
}

// The sum over entries [b, e) of one line in order: step(p, acc) -> acc with entry p's term added.
// Past kLine terms, pieces of kLine summed in order and joined along the tree; `parts` is scratch.
template <class T, class Step>
inline T ordered_sum(std::int64_t b, std::int64_t e, const Step& step, std::vector<T>& parts) {
    const std::int64_t n = e - b;
    if (n <= kLine) {
        T acc{};
        for (std::int64_t p = b; p < e; ++p) acc = step(p, acc);
        return acc;
    }
    const std::int64_t nch = (n + kLine - 1) / kLine;
    parts.resize(static_cast<std::size_t>(nch));
    for (std::int64_t ch = 0; ch < nch; ++ch) {
        T acc{};
        const std::int64_t pe = std::min(e, b + (ch + 1) * kLine);
        for (std::int64_t p = b + ch * kLine; p < pe; ++p) acc = step(p, acc);
        parts[static_cast<std::size_t>(ch)] = acc;
    }
    return tree_sum(parts.data(), nch);
}

// y[r * ys] = sum over line r's entries p of v[p] * x[ix[p] * xs] (ordered_sum's sum, bit for bit)
// for the lines r in [lo, hi): four lines at a time, each line's terms in its own order and its own
// accumulator, so the four dependent chains of fused multiply-adds overlap (a line of a few entries
// is otherwise one chain's latency per term). Lines past kLine terms take ordered_sum.
template <class T>
inline void line_dots(
    std::int64_t lo,
    std::int64_t hi,
    const std::int64_t* ip,
    const std::int64_t* ix,
    const T* v,
    const T* x,
    std::int64_t xs,
    T* y,
    std::int64_t ys,
    std::vector<T>& parts
) {
    auto step = [v, ix, x, xs](std::int64_t p, T acc) { return madd(v[p], x[ix[p] * xs], acc); };
    std::int64_t r = lo;
    for (; r + 4 <= hi; r += 4) {
        const std::int64_t s0 = ip[r], s1 = ip[r + 1], s2 = ip[r + 2], s3 = ip[r + 3], e3 = ip[r + 4];
        const std::int64_t n0 = s1 - s0, n1 = s2 - s1, n2 = s3 - s2, n3 = e3 - s3;
        if (std::max(std::max(n0, n1), std::max(n2, n3)) > kLine) {
            for (std::int64_t q = r; q < r + 4; ++q) y[q * ys] = ordered_sum<T>(ip[q], ip[q + 1], step, parts);
            continue;
        }
        const std::int64_t m = std::min(std::min(n0, n1), std::min(n2, n3));
        T a0{}, a1{}, a2{}, a3{};
        for (std::int64_t j = 0; j < m; ++j) {
            a0 = step(s0 + j, a0);
            a1 = step(s1 + j, a1);
            a2 = step(s2 + j, a2);
            a3 = step(s3 + j, a3);
        }
        for (std::int64_t p = s0 + m; p < s1; ++p) a0 = step(p, a0);
        for (std::int64_t p = s1 + m; p < s2; ++p) a1 = step(p, a1);
        for (std::int64_t p = s2 + m; p < s3; ++p) a2 = step(p, a2);
        for (std::int64_t p = s3 + m; p < e3; ++p) a3 = step(p, a3);
        y[r * ys] = a0;
        y[(r + 1) * ys] = a1;
        y[(r + 2) * ys] = a2;
        y[(r + 3) * ys] = a3;
    }
    for (; r < hi; ++r) y[r * ys] = ordered_sum<T>(ip[r], ip[r + 1], step, parts);
}

}  // namespace sd::sparse
