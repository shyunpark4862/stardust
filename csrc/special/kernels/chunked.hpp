// The kernel shape of the NEON functions (docs/design/special.md §3.4, Q16):
// the input is processed in chunks of 1024 elements (8 KB, on the stack);
// each chunk is one call of Op::chunk(x, out, n), which mixes NEON passes
// with whole-chunk vForce calls (vvexp, vvlog, ...).
//
// Every layout takes this path: a strided input or a broadcast scalar is
// gathered into the chunk buffer first, a strided output scattered from one,
// and a single element is a chunk of n = 1. vForce's result for an element
// does not depend on its position or the call's length (docs/design/array.md
// §5.4), and the NEON passes are elementwise, so f(x)[i] == f(x[i]) holds
// whatever the layout -- there is no separate scalar definition to match.
#pragma once

#include "accelerate/vforce.hpp"
#include "array/kernels/elementwise.hpp"

#include <algorithm>
#include <cstdint>

namespace sd::special {

inline constexpr int kChunk = static_cast<int>(vforce_detail::kChunk);  // 1024

// One input, one output. Op::chunk may write out[i] only after it has read
// x[i] for the last time (out may be x: the very same view, ufunc.cpp).
template <typename Op>
void chunked_loop(std::byte* const* args, std::int64_t n, const std::int64_t* s, void*) {
    using namespace vforce_detail;
    double in[kChunk], out[kChunk];
    for (std::int64_t off = 0; off < n; off += kChunk) {
        const int m = static_cast<int>(std::min<std::int64_t>(kChunk, n - off));
        const double* src = gather(args[0] + off * s[0], s[0], m, in);
        const bool direct = s[1] == sizeof(double);
        double* dst = direct ? reinterpret_cast<double*>(args[1] + off * s[1]) : out;
        Op::chunk(src, dst, m);
        if (!direct) scatter(out, args[1] + off * s[1], s[1], m);
    }
}

// The loop table of such an Op: float64 only (bool and int64 inputs are
// converted on the way in, ufunc.cpp).
template <typename Op>
inline constexpr LoopTable chunked_table = {
    LoopEntry{}, LoopEntry{}, LoopEntry{&chunked_loop<Op>, DType::Float64}, LoopEntry{}
};

}  // namespace sd::special
