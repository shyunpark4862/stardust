// sd.stats' distribution methods over an array from sd.htest's kernels (the p-values of a batch,
// the fitted distribution's values at a sample), with the bits of one call per element.
#pragma once

#include "array/ops/pieces.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sd::htest {

// Method M of T at x[0, n) into out with the parameters a (one set for every element): T's chunk
// kernel over the whole array (stats/engine.hpp chunk_loop, the parameters at stride 0) where T
// has one, else method_of per element. The chunk kernel's value of an element depends on that
// element alone, and method_of is that kernel on a chunk of one, so every element has
// method_of's bits; the per-call cost of a chunk of one (~40 ns of a normal logcdf's ~50) goes.
template <class T, class M, class... A>
inline void method_array(const double* x, std::int64_t n, double* out, A... a) {
    constexpr std::size_t NP = sizeof...(A);
    if constexpr (M::template chunked<T, stats::Chunk<NP>>) {
        double pv[NP] = {a...};
        std::byte* args[NP + 2];
        std::int64_t st[NP + 2];
        for (std::size_t k = 0; k < NP; ++k) args[k] = reinterpret_cast<std::byte*>(&pv[k]), st[k] = 0;
        args[NP] = reinterpret_cast<std::byte*>(const_cast<double*>(x)), st[NP] = sizeof(double);
        args[NP + 1] = reinterpret_cast<std::byte*>(out), st[NP + 1] = sizeof(double);
        stats::chunk_loop<T, M>(args, n, st, nullptr);
    } else {
        for (std::int64_t i = 0; i < n; ++i) out[i] = stats::method_of<T, M>(x[i], a...);
    }
}

// The normal p-value of each z (NaN for NaN; statsmodels' _zstat_generic2, scipy's): alt 2
// ('greater') the upper tail, 1 ('less') the lower, 0 two-sided twice the upper tail of |z|
// (at most 1): Normal.ccdf by its chunk kernel, the bits of one method_of call per element.
inline void z_pvalues(const double* z, std::int64_t n, int alt, double* out) {
    for_pieces(n, 1 << 12, n * 16, [&](std::int64_t, std::int64_t b, std::int64_t e) {
        double arg[256];
        for (std::int64_t i0 = b; i0 < e; i0 += 256) {
            const std::int64_t m = std::min<std::int64_t>(256, e - i0);
            for (std::int64_t i = 0; i < m; ++i) {
                const double v = z[i0 + i];
                arg[i] = alt == 2 ? v : (alt == 1 ? -v : std::fabs(v));
            }
            method_array<stats::Normal, stats::method::Ccdf>(arg, m, out + i0, 0.0, 1.0);
            for (std::int64_t i = 0; i < m; ++i) {
                const double v = z[i0 + i];
                if (v != v) out[i0 + i] = v;
                else if (alt == 0) out[i0 + i] = std::min(1.0, 2.0 * out[i0 + i]);
            }
        }
    });
}

}  // namespace sd::htest
