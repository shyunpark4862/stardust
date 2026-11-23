// The complete elliptic integrals (docs/api/special.md §10,
// docs/design/special.md §5.10; S5), in scipy's parameter m = k^2:
//
//     ellipk(m) = K(m) = int_0^(pi/2) (1 - m sin^2 t)^(-1/2) dt
//     ellipe(m) = E(m) = int_0^(pi/2) (1 - m sin^2 t)^(1/2) dt
//
// Public: Cephes' polynomials in y = 1 - m (coeffs/cephes_ellip.hpp),
//     K = P(y) - log(y) Q(y),   E = P(y) - log(y) y Q(y),
// with the log singularity at m = 1 factored out, so nothing cancels; 1 - m
// is exact from m = 1/2 up (Sterbenz). For m < 0, y = 1/(1-m) formed
// directly and K(m) = K(y) / sqrt(1-m), E(m) = E(y) sqrt(1-m) (scipy forms
// 1 - 1/x and then 1 minus that again). Measured against the design's methods
// (S5 user decision, docs/design/special.md §5.10) -- K by the arithmetic-
// geometric mean, E by Carlson's symmetric form with two positive terms
// (DLMF 19.25.1) -- as accurate (max 2.3 / 2.0 ULP against their 2.9 / 4.2)
// and 4-8x faster.
#pragma once

#include "special/coeffs/cephes_ellip.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/kernels/chunked.hpp"
#include "special/kernels/simd.hpp"

#include <Accelerate/Accelerate.h>

#include <array>
#include <cmath>

namespace sd::special {

namespace ellip_detail {

// Cephes ellpk, ellpe at y = 1 - m, 0 < y <= 1.
inline double ellpk(double y) {
    if (y <= kEps) return cephes::kEllpkC1 - 0.5 * std::log(y);  // MACHEP
    return horner(y, cephes::kEllpkP) - std::log(y) * horner(y, cephes::kEllpkQ);
}
inline double ellpe(double y) { return horner(y, cephes::kEllpeP) - std::log(y) * (y * horner(y, cephes::kEllpeQ)); }

}  // namespace ellip_detail

// The NEON + vForce chunk shared by K and E (kernels/chunked.hpp): y = 1 - m for m >= 0 (1/(1-m)
// for m < 0) in lanes, log y by one vvlog over the chunk, the polynomials in lanes (simd::horner,
// the scalar's roundings), and the special values (NaN, m >= 1, -inf) element by element: found
// before anything is written (o may be m itself, out= the input) and stored after.
// 2.5x the scalar libm loop.
template <bool kE>
inline void ellip_chunk(const double* m, double* o, int n) {
    using simd::F2;
    alignas(16) double y[kChunk], ly[kChunk];
    auto ybar = [](auto mv) {
        using V = decltype(mv);
        const V w = simd::cst<V>(1.0) - mv;
        return simd::select(mv >= simd::cst<V>(0.0), w, simd::cst<V>(1.0) / w);
    };
    int special_at[kChunk], specials = 0;
    double special_value[kChunk];
    for (int j = 0; j < n; ++j) {
        const double v = m[j];
        if (v < 1.0 && v > -kInf) [[likely]]
            continue;
        special_at[specials] = j;
        if (std::isnan(v) || v > 1.0) special_value[specials] = kNaN;
        else if (v == 1.0) special_value[specials] = kE ? 1.0 : kInf;
        else special_value[specials] = kE ? kInf : 0.0;  // -inf
        ++specials;
    }
    int i = 0;
    for (; i + 2 <= n; i += 2) simd::store(y + i, ybar(simd::load(m + i, F2{})));
    for (; i < n; ++i) y[i] = ybar(m[i]);
    vvlog(ly, y, &n);
    auto value = [](auto mv, auto yv, auto lv) {
        using V = decltype(mv);
        const V w = simd::cst<V>(1.0) - mv;
        if constexpr (kE) {
            const V e = simd::horner(yv, cephes::kEllpeP) - lv * (yv * simd::horner(yv, cephes::kEllpeQ));
            return simd::select(mv >= simd::cst<V>(0.0), e, simd::sqrt(w) * e);
        } else {
            const V r = simd::horner(yv, cephes::kEllpkP) - lv * simd::horner(yv, cephes::kEllpkQ);
            const V k = simd::select(
                yv <= simd::cst<V>(kEps), simd::cst<V>(cephes::kEllpkC1) - simd::cst<V>(0.5) * lv, r
            );  // (MACHEP: Cephes' form there)
            return simd::select(mv >= simd::cst<V>(0.0), k, k / simd::sqrt(w));
        }
    };
    i = 0;
    for (; i + 2 <= n; i += 2)
        simd::store(o + i, value(simd::load(m + i, F2{}), simd::load(y + i, F2{}), simd::load(ly + i, F2{})));
    for (; i < n; ++i) o[i] = value(m[i], y[i], ly[i]);
    for (int j = 0; j < specials; ++j) o[special_at[j]] = special_value[j];
}

// K(m). ellipk(1) = inf, ellipk(-inf) = 0, m > 1 is NaN (scipy).
struct Ellipk : Real {
    static constexpr const char* name = "ellipk";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"m"};
    static constexpr double cost_ns = cost::kEllipk;
    static void chunk(const double* m, double* o, int n) { ellip_chunk<false>(m, o, n); }
};

// E(m). ellipe(1) = 1, ellipe(-inf) = inf, m > 1 is NaN (scipy).
struct Ellipe : Real {
    static constexpr const char* name = "ellipe";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"m"};
    static constexpr double cost_ns = cost::kEllipe;
    static void chunk(const double* m, double* o, int n) { ellip_chunk<true>(m, o, n); }
};

}  // namespace sd::special
