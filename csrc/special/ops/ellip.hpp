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

// K(m). ellipk(1) = inf, ellipk(-inf) = 0, m > 1 is NaN (scipy).
struct Ellipk : Real {
    static constexpr const char* name = "ellipk";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"m"};
    static constexpr double cost_ns = cost::kEllipk;
    static double f(double m) {
        if (std::isnan(m) || m > 1.0) return kNaN;
        if (m == 1.0) return kInf;
        if (m == -kInf) return 0.0;
        if (m >= 0.0) return ellip_detail::ellpk(1.0 - m);
        const double w = 1.0 - m;  // > 1
        return ellip_detail::ellpk(1.0 / w) / std::sqrt(w);
    }
};

// E(m). ellipe(1) = 1, ellipe(-inf) = inf, m > 1 is NaN (scipy).
struct Ellipe : Real {
    static constexpr const char* name = "ellipe";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"m"};
    static constexpr double cost_ns = cost::kEllipe;
    static double f(double m) {
        if (std::isnan(m) || m > 1.0) return kNaN;
        if (m == 1.0) return 1.0;
        if (m == -kInf) return kInf;
        if (m >= 0.0) return ellip_detail::ellpe(1.0 - m);
        const double w = 1.0 - m;
        return std::sqrt(w) * ellip_detail::ellpe(1.0 / w);
    }
};

}  // namespace sd::special
