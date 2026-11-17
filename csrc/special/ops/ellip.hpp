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
// 1 - 1/x and then 1 minus that again). As accurate as the two methods below
// (max 2.3 / 2.0 ULP against their 2.9 / 4.2) and 4-8x faster (S5 user
// decision).
//
// Private, for comparison (the design's methods):
// _ellipk_agm: the arithmetic-geometric mean, K = pi / (2 AGM(1, sqrt(1 - m))),
// quadratic convergence in 5-7 steps.
// _ellipe_carlson: Carlson's symmetric form. The textbook one,
//     E = R_F(0, y, 1) - (m/3) R_D(0, y, 1),
// is K - (K - E): near m = 1 both terms grow like log(1/y) while E -> 1, a
// cancellation of ~20x at y = 1e-16. The form with two positive terms
// (DLMF 19.25.1, checked with mpmath)
//     E = (y/3) (R_D(0, y, 1) + R_D(0, 1, y))
// cancels nothing. For y < 2^-34 the expansion about m = 1,
//     E = 1 + (y/4)(log(16/y) - 1) + (3y^2/32)(log(16/y) - 13/6),
// is exact to rounding (and R_D's duplication would need log4(1/y) steps).
// R_D by Carlson's duplication (Carlson 1995, Numer. Algorithms 10), iterated
// until the remaining relative error is below 2^-56.
#pragma once

#include "special/coeffs/cephes_ellip.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/gamma_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace sd::special {

namespace ellip_detail {

// R_D(x, y, z) = (3/2) int_0^inf dt / ((t+x)^(1/2) (t+y)^(1/2) (t+z)^(3/2)), x, y >= 0, z > 0.
inline double carlson_rd(double x, double y, double z) {
    constexpr double kTolFactor = 812.7;  // (r / 4)^(-1/6) = 2^(58/6) for r = 2^-56
    const double x0 = x, y0 = y;
    double A = (x + y + 3.0 * z) / 5.0;
    const double A0 = A;
    double Q = kTolFactor * std::max({std::fabs(A - x), std::fabs(A - y), std::fabs(A - z)});
    double sum = 0.0, fac = 1.0, pow4 = 1.0;
    int it = 0;
    for (; it < 200 && pow4 * Q >= std::fabs(A); ++it) {  // pow4 falls 4x a turn: ~30 turns
        const double sx = std::sqrt(x), sy = std::sqrt(y), sz = std::sqrt(z);
        const double lam = sx * sy + sx * sz + sy * sz;
        sum += fac / (sz * (z + lam));
        fac *= 0.25;
        x = 0.25 * (x + lam);
        y = 0.25 * (y + lam);
        z = 0.25 * (z + lam);
        A = 0.25 * (A + lam);
        pow4 *= 0.25;
    }
    if (it == 200) return kNaN;
    const double X = (A0 - x0) * pow4 / A, Y = (A0 - y0) * pow4 / A;
    const double Z = -(X + Y) / 3.0;
    const double XY = X * Y, Z2 = Z * Z;
    const double E2 = XY - 6.0 * Z2;
    const double E3 = (3.0 * XY - 8.0 * Z2) * Z;
    const double E4 = 3.0 * (XY - Z2) * Z2;
    const double E5 = XY * Z2 * Z;
    const double series = 1.0 - 3.0 * E2 / 14.0 + E3 / 6.0 + 9.0 * E2 * E2 / 88.0 - 3.0 * E4 / 22.0 -
                          9.0 * E2 * E3 / 52.0 + 3.0 * E5 / 26.0;
    return pow4 / (A * std::sqrt(A)) * series + 3.0 * sum;
}

// E as a function of y = 1 - m, 0 < y <= 1.
inline double ellipe_y(double y) {
    if (y < 0x1p-34) {
        const double L = std::log(16.0 / y);
        return 1.0 + 0.25 * y * (L - 1.0) + (3.0 / 32.0) * y * y * (L - 13.0 / 6.0);
    }
    return (y / 3.0) * (carlson_rd(0.0, y, 1.0) + carlson_rd(0.0, 1.0, y));
}

// AGM(1, b) for b > 0 finite.
inline double agm1(double b) {
    double a = 1.0;
    for (int it = 0; it < 64; ++it) {  // quadratic: ~10 turns from any b
        if (std::fabs(a - b) <= 0x1p-52 * a) return 0.5 * (a + b);
        const double an = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = an;
    }
    return std::fabs(a - b) <= 0x1p-48 * a ? 0.5 * (a + b) : kNaN;
}

// Cephes ellpk, ellpe at y = 1 - m, 0 < y <= 1.
inline double ellpk(double y) {
    if (y <= kEps) return cephes::kEllpkC1 - 0.5 * std::log(y);  // MACHEP
    return horner(y, cephes::kEllpkP) - std::log(y) * horner(y, cephes::kEllpkQ);
}
inline double ellpe(double y) { return horner(y, cephes::kEllpeP) - std::log(y) * (y * horner(y, cephes::kEllpeQ)); }

}  // namespace ellip_detail

// K(m) by the AGM (private, for comparison).
struct EllipkAgm : Real {
    static constexpr const char* name = "_ellipk_agm";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"m"};
    static constexpr double cost_ns = cost::kEllipkAgm;
    static double f(double m) {
        if (std::isnan(m) || m > 1.0) return kNaN;
        if (m == 1.0) return kInf;
        if (m == -kInf) return 0.0;
        // AGM(1, sqrt(y)) with y up to 1e300: sqrt(a b) of the first step stays in range
        return kPi / (2.0 * ellip_detail::agm1(std::sqrt(1.0 - m)));
    }
};

// E(m) by Carlson's positive form (private, for comparison).
struct EllipeCarlson : Real {
    static constexpr const char* name = "_ellipe_carlson";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"m"};
    static constexpr double cost_ns = cost::kEllipeCarlson;
    static double f(double m) {
        if (std::isnan(m) || m > 1.0) return kNaN;
        if (m == 1.0) return 1.0;
        if (m == 0.0) return 0.5 * kPi;  // exactly pi/2, as scipy (the duplication gives it to 2 ULP)
        if (m == -kInf) return kInf;
        if (m >= 0.0) return ellip_detail::ellipe_y(1.0 - m);
        const double w = 1.0 - m;  // > 1
        return std::sqrt(w) * ellip_detail::ellipe_y(1.0 / w);
    }
};

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
