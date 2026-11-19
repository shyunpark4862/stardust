// The prefixes of the incomplete gamma and beta functions
// (docs/design/special.md §5.4; S4):
//
//     D(a, x)    = x^a e^-x / Gamma(a + 1)
//     B(a, b, x) = x^a (1 - x)^b / B(a, b)
//
// Each is a series or continued fraction away from the function, and its
// derivative (D a / x, B / (x (1 - x))), which the inverses' Halley steps use.
//
// Formed directly they overflow and lose accuracy: a log x - x - lgamma(a+1)
// cancels terms of size a log a down to the exponent E, and every bit lost
// there is multiplied by E in e^-E. So both are written with Stirling's
// normalisation Gamma(z) = sqrt(2 pi / z) (z/e)^z Gamma*(z), which puts every
// large part into one exponent that cannot cancel:
//
//     D(a, x)    = e^-E / (sqrt(2 pi a) Gamma*(a)),     E = a mu(x/a - 1)
//     B(a, b, x) = sqrt(ab / (2 pi c)) Gamma*(c) / (Gamma*(a) Gamma*(b)) e^-E,
//                  E = a mu(x c/a - 1) + b mu((1-x) c/b - 1),  c = a + b
//
// with mu(t) = t - log(1 + t) >= 0 (Temme's form; Boost's
// regularised_gamma_prefix and ibeta_power_terms do the same with Lanczos).
// E is a double-double (special/dd.hpp) from the exact inputs; Gamma*(z) is
// 1 + O(1/z) and harmless.
#pragma once

#include "special/common.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"

#include <algorithm>
#include <cmath>

namespace sd::special {

namespace prefix_detail {

inline constexpr double kSqrt2Pi = 2.50662827463100050242;
inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;

// Gamma(x) for 0 < x < 13 (Cephes gamma.c, as gamma_detail::gamma_cephes_pos).
inline double gamma_small(double x) {
    double z = 1.0;
    while (x >= 3.0) {
        x -= 1.0;
        z *= x;
    }
    while (x < 2.0) {
        if (x < 1e-9) return z / ((1.0 + kEuler * x) * x);
        z /= x;
        x += 1.0;
    }
    if (x == 2.0) return z;
    x -= 2.0;
    return z * horner(x, cephes::kGammaP) / horner(x, cephes::kGammaQ);
}

}  // namespace prefix_detail

// Gamma*(z) = Gamma(z) / (sqrt(2 pi / z) (z/e)^z), z > 0: exp of Stirling's
// series from 13 (Gamma*(inf) = 1), else Gamma(z + 1) e^(z (1 - log z)) / sqrt(2 pi z)
// with the exponent in double-double. Gamma(z + 1) = 1 / (1 + euler z) below 1e-9.
inline double gamma_star(double z) {
    if (z >= 13.0) {
        const double w = 1.0 / z;
        return std::exp(horner(w * w, cephes::kLgamA) * w);
    }
    const double g1 = z < 1e-9 ? 1.0 / (1.0 + kEuler * z) : z * prefix_detail::gamma_small(z);
    const dd::DD e = (dd::DD{1.0, 0.0} - dd::log(z)) * z;
    return g1 * (std::exp(e.hi) * (1.0 + e.lo)) / std::sqrt(2.0 * kPi * z);
}

// ---- incomplete gamma ---------------------------------------------------

// E = a mu(x/a - 1) = x - a - a log(x/a) >= 0 for a, x > 0 finite. lambda =
// x/a as a double-double while it is a normal number; else the logs apart.
inline dd::DD gamma_exponent(double a, double x) {
    const double q = x / a;
    if (q > 0x1p-1000 && q < 0x1p1000) return dd::a_mu(a, dd::DD{x, 0.0} / a);
    return dd::two_sum(x, -a) - (dd::log(x) - dd::log(a)) * a;
}

// e^-E (1 - E.lo) as m 2^e: D(a, x) without over- or underflow.
inline Scaled exp_minus(dd::DD E) {
    // 0 from 2^52 up (also E = inf): far below any m 2^e in use, and from 2^53 the low
    // part is no longer below 1, where 1 - E.lo would turn the sign
    if (!(E.hi < 0x1p52)) return {0.0, 0};
    if (std::fabs(E.hi) < 700.0) return scaled(std::exp(-E.hi) * (1.0 - E.lo));  // a normal double: no split
    return exp_scaled(-E.hi) * scaled(1.0 - E.lo);
}

// D(a, x) = x^a e^-x / Gamma(a + 1) as m 2^e.
inline Scaled gamma_prefix(double a, dd::DD E) {  // E = gamma_exponent(a, x)
    if (std::fabs(E.hi) < 700.0 && a < 0x1p900)
        return scaled(std::exp(-E.hi) * (1.0 - E.lo) / (std::sqrt(2.0 * kPi * a) * gamma_star(a)));
    return exp_minus(E) / scaled(std::sqrt(2.0 * kPi * a) * gamma_star(a));
}

// D(a, x) from a and x. Below a = 13 the Stirling form buys nothing (Gamma* there is
// Gamma(a + 1) e^(a (1 - log a)), a second exponent): directly e^(a log x - x) /
// Gamma(a + 1), the exponent a double-double (132 ns -> 40 with the table log).
inline Scaled gamma_prefix(double a, double x) {
    if (a < 13.0 && a >= 1e-9 && x > 0x1p-1000 && x < 0x1p1000) {
        const dd::DD L = dd::log(x) * a - x;
        const double g1 = a * prefix_detail::gamma_small(a);  // Gamma(a + 1)
        if (std::fabs(L.hi) < 700.0) return scaled(std::exp(L.hi) * (1.0 + L.lo) / g1);
        return exp_scaled(L.hi) * scaled((1.0 + L.lo) / g1);
    }
    return gamma_prefix(a, gamma_exponent(a, x));
}

// ---- incomplete beta ----------------------------------------------------

// a mu(lambda - 1) for lambda = x c / a = x + x (b / a), x = hi + lo: the
// double-double lambda while it is a normal number, else
// a (lambda - 1) = x a + x b - a and log lambda = log x + log c - log a,
// with log c = log max + log(1 + min / max) (c may overflow).
inline dd::DD beta_term(double a, double b, dd::DD x) {
    const dd::DD r = dd::DD{b, 0.0} / a;
    if (std::isfinite(r.hi)) {
        const dd::DD lam = x + x * r;
        if (lam.hi > 0x1p-1000 && lam.hi < 0x1p1000) return dd::a_mu(a, lam);
    }
    const double big = std::max(a, b), small = std::min(a, b);
    const dd::DD logc = dd::log(big) + dd::log(dd::DD{1.0, 0.0} + dd::DD{small, 0.0} / big);
    const dd::DD lin = x * a + x * b - a;
    return lin - (dd::log(x) + logc - dd::log(a)) * a;
}

// E = a mu(x c/a - 1) + b mu(y c/b - 1), y = 1 - x; both double-doubles with
// x + y = 1 exactly.
inline dd::DD beta_exponent(double a, double b, dd::DD x, dd::DD y) { return beta_term(a, b, x) + beta_term(b, a, y); }

// sqrt(ab / (2 pi c)) Gamma*(c) / (Gamma*(a) Gamma*(b)), c = a + b:
// ab / c = min / (1 + min / max) and Gamma*(c) at c rounded (its
// sensitivity is 1/(12 c^2): nothing).
inline double beta_factor(double a, double b) {
    const double big = std::max(a, b), small = std::min(a, b);
    const double abc = small / (1.0 + small / big);
    return std::sqrt(abc) * prefix_detail::kInvSqrt2Pi * gamma_star(a + b) / (gamma_star(a) * gamma_star(b));
}

// B(a, b, x) = x^a y^b / B(a, b) as m 2^e.
inline Scaled beta_prefix(double a, double b, dd::DD E) { return exp_minus(E) * scaled(beta_factor(a, b)); }

}  // namespace sd::special
