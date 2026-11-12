// Continued fractions by the modified Lentz method (docs/design/special.md
// §4; Numerical Recipes, 3rd ed., §5.2; Lentz 1976, Thompson & Barnett 1986).
//
//     f = b0 + a1 / (b1 + a2 / (b2 + a3 / (b3 + ...)))
//
// Evaluating from the tail needs the number of terms in advance. Lentz runs
// forward instead, keeping the ratios C_n = A_n / A_(n-1) and
// D_n = B_(n-1) / B_n of the convergents A_n / B_n, so f_n = f_(n-1) C_n D_n,
// and stops when C_n D_n is 1 to within `tol`. A zero denominator on the way
// would divide by zero; the "modified" method replaces it by a tiny number,
// which the next step cancels.
//
// With b0 = 0 the start value is that tiny number and every later f_n a
// multiple of it: a result much smaller than `tiny` (2^-1000) comes out
// wrong. Evaluate the tail b1 + a2 / (b2 + ...) as its own fraction then, and
// divide: f = a1 / tail.
//
// The incomplete gamma and beta functions (S4) and the Bessel ratios (S6)
// use it.
#pragma once

#include "special/common.hpp"

#include <cmath>
#include <utility>

namespace sd::special {

struct CfResult {
    double value;
    int terms;       // terms used
    bool converged;  // false: stopped at max_terms
};

// `term(n)` returns {a_n, b_n} for n = 1, 2, ...
template <class Term>
CfResult lentz(double b0, Term&& term, int max_terms, double tol = kEps) {
    constexpr double tiny = 0x1p-1000;  // stands in for a zero denominator
    double f = b0 == 0.0 ? tiny : b0;
    double c = f;
    double d = 0.0;
    for (int n = 1; n <= max_terms; ++n) {
        const auto [a, b] = term(n);
        d = b + a * d;
        if (d == 0.0) d = tiny;
        c = b + a / c;
        if (c == 0.0) c = tiny;
        d = 1.0 / d;
        const double delta = c * d;
        f *= delta;
        if (std::fabs(delta - 1.0) <= tol) return {f, n, true};
    }
    return {f, max_terms, false};
}

}  // namespace sd::special
