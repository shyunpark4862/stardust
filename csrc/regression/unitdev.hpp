// The unit deviances' pieces without cancellation, shared by the GLM (regression/glm.cpp) and
// the penalized GLM (regression/penalized.cpp).
#pragma once

#include <cmath>

namespace sd::regression::unitdev {

// sum_{j=0}^{17} s2^j / (2 j + 3) by Estrin's scheme (a depth of five multiply-adds instead
// of Horner's eighteen: the rows' evaluations are latency-bound). Every term is positive, so
// the order adds no cancellation (a few units in the last place, as Horner's).
inline double atanh_tail(double x) {
    constexpr double c[18] = {1.0 / 3,  1.0 / 5,  1.0 / 7,  1.0 / 9,  1.0 / 11, 1.0 / 13, 1.0 / 15, 1.0 / 17, 1.0 / 19,
                              1.0 / 21, 1.0 / 23, 1.0 / 25, 1.0 / 27, 1.0 / 29, 1.0 / 31, 1.0 / 33, 1.0 / 35, 1.0 / 37};
    const double x2 = x * x, x4 = x2 * x2, x8 = x4 * x4, x16 = x8 * x8;
    const double p0 = c[0] + c[1] * x, p1 = c[2] + c[3] * x, p2 = c[4] + c[5] * x, p3 = c[6] + c[7] * x,
                 p4 = c[8] + c[9] * x, p5 = c[10] + c[11] * x, p6 = c[12] + c[13] * x, p7 = c[14] + c[15] * x,
                 p8 = c[16] + c[17] * x;
    const double q0 = p0 + p1 * x2, q1 = p2 + p3 * x2, q2 = p4 + p5 * x2, q3 = p6 + p7 * x2;
    const double r0 = q0 + q1 * x4, r1 = q2 + q3 * x4;
    return (r0 + r1 * x8) + p8 * x16;
}

// r - log1p(r) (>= 0) for r > -1, x = 1 + r given exactly. On [-1/2, 1] by the atanh series
// in s = r/(2 + r): log1p(r) = 2 atanh(s) and r - 2s = r s, so r - log1p(r) = r s - 2 s^3
// (1/3 + s^2/5 + ...), no cancellation (|s| <= 1/3: 18 terms leave < 2^-56).
inline double mu_fn(double r, double x) {
    if (r >= -0.5 && r <= 1.0) {
        const double s = r / (2.0 + r), s2 = s * s;
        return r * s - 2.0 * s * s2 * atanh_tail(s2);
    }
    return r > 1.0 ? r - std::log1p(r) : r - std::log(x);
}

// (1 + r) log1p(r) - r = x log x - (x - 1) (>= 0), x = 1 + r = y/mu: r^2 - (1 + r) mu_fn(r)
// on [-1/2, 1] (cancellation at most 4 times), x log x - r outside.
inline double phi_fn(double r, double x) {
    if (r >= -0.5 && r <= 1.0) return r * r - (1.0 + r) * mu_fn(r, x);
    return x * std::log(x) - r;
}

}  // namespace sd::regression::unitdev
