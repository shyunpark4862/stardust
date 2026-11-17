// The complex helpers of the S8 functions (ops/wofz.hpp, loggamma.hpp, lambertw.hpp):
// numpy's multiply and divide (array/kernels/scalar_ops.hpp: no C99 Annex G recovery,
// which these functions handle themselves), and log, log1p, exp, sqrt that keep the
// sign of a zero imaginary part (the side of a branch cut).
#pragma once

#include "array/kernels/scalar_ops.hpp"

#include <cmath>
#include <complex>

namespace sd::special::cx {

using cplx = std::complex<double>;

inline constexpr double kTwoPi = 6.283185307179586477;

inline cplx mul(cplx a, cplx b) { return ops::cmul(a, b); }
inline cplx div(cplx a, cplx b) { return ops::cdiv(a, b); }
// r / t for a real r (Smith's algorithm without the complex numerator's products).
inline cplx rdiv(double r, cplx t) {
    const double tr = t.real(), ti = t.imag();
    if (std::fabs(tr) >= std::fabs(ti)) {
        const double q = ti / tr, d = tr + ti * q;
        return {r / d, -r * q / d};
    }
    const double q = tr / ti, d = tr * q + ti;
    return {r * q / d, -r / d};
}
inline double norm1(cplx a) { return std::fabs(a.real()) + std::fabs(a.imag()); }
// log z: log|z| + i atan2(Im z, Re z) (atan2 keeps the side of the negative axis)
// (|z| past 2^1000 or below 2^-900 through a power of two: the modulus would overflow at
// the largest doubles and has only a few bits among the subnormals)
inline cplx clog(cplx z) {
    const double x = z.real(), y = z.imag();
    const double m = std::max(std::fabs(x), std::fabs(y));
    if (m > 0x1p1000 && m < kInf) return {std::log(std::hypot(x * 0x1p-100, y * 0x1p-100)) + 100.0 * 0.69314718055994530942, std::atan2(y, x)};
    if (m < 0x1p-900 && m > 0.0) return {std::log(std::hypot(x * 0x1p200, y * 0x1p200)) - 200.0 * 0.69314718055994530942, std::atan2(y, x)};
    return {std::log(std::hypot(x, y)), std::atan2(y, x)};
}
// log(1 + w) for small w: Re = log1p(2 Re w + |w|^2) / 2, Im = atan2(Im w, 1 + Re w)
inline cplx clog1p(cplx w) {
    const double a = w.real(), b = w.imag();
    return {0.5 * std::log1p(std::fma(a, a + 2.0, b * b)), std::atan2(b, 1.0 + a)};
}
inline cplx cexp(cplx z) {
    const double m = std::exp(z.real());
    return {m * std::cos(z.imag()), m * std::sin(z.imag())};
}
// The principal square root (Kahan): the sign of a zero imaginary part picks the side of the cut.
inline cplx csqrt(cplx z) {
    const double x = z.real(), y = z.imag();
    if (x == 0.0 && y == 0.0) return {0.0, y};
    const double t = std::sqrt(0.5 * (std::fabs(x) + std::hypot(x, y)));
    if (x >= 0.0) return {t, y / (2.0 * t)};
    return {std::fabs(y) / (2.0 * t), std::copysign(t, y)};
}

}  // namespace sd::special::cx
