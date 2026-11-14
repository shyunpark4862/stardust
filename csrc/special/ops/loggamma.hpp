// loggamma: the principal branch of log Gamma (docs/api/special.md §12,
// docs/design/special.md §5.13; S8). For complex z the branch is the one
// continuous in the plane cut along the negative real axis (Hare 1997; scipy and
// mpmath): Im loggamma(x + 0i) = -k pi on (-k, -k + 1), and loggamma(conj z) =
// conj loggamma(z), with the sign of a zero imaginary part choosing the side.
//
//   - Re z > 7 or |Im z| > 7: Stirling's series, 8 terms.
//   - 0.1 <= Re z <= 4, |Im z| <= 3 (around the zeros at 1 and 2, where the
//     recurrence below cancels: 293 ULP measured, scipy 439): steps down to
//     Re z <= 1.5 by loggamma(z) = log1p(z - 2) + loggamma(z - 1), then with
//     w = z - 1
//         loggamma(1 + w) = -euler w + sum_{j<=6} mu(w/j) + sum_k d_k w^k,
//     mu(t) = t - log(1 + t), d_k = (-1)^k zeta(k, 7) / k (coeffs/loggamma_taylor.hpp):
//     no part cancels another except at the zeros themselves, which the
//     log1p step keeps exact to first order.
//   - Re z < 0.1: the reflection log pi - log sin(pi z) - loggamma(1 - z) with the
//     branch's 2 pi i k (Hare, proposition 3.1).
//   - otherwise: the recurrence loggamma(z) = loggamma(z + n) - log(z (z+1) ...
//     (z+n-1)), counting the product's crossings of the negative axis.
// Contract (S8 user decision): relative 1e-14 for Re z > 0; |error| <= 1e-14
// max(1, |loggamma|) for Re z <= 0. A real argument: gammaln for x > 0, inf at
// +-0, NaN for x < 0 (scipy).
#pragma once

#include "special/coeffs/loggamma_taylor.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/gamma_core.hpp"
#include "special/ops/gamma.hpp"

#include <array>
#include <cmath>
#include <complex>

namespace sd::special {

namespace loggamma_detail {

using cplx = std::complex<double>;

inline constexpr double kHalfLog2Pi = 0.91893853320467274178;  // log(2 pi) / 2
inline constexpr double kLogPi = 1.1447298858494001741;
inline constexpr double kTwoPi = 6.283185307179586477;

inline cplx mul(cplx a, cplx b) { return ops::cmul(a, b); }
inline cplx div(cplx a, cplx b) { return ops::cdiv(a, b); }
inline cplx clog(cplx z) { return {std::log(std::hypot(z.real(), z.imag())), std::atan2(z.imag(), z.real())}; }

// log(1 + w) for small w: Re = log1p(2 Re w + |w|^2) / 2, Im = atan2(Im w, 1 + Re w).
inline cplx clog1p(cplx w) {
    const double a = w.real(), b = w.imag();
    return {0.5 * std::log1p(std::fma(a, a + 2.0, b * b)), std::atan2(b, 1.0 + a)};
}

// Stirling: (z - 1/2) log z - z + log(2 pi)/2 + sum B_2k / (2k (2k-1) z^(2k-1)).
inline constexpr std::array<double, 8> kStirling{
    -2.955065359477124183e-2, 6.4102564102564102564e-3, -1.9175269175269175269e-3, 8.4175084175084175084e-4,
    -5.952380952380952381e-4, 7.9365079365079365079e-4, -2.7777777777777777778e-3, 8.3333333333333333333e-2};
inline cplx stirling(cplx z) {
    const cplx rz = div(1.0, z), rzz = mul(rz, rz);
    cplx p = kStirling[0];
    for (std::size_t i = 1; i < kStirling.size(); ++i) p = mul(p, rzz) + kStirling[i];
    return mul(z - 0.5, clog(z)) - z + kHalfLog2Pi + mul(rz, p);
}

// mu(t) = t - log(1 + t): the series sum_{m>=2} (-1)^m t^m / m for |t| < 1/4
// (to t^28: 0.25^28/28 is 1e-19 of t^2/2), else directly.
inline cplx mu(cplx t) {
    if (std::fabs(t.real()) + std::fabs(t.imag()) < 0.25) {
        cplx s = 1.0 / 28;
        for (int m = 27; m >= 2; --m) s = (m % 2 == 0 ? 1.0 : -1.0) / m + mul(t, s);
        return mul(mul(t, t), s);
    }
    return t - clog1p(t);
}

// 0.1 <= Re z <= 4, |Im z| <= 3
inline cplx blob(cplx z) {
    cplx acc = 0.0;
    while (z.real() > 1.5) {
        acc += clog1p(z - 2.0);  // log(z - 1)
        z -= 1.0;
    }
    const cplx w = z - 1.0;  // |w| <= 3.14: the tail's terms fall like 0.45^k (to k = 50)
    cplx s = -kEuler * w;
    for (int j = 1; j <= kLoggammaJ; ++j) s += mu(w / static_cast<double>(j));
    cplx t = kLoggammaTail[0];
    for (std::size_t i = 1; i < kLoggammaTail.size(); ++i) t = mul(t, w) + kLoggammaTail[i];
    return acc + (s + mul(mul(w, w), t));
}

// Im z >= +0, 0.1 <= Re z <= 7
inline cplx recurrence(cplx z) {
    int flips = 0;
    bool neg = false;
    cplx prod = z;
    z += 1.0;
    while (z.real() <= 7.0) {
        prod = mul(prod, z);
        const bool n = std::signbit(prod.imag());
        if (n && !neg) ++flips;
        neg = n;
        z += 1.0;
    }
    return stirling(z) - clog(prod) - cplx{0.0, kTwoPi * flips};
}

inline cplx sinpi(cplx z) {  // sin(pi z) with exact reduction of Re z
    const double x = z.real(), py = kPi * z.imag();
    return {special::sinpi(x) * std::cosh(py), special::cospi(x) * std::sinh(py)};
}

inline cplx loggamma(cplx z);

inline cplx finite(cplx z) {
    const double x = z.real(), y = z.imag();
    if (x > 7.0 || std::fabs(y) > 7.0) return stirling(z);
    if (x < 0.1) {
        const double t = std::copysign(kTwoPi, y) * std::floor(0.5 * x + 0.25);
        return cplx{kLogPi, t} - clog(sinpi(z)) - loggamma(cplx{1.0 - x, -y});
    }
    if (x <= 4.0 && std::fabs(y) <= 3.0) return blob(z);
    if (!std::signbit(y)) return recurrence(z);
    return std::conj(recurrence(std::conj(z)));
}

inline cplx loggamma(cplx z) {
    const double x = z.real(), y = z.imag();
    if (std::isnan(x) || std::isnan(y)) return {kNaN, kNaN};
    if (std::isinf(x) || std::isinf(y)) {  // the limits where the direction fixes one (S8 user decision)
        if (std::isinf(y)) return {x == kInf ? kInf : -kInf, y};
        if (x > 0) return {kInf, y == 0.0 ? y : std::copysign(kInf, y)};
        if (y == 0.0) return {kNaN, kNaN};  // poles all along the negative axis
        return {-kInf, std::copysign(kInf, -y)};
    }
    if (y == 0.0 && x <= 0.0 && x == std::floor(x)) return {kInf, kNaN};  // a pole: Im depends on the direction
    return finite(z);
}

}  // namespace loggamma_detail

struct Loggamma {
    static constexpr const char* name = "loggamma";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kLoggammaC;
    template <typename T> static constexpr bool ok = ops::is_float<T> || ops::is_cplx<T>;
    static double f(double x) {
        if (std::isnan(x)) return x;
        if (x == 0.0) return kInf;  // both zeros (scipy)
        if (x < 0.0) return kNaN;
        return gamma_detail::lgamma_abs(x);
    }
    static std::complex<double> f(std::complex<double> z) { return loggamma_detail::loggamma(z); }
};

}  // namespace sd::special
