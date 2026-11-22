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
#include "special/cplx.hpp"
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
using cx::clog;
using cx::clog1p;
using cx::div;
using cx::kTwoPi;
using cx::mul;

// Stirling: (z - 1/2) log z - z + log(2 pi)/2 + sum B_2k / (2k (2k-1) z^(2k-1)).
inline constexpr std::array<double, 8> kStirling{-2.955065359477124183e-2,  6.4102564102564102564e-3,
                                                 -1.9175269175269175269e-3, 8.4175084175084175084e-4,
                                                 -5.952380952380952381e-4,  7.9365079365079365079e-4,
                                                 -2.7777777777777777778e-3, 8.3333333333333333333e-2};
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

// The tails for a split J < kLoggammaJ: sum_{j>J} mu(w / j) = sum_k (-1)^k zeta(k, J + 1) / k w^k,
// zeta(k, J + 1) = zeta(k, 7) + sum_{j=J+1}^{6} j^-k (positive parts: no cancellation), from the
// table's d_k (coeffs/loggamma_taylor.hpp) at compile time; k = 2..29, highest power first. Used
// while |w| / (J + 1) < 1/4, where the terms past k = 29 are below 2^-60 of the first.
inline constexpr int kShortTail = 28;
inline constexpr std::array<std::array<double, kShortTail>, kLoggammaJ> kShortTails = [] {
    std::array<std::array<double, kShortTail>, kLoggammaJ> t{};
    constexpr std::size_t n = kLoggammaTail.size();  // d_k at index n - 1 - (k - 2)
    for (int J = 0; J < kLoggammaJ; ++J)
        for (int k = 2; k < 2 + kShortTail; ++k) {
            double add = 0.0;
            for (int j = J + 1; j <= kLoggammaJ; ++j) {
                double p = 1.0;
                for (int i = 0; i < k; ++i) p /= j;
                add += p;
            }
            const double d = kLoggammaTail[n - 1 - static_cast<std::size_t>(k - 2)];
            t[static_cast<std::size_t>(J)][static_cast<std::size_t>(kShortTail - 1 - (k - 2))] =
                d + (k % 2 == 0 ? add : -add) / k;
        }
    return t;
}();

// log Gamma(1 + w) + euler w for Re w > -1, |w| <= 3.14 (blob): sum_{j<=J} mu(w / j) + the tail of
// the rest, J = min(6, floor(4 |w|_1)) so that each split-off t = w / j has |t|_1 >= 1/4 and the
// tail's ratio is |w| / (J + 1) (below 1/4 for J < 6, up to 0.45 at J = 6: 49 terms). The split-
// off mu = t - log(1 + t) as sum t - log prod (1 + t): one logarithm for them all (log1p of the
// product less 1, carried apart, where that is below 1/2). Every factor has
// Re > 0 and an argument of Im w's sign, so the product turns one way and its crossings of the
// negative axis are counted (as recurrence). (Before: each j apart, the small t by a 26-term
// series and the 49-term tail always; a third of the time now.)
inline cplx log_gamma1p_plus_euler(cplx w) {
    if (std::signbit(w.imag())) return std::conj(log_gamma1p_plus_euler(std::conj(w)));
    const int J = std::min(kLoggammaJ, static_cast<int>(4.0 * (std::fabs(w.real()) + std::fabs(w.imag()))));
    // the product as it is, and q = prod - 1 apart from the 1 (log1p's accuracy where it is small)
    cplx ts = 0.0, prod = 1.0, q = 0.0;
    int flips = 0;
    bool neg = false;
    for (int j = 1; j <= J; ++j) {
        const cplx t = w / static_cast<double>(j);
        ts += t;
        q += t + mul(q, t);
        prod = mul(prod, 1.0 + t);
        const bool n = std::signbit(prod.imag());
        if (n && !neg) ++flips;
        neg = n;
    }
    cplx tail;
    if (J == kLoggammaJ) {
        // the terms beyond k = K left out where they are below 2^-56 of the first: |d_k w^k| ~
        // rho^k / k, rho = |w| / 7, so K = 1 + 49 / log2(1/rho), by thresholds on |w|^2
        const double m2 = w.real() * w.real() + w.imag() * w.imag();
        const std::size_t K = m2 < 49.0 * 0.0625   ? 27
                              : m2 < 49.0 * 0.09   ? 31
                              : m2 < 49.0 * 0.1225 ? 35
                              : m2 < 49.0 * 0.16   ? 39
                              : m2 < 49.0 * 0.2025 ? 45
                                                   : kLoggammaTail.size() + 1;
        const std::size_t first = kLoggammaTail.size() + 1 - K;  // d_K's index (d_k at n - 1 - (k - 2))
        tail = kLoggammaTail[first];
        for (std::size_t i = first + 1; i < kLoggammaTail.size(); ++i) tail = mul(tail, w) + kLoggammaTail[i];
    } else {
        const auto& c = kShortTails[static_cast<std::size_t>(J)];
        tail = c[0];
        for (std::size_t i = 1; i < c.size(); ++i) tail = mul(tail, w) + c[i];
    }
    tail = mul(mul(w, w), tail);
    if (J == 0) return tail;
    const cplx l = std::fabs(q.real()) + std::fabs(q.imag()) < 0.5 ? clog1p(q) : clog(prod);
    return (ts - cplx{l.real(), l.imag() + kTwoPi * flips}) + tail;
}

// 0.1 <= Re z <= 4, |Im z| <= 3
inline cplx blob(cplx z) {
    cplx acc = 0.0;
    while (z.real() > 1.5) {
        acc += clog1p(z - 2.0);  // log(z - 1)
        z -= 1.0;
    }
    const cplx w = z - 1.0;  // |w| <= 3.14
    return acc + (-kEuler * w + log_gamma1p_plus_euler(w));
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

// log sin(pi z). Within 2^-500 of an integer n, sin(pi z) = (-1)^n pi d (d = z - n) to
// rounding: log pi + log d, the logarithm scaled (cx::clog) -- pi times a subnormal part
// rounds by percents (loggamma(-34 - 5e-324i) was off by 0.046).
inline cplx log_sinpi(cplx z) {
    const double n = std::nearbyint(z.real()), dx = z.real() - n;
    if (std::max(std::fabs(dx), std::fabs(z.imag())) < 0x1p-500) {
        const double sgn = std::fmod(n, 2.0) == 0.0 ? 1.0 : -1.0;
        const cplx l = clog(cplx{sgn * dx, sgn * z.imag()});
        return {kLogPi + l.real(), l.imag()};
    }
    return clog(sinpi(z));
}

inline cplx loggamma(cplx z);

inline cplx finite(cplx z) {
    const double x = z.real(), y = z.imag();
    if (x > 7.0 || std::fabs(y) > 7.0) return stirling(z);
    if (x < 0.1) {
        const double t = std::copysign(kTwoPi, y) * std::floor(0.5 * x + 0.25);
        return cplx{kLogPi, t} - log_sinpi(z) - loggamma(cplx{1.0 - x, -y});
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

// A real argument: gammaln for x > 0, inf at +-0, NaN for x < 0 (scipy). Its own Op so
// that a float64 call runs with gammaln's weight, not the complex one's.
struct LoggammaReal : Real {
    static constexpr const char* name = "loggamma";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kGammaln;
    static double f(double x) {
        if (std::isnan(x)) return x;
        if (x == 0.0) return kInf;  // both zeros (scipy)
        if (x < 0.0) return kNaN;
        return gamma_detail::lgamma_abs(x);
    }
};
struct Loggamma {
    static constexpr const char* name = "loggamma";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"z"};
    static constexpr double cost_ns = cost::kLoggammaC;
    template <typename T> static constexpr bool ok = ops::is_cplx<T>;
    static std::complex<double> f(std::complex<double> z) { return loggamma_detail::loggamma(z); }
};

}  // namespace sd::special
