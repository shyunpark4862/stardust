// The Riemann and Hurwitz zeta functions (docs/api/special.md §7,
// docs/design/special.md §5.6; S3). Hurwitz: Cephes zeta.c (Euler-Maclaurin:
// the first terms summed, then the tail as an integral and Bernoulli
// corrections). Riemann: Boost for x >= -1 and at the negative integers, a
// table at the even integers 2..52, the functional equation below -1 (the
// routes are listed above struct Zeta).
#pragma once

#include "special/boost.hpp"

#include "special/coeffs/cephes_gamma.hpp"
#include "special/coeffs/zeta_even.hpp"
#include "special/cost.hpp"
#include "special/gamma_core.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace zeta_detail {

// sum_k (k + q)^-s by Euler-Maclaurin (Cephes zeta.c), for s != 1. sm1 is
// s - 1 given exactly by the caller: the integral of the tail is
// (q + N)^(1-s) / (s - 1), and near s = 1 a rounded s - 1 would be the whole
// error. Cephes stops the direct sum once a term is negligible; for s < 1
// (Riemann only) the terms grow, and the sum runs to N = 9 before the
// corrections continue it analytically.
inline double euler_maclaurin(double s, double sm1, double q) {
    double sum = std::pow(q, -s);
    double a = q, b = 0.0;
    int i = 0;
    if (q > 0.0 && (sum == 0.0 || sum == kInf)) return sum;  // q^-s beyond the range: the later terms are nothing to it
    while (i < 9 || a <= 9.0) {  // q >= -kDirectBelow (hurwitz): at most 9 + 9 + 64 turns
        ++i;
        a += 1.0;
        b = std::pow(a, -s);
        sum += b;
        if (std::fabs(b / sum) < kEps && std::fabs(b * a / (sm1 * sum)) < kEps) return sum;
    }
    const double w = a;
    sum += b * w / sm1;
    sum -= 0.5 * b;
    double fac = 1.0, k = 0.0;
    for (int j = 0; j < 12; ++j) {
        fac *= s + k;
        b /= w;
        const double t = fac * b / cephes::kZetaA[static_cast<std::size_t>(j)];
        sum += t;
        if (std::fabs(t / sum) < kEps) break;
        k += 1.0;
        fac *= s + k;
        b /= w;
        k += 1.0;
    }
    return sum;
}


// A negative q at most this far from 0 is summed directly (euler_maclaurin's first loop:
// 9 - q terms at most).
inline constexpr double kDirectBelow = 64.0;

inline Scaled hurwitz_negative(double s, double sm1, double q);

// q < 0 is a half-integer: 2 (q + N) - 1 = 0, N = ceil(-q), formed exactly (q + N itself
// rounds for -1/2 < q < 0: q = -0.49999999999999994 gave 1/2).
inline double twice_fraction_less_one(double q) { return 2.0 * q + (2.0 * std::ceil(-q) - 1.0); }
inline bool half_integer(double q) { return twice_fraction_less_one(q) == 0.0; }

// Hurwitz zeta(s, q), s >= 1 (s = 1: inf), with Cephes' conventions: NaN for
// s < 1; inf at q = 0, -1, -2, ...; NaN for q < 0 when s is not an integer
// (q^-s is complex); q > 1e8 by the first terms of the asymptotic series.
// s = inf is the limit (S3 user decision: zeta(inf, q > 1) = 0; scipy NaN).
inline double hurwitz(double s, double sm1, double q) {
    if (std::isnan(s) || std::isnan(q)) return kNaN;
    if (sm1 == 0.0) return kInf;  // s - 1, not s: 1 + 1e-323 rounds to 1
    if (sm1 < 0.0) return kNaN;
    if (q <= 0.0) {
        if (q == std::floor(q)) return kInf;
        if (s != std::floor(s)) return kNaN;
    }
    if (s == kInf) {
        if (q > 1.0) return 0.0;
        if (q == 1.0) return 1.0;
        return kInf;
    }
    if (q > 1e8) {  // 1/(s-1) + 1/(2q) + s/(12 q^2), without overflowing q^2
        const double sq = s / q;
        return (1.0 / sm1 + (6.0 + sq) / q / 12.0) * std::pow(q, -sm1);
    }
    if (q < 0.0) {
        // (s an integer here.) The N = ceil(-q) terms below 0 are a difference of two sums
        // over positive arguments: with d = q + N in (0, 1),
        //     zeta(s, q) = zeta(s, d) + (-1)^s (zeta(s, 1 - d) - zeta(s, 1 - q)).
        // (d is exact for q <= -1/2; above, 1 + q may round)
        const double d = q + std::ceil(-q);
        const double sgn = std::fmod(s, 2.0) == 0.0 ? 1.0 : -1.0;
        // At a half-integer q with an odd s the two sums near 0 cancel exactly; summed term by
        // term they leave rounding noise 2^s 2^-53 (polygamma(20, -2.5) was -4.2e8 for -9.2e6),
        // or inf - inf for a large s (zeta(1025, -0.5)).
        if (half_integer(q) && sgn < 0.0) return hurwitz(s, sm1, 1.0 - q);
        // Far below 0 the direct sum would take |q| terms.
        const double r = q < -kDirectBelow ? (euler_maclaurin(s, sm1, d) + sgn * euler_maclaurin(s, sm1, 1.0 - d)) -
                                                 sgn * hurwitz(s, sm1, 1.0 - q)
                                           : euler_maclaurin(s, sm1, q);
        // The terms next to 0 beyond the doubles with opposite signs (an odd s): inf - inf.
        if (std::isnan(r)) return value(hurwitz_negative(s, sm1, q));
        return r;
    }
    return euler_maclaurin(s, sm1, q);
}

// zeta(s, q) as m 2^e, for q > 0 and s > 1 where q^-s may underflow
// (polygamma's n! zeta(n + 1, x) for large n and x): every term is taken
// relative to the first, q^-s, from pow_scaled, so the sum is an ordinary
// number and q^-s is applied once, scaled.
// zeta(s, q) q^s for 0 < q <= 1e8 and s > 1: the sum with every term relative to the
// first, (q / (q + k))^s <= 1.
inline double hurwitz_relative(double s, double sm1, double q) {
    // (a/q)^-s = e^(-s log1p(k/q)): no power that leaves the range
    auto rel = [&](double a) { return std::exp(-s * std::log1p((a - q) / q)); };
    double sum = 1.0, a = q, b = 0.0;
    int i = 0;
    while (i < 9) {
        ++i;
        a += 1.0;
        b = rel(a);
        sum += b;
        if (std::fabs(b / sum) < kEps && std::fabs(b * a / (sm1 * sum)) < kEps) return sum;
    }
    const double w = a;
    sum += b * w / sm1;
    sum -= 0.5 * b;
    double fac = 1.0, k = 0.0;
    for (int j = 0; j < 12; ++j) {
        fac *= s + k;
        b /= w;
        const double t = fac * b / cephes::kZetaA[static_cast<std::size_t>(j)];
        sum += t;
        if (std::fabs(t / sum) < kEps) break;
        k += 1.0;
        fac *= s + k;
        b /= w;
        k += 1.0;
    }
    return sum;
}

inline Scaled hurwitz_scaled(double s, double sm1, double q) {
    if (q < 0.0 && q != std::floor(q) && half_integer(q) && std::fmod(s, 2.0) == 1.0)
        return hurwitz_scaled(s, sm1, 1.0 - q);  // (hurwitz: the sums near 0 cancel, zeta(s, 1 - q) is left)
    if (q > 1e8 && s * std::log2(q) >= 900.0) {  // hurwitz's three terms, q^-(s-1) apart
        const double sq = s / q;
        return scaled(1.0 / sm1 + (6.0 + sq) / q / 12.0) * pow_scaled(q, -sm1);
    }
    if (q <= 0.0 || q > 1e8 || s * std::log2(q) < 900.0) return scaled(hurwitz(s, sm1, q));
    const Scaled first = pow_scaled(q, -s);
    if (first.m == 0.0) return first;  // q^-s beyond every range
    auto rel = [&](double a) { return value(pow_scaled(a, -s) / first); };  // (a/q)^-s, <= 1
    double sum = 1.0, a = q, b = 0.0;
    int i = 0;
    while (i < 9) {
        ++i;
        a += 1.0;
        b = rel(a);
        sum += b;
        if (std::fabs(b / sum) < kEps && std::fabs(b * a / (sm1 * sum)) < kEps) return first * scaled(sum);
    }
    const double w = a;
    sum += b * w / sm1;
    sum -= 0.5 * b;
    double fac = 1.0, k = 0.0;
    for (int j = 0; j < 12; ++j) {
        fac *= s + k;
        b /= w;
        const double t = fac * b / cephes::kZetaA[static_cast<std::size_t>(j)];
        sum += t;
        if (std::fabs(t / sum) < kEps) break;
        k += 1.0;
        fac *= s + k;
        b /= w;
        k += 1.0;
    }
    return first * scaled(sum);
}

// a + b as m 2^e.
inline Scaled add_scaled(Scaled a, Scaled b) {
    if (!std::isfinite(a.m) || !std::isfinite(b.m)) return {a.m + b.m, 0};
    if (a.m == 0.0) return b;
    if (b.m == 0.0) return a;
    const std::int64_t e = std::max(a.e, b.e);
    auto at = [&](Scaled v) { return std::ldexp(v.m, static_cast<int>(std::max<std::int64_t>(v.e - e, -1100))); };
    Scaled r = scaled(at(a) + at(b));
    r.e += e;
    return r;
}

// zeta(s, q) for q < 0 not an integer and s an integer, as m 2^e: hurwitz's three sums,
// zeta(s, d) + (-1)^s (zeta(s, e) - zeta(s, 1 - q)), d = q + N, e = 1 - d = -(q + N - 1)
// (exact), with the two next to 0 taken relative to the nearer one's first term x1^-s
// (x1 = min(d, e), x2 the other):
//     zeta(s, x1) +- zeta(s, x2) = x1^-s (1 +- r + R1 +- r R2),  r = (x1 / x2)^s,
// R the rest of each sum (hurwitz_relative - 1). For an odd s the pair is a difference,
// 1 - r = -expm1(s log1p((x1 - x2) / x2)) with x1 - x2 = -|2 d - 1| exact next to d = 1/2,
// where the two first terms nearly cancel. For a large s, where both are beyond the doubles
// (summed as doubles: inf - inf, polygamma(3000, -2.25) was NaN for inf).
inline Scaled hurwitz_negative(double s, double sm1, double q) {
    const double n = std::ceil(-q);
    const double d = q + n, e = -(q + (n - 1.0));
    const bool odd = std::fmod(s, 2.0) != 0.0;
    const double x1 = std::min(d, e), x2 = std::max(d, e);
    const double R1 = hurwitz_relative(s, sm1, x1) - 1.0, R2 = hurwitz_relative(s, sm1, x2) - 1.0;
    const double l = s * std::log1p(-std::fabs(twice_fraction_less_one(q)) / x2), r = std::exp(l);
    const double pair = odd ? -std::expm1(l) + (R1 - r * R2) : 1.0 + r + (R1 + r * R2);
    // the odd pair is zeta(s, d) - zeta(s, 1 - d): negative when 1 - d is the nearer
    const Scaled near = pow_scaled(x1, -s) * scaled(odd && x1 == e ? -pair : pair);
    const Scaled far = hurwitz_scaled(s, sm1, 1.0 - q);
    return add_scaled(near, odd ? far : Scaled{-far.m, far.e});
}

// zeta(x) for -60 < x < -1, x not an integer, by the functional equation
//     zeta(x) = (2 pi)^x / pi  sin(pi x / 2)  Gamma(1 + y)  zeta(1 + y),  y = -x,
// in plain doubles (every factor and partial product is far inside the range).
// sin_half is sinpi(x / 2), relatively accurate next to the trivial zeros.
// Gamma(1 + y) = y Gamma(y) with Lanczos, Gamma(y) = L(y) zgh^t e^-zgh,
// zgh = y + g - 1/2, t = y - 1/2 (exact for y > 1): y enters exactly, where
// rounding 1 + y first would cost up to y psi(y) 2^-53 (60 ULP for y in
// (31, 32), whose doubles are twice as dense as those of 1 + y). The two
// exponentials are one, e^(x log(2 pi) - zgh), with the exponent as a sum plus
// a remainder: log(2 pi) in two parts, zgh's rounding, and the sum's rounding
// are all applied as one first-order factor, together with zgh's rounding in
// the power, (1 + lo/hi)^t ~ 1 + t lo/hi. Only zeta(1 + y) sees the rounded
// 1 + y, and above 2 it moves by less than 1 ULP for that.
inline double reflect(double x, double sin_half) {
    const double y = -x;
    const double p = x * kLog2PiHi;
    const double perr = std::fma(x, kLog2PiHi, -p) + x * kLog2PiLo;
    const TwoSum zgh = two_sum(y, lanczos::kG - 0.5);  // g - 1/2 is exact
    const double t = y - 0.5;
    const TwoSum e = two_sum(p, -zgh.s);
    const double corr = (e.err + perr - zgh.err) + t * (zgh.err / zgh.s);
    const double g = std::pow(zgh.s, t) * std::exp(e.s);
    const double a = sin_half * bst::zeta(1.0 + y) * (y * lanczos_sum(y)) / kPi;
    return a * (g + g * corr);
}

}  // namespace zeta_detail

// Riemann zeta(x), by region:
//     |x| < 2^-30                  -1/2 - log(2 pi) x / 2
//     x = 2, 4, ..., 52            the table of correctly rounded values (coeffs/zeta_even.hpp)
//     x = -2, -4, ...              0 (the trivial zeros)
//     other x >= -1                Boost
//     x = -3, -5, ..., -59         Boost (its table of Bernoulli numbers)
//     other -60 < x < -1           the functional equation in plain doubles (zeta_detail::reflect)
//     x <= -60                     the functional equation with every factor scaled (m 2^e), so the
//                                  product overflows only where zeta does (beyond x = -260);
//                                  zeta(1 - x) by Euler-Maclaurin with s - 1 = -x exact
// A Boost result that is not a normal number is not used: x >= 0 then takes Euler-Maclaurin with
// q = 1, x < 0 the scaled functional equation.
struct Zeta : Real {
    static constexpr const char* name = "zeta";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kZeta;
    static double f(double x) {
        // x > 1 not an even integer of the table: Boost (below), first and without the other tests
        // (the rest kept out of line: 40% less time there)
        // (and -1 <= x < 1 away from 0, which general also gives to Boost first)
        if ((x > 1.0 && x < kInf && !(x <= kZetaEvenLast && 0.5 * x == std::floor(0.5 * x))) ||
            (x >= -1.0 && x < 1.0 && std::fabs(x) >= 0x1p-30)) [[likely]] {
            const double r = bst::zeta(x);
            if (bst::normal(r)) [[likely]]
                return r;
        }
        return general(x);
    }
    [[gnu::noinline]] static double general(double x) {
        if (std::isnan(x)) return x;
        if (x == 1.0) return kInf;
        if (x == kInf) return 1.0;
        if (x == -kInf) return kNaN;
        if (std::fabs(x) < 0x1p-30) return -0.5 - kLogSqrt2Pi * x;  // zeta(x) = -1/2 - log(2 pi) x / 2 + O(x^2)
        // The even integers 2..52 from the table: Boost takes the closed form
        // 2^(x-1) pi^x |B_x| / x! there and loses up to 9 ULP in pi^x, enough to put zeta(50)
        // below 1. Every other x > 1 is within 1.3 ULP from Boost (measured).
        if (x >= 2.0 && x <= kZetaEvenLast && 0.5 * x == std::floor(0.5 * x)) {
            return kZetaEven[static_cast<std::size_t>(0.5 * x) - 1];
        }
        // Boost down to -1: its own reflection is within 11 ULP there, 1 - x being exact or
        // nearly so. Below -1 it rounds 1 - x before Gamma and raises a rounded 2 pi to a power
        // (65 ULP in (-32, -31)), so only the integers, which it reads from a table, stay with it.
        if (x >= -1.0) {
            const double r = bst::zeta(x);
            if (bst::normal(r)) return r;
            if (x >= 0.0) return zeta_detail::euler_maclaurin(x, x - 1.0, 1.0);
        }
        const double s = sinpi(0.5 * x);
        if (s == 0.0) return 0.0;
        const double y = -x;
        if (y < 60.0) {
            if (y != std::floor(y)) {
                if (y > 1.0) return zeta_detail::reflect(x, s);
            } else {
                const double r = bst::zeta(x);
                if (bst::normal(r)) return r;
            }
        }
        if (y > 400.0) return std::copysign(kInf, s);
        const double p = x * kLog2PiHi;
        const double perr = std::fma(x, kLog2PiHi, -p) + x * kLog2PiLo;
        Scaled r = exp_scaled(p) * scaled(1.0 + perr);
        r = r * scaled(s * zeta_detail::hurwitz(1.0 + y, y, 1.0) / kPi);
        r = r * scaled(y) * gamma_scaled(y);
        return value(r);
    }
};

// Hurwitz zeta(x, q) (zeta's two-argument form, docs/api/special.md §7).
struct Hurwitz : Real {
    static constexpr const char* name = "zeta";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"x", "q"};
    static constexpr double cost_ns = cost::kHurwitz;
    static double f(double x, double q) { return zeta_detail::hurwitz(x, x - 1.0, q); }
};

}  // namespace sd::special
