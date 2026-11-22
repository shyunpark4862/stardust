// The machinery under the gamma family (docs/design/special.md §5.2; S3).
//
// Three ideas carry the accuracy:
//
// 1. Exact arguments. Gamma's relative sensitivity to its argument is
//    psi(z) z: rounding z to float64 (2^-53 relative) costs z log z / 2 ULP --
//    436 ULP at z = 170. So no formula forms 1 - x, a + b, n - k + 1 and then
//    calls Gamma on it. The reflection uses Gamma(x) Gamma(-x) = -pi / (x
//    sin(pi x)) (-x is exact), Gamma(x + 1) is x Gamma(x), and a sum that
//    must be formed is kept as hi + lo (two_sum) and corrected to first order:
//    Gamma(hi + lo) = Gamma(hi) (1 + psi(hi) lo).
// 2. sinpi, cospi with exact argument reduction: x mod 2 is exact in floating
//    point, so sin(pi x) keeps its relative accuracy next to every integer.
// 3. Scaled values m 2^e. Gamma(z) overflows at 171.6, but the quotients it
//    builds (1/Gamma down to subnormals, beta, zeta(x < 0), polygamma) are
//    ordinary numbers; carrying the binary exponent apart from the mantissa
//    lets them be formed without overflow and rounded once at the end.
//    Powers and exponentials with large results are split the same way
//    (pow_scaled, exp_scaled), keeping libm's accuracy on each piece.
#pragma once

#include "special/coeffs/cephes_gamma.hpp"
#include "special/coeffs/lanczos13m53.hpp"
#include "special/coeffs/lgamma_taylor2.hpp"
#include "special/common.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace sd::special {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kEuler = 0.57721566490153286061;
inline constexpr double kLogSqrt2Pi = 0.91893853320467274178;  // log(2 pi) / 2 (Cephes LS2PI)
inline constexpr double kSqrtHalfG = 0.70710678118654752440;
inline constexpr double kLn2Hi = 0.6931471805599453, kLn2Lo = 2.3190468138462996e-17;     // log 2 = hi + lo
inline constexpr double kLog2PiHi = 1.8378770664093456, kLog2PiLo = -7.756588316134483e-17;  // log(2 pi)

// ---- exact sums ------------------------------------------------------------

// s + err = a + b exactly (Knuth's TwoSum).
struct TwoSum {
    double s, err;
};
inline TwoSum two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

// ---- sinpi, cospi ----------------------------------------------------------

// sin(pi x) with exact argument reduction: r = x mod 2 (fmod is exact), then
// folded into [-1/2, 1/2] by exact subtractions (Sterbenz), so only pi r
// rounds. sinpi(integer) = +-0.
// On Apple the libm's __sinpi / __cospi (exact reduction too): 1 ULP at most against Arb where
// these take 2, at half the time; the same zeros, except the sign of sinpi at a negative odd
// integer (-0 there, IEEE 754's sinPi).
inline double sinpi(double x) {
#if defined(__APPLE__)
    return std::isfinite(x) ? __sinpi(x) : kNaN;
#endif
    if (!std::isfinite(x)) return kNaN;
    double r = std::fmod(x, 2.0);
    if (r > 1.0) r -= 2.0;
    else if (r < -1.0) r += 2.0;
    if (r > 0.5) r = 1.0 - r;
    else if (r < -0.5) r = -1.0 - r;
    return std::sin(kPi * r);
}

// cos(pi x) = sin(pi (1/2 - r)), r = |x| mod 2 folded into [0, 1].
inline double cospi(double x) {
#if defined(__APPLE__)
    return std::isfinite(x) ? __cospi(x) : kNaN;
#endif
    if (!std::isfinite(x)) return kNaN;
    double r = std::fmod(std::fabs(x), 2.0);
    if (r > 1.0) r = 2.0 - r;
    return std::sin(kPi * (0.5 - r));
}

// ---- scaled values m 2^e -------------------------------------------------

struct Scaled {
    double m;        // 0.5 <= |m| < 1, or 0, inf, NaN
    std::int64_t e;
};
// (frexp by the bits for a normal v: frexp and ldexp are calls, and a Scaled product, quotient or
// value made a third of gamma_scaled's time; the same m and e.)
inline Scaled scaled(double v) {
    const auto bits = std::bit_cast<std::uint64_t>(v);
    const auto be = static_cast<std::int64_t>((bits >> 52) & 0x7ff);
    if (be != 0 && be != 0x7ff) [[likely]]
        return {std::bit_cast<double>((bits & 0x800fffffffffffffULL) | 0x3fe0000000000000ULL), be - 1022};
    int e = 0;
    const double m = std::frexp(v, &e);
    return {m, e};
}
inline Scaled operator*(Scaled a, Scaled b) {
    Scaled r = scaled(a.m * b.m);
    r.e += a.e + b.e;
    return r;
}
inline Scaled operator/(Scaled a, Scaled b) {
    Scaled r = scaled(a.m / b.m);
    r.e += a.e - b.e;
    return r;
}
// Rounded to float64 once (ldexp rounds into the subnormals correctly).
inline double value(Scaled s) {
    if (s.m == 0.0 || !std::isfinite(s.m)) return s.m;
    // m in [1/2, 1) and a normal result: m times 2^e built from its bits (exact, as ldexp)
    if (s.e >= -1021 && s.e <= 1024 && std::fabs(s.m) >= 0.5 && std::fabs(s.m) < 1.0) [[likely]]
        return s.m * 2.0 * std::bit_cast<double>(static_cast<std::uint64_t>(s.e - 1 + 1023) << 52);
    const std::int64_t e = std::clamp<std::int64_t>(s.e, -2200, 2200);
    return std::ldexp(s.m, static_cast<int>(e));
}
// log|m 2^e| = log|m| + e log 2
inline double log_abs(Scaled s) {
    const double e = static_cast<double>(s.e);
    return (std::log(std::fabs(s.m)) + e * kLn2Lo) + e * kLn2Hi;
}

// e^x for any finite x: 2^n e^r with r = x - n log 2 (Cody-Waite, log 2 in two parts).
inline Scaled exp_scaled(double x) {
    // An exponent past 2^57 is beyond every result that can come back into range (as in
    // pow_scaled below): e^x is inf or 0 itself, and a product of one with the other is
    // NaN rather than a number made of two clamped exponents (hyp1f1(100.5, 1e16, 1e300)
    // was 0 for inf). Also keeps n's conversion defined.
    if (!(std::fabs(x) < 0x1p57)) return {x > 0.0 ? kInf : (x < 0.0 ? 0.0 : x), 0};
    if (std::fabs(x) < 700.0) return scaled(std::exp(x));  // in range: libm's exp itself
    const double n = std::nearbyint(x / kLn2Hi);
    double r = std::fma(-n, kLn2Hi, x);
    r = std::fma(-n, kLn2Lo, r);
    Scaled s = scaled(std::exp(r));
    s.e += static_cast<std::int64_t>(n);
    return s;
}

inline constexpr double kExpLimit = 0x1p58;
inline constexpr std::int64_t kExpLimitInt = std::int64_t{1} << 58;

// a^t for a > 0 and finite t, without overflow. a = m 2^ea with m in
// [1/sqrt 2, sqrt 2): a^t = m^t 2^(ea t). ea t is split exactly (fma) into
// an integer and a fraction; m^t is computed in k equal parts (k a power of
// two, each part within range), so libm's pow keeps its accuracy on each.
inline Scaled pow_scaled(double a, double t) {
    {  // a result in range: libm's pow itself (under 1 ULP there)
        const double r = std::pow(a, t);
        if (r >= 0x1p-1022 && r <= 0x1p1022) return scaled(r);
    }
    int ea = 0;
    double m = std::frexp(a, &ea);
    if (m < kSqrtHalfG) {
        m *= 2.0;
        --ea;
    }
    const double p = static_cast<double>(ea) * t;
    // An exponent past 2^58 is beyond every result that can come back into range (and beyond
    // the int64 sums of a few of them): the power is 0 or inf itself. |ea t| >= 2 |t log2 m|,
    // so p's sign is the result's.
    if (!(std::fabs(p) < kExpLimit)) return {p > 0.0 ? kInf : 0.0, 0};
    const double perr = std::fma(static_cast<double>(ea), t, -p);
    double ip = std::floor(p);
    double f;
    if (std::fabs(p) < 0x1p52) {
        f = std::exp2(p - ip);  // p - ip is exact
        f += f * (perr * kLn2Hi);
    } else {
        // p is an integer and the product's rounding perr (up to 2^5 here) holds the
        // fraction: its own integer part goes to the exponent (2^perr ~ 1 + perr log 2
        // gave a negative power for perr < -1.44)
        const double pi = std::floor(perr);
        ip += pi;
        f = std::exp2(perr - pi);
    }
    Scaled r = scaled(f);
    r.e += static_cast<std::int64_t>(ip);
    double tk = t;
    int halvings = 0;
    while (std::fabs(tk) * 0.5 > 960.0) {  // at most 1014 times: |t| is a finite double
        tk *= 0.5;
        ++halvings;
    }
    Scaled part = scaled(std::pow(m, tk));
    if (halvings <= 10) {
        for (int i = 0; i < (1 << halvings); ++i) r = r * part;
        return r;
    }
    // m^t = (m^tk)^(2^halvings) by squaring (a product of 2^halvings parts would not return
    // for a huge t).
    for (int i = 0; i < halvings; ++i) {
        part = part * part;
        if (part.e > kExpLimitInt || part.e < -kExpLimitInt) return {part.e > 0 ? kInf : 0.0, 0};
    }
    return r * part;
}

// ---- Lanczos ---------------------------------------------------------------

// L(z) = num(z) / den(z), evaluated in 1/z for z > 1 so neither overflows
// (Boost's evaluate_rational).
inline double lanczos_sum(double z) {
    double s1, s2;
    if (z <= 1.0) {
        s1 = lanczos::kNum[12];
        s2 = lanczos::kDen[12];
        for (int i = 11; i >= 0; --i) {
            s1 = s1 * z + lanczos::kNum[i];
            s2 = s2 * z + lanczos::kDen[i];
        }
    } else {
        const double w = 1.0 / z;
        s1 = lanczos::kNum[0];
        s2 = lanczos::kDen[0];
        for (int i = 1; i <= 12; ++i) {
            s1 = s1 * w + lanczos::kNum[i];
            s2 = s2 * w + lanczos::kDen[i];
        }
    }
    return s1 / s2;
}

// (n - 1)! for n = 1..23: exact in float64 (22! = 2^19 * an odd 51-bit number).
inline constexpr double kFactorial[23] = {
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
    51090942171709440000.0,
    1124000727777607680000.0
};

// Gamma(z) for z > 0 as m 2^e (candidate A, Lanczos):
//     Gamma(z) = L(z) zgh^(z - 1/2) e^-zgh,  zgh = z + g - 1/2
// zgh is kept as hi + lo: rounding it moves the power by (z - 1/2) 2^-53 and
// the exponential by zgh 2^-53 relative; the two cancel down to g 2^-53 (3 ULP).
// Then zgh^t = hi^t (1 + lo/hi)^t ~ hi^t (1 + t lo/hi) and e^-zgh = e^-hi (1 - lo).
// Integers up to 23 come from the table (exact), tiny z from 1/z - euler.
inline double gamma_plain(double z);
inline Scaled gamma_scaled(double z) {
    if (z < 0x1p-30) return Scaled{1.0, 0} / scaled(z) * scaled(1.0 - kEuler * z);
    if (z <= 170.0) return scaled(gamma_plain(z));  // the same form in float64 where it is in range
    const TwoSum zgh = two_sum(z, lanczos::kG - 0.5);  // g - 1/2 is exact
    const double t = z - 0.5;
    Scaled r = scaled(lanczos_sum(z)) * pow_scaled(zgh.s, t);
    r = r * scaled(1.0 + t * (zgh.err / zgh.s));
    r = r * exp_scaled(-zgh.s) * scaled(1.0 - zgh.err);
    return r;
}

// The same Lanczos form in float64 for 0x1p-30 <= z <= 170 (the value in range; the power split
// in two halves from 100 so that it does not overflow): gamma_scaled's value without the m 2^e
// arithmetic (5x faster; against Arb on [1, 170] both 6 ULP at most, Boost's tgamma 9).
inline double gamma_plain(double z) {
    if (z <= 23.0 && z == std::floor(z)) return kFactorial[static_cast<int>(z) - 1];
    const TwoSum zgh = two_sum(z, lanczos::kG - 0.5);
    const double t = z - 0.5;
    const double r = lanczos_sum(z) * ((1.0 + t * (zgh.err / zgh.s)) * (1.0 - zgh.err));
    if (z < 100.0) return r * (std::pow(zgh.s, t) * std::exp(-zgh.s));
    const double hp = std::pow(zgh.s, 0.5 * t);
    return r * hp * std::exp(-zgh.s) * hp;
}

// ---- log Gamma for x > 0 -------------------------------------------------

// lgamma(2 + e) for |e| <= 1/2: the Taylor series about 2 (coeffs/lgamma_taylor2.hpp).
inline double lgamma_taylor2(double e) { return e * horner(e, kLgammaTaylor2); }

// log Gamma(x), x > 0, in two approximations and exact reductions:
//     x < 2^-30         -log x - euler x
//     x < 1/2           lgamma(2 + x) - log1p(x) - log(x)        (lgamma(x) = lgamma(1 + x) - log x)
//     x < 3/2           lgamma(2 + e) - log1p(e),  e = x - 1    (exact: Sterbenz)
//     x <= 5/2          lgamma(2 + e),             e = x - 2
//     x < 13            log(prod) + lgamma(2 + e): down to (3/2, 5/2] by Gamma(u + 1) = u Gamma(u)
//     x >= 13           Stirling (x - 1/2) log x - x + log sqrt(2 pi) + A(1/x^2)/x
// Near 1 and 2, where log Gamma is 0, every term is small and computed from
// the exact e: relative accuracy, not just absolute (scipy: 2.9e5 ULP near 1).
inline double lgamma_pos(double x) {
    if (x < 0x1p-30) return -std::log(x) - kEuler * x;
    if (x < 0.5) return lgamma_taylor2(x) - std::log1p(x) - std::log(x);
    if (x < 1.5) {
        const double e = x - 1.0;
        return lgamma_taylor2(e) - std::log1p(e);
    }
    if (x <= 2.5) return lgamma_taylor2(x - 2.0);
    if (x < 13.0) {
        double z = 1.0, u = x;
        while (u > 2.5) {
            u -= 1.0;
            z *= u;
        }
        return std::log(z) + lgamma_taylor2(u - 2.0);
    }
    if (x > 2.556348e305) return kInf;  // Cephes MAXLGM: (x - 1/2) log x overflows
    const double q = (x - 0.5) * std::log(x) - x + kLogSqrt2Pi;
    if (x > 1e8) return q;
    const double w = 1.0 / x;
    return q + horner(w * w, cephes::kLgamA) * w;
}

// ---- digamma -------------------------------------------------------------

// psi(x) for x > 0 (Cephes psi.c with Boost's rational on [1, 2]):
// recurrence into [1, 2], where psi(x) = (x - root)(Y + P/Q) has the
// positive root factored out (relative accuracy at x0 = 1.4616...), or the
// asymptotic series log x - 1/(2x) - A(1/x^2) from 10.
inline double digamma_pos(double x) {
    double y = 0.0;
    if (x <= 10.0 && x == std::floor(x)) {  // psi(n) = H(n-1) - euler
        for (int i = 1; i < static_cast<int>(x); ++i) y += 1.0 / i;
        return y - kEuler;
    }
    if (x < 1.0) {
        y = -1.0 / x;
        x += 1.0;
    } else if (x < 10.0) {
        while (x > 2.0) {
            x -= 1.0;
            y += 1.0 / x;
        }
    }
    if (x >= 1.0 && x <= 2.0) {
        double g = x - cephes::kPsiRoot1;
        g -= cephes::kPsiRoot2;
        g -= cephes::kPsiRoot3;
        const double r = horner(x - 1.0, cephes::kPsiP) / horner(x - 1.0, cephes::kPsiQ);
        return y + (g * cephes::kPsiY + g * r);
    }
    const double w = x < 1e17 ? 1.0 / (x * x) : 0.0;
    return y + (std::log(x) - 0.5 / x - w * horner(w, cephes::kPsiA));
}

// ---- Gamma and 1/Gamma at a double-double argument -------------------------

// Gamma(hi + lo), hi > 0, to first order in lo (|lo| <= ulp(hi)).
inline Scaled gamma_dd(double hi, double lo) {
    Scaled g = gamma_scaled(hi);
    if (lo != 0.0) g = g * scaled(1.0 + digamma_pos(hi) * lo);
    return g;
}

}  // namespace sd::special
