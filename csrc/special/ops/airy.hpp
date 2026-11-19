// The Airy functions (docs/api/special.md §9, docs/design/special.md §5.9; S6):
// airy(x) -> (Ai, Ai', Bi, Bi'), all four from one evaluation.
//
//     -2 <= x <= 1     Maclaurin: Ai = c1 f - c2 g, Bi = sqrt3 (c1 f + c2 g)
//     x > 1            Bessel functions of order 1/3, 2/3 at zeta = (2/3) x^(3/2)
//                      (DLMF 9.6.2-9.6.5, written with K and I only):
//                        Ai  = sqrt(x/3) K_1/3 / pi,       Ai' = -x K_2/3 / (pi sqrt3)
//                        Bi  = sqrt(x/3) (2 I_1/3 + sqrt3 K_1/3 / pi)
//                        Bi' = x/sqrt3 (2 I_2/3 + sqrt3 K_2/3 / pi)
//                      every term positive; I and K scaled (bessel.hpp), e^+-zeta
//                      as m 2^e: Ai's subnormal tail and Bi's overflow come out
//     -9.7 < x < -2    J and Y of order 1/3, 2/3 (DLMF 9.6.6-9.6.9)
//     x <= -9.7        the oscillatory expansions (DLMF 9.7.9-9.7.12) about
//                      zeta - pi/4, zeta >= 20 (smallest term ~e^-2zeta)
//
// zeta is a double-double: rounding it would move the phase by zeta 2^-53
// (scipy loses 5e5 of the envelope at x = -1e4). The Bessel functions take its
// high part and a first-order correction (their derivatives); the
// oscillatory phase is sin/cos(zeta_hi) from libm (exact reduction) combined
// with zeta_lo - pi/4. That is exact to ~zeta 2^-106 rad: for |x| beyond
// ~1e10 (zeta past 2^50) the phase itself is no longer resolved (S6 note).
#pragma once

#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"
#include "special/ops/bessel.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace airy_detail {

inline constexpr double kC1 = 0.35502805388781723926;   // Ai(0)
inline constexpr double kC2 = 0.25881940379280679840;   // -Ai'(0)
inline constexpr double kSqrt3 = 1.73205080756887729353;
inline constexpr double kInvSqrtPi = 0.56418958354775628695;

struct Airy {
    double ai, aip, bi, bip;
};

// Maclaurin series f, g and their derivatives.
inline Airy maclaurin(double x) {
    if (x == 0.0) return {kC1, -kC2, kSqrt3 * kC1, kSqrt3 * kC2};
    const double x3 = x * x * x;
    double f = 1.0, g = x, fp = 0.0, gp = 1.0;
    double tf = 1.0, tg = x;
    for (int k = 1; k < 200; ++k) {
        tf *= x3 / ((3.0 * k) * (3.0 * k - 1.0));  // x^3k-term of f
        tg *= x3 / ((3.0 * k + 1.0) * (3.0 * k));  // x^(3k+1)-term of g
        f += tf;
        g += tg;
        // derivatives term by term: d/dx x^n = n x^(n-1)
        fp += tf * (3.0 * k) / x;
        gp += tg * (3.0 * k + 1.0) / x;
        if (std::fabs(tf) + std::fabs(tg) <= 0x1p-56 * (std::fabs(f) + std::fabs(g)) &&
            std::fabs(tf * 3.0 * k / x) + std::fabs(tg * (3.0 * k + 1.0) / x) <=
                0x1p-56 * (std::fabs(fp) + std::fabs(gp)))
            break;
    }
    return {kC1 * f - kC2 * g, kC1 * fp - kC2 * gp, kSqrt3 * (kC1 * f + kC2 * g), kSqrt3 * (kC1 * fp + kC2 * gp)};
}

// zeta = (2/3) z^(3/2) for z > 0 as a double-double.
inline dd::DD zeta_of(double z) {
    const double s = std::sqrt(z);
    const dd::DD sq{s, std::fma(-s, s, z) / (2.0 * s)};  // sqrt(z), one Newton step
    return sq * z * dd::kTwoThirds;
}

// x > 1: the K, I form.
inline Airy positive(double x) {
    using namespace bessel_detail;
    const dd::DD zt = zeta_of(x);
    const double zh = zt.hi, zl = zt.lo;
    const IK a = ik(1.0 / 3.0, zh), b = ik(2.0 / 3.0, zh);
    // scaled values at zeta = hi + lo: f(hi + lo) = f(hi) + lo f'(hi); for the
    // scaled e^zeta K: (e^z K)' = e^z K' + e^z K, for e^-zeta I: e^-z I' - e^-z I
    const double k13 = value(a.k) + zl * (value(a.kp) + value(a.k));
    const double k23 = value(b.k) + zl * (value(b.kp) + value(b.k));
    const double i13 = value(a.i) + zl * (value(a.ip) - value(a.i));
    const double i23 = value(b.i) + zl * (value(b.ip) - value(b.i));
    const Scaled em = exp_scaled(-zh) * scaled(1.0 - zl), ep = exp_scaled(zh) * scaled(1.0 + zl);
    const double r = std::sqrt(x / 3.0);
    return {
        value(em * scaled(r * k13 / kPi)), value(em * scaled(-x * k23 / (kPi * kSqrt3))),
        value(ep * scaled(r * 2.0 * i13)) + value(em * scaled(r * kSqrt3 * k13 / kPi)),
        value(ep * scaled(x / kSqrt3 * 2.0 * i23)) + value(em * scaled(x * k23 / kPi))
    };
}

// -9.7 < x < -2: the J, Y form at zeta for z = -x.
inline Airy negative_bessel(double z) {
    using namespace bessel_detail;
    const dd::DD zt = zeta_of(z);
    const double zh = zt.hi, zl = zt.lo;
    const JY a = jy(1.0 / 3.0, zh), b = jy(2.0 / 3.0, zh);
    const double j13 = value(a.j) + zl * value(a.jp), y13 = value(a.y) + zl * value(a.yp);
    const double j23 = value(b.j) + zl * value(b.jp), y23 = value(b.y) + zl * value(b.yp);
    const double sz = std::sqrt(z);
    return {
        0.5 * sz * (j13 - y13 / kSqrt3), 0.5 * z * (j23 + y23 / kSqrt3), -0.5 * sz * (j13 / kSqrt3 + y13),
        0.5 * z * (j23 / kSqrt3 - y23)
    };
}

// x <= -9.7: the oscillatory expansions in 1/zeta with u_k, v_k (DLMF 9.7.2)
inline Airy negative_asymptotic(double z) {
    const dd::DD zt = zeta_of(z);
    const double zh = zt.hi, zl = zt.lo;
    // sums over u_k, v_k: P = sum (-1)^k u_2k / zeta^2k, Q = sum (-1)^k u_(2k+1) / zeta^(2k+1)
    double Pu = 1.0, Qu = 0.0, Pv = 1.0, Qv = 0.0;
    double u = 1.0, zk = 1.0, prev = kInf;
    for (int k = 1; k < 200; ++k) {
        u *= (6.0 * k - 5.0) * (6.0 * k - 3.0) * (6.0 * k - 1.0) / ((2.0 * k - 1.0) * 216.0 * k);
        const double v = -(6.0 * k + 1.0) / (6.0 * k - 1.0) * u;
        zk /= zh;
        const double tu = u * zk, tv = v * zk;
        if (std::fabs(tu) > prev) break;  // the asymptotic tail turns up (does not happen for zeta >= 20 before 2^-56)
        prev = std::fabs(tu);
        const double sgn = (k / 2) % 2 == 0 ? 1.0 : -1.0;  // (-1)^floor(k/2)
        if (k % 2 == 0) {
            Pu += sgn * tu;
            Pv += sgn * tv;
        } else {
            Qu += sgn * tu;
            Qv += sgn * tv;
        }
        if (std::fabs(tu) + std::fabs(tv) <= 0x1p-56) break;
    }
    // theta = zeta - pi/4 = zh + (zl - pi/4): sin, cos of zh from libm, of the rest directly
    const double sh = std::sin(zh), ch = std::cos(zh);
    const double r = zl - 0.25 * kPi;
    const double sr = std::sin(r), cr = std::cos(r);
    const double st = sh * cr + ch * sr, ct = ch * cr - sh * sr;
    const double q4 = std::sqrt(std::sqrt(z));  // z^(1/4)
    const double a = kInvSqrtPi / q4, ad = kInvSqrtPi * q4;
    return {a * (ct * Pu + st * Qu), ad * (st * Pv - ct * Qv), a * (-st * Pu + ct * Qu), ad * (ct * Pv + st * Qv)};
}

inline Airy airy(double x) {
    if (std::isnan(x)) return {kNaN, kNaN, kNaN, kNaN};
    if (x == kInf) return {0.0, -0.0, kInf, kInf};
    if (x == -kInf) return {0.0, kNaN, 0.0, kNaN};  // the derivatives' amplitude grows: no limit
    if (x >= -2.0 && x <= 1.0) return maclaurin(x);
    // zeta = (2/3) x^(3/2) > 1885: Ai, Ai' are far below the doubles and Bi, Bi' far above
    // (and zeta itself overflows from x = 1e206: the Bessel functions did not return)
    if (x >= 200.0) return {0.0, -0.0, kInf, kInf};
    if (x > 1.0) return positive(x);
    if (x > -9.7) return negative_bessel(-x);
    // Below -1e20 the phase zeta = (2/3) |x|^(3/2), a double-double, is off by more than 1e-3
    // rad (zeta 2^-106) and by whole turns from 2.4e21: no value (the envelope alone is known).
    if (x < -1e20) return {kNaN, kNaN, kNaN, kNaN};
    return negative_asymptotic(-x);
}

}  // namespace airy_detail

// airy(x) -> (Ai, Ai', Bi, Bi') (the Python wrapper makes it an AiryResult).
struct Airy : Real {
    static constexpr const char* name = "airy";
    static constexpr int nin = 1, nout = 4;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kAiry;
    static std::array<double, 4> f(double x) {
        const airy_detail::Airy a = airy_detail::airy(x);
        return {a.ai, a.aip, a.bi, a.bip};
    }
};

}  // namespace sd::special
