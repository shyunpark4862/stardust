// Bessel functions of real order (docs/api/special.md §8.2,
// docs/design/special.md §5.8; S6): jv yv iv kv and the scaled jve yve ive kve.
//
// J and Y of an order |nu| < 50 take the plain-double path first (fast_j, fast_y below: the
// power series, Hankel, Miller's recurrence with Neumann's sum, the closed forms at integer and
// half-integer orders, Steed's CF2 for Y); what it does not give as a normal double, and
// every larger order, goes the general way that follows.
//
// Temme's method (Numerical Recipes' bessjy and bessik), for nu >= 0, x > 0:
//   - nu = n + mu with |mu| <= 1/2.
//   - CF1 gives J'_nu / J_nu (I'_nu / I_nu); the downward recurrence carries
//     the unnormalised pair to mu.
//   - x < 2: Temme's series for Y_mu, Y_mu+1 (K_mu, K_mu+1). Gamma1 and
//     Gamma2 come from the Taylor series of 1/Gamma(1+z)
//     (coeffs/rgamma_taylor.hpp) rather than NR's fitted Chebyshev series.
//   - x >= 2: Steed's CF2 for the same.
//   - The Wronskian normalises J (I); Y (K) recur upward to nu.
//   - x >= 30: Hankel's asymptotic series where hankel_pq accepts it (the terms
//     reach 2^-56 of the sum before they grow again and none exceeds 2; CF1
//     needs ~x terms there). J, Y with nu < x where it is refused at nu:
//     Hankel at nu - floor(nu), then the recurrence upward (jy_upward).
// I, K with |nu| >= 30: Debye's uniform expansion (debye_value). J, Y have no
// uniform expansion: a large nu is right but costs O(nu).
// iv, kv (and ive, kve) with |nu| < 30 and x < 700 take a float64 path first (fast_iv_kv: I's
// power series for x^2 < 4 (nu + 1), else K by Temme/Steed and its upward recurrence, I by the
// Wronskian; faster than Boost and within 16 ULP on Arb sets where Boost reached 40); then Boost
// (iv |nu| < 30, kv |nu| < 400; iv_route, kv_route), then the general path.
//
// Changed from NR:
//   - CF1's value is the backward pass over the terms Lentz counted (cf.hpp):
//     the forward ratios lose ~sqrt(terms) ULP.
//   - The recurrences keep the binary exponent apart (m 2^e) so a large nu
//     neither overflows the unnormalised J nor loses the result to
//     underflow.
//   - I and K are computed scaled (e^-x I, e^x K) from the start; ive, kve
//     are those, iv, kv multiply back with exp as m 2^e.
//   - Hankel's phase chi = x - (nu/2 + 1/4) pi is never rounded: cos and sin
//     of x from libm (exact reduction) and of pi nu/2 from cospi/sinpi.
// nu < 0 by reflection with sinpi, cospi (exact zeros at the integers and
// half-integers: yv(-1/2, 0) = 0, scipy -inf).
#pragma once

#include "base/sincos.hpp"
#include "special/boost.hpp"

#include "special/cf.hpp"
#include "special/coeffs/debye_u.hpp"
#include "special/coeffs/rgamma_taylor.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"
#include "special/ops/bessel01.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace sd::special {

namespace bessel_detail {

inline constexpr double kTwoOverPi = 0.63661977236758134308;
inline constexpr double kTiny = 0x1p-1000;

// The loops below are bounded (docs/design/stats.md 1.3): a recurrence over the order takes
// at most kMaxOrderSteps steps and a continued fraction at most kMaxCfTerms terms (about
// 0.25 s each: yv(2e7, 1e7), 1e7 steps, takes 0.23 s; jv there is 0 at once, j_underflows);
// a call that would need more is NaN.
// J and Y have no uniform expansion here, so that is x > |nu| > 1e7 below the Hankel
// expansion's reach (x < (nu + 1)^2 / 4, jy_upward), and |nu| - x > 1e7 where J is not
// proven to underflow (j_underflows; order_saturates gives J = 0 and Y = -inf at once).
inline constexpr double kMaxOrderSteps = 1e7;
inline constexpr int kMaxCfTerms = 10000000;
inline constexpr int kMaxCf2Terms = 100000;

// Temme's gamma functions of mu, |mu| <= 1/2.
struct Gammas {
    double g1, g2, gpl, gmi;  // Gamma1, Gamma2, 1/Gamma(1+mu), 1/Gamma(1-mu)
};
inline Gammas temme_gammas(double mu) {
    const double m2 = mu * mu;
    const double g1 = -horner(m2, kRgammaOdd), g2 = horner(m2, kRgammaEven);
    return {g1, g2, g2 - mu * g1, g2 + mu * g1};
}

// A value as m 2^e, multiplied and rescaled freely (gamma_core.hpp's Scaled).
inline Scaled sc(double v) { return scaled(v); }

// The recurrences run in double-double: over n steps float64 loses ~n/2 ULP
// (86 ULP at nu = 286, x = 100, measured), from the products and from the
// rounded coefficients v/x alike. The orders v = nu - k are exact (k an
// integer below nu), and 1/x is a double-double.
inline dd::DD recip(double x) {
    const double h = 1.0 / x;
    return {h, std::fma(-h, x, 1.0) / x};
}
inline void rescale_down(dd::DD& a, dd::DD& b, std::int64_t& ex) {
    if (std::fabs(a.hi) > 0x1p500) {
        a = dd::ldexp(a, -500);
        b = dd::ldexp(b, -500);
        ex += 500;
    }
}

// ---- J and Y ------------------------------------------------------------------

struct JY {
    Scaled j, y, jp, yp;  // J_nu, Y_nu and their derivatives
};
inline constexpr JY kNoJY{{kNaN, 0}, {kNaN, 0}, {kNaN, 0}, {kNaN, 0}};

// Hankel: P, Q in 1/x. The terms a_k / x^k first grow while 4 nu^2 > 8 k x, then
// fall (the factors 4nu^2 - (2k-1)^2 shrink), and grow again for k past ~2x.
// Accepted when they reach 2^-56 of the sum before growing again and never
// exceed 2 (the leading term is 1).
inline bool hankel_pq(double nu, double x, double& P, double& Q) {
    const double m = 4.0 * nu * nu;
    double term = 1.0;  // a_k / x^k
    P = 1.0;
    Q = 0.0;
    double big = 1.0, prev = 1.0;
    bool falling = false;
    for (int k = 1; k < 400; ++k) {
        term *= (m - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (8.0 * k * x);
        const double a = std::fabs(term);
        if (a < prev) falling = true;
        else if (falling && a > prev) return false;  // the asymptotic tail turns up
        prev = a;
        big = std::max(big, a);
        // P = 1 - a2/x^2 + a4/x^4 ..., Q = a1/x - a3/x^3 + ...
        const int r = k % 4;
        if (k % 2 == 0) P += (r == 0 ? term : -term);
        else Q += (r == 1 ? term : -term);
        // the leading term is 1 and P^2 + Q^2 ~ 1: a term past 2 means cancellation, and each
        // term carries ~k ulp of its product's rounding (measured: 400 ULP at a bound of 16)
        if (!(big <= 2.0)) return false;
        if (a <= 0x1p-56 * std::max(std::fabs(P), std::fabs(Q))) return true;
    }
    return false;
}

// J, Y by Hankel when it converges; the derivatives from order nu + 1.
inline bool jy_hankel(double nu, double x, JY& out) {
    double P0, Q0, P1, Q1;
    if (!hankel_pq(nu, x, P0, Q0) || !hankel_pq(nu + 1.0, x, P1, Q1)) return false;
    const auto [s, c] = sincos(x);
    const double amp = std::sqrt(kTwoOverPi / x);
    auto jy = [&](double v, double P, double Q, double& J, double& Y) {
        // chi = x - pi v/2 - pi/4: cos chi = (c (cv + sv) + s (sv... written via phi = pi v/2:
        const double cv = cospi(0.5 * v), sv = sinpi(0.5 * v);
        // cos(x - phi - pi/4) = [cos(x - phi) + sin(x - phi)] / sqrt2, cos(x - phi) = c cv + s sv
        const double cx = c * cv + s * sv, sx = s * cv - c * sv;  // cos, sin of x - phi
        const double cchi = (cx + sx) * kSqrtHalfG, schi = (sx - cx) * kSqrtHalfG;
        J = amp * (P * cchi - Q * schi);
        Y = amp * (P * schi + Q * cchi);
    };
    double J0, Y0, J1, Y1;
    jy(nu, P0, Q0, J0, Y0);
    jy(nu + 1.0, P1, Q1, J1, Y1);
    out = {sc(J0), sc(Y0), sc(nu / x * J0 - J1), sc(nu / x * Y0 - Y1)};
    return true;
}

// Temme's series for Y_mu and Y_(mu+1), |mu| <= 1/2, 0 < x < 2 (NR bessjy).
inline bool temme_y(double mu, double x, double& ymu, double& y1) {
    const double mu2 = mu * mu, xi2 = 2.0 * (1.0 / x);
    const double x2 = 0.5 * x, pimu = kPi * mu;
    const double fact = std::fabs(mu) < 1e-300 ? 1.0 : pimu / sinpi(mu);
    const double d = -std::log(x2);
    double e = mu * d;
    const double fact2 = std::fabs(e) < 1e-300 ? 1.0 : std::sinh(e) / e;
    const Gammas g = temme_gammas(mu);
    double ff = kTwoOverPi * fact * (g.g1 * std::cosh(e) + g.g2 * fact2 * d);
    e = std::exp(e);
    double p = e / (g.gpl * kPi), q = 1.0 / (e * kPi * g.gmi);
    const double fact3 = std::fabs(mu) < 1e-300 ? 1.0 : sinpi(0.5 * mu) / (0.5 * pimu);
    const double r = kPi * (0.5 * pimu) * fact3 * fact3;
    double c = 1.0;
    const double dd = -x2 * x2;
    double sum = ff + r * q, sum1 = p;
    for (int i = 1; i < 1000; ++i) {  // x < 2: c falls like 1/i!
        ff = (i * ff + p + q) / (i * static_cast<double>(i) - mu2);
        c *= dd / i;
        p /= (i - mu);
        q /= (i + mu);
        const double del = c * (ff + r * q);
        sum += del;
        const double del1 = c * p - i * del;
        sum1 += del1;
        if (std::fabs(del) <= (1.0 + std::fabs(sum)) * 0x1p-56) {
            ymu = -sum;
            y1 = -sum1 * xi2;
            return true;
        }
    }
    return false;
}

// Steed's CF2 for x >= 2 (NR bessjy): p + i q = (J'_mu + i Y'_mu) / (J_mu + i Y_mu).
inline bool steed_cf2(double mu, double x, double& p_out, double& q_out) {
    const double xi = 1.0 / x;
    double a = 0.25 - mu * mu, p = -0.5 * xi, q = 1.0;
    const double br = 2.0 * x;
    double bi = 2.0;
    double fct = a * xi / (p * p + q * q);
    double cr = br + q * fct, ci = bi + p * fct;
    double den = br * br + bi * bi;
    double dr = br / den, di = -bi / den;
    double dlr = cr * dr - ci * di, dli = cr * di + ci * dr;
    double temp = p * dlr - q * dli;
    q = p * dli + q * dlr;
    p = temp;
    for (int i = 1; i < kMaxCf2Terms; ++i) {  // x >= 2: a few hundred terms at most
        a += 2 * i;
        bi += 2.0;
        dr = a * dr + br;
        di = a * di + bi;
        if (std::fabs(dr) + std::fabs(di) < kTiny) dr = kTiny;
        fct = a / (cr * cr + ci * ci);
        cr = br + cr * fct;
        ci = bi - ci * fct;
        if (std::fabs(cr) + std::fabs(ci) < kTiny) cr = kTiny;
        den = dr * dr + di * di;
        dr /= den;
        di /= -den;
        dlr = cr * dr - ci * di;
        dli = cr * di + ci * dr;
        temp = p * dlr - q * dli;
        q = p * dli + q * dlr;
        p = temp;
        if (std::fabs(dlr - 1.0) + std::fabs(dli) <= kEps) {
            p_out = p;
            q_out = q;
            return true;
        }
    }
    return false;
}

// NR bessjy for nu >= 0, 0 < x < inf.
inline JY jy_temme(double nu, double x) {
    if (!((x < 2.0 ? nu : nu - x) <= kMaxOrderSteps)) return kNoJY;  // the recurrences' length
    // CF1 known not to settle within kMaxCfTerms (jv(1e300, 1e300) walked all of them, 45 ms,
    // for NaN): from x = 2^53 its step 2/x is below half an ulp of b = 2 nu / x >= 2 (nu >= x
    // here), so b stays fl(2 nu / x) = 2 + delta, and the fraction of a constant b settles
    // after about 18 / sqrt(delta) terms (its ratio is 1 - sqrt(delta)): more than kMaxCfTerms
    // for delta = 2 (nu - x) / x below 2e-12.
    if (x >= 0x1p53 && nu - x < 1e-12 * x) return kNoJY;
    const std::int64_t nl = x < 2.0 ? static_cast<std::int64_t>(nu + 0.5)
                                    : std::max<std::int64_t>(0, static_cast<std::int64_t>(nu - x + 1.5));
    const double mu = nu - nl;
    const double xi = 1.0 / x, xi2 = 2.0 * xi, w = xi2 / kPi;
    // CF1: J'_nu / J_nu = nu/x - 1/(2(nu+1)/x - 1/(2(nu+2)/x - ...)). The forward pass
    // (Lentz) only counts the terms.
    int terms = 0;
    {
        double b = xi2 * nu, d = 0.0, c = std::max(nu * xi, kTiny);
        bool converged = false;
        for (int i = 1; i <= kMaxCfTerms; ++i) {
            b += xi2;
            d = b - d;
            if (std::fabs(d) < kTiny) d = kTiny;
            c = b - 1.0 / c;
            if (std::fabs(c) < kTiny) c = kTiny;
            d = 1.0 / d;
            const double del = c * d;
            terms = i;
            if (std::fabs(del - 1.0) <= kEps) {
                converged = true;
                break;
            }
        }
        if (!converged) return kNoJY;  // (~x terms for x > nu: x past kMaxCfTerms)
    }
    // The value from the backward pass: den_k = 2(nu+k)/x - 1/den_(k+1) is J_(nu+k-1) / J_(nu+k).
    // J at a large order is positive, so the negative den_k give the signs of J_(nu+1) and J_nu;
    // the forward pass can place x on the other side of a zero of J_nu than this pass does,
    // and a sign from there would flip J'_nu (and Y_nu with it).
    // The pair (J_nu, J'_nu) is kept unnormalised and never as the ratio J'/J: at the float64
    // nearest a zero of J_nu den_1 can be exactly 0, the ratio infinite and every product
    // with it NaN. J'_nu = (nu/x) J_nu - J_(nu+1).
    // While nu + k >= x + 2 every den_k is above 1 (den_k > 2 - 1/den_(k+1), from den = inf),
    // so only the steps below that are counted; counted without a branch, as the signs
    // alternate unpredictably for x > nu.
    double den = kInf;
    int negatives = 0, k = terms + 4;
    const double first_counted = std::max(1.0, x - nu + 2.0);
    const int kcount = x < 0x1p40 && first_counted < k ? static_cast<int>(first_counted) : k;
    for (; k > kcount; --k) den = 2.0 * (nu + k) * xi - 1.0 / den;
    for (; k >= 2; --k) {
        den = 2.0 * (nu + k) * xi - 1.0 / den;
        negatives += den < 0.0;
    }
    const double s1 = negatives % 2 ? -1.0 : 1.0;  // the sign of J_(nu+1)
    den = 2.0 * (nu + 1.0) * xi - 1.0 / den;  // J_nu / J_(nu+1)
    double rjl1, rjp1;                        // J_nu, J'_nu up to a common positive factor
    if (std::fabs(den) <= 0x1p500) {          // in units of |J_(nu+1)|
        rjl1 = s1 * den;
        rjp1 = nu * xi * rjl1 - s1;
    } else {  // J_(nu+1) is 0 or nearly: in units of |J_nu|
        rjl1 = den < 0.0 ? -s1 : s1;
        rjp1 = (nu * xi - 1.0 / den) * rjl1;
    }
    // downward recurrence from nu to mu, with the exponent kept apart
    const dd::DD xid = recip(x);
    dd::DD rjl{rjl1, 0.0}, rjpl{rjp1, 0.0};
    std::int64_t ex = 0;  // true = stored * 2^ex
    for (std::int64_t l = nl - 1; l >= 0; --l) {
        const double v1 = nu - static_cast<double>(nl - l - 1), v0 = nu - static_cast<double>(nl - l);  // mu+l+1, mu+l
        const dd::DD rjtemp = xid * v1 * rjl + rjpl;
        rjpl = xid * v0 * rjtemp - rjl;
        rjl = rjtemp;
        rescale_down(rjl, rjpl, ex);
    }
    // (ja, jb) = (J_mu, J'_mu) / c with the larger of the two about 1 in magnitude; c > 0 comes from
    // the Wronskian J Y' - J' Y = 2/(pi x) below, again without dividing by J_mu (mu = -1/2
    // has a zero at pi/2).
    const double jscale = 1.0 / std::max(std::fabs(rjl.hi), std::fabs(rjpl.hi));
    const double ja = rjl.hi * jscale, jb = rjpl.hi * jscale;
    double cj, rymu, rymup, ry1;
    if (x < 2.0) {
        if (!temme_y(mu, x, rymu, ry1)) return kNoJY;
        rymup = mu * xi * rymu - ry1;
        cj = w / (ja * rymup - jb * rymu);
    } else {
        double p, q;
        if (!steed_cf2(mu, x, p, q)) return kNoJY;
        // J' = p J - q Y and Y' = q J + p Y: Y_mu = c yr with yr = (p ja - jb) / q, and the
        // Wronskian is q (J^2 + Y^2)
        const double yr = (p * ja - jb) / q;
        cj = std::sqrt(w / (q * (ja * ja + yr * yr)));
        rymu = cj * yr;
        rymup = q * (cj * ja) + p * rymu;
        ry1 = mu * xi * rymu - rymup;
    }
    // J_nu = rjl1 c jscale / 2^ex
    const Scaled fact = sc(cj) * sc(jscale) * Scaled{0.5, 1 - ex};
    const Scaled rj = sc(rjl1) * fact, rjp = sc(rjp1) * fact;
    // Y upward from mu to nu, exponent apart
    std::int64_t ey = 0;
    dd::DD y0{rymu, 0.0}, y1{ry1, 0.0};
    const dd::DD xi2d = xid * 2.0;
    for (std::int64_t i = 1; i <= nl; ++i) {
        const double v = nu - static_cast<double>(nl - i);  // mu + i
        const dd::DD y2 = xi2d * v * y1 - y0;
        y0 = y1;
        y1 = y2;
        rescale_down(y1, y0, ey);
    }
    const Scaled two_ey{0.5, 1 + ey};
    const Scaled ry = sc(y0.hi) * two_ey, ryp = sc(nu * xi * y0.hi - y1.hi) * two_ey;
    return {rj, ry, rjp, ryp};
}

// ---- I and K, scaled ---------------------------------------------------------

struct IK {
    Scaled i, k, ip, kp;  // e^-x I_nu, e^x K_nu, e^-x I'_nu, e^x K'_nu
};
inline constexpr IK kNoIK{{kNaN, 0}, {kNaN, 0}, {kNaN, 0}, {kNaN, 0}};

// The large-x series: e^-x I ~ sum (-1)^k a_k / x^k / sqrt(2 pi x), e^x K ~ sqrt(pi/(2x)) sum a_k / x^k,
// accepted as hankel_pq accepts.
inline bool ik_asymptotic(double nu, double x, double& si, double& sk) {
    const double m = 4.0 * nu * nu;
    double term = 1.0, prev = 1.0, big = 1.0;
    bool falling = false;
    si = sk = 1.0;
    for (int k = 1; k < 400; ++k) {
        term *= (m - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (8.0 * k * x);
        const double a = std::fabs(term);
        if (a < prev) falling = true;
        else if (falling && a > prev) return false;
        prev = a;
        big = std::max(big, a);
        sk += term;
        si += (k % 2 ? -term : term);
        if (!(big <= 2.0)) return false;  // the leading term is 1 (as hankel_pq)
        if (a <= 0x1p-56 * std::min(std::fabs(si), std::fabs(sk))) return true;
    }
    return false;
}

inline bool ik_hankel(double nu, double x, IK& out) {
    double si0, sk0, si1, sk1;
    if (!ik_asymptotic(nu, x, si0, sk0) || !ik_asymptotic(nu + 1.0, x, si1, sk1)) return false;
    // (the roots apart: 2 x overflows for the largest x)
    const double sx = std::sqrt(x), ai = 0.39894228040143267794 / sx, ak = 1.25331413731550025121 / sx;
    const double I0 = ai * si0, I1 = ai * si1, K0 = ak * sk0, K1 = ak * sk1;
    // I' = I_(nu+1) + (nu/x) I, K' = (nu/x) K - K_(nu+1)
    out = {sc(I0), sc(K0), sc(I1 + nu / x * I0), sc(nu / x * K0 - K1)};
    return true;
}

// The large-x series of K alone, sum_k a_k(nu) / x^k (e^x K = sqrt(pi / (2 x)) times it), accepted
// as ik_asymptotic accepts.
inline constexpr double kKAsymFrom = 20.0;
// 1 / k for k < kRecips (the loops below take their small integers' reciprocals from it: a
// division a term was their throughput's bound, 4 cycles each)
inline constexpr int kRecips = 512;
inline constexpr auto kRecip = [] {
    std::array<double, kRecips> r{};
    for (int k = 1; k < kRecips; ++k) r[k] = 1.0 / k;
    return r;
}();
inline bool k_asymptotic(double nu, double x, double& sk) {
    const double m = 4.0 * nu * nu, inv8x = 0.125 / x;
    double term = 1.0, prev = 1.0;
    bool falling = false;
    sk = 1.0;
    for (int k = 1; k < kRecips; ++k) {
        const double odd = 2.0 * k - 1.0;
        term *= (m - odd * odd) * inv8x * kRecip[k];
        const double a = std::fabs(term);
        if (a < prev) falling = true;
        else if (falling && a > prev) return false;
        prev = a;
        if (!(a <= 2.0)) return false;
        sk += term;
        if (a <= 0x1p-56 * std::fabs(sk)) return true;
    }
    return false;
}

// e^x K_mu and e^x K_(mu+1), |mu| <= 1/2, x > 0 (NR bessik): Temme's series for x < 2, the
// large-x series from kKAsymFrom, Steed's CF2 (Temme's form) between. False if a loop did not end.
inline bool k_pair_scaled(double mu, double x, double& rkmu, double& rk1) {
    const double mu2 = mu * mu, xi = 1.0 / x, xi2 = 2.0 * xi;
    if (x < 2.0) {
        const double x2 = 0.5 * x, pimu = kPi * mu;
        const double fact = std::fabs(mu) < 1e-300 ? 1.0 : pimu / sinpi(mu);
        const double d = -std::log(x2);
        double e = mu * d;
        const double fact2 = std::fabs(e) < 1e-300 ? 1.0 : std::sinh(e) / e;
        const Gammas g = temme_gammas(mu);
        double ff = fact * (g.g1 * std::cosh(e) + g.g2 * fact2 * d);
        double sum = ff;
        e = std::exp(e);
        double p = 0.5 * e / g.gpl, q = 0.5 / (e * g.gmi), c = 1.0;
        const double dd = x2 * x2;
        double sum1 = p;
        bool converged = false;
        for (int i = 1; i < 1000; ++i) {  // x < 2: c falls like 1/i!
            ff = (i * ff + p + q) / (i * static_cast<double>(i) - mu2);
            c *= dd / i;
            p /= (i - mu);
            q /= (i + mu);
            const double del = c * ff;
            sum += del;
            sum1 += c * (p - i * ff);
            if (std::fabs(del) <= std::fabs(sum) * 0x1p-56) {
                converged = true;
                break;
            }
        }
        if (!converged) return false;
        const double ex2 = std::exp(x);
        rkmu = sum * ex2;
        rk1 = sum1 * xi2 * ex2;
    } else if (x >= kKAsymFrom && k_asymptotic(mu, x, rkmu) && k_asymptotic(mu + 1.0, x, rk1)) {
        // the large-x series (its smallest term ~ e^-2x: below 2^-57 from x = 20)
        const double r = std::sqrt(kPi / (2.0 * x));
        rkmu *= r, rk1 *= r;
    } else {
        // Steed's CF2 (Temme's form) for K_mu: no e^-x, so e^x K_mu. Its d_i = 1 / (b_i + a_i d_(i-1))
        // put a division on the loop's chain (18 cycles a term); as d_i = D_(i-1) / D_i with
        // D_i = b_i D_(i-1) + a_i D_(i-2), delh_i = (b_i d_i - 1) delh_(i-1) = P_i / (D_i D_(i-1)),
        // P_i = -a_i P_(i-1): the chains are an fma and products, the divisions off them; and one
        // division a term for both 1 / a_i and 1 / (D_i D_(i-1)), 1 / (i + 1) from kRecip (the
        // divisions' throughput was the loop's bound). D and P kept within 2^+-400 together, P by
        // the square of D's factor.
        double b = 2.0 * (1.0 + x), D1 = 1.0, D0 = b, P = 1.0, delh = 1.0 / b, hh = delh, q1 = 0.0, q2 = 1.0;
        const double a1 = 0.25 - mu2;
        double q = a1, c = a1, a = -a1, s = 1.0 + q * delh;
        bool converged = false;
        for (int i = 1; i < kMaxCf2Terms; ++i) {  // x >= 2: a few hundred terms at most
            a -= 2 * i;
            c *= -a * (i + 1 < kRecips ? kRecip[i + 1] : 1.0 / (i + 1.0));
            b += 2.0;
            const double Dn = std::fma(b, D0, a * D1), DD = Dn * D0;
            const double inv = 1.0 / (DD * a);
            const double qnew = (q1 - (b - 2.0) * q2) * (DD * inv);
            q1 = q2;
            q2 = qnew;
            q += c * qnew;
            P *= -a;
            delh = P * (a * inv);
            D1 = D0, D0 = Dn;
            hh += delh;
            const double dels = q * delh;
            s += dels;
            if (std::fabs(dels) <= kEps * std::fabs(s)) {
                converged = true;
                break;
            }
            if (std::fabs(D0) > 0x1p400) D0 *= 0x1p-400, D1 *= 0x1p-400, P *= 0x1p-800;
        }
        if (!converged) return false;
        hh = a1 * hh;
        rkmu = std::sqrt(kPi / (2.0 * x)) / s;
        rk1 = rkmu * (mu + x + 0.5 - hh) * xi;
    }
    return true;
}

// NR bessik for nu >= 0, 0 < x < inf, scaled.
inline IK ik_temme(double nu, double x) {
    if (!(nu <= kMaxOrderSteps)) return kNoIK;  // (the public functions call with nu < kDebyeOrder)
    const std::int64_t nl = static_cast<std::int64_t>(nu + 0.5);
    const double mu = nu - nl;
    const double xi = 1.0 / x;
    // CF1: I'_nu / I_nu = nu/x + 1/(2(nu+1)/x + 1/(2(nu+2)/x + ...)): positive terms
    const CfResult cf =
        lentz_backward(nu * xi, [&](int k) { return std::pair{1.0, 2.0 * (nu + k) * xi}; }, kMaxCfTerms);
    if (!cf.converged) return kNoIK;
    const double h = cf.value;
    const dd::DD xid = recip(x);
    dd::DD ril{1.0, 0.0}, ripl{h, 0.0};
    const double ril1 = ril.hi, rip1 = ripl.hi;
    std::int64_t ex = 0;
    for (std::int64_t l = nl - 1; l >= 0; --l) {
        const double v1 = nu - static_cast<double>(nl - l - 1), v0 = nu - static_cast<double>(nl - l);
        const dd::DD ritemp = xid * v1 * ril + ripl;
        ripl = xid * v0 * ritemp + ril;
        ril = ritemp;
        rescale_down(ril, ripl, ex);
    }
    const double f = (ripl / ril).hi;
    double rkmu, rk1;  // scaled: e^x K
    if (!k_pair_scaled(mu, x, rkmu, rk1)) return kNoIK;
    const double rkmup = mu * xi * rkmu - rk1;
    const double rimu = xi / (f * rkmu - rkmup);  // e^-x I_mu (the Wronskian with the scaled K)
    const Scaled fact = sc(rimu) / sc(ril.hi) * Scaled{0.5, 1 - ex};
    const Scaled ri = sc(ril1) * fact, rip = sc(rip1) * fact;
    std::int64_t ek = 0;
    dd::DD k0{rkmu, 0.0}, k1{rk1, 0.0};
    const dd::DD xi2d = xid * 2.0;
    for (std::int64_t i = 1; i <= nl; ++i) {
        const double v = nu - static_cast<double>(nl - i);  // mu + i
        const dd::DD k2 = xi2d * v * k1 + k0;
        k0 = k1;
        k1 = k2;
        rescale_down(k1, k0, ek);
    }
    rkmu = k0.hi;
    rk1 = k1.hi;
    const Scaled two_ek{0.5, 1 + ek};
    return {ri, sc(rkmu) * two_ek, rip, sc(nu * xi * rkmu - rk1) * two_ek};
}

// ---- the power series for x small next to nu ---------------------------------

// J_nu (sign = -1) or I_nu (sign = +1) for nu >= 0 and x^2 < nu + 1:
//     (x/2)^nu / Gamma(nu+1) sum_k (sign x^2/4)^k / (k! (nu+1)_k)
// The terms fall from the first (no cancellation worth a bit), and the prefix
// is formed without its exponent rounding: pow_scaled and Gamma as m 2^e. The
// derivative from the order nu + 1 (J' = (nu/x) J - J_(nu+1), I' = (nu/x) I + I_(nu+1)).
inline Scaled power_series(double nu, double x, double sign, double* deriv = nullptr) {
    auto sum_at = [&](double v) {
        const double q = sign * 0.25 * x * x;
        double s = 1.0, t = 1.0;
        for (int k = 1; k < 500; ++k) {  // |q| < (v + 1) / 4: the terms fall at least 4x a turn
            t *= q / (k * (v + k));
            s += t;
            if (std::fabs(t) <= std::fabs(s) * 0x1p-56) return s;
        }
        return kNaN;
    };
    // Gamma(nu + 1) = nu Gamma(nu): rounding nu + 1 would cost psi(nu) nu / 2 ULP (S3)
    const Scaled g1 = nu == 0.0 ? scaled(1.0) : gamma_scaled(nu) * scaled(nu);
    const Scaled pre = pow_scaled(0.5 * x, nu) / g1;
    const Scaled r = pre * scaled(sum_at(nu));
    if (deriv) {
        const Scaled r1 = pre * scaled(0.5 * x / (nu + 1.0) * sum_at(nu + 1.0));  // order nu + 1
        *deriv = nu / x * value(r) + sign * value(r1);
    }
    return r;
}
inline bool use_power_series(double nu, double x) { return x * x < nu + 1.0; }

// ---- the plain-double path for an order below kFastOrder ------------------------
//
// J_nu and Y_nu each on its own (jv needs no Y, yv no J), in float64 throughout, for
// 0 <= nu < kFastOrder and x >= kTinyArg; false where the result leaves the normal range
// (the general path below takes it). Below order 500 float64 recurrences lose no more than the
// double-double ones of jy_temme did (reference files: 13 and 15 envelope ULP at most for jv and
// yv, as before; 3-4x faster); past it they lose ~n/30 ULP (29 and 41 with the bound at 1100),
// so larger orders stay on the general path. nu = n + mu, n = round(nu), |mu| <= 1/2.
//   J: the power series where x^2 < 2 (nu + 1) (its terms fall at least 2x a turn, so they
//      cancel to a factor ~3 at most); Hankel at nu where it is accepted (x >= kHankelFrom);
//      for nu < x the upward recurrence from J_mu, J_(mu+1) where those are cheap (the order
//      0 and 1 rational fits, the closed forms at mu = -1/2, Hankel for x >= kHankelFrom),
//      which is neutral for orders below x; else Miller's backward recurrence from an order
//      N past max(nu, x), normalised by Neumann's sum (DLMF 10.23.15)
//          (x/2)^mu / Gamma(mu + 1) = J_mu + sum_k>=1 (mu + 2k) (mu + 1)_(k-1) / k! J_(mu+2k).
//   Y: Hankel at nu where accepted; else Y_mu, Y_(mu+1) (the fits at mu = 0, the closed forms
//      at mu = -1/2, Hankel for x >= kHankelFrom, Temme's series for x < kTemmeBelow, else Steed's CF2
//      with J_mu, J'_mu from Miller up to the common factor that the Wronskian fixes) and the
//      upward recurrence, which Y (the dominant solution) keeps stable.
inline constexpr double kFastOrder = 500.0;
inline constexpr double kHankelFrom = 25.0;
// Temme's series for Y below this x, Steed's CF2 above: CF2 takes ~40 terms at x = 2 and ~30
// at 3, and loses more there than the series does (focused Arb set, nu < 3, x in 1.5-6: max 32
// envelope ULP with the change at 2, 21 at 3, 35 at 4).
inline constexpr double kTemmeBelow = 3.0;

// sin x, cos x and sqrt(2 / (pi x)), shared by the Hankel values of one x.
struct HankelX {
    double s, c, amp;
};
inline HankelX hankel_x(double x) {
    const SinCos sc = sincos(x);
    return {sc.s, sc.c, std::sqrt(kTwoOverPi / x)};
}
// J and Y from Hankel's P, Q at an order v with cv = cos(pi v / 2), sv = sin(pi v / 2): chi =
// x - pi v / 2 - pi / 4 (as jy_hankel).
inline void hankel_jy_at(const HankelX& h, double cv, double sv, double P, double Q, double& J, double& Y) {
    const double cx = h.c * cv + h.s * sv, sx = h.s * cv - h.c * sv;  // cos, sin of x - pi v / 2
    const double cchi = (cx + sx) * kSqrtHalfG, schi = (sx - cx) * kSqrtHalfG;
    J = h.amp * (P * cchi - Q * schi);
    Y = h.amp * (P * schi + Q * cchi);
}
// sin(pi v), cos(pi v) with exact reduction (Apple's libm has them as one call).
inline SinCos sincospi(double v) {
#if defined(__APPLE__)
    const struct __double2 r = __sincospi_stret(v);
    return {r.__sinval, r.__cosval};
#else
    return {sinpi(v), cospi(v)};
#endif
}
// Hankel's P, Q at mu and mu + 1 for |mu| <= 1/2 and x >= kHankelFrom, where hankel_pq accepts
// both: the terms fall from the first (|4 v^2 - 1| / (8x) <= 1/25) and keep falling past the
// 2^-56 that ends the sum (they turn up from k ~ 2x - 2 >= 48 only, reached below 1e-21). So no
// test but the end, and the two orders' products in one loop, two terms (a Q's and a P's) a turn.
inline void hankel_pq_pair(double mu, double x, double& P0, double& Q0, double& P1, double& Q1) {
    const double m0 = 4.0 * mu * mu, m1 = 4.0 * (mu + 1.0) * (mu + 1.0), x8 = 8.0 * x;
    double t0 = 1.0, t1 = 1.0, p0 = 1.0, p1 = 1.0, q0 = 0.0, q1 = 0.0, sign = 1.0;
    for (int k = 1; k < 200; k += 2) {
        // k odd: Q += sign t; k + 1: P -= sign t (Q = a1/x - a3/x^3 ..., P = 1 - a2/x^2 + ...)
        const double s1 = (2.0 * k - 1.0) * (2.0 * k - 1.0), s2 = (2.0 * k + 1.0) * (2.0 * k + 1.0);
        const double w1 = 1.0 / (x8 * k), w2 = 1.0 / (x8 * (k + 1));
        t0 *= (m0 - s1) * w1;
        t1 *= (m1 - s1) * w1;
        q0 = std::fma(sign, t0, q0);
        q1 = std::fma(sign, t1, q1);
        t0 *= (m0 - s2) * w2;
        t1 *= (m1 - s2) * w2;
        p0 = std::fma(-sign, t0, p0);
        p1 = std::fma(-sign, t1, p1);
        sign = -sign;
        if (std::max(std::fabs(t0), std::fabs(t1)) <= 0x1p-57) break;  // |P| > 0.99
    }
    P0 = p0, Q0 = q0, P1 = p1, Q1 = q1;
}
// J, Y at mu and mu + 1 by Hankel, |mu| <= 1/2, x >= kHankelFrom (cos and sin of pi (mu + 1) / 2
// are -sin, cos of pi mu / 2).
inline void hankel_pair(double mu, double x, double& j0, double& j1, double& y0, double& y1) {
    double P0, Q0, P1, Q1;
    hankel_pq_pair(mu, x, P0, Q0, P1, Q1);
    const HankelX h = hankel_x(x);
    const SinCos v = sincospi(0.5 * mu);
    hankel_jy_at(h, v.c, v.s, P0, Q0, j0, y0);
    hankel_jy_at(h, -v.s, v.c, P1, Q1, j1, y1);
}
// J and Y at mu, mu + 1 in closed form: mu = 0 by the rational fits (bessel01.hpp), mu = -1/2
// by J_-1/2 = Y_1/2 ... (J_-1/2 = A cos x, J_1/2 = A sin x, Y_-1/2 = A sin x, Y_1/2 = -A cos x,
// A = sqrt(2 / (pi x))). False for other mu.
inline bool closed_pair(double mu, double x, bool want_y, double& c0, double& c1) {
    if (mu == 0.0) {
        c0 = want_y ? Y0::f(x) : J0::f(x);
        c1 = want_y ? Y1::f(x) : J1::f(x);
        return true;
    }
    if (mu == -0.5) {
        const HankelX h = hankel_x(x);
        c0 = h.amp * (want_y ? h.s : h.c);
        c1 = h.amp * (want_y ? -h.c : h.s);
        return true;
    }
    return false;
}
// The coefficient 2v / x of the recurrences as 2v (xh + xl), 1/x = xh + xl (recip): rounded once,
// with no division (2v is exact: v = mu + k is a part of nu). A rounded 1/x alone would move x
// itself by an ULP, and the result by x J'/J ULP.
inline double coef(double v2, const dd::DD& xr) { return std::fma(v2, xr.hi, v2 * xr.lo); }
// C_(mu+n) from C_mu, C_(mu+1): C_(v+1) = (2v/x) C_v - C_(v-1), each step one fma.
inline double upward(double mu, double x, std::int64_t n, double c0, double c1) {
    if (n == 0) return c0;
    const dd::DD xr = recip(x);
    for (std::int64_t k = 1; k < n; ++k) {
        const double c2 = std::fma(coef(2.0 * (mu + static_cast<double>(k)), xr), c1, -c0);
        c0 = c1;
        c1 = c2;
    }
    return c1;
}

// Miller's recurrence f_(k-1) = (2 (mu + k) / x) f_k - f_(k+1) from f_N = 1, f_(N+1) = 0 down to
// order mu: f_k = c (J_(mu+k) + e_k), c > 0 (J of an order past x is positive), with e_k / J_k
// ~ (J_N / Y_N) (Y_k / J_k), below 2^-53 when the exponent nu (a - tanh a) of J_N / J_top
// (Debye, sech a = x / N) has fallen by 19 past top = max(x, recorded order): near the turning
// point that takes N - top ~ 7.4 top^(1/3). Measured (float64 runs against a long start, top
// 0.5-100): N - top = 7 top^(1/3) + 3 settles f_0 / f_1; N below is 8 top^(1/3) + 6 for those.
// Neumann's sum also takes the orders up to ~top + 2 top^(1/3), where J_k is still ~top^(-1/3)
// and its error (J_N / Y_N) Y_k grows with k: against Arb, 8 top^(1/3) + 6 left 94 ULP, 9 top^(1/3)
// + 8 none; with the sum N is 10 top^(1/3) + 8.
// Returns f_n (n the recorded order offset), f_0, f_1, and Neumann's sum in the same units.
struct MillerOut {
    double fn, f0, f1, sum;
};
template <bool kSum>
inline MillerOut miller(double mu, double x, std::int64_t n, double top) {
    const double ca = kSum ? 10.0 : 8.0, cb = kSum ? 8.0 : 6.0;
    const auto M = static_cast<std::int64_t>((top + ca * std::cbrt(top) + cb) * 0.5) + 1;  // N = 2M
    const dd::DD xr = recip(x);
    double fk1 = 0.0, fk = 1.0;  // f_(2m+1), f_(2m)
    double fn = n == 2 * M ? 1.0 : 0.0;
    // the sum's coefficients (mu + 2m) b_m, b_m = (mu + 1)_(m-1) / m!, in units of b_M: r_m = b_m / b_M,
    // r_(m-1) = r_m m / (mu + m - 1)
    double r = 1.0, s = 0.0;
    for (std::int64_t m = M; m >= 1; --m) {
        const double dm = static_cast<double>(m);
        if constexpr (kSum) {
            s = std::fma((mu + 2.0 * dm) * r, fk, s);
            if (m >= 2) r *= dm / (mu + (dm - 1.0));
        }
        const double fo = std::fma(coef(2.0 * (mu + 2.0 * dm), xr), fk, -fk1);  // f_(2m-1)
        const double fe = std::fma(coef(2.0 * (mu + (2.0 * dm - 1.0)), xr), fo, -fk);  // f_(2m-2)
        if (2 * m - 1 == n) fn = fo;
        if (2 * m - 2 == n) fn = fe;
        fk1 = fo;
        fk = fe;
        if (std::fabs(fk) > 0x1p600) {  // (x small next to N)
            fk *= 0x1p-600;
            fk1 *= 0x1p-600;
            s *= 0x1p-600;
            fn *= 0x1p-600;
        }
    }
    return {fn, fk, fk1, kSum ? fk + s / r : 0.0};
}

// I_(nu+1) / I_nu for nu > -1, x > 0, by the backward recurrence f_(k-1) = (2 (nu + k) / x) f_k +
// f_(k+1) from f_N = 1, f_(N+1) = 0: I_(nu+k) is its minimal solution, every term is positive
// (nothing cancels), and the start's error falls as I_N / K_N: below 2^-53 from N - nu ~
// sqrt(30 x) + 4 at the most (float64 runs against a long start, x 0.5-100, nu in [-1/2, 1/2]);
// N - nu = sqrt(40 x) + 10 here.
inline double i_ratio(double nu, double x) {
    const int N = static_cast<int>(std::sqrt(40.0 * x)) + 10;
    const dd::DD xr = recip(x);
    double fk1 = 0.0, fk = 1.0;
    for (int k = N; k >= 1; --k) {
        const double f = std::fma(coef(2.0 * (nu + k), xr), fk, fk1);
        fk1 = fk;
        fk = f;
        if (fk > 0x1p600) {  // (x small next to N)
            fk *= 0x1p-600;
            fk1 *= 0x1p-600;
        }
    }
    return fk1 / fk;
}

// e^x K_nu and (want_i) e^-x I_nu for 0 <= nu < kDebyeOrder, x > 0, in float64: K_mu, K_(mu+1)
// (k_pair_scaled, nu = n + mu, |mu| <= 1/2) and K's upward recurrence K_(v+1) = (2v/x) K_v +
// K_(v-1) (K is the dominant solution: stable); I from the Wronskian I_v K_(v+1) + I_(v+1) K_v
// = 1/x with I_(nu+1) / I_nu from i_ratio. False if a loop did not end or a value is not
// normal (the general path takes it).
inline bool fast_ik_scaled(double nu, double x, bool want_i, double& ek, double& ei) {
    const auto n = static_cast<std::int64_t>(std::round(nu));
    const double mu = nu - static_cast<double>(n);
    double k0, k1;
    if (!k_pair_scaled(mu, x, k0, k1)) return false;
    const dd::DD xr = recip(x);
    for (std::int64_t k = 1; k <= n; ++k) {  // to K_nu, K_(nu+1)
        const double k2 = std::fma(coef(2.0 * (mu + static_cast<double>(k)), xr), k1, k0);
        k0 = k1;
        k1 = k2;
    }
    ek = k0;
    if (!bst::normal(ek) || !(k1 < kInf)) return false;
    if (want_i) {
        ei = 1.0 / (x * (k1 + i_ratio(nu, x) * k0));
        if (!bst::normal(ei)) return false;
    }
    return true;
}

// J_nu by the cases above; false: not this path.
inline bool fast_j(double nu, double x, double& out) {
    if (x * x < 2.0 * (nu + 1.0)) {
        // the power series, (x/2)^nu / Gamma(nu + 1) sum_k (-x^2/4)^k / (k! (nu+1)_k); Gamma(nu + 1)
        // as nu Gamma(nu) (rounding nu + 1 would cost psi(nu) nu / 2 ULP)
        const double q = -0.25 * x * x;
        double s = 1.0, t = 1.0;
        for (int k = 1; k < 200; ++k) {
            t *= q / (k * (nu + k));
            s += t;
            if (std::fabs(t) <= std::fabs(s) * 0x1p-56) break;
        }
        const double g = nu < 1.0 ? bst::tgamma(nu + 1.0) : nu * bst::tgamma(nu);
        out = std::pow(0.5 * x, nu) / g * s;
        return bst::normal(out);
    }
    double P, Q;
    if (x >= kHankelFrom && hankel_pq(nu, x, P, Q)) {
        const HankelX h = hankel_x(x);
        const SinCos v = sincospi(0.5 * nu);
        double Y;
        hankel_jy_at(h, v.c, v.s, P, Q, out, Y);
        return true;
    }
    const auto n = static_cast<std::int64_t>(std::round(nu));
    const double mu = nu - static_cast<double>(n);
    if (mu == 0.0 && n <= 1) {
        out = n == 0 ? J0::f(x) : J1::f(x);
        return true;
    }
    if (nu < x) {
        double c0, c1, y0, y1;
        if (closed_pair(mu, x, false, c0, c1) || (x >= kHankelFrom && (hankel_pair(mu, x, c0, c1, y0, y1), true))) {
            out = upward(mu, x, n, c0, c1);
            return true;
        }
    }
    const MillerOut m = miller<true>(mu, x, n, std::max(nu, x));
    out = m.fn * (std::pow(0.5 * x, mu) * temme_gammas(mu).gpl / m.sum);
    return bst::normal(out);
}

// Y_nu by the cases above; false: not this path.
inline bool fast_y(double nu, double x, double& out) {
    double P, Q;
    if (x >= kHankelFrom && hankel_pq(nu, x, P, Q)) {
        const HankelX h = hankel_x(x);
        const SinCos v = sincospi(0.5 * nu);
        double J;
        hankel_jy_at(h, v.c, v.s, P, Q, J, out);
        return true;
    }
    const auto n = static_cast<std::int64_t>(std::round(nu));
    const double mu = nu - static_cast<double>(n);
    double y0, y1, j0, j1;
    if (closed_pair(mu, x, true, y0, y1) || (x >= kHankelFrom && (hankel_pair(mu, x, j0, j1, y0, y1), true))) {
        // the seeds as they are
    } else if (x < kTemmeBelow) {
        if (!temme_y(mu, x, y0, y1)) return false;
    } else {
        double p, q;
        if (!steed_cf2(mu, x, p, q)) return false;
        const MillerOut m = miller<false>(mu, x, 0, std::max(x, 1.0));
        // (ja, jb) = (J_mu, J'_mu) / c, scaled to about 1; Y_mu = c yr and the Wronskian
        // J Y' - J' Y = q (J^2 + Y^2) = 2 / (pi x) fix c > 0 (as jy_temme)
        const double xi = 1.0 / x;
        double ja = m.f0, jb = mu * xi * m.f0 - m.f1;
        const double js = 1.0 / std::max(std::fabs(ja), std::fabs(jb));
        ja *= js;
        jb *= js;
        const double yr = (p * ja - jb) / q;
        const double cj = std::sqrt(2.0 * xi / kPi / (q * (ja * ja + yr * yr)));
        y0 = cj * yr;
        y1 = mu * xi * y0 - (q * (cj * ja) + p * y0);  // Y_(mu+1) = (mu/x) Y_mu - Y'_mu
    }
    out = upward(mu, x, n, y0, y1);
    return bst::normal(out);
}

// J and Y for x >= 30 > nu where Hankel is refused at nu itself (at a large order its
// first term (4 nu^2 - 1) / (8x) passes hankel_pq's bound of 2, so with the order nu + 1
// that jy_hankel also takes, x below ((2 nu + 2)^2 - 1) / 16 ~ nu^2 / 4: measured
// nu^2 / x = 4.0 at nu = 1.2e7-2e7): Hankel at mu = nu - floor(nu) and mu + 1, then the three-term
// recurrence up to nu. For nu < x the recurrence is neutral (its roots have
// modulus 1) in both directions, while CF1 there loses ~0.3 x ULP to the
// same neutrality (measured).
inline bool jy_upward(double nu, double x, JY& out) {
    if (!(nu <= kMaxOrderSteps)) return false;
    const double mu = nu - std::floor(nu);
    JY a, b;
    if (!jy_hankel(mu, x, a) || !jy_hankel(mu + 1.0, x, b)) return false;
    dd::DD j0{value(a.j), 0.0}, j1{value(b.j), 0.0}, y0{value(a.y), 0.0}, y1{value(b.y), 0.0};
    const std::int64_t n = static_cast<std::int64_t>(std::floor(nu));
    const dd::DD xi2d = recip(x) * 2.0;
    // order mu + k -> mu + k + 1: C_(v+1) = (2v/x) C_v - C_(v-1), v = mu + k exact
    for (std::int64_t k = 1; k < n; ++k) {
        const dd::DD c = xi2d * (nu - static_cast<double>(n - k));
        const dd::DD j2 = c * j1 - j0, y2 = c * y1 - y0;
        j0 = j1;
        j1 = j2;
        y0 = y1;
        y1 = y2;
    }
    if (n == 0) {
        out = a;
        return true;
    }
    // C_nu = j1 (order mu + n); C_(nu-1) = j0: C'_nu = C_(nu-1) - (nu/x) C_nu
    out = {scaled(j1.hi), scaled(y1.hi), scaled(j0.hi - nu / x * j1.hi), scaled(y0.hi - nu / x * y1.hi)};
    return true;
}

// J_nu, Y_nu (and derivatives) for nu >= 0, 0 < x < inf: Hankel for large x,
// the upward recurrence from Hankel at a small order for 30 <= x with nu < x,
// else Temme; J from the power series when x^2 < nu + 1 (the downward
// recurrence over ~nu steps loses ~nu/2 ULP there).
inline JY jy(double nu, double x) {
    JY r;
    if (x >= 30.0) {
        if (jy_hankel(nu, x, r)) return r;
        // (an order past kMaxOrderSteps: no value. Temme's CF1 at nu < x would lose ~0.3 x ULP)
        if (nu < x) return jy_upward(nu, x, r) ? r : kNoJY;
    }
    r = jy_temme(nu, x);
    if (use_power_series(nu, x)) {
        double jp;
        r.j = power_series(nu, x, -1.0, &jp);
        r.jp = scaled(jp);
    }
    return r;
}

// e^-x I_nu, e^x K_nu (and derivatives), the same way.
inline IK ik(double nu, double x) {
    IK r;
    if (x >= 30.0 && ik_hankel(nu, x, r)) return r;
    r = ik_temme(nu, x);
    if (use_power_series(nu, x)) {
        double ip;
        const Scaled e = exp_scaled(-x);
        r.i = power_series(nu, x, 1.0, &ip) * e;
        r.ip = scaled(ip) * e;
    }
    return r;
}

// ---- the shortcuts for an order far beyond x ---------------------------------

// log of (x/2)^nu / Gamma(nu + 1), the size of J_nu(x) and I_nu(x) for nu >> x.
inline double log_small_order_term(double nu, double x) {
    const double lx = std::log(x) - kLn2Hi;  // log(x / 2), also for the smallest x
    // from 1e8 by Stirling, nu log(e x / (2 nu)) - log(2 pi nu) / 2 (1e-9 absolute): the
    // difference of the logs is formed before nu multiplies it, so nothing overflows
    if (nu > 1e8) return nu * ((lx - std::log(nu)) + 1.0) - 0.5 * (kLog2PiHi + std::log(nu));
    return nu * lx - lgamma_pos(nu + 1.0);
}

// True when nu is so far beyond x that J (modified: I) underflows to 0 and Y (K) overflows
// (avoids an O(nu) loop for an answer of 0 or inf). The bounds on the small solution:
// |J_nu(x)| <= (x/2)^nu / Gamma(nu + 1) for real x (DLMF 10.14.4), and I_nu(x) <= that times
// e^x. For x < nu / 2, I K ~ 1/(2 nu) and |J Y| ~ 1/(pi nu tanh a) with tanh a > 0.86 (Debye,
// sech a = x / nu): past -760 - log(2 nu) the large partner overflows as well. (J without the
// e^x: jv(3e7, 1.49e7), e^-1.4e7, was a 1.5e7-step recurrence and NaN.)
inline bool order_saturates(double nu, double x, bool modified) {
    return nu > 50.0 && x < 0.5 * nu &&
           log_small_order_term(nu, x) + (modified ? x : 0.0) < -760.0 - kLn2Hi - std::log(nu);
}

// True when J_nu(x) (nu > 0, 0 < x < nu) is proven to round to 0: an upper bound below
// e^-746.2, under half the smallest subnormal (2^-1075 = e^-745.13; the nat of margin covers
// the bounds' own rounding many times over). Two rigorous bounds (user decision 2026-11-19):
//   DLMF 10.14.5: |J_nu(nu sech a)| <= e^(-nu (a - tanh a)), nu >= 0 (Debye's exponent),
//   DLMF 10.14.4: |J_nu(x)| <= (x/2)^nu / Gamma(nu + 1) (sharper for x << nu).
// a - tanh a = atanh(t) - t with t = tanh a = sqrt(1 - (x/nu)^2): below t = 1/2 the positive
// series t^3/3 + t^5/5 + ... (no cancellation as x nears nu; t from nu - x, exact for
// x >= nu / 2, so that nu ~ 1e30 next to x keeps its t), above it log(nu/x) + log1p(t) - t. Y has no such bound in the
// other direction (DLMF 10.14 bounds J only): jv(3e7, 1.9e7) = 0 (e^-7.6e6) while yv there
// stays a 1e7-step recurrence, NaN. (order_saturates above, the older test, also gives Y.)
inline constexpr double kJZeroLog = -746.2;
inline bool j_underflows(double nu, double x) {
    if (!(nu > 0.0 && x > 0.0 && x < nu)) return false;
    const double t = std::sqrt((nu - x) / nu) * std::sqrt((nu + x) / nu);
    double g;  // a - tanh a
    if (t < 0.5) {
        const double t2 = t * t;
        double term = t * t2, s = 0.0;
        for (int k = 3; k < 200; k += 2) {  // ratio t^2 <= 1/4: 2^-60 by k = 63
            s += term / k;
            term *= t2;
            if (term < 0x1p-60 * s) break;
        }
        g = s;
    } else {
        // a = acosh(nu / x) = log(nu / x) + log1p(t), the logs apart (nu / x may overflow; and
        // atanh(t) loses t's last bits as t rounds to 1 for a tiny x)
        g = (std::log(nu) - std::log(x)) + std::log1p(t) - t;
    }
    if (-nu * g < kJZeroLog) return true;
    return log_small_order_term(nu, x) < kJZeroLog;
}

// ---- an argument below 2^-200 ---------------------------------------------------
//
// There every series is its first term to 2^-400 (the next is x^2 / 4 of it):
//     J_a, I_a = (x/2)^a / Gamma(a + 1)
//     Y_a = -Gamma(a) (2/x)^a / pi,  K_a = Gamma(a) (2/x)^a / 2 where the other solution's
//                part, (x/2)^(2a) of it, is nothing;
//     else       Temme's first terms (jy_temme, ik_temme at i = 0), which hold the two
//                powers x^+-a together as a -> 0 (Y_0, K_0 ~ log x)
// The recurrences' 1/x overflows for a subnormal x, and Temme's cosh and sinh of
// e = a log(2/x) carry e ULP (170 ULP at a = 0.49, x = 1e-300, measured).
inline constexpr double kTinyArg = 0x1p-200;
struct TinyPair {
    double small, big;  // (J_a, Y_a) or (I_a, K_a), a >= 0
};
inline TinyPair tiny_argument(double a, double x, bool modified) {
    const Scaled xa = pow_scaled(x, a) * pow_scaled(0.5, a);  // (x/2)^a
    const double small = a == 0.0 ? 1.0 : value(xa / (gamma_scaled(a) * scaled(a)));
    const double d = -(std::log(x) - kLn2Hi);  // -log(x/2)
    const double e = a * d;
    // (x/2)^(2a) = e^-2e below 2^-70: the one power alone (also for a < 1/2: Temme's cosh e
    // and sinh e would carry e ULP)
    if (e > 25.0) {
        const Scaled g = gamma_scaled(a) / xa;
        return {small, modified ? value(g * scaled(0.5)) : -value(g * scaled(1.0 / kPi))};
    }
    const double pimu = kPi * a;
    const double fact = a < 1e-300 ? 1.0 : pimu / sinpi(a);
    const double fact2 = std::fabs(e) < 1e-300 ? 1.0 : std::sinh(e) / e;
    const Gammas g = temme_gammas(a);
    const double ff = fact * (g.g1 * std::cosh(e) + g.g2 * fact2 * d);
    if (modified) return {small, ff};
    const double q = 1.0 / (std::exp(e) * kPi * g.gmi);
    const double fact3 = a < 1e-300 ? 1.0 : sinpi(0.5 * a) / (0.5 * pimu);
    const double r = kPi * (0.5 * pimu) * fact3 * fact3;
    return {small, -(kTwoOverPi * ff + r * q)};
}

inline bool is_int(double v) { return v == std::floor(v); }

}  // namespace bessel_detail

// ---- the public functions -----------------------------------------------------
//
// Conventions (scipy, and the S6 user decision for its errors): x < 0 is
// defined for an integer order only (J, I; Y, K NaN); x = 0 gives the limits
// (J_0 = I_0 = 1, J_nu = I_nu = 0 for nu > 0, Y = -inf, K = inf, and for nu < 0
// the reflected limits); x = inf gives 0 (J, Y, K, the scaled I, K) and inf
// (I); nu = inf gives 0 (J, I), -inf (Y), inf (K).

namespace bessel_detail {

// J_v(x) or Y_v(x) (want_y), any real v, x >= 0 finite or inf; NaN for x < 0 except J at an integer v.
inline double jv_or_yv(double v, double x, bool want_y) {
    if (std::isnan(v) || std::isnan(x)) return kNaN;
    if (x < 0.0) {
        if (want_y || !is_int(v)) return kNaN;
        const double r = jv_or_yv(v, -x, false);
        return std::fmod(std::fabs(v), 2.0) == 1.0 ? -r : r;
    }
    if (std::isinf(v)) {
        if (v > 0.0) return want_y ? -kInf : 0.0;
        return want_y ? kNaN : 0.0;  // scipy: jv(-inf, x) = 0, yv(-inf, x) NaN
    }
    if (std::isinf(x)) return 0.0;
    const double a = std::fabs(v);
    if (x == 0.0) {
        // J_a(0) = 1 (a = 0), 0 (a > 0); Y_a(0) = -inf. Reflected for v < 0.
        if (v >= 0.0) return want_y ? -kInf : (v == 0.0 ? 1.0 : 0.0);
        const double cv = cospi(a), sv = sinpi(a);
        if (!want_y) {
            // J_-a(0) = cos(a pi) J_a(0) - sin(a pi) Y_a(0) = sign(sin(a pi)) inf; (-1)^n 0 at an integer
            if (is_int(a)) return cv * 0.0;
            return sv > 0.0 ? kInf : -kInf;
        }
        // Y_-a(0) = sin(a pi) J_a(0) + cos(a pi) Y_a(0) = -cos(a pi) inf, 0 at half-integers
        if (cv == 0.0) return 0.0;
        return cv > 0.0 ? -kInf : kInf;
    }
    if (a < kFastOrder && x >= kTinyArg) {  // the plain-double path, where its result is normal
        double r, J = 0.0, Y = 0.0;
        if (v >= 0.0) {
            if (want_y ? fast_y(a, x, r) : fast_j(a, x, r)) return r;
        } else {
            // J_-a = cos(a pi) J_a - sin(a pi) Y_a, Y_-a = sin(a pi) J_a + cos(a pi) Y_a (a zero
            // factor leaves its function out: the integers and half-integers)
            const double cv = cospi(a), sv = sinpi(a);
            const bool need_j = want_y ? sv != 0.0 : cv != 0.0, need_y = want_y ? cv != 0.0 : sv != 0.0;
            if ((!need_j || fast_j(a, x, J)) && (!need_y || fast_y(a, x, Y))) {
                if (!want_y) return (need_j ? cv * J : 0.0) - (need_y ? sv * Y : 0.0);
                return (need_j ? sv * J : 0.0) + (need_y ? cv * Y : 0.0);
            }
        }
    }
    if (order_saturates(a, x, false)) {
        // J_a = 0, Y_a = -inf; reflected: J_-a = -sin(a pi) Y_a, Y_-a = cos(a pi) Y_a
        if (v >= 0.0) return want_y ? -kInf : 0.0;
        const double cv = cospi(a), sv = sinpi(a);
        if (!want_y) return sv == 0.0 ? cv * 0.0 : (sv > 0.0 ? kInf : -kInf);
        return cv == 0.0 ? 0.0 : (cv > 0.0 ? -kInf : kInf);
    }
    // J alone proven below the subnormals (j_underflows); a negative non-integer order also
    // takes Y_a (J_-a = cos(a pi) J_a - sin(a pi) Y_a): computed as before
    if (!want_y && (v >= 0.0 || is_int(a)) && j_underflows(a, x)) return v >= 0.0 ? 0.0 : cospi(a) * 0.0;
    double J, Y;
    if (x < kTinyArg) {
        const TinyPair t = tiny_argument(a, x, false);
        J = t.small, Y = t.big;
    } else {
        const JY r = jy(a, x);
        J = value(r.j), Y = value(r.y);
    }
    if (v >= 0.0) return want_y ? Y : J;
    const double cv = cospi(a), sv = sinpi(a);
    if (!want_y) return (cv == 0.0 ? 0.0 : cv * J) - (sv == 0.0 ? 0.0 : sv * Y);
    return (sv == 0.0 ? 0.0 : sv * J) + (cv == 0.0 ? 0.0 : cv * Y);
}

// I_v(x) (scaled: e^-|x| I_v) or K_v(x) (scaled: e^x K_v).
// Debye's uniform expansion for large order (DLMF 10.41.3-4), nu >= kDebyeOrder: with
// s = sqrt(nu^2 + x^2), p = nu / s and eta nu = s - nu log((nu + s) / x),
//     I_nu(x) ~ e^(nu eta) / sqrt(2 pi s) sum U_k(p) / nu^k,
//     K_nu(x) ~ sqrt(pi / (2 s)) e^(-nu eta) sum (-1)^k U_k(p) / nu^k,
// U_0..U_10 (coeffs/debye_u.hpp; the next term is below 2e-16 relative from nu = 30). The recurrences above take
// O(nu) steps (iv(1e9, 1e9) took 26 s and lost the answer to CF1's cap); here the exponent
// nu eta (-+ x for the scaled forms), with nu and s ~ 1e9 cancelling to a few hundred, is
// formed in double-double and exponentiated once.
inline constexpr double kDebyeOrder = 30.0;
// The value itself (not e^-+x times the scaled one: with x ~ 1e9 both factors leave
// exp_scaled's range and their saturated exponents would cancel). nu >= kDebyeOrder, x > 0.
inline double debye_value(double v, double x, bool want_k, bool scaled_out) {
    const double nu = std::fabs(v);
    auto sum_at = [&](double p, double sign) {  // sum_k (sign^k) U_k(p) / nu^k
        const double p2 = p * p, inv = sign / nu;
        double sum = 0.0;
        for (int k = kDebyeTerms - 1; k >= 0; --k) {
            const auto& c = kDebyeV[static_cast<std::size_t>(k)];
            double u = 0.0;
            for (std::size_t j = c.size() - static_cast<std::size_t>(k) - 1; j < c.size(); ++j) u = u * p2 + c[j];
            sum = sum * inv + u * std::pow(p, k);
        }
        return sum;
    };
    const double z = x / nu;
    if (z > 0x1p30) {
        // x far beyond nu: s - x and the logarithm, each ~nu^2 / x, would be lost in s ~ x (and
        // x^2 overflows from 1e154). With q = 1 / (z + sqrt(1 + z^2)),
        //     nu eta - x = nu (q - log1p((1 + q) / z)) = -nu^2 / (2x) (1 + O(1/z^2)),
        // in plain doubles. I itself is inf and K is 0 here (x > 3e10).
        if (!scaled_out) return want_k ? 0.0 : kInf;
        const double q = 1.0 / (z + std::sqrt(1.0 + z * z));
        const double e = nu * (q - std::log1p((1.0 + q) / z));  // <= 0
        const double s = x * std::sqrt(1.0 + 1.0 / (z * z)), p = nu / s;
        if (want_k) return value(exp_scaled(-e) * sc(1.25331413731550025121 / std::sqrt(s) * sum_at(p, -1.0)));
        // (v < 0, not an integer: the K term carries e^-2x, nothing here)
        return value(exp_scaled(e) * sc(0.39894228040143267794 / std::sqrt(s) * sum_at(p, 1.0)));
    }
    if (nu > 0x1p500 || x > 0x1p500) {
        // Both beyond the doubles' squares, x / nu <= 2^30: the exponent nu eta, eta(z) =
        // sqrt(1 + z^2) + log(z / (1 + sqrt(1 + z^2))), is past every range, so the value is
        // 0 or inf by its sign alone (the scaled forms: nu (eta - z) < 0 always).
        const double r = std::sqrt(1.0 + z * z), eta = r + std::log(z / (1.0 + r));
        if (scaled_out) return want_k ? kInf : (v >= 0.0 || is_int(nu) ? 0.0 : kNaN);
        if (!(std::fabs(eta) * nu > 1e4)) return kNaN;  // next to eta's zero (z = 0.6627): not resolved
        if (want_k) return eta > 0.0 ? 0.0 : kInf;
        return v >= 0.0 || is_int(nu) ? (eta > 0.0 ? kInf : 0.0) : kNaN;
    }
    const dd::DD s2 = dd::two_prod(nu, nu) + dd::two_prod(x, x);
    const double sh = std::sqrt(s2.hi);
    const dd::DD sh2 = dd::two_prod(sh, sh);
    const dd::DD s = dd::DD{sh, 0.0} + ((s2 - sh2) / (2.0 * sh)).hi;  // one Newton step
    const dd::DD lg = dd::log(s + nu) - dd::log(x);  // log((nu + s) / x)
    const dd::DD nueta = s - lg * nu;
    const double p = nu / sh, p2 = p * p, inv = 1.0 / nu;
    // sum_k U_k(p) / nu^k and sum_k (-1)^k U_k(p) / nu^k, U_k = p^k V_k(p^2) (coeffs/debye_u.hpp)
    double sum_i = 0.0, sum_k = 0.0;
    for (int k = kDebyeTerms - 1; k >= 0; --k) {
        const auto& c = kDebyeV[static_cast<std::size_t>(k)];
        double v = 0.0;
        for (std::size_t j = c.size() - static_cast<std::size_t>(k) - 1; j < c.size(); ++j) v = v * p2 + c[j];
        const double u = v * std::pow(p, k);
        sum_i = sum_i * inv + u;
        sum_k = sum_k * -inv + u;
    }
    auto expo = [](dd::DD e) {  // e^(hi + lo); beyond exp_scaled's range the result is 0 or inf anyway
        Scaled r = exp_scaled(e.hi);
        if (std::isfinite(r.m)) r.m = std::fma(r.m, e.lo, r.m);  // e^lo = 1 + lo where the result is in range
        return r;
    };
    // exponents: I ~ e^(nu eta), K ~ e^(-nu eta); the scaled ones e^-x I, e^x K
    const dd::DD ei = scaled_out ? nueta - x : nueta, ek = scaled_out ? -nueta + x : -nueta;
    const Scaled k = expo(ek) * sc(std::sqrt(kPi / (2.0 * sh)) * sum_k);
    if (want_k) return value(k);  // K_-nu = K_nu
    const Scaled i = expo(ei) * sc(sum_i / std::sqrt(2.0 * kPi * sh));
    if (v >= 0.0 || is_int(nu)) return value(i);
    // I_-nu = I_nu + (2/pi) sin(nu pi) K_nu; scaled, the K term is e^-x K_nu: exponent -nu eta - x
    const Scaled kterm = (scaled_out ? expo(-nueta - x) : expo(-nueta)) *
                         sc(std::sqrt(kPi / (2.0 * sh)) * sum_k * kTwoOverPi * sinpi(nu));
    return value(i) + value(kterm);
}

inline double iv_or_kv(double v, double x, bool want_k, bool scaled_out) {
    if (std::isnan(v) || std::isnan(x)) return kNaN;
    if (x < 0.0) {
        if (want_k || !is_int(v)) return kNaN;
        const double r = iv_or_kv(v, -x, false, scaled_out);
        return std::fmod(std::fabs(v), 2.0) == 1.0 ? -r : r;
    }
    const double a = std::fabs(v);
    if (std::isinf(v)) {
        if (want_k) return kInf;  // K_-v = K_v
        return 0.0;  // scipy: iv(+-inf, x) = 0
    }
    if (std::isinf(x)) {
        if (want_k) return 0.0;
        return scaled_out ? 0.0 : kInf;
    }
    if (x == 0.0) {
        if (want_k) return kInf;
        if (v >= 0.0 || is_int(a)) return v == 0.0 ? 1.0 : 0.0;
        // I_-a(0) = I_a(0) + (2/pi) sin(a pi) K_a(0) = sign(sin(a pi)) inf
        return sinpi(a) > 0.0 ? kInf : -kInf;
    }
    if (order_saturates(a, x, true) && (v >= 0.0 || want_k || is_int(a))) return want_k ? kInf : 0.0;
    if (x < kTinyArg) {  // (e^+-x = 1: the scaled forms are the same)
        const TinyPair t = tiny_argument(a, x, true);
        if (want_k) return t.big;
        if (v >= 0.0 || is_int(a)) return t.small;
        const double sv = sinpi(a);  // I_-a = I_a + (2/pi) sin(a pi) K_a
        return sv == 0.0 ? t.small : t.small + kTwoOverPi * sv * t.big;
    }
    if (a >= kDebyeOrder) return debye_value(v, x, want_k, scaled_out);
    const IK r = ik(a, x);
    // scaled values; unscaled ones multiply by e^+-x as m 2^e
    if (want_k) return value(scaled_out ? r.k : r.k * exp_scaled(-x));
    Scaled I = r.i;
    if (v < 0.0 && !is_int(a)) {
        // I_-a = I_a + (2/pi) sin(a pi) K_a; scaled: e^-x K_a = (e^x K_a) e^-2x
        const Scaled kterm = r.k * exp_scaled(-2.0 * x) * sc(kTwoOverPi * sinpi(a));
        I = sc(value(I) + value(kterm));  // both in range: e^-x I and e^-x K
    }
    return value(scaled_out ? I : I * exp_scaled(x));
}

// iv for |v| < 30 and kv for |v| < 400 (x > 0): Boost (special/boost.hpp; 2-7x faster
// there, as accurate); ours elsewhere and where Boost's result is not normal.
inline constexpr double kBoostArgMin = kTinyArg, kBoostArgMax = 1e4;
// The float64 path (fast_ik_scaled) before Boost, for |v| < kDebyeOrder and x in [kTinyArg, 700)
// (e^+-x in range), where its values are normal: a negative non-integer order takes both
// (I_-a = I_a + (2/pi) sin(a pi) K_a; K_-a = K_a).
inline bool fast_iv_kv(double v, double x, bool want_k, bool scaled_out, double& out) {
    if (!(std::fabs(v) < kDebyeOrder && x >= kTinyArg && x < 700.0)) return false;
    const double a = std::fabs(v);
    const bool reflect = !want_k && v < 0.0 && !is_int(a);
    if (!want_k && !reflect && x * x < 4.0 * (a + 1.0)) {
        // I alone where its power series' terms fall from the first (positive, nothing cancels):
        // (x/2)^a / Gamma(a + 1) sum (x^2/4)^k / (k! (a+1)_k), Gamma(a + 1) = a Gamma(a) (as fast_j)
        const double q = 0.25 * x * x;
        double sum = 1.0, t = 1.0;
        for (int k = 1; k < 200; ++k) {
            t *= q / (k * (a + k));
            sum += t;
            if (t <= sum * 0x1p-56) break;
        }
        const double g = a < 1.0 ? bst::tgamma(a + 1.0) : a * bst::tgamma(a);
        out = std::pow(0.5 * x, a) / g * sum;
        if (scaled_out) out *= std::exp(-x);
        return bst::normal(out);
    }
    double ek, ei = 0.0;
    if (!fast_ik_scaled(a, x, !want_k, ek, ei)) return false;
    if (want_k) {
        out = scaled_out ? ek : ek * std::exp(-x);
    } else {
        if (reflect) ei += kTwoOverPi * sinpi(a) * ek * std::exp(-2.0 * x);
        out = scaled_out ? ei : ei * std::exp(x);
    }
    return bst::normal(out);
}
inline double iv_route(double v, double x) {
    double r;
    if (fast_iv_kv(v, x, false, false, r)) return r;
    // (x within kBoostArgMin..kBoostArgMax: beyond, I overflows -- Boost did not return for
    // x >= 1e12 -- and below, the first terms are the value)
    if (std::fabs(v) < 30.0 && x >= kBoostArgMin && x < kBoostArgMax) {
        const double r = bst::i(v, x);
        if (bst::normal(r)) return r;
    }
    return iv_or_kv(v, x, false, false);
}
inline double kv_route(double v, double x) {
    double r;
    if (fast_iv_kv(v, x, true, false, r)) return r;
    if (std::fabs(v) < 400.0 && x >= kBoostArgMin && x < kBoostArgMax) {
        const double r = bst::k(v, x);
        if (bst::normal(r)) return r;
    }
    return iv_or_kv(v, x, true, false);
}
// ive, kve: Boost's iv, kv times e^-+x in the same regions for 0 < x < 700 (both factors in
// range there; one more rounding).
inline double ive_route(double v, double x) {
    double r;
    if (fast_iv_kv(v, x, false, true, r)) return r;
    if (std::fabs(v) < 30.0 && x >= kBoostArgMin && x < 700.0) {
        const double r = bst::i(v, x);
        if (bst::normal(r)) {
            const double s = r * std::exp(-x);
            if (bst::normal(s)) return s;
        }
    }
    return iv_or_kv(v, x, false, true);
}
inline double kve_route(double v, double x) {
    double r;
    if (fast_iv_kv(v, x, true, true, r)) return r;
    if (std::fabs(v) < 400.0 && x >= kBoostArgMin && x < 700.0) {
        const double r = bst::k(v, x);
        if (bst::normal(r)) {
            const double s = r * std::exp(x);
            if (bst::normal(s)) return s;
        }
    }
    return iv_or_kv(v, x, true, true);
}

}  // namespace bessel_detail

#define SD_BESSEL_OP(Struct, Name, Cost, Body)                      \
    struct Struct : Real {                                          \
        static constexpr const char* name = Name;                   \
        static constexpr int nin = 2, nout = 1;                     \
        static constexpr std::array<const char*, 2> args{"v", "x"}; \
        static constexpr double cost_ns = cost::Cost;               \
        static double f(double v, double x) { return Body; }        \
    };

SD_BESSEL_OP(Jv, "jv", kJv, bessel_detail::jv_or_yv(v, x, false))
SD_BESSEL_OP(Yv, "yv", kYv, bessel_detail::jv_or_yv(v, x, true))
SD_BESSEL_OP(Jve, "jve", kJv, bessel_detail::jv_or_yv(v, x, false))  // real x: exp(-|Im x|) = 1
SD_BESSEL_OP(Yve, "yve", kYv, bessel_detail::jv_or_yv(v, x, true))
SD_BESSEL_OP(Iv, "iv", kIv, bessel_detail::iv_route(v, x))
SD_BESSEL_OP(Kv, "kv", kKv, bessel_detail::kv_route(v, x))
SD_BESSEL_OP(Ive, "ive", kIv, bessel_detail::ive_route(v, x))
SD_BESSEL_OP(Kve, "kve", kKv, bessel_detail::kve_route(v, x))
#undef SD_BESSEL_OP

}  // namespace sd::special
