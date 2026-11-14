// Bessel functions of real order (docs/api/special.md §8.2,
// docs/design/special.md §5.8; S6): jv yv iv kv and the scaled jve yve ive kve.
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
//   - x >= max(30, nu^2): Hankel's asymptotic series (CF1 needs ~x terms there).
// Debye's uniform expansion for large nu is not used (design): a large nu
// is right but costs O(nu).
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

#include "special/boost.hpp"

#include "special/cf.hpp"
#include "special/coeffs/rgamma_taylor.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace sd::special {

namespace bessel_detail {

inline constexpr double kTwoOverPi = 0.63661977236758134308;
inline constexpr double kTiny = 0x1p-1000;

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
    const double s = std::sin(x), c = std::cos(x);
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

// NR bessjy for nu >= 0, 0 < x < inf.
inline JY jy_temme(double nu, double x) {
    const std::int64_t nl = x < 2.0 ? static_cast<std::int64_t>(nu + 0.5)
                                    : std::max<std::int64_t>(0, static_cast<std::int64_t>(nu - x + 1.5));
    const double mu = nu - nl, mu2 = mu * mu;
    const double xi = 1.0 / x, xi2 = 2.0 * xi, w = xi2 / kPi;
    // CF1: f = J'_nu / J_nu = nu/x - 1/(2(nu+1)/x - 1/(2(nu+2)/x - ...)); isign = sign of J_nu
    // relative to the downward recurrence's start (NR), from the forward pass.
    int isign = 1, terms = 0;
    {
        double b = xi2 * nu, d = 0.0, c = std::max(nu * xi, kTiny);
        for (int i = 1; i <= 50000000; ++i) {
            b += xi2;
            d = b - d;
            if (std::fabs(d) < kTiny) d = kTiny;
            c = b - 1.0 / c;
            if (std::fabs(c) < kTiny) c = kTiny;
            d = 1.0 / d;
            const double del = c * d;
            if (d < 0.0) isign = -isign;
            terms = i;
            if (std::fabs(del - 1.0) <= kEps) break;
        }
    }
    const double h = cf_backward(nu * xi, [&](int k) { return std::pair{-1.0, 2.0 * (nu + k) * xi}; }, terms + 4);
    // downward recurrence from nu to mu, with the exponent kept apart
    const dd::DD xid = recip(x);
    dd::DD rjl{static_cast<double>(isign), 0.0}, rjpl{h * isign, 0.0};
    const double rjl1 = rjl.hi, rjp1 = rjpl.hi;
    std::int64_t ex = 0;  // true = stored * 2^ex
    for (std::int64_t l = nl - 1; l >= 0; --l) {
        const double v1 = nu - static_cast<double>(nl - l - 1), v0 = nu - static_cast<double>(nl - l);  // mu+l+1, mu+l
        const dd::DD rjtemp = xid * v1 * rjl + rjpl;
        rjpl = xid * v0 * rjtemp - rjl;
        rjl = rjtemp;
        rescale_down(rjl, rjpl, ex);
    }
    if (rjl.hi == 0.0) rjl = {kEps, 0.0};
    const double f = (rjpl / rjl).hi;
    double rjmu, rymu, rymup, ry1;
    if (x < 2.0) {
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
        for (int i = 1; i < 1000; ++i) {
            ff = (i * ff + p + q) / (i * static_cast<double>(i) - mu2);
            c *= dd / i;
            p /= (i - mu);
            q /= (i + mu);
            const double del = c * (ff + r * q);
            sum += del;
            const double del1 = c * p - i * del;
            sum1 += del1;
            if (std::fabs(del) <= (1.0 + std::fabs(sum)) * 0x1p-56) break;
        }
        rymu = -sum;
        ry1 = -sum1 * xi2;
        rymup = mu * xi * rymu - ry1;
        rjmu = w / (rymup - f * rymu);
    } else {
        // Steed's CF2: p + i q = (J'_mu + i Y'_mu) / (J_mu + i Y_mu)
        double a = 0.25 - mu2, p = -0.5 * xi, q = 1.0;
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
        for (int i = 1; i < 50000000; ++i) {
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
            if (std::fabs(dlr - 1.0) + std::fabs(dli) <= kEps) break;
        }
        const double gam = (p - f) / q;
        rjmu = std::sqrt(w / ((p - f) * gam + q));
        rjmu = std::copysign(rjmu, rjl.hi);
        rymu = rjmu * gam;
        rymup = rymu * (p + q / gam);
        ry1 = mu * xi * rymu - rymup;
    }
    // J_nu = rjl1 * rjmu / (rjl 2^ex)
    const Scaled fact = sc(rjmu) / sc(rjl.hi) * Scaled{0.5, 1 - ex};
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
    const double ai = 1.0 / std::sqrt(2.0 * kPi * x), ak = std::sqrt(kPi / (2.0 * x));
    const double I0 = ai * si0, I1 = ai * si1, K0 = ak * sk0, K1 = ak * sk1;
    // I' = I_(nu+1) + (nu/x) I, K' = (nu/x) K - K_(nu+1)
    out = {sc(I0), sc(K0), sc(I1 + nu / x * I0), sc(nu / x * K0 - K1)};
    return true;
}

// NR bessik for nu >= 0, 0 < x < inf, scaled.
inline IK ik_temme(double nu, double x) {
    const std::int64_t nl = static_cast<std::int64_t>(nu + 0.5);
    const double mu = nu - nl, mu2 = mu * mu;
    const double xi = 1.0 / x, xi2 = 2.0 * xi;
    // CF1: I'_nu / I_nu = nu/x + 1/(2(nu+1)/x + 1/(2(nu+2)/x + ...)): positive terms
    const CfResult cf =
        lentz_backward(nu * xi, [&](int k) { return std::pair{1.0, 2.0 * (nu + k) * xi}; }, 50000000);
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
        for (int i = 1; i < 1000; ++i) {
            ff = (i * ff + p + q) / (i * static_cast<double>(i) - mu2);
            c *= dd / i;
            p /= (i - mu);
            q /= (i + mu);
            const double del = c * ff;
            sum += del;
            sum1 += c * (p - i * ff);
            if (std::fabs(del) <= std::fabs(sum) * 0x1p-56) break;
        }
        const double ex2 = std::exp(x);
        rkmu = sum * ex2;
        rk1 = sum1 * xi2 * ex2;
    } else {
        // Steed's CF2 (Temme's form) for K_mu: no e^-x, so e^x K_mu
        double b = 2.0 * (1.0 + x), d = 1.0 / b, hh = d, delh = d, q1 = 0.0, q2 = 1.0;
        const double a1 = 0.25 - mu2;
        double q = a1, c = a1, a = -a1, s = 1.0 + q * delh;
        for (int i = 1; i < 50000000; ++i) {
            a -= 2 * i;
            c = -a * c / (i + 1.0);
            const double qnew = (q1 - b * q2) / a;
            q1 = q2;
            q2 = qnew;
            q += c * qnew;
            b += 2.0;
            d = 1.0 / (b + a * d);
            delh = (b * d - 1.0) * delh;
            hh += delh;
            const double dels = q * delh;
            s += dels;
            if (std::fabs(dels / s) <= kEps) break;
        }
        hh = a1 * hh;
        rkmu = std::sqrt(kPi / (2.0 * x)) / s;
        rk1 = rkmu * (mu + x + 0.5 - hh) * xi;
    }
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
        for (int k = 1; k < 500; ++k) {
            t *= q / (k * (v + k));
            s += t;
            if (std::fabs(t) <= std::fabs(s) * 0x1p-56) break;
        }
        return s;
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

// J and Y for x >= 30 > nu where Hankel does not converge at nu itself (nu^2 >
// ~16 x): Hankel at mu = nu - floor(nu) and mu + 1, then the three-term
// recurrence up to nu. For nu < x the recurrence is neutral (its roots have
// modulus 1) in both directions, while CF1 there loses ~0.3 x ULP to the
// same neutrality (measured, docs/learn/S6-bessel-airy.md).
inline bool jy_upward(double nu, double x, JY& out) {
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
        if (nu < x && jy_upward(nu, x, r)) return r;
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
inline double log_small_order_term(double nu, double x) { return nu * std::log(0.5 * x) - lgamma_pos(nu + 1.0); }

// True when nu is so far beyond x that J and I underflow to 0 and Y, K overflow
// (avoids an O(nu) loop for an answer of 0 or inf).
inline bool order_saturates(double nu, double x) {
    // I K ~ 1/(2 nu) and J Y ~ -1/(pi nu) for nu >> x: past -760 - log(2 nu) the
    // large partner overflows as well
    return nu > 50.0 && x < 0.5 * nu && log_small_order_term(nu, x) + x < -760.0 - std::log(2.0 * nu);
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
    if (order_saturates(a, x)) {
        // J_a = 0, Y_a = -inf; reflected: J_-a = -sin(a pi) Y_a, Y_-a = cos(a pi) Y_a
        if (v >= 0.0) return want_y ? -kInf : 0.0;
        const double cv = cospi(a), sv = sinpi(a);
        if (!want_y) return sv == 0.0 ? cv * 0.0 : (sv > 0.0 ? kInf : -kInf);
        return cv == 0.0 ? 0.0 : (cv > 0.0 ? -kInf : kInf);
    }
    const JY r = jy(a, x);
    const double J = value(r.j), Y = value(r.y);
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
// U_0..U_4 (the next term is below 1e-20 relative from nu = 1e4). The recurrences above take
// O(nu) steps (iv(1e9, 1e9) took 26 s and lost the answer to CF1's cap); here the exponent
// nu eta (-+ x for the scaled forms), with nu and s ~ 1e9 cancelling to a few hundred, is
// formed in double-double and exponentiated once.
inline constexpr double kDebyeOrder = 1e4;
// The value itself (not e^-+x times the scaled one: with x ~ 1e9 both factors leave
// exp_scaled's range and their saturated exponents would cancel). nu >= kDebyeOrder, x > 0.
inline double debye_value(double v, double x, bool want_k, bool scaled_out) {
    const double nu = std::fabs(v);
    const dd::DD s2 = dd::two_prod(nu, nu) + dd::two_prod(x, x);
    const double sh = std::sqrt(s2.hi);
    const dd::DD sh2 = dd::two_prod(sh, sh);
    const dd::DD s = dd::DD{sh, 0.0} + ((s2 - sh2) / (2.0 * sh)).hi;  // one Newton step
    const dd::DD lg = dd::log(s + nu) - dd::log(x);  // log((nu + s) / x)
    const dd::DD nueta = s - lg * nu;
    const double p = nu / sh, p2 = p * p, inv = 1.0 / nu;
    const double u1 = p * (3.0 - 5.0 * p2) / 24.0;
    const double u2 = p2 * (81.0 + p2 * (-462.0 + 385.0 * p2)) / 1152.0;
    const double u3 = p * p2 * (30375.0 + p2 * (-369603.0 + p2 * (765765.0 - 425425.0 * p2))) / 414720.0;
    const double u4 =
        p2 * p2 * (4465125.0 + p2 * (-94121676.0 + p2 * (349922430.0 + p2 * (-446185740.0 + 185910725.0 * p2)))) /
        39813120.0;
    auto expo = [](dd::DD e) {  // e^(hi + lo); beyond exp_scaled's range the result is 0 or inf anyway
        Scaled r = exp_scaled(e.hi);
        r.m = std::fma(r.m, e.lo, r.m);  // e^lo = 1 + lo where the result is in range
        return r;
    };
    // exponents: I ~ e^(nu eta), K ~ e^(-nu eta); the scaled ones e^-x I, e^x K
    const dd::DD ei = scaled_out ? nueta - x : nueta, ek = scaled_out ? -nueta + x : -nueta;
    const Scaled k = expo(ek) * sc(std::sqrt(kPi / (2.0 * sh)) * (1.0 + inv * (-u1 + inv * (u2 + inv * (-u3 + inv * u4)))));
    if (want_k) return value(k);  // K_-nu = K_nu
    const Scaled i = expo(ei) * sc((1.0 + inv * (u1 + inv * (u2 + inv * (u3 + inv * u4)))) / std::sqrt(2.0 * kPi * sh));
    if (v >= 0.0 || is_int(nu)) return value(i);
    // I_-nu = I_nu + (2/pi) sin(nu pi) K_nu; scaled, the K term is e^-x K_nu: exponent -nu eta - x
    const Scaled kterm = (scaled_out ? expo(-nueta - x) : expo(-nueta)) *
                         sc(std::sqrt(kPi / (2.0 * sh)) * (1.0 + inv * (-u1 + inv * (u2 + inv * (-u3 + inv * u4)))) *
                            kTwoOverPi * sinpi(nu));
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
        return 0.0;               // scipy: iv(+-inf, x) = 0
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
    if (order_saturates(a, x) && (v >= 0.0 || want_k || is_int(a))) return want_k ? kInf : 0.0;
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
inline double iv_route(double v, double x) {
    if (std::fabs(v) < 30.0 && x > 0.0 && std::isfinite(x)) {
        const double r = bst::i(v, x);
        if (bst::normal(r)) return r;
    }
    return iv_or_kv(v, x, false, false);
}
inline double kv_route(double v, double x) {
    if (std::fabs(v) < 400.0 && x > 0.0 && std::isfinite(x)) {
        const double r = bst::k(v, x);
        if (bst::normal(r)) return r;
    }
    return iv_or_kv(v, x, true, false);
}

}  // namespace bessel_detail

#define SD_BESSEL_OP(Struct, Name, Cost, Body)                                 \
    struct Struct : Real {                                                     \
        static constexpr const char* name = Name;                              \
        static constexpr int nin = 2, nout = 1;                                \
        static constexpr std::array<const char*, 2> args{"v", "x"};            \
        static constexpr double cost_ns = cost::Cost;                          \
        static double f(double v, double x) { return Body; }                   \
    };

SD_BESSEL_OP(Jv, "jv", kJv, bessel_detail::jv_or_yv(v, x, false))
SD_BESSEL_OP(Yv, "yv", kYv, bessel_detail::jv_or_yv(v, x, true))
SD_BESSEL_OP(Jve, "jve", kJv, bessel_detail::jv_or_yv(v, x, false))  // real x: exp(-|Im x|) = 1
SD_BESSEL_OP(Yve, "yve", kYv, bessel_detail::jv_or_yv(v, x, true))
SD_BESSEL_OP(Iv, "iv", kIv, bessel_detail::iv_route(v, x))
SD_BESSEL_OP(Kv, "kv", kKv, bessel_detail::kv_route(v, x))
SD_BESSEL_OP(Ive, "ive", kIv, bessel_detail::iv_or_kv(v, x, false, true))
SD_BESSEL_OP(Kve, "kve", kKv, bessel_detail::iv_or_kv(v, x, true, true))
#undef SD_BESSEL_OP

}  // namespace sd::special
