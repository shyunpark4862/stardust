// The regularised incomplete beta function and its inverses
// (docs/api/special.md §3, docs/design/special.md §5.4-5.5; S4):
//
//     I_x(a, b) = B(x; a, b) / B(a, b)     betainc
//     1 - I_x(a, b) = I_(1-x)(b, a)        betaincc
//
// The method is DiDonato & Morris's BRATIO (ACM TOMS 708, 1992), whose
// choice of side and method follows a, b and x (ibeta below):
//     BPSER  the power series in x, for b <= 1 or b x <= 0.7
//     BUP    I_x(a, b) - I_x(a+n, b), a finite sum: shifts a or b to where
//            another method applies
//     BGRAT  a large, b <= 1, x near 1: an expansion in incomplete gamma
//            functions (Q(b, u) from incgamma.hpp)
//     BFRAC  a continued fraction, x below the mean
//     BASYM  a, b > 100 near the mean: Temme's uniform expansion around
//            erfc(sqrt(E))
// One of I, 1 - I is computed and the other is 1 minus it. y = 1 - x is an
// exact double-double (two_sum), and every prefix x^a y^b / B(a, b) and
// exponent comes from special/prefix.hpp with E in double-double.
//
// Two places differ from BRATIO's choice (both measured against Arb,
// bench/stats/opt/f):
//   - a, b <= 1: BRATIO takes BPSER for I whenever x^a <= 0.9 (or a >= min(0.2, b)),
//     so I reaches 0.9 and 1 - I, formed by subtraction, carries 9 times the series'
//     error (7.5e-15 at a = 2.7e-4, b = 0.014, x = 1.4e-188; Boost, which copies the
//     rule, 6.0e-15). Here the one computed is always the smaller of the pair (ibeta).
//   - x below 2^-1000: from the value at 2^-1000 (ibeta_tiny), which also serves an
//     argument below the doubles (the F distribution's w for a tiny x).
//
// S4 measured on the way: Numerical Recipes' betacf loses ~2000 ULP to a
// small denominator near its switch point for large a and b (BFRAC's b0 =
// a (lambda + 1) / (a + 1) keeps away from it); without BASYM, BFRAC reached
// 82 ULP at a, b ~ 6e4 near the mean; without BGRAT, b ~ 1e-3 with a ~ 1e5
// near x = 1 lost 2800 ULP.
//
// The inverses solve I = p (or 1 - I = q, the smaller) by Halley's method in
// x, or in y = 1 - x when the root is above 1/2, from the initial values of
// AS 109 (Cran, Martin & Thomas 1977).
#pragma once

#include "special/boost.hpp"

#include "special/cf.hpp"
#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/halley.hpp"
#include "special/ops/erf.hpp"
#include "special/ops/gamma.hpp"
#include "special/ops/incgamma.hpp"
#include "special/prefix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>

namespace sd::special {

namespace ibeta_detail {

// I, 1 - I and x dI/dx = x^a y^(b-1) / B(a, b) (= B / y: no overflow for small x).
struct IB {
    double i, ic, xderiv;
};

inline Scaled prefix_of(double a, double b, dd::DD x, dd::DD y) { return beta_prefix(a, b, beta_exponent(a, b, x, y)); }

// BPSER: I_x(a, b) = (B y^-b / a) sum_n (1-b)_n / n! a/(a+n) x^n, B the
// prefix of (a, b) at x. The low part of x enters to first order (dS/dx).
// In two parts, the factor y^-b and the series, so that a caller can bound I from the
// factor alone before it sums anything (ibeta, a, b <= 1).
inline double bpser_power(double b, dd::DD y) {
    const dd::DD e = dd::log(y) * b;  // y^-b = e^-(b log y)
    return std::exp(-e.hi) * (1.0 - e.lo);
}
inline double bpser_series(double a, double b, dd::DD x) {
    const double xh = x.hi;
    double s = 1.0, sd = 0.0, u = 1.0;
    for (int n = 1; n < 100000; ++n) {  // (b x <= 0.7 or x <= 1/2 where ibeta sends it: under 100 terms)
        u *= (n - b) * xh / n;
        const double term = u * (a / (a + n));
        s += term;
        sd += n * term;
        if (std::fabs(term) <= std::fabs(s) * 0x1p-56) return s + x.lo * (sd / xh);
    }
    return kNaN;  // not ended: no value
}
inline double bpser(double a, double b, dd::DD x, dd::DD y, Scaled pref) {
    if (pref.m == 0.0) return 0.0;
    const double s = bpser_series(a, b, x);
    return value(pref * scaled(bpser_power(b, y) * s / a));
}

// BUP: I_x(a, b) - I_x(a+n, b) = (B / a) sum_{i<n} prod_{j<i} (a+b+j) x / (a+1+j),
// all terms positive.
inline double bup(double a, double b, dd::DD x, dd::DD y, int n) {
    const Scaled pref = prefix_of(a, b, x, y);
    if (pref.m == 0.0) return 0.0;
    double d = 1.0, w = 1.0;
    for (int i = 0; i < n - 1; ++i) {
        d *= (a + b + i) * x.hi / (a + 1.0 + i);
        w += d;
    }
    return value(pref * scaled(w / a));
}

// BGRAT, a >= 15, b <= 1: with nu = a + (b-1)/2 and z = -nu log x,
//     I_x(a, b) = G (Q(b, z) + r sum_n d_n J_n),   r = z^b e^-z / Gamma(b),
//     G = Gamma(a+b) / (Gamma(a) nu^b)
// (TOMS 708 bgrat; J_n here carries the factor r, so nothing divides by
// it). z is a double-double: e^-z is the result's scale, and z reaches
// 1e5. w is the part already summed (for the stopping test); returns w + I.
// BGRAT's pieces: nu, log x (from the small one of the pair: x near 1 carries only
// the double-double's absolute accuracy (~1e-32) in its distance from 1, which is y's
// whole value when x came from a division (fdtr's w = 1 - 4e-26: z = -nu log x off by
// 2.5e-7, the result by 1.5e-9); log x = log(1 - y) = -(y + mu(-y))) and z = -nu log x.
struct BgratArgs {
    dd::DD nu, lnx, z;
};
inline BgratArgs bgrat_args(double a, double b, dd::DD x, dd::DD y) {
    const dd::DD nu = dd::two_sum(a, 0.5 * (b - 1.0));
    const dd::DD lnx = y.hi < 0.5 ? -(y + dd::log1pmx_neg(-y)) : dd::log(x);
    return {nu, lnx, -(nu * lnx)};
}
// G = sqrt(a/c) Gamma*(c) / Gamma*(a) e^(-a mu(b/a) + b log1p((b+1) / (2 nu))), c = a + b
inline double bgrat_g(double a, double b, double nu) {
    const double c = a + b;
    const double mu = dd::log1pmx_neg(dd::DD{b / a, 0.0}).hi;
    return std::sqrt(a / c) * gamma_star(c) / gamma_star(a) *
           std::exp(-a * mu + b * std::log1p((b + 1.0) / (2.0 * nu)));
}
// Q + sum_n d_n J_n from Q and r (both in any one scale; linear in the pair), tail the
// part already summed over G in that scale (for the stopping test).
inline double bgrat_series(double b, const BgratArgs& g, double Q, double r, double tail) {
    const double bm1 = b - 1.0;
    const double v = 0.25 / (g.nu.hi * g.nu.hi);
    const double t2 = 0.25 * g.lnx.hi * g.lnx.hi;
    double cs[31], ds[31];
    double sum = Q, K = Q, t = r, cn = 1.0, n2 = 0.0;
    for (int n = 1; n <= 30; ++n) {
        const double bp2n = b + n2;
        K = (bp2n * (bp2n + 1.0) * K + (g.z.hi + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        cs[n] = cn;
        double s = 0.0, coef = b - n;
        for (int i = 1; i < n; ++i) {
            s += coef * cs[i] * ds[n - i];
            coef += b;
        }
        ds[n] = bm1 * cn + s / n;
        const double dj = ds[n] * K;
        sum += dj;
        if (sum <= 0.0 || std::fabs(dj) <= 0x1p-56 * (sum + tail)) break;
    }
    return sum;
}
inline double bgrat(double a, double b, dd::DD x, dd::DD y, double w) {
    const BgratArgs g = bgrat_args(a, b, x, y);
    const dd::DD z = g.z;
    if (!(z.hi > 0.0)) return w;
    // Q(b, z) and r at z = hi + lo, to first order in lo
    const igam_detail::PQ q = igam_detail::igam(b, z.hi);
    // (z.lo / z.hi first: xderiv / z.hi and b / z.hi overflow for a z near the subnormals)
    const double zr = z.lo / z.hi;
    const double Q = q.q - zr * q.xderiv;
    dd::DD Ez = gamma_exponent(b, z.hi);
    Ez = Ez + (z.lo - b * zr);
    const double r = b * value(gamma_prefix(b, Ez));
    if (Q == 0.0 && r == 0.0) return w;
    const double G = bgrat_g(a, b, g.nu.hi);
    return w + G * bgrat_series(b, g, Q, r, w / G);
}
// log I_x(a, b) by BGRAT where I underflows (z large): Q and r from their logs
// (incgamma.hpp's log_tail for log Q), both scaled by 1 / Q.
inline double log_bgrat(double a, double b, dd::DD x, dd::DD y) {
    const BgratArgs g = bgrat_args(a, b, x, y);
    const dd::DD z = g.z;
    if (!(z.hi > 0.0)) return -kInf;
    const auto [lq0, dlq] = igam_detail::log_tail(b, z.hi, false, dd::DD{0.0, 0.0});
    const double lq = lq0 + z.lo * dlq;
    const dd::DD Ez = gamma_exponent(b, z.hi) + (z.lo - b * (z.lo / z.hi));
    const double lr = std::log(b) - Ez.hi - (Ez.lo + std::log(std::sqrt(2.0 * kPi * b) * gamma_star(b)));
    const double sum = bgrat_series(b, g, 1.0, std::exp(lr - lq), 0.0);
    return lq + (std::log(bgrat_g(a, b, g.nu.hi)) + std::log(sum));
}

// BFRAC (TOMS 708 bfrac; Boost's ibeta_fraction2), x at most the mean:
//     I_x(a, b) = B / (b0 + a1 / (b1 + a2 / (b2 + ...)))
//     a_m = (a+m-1)(a+b+m-1) m (b-m) x^2 / (a+2m-1)^2
//     b_m = m + m (b-m) x / (a+2m-1) + (a+m)(a y - b x + 1 + m (2 - x)) / (a+2m+1)
// No correction for x's low part (x = 1 - y when the roles are turned): the
// fraction takes y, the small and exact one, in a y - b x + 1; measured, the
// error is ~5 ULP without the first-order term and 3300 ULP with it.
// The fraction itself (I = B / it), NaN if it has not converged in 1e5 terms.
inline double bfrac_fraction(double a, double b, dd::DD x, dd::DD y) {
    const double xh = x.hi, yh = y.hi;
    auto gen = [&](int m) {
        const double md = m;
        const double den = a + 2.0 * md - 1.0;
        // (... x)(... x): with b = 1e300 the products of the parameters overflow and x^2 underflows
        const double an = ((a + md - 1.0) * (a + b + md - 1.0) * xh) * (md * (b - md) * xh) / (den * den);
        const double bn =
            md + md * (b - md) * xh / den + (a + md) * (a * yh - b * xh + 1.0 + md * (2.0 - xh)) / (a + 2.0 * md + 1.0);
        return std::pair{an, bn};
    };
    const CfResult r = lentz_backward(gen(0).second, gen, 100000);
    return r.converged ? r.value : kNaN;
}
inline double bfrac(double a, double b, dd::DD x, dd::DD y, Scaled pref) {
    if (pref.m == 0.0) return 0.0;
    return value(pref / scaled(bfrac_fraction(a, b, x, y)));
}

// BASYM (TOMS 708 basym), a, b > 100 and lambda = a - (a+b) x small and >= 0:
//     I_x(a, b) = u (erfc(sqrt E) / 2 + e0 e^-E sum_n (d_n w^n J_n)),
//     u = Gamma*(a+b) / (Gamma*(a) Gamma*(b)),  E = the prefix's exponent.
// The coefficients d_n come from a recursion (no tables); the leading term
// is taken from erfc at the double-double E.
inline double basym(double a, double b, dd::DD E) {
    constexpr double e0 = 1.12837916709551257390;  // 2 / sqrt(pi)
    constexpr double e1 = 0.35355339059327376220;  // 2^(-3/2)
    constexpr int num = 20;
    double h, r0, r1, w0;
    if (a < b) {
        h = a / b;
        r0 = 1.0 / (1.0 + h);
        r1 = (b - a) / b;
        w0 = 1.0 / std::sqrt(a * (1.0 + h));
    } else {
        h = b / a;
        r0 = 1.0 / (1.0 + h);
        r1 = (b - a) / a;
        w0 = 1.0 / std::sqrt(b * (1.0 + h));
    }
    const double f = E.hi;
    const double t = std::exp(-E.hi) * (1.0 - E.lo);
    const double half_erfc = 0.5 * igam_detail::erfc_sqrt(E);
    if (t == 0.0 && half_erfc == 0.0) return 0.0;
    const double z0 = std::sqrt(f);
    const double z = 0.5 * (z0 / e1);
    const double z2 = f + f;
    double a0[num + 2], b0[num + 2], c[num + 2], d[num + 2];
    a0[1] = (2.0 / 3.0) * r1;
    c[1] = -0.5 * a0[1];
    d[1] = -c[1];
    const double lead = (0.5 / e0) * Erfcx::f(z0);  // the sum's leading term (for the stopping test)
    double j0 = lead;
    double j1 = e1;
    double corr = d[1] * w0 * j1;  // the sum without its leading term j0
    double s = 1.0, h2 = h * h, hn = 1.0, w = w0, znm1 = z, zn = z2;
    for (int n = 2; n <= num; n += 2) {
        hn *= h2;
        a0[n] = 2.0 * r0 * (1.0 + h * hn) / (n + 2.0);
        const int np1 = n + 1;
        s += hn;
        a0[np1] = 2.0 * r1 * s / (n + 3.0);
        for (int i = n; i <= np1; ++i) {
            const double rr = -0.5 * (i + 1.0);
            b0[1] = rr * a0[1];
            for (int m = 2; m <= i; ++m) {
                double bsum = 0.0;
                for (int j = 1; j < m; ++j) bsum += (j * rr - (m - j)) * a0[j] * b0[m - j];
                b0[m] = rr * a0[m] + bsum / m;
            }
            c[i] = b0[i] / (i + 1.0);
            double dsum = 0.0;
            for (int j = 1; j < i; ++j) dsum += d[i - j] * c[j];
            d[i] = -(dsum + c[i]);
        }
        j0 = e1 * znm1 + (n - 1.0) * j0;
        j1 = e1 * zn + n * j1;
        znm1 *= z2;
        zn *= z2;
        w *= w0;
        const double t0 = d[n] * w * j0;
        w *= w0;
        const double t1 = d[np1] * w * j1;
        corr += t0 + t1;
        if (std::fabs(t0) + std::fabs(t1) <= 0x1p-56 * (lead + corr)) break;
    }
    const double u = gamma_star(a + b) / (gamma_star(a) * gamma_star(b));
    return u * (half_erfc + e0 * t * corr);
}

// Below this an argument goes through ibeta_tiny.
inline constexpr double kTinyX = 0x1p-1000;
inline IB ibeta_tiny(double a, double b, dd::DD lx, bool deriv);
// (I, 1 - I, x dI/dx) at x = e^lx < 2^-1000 from the triple r at 2^-1000, for an I that is
// a constant times x^a over that range: with d = a (lx - log 2^-1000) <= 0,
//     I_x = I_r e^d,      1 - I_x = (1 - I_r) + I_r (1 - e^d),
// both sums of positive terms.
inline IB power_down(const IB& r, double a, dd::DD lx) {
    // d as a double-double: its rounding to a double would enter e^d |d| times (d ~ -40 a
    // for the smallest x).
    const dd::DD d = (lx + dd::kLn2 * 1000.0) * a;
    if (d.hi >= 0.0) return r;  // (x at the reference to rounding)
    // e^d = 0. Also a NaN d: a log of -inf (x = 0 to the caller) does not survive the
    // double-double sums on the way here.
    if (!(d.hi >= -800.0)) return {0.0, 1.0, 0.0};
    const double eh = std::exp(d.hi);
    const double e = eh + eh * d.lo;
    return {r.i * e, r.ic + r.i * (-std::expm1(d.hi) - eh * d.lo), r.xderiv * e};
}

// I_x(s, c) and 1 - I_x(s, c) for c >= 2^80 max(1, s)^2 (limit_shapes), where the beta
// distribution is its gamma limit: with z = c x,
//     I_x(s, c) = P(s, z) (1 + O((s^2 + z^2) / c))
// (t = u / c in the integral: (1 - u/c)^(c-1) = e^-u (1 + O(u^2 / c)), and Gamma(s + c) /
// (Gamma(c) c^s) = 1 + O(s^2 / c)); past z = zcut = 2^-30 sqrt(c) >= 1024 max(1, s) the pair
// is 1 and 0 (Q(s, z) is below e^-1000 there), so the error is 2^-60 at most. z = hi + lo
// enters P to first order in lo (z P' = xderiv); below 2^-1000, where it may not be a
// double, from its log lz and the pair at 2^-1000 (power_down: P = z^s / Gamma(s + 1)
// (1 + O(z)) there). BRATIO's own methods are not made for these shapes (their prefix is
// NaN at a = 8, b = 4e277, x = 1e-276, and from b = 1e200 at a = 14).
inline constexpr double kLimitShape = 0x1p900, kLimitOther = 0x1p400;
// A shape from which the pair may be one of the two limits below (ibeta's one test).
inline constexpr double kPlainShape = 0x1p60;
// The smaller shape up to which the gamma limit is taken (beyond: both_large).
inline bool limit_shapes(double s, double c) { return s < kPlainShape && c >= 0x1p80 * std::max(1.0, s * s); }
inline IB gamma_limit(double s, dd::DD z, dd::DD lz, double zcut = kLimitOther) {
    if (!(z.hi < zcut)) return {1.0, 0.0, 0.0};
    if (!(z.hi >= kTinyX)) {
        const igam_detail::PQ r = igam_detail::igam(s, kTinyX);
        return power_down({r.p, r.q, r.xderiv}, s, lz);
    }
    const igam_detail::PQ r = igam_detail::igam(s, z.hi);
    const double dz = r.xderiv * (z.lo / z.hi);
    if (!(dz == dz) || dz == 0.0) return {r.p, r.q, r.xderiv};  // (xderiv past the doubles: P or Q is 0 there)
    if (r.p <= r.q) return {r.p + dz, 1.0 - (r.p + dz), r.xderiv};
    return {1.0 - (r.q - dz), r.q - dz, r.xderiv};
}

// x b - a y = x (a + b) - a (x + y = 1), the distance from the mean in units of 1 / (a + b),
// as a double-double from the exact sum of its eight partial products (Shewchuk's
// expansion: no rounding until the end). Next to the mean it is 2^-100 of its terms.
inline dd::DD mean_distance(double a, double b, dd::DD x, dd::DD y) {
    double part[10];
    int n = 0;
    auto add = [&](double t) {
        int m = 0;
        for (int i = 0; i < n; ++i) {
            const dd::DD r = dd::two_sum(part[i], t);
            if (r.lo != 0.0) part[m++] = r.lo;
            t = r.hi;
        }
        part[m++] = t;
        n = m;
    };
    auto product = [&](double p, double q, double sign) {
        const dd::DD t = dd::two_prod(p, q);
        add(sign * t.hi);
        add(sign * t.lo);
    };
    product(x.hi, b, 1.0);
    product(x.lo, b, 1.0);
    product(a, y.hi, -1.0);
    product(a, y.lo, -1.0);
    dd::DD r{0.0, 0.0};
    for (int i = 0; i < n; ++i) r = r + part[i];  // increasing in magnitude
    return r;
}

// Both shapes from 2^60 up. The distribution is narrower than x resolves (sigma / mean <=
// 2^-30), so the prefix's exponent is formed from the exact distance D = x (a + b) - a,
//     E = a mu(D / a) + b mu(-D / b),  mu(t) = t - log(1 + t)
// (lambda - 1 of prefix.hpp's beta_term is D / a: as a double-double lambda it would carry
// 2^-105, and E would be off by a 2^-105 sqrt(E a)), and BASYM with it gives the side
// below the mean; the other is 1 minus it. Away from the mean E is beyond 800 and the pair
// is 0 and 1 (BASYM's own range, lambda <= 0.03 min(a, b), ends at E ~ 2^48).
inline IB both_large(double a, double b, dd::DD x, dd::DD y, bool deriv) {
    const dd::DD D = mean_distance(a, b, x, y);
    const bool below = D.hi <= 0.0;  // x at or below the mean: I is the small side
    const dd::DD ta = D / a, tb = (-D) / b;
    auto term = [](double shape, dd::DD t) {
        // (1 + t outside 2^+-900: the term is beyond every exponent in use)
        if (!(t.hi > -1.0 + 0x1p-900 && t.hi < 0x1p900)) return dd::DD{kInf, 0.0};
        return dd::log1pmx_neg(t) * shape;
    };
    const dd::DD E = term(a, ta) + term(b, tb);
    if (!(E.hi < 800.0)) return below ? IB{0.0, 1.0, 0.0} : IB{1.0, 0.0, 0.0};  // (also a NaN E: inf - inf)
    const double w = below ? basym(a, b, E) : basym(b, a, E);
    const double xd = deriv ? value(beta_prefix(a, b, E)) / y.hi : 0.0;
    return below ? IB{w, 1.0 - w, xd} : IB{1.0 - w, w, xd};
}

// The arguments BRATIO's methods are not given (one test in ibeta sends them here); false
// if (a, b, x, y) is none of them.
inline bool ibeta_outside(double a, double b, dd::DD x, dd::DD y, bool deriv, IB& out) {
    // An argument that reached here as 0 (a double-double that underflowed on the way):
    // the value at 0, I = 0 or 1 - I = 0 exactly.
    if (x.hi <= 0.0) return out = {0.0, 1.0, 0.0}, true;
    if (y.hi <= 0.0) return out = {1.0, 0.0, 0.0}, true;
    if (std::min(a, b) >= kPlainShape) return out = both_large(a, b, x, y, deriv), true;
    // One shape far beyond the other: the gamma limit in c x (lower) or in c y (upper).
    if (limit_shapes(a, b))
        return out = gamma_limit(a, x * b, dd::log_pos(x) + dd::log(b), 0x1p-30 * std::sqrt(b)), true;
    if (limit_shapes(b, a)) {
        const IB r = gamma_limit(b, y * a, dd::log_pos(y) + dd::log(a), 0x1p-30 * std::sqrt(a));
        return out = {r.ic, r.i, deriv ? r.xderiv * (x.hi / y.hi) : 0.0}, true;
    }
    // An argument below 2^-1000 (ibeta_tiny; the other shape below 2^900, so that its
    // product with the argument is nothing).
    if (x.hi < kTinyX && b < kLimitShape) return out = ibeta_tiny(a, b, dd::log_pos(x), deriv), true;
    if (y.hi < kTinyX && a < kLimitShape) {
        const IB r = ibeta_tiny(b, a, dd::log_pos(y), deriv);
        return out = {r.ic, r.i, deriv ? r.xderiv * (x.hi / y.hi) : 0.0}, true;  // x^a y^(b-1) / B from x^(a-1) y^b / B
    }
    return false;
}

// I and 1 - I for a, b > 0 finite, 0 < x < 1, y = 1 - x exactly: TOMS 708
// BRATIO's choice. (a0, b0, x0, y0) is (a, b, x, y) or, `ind`, (b, a, y, x);
// w = I_x0(a0, b0) and w1 = 1 - w, one computed and the other 1 minus it.
inline IB ibeta(double a, double b, dd::DD x, dd::DD y, bool deriv = false) {
    if (!(x.hi >= kTinyX && y.hi >= kTinyX && a < kPlainShape && b < kPlainShape)) {
        IB r;
        if (ibeta_outside(a, b, x, y, deriv, r)) return r;
    }
    // The prefix (symmetric: the same for (b, a, y, x)), made once and only by the
    // branches that use it: bgrat and basym do not, and xderiv is the inverses' alone.
    std::optional<Scaled> made;
    auto prefix = [&] {
        if (!made) made = prefix_of(a, b, x, y);
        return *made;
    };
    double a0 = a, b0 = b;
    dd::DD x0 = x, y0 = y;
    bool ind = false;
    auto turn = [&] {
        std::swap(a0, b0);
        std::swap(x0, y0);
        ind = !ind;
    };
    double w, w1;
    auto direct = [&](double v) { w = v, w1 = 1.0 - v; };
    auto other = [&](double v) { w1 = v, w = 1.0 - v; };
    if (std::min(a, b) <= 1.0) {
        // Both shapes below 2^-700: I_x(a, b) = b / (a + b) + a b / (a + b) log(x / y) + ..., the
        // second term below 2^-680 of the first at any x, y a double can hold. (The series'
        // prefix and a B(a, b) are products of the two shapes: with a subnormal one they lost
        // their bits or vanished, betainc(5e-324, 5e-324, x) was 1 for 1/2.)
        if (std::max(a, b) < 0x1p-700) {
            const double sum = a + b;
            return {b / sum, a / sum, deriv ? a * (b / sum) / y.hi : 0.0};
        }
        if (x.hi > 0.5) turn();
        if (std::max(a0, b0) <= 1.0) {
            // BPSER for I where BRATIO takes it, kept while I <= 1/2: above that 1 - I is the
            // small one and would carry I / (1 - I) times the series' error (up to 9 times,
            // at x^a = 0.9), so it is computed itself, by BRATIO's other two ways. (For
            // a0 >= b0, I <= 1/2 at any x0 <= 1/2: the density's mass leans to the right.)
            // The series is at least 1 (b0 <= 1: every term is positive), so its factor
            // x^a / (a B(a, b)) alone is a lower bound of I: where that is already above 1/2
            // (by 2^-40, far more than its rounding) the series is not summed.
            w = 1.0;
            if (a0 >= std::min(0.2, b0) || std::pow(x0.hi, a0) <= 0.9) {
                const Scaled pref = prefix();
                const double power = pref.m == 0.0 ? 0.0 : bpser_power(b0, y0);
                // (the bound in plain doubles, 2^e from its bits: only a prefix near 1 can pass)
                bool above = false;
                if (a0 < b0 && pref.e > -64 && pref.e <= 0) {
                    const std::uint64_t bits = static_cast<std::uint64_t>(1023 + pref.e) << 52;
                    double two_e;
                    std::memcpy(&two_e, &bits, sizeof two_e);
                    above = pref.m * two_e * power > (0.5 + 0x1p-40) * a0;
                }
                if (!above) direct(pref.m == 0.0 ? 0.0 : value(pref * scaled(power * bpser_series(a0, b0, x0) / a0)));
            }
            if (w > 0.5 && a0 < b0) {
                if (x0.hi >= 0.3) other(bpser(b0, a0, y0, x0, prefix()));
                else other(bgrat(b0 + 20.0, a0, y0, x0, bup(b0, a0, y0, x0, 20)));
            }
        } else if (b0 <= 1.0) {
            direct(bpser(a0, b0, x0, y0, prefix()));
        } else if (x0.hi >= 0.3) {
            other(bpser(b0, a0, y0, x0, prefix()));
        } else if (x0.hi < 0.1 && std::pow(x0.hi * b0, a0) <= 0.7) {
            direct(bpser(a0, b0, x0, y0, prefix()));
        } else if (b0 > 15.0) {
            other(bgrat(b0, a0, y0, x0, 0.0));
        } else {
            other(bgrat(b0 + 20.0, a0, y0, x0, bup(b0, a0, y0, x0, 20)));
        }
    } else {
        double lambda = a > b ? (a + b) * y.hi - b : a - (a + b) * x.hi;
        if (lambda < 0.0) {
            turn();
            lambda = -lambda;
        }
        if (b0 < 40.0 && b0 * x0.hi <= 0.7) {
            direct(bpser(a0, b0, x0, y0, prefix()));
        } else if (std::min(a0, b0) > 100.0 && lambda <= 0.03 * std::min(a0, b0)) {
            direct(basym(a0, b0, beta_exponent(a0, b0, x0, y0)));
        } else {
            direct(bfrac(a0, b0, x0, y0, prefix()));
        }
    }
    if (ind) std::swap(w, w1);
    return {w, w1, deriv ? value(prefix()) / y.hi : 0.0};
}
inline IB ibeta(double a, double b, double x, bool deriv = false) {
    return ibeta(a, b, dd::DD{x, 0.0}, dd::two_sum(1.0, -x), deriv);
}

// I, 1 - I and x dI/dx at a tiny argument x = e^lx < 2^-1000 (x itself may be below the
// doubles: only its log is used), for b 2^-1000 negligible. There
//     I_x(a, b) = x^a / (a B(a, b)) (1 + O(b x)),
// so the pair comes from the one at r = 2^-1000 (power_down), without a subtraction: for a
// tiny a, I is near 1 at every double (x^a = 0.93 at a = 1e-4, x = 1e-308) and 1 - I is
// the small one, which ibeta gives at r by BGRAT. This
// replaces, for these arguments, the series' prefix from log x (the double-double log of a
// subnormal x) and BGRAT's z = nu x, a subnormal product that had lost up to half its bits
// (NaN from its low part's correction at most of them).
inline IB ibeta_tiny(double a, double b, dd::DD lx, bool deriv) {
    if (!(lx.hi > -kInf))
        return {0.0, 1.0, 0.0};  // x = 0 to the caller (-inf, or the NaN double-double sums make of it)
    return power_down(ibeta(a, b, dd::DD{kTinyX, 0.0}, dd::DD{1.0, -kTinyX}, deriv), a, lx);
}

// scipy's conventions; returns true with `out` set on an edge.
//   NaN: any NaN, a < 0, b < 0, x outside [0, 1], a = b = 0, a = b = inf
//   x = 0 -> 0, x = 1 -> 1, then a = 0 -> 1, b = 0 -> 0, a = inf -> 0, b = inf -> 1
// (for 1 - I, the complements).
inline bool ibeta_edge(double a, double b, double x, bool upper, double& out) {
    double v;
    if (std::isnan(a) || std::isnan(b) || std::isnan(x) || a < 0.0 || b < 0.0 || x < 0.0 || x > 1.0 ||
        (a == 0.0 && b == 0.0) || (std::isinf(a) && std::isinf(b))) {
        out = kNaN;
        return true;
    }
    if (x == 0.0) v = 0.0;
    else if (x == 1.0) v = 1.0;
    else if (a == 0.0) v = 1.0;
    else if (b == 0.0) v = 0.0;
    else if (std::isinf(a)) v = 0.0;
    else if (std::isinf(b)) v = 1.0;
    else return false;
    out = upper ? 1.0 - v : v;
    return true;
}

// ---- inverses ---------------------------------------------------------------

// log(I_x(a, b) / t) and x I' / I for a lower tail I far below the normal doubles (the
// inverses' deep tail, t = e^lt), x below the mean: I = prefix S with the prefix's exponent
// E against lt in double-double (both ~ -740, their difference the residual) and S the O(1)
// factor of BPSER (b <= 1 or b x <= 0.7) or of BFRAC (a, b > 1). x I' / I = 1 / (y S).
// NaN where neither applies or it does not end.
inline std::array<double, 2> log_tail(double a, double b, dd::DD x, dd::DD y, dd::DD lt) {
    double S;
    if (b <= 1.0 || b * x.hi <= 0.7) S = bpser_power(b, y) * bpser_series(a, b, x) / a;
    else if (a > 1.0) S = 1.0 / bfrac_fraction(a, b, x, y);
    else return {kNaN, kNaN};
    if (!(S > 0.0)) return {kNaN, kNaN};
    const dd::DD lr = -beta_exponent(a, b, x, y) - lt;
    return {lr.hi + (lr.lo + std::log(beta_factor(a, b) * S)), 1.0 / (y.hi * S)};
}
// A target below this is solved on the logs (ibeta_inverse): the tail rounded to a double has
// only a few bits there.
inline constexpr double kDeepTail = 0x1p-900;

// AS 109's initial value for I_u(p, q) = t, t <= 1/2, as u and v = 1 - u
// each from its own closed form (so neither loses digits near 1).
struct UV {
    double u, v;
};
inline UV as109(double p, double q, double t) {
    double z;
    {
        double tt = t, o;
        Ndtri::chunk(&tt, &o, 1);
        z = -o;  // the upper deviate: z >= 0
    }
    const double lbeta = Betaln::f(p, q);
    if (p > 1.0 && q > 1.0) {
        const double r = (z * z - 3.0) / 6.0;
        const double s = 1.0 / (p + p - 1.0), u = 1.0 / (q + q - 1.0);
        const double h = 2.0 / (s + u);
        const double w = z * std::sqrt(h + r) / h - (u - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
        const double e = q * std::exp(w + w);
        if (std::isinf(e)) return {std::exp((std::log(t * p) + lbeta) / p), 1.0};
        return {p / (p + e), e / (p + e)};
    }
    // The two end approximations: I ~ x^p / (p B) near x = 0, and
    // 1 - I ~ (1 - x)^q / (q B) near x = 1. AS109 picks one by its own test,
    // but for p, q < 1 (a U-shaped density) the pick can be the wrong end and
    // give a value outside [0, 1) -- e.g. 1 - x = 5400 for p = 0.038,
    // q = 0.0058, t = 0.088 -- which used to be read as an underflowing root
    // (betaincinv returned 0 for 8% of a, b < 1; T1 review). A value outside
    // [0, 1) falls back to the other end, then to the middle.
    auto lower_end = [&] { return std::exp((std::log(t * p) + lbeta) / p); };  // x
    auto upper_end = [&] { return std::exp((std::log1p(-t) + std::log(q) + lbeta) / q); };  // 1 - x
    auto from_lower = [&](double x) -> std::optional<UV> {
        if (x >= 0.0 && x < 1.0) return UV{x, 1.0 - x};
        return std::nullopt;
    };
    // (v = 1 is no start either: it reads as a root x = 0 that underflows, and for a tiny q
    // the exponent above is log q + log B cancelling to rounding noise over q --
    // betaincinv(2, 1e-15, 1e-15) was 0 for 0.84)
    auto from_upper = [&](double v) -> std::optional<UV> {
        if (v > 0.0 && v < 1.0) return UV{1.0 - v, v};
        return std::nullopt;
    };
    const double r = q + q;
    double u = 1.0 / (9.0 * q);
    u = r * std::pow(1.0 - u + z * std::sqrt(u), 3);
    if (u <= 0.0) {
        if (auto g = from_upper(upper_end())) return *g;
        if (auto g = from_lower(lower_end())) return *g;
        return {0.5, 0.5};
    }
    u = (4.0 * p + r - 2.0) / u;
    if (u <= 1.0) {
        if (auto g = from_lower(lower_end())) return *g;
        if (auto g = from_upper(upper_end())) return *g;
        return {0.5, 0.5};
    }
    return {1.0 - 2.0 / (u + 1.0), 2.0 / (u + 1.0)};
}

// The x with I_x(a, b) = t (upper: 1 - I = t), 0 < t <= 1/2, a, b > 0 finite.
inline double ibeta_inverse(double a, double b, double t, bool upper) {
    // upper: 1 - I_x(a, b) = I_y(b, a) = t, a lower problem in y
    const double p = upper ? b : a, q = upper ? a : b;
    // One shape far beyond the other (limit_shapes): the gamma limit's root, I_u(p, q) =
    // P(p, q u) for a huge q and 1 - I_u(p, q) = P(q, p (1 - u)) for a huge p, to 2^-60.
    // (AS 109's start is 0 or NaN for these shapes.)
    if (limit_shapes(p, q)) {
        const double u = Gammaincinv::f(p, t) / q;
        return upper ? 1.0 - u : u;
    }
    if (limit_shapes(q, p)) {
        const double v = Gammainccinv::f(q, t) / p;
        return upper ? v : 1.0 - v;
    }
    UV g;
    if (std::min(p, q) >= kPlainShape) {
        // both_large's shapes: the normal limit's start, u = m + z sigma about the mean m =
        // p / (p + q), sigma = m sqrt((1 - m) / p) (no p + q, which may overflow)
        double tt = t, z;
        Ndtri::chunk(&tt, &z, 1);
        const double mu = 1.0 / (1.0 + q / p), mv = 1.0 / (1.0 + p / q);
        const double d = z * mu * std::sqrt(mv / p);
        g = {mu + d, mv - d};
        if (!(g.u > 0.0 && g.v > 0.0)) g = {mu, mv};
    } else {
        g = as109(p, q, t);  // I_u(p, q) = t: u is x (lower) or y (upper)
    }
    // iterate in whichever of u, v is below 1/2; s = the variable's parameters
    bool in_u = g.u <= 0.5;
    double sa = in_u ? p : q, sb = in_u ? q : p;
    // in v = 1 - u: I_v(q, p) = 1 - t, i.e. the upper function equals t
    bool inc = in_u;
    double w0 = in_u ? g.u : g.v;
    if (!(w0 > 0.0)) w0 = 0.0;
    if (w0 == 0.0) {  // the root underflows
        const double w = 0.0;
        return (in_u != upper) ? w : 1.0 - w;
    }
    w0 = std::min(w0, 0.5);
    auto solve = [&](double start) {
        return halley(
            start, t, inc,
            [&](double v) {
                const IB r = ibeta(sa, sb, v, true);
                const double h = (sa - 1.0) / v - (sb - 1.0) / (1.0 - v);
                return std::array<double, 3>{inc ? r.i : r.ic, inc ? r.xderiv : -r.xderiv, h};
            },
            0.0, 1.0
        );
    };
    double w = solve(w0);
    if (w > 0.5) {
        // The start was in the wrong half (AS 109 gave u = 3e-6 for 1 - 7e-5 at p = 1e7, q =
        // 2.5, t = 1e-300): the root is next to 1 in this variable, where 1 - w would round
        // (1.3e-13 of x there). Solved again in the other one, from 1 - w.
        in_u = !in_u;
        inc = in_u;
        std::swap(sa, sb);
        w = solve(w < 1.0 ? 1.0 - w : 0x1p-54);  // (w = 1: the root is below the spacing there)
    }
    double root = w;
    if (t < kDeepTail && w > 0.0 && w < 1.0) {
        // Newton on log F - log t from Halley's w (right to a few digits there): in log w
        // for the rising side (F = I_w(sa, sb) ~ w^sa), in w for the falling one (F =
        // I_(1-w)(sb, sa)). At most 100 steps; NaN if they do not settle.
        const dd::DD lt = dd::log_pos(t);
        double step = kInf;
        bool settled = false;
        for (int it = 0; it < 100; ++it) {
            const dd::DD xw{root, 0.0}, yw = dd::two_sum(1.0, -root);
            const auto [g, xd] = inc ? log_tail(sa, sb, xw, yw, lt) : log_tail(sb, sa, yw, xw, lt);
            if (std::isnan(g)) {
                // neither BPSER nor BFRAC (a large shape with the other <= 1 next to x = 1,
                // BGRAT's ground): Halley's root stands for a normal t, where the tail still
                // has its bits; for a subnormal t there is no value
                if (t < 0x1p-1022) return kNaN;
                root = w;
                settled = true;
                break;
            }
            double wn = inc ? root * std::exp(-g / xd) : root + g * (yw.hi / xd);
            if (inc && wn == 0.0) {  // the root underflows
                root = 0.0;
                settled = true;
                break;
            }
            if (!(wn > 0.0)) wn = 0.5 * root;
            if (!(wn < 1.0)) wn = 0.5 * (root + 1.0);
            const double moved = std::fabs(wn - root);
            step = moved / wn;
            root = wn;
            if (step <= 0x1p-53 || moved <= 0x1p-1073) {  // (or between two subnormal neighbours)
                settled = true;
                break;
            }
        }
        if (!settled && !(step <= 0x1p-48)) return kNaN;
    }
    // root is x when (in_u and lower) or (in v and upper); else it is y
    return (in_u != upper) ? root : 1.0 - root;
}

// ---- where Boost's ibeta and ibetac are taken -------------------------------------------
//
// Shapes at most 6 (special/boost.hpp, docs/design/special.md 5.14), except, all measured
// against Arb (bench/stats/opt/f):
//   - a subnormal x: Boost's value is off by up to 25% there (ours: ibeta_tiny);
//   - a result below 2^-960: Boost loses digits below 2^-1016 (2e-13 there, an
//     intermediate in the subnormals; 2.5e-15 from 2^-1016 up over 3e5 points). The floor
//     is the one sd.stats uses for the same reason (beta_detail::kDeep), 56 binades above
//     the measured edge;
//   - a, b <= 1 and the result the one Boost forms as 1 - (BPSER's series) with the series
//     well above 1/2 (BRATIO's rule for BPSER, which Boost follows: the header's first
//     point). Known before any evaluation (boost_subtracts): with (a0, b0, x0) the
//     parameters turned so that x0 <= 1/2, Boost sums the series for I_x0(a0, b0) when
//     x0^a0 <= 0.9 (or a0 >= min(0.2, b0)) and gives the other side as 1 minus it when
//     x0 < 0.02 (from 0.02 up, where the series passes 1/2, by a continued fraction
//     instead: 1.7e-15 at most over 1.4e4 such points). For a0 >= b0 the series stays
//     below 1/2. Else I_x0(a0, b0) <= x0^a0 / y0 (the integrand's (1 - t)^(b0 - 1) is at
//     most 1 / y0, and 1 / (a0 B(a0, b0)) <= 1 for b0 <= 1), so x0^a0 <= 0.6 bounds the
//     series by 0.6 / 0.98 = 0.62 and the subtraction's error by 1.6 times the series':
//     measured 1.2e-15 at most there (1.9e4 points), against 6.8e-15 for x0^a0 in
//     (0.6, 0.9], where the side is ibeta's (1.3e-15 on those 1.9e4 points).
inline constexpr double kBoostFloor = 0x1p-960;
// (not with both shapes below 2^-700, ibeta's b / (a + b): Boost gave 1 for 1/2 at two subnormal ones)
inline bool boost_region(double a, double b, double x) {
    return a <= 6.0 && b <= 6.0 && x >= 0x1p-1022 && (a >= 0x1p-700 || b >= 0x1p-700);
}
// The side asked (upper: 1 - I) is the one Boost would form by subtraction from a series above ~0.6.
inline bool boost_subtracts(double a, double b, double x, bool upper) {
    if (!(a <= 1.0 && b <= 1.0) || (x > 0.5) == upper) return false;  // BPSER's own side: x <= 1/2 lower, else upper
    const double a0 = upper ? a : b, b0 = upper ? b : a, x0 = upper ? x : 1.0 - x;  // (1 - x exact: x > 1/2)
    if (!(a0 < b0 && x0 < 0.02 && a0 < 0.14)) return false;  // (0.02^a0 <= 0.6 from a0 = 0.131 up: no pow needed)
    const double p = std::pow(x0, a0);
    return p > 0.6 && p <= 0.9;  // (above 0.9, with a0 < 0.2 here, Boost computes this side itself)
}
inline bool boost_kept(double r) { return r >= kBoostFloor && r <= 1.0; }

}  // namespace ibeta_detail

struct Betainc : Real {
    static constexpr const char* name = "betainc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"a", "b", "x"};
    static constexpr double cost_ns = cost::kBetainc;
    static double f(double a, double b, double x) {
        double out;
        if (ibeta_detail::ibeta_edge(a, b, x, false, out)) return out;
        if (ibeta_detail::boost_region(a, b, x) && !ibeta_detail::boost_subtracts(a, b, x, false)) {
            const double r = bst::ibeta(a, b, x);
            if (ibeta_detail::boost_kept(r)) return r;
        }
        return ibeta_detail::ibeta(a, b, x).i;
    }
};

struct Betaincc : Real {
    static constexpr const char* name = "betaincc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"a", "b", "x"};
    static constexpr double cost_ns = cost::kBetaincc;
    static double f(double a, double b, double x) {
        double out;
        if (ibeta_detail::ibeta_edge(a, b, x, true, out)) return out;
        if (ibeta_detail::boost_region(a, b, x) && !ibeta_detail::boost_subtracts(a, b, x, true)) {
            const double r = bst::ibetac(a, b, x);
            if (ibeta_detail::boost_kept(r)) return r;
        }
        return ibeta_detail::ibeta(a, b, x).ic;
    }
};

// scipy's ends: p = 0 -> 0, p = 1 -> 1 (betainccinv: the other way), also for
// an infinite a or b; NaN for a or b not > 0, NaN inputs, p outside [0, 1],
// and an infinite a or b inside (0, 1).
inline bool ibeta_inverse_edge(double a, double b, double p, bool upper, double& out) {
    if (std::isnan(a) || std::isnan(b) || std::isnan(p) || !(a > 0.0) || !(b > 0.0) || p < 0.0 || p > 1.0) {
        out = kNaN;
    } else if (p == 0.0) {
        out = upper ? 1.0 : 0.0;
    } else if (p == 1.0) {
        out = upper ? 0.0 : 1.0;
    } else if (std::isinf(a) || std::isinf(b)) {
        out = kNaN;
    } else {
        return false;
    }
    return true;
}

// Boost's inverses are taken for shapes up to here (measured to 1e5; at b = 1e200 they gave
// the root of another probability, 12.96 / b for 15.19 / b at a = 14.38, p = 0.617).
inline constexpr double kBoostInverseShape = 1e6;

// The smaller of p, 1 - p (exact) against its own function.
inline double ibeta_inverse_dispatch(double a, double b, double p, bool upper) {
    if (p <= 0.5) return ibeta_detail::ibeta_inverse(a, b, p, upper);
    return ibeta_detail::ibeta_inverse(a, b, 1.0 - p, !upper);
}

struct Betaincinv : Real {
    static constexpr const char* name = "betaincinv";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"a", "b", "p"};
    static constexpr double cost_ns = cost::kBetaincinv;
    static double f(double a, double b, double p) {
        double out;
        if (ibeta_inverse_edge(a, b, p, false, out)) return out;
        // Boost (special/boost.hpp); not for a subnormal-range p (27 times the root at a = 10,
        // b = 14.38, p = 5e-324)
        if (a >= 10.0 && b >= 10.0 && a <= kBoostInverseShape && b <= kBoostInverseShape && p >= 1e-300) {
            const double r = bst::ibeta_inv(a, b, p);
            if (bst::normal(r)) return r;
        }
        return ibeta_inverse_dispatch(a, b, p, false);
    }
};

struct Betainccinv : Real {
    static constexpr const char* name = "betainccinv";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"a", "b", "q"};
    static constexpr double cost_ns = cost::kBetainccinv;
    static double f(double a, double b, double q) {
        double out;
        if (ibeta_inverse_edge(a, b, q, true, out)) return out;
        if (a >= 1.0 && b >= 1.0 && a <= kBoostInverseShape && b <= kBoostInverseShape &&
            q >= 1e-300) {  // Boost (special/boost.hpp)
            const double r = bst::ibetac_inv(a, b, q);
            if (bst::normal(r)) return r;
        }
        return ibeta_inverse_dispatch(a, b, q, true);
    }
};

}  // namespace sd::special
