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
#include <optional>
#include <utility>

namespace sd::special {

namespace ibeta_detail {

// I, 1 - I and x dI/dx = x^a y^(b-1) / B(a, b) (= B / y: no overflow for small x).
struct IB {
    double i, ic, xderiv;
};

inline Scaled prefix_of(double a, double b, dd::DD x, dd::DD y) {
    return beta_prefix(a, b, beta_exponent(a, b, x, y));
}

// BPSER: I_x(a, b) = (B y^-b / a) sum_n (1-b)_n / n! a/(a+n) x^n, B the
// prefix of (a, b) at x. The low part of x enters to first order (dS/dx).
inline double bpser(double a, double b, dd::DD x, dd::DD y, Scaled pref) {
    if (pref.m == 0.0) return 0.0;
    const double xh = x.hi;
    double s = 1.0, sd = 0.0, u = 1.0;
    for (int n = 1; n < 100000; ++n) {
        u *= (n - b) * xh / n;
        const double term = u * (a / (a + n));
        s += term;
        sd += n * term;
        if (std::fabs(term) <= std::fabs(s) * 0x1p-56) break;
    }
    s += x.lo * (sd / xh);
    const dd::DD e = dd::log(y) * b;  // y^-b = e^-(b log y)
    return value(pref * scaled(std::exp(-e.hi) * (1.0 - e.lo) * s / a));
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
inline double bgrat(double a, double b, dd::DD x, dd::DD y, double w) {
    const double bm1 = b - 1.0;
    const dd::DD nu = dd::two_sum(a, 0.5 * bm1);
    // log x from the small one of the pair: x near 1 carries only the double-double's
    // absolute accuracy (~1e-32) in its distance from 1, which is y's whole value when x
    // came from a division (fdtr's w = 1 - 4e-26: z = -nu log x off by 2.5e-7, the result
    // by 1.5e-9). log x = log(1 - y) = -(y + mu(-y)).
    const dd::DD lnx = y.hi < 0.5 ? -(y + dd::log1pmx_neg(-y)) : dd::log(x);
    const dd::DD z = -(nu * lnx);
    if (!(z.hi > 0.0)) return w;
    // Q(b, z) and r at z = hi + lo, to first order in lo
    const igam_detail::PQ q = igam_detail::igam(b, z.hi);
    const double Q = q.q - z.lo * (q.xderiv / z.hi);
    dd::DD Ez = gamma_exponent(b, z.hi);
    Ez = Ez + (1.0 - b / z.hi) * z.lo;
    const double r = b * value(gamma_prefix(b, Ez));
    if (Q == 0.0 && r == 0.0) return w;
    // G = sqrt(a/c) Gamma*(c) / Gamma*(a) e^(-a mu(b/a) + b log1p((b+1) / (2 nu))), c = a + b
    const double c = a + b;
    const double mu = dd::log1pmx_neg(dd::DD{b / a, 0.0}).hi;
    const double G = std::sqrt(a / c) * gamma_star(c) / gamma_star(a) *
                     std::exp(-a * mu + b * std::log1p((b + 1.0) / (2.0 * nu.hi)));
    const double v = 0.25 / (nu.hi * nu.hi);
    const double t2 = 0.25 * lnx.hi * lnx.hi;
    double cs[31], ds[31];
    double sum = Q, K = Q, t = r, cn = 1.0, n2 = 0.0;
    const double tail = w / G;
    for (int n = 1; n <= 30; ++n) {
        const double bp2n = b + n2;
        K = (bp2n * (bp2n + 1.0) * K + (z.hi + bp2n + 1.0) * t) * v;
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
    return w + G * sum;
}

// BFRAC (TOMS 708 bfrac; Boost's ibeta_fraction2), x at most the mean:
//     I_x(a, b) = B / (b0 + a1 / (b1 + a2 / (b2 + ...)))
//     a_m = (a+m-1)(a+b+m-1) m (b-m) x^2 / (a+2m-1)^2
//     b_m = m + m (b-m) x / (a+2m-1) + (a+m)(a y - b x + 1 + m (2 - x)) / (a+2m+1)
// No correction for x's low part (x = 1 - y when the roles are turned): the
// fraction takes y, the small and exact one, in a y - b x + 1; measured, the
// error is ~5 ULP without the first-order term and 3300 ULP with it.
inline double bfrac(double a, double b, dd::DD x, dd::DD y, Scaled pref) {
    if (pref.m == 0.0) return 0.0;
    const double xh = x.hi, yh = y.hi;
    auto gen = [&](int m) {
        const double md = m;
        const double den = a + 2.0 * md - 1.0;
        // (... x)(... x): with b = 1e300 the products of the parameters overflow and x^2 underflows
        const double an = ((a + md - 1.0) * (a + b + md - 1.0) * xh) * (md * (b - md) * xh) / (den * den);
        const double bn = md + md * (b - md) * xh / den +
                          (a + md) * (a * yh - b * xh + 1.0 + md * (2.0 - xh)) / (a + 2.0 * md + 1.0);
        return std::pair{an, bn};
    };
    const CfResult r = lentz_backward(gen(0).second, gen, 100000);
    return value(pref / scaled(r.value));
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

// I and 1 - I for a, b > 0 finite, 0 < x < 1, y = 1 - x exactly: TOMS 708
// BRATIO's choice. (a0, b0, x0, y0) is (a, b, x, y) or, `ind`, (b, a, y, x);
// w = I_x0(a0, b0) and w1 = 1 - w, one computed and the other 1 minus it.
inline IB ibeta(double a, double b, dd::DD x, dd::DD y, bool deriv = false) {
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
        if (x.hi > 0.5) turn();
        if (std::max(a0, b0) <= 1.0) {
            if (a0 >= std::min(0.2, b0) || std::pow(x0.hi, a0) <= 0.9) direct(bpser(a0, b0, x0, y0, prefix()));
            else if (x0.hi >= 0.3) other(bpser(b0, a0, y0, x0, prefix()));
            else other(bgrat(b0 + 20.0, a0, y0, x0, bup(b0, a0, y0, x0, 20)));
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
    auto lower_end = [&] { return std::exp((std::log(t * p) + lbeta) / p); };              // x
    auto upper_end = [&] { return std::exp((std::log1p(-t) + std::log(q) + lbeta) / q); };  // 1 - x
    auto from_lower = [&](double x) -> std::optional<UV> {
        if (x >= 0.0 && x < 1.0) return UV{x, 1.0 - x};
        return std::nullopt;
    };
    auto from_upper = [&](double v) -> std::optional<UV> {
        if (v > 0.0 && v <= 1.0) return UV{1.0 - v, v};
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
    const UV g = as109(p, q, t);  // I_u(p, q) = t: u is x (lower) or y (upper)
    // iterate in whichever of u, v is below 1/2; s = the variable's parameters
    const bool in_u = g.u <= 0.5;
    const double sa = in_u ? p : q, sb = in_u ? q : p;
    // in v = 1 - u: I_v(q, p) = 1 - t, i.e. the upper function equals t
    const bool inc = in_u;
    double w0 = in_u ? g.u : g.v;
    if (!(w0 > 0.0)) w0 = 0.0;
    if (w0 == 0.0) {  // the root underflows
        const double w = 0.0;
        return (in_u != upper) ? w : 1.0 - w;
    }
    w0 = std::min(w0, 0.5);
    const double w = halley(
        w0, t, inc,
        [&](double v) {
            const IB r = ibeta(sa, sb, v, true);
            const double h = (sa - 1.0) / v - (sb - 1.0) / (1.0 - v);
            return std::array<double, 3>{inc ? r.i : r.ic, inc ? r.xderiv : -r.xderiv, h};
        },
        0.0, 1.0);
    // w is x when (in_u and lower) or (in v and upper); else w is y
    return (in_u != upper) ? w : 1.0 - w;
}

}  // namespace ibeta_detail

struct Betainc : Real {
    static constexpr const char* name = "betainc";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"a", "b", "x"};
    static constexpr double cost_ns = cost::kBetainc;
    static double f(double a, double b, double x) {
        double out;
        if (ibeta_detail::ibeta_edge(a, b, x, false, out)) return out;
        if (a <= 6.0 && b <= 6.0) {  // Boost (special/boost.hpp; the same region as betaincc, so
                                     // the pair still adds to 1); subnormal tails are ours
            const double r = bst::ibeta(a, b, x);
            if (bst::normal(r)) return r;
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
        if (a <= 6.0 && b <= 6.0) {  // Boost (special/boost.hpp); subnormal tails are ours
            const double r = bst::ibetac(a, b, x);
            if (bst::normal(r)) return r;
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
        if (a >= 10.0 && b >= 10.0) {  // Boost (special/boost.hpp)
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
        if (a >= 1.0 && b >= 1.0) {  // Boost (special/boost.hpp)
            const double r = bst::ibetac_inv(a, b, q);
            if (bst::normal(r)) return r;
        }
        return ibeta_inverse_dispatch(a, b, q, true);
    }
};

}  // namespace sd::special
