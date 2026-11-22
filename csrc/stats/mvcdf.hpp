// The cumulative distribution functions of the multivariate normal and t
// (T9; docs/design/stats.md §6.2, decisions T16 and T29): P(X <= x) with every
// lower limit -inf, for X ~ N(0, Sigma) or X = Z / sqrt(S / nu) (Z ~ N(0, Sigma),
// S ~ chi^2_nu independent). The caller has subtracted the mean.
//
//   normal, d = 1   Phi (sd.special's ndtr)
//   normal, d = 2   Plackett's identity integrated along the correlation (bvn below)
//   normal, d = 3   Plackett's identity along a path of correlations, or
//                   conditioning on one variable where that cancels (tvn below)
//   normal, d >= 4  Genz's quasi-Monte Carlo (qmc below)
//   t,      d = 1   sd.special's stdtr
//   t,      d = 2   Plackett's identity mixed over the chi variable: a closed-form
//                   integrand, plus a 1-D mixture of Phi's (bvt below)
//   t,      d = 3   the scale mixture E[Phi_3(R h)], one more 1-D quadrature
//                   around tvn (tvt below)
//   t,      d >= 4  Genz's quasi-Monte Carlo with the chi variable
//
// The exact formulas (d <= 3) are written so that the sums they form do not
// cancel: every integrand has a fixed sign, and the starting point of each
// integral is chosen so that the integral adds to it. That keeps the relative
// accuracy in the lower tails, where Genz's TVPACK (bvnd, tvnd) reaches ~1e-15
// absolute only. Measured (bench/stats/t9/cdf_accuracy.md at commit 4151d2a): the relative error is
// within a few units of eps times the problem's condition number (the sum of
// |d log P / d log x| over the inputs), which is ~ -log P in the tails.
#pragma once

#include "array/kernels/vmath.hpp"
#include "array/ops/pieces.hpp"
#include "parallel/pool.hpp"
#include "random/philox.hpp"
#include "special/dd.hpp"
#include "special/ops/distr.hpp"
#include "special/ops/erf.hpp"
#include "special/ops/incgamma.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace sd::stats::mvcdf {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kPi = 3.141592653589793238;
inline constexpr double kHalfPi = 1.570796326794896619;
inline constexpr double kTwoPi = 6.283185307179586477;
inline constexpr double kSqrt2OverPi = 0.7978845608028653559;  // sqrt(2 / pi)
inline constexpr double kSqrtHalf = 0.7071067811865475244;

inline double Phi(double x) { return special::distr_detail::one<special::Ndtr>(x); }

// min(1, v) for sums that may be NaN (std::min(1.0, NaN) is 1.0).
inline double cap1(double v) { return v > 1.0 ? 1.0 : v; }

// phi(t) / Phi(t) (the inverse Mills ratio of the lower tail) without underflow:
// sqrt(2 / pi) / erfcx(-t / sqrt 2), since Phi(t) = erfc(-t / sqrt 2) / 2.
inline double mills_lower(double t) { return kSqrt2OverPi / special::Erfcx::f(-t * kSqrtHalf); }

// ------------------------------------------------- adaptive Gauss-Kronrod

// QUADPACK's 21-point Kronrod rule and its embedded 10-point Gauss rule (qk21;
// Piessens, de Doncker, Ueberhuber, Kahaner, "QUADPACK", 1983): the nodes in
// (0, 1) from the outside in, then 0; the Gauss nodes are xgk[1], xgk[3], ..., xgk[9].
namespace gk {
// clang-format off
inline constexpr double xgk[11] = {
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452, 0.930157491355708226001207180059508,
    0.865063366688984510732096688423493, 0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784, 0.294392862701460198131126603103866,
    0.148874338981631210884826001129720, 0.0};
inline constexpr double wgk[11] = {
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390, 0.054755896574351996031381300244580,
    0.075039674810919952767043140916190, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077208980632301, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717,
    0.147739104901338491374841515972068, 0.149445554002916905664936468389821};
inline constexpr double wg[5] = {0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
                                 0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
                                 0.295524224714752870173892994651244};
// clang-format on
}  // namespace gk

struct Panel {
    double a, b, val, err;
};

// One panel [a, b]: f(nodes, values, 21) evaluates the integrand at all nodes at
// once (so it can call the chunked special functions). The error is QUADPACK's
// estimate resasc min(1, (200 |K - G| / resasc)^1.5) without its round-off
// floor (50 eps resabs): that floor would stop the refinement at ~1e-14, and the
// integrands here are smooth enough for the estimate to keep meaning below it.
template <class F>
Panel gk21(F& f, double a, double b) {
    const double c = 0.5 * (a + b), hl = 0.5 * (b - a);
    double t[21], v[21];
    t[0] = c;
    for (int j = 0; j < 10; ++j) t[1 + 2 * j] = c - hl * gk::xgk[j], t[2 + 2 * j] = c + hl * gk::xgk[j];
    f(t, v, 21);
    double resk = gk::wgk[10] * v[0], resg = 0.0;
    for (int j = 0; j < 10; ++j) {
        const double s = v[1 + 2 * j] + v[2 + 2 * j];
        resk += gk::wgk[j] * s;
        if (j & 1) resg += gk::wg[j >> 1] * s;
    }
    const double mean = 0.5 * resk;
    double resasc = gk::wgk[10] * std::fabs(v[0] - mean);
    for (int j = 0; j < 10; ++j)
        resasc += gk::wgk[j] * (std::fabs(v[1 + 2 * j] - mean) + std::fabs(v[2 + 2 * j] - mean));
    const double ah = std::fabs(hl);
    double err = std::fabs(resk - resg) * ah;
    resasc *= ah;
    if (resasc != 0.0 && err != 0.0) err = resasc * std::min(1.0, std::pow(200.0 * err / resasc, 1.5));
    return {a, b, resk * hl, err};
}

inline constexpr double kQuadRel = 1e-16;  // the integrals' target, relative to base + |integral|
inline constexpr int kMaxPanels = 200;

// adapt: the integral of f over [a, b] (0 <= a < b, f of one sign) from the first
// panels p[0, n), refined at the panel
// with the largest error estimate until the estimates sum to kQuadRel (|base| +
// |integral|): base is what the integral is added to, so the sum is accurate to
// ~1e-16 of its terms. Panels are bisected in a fixed order: the result depends on
// the inputs alone.
template <class F>
double adapt(F& f, Panel* p, int n, double base, int cap = kMaxPanels) {
    for (;;) {  // at most cap splits: every pass that does not return adds a panel
        double v = 0.0, e = 0.0;
        int worst = 0;
        for (int i = 0; i < n; ++i) {
            v += p[i].val, e += p[i].err;
            if (p[i].err > p[worst].err) worst = i;
        }
        if (!(e > kQuadRel * (std::fabs(base) + std::fabs(v))) || n >= cap) return v;
        const double lo = p[worst].a, hi = p[worst].b, m = 0.5 * (lo + hi);
        if (!(lo < m && m < hi)) return v;  // the panel cannot be split further
        p[worst] = gk21(f, lo, m);
        p[n++] = gk21(f, m, hi);
    }
}

// The integral over [a, b] from first panels cut at the sorted breakpoints in (a, b).
template <class F>
double integrate_at(F&& f, double a, double b, const double* cuts, int ncuts, double base) {
    Panel p[kMaxPanels];
    int n = 0;
    double lo = a;
    for (int i = 0; i < ncuts && n < kMaxPanels / 2; ++i)
        if (cuts[i] > lo && cuts[i] < b) p[n++] = gk21(f, lo, cuts[i]), lo = cuts[i];
    p[n++] = gk21(f, lo, b);
    return adapt(f, p, n, base);
}

// ---------------------------------------------------------------- bivariate

// The path variable. Plackett's identity (Plackett 1954; Drezner and Wesolowsky
// 1990, "On the computation of the bivariate normal integral"; Genz 2004,
// "Numerical computation of rectangular bivariate and trivariate normal and t
// probabilities", Stat. Comput. 14) is
//     d Phi2(h, k; r) / dr = exp(-q(r)) / (2 pi sqrt(1 - r^2)),
//     q(r) = (h^2 - 2 r h k + k^2) / (2 (1 - r^2)),
// and r = sin(theta) removes the square root: d Phi2 / d theta = exp(-q) / (2 pi).
// The integrals below run in t = sqrt(1 - |s|), the distance from the nearer pole:
// s = 1 - t^2 (s >= 0) or t^2 - 1 (s <= 0), t in [0, 1], d theta = 2 dt / sqrt(2 - t^2)
// (smooth: no square root singularity at the pole), and 1 - s, 1 + s are t^2 and
// 2 - t^2, both accurate even where s is within 1e-16 of +-1.
//
// In the tails the integrand peaks at the end of the path, s = r, t = T = sqrt(1 -+ r),
// with slope |d log f / dt| ~ q (hundreds), and the probability's relative condition
// number in r is about as large. Two roundings would each move r by an ulp there:
//   * the end itself: T is known as hi + lo (dd_sqrt, from the exact 1 -+ r), and the
//     integral to T is the one to hi plus f(hi) lo;
//   * the nodes: a node c + x hl near t ~ 0.5 is off by up to 2^-54, all of a panel's
//     nodes by the same rounding of c (one panel at the end, slope 600, lost 3.6e-14).
//     So the nodes are u = |t - T.hi| >= 0, measured from the end (small where the
//     integrand is large, and so exact to their own size), and t^2 = (T.hi +- u)^2 is
//     formed as hi + lo from u.
namespace dd = special::dd;
using dd::DD;

// sqrt(x) for x = hi + lo >= 0, as hi + lo (one Newton step from the exact remainder).
inline DD dd_sqrt(DD x) {
    if (!(x.hi > 0.0)) return {0.0, 0.0};
    const double r = std::sqrt(x.hi);
    return dd::quick_two_sum(r, (std::fma(-r, r, x.hi) + x.lo) / (2.0 * r));
}

// The integral over t in [ta, tb] (0 <= ta <= tb <= 1, exact doubles) of f, given t^2 as
// hi + lo (f(t2, n, out): n values), added to base. The nodes are u = t - ta (from_low)
// or tb - t.
//
// delta is the width of a step at the pole t = 0: the normal's integrands behave there
// like exp(-delta^2 / (4 t^2)), 0 below t ~ delta and smooth above, varying on the scale
// of t itself. A first panel whose nodes all lie above the step cannot see it (it lost
// 2e-8 relative at h + k = 1e-8), so the first panels are geometric in t from
// max(ta, delta): [ta, c], [c, 4c], [4c, 16c], ..., each resolving the scale of its t. A
// step below 1e-18 wide holds less than 1e-18 of the integral and is ignored. The t's
// integrands go like t^(2a) below the step instead, not smooth at 0 for small a:
// pole_levels more geometric panels below it.
template <class F>
double path_integral(F&& f, double ta, double tb, bool from_low, double base, double delta, int pole_levels = 0) {
    if (!(tb > ta)) return 0.0;
    const double T = from_low ? ta : tb, dir = from_low ? 1.0 : -1.0;
    const DD TT = dd::two_prod(T, T);
    const DD L = dd::two_sum(tb, -ta);  // the u range, as hi + lo
    auto g = [&](const double* u, double* o, int n) {
        DD t2[21];
        for (int i = 0; i < n; ++i) t2[i] = TT + dd::two_prod(2.0 * dir * T, u[i]) + dd::two_prod(u[i], u[i]);
        f(t2, n, o);
    };
    // the cuts in t, as integrate places them, then in u
    double tc[64];
    int nt = 0;
    double c = delta >= 1e-18 && delta < tb ? std::max(ta, delta) : ta;
    if (pole_levels > 0 && ta == 0.0 && delta >= 1e-18) {
        c = std::min(delta, tb);
        for (int l = pole_levels; l >= 1; --l) tc[nt++] = std::ldexp(c, -2 * l);
    }
    if (c > ta && c < tb) tc[nt++] = c;
    while (c > 0.0 && 4.0 * c < tb && nt < 60) tc[nt++] = c *= 4.0;
    double uc[64];
    int nu = 0;
    for (int i = 0; i < nt; ++i)
        if (tc[i] > ta && tc[i] < tb) uc[nu++] = from_low ? tc[i] - ta : tb - tc[i];
    std::sort(uc, uc + nu);
    const double J = integrate_at(g, 0.0, L.hi, uc, nu, base);
    if (L.lo == 0.0) return J;
    double uf[1] = {L.hi}, vf[1];
    g(uf, vf, 1);
    return J + vf[0] * L.lo;
}

// The part of the integral between T.hi and the true end T.hi + T.lo: f(T.hi) T.lo.
template <class F>
double end_part(F&& f, DD T) {
    if (T.lo == 0.0) return 0.0;
    const DD t2[1] = {dd::two_prod(T.hi, T.hi)};
    double v[1];
    f(t2, 1, v);
    return v[0] * T.lo;
}

struct Sides {
    double s, sm, sp, jac;  // s, 1 - s, 1 + s, d theta / dt
};
inline Sides sides(DD t2, bool minus) {
    const double near = t2.hi, far = (DD{2.0, 0.0} - t2).hi, jac = 2.0 / std::sqrt(far);
    const double s = (DD{1.0, 0.0} - t2).hi;
    return minus ? Sides{-s, far, near, jac} : Sides{s, near, far, jac};
}

// q of (h, k) at correlation s, as a sum of two nonnegative terms (no cancellation):
//     hk >= 0:  (h - k)^2 / (2 (1 - s^2)) + hk / (1 + s)
//     hk <  0:  (h + k)^2 / (2 (1 - s^2)) - hk / (1 - s)
// (from h^2 + k^2 - 2hks = (h -+ k)^2 +- 2hk (1 -+ s)). A zero numerator stays 0 at
// the poles, where the true limit is finite.
inline double qform(double h, double k, double sm, double sp) {
    const double hk = h * k;
    if (hk >= 0.0) {
        const double d = h - k, t = d * d;
        return (t == 0.0 ? 0.0 : t / (2.0 * sm * sp)) + (hk == 0.0 ? 0.0 : hk / sp);
    }
    const double d = h + k, t = d * d;
    return (t == 0.0 ? 0.0 : t / (2.0 * sm * sp)) - hk / sm;
}

// P(X <= h, Y <= k) at correlation -1 (Y = -X): P(-k <= X <= h), each case by the
// tails it is made of.
inline double bvn_minus_one(double h, double k) {
    if (!(h > -k)) return 0.0;
    if (h <= 0.0) return Phi(h) - Phi(-k);    // -k < h <= 0: two lower tails
    if (k <= 0.0) return Phi(k) - Phi(-h);    // 0 <= -k < h: two upper tails
    return (1.0 - Phi(-h)) - Phi(-k);         // -k < 0 < h
}

// The integral of exp(-q) dtheta on the given side over t from the end T (hi + lo) to the
// pole (to_pole) or to s = 0, added to base.
inline double bvn_integral(double h, double k, bool minus, DD T, bool to_pole, double base) {
    auto f = [&](const DD* t2, int n, double* o) {
        for (int i = 0; i < n; ++i) {
            const Sides s = sides(t2[i], minus);
            o[i] = s.jac * std::exp(-qform(h, k, s.sm, s.sp));
        }
    };
    // the step at the pole t = 0: s -> 1 (plus side) has width |h - k| / 2, s -> -1 |h + k| / 2
    const double delta = 0.5 * std::fabs(minus ? h + k : h - k);
    return to_pole ? path_integral(f, 0.0, T.hi, false, base * kTwoPi, delta) + end_part(f, T)
                   : path_integral(f, T.hi, 1.0, true, base * kTwoPi, delta) - end_part(f, T);
}

// Below this r < 0 the integral starts at r = -1 directly (the start at 0 cancels).
inline constexpr double kBvnSwitch = -0.5;

// t = sqrt(1 - |r|) for r = hi + lo, as hi + lo (0 at |r| >= 1).
inline DD path_end(DD r) {
    return dd_sqrt(r.hi > 0.0 ? dd::two_sum(1.0, -r.hi) - r.lo : dd::two_sum(1.0, r.hi) + r.lo);
}

// Phi2(h, k; r) = P(X <= h, Y <= k), unit variances, finite h and k, |r| <= 1, for r
// given as hi + lo (tvn's partial correlation; a double r is {r, 0}).
//     r >= 0:  Phi(h) Phi(k) + (1 / 2pi) int_0^asin r exp(-q) dtheta
//              (both terms >= 0)
//     r <  0:  Phi(h) Phi(k) - (1 / 2pi) int_asin r^0 exp(-q) dtheta
//              when that loses at most two bits (r > -1/2 only), else
//              Phi2(h, k; -1) + (1 / 2pi) int_-pi/2^asin r exp(-q) dtheta
//              (both terms >= 0 again)
inline double bvn(double h, double k, DD r) {
    if (std::isnan(h) || std::isnan(k) || std::isnan(r.hi)) return kNaN;
    const double ph = Phi(h), pk = Phi(k);
    if (r.hi == 0.0) return ph * pk;
    const DD T = path_end(r);
    if (r.hi > 0.0) {
        const double p0 = ph * pk;
        return cap1(p0 + bvn_integral(h, k, false, T, false, p0) / kTwoPi);
    }
    if (r.hi > kBvnSwitch) {
        const double p0 = ph * pk;
        const double v = p0 - bvn_integral(h, k, true, T, false, p0) / kTwoPi;
        if (p0 <= 4.0 * v) return v;
    }
    const double pm = bvn_minus_one(h, k);
    return cap1(pm + bvn_integral(h, k, true, T, true, pm) / kTwoPi);
}
inline double bvn(double h, double k, double r) { return bvn(h, k, DD{std::clamp(r, -1.0, 1.0), 0.0}); }

// --------------------------------------------------------------- trivariate

// One term of the path integral for the trivariate (Plackett 1954; Genz 2004 §3):
// keep sigma_ij = a, move sigma_ik = t c1 and sigma_jk = t c2 together for t in
// [0, 1]. Then dP/dt = c1 dP/dsigma_ik + c2 dP/dsigma_jk, and
//     dP / dsigma_ik = phi2(h_i, h_k; sigma_ik) P(X_j <= h_j | X_i = h_i, X_k = h_k).
// With sigma_ik = s = sin(theta) the c1 term is
//     (1 / 2pi) int_0^asin c1 exp(-q(h_i, h_k; s)) Phi(u_j(s)) dtheta,
//     u_j = (h_j (1 - s^2) - a (h_i - s h_k) - b (h_k - s h_i)) / sqrt((1 - s^2) det),
//     b = s c2 / c1 (= sigma_jk),  det = (1 - a^2)(1 - s^2) - (b - a s)^2 (> 0 on the
// path: it is det Sigma(t), a convex combination of positive definite matrices).
// Returns the integral without the 1 / 2pi (negative for c1 < 0).
inline double tvn_term(double hi, double hk, double hj, double a, double c1, double c2, double base) {
    if (c1 == 0.0) return 0.0;
    const bool minus = c1 < 0.0;
    // det = (1 - a^2)(1 - s^2) - (b - a s)^2 = (1 - s^2) K - g^2 with g = (c2 - a c1) / c1
    // (b - a s = s g) and K = 1 - a^2 + g^2, the constants as hi + lo and 1 - s^2 =
    // t^2 (2 - t^2) from t^2 as hi + lo: det has no rounding beyond the last, where the
    // matrix is nearly singular and det cancels
    const DD oma = dd::two_sum(1.0, -a) * dd::two_sum(1.0, a);
    const DD g = (DD{c2, 0.0} - dd::two_prod(a, c1)) / c1;
    const DD g2 = g * g, K = oma + g2;
    auto f = [&](const DD* t2, int n, double* o) {
        double u[21], ph[21];
        for (int i = 0; i < n; ++i) {
            const Sides sd = sides(t2[i], minus);
            const double s = sd.s, oms = sd.sm * sd.sp, b = s * c2 / c1;
            const double det = ((t2[i] * (DD{2.0, 0.0} - t2[i])) * K - g2).hi;
            const double num = hj * oms - a * (hi - s * hk) - b * (hk - s * hi);
            u[i] = det > 0.0 ? num / std::sqrt(oms * det) : (num > 0.0 ? kInf : (num < 0.0 ? -kInf : 0.0));
            o[i] = sd.jac * std::exp(-qform(hi, hk, sd.sm, sd.sp));
        }
        special::Ndtr::chunk(u, ph, n);
        for (int i = 0; i < n; ++i) o[i] *= ph[i];
    };
    // t from T = sqrt(1 - |c1|) (as hi + lo) to 1: sigma_ik from c1 to 0
    const DD T = dd_sqrt(dd::two_sum(1.0, -std::fabs(c1)));
    const double J =
        path_integral(f, T.hi, 1.0, true, base * kTwoPi, 0.5 * std::fabs(minus ? hi + hk : hi - hk)) - end_part(f, T);
    return minus ? -J : J;
}

struct Tri {
    double h[3];
    double r[3][3];
    // The bvn evaluations tvn_condition may still spend for the call this problem belongs
    // to (the t's mixture calls tvn at up to 4200 nodes, each conditioning integral takes
    // hundreds to thousands of bvn's): at 0 the value is NaN. Null: no limit (one tvn).
    std::int64_t* budget = nullptr;
};

// The path from (r_ij, 0, 0) with the pair (i, j) kept: value and the sum of the
// magnitudes of its terms (their ratio is the cancellation).
inline std::pair<double, double> tvn_direct(const Tri& T, int i, int j) {
    const int k = 3 - i - j;
    const double p0 = bvn(T.h[i], T.h[j], T.r[i][j]) * Phi(T.h[k]);
    const double J1 = tvn_term(T.h[i], T.h[k], T.h[j], T.r[i][j], T.r[i][k], T.r[j][k], p0);
    const double J2 = tvn_term(T.h[j], T.h[k], T.h[i], T.r[i][j], T.r[j][k], T.r[i][k], p0);
    return {p0 + (J1 + J2) / kTwoPi, p0 + (std::fabs(J1) + std::fabs(J2)) / kTwoPi};
}

// The pair (i, j) to keep: with at most one negative correlation, keep it (the two
// moved ones are >= 0, so both terms add); with none, keep the largest (the
// shortest paths).
inline constexpr int kPairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
inline void keep_pair(const Tri& T, int& i, int& j) {
    int best = -1;
    for (int q = 0; q < 3; ++q)
        if (T.r[kPairs[q][0]][kPairs[q][1]] < 0.0) best = q;  // at most one
    if (best < 0) {
        best = 0;
        for (int q = 1; q < 3; ++q)
            if (T.r[kPairs[q][0]][kPairs[q][1]] > T.r[kPairs[best][0]][kPairs[best][1]]) best = q;
    }
    i = kPairs[best][0], j = kPairs[best][1];
}

// Phi3 by conditioning on X_k (Genz 2004 §3; the textbook reduction): with
//     a(z) = (h_i - r_ik z) / s_i,  b(z) = (h_j - r_jk z) / s_j,  s = sqrt(1 - r^2),
//     rho = (r_ij - r_ik r_jk) / (s_i s_j)  (the partial correlation),
//     Phi3 = int_-inf^h_k phi(z) Phi2(a(z), b(z); rho) dz,
// a positive integrand: no cancellation at all, with bvn's relative accuracy inside;
// but a bvn per node (some 200-600), so it is the last resort of tvn. k is the
// smallest h (often the binding constraint). The range is [h_k - L, h_k], L doubled
// (five times at most) until the part left out, at most Phi(h_k - L) min(Phi(a), Phi(b))
// at h_k - L (Phi2 <= min(Phi(a), Phi(b)), and a(z) is increasing in z when r_ik < 0), is
// below 1e-17 of the integral.
//
// Where the integrand lives. Near a singular matrix (|r_ik|, |r_jk| or |rho| near 1) it
// is a spike far narrower than the range: uniform first panels see zeros only and the
// integral came out 0 (h ~ 1e-6, smallest eigenvalue 1.6e-9: the truth is 3.3e-10). The
// integrand is log-concave in z (Prekopa: a Gaussian measure of a convex set moved
// affinely, times phi), so it has one peak, and
//     Phi2(a, b; rho) <= Phi(-d) <= exp(-d^2 / 2) / 2,
// d the Mahalanobis distance from the origin to the quadrant {u <= (a, b)} (a convex set
// lies in a half-space at its distance). E(z) = z^2 + d^2(z) is convex and piecewise
// quadratic in closed form: its minimum over z <= h_k (bisection on its derivative, 100
// steps) is the peak to within its own width, and the first panels are cut
// geometrically around it, from the peak's width (the curvature of E there, or the slope
// at the end h_k) up to the range. The same cuts go around the three places where a
// factor turns: a = 0, b = 0 (widths s / |r|) and the ridge a + b = 0 (rho < 0) or a = b
// (rho > 0), sqrt(1 - rho^2) over the slope wide. A turn narrower than the spacing of a
// panel's outer nodes is invisible to the error estimate whatever its size: the ridge
// 2.4e-4 wide at the end of a panel 1.25 long, 1e-3 of the integrand high, left 1.7e-7 of
// the result out (kappa = 7.7e7, P = 0.07, a well conditioned value). Only widths below
// 1/64 of the range add cuts (wider ones span several nodes of a uniform panel): an
// ordinary matrix keeps the eight uniform panels.
// Nodes with E / 2 > 745.2 are 0 without their bvn (the bound above is below the
// smallest double there).
inline constexpr int kCondLevels = 9;  // geometric cuts per side, ratio 64: 64^9 > 160 / 2^-48
inline double tvn_condition(const Tri& T) {
    int k = 0;
    for (int q = 1; q < 3; ++q)
        if (T.h[q] < T.h[k]) k = q;
    const int i = k == 0 ? 1 : 0, j = 3 - k - i;
    const double rik = T.r[i][k], rjk = T.r[j][k];
    // s_i, s_j, 1 / s and rho as hi + lo: rho's numerator cancels when the matrix is
    // nearly singular, and the bvn's end is sqrt(1 -+ rho)
    const DD si = dd_sqrt(dd::two_sum(1.0, -rik) * dd::two_sum(1.0, rik));
    const DD sj = dd_sqrt(dd::two_sum(1.0, -rjk) * dd::two_sum(1.0, rjk));
    const DD isi = DD{1.0, 0.0} / si, isj = DD{1.0, 0.0} / sj;
    DD rho = (DD{T.r[i][j], 0.0} - dd::two_prod(rik, rjk)) / (si * sj);
    if (rho.hi > 1.0) rho = {1.0, 0.0};
    if (rho.hi < -1.0) rho = {-1.0, 0.0};
    const double hi = T.h[i], hj = T.h[j], hk = T.h[k];
    constexpr double kInvSqrt2Pi = 0.3989422804014326779;

    // a(z) = a0 + a1 z, b(z) = b0 + b1 z; omr = 1 - rho^2
    const double a0 = hi * isi.hi, a1 = -rik * isi.hi, b0 = hj * isj.hi, b1 = -rjk * isj.hi;
    const double rh = rho.hi, omr = ((DD{1.0, 0.0} - rho) * (DD{1.0, 0.0} + rho)).hi;
    const bool proxy = omr > 0.0 && std::isfinite(a0) && std::isfinite(a1) && std::isfinite(b0) && std::isfinite(b1);
    // E(z) = z^2 + d^2 (v = 0), dE/dz (v = 1) or d^2E/dz^2 (v = 2). d^2: 0 in the quadrant;
    // a^2 (or b^2) where the nearest point lies on one edge; the form in both where it is
    // the corner.
    auto E = [&](double z, int v) {
        const double a = a0 + a1 * z, b = b0 + b1 * z;
        const double base = v == 0 ? z * z : (v == 1 ? 2.0 * z : 2.0);
        if (a >= 0.0 && b >= 0.0) return base;
        if (a < 0.0 && rh * a <= b) return base + (v == 0 ? a * a : (v == 1 ? 2.0 * a * a1 : 2.0 * a1 * a1));
        if (b < 0.0 && rh * b <= a) return base + (v == 0 ? b * b : (v == 1 ? 2.0 * b * b1 : 2.0 * b1 * b1));
        const double pa = a - rh * b, pb = b - rh * a;
        if (v == 0) return base + (a * pa + b * pb) / omr;
        if (v == 1) return base + 2.0 * (pa * a1 + pb * b1) / omr;
        return base + 2.0 * (a1 * (a1 - rh * b1) + b1 * (b1 - rh * a1)) / omr;
    };
    auto f = [&](const double* z, double* o, int n) {
        for (int q = 0; q < n; ++q) {
            if (proxy && 0.5 * E(z[q], 0) > 745.2) {
                o[q] = 0.0;
                continue;
            }
            if (T.budget && --*T.budget < 0) {
                o[q] = kNaN;
                continue;
            }
            const double a = ((DD{hi, 0.0} - dd::two_prod(rik, z[q])) * isi).hi;
            const double b = ((DD{hj, 0.0} - dd::two_prod(rjk, z[q])) * isj).hi;
            o[q] = kInvSqrt2Pi * std::exp(-0.5 * z[q] * z[q]) * bvn(a, b, rho);
        }
    };

    // The places to cut around, (center, width): the three turns, and the peak. E' is
    // increasing: the peak is its zero in [h_k - 320, h_k], or the end h_k. Its width is the
    // distance to where E has risen by 2 (the bound down by 1/e), the nearer side: E is
    // piecewise, so the curvature at the peak says nothing about a wall a step away (a
    // flat piece 3e-7 long before a curvature of 1e15).
    double fc[4], fw[4];
    int nf = 0;
    auto feature = [&](double c, double w) {
        if (std::isfinite(c) && w >= 0.0) fc[nf] = c, fw[nf++] = w;
    };
    if (proxy) {
        if (a1 != 0.0) feature(-a0 / a1, 1.0 / std::fabs(a1));
        if (b1 != 0.0) feature(-b0 / b1, 1.0 / std::fabs(b1));
        const double slope = rh < 0.0 ? a1 + b1 : a1 - b1, off = rh < 0.0 ? a0 + b0 : a0 - b0;
        if (slope != 0.0) feature(-off / slope, std::sqrt(omr) / std::fabs(slope));
        auto bisect = [&](double zl, double zr, auto&& left) {  // 100 halvings at most; returns zr
            for (int it = 0; it < 100; ++it) {
                const double m = 0.5 * (zl + zr);
                if (!(zl < m && m < zr)) break;
                (left(m) ? zl : zr) = m;
            }
            return zr;
        };
        const double far = hk - 320.0;
        bool found = true;
        double pk = hk;
        if (E(hk, 1) <= 0.0) pk = hk;
        else if (E(far, 1) < 0.0) pk = bisect(far, hk, [&](double m) { return E(m, 1) < 0.0; });
        else found = false;  // the peak is left of every range below
        if (found) {
            const double top = E(pk, 0) + 2.0;
            // left of the peak E decreases toward it; right of it E increases
            const double wl =
                E(pk - 320.0, 0) < top ? 320.0 : pk - bisect(pk - 320.0, pk, [&](double m) { return E(m, 0) >= top; });
            const double wr = !(E(hk, 0) >= top) ? kInf : bisect(pk, hk, [&](double m) { return E(m, 0) < top; }) - pk;
            feature(pk, std::min(wl, wr));
        }
    }

    double L = 10.0;
    for (int round = 0;; ++round, L *= 2.0) {  // L = 10, 20, 40, 80, 160
        const double lo = hk - L;
        double cuts[8 + 4 * (1 + 2 * kCondLevels)];
        int nc = 0;
        for (int q = 0; q < 7; ++q) cuts[nc++] = lo + L * (q + 1) / 8.0;
        for (int q = 0; q < nf; ++q) {
            if (!(fw[q] < L / 64.0) || !(fc[q] < hk + L)) continue;
            auto add = [&](double z) {
                if (z > lo && z < hk) cuts[nc++] = z;
            };
            add(fc[q]);
            double w = std::max(fw[q], L * 0x1p-48);
            for (int l = 0; l < kCondLevels && w < 2.0 * L; ++l, w *= 64.0) add(fc[q] - w), add(fc[q] + w);
        }
        if (nc > 7) std::sort(cuts, cuts + nc);
        // its own panels: the cuts must not use up the refinements (kMaxPanels - 8, as without them)
        Panel p[kMaxPanels + 4 * (1 + 2 * kCondLevels)];
        int n = 0;
        double from = lo;
        for (int q = 0; q < nc; ++q)
            if (cuts[q] > from) p[n++] = gk21(f, from, cuts[q]), from = cuts[q];
        p[n++] = gk21(f, from, hk);
        const double I = adapt(f, p, n, 0.0, n + kMaxPanels - 8);
        if (std::isnan(I)) return I;  // the budget ran out
        double tail = 1.0;
        if (rik < 0.0) tail = std::min(tail, Phi((hi - rik * lo) * isi.hi));
        if (rjk < 0.0) tail = std::min(tail, Phi((hj - rjk * lo) * isj.hi));
        if (Phi(lo) * tail <= 1e-17 * I || round == 4) return I;
    }
}

// Phi3(h; R) = P(X1 <= h1, X2 <= h2, X3 <= h3), unit variances, finite h.
// At most one negative correlation: the direct path, no cancellation. Two or three:
// the direct path (keeping the nonnegative one, or the largest) subtracts; when that
// loses more than two bits, also try flipping X_k (k common to two negative pairs;
// with three, the largest h_k, whose complement is the smallest):
//     P = Phi2(h_i, h_j; r_ij) - P(X_i <= h_i, X_j <= h_j, -X_k < -h_k),
// whose second term has at most one negative correlation; keep the better
// conditioned of the two. When both lose more than two bits (the lower tails with
// negative correlations: both subtract nearly equal terms), condition on a variable
// instead (tvn_condition), which never cancels.
inline double tvn(const Tri& T) {
    int negs = 0;
    for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 3; ++b) negs += T.r[a][b] < 0.0;
    int i, j;
    if (negs <= 1) {
        keep_pair(T, i, j);
        return std::clamp(tvn_direct(T, i, j).first, 0.0, 1.0);
    }
    // the direct path: keep the nonnegative pair (two negative) or the largest |r| (three)
    {
        int best = -1;
        for (int q = 0; q < 3; ++q) {
            const double v = T.r[kPairs[q][0]][kPairs[q][1]];
            if (negs == 2 ? v >= 0.0 : (best < 0 || std::fabs(v) > std::fabs(T.r[kPairs[best][0]][kPairs[best][1]])))
                best = q;
        }
        i = kPairs[best][0], j = kPairs[best][1];
    }
    const auto [va, ma] = tvn_direct(T, i, j);
    if (ma <= 4.0 * va) return std::clamp(va, 0.0, 1.0);
    int k;
    if (negs == 2) {
        k = 3 - i - j;  // the variable in both negative pairs
    } else {
        k = 0;
        for (int q = 1; q < 3; ++q)
            if (T.h[q] > T.h[k]) k = q;
    }
    const int ii = k == 0 ? 1 : 0, jj = 3 - k - ii;
    Tri F = T;
    F.h[k] = -T.h[k];
    for (int q = 0; q < 3; ++q)
        if (q != k) F.r[q][k] = F.r[k][q] = -T.r[q][k];
    keep_pair(F, i, j);
    const double inner = tvn_direct(F, i, j).first;
    const double b2 = bvn(T.h[ii], T.h[jj], T.r[ii][jj]);
    const double vb = b2 - inner, mb = b2 + inner;
    const bool use_b = vb > 0.0 && (!(va > 0.0) || mb * va < ma * vb);
    const double v = use_b ? vb : va, m = use_b ? mb : ma;
    if (v > 0.0 && m <= 4.0 * v) return std::clamp(v, 0.0, 1.0);
    return std::clamp(tvn_condition(T), 0.0, 1.0);
}

// ------------------------------------------------- the t at d = 2, 3, exactly

// The t as a scale mixture: X = Z / R, R = sqrt(S / nu), S ~ chi^2_nu, so
//     P(X <= h) = E[Phi_d(R h)] = int_0^inf Gamma(a)^-1 y^(a-1) e^-y Phi_d(sqrt(y / a) h) dy,
// a = nu / 2, y = nu R^2 / 2 (Genz and Bretz 2009 §4.2 write the same mixture over the
// chi variable). In v = log(y / a) = log R^2 the weight is
//     exp(c(a) - a (e^v - 1 - v)),  c(a) = a log a - a - lgamma(a),
// peaked at v = 0 with width ~ 1 / sqrt(a), or a left tail ~ e^(a v) for small a.
//
// In the tails the exponents are large (a (e^v - 1 - v) and R^2 h^2 / 2 reach ~700 where
// P ~ 1e-300), and a rounding of eps in one becomes 700 eps in the integrand. So each node
// is evaluated at x = e^v as rounded, which is exact: the exponent a (x - 1 - log x) is
// formed as hi + lo (special/dd.hpp), and so are the integrand's own (MixTerm.e below); the
// node v' = log x, off v by ~1e-16, is moved back to v to first order by the integrand's
// logarithmic derivative (MixTerm.dlog), whose own error is then negligible.
//
// c(a) as hi + lo (it is an exponent too: |c| ~ log a reaches 700 for a = 1e+-300, and its
// rounding alone moved the result by 2e-14 at df = 1e300): below a = 10 from its three
// terms in double-double (in double they cancel to ~eps a log a); above, Stirling's series (lgamma(a) = (a - 1/2) log a - a + log(2 pi)/2 +
// sum B_2k / (2k (2k-1) a^(2k-1)); seven terms, the next < 3e-17 at a = 10).
inline DD stirling_c(double a) {
    if (a < 10.0) return dd::log(a) * a - a - dd::loggamma(DD{a, 0.0});
    const double r = 1.0 / a, r2 = r * r;
    const double s =
        r *
        (1.0 / 12 +
         r2 * (-1.0 / 360 +
               r2 * (1.0 / 1260 + r2 * (-1.0 / 1680 + r2 * (1.0 / 1188 + r2 * (-691.0 / 360360 + r2 * (1.0 / 156)))))));
    return dd::log(a) * 0.5 - dd::kLogSqrt2Pi - s;
}

// One integrand value g(R) = exp(-e) f at R^2 = x, with d log g / dv (v = log x).
struct MixTerm {
    DD e{0.0, 0.0};
    double f = 1.0, dlog = 0.0;
};

// One node of the mixture, R^2 = x = e^v, in one of three forms:
//   near   x the rounded exp(v) (lo = 0): the exponents are formed from x itself and the
//          node is moved back by dlog (above);
//   one    a > kMixBigA: the weight is 1 / sqrt(a) wide around v = 0, and the move from the
//          rounded exp(v) (up to 1.1e-16) is no longer small against it (a dv^2 ~ 1 at
//          a ~ 1e32: the integrand came out constant and the result 1). There x = 1 +
//          expm1(v) as hi + lo, exact to eps |v|: nothing to move;
//   far    v < kMixFarV: exp(v) is below the normal doubles (limits beyond 1e150 put the
//          action at R ~ 1 / |h|; exp(v) rounded to 0 gave log(0) and results like -70).
//          sqrt(x) h is formed in logs and the exponents from v directly.
inline constexpr double kMixFarV = -700.0;
inline constexpr double kMixBigA = 1e13;
struct MixNode {
    DD x{0.0, 0.0};
    double half_v = 0.0;
    bool far = false;
    // sqrt(x) h
    double times(double h) const {
        if (!far) return std::sqrt(x.hi) * h;
        if (h == 0.0) return 0.0;
        return std::copysign(std::exp(half_v + std::log(std::fabs(h))), h);
    }
};

// E[g(R)] for g given at a node (above), the orthant p0 = Phi_d(0), the limits hs[0, nh)
// and sum |h_i| = habs. The left end: for R <= r_c, |Phi_d(R h) - p0| <= R habs / sqrt(2 pi)
// (each partial derivative of Phi_d is at most phi(0)), so with r_c = 1e-17 p0 / (0.4 habs)
// that part is p0 P(R <= r_c) to 1e-17 relative (sd.special's gammainc; where its argument
// is below the normal doubles, the first term of its series, y^a / Gamma(a + 1), in logs).
// The right end at y = a + 40 sqrt(a) + 750 leaves out P(S > that) < 1e-300. The first
// panels are cut at v = +-sigma 4^k, sigma = min(1, 1 / sqrt(a)): the weight's peak and its
// tails at their own scales; and around v_i = -2 log |h_i|, where R |h_i| ~ 1 and Phi_d
// turns, when that is far from the weight's peak (|v_i| > 12: limits above 400 or below
// 1/400).
template <class G>
double t_mixture(double nu, double p0, double habs, const double* hs, int nh, G&& g) {
    if (habs == 0.0) return p0;
    const double a = 0.5 * nu;
    if (!(a > 0.0) || std::isnan(p0)) return kNaN;  // a subnormal nu halves to 0
    const DD c = stirling_c(a);
    double vhi = std::log1p((40.0 * std::sqrt(a) + 750.0) / a);
    if (!(vhi < kInf)) vhi = std::log(a + 40.0 * std::sqrt(a) + 750.0) - std::log(a);  // the quotient overflowed
    const double rc = 1e-17 * p0 / (0.4 * habs);
    constexpr double kTiny = std::numeric_limits<double>::min(), kHuge = std::numeric_limits<double>::max();
    const double vc =
        rc >= kTiny ? 2.0 * std::log(rc) : 2.0 * (std::log(1e-17 * p0 / 0.4) - std::log(std::min(habs, kHuge)));
    if (!(vc < vhi)) return p0;
    const double yc = a * std::exp(vc);
    const double left =
        yc >= kTiny ? p0 * special::Gammainc::f(a, yc) : p0 * std::exp(a * (std::log(a) + vc) - std::lgamma(a + 1.0));
    const bool big = a > kMixBigA;
    auto f = [&](const double* v, double* o, int n) {
        for (int i = 0; i < n; ++i) {
            MixNode nd;
            DD ew;            // a (x - 1 - log x)
            double dv = 0.0;  // v - log x (near)
            if (v[i] < kMixFarV) {
                nd.far = true, nd.half_v = 0.5 * v[i];
                ew = dd::two_sum(-1.0, -v[i]) * a;  // x < 1e-304 adds nothing
            } else if (big) {
                if (std::fabs(v[i]) > 1e-3) {  // a (e^v - 1 - v) > 5e6
                    o[i] = 0.0;
                    continue;
                }
                nd.x = dd::two_sum(1.0, std::expm1(v[i]));
                ew = dd::a_mu(a, nd.x);
            } else {
                const double x = std::exp(v[i]);
                const DD lx = dd::log(x);
                dv = (v[i] - lx.hi) - lx.lo;
                nd.x = DD{x, 0.0};
                ew = dd::a_mu(a, nd.x);
            }
            if ((ew - c).hi > 745.2) {  // t.e >= 0: the weight is 0 whatever g is
                o[i] = 0.0;
                continue;
            }
            const MixTerm t = g(nd);
            const DD e = ew + t.e - c;
            const double w = e.hi > 745.2 || t.f == 0.0 ? 0.0 : std::exp(-e.hi) * (1.0 - e.lo);
            o[i] = dv == 0.0 ? w * t.f : w * t.f * (1.0 + (t.dlog - a * (nd.x.hi - 1.0)) * dv);
        }
    };
    const double sigma = std::min(1.0, 1.0 / std::sqrt(a));
    double cuts[96];
    int nc = 0;
    for (double t = sigma * 0x1p40; t >= sigma && nc < 20; t *= 0.25)  // the negative side, outside in
        if (-t > vc) cuts[nc++] = -t;
    cuts[nc++] = 0.0;
    for (double t = sigma; t < vhi && nc < 40; t *= 4.0) cuts[nc++] = t;
    for (int q = 0; q < nh && q < 4; ++q) {
        if (hs[q] == 0.0) continue;
        const double vi = -2.0 * std::log(std::fabs(hs[q]));
        if (!(std::fabs(vi) > 12.0)) continue;
        auto add = [&](double t) {
            if (t > vc && t < vhi) cuts[nc++] = t;
        };
        add(vi);
        for (double t = 1.0; t <= 1024.0; t *= 4.0) add(vi - t), add(vi + t);  // 6 a side
    }
    std::sort(cuts, cuts + nc);
    return cap1(left + integrate_at(f, vc, vhi, cuts, nc, left));
}

// The bivariate t, nu finite, by Plackett's identity mixed over R (Genz 2004 §4 writes
// the t's version; tools/mvcdfref.py's bvt is the same mixture term by term):
//     d P / d theta = E[exp(-R^2 q(s))] / (2 pi) = (1 + q / a)^-a / (2 pi),  s = sin theta,
// a closed form, so P is one integral in the path variable t (as bvn) plus its start:
//     r >= 0:  E[Phi(R h) Phi(R k)] + (1 / 2pi) int_0^asin r (1 + q/a)^-a dtheta
//     r <  0:  E[Phi(R h) Phi(R k)] - (1 / 2pi) int_asin r^0 ...  (r > -1/2, losing at
//              most two bits), else
//              P(X <= h, Y <= k; -1) + (1 / 2pi) int_-pi/2^asin r ...
// with the start at r = 0 one mixture over R of a product of Phi's (t_mixture) and at
// r = -1 the t's P(-k <= X <= h) (sd.special's stdtr). The integrand's exponent
// a log(1 + q / a) is formed in double-double from the exact t, h and k: it reaches ~700
// in the tails, where the bivariate t's relative condition in r is ~200 too.
inline double bvt_density(double h, double k, double a, bool minus, DD t2) {
    const DD far = DD{2.0, 0.0} - t2;
    if (t2.hi == 0.0) return 0.0;  // the pole (an end, never a node)
    const DD sm = minus ? far : t2, sp = minus ? t2 : far;
    // limits beyond 2^400 (their squares, over 1 - s^2, overflow): q of the limits scaled by 2^-E, exactly
    int E = 0;
    const double big = std::max(std::fabs(h), std::fabs(k));
    if (big > 0x1p400) E = std::ilogb(big), h = std::ldexp(h, -E), k = std::ldexp(k, -E);
    const DD hk = dd::two_prod(h, k);
    const DD d = dd::two_sum(h, hk.hi >= 0.0 ? -k : k);
    const DD qa = d * d / (sm * sp * 2.0);
    const DD q = hk.hi >= 0.0 ? qa + hk / sp : qa - hk / sm;
    const DD x = q / a;
    if (E != 0 || !(x.hi < 0x1p900)) {
        // log(1 + q 4^E / a) from the logs of its pieces (the quotient is beyond the doubles,
        // or near it): plain double, a log(...) <= 745 where the density is not 0. q is a
        // sum of non-negative terms: not finite means it overflowed near the pole.
        if (!std::isfinite(q.hi)) return 0.0;
        const double lx = std::log(q.hi) + 2.0 * E * 0.6931471805599453 - std::log(a);
        const double e = a * (lx > 40.0 ? lx : std::log1p(std::exp(lx)));
        return e > 745.2 ? 0.0 : std::exp(-e) * 2.0 / std::sqrt(far.hi);
    }
    // a log(1 + x) as hi + lo: log1p by its series form for small x (1 + x would drop x's low bits)
    const DD L = x.hi < 0x1p-8 ? x - dd::log1pmx_neg(x) : dd::log(DD{1.0, 0.0} + x);
    const DD E2 = L * a;
    if (E2.hi > 745.2) return 0.0;
    return std::exp(-E2.hi) * (1.0 - E2.lo) * 2.0 / std::sqrt(far.hi);
}

// The integral of (1 + q/a)^-a dtheta on the given side over t from the end T (hi + lo)
// to the pole (to_pole) or to s = 0, added to base. Near the pole (t below |h -+ k| / 2)
// the integrand goes like t^(2a), not smooth at 0 for small a: the panels are also cut
// geometrically below the step, down to where t^(2a+1) is below 1e-17.
inline double bvt_integral(double h, double k, double a, bool minus, DD T, bool to_pole, double base) {
    auto f = [&](const DD* t2, int n, double* o) {
        for (int i = 0; i < n; ++i) o[i] = bvt_density(h, k, a, minus, t2[i]);
    };
    const double delta = 0.5 * std::fabs(minus ? h + k : h - k);
    if (!to_pole) return path_integral(f, T.hi, 1.0, true, base * kTwoPi, delta) - end_part(f, T);
    const int levels = std::min(30, static_cast<int>(std::ceil(17.0 * 2.302585 / ((2.0 * a + 1.0) * 1.386294))));
    return path_integral(f, 0.0, T.hi, false, base * kTwoPi, delta, levels) + end_part(f, T);
}

// P(X <= h, Y <= k) for the t at correlation -1: P(-k <= X <= h), by the tails it is made of.
inline double bvt_minus_one(double h, double k, double nu) {
    if (!(h > -k)) return 0.0;
    auto T = [&](double x) { return special::Stdtr::f(nu, x); };
    if (h <= 0.0) return T(h) - T(-k);
    if (k <= 0.0) return T(k) - T(-h);
    return (1.0 - T(-h)) - T(-k);
}

// Phi(sqrt(x) h) for the mixture: e^(-x h^2 / 2) erfcx(...) / 2 in the lower tail, its
// exponent exact (from h and x; from y = sqrt(x) h itself at a far node, or where h^2 would
// overflow). Below y = -40 the factor is 0 in any double.
inline void mix_phi(const MixNode& nd, double h, MixTerm& t) {
    const double y = nd.times(h);
    if (h < 0.0) {
        if (!(y > -40.0)) {
            t.f = 0.0;
            return;
        }
        if (nd.far || !(h > -0x1p500)) t.e = t.e + dd::two_prod(y, y) * 0.5;
        else t.e = t.e + (nd.x.lo == 0.0 ? dd::two_prod(h, h) * (0.5 * nd.x.hi) : dd::two_prod(h, h) * (nd.x * 0.5));
        t.f *= 0.5 * special::Erfcx::f(-y * kSqrtHalf);
    } else {
        t.f *= Phi(y);
    }
    if (!(y > 40.0)) t.dlog += 0.5 * y * mills_lower(y);  // beyond: 0 (and y may be inf, inf * 0)
}

// Phi_3(0), the orthant probability. Sheppard's 1/8 + (asin r12 + asin r13 + asin r23) /
// (4 pi) (Plackett 1954) cancels where the matrix is nearly singular with negative
// correlations (the sum of the asin's near -pi/2: the truth 3.3e-10 came out with seven
// digits). The orthant is the cone {z : L z <= 0}; its edges are the columns of -L^-1,
// whose Gram matrix is R^-1, and the solid angle of a cone with unit edges e1, e2, e3 is
// (van Oosterom and Strackee 1983)
//     tan(Omega / 2) = |det(e1 e2 e3)| / (1 + e1.e2 + e1.e3 + e2.e3).
// With A = adj R (A_ii = 1 - r_jk^2, A_ij = r_ik r_jk - r_ij): e_i.e_j = A_ij / sqrt(A_ii A_jj)
// and det = det R / sqrt(A11 A22 A33), every piece a product or a short sum formed as
// hi + lo: no cancellation is left in Phi_3(0) = Omega / (4 pi).
inline double orthant3(const Tri& T) {
    const double r12 = T.r[0][1], r13 = T.r[0][2], r23 = T.r[1][2];
    auto om = [](double r) { return dd::two_sum(1.0, -r) * dd::two_sum(1.0, r); };
    const DD A11 = om(r23), A22 = om(r13), A33 = om(r12);
    const DD A12 = dd::two_prod(r13, r23) - r12, A13 = dd::two_prod(r12, r23) - r13, A23 = dd::two_prod(r12, r13) - r23;
    const DD det = A11 * A22 - A12 * A12;  // det R (R33 = 1)
    if (!(det.hi > 0.0 && A11.hi > 0.0 && A22.hi > 0.0 && A33.hi > 0.0)) return kNaN;
    const DD s1 = dd_sqrt(A11), s2 = dd_sqrt(A22), s3 = dd_sqrt(A33);
    const DD num = det / (s1 * s2 * s3);
    const DD den = DD{1.0, 0.0} + A12 / (s1 * s2) + A13 / (s1 * s3) + A23 / (s2 * s3);
    return std::atan2(num.hi, den.hi) / kTwoPi;
}

// The bivariate and trivariate t, nu finite: Phi_2(0) = 1/4 + asin(r) / (2 pi) (Sheppard)
// and Phi_3(0) above.
// r as hi + lo (tvt_condition's partial correlation; a double r is {r, 0}).
inline double bvt(double h, double k, DD r, double nu) {
    const double a = 0.5 * nu;
    if (!(a > 0.0) || std::isnan(h) || std::isnan(k) || std::isnan(r.hi)) return kNaN;
    auto start = [&] {
        const double hs[2] = {h, k};
        return t_mixture(nu, 0.25, std::fabs(h) + std::fabs(k), hs, 2, [&](const MixNode& nd) {
            MixTerm t;
            mix_phi(nd, h, t);
            if (t.f != 0.0) mix_phi(nd, k, t);
            return t;
        });
    };
    if (r.hi == 0.0) return start();
    if (r.hi >= 1.0) r = {1.0, 0.0};
    if (r.hi <= -1.0) r = {-1.0, 0.0};
    const DD T = path_end(r);
    if (r.hi > 0.0) {
        const double p0 = start();
        return cap1(p0 + bvt_integral(h, k, a, false, T, false, p0) / kTwoPi);
    }
    if (r.hi > kBvnSwitch) {
        const double p0 = start();
        const double v = p0 - bvt_integral(h, k, a, true, T, false, p0) / kTwoPi;
        if (p0 <= 4.0 * v) return v;
    }
    const double pm = bvt_minus_one(h, k, nu);
    return cap1(pm + bvt_integral(h, k, a, true, T, true, pm) / kTwoPi);
}
inline double bvt(double h, double k, double r, double nu) { return bvt(h, k, DD{r, 0.0}, nu); }

// The trivariate t by conditioning on X_k, the t's own version of tvn_condition. Given
// X_k = x (a univariate t_nu), (X_i, X_j) is a bivariate t with nu + 1 degrees of freedom,
// location (r_ik, r_jk) x, the partial correlation rho and scales s_i tau, s_j tau,
// tau(x) = sqrt((nu + x^2) / (nu + 1)) (Kotz and Nadarajah 2004, "Multivariate t
// Distributions and Their Applications", §1.11):
//     P = int_-inf^h_k f_nu(x) T2_{nu+1}((h_i - r_ik x) / (s_i tau), (h_j - r_jk x) / (s_j tau); rho) dx,
// a positive integrand with bvt's accuracy inside. tvt uses it where the mixture over tvn
// would itself condition at too many nodes (a nearly singular matrix: each tvn is then an
// integral of bvn's, and the three nested quadratures took 45-215 s); here the inner
// function is bvt, two quadratures of closed forms.
//
// The variable is u = asinh(h_k - x): linear near the end, logarithmic in the tail (the
// t's tails are powers: smooth in log x), up to x = h_k - X, X = 1e17 max(1, sqrt nu, |h|),
// beyond which the limits of T2 are their values at -inf to 1e-17 and the rest is
// F_nu(h_k - X) T2(r_ik sqrt(nu+1) / s_i, r_jk sqrt(nu+1) / s_j; rho). First panels: every
// log 16 in u, and geometric cuts (ratio 64 in x) around the places where a factor turns
// within a width smaller than its distance from the end: x = 0 (the density's core, sqrt
// nu wide), the zeros of the two limits (s tau / |r| wide) and the ridge of rho, where the
// limits cross (sqrt(1 - rho^2) tau over the slope wide).
inline DD dd_log1p(DD x) { return x.hi < 0x1p-8 ? x - dd::log1pmx_neg(x) : dd::log(DD{1.0, 0.0} + x); }
inline constexpr double kCondTMax = 1e100;  // limits and sqrt(nu) up to here (x^2 must not overflow)
inline double tvt_condition(const Tri& T, double nu) {
    int k = 0;
    for (int q = 1; q < 3; ++q)
        if (T.h[q] < T.h[k]) k = q;
    const int i = k == 0 ? 1 : 0, j = 3 - k - i;
    const double rik = T.r[i][k], rjk = T.r[j][k];
    const double hi = T.h[i], hj = T.h[j], hk = T.h[k];
    const double hmax = std::max({std::fabs(hi), std::fabs(hj), std::fabs(hk)});
    const double a = 0.5 * nu, rnu = std::sqrt(nu);
    if (!(a > 0.0) || !(hmax <= kCondTMax) || !(rnu <= kCondTMax) || !(nu >= 0x1p-500)) return kNaN;
    const DD si = dd_sqrt(dd::two_sum(1.0, -rik) * dd::two_sum(1.0, rik));
    const DD sj = dd_sqrt(dd::two_sum(1.0, -rjk) * dd::two_sum(1.0, rjk));
    const DD isi = DD{1.0, 0.0} / si, isj = DD{1.0, 0.0} / sj;
    DD rho = (DD{T.r[i][j], 0.0} - dd::two_prod(rik, rjk)) / (si * sj);
    if (rho.hi > 1.0) rho = {1.0, 0.0};
    if (rho.hi < -1.0) rho = {-1.0, 0.0};
    if (!(si.hi > 0.0 && sj.hi > 0.0) || std::isnan(rho.hi)) return kNaN;
    // log of the density's constant, Gamma(a + 1/2) / (Gamma(a) sqrt(nu pi)), from c(a) (above):
    // log Gamma(a + 1/2) - log Gamma(a) = a log1p(1 / 2a) + log(a + 1/2) / 2 - 1/2 - c(a + 1/2) + c(a)
    const DD lc = dd_log1p(DD{0.5, 0.0} / a) * a + dd::log(dd::two_sum(a, 0.5)) * 0.5 - 0.5 - stirling_c(a + 0.5) +
                  stirling_c(a) - (dd::log(nu) + dd::log(kPi)) * 0.5;
    const double nu1 = nu + 1.0;
    auto tau = [&](double x) { return std::sqrt((nu + x * x) / nu1); };
    // f_nu(x) T2(...) at x
    auto g = [&](double x) {
        const DD x2 = dd::two_prod(x, x);
        const DD e = dd_log1p(x2 / nu) * (a + 0.5) - lc;
        if (e.hi > 745.2) return 0.0;
        const DD itau = DD{1.0, 0.0} / dd_sqrt((x2 + nu) / nu1);
        const double A = ((DD{hi, 0.0} - dd::two_prod(rik, x)) * isi * itau).hi;
        const double B = ((DD{hj, 0.0} - dd::two_prod(rjk, x)) * isj * itau).hi;
        return std::exp(-e.hi) * (1.0 - e.lo) * bvt(A, B, rho, nu1);
    };
    const double c0 = std::max(1.0, rnu);  // the scale of the end
    auto f = [&](const double* u, double* o, int n) {
        for (int q = 0; q < n; ++q) o[q] = g(hk - c0 * std::sinh(u[q])) * c0 * std::cosh(u[q]);
    };
    const double X = 1e17 * std::max({1.0, rnu, hmax});
    const double umax = std::asinh(X / c0);
    double cuts[kMaxPanels];
    int nc = 0;
    const int nuni = static_cast<int>(std::min(30.0, std::ceil(umax / 2.772588722239781)));
    for (int q = 1; q < nuni; ++q) cuts[nc++] = umax * q / nuni;
    // around a turn at x = c of width w
    auto feature = [&](double c, double w) {
        if (!std::isfinite(c) || !(w > 0.0) || !(c < hk + 64.0 * w)) return;
        auto add = [&](double x) {
            if (x < hk && nc < kMaxPanels / 2 - 4) {
                const double uu = std::asinh((hk - x) / c0);
                if (uu > 0.0 && uu < umax) cuts[nc++] = uu;
            }
        };
        add(c);
        double d = w;
        for (int l = 0; l < 8; ++l, d *= 64.0) add(c - d), add(c + d);
    };
    const double a0 = hi * isi.hi, a1 = -rik * isi.hi, b0 = hj * isj.hi, b1 = -rjk * isj.hi;
    const double omr = ((DD{1.0, 0.0} - rho) * (DD{1.0, 0.0} + rho)).hi;
    if (rnu < 1.0) feature(0.0, rnu);
    if (a1 != 0.0) feature(-a0 / a1, tau(-a0 / a1) / std::fabs(a1));
    if (b1 != 0.0) feature(-b0 / b1, tau(-b0 / b1) / std::fabs(b1));
    {
        const double slope = rho.hi < 0.0 ? a1 + b1 : a1 - b1, off = rho.hi < 0.0 ? a0 + b0 : a0 - b0;
        if (slope != 0.0 && omr >= 0.0)
            feature(-off / slope, std::max(std::sqrt(omr), 0x1p-52) * tau(-off / slope) / std::fabs(slope));
    }
    std::sort(cuts, cuts + nc);
    const double I = integrate_at(f, 0.0, umax, cuts, nc, 0.0);
    const double far = special::Stdtr::f(nu, hk - X);
    const double rest =
        far > 0.0 ? far * bvt(rik * isi.hi * std::sqrt(nu1), rjk * isj.hi * std::sqrt(nu1), rho, nu1) : 0.0;
    return cap1(I + rest);
}

// The trivariate t: the mixture over tvn where tvn's own paths do not cancel (0.2-0.5 ms,
// 7e-16 against Arb). Where they do, its nodes condition (tvn_condition), hundreds of
// bvn's each: they may spend kTvtBudget bvn's together (some tens of ms), and past that
// the whole value is tvt_condition's (5-120 ms, 2.4e-15 against Arb on the same points;
// the mixture alone took up to 0.6 s on them, and minutes near singular matrices).
inline constexpr std::int64_t kTvtBudget = 20000;
// The limits of the inner normal are clamped at +-1e100 (far past where Phi_3 is 0 or 1 in
// any double; R |h| itself can overflow).
inline double tvt(const Tri& T, double nu) {
    const double p0 = orthant3(T);
    if (!(p0 > 0.0)) return kNaN;  // not positive definite to working precision
    std::int64_t budget = kTvtBudget;
    const double v =
        t_mixture(nu, p0, std::fabs(T.h[0]) + std::fabs(T.h[1]) + std::fabs(T.h[2]), T.h, 3, [&](const MixNode& nd) {
            Tri U = T;
            U.budget = &budget;
            if (!nd.far) {
                const double s = std::sqrt(nd.x.hi);
                for (double& v : U.h) v = std::clamp(v * s, -1e100, 1e100);
            } else {
                for (double& v : U.h) v = std::clamp(nd.times(v), -1e100, 1e100);
            }
            MixTerm t;
            t.f = tvn(U);
            return t;
        });
    return budget < 0 ? tvt_condition(T, nu) : v;
}

// ---------------------------------------------------------- quasi-Monte Carlo

// Genz's method (Genz 1992, "Numerical computation of multivariate normal
// probabilities", J. Comput. Graph. Stat. 1; Genz and Bretz 2002, "Comparison of
// methods for the computation of multivariate t probabilities", JCGS 11; Genz and
// Bretz 2009, "Computation of Multivariate Normal and t Probabilities", ch. 4).
// After the Cholesky factor L of the (reordered) covariance, the probability is
//     E_w[ e_1 e_2 ... e_n ],  e_i = Phi((r b_i - sum_{j<i} L_ij y_j) / L_ii),
//                              y_j = Phi^-1(w_j e_j),
// over w in [0, 1]^(n-1) (normal, r = 1) or [0, 1]^n for the t, whose extra
// coordinate gives r = sqrt(chi^2_nu^-1(w_0) / nu) (X <= b iff Z <= r b).
//
// The points are randomly shifted rank-1 lattice rules, x_i = frac(i z / N + Delta_m)
// for i < N (N prime), z from the fast component-by-component construction of
// Nuyens and Cools ("Fast algorithms for component-by-component construction of
// rank-1 lattice rules in shift-invariant reproducing kernel Hilbert spaces", Math.
// Comp. 75, 2006; scipy.stats' _cbc_lattice, whose product weights 0.8^j and kernel
// B_2 we use), periodized by the baker's transformation |2x - 1|, with M random
// shifts Delta_m. Measured against the Richtmyer (Kronecker, sqrt of the primes) rule
// of Genz's qsimvnv, at equal evaluations: 7.5x smaller error at d = 5 (and antithetic
// pairs 1 - w gained nothing on top of |2x - 1|, so there are none).
//
// A level is one N for every shift: its estimate is the mean of the M shift means,
// its variance that of the mean. Lattice rules do not extend, so the levels are
// independent (new N, new shifts) and combine by inverse-variance weighting, as
// Genz's DKBVRC (and scipy's _qauto) do. The error is 3 standard errors of the
// combination ("3 sigma"); N grows by the predicted factor (error ~ N^-1, between
// 1.5x and 8x, the next prime) until the error is at most atol or the evaluations
// reach max_points.
//
// The shifts are Philox4x64-10 (random/philox.hpp) under the fixed key {0x5eed, n}
// (n the number of variables integrated): coordinate j of shift m at level l is
// to_unit of word j % 4 of the block at counter (m, j / 4, l, 0). The lattices are
// functions of (D, N) alone. So the result is a function of the input alone, the
// same bits for any thread count.
inline constexpr int kShifts = 12;               // Genz's qsimvnv: 12 shifts
inline constexpr std::int64_t kQmcBlock = 128;   // points evaluated together
inline constexpr std::int64_t kQmcItem = 1024;   // points per task of the pool (a multiple of kQmcBlock)
inline constexpr std::int64_t kQmcStart = 1009;  // N at the first level (a prime)
inline constexpr std::uint64_t kShiftKey = 0x5eed;

namespace cbc {

inline bool is_prime(std::int64_t n) {
    if (n < 2) return false;
    for (std::int64_t q = 2; q * q <= n; ++q)
        if (n % q == 0) return false;
    return true;
}
inline std::int64_t next_prime(std::int64_t n) {  // the smallest prime >= n
    while (!is_prime(n)) ++n;
    return n;
}
inline std::int64_t prev_prime(std::int64_t n) {  // the largest prime <= n (n >= 2)
    while (n > 2 && !is_prime(n)) --n;
    return n;
}
inline std::int64_t mulmod(std::int64_t a, std::int64_t b, std::int64_t n) {
    return static_cast<std::int64_t>(static_cast<unsigned __int128>(a) * static_cast<std::uint64_t>(b) % n);
}
inline std::int64_t powmod(std::int64_t g, std::int64_t e, std::int64_t n) {
    std::int64_t r = 1;
    for (g %= n; e; e >>= 1, g = mulmod(g, g, n))
        if (e & 1) r = mulmod(r, g, n);
    return r;
}
// The smallest primitive root of the prime p: g with g^((p-1)/q) != 1 for each prime q | p - 1.
inline std::int64_t primitive_root(std::int64_t p) {
    std::vector<std::int64_t> fs;
    std::int64_t m = p - 1;
    for (std::int64_t q = 2; q * q <= m; ++q)
        if (m % q == 0) {
            fs.push_back(q);
            while (m % q == 0) m /= q;
        }
    if (m > 1) fs.push_back(m);
    for (std::int64_t g = 2;; ++g) {
        bool ok = true;
        for (std::int64_t q : fs) ok = ok && powmod(g, (p - 1) / q, p) != 1;
        if (ok) return g;
    }
}

// Circular convolutions of length m with one fixed operand, as linear convolutions on
// vDSP's power-of-2 complex DFT (length L >= 2m) folded back: (c * q)[k] = lin[k] + lin[k + m].
class Convolver {
public:
    Convolver(const std::vector<double>& c) : m_(static_cast<int>(c.size())) {
        L_ = 16;
        while (L_ < 2 * m_) L_ <<= 1;
        fwd_ = vDSP_DFT_zop_CreateSetupD(nullptr, static_cast<vDSP_Length>(L_), vDSP_DFT_FORWARD);
        inv_ = vDSP_DFT_zop_CreateSetupD(fwd_, static_cast<vDSP_Length>(L_), vDSP_DFT_INVERSE);
        fr_.assign(L_, 0.0), fi_.assign(L_, 0.0), ar_.assign(L_, 0.0), ai_.assign(L_, 0.0), br_.assign(L_, 0.0),
            bi_.assign(L_, 0.0);
        std::copy(c.begin(), c.end(), ar_.begin());
        vDSP_DFT_ExecuteD(fwd_, ar_.data(), ai_.data(), fr_.data(), fi_.data());
    }
    ~Convolver() {
        vDSP_DFT_DestroySetupD(inv_);
        vDSP_DFT_DestroySetupD(fwd_);
    }
    Convolver(const Convolver&) = delete;
    Convolver& operator=(const Convolver&) = delete;
    // out[k] = sum_j c[j] q[(k - j) mod m]
    void apply(const std::vector<double>& q, std::vector<double>& out) {
        std::fill(ar_.begin(), ar_.end(), 0.0), std::fill(ai_.begin(), ai_.end(), 0.0);
        std::copy(q.begin(), q.end(), ar_.begin());
        vDSP_DFT_ExecuteD(fwd_, ar_.data(), ai_.data(), br_.data(), bi_.data());
        for (int k = 0; k < L_; ++k) {
            const double re = br_[k] * fr_[k] - bi_[k] * fi_[k], im = br_[k] * fi_[k] + bi_[k] * fr_[k];
            br_[k] = re, bi_[k] = im;
        }
        vDSP_DFT_ExecuteD(inv_, br_.data(), bi_.data(), ar_.data(), ai_.data());
        out.resize(m_);
        for (int k = 0; k < m_; ++k) out[k] = (ar_[k] + (k + m_ < L_ ? ar_[k + m_] : 0.0)) / L_;
    }

private:
    int m_, L_;
    vDSP_DFT_SetupD fwd_ = nullptr, inv_ = nullptr;
    std::vector<double> fr_, fi_, ar_, ai_, br_, bi_;
};

// The generating vector z (D components) of the N-point rule, N an odd prime: scipy's
// _cbc_lattice step for step (Nuyens and Cools 2006, Algorithm 2: the components
// z_s = g^w mod N, the kernel B_2(x) = x^2 - x + 1/6 permuted by the primitive root g
// into a circulant, each step one circular convolution and an argmin; weights 1, 1,
// 0.8, 0.64, ...). Components agree with a longer construction's first D (CBC is
// greedy), so a cached longer vector serves shorter requests.
inline std::vector<std::int64_t> construct(int D, std::int64_t N) {
    std::vector<std::int64_t> z(static_cast<std::size_t>(std::max(D, 1)), 1);
    if (D <= 1) return z;
    const int m = static_cast<int>((N - 1) / 2);
    const std::int64_t g = primitive_root(N);
    std::vector<std::int64_t> perm(m);
    perm[0] = 1;
    for (int j = 0; j + 1 < m; ++j) perm[j + 1] = mulmod(g, perm[j], N);
    std::vector<double> c(m);
    for (int j = 0; j < m; ++j) {
        perm[j] = std::min(N - perm[j], perm[j]);
        const double pn = static_cast<double>(perm[j]) / static_cast<double>(N);
        c[j] = pn * pn - pn + 1.0 / 6.0;
    }
    Convolver conv(c);
    std::vector<double> q(m, 1.0), out;
    int w = 0;
    double gm = 1.0;
    for (int s = 1; s < D; ++s) {
        if (s >= 3) gm *= 0.8;
        // reordered = c[w], c[w-1], ..., c[0], c[m-1], ..., c[w+1]
        for (int k = 0; k < m; ++k) {
            const int idx = k <= w ? w - k : m - 1 - (k - w - 1);
            q[k] *= 1.0 + gm * c[idx];
        }
        conv.apply(q, out);
        w = static_cast<int>(std::min_element(out.begin(), out.end()) - out.begin());
        z[s] = perm[w];
    }
    return z;
}

// The vectors already built, by N (the longest one for each), under a mutex: tasks of
// the pool ask concurrently. Two may build the same one (outside the lock); they are
// equal. At most kCacheEntries N's are kept (then the cache starts over): a few dozen
// int64 per entry.
inline constexpr std::size_t kCacheEntries = 512;
inline std::vector<std::int64_t> lattice(int D, std::int64_t N) {
    static std::mutex mu;
    static std::map<std::int64_t, std::vector<std::int64_t>> cache;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(N);
        if (it != cache.end() && static_cast<int>(it->second.size()) >= D)
            return std::vector<std::int64_t>(it->second.begin(), it->second.begin() + std::max(D, 1));
    }
    std::vector<std::int64_t> z = construct(D, N);
    std::lock_guard<std::mutex> lock(mu);
    if (cache.size() >= kCacheEntries) cache.clear();
    auto& slot = cache[N];
    if (slot.size() < z.size()) slot = z;
    return z;
}

}  // namespace cbc

// The chi variable of the t's QMC, r(w) = sqrt(P^-1(a, w) / a), a = nu / 2 (sd.special's
// gammaincinv, ~250 ns: four times the rest of a d = 5 evaluation). Per df, piecewise
// Chebyshev interpolants of log r replace it, in variables where log r is smooth and the
// variable itself is accurate:
//   w <= Phi(-2):  y = log w  (log r ~ (y + lgamma(a + 1)) / (2a) at the left end; an
//                  error in y moves log r by 1 / (2a) of it)
//   w >  Phi(-2):  z = Phi^-1(w)  (log r ~ log z on the right; fitted from the complement
//                  Phi(-z) with gammainccinv, since w = Phi(z) is a staircase there)
// In z the left end would not do: ndtri's few-ulp error at z = -8.8 is amplified by
// |z| / (2a) and cost 4e-14 at df = 3.
// Pieces of kChiNodes nodes, uniform in each variable, 8 of them doubled up to 512 until
// the a posteriori check passes: at every piece's ends and the kChiNodes - 1 points between
// its nodes, the whole path w -> (y or z) -> table -> r against gammaincinv at that same w,
// |r_table / r - 1| <= 1e-14. Otherwise (df below ~0.1, where r underflows at small w) the
// exact inverse is used.
//
// The tables' bias: r is off by at most ~1e-14 relative (checked at 33 points per piece of
// a 16-node interpolant; 1.4e-14 the worst of 512k dense samples, at df = 1), which moves
// each e_i = Phi((r b_i - ...) / L_ii) by at most 1e-14 |r b_i| phi / L_ii, so the
// probability by less than 1e-13 for d <= 100. Measured against the exact inverse on the
// same lattice (2e5 evaluations, d = 4..100, df = 1..30): at most 8.3e-17, the errors being
// of both signs. Far below any atol the QMC reaches. The tables cost ~0.3 ms per df and
// 15 ns per value against gammaincinv's 260 ns (d = 5, df = 3: 183 -> 37 ms per point).
inline constexpr int kChiNodes = 16;
inline constexpr double kChiLo = 0x1p-60, kChiHi = 1.0 - 0x1p-53;  // the clamp of w
inline constexpr double kChiSplitZ = -2.0;

struct Cheb {  // a piecewise Chebyshev series on [x0, x0 + pieces / inv_h]
    double x0 = 0.0, inv_h = 0.0;
    int pieces = 0;
    std::vector<double> c;
    double eval(double x) const {
        const double s = (x - x0) * inv_h;
        const int k = std::clamp(static_cast<int>(s), 0, pieces - 1);
        const double t = 2.0 * (s - k) - 1.0;
        const double* ck = c.data() + static_cast<std::size_t>(k) * kChiNodes;
        double b1 = 0.0, b2 = 0.0;
        for (int j = kChiNodes - 1; j >= 1; --j) {  // Clenshaw; the constant term added last
            const double b0 = 2.0 * t * b1 - b2 + ck[j];
            b2 = b1, b1 = b0;
        }
        return 0.5 * ck[0] + (t * b1 - b2);
    }
    // fit g on [x0, x1], then check(x, table) at the ends and midpoints of every piece
    template <class G, class C>
    bool fit(double x0_, double x1, G&& g, C&& check) {
        const double N = kChiNodes;
        for (int p = 8; p <= 512; p *= 2) {
            const double h = (x1 - x0_) / p;
            x0 = x0_, inv_h = 1.0 / h, pieces = p;
            c.assign(static_cast<std::size_t>(p) * kChiNodes, 0.0);
            bool ok = true;
            for (int k = 0; k < p && ok; ++k) {
                double f[kChiNodes];
                for (int m = 0; m < kChiNodes; ++m) {
                    f[m] = g(x0 + (k + 0.5 * (std::cos(kPi * (m + 0.5) / N) + 1.0)) * h);
                    ok = ok && std::isfinite(f[m]);
                }
                // the coefficients of f - mean(f): log r is ~ -14 at the left end, and the sums
                // of the raw values left eps |f| N ~ 2e-14 of noise in every coefficient
                double mean = 0.0;
                for (int m = 0; m < kChiNodes; ++m) mean += f[m];
                mean /= N;
                for (int j = 0; j < kChiNodes; ++j) {
                    double sum = 0.0;
                    for (int m = 0; m < kChiNodes; ++m) sum += (f[m] - mean) * std::cos(kPi * j * (m + 0.5) / N);
                    c[static_cast<std::size_t>(k) * kChiNodes + j] = 2.0 * sum / N + (j == 0 ? 2.0 * mean : 0.0);
                }
            }
            for (int k = 0; k < p && ok; ++k)
                for (int m = 0; m <= kChiNodes && ok; ++m)
                    ok = check(x0 + (k + 0.5 * (std::cos(kPi * m / N) + 1.0)) * h);
            if (ok) return true;
        }
        return false;
    }
};

struct ChiTable {
    double a = 0.0, wsplit = 0.0;
    bool exact = true;
    Cheb left, right;  // log r in y = log w (w <= wsplit), in z = Phi^-1(w) (w > wsplit)

    double exact_r(double w) const { return std::sqrt(special::Gammaincinv::f(a, w) / a); }
};

inline ChiTable chi_table(double nu) {
    ChiTable T;
    T.a = 0.5 * nu;
    const double a = T.a;
    auto ndtri = [](double w) { return special::distr_detail::one<special::Ndtri>(w); };
    auto good = [](double rt, double r) { return r > 1e-290 && std::isfinite(r) && std::fabs(rt - r) <= 1e-14 * r; };
    T.wsplit = Phi(kChiSplitZ);
    const double z0 = kChiSplitZ, z1 = ndtri(kChiHi);
    const double y0 = std::log(kChiLo), y1 = std::log(T.wsplit);
    const bool okl = T.left.fit(
        y0, y1, [&](double y) { return std::log(T.exact_r(std::exp(y))); },
        [&](double y) {
            const double w = std::clamp(std::exp(y), kChiLo, T.wsplit);
            return good(std::exp(T.left.eval(std::log(w))), T.exact_r(w));
        }
    );
    const bool okr =
        okl && T.right.fit(
                   z0, z1,
                   [&](double z) {
                       return 0.5 * std::log(
                                        (z <= 0.0 ? special::Gammaincinv::f(a, Phi(z))
                                                  : special::Gammainccinv::f(a, std::max(Phi(-z), 1.0 - kChiHi))) /
                                        a
                                    );
                   },
                   [&](double z) {
                       const double w = std::clamp(Phi(z), T.wsplit, kChiHi);
                       return good(std::exp(T.right.eval(ndtri(w))), T.exact_r(w));
                   }
               );
    T.exact = !(okl && okr);
    return T;
}

// r for E clamped uniforms u (in [kChiLo, kChiHi]); y and z are scratch of E values.
// Phi^-1 for every element (the right table's ~98%), the log for the left's alone.
inline void chi_r(const ChiTable& T, const double* u, double* r, double* y, double* z, int E);


struct Problem {
    int n = 0;
    double nu = kInf;
    const ChiTable* chi = nullptr;  // the t's chi table (nullptr or exact: gammaincinv)
    std::vector<double> L, b, inv_diag;
};

// Cholesky with Genz and Bretz's variable reordering (2002 §2; 2009 §4.1.3; scipy's
// _permuted_cholesky): at step i choose, among the remaining variables, the one
// whose conditional upper limit (b_j - sum_k L_jk y_k) / s_j has the smallest
// probability, then take y_i = E[Z | Z <= that limit] = -phi/Phi for the later
// steps. The innermost variables are then the least constraining ones, which
// leaves the integrand nearly constant in them. false when a pivot is not
// positive (not positive definite in floating point).
inline bool reorder(std::vector<double> S, std::vector<double> b, int n, double nu, Problem& P) {
    P.n = n, P.nu = nu;
    P.L.assign(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    auto Sx = [&](int r, int c) -> double& { return S[static_cast<std::size_t>(r) * n + c]; };
    auto Lx = [&](int r, int c) -> double& { return P.L[static_cast<std::size_t>(r) * n + c]; };
    for (int i = 0; i < n; ++i) {
        int best = -1;
        double best_p = kInf, best_s = 0.0, best_t = 0.0;
        for (int j = i; j < n; ++j) {
            double v = Sx(j, j), m = b[j];
            for (int k = 0; k < i; ++k) v -= Lx(j, k) * Lx(j, k), m -= Lx(j, k) * y[k];
            if (!(v > 0.0)) return false;
            const double s = std::sqrt(v), t = m / s, p = Phi(t);
            if (p < best_p || best < 0) best = j, best_p = p, best_s = s, best_t = t;
        }
        if (best != i) {  // swap variables i and best (the factor's finished columns move with them)
            for (int c = 0; c < n; ++c) std::swap(Sx(i, c), Sx(best, c));
            for (int r = 0; r < n; ++r) std::swap(Sx(r, i), Sx(r, best));
            for (int k = 0; k < i; ++k) std::swap(Lx(i, k), Lx(best, k));
            std::swap(b[i], b[best]);
        }
        Lx(i, i) = best_s;
        for (int j = i + 1; j < n; ++j) {
            double v = Sx(j, i);
            for (int k = 0; k < i; ++k) v -= Lx(j, k) * Lx(i, k);
            Lx(j, i) = v / best_s;
        }
        y[i] = -mills_lower(best_t);
    }
    P.b = std::move(b);
    P.inv_diag.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) P.inv_diag[i] = 1.0 / Lx(i, i);
    return true;
}

// ---- the integrand's Phi and Phi^-1
//
// sd.special's algorithms (Cephes ndtr's rationals with exp(-x^2 / 2) split as hi + lo; Wichura's
// AS241) in NEON lanes, two elements a vector: a pair computes the regions its two elements
// need (one when they share a region, the usual case) and selects. exp is array/kernels/vmath.hpp's
// (0.52 ULP against vvexp's 1), the tail's log log01 below (vvlog was 17% of the integrand).
// The chunk kernels branch per element on scalars after a vForce pass; on mixed regions the
// lanes measured about 1.25x (Phi) and 2.8x (Phi^-1) faster. Computing every region branch-free,
// interleaving four pairs, and loading the coefficients from memory all measured slower. Each
// element's value depends on that element alone (the odd last one takes both lanes).
namespace qv {

template <std::size_t N>
inline float64x2_t horner(float64x2_t x, const std::array<double, N>& c) {
    float64x2_t r = vdupq_n_f64(c[0]);
    for (std::size_t i = 1; i < N; ++i) r = vfmaq_f64(vdupq_n_f64(c[i]), r, x);
    return r;
}
template <std::size_t N>
inline float64x2_t horner1(float64x2_t x, const std::array<double, N>& c) {  // with a leading 1
    float64x2_t r = vaddq_f64(x, vdupq_n_f64(c[0]));
    for (std::size_t i = 1; i < N; ++i) r = vfmaq_f64(vdupq_n_f64(c[i]), r, x);
    return r;
}
inline bool any(uint64x2_t m) { return (vgetq_lane_u64(m, 0) | vgetq_lane_u64(m, 1)) != 0; }
inline bool all(uint64x2_t m) { return (vgetq_lane_u64(m, 0) & vgetq_lane_u64(m, 1)) != 0; }
inline float64x2_t splat(double c) { return vdupq_n_f64(c); }

// e^x for x <= 0 (NaN passes): the lanes' exp where x >= -kExpMax, libm's below (subnormal or 0)
inline float64x2_t exp_neg(float64x2_t x) {
    const uint64x2_t in = vcgeq_f64(x, splat(-vmath::kExpMax));  // false for NaN
    float64x2_t y = vmath::exp_lanes(vbslq_f64(in, x, splat(0.0)));
    if (!all(in)) [[unlikely]] {
        double a[2];
        vst1q_f64(a, y);
        if (vgetq_lane_u64(in, 0) == 0) a[0] = std::exp(vgetq_lane_f64(x, 0));
        if (vgetq_lane_u64(in, 1) == 0) a[1] = std::exp(vgetq_lane_f64(x, 1));
        y = vld1q_f64(a);
    }
    return y;
}

// Phi(x) = ndtr(x): 1/2 + erf(t) / 2 for |t| < 1 (t = x / sqrt 2), else y = exp(-x^2 / 2) P/Q(|t|) / 2
// (R/S from |t| = 8), y for x < 0 and 1 - y above. Each region's value is a quotient N / D;
// a pair that mixes regions selects N and D per lane and divides once.
inline float64x2_t phi(float64x2_t x) {
    using special::cephes::kNdtrP, special::cephes::kNdtrQ, special::cephes::kNdtrR, special::cephes::kNdtrS;
    using special::cephes::kNdtrT, special::cephes::kNdtrU;
    const float64x2_t t = vmulq_f64(x, splat(special::kSqrtHalf)), a = vabsq_f64(t);
    const uint64x2_t cen = vcltq_f64(a, splat(1.0));
    float64x2_t nc = splat(0.0), dc = splat(1.0);
    if (any(cen)) {  // erf(t) = t T(t^2) / U(t^2)
        const float64x2_t z = vmulq_f64(t, t);
        nc = vmulq_f64(t, horner(z, kNdtrT)), dc = horner1(z, kNdtrU);
        if (all(cen)) return vfmaq_f64(splat(0.5), splat(0.5), vdivq_f64(nc, dc));
    }
    const float64x2_t hi = vmulq_f64(x, x);
    const float64x2_t lo = vmulq_f64(splat(0.5), vfmaq_f64(vnegq_f64(hi), x, x));
    const float64x2_t e = exp_neg(vmulq_f64(splat(-0.5), hi));
    const uint64x2_t far = vcgeq_f64(a, splat(8.0));
    float64x2_t p = horner(a, kNdtrP), q = horner1(a, kNdtrQ);
    if (any(far)) [[unlikely]] {  // |x| >= 11.3
        p = vbslq_f64(far, horner(a, kNdtrR), p);
        q = vbslq_f64(far, horner1(a, kNdtrS), q);
    }
    // erfc = (e (p - p lo)) / q, 0 where e is 0 (also x = +-inf, where p, q are inf and lo NaN)
    const float64x2_t nt = vmulq_f64(e, vfmsq_f64(p, p, lo));
    const float64x2_t r = vdivq_f64(vbslq_f64(cen, nc, nt), vbslq_f64(cen, dc, q));
    float64x2_t y = vmulq_f64(splat(0.5), r);
    y = vbslq_f64(vceqq_f64(e, splat(0.0)), splat(0.0), y);
    const float64x2_t rt = vbslq_f64(vcltq_f64(x, splat(0.0)), y, vsubq_f64(splat(1.0), y));
    return vbslq_f64(cen, vfmaq_f64(splat(0.5), splat(0.5), r), rt);
}

// log x for x in [2^-1022, 1] (normal, positive; the integrand's clamped probabilities):
// fdlibm's e_log.c in lanes. x = 2^k m, m in [sqrt(1/2), sqrt 2), f = m - 1, s = f / (2 + f),
// log m = f - (f^2 / 2 - s (f^2 / 2 + R(s^2))), R its degree-14 minimax polynomial (Lg1..Lg7);
// k log 2 in two parts (fma where fdlibm rounds twice). Measured against Arb on 7000 points of
// (2^-1000, 1]: at most 2 units of 2^-53 relative (1 ULP); a measured figure, not a proof.
inline float64x2_t log01(float64x2_t x) {
    const uint64x2_t bits = vreinterpretq_u64_f64(x);
    // the mantissa as a double in [1, 2), then halved above sqrt 2 (k one more)
    const uint64x2_t mb =
        vorrq_u64(vandq_u64(bits, vdupq_n_u64(0x000fffffffffffffULL)), vdupq_n_u64(0x3ff0000000000000ULL));
    float64x2_t m = vreinterpretq_f64_u64(mb);
    int64x2_t k = vsubq_s64(vreinterpretq_s64_u64(vshrq_n_u64(bits, 52)), vdupq_n_s64(1023));
    const uint64x2_t big = vcgtq_f64(m, splat(1.4142135623730951));
    m = vbslq_f64(big, vmulq_f64(m, splat(0.5)), m);
    k = vsubq_s64(k, vreinterpretq_s64_u64(big));  // big is all ones (-1): k + 1
    const float64x2_t f = vsubq_f64(m, splat(1.0));
    const float64x2_t sv = vdivq_f64(f, vaddq_f64(splat(2.0), f));
    const float64x2_t z = vmulq_f64(sv, sv), w = vmulq_f64(z, z);
    static constexpr std::array<double, 3> kOdd = {
        1.531383769920937332e-01, 2.222219843214978396e-01, 3.999999999940941908e-01
    };
    static constexpr std::array<double, 4> kEven = {
        1.479819860511658591e-01, 1.818357216161805012e-01, 2.857142874366239149e-01, 6.666666666666735130e-01
    };
    const float64x2_t t1 = vmulq_f64(w, horner(w, kOdd));   // w (Lg2 + w (Lg4 + w Lg6))
    const float64x2_t t2 = vmulq_f64(z, horner(w, kEven));  // z (Lg1 + w (Lg3 + w (Lg5 + w Lg7)))
    const float64x2_t Rv = vaddq_f64(t2, t1);
    const float64x2_t hfsq = vmulq_f64(splat(0.5), vmulq_f64(f, f));
    const float64x2_t dk = vcvtq_f64_s64(k);
    // k ln2_hi - ((hfsq - (s (hfsq + R) + k ln2_lo)) - f)
    const float64x2_t inner = vfmaq_f64(vmulq_f64(dk, splat(1.90821492927058770002e-10)), sv, vaddq_f64(hfsq, Rv));
    return vfmaq_f64(vsubq_f64(f, vsubq_f64(hfsq, inner)), dk, splat(6.93147180369123816490e-01));
}

// Phi^-1(p) = ndtri(p): q = p - 1/2, q A/B(0.180625 - q^2) for |q| <= 0.425, else from
// L = log(min(p, 1 - p)) t = C/D(s - 1.6) (s = sqrt(-L) <= 5) or E/F(s - 5), -t for q < 0.
// p in [2^-1022, 1 - 2^-53] (the integrand's clamp). One division a pair, as phi.
inline float64x2_t ndtri(float64x2_t p) {
    namespace as = special::as241;
    const float64x2_t q = vsubq_f64(p, splat(0.5));
    const uint64x2_t cen = vcleq_f64(vabsq_f64(q), splat(as::kSplit1));
    float64x2_t nc = splat(0.0), dc = splat(1.0);
    if (any(cen)) {
        const float64x2_t r = vfmsq_f64(splat(as::kConst1), q, q);
        nc = horner(r, as::kA), dc = horner(r, as::kB);
        if (all(cen)) return vmulq_f64(q, vdivq_f64(nc, dc));
    }
    // 1 - p is exact for p >= 1/2; a central lane's value here is not used
    const float64x2_t L = log01(vminq_f64(p, vsubq_f64(splat(1.0), p)));
    const float64x2_t s = vsqrtq_f64(vnegq_f64(L));
    const uint64x2_t near = vcleq_f64(s, splat(as::kSplit2));
    const float64x2_t s1 = vsubq_f64(s, splat(as::kConst2));
    float64x2_t nt = horner(s1, as::kC), dt = horner(s1, as::kD);
    if (!all(near)) [[unlikely]] {  // s > 5: p < 1.4e-11
        const float64x2_t s2 = vsubq_f64(s, splat(as::kSplit2));
        nt = vbslq_f64(near, nt, horner(s2, as::kE)), dt = vbslq_f64(near, dt, horner(s2, as::kF));
    }
    const float64x2_t r = vdivq_f64(vbslq_f64(cen, nc, nt), vbslq_f64(cen, dc, dt));
    const float64x2_t rt = vbslq_f64(vcltq_f64(q, splat(0.0)), vnegq_f64(r), r);
    return vbslq_f64(cen, vmulq_f64(q, r), rt);
}

// o[i] = Phi(x[i]), i < n
inline void phi(const double* x, double* o, int n) {
    int i = 0;
    for (; i + 2 <= n; i += 2) vst1q_f64(o + i, phi(vld1q_f64(x + i)));
    if (i < n) o[i] = vgetq_lane_f64(phi(vdupq_n_f64(x[i])), 0);
}

// o[i] = Phi^-1(p[i]), i < n, p[i] in [2^-1022, 1 - 2^-53]
inline void ndtri(const double* p, double* o, int n) {
    int i = 0;
    for (; i + 2 <= n; i += 2) vst1q_f64(o + i, ndtri(vld1q_f64(p + i)));
    if (i < n) o[i] = vgetq_lane_f64(ndtri(vdupq_n_f64(p[i])), 0);
}

}  // namespace qv

inline void chi_r(const ChiTable& T, const double* u, double* r, double* y, double* z, int E) {
    if (T.exact) {
        for (int q = 0; q < E; ++q) r[q] = T.exact_r(u[q]);
        return;
    }
    static_cast<void>(y);
    qv::ndtri(u, z, E);
    for (int q = 0; q < E; ++q) r[q] = u[q] <= T.wsplit ? T.left.eval(std::log(u[q])) : T.right.eval(z[q]);
    vmath::exp(r, r, E);
}

// Scratch for one task: the coordinates, y's and products of a block.
struct QmcWork {
    std::vector<double> W, Y, f, e, arg, p, r;
    std::vector<std::int64_t> k;
    void size(int D, int n) {
        const std::size_t E = kQmcBlock;
        W.resize(E * static_cast<std::size_t>(std::max(D, 1)));
        Y.resize(E * static_cast<std::size_t>(std::max(n, 1)));
        f.resize(E), e.resize(E), arg.resize(E), p.resize(E), r.resize(E);
        k.resize(static_cast<std::size_t>(std::max(D, 1)));
    }
};

// The integrand summed over the points [i0, i1) of the N-point rule z shifted by Dm.
inline double qmc_sum(
    const Problem& P,
    const std::int64_t* z,
    std::int64_t N,
    const double* Dm,
    std::int64_t i0,
    std::int64_t i1,
    QmcWork& w
) {
    const int n = P.n;
    const bool t = std::isfinite(P.nu);
    const int D = t ? n : n - 1;  // the t's coordinate 0 is the chi variable
    const int off = t ? 1 : 0;    // coordinate of y_0's uniform is off
    w.size(D, n);
    const double* L = P.L.data();
    const double e0 = t ? 0.0 : Phi(P.b[0] * P.inv_diag[0]);
    const double half_nu = 0.5 * P.nu, invN = 1.0 / static_cast<double>(N);
    for (int j = 0; j < D; ++j) w.k[j] = cbc::mulmod(i0 % N, z[j] % N, N);  // i z_j mod N, then stepped
    double total = 0.0;
    for (std::int64_t blk = i0; blk < i1; blk += kQmcBlock) {
        const int E = static_cast<int>(std::min<std::int64_t>(kQmcBlock, i1 - blk));
        for (int j = 0; j < D; ++j) {
            double* Wj = w.W.data() + static_cast<std::size_t>(j) * kQmcBlock;
            std::int64_t k = w.k[j];
            const std::int64_t zj = z[j] % N;
            const double dj = Dm[j];
            // two points a vector: i z_j mod N for i, i + 1, stepped by 2 z_j mod N (the same
            // operations as one point at a time)
            const std::int64_t k1 = k + zj >= N ? k + zj - N : k + zj, z2 = 2 * zj >= N ? 2 * zj - N : 2 * zj;
            int64x2_t kv = vcombine_s64(vdup_n_s64(k), vdup_n_s64(k1));
            const int64x2_t z2v = vdupq_n_s64(z2), Nv = vdupq_n_s64(N);
            const float64x2_t inv = vdupq_n_f64(invN), dv = vdupq_n_f64(dj), one = vdupq_n_f64(1.0);
            int q = 0;
            for (; q + 2 <= E; q += 2) {
                float64x2_t x = vaddq_f64(vmulq_f64(vcvtq_f64_s64(kv), inv), dv);
                x = vbslq_f64(vcgeq_f64(x, one), vsubq_f64(x, one), x);
                vst1q_f64(Wj + q, vabsq_f64(vsubq_f64(vmulq_f64(vdupq_n_f64(2.0), x), one)));
                kv = vaddq_s64(kv, z2v);
                kv = vbslq_s64(vcgeq_s64(kv, Nv), vsubq_s64(kv, Nv), kv);
            }
            k = vgetq_lane_s64(kv, 0);
            for (; q < E; ++q) {
                double x = static_cast<double>(k) * invN + dj;
                if (x >= 1.0) x -= 1.0;
                Wj[q] = std::fabs(2.0 * x - 1.0);
                k += zj;
                if (k >= N) k -= N;
            }
            w.k[j] = k;
        }
        double* f = w.f.data();
        double* e = w.e.data();
        double* arg = w.arg.data();
        double* p = w.p.data();
        double* r = w.r.data();
        if (t) {
            const double* W0 = w.W.data();
            if (P.chi) {
                for (int q = 0; q < E; ++q) p[q] = std::clamp(W0[q], kChiLo, kChiHi);
                chi_r(*P.chi, p, r, arg, e, E);
            } else {
                for (int q = 0; q < E; ++q) {
                    const double u0 = std::clamp(W0[q], kChiLo, kChiHi);
                    r[q] = std::sqrt(special::Gammaincinv::f(half_nu, u0) / half_nu);
                }
            }
            for (int q = 0; q < E; ++q) arg[q] = r[q] * P.b[0] * P.inv_diag[0];
            qv::phi(arg, e, E);
            for (int q = 0; q < E; ++q) f[q] = e[q];
        } else {
            for (int q = 0; q < E; ++q) r[q] = 1.0, e[q] = e0, f[q] = e0;
        }
        for (int i = 1; i < n; ++i) {
            const double* Wi = w.W.data() + static_cast<std::size_t>(i - 1 + off) * kQmcBlock;
            double* Yp = w.Y.data() + static_cast<std::size_t>(i - 1) * kQmcBlock;
            for (int q = 0; q < E; ++q) p[q] = std::clamp(Wi[q] * e[q], 0x1p-1000, 1.0 - 0x1p-53);
            qv::ndtri(p, Yp, E);
            const double bi = P.b[i], id = P.inv_diag[i];
            for (int q = 0; q < E; ++q) arg[q] = r[q] * bi;
            for (int j = 0; j < i; ++j) {
                const double l = L[static_cast<std::size_t>(i) * n + j];
                const double* Yj = w.Y.data() + static_cast<std::size_t>(j) * kQmcBlock;
                for (int q = 0; q < E; ++q) arg[q] -= l * Yj[q];
            }
            for (int q = 0; q < E; ++q) arg[q] *= id;
            qv::phi(arg, e, E);
            for (int q = 0; q < E; ++q) f[q] *= e[q];
        }
        double s = 0.0;
        for (int q = 0; q < E; ++q) s += f[q];
        total += s;
    }
    return total;
}

struct Estimate {
    double value, error;
    std::int64_t evals;
};

struct Neumaier {
    double s = 0.0, c = 0.0;
    void add(double x) {
        const double t = s + x;
        c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    double value() const { return s + c; }
};

// The QMC estimate of a reordered problem (n >= 2), level by level: QmcRun holds one
// problem's state; a level is prepare() (its N's lattice and shifts), the integrand over its
// items (shift, piece of kQmcItem points: item(it), any order, on any thread) and finish()
// (the pieces added per shift in order, the levels combined, the stopping test). qmc() runs
// one problem with each level's items on the pool -- serially when the caller is itself a
// task; mv_cdf (bind/stats_mvcdf.cpp) runs the levels of all its points together, their items
// on the pool at once. Either way the result is a function of the problem alone: the items'
// values do not depend on where they run, and finish() reads them in a fixed order, so the
// stopping decision and every bit are the same on any thread count (guardrail d).
//
// Nearly singular problems (a variable whose conditional standard deviation given the
// ones before it is below kQmcHardRatio of its own: R^2 > 0.9975) have integrands with a
// thin region that most shifted lattices miss: twelve shifts that all miss it agree with
// each other to 1e-8 and are all off by 2e-6 (measured at kappa = 2e5, N = 1009: the
// estimate was 29 times too small, and its weight 1 / var kept every later level out).
// For those problems only: at least kQmcHardLevels levels; a level's variance is never
// taken below a finer level's scaled by the ratio of their sizes (a coarser rule is not
// the more accurate one), so one lucky level cannot carry the result; and never below
// (1 / (36 N))^2, the resolution of 12 N bounded values (so the error estimate is at
// least 1 / (12 N): shifts that all returned 0 said "0, error 0" where the truth was 3e-6).
inline constexpr double kQmcHardRatio = 0.05;
inline constexpr int kQmcHardLevels = 3;
inline constexpr int kQmcMaxLevels = 96;  // N grows by 1.5 at least: more than max_evals = 2^63 allows

class QmcRun {
public:
    QmcRun(const Problem& P, double atol, std::int64_t max_evals)
        : P_(&P),
          atol_(atol),
          max_evals_(max_evals),
          n_(P.n),
          D_(std::isfinite(P.nu) ? P.n : P.n - 1),
          key_{kShiftKey, static_cast<std::uint64_t>(P.n)} {
        for (int i = 1; i < n_; ++i) {
            double s2 = 0.0;
            for (int k = 0; k <= i; ++k)
                s2 += P.L[static_cast<std::size_t>(i) * n_ + k] * P.L[static_cast<std::size_t>(i) * n_ + k];
            hard_ = hard_ || P.L[static_cast<std::size_t>(i) * n_ + i] < kQmcHardRatio * std::sqrt(s2);
        }
        N_ = cbc::next_prime(kQmcStart);
        if (kShifts * N_ > max_evals) N_ = cbc::prev_prime(std::max<std::int64_t>(3, max_evals / kShifts));
        shifts_.resize(static_cast<std::size_t>(kShifts) * D_);
    }
    bool done() const { return done_; }
    const Estimate& result() const { return res_; }
    // the level's lattice and shifts (thread-safe across runs: the lattice cache locks)
    void prepare() {
        z_ = cbc::lattice(D_, N_);
        for (int m = 0; m < kShifts; ++m)
            for (int j = 0; j < D_; j += 4) {
                const random::Block blk = random::philox(
                    {static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(j >> 2),
                     static_cast<std::uint64_t>(level_), 0},
                    key_
                );
                for (int q = 0; q < 4 && j + q < D_; ++q)
                    shifts_[static_cast<std::size_t>(m) * D_ + j + q] = random::to_unit(blk[q]);
            }
        per_ = (N_ + kQmcItem - 1) / kQmcItem;
    }
    std::int64_t items() const { return kShifts * per_; }
    std::int64_t work() const { return items() * kQmcItem * n_ * 8; }  // for the pool's threshold
    // the integrand summed over item it of the level
    double item(std::int64_t it, QmcWork& w) const {
        const std::int64_t m = it / per_, k = it % per_;
        const std::int64_t lo = k * kQmcItem, hi = std::min(N_, lo + kQmcItem);
        return qmc_sum(*P_, z_.data(), N_, shifts_.data() + static_cast<std::size_t>(m) * D_, lo, hi, w);
    }
    // the level's items' values (items() of them, in order): the estimate so far, and done()
    // or the next level's N
    void finish(const double* part) {
        const std::int64_t N = N_, per = per_;
        const int level = level_;
        double mu[kShifts], mean = 0.0;
        for (int m = 0; m < kShifts; ++m) {
            Neumaier s;
            for (std::int64_t k = 0; k < per; ++k) s.add(part[m * per + k]);
            mu[m] = s.value() / static_cast<double>(N), mean += mu[m];
        }
        mean /= kShifts;
        double var = 0.0;
        for (int m = 0; m < kShifts; ++m) var += (mu[m] - mean) * (mu[m] - mean);
        var /= static_cast<double>(kShifts) * (kShifts - 1);
        evals_ += kShifts * N;
        if (!std::isfinite(mean)) return stop({kNaN, kNaN, evals_});
        if (hard_) {
            // 12 N values in [0, 1] cannot rule out a region of volume ~1 / (12 N): shifts that
            // agree (or all see 0) are no evidence below that
            const double floor = 1.0 / (static_cast<double>(kShifts) * static_cast<double>(N));
            var = std::max(var, floor * floor / 9.0);
        }
        if (var == 0.0) return stop({std::clamp(mean, 0.0, 1.0), 0.0, evals_});  // a constant integrand
        if (hard_) {
            lvar_[level] = var, lmean_[level] = mean, lN_[level] = static_cast<double>(N);
            prec_ = 0.0, wsum_ = 0.0;
            for (int k = 0; k <= level; ++k) {
                const double vk = std::max(lvar_[k], var * lN_[level] / lN_[k]);
                prec_ += 1.0 / vk, wsum_ += lmean_[k] / vk;
            }
        } else {
            prec_ += 1.0 / var, wsum_ += mean / var;
        }
        const double value = wsum_ / prec_, err = 3.0 / std::sqrt(prec_);
        const std::int64_t room = (max_evals_ - evals_) / kShifts;
        const bool enough = !hard_ || level + 1 >= kQmcHardLevels;
        if ((!(err > atol_) && enough) || room < kQmcStart || level + 1 == kQmcMaxLevels)
            return stop({std::clamp(value, 0.0, 1.0), err, evals_});
        // the next level alone should reach atol: its error ~ N^-1 (lattice rules with
        // the baker's transformation do at least that on these integrands)
        const double lvl = 3.0 * std::sqrt(var);
        const double g = std::clamp(1.2 * lvl / atol_, 1.5, 8.0);
        const std::int64_t want = static_cast<std::int64_t>(std::ceil(static_cast<double>(N) * g));
        N_ = want <= room ? cbc::next_prime(want) : cbc::prev_prime(room);
        if (N_ > room) N_ = cbc::prev_prime(room);
        ++level_;
    }

private:
    void stop(Estimate e) { res_ = e, done_ = true; }
    const Problem* P_;
    double atol_;
    std::int64_t max_evals_;
    int n_, D_;
    random::Key key_;
    bool hard_ = false, done_ = false;
    int level_ = 0;
    std::int64_t N_ = 0, per_ = 0, evals_ = 0;
    double prec_ = 0.0, wsum_ = 0.0;  // the combination: sum 1 / var_l and sum v_l / var_l
    double lvar_[kQmcMaxLevels] = {}, lmean_[kQmcMaxLevels] = {}, lN_[kQmcMaxLevels] = {};
    std::vector<std::int64_t> z_;
    std::vector<double> shifts_;
    Estimate res_{kNaN, kNaN, 0};
};

inline Estimate qmc(const Problem& P, double atol, std::int64_t max_evals) {
    QmcRun run(P, atol, max_evals);
    std::vector<double> part;
    while (!run.done()) {
        run.prepare();
        const std::int64_t items = run.items();
        part.assign(static_cast<std::size_t>(items), 0.0);
        for_pieces(items, 1, run.work(), [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
            QmcWork w;
            for (std::int64_t it = b0; it < b1; ++it) part[static_cast<std::size_t>(it)] = run.item(it, w);
        });
        run.finish(part.data());
    }
    return run.result();
}

}  // namespace sd::stats::mvcdf
