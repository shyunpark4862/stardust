// The cumulative distribution functions of the multivariate normal and t
// (T9; docs/design/stats.md §6.2, decisions T16 and T29): P(X <= x) with every
// lower limit -inf, for X ~ N(0, Sigma) or X = Z / sqrt(S / nu) (Z ~ N(0, Sigma),
// S ~ chi^2_nu independent). The caller has subtracted the mean.
//
//   normal, d = 1   Phi (sd.special's ndtr)
//   normal, d = 2   Plackett's identity integrated in an angle (bvn below)
//   normal, d = 3   Plackett's identity along a path of correlations, or
//                   conditioning on one variable where that cancels (tvn below)
//   normal, d >= 4  Genz's quasi-Monte Carlo (qmc below)
//   t,      d = 1   sd.special's stdtr
//   t,      d = 2, 3  the scale mixture E[Phi_d(R h)], one more 1-D quadrature
//                   around bvn / tvn (bvt, tvt below)
//   t,      d >= 4  Genz's quasi-Monte Carlo with the chi variable
//
// The exact formulas (d <= 3) are written so that the sums they form do not
// cancel: every integrand has a fixed sign, and the starting point of each
// integral is chosen so that the integral adds to it. That keeps the relative
// accuracy in the lower tails, where Genz's TVPACK (bvnd, tvnd) reaches ~1e-15
// absolute only. Measured (bench/stats/t9/cdf_accuracy.md): the relative error is
// within a few units of eps times the problem's condition number (the sum of
// |d log P / d log x| over the inputs), which is ~ -log P in the tails.
#pragma once

#include "array/ops/pieces.hpp"
#include "parallel/pool.hpp"
#include "random/philox.hpp"
#include "special/ops/distr.hpp"
#include "special/ops/erf.hpp"
#include "special/ops/incgamma.hpp"

#include <algorithm>
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

// phi(t) / Phi(t) (the inverse Mills ratio of the lower tail) without underflow:
// sqrt(2 / pi) / erfcx(-t / sqrt 2), since Phi(t) = erfc(-t / sqrt 2) / 2.
inline double mills_lower(double t) { return kSqrt2OverPi / special::Erfcx::f(-t * kSqrtHalf); }

// ------------------------------------------------- adaptive Gauss-Kronrod

// QUADPACK's 21-point Kronrod rule and its embedded 10-point Gauss rule (qk21;
// Piessens, de Doncker, Ueberhuber, Kahaner, "QUADPACK", 1983): the nodes in
// (0, 1) from the outside in, then 0; the Gauss nodes are xgk[1], xgk[3], ..., xgk[9].
namespace gk {
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
//
// delta is the width of a step at w = 0: the integrands below behave there like
// exp(-delta^2 / (2 w^2)), 0 below w ~ delta and smooth above, varying on the scale
// of w itself. A first panel whose nodes all lie above the step cannot see it (it
// lost 2e-8 relative at h + k = 1e-8), so the first panels are geometric from
// max(a, delta): [a, c], [c, 4c], [4c, 16c], ..., each resolving the scale of its w.
// A step below 1e-18 wide holds less than 1e-18 of the integral and is ignored.
template <class F>
double adapt(F& f, Panel* p, int n, double base) {
    for (;;) {
        double v = 0.0, e = 0.0;
        int worst = 0;
        for (int i = 0; i < n; ++i) {
            v += p[i].val, e += p[i].err;
            if (p[i].err > p[worst].err) worst = i;
        }
        if (!(e > kQuadRel * (std::fabs(base) + std::fabs(v))) || n == kMaxPanels) return v;
        const double lo = p[worst].a, hi = p[worst].b, m = 0.5 * (lo + hi);
        if (!(lo < m && m < hi)) return v;  // the panel cannot be split further
        p[worst] = gk21(f, lo, m);
        p[n++] = gk21(f, m, hi);
    }
}

// The first panels for a step of width delta at 0 (above), then adapt.
template <class F>
double integrate(F&& f, double a, double b, double base, double delta = 0.0) {
    Panel p[kMaxPanels];
    int n = 0;
    double c = delta >= 1e-18 && delta < b ? std::max(a, delta) : a;
    if (c > a) p[n++] = gk21(f, a, c);
    while (c > 0.0 && 4.0 * c < b && n < kMaxPanels / 2) {
        p[n++] = gk21(f, c, 4.0 * c);
        c *= 4.0;
    }
    p[n++] = gk21(f, c, b);
    return adapt(f, p, n, base);
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

// The angle variable. Plackett's identity (Plackett 1954; Drezner and Wesolowsky
// 1990, "On the computation of the bivariate normal integral"; Genz 2004,
// "Numerical computation of rectangular bivariate and trivariate normal and t
// probabilities", Stat. Comput. 14) is
//     d Phi2(h, k; r) / dr = exp(-q(r)) / (2 pi sqrt(1 - r^2)),
//     q(r) = (h^2 - 2 r h k + k^2) / (2 (1 - r^2)),
// and r = sin(theta) removes the square root: d Phi2 / d theta = exp(-q) / (2 pi).
// The integrals below run in w, the angle from the nearer pole: s = r = cos w
// (r >= 0) or -cos w (r <= 0), w in [0, pi/2], so that 1 - s and 1 + s are both
// accurate (2 sin^2(w / 2) on the near side) even where s is within 1e-16 of +-1.
struct Sides {
    double s, sm, sp;  // s, 1 - s, 1 + s
};
inline Sides sides(double w, bool minus) {
    const double c = std::cos(w), h = std::sin(0.5 * w);
    const double near = 2.0 * h * h, far = 1.0 + c;
    return minus ? Sides{-c, far, near} : Sides{c, near, far};
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

// The integral of exp(-q) over w in [a, b] on the given side, added to base.
inline double bvn_integral(double h, double k, bool minus, double a, double b, double base) {
    auto f = [&](const double* w, double* o, int n) {
        for (int i = 0; i < n; ++i) {
            const Sides s = sides(w[i], minus);
            o[i] = std::exp(-qform(h, k, s.sm, s.sp));
        }
    };
    // the step at the pole w = 0: s -> 1 (plus side) has width |h - k|, s -> -1 |h + k|
    return integrate(f, a, b, base * kTwoPi, std::fabs(minus ? h + k : h - k));
}

// Below this r < 0 the integral starts at r = -1 directly (the start at 0 cancels).
inline constexpr double kBvnSwitch = -0.5;

// Phi2(h, k; r) = P(X <= h, Y <= k), unit variances, finite h and k, |r| <= 1.
//     r >= 0:  Phi(h) Phi(k) + (1 / 2pi) int_0^asin r exp(-q) dtheta
//              (both terms >= 0)
//     r <  0:  Phi(h) Phi(k) - (1 / 2pi) int_asin r^0 exp(-q) dtheta
//              when that loses at most two bits (r > -1/2 only), else
//              Phi2(h, k; -1) + (1 / 2pi) int_-pi/2^asin r exp(-q) dtheta
//              (both terms >= 0 again)
inline double bvn(double h, double k, double r) {
    if (std::isnan(h) || std::isnan(k) || std::isnan(r)) return kNaN;
    const double ph = Phi(h), pk = Phi(k);
    if (r == 0.0) return ph * pk;
    r = std::clamp(r, -1.0, 1.0);
    if (r > 0.0) {
        const double p0 = ph * pk;
        return std::min(1.0, p0 + bvn_integral(h, k, false, std::acos(r), kHalfPi, p0) / kTwoPi);
    }
    const double wr = std::acos(-r);  // (0, pi/2]: theta = asin r is w = wr on the minus side
    if (r > kBvnSwitch) {
        const double p0 = ph * pk;
        const double v = p0 - bvn_integral(h, k, true, wr, kHalfPi, p0) / kTwoPi;
        if (p0 <= 4.0 * v) return v;
    }
    const double pm = bvn_minus_one(h, k);
    return std::min(1.0, pm + bvn_integral(h, k, true, 0.0, wr, pm) / kTwoPi);
}

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
    const double ratio = c2 / c1, oma = (1.0 - a) * (1.0 + a);
    auto f = [&](const double* w, double* o, int n) {
        double u[21], ph[21];
        for (int i = 0; i < n; ++i) {
            const Sides sd = sides(w[i], minus);
            const double s = sd.s, oms = sd.sm * sd.sp, b = s * ratio;
            const double g = b - a * s, det = oma * oms - g * g;
            const double num = hj * oms - a * (hi - s * hk) - b * (hk - s * hi);
            u[i] = det > 0.0 ? num / std::sqrt(oms * det) : (num > 0.0 ? kInf : (num < 0.0 ? -kInf : 0.0));
            o[i] = std::exp(-qform(hi, hk, sd.sm, sd.sp));
        }
        special::Ndtr::chunk(u, ph, n);
        for (int i = 0; i < n; ++i) o[i] *= ph[i];
    };
    const double J = integrate(f, std::acos(std::fabs(c1)), kHalfPi, base * kTwoPi, std::fabs(minus ? hi + hk : hi - hk));
    return minus ? -J : J;
}

struct Tri {
    double h[3];
    double r[3][3];
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
// until the part left out, at most Phi(h_k - L) min(Phi(a), Phi(b)) at h_k - L (Phi2 <=
// min(Phi(a), Phi(b)), and a(z) is increasing in z when r_ik < 0), is below 1e-17 of
// the integral.
inline double tvn_condition(const Tri& T) {
    int k = 0;
    for (int q = 1; q < 3; ++q)
        if (T.h[q] < T.h[k]) k = q;
    const int i = k == 0 ? 1 : 0, j = 3 - k - i;
    const double rik = T.r[i][k], rjk = T.r[j][k];
    const double si = std::sqrt((1.0 - rik) * (1.0 + rik)), sj = std::sqrt((1.0 - rjk) * (1.0 + rjk));
    const double rho = std::clamp((T.r[i][j] - rik * rjk) / (si * sj), -1.0, 1.0);
    const double hi = T.h[i], hj = T.h[j], hk = T.h[k];
    constexpr double kInvSqrt2Pi = 0.3989422804014326779;
    auto f = [&](const double* z, double* o, int n) {
        for (int q = 0; q < n; ++q)
            o[q] = kInvSqrt2Pi * std::exp(-0.5 * z[q] * z[q]) * bvn((hi - rik * z[q]) / si, (hj - rjk * z[q]) / sj, rho);
    };
    for (double L = 10.0;; L *= 2.0) {
        const double lo = hk - L;
        double cuts[7];
        for (int q = 0; q < 7; ++q) cuts[q] = lo + L * (q + 1) / 8.0;
        const double I = integrate_at(f, lo, hk, cuts, 7, 0.0);
        double tail = 1.0;
        if (rik < 0.0) tail = std::min(tail, Phi((hi - rik * lo) / si));
        if (rjk < 0.0) tail = std::min(tail, Phi((hj - rjk * lo) / sj));
        if (Phi(lo) * tail <= 1e-17 * I || L >= 160.0) return I;
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
            if (negs == 2 ? v >= 0.0
                          : (best < 0 || std::fabs(v) > std::fabs(T.r[kPairs[best][0]][kPairs[best][1]])))
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
//     P(X <= h) = E[Phi_d(R h)] = int_0^inf Gamma(a)^-1 x^(a-1) e^-x Phi_d(sqrt(x / a) h) dx,
// a = nu / 2, x = nu R^2 / 2 (Genz and Bretz 2009 §4.2 write the same mixture over the
// chi variable). With Phi_d by bvn or tvn the only error is that of one more 1-D
// quadrature, in v = log(x / a), where the weight is
//     exp(c(a) - a (e^v - 1 - v)),  c(a) = a log a - a - lgamma(a):
// written this way it has no cancellation for large a (a v - e^v - lgamma(a) directly
// loses eps a log a: 7e-10 at nu = 1e6), and it peaks at v = 0 with width
// ~ 1 / sqrt(a), or a left tail ~ e^(a v) for small a.
//
// c(a) for a >= 10 by Stirling's series (lgamma(a) = (a - 1/2) log a - a + log(2 pi)/2 +
// sum B_2k / (2k (2k-1) a^(2k-1)); seven terms, the next < 3e-17 at a = 10).
inline double stirling_c(double a) {
    if (a < 10.0) return a * std::log(a) - a - std::lgamma(a);
    const double r = 1.0 / a, r2 = r * r;
    const double s = r * (1.0 / 12 + r2 * (-1.0 / 360 + r2 * (1.0 / 1260 + r2 * (-1.0 / 1680 + r2 * (1.0 / 1188 +
                                                                                                     r2 * (-691.0 / 360360 + r2 * (1.0 / 156)))))));
    return 0.5 * std::log(a / kTwoPi) - s;
}

// e^v - 1 - v without cancellation (Taylor below |v| = 1/2: 20 terms, the next < 1e-26).
inline double expm1mx(double v) {
    if (std::fabs(v) >= 0.5) return std::expm1(v) - v;
    double t = 1.0;  // v^2/2 (1 + v/3 (1 + v/4 (1 + ...)))
    for (int k = 21; k >= 3; --k) t = 1.0 + v / k * t;
    return 0.5 * v * v * t;
}

// E[Phi_d(R h)] for Phi_d = g (a function of the scale r), the orthant p0 = Phi_d(0) and
// sum |h_i| = habs. The left end: for r <= r_c, |Phi_d(r h) - p0| <= r habs / sqrt(2 pi)
// (each partial derivative of Phi_d is at most phi(0)), so with r_c = 1e-17 p0 / (0.4 habs)
// that part is p0 P(R <= r_c) to 1e-17 relative (sd.special's gammainc). The right end at
// x = a + 40 sqrt(a) + 750 leaves out P(S > that) < 1e-300. The first panels are cut at
// v = +-sigma 4^k, sigma = min(1, 1 / sqrt(a)): the weight's peak and its tails at their
// own scales.
template <class G>
double t_mixture(double nu, double p0, double habs, G&& g) {
    if (habs == 0.0) return p0;
    const double a = 0.5 * nu;
    const double c = stirling_c(a);
    const double vhi = std::log1p((40.0 * std::sqrt(a) + 750.0) / a);
    const double rc = 1e-17 * p0 / (0.4 * habs);
    const double vc = 2.0 * std::log(rc);
    if (!(vc < vhi)) return p0;
    const double left = p0 * special::Gammainc::f(a, a * std::exp(vc));
    auto f = [&](const double* v, double* o, int n) {
        for (int i = 0; i < n; ++i) {
            const double w = std::exp(c - a * expm1mx(v[i]));
            o[i] = w == 0.0 ? 0.0 : w * g(std::exp(0.5 * v[i]));
        }
    };
    const double sigma = std::min(1.0, 1.0 / std::sqrt(a));
    double cuts[40];
    int nc = 0;
    for (double t = sigma * 0x1p40; t >= sigma && nc < 20; t *= 0.25)  // the negative side, outside in
        if (-t > vc) cuts[nc++] = -t;
    cuts[nc++] = 0.0;
    for (double t = sigma; t < vhi && nc < 40; t *= 4.0) cuts[nc++] = t;
    return std::min(1.0, left + integrate_at(f, vc, vhi, cuts, nc, left));
}

// The bivariate and trivariate t, nu finite: Phi_2(0) = 1/4 + asin(r) / (2 pi) and
// Phi_3(0) = 1/8 + (asin r12 + asin r13 + asin r23) / (4 pi) (Sheppard; Plackett 1954).
inline double bvt(double h, double k, double r, double nu) {
    const double p0 = 0.25 + std::asin(r) / kTwoPi;
    return t_mixture(nu, p0, std::fabs(h) + std::fabs(k), [&](double s) { return bvn(s * h, s * k, r); });
}
inline double tvt(const Tri& T, double nu) {
    const double p0 = 0.125 + (std::asin(T.r[0][1]) + std::asin(T.r[0][2]) + std::asin(T.r[1][2])) / (2.0 * kTwoPi);
    return t_mixture(nu, p0, std::fabs(T.h[0]) + std::fabs(T.h[1]) + std::fabs(T.h[2]), [&](double s) {
        Tri U = T;
        for (double& x : U.h) x *= s;
        return tvn(U);
    });
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
                for (int m = 0; m <= kChiNodes && ok; ++m) ok = check(x0 + (k + 0.5 * (std::cos(kPi * m / N) + 1.0)) * h);
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
        });
    const bool okr = okl && T.right.fit(
        z0, z1,
        [&](double z) {
            return 0.5 * std::log((z <= 0.0 ? special::Gammaincinv::f(a, Phi(z))
                                            : special::Gammainccinv::f(a, std::max(Phi(-z), 1.0 - kChiHi))) / a);
        },
        [&](double z) {
            const double w = std::clamp(Phi(z), T.wsplit, kChiHi);
            return good(std::exp(T.right.eval(ndtri(w))), T.exact_r(w));
        });
    T.exact = !(okl && okr);
    return T;
}

// r for E clamped uniforms u (in [kChiLo, kChiHi]); y and z are scratch of E values.
inline void chi_r(const ChiTable& T, const double* u, double* r, double* y, double* z, int E) {
    if (T.exact) {
        for (int q = 0; q < E; ++q) r[q] = T.exact_r(u[q]);
        return;
    }
    vvlog(y, u, &E);
    special::Ndtri::chunk(u, z, E);
    for (int q = 0; q < E; ++q) r[q] = std::exp(u[q] <= T.wsplit ? T.left.eval(y[q]) : T.right.eval(z[q]));
}

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
inline double qmc_sum(const Problem& P, const std::int64_t* z, std::int64_t N, const double* Dm, std::int64_t i0,
                      std::int64_t i1, QmcWork& w) {
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
            for (int q = 0; q < E; ++q) {
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
            special::Ndtr::chunk(arg, e, E);
            for (int q = 0; q < E; ++q) f[q] = e[q];
        } else {
            for (int q = 0; q < E; ++q) r[q] = 1.0, e[q] = e0, f[q] = e0;
        }
        for (int i = 1; i < n; ++i) {
            const double* Wi = w.W.data() + static_cast<std::size_t>(i - 1 + off) * kQmcBlock;
            double* Yp = w.Y.data() + static_cast<std::size_t>(i - 1) * kQmcBlock;
            for (int q = 0; q < E; ++q) p[q] = std::clamp(Wi[q] * e[q], 0x1p-1000, 1.0 - 0x1p-53);
            special::Ndtri::chunk(p, Yp, E);
            const double bi = P.b[i], id = P.inv_diag[i];
            for (int q = 0; q < E; ++q) arg[q] = r[q] * bi;
            for (int j = 0; j < i; ++j) {
                const double l = L[static_cast<std::size_t>(i) * n + j];
                const double* Yj = w.Y.data() + static_cast<std::size_t>(j) * kQmcBlock;
                for (int q = 0; q < E; ++q) arg[q] -= l * Yj[q];
            }
            for (int q = 0; q < E; ++q) arg[q] *= id;
            special::Ndtr::chunk(arg, e, E);
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

// The QMC estimate of a reordered problem (n >= 2). The work of a level is cut into
// (shift, piece of kQmcItem points) tasks on the pool -- serially when the caller is
// itself a task -- and the pieces are added per shift in order.
inline Estimate qmc(const Problem& P, double atol, std::int64_t max_evals) {
    const int n = P.n;
    const int D = std::isfinite(P.nu) ? n : n - 1;
    const random::Key key{kShiftKey, static_cast<std::uint64_t>(n)};
    std::vector<double> shifts(static_cast<std::size_t>(kShifts) * D);
    double prec = 0.0, wsum = 0.0;  // the combination: sum 1 / var_l and sum v_l / var_l
    std::int64_t evals = 0;
    std::int64_t N = cbc::next_prime(kQmcStart);
    if (kShifts * N > max_evals) N = cbc::prev_prime(std::max<std::int64_t>(3, max_evals / kShifts));
    std::vector<double> part;
    for (int level = 0;; ++level) {
        const std::vector<std::int64_t> z = cbc::lattice(D, N);
        for (int m = 0; m < kShifts; ++m)
            for (int j = 0; j < D; j += 4) {
                const random::Block blk = random::philox(
                    {static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(j >> 2), static_cast<std::uint64_t>(level), 0},
                    key);
                for (int q = 0; q < 4 && j + q < D; ++q) shifts[static_cast<std::size_t>(m) * D + j + q] = random::to_unit(blk[q]);
            }
        const std::int64_t per = (N + kQmcItem - 1) / kQmcItem;
        const std::int64_t items = kShifts * per;
        part.assign(static_cast<std::size_t>(items), 0.0);
        for_pieces(items, 1, items * kQmcItem * n * 8, [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
            QmcWork w;
            for (std::int64_t it = b0; it < b1; ++it) {
                const std::int64_t m = it / per, k = it % per;
                const std::int64_t lo = k * kQmcItem, hi = std::min(N, lo + kQmcItem);
                part[static_cast<std::size_t>(it)] =
                    qmc_sum(P, z.data(), N, shifts.data() + static_cast<std::size_t>(m) * D, lo, hi, w);
            }
        });
        double mu[kShifts], mean = 0.0;
        for (int m = 0; m < kShifts; ++m) {
            Neumaier s;
            for (std::int64_t k = 0; k < per; ++k) s.add(part[static_cast<std::size_t>(m * per + k)]);
            mu[m] = s.value() / static_cast<double>(N), mean += mu[m];
        }
        mean /= kShifts;
        double var = 0.0;
        for (int m = 0; m < kShifts; ++m) var += (mu[m] - mean) * (mu[m] - mean);
        var /= static_cast<double>(kShifts) * (kShifts - 1);
        evals += kShifts * N;
        if (!std::isfinite(mean)) return {kNaN, kNaN, evals};
        if (var == 0.0) return {std::clamp(mean, 0.0, 1.0), 0.0, evals};  // a constant integrand
        prec += 1.0 / var, wsum += mean / var;
        const double value = wsum / prec, err = 3.0 / std::sqrt(prec);
        const std::int64_t room = (max_evals - evals) / kShifts;
        if (!(err > atol) || room < kQmcStart) return {std::clamp(value, 0.0, 1.0), err, evals};
        // the next level alone should reach atol: its error ~ N^-1 (lattice rules with
        // the baker's transformation do at least that on these integrands)
        const double lvl = 3.0 * std::sqrt(var);
        const double g = std::clamp(1.2 * lvl / atol, 1.5, 8.0);
        const std::int64_t want = static_cast<std::int64_t>(std::ceil(static_cast<double>(N) * g));
        N = want <= room ? cbc::next_prime(want) : cbc::prev_prime(room);
        if (N > room) N = cbc::prev_prime(room);
    }
}

}  // namespace sd::stats::mvcdf
