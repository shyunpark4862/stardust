// The studentized range distribution of tukey_hsd (milestone H8, docs/design/htest.md §3.9;
// H14, H107: inside htest, not a public sd.stats distribution), and the root finder that turns
// an upper tail into a critical value (tukey_hsd's and dunnett's intervals).
//
// Q = W / R for the range W of k independent standard normals and R = sqrt(S / nu), S ~ chi^2_nu
// independent of them. Its upper tail is the scale mixture
//     P(Q > q) = E[P(W > R q)],
// taken by sd.stats' t mixture (stats/mvcdf.hpp t_mixture: the chi variable in v = log R^2,
// adaptive Gauss-Kronrod, deterministic), and the range's own upper tail is one integral over
// the smallest of the k values, z:
//     P(W > w) = k int phi(z) [A^(k-1) - (A - C)^(k-1)] dz,   A = Phi(-z), C = Phi(-(z + w))
// (P(W <= w) = k int phi(z) (Phi(z + w) - Phi(z))^(k-1) dz, and k int phi(z) A^(k-1) dz = 1).
// The bracket is written A^(k-1) (1 - (1 - C/A)^(k-1)) = A^(k-1) (-expm1((k-1) log1p(-C/A))):
// a positive integrand without cancellation, so the tail keeps its relative accuracy where
// 1 - P(W <= w) would lose it (scipy's studentized_range.sf is 0 at q = 30, k = 3, nu = 27;
// the value is 5.3e-21). phi(z) A^(k-1) is exp(-z^2/2 + (k-1) log A) with -z^2/2 a
// double-double; log A is log1p(-Phi(z)) for z < 0.
//
// The integral over z runs over [-w - 12, 9]: below, the integrand is at most k (k-1) phi(z)
// (A^m - (A - C)^m <= m C), a part below Phi(-w - 12) = e^-(w+12)^2/2 against the tail's
// e^-w^2/4; above, at most k (k-1) phi(z) Phi(-(z+w)), below Phi(-9) Phi(-9 - w). First panels
// are cut around the peak at z = -w/2 (the pair at z and z + w) and at 0.
#pragma once

#include "special/dd.hpp"
#include "special/ops/distr.hpp"
#include "special/ops/erf.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/engine.hpp"
#include "stats/mvcdf.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sd::htest::srange {

namespace mv = stats::mvcdf;
namespace dd = special::dd;
using dd::DD;

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kSqrt2 = 1.4142135623730951;

// Phi(x) of the nodes x[0, n) (n <= 21)
inline void ndtr(const double* x, double* o, int n) { special::Ndtr::chunk(x, o, n); }

// phi(x) / Phi(-x), the upper tail's hazard, without underflow
inline double hazard(double x) { return mv::kSqrt2OverPi / special::Erfcx::f(x * mv::kSqrtHalf); }

// P(W > w): the range of k standard normals beyond w >= 0 (k >= 2).
inline double range_sf(double k, double w) {
    if (w != w) return kNaN;
    if (!(w > 0.0)) return 1.0;
    if (std::isinf(w)) return 0.0;
    const double m = k - 1.0;
    auto f = [&](const double* z, double* o, int n) {
        double a1[21], a2[21], a3[21], A[21], C[21], P[21];
        for (int i = 0; i < n; ++i) a1[i] = -z[i], a2[i] = -(z[i] + w), a3[i] = z[i];
        ndtr(a1, A, n);
        ndtr(a2, C, n);
        ndtr(a3, P, n);
        for (int i = 0; i < n; ++i) {
            if (!(A[i] > 0.0) || !(C[i] > 0.0)) {
                o[i] = 0.0;
                continue;
            }
            const double L = z[i] < 0.0 ? std::log1p(-P[i]) : std::log(A[i]);
            const double r = std::min(1.0, C[i] / A[i]);
            const double D = -std::expm1(m * std::log1p(-r));
            const DD e = dd::two_prod(z[i], z[i]) * -0.5 + (m * L - dd::kLogSqrt2Pi.hi) - dd::kLogSqrt2Pi.lo;
            o[i] = k * special::distr_detail::exp_dd(e) * D;
        }
    };
    const double lo = -w - 12.0, hi = 9.0, c = -0.5 * w;
    double cuts[8];
    int nc = 0;
    for (double t : {-w - 6.0, -w, c - 3.0, c, c + 3.0, 0.0, 3.0})
        if (t > lo && t < hi) cuts[nc++] = t;
    std::sort(cuts, cuts + nc);
    nc = static_cast<int>(std::unique(cuts, cuts + nc) - cuts);
    return std::min(1.0, mv::integrate_at(f, lo, hi, cuts, nc, 0.0));
}

// P(Q > q) for k groups and nu degrees of freedom (nu = inf: the range itself).
inline double sf(double q, double k, double nu) {
    if (q != q || k != k || nu != nu || !(k >= 2.0) || !(nu > 0.0)) return kNaN;
    if (!(q > 0.0)) return 1.0;
    if (std::isinf(q)) return 0.0;
    if (std::isinf(nu)) return range_sf(k, q);
    const double hs[1] = {0.5 * q};
    // |P(W > R q) - 1| = P(W <= R q) <= P(|Z1 - Z2| <= R q) <= R q / sqrt(pi) = R (q sqrt 2) / sqrt(2 pi)
    return mv::t_mixture(nu, 1.0, q * kSqrt2, hs, 1, [&](const mv::MixNode& nd) {
        mv::MixTerm t;
        const double w = nd.times(q);
        t.f = range_sf(k, w);
        // d log g / dv = (w / 2) g'(w) / g(w), about -(w / 2) times the hazard of the pair's
        // difference N(0, 2) (exact for k = 2, the tail's rate for any k): only the first-order
        // move of the node (~1e-16) uses it
        t.dlog = -0.5 * w * hazard(w * mv::kSqrtHalf) * mv::kSqrtHalf;
        return t;
    });
}

// The root q of a decreasing tail: log tail(q) = log a, from a bracket [lo, hi] with tail(lo) >=
// a >= tail(hi) (Illinois' regula falsi on the logarithm, bisection when a step leaves the
// bracket), to 2^-50 of q.
template <class F>
double solve_tail(F&& tail, double a, double lo, double hi) {
    if (!(lo <= hi)) return kNaN;
    const double la = std::log(a);
    auto g = [&](double x) {
        const double t = tail(x);
        return t > 0.0 ? std::log(t) - la : -kInf;
    };
    double glo = g(lo), ghi = g(hi);
    if (glo != glo || ghi != ghi) return kNaN;
    if (glo <= 0.0) return lo;
    if (ghi >= 0.0) return hi;
    int side = 0;
    for (int it = 0; it < 200; ++it) {
        if (hi - lo <= 0x1p-50 * std::fabs(hi)) break;
        double x = std::isfinite(ghi) ? hi - ghi * (hi - lo) / (ghi - glo) : 0.5 * (lo + hi);
        if (!(x > lo && x < hi)) x = 0.5 * (lo + hi);
        const double gx = g(x);
        if (gx != gx) return kNaN;
        if (gx == 0.0) return x;
        if (gx > 0.0) {
            lo = x, glo = gx;
            if (side == 1) ghi *= 0.5;
            side = 1;
        } else {
            hi = x, ghi = gx;
            if (side == -1) glo *= 0.5;
            side = -1;
        }
    }
    return 0.5 * (lo + hi);
}

// The Student t's upper a-quantile
inline double t_isf(double a, double nu) { return stats::method_of<stats::StudentT, stats::method::Iccdf>(a, nu); }

// The q with P(Q > q) = a (0 < a < 1): between the quantile of one pair's |T| sqrt 2 (P(Q > q)
// is at least that pair's tail) and Bonferroni's over the k (k - 1) / 2 pairs.
inline double isf(double a, double k, double nu) {
    if (a != a || k != k || nu != nu || !(k >= 2.0) || !(nu > 0.0) || !(a > 0.0 && a < 1.0)) return kNaN;
    const double pairs = 0.5 * k * (k - 1.0);
    double lo = kSqrt2 * t_isf(0.5 * a, nu), hi = kSqrt2 * t_isf(0.5 * a / pairs, nu);
    if (!(lo < hi)) return lo;  // k = 2: Q = sqrt 2 |T|
    return solve_tail([&](double q) { return sf(q, k, nu); }, a, lo, hi);
}

}  // namespace sd::htest::srange
