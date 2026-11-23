// The studentized range distribution of tukey_hsd (milestone H8, docs/design/htest.md §3.9;
// H14, H107: inside htest, not a public sd.stats distribution), and the root finder that turns
// an upper tail into a critical value (tukey_hsd's and dunnett's intervals).
//
// Q = W / R for the range W of k independent standard normals and R = sqrt(S / nu), S ~ chi^2_nu
// independent of them. Its upper tail is the scale mixture
//     P(Q > q) = E[P(W > R q)],
// taken in the chi variable v = log R^2 with stats/mvcdf.hpp t_mixture's integrand (lattice_mixture
// below: the trapezoid rule, halved to agreement; t_mixture's adaptive Gauss-Kronrod rule for
// nu < 2), and the range's own upper tail is one integral over the smallest of the k values, z:
//     P(W > w) = k int phi(z) [A^(k-1) - (A - C)^(k-1)] dz,   A = Phi(-z), C = Phi(-(z + w))
// (P(W <= w) = k int phi(z) (Phi(z + w) - Phi(z))^(k-1) dz, and k int phi(z) A^(k-1) dz = 1).
// The bracket is written A^(k-1) (1 - (1 - C/A)^(k-1)) = A^(k-1) (-expm1((k-1) log1p(-C/A))):
// a positive integrand without cancellation, so the tail keeps its relative accuracy where
// 1 - P(W <= w) would lose it (scipy's studentized_range.sf is 0 at q = 30, k = 3, nu = 27;
// the value is 5.3e-21). phi(z) A^(k-1) is exp(-z^2/2 + (k-1) log A) with -z^2/2 a
// double-double; log A is log1p(-Phi(z)) for z < 0.
//
// The integral over z runs inside [-w - 12, 9]: below, the integrand is at most k (k-1) phi(z)
// (A^m - (A - C)^m <= m C), a part below Phi(-w - 12) = e^-(w+12)^2/2 against the tail's
// e^-w^2/4; above, at most k (k-1) phi(z) Phi(-(z+w)), below Phi(-9) Phi(-9 - w). It is taken by
// the trapezoid rule on a lattice walked out from the peak at z = -w/2 (lattice_sum): both
// integrands are analytic with Gaussian tails, where that rule converges exponentially, ~70 nodes
// against ~370 for adaptive Gauss-Kronrod panels to the same accuracy (measured against Arb at
// w = 2 to 31: within 2e-16, where the panels were within 1.5e-15).
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

// Phi(x) of the nodes x[0, n) (n <= 1024, the chunk)
inline void ndtr(const double* x, double* o, int n) { special::Ndtr::chunk(x, o, n); }

// phi(x) / Phi(-x), the upper tail's hazard, without underflow
inline double hazard(double x) { return mv::kSqrt2OverPi / special::Erfcx::f(x * mv::kSqrtHalf); }

// The trapezoid rule on the lattice z0 + j h, walked out from z0 in blocks until the terms are
// negligible: term(z, o, n) fills the integrand at n <= kBlock nodes. The integrands here are
// analytic, positive and unimodal with Gaussian tails, where the rule's error is exponentially
// small in 1 / h (Trefethen and Weideman, "The exponentially convergent trapezoidal rule", SIAM
// Review 56, 2014): far fewer nodes than adaptive Gauss-Kronrod panels to the same accuracy. A side
// stops at the window's end [lo, hi], or at a block whose last term is below 2^-64 of the largest
// so far and below its first (past the peak: what is left falls off at least as fast), and not
// inside [zl, zr] (where the caller knows the peaks lie). z0 and h are multiples of 2^-8, so every
// node is exact.
inline constexpr int kBlock = 16;
inline double lattice_step(double h) { return std::max(0x1p-8, std::ldexp(std::nearbyint(std::ldexp(h, 8)), -8)); }
template <class T>
double lattice_sum(T&& term, double z0, double h, double lo, double hi, double zl, double zr) {
    z0 = std::ldexp(std::nearbyint(std::ldexp(z0, 8)), -8);
    h = lattice_step(h);
    DD sum{0.0, 0.0};
    double mx = 0.0;
    for (int side = 0; side < 2; ++side) {
        for (int b = 0;; ++b) {
            double z[kBlock], o[kBlock];
            int n = 0;
            for (int i = 0; i < kBlock; ++i) {
                const double j = side ? -static_cast<double>(b * kBlock + i + 1) : static_cast<double>(b * kBlock + i);
                const double t = z0 + j * h;
                if (t < lo || t > hi) break;
                z[n++] = t;
            }
            if (n == 0) break;
            term(z, o, n);
            for (int i = 0; i < n; ++i) {
                sum = sum + o[i];
                mx = std::max(mx, o[i]);
            }
            if (n < kBlock) break;
            const bool outside = side ? z[n - 1] < zl : z[n - 1] > zr;
            if (outside && o[n - 1] <= 0x1p-64 * mx && o[n - 1] <= o[0]) break;
        }
    }
    return sum.hi * h;
}

// P(W > w): the range of k standard normals beyond w >= 0 (k >= 2), by the lattice rule from the
// peak at z = -w/2 (the pair at z and z + w). The step: the integrand's narrowest feature is
// A^(k-1)'s fall, sharper as k grows; 0.4 / log(k)^0.75 keeps the rule's error below the
// integrand's own rounding (~1e-15 relative, measured against the adaptive Gauss-Kronrod rule
// over w in [1e-3, 60] for k = 2 to 1000, with at least a factor 1.2 in h to spare).
inline double range_step(double k) { return std::min(0.4, 0.4 / std::pow(std::log(k), 0.75)); }

inline double range_sf(double k, double w) {
    if (w != w) return kNaN;
    if (!(w > 0.0)) return 1.0;
    if (std::isinf(w)) return 0.0;
    const double m = k - 1.0;
    auto f = [&](const double* z, double* o, int n) {
        double a1[kBlock], a2[kBlock], P[kBlock], C[kBlock];
        for (int i = 0; i < n; ++i) a1[i] = -std::fabs(z[i]), a2[i] = -(z[i] + w);
        ndtr(a1, P, n);  // Phi(-|z|): A = Phi(-z) for z >= 0, Phi(z) = 1 - A below
        ndtr(a2, C, n);
        for (int i = 0; i < n; ++i) {
            const double A = z[i] < 0.0 ? 1.0 - P[i] : P[i];
            if (!(A > 0.0) || !(C[i] > 0.0)) {
                o[i] = 0.0;
                continue;
            }
            const double L = z[i] < 0.0 ? std::log1p(-P[i]) : std::log(P[i]);
            const double r = std::min(1.0, C[i] / A);
            const double D = -std::expm1(m * std::log1p(-r));
            const DD e = dd::two_prod(z[i], z[i]) * -0.5 + (m * L - dd::kLogSqrt2Pi.hi) - dd::kLogSqrt2Pi.lo;
            o[i] = k * special::distr_detail::exp_dd(e) * D;
        }
    };
    return std::min(1.0, lattice_sum(f, -0.5 * w, range_step(k), -w - 12.0, 9.0, -0.5 * w, -0.5 * w));
}

// E[g(R)] over R = sqrt(chi^2_nu / nu) by the trapezoid rule in v = log R^2, the integrand of
// stats/mvcdf.hpp's t_mixture (the weight exp(c(a) - a (x - 1 - log x)), x = e^v, a = nu / 2, each
// node at the rounded x moved back to v by the integrand's logarithmic derivative), on a lattice
// halved until two levels agree: the rule's error falls as e^(-const / h), so a level within 2^-40
// of the coarser one is within ~2^-80 itself. The integrand is analytic in v and unimodal: the
// weight peaks at v = 0 with width 1 / sqrt(a), and g, where it falls like exp(-beta R^2 q^2),
// moves the peak to v = -log1p(beta q^2 / a), the same width. The first step 1.5 / sqrt(a) is about
// twice the one that reaches 1e-18 for the weight alone (2 pi / sqrt(82 a)); a g that turns faster
// (many groups) takes more levels. Each level walks out from the peak until a term is below
// 2^-64 of the largest and decreasing, outside [min(peak, 0) - width, width]. Below v_c (R below
// t_mixture's r_c) g is p0 to 1e-17 and is not called. Small a (a < 1, nu < 2: the weight's left
// tail e^(a v) is long for one step) and very large a (above 1e12), or no agreement in six levels:
// t_mixture's adaptive Gauss-Kronrod rule.
template <class G>
double lattice_mixture(double nu, double p0, double habs, double beta_q2, const double* hs, int nh, G&& g) {
    const double a = 0.5 * nu;
    auto adaptive = [&] { return mv::t_mixture(nu, p0, habs, hs, nh, g); };
    if (habs == 0.0) return p0;
    if (!(a >= 1.0 && a <= 1e12) || !(p0 > 0.0) || !(habs < 1e300) || !(beta_q2 < 1e300)) return adaptive();
    const DD c = mv::stirling_c(a);
    const double rc = 1e-17 * p0 / (0.4 * habs);
    if (!(rc >= std::numeric_limits<double>::min())) return adaptive();
    const double vc = 2.0 * std::log(rc);
    const double width = 1.0 / std::sqrt(a);
    const double vpk = -std::log1p(beta_q2 / a);
    const double vl = std::min(vpk, 0.0) - width, vr = width;
    // node value: the weighted integrand at v (an exact lattice point)
    auto term = [&](double v) {
        if (v < mv::kMixFarV) return 0.0;
        mv::MixNode nd;
        const double x = std::exp(v);
        const DD lx = dd::log(x);
        const double dv = (v - lx.hi) - lx.lo;
        nd.x = DD{x, 0.0};
        const DD ew = dd::a_mu(a, nd.x);
        if ((ew - c).hi > 745.2) return 0.0;
        mv::MixTerm t;
        if (v < vc) t.f = p0;
        else t = g(nd);
        const DD e = ew + t.e - c;
        const double w = e.hi > 745.2 || t.f == 0.0 ? 0.0 : std::exp(-e.hi) * (1.0 - e.lo);
        return dv == 0.0 ? w * t.f : w * t.f * (1.0 + (t.dlog - a * (x - 1.0)) * dv);
    };
    // the lattice: v0 + j h, v0 and h multiples of 2^-20 (exact nodes)
    auto grid = [](double t) { return std::ldexp(std::nearbyint(std::ldexp(t, 20)), -20); };
    const double h0 = grid(std::min(1.0, 1.5 * width));
    if (!(h0 > 0.0)) return adaptive();
    const double v0 = grid(std::nearbyint(vpk / h0) * h0);
    DD sum{0.0, 0.0};
    double mx = 0.0, prev = 0.0;
    for (int level = 0; level < 6; ++level) {
        // level 0: v0 + j h0; level L: the odd multiples of h0 / 2^L
        const double step = std::ldexp(h0, -level), stride = level == 0 ? step : 2.0 * step;
        const double first = level == 0 ? v0 : v0 + step;
        for (int side = 0; side < 2; ++side) {
            double last = kInf;
            for (int j = 0; j < 1 << 14; ++j) {
                const double v = side ? first - (j + 1) * stride : first + j * stride;
                const double o = term(v);
                if (o != o) return kNaN;
                sum = sum + o;
                mx = std::max(mx, o);
                const bool outside = side ? v < vl : v > vr;
                if (outside && o <= 0x1p-64 * mx && o <= last) break;
                last = o;
            }
        }
        const double T = sum.hi * step;
        if (level > 0 && std::fabs(T - prev) <= 0x1p-40 * T) return mv::cap1(T);
        prev = T;
    }
    return adaptive();
}

// P(Q > q) for k groups and nu degrees of freedom (nu = inf: the range itself).
inline double sf(double q, double k, double nu) {
    if (q != q || k != k || nu != nu || !(k >= 2.0) || !(nu > 0.0)) return kNaN;
    if (!(q > 0.0)) return 1.0;
    if (std::isinf(q)) return 0.0;
    if (std::isinf(nu)) return range_sf(k, q);
    const double hs[1] = {0.5 * q};
    // |P(W > R q) - 1| = P(W <= R q) <= P(|Z1 - Z2| <= R q) <= R q / sqrt(pi) = R (q sqrt 2) / sqrt(2 pi)
    return lattice_mixture(nu, 1.0, q * kSqrt2, 0.25 * q * q, hs, 1, [&](const mv::MixNode& nd) {
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
