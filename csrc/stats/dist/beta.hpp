// The beta distribution on [0, 1] with shapes a, b, and the incomplete-beta
// pieces the t and F distributions share.
//
// The density is the incomplete beta's prefix x^a y^b / B(a, b) (special/prefix.hpp:
// its exponent in double-double, no over- or underflow) divided by x y; the cdfs
// are sd.special's I_x(a, b) and 1 - I_x(a, b) (ibeta, TOMS 708). Below the
// normal range the log of the small tail is the log of the prefix (from its
// exponent) plus the log of the series or continued fraction that multiplies it
// (BPSER, BFRAC), so logcdf keeps going where the tail itself underflows.
#pragma once

#include "special/ops/gamma.hpp"
#include "special/ops/incbeta.hpp"
#include "special/prefix.hpp"
#include "stats/chunk.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/two_point.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace beta_detail {

using num::DD;
namespace dd = num::dd;

// log of the prefix x^a y^b / B(a, b), x + y = 1 (double-doubles).
inline DD log_prefix(double a, double b, DD x, DD y) {
    thread_local num::Memo memo;
    const DD f = memo.get(a, b, [&] { return DD{std::log(special::beta_factor(a, b)), 0.0}; });
    return -special::beta_exponent(a, b, x, y) + f;
}

// log I_x(a, b) for a deep lower tail (x well below the mean): the log prefix plus
// log(I / prefix), the factor by BFRAC when a, b > 1 (TOMS 708's choice away from the
// mean) and by BPSER otherwise, both given a unit prefix; BGRAT (in logs) for a > 15,
// b <= 1 and x within 1e-3 of 1, where BPSER's series in x needs ~40 / (1 - x) terms and
// stops at its cap (the negative binomial's ccdf at k = 3.4e7, p = 1.2e-4 was 7e-10 off).
// Above that BPSER stays: measured on the t's deep tails (y ~ 2e-3), it is 2e-16 to
// BGRAT's 4e-16.
inline double log_lower_deep(double a, double b, DD x, DD y) {
    if (b <= 1.0 && a > 15.0 && y.hi < 1e-3) return special::ibeta_detail::log_bgrat(a, b, x, y);
    // BPSER's series in x stops at its cap for x next to 1; with b below the measured shapes
    // (1e-2) a deep tail there was 9e-4 off (a = 1, b = 1e-300, x = 1 - 1e-10): NaN
    if (b < 1e-3 && x.hi > 0.9) return kNaN;
    const special::Scaled unit = special::scaled(1.0);
    double ratio = (a > 1.0 && b > 1.0) ? special::ibeta_detail::bfrac(a, b, x, y, unit)
                                        : special::ibeta_detail::bpser(a, b, x, y, unit);
    // BFRAC's fraction gives NaN for shapes far beyond the guaranteed range (a = 1e281); the
    // power series does not, and for a small x it is a few terms
    if (!(ratio > 0.0 && std::isfinite(ratio)) && x.hi < 0.25) ratio = special::ibeta_detail::bpser(a, b, x, y, unit);
    if (!(ratio > 0.0 && std::isfinite(ratio))) return kNaN;
    const DD l = log_prefix(a, b, x, y);
    return l.hi + (l.lo + std::log(ratio));
}

// Below this a tail is "deep": its log comes from the pieces, not from its value.
inline constexpr double kDeep = 0x1p-960;

// log I_x(a, b) (upper: log(1 - I_x(a, b))).
inline double log_tail(double a, double b, DD x, DD y, bool upper) {
    const special::ibeta_detail::IB r = special::ibeta_detail::ibeta(a, b, x, y);
    const bool lower_small = r.i <= r.ic;
    const double small = lower_small ? r.i : r.ic;
    if (upper != lower_small) {  // the small side is asked for
        if (small >= kDeep) return std::log(small);
        return lower_small ? log_lower_deep(a, b, x, y) : log_lower_deep(b, a, y, x);
    }
    return std::log1p(-small);
}

// log B(a, b) in double-double (log-gammas: the start and the leading term below).
inline DD log_beta(double a, double b) {
    return num::log_gamma(DD{a, 0.0}) + num::log_gamma(DD{b, 0.0}) - num::log_gamma(dd::two_sum(a, b));
}

// log x for the x with log I_x(a, b) = lt (a double-double), deep below the normal
// range. For a small root, the series I = x^a / (a B(a, b)) (1 + c x + O(x^2)),
// c = a (1 - b) / (a + 1): log x = (lt + log a + log B) / a - c x / a, exact once
// c x < 1e-9 (the next term is below 2^-60). Otherwise Newton in log x on log I from
// the pieces, d log I / d log x = x f / I = prefix / (y I). Newton stops at a step of
// 2^-54 (of max(1, |log x|): log x's own spacing is 2^-52 of it, and log I's rounding
// keeps the steps from shrinking much below); not there in 100 steps, the root is taken
// when the last step was at most 2^-48, else NaN (as the gamma's deep inverse, never a value
// that has not converged).
inline DD deep_log_inverse(double a, double b, DD lt) {
    const DD lx0 = (lt + num::log_dd(a) + log_beta(a, b)) / a;
    const double c = a * (1.0 - b) / (a + 1.0);
    const double x0 = std::exp(lx0.hi);
    if (std::fabs(c) * x0 < 1e-9) return lx0 - c * x0 / a;
    double lx = lx0.hi, step = kInf;
    for (int it = 0; it < 100; ++it) {
        const double x = std::exp(lx);
        const DD xd{x, 0.0}, yd = dd::two_sum(1.0, -x);
        const double li = log_lower_deep(a, b, xd, yd);
        const DD lp = log_prefix(a, b, xd, yd);
        const double slope = std::exp(lp.hi - std::log(yd.hi) - li);
        const DD g = DD{li, 0.0} - lt;
        step = (g.hi + g.lo) / slope;
        lx -= step;
        if (std::fabs(step) <= 0x1p-54 * std::max(1.0, std::fabs(lx))) return DD{lx, 0.0};
    }
    if (!(std::fabs(step) <= 0x1p-48 * std::max(1.0, std::fabs(lx)))) return DD{kNaN, 0.0};
    return DD{lx, 0.0};
}

// Both logs of the deep root, log x and log(1 - x) (the t's and the F's quantiles are
// ratios of x and 1 - x). Where the root is near 1 (x > 1/2), 1 - x from log x loses what
// log x has of x's distance from 1: Newton stops at 2^-54 of log x itself (the spacing near
// 1 allows no better), which is 2^-54 / |log x| of 1 - x (t's quantile at df = 1e6 and p ~
// 1e-305 was 1.6e-14 off, x = 1 - 1.4e-3). There the root is refined in log y, y = 1 - x:
// d log I / d log y = -(x f / I)(y / x) = -prefix / (x I), the same stops (2^-54 of
// max(1, |log y|); not there in 100 steps: the last step at most 2^-48, else NaN).
struct LogPair {
    DD lx, ly;
};
inline LogPair deep_log_inverse_both(double a, double b, DD lt) {
    const DD lx = deep_log_inverse(a, b, lt);
    if (!(lx.hi == lx.hi)) return {lx, lx};
    if (lx.hi < -num::kLn2.hi) {  // x < 1/2: 1 - x is well defined from log x
        const double x = std::exp(lx.hi);
        return {lx, x < 0x1p-60 ? DD{-x, 0.0} : num::log_dd(-std::expm1(lx.hi))};
    }
    double ly = std::log(-std::expm1(lx.hi)), step = kInf;
    for (int it = 0; it < 100; ++it) {
        const double y = std::exp(ly);
        const DD yd{y, 0.0}, xd = dd::two_sum(1.0, -y);
        const double li = log_lower_deep(a, b, xd, yd);
        const DD lp = log_prefix(a, b, xd, yd);
        const double slope = -std::exp(lp.hi - std::log(xd.hi) - li);
        const DD g = DD{li, 0.0} - lt;
        step = (g.hi + g.lo) / slope;
        ly -= step;
        if (std::fabs(step) <= 0x1p-54 * std::max(1.0, std::fabs(ly))) {
            step = 0.0;
            break;
        }
    }
    if (!(std::fabs(step) <= 0x1p-48 * std::max(1.0, std::fabs(ly)))) return {DD{kNaN, 0.0}, DD{kNaN, 0.0}};
    return {dd::log(dd::two_sum(1.0, -std::exp(ly))), DD{ly, 0.0}};
}

// Below this a probability goes to the deep inverse (from its log in double-double).
inline constexpr double kDeepP = 0x1p-1000;

// x with I_x(a, b) = p (p <= 1/2 the small side, lt = log p): sd.special's betaincinv
// while p and the root are normal numbers, else the deep inverse. Returns log x too
// when it came from the deep side (lx.hi = NaN otherwise).
struct Root {
    double x;
    DD lx;
};
inline Root lower_root(double a, double b, double p, DD lt) {
    if (p >= kDeepP) {
        const double est = (lt.hi + std::log(a) + special::Betaln::f(a, b)) / a;  // log of the leading-term root
        if (est > -40.0) return {special::Betaincinv::f(a, b, p), DD{kNaN, 0.0}};
    }
    const DD lx = deep_log_inverse(a, b, lt);
    return {num::exp_dd(lx), lx};
}

// ---- pieces of the densities' chunk kernels (beta, t, F) ----------------------------------

// log B(a, b) as a double-double for the kernels' constants (gamma_detail::log_gamma_dd;
// log Gamma(z) = log Gamma(1 + z) - log z keeps a tiny z out of the shift's product). NaN
// past gamma_detail::kPlainMax, where the log-gammas' cancellation is no longer free.
inline DD log_beta_dd(double a, double b) {
    if (!(a > 0.0 && b > 0.0 && a <= gamma_detail::kPlainMax && b <= gamma_detail::kPlainMax)) return {kNaN, 0.0};
    const DD one{1.0, 0.0};
    const DD c = dd::two_sum(a, b);
    return gamma_detail::log_gamma_dd(one + a) - num::log_dd(a) +
           (gamma_detail::log_gamma_dd(one + b) - num::log_dd(b)) - (gamma_detail::log_gamma_dd(c + 1.0) - dd::log(c));
}

// ---- the inverse for small shapes ----------------------------------------------------------------
//
// sd.special's betaincinv runs Halley's method on the double-double ibeta with its
// derivative: four or five evaluations of 250 ns (1.2 us at a = 2.5, b = 0.7). For shapes
// at most 6 the same iteration is run here on betainc and betaincc themselves (Boost's
// there, as the cdfs: Beta::tail), from the same start (AS 109) and in the same variable
// (the smaller of x and 1 - x, against the smaller of p and 1 - p):
//   - x F'(x) = x^a (1 - x)^(b - 1) / B(a, b) is formed in double from libm's logs: it
//     only steers the steps (an error in it changes where a step lands, to second order,
//     not the fixed point);
//   - a Halley step is cubic, so once a step is below 1e-6 of x what it leaves is of the
//     order of (a / x)^2 step^3 <= 1e-16 x for these shapes, and the iteration stops
//     without evaluating again (as special's t_quantile): two or three evaluations.
// The result's error is the function's (2.7e-15 measured, as ibeta's own) times
// F / (x F'), the inverse's condition, like the double-double way's. Anything that does
// not settle in 8 steps, or leaves the bracket, returns false: the caller's general way.
struct XY {
    double x, y;  // the root and 1 - root, the smaller of the two not rounded from the other
};
inline bool fast_root(double a, double b, double p, bool upper, XY& out) {
    // the smaller of p, 1 - p against its own function (special's ibeta_inverse_dispatch)
    const double t = p <= 0.5 ? p : 1.0 - p;
    const bool up = p <= 0.5 ? upper : !upper;
    const double P = up ? b : a, Q = up ? a : b;  // I_u(P, Q) = t
    const special::ibeta_detail::UV g = special::ibeta_detail::as109(P, Q, t);
    const bool in_u = g.u <= 0.5;
    const double sa = in_u ? P : Q, sb = in_u ? Q : P;  // the iteration variable's shapes
    const bool inc = in_u;                              // F = I_w(sa, sb), or 1 - I_w(sa, sb) in v = 1 - u
    double w = in_u ? g.u : g.v;
    if (!(w > 0x1p-100 && w <= 0.5)) return false;
    thread_local dens::Const cst;
    const double lbeta = cst.get(sa, sb, [&] { return DD{special::Betaln::f(sa, sb), 0.0}; }).hi;
    double lo = 0.0, hi = 0.75;
    for (int it = 0; it < 8; ++it) {
        const double F = inc ? special::Betainc::f(sa, sb, w) : special::Betaincc::f(sa, sb, w);
        if (!(F >= kDeep && F < 1.0)) return false;
        const double f = F - t;
        if (f == 0.0) break;
        if ((f > 0.0) == inc) hi = w;
        else lo = w;
        const double xfp = std::exp(sa * std::log(w) + (sb - 1.0) * std::log1p(-w) - lbeta);  // w dI/dw
        if (!(xfp > 0.0) || !std::isfinite(xfp)) return false;
        double wn;
        const bool near = F <= 2.0 * t && F >= 0.5 * t;
        if (near) {
            const double u = w * (f / xfp) * (inc ? 1.0 : -1.0);
            const double h = (sa - 1.0) / w - (sb - 1.0) / (1.0 - w);  // F'' / F'
            const double den = 1.0 - 0.5 * u * h;
            wn = w - (den > 0.5 && den < 2.0 ? u / den : u);
        } else {  // far: Newton on log F (in log w for the increasing one), as special::halley
            const double gl = std::log(F / t);
            wn = inc ? w * std::exp(-gl * (F / xfp)) : w + w * (gl * (F / xfp));
        }
        if (!(wn > lo && wn < hi)) return false;
        const bool done = near && std::fabs(wn - w) <= 1e-6 * wn;
        w = wn;
        if (done) break;
        if (it == 7) return false;
    }
    if (!(w > 0x1p-100 && w <= 0.75)) return false;
    // w is x when (in u and lower) or (in v and upper), as ibeta_inverse
    out = (in_u != up) ? XY{w, 1.0 - w} : XY{1.0 - w, w};
    return true;
}
// The shapes it is used for (below 0.05 the density's ends are too steep for the start).
inline bool fast_shapes(double a, double b) { return a <= 6.0 && b <= 6.0 && a >= 0.05 && b >= 0.05; }

}  // namespace beta_detail

struct Beta {
    static constexpr char name[] = "beta";
    static constexpr std::array<const char*, 2> params{"a", "b"};

    static bool valid(double a, double b) { return a > 0.0 && std::isfinite(a) && b > 0.0 && std::isfinite(b); }
    // A shape below 2^-1021: the shape's limit at 0 (user decision 2026-11-19, T50), the point
    // mass at 0 (a), at 1 (b), or both atoms with P(1) = a / (a + b), every method
    // (stats/dist/two_point.hpp).
    static bool at_limit(double a, double b) {
        return valid(a, b) && (a < two_point::kMinShape || b < two_point::kMinShape);
    }
    template <class M>
    static double limit_in(double x, double a, double b) {
        return two_point::in<M>(x, two_point::of_shapes(a, b, 0.0, 1.0, false));
    }
    template <class M>
    static double limit_par(double a, double b) {
        return two_point::par<M>(two_point::of_shapes(a, b, 0.0, 1.0, false));
    }

    // log density at 0 < x < 1: log prefix - log x - log(1 - x).
    static num::DD log_density(double x, double a, double b) {
        const num::DD xd{x, 0.0}, yd = num::dd::two_sum(1.0, -x);
        return beta_detail::log_prefix(a, b, xd, yd) - num::log_dd(x) - num::dd::log(yd);
    }
    // The density at an end: x = 0 is (a < 1: inf, a = 1: b, a > 1: 0), x = 1 mirrored.
    static double end_density(double s, double other) { return s < 1.0 ? kInf : (s == 1.0 ? other : 0.0); }
    // The double-double definitions (the kernel's fallback: the tails, huge shapes, the ends).
    static double pdf_dd(double x, double a, double b) {
        if (x < 0.0 || x > 1.0) return 0.0;
        if (x == 0.0) return end_density(a, b);
        if (x == 1.0) return end_density(b, a);
        return num::exp_dd(log_density(x, a, b));
    }
    static double logpdf_dd(double x, double a, double b) {
        if (x < 0.0 || x > 1.0) return -kInf;
        if (x == 0.0) return std::log(end_density(a, b));
        if (x == 1.0) return std::log(end_density(b, a));
        const num::DD l = log_density(x, a, b);
        return l.hi + l.lo;
    }
    // The chunk kernel of pdf and logpdf (dens::kernel, stats/dist/gamma.hpp):
    //     l = (a - 1) log x + (b - 1) log(1 - x) - log B(a, b),
    // 1 - x a double-double (two_sum).
    static void density_chunk(const Chunk<2>& c, double* out, bool log) {
        const double *a = c.p[0], *b = c.p[1];
        thread_local dens::Const cst;
        dens::kernel<true>(
            c.n, log, out,
            [&](int i) {
                dens::In in = dens::kBlank;
                in.c = -cst.get(a[i], b[i], [&] { return beta_detail::log_beta_dd(a[i], b[i]); });
                in.k1 = num::dd::two_sum(a[i], -1.0), in.x1 = {c.x[i], 0.0};
                in.k2 = num::dd::two_sum(b[i], -1.0), in.x2 = num::dd::two_sum(1.0, -c.x[i]);
                in.ok = c.x[i] >= 0x1p-1000 && c.x[i] < 1.0 && in.c.hi == in.c.hi;  // false for a NaN
                return in;
            },
            [&](int i) {
                if (!(c.x[i] == c.x[i] && valid(a[i], b[i]))) return kNaN;
                return log ? logpdf_dd(c.x[i], a[i], b[i]) : pdf_dd(c.x[i], a[i], b[i]);
            }
        );
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, false); }
    static void logpdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, true); }
    // The scalar forms: the kernel on one element (every caller gets the same bits).
    static double density_one(double x, double a, double b, bool log) {
        double out;
        density_chunk(Chunk<2>{1, &x, {&a, &b}, {true, true}}, &out, log);
        return out;
    }
    static double pdf(double x, double a, double b) { return density_one(x, a, b, false); }
    static double logpdf(double x, double a, double b) { return density_one(x, a, b, true); }
    // Both shapes at most 6: sd.special's betainc and betaincc, which go to Boost there
    // (special/ops/incbeta.hpp, docs/design/special.md §5.14: measured inside the contract on
    // every reference point, and against Arb on 4e4 points of this region as accurate as the
    // double-double ibeta, 2.7e-15 to its 4.8e-15; bench/stats/opt/b) at half the time or
    // less. x is the input itself, so nothing is rounded on the way in. Larger shapes, and
    // a result below 2^-960, are ibeta's.
    static bool small_shapes(double a, double b) { return a <= 6.0 && b <= 6.0; }
    // Both shapes past 2^100: the distribution's width, 1 / sqrt(a + b), is below 2^-50 and the
    // cdf a step between neighbouring doubles; special's incomplete beta is not measured near
    // there (F(1.7e308, 1.7e308).cdf(0.5) was 1). The cdfs and inverses are NaN.
    static bool degenerate(double a, double b) { return a > 0x1p100 && b > 0x1p100; }
    static double tail(double x, double a, double b, bool upper) {
        if (degenerate(a, b)) return kNaN;
        if (x <= 0.0) return upper ? 1.0 : 0.0;
        if (x >= 1.0) return upper ? 0.0 : 1.0;
        if (small_shapes(a, b)) {
            // (kept from 2^-960 up: Boost's value loses digits just above the subnormals,
            // 1.7e-13 at 1e-308)
            const double v = upper ? special::Betaincc::f(a, b, x) : special::Betainc::f(a, b, x);
            if (v >= beta_detail::kDeep) return v;
        }
        const special::ibeta_detail::IB r = special::ibeta_detail::ibeta(a, b, x);
        return upper ? r.ic : r.i;
    }
    static double cdf(double x, double a, double b) { return tail(x, a, b, false); }
    static double ccdf(double x, double a, double b) { return tail(x, a, b, true); }
    // The log of a tail from the smaller side's value (its log, or log1p of minus it for the
    // other side), as beta_detail::log_tail; for small shapes the smaller side is betainc's
    // or betaincc's: the side the mean suggests first, the other if that one is above 1/2.
    // A deep tail (below beta_detail::kDeep) is beta_detail::log_tail's, from the pieces.
    static double log_tail(double x, double a, double b, bool upper) {
        if (degenerate(a, b)) return kNaN;
        if (x <= 0.0) return upper ? 0.0 : -kInf;
        if (x >= 1.0) return upper ? -kInf : 0.0;
        if (small_shapes(a, b)) {
            bool lower_small = x * (a + b) < a;
            double small = lower_small ? special::Betainc::f(a, b, x) : special::Betaincc::f(a, b, x);
            if (small > 0.5) {
                lower_small = !lower_small;
                small = lower_small ? special::Betainc::f(a, b, x) : special::Betaincc::f(a, b, x);
            }
            if (small >= beta_detail::kDeep) return upper != lower_small ? std::log(small) : std::log1p(-small);
        }
        return beta_detail::log_tail(a, b, num::DD{x, 0.0}, num::dd::two_sum(1.0, -x), upper);
    }
    static double logcdf(double x, double a, double b) { return log_tail(x, a, b, false); }
    static double logccdf(double x, double a, double b) { return log_tail(x, a, b, true); }

    // x with I_x(a, b) = t (upper: 1 - I_x(a, b) = t) for a normal t, lt = log t: sd.special's
    // inverses, which choose the side by the root (x or 1 - x, whichever is small), unless
    // t is tiny or the root is (beta_detail::lower_root: the deep inverse on the log side).
    // For the upper side a tiny t means 1 - x tiny: solved as the lower side of (b, a).
    static double solve(double t, num::DD lt, double a, double b, bool upper) {
        if (degenerate(a, b)) return kNaN;
        const double x = solve_raw(t, lt, a, b, upper);
        return checked_shapes(a, b) ? x : verify(x, lt, a, b, upper);
    }
    // The shapes whose inverses are measured (tests/stats/ref: 1e-2 to 1e4; guaranteed to 1e5).
    // Outside, a root is returned only if the forward log tail reproduces the target
    // (num::accept_root); a root at an end only if the tail at the last double inside is
    // already past the target.
    static bool checked_shapes(double a, double b) { return a >= 1e-3 && b >= 1e-3 && a <= 1e5 && b <= 1e5; }
    static double verify(double x, num::DD lt, double a, double b, bool upper) {
        if (!(x >= 0.0 && x <= 1.0)) return kNaN;
        if (x == 0.0 || x == 1.0) {
            const bool low = x == 0.0;
            const double lf = log_tail(low ? 0x1p-1074 : 1.0 - 0x1p-53, a, b, upper);
            const double slack = 1e-9 * std::max(1.0, std::fabs(lt.hi));
            // the lower tail grows with x: a root below the first double needs F there >= target
            const bool ok = (low != upper) ? lf >= lt.hi - slack : lf <= lt.hi + slack;
            return ok ? x : kNaN;
        }
        const num::DD ld = log_density(x, a, b);
        return num::accept_root(x, log_tail(x, a, b, upper), ld.hi + std::log(x), lt.hi);
    }
    static double solve_raw(double t, num::DD lt, double a, double b, bool upper) {
        if (beta_detail::fast_shapes(a, b) && t >= 0x1p-200) {  // (beta_detail::fast_root)
            beta_detail::XY r;
            if (beta_detail::fast_root(a, b, t, upper, r)) return r.x;
        }
        if (!upper) return beta_detail::lower_root(a, b, t, lt).x;
        if (t >= beta_detail::kDeepP) {
            const double est = (lt.hi + std::log(b) + special::Betaln::f(a, b)) / b;  // log of a tiny 1 - x
            if (est > -40.0) return special::Betainccinv::f(a, b, t);
        }
        return 1.0 - beta_detail::lower_root(b, a, t, lt).x;
    }
    static double icdf(double p, double a, double b) {
        if (!(p > 0.0 && p < 1.0)) return special::Betaincinv::f(a, b, p);
        return solve(p, num::log_dd(p), a, b, false);
    }
    static double iccdf(double p, double a, double b) {
        if (!(p > 0.0 && p < 1.0)) return special::Betainccinv::f(a, b, p);
        return solve(p, num::log_dd(p), a, b, true);
    }
    // From log p: e^lp (lp exact) for the side asked, or, above log 1/2, 1 - e^lp for the
    // other side (the probability there is the small one).
    static double from_log(double lp, double a, double b, bool upper) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return upper ? 0.0 : 1.0;
        if (lp == -kInf) return upper ? 1.0 : 0.0;
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return solve(q, num::log_dd(q), a, b, !upper);
        }
        return solve(std::exp(lp), num::DD{lp, 0.0}, a, b, upper);
    }
    static double ilogcdf(double lp, double a, double b) { return from_log(lp, a, b, false); }
    static double ilogccdf(double lp, double a, double b) { return from_log(lp, a, b, true); }

    static double mean(double a, double b) {
        const double c = a + b;
        if (std::isinf(c)) return a >= b ? 1.0 / (1.0 + b / a) : (a / b) / (1.0 + a / b);
        return a / c;
    }
    static double variance(double a, double b) {
        const double c = a + b;
        if (std::isinf(c)) return 0.0;  // below 1 / c
        return (a / c) * (b / c) / (c + 1.0);
    }
    // sqrt of it by factors (the variance itself may be subnormal where its root is not).
    static double standard_deviation(double a, double b) {
        const double c = a + b;
        if (std::isinf(c)) return 0.0;
        const double v = (a / c) * (b / c) / (c + 1.0);
        if (v >= 0x1p-900) return std::sqrt(v);
        return std::sqrt(a) * std::sqrt(b) / c / std::sqrt(c + 1.0);  // (a / c may round in the subnormals)
    }
    static double skewness(double a, double b) {
        const double c = a + b, ab = a * b;
        if (std::isinf(c)) return 0.0;  // 2 (b - a) / (c sqrt(a b / c)) is below 2 / sqrt(c)
        if (ab >= 0x1p-900 && ab <= 0x1p900) return 2.0 * (b - a) * std::sqrt(c + 1.0) / ((c + 2.0) * std::sqrt(ab));
        return 2.0 * ((b - a) / std::sqrt(a)) / std::sqrt(b) *
               (std::sqrt(c + 1.0) / (c + 2.0));  // a b leaves the range
    }
    // 6 ((a - b)^2 (c + 1) - a b (c + 2)) / (a b (c + 2) (c + 3)): the numerator crosses 0,
    // so it is formed in double-double.
    static double kurtosis(double a, double b) {
        const double p = a * b;
        if (!(p >= 0x1p-900 && p <= 0x1p900 && a + b <= 0x1p900)) {
            // a b or the sum leaves the range: 6 ((a/b + b/a - 2) (c + 1) / (c + 2) - 1) / (c + 3)
            // in double (the numerator's crossing of 0 is not near such shapes)
            const double cs = a + b;
            if (std::isinf(cs)) return 0.0;  // O(1 / c)
            const double d =
                cs + 3.0;  // (a/b + b/a - 2) / d term by term: the sum may overflow where the quotient does not
            // (the quotient first where it is a double: a / d rounds to nothing for a subnormal
            // a, and the kurtosis of Beta(5e-324, 5e-324) was -4 for -2)
            auto over = [d](double x, double y) {
                const double q = x / y;
                return std::isfinite(q) && q != 0.0 ? q / d : (x / d) / y;
            };
            const double r = over(a, b) + over(b, a) - 2.0 / d;
            return 6.0 * (r * ((cs + 1.0) / (cs + 2.0)) - 1.0 / d);
        }
        const num::DD c = num::dd::two_sum(a, b);
        const num::DD d = num::dd::two_sum(a, -b);
        const num::DD ab = num::dd::two_prod(a, b);
        const num::DD n = d * d * (c + 1.0) - ab * (c + 2.0);
        const num::DD r = n * 6.0 / (ab * (c + 2.0) * (c + 3.0));
        return r.hi + r.lo;
    }
    // The single largest point (T21): (a - 1)/(a + b - 2) inside; an end where only that
    // end is largest; NaN for two ends (a, b < 1) or the uniform a = b = 1.
    static double mode(double a, double b) {
        if (a > 1.0 && b > 1.0) return (a - 1.0) / (a + b - 2.0);
        if (a <= 1.0 && b <= 1.0) {
            if (a == 1.0 && b < 1.0) return 1.0;
            if (b == 1.0 && a < 1.0) return 0.0;
            return kNaN;
        }
        return a <= 1.0 ? 0.0 : 1.0;
    }
    // log B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (c - 2) psi(c)
    // = G(a) + G(b) + Gc(c) + log(a)/2 + log(b)/2 - 3 log(c)/2 + log(2 pi)/2 (numeric.hpp).
    static double entropy(double a, double b) {
        if (std::isinf(1.0 / a) || std::isinf(1.0 / b)) return -kInf;  // psi(a) ~ -1/a beyond the range
        const num::DD c = num::dd::two_sum(a, b);
        const num::DD logs = (num::log_dd(a) + num::log_dd(b) - num::dd::log(c) * 3.0) * 0.5 + num::kLogSqrt2Pi;
        const double g = num::entropy_g(a, special::Digamma::f) + num::entropy_g(b, special::Digamma::f) +
                         num::entropy_gc(c, special::Digamma::f);
        const num::DD h = logs + g;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return 1.0; }
};

}  // namespace sd::stats
