// The gamma distribution with shape a and scale s (T18), and the chi-square
// distribution with df degrees of freedom (= gamma(df/2, scale 2)).
//
// The cdfs are sd.special's regularized incomplete gamma P(a, y), Q(a, y) at
// y = x / s, which is a double-double (stats/numeric.hpp): the rounded y.hi
// goes to the incomplete gamma and y.lo enters to first order through
// y P'(y) = y^a e^-y / Gamma(a) (the condition number of P in y reaches a).
// Below the normal range the log tails come from the unrounded pieces
// (igam_detail::log_tail: the prefix's exponent in double-double times the
// series or continued fraction), as the inverses' deep-tail iteration does.
#pragma once

#include "special/ops/gamma.hpp"
#include "special/ops/incgamma.hpp"
#include "special/prefix.hpp"
#include "stats/chunk.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace sd::stats {

namespace gamma_detail {

using num::DD;
namespace dd = num::dd;

// log of the standard density y^(a-1) e^-y / Gamma(a) at y > 0 finite (a double-double).
// For a >= 13 through the prefix's exponent E = y - a - a log(y/a) (Temme's form,
// special/prefix.hpp): log(a D / y) = log a - log y - E - log(sqrt(2 pi a) Gamma*(a));
// E at y.hi in double-double, y.lo to first order (dE/dy = 1 - a/y).
inline DD log_density_std(double a, DD y) {
    if (a >= 13.0) {
        DD e = special::gamma_exponent(a, y.hi);
        if (y.lo != 0.0) e = e + (y.lo - a * (y.lo / y.hi));  // (a / y.hi alone overflows for a tiny y)
        thread_local num::Memo memo;  // log a - log(sqrt(2 pi a) Gamma*(a))
        const DD c = memo.get(a, 0.0, [&] {
            return num::log_dd(a) - std::log(std::sqrt(2.0 * special::kPi * a) * special::gamma_star(a));
        });
        return c - dd::log(y) - e;
    }
    // (a - 1) log y - y - log Gamma(a); log Gamma(a) = log Gamma(1 + a) - log a (no
    // large terms to round for a tiny a). (a - 1) log y as a log y - log y: a - 1 is not
    // exact below a = 1/2, and log y reaches 745 (39 ULP).
    const DD ly = y.hi >= 0x1p-1000 ? dd::log(y) : num::log_dd(y.hi);
    thread_local num::Memo memo;
    const DD lg = memo.get(a, 0.0, [&] { return num::log_gamma(DD{1.0, 0.0} + a) - num::log_dd(a); });
    return ly * a - ly - y - lg;
}

// P(a, y) (upper: Q) at y = hi + lo > 0 finite.
inline double tail(double a, DD y, bool upper) {
    const special::igam_detail::PQ r = special::igam_detail::igam(a, y.hi);
    const double d = r.xderiv * (y.lo / y.hi);  // y P'(y) dy / y
    return upper ? r.q - d : r.p + d;
}

// log P(a, y) (upper: log Q). The smaller side: log of its value, or below the
// normal range log_tail's unrounded pieces; the larger: log1p(-smaller).
inline double log_tail(double a, DD y, bool upper) {
    const special::igam_detail::PQ r = special::igam_detail::igam(a, y.hi);
    const double d = r.xderiv * (y.lo / y.hi);
    const double p = r.p + d, q = r.q - d;
    const bool lower_small = p <= q;
    const double small = lower_small ? p : q, large = lower_small ? q : p;
    if (upper == !lower_small) {  // the small side is asked for
        if (small >= 0x1p-1000) return std::log(small);
        const auto [l, dl] = special::igam_detail::log_tail(a, y.hi, lower_small, DD{0.0, 0.0});
        return l + dl * y.lo;
    }
    (void)large;
    return std::log1p(-small);
}

// Past this shape the cdfs and their inverses are NaN: the distribution's relative width,
// 1 / sqrt(a), is below 2^-50, so the cdf goes from 0 to 1 within a few doubles, and special's
// incomplete gamma is not measured anywhere near (its values there left [0, 1]).
inline constexpr double kMaxShape = 0x1p100;

// A shape below the normal range has lost bits itself, and so do the incomplete gamma's
// pieces at it (a / tail, sqrt(2 pi a), y / a beyond the doubles): its tails come from the
// reference shape a0 = 2^-600 instead. For a tiny shape
//     Q(a, y) = a E1(y) (1 + O(a (1 + log^2 y))),
// so Q(a, y) = (a / a0) Q(a0, y) to 2^-570 relatively for every y a quotient of doubles
// (|log y| < 1500), and P = 1 - Q is 1 to 2^-1010.
inline constexpr double kMinNormalShape = 0x1p-1022;
inline constexpr double kRefShape = 0x1p-600;
// log(a / a0) for a subnormal a.
inline DD log_shape_ratio(double a) { return num::log_dd(a) + num::kLn2 * 600.0; }
// Q(a, y) = a E1(y) at or above 1600 a (a <= a0) has its root y below 2^-2300: s y is below
// every double for any scale s.
inline constexpr double kLogRootZero = 7.3777589082278725;  // log 1600

// log P(a, y) for y below 2^-960, from log y: the leading term y^a / Gamma(a + 1) (the next
// is a y / (a + 1) of it). NaN where a log y overflows.
inline DD log_lower_tiny(double a, DD ly) { return ly * a - num::lgamma1p(a); }

// ---- Q(a, y) for an integer or half-integer a by its finite sum ---------------------------------
//
// For 2a an integer (every chi-square with integer df) the upper tail is a finite sum of
// positive terms (DLMF 8.4.10, 8.4.14; Boost's finite_gamma_q and finite_half_gamma_q):
//     Q(n, y)       = e^-y sum_{k<n} y^k / k!
//     Q(n + 1/2, y) = erfc(sqrt y) + e^-y / sqrt(pi y) sum_{k=1..n} y^k / ((1/2)(3/2)...(k - 1/2))
// against the continued fraction's ~26 terms and two passes (150 ns at a = 2.5). Here the
// sums are Horner forms in double-double (each ratio y / k from the division's remainder, so
// y = hi + lo enters exactly and the sums are exact to 2^-100), and the rest is counted in
// units of 2^-53:
//     integer: e^-y.hi from libm (2) times the sum, one fma (1), second order (1): 4;
//     half:    z = sqrt(y.hi) and d = (y - z^2) / (2 z) exactly to first order, so
//              erfc(sqrt y) = erfc(z) - (2 / sqrt pi) e^-y d: libm's erfc (2) and one sum (1);
//              the other term is (2 / sqrt pi) e^-y (z + 2 d) s: e^-y (2), the constant (1),
//              three products (3); both positive, and their sum (1): 8 at most.
// Used on the side where Q is the smaller one, y >= a (the median is below the mean), so
// P = 1 - Q is within the same bound plus one rounding; and while e^-y is a normal number
// with room (y <= 690). Elsewhere, and for other a, igam.
inline constexpr double kFiniteMax = 30.0;  // the largest a (60 terms' worth of Horner steps at most)
inline bool finite_upper(double a, DD y, double& q) {
    const double a2 = 2.0 * a;
    if (!(a <= kFiniteMax && y.hi >= a && y.hi <= 690.0) || a2 != std::floor(a2)) return false;
    const int n2 = static_cast<int>(a2);
    const double e = std::exp(-y.hi);
    DD s{1.0, 0.0};
    if (n2 % 2 == 0) {
        // s = 1 + y/1 (1 + y/2 (... (1 + y/(n-1))))
        for (int k = n2 / 2 - 1; k >= 1; --k) s = s * dd::DD(y / static_cast<double>(k)) + 1.0;
        s.lo -= s.hi * y.lo;  // e^-y = e^-y.hi (1 - y.lo)
        q = std::fma(e, s.hi, e * s.lo);
        return true;
    }
    constexpr double kTwoOverSqrtPi = 1.1283791670955126;
    const double z = std::sqrt(y.hi);
    const double d = (std::fma(-z, z, y.hi) + y.lo) / (2.0 * z);  // sqrt(y) - z
    const double head = std::erfc(z) - kTwoOverSqrtPi * e * d;
    const int n = (n2 - 1) / 2;
    if (n == 0) {
        q = head;
        return true;
    }
    // s = 1 + y/(3/2) (1 + y/(5/2) (... (1 + y/(n - 1/2)))); the sum is (y / (1/2)) s
    for (int k = n; k >= 2; --k) s = s * dd::DD(y / (static_cast<double>(k) - 0.5)) + 1.0;
    // (2 / sqrt pi) e^-y sqrt(y) s, sqrt(y) = z + d... and y / sqrt(y) (1/2)^-1 = 2 sqrt(y)
    const double m = z * s.hi + (z * s.lo + d * s.hi - z * s.hi * y.lo);
    q = head + kTwoOverSqrtPi * (e * m);
    return true;
}

// P(a, y) or Q(a, y) for the distributions (the gamma, chi-square and inverse gamma cdfs):
// the finite sum where it applies, else `tail`.
inline double dist_tail(double a, DD y, bool upper) {
    double q;
    if (finite_upper(a, y, q)) return upper ? q : 1.0 - q;
    return tail(a, y, upper);
}
// log of it: Q is the smaller side wherever the finite sum is used.
inline double dist_log_tail(double a, DD y, bool upper) {
    double q;
    if (finite_upper(a, y, q)) return upper ? std::log(q) : std::log1p(-q);
    return log_tail(a, y, upper);
}

// P(a, nu / den) (upper: Q), or its log, for nu, den > 0 finite, where the shape is past
// kMaxShape (NaN) or the quotient y leaves [2^-960, max]: beyond the double range the limits;
// below 2^-960 (where y's double-double has lost bits, or is 0) the leading term from
// log nu - log den. Out of line: the cdfs' hot path is ratio_tail.
inline double ratio_tail(double a, double nu, double den, bool upper, bool log);
[[gnu::noinline, gnu::cold]] inline double ratio_tail_edge(double a, double nu, double den, DD y, bool upper, bool log) {
    if (!(a <= kMaxShape) || y.hi != y.hi) return kNaN;
    if (std::isinf(y.hi)) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
    if (a < kMinNormalShape) {
        // a subnormal shape, from the reference shape: log Q = log Q(a0, y) + log(a / a0);
        // Q = a (Q(a0, y) / a0), one rounding in the subnormals; P = 1 - Q, log P = -Q
        if (upper && log) {
            const DD l = log_shape_ratio(a) + ratio_tail(kRefShape, nu, den, true, true);
            return l.hi + l.lo;
        }
        const double q = a * (0x1p600 * ratio_tail(kRefShape, nu, den, true, false));
        return upper ? q : (log ? -q : 1.0 - q);
    }
    const DD lp = log_lower_tiny(a, num::log_dd(nu) - num::log_dd(den));
    const double l = lp.hi + lp.lo;
    if (!upper) return log ? l : num::exp_dd(lp);
    // 1 - P: from P itself below 1/2 (exp of the double-double: e^l of the rounded l is off
    // by |l| ULP), else from -expm1(l)
    if (l < -num::kLn2.hi) return log ? std::log1p(-num::exp_dd(lp)) : 1.0 - num::exp_dd(lp);
    return log ? std::log(-std::expm1(l)) : -std::expm1(l);
}
// P(a, nu / den) (upper: Q), or its log, for nu, den > 0 finite. A value outside [0, 1] (a
// positive log) is the first-order correction for y.lo failing next to the mean of a huge
// shape: NaN.
inline double ratio_tail(double a, double nu, double den, bool upper, bool log) {
    const DD y = num::ratio(nu, den);
    if (!(a <= kMaxShape && a >= kMinNormalShape && y.hi >= 0x1p-960 && y.hi <= std::numeric_limits<double>::max()))
        [[unlikely]]
        return ratio_tail_edge(a, nu, den, y, upper, log);
    if (log) {
        const double l = dist_log_tail(a, y, upper);
        return l <= 0.0 ? l : kNaN;
    }
    const double v = dist_tail(a, y, upper);
    return v >= 0.0 && v <= 1.0 ? v : kNaN;
}

// s y for the y with log P(a, y) = lt (lower) or log Q(a, y) = lt (upper), lt a
// double-double far below the normal range: Newton on log F from the unrounded pieces
// (igam_detail::log_tail), in log y for P (y tiny) and in y for Q (y large), from the
// tails' leading terms. A small lower root comes from the series
// P = y^a / Gamma(a + 1) (1 - a y / (a + 1) + O(y^2)):
// log y = (lt + log Gamma(a + 1)) / a + y / (a + 1), exact once y < 1e-9, in
// double-double with s folded into the exponent (y may be subnormal while s y is not).
// The iteration is bounded: if it has not settled after 100 steps (a cycle in the last bits
// counts as settled) there is no value, NaN.
inline double deep_inverse(double a, DD lt, bool lower, double s) {
    using special::igam_detail::log_tail;
    double y;
    if (lower) {
        const DD ly = (lt + num::log_gamma(DD{1.0, 0.0} + a)) / a;
        const double y0 = std::exp(ly.hi);
        if (y0 < 1e-9) return num::exp_dd(ly + y0 / (a + 1.0) + num::log_dd(s));
        y = std::exp(ly.hi);
    } else {  // Q ~ y^(a-1) e^-y / Gamma(a): y = -L + (a - 1) log y
        const double L = lt.hi + special::Gammaln::f(a);
        y = std::max(1.0, -L);
        for (int i = 0; i < 8; ++i) y = std::max(1.0, -L + (a - 1.0) * std::log(y));
    }
    if (!std::isfinite(y)) return s * y;
    double step = kInf;  // the last step, relative
    for (int it = 0; it < 100; ++it) {
        const auto [g, dlf] = log_tail(a, y, lower, lt);
        double yn = lower ? y * std::exp(-g / (y * dlf)) : y - g / dlf;
        if (!(yn > 0.0) || !std::isfinite(yn)) yn = lower ? 0.5 * y : 2.0 * y;
        if (std::fabs(yn - y) <= 0x1p-54 * yn) return s * yn;
        step = std::fabs(yn - y) / yn;
        y = yn;
    }
    return step <= 0x1p-48 ? s * y : kNaN;
}

// Below this a probability goes to the deep inverses (from its log in double-double).
inline constexpr double kDeepP = 0x1p-1000;

// s y with P(a, y) = p, given p and lt = log p: sd.special's gammaincinv while p and
// the root are normal numbers, else the deep inverse (a root below 1e-250 loses bits
// as a float64 before s scales it).
inline double lower_inverse(double a, double s, double p, DD lt) {
    if (p >= kDeepP) {
        const double est = (lt.hi + special::Gammaln::f(a + 1.0)) / a;  // log of the leading-term root
        if (est > -40.0) return s * special::Gammaincinv::f(a, p);
    }
    return deep_inverse(a, lt, true, s);
}
// The upper side: sd.special's gammainccinv (Boost's inverse and one Halley step on igam,
// special/ops/incgamma.hpp). Where the finite sum gives Q (finite_upper: 2a an integer, the
// root at or above a) that step is taken on it instead of igam's continued fraction: the
// same step from a value at least as accurate, with x Q' = -y^a e^-y / Gamma(a) in double
// (it multiplies a step of a few 1e-15 x: 1e-10 would do).
inline double upper_inverse(double a, double s, double q, DD lt) {
    if (q < kDeepP) return deep_inverse(a, lt, false, s);
    if (a <= kFiniteMax && 2.0 * a == std::floor(2.0 * a)) {
        const double r = special::bst::gamma_q_inv(a, q);
        double F;
        if (special::bst::normal(r) && finite_upper(a, DD{r, 0.0}, F)) {
            thread_local num::Memo memo;
            const double lg = memo.get(a, 0.0, [&] { return DD{special::Gammaln::f(a), 0.0}; }).hi;
            const double xfp = -std::exp(a * std::log(r) - r - lg);
            if (F > 0.0 && xfp < 0.0 && std::isfinite(xfp)) {
                const double u = r * ((F - q) / xfp);
                const double den = 1.0 - 0.5 * u * ((a - 1.0) / r - 1.0);
                const double rn = r - (den > 0.5 && den < 2.0 ? u / den : u);
                if (rn > 0.0 && std::isfinite(rn)) return s * rn;
            }
        }
    }
    return s * special::Gammainccinv::f(a, q);
}

// The shapes whose inverses are measured (tests/stats/ref: 1e-5 to 1e7; guaranteed to 1e6):
// outside, an inverse is returned only if the forward tail at it reproduces the target.
inline constexpr double kGuaranteedA = 1e6;
inline constexpr double kMeasuredMinA = 1e-6;

// The inverses (lower: of P, else of Q; r = s y for the target log probability lt) for a shape
// outside [kMeasuredMinA, kGuaranteedA]: NaN past kMaxShape; else the root, returned only if
// the log tail at y = r / s meets lt within the tail's change over 8 ULP of y and
// 1e-12 max(1, |lt|). The tail's relative slope y f / F is taken over the larger of the
// tail and the target (num::accept_root's rule: a tail far below the target, -inf for one
// that underflowed, must not widen its own tolerance)
// and is never above its bound: y f / P <= a (P >= y^a e^-y / Gamma(a + 1)) and
// y f / Q <= y + 1 (Q >= y^a e^-y / (Gamma(a) (y + 1)), the fraction's first convergent for
// a < 1, and without the + 1 above).
// A subnormal shape is solved at the reference shape (log_shape_ratio): P(a, y) = p <= 1/2
// has log y = log(p) / a, a root below every double; Q(a, y) = e^lt is
// Q(a0, y) = e^(lt - log(a / a0)). For a <= a0 a target Q >= 1600 a has the root 0
// (kLogRootZero). Out of line.
[[gnu::noinline, gnu::cold]] inline double inverse_beyond(double a, double s, double p, DD lt, bool lower) {
    if (!(a <= kMaxShape)) return kNaN;
    const bool subnormal = a < kMinNormalShape;
    if (subnormal) {
        if (lower) return lt.hi <= -0x1p-900 ? 0.0 : kNaN;
        lt = lt - log_shape_ratio(a);
        a = kRefShape;
    }
    if (!lower && a <= kRefShape && lt.hi >= std::log(a) + kLogRootZero) return 0.0;
    if (subnormal) p = num::exp_dd(lt);  // the shifted target: below 1600 a0
    const double r = lower ? lower_inverse(a, s, p, lt) : upper_inverse(a, s, p, lt);
    if (r != r || r < 0.0) return kNaN;
    if (r == 0.0 || std::isinf(r)) {
        // a root beyond the doubles: the tail at the last double on that side is already past
        // the target (P grows with y, Q falls)
        const bool below = r == 0.0;
        const double lf = ratio_tail(a, below ? 0x1p-1074 : std::numeric_limits<double>::max(), s, !lower, true);
        const double slack = 1e-9 * std::max(1.0, std::fabs(lt.hi));
        return ((below == lower) ? lf >= lt.hi - slack : lf <= lt.hi + slack) ? r : kNaN;
    }
    const DD y = num::ratio(r, s);
    if (!(y.hi >= 0x1p-960) || std::isinf(y.hi)) return kNaN;
    const double lf = dist_log_tail(a, y, !lower);
    const DD ld = log_density_std(a, y) + dd::log(y);  // log(y f(y))
    const double slope = std::min(std::exp(ld.hi - std::max(lf, lt.hi)), lower ? a : y.hi + 1.0);
    // (a subnormal r is itself rounded by up to 2^-1075 / r)
    const double tol = (0x1p-50 + 0x1p-1073 / r) * slope + 1e-12 * std::max(1.0, std::fabs(lt.hi));
    return std::fabs(lf - lt.hi) <= tol ? r : kNaN;  // false for NaN
}
inline double lower_inverse_checked(double a, double s, double p, DD lt) {
    if (!(a <= kGuaranteedA && a >= kMeasuredMinA)) [[unlikely]] return inverse_beyond(a, s, p, lt, true);
    return lower_inverse(a, s, p, lt);
}
inline double upper_inverse_checked(double a, double s, double q, DD lt) {
    if (!(a <= kGuaranteedA && a >= kMeasuredMinA)) [[unlikely]] return inverse_beyond(a, s, q, lt, false);
    return upper_inverse(a, s, q, lt);
}

// The differential entropy of gamma(a, 1): a + log Gamma(a) + (1 - a) psi(a)
// = G(a) + log(a)/2 + log(2 pi)/2 (stats/numeric.hpp: G is O(1); the terms grow
// like a log a).
inline DD entropy_std(double a) {
    return num::log_dd(a) * 0.5 + num::kLogSqrt2Pi + num::entropy_g(a, special::Digamma::f);
}

// log Gamma(z) for z > 0 a double-double, as num::log_gamma (to ~1e-19 absolutely) at less
// than half its cost for a small z: the argument is shifted to w >= 13 only (at most 13
// double-double products, not 30) and Stirling's series S(w) = sum B_2k / (2k (2k-1) w^(2k-1))
// runs to k = 8, whose first omitted term is 2e-20 at w = 13. The densities' chunk kernels
// take their constants from it (once per distinct parameters).
inline DD log_gamma_dd(DD z) {
    DD w = z, prod{1.0, 0.0};
    while (w.hi < 13.0) {
        prod = prod * w;
        w = w + 1.0;
    }
    const DD lw = dd::log(w);
    const double r = 1.0 / w.hi, r2 = r * r;
    double series = 0.0;
    for (int k = 8; k >= 1; --k) series = series * r2 + num::kBernoulli2k[k - 1] / (2.0 * k * (2.0 * k - 1.0));
    DD l = w * lw - w - lw * 0.5 + num::kLogSqrt2Pi + series * r;
    if (prod.hi != 1.0 || prod.lo != 0.0) l = l - dd::log(prod);
    return l;
}

// log Gamma(a) + log s as a double-double (the density kernel's constant);
// log Gamma(a) = log Gamma(1 + a) - log a (nothing large to round for a tiny a).
inline DD log_norm(double a, double s) { return log_gamma_dd(DD{1.0, 0.0} + a) - num::log_dd(a) + num::log_dd(s); }

// Past this a parameter's log-gammas (terms of a log a, double-doubles) are no longer good to
// a hundredth of a unit: the kernels leave such parameters to the scalar definitions.
inline constexpr double kPlainMax = 1e10;

}  // namespace gamma_detail

// ---- the densities' chunk kernel (gamma, chi-square, inverse gamma, beta, t, F) ----------------
//
// Each of these log densities is a sum
//     l = k1 log(x1) + k2 log(x2) - lin + c
// of at most two logs with coefficients from the shapes, a linear term and a constant of
// the parameters (log-gammas). The scalar definitions form it in double-double from
// Stirling-normalised pieces (special/prefix.hpp), which is what the far tails and huge
// shapes need, at 20 to 80 ns. The kernel forms the same sum directly: k, x and lin are
// double-doubles from exact operations (two_sum, two_prod, a division's remainder by fma),
// c is a double-double (once per distinct parameters), the products k log x are two_prod's
// and the sum is two_sum's, with the low parts gathered in a double (their own rounding is
// 2^-53 of 2^-53 of the terms). What is left is the logs' error, so the absolute error of
// l = lh + ll is at most
//     T = e (|k1 log x1| + |k2 log x2|) + 1   units of 2^-53
// (the 1 for everything of second order), with the logs in two tiers:
//     1. vForce's log of x.hi plus x.lo / x.hi: e = kVfLog. Enough for small shapes in the
//        bulk (|k log x| up to ~2.6), at ~1 ns a log. Taken where T passes for the density
//        (T + kVfExp + 2 <= kFastBudget), for logpdf too: one rule from the logs alone;
//     2. elsewhere dens::log below (2^-63 relative): e = 2^-10, which holds until a term
//        passes ~1e4.
// The scalar definition keeps what is left: huge shapes near the mode, far tails, the ends
// of the support, non-finite values.
// logpdf returns lh + ll rounded (one more unit of |l|) under the mixed bound (T20):
// T + |l| <= kFastBudget max(1, |l|). pdf returns e^lh (1 + ll) with vForce's exp: T, kVfExp
// and two roundings, while the result is a normal number.
namespace dens {

using num::DD;
namespace dd = num::dd;

// log(x) for x = hi + lo > 0, hi a normal number, as a double-double: special/dd.hpp's
// table reduction (x.hi = 2^k c (1 + r), c = i/128, |r| <= 2^-7.58, r = rh + rl exactly to
// 2^-106) with the series cut to what a density needs,
//     log(1 + r) = r - r^2/2 + r^3 (1/3 - r/4 + ... - r^7/10),
// r^2 by two_prod and the cubic tail in double, plus x.lo / x.hi. The tail's rounding
// (2^-49 |r|^3 <= 2^-71.7 absolutely, with the low sum's) is the error; |log x| is at least
// 2^-8.5 unless c = 1 and k = 0, where log x ~ r and the error is 2^-64 of it: 2^-63
// relative everywhere (measured against dd::log on 2e6 points: 2^-68.4). 3 ns to dd::log's 8.
inline DD log(DD x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x.hi, sizeof bits);
    int k = static_cast<int>(bits >> 52) - 1023;
    bits = (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m;
    std::memcpy(&m, &bits, sizeof m);  // in [1, 2)
    const bool up = m >= 1.5;
    m = up ? 0.5 * m : m;
    k += up;
    const int i = static_cast<int>(m * 128.0 + 0.5);  // 96 .. 192
    const dd::LogPoint& c = dd::kLogTable[static_cast<std::size_t>(i - 96)];
    const double cv = i * 0x1p-7;
    const double d = m - cv;  // exact
    const double rh = d * c.inv;
    const double rl = std::fma(-rh, cv, d) * c.inv;
    const DD r2 = dd::two_prod(rh, rh);
    const double tail =
        rh * r2.hi *
        (1.0 / 3 + rh * (-0.25 + rh * (0.2 + rh * (-1.0 / 6 + rh * (1.0 / 7 + rh * (-0.125 + rh * (1.0 / 9 + rh * -0.1)))))));
    const double kd = k;
    const DD kl = dd::two_prod(dd::kLn2.hi, kd);
    DD s = dd::two_sum(kl.hi, c.hi);
    double lo = s.lo;
    s = dd::two_sum(s.hi, rh), lo += s.lo;
    s = dd::two_sum(s.hi, -0.5 * r2.hi), lo += s.lo;
    lo += (kl.lo + dd::kLn2.lo * kd + c.lo) + (rl - rh * rl - 0.5 * r2.lo) + tail + x.lo / x.hi;
    return dd::quick_two_sum(s.hi, lo);
}
inline constexpr double kLog = 0x1p-10;  // its error in units of 2^-53 per unit of |log x|

struct In {
    DD k1, x1, k2, x2, lin, c;
    bool ok;  // the element may take the kernel: x1.hi, x2.hi normal and positive, c finite
};
inline constexpr In kBlank = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, false};

// lh + ll = k1 L1 + k2 L2 - lin + c; returns |k1 L1| + |k2 L2|.
inline double combine(const In& in, DD L1, DD L2, double& lh, double& ll) {
    const DD p1 = dd::two_prod(in.k1.hi, L1.hi), p2 = dd::two_prod(in.k2.hi, L2.hi);
    DD s = dd::two_sum(p1.hi, p2.hi);
    double lo = s.lo + (p1.lo + p2.lo);
    s = dd::two_sum(s.hi, -in.lin.hi), lo += s.lo;
    s = dd::two_sum(s.hi, in.c.hi), lo += s.lo;
    lo += (in.k1.hi * L1.lo + in.k1.lo * L1.hi) + (in.k2.hi * L2.lo + in.k2.lo * L2.hi) + (in.c.lo - in.lin.lo);
    lh = s.hi + lo;
    ll = lo - (lh - s.hi);
    return std::fabs(p1.hi) + std::fabs(p2.hi);
}
// Whether l's error bound t (units of 2^-53) leaves the result inside the budget.
inline bool within(bool log, double t, double al) {
    return log ? al <= 0x1p1000 && t + al <= kFastBudget * std::max(1.0, al)
               : t + ck::kVfExp + 2.0 <= kFastBudget && al < 700.0;
}

inline constexpr int kBlock = 128;  // the working set stays in the first-level cache

// args(i) gives element i's In; precise(i) the scalar definition (for any x: NaN and
// invalid parameters included). Two: whether the second log is used.
template <bool Two, class Args, class Precise>
inline void kernel(int n, bool log, double* out, Args&& args, Precise&& precise) {
    In in[kBlock];
    double l1[kBlock], l2[kBlock], one_ll[kBlock];
    bool fast[kBlock];
    for (int off = 0; off < n; off += kBlock) {
        const int m = std::min(kBlock, n - off);
        double* o = out + off;
        for (int i = 0; i < m; ++i) {
            in[i] = args(off + i);
            l1[i] = in[i].x1.hi;
            if constexpr (Two) l2[i] = in[i].x2.hi;
        }
        ck::log(l1, l1, m);
        if constexpr (Two) ck::log(l2, l2, m);
        for (int i = 0; i < m; ++i) {
            const In& e = in[i];
            double lh = 0.0, ll = 0.0;
            fast[i] = false;
            if (e.ok) {
                // the first tier where its bound passes for the density (then it does for logpdf
                // too: T + |l| <= 11.5 + |l| <= kFastBudget max(1, |l|)); the test needs only the logs
                const double t1 = ck::kVfLog * (std::fabs(e.k1.hi * l1[i]) + (Two ? std::fabs(e.k2.hi * l2[i]) : 0.0)) + 1.0;
                if (t1 + ck::kVfExp + 2.0 <= kFastBudget) {
                    combine(e, DD{l1[i], e.x1.lo / e.x1.hi}, Two ? DD{l2[i], e.x2.lo / e.x2.hi} : DD{0.0, 0.0}, lh, ll);
                    fast[i] = std::fabs(lh) < 700.0;
                } else {
                    const double t2 = combine(e, dens::log(e.x1), Two ? dens::log(e.x2) : DD{0.0, 0.0}, lh, ll);
                    fast[i] = within(log, kLog * t2 + 1.0, std::fabs(lh));
                }
            }
            o[i] = lh, one_ll[i] = 1.0 + ll;
        }
        if (!log) {
            ck::exp(o, o, m);
            for (int i = 0; i < m; ++i) o[i] *= one_ll[i];
        }
        ck::fallback(m, fast, o, [&](int i) { return precise(off + i); });
    }
}

// The constant of the last parameters seen: one evaluation per run of equal parameters (the
// kernels keep one per thread, so a scalar call per element does not recompute it either).
// Always the same function of the parameters, whatever the layout.
struct Const {
    double a = kNaN, b = kNaN;
    DD v{kNaN, 0.0};
    template <class F>
    DD get(double x, double y, F&& f) {
        if (!(x == a && y == b)) v = f(), a = x, b = y;
        return v;
    }
};

// num / den as a double-double (the remainder by fma); the caller checks the range.
inline DD quotient(double num, double den) {
    const double q = num / den;
    return {q, std::fma(-q, den, num) / den};
}

// The gamma-type densities: y = x / s, k = a - 1 for the gamma; y = s / x, k = a + 1 for the
// inverse gamma (its density is the gamma's at y times y^2 / s):
//     l = k log y - y - (log Gamma(a) + log s).
template <class Precise>
inline void gamma_kernel(int n, const double* x, const double* a, const double* s, bool inverse, bool log, double* out,
                         Precise&& precise) {
    thread_local Const cst_gamma, cst_inverse;  // (one each: a gamma and an inverse gamma may alternate)
    Const& cst = inverse ? cst_inverse : cst_gamma;
    kernel<false>(
        n, log, out,
        [&](int i) {
            In in = kBlank;
            const DD c = cst.get(a[i], s[i], [&] {
                const bool valid = a[i] > 0.0 && a[i] <= gamma_detail::kPlainMax && s[i] > 0.0 && std::isfinite(s[i]);
                return valid ? gamma_detail::log_norm(a[i], s[i]) : DD{kNaN, 0.0};
            });
            in.x1 = inverse ? quotient(s[i], x[i]) : quotient(x[i], s[i]);
            in.k1 = dd::two_sum(a[i], inverse ? 1.0 : -1.0);
            in.lin = in.x1;
            in.c = -c;
            in.ok = x[i] > 0.0 && in.x1.hi >= 0x1p-1000 && in.x1.hi <= 0x1p1000 && c.hi == c.hi;  // false for a NaN
            return in;
        },
        precise);
}

}  // namespace dens

namespace gamma_detail {

}  // namespace gamma_detail

struct Gamma {
    static constexpr char name[] = "gamma";
    static constexpr std::array<const char*, 2> params{"a", "scale"};

    static bool valid(double a, double s) { return a > 0.0 && std::isfinite(a) && s > 0.0 && std::isfinite(s); }

    // log density of X at x > 0 finite.
    static num::DD log_density(double x, double a, double s) {
        const num::DD y = num::ratio(x, s);
        if (std::isinf(y.hi)) return {-kInf, 0.0};  // x / s overflows: e^-y is 0
        if (y.hi < 0x1p-960) {
            // y has lost bits (or is 0): (a - 1) log y - log Gamma(a) - log s from the logs (y
            // itself is nothing); log Gamma(a) = log Gamma(1 + a) - log a
            const num::DD ly = num::log_dd(x) - num::log_dd(s);
            return ly * a - ly - (num::lgamma1p(a) - num::log_dd(a)) - num::log_dd(s);
        }
        thread_local num::Memo memo;
        return gamma_detail::log_density_std(a, y) - memo.get(s, 0.0, [&] { return num::log_dd(s); });
    }
    // The double-double definitions (the kernel's fallback: the tails, huge shapes, the ends).
    static double pdf_dd(double x, double a, double s) {
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return a < 1.0 ? kInf : (a == 1.0 ? 1.0 / s : 0.0);
        return num::exp_dd(log_density(x, a, s));
    }
    static double logpdf_dd(double x, double a, double s) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return a < 1.0 ? kInf : (a == 1.0 ? -std::log(s) : -kInf);
        const num::DD l = log_density(x, a, s);
        return l.hi + l.lo;
    }
    // pdf and logpdf: dens::kernel, and the definitions above for what it leaves. The scalar
    // forms are the kernel on one element, so every caller gets the same bits.
    static void density_chunk(const Chunk<2>& c, double* out, bool log) {
        dens::gamma_kernel(c.n, c.x, c.p[0], c.p[1], false, log, out, [&](int i) {
            if (!(c.x[i] == c.x[i] && valid(c.p[0][i], c.p[1][i]))) return kNaN;  // (the engine's NaNs)
            return log ? logpdf_dd(c.x[i], c.p[0][i], c.p[1][i]) : pdf_dd(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, false); }
    static void logpdf_chunk(const Chunk<2>& c, double* out) { density_chunk(c, out, true); }
    static double density_one(double x, double a, double s, bool log) {
        double out;
        density_chunk(Chunk<2>{1, &x, {&a, &s}, {true, true}}, &out, log);
        return out;
    }
    static double pdf(double x, double a, double s) { return density_one(x, a, s, false); }
    static double logpdf(double x, double a, double s) { return density_one(x, a, s, true); }
    static double cdf(double x, double a, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return 1.0;
        return gamma_detail::ratio_tail(a, x, s, false, false);
    }
    static double ccdf(double x, double a, double s) {
        if (x <= 0.0) return 1.0;
        if (std::isinf(x)) return 0.0;
        return gamma_detail::ratio_tail(a, x, s, true, false);
    }
    static double logcdf(double x, double a, double s) {
        if (x <= 0.0) return -kInf;
        if (std::isinf(x)) return 0.0;
        return gamma_detail::ratio_tail(a, x, s, false, true);
    }
    static double logccdf(double x, double a, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return -kInf;
        return gamma_detail::ratio_tail(a, x, s, true, true);
    }

    // The side solved is the smaller of p and 1 - p (1 - p exact for p >= 1/2).
    static double icdf(double p, double a, double s) {
        if (a > gamma_detail::kMaxShape) return kNaN;
        if (!(p > 0.0 && p < 1.0)) return special::Gammaincinv::f(a, p) * s;  // ends and NaN
        if (p > 0.5) return gamma_detail::upper_inverse_checked(a, s, 1.0 - p, num::log_dd(1.0 - p));
        return gamma_detail::lower_inverse_checked(a, s, p, num::log_dd(p));
    }
    static double iccdf(double p, double a, double s) {
        if (a > gamma_detail::kMaxShape) return kNaN;
        if (!(p > 0.0 && p < 1.0)) return special::Gammainccinv::f(a, p) * s;
        if (p > 0.5) return gamma_detail::lower_inverse_checked(a, s, 1.0 - p, num::log_dd(1.0 - p));
        return gamma_detail::upper_inverse_checked(a, s, p, num::log_dd(p));
    }
    // From log p: the side with the smaller probability, e^lp (lp exact) or 1 - e^lp.
    static double from_log(double lp, double a, double s, bool upper) {
        if (!(lp <= 0.0) || a > gamma_detail::kMaxShape) return kNaN;
        if (lp == 0.0) return upper ? 0.0 : kInf;
        if (lp == -kInf) return upper ? kInf : 0.0;
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return upper ? gamma_detail::lower_inverse_checked(a, s, q, num::log_dd(q))
                         : gamma_detail::upper_inverse_checked(a, s, q, num::log_dd(q));
        }
        const double p = std::exp(lp);
        return upper ? gamma_detail::upper_inverse_checked(a, s, p, num::DD{lp, 0.0})
                     : gamma_detail::lower_inverse_checked(a, s, p, num::DD{lp, 0.0});
    }
    static double ilogcdf(double lp, double a, double s) { return from_log(lp, a, s, false); }
    static double ilogccdf(double lp, double a, double s) { return from_log(lp, a, s, true); }

    static double mean(double a, double s) { return a * s; }
    static double variance(double a, double s) { return a * s * s; }
    static double standard_deviation(double a, double s) { return std::sqrt(a) * s; }
    static double skewness(double a, double) { return 2.0 / std::sqrt(a); }
    static double kurtosis(double a, double) { return 6.0 / a; }
    static double mode(double a, double s) { return a > 1.0 ? (a - 1.0) * s : 0.0; }  // a - 1 exact
    static double entropy(double a, double s) {
        const num::DD h = gamma_detail::entropy_std(a) + num::log_dd(s);
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

// chi-square(df) = gamma(df/2, 2): y = x/2 is exact.
struct ChiSquared {
    static constexpr char name[] = "chisquared";
    static constexpr std::array<const char*, 1> params{"df"};

    static bool valid(double df) { return df > 0.0 && std::isfinite(df); }

    // gamma(df/2, 2)'s kernel (Gamma::density_chunk: the same bits).
    static void density_chunk_raw(const Chunk<1>& c, double* out, bool log) {
        double a[kChunkSize], s[kChunkSize];
        for (int i = 0; i < c.n; ++i) a[i] = 0.5 * c.p[0][i], s[i] = 2.0;
        dens::gamma_kernel(c.n, c.x, a, s, false, log, out, [&](int i) {
            if (!(c.x[i] == c.x[i] && valid(c.p[0][i]))) return kNaN;
            return log ? Gamma::logpdf_dd(c.x[i], a[i], 2.0) : Gamma::pdf_dd(c.x[i], a[i], 2.0);
        });
    }
    // A df below 2^-1021 has no exact half (the beta and gamma shapes are df / 2): NaN, as the
    // cdfs. Checked once for broadcast parameters.
    static void density_chunk(const Chunk<1>& c, double* out, bool log) {
        density_chunk_raw(c, out, log);
        if (c.same[0]) {
            if (!(c.p[0][0] < 0x1p-1021)) return;
            for (int i = 0; i < c.n; ++i) out[i] = kNaN;
            return;
        }
        for (int i = 0; i < c.n; ++i) out[i] = (c.p[0][i] < 0x1p-1021) ? kNaN : out[i];
    }
    static void pdf_chunk(const Chunk<1>& c, double* out) { density_chunk(c, out, false); }
    static void logpdf_chunk(const Chunk<1>& c, double* out) { density_chunk(c, out, true); }
    static double density_one(double x, double df, bool log) {
        double out;
        density_chunk(Chunk<1>{1, &x, {&df}, {true}}, &out, log);
        return out;
    }
    static double pdf(double x, double df) { return density_one(x, df, false); }
    static double logpdf(double x, double df) { return density_one(x, df, true); }
    // The shape is df / 2: below 2^-1021 a df has no exact half (the smallest subnormals), and
    // the methods through the gamma functions are NaN.
    static bool halvable(double df) { return df >= 0x1p-1021; }
    static double cdf(double x, double df) { return halvable(df) ? Gamma::cdf(x, 0.5 * df, 2.0) : kNaN; }
    static double ccdf(double x, double df) { return halvable(df) ? Gamma::ccdf(x, 0.5 * df, 2.0) : kNaN; }
    static double logcdf(double x, double df) { return halvable(df) ? Gamma::logcdf(x, 0.5 * df, 2.0) : kNaN; }
    static double logccdf(double x, double df) { return halvable(df) ? Gamma::logccdf(x, 0.5 * df, 2.0) : kNaN; }
    static double icdf(double p, double df) { return halvable(df) ? Gamma::icdf(p, 0.5 * df, 2.0) : kNaN; }
    static double iccdf(double p, double df) { return halvable(df) ? Gamma::iccdf(p, 0.5 * df, 2.0) : kNaN; }
    static double ilogcdf(double lp, double df) { return halvable(df) ? Gamma::ilogcdf(lp, 0.5 * df, 2.0) : kNaN; }
    static double ilogccdf(double lp, double df) { return halvable(df) ? Gamma::ilogccdf(lp, 0.5 * df, 2.0) : kNaN; }

    static double mean(double df) { return df; }
    static double variance(double df) { return 2.0 * df; }
    static double standard_deviation(double df) { return std::sqrt(2.0 * df); }
    static double skewness(double df) {
        if (std::isinf(8.0 / df)) return 2.8284271247461903 / std::sqrt(df);  // sqrt 8 / sqrt df
        return std::sqrt(8.0 / df);
    }
    static double kurtosis(double df) { return 12.0 / df; }
    static double median(double df) { return halvable(df) ? Gamma::icdf(0.5, 0.5 * df, 2.0) : kNaN; }
    static double mode(double df) { return df > 2.0 ? df - 2.0 : 0.0; }
    static double entropy(double df) { return halvable(df) ? Gamma::entropy(0.5 * df, 2.0) : kNaN; }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double) { return kInf; }
};

}  // namespace sd::stats
