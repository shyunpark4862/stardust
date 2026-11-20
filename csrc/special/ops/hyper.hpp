// The confluent and Gauss hypergeometric functions, real arguments
// (docs/api/special.md §11, docs/design/special.md §5.11; S7).
//
// hyp1f1(a, b, x) = 1F1(a; b; x) = M(a, b, x):
//   x >= 0   the Maclaurin series sum (a)_k / (b)_k x^k / k!, carried with its
//            binary exponent apart (the terms reach e^x); for x past
//            600 + 2(|a| + |b|) the asymptotic expansion
//            Gamma(b)/Gamma(a) e^x x^(a-b) sum (b-a)_s (1-a)_s / s! x^-s
//   x < 0    Kummer's transformation M(a, b, x) = e^x M(b - a, b, -x)
// The guaranteed region (S7 user decision) is where one of these has positive
// terms only: a >= 0, b > 0 and (x >= 0 or b >= a). Elsewhere (the loose region) both
// forms alternate, and a value is given only where its loss is known (m_loose): a series
// that kept kMaxCancel, the recurrence in a for b > 0, the large-argument expansions,
// Boost for |a|, |b| <= kBoostLoose, Tricomi's expansion; else NaN.
//
// hyp2f1(a, b, c, z), z < 1:
//   0 <= z <= 0.9    the Maclaurin series
//   -1 <= z < 0      Pfaff: (1-z)^-a 2F1(a, c-b; c; z/(z-1)), the larger of a, b
//                    first so that c - b >= 0 when c >= min(a, b)
//   z > 0.9          the transformation to 1 - z (AS 15.3.6, every Gamma as m 2^e at its
//                    exact hi + lo argument), and its limit forms with digamma when
//                    c - a - b is an integer (AS 15.3.10-12)
//   z < -1           Pfaff to w = z/(z-1) in (1/2, 1), then as above in w
// The guaranteed region: a, b >= 0, c > 0, c >= min(a, b), -1 <= z <= 0.9.
// Polynomials (a or b a non-positive integer) are summed exactly as far as
// they go, for any z; c a non-positive integer is a pole (inf) unless the
// polynomial ends first. Outside the guaranteed region a sum that lost more than
// kMaxCancel2 to cancellation, or a transformation whose two terms cancel more than
// kMaxTransformCancel, is NaN.
#pragma once

#include "special/boost.hpp"

#include "special/common.hpp"
#include "special/cost.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"
#include "special/ops/bessel.hpp"
#include "special/ops/gamma.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace sd::special {

namespace hyper_detail {

inline bool nonpos_int(double v) { return v <= 0.0 && v == std::floor(v); }

// For the sign of a polynomial's leading coefficient without a loop over its degree:
// how many of v, v+1, ..., v+n-1 are negative (n an integer-valued count), and whether an
// integer-valued v is odd (every double from 2^53 is even).
inline double neg_count(double v, double n) { return -v > 0.0 ? std::min(n, std::ceil(-v)) : 0.0; }
inline bool is_odd(double v) { return std::fabs(v) < 0x1p53 && std::fmod(v, 2.0) != 0.0; }

// ---- 1F1 ------------------------------------------------------------------------

// sum (a)_k / (b)_k y^k / k! as m 2^e, and the ratio sum|terms| / |sum| (the
// cancellation). y >= 0 or < 0 (alternating); stops at 2^-56 once the terms fall.
struct Series {
    Scaled s;
    double cancel;
    double lo = 0.0;  // the sum is (s.m + lo) 2^s.e (the long series; 0 from the float64 pass)
};
// The float64 pass of a series is kept while sum (k+1)|t_k| <= kFastSeries |s|. A step
// rounds about six times (a + k, the products, b + k, the division), so term k carries
// at most ~3 (k + 1) ULP and the accepted sum at most ~3 kFastSeries = 12 ULP (well
// inside the 45 of the contract; measured: 4.1 ULP at most, bench/special/opt3 at commit 4151d2a).
inline constexpr double kFastSeries = 4.0;
// A series that has not ended after kMaxSeriesTerms terms gives NaN (about 20 ms).
inline constexpr int kMaxSeriesTerms = 1000000;
inline constexpr Series kNoSeries{{kNaN, 0}, kInf};
// A compensated sum that came out 0: the true sum is below 2^-104 sum|terms|. With terms of
// modest size that is an exact zero of a short polynomial (1 - x at x = 1), kept as 0 (no
// cancellation reported); with large ones it is every digit lost (4e18 - 4e18 for 1.31).
inline double zero_sum_cancel(double abssum) { return abssum <= 1024.0 ? 1.0 : kInf; }
// The series with positive terms only (a >= 0, b > 0, y > 0: every sum of the guaranteed
// region): nothing cancels and the terms rise once and fall once.
// positive_only: the caller wants the sum itself and every term is positive (a, b, y > 0):
// once the partial sum is past every double the sum is inf, without the remaining terms.
inline Series m_series_positive(double a, double b, double y, double a_lo, bool positive_only) {
    // The terms in double-double: by successive products a float64 term k
    // carries ~k ULP (a + k, b + k rounded, then multiplied), and near x ~ 500
    // the sum is made of terms around k ~ x: 120-180 ULP, measured. a + k and
    // b + k are exact as two_sums.
    if (a_lo == 0.0) {
        // First in float64 with a bound on what the products cost (kFastSeries): where
        // it is small (the terms fall from the start: most inputs) this is the answer, at
        // a fifth of the compensated loop's time.
        dd::Sum fsum(1.0);
        double ft = 1.0, fabs_sum = 1.0, weighted = 1.0;
        bool done = false;
        for (int k = 0; k < 2000; ++k) {
            if (a + k == 0.0) {
                done = true;
                break;
            }
            ft *= (a + k) * y / ((b + k) * (k + 1.0));
            fsum.add(ft);
            const double at = std::fabs(ft);
            fabs_sum += at;
            weighted += (k + 2.0) * at;
            if (!(at < 0x1p600)) break;
            if (ft == 0.0 || (k > std::fabs(y) && k > -b && at <= 0x1p-56 * std::fabs(fsum.hi))) {
                done = true;
                break;
            }
        }
        const double fs = fsum.total();
        if (done && weighted <= kFastSeries * std::fabs(fs))
            return {scaled(fs), fs == 0.0 ? kInf : fabs_sum / std::fabs(fs)};
    }
    // The long series, compensated (as f_series2 below: the roundings of the product
    // and the quotient by fma, to first order; the term and the sum as hi + lo).
    double th = 1.0, tl = 0.0, abssum = 1.0;
    dd::Sum sum(1.0);
    std::int64_t e = 0;
    bool ended = false;
    for (int k = 0; k < kMaxSeriesTerms; ++k) {
        if (a + k == 0.0) {  // a polynomial ends (before b + k = 0 can divide)
            ended = true;
            break;
        }
        const dd::DD A = dd::two_sum(a, k), B = dd::two_sum(b, k);
        const double kp = k + 1.0;
        const double n = A.hi * y, e1 = std::fma(A.hi, y, -n) + (A.lo + a_lo) * y;
        const double d = B.hi * kp, e2 = std::fma(B.hi, kp, -d) + B.lo * kp;
        const double rd = 1.0 / d, q = n * rd;
        const double ql = (std::fma(-q, d, n) + e1 - q * e2) * rd;  // the ratio is q + ql
        const double tn = th * q;
        tl = std::fma(th, q, -tn) + tl * q + th * ql;
        th = tn;
        sum.add(th, tl);
        abssum += std::fabs(th);
        if (th == 0.0) {
            ended = true;
            break;
        }
        if (std::fabs(sum.hi) > 0x1p600 || std::fabs(th) > 0x1p600) {
            sum.scale(-600);
            th *= 0x1p-600, tl *= 0x1p-600;
            abssum *= 0x1p-600;
            e += 600;
            if (positive_only && e > 1800) return {{kInf, 0}, 1.0};
        }
        // past the peak (k > |y|), and past b + k = 0 where a negative b makes the terms grow again
        if (k > std::fabs(y) && k > -b && std::fabs(th) <= 0x1p-56 * std::fabs(sum.hi)) {
            ended = true;
            break;
        }
    }
    if (!ended || !std::isfinite(sum.hi)) return kNoSeries;
    Scaled r = scaled(sum.total());
    r.e += e;
    return {r, sum.hi == 0.0 ? kInf : abssum / std::fabs(sum.hi)};
}

// The series for any signs.
// max_terms, max_cancel: a caller that only wants a cheap, well-conditioned sum gets kNoSeries
// instead of the long loop when the series needs more terms than max_terms, or when the float64
// pass already proves the cancellation above max_cancel.
// precise: the long series at once, for a caller that wants the sum beyond float64 (lo).
inline Series m_series(
    double a,
    double b,
    double y,
    double a_lo = 0.0,
    bool positive_only = false,
    int max_terms = kMaxSeriesTerms,
    double max_cancel = kInf,
    bool precise = false
) {
    if (a >= 0.0 && b > 0.0 && y > 0.0 && !precise) {
        const Series r = m_series_positive(a, b, y, a_lo, positive_only);
        if (!std::isnan(r.s.m)) return r;  // (NaN: a subnormal b, below)
    }
    // The terms in double-double: by successive products a float64 term k
    // carries ~k ULP (a + k, b + k rounded, then multiplied), and near x ~ 500
    // the sum is made of terms around k ~ x: 120-180 ULP, measured. a + k and
    // b + k are exact as two_sums.
    if (a_lo == 0.0 && !precise) {
        // First in float64 with a bound on what the products cost (kFastSeries): where
        // it is small (the terms fall from the start: most inputs) this is the answer, at
        // a fifth of the compensated loop's time.
        dd::Sum fsum(1.0);
        double ft = 1.0, fabs_sum = 1.0, weighted = 1.0;
        bool done = false;
        for (int k = 0; k < 2000; ++k) {
            if (a + k == 0.0) {
                done = true;
                break;
            }
            const double ratio = (a + k) * y / ((b + k) * (k + 1.0));
            ft *= ratio;
            fsum.add(ft);
            const double at = std::fabs(ft);
            fabs_sum += at;
            weighted += (k + 2.0) * at;
            if (!(at < 0x1p600)) break;
            if (at < 0x1p-700 && k < -b) break;  // (the long series keeps such terms scaled)
            if (ft == 0.0 ||
                (k > std::fabs(y) && k > -b && std::fabs(ratio) < 1.0 && at <= 0x1p-56 * std::fabs(fsum.hi))) {
                done = true;
                break;
            }
        }
        const double fs = fsum.total();
        if (done && weighted <= kFastSeries * std::fabs(fs))
            return {scaled(fs), fs == 0.0 ? kInf : fabs_sum / std::fabs(fs)};
        // |fs - sum| <= 2^-51 weighted (the bound above), so |sum| <= |fs| + that
        if (done && fabs_sum > max_cancel * (std::fabs(fs) + 0x1p-51 * weighted)) return kNoSeries;
    }
    // The long series, compensated (as f_series2 below: the roundings of the product
    // and the quotient by fma, to first order; the term and the sum as hi + lo).
    // The sum is sum 2^e and the term (th + tl) 2^(e - lag). lag > 0 only for b < 0, while the
    // terms fall below the normal range before b + k = 0: past it they grow again by
    // 1/|b + k| factors and can be the whole sum, so they must not be built on a rounded or
    // vanished term (hyp1f1(0.0106, -2370.1, 731) was 0.997 for -1.75e132).
    double th = 1.0, tl = 0.0, abssum = 1.0;
    dd::Sum sum(1.0);
    std::int64_t e = 0;
    int lag = 0;
    bool ended = false;
    for (int k = 0; k < max_terms; ++k) {
        if (a + k == 0.0) {  // a polynomial ends (before b + k = 0 can divide)
            ended = true;
            break;
        }
        const dd::DD A = dd::two_sum(a, k), B = dd::two_sum(b, k);
        const double kp = k + 1.0;
        const double n = A.hi * y, e1 = std::fma(A.hi, y, -n) + (A.lo + a_lo) * y;
        const double d = B.hi * kp, e2 = std::fma(B.hi, kp, -d) + B.lo * kp;
        // (a subnormal b: 1/d would overflow where the quotient does not)
        const bool tiny = std::fabs(d) < 0x1p-1000;
        const double rd = 1.0 / d, q = tiny ? n / d : n * rd;
        const double rem = std::fma(-q, d, n) + e1 - q * e2;
        const double ql = tiny ? rem / d : rem * rd;  // the ratio is q + ql
        const double tn = th * q;
        tl = std::fma(th, q, -tn) + tl * q + th * ql;
        th = tn;
        double at = std::fabs(th);
        if (lag == 0 && !(at < 0x1p-700 && k < -b)) [[likely]] {
            sum.add(th, tl);
            abssum += at;
        } else {
            if (at < 0x1p-700 && k < -b) {
                th *= 0x1p600, tl *= 0x1p600;
                lag += 600;
            } else if (at > 0x1p100) {
                th *= 0x1p-600, tl *= 0x1p-600;
                lag -= 600;
            }
            // (a term 2^-1200 below the sum's scale adds nothing: left out)
            at = lag > 1200 ? 0.0 : std::ldexp(std::fabs(th), -lag);
            if (lag <= 1200) sum.add(std::ldexp(th, -lag), std::ldexp(tl, -lag));
            abssum += at;
        }
        if (th == 0.0) {
            ended = true;
            break;
        }
        if (std::fabs(sum.hi) > 0x1p600 || std::fabs(th) > 0x1p600) {  // (th: only with lag = 0)
            sum.scale(-600);
            th *= 0x1p-600, tl *= 0x1p-600;
            abssum *= 0x1p-600;
            e += 600;
            if (positive_only && e > 1800) return {{kInf, 0}, 1.0};
        }
        // past the peak (k > |y|), and past b + k = 0 where a negative b makes the terms grow
        // again: they keep growing while the ratio is above 1 (right after b + k = 0 they are
        // still far below the sum). Against hi + lo: under cancellation hi alone is the
        // uncompensated sum (off by 2^-53 sum|terms|), which would end the series early.
        if (k > std::fabs(y) && k > -b && std::fabs(q) < 1.0 && at <= 0x1p-56 * std::fabs(sum.hi + sum.lo)) {
            ended = true;
            break;
        }
    }
    if (!ended || !std::isfinite(sum.hi)) return kNoSeries;
    const double total = sum.total();
    int te = 0;
    const double tm = std::frexp(total, &te);
    return {
        {tm, te + e},
        total == 0.0 ? zero_sum_cancel(abssum) : abssum / std::fabs(total),
        std::ldexp((sum.hi - total) + sum.lo, -te)
    };
}

// Gamma(b) / Gamma(a) with its sign, as m 2^e (Cephes Gamma up to 170; beyond it each Gamma
// as m 2^e, a negative argument through the reflection: the difference of two logs of size
// z log z would carry z log z 2^-53, 900 ULP at 1000).
inline Scaled gamma_ratio_signed(double b, double a) {
    if (std::fabs(a) < 170.0 && std::fabs(b) < 170.0) return scaled(Rgamma::f(a)) / scaled(Rgamma::f(b));
    const Scaled num = b > 0.0 ? gamma_scaled(b) : gamma_detail::gamma_neg(b);
    if (a > 0.0) return num / gamma_scaled(a);
    return nonpos_int(a) ? Scaled{0.0, 0} : num * gamma_detail::rgamma_neg(a);
}

// Large y > 0: Gamma(b)/Gamma(a) e^y y^(a-b) sum (b-a)_s (1-a)_s / s! y^-s; false if not converged.
inline bool m_asymptotic(double a, double b, double y, Scaled& out) {
    double s = 1.0, t = 1.0, prev = 1.0;
    for (int k = 0; k < 500; ++k) {
        t *= (b - a + k) * (1.0 - a + k) / ((k + 1.0) * y);
        const double at = std::fabs(t);
        if (at > prev && k > 2) return false;
        prev = at;
        s += t;
        if (t == 0.0 || at <= 0x1p-56 * std::fabs(s)) {
            out = gamma_ratio_signed(b, a) * exp_scaled(y) * pow_scaled(y, a - b) * scaled(s);
            return true;
        }
    }
    return false;
}

// M(a, b, y) for y >= 0, finite, b not a pole; a = a_hi + a_lo (Kummer's b - a).
inline Series m_nonneg(double a, double b, double y, double a_lo = 0.0, bool positive_only = false) {
    if (!nonpos_int(a) && a_lo == 0.0 && y > 600.0 + 2.0 * (std::fabs(a) + std::fabs(b))) {
        Scaled r;
        if (m_asymptotic(a, b, y, r) && !std::isnan(r.m)) return {r, 1.0};  // (NaN: its factors left the range apart)
    }
    return m_series(a, b, y, a_lo, positive_only);
}

// x < 0 far out: M(a, b, x) ~ Gamma(b)/Gamma(b-a) |x|^-a sum (a)_s (1+a-b)_s / s! |x|^-s
// (Kummer's form of the large-argument expansion, the e^x and e^|x| cancelled
// analytically). b - a = hi + lo enters Gamma to first order: rounding it
// would cost psi(b-a) (b-a)/2 ULP (S3's lesson again). False if not converged.
inline bool m_negative_asymptotic(double a, double b, double x, double& out) {
    const double y = -x;
    double s = 1.0, t = 1.0, prev = 1.0;
    for (int k = 0; k < 500; ++k) {
        t *= (a + k) * (1.0 + a - b + k) / ((k + 1.0) * y);
        const double at = std::fabs(t);
        if (at > prev && k > 2) return false;
        prev = at;
        s += t;
        if (t == 0.0 || at <= 0x1p-56 * std::fabs(s)) break;
        if (k == 499) return false;
    }
    const dd::DD bma = dd::two_sum(b, -a);
    // 1/Gamma(b - a) = 0: only the e^x part is left, a polynomial in y times e^-y (not this
    // expansion; it can still be a subnormal: hyp1f1(3, 2, -700) = -3.4e-302)
    if (nonpos_int(bma.hi) && bma.lo == 0.0) return false;
    // 1/Gamma(hi + lo) = (1/Gamma(hi)) (1 - psi(hi) lo); as m 2^e past Gamma's own range on
    // both sides (1/Gamma(1002) is 0 as a double: hyp1f1(-1000, 2, -1e300) was 0 for inf;
    // 1/Gamma(-241.5) is inf: hyp1f1(243, 1.5, -532863) was inf for 0)
    const Scaled rg0 = bma.hi > 170.0 ? Scaled{1.0, 0} / gamma_scaled(bma.hi) : scaled(Rgamma::f(bma.hi));
    const Scaled rg = bma.hi < -170.0 ? gamma_detail::rgamma_dd(bma.hi, bma.lo)
                                      : rg0 * scaled(1.0 - (bma.lo != 0.0 ? Digamma::f(bma.hi) * bma.lo : 0.0));
    const Scaled gb = b > 170.0 ? gamma_scaled(b) : scaled(Gamma::f(b));
    out = value(gb * rg * pow_scaled(y, -a) * scaled(s));
    return !std::isnan(out);
}

inline double hyp1f1(double a, double b, double x);
inline constexpr double kReach = 1e6;  // |a|, |b|, |x| the loose region is computed for
inline constexpr double kMaxCancel = 1e13;  // sum|terms| / |sum| a loose-region series may have

// Tricomi's expansion in Bessel functions for a -> -inf, b and x > 0 fixed (A&S 13.3.7):
//     M(a, b, x) = Gamma(b) e^(x/2) (k x)^((1-b)/2) sum_n A_n r^n J_(b-1+n)(2 sqrt(k x)),
//     k = b/2 - a, r = (x/2) / sqrt(k x), A_0 = 1, A_1 = 0, A_2 = b/2,
//     (n+1) A_(n+1) = (n+b-1) A_(n-1) + (2a-b) A_(n-2).
// A_n r^n falls like |a|^(-n/6): for the huge |a| of the loose region, where both our series
// and Boost overflow (hyp1f1(-1e12, 2, 0.5) was inf; the value is ~1e-9). J by Jv, then upward
// in order (stable: the order stays far below the argument).
inline bool m_tricomi(double a, double b, double x, double& out) {
    const double kappa = 0.5 * b - a;
    if (!(kappa > 0.0) || !(b > 0.0) || b > 170.0 || !(x > 0.0)) return false;
    const double kx = kappa * x, w = 2.0 * std::sqrt(kx), r = 0.5 * x / std::sqrt(kx);
    if (!(w > 4.0 * (b + 60.0))) return false;  // the upward recurrence needs order << argument
    double jm = Jv::f(b - 1.0, w), j = Jv::f(b, w);  // J_(b-1), J_b
    double a2 = 0.0, a1 = 0.0, a0 = 1.0;  // A_(n-2), A_(n-1), A_n at n = 0
    double rn = 1.0, sum = jm, big = std::fabs(jm);
    bool converged = false;
    for (int n = 1; n < 400; ++n) {
        // A_n from A_(n-1), A_(n-2), A_(n-3)
        double an;
        if (n == 1) an = 0.0;
        else if (n == 2) an = 0.5 * b;
        else an = ((n - 1 + b - 1) * a1 + (2.0 * a - b) * a2) / n;  // (n) A_n = (n+b-2) A_(n-2) + (2a-b) A_(n-3)
        a2 = a1;
        a1 = a0;
        a0 = an;
        rn *= r;
        const double t = an * rn * j;
        sum += t;
        big = std::max(big, std::fabs(t));
        if (n > 4 && std::fabs(t) <= 0x1p-56 * std::fabs(sum) && std::fabs(an * rn) <= 0x1p-56 * big) {
            converged = true;
            break;
        }
        const double nu = b - 1.0 + n;  // J_(nu+1) = (2 nu / w) J_nu - J_(nu-1)
        if (!(4.0 * nu < w)) break;  // the order has come up to the argument: not for this expansion
        const double jn = 2.0 * nu / w * j - jm;
        jm = j;
        j = jn;
    }
    if (!converged) return false;  // (x not small against |a|: the terms do not fall)
    out = value(scaled(Gamma::f(b)) * exp_scaled(0.5 * x) * pow_scaled(kx, 0.5 * (1.0 - b)) * scaled(sum));
    return std::isfinite(out);
}

// M(a, b, x) for a = a_hi + a_lo <= -2, b > 0, where the power series cancels (like
// e^(2 sqrt(|a| x)) for x > 0): by the recurrence in a (DLMF 13.3.1), in double-double,
//     (b - a) M(a - 1) = (b - 2a - x) M(a) + a M(a + 1).
// A polynomial (a = -n) starts from M(0) = 1, M(-1) = 1 - x/b. Any other a starts from two
// series values at a + t and a + t - 1 with a + t just below -k0, k0 = (x - b)^2 / (4x) for
// x > b: at -a < k0 the argument is beyond the oscillations of M (x > (sqrt(k + b) + sqrt k)^2)
// and M(a - k) is the vanishing solution downward in a, which the recurrence would lose; from
// k0 on, and for a polynomial from the start, it is not: the product of the two characteristic
// ratios is k / (b + k) < 1 and M follows the larger, then both oscillate. So a step's
// rounding (2^-102) stays that small against the size of the solutions.
// What is lost is how far the result itself is below that size (near one of its zeros):
// beside M runs the solution D started with 0, 1, in float64, as the measure of that size.
// False if fewer than 53 bits are left, or if the starting series lose more than kMaxCancel.
// (Not for b < 0: there the second solution falls up to k = -b/2 and grows after it, and a
// rounding made in between is amplified without D showing it. Boost's float64 recurrence was
// off by 2e-9 at a = -1e5, and by orders of magnitude for b < 0.)
inline constexpr int kStartTerms = 4000;  // terms of each starting series
inline bool m_a_recurrence(double a_hi, double a_lo, double b, double x, Scaled& out) {
    const bool poly = a_lo == 0.0 && a_hi == std::floor(a_hi);
    double top = 0.0;  // the recurrence starts at top (cur at top - 1, prev at top), top - a_hi an integer
    dd::DD prev{1.0, 0.0};
    dd::DD cur = dd::DD{1.0, 0.0} - dd::DD{x, 0.0} / dd::DD{b, 0.0};
    std::int64_t e = 0;
    double start_error = 0.0;  // the relative error of the starting values
    if (!poly) {
        const double below = (x > b ? std::ceil(0.25 * (x - b) * (x - b) / x) : 0.0) + 2.0;
        const double t = std::floor(-a_hi) - below;
        if (!(t >= 2.0)) return false;  // (the series itself is the way there)
        top = a_hi + t;  // exact: a smaller number with the same last bit
        const Series s0 = m_series(top, b, x, a_lo, false, kStartTerms, kMaxCancel, true);
        const Series s1 = m_series(top - 1.0, b, x, a_lo, false, kStartTerms, kMaxCancel, true);
        if (!(s0.cancel <= kMaxCancel) || !(s1.cancel <= kMaxCancel)) return false;
        // both on the scale 2^s1.s.e (they are one step apart: within the range of a double)
        const std::int64_t gap = s0.s.e - s1.s.e;
        if (gap > 900 || gap < -900) return false;
        prev = dd::ldexp(dd::quick_two_sum(s0.s.m, s0.lo), static_cast<int>(gap));
        cur = dd::quick_two_sum(s1.s.m, s1.lo);
        e = s1.s.e;
        start_error = 0x1p-96 * std::max(s0.cancel, s1.cancel);  // (the compensated sums: cancel 2^-100, kept above)
    }
    const dd::DD bmx = dd::two_sum(b, -x);
    const double c0 = b - x;
    double dprev = 0.0, dcur = 1.0;  // D at top, top - 1, on M's scale there
    std::int64_t ed = e;  // M is cur 2^e, D is dcur 2^ed
    const std::int64_t count = static_cast<std::int64_t>(top - a_hi);
    for (std::int64_t k = 1; k < count; ++k) {
        const double ak = top - static_cast<double>(k);  // exact; the parameter is ak + a_lo
        const dd::DD den = dd::two_sum(b, -ak) + (-a_lo);
        const dd::DD next = ((bmx + (-2.0 * ak) + (-2.0 * a_lo)) * cur + prev * dd::DD{ak, a_lo}) / den;
        const double dnext = ((c0 - 2.0 * ak) * dcur + ak * dprev) / den.hi;
        prev = cur, cur = next;
        dprev = dcur, dcur = dnext;
        const double size = std::max(std::fabs(cur.hi), std::fabs(prev.hi));
        if (size > 0x1p600) {
            cur = dd::ldexp(cur, -600), prev = dd::ldexp(prev, -600);
            e += 600;
        } else if (size < 0x1p-600) {
            cur = dd::ldexp(cur, 600), prev = dd::ldexp(prev, 600);
            e -= 600;
        }
        const double dsize = std::max(std::fabs(dcur), std::fabs(dprev));
        if (dsize > 0x1p600) {
            dcur *= 0x1p-600, dprev *= 0x1p-600;
            ed += 600;
        } else if (dsize < 0x1p-600) {
            dcur *= 0x1p600, dprev *= 0x1p600;
            ed -= 600;
        }
    }
    const double dsize = std::max(std::fabs(dcur), std::fabs(dprev));
    if (!std::isfinite(cur.hi) || !(dsize > 0.0) || !std::isfinite(dsize)) return false;
    // count roundings of 2^-102 and the starting values' error, each of the solutions' size,
    // against 2^-54 of the result
    const std::int64_t shift = ed - e;
    if (shift > 2000) return false;
    const double dn = shift < -2000 ? 0.0 : std::ldexp(dsize, static_cast<int>(shift));
    const double error = static_cast<double>(count) * 0x1p-102 + start_error;
    if (!(error * std::max(dn, std::fabs(prev.hi)) <= 0x1p-54 * std::fabs(cur.hi))) return false;
    out = scaled(cur.hi + cur.lo);
    out.e += e;
    return true;
}

// The loose region by our own series, where one of the two forms (the Maclaurin series, or
// Kummer's e^x M(b - a, b, -x)) is short and loses at most kMaxCancel to cancellation: then
// the compensated sum is as accurate as in the guaranteed region. NaN where neither is (the
// caller goes on to Boost and the expansions). Boost alone was wrong by orders of magnitude at
// such points (hyp1f1(-1000, -100.5, 0.999) = 4.3e152 for 34799.14).
inline constexpr int kOwnTerms = 4000;  // terms of each form here: 30 us at most
inline double m_loose_series(double a, double b, double x) {
    const dd::DD bma = dd::two_sum(b, -a);
    auto accept = [](const Series& r, Scaled v, double& out) {
        if (!(r.cancel <= kMaxCancel)) return false;
        out = value(v);
        return std::isfinite(out);
    };
    double out;
    if (x > 0.0) {
        const Series direct = m_series(a, b, x, 0.0, false, kOwnTerms, kMaxCancel);
        if (accept(direct, direct.s, out)) return out;
        const Series k = m_series(bma.hi, b, -x, bma.lo, false, kOwnTerms, kMaxCancel);
        return accept(k, exp_scaled(x) * k.s, out) ? out : kNaN;
    }
    const double y = -x;
    // far out: the large-argument expansion (the e^x of Kummer's form cancelled analytically)
    if (y > 600.0 + 2.0 * (std::fabs(a) + std::fabs(b)) && (b > 0.0 || b > -170.0) &&
        m_negative_asymptotic(a, b, x, out) && std::isfinite(out))
        return out;
    const Series k = m_series(bma.hi, b, y, bma.lo, false, kOwnTerms, kMaxCancel);
    if (accept(k, exp_scaled(x) * k.s, out)) return out;
    const Series direct = m_series(a, b, x, 0.0, false, kOwnTerms, kMaxCancel);
    return accept(direct, direct.s, out) ? out : kNaN;
}

inline constexpr double kBoostX = 16.0;  // x below which Boost is used in the guaranteed region
inline constexpr double kBoostA = 100.0;  // a up to which Boost is used in the guaranteed region
inline constexpr double kBoostLoose = 100.0;  // |a|, |b| up to which Boost is used in the loose region

// The loose region with |a|, |b|, |x| <= kReach. Every value comes from a form whose loss is
// known: a series that kept kMaxCancel, the recurrence in a (b > 0), the large-argument
// expansions; then Boost where its error was measured small (|a|, |b| <= kBoostLoose:
// 1.2e-12 at most; beyond it Boost returned values wrong by orders of magnitude, mostly for
// b < 0), and Tricomi's expansion. Anything else is NaN.
inline double m_loose(double a, double b, double x) {
    const double own = m_loose_series(a, b, x);
    if (!std::isnan(own)) return own;
    const dd::DD bma = dd::two_sum(b, -a);
    if (b > 0.0) {
        Scaled r;
        if (x > 0.0 && a <= -2.0 && m_a_recurrence(a, 0.0, b, x, r)) return value(r);
        if (x < 0.0 && bma.hi <= -2.0 && m_a_recurrence(bma.hi, bma.lo, b, -x, r)) {  // Kummer's form
            const double v = value(exp_scaled(x) * r);
            if (std::isfinite(v)) return v;
        }
    }
    if (std::max(std::fabs(a), std::fabs(b)) <= kBoostLoose) {
        const double r = bst::hyp1f1(a, b, x);
        if (std::isfinite(r)) return r;
    }
    double t;
    if (a < -1e3 && m_tricomi(a, b, x, t)) return t;
    // the series at full length (a sum that overflowed is no value: its sign is the last
    // term's, not the value's)
    auto kept = [](const Series& r, Scaled v) {
        const double out = value(v);
        return r.cancel <= kMaxCancel && std::isfinite(out) ? out : kNaN;
    };
    if (x > 0.0) {
        const Series direct = m_nonneg(a, b, x);
        if (direct.cancel <= kMaxCancel) return kept(direct, direct.s);
        const Series k = m_series(bma.hi, b, -x, bma.lo);
        return kept(k, exp_scaled(x) * k.s);
    }
    if (x < -(600.0 + 2.0 * (std::fabs(a) + std::fabs(b))) && std::fabs(b) < 170.0 && m_negative_asymptotic(a, b, x, t))
        return t;
    const Series kummer = m_nonneg(bma.hi, b, -x, bma.lo);
    if (kummer.cancel <= kMaxCancel) return kept(kummer, exp_scaled(x) * kummer.s);
    const Series direct = m_series(a, b, x);
    return kept(direct, direct.s);
}

// The loose region beyond the reach: Tricomi's expansion for a large negative a, a short
// polynomial, the large-argument expansion for x < 0; else NaN.
inline double m_beyond(double a, double b, double x) {
    double t;
    if (a < -1e3 && m_tricomi(a, b, x, t)) return t;
    if (!(nonpos_int(a) && a >= -1e3)) return kNaN;
    auto kept = [](const Series& r, Scaled v) {
        const double out = value(v);
        return r.cancel <= kMaxCancel && std::isfinite(out) ? out : kNaN;
    };
    const dd::DD bma = dd::two_sum(b, -a);
    if (x > 0.0) {
        const Series direct = m_nonneg(a, b, x);
        if (direct.cancel <= kMaxCancel) return kept(direct, direct.s);
        const Series k = m_series(bma.hi, b, -x, bma.lo);
        return kept(k, exp_scaled(x) * k.s);
    }
    if (x < -(600.0 + 2.0 * (std::fabs(a) + std::fabs(b))) && std::fabs(b) < 170.0 && m_negative_asymptotic(a, b, x, t))
        return t;
    const Series kummer = m_nonneg(bma.hi, b, -x, bma.lo);
    if (kummer.cancel <= kMaxCancel) return kept(kummer, exp_scaled(x) * kummer.s);
    const Series direct = m_series(a, b, x);
    return kept(direct, direct.s);
}

inline double hyp1f1(double a, double b, double x) {
    if (std::isnan(a) || std::isnan(b) || std::isnan(x)) return kNaN;
    if (a == 0.0 || x == 0.0) return 1.0;  // hyp1f1(0, 0, x) = 1 too (S7 user decision; scipy inf)
    if (nonpos_int(b) && !(nonpos_int(a) && a >= b)) return kInf;  // a pole (scipy inf)
    if (std::isinf(a) || std::isinf(b)) return kNaN;
    if (std::isinf(x)) {
        // x -> inf: Gamma(b)/Gamma(a) e^x x^(a-b) (a polynomial: its leading term);
        // x -> -inf: Gamma(b)/Gamma(b-a) |x|^-a. The sign of the coefficient, or 0.
        if (nonpos_int(a)) {
            // degree n = -a: leading coefficient (a)_n / ((b)_n n!), times x^n. All n factors
            // a + k are negative; b + k is negative for the first neg_count(b, n).
            const double n = -a;
            const bool neg = is_odd(n + neg_count(b, n)) != (x < 0.0 && is_odd(n));
            return neg ? -kInf : kInf;
        }
        if (x > 0.0) return (gamma_detail::gamma_sign(a) * gamma_detail::gamma_sign(b) > 0.0) ? kInf : -kInf;
        if (a > 0.0) return 0.0;
        if (nonpos_int(b - a)) return 0.0;  // 1/Gamma(b - a) = 0: the e^x term (0)
        return (gamma_detail::gamma_sign(b) * gamma_detail::gamma_sign(b - a) > 0.0) ? kInf : -kInf;
    }
    // Boost (special/boost.hpp) is called only for |a|, |b|, |x| <= kReach (its time grows with
    // them: 50 ms at 1e7, no return at 1e12) and |a| >= 2^-512 (a = 1e-300 with b < 0
    // overflowed its stack).
    const bool guaranteed = a >= 0.0 && b > 0.0 && (x >= 0.0 || b >= a);
    const bool within = std::max({std::fabs(a), std::fabs(b), std::fabs(x)}) <= kReach;
    if (std::fabs(a) < 0x1p-512 && !guaranteed) {
        // M = 1 + a S(b, x) to O(a^2): S from the value at a' = +-2^-26 (1e-8 relative).
        // The term is kept while it is below 1e-9, so that S's error is nothing in M; a
        // larger one is no value. (Boost overflowed its stack at a = 1e-300 with b < 0.)
        const double ap = std::copysign(0x1p-26, a);
        const double term = (hyp1f1(ap, b, x) - 1.0) * (a / ap);
        return std::fabs(term) <= 1e-9 ? 1.0 + term : kNaN;
    }
    if (!guaranteed) return within ? m_loose(a, b, x) : m_beyond(a, b, x);
    // The guaranteed region. Boost for 0 <= x < 16 with b < 11 (twice as fast there, inside
    // the contract where it was measured: a <= 100; at a = 1e5 it was 83 ULP off, and from
    // x = 18 on it exceeds 45 ULP on 4% of the points, up to 297 ULP near x = 500); a
    // non-finite or subnormal result is ours.
    if (x >= 0.0 && x < kBoostX && b < 11.0 && a <= kBoostA && within) {
        const double r = bst::hyp1f1(a, b, x);
        if (bst::normal(r)) return r;
    }
    if (x > 0.0) return value(m_nonneg(a, b, x, 0.0, true).s);
    // x < 0: Kummer, with b - a carried exactly
    if (x < -(600.0 + 2.0 * (std::fabs(a) + std::fabs(b))) && std::fabs(b) < 170.0) {
        double r;
        if (m_negative_asymptotic(a, b, x, r)) return r;
    }
    const dd::DD bma = dd::two_sum(b, -a);
    return value(exp_scaled(x) * m_nonneg(bma.hi, b, -x, bma.lo).s);
}

// ---- 2F1 ------------------------------------------------------------------------

// The series with positive terms only (a, b, c, z > 0: every sum of the guaranteed region), in
// plain float64 range; the sum, inf or NaN if the terms leave the range (the caller then takes
// the scaled series below).
struct PlainSum {
    double s, cancel;
};
inline PlainSum f_series_positive(double a, double b, double c, double z) {
    {  // the float64 pass (as m_series)
        dd::Sum fsum(1.0);
        double ft = 1.0, fabs_sum = 1.0, weighted = 1.0;
        bool done = false;
        for (int k = 0; k < 2000; ++k) {
            if (a + k == 0.0 || b + k == 0.0) {
                done = true;
                break;
            }
            ft *= (a + k) * (b + k) * z / ((c + k) * (k + 1.0));
            fsum.add(ft);
            const double at = std::fabs(ft);
            fabs_sum += at;
            weighted += (k + 2.0) * at;
            if (!(at < 0x1p600)) break;
            if (ft == 0.0 || (k > 2 && k > -c && at <= 0x1p-56 * std::fabs(fsum.hi))) {
                // (not while the terms still rise: a tiny a b / c makes the first ones negligible
                // before a large b lifts them, hyp2f1(1e-25, 1e6, 1, 0.5))
                if (ft != 0.0 && (a + k) * (b + k) * z / ((c + k) * (k + 1.0)) >= 1.0) continue;
                done = true;
                break;
            }
        }
        const double fs = fsum.total();
        if (done && weighted <= kFastSeries * std::fabs(fs)) return {fs, fs == 0.0 ? kInf : fabs_sum / std::fabs(fs)};
    }
    // The long series: each term from the last by one division, the roundings of the
    // products and of the quotient recovered by fma and carried to first order (the
    // term and the sum as hi + lo). What is left is second order, k eps^2: as accurate
    // as the double-double loop this replaces (2.7 ULP at most over z in [0.5, 0.9],
    // both, measured) at a third of its time.
    double th = 1.0, tl = 0.0, abssum = 1.0;
    dd::Sum sum(1.0);
    bool ended = false;
    for (int k = 0; k < kMaxSeriesTerms; ++k) {
        if (a + k == 0.0 || b + k == 0.0) {  // a polynomial ends
            ended = true;
            break;
        }
        const dd::DD A = dd::two_sum(a, k), B = dd::two_sum(b, k), C = dd::two_sum(c, k);
        const double kp = k + 1.0;
        const double n1 = A.hi * B.hi, e1 = std::fma(A.hi, B.hi, -n1) + A.hi * B.lo + A.lo * B.hi;
        const double n2 = n1 * z, e2 = std::fma(n1, z, -n2) + e1 * z;
        const double d1 = C.hi * kp, e3 = std::fma(C.hi, kp, -d1) + C.lo * kp;
        const double rd = 1.0 / d1, q = n2 * rd;
        const double ql = (std::fma(-q, d1, n2) + e2 - q * e3) * rd;  // the ratio is q + ql
        const double tn = th * q;
        tl = std::fma(th, q, -tn) + tl * q + th * ql;
        th = tn;
        sum.add(th, tl);
        abssum += std::fabs(th);
        if (th == 0.0 ||
            (k > 2 && k > -c && std::fabs(th) <= 0x1p-56 * std::fabs(sum.hi) && q < 1.0)) {  // past c + k = 0
            ended = true;
            break;
        }
        if (!std::isfinite(th)) {  // the terms left the range: the sum is inf or NaN as it stands
            ended = true;
            break;
        }
    }
    if (!ended) return {kNaN, kInf};
    const double v = sum.total();
    return {v, v == 0.0 ? kInf : abssum / std::fabs(v)};
}
// sum (a)_k (b)_k / ((c)_k k!) z^k, |z| < 1 (or a polynomial), in
// double-double (the terms' products would carry ~k ULP: 24 ULP at z = 0.885,
// measured), as m 2^e (the terms reach (1 - z)^(c - a - b)), with the cancellation
// sum|terms| / |sum|. Each argument may carry a low part (Pfaff's c - b and z / (z - 1), the
// c - a, c - b, 1 +- (c - a - b) of the 1 - z transformation: none need be a double).
struct FSum {
    Scaled s;
    double cancel;
};
struct Lows {
    double a = 0.0, b = 0.0, c = 0.0, z = 0.0;
};
inline constexpr FSum kNoSum{{kNaN, 0}, kInf};
// positive_only: the caller wants the sum itself and every term is positive: once the partial
// sum is past every double the sum is inf, without the remaining terms.
inline FSum f_series2(double a, double b, double c, double z, const Lows& low = {}, bool positive_only = false) {
    if (low.a == 0.0 && low.b == 0.0 && low.c == 0.0 && low.z == 0.0) {  // the float64 pass (as m_series)
        dd::Sum fsum(1.0);
        double ft = 1.0, fabs_sum = 1.0, weighted = 1.0;
        bool done = false;
        for (int k = 0; k < 2000; ++k) {
            if (a + k == 0.0 || b + k == 0.0) {
                done = true;
                break;
            }
            const double ratio = (a + k) * (b + k) * z / ((c + k) * (k + 1.0));
            ft *= ratio;
            fsum.add(ft);
            const double at = std::fabs(ft);
            fabs_sum += at;
            weighted += (k + 2.0) * at;
            if (!(at < 0x1p600)) break;
            if (at < 0x1p-700 && k < -c) break;  // (the long series keeps such terms scaled)
            if (ft == 0.0 || (k > 2 && k > -c && std::fabs(ratio) < 1.0 && at <= 0x1p-56 * std::fabs(fsum.hi))) {
                done = true;
                break;
            }
        }
        const double fs = fsum.total();
        if (done && weighted <= kFastSeries * std::fabs(fs))
            return {scaled(fs), fs == 0.0 ? kInf : fabs_sum / std::fabs(fs)};
    }
    // The long series: each term from the last by one division, the roundings of the
    // products and of the quotient recovered by fma and carried to first order (the
    // term and the sum as hi + lo). What is left is second order, k eps^2: as accurate
    // as the double-double loop this replaces (2.7 ULP at most over z in [0.5, 0.9],
    // both, measured) at a third of its time.
    // The sum is sum 2^e and the term (th + tl) 2^(e - lag), as in m_series: lag > 0 only for
    // c < 0, while the terms fall below the normal range before c + k = 0.
    double th = 1.0, tl = 0.0, abssum = 1.0;
    dd::Sum sum(1.0);
    std::int64_t e = 0;
    int lag = 0;
    bool ended = false;
    for (int k = 0; k < kMaxSeriesTerms; ++k) {
        if ((a + k == 0.0 && low.a == 0.0) || (b + k == 0.0 && low.b == 0.0)) {  // a polynomial ends
            ended = true;
            break;
        }
        const dd::DD A = dd::two_sum(a, k), B = dd::two_sum(b, k), C = dd::two_sum(c, k);
        const double kp = k + 1.0;
        const double n1 = A.hi * B.hi, e1 = std::fma(A.hi, B.hi, -n1) + A.hi * (B.lo + low.b) + (A.lo + low.a) * B.hi;
        const double n2 = n1 * z, e2 = std::fma(n1, z, -n2) + e1 * z + n1 * low.z;
        const double d1 = C.hi * kp, e3 = std::fma(C.hi, kp, -d1) + (C.lo + low.c) * kp;
        // (a subnormal c: 1/d1 would overflow where the quotient does not)
        const bool tiny = std::fabs(d1) < 0x1p-1000;
        const double rd = 1.0 / d1, q = tiny ? n2 / d1 : n2 * rd;
        const double rem = std::fma(-q, d1, n2) + e2 - q * e3;
        const double ql = tiny ? rem / d1 : rem * rd;  // the ratio is q + ql
        const double tn = th * q;
        tl = std::fma(th, q, -tn) + tl * q + th * ql;
        th = tn;
        double at = std::fabs(th);
        if (lag == 0 && !(at < 0x1p-700 && k < -c)) [[likely]] {
            sum.add(th, tl);
            abssum += at;
        } else {
            if (at < 0x1p-700 && k < -c) {
                th *= 0x1p600, tl *= 0x1p600;
                lag += 600;
            } else if (at > 0x1p100) {
                th *= 0x1p-600, tl *= 0x1p-600;
                lag -= 600;
            }
            // (a term 2^-1200 below the sum's scale adds nothing: left out)
            at = lag > 1200 ? 0.0 : std::ldexp(std::fabs(th), -lag);
            if (lag <= 1200) sum.add(std::ldexp(th, -lag), std::ldexp(tl, -lag));
            abssum += at;
        }
        // past c + k = 0, and not while the terms still grow (right after c + k = 0 they are
        // far below the sum and rising); against hi + lo, the compensated sum
        if (th == 0.0 || (k > 2 && k > -c && std::fabs(q) < 1.0 && at <= 0x1p-56 * std::fabs(sum.hi + sum.lo))) {
            ended = true;
            break;
        }
        if (std::fabs(sum.hi) > 0x1p600 || std::fabs(th) > 0x1p600) {  // (th: only with lag = 0)
            sum.scale(-600);
            th *= 0x1p-600, tl *= 0x1p-600;
            abssum *= 0x1p-600;
            e += 600;
            if (positive_only && e > 1800) return {{kInf, 0}, 1.0};
        }
    }
    if (!ended || !std::isfinite(sum.hi)) return kNoSum;
    const double v = sum.total();
    Scaled r = scaled(v);
    r.e += e;
    return {r, v == 0.0 ? zero_sum_cancel(abssum) : abssum / std::fabs(v)};
}
// (the compensated sum carries ~cancel 2^-105 a term: 1e16 leaves 1e-13 after 400 terms)
inline constexpr double kMaxCancel2 = 1e16;
// The sum, or NaN where it lost more than kMaxCancel2 to cancellation (the loose region:
// hyp2f1(10, -100.5, 1e-10, 0.9) was 7e16 for 2e-13).
inline Scaled f_series_scaled(double a, double b, double c, double z, const Lows& low = {}) {
    const FSum r = f_series2(a, b, c, z, low);
    return r.cancel <= kMaxCancel2 ? r.s : Scaled{kNaN, 0};
}
inline double f_series(double a, double b, double c, double z) {
    if (a > 0.0 && b > 0.0 && c > 0.0 && z > 0.0) {
        const double v = f_series_positive(a, b, c, z).s;
        if (std::isfinite(v)) return v;
        return value(f_series2(a, b, c, z, {}, true).s);  // beyond the range: inf
    }
    return value(f_series_scaled(a, b, c, z));
}

inline constexpr int kMaxTailTerms = 100000;  // the limit forms' sums: NaN if not ended by then

inline double gam(double v) { return Gamma::f(v); }
inline double rgam(double v) { return Rgamma::f(v); }
inline double psi(double v) { return Digamma::f(v); }

// x + y as m 2^e, and how much of the two was lost: (|x| + |y|) / |x + y|.
inline Scaled add_scaled(Scaled x, Scaled y, double& cancel) {
    cancel = 1.0;
    if (x.m == 0.0) return y;
    if (y.m == 0.0) return x;
    if (!std::isfinite(x.m) || !std::isfinite(y.m)) return {kNaN, 0};
    const std::int64_t e = std::max(x.e, y.e);
    const double xv = x.e - e < -2000 ? 0.0 : std::ldexp(x.m, static_cast<int>(x.e - e));
    const double yv = y.e - e < -2000 ? 0.0 : std::ldexp(y.m, static_cast<int>(y.e - e));
    const double sum = xv + yv;
    cancel = sum == 0.0 ? kInf : (std::fabs(xv) + std::fabs(yv)) / std::fabs(sum);
    Scaled r = scaled(sum);
    r.e += e;
    return r;
}

// The two terms of the transformation to y = 1 - z may cancel this much (each carries a
// few tens of ULP from its Gammas: 7e-12 of the sum is left at worst); more is no value.
inline constexpr double kMaxTransformCancel = 1024.0;

// 2F1(a, b; c; 1 - y), c - a - b not an integer, by AS 15.3.6:
//     Gamma(c) Gamma(s) / (Gamma(c-a) Gamma(c-b)) F(a, b; 1 - s; y)
//     + y^s Gamma(c) Gamma(-s) / (Gamma(a) Gamma(b)) F(c-a, c-b; 1 + s; y),   s = c - a - b.
// b and y as hi + lo (Pfaff's c - b and 1 / (1 - z)); c - a, c - b and s are formed exactly
// and every Gamma is taken at its hi + lo argument as m 2^e: a Gamma beyond 171 does not
// overflow, one next to a pole keeps its relative accuracy, and (1 - z)^-a's underflow in the
// caller is met by this sum's size. NaN if the two terms cancel more than kMaxTransformCancel
// (s next to an integer).
inline Scaled f_one_minus_general(double a, dd::DD b, double c, dd::DD y) {
    using gamma_detail::rgamma_dd;
    const dd::DD ca = dd::two_sum(c, -a);
    const dd::DD cb = dd::two_sum(c, -b.hi) + (-b.lo);
    const dd::DD s = ca - b;
    const double ly = std::log(y.hi) + y.lo / y.hi;
    const Scaled rc = rgamma_dd(c, 0.0);
    // 1/Gamma is 0 at a pole (the term is absent); a 0 or inf anywhere else is a factor that
    // left the range (a Gamma beyond 2^(2^58)): no value
    auto pole = [](dd::DD v) { return nonpos_int(v.hi) && v.lo == 0.0; };
    auto usable = [](Scaled v) { return std::isfinite(v.m) && v.m != 0.0; };
    if (!usable(rc)) return {kNaN, 0};
    Scaled t1{0.0, 0}, t2{0.0, 0};
    if (!pole(ca) && !pole(cb)) {
        const dd::DD c1 = dd::DD{1.0, 0.0} - s;
        const Scaled g = rgamma_dd(ca.hi, ca.lo) * rgamma_dd(cb.hi, cb.lo), gs = rgamma_dd(s.hi, s.lo);
        if (!usable(g) || !usable(gs)) return {kNaN, 0};
        t1 = g / (rc * gs) * f_series_scaled(a, b.hi, c1.hi, y.hi, {0.0, b.lo, c1.lo, y.lo});
    }
    if (!pole({a, 0.0}) && !pole(b)) {
        const dd::DD c2 = s + 1.0;
        const Scaled g = rgamma_dd(a, 0.0) * rgamma_dd(b.hi, b.lo), gs = rgamma_dd(-s.hi, -s.lo);
        if (!usable(g) || !usable(gs)) return {kNaN, 0};
        const Scaled ys = pow_scaled(y.hi, s.hi) * scaled(1.0 + s.lo * ly + s.hi * (y.lo / y.hi));
        t2 = g / (rc * gs) * ys * f_series_scaled(ca.hi, cb.hi, c2.hi, y.hi, {ca.lo, cb.lo, c2.lo, y.lo});
    }
    if (std::isnan(t1.m) || std::isnan(t2.m)) return {kNaN, 0};
    double cancel;
    const Scaled r = add_scaled(t1, t2, cancel);
    return cancel <= kMaxTransformCancel ? r : Scaled{kNaN, 0};
}

// The limit forms' sums are plain float64 sums: one that lost more than this to cancellation
// (terms of both signs: a negative a or b) is no value.
inline constexpr double kMaxPlainCancel = 64.0;

// 2F1(a, b; c; 1 - y) for 0 < y <= 1/2 and c - a - b an integer m (up to the rounding of
// a + b): the limit forms with digamma (AS 15.3.10-12). NaN where a Gamma leaves the range.
inline double f_one_minus_integer(double a, double b, double c, double y) {
    const double s = c - a - b;
    const double ly = std::log(y);
    // s is m up to the rounding of a + b; the degenerate forms sum m terms, so a huge m
    // (only with |a + b| huge, the loose region) takes the series instead
    if (std::fabs(s) > 1e5) return f_series(a, b, c, 1.0 - y);
    const int m = static_cast<int>(std::fabs(std::nearbyint(s)));
    // A product whose factors left the range (a Gamma beyond 171 against a 1/Gamma that
    // underflowed) is inf, 0 or NaN whatever the value is: none of them is returned.
    auto kept = [](double v) { return std::isfinite(v) && v != 0.0 ? v : kNaN; };
    auto tail = [&](double a0, double b0, double pa, double pb) {
        // sum_n (a0)_n (b0)_n / (n! (n+m)!) y^n [ly - psi(n+1) - psi(n+m+1) + psi(pa+n) + psi(pb+n)]
        double sum = 0.0, t = 1.0, abs_sum = 0.0;
        for (int i = 1; i <= m; ++i) t /= i;  // 1/m!
        // psi(n + 1) = H_n - euler and psi(n + m + 1): the harmonic numbers as running sums
        double hn = 0.0, hm = 0.0;
        for (int i = 1; i <= m; ++i) hm += 1.0 / i;
        for (int n = 0; n < kMaxTailTerms; ++n) {
            const double term = t * (ly - (hn - kEuler) - (hm - kEuler) + psi(pa + n) + psi(pb + n));
            sum += term;
            abs_sum += std::fabs(term);
            if (n > 2 && std::fabs(term) <= 0x1p-56 * std::fabs(sum))
                return abs_sum <= kMaxPlainCancel * std::fabs(sum) ? sum : kNaN;
            t *= (a0 + n) * (b0 + n) / ((n + 1.0) * (n + m + 1.0)) * y;
            if (t == 0.0) return abs_sum <= kMaxPlainCancel * std::fabs(sum) ? sum : kNaN;
            hn += 1.0 / (n + 1.0);
            hm += 1.0 / (n + m + 1.0);
        }
        return kNaN;  // not ended (huge a0 b0 y): no value
    };
    if (m == 0) {
        // AS 15.3.10
        double sum = 0.0, t = 1.0, hn = 0.0, abs_sum = 0.0;  // hn = H_n: psi(n + 1) = H_n - euler
        bool ended = false;
        for (int n = 0; n < kMaxTailTerms; ++n) {
            const double term = t * (2.0 * (hn - kEuler) - psi(a + n) - psi(b + n) - ly);
            sum += term;
            abs_sum += std::fabs(term);
            t *= (a + n) * (b + n) / ((n + 1.0) * (n + 1.0)) * y;
            if ((n > 2 && std::fabs(term) <= 0x1p-56 * std::fabs(sum)) || t == 0.0) {
                ended = true;
                break;
            }
            hn += 1.0 / (n + 1.0);
        }
        if (!ended || !(abs_sum <= kMaxPlainCancel * std::fabs(sum)))
            return kNaN;  // huge a b y, or cancelled: no value
        return kept(gam(a + b) * rgam(a) * rgam(b) * sum);
    }
    // finite part: sum_{n<m} (p)_n (q)_n / (n! (1-m)_n) y^n
    auto finite = [&](double p, double q) {
        double sum = 0.0, t = 1.0, abs_sum = 0.0;
        for (int n = 0; n < m; ++n) {
            sum += t;
            abs_sum += std::fabs(t);
            t *= (p + n) * (q + n) / ((n + 1.0) * (1.0 - m + n)) * y;
        }
        return abs_sum <= kMaxPlainCancel * std::fabs(sum) ? sum : kNaN;
    };
    const double gm = m <= 23 ? kFactorial[m - 1] : gam(m);  // Gamma(m) = (m-1)!; kFactorial[k] = k!
    if (s > 0.0) {
        // AS 15.3.11, c = a + b + m
        const double f = gm * gam(c) * rgam(a + m) * rgam(b + m) * finite(a, b);
        const double r = rgam(a) * rgam(b);
        // (r = 0 at a pole of Gamma(a) or Gamma(b): the second part is absent; an underflowed r is no value)
        if (r == 0.0) return nonpos_int(a) || nonpos_int(b) ? kept(f) : kNaN;
        return kept(f - std::pow(-y, m) * gam(c) * r * tail(a + m, b + m, a + m, b + m));
    }
    // AS 15.3.12, c = a + b - m
    const double f = gm * gam(c) * rgam(a) * rgam(b) * std::pow(y, -m) * finite(a - m, b - m);
    const double r = rgam(a - m) * rgam(b - m);
    if (r == 0.0) return nonpos_int(a - m) || nonpos_int(b - m) ? kept(f) : kNaN;
    return kept(f - ((m % 2) ? -1.0 : 1.0) * gam(c) * r * tail(a, b, a, b));
}

// 2F1(a, b; c; 1 - y) for 0 < y < 0.1, as m 2^e; b = b.hi + b.lo, y = y.hi + y.lo; z = 1 - y
// as a double (for the series). The transformation to y: its limit forms when s = c - a - b is
// an integer or within 2^-48 of one (they take the integer: an error of that size), the
// general form otherwise, which gives NaN when s is so near an integer that its two terms
// cancel more than kMaxTransformCancel. There the series in z itself serves while y >= 1e-3
// (~4e4 terms at most); nearer 1 there is no value.
inline Scaled f_near_one(double a, dd::DD b, double c, dd::DD y, double z) {
    const dd::DD s = dd::two_sum(c, -a) - b;
    const double m = std::nearbyint(s.hi);
    const double off = std::fabs((s.hi - m) + s.lo);
    if (off <= 0x1p-48) {
        // (the integer forms sum |m| terms: a huge m, only with |a + b| huge, takes the series)
        if (std::fabs(m) > 1e5) return f_series_scaled(a, b.hi, c, z, {0.0, b.lo, 0.0, 0.0});
        return scaled(f_one_minus_integer(a, b.hi, a + b.hi + m, y.hi));
    }
    const Scaled r = f_one_minus_general(a, b, c, y);
    if (!std::isnan(r.m)) return r;
    if (y.hi >= 1e-3) return f_series_scaled(a, b.hi, c, z, {0.0, b.lo, 0.0, 0.0});
    return {kNaN, 0};
}

// 2F1 for 0 <= z < 1 (non-polynomial).
inline double f_unit(double a, double b, double c, double z) {
    if (z <= 0.9) return f_series(a, b, c, z);
    // c far beyond a and b: the terms fall like (a b / c)(k / c)^k from the start: the series directly
    if (c > 1e3 * std::max({1.0, std::fabs(a), std::fabs(b)})) return f_series(a, b, c, z);
    return value(f_near_one(a, {b, 0.0}, c, {1.0 - z, 0.0}, z));  // 1 - z exact (Sterbenz)
}

inline double hyp2f1(double a, double b, double c, double z) {
    if (std::isnan(a) || std::isnan(b) || std::isnan(c) || std::isnan(z)) return kNaN;
    if (a == 0.0 || b == 0.0 || z == 0.0) return 1.0;
    if (std::isinf(a) || std::isinf(b) || std::isinf(c)) return kNaN;
    // a polynomial: a or b a non-positive integer (the one nearer 0 ends it)
    const bool pa = nonpos_int(a), pb = nonpos_int(b);
    const double n = pa && pb ? std::max(a, b) : (pa ? a : (pb ? b : 1.0));  // -degree
    const bool poly = pa || pb;
    if (nonpos_int(c) && !(poly && n >= c)) return kInf;  // a pole (scipy inf)
    if (std::isinf(z)) {
        if (poly) {
            // the leading term (p)_N (q)_N / ((c)_N N!) z^N, N = -n: its sign times inf
            const double p = pa && n == a ? a : b, q = (p == a) ? b : a, N = -n;
            // all N factors p + k are negative; q + k and c + k for the first neg_count of each
            const bool neg = is_odd(N + neg_count(q, N) + neg_count(c, N)) != (z < 0.0 && is_odd(N));
            return neg ? -kInf : kInf;
        }
        if (z > 0.0) return kNaN;
        // z -> -inf (S7 user decision: the limit)
        const double lo = std::min(a, b), hi = std::max(a, b);
        if (lo > 0.0) return 0.0;
        if (lo == hi) return kNaN;
        // dominant (-z)^-lo with Gamma(c) Gamma(hi - lo) / (Gamma(hi) Gamma(c - lo))
        const double sg = gamma_detail::gamma_sign(c) * gamma_detail::gamma_sign(hi - lo) *
                          gamma_detail::gamma_sign(hi) * gamma_detail::gamma_sign(c - lo);
        if (nonpos_int(hi) || nonpos_int(c - lo)) return 0.0;
        return sg > 0.0 ? kInf : -kInf;
    }
    if (poly) {
        if (z == 1.0 || z > 1.0 || z < -1.0) {
            // finite sum, any z
            const double p = pa && n == a ? a : b, q = (p == a) ? b : a;
            return f_series(p, q, c, z);  // ends after -p terms
        }
    }
    if (z == 1.0) {
        const double s = c - a - b;
        if (s > 0.0) return gam(c) * gam(s) * rgam(c - a) * rgam(c - b);  // Gauss
        return kNaN;  // diverges (docs/api/special.md §11)
    }
    if (z > 1.0) return kNaN;
    if (poly) {
        const double r = f_series(a, b, c, z);
        if (!std::isnan(r) || !(z > 0.0)) return r;
        // The sum lost more than kMaxCancel2 for 0 < z < 1 (the alternating (p)_k: from degree
        // 60-200, hyp2f1(-200, 1, 2, 0.3) was NaN). Pfaff, 2F1(p, q; c; z) = (1 - z)^-p
        // 2F1(p, c - q; c; w), w = z / (z - 1) < 0, is the same polynomial in w, and for
        // c > 0, c - q > 0 every term of it is positive (Jacobi and Gegenbauer polynomials
        // are this form). Else NaN as before. (A transformation, not a new region's branch.)
        const double p = pa && n == a ? a : b, q = (p == a) ? b : a;
        const dd::DD cq = dd::two_sum(c, -q);  // c - q, exact
        if (!(c > 0.0 && cq.hi > 0.0)) return r;
        const dd::DD omz = dd::two_sum(1.0, -z);  // 1 - z, exact
        const dd::DD wq = dd::DD{-z, 0.0} / omz;
        const double tilt = -p * omz.lo / omz.hi;  // (1 - z)^-p = hi^-p (1 + lo/hi)^-p
        const double low_power = std::fabs(tilt) < 0x1p-27 ? 1.0 + tilt : std::exp(-p * std::log1p(omz.lo / omz.hi));
        return value(
            pow_scaled(omz.hi, -p) * scaled(low_power) * f_series_scaled(p, cq.hi, c, wq.hi, {0.0, cq.lo, 0.0, wq.lo})
        );
    }
    if (z > 0.0) return f_unit(a, b, c, z);
    // z < 0: Pfaff with the larger of a, b first: (1-z)^-a 2F1(a, c-b; c; w)
    const double A = std::max(a, b), B = std::min(a, b);
    const double w = z / (z - 1.0);
    const dd::DD omz = dd::two_sum(1.0, -z);  // 1 - z, exact
    if (w <= 0.9 && B > 0.0 && c >= B && z >= -1.0 && A <= 32.0) {
        // The guaranteed region up to a = 32: the plain positive series and the power in float64
        // (the rounding of c - b and w costs a few ULP at most there, measured). A factor
        // outside the normal range goes on to the general way below.
        const double power = std::pow(omz.hi, -A) * (1.0 - A * omz.lo / omz.hi);
        const double sum = f_series_positive(A, c - B, c, w).s;
        if (bst::normal(power) && bst::normal(sum)) return power * sum;
    }
    const dd::DD cb = dd::two_sum(c, -B);  // c - b, exact
    // (1 - z)^-a = hi^-a (1 + lo/hi)^-a: to first order while a lo/hi is small, else by exp
    // (a = 1e100 with z = -1e-100: the first order gave 0 for e^-1)
    const double tilt = A * omz.lo / omz.hi;
    const double low_power = std::fabs(tilt) < 0x1p-27 ? 1.0 - tilt : std::exp(-A * std::log1p(omz.lo / omz.hi));
    if (cb.hi == 0.0) return value(pow_scaled(omz.hi, -A) * scaled(low_power));  // 2F1(a, 0; c; w) = 1
    const double pw = std::pow(omz.hi, -A) * low_power;
    // (1 - z)^-a can leave the range while the product does not (a large a: the sum is its
    // inverse's size): then both as m 2^e
    auto times_power = [&](Scaled f) {
        const double fv = value(f);
        if (bst::normal(pw) && bst::normal(fv)) return pw * fv;
        return value(pow_scaled(omz.hi, -A) * scaled(low_power) * f);
    };
    if (w <= 0.9) {
        // c - b and w as hi + lo: the sum's relative sensitivity to each is up to
        // a (c - b) w / (c (1 - w)) <= a in the guaranteed region (hyp2f1(27119, 0.0156, 404.7,
        // -0.9957) was 1e-12 off with them rounded).
        const dd::DD wq = dd::DD{-z, 0.0} / omz;
        return times_power(f_series_scaled(A, cb.hi, c, wq.hi, {0.0, cb.lo, 0.0, wq.lo}));
    }
    if (nonpos_int(cb.hi) && cb.lo == 0.0) return times_power(f_series_scaled(A, cb.hi, c, w));  // a polynomial in w
    const dd::DD wq = dd::DD{-z, 0.0} / omz;
    return times_power(f_near_one(A, cb, c, dd::DD{1.0, 0.0} / omz, wq.hi));  // 1 - w = 1/(1 - z)
}

}  // namespace hyper_detail

struct Hyp1f1 : Real {
    static constexpr const char* name = "hyp1f1";
    static constexpr int nin = 3, nout = 1;
    static constexpr std::array<const char*, 3> args{"a", "b", "x"};
    static constexpr double cost_ns = cost::kHyp1f1;
    static double f(double a, double b, double x) { return hyper_detail::hyp1f1(a, b, x); }
};

struct Hyp2f1 : Real {
    static constexpr const char* name = "hyp2f1";
    static constexpr int nin = 4, nout = 1;
    static constexpr std::array<const char*, 4> args{"a", "b", "c", "z"};
    static constexpr double cost_ns = cost::kHyp2f1;
    static double f(double a, double b, double c, double z) { return hyper_detail::hyp2f1(a, b, c, z); }
};

}  // namespace sd::special
