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
// terms only: a >= 0, b > 0 and (x >= 0 or b >= a). Elsewhere both forms
// alternate and the one with less cancellation is used (the loose region).
//
// hyp2f1(a, b, c, z), z < 1:
//   0 <= z <= 0.9    the Maclaurin series
//   -1 <= z < 0      Pfaff: (1-z)^-a 2F1(a, c-b; c; z/(z-1)), the larger of a, b
//                    first so that c - b >= 0 when c >= min(a, b)
//   z > 0.9          the transformation to 1 - z (AS 15.3.6), and its limit
//                    forms with digamma when c - a - b is an integer (AS 15.3.10-12)
//   z < -1           Pfaff to w = z/(z-1) in (1/2, 1), then as above in w
// The guaranteed region: a, b >= 0, c > 0, c >= min(a, b), -1 <= z <= 0.9.
// Polynomials (a or b a non-positive integer) are summed exactly as far as
// they go, for any z; c a non-positive integer is a pole (inf) unless the
// polynomial ends first.
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
};
// The float64 pass of a series is kept while sum (k+1)|t_k| <= kFastSeries |s|: at
// most ~5 ULP from the products (measured in bench/special/opt3).
inline constexpr double kFastSeries = 4.0;
inline Series m_series(double a, double b, double y, double a_lo = 0.0) {
    // The terms in double-double: by successive products a float64 term k
    // carries ~k ULP (a + k, b + k rounded, then multiplied), and near x ~ 500
    // the sum is made of terms around k ~ x: 120-180 ULP, measured. a + k and
    // b + k are exact as two_sums.
    if (a_lo == 0.0) {
        // First in float64 with a bound on what the products cost: term k carries at
        // most ~1.25 (k + 1) ULP, so the sum's error is at most 1.25 sum (k+1)|t_k| / |s|
        // ULP. Where that is small (the terms fall from the start: most inputs) this is
        // the answer, at a fifth of the double-double loop's time.
        double fs = 1.0, fc = 0.0, ft = 1.0, fabs_sum = 1.0, weighted = 1.0;  // fs + fc: a compensated sum
        bool done = false;
        for (int k = 0; k < 2000; ++k) {
            if (a + k == 0.0) {
                done = true;
                break;
            }
            ft *= (a + k) * y / ((b + k) * (k + 1.0));
            const dd::DD two = dd::two_sum(fs, ft);
            fs = two.hi;
            fc += two.lo;
            const double at = std::fabs(ft);
            fabs_sum += at;
            weighted += (k + 2.0) * at;
            if (!(at < 0x1p600)) break;
            if (ft == 0.0 || (k > std::fabs(y) && k > -b && at <= 0x1p-56 * std::fabs(fs))) {
                done = true;
                break;
            }
        }
        fs += fc;
        if (done && weighted <= kFastSeries * std::fabs(fs)) return {scaled(fs), fs == 0.0 ? kInf : fabs_sum / std::fabs(fs)};
    }
    // The long series, compensated (as f_series2 below: the roundings of the product
    // and the quotient by fma, to first order; the term and the sum as hi + lo).
    double th = 1.0, tl = 0.0, sh = 1.0, sl = 0.0, abssum = 1.0;
    std::int64_t e = 0;
    for (int k = 0; k < 1000000; ++k) {
        if (a + k == 0.0) break;  // a polynomial ends (before b + k = 0 can divide)
        const dd::DD A = dd::two_sum(a, k), B = dd::two_sum(b, k);
        const double kp = k + 1.0;
        const double n = A.hi * y, e1 = std::fma(A.hi, y, -n) + (A.lo + a_lo) * y;
        const double d = B.hi * kp, e2 = std::fma(B.hi, kp, -d) + B.lo * kp;
        const double rd = 1.0 / d, q = n * rd;
        const double ql = (std::fma(-q, d, n) + e1 - q * e2) * rd;  // the ratio is q + ql
        const double tn = th * q;
        tl = std::fma(th, q, -tn) + tl * q + th * ql;
        th = tn;
        const dd::DD two = dd::two_sum(sh, th);
        sh = two.hi;
        sl += two.lo + tl;
        abssum += std::fabs(th);
        if (th == 0.0) break;
        if (std::fabs(sh) > 0x1p600 || std::fabs(th) > 0x1p600) {
            sh *= 0x1p-600, sl *= 0x1p-600, th *= 0x1p-600, tl *= 0x1p-600;
            abssum *= 0x1p-600;
            e += 600;
        }
        // past the peak (k > |y|), and past b + k = 0 where a negative b makes the terms grow again
        if (k > std::fabs(y) && k > -b && std::fabs(th) <= 0x1p-56 * std::fabs(sh)) break;
    }
    Scaled r = scaled(sh + sl);
    r.e += e;
    return {r, sh == 0.0 ? kInf : abssum / std::fabs(sh)};
}

// Gamma(b) / Gamma(a) with its sign, as m 2^e (Cephes Gamma up to 170, else via log).
inline Scaled gamma_ratio_signed(double b, double a) {
    if (std::fabs(a) < 170.0 && std::fabs(b) < 170.0) return scaled(Rgamma::f(a)) / scaled(Rgamma::f(b));
    const double s = gamma_detail::gamma_sign(a) * gamma_detail::gamma_sign(b);
    return exp_scaled(gamma_detail::lgamma_abs(b) - gamma_detail::lgamma_abs(a)) * scaled(s);
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
inline Series m_nonneg(double a, double b, double y, double a_lo = 0.0) {
    if (!nonpos_int(a) && a_lo == 0.0 && y > 600.0 + 2.0 * (std::fabs(a) + std::fabs(b))) {
        Scaled r;
        if (m_asymptotic(a, b, y, r)) return {r, 1.0};
    }
    return m_series(a, b, y, a_lo);
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
    if (nonpos_int(bma.hi) && bma.lo == 0.0) {
        out = 0.0;  // 1/Gamma(b - a) = 0: only the e^x part, below every float64 here
        return true;
    }
    // 1/Gamma(hi + lo) = (1/Gamma(hi)) (1 - psi(hi) lo)
    const double rg = Rgamma::f(bma.hi) * (1.0 - (bma.lo != 0.0 ? Digamma::f(bma.hi) * bma.lo : 0.0));
    out = value(scaled(Gamma::f(b)) * scaled(rg) * pow_scaled(y, -a) * scaled(s));
    return true;
}

inline double hyp1f1(double a, double b, double x);
inline constexpr double kMaxRecurrence = 1e6;  // steps of the b recurrence at most

// b < 0 (not an integer) with heavy cancellation in both forms: from b' = b + n
// in (0, 1] and b' + 1, where the forms are clean, down in b by DLMF 13.3.2:
//     b(b-1) M(a, b-1) + b(1-b-x) M(a, b) + x(b-a) M(a, b+1) = 0
// (measured stable in this direction; the loose region).
inline double m_b_recurrence(double a, double b, double x) {
    // callers keep n = ceil(-b) at most kMaxRecurrence (beyond it, the better series instead)
    const std::int64_t n = static_cast<std::int64_t>(std::ceil(-b));
    const double bp = b + n;  // (0, 1]: b + n is exact (b's bits below 1 survive)
    double cur = hyp1f1(a, bp, x), nxt = hyp1f1(a, bp + 1.0, x), bb = bp;
    for (std::int64_t k = 0; k < n; ++k) {
        const double prev = -(bb * (1.0 - bb - x) * cur + x * (bb - a) * nxt) / (bb * (bb - 1.0));
        nxt = cur;
        cur = prev;
        bb -= 1.0;
    }
    return cur;
}

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
        if (n > 4 && std::fabs(t) <= 0x1p-56 * std::fabs(sum) && std::fabs(an * rn) <= 0x1p-56 * big) break;
        const double nu = b - 1.0 + n;  // J_(nu+1) = (2 nu / w) J_nu - J_(nu-1)
        const double jn = 2.0 * nu / w * j - jm;
        jm = j;
        j = jn;
    }
    out = value(scaled(Gamma::f(b)) * exp_scaled(0.5 * x) * pow_scaled(kx, 0.5 * (1.0 - b)) * scaled(sum));
    return std::isfinite(out);
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
    // Boost (special/boost.hpp): the loose region, where every form of ours cancels (a < 0, x > 0
    // measured 3.9e-3, Boost 3e-12), and inside the guaranteed one x >= 0 with b < 11 (8x faster,
    // as accurate). A non-finite Boost result, or a subnormal one in the guaranteed region, is ours.
    {
        const bool guaranteed = a >= 0.0 && b > 0.0 && (x >= 0.0 || b >= a);
        if (!guaranteed || (x >= 0.0 && b < 11.0)) {
            const double r = bst::hyp1f1(a, b, x);
            if (guaranteed ? bst::normal(r) : std::isfinite(r)) return r;
            double t;  // Boost gave no value: for a large negative a, Tricomi's expansion
            if (!guaranteed && a < -1e3 && m_tricomi(a, b, x, t)) return t;
        }
    }
    if (x > 0.0) {
        const Series direct = m_nonneg(a, b, x);
        if (a >= 0.0 || direct.cancel < 4.0) return value(direct.s);
        // a < 0: Kummer's form alternates too; keep the one with less cancellation
        const dd::DD bma = dd::two_sum(b, -a);
        const Series k = m_series(bma.hi, b, -x, bma.lo);
        const double best = std::min(k.cancel, direct.cancel);
        if (b < 0.0 && b > -kMaxRecurrence && best > 1e4) return m_b_recurrence(a, b, x);
        return k.cancel < direct.cancel ? value(exp_scaled(x) * k.s) : value(direct.s);
    }
    // x < 0: Kummer, with b - a carried exactly
    if (x < -(600.0 + 2.0 * (std::fabs(a) + std::fabs(b))) && std::fabs(b) < 170.0) {
        double r;
        if (m_negative_asymptotic(a, b, x, r)) return r;
    }
    const dd::DD bma = dd::two_sum(b, -a);
    const Series kummer = m_nonneg(bma.hi, b, -x, bma.lo);
    if (b - a >= 0.0 || kummer.cancel < 4.0) return value(exp_scaled(x) * kummer.s);
    const Series direct = m_series(a, b, x);
    if (b < 0.0 && b > -kMaxRecurrence && std::min(direct.cancel, kummer.cancel) > 1e4) return m_b_recurrence(a, b, x);
    return direct.cancel < kummer.cancel ? value(direct.s) : value(exp_scaled(x) * kummer.s);
}

// ---- 2F1 ------------------------------------------------------------------------

// sum (a)_k (b)_k / ((c)_k k!) z^k, |z| < 1 (or a polynomial), in
// double-double (the terms' products would carry ~k ULP: 24 ULP at z = 0.885,
// measured), with the cancellation sum|terms| / |sum|.
struct FSum {
    double s, cancel;
};
inline FSum f_series2(double a, double b, double c, double z) {
    {  // the float64 pass (as m_series)
        double fs = 1.0, fc = 0.0, ft = 1.0, fabs_sum = 1.0, weighted = 1.0;  // fs + fc: a compensated sum
        bool done = false;
        for (int k = 0; k < 2000; ++k) {
            if (a + k == 0.0 || b + k == 0.0) {
                done = true;
                break;
            }
            ft *= (a + k) * (b + k) * z / ((c + k) * (k + 1.0));
            const dd::DD two = dd::two_sum(fs, ft);
            fs = two.hi;
            fc += two.lo;
            const double at = std::fabs(ft);
            fabs_sum += at;
            weighted += (k + 2.0) * at;
            if (!(at < 0x1p600)) break;
            if (ft == 0.0 || (k > 2 && k > -c && at <= 0x1p-56 * std::fabs(fs))) {
                done = true;
                break;
            }
        }
        fs += fc;
        if (done && weighted <= kFastSeries * std::fabs(fs)) return {fs, fs == 0.0 ? kInf : fabs_sum / std::fabs(fs)};
    }
    // The long series: each term from the last by one division, the roundings of the
    // products and of the quotient recovered by fma and carried to first order (the
    // term and the sum as hi + lo). What is left is second order, k eps^2: as accurate
    // as the double-double loop this replaces (2.7 ULP at most over z in [0.5, 0.9],
    // both, measured) at a third of its time.
    double th = 1.0, tl = 0.0, sh = 1.0, sl = 0.0, abssum = 1.0;
    for (int k = 0; k < 10000000; ++k) {
        if (a + k == 0.0 || b + k == 0.0) break;  // a polynomial ends
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
        const dd::DD two = dd::two_sum(sh, th);
        sh = two.hi;
        sl += two.lo + tl;
        abssum += std::fabs(th);
        if (th == 0.0) break;
        if (k > 2 && k > -c && std::fabs(th) <= 0x1p-56 * std::fabs(sh)) break;  // past c + k = 0
    }
    const double v = sh + sl;
    return {v, v == 0.0 ? kInf : abssum / std::fabs(v)};
}
inline double f_series(double a, double b, double c, double z) { return f_series2(a, b, c, z).s; }

inline double psi_n1(int n) {  // psi(n + 1) = H_n - euler
    double h = 0.0;
    for (int i = 1; i <= n; ++i) h += 1.0 / i;
    return h - kEuler;
}

inline double gam(double v) { return Gamma::f(v); }
inline double rgam(double v) { return Rgamma::f(v); }
inline double psi(double v) { return Digamma::f(v); }

// 2F1(a, b; c; 1 - y) for 0 < y <= 1/2 by the transformation to y (AS 15.3.6, 15.3.10-12).
inline double f_one_minus(double a, double b, double c, double y) {
    const double s = c - a - b;
    const double ly = std::log(y);
    if (s != std::floor(s) && std::fabs(a + b + std::nearbyint(s) - c) != 0.0) {
        const double t1 = gam(c) * gam(s) * rgam(c - a) * rgam(c - b) * f_series(a, b, 1.0 - s, y);
        const double t2 = std::pow(y, s) * gam(c) * gam(-s) * rgam(a) * rgam(b) * f_series(c - a, c - b, 1.0 + s, y);
        return t1 + t2;
    }
    // s is m up to the rounding of a + b; the degenerate forms sum m terms, so a huge m
    // (only with |a + b| huge, the loose region) takes the series instead
    if (std::fabs(s) > 1e5) return f_series(a, b, c, 1.0 - y);
    const int m = static_cast<int>(std::fabs(std::nearbyint(s)));
    auto tail = [&](double a0, double b0, double pa, double pb) {
        // sum_n (a0)_n (b0)_n / (n! (n+m)!) y^n [ly - psi(n+1) - psi(n+m+1) + psi(pa+n) + psi(pb+n)]
        double sum = 0.0, t = 1.0;
        for (int i = 1; i <= m; ++i) t /= i;  // 1/m!
        for (int n = 0; n < 100000; ++n) {
            const double term = t * (ly - psi_n1(n) - psi_n1(n + m) + psi(pa + n) + psi(pb + n));
            sum += term;
            if (n > 2 && std::fabs(term) <= 0x1p-56 * std::fabs(sum)) break;
            t *= (a0 + n) * (b0 + n) / ((n + 1.0) * (n + m + 1.0)) * y;
            if (t == 0.0) break;
        }
        return sum;
    };
    if (m == 0) {
        // AS 15.3.10
        double sum = 0.0, t = 1.0;
        for (int n = 0; n < 100000; ++n) {
            const double term = t * (2.0 * psi_n1(n) - psi(a + n) - psi(b + n) - ly);
            sum += term;
            if (n > 2 && std::fabs(term) <= 0x1p-56 * std::fabs(sum)) break;
            t *= (a + n) * (b + n) / ((n + 1.0) * (n + 1.0)) * y;
            if (t == 0.0) break;
        }
        return gam(a + b) * rgam(a) * rgam(b) * sum;
    }
    // finite part: sum_{n<m} (p)_n (q)_n / (n! (1-m)_n) y^n
    auto finite = [&](double p, double q) {
        double sum = 0.0, t = 1.0;
        for (int n = 0; n < m; ++n) {
            sum += t;
            t *= (p + n) * (q + n) / ((n + 1.0) * (1.0 - m + n)) * y;
        }
        return sum;
    };
    const double gm = m <= 23 ? kFactorial[m - 1] : gam(m);  // Gamma(m) = (m-1)!; kFactorial[k] = k!
    if (s > 0.0) {
        // AS 15.3.11, c = a + b + m
        const double f = gm * gam(c) * rgam(a + m) * rgam(b + m) * finite(a, b);
        const double r = rgam(a) * rgam(b);
        if (r == 0.0) return f;
        return f - std::pow(-y, m) * gam(c) * r * tail(a + m, b + m, a + m, b + m);
    }
    // AS 15.3.12, c = a + b - m
    const double f = gm * gam(c) * rgam(a) * rgam(b) * std::pow(y, -m) * finite(a - m, b - m);
    const double r = rgam(a - m) * rgam(b - m);
    if (r == 0.0) return f;
    return f - ((m % 2) ? -1.0 : 1.0) * gam(c) * r * tail(a, b, a, b);
}

// 2F1 for 0 <= z < 1 (non-polynomial). Near z = 1 the transformation to 1 - z;
// when c - a - b is within 1e-3 of an integer m without being one, its two
// terms Gamma(+-s) cancel ~1/|s - m|: then the series itself while 1 - z >=
// 1e-3 (~4e4 terms at most), and the limit form at m once |s - m| < 1e-8
// (its error ~|s - m|).
inline double f_unit(double a, double b, double c, double z) {
    if (z <= 0.9) return f_series(a, b, c, z);
    // c far beyond a and b: the terms fall like (a b / c)(k / c)^k from the start, and the
    // 1 - z transformation's Gammas would overflow (NaN) -- the series directly
    if (c > 1e3 * std::max({1.0, std::fabs(a), std::fabs(b)})) return f_series(a, b, c, z);
    const double s = c - a - b, m = std::nearbyint(s), off = std::fabs(s - m);
    if (off != 0.0 && off < 1e-3) {
        if (1.0 - z >= 1e-3) return f_series(a, b, c, z);  // at most ~4e4 terms (rare)
        if (off < 1e-8) {
            // the limit form at c' = a + b + m (rounding c by ~|s - m|)
            return f_one_minus(a, b, a + b + m, 1.0 - z);
        }
    }
    return f_one_minus(a, b, c, 1.0 - z);  // 1 - z exact (Sterbenz)
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
        const double sg = gamma_detail::gamma_sign(c) * gamma_detail::gamma_sign(hi - lo) * gamma_detail::gamma_sign(hi) *
                          gamma_detail::gamma_sign(c - lo);
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
    if (poly) return f_series(a, b, c, z);
    if (z > 0.0) return f_unit(a, b, c, z);
    // z < 0: Pfaff with the larger of a, b first: (1-z)^-a 2F1(a, c-b; c; w)
    const double A = std::max(a, b), B = std::min(a, b);
    const double w = z / (z - 1.0);
    const dd::DD omz = dd::two_sum(1.0, -z);  // 1 - z, exact
    const double pw = std::pow(omz.hi, -A) * (1.0 - A * omz.lo / omz.hi);
    if (w <= 0.9) return pw * f_series(A, c - B, c, w);
    return pw * f_one_minus(A, c - B, c, 1.0 / omz.hi);  // 1 - w = 1/(1 - z)
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
