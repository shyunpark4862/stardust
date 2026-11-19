// The gamma family (docs/api/special.md §3, docs/design/special.md §5.2; S3):
// gamma rgamma gammaln digamma polygamma beta betaln binom. Scalar
// definitions on libm through the n-ary loop; the machinery is
// special/gamma_core.hpp.
//
// Special values follow scipy except where docs/design/special.md §2.4 lists
// them (S3, user decision): gammaln(+-5e-324) and betaln(5e-324, 1) are
// finite; rgamma(-5e-324) = -5e-324; deep negative gamma and rgamma keep their
// subnormal values and signs; NaN in beta and betaln is NaN; binom with an
// integer k >= 0 is the falling factorial for every n.
#pragma once

#include "special/cost.hpp"
#include "special/boost.hpp"
#include "special/dd.hpp"
#include "special/gamma_core.hpp"
#include "special/ops/zeta.hpp"

#include <array>
#include <cmath>

namespace sd::special {

namespace gamma_detail {

inline bool is_nonpos_int(double x) { return x <= 0.0 && x == std::floor(x); }

// The sign of Gamma(x) away from its poles: + for x > 0, sign(sinpi(x)) for
// x < 0 (from Gamma(x) = -pi / (x sinpi(x) Gamma(-x))).
inline double gamma_sign(double x) { return x > 0.0 || sinpi(x) > 0.0 ? 1.0 : -1.0; }

// Candidate B for x > 0: Cephes gamma.c (recurrence into [2, 3] and P/Q,
// Stirling from 33). Measured against A in S3.
inline double gamma_cephes_pos(double x) {
    if (x > 171.62437695630272) return kInf;
    if (x >= 33.0) {
        double w = 1.0 / x;
        w = 1.0 + w * horner(w, cephes::kGammaStir);
        double y = std::exp(x);
        if (x > cephes::kMaxStir) {
            const double v = std::pow(x, 0.5 * x - 0.25);
            y = v * (v / y);
        } else {
            y = std::pow(x, x - 0.5) / y;
        }
        return 2.50662827463100050242 * y * w;  // sqrt(2 pi)
    }
    double z = 1.0;
    while (x >= 3.0) {
        x -= 1.0;
        z *= x;
    }
    while (x < 2.0) {
        if (x < 1e-9) return z / ((1.0 + kEuler * x) * x);
        z /= x;
        x += 1.0;
    }
    if (x == 2.0) return z;
    x -= 2.0;
    return z * horner(x, cephes::kGammaP) / horner(x, cephes::kGammaQ);
}

// Gamma(y), y > 0, as m 2^e: candidate B where it is finite (S3 chose it:
// more accurate and 3.4x faster), the scaled
// Lanczos form A beyond.
inline Scaled gamma_pos_scaled(double y) { return y <= 171.6 ? scaled(gamma_cephes_pos(y)) : gamma_scaled(y); }

// Gamma(x) for x < 0, not an integer, as m 2^e: -pi / (x sinpi(x) Gamma(-x)).
inline Scaled gamma_neg(double x) {
    // (no shortcut to 0 for a large |x|: as m 2^e the value is still a factor of quotients
    // that come back into range, binom(0.1, 1000) and beta(-500.5, 1000.5))
    const double y = -x;
    if (y < 0x1p-30)
        return Scaled{1.0, 0} / scaled(x) * scaled(1.0 - kEuler * x);  // 1/x - euler (x sinpi(x) underflows)
    return scaled(-kPi / (x * sinpi(x))) / gamma_pos_scaled(y);
}
// The same as a double where Gamma(-x) is finite: one division, rounded once
// (also into the subnormals), without the scaled arithmetic.
inline double gamma_neg_direct(double x) {
    const double y = -x;
    if (y > 170.0) return value(gamma_neg(x));
    return (-kPi / (x * sinpi(x))) / gamma_cephes_pos(y);
}

// 1/Gamma(x) for x < 0, not an integer: -x sinpi(x) Gamma(-x) / pi.
inline Scaled rgamma_neg(double x) {
    const double y = -x;
    if (y < 0x1p-30) return scaled(x) * scaled(1.0 + kEuler * x);
    return scaled(-x * sinpi(x) / kPi) * gamma_pos_scaled(y);
}

// The sign and size of 1/Gamma(hi + lo) for any hi: for hi < 0 through the
// reflection, with sinpi(hi + lo) = sinpi(hi) cospi(lo) + cospi(hi) sinpi(lo),
// which keeps the relative accuracy next to the poles (where 1/Gamma is near
// 0) even when lo carries the whole fraction (hi + lo = -1e300 + 0.5).
inline Scaled rgamma_dd(double hi, double lo) {
    if (hi > 0.0) return Scaled{1.0, 0} / gamma_dd(hi, lo);
    if (hi == std::floor(hi)) {  // at a pole: 0, or next to it 1/Gamma(-n + lo) = (-1)^n n! lo
        if (lo == 0.0) return {0.0, 0};
        const double n = -hi;
        const double sign = std::fmod(n, 2.0) == 0.0 ? 1.0 : -1.0;
        return scaled(sign * lo) * gamma_pos_scaled(n + 1.0);
    }
    if (hi > -0x1p-30) return scaled(hi + lo) * scaled(1.0 + kEuler * hi);  // x (1 + euler x)
    const double s = sinpi(hi) * cospi(lo) + cospi(hi) * sinpi(lo);  // exact identity, any lo
    const double y = -hi;
    return scaled(-(hi + lo) * s / kPi) * gamma_dd(y, -lo);
}

// Gamma(x) Gamma(y) / Gamma(x + y) as m 2^e, for x, y not non-positive
// integers. x + y is kept as hi + lo (its rounding would cost (x + y)
// log(x + y) / 2 ULP).
inline Scaled beta_scaled(double x, double y) {
    auto gam = [](double v) { return v > 0.0 ? gamma_scaled(v) : gamma_neg(v); };
    const TwoSum c = two_sum(x, y);
    return gam(x) * gam(y) * rgamma_dd(c.s, c.err);
}

// log|Gamma(x)| for any x (gammaln's definition, below).
inline double lgamma_abs(double x) {
    if (std::isnan(x) || std::isinf(x)) return x;
    if (x == 0.0) return kInf;
    if (x > 0.0) return lgamma_pos(x);
    if (x == std::floor(x)) return kInf;
    if (x > -0x1p-30) return -std::log(-x) - kEuler * x;  // -log|x| + euler |x|
    return std::log(kPi / std::fabs(x * sinpi(x))) - lgamma_pos(-x);
}

// log|B(a, b)| for a large (beyond the scaled range) and |b| <= a, from
// Stirling's series: with c = a + b,
//     log B = log Gamma(b) + (a - 1/2) log(a/c) - b log c + b + mu(a) - mu(c)
// where mu is Stirling's correction A(1/x^2)/x and log(a/c) = log1p(-b/c).
// For b >= 10 log Gamma(b) is expanded as well and the terms regrouped, since
// log Gamma(b) and b log c are both huge and cancel.
inline double lbeta_large(double a, double b) {
    auto mu = [](double x) {
        const double w = 1.0 / x;
        return horner(w * w, cephes::kLgamA) * w;
    };
    const double c = a + b;
    if (c == kInf) {
        // a + b beyond the doubles (so both are huge and positive): b / c and log c through
        // r = b / a <= 1 (betaln(1e300, 1.8e308) was -inf for -2.0e301)
        const double r = b / a, bc = r / (1.0 + r), lc = std::log(a) + std::log1p(r);
        return (b - 0.5) * std::log(bc) - 0.5 * lc + (a - 0.5) * std::log1p(-bc) + kLogSqrt2Pi + (mu(a) + mu(b));
    }
    if (b < 0.0) return lgamma_abs(b) + (a - 0.5) * std::log1p(-b / c) - b * std::log(c) + b + mu(a) - mu(c);
    if (b >= 10.0) {
        // with Stirling for log Gamma(b) too, the b log b and b log c terms meet as
        // (b - 1/2) log(b/c): every term then has the same sign
        return (b - 0.5) * std::log(b / c) - 0.5 * std::log(c) + (a - 0.5) * std::log1p(-b / c) + kLogSqrt2Pi +
               (mu(a) + mu(b) - mu(c));
    }
    return lgamma_pos(b) + (a - 0.5) * std::log1p(-b / c) - b * std::log(c) + b + mu(a) - mu(c);
}

// The largest argument the scaled path takes (pow_scaled then uses at most 8 parts).
inline constexpr double kScaledMax = 7000.0;

// log1p(u) - u, without the cancellation for small u: the series
// -u^2/2 + u^3/3 - ... below |u| = 1/8 (25 terms leave < 2^-60), log1p beyond.
inline double log1p_minus(double u) {
    if (std::fabs(u) >= 0.125) return std::log1p(u) - u;
    double r = 0.0;
    for (int k = 26; k >= 2; --k) r = (k % 2 ? 1.0 : -1.0) / k + u * r;
    return u * u * r;
}

// Gamma(a) / Gamma(a + b) for a > 7000 >= |b|, as m 2^e (c = a + b):
//     Gamma(a) / Gamma(c) = L(a)/L(c) e^E cgh^-b
//     E = b (b + g) / cgh + (a - 1/2) (log1p(u) - u),  u = -b/cgh,  cgh = c + g - 1/2
// (from agh / cgh = 1 + u and cgh - agh = b exactly). E is small wherever B is
// representable (b (b + g) / cgh < 1 there), so e^E does not amplify its
// rounding, unlike exp(log B); cgh^-b comes from pow with cgh kept as hi + lo.
inline Scaled gamma_ratio(double a, double b) {
    const TwoSum c = two_sum(a, b);
    const TwoSum cgh0 = two_sum(c.s, lanczos::kG - 0.5);
    const TwoSum cgh = two_sum(cgh0.s, cgh0.err + c.err);
    const double u = -b / cgh.s;
    const double E = b * (b + lanczos::kG) / cgh.s + (a - 0.5) * log1p_minus(u);
    Scaled r = scaled(lanczos_sum(a) / lanczos_sum(c.s) * std::exp(E));
    return r * pow_scaled(cgh.s, -b) * scaled(1.0 - b * (cgh.err / cgh.s));
}

// B(a, b) for a > 7000 >= b > 0: Gamma(b) Gamma(a) / Gamma(a + b).
inline Scaled beta_large(double a, double b) { return gamma_ratio(a, b) * gamma_scaled(b); }

}  // namespace gamma_detail

// Gamma(x): candidate B (Cephes) for x > 0, chosen over A (Lanczos) by the
// rule of docs/design/special.md §1.3; the reflection for x < 0. Poles: +-inf
// at +-0, NaN at the negative integers and -inf (scipy).
struct Gamma : Real {
    static constexpr const char* name = "gamma";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kGamma;
    static double f(double x) {
        using namespace gamma_detail;
        if (std::isnan(x)) return x;
        if (x == kInf) return x;
        if (x == 0.0) return std::copysign(kInf, x);
        if (x > 171.62437695630272) return kInf;
        if (std::fabs(x) < 0x1p-30) return 1.0 / x - kEuler;  // -5e-324: -inf, not inf / inf
        if (x > -170.0 && (x > 0.0 || !is_nonpos_int(x))) {  // Boost (special/boost.hpp): no subnormal tail
            const double r = bst::tgamma(x);
            if (bst::normal(r)) return r;
        }
        if (x > 0.0) return gamma_cephes_pos(x);
        if (is_nonpos_int(x)) return kNaN;  // and -inf
        return gamma_neg_direct(x);
    }
};

// The same with candidate A (Lanczos) for x > 0; private, measured in S3.
struct GammaLanczos : Real {
    static constexpr const char* name = "_gamma_lanczos";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kGamma;
    static double f(double x) {
        if (x > 0.0 && x <= 171.6) return value(gamma_scaled(x));
        return Gamma::f(x);
    }
};

// 1/Gamma(x): 0 at the poles, +-0 at +-0, 0 at +-inf (scipy). Its own scaled
// path, so it is a value (not 0) up to x = 177 and keeps the subnormals of
// rgamma(x) = x (1 + euler x) for tiny x (rgamma(-5e-324) = -5e-324).
struct Rgamma : Real {
    static constexpr const char* name = "rgamma";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kRgamma;
    static double f(double x) {
        using namespace gamma_detail;
        if (std::isnan(x)) return x;
        if (std::isinf(x)) return 0.0;
        if (x == 0.0) return x;  // keeps the sign
        if (std::fabs(x) < 0x1p-30) return x + kEuler * x * x;
        if (x > 0.0) {
            if (x <= 171.6) return 1.0 / gamma_cephes_pos(x);
            return x > 400.0 ? 0.0 : value(Scaled{1.0, 0} / gamma_scaled(x));
        }
        if (x == std::floor(x)) return 0.0;
        if (x >= -170.0) return (-x * sinpi(x) / kPi) * gamma_cephes_pos(-x);
        return value(rgamma_neg(x));
    }
};

// log|Gamma(x)| (the sign is not returned, as scipy). Poles: inf at 0 and the
// negative integers; gammaln(-inf) = -inf (scipy). x < 0:
//     log(pi / |x sinpi(x)|) - log Gamma(-x)
struct Gammaln : Real {
    static constexpr const char* name = "gammaln";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kGammaln;
    static double f(double x) { return gamma_detail::lgamma_abs(x); }
};

// psi(x) = Gamma'(x) / Gamma(x). Poles: -inf at +0, +inf at -0, NaN at the
// negative integers and -inf (scipy). x < 0:
//     (-1, 0):  psi(x + 1) - 1/x                       (no cancellation near 0)
//     <= -1:    psi(-x) - 1/x - pi cospi(x) / sinpi(x)  (reflection, -x exact)
struct Digamma : Real {
    static constexpr const char* name = "digamma";
    static constexpr int nin = 1, nout = 1;
    static constexpr std::array<const char*, 1> args{"x"};
    static constexpr double cost_ns = cost::kDigamma;
    static double f(double x) {
        if (std::isnan(x)) return x;
        if (x == kInf) return x;
        if (x == -kInf) return kNaN;
        if (x == 0.0) return std::copysign(kInf, -x);
        if (x > 0.0) {  // Boost (special/boost.hpp); x < 0 is ours (its reflection loses at the poles)
            const double r = bst::digamma(x);
            if (std::isfinite(r)) return r;
            return digamma_pos(x);
        }
        if (x == std::floor(x)) return kNaN;
        if (x > -1.0) return digamma_pos(x + 1.0) - 1.0 / x;
        return digamma_pos(-x) - 1.0 / x - kPi * cospi(x) / sinpi(x);
    }
};

// psi^(n)(x) = (-1)^(n+1) n! zeta(n + 1, x) for n = 1, 2, ...; digamma for n
// = 0. n must be a non-negative integer (int64 or float64), else NaN (Q17).
// n! comes scaled for n > 22, so n! zeta stays finite where it is.
struct Polygamma : Real {
    static constexpr const char* name = "polygamma";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"n", "x"};
    static constexpr double cost_ns = cost::kPolygamma;
    // n! zeta(n + 1, x) for n > 7000 and x > 0, where n! alone is far beyond the range:
    // from its log E = log Gamma(n + 1) - (n + 1) log x in double-double (zeta(s, x) =
    // x^-s times a sum between 1 and 1 + x / (s - 1)). inf or 0 where E says so; the
    // window between them (x near n / e) is computed up to n = 1e6 and is NaN beyond
    // (E's own error grows with n).
    static double large_order(double n, double x) {
        if (x == kInf) return 0.0;
        if (n >= 0x1p53) {
            // log(n!) - n log x = n (log n - 1 - log x) to 1/n relative: the sign decides
            const double t = std::log(n) - 1.0 - std::log(x);
            if (std::fabs(t) * n < 1e4) return kNaN;
            return t > 0.0 ? kInf : 0.0;
        }
        const dd::DD E = dd::loggamma(dd::DD{n, 0.0} + 1.0) - dd::log(dd::DD{x, 0.0}) * (n + 1.0);
        // the sum is at most 1 + x / n: 2^62 x / n would have to pass e^-800 to matter below
        if (E.hi > 720.0) return kInf;
        if (E.hi < -800.0 - 45.0) return 0.0;
        if (n > 1e6 || x > 1e8) return kNaN;
        const double rel = zeta_detail::hurwitz_relative(n + 1.0, n, x);
        return value(exp_scaled(E.hi) * scaled((1.0 + E.lo) * rel));
    }
    static double f(double n, double x) {
        if (std::isnan(n) || std::isnan(x)) return kNaN;
        if (n < 0.0 || n != std::floor(n) || std::isinf(n)) return kNaN;
        if (n == 0.0) return Digamma::f(x);
        const double sign = std::fmod(n, 2.0) == 1.0 ? 1.0 : -1.0;  // (-1)^(n+1)
        if (n > gamma_detail::kScaledMax && x > 0.0) return sign * large_order(n, x);
        if (n > gamma_detail::kScaledMax && x < 0.0 && x != std::floor(x) && std::isfinite(x)) {
            // zeta(s, x) = zeta(s, d) + (-1)^s (zeta(s, 1 - d) - zeta(s, 1 - x)), d = x's fraction
            // (zeta.hpp), s = n + 1: for a large s the nearer pole's term alone, d^-s or
            // (-1)^s (1 - d)^-s, and n! times it is beyond the range. Only at d = 1/2 with an
            // odd s do the two cancel, leaving zeta(s, 1 - x).
            const double d = x + std::ceil(-x);
            const bool s_even = sign > 0.0;  // n odd
            if (d == 0.5 && !s_even) return sign * large_order(n, 1.0 - x);
            return (d < 0.5 || s_even) ? sign * kInf : -sign * kInf;
        }
        const Scaled z = zeta_detail::hurwitz_scaled(n + 1.0, n, x);  // s - 1 = n exactly
        if (z.m == 0.0 || !std::isfinite(z.m)) return sign * z.m;
        if (n > gamma_detail::kScaledMax) return sign * std::copysign(kInf, z.m);
        return sign * value(gamma_scaled(n + 1.0) * z);
    }
};

// log|B(a, b)|. Accuracy: |error| <= 1e-14 max(1, |value|) (S3 user decision):
// near B = 1 the value is 0 and only an absolute bound is possible.
struct Betaln : Real {
    static constexpr const char* name = "betaln";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"a", "b"};
    static constexpr double cost_ns = cost::kBetaln;
    static double negint(double a, double b) {
        if (b == std::floor(b) && 1.0 - a - b > 0.0) return f(1.0 - a - b, b);
        return kInf;
    }
    static double f(double a, double b) {
        using namespace gamma_detail;
        if (std::isnan(a) || std::isnan(b)) return kNaN;
        if (is_nonpos_int(a)) return negint(a, b);
        if (is_nonpos_int(b)) return negint(b, a);
        if (a > 0.0 && b > 0.0 && a + b < 40.0) {  // log of Boost's beta (special/boost.hpp)
            const double r = bst::beta(a, b);
            if (bst::normal(r)) return std::log(r);
        }
        if (std::fabs(a) < std::fabs(b)) std::swap(a, b);
        if (a == kInf) return b == kInf ? kNaN : (b > 0.0 ? -kInf : kInf);
        if (std::fabs(a) <= kScaledMax && std::fabs(b) <= kScaledMax) return log_abs(beta_scaled(a, b));
        if (a > 0.0) return lbeta_large(a, b);
        // a < -7000: log|Gamma| term by term (mixed-bound region)
        const TwoSum c = two_sum(a, b);
        return lgamma_abs(a) + lgamma_abs(b) - lgamma_abs(c.s);
    }
};

// B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b). Poles follow scipy (Cephes
// beta.c): a non-positive integer a gives (-1)^b B(1 - a - b, b) when b is an
// integer with 1 - a - b > 0, else inf; NaN in, NaN out. Elsewhere the scaled
// path, or for a huge argument exp(log B) from lbeta_large.
struct Beta : Real {
    static constexpr const char* name = "beta";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"a", "b"};
    static constexpr double cost_ns = cost::kBeta;
    static double negint(double a, double b) {
        if (b == std::floor(b) && 1.0 - a - b > 0.0) {
            const double s = std::fmod(b, 2.0) == 0.0 ? 1.0 : -1.0;
            return s * f(1.0 - a - b, b);
        }
        return kInf;
    }
    static double f(double a, double b) {
        using namespace gamma_detail;
        if (std::isnan(a) || std::isnan(b)) return kNaN;
        if (is_nonpos_int(a)) return negint(a, b);
        if (is_nonpos_int(b)) return negint(b, a);
        if (a > 0.0 && b > 0.0 && a + b < 40.0) {  // Boost (special/boost.hpp)
            const double r = bst::beta(a, b);
            if (bst::normal(r)) return r;
        }
        if (std::fabs(a) < std::fabs(b)) std::swap(a, b);
        if (a == kInf)
            return b == kInf ? kNaN : (b > 0.0 ? 0.0 : (Gamma::f(b) > 0 ? kInf : -kInf));  // B(inf, inf): NaN (scipy)
        if (a > 0.0 && b > 0.0 && a + b <= 170.0) {
            // no Gamma overflows: Cephes' Gamma three times, a + b kept as hi + lo
            // (Gamma(a)/Gamma(c) <= 1 for c > a, so the order cannot overflow)
            const TwoSum c = two_sum(a, b);
            const double r = gamma_cephes_pos(a) / gamma_cephes_pos(c.s) * gamma_cephes_pos(b);
            return c.err == 0.0 ? r : r * (1.0 - digamma_pos(c.s) * c.err);
        }
        if (std::fabs(a) <= kScaledMax && std::fabs(b) <= kScaledMax) return value(beta_scaled(a, b));
        if (a > 0.0 && b > 0.0) return b > kScaledMax ? 0.0 : value(beta_large(a, b));
        // a huge, b < 0 within the scaled range: Gamma(a) / Gamma(a + b) as a ratio, times
        // Gamma(b) (exp(log|B|) below carries |log B| ULP: 2.4e-14 at B = 1e300)
        if (a > kScaledMax && std::fabs(b) <= kScaledMax) return value(gamma_ratio(a, b) * gamma_neg(b));
        // a negative argument beyond the scaled range (mixed-bound region): the size
        // from log|B|, the sign from the three Gammas (Gamma(a)/Gamma(a + b) > 0 for a huge a)
        const double sign = a > 0.0 ? gamma_sign(b) : gamma_sign(a) * gamma_sign(b) * gamma_sign(a + b);
        return sign * std::exp(Betaln::f(a, b));
    }
};

// binom(n, k) = Gamma(n + 1) / (Gamma(k + 1) Gamma(n - k + 1)), the
// generalized binomial coefficient.
//     k integer >= 0:   n (n-1) ... (n-k+1) / k! for every real n (S3 user
//                       decision: binom(-5, 2) = 15), as scipy's product below
//                       k = 20 (with its symmetric k for integer n); beyond, a
//                       negative integer n turns into (-1)^k binom(k - n - 1, k)
//     k negative integer:  0 (NaN for a negative integer n, scipy)
//     otherwise:        Gamma(n + 1) / Gamma(k + 1) / Gamma(n - k + 1) with
//                       Gamma(v + 1) = v Gamma(v) (exact arguments) and
//                       n - k + 1 kept as hi + lo for 1/Gamma (rgamma_dd)
struct Binom : Real {
    static constexpr const char* name = "binom";
    static constexpr int nin = 2, nout = 1;
    static constexpr std::array<const char*, 2> args{"n", "k"};
    static constexpr double cost_ns = cost::kBinom;
    static Scaled gamma1p(double v) {  // Gamma(v + 1)
        using namespace gamma_detail;
        if (v >= 1.0) return scaled(v) * gamma_scaled(v);
        const double w = v + 1.0;
        if (w > 0.0) return gamma_scaled(w);
        return Scaled{1.0, 0} / rgamma_dd(w, 0.0);
    }
    // With n or k above kScaledMax, a smaller one of k, n - k (or n + 1, |k|, k - n) above this
    // puts the result beyond the doubles: binom(2 m, m) ~ 4^m / sqrt(pi m) > 2^1024 from m = 515.
    static constexpr double kBothLarge = 1100.0;
    // psi(x + y) - psi(x) for x, y > 0 with one of them above kScaledMax, to 1e-4 relative: for
    // the first-order terms of a low part (a difference of two digammas would lose it all).
    static double psi_step(double x, double y) {
        using namespace gamma_detail;
        return x > kScaledMax ? std::log1p(y / x) : std::log(x + y) - digamma_pos(x);
    }
    // B(x, y) = Gamma(x) Gamma(y) / Gamma(x + y) for x = xh + xl, y = yh + yl > 0, at least one
    // above kScaledMax, as m 2^e; 0 where the smaller is above kBothLarge (below every double:
    // B(x, y) <= B(y, y) ~ 4^-y).
    static Scaled beta_big(double xh, double xl, double yh, double yl) {
        using namespace gamma_detail;
        if (xh < yh) {
            std::swap(xh, yh);
            std::swap(xl, yl);
        }
        if (yh > kBothLarge) return {0.0, 0};
        // d log B = (psi(x) - psi(x + y)) dx + (psi(y) - psi(x + y)) dy
        const double low = (xl != 0.0 ? -psi_step(xh, yh) * xl : 0.0) + (yl != 0.0 ? -psi_step(yh, xh) * yl : 0.0);
        return gamma_ratio(xh, yh) * gamma_scaled(yh) * scaled(1.0 + low);
    }
    static double f(double n, double k) {
        using namespace gamma_detail;
        if (std::isnan(n) || std::isnan(k)) return kNaN;
        const bool n_negint = n < 0.0 && n == std::floor(n);
        if (k == std::floor(k) && std::isfinite(k)) {
            if (k < 0.0) return n_negint ? kNaN : std::copysign(0.0, n + 1.0);  // scipy: 1 / ((n + 1) inf)
            // scipy's integer-k path, kept so its signed zeros match: the symmetric k
            // for a positive integer n (k' = n - k may be negative: then the general
            // formula), and the product num / den below k' = 20
            double kx = k;
            if (n == std::floor(n) && kx > n / 2.0 && n > 0.0) kx = n - kx;
            if (kx >= 0.0 && kx < 20.0 && (std::fabs(n) > 1e-8 || n == 0.0)) {
                double num = 1.0, den = 1.0;
                for (int i = 1; i < 1 + static_cast<int>(kx); ++i) {
                    // n - (kx - i): one rounding, none next to an integer n ((i + n) - kx rounds
                    // i + n first: binom(1 - 2^-53, 7) lost its factor n - 1 to it)
                    num *= (n - (kx - i)) + 0.0;  // (+ 0.0: a factor -0.0 at n = -0.0 is scipy's +0.0)
                    den *= i;
                    if (std::fabs(num) > 1e50) {
                        num /= den;
                        den = 1.0;
                    }
                }
                return num / den;
            }
            if (n_negint) {  // the falling factorial (S3 user decision): (-1)^k binom(k - n - 1, k)
                const double s = std::fmod(k, 2.0) == 0.0 ? 1.0 : -1.0;
                return s * f(k - n - 1.0, k);
            }
            if (n == std::floor(n) && n >= 0.0 && k > n) return 0.0;
        }
        if (n_negint) return kNaN;
        // k = +inf: no limit. k = -inf: |binom| ~ Gamma(n + 1) |k|^-(n+1) / pi with an oscillating
        // sign, so the limit is 0 for finite n > -1 (user decision 2026-11-17) and none below.
        if (std::isinf(k)) return k < 0.0 && n > -1.0 && std::isfinite(n) ? 0.0 : kNaN;
        if (std::isinf(n))
            return 1.0 / ((n + 1.0) * Beta::f(n - k + 1.0, k + 1.0));  // scipy's formula, its conventions
        const TwoSum d1 = two_sum(n, -k);
        const TwoSum d = two_sum(d1.s, 1.0);
        const double dlo = d.err + d1.err;
        // n - k + 1 a pole of Gamma: 0, signed as scipy's 1 / ((n + 1) inf)
        if (is_nonpos_int(d.s) && dlo == 0.0) return std::copysign(0.0, n + 1.0);
        if (n >= 0.0 && (n > kScaledMax || std::fabs(k) > kScaledMax)) {
            // The guaranteed region with a huge n or k. Every form below keeps its arguments
            // exact: a sum that is not a double (n + 1, k - n, n - k + 1) enters as hi + lo, the
            // Gamma quotient at hi and the low part to first order (a relative 2^-53 in an
            // argument x of B(x, y) moves it by up to 2^-53 x log(1 + y/x): 200 ULP at
            // k = 1e300, 3000 at a 7000 beside 1e7), and no log Gamma is subtracted from another.
            const TwoSum np1 = two_sum(n, 1.0);
            if (k < 0.0) {
                // Gamma(n+1) / (Gamma(k+1) Gamma(n-k+1)) with 1/Gamma(k+1) reflected:
                //     sin(pi (k + 1)) / pi  B(n + 1, -k);  sin(pi (k + 1)) = -sin(pi k), k reduced exactly
                // (sin(pi k) / pi = k below 2^-30, also where pi k is subnormal)
                const double sk = k > -0x1p-30 ? k : sinpi(k) / kPi;
                return value(scaled(-sk) * beta_big(np1.s, np1.err, -k, 0.0));
            }
            if (d1.s < 0.0) {
                // k > n, with 1/Gamma(n-k+1) reflected: sin(pi (n - k + 1)) / pi  B(n + 1, k - n).
                // sin(pi (d.s + d.err + d1.err)), the three parts each reduced exactly (their sum
                // dlo rounds: at n = 1e-10 the 1 + n lost n's digits, 8e-8 of the result)
                const double s1 = sinpi(d.s) * cospi(d.err) + cospi(d.s) * sinpi(d.err);
                const double c1 = cospi(d.s) * cospi(d.err) - sinpi(d.s) * sinpi(d.err);
                const double sn = s1 * cospi(d1.err) + c1 * sinpi(d1.err);
                if (n <= kScaledMax && -d1.s <= kScaledMax) return value(gamma1p(n) / gamma1p(k) * rgamma_dd(d.s, dlo));
                return value(scaled(sn / kPi) * beta_big(np1.s, np1.err, -d1.s, -d1.err));
            }
            // 0 <= k <= n (so n is the huge one): Gamma(n+1) / (Gamma(k+1) Gamma(n-k+1)), the
            // smaller of k, n - k apart and the other against Gamma(n+1) as a quotient
            if (std::min(k, d1.s) > kBothLarge) return kInf;  // binom(n, k) >= binom(2 kk, kk) ~ 4^kk
            if (k <= d1.s) {
                // Gamma(n-k+1) / Gamma(n+1) = ratio(a, k), a = n - k + 1 = d.s + dlo
                const Scaled q = gamma_ratio(d.s, k) * scaled(1.0 - psi_step(d.s, k) * dlo);
                return value(Scaled{1.0, 0} / q / gamma1p(k));
            }
            // Gamma(k+1) / Gamma(n+1) = ratio(a, b), a = k + 1, b = n - k = d1.s + d1.err; and Gamma(b + 1)
            const TwoSum kp1 = two_sum(k, 1.0);
            const double low = -psi_step(kp1.s, d1.s) * kp1.err - std::log(kp1.s + d1.s) * d1.err;
            const Scaled q = gamma_ratio(kp1.s, d1.s) * scaled(1.0 + low);
            const Scaled gb = gamma1p(d1.s) * scaled(1.0 + (d1.err != 0.0 ? digamma_pos(d1.s + 1.0) * d1.err : 0.0));
            return value(Scaled{1.0, 0} / q / gb);
        }
        if (std::fabs(n) > kScaledMax || std::fabs(k) > kScaledMax) {
            // n < 0 (the mixed-bound region)
            if (k > kScaledMax && std::fabs(n + 1.0) <= kScaledMax && k - n > kScaledMax) {
                // n moderate, k huge (so n - k + 1 < 0): with the reflection,
                //     binom = Gamma(n + 1) / pi  sinpi(n - k + 1)  Gamma(k - n) / Gamma(k + 1)
                // the last factor a Gamma ratio (no huge log Gamma cancels another); the sine
                // and the low part of n + 1 as above
                const TwoSum np1 = two_sum(n, 1.0);
                const double s1 = sinpi(d.s) * cospi(d.err) + cospi(d.s) * sinpi(d.err);
                const double c1 = cospi(d.s) * cospi(d.err) - sinpi(d.s) * sinpi(d.err);
                const double sn = s1 * cospi(d1.err) + c1 * sinpi(d1.err);
                const double low = np1.err != 0.0 ? -std::log(k) * np1.err : 0.0;
                return value(gamma1p(n) * scaled(sn / kPi) * gamma_ratio(k - n, np1.s) * scaled(1.0 + low));
            }
            // log|Gamma| term by term and the signs
            const double sign = gamma_sign(n + 1.0) * gamma_sign(k + 1.0) * gamma_sign(d.s);
            return sign * std::exp(lgamma_abs(n + 1.0) - lgamma_abs(k + 1.0) - lgamma_abs(d.s));
        }
        return value(gamma1p(n) / gamma1p(k) * rgamma_dd(d.s, dlo));
    }
};

}  // namespace sd::special
