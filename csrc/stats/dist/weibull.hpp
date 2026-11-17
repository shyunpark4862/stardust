// The Weibull distribution with shape c and scale s (scipy's weibull_min; T18):
// P(X > x) = exp(-t), t = (x/s)^c. t = exp(c log(x/s)) with the exponent a
// double-double (exp multiplies its rounding by |c log(x/s)|), and the
// inverses x = s exp(log(t) / c) the same way.
//
// The moments are m_k = s^k Gamma(1 + k/c). As c grows X concentrates
// (X ~ s (1 + log E / c)) and the central moments cancel: the variance
// Gamma(1 + 2x) - Gamma(1 + x)^2 (x = 1/c) is O(x^2) out of terms of size 1.
// So they are written through
//     D_k = log Gamma(1 + k x) - k log Gamma(1 + x)
//         = sum_{j>=2} (-1)^j zeta(j) / j ((k x)^j - k x^j)       (k x < 1)
// in which the linear terms cancel exactly, and with E_k = e^(D_k) - 1:
//     mu_2 / m_1^2 = E_2,   mu_3 / m_1^3 = E_3 - 3 E_2,   mu_4 / m_1^4 = E_4 - 4 E_3 + 6 E_2,
// all in double-double (the skewness crosses 0 near c = 3.6, the excess
// kurtosis near c = 2.25 and 5.77). Where k x is too large for the series,
// D_k comes from gammaln directly.
#pragma once

#include "special/ops/gamma.hpp"
#include "stats/coeffs/zeta_minus_one.hpp"
#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace weibull_detail {

using num::DD;
namespace dd = num::dd;

// log(x / s) as a double-double, for x, s > 0 finite.
inline DD log_ratio(double x, double s) {
    if (x < 0x1p-960) [[unlikely]] {
        // the quotient's remainder would be subnormal: both scaled up (exact) when s is small
        // too, else the quotient is far below 1 and the two logs do not cancel
        if (s < 0x1p-700) return log_ratio(x * 0x1p200, s * 0x1p200);
        return num::log_dd(x) - num::log_dd(s);
    }
    const DD y = num::ratio(x, s);
    if (std::isfinite(y.hi) && y.hi >= 0x1p-960) return dd::log(y);  // (below, y's low part is subnormal)
    return num::log_dd(x) - num::log_dd(s);
}

// D_k by the series, x = 1/c a double-double, k x <= 0.85.
inline DD lgamma_diff(int k, DD x) {
    const DD kx = x * static_cast<double>(k);
    DD pk = kx * kx, p1 = x * x, sum{0.0, 0.0};
    for (int j = 2; j < 4000; ++j) {
        DD zeta{1.0, 0.0};
        if (j <= kZetaMinusOneMax) zeta = zeta + DD{kZetaMinusOne[j - 2][0], kZetaMinusOne[j - 2][1]};
        DD term = (zeta / static_cast<double>(j)) * (pk - p1 * static_cast<double>(k));
        if (j % 2 == 1) term = -term;
        sum = sum + term;
        if (std::fabs(term.hi) <= std::fabs(sum.hi) * 0x1p-108) break;
        pk = pk * kx;
        p1 = p1 * x;
    }
    return sum;
}

inline constexpr double kSeriesMax = 0.85;  // the largest k x for the series

// D_k = log Gamma(1 + k x) - k log Gamma(1 + x): the series while |k x| is small, else
// the Stirling forms in double-double (no cancellation left that matters: D_k is then of
// the size of its terms). x may be negative (GEV's c; 1 + k x > 0 is the caller's).
inline DD lgamma_ratio_x(int k, DD x) {
    if (std::fabs(k * x.hi) <= kSeriesMax) return lgamma_diff(k, x);
    return num::log_gamma(x * static_cast<double>(k) + 1.0) - num::log_gamma(x + 1.0) * static_cast<double>(k);
}
// The Weibull's D_k, x = 1/c.
inline DD lgamma_ratio(int k, double c) { return lgamma_ratio_x(k, num::div(DD{1.0, 0.0}, c)); }

// The skewness of W with E[W^k] = Gamma(1 + k x): (E_3 - 3 E_2) / E_2^(3/2); past
// D_3 = 600 scaled by e^-D_3 (the E's overflow).
inline double skewness_x(DD x) {
    // About x = 0 the numerator E_3 - 3 E_2 E_1 + ... cancels to O(x^3) of terms O(x^2): below
    // 2^-40 the limit (log of an Exp(1): -+12 sqrt 6 zeta(3) / pi^3) and its slope (from the
    // Gamma form in Arb; the next term is O(x^2) < 2^-78).
    if (std::fabs(x.hi) < 0x1p-40) {
        const double s = -1.1395470994046486 + 5.966612412667091 * std::fabs(x.hi);
        return x.hi > 0.0 ? s : -s;
    }
    if (x.hi > 1e17) return kInf;  // Gamma(1 + 3x) / Gamma(1 + 2x)^(3/2) beyond the range
    const DD d2 = lgamma_ratio_x(2, x), d3 = lgamma_ratio_x(3, x);
    if (d3.hi > 600.0) {
        const double a = d2.hi, b = d3.hi;
        const double n = -std::expm1(-b) - 3.0 * (std::exp(a - b) - std::exp(-b));
        const double den = std::pow(-std::expm1(-a), 1.5);
        return std::exp(b - 1.5 * a) * n / den;
    }
    const DD e2 = num::expm1_full(d2), e3 = num::expm1_full(d3);
    const DD n = e3 - e2 * 3.0;
    return (n / (e2 * std::sqrt(e2.hi))).hi * (1.0 - 0.5 * e2.lo / e2.hi);
}
// Its excess kurtosis: (E_4 - 4 E_3 + 6 E_2) / E_2^2 - 3; past D_4 = 600 scaled by e^-D_4.
// The numerator cancels its x^2 terms (O(x^4) out of O(x^2): 1e-14 left of double-double at
// x = 1e-9), so below |x| = 1e-4 the Taylor series about x = 0 (its coefficients from the
// exact Gamma form; against Arb the residual at |x| = 1e-3 is the next term's, 1e-16; radius
// 1/4, the next term below 1e-22 at |x| = 1e-4).
inline double kurtosis_x(DD x) {
    if (std::fabs(x.hi) < 1e-4) {
        constexpr double k[7] = {2.4, -28.917688340799625, 152.61603300554222, -666.448329815961,
                                 2877.87676765926, -12120.830749353447, 50326.42880216044};
        double r = k[6];
        for (int i = 5; i >= 0; --i) r = r * x.hi + k[i];
        return r;
    }
    if (x.hi > 1e17) return kInf;  // Gamma(1 + 4x) / Gamma(1 + 2x)^2 beyond the range
    const DD d2 = lgamma_ratio_x(2, x), d3 = lgamma_ratio_x(3, x), d4 = lgamma_ratio_x(4, x);
    if (d4.hi > 600.0) {
        const double a = d2.hi, b = d3.hi, e = d4.hi;
        const double em = std::exp(-e);
        const double n = (1.0 - em) - 4.0 * (std::exp(b - e) - em) + 6.0 * (std::exp(a - e) - em) -
                         3.0 * std::exp(2.0 * a - e) * std::pow(-std::expm1(-a), 2.0);
        const double den = std::exp(2.0 * a - e) * std::pow(-std::expm1(-a), 2.0);
        return n / den - 3.0;
    }
    const DD e2 = num::expm1_full(d2), e3 = num::expm1_full(d3), e4 = num::expm1_full(d4);
    const DD den = e2 * e2;
    const DD n = e4 - e3 * 4.0 + e2 * 6.0 - den * 3.0;
    const DD r = n / den;
    return r.hi + r.lo;
}

// ---- chunk kernels of the distributions with P = e^-t, t = e^E (Weibull, Gumbel, GEV) ------
//
// t = th + tl comes with a bound tau on its relative error (in kUlp). From E = log t, given
// as hi + lo with an absolute error bound eu, in two grades:
//   plain:  th = exp(E.hi) by vForce, tl = th E.lo: relative error tau = eu + kVfExp + 1;
//   folded: also r = E - log(th) (ckd::log_dd; E.hi - log(th).hi is exact, the two being
//           within a few ULP) and tl = th r, the scalar definition's own fold: exp's rounding
//           is gone and tau = eu + kLq |E| + 1/2 (log_dd's error, and the rounding of r and
//           th r, terms below 2^-51 of t).
// NaN (th and tau, and so every bound) where th leaves [2^-999, 2^999].
inline void t_chunk(const double* eh, const double* el, const double* eu, bool fold, double* th, double* tl,
                    double* tau, int n) {
    ck::exp(th, eh, n);
    if (!fold) {
        for (int i = 0; i < n; ++i) {
            const bool ok = th[i] >= 0x1p-999 && th[i] <= 0x1p999;
            tl[i] = th[i] * el[i];
            tau[i] = ok ? eu[i] + ck::kVfExp + 1.0 : kNaN;
            th[i] = ok ? th[i] : kNaN;
        }
        return;
    }
    double lh[kChunkSize], ll[kChunkSize];
    for (int i = 0; i < n; ++i) tl[i] = 0.0;
    ckd::log_dd(th, tl, lh, ll, n);  // NaN outside [2^-999, 2^999]
    for (int i = 0; i < n; ++i) {
        const double r = (eh[i] - lh[i]) + (el[i] - ll[i]);
        const bool ok = lh[i] == lh[i];
        tl[i] = th[i] * r;
        tau[i] = ok ? eu[i] + ckd::kLq * std::fabs(eh[i]) + 0.5 : kNaN;
        th[i] = ok ? th[i] : kNaN;
    }
}

// The four tails of P = e^-t from t = th + tl (relative error tau), with a bound on the
// result's relative error in kUlp (NaN or above the budget: not this path's element).
//   kExp:     e = exp(-th) (1 - tl) by vForce: t tau + kVfExp + 2. Results below 2^-1000 are
//             refused.
//   kOneExp:  f = m + (1 - m) tl, m = 1 - e^-th (ckd::one_minus_exp, with its bound per
//             element: tanh's kTanh reaches m times 1 / (1 + T) = 1 - m/2, plus 2; 1 - m =
//             e^-th only scales the small correction): tau times the condition
//             t e^-t / (1 - e^-t) <= 1, plus kTanh (1 - m/2) + 3.
//   kLogExp:  -t: tau + 1.
//   kLogOne:  log(1 - e^-t), all by libm (2 units a call; vForce's bounds would leave no
//             room near t = log 2): below t = log 2 log(f) with f = -expm1(-th) + e^-th tl
//             (tau times the condition, plus 3) over |log f| >= log 2; from there
//             log1p(-e), e = exp(-th) (1 - tl) (t tau + 3) times e / ((1 - e) |log1p(-e)|) <=
//             1 / (1 - e).
enum class Tail { kExp, kOneExp, kLogExp, kLogOne };
inline void tail_chunk(Tail kind, const double* th, const double* tl, const double* tau, double* out,
                       double* bound, int n) {
    if (kind == Tail::kLogExp) {
        for (int i = 0; i < n; ++i) {
            out[i] = -(th[i] + tl[i]);
            bound[i] = tau[i] + 1.0;
        }
        return;
    }
    if (kind == Tail::kLogOne) {
        for (int i = 0; i < n; ++i) {
            if (th[i] < num::kLn2.hi) {
                const double m = -std::expm1(-th[i]);
                const double f = m + (1.0 - m) * tl[i];
                out[i] = std::log(f);
                bound[i] = (tau[i] * std::min(1.0, th[i] * (1.0 - m) / m) + 3.0) / std::fabs(out[i]) + 2.0;
            } else {
                const double e = std::exp(-th[i]) * (1.0 - tl[i]);
                out[i] = std::log1p(-e);
                bound[i] = (th[i] * tau[i] + 3.0) / (1.0 - e) + 2.0;
            }
        }
        return;
    }
    if (kind == Tail::kExp) {
        for (int i = 0; i < n; ++i) out[i] = -th[i];
        ck::exp(out, out, n);
        for (int i = 0; i < n; ++i) {
            out[i] *= 1.0 - tl[i];
            bound[i] = out[i] >= 0x1p-1000 ? th[i] * tau[i] + ck::kVfExp + 2.0 : kNaN;
        }
        return;
    }
    ckd::one_minus_exp(out, th, n);
    for (int i = 0; i < n; ++i) {
        const double m = out[i];
        out[i] = m + (1.0 - m) * tl[i];
        bound[i] = tau[i] * std::min(1.0, th[i] * (1.0 - m) / m) + ckd::kTanh * (1.0 - 0.5 * m) + 3.0;
    }
}

}  // namespace weibull_detail

struct Weibull {
    static constexpr char name[] = "weibull";
    static constexpr std::array<const char*, 2> params{"c", "scale"};

    static bool valid(double c, double s) { return c > 0.0 && std::isfinite(c) && s > 0.0 && std::isfinite(s); }

    // t = (x/s)^c as a double-double, and E = c log(x/s). exp(E.hi) is rounded, and e^-t
    // multiplies that (relative) rounding by t: past t = 4 (more than ~4 ULP) the
    // residual E - log(exp(E.hi)) is folded into the low part.
    struct T {
        num::DD t, e;
    };
    // c log(x/s); +-inf where the product overflows (the double-double product is NaN there).
    static num::DD e_of(num::DD ly, double c) {
        const double a = ly.hi * c;
        return std::isinf(a) ? num::DD{a, 0.0} : ly * c;
    }
    static T t_of(double x, double c, double s) {
        const num::DD e = e_of(weibull_detail::log_ratio(x, s), c);
        const double th = std::exp(e.hi);
        if (th == 0.0 || std::isinf(th)) return {num::DD{th, 0.0}, e};
        if (th > 4.0) {
            const num::DD r = e - num::log_dd(th);
            return {num::DD{th, th * (r.hi + r.lo)}, e};
        }
        return {num::DD{th, th * e.lo}, e};
    }

    // log c - log x + E - t: the log density at x > 0 finite.
    // log c - log x + E - t = (log c - log s) + (c - 1) log y - t, y = x/s, with
    // (c - 1) log y as E - log y (c - 1 is not exact below 1/2) and log c - log s
    // remembered per parameters (num::Memo): one log per element.
    static num::DD log_density(double x, double c, double s) {
        const num::DD ly = weibull_detail::log_ratio(x, s);
        const num::DD e = e_of(ly, c);
        if (std::isinf(e.hi)) return {-kInf, 0.0};  // c of 1e305 or more: (c - 1) log y or t beyond the range
        const double th = std::exp(e.hi);
        if (std::isinf(th)) return {-kInf, 0.0};
        num::DD t{th, th * e.lo};
        if (th > 4.0) {
            const num::DD r = e - num::log_dd(th);
            t = num::DD{th, th * (r.hi + r.lo)};
        }
        thread_local num::Memo memo;
        const num::DD k = memo.get(c, s, [&] { return num::log_dd(c) - num::log_dd(s); });
        return k + e - ly - t;
    }
    static double pdf(double x, double c, double s) {
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return c < 1.0 ? kInf : (c == 1.0 ? 1.0 / s : 0.0);
        const num::DD l = log_density(x, c, s);
        return std::isinf(l.hi) ? 0.0 : num::exp_dd(l);
    }
    static double logpdf(double x, double c, double s) {
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return c < 1.0 ? kInf : (c == 1.0 ? -std::log(s) : -kInf);
        const num::DD l = log_density(x, c, s);
        return l.hi + l.lo;
    }
    static double cdf(double x, double c, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return 1.0;
        const num::DD t = t_of(x, c, s).t;
        return -std::expm1(-t.hi) + std::exp(-t.hi) * t.lo;
    }
    static double ccdf(double x, double c, double s) {
        if (x <= 0.0) return 1.0;
        if (std::isinf(x)) return 0.0;
        const num::DD t = t_of(x, c, s).t;
        return std::isinf(t.hi) ? 0.0 : num::exp_dd(-t);
    }
    static double logcdf(double x, double c, double s) {
        if (x <= 0.0) return -kInf;
        if (std::isinf(x)) return 0.0;
        const T v = t_of(x, c, s);
        if (v.t.hi < 0x1p-60) {  // log(1 - e^-t) = log t - t/2 + ...: E to the last bit
            if (std::isinf(v.e.hi)) return -kInf;
            const num::DD l = v.e - v.t * 0.5;
            return l.hi + l.lo;
        }
        if (v.t.hi < num::kLn2.hi) return std::log(-std::expm1(-v.t.hi) + std::exp(-v.t.hi) * v.t.lo);
        return std::isinf(v.t.hi) ? 0.0 : std::log1p(-num::exp_dd(-v.t));
    }
    static double logccdf(double x, double c, double s) {
        if (x <= 0.0) return 0.0;
        if (std::isinf(x)) return -kInf;
        const num::DD t = t_of(x, c, s).t;
        return -(t.hi + t.lo);
    }

    // ---- chunk kernels ----------------------------------------------------------------
    //
    // t = (x/s)^c for the chunk by vForce's pow (ckd::pow_dd on the scalar definition's
    // double-double x/s): relative error tau = kPowDd whatever c and t. NaN (both) where x/s or
    // t leaves [2^-999, 2^999] or c > 2^30.
    static void t_pow_chunk(const Chunk<2>& c, double* th, double* tl, double* tau) {
        double yh[kChunkSize], yl[kChunkSize];
        ckd::ratio(c.x, c.p[1], yh, yl, c.n);
        ckd::pow_dd(th, yh, yl, c.p[0], c.n);
        for (int i = 0; i < c.n; ++i) {
            const bool ok = yh[i] >= 0x1p-999 && yh[i] <= 0x1p999 && th[i] >= 0x1p-999 && th[i] <= 0x1p999 &&
                            c.p[0][i] <= 0x1p30;
            th[i] = ok ? th[i] : kNaN;
            tl[i] = 0.0;
            tau[i] = ok ? ckd::kPowDd : kNaN;
        }
    }
    // E = c log(x/s) and log(x/s) for the chunk in the scalar definition's double-doubles
    // (ckd::ratio, ckd::log_dd for dd::log): E's absolute error is eu = kLq |E|. NaN where x/s
    // is not a positive number of the normal range.
    static void log_t_chunk(const Chunk<2>& c, double* eh, double* el, double* eu, double* lh, double* ll) {
        ckd::ratio(c.x, c.p[1], eh, el, c.n);
        ckd::log_dd(eh, el, lh, ll, c.n);
        for (int i = 0; i < c.n; ++i) {
            const num::DD e = num::DD{lh[i], ll[i]} * c.p[0][i];
            eh[i] = e.hi, el[i] = e.lo;
            eu[i] = ckd::kLq * std::fabs(e.hi);
        }
    }
    // One tail (weibull_detail::tail_chunk) in one grade:
    //   0: t by pow (tau = kPowDd);
    //   1: E in double-double, t = e^E plain (tau = kVfExp + 1: no better than grade 0, but for
    //      c > 2^30);
    //   2: t folded (tau = 1/2): for a large t, where e^-t multiplies tau by t.
    // fast[i] where x > 0 and the bound is inside the budget.
    static void tail_kernel(const Chunk<2>& c, double* out, bool* fast, weibull_detail::Tail kind, int grade) {
        double eh[kChunkSize], el[kChunkSize], eu[kChunkSize], th[kChunkSize], tl[kChunkSize];
        if (grade == 0) {
            t_pow_chunk(c, th, tl, eu);
        } else {
            log_t_chunk(c, eh, el, eu, th, tl);
            weibull_detail::t_chunk(eh, el, eu, grade > 1, th, tl, eu, c.n);
        }
        weibull_detail::tail_chunk(kind, th, tl, eu, out, eh, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = c.x[i] > 0.0 && eh[i] <= kFastBudget;  // false for NaN
    }
    // Grade 0, then grade 2 for what it refuses, then the scalar definition (x <= 0, inf,
    // results in the subnormals).
    template <class F>
    static void tail_run(const Chunk<2>& c, double* out, weibull_detail::Tail kind, F&& precise) {
        bool fast[kChunkSize];
        tail_kernel(c, out, fast, kind, 0);
        ckd::retry(c, out, fast, [&](const Chunk<2>& s, double* o, bool* f) { tail_kernel(s, o, f, kind, 2); });
        ck::fallback(c.n, fast, out, [&](int i) { return precise(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kOneExp, cdf); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kExp, ccdf); }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogOne, logcdf); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogExp, logccdf); }

    // The density in the same grades (k = log c - log s, a double-double per parameters).
    //   0: pdf = c t e^-t / x in double, t by pow: t's error reaches it times |1 - t|, then exp
    //      (kVfExp) and three operations: tau |1 - t| + kVfExp + 3. logpdf is its log by
    //      vForce: that bound plus kVfLog |value|, against logpdf's mixed bound max(1, |value|)
    //      (T20). Refused where c t e^-t or the density leaves the normal range.
    //   2: k + E - log y - t in the scalar definition's double-doubles (ckd::log_dd for
    //      dd::log), t folded: the exponent's absolute error is t tau + kLq (|E| + |log y|) and
    //      the last bit of the sum (1); pdf adds exp's kVfExp + 1.
    static void density_kernel(const Chunk<2>& c, double* out, bool* fast, bool log, int grade) {
        double eh[kChunkSize], el[kChunkSize], tau[kChunkSize], th[kChunkSize], tl[kChunkSize], lh[kChunkSize],
            ll[kChunkSize];
        if (grade == 0) {
            t_pow_chunk(c, th, tl, tau);
            for (int i = 0; i < c.n; ++i) el[i] = -th[i];
            ck::exp(el, el, c.n);
            for (int i = 0; i < c.n; ++i) {
                const double m = c.p[0][i] * th[i] * el[i];
                out[i] = m / c.x[i];
                tau[i] = tau[i] * std::fabs(1.0 - th[i]) + ck::kVfExp + 3.0;
                fast[i] = c.x[i] > 0.0 && m >= 0x1p-900 && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;
            }
            if (log) ck::log(out, out, c.n);
            for (int i = 0; i < c.n; ++i) {
                const double v = std::fabs(out[i]);
                fast[i] = fast[i] && (log ? tau[i] + ck::kVfLog * v <= kFastBudget * std::max(1.0, v) : tau[i] <= kFastBudget);
            }
            return;
        }
        log_t_chunk(c, eh, el, tau, lh, ll);
        weibull_detail::t_chunk(eh, el, tau, grade > 1, th, tl, tau, c.n);
        double* kh = out;  // k's high part, until the result replaces it
        double kl[kChunkSize];
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[0][i]) - num::log_dd(c.p[1][i]); });
        for (int i = 0; i < c.n; ++i) {
            const num::DD l = num::DD{kh[i], kl[i]} + num::DD{eh[i], el[i]} - num::DD{lh[i], ll[i]} - num::DD{th[i], tl[i]};
            const double abs = th[i] * tau[i] + ckd::kLq * (std::fabs(eh[i]) + std::fabs(lh[i])) + 1.0;
            if (log) {
                out[i] = l.hi + l.lo;
                fast[i] = c.x[i] > 0.0 && out[i] == out[i] && abs <= kFastBudget * std::max(1.0, std::fabs(out[i]));
            } else {
                eh[i] = l.hi, el[i] = l.lo;
                tau[i] = abs + ck::kVfExp + 1.0;
            }
        }
        if (log) return;
        ckd::exp_dd(out, eh, el, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = c.x[i] > 0.0 && tau[i] <= kFastBudget && out[i] >= 0x1p-1000;
    }
    template <class F>
    static void density_run(const Chunk<2>& c, double* out, bool log, F&& precise) {
        bool fast[kChunkSize];
        density_kernel(c, out, fast, log, 0);
        ckd::retry(c, out, fast, [&](const Chunk<2>& s, double* o, bool* f) { density_kernel(s, o, f, log, 2); });
        ck::fallback(c.n, fast, out, [&](int i) { return precise(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) { density_run(c, out, false, pdf); }
    static void logpdf_chunk(const Chunk<2>& c, double* out) { density_run(c, out, true, logpdf); }

    // x = s exp(log(t) / c) in double from t > 0 with relative error tu (kUlp): the log by
    // vForce (absolute error tu + kVfLog |log t|), the quotient e (|e| more), exp (kVfExp) and
    // the product (1). Taken while that is inside the budget and x is in the normal range.
    static void from_t_chunk(const Chunk<2>& c, const double* t, double tu, double* out, bool* fast) {
        double e[kChunkSize], bound[kChunkSize];
        ck::log(e, t, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double l = e[i];
            e[i] = l / c.p[0][i];
            bound[i] = (tu + ck::kVfLog * std::fabs(l)) / c.p[0][i] + std::fabs(e[i]) + ck::kVfExp + 1.0;
        }
        ck::exp(out, e, c.n);
        for (int i = 0; i < c.n; ++i) {
            out[i] *= c.p[1][i];
            fast[i] = fast[i] && bound[i] <= kFastBudget && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;
        }
    }
    // t = -log(1 - p): vForce's log of 1 - p from p = 1/4 (1 - p's rounding, 1 unit, is 3.5
    // of t >= 0.2877 there; kVfLog more), libm's log1p below (2): tu = 7.5.
    static void icdf_chunk(const Chunk<2>& c, double* out) {
        double t[kChunkSize];
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) t[i] = 1.0 - c.x[i];
        ck::log(t, t, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i];
            t[i] = p < 0.25 ? -std::log1p(-p) : -t[i];
            fast[i] = p > 0.0 && p < 1.0;
        }
        from_t_chunk(c, t, 7.5, out, fast);
        ck::fallback(c.n, fast, out, [&](int i) { return icdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // t = -log p by vForce: tu = kVfLog.
    static void iccdf_chunk(const Chunk<2>& c, double* out) {
        double t[kChunkSize];
        bool fast[kChunkSize];
        ck::log(t, c.x, c.n);
        for (int i = 0; i < c.n; ++i) {
            t[i] = -t[i];
            fast[i] = c.x[i] > 0.0 && c.x[i] < 1.0;
        }
        from_t_chunk(c, t, ck::kVfLog, out, fast);
        ck::fallback(c.n, fast, out, [&](int i) { return iccdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }
    // t = -lp, exact.
    static void ilogccdf_chunk(const Chunk<2>& c, double* out) {
        double t[kChunkSize];
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) {
            t[i] = -c.x[i];
            fast[i] = c.x[i] < 0.0 && c.x[i] > -kInf;
        }
        from_t_chunk(c, t, 0.0, out, fast);
        ck::fallback(c.n, fast, out, [&](int i) { return ilogccdf(c.x[i], c.p[0][i], c.p[1][i]); });
    }

    // s exp(log(t) / c) from log t (a double-double); s goes into the exponent when the
    // power alone would leave the normal range.
    static double from_log_t(num::DD lt, double c, double s) {
        if (std::isinf(lt.hi)) return lt.hi > 0.0 ? kInf : 0.0;
        const num::DD e = num::div(lt, c);
        if (std::isinf(e.hi)) return e.hi > 0.0 ? kInf : 0.0;
        if (std::fabs(e.hi) < 700.0) return s * num::exp_dd(e);
        return num::exp_dd(e + num::log_dd(s));
    }
    static double icdf(double p, double c, double s) {  // t = -log(1 - p)
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return 0.0;
        if (p == 1.0) return kInf;
        if (p > 0.5) return from_log_t(weibull_detail::dd::log(-num::log_dd(1.0 - p)), c, s);
        if (p < 0x1p-60) return from_log_t(num::log_dd(p) + 0.5 * p, c, s);  // t = p + p^2/2 + ...
        return from_log_t(num::log_dd(-std::log1p(-p)), c, s);
    }
    static double iccdf(double p, double c, double s) {  // t = -log p
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0) return kInf;
        if (p == 1.0) return 0.0;
        if (p > 0.5) return icdf(1.0 - p, c, s);
        return from_log_t(weibull_detail::dd::log(-num::log_dd(p)), c, s);
    }
    static double ilogcdf(double lp, double c, double s) {  // t = -log(1 - e^lp)
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return kInf;
        if (lp == -kInf) return 0.0;
        if (lp < -40.0) return from_log_t(num::DD{lp, 0.5 * std::exp(lp)}, c, s);  // log t = lp + p/2 + ...
        if (lp < -num::kLn2.hi) return from_log_t(num::log_dd(-std::log1p(-std::exp(lp))), c, s);
        return from_log_t(weibull_detail::dd::log(-num::log_dd(-std::expm1(lp))), c, s);
    }
    static double ilogccdf(double lp, double c, double s) {  // t = -lp
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return 0.0;
        if (lp == -kInf) return kInf;
        return from_log_t(num::log_dd(-lp), c, s);
    }

    // Gamma(1 + 1/c), 1/c a double-double: Gamma(1 + x.hi) (1 + psi(1 + x.hi) x.lo). The
    // rounding of 1/c alone would cost psi(1 + x) x ULP (300 at c = 0.013).
    static double gamma1(double c) {
        const num::DD x = num::div(num::DD{1.0, 0.0}, c);
        const double g = special::Gamma::f(1.0 + x.hi);
        const num::DD a = num::dd::two_sum(1.0, x.hi);  // 1 + x.hi, its rounding too
        return g + g * (special::Digamma::f(a.hi) * (a.lo + x.lo));
    }
    static double mean(double c, double s) {
        if (c < 1e-17) return kInf;  // s Gamma(1 + 1/c) beyond the range for every s
        const double g = gamma1(c);
        if (std::isfinite(g) && s * g >= 0x1p-1000) return s * g;
        const num::DD l = num::log_gamma(num::div(num::DD{1.0, 0.0}, c) + 1.0) + num::log_dd(s);
        return num::exp_dd(l);
    }
    // s^2 Gamma(1 + 1/c)^2 E_2, through logs when the factors overflow.
    static double variance(double c, double s) {
        using namespace weibull_detail;
        if (c < 1e-17) return kInf;  // s^2 Gamma(1 + 2/c) beyond the range for every s
        if (c > 0x1p56) {  // (pi^2 / 6) (s/c)^2 to first order in 1/c (E_2 underflows past c = 1e154)
            const double r = s / c;
            return r * (r * 1.6449340668482264);
        }
        const num::DD d2 = lgamma_ratio(2, c);
        const double m1 = s * gamma1(c);
        if (std::isfinite(m1 * m1) && m1 * m1 >= 0x1p-900 && d2.hi < 600.0) return m1 * m1 * num::expm1_full(d2).hi;
        // log var = 2 log s + log Gamma(1 + 2/c) + log(1 - e^-D_2)
        const num::DD l = num::log_gamma(num::div(num::DD{2.0, 0.0}, c) + 1.0) + num::log_dd(s) * 2.0;
        return num::exp_dd(l + std::log(-std::expm1(-d2.hi)));
    }
    static double standard_deviation(double c, double s) {
        using namespace weibull_detail;
        if (c < 1e-17) return kInf;
        if (c > 0x1p56) return (s / c) * 1.2825498301618641;  // pi / sqrt 6
        const num::DD d2 = lgamma_ratio(2, c);
        const double m1 = s * gamma1(c);
        if (std::isfinite(m1) && m1 >= 0x1p-900 && d2.hi < 600.0) return m1 * std::sqrt(num::expm1_full(d2).hi);
        const num::DD l = num::log_gamma(num::div(num::DD{2.0, 0.0}, c) + 1.0) + num::log_dd(s) * 2.0;
        return num::exp_dd((l + std::log(-std::expm1(-d2.hi))) * 0.5);
    }
    static double skewness(double c, double) { return weibull_detail::skewness_x(num::div(num::DD{1.0, 0.0}, c)); }
    static double kurtosis(double c, double) { return weibull_detail::kurtosis_x(num::div(num::DD{1.0, 0.0}, c)); }
    static double median(double c, double s) {  // s (log 2)^(1/c)
        constexpr num::DD kLogLn2 = {-0.36651292058166435, 2.0606571710351486e-17};
        return s * num::exp_dd(num::div(kLogLn2, c));
    }
    static double mode(double c, double s) {  // s ((c - 1)/c)^(1/c) for c > 1, else 0
        if (c <= 1.0) return 0.0;
        const num::DD l = num::log_dd(c - 1.0) - num::log_dd(c);  // c - 1 exact
        return s * num::exp_dd(num::div(l, c));
    }
    static double entropy(double c, double s) {  // euler (1 - 1/c) + log(s / c) + 1
        constexpr num::DD kEuler = {0.5772156649015329, -4.942915152430645e-18};
        const num::DD x = num::div(num::DD{1.0, 0.0}, c);
        if (std::isinf(x.hi)) return -kInf;
        const num::DD h = kEuler - kEuler * x + num::log_dd(s) - num::log_dd(c) + 1.0;
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
