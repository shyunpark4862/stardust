// The Pareto, Rayleigh and inverse gamma distributions (docs/api/stats.md §2.1,
// docs/design/stats.md T26).
//
//   * Pareto (b, scale s; x >= s): P(X > x) = e^-L, L = b log(x/s), log(x/s) as
//     log1p((x - s)/s) up to x = 2s (x - s exact) and the double-double log beyond; the
//     inverses x = s exp(L / b) with L in double-double (exp multiplies its error by L/b).
//   * Rayleigh (scale s): P(X > x) = e^-t, t = y^2 / 2 (y = x/s a double-double); the
//     inverses x = s sqrt(2t).
//   * inverse gamma (a, s): X = s / Y, Y ~ gamma(a, 1): P(X <= x) = Q(a, s/x), and the
//     density, inverses and entropy through the gamma's pieces (stats/dist/gamma.hpp).
#pragma once

#include "special/ops/gamma.hpp"
#include "stats/chunk.hpp"
#include "stats/chunk_dd.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/weibull.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

namespace lifetime_detail {

using num::DD;
namespace dd = num::dd;

// log(1 + u) for a double-double u > -1, to the last bit for a tiny u.
inline DD log1p_dd(DD u) { return u - dd::log1pmx_neg(u); }

// P(X > x) = e^-L (upper) or P(X <= x) = 1 - e^-L, or their logs, for L >= 0 a
// double-double with log L given (lnl, for the lower log near L = 0).
inline double tail_exp(DD L, DD lnl, bool upper, bool log) {
    if (upper) {
        if (log) return -(L.hi + L.lo);
        return std::isinf(L.hi) ? 0.0 : num::exp_dd(-L);
    }
    if (!log) return -std::expm1(-L.hi) + std::exp(-L.hi) * L.lo;
    if (L.hi < 0x1p-60) {  // log(1 - e^-L) = log L - L/2 + ...
        const DD l = lnl - L * 0.5;
        return l.hi + l.lo;
    }
    if (L.hi < num::kLn2.hi) return std::log(-std::expm1(-L.hi) + std::exp(-L.hi) * L.lo);
    return std::isinf(L.hi) ? 0.0 : std::log1p(-num::exp_dd(-L));
}

// -log(1 - p) and -log p as double-doubles (0 < p < 1).
inline DD neglog1m(double p) {
    if (p > 0.5) return -num::log_dd(1.0 - p);  // 1 - p exact
    return -log1p_dd(DD{-p, 0.0});
}
inline DD neglog(double p) { return p > 0.5 ? neglog1m(1.0 - p) : -num::log_dd(p); }
// -log(1 - e^lq) for lq < 0
inline DD neglog1m_exp(double lq) {
    if (lq < -num::kLn2.hi) return -log1p_dd(DD{-std::exp(lq), 0.0});
    return -num::log_dd(-std::expm1(lq));
}

}  // namespace lifetime_detail

struct Pareto {
    static constexpr char name[] = "pareto";
    static constexpr std::array<const char*, 2> params{"b", "scale"};

    static bool valid(double b, double s) { return b > 0.0 && std::isfinite(b) && s > 0.0 && std::isfinite(s); }

    // log(x / s) for x >= s finite.
    static num::DD log_ratio(double x, double s) {
        if (s < 0x1p-700 && x < 0x1p-600) [[unlikely]] return log_ratio(x * 0x1p200, s * 0x1p200);  // (x - s may be subnormal: scaled, exactly)
        if (x <= 2.0 * s) return lifetime_detail::log1p_dd(num::ratio(x - s, s));  // x - s exact
        return weibull_detail::log_ratio(x, s);
    }
    static double pdf(double x, double b, double s) {
        if (!(x >= s) || std::isinf(x)) return 0.0;
        return num::exp_dd(log_density(x, b, s));
    }
    // log b - log s - (b + 1) log(x/s), (b + 1) log(x/s) as L + log(x/s).
    static num::DD log_density(double x, double b, double s) {
        const num::DD lr = log_ratio(x, s);
        if (std::isinf(lr.hi * b)) return {-kInf, 0.0};  // b log(x/s) beyond the range
        thread_local num::Memo memo;
        return memo.get(b, s, [&] { return num::log_dd(b) - num::log_dd(s); }) - lr * b - lr;
    }
    static double logpdf(double x, double b, double s) {
        if (!(x >= s) || std::isinf(x)) return -kInf;
        const num::DD l = log_density(x, b, s);
        return l.hi + l.lo;
    }
    static double tail(double x, double b, double s, bool upper, bool log) {
        if (!(x > s)) return upper ? (log ? 0.0 : 1.0) : (log ? -kInf : 0.0);
        if (std::isinf(x)) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
        const num::DD lr = log_ratio(x, s);
        const num::DD L = std::isinf(lr.hi * b) ? num::DD{kInf, 0.0} : lr * b;  // (the product's low part is NaN there)
        const num::DD lnl = L.hi < 0x1p-60 ? num::log_dd(b) + num::dd::log(lr) : num::DD{0.0, 0.0};
        return lifetime_detail::tail_exp(L, lnl, upper, log);
    }
    static double cdf(double x, double b, double s) { return tail(x, b, s, false, false); }
    static double ccdf(double x, double b, double s) { return tail(x, b, s, true, false); }
    static double logcdf(double x, double b, double s) { return tail(x, b, s, false, true); }
    static double logccdf(double x, double b, double s) { return tail(x, b, s, true, true); }

    // ---- chunk kernels --------------------------------------------------------------------
    //
    // S = (s/x)^b by vForce's pow (ckd::pow_dd on the double-double s/x): relative error
    // kPowDd whatever x. NaN outside x > s, b <= 2^30 and S >= 2^-999.
    static void power_chunk(const Chunk<2>& c, double* out) {
        double qh[kChunkSize], ql[kChunkSize];
        ckd::ratio(c.p[1], c.x, qh, ql, c.n);
        ckd::pow_dd(out, qh, ql, c.p[0], c.n);
        for (int i = 0; i < c.n; ++i) {
            const bool ok = c.x[i] > c.p[1][i] && c.p[0][i] <= 0x1p30 && qh[i] >= 0x1p-999 && out[i] >= 0x1p-999;
            out[i] = ok ? out[i] : kNaN;
        }
    }
    // log(x/s) for the chunk as the scalar definition's double-double, ckd::log_dd for its two
    // forms: log(1 + u), u = (x - s)/s, up to x = 2s (that needs u >= 2^-30: 1 + u keeps u to
    // 2^-106 absolute, 2^-76 of the log) and log(x/s) beyond. Relative error 2 kLq. NaN
    // outside x > s (1 + 2^-30) and the normal range.
    static void log_ratio_chunk(const Chunk<2>& c, double* lh, double* ll) {
        double yh[kChunkSize], yl[kChunkSize];
        for (int i = 0; i < c.n; ++i) lh[i] = c.x[i] <= 2.0 * c.p[1][i] ? c.x[i] - c.p[1][i] : c.x[i];
        ckd::ratio(lh, c.p[1], yh, yl, c.n);
        for (int i = 0; i < c.n; ++i) {
            if (c.x[i] <= 2.0 * c.p[1][i]) {
                const num::DD y = num::dd::two_sum(1.0, yh[i]) + yl[i];
                yh[i] = yh[i] >= 0x1p-30 ? y.hi : kNaN, yl[i] = y.lo;
            } else if (!(c.x[i] > c.p[1][i])) {
                yh[i] = kNaN;
            }
        }
        ckd::log_dd(yh, yl, lh, ll, c.n);
    }
    // The tails. Grade 0 from S: ccdf = S (kPowDd), cdf = 1 - S while S <= 3/4 (S's error
    // times S / (1 - S) <= 3, plus 1: 14.5) and above it, up to x = 2s, -expm1(-b log1p(u)) with
    // u = (x - s)/s by libm (u: 1 unit, x - s being exact; log1p: its condition <= 1 and 2;
    // the product; expm1: its condition <= 1 and 2: 6). Grade 1 from L = b log(x/s) in
    // double-double (relative error tau = 2 kLq) through weibull_detail::tail_chunk with t = L.
    static void tail_kernel(const Chunk<2>& c, double* out, bool* fast, weibull_detail::Tail kind, int grade) {
        using weibull_detail::Tail;
        if (grade == 0) {
            if (kind != Tail::kExp && kind != Tail::kOneExp) {
                for (int i = 0; i < c.n; ++i) fast[i] = false;
                return;
            }
            power_chunk(c, out);
            for (int i = 0; i < c.n; ++i) {
                if (kind == Tail::kExp) {
                    fast[i] = out[i] >= 0x1p-999;  // false for NaN
                } else if (out[i] <= 0.75) {
                    fast[i] = true;
                    out[i] = 1.0 - out[i];
                } else {
                    const double x = c.x[i], s = c.p[1][i];
                    fast[i] = out[i] == out[i] && x <= 2.0 * s;
                    out[i] = -std::expm1(-c.p[0][i] * std::log1p((x - s) / s));
                }
            }
            return;
        }
        double lh[kChunkSize], ll[kChunkSize], tau[kChunkSize], bound[kChunkSize];
        log_ratio_chunk(c, lh, ll);
        for (int i = 0; i < c.n; ++i) {
            const num::DD L = num::DD{lh[i], ll[i]} * c.p[0][i];
            const bool ok = L.hi >= 0x1p-999 && L.hi <= 0x1p999;
            lh[i] = L.hi, ll[i] = L.lo;
            tau[i] = ok ? 2.0 * ckd::kLq : kNaN;
        }
        weibull_detail::tail_chunk(kind, lh, ll, tau, out, bound, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = bound[i] <= kFastBudget;  // false for NaN
    }
    static void tail_run(const Chunk<2>& c, double* out, weibull_detail::Tail kind, bool upper, bool log) {
        bool fast[kChunkSize];
        tail_kernel(c, out, fast, kind, 0);
        ckd::retry(c, out, fast, [&](const Chunk<2>& s, double* o, bool* f) { tail_kernel(s, o, f, kind, 1); });
        ck::fallback(c.n, fast, out, [&](int i) { return tail(c.x[i], c.p[0][i], c.p[1][i], upper, log); });
    }
    static void cdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kOneExp, false, false); }
    static void ccdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kExp, true, false); }
    static void logcdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogOne, false, true); }
    static void logccdf_chunk(const Chunk<2>& c, double* out) { tail_run(c, out, weibull_detail::Tail::kLogExp, true, true); }
    // The density. Grade 0: b S / x in double (kPowDd + 2), logpdf its log by vForce
    // (kVfLog |value| more, against the mixed bound of T20). Grade 1: k - b log(x/s) - log(x/s)
    // in the scalar definition's double-doubles: absolute error 2 kLq (b + 1) log(x/s) and the
    // last bit; pdf adds exp's kVfExp + 1.
    static void density_kernel(const Chunk<2>& c, double* out, bool* fast, bool log, int grade) {
        if (grade == 0) {
            power_chunk(c, out);
            for (int i = 0; i < c.n; ++i) {
                const double m = c.p[0][i] * out[i];
                out[i] = m / c.x[i];
                fast[i] = m >= 0x1p-900 && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;
            }
            if (log) ck::log(out, out, c.n);
            for (int i = 0; i < c.n; ++i) {
                const double v = std::fabs(out[i]);
                fast[i] = fast[i] && (!log || ckd::kPowDd + 2.0 + ck::kVfLog * v <= kFastBudget * std::max(1.0, v));
            }
            return;
        }
        double lh[kChunkSize], ll[kChunkSize], kh[kChunkSize], kl[kChunkSize], bound[kChunkSize];
        log_ratio_chunk(c, lh, ll);
        ckd::constant(c, kh, kl, [&](int i) { return num::log_dd(c.p[0][i]) - num::log_dd(c.p[1][i]); });
        for (int i = 0; i < c.n; ++i) {
            const num::DD lr{lh[i], ll[i]};
            const num::DD l = num::DD{kh[i], kl[i]} - lr * c.p[0][i] - lr;
            const double abs = 2.0 * ckd::kLq * (c.p[0][i] + 1.0) * std::fabs(lr.hi) + 1.0;
            if (log) {
                out[i] = l.hi + l.lo;
                fast[i] = out[i] == out[i] && abs <= kFastBudget * std::max(1.0, std::fabs(out[i]));
            } else {
                lh[i] = l.hi, ll[i] = l.lo;
                bound[i] = abs + ck::kVfExp + 1.0;
            }
        }
        if (log) return;
        ckd::exp_dd(out, lh, ll, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = bound[i] <= kFastBudget && out[i] >= 0x1p-1000;
    }
    static void density_run(const Chunk<2>& c, double* out, bool log) {
        bool fast[kChunkSize];
        density_kernel(c, out, fast, log, 0);
        ckd::retry(c, out, fast, [&](const Chunk<2>& s, double* o, bool* f) { density_kernel(s, o, f, log, 1); });
        ck::fallback(c.n, fast, out, [&](int i) {
            return log ? logpdf(c.x[i], c.p[0][i], c.p[1][i]) : pdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void pdf_chunk(const Chunk<2>& c, double* out) { density_run(c, out, false); }
    static void logpdf_chunk(const Chunk<2>& c, double* out) { density_run(c, out, true); }
    // x = s q^(-1/b) by vForce's pow, q = 1 - p (icdf) or p (iccdf). The exponent's rounding
    // moves x by |log q| / b units, q's own (1 - p below p = 1/2) by 1 / b, and |log q| <=
    // (1 - q) / q; then pow and the product: ((1 - q) / q + 1) / b + kPow + 1.
    static void quantile_chunk(const Chunk<2>& c, double* out, bool upper) {
        double q[kChunkSize], e[kChunkSize];
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) {
            q[i] = upper ? c.x[i] : 1.0 - c.x[i];
            e[i] = -1.0 / c.p[0][i];
        }
        ck::pow(out, q, e, c.n);
        for (int i = 0; i < c.n; ++i) {
            out[i] *= c.p[1][i];
            const double bound = ((1.0 - q[i]) / q[i] + 1.0) / c.p[0][i] + ckd::kPow + 1.0;
            fast[i] = c.x[i] > 0.0 && c.x[i] < 1.0 && bound <= kFastBudget && out[i] <= 0x1p1000;  // false for NaN
        }
        ck::fallback(c.n, fast, out, [&](int i) {
            return upper ? iccdf(c.x[i], c.p[0][i], c.p[1][i]) : icdf(c.x[i], c.p[0][i], c.p[1][i]);
        });
    }
    static void icdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, false); }
    static void iccdf_chunk(const Chunk<2>& c, double* out) { quantile_chunk(c, out, true); }

    // s exp(L / b); s into the exponent when the power alone leaves the normal range.
    static double from_L(num::DD L, double b, double s) {
        if (std::isinf(L.hi)) return kInf;
        return from_e(num::div(L, b), s);
    }
    // s exp(e), e = L / b.
    static double from_e(num::DD e, double s) {
        if (std::isinf(e.hi)) return kInf;
        if (e.hi < 700.0) return s * num::exp_dd(e);
        return num::exp_dd(e + num::log_dd(s));
    }
    static double icdf(double p, double b, double s) {  // L = -log(1 - p)
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? s : kInf;
        return from_L(lifetime_detail::neglog1m(p), b, s);
    }
    static double iccdf(double p, double b, double s) {  // L = -log p
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? kInf : s;
        return from_L(lifetime_detail::neglog(p), b, s);
    }
    static double ilogcdf(double lp, double b, double s) {  // L = -log(1 - e^lp)
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? kInf : s;
        if (lp < -700.0) {  // L = e^lp may be subnormal: L / b = e^(lp - log b)
            return from_e(num::DD{num::exp_dd(num::DD{lp, 0.0} - num::log_dd(b)), 0.0}, s);
        }
        return from_L(lifetime_detail::neglog1m_exp(lp), b, s);
    }
    static double ilogccdf(double lp, double b, double s) {  // L = -lp
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? s : kInf;
        return from_L(num::DD{-lp, 0.0}, b, s);
    }

    // The moments by the rule of docs/api/stats.md §2: the mean +inf for b <= 1; the
    // variance +inf for 1 < b <= 2 and NaN below; the ratios +inf while the variance is
    // finite and their moment is not, NaN when the variance is not finite.
    static double mean(double b, double s) { return b > 1.0 ? s * (b / (b - 1.0)) : kInf; }
    static double variance(double b, double s) {
        if (!(b > 2.0)) return b > 1.0 ? kInf : kNaN;
        const double s2 = s * s;
        if (s2 >= 0x1p-900 && s2 <= 0x1p900 && b < 1e100) return s2 * (b / ((b - 1.0) * (b - 1.0) * (b - 2.0)));
        // s^2 or (b - 1)^2 leaves the range where the variance may not: (s / (b - 1))^2 b / (b - 2)
        const double r = s / (b - 1.0);
        return r * (r * (b / (b - 2.0)));
    }
    static double standard_deviation(double b, double s) {
        if (!(b > 2.0)) return b > 1.0 ? kInf : kNaN;
        return s * (std::sqrt(b / (b - 2.0)) / (b - 1.0));
    }
    static double skewness(double b, double) {  // 2 (1 + b) / (b - 3) sqrt((b - 2) / b)
        if (!(b > 3.0)) return b > 2.0 ? kInf : kNaN;
        if (b > 1e300) return 2.0 * ((1.0 + b) / (b - 3.0)) * std::sqrt((b - 2.0) / b);  // 2 (1 + b) overflows
        return 2.0 * (1.0 + b) / (b - 3.0) * std::sqrt((b - 2.0) / b);
    }
    static double kurtosis(double b, double) {  // 6 (b^3 + b^2 - 6b - 2) / (b (b - 3) (b - 4))
        if (!(b > 4.0)) return b > 2.0 ? kInf : kNaN;
        if (b > 1e50) return 6.0 * (1.0 + 1.0 / b) / (1.0 - 7.0 / b);  // b^3 overflows; the 1/b^2 terms are below 1e-100
        return 6.0 * (b * (b * (b + 1.0) - 6.0) - 2.0) / (b * (b - 3.0) * (b - 4.0));
    }
    static double median(double b, double s) { return from_L(num::kLn2, b, s); }  // s 2^(1/b)
    static double mode(double, double s) { return s; }
    static double entropy(double b, double s) {  // log(s / b) + 1/b + 1
        if (std::isinf(1.0 / b)) return kInf;
        const num::DD h = num::log_dd(s) - num::log_dd(b) + num::div(num::DD{1.0, 0.0}, b) + 1.0;
        return h.hi + h.lo;
    }
    static double support_lo(double, double s) { return s; }
    static double support_hi(double, double) { return kInf; }
};

struct Rayleigh {
    static constexpr char name[] = "rayleigh";
    static constexpr std::array<const char*, 1> params{"scale"};

    static bool valid(double s) { return s > 0.0 && std::isfinite(s); }

    // t = y^2 / 2, y = x / s.
    static num::DD t_of(num::DD y) {
        if (std::isinf(y.hi * y.hi)) return {kInf, 0.0};  // (the double-double square is NaN there)
        return y * y * 0.5;
    }
    static double pdf(double x, double s) {
        if (!(x > 0.0) || std::isinf(x)) return 0.0;
        const num::DD l = log_density(x, s);
        return std::isinf(l.hi) ? 0.0 : num::exp_dd(l);
    }
    // log y - log s - t
    static num::DD log_density(double x, double s) {
        const num::DD y = num::ratio(x, s);
        const num::DD t = t_of(y);
        if (std::isinf(t.hi)) return {-kInf, 0.0};
        const num::DD ly = y.hi >= 0x1p-960 ? num::dd::log(y) : num::log_dd(x) - num::log_dd(s);
        thread_local num::Memo memo;
        return ly - t - memo.get(s, 0.0, [&] { return num::log_dd(s); });
    }
    static double logpdf(double x, double s) {
        if (!(x > 0.0) || std::isinf(x)) return -kInf;
        const num::DD l = log_density(x, s);
        return l.hi + l.lo;
    }
    static double tail(double x, double s, bool upper, bool log) {
        if (!(x > 0.0)) return upper ? (log ? 0.0 : 1.0) : (log ? -kInf : 0.0);
        if (std::isinf(x)) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
        const num::DD y = num::ratio(x, s);
        const num::DD t = t_of(y);
        num::DD lnt{0.0, 0.0};
        if (!upper && log && t.hi < 0x1p-60)  // log t = 2 log y - log 2
            lnt = (y.hi >= 0x1p-960 ? num::dd::log(y) : num::log_dd(x) - num::log_dd(s)) * 2.0 - num::kLn2;
        return lifetime_detail::tail_exp(t, lnt, upper, log);
    }
    static double cdf(double x, double s) { return tail(x, s, false, false); }
    static double ccdf(double x, double s) { return tail(x, s, true, false); }
    static double logcdf(double x, double s) { return tail(x, s, false, true); }
    static double logccdf(double x, double s) { return tail(x, s, true, true); }

    // ---- chunk kernels: t = y^2 / 2 is the scalar definition's double-double (ckd::ratio and
    // its products: tau = 0), the tails weibull_detail::tail_chunk's. NaN (tau) where t leaves
    // [2^-999, 2^999].
    static void t_chunk(const Chunk<1>& c, double* th, double* tl, double* tau) {
        ckd::ratio(c.x, c.p[0], th, tl, c.n);
        for (int i = 0; i < c.n; ++i) {
            const num::DD t = t_of(num::DD{th[i], tl[i]});
            const bool ok = c.x[i] > 0.0 && t.hi >= 0x1p-999 && t.hi <= 0x1p999;
            th[i] = t.hi, tl[i] = t.lo;
            tau[i] = ok ? 0.0 : kNaN;
        }
    }
    static void tail_chunk(const Chunk<1>& c, double* out, weibull_detail::Tail kind, bool upper, bool log) {
        double th[kChunkSize], tl[kChunkSize], tau[kChunkSize];
        bool fast[kChunkSize];
        t_chunk(c, th, tl, tau);
        weibull_detail::tail_chunk(kind, th, tl, tau, out, tau, c.n);
        for (int i = 0; i < c.n; ++i) fast[i] = tau[i] <= kFastBudget;  // false for NaN
        ck::fallback(c.n, fast, out, [&](int i) { return tail(c.x[i], c.p[0][i], upper, log); });
    }
    static void cdf_chunk(const Chunk<1>& c, double* out) { tail_chunk(c, out, weibull_detail::Tail::kOneExp, false, false); }
    static void ccdf_chunk(const Chunk<1>& c, double* out) { tail_chunk(c, out, weibull_detail::Tail::kExp, true, false); }
    static void logcdf_chunk(const Chunk<1>& c, double* out) { tail_chunk(c, out, weibull_detail::Tail::kLogOne, false, true); }
    static void logccdf_chunk(const Chunk<1>& c, double* out) { tail_chunk(c, out, weibull_detail::Tail::kLogExp, true, true); }
    // (y / s) e^-t in double, e^-t from the double-double t: kVfExp + 1 and y's rounding, the
    // quotient and the product: kVfExp + 4. logpdf is its log by vForce (kVfLog |value| more:
    // inside the mixed bound of T20 for any value).
    static void density_chunk(const Chunk<1>& c, double* out, bool log) {
        double th[kChunkSize], tl[kChunkSize], tau[kChunkSize];
        bool fast[kChunkSize];
        t_chunk(c, th, tl, tau);
        for (int i = 0; i < c.n; ++i) th[i] = -th[i], tl[i] = -tl[i];
        ckd::exp_dd(out, th, tl, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double m = (c.x[i] / c.p[0][i]) / c.p[0][i];
            fast[i] = tau[i] == 0.0 && out[i] >= 0x1p-900 && m >= 0x1p-900 && m <= 0x1p900;
            out[i] *= m;
            fast[i] = fast[i] && out[i] >= 0x1p-1000 && out[i] <= 0x1p1000;
        }
        if (log) ck::log(out, out, c.n);
        ck::fallback(c.n, fast, out, [&](int i) { return log ? logpdf(c.x[i], c.p[0][i]) : pdf(c.x[i], c.p[0][i]); });
    }
    static void pdf_chunk(const Chunk<1>& c, double* out) { density_chunk(c, out, false); }
    static void logpdf_chunk(const Chunk<1>& c, double* out) { density_chunk(c, out, true); }
    // s sqrt(2t) in double from t with relative error tu: tu / 2, the square root and the
    // product: at most 5.75 (icdf: t = -log(1 - p) as Weibull::icdf_chunk's, tu = 7.5; iccdf:
    // t = -log p by vForce, tu = kVfLog).
    static void quantile_chunk(const Chunk<1>& c, double* out, bool upper) {
        bool fast[kChunkSize];
        for (int i = 0; i < c.n; ++i) out[i] = upper ? c.x[i] : 1.0 - c.x[i];
        ck::log(out, out, c.n);
        for (int i = 0; i < c.n; ++i) {
            const double p = c.x[i];
            const double t = !upper && p < 0.25 ? -std::log1p(-p) : -out[i];
            out[i] = c.p[0][i] * std::sqrt(2.0 * t);
            fast[i] = p >= 0x1p-1000 && p < 1.0;
        }
        ck::fallback(c.n, fast, out, [&](int i) { return upper ? iccdf(c.x[i], c.p[0][i]) : icdf(c.x[i], c.p[0][i]); });
    }
    static void icdf_chunk(const Chunk<1>& c, double* out) { quantile_chunk(c, out, false); }
    static void iccdf_chunk(const Chunk<1>& c, double* out) { quantile_chunk(c, out, true); }

    // s sqrt(2t), without overflowing 2t.
    static double from_t(num::DD t, double s) {
        if (std::isinf(t.hi)) return kInf;
        if (t.hi == 0.0) return 0.0;
        const double r = t.hi < 1e300 ? std::sqrt(2.0 * t.hi) : std::sqrt(2.0) * std::sqrt(t.hi);
        return s * (r + t.lo / r);  // sqrt(2 (hi + lo)) = r (1 + lo / (2 hi))
    }
    static double icdf(double p, double s) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? 0.0 : kInf;
        return from_t(lifetime_detail::neglog1m(p), s);
    }
    static double iccdf(double p, double s) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? kInf : 0.0;
        return from_t(lifetime_detail::neglog(p), s);
    }
    static double ilogcdf(double lp, double s) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? kInf : 0.0;
        if (lp < -40.0) {  // t = e^lp (1 + e^lp / 2 + ...), maybe below the doubles: x = s exp((lp + log 2) / 2)
            const num::DD l = (num::DD{lp, 0.5 * std::exp(lp)} + num::kLn2) * 0.5;
            if (l.hi < -690.0) return num::exp_dd(l + num::log_dd(s));  // the power alone is subnormal
            return s * num::exp_dd(l);
        }
        return from_t(lifetime_detail::neglog1m_exp(lp), s);
    }
    static double ilogccdf(double lp, double s) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0 || lp == -kInf) return lp == 0.0 ? 0.0 : kInf;
        return from_t(num::DD{-lp, 0.0}, s);
    }

    static double mean(double s) { return s * 1.2533141373155003; }                   // sqrt(pi / 2)
    static double variance(double s) { return s * s * 0.4292036732051034; }           // (4 - pi) / 2
    static double standard_deviation(double s) { return s * 0.6551363775620336; }
    static double skewness(double) { return 0.6311106578189372; }   // 2 sqrt(pi) (pi - 3) / (4 - pi)^(3/2)
    static double kurtosis(double) { return 0.24508930068763807; }  // -(6 pi^2 - 24 pi + 16) / (4 - pi)^2
    static double median(double s) { return s * 1.1774100225154747; }  // sqrt(2 log 2)
    static double mode(double s) { return s; }
    static double entropy(double s) {  // log s + 1 - log(2)/2 + euler/2
        const num::DD h = num::log_dd(s) + num::DD{0.9420342421707938, -1.406669164544682e-17};
        return h.hi + h.lo;
    }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double) { return kInf; }
};

struct InverseGamma {
    static constexpr char name[] = "inversegamma";
    static constexpr std::array<const char*, 2> params{"a", "scale"};

    static bool valid(double a, double s) { return a > 0.0 && std::isfinite(a) && s > 0.0 && std::isfinite(s); }

    // y = s / x; the density of X is f_Y(y) y / x = f_Y(y) y^2 / s.
    static num::DD log_density(double x, double a, double s) {
        const num::DD y = num::ratio(s, x);
        if (std::isinf(y.hi)) return {-kInf, 0.0};  // s / x overflows: e^-y is 0
        if (y.hi < 0x1p-960) {
            // y has lost bits (or is 0): (a + 1) log y - log Gamma(a) - log s from the logs
            const num::DD ly = num::log_dd(s) - num::log_dd(x);
            return ly * a + ly - (num::lgamma1p(a) - num::log_dd(a)) - num::log_dd(s);
        }
        const num::DD ly = num::dd::log(y);
        thread_local num::Memo memo;
        return gamma_detail::log_density_std(a, y) + ly * 2.0 - memo.get(s, 0.0, [&] { return num::log_dd(s); });
    }
    static bool y_ok(double x, double s) {  // y = s / x a finite positive double
        const double y = s / x;
        return std::isfinite(y) && y > 0.0;
    }
    // The double-double definitions (the kernel's fallback: the tails, huge shapes, the ends).
    static double pdf_dd(double x, double a, double s) {
        if (!(x > 0.0) || std::isinf(x)) return 0.0;
        const num::DD l = log_density(x, a, s);
        return std::isinf(l.hi) || std::isnan(l.hi) ? 0.0 : num::exp_dd(l);
    }
    static double logpdf_dd(double x, double a, double s) {
        if (!(x > 0.0) || std::isinf(x)) return -kInf;
        const num::DD l = log_density(x, a, s);
        return l.hi + l.lo;
    }
    // pdf and logpdf: the gamma's density kernel at y = s / x with k = a + 1 (dens::gamma_kernel,
    // stats/dist/gamma.hpp); the scalar forms are the kernel on one element.
    static void density_chunk(const Chunk<2>& c, double* out, bool log) {
        dens::gamma_kernel(c.n, c.x, c.p[0], c.p[1], true, log, out, [&](int i) {
            if (!(c.x[i] == c.x[i] && valid(c.p[0][i], c.p[1][i]))) return kNaN;
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
    // P(X <= x) = Q(a, s/x), P(X > x) = P(a, s/x).
    static double tail(double x, double a, double s, bool upper, bool log) {
        if (!(x > 0.0)) return upper ? (log ? 0.0 : 1.0) : (log ? -kInf : 0.0);
        if (std::isinf(x)) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
        return gamma_detail::ratio_tail(a, s, x, !upper, log);
    }
    static double cdf(double x, double a, double s) { return tail(x, a, s, false, false); }
    static double ccdf(double x, double a, double s) { return tail(x, a, s, true, false); }
    static double logcdf(double x, double a, double s) { return tail(x, a, s, false, true); }
    static double logccdf(double x, double a, double s) { return tail(x, a, s, true, true); }

    // x = s / y. The gamma's inverses give y (scale 1); a y outside the normal range
    // comes back with 1/s folded into its exponent (x = 1 / ((1/s) y)).
    template <class Inv>
    static double invert(double s, Inv&& inv) {
        const double y = inv(1.0);
        if (y >= 0x1p-1000 && y <= 0x1p1000) return s / y;
        if (y != y) return kNaN;
        // y is outside [2^-1000, 2^1000] (maybe 0 or inf as a double): a factor k folded into
        // its exponent so that w = k y is a normal double, x = s k / w; 1/s first (x = 1 / w),
        // then a power of two
        constexpr double kMax = std::numeric_limits<double>::max();
        const double k1 = 1.0 / s;
        if (k1 >= 0x1p-1000 && k1 <= 0x1p1000) {
            const double w = inv(k1);
            if (w >= 0x1p-1022 && w <= kMax) return 1.0 / w;
        }
        const double k2 = y < 1.0 ? 0x1p1000 : 0x1p-1000;
        const double w = inv(k2);
        if (w >= 0x1p-1022 && w <= kMax) {
            const double r = (s / w) * k2;  // or the other order, where this one leaves the range on the way
            return (r >= 0x1p-1022 && r <= kMax) ? r : (s * k2) / w;
        }
        // beyond both: x = s / y with y outside 2^+-2000 is beyond the doubles unless s is as far the other way
        if (y < 1.0) return s >= 0x1p-900 ? kInf : kNaN;
        return s <= 0x1p900 ? 0.0 : kNaN;
    }
    // P(X <= x) = p: Q(a, y) = p (y's upper inverse); the smaller of p and 1 - p is solved.
    static double icdf(double p, double a, double s) {
        if (!(p >= 0.0 && p <= 1.0) || a > gamma_detail::kMaxShape) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? 0.0 : kInf;
        if (p > 0.5) return invert(s, [&](double k) { return gamma_detail::lower_inverse_checked(a, k, 1.0 - p, num::log_dd(1.0 - p)); });
        return invert(s, [&](double k) { return gamma_detail::upper_inverse_checked(a, k, p, num::log_dd(p)); });
    }
    static double iccdf(double p, double a, double s) {
        if (!(p >= 0.0 && p <= 1.0) || a > gamma_detail::kMaxShape) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? kInf : 0.0;
        if (p > 0.5) return invert(s, [&](double k) { return gamma_detail::upper_inverse_checked(a, k, 1.0 - p, num::log_dd(1.0 - p)); });
        return invert(s, [&](double k) { return gamma_detail::lower_inverse_checked(a, k, p, num::log_dd(p)); });
    }
    static double from_log(double lp, double a, double s, bool upper) {
        if (!(lp <= 0.0) || a > gamma_detail::kMaxShape) return kNaN;
        if (lp == 0.0) return upper ? 0.0 : kInf;
        if (lp == -kInf) return upper ? kInf : 0.0;
        // X's lower tail is Y's upper tail
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return invert(s, [&](double k) {
                return upper ? gamma_detail::upper_inverse_checked(a, k, q, num::log_dd(q))
                             : gamma_detail::lower_inverse_checked(a, k, q, num::log_dd(q));
            });
        }
        const double p = std::exp(lp);
        return invert(s, [&](double k) {
            return upper ? gamma_detail::lower_inverse_checked(a, k, p, num::DD{lp, 0.0})
                         : gamma_detail::upper_inverse_checked(a, k, p, num::DD{lp, 0.0});
        });
    }
    static double ilogcdf(double lp, double a, double s) { return from_log(lp, a, s, false); }
    static double ilogccdf(double lp, double a, double s) { return from_log(lp, a, s, true); }

    // The moments by the rule of docs/api/stats.md §2 (a - k exact near the thresholds).
    static double mean(double a, double s) { return a > 1.0 ? s / (a - 1.0) : kInf; }
    static double variance(double a, double s) {
        if (!(a > 2.0)) return a > 1.0 ? kInf : kNaN;
        const double m = s / (a - 1.0);
        return m * m / (a - 2.0);
    }
    static double standard_deviation(double a, double s) {
        if (!(a > 2.0)) return a > 1.0 ? kInf : kNaN;
        return s / (a - 1.0) / std::sqrt(a - 2.0);
    }
    static double skewness(double a, double) {  // 4 sqrt(a - 2) / (a - 3)
        if (!(a > 3.0)) return a > 2.0 ? kInf : kNaN;
        return 4.0 * std::sqrt(a - 2.0) / (a - 3.0);
    }
    static double kurtosis(double a, double) {  // 6 (5a - 11) / ((a - 3)(a - 4))
        if (!(a > 4.0)) return a > 2.0 ? kInf : kNaN;
        return 6.0 * (5.0 * a - 11.0) / ((a - 3.0) * (a - 4.0));
    }
    static double mode(double a, double s) { return s / (a + 1.0); }
    // The gamma's entropy + log s - 2 psi(a) (E log |dx/dy| = log s - 2 E log Y).
    static double entropy(double a, double s) {
        const num::DD h = gamma_detail::entropy_std(a) + num::log_dd(s) - 2.0 * special::Digamma::f(a);
        return h.hi + h.lo;
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
