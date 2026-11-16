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
        const num::DD L = lr * b;
        const num::DD lnl = L.hi < 0x1p-60 ? num::log_dd(b) + num::dd::log(lr) : num::DD{0.0, 0.0};
        return lifetime_detail::tail_exp(L, lnl, upper, log);
    }
    static double cdf(double x, double b, double s) { return tail(x, b, s, false, false); }
    static double ccdf(double x, double b, double s) { return tail(x, b, s, true, false); }
    static double logcdf(double x, double b, double s) { return tail(x, b, s, false, true); }
    static double logccdf(double x, double b, double s) { return tail(x, b, s, true, true); }

    // s exp(L / b); s into the exponent when the power alone leaves the normal range.
    static double from_L(num::DD L, double b, double s) {
        if (std::isinf(L.hi)) return kInf;
        const num::DD e = num::div(L, b);
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
        return s * s * (b / ((b - 1.0) * (b - 1.0) * (b - 2.0)));
    }
    static double standard_deviation(double b, double s) {
        if (!(b > 2.0)) return b > 1.0 ? kInf : kNaN;
        return s * (std::sqrt(b / (b - 2.0)) / (b - 1.0));
    }
    static double skewness(double b, double) {  // 2 (1 + b) / (b - 3) sqrt((b - 2) / b)
        if (!(b > 3.0)) return b > 2.0 ? kInf : kNaN;
        return 2.0 * (1.0 + b) / (b - 3.0) * std::sqrt((b - 2.0) / b);
    }
    static double kurtosis(double b, double) {  // 6 (b^3 + b^2 - 6b - 2) / (b (b - 3) (b - 4))
        if (!(b > 4.0)) return b > 2.0 ? kInf : kNaN;
        return 6.0 * (b * (b * (b + 1.0) - 6.0) - 2.0) / (b * (b - 3.0) * (b - 4.0));
    }
    static double median(double b, double s) { return from_L(num::kLn2, b, s); }  // s 2^(1/b)
    static double mode(double, double s) { return s; }
    static double entropy(double b, double s) {  // log(s / b) + 1/b + 1
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
    static num::DD t_of(num::DD y) { return y * y * 0.5; }
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
        const num::DD ly = y.hi >= 0x1p-1000 ? num::dd::log(y) : num::log_dd(x) - num::log_dd(s);
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
            lnt = (y.hi >= 0x1p-1000 ? num::dd::log(y) : num::log_dd(x) - num::log_dd(s)) * 2.0 - num::kLn2;
        return lifetime_detail::tail_exp(t, lnt, upper, log);
    }
    static double cdf(double x, double s) { return tail(x, s, false, false); }
    static double ccdf(double x, double s) { return tail(x, s, true, false); }
    static double logcdf(double x, double s) { return tail(x, s, false, true); }
    static double logccdf(double x, double s) { return tail(x, s, true, true); }

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
        const num::DD ly = y.hi >= 0x1p-1000 && std::isfinite(y.hi) ? num::dd::log(y) : num::log_dd(s) - num::log_dd(x);
        thread_local num::Memo memo;
        return gamma_detail::log_density_std(a, y) + ly * 2.0 - memo.get(s, 0.0, [&] { return num::log_dd(s); });
    }
    static bool y_ok(double x, double s) {  // y = s / x a finite positive double
        const double y = s / x;
        return std::isfinite(y) && y > 0.0;
    }
    static double pdf(double x, double a, double s) {
        if (!(x > 0.0) || std::isinf(x) || !y_ok(x, s)) return 0.0;
        const num::DD l = log_density(x, a, s);
        return std::isinf(l.hi) || std::isnan(l.hi) ? 0.0 : num::exp_dd(l);
    }
    static double logpdf(double x, double a, double s) {
        if (!(x > 0.0) || std::isinf(x) || !y_ok(x, s)) return -kInf;
        const num::DD l = log_density(x, a, s);
        return l.hi + l.lo;
    }
    // P(X <= x) = Q(a, s/x), P(X > x) = P(a, s/x).
    static double tail(double x, double a, double s, bool upper, bool log) {
        if (!(x > 0.0)) return upper ? (log ? 0.0 : 1.0) : (log ? -kInf : 0.0);
        if (std::isinf(x)) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
        const num::DD y = num::ratio(s, x);
        if (std::isinf(y.hi)) return upper ? (log ? 0.0 : 1.0) : (log ? -kInf : 0.0);  // y = inf: X ~ 0
        if (y.hi == 0.0) return upper ? (log ? -kInf : 0.0) : (log ? 0.0 : 1.0);
        return log ? gamma_detail::log_tail(a, y, !upper) : gamma_detail::tail(a, y, !upper);
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
        if (y == 0.0) return kInf;
        if (std::isinf(y)) return 0.0;
        return 1.0 / inv(1.0 / s);
    }
    // P(X <= x) = p: Q(a, y) = p (y's upper inverse); the smaller of p and 1 - p is solved.
    static double icdf(double p, double a, double s) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? 0.0 : kInf;
        if (p > 0.5) return invert(s, [&](double k) { return gamma_detail::lower_inverse(a, k, 1.0 - p, num::log_dd(1.0 - p)); });
        return invert(s, [&](double k) { return gamma_detail::upper_inverse(a, k, p, num::log_dd(p)); });
    }
    static double iccdf(double p, double a, double s) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (p == 0.0 || p == 1.0) return p == 0.0 ? kInf : 0.0;
        if (p > 0.5) return invert(s, [&](double k) { return gamma_detail::upper_inverse(a, k, 1.0 - p, num::log_dd(1.0 - p)); });
        return invert(s, [&](double k) { return gamma_detail::lower_inverse(a, k, p, num::log_dd(p)); });
    }
    static double from_log(double lp, double a, double s, bool upper) {
        if (!(lp <= 0.0)) return kNaN;
        if (lp == 0.0) return upper ? 0.0 : kInf;
        if (lp == -kInf) return upper ? kInf : 0.0;
        // X's lower tail is Y's upper tail
        if (lp > -num::kLn2.hi) {
            const double q = -std::expm1(lp);
            return invert(s, [&](double k) {
                return upper ? gamma_detail::upper_inverse(a, k, q, num::log_dd(q))
                             : gamma_detail::lower_inverse(a, k, q, num::log_dd(q));
            });
        }
        const double p = std::exp(lp);
        return invert(s, [&](double k) {
            return upper ? gamma_detail::lower_inverse(a, k, p, num::DD{lp, 0.0})
                         : gamma_detail::upper_inverse(a, k, p, num::DD{lp, 0.0});
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
