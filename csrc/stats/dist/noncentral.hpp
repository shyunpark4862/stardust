// The noncentral chi-square, Student t and F distributions (docs/api/stats.md §2.1,
// docs/design/stats.md T26). nc = 0 is the central distribution exactly.
//
//   * cdfs and inverses: sd.special's chndtr, nctdtr, ncfdtr and their complements (both
//     tails summed directly, measured to probability 1e-300: the guaranteed region); the
//     upper-tail inverses solve on the given upper probability (1 - q is not formed). The
//     log forms and the inverses take the tails as m 2^e (special's *_tail, the sums before
//     their rounding to a double), so a tail far below the double range keeps its value and
//     a log target e^lp is never formed (it underflows below lp = -745).
//   * densities: chi-square and F as Poisson mixtures of gamma and beta densities, every
//     term positive, summed from the largest term outwards in logs (double-double);
//     the t as the integral f(t) = E[S phi(t S - nc)], S = sqrt(V / df), whose integrand is
//     positive everywhere (the series' two families cancel in the tail away from nc),
//     by the trapezoid rule on a sinh map about its peak, as sd.special's nct_opposite.
//   * moments: chi-square in closed form; F and t as products of independent factors,
//     F = (X / dfn)(dfd / V) and T = (Z + nc) R, R = sqrt(df / V), whose central moments
//     expand into positive terms; R's moment ratios through D_k = log E[R^k] - k log E[R]
//     by their 1/df series (coeffs/nct_moments.hpp), the linear terms cancelled exactly.
//   * mode: the root of the density's derivative (1e-14, T26); entropy: -int f log f by the
//     trapezoid rule, halving the step to convergence (its accuracy documented as loose).
#pragma once

#include "special/ops/distr.hpp"
#include "special/ops/noncentral.hpp"
#include "stats/coeffs/nct_moments.hpp"
#include "stats/discrete.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace sd::stats {

namespace ncd {

using num::DD;
namespace dd = num::dd;

// log sum_j t_j for a positive, log-concave series given log t_j (double-double, costly)
// and t_(j+1) / t_j (cheap): from the largest term j* (found from the estimate j0 by
// the ratio) outwards, the ratios resynchronized every 8 terms. Also E[j] under the
// weights t_j / sum (for the modes).
struct Series {
    DD log;
    double mean_j;
};
template <class LogTerm, class Ratio>
Series log_series(double j0, double jmin, LogTerm&& lt, Ratio&& ratio) {
    double j = std::max(jmin, std::floor(j0));
    for (int i = 0; i < 64 && ratio(j) > 1.0; ++i) j += 1.0;
    for (int i = 0; i < 64 && j > jmin && ratio(j - 1.0) < 1.0; ++i) j -= 1.0;
    const DD l0 = lt(j);
    disc::KahanSum sum, sumj;
    sum.add(1.0);
    sumj.add(j);
    double t = 1.0;
    for (int i = 1;; ++i) {  // up
        const double k = j + i;
        t *= ratio(k - 1.0);
        if (i % 8 == 0) {
            const DD d = lt(k) - l0;
            t = std::exp(d.hi + d.lo);
        }
        sum.add(t);
        sumj.add(t * k);
        if (t <= sum.s * 0x1p-60 || i > 100000000) break;
    }
    t = 1.0;
    for (int i = 1; j - i >= jmin; ++i) {  // down
        const double k = j - i;
        t /= ratio(k);
        if (i % 8 == 0) {
            const DD d = lt(k) - l0;
            t = std::exp(d.hi + d.lo);
        }
        sum.add(t);
        sumj.add(t * k);
        if (t <= sum.s * 0x1p-60) break;
    }
    return {l0 + std::log(sum.value()), sumj.value() / sum.value()};
}

using special::Scaled;

// The tail target of an inverse: the side whose probability is small, and that probability
// as m 2^e (the log forms' targets e^lp reach e^-1e10).
struct Target {
    bool upper;  // the side solved
    Scaled p;
};
// icdf (upper: iccdf) at p, 0 < p < 1: the smaller side (1 - p exact for p >= 1/2)
inline Target target_p(double p, bool upper) {
    const bool small = p <= 0.5;
    return {upper == small, special::scaled(small ? p : 1.0 - p)};
}
// ilogcdf (upper: ilogccdf) at lp < 0: e^lp itself below 1/2, else the other side 1 - e^lp
inline Target target_log(double lp, bool upper) {
    if (lp > -num::kLn2.hi) return {!upper, special::scaled(-std::expm1(lp))};
    return {upper, special::nc_detail::exp_wide(lp)};
}

// The x (positive: on a log scale; else real, on asinh) whose tail (tail(x, upper) as
// m 2^e: the sd.special sums before their rounding to a double) is the target, by
// sd.special's bracketing and zeroin (special/ops/noncentral.hpp) on log(tail / target):
// a ratio of m 2^e values, so a target far below the double range (or a subnormal one, of
// few bits as a double result) is solved as 0.3 is.
template <class Tail>
double invert(const Tail& tail, Target tg, double x0, bool positive) {
    auto g = [&](double x) {
        const double l = special::log_abs(tail(x, tg.upper) / tg.p);
        return tg.upper ? -l : l;  // increasing in x
    };
    namespace nd = special::nc_detail;
    if (positive) {
        if (!(x0 > 0.0 && x0 < kInf)) x0 = 1.0;
        return nd::solve(g, x0, [](double x) { return std::log(x); }, [](double u) { return std::exp(u); }, 0.125, 0.0,
                         kInf);
    }
    if (!std::isfinite(x0)) x0 = 0.0;
    return nd::solve(g, x0, [](double t) { return std::asinh(t); }, [](double u) { return std::sinh(u); }, 0.125, -kInf,
                     kInf);
}

// log P: the asked tail's log from its m 2^e value where that is at most 1/2 (its
// exponent kept: a tail of e^-1e4 is no 0), else log1p of minus the other side.
template <class Tail>
double log_tail(const Tail& tail, bool upper) {
    const Scaled v = tail(upper);
    if (special::value(v) <= 0.5) return v.m == 0.0 ? -kInf : special::log_abs(v);
    return std::log1p(-special::value(tail(!upper)));
}

// -int f log f dx by the trapezoid rule in u, x = to_x(u) (dx = jac(u) du), the step
// halved until two estimates agree to 2^-50 (logpdf: a double-double log density).
template <class LogPdf, class ToX, class Jac>
double entropy(LogPdf&& logpdf, ToX&& to_x, Jac&& jac) {
    auto g = [&](double u) {
        const double x = to_x(u);
        const DD l = logpdf(x);
        const double lv = l.hi + l.lo;
        if (!std::isfinite(lv)) return 0.0;
        return -num::exp_dd(l) * lv * jac(u);
    };
    auto sweep = [&](double h, double offset) {  // sum over u = offset + k h, k in Z
        disc::KahanSum s;
        double mag = 0.0;
        for (int dir = -1; dir <= 1; dir += 2) {
            for (int k = (dir < 0 ? 0 : 1); k < 1000000; ++k) {
                const double v = g(offset + dir * k * h);
                s.add(v);
                mag = std::max(mag, std::fabs(v));
                if (std::fabs(v) <= 1e-22 * mag && k > 16) break;
            }
        }
        return s.value();
    };
    double h = 0.5;
    double total = sweep(h, 0.0);  // sum at the grid points
    double est = total * h;
    for (int it = 0; it < 12; ++it) {
        total += sweep(h, 0.5 * h);  // the midpoints
        h *= 0.5;
        const double next = total * h;
        if (std::fabs(next - est) <= 0x1p-50 * std::fabs(next) && it > 1) return next;
        est = next;
    }
    return est;
}

}  // namespace ncd

// ---------------------------------------------------------------- chi-square

struct NoncentralChiSquared {
    static constexpr char name[] = "noncentralchisquared";
    static constexpr std::array<const char*, 2> params{"df", "nc"};

    static bool valid(double df, double nc) { return df > 0.0 && std::isfinite(df) && nc >= 0.0 && std::isfinite(nc); }

    // f(x) = sum_j Pois(j; nc/2) chi2(x; df + 2j), chi2(x; k) = g(x/2; k/2) / 2 (g the standard
    // gamma density, gamma_detail::log_density_std): every term positive. `shift` lowers
    // the gamma shapes by `shift` (the mode's f_{df-2}). The shape a + j is a double-double
    // (rounded, it moved the density by s eps |log y - psi(s)|: 2.4e-13 at df = 7e3): the
    // density at its double part, moved by the rest to first order. Below x = 2^-1020, x/2
    // would round in the subnormals (1e-5 of the density at x = 4e-319): the terms from
    // log y = log x - log 2 directly, (s - 1) log y - y - log Gamma(s).
    static ncd::Series series(double x, double df, double nc, double shift = 0.0) {
        const double mu = 0.5 * nc, y = 0.5 * x, a = 0.5 * df - shift;
        const num::DD yd{y, 0.0};
        const bool tiny = x < 0x1p-1020;
        const num::DD ly = num::log_dd(x) - num::kLn2;
        auto lt = [&](double j) {
            const num::DD sh = num::dd::two_sum(a, j);
            num::DD l = gamma_detail::log_density_std(j + 1.0, num::DD{mu, 0.0}) - num::kLn2;
            if (tiny) return l + (sh - 1.0) * ly - y - special::nc_detail::lgamma_dd(sh);
            l = l + gamma_detail::log_density_std(sh.hi, yd);
            if (sh.lo != 0.0) l = l + sh.lo * (ly.hi - special::Digamma::f(sh.hi));
            return l;
        };
        auto ratio = [&](double j) { return mu * y / ((j + 1.0) * (a + j)); };  // t_(j+1) / t_j
        const double jmin = a > 0.0 ? 0.0 : 1.0;  // shape a + j > 0
        const double b = a + 1.0;  // the peak: (j + 1)(j + a) = mu y
        const double j0 = 0.5 * (-b + std::sqrt((a - 1.0) * (a - 1.0) + 4.0 * mu * y));
        return ncd::log_series(j0, jmin, lt, ratio);
    }
    // the density at x = 0: df < 2 inf, df = 2 e^(-nc/2) / 2, df > 2 zero
    static double density0(double df, double nc) { return df < 2.0 ? kInf : (df == 2.0 ? 0.5 * std::exp(-0.5 * nc) : 0.0); }
    static double pdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::pdf(x, df);
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return density0(df, nc);
        return num::exp_dd(series(x, df, nc).log);
    }
    static double logpdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::logpdf(x, df);
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return std::log(density0(df, nc));
        const num::DD l = series(x, df, nc).log;
        return l.hi + l.lo;
    }
    static double cdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::cdf(x, df);
        return x <= 0.0 ? 0.0 : special::Chndtr::f(df, nc, x);
    }
    static double ccdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::ccdf(x, df);
        return x <= 0.0 ? 1.0 : special::Chndtrc::f(df, nc, x);
    }
    // the tail as m 2^e (sd.special's chndtr sum before its rounding: the log forms and
    // inverses reach tails far below the double range)
    static special::Scaled tail(double x, double df, double nc, bool upper) {
        return special::nc_detail::chndtr_tail(df, nc, x, upper);
    }
    static double logcdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::logcdf(x, df);
        if (x <= 0.0) return -kInf;
        return ncd::log_tail([&](bool u) { return tail(x, df, nc, u); }, false);
    }
    static double logccdf(double x, double df, double nc) {
        if (nc == 0.0) return ChiSquared::logccdf(x, df);
        if (x <= 0.0) return 0.0;
        return ncd::log_tail([&](bool u) { return tail(x, df, nc, u); }, true);
    }
    static double inv(double p, double df, double nc, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (nc == 0.0) return upper ? ChiSquared::iccdf(p, df) : ChiSquared::icdf(p, df);
        if (p == 0.0) return upper ? kInf : 0.0;
        if (p == 1.0) return upper ? 0.0 : kInf;
        const double x0 = special::nc_detail::ncx2_start(df, nc, upper ? 1.0 - p : p);
        return ncd::invert([&](double x, bool u) { return tail(x, df, nc, u); }, ncd::target_p(p, upper), x0, true);
    }
    static double icdf(double p, double df, double nc) { return inv(p, df, nc, false); }
    static double iccdf(double p, double df, double nc) { return inv(p, df, nc, true); }
    // log p itself as the target (e^lp is not formed: it underflows below lp = -745)
    static double from_log(double lp, double df, double nc, bool upper) {
        if (nc == 0.0) return upper ? ChiSquared::ilogccdf(lp, df) : ChiSquared::ilogcdf(lp, df);
        if (!(lp <= 0.0)) return kNaN;
        if (lp == -kInf) return inv(0.0, df, nc, upper);
        const ncd::Target tg = ncd::target_log(lp, upper);
        if (tg.p.m == 0.0) return inv(1.0, df, nc, upper);  // 1 - e^lp rounds to 0: p = 1
        // start: Patnaik's c chi-square(k) (ncx2_start) at the log target
        const double c = (df + 2.0 * nc) / (df + nc), k = (df + nc) / c;
        const double x0 = c * (upper ? ChiSquared::ilogccdf(lp, k) : ChiSquared::ilogcdf(lp, k));
        return ncd::invert([&](double x, bool u) { return tail(x, df, nc, u); }, tg, x0, true);
    }
    static double ilogcdf(double lp, double df, double nc) { return from_log(lp, df, nc, false); }
    static double ilogccdf(double lp, double df, double nc) { return from_log(lp, df, nc, true); }

    static double mean(double df, double nc) { return df + nc; }
    static double variance(double df, double nc) { return 2.0 * (df + 2.0 * nc); }
    static double standard_deviation(double df, double nc) { return std::sqrt(2.0 * (df + 2.0 * nc)); }
    static double skewness(double df, double nc) {  // 2^(3/2) (df + 3 nc) / (df + 2 nc)^(3/2)
        const double v = df + 2.0 * nc;
        return 2.8284271247461903 * (df + 3.0 * nc) / (v * std::sqrt(v));
    }
    static double kurtosis(double df, double nc) {  // 12 (df + 4 nc) / (df + 2 nc)^2
        const double v = df + 2.0 * nc;
        return 12.0 * (df + 4.0 * nc) / (v * v);
    }
    // f' = (f_(df-2) - f_df) / 2, so the mode solves y = A(y) - 1, A = E[shape] under the
    // weights of f_df's terms at y = x/2 (each term's log-derivative is (a_j - 1)/y - 1):
    // phi(y) = A(y) - 1 - y falls from positive to negative. df < 2: the density is
    // infinite at 0 (T21: that point); df = 2 with nc <= 2 too has its peak at 0.
    static double mode(double df, double nc) {
        if (nc == 0.0) return ChiSquared::mode(df);
        if (df < 2.0) return 0.0;
        auto phi = [&](double x) {
            const ncd::Series s = series(x, df, nc);
            return 2.0 * ((0.5 * df - 1.0) + s.mean_j) - x;  // in x = 2y (df/2 - 1 exact)
        };
        const double lo = 1e-300;
        if (!(phi(lo) > 0.0)) return 0.0;
        double x0 = std::max(df + nc - 2.0, 1.0);
        auto g = [&](double x) { return -phi(x); };  // increasing
        return special::nc_detail::solve(g, x0, [](double x) { return std::log(x); },
                                         [](double u) { return std::exp(u); }, 0.125, 0.0, kInf);
    }
    static double entropy(double df, double nc) {
        if (nc == 0.0) return ChiSquared::entropy(df);
        const double m = std::max(df + nc, 1e-3);  // x = m e^u
        return ncd::entropy([&](double x) { return x > 0.0 && std::isfinite(x) ? series(x, df, nc).log : num::DD{-kInf, 0.0}; },
                            [&](double u) { return m * std::exp(u); }, [&](double u) { return m * std::exp(u); });
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }
};

// ---------------------------------------------------------------- F

namespace ncf_detail {

using num::DD;

// The central moments 2..4 of P = bX + aY + XY for independent centred X, Y with central
// moments mx[i], my[i] (mx[0] = 1, mx[1] = 0), a, b > 0: the expansion of P^k into
// X^i Y^j, each term c_ij mx[i] my[j]; all positive here (a, b > 0, odd moments >= 0).
struct Central {
    double m2, m3, m4;
};
inline Central product_central(double a, double b, const double mx[5], const double my[5]) {
    // P^k as coefficients c[i][j] of X^i Y^j, P = b X + a Y + X Y
    double c[5][5] = {{1.0}}, n[5][5];
    double out[5] = {0.0};
    for (int k = 1; k <= 4; ++k) {
        for (auto& r : n)
            for (double& v : r) v = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                if (c[i][j] == 0.0) continue;
                n[i + 1][j] += c[i][j] * b;
                n[i][j + 1] += c[i][j] * a;
                n[i + 1][j + 1] += c[i][j];
            }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) c[i][j] = n[i][j];
        double s = 0.0;
        for (int i = 0; i <= 4; ++i)
            for (int j = 0; j <= 4; ++j)
                if (c[i][j] != 0.0) s += c[i][j] * mx[i] * my[j];
        out[k] = s;
    }
    return {out[2], out[3], out[4]};
}

}  // namespace ncf_detail

struct NoncentralF {
    static constexpr char name[] = "noncentralf";
    static constexpr std::array<const char*, 3> params{"dfn", "dfd", "nc"};

    static bool valid(double dfn, double dfd, double nc) {
        return dfn > 0.0 && std::isfinite(dfn) && dfd > 0.0 && std::isfinite(dfd) && nc >= 0.0 && std::isfinite(nc);
    }

    // f(x) = sum_j Pois(j; nc/2) beta(w; dfn/2 + j, dfd/2) dw/dx, w = dfn x / (dfn x + dfd):
    // log term = log Pois + log(w^a v^b / B(a, b)) - log x (beta_detail::log_prefix; dw/dx =
    // w v / x). The shape a0 + j a double-double, as the chi-square's (d log prefix / da =
    // log w - psi(a) + psi(a + b)); below x = 2^-1020, where dfn x rounds in the subnormals
    // (2e-6 of the density at x = 5e-319), the terms from log w = log dfn + log x -
    // log(dfn x + dfd) and log v = -log1p(dfn x / dfd) directly, log B by lgamma_dd.
    static ncd::Series series(double x, double dfn, double dfd, double nc) {
        const special::distr_detail::Arg g = special::distr_detail::f_arg(dfn, dfd, x);
        const double mu = 0.5 * nc, a0 = 0.5 * dfn, b = 0.5 * dfd, w = g.x.hi;
        const num::DD lx = num::log_dd(x);
        const bool tiny = x < 0x1p-1020;
        const num::DD lw = num::log_dd(dfn) + lx - num::log_dd(dfn * x + dfd);
        const double lv = -std::log1p(dfn * x / dfd), lw1 = lw.hi;
        namespace nd = special::nc_detail;
        auto lt = [&](double j) {
            const num::DD sh = num::dd::two_sum(a0, j);
            const num::DD lp = gamma_detail::log_density_std(j + 1.0, num::DD{mu, 0.0}) - lx;
            if (tiny) {
                const num::DD lbeta = nd::lgamma_dd(sh) + nd::lgamma_dd(num::DD{b, 0.0}) - nd::lgamma_dd(sh + b);
                return lp + lw * sh + lv * b - lbeta;
            }
            num::DD l = lp + beta_detail::log_prefix(sh.hi, b, g.x, g.y);
            if (sh.lo != 0.0)
                l = l + sh.lo * (lw1 - special::Digamma::f(sh.hi) + special::Digamma::f(sh.hi + b));
            return l;
        };
        auto ratio = [&](double j) { return mu * w * (a0 + j + b) / ((j + 1.0) * (a0 + j)); };
        // the peak: (j + 1)(j + a0) = mu w (j + a0 + b)
        const double B = a0 + 1.0 - mu * w, C = a0 - mu * w * (a0 + b);
        const double j0 = 0.5 * (-B + std::sqrt(std::max(0.0, B * B - 4.0 * C)));
        return ncd::log_series(j0, 0.0, lt, ratio);
    }
    // the density at 0: dfn < 2 inf, dfn = 2 e^(-nc/2) (the j = 0 term alone), dfn > 2 zero
    static double density0(double dfn, double nc) { return dfn < 2.0 ? kInf : (dfn == 2.0 ? std::exp(-0.5 * nc) : 0.0); }
    static double pdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::pdf(x, dfn, dfd);
        if (x < 0.0 || std::isinf(x)) return 0.0;
        if (x == 0.0) return density0(dfn, nc);
        return num::exp_dd(series(x, dfn, dfd, nc).log);
    }
    static double logpdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::logpdf(x, dfn, dfd);
        if (x < 0.0 || std::isinf(x)) return -kInf;
        if (x == 0.0) return dfn == 2.0 ? -0.5 * nc : std::log(density0(dfn, nc));
        const num::DD l = series(x, dfn, dfd, nc).log;
        return l.hi + l.lo;
    }
    static double cdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::cdf(x, dfn, dfd);
        return x <= 0.0 ? 0.0 : special::Ncfdtr::f(dfn, dfd, nc, x);
    }
    static double ccdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::ccdf(x, dfn, dfd);
        return x <= 0.0 ? 1.0 : special::Ncfdtrc::f(dfn, dfd, nc, x);
    }
    static special::Scaled tail(double x, double dfn, double dfd, double nc, bool upper) {
        return special::nc_detail::ncfdtr_tail(dfn, dfd, nc, x, upper);
    }
    static double logcdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::logcdf(x, dfn, dfd);
        if (x <= 0.0) return -kInf;
        return ncd::log_tail([&](bool u) { return tail(x, dfn, dfd, nc, u); }, false);
    }
    static double logccdf(double x, double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::logccdf(x, dfn, dfd);
        if (x <= 0.0) return 0.0;
        return ncd::log_tail([&](bool u) { return tail(x, dfn, dfd, nc, u); }, true);
    }
    static double inv(double p, double dfn, double dfd, double nc, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (nc == 0.0) return upper ? F::iccdf(p, dfn, dfd) : F::icdf(p, dfn, dfd);
        if (p == 0.0) return upper ? kInf : 0.0;
        if (p == 1.0) return upper ? 0.0 : kInf;
        const double c = (dfn + 2.0 * nc) / (dfn + nc), k = (dfn + nc) / c;  // Patnaik's start
        const double x0 = c * k / dfn * special::Fdtri::f(k, dfd, upper ? 1.0 - p : p);
        return ncd::invert([&](double x, bool u) { return tail(x, dfn, dfd, nc, u); }, ncd::target_p(p, upper), x0, true);
    }
    static double icdf(double p, double dfn, double dfd, double nc) { return inv(p, dfn, dfd, nc, false); }
    static double iccdf(double p, double dfn, double dfd, double nc) { return inv(p, dfn, dfd, nc, true); }
    static double from_log(double lp, double dfn, double dfd, double nc, bool upper) {
        if (nc == 0.0) return upper ? F::ilogccdf(lp, dfn, dfd) : F::ilogcdf(lp, dfn, dfd);
        if (!(lp <= 0.0)) return kNaN;
        if (lp == -kInf) return inv(0.0, dfn, dfd, nc, upper);
        const ncd::Target tg = ncd::target_log(lp, upper);
        if (tg.p.m == 0.0) return inv(1.0, dfn, dfd, nc, upper);
        const double c = (dfn + 2.0 * nc) / (dfn + nc), k = (dfn + nc) / c;  // Patnaik's start, at the log target
        const double x0 = c * k / dfn * (upper ? F::ilogccdf(lp, k, dfd) : F::ilogcdf(lp, k, dfd));
        return ncd::invert([&](double x, bool u) { return tail(x, dfn, dfd, nc, u); }, tg, x0, true);
    }
    static double ilogcdf(double lp, double dfn, double dfd, double nc) { return from_log(lp, dfn, dfd, nc, false); }
    static double ilogccdf(double lp, double dfn, double dfd, double nc) { return from_log(lp, dfn, dfd, nc, true); }

    // F = A B, A = X / dfn (X ~ chi-square(dfn, nc)), B = dfd / V (V ~ chi-square(dfd)).
    // A's central moments from X's cumulants 2^(r-1) (r-1)! (dfn + r nc); B = dfd W, W an
    // inverse gamma (dfd/2, 1/2).
    static ncf_detail::Central central(double dfn, double dfd, double nc, double& ma, double& mb) {
        const double s2 = 2.0 * (dfn + 2.0 * nc), s3 = 8.0 * (dfn + 3.0 * nc);
        const double s4 = 48.0 * (dfn + 4.0 * nc) + 3.0 * s2 * s2;
        const double d = dfn;
        const double mx[5] = {1.0, 0.0, s2 / (d * d), s3 / (d * d * d), s4 / (d * d * d * d)};
        const double al = 0.5 * dfd;  // W: alpha = dfd/2, beta = 1/2; B = dfd W
        const double r1 = al - 1.0, r2 = al - 2.0, r3 = al - 3.0, r4 = al - 4.0;
        const double bw = 0.5 * dfd;  // beta times dfd
        const double v2 = bw * bw / (r1 * r1 * r2);
        const double v3 = 4.0 * bw * bw * bw / (r1 * r1 * r1 * r2 * r3);
        const double v4 = 3.0 * (al + 5.0) * bw * bw * bw * bw / (r1 * r1 * r1 * r1 * r2 * r3 * r4);
        const double my[5] = {1.0, 0.0, v2, dfd > 6.0 ? v3 : 0.0, dfd > 8.0 ? v4 : 0.0};
        ma = (dfn + nc) / dfn;
        mb = bw / r1;  // dfd / (dfd - 2)
        return ncf_detail::product_central(ma, mb, mx, my);
    }
    static double mean(double dfn, double dfd, double nc) {
        return dfd > 2.0 ? (dfn + nc) / dfn * (dfd / (dfd - 2.0)) : kInf;
    }
    static double variance(double dfn, double dfd, double nc) {
        if (!(dfd > 4.0)) return dfd > 2.0 ? kInf : kNaN;
        double ma, mb;
        return central(dfn, dfd, nc, ma, mb).m2;
    }
    static double standard_deviation(double dfn, double dfd, double nc) {
        if (!(dfd > 4.0)) return dfd > 2.0 ? kInf : kNaN;
        double ma, mb;
        return std::sqrt(central(dfn, dfd, nc, ma, mb).m2);
    }
    static double skewness(double dfn, double dfd, double nc) {
        if (!(dfd > 6.0)) return dfd > 4.0 ? kInf : kNaN;
        double ma, mb;
        const ncf_detail::Central c = central(dfn, dfd, nc, ma, mb);
        return c.m3 / (c.m2 * std::sqrt(c.m2));
    }
    // m4 / m2^2 - 3 cancels (to 2e-13 at a kurtosis of 0.016): the excess m4 - 3 m2^2 of
    // P = b X + a Y + X Y in the cumulants of X and Y, every term positive (X a scaled
    // noncentral chi-square, Y a scaled inverse gamma: all their cumulants positive):
    //   k4X k4Y + 3 k4X m2Y^2 + 3 m2X^2 k4Y + 6 m2X^2 m2Y^2 + 4 b k4X m3Y + 12 b m2X^2 m3Y
    //   + 6 b^2 k4X m2Y + 12 b^2 m2X^2 m2Y + b^4 k4X + 4 a m3X k4Y + 12 a m3X m2Y^2
    //   + 12 a b m3X m3Y + 12 a b^2 m3X m2Y + 6 a^2 m2X k4Y + 12 a^2 m2X m2Y^2 + 12 a^2 b m2X m3Y + a^4 k4Y,
    // k4X = 48 (dfn + 4 nc) / dfn^4, k4Y = m2Y^2 (30 al - 66) / ((al - 3)(al - 4)), al = dfd/2.
    static double kurtosis(double dfn, double dfd, double nc) {
        if (!(dfd > 8.0)) return dfd > 4.0 ? kInf : kNaN;
        double a, b;
        const ncf_detail::Central c = central(dfn, dfd, nc, a, b);
        const double d = dfn, d2 = d * d;
        const double m2 = 2.0 * (dfn + 2.0 * nc) / d2, m3 = 8.0 * (dfn + 3.0 * nc) / (d2 * d);
        const double k4x = 48.0 * (dfn + 4.0 * nc) / (d2 * d2);
        const double al = 0.5 * dfd, r1 = al - 1.0, r2 = al - 2.0, r3 = al - 3.0, r4 = al - 4.0, bw = 0.5 * dfd;
        const double n2 = bw * bw / (r1 * r1 * r2), n3 = 4.0 * bw * bw * bw / (r1 * r1 * r1 * r2 * r3);
        const double k4y = n2 * n2 * (30.0 * al - 66.0) / (r3 * r4);
        const double m22 = m2 * m2, n22 = n2 * n2, b2 = b * b, a2 = a * a;
        const double ex = k4x * k4y + 3.0 * k4x * n22 + 3.0 * m22 * k4y + 6.0 * m22 * n22 + 4.0 * b * k4x * n3 +
                          12.0 * b * m22 * n3 + 6.0 * b2 * k4x * n2 + 12.0 * b2 * m22 * n2 + b2 * b2 * k4x +
                          4.0 * a * m3 * k4y + 12.0 * a * m3 * n22 + 12.0 * a * b * m3 * n3 + 12.0 * a * b2 * m3 * n2 +
                          6.0 * a2 * m2 * k4y + 12.0 * a2 * m2 * n22 + 12.0 * a2 * b * m2 * n3 + a2 * a2 * k4y;
        return ex / (c.m2 * c.m2);
    }
    // f'/f = E[(a_j - 1)/x - (a_j + b) dfn / (dfn x + dfd)] under the terms' weights, so the
    // mode solves x = (A - 1) dfd / (dfn (1 + b)), A = E[a_j]; dfn < 2: the density is
    // infinite at 0 (T21).
    static double mode(double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::mode(dfn, dfd);
        if (dfn < 2.0) return 0.0;
        const double b = 0.5 * dfd;
        auto phi = [&](double x) {
            const ncd::Series s = series(x, dfn, dfd, nc);
            return ((0.5 * dfn - 1.0) + s.mean_j) * dfd - dfn * (1.0 + b) * x;
        };
        if (!(phi(1e-300) > 0.0)) return 0.0;
        auto g = [&](double x) { return -phi(x) / (dfn * (1.0 + b) * x); };  // increasing, scale-free
        const double x0 = std::max(1e-3, (0.5 * dfn + 0.5 * nc - 1.0) * dfd / (dfn * (1.0 + b)));
        return special::nc_detail::solve(g, x0, [](double x) { return std::log(x); },
                                         [](double u) { return std::exp(u); }, 0.125, 0.0, kInf);
    }
    static double entropy(double dfn, double dfd, double nc) {
        if (nc == 0.0) return F::entropy(dfn, dfd);
        const double m = std::max((dfn + nc) / dfn, 1e-3);
        return ncd::entropy(
            [&](double x) { return x > 0.0 && std::isfinite(x) ? series(x, dfn, dfd, nc).log : num::DD{-kInf, 0.0}; },
            [&](double u) { return m * std::exp(u); }, [&](double u) { return m * std::exp(u); });
    }
    static double support_lo(double, double, double) { return 0.0; }
    static double support_hi(double, double, double) { return kInf; }
};

// ---------------------------------------------------------------- t

namespace nct_detail {

using num::DD;
namespace dd = num::dd;

// log of M_k(t) = int_0^inf s^k phi(t s - nc) f_S(s) ds, S = sqrt(V / df) (k = 1: the
// density). With s = e^v and z = t s - nc,
//     s^k phi(z) f_S(s) ds = e^F(v) dv / sqrt(2 pi),
//     F(v) = -z^2/2 + log(df/pi)/2 - log Gamma*(df/2) - (df/2)(e^2v - 1 - 2v) + (k - 1) v + v
// (as special's nct_opposite, with s^k): split at the stationary point s*, the root of
// (t^2 + df) s^2 - nc t s - (df + k) = 0, as F* (double-double) plus a remainder without
// cancellation; trapezoid in u over v - v* = c A sinh(u / A).
struct Moment {
    DD log;         // log M_k(t)
    double mean_z;  // E[t S - nc] under M_k's weights (computed when asked)
};
inline Moment moment(int k, double df, double nc, double t, bool want_z = false) {
    const double a = -nc;
    const double C0 = df + k;
    // s*: in w = |t| s for |t| > 1 ((1 + df/t^2) w^2 + a sign(t) w - C0 = 0): t^2 overflows past 1e154
    double sstar;
    if (std::fabs(t) > 1.0) {
        const double at = std::fabs(t), A = 1.0 + df / at / at, B = t > 0.0 ? a : -a;
        const double disc = std::sqrt(B * B + 4.0 * A * C0);
        sstar = (B >= 0.0 ? 2.0 * C0 / (B + disc) : (-B + disc) / (2.0 * A)) / at;
    } else {
        const double A2 = t * t + df, B1 = a * t;
        const double disc = std::sqrt(B1 * B1 + 4.0 * A2 * C0);
        sstar = B1 >= 0.0 ? 2.0 * C0 / (B1 + disc) : (-B1 + disc) / (2.0 * A2);
    }
    const double p = t * sstar;
    const DD zdd = dd::two_prod(t, sstar) + a;
    const double z = zdd.hi;
    const DD s2 = dd::two_prod(sstar, sstar);
    const DD lnS = dd::log(sstar);
    const DD hstar = s2 - 1.0 - lnS * 2.0;
    const DD Fstar = -(zdd * zdd) * 0.5 + (dd::log(df) - dd::log(special::kPi)) * 0.5 - hstar * (0.5 * df) -
                     std::log(special::gamma_star(0.5 * df)) + lnS * static_cast<double>(k);
    const double c1 = (-(zdd * DD{p, 0.0}) - (s2 - 1.0) * df).hi + k;
    const double kappa = p * p + z * p + 2.0 * df * sstar * sstar;
    const double sigma = 1.0 / std::sqrt(std::max(kappa, 1e-300));
    using special::nc_detail::em1mx;
    auto integrand = [&](double delta) {
        const double em = em1mx(delta), q = p * std::expm1(delta);
        const double dF = c1 * delta - z * p * em - 0.5 * q * q - 0.5 * df * (sstar * sstar) * em1mx(2.0 * delta);
        return std::exp(dF);
    };
    constexpr double kStep = 0.25, kA = 8.0;
    const double c = std::min(sigma, 0.4);
    // z at a node, z* + p (e^delta - 1) (z* to double-double: no cancellation near z = 0)
    auto zat = [&](double delta) { return zdd.hi + (zdd.lo + p * std::expm1(delta)); };
    disc::KahanSum I, Iz;
    I.add(integrand(0.0));
    if (want_z) Iz.add(integrand(0.0) * zat(0.0));
    for (int dir = -1; dir <= 1; dir += 2) {
        for (int i = 1; i < 100000; ++i) {
            const double u = dir * i * kStep / kA;
            const double delta = c * kA * std::sinh(u);
            const double v = integrand(delta) * std::cosh(u);
            I.add(v);
            if (want_z) Iz.add(v * zat(delta));
            if (v <= 0x1p-60 * I.s && i > 8) break;
        }
    }
    return {Fstar - num::kLogSqrt2Pi + std::log(I.value() * kStep * c), want_z ? Iz.value() / I.value() : 0.0};
}
inline DD log_moment(int k, double df, double nc, double t) { return moment(k, df, nc, t).log; }

// R = sqrt(df / V), x = df / 2: L_1 = log E[R] and D_k = log E[R^k] - k L_1 (k = 2..4) as
// double-doubles: the 1/x series (coeffs/nct_moments.hpp) from x = 32, else from
// log_gamma (no cancellation that matters there). (From x = 8 the series' truncation left
// 2e-13 of the moments at x = 8.6: its coefficients grow faster than its terms' x^-n fall
// there; at 32 it is below 1e-19.)
inline DD series_term(int r, double x) {
    DD s{0.0, 0.0};
    const DD inv = num::div(DD{1.0, 0.0}, x);
    DD p = inv;
    double prev = kInf;
    for (int n = 0; n < kNctTerms; ++n) {
        const DD term = DD{kNct[r][n][0], kNct[r][n][1]} * p;
        const double m = std::fabs(term.hi);
        if (m > prev) break;  // the asymptotic series turns
        s = s + term;
        if (m <= std::fabs(s.hi) * 0x1p-110) break;
        prev = m;
        p = p * inv;
    }
    return s;
}
inline DD log_raw(double k, double x) {  // log E[R^k] = log Gamma(x - k/2) - log Gamma(x) + (k/2) log x
    return num::log_gamma(DD{x, 0.0} - 0.5 * k) - num::log_gamma(DD{x, 0.0}) + num::log_dd(x) * (0.5 * k);
}
struct RMoments {
    DD l1, d[5];  // d[2..4]
};
inline RMoments r_moments(double df, int kmax) {
    const double x = 0.5 * df;
    RMoments m{};
    if (x >= 32.0) {
        m.l1 = series_term(0, x);
        for (int k = 2; k <= kmax; ++k) m.d[k] = series_term(k - 1, x);
    } else {
        m.l1 = log_raw(1.0, x);
        for (int k = 2; k <= kmax; ++k) m.d[k] = log_raw(k, x) - m.l1 * static_cast<double>(k);
    }
    return m;
}

}  // namespace nct_detail

struct NoncentralT {
    static constexpr char name[] = "noncentralt";
    static constexpr std::array<const char*, 2> params{"df", "nc"};

    static bool valid(double df, double nc) { return df > 0.0 && std::isfinite(df) && std::isfinite(nc); }

    static double pdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::pdf(t, df);
        if (std::isinf(t)) return 0.0;
        return num::exp_dd(nct_detail::log_moment(1, df, nc, t));
    }
    static double logpdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::logpdf(t, df);
        if (std::isinf(t)) return -kInf;
        const num::DD l = nct_detail::log_moment(1, df, nc, t);
        return l.hi + l.lo;
    }
    static double cdf(double t, double df, double nc) {
        return nc == 0.0 ? StudentT::cdf(t, df) : special::Nctdtr::f(df, nc, t);
    }
    static double ccdf(double t, double df, double nc) {
        return nc == 0.0 ? StudentT::ccdf(t, df) : special::Nctdtrc::f(df, nc, t);
    }
    static special::Scaled tail(double t, double df, double nc, bool upper) {
        return special::nc_detail::nctdtr_tail(df, nc, t, upper);
    }
    static double logcdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::logcdf(t, df);
        return ncd::log_tail([&](bool u) { return tail(t, df, nc, u); }, false);
    }
    static double logccdf(double t, double df, double nc) {
        if (nc == 0.0) return StudentT::logccdf(t, df);
        return ncd::log_tail([&](bool u) { return tail(t, df, nc, u); }, true);
    }
    static double inv(double p, double df, double nc, bool upper) {
        if (!(p >= 0.0 && p <= 1.0)) return kNaN;
        if (nc == 0.0) return upper ? StudentT::iccdf(p, df) : StudentT::icdf(p, df);
        if (p == 0.0) return upper ? kInf : -kInf;
        if (p == 1.0) return upper ? -kInf : kInf;
        // start: the special's (P(T <= t) ~ Phi of a normalized t)
        const double pl = upper ? 1.0 - p : p;
        const double z = special::distr_detail::one<special::Ndtri>(std::clamp(pl, 1e-300, 1.0 - 1e-16));
        const double A = 1.0 - 0.25 / df, qa = A * A - z * z / (2.0 * df);
        double t0 = nc + z;
        if (qa > 0.0) {
            const double d = A * A * nc * nc - qa * (nc * nc - z * z);
            if (d >= 0.0) t0 = (A * nc + std::copysign(std::sqrt(d), z)) / qa;
        }
        return ncd::invert([&](double t, bool u) { return tail(t, df, nc, u); }, ncd::target_p(p, upper), t0, false);
    }
    static double icdf(double p, double df, double nc) { return inv(p, df, nc, false); }
    static double iccdf(double p, double df, double nc) { return inv(p, df, nc, true); }
    static double from_log(double lp, double df, double nc, bool upper) {
        if (nc == 0.0) return upper ? StudentT::ilogccdf(lp, df) : StudentT::ilogcdf(lp, df);
        if (!(lp <= 0.0)) return kNaN;
        if (lp == -kInf) return inv(0.0, df, nc, upper);
        const ncd::Target tg = ncd::target_log(lp, upper);
        if (tg.p.m == 0.0) return inv(1.0, df, nc, upper);
        // start: the central t's quantile at the log target, shifted by nc
        const double t0 = nc + (upper ? StudentT::ilogccdf(lp, df) : StudentT::ilogcdf(lp, df));
        return ncd::invert([&](double t, bool u) { return tail(t, df, nc, u); }, tg, t0, false);
    }
    static double ilogcdf(double lp, double df, double nc) { return from_log(lp, df, nc, false); }
    static double ilogccdf(double lp, double df, double nc) { return from_log(lp, df, nc, true); }

    // T = U R, U ~ N(nc, 1), R = sqrt(df / V) with r = E[R] and central moments y2..y4
    // (= r^k times E_2, E_3 - 3 E_2, E_4 - 4 E_3 + 6 E_2, E_k = e^(D_k) - 1):
    //   Var T = nc^2 y2 + E[R^2],                       E[R^2] = df / (df - 2)
    //   mu_3  = nc (nc^2 y3 + 3 y3 + 6 r y2)
    //   mu_4 - 3 Var^2 = nc^4 k4(R) + 6 nc^2 (2 r y3 + y4 - y2^2) + 3 Var(R^2),
    //   k4(R) = r^4 (E_4 - 4 E_3 + 6 E_2 - 3 E_2^2),    Var(R^2) = 2 df^2 / ((df - 2)^2 (df - 4)),
    // every term positive (T - nc r = nc (R - r) + R Z, Z ~ N(0, 1) independent of R).
    struct Moments {
        double r, y2, y3, y4, k4;
    };
    static Moments moments(double df, int kmax) {
        const nct_detail::RMoments m = nct_detail::r_moments(df, kmax);
        Moments o{};
        o.r = num::exp_dd(m.l1);
        const num::DD e2 = num::expm1_full(m.d[2]);
        o.y2 = o.r * o.r * (e2.hi + e2.lo);
        if (kmax >= 3) {
            const num::DD e3 = num::expm1_full(m.d[3]);
            const num::DD n3 = e3 - e2 * 3.0;
            o.y3 = o.r * o.r * o.r * (n3.hi + n3.lo);
            if (kmax >= 4) {
                const num::DD e4 = num::expm1_full(m.d[4]);
                const num::DD n4 = e4 - e3 * 4.0 + e2 * 6.0;
                const num::DD k4 = n4 - e2 * e2 * 3.0;
                const double r4 = o.r * o.r * o.r * o.r;
                o.y4 = r4 * (n4.hi + n4.lo);
                o.k4 = r4 * (k4.hi + k4.lo);
            }
        }
        return o;
    }
    static double mean(double df, double nc) {
        if (!(df > 1.0)) return kNaN;
        if (nc == 0.0) return 0.0;
        return nc * num::exp_dd(nct_detail::r_moments(df, 1).l1);
    }
    static double variance(double df, double nc) {
        if (!(df > 2.0)) return df > 1.0 ? kInf : kNaN;
        const Moments m = moments(df, 2);
        return nc * nc * m.y2 + df / (df - 2.0);
    }
    static double standard_deviation(double df, double nc) {
        if (!(df > 2.0)) return df > 1.0 ? kInf : kNaN;
        return std::sqrt(variance(df, nc));
    }
    static double skewness(double df, double nc) {
        if (!(df > 3.0)) return kNaN;
        if (nc == 0.0) return 0.0;
        const Moments m = moments(df, 3);
        const double v = nc * nc * m.y2 + df / (df - 2.0);
        const double mu3 = nc * (nc * nc * m.y3 + 3.0 * m.y3 + 6.0 * m.r * m.y2);
        return mu3 / (v * std::sqrt(v));
    }
    static double kurtosis(double df, double nc) {
        if (!(df > 4.0)) return df > 2.0 ? kInf : kNaN;
        if (nc == 0.0) return StudentT::kurtosis(df);
        const Moments m = moments(df, 4);
        const double v = nc * nc * m.y2 + df / (df - 2.0);
        const double vr2 = 2.0 * df * df / ((df - 2.0) * (df - 2.0) * (df - 4.0));
        const double n2 = nc * nc;
        const double ex = n2 * n2 * m.k4 + 6.0 * n2 * (2.0 * m.r * m.y3 + m.y4 - m.y2 * m.y2) + 3.0 * vr2;
        return ex / (v * v);
    }
    // f'(t) = -(t M_3 - nc M_2) = -M_2 E[t S - nc] (the expectation under M_2's weights
    // s^2 phi(t s - nc) f_S(s)): the mode solves E[z] = 0, z = t s - nc, summed on one grid.
    // (As t M_3 / M_2 = nc, two moments' independent rounding, nc eps, met a flat g: 4e-13 of
    // the mode at df = 0.6, nc = 75; E[z] carries E|z| eps.)
    static double mode(double df, double nc) {
        if (nc == 0.0) return 0.0;
        auto g = [&](double t) { return nct_detail::moment(2, df, nc, t, true).mean_z; };
        const double t0 = nc * std::sqrt(df / (df + 2.5));
        return special::nc_detail::solve(g, t0, [](double t) { return std::asinh(t); },
                                         [](double u) { return std::sinh(u); }, 0.125, -kInf, kInf);
    }
    static double entropy(double df, double nc) {
        if (nc == 0.0) return StudentT::entropy(df);
        const double m = mode(df, nc);
        const double s = df > 2.0 ? std::sqrt(variance(df, nc)) : 1.0 + std::fabs(nc);
        return ncd::entropy([&](double t) { return std::isfinite(t) ? nct_detail::log_moment(1, df, nc, t) : num::DD{-kInf, 0.0}; },
                            [&](double u) { return m + s * std::sinh(u); }, [&](double u) { return s * std::cosh(u); });
    }
    static double support_lo(double, double) { return -kInf; }
    static double support_hi(double, double) { return kInf; }
};

}  // namespace sd::stats
