// The fits of goodness_of_fit and anderson (milestone H9, decisions H40 and H121 in
// docs/design/htest.md §1.1): maximum likelihood for the sd.stats families, solved to the last
// digits by bracketing (every score equation here has one root in its parameter, so no starting
// value is needed), and the statistics of a fitted distribution.
//
// Families and their parameters (sd.stats' names and order):
//   0 Normal (mu, sigma)       scipy's _fit_norm: the mean and the standard deviation (n - 1);
//                              mu given: sigma = sqrt(mean (x - mu)^2); sigma given: the mean
//   1 Exponential (scale)      the mean (location 0)
//   2 Logistic (mu, sigma)     sum tanh(t/2) = 0 and sum t tanh(t/2) = n, t = (x - mu)/sigma
//   3 Gumbel (mu, sigma)       the maximum's: sigma from mean(x) - sum x e^(-x/s)/sum e^(-x/s) = s,
//                              mu = -sigma log mean e^(-x/sigma); mu given: sum t (1 - e^-t) = n
//   4 Weibull (c, scale)       1/c + mean log x = sum x^c log x / sum x^c, scale = mean(x^c)^(1/c)
//   5 LogNormal (mu, sigma)    the mean and the standard deviation (n) of log x
//   6 Laplace (mu, sigma)      the median, the mean absolute deviation from it
//   7 Rayleigh (scale)         sqrt(sum x^2 / 2n)
//   8 Weibull with a location  (c, loc, scale): anderson's (scipy's weibull_min): the profile
//                              likelihood over loc < min x, its largest interior maximum with c >= 1
// Each score equation is monotone in its parameter (decreasing), so the root is bracketed and
// refined by bisection and the secant (Illinois) to a relative 2^-52. A fit that does not exist
// (data outside the support, a constant sample, no interior maximum) is false (NaN parameters).
#pragma once

#include "htest/dist_array.hpp"
#include "htest/moments.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/exponential.hpp"
#include "stats/dist/extreme.hpp"
#include "stats/dist/laplace.hpp"
#include "stats/dist/lifetime.hpp"
#include "stats/dist/lognormal.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/weibull.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::htest::mle {

using i64 = std::int64_t;
namespace sm = stats::method;

enum Family {
    kNormal = 0,
    kExponential = 1,
    kLogistic = 2,
    kGumbel = 3,
    kWeibull = 4,
    kLogNormal = 5,
    kLaplace = 6,
    kRayleigh = 7,
    kWeibull3 = 8
};

inline int n_params(int f) {
    switch (f) {
        case kExponential:
        case kRayleigh: return 1;
        case kWeibull3: return 3;
        default: return 2;
    }
}

// The root of a decreasing f on [lo, hi] (f(lo) > 0 > f(hi)): bisection and Illinois' secant.
template <class F>
double root(F&& f, double lo, double hi, double flo, double fhi) {
    int side = 0;
    for (int it = 0; it < 400; ++it) {
        if (!(hi - lo > 4e-16 * std::max(std::fabs(lo), std::fabs(hi)))) break;
        double x = lo + flo * (hi - lo) / (flo - fhi);  // the secant
        if (!(x > lo && x < hi) || it % 4 == 3) x = 0.5 * (lo + hi);
        const double v = f(x);
        if (v == 0.0) return x;
        if (v > 0.0) {
            lo = x, flo = v;
            if (side == 1) fhi /= 2.0;
            side = 1;
        } else {
            hi = x, fhi = v;
            if (side == -1) flo /= 2.0;
            side = -1;
        }
        if (v != v) return std::numeric_limits<double>::quiet_NaN();
    }
    return 0.5 * (lo + hi);
}

// The root of a decreasing f of a positive parameter, from a guess: the bracket by doubling
// and halving (at most 2^+-1000).
template <class F>
bool positive_root(F&& f, double guess, double& out) {
    double lo = guess, hi = guess;
    double flo = f(lo), fhi = flo;
    if (flo != flo) return false;
    int i = 0;
    while (flo <= 0.0 && i++ < 2000) {
        hi = lo, fhi = flo;
        lo /= 2.0, flo = f(lo);
        if (flo != flo || lo == 0.0) return false;
    }
    i = 0;
    while (fhi >= 0.0 && i++ < 2000) {
        lo = hi, flo = fhi;
        hi *= 2.0, fhi = f(hi);
        if (fhi != fhi || std::isinf(hi)) return false;
    }
    if (!(flo > 0.0) || !(fhi < 0.0)) return false;
    out = root(f, lo, hi, flo, fhi);
    return out == out && out > 0.0;
}

inline double mean_of(const std::vector<double>& v) {
    Acc s;
    for (double x : v) s.add(x);
    return to_double(s.value()) / static_cast<double>(v.size());
}

// ---- the logistic and the Gumbel of standardized values (anderson's, H2) ----

// the logistic's maximum likelihood (location, scale) of standardized values z (mean 0, sd 1),
// in the parameters (alpha, beta) of w = beta z - alpha, where the log-likelihood is concave:
// Newton's steps with halving
inline bool logistic_fit(const std::vector<double>& z, double& alpha, double& beta) {
    const double N = static_cast<double>(z.size());
    auto loglik = [&](double al, double be) {
        Acc s;
        for (double v : z) {
            const double w = std::fabs(be * v - al);
            s.add(-w - 2.0 * std::log1p(std::exp(-w)));
        }
        return to_double(s.value()) + N * std::log(be);
    };
    alpha = 0.0, beta = 1.8137993642342178;  // pi / sqrt 3
    double ll = loglik(alpha, beta);
    for (int it = 0; it < 200; ++it) {
        Acc ga, gb, haa, hab, hbb;
        for (double v : z) {
            const double w = beta * v - alpha;
            const double th = std::tanh(w / 2.0);
            const double h = -0.5 * (1.0 - th * th);
            ga.add(th);
            gb.add(-v * th);
            haa.add(h);
            hab.add(-h * v);
            hbb.add(h * v * v);
        }
        const double g1 = to_double(ga.value()), g2 = to_double(gb.value()) + N / beta;
        const double h11 = to_double(haa.value()), h12 = to_double(hab.value()),
                     h22 = to_double(hbb.value()) - N / (beta * beta);
        const double det = h11 * h22 - h12 * h12;
        if (!(det > 0.0)) return false;
        double da = -(h22 * g1 - h12 * g2) / det, db = -(-h12 * g1 + h11 * g2) / det;
        double step = 1.0;
        double na = alpha + da, nbv = beta + db, nl;
        while (true) {
            na = alpha + step * da, nbv = beta + step * db;
            nl = nbv > 0.0 ? loglik(na, nbv) : -std::numeric_limits<double>::infinity();
            if (nl >= ll - 1e-12 * std::fabs(ll) || step < 0x1p-30) break;
            step /= 2.0;
        }
        const bool small =
            std::fabs(step * da) <= 4e-16 * (1.0 + std::fabs(alpha)) && std::fabs(step * db) <= 4e-16 * beta;
        alpha = na, beta = nbv, ll = std::max(ll, nl);
        if (small) return true;
    }
    return false;
}

// the Gumbel's (maximum's) maximum likelihood of standardized values z: the scale b solves
// mean(z) - sum z e^(-z/b) / sum e^(-z/b) - b = 0 (decreasing in b), the location
// a = -b log(mean(e^(-z/b))).
inline bool gumbel_fit(const std::vector<double>& z, double& loc, double& scale) {
    const double N = static_cast<double>(z.size());
    const double zmin = *std::min_element(z.begin(), z.end());
    Acc sm0;
    for (double v : z) sm0.add(v);
    const double mean = to_double(sm0.value()) / N;
    auto f = [&](double b, double* df) {
        Acc s0, s1, s2;
        for (double v : z) {
            const double e = std::exp(-(v - zmin) / b);
            s0.add(e), s1.add(v * e), s2.add(v * v * e);
        }
        const double w0 = to_double(s0.value()), m1 = to_double(s1.value()) / w0, m2 = to_double(s2.value()) / w0;
        if (df) *df = -(m2 - m1 * m1) / (b * b) - 1.0;
        return mean - m1 - b;
    };
    double lo = 0.39, hi = 1.56;  // around sqrt 6 / pi
    for (int i = 0; i < 200 && f(lo, nullptr) <= 0.0; ++i) lo /= 2.0;
    for (int i = 0; i < 200 && f(hi, nullptr) >= 0.0; ++i) hi *= 2.0;
    if (!(f(lo, nullptr) > 0.0) || !(f(hi, nullptr) < 0.0)) return false;
    double b = 0.5 * (lo + hi);
    for (int it = 0; it < 300; ++it) {
        double d;
        const double v = f(b, &d);
        if (v == 0.0) break;
        (v > 0.0 ? lo : hi) = b;
        double nb2 = b - v / d;
        if (!(nb2 > lo && nb2 < hi)) nb2 = 0.5 * (lo + hi);
        if (std::fabs(nb2 - b) <= 2e-16 * b) {
            b = nb2;
            break;
        }
        b = nb2;
        if (hi - lo <= 2e-16 * hi) break;
    }
    Acc s0;
    for (double v : z) s0.add(std::exp(-(v - zmin) / b));
    loc = -b * (-zmin / b + std::log(to_double(s0.value())) - std::log(N));
    scale = b;
    return std::isfinite(loc) && b > 0.0;
}

// ---- the Weibull ----

// the shape of the Weibull with location u (z = x - u > 0, sorted): the root of
// 1/c + mean log z - sum z^c log z / sum z^c (decreasing in c); false when the logs are all equal
inline bool weibull_shape(const std::vector<double>& x, double u, double& c) {
    const auto n = static_cast<double>(x.size());
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double z = x[i] - u;
        if (!(z > 0.0) || !std::isfinite(z)) return false;
        y[i] = std::log(z);
    }
    const double ymax = *std::max_element(y.begin(), y.end()), ymin = *std::min_element(y.begin(), y.end());
    if (!(ymax > ymin)) return false;
    Acc sy;
    for (double v : y) sy.add(v);
    const double my = to_double(sy.value()) / n;
    auto h = [&](double cc) {
        Acc s0, s1;
        for (double v : y) {
            const double w = std::exp(cc * (v - ymax));
            s0.add(w), s1.add(w * (v - ymax));
        }
        return 1.0 / cc + (my - ymax) - to_double(s1.value()) / to_double(s0.value());
    };
    return positive_root(h, 1.0 / (ymax - ymin) + 0.5, c);
}

inline double weibull_scale(const std::vector<double>& x, double u, double c) {
    double ymax = -std::numeric_limits<double>::infinity();
    for (double v : x) ymax = std::max(ymax, std::log(v - u));
    Acc s0;
    for (double v : x) s0.add(std::exp(c * (std::log(v - u) - ymax)));
    return std::exp(ymax + std::log(to_double(s0.value()) / static_cast<double>(x.size())) / c);
}

// the Weibull log-likelihood at (c, u, s) per observation's sum
inline double weibull_loglik(const std::vector<double>& x, double c, double u, double s) {
    Acc l;
    for (double v : x) {
        const double t = (v - u) / s;
        l.add(std::log(c / s) + (c - 1.0) * std::log(t) - std::pow(t, c));
    }
    return to_double(l.value());
}

// anderson's three-parameter Weibull (scipy's weibull_min.fit, its location free): the profile
// log-likelihood over the location u = min x - tau on a grid of tau from 1e-8 to 1e4 times the
// range (a quarter decade apart), its interior local maxima refined by golden sections in log
// tau, the largest of those with c >= 1 (scipy refuses c < 1 and a location at the minimum: the
// table of critical values does not cover them).
inline bool weibull3_fit(const std::vector<double>& x, double& c, double& loc, double& scale) {
    const double xmin = x.front(), range = x.back() - x.front();
    if (!(range > 0.0) || !std::isfinite(range)) return false;
    auto prof = [&](double lt, double* cc) {
        const double u = xmin - range * std::exp(lt);
        double cv;
        if (!weibull_shape(x, u, cv)) return -std::numeric_limits<double>::infinity();
        if (cc) *cc = cv;
        return weibull_loglik(x, cv, u, weibull_scale(x, u, cv));
    };
    constexpr int kGrid = 49;
    const double l10 = std::log(10.0);
    std::vector<double> lt(kGrid), lp(kGrid);
    for (int i = 0; i < kGrid; ++i) lt[i] = (-8.0 + 0.25 * i) * l10, lp[i] = prof(lt[i], nullptr);
    bool found = false;
    double best = -std::numeric_limits<double>::infinity();
    for (int i = 1; i + 1 < kGrid; ++i) {
        if (!(lp[i] >= lp[i - 1] && lp[i] >= lp[i + 1]) || !std::isfinite(lp[i])) continue;
        // golden sections on [lt[i-1], lt[i+1]]
        double a = lt[i - 1], b = lt[i + 1];
        const double g = 0.3819660112501051;
        double x1 = a + g * (b - a), x2 = b - g * (b - a);
        double f1 = prof(x1, nullptr), f2 = prof(x2, nullptr);
        for (int it = 0; it < 80 && b - a > 1e-13 * std::max(1.0, std::fabs(a)); ++it) {
            if (f1 >= f2) {
                b = x2, x2 = x1, f2 = f1;
                x1 = a + g * (b - a), f1 = prof(x1, nullptr);
            } else {
                a = x1, x1 = x2, f1 = f2;
                x2 = b - g * (b - a), f2 = prof(x2, nullptr);
            }
        }
        const double lm = 0.5 * (a + b);
        double cv = 0.0;
        const double v = prof(lm, &cv);
        if (!(cv >= 1.0) || !(v > best)) continue;
        best = v, found = true;
        loc = xmin - range * std::exp(lm), c = cv;
    }
    if (!found) return false;
    scale = weibull_scale(x, loc, c);
    return std::isfinite(scale) && scale > 0.0;
}

// ---- the fits ----

// x sorted; fixed[q] the given parameters (NaN: fitted); th the parameters
inline bool fit(int f, const std::vector<double>& x, const double* fixed, double* th) {
    const auto n = static_cast<double>(x.size());
    if (x.empty() || !std::isfinite(x.front()) || !std::isfinite(x.back())) return false;
    const int k = n_params(f);
    for (int q = 0; q < k; ++q) th[q] = fixed[q];
    auto is_free = [&](int q) { return fixed[q] != fixed[q]; };
    switch (f) {
        case kNormal: {
            const Moments mo = moments_of(x.data(), nullptr, static_cast<i64>(x.size()));
            if (is_free(0)) th[0] = to_double(mo.mean);
            if (is_free(1)) {
                if (is_free(0)) {
                    if (x.size() < 2) return false;
                    th[1] = std::sqrt(to_double(mo.ss) / (n - 1.0));
                } else {
                    Acc s;
                    for (double v : x) s.add(sq_dev(v, 0.0, DD{th[0], 0.0}));
                    th[1] = std::sqrt(to_double(s.value()) / n);
                }
            }
            return th[1] > 0.0 && std::isfinite(th[1]) && std::isfinite(th[0]);
        }
        case kExponential:
            if (x.front() < 0.0) return false;
            if (is_free(0)) th[0] = mean_of(x);
            return th[0] > 0.0 && std::isfinite(th[0]);
        case kRayleigh: {
            if (x.front() < 0.0) return false;
            if (is_free(0)) {
                Acc s;
                for (double v : x) s.add(dd::two_prod(v, v));
                th[0] = std::sqrt(to_double(s.value()) / (2.0 * n));
            }
            return th[0] > 0.0 && std::isfinite(th[0]);
        }
        case kLogNormal: {
            if (!(x.front() > 0.0)) return false;
            std::vector<double> y(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::log(x[i]);
            if (is_free(0)) th[0] = mean_of(y);
            if (is_free(1)) {
                Acc s;
                for (double v : y) s.add(sq_dev(v, 0.0, DD{th[0], 0.0}));
                th[1] = std::sqrt(to_double(s.value()) / n);
            }
            return th[1] > 0.0 && std::isfinite(th[1]) && std::isfinite(th[0]);
        }
        case kLaplace: {
            const std::size_t h = x.size() / 2;
            if (is_free(0)) th[0] = x.size() % 2 ? x[h] : 0.5 * x[h - 1] + 0.5 * x[h];
            if (is_free(1)) {
                Acc s;
                for (double v : x) s.add(std::fabs(v - th[0]));
                th[1] = to_double(s.value()) / n;
            }
            return th[1] > 0.0 && std::isfinite(th[1]) && std::isfinite(th[0]);
        }
        case kLogistic:
        case kGumbel: {
            const Moments mo = moments_of(x.data(), nullptr, static_cast<i64>(x.size()));
            if (mo.constant() || x.size() < 2) return false;
            const double mean = to_double(mo.mean), sd = std::sqrt(to_double(mo.ss) / (n - 1.0));
            if (is_free(0) && is_free(1)) {
                std::vector<double> z(x.size());
                for (std::size_t i = 0; i < x.size(); ++i) {
                    const DD d = dd::two_sum(x[i], -mo.mean.hi) - mo.mean.lo;
                    z[i] = (d.hi + d.lo) / sd;
                }
                double a, b;
                if (f == kLogistic) {
                    if (!logistic_fit(z, a, b)) return false;
                    th[1] = sd / b, th[0] = mean + sd * a / b;
                } else {
                    if (!gumbel_fit(z, a, b)) return false;
                    th[0] = mean + sd * a, th[1] = sd * b;
                }
            } else if (is_free(0)) {  // the location given the scale
                const double s = th[1];
                if (f == kLogistic) {
                    auto g = [&](double mu) {
                        Acc a;
                        for (double v : x) a.add(std::tanh((v - mu) / (2.0 * s)));
                        return to_double(a.value());
                    };
                    double lo = x.front(), hi = x.back();
                    const double glo = g(lo), ghi = g(hi);
                    if (!(glo > 0.0) || !(ghi < 0.0)) return false;
                    th[0] = root(g, lo, hi, glo, ghi);
                } else {
                    Acc a;
                    for (double v : x) a.add(std::exp(-(v - x.front()) / s));
                    th[0] = x.front() - s * std::log(to_double(a.value()) / n);
                }
            } else if (is_free(1)) {  // the scale given the location
                const double mu = th[0];
                auto g = [&](double s) {
                    Acc a;
                    for (double v : x) {
                        const double t = (v - mu) / s;
                        a.add(f == kLogistic ? t * std::tanh(t / 2.0) : -t * std::expm1(-t));
                    }
                    return to_double(a.value()) - n;
                };
                if (!positive_root(g, sd, th[1])) return false;
            }
            return th[1] > 0.0 && std::isfinite(th[1]) && std::isfinite(th[0]);
        }
        case kWeibull: {
            if (!(x.front() > 0.0)) return false;
            if (is_free(0) && is_free(1)) {
                if (!weibull_shape(x, 0.0, th[0])) return false;
                th[1] = weibull_scale(x, 0.0, th[0]);
            } else if (is_free(1)) {
                th[1] = weibull_scale(x, 0.0, th[0]);
            } else if (is_free(0)) {
                const double s = th[1];
                std::vector<double> z(x.size());
                for (std::size_t i = 0; i < x.size(); ++i) z[i] = std::log(x[i] / s);
                Acc sz;
                for (double v : z) sz.add(v);
                const double tz = to_double(sz.value());
                auto g = [&](double cc) {
                    Acc a;
                    for (double v : z) a.add(std::exp(cc * v) * v);
                    return n / cc + tz - to_double(a.value());
                };
                if (!positive_root(g, 1.0, th[0])) return false;
            }
            return th[0] > 0.0 && th[1] > 0.0 && std::isfinite(th[0]) && std::isfinite(th[1]);
        }
        case kWeibull3: return weibull3_fit(x, th[0], th[1], th[2]);
        default: return false;
    }
}

// ---- the fitted distribution's functions ----

template <class M>
inline double fn(int f, const double* th, double v) {
    switch (f) {
        case kNormal: return stats::method_of<stats::Normal, M>(v, th[0], th[1]);
        case kExponential: return stats::method_of<stats::Exponential, M>(v, th[0]);
        case kLogistic: return stats::method_of<stats::Logistic, M>(v, th[0], th[1]);
        case kGumbel: return stats::method_of<stats::Gumbel, M>(v, th[0], th[1]);
        case kWeibull: return stats::method_of<stats::Weibull, M>(v, th[0], th[1]);
        case kLogNormal: return stats::method_of<stats::LogNormal, M>(v, th[0], th[1]);
        case kLaplace: return stats::method_of<stats::Laplace, M>(v, th[0], th[1]);
        case kRayleigh: return stats::method_of<stats::Rayleigh, M>(v, th[0]);
        case kWeibull3: {
            if constexpr (std::is_same_v<M, sm::Icdf>)
                return th[1] + stats::method_of<stats::Weibull, M>(v, th[0], th[2]);
            else return stats::method_of<stats::Weibull, M>(v - th[1], th[0], th[2]);
        }
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

// fn<M>(f, th, x[i]) for i in [0, n) into out (method_array: the same bits)
template <class M>
inline void fn_array(int f, const double* th, const double* x, i64 n, double* out) {
    switch (f) {
        case kNormal: return method_array<stats::Normal, M>(x, n, out, th[0], th[1]);
        case kExponential: return method_array<stats::Exponential, M>(x, n, out, th[0]);
        case kLogistic: return method_array<stats::Logistic, M>(x, n, out, th[0], th[1]);
        case kGumbel: return method_array<stats::Gumbel, M>(x, n, out, th[0], th[1]);
        case kWeibull: return method_array<stats::Weibull, M>(x, n, out, th[0], th[1]);
        case kLogNormal: return method_array<stats::LogNormal, M>(x, n, out, th[0], th[1]);
        case kLaplace: return method_array<stats::Laplace, M>(x, n, out, th[0], th[1]);
        case kRayleigh: return method_array<stats::Rayleigh, M>(x, n, out, th[0]);
        default:
            for (i64 i = 0; i < n; ++i) out[i] = fn<M>(f, th, x[i]);  // (kWeibull3: the shift)
    }
}

// the medians of the uniform order statistics, Beta(i, n + 1 - i) (Filliben's, scipy's exact ones)
inline std::vector<double> beta_medians(i64 n) {
    std::vector<double> m(static_cast<std::size_t>(n));
    for (i64 i = 1; i <= n; ++i)
        m[static_cast<std::size_t>(i - 1)] =
            stats::method_of<stats::Beta, sm::Icdf>(0.5, static_cast<double>(i), static_cast<double>(n + 1 - i));
    return m;
}

// stat 0: Anderson-Darling's A^2, 1: Kolmogorov-Smirnov's D, 2: Cramér-von Mises' W^2, 3:
// Filliben's correlation (scipy's _compare_dict); x sorted
inline double statistic(
    int f, const double* th, const std::vector<double>& x, int stat, const std::vector<double>& med
) {
    const auto n = static_cast<i64>(x.size());
    const double N = static_cast<double>(n);
    if (stat == 0) {
        std::vector<double> L(static_cast<std::size_t>(n)), U(static_cast<std::size_t>(n));
        fn_array<sm::Logcdf>(f, th, x.data(), n, L.data());
        fn_array<sm::Logccdf>(f, th, x.data(), n, U.data());
        Acc s;
        for (i64 i = 0; i < n; ++i) {
            const double c = (2.0 * static_cast<double>(i + 1) - 1.0) / N;
            s.add(c * (L[static_cast<std::size_t>(i)] + U[static_cast<std::size_t>(n - 1 - i)]));
        }
        return -N - to_double(s.value());
    }
    if (stat == 1 || stat == 2) {
        std::vector<double> F(static_cast<std::size_t>(n));
        fn_array<sm::Cdf>(f, th, x.data(), n, F.data());
        if (stat == 1) {
            double d = 0.0;
            for (i64 i = 0; i < n; ++i) {
                const double Fi = F[static_cast<std::size_t>(i)];
                if (Fi != Fi) return Fi;
                d = std::max(d, std::max(static_cast<double>(i + 1) / N - Fi, Fi - static_cast<double>(i) / N));
            }
            return d;
        }
        Acc s;
        for (i64 i = 0; i < n; ++i) {
            const double u = (2.0 * static_cast<double>(i + 1) - 1.0) / (2.0 * N);
            const double d = u - F[static_cast<std::size_t>(i)];
            s.add(dd::two_prod(d, d));
        }
        return 1.0 / (12.0 * N) + to_double(s.value());
    }
    std::vector<double> M(static_cast<std::size_t>(n));
    fn_array<sm::Icdf>(f, th, med.data(), n, M.data());
    const Moments mx = moments_of(x.data(), nullptr, n), mm = moments_of(M.data(), nullptr, n);
    Acc sxy;
    for (i64 i = 0; i < n; ++i) {
        const DD dx = dd::two_sum(x[static_cast<std::size_t>(i)], -mx.mean.hi) - mx.mean.lo;
        const DD dm = dd::two_sum(M[static_cast<std::size_t>(i)], -mm.mean.hi) - mm.mean.lo;
        sxy.add(dx * dm);
    }
    return to_double(sxy.value()) / std::sqrt(to_double(mx.ss) * to_double(mm.ss));
}

}  // namespace sd::htest::mle
