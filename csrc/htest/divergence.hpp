// The terms of the Cressie-Read power divergence statistics (docs/design/htest.md §3.3), shared
// by power_divergence (htest/gof.cpp) and the contingency tables' chi-squared tests
// (median_test's, htest/rank.cpp; chi2_contingency later).
//
// With delta = (O - E)/E and a = lambda + 1, every Cressie-Read statistic is
// 2 sum E g_a(delta) + (2/a) sum (O - E), with
//     g_a(delta) = ((1 + delta)^a - 1 - a delta) / (a (a - 1)) >= 0
// (g_1 = (1 + delta) log(1 + delta) - delta: the G test; the limit a = 0 is the modified one,
// 2 sum E log(E/O) = 2 sum E g_0 - 2 sum (O - E)). The terms of the first sum are never negative,
// so a statistic near 0 (O close to E) keeps its digits.
#pragma once

#include "htest/moments.hpp"
#include "special/dd.hpp"

#include <cmath>
#include <limits>

namespace sd::htest::div {

// mu(t) = t - log(1 + t) >= 0 as a double-double (special/dd.hpp)
inline DD mu_dd(double t) { return special::dd::log1pmx_neg(DD{t, 0.0}); }

// g_1(delta) = (1 + delta) log1p(delta) - delta = delta^2 - (1 + delta) mu(delta)
inline double g1(double t) {
    if (t == -1.0) return 1.0;
    if (!std::isfinite(t)) return t;
    const DD mu = mu_dd(t);
    const DD v = dd::two_prod(t, t) - mu - mu * t;  // (1 + t) mu = mu + t mu
    return v.hi + v.lo;
}

// psi(y) = (e^y - 1 - y) / y
inline double psi(double y) {
    if (std::fabs(y) < 0.5) {
        double term = y / 2.0, s = term;
        for (int k = 3; k < 40; ++k) {
            term *= y / k;
            s += term;
            if (std::fabs(term) <= 0x1p-60 * std::fabs(s)) break;
        }
        return s;
    }
    return (std::expm1(y) - y) / y;
}

// g_a(delta) for delta > -1 (O > 0), a != 1
inline double ga(double a, double t) {
    if (t == -1.0) return a > 0.0 ? 1.0 / a : std::numeric_limits<double>::infinity();
    if (a < 0.5) {
        // g_a(t) = (1 + t) g_(1-a)(-t / (1 + t))
        return (1.0 + t) * ga(1.0 - a, -t / (1.0 + t));
    }
    if (a == 1.0) return g1(t);
    if (std::fabs(t) <= 0x1p-4 && std::fabs(a) <= 16.0) {
        // sum_{k >= 2} c_k t^k, c_2 = 1/2, c_(k+1) = c_k (a - k) / (k + 1)
        double c = 0.5, p = t * t, s = c * p;
        for (int k = 2; k < 200; ++k) {
            c *= (a - k) / (k + 1.0);
            p *= t;
            const double term = c * p;
            s += term;
            if (std::fabs(term) <= 0x1p-60 * std::fabs(s)) break;
        }
        return s;
    }
    const double L = std::log1p(t);
    return (g1(t) + (1.0 + t) * L * psi((a - 1.0) * L)) / a;
}

// scipy's term for the pairs a rearranged form does not cover (E <= 0 or O < 0)
inline double xlogy(double x, double y) { return (x == 0.0 && y == y) ? 0.0 : x * std::log(y); }
inline double scipy_term(double lam, double o, double e) {
    if (lam == 1.0) return (o - e) * (o - e) / e;
    if (lam == 0.0) return 2.0 * xlogy(o, o / e);
    if (lam == -1.0) return 2.0 * xlogy(e, e / o);
    return o * (std::pow(o / e, lam) - 1.0) / (0.5 * lam * (lam + 1.0));
}

// The statistic of cells with expected counts E (double-doubles, > 0) and observed counts
// E + d (d the exact difference O - E as a double-double, O >= 0): the rearranged sum above.
struct Divergence {
    double lam;
    Acc pos, lin;
    explicit Divergence(double lambda) : lam(lambda) {}
    void add(DD E, DD d) {
        if (lam == 1.0) {
            pos.add((d * d) / E);
            return;
        }
        const DD q = d / E;
        const double t = q.hi + q.lo;
        const double aa = lam + 1.0;
        if (lam == 0.0) {
            pos.add(E * (2.0 * g1(t)));
        } else if (lam == -1.0) {
            pos.add(E * (2.0 * ga(0.0, t)));
        } else {
            pos.add(E * (2.0 * ga(aa, t)));
        }
        lin.add(d);
    }
    double value() const {
        const double coef = lam == -1.0 ? -2.0 : (lam == 1.0 ? 0.0 : 2.0 / (lam + 1.0));
        Acc tot = pos;
        tot.add(lin.value() * coef);
        return to_double(tot.value());
    }
};

}  // namespace sd::htest::div
