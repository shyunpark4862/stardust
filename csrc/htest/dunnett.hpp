// The multivariate t tail of dunnett (milestone H8, docs/design/htest.md §3.9; H108: deterministic,
// no rng). Dunnett's statistics T_j = Z_j / R (j = 1..d, the treatments against one control) have
// the one-factor correlation corr(Z_i, Z_j) = l_i l_j, l_j = sqrt(n_j / (n_j + n_0)): every
// difference shares the control's mean, so Z_j = l_j Z_0 + s_j E_j (s_j = sqrt(n_0 / (n_j + n_0)),
// Z_0 and E_j independent standard normals). Given Z_0 = z the Z_j are independent, and
//     P(max_j |Z_j| > c) = int phi(z) [1 - prod_j (1 - u_j(z))] dz,
//     u_j(z) = Phi(-(c - l_j z) / s_j) + Phi(-(c + l_j z) / s_j)          (two-sided),
//     u_j(z) = Phi(-(c - l_j z) / s_j)                                    (one-sided, max Z_j > c),
// with 1 - prod (1 - u_j) = -expm1(sum log1p(-u_j)): positive, no cancellation, so the p-value
// keeps its relative accuracy in the tail (1 - cdf would not). The t's mixture over R is sd.stats'
// (stats/mvcdf.hpp t_mixture's integrand on srange::lattice_mixture's halved trapezoid lattice), as
// for the studentized range (htest/studentized_range.hpp).
//
// This is the one-factor form of the multivariate t cdf that sd.stats' MultivariateT.cdf takes
// through Genz's quasi-Monte Carlo for d >= 4 (atol 1e-6) and exact formulas for d <= 3 (with
// lower limits -inf only): here every d is a two-level quadrature to ~1e-15, and the two-sided
// rectangle needs no orthant sums. The integral over z runs inside [-|c| - 12, |c| + 12] (the
// rest is below 2 Phi(-|c| - 12) against the tail's Phi(-|c|)), by the trapezoid lattice of the
// studentized range (srange::lattice_sum) walked out past the peaks z = +-l_j c.
#pragma once

#include "htest/studentized_range.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sd::htest::dunnett {

namespace mv = stats::mvcdf;
namespace dd = special::dd;
using dd::DD;
using srange::kInf;
using srange::kNaN;

// The same by adaptive Gauss-Kronrod panels cut at the peaks: for unequal sizes where some u_j turns
// much faster than phi (s_j / l_j below 0.08 two-sided, 0.1 one-sided: a control of under 1/150
// of a treatment), where one lattice step for the whole line would be so narrow that the panels
// cost less (measured from s / l = 0.014 to 0.7, d = 1 to 5; both within 4e-15 of each other).
inline double normal_tail_adaptive(const double* l, const double* s, int d, double c, bool two) {
    if (c != c) return kNaN;
    if (two && !(c > 0.0)) return 1.0;
    if (std::isinf(c)) return c > 0.0 ? 0.0 : 1.0;
    auto f = [&](const double* z, double* o, int n) {
        double x[21], u[21], acc[21];
        for (int i = 0; i < n; ++i) acc[i] = 0.0;
        for (int j = 0; j < d; ++j) {
            const double lj = l[j], sj = s[j];
            for (int i = 0; i < n; ++i) x[i] = -(c - lj * z[i]) / sj;
            srange::ndtr(x, u, n);
            if (two) {
                double v[21];
                for (int i = 0; i < n; ++i) x[i] = -(c + lj * z[i]) / sj;
                srange::ndtr(x, v, n);
                for (int i = 0; i < n; ++i) u[i] += v[i];
            }
            for (int i = 0; i < n; ++i) acc[i] += std::log1p(-std::min(1.0, u[i]));
        }
        for (int i = 0; i < n; ++i) {
            const DD e = dd::two_prod(z[i], z[i]) * -0.5 - dd::kLogSqrt2Pi;
            o[i] = special::distr_detail::exp_dd(e) * -std::expm1(acc[i]);
        }
    };
    const double a = std::fabs(c), lo = -a - 12.0, hi = a + 12.0;
    std::vector<double> cuts;
    cuts.push_back(0.0);
    for (int j = 0; j < d && j < 16; ++j) {
        cuts.push_back(l[j] * c);
        if (two) cuts.push_back(-l[j] * c);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    return std::min(1.0, mv::integrate_at(f, lo, hi, cuts.data(), static_cast<int>(cuts.size()), 0.0));
}

// P(max_j |Z_j| > c) (two) or P(max_j Z_j > c), l[0, d) and s[0, d) as above, by the trapezoid
// lattice (srange::lattice_sum). The integrand peaks around z = l_j c (and phi's 0): the walk
// stops only outside them. Two-sided it is even in z: the lattice from 0, the nodes past 0 twice.
inline double normal_tail(const double* l, const double* s, int d, double c, bool two) {
    if (c != c) return kNaN;
    if (two && !(c > 0.0)) return 1.0;
    if (std::isinf(c)) return c > 0.0 ? 0.0 : 1.0;
    auto f = [&](const double* z, double* o, int n) {
        double x[srange::kBlock], u[srange::kBlock], acc[srange::kBlock];
        for (int i = 0; i < n; ++i) acc[i] = 0.0;
        for (int j = 0; j < d; ++j) {
            const double lj = l[j], sj = s[j];
            for (int i = 0; i < n; ++i) x[i] = -(c - lj * z[i]) / sj;
            srange::ndtr(x, u, n);
            if (two) {
                double v[srange::kBlock];
                for (int i = 0; i < n; ++i) x[i] = -(c + lj * z[i]) / sj;
                srange::ndtr(x, v, n);
                for (int i = 0; i < n; ++i) u[i] += v[i];
            }
            for (int i = 0; i < n; ++i) acc[i] += std::log1p(-std::min(1.0, u[i]));
        }
        for (int i = 0; i < n; ++i) {
            const DD e = dd::two_prod(z[i], z[i]) * -0.5 - dd::kLogSqrt2Pi;
            const double v = special::distr_detail::exp_dd(e) * -std::expm1(acc[i]);
            o[i] = two && z[i] != 0.0 ? 2.0 * v : v;
        }
    };
    const double a = std::fabs(c);
    double zl = 0.0, zr = 0.0;
    for (int j = 0; j < d; ++j) zl = std::min(zl, l[j] * c), zr = std::max(zr, l[j] * c);
    double sl = 1.0;
    for (int j = 0; j < d; ++j)
        if (l[j] > 0.0) sl = std::min(sl, s[j] / l[j]);
    if (sl < (two ? 0.08 : 0.1)) return normal_tail_adaptive(l, s, d, c, two);
    const double h = srange::lattice_step(sl * srange::range_step(static_cast<double>(two ? 2 * d + 1 : d + 1)));
    // from the farthest peak, on a lattice through 0 (two-sided: the fold's center)
    const double z0 = std::nearbyint(zr / h) * h;
    if (two) return std::min(1.0, srange::lattice_sum(f, z0, h, 0.0, a + 12.0, zl, zr));
    return std::min(1.0, srange::lattice_sum(f, z0, h, -a - 12.0, a + 12.0, zl, zr));
}

// The t's tail: E[normal_tail(R c)] over R = sqrt(chi^2_nu / nu) (nu = inf: the normal's).
inline double tail(const double* l, const double* s, int d, double c, double nu, bool two) {
    if (c != c || nu != nu || !(nu > 0.0)) return kNaN;
    if (two && !(c > 0.0)) return 1.0;
    if (std::isinf(c)) return c > 0.0 ? 0.0 : 1.0;
    if (std::isinf(nu)) return normal_tail(l, s, d, c, two);
    // g(0): 1 (two-sided), else 1 - the orthant P(all Z_j <= 0); |g(R c) - g(0)| is at most
    // R |c| / sqrt(2 pi) per coordinate (twice that two-sided, where one coordinate bounds it)
    const double p0 = two ? 1.0 : normal_tail(l, s, d, 0.0, false);
    const double habs = two ? 2.0 * c : static_cast<double>(d) * std::fabs(c);
    const double hs[1] = {c};
    // g falls like exp(-R^2 c^2 / 2) for c > 0 (rises for a one-sided c < 0)
    const double beta_q2 = c > 0.0 ? 0.5 * c * c : 0.0;
    return srange::lattice_mixture(nu, p0, habs, beta_q2, hs, 1, [&](const mv::MixNode& nd) {
        mv::MixTerm t;
        const double w = nd.times(c);
        t.f = normal_tail(l, s, d, w, two);
        // d log g / dv ~ -(w / 2) phi(w) / Phi(-w), the marginal tail's rate (first order only)
        t.dlog = w > 0.0 ? -0.5 * w * srange::hazard(w) : 0.0;
        return t;
    });
}

// The critical value c with tail(c) = a: between one comparison's t quantile (the tail is at
// least one coordinate's) and Bonferroni's over the d comparisons.
inline double isf(const double* l, const double* s, int d, double a, double nu, bool two) {
    if (a != a || nu != nu || !(nu > 0.0) || !(a > 0.0 && a < 1.0)) return kNaN;
    const double one = two ? 0.5 * a : a;
    double lo = srange::t_isf(one, nu), hi = srange::t_isf(one / static_cast<double>(d), nu);
    if (!(lo < hi)) return lo;  // d = 1: the t itself
    return srange::solve_tail([&](double c) { return tail(l, s, d, c, nu, two); }, a, lo, hi);
}

}  // namespace sd::htest::dunnett
