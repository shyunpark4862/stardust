// The tests of goodness of fit to a given distribution (milestone H2, docs/design/htest.md §3.3).
//
// The Python side (stardust.htest._gof) checks the arguments and arranges every sample as a 2-D
// float64 array, one lane (one test of a batch) per row (sorted, with the cdf's values beside it,
// for ks_1samp and cramervonmises); each kernel returns the per-lane results as 1-D arrays. Lanes
// are independent: the same lane gives the same bits alone or in a batch, on any thread count.
//
//   power_divergence  chisquare and the Cressie-Read family: statistic, p-value, df, and the
//                     relative difference of the sums (scipy's sum_check)
//   jarque_bera       statistic, p-value (htest/normality.hpp; sd.regression's too: jb_of_sums)
//   dagostino         skewtest (kind 0), kurtosistest (1), normaltest (2): statistic, p-value
//   shapiro           W, p-value (Royston's AS R94)
//   anderson          A^2, the interpolated p-value
//   ks                D, p-value, the location and sign of D
//   cvm               W^2, p-value
//
// The p-values are sd.stats' distributions (Normal, ChiSquared) called through their own methods,
// and the null distributions of the Kolmogorov-Smirnov and Cramér-von Mises statistics
// (htest/ks.hpp, htest/cvm.hpp). Degenerate lanes give NaN without warnings (H11): too few
// observations, a constant sample (sd.stats skew's rule, H20), a NaN under 'propagate'.
#include "htest/gof.hpp"

#include "htest/cvm.hpp"
#include "htest/divergence.hpp"
#include "htest/ks.hpp"
#include "htest/lanes.hpp"
#include "htest/mle.hpp"
#include "htest/moments.hpp"
#include "htest/normality.hpp"
#include "special/dd.hpp"
#include "special/kernels/chunked.hpp"
#include "special/ops/erf.hpp"
#include "stats/dist/exponential.hpp"
#include "stats/dist/extreme.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;

double chi2_sf(double x, double df) { return stats::method_of<stats::ChiSquared, m::Ccdf>(x, df); }
double norm_sf(double z) { return stats::method_of<stats::Normal, m::Ccdf>(z, 0.0, 1.0); }
double norm_cdf(double z) { return stats::method_of<stats::Normal, m::Cdf>(z, 0.0, 1.0); }

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };

// scipy's _get_pvalue for the standard normal (symmetric)
double normal_p(double z, int alt) {
    if (alt == kTwoSided) return z == z ? std::min(1.0, 2.0 * norm_sf(std::fabs(z))) : z;
    return alt == kLess ? norm_cdf(z) : norm_sf(z);
}

// A lane's values without its NaNs; true when a NaN met under 'propagate' (the lane is NaN).
bool lane_values(const Rows& a, std::int64_t r, bool omit, std::vector<double>& v) {
    if (gather(a, r, omit, v)) return true;
    return false;
}

// ---- power_divergence --------------------------------------------------------------------
//
// With delta = (O - E)/E (O - E exact as a double-double) and a = lambda + 1, every Cressie-Read
// statistic is 2 sum E g_a(delta) + (2/a) sum (O - E), with
//     g_a(delta) = ((1 + delta)^a - 1 - a delta) / (a (a - 1)) >= 0
// (g_1 = (1 + delta) log(1 + delta) - delta: the G test; the limit a = 0 is the modified one,
// 2 sum E log(E/O) = 2 sum E g_0 - 2 sum (O - E)). The terms of the first sum are never negative,
// so a statistic near 0 (O close to E) keeps its digits: scipy's direct sum of O log(O/E) cancels
// to the first order (9.9e-8 relative at O - E = 1e-3, E = 1e6). Pearson's (lambda = 1) is
// scipy's sum (O - E)^2 / E. The terms g_a are htest/divergence.hpp's.

using div::g1;
using div::ga;
using div::scipy_term;

struct PdLane {
    double stat = kNaN, p = kNaN, df = kNaN, rel = 0.0;
};

PdLane pd_lane(const Rows& o, const Rows* e, std::int64_t r, double lam, double ddof, bool omit) {
    namespace dd = special::dd;
    PdLane out;
    std::vector<double> ov, ev;
    for (std::int64_t j = 0; j < o.n; ++j) {
        const double x = o.at(r, j), y = e ? e->at(r, j) : 0.0;
        if (x != x || y != y) {
            if (omit) continue;
            return out;
        }
        ov.push_back(x);
        if (e) ev.push_back(y);
    }
    const auto k = static_cast<std::int64_t>(ov.size());
    if (k == 0) return out;
    out.df = static_cast<double>(k) - 1.0 - ddof;
    Acc so;
    for (double x : ov) so.add(x);
    DD emean{0.0, 0.0};
    if (e) {
        Acc se;
        for (double y : ev) se.add(y);
        const double a = to_double(so.value()), b = to_double(se.value());
        out.rel = std::fabs(a - b) / std::min(a, b);
    } else {
        emean = so.value() / static_cast<double>(k);
    }
    Acc pos, lin, raw;
    const double aa = lam + 1.0;
    // cell i's terms: into raw (scipy's term), pos alone (Pearson's), or pos and lin
    struct Term {
        DD a, d;
        int kind;
    };
    auto term = [&](std::int64_t i) -> Term {
        const double x = ov[static_cast<std::size_t>(i)];
        const DD E = e ? DD{ev[static_cast<std::size_t>(i)], 0.0} : emean;
        if (!(E.hi > 0.0) || x < 0.0 || !std::isfinite(E.hi) || !std::isfinite(x))
            return {DD{scipy_term(lam, x, E.hi), 0.0}, DD{0.0, 0.0}, 0};
        const DD d = dd::two_sum(x, -E.hi) - E.lo;  // O - E
        if (lam == 1.0) return {(d * d) / E.hi, d, 1};
        const double t = (d.hi + d.lo) / E.hi;
        const double g = lam == 0.0 ? g1(t) : (lam == -1.0 ? ga(0.0, t) : ga(aa, t));
        return {DD{2.0 * E.hi * g, 0.0}, d, 2};
    };
    auto add = [&](const Term& t) {
        if (t.kind == 0) raw.add(t.a.hi);
        else if (t.kind == 1) pos.add(t.a);
        else pos.add(t.a.hi), lin.add(t.d);
    };
    // the terms on the pool for a long lane alone, added in order (the same bits)
    const std::int64_t cost = lam == 1.0 ? 8 : 48;
    if (k * cost < kParallelThreshold || k < 2 * kTask || !parallel::available()) {
        for (std::int64_t i = 0; i < k; ++i) {  // (term and add in one, as written out)
            const double x = ov[static_cast<std::size_t>(i)];
            const DD E = e ? DD{ev[static_cast<std::size_t>(i)], 0.0} : emean;
            if (!(E.hi > 0.0) || x < 0.0 || !std::isfinite(E.hi) || !std::isfinite(x)) {
                raw.add(scipy_term(lam, x, E.hi));
                continue;
            }
            const DD d = dd::two_sum(x, -E.hi) - E.lo;  // O - E
            if (lam == 1.0) {
                pos.add((d * d) / E.hi);
                continue;
            }
            const double t = (d.hi + d.lo) / E.hi;
            pos.add(2.0 * E.hi * (lam == 0.0 ? g1(t) : (lam == -1.0 ? ga(0.0, t) : ga(aa, t))));
            lin.add(d);
        }
    } else {
        std::vector<Term> ts(static_cast<std::size_t>(k));
        for_pieces(k, kTask, k * cost, [&](std::int64_t, std::int64_t b, std::int64_t f) {
            for (std::int64_t i = b; i < f; ++i) ts[static_cast<std::size_t>(i)] = term(i);
        });
        for (const Term& t : ts) add(t);
    }
    const double coef = lam == -1.0 ? -2.0 : (lam == 1.0 ? 0.0 : 2.0 / aa);
    Acc tot = pos;
    tot.add(lin.value() * coef);
    tot.add(raw.value());
    out.stat = to_double(tot.value());
    out.p = (out.df > 0.0 && out.stat == out.stat) ? chi2_sf(out.stat, out.df) : kNaN;
    return out;
}

nb::tuple power_divergence(nb::handle obs, nb::handle exp, double lam, double ddof, bool omit) {
    const Rows o(rows_arg(obs));
    Rows e;
    const bool has_e = !exp.is_none();
    if (has_e) e = Rows(rows_arg(exp));
    Outs out(4, o.lanes);
    each_lane(o.lanes, 8 * o.n, [&](std::int64_t r) {
        const PdLane l = pd_lane(o, has_e ? &e : nullptr, r, lam, ddof, omit);
        out.p[0][r] = l.stat, out.p[1][r] = l.p, out.p[2][r] = l.df, out.p[3][r] = l.rel;
    });
    return out.tuple();
}

// ---- the moment tests ----------------------------------------------------------------------

// The central sums of one lane (htest/normality.hpp); false when the lane is NaN.
bool lane_sums(
    const Rows& a, std::int64_t r, bool omit, std::vector<double>& v, std::vector<double>& part, CentralSums& cs
) {
    if (lane_values(a, r, omit, v)) return false;
    const auto n = static_cast<std::int64_t>(v.size());
    part.assign(static_cast<std::size_t>(8 * std::max<std::int64_t>(1, (n + kSumChunk - 1) / kSumChunk)), 0.0);
    cs = central_sums(v.data(), n, part.data());
    return true;
}

nb::tuple jarque_bera(nb::handle x, bool omit) {
    const Rows a(rows_arg(x));
    Outs out(2, a.lanes);
    each_lane(a.lanes, 16 * a.n, [&](std::int64_t r) {
        std::vector<double> v, part;
        CentralSums cs;
        double stat = kNaN, p = kNaN;
        if (lane_sums(a, r, omit, v, part, cs)) {
            const JarqueBera jb =
                jarque_bera_of(static_cast<double>(v.size()), cs.s2.hi, cs.s3.hi, cs.s4.hi, cs.mean.hi);
            stat = jb.stat, p = jb.p;
        }
        out.p[0][r] = stat, out.p[1][r] = p;
    });
    return out.tuple();
}

// Jarque-Bera from sums already formed (sd.regression's resid_stats): 1-D arrays of one length.
nb::tuple jb_of_sums(nb::handle n, nb::handle s2, nb::handle s3, nb::handle s4, nb::handle mean) {
    const NDArray& na = require_ndarray(n);
    const std::int64_t k = na.ndim == 1 ? na.shape[0] : -1;
    const NDArray &a = vec_arg(n, k), &b = vec_arg(s2, k), &c = vec_arg(s3, k), &d = vec_arg(s4, k),
                  &e = vec_arg(mean, k);
    Outs out(2, k);
    for (std::int64_t i = 0; i < k; ++i) {
        const JarqueBera jb = jarque_bera_of(vec_at(a, i), vec_at(b, i), vec_at(c, i), vec_at(d, i), vec_at(e, i));
        out.p[0][i] = jb.stat, out.p[1][i] = jb.p;
    }
    return out.tuple();
}

// D'Agostino's skewness test: Z = delta asinh(y / alpha) (scipy's log(y/alpha + sqrt(...)), which
// loses digits for y < 0 and, as written in scipy 1.18.1, puts 1 for y = 0).
double skewtest_z(double n, double b) {
    const double y = b * std::sqrt((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0)));
    // beta2 - 3 = 36 (n^3 - 5n^2 - 19n + 35) / ((n - 2)(n + 5)(n + 7)(n + 9)): W^2 - 1 without the
    // cancellation of -1 + sqrt(2 (beta2 - 1)) against 1
    const double b3 = 36.0 * (((n - 5.0) * n - 19.0) * n + 35.0) / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    const double w2m1 = 2.0 * b3 / (std::sqrt(2.0 * (b3 + 2.0)) + 2.0);  // W^2 - 1
    const double delta = 1.0 / std::sqrt(0.5 * std::log1p(w2m1));
    const double alpha = std::sqrt(2.0 / w2m1);
    return delta * std::asinh(y / alpha);
}

// D'Agostino's kurtosis test (Anscombe and Glynn): Z = (term1 - term2) / sqrt(2/(9A)) with
// term1 - term2 formed from term1^3 - (1 - 2/A)/(1 + x c) as one difference, x from b2 - E in
// double-double.
double kurtosistest_z(double n, special::dd::DD b2) {
    namespace dd = special::dd;
    const DD E = DD{3.0 * (n - 1.0), 0.0} / (n + 1.0);
    const double varb2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    const DD diff = b2 - E;
    const double x = (diff.hi + diff.lo) / std::sqrt(varb2);
    const double sb1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0)) *
                       std::sqrt(6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0)));
    const double A = 6.0 + 8.0 / sb1 * (2.0 / sb1 + std::sqrt(1.0 + 4.0 / (sb1 * sb1)));
    const double t = 2.0 / (9.0 * A);
    const double c = std::sqrt(2.0 / (A - 4.0));
    const double denom = 1.0 + x * c;
    if (denom == 0.0 || denom != denom) return kNaN;
    const double term1 = 1.0 - t;
    const double term2 = std::cbrt((1.0 - 2.0 / A) / denom);
    const double t13 = term1 * term1 * term1;
    // term1^3 (1 + x c) - (1 - 9t) = 6t + 3t^2 - t^3 + term1^3 x c
    const double num = ((6.0 + 3.0 * t - t * t) * t + t13 * x * c) / denom;
    const double diff13 = num / (term1 * term1 + term1 * term2 + term2 * term2);
    return diff13 / std::sqrt(t);
}

nb::tuple dagostino(int kind, nb::handle x, int alt, bool omit) {
    namespace dd = special::dd;
    const Rows a(rows_arg(x));
    Outs out(2, a.lanes);
    each_lane(a.lanes, 16 * a.n, [&](std::int64_t r) {
        std::vector<double> v, part;
        CentralSums cs;
        double stat = kNaN, p = kNaN;
        if (lane_sums(a, r, omit, v, part, cs)) {
            const auto n = static_cast<double>(v.size());
            const double mu2 = cs.s2.hi / n;
            const double need = kind == 1 ? 5.0 : 8.0;
            if (n >= need && !zero_variance(mu2, cs.mean.hi) && std::isfinite(mu2)) {
                double zs = 0.0, zk = 0.0;
                if (kind != 1) {
                    const double b = (cs.s3.hi / n) / (mu2 * std::sqrt(mu2));
                    zs = skewtest_z(n, b);
                }
                if (kind != 0) {
                    const DD b2 = cs.s4 * n / (cs.s2 * cs.s2);
                    zk = kurtosistest_z(n, b2);
                }
                if (kind == 0) {
                    stat = zs, p = normal_p(zs, alt);
                } else if (kind == 1) {
                    stat = zk, p = normal_p(zk, alt);
                } else {
                    stat = zs * zs + zk * zk;
                    p = chi2_sf(stat, 2.0);
                }
            }
        }
        out.p[0][r] = stat, out.p[1][r] = p;
    });
    return out.tuple();
}

// ---- shapiro: AS R94 (Royston 1995), as scipy 1.18.1 and R's swilk -----------------------------
//
// The coefficients: m_i = ndtri((i - 3/8) / (n + 1/4)) for the lower half (the upper half by
// symmetry: m_(n+1-i) = -m_i, so no ndtri of a p near 1), Royston's polynomial corrections for the
// two (n <= 5: one) outer coefficients and the others m_i / sqrt(phi). W = (sum a_i x_(i))^2 / SST
// and 1 - W = (SST - (sum a_i x_(i))^2) / SST, both from double-double sums (scipy and R form W in
// double and then 1 - W, which for large n loses the digits of log(1 - W): 1.6e-12 at n = 20000).

struct SwLane {
    double w = kNaN, p = kNaN;
};

SwLane shapiro_lane(std::vector<double>& x) {
    namespace dd = special::dd;
    SwLane o;
    const auto n = static_cast<std::int64_t>(x.size());
    if (n < 3) return o;
    sort_values(x);
    const Moments mo = moments_of(x.data(), nullptr, n);
    if (mo.constant() || !std::isfinite(mo.ss.hi)) return o;
    const double nn = static_cast<double>(n);
    const std::int64_t half = n / 2;
    std::vector<double> a(static_cast<std::size_t>(n), 0.0);
    if (n == 3) {
        const double c = std::sqrt(0.5);
        a[0] = -c, a[2] = c;
    } else {
        std::vector<double> pr(static_cast<std::size_t>(half)), mm(static_cast<std::size_t>(half));
        for (std::int64_t i = 0; i < half; ++i)
            pr[static_cast<std::size_t>(i)] = (static_cast<double>(i + 1) - 0.375) / (nn + 0.25);
        for (std::int64_t i0 = 0; i0 < half; i0 += special::kChunk) {
            const int len = static_cast<int>(std::min<std::int64_t>(special::kChunk, half - i0));
            special::Ndtri::chunk(pr.data() + i0, mm.data() + i0, len);
        }
        dd::Sum s2;
        for (double v : mm) s2.add(dd::two_prod(v, v));
        const double mtm = 2.0 * (s2.value().hi + s2.value().lo);  // sum of m_i^2 (the middle is 0)
        const double u = 1.0 / std::sqrt(nn);
        const double mn = -mm[0], cn = mn / std::sqrt(mtm);  // m_n = -m_1 > 0
        const double an = cn + 0.221157 * u - 0.147981 * u * u - 2.071190 * u * u * u + 4.434685 * u * u * u * u -
                          2.706056 * u * u * u * u * u;
        double phi;
        double anm1 = 0.0;
        if (n > 5) {
            const double mnm1 = -mm[1], cnm1 = mnm1 / std::sqrt(mtm);
            anm1 = cnm1 + 0.042981 * u - 0.293762 * u * u - 1.752461 * u * u * u + 5.682633 * u * u * u * u -
                   3.582633 * u * u * u * u * u;
            phi = (mtm - 2.0 * mn * mn - 2.0 * mnm1 * mnm1) / (1.0 - 2.0 * an * an - 2.0 * anm1 * anm1);
        } else {
            phi = (mtm - 2.0 * mn * mn) / (1.0 - 2.0 * an * an);
        }
        const double rs = 1.0 / std::sqrt(phi);
        for (std::int64_t i = 0; i < half; ++i) {
            const double v = mm[static_cast<std::size_t>(i)] * rs;
            a[static_cast<std::size_t>(i)] = v;
            a[static_cast<std::size_t>(n - 1 - i)] = -v;
        }
        a[0] = -an, a[static_cast<std::size_t>(n - 1)] = an;
        if (n > 5) a[1] = -anm1, a[static_cast<std::size_t>(n - 2)] = anm1;
    }
    // S = sum a_i x_(i), exact products in double-double (sum a_i = 0: no shift needed). The exact
    // coefficients have sum a_i^2 = 1 (phi is chosen so); the rounded ones are off by a common
    // factor of a few ulps, which would move 1 - W by that factor over 1 - W (1e-12 at n = 20000):
    // W is taken as S^2 / (SST sum a_i^2).
    dd::Sum s, aa;
    for (std::int64_t i = 0; i < n; ++i) {
        const double ai = a[static_cast<std::size_t>(i)];
        s.add(dd::two_prod(ai, x[static_cast<std::size_t>(i)]));
        aa.add(dd::two_prod(ai, ai));
    }
    const DD S = s.value();
    const DD S2 = S * S;
    const DD sst = mo.ss * aa.value();
    const DD w1 = (sst - S2) / sst;  // 1 - W
    const DD W = S2 / sst;
    o.w = W.hi + W.lo;
    const double one_m_w = w1.hi + w1.lo;
    if (n == 3) {
        // p = 6/pi (asin sqrt W - pi/3) = 6/pi asin((W - 3/4) / (sqrt(W)/2 + sqrt(3)/2 sqrt(1 - W))),
        // W clipped to [3/4, 1] as scipy
        const double wc = std::clamp(o.w, 0.75, 1.0);
        o.w = wc;
        const DD wm = (wc == o.w && wc == W.hi + W.lo) ? W - 0.75 : DD{wc - 0.75, 0.0};
        const double om = std::max(0.0, wc == 1.0 ? 0.0 : one_m_w);
        const double arg = (wm.hi + wm.lo) / (std::sqrt(wc) / 2.0 + std::sqrt(3.0) / 2.0 * std::sqrt(om));
        o.p = std::clamp(6.0 / ks::kPi * std::asin(std::min(1.0, arg)), 0.0, 1.0);
        return o;
    }
    double gw, mu, sigma;
    if (n <= 11) {
        const double g = -2.273 + 0.459 * nn;
        mu = 0.5440 - 0.39978 * nn + 0.025054 * nn * nn - 0.0006714 * nn * nn * nn;
        sigma = std::exp(1.3822 - 0.77857 * nn + 0.062767 * nn * nn - 0.0020322 * nn * nn * nn);
        const double y = std::log(one_m_w);
        if (!(g - y > 0.0)) {
            o.p = y == y ? 0.0 : kNaN;  // the limit of -log(gamma - y) as y -> gamma (scipy: NaN, R: 1e-99)
            return o;
        }
        gw = -std::log(g - y);
    } else {
        const double ln = std::log(nn);
        mu = -1.5861 - 0.31082 * ln - 0.083751 * ln * ln + 0.0038915 * ln * ln * ln;
        sigma = std::exp(-0.4803 - 0.082676 * ln + 0.0030302 * ln * ln);
        gw = std::log(one_m_w);
    }
    o.p = norm_sf((gw - mu) / sigma);
    return o;
}

nb::tuple shapiro(nb::handle x, bool omit) {
    const Rows a(rows_arg(x));
    Outs out(2, a.lanes);
    each_lane(a.lanes, 64 * a.n, [&](std::int64_t r) {
        std::vector<double> v;
        SwLane l;
        if (!lane_values(a, r, omit, v)) l = shapiro_lane(v);
        out.p[0][r] = l.w, out.p[1][r] = l.p;
    });
    return out.tuple();
}

// ---- anderson ------------------------------------------------------------------------------
//
// A^2 = -N - sum (2i - 1)/N (log F(w_(i)) + log(1 - F(w_(N+1-i)))) with the fitted distribution's
// logcdf and logccdf (sd.stats' Normal, Exponential, Logistic, Gumbel), so a tail observation's
// log(1 - F) is not formed from 1 - F. The fits: the normal's mean and standard deviation (n - 1),
// the exponential's mean (location 0), and for the logistic and the Gumbel the maximum likelihood
// estimates solved to the last digit (scipy: fsolve with xtol 1e-5 for the logistic, root_scalar
// for the Gumbel). The p-value interpolates scipy's tables of critical values (Stephens) as
// scipy's np.interp does, so it lies in the table's range of significance levels.

enum Dist { kNormalD = 0, kExponentialD = 1, kLogisticD = 2, kGumbelD = 3, kWeibullD = 4 };

// the logistic's and the Gumbel's maximum likelihood of standardized values (htest/mle.hpp, shared
// with goodness_of_fit)
using mle::gumbel_fit;
using mle::logistic_fit;

double round3(double v) { return std::nearbyint(v * 1000.0) / 1000.0; }

// numpy's interp(x, xp, fp) at one point, xp increasing
double interp(double x, const double* xp, const double* fp, int k) {
    if (x != x) return x;
    if (x < xp[0]) return fp[0];
    if (x > xp[k - 1]) return fp[k - 1];
    if (x == xp[k - 1]) return fp[k - 1];
    int j = 0;
    while (j + 1 < k && xp[j + 1] <= x) ++j;
    if (x == xp[j]) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    return slope * (x - xp[j]) + fp[j];
}

// the three-parameter Weibull's (Lockhart and Stephens 1994, table 1; scipy's _Avals_weibull):
// the critical values at c = 1/m = 0, 0.05, ..., 0.5 interpolated linearly in c (scipy's interp1d,
// the last row beyond 0.5), rounded up (+0.0005, then to 3 decimals), and the p-value interpolated
// in them over the levels 1 - (0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995)
double anderson_weibull_p(double a2, double m) {
    static constexpr double A[11][8] = {{0.292, 0.395, 0.467, 0.522, 0.617, 0.711, 0.836, 0.931},
                                        {0.295, 0.399, 0.471, 0.527, 0.623, 0.719, 0.845, 0.941},
                                        {0.298, 0.403, 0.476, 0.534, 0.631, 0.728, 0.856, 0.954},
                                        {0.301, 0.408, 0.483, 0.541, 0.640, 0.738, 0.869, 0.969},
                                        {0.305, 0.414, 0.490, 0.549, 0.650, 0.751, 0.885, 0.986},
                                        {0.309, 0.421, 0.498, 0.559, 0.662, 0.765, 0.902, 1.007},
                                        {0.314, 0.429, 0.508, 0.570, 0.676, 0.782, 0.923, 1.030},
                                        {0.320, 0.438, 0.519, 0.583, 0.692, 0.802, 0.947, 1.057},
                                        {0.327, 0.448, 0.532, 0.598, 0.711, 0.824, 0.974, 1.089},
                                        {0.334, 0.469, 0.547, 0.615, 0.732, 0.850, 1.006, 1.125},
                                        {0.342, 0.472, 0.563, 0.636, 0.757, 0.879, 1.043, 1.167}};
    static constexpr double sig[] = {0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995};
    const double c = 1.0 / m;
    double crit[8], lev[8];
    int hi = 1;
    while (hi < 10 && static_cast<double>(hi) * 0.05 < c) ++hi;  // searchsorted(x, c, 'left'), clipped to 1..10
    const double xlo = static_cast<double>(hi - 1) * 0.05, xhi = static_cast<double>(hi) * 0.05;
    for (int q = 0; q < 8; ++q) {
        double v;
        if (c > 0.5) {
            v = A[10][q];
        } else {
            const double slope = (A[hi][q] - A[hi - 1][q]) / (xhi - xlo);
            v = slope * (c - xlo) + A[hi - 1][q];
        }
        crit[q] = round3(v + 0.0005);
        lev[q] = 1.0 - sig[q];
    }
    return interp(a2, crit, lev, 8);
}

double anderson_p(int dist, double a2, double N) {
    static constexpr double norm[] = {0.561, 0.631, 0.752, 0.873, 1.035};
    static constexpr double expo[] = {0.916, 1.062, 1.321, 1.591, 1.959};
    static constexpr double logi[] = {0.426, 0.563, 0.660, 0.769, 0.906, 1.010};
    static constexpr double gumb[] = {0.474, 0.637, 0.757, 0.877, 1.038};
    static constexpr double sig5[] = {15.0, 10.0, 5.0, 2.5, 1.0};
    static constexpr double sig6[] = {25.0, 10.0, 5.0, 2.5, 1.0, 0.5};
    static constexpr double sigg[] = {25.0, 10.0, 5.0, 2.5, 1.0};
    double crit[6], sig[6];
    int k = 5;
    const double* av = norm;
    const double* sv = sig5;
    double den = 1.0;
    if (dist == kNormalD) {
        den = 1.0 + 0.75 / N + 2.25 / N / N;
    } else if (dist == kExponentialD) {
        av = expo, den = 1.0 + 0.6 / N;
    } else if (dist == kLogisticD) {
        av = logi, sv = sig6, k = 6, den = 1.0 + 0.25 / N;
    } else {
        av = gumb, sv = sigg, den = 1.0 + 0.2 / std::sqrt(N);
    }
    for (int i = 0; i < k; ++i) crit[i] = round3(av[i] / den), sig[i] = sv[i] / 100.0;
    return interp(a2, crit, sig, k);
}

struct AdLane {
    double a2 = kNaN, p = kNaN;
};

AdLane anderson_lane(std::vector<double>& x, int dist) {
    AdLane o;
    const auto n = static_cast<std::int64_t>(x.size());
    if (n < 2) return o;
    sort_values(x);
    const Moments mo = moments_of(x.data(), nullptr, n);
    if (mo.constant() || !std::isfinite(mo.ss.hi)) return o;
    const double N = static_cast<double>(n);
    if (dist == kWeibullD) {  // the three-parameter Weibull (H40): mle.hpp's profile likelihood
        double th[3];
        if (!mle::weibull3_fit(x, th[0], th[1], th[2])) return o;
        const Acc s = ordered_sum(n, 64, [&](std::int64_t i) {
            const double c = (2.0 * static_cast<double>(i + 1) - 1.0) / N;
            return c * (mle::fn<m::Logcdf>(mle::kWeibull3, th, x[static_cast<std::size_t>(i)]) +
                        mle::fn<m::Logccdf>(mle::kWeibull3, th, x[static_cast<std::size_t>(n - 1 - i)]));
        });
        o.a2 = -N - to_double(s.value());
        o.p = anderson_weibull_p(o.a2, th[0]);
        return o;
    }
    std::vector<double> w(static_cast<std::size_t>(n));
    const double sd = std::sqrt(to_double(mo.ss) / (N - 1.0));
    auto standardized = [&](std::vector<double>& z) {
        z.resize(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            const DD d = dd::two_sum(x[static_cast<std::size_t>(i)], -mo.mean.hi) - mo.mean.lo;
            z[static_cast<std::size_t>(i)] = (d.hi + d.lo) / sd;
        }
    };
    if (dist == kNormalD) {
        standardized(w);
    } else if (dist == kExponentialD) {
        for (std::int64_t i = 0; i < n; ++i) {
            const DD q = DD{x[static_cast<std::size_t>(i)], 0.0} / mo.mean;
            w[static_cast<std::size_t>(i)] = q.hi + q.lo;
        }
    } else {
        std::vector<double> z;
        standardized(z);
        if (dist == kLogisticD) {
            double al, be;
            if (!logistic_fit(z, al, be)) return o;
            for (std::int64_t i = 0; i < n; ++i)
                w[static_cast<std::size_t>(i)] = be * z[static_cast<std::size_t>(i)] - al;
        } else {
            double loc, scale;
            if (!gumbel_fit(z, loc, scale)) return o;
            for (std::int64_t i = 0; i < n; ++i)
                w[static_cast<std::size_t>(i)] = (z[static_cast<std::size_t>(i)] - loc) / scale;
        }
    }
    // the standard distribution's logcdf and logccdf of every w, by its chunk kernel (mle.hpp
    // method_array: method_of's bits), on the pool for a long sample
    std::vector<double> L(static_cast<std::size_t>(n)), U(static_cast<std::size_t>(n));
    for_pieces(n, kTask, n * 32, [&](std::int64_t, std::int64_t b, std::int64_t e) {
        const double* wb = w.data() + b;
        double *lb = L.data() + b, *ub = U.data() + b;
        const std::int64_t len = e - b;
        switch (dist) {
            case kNormalD:
                method_array<stats::Normal, m::Logcdf>(wb, len, lb, 0.0, 1.0);
                method_array<stats::Normal, m::Logccdf>(wb, len, ub, 0.0, 1.0);
                break;
            case kExponentialD:
                method_array<stats::Exponential, m::Logcdf>(wb, len, lb, 1.0);
                method_array<stats::Exponential, m::Logccdf>(wb, len, ub, 1.0);
                break;
            case kLogisticD:
                method_array<stats::Logistic, m::Logcdf>(wb, len, lb, 0.0, 1.0);
                method_array<stats::Logistic, m::Logccdf>(wb, len, ub, 0.0, 1.0);
                break;
            default:
                method_array<stats::Gumbel, m::Logcdf>(wb, len, lb, 0.0, 1.0);
                method_array<stats::Gumbel, m::Logccdf>(wb, len, ub, 0.0, 1.0);
        }
    });
    Acc s;
    for (std::int64_t i = 0; i < n; ++i) {
        const double c = (2.0 * static_cast<double>(i + 1) - 1.0) / N;
        s.add(c * (L[static_cast<std::size_t>(i)] + U[static_cast<std::size_t>(n - 1 - i)]));
    }
    o.a2 = -N - to_double(s.value());
    o.p = anderson_p(dist, o.a2, N);
    return o;
}

nb::tuple anderson(nb::handle x, int dist, bool omit) {
    const Rows a(rows_arg(x));
    Outs out(2, a.lanes);
    each_lane(a.lanes, 64 * a.n, [&](std::int64_t r) {
        std::vector<double> v;
        AdLane l;
        if (!lane_values(a, r, omit, v)) l = anderson_lane(v, dist);
        out.p[0][r] = l.a2, out.p[1][r] = l.p;
    });
    return out.tuple();
}

// ---- ks_1samp and cramervonmises -------------------------------------------------------------
//
// x: the lanes sorted (NaNs last), F: the cdf at them. Under 'propagate' a lane with a NaN is NaN.
// A NaN cdf value of a number is NaN too.

// The lane's (x, F) pairs; false when the lane is NaN.
bool sorted_pairs(
    const Rows& x, const Rows& F, std::int64_t r, bool omit, std::vector<double>& xs, std::vector<double>& fs
) {
    xs.clear(), fs.clear();
    for (std::int64_t j = 0; j < x.n; ++j) {
        const double v = x.at(r, j);
        if (v != v) {
            if (omit) continue;
            return false;
        }
        const double f = F.at(r, j);
        if (f != f) return false;
        xs.push_back(v), fs.push_back(f);
    }
    return true;
}

enum KsMethod { kExact = 0, kApprox = 1, kAsymp = 2 };

nb::tuple ks_test(nb::handle xo, nb::handle fo, int alt, int method, bool omit) {
    const Rows x(rows_arg(xo)), F(rows_arg(fo));
    Outs out(4, x.lanes);
    each_lane(x.lanes, 4 * x.n, [&](std::int64_t r) {
        std::vector<double> xs, fs;
        double D = kNaN, p = kNaN, loc = kNaN, sign = 0.0;
        if (sorted_pairs(x, F, r, omit, xs, fs) && !xs.empty()) {
            const auto n = static_cast<std::int64_t>(xs.size());
            const double nn = static_cast<double>(n);
            double dp = -std::numeric_limits<double>::infinity(), dm = dp;
            std::int64_t ip = 0, im = 0;
            for (std::int64_t i = 0; i < n; ++i) {
                const double f = fs[static_cast<std::size_t>(i)];
                const double up = static_cast<double>(i + 1) / nn - f, dn = f - static_cast<double>(i) / nn;
                if (up > dp) dp = up, ip = i;
                if (dn > dm) dm = dn, im = i;
            }
            if (alt == kGreater) {
                D = dp, loc = xs[static_cast<std::size_t>(ip)], sign = 1.0;
                p = ks::smirnov_sf(n, D);
            } else if (alt == kLess) {
                D = dm, loc = xs[static_cast<std::size_t>(im)], sign = -1.0;
                p = ks::smirnov_sf(n, D);
            } else {
                const bool plus = dp > dm;
                D = plus ? dp : dm;
                loc = xs[static_cast<std::size_t>(plus ? ip : im)];
                sign = plus ? 1.0 : -1.0;
                if (method == kExact) {
                    p = ks::kstwo_sf(n, D);
                } else if (method == kAsymp) {
                    p = ks::kolmogorov_sf(D * std::sqrt(nn));
                } else {
                    p = std::min(1.0, 2.0 * ks::smirnov_sf(n, D));
                }
            }
        }
        out.p[0][r] = D, out.p[1][r] = p, out.p[2][r] = loc, out.p[3][r] = sign;
    });
    return out.tuple();
}

nb::tuple cvm_test(nb::handle xo, nb::handle fo, bool omit) {
    const Rows x(rows_arg(xo)), F(rows_arg(fo));
    Outs out(2, x.lanes);
    each_lane(x.lanes, 4 * x.n, [&](std::int64_t r) {
        std::vector<double> xs, fs;
        double W = kNaN, p = kNaN;
        if (sorted_pairs(x, F, r, omit, xs, fs) && !xs.empty()) {
            const auto n = static_cast<std::int64_t>(xs.size());
            const double nn = static_cast<double>(n);
            Acc s;
            s.add(1.0 / (12.0 * nn));
            for (std::int64_t i = 0; i < n; ++i) {
                const double u = (2.0 * static_cast<double>(i + 1) - 1.0) / (2.0 * nn);
                const double d = u - fs[static_cast<std::size_t>(i)];
                s.add(dd::two_prod(d, d));
            }
            W = to_double(s.value());
            p = n >= 2 ? cvm::sf(W, nn) : kNaN;
            if (n < 2) W = kNaN;
        }
        out.p[0][r] = W, out.p[1][r] = p;
    });
    return out.tuple();
}

// the null distributions at points (tests and the reference comparisons)
double dist_sf(int kind, double n, double x) {
    const auto ni = static_cast<std::int64_t>(n);
    switch (kind) {
        case 0: return ks::kstwo_sf(ni, x);
        case 1: return ks::smirnov_sf(ni, x);
        case 2: return ks::kolmogorov_sf(x);
        default: return cvm::sf(x, n);
    }
}

}  // namespace

void bind_gof(nb::module_& m) {
    m.def(
        "power_divergence", &power_divergence, nb::arg("obs"), nb::arg("exp").none(), nb::arg("lam"), nb::arg("ddof"),
        nb::arg("omit")
    );
    m.def("jarque_bera", &jarque_bera);
    m.def("jb_of_sums", &jb_of_sums);
    m.def("dagostino", &dagostino);
    m.def("shapiro", &shapiro);
    m.def("anderson", &anderson);
    m.def("ks", &ks_test);
    m.def("cvm", &cvm_test);
    m.def("dist_sf", &dist_sf);
}

}  // namespace sd::htest
