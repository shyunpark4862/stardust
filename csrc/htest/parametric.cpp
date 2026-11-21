// The tests of means and variances (milestone H1, docs/design/htest.md §3.2).
//
// The Python side (stardust.htest._parametric) checks the arguments and arranges every sample as a
// 2-D float64 array, one lane (one test of a batch) per row; each kernel returns the per-lane
// results as 1-D arrays. Lanes are independent tests: the same lane gives the same bits alone or
// in a batch, on any thread count (stats' fixed blocks, htest/moments.hpp).
//
//   ttest        ttest_1samp (kind 0), ttest_rel (1), ttest_ind (2, with Yuen's trim): statistic,
//                p-value, df, the estimate (hi + lo) and its standard error (for conf_int)
//   ttest_stats  ttest_ind_from_stats: the same from summary statistics
//   kgroups      f_oneway (0), Welch's one-way ANOVA (1), alexandergovern (2), bartlett (3),
//                levene (4), fligner (5): statistic, p-value, the degrees of freedom
//
// The p-values are sd.stats' distributions (StudentT, F, ChiSquared) called through their own
// methods (stats::method_of): the bits of StudentT(df).ccdf(t) in Python, and of
// sd.regression's p-values for the same arguments.
//
// Degenerate lanes (H11) give NaN without warnings: too few observations, a zero standard error
// or within-group sum of squares (every sample constant by sd.stats skew's rule, H20), a NaN under
// nan_policy='propagate' (every field NaN, as scipy).
#include "htest/parametric.hpp"

#include "array/ndarray.hpp"
#include "array/ops/pieces.hpp"
#include "array/parallel_limits.hpp"
#include "bind/pyutil.hpp"
#include "htest/lanes.hpp"
#include "htest/moments.hpp"
#include "special/dd.hpp"
#include "special/ops/erf.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

// The lanes' unit exponents (htest/moments.hpp) from their largest magnitudes; empty when every
// lane is in range (the usual data: no scaling).
std::vector<int> units_of(const std::vector<double>& mx) {
    std::vector<int> u(mx.size(), 0);
    bool any = false;
    for (std::size_t r = 0; r < mx.size(); ++r) any = (u[r] = unit_exponent(mx[r])) != 0 || any;
    if (!any) u.clear();
    return u;
}
const int* units_ptr(const std::vector<int>& u) { return u.empty() ? nullptr : u.data(); }
int unit_at(const std::vector<int>& u, std::int64_t r) { return u.empty() ? 0 : u[static_cast<std::size_t>(r)]; }

// the samples of one lane in one unit (unit_exponent of their largest magnitude)
int to_common_unit(std::vector<std::vector<double>*> vs) {
    double m = 0.0;
    for (const auto* v : vs)
        for (double x : *v) m = std::max(m, std::fabs(x));
    const int e = unit_exponent(m);
    if (e) {
        const double f = std::ldexp(1.0, -e);
        for (auto* v : vs)
            for (double& x : *v) x *= f;
    }
    return e;
}

// ---- p-values ------------------------------------------------------------------------------

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };

double t_pvalue(double t, double df, int alt) {
    using stats::StudentT;
    namespace m = stats::method;
    if (alt == kTwoSided) return 2.0 * stats::method_of<StudentT, m::Ccdf>(std::fabs(t), df);
    if (alt == kLess) return stats::method_of<StudentT, m::Cdf>(t, df);
    return stats::method_of<StudentT, m::Ccdf>(t, df);
}
double f_sf(double x, double dfn, double dfd) { return stats::method_of<stats::F, stats::method::Ccdf>(x, dfn, dfd); }
double chi2_sf(double x, double df) { return stats::method_of<stats::ChiSquared, stats::method::Ccdf>(x, df); }

// ---- the t tests ---------------------------------------------------------------------------

// One lane's t test: o[0..5] = statistic, p-value, df, estimate hi, lo, standard error.
struct TLane {
    double stat = kNaN, p = kNaN, df = kNaN, est_hi = kNaN, est_lo = kNaN, se = kNaN;
    void put(const std::vector<double*>& o, std::int64_t r) const {
        o[0][r] = stat, o[1][r] = p, o[2][r] = df, o[3][r] = est_hi, o[4][r] = est_lo, o[5][r] = se;
    }
    // the estimate and the standard error back in the data's unit 2^e
    TLane& unit(int e) {
        if (e) est_hi = std::ldexp(est_hi, e), est_lo = std::ldexp(est_lo, e), se = std::ldexp(se, e);
        return *this;
    }
};

// t = diff / sqrt(se2) with `df` degrees of freedom. zero: the standard error is 0 (every
// sample constant): t and p are NaN and the interval has width 0 (user decision 2026-11-20).
TLane t_finish(DD diff, DD est, double se2, double df, bool zero, int alt) {
    TLane t;
    t.df = df;
    t.est_hi = est.hi, t.est_lo = est.lo;
    if (zero) {
        t.se = 0.0;
        return t;
    }
    if (!(se2 >= 0.0) || !(df > 0.0)) return t;
    t.se = std::sqrt(se2);
    t.stat = to_double(diff) / t.se;
    if (alt >= 0) t.p = t_pvalue(t.stat, df, alt);  // (alt < 0: the statistic alone, a resample's)
    return t;
}

// One sample's summary for the two-sample test: its size (h for Yuen), mean (trimmed), sum of
// squares (winsorized) and whether it is constant.
struct Summary {
    double n = 0.0;
    DD mean{kNaN, 0.0};
    DD ss{kNaN, 0.0};
    bool constant = false, bad = false;
};
Summary summary_of(const Moments& m) { return {static_cast<double>(m.n), m.mean, m.ss, m.constant(), m.bad}; }

// Yuen's trimmed sample (scipy's _ttest_trim_var_mean_len): g = int(n trim) of the double product,
// the mean of the middle h = n - 2g order statistics, and the sum of squares of the sample with
// its g smallest values replaced by x_(g+1) and its g largest by x_(n-g) (winsorized). v is
// reordered (two selections).
Summary trimmed(std::vector<double>& v, double trim) {
    Summary s;
    const auto n = static_cast<std::int64_t>(v.size());
    if (n == 0) return s;
    const auto g = static_cast<std::int64_t>(static_cast<double>(n) * trim);
    if (g > 0) {
        std::nth_element(v.begin(), v.begin() + g, v.end());
        std::nth_element(v.begin() + g + 1, v.begin() + (n - g - 1), v.end());
    }
    const std::int64_t h = n - 2 * g;
    Acc mid;
    for (std::int64_t j = g; j < n - g; ++j) mid.add(v[static_cast<std::size_t>(j)]);
    s.n = static_cast<double>(h);
    s.mean = mid.value() / static_cast<double>(h);
    const double lo = v[static_cast<std::size_t>(g)], hi = v[static_cast<std::size_t>(n - g - 1)];
    const auto gd = static_cast<double>(g);
    Acc w = mid;
    if (g > 0) {
        w.add(dd::two_prod(gd, lo));
        w.add(dd::two_prod(gd, hi));
    }
    const DD wm = w.value() / static_cast<double>(n);
    if (!std::isfinite(wm.hi)) {
        s.ss = {kNaN, 0.0};
        return s;
    }
    Acc q;
    for (std::int64_t j = g; j < n - g; ++j) q.add(sq_dev(v[static_cast<std::size_t>(j)], 0.0, wm));
    if (g > 0) {
        q.add(sq_dev(lo, 0.0, wm) * gd);
        q.add(sq_dev(hi, 0.0, wm) * gd);
    }
    s.ss = q.value();
    const double m2 = s.ss.hi / static_cast<double>(n), e = kEps * wm.hi;
    s.constant = m2 <= e * e;
    return s;
}

// The two-sample test from the samples' summaries (scipy's _equal_var_ttest_denom and
// _unequal_var_ttest_denom; a sample of one has variance 0 in Student's).
TLane two_sample(const Summary& a, const Summary& b, bool equal_var, int alt) {
    if (a.bad || b.bad || a.n < 1.0 || b.n < 1.0) return {};
    const DD diff = a.mean - b.mean;
    if (equal_var) {
        const double df = a.n + b.n - 2.0;
        if (!(df > 0.0)) return {kNaN, kNaN, df, diff.hi, diff.lo, kNaN};
        const double svar = to_double(a.ss + b.ss) / df;
        const double se2 = svar * ((a.n + b.n) / (a.n * b.n));
        return t_finish(diff, diff, se2, df, a.constant && b.constant, alt);
    }
    if (a.n < 2.0 || b.n < 2.0) return {};
    const double vn1 = to_double(a.ss) / (a.n - 1.0) / a.n, vn2 = to_double(b.ss) / (b.n - 1.0) / b.n;
    const double se2 = vn1 + vn2;
    const double df = se2 * se2 / (vn1 * vn1 / (a.n - 1.0) + vn2 * vn2 / (b.n - 1.0));
    return t_finish(diff, diff, se2, df, a.constant && b.constant, alt);
}

// kind 0: ttest_1samp(a, popmean), 1: ttest_rel(a, b), 2: ttest_ind(a, b). popmean: a (lanes,)
// array (kind 0) or None. Returns (statistic, pvalue, df, estimate hi, estimate lo, se).
nb::tuple ttest(
    int kind, nb::handle a_obj, nb::handle b_obj, nb::handle mu_obj, bool equal_var, double trim, int alt, bool omit
) {
    const Rows a(rows_arg(a_obj));
    const std::int64_t lanes = a.lanes;
    Rows b;
    if (kind != 0) b = Rows(rows_arg(b_obj));
    const NDArray* mu = kind == 0 ? &vec_arg(mu_obj, lanes) : nullptr;
    Outs out(6, lanes);
    std::vector<double> mus(static_cast<std::size_t>(lanes), 0.0);
    if (mu)
        for (std::int64_t r = 0; r < lanes; ++r) mus[static_cast<std::size_t>(r)] = vec_at(*mu, r);
    if (kind != 2 || trim == 0.0) {
        std::vector<Moments> ma(static_cast<std::size_t>(lanes)), mb;
        std::vector<double> mx(static_cast<std::size_t>(lanes), 0.0);
        if (kind == 1) {
            lane_maxabs(Diff{a, b}, lanes, a.n, mx.data());
        } else {
            lane_maxabs(Plain{a}, lanes, a.n, mx.data());
            if (kind == 2) lane_maxabs(Plain{b}, lanes, b.n, mx.data());
        }
        const std::vector<int> u = units_of(mx);
        if (kind == 1) {
            lane_moments(Diff{a, b}, lanes, a.n, omit, ma.data(), units_ptr(u));
        } else {
            lane_moments(Plain{a}, lanes, a.n, omit, ma.data(), units_ptr(u));
            if (kind == 2) {
                mb.resize(static_cast<std::size_t>(lanes));
                lane_moments(Plain{b}, lanes, b.n, omit, mb.data(), units_ptr(u));
            }
        }
        each_lane(lanes, 64, [&](std::int64_t r) {
            const Moments& m = ma[static_cast<std::size_t>(r)];
            TLane t;
            if (kind == 2) {
                t = two_sample(summary_of(m), summary_of(mb[static_cast<std::size_t>(r)]), equal_var, alt);
            } else if (!m.bad && m.n >= 1) {
                const auto n = static_cast<double>(m.n);
                if (m.n < 2) {
                    t.df = n - 1.0, t.est_hi = m.mean.hi, t.est_lo = m.mean.lo;
                } else {
                    const DD diff = m.mean - std::ldexp(mus[static_cast<std::size_t>(r)], -unit_at(u, r));
                    t = t_finish(diff, m.mean, m.var() / n, n - 1.0, m.constant(), alt);
                }
            }
            t.unit(unit_at(u, r)).put(out.p, r);
        });
        return out.tuple();
    }
    // Yuen's trimmed test: each lane's samples gathered and selected
    each_lane(lanes, a.n + b.n, [&](std::int64_t r) {
        std::vector<double> va, vb;
        const bool bad_a = gather(a, r, omit, va), bad_b = gather(b, r, omit, vb);  // (both gathered)
        const bool bad = bad_a || bad_b;
        TLane t;
        if (!bad) {
            const int e = to_common_unit({&va, &vb});
            const Summary sa = trimmed(va, trim), sb = trimmed(vb, trim);
            t = two_sample(sa, sb, equal_var, alt).unit(e);
        }
        t.put(out.p, r);
    });
    return out.tuple();
}

// ttest_ind_from_stats: six (L,) arrays (broadcast by the caller).
nb::tuple ttest_stats(
    nb::handle m1o,
    nb::handle s1o,
    nb::handle n1o,
    nb::handle m2o,
    nb::handle s2o,
    nb::handle n2o,
    bool equal_var,
    int alt
) {
    const NDArray& m1a = require_ndarray(m1o);
    const std::int64_t L = m1a.ndim == 1 ? m1a.shape[0] : -1;
    const NDArray *m1 = &vec_arg(m1o, L), *s1 = &vec_arg(s1o, L), *n1 = &vec_arg(n1o, L), *m2 = &vec_arg(m2o, L),
                  *s2 = &vec_arg(s2o, L), *n2 = &vec_arg(n2o, L);
    Outs out(6, L);
    each_lane(L, 64, [&](std::int64_t r) {
        const double na = vec_at(*n1, r), nb_ = vec_at(*n2, r), sa = vec_at(*s1, r), sb = vec_at(*s2, r);
        const DD diff = dd::two_sum(vec_at(*m1, r), -vec_at(*m2, r));
        TLane t;
        if (na > 0.0 && nb_ > 0.0) {
            if (equal_var) {
                const double v1 = na == 1.0 ? 0.0 : sa * sa, v2 = nb_ == 1.0 ? 0.0 : sb * sb;
                const double df = na + nb_ - 2.0;
                const double se2 = ((na - 1.0) * v1 + (nb_ - 1.0) * v2) / df * ((na + nb_) / (na * nb_));
                t = df > 0.0 ? t_finish(diff, diff, se2, df, se2 == 0.0, alt)
                             : TLane{kNaN, kNaN, df, diff.hi, diff.lo, kNaN};
            } else if (na > 1.0 && nb_ > 1.0) {
                const double vn1 = sa * sa / na, vn2 = sb * sb / nb_, se2 = vn1 + vn2;
                const double df = se2 * se2 / (vn1 * vn1 / (na - 1.0) + vn2 * vn2 / (nb_ - 1.0));
                t = t_finish(diff, diff, se2, df, se2 == 0.0, alt);
            }
        }
        t.put(out.p, r);
    });
    return out.tuple();
}

// ---- k samples -----------------------------------------------------------------------------

enum Kind { kAnova = 0, kWelch = 1, kAlexanderGovern = 2, kBartlett = 3, kLevene = 4, kFligner = 5 };
enum Center { kMedian = 0, kMean = 1, kTrimmed = 2 };

struct KLane {
    double stat = kNaN, p = kNaN, dfn = kNaN, dfd = kNaN;
    void put(const std::vector<double*>& o, std::int64_t r) const {
        o[0][r] = stat, o[1][r] = p, o[2][r] = dfn, o[3][r] = dfd;
    }
};

// One-way ANOVA from the groups' moments (scipy's f_oneway): F = (SSB / (k - 1)) / (SSW / (N - k)),
// SSB = sum n_i (m_i - m)^2 from double-double means. An empty group or N <= k: NaN; every group
// constant (SSW = 0, H20; scipy: inf, or NaN): NaN.
KLane anova(const Moments* g, std::int64_t k) {
    KLane o;
    double N = 0.0;
    bool all_constant = true;
    for (std::int64_t i = 0; i < k; ++i) {
        if (g[i].bad) return o;
        N += static_cast<double>(g[i].n);
    }
    o.dfn = static_cast<double>(k - 1), o.dfd = N - static_cast<double>(k);
    Acc tot, ssw;
    for (std::int64_t i = 0; i < k; ++i) {
        if (g[i].n == 0) return o;
        const auto n = static_cast<double>(g[i].n);
        tot.add(g[i].mean * n);
        ssw.add(g[i].ss);
        all_constant = all_constant && g[i].constant();
    }
    if (!(o.dfd > 0.0) || all_constant) return o;
    const DD grand = tot.value() / N;
    double ssb = 0.0;
    for (std::int64_t i = 0; i < k; ++i) {
        const double d = to_double(g[i].mean - grand);
        ssb += static_cast<double>(g[i].n) * d * d;
    }
    o.stat = (ssb / o.dfn) / (to_double(ssw.value()) / o.dfd);
    o.p = f_sf(o.stat, o.dfn, o.dfd);
    return o;
}

// Welch's one-way ANOVA (scipy's f_oneway(equal_var=False), R's oneway.test(var.equal=FALSE)):
// weights w_i = n_i / s_i^2, the weighted grand mean, F = [sum w_i (m_i - m_w)^2 / (k - 1)] /
// [1 + 2 (k - 2) / (k^2 - 1) T], df_denom = (k^2 - 1) / (3 T), T = sum (1 - w_i / W)^2 / (n_i - 1)
// with 1 - w_i / W as the other weights' sum over W. A group of fewer than two or a constant
// group: NaN.
KLane welch(const Moments* g, std::int64_t k) {
    KLane o;
    for (std::int64_t i = 0; i < k; ++i)
        if (g[i].bad) return o;
    const auto kd = static_cast<double>(k);
    o.dfn = kd - 1.0;
    std::vector<double> w(static_cast<std::size_t>(k));
    for (std::int64_t i = 0; i < k; ++i) {
        if (g[i].n < 2 || g[i].constant()) return o;
        w[static_cast<std::size_t>(i)] = static_cast<double>(g[i].n) / g[i].var();
    }
    Acc sw, swm;
    for (std::int64_t i = 0; i < k; ++i) {
        const double wi = w[static_cast<std::size_t>(i)];
        sw.add(wi);
        swm.add(g[i].mean * wi);
    }
    // (both sums double-doubles: a rounded W would move m_w by W's rounding times the means' size)
    const double W = to_double(sw.value());
    const DD mw = swm.value() / sw.value();
    double num = 0.0, tmp = 0.0;
    for (std::int64_t i = 0; i < k; ++i) {
        const double d = to_double(g[i].mean - mw);
        num += w[static_cast<std::size_t>(i)] * d * d;
        double others = 0.0;
        for (std::int64_t j = 0; j < k; ++j)
            if (j != i) others += w[static_cast<std::size_t>(j)];
        const double f = others / W;
        tmp += f * f / static_cast<double>(g[i].n - 1);
    }
    num /= kd - 1.0;
    o.stat = num / (1.0 + 2.0 * (kd - 2.0) / (kd * kd - 1.0) * tmp);
    o.dfd = (kd * kd - 1.0) / (3.0 * tmp);
    o.p = f_sf(o.stat, o.dfn, o.dfd);
    return o;
}

// Alexander and Govern (scipy's alexandergovern): each group's t against the variance-weighted
// mean, Hill's normalizing transformation z(t), A = sum z^2, chi^2 (k - 1). A group of fewer than
// two or a constant group: NaN.
KLane alexander_govern(const Moments* g, std::int64_t k) {
    KLane o;
    for (std::int64_t i = 0; i < k; ++i)
        if (g[i].bad) return o;
    o.dfn = static_cast<double>(k - 1);
    std::vector<double> se2(static_cast<std::size_t>(k));
    Acc inv;
    for (std::int64_t i = 0; i < k; ++i) {
        if (g[i].n < 2 || g[i].constant()) return o;
        se2[static_cast<std::size_t>(i)] = g[i].var() / static_cast<double>(g[i].n);
        inv.add(1.0 / se2[static_cast<std::size_t>(i)]);
    }
    // the weighted mean as sum (m_i / se_i^2) / sum (1 / se_i^2), both double-doubles: rounded
    // normalized weights (scipy's) move it by their rounding times the means' size
    Acc mwa;
    for (std::int64_t i = 0; i < k; ++i) mwa.add(g[i].mean * (1.0 / se2[static_cast<std::size_t>(i)]));
    const DD mw = mwa.value() / inv.value();
    double A = 0.0;
    for (std::int64_t i = 0; i < k; ++i) {
        const double t = to_double(g[i].mean - mw) / std::sqrt(se2[static_cast<std::size_t>(i)]);
        const auto v = static_cast<double>(g[i].n - 1);
        const double a = v - 0.5, b = 48.0 * a * a;
        const double c = std::sqrt(a * std::log1p(t * t / v));
        const double c2 = c * c, c3 = c2 * c, c4 = c2 * c2, c5 = c4 * c, c7 = c4 * c3;
        const double z = c + (c3 + 3.0 * c) / b -
                         (4.0 * c7 + 33.0 * c5 + 240.0 * c3 + 855.0 * c) / (b * b * 10.0 + 8.0 * b * c4 + 1000.0 * b);
        A += z * z;
    }
    o.stat = A;
    o.p = chi2_sf(A, o.dfn);
    return o;
}

// Bartlett's T = [(N - k) log s_p^2 - sum (n_i - 1) log s_i^2] / C as a sum of non-negative terms:
// with d_i = s_i^2 / s_p^2 - 1, sum (n_i - 1) d_i = 0, so the numerator is
// sum (n_i - 1) (d_i - log1p(d_i)) (no cancellation; scipy subtracts the two logarithms, 53 times
// too large when the variances agree to 1e-9). s_i^2 and s_p^2 are double-doubles. C = 1 + (sum
// 1/(n_i - 1) - 1/(N - k)) / (3 (k - 1)). A group of fewer than two or a constant group: NaN
// (scipy: inf).
KLane bartlett(const Moments* g, std::int64_t k) {
    KLane o;
    for (std::int64_t i = 0; i < k; ++i)
        if (g[i].bad) return o;
    o.dfn = static_cast<double>(k - 1);
    double N = 0.0, inv = 0.0;
    Acc ss;
    for (std::int64_t i = 0; i < k; ++i) {
        if (g[i].n < 2 || g[i].constant()) return o;
        N += static_cast<double>(g[i].n);
        inv += 1.0 / static_cast<double>(g[i].n - 1);
        ss.add(g[i].ss);
    }
    const double dfw = N - static_cast<double>(k);
    const DD sp2 = ss.value() / dfw;
    Acc num;
    for (std::int64_t i = 0; i < k; ++i) {
        const auto ni1 = static_cast<double>(g[i].n - 1);
        const DD d = (g[i].ss / ni1 - sp2) / sp2;
        num.add(dd::log1pmx_neg(d) * ni1);
    }
    const double C = 1.0 + (inv - 1.0 / dfw) / (3.0 * static_cast<double>(k - 1));
    o.stat = to_double(num.value()) / C;
    o.p = chi2_sf(o.stat, o.dfn);
    return o;
}

// The centre of a sample (levene's and fligner's `center`) as a double-double: the median (the
// middle value, or the exact half sum of the two middle ones), the mean, or scipy's trim_mean
// (g = int(proportiontocut n), the mean of the middle n - 2g order statistics). v is reordered
// by the trimmed mean; the median is a selection in a copy (a long sample sorted on the pool, else
// nth_element: the same values), so v keeps the data's order.
DD center_of(std::vector<double>& v, int center, double ptc) {
    const auto n = static_cast<std::int64_t>(v.size());
    if (n == 0) return {kNaN, 0.0};
    if (center == kMean) return moments_of(v.data(), nullptr, n).mean;
    if (center == kTrimmed) return trimmed(v, ptc).mean;
    const std::int64_t h = n / 2;
    std::vector<double> w(v);
    double hi, lo = kNaN;
    if (n >= (1 << 16) && parallel::available()) {
        sort_values(w);
        hi = w[static_cast<std::size_t>(h)];
        if (n % 2 == 0) lo = w[static_cast<std::size_t>(h - 1)];
    } else {
        std::nth_element(w.begin(), w.begin() + h, w.end());
        hi = w[static_cast<std::size_t>(h)];
        if (n % 2 == 0) lo = *std::max_element(w.begin(), w.begin() + h);
    }
    if (n % 2) return {hi, 0.0};
    const DD s = dd::two_sum(lo, hi);
    return {0.5 * s.hi, 0.5 * s.lo};
}

// |x - c| as a double-double
DD abs_dev(double x, DD c) { return abs_dd(dd::two_sum(x, -c.hi) + (-c.lo)); }

// fligner's score of the midrank r = r2 / 2 among N: Phi^-1(1/2 + r / (2 (N + 1))). The argument
// is not formed: below the middle (q = r / (2 (N + 1)) <= 1/4) AS241's central form at q, above it
// -ndtri((N + 1 - r) / (2 (N + 1))), each from one rounded quotient of exact integers.
double fligner_score(double r2, double N) {
    using special::erf_detail::ndtri_parts;
    if (r2 <= N + 1.0) {
        const double q = r2 / (4.0 * (N + 1.0));
        return q * ndtri_parts(q, 0.0).ratio;
    }
    const double p = (2.0 * N + 2.0 - r2) / (4.0 * (N + 1.0));
    const double q = p - 0.5;
    const special::erf_detail::NdtriParts r = ndtri_parts(q, std::log(p));
    return -(r.central ? q * r.ratio : r.tail);
}

// Levene's W (scipy's levene): the one-way ANOVA of z = |x - c_i| (double-doubles), groups gathered
// per lane. Fligner-Killeen (scipy's fligner): the midranks of |x - c_i| over all groups (ties by
// exact double-double equality), their normal scores, sum n_i (mean_i - mean)^2 / var, chi^2
// (k - 1).
KLane levene_lane(const std::vector<Rows>& s, std::int64_t r, bool omit, int center, double ptc, bool fligner) {
    const auto k = static_cast<std::int64_t>(s.size());
    KLane o;
    std::vector<std::vector<double>> v(static_cast<std::size_t>(k));
    std::vector<DD> c(static_cast<std::size_t>(k));
    for (std::int64_t i = 0; i < k; ++i) {
        if (gather(s[static_cast<std::size_t>(i)], r, omit, v[static_cast<std::size_t>(i)])) return o;
    }
    std::vector<std::vector<double>*> all_v;
    for (auto& x : v) all_v.push_back(&x);
    to_common_unit(all_v);
    double N = 0.0;
    for (std::int64_t i = 0; i < k; ++i) {
        // (in place: the order of a group's values does not matter below, and is the same for
        // the lane alone and in a batch)
        c[static_cast<std::size_t>(i)] = center_of(v[static_cast<std::size_t>(i)], center, ptc);
        N += static_cast<double>(v[static_cast<std::size_t>(i)].size());
    }
    o.dfn = static_cast<double>(k - 1);
    if (!fligner) o.dfd = N - static_cast<double>(k);
    for (std::int64_t i = 0; i < k; ++i)
        if (v[static_cast<std::size_t>(i)].empty()) return o;
    if (!fligner) {
        std::vector<Moments> m(static_cast<std::size_t>(k));
        std::vector<double> zh, zl;
        for (std::int64_t i = 0; i < k; ++i) {
            const std::vector<double>& x = v[static_cast<std::size_t>(i)];
            const auto nx = static_cast<std::int64_t>(x.size());
            const DD ci = c[static_cast<std::size_t>(i)];
            zh.resize(x.size()), zl.resize(x.size());
            for_pieces(nx, kTask, nx, [&](std::int64_t, std::int64_t b, std::int64_t e) {
                for (std::int64_t j = b; j < e; ++j) {
                    const DD z = abs_dev(x[static_cast<std::size_t>(j)], ci);
                    zh[static_cast<std::size_t>(j)] = z.hi, zl[static_cast<std::size_t>(j)] = z.lo;
                }
            });
            // (the blocked kernel: on the pool for a long lane alone, the same bits inside a batch)
            lane_moments(
                Vec{zh.data(), zl.data()}, 1, static_cast<std::int64_t>(x.size()), false,
                &m[static_cast<std::size_t>(i)]
            );
        }
        return anova(m.data(), k);
    }
    std::vector<Key> keys(static_cast<std::size_t>(N));
    std::int64_t at = 0;
    for (std::int64_t i = 0; i < k; ++i) {
        const std::vector<double>& x = v[static_cast<std::size_t>(i)];
        const auto nx = static_cast<std::int64_t>(x.size());
        const DD ci = c[static_cast<std::size_t>(i)];
        for_pieces(nx, kTask, nx, [&](std::int64_t, std::int64_t b, std::int64_t e) {
            for (std::int64_t j = b; j < e; ++j) {
                const DD z = abs_dev(x[static_cast<std::size_t>(j)], ci);
                keys[static_cast<std::size_t>(at + j)] = {z.hi, z.lo, static_cast<std::int32_t>(i)};
            }
        });
        at += nx;
    }
    sort_keys(keys);
    // the tie runs (first and last position of each element's run), the scores on the pool (a long
    // lane alone; elementwise, so any split gives the same bits), the sums in order
    const auto nk = static_cast<std::int64_t>(keys.size());
    std::vector<std::int64_t> first(static_cast<std::size_t>(nk)), last(static_cast<std::size_t>(nk));
    for (std::int64_t i = 0; i < nk; ++i) {
        const bool tie = i > 0 && key_equal(keys[static_cast<std::size_t>(i - 1)], keys[static_cast<std::size_t>(i)]);
        first[static_cast<std::size_t>(i)] = tie ? first[static_cast<std::size_t>(i - 1)] : i;
    }
    for (std::int64_t i = nk - 1; i >= 0; --i) {
        const bool tie =
            i + 1 < nk && key_equal(keys[static_cast<std::size_t>(i + 1)], keys[static_cast<std::size_t>(i)]);
        last[static_cast<std::size_t>(i)] = tie ? last[static_cast<std::size_t>(i + 1)] : i;
    }
    std::vector<double> score(keys.size());
    for_pieces(nk, kTask, nk * 16, [&](std::int64_t, std::int64_t b, std::int64_t e) {
        for (std::int64_t i = b; i < e; ++i) {
            const auto r2 =
                static_cast<double>(first[static_cast<std::size_t>(i)] + last[static_cast<std::size_t>(i)] + 2);
            score[static_cast<std::size_t>(i)] = fligner_score(r2, N);
        }
    });
    std::vector<Acc> gs(static_cast<std::size_t>(k));
    for (std::int64_t i = 0; i < nk; ++i)
        gs[static_cast<std::size_t>(keys[static_cast<std::size_t>(i)].tag)].add(score[static_cast<std::size_t>(i)]);
    Moments sm;
    lane_moments(Vec{score.data(), nullptr}, 1, nk, false, &sm);
    const DD mean = sm.mean, ss = sm.ss;
    const double m2 = ss.hi / N, e = kEps * mean.hi;
    if (N < 2.0 || m2 <= e * e) return o;
    double num = 0.0;
    for (std::int64_t i = 0; i < k; ++i) {
        const auto ni = static_cast<double>(v[static_cast<std::size_t>(i)].size());
        const double d = to_double(gs[static_cast<std::size_t>(i)].value() / ni - mean);
        num += ni * d * d;
    }
    o.stat = num / (to_double(ss) / (N - 1.0));
    o.p = chi2_sf(o.stat, o.dfn);
    return o;
}

// kind: Kind. samples: a list of (lanes, n_i) arrays (lanes equal). Returns (statistic, pvalue,
// df_num, df_denom).
nb::tuple kgroups(int kind, nb::list samples, bool omit, int center, double ptc) {
    const auto k = static_cast<std::int64_t>(nb::len(samples));
    std::vector<Rows> s;
    std::int64_t total = 0;
    for (nb::handle h : samples) {
        s.emplace_back(rows_arg(h));
        total += s.back().n;
    }
    const std::int64_t lanes = s.front().lanes;
    Outs out(4, lanes);
    if (kind == kLevene || kind == kFligner) {
        each_lane(lanes, total * (kind == kFligner ? 16 : 4), [&](std::int64_t r) {
            levene_lane(s, r, omit, center, ptc, kind == kFligner).put(out.p, r);
        });
        return out.tuple();
    }
    // the groups' moments, lane by lane (group i of lane r at m[r * k + i])
    std::vector<Moments> m(static_cast<std::size_t>(lanes * k)), mi(static_cast<std::size_t>(lanes));
    std::vector<double> mx(static_cast<std::size_t>(lanes), 0.0);
    for (const Rows& g : s) lane_maxabs(Plain{g}, lanes, g.n, mx.data());
    const std::vector<int> u = units_of(mx);
    for (std::int64_t i = 0; i < k; ++i) {
        const Rows& g = s[static_cast<std::size_t>(i)];
        lane_moments(Plain{g}, lanes, g.n, omit, mi.data(), units_ptr(u));
        for (std::int64_t r = 0; r < lanes; ++r)
            m[static_cast<std::size_t>(r * k + i)] = mi[static_cast<std::size_t>(r)];
    }
    each_lane(lanes, 32 * k, [&](std::int64_t r) {
        const Moments* g = m.data() + r * k;
        KLane o = kind == kAnova             ? anova(g, k)
                  : kind == kWelch           ? welch(g, k)
                  : kind == kAlexanderGovern ? alexander_govern(g, k)
                                             : bartlett(g, k);
        o.put(out.p, r);
    });
    return out.tuple();
}

}  // namespace

void bind_parametric(nb::module_& m) {
    m.def(
        "ttest", &ttest, nb::arg("kind"), nb::arg("a"), nb::arg("b").none(), nb::arg("popmean").none(),
        nb::arg("equal_var"), nb::arg("trim"), nb::arg("alternative"), nb::arg("omit")
    );
    m.def("ttest_stats", &ttest_stats);
    m.def("kgroups", &kgroups);
}

}  // namespace sd::htest
