// Multiple comparisons and corrections (milestone H8, docs/design/htest.md §3.9).
//
//   multitest       the corrected p-values of every lane of a (lanes, m) array (htest/multitest.hpp):
//                   multipletests' ten methods and false_discovery_control's two, one kernel
//   combine         combine_pvalues' five methods per lane: Fisher's -2 sum log p (chi^2 with 2n
//                   degrees of freedom), Pearson's 2 sum log1p(-p) (its lower tail), Tippett's min p
//                   (1 - (1 - min)^n), Stouffer's sum w_i isf(p_i) / |w| (the normal's upper tail)
//                   and Mudholkar and George's -sum log(p / (1 - p)) (a t with 5n + 4 degrees of
//                   freedom, scipy's scaling). The sums of logarithms are double-doubles (each log
//                   in double-double, special/dd.hpp) and the tails take the sum's low part to first
//                   order, so a p-value deep in a tail keeps its digits (a rounded -2 sum log p of
//                   1000 moves the tail by 500 eps)
//   tukey           tukey_hsd's mean differences, standard errors, degrees of freedom (Tukey-Kramer's
//                   pooled N - k or Games and Howell's Welch df per pair) and the studentized range's
//                   upper tail per pair, the pairs on the pool
//   srange_isf      the studentized range's critical values (the intervals)
//   dunnett         Dunnett's t statistics against the control and their multivariate t tails
//   dunnett_isf     the critical value of the intervals
//
// The means and the sums of squares are double-doubles (htest/moments.hpp; the groups in one unit,
// the samples' largest magnitude's), so the mean differences keep their digits under an offset.
#include "htest/multicomp.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "htest/dunnett.hpp"
#include "htest/lanes.hpp"
#include "htest/moments.hpp"
#include "htest/multitest.hpp"
#include "htest/studentized_range.hpp"
#include "special/ops/incgamma.hpp"
#include "stats/dist/normal.hpp"
#include "stats/dist/t_f.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;

constexpr double kPi = 3.141592653589793;

DD D(double v) { return DD{v, 0.0}; }
DD sqrt_dd(DD a) {
    if (!(a.hi > 0.0)) return D(a.hi == 0.0 ? 0.0 : (a.hi == a.hi ? kNaN : a.hi));
    if (std::isinf(a.hi)) return D(kInf);
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    return dd::two_sum(s, ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s));
}

// ---- the corrections ----------------------------------------------------------------------

// Returns (corrected, reject): each (lanes, m), reject 1 or 0 (the procedure's decision at alpha).
nb::tuple correct_rows(nb::handle p, int method, double alpha, std::int64_t max_iter) {
    const NDArray& a = rows_arg(p);
    const Rows r(a);
    const std::int64_t shape[] = {r.lanes, r.n};
    NDArray out = new_array(DType::Float64, shape), rej = new_array(DType::Float64, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    auto* j0 = reinterpret_cast<double*>(rej.data);
    each_lane(r.lanes, std::max<std::int64_t>(1, r.n) * 40, [&](std::int64_t l) {
        std::vector<double> x(static_cast<std::size_t>(r.n));
        for (std::int64_t j = 0; j < r.n; ++j) x[static_cast<std::size_t>(j)] = r.at(l, j);
        multitest::Scratch w;
        multitest::correct(x.data(), 1, r.n, method, alpha, max_iter, o + l * r.n, j0 + l * r.n, w);
    });
    nb::list lst;
    lst.append(wrap(std::move(out)));
    lst.append(wrap(std::move(rej)));
    return nb::tuple(lst);
}

// ---- combine_pvalues ----------------------------------------------------------------------

enum Combine { kFisher = 0, kPearson = 1, kTippett = 2, kStouffer = 3, kMudholkarGeorge = 4 };

using multitest::log1m;  // log1p(-p) as a double-double
// log p as a double-double (subnormal p too)
DD logp(double p) { return p > 0.0 ? dd::log_pos(p) : D(p == 0.0 ? -kInf : kNaN); }

// The upper tail Q(a, x) and the lower P(a, x) of the gamma at x = hi + lo (first order in lo).
double gamma_pdf(double a, double x) {
    if (!(x > 0.0) || !std::isfinite(x)) return 0.0;
    return std::exp((a - 1.0) * std::log(x) - x - std::lgamma(a));
}
double gamma_q(double a, DD x) {
    const double q = special::Gammaincc::f(a, x.hi);
    if (!(q > 0.0) || x.lo == 0.0) return q;
    return q - x.lo * gamma_pdf(a, x.hi);
}
double gamma_p(double a, DD x) {
    const double p = special::Gammainc::f(a, x.hi);
    if (!(p > 0.0) || x.lo == 0.0) return p;
    return p + x.lo * gamma_pdf(a, x.hi);
}

struct Pair {
    double stat = kNaN, p = kNaN;
};

// One lane's p-values v (and weights w, or empty), no NaNs.
Pair combine_lane(const std::vector<double>& v, const std::vector<double>& w, int method) {
    Pair o;
    const auto n = static_cast<double>(v.size());
    if (v.empty()) return o;
    if (method == kFisher || method == kPearson || method == kMudholkarGeorge) {
        Acc s;
        for (double p : v) {
            DD t = method == kFisher ? logp(p) : (method == kPearson ? log1m(p) : log1m(p) - logp(p));
            s.add(t);
        }
        DD sum = s.value();
        if (!std::isfinite(sum.hi)) sum = D(sum.hi);  // a p of 0 or 1: the products below keep the infinity
        if (method == kFisher) {
            const DD S = std::isfinite(sum.hi) ? sum * -2.0 : D(sum.hi * -2.0);
            o.stat = S.hi;
            o.p = std::isfinite(S.hi) ? gamma_q(n, S * 0.5) : (S.hi > 0.0 ? 0.0 : kNaN);
        } else if (method == kPearson) {
            const DD S = std::isfinite(sum.hi) ? sum * 2.0 : D(sum.hi * 2.0);
            o.stat = S.hi;
            o.p = std::isfinite(S.hi) ? gamma_p(n, S * -0.5) : (S.hi < 0.0 ? 1.0 : kNaN);
        } else {  // T = -sum log(p / (1 - p)); scipy's t: T sqrt(3 / n) / pi sqrt(nu / (nu - 2))
            o.stat = sum.hi;
            const double nu = 5.0 * n + 4.0;
            const DD c = sqrt_dd(DD{3.0, 0.0} / n * (D(nu) / (nu - 2.0))) / kPi;
            const DD x = std::isfinite(sum.hi) ? sum * c : D(sum.hi);
            if (!std::isfinite(x.hi)) {
                o.p = x.hi > 0.0 ? 0.0 : (x.hi < 0.0 ? 1.0 : kNaN);
            } else {
                const double sf = stats::method_of<stats::StudentT, m::Ccdf>(x.hi, nu);
                const double pdf = stats::method_of<stats::StudentT, m::Pdf>(x.hi, nu);
                o.p = sf > 0.0 ? sf - x.lo * pdf : sf;
            }
        }
        return o;
    }
    if (method == kTippett) {
        double mn = v[0];
        for (double p : v) mn = std::min(mn, p);
        o.stat = mn;
        o.p = -std::expm1(n * std::log1p(-mn));
        return o;
    }
    // Stouffer
    Acc z, ww;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double wi = w.empty() ? 1.0 : w[i];
        const double zi = stats::method_of<stats::Normal, m::Iccdf>(v[i], 0.0, 1.0);
        z.add(dd::two_prod(wi, zi));
        ww.add(dd::two_prod(wi, wi));
    }
    const DD zs = z.value(), nw = sqrt_dd(ww.value());
    if (!std::isfinite(zs.hi) || !(nw.hi > 0.0)) {
        o.stat = zs.hi / nw.hi;
        o.p = stats::method_of<stats::Normal, m::Ccdf>(o.stat, 0.0, 1.0);
        return o;
    }
    const DD Z = zs / nw;
    o.stat = Z.hi;
    const double sf = stats::method_of<stats::Normal, m::Ccdf>(Z.hi, 0.0, 1.0);
    o.p = sf > 0.0 ? sf * (1.0 - Z.lo * srange::hazard(Z.hi)) : sf;
    return o;
}

nb::tuple combine(nb::handle p, nb::handle w, int method, bool omit) {
    const Rows r(rows_arg(p));
    const bool weighted = !w.is_none();
    Rows rw;
    if (weighted) rw = Rows(rows_arg(w));
    Outs out(2, r.lanes);
    each_lane(r.lanes, std::max<std::int64_t>(1, r.n) * 64, [&](std::int64_t l) {
        std::vector<double> v, ws;
        bool nan = false;
        for (std::int64_t j = 0; j < r.n; ++j) {
            const double x = r.at(l, j), y = weighted ? rw.at(l, j) : 1.0;
            if (x != x || y != y) {
                nan = true;
                continue;
            }
            v.push_back(x);
            if (weighted) ws.push_back(y);
        }
        Pair o;
        if (!nan || omit) o = combine_lane(v, ws, method);
        out.p[0][l] = o.stat, out.p[1][l] = o.p;
    });
    return out.tuple();
}

// ---- the groups ---------------------------------------------------------------------------

// Every group's moments in one unit (the samples' largest magnitude's exponent), the unit's
// exponent returned: a mean or a difference times 2^e is the data's.
int group_moments(nb::list samples, std::vector<Moments>& g) {
    std::vector<Rows> s;
    for (nb::handle h : samples) s.emplace_back(rows_arg(h));
    double mx = 0.0;
    for (const Rows& r : s) lane_maxabs(Plain{r}, 1, r.n, &mx);
    const int u = unit_exponent(mx);
    g.resize(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) lane_moments(Plain{s[i]}, 1, s[i].n, false, &g[i], u ? &u : nullptr);
    return u;
}

// m_i - m_j in the data's unit
double diff_of(const Moments& a, const Moments& b, int u) { return std::ldexp(to_double(a.mean - b.mean), u); }

// ---- tukey_hsd ----------------------------------------------------------------------------

// Returns (statistic, pvalue, stderr, df), each k * k (row-major (i, j)); stderr in the data's unit.
nb::tuple tukey(nb::list samples, bool equal_var) {
    std::vector<Moments> g;
    const int u = group_moments(samples, g);
    const auto k = static_cast<std::int64_t>(g.size());
    Outs out(4, k * k);
    double *stat = out.p[0], *pv = out.p[1], *se = out.p[2], *df = out.p[3];
    double N = 0.0;
    bool all_constant = true;
    Acc ssw;
    for (const Moments& m : g) {
        N += static_cast<double>(m.n);
        ssw.add(m.ss);
        all_constant = all_constant && m.constant();
    }
    const double dfp = N - static_cast<double>(k);
    const DD mse = dfp > 0.0 ? ssw.value() / dfp : D(kNaN);
    // each group's variance over its size (Games-Howell), a double-double in the unit
    std::vector<DD> vn(static_cast<std::size_t>(k));
    for (std::int64_t i = 0; i < k; ++i) {
        const Moments& m = g[static_cast<std::size_t>(i)];
        const auto n = static_cast<double>(m.n);
        vn[static_cast<std::size_t>(i)] = m.n > 1 ? m.ss / (n - 1.0) / n : D(kNaN);
    }
    std::vector<std::int64_t> pi, pj;
    for (std::int64_t i = 0; i < k; ++i)
        for (std::int64_t j = i; j < k; ++j) pi.push_back(i), pj.push_back(j);
    const auto np = static_cast<std::int64_t>(pi.size());
    const double fk = static_cast<double>(k);
    each_lane(np, 1 << 20, [&](std::int64_t t) {
        const std::int64_t i = pi[static_cast<std::size_t>(t)], j = pj[static_cast<std::size_t>(t)];
        const Moments &a = g[static_cast<std::size_t>(i)], &b = g[static_cast<std::size_t>(j)];
        const double d = diff_of(a, b, u);
        DD var;  // the squared standard error in the unit
        double nu;
        bool zero;
        if (equal_var) {
            const auto na = static_cast<double>(a.n), nb_ = static_cast<double>(b.n);
            var = mse * (na + nb_) / (na * nb_) * 0.5;
            nu = dfp;
            zero = all_constant;
        } else {
            const DD s = vn[static_cast<std::size_t>(i)] + vn[static_cast<std::size_t>(j)];
            var = s * 0.5;
            const DD vi = vn[static_cast<std::size_t>(i)], vj = vn[static_cast<std::size_t>(j)];
            const DD den = vi * vi / static_cast<double>(a.n - 1) + vj * vj / static_cast<double>(b.n - 1);
            nu = to_double(s * s / den);
            zero = a.constant() && b.constant();
        }
        const DD sed = sqrt_dd(var);
        const double q = d != d ? kNaN : to_double(abs_dd(a.mean - b.mean) / sed);
        const double p = (zero || !(sed.hi > 0.0)) ? kNaN : srange::sf(q, fk, nu);
        const double e = std::ldexp(to_double(sed), u);
        for (int side = 0; side < 2; ++side) {
            const std::int64_t r = side ? j * k + i : i * k + j;
            stat[r] = side ? (d == 0.0 ? 0.0 : -d) : d;
            pv[r] = p, se[r] = e, df[r] = nu;
        }
    });
    return out.tuple();
}

// The studentized range's critical values q with P(Q > q) = alpha, for the degrees of freedom df.
nb::object srange_isf(double alpha, double k, nb::handle df) {
    const NDArray& a = require_ndarray(df);
    const std::int64_t n = a.shape[0];
    Outs out(1, n);
    each_lane(n, 1 << 20, [&](std::int64_t i) { out.p[0][i] = srange::isf(alpha, k, vec_at(a, i)); });
    return wrap(std::move(out.arrays[0]));
}

// ---- dunnett ------------------------------------------------------------------------------

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };

// The comparisons' l_j = sqrt(n_j / (n_j + n_0)) and s_j = sqrt(n_0 / (n_j + n_0)).
void factors(const std::vector<double>& n, double n0, std::vector<double>& l, std::vector<double>& s) {
    l.resize(n.size()), s.resize(n.size());
    for (std::size_t j = 0; j < n.size(); ++j) {
        l[j] = std::sqrt(n[j] / (n[j] + n0));
        s[j] = std::sqrt(n0 / (n[j] + n0));
    }
}

// samples: the d treatments, then the control. Returns (statistic, pvalue, difference, scale, df):
// scale = std sqrt(1/n_i + 1/n_0) in the data's unit (the interval's allowance per critical value).
nb::tuple dunnett_test(nb::list samples, int alt) {
    std::vector<Moments> g;
    const int u = group_moments(samples, g);
    const auto d = static_cast<std::int64_t>(g.size()) - 1;
    const Moments& c = g.back();
    double N = 0.0;
    bool all_constant = true;
    Acc ss;
    for (const Moments& m : g) {
        N += static_cast<double>(m.n);
        ss.add(m.ss);
        all_constant = all_constant && m.constant();
    }
    const double df = N - static_cast<double>(d) - 1.0;
    const DD var = df > 0.0 ? ss.value() / df : D(kNaN);
    const DD sd_ = sqrt_dd(var);
    const auto n0 = static_cast<double>(c.n);
    std::vector<double> ns(static_cast<std::size_t>(d)), l, s;
    for (std::int64_t i = 0; i < d; ++i) ns[static_cast<std::size_t>(i)] = static_cast<double>(g[i].n);
    factors(ns, n0, l, s);
    bool empty = c.n == 0;
    for (double n : ns) empty = empty || n == 0.0;
    Outs out(4, d);
    std::vector<double> t(static_cast<std::size_t>(d));
    for (std::int64_t i = 0; i < d; ++i) {
        const Moments& a = g[static_cast<std::size_t>(i)];
        const double ni = ns[static_cast<std::size_t>(i)];
        const DD w = sqrt_dd((D(ni) + n0) / (ni * n0));  // sqrt(1/n_i + 1/n_0)
        const DD scale = sd_ * w;
        out.p[2][i] = diff_of(a, c, u);
        out.p[3][i] = std::ldexp(to_double(scale), u);
        t[static_cast<std::size_t>(i)] =
            (all_constant || !(scale.hi > 0.0)) ? kNaN : to_double((a.mean - c.mean) / scale);
        out.p[0][i] = t[static_cast<std::size_t>(i)];
    }
    each_lane(d, 1 << 20, [&](std::int64_t i) {
        const double ti = t[static_cast<std::size_t>(i)];
        double p;
        if (ti != ti || empty) p = kNaN;
        else if (alt == kTwoSided) p = dunnett::tail(l.data(), s.data(), static_cast<int>(d), std::fabs(ti), df, true);
        else p = dunnett::tail(l.data(), s.data(), static_cast<int>(d), alt == kGreater ? ti : -ti, df, false);
        out.p[1][i] = p;
    });
    nb::list lst;
    for (NDArray& a : out.arrays) lst.append(wrap(std::move(a)));
    lst.append(nb::float_(df));
    return nb::tuple(lst);
}

// The critical value c of the 1 - alpha intervals: P(max |T_j| > c) = alpha (two-sided) or
// P(max T_j > c) = alpha, for the treatments' sizes n and the control's n0.
double dunnett_isf(double alpha, nb::handle n, double n0, double df, bool two) {
    const NDArray& a = require_ndarray(n);
    std::vector<double> ns(static_cast<std::size_t>(a.shape[0])), l, s;
    for (std::int64_t i = 0; i < a.shape[0]; ++i) ns[static_cast<std::size_t>(i)] = vec_at(a, i);
    factors(ns, n0, l, s);
    return dunnett::isf(l.data(), s.data(), static_cast<int>(ns.size()), alpha, df, two);
}

// The tails themselves (the tests' checks against sd.stats' MultivariateT): P(Q > q) and the
// one-factor multivariate t's.
double srange_sf(double q, double k, double nu) { return srange::sf(q, k, nu); }
double dunnett_tail(double c, nb::handle n, double n0, double df, bool two) {
    const NDArray& a = require_ndarray(n);
    std::vector<double> ns(static_cast<std::size_t>(a.shape[0])), l, s;
    for (std::int64_t i = 0; i < a.shape[0]; ++i) ns[static_cast<std::size_t>(i)] = vec_at(a, i);
    factors(ns, n0, l, s);
    return dunnett::tail(l.data(), s.data(), static_cast<int>(ns.size()), c, df, two);
}

}  // namespace

void bind_multicomp(nb::module_& m) {
    m.def("multitest", &correct_rows);
    m.def("combine", &combine, nb::arg("p"), nb::arg("w").none(), nb::arg("method"), nb::arg("omit"));
    m.def("tukey", &tukey);
    m.def("srange_isf", &srange_isf);
    m.def("srange_sf", &srange_sf);
    m.def("dunnett", &dunnett_test);
    m.def("dunnett_isf", &dunnett_isf);
    m.def("dunnett_tail", &dunnett_tail);
}

}  // namespace sd::htest
