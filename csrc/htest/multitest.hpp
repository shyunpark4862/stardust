// The multiple-testing corrections of sd.htest (milestone H8, docs/design/htest.md §3.9): one
// kernel for statsmodels' multipletests (its ten methods, H102, H106) and scipy's
// false_discovery_control ('bh' and 'by' are fdr_bh and fdr_by here), so the two give the same
// bits for the same p-values.
//
// One family of m p-values (in [0, 1], no NaN: the caller checks) is sorted (stably: equal
// p-values get equal corrections whatever their order) and corrected in that order:
//
//   bonferroni      m p
//   sidak           1 - (1 - p)^m = -expm1(m log1p(-p))  (no cancellation for small p)
//   holm-sidak      the running maximum of 1 - (1 - p_(i))^(m - i + 1)   (step down)
//   holm            the running maximum of (m - i + 1) p_(i)
//   simes-hochberg  the running minimum from the top of (m - i + 1) p_(i)  (step up)
//   hommel          Hommel's adjusted p-values (below), O(m log m)
//   fdr_bh          the running minimum from the top of m p_(i) / i
//   fdr_by          the same times the harmonic number H_m
//   fdr_tsbh/tsbky  the two-stage procedures of Benjamini, Krieger and Yekutieli (statsmodels'
//                   fdrcorrection_twostage): fdr_bh's values times m0 / m (times 1 + alpha for
//                   tsbky), m0 the estimated number of true nulls after max_iter stages
//
// each clipped at 1. Every corrected value is carried as a double-double (the products and
// quotients m p / i and H_m; for the Sidak forms the exponent y = n log1p(-p), the value being
// -expm1(y)), rounded once at the end, and the rejection is the corrected value's against alpha
// in double-double (H111): the procedure's decision on the given doubles, not on a rounded
// corrected value (fdr_tsbh on the problems' p-values: 0.07 of 20 is at 0.05 + 5e-18, rounded to
// 0.05 = alpha, and is not rejected).
//
// Hommel's adjusted p-value (Hommel 1988; Wright 1992's formula, R's p.adjust and statsmodels'
// O(m^2) loops give the same values) is p~_i = min over k = 0..m of max(S_(k+1), k p_i), where
// s_k = k min_(1<=j<=k) p_(m-k+j) / j is Simes' p-value of the k largest p-values and S_k the
// largest s_k' with k' >= k (S_(m+1) = 0): H_i is rejected at alpha when p_i <= alpha / h(alpha),
// h(alpha) the largest k with s_k > alpha, and h(alpha) = k exactly on [S_(k+1), S_k). The
// minimum over j of q_t / (t - a + 1) (a = m - k + 1) is the least slope from the point (a, 0)
// to the points (t + 1, q_t), t >= a: a tangent to their lower convex hull, kept as a stack while
// a goes down and searched by bisection (Meijer, Krebs and Goeman 2019 reach the same
// quantities in linear time). The minimum over k is where k p_i crosses S_(k+1): a bisection.
#pragma once

#include "special/dd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::htest::multitest {

namespace dd = special::dd;
using dd::DD;

enum Method {
    kBonferroni = 0,
    kSidak = 1,
    kHolmSidak = 2,
    kHolm = 3,
    kSimesHochberg = 4,
    kHommel = 5,
    kFdrBH = 6,
    kFdrBY = 7,
    kFdrTSBH = 8,
    kFdrTSBKY = 9,
};

inline double cap1(double v) { return v > 1.0 ? 1.0 : v; }

inline bool le(DD a, DD b) {
    const DD d = a - b;
    return d.hi < 0.0 || (d.hi == 0.0 && d.lo <= 0.0);
}
inline DD dmax(DD a, DD b) { return le(a, b) ? b : a; }
inline DD dmin(DD a, DD b) { return le(a, b) ? a : b; }

// The harmonic number H_m as a double-double (fdr_by's c(m)): the terms from the small end.
inline DD harmonic(std::int64_t m) {
    DD s{0.0, 0.0};
    for (std::int64_t i = m; i >= 1; --i) s = s + DD{1.0, 0.0} / static_cast<double>(i);
    return s;
}

// log1p(-p) as a double-double (p in [0, 1]): -p - mu(-p), mu(t) = t - log(1 + t)
inline DD log1m(double p) {
    if (p >= 1.0) return DD{-std::numeric_limits<double>::infinity(), 0.0};
    if (p == 0.0) return DD{0.0, 0.0};
    return DD{-p, 0.0} - dd::log1pmx_neg(DD{-p, 0.0});
}
// n log1p(-p) (-inf at p = 1)
inline DD sidak_exponent(double p, double n) {
    const DD l = log1m(p);
    return std::isfinite(l.hi) ? l * n : DD{l.hi, 0.0};
}
// -expm1(y) for y = hi + lo <= 0, rounded once (first order in lo)
inline double sidak_value(DD y) {
    if (!std::isfinite(y.hi)) return y.hi < 0.0 ? 1.0 : std::numeric_limits<double>::quiet_NaN();
    return -std::expm1(y.hi) - std::exp(y.hi) * y.lo;
}

// Hommel's adjusted p-values of q (sorted ascending, m of them) into c (the same order).
inline void hommel(const double* q, std::int64_t m, DD* c) {
    std::vector<DD> S(static_cast<std::size_t>(m + 2), DD{0.0, 0.0});
    std::vector<std::int64_t> st;  // the lower hull's points t (1-based), st.back() the leftmost
    st.reserve(static_cast<std::size_t>(m));
    auto X = [](std::int64_t t) { return static_cast<double>(t + 1); };
    auto Y = [&](std::int64_t t) { return q[t - 1]; };
    for (std::int64_t a = m; a >= 1; --a) {
        while (st.size() >= 2) {  // the top is not below the segment from the new point to the next
            const std::int64_t u = st[st.size() - 1], v = st[st.size() - 2];
            const double cr = (X(u) - X(a)) * (Y(v) - Y(a)) - (Y(u) - Y(a)) * (X(v) - X(a));
            if (cr <= 0.0) st.pop_back();
            else break;
        }
        st.push_back(a);
        // the slope from (a, 0) along the hull from its left end is unimodal: bisection
        const auto n = static_cast<std::int64_t>(st.size());
        auto vert = [&](std::int64_t i) { return st[static_cast<std::size_t>(n - 1 - i)]; };
        auto slope = [&](std::int64_t i) {
            const std::int64_t t = vert(i);
            return Y(t) / static_cast<double>(t + 1 - a);
        };
        std::int64_t lo = 0, hi = n - 1;
        while (lo < hi) {
            const std::int64_t mid = (lo + hi) / 2;
            if (slope(mid + 1) < slope(mid)) lo = mid + 1;
            else hi = mid;
        }
        const std::int64_t k = m - a + 1, t = vert(lo);
        S[static_cast<std::size_t>(k)] = dd::two_prod(Y(t), static_cast<double>(k)) / static_cast<double>(t + 1 - a);
    }
    for (std::int64_t k = m; k >= 1; --k)  // S_k = max_(k' >= k) s_k'
        S[static_cast<std::size_t>(k)] = dmax(S[static_cast<std::size_t>(k)], S[static_cast<std::size_t>(k + 1)]);
    for (std::int64_t i = 0; i < m; ++i) {
        // the least k >= 0 with k q_i >= S_(k+1) (k = m always is: S_(m+1) = 0)
        std::int64_t lo = 0, hi = m;
        while (lo < hi) {
            const std::int64_t mid = (lo + hi) / 2;
            if (le(S[static_cast<std::size_t>(mid + 1)], dd::two_prod(static_cast<double>(mid), q[i]))) hi = mid;
            else lo = mid + 1;
        }
        const DD v =
            lo == 0 ? S[1] : dmin(dd::two_prod(static_cast<double>(lo), q[i]), S[static_cast<std::size_t>(lo)]);
        c[i] = dmax(v, DD{q[i], 0.0});
    }
}

// fdr_bh's corrected values of the sorted q times factor / m (the running minimum from the top
// of factor q_(i) / i), into c.
inline void bh_sorted(const double* q, std::int64_t m, DD factor, DD* c) {
    DD run{0.0, 0.0};
    for (std::int64_t i = m - 1; i >= 0; --i) {
        const DD v = factor * q[i] / static_cast<double>(i + 1);
        run = i == m - 1 ? v : dmin(run, v);
        c[i] = run;
    }
}

// The rejections of the BH step-up rule at the two-stage level alpha' m / n0 (alpha' = alpha /
// (1 + alpha) for tsbky, alpha for tsbh) on the sorted q: the largest i with q_(i) n0 (1 +
// alpha) <= i alpha (tsbky; q_(i) n0 <= i alpha for tsbh), in double-double.
inline std::int64_t bh_rejections(const double* q, std::int64_t m, double n0, double alpha, bool bky) {
    for (std::int64_t i = m; i >= 1; --i) {
        DD lhs = dd::two_prod(q[i - 1], n0);
        if (bky) lhs = lhs + lhs * alpha;
        if (le(lhs, dd::two_prod(static_cast<double>(i), alpha))) return i;
    }
    return 0;
}

// statsmodels' fdrcorrection_twostage on the sorted q: the corrected values into c.
inline void two_stage(const double* q, std::int64_t m, double alpha, std::int64_t max_iter, bool bky, DD* c) {
    const double fm = static_cast<double>(m);
    bh_sorted(q, m, DD{fm, 0.0}, c);
    for (std::int64_t i = 0; i < m; ++i)  // (fdrcorrection's own clip)
        if (!le(c[i], DD{1.0, 0.0})) c[i] = DD{1.0, 0.0};
    std::int64_t r = bh_rejections(q, m, fm, alpha, bky);
    double n0 = fm;
    if (r != 0 && r != m) {
        std::int64_t old = r;
        for (std::int64_t it = 0; it < max_iter; ++it) {
            n0 = fm - static_cast<double>(old);
            r = bh_rejections(q, m, n0, alpha, bky);
            if (it >= max_iter - 1 || r == old) break;
            old = r;
        }
    }
    DD f = DD{n0, 0.0} / fm;  // n0 / m, times 1 + alpha for tsbky
    if (bky) f = f + f * alpha;
    for (std::int64_t i = 0; i < m; ++i) c[i] = f * c[i];
}

// One family p[0, m) (stride apart): the corrected p-values into out and the rejections at
// alpha (1 or 0) into rej (null: not wanted), in the input's order. max_iter matters to the
// two-stage methods only. The scratch is reused between families.
struct Scratch {
    std::vector<std::int64_t> order;
    std::vector<double> q;
    std::vector<DD> c;
};
inline void correct(
    const double* p,
    std::int64_t stride,
    std::int64_t m,
    int method,
    double alpha,
    std::int64_t max_iter,
    double* out,
    double* rej,
    Scratch& w
) {
    if (m == 0) return;
    w.order.resize(static_cast<std::size_t>(m));
    w.q.resize(static_cast<std::size_t>(m));
    w.c.resize(static_cast<std::size_t>(m));
    std::iota(w.order.begin(), w.order.end(), std::int64_t{0});
    std::stable_sort(w.order.begin(), w.order.end(), [&](std::int64_t a, std::int64_t b) {
        return p[a * stride] < p[b * stride];
    });
    double* q = w.q.data();
    DD* c = w.c.data();
    for (std::int64_t i = 0; i < m; ++i) q[i] = p[w.order[static_cast<std::size_t>(i)] * stride];
    const double fm = static_cast<double>(m);
    const bool logs = method == kSidak || method == kHolmSidak;  // c holds y, the value -expm1(y)
    switch (method) {
        case kBonferroni:
            for (std::int64_t i = 0; i < m; ++i) c[i] = dd::two_prod(q[i], fm);
            break;
        case kSidak:
            for (std::int64_t i = 0; i < m; ++i) c[i] = sidak_exponent(q[i], fm);
            break;
        case kHolmSidak: {  // the running maximum of -expm1(y): the running minimum of y
            DD run{0.0, 0.0};
            for (std::int64_t i = 0; i < m; ++i) {
                const DD y = sidak_exponent(q[i], fm - static_cast<double>(i));
                run = i == 0 ? y : dmin(run, y);
                c[i] = run;
            }
            break;
        }
        case kHolm: {
            DD run{0.0, 0.0};
            for (std::int64_t i = 0; i < m; ++i) {
                run = dmax(run, dd::two_prod(q[i], fm - static_cast<double>(i)));
                c[i] = run;
            }
            break;
        }
        case kSimesHochberg: {
            DD run{0.0, 0.0};
            for (std::int64_t i = m - 1; i >= 0; --i) {
                const DD v = dd::two_prod(q[i], fm - static_cast<double>(i));
                run = i == m - 1 ? v : dmin(run, v);
                c[i] = run;
            }
            break;
        }
        case kHommel: hommel(q, m, c); break;
        case kFdrBH: bh_sorted(q, m, DD{fm, 0.0}, c); break;
        case kFdrBY: bh_sorted(q, m, harmonic(m) * fm, c); break;
        case kFdrTSBH:
        case kFdrTSBKY: two_stage(q, m, alpha, max_iter, method == kFdrTSBKY, c); break;
        default: break;
    }
    // -expm1(y) <= alpha  <=>  y >= log1p(-alpha)
    const DD la = logs ? log1m(alpha) : DD{alpha, 0.0};
    for (std::int64_t i = 0; i < m; ++i) {
        const std::int64_t at = w.order[static_cast<std::size_t>(i)];
        out[at] = cap1(logs ? sidak_value(c[i]) : c[i].hi + c[i].lo);
        if (rej) rej[at] = (logs ? le(la, c[i]) : le(c[i], la)) ? 1.0 : 0.0;
    }
}

}  // namespace sd::htest::multitest
