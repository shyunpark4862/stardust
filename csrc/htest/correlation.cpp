// The correlation tests (milestone H5, docs/design/htest.md §3.6).
//
// The Python side (stardust.htest._correlation) checks the arguments and arranges the paired
// samples as 2-D float64 arrays, one lane (one test of a batch) per row; each kernel returns the
// per-lane results as 1-D arrays. Lanes are independent: the same lane gives the same bits alone
// or in a batch, on any thread count.
//
//   pearson      r, p-value, Fisher's z = atanh r, n (pearsonr, pointbiserialr; ranks=True:
//                spearmanrho)
//   corr_matrix  spearmanr's matrix: every pair of variables by the same lane computation
//   kendall      tau-b or tau-c, p-value, method, exact work, a flag for 'exact' with ties
//   weightedtau  Vigna's weighted tau (p-value: none, scipy's)
//   somersd      Somers' D(Y|X) and its p-value from a contingency table; crosstab builds one
//   chatterjee   Chatterjee's xi and its p-value
//
// Pearson's r: the values in a unit of 2^-e (e the exponent of the largest magnitude; exact),
// the means, the centred values, Sxx, Syy and Sxy in double-doubles, so r = Sxy / sqrt(Sxx Syy)
// keeps its digits whatever the offset. The p-value depends on q = 1 - r^2, which is formed
// without cancellation: 1 - r^2 from the double-double r while r^2 < 1/2, else the residual of
// the centred y projected on the centred x, q = |y_c - b x_c|^2 / Syy with b = Sxy / Sxx (the
// regression's way). Its error is ~2^-105 / sqrt(q) relative, where scipy's 1 - r^2 from a
// rounded r loses everything near |r| = 1 (p 0.11 off at r = 1 - 1e-14, docs/design/htest.md
// §7). The p-value is the beta distribution of r (scipy's): two-sided I_q((n - 2)/2, 1/2), from
// sd.special's incomplete beta given both q and 1 - q (no rounding of 1 - q on the way in).
// Fisher's z is (1/2) log((1 + |r|)/(1 - |r|)) with 1 - |r| = q / (1 + |r|).
//
// Zero variance (H20): a sample with m2 <= (eps mean)^2 (sd.stats skew's rule) gives NaN, as do
// fewer than two pairs, an infinite value (Pearson's r of an inf is undefined) and a NaN under
// 'propagate'. n = 2 gives r = +-1 and p = 1 (scipy's).
#include "htest/correlation.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "htest/kendall.hpp"
#include "htest/lanes.hpp"
#include "htest/moments.hpp"
#include "htest/rank_exact.hpp"
#include "special/ops/incbeta.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;
using exact::i128;

double norm_sf(double z) { return stats::method_of<stats::Normal, m::Ccdf>(z, 0.0, 1.0); }
double norm_cdf(double z) { return stats::method_of<stats::Normal, m::Cdf>(z, 0.0, 1.0); }
// scipy's _get_pvalue for the standard normal
double normal_p(double z, int alt) {
    if (z != z) return z;
    if (alt == 0) return std::min(1.0, 2.0 * norm_sf(std::fabs(z)));
    return alt == 1 ? norm_cdf(z) : norm_sf(z);
}

constexpr double kUsedExact = 0.0, kUsedAsymptotic = 1.0;
// the exact distribution's memory: at most this many entries at once (H13, with the work cap)
constexpr double kStateCap = 1e7;

DD sqrt_dd(DD a) {
    if (!(a.hi > 0.0)) return DD{a.hi == 0.0 ? 0.0 : kNaN, 0.0};
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    return dd::two_sum(s, ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s));
}
DD dd_of(i128 v) { return exact::to_dd(v); }
DD dd_of(std::int64_t v) { return exact::to_dd(static_cast<i128>(v)); }
DD clip_unit(DD r) {
    if (r.hi > 1.0 || (r.hi == 1.0 && r.lo > 0.0)) return DD{1.0, 0.0};
    if (r.hi < -1.0 || (r.hi == -1.0 && r.lo < 0.0)) return DD{-1.0, 0.0};
    return r;
}

// Lane r's pairs (x_j, y_j) without those holding a NaN; true when one was met.
bool lane_pairs(const Rows& a, const Rows& b, std::int64_t r, std::vector<double>& x, std::vector<double>& y) {
    x.clear();
    y.clear();
    bool nan = false;
    for (std::int64_t j = 0; j < a.n; ++j) {
        const double u = a.at(r, j), v = b.at(r, j);
        if (u != u || v != v) {
            nan = true;
            continue;
        }
        x.push_back(u);
        y.push_back(v);
    }
    return nan;
}

// The doubled midranks of v in v's order (a tie run of t values from sorted position b gets
// 2b + t + 1; exact integers). The sort is htest's sort_keys (on the pool for a long lane).
void midranks2(const std::vector<double>& v, std::vector<double>& out) {
    const auto n = static_cast<std::int64_t>(v.size());
    std::vector<Key> k(static_cast<std::size_t>(n));
    for (std::int64_t j = 0; j < n; ++j)
        k[static_cast<std::size_t>(j)] = Key{v[static_cast<std::size_t>(j)], 0.0, static_cast<std::int32_t>(j)};
    sort_keys(k);
    out.assign(static_cast<std::size_t>(n), 0.0);
    std::int64_t i = 0;
    while (i < n) {
        std::int64_t e = i + 1;
        while (e < n && key_equal(k[static_cast<std::size_t>(e)], k[static_cast<std::size_t>(i)])) ++e;
        const double rk = static_cast<double>(i + e + 1);  // 2b + t + 1, t = e - i
        for (std::int64_t t = i; t < e; ++t) out[static_cast<std::size_t>(k[static_cast<std::size_t>(t)].tag)] = rk;
        i = e;
    }
}

// ---- Pearson's r and q = 1 - r^2 ----------------------------------------------------------

struct RQ {
    DD r{kNaN, 0.0}, q{kNaN, 0.0}, w{kNaN, 0.0};  // r, 1 - r^2 and r^2 (q + w = 1)
    bool ok = false;
};

// sd.stats skew's zero-variance rule (H20): m2 = ss / n at most (eps mean)^2
bool constant(DD ss, DD mean, std::int64_t n) {
    const double m2 = ss.hi / static_cast<double>(n), e = kEps * mean.hi;
    return m2 <= e * e;
}

RQ corr_rq(const double* x, const double* y, std::int64_t n) {
    RQ o;
    if (n < 2) return o;
    double mx = 0.0, my = 0.0;
    for (std::int64_t j = 0; j < n; ++j) {
        if (!std::isfinite(x[j]) || !std::isfinite(y[j])) return o;
        mx = std::max(mx, std::fabs(x[j]));
        my = std::max(my, std::fabs(y[j]));
    }
    if (!(mx > 0.0) || !(my > 0.0)) return o;  // a sample of zeros: constant
    const double fx = std::ldexp(1.0, -std::ilogb(mx)), fy = std::ldexp(1.0, -std::ilogb(my));
    Acc4 sx, sy;
    for (std::int64_t j = 0; j < n; ++j) {
        sx.add(j, x[j] * fx, 0.0);
        sy.add(j, y[j] * fy, 0.0);
    }
    const DD mxd = sx.total().value() / static_cast<double>(n), myd = sy.total().value() / static_cast<double>(n);
    std::vector<DD> xc(static_cast<std::size_t>(n)), yc(static_cast<std::size_t>(n));
    Acc4 sxx, syy, sxy;
    for (std::int64_t j = 0; j < n; ++j) {
        const DD a = dd::two_sum(x[j] * fx, -mxd.hi) + (-mxd.lo), b = dd::two_sum(y[j] * fy, -myd.hi) + (-myd.lo);
        xc[static_cast<std::size_t>(j)] = a;
        yc[static_cast<std::size_t>(j)] = b;
        const DD aa = a * a, bb = b * b, ab = a * b;
        sxx.add(j, aa.hi, aa.lo);
        syy.add(j, bb.hi, bb.lo);
        sxy.add(j, ab.hi, ab.lo);
    }
    const DD Sxx = sxx.total().value(), Syy = syy.total().value(), Sxy = sxy.total().value();
    if (!(Sxx.hi > 0.0) || !(Syy.hi > 0.0) || constant(Sxx, mxd, n) || constant(Syy, myd, n)) return o;
    o.r = clip_unit(Sxy / sqrt_dd(Sxx * Syy));
    const DD w = o.r * o.r;
    if (w.hi < 0.5) {
        o.w = w;
        o.q = DD{1.0, 0.0} - w;
    } else {
        const DD b = Sxy / Sxx;
        Acc4 se;
        for (std::int64_t j = 0; j < n; ++j) {
            const DD e = yc[static_cast<std::size_t>(j)] - b * xc[static_cast<std::size_t>(j)];
            const DD e2 = e * e;
            se.add(j, e2.hi, e2.lo);
        }
        o.q = se.total().value() / Syy;
        if (o.q.hi > 0.5) o.q = DD{1.0, 0.0} - w;  // (rounding past the branch's side: take 1 - r^2)
        if (o.q.hi < 0.0) o.q = DD{0.0, 0.0};
        o.w = DD{1.0, 0.0} - o.q;
    }
    o.ok = true;
    return o;
}

// The beta distribution of r (scipy's pearsonr): alt 0 two-sided, 1 less, 2 greater
double corr_p(const RQ& c, std::int64_t n, int alt) {
    if (!c.ok) return kNaN;
    if (n == 2) return 1.0;
    double i, ic;  // I_q((n - 2)/2, 1/2) and its complement
    if (c.q.hi <= 0.0) {
        i = 0.0, ic = 1.0;
    } else if (c.w.hi <= 0.0) {
        i = 1.0, ic = 0.0;
    } else {
        const special::ibeta_detail::IB b =
            special::ibeta_detail::ibeta(0.5 * static_cast<double>(n - 2), 0.5, c.q, c.w);
        i = b.i, ic = b.ic;
    }
    if (alt == 0) return i;
    const bool upper = (c.r.hi >= 0.0) == (alt == 2);
    return upper ? 0.5 * i : 0.5 + 0.5 * ic;
}

// Fisher's z = atanh r
double fisher_z(const RQ& c) {
    if (!c.ok) return kNaN;
    if (c.q.hi <= 0.0) return std::copysign(kInf, c.r.hi);
    const DD a = abs_dd(c.r);
    if (a.hi == 0.0) return 0.0;
    const DD up = DD{1.0, 0.0} + a, dn = c.q / up;
    const double z = to_double(dd::log(up / dn) * 0.5);
    return c.r.hi < 0.0 ? -z : z;
}

struct Pearson {
    double r = kNaN, p = kNaN, z = kNaN, n = kNaN;
};

Pearson pearson_of(const double* x, const double* y, std::int64_t n, int alt) {
    const RQ c = corr_rq(x, y, n);
    Pearson o;
    o.n = static_cast<double>(n);
    if (!c.ok) return o;
    o.r = to_double(c.r);
    if (n == 2) o.r = o.r > 0.0 ? 1.0 : -1.0;
    o.p = corr_p(c, n, alt);
    o.z = fisher_z(c);
    return o;
}

nb::tuple pearson(nb::handle x, nb::handle y, int alt, bool omit, bool ranks) {
    const Rows a(rows_arg(x)), b(rows_arg(y));
    Outs out(4, a.lanes);
    each_lane(a.lanes, (ranks ? 40 : 12) * a.n, [&](std::int64_t r) {
        std::vector<double> u, v, ru, rv;
        const bool nan = lane_pairs(a, b, r, u, v);
        Pearson o;
        if (!(nan && !omit)) {
            if (ranks) {
                midranks2(u, ru);
                midranks2(v, rv);
                u.swap(ru);
                v.swap(rv);
            }
            o = pearson_of(u.data(), v.data(), static_cast<std::int64_t>(u.size()), alt);
        }
        out.p[0][r] = o.r, out.p[1][r] = o.p, out.p[2][r] = o.z, out.p[3][r] = o.n;
    });
    return out.tuple();
}

// spearmanr's matrix of k variables (one per row of cols, n observations): entry (i, j) is the
// lane computation of spearmanrho(cols[i], cols[j]) (the same bits), mirrored; the diagonal is
// 1 and p 0 for a variable of at least two values that are not all equal. 'propagate': a
// variable holding a NaN has a NaN row and column. 'omit': each pair drops the observations
// where either is NaN (scipy's mstats, pairwise).
nb::tuple corr_matrix(nb::handle cols, int alt, bool omit) {
    const Rows c(rows_arg(cols));
    const std::int64_t k = c.lanes, n = c.n;
    std::vector<std::vector<double>> rk(static_cast<std::size_t>(k));
    std::vector<char> nan(static_cast<std::size_t>(k), 0), diag(static_cast<std::size_t>(k), 0);
    each_lane(k, 40 * n, [&](std::int64_t i) {
        std::vector<double> v;
        bool has = false;
        for (std::int64_t j = 0; j < n; ++j) {
            const double t = c.at(i, j);
            if (t != t) {
                has = true;
                continue;
            }
            v.push_back(t);
        }
        nan[static_cast<std::size_t>(i)] = has;
        bool varies = false;
        for (std::size_t j = 1; j < v.size(); ++j) varies = varies || !(v[j] == v[0]);
        diag[static_cast<std::size_t>(i)] = varies && (omit || !has);
        if (!has) midranks2(v, rk[static_cast<std::size_t>(i)]);
    });
    Outs out(2, k * k);
    for (std::int64_t i = 0; i < k; ++i) {
        const bool d = diag[static_cast<std::size_t>(i)];
        out.p[0][i * k + i] = d ? 1.0 : kNaN, out.p[1][i * k + i] = d ? 0.0 : kNaN;
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
    for (std::int64_t i = 0; i < k; ++i)
        for (std::int64_t j = i + 1; j < k; ++j) pairs.emplace_back(i, j);
    each_lane(static_cast<std::int64_t>(pairs.size()), 12 * n, [&](std::int64_t q) {
        const auto [i, j] = pairs[static_cast<std::size_t>(q)];
        Pearson o;
        const bool ni = nan[static_cast<std::size_t>(i)], nj = nan[static_cast<std::size_t>(j)];
        if (!ni && !nj) {
            o = pearson_of(rk[static_cast<std::size_t>(i)].data(), rk[static_cast<std::size_t>(j)].data(), n, alt);
        } else if (omit) {
            std::vector<double> u, v, ru, rv;
            for (std::int64_t t = 0; t < n; ++t) {
                const double a = c.at(i, t), b = c.at(j, t);
                if (a == a && b == b) u.push_back(a), v.push_back(b);
            }
            midranks2(u, ru);
            midranks2(v, rv);
            o = pearson_of(ru.data(), rv.data(), static_cast<std::int64_t>(ru.size()), alt);
        }
        out.p[0][i * k + j] = out.p[0][j * k + i] = o.r;
        out.p[1][i * k + j] = out.p[1][j * k + i] = o.p;
    });
    return out.tuple();
}

// ---- Kendall's tau ------------------------------------------------------------------------

// The dense codes (0, 1, ...) of v's values in v's order, and the tie sums of the runs:
// sum t(t - 1)/2, sum t(t - 1)(t - 2), sum t(t - 1)(2t + 5), the number of distinct values.
struct Ties {
    i128 pairs = 0, t3 = 0, t5 = 0;
    std::int64_t distinct = 0;
};
Ties dense_codes(const std::vector<double>& v, std::vector<std::uint32_t>& code) {
    const auto n = static_cast<std::int64_t>(v.size());
    std::vector<Key> k(static_cast<std::size_t>(n));
    for (std::int64_t j = 0; j < n; ++j)
        k[static_cast<std::size_t>(j)] = Key{v[static_cast<std::size_t>(j)], 0.0, static_cast<std::int32_t>(j)};
    sort_keys(k);
    code.assign(static_cast<std::size_t>(n), 0);
    Ties t;
    std::int64_t i = 0;
    while (i < n) {
        std::int64_t e = i + 1;
        while (e < n && key_equal(k[static_cast<std::size_t>(e)], k[static_cast<std::size_t>(i)])) ++e;
        const i128 c = e - i;
        t.pairs += c * (c - 1) / 2;
        t.t3 += c * (c - 1) * (c - 2);
        t.t5 += c * (c - 1) * (2 * c + 5);
        for (std::int64_t s = i; s < e; ++s)
            code[static_cast<std::size_t>(k[static_cast<std::size_t>(s)].tag)] = static_cast<std::uint32_t>(t.distinct);
        ++t.distinct;
        i = e;
    }
    return t;
}

// the pairs i < j with v_i > v_j (a bottom-up merge sort)
std::int64_t inversions(std::vector<std::uint32_t>& v) {
    const auto n = static_cast<std::int64_t>(v.size());
    std::vector<std::uint32_t> buf(v.size());
    std::int64_t inv = 0;
    for (std::int64_t w = 1; w < n; w *= 2) {
        for (std::int64_t lo = 0; lo < n; lo += 2 * w) {
            const std::int64_t mid = std::min(n, lo + w), hi = std::min(n, lo + 2 * w);
            std::int64_t i = lo, j = mid, o = lo;
            while (i < mid && j < hi) {
                if (v[static_cast<std::size_t>(j)] < v[static_cast<std::size_t>(i)]) {
                    inv += mid - i;
                    buf[static_cast<std::size_t>(o++)] = v[static_cast<std::size_t>(j++)];
                } else {
                    buf[static_cast<std::size_t>(o++)] = v[static_cast<std::size_t>(i++)];
                }
            }
            while (i < mid) buf[static_cast<std::size_t>(o++)] = v[static_cast<std::size_t>(i++)];
            while (j < hi) buf[static_cast<std::size_t>(o++)] = v[static_cast<std::size_t>(j++)];
        }
        v.swap(buf);
    }
    return inv;
}

struct KOut {
    double tau = kNaN, p = kNaN, method = kUsedAsymptotic, work = 0.0, tie_exact = 0.0;
};

// variant 0 'b', 1 'c'; req 0 auto, 1 exact, 2 asymptotic (scipy's rules, H46). nan: the lane's
// results are NaN ('propagate'), the method still decided on its other values (H53).
KOut kendall_lane(
    const std::vector<double>& x, const std::vector<double>& y, int variant, int alt, int req, double cap, bool nan
) {
    KOut o;
    const auto n = static_cast<std::int64_t>(x.size());
    std::vector<std::uint32_t> cx, cy;
    const Ties tx = dense_codes(x, cx), ty = dense_codes(y, cy);
    // sorted by (x, y): the joint ties and the y sequence whose inversions are the discordant pairs
    std::vector<std::uint64_t> pk(static_cast<std::size_t>(n));
    for (std::int64_t j = 0; j < n; ++j)
        pk[static_cast<std::size_t>(j)] =
            (static_cast<std::uint64_t>(cx[static_cast<std::size_t>(j)]) << 32) | cy[static_cast<std::size_t>(j)];
    std::sort(pk.begin(), pk.end());
    i128 ntie = 0;
    for (std::int64_t i = 0; i < n;) {
        std::int64_t e = i + 1;
        while (e < n && pk[static_cast<std::size_t>(e)] == pk[static_cast<std::size_t>(i)]) ++e;
        const i128 c = e - i;
        ntie += c * (c - 1) / 2;
        i = e;
    }
    std::vector<std::uint32_t> ys(static_cast<std::size_t>(n));
    for (std::int64_t j = 0; j < n; ++j)
        ys[static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(pk[static_cast<std::size_t>(j)] & 0xffffffffu);
    const i128 dis = inversions(ys);
    const i128 tot = static_cast<i128>(n) * (n - 1) / 2;
    const bool ties = tx.pairs != 0 || ty.pairs != 0;
    // the method (scipy's 'auto': exact without ties when n <= 33 or min(dis, tot - dis) <= 1),
    // within the exact work cap
    const i128 cm = std::min(dis, tot - dis);
    const double work = static_cast<double>(n) * (static_cast<double>(cm) + 1.0);
    const bool within = work <= cap && static_cast<double>(cm) + 1.0 <= kStateCap;
    bool use_exact = false;
    if (req == 0) use_exact = !ties && (n <= 33 || cm <= 1) && within;
    if (req == 1) {
        if (ties) {
            o.tie_exact = 1.0;
        } else {
            use_exact = true;
            if (!within) o.work = work;
        }
    }
    o.method = use_exact ? kUsedExact : kUsedAsymptotic;
    if (nan || n < 2 || tx.pairs == tot || ty.pairs == tot || o.tie_exact != 0.0 || o.work != 0.0) return o;
    const i128 cmd = tot - tx.pairs - ty.pairs + ntie - 2 * dis;
    DD tau;
    if (variant == 0) {
        tau = dd_of(cmd) / sqrt_dd(dd_of(tot - tx.pairs) * dd_of(tot - ty.pairs));
    } else {
        const i128 mc = std::min(tx.distinct, ty.distinct);
        tau = dd_of(2 * cmd * mc) / dd_of(static_cast<i128>(n) * n * (mc - 1));
    }
    o.tau = to_double(clip_unit(tau));
    if (use_exact) {
        o.p = kendall::exact_p(n, static_cast<std::int64_t>(tot - dis), alt);
    } else {
        // scipy's tie-corrected variance of con - dis
        const DD mm = dd_of(static_cast<i128>(n) * (n - 1));
        DD var = dd_of(static_cast<i128>(n) * (n - 1) * (2 * n + 5) - tx.t5 - ty.t5) / 18.0;
        var = var + dd_of(2 * tx.pairs) * dd_of(ty.pairs) / mm;
        if (n > 2 && tx.t3 != 0 && ty.t3 != 0)
            var = var + dd_of(tx.t3) * dd_of(ty.t3) / (mm * dd_of(9 * static_cast<i128>(n - 2)));
        o.p = normal_p(to_double(dd_of(cmd) / sqrt_dd(var)), alt);
    }
    return o;
}

nb::tuple kendall_tau(nb::handle x, nb::handle y, int variant, int alt, int req, double cap, bool omit) {
    const Rows a(rows_arg(x)), b(rows_arg(y));
    Outs out(5, a.lanes);
    each_lane(a.lanes, 40 * a.n, [&](std::int64_t r) {
        std::vector<double> u, v;
        const bool nan = lane_pairs(a, b, r, u, v);
        const KOut o = kendall_lane(u, v, variant, alt, req, cap, nan && !omit);
        out.p[0][r] = o.tau, out.p[1][r] = o.p, out.p[2][r] = o.method, out.p[3][r] = o.work, out.p[4][r] = o.tie_exact;
    });
    return out.tuple();
}

// ---- the weighted tau (Vigna 2015; scipy's _weightedrankedtau) -----------------------------
//
// The elements are sorted by (x, y) (and their position: a stable sort), the weighted ties and
// the weighted exchanges of a merge sort on y counted, every sum a double-double (the
// exchanges' running residual subtracts weights: rounded to doubles it loses the small ones).

struct WTau {
    const double *x, *y;
    const DD* w;  // per element
    bool additive;
    std::vector<std::int64_t> perm, temp;
    DD exch{0.0, 0.0};

    DD weigh(std::int64_t off, std::int64_t len) {
        if (len == 1) return w[perm[static_cast<std::size_t>(off)]];
        const std::int64_t len0 = len / 2, len1 = len - len0, mid = off + len0;
        DD residual = weigh(off, len0);
        const DD weight = weigh(mid, len1) + residual;
        if (y[perm[static_cast<std::size_t>(mid - 1)]] < y[perm[static_cast<std::size_t>(mid)]]) return weight;
        std::int64_t i = 0, j = 0, k = 0;
        while (j < len0 && k < len1) {
            const std::int64_t a = perm[static_cast<std::size_t>(off + j)], b = perm[static_cast<std::size_t>(mid + k)];
            if (y[a] <= y[b]) {
                temp[static_cast<std::size_t>(i)] = a;
                residual = residual - w[a];
                ++j;
            } else {
                temp[static_cast<std::size_t>(i)] = b;
                exch = exch + (additive ? w[b] * static_cast<double>(len0 - j) + residual : w[b] * residual);
                ++k;
            }
            ++i;
        }
        std::copy_backward(perm.begin() + off + j, perm.begin() + off + len0, perm.begin() + off + i + len0 - j);
        std::copy(temp.begin(), temp.begin() + i, perm.begin() + off);
        return weight;
    }

    // the weighted ties of perm's runs equal by eq
    template <class Eq>
    DD tie_weight(std::int64_t n, Eq eq, std::int64_t& first) {
        first = 0;
        DD t{0.0, 0.0}, s = w[perm[0]], sq = s * s;
        auto close = [&](std::int64_t i) {
            t = t + (additive ? s * static_cast<double>(i - first - 1) : (s * s - sq) * 0.5);
        };
        for (std::int64_t i = 1; i < n; ++i) {
            if (!eq(perm[static_cast<std::size_t>(first)], perm[static_cast<std::size_t>(i)])) {
                close(i);
                first = i;
                s = sq = DD{0.0, 0.0};
            }
            const DD v = w[perm[static_cast<std::size_t>(i)]];
            s = s + v;
            sq = sq + v * v;
        }
        close(n);
        return t;
    }
};

// tau of n elements, perm their order by (x, y, position)
DD wtau_core(
    const double* x, const double* y, const DD* w, std::int64_t n, bool additive, std::vector<std::int64_t> perm
) {
    const DD nan{kNaN, 0.0};
    if (n < 1) return nan;
    WTau s{x, y, w, additive, std::move(perm), std::vector<std::int64_t>(static_cast<std::size_t>(n)), DD{0.0, 0.0}};
    std::int64_t first;
    const DD t = s.tie_weight(n, [&](std::int64_t a, std::int64_t b) { return x[a] == x[b] && y[a] == y[b]; }, first);
    const DD u = s.tie_weight(n, [&](std::int64_t a, std::int64_t b) { return x[a] == x[b]; }, first);
    if (first == 0) return nan;  // x constant
    s.weigh(0, n);
    const DD v = s.tie_weight(n, [&](std::int64_t a, std::int64_t b) { return y[a] == y[b]; }, first);
    if (first == 0) return nan;  // y constant
    DD sum{0.0, 0.0}, sq{0.0, 0.0};
    for (std::int64_t i = 0; i < n; ++i) {
        const DD e = w[s.perm[static_cast<std::size_t>(i)]];
        sum = sum + e;
        sq = sq + e * e;
    }
    const DD tot = additive ? sum * static_cast<double>(n - 1) : (sum * sum - sq) * 0.5;
    const DD num = (tot - (v + u - t)) - s.exch * 2.0;
    return clip_unit(num / sqrt_dd(tot - u) / sqrt_dd(tot - v));
}

// the order by (a, b, position)
std::vector<std::int64_t> lexorder(const double* a, const double* b, std::int64_t n) {
    std::vector<std::int64_t> p(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) p[static_cast<std::size_t>(i)] = i;
    std::sort(p.begin(), p.end(), [&](std::int64_t i, std::int64_t j) {
        if (a[i] != a[j]) return a[i] < a[j];
        if (b[i] != b[j]) return b[i] < b[j];
        return i < j;
    });
    return p;
}

DD hyperbolic(double rank) { return DD{1.0, 0.0} / (rank + 1.0); }

// mode 0: rank=True (the mean of the lexicographic ranks by (x, y) and by (y, x)); 1: rank=None
// (by (x, y)); 2: the given ranks (rank=False: the positions), as scipy's _toint64 makes them:
// their dense codes 0, 1, ... among the lane's elements. The weight of rank r is table[r] (a
// weigher's values, table None for none) or 1/(r + 1) (the hyperbolic weigher, as double-doubles).
nb::object weightedtau(
    nb::handle x, nb::handle y, int mode, nb::handle table_obj, nb::handle rank_obj, bool additive, bool omit
) {
    const Rows a(rows_arg(x)), b(rows_arg(y));
    const bool has_table = !table_obj.is_none(), has_rank = !rank_obj.is_none();
    const NDArray* table = has_table ? &require_ndarray(table_obj) : nullptr;
    Rows rk;
    if (has_rank) rk = Rows(rows_arg(rank_obj));
    Outs out(1, a.lanes);
    auto weight_of = [&](std::int64_t r) {
        return has_table ? DD{vec_at(*table, r), 0.0} : hyperbolic(static_cast<double>(r));
    };
    each_lane(a.lanes, 40 * a.n, [&](std::int64_t r) {
        std::vector<double> u, v, rr;
        bool nan = false;
        for (std::int64_t j = 0; j < a.n; ++j) {
            const double p = a.at(r, j), q = b.at(r, j);
            if (p != p || q != q) {
                nan = true;
                continue;
            }
            u.push_back(p), v.push_back(q);
            if (has_rank) rr.push_back(rk.at(r, j));
        }
        const auto n = static_cast<std::int64_t>(u.size());
        double tau = kNaN;
        if (!(nan && !omit) && n > 0) {
            std::vector<DD> w(static_cast<std::size_t>(n));
            if (mode == 2) {
                std::vector<double> levels(rr);
                std::sort(levels.begin(), levels.end());
                levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
                for (std::int64_t j = 0; j < n; ++j) {
                    const auto c = std::lower_bound(levels.begin(), levels.end(), rr[static_cast<std::size_t>(j)]) -
                                   levels.begin();
                    w[static_cast<std::size_t>(j)] = weight_of(c);
                }
                tau = to_double(wtau_core(u.data(), v.data(), w.data(), n, additive, lexorder(u.data(), v.data(), n)));
            } else {
                auto one = [&](const double* p, const double* q) {
                    std::vector<std::int64_t> perm = lexorder(p, q, n);
                    // the decreasing lexicographic rank: the largest (p, q) gets 0
                    for (std::int64_t i = 0; i < n; ++i)
                        w[static_cast<std::size_t>(perm[static_cast<std::size_t>(n - 1 - i)])] = weight_of(i);
                    return wtau_core(p, q, w.data(), n, additive, std::move(perm));
                };
                const DD t1 = one(u.data(), v.data());
                tau = mode == 1 ? to_double(t1) : to_double((t1 + one(v.data(), u.data())) * 0.5);
            }
        }
        out.p[0][r] = tau;
    });
    return wrap(std::move(out.arrays[0]));
}

// ---- Somers' D ----------------------------------------------------------------------------

// The table of x's distinct values (rows, ascending) by y's (columns): (table, rows, cols), the
// table None when rows x cols exceeds cap (nothing allocated).
nb::tuple crosstab(nb::handle xo, nb::handle yo, double cap) {
    const NDArray& xa = require_ndarray(xo);
    const NDArray& ya = require_ndarray(yo);
    const std::int64_t n = xa.shape[0];
    std::vector<double> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        x[static_cast<std::size_t>(i)] = vec_at(xa, i), y[static_cast<std::size_t>(i)] = vec_at(ya, i);
    std::vector<double> ux(x), uy(y);
    std::sort(ux.begin(), ux.end());
    ux.erase(std::unique(ux.begin(), ux.end()), ux.end());
    std::sort(uy.begin(), uy.end());
    uy.erase(std::unique(uy.begin(), uy.end()), uy.end());
    const auto R = static_cast<std::int64_t>(ux.size()), C = static_cast<std::int64_t>(uy.size());
    if (static_cast<double>(R) * static_cast<double>(C) > cap) return nb::make_tuple(nb::none(), R, C);
    const std::int64_t shape[] = {R, C};
    NDArray t = new_array(DType::Float64, shape);
    double* p = reinterpret_cast<double*>(t.data);
    std::fill(p, p + R * C, 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        const auto ix = std::lower_bound(ux.begin(), ux.end(), x[static_cast<std::size_t>(i)]) - ux.begin();
        const auto iy = std::lower_bound(uy.begin(), uy.end(), y[static_cast<std::size_t>(i)]) - uy.begin();
        p[ix * C + iy] += 1.0;
    }
    return nb::make_tuple(wrap(std::move(t)), R, C);
}

// D(Y|X) = (P - Q) / (N^2 - sum of the squared row sums) and the normal p-value of
// Z = (P - Q) / sqrt(4 S), S = sum a_ij (A_ij - D_ij)^2 - (P - Q)^2 / N (scipy's _somers_d; P and
// Q twice the concordant and discordant pairs), every sum an exact integer from the table's
// two-dimensional prefix sums. A table with one row or column, or S = 0, gives NaN.
nb::tuple somersd(nb::handle table_obj, int alt) {
    const NDArray& ta = require_ndarray(table_obj);
    if (ta.dtype != DType::Float64 || ta.ndim != 2) throw nb::type_error("htest: expects a 2-D float64 table");
    const std::int64_t R = ta.shape[0], C = ta.shape[1];
    if (R <= 1 || C <= 1) return nb::make_tuple(kNaN, kNaN);
    auto cell = [&](std::int64_t i, std::int64_t j) {
        return static_cast<std::int64_t>(
            *reinterpret_cast<const double*>(ta.data + i * ta.strides[0] + j * ta.strides[1])
        );
    };
    // cum(i, j) = sum over rows <= i - 1, columns <= j - 1 (an (R + 1) x (C + 1) table)
    std::vector<std::int64_t> cum(static_cast<std::size_t>((R + 1) * (C + 1)), 0);
    auto at = [&](std::int64_t i, std::int64_t j) -> std::int64_t& {
        return cum[static_cast<std::size_t>(i * (C + 1) + j)];
    };
    for (std::int64_t i = 0; i < R; ++i)
        for (std::int64_t j = 0; j < C; ++j) at(i + 1, j + 1) = cell(i, j) + at(i, j + 1) + at(i + 1, j) - at(i, j);
    const std::int64_t N = at(R, C);
    i128 P = 0, Q = 0, T = 0, rows2 = 0;
    for (std::int64_t i = 0; i < R; ++i) {
        const i128 ri = at(i + 1, C) - at(i, C);
        rows2 += ri * ri;
        for (std::int64_t j = 0; j < C; ++j) {
            const std::int64_t a = cell(i, j);
            if (a == 0) continue;
            const std::int64_t ul = at(i, j), lr = N - at(i + 1, C) - at(R, j + 1) + at(i + 1, j + 1);
            const std::int64_t ll = at(R, j) - at(i + 1, j), ur = at(i, C) - at(i, j + 1);
            const i128 A = ul + lr, D = ll + ur;
            P += a * A;
            Q += a * D;
            T += a * (A - D) * (A - D);
        }
    }
    const i128 pq = P - Q;
    const double d = to_double(dd_of(pq) / dd_of(static_cast<i128>(N) * N - rows2));
    // S N = T N - (P - Q)^2: exact in 128 bits up to N ~ 1e9, else double-doubles
    DD SN;
    if (N < 1'000'000'000) SN = dd_of(T * N - pq * pq);
    else SN = dd_of(T) * static_cast<double>(N) - dd_of(pq) * dd_of(pq);
    double p = kNaN;
    if (SN.hi > 0.0)
        p = normal_p(to_double(dd_of(pq) * sqrt_dd(DD{static_cast<double>(N), 0.0}) / (sqrt_dd(SN) * 2.0)), alt);
    return nb::make_tuple(d, p);
}

// ---- Chatterjee's xi ----------------------------------------------------------------------
//
// The pairs ordered by x (ties by position: scipy breaks them by its argsort, arbitrarily), r_i
// the number of y_j <= y_(i), l_i of y_j >= y_(i); xi = 1 - n sum |r_(i+1) - r_i| / (2 sum l (n - l))
// (1 - 3 sum / (n^2 - 1) when y is continuous) as one exact rational, and the p-value of the
// normal approximation P(Z >= xi / sd), sd from Chatterjee's variance (scipy's _xi_std) with its
// sums as integers: tau^2 = (A n^2 - 2 B n + C^2) / D^2.
nb::tuple chatterjee(nb::handle x, nb::handle y, bool cont, bool omit) {
    const Rows a(rows_arg(x)), b(rows_arg(y));
    Outs out(2, a.lanes);
    each_lane(a.lanes, 40 * a.n, [&](std::int64_t r) {
        std::vector<double> u, v;
        const bool nan = lane_pairs(a, b, r, u, v);
        const auto n = static_cast<std::int64_t>(u.size());
        double xi = kNaN, p = kNaN;
        if (!(nan && !omit) && n >= 2) {
            std::vector<std::pair<double, std::int64_t>> ox(static_cast<std::size_t>(n));
            for (std::int64_t j = 0; j < n; ++j) ox[static_cast<std::size_t>(j)] = {u[static_cast<std::size_t>(j)], j};
            std::sort(ox.begin(), ox.end());
            std::vector<double> ys(static_cast<std::size_t>(n)), sy;
            for (std::int64_t j = 0; j < n; ++j)
                ys[static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(ox[static_cast<std::size_t>(j)].second)];
            sy = ys;
            std::sort(sy.begin(), sy.end());
            std::vector<std::int64_t> rr(static_cast<std::size_t>(n)), ll(static_cast<std::size_t>(n));
            for (std::int64_t j = 0; j < n; ++j) {
                const double t = ys[static_cast<std::size_t>(j)];
                rr[static_cast<std::size_t>(j)] = std::upper_bound(sy.begin(), sy.end(), t) - sy.begin();
                ll[static_cast<std::size_t>(j)] = n - (std::lower_bound(sy.begin(), sy.end(), t) - sy.begin());
            }
            i128 num = 0;
            for (std::int64_t j = 0; j + 1 < n; ++j) {
                const std::int64_t d = rr[static_cast<std::size_t>(j + 1)] - rr[static_cast<std::size_t>(j)];
                num += d < 0 ? -d : d;
            }
            const i128 nn = n;
            if (cont) {
                const i128 den = nn * nn - 1;
                const DD xd = dd_of(den - 3 * num) / dd_of(den);
                xi = to_double(xd);
                p = norm_sf(to_double(xd * sqrt_dd(DD{2.5 * static_cast<double>(n), 0.0})));
            } else {
                i128 D = 0;
                for (std::int64_t j = 0; j < n; ++j)
                    D += static_cast<i128>(ll[static_cast<std::size_t>(j)]) * (n - ll[static_cast<std::size_t>(j)]);
                if (D > 0) {
                    const DD xd = dd_of(2 * D - nn * num) / dd_of(2 * D);
                    xi = to_double(xd);
                    std::vector<std::int64_t> us(rr);
                    std::sort(us.begin(), us.end());
                    Acc A, B, Cc;
                    i128 cum = 0;
                    for (std::int64_t i = 1; i <= n; ++i) {
                        const i128 ui = us[static_cast<std::size_t>(i - 1)];
                        cum += ui;
                        const i128 f = 2 * nn - 2 * i + 1;
                        A.add(dd_of(f) * dd_of(ui * ui));
                        const DD g = dd_of(cum + (nn - i) * ui);
                        B.add(g * g);
                        Cc.add(dd_of(f * ui));
                    }
                    const DD c = Cc.value();
                    const DD dn = DD{static_cast<double>(n), 0.0};
                    const DD tnum = A.value() * dn * dn - B.value() * dn * 2.0 + c * c;
                    if (tnum.hi > 0.0) p = norm_sf(to_double(xd * sqrt_dd(dn) * dd_of(D) / sqrt_dd(tnum)));
                }
            }
        }
        out.p[0][r] = xi, out.p[1][r] = p;
    });
    return out.tuple();
}

// the exact Kendall tail against independent counts (tests): (P(K <= c), P(K = c)) by the
// integer counts (n <= 33) or the double-double probabilities
nb::tuple kendall_tail(std::int64_t n, std::int64_t c, bool dd_path) {
    const kendall::Tail t = dd_path ? kendall::exact_tail_dd(n, c) : kendall::exact_tail_int(n, c);
    return nb::make_tuple(t.le, t.at);
}

}  // namespace

void bind_correlation(nb::module_& m) {
    m.def("pearson", &pearson);
    m.def("corr_matrix", &corr_matrix);
    m.def("kendall", &kendall_tau);
    m.def(
        "weightedtau", &weightedtau, nb::arg("x"), nb::arg("y"), nb::arg("mode"), nb::arg("table").none(),
        nb::arg("rank").none(), nb::arg("additive"), nb::arg("omit")
    );
    m.def("crosstab", &crosstab);
    m.def("somersd", &somersd);
    m.def("chatterjee", &chatterjee);
    m.def("kendall_tail", &kendall_tail);
}

}  // namespace sd::htest
