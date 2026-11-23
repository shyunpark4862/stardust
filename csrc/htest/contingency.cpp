// The contingency tables (milestone H6, docs/design/htest.md §3.7).
//
// The Python side (stardust.htest._contingency) checks the arguments and arranges the tables as a
// 3-D float64 array, one table (one test of a batch) per lane: (lanes, r, c). Each kernel returns
// the per-lane results as 1-D arrays (and chi2_contingency the expected counts as a 3-D one).
// Lanes are independent: the same lane gives the same bits alone or in a batch, on any thread
// count.
//
//   chi2_contingency  statistic, p-value, df, expected counts (weighted tables too: H80)
//   fisher22          the sample odds ratio and the hypergeometric p-value of a 2 x 2 table
//   fisher_rxc        the observed table's probability, the network algorithm's p-value
//                     (htest/fexact.hpp, H77), a flag for a lane past the work cap
//   barnard, boschloo the Wald statistic or Fisher's one-sided p-value, and the supremum over
//                     the nuisance parameter (htest/nuisance.hpp, H78)
//   mcnemar           statsmodels' statistic and p-value (binomial or chi^2)
//   cochrans_q        Q, its p-value and df
//   odds_ratio        the conditional MLE (htest/nchg.hpp, H79) or the sample odds ratio, and
//   odds_ratio_ci     the exact (conditional) or the Woolf (sample) interval
//
// Counts are exact: margins, products and the statistics that are rational functions of them
// are 128-bit integers or double-doubles formed once from them; the chi^2 statistics are the
// Cressie-Read sums of htest/divergence.hpp on double-double expected counts.
#include "htest/contingency.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "htest/divergence.hpp"
#include "htest/fexact.hpp"
#include "htest/lanes.hpp"
#include "htest/moments.hpp"
#include "htest/nchg.hpp"
#include "htest/nuisance.hpp"
#include "htest/rank_exact.hpp"
#include "stats/dist/discrete_basic.hpp"
#include "stats/dist/discrete_more.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;
using exact::i128;

double chi2_sf(double x, double df) { return stats::method_of<stats::ChiSquared, m::Ccdf>(x, df); }
double norm_upper_q(double p) { return stats::method_of<stats::Normal, m::Iccdf>(p, 0.0, 1.0); }

DD dd_of(i128 v) { return exact::to_dd(v); }
DD sqrt_dd(DD a) {
    if (!(a.hi > 0.0)) return DD{a.hi == 0.0 ? 0.0 : kNaN, 0.0};
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    return dd::two_sum(s, ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s));
}
double sgn(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };

// The tables: a 3-D float64 array (lanes, r, c), any strides.
struct Tabs {
    const std::byte* data = nullptr;
    std::int64_t s0 = 0, s1 = 0, s2 = 0, lanes = 0, r = 0, c = 0;
    explicit Tabs(nb::handle o) {
        const NDArray& a = require_ndarray(o);
        if (a.dtype != DType::Float64 || a.ndim != 3)
            throw nb::type_error("htest: expects a 3-D float64 array of tables");
        data = a.data, s0 = a.strides[0], s1 = a.strides[1], s2 = a.strides[2];
        lanes = a.shape[0], r = a.shape[1], c = a.shape[2];
    }
    double at(std::int64_t l, std::int64_t i, std::int64_t j) const {
        return *reinterpret_cast<const double*>(data + l * s0 + i * s1 + j * s2);
    }
    std::int64_t count(std::int64_t l, std::int64_t i, std::int64_t j) const {
        return static_cast<std::int64_t>(at(l, i, j));
    }
};

// ---- chi2_contingency ----------------------------------------------------------------------
//
// scipy's: E = R_i C_j / N (here double-double margins and one division), df = (r - 1)(c - 1),
// df = 0 gives statistic 0 and p-value 1; df = 1 with the correction moves each O towards E by
// min(1/2, |O - E|) (Yates); the statistic is power_divergence's with lambda_. A zero margin
// (an expected count of 0; scipy raises) gives NaN (H11).
nb::tuple chi2_contingency(nb::handle tables, bool correction, double lam) {
    const Tabs t(tables);
    Outs out(3, t.lanes);
    const std::int64_t shape[] = {t.lanes, t.r, t.c};
    NDArray expected = new_array(DType::Float64, shape);
    double* ex = reinterpret_cast<double*>(expected.data);
    const double df = static_cast<double>((t.r - 1) * (t.c - 1));
    each_lane(t.lanes, 8 * t.r * t.c, [&](std::int64_t l) {
        std::vector<Acc> R(static_cast<std::size_t>(t.r)), C(static_cast<std::size_t>(t.c));
        Acc Nacc;
        for (std::int64_t i = 0; i < t.r; ++i)
            for (std::int64_t j = 0; j < t.c; ++j) {
                const double o = t.at(l, i, j);
                R[static_cast<std::size_t>(i)].add(o), C[static_cast<std::size_t>(j)].add(o), Nacc.add(o);
            }
        const DD N = Nacc.value();
        const bool empty = !(N.hi > 0.0);
        bool zero = empty;
        std::vector<DD> E(static_cast<std::size_t>(t.r * t.c));
        double* row = ex + l * t.r * t.c;
        for (std::int64_t i = 0; i < t.r; ++i)
            for (std::int64_t j = 0; j < t.c; ++j) {
                const DD e =
                    empty ? DD{0.0, 0.0}
                          : (R[static_cast<std::size_t>(i)].value() * C[static_cast<std::size_t>(j)].value()) / N;
                E[static_cast<std::size_t>(i * t.c + j)] = e;
                row[i * t.c + j] = to_double(e);
                zero = zero || !(e.hi > 0.0);
            }
        if (zero) {  // (the expected counts of a zero total: 0 / 0)
            if (empty)
                for (std::int64_t q = 0; q < t.r * t.c; ++q) row[q] = kNaN;
            out.p[0][l] = kNaN, out.p[1][l] = kNaN, out.p[2][l] = df;
            return;
        }
        double stat = 0.0, p = 1.0;
        if (df > 0.0) {
            div::Divergence dv(lam);
            for (std::int64_t i = 0; i < t.r; ++i)
                for (std::int64_t j = 0; j < t.c; ++j) {
                    const DD e = E[static_cast<std::size_t>(i * t.c + j)];
                    DD d = DD{t.at(l, i, j), 0.0} - e;
                    if (correction && df == 1.0) {
                        const DD ad = abs_dd(d);
                        d = (ad.hi < 0.5 || (ad.hi == 0.5 && ad.lo <= 0.0)) ? DD{0.0, 0.0}
                                                                            : d - DD{0.5 * sgn(d.hi), 0.0};
                    }
                    dv.add(e, d);
                }
            stat = dv.value();
            p = stat == stat ? chi2_sf(stat, df) : kNaN;
        }
        out.p[0][l] = stat, out.p[1][l] = p, out.p[2][l] = df;
    });
    nb::tuple r = out.tuple();
    return nb::make_tuple(r[0], r[1], r[2], wrap(std::move(expected)));
}

// ---- Fisher's exact test, 2 x 2 ------------------------------------------------------------
//
// scipy's: X the top-left cell, hypergeometric with total N, the first row's sum as successes and
// the first column's as draws. 'less' P(X <= a), 'greater' P(X >= a), both sd.stats'
// Hypergeometric tails. 'two-sided': p = 1 when the observed table is as probable as the mode to
// a relative 1e-14; else the observed tail and the other side's tables less probable than
// (1 + 1e-14) times the observed one (at most as probable below the mode): scipy's binary
// search. Within kWalk of the mode the comparisons are on the exact pmf ratios to the observed
// table (double-doubles kept with a power of two), walked one table at a time; farther (H150) a
// binary search on sd.stats' double-double log pmf (its error, about 1e-30, is far below the
// 1e-14 of the rule). Where a hypergeometric tail is NaN although the pmf is not (sd.stats gives
// a long tail by its smooth sum, and NaN where that does not apply) and the tables between the
// two boundaries are at most kMiddle, the p-value is 1 minus their probability when that is at
// most 1/2, else NaN. Past a total of 2^52 sd.stats has no pmf and every p-value is NaN. The
// statistic is the sample odds ratio ad / bc (inf for bc = 0); a zero margin gives NaN and p = 1.
// The counts are below 2^53 (checked by the caller), the mode a 128-bit product.
constexpr std::int64_t kWalk = 1 << 16;
constexpr std::int64_t kMiddle = 1 << 21;

struct Ratio {  // a positive value m 2^e
    DD m{1.0, 0.0};
    int e = 0;
    void mul(DD f) {
        m = m * f;
        const int k = std::ilogb(m.hi);
        if (k > 600 || k < -600) {
            m = dd::ldexp(m, -k);
            e += k;
        }
    }
    // compared with a double-double g in [1, 2): -1, 0, 1 (by the exponent alone when the value
    // is at least 4 or below 1/2: m 2^e itself may be past the doubles)
    int cmp(DD g) const {
        if (!(m.hi > 0.0)) return -1;
        const int k = e + std::ilogb(m.hi);
        if (k >= 2) return 1;
        if (k < -1) return -1;
        const DD v = dd::ldexp(m, e);
        const DD d = v - g;
        return d.hi > 0.0 ? 1 : (d.hi < 0.0 ? -1 : (d.lo > 0.0 ? 1 : (d.lo < 0.0 ? -1 : 0)));
    }
};

int dd_cmp(DD a, DD b) {
    const DD d = a - b;
    return d.hi > 0.0 ? 1 : (d.hi < 0.0 ? -1 : (d.lo > 0.0 ? 1 : (d.lo < 0.0 ? -1 : 0)));
}

struct Pair {
    double stat, p;
};

// The margins of a 2 x 2 table and the hypergeometric of its top-left cell.
struct Hyper22 {
    std::int64_t r0, r1, c0, c1, N, lo, hi, mode;
    Hyper22(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d)
        : r0(a + b), r1(c + d), c0(a + c), c1(b + d), N(r0 + r1) {
        lo = std::max<std::int64_t>(0, c0 - r1), hi = std::min(r0, c0);
        const i128 m = N + 2 > 0 ? (static_cast<i128>(c0) + 1) * (static_cast<i128>(r0) + 1) / (N + 2) : 0;
        mode = static_cast<std::int64_t>(std::clamp<i128>(m, lo, hi));
    }
    // the standard deviation of the cell (a rough size of the sums)
    double sd() const {
        if (N < 2) return 0.0;
        const double Nd = static_cast<double>(N);
        return std::sqrt(
            static_cast<double>(r0) / Nd * (static_cast<double>(r1) / Nd) * static_cast<double>(c0) *
            (static_cast<double>(c1) / (Nd - 1.0))
        );
    }
};

// the work of one lane in elements (each_lane's cost; the pool and the GIL go by it)
std::int64_t fisher22_cost(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) {
    const Hyper22 h(a, b, c, d);
    const std::int64_t walk = std::min<std::int64_t>(std::abs(a - h.mode), kWalk);
    const double sums = std::min(40.0 * h.sd(), static_cast<double>(1 << 21));
    return 64 + 8 * walk + 4 * static_cast<std::int64_t>(sums);
}

Pair fisher22_lane(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d, int alt) {
    const Hyper22 h(a, b, c, d);
    const std::int64_t r0 = h.r0, c0 = h.c0, N = h.N, lo = h.lo, hi = h.hi, mode = h.mode;
    if (r0 == 0 || h.r1 == 0 || c0 == 0 || h.c1 == 0) return {kNaN, 1.0};
    const double stat =
        (b > 0 && c > 0) ? to_double(dd_of(static_cast<i128>(a) * d) / dd_of(static_cast<i128>(b) * c)) : kInf;
    const double Nd = static_cast<double>(N), K = static_cast<double>(r0), n = static_cast<double>(c0);
    using H = stats::Hypergeometric;
    if (alt == kLess) return {stat, H::cdf(static_cast<double>(a), Nd, K, n)};
    if (alt == kGreater) return {stat, H::ccdf(static_cast<double>(a - 1), Nd, K, n)};
    // pmf(x + 1) / pmf(x) = (r0 - x)(c0 - x) / ((x + 1)(N - r0 - c0 + x + 1)), and its inverse
    auto up = [&](std::int64_t x) {
        return dd::two_prod(static_cast<double>(r0 - x), static_cast<double>(c0 - x)) /
               dd::two_prod(static_cast<double>(x + 1), static_cast<double>(N - r0 - c0 + x + 1));
    };
    auto down = [&](std::int64_t x) {  // pmf(x - 1) / pmf(x)
        return dd::two_prod(static_cast<double>(x), static_cast<double>(N - r0 - c0 + x)) /
               dd::two_prod(static_cast<double>(r0 - x + 1), static_cast<double>(c0 - x + 1));
    };
    auto lpmf = [&](std::int64_t x) { return H::log_pmf(static_cast<double>(x), Nd, K, n); };
    const double eps = 1e-14, gamma = 1.0 + eps;
    const DD lim = DD{1.0, 0.0} / dd::two_sum(1.0, -eps);  // rho <= 1 / (1 - eps): p = 1
    const DD g{gamma, 0.0};
    const bool near = std::abs(a - mode) <= kWalk;
    // The other side's boundary by the log pmf: a < mode the smallest x in [from, hi] with
    // pmf(x) < gamma pmf(a) (hi + 1 for none), else the largest x in [lo, from] with pmf(x) <=
    // gamma pmf(a) (lo - 1 for none); the log pmf falls away from the mode.
    auto search = [&](std::int64_t from) {
        const DD thr = lpmf(a) + dd::log(g);
        if (a < mode) {
            std::int64_t L = from, R = hi + 1;  // L: not below (from is not), R: below
            while (R - L > 1) {
                const std::int64_t mid = L + (R - L) / 2;
                if (dd_cmp(lpmf(mid), thr) < 0) R = mid;
                else L = mid;
            }
            return R;
        }
        std::int64_t L = lo - 1, R = from;  // L: at most, R: above (from is)
        while (R - L > 1) {
            const std::int64_t mid = L + (R - L) / 2;
            if (dd_cmp(lpmf(mid), thr) <= 0) L = mid;
            else R = mid;
        }
        return L;
    };
    std::int64_t x;
    if (near) {
        // rho = pmf(mode) / pmf(a)
        Ratio rho;
        if (a < mode)
            for (std::int64_t y = a; y < mode; ++y) rho.mul(up(y));
        else
            for (std::int64_t y = a; y > mode; --y) rho.mul(down(y));
        if (rho.cmp(lim) <= 0) return {stat, 1.0};
        Ratio v = rho;
        x = mode;
        std::int64_t steps = 0;
        if (a < mode) {
            while (x <= hi && v.cmp(g) >= 0) {
                if (x == hi) {
                    ++x;
                    break;
                }
                if (++steps > 4 * kWalk) {  // a long side: the rest by the search
                    x = search(x);
                    break;
                }
                v.mul(up(x));
                ++x;
            }
        } else {
            while (x >= lo && v.cmp(g) > 0) {
                if (x == lo) {
                    --x;
                    break;
                }
                if (++steps > 4 * kWalk) {
                    x = search(x);
                    break;
                }
                v.mul(down(x));
                --x;
            }
        }
    } else {
        if (dd_cmp(lpmf(mode) - lpmf(a), dd::log(lim)) <= 0) return {stat, 1.0};
        x = search(mode);
    }
    // a < mode: P(X <= a) + P(X >= x); else P(X >= a) + P(X <= x)
    double p;
    if (a < mode)
        p = H::cdf(static_cast<double>(a), Nd, K, n) + (x > hi ? 0.0 : H::ccdf(static_cast<double>(x - 1), Nd, K, n));
    else p = H::ccdf(static_cast<double>(a - 1), Nd, K, n) + (x < lo ? 0.0 : H::cdf(static_cast<double>(x), Nd, K, n));
    if (p != p) {
        // 1 - the probability of the tables strictly between a and x (the mode among them)
        const std::int64_t b0 = a < mode ? a + 1 : x + 1, b1 = a < mode ? x - 1 : a - 1;
        if (b1 - b0 + 1 > kMiddle) return {stat, kNaN};
        Acc s;
        s.add(1.0);
        DD t{1.0, 0.0};
        for (std::int64_t y = mode; y < b1; ++y) {
            t = t * up(y);
            s.add(t);
        }
        t = DD{1.0, 0.0};
        for (std::int64_t y = mode; y > b0; --y) {
            t = t * down(y);
            s.add(t);
        }
        const DD mid = s.value() * stats::num::exp_full(lpmf(mode));
        p = mid.hi <= 0.5 ? to_double(DD{1.0, 0.0} - mid) : kNaN;
    }
    return {stat, std::min(p, 1.0)};
}

nb::tuple fisher22(nb::handle tables, int alt) {
    const Tabs t(tables);
    Outs out(2, t.lanes);
    std::int64_t cost = 64;
    for (std::int64_t l = 0; l < t.lanes; ++l)
        cost = std::max(cost, fisher22_cost(t.count(l, 0, 0), t.count(l, 0, 1), t.count(l, 1, 0), t.count(l, 1, 1)));
    each_lane(t.lanes, cost, [&](std::int64_t l) {
        const Pair r = fisher22_lane(t.count(l, 0, 0), t.count(l, 0, 1), t.count(l, 1, 0), t.count(l, 1, 1), alt);
        out.p[0][l] = r.stat, out.p[1][l] = r.p;
    });
    return out.tuple();
}

// ---- Fisher's exact test, r x c (htest/fexact.hpp) -----------------------------------------

nb::tuple fisher_rxc(nb::handle tables, double cap, double hold) {
    const Tabs t(tables);
    Outs out(3, t.lanes);
    each_lane(t.lanes, 1 << 20, [&](std::int64_t l) {
        std::vector<std::int64_t> cells(static_cast<std::size_t>(t.r * t.c));
        for (std::int64_t i = 0; i < t.r; ++i)
            for (std::int64_t j = 0; j < t.c; ++j) cells[static_cast<std::size_t>(i * t.c + j)] = t.count(l, i, j);
        const fexact::Result r = fexact::test(t.r, t.c, cells, cap, hold);
        out.p[0][l] = r.statistic, out.p[1][l] = r.pvalue, out.p[2][l] = r.capped ? 1.0 : 0.0;
    });
    return out.tuple();
}

// ---- Barnard's and Boschloo's exact tests (htest/nuisance.hpp) -----------------------------
//
// scipy's orientation: the columns are the two binomial samples, x1 = t[0, 0] of n1 = the first
// column's sum, x2 = t[0, 1] of n2. A zero column gives NaN and (Barnard) p = 1 or (Boschloo)
// NaN, scipy's. Past (n1 + 1)(n2 + 1) = 1e7 tables the p-value is NaN (the work cap).
constexpr double kTablesCap = 1e7;
constexpr double kSupCap = 4e8;  // the supremum's evaluations times N + 1

// Barnard's Wald statistic as the exact T^2 = num / den (den 0 with num > 0: infinite) and its
// sign: pooled T^2 = D^2 N / (n1 n2 s (N - s)), unpooled D^2 n1 n2 / V, D = x1 n2 - x2 n1,
// V = x1 (n1 - x1) n2^3 + x2 (n2 - x2) n1^3. The comparisons drop the common factors (N / (n1 n2),
// n1 n2): num' = D^2, den' = s (N - s) or V, every product within 128 bits for n1, n2 <= 1e4.
struct Wald {
    i128 num, den;  // T^2 up to the common factor
    int sign;
};
Wald wald(std::int64_t x1, std::int64_t x2, std::int64_t n1, std::int64_t n2, bool pooled) {
    const i128 D = static_cast<i128>(x1) * n2 - static_cast<i128>(x2) * n1;
    if (D == 0) return {0, 1, 0};
    i128 den;
    if (pooled) {
        const i128 s = x1 + x2, N = n1 + n2;
        den = s * (N - s);
    } else {
        const i128 a = static_cast<i128>(n1) * n1 * n1, b = static_cast<i128>(n2) * n2 * n2;
        den = static_cast<i128>(x1) * (n1 - x1) * b + static_cast<i128>(x2) * (n2 - x2) * a;
    }
    return {D * D, den, D > 0 ? 1 : -1};
}
// |T_a| vs |T_b|: -1, 0, 1
int cmp_abs(const Wald& a, const Wald& b) {
    if (a.num == 0 || b.num == 0) return a.num == 0 ? (b.num == 0 ? 0 : -1) : 1;
    if (a.den == 0 || b.den == 0) return a.den == 0 ? (b.den == 0 ? 0 : 1) : -1;
    const i128 l = a.num * b.den, r = b.num * a.den;
    return l > r ? 1 : (l < r ? -1 : 0);
}
int cmp_signed(const Wald& a, const Wald& b) {
    if (a.sign != b.sign) return a.sign < b.sign ? -1 : 1;
    const int c = cmp_abs(a, b);
    return a.sign >= 0 ? c : -c;
}
double wald_value(const Wald& w, std::int64_t n1, std::int64_t n2, bool pooled) {
    if (w.num == 0) return 0.0;
    if (w.den == 0) return w.sign * kInf;
    // T^2 = num' f / den', f = N / (n1 n2) pooled, n1 n2 unpooled
    const i128 N = n1 + n2, p = static_cast<i128>(n1) * n2;
    const DD t2 =
        pooled ? (dd_of(w.num) * dd_of(N)) / (dd_of(w.den) * dd_of(p)) : (dd_of(w.num) * dd_of(p)) / dd_of(w.den);
    return w.sign * to_double(sqrt_dd(t2));
}

double supremum(const std::vector<DD>& lbeta, std::int64_t n) {
    nuisance::Sup sup(lbeta);
    const DD l = sup.run(n, kSupCap);
    if (l.hi != l.hi) return kNaN;
    return std::min(1.0, stats::num::exp_dd(l));
}

Pair barnard_lane(
    std::int64_t t00, std::int64_t t01, std::int64_t t10, std::int64_t t11, int alt, bool pooled, std::int64_t n
) {
    const std::int64_t n1 = t00 + t10, n2 = t01 + t11;
    if (n1 == 0 || n2 == 0) return {kNaN, 1.0};
    const Wald obs = wald(t00, t01, n1, n2, pooled);
    const double stat = wald_value(obs, n1, n2, pooled);
    if (static_cast<double>(n1 + 1) * static_cast<double>(n2 + 1) > kTablesCap) return {stat, kNaN};
    const std::int64_t N = n1 + n2;
    std::vector<DD> lbeta(static_cast<std::size_t>(N + 1));
    for (std::int64_t s = 0; s <= N; ++s) {
        const nuisance::Total tot(n1, n2, s);
        lbeta[static_cast<std::size_t>(s)] = tot.log_sum([&](std::int64_t x1) {
            const Wald w = wald(x1, s - x1, n1, n2, pooled);
            if (alt == kTwoSided) return cmp_abs(w, obs) >= 0;
            if (alt == kLess) return cmp_signed(w, obs) <= 0;
            return cmp_signed(w, obs) >= 0;
        });
    }
    return {stat, supremum(lbeta, n)};
}

// Boschloo: the one-sided Fisher p-value of every table (P(X1 <= x1 | s) for 'less', P(X1 >= x1 | s)
// for 'greater', each as a double-double log from the hypergeometric terms of its total, the
// partial sums relative to the current term), the region the tables at most (1 + 1e-13) times as
// significant as the observed one (scipy's tolerance); the statistic is the observed one's, by
// sd.stats' Hypergeometric (scipy's parametrization). Two-sided: the smaller one-sided p-value
// doubled (ties within 1e-13 take 'greater', as scipy's strict '<').
Pair boschloo_side(std::int64_t t00, std::int64_t t01, std::int64_t n1, std::int64_t n2, bool less, std::int64_t n) {
    const std::int64_t N = n1 + n2, so = t00 + t01;
    using H = stats::Hypergeometric;
    const double stat =
        less
            ? H::cdf(static_cast<double>(t00), static_cast<double>(N), static_cast<double>(so), static_cast<double>(n1))
            : H::cdf(
                  static_cast<double>(t01), static_cast<double>(N), static_cast<double>(so), static_cast<double>(n2)
              );
    if (static_cast<double>(n1 + 1) * static_cast<double>(n2 + 1) > kTablesCap) return {stat, kNaN};
    // the log one-sided p-values of the tables of total s into v
    auto side = [&](const nuisance::Total& T, std::int64_t s, std::vector<DD>& v) {
        v.resize(static_cast<std::size_t>(T.hi - T.lo + 1));
        DD u{1.0, 0.0};
        if (less) {
            for (std::int64_t x = T.lo; x <= T.hi; ++x) {
                if (x > T.lo)  // u_x = 1 + u_(x-1) P(x - 1) / P(x)
                    u = DD{1.0, 0.0} +
                        u * (dd::two_prod(static_cast<double>(x), static_cast<double>(n2 - s + x)) /
                             dd::two_prod(static_cast<double>(n1 - x + 1), static_cast<double>(s - x + 1)));
                const auto i = static_cast<std::size_t>(x - T.lo);
                v[i] = T.lmode + dd::log(T.r[i] * u) + stats::num::kLn2 * static_cast<double>(T.e[i]);
            }
        } else {
            for (std::int64_t x = T.hi; x >= T.lo; --x) {
                if (x < T.hi)  // u_x = 1 + u_(x+1) P(x + 1) / P(x)
                    u = DD{1.0, 0.0} +
                        u * (dd::two_prod(static_cast<double>(n1 - x), static_cast<double>(s - x)) /
                             dd::two_prod(static_cast<double>(x + 1), static_cast<double>(n2 - s + x + 1)));
                const auto i = static_cast<std::size_t>(x - T.lo);
                v[i] = T.lmode + dd::log(T.r[i] * u) + stats::num::kLn2 * static_cast<double>(T.e[i]);
            }
        }
    };
    std::vector<DD> v;
    const nuisance::Total To(n1, n2, so);
    side(To, so, v);
    const DD thr = v[static_cast<std::size_t>(t00 - To.lo)] + dd::log(DD{1.0 + 1e-13, 0.0});
    std::vector<DD> lbeta(static_cast<std::size_t>(N + 1));
    for (std::int64_t s = 0; s <= N; ++s) {
        const nuisance::Total T(n1, n2, s);
        side(T, s, v);
        lbeta[static_cast<std::size_t>(s)] = T.log_sum([&](std::int64_t x1) {
            const DD d = v[static_cast<std::size_t>(x1 - T.lo)] - thr;
            return d.hi < 0.0 || (d.hi == 0.0 && d.lo <= 0.0);
        });
    }
    return {stat, supremum(lbeta, n)};
}

Pair boschloo_lane(std::int64_t t00, std::int64_t t01, std::int64_t t10, std::int64_t t11, int alt, std::int64_t n) {
    const std::int64_t n1 = t00 + t10, n2 = t01 + t11;
    if (n1 == 0 || n2 == 0) return {kNaN, kNaN};
    if (alt == kLess) return boschloo_side(t00, t01, n1, n2, true, n);
    if (alt == kGreater) return boschloo_side(t00, t01, n1, n2, false, n);
    const Pair l = boschloo_side(t00, t01, n1, n2, true, n), g = boschloo_side(t00, t01, n1, n2, false, n);
    if (l.p != l.p || g.p != g.p) return {l.p < g.p ? l.stat : g.stat, kNaN};
    const Pair& r = l.p < g.p * (1.0 - 1e-13) ? l : g;
    return {r.stat, std::min(1.0, 2.0 * r.p)};
}

nb::tuple barnard(nb::handle tables, int alt, bool pooled, std::int64_t n) {
    const Tabs t(tables);
    Outs out(2, t.lanes);
    each_lane(t.lanes, 1 << 20, [&](std::int64_t l) {
        const Pair r =
            barnard_lane(t.count(l, 0, 0), t.count(l, 0, 1), t.count(l, 1, 0), t.count(l, 1, 1), alt, pooled, n);
        out.p[0][l] = r.stat, out.p[1][l] = r.p;
    });
    return out.tuple();
}

nb::tuple boschloo(nb::handle tables, int alt, std::int64_t n) {
    const Tabs t(tables);
    Outs out(2, t.lanes);
    each_lane(t.lanes, 1 << 20, [&](std::int64_t l) {
        const Pair r = boschloo_lane(t.count(l, 0, 0), t.count(l, 0, 1), t.count(l, 1, 0), t.count(l, 1, 1), alt, n);
        out.p[0][l] = r.stat, out.p[1][l] = r.p;
    });
    return out.tuple();
}

// ---- McNemar and Cochran's Q (statsmodels) -------------------------------------------------
//
// mcnemar: b = t[0, 1], c = t[1, 0]. exact: statistic min(b, c), p = min(1, 2 P(B <= min(b, c)))
// for B ~ Binomial(b + c, 1/2) (sd.stats' tail). Otherwise (|b - c| - correction)^2 / (b + c),
// exact, and its chi^2(1) tail; b + c = 0 gives NaN (statsmodels 0/0 or inf and 0; H11).
nb::tuple mcnemar(nb::handle tables, bool exact, bool correction) {
    const Tabs t(tables);
    Outs out(2, t.lanes);
    each_lane(t.lanes, 64, [&](std::int64_t l) {
        const std::int64_t b = t.count(l, 0, 1), c = t.count(l, 1, 0), n = b + c;
        double stat, p;
        if (exact) {
            const std::int64_t k = std::min(b, c);
            stat = static_cast<double>(k);
            p = std::min(1.0, 2.0 * stats::Binomial::cdf(static_cast<double>(k), static_cast<double>(n), 0.5));
        } else if (n == 0) {
            stat = p = kNaN;
        } else {
            const i128 d = static_cast<i128>(b > c ? b - c : c - b) - (correction ? 1 : 0);
            stat = to_double(dd_of(d * d) / dd_of(static_cast<i128>(n)));
            p = chi2_sf(stat, 1.0);
        }
        out.p[0][l] = stat, out.p[1][l] = p;
    });
    return out.tuple();
}

// cochrans_q: x (N, k) of 0 and 1, R_i the row sums, C_j the column sums, T their total:
// Q = (k - 1)(k sum C_j^2 - T^2) / (k T - sum R_i^2) exact (statsmodels' SAS form), the chi^2
// (k - 1) tail; a zero denominator (every row all 0 or all 1) gives NaN.
nb::tuple cochrans_q(nb::handle tables) {
    const Tabs t(tables);
    Outs out(3, t.lanes);
    const std::int64_t N = t.r, k = t.c;
    each_lane(t.lanes, 4 * N * k, [&](std::int64_t l) {
        std::vector<i128> C(static_cast<std::size_t>(k), 0);
        i128 T = 0, R2 = 0;
        for (std::int64_t i = 0; i < N; ++i) {
            i128 ri = 0;
            for (std::int64_t j = 0; j < k; ++j) {
                const std::int64_t v = t.count(l, i, j);
                ri += v;
                C[static_cast<std::size_t>(j)] += v;
            }
            T += ri, R2 += ri * ri;
        }
        i128 C2 = 0;
        for (const i128 v : C) C2 += v * v;
        const i128 num = static_cast<i128>(k - 1) * (static_cast<i128>(k) * C2 - T * T),
                   den = static_cast<i128>(k) * T - R2;
        double q = kNaN, p = kNaN;
        if (den > 0) {
            q = to_double(dd_of(num) / dd_of(den));
            p = chi2_sf(q, static_cast<double>(k - 1));
        }
        out.p[0][l] = q, out.p[1][l] = p, out.p[2][l] = static_cast<double>(k - 1);
    });
    return out.tuple();
}

// ---- the odds ratio (scipy.stats.contingency) ----------------------------------------------
//
// kind 0 'conditional': the conditional MLE of Fisher's noncentral hypergeometric (htest/
// nchg.hpp; 0 at the support's low end, inf at its high end), its exact interval (the omegas
// with P(X >= x) = alpha and P(X <= x) = alpha; 0 or inf at the ends of the support). kind 1
// 'sample': ad / bc (inf for bc = 0 and ad > 0) and Woolf's interval OR exp(-+ z se), se^2 =
// 1/a + 1/b + 1/c + 1/d (woolf_variance; inf for a zero cell, so the ends follow IEEE as scipy's: 0
// and NaN, NaN and inf). A zero margin: statistic NaN, interval (0, inf).
double sample_or(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) {
    if (b > 0 && c > 0) return to_double(dd_of(static_cast<i128>(a) * d) / dd_of(static_cast<i128>(b) * c));
    if (a == 0 || d == 0) return kNaN;
    return kInf;
}
bool zero_margin(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) {
    return a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0;
}
// 1/a + 1/b + 1/c + 1/d for positive counts as a double-double: one quotient of exact 128-bit
// integers while the product abcd fits (every count below 2^31: abcd < 2^124 and the numerator
// < 2^95), else the four quotients added (each within 2^-104 of itself, the terms positive; the
// counts are below 2^53, exact doubles).
DD woolf_variance(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) {
    if (std::max({a, b, c, d}) < (std::int64_t{1} << 31)) {
        const i128 A = a, B = b, Cc = c, Dd = d;
        return dd_of(B * Cc * Dd + A * Cc * Dd + A * B * Dd + A * B * Cc) / dd_of(A * B * Cc * Dd);
    }
    const DD one{1.0, 0.0};
    auto inv = [&](std::int64_t v) { return one / DD{static_cast<double>(v), 0.0}; };
    return (inv(a) + inv(d)) + (inv(b) + inv(c));
}

// the work of one conditional lane in elements: the support or the window, times the solver's
// evaluations
std::int64_t odds_ratio_cost(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d) {
    const Hyper22 h(a, b, c, d);
    const double pts = std::min(static_cast<double>(h.hi - h.lo + 1), 40.0 * h.sd() + 64.0);
    return 64 * (1 + static_cast<std::int64_t>(std::min(pts, static_cast<double>(nchg::Dist::kMaxWindow))));
}
std::int64_t odds_ratio_cost(const Tabs& t, int kind) {
    std::int64_t cost = 1 << 12;
    if (kind == 0)
        for (std::int64_t l = 0; l < t.lanes; ++l)
            cost =
                std::max(cost, odds_ratio_cost(t.count(l, 0, 0), t.count(l, 0, 1), t.count(l, 1, 0), t.count(l, 1, 1)));
    return cost;
}

// (statistic, too big): the second 1 where the conditional estimate's window passed its cap
nb::tuple odds_ratio(nb::handle tables, int kind) {
    const Tabs t(tables);
    Outs out(2, t.lanes);
    each_lane(t.lanes, odds_ratio_cost(t, kind), [&](std::int64_t l) {
        const std::int64_t a = t.count(l, 0, 0), b = t.count(l, 0, 1), c = t.count(l, 1, 0), d = t.count(l, 1, 1);
        double v, big = 0.0;
        if (zero_margin(a, b, c, d)) v = kNaN;
        else if (kind == 1) v = sample_or(a, b, c, d);
        else {
            nchg::Dist D(a + b + c + d, a + b, a + c);
            v = D.mle(a);
            if (D.too_big()) v = kNaN, big = 1.0;
        }
        out.p[0][l] = v, out.p[1][l] = big;
    });
    return out.tuple();
}

// (lower, upper, too big)
nb::tuple odds_ratio_ci(nb::handle tables, int kind, double alpha, int alt) {
    const Tabs t(tables);
    Outs out(3, t.lanes);
    each_lane(t.lanes, 2 * odds_ratio_cost(t, kind), [&](std::int64_t l) {
        const std::int64_t a = t.count(l, 0, 0), b = t.count(l, 0, 1), c = t.count(l, 1, 0), d = t.count(l, 1, 1);
        double lo = 0.0, hi = kInf, big = 0.0;
        const double tail = alt == kTwoSided ? 0.5 * alpha : alpha;
        if (!zero_margin(a, b, c, d)) {
            if (kind == 0) {
                nchg::Dist D(a + b + c + d, a + b, a + c);
                if (alt != kLess) lo = D.end(a, tail, true);
                if (alt != kGreater) hi = D.end(a, tail, false);
                if (D.too_big()) lo = kNaN, hi = kNaN, big = 1.0;
            } else {
                const double r = sample_or(a, b, c, d);
                double se = kInf;
                if (a > 0 && b > 0 && c > 0 && d > 0) se = to_double(sqrt_dd(woolf_variance(a, b, c, d)));
                const double z = norm_upper_q(tail);
                if (alt != kLess) lo = r * std::exp(-z * se);
                if (alt != kGreater) hi = r * std::exp(z * se);
            }
        }
        out.p[0][l] = lo, out.p[1][l] = hi, out.p[2][l] = big;
    });
    return out.tuple();
}

}  // namespace

void bind_contingency(nb::module_& m) {
    m.def("chi2_contingency", &chi2_contingency);
    m.def("fisher22", &fisher22);
    m.def("fisher_rxc", &fisher_rxc);
    m.def("barnard", &barnard);
    m.def("boschloo", &boschloo);
    m.def("mcnemar", &mcnemar);
    m.def("cochrans_q", &cochrans_q);
    m.def("odds_ratio", &odds_ratio);
    m.def("odds_ratio_ci", &odds_ratio_ci);
    m.attr("ODDS_RATIO_WINDOW") = static_cast<double>(nchg::Dist::kMaxWindow);
}

}  // namespace sd::htest
