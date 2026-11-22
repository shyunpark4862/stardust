// The proportions and the binomial (milestone H7, docs/design/htest.md §3.8).
//
// The Python side (stardust.htest._proportion) checks the arguments and broadcasts the counts;
// each kernel takes 1-D float64 arrays of the lanes (one test or one interval of a batch) and
// returns 1-D arrays, or, for the scalar tests (binomtest, poisson_means_test), Python floats.
// Lanes are independent: the same lane gives the same bits alone or in a batch, on any thread
// count.
//
//   binomtest          scipy's test: the binomial tails, the two-sided rule (the far side's
//                      terms at most (1 + 1e-7) pmf(k)) decided on double-double log pmfs
//   binom_ci           the intervals of one proportion: normal, agresti_coull, exact
//                      (Clopper-Pearson), wilson, wilsoncc, jeffreys, binom_test (the
//                      two-sided binomial test inverted)
//   quantile_test      the counts of x <= q and x < q and their binomial p-value
//   quantile_ci        the order statistics of the interval
//   poisson_means      the E-test of two Poisson means (the double sum as one sum of Poisson
//                      tails)
//   prop_ztest         statsmodels' z test of one or two proportions
//   prop_chisquare     the chi^2 test of proportions against a null proportion
//   prop2_test         the tests of two independent proportions (diff, ratio, odds ratio)
//   prop2_ci           their intervals
//
// Proportions, differences and the variances are formed from the integer counts as
// double-doubles (one rounding at the end), so a difference near the null value keeps its
// digits; the score statistics' constrained estimates are polished to double-double by Newton's
// method on their polynomial.
#include "htest/proportion.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "htest/dist_array.hpp"
#include "htest/lanes.hpp"
#include "htest/moments.hpp"
#include "htest/rank_exact.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/discrete_basic.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/dist/normal.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace nb = nanobind;

namespace sd::htest {

namespace {

namespace m = stats::method;
namespace num = stats::num;
using exact::i128;
using stats::Binomial;

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };
constexpr double kRerr = 1.0 + 1e-7;  // scipy's and R's relative tolerance of the two-sided rule
constexpr double kPi = 3.141592653589793;

double nsf(double z) { return stats::method_of<stats::Normal, m::Ccdf>(z, 0.0, 1.0); }
double zq(double a) { return stats::method_of<stats::Normal, m::Iccdf>(a, 0.0, 1.0); }  // the upper a-quantile
double chi2_sf(double x, double df) { return stats::method_of<stats::ChiSquared, m::Ccdf>(x, df); }
double beta_icdf(double p, double a, double b) { return stats::method_of<stats::Beta, m::Icdf>(p, a, b); }
double beta_iccdf(double p, double a, double b) { return stats::method_of<stats::Beta, m::Iccdf>(p, a, b); }

// The normal p-value of z (statsmodels' _zstat_generic2, scipy's): 'greater' the upper tail,
// 'less' the lower, two-sided twice the upper tail of |z|.
double z_pvalue(double z, int alt) {
    if (z != z) return kNaN;
    if (alt == kGreater) return nsf(z);
    if (alt == kLess) return nsf(-z);
    return std::min(1.0, 2.0 * nsf(std::fabs(z)));
}

DD D(double v) { return DD{v, 0.0}; }
DD quot(double a, double b) { return num::div(DD{a, 0.0}, b); }
DD sqrt_dd(DD a) {
    if (!(a.hi > 0.0)) return DD{a.hi == 0.0 ? 0.0 : (a.hi == a.hi ? kNaN : a.hi), 0.0};
    if (std::isinf(a.hi)) return DD{kInf, 0.0};
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    return dd::two_sum(s, ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s));
}
bool dd_le(DD a, DD b) {
    const DD d = a - b;
    return d.hi < 0.0 || (d.hi == 0.0 && d.lo <= 0.0);
}
// a / sqrt(v): the statistic of a difference and its variance; NaN for a zero (or NaN) variance
// (H24)
double over_sqrt(DD a, DD v) {
    if (!(v.hi > 0.0)) return kNaN;
    return to_double(a / sqrt_dd(v));
}
// a / b with IEEE's ends where b is 0 (a zero count: inf, or NaN for 0 / 0) or a quotient
// overflows
DD sdiv(DD a, DD b) {
    if (b.hi == 0.0) return D(a.hi == 0.0 ? kNaN : std::copysign(kInf, a.hi) * std::copysign(1.0, b.hi));
    const DD q = a / b;
    return std::isfinite(q.hi) ? q : D(a.hi / b.hi);
}
// a + b with IEEE's sum where either is not finite (a variance with a zero count is inf)
DD sadd(DD a, DD b) { return std::isfinite(a.hi) && std::isfinite(b.hi) ? a + b : D(a.hi + b.hi); }
double clip01(double v) { return v != v ? v : std::min(1.0, std::max(0.0, v)); }

// ---- bisection on the doubles' order ------------------------------------------------------

std::int64_t okey(double x) {
    std::int64_t i;
    std::memcpy(&i, &x, sizeof i);
    return i >= 0 ? i : std::numeric_limits<std::int64_t>::min() - i;
}
double unkey(std::int64_t k) {
    const std::int64_t i = k >= 0 ? k : std::numeric_limits<std::int64_t>::min() - k;
    double x;
    std::memcpy(&x, &i, sizeof x);
    return x;
}
// pred(lo) true, pred(hi) false: the largest double in [lo, hi) where pred holds, if pred is
// monotone (true, then false): at most 64 evaluations.
template <class P>
double last_true(double lo, double hi, P&& pred) {
    std::int64_t a = okey(lo), b = okey(hi);
    while (static_cast<i128>(b) - a > 1) {
        const std::int64_t mid = static_cast<std::int64_t>(a + (static_cast<i128>(b) - a) / 2);
        if (pred(unkey(mid))) a = mid;
        else b = mid;
    }
    return unkey(a);
}
// pred(lo) false, pred(hi) true: the smallest double in (lo, hi] where pred holds.
template <class P>
double first_true(double lo, double hi, P&& pred) {
    std::int64_t a = okey(lo), b = okey(hi);
    while (static_cast<i128>(b) - a > 1) {
        const std::int64_t mid = static_cast<std::int64_t>(a + (static_cast<i128>(b) - a) / 2);
        if (pred(unkey(mid))) b = mid;
        else a = mid;
    }
    return unkey(b);
}

// ---- the binomial test ----------------------------------------------------------------------

// ceil and floor of the exact product p n (fma gives its sign against an integer exactly)
double ceil_prod(double p, double n) {
    double c = std::ceil(p * n);
    while (std::fma(p, n, -(c - 1.0)) <= 0.0) c -= 1.0;
    while (std::fma(p, n, -c) > 0.0) c += 1.0;
    return c;
}
double floor_prod(double p, double n) {
    double f = std::floor(p * n);
    while (std::fma(p, n, -f) < 0.0) f -= 1.0;
    while (std::fma(p, n, -(f + 1.0)) >= 0.0) f += 1.0;
    return f;
}

DD binom_lpmf(double k, double n, double p, DD q) { return stats::binom_detail::log_pmf(k, n, p, q); }

// scipy's two-sided p-value: k below p n, P(X <= k) plus the far side's terms from ceil(p n)
// on whose pmf is at most (1 + 1e-7) pmf(k) (the pmf does not increase there); k at or above
// p n, the same on the side below floor(p n). The comparisons on double-double log pmfs.
double binom_two_sided(double k, double n, double p) {
    if (p == 0.0) return k == 0.0 ? 1.0 : 0.0;
    if (p == 1.0) return k == n ? 1.0 : 0.0;
    const DD q = dd::two_sum(1.0, -p);
    const DD lthr = binom_lpmf(k, n, p, q) + dd::log(DD{kRerr, 0.0});
    auto small = [&](double x) { return dd_le(binom_lpmf(x, n, p, q), lthr); };
    double pv;
    if (std::fma(p, n, -k) > 0.0) {
        double a = ceil_prod(p, n), b = n + 1.0;  // the first x in [a, n] with small(x); n + 1 for none
        while (a < b) {  // (n < 2^53, checked by the caller: mid + 1 and mid - 1 move; the guard is for safety)
            const double mid = std::floor((a + b) / 2.0);
            const double a0 = a, b0 = b;
            if (small(mid)) b = mid;
            else a = mid + 1.0;
            if (a == a0 && b == b0) return kNaN;
        }
        pv = Binomial::cdf(k, n, p) + Binomial::ccdf(a - 1.0, n, p);
    } else {
        double a = -1.0, b = floor_prod(p, n);  // the last x in [0, b] with small(x); -1 for none
        while (a < b) {
            const double mid = std::floor((a + b + 1.0) / 2.0);
            const double a0 = a, b0 = b;
            if (small(mid)) a = mid;
            else b = mid - 1.0;
            if (a == a0 && b == b0) return kNaN;
        }
        pv = Binomial::cdf(a, n, p) + Binomial::ccdf(k - 1.0, n, p);
    }
    return pv < 1.0 ? pv : (pv >= 1.0 ? 1.0 : kNaN);  // (NaN kept: a tail past sd.stats' range)
}

double binom_p(double k, double n, double p, int alt) {
    if (alt == kLess) return Binomial::cdf(k, n, p);
    if (alt == kGreater) return Binomial::ccdf(k - 1.0, n, p);
    return binom_two_sided(k, n, p);
}

nb::tuple binomtest(double k, double n, double p, int alt) { return nb::make_tuple(k / n, binom_p(k, n, p, alt)); }

// ---- the intervals of one proportion --------------------------------------------------------

enum Ci { kNormal = 0, kAgresti = 1, kExact = 2, kWilson = 3, kWilsonCC = 4, kJeffreys = 5, kBinomTest = 6 };

// The Wilson quadratic (n + z^2) t^2 - (2x + z^2) t + x^2 / n = 0 of x successes (x a count, or a
// count -+ 1/2 for the continuity correction): its larger root, a sum of positive terms, and the
// smaller one as the product x^2 / (n (n + z^2)) over it (no cancellation).
// Wilson's ends in double: every operation adds, multiplies, divides or takes the square root of
// positive numbers (the lower end as x^2 / ((z^2 + n) n) / upper, not as a difference), so each end
// is within ~4 ulp of its value (six roundings, sqrt halving its argument's), well inside the
// 1e-14 of the contract (H7: each end against itself); the double-doubles bought nothing there.
double wilson_big(double x, double n, double z, double z2) {
    const double disc = z2 + 4.0 * (x * (n - x)) / n;
    return (2.0 * x + z2 + std::sqrt(disc) * z) / (2.0 * (z2 + n));
}
double wilson_small(double x, double n, double z, double z2) {
    return x * x / ((z2 + n) * n) / wilson_big(x, n, z, z2);
}

// log C(n, j) as a double-double
DD log_choose(double n, double j) {
    return dd::loggamma(D(n + 1.0)) - dd::loggamma(D(j + 1.0)) - dd::loggamma(D(n - j + 1.0));
}
// 1 / (1 + e^-w): the q with log(q / (1 - q)) = w
double sigmoid(DD w) {
    if (w.hi >= 0.0) return 1.0 / (1.0 + num::exp_dd(-w));
    const double e = num::exp_dd(w);
    return e / (1.0 + e);
}

// The ends of the two-sided binomial test inverted (statsmodels' 'binom_test', alpha not halved):
// lower = sup{q < k/n: p(q) <= alpha}, upper = inf{q > k/n: p(q) <= alpha}, p(q) scipy's two-sided
// p-value of k at q. Below k/n, p(q) = P(X >= k) + P(X <= j), j the far side's last count, which
// steps up with q at known points s_j = max(j / n, t_j) (t_j: pmf(j) = (1 + 1e-7) pmf(k)); on
// each piece p is decreasing then increasing (p' = n (pmf_{n-1}(k - 1) - pmf_{n-1}(j)), a ratio
// monotone in q), so the pieces are scanned from k/n outwards and the first one that holds a
// rejection gives the end: its right end, or the root of p = alpha on its increasing part. Above
// k/n the same with P(X <= k) + P(X >= m).
struct BinomInversion {
    double k, n, alpha;
    DD lrerr = dd::log(DD{kRerr, 0.0});

    double lower() const {
        if (k == 0.0) return 0.0;
        double b = k / n;
        const DD lck = log_choose(n, k);
        for (double j = k - 1.0; j >= -1.0; j -= 1.0) {
            double a = 0.0;
            if (j >= 0.0) a = std::max(j / n, sigmoid((log_choose(n, j) - lck - lrerr) / (k - j)));
            a = std::min(a, b);
            auto pv = [&](double q) {
                return Binomial::ccdf(k - 1.0, n, q) + (j >= 0.0 ? Binomial::cdf(j, n, q) : 0.0);
            };
            if (pv(b) <= alpha) return b;
            if (j < k - 1.0) {
                double mm = a;
                if (j >= 0.0) {
                    const DD w = (log_choose(n - 1.0, j) - log_choose(n - 1.0, k - 1.0)) / (k - 1.0 - j);
                    mm = std::clamp(sigmoid(w), a, b);
                }
                if (pv(mm) <= alpha) return last_true(mm, b, [&](double q) { return pv(q) <= alpha; });
            }
            b = a;
            if (!(b > 0.0)) break;
        }
        return 0.0;
    }

    double upper() const {
        if (k == n) return 1.0;
        double a = k / n;
        const DD lck = log_choose(n, k);
        for (double mo = k + 1.0; mo <= n + 1.0; mo += 1.0) {
            double b = 1.0;
            if (mo <= n) b = std::min(mo / n, sigmoid((lrerr + lck - log_choose(n, mo)) / (mo - k)));
            b = std::max(b, a);
            auto pv = [&](double q) {
                return Binomial::cdf(k, n, q) + (mo <= n ? Binomial::ccdf(mo - 1.0, n, q) : 0.0);
            };
            if (pv(a) <= alpha) return a;
            if (mo > k + 1.0) {
                double mm = b;
                if (mo <= n) {
                    const DD w = (log_choose(n - 1.0, k) - log_choose(n - 1.0, mo - 1.0)) / (mo - 1.0 - k);
                    mm = std::clamp(sigmoid(w), a, b);
                }
                if (pv(mm) <= alpha) return first_true(a, mm, [&](double q) { return pv(q) <= alpha; });
            }
            a = b;
            if (!(a < 1.0)) break;
        }
        return 1.0;
    }
};

// The normal quantile of the intervals' tail: alpha / 2 two-sided, alpha one-sided.
double ci_z(double alpha, int alt) { return zq(alt == kTwoSided ? alpha / 2.0 : alpha); }

// One proportion's interval, alpha the level's complement: two-sided intervals put alpha / 2 in
// each tail (but the inverted binomial test, whose two-sided p-value is already two-sided), a
// one-sided one alpha in its tail; 'less' is [0, upper], 'greater' [lower, 1]. zt: ci_z(alpha,
// alt) when the caller has it (a batch's lanes share it), else NaN.
void binom_interval(double k, double n, double alpha, int method, int alt, double& lo, double& hi, double zt = kNaN) {
    if (!(n > 0.0)) {
        lo = hi = kNaN;
        return;
    }
    const double as = alt == kTwoSided ? alpha / 2.0 : alpha;
    const bool normal = method == kNormal || method == kAgresti || method == kWilson || method == kWilsonCC;
    const double zn = normal ? (zt == zt ? zt : zq(as)) : kNaN;
    switch (method) {
        case kNormal: {
            const double z = zn;
            const DD qd = quot(k, n);
            const DD half = sqrt_dd(dd::two_prod(k, n - k) / n / n / n) * z;
            lo = clip01(to_double(qd - half)), hi = clip01(to_double(qd + half));
            break;
        }
        case kAgresti: {
            const double z = zn;
            const DD z2 = dd::two_prod(z, z), nc = z2 + n;
            const DD c = z2 * 0.5 + k, f = z2 * 0.5 + (n - k);
            const DD qc = c / nc;
            const DD half = sqrt_dd(c * f / (nc * nc * nc)) * z;
            lo = clip01(to_double(qc - half)), hi = clip01(to_double(qc + half));
            break;
        }
        case kWilson: {
            const double z = zn, z2 = z * z;
            const double big = wilson_big(k, n, z, z2);  // (wilson_small(k, ...)'s denominator too)
            lo = k == 0.0 ? 0.0 : clip01(k * k / ((z2 + n) * n) / big);
            hi = k == n ? 1.0 : clip01(big);
            break;
        }
        case kWilsonCC: {
            const double z = zn, z2 = z * z;
            lo = k == 0.0 ? 0.0 : wilson_small(k - 0.5, n, z, z2);
            hi = k == n ? 1.0 : wilson_big(k + 0.5, n, z, z2);
            break;
        }
        case kJeffreys: lo = beta_icdf(as, k + 0.5, n - k + 0.5), hi = beta_iccdf(as, k + 0.5, n - k + 0.5); break;
        case kBinomTest:
            if (alt == kTwoSided) {
                const BinomInversion inv{k, n, alpha};
                lo = inv.lower(), hi = inv.upper();
                break;
            }
            [[fallthrough]];  // one-sided: Clopper-Pearson's (statsmodels')
        default:              // kExact
            lo = k == 0.0 ? 0.0 : beta_icdf(as, k, n - k + 1.0);
            hi = k == n ? 1.0 : beta_iccdf(as, k + 1.0, n - k);
    }
    if (alt == kLess) lo = 0.0;
    if (alt == kGreater) hi = 1.0;
}

nb::tuple binom_ci(nb::handle count, nb::handle nobs, double alpha, int method, int alt) {
    const NDArray& c = require_ndarray(count);
    const std::int64_t lanes = c.shape[0];
    const NDArray& n = vec_arg(nobs, lanes);
    Outs out(2, lanes);
    const std::int64_t cost = method == kBinomTest ? 100000 : (method == kExact || method == kJeffreys ? 2000 : 50);
    const double z = ci_z(alpha, alt);  // (one normal quantile for every lane)
    each_lane(lanes, cost, [&](std::int64_t l) {
        binom_interval(vec_at(c, l), vec_at(n, l), alpha, method, alt, out.p[0][l], out.p[1][l], z);
    });
    return out.tuple();
}

// ---- quantile_test ----------------------------------------------------------------------------

// The lane's values (NaNs dropped under 'omit'); false when a NaN propagates
bool lane_values(const Rows& r, std::int64_t l, bool omit, std::vector<double>& v) {
    v.clear();
    for (std::int64_t j = 0; j < r.n; ++j) {
        const double x = r.at(l, j);
        if (x != x) {
            if (!omit) return false;
            continue;
        }
        v.push_back(x);
    }
    return true;
}

// scipy's: T1 = #(x <= q), T2 = #(x < q) and Y ~ Binomial(n, p); 'less' P(Y >= T2) with T2,
// 'greater' P(Y <= T1) with T1, two-sided twice the smaller (at most 1) with its count (T2, type
// 2, where P(Y <= T1) is the larger).
nb::tuple quantile_test(nb::handle rows, double q, double p, int alt, bool omit) {
    const Rows r(rows_arg(rows));
    Outs out(3, r.lanes);
    each_lane(r.lanes, r.n + 64, [&](std::int64_t l) {
        double* o[3] = {out.p[0] + l, out.p[1] + l, out.p[2] + l};
        *o[0] = *o[1] = *o[2] = kNaN;
        std::int64_t n = 0, t1 = 0, t2 = 0;
        for (std::int64_t j = 0; j < r.n; ++j) {
            const double x = r.at(l, j);
            if (x != x) {
                if (!omit) return;
                continue;
            }
            ++n;
            t1 += x <= q;
            t2 += x < q;
        }
        if (n == 0 || q != q) return;
        const double nn = static_cast<double>(n), T1 = static_cast<double>(t1), T2 = static_cast<double>(t2);
        const double lo = Binomial::cdf(T1, nn, p), up = Binomial::ccdf(T2 - 1.0, nn, p);
        if (alt == kLess) *o[0] = T2, *o[1] = 2.0, *o[2] = up;
        else if (alt == kGreater) *o[0] = T1, *o[1] = 1.0, *o[2] = lo;
        else {
            const bool two = lo > up;
            *o[0] = two ? T2 : T1, *o[1] = two ? 2.0 : 1.0, *o[2] = std::min(1.0, 2.0 * std::min(lo, up));
        }
    });
    return out.tuple();
}

// scipy's interval of the quantile: the order statistics x[ppf(a) - 1] and x[isf(a)] of
// Binomial(n, p) (a = alpha / 2 two-sided, alpha one-sided), NaN past the sample's ends, -inf or
// inf for the open side of a one-sided test.
nb::tuple quantile_ci(nb::handle rows, double p, double alpha, int alt, bool omit) {
    const Rows r(rows_arg(rows));
    Outs out(2, r.lanes);
    each_lane(r.lanes, 4 * r.n + 64, [&](std::int64_t l) {
        std::vector<double> v;
        out.p[0][l] = out.p[1][l] = kNaN;
        if (!lane_values(r, l, omit, v) || v.empty()) return;
        std::sort(v.begin(), v.end());
        const double nn = static_cast<double>(v.size());
        const double a = alt == kTwoSided ? alpha / 2.0 : alpha;
        double lo = -kInf, hi = kInf;
        if (alt != kLess) {
            const double i = stats::method_of<Binomial, m::Icdf>(a, nn, p) - 1.0;
            lo = i >= 0.0 ? v[static_cast<std::size_t>(i)] : kNaN;
        }
        if (alt != kGreater) {
            const double i = stats::method_of<Binomial, m::Iccdf>(a, nn, p);
            hi = i < nn ? v[static_cast<std::size_t>(i)] : kNaN;
        }
        out.p[0][l] = lo, out.p[1][l] = hi;
    });
    return out.tuple();
}

// ---- poisson_means_test (the E-test) ------------------------------------------------------------
//
// scipy's definition: lambda2 = (k1 + k2 - diff n1) / (n1 + n2) the null estimate (<= 0: the
// result (0, 1)), T(x1, x2) = (x1/n1 - x2/n2 - diff) / sqrt(x1/n1^2 + x2/n2^2) and the p-value
// the Poisson(n1 (lambda2 + diff)) x Poisson(n2 lambda2) probability of the (x1, x2) with T at
// least as extreme as T(k1, k2) (T(0, 0) = -inf for diff > 0, and for diff = 0 undefined: never
// counted). For a fixed x2, T increases with x1, so each x2 adds pmf(x2) times one or two
// Poisson tails of X1 (no double sum; scipy truncates both sums to [ppf(1e-10), ppf(1 - 1e-16)]).
// The sum over x2 runs out from the mode until the mass left in X2's tails (a geometric bound of
// the pmf) is below 2^-60 of the sum. Ties: T within 2^-90 of the observed one's scale (the
// double-double T's error is ~2^-104 of it) counts as equal.
struct EStat {
    double n1, n2, diff;
    // T as a double-double and its scale; nan for (0, 0) with diff = 0
    DD t(double x1, double x2, double& scale, bool& nan) const {
        nan = false;
        const DD a = quot(x1, n1), b = quot(x2, n2);
        const DD d = a - b - diff;
        const DD v = num::div(a, n1) + num::div(b, n2);
        if (!(v.hi > 0.0)) {
            scale = 0.0;
            if (d.hi < 0.0) return DD{-kInf, 0.0};
            nan = true;
            return DD{kNaN, 0.0};
        }
        const DD s = sqrt_dd(v);
        scale = (a.hi + b.hi + diff) / s.hi;
        return d / s;
    }
};

// The x2 steps of poisson_means at most (H154): past them a ValueError (the third value 1).
constexpr std::int64_t kPoissonSteps = 20000000;

struct PoissonOut {
    double stat, p;
    bool capped;
};
PoissonOut poisson_means_body(double k1, double n1, double k2, double n2, double diff, int alt);

// (statistic, p-value, capped): the body without the GIL when it is long (sqrt(mu2) x2 steps,
// each a few bisections)
nb::tuple poisson_means(double k1, double n1, double k2, double n2, double diff, int alt) {
    const double mu2 = std::max(0.0, (k1 + k2 - diff * n1) / (n1 + n2) * n2);
    PoissonOut o;
    {
        const parallel::Unlocked unlocked(std::sqrt(mu2) * 400.0 >= static_cast<double>(kParallelThreshold));
        o = poisson_means_body(k1, n1, k2, n2, diff, alt);
    }
    return nb::make_tuple(o.stat, o.p, o.capped ? 1.0 : 0.0);
}

PoissonOut poisson_means_body(double k1, double n1, double k2, double n2, double diff, int alt) {
    const DD num_ = D(k1 + k2) - dd::two_prod(diff, n1);
    const DD lam2 = num_ / dd::two_sum(n1, n2);
    if (!(lam2.hi > 0.0)) return {0.0, 1.0, false};
    const EStat E{n1, n2, diff};
    double sobs;
    bool nanobs;
    const DD tobs = E.t(k1, k2, sobs, nanobs);
    const double stat = to_double(tobs);
    const DD mu1 = (lam2 + diff) * n1, mu2 = lam2 * n2;
    const DD c = alt == kTwoSided ? abs_dd(tobs) : tobs;
    // x1 is in the upper region: T >= c (ties within tolerance); the lower region T <= -c
    // (two-sided) or T <= c ('less')
    auto upper_ok = [&](double x1, double x2) {
        double s;
        bool nan;
        const DD t = E.t(x1, x2, s, nan);
        if (nan) return false;
        if (!std::isfinite(t.hi)) return t.hi > 0.0;  // T(0, 0) = -inf
        const DD dlt = t - c;
        return dlt.hi >= 0.0 || std::fabs(dlt.hi) <= 0x1p-90 * (s + sobs);
    };
    const DD cl = alt == kTwoSided ? -c : c;
    auto lower_ok = [&](double x1, double x2) {
        double s;
        bool nan;
        const DD t = E.t(x1, x2, s, nan);
        if (nan) return false;
        if (!std::isfinite(t.hi)) return t.hi < 0.0;
        const DD dlt = t - cl;
        return dlt.hi <= 0.0 || std::fabs(dlt.hi) <= 0x1p-90 * (s + sobs);
    };
    // the smallest x1 >= 0 in the upper region (T increases with x1: galloping from a guess, then
    // bisection; inf when past 2^52)
    auto first_upper = [&](double x2, double guess) {
        const double g = std::max(0.0, std::floor(guess));
        double lo, hi;  // lo: not in the region (-1: below 0), hi: in it
        if (upper_ok(g, x2)) {
            hi = g;
            for (double step = 1.0;; step *= 2.0) {
                const double t = hi - step;
                if (t < 0.0) {
                    lo = -1.0;
                    break;
                }
                if (!upper_ok(t, x2)) {
                    lo = t;
                    break;
                }
                hi = t;
            }
        } else {
            lo = g;
            for (double step = 1.0;; step *= 2.0) {
                const double t = g + step;
                if (t > 0x1p52) return kInf;
                if (upper_ok(t, x2)) {
                    hi = t;
                    break;
                }
                lo = t;
            }
        }
        while (hi - lo > 1.0) {
            const double mid = std::floor((lo + hi) / 2.0);
            if (upper_ok(mid, x2)) hi = mid;
            else lo = mid;
        }
        return hi;
    };
    // the largest x1 >= s0 in the lower region, s0 - 1 for none; s0 = 1 at x2 = 0 when T(0, 0) is
    // undefined (diff = 0: the region is then [1, b])
    auto last_lower = [&](double x2, double guess, double s0) {
        if (!lower_ok(s0, x2)) return s0 - 1.0;
        const double g = std::max(s0, std::floor(guess));
        double lo, hi;  // lo: in the region, hi: not
        if (lower_ok(g, x2)) {
            lo = g;
            for (double step = 1.0;; step *= 2.0) {
                const double t = g + step;
                if (t > 0x1p52) return kInf;
                if (!lower_ok(t, x2)) {
                    hi = t;
                    break;
                }
                lo = t;
            }
        } else {
            hi = g;
            for (double step = 1.0;; step *= 2.0) {
                const double t = hi - step;
                if (t <= s0) {
                    lo = s0;
                    break;
                }
                if (lower_ok(t, x2)) {
                    lo = t;
                    break;
                }
                hi = t;
            }
        }
        while (hi - lo > 1.0) {
            const double mid = std::floor((lo + hi) / 2.0);
            if (lower_ok(mid, x2)) lo = mid;
            else hi = mid;
        }
        return lo;
    };
    // Poisson(mu1) tails: P(X1 >= a) = P(a, mu1), P(X1 <= b) = Q(b + 1, mu1)
    auto up_tail = [&](double a) {
        if (a <= 0.0) return 1.0;
        if (std::isinf(a)) return 0.0;
        return stats::gamma_detail::tail(a, mu1, false);
    };
    auto low_tail = [&](double b) {
        if (b < 0.0) return 0.0;
        if (std::isinf(b)) return 1.0;
        return stats::gamma_detail::tail(b + 1.0, mu1, true);
    };
    auto pmf2 = [&](double x2) { return num::exp_dd(stats::gamma_detail::log_density_std(x2 + 1.0, mu2)); };
    const bool want_up = alt != kLess, want_low = alt != kGreater;
    if (alt == kTwoSided && std::fabs(tobs.hi) <= 0x1p-90 * sobs) {
        // |T| >= 0 holds everywhere but at (0, 0), which counts only for diff > 0 (T = -inf)
        const double p00 = diff > 0.0 ? 0.0 : num::exp_dd(-(mu1 + mu2));
        return {stat, 1.0 - p00, false};
    }
    const double mode = std::floor(mu2.hi);
    double sum = 0.0;
    std::int64_t steps = 0;
    const std::int64_t cap = kPoissonSteps;
    // the sums reach about 9.2 sqrt(mu2) on each side of the mode (a tail of 2^-60): far past
    // the cap, no need to start
    if (16.0 * std::sqrt(mu2.hi) > 1.5 * static_cast<double>(cap)) return {stat, kNaN, true};
    double gu = to_double(mu1), gl = gu;
    auto term = [&](double x2, double pm, double& gup, double& glo) {
        double t = 0.0;
        if (want_up) {
            const double a = first_upper(x2, gup);
            if (std::isfinite(a)) gup = a;
            t += up_tail(a);
        }
        if (want_low) {
            const double s0 = (x2 == 0.0 && diff == 0.0) ? 1.0 : 0.0;
            const double b = last_lower(x2, glo, s0);
            if (b >= s0 && std::isfinite(b)) {
                glo = b;
                t += low_tail(b) - (s0 > 0.0 ? num::exp_dd(-mu1) : 0.0);
            }
        }
        return pm * t;
    };
    // upwards from the mode
    double gup = gu, glo = gl;
    for (double x2 = mode;; x2 += 1.0) {
        const double pm = pmf2(x2);
        sum += term(x2, pm, gup, glo);
        if (++steps > cap) return {stat, kNaN, true};
        if (x2 + 1.0 > mu2.hi) {  // the mass above x2: at most pmf(x2 + 1) / (1 - mu2 / (x2 + 2))
            const double rest = pm * (mu2.hi / (x2 + 1.0)) / (1.0 - mu2.hi / (x2 + 2.0));
            if (rest <= 0x1p-60 * sum || rest < 1e-320) break;
        }
    }
    gup = gu, glo = gl;
    for (double x2 = mode - 1.0; x2 >= 0.0; x2 -= 1.0) {
        const double pm = pmf2(x2);
        sum += term(x2, pm, gup, glo);
        if (++steps > cap) return {stat, kNaN, true};
        if (x2 < mu2.hi) {  // the mass below x2: at most pmf(x2 - 1) / (1 - (x2 - 1) / mu2)
            const double rest = x2 >= 1.0 ? pm * (x2 / mu2.hi) / (1.0 - (x2 - 1.0) / mu2.hi) : 0.0;
            if (rest <= 0x1p-60 * sum || rest < 1e-320) break;
        }
    }
    return {stat, std::min(1.0, sum), false};
}

// ---- statsmodels' proportions_ztest and proportions_chisquare -----------------------------------

// count and nobs as (lanes, k), k = 1 or 2: one sample, (count / nobs - value) / sqrt(pp (1 - pp)
// / nobs), pp = count / nobs (or prop_var); two samples, (p1 - p2 - value) / sqrt(pp (1 - pp) (1/n1
// + 1/n2)), pp the pooled proportion (or prop_var). NaN prop_var: not given.
nb::tuple prop_ztest(nb::handle count, nb::handle nobs, double value, double prop_var, int alt) {
    const Rows c(rows_arg(count)), n(rows_arg(nobs));
    Outs out(2, c.lanes);
    each_lane(c.lanes, 64, [&](std::int64_t l) {
        double z;
        if (c.n == 1) {
            const double k = c.at(l, 0), nn = n.at(l, 0);
            if (!(nn > 0.0)) {
                out.p[0][l] = out.p[1][l] = kNaN;
                return;
            }
            const DD d = quot(k, nn) - value;
            const DD v = prop_var == prop_var ? D(prop_var) * dd::two_sum(1.0, -prop_var) / nn
                                              : dd::two_prod(k, nn - k) / nn / nn / nn;
            z = over_sqrt(d, v);
        } else {
            const double k1 = c.at(l, 0), k2 = c.at(l, 1), n1 = n.at(l, 0), n2 = n.at(l, 1);
            if (!(n1 > 0.0 && n2 > 0.0)) {
                out.p[0][l] = out.p[1][l] = kNaN;
                return;
            }
            const double N = n1 + n2, K = k1 + k2;
            const DD d = quot(k1, n1) - quot(k2, n2) - value;
            const DD v = prop_var == prop_var ? D(prop_var) * dd::two_sum(1.0, -prop_var) * N / dd::two_prod(n1, n2)
                                              : dd::two_prod(K, N - K) / N / dd::two_prod(n1, n2);
            z = over_sqrt(d, v);
        }
        out.p[0][l] = z;
    });
    z_pvalues(out.p[0], c.lanes, alt, out.p[1]);  // (z_pvalue's bits, by the normal's chunk kernel)
    return out.tuple();
}

// statsmodels' chi^2 test of k proportions against the null value v: the 2 k cells (count,
// nobs - count) against (nobs v, nobs (1 - v)), sum (c - n v)^2 / (n v (1 - v)), df = k; the
// expected counts as (lanes, k, 2).
nb::tuple prop_chisquare(nb::handle count, nb::handle nobs, double v) {
    const Rows c(rows_arg(count)), n(rows_arg(nobs));
    Outs out(3, c.lanes);
    const std::int64_t shape[] = {c.lanes, c.n, 2};
    NDArray expected = new_array(DType::Float64, shape);
    double* ex = reinterpret_cast<double*>(expected.data);
    const DD vq = D(v) * dd::two_sum(1.0, -v);
    each_lane(c.lanes, 16 * c.n, [&](std::int64_t l) {
        Acc s;
        bool bad = false;
        for (std::int64_t i = 0; i < c.n; ++i) {
            const double k = c.at(l, i), nn = n.at(l, i);
            ex[(l * c.n + i) * 2] = to_double(dd::two_prod(nn, v));
            ex[(l * c.n + i) * 2 + 1] = to_double(dd::two_sum(1.0, -v) * nn);
            if (!(nn > 0.0)) bad = true;
            const DD d = D(k) - dd::two_prod(nn, v);
            const DD den = vq * nn;
            if (den.hi == 0.0) s.add(d.hi == 0.0 ? kNaN : kInf);
            else s.add(d * d / den);
        }
        const double stat = bad ? kNaN : to_double(s.value());
        out.p[0][l] = stat, out.p[1][l] = chi2_sf(stat, static_cast<double>(c.n)),
        out.p[2][l] = static_cast<double>(c.n);
    });
    nb::tuple t = out.tuple();
    return nb::make_tuple(t[0], t[1], t[2], wrap(std::move(expected)));
}

// ---- two independent proportions (statsmodels' test_proportions_2indep and
// confint_proportions_2indep) ---------------------------------------------------------------------

enum Compare { kDiff = 0, kRatio = 1, kOdds = 2 };
enum Method2 {
    kWald = 0,
    kAgrestiCaffo = 1,
    kScore = 2,
    kLog = 3,
    kLogAdjusted = 4,
    kLogit = 5,
    kLogitAdjusted = 6,
    kLogitSmoothed = 7,
    kNewcomb = 8
};

// A root of the polynomial c[0] x^d + ... + c[d] polished from x0 by Newton's method in
// double-double (its value at x0 is a double-double Horner sum)
DD polish(const DD* c, int d, double x0) {
    DD x = D(x0);
    for (int it = 0; it < 4; ++it) {
        DD f = c[0];
        double fp = 0.0;
        for (int i = 1; i <= d; ++i) {
            fp = fp * x.hi + f.hi;
            f = f * x + c[i];
        }
        if (!(fp != 0.0) || !std::isfinite(fp) || !std::isfinite(f.hi)) break;
        const DD dx = f / fp;
        const DD nx = x - dx;
        if (!std::isfinite(nx.hi)) break;
        x = nx;
        if (std::fabs(dx.hi) <= 0x1p-104 * std::fabs(x.hi)) break;
    }
    return x;
}

struct Counts {
    double c1, n1, c2, n2;
};

// The score test (Miettinen and Nurminen; statsmodels' score_test_proportions_2indep): the
// constrained estimates p1, p0 under the null value v (the pooled proportion at v = 0 or 1),
// the variance (times N / (N - 1) with the correction) and z.
struct Score {
    double z, var, p1, p0;
};
Score score(const Counts& k, int compare, double v, bool corr) {
    const double c1 = k.c1, n1 = k.n1, c0 = k.c2, n0 = k.n2;
    const double N = n1 + n0, C = c1 + c0;
    DD p1, p0, dstat, var;
    if (compare == kDiff) {
        if (v == 0.0) {
            p0 = p1 = quot(C, N);
        } else {
            // The constrained maximum likelihood estimate of p0 (p1 = p0 + v): the log likelihood
            // c1 log(p1) + (n1 - c1) log(1 - p1) + c0 log(p0) + (n0 - c0) log(1 - p0) is concave on
            // the feasible (max(0, -v), min(1, 1 - v)), so its derivative has at most one root there
            // (bisection on its sign), a root of Miettinen and Nurminen's cubic N x^3 + L2 x^2 +
            // L1 x + L0, L1 = (n0 v - N - 2 c0) v + C (polished on it in double-double); without one
            // the estimate is the end the
            // derivative's sign points to. (statsmodels 0.15.0's cubic has c0 v for n0 v in L1, so its
            // root is not the estimate unless v = 0: H98.)
            const DD c2d = dd::two_prod(n1 + 2.0 * n0, v) - (N + C);
            const DD c1d = (dd::two_prod(n0, v) - (N + 2.0 * c0)) * v + C;
            const DD c0d = dd::two_prod(c0, v) * dd::two_sum(1.0, -v);
            const DD cs[4] = {D(N), c2d, c1d, c0d};
            auto dl = [&](double x) {  // the log likelihood's derivative in p0
                const double x1 = x + v;
                double s_ = 0.0;
                if (c1 > 0.0) s_ += c1 / x1;
                if (n1 > c1) s_ -= (n1 - c1) / (1.0 - x1);
                if (c0 > 0.0) s_ += c0 / x;
                if (n0 > c0) s_ -= (n0 - c0) / (1.0 - x);
                return s_;
            };
            const double lo = std::max(0.0, -v), hi = std::min(1.0, 1.0 - v);
            const double ilo = std::nextafter(lo, hi), ihi = std::nextafter(hi, lo);
            double r0;
            bool interior = false;
            if (!(dl(ilo) > 0.0)) {
                r0 = lo;
            } else if (!(dl(ihi) < 0.0)) {
                r0 = hi;
            } else {
                r0 = last_true(ilo, ihi, [&](double x) { return dl(x) > 0.0; });
                interior = true;
            }
            if (interior) {
                p0 = polish(cs, 3, r0);
                if (!(p0.hi > lo && p0.hi < hi)) p0 = D(r0);
            } else {
                p0 = D(r0);
            }
            p1 = p0 + v;
        }
        var = p1 * (D(1.0) - p1) / n1 + p0 * (D(1.0) - p0) / n0;
        dstat = quot(c1, n1) - quot(c0, n0) - v;
    } else if (compare == kRatio) {
        if (v == 1.0) {
            p0 = p1 = quot(C, N);
        } else {
            const DD a = dd::two_prod(N, v);
            const DD b = -(dd::two_prod(n1, v) + (c1 + n0) + dd::two_prod(c0, v));
            const DD disc = b * b - a * (4.0 * C);
            const DD r = disc.hi >= 0.0 ? D(2.0 * C) / (-b + sqrt_dd(disc)) : D(kNaN);
            const DD cs[3] = {a, b, D(C)};
            p0 = r.hi == r.hi && r.hi != 0.0 ? polish(cs, 2, r.hi) : r;
            p1 = p0 * v;
        }
        var = p1 * (D(1.0) - p1) / n1 + dd::two_prod(v, v) * p0 * (D(1.0) - p0) / n0;
        dstat = quot(c1, n1) - quot(c0, n0) * v;
    } else {
        if (v == 1.0) {
            p0 = p1 = quot(C, N);
        } else {
            const DD a = D(n0) * dd::two_sum(v, -1.0);
            const DD b = dd::two_prod(n1, v) + n0 - dd::two_sum(v, -1.0) * C;
            const DD disc = b * b + a * (4.0 * C);
            DD r = D(kNaN);
            if (disc.hi >= 0.0) {
                const DD s = sqrt_dd(disc);
                r = b.hi > 0.0 ? D(2.0 * C) / (b + s) : (-b + s) / (a * 2.0);
            }
            const DD cs[3] = {a, b, D(-C)};
            p0 = r.hi == r.hi && r.hi != 0.0 ? polish(cs, 2, r.hi) : r;
            p1 = p0 * v / (D(1.0) + p0 * dd::two_sum(v, -1.0));
        }
        // statsmodels keeps the estimates away from 0 and 1
        constexpr double eps = 1e-10;
        auto clip = [&](DD x) { return x.hi < eps ? D(eps) : (x.hi > 1.0 - eps ? D(1.0 - eps) : x); };
        p0 = clip(p0), p1 = clip(p1);
        const DD q1 = p1 * (D(1.0) - p1), q0 = p0 * (D(1.0) - p0);
        var = D(1.0) / (q1 * n1) + D(1.0) / (q0 * n0);
        dstat = (quot(c1, n1) - p1) / q1 - (quot(c0, n0) - p0) / q0;
    }
    if (corr) var = var * N / (N - 1.0);
    return {over_sqrt(dstat, var), to_double(var), to_double(p1), to_double(p0)};
}

// The adjusted counts of the Wald-type statistics: success and failure counts of each sample
struct Adj {
    DD s1, f1, s2, f2;
};
Adj adjusted(const Counts& k, int method) {
    const double c1 = k.c1, n1 = k.n1, c2 = k.c2, n2 = k.n2;
    if (method == kLogitSmoothed) {  // statsmodels' _shrink_prob: 2 times the independence table
        const double N = n1 + n2, C = c1 + c2, N2 = N * N;
        const DD a1 = dd::two_prod(2.0 * n1, C) / N2, b1 = dd::two_prod(2.0 * n1, N - C) / N2;
        const DD a2 = dd::two_prod(2.0 * n2, C) / N2, b2 = dd::two_prod(2.0 * n2, N - C) / N2;
        return {a1 + c1, b1 + (n1 - c1), a2 + c2, b2 + (n2 - c2)};
    }
    const double h = method == kAgrestiCaffo ? 1.0 : (method == kLogAdjusted || method == kLogitAdjusted ? 0.5 : 0.0);
    return {D(c1 + h), D(n1 - c1 + h), D(c2 + h), D(n2 - c2 + h)};
}

double log_dd_ratio(DD r) {  // log of a positive double-double (r = 0: -inf, inf: inf, NaN: NaN)
    if (r.hi != r.hi) return kNaN;
    if (r.hi == 0.0) return -kInf;
    if (std::isinf(r.hi)) return r.hi;
    return to_double(dd::log(r));
}
DD log_ratio_dd(DD r) {
    if (!(r.hi > 0.0) || std::isinf(r.hi)) return D(log_dd_ratio(r));
    return dd::log(r);
}

// The test statistic of a Wald-type method (and its variance)
void wald_type(const Counts& k, int compare, int method, double v, double& z, double& var) {
    if (compare == kDiff) {  // wald, agresti-caffo: the adjusted proportions' difference
        const double a = method == kAgrestiCaffo ? 1.0 : 0.0;
        const double c1 = k.c1 + a, n1 = k.n1 + 2.0 * a, c2 = k.c2 + a, n2 = k.n2 + 2.0 * a;
        if (v == 0.0 && n1 * n2 < 0x1p53) {
            // integer counts whose products are exact in double: the difference is one rounded
            // quotient of exact integers (no cancellation left), the variance a sum of positive
            // terms, z within a few ulp
            const double d = (c1 * n2 - c2 * n1) / (n1 * n2);
            const double vv = c1 * (n1 - c1) / n1 / n1 / n1 + c2 * (n2 - c2) / n2 / n2 / n2;
            z = vv > 0.0 ? d / std::sqrt(vv) : kNaN, var = vv;
            return;
        }
        const DD d = quot(c1, n1) - quot(c2, n2) - v;
        const DD vv = dd::two_prod(c1, n1 - c1) / n1 / n1 / n1 + dd::two_prod(c2, n2 - c2) / n2 / n2 / n2;
        z = over_sqrt(d, vv), var = to_double(vv);
        return;
    }
    if (compare == kRatio) {  // log, log-adjusted: count + h over nobs + h (statsmodels')
        const double h = method == kLogAdjusted ? 0.5 : 0.0;
        const double c1 = k.c1 + h, n1 = k.n1 + h, c2 = k.c2 + h, n2 = k.n2 + h;
        const DD r = sdiv(sdiv(dd::two_prod(c1, n2), dd::two_prod(n1, c2)), D(v));
        const DD vv = sadd(sdiv(D(k.n1 - k.c1), dd::two_prod(c1, n1)), sdiv(D(k.n2 - k.c2), dd::two_prod(c2, n2)));
        const DD l = log_ratio_dd(r);
        z = std::isfinite(vv.hi) ? over_sqrt(l, vv) : to_double(l) / vv.hi;
        var = to_double(vv);
        return;
    }
    const Adj a = adjusted(k, method);
    const DD r = sdiv(sdiv(a.s1 * a.f2, a.f1 * a.s2), D(v));
    const DD vv = sadd(sadd(sdiv(D(1.0), a.s1), sdiv(D(1.0), a.f1)), sadd(sdiv(D(1.0), a.s2), sdiv(D(1.0), a.f2)));
    const DD l = log_ratio_dd(r);
    z = std::isfinite(vv.hi) ? over_sqrt(l, vv) : to_double(l) / vv.hi;
    var = to_double(vv);
}

Counts counts_at(const NDArray* const* a, std::int64_t l) {
    return {vec_at(*a[0], l), vec_at(*a[1], l), vec_at(*a[2], l), vec_at(*a[3], l)};
}

nb::tuple prop2_test(
    nb::handle c1, nb::handle n1, nb::handle c2, nb::handle n2, int compare, int method, double v, int alt, bool corr
) {
    const std::int64_t lanes = require_ndarray(c1).shape[0];
    const NDArray* a[4] = {&vec_arg(c1, lanes), &vec_arg(n1, lanes), &vec_arg(c2, lanes), &vec_arg(n2, lanes)};
    Outs out(8, lanes);
    each_lane(lanes, 256, [&](std::int64_t l) {
        const Counts k = counts_at(a, l);
        double z, var, p1 = kNaN, p0 = kNaN;
        if (!(k.n1 > 0.0 && k.n2 > 0.0)) {
            z = var = kNaN;
        } else if (method == kScore) {
            const Score s = score(k, compare, v, corr);
            z = s.z, var = s.var, p1 = s.p1, p0 = s.p0;
        } else {
            wald_type(k, compare, method, v, z, var);
        }
        // the sample's difference, ratio and odds ratio: exact rationals rounded once (counts whose
        // products stay below 2^53: exact in double, and one division rounds correctly)
        if (k.n1 * k.n2 < 0x1p53) {
            auto fq = [](double nn, double d) { return d == 0.0 ? (nn == 0.0 ? kNaN : kInf) : nn / d; };
            out.p[0][l] = z, out.p[2][l] = var, out.p[3][l] = p1, out.p[4][l] = p0;
            out.p[5][l] = (k.c1 * k.n2 - k.c2 * k.n1) / (k.n1 * k.n2);
            out.p[6][l] = fq(k.c1 * k.n2, k.n1 * k.c2), out.p[7][l] = fq(k.c1 * (k.n2 - k.c2), (k.n1 - k.c1) * k.c2);
            return;
        }
        const i128 C1 = static_cast<i128>(k.c1), N1 = static_cast<i128>(k.n1), C2 = static_cast<i128>(k.c2),
                   N2 = static_cast<i128>(k.n2);
        const double diff = to_double(exact::to_dd(C1 * N2 - C2 * N1) / exact::to_dd(N1 * N2));
        const i128 rn = C1 * N2, rd = N1 * C2, on = C1 * (N2 - C2), od = (N1 - C1) * C2;
        auto frac = [](i128 nn, i128 d) {
            if (d == 0) return nn == 0 ? kNaN : kInf;
            return to_double(exact::to_dd(nn) / exact::to_dd(d));
        };
        out.p[0][l] = z, out.p[2][l] = var, out.p[3][l] = p1, out.p[4][l] = p0;
        out.p[5][l] = diff, out.p[6][l] = frac(rn, rd), out.p[7][l] = frac(on, od);
    });
    z_pvalues(out.p[0], lanes, alt, out.p[1]);  // (z_pvalue's bits, by the normal's chunk kernel)
    return out.tuple();
}

// The interval of a score test inverted (statsmodels' _score_confint_inversion: the two-sided
// p-value equal to alpha, |z(v)| = z_{alpha/2}; z decreases in v): each end the v where z crosses
// its bound, found from an anchor (the sample's value; for an odds ratio of 0 or inf the
// logit-adjusted one) by steps out (doubling; for a difference at most half the way left to -1 or
// 1, for an odds ratio factors e^h) to the first v on the far side, then bisection; the domain's
// end where z does not reach the bound (statsmodels' brentq raises there). An odds ratio of 0 has
// the lower end 0, of inf the upper end inf.
void score_inversion(const Counts& k, int compare, double alpha, bool corr, double& lo, double& hi) {
    const double zc = zq(alpha / 2.0);
    auto z = [&](double v) { return score(k, compare, v, corr).z; };
    const bool diff = compare == kDiff;
    double param;
    if (diff) {
        param = to_double(quot(k.c1, k.n1) - quot(k.c2, k.n2));
    } else {
        const double on = k.c1 * (k.n2 - k.c2), od = (k.n1 - k.c1) * k.c2;
        param = od == 0.0 ? (on == 0.0 ? kNaN : kInf) : on / od;
    }
    if (param != param) {
        lo = hi = kNaN;
        return;
    }
    double anchor = param;
    if (!diff && !(param > 0.0 && param < kInf)) {
        const DD r = DD{k.c1 + 0.5, 0.0} * (k.n2 - k.c2 + 0.5) / (DD{k.n1 - k.c1 + 0.5, 0.0} * (k.c2 + 0.5));
        anchor = to_double(r);
    }
    const DD vw =
        dd::two_prod(k.c1, k.n1 - k.c1) / k.n1 / k.n1 / k.n1 + dd::two_prod(k.c2, k.n2 - k.c2) / k.n2 / k.n2 / k.n2;
    const double h0 = diff ? std::max(0.25 * zc * std::sqrt(std::max(vw.hi, 0.0)), 1e-6) : 0.25;
    const double vlo = diff ? -1.0 : 0.0, vhi = diff ? 1.0 : kInf;
    // the boundary where below(v) turns from true (smaller v) to false
    auto boundary = [&](auto&& below) {
        const bool up = below(anchor);  // step towards larger v while below holds
        double prev = anchor, h = h0;
        for (int i = 0; i < 200; ++i, h *= 2.0) {
            double v;
            const double end = up ? vhi : vlo;
            if (diff) {
                v = anchor + (up ? h : -h);
                if (!(up ? v < end : v > end)) v = prev + (end - prev) / 2.0;
            } else {
                v = anchor * std::exp(up ? h : -h);
            }
            if (v == prev || !(v > vlo && v < vhi)) return end;
            if (below(v) != up) return up ? last_true(prev, v, below) : last_true(v, prev, below);
            prev = v;
        }
        return up ? vhi : vlo;
    };
    lo = (!diff && param == 0.0) ? 0.0 : boundary([&](double v) { return z(v) >= zc; });
    hi = (!diff && param == kInf) ? kInf : boundary([&](double v) { return z(v) > -zc; });
}

// The real roots of a1 x^3 + a2 x^2 + a3 x + a4 (double-double coefficients), sorted; false
// unless three are real. Trigonometric starts, polished in double-double.
bool cubic_roots(const DD* c, DD* r) {
    const double a = c[0].hi, b = c[1].hi / a, cc = c[2].hi / a, d = c[3].hi / a;
    if (c[3].hi == 0.0) {  // a root at 0 and the quadratic's two
        const DD disc = c[1] * c[1] - c[0] * c[2] * 4.0;
        if (disc.hi < 0.0) return false;
        const DD s = sqrt_dd(disc);
        const DD qd = c[1].hi >= 0.0 ? -(c[1] + s) * 0.5 : (s - c[1]) * 0.5;  // q = -(b + sign(b) s) / 2
        DD x1 = qd / c[0], x2 = qd.hi != 0.0 ? c[2] / qd : D(0.0);
        DD v[3] = {D(0.0), x1, x2};
        std::sort(v, v + 3, [](DD u, DD w) { return u.hi < w.hi || (u.hi == w.hi && u.lo < w.lo); });
        for (int i = 0; i < 3; ++i) r[i] = v[i];
        return true;
    }
    const double p = cc - b * b / 3.0, q = 2.0 * b * b * b / 27.0 - b * cc / 3.0 + d;
    // three real roots (a double one within the rounding of p and q counts: numpy's roots give it
    // as a pair with a tiny imaginary part, statsmodels takes the real part)
    const double disc = 4.0 * p * p * p + 27.0 * q * q;
    if (!(p < 0.0) || disc > 1e-12 * (std::fabs(4.0 * p * p * p) + 27.0 * q * q)) return false;
    const double m2 = 2.0 * std::sqrt(-p / 3.0);
    const double th = std::acos(std::clamp(3.0 * q / (p * m2), -1.0, 1.0)) / 3.0;
    DD v[3];
    for (int i = 0; i < 3; ++i) v[i] = polish(c, 3, m2 * std::cos(th - 2.0 * kPi * i / 3.0) - b / 3.0);
    std::sort(v, v + 3, [](DD u, DD w) { return u.hi < w.hi || (u.hi == w.hi && u.lo < w.lo); });
    for (int i = 0; i < 3; ++i) r[i] = v[i];
    return true;
}

// Koopman's interval of the risk ratio (statsmodels' _confint_riskratio_koopman, Nam's
// equations 5 and 6): the two smallest roots of a cubic in the null p0, each mapped to a ratio
void koopman(const Counts& k, double alpha, bool corr, double& lo, double& hi) {
    const double x0 = k.c2, x1 = k.c1, n0 = k.n2, n1 = k.n1, x = x0 + x1, n = n0 + n1;
    const double zc = zq(alpha / 2.0);
    DD z = dd::two_prod(zc, zc);
    if (corr) z = z * n / (n - 1.0);
    auto I = [](double v) { return static_cast<i128>(v); };
    auto T = [](i128 v) { return exact::to_dd(v); };
    const DD a1 = (T(I(n0) * I(n0) * I(n) * I(x1)) + T(I(n0) * I(n1) * (I(n0) + I(x1))) * z);
    const DD a2 =
        -(T(I(n0) * I(n0) * I(n1) * I(x) + 2 * I(n0) * I(n) * I(x0) * I(x1)) +
          T(I(n0) * I(n1) * (I(n0) + I(x0) + 2 * I(x1))) * z);
    const DD a3 = T(2 * I(n0) * I(n1) * I(x0) * I(x) + I(n) * I(x0) * I(x0) * I(x1)) + T(I(n0) * I(n1) * I(x)) * z;
    const DD a4 = T(-I(n1) * I(x0) * I(x0) * I(x));
    const DD c[4] = {a1, a2, a3, a4};
    DD r[3];
    const bool full = x0 == n0;  // p0 = 1 is then a double root (and the third -a4 / a1)
    if (full) {
        r[0] = r[1] = D(1.0), r[2] = -a4 / a1;
        std::sort(r, r + 3, [](DD u, DD w) { return u.hi < w.hi || (u.hi == w.hi && u.lo < w.lo); });
    } else if (!(a1.hi != 0.0) || !cubic_roots(c, r)) {
        lo = hi = kNaN;
        return;
    }
    auto g = [&](DD p) {  // (1 - (n1 - x1)(1 - p) / (x0 + n1 - n p)) / p
        if (full && p.hi == 1.0 && p.lo == 0.0) return to_double(D(1.0) - D(n1 - x1) / n);  // (its limit at 1)
        const DD num_ = D(1.0) - (D(1.0) - p) * (n1 - x1) / (D(x0 + n1) - p * n);
        if (p.hi == 0.0) return num_.hi == 0.0 ? kNaN : std::copysign(kInf, num_.hi);
        return to_double(num_ / p);
    };
    lo = std::max(0.0, g(r[1])), hi = std::max(0.0, g(r[0]));  // (a ratio: the rounding of an end at 0)
    if (x1 == 0.0 && !full) lo = 0.0;  // p0 = x0 / n0 is then the root, and its ratio is 0 exactly
}

// zt: zq(alpha / 2) when the caller has it (a batch's lanes share it), else NaN
void prop2_interval(
    const Counts& k, int compare, int method, double alpha, bool corr, double& lo, double& hi, double zt = kNaN
) {
    if (!(k.n1 > 0.0 && k.n2 > 0.0)) {
        lo = hi = kNaN;
        return;
    }
    if (method == kScore) {
        if (compare == kRatio) koopman(k, alpha, corr, lo, hi);
        else score_inversion(k, compare, alpha, corr, lo, hi);
        return;
    }
    const double z = zt == zt ? zt : zq(alpha / 2.0);
    if (compare == kDiff && method == kNewcomb) {  // Wilson's interval of each (alpha / 2 a side)
        double l1, u1, l2, u2;
        binom_interval(k.c1, k.n1, alpha, kWilson, kTwoSided, l1, u1, z);
        binom_interval(k.c2, k.n2, alpha, kWilson, kTwoSided, l2, u2, z);
        if (k.n1 * k.n2 < 0x1p53) {
            // counts whose products are exact in double: d one rounded quotient of exact integers,
            // p_i = q_i + r_i from the quotient's exact remainder, the half widths sums of
            // positive squares; each end within a few ulp of itself and of the half width (the
            // Wilson ends' own ~4 ulp dominate, as their rounding did in double-doubles)
            const double q1 = k.c1 / k.n1, r1 = std::fma(-q1, k.n1, k.c1) / k.n1;
            const double q2 = k.c2 / k.n2, r2 = std::fma(-q2, k.n2, k.c2) / k.n2;
            const double d = (k.c1 * k.n2 - k.c2 * k.n1) / (k.n1 * k.n2);
            const double e1 = (q1 - l1) + r1, e2 = (u2 - q2) - r2, e3 = (q2 - l2) + r2, e4 = (u1 - q1) - r1;
            lo = d - std::sqrt(e1 * e1 + e2 * e2), hi = d + std::sqrt(e3 * e3 + e4 * e4);
            return;
        }
        const DD p1 = quot(k.c1, k.n1), p2 = quot(k.c2, k.n2), d = p1 - p2;
        const DD e1 = p1 - l1, e2 = D(u2) - p2, e3 = p2 - l2, e4 = D(u1) - p1;
        lo = to_double(d - sqrt_dd(e1 * e1 + e2 * e2)), hi = to_double(d + sqrt_dd(e3 * e3 + e4 * e4));
        return;
    }
    if (compare == kDiff) {
        const double a = method == kAgrestiCaffo ? 1.0 : 0.0;
        const double c1 = k.c1 + a, n1 = k.n1 + 2.0 * a, c2 = k.c2 + a, n2 = k.n2 + 2.0 * a;
        const DD d = quot(c1, n1) - quot(c2, n2);
        const DD vv = dd::two_prod(c1, n1 - c1) / n1 / n1 / n1 + dd::two_prod(c2, n2 - c2) / n2 / n2 / n2;
        const DD half = sqrt_dd(vv) * z;
        lo = to_double(d - half), hi = to_double(d + half);
        return;
    }
    DD r, vv;
    if (compare == kRatio) {
        const double h = method == kLogAdjusted ? 0.5 : 0.0;
        const double c1 = k.c1 + h, n1 = k.n1 + h, c2 = k.c2 + h, n2 = k.n2 + h;
        r = sdiv(dd::two_prod(c1, n2), dd::two_prod(n1, c2));
        vv = sadd(sdiv(D(k.n1 - k.c1), dd::two_prod(c1, n1)), sdiv(D(k.n2 - k.c2), dd::two_prod(c2, n2)));
    } else {
        const Adj a = adjusted(k, method);
        r = sdiv(a.s1 * a.f2, a.f1 * a.s2);
        vv = sadd(sadd(sdiv(D(1.0), a.s1), sdiv(D(1.0), a.f1)), sadd(sdiv(D(1.0), a.s2), sdiv(D(1.0), a.f2)));
    }
    // exp(log r -+ z sqrt(var)) = r exp(-+ d)
    const double rr = to_double(r), dl = z * to_double(sqrt_dd(vv));
    if (rr > 0.0 && std::isfinite(rr) && std::isfinite(dl)) {
        lo = rr * std::exp(-dl), hi = rr * std::exp(dl);
    } else {  // a zero count: IEEE's ends of exp(log r -+ d), statsmodels' (0 and NaN, ...)
        lo = std::exp(log_dd_ratio(r) - dl), hi = std::exp(log_dd_ratio(r) + dl);
    }
}

nb::tuple prop2_ci(
    nb::handle c1, nb::handle n1, nb::handle c2, nb::handle n2, int compare, int method, double alpha, bool corr
) {
    const std::int64_t lanes = require_ndarray(c1).shape[0];
    const NDArray* a[4] = {&vec_arg(c1, lanes), &vec_arg(n1, lanes), &vec_arg(c2, lanes), &vec_arg(n2, lanes)};
    Outs out(2, lanes);
    const double z = zq(alpha / 2.0);  // (one normal quantile for every lane)
    each_lane(lanes, method == kScore ? 100000 : 256, [&](std::int64_t l) {
        prop2_interval(counts_at(a, l), compare, method, alpha, corr, out.p[0][l], out.p[1][l], z);
    });
    return out.tuple();
}

}  // namespace

void bind_proportion(nb::module_& m) {
    m.def("binomtest", &binomtest);
    m.def("binom_ci", &binom_ci);
    m.def("quantile_test", &quantile_test);
    m.def("quantile_ci", &quantile_ci);
    m.def("poisson_means", &poisson_means);
    m.def("prop_ztest", &prop_ztest);
    m.def("prop_chisquare", &prop_chisquare);
    m.def("prop2_test", &prop2_test);
    m.def("prop2_ci", &prop2_ci);
}

}  // namespace sd::htest
