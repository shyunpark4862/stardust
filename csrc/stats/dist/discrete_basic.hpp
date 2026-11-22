// The binomial, Bernoulli, Poisson, geometric and negative binomial distributions
// (docs/api/stats.md §2.1, T25).
//
// The pmfs are the incomplete functions' prefixes, whose exponents are formed in
// double-double (special/prefix.hpp): for 1 <= k <= n - 1
//     binomial  C(n, k) p^k q^(n-k)  = B(k, n - k; p) n / (k (n - k)),   B(a, b; x) = x^a y^b / B(a, b)
//     neg. bin. Gamma(k+n)/(Gamma(n) k!) p^n q^k = B(n, k; p) / k
//     Poisson   mu^k e^-mu / k!       = D(k, mu),  D(a, x) = x^a e^-x / Gamma(a + 1)
// (Loader's saddle-point form with the deviance in double-double), the ends from
// k log q and n log p. The cdfs are sd.special's bdtr, nbdtr, pdtr and their
// complements; the log tails below the normal range come from the same pieces as the
// beta and gamma distributions' (stats/dist/beta.hpp, gamma.hpp).
//
// Those are the scalar definitions. Over arrays the methods run through chunk kernels
// (stats/engine.hpp): the pmfs near the mean of a wide distribution in float64 with their
// error terms (saddle::, below), the log tails with one vForce log per chunk, the geometric
// with its logs taken once per chunk, each handing what its bound does not cover to the
// scalar definition; and behind the per-thread memos of stats/discrete.hpp, so repeated
// values of k (a small support) are evaluated once.
#pragma once

#include "special/ops/distr.hpp"
#include "special/prefix.hpp"
#include "special/coeffs/cephes_gamma.hpp"
#include "stats/chunk.hpp"
#include "stats/discrete.hpp"
#include "stats/dist/beta.hpp"
#include "stats/dist/gamma.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <array>
#include <cmath>

namespace sd::stats {

// ---- the fast pmfs: the saddle-point form in float64 with its error terms -------------------
//
// The pmfs of this file are e^-E times a slowly varying factor (special/prefix.hpp; Loader,
// "Fast and accurate computation of binomial probabilities", 2000), E a sum of terms
//
//     a mu(t),  mu(t) = t - log(1 + t) = t^2 P(t),  P(t) = 1/2 - t R(t),  R(t) = 1/3 - t/4 + t^2/5 - ...
//
// with a t = D / a, D the distance of k from the mean in the parameter's own units (n p - k
// for the binomial). The precise definitions form E in double-double with a double-double
// log. Near the mean t is small, the series for R needs no log, and the rest of E is a
// few products and quotients whose rounding errors come out of fma exactly. So for
// |t| <= 1/8 the kernels below form X = S - E (S: Stirling's corrections, below 0.02) as
// a float64 with a correction, X = xh + xl, and the pmf is sqrt(r) e^xh (1 + xl).
//
// Error of X (u = 2^-53; every operation rounds to within u of its result):
//   * D = dh + dl: the products and sums are error-free (two_prod, two_sum), except the
//     sum of the low parts, off by at most 3u in absolute terms for arguments up to 2^53.
//     That moves E by 2 |D| W 3u with |D| W = |t1| P(t1) + |t2| P(t2) <= 0.14: 0.9u absolute.
//   * t = dh / a: 2u. R by Horner, 17 terms (the 18th is (1/8)^17 / 20 = 2^-55): 2u with
//     t's error and the truncation. x = t R: 5u, and |x| <= 0.044 against P >= 0.45, so P =
//     1/2 - x is off by 0.5u of itself; the subtraction itself is error-free (two_sum).
//   * P / a with the division's remainder (fma), the sum of the two quotients by two_sum,
//     the product with D^2 by two_prod: exact up to terms of u^2.
//   * S: three (or one) Stirling series of size <= 1/156 each to 4u: 0.08u absolute.
// Together: X is within 0.5u E + 1.1u of its value. The kernels then add ck::exp or
// log and their own factor's roundings, written at each.

namespace saddle {

using num::DD;
namespace dd = num::dd;

inline constexpr double kTMax = 0.125;      // the series' range, |t| <= 1/8
inline constexpr double kLargest = 0x1p53;  // counts and shapes up to here (their sums exact or to u)
inline constexpr double kStirlingMin = 13.0;  // Stirling's series from here (as special::gamma_star)

// R(t) = sum_{j >= 0} (-t)^j / (j + 3), |t| <= 1/8: 17 terms, the even and the odd ones as
// two Horner chains in t^2 (half the latency of one chain).
inline double series_r(double t) {
    constexpr double ce[9] = {1.0 / 3, 1.0 / 5, 1.0 / 7, 1.0 / 9, 1.0 / 11, 1.0 / 13, 1.0 / 15, 1.0 / 17, 1.0 / 19};
    constexpr double co[8] = {-1.0 / 4, -1.0 / 6, -1.0 / 8, -1.0 / 10, -1.0 / 12, -1.0 / 14, -1.0 / 16, -1.0 / 18};
    const double t2 = t * t;
    double e = ce[8], o = co[7];
    for (int j = 7; j >= 0; --j) e = e * t2 + ce[j];
    for (int j = 6; j >= 0; --j) o = o * t2 + co[j];
    return e + t * o;
}

// P(t) / a as wh + wl (a > 0).
inline DD p_over(double t, double a) {
    const DD p = dd::two_sum(0.5, -(t * series_r(t)));
    const double wh = p.hi / a;
    return {wh, (std::fma(-wh, a, p.hi) + p.lo) / a};
}

// log Gamma*(z) for z >= 13: Stirling's series 1/(12 z) - 1/(360 z^3) + ... (Cephes' fit,
// the polynomial of special::gamma_star).
inline double stirling(double z) {
    const double w = 1.0 / z;
    return special::horner(w * w, special::cephes::kLgamA) * w;
}

// X = xh + xl and the factor r of a pmf sqrt(r) e^X; ok: inside the fast path's range.
struct Parts {
    double xh, xl, r, e;  // e: E (for the bounds)
    bool ok;
};

// E = D^2 (P(t1) / a + P(t2) / b), t1 = D / a, t2 = -D / b, D = dh + dl; then X = s - E.
inline Parts finish(double dh, double dl, double a, double b, double s, double r, bool ok) {
    const double t1 = dh / a, t2 = -dh / b;
    const DD w1 = p_over(t1, a), w2 = p_over(t2, b);
    const DD w = dd::two_sum(w1.hi, w2.hi);
    const double wl = w.lo + (w1.lo + w2.lo);
    const DD sq = dd::two_prod(dh, dh);
    const double sl = sq.lo + 2.0 * dh * dl;
    const DD e = dd::two_prod(sq.hi, w.hi);
    const double el = e.lo + (sq.hi * wl + sl * w.hi);
    const DD x = dd::two_sum(s, -e.hi);
    return {x.hi, x.lo - el, r, e.hi, ok && std::fabs(t1) <= kTMax && std::fabs(t2) <= kTMax};
}

// Binomial(n, p) at k: 13 <= k <= n - 13, n <= 2^53, 0 < p < 1 (ok false otherwise; the
// arithmetic is safe for any input). D = n p - k; r = n / (2 pi k (n - k)).
inline Parts binomial(double k, double n, double p) {
    const double m = n - k;
    const bool ok = k >= kStirlingMin && m >= kStirlingMin && n <= kLargest && p > 0.0 && p < 1.0 && std::floor(k) == k;
    const DD h = dd::two_prod(n, p);
    const DD a = dd::two_sum(h.hi, -k);
    const DD d = dd::two_sum(a.hi, a.lo + h.lo);
    const double s = stirling(n) - stirling(k) - stirling(m);
    return finish(d.hi, d.lo, k, m, s, n / (2.0 * special::kPi * (k * m)), ok);
}

// Negative binomial(n, p) at k: n, k >= 13 and <= 2^53, 0 < p < 1. With a = n, b = k, c = n + k:
// D = p c - n = p k - q n (q = 1 - p as a double-double: exact); r = n / (2 pi c k).
inline Parts negative_binomial(double k, double n, double p) {
    const bool ok = k >= kStirlingMin && n >= kStirlingMin && k <= kLargest && n <= kLargest && p > 0.0 && p < 1.0 &&
                    std::floor(k) == k;
    const DD q = dd::two_sum(1.0, -p);
    const DD h1 = dd::two_prod(p, k), h2 = dd::two_prod(q.hi, n);
    const DD a = dd::two_sum(h1.hi, -h2.hi);
    const DD d = dd::two_sum(a.hi, a.lo + ((h1.lo - h2.lo) - q.lo * n));
    const double c = n + k;
    const double s = stirling(c) - stirling(n) - stirling(k);
    return finish(d.hi, d.lo, n, k, s, n / (2.0 * special::kPi * (c * k)), ok);
}

// Poisson(mu) at k >= 13: one term, D = mu - k (exact), E = D^2 P(D / k) / k, S = -log Gamma*(k),
// r = 1 / (2 pi k).
inline Parts poisson(double k, double mu) {
    const bool ok = k >= kStirlingMin && mu > 0.0 && std::isfinite(k) && std::isfinite(mu) && std::floor(k) == k;
    const DD d = dd::two_sum(mu, -k);
    const double t = d.hi / k;
    const DD w = p_over(t, k);
    const DD sq = dd::two_prod(d.hi, d.hi);
    const double sl = sq.lo + 2.0 * d.hi * d.lo;
    const DD e = dd::two_prod(sq.hi, w.hi);
    const double el = e.lo + (sq.hi * w.lo + sl * w.hi);
    const DD x = dd::two_sum(-stirling(k), -e.hi);
    return {x.hi, x.lo - el, 1.0 / (2.0 * special::kPi * k), e.hi, ok && std::fabs(t) <= kTMax};
}

// The kernels' common shape. parts(i) -> Parts of element i; precise(i) -> the scalar
// definition (called for the elements outside the fast path; it must be safe to call).
//
// pmf = sqrt(r) e^xh (1 + xl): X's 0.5u E + 1.1u, ck::exp (ck::kVfExp = 2.5u), the
// factor 1 + xl and the two products 3u, sqrt(r) (r_err / 2 + 1)u. With r_err <= 4.5 (the
// negative binomial's: c, c k, 2 pi (.), n / (.)) that is 0.5 E + 9.9 units, inside
// kFastBudget for E <= 12; a caller whose factor takes more operations passes a smaller e_max.
template <class PartsOf, class Precise>
void pmf_kernel(int n, double* out, PartsOf&& parts, Precise&& precise, double e_max = 12.0) {
    double xh[kChunkSize], xl[kChunkSize], r[kChunkSize];
    bool fast[kChunkSize];
    for (int i = 0; i < n; ++i) {
        const Parts q = parts(i);
        fast[i] = q.ok && q.e <= e_max;
        xh[i] = fast[i] ? q.xh : 0.0;
        xl[i] = q.xl;
        r[i] = q.r;
    }
    ck::exp(xh, xh, n);
    for (int i = 0; i < n; ++i) out[i] = std::sqrt(r[i]) * (xh[i] * (1.0 + xl[i]));
    ck::fallback(n, fast, out, precise);
}

// logpmf = X + log(r) / 2, both negative (r <= 1 / (26 pi): log(r) / 2 <= -2.2; X <= 0.007):
// r's 4.5u and vForce's log (ck::kVfLog = 4u of |log r|), halved; X's 0.5u E + 1.1u; the two
// sums 2u |v|. With |v| >= max(2.2, E, |log r| / 2): (2.25 + 1.1) / 2.2 + 4 + 0.5 + 2 = 8.1
// units of |v| at any E (the mixed bound of the log pmf is relative here: |v| > 1).
template <class PartsOf, class Precise>
void logpmf_kernel(int n, double* out, PartsOf&& parts, Precise&& precise) {
    double xh[kChunkSize], xl[kChunkSize], r[kChunkSize];
    bool fast[kChunkSize];
    for (int i = 0; i < n; ++i) {
        const Parts q = parts(i);
        fast[i] = q.ok;
        xh[i] = q.xh;
        xl[i] = q.xl;
        r[i] = fast[i] ? q.r : 1.0;
    }
    ck::log(r, r, n);
    for (int i = 0; i < n; ++i) out[i] = (xh[i] + 0.5 * r[i]) + xl[i];
    ck::fallback(n, fast, out, precise);
}

}  // namespace saddle

namespace binom_detail {

using num::DD;

// log C(n, k) p^k q^(n-k) for integers 0 <= k <= n, 0 < p < 1, q = 1 - p as a
// double-double (exact).
inline DD log_pmf(double k, double n, double p, DD q) {
    if (k == 0.0) return num::dd::log(q) * n;
    if (k == n) return num::log_dd(p) * n;
    return beta_detail::log_prefix(k, n - k, DD{p, 0.0}, q) + num::log_dd(n) - num::log_dd(k) - num::log_dd(n - k);
}

// True when x y (x, y doubles) is an integer: the product's two parts are exact.
inline bool product_is_int(double x, double y) {
    const DD m = num::dd::two_prod(x, y);
    return m.lo == 0.0 && std::floor(m.hi) == m.hi;
}

}  // namespace binom_detail

struct Binomial {
    static constexpr char name[] = "binomial";
    static constexpr std::array<const char*, 2> params{"n", "p"};

    static bool valid(double n, double p) { return n >= 0.0 && disc::is_int(n) && p >= 0.0 && p <= 1.0; }

    // The degenerate cases (n = 0, p = 0, p = 1): the point mass at `at`.
    static bool degenerate(double n, double p, double& at) {
        if (n == 0.0 || p == 0.0) {
            at = 0.0;
            return true;
        }
        if (p == 1.0) {
            at = n;
            return true;
        }
        return false;
    }

    // n - k is a float64 (always up to n = 2^53; past it only for some k, and the pmf at the
    // others is NaN: it would be the pmf of another k).
    static bool exact_rest(double k, double n) { return n <= disc::kMaxCount || num::dd::two_sum(n, -k).lo == 0.0; }

    static double logpmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0 || k > n) return -kInf;
        double at;
        if (degenerate(n, p, at)) return k == at ? 0.0 : -kInf;
        if (!exact_rest(k, n)) return kNaN;
        const num::DD l = binom_detail::log_pmf(k, n, p, num::dd::two_sum(1.0, -p));
        return l.hi + l.lo;
    }
    static double pmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0 || k > n) return 0.0;
        double at;
        if (degenerate(n, p, at)) return k == at ? 1.0 : 0.0;
        if (!exact_rest(k, n)) return kNaN;
        return num::exp_dd(binom_detail::log_pmf(k, n, p, num::dd::two_sum(1.0, -p)));
    }
    // P(X <= floor(x)) (upper: P(X > floor(x))); NaN inside the support of an n past
    // disc::kMaxCount (the incomplete beta's n - k, k + 1 and n + 1 are not all float64s there).
    static double tail(double x, double n, double p, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 1.0 : 0.0;
        if (k >= n) return upper ? 0.0 : 1.0;
        double at;
        if (degenerate(n, p, at)) return (k >= at) != upper ? 1.0 : 0.0;
        if (n > disc::kMaxCount) return kNaN;
        return upper ? special::Bdtrc::f(k, n, p) : special::Bdtr::f(k, n, p);
    }
    static double cdf(double x, double n, double p) { return tail(x, n, p, false); }
    static double ccdf(double x, double n, double p) { return tail(x, n, p, true); }
    // log P(X <= k) = log I_q(n - k, k + 1) (upper: log of its complement).
    static double log_tail(double x, double n, double p, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 0.0 : -kInf;
        if (k >= n) return upper ? -kInf : 0.0;
        double at;
        if (degenerate(n, p, at)) return (k >= at) != upper ? 0.0 : -kInf;
        if (n > disc::kMaxCount) return kNaN;
        return beta_detail::log_tail(n - k, k + 1.0, num::dd::two_sum(1.0, -p), num::DD{p, 0.0}, upper);
    }
    static double logcdf(double x, double n, double p) { return log_tail(x, n, p, false); }
    static double logccdf(double x, double n, double p) { return log_tail(x, n, p, true); }

    struct Fns {
        double n, p, lo = 0.0, hi, guess;
        double cdf(double k) const { return Binomial::cdf(k, n, p); }
        double ccdf(double k) const { return Binomial::ccdf(k, n, p); }
        double logcdf(double k) const { return Binomial::logcdf(k, n, p); }
        double logccdf(double k) const { return Binomial::logccdf(k, n, p); }
    };
    static Fns fns(double n, double p) { return {n, p, 0.0, n, std::floor(n * p)}; }
    static auto searcher(disc::Inv which, double t, disc::InvMemo* m, double n, double p) {
        auto f = disc::Memoized(fns(n, p), m);
        const double sd = std::sqrt(variance(n, p));
        if (sd >= disc::kWideSd) disc::normal_start(f, which, t, n * p, sd, (1.0 - 2.0 * p) / sd);
        return f;
    }
    SD_DISC_INVERSES(Binomial, 2)
    // the value methods behind their memos (stats/discrete.hpp)
    // pmf and logpmf: the saddle-point fast path (saddle::, above), else the scalar definition
    static double precise_at(const Chunk<2>& c, int i, double (*f)(double, double, double)) {
        return valid(c.p[0][i], c.p[1][i]) ? f(c.x[i], c.p[0][i], c.p[1][i]) : kNaN;
    }
    static void pmf_kernel(const Chunk<2>& c, double* out) {
        saddle::pmf_kernel(
            c.n, out, [&](int i) { return saddle::binomial(c.x[i], c.p[0][i], c.p[1][i]); },
            [&](int i) { return precise_at(c, i, &Binomial::pmf); }
        );
    }
    static void logpmf_kernel(const Chunk<2>& c, double* out) {
        saddle::logpmf_kernel(
            c.n, out, [&](int i) { return saddle::binomial(c.x[i], c.p[0][i], c.p[1][i]); },
            [&](int i) { return precise_at(c, i, &Binomial::logpmf); }
        );
    }
    SD_DISC_MEMO_KERNEL(Binomial, 2, pmf, false)
    SD_DISC_MEMO_KERNEL(Binomial, 2, logpmf, false)
    SD_DISC_MEMO_SCALAR(Binomial, 2, cdf, true)
    SD_DISC_MEMO_SCALAR(Binomial, 2, ccdf, true)
    // logcdf and logccdf from both tails of one incomplete beta (disc::log_tail_kernel):
    // P(X <= k) = I_q(n - k, k + 1).
    static void log_tail_kernel(const Chunk<2>& c, double* out, bool upper) {
        disc::log_tail_kernel(
            c.n, out, upper, beta_detail::kDeep,
            [&](int i, double& lower, double& higher) {
                const double k = std::floor(c.x[i]), n = c.p[0][i], p = c.p[1][i];
                if (!(k >= 0.0 && k < n && n <= disc::kMaxCount && p > 0.0 && p < 1.0)) return false;
                const special::ibeta_detail::IB r =
                    special::ibeta_detail::ibeta(n - k, k + 1.0, num::dd::two_sum(1.0, -p), num::DD{p, 0.0});
                lower = r.i, higher = r.ic;
                return true;
            },
            [&](int i) {
                return valid(c.p[0][i], c.p[1][i]) && c.x[i] == c.x[i] ? log_tail(c.x[i], c.p[0][i], c.p[1][i], upper)
                                                                       : kNaN;
            }
        );
    }
    static void logcdf_kernel(const Chunk<2>& c, double* out) { log_tail_kernel(c, out, false); }
    static void logccdf_kernel(const Chunk<2>& c, double* out) { log_tail_kernel(c, out, true); }
    SD_DISC_MEMO_KERNEL(Binomial, 2, logcdf, true)
    SD_DISC_MEMO_KERNEL(Binomial, 2, logccdf, true)

    static double mean(double n, double p) { return n * p; }
    static double variance(double n, double p) { return n * p * (1.0 - p); }
    // (q - p) / sqrt(n p q); a point mass has no skewness or kurtosis (NaN).
    static double skewness(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return kNaN;
        return (1.0 - 2.0 * p) / std::sqrt(n * p * (1.0 - p));
    }
    // (1 - 6 p q) / (n p q), the numerator in double-double (it crosses 0 at p = 0.211, 0.789).
    static double kurtosis(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return kNaN;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD pq = q * p;
        const num::DD num_ = num::DD{1.0, 0.0} - pq * 6.0;
        return (num_.hi + num_.lo) / (n * (pq.hi + pq.lo));
    }
    // floor((n + 1) p); two modes ((n + 1) p and one below) when (n + 1) p is an integer (T25).
    // The floor is of the exact product (its two parts): rounded to float64 a product within
    // an ULP below an integer is that integer (n = 9, p = 0.3: 2.9999999999999998 -> 3).
    // NaN beyond n = disc::kMaxCount (the cdfs' limit: n = 2^52 itself still has them).
    static double mode(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return at;
        if (n > disc::kMaxCount) return kNaN;
        const num::DD m = num::dd::two_prod(n + 1.0, p);
        const double f = std::floor(m.hi);
        if (f != m.hi) return f;
        return m.lo == 0.0 ? kNaN : (m.lo < 0.0 ? f - 1.0 : f);
    }
    static double entropy(double n, double p) {
        double at;
        if (degenerate(n, p, at)) return 0.0;
        if (!(variance(n, p) <= disc::kMaxEntropyVariance)) return kNaN;  // a sum past its bound
        if (n > disc::kMaxCount) return kNaN;                             // n - k is not exact
        const num::DD q = num::dd::two_sum(1.0, -p);
        return disc::entropy_sum(
            std::floor((n + 1.0) * p), 0.0, n, [&](double k) { return binom_detail::log_pmf(k, n, p, q); },
            // pmf(k + 1) / pmf(k) = (n - k) p / ((k + 1) q); one step down, its reciprocal at k - 1
            [&](double k, double dir) {
                return dir > 0.0 ? (num::DD{n - k, 0.0} * p) / (q * (k + 1.0))
                                 : (q * k) / (num::DD{n - k + 1.0, 0.0} * p);
            }
        );
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double n, double) { return n; }
};

// Bernoulli(p) = binomial(1, p).
struct Bernoulli {
    static constexpr char name[] = "bernoulli";
    static constexpr std::array<const char*, 1> params{"p"};

    static bool valid(double p) { return p >= 0.0 && p <= 1.0; }
    // Closed forms: 1 - p is exact for p >= 1/2 and relative to within an ULP below.
    static double pmf(double k, double p) { return k == 0.0 ? 1.0 - p : (k == 1.0 ? p : 0.0); }
    static double logpmf(double k, double p) { return k == 0.0 ? std::log1p(-p) : (k == 1.0 ? std::log(p) : -kInf); }
    static double cdf(double x, double p) { return x < 0.0 ? 0.0 : (x < 1.0 ? 1.0 - p : 1.0); }
    static double ccdf(double x, double p) { return x < 0.0 ? 1.0 : (x < 1.0 ? p : 0.0); }
    static double logcdf(double x, double p) { return x < 0.0 ? -kInf : (x < 1.0 ? std::log1p(-p) : 0.0); }
    static double logccdf(double x, double p) { return x < 0.0 ? 0.0 : (x < 1.0 ? std::log(p) : -kInf); }
    // The chunk kernels: the closed forms as branch-free loops, the two logs once per chunk for a
    // broadcast scalar p (the same libm calls: the same bits as the scalar definitions).
    template <class F>
    static void plain_chunk(const Chunk<1>& c, double* out, F&& f) {
        for (int i = 0; i < c.n; ++i) out[i] = f(c.x[i], c.p[0][i]);
    }
    static void pmf_chunk(const Chunk<1>& c, double* out) {
        plain_chunk(c, out, [](double k, double p) { return pmf(k, p); });
    }
    static void cdf_chunk(const Chunk<1>& c, double* out) {
        plain_chunk(c, out, [](double x, double p) { return cdf(x, p); });
    }
    static void ccdf_chunk(const Chunk<1>& c, double* out) {
        plain_chunk(c, out, [](double x, double p) { return ccdf(x, p); });
    }
    // a, b, d: the values at x < 0, x in [0, 1) and x >= 1 (logpmf: elsewhere, k = 0, k = 1)
    template <class F>
    static void log_chunk(const Chunk<1>& c, double* out, bool pmf_like, F&& scalar) {
        if (!c.same[0]) {
            for (int i = 0; i < c.n; ++i) out[i] = scalar(c.x[i], c.p[0][i]);
            return;
        }
        const double p = c.p[0][0];
        const double a = scalar(-1.0, p), b = scalar(0.0, p), d = scalar(1.0, p);
        for (int i = 0; i < c.n; ++i) {
            const double x = c.x[i];
            out[i] = pmf_like ? (x == 0.0 ? b : (x == 1.0 ? d : a)) : (x < 0.0 ? a : (x < 1.0 ? b : d));
        }
    }
    static void logpmf_chunk(const Chunk<1>& c, double* out) {
        log_chunk(c, out, true, [](double k, double p) { return logpmf(k, p); });
    }
    static void logcdf_chunk(const Chunk<1>& c, double* out) {
        log_chunk(c, out, false, [](double x, double p) { return logcdf(x, p); });
    }
    static void logccdf_chunk(const Chunk<1>& c, double* out) {
        log_chunk(c, out, false, [](double x, double p) { return logccdf(x, p); });
    }
    static double icdf(double t, double p) {
        return disc::settle<Bernoulli>(disc::Inv::icdf, t, Binomial::icdf(t, 1.0, p), nullptr, p);
    }
    static double iccdf(double t, double p) {
        return disc::settle<Bernoulli>(disc::Inv::iccdf, t, Binomial::iccdf(t, 1.0, p), nullptr, p);
    }
    static double ilogcdf(double t, double p) {
        return disc::settle<Bernoulli>(disc::Inv::ilogcdf, t, Binomial::ilogcdf(t, 1.0, p), nullptr, p);
    }
    static double ilogccdf(double t, double p) {
        return disc::settle<Bernoulli>(disc::Inv::ilogccdf, t, Binomial::ilogccdf(t, 1.0, p), nullptr, p);
    }
    static double mean(double p) { return p; }
    static double variance(double p) { return p * (1.0 - p); }
    static double skewness(double p) { return Binomial::skewness(1.0, p); }
    static double kurtosis(double p) { return Binomial::kurtosis(1.0, p); }
    static double median(double p) { return Binomial::icdf(0.5, 1.0, p); }
    static double mode(double p) { return Binomial::mode(1.0, p); }
    static double entropy(double p) {  // entr(p) + entr(q)
        if (p == 0.0 || p == 1.0) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD h = -(num::log_dd(p) * p) - num::dd::log(q) * q;
        return h.hi + h.lo;
    }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double) { return 1.0; }
};

namespace poisson_detail {

using num::DD;

// log(mu^k e^-mu / k!) for an integer k >= 1, mu > 0: for k < 13 directly, k! exact;
// from 13 the prefix's form -E - log(sqrt(2 pi k) Gamma*(k)), E = mu - k - k log(mu/k)
// (special/prefix.hpp) in double-double.
inline DD log_pmf(double k, double mu) {
    if (k < 13.0) {
        double f = 1.0;
        for (double j = 2.0; j <= k; j += 1.0) f *= j;
        return num::log_dd(mu) * k - mu - num::log_dd(f);
    }
    const double c = std::log(std::sqrt(2.0 * special::kPi * k) * special::gamma_star(k));
    return -special::gamma_exponent(k, mu) - c;
}

}  // namespace poisson_detail

struct Poisson {
    static constexpr char name[] = "poisson";
    static constexpr std::array<const char*, 1> params{"mu"};

    static bool valid(double mu) { return mu >= 0.0 && std::isfinite(mu); }

    static num::DD log_pmf_dd(double k, double mu) {
        if (k == 0.0) return num::DD{-mu, 0.0};
        const num::DD l = poisson_detail::log_pmf(k, mu);
        if (l.hi + l.lo == l.hi + l.lo) return l;
        // A k so large that the double-double exponent is inf - inf: Stirling's form in
        // float64 (-inf, or within a few ULP where it is still in range).
        return {k * (std::log(mu) - std::log(k) + 1.0) - mu - 0.5 * std::log(2.0 * special::kPi * k), 0.0};
    }
    static double logpmf(double k, double mu) {
        if (!disc::is_int(k) || k < 0.0) return -kInf;
        if (mu == 0.0) return k == 0.0 ? 0.0 : -kInf;
        const num::DD l = log_pmf_dd(k, mu);
        return l.hi + l.lo;
    }
    static double pmf(double k, double mu) {
        if (!disc::is_int(k) || k < 0.0) return 0.0;
        if (mu == 0.0) return k == 0.0 ? 1.0 : 0.0;
        return num::exp_dd(log_pmf_dd(k, mu));
    }
    static double cdf(double x, double mu) {
        const double k = std::floor(x);
        if (k < 0.0) return 0.0;
        if (mu == 0.0 || std::isinf(k)) return 1.0;
        const double v = special::Pdtr::f(k, mu);
        return k < 0x1p53 ? v : disc::checked_past_counts(v, pmf(k, mu));
    }
    static double ccdf(double x, double mu) {
        const double k = std::floor(x);
        if (k < 0.0) return 1.0;
        if (mu == 0.0 || std::isinf(k)) return 0.0;
        const double v = special::Pdtrc::f(k, mu);
        return k < 0x1p53 ? v : disc::checked_past_counts(v, pmf(k, mu));
    }
    // log P(X <= k) = log Q(k + 1, mu) (upper: log P(k + 1, mu)).
    static double log_tail(double x, double mu, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 0.0 : -kInf;
        if (mu == 0.0 || std::isinf(k)) return upper ? -kInf : 0.0;
        const double v = gamma_detail::log_tail(k + 1.0, num::DD{mu, 0.0}, !upper);
        // Past the range of the incomplete gamma's exponent (k beyond 2^53 mu): the upper
        // tail is its first term, pmf(k + 1) (1 + mu / (k + 2) + ...), to 2^-53.
        if (v != v && upper && k > 0x1p53 * mu) return logpmf(k + 1.0, mu);
        return k < 0x1p53 ? v : disc::checked_log_past_counts(v, pmf(k, mu));
    }
    static double logcdf(double x, double mu) { return log_tail(x, mu, false); }
    static double logccdf(double x, double mu) { return log_tail(x, mu, true); }

    struct Fns {
        double mu, lo = 0.0, hi = kInf, guess;
        double cdf(double k) const { return Poisson::cdf(k, mu); }
        double ccdf(double k) const { return Poisson::ccdf(k, mu); }
        double logcdf(double k) const { return Poisson::logcdf(k, mu); }
        double logccdf(double k) const { return Poisson::logccdf(k, mu); }
    };
    // (the family's support, [0, inf), also for the point mass mu = 0: docs/api/stats.md §2)
    static Fns fns(double mu) { return {mu, 0.0, kInf, std::floor(mu)}; }
    static auto searcher(disc::Inv which, double t, disc::InvMemo* m, double mu) {
        auto f = disc::Memoized(fns(mu), m);
        const double sd = std::sqrt(mu);
        if (sd >= disc::kWideSd) disc::normal_start(f, which, t, mu, sd, 1.0 / sd);
        return f;
    }
    SD_DISC_INVERSES(Poisson, 1)
    // the value methods behind their memos (stats/discrete.hpp)
    static double precise_at(const Chunk<1>& c, int i, double (*f)(double, double)) {
        return valid(c.p[0][i]) ? f(c.x[i], c.p[0][i]) : kNaN;
    }
    static void pmf_kernel(const Chunk<1>& c, double* out) {
        saddle::pmf_kernel(
            c.n, out, [&](int i) { return saddle::poisson(c.x[i], c.p[0][i]); },
            [&](int i) { return precise_at(c, i, &Poisson::pmf); }
        );
    }
    static void logpmf_kernel(const Chunk<1>& c, double* out) {
        saddle::logpmf_kernel(
            c.n, out, [&](int i) { return saddle::poisson(c.x[i], c.p[0][i]); },
            [&](int i) { return precise_at(c, i, &Poisson::logpmf); }
        );
    }
    SD_DISC_MEMO_KERNEL(Poisson, 1, pmf, false)
    SD_DISC_MEMO_KERNEL(Poisson, 1, logpmf, false)
    SD_DISC_MEMO_SCALAR(Poisson, 1, cdf, true)
    SD_DISC_MEMO_SCALAR(Poisson, 1, ccdf, true)
    // logcdf and logccdf from both tails of one incomplete gamma (disc::log_tail_kernel).
    static void log_tail_kernel(const Chunk<1>& c, double* out, bool upper) {
        disc::log_tail_kernel(
            c.n, out, upper, 0x1p-1000,
            [&](int i, double& lower, double& higher) {
                const double k = std::floor(c.x[i]), mu = c.p[0][i];
                if (!(k >= 0.0 && k < 0x1p53 && mu > 0.0 && std::isfinite(mu))) return false;
                // logcdf at k = 0, log Q(1, mu) = -mu: the scalar log_tail gives it to the last bit
                // (gamma_detail::finite_log_upper); the log of the rounded Q was an ULP off it
                if (!upper && k == 0.0) return false;
                const special::igam_detail::PQ r = special::igam_detail::igam(k + 1.0, mu);
                lower = r.q, higher = r.p;  // P(X <= k) = Q(k + 1, mu)
                return true;
            },
            [&](int i) { return valid(c.p[0][i]) && c.x[i] == c.x[i] ? log_tail(c.x[i], c.p[0][i], upper) : kNaN; }
        );
    }
    static void logcdf_kernel(const Chunk<1>& c, double* out) { log_tail_kernel(c, out, false); }
    static void logccdf_kernel(const Chunk<1>& c, double* out) { log_tail_kernel(c, out, true); }
    SD_DISC_MEMO_KERNEL(Poisson, 1, logcdf, true)
    SD_DISC_MEMO_KERNEL(Poisson, 1, logccdf, true)

    static double mean(double mu) { return mu; }
    static double variance(double mu) { return mu; }
    static double skewness(double mu) { return mu == 0.0 ? kNaN : 1.0 / std::sqrt(mu); }
    static double kurtosis(double mu) { return mu == 0.0 ? kNaN : 1.0 / mu; }
    // floor(mu); an integer mu >= 1 has two modes, mu - 1 and mu (T25).
    static double mode(double mu) {
        if (mu == 0.0) return 0.0;
        return std::floor(mu) == mu ? kNaN : std::floor(mu);
    }
    // The sum over the support, or from mu = 1e4 the asymptotic series
    // (1 + log 2 pi mu)/2 - 1/(12 mu) - 1/(24 mu^2) - 19/(360 mu^3) + ...
    static double entropy(double mu) {
        if (mu == 0.0) return 0.0;
        if (mu >= 1e4) {
            const double r = 1.0 / mu;
            const num::DD h = num::kNormalEntropy + num::log_dd(mu) * 0.5 +
                              r * (-1.0 / 12.0 + r * (-1.0 / 24.0 + r * (-19.0 / 360.0)));
            return h.hi + h.lo;
        }
        return disc::entropy_sum(
            std::floor(mu), 0.0, kInf, [&](double k) { return log_pmf_dd(k, mu); },
            // pmf(k + 1) / pmf(k) = mu / (k + 1)
            [&](double k, double dir) { return dir > 0.0 ? num::DD{mu, 0.0} / (k + 1.0) : num::DD{k, 0.0} / mu; }
        );
    }
    static double support_lo(double) { return 0.0; }
    static double support_hi(double) { return kInf; }  // (the family's: mu = 0 too)
};

struct Geometric {
    static constexpr char name[] = "geometric";
    static constexpr std::array<const char*, 1> params{"p"};

    static bool valid(double p) { return p > 0.0 && p <= 1.0; }

    // log q = log(1 - p), 1 - p exact as a double-double (per thread, for the last p).
    static num::DD log_q(double p) {
        thread_local num::Memo memo;
        return memo.get(p, 0.0, [&] { return num::dd::log(num::dd::two_sum(1.0, -p)); });
    }

    static num::DD log_pmf_dd(double k, double p) { return log_q(p) * (k - 1.0) + num::log_dd(p); }
    static double logpmf(double k, double p) {
        if (!disc::is_int(k) || k < 1.0) return -kInf;
        if (p == 1.0) return k == 1.0 ? 0.0 : -kInf;
        if (k >= disc::kFarCount && disc::overflows(log_q(p).hi, k - 1.0)) return -kInf;
        const num::DD l = log_pmf_dd(k, p);
        return l.hi + l.lo;
    }
    static double pmf(double k, double p) {
        if (!disc::is_int(k) || k < 1.0) return 0.0;
        if (p == 1.0) return k == 1.0 ? 1.0 : 0.0;
        if (k >= disc::kFarCount && disc::overflows(log_q(p).hi, k - 1.0)) return 0.0;
        return num::exp_dd(log_pmf_dd(k, p));
    }
    // q^k is below every float64, its log past the float64 range: the float64 product
    // k log q is -inf (an infinite k included), where the double-double one would be inf - inf.
    // The test is on a product the double-double one starts with.
    static bool beyond(double k, num::DD lq) { return !(k * lq.hi > -kInf); }
    // P(X > k) = q^k, P(X <= k) = 1 - q^k = -expm1(k log q).
    static double cdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (p == 1.0) return 1.0;
        const num::DD lq = log_q(p);
        if (beyond(k, lq)) return 1.0;
        const num::DD e = lq * k;
        return -std::expm1(e.hi) - std::exp(e.hi) * e.lo;
    }
    static double ccdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 1.0;
        if (p == 1.0) return 0.0;
        const num::DD lq = log_q(p);
        if (beyond(k, lq)) return 0.0;
        return num::exp_dd(lq * k);
    }
    static double logccdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return 0.0;
        if (p == 1.0) return -kInf;
        const num::DD lq = log_q(p);
        if (beyond(k, lq)) return -kInf;
        const num::DD e = lq * k;
        return e.hi + e.lo;
    }
    static double logcdf(double x, double p) {
        const double k = std::floor(x);
        if (k < 1.0) return -kInf;
        if (p == 1.0 || std::isinf(k)) return 0.0;
        const double c = ccdf(x, p);
        return c <= 0.5 ? std::log1p(-c) : std::log(cdf(x, p));
    }

    struct Fns {
        double p, lo = 1.0, hi, guess;
        double cdf(double k) const { return Geometric::cdf(k, p); }
        double ccdf(double k) const { return Geometric::ccdf(k, p); }
        double logcdf(double k) const { return Geometric::logcdf(k, p); }
        double logccdf(double k) const { return Geometric::logccdf(k, p); }
    };
    static Fns fns(double p) { return {p, 1.0, kInf, std::floor(1.0 / p)}; }  // (p = 1 too: the family's support)
    // The search starts at the closed form ceil(log P(X > k) / log q) (q^k <= the upper tail).
    static Fns at(double p, double log_upper) {
        return disc::with_guess(fns(p), std::ceil(log_upper / std::log1p(-p)));
    }
    static double icdf(double t, double p) {
        return disc::settle<Geometric>(disc::Inv::icdf, t, disc::icdf(t, at(p, std::log1p(-t))), nullptr, p);
    }
    static double iccdf(double t, double p) {
        return disc::settle<Geometric>(disc::Inv::iccdf, t, disc::iccdf(t, at(p, std::log(t))), nullptr, p);
    }
    static double ilogcdf(double t, double p) {
        return disc::settle<Geometric>(
            disc::Inv::ilogcdf, t, disc::ilogcdf(t, at(p, std::log(-std::expm1(t)))), nullptr, p
        );
    }
    static double ilogccdf(double t, double p) {
        return disc::settle<Geometric>(disc::Inv::ilogccdf, t, disc::ilogccdf(t, at(p, t)), nullptr, p);
    }

    // ---- chunk kernels: log q and log p once per chunk for a scalar p, ck::exp ----

    // The logs of element i: {log q, log p} (double-doubles), from `hoisted` for a scalar p.
    struct Logs {
        num::DD lq, lp;
    };
    static Logs logs(double p) { return {num::dd::log(num::dd::two_sum(1.0, -p)), num::log_dd(p)}; }
    // Elements the kernels leave to the scalar definitions: p outside (0, 1), k outside the
    // support (as given; a cdf's is floored by the caller) or from disc::kFarCount on (where
    // k log q may pass the float64 range), NaN.
    static bool plain(double k, double p) { return p > 0.0 && p < 1.0 && k >= 1.0 && k < disc::kFarCount; }
    template <class F>
    static double scalar_at(const Chunk<1>& c, int i, F&& f) {
        return valid(c.p[0][i]) && c.x[i] == c.x[i] ? f(c.x[i], c.p[0][i]) : kNaN;
    }
    // exp(e) is taken from vForce down to here; below, the scalar definition (whose results
    // run into the subnormals).
    static constexpr double kExpMin = -700.0;

    // pmf = exp((k - 1) log q + log p): the exponent in double-double as in the scalar
    // definition (exact to 2^-100 of itself), ck::exp 2.5u, the factor 1 + lo and the
    // product 2u: 4.5 units.
    static void pmf_kernel(const Chunk<1>& c, double* out) {
        double eh[kChunkSize], el[kChunkSize];
        bool fast[kChunkSize];
        const bool same = c.same[0];
        Logs l = logs(same && c.p[0][0] > 0.0 && c.p[0][0] < 1.0 ? c.p[0][0] : 0.5);
        for (int i = 0; i < c.n; ++i) {
            const double k = c.x[i], p = c.p[0][i];
            fast[i] = plain(k, p) && std::floor(k) == k;
            eh[i] = el[i] = 0.0;
            if (!fast[i]) continue;
            if (!same) l = logs(p);
            const num::DD e = l.lq * (k - 1.0) + l.lp;
            fast[i] = e.hi > kExpMin;
            if (fast[i]) eh[i] = e.hi, el[i] = e.lo;
        }
        ck::exp(eh, eh, c.n);
        for (int i = 0; i < c.n; ++i) out[i] = eh[i] * (1.0 + el[i]);
        ck::fallback(c.n, fast, out, [&](int i) { return scalar_at(c, i, &Geometric::pmf); });
    }
    // The scalar definition's own operations with the logs hoisted: the same bits.
    static void logpmf_kernel(const Chunk<1>& c, double* out) {
        const bool same = c.same[0];
        Logs l = logs(same && c.p[0][0] > 0.0 && c.p[0][0] < 1.0 ? c.p[0][0] : 0.5);
        for (int i = 0; i < c.n; ++i) {
            const double k = c.x[i], p = c.p[0][i];
            if (!(plain(k, p) && std::floor(k) == k)) {
                out[i] = scalar_at(c, i, &Geometric::logpmf);
                continue;
            }
            if (!same) l = logs(p);
            const num::DD e = l.lq * (k - 1.0) + l.lp;
            out[i] = e.hi + e.lo;
        }
    }
    // log P(X <= k) with c = q^k = exp(k log q) from vForce (c (1 + lo): 4.5u):
    //   c <= 1/2: log1p(-c), whose condition in c is c / ((1 - c) |log(1 - c)|) <= 1.45: 6.5u,
    //             and log1p's own 2u: 8.5 units;
    //   c > 1/2:  P = -expm1(hi) - c lo (expm1 2u, the correction's error second order, the
    //             difference 1u), then log P with |log P| > log 2: 3.1 / 0.69 + 2 = 6.5 units.
    static void logcdf_kernel(const Chunk<1>& c, double* out) {
        double eh[kChunkSize], el[kChunkSize], y[kChunkSize];
        bool fast[kChunkSize];
        const bool same = c.same[0];
        Logs l = logs(same && c.p[0][0] > 0.0 && c.p[0][0] < 1.0 ? c.p[0][0] : 0.5);
        for (int i = 0; i < c.n; ++i) {
            const double k = std::floor(c.x[i]), p = c.p[0][i];
            fast[i] = plain(k, p);
            eh[i] = el[i] = 0.0;
            if (!fast[i]) continue;
            if (!same) l = logs(p);
            const num::DD e = l.lq * k;
            fast[i] = e.hi > kExpMin;
            if (fast[i]) eh[i] = e.hi, el[i] = e.lo;
        }
        ck::exp(y, eh, c.n);
        for (int i = 0; i < c.n; ++i) {
            if (!fast[i]) {
                out[i] = scalar_at(c, i, &Geometric::logcdf);
                continue;
            }
            const double cc = y[i] * (1.0 + el[i]);
            out[i] = cc <= 0.5 ? std::log1p(-cc) : std::log(-std::expm1(eh[i]) - y[i] * el[i]);
        }
    }
    SD_DISC_MEMO_KERNEL(Geometric, 1, pmf, false)
    SD_DISC_MEMO_KERNEL(Geometric, 1, logpmf, false)
    SD_DISC_MEMO_SCALAR(Geometric, 1, cdf, true)
    SD_DISC_MEMO_KERNEL(Geometric, 1, logcdf, true)

    // icdf and iccdf. The search (disc::icdf) starts at g = ceil(r), r = log U / log q (U the
    // upper tail asked for: 1 - t or t), and then compares q^k with the target at g and
    // g - 1. Its comparisons are the exact ones up to the few units u of exp and log, which
    // in k is e / |log q| + e r; so when r is further than 64u (1 / |log q| + r) from both
    // g - 1 and g, the search can only return g, and the kernel returns it without the two
    // evaluations. Everything else (the ends t = 0 and 1, tiny or huge r, p = 1) is searched.
    // The g of the fast path is still checked one step below (disc::settle, D-13): a target that
    // is a rounded cdf(k) can be far from k in r when it is near 1.
    template <class LogUpper, class Scalar>
    static void inverse_kernel(const Chunk<1>& c, double* out, LogUpper&& log_upper, Scalar&& scalar, disc::Inv which) {
        const bool same = c.same[0];
        double lq = same ? std::log1p(-c.p[0][0]) : 0.0;
        disc::InvMemo* m = disc::inv_memo_for<Geometric>(c);  // settle's values at g - 1, for scalar p
        for (int i = 0; i < c.n; ++i) {
            const double t = c.x[i], p = c.p[0][i];
            if (t > 0.0 && t < 1.0 && p > 0.0 && p < 1.0) {
                if (!same) lq = std::log1p(-p);
                const double r = log_upper(t) / lq, g = std::ceil(r);
                const double margin = 64.0 * kUlp * (r - 1.0 / lq);
                if (g - r > margin && r - (g - 1.0) > margin && g < 0x1p52) {
                    out[i] = disc::settle<Geometric>(which, t, g, m, p);
                    continue;
                }
            }
            out[i] = scalar_at(c, i, scalar);
        }
    }
    static void icdf_chunk(const Chunk<1>& c, double* out) {
        inverse_kernel(c, out, [](double t) { return std::log1p(-t); }, &Geometric::icdf, disc::Inv::icdf);
    }
    static void iccdf_chunk(const Chunk<1>& c, double* out) {
        inverse_kernel(c, out, [](double t) { return std::log(t); }, &Geometric::iccdf, disc::Inv::iccdf);
    }

    static double mean(double p) { return 1.0 / p; }
    static double variance(double p) { return (1.0 - p) / (p * p); }
    static double skewness(double p) { return p == 1.0 ? kNaN : (2.0 - p) / std::sqrt(1.0 - p); }
    static double kurtosis(double p) { return p == 1.0 ? kNaN : 6.0 + p * p / (1.0 - p); }
    static double mode(double) { return 1.0; }
    static double entropy(double p) {  // (entr(p) + entr(q)) / p
        if (p == 1.0) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD h = (-(num::log_dd(p) * p) - num::dd::log(q) * q) / p;
        return h.hi + h.lo;
    }
    static double support_lo(double) { return 1.0; }
    static double support_hi(double) { return kInf; }  // (the family's: p = 1 too)
};

namespace nbinom_detail {

using num::DD;

// log of Gamma(k+n) / (Gamma(n) k!) p^n q^k for an integer k >= 0, n > 0, 0 < p < 1.
inline DD log_pmf(double k, double n, double p, DD q) {
    if (k == 0.0) return num::log_dd(p) * n;
    return beta_detail::log_prefix(n, k, DD{p, 0.0}, q) - num::log_dd(k);
}

}  // namespace nbinom_detail

struct NegativeBinomial {
    static constexpr char name[] = "negativebinomial";
    static constexpr std::array<const char*, 2> params{"n", "p"};

    static bool valid(double n, double p) { return n > 0.0 && std::isfinite(n) && p > 0.0 && p <= 1.0; }
    // n = 0 (no successes to wait for: no failures) is the point mass at 0, as Binomial's n = 0
    // and Poisson's mu = 0 (user decision 2026-11-19), and as special.nbdtr(k, 0, p) = 1: every
    // method is that of the point mass p = 1 (the family's support [0, inf) included).
    static bool at_limit(double n, double p) { return n == 0.0 && p > 0.0 && p <= 1.0; }
    template <class M>
    static double limit_in(double x, double, double) {
        return method_of<NegativeBinomial, M>(x, 1.0, 1.0);
    }
    template <class M>
    static double limit_par(double, double) {
        return method_of<NegativeBinomial, M>(1.0, 1.0);
    }

    // The smallest n with a pmf. The prefix's exponent holds log n to an ULP, so the pmf's
    // relative error grows like 5e-17 |log n| (1.1e-14 at n = 1e-100, measured against Arb),
    // and next to the subnormals the log of the upper tail (~ n |log p|) is lost: below this n
    // the pmf, the cdfs, the quantiles and the entropy are NaN.
    static constexpr double kMinN = 0x1p-128;

    // The log pmf; for a k so large that the prefix's double-double exponent is inf - inf (a
    // NaN), its float64 form for k > 2^60 n: k log q + (n - 1) log k + n log p - log Gamma(n)
    // (log Gamma(k + n) - log Gamma(k + 1) = (n - 1) log k to n / k), -inf or within a few ULP.
    static num::DD log_pmf_dd(double k, double n, double p) {
        if (n < kMinN) return {kNaN, 0.0};
        const num::DD l = nbinom_detail::log_pmf(k, n, p, num::dd::two_sum(1.0, -p));
        if (l.hi + l.lo == l.hi + l.lo || !(k > 0x1p60 * n)) return l;
        return {k * std::log1p(-p) + (n - 1.0) * std::log(k) + n * std::log(p) - std::lgamma(n), 0.0};
    }
    static double logpmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0) return -kInf;
        if (p == 1.0) return k == 0.0 ? 0.0 : -kInf;
        const num::DD l = log_pmf_dd(k, n, p);
        return l.hi + l.lo;
    }
    static double pmf(double k, double n, double p) {
        if (!disc::is_int(k) || k < 0.0) return 0.0;
        if (p == 1.0) return k == 0.0 ? 1.0 : 0.0;
        return num::exp_dd(log_pmf_dd(k, n, p));
    }
    static double cdf(double x, double n, double p) {
        const double k = std::floor(x);
        if (k < 0.0) return 0.0;
        if (p == 1.0 || std::isinf(k)) return 1.0;
        if (n < kMinN) return kNaN;
        const double v = special::Nbdtr::f(k, n, p);
        return k < 0x1p53 ? v : disc::checked_past_counts(v, pmf(k, n, p));
    }
    static double ccdf(double x, double n, double p) {
        const double k = std::floor(x);
        if (k < 0.0) return 1.0;
        if (p == 1.0 || std::isinf(k)) return 0.0;
        if (n < kMinN) return kNaN;
        const double v = special::Nbdtrc::f(k, n, p);
        return k < 0x1p53 ? v : disc::checked_past_counts(v, pmf(k, n, p));
    }
    // log P(X <= k) = log I_p(n, k + 1).
    static double log_tail(double x, double n, double p, bool upper) {
        const double k = std::floor(x);
        if (k < 0.0) return upper ? 0.0 : -kInf;
        if (p == 1.0 || std::isinf(k)) return upper ? -kInf : 0.0;
        if (n < kMinN) return kNaN;
        const double v = beta_detail::log_tail(n, k + 1.0, num::DD{p, 0.0}, num::dd::two_sum(1.0, -p), upper);
        // Past the range of the incomplete beta's exponent: the upper tail is pmf(k + 1) / p,
        // the later ratios (k + j + n) q / (k + j + 1) being q to n / k. That is used where the
        // n / (p k) it leaves out is below 2^-60 of the result (which is at least k p in size).
        if (v != v && upper && n <= 0x1p-60 * (p * k) * std::fmax(1.0, p * k))
            return logpmf(k + 1.0, n, p) - std::log(p);
        return k < 0x1p53 ? v : disc::checked_log_past_counts(v, pmf(k, n, p));
    }
    static double logcdf(double x, double n, double p) { return log_tail(x, n, p, false); }
    static double logccdf(double x, double n, double p) { return log_tail(x, n, p, true); }

    struct Fns {
        double n, p, lo = 0.0, hi, guess;
        double cdf(double k) const { return NegativeBinomial::cdf(k, n, p); }
        double ccdf(double k) const { return NegativeBinomial::ccdf(k, n, p); }
        double logcdf(double k) const { return NegativeBinomial::logcdf(k, n, p); }
        double logccdf(double k) const { return NegativeBinomial::logccdf(k, n, p); }
    };
    static Fns fns(double n, double p) { return {n, p, 0.0, kInf, std::floor(n * (1.0 - p) / p)}; }  // (p = 1 too)
    static auto searcher(disc::Inv which, double t, disc::InvMemo* m, double n, double p) {
        auto f = disc::Memoized(fns(n, p), m);
        const double sd = std::sqrt(variance(n, p));
        if (sd >= disc::kWideSd && p < 1.0) disc::normal_start(f, which, t, mean(n, p), sd, skewness(n, p));
        return f;
    }
    SD_DISC_INVERSES(NegativeBinomial, 2)
    // the value methods behind their memos (stats/discrete.hpp)
    static double precise_at(const Chunk<2>& c, int i, double (*f)(double, double, double)) {
        return valid(c.p[0][i], c.p[1][i]) ? f(c.x[i], c.p[0][i], c.p[1][i]) : kNaN;
    }
    static void pmf_kernel(const Chunk<2>& c, double* out) {
        saddle::pmf_kernel(
            c.n, out, [&](int i) { return saddle::negative_binomial(c.x[i], c.p[0][i], c.p[1][i]); },
            [&](int i) { return precise_at(c, i, &NegativeBinomial::pmf); }
        );
    }
    static void logpmf_kernel(const Chunk<2>& c, double* out) {
        saddle::logpmf_kernel(
            c.n, out, [&](int i) { return saddle::negative_binomial(c.x[i], c.p[0][i], c.p[1][i]); },
            [&](int i) { return precise_at(c, i, &NegativeBinomial::logpmf); }
        );
    }
    SD_DISC_MEMO_KERNEL(NegativeBinomial, 2, pmf, false)
    SD_DISC_MEMO_KERNEL(NegativeBinomial, 2, logpmf, false)
    SD_DISC_MEMO_SCALAR(NegativeBinomial, 2, cdf, true)
    SD_DISC_MEMO_SCALAR(NegativeBinomial, 2, ccdf, true)
    // logcdf and logccdf from both tails of one incomplete beta (disc::log_tail_kernel):
    // P(X <= k) = I_p(n, k + 1).
    static void log_tail_kernel(const Chunk<2>& c, double* out, bool upper) {
        disc::log_tail_kernel(
            c.n, out, upper, beta_detail::kDeep,
            [&](int i, double& lower, double& higher) {
                const double k = std::floor(c.x[i]), n = c.p[0][i], p = c.p[1][i];
                if (!(k >= 0.0 && k < 0x1p53 && n >= kMinN && n < kInf && p > 0.0 && p < 1.0)) return false;
                const special::ibeta_detail::IB r =
                    special::ibeta_detail::ibeta(n, k + 1.0, num::DD{p, 0.0}, num::dd::two_sum(1.0, -p));
                lower = r.i, higher = r.ic;
                return true;
            },
            [&](int i) {
                return valid(c.p[0][i], c.p[1][i]) && c.x[i] == c.x[i] ? log_tail(c.x[i], c.p[0][i], c.p[1][i], upper)
                                                                       : kNaN;
            }
        );
    }
    static void logcdf_kernel(const Chunk<2>& c, double* out) { log_tail_kernel(c, out, false); }
    static void logccdf_kernel(const Chunk<2>& c, double* out) { log_tail_kernel(c, out, true); }
    SD_DISC_MEMO_KERNEL(NegativeBinomial, 2, logcdf, true)
    SD_DISC_MEMO_KERNEL(NegativeBinomial, 2, logccdf, true)

    static double mean(double n, double p) { return n * (1.0 - p) / p; }
    // n q / p^2 and (2 - p) / sqrt(n q); factor by factor where p^2 or n q would leave the
    // normal range (a tiny p or n).
    static double variance(double n, double p) {
        const double pp = p * p;
        return pp >= 0x1p-1000 ? n * (1.0 - p) / pp : n * (1.0 - p) / p / p;
    }
    static double skewness(double n, double p) {
        if (p == 1.0) return kNaN;
        const double v = n * (1.0 - p);
        return v >= 0x1p-1000 ? (2.0 - p) / std::sqrt(v) : (2.0 - p) / (std::sqrt(n) * std::sqrt(1.0 - p));
    }
    static double kurtosis(double n, double p) { return p == 1.0 ? kNaN : 6.0 / n + p * p / (n * (1.0 - p)); }
    // floor((n - 1) q / p) for n > 1 (else 0); two modes when that is a positive integer (T25).
    // Both from the remainder r = (n - 1) q - m p at the nearest integer m, every product's
    // parts exact: r = 0 is the tie, and the floor is m - 1 for r < 0 (the quotient rounded to
    // float64 is m itself when it is within an ULP below m).
    static double mode(double n, double p) {
        if (n <= 1.0 || p == 1.0) return 0.0;
        const num::DD q = num::dd::two_sum(1.0, -p);
        const num::DD a = q * (n - 1.0);  // (n - 1) q (n - 1 exact)
        const double m = std::round((a.hi + a.lo) / p);
        if (!(m > 0.0)) return 0.0;
        const num::DD r = num::dd::two_prod(n - 1.0, q.hi) + num::dd::two_prod(n - 1.0, q.lo) - num::dd::two_prod(m, p);
        if (r.hi == 0.0 && r.lo == 0.0) return kNaN;
        return r.hi < 0.0 || (r.hi == 0.0 && r.lo < 0.0) ? m - 1.0 : m;
    }
    // The sum from the mode outwards. Its right tail falls like q^k, so the terms count down
    // to 2^-60 of the sum over about 42 / p steps past the bulk: below p = 2^-20, as above
    // the largest variance, the sum would pass its bound (stats/discrete.hpp) and is NaN.
    static double entropy(double n, double p) {
        if (p == 1.0) return 0.0;
        if (!(variance(n, p) <= disc::kMaxEntropyVariance) || p < 0x1p-20 || n < kMinN) return kNaN;
        const num::DD q = num::dd::two_sum(1.0, -p);
        return disc::entropy_sum(
            n > 1.0 ? std::floor((n - 1.0) * (1.0 - p) / p) : 0.0, 0.0, kInf,
            [&](double k) { return nbinom_detail::log_pmf(k, n, p, q); },
            // pmf(k + 1) / pmf(k) = (k + n) q / (k + 1)
            [&](double k, double dir) {
                return dir > 0.0 ? (num::dd::two_sum(k, n) * q) / (k + 1.0)
                                 : num::DD{k, 0.0} / (num::dd::two_sum(k - 1.0, n) * q);
            }
        );
    }
    static double support_lo(double, double) { return 0.0; }
    static double support_hi(double, double) { return kInf; }  // (the family's: p = 1 too)
};

}  // namespace sd::stats
