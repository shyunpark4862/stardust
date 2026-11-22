// The null distributions of the one-sample Kolmogorov-Smirnov test (docs/design/htest.md §3.3;
// H14: inside sd.htest, not public distributions):
//
//   smirnov_sf(n, d)     P(D_n^+ >= d), the one-sided statistic: Birnbaum and Tingey's finite sum
//                        d sum_{j <= n(1-d)} C(n, j) (1 - d - j/n)^(n-j) (d + j/n)^(j-1), positive
//                        terms. Each term's logarithm is a double-double (special/dd.hpp's log and
//                        loggamma), so a term is right to an ulp or two whatever n: in double the
//                        logarithms are of size n and a term would lose n ulps. The terms that
//                        matter are found first in double (those within e^-100 of the largest),
//                        then summed with their double-double logarithms in fixed chunks, in
//                        parallel for many terms. Down to the subnormals: no cut to 0. From n = 4096,
//                        where the terms turn over thousands of j, the sum is first taken as the
//                        integral of their analytic continuation (smirnov_integral below; ~100
//                        terms instead of n, the same value to 3e-16).
//   kolmogorov_sf(x)     Kolmogorov's limit Q(x) = 2 sum_{k >= 1} (-1)^(k-1) e^(-2 k^2 x^2) for x >= 1
//                        (x^2 split as hi + lo, so e^(-2x^2) is right to an ulp in the deep tail),
//                        1 - sqrt(2 pi)/x sum_{k >= 1} e^(-(2k-1)^2 pi^2 / (8 x^2)) below.
//   kstwo_sf(n, d)       P(D_n >= d), the two-sided statistic (Simard and L'Ecuyer's selection, as
//                        scipy's kstwo, made exact where scipy is not):
//                          - Ruben and Gambino's closed forms (n d <= 1, n d >= n - 1), and
//                            2 smirnov_sf for d >= 1/2 (both tails cannot be crossed): exact;
//                          - n d^2 >= 6.5: 2 smirnov_sf, which leaves out P(D^+ >= d, D^- >= d),
//                            below 1e-17 of the result there (about e^(-6 n d^2), measured against
//                            Arb for n = 20 to 1000);
//                          - else 1 - P(D_n < d) with P(D_n < d) from Durbin's matrix (Marsaglia,
//                            Tsang and Wang) in double-double: n steps of a row vector times the
//                            m x m matrix (m = 2 ceil(n d) - 1), n m^2 / 2 multiply-adds, within
//                            kDmtwWork; 1 - P keeps 1e-14 of a p-value down to 1e-16;
//                          - beyond that work (n above about 1000): Pelz and Good's expansion for
//                            n d^2 < 3 and 2 smirnov_sf above, an approximation (1e-6 to 1e-9
//                            relative for n = 1000 to 3000, better as n grows; the loose region of
//                            docs/design/htest.md §4).
//
// scipy's kstwo takes 1 - cdf from Pomeranz's recursion in double (8e-12 relative at n = 100,
// n d^2 = 4), 2 smirnov from n d^2 = 2.2 for n > 140 (1.6e-6) and 0 from n d^2 = 370.
#pragma once

#include "parallel/pool.hpp"
#include "special/dd.hpp"
#include "special/ops/gamma.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::htest::ks {

namespace dd = special::dd;
using dd::DD;

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTailLam2 = 6.5;  // n d^2 from which 2 smirnov_sf is the two-sided tail
inline constexpr double kPelzGoodLam2 = 3.0;  // beyond the matrix's work: Pelz-Good below, 2 smirnov above
inline constexpr double kDmtwWork = 4e7;  // multiply-adds of the double-double matrix (~0.1 s on one thread)

// e^(hi + lo), lo below an ulp of hi.
inline double exp_dd(DD x) { return std::exp(x.hi) * (1.0 + x.lo); }

// ---- the one-sided tail ----------------------------------------------------------------------

namespace detail {

// log of one Birnbaum-Tingey term with the leading d folded in: log C(n, j) + (n - j) log a_j +
// (j - 1) log b_j + log d, a_j = (c - j)/n, b_j = (nd + j)/n, c = n (1 - d); -inf for a_j = 0.
inline DD bt_log(double n, double j, DD c, DD nd, DD lgn, DD logd) {
    const DD a = (c - j) / n;
    if (!(a.hi > 0.0)) return {-std::numeric_limits<double>::infinity(), 0.0};
    const DD lc = lgn - dd::loggamma(DD{j + 1.0, 0.0}) - dd::loggamma(DD{n - j + 1.0, 0.0});
    DD l = lc + dd::log_pos(a) * (n - j);
    if (j == 0.0) return l;  // (j - 1) log d + log d = 0
    const DD b = (nd + j) / n;
    return l + dd::log_pos(b) * (j - 1.0) + logd;
}

}  // namespace detail

// The term's logarithm at a real x (the analytic continuation: log C(n, x) by loggamma), and its
// derivative in double (to find the peak).
inline DD bt_log_x(double x, double n, DD c, DD nd, DD lgn, DD logd) {
    const DD X{x, 0.0}, N{n, 0.0};
    const DD a = (c - X) / n, b = (nd + X) / n;
    const DD lc = lgn - dd::loggamma(X + 1.0) - dd::loggamma(N - X + 1.0);
    return lc + dd::log_pos(a) * (N - X) + dd::log_pos(b) * (X - 1.0) + logd;
}
inline double bt_dlog_x(double x, double n, double c, double nd) {
    using special::Digamma;
    return Digamma::f(n - x + 1.0) - Digamma::f(x + 1.0) - std::log((c - x) / n) - (n - x) / (c - x) +
           std::log((nd + x) / n) + (x - 1.0) / (nd + x);
}

// The Birnbaum-Tingey sum of a large n as an integral: the terms are the values at the integers
// of an analytic function T(x) (log C(n, x) by loggamma) that turns over a width sigma of
// thousands, and where T and its derivatives are negligible at both ends (x = 0, x = J), the
// Euler-Maclaurin formula leaves sum_j T(j) = int_0^J T(x) dx up to an exponentially small
// remainder (Poisson summation: about e^(-2 pi^2 sigma^2) for a bell of width sigma). The
// integral is the trapezoid rule on a lattice from the peak (found by bisection on T'/T),
// halved until two levels agree to 2^-44 (the rule's error falls as e^(-const / h)), each level
// walked out until a term is below 2^-70 of the largest; a walk that leaves [0, J] while its last
// term is not that small leaves the level unresolved. false where it does not apply: the peak
// at an end, sigma < 30, T(0) not negligible, or no agreement within 4096 terms (then the sum).
// Measured against the sum for n = 3e3 to 1e6 and n d^2 from 0.1 to 400: within 3e-16 relative,
// in ~100 terms (n d^2 >= 1) to ~5000 (n d^2 = 0.09), against n terms for the sum.
inline bool smirnov_integral(std::int64_t ni, double d, DD c, DD nd, double J, double& out) {
    const double n = static_cast<double>(ni);
    double lo = 0.5, hi = J - 0.5;
    if (!(hi > lo) || !(bt_dlog_x(lo, n, c.hi, nd.hi) > 0.0) || !(bt_dlog_x(hi, n, c.hi, nd.hi) < 0.0)) return false;
    for (int it = 0; it < 200 && hi - lo > 1e-6 * std::max(1.0, lo); ++it) {
        const double m = 0.5 * (lo + hi);
        (bt_dlog_x(m, n, c.hi, nd.hi) > 0.0 ? lo : hi) = m;
    }
    const double xs = 0.5 * (lo + hi), h = std::max(1.0, 1e-4 * xs);
    const double d2 = (bt_dlog_x(xs + h, n, c.hi, nd.hi) - bt_dlog_x(xs - h, n, c.hi, nd.hi)) / (2.0 * h);
    if (!(d2 < 0.0)) return false;
    const double sig = 1.0 / std::sqrt(-d2);
    if (!(sig >= 30.0)) return false;
    const DD lgn = dd::loggamma(DD{n + 1.0, 0.0}), logd = dd::log_pos(d);
    const DD Lpk = bt_log_x(xs, n, c, nd, lgn, logd);
    auto term = [&](double x) {
        const DD l = bt_log_x(x, n, c, nd, lgn, logd) - Lpk;
        return l.hi < -745.0 ? 0.0 : exp_dd(l);
    };
    if (!(term(0.0) <= 0x1p-70)) return false;  // the first term counts: the sum, not the integral
    const double H0 = 0.25 * sig;
    dd::Sum sum;
    double mx = 0.0, prev = 0.0;
    int count = 0;
    for (int level = 0; level < 12 && count < 4096; ++level) {
        const double step = std::ldexp(H0, -level), stride = level ? 2.0 * step : step;
        const double first = level ? xs + step : xs;
        bool resolved = true;
        for (int side = 0; side < 2; ++side) {
            double last = std::numeric_limits<double>::infinity();
            for (int j = 0;; ++j) {
                const double x = side ? first - (j + 1) * stride : first + j * stride;
                if (x < 0.0 || x > J) {
                    resolved = resolved && last <= 0x1p-70 * mx;
                    break;
                }
                const double o = term(x);
                ++count;
                sum.add(o);
                mx = std::max(mx, o);
                const bool outside = side ? x < xs - sig : x > xs + sig;
                if (outside && o <= 0x1p-70 * mx && o <= last) break;
                last = o;
            }
        }
        const double T = sum.total() * step;
        if (level > 0 && resolved && std::fabs(T - prev) <= 0x1p-44 * T) {
            out = exp_dd(Lpk + dd::log(DD{T, 0.0}));
            return true;
        }
        prev = T;
    }
    return false;
}
inline constexpr std::int64_t kSmirnovIntegralN = 4096;  // from this n the integral is tried first

inline double smirnov_sf(std::int64_t n, double d) {
    if (n <= 0 || d != d) return kNaN;
    if (d <= 0.0) return 1.0;
    if (d >= 1.0) return 0.0;
    if (n == 1) return 1.0 - d;
    const double nn = static_cast<double>(n);
    const DD c = dd::two_sum(1.0, -d) * nn;  // n (1 - d)
    double top = std::floor(c.hi);
    if (top == c.hi && c.lo < 0.0) top -= 1.0;
    const auto J = static_cast<std::int64_t>(top);
    const DD nd = dd::two_prod(nn, d);
    if (n >= kSmirnovIntegralN) {
        double v;
        if (smirnov_integral(n, d, c, nd, top, v)) return v;
    }
    // screening in double: log C(n, j) by its ratios, the logarithms of a and b in double
    std::vector<double> L(static_cast<std::size_t>(J + 1));
    const double ld = std::log(d), cn = c.hi;
    double lc = 0.0, lmax = -std::numeric_limits<double>::infinity();
    for (std::int64_t j = 0; j <= J; ++j) {
        const double jd = static_cast<double>(j);
        if (j > 0) lc += std::log((nn - jd + 1.0) / jd);
        const double a = (cn - jd) / nn, b = (nd.hi + jd) / nn;
        double v = a > 0.0 ? lc + (nn - jd) * std::log(a) : -std::numeric_limits<double>::infinity();
        if (j > 0) v += (jd - 1.0) * std::log(b) + ld;
        L[static_cast<std::size_t>(j)] = v;
        lmax = std::max(lmax, v);
    }
    if (!(lmax > -std::numeric_limits<double>::infinity())) return 0.0;
    std::int64_t j0 = J, j1 = 0;
    for (std::int64_t j = 0; j <= J; ++j)
        if (L[static_cast<std::size_t>(j)] >= lmax - 100.0) j0 = std::min(j0, j), j1 = std::max(j1, j);
    // the kept terms with double-double logarithms, e^(L - lmax), in fixed chunks added in order
    const DD lgn = dd::loggamma(DD{nn + 1.0, 0.0}), logd = dd::log_pos(d);
    constexpr std::int64_t kChunk = 2048;
    const std::int64_t count = j1 - j0 + 1, chunks = (count + kChunk - 1) / kChunk;
    std::vector<DD> part(static_cast<std::size_t>(chunks));
    auto body = [&](std::int64_t ch) {
        dd::Sum s;
        const std::int64_t a = j0 + ch * kChunk, b = std::min(j1 + 1, a + kChunk);
        for (std::int64_t j = a; j < b; ++j) {
            const DD l = detail::bt_log(nn, static_cast<double>(j), c, nd, lgn, logd);
            if (l.hi == -std::numeric_limits<double>::infinity()) continue;
            s.add(exp_dd(l - lmax));
        }
        part[static_cast<std::size_t>(ch)] = s.value();
    };
    if (count >= (1 << 14)) {
        parallel::parallel_for(chunks, body);
    } else {
        for (std::int64_t ch = 0; ch < chunks; ++ch) body(ch);
    }
    dd::Sum s;
    for (const DD& p : part) s.add(p);
    const DD tot = s.value();
    if (!(tot.hi > 0.0)) return 0.0;
    return exp_dd(DD{lmax, 0.0} + dd::log(tot));
}

// ---- Kolmogorov's limit ------------------------------------------------------------------------

inline double kolmogorov_sf(double x) {
    if (x != x) return kNaN;
    if (x <= 0.0) return 1.0;
    if (x < 1.0) {
        // 1 - K(x), K(x) = sqrt(2 pi)/x sum e^(-(2k-1)^2 w), w = pi^2/(8 x^2)
        const double w = kPi * kPi / (8.0 * x * x);
        if (w > 745.0) return 1.0;
        double s = 0.0;
        for (int k = 1; k < 100; ++k) {
            const double m = 2.0 * k - 1.0;
            const double t = std::exp(-m * m * w);
            s += t;
            if (t <= 0x1p-60 * s) break;
        }
        return 1.0 - std::sqrt(2.0 * kPi) / x * s;
    }
    // 2 sum (-1)^(k-1) e^(-2 k^2 x^2) with x^2 = h + l: e^(-2k^2 h) (1 - 2 k^2 l)
    const DD x2 = dd::two_prod(x, x);
    double s = 0.0;
    for (int k = 1; k < 100; ++k) {
        const double kk = 2.0 * k * k;
        const double t = std::exp(-kk * x2.hi) * (1.0 - kk * x2.lo);
        s += (k & 1) ? t : -t;
        if (t <= 0x1p-60 * std::fabs(s)) break;
    }
    return 2.0 * s;
}

// ---- the two-sided distribution ----------------------------------------------------------------

namespace detail {

// P(D_n < d) by Durbin's matrix (Marsaglia, Tsang and Wang 2003) in double-double: k = ceil(n d),
// h = k - n d, the m x m matrix H (m = 2k - 1) and n!/n^n (H^n)_{kk}, the latter as e_k^T H^n
// one row vector step at a time (each step also times t/n, so n!/n^n comes along), with the
// binary exponent kept apart. The work is n m^2 / 2 multiply-adds.
inline DD dmtw_cdf(std::int64_t n, double d) {
    const double nn = static_cast<double>(n);
    const DD nd = dd::two_prod(nn, d);
    double kf = std::ceil(nd.hi);
    if (kf == nd.hi && nd.lo > 0.0) kf += 1.0;
    const auto k = static_cast<std::int64_t>(kf);
    const DD h = DD{kf, 0.0} - nd;  // in [0, 1)
    const std::int64_t m = 2 * k - 1;
    std::vector<DD> inv(static_cast<std::size_t>(m + 2));  // 1/i!
    inv[0] = {1.0, 0.0};
    for (std::int64_t i = 1; i <= m + 1; ++i)
        inv[static_cast<std::size_t>(i)] = inv[static_cast<std::size_t>(i - 1)] / static_cast<double>(i);
    std::vector<DD> hp(static_cast<std::size_t>(m + 1));  // h^i
    hp[0] = {1.0, 0.0};
    for (std::int64_t i = 1; i <= m; ++i) hp[static_cast<std::size_t>(i)] = hp[static_cast<std::size_t>(i - 1)] * h;
    const auto mz = static_cast<std::size_t>(m);
    std::vector<DD> H(mz * mz, DD{0.0, 0.0});
    auto at = [&](std::int64_t i, std::int64_t j) -> DD& {
        return H[static_cast<std::size_t>(i) * mz + static_cast<std::size_t>(j)];
    };
    for (std::int64_t i = 0; i < m; ++i)
        for (std::int64_t j = 0; j < m; ++j)
            if (i - j + 1 >= 0) at(i, j) = inv[static_cast<std::size_t>(i - j + 1)];
    for (std::int64_t i = 0; i < m; ++i) {
        at(i, 0) = at(i, 0) - hp[static_cast<std::size_t>(i + 1)] * inv[static_cast<std::size_t>(i + 1)];
        at(m - 1, i) = at(m - 1, i) - hp[static_cast<std::size_t>(m - i)] * inv[static_cast<std::size_t>(m - i)];
    }
    const DD g = h * 2.0 - 1.0;
    if (g.hi > 0.0) {
        DD gp{1.0, 0.0};
        for (std::int64_t i = 0; i < m; ++i) gp = gp * g;
        at(m - 1, 0) = at(m - 1, 0) + gp * inv[mz];
    }
    // the matrix's high and low parts apart, the vector's too: v H as rows of axpys into the
    // columns' sums (Ogita, Rump and Oishi's Dot2: the product's error by fma, the sum's by
    // two_sum, both into the low sum), each column's terms still added in row order. Independent
    // columns in the inner loop vectorize; the old column-wise dd::Sum chained every term.
    std::vector<double> Hh(mz * mz), Hl(mz * mz);
    for (std::size_t i = 0; i < mz * mz; ++i) Hh[i] = H[i].hi, Hl[i] = H[i].lo;
    std::vector<double> vh(mz, 0.0), vl(mz, 0.0), sh(mz), sl(mz);
    vh[static_cast<std::size_t>(k - 1)] = 1.0;
    std::int64_t ex = 0;  // the vector is v 2^ex
    // columns [j0, j1) of v H into (sh, sl): the rows from j0 - 1 on (above the subdiagonal H is
    // zero, but in column 0)
    auto columns = [&](std::size_t j0, std::size_t j1) {
        for (std::size_t j = j0; j < j1; ++j) sh[j] = 0.0, sl[j] = 0.0;
        for (std::size_t i = j0 == 0 ? 0 : j0 - 1; i < mz; ++i) {
            const double a = vh[i], b = vl[i];
            const double* __restrict hh = Hh.data() + i * mz;
            const double* __restrict hl = Hl.data() + i * mz;
            double* __restrict s = sh.data();
            double* __restrict c = sl.data();
            const std::size_t je = std::min(j1, i + 2);
            for (std::size_t j = j0; j < je; ++j) {
                const double p = a * hh[j];
                double e = std::fma(a, hh[j], -p);
                e = std::fma(a, hl[j], e);
                e = std::fma(b, hh[j], e);
                const double x = s[j] + p, bb = x - s[j];
                c[j] += ((s[j] - (x - bb)) + (p - bb)) + e;
                s[j] = x;
            }
        }
    };
    // many columns: fixed blocks of 32 on the pool (each column's sum is the same whatever the
    // thread count)
    constexpr std::size_t kCols = 32;
    const std::size_t blocks = (mz + kCols - 1) / kCols;
    const bool split = m >= 96;
    for (std::int64_t t = 1; t <= n; ++t) {
        if (split) {
            parallel::parallel_for(static_cast<std::int64_t>(blocks), [&](std::int64_t b) {
                const auto j0 = static_cast<std::size_t>(b) * kCols;
                columns(j0, std::min(mz, j0 + kCols));
            });
        } else {
            columns(0, mz);
        }
        double big = 0.0;
        for (std::size_t j = 0; j < mz; ++j) {
            const DD r = dd::quick_two_sum(sh[j], sl[j]) * static_cast<double>(t) / nn;
            vh[j] = r.hi, vl[j] = r.lo;
            big = std::max(big, std::fabs(r.hi));
        }
        if (big > 0x1p300 || (big > 0.0 && big < 0x1p-300)) {
            const int e = std::ilogb(big);
            for (std::size_t j = 0; j < mz; ++j) vh[j] = std::ldexp(vh[j], -e), vl[j] = std::ldexp(vl[j], -e);
            ex += e;
        }
    }
    const DD r{vh[static_cast<std::size_t>(k - 1)], vl[static_cast<std::size_t>(k - 1)]};
    return dd::ldexp(r, static_cast<int>(std::clamp<std::int64_t>(ex, -3000, 3000)));
}

// The work of dmtw_cdf in multiply-adds.
inline double dmtw_work(std::int64_t n, double d) {
    const double k = std::ceil(static_cast<double>(n) * d) + 1.0;
    const double m = 2.0 * k - 1.0;
    return static_cast<double>(n) * m * m / 2.0;
}

// Pelz and Good's expansion of P(D_n <= d) (scipy's _kolmogn_PelzGood, its terms in the same
// order), as the upper tail.
inline double pelz_good_sf(std::int64_t n, double x) {
    if (x <= 0.0) return 1.0;
    if (x >= 1.0) return 0.0;
    const double nn = static_cast<double>(n);
    const double pi2 = kPi * kPi, pi4 = pi2 * pi2, pi6 = pi4 * pi2;
    const double sqrt2pi = std::sqrt(2.0 * kPi), sqrt3 = std::sqrt(3.0);
    const double z = std::sqrt(nn) * x;
    const double z2 = z * z, z3 = z2 * z, z4 = z2 * z2, z6 = z4 * z2;
    const double qlog = -pi2 / 8.0 / z2;
    if (qlog < -708.0) return 1.0;
    double q = std::exp(qlog);
    const double k1a = -z2, k1b = pi2 / 4.0;
    const double k2a = 6.0 * z6 + 2.0 * z4, k2b = (2.0 * z4 - 5.0 * z2) * pi2 / 4.0,
                 k2c = pi4 * (1.0 - 2.0 * z2) / 16.0;
    const double k3d = pi6 * (5.0 - 30.0 * z2) / 64.0, k3c = pi4 * (-60.0 * z2 + 212.0 * z4) / 16.0;
    const double k3b = pi2 * (135.0 * z4 - 96.0 * z6) / 4.0, k3a = -30.0 * z6 - 90.0 * z4 * z4;
    double K[4] = {0.0, 0.0, 0.0, 0.0};
    const auto maxk = static_cast<std::int64_t>(std::ceil(16.0 * z / kPi));
    for (std::int64_t k = maxk; k >= 1; --k) {
        const double m = 2.0 * static_cast<double>(k) - 1.0, m2 = m * m, m4 = m2 * m2, m6 = m4 * m2;
        const double qp = std::pow(q, 8.0 * static_cast<double>(k));
        const double c[4] = {1.0, k1a + k1b * m2, k2a + k2b * m2 + k2c * m4, k3a + k3b * m2 + k3c * m4 + k3d * m6};
        for (int i = 0; i < 4; ++i) K[i] = K[i] * qp + c[i];
    }
    const double den[4] = {z, 6.0 * z4, 72.0 * z4 * z3, 6480.0 * z4 * z6};
    for (int i = 0; i < 4; ++i) K[i] = K[i] * q * sqrt2pi / den[i];
    q = std::exp(-pi2 / 2.0 / z2);
    double e2 = 0.0, e3 = 0.0;
    for (std::int64_t k = maxk; k >= 1; --k) {
        const double kd = static_cast<double>(k), ks = kd * kd, qk = std::pow(q, ks);
        e2 += ks * qk;
        e3 += (sqrt3 * z + kPi * kd) * (sqrt3 * z - kPi * kd) * ks * qk;
    }
    K[2] += e2 * pi2 * sqrt2pi / (-36.0 * z3);
    K[3] += e3 * pi2 * sqrt2pi / (216.0 * z6);
    for (int i = 0; i < 4; ++i) K[i] /= std::pow(nn, i / 2.0);
    // the upper tail: 1 - K0 - K1 - K2 - K3
    return (1.0 - K[0]) - K[1] - K[2] - K[3];
}

}  // namespace detail

// P(D_n >= d), the two-sided one-sample statistic.
inline double kstwo_sf(std::int64_t n, double d) {
    if (n <= 0 || d != d) return kNaN;
    if (d >= 1.0) return 0.0;
    if (d <= 0.0) return 1.0;
    if (n == 1) return d <= 0.5 ? 1.0 : 2.0 * (1.0 - d);
    const double nn = static_cast<double>(n);
    const double t = nn * d;
    if (t <= 0.5) return 1.0;
    if (t <= 1.0) {
        // P(D_n < d) = n!/n^n (2 n d - 1)^n (Ruben and Gambino)
        const DD g = dd::two_prod(2.0 * nn, d) - 1.0;
        const double lp = std::lgamma(nn + 1.0) - nn * std::log(nn) + nn * std::log(g.hi + g.lo);
        return 1.0 - std::exp(lp);
    }
    if (t >= nn - 1.0) return 2.0 * std::pow(1.0 - d, nn);
    if (d >= 0.5) return 2.0 * smirnov_sf(n, d);
    const double lam2 = t * d;
    if (lam2 >= kTailLam2) return 2.0 * smirnov_sf(n, d);
    if (detail::dmtw_work(n, d) <= kDmtwWork) {
        const DD cdf = detail::dmtw_cdf(n, d);
        const DD sf = DD{1.0, 0.0} - cdf;
        return std::clamp(sf.hi + sf.lo, 0.0, 1.0);
    }
    if (lam2 >= kPelzGoodLam2) return 2.0 * smirnov_sf(n, d);
    return std::clamp(detail::pelz_good_sf(n, d), 0.0, 1.0);
}

}  // namespace sd::htest::ks
