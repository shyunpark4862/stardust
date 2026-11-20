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
//                        parallel for many terms. Down to the subnormals: no cut to 0.
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
    std::vector<DD> v(mz, DD{0.0, 0.0}), w(mz);
    v[static_cast<std::size_t>(k - 1)] = {1.0, 0.0};
    std::int64_t ex = 0;  // the vector is v 2^ex
    const bool split = m >= 96;
    for (std::int64_t t = 1; t <= n; ++t) {
        // column j of v H: the rows from j - 1 on (above it H is zero, but in column 0)
        auto column = [&](std::int64_t j) {
            dd::Sum s;
            for (std::int64_t i = std::max<std::int64_t>(0, j - 1); i < m; ++i)
                s.add(v[static_cast<std::size_t>(i)] * at(i, j));
            w[static_cast<std::size_t>(j)] = s.value();
        };
        if (split) {
            parallel::parallel_for(m, [&](std::int64_t j) { column(j); });
        } else {
            for (std::int64_t j = 0; j < m; ++j) column(j);
        }
        double big = 0.0;
        for (std::int64_t j = 0; j < m; ++j) {
            v[static_cast<std::size_t>(j)] = w[static_cast<std::size_t>(j)] * static_cast<double>(t) / nn;
            big = std::max(big, std::fabs(v[static_cast<std::size_t>(j)].hi));
        }
        if (big > 0x1p300 || (big > 0.0 && big < 0x1p-300)) {
            const int e = std::ilogb(big);
            for (DD& x : v) x = dd::ldexp(x, -e);
            ex += e;
        }
    }
    const DD r = v[static_cast<std::size_t>(k - 1)];
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
