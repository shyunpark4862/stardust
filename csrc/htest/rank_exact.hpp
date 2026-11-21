// The exact null distributions of the rank tests (milestone H3, docs/design/htest.md §3.4): counts
// of the arrangements whose statistic is at most (or at least) the observed one, so the exact
// p-value is one quotient of two counts.
//
// The statistics are sums of scores, every score an integer (doubled midranks with ties: decision
// H45, the distribution conditional on the observed ranks). Three ways:
//
//   - exact integers (128 bits) where the total count stays below 2^124: the Gaussian binomial
//     coefficients of the Mann-Whitney U and of Ansari-Bradley's score halves by the product
//     formula [n, j]_q = [n, j - 1]_q (1 - q^(n-j+1)) / (1 - q^j) (each factor's subtraction and
//     prefix sum exact in integers; truncated at the degree the tail needs), and the subset sums
//     of the signed rank scores;
//   - double-double counts elsewhere (ties, or totals past 2^124): every step adds positive terms
//     (a relative error of ~2^-104 per step), so a tail of n steps is good to n 2^-104;
//   - Page's L: each row's distribution over the permutations of its values (a subset dynamic
//     programme in 64-bit integers), the rows convolved in double-doubles.
//
// The work of each is counted beforehand (H13: method='exact' past its cap is an error, 'auto'
// takes the normal approximation there).
#pragma once

#include "htest/moments.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sd::htest::exact {

using i128 = __int128;
using u128 = unsigned __int128;

// the exact integer path needs every count (and twice it, mid-product) below 2^126
inline constexpr double kIntBits = 124.0;

// a nonnegative integer below 2^126 as a double-double (exact below 2^106)
inline DD to_dd(u128 v) {
    const double hi = static_cast<double>(v);
    const auto back = static_cast<u128>(hi);
    const double lo = v >= back ? static_cast<double>(v - back) : -static_cast<double>(back - v);
    return dd::two_sum(hi, lo);
}
inline DD to_dd(i128 v) { return v < 0 ? -to_dd(static_cast<u128>(-v)) : to_dd(static_cast<u128>(v)); }

inline double log2_binom(double n, double k) {
    return (std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0)) / std::log(2.0);
}

inline u128 gcd_u128(u128 a, u128 b) {
    while (b != 0) {
        const u128 t = a % b;
        a = b, b = t;
    }
    return a;
}

// C(n, k) in 128 bits (the caller keeps it below 2^126): C(n, i) = C(n, i - 1) (n - i + 1) / i with
// the common factor of C(n, i - 1) and i taken out first, so no intermediate exceeds the result.
inline u128 binom_u128(std::int64_t n, std::int64_t k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    u128 c = 1;
    for (std::int64_t i = 1; i <= k; ++i) {
        const u128 g = gcd_u128(c, static_cast<u128>(i));
        c /= g;
        c *= static_cast<u128>(n - i + 1) / (static_cast<u128>(i) / g);
    }
    return c;
}

// C(t, c) as a double-double
inline DD binom_dd(std::int64_t t, std::int64_t c) {
    c = std::min(c, t - c);
    DD v{1.0, 0.0};
    for (std::int64_t i = 1; i <= c; ++i) v = (v * static_cast<double>(t - c + i)) / static_cast<double>(i);
    return v;
}

// ---- Gaussian binomial coefficients -------------------------------------------------------

// P <- P (1 - q^(n-j+1)) / (1 - q^j): [n, j-1]_q to [n, j]_q, truncated at deg P.size() - 1
inline void qbinom_step(std::vector<i128>& P, std::int64_t n, std::int64_t j) {
    const auto b = static_cast<std::int64_t>(P.size()) - 1;
    const std::int64_t a = n - j + 1;
    for (std::int64_t v = b; v >= a; --v) P[static_cast<std::size_t>(v)] -= P[static_cast<std::size_t>(v - a)];
    for (std::int64_t v = j; v <= b; ++v) P[static_cast<std::size_t>(v)] += P[static_cast<std::size_t>(v - j)];
}

// #{U <= b} for U the Mann-Whitney statistic of samples of c and N - c (coefficients of
// [N, c]_q), C(N, c) below 2^124. Work: c (b + 1).
inline u128 mwu_lower(std::int64_t N, std::int64_t c, std::int64_t b) {
    if (b < 0) return 0;
    if (b >= c * (N - c)) return binom_u128(N, c);
    std::vector<i128> P(static_cast<std::size_t>(b + 1), 0);
    P[0] = 1;
    for (std::int64_t j = 1; j <= c; ++j) qbinom_step(P, N, j);
    u128 s = 0;
    for (i128 v : P) s += static_cast<u128>(v);
    return s;
}

// Ansari-Bradley without ties: the scores of N = 2k + odd observations are 1..k twice (and k + 1
// once when N is odd). Choosing j of the first copy and c - j of the second, the sum is
// j(j + 1)/2 + (c - j)(c - j + 1)/2 plus two independent U's of [k, j]_q and [k, c - j]_q.
struct AnsariCounts {
    std::int64_t k = 0;
    std::vector<std::vector<u128>> A, cum;  // [k, j]_q and its prefix sums, j = 0..k
    explicit AnsariCounts(std::int64_t k_) : k(k_) {
        std::vector<i128> P(static_cast<std::size_t>(k * k / 4 + 1), 0);
        P[0] = 1;
        for (std::int64_t j = 0; j <= k; ++j) {
            if (j > 0) qbinom_step(P, k, j);
            const std::int64_t deg = j * (k - j);
            std::vector<u128> a(static_cast<std::size_t>(deg + 1)), c(static_cast<std::size_t>(deg + 1));
            u128 s = 0;
            for (std::int64_t v = 0; v <= deg; ++v) {
                a[static_cast<std::size_t>(v)] = static_cast<u128>(P[static_cast<std::size_t>(v)]);
                s += a[static_cast<std::size_t>(v)];
                c[static_cast<std::size_t>(v)] = s;
            }
            A.push_back(std::move(a));
            cum.push_back(std::move(c));
        }
    }
    // #{sum of c scores from 1..k twice <= b}
    u128 even_lower(std::int64_t c, std::int64_t b) const {
        u128 tot = 0;
        if (c < 0) return 0;
        for (std::int64_t j = std::max<std::int64_t>(0, c - k); j <= std::min(c, k); ++j) {
            const std::int64_t base = j * (j + 1) / 2 + (c - j) * (c - j + 1) / 2;
            const std::vector<u128>& a = A[static_cast<std::size_t>(j)];
            const std::vector<u128>& cb = cum[static_cast<std::size_t>(c - j)];
            const auto degb = static_cast<std::int64_t>(cb.size()) - 1;
            for (std::int64_t u = 0; u < static_cast<std::int64_t>(a.size()); ++u) {
                const std::int64_t idx = b - base - u;
                if (idx < 0) break;
                tot += a[static_cast<std::size_t>(u)] * cb[static_cast<std::size_t>(std::min(idx, degb))];
            }
        }
        return tot;
    }
    // #{AB <= b} for c scores of N observations (N = 2k or 2k + 1)
    u128 lower(std::int64_t N, std::int64_t c, std::int64_t b) const {
        u128 v = even_lower(c, b);
        if (N % 2 == 1) v += even_lower(c - 1, b - (k + 1));
        return v;
    }
};

// ---- double-double counts -----------------------------------------------------------------

// The distribution of the sum of c scores drawn without replacement from groups of equal scores
// (score v[g] >= 0, multiplicity t[g]): counts by sum s = 0..smax (smax >= the largest sum). A
// dynamic programme over the groups, the number chosen and the sum; every update adds positive
// terms. Work: about (sum of t) c (smax + 1).
inline std::vector<DD> grouped_counts(
    const std::vector<std::int64_t>& v, const std::vector<std::int64_t>& t, std::int64_t c, std::int64_t smax
) {
    const std::int64_t S = smax + 1;
    std::vector<DD> dp(static_cast<std::size_t>((c + 1) * S), DD{0.0, 0.0});
    std::vector<std::int64_t> reach(static_cast<std::size_t>(c + 1), -1);  // the largest sum of j so far
    dp[0] = DD{1.0, 0.0};
    reach[0] = 0;
    std::vector<DD> w;
    for (std::size_t g = 0; g < v.size(); ++g) {
        const std::int64_t tg = t[g], vg = v[g];
        w.assign(static_cast<std::size_t>(std::min(tg, c) + 1), DD{0.0, 0.0});
        for (std::int64_t cc = 1; cc <= std::min(tg, c); ++cc) w[static_cast<std::size_t>(cc)] = binom_dd(tg, cc);
        for (std::int64_t j = c; j >= 1; --j) {
            DD* out = dp.data() + j * S;
            std::int64_t r = reach[static_cast<std::size_t>(j)];
            for (std::int64_t cc = 1; cc <= std::min(tg, j); ++cc) {
                const std::int64_t from = reach[static_cast<std::size_t>(j - cc)];
                if (from < 0) continue;
                const DD* in = dp.data() + (j - cc) * S;
                const std::int64_t shift = cc * vg, hi = std::min(smax, from + shift);
                const DD wc = w[static_cast<std::size_t>(cc)];
                if (wc.hi == 1.0 && wc.lo == 0.0) {
                    for (std::int64_t s = shift; s <= hi; ++s) out[s] = out[s] + in[s - shift];
                } else {
                    for (std::int64_t s = shift; s <= hi; ++s) out[s] = out[s] + in[s - shift] * wc;
                }
                r = std::max(r, hi);
            }
            reach[static_cast<std::size_t>(j)] = r;
        }
    }
    return std::vector<DD>(dp.begin() + c * S, dp.begin() + (c + 1) * S);
}

// The signed rank sums: sum of a_i B_i over independent fair B_i (a_i >= 1), by sum s = 0..sum a.
// T u128: counts (2^n below 2^126); T DD: probabilities, each step (x(s) + x(s - a)) / 2 (the
// halving is exact). Work: n (sum a + 1).
inline std::vector<u128> signed_rank_counts(const std::vector<std::int64_t>& a) {
    std::int64_t S = 0;
    for (std::int64_t x : a) S += x;
    std::vector<u128> c(static_cast<std::size_t>(S + 1), 0);
    c[0] = 1;
    std::int64_t top = 0;
    for (std::int64_t x : a) {
        top += x;
        for (std::int64_t s = top; s >= x; --s) c[static_cast<std::size_t>(s)] += c[static_cast<std::size_t>(s - x)];
    }
    return c;
}
inline std::vector<DD> signed_rank_probs(const std::vector<std::int64_t>& a) {
    std::int64_t S = 0;
    for (std::int64_t x : a) S += x;
    std::vector<DD> c(static_cast<std::size_t>(S + 1), DD{0.0, 0.0});
    c[0] = DD{1.0, 0.0};
    std::int64_t top = 0;
    for (std::int64_t x : a) {
        top += x;
        for (std::int64_t s = top; s >= 0; --s) {
            DD v = c[static_cast<std::size_t>(s)];
            if (s >= x) v = v + c[static_cast<std::size_t>(s - x)];
            c[static_cast<std::size_t>(s)] = DD{0.5 * v.hi, 0.5 * v.lo};
        }
    }
    return c;
}

// ---- Page's L -----------------------------------------------------------------------------

// One row's distribution: counts of sum_j (j + 1) r[pi(j)] over the n! permutations pi of the
// row's values r (integers >= 0), by that sum minus `low` (the smallest: the values in descending
// order against the weights 1..n). A dynamic programme over the subsets of the values placed in
// columns 1..|subset|, layer by layer; n <= 20 (n! < 2^63). Work: 2^n n (range + 1).
inline std::vector<std::uint64_t> page_row_counts(const std::vector<std::int64_t>& r, std::int64_t& low) {
    const auto n = static_cast<std::int64_t>(r.size());
    std::vector<std::int64_t> up(r), dn(r);
    std::sort(up.begin(), up.end());
    std::sort(dn.begin(), dn.end(), std::greater<>());
    std::int64_t hi = 0;
    low = 0;
    for (std::int64_t j = 0; j < n; ++j)
        low += (j + 1) * dn[static_cast<std::size_t>(j)], hi += (j + 1) * up[static_cast<std::size_t>(j)];
    const std::int64_t R = hi - low + 1;
    const std::uint32_t full = (1u << n) - 1u;
    // dp over masks of the current layer, keyed by the mask's index in a dense table
    std::vector<std::vector<std::uint64_t>> cur(static_cast<std::size_t>(full + 1));
    cur[0].assign(static_cast<std::size_t>(hi + 1), 0);  // by the absolute sum 0..hi
    cur[0][0] = 1;
    for (std::int64_t layer = 0; layer < n; ++layer) {
        const std::int64_t w = layer + 1;
        for (std::uint32_t mask = 0; mask <= full; ++mask) {
            if (std::popcount(mask) != layer || cur[mask].empty()) continue;
            const std::vector<std::uint64_t>& src = cur[mask];
            for (std::int64_t i = 0; i < n; ++i) {
                if (mask & (1u << i)) continue;
                const std::uint32_t nm = mask | (1u << i);
                std::vector<std::uint64_t>& dst = cur[nm];
                if (dst.empty()) dst.assign(static_cast<std::size_t>(hi + 1), 0);
                const std::int64_t add = w * r[static_cast<std::size_t>(i)];
                for (std::int64_t s = 0; s + add <= hi; ++s)
                    if (src[static_cast<std::size_t>(s)])
                        dst[static_cast<std::size_t>(s + add)] += src[static_cast<std::size_t>(s)];
            }
            std::vector<std::uint64_t>().swap(cur[mask]);
        }
    }
    std::vector<std::uint64_t> out(static_cast<std::size_t>(R), 0);
    for (std::int64_t s = low; s <= hi; ++s)
        out[static_cast<std::size_t>(s - low)] = cur[full][static_cast<std::size_t>(s)];
    return out;
}

}  // namespace sd::htest::exact
