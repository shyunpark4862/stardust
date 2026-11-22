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
#include "parallel/pool.hpp"
#include "htest/vdd.hpp"

#include <arm_neon.h>

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

// a + b of two nonnegative double-doubles: the high parts' two_sum, the low parts added to its
// error once, renormalized (the counts are sums of positive terms: ~2^-104 a step whatever the
// order, without operator+'s second two_sum)
inline vdd::D2 add_pos(vdd::D2 a, vdd::D2 b) {
    const vdd::D2 s = vdd::two_sum(a.hi, b.hi);
    return vdd::quick_two_sum(s.hi, vaddq_f64(s.lo, vaddq_f64(a.lo, b.lo)));
}
inline DD add_pos(DD a, DD b) {
    const DD s = dd::two_sum(a.hi, b.hi);
    return dd::quick_two_sum(s.hi, s.lo + (a.lo + b.lo));
}

// out[s] = out[s] + in[s] (* w, unless w is 1) for s in [0, n): two entries a NEON register (vld2
// splits the double-doubles' high and low parts), add_pos and vdd.hpp's operator*, the scalar
// operations lane by lane; the rows do not overlap.
inline void dd_axpy(DD* out, const DD* in, std::int64_t n, DD w) {
    const bool one = w.hi == 1.0 && w.lo == 0.0;
    std::int64_t s = 0;
    const vdd::D2 wv{vdupq_n_f64(w.hi), vdupq_n_f64(w.lo)};
    for (; s + 2 <= n; s += 2) {
        const float64x2x2_t o = vld2q_f64(&out[s].hi), x = vld2q_f64(&in[s].hi);
        const vdd::D2 xv{x.val[0], x.val[1]};
        const vdd::D2 r = add_pos(vdd::D2{o.val[0], o.val[1]}, one ? xv : vdd::mul(xv, wv));
        float64x2x2_t st;
        st.val[0] = r.hi, st.val[1] = r.lo;
        vst2q_f64(&out[s].hi, st);
    }
    for (; s < n; ++s) out[s] = add_pos(out[s], one ? in[s] : in[s] * w);
}

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
                dd_axpy(out + shift, in, hi - shift + 1, wc);
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
// The same as probabilities P(T = s) for s = 0..L only, L <= floor(sum a / 2): the lower half of
// a distribution symmetric about sum a / 2. Each step's distribution is symmetric about its own
// half sum c / 2 and is kept for s <= min(L, c / 2) alone; the one or two entries the next step
// needs above that are the mirrored ones below. Each entry is (x(s) + x(s - a)) / 2 of positive
// double-doubles (the sum without renormalizing its low parts twice; the halving is exact), in
// NEON two entries a register, four at once, from the top down (an entry's inputs are at or below
// it, read before the chunk is written); a long one's steps on the pool, into a second array.
// Work: about n min(L, sum a / 2), a quarter of signed_rank_counts' n (sum a) / 2 per entry in
// double-doubles.
inline void signed_rank_lower(
    const std::vector<std::int64_t>& a, std::int64_t L, std::vector<double>& ph, std::vector<double>& pl
) {
    ph.assign(static_cast<std::size_t>(L + 1), 0.0), pl.assign(static_cast<std::size_t>(L + 1), 0.0);
    double* h = ph.data();
    double* l = pl.data();
    h[0] = 1.0;
    std::int64_t cum = 0, H = 0;  // the sum so far, and the top entry kept: min(L, cum / 2)
    auto one = [&](std::int64_t s, double oh, double ol, double xh, double xl) {
        const DD t = dd::two_sum(oh, xh);
        const DD v = dd::quick_two_sum(t.hi, t.lo + (ol + xl));
        h[s] = 0.5 * v.hi, l[s] = 0.5 * v.lo;
    };
    // a long distribution on the pool: each step from the old array into a second one (the same
    // operations per entry, so the same bits), fixed pieces of kPiece entries
    constexpr std::int64_t kPiece = 1 << 13;
    std::vector<double> qh, ql;
    for (const std::int64_t x : a) {
        const std::int64_t c2 = cum + x, H2 = std::min(L, c2 / 2);
        if (H2 >= 4 * kPiece && parallel::available()) {
            qh.resize(ph.size()), ql.resize(pl.size());
            double *nh = qh.data(), *nl = ql.data();
            const double *oh = h, *ol = l;
            for_pieces_woken(H2 + 1, kPiece, (H2 + 1) * 8, [&](std::int64_t, std::int64_t s0, std::int64_t s1) {
                const float64x2_t half = vdupq_n_f64(0.5);
                std::int64_t s = s0;
                auto put = [&](std::int64_t q, double ah, double al, double bh, double bl) {
                    const DD t = dd::two_sum(ah, bh);
                    const DD v = dd::quick_two_sum(t.hi, t.lo + (al + bl));
                    nh[q] = 0.5 * v.hi, nl[q] = 0.5 * v.lo;
                };
                for (; s < s1 && s < x && s <= H; ++s) nh[s] = 0.5 * oh[s], nl[s] = 0.5 * ol[s];
                for (; s + 4 <= s1 && s + 3 <= H; s += 4)
                    for (int k = 0; k < 4; k += 2) {
                        const float64x2_t ah = vld1q_f64(oh + s + k), al = vld1q_f64(ol + s + k);
                        const float64x2_t bh = vld1q_f64(oh + s + k - x), bl = vld1q_f64(ol + s + k - x);
                        const float64x2_t t = vaddq_f64(ah, bh), bb = vsubq_f64(t, ah);
                        const float64x2_t e = vaddq_f64(vsubq_f64(ah, vsubq_f64(t, bb)), vsubq_f64(bh, bb));
                        const float64x2_t w = vaddq_f64(e, vaddq_f64(al, bl));
                        const float64x2_t vh = vaddq_f64(t, w), vl = vsubq_f64(w, vsubq_f64(vh, t));
                        vst1q_f64(nh + s + k, vmulq_f64(half, vh)), vst1q_f64(nl + s + k, vmulq_f64(half, vl));
                    }
                for (; s < s1; ++s) {
                    const std::int64_t m = s <= H ? s : cum - s;  // above H: the mirrored entry
                    put(s, m >= 0 ? oh[m] : 0.0, m >= 0 ? ol[m] : 0.0, s >= x ? oh[s - x] : 0.0,
                        s >= x ? ol[s - x] : 0.0);
                }
            });
            ph.swap(qh), pl.swap(ql);
            h = ph.data(), l = pl.data();
            cum = c2, H = H2;
            continue;
        }
        // the entries above H: their own old value is the mirror cum - s (below H, not yet written)
        std::int64_t s = H2;
        for (; s > H; --s) {
            const std::int64_t m = cum - s;
            const double xh = s >= x ? h[s - x] : 0.0, xl = s >= x ? l[s - x] : 0.0;
            one(s, m >= 0 ? h[m] : 0.0, m >= 0 ? l[m] : 0.0, xh, xl);
        }
        // s <= H: both inputs kept
        const float64x2_t half = vdupq_n_f64(0.5);
        for (; s - 3 >= x; s -= 4) {
            const std::int64_t b = s - 3;
            const float64x2_t oh0 = vld1q_f64(h + b), oh1 = vld1q_f64(h + b + 2);
            const float64x2_t ol0 = vld1q_f64(l + b), ol1 = vld1q_f64(l + b + 2);
            const float64x2_t xh0 = vld1q_f64(h + b - x), xh1 = vld1q_f64(h + b - x + 2);
            const float64x2_t xl0 = vld1q_f64(l + b - x), xl1 = vld1q_f64(l + b - x + 2);
            auto vone = [&](float64x2_t oh, float64x2_t ol, float64x2_t xh, float64x2_t xl, double* ho, double* lo) {
                const float64x2_t t = vaddq_f64(oh, xh), bb = vsubq_f64(t, oh);
                const float64x2_t e = vaddq_f64(vsubq_f64(oh, vsubq_f64(t, bb)), vsubq_f64(xh, bb));
                const float64x2_t w = vaddq_f64(e, vaddq_f64(ol, xl));
                const float64x2_t vh = vaddq_f64(t, w), vl = vsubq_f64(w, vsubq_f64(vh, t));
                vst1q_f64(ho, vmulq_f64(half, vh)), vst1q_f64(lo, vmulq_f64(half, vl));
            };
            vone(oh0, ol0, xh0, xl0, h + b, l + b);
            vone(oh1, ol1, xh1, xl1, h + b + 2, l + b + 2);
        }
        for (; s >= x; --s) one(s, h[s], l[s], h[s - x], l[s - x]);
        for (; s >= 0; --s) h[s] *= 0.5, l[s] *= 0.5;  // (exact)
        cum = c2, H = H2;
    }
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
