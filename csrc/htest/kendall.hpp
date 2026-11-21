// Kendall's exact null distribution (milestone H5, docs/design/htest.md §3.6): the number of
// inversions K of a random permutation of n (the Mahonian numbers), whose lower tail gives
// kendalltau's exact p-value without ties (scipy's _kendall_p_exact).
//
// The counts satisfy M_j(k) = sum_{i < j} M_{j-1}(k - i). Two ways, both kept to k <= c (the
// tail the p-value needs; the distribution is symmetric, so the tail is always the lower one):
//
//   - exact 128-bit integers while n! < 2^127 (n <= 33, scipy's 'auto' range): the window sum by
//     a running sum (an exact subtraction);
//   - probabilities g_j = M_j / j! in double-doubles beyond: g_j(k) = (1/j) sum_{i < j} g_{j-1}(k - i),
//     the window [k - j + 1, k] taken as the suffix of one block of j entries and the prefix of
//     the next (blocks of the window's length: a window spans at most two), so every step adds
//     positive terms only (no subtraction, a relative error of a few 2^-106 per step). The
//     vector is rescaled by 2^600 whenever its largest entry falls below 2^-600, so tails below
//     the doubles' range come out as subnormals or 0 instead of losing their digits on the way.
//
// The work is n (c + 1) entry updates (H13: kendalltau caps it at 1e9, and the entries held at
// 1e7).
#pragma once

#include "htest/moments.hpp"
#include "htest/rank_exact.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::htest::kendall {

using exact::i128;
using exact::u128;

// (P(K <= c), P(K = c)) for K the inversions of a random permutation of n, c <= n(n - 1)/4
struct Tail {
    double le, at;
};

inline Tail exact_tail_int(std::int64_t n, std::int64_t c) {
    std::vector<u128> m(static_cast<std::size_t>(c + 1), 0), next(static_cast<std::size_t>(c + 1));
    m[0] = 1;
    for (std::int64_t j = 2; j <= n; ++j) {
        u128 run = 0;
        for (std::int64_t k = 0; k <= c; ++k) {
            run += m[static_cast<std::size_t>(k)];
            if (k - j >= 0) run -= m[static_cast<std::size_t>(k - j)];
            next[static_cast<std::size_t>(k)] = run;
        }
        m.swap(next);
    }
    u128 fact = 1;
    for (std::int64_t j = 2; j <= n; ++j) fact *= static_cast<u128>(j);
    u128 le = 0;
    for (std::int64_t k = 0; k <= c; ++k) le += m[static_cast<std::size_t>(k)];
    const DD f = exact::to_dd(fact);
    return {to_double(exact::to_dd(le) / f), to_double(exact::to_dd(m[static_cast<std::size_t>(c)]) / f)};
}

inline Tail exact_tail_dd(std::int64_t n, std::int64_t c) {
    const auto len = static_cast<std::size_t>(c + 1);
    std::vector<DD> g(len, DD{0.0, 0.0}), pre(len), suf(len);
    g[0] = DD{1.0, 0.0};
    int scale = 0;  // the entries are the probabilities times 2^scale
    for (std::int64_t j = 2; j <= n; ++j) {
        // block sums within blocks [b j, (b + 1) j): prefixes and suffixes, positive additions
        for (std::int64_t s = 0; s <= c; s += j) {
            const std::int64_t e = std::min(c, s + j - 1);
            DD acc{0.0, 0.0};
            for (std::int64_t k = s; k <= e; ++k)
                pre[static_cast<std::size_t>(k)] = acc = acc + g[static_cast<std::size_t>(k)];
            acc = DD{0.0, 0.0};
            for (std::int64_t k = e; k >= s; --k)
                suf[static_cast<std::size_t>(k)] = acc = acc + g[static_cast<std::size_t>(k)];
        }
        const double inv = 1.0 / static_cast<double>(j);
        double top = 0.0;
        for (std::int64_t k = 0; k <= c; ++k) {
            const std::int64_t a = k - j + 1;  // the window [a, k]
            DD w = pre[static_cast<std::size_t>(k)];
            if (a > 0 && a % j != 0) w = suf[static_cast<std::size_t>(a)] + w;
            // (the division by j: exact for j a power of 2, else the double-double quotient)
            g[static_cast<std::size_t>(k)] =
                (j & (j - 1)) == 0 ? DD{w.hi * inv, w.lo * inv} : w / static_cast<double>(j);
            top = std::max(top, g[static_cast<std::size_t>(k)].hi);
        }
        if (top > 0.0 && top < 0x1p-600) {
            for (DD& v : g) v = dd::ldexp(v, 600);
            scale += 600;
        }
    }
    DD le{0.0, 0.0};
    for (const DD& v : g) le = le + v;
    return {std::ldexp(to_double(le), -scale), std::ldexp(to_double(g[static_cast<std::size_t>(c)]), -scale)};
}

// n! < 2^127 up to n = 33
inline Tail exact_tail(std::int64_t n, std::int64_t c) { return n <= 33 ? exact_tail_int(n, c) : exact_tail_dd(n, c); }

// scipy's _kendall_p_exact: con the concordant pairs (no ties), alt 0 two-sided, 1 less,
// 2 greater. The two-sided p-value is twice the near tail (clipped to 1); a one-sided one is
// half of it when the observation lies on the alternative's side, else 1 - half + P(K = c).
inline double exact_p(std::int64_t n, std::int64_t con, int alt) {
    const std::int64_t tot = n * (n - 1) / 2;
    const bool right = con >= tot - con;
    const std::int64_t c = std::min(con, tot - con);
    if (n <= 2) {
        const double at = n == 2 ? 0.5 : 1.0;
        return alt == 0 ? 1.0 : (right == (alt == 2) ? 0.5 : std::min(1.0, 0.5 + at));
    }
    const Tail t = exact_tail(n, c);
    const double two = std::min(1.0, 2.0 * t.le);
    if (alt == 0) return two;
    if (right == (alt == 2)) return std::min(1.0, t.le);
    return std::min(1.0, std::max(0.0, 1.0 - t.le + t.at));
}

}  // namespace sd::htest::kendall
