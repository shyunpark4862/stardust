// The exact null distribution of the two-sample Kolmogorov-Smirnov statistic (milestone H4,
// docs/design/htest.md §3.5; H14: inside sd.htest).
//
// With sizes na and nb, L = lcm(na, nb), a = L/na and b = L/nb, the gap of the two empirical
// distribution functions after i values of the first sample and j of the second is
// g = i a - j b, an integer in units of 1/L, so D = h/L and "at least as extreme" compares
// integers (docs/design/htest.md §3.1). The pooled sorted values come in blocks: a block is one
// value (no ties) or a run of t equal values (ties), and the functions are observed only at the
// end of each block. Under the null every arrangement of the labels is equally likely; with
// ties, the distribution is conditional on the observed blocks (decision H55: R's psmirnov with
// ties, R >= 4.3), each block taking k of the indexed sample's values with the hypergeometric
// probability C(t, k) C(R - t, r - k) / C(R, r) (R the positions left, r the indexed sample's
// values left).
//
//   tail(...)   P(D >= h/L) for 'two-sided' (|g| >= h), 'greater' (g >= h) or 'less' (-g >= h):
//               a dynamic programme over the blocks whose state is the count of the smaller
//               sample's values so far and whose weight is the probability of that prefix.
//               A state on or past the boundary is absorbed: its weight is added to the tail
//               and it is dropped. Every weight and every addition is a positive double-double,
//               so the tail is a sum of positive terms (no 1 - P(inside)), good to about
//               (number of blocks) 2^-104 relative, and the prefix probabilities never
//               overflow (scipy counts paths in doubles and gives up past 1e308). The states
//               whose weight underflows to 0 drop out of the band, so the work is about
//               (na + nb) x (the band's width) multiply-adds.
//   equal_tail(n, h, two_sided)  na = nb = n without ties: Hodges' closed forms,
//               C(2n, n - h)/C(2n, n) one-sided and 2 sum_k (-1)^(k-1) C(2n, n - k h)/C(2n, n)
//               two-sided, the ratios as products of double-doubles (O(n)).
//   work(...)   an upper bound of tail()'s multiply-adds, for the cap (H13).
#pragma once

#include "parallel/pool.hpp"
#include "special/dd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::htest::ks2 {

namespace dd = special::dd;
using dd::DD;

enum Alt { kTwoSided = 0, kLess = 1, kGreater = 2 };

// A positive double-double sum (Neumaier's, with the low parts).
struct PosSum {
    double hi = 0.0, lo = 0.0;
    void add(DD v) {
        const DD s = dd::two_sum(hi, v.hi);
        hi = s.hi;
        lo += s.lo + v.lo;
    }
    DD value() const { return dd::two_sum(hi, lo); }
};

// The hypergeometric probabilities of k = kmin..kmax of a block of t positions holding k of the
// r indexed values left among R positions, as double-doubles, by the ratios of consecutive terms
// from the mode (each ratio at most 1 going away from it: no overflow) and normalized by their
// sum (which is 1).
inline void hypergeometric(std::int64_t R, std::int64_t r, std::int64_t t, std::int64_t& kmin, std::vector<DD>& h) {
    kmin = std::max<std::int64_t>(0, t - (R - r));
    const std::int64_t kmax = std::min(t, r);
    h.assign(static_cast<std::size_t>(kmax - kmin + 1), DD{0.0, 0.0});
    if (kmax == kmin) {
        h[0] = {1.0, 0.0};
        return;
    }
    // ratio(k) = P(k + 1) / P(k) = (t - k)(r - k) / ((k + 1)(R - t - r + k + 1))
    auto ratio = [&](std::int64_t k) {
        const DD num = dd::two_prod(static_cast<double>(t - k), static_cast<double>(r - k));
        const DD den = dd::two_prod(static_cast<double>(k + 1), static_cast<double>(R - t - r + k + 1));
        return num / den;
    };
    auto mode = static_cast<std::int64_t>(
        std::floor(static_cast<double>(t + 1) * static_cast<double>(r + 1) / static_cast<double>(R + 2))
    );
    mode = std::clamp(mode, kmin, kmax);
    const auto at = [&](std::int64_t k) -> DD& { return h[static_cast<std::size_t>(k - kmin)]; };
    at(mode) = {1.0, 0.0};
    for (std::int64_t k = mode; k < kmax; ++k) {
        at(k + 1) = at(k) * ratio(k);
        if (at(k + 1).hi == 0.0) break;
    }
    for (std::int64_t k = mode; k > kmin; --k) {
        at(k - 1) = at(k) / ratio(k - 1);
        if (at(k - 1).hi == 0.0) break;
    }
    PosSum s;
    for (const DD& v : h) s.add(v);
    const DD tot = s.value();
    for (DD& v : h) v = v / tot;
}

// The problem: the sizes, the blocks (sizes of the runs of equal pooled values, in order; all 1
// without ties) and the boundary h for the alternative.
struct Problem {
    std::int64_t na = 0, nb = 0, h = 0;
    int alt = kTwoSided;
    const std::vector<std::int64_t>* blocks = nullptr;  // nullptr: no ties (every block one value)
};

// The DP's frame: the indexed sample is the smaller one (fewer states), c its count; the gap is
// g = sgn (c (a + b) - p q) after p positions (sgn = +1, q = b when the first sample is indexed).
struct Frame {
    std::int64_t nc, no, N, apb, q, h;
    int sgn, alt;
    explicit Frame(const Problem& P) {
        const std::int64_t L = std::lcm(P.na, P.nb), a = L / P.na, b = L / P.nb;
        const bool first = P.na <= P.nb;
        nc = first ? P.na : P.nb, no = first ? P.nb : P.na, N = P.na + P.nb;
        apb = a + b, q = first ? b : a, sgn = first ? 1 : -1, h = P.h, alt = P.alt;
    }
    // the c inside the boundary after p positions: [lo, hi] (empty when lo > hi)
    void inside(std::int64_t p, std::int64_t& lo, std::int64_t& hi) const {
        // g = sgn (c apb - p q); inside: g < h ('greater'), -g < h ('less'), both (two-sided)
        auto fdiv = [](std::int64_t x, std::int64_t y) { return x >= 0 ? x / y : -((-x + y - 1) / y); };
        const std::int64_t pq = p * q;
        // c apb - pq < h  <=>  c <= floor((pq + h - 1) / apb);  c apb - pq > -h  <=>  c >= floor((pq - h) / apb) + 1
        const std::int64_t up = fdiv(pq + h - 1, apb), down = fdiv(pq - h, apb) + 1;
        const bool need_up = alt == kTwoSided || (alt == kGreater) == (sgn > 0);
        const bool need_down = alt == kTwoSided || (alt == kGreater) != (sgn > 0);
        lo = need_down ? down : std::numeric_limits<std::int64_t>::min() / 2;
        hi = need_up ? up : std::numeric_limits<std::int64_t>::max() / 2;
    }
    bool outside(std::int64_t c, std::int64_t p) const {
        const std::int64_t g = sgn * (c * apb - p * q);
        if (alt == kGreater) return g >= h;
        if (alt == kLess) return -g >= h;
        return g >= h || -g >= h;
    }
};

// An upper bound of tail()'s multiply-adds: the reachable states of each block times its choices
// (the states inside the band only, for the two-sided test).
inline double work(const Problem& P) {
    const Frame F(P);
    const auto nblocks = P.blocks ? static_cast<std::int64_t>(P.blocks->size()) : F.N;
    double w = 0.0;
    std::int64_t p = 0;
    // the band |c (a + b) - p q| < h holds at most 2h/(a + b) + 1 states
    const double band = static_cast<double>(2 * F.h) / static_cast<double>(F.apb) + 2.0;
    for (std::int64_t i = 0; i < nblocks; ++i) {
        const std::int64_t t = P.blocks ? (*P.blocks)[static_cast<std::size_t>(i)] : 1;
        const std::int64_t lo = std::max<std::int64_t>(0, p - F.no), hi = std::min(F.nc, p);
        double states = static_cast<double>(hi - lo + 1);
        if (P.alt == kTwoSided) states = std::min(states, band);
        w += states * static_cast<double>(std::min(t, F.nc) + 1);
        p += t;
    }
    return w;
}

// P(D >= h/L) (module comment). h <= 0 gives 1.
inline double tail(const Problem& P) {
    if (P.h <= 0) return 1.0;
    const Frame F(P);
    const auto nblocks = P.blocks ? static_cast<std::int64_t>(P.blocks->size()) : F.N;
    std::vector<DD> w(static_cast<std::size_t>(F.nc + 1), DD{0.0, 0.0}), nw = w;
    w[0] = {1.0, 0.0};
    std::int64_t lo = 0, hi = 0, p = 0;
    PosSum absorbed;
    std::vector<DD> hg;
    for (std::int64_t blk = 0; blk < nblocks; ++blk) {
        const std::int64_t t = P.blocks ? (*P.blocks)[static_cast<std::size_t>(blk)] : 1;
        const std::int64_t R = F.N - p, np = p + t;
        const std::int64_t nlo = std::max<std::int64_t>(lo, np - F.no), nhi = std::min(F.nc, hi + t);
        for (std::int64_t c = nlo; c <= nhi; ++c) nw[static_cast<std::size_t>(c)] = {0.0, 0.0};
        if (t == 1) {
            // pulled: nw[c] = (w[c - 1] r_(c-1) + w[c] (R - r_c)) / R, r_c = nc - c the indexed
            // values left; the states outside the band (an interval of c at np) absorbed
            const DD inv = DD{1.0, 0.0} / static_cast<double>(R);
            std::int64_t in_lo, in_hi;
            F.inside(np, in_lo, in_hi);
            auto pulled = [&](std::int64_t c) {
                DD v{0.0, 0.0};
                if (c - 1 >= lo && c - 1 <= hi)
                    v = w[static_cast<std::size_t>(c - 1)] * static_cast<double>(F.nc - c + 1);
                if (c >= lo && c <= hi) v = v + w[static_cast<std::size_t>(c)] * static_cast<double>(R - (F.nc - c));
                return v * inv;
            };
            const std::int64_t a0 = std::max(nlo, in_lo), a1 = std::min(nhi, in_hi);
            for (std::int64_t c = nlo; c <= std::min(nhi, a0 - 1); ++c) absorbed.add(pulled(c));
            for (std::int64_t c = std::max(nlo, a1 + 1); c <= nhi; ++c)
                if (c >= a0) absorbed.add(pulled(c));
            if (a0 <= a1) {
                // the interior: both neighbours inside [lo, hi] but at the ends
                std::int64_t c = a0;
                for (; c <= a1 && (c - 1 < lo || c > hi); ++c) nw[static_cast<std::size_t>(c)] = pulled(c);
                const std::int64_t mid_end = std::min(a1, hi);
                const DD* wp = w.data();
                DD* out = nw.data();
                for (; c <= mid_end; ++c) {
                    const DD v =
                        wp[c - 1] * static_cast<double>(F.nc - c + 1) + wp[c] * static_cast<double>(R - (F.nc - c));
                    out[c] = v * inv;
                }
                for (; c <= a1; ++c) nw[static_cast<std::size_t>(c)] = pulled(c);
            }
        } else {
            for (std::int64_t c = lo; c <= hi; ++c) {
                const DD wc = w[static_cast<std::size_t>(c)];
                if (wc.hi == 0.0) continue;
                std::int64_t kmin;
                hypergeometric(R, F.nc - c, t, kmin, hg);
                for (std::size_t i = 0; i < hg.size(); ++i) {
                    if (hg[i].hi == 0.0) continue;
                    const std::int64_t c2 = c + kmin + static_cast<std::int64_t>(i);
                    const DD v = wc * hg[i];
                    if (F.outside(c2, np)) absorbed.add(v);
                    else nw[static_cast<std::size_t>(c2)] = nw[static_cast<std::size_t>(c2)] + v;
                }
            }
        }
        // the live range: the states whose weight did not underflow to 0
        lo = nhi + 1, hi = nlo - 1;
        for (std::int64_t c = nlo; c <= nhi; ++c) {
            if (nw[static_cast<std::size_t>(c)].hi != 0.0) {
                lo = std::min(lo, c);
                hi = std::max(hi, c);
            }
        }
        std::swap(w, nw);
        p = np;
        if (lo > hi) break;  // everything absorbed
    }
    const DD r = absorbed.value();
    return std::clamp(r.hi + r.lo, 0.0, 1.0);
}

// Hodges' closed forms for na = nb = n without ties (module comment): A_k = C(2n, n - k h) /
// C(2n, n) = prod_{j < k h} (n - j)/(n + j + 1).
inline double equal_tail(std::int64_t n, std::int64_t h, bool two_sided) {
    if (h <= 0) return 1.0;
    if (h > n) return 0.0;
    DD A{1.0, 0.0};
    std::int64_t j = 0;
    auto advance = [&](std::int64_t to) {
        for (; j < to; ++j) A = A * (DD{static_cast<double>(n - j), 0.0} / static_cast<double>(n + j + 1));
    };
    advance(h);
    if (!two_sided) return A.hi + A.lo;
    DD s{0.0, 0.0};
    for (std::int64_t k = 1; k * h <= n; ++k) {
        advance(k * h);
        if (A.hi == 0.0) break;
        s = (k & 1) ? s + A : s - A;
    }
    const DD r = s * 2.0;
    return std::clamp(r.hi + r.lo, 0.0, 1.0);
}

}  // namespace sd::htest::ks2
