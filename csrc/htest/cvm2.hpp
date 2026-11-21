// The exact null distribution of the two-sample Cramér-von Mises statistic (milestone H4,
// docs/design/htest.md §3.5; decision H56; H14: inside sd.htest).
//
// The statistic is T = U/(k N) - (4k - 1)/(6N), k = nx ny, N = nx + ny, with
//     U = nx sum_i (r_i - i)^2 + ny sum_j (s_j - j)^2
// (r_i the midrank of x's i-th smallest value in the pooled sample, s_j y's; Anderson 1962, eq.
// 10). 4U is an integer (midranks are halves), so the exact p-value P(U >= u) compares integers.
// The pooled sorted values come in blocks (one value without ties, a run of t equal values with
// them); a block at position p0 holding k values of a sample whose count so far is c adds
// n sum_{m=1..k} (2R - 2c - 2m)^2 to 4U (n that sample's size, 2R = 2 p0 + t + 1 the block's
// doubled midrank). Without ties this is Burr's (1963) lattice-path count; with ties it is the
// distribution conditional on the observed blocks (decision H58, the rank tests' rule H45), each block taking k of the indexed sample's values with
// the hypergeometric probability (htest/ks2.hpp's).
//
//   tail(...)  P(4U/G >= v) (G = gcd(nx, ny), which divides every increment; without ties every
//              increment is also a multiple of 4, so the unit is 4U/G / 4): a dynamic programme
//              over the blocks whose state is the smaller sample's count so far with, per state,
//              the sorted list of the partial sums and their probabilities (positive
//              double-doubles). The increments are never negative, and the least and the
//              largest increment still to come from each state are known (a min-plus and a
//              max-plus pass over the blocks beforehand): a partial sum that will reach v
//              whatever follows is absorbed (its probability added to the tail at once), one
//              that cannot reach it is dropped. So a state keeps only the window of partial
//              sums still undecided, which bounds the work; scipy enumerates every partial sum
//              (2.2 ms at nx = ny = 20, seconds at 40 x 35), so its 'auto' stays at
//              max(nx, ny) <= 20.
//   plan(...)  the passes and an upper bound of tail()'s list entries (for the cap, H13), with
//              the largest number of entries held at once.
#pragma once

#include "htest/ks2.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace sd::htest::cvm2 {

using ks2::DD;
using ks2::PosSum;

struct Problem {
    std::int64_t nx = 0, ny = 0;
    std::int64_t v = 0;  // the observed 4U / G
    const std::vector<std::int64_t>* blocks = nullptr;  // run sizes in order; nullptr without ties
};

// sum_{m=1..k} (A - 2m)^2 = k A^2 - 2 A k (k + 1) + 2 k (k + 1)(2k + 1)/3
inline std::int64_t run_sum(std::int64_t A, std::int64_t k) {
    return k * A * A - 2 * A * k * (k + 1) + 2 * k * (k + 1) * (2 * k + 1) / 3;
}

inline constexpr std::int64_t kNone = std::numeric_limits<std::int64_t>::max() / 4;

// The blocks, the units and the passes of one problem.
struct Plan {
    std::int64_t nc = 0, no = 0, N = 0, G = 1, unit = 1, v = 0;  // v in units
    std::vector<std::int64_t> t, p0;  // the blocks' sizes and first positions
    // rem_lo[b][c], rem_hi[b][c]: the least and the largest sum of the increments still to come
    // from the state c at the start of block b (kNone where c cannot be there)
    std::vector<std::int64_t> rem_lo, rem_hi;
    double entries = 0.0, held = 0.0;  // the bounds of tail()'s work
    bool ok = false;

    std::int64_t lo_at(std::size_t b, std::int64_t c) const {
        return rem_lo[b * static_cast<std::size_t>(nc + 1) + static_cast<std::size_t>(c)];
    }
    std::int64_t hi_at(std::size_t b, std::int64_t c) const {
        return rem_hi[b * static_cast<std::size_t>(nc + 1) + static_cast<std::size_t>(c)];
    }
    // the block's increment (in units) when it takes k of the indexed sample's t values, the
    // indexed count c (the other's p0 - c) before it
    std::int64_t inc(std::size_t b, std::int64_t c, std::int64_t k) const {
        const std::int64_t tt = t[b], twoR = 2 * p0[b] + tt + 1;
        return (nc * run_sum(twoR - 2 * c, k) + no * run_sum(twoR - 2 * (p0[b] - c), tt - k)) / (G * unit);
    }
    std::int64_t kmin(std::size_t b, std::int64_t c) const {
        return std::max<std::int64_t>(0, t[b] - (no - (p0[b] - c)));
    }
    std::int64_t kmax(std::size_t b, std::int64_t c) const { return std::min(t[b], nc - c); }
};

// The plan: false (and the bounds' rough size) when the states alone are past `max_states`.
inline Plan plan(const Problem& P, double max_states) {
    Plan L;
    L.nc = std::min(P.nx, P.ny), L.no = std::max(P.nx, P.ny), L.N = P.nx + P.ny, L.G = std::gcd(P.nx, P.ny);
    L.unit = P.blocks ? 1 : 4;
    L.v = (P.v + L.unit - 1) / L.unit;  // 4U/G >= v  <=>  (4U/G)/unit >= ceil(v/unit)
    const auto B = P.blocks ? P.blocks->size() : static_cast<std::size_t>(L.N);
    const double states = static_cast<double>(L.nc + 1) * static_cast<double>(B + 1);
    if (states > max_states) {
        L.entries = L.held = states * 256.0;  // (far past the cap: its size only)
        return L;
    }
    L.t.resize(B), L.p0.resize(B);
    std::int64_t p = 0;
    for (std::size_t b = 0; b < B; ++b) {
        L.t[b] = P.blocks ? (*P.blocks)[b] : 1;
        L.p0[b] = p;
        p += L.t[b];
    }
    const auto W = static_cast<std::size_t>(L.nc + 1);
    L.rem_lo.assign((B + 1) * W, kNone);
    L.rem_hi.assign((B + 1) * W, -kNone);
    L.rem_lo[B * W + static_cast<std::size_t>(L.nc)] = L.rem_hi[B * W + static_cast<std::size_t>(L.nc)] = 0;
    for (std::size_t b = B; b-- > 0;) {
        const std::int64_t lo = std::max<std::int64_t>(0, L.p0[b] - L.no), hi = std::min(L.nc, L.p0[b]);
        for (std::int64_t c = lo; c <= hi; ++c) {
            std::int64_t mn = kNone, mx = -kNone;
            for (std::int64_t k = L.kmin(b, c); k <= L.kmax(b, c); ++k) {
                const std::size_t nxt = (b + 1) * W + static_cast<std::size_t>(c + k);
                if (L.rem_lo[nxt] == kNone) continue;
                const std::int64_t d = L.inc(b, c, k);
                mn = std::min(mn, d + L.rem_lo[nxt]);
                mx = std::max(mx, d + L.rem_hi[nxt]);
            }
            L.rem_lo[b * W + static_cast<std::size_t>(c)] = mn;
            L.rem_hi[b * W + static_cast<std::size_t>(c)] = mx;
        }
    }
    // forward: the least and largest partial sums at each state, and the window kept there
    std::vector<std::int64_t> flo(W, kNone), fhi(W, -kNone), glo(W), ghi(W);
    flo[0] = fhi[0] = 0;
    for (std::size_t b = 0; b < B; ++b) {
        std::fill(glo.begin(), glo.end(), kNone);
        std::fill(ghi.begin(), ghi.end(), -kNone);
        const std::int64_t lo = std::max<std::int64_t>(0, L.p0[b] - L.no), hi = std::min(L.nc, L.p0[b]);
        for (std::int64_t c = lo; c <= hi; ++c) {
            if (flo[static_cast<std::size_t>(c)] == kNone) continue;
            for (std::int64_t k = L.kmin(b, c); k <= L.kmax(b, c); ++k) {
                const std::int64_t d = L.inc(b, c, k);
                auto& a = glo[static_cast<std::size_t>(c + k)];
                auto& z = ghi[static_cast<std::size_t>(c + k)];
                a = std::min(a, flo[static_cast<std::size_t>(c)] + d);
                z = std::max(z, fhi[static_cast<std::size_t>(c)] + d);
            }
        }
        // the entries a state keeps: its partial sums s with v - rem_hi <= s < v - rem_lo
        double held = 0.0;
        for (std::int64_t c = 0; c <= L.nc; ++c) {
            if (glo[static_cast<std::size_t>(c)] == kNone) continue;
            const std::int64_t rl = L.lo_at(b + 1, c), rh = L.hi_at(b + 1, c);
            if (rl == kNone) continue;
            const std::int64_t a = std::max(glo[static_cast<std::size_t>(c)], L.v - rh);
            const std::int64_t z = std::min(ghi[static_cast<std::size_t>(c)], L.v - rl - 1);
            if (z >= a) held += static_cast<double>(z - a + 1);
        }
        L.held = std::max(L.held, held);
        L.entries += held * static_cast<double>(std::min(L.t[b], L.nc) + 1);
        flo.swap(glo), fhi.swap(ghi);
    }
    L.ok = true;
    return L;
}

using List = std::vector<std::pair<std::int64_t, DD>>;  // (partial sum, probability), ascending

// out += src shifted by s and scaled by f: the sums below `keep` dropped (they cannot reach v),
// those at `absorb` or above absorbed (they reach it whatever follows)
inline void merge_into(
    List& out,
    List& tmp,
    const List& src,
    std::int64_t s,
    DD f,
    std::int64_t keep,
    std::int64_t absorb,
    PosSum& absorbed
) {
    tmp.clear();
    tmp.reserve(out.size() + src.size());
    std::size_t i = 0, j = 0;
    while (j < src.size() && src[j].first + s < keep) ++j;
    while (j < src.size() && src[j].first + s < absorb) {
        const std::int64_t key = src[j].first + s;
        while (i < out.size() && out[i].first < key) tmp.push_back(out[i++]);
        const DD add = src[j].second * f;
        if (i < out.size() && out[i].first == key) {
            tmp.emplace_back(key, out[i].second + add);
            ++i;
        } else {
            tmp.emplace_back(key, add);
        }
        ++j;
    }
    while (i < out.size()) tmp.push_back(out[i++]);
    for (; j < src.size(); ++j) absorbed.add(src[j].second * f);
    out.swap(tmp);
}

// P(4U/G >= v) (module comment) on a plan made for it (L.ok); -1 when the entries merged pass
// `budget` or those held at once pass `hold` (the plan's bounds are upper bounds, often far above
// the work done: a problem past them is tried within these limits).
inline double tail(
    const Plan& L,
    double budget = std::numeric_limits<double>::infinity(),
    double hold = std::numeric_limits<double>::infinity()
) {
    if (L.v <= 0) return 1.0;
    const std::size_t B = L.t.size();
    std::vector<List> w(static_cast<std::size_t>(L.nc + 1)), nw(static_cast<std::size_t>(L.nc + 1));
    w[0].emplace_back(0, DD{1.0, 0.0});
    PosSum absorbed;
    List tmp;
    std::vector<DD> hg;
    std::int64_t lo = 0, hi = 0;
    double merged = 0.0;
    for (std::size_t b = 0; b < B; ++b) {
        const std::int64_t t = L.t[b], R = L.N - L.p0[b], np = L.p0[b] + t;
        const std::int64_t nlo = std::max<std::int64_t>(lo, np - L.no), nhi = std::min(L.nc, hi + t);
        for (std::int64_t c = nlo; c <= nhi; ++c) nw[static_cast<std::size_t>(c)].clear();
        for (std::int64_t c = lo; c <= hi; ++c) {
            const List& src = w[static_cast<std::size_t>(c)];
            if (src.empty()) continue;
            std::int64_t kmin;
            ks2::hypergeometric(R, L.nc - c, t, kmin, hg);
            for (std::size_t i = 0; i < hg.size(); ++i) {
                if (hg[i].hi == 0.0) continue;
                const std::int64_t k = kmin + static_cast<std::int64_t>(i), c2 = c + k;
                const std::int64_t rl = L.lo_at(b + 1, c2), rh = L.hi_at(b + 1, c2);
                if (rl == kNone) continue;
                merge_into(
                    nw[static_cast<std::size_t>(c2)], tmp, src, L.inc(b, c, k), hg[i], L.v - rh, L.v - rl, absorbed
                );
                merged += static_cast<double>(src.size());
            }
            if (merged > budget) return -1.0;
        }
        double held = 0.0;
        for (std::int64_t c = nlo; c <= nhi; ++c) held += static_cast<double>(nw[static_cast<std::size_t>(c)].size());
        if (held > hold) return -1.0;
        lo = nhi + 1, hi = nlo - 1;
        for (std::int64_t c = nlo; c <= nhi; ++c)
            if (!nw[static_cast<std::size_t>(c)].empty()) lo = std::min(lo, c), hi = std::max(hi, c);
        std::swap(w, nw);
        if (lo > hi) break;
    }
    const DD r = absorbed.value();
    return std::clamp(r.hi + r.lo, 0.0, 1.0);
}

}  // namespace sd::htest::cvm2
