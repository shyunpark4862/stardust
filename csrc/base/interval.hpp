// The interval search that the array's sd.interp (array/ops/interp.cpp) and sd.interpolate
// (interpolate/search.hpp) share (docs/design/interpolate.md §3.1; decision IP37): the number of nodes at
// or below a point, in nondecreasing finite nodes xp[0 .. n). Both searches below give that one number, so
// the interval of a point never depends on which one ran (nor on the thread that ran it).
//
//   count_le_lanes  branch-free halving, L points side by side: every point of a group halves the same
//                   lengths and their probes are independent loads, so L cache misses are in flight at once
//                   (searchsorted's search, array/ops/searching.cpp)
//   count_le_from   galloping from a count known not to be larger (the previous point's, for sorted
//                   points), then halving over the last doubling
//
// A NaN point counts 0 (no comparison with it holds); the callers give NaN points their own value.
#pragma once

#include <cstdint>

namespace sd {

// out[l] = the number of xp[i] <= v[l], for l < L (n >= 1).
template <int L>
inline void count_le_lanes(const double* xp, std::int64_t n, const double* v, std::int64_t* out) {
    std::int64_t base[L] = {};
    for (std::int64_t len = n; len > 1;) {
        const std::int64_t half = len / 2;
        for (int l = 0; l < L; ++l) base[l] = (xp[base[l] + half] <= v[l]) ? base[l] + half : base[l];
        len -= half;
    }
    for (int l = 0; l < L; ++l) out[l] = base[l] + (xp[base[l]] <= v[l]);
}

// The number of xp[i] <= v, given that it is at least `from` (xp[from - 1] <= v, or from == 0).
inline std::int64_t count_le_from(const double* xp, std::int64_t n, double v, std::int64_t from) {
    std::int64_t lo = from, hi = n;  // the count is in [lo, hi]
    for (std::int64_t step = 1;; step *= 2) {
        const std::int64_t probe = lo + step - 1;
        if (probe >= n) break;
        if (!(xp[probe] <= v)) {
            hi = probe;
            break;
        }
        lo = probe + 1;
    }
    // xp[lo - 1] <= v < xp[hi] (as far as they exist): halve [lo, hi)
    for (std::int64_t len = hi - lo; len > 0;) {
        const std::int64_t half = len / 2;
        const bool below = xp[lo + half] <= v;
        lo = below ? lo + half + 1 : lo;
        len = below ? len - half - 1 : half;
    }
    return lo;
}

// Where the last sorted block ended: its last point and that point's count.
struct GallopState {
    double x = __builtin_nan("");
    std::int64_t c = 0;
};

// c[i] = the number of xp <= v[i] for a block of m >= 1 points. A sorted block (no NaN) gallops from point
// to point, starting from the previous sorted block's last count when it continues that block; any other
// block searches kLanes points side by side. Callers cut the points into blocks by position alone.
inline void count_le_block(
    const double* xp, std::int64_t n, const double* v, std::int64_t m, std::int64_t* c, GallopState& s
) {
    constexpr int kLanes = 8;
    bool sorted = v[0] == v[0];
    for (std::int64_t i = 1; i < m; ++i) sorted &= v[i - 1] <= v[i];  // a NaN breaks it
    if (sorted) {
        std::int64_t from = s.x <= v[0] ? s.c : 0;
        for (std::int64_t i = 0; i < m; ++i) c[i] = from = count_le_from(xp, n, v[i], from);
        s.x = v[m - 1];
        s.c = c[m - 1];
        return;
    }
    std::int64_t i = 0;
    for (; i + kLanes <= m; i += kLanes) count_le_lanes<kLanes>(xp, n, v + i, c + i);
    for (; i < m; ++i) count_le_lanes<1>(xp, n, v + i, c + i);
    s = GallopState{};
}

}  // namespace sd
