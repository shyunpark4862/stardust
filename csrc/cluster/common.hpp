// What the cluster area's kernels share (docs/design/cluster.md §3.10, §6; decisions CL7, CL43).
//
// - The key of decision CL7: the points sorted by their coordinates (the first column first) and then by
//   their index; a point's rank in that order is its key. Every tie rule of the area compares keys, so the
//   same points in another order give the same partition. lex_order() gives order[r] = the input index of
//   rank r (a total order, so std::sort's result is one and the same).
// - The condensed index of a pair i < j of n points (scipy's and sd.spatial.pdist's layout).
// - The power of two of decision CL43: data whose largest |x| lies outside [2^-250, 2^250] are scaled into
//   [1, 2) before squared distances are taken (exact), the results scaled back.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sd::cluster {

using std::int64_t;

inline constexpr double kU = 0x1p-53;
inline constexpr int64_t kSplitWork = int64_t{1} << 15;  // the parallel threshold of API.md §3 (flops)

inline int64_t cidx(int64_t n, int64_t i, int64_t j) {  // i < j
    return n * i - i * (i + 1) / 2 + (j - i - 1);
}

inline int64_t cpair(int64_t n, int64_t i, int64_t j) { return i < j ? cidx(n, i, j) : cidx(n, j, i); }

// order[r]: the input index of the point of rank r (coordinates lexicographically, then the index).
inline std::vector<int64_t> lex_order(const double* X, int64_t n, int64_t p) {
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        const double* x = X + a * p;
        const double* y = X + b * p;
        for (int64_t d = 0; d < p; ++d)
            if (x[d] != y[d]) return x[d] < y[d];
        return a < b;
    });
    return order;
}

// The exponent e of CL43: 0 when max |x| is in [2^-250, 2^250] (or 0), else the e with max |x| 2^e in [1, 2).
inline int scale_exponent(double maxabs) {
    if (maxabs == 0.0 || (maxabs >= 0x1p-250 && maxabs <= 0x1p250)) return 0;
    int e;
    (void)std::frexp(maxabs, &e);  // maxabs = f 2^e, f in [1/2, 1)
    return 1 - e;
}

// A union-find over 0 .. n - 1 whose root is the smallest member (so a cluster's root is its key).
struct MinUnionFind {
    std::vector<int64_t> parent;
    explicit MinUnionFind(int64_t n) : parent(static_cast<size_t>(n)) {
        std::iota(parent.begin(), parent.end(), int64_t{0});
    }
    int64_t find(int64_t x) {
        while (parent[static_cast<size_t>(x)] != x) {
            const int64_t g = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            parent[static_cast<size_t>(x)] = g;
            x = g;
        }
        return x;
    }
    int64_t unite(int64_t a, int64_t b) {  // the roots of two different sets; returns the new root
        if (a > b) std::swap(a, b);
        parent[static_cast<size_t>(b)] = a;
        return a;
    }
};

}  // namespace sd::cluster
