// What the cluster area's kernels share (docs/design/cluster.md §3.10, §6; decisions CL7, CL43).
//
// - The key order (CL7), the chunk sums (CL48) and the random elements (CL41) are the learning areas' shared
//   machinery in base/keyorder.hpp (moved there by sd.embed's milestone EM0, docs/design/embed.md EM44); the
//   cluster namespace takes them by `using` declarations, so the area's code and bits are unchanged.
// - The condensed index of a pair i < j of n points (scipy's and sd.spatial.pdist's layout).
// - The power of two of decision CL43: data whose largest |x| lies outside [2^-250, 2^250] are scaled into
//   [1, 2) before squared distances are taken (exact), the results scaled back.
// - A union-find whose root is the smallest member.
#pragma once

#include "base/keyorder.hpp"
#include "parallel/pool.hpp"
#include "random/philox.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::cluster {

using std::int64_t;

inline constexpr double kU = 0x1p-53;

using keyorder::chunks_of;
using keyorder::draw;
using keyorder::element;
using keyorder::kChunk;
using keyorder::key_of;
using keyorder::kSplitWork;  // the parallel threshold of API.md §3 (flops)
using keyorder::kStartBits;
using keyorder::lex_order;
using keyorder::pairwise_chunks;
using keyorder::PairwiseStack;
using keyorder::pick;
using keyorder::running_sums;
using keyorder::uniform;
using keyorder::value_key;

inline int64_t cidx(int64_t n, int64_t i, int64_t j) {  // i < j
    return n * i - i * (i + 1) / 2 + (j - i - 1);
}

inline int64_t cpair(int64_t n, int64_t i, int64_t j) { return i < j ? cidx(n, i, j) : cidx(n, j, i); }

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
