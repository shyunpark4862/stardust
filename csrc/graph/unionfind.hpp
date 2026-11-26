// The union-find of the graph area's weak components (docs/design/graph.md §3.3, decision GR31), shared with the
// cluster area's DBSCAN (docs/design/cluster.md §3.6, §6): the root of a set is its smallest node, and a union links
// the larger root under the smaller.
//
//   find(parent, x)               the root of x, halving the path (one thread)
//   unite(parent, a, b)           the union of a's and b's sets (one thread)
//   find_shared(parent, x)        find for tasks running side by side: the reads and the path halving are relaxed
//                                 atomics, and a pointer only ever moves to a smaller node of the same set
//   unite_shared(parent, a, b)    the union for tasks: a root is linked under a smaller root by compare-and-swap
//
// The partition a set of unions defines is unique, so any interleaving of unite_shared ends with it, and the root
// of every set is its smallest node either way.
#pragma once

#include <atomic>
#include <cstdint>
#include <utility>

namespace sd::graph {

inline std::int64_t find(std::int64_t* parent, std::int64_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

inline void unite(std::int64_t* parent, std::int64_t a, std::int64_t b) {
    a = find(parent, a);
    b = find(parent, b);
    if (a == b) return;
    if (a < b) parent[b] = a;
    else parent[a] = b;
}

inline std::int64_t find_shared(std::int64_t* parent, std::int64_t x) {
    for (;;) {
        const std::int64_t p = std::atomic_ref<std::int64_t>(parent[x]).load(std::memory_order_relaxed);
        if (p == x) return x;
        const std::int64_t gp = std::atomic_ref<std::int64_t>(parent[p]).load(std::memory_order_relaxed);
        if (gp != p) std::atomic_ref<std::int64_t>(parent[x]).store(gp, std::memory_order_relaxed);
        x = gp;
    }
}

inline void unite_shared(std::int64_t* parent, std::int64_t a, std::int64_t b) {
    for (;;) {
        a = find_shared(parent, a);
        b = find_shared(parent, b);
        if (a == b) return;
        if (a < b) std::swap(a, b);  // link the larger root a under b
        std::int64_t expected = a;
        if (std::atomic_ref<std::int64_t>(parent[a]).compare_exchange_weak(
                expected, b, std::memory_order_relaxed, std::memory_order_relaxed
            ))
            return;
    }
}

}  // namespace sd::graph
