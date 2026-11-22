// The shortest-path pieces shared by paths.cpp and yen.cpp (docs/design/graph.md §3.2; milestone GR2).
//
// A distance is the least fixed point of d(v) = min_u fl(d(u) + w(u, v)) (the sources 0): the smallest
// left-to-right sum over the paths to v. Every relaxation algorithm that stops with no edge left to relax
// ends there, whatever its order, so the indexed-heap Dijkstra, the queue Bellman-Ford, the label-correcting
// search on Johnson's potentials and the heap-free dense Dijkstra give the same bits. The predecessors are
// fixed by the distances alone (predecessors()): the tight edges, the fewest edges, the smallest node.
//
// No function here touches a Python object: they run without the GIL.
#pragma once

#include "graph/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::graph::paths {

using std::int64_t;

constexpr double kInf = std::numeric_limits<double>::infinity();

// A graph read for paths: canonical CSR, w null when every edge weighs 1 (unweighted).
struct WGraph {
    int64_t n = 0;
    const int64_t* ip = nullptr;
    const int64_t* ix = nullptr;
    const double* w = nullptr;
    double weight(int64_t k) const { return w ? w[k] : 1.0; }
    int64_t nnz() const { return ip[n]; }
};

// A CSR made here (the undirected reading).
struct Owned {
    std::vector<int64_t> ip, ix;
    std::vector<double> w;
};

// The undirected reading of g (docs/design/graph.md §2.1, §3.1): row v is the union of row v and column v,
// ascending, with undirected_weight of the two directions where both are stored (weights false: the
// pattern only, w left empty).
Owned undirected(const View& g, bool weights);

// The graph a path function reads: g itself (directed, or undirected when g is symmetric: its own
// undirected reading) or its undirected reading kept in `own`.
WGraph reading(const View& g, bool directed, bool weights, Owned& own);

// The weight of the edge u -> v by a binary search of row u; false when there is none.
bool edge_weight(const WGraph& g, int64_t u, int64_t v, double& w);

// mark[v] = 1 for the nodes reachable from the m sources (the pattern), 0 elsewhere.
void reach(const WGraph& g, const int64_t* src, int64_t m, std::vector<uint8_t>& mark);

// The negative-cycle check every method shares (docs/design/graph.md §2.3): a queue Bellman-Ford from a
// virtual root joined by 0-weight edges to the nodes marked in `in` (a set closed under the edges), the
// marked nodes queued in ascending order. Returns -1 when it ends (then h holds the potentials of the
// marked nodes: h(v) <= fl(h(u) + w) on every edge) or the smallest node of the negative cycle it found
// (a negative self loop is one by itself). Deterministic: one serial sweep in a fixed order.
int64_t negative_cycle(const WGraph& g, const std::vector<uint8_t>& in, std::vector<double>& h);

// Scratch of one search, sized for n nodes.
struct Work {
    struct Item {
        double key;
        int64_t v;
    };
    std::vector<uint8_t> flag;
    std::vector<int64_t> order, level, count;
    std::vector<double> key;
    explicit Work(int64_t n)
        : flag(static_cast<size_t>(n)),
          order(static_cast<size_t>(n)),
          level(static_cast<size_t>(n)),
          count(static_cast<size_t>(n)),
          key(static_cast<size_t>(n)),
          hk(static_cast<size_t>(n)),
          hv(static_cast<size_t>(n)),
          pos(static_cast<size_t>(n), -1) {}

    // An indexed 4-ary heap of (key, node), one entry per node at most (a smaller key moves the entry up
    // in place): Dijkstra's queue. pos[v] is v's slot, -1 when v is not in it; every pos is -1 between
    // searches (q_clear empties a search left early).
    std::vector<double> hk;
    std::vector<int64_t> hv, pos;
    int64_t hn = 0;
    bool q_empty() const { return hn == 0; }
    void q_set(int64_t v, double k) {
        int64_t i = pos[static_cast<size_t>(v)];
        if (i < 0) i = hn++;
        while (i > 0) {
            const int64_t p = (i - 1) >> 2;
            if (!(k < hk[static_cast<size_t>(p)])) break;
            place(i, hk[static_cast<size_t>(p)], hv[static_cast<size_t>(p)]);
            i = p;
        }
        place(i, k, v);
    }
    Item q_pop() {
        const Item top{hk[0], hv[0]};
        pos[static_cast<size_t>(top.v)] = -1;
        if (--hn > 0) {
            const double k = hk[static_cast<size_t>(hn)];
            const int64_t v = hv[static_cast<size_t>(hn)];
            int64_t i = 0;
            for (;;) {
                const int64_t c = 4 * i + 1;
                if (c >= hn) break;
                int64_t m = c;
                for (int64_t x = c + 1, e = std::min(c + 4, hn); x < e; ++x)
                    if (hk[static_cast<size_t>(x)] < hk[static_cast<size_t>(m)]) m = x;
                if (!(hk[static_cast<size_t>(m)] < k)) break;
                place(i, hk[static_cast<size_t>(m)], hv[static_cast<size_t>(m)]);
                i = m;
            }
            place(i, k, v);
        }
        return top;
    }
    void q_clear() {
        for (int64_t i = 0; i < hn; ++i) pos[static_cast<size_t>(hv[static_cast<size_t>(i)])] = -1;
        hn = 0;
    }

private:
    void place(int64_t i, double k, int64_t v) {
        hk[static_cast<size_t>(i)] = k;
        hv[static_cast<size_t>(i)] = v;
        pos[static_cast<size_t>(v)] = i;
    }
};

// Distances from the m sources together (each 0) by Dijkstra on the indexed heap (non-negative weights):
// d (n) gets fl sums up to `limit` (inclusive) and inf beyond.
void dijkstra(const WGraph& g, const int64_t* src, int64_t m, double limit, double* d, Work& w);

// Edge counts from the m sources by a breadth-first search, up to `limit`: the unweighted distances.
void bfs(const WGraph& g, const int64_t* src, int64_t m, double limit, double* d, Work& w);

// Distances from s by a queue Bellman-Ford (any weights). False when a node is queued more than n times.
bool bellman_ford(const WGraph& g, int64_t s, double* d, Work& w);

// Distances from s by a Dijkstra on the keys fl(d - h) that queues a node again whenever its distance
// drops (Johnson's potentials order the search only; the distances are the original weights' sums).
// False when a node is taken more than n + 1 times.
bool johnson(const WGraph& g, int64_t s, const double* h, double* d, Work& w);

// The predecessors of docs/design/graph.md §3.2 from the distances d of the m sources: T the tight edges
// (fl(d(u) + w) == d(v), u != v, d(v) finite), h the breadth-first edge count over T from the sources,
// pred(v) the smallest u with (u, v) in T and h(u) = h(v) - 1; kNoNode for the sources and the nodes with
// no finite distance. source_of (when not null) gets the source each predecessor chain ends at.
void predecessors(
    const WGraph& g, const double* d, const int64_t* src, int64_t m, int64_t* pred, int64_t* source_of, Work& w
);

}  // namespace sd::graph::paths
