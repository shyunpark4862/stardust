// Minimum spanning trees of the cluster area (docs/design/cluster.md §3.2, §3.6, §3.10; decisions CL17, CL18, CL47,
// CL54, CL67): single linkage's (the distances) and HDBSCAN's (the mutual reachability).
//
// An edge (w, a, b), a < b, joins points a and b (their keys: the rows of the rank-ordered data) at the distance
// w of sd.spatial (euclidean: the pair kernel's sum and its careful finish, the bits of pdist). Given core
// distances (core, by key) the weight is HDBSCAN's mutual reachability max(core_a, core_b, d / alpha) instead
// (d / alpha rounded once; d itself when alpha = 1). Edges are ordered by (w, a, b), a total order, so the minimum
// spanning tree is unique and every method below returns it:
//
//   mst_prim          Prim without a matrix: O(n^2 p) time, O(n) memory. Each step offers the last vertex to
//                     every outside vertex (the smaller other end on equal distances) and takes the outside
//                     vertex of smallest edge; the step's scan runs in point tasks fixed by n and p.
//   mst_prim_matrix   the same on a condensed distance vector (the distances as given).
//   mst_boruvka       Boruvka on spatial's KD tree: each round finds every point's smallest edge to another
//                     component and each component takes the smallest of its points'. A point searches the tree
//                     from the root, the nearer child first, and cuts a node when it lies in the point's own
//                     component or when spatial's proven bound puts all its points strictly beyond the point's
//                     radius: the smaller of its own best and its component's best so far (a value other tasks
//                     lower concurrently; it only cuts, and strictly, so no edge and no tie is lost and the edges
//                     chosen do not depend on the timing). The query leaves, in groups fixed by n, are the tasks.
//                     dual = true walks (query node, reference node) pairs instead (Curtin et al.'s dual tree,
//                     McInnes and Healy 2017), kept for tools/cluster_probe.py proto.mst: it measured slower here
//                     (decision CL54).
//   boruvka_shape     the path by shape (CL54, which replaced CL47's draft): Boruvka for p <= 3 and n >= 1024,
//                     p <= 5 and n >= 2048, p = 6 and n >= 4096, p <= 8 and n >= 16384; Prim otherwise.
//
// The mutual reachability prunes more (CL67): an edge of point a weighs at least core_a, a node's points at least
// their smallest core distance (per node), and a distance d > T_d(r) = up(up(r) alpha) has d / alpha > r once
// rounded, so a node is cut when its smallest core distance, the point's own or spatial's bound on d exceeds the
// radius r (strictly, as above).
#pragma once

#include "spatial/kdtree.hpp"

#include <cstdint>
#include <vector>

namespace sd::cluster {

using std::int64_t;

struct Edge {
    double w;
    int64_t a, b;  // a < b
};

inline bool edge_less(const Edge& x, const Edge& y) {
    if (x.w != y.w) return x.w < y.w;
    if (x.a != y.a) return x.a < y.a;
    return x.b < y.b;
}

inline constexpr int64_t kLeafSize = 16;  // the KD tree's leaves (spatial's kd_build)

bool boruvka_shape(int64_t n, int64_t p);
// metric: 0 euclidean, 1 cityblock, 2 chebyshev (spatial's pair kernels); core null: the distances themselves
std::vector<Edge> mst_prim(
    const double* X, int64_t n, int64_t p, const double* core = nullptr, double alpha = 1.0, int metric = 0
);
std::vector<Edge> mst_prim_matrix(const double* D, int64_t n, const double* core = nullptr, double alpha = 1.0);
// Euclidean; dual (the measurement's dual tree) only without core distances
std::vector<Edge> mst_boruvka(
    const spatial::KdTree& T, bool dual = false, const double* core = nullptr, double alpha = 1.0
);

// The mutual reachability's pieces (CL67): d / alpha, and the distance bound T_d(r) above which d / alpha > r.
inline double over_alpha(double d, double alpha) { return alpha == 1.0 ? d : d / alpha; }
double reach_bound(double r, double alpha);

}  // namespace sd::cluster
