// Density clustering (docs/design/cluster.md §2.4, §3.6, §3.10, §4.4; decisions CL28-CL30, CL67-CL74): DBSCAN,
// HDBSCAN and OPTICS. Points are numbered by their key (decision CL7: the caller passes observations in key order;
// precomputed distances keep the input index), and every tie goes to the smaller key.
//
// The distances: observations with metric 0 euclidean, 1 cityblock or 2 chebyshev take spatial's pair kernel (the
// bits of sd.spatial.cdist) through a KD tree (spatial's, built on the key-ordered rows) or a scan of every pair
// (the tree by shape, kd_shape: CL72);
// precomputed distances (and the observations of any other metric, through sd.spatial.pdist) are a condensed matrix
// in key order. The tree and the scan visit the same pairs with the same kernel, so they give the same bits; the
// path is chosen by shape alone (kd_shape) or forced (path 1 scan, 2 tree) for the tests.
//
//   DBSCAN     the points within eps of a point (d <= eps, the point itself counted; weights summed in key order);
//              core when that count reaches min_samples; the core points' components (a pair of core points within
//              eps joins them) by the union-find of graph/unionfind.hpp, side by side; a border point (within eps of
//              a core point, not core) joins the component of its nearest core point (smallest (d, key); CL28).
//   core       a point's core distance: the k-th smallest distance from it, itself counted (spatial's k nearest,
//              or the k-th of the scanned distances: the same value).
//   HDBSCAN    the minimum spanning tree of the mutual reachability max(core_a, core_b, d / alpha) (mst.cpp:
//              Boruvka on the core distances' KD tree, else Prim; the edge order (w, a, b) of CL50); its single linkage
//              tree by Kruskal in that order (CL67; Z with each merge's clusters (smaller, larger)); the condensed
//              tree with the merges of equal height taken together (Campello et al. 2013; CL68); stabilities,
//              cluster selection ('eom' or 'leaf', cluster_selection_epsilon, max_cluster_size,
//              allow_single_cluster), labels and probabilities by scikit-learn 1.9.1's _tree.pyx on that tree
//              (CL68); GLOSH outlier scores (Campello et al. 2015) and persistence (CL69).
//   OPTICS     the order of scikit-learn's compute_optics_graph: each step takes the unprocessed point of smallest
//              (reachability, key) and, when it is core (core distance <= max_eps), lowers the reachability of the
//              unprocessed points within max_eps to max(core distance, d) when that is strictly smaller (its
//              predecessor then the step's point). One scan of the unprocessed points per step does both. No
//              rounding to 15 decimals (scikit-learn's np.around; CL70).
//   xi         scikit-learn's _xi_cluster and _extract_xi_labels on the reachability plot, the same floating-point
//              operations (CL71); the predecessor correction finds a point's place in O(1).
//
// _core.cluster:
//   dbscan(X, D, eps, min_samples, w, metric, path) -> (labels (n,) cluster slot or -1, core (n,) bool), by key
//   core_distances(X, D, k, metric, path) -> (n,) by key
//   hdbscan(X, D, order, min_samples, alpha, metric, path, min_cluster_size, method, allow_single, epsilon,
//           max_cluster_size)
//       -> (labels (n,) slot or -1 by key, probabilities (n,) by key, outlier scores (n,) by key, Z (n - 1, 4) in
//           the input's numbering, condensed (parent, child, lambda, size) with the children points' input
//           numbers, stability (m,) per condensed cluster, selected (K,) the condensed cluster of each slot,
//           persistence (K,) per slot)
//   hdbscan_cut(Z, cut, min_cluster_size) -> (n,) labels (first-seen numbering, -1 noise) of the merges below cut
//   hdbscan_centres(X, labels, strength, K, metric) -> (centroids (K, p), medoids (K,) keys) of key-ordered data
//       (metric 0, 1, 2 as above; 3 the centroids alone, the medoids -1)
//   weighted_medoid(D, strength) -> the point of smallest sum_j d(i, j) s_j (the smaller on a tie)
//   optics(X, D, min_samples, max_eps, metric, path) -> (ordering keys (n,), reach (n,), core (n,), pred (n,) key
//           or -1), by key
//   optics_xi(reach_plot, pred_plot, ordering, min_samples, min_cluster_size, xi, correction)
//       -> (labels (n,) by input index, clusters (m, 2) positions in the ordering)
//   optics_dbscan(reach, core, ordering, eps) -> (n,) labels by input index (scikit-learn's numbering)
#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "cluster/cluster.hpp"
#include "cluster/common.hpp"
#include "cluster/mst.hpp"
#include "cluster/partition.hpp"
#include "graph/unionfind.hpp"
#include "parallel/pool.hpp"
#include "spatial/kdtree.hpp"
#include "spatial/metric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace sd::cluster {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int64_t kTask = 64;  // points of a task

// The KD tree for observations of the tree's metrics from these shapes (CL72; measured, docs/design/cluster.md §5):
// the radius searches of DBSCAN up to p = 12, the k nearest of the core distances (and HDBSCAN's Boruvka on the same
// tree) up to p = 16, from 64 points.
bool kd_shape(int64_t n, int64_t p, bool knn) { return n >= 64 && p <= (knn ? 16 : 12); }

bool use_tree(int64_t n, int64_t p, int metric, int path, bool matrix, bool knn = false) {
    if (matrix || metric > 2) return false;
    return path == 2 || (path != 1 && kd_shape(n, p, knn));
}

// Runs body(t) over ceil(n / kTask) tasks, split when the work n * cost reaches the parallel threshold.
template <class F>
void for_points(int64_t n, int64_t cost, F&& body) {
    const int64_t nt = (n + kTask - 1) / kTask;
    auto task = [&](int64_t t) { body(t * kTask, std::min(n, (t + 1) * kTask)); };
    if (nt > 1 && n * cost >= kSplitWork) parallel::parallel_for(nt, task, kTask * cost);
    else
        for (int64_t t = 0; t < nt; ++t) task(t);
}

// ---- neighbours ----------------------------------------------------------------------------------------------------

// The points of a KD tree within r of x: f(key, d) for every point with d <= r (d spatial's distance).
template <class Dist>
struct KdNbrs {
    const spatial::KdTree& T;
    Dist E;
    spatial::KdCut cut;
    int64_t p;

    KdNbrs(const spatial::KdTree& t, Dist e, bool cheb) : T(t), E(e), cut(t.m, cheb), p(t.m) {}

    struct Buf {
        std::vector<int64_t> stack;
        std::vector<double> clamp;
        spatial::KdPair pair;
        explicit Buf(int64_t m) : clamp(static_cast<size_t>(m)), pair(m) {}
    };

    bool cut_node(const double* x, int64_t id, double T1, Buf& b) const {
        const double* lo = T.lo(id);
        const double* hi = T.hi(id);
        for (int64_t d = 0; d < p; ++d) b.clamp[static_cast<size_t>(d)] = std::min(std::max(x[d], lo[d]), hi[d]);
        return spatial::kd_beyond(E, x, b.clamp.data(), T1);
    }

    template <class F>
    void visit(const double* x, double r, Buf& b, F&& f) const {
        const double T1 = cut(r);
        b.stack.assign(1, 0);
        while (!b.stack.empty()) {
            const int64_t id = b.stack.back();
            b.stack.pop_back();
            if (cut_node(x, id, T1, b)) continue;
            const spatial::KdNode& nd = T.nodes[static_cast<size_t>(id)];
            if (nd.right >= 0) {
                b.stack.push_back(nd.right);
                b.stack.push_back(id + 1);
                continue;
            }
            const double* y = T.data.data() + nd.start * p;
            for (int64_t q = nd.start; q < nd.end; ++q, y += p) {
                const double s = E.raw(x, y);
                if (Dist::plain(s) && s > Dist::above(r)) continue;
                const double d = E.finish(s, x, y);
                if (d <= r) f(T.idx[static_cast<size_t>(q)], d);
            }
        }
    }

    // The number of points within r: a node whose farthest corner is proven within r counts whole (spatial's far
    // bound, docs/design/spatial.md §3.4).
    int64_t count(const double* x, double r, Buf& b) const {
        const double T1 = cut(r);
        int64_t c = 0;
        b.stack.assign(1, 0);
        while (!b.stack.empty()) {
            const int64_t id = b.stack.back();
            b.stack.pop_back();
            if (cut_node(x, id, T1, b)) continue;
            const spatial::KdNode& nd = T.nodes[static_cast<size_t>(id)];
            b.pair.far(x, x, T.lo(id), T.hi(id), p);
            if (cut(spatial::kd_distance(E, b.pair.a.data(), b.pair.b.data())) <= r) {
                c += nd.end - nd.start;
                continue;
            }
            if (nd.right >= 0) {
                b.stack.push_back(nd.right);
                b.stack.push_back(id + 1);
                continue;
            }
            const double* y = T.data.data() + nd.start * p;
            for (int64_t q = nd.start; q < nd.end; ++q, y += p) {
                const double s = E.raw(x, y);
                if (Dist::plain(s) && s > Dist::above(r)) continue;
                if (E.finish(s, x, y) <= r) ++c;
            }
        }
        return c;
    }
};

// The same by a scan of every point (observations in key order).
template <class Dist>
struct ScanNbrs {
    const double* X;
    int64_t n, p;
    Dist E;
    struct Buf {
        explicit Buf(int64_t) {}
    };
    template <class F>
    void visit(const double* x, double r, Buf&, F&& f) const {
        const double* y = X;
        for (int64_t j = 0; j < n; ++j, y += p) {
            const double s = E.raw(x, y);
            if (Dist::plain(s) && s > Dist::above(r)) continue;
            const double d = E.finish(s, x, y);
            if (d <= r) f(j, d);
        }
    }
    int64_t count(const double* x, double r, Buf& b) const {
        int64_t c = 0;
        visit(x, r, b, [&](int64_t, double) { ++c; });
        return c;
    }
};

// The same on a condensed matrix (the row of point i; x is unused, i comes through `at`).
struct MatNbrs {
    const double* D;
    int64_t n;
    struct Buf {
        explicit Buf(int64_t) {}
    };
    template <class F>
    void visit_row(int64_t i, double r, F&& f) const {
        for (int64_t j = 0; j < n; ++j) {
            const double d = i == j ? 0.0 : D[cpair(n, i, j)];
            if (d <= r) f(j, d);
        }
    }
};

// One interface over the three: visit(i, r, buf, f), count(i, r, buf).
template <class N>
struct PointNbrs {
    const N& nb;
    const double* X;
    int64_t p;
    using Buf = typename N::Buf;
    template <class F>
    void visit(int64_t i, double r, Buf& b, F&& f) const {
        nb.visit(X + i * p, r, b, f);
    }
    int64_t count(int64_t i, double r, Buf& b) const { return nb.count(X + i * p, r, b); }
};

struct RowNbrs {
    const MatNbrs& nb;
    using Buf = MatNbrs::Buf;
    template <class F>
    void visit(int64_t i, double r, Buf&, F&& f) const {
        nb.visit_row(i, r, f);
    }
    int64_t count(int64_t i, double r, Buf&) const {
        int64_t c = 0;
        nb.visit_row(i, r, [&](int64_t, double) { ++c; });
        return c;
    }
};

// Calls f(nbrs, buf_width) with the neighbour finder of the data's path.
template <class F>
void with_nbrs(const double* X, int64_t n, int64_t p, const double* D, int metric, int path, F&& f) {
    if (D) {
        const MatNbrs m{D, n};
        f(RowNbrs{m}, int64_t{1});
        return;
    }
    auto go = [&](auto E, bool cheb) {
        if (use_tree(n, p, metric, path, false)) {
            const spatial::KdTree T = spatial::kd_build(X, n, p, p, kLeafSize);
            using Dist = decltype(E);
            const KdNbrs<Dist> kn(T, E, cheb);
            f(PointNbrs<KdNbrs<Dist>>{kn, X, p}, p);
        } else {
            using Dist = decltype(E);
            const ScanNbrs<Dist> sn{X, n, p, E};
            f(PointNbrs<ScanNbrs<Dist>>{sn, X, p}, p);
        }
    };
    if (metric == 1) go(spatial::KdCity{p}, false);
    else if (metric == 2) go(spatial::KdCheb{p}, true);
    else go(spatial::KdEuclid{p}, false);
}

// The cost of one point's search in pair-terms (for the split): a tree search about 32 pairs, a scan n.
int64_t search_cost(int64_t n, int64_t p, int metric, int path, bool matrix) {
    if (matrix) return n;
    return use_tree(n, p, metric, path, false) ? 32 * (p + 2) : n * (p + 2);
}

// ---- DBSCAN --------------------------------------------------------------------------------------------------------

struct Dbscan {
    std::vector<int64_t> label;  // cluster slot (0 .. K - 1 by the components' smallest keys) or -1
    std::vector<char> core;
};

template <class Nb>
Dbscan dbscan_run(const Nb& nb, int64_t n, int64_t width, int64_t cost, double eps, double ms, const double* w) {
    Dbscan out;
    out.core.assign(static_cast<size_t>(n), 0);
    // 1. the core points: the count (weights: their sum in key order) within eps
    for_points(n, cost, [&](int64_t i0, int64_t i1) {
        typename Nb::Buf b(width);
        std::vector<int64_t> keys;
        for (int64_t i = i0; i < i1; ++i) {
            if (!w) {
                out.core[static_cast<size_t>(i)] = static_cast<double>(nb.count(i, eps, b)) >= ms;
                continue;
            }
            keys.clear();
            nb.visit(i, eps, b, [&](int64_t j, double) { keys.push_back(j); });
            std::sort(keys.begin(), keys.end());
            double s = 0.0;
            for (int64_t j : keys) s += w[j];
            out.core[static_cast<size_t>(i)] = s >= ms;
        }
    });
    // 2. the components of the core points (each pair once, from its smaller key)
    std::vector<int64_t> parent(static_cast<size_t>(n));
    std::iota(parent.begin(), parent.end(), int64_t{0});
    for_points(n, cost, [&](int64_t i0, int64_t i1) {
        typename Nb::Buf b(width);
        for (int64_t i = i0; i < i1; ++i) {
            if (!out.core[static_cast<size_t>(i)]) continue;
            nb.visit(i, eps, b, [&](int64_t j, double) {
                if (j > i && out.core[static_cast<size_t>(j)]) graph::unite_shared(parent.data(), i, j);
            });
        }
    });
    // slots by the components' smallest keys
    out.label.assign(static_cast<size_t>(n), -1);
    std::vector<int64_t> slot(static_cast<size_t>(n), -1);
    int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!out.core[static_cast<size_t>(i)]) continue;
        const int64_t r = graph::find(parent.data(), i);
        if (slot[static_cast<size_t>(r)] < 0) slot[static_cast<size_t>(r)] = k++;
        out.label[static_cast<size_t>(i)] = slot[static_cast<size_t>(r)];
    }
    // 3. a border point joins its nearest core point's cluster (smallest (d, key); CL28)
    for_points(n, cost, [&](int64_t i0, int64_t i1) {
        typename Nb::Buf b(width);
        for (int64_t i = i0; i < i1; ++i) {
            if (out.core[static_cast<size_t>(i)]) continue;
            double bd = kInf;
            int64_t bj = -1;
            nb.visit(i, eps, b, [&](int64_t j, double d) {
                if (!out.core[static_cast<size_t>(j)]) return;
                if (d < bd || (d == bd && j < bj)) bd = d, bj = j;
            });
            if (bj >= 0) out.label[static_cast<size_t>(i)] = out.label[static_cast<size_t>(bj)];
        }
    });
    return out;
}

// ---- core distances ------------------------------------------------------------------------------------------------

template <class Dist>
std::vector<double> core_tree(const spatial::KdTree& T, const Dist& E, bool cheb, const double* X, int64_t k) {
    const int64_t n = T.n, p = T.m;
    std::vector<double> core(static_cast<size_t>(n));
    const spatial::KdCut cut(p, cheb);
    for_points(n, (k + 32) * (p + 2), [&](int64_t i0, int64_t i1) {
        spatial::KdSearch<Dist> s(T, E, cut, k);
        std::vector<std::pair<double, int64_t>> out;
        for (int64_t i = i0; i < i1; ++i) {
            s.run(X + i * p, kInf, out);
            core[static_cast<size_t>(i)] = out[static_cast<size_t>(k - 1)].first;
        }
    });
    return core;
}

template <class Dist>
std::vector<double> core_scan(const double* X, int64_t n, int64_t p, const Dist& E, int64_t k) {
    std::vector<double> core(static_cast<size_t>(n));
    for_points(n, n * (p + 2), [&](int64_t i0, int64_t i1) {
        std::vector<double> d(static_cast<size_t>(n));
        for (int64_t i = i0; i < i1; ++i) {
            const double* x = X + i * p;
            for (int64_t j = 0; j < n; ++j) d[static_cast<size_t>(j)] = spatial::kd_distance(E, x, X + j * p);
            std::nth_element(d.begin(), d.begin() + (k - 1), d.end());
            core[static_cast<size_t>(i)] = d[static_cast<size_t>(k - 1)];
        }
    });
    return core;
}

std::vector<double> core_matrix(const double* D, int64_t n, int64_t k) {
    std::vector<double> core(static_cast<size_t>(n));
    for_points(n, n, [&](int64_t i0, int64_t i1) {
        std::vector<double> d(static_cast<size_t>(n));
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < n; ++j) d[static_cast<size_t>(j)] = i == j ? 0.0 : D[cpair(n, i, j)];
            std::nth_element(d.begin(), d.begin() + (k - 1), d.end());
            core[static_cast<size_t>(i)] = d[static_cast<size_t>(k - 1)];
        }
    });
    return core;
}

// The core distances of the data's path; T is built (and kept for the caller) when the path is the tree.
std::vector<double> core_distances(
    const double* X, int64_t n, int64_t p, const double* D, int metric, int path, int64_t k, spatial::KdTree& T
) {
    if (D) return core_matrix(D, n, k);
    if (use_tree(n, p, metric, path, false, true)) {
        T = spatial::kd_build(X, n, p, p, kLeafSize);
        if (metric == 1) return core_tree(T, spatial::KdCity{p}, false, X, k);
        if (metric == 2) return core_tree(T, spatial::KdCheb{p}, true, X, k);
        return core_tree(T, spatial::KdEuclid{p}, false, X, k);
    }
    if (metric == 1) return core_scan(X, n, p, spatial::KdCity{p}, k);
    if (metric == 2) return core_scan(X, n, p, spatial::KdCheb{p}, k);
    return core_scan(X, n, p, spatial::KdEuclid{p}, k);
}

// ---- HDBSCAN -------------------------------------------------------------------------------------------------------

// The single linkage tree of the sorted edges (Kruskal; CL67): row i merges the clusters of edge i, (smaller,
// larger) cluster numbers, the points 0 .. n - 1 by key and merge i n + i.
struct Tree {
    std::vector<int64_t> left, right, size;
    std::vector<double> h;
};

Tree kruskal(const std::vector<Edge>& E, int64_t n) {
    Tree t;
    const int64_t m = static_cast<int64_t>(E.size());
    t.left.resize(static_cast<size_t>(m));
    t.right.resize(static_cast<size_t>(m));
    t.size.resize(static_cast<size_t>(m));
    t.h.resize(static_cast<size_t>(m));
    MinUnionFind uf(n);
    std::vector<int64_t> id(static_cast<size_t>(n)), sz(static_cast<size_t>(n), 1);
    std::iota(id.begin(), id.end(), int64_t{0});
    for (int64_t i = 0; i < m; ++i) {
        const Edge& e = E[static_cast<size_t>(i)];
        const int64_t a = uf.find(e.a), b = uf.find(e.b);
        const int64_t ca = id[static_cast<size_t>(a)], cb = id[static_cast<size_t>(b)];
        t.left[static_cast<size_t>(i)] = std::min(ca, cb);
        t.right[static_cast<size_t>(i)] = std::max(ca, cb);
        t.h[static_cast<size_t>(i)] = e.w;
        const int64_t s = sz[static_cast<size_t>(a)] + sz[static_cast<size_t>(b)];
        t.size[static_cast<size_t>(i)] = s;
        const int64_t r = uf.unite(a, b);
        id[static_cast<size_t>(r)] = n + i;
        sz[static_cast<size_t>(r)] = s;
    }
    return t;
}

// The condensed tree (Campello et al. 2013; scikit-learn's _condense_tree on a tree without equal heights; CL68):
// from the root down, a cluster at a merge of height h takes at once every merge of that same height below it
// (Campello's edges of equal weight removed together), which leaves it in components (the subtrees below h, by
// their smallest key). Components of min_cluster_size or more points are the significant ones: two or more of them
// become new clusters (numbered on from n + 1 in the order they are made, the root n), exactly one carries the
// cluster on, and the points of the others leave the cluster at lambda = 1 / h (inf at h = 0), each component's
// points breadth first. Without equal heights this is scikit-learn's procedure; with them scikit-learn's result
// depends on the order its minimum spanning tree met the equal edges, and this one does not.
struct Condensed {
    std::vector<int64_t> parent, child, size;
    std::vector<double> lambda;
    int64_t nclusters = 0;  // the clusters n .. n + nclusters - 1
};

Condensed condense(const Tree& t, int64_t n, int64_t mcs) {
    Condensed c;
    const int64_t root = 2 * (n - 1);
    auto count = [&](int64_t v) { return v >= n ? t.size[static_cast<size_t>(v - n)] : int64_t{1}; };
    auto height = [&](int64_t v) { return v >= n ? t.h[static_cast<size_t>(v - n)] : -kInf; };
    // the smallest key below each node (the merges come after their children)
    std::vector<int64_t> minkey(static_cast<size_t>(root + 1));
    for (int64_t v = 0; v < n; ++v) minkey[static_cast<size_t>(v)] = v;
    for (int64_t i = 0; i < n - 1; ++i)
        minkey[static_cast<size_t>(n + i)] = std::min(
            minkey[static_cast<size_t>(t.left[static_cast<size_t>(i)])],
            minkey[static_cast<size_t>(t.right[static_cast<size_t>(i)])]
        );
    std::vector<int64_t> sub, group, frontier;
    auto push = [&](int64_t par, int64_t ch, double lam, int64_t s) {
        c.parent.push_back(par);
        c.child.push_back(ch);
        c.lambda.push_back(lam);
        c.size.push_back(s);
    };
    auto drop = [&](int64_t v, int64_t par, double lam) {  // v's points, breadth first
        sub.assign(1, v);
        for (size_t at = 0; at < sub.size(); ++at) {
            const int64_t u = sub[at];
            if (u < n) {
                push(par, u, lam, 1);
            } else {
                sub.push_back(t.left[static_cast<size_t>(u - n)]);
                sub.push_back(t.right[static_cast<size_t>(u - n)]);
            }
        }
    };
    std::vector<std::pair<int64_t, int64_t>> queue{{root, n}};  // (merge, its cluster)
    int64_t next = n + 1;
    for (size_t at = 0; at < queue.size(); ++at) {
        const auto [node, me] = queue[at];
        const double h = height(node);
        const double lam = h > 0.0 ? 1.0 / h : kInf;
        group.assign(1, node);
        frontier.clear();
        for (size_t g = 0; g < group.size(); ++g) {
            const int64_t v = group[g];
            for (const int64_t ch : {t.left[static_cast<size_t>(v - n)], t.right[static_cast<size_t>(v - n)]}) {
                if (height(ch) == h) group.push_back(ch);
                else frontier.push_back(ch);
            }
        }
        std::sort(frontier.begin(), frontier.end(), [&](int64_t a, int64_t b) {
            return minkey[static_cast<size_t>(a)] < minkey[static_cast<size_t>(b)];
        });
        int64_t big = 0;
        for (int64_t f : frontier) big += count(f) >= mcs;
        for (int64_t f : frontier) {
            if (count(f) < mcs) {
                drop(f, me, lam);
            } else if (big >= 2) {
                const int64_t lab = next++;
                push(me, lab, lam, count(f));
                queue.emplace_back(f, lab);
            } else {
                queue.emplace_back(f, me);
            }
        }
    }
    c.nclusters = next - n;
    return c;
}


struct Hdbscan {
    std::vector<int64_t> label;  // slot or -1, by key
    std::vector<double> prob, outlier;
    std::vector<double> stability;  // per condensed cluster
    std::vector<int64_t> selected;  // per slot: the condensed cluster (n + c)
    std::vector<double> persistence;
};

// scikit-learn's _compute_stability, _get_clusters, _do_labelling and get_probabilities on the condensed tree
// (CL68), with GLOSH and persistence (CL69).
Hdbscan select(const Condensed& c, int64_t n, int method, bool allow_single, double eps, int64_t max_size) {
    Hdbscan out;
    const int64_t m = c.nclusters;
    const int64_t rows = static_cast<int64_t>(c.parent.size());
    auto ci = [&](int64_t v) { return static_cast<size_t>(v - n); };
    // births, the row of each child, the stabilities in row order
    std::vector<double> birth(static_cast<size_t>(m), 0.0);
    std::vector<int64_t> row_of_cluster(static_cast<size_t>(m), -1), row_of_point(static_cast<size_t>(n), -1);
    for (int64_t q = 0; q < rows; ++q) {
        const int64_t ch = c.child[static_cast<size_t>(q)];
        if (ch >= n) {
            birth[ci(ch)] = c.lambda[static_cast<size_t>(q)];
            row_of_cluster[ci(ch)] = q;
        } else {
            row_of_point[static_cast<size_t>(ch)] = q;
        }
    }
    std::vector<double> stab(static_cast<size_t>(m), 0.0);
    for (int64_t q = 0; q < rows; ++q) {
        const int64_t par = c.parent[static_cast<size_t>(q)];
        stab[ci(par)] +=
            (c.lambda[static_cast<size_t>(q)] - birth[ci(par)]) * static_cast<double>(c.size[static_cast<size_t>(q)]);
    }
    out.stability = stab;
    // the cluster rows (cluster_tree) and each cluster's cluster children in row order
    std::vector<std::vector<int64_t>> kids(static_cast<size_t>(m));
    std::vector<int64_t> csize(static_cast<size_t>(m), 0);
    for (int64_t q = 0; q < rows; ++q) {
        const int64_t ch = c.child[static_cast<size_t>(q)];
        if (c.size[static_cast<size_t>(q)] > 1) {
            kids[ci(c.parent[static_cast<size_t>(q)])].push_back(ch);
            csize[ci(ch)] = c.size[static_cast<size_t>(q)];
        }
    }
    const int64_t rootc = n;
    if (allow_single) {
        int64_t s = 0;
        for (int64_t ch : kids[0]) s += csize[ci(ch)];
        csize[0] = s;
    }
    const int64_t maxc = max_size < 0 ? n + 1 : max_size;
    // is_cluster over scikit-learn's node list (every cluster but the root unless allow_single)
    std::vector<char> in_list(static_cast<size_t>(m), 1), is_cluster(static_cast<size_t>(m), 1);
    if (!allow_single) in_list[0] = 0, is_cluster[0] = 0;
    auto descendants = [&](int64_t v, std::vector<int64_t>& outv) {  // v's cluster subtree, v first
        outv.assign(1, v);
        for (size_t at = 0; at < outv.size(); ++at)
            for (int64_t ch : kids[ci(outv[at])]) outv.push_back(ch);
    };
    std::vector<int64_t> sub;
    bool any_rows = false;
    for (int64_t v = 0; v < m; ++v) any_rows = any_rows || !kids[static_cast<size_t>(v)].empty();
    // a cluster's birth epsilon: 1 / the lambda of its row
    auto birth_eps = [&](int64_t v) { return 1.0 / birth[ci(v)]; };
    auto parent_of = [&](int64_t v) { return c.parent[static_cast<size_t>(row_of_cluster[ci(v)])]; };
    auto traverse_up = [&](int64_t leaf) {
        for (;;) {
            const int64_t par = parent_of(leaf);
            if (par == rootc) return allow_single ? par : leaf;
            if (birth_eps(par) > eps) return par;
            leaf = par;
        }
    };
    auto epsilon_search = [&](const std::vector<int64_t>& leaves) {  // ascending cluster numbers
        std::vector<char> processed(static_cast<size_t>(m), 0), chosen(static_cast<size_t>(m), 0);
        for (int64_t leaf : leaves) {
            if (birth_eps(leaf) < eps) {
                if (processed[ci(leaf)]) continue;
                const int64_t e = traverse_up(leaf);
                chosen[ci(e)] = 1;
                descendants(e, sub);
                for (size_t a = 1; a < sub.size(); ++a) processed[ci(sub[a])] = 1;
            } else {
                chosen[ci(leaf)] = 1;
            }
        }
        return chosen;
    };
    if (method == 0) {  // eom
        for (int64_t v = m - 1; v >= 0; --v) {
            if (!in_list[static_cast<size_t>(v)]) continue;
            double sub_stab = 0.0;
            for (int64_t ch : kids[static_cast<size_t>(v)]) sub_stab += stab[ci(ch)];
            if (sub_stab > stab[static_cast<size_t>(v)] || csize[static_cast<size_t>(v)] > maxc) {
                is_cluster[static_cast<size_t>(v)] = 0;
                stab[static_cast<size_t>(v)] = sub_stab;
            } else {
                descendants(n + v, sub);
                for (size_t a = 1; a < sub.size(); ++a) is_cluster[ci(sub[a])] = 0;
            }
        }
        if (eps != 0.0 && any_rows) {
            std::vector<int64_t> eom;
            for (int64_t v = 0; v < m; ++v)
                if (in_list[static_cast<size_t>(v)] && is_cluster[static_cast<size_t>(v)]) eom.push_back(n + v);
            std::vector<char> chosen(static_cast<size_t>(m), 0);
            if (eom.size() == 1 && eom[0] == rootc) {
                if (allow_single) chosen[0] = 1;
            } else {
                chosen = epsilon_search(eom);
            }
            for (int64_t v = 0; v < m; ++v)
                if (in_list[static_cast<size_t>(v)])
                    is_cluster[static_cast<size_t>(v)] = chosen[static_cast<size_t>(v)];
        }
    } else {  // leaf
        std::vector<int64_t> leaves;
        if (any_rows) {
            descendants(rootc, sub);
            for (int64_t v : sub)
                if (v != rootc && kids[ci(v)].empty()) leaves.push_back(v);
            std::sort(leaves.begin(), leaves.end());
        }
        // no leaves: scikit-learn marks the root, then keeps only the selected leaves, so nothing is selected
        std::vector<char> chosen(static_cast<size_t>(m), 0);
        if (eps != 0.0) chosen = epsilon_search(leaves);
        else
            for (int64_t v : leaves) chosen[ci(v)] = 1;
        for (int64_t v = 0; v < m; ++v)
            if (in_list[static_cast<size_t>(v)]) is_cluster[static_cast<size_t>(v)] = chosen[static_cast<size_t>(v)];
    }
    std::vector<char> sel(static_cast<size_t>(m), 0);
    int64_t nsel = 0;
    for (int64_t v = 0; v < m; ++v)
        if (in_list[static_cast<size_t>(v)] && is_cluster[static_cast<size_t>(v)])
            sel[static_cast<size_t>(v)] = 1, ++nsel;
    // the labels: a point's nearest selected ancestor; under the root alone the single cluster's threshold
    std::vector<int64_t> slot(static_cast<size_t>(m), -1);
    int64_t k = 0;
    for (int64_t v = 0; v < m; ++v)
        if (sel[static_cast<size_t>(v)]) slot[static_cast<size_t>(v)] = k++, out.selected.push_back(n + v);
    double thresh = 0.0;
    if (nsel == 1 && allow_single) {
        if (eps != 0.0) {
            thresh = 1.0 / eps;
        } else {
            thresh = -kInf;
            for (int64_t q = 0; q < rows; ++q)
                if (c.parent[static_cast<size_t>(q)] == rootc)
                    thresh = std::max(thresh, c.lambda[static_cast<size_t>(q)]);
        }
    }
    // deaths: the largest lambda of a cluster's own rows; GLOSH's: of its whole subtree
    std::vector<double> death(static_cast<size_t>(m), 0.0);
    for (int64_t q = 0; q < rows; ++q) {
        double& d = death[ci(c.parent[static_cast<size_t>(q)])];
        d = std::max(d, c.lambda[static_cast<size_t>(q)]);
    }
    std::vector<double> deep(death);
    for (int64_t v = m - 1; v >= 1; --v) {  // children are numbered after their parents
        double& d = deep[ci(parent_of(n + v))];
        d = std::max(d, deep[static_cast<size_t>(v)]);
    }
    out.label.assign(static_cast<size_t>(n), -1);
    out.prob.assign(static_cast<size_t>(n), 0.0);
    out.outlier.assign(static_cast<size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t q = row_of_point[static_cast<size_t>(i)];
        const double lam = c.lambda[static_cast<size_t>(q)];
        const int64_t par = c.parent[static_cast<size_t>(q)];
        int64_t v = par;
        while (v != rootc && !sel[ci(v)]) v = parent_of(v);
        int64_t lab = -1;
        if (v != rootc) {
            lab = slot[ci(v)];
        } else if (nsel == 1 && allow_single && sel[0]) {
            if (lam >= thresh) lab = slot[0];
        }
        out.label[static_cast<size_t>(i)] = lab;
        if (lab >= 0) {
            const double mx = death[ci(v)];
            out.prob[static_cast<size_t>(i)] = (mx == 0.0 || std::isinf(lam)) ? 1.0 : std::min(lam, mx) / mx;
        }
        // GLOSH: 1 - lambda / lambda_max of the cluster the point leaves (0 at lambda inf; 1 when lambda_max is
        // inf and lambda is not)
        const double lm = deep[ci(par)];
        double g;
        if (std::isinf(lam) || lm == 0.0) g = 0.0;
        else if (std::isinf(lm)) g = 1.0;
        else g = (lm - lam) / lm;
        out.outlier[static_cast<size_t>(i)] = g;
    }
    // persistence: the stability over (size times the largest lambda of the tree); 1 when that is 0 or inf
    double lmax = 0.0;
    for (double v : c.lambda) lmax = std::max(lmax, v);
    std::vector<int64_t> count(static_cast<size_t>(k), 0);
    for (int64_t v : out.label)
        if (v >= 0) ++count[static_cast<size_t>(v)];
    out.persistence.resize(static_cast<size_t>(k));
    for (int64_t s = 0; s < k; ++s) {
        const int64_t v = out.selected[static_cast<size_t>(s)];
        const double cs = static_cast<double>(count[static_cast<size_t>(s)]);
        out.persistence[static_cast<size_t>(s)] =
            (std::isinf(lmax) || lmax == 0.0 || cs == 0.0) ? 1.0 : stab[ci(v)] / (cs * lmax);
    }
    return out;
}

// ---- OPTICS --------------------------------------------------------------------------------------------------------

struct Optics {
    std::vector<int64_t> ordering, pred;
    std::vector<double> reach, core;
};

// The order of scikit-learn's compute_optics_graph (CL30, CL70). dle(i, j, T, d) says whether d(i, j) <= T and
// sets d when it is.
template <class DLe>
Optics optics_run(int64_t n, int64_t cost, std::vector<double> core, double max_eps, const DLe& dle) {
    Optics o;
    for (double& c : core)
        if (c > max_eps) c = kInf;
    o.core = core;
    o.reach.assign(static_cast<size_t>(n), kInf);
    o.pred.assign(static_cast<size_t>(n), -1);
    o.ordering.reserve(static_cast<size_t>(n));
    std::vector<int64_t> open(static_cast<size_t>(n)), at(static_cast<size_t>(n));  // the unprocessed points (any
    std::iota(open.begin(), open.end(), int64_t{0});  // order) and their places
    std::iota(at.begin(), at.end(), int64_t{0});
    const int64_t task = std::max<int64_t>(64, (int64_t{1} << 12) / std::max<int64_t>(cost, 1));
    std::vector<int64_t> ti;
    auto better = [&](int64_t a, int64_t b) {  // (reach, key) of a below b's
        const double ra = o.reach[static_cast<size_t>(a)], rb = o.reach[static_cast<size_t>(b)];
        return ra < rb || (ra == rb && a < b);
    };
    int64_t next = 0;  // the first step takes key 0 (every reachability inf)
    while (true) {
        const int64_t pt = next;
        o.ordering.push_back(pt);
        // remove pt from the open list
        const int64_t last = open.back();
        open[static_cast<size_t>(at[static_cast<size_t>(pt)])] = last;
        at[static_cast<size_t>(last)] = at[static_cast<size_t>(pt)];
        open.pop_back();
        const int64_t m = static_cast<int64_t>(open.size());
        if (m == 0) break;
        const double cp = core[static_cast<size_t>(pt)];
        const bool upd = cp != kInf;
        const int64_t nt = (m + task - 1) / task;
        ti.assign(static_cast<size_t>(nt), -1);
        auto body = [&](int64_t t) {
            int64_t best = -1;
            for (int64_t a = t * task; a < std::min(m, (t + 1) * task); ++a) {
                const int64_t j = open[static_cast<size_t>(a)];
                if (upd) {
                    double& rj = o.reach[static_cast<size_t>(j)];
                    if (rj > cp) {  // else max(d, cp) >= rj: not lowered
                        double d;
                        if (dle(pt, j, std::min(rj, max_eps), d)) {
                            const double rd = std::max(d, cp);
                            if (rd < rj) rj = rd, o.pred[static_cast<size_t>(j)] = pt;
                        }
                    }
                }
                if (best < 0 || better(j, best)) best = j;
            }
            ti[static_cast<size_t>(t)] = best;
        };
        if (nt > 1 && m * (upd ? cost : 1) >= kSplitWork) parallel::parallel_for(nt, body, task * cost);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
        int64_t best = -1;
        for (int64_t t = 0; t < nt; ++t) {
            const int64_t j = ti[static_cast<size_t>(t)];
            if (j >= 0 && (best < 0 || better(j, best))) best = j;
        }
        next = best;
    }
    return o;
}

// scikit-learn's _xi_cluster (CL71): the same operations on the plot (reach_plot with inf appended).
struct XiOut {
    std::vector<int64_t> labels;  // by input index
    std::vector<int64_t> clusters;  // (start, end) pairs
};

XiOut optics_xi(
    const double* rp0,
    const int64_t* pp,
    const int64_t* ordering,
    int64_t n,
    int64_t min_samples,
    int64_t mcs,
    double xi,
    bool correction
) {
    std::vector<double> rp(rp0, rp0 + n);
    rp.push_back(kInf);
    const double xc = 1.0 - xi;
    const double inv = 1.0 / xc;
    std::vector<char> up_s(static_cast<size_t>(n)), down_s(static_cast<size_t>(n)), down(static_cast<size_t>(n)),
        up(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double r = rp[static_cast<size_t>(i)] / rp[static_cast<size_t>(i + 1)];
        up_s[static_cast<size_t>(i)] = r <= xc;
        down_s[static_cast<size_t>(i)] = r >= inv;
        down[static_cast<size_t>(i)] = r > 1.0;
        up[static_cast<size_t>(i)] = r < 1.0;
    }
    std::vector<int64_t> pos(static_cast<size_t>(n));  // the place of each input index in the ordering
    for (int64_t i = 0; i < n; ++i) pos[static_cast<size_t>(ordering[i])] = i;
    auto extend = [&](const std::vector<char>& steep, const std::vector<char>& xward, int64_t start) {
        int64_t non = 0, end = start;
        for (int64_t index = start; index < n; ++index) {
            if (steep[static_cast<size_t>(index)]) {
                non = 0;
                end = index;
            } else if (!xward[static_cast<size_t>(index)]) {
                if (++non > min_samples) break;
            } else {
                return end;
            }
        }
        return end;
    };
    struct Sda {
        int64_t start, end;
        double mib;
    };
    std::vector<Sda> sdas;
    auto filter = [&](double mib) {
        if (std::isinf(mib)) {
            sdas.clear();
            return;
        }
        std::vector<Sda> res;
        for (const Sda& s : sdas)
            if (mib <= rp[static_cast<size_t>(s.start)] * xc) res.push_back(s);
        for (Sda& s : res) s.mib = std::max(s.mib, mib);
        sdas.swap(res);
    };
    auto correct = [&](int64_t s, int64_t e, int64_t& os, int64_t& oe) {
        while (s < e) {
            if (rp[static_cast<size_t>(s)] > rp[static_cast<size_t>(e)]) {
                os = s, oe = e;
                return true;
            }
            const int64_t pe = pp[e];
            if (pe >= 0) {
                const int64_t at = pos[static_cast<size_t>(pe)];
                if (at >= s && at < e) {
                    os = s, oe = e;
                    return true;
                }
            }
            --e;
        }
        return false;
    };
    XiOut out;
    int64_t index = 0;
    double mib = 0.0;
    for (int64_t si = 0; si < n; ++si) {
        if (!(up_s[static_cast<size_t>(si)] || down_s[static_cast<size_t>(si)])) continue;
        if (si < index) continue;
        double mx = rp[static_cast<size_t>(index)];
        for (int64_t a = index + 1; a <= si; ++a) mx = std::max(mx, rp[static_cast<size_t>(a)]);
        mib = std::max(mib, mx);
        if (down_s[static_cast<size_t>(si)]) {
            filter(mib);
            const int64_t d_end = extend(down_s, up, si);
            sdas.push_back({si, d_end, 0.0});
            index = d_end + 1;
            mib = rp[static_cast<size_t>(index)];
        } else {
            filter(mib);
            const int64_t u_start = si;
            const int64_t u_end = extend(up_s, down, u_start);
            index = u_end + 1;
            mib = rp[static_cast<size_t>(index)];
            std::vector<std::pair<int64_t, int64_t>> uc;
            for (const Sda& D : sdas) {
                int64_t cs = D.start, ce = u_end;
                if (rp[static_cast<size_t>(ce + 1)] * xc < D.mib) continue;
                const double dmax = rp[static_cast<size_t>(D.start)];
                if (dmax * xc >= rp[static_cast<size_t>(ce + 1)]) {
                    while (rp[static_cast<size_t>(cs + 1)] > rp[static_cast<size_t>(ce + 1)] && cs < D.end) ++cs;
                } else if (rp[static_cast<size_t>(ce + 1)] * xc >= dmax) {
                    while (ce > u_start && rp[static_cast<size_t>(ce - 1)] > dmax) --ce;
                }
                if (correction) {
                    int64_t a, b;
                    if (!correct(cs, ce, a, b)) continue;
                    cs = a, ce = b;
                }
                if (ce - cs + 1 < mcs) continue;
                if (cs > D.end) continue;
                if (ce < u_start) continue;
                uc.emplace_back(cs, ce);
            }
            for (auto it = uc.rbegin(); it != uc.rend(); ++it) {
                out.clusters.push_back(it->first);
                out.clusters.push_back(it->second);
            }
        }
    }
    // _extract_xi_labels
    std::vector<int64_t> lab(static_cast<size_t>(n), -1);
    int64_t label = 0;
    for (size_t c = 0; c < out.clusters.size(); c += 2) {
        const int64_t a = out.clusters[c], b = out.clusters[c + 1];
        bool free = true;
        for (int64_t i = a; i <= b && free; ++i) free = lab[static_cast<size_t>(i)] == -1;
        if (!free) continue;
        for (int64_t i = a; i <= b; ++i) lab[static_cast<size_t>(i)] = label;
        ++label;
    }
    out.labels.assign(static_cast<size_t>(n), -1);
    for (int64_t i = 0; i < n; ++i) out.labels[static_cast<size_t>(ordering[i])] = lab[static_cast<size_t>(i)];
    return out;
}

// ---- bindings ------------------------------------------------------------------------------------------------------

struct Input {
    const double* X = nullptr;
    const double* D = nullptr;
    int64_t n = 0, p = 0;
};

Input input_of(nb::handle Xo, nb::handle Do) {
    Input in;
    if (!Xo.is_none()) {
        in.X = matrix_arg(Xo, "X", in.n, in.p);
        return in;
    }
    int64_t m;
    in.D = vector_arg(Do, "D", m);
    in.n = (1 + static_cast<int64_t>(std::llround(std::sqrt(1.0 + 8.0 * static_cast<double>(m))))) / 2;
    if (in.n * (in.n - 1) / 2 != m) throw nb::value_error("D: condensed");
    return in;
}

NDArray bool_array(const std::vector<char>& v) {
    const int64_t shape[1] = {static_cast<int64_t>(v.size())};
    NDArray a = new_array(DType::Bool, shape, false);
    std::copy(v.begin(), v.end(), reinterpret_cast<char*>(a.data));
    return a;
}

nb::object dbscan_py(nb::handle Xo, nb::handle Do, double eps, double ms, nb::handle wo, int metric, int path) {
    const Input in = input_of(Xo, Do);
    const double* w = nullptr;
    if (!wo.is_none()) {
        int64_t m;
        w = vector_arg(wo, "w", m);
        if (m != in.n) throw nb::value_error("w");
    }
    Dbscan r;
    {
        const int64_t cost = search_cost(in.n, in.p, metric, path, in.D != nullptr);
        const parallel::Unlocked unlocked(in.n * cost >= kSplitWork);
        with_nbrs(in.X, in.n, in.p, in.D, metric, path, [&](const auto& nb, int64_t width) {
            r = dbscan_run(nb, in.n, width, cost, eps, ms, w);
        });
    }
    return nb::make_tuple(wrap(int_array(r.label, in.n)), wrap(bool_array(r.core)));
}

std::vector<double> core_of(const Input& in, int metric, int path, int64_t k, spatial::KdTree& T) {
    return core_distances(in.X, in.n, in.p, in.D, metric, path, k, T);
}

nb::object core_distances_py(nb::handle Xo, nb::handle Do, int64_t k, int metric, int path) {
    const Input in = input_of(Xo, Do);
    if (k < 1 || k > in.n) throw nb::value_error("k");
    std::vector<double> core;
    {
        const parallel::Unlocked unlocked(in.n * search_cost(in.n, in.p, metric, path, in.D != nullptr) >= kSplitWork);
        spatial::KdTree T;
        core = core_of(in, metric, path, k, T);
    }
    return wrap(float_array(core, in.n));
}

nb::object hdbscan_py(
    nb::handle Xo,
    nb::handle Do,
    nb::handle oo,
    int64_t min_samples,
    double alpha,
    int metric,
    int path,
    int64_t mcs,
    int method,
    bool allow_single,
    double eps,
    int64_t max_size
) {
    const Input in = input_of(Xo, Do);
    const int64_t n = in.n, p = in.p;
    int64_t no;
    const int64_t* order = index_arg(oo, "order", no);
    if (no != n || n < 2) throw nb::value_error("order");
    if (min_samples < 1 || min_samples > n || mcs < 2) throw nb::value_error("min_samples");
    Hdbscan h;
    Tree t;
    Condensed c;
    {
        const parallel::Unlocked unlocked(n * search_cost(n, p, metric, path, in.D != nullptr) >= kSplitWork);
        spatial::KdTree T;
        const std::vector<double> core = core_of(in, metric, path, min_samples, T);
        std::vector<Edge> E;
        if (in.D) {
            E = mst_prim_matrix(in.D, n, core.data(), alpha);
        } else if (metric == 0 && use_tree(n, p, metric, path, false, true)) {
            E = mst_boruvka(T, false, core.data(), alpha);  // the core distances' tree
        } else {
            E = mst_prim(in.X, n, p, core.data(), alpha, metric);
        }
        std::sort(E.begin(), E.end(), edge_less);
        t = kruskal(E, n);
        c = condense(t, n, mcs);
        h = select(c, n, method, allow_single, eps, max_size);
    }
    // Z in the input's numbering: (smaller, larger) of the mapped clusters
    std::vector<double> z(static_cast<size_t>(4 * (n - 1)));
    auto map = [&](int64_t v) { return v < n ? order[v] : v; };
    for (int64_t i = 0; i < n - 1; ++i) {
        const int64_t a = map(t.left[static_cast<size_t>(i)]), b = map(t.right[static_cast<size_t>(i)]);
        z[static_cast<size_t>(4 * i)] = static_cast<double>(std::min(a, b));
        z[static_cast<size_t>(4 * i + 1)] = static_cast<double>(std::max(a, b));
        z[static_cast<size_t>(4 * i + 2)] = t.h[static_cast<size_t>(i)];
        z[static_cast<size_t>(4 * i + 3)] = static_cast<double>(t.size[static_cast<size_t>(i)]);
    }
    std::vector<int64_t> child(c.child);
    for (int64_t& v : child) v = map(v);
    const int64_t rows = static_cast<int64_t>(child.size());
    const int64_t k = static_cast<int64_t>(h.selected.size());
    return nb::make_tuple(
        wrap(int_array(h.label, n)), wrap(float_array(h.prob, n)), wrap(float_array(h.outlier, n)),
        wrap(float_array(z, n - 1, 4)),
        nb::make_tuple(
            wrap(int_array(c.parent, rows)), wrap(int_array(child, rows)), wrap(float_array(c.lambda, rows)),
            wrap(int_array(c.size, rows))
        ),
        wrap(float_array(h.stability, c.nclusters)), wrap(int_array(h.selected, k)), wrap(float_array(h.persistence, k))
    );
}

// scikit-learn's labelling_at_cut: the merges of height < cut joined, components of fewer than min_cluster_size
// points noise; labels by first appearance.
nb::object hdbscan_cut_py(nb::handle Zo, double cut, int64_t mcs) {
    int64_t m, four;
    const double* Z = matrix_arg(Zo, "Z", m, four);
    const int64_t n = m + 1;
    MinUnionFind uf(2 * n - 1);
    for (int64_t i = 0; i < m; ++i) {
        if (!(Z[4 * i + 2] < cut)) continue;
        const int64_t v = n + i;
        for (int s = 0; s < 2; ++s) {
            const int64_t a = uf.find(static_cast<int64_t>(Z[4 * i + s])), b = uf.find(v);
            if (a != b) uf.unite(a, b);
        }
    }
    std::vector<int64_t> root(static_cast<size_t>(n)), size(static_cast<size_t>(2 * n - 1), 0);
    for (int64_t i = 0; i < n; ++i) ++size[static_cast<size_t>(root[static_cast<size_t>(i)] = uf.find(i))];
    std::vector<int64_t> num(static_cast<size_t>(2 * n - 1), -1), lab(static_cast<size_t>(n), -1);
    int64_t next = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = root[static_cast<size_t>(i)];
        if (size[static_cast<size_t>(r)] < mcs) continue;
        if (num[static_cast<size_t>(r)] < 0) num[static_cast<size_t>(r)] = next++;
        lab[static_cast<size_t>(i)] = num[static_cast<size_t>(r)];
    }
    return wrap(int_array(lab, n));
}

// The centroids (the strength-weighted means, each sum in key order) and the medoids (the point of smallest
// sum_j d(i, j) s_j over its cluster, sums in key order, the smaller key on a tie) of key-ordered data.
nb::object hdbscan_centres_py(nb::handle Xo, nb::handle lo, nb::handle so, int64_t k, int metric) {
    int64_t n, p, nl, ns;
    const double* X = matrix_arg(Xo, "X", n, p);
    const int64_t* lab = index_arg(lo, "labels", nl);
    const double* s = vector_arg(so, "strength", ns);
    if (nl != n || ns != n) throw nb::value_error("labels");
    std::vector<double> C(static_cast<size_t>(k * p), 0.0), W(static_cast<size_t>(k), 0.0);
    std::vector<int64_t> med(static_cast<size_t>(k), -1);
    std::vector<std::vector<int64_t>> members(static_cast<size_t>(k));
    for (int64_t i = 0; i < n; ++i)
        if (lab[i] >= 0) members[static_cast<size_t>(lab[i])].push_back(i);
    {
        int64_t work = 0;
        for (const auto& v : members) work += static_cast<int64_t>(v.size()) * static_cast<int64_t>(v.size());
        const parallel::Unlocked unlocked(work * p >= kSplitWork);
        for (int64_t j = 0; j < k; ++j) {
            double w = 0.0;
            for (int64_t i : members[static_cast<size_t>(j)]) {
                w += s[i];
                for (int64_t d = 0; d < p; ++d) C[static_cast<size_t>(j * p + d)] += s[i] * X[i * p + d];
            }
            for (int64_t d = 0; d < p; ++d) C[static_cast<size_t>(j * p + d)] /= w;
        }
        auto medoid = [&](auto E) {
            for (int64_t j = 0; j < k; ++j) {
                const auto& mem = members[static_cast<size_t>(j)];
                const int64_t m = static_cast<int64_t>(mem.size());
                std::vector<double> tot(static_cast<size_t>(m));
                auto body = [&](int64_t a0, int64_t a1) {
                    for (int64_t a = a0; a < a1; ++a) {
                        const double* x = X + mem[static_cast<size_t>(a)] * p;
                        double t = 0.0;
                        for (int64_t b = 0; b < m; ++b) {
                            const int64_t i = mem[static_cast<size_t>(b)];
                            t += spatial::kd_distance(E, x, X + i * p) * s[i];
                        }
                        tot[static_cast<size_t>(a)] = t;
                    }
                };
                for_points(m, m * (p + 2), body);
                int64_t best = 0;
                for (int64_t a = 1; a < m; ++a)
                    if (tot[static_cast<size_t>(a)] < tot[static_cast<size_t>(best)]) best = a;
                med[static_cast<size_t>(j)] = m ? mem[static_cast<size_t>(best)] : -1;
            }
        };
        if (metric == 1) medoid(spatial::KdCity{p});
        else if (metric == 2) medoid(spatial::KdCheb{p});
        else if (metric == 0) medoid(spatial::KdEuclid{p});  // 3: the centroids alone
    }
    return nb::make_tuple(wrap(float_array(C, k, p)), wrap(int_array(med, k)));
}

nb::object weighted_medoid_py(nb::handle Do, nb::handle so) {
    int64_t m, ns;
    const double* D = vector_arg(Do, "D", m);
    const double* s = vector_arg(so, "strength", ns);
    const int64_t n = ns;
    if (n * (n - 1) / 2 != m) throw nb::value_error("D");
    int64_t best = 0;
    double bt = kInf;
    for (int64_t i = 0; i < n; ++i) {
        double t = 0.0;
        for (int64_t j = 0; j < n; ++j) t += (i == j ? 0.0 : D[cpair(n, i, j)]) * s[j];
        if (i == 0 || t < bt) bt = t, best = i;
    }
    return nb::int_(best);
}

nb::object optics_py(nb::handle Xo, nb::handle Do, int64_t ms, double max_eps, int metric, int path) {
    const Input in = input_of(Xo, Do);
    const int64_t n = in.n, p = in.p;
    if (ms < 1 || ms > n) throw nb::value_error("min_samples");
    Optics o;
    {
        const parallel::Unlocked unlocked(n * n * (p + 2) >= kSplitWork);
        spatial::KdTree T;
        std::vector<double> core = core_of(in, metric, path, ms, T);
        if (in.D) {
            const double* D = in.D;
            o = optics_run(n, 2, std::move(core), max_eps, [&](int64_t i, int64_t j, double T1, double& d) {
                d = D[cpair(n, i, j)];
                return d <= T1;
            });
        } else {
            auto go = [&](auto E) {
                using Dist = decltype(E);
                const double* X = in.X;
                o = optics_run(n, p + 2, std::move(core), max_eps, [&](int64_t i, int64_t j, double T1, double& d) {
                    const double* x = X + i * p;
                    const double* y = X + j * p;
                    const double s = E.raw(x, y);
                    if (Dist::plain(s) && s > Dist::above(T1)) return false;
                    d = E.finish(s, x, y);
                    return d <= T1;
                });
            };
            if (metric == 1) go(spatial::KdCity{p});
            else if (metric == 2) go(spatial::KdCheb{p});
            else go(spatial::KdEuclid{p});
        }
    }
    return nb::make_tuple(
        wrap(int_array(o.ordering, n)), wrap(float_array(o.reach, n)), wrap(float_array(o.core, n)),
        wrap(int_array(o.pred, n))
    );
}

nb::object optics_xi_py(
    nb::handle ro, nb::handle po, nb::handle oo, int64_t min_samples, int64_t mcs, double xi, bool correction
) {
    int64_t n, np_, no;
    const double* rp = vector_arg(ro, "reach_plot", n);
    const int64_t* pp = index_arg(po, "pred_plot", np_);
    const int64_t* ord = index_arg(oo, "ordering", no);
    if (np_ != n || no != n) throw nb::value_error("lengths");
    XiOut x;
    {
        const parallel::Unlocked unlocked(n >= kSplitWork);
        x = optics_xi(rp, pp, ord, n, min_samples, mcs, xi, correction);
    }
    return nb::make_tuple(
        wrap(int_array(x.labels, n)), wrap(int_array(x.clusters, static_cast<int64_t>(x.clusters.size()) / 2, 2))
    );
}

// scikit-learn's cluster_optics_dbscan
nb::object optics_dbscan_py(nb::handle ro, nb::handle co, nb::handle oo, double eps) {
    int64_t n, nc, no;
    const double* reach = vector_arg(ro, "reach", n);
    const double* core = vector_arg(co, "core", nc);
    const int64_t* ord = index_arg(oo, "ordering", no);
    if (nc != n || no != n) throw nb::value_error("lengths");
    std::vector<int64_t> lab(static_cast<size_t>(n));
    int64_t cum = 0;
    for (int64_t a = 0; a < n; ++a) {
        const int64_t i = ord[a];
        if (reach[i] > eps && core[i] <= eps) ++cum;
        lab[static_cast<size_t>(i)] = cum - 1;
    }
    for (int64_t i = 0; i < n; ++i)
        if (reach[i] > eps && !(core[i] <= eps)) lab[static_cast<size_t>(i)] = -1;
    return wrap(int_array(lab, n));
}

}  // namespace

void bind_density(nb::module_& m) {
    m.def(
        "dbscan", &dbscan_py, nb::arg("X").none(), nb::arg("D").none(), nb::arg("eps"), nb::arg("min_samples"),
        nb::arg("w").none(), nb::arg("metric"), nb::arg("path")
    );
    m.def(
        "core_distances", &core_distances_py, nb::arg("X").none(), nb::arg("D").none(), nb::arg("k"), nb::arg("metric"),
        nb::arg("path")
    );
    m.def(
        "hdbscan", &hdbscan_py, nb::arg("X").none(), nb::arg("D").none(), nb::arg("order"), nb::arg("min_samples"),
        nb::arg("alpha"), nb::arg("metric"), nb::arg("path"), nb::arg("min_cluster_size"), nb::arg("method"),
        nb::arg("allow_single"), nb::arg("epsilon"), nb::arg("max_cluster_size")
    );
    m.def("hdbscan_cut", &hdbscan_cut_py, nb::arg("Z"), nb::arg("cut"), nb::arg("min_cluster_size"));
    m.def(
        "hdbscan_centres", &hdbscan_centres_py, nb::arg("X"), nb::arg("labels"), nb::arg("strength"), nb::arg("k"),
        nb::arg("metric")
    );
    m.def("weighted_medoid", &weighted_medoid_py, nb::arg("D"), nb::arg("strength"));
    m.def(
        "optics", &optics_py, nb::arg("X").none(), nb::arg("D").none(), nb::arg("min_samples"), nb::arg("max_eps"),
        nb::arg("metric"), nb::arg("path")
    );
    m.def(
        "optics_xi", &optics_xi_py, nb::arg("reach_plot"), nb::arg("pred_plot"), nb::arg("ordering"),
        nb::arg("min_samples"), nb::arg("min_cluster_size"), nb::arg("xi"), nb::arg("correction")
    );
    m.def("optics_dbscan", &optics_dbscan_py, nb::arg("reach"), nb::arg("core"), nb::arg("ordering"), nb::arg("eps"));
}

}  // namespace sd::cluster
