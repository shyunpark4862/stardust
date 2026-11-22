// The KD tree of sd.spatial (docs/design/spatial.md §2.4, §3.3, §3.4; decisions SPT50-SPT62).
//
// Build (§3.3): every node of more than `leafsize` points is split at the median of its widest axis (the
// smallest such axis on a tie) under the total order (coordinate, index), so the point sets of the two
// halves are one and the same whatever nth_element does (SPT50). Nodes are split even when their box has
// no width, so the node count and the preorder numbering follow from n and leafsize alone. A node's box
// is the range of its own points. Leaf points are kept in index order. The tree holds a copy of the data
// in leaf order and the original indices; the top of the tree is built on the calling thread, the
// subtrees below as tasks cut by size alone, each writing only its own nodes and points, so the tree is
// the same bytes for any thread count (SPT53).
//
// Queries (§3.4): every distance a query returns or compares is the pair kernel of csrc/spatial/metric.hpp
// for the metric of p (1 cityblock, 2 euclidean, inf chebyshev, else minkowski(p)), so it is the bit
// of cdist / minkowski for that pair. The k nearest are kept in a max-heap ordered by (distance, index):
// the result is the first k of all points sorted by (distance, index), the brute force with the same
// kernel. The search goes depth first, the child with the smaller bound first.
//
// Pruning (SPT51): the bound of a node is the kernel distance from x to c, the point of the node's box
// nearest x (c_i = min(max(x_i, lo_i), hi_i), exact). For every point y in the box the true distances
// satisfy D(x, y) >= D(x, c) coordinate by coordinate. With the distance contract |d^ - D| <= kappa D +
// 2^-1074 (docs/design/spatial.md §4; kappa <= (m + 11) u for p >= 1, u = 2^-53), a node is cut only
// when its bound b exceeds T1 = fl(fl(T c) + eta), c = 1 + 4 (m + 12) u, eta = 2^-1070, T the distance a
// point must beat. Then d^(y) >= (1 - kappa) D(x, c) - 2^-1074 >= (1 - kappa) (b - 2^-1074) / (1 + kappa)
// - 2^-1074 > T, since T1 >= T c (1 - u)^2 + (eta - 2^-1075) (1 - u) (the product rounds with an absolute
// error of at most 2^-1075, the sum without one), eta > 4 2^-1074 and c (1 - u)^2 (1 - kappa) / (1 + kappa)
// >= 1 + (2m + 24) u - O(m^2 u^2) > 1:
// every point of a cut node is strictly farther than T, so neither a neighbour nor a tie (equal distance,
// smaller index) is ever dropped. For p = inf the kernel is a correctly rounded maximum of correctly
// rounded differences, monotone in each |y_i - x_i|, so d^(y) >= b exactly and the test is b > T.
#pragma once

#include "spatial/metric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace sd::spatial {

using std::int64_t;

struct KdNode {
    int64_t start = 0, end = 0;  // the node's points [start, end) in leaf order
    int64_t right = -1;          // the right child's index (the left child is the next node); -1: a leaf
};

struct KdTree {
    int64_t n = 0, m = 0, leafsize = 10;
    std::vector<double> data;   // (n, m) in leaf order
    std::vector<int64_t> idx;   // the original index of each point in leaf order
    std::vector<KdNode> nodes;  // preorder
    std::vector<double> box;    // per node: lo[0 .. m), hi[0 .. m)

    const double* lo(int64_t node) const { return box.data() + 2 * m * node; }
    const double* hi(int64_t node) const { return box.data() + 2 * m * node + m; }
};

// The number of nodes of a tree over c points (SPT50: a function of c and leafsize alone).
int64_t kd_node_count(int64_t c, int64_t leafsize);

// Builds the tree over src (n rows of m doubles, row stride ld doubles), all finite (checked by the
// caller). Runs on the pool from the parallel threshold; touches no Python object.
KdTree kd_build(const double* src, int64_t n, int64_t m, int64_t ld, int64_t leafsize);

// ---- the distance of a query and a point, by metric ------------------------------------------------
//
// raw(x, y) is the kernel's sum and finish(r, x, y) the distance from it (d^ = finish(raw)). For p = 2
// the search compares sums first: when r is in the plain range [kTiny, kHuge] (no careful path, so d^ =
// RN(sqrt(r))) and r > above(T) = fl(fl(T T) (1 + 8u)), then d^ > T strictly (fl(T T) >= T^2 (1 - u) - 2^-1075,
// so r > T^2 (1 + 5u) when T^2 is normal and sqrt(r) (1 - u) > T; when T^2 is below the normal range,
// sqrt(r) >= 2^-484 > T anyway; T = inf gives inf). The square root is taken only for the sums that may
// still enter the result. For the other metrics raw is the distance and above(T) is T.

struct KdEuclid {  // p = 2
    int64_t m;
    double raw(const double* x, const double* y) const {
        double s;
        pair_scalar<SqOp, false>(x, y, nullptr, m, &s);
        return s;
    }
    double finish(double s, const double* x, const double* y) const {
        return euclid_finish<false>(s, x, y, nullptr, m);
    }
    static bool plain(double s) { return s >= kTiny && s <= kHuge; }
    static double above(double T) { return (T * T) * (1.0 + 0x1p-50); }
    // A plain sum s <= below(T) has d^ <= T: s <= fl(T T) (1 - 8u) (1 + u) < T^2 when T^2 is normal, so
    // sqrt(s) < T and RN(sqrt(s)) <= T. Outside the normal range of T^2 no sum passes (-1).
    static double below(double T) {
        const double t = T * T;
        return t >= 0x1p-1022 && t <= kHuge ? t * (1.0 - 0x1p-50) : -1.0;
    }
};
struct KdCity {  // p = 1
    int64_t m;
    double raw(const double* x, const double* y) const {
        double s;
        pair_scalar<AbsOp, false>(x, y, nullptr, m, &s);
        return s;
    }
    double finish(double s, const double*, const double*) const { return s; }
    static bool plain(double) { return true; }
    static double above(double T) { return T; }
    static double below(double T) { return T; }
};
struct KdCheb {  // p = inf
    int64_t m;
    double raw(const double* x, const double* y) const {
        double s;
        pair_scalar<MaxOp, false>(x, y, nullptr, m, &s);
        return s;
    }
    double finish(double s, const double*, const double*) const { return s; }
    static bool plain(double) { return true; }
    static double above(double T) { return T; }
    static double below(double T) { return T; }
};
struct KdMink {  // any other p >= 1
    int64_t m;
    Root root;
    double raw(const double* x, const double* y) const { return minkowski_pair<false>(x, y, nullptr, m, root); }
    double finish(double s, const double*, const double*) const { return s; }
    static bool plain(double) { return true; }
    static double above(double T) { return T; }
    static double below(double T) { return T; }
};

// The threshold a node's bound must exceed to be cut, for a search radius T (see the top of this file).
struct KdCut {
    double c = 1.0;      // 1 + 4 (m + 12) u, exact (spacing 2u above 1)
    bool exact = false;  // p = inf: the bound is b > T itself
    KdCut(int64_t m, bool chebyshev) : c(1.0 + static_cast<double>(m + 12) * 0x1p-51), exact(chebyshev) {}
    double operator()(double T) const { return exact ? T : (T * c) + 0x1p-1070; }  // inf stays inf
};

// ---- bounds between two boxes (SPT4: the radius, pair and count queries; SPT63) ---------------------------
//
// A box is (lo, hi) per axis; a query point x is the box (x, x). The bounds are kernel distances of two
// points a, b made of box corners, so they are computed like any pair and the proof above applies.
//
// near(A, B): a_i, b_i are the corners across the gap of axis i (hi^A_i and lo^B_i when lo^B_i > hi^A_i,
// lo^A_i and hi^B_i when lo^A_i > hi^B_i, else 0 and 0), so |a_i - b_i| is the gap exactly and D(a, b) <=
// D(x, y) for every x in A and y in B. A node pair is cut for a radius T when the bound exceeds KdCut(T):
// every pair of it then has d^ > T (the proof above, with D(x, c) replaced by D(a, b)). For a point and
// a box this is SPT51's clamp point.
//
// far(A, B): a_i, b_i are the corners of the larger rounded difference of fl(hi^B_i - lo^A_i) and
// fl(hi^A_i - lo^B_i) (the first on a tie). The largest true |x_i - y_i| over the boxes, f_i, is the larger
// of the two true differences; rounding is monotone, so the chosen true difference t_i is f_i unless both
// round to the same R, and then t_i >= f_i (1 - u) / (1 + u) >= f_i (1 - 2u) (a subnormal difference is
// exact). The p-norms are monotone and homogeneous, so D(x, y) <= D(f) <= D(a, b) / (1 - 2u). With the
// distance contract d^(x, y) <= (1 + kappa) D(x, y) + 2^-1074 and D(a, b) <= (b^ + 2^-1074) / (1 - kappa)
// for the bound b^ = d^(a, b):
//     d^(x, y) <= (1 + kappa) / ((1 - kappa) (1 - 2u)) (b^ + 2^-1074) + 2^-1074 <= fl(fl(b^ c) + eta),
// since c (1 - u)^2 >= 1 + (4m + 46) u - O(u^2) exceeds (1 + kappa) / ((1 - kappa) (1 - 2u)) <= 1 + (2m + 24) u
// + O(u^2) (kappa <= (m + 11) u) and (eta - 2^-1075) (1 - u) > 15 2^-1074 covers the absolute terms. So
// every pair of a node pair has d^ <= T as soon as KdCut(b^) <= T: the whole node pair is inside. A
// difference that overflows makes b^ = inf, inside only for T = inf, which holds for any finite data. For
// p = inf the kernel is the largest correctly rounded |x_i - y_i|, monotone in each, so d^(x, y) <= b^
// exactly and the test is b^ <= T (KdCut is exact there).
struct KdPair {
    std::vector<double> a, b;
    explicit KdPair(int64_t m) : a(static_cast<size_t>(m)), b(static_cast<size_t>(m)) {}

    void near(const double* loA, const double* hiA, const double* loB, const double* hiB, int64_t m) {
        for (int64_t i = 0; i < m; ++i) {
            const size_t k = static_cast<size_t>(i);
            if (loB[i] > hiA[i]) {
                a[k] = hiA[i];
                b[k] = loB[i];
            } else if (loA[i] > hiB[i]) {
                a[k] = loA[i];
                b[k] = hiB[i];
            } else {
                a[k] = 0.0;
                b[k] = 0.0;
            }
        }
    }
    void far(const double* loA, const double* hiA, const double* loB, const double* hiB, int64_t m) {
        for (int64_t i = 0; i < m; ++i) {
            const size_t k = static_cast<size_t>(i);
            if (hiB[i] - loA[i] >= hiA[i] - loB[i]) {
                a[k] = loA[i];
                b[k] = hiB[i];
            } else {
                a[k] = hiA[i];
                b[k] = loB[i];
            }
        }
    }
};

// The kernel distance of a pair (the sum, then the finish).
template <class Dist>
inline double kd_distance(const Dist& dist, const double* x, const double* y) {
    return dist.finish(dist.raw(x, y), x, y);
}

// Whether the pair (x, y) is proven beyond T1 (a plain sum above T1's square says so without the root;
// a plain sum at or below it is not cut, which only prunes less).
template <class Dist>
inline bool kd_beyond(const Dist& dist, const double* x, const double* y, double T1) {
    const double s = dist.raw(x, y);
    if (Dist::plain(s)) return s > Dist::above(T1);
    return dist.finish(s, x, y) > T1;
}

// One query's search: the k nearest (k >= 1, at most n kept) with distance < ub, as (distance, index).
template <class Dist>
class KdSearch {
public:
    KdSearch(const KdTree& t, const Dist& dist, const KdCut& cut, int64_t k)
        : t_(t), dist_(dist), cut_(cut), k_(std::min(k, t.n)), clamp_(static_cast<size_t>(t.m)) {
        heap_.reserve(static_cast<size_t>(std::max<int64_t>(k_, 1)));
    }

    // The neighbours of x sorted by (distance, index) into `out` (cleared first).
    void run(const double* x, double ub, std::vector<std::pair<double, int64_t>>& out) {
        heap_.clear();
        x_ = x;
        ub_ = ub;
        if (k_ > 0 && !t_.nodes.empty()) {
            const double r = bound(0);
            if (!cut(r)) visit(0);
        }
        out.assign(heap_.begin(), heap_.end());
        std::sort(out.begin(), out.end());
    }

private:
    const KdTree& t_;
    const Dist& dist_;
    const KdCut& cut_;
    int64_t k_;
    std::vector<double> clamp_;
    std::vector<std::pair<double, int64_t>> heap_;  // max-heap by (distance, index)
    const double* x_ = nullptr;
    double ub_ = 0.0;

    double radius() const { return static_cast<int64_t>(heap_.size()) < k_ ? ub_ : heap_.front().first; }

    // The kernel's sum from x to the point of the node's box nearest x.
    double bound(int64_t node) {
        const double *lo = t_.lo(node), *hi = t_.hi(node);
        for (int64_t i = 0; i < t_.m; ++i) clamp_[static_cast<size_t>(i)] = std::min(std::max(x_[i], lo[i]), hi[i]);
        return dist_.raw(x_, clamp_.data());
    }

    // Whether a node whose bound has the sum r is cut: its distance exceeds cut_(radius) (a plain sum above
    // the threshold's square says so without the root; otherwise the distance itself is compared).
    bool cut(double r) {
        const double T1 = cut_(radius());
        if (Dist::plain(r)) return r > Dist::above(T1);
        return dist_.finish(r, x_, clamp_.data()) > T1;
    }

    void offer(double r, const double* y, int64_t i) {
        const bool full = static_cast<int64_t>(heap_.size()) >= k_;
        const double T = full ? heap_.front().first : ub_;
        if (Dist::plain(r) && r > Dist::above(T)) return;  // d^ > T: neither below ub nor before the top
        const std::pair<double, int64_t> e{dist_.finish(r, x_, y), i};
        if (!full) {
            if (!(e.first < ub_)) return;
            heap_.push_back(e);
            std::push_heap(heap_.begin(), heap_.end());
        } else if (e < heap_.front()) {
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.back() = e;
            std::push_heap(heap_.begin(), heap_.end());
        }
    }

    void visit(int64_t node) {
        const KdNode& nd = t_.nodes[static_cast<size_t>(node)];
        if (nd.right < 0) {
            const double* y = t_.data.data() + nd.start * t_.m;
            for (int64_t i = nd.start; i < nd.end; ++i, y += t_.m)
                offer(dist_.raw(x_, y), y, t_.idx[static_cast<size_t>(i)]);
            return;
        }
        const int64_t l = node + 1, r = nd.right;
        const double bl = bound(l), br = bound(r);
        const bool left_first = !(br < bl);
        if (left_first) {
            if (!cut_sum(bl, l)) visit(l);
            if (!cut_sum(br, r)) visit(r);
        } else {
            if (!cut_sum(br, r)) visit(r);
            if (!cut_sum(bl, l)) visit(l);
        }
    }

    // cut() for a child whose bound sum r was computed before: the careful path needs the clamp point again.
    bool cut_sum(double r, int64_t node) {
        if (Dist::plain(r)) return r > Dist::above(cut_(radius()));
        (void)bound(node);
        return cut(r);
    }
};

}  // namespace sd::spatial
