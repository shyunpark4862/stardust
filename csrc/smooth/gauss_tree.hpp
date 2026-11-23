// The KD-tree path of the Gaussian sums over several variables (docs/design/smooth.md §3.4, §3.5; decisions SM80,
// SM115, SM116): the candidates of one query point from the spatial area's tree (spatial/kdtree.hpp), so that the
// caller keeps exactly the points that scanning every point would keep, in the same (index) order.
//
// The tree is built over whitened coordinates W_j (n rows of p; the caller's transform of the data), and a query
// point has W0 by the same transform. The caller's exact squared distance D_j of a pair (its own formula, from the
// data's coordinates, not from W) differs from |W_j - W0|^2 only by rounding: every coordinate of W_j - W0 is within
// `eps` of the exact z_j that D_j sums, where eps is the caller's bound for that query (decision SM115). Then
//
//   1. an upper bound D_ub of the nearest D_j: the tree is descended to the leaf of the nearer child at every node
//      and D_j is computed for that leaf's points (any point's D is an upper bound);
//   2. every point whose exact D_j is at most (D_ub + T)(1 + 2^-30) lies within the radius
//      r = (sqrt(D_ub + T) + sqrt(p) eps)(1 + 2^-30) of W0 in W (the triangle inequality on the coordinates, and
//      2^-30 covers the roundings of D_j, of the sums of squares and of the box bounds); the nodes whose box is
//      farther than r are cut and the points within r of W0 are collected;
//   3. the collected indices are put in ascending order (a sort, or a byte mask over n when there are many).
//
// The caller then computes D_j for the candidates only and keeps them by its own rule; a rule that keeps only
// points with D_j <= D0 + T (to within a relative 2^-40) keeps a subset of the candidates, and the nearest point
// (the first of the smallest D_j) is among them, so the kept set and its order are those of the scan and every sum
// has the scan's bits. A query whose D_ub is not finite (or whose W0 is not) returns false: the caller scans.
#pragma once

#include "spatial/kdtree.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::smooth::gauss {

// Which path the sums over several variables take: 0 the shape's rule, 1 always the scan, 2 the tree whenever it
// can (tests compare the two paths bit for bit; smooth/kde.cpp's kde_tree_mode sets it).
extern std::atomic<int> g_tree_mode;

struct TreeScratch {
    std::vector<std::int64_t> cand, stack;
    std::vector<std::uint8_t> mask;
};

class TreePath {
public:
    static constexpr std::int64_t kLeaf = 16;

    // The tree over W (n x p, row-major); false when a coordinate is not finite (the caller then scans).
    bool build(const double* W, std::int64_t n, std::int64_t p) {
        n_ = n, p_ = p, ok_ = false;
        for (std::int64_t i = 0; i < n * p; ++i)
            if (!std::isfinite(W[i])) return false;
        tree_ = spatial::kd_build(W, n, p, p, kLeaf);
        wmax_ = 0;
        for (std::int64_t i = 0; i < n * p; ++i) wmax_ = std::max(wmax_, std::fabs(W[i]));
        ok_ = true;
        return true;
    }

    bool ok() const { return ok_; }
    double wmax() const { return wmax_; }  // the largest |W_jl|

    // The candidates of the query W0 (ascending indices into out.cand, `skip` left out); D(j) is the caller's
    // exact squared distance (+inf for skip), T the cut, eps the coordinate bound. False: scan instead.
    template <class Dist>
    bool candidates(const double* W0, double T, double eps, std::int64_t skip, const Dist& D, TreeScratch& S) const {
        S.cand.clear();
        if (!ok_ || n_ == 0) return false;
        for (std::int64_t l = 0; l < p_; ++l)
            if (!std::isfinite(W0[l])) return false;
        // 1. the leaf of the nearer child at every node
        std::int64_t node = 0;
        while (tree_.nodes[node].right >= 0) {
            const std::int64_t a = node + 1, b = tree_.nodes[node].right;
            node = box_sq(b, W0) < box_sq(a, W0) ? b : a;
        }
        double dub = std::numeric_limits<double>::infinity();
        for (std::int64_t i = tree_.nodes[node].start; i < tree_.nodes[node].end; ++i) {
            const std::int64_t j = tree_.idx[i];
            if (j != skip) dub = std::min(dub, D(j));
        }
        if (!(dub < std::numeric_limits<double>::infinity()) || !std::isfinite(eps)) return false;
        const double grow = 1.0 + 0x1p-30;
        const double r = (std::sqrt((dub + T) * grow) + std::sqrt(static_cast<double>(p_)) * eps) * grow;
        const double r2 = r * r * grow;
        if (!std::isfinite(r2)) return false;
        // 2. the points within r
        S.stack.clear();
        S.stack.push_back(0);
        while (!S.stack.empty()) {
            const std::int64_t nd = S.stack.back();
            S.stack.pop_back();
            if (box_sq(nd, W0) > r2) continue;
            const spatial::KdNode& k = tree_.nodes[nd];
            if (k.right >= 0) {
                S.stack.push_back(k.right);
                S.stack.push_back(nd + 1);
                continue;
            }
            const double* y = tree_.data.data() + k.start * p_;
            for (std::int64_t i = k.start; i < k.end; ++i, y += p_) {
                double s = 0;
                for (std::int64_t l = 0; l < p_; ++l) {
                    const double d = y[l] - W0[l];
                    s += d * d;
                }
                const std::int64_t j = tree_.idx[i];
                if (s <= r2 && j != skip) S.cand.push_back(j);
            }
        }
        // 3. ascending order
        const std::int64_t m = static_cast<std::int64_t>(S.cand.size());
        if (m * 16 < n_) {
            std::sort(S.cand.begin(), S.cand.end());
        } else {
            S.mask.assign(n_, 0);
            for (const std::int64_t j : S.cand) S.mask[j] = 1;
            std::int64_t k = 0;
            for (std::int64_t j = 0; j < n_; ++j)
                if (S.mask[j]) S.cand[k++] = j;
        }
        return true;
    }

private:
    spatial::KdTree tree_;
    std::int64_t n_ = 0, p_ = 0;
    double wmax_ = 0;
    bool ok_ = false;

    // The squared distance from W0 to the node's box (0 inside).
    double box_sq(std::int64_t node, const double* W0) const {
        const double *lo = tree_.lo(node), *hi = tree_.hi(node);
        double s = 0;
        for (std::int64_t l = 0; l < p_; ++l) {
            const double d = W0[l] < lo[l] ? lo[l] - W0[l] : (W0[l] > hi[l] ? W0[l] - hi[l] : 0.0);
            s += d * d;
        }
        return s;
    }
};

}  // namespace sd::smooth::gauss
