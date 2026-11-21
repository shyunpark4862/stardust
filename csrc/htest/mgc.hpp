// Multiscale graph correlation (multiscale_graphcorr, milestone H9; docs/design/htest.md §3.10):
// scipy's _mgc_stat on two distance matrices, and its permutations of the second.
//
// Each distance matrix is ranked column by column (dense ranks: ties share a rank, the next value
// takes the next one; scipy's _rank_distance_matrix) and
// centred by its column sums, c_ij = d_ij - (sum_i d_ij)/(n - 1) with a zero diagonal (scipy's
// 'mgc' transform). The local covariance of the scales (k, l) sums a_ij b_ji over the pairs whose
// ranks are at most k in x (a_ij's) and l in y (b_ji's: scipy pairs x with y's transpose), less
// the product of the two sums over those pairs / n^2;
// the local correlation divides by the square roots of the two local variances (0 where one is
// not positive, at most 1). The statistic is the largest local correlation in the largest
// 4-connected region of correlations above the threshold (the beta approximation's quantile
// 1 - 0.02/(n - 1), or the global correlation if larger) when the region is large enough, else the
// global correlation (the map's last entry): scipy's _threshold_mgc_map and _smooth_mgc_map.
//
// A permutation p of the second sample permutes the rows and columns of its matrix: its ranks and
// centred values are those of the original at (p_i, p_j), its local variances and its sums by
// rank do not change, so only the covariances are recomputed (O(n^2) per permutation).
#pragma once

#include "htest/moments.hpp"
#include "stats/dist/beta.hpp"
#include "stats/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sd::htest::mgc {

using i64 = std::int64_t;

struct Prepared {
    i64 n = 0, m = 0;  // m: the largest rank (the map's side)
    std::vector<double> cent;  // (n, n) centred distances
    std::vector<std::int32_t> rank;  // (n, n) dense ranks - 1 within the columns
    std::vector<double> expect;  // (m,) the sums of the centred values of ranks <= k
    std::vector<double> var;  // (m,) the local variances
};

inline Prepared prepare(const Rows& D) {
    Prepared P;
    const i64 n = D.lanes;
    P.n = n;
    P.cent.resize(static_cast<std::size_t>(n * n));
    P.rank.resize(static_cast<std::size_t>(n * n));
    std::vector<i64> order(static_cast<std::size_t>(n));
    std::int32_t top = 0;
    for (i64 j = 0; j < n; ++j) {
        double colsum = 0.0;
        for (i64 i = 0; i < n; ++i) colsum += D.at(i, j);
        const double mean = colsum / static_cast<double>(n) * static_cast<double>(n) / static_cast<double>(n - 1);
        for (i64 i = 0; i < n; ++i) P.cent[static_cast<std::size_t>(i * n + j)] = i == j ? 0.0 : D.at(i, j) - mean;
        std::iota(order.begin(), order.end(), i64{0});
        std::stable_sort(order.begin(), order.end(), [&](i64 a, i64 b) { return D.at(a, j) < D.at(b, j); });
        std::int32_t g = 0;  // dense ranks (scipy's _rank_distance_matrix): the tie groups counted
        for (i64 s = 0; s < n; ++g) {
            i64 e = s + 1;
            while (e < n && D.at(order[static_cast<std::size_t>(e)], j) == D.at(order[static_cast<std::size_t>(s)], j))
                ++e;
            for (i64 q = s; q < e; ++q)
                P.rank[static_cast<std::size_t>(order[static_cast<std::size_t>(q)] * n + j)] = g;
            s = e;
        }
        top = std::max(top, g);
    }
    P.m = top;
    // the local variance of scale k pairs entry (i, j) with its transpose (j, i), as scipy's
    // _local_covariance(c, c.T, r, r.T): the diagonal of its map sums c_ij c_ji over the pairs
    // whose two ranks are at most k
    std::vector<double> sum(static_cast<std::size_t>(P.m), 0.0), sq(static_cast<std::size_t>(P.m), 0.0);
    for (i64 i = 0; i < n; ++i)
        for (i64 j = 0; j < n; ++j) {
            const std::size_t q = static_cast<std::size_t>(i * n + j), t = static_cast<std::size_t>(j * n + i);
            sum[static_cast<std::size_t>(P.rank[q])] += P.cent[q];
            sq[static_cast<std::size_t>(std::max(P.rank[q], P.rank[t]))] += P.cent[q] * P.cent[t];
        }
    P.expect.resize(static_cast<std::size_t>(P.m));
    P.var.resize(static_cast<std::size_t>(P.m));
    double cs = 0.0, cq = 0.0;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);
    for (i64 k = 0; k < P.m; ++k) {
        cs += sum[static_cast<std::size_t>(k)], cq += sq[static_cast<std::size_t>(k)];
        P.expect[static_cast<std::size_t>(k)] = cs;
        P.var[static_cast<std::size_t>(k)] = cq - cs * cs / n2;
    }
    return P;
}

struct Scratch {
    std::vector<double> cov, corr;
    std::vector<std::int32_t> label;
    std::vector<i64> stack, size;
    explicit Scratch(i64 n)
        : cov(static_cast<std::size_t>(n * n)),
          corr(static_cast<std::size_t>(n * n)),
          label(static_cast<std::size_t>(n * n)) {}
};

struct Stat {
    double stat = 0.0;
    i64 k = 0, l = 0;  // the optimal scale (1-based); l = -1: the scalar m * n of a one-row map
};

// The beta approximation's threshold of a sample of n (scipy: beta.ppf(1 - 0.02/(n-1), a, a)*2 - 1).
inline double threshold(i64 n) {
    const double s = static_cast<double>(n - 1);
    const double per = 1.0 - 0.02 / s;
    const double a = s * (s - 3.0) / 4.0 - 0.5;
    return stats::method_of<stats::Beta, stats::method::Icdf>(per, a, a) * 2.0 - 1.0;
}

// the statistic of x against y with y's observations permuted by perm (identity: the observed one);
// sc.corr holds the map (x.m, y.m) after the call
inline Stat statistic(const Prepared& x, const Prepared& y, const i64* perm, Scratch& sc, double thr0) {
    const i64 n = x.n, mx = x.m, my = y.m;
    std::fill(sc.cov.begin(), sc.cov.begin() + mx * my, 0.0);
    for (i64 i = 0; i < n; ++i) {
        const i64 pi = perm[i];
        for (i64 j = 0; j < n; ++j) {
            // y's entry (j, i): scipy's _local_covariance(cent_x, cent_y.T, rank_x, rank_y.T)
            const std::size_t qx = static_cast<std::size_t>(i * n + j), qy = static_cast<std::size_t>(perm[j] * n + pi);
            sc.cov[static_cast<std::size_t>(x.rank[qx] * my + y.rank[qy])] += x.cent[qx] * y.cent[qy];
        }
    }
    double* c = sc.cov.data();
    for (i64 k = 1; k < mx; ++k) c[k * my] += c[(k - 1) * my];
    for (i64 l = 1; l < my; ++l) c[l] += c[l - 1];
    for (i64 k = 0; k + 1 < mx; ++k)
        for (i64 l = 0; l + 1 < my; ++l)
            c[(k + 1) * my + l + 1] += c[(k + 1) * my + l] + c[k * my + l + 1] - c[k * my + l];
    const double n2 = static_cast<double>(n) * static_cast<double>(n);
    double* r = sc.corr.data();
    for (i64 k = 0; k < mx; ++k) {
        const double vx = x.var[static_cast<std::size_t>(k)];
        for (i64 l = 0; l < my; ++l) {
            const double vy = y.var[static_cast<std::size_t>(l)];
            const double cv =
                c[k * my + l] - x.expect[static_cast<std::size_t>(k)] * y.expect[static_cast<std::size_t>(l)] / n2;
            double v = cv / std::sqrt(vx * vy);
            if (v > 1.0) v = 1.0;
            if (!(vx > 0.0) || !(vy > 0.0)) v = 0.0;
            r[k * my + l] = v;
        }
    }
    Stat out;
    const double global = r[mx * my - 1];
    out.stat = global;
    if (mx == 1 || my == 1) {
        out.k = mx * my, out.l = -1;
        return out;
    }
    out.k = mx, out.l = my;
    const double thr = std::max(thr0, global);
    // the 4-connected regions above the threshold, labelled in raster order of their first cell
    // (scipy.ndimage.label's numbering); the largest, the first of equal ones
    std::int32_t* lab = sc.label.data();
    std::fill(lab, lab + mx * my, 0);
    sc.size.assign(1, 0);
    std::int32_t next = 0;
    for (i64 q0 = 0; q0 < mx * my; ++q0) {
        if (!(r[q0] > thr) || lab[q0] != 0) continue;
        ++next;
        i64 cnt = 0;
        sc.stack.assign(1, q0);
        lab[q0] = next;
        while (!sc.stack.empty()) {
            const i64 q = sc.stack.back();
            sc.stack.pop_back();
            ++cnt;
            const i64 k = q / my, l = q % my;
            const i64 nb4[4] = {
                k > 0 ? q - my : -1, k + 1 < mx ? q + my : -1, l > 0 ? q - 1 : -1, l + 1 < my ? q + 1 : -1
            };
            for (i64 w : nb4)
                if (w >= 0 && lab[w] == 0 && r[w] > thr) lab[w] = next, sc.stack.push_back(w);
        }
        sc.size.push_back(cnt);
    }
    if (next == 0) return out;
    std::int32_t best = 1;
    for (std::int32_t g = 2; g <= next; ++g)
        if (sc.size[static_cast<std::size_t>(g)] > sc.size[static_cast<std::size_t>(best)]) best = g;
    const i64 area = sc.size[static_cast<std::size_t>(best)];
    if (static_cast<double>(area) <
        std::ceil(0.02 * static_cast<double>(std::max(mx, my))) * static_cast<double>(std::min(mx, my)))
        return out;
    double mc = -std::numeric_limits<double>::infinity();
    for (i64 q = 0; q < mx * my; ++q)
        if (lab[q] == best) mc = std::max(mc, r[q]);
    if (mc >= global) {
        i64 last = -1;
        for (i64 q = 0; q < mx * my; ++q)
            if (lab[q] == best && r[q] >= mc) last = q;
        out.stat = mc;
        out.k = last / my + 1, out.l = last % my + 1;
    }
    return out;
}

}  // namespace sd::htest::mgc
