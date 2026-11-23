// The convex hull of 2-D and 3-D points (docs/design/spatial_geometry.md §2.4, §3.6, §3.10, §3.11; decisions
// SPT108, SPT109, SPT140-SPT153). The private binding behind sd.spatial.ConvexHull:
//
//   _convex_hull(points) -> (points, vertices, simplices, neighbors, equations, area, volume, min_bound, max_bound)
//       points (n, d) float64, d = 2 or 3 (checked by the caller); the first item is the copy the hull was
//       computed from. Raises ValueError for non-finite points, too few distinct points, all on one line
//       (d = 2) or one plane (d = 3).
//
// Every combinatorial answer -- which points are extreme, which facets exist, which facets are flat
// together -- is the exact one for the inputs read as rationals (predicates.hpp). The hull is a function of
// the point set: neither the input order, the order points are added in, nor the thread count changes a byte.
//
// STEPS
//   1. Scan (tasks of kChunk points): finiteness, the range of the coordinates, the box, and the extreme
//      point in each of 8 (d = 2) or 14 (d = 3) directions, ties by (lexicographic order, index) (SPT141).
//   2. Start: the exact hull of those candidates -- a polygon (d = 2: monotone chain) or a polytope
//      (d = 3: the quickhull below over the candidates alone). Its vertices are input points, so it lies
//      inside the hull.
//   3. Pre-filter (tasks): a point strictly inside the start polygon / polytope is not extreme and is
//      dropped. The test is a static filter (SPT142): one float64 plane per edge or facet and a bound
//      proved like the predicates' filters (below); a point it cannot place is kept (d = 2) or decided by
//      the exact orient3d (d = 3).
//   4. d = 2: the kept points in lexicographic order, equal points merged (the smallest index stays,
//      SPT96), Andrew's monotone chain with strict turns (exact orient2d > 0): points on an edge and equal
//      points drop out. d = 3: quickhull (SPT143) from the start polytope: every kept point waits in the
//      conflict list of the first facet (in facet order) it is strictly outside; the farthest point of a
//      facet (float64 distance; ties to the smallest index) is added, the facets it strictly sees are
//      removed (exact orient3d > 0; a facet whose plane holds it stays), the horizon is joined to it, and
//      the removed facets' points move to the first new facet they are strictly outside, or drop out.
//   5. d = 3 flat facets (SPT144): neighbouring triangles on exactly one plane (orient3d = 0) are merged
//      into polygons; a polygon's vertices inside it or in the middle of a side (exactly collinear with
//      its two neighbours on the boundary) are not extreme and drop out; each polygon is a fan from its
//      smallest index.
//   6. Canonical form (§2.2), equations, area and volume (SPT145-SPT148).
//
// THE STATIC FILTER (SPT142). The point set satisfies Range::safe(q) (q = 2 for d = 2, 3 for d = 3), so the
// lemma of predicates.hpp holds: an expression tree of float64 operations over differences of input values
// has |x^ - x| <= u |x^| + theta_(c-1) S (c the count of the tree, S its exact permanent), and the sign of x^
// is the sign of x once (1 - u)|x^| > theta_(c-1) S. Coordinates are measured from o = the box's lower corner
// (each o_k is an input coordinate, so every difference is a multiple of 2^lo as the lemma needs).
//   d = 2, edge (a, b), point p: e = fl(b - a), q = fl(p - o), a' = fl(a - o) (counts 1),
//     s^ = fl(fl(fl(e_x q_y) - fl(e_y q_x)) - h),  h = fl(fl(e_x a'_y) - fl(e_y a'_x)): count 5, and the exact
//     value e (q - a') is orient2d(a, b, p). S = |e_x|(|q_y| + |a'_y|) + |e_y|(|q_x| + |a'_x|)
//     <= 2 (|e_x| E_y + |e_y| E_x), E the box's extent. With B = fl(10u fl(fl(|e^_x| E^_y) + fl(|e^_y| E^_x)))
//     and |e^| >= (1 - u)|e|, E^ >= (1 - u) E: B >= 5u (1 - u)^5 S, so (1 - u) B > theta_4 S = (4u + 6u^2 + ..) S.
//   d = 3, facet (a, b, c): e1 = fl(b - a), e2 = fl(c - a), n_x = fl(fl(e1_y e2_z) - fl(e1_z e2_y)) (count 4)
//     and so on, h = fl(fl(n_x a'_x + n_y a'_y) + n_z a'_z) (count 8), t = the same with q (count 8),
//     s^ = fl(t - h): count 9, exact value n (q - a') = orient3d(a, b, c, p). S = sum_k N_k (|q_k| + |a'_k|)
//     <= 2 sum_k N_k E_k with N_x = |e1_y e2_z| + |e1_z e2_y|. B = fl(18u ((N^_x E^_x + N^_y E^_y) + N^_z E^_z)),
//     N^_x = fl(fl(|e^1_y| |e^2_z|) + fl(|e^1_z| |e^2_y|)): B >= 9u (1 - u)^9 S, (1 - u) B > theta_8 S.
//   A plane whose bound is not a normal float64 (B < 2^-1000 or not finite) places nothing. So s^ > B (d = 2:
//   left of the edge, inside) and s^ < -B (d = 3: inside the facet's plane) are certain.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/metric.hpp"
#include "spatial/predicates.hpp"
#include "spatial/sfc.hpp"
#include "spatial/triangulation.hpp"
#include "spatial/spatial.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sd::spatial {

namespace nb = nanobind;

namespace {

using i32 = std::int32_t;
using i64 = std::int64_t;

constexpr i64 kSplitWork = 1 << 13;  // split (and drop the GIL) from n d = 2^13 coordinates: measured (SPT186)
constexpr i64 kChunk = 1 << 12;         // points a task scans (steps 1 and 3; SPT186)
constexpr i64 kFacetChunk = 1 << 12;    // facets a task handles (coplanar tests, equations, terms)
constexpr i64 kSumBlock = 4096;         // terms of one partial sum (§3.6)
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kBoundMin = 0x1p-1000;  // a static filter bound below this places nothing

template <class F>
void run_tasks(i64 ntasks, bool split, F&& fn) {
    if (!split || ntasks <= 1) {
        for (i64 t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

// ---------------------------------------------------------------- errors

struct HullError {
    enum Kind { None, NotFinite, TooFew, Collinear, Coplanar } kind = None;
    i64 distinct = 0;
};

// ---------------------------------------------------------------- small helpers

// The euclidean pair kernel of cdist (metric.hpp): the length of x - y over n coordinates.
inline double euclid(const double* x, const double* y, int n) {
    double s[1];
    pair_scalar<SqOp, false>(x, y, nullptr, n, s);
    return euclid_finish<false>(s[0], x, y, nullptr, n);
}

// floor(log2 |w 2^e|) for w != 0 (to_double probed 2^1000 at a time).
int wide_top(const WideInt& w, i64 e) {
    i64 s = 0;
    for (;;) {
        const double v = w.to_double(e - s);
        if (std::isinf(v)) {
            s += 1000;
        } else if (std::fabs(v) < 0x1p-1000) {
            s -= 1000;
        } else {
            return static_cast<int>(std::ilogb(v) + s);
        }
    }
}

// The number of distinct rows (-0.0 equals 0.0): the error path only.
i64 count_distinct(const double* p, i64 n, int d) {
    std::vector<i64> order(static_cast<size_t>(n));
    for (i64 i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [=](i64 i, i64 j) { return lex_less(p + i * d, p + j * d, d); });
    i64 m = 0;
    for (i64 k = 0; k < n; ++k)
        if (k == 0 || !same_point(p + order[static_cast<size_t>(k)] * d, p + order[static_cast<size_t>(k - 1)] * d, d))
            ++m;
    return m;
}

// The sum of the terms in order as partial sums of kSumBlock terms added one after another (§3.6): the
// blocks are tasks; the result does not depend on the thread count.
double blocked_sum(const std::vector<double>& t, bool split) {
    const i64 n = static_cast<i64>(t.size());
    if (n == 0) return 0.0;
    const i64 nb = (n + kSumBlock - 1) / kSumBlock;
    std::vector<double> part(static_cast<size_t>(nb));
    run_tasks(nb, split && nb > 1, [&](i64 b) {
        const i64 s = b * kSumBlock, e = std::min(n, s + kSumBlock);
        double c = t[static_cast<size_t>(s)];
        for (i64 i = s + 1; i < e; ++i) c += t[static_cast<size_t>(i)];
        part[static_cast<size_t>(b)] = c;
    });
    double s = part[0];
    for (i64 b = 1; b < nb; ++b) s += part[static_cast<size_t>(b)];
    return s;
}

// ---------------------------------------------------------------- step 1: the scan

constexpr int kDirs2 = 8, kDirs3 = 14;

// The score of point p in direction k: d = 2: x, -x, y, -y, x + y, -(x + y), x - y, y - x;
// d = 3: x, -x, y, -y, z, -z, then s_x x + s_y y + s_z z for the 8 sign patterns.
template <int D>
inline double score(const double* p, int k) {
    if (k < 2 * D) return (k & 1) ? -p[k >> 1] : p[k >> 1];
    if constexpr (D == 2) {
        switch (k) {
            case 4: return p[0] + p[1];
            case 5: return -p[0] - p[1];
            case 6: return p[0] - p[1];
            default: return p[1] - p[0];
        }
    } else {
        const int m = k - 6;
        const double x = (m & 1) ? -p[0] : p[0], y = (m & 2) ? -p[1] : p[1], z = (m & 4) ? -p[2] : p[2];
        return (x + y) + z;
    }
}

template <int D>
struct Scan {
    static constexpr int K = D == 2 ? kDirs2 : kDirs3;
    bool finite = true;
    double amax = 0.0, amin = kInf;  // the largest |x| and the smallest non-zero |x|
    double lo[D], hi[D];
    i64 best[K];
    double bestv[K];

    Scan() {
        for (int k = 0; k < D; ++k) lo[k] = kInf, hi[k] = -kInf;
        for (int k = 0; k < K; ++k) best[k] = -1, bestv[k] = -kInf;
    }
};

// Whether (v, point i) beats the current best (bv, point j): a higher score, then the lexicographically
// smaller point, then the smaller index (a strict total order, so the chunks combine in any grouping).
template <int D>
inline bool beats(const double* p, double v, i64 i, double bv, i64 j) {
    if (j < 0) return true;
    if (v != bv) return v > bv;
    const double *a = p + i * D, *b = p + j * D;
    if (lex_less(a, b, D)) return true;
    if (lex_less(b, a, D)) return false;
    return i < j;
}

template <int D>
void scan_points(const double* p, i64 s, i64 e, Scan<D>& r) {
    for (i64 i = s; i < e; ++i) {
        const double* x = p + i * D;
        for (int k = 0; k < D; ++k) {
            const double a = std::fabs(x[k]);
            if (!(a <= std::numeric_limits<double>::max())) {
                r.finite = false;
                return;
            }
            r.amax = std::max(r.amax, a);
            if (a != 0.0) r.amin = std::min(r.amin, a);
            if (x[k] < r.lo[k]) r.lo[k] = x[k];
            if (x[k] > r.hi[k]) r.hi[k] = x[k];
        }
        for (int k = 0; k < Scan<D>::K; ++k) {
            const double v = score<D>(x, k);
            if (beats<D>(p, v, i, r.bestv[k], r.best[k])) r.bestv[k] = v, r.best[k] = i;
        }
    }
}

template <int D>
void merge_scan(const double* p, Scan<D>& r, const Scan<D>& c) {
    r.finite = r.finite && c.finite;
    r.amax = std::max(r.amax, c.amax);
    r.amin = std::min(r.amin, c.amin);
    for (int k = 0; k < D; ++k) {
        if (c.lo[k] < r.lo[k]) r.lo[k] = c.lo[k];
        if (c.hi[k] > r.hi[k]) r.hi[k] = c.hi[k];
    }
    for (int k = 0; k < Scan<D>::K; ++k)
        if (c.best[k] >= 0 && beats<D>(p, c.bestv[k], c.best[k], r.bestv[k], r.best[k]))
            r.bestv[k] = c.bestv[k], r.best[k] = c.best[k];
}

template <int D>
Scan<D> scan(const double* p, i64 n, bool split) {
    const i64 nt = (n + kChunk - 1) / kChunk;
    std::vector<Scan<D>> part(static_cast<size_t>(std::max<i64>(nt, 1)));
    run_tasks(nt, split, [&](i64 t) { scan_points<D>(p, t * kChunk, std::min(n, (t + 1) * kChunk), part[t]); });
    Scan<D> r;
    for (i64 t = 0; t < nt; ++t) merge_scan<D>(p, r, part[static_cast<size_t>(t)]);
    return r;
}

// A range for the predicates from the largest and smallest non-zero magnitudes (SPT141): hi is exact; lo is
// bounded below by the smallest magnitude's exponent minus 52 (a lowest set bit of x is at least ulp(x)), so
// Range::safe stays a sufficient condition.
template <int D>
Range range_of(const Scan<D>& s) {
    Range r;
    if (s.amax == 0.0) return r;
    r.hi = std::ilogb(s.amax);
    r.lo = std::max(-1074, std::ilogb(s.amin) - 52);
    return r;
}

// The candidates: the distinct points of the directions, in direction order.
template <int D>
std::vector<i64> candidates(const Scan<D>& s) {
    std::vector<i64> c;
    for (int k = 0; k < Scan<D>::K; ++k)
        if (s.best[k] >= 0 && std::find(c.begin(), c.end(), s.best[k]) == c.end()) c.push_back(s.best[k]);
    return c;
}

// ---------------------------------------------------------------- d = 2

// Andrew's monotone chain over idx (sorted lexicographically, no two equal), strict turns: the extreme
// points counterclockwise from the lexicographically smallest.
std::vector<i64> monotone_chain(const double* p, const std::vector<i64>& idx, const Predicates& P) {
    const size_t m = idx.size();
    if (m < 3) return idx;
    std::vector<i64> h(2 * m);
    size_t k = 0;
    for (size_t i = 0; i < m; ++i) {
        while (k >= 2 && P.orient2d(p + 2 * h[k - 2], p + 2 * h[k - 1], p + 2 * idx[i]) <= 0) --k;
        h[k++] = idx[i];
    }
    for (size_t i = m - 1, t = k + 1; i-- > 0;) {
        while (k >= t && P.orient2d(p + 2 * h[k - 2], p + 2 * h[k - 1], p + 2 * idx[i]) <= 0) --k;
        h[k++] = idx[i];
    }
    h.resize(k - 1);
    return h;
}

void sort_lex(const double* p, int d, std::vector<i64>& idx) {
    std::sort(idx.begin(), idx.end(), [=](i64 i, i64 j) {
        const double *a = p + i * d, *b = p + j * d;
        if (lex_less(a, b, d)) return true;
        if (lex_less(b, a, d)) return false;
        return i < j;
    });
}

// Keeps the first of each run of equal points (the smallest index, as idx is sorted by (point, index)).
void unique_points(const double* p, int d, std::vector<i64>& idx) {
    size_t w = 0;
    for (size_t r = 0; r < idx.size(); ++r)
        if (w == 0 || !same_point(p + idx[r] * d, p + idx[w - 1] * d, d)) idx[w++] = idx[r];
    idx.resize(w);
}

struct Edge2 {
    double ex, ey, h, B;
};

std::vector<i64> hull2(const double* p, i64 n, const Scan<2>& sc, const Predicates& P, bool safe, bool split) {
    std::vector<i64> cand = candidates(sc);
    sort_lex(p, 2, cand);
    unique_points(p, 2, cand);
    const std::vector<i64> poly = monotone_chain(p, cand, P);
    std::vector<Edge2> edges;
    if (safe && poly.size() >= 3) {
        const double o[2] = {sc.lo[0], sc.lo[1]};
        const double E[2] = {sc.hi[0] - sc.lo[0], sc.hi[1] - sc.lo[1]};
        for (size_t k = 0; k < poly.size(); ++k) {
            const double *a = p + 2 * poly[k], *b = p + 2 * poly[(k + 1) % poly.size()];
            Edge2 g;
            g.ex = b[0] - a[0];
            g.ey = b[1] - a[1];
            const double ax = a[0] - o[0], ay = a[1] - o[1];
            g.h = g.ex * ay - g.ey * ax;
            g.B = (10.0 * kU) * (std::fabs(g.ex) * E[1] + std::fabs(g.ey) * E[0]);
            if (!(g.B >= kBoundMin && g.B <= std::numeric_limits<double>::max())) {
                edges.clear();
                break;
            }
            edges.push_back(g);
        }
    }
    std::vector<i64> kept;
    if (edges.empty()) {
        kept.resize(static_cast<size_t>(n));
        for (i64 i = 0; i < n; ++i) kept[static_cast<size_t>(i)] = i;
    } else {
        const double ox = sc.lo[0], oy = sc.lo[1];
        const i64 nt = (n + kChunk - 1) / kChunk;
        std::vector<std::vector<i64>> part(static_cast<size_t>(nt));
        const Edge2* eg = edges.data();
        const size_t ne = edges.size();
        run_tasks(nt, split, [&](i64 t) {
            auto& out = part[static_cast<size_t>(t)];
            for (i64 i = t * kChunk, e = std::min(n, (t + 1) * kChunk); i < e; ++i) {
                const double qx = p[2 * i] - ox, qy = p[2 * i + 1] - oy;
                bool inside = true;
                for (size_t k = 0; k < ne; ++k) {
                    const double s = (eg[k].ex * qy - eg[k].ey * qx) - eg[k].h;
                    if (!(s > eg[k].B)) {
                        inside = false;
                        break;
                    }
                }
                if (!inside) out.push_back(i);
            }
        });
        for (auto& v : part) kept.insert(kept.end(), v.begin(), v.end());
    }
    // Every point may be kept (all extreme, as on a circle): the merge sort by tasks, the total order's result.
    tri::sort_parallel(
        kept,
        [=](i64 i, i64 j) {
            const double *a = p + 2 * i, *b = p + 2 * j;
            if (lex_less(a, b, 2)) return true;
            if (lex_less(b, a, 2)) return false;
            return i < j;
        },
        split
    );  // fmt: skip
    unique_points(p, 2, kept);
    return monotone_chain(p, kept, P);
}

// ---------------------------------------------------------------- d = 3: quickhull

struct Face {
    i32 v[3];   // point indices, counterclockwise seen from outside
    i32 nb[3];  // nb[k]: the face across the edge opposite v[k]
    double n[3], off;  // the float64 plane for the farthest-point choice: n.p - off
    i32 head, count, far;
    double fard;
    i64 mark;
    bool alive;
};

struct StaticPlane {
    double n[3], h, B;
    bool usable;
};

class Quickhull {
public:
    Quickhull(const double* p, i64 n, const Predicates& P) : p_(p), P_(P), next_(static_cast<size_t>(n), -1) {
        first_.assign(static_cast<size_t>(n), -1);
        second_.assign(static_cast<size_t>(n), -1);
        stampv_.assign(static_cast<size_t>(n), 0);
    }

    // The tetrahedron a, b, c, d (not coplanar).
    void start(i32 a, i32 b, i32 c, i32 d) {
        const i32 t[4] = {a, b, c, d};
        i32 ids[4];
        for (int k = 0; k < 4; ++k) {
            i32 v[3], m = 0;
            for (int j = 0; j < 4; ++j)
                if (j != k) v[m++] = t[j];
            if (orient(v[0], v[1], v[2], t[k]) > 0) std::swap(v[1], v[2]);
            ids[k] = new_face(v[0], v[1], v[2]);
        }
        for (int k = 0; k < 4; ++k) {
            Face& f = f_[ids[k]];
            for (int e = 0; e < 3; ++e) {
                const i32 u = f.v[(e + 1) % 3], w = f.v[(e + 2) % 3];
                for (int j = 0; j < 4; ++j) {
                    if (j == k) continue;
                    const Face& g = f_[ids[j]];
                    for (int q = 0; q < 3; ++q)
                        if (g.v[(q + 1) % 3] == w && g.v[(q + 2) % 3] == u) f.nb[e] = ids[j];
                }
            }
        }
    }

    // Puts each point in the list of the first live face (in index order) it is strictly outside.
    void assign_exact(const std::vector<i32>& pts) {
        const std::vector<i32> live = live_faces();
        for (const i32 q : pts)
            for (const i32 fi : live)
                if (orient_face(fi, q) > 0) {
                    push(fi, q);
                    break;
                }
        for (const i32 fi : live)
            if (f_[fi].count > 0) pending_.push_back(fi);
    }

    // The same for all n points with the static filter (tasks of kChunk points; step 3).
    void assign_all(i64 n, const double* lo, const double* hi, bool safe, bool split) {
        const std::vector<i32> live = live_faces();
        std::vector<StaticPlane> sp(live.size());
        const double E[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
        for (size_t k = 0; k < live.size(); ++k) sp[k] = static_plane(f_[live[k]], lo, E, safe);
        const i64 nt = (n + kChunk - 1) / kChunk;
        std::vector<std::vector<std::pair<i32, i32>>> part(static_cast<size_t>(nt));
        run_tasks(nt, split, [&](i64 t) {
            auto& out = part[static_cast<size_t>(t)];
            for (i64 i = t * kChunk, e = std::min(n, (t + 1) * kChunk); i < e; ++i) {
                const double* x = p_ + 3 * i;
                const double qx = x[0] - lo[0], qy = x[1] - lo[1], qz = x[2] - lo[2];
                for (size_t k = 0; k < live.size(); ++k) {
                    const StaticPlane& s = sp[k];
                    int side = 0;  // 1 outside, -1 inside or on the plane
                    if (s.usable) {
                        const double v = ((s.n[0] * qx + s.n[1] * qy) + s.n[2] * qz) - s.h;
                        if (v > s.B) side = 1;
                        else if (v < -s.B) side = -1;
                    }
                    if (side == 0) side = orient_face(live[k], static_cast<i32>(i)) > 0 ? 1 : -1;
                    if (side > 0) {
                        out.emplace_back(static_cast<i32>(i), live[k]);
                        break;
                    }
                }
            }
        });
        for (const auto& v : part)
            for (const auto& [q, fi] : v) push(fi, q);
        for (const i32 fi : live)
            if (f_[fi].count > 0) pending_.push_back(fi);
    }

    // Adds conflict points until no face has any.
    void run() {
        while (!pending_.empty()) {
            const i32 fi = pending_.back();
            pending_.pop_back();
            if (!f_[fi].alive || f_[fi].count == 0) continue;
            add(f_[fi].far, fi);
        }
    }

    std::vector<i32> live_faces() const {
        std::vector<i32> out;
        for (i32 k = 0; k < static_cast<i32>(f_.size()); ++k)
            if (f_[k].alive) out.push_back(k);
        return out;
    }
    const Face& face(i32 k) const { return f_[k]; }

private:
    const double* p_;
    const Predicates& P_;
    std::vector<Face> f_;
    std::vector<i32> free_, pending_, next_;
    std::vector<i32> first_, second_;  // the new face whose horizon edge starts / ends at a point
    std::vector<i64> stampv_;
    i64 stamp_ = 0;
    std::vector<i32> visible_, newf_, orphans_;

    struct Horizon {
        i32 u, w, outer, inner;
    };
    std::vector<Horizon> horizon_;

    const double* pt(i32 i) const { return p_ + 3 * static_cast<i64>(i); }
    int orient(i32 a, i32 b, i32 c, i32 d) const { return P_.orient3d(pt(a), pt(b), pt(c), pt(d)); }
    int orient_face(i32 fi, i32 q) const {
        const Face& f = f_[fi];
        return orient(f.v[0], f.v[1], f.v[2], q);
    }

    i32 new_face(i32 a, i32 b, i32 c) {
        i32 id;
        if (!free_.empty()) {
            id = free_.back();
            free_.pop_back();
        } else {
            id = static_cast<i32>(f_.size());
            f_.emplace_back();
        }
        Face& f = f_[id];
        f.v[0] = a, f.v[1] = b, f.v[2] = c;
        f.nb[0] = f.nb[1] = f.nb[2] = -1;
        f.head = -1, f.count = 0, f.far = -1, f.fard = -kInf;
        f.mark = 0;
        f.alive = true;
        const double *A = pt(a), *B = pt(b), *C = pt(c);
        const double e1[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
        const double e2[3] = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};
        f.n[0] = e1[1] * e2[2] - e1[2] * e2[1];
        f.n[1] = e1[2] * e2[0] - e1[0] * e2[2];
        f.n[2] = e1[0] * e2[1] - e1[1] * e2[0];
        f.off = (f.n[0] * A[0] + f.n[1] * A[1]) + f.n[2] * A[2];
        return id;
    }

    void push(i32 fi, i32 q) {
        Face& f = f_[fi];
        const double* x = pt(q);
        const double dist = ((f.n[0] * x[0] + f.n[1] * x[1]) + f.n[2] * x[2]) - f.off;
        next_[q] = f.head;
        f.head = q;
        ++f.count;
        if (f.far < 0 || dist > f.fard || (dist == f.fard && q < f.far)) f.far = q, f.fard = dist;
    }

    StaticPlane static_plane(const Face& f, const double* o, const double* E, bool safe) const {
        StaticPlane s{};
        s.usable = false;
        if (!safe) return s;
        const double *a = pt(f.v[0]), *b = pt(f.v[1]), *c = pt(f.v[2]);
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        double N[3];
        for (int k = 0; k < 3; ++k) {
            const int i = (k + 1) % 3, j = (k + 2) % 3;
            const double l = e1[i] * e2[j], r = e1[j] * e2[i];
            s.n[k] = l - r;
            N[k] = std::fabs(l) + std::fabs(r);
        }
        const double ax = a[0] - o[0], ay = a[1] - o[1], az = a[2] - o[2];
        s.h = (s.n[0] * ax + s.n[1] * ay) + s.n[2] * az;
        s.B = (18.0 * kU) * ((N[0] * E[0] + N[1] * E[1]) + N[2] * E[2]);
        s.usable = s.B >= kBoundMin && s.B <= std::numeric_limits<double>::max();
        return s;
    }

    void touch(i32 v) {
        if (stampv_[v] != stamp_) stampv_[v] = stamp_, first_[v] = second_[v] = -1;
    }

    // Adds point e, which is strictly outside face f0.
    void add(i32 e, i32 f0) {
        ++stamp_;
        const i64 vis = 2 * stamp_, not_vis = 2 * stamp_ + 1;
        visible_.clear();
        visible_.push_back(f0);
        f_[f0].mark = vis;
        for (size_t k = 0; k < visible_.size(); ++k) {
            const i32 fi = visible_[k];
            for (int j = 0; j < 3; ++j) {
                const i32 g = f_[fi].nb[j];
                if (f_[g].mark == vis || f_[g].mark == not_vis) continue;
                if (orient_face(g, e) > 0) {
                    f_[g].mark = vis;
                    visible_.push_back(g);
                } else {
                    f_[g].mark = not_vis;
                }
            }
        }
        horizon_.clear();
        for (const i32 fi : visible_)
            for (int j = 0; j < 3; ++j) {
                const i32 g = f_[fi].nb[j];
                if (f_[g].mark != vis) horizon_.push_back({f_[fi].v[(j + 1) % 3], f_[fi].v[(j + 2) % 3], g, fi});
            }
        newf_.clear();
        for (const Horizon& h : horizon_) {
            const i32 nf = new_face(h.u, h.w, e);
            f_[nf].nb[2] = h.outer;
            Face& g = f_[h.outer];
            for (int j = 0; j < 3; ++j)
                if (g.nb[j] == h.inner && g.v[(j + 1) % 3] == h.w && g.v[(j + 2) % 3] == h.u) g.nb[j] = nf;
            touch(h.u);
            touch(h.w);
            if (first_[h.u] >= 0 || second_[h.w] >= 0) throw std::logic_error("ConvexHull(): internal error");
            first_[h.u] = nf;
            second_[h.w] = nf;
            newf_.push_back(nf);
        }
        for (const i32 nf : newf_) {
            Face& f = f_[nf];
            f.nb[0] = first_[f.v[1]];   // across (w, e): the new face starting at w
            f.nb[1] = second_[f.v[0]];  // across (e, u): the new face ending at u
            if (f.nb[0] < 0 || f.nb[1] < 0) throw std::logic_error("ConvexHull(): internal error");
        }
        orphans_.clear();
        for (const i32 fi : visible_) {
            for (i32 q = f_[fi].head; q >= 0; q = next_[q])
                if (q != e) orphans_.push_back(q);
            f_[fi].alive = false;
            f_[fi].head = -1;
            f_[fi].count = 0;
            free_.push_back(fi);
        }
        for (const i32 q : orphans_)
            for (const i32 nf : newf_)
                if (orient_face(nf, q) > 0) {
                    push(nf, q);
                    break;
                }
        for (const i32 nf : newf_)
            if (f_[nf].count > 0) pending_.push_back(nf);
    }
};

// One output facet (a fan triangle) and the polygon it comes from.
struct Tri {
    i32 v[3];
    i32 group;
};

struct DisjointSets {
    std::vector<i32> parent;
    explicit DisjointSets(i32 n) : parent(static_cast<size_t>(n)) {
        for (i32 i = 0; i < n; ++i) parent[static_cast<size_t>(i)] = i;
    }
    i32 find(i32 x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    }
    void unite(i32 a, i32 b) {
        a = find(a), b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
};

struct Hull3Out {
    std::vector<Tri> tris;              // canonical facets (not yet sorted)
    std::vector<std::array<i32, 3>> plane;  // per group: the polygon's first fan triangle
};

// Steps 2-5 for d = 3. Returns false when the points are all on one plane.
bool hull3(const double* p, i64 n, const Scan<3>& sc, const Predicates& P, bool safe, bool split, Hull3Out& out) {
    const double* pts = p;
    auto at = [&](i64 i) { return pts + 3 * i; };
    auto noncollinear = [&](i64 a, i64 b, i64 c) {
        for (int k = 0; k < 3; ++k) {
            const int i = k, j = (k + 1) % 3;
            const double A[2] = {at(a)[i], at(a)[j]}, B[2] = {at(b)[i], at(b)[j]}, C[2] = {at(c)[i], at(c)[j]};
            if (P.orient2d(A, B, C) != 0) return true;
        }
        return false;
    };
    // The starting tetrahedron: from the candidates when they span space, else by exact scans of all points.
    const std::vector<i64> cand = candidates(sc);
    i64 a = cand.empty() ? -1 : cand[0], b = -1, c = -1, d = -1;
    auto pick = [&](const std::vector<i64>* from) {
        auto each = [&](auto&& fn) {
            if (from) {
                for (const i64 i : *from)
                    if (fn(i)) return;
            } else {
                for (i64 i = 0; i < n; ++i)
                    if (fn(i)) return;
            }
        };
        if (a < 0) return;
        b = c = d = -1;
        each([&](i64 i) { return !same_point(at(i), at(a), 3) && (b = i, true); });
        if (b < 0) return;
        each([&](i64 i) { return noncollinear(a, b, i) && (c = i, true); });
        if (c < 0) return;
        each([&](i64 i) { return P.orient3d(at(a), at(b), at(c), at(i)) != 0 && (d = i, true); });
    };
    pick(&cand);
    if (d < 0) pick(nullptr);
    if (d < 0) return false;
    Quickhull q(p, n, P);
    q.start(static_cast<i32>(a), static_cast<i32>(b), static_cast<i32>(c), static_cast<i32>(d));
    std::vector<i32> rest;
    for (const i64 i : cand)
        if (i != a && i != b && i != c && i != d) rest.push_back(static_cast<i32>(i));
    q.assign_exact(rest);
    q.run();
    q.assign_all(n, sc.lo, sc.hi, safe, split);
    q.run();

    // Step 5: the live faces, compacted, and their coplanar neighbours merged.
    const std::vector<i32> live = q.live_faces();
    const i32 F = static_cast<i32>(live.size());
    std::vector<i32> id(live.empty() ? 0 : static_cast<size_t>(*std::max_element(live.begin(), live.end()) + 1), -1);
    for (i32 k = 0; k < F; ++k) id[live[k]] = k;
    std::vector<std::array<i32, 3>> tv(static_cast<size_t>(F)), tn(static_cast<size_t>(F));
    for (i32 k = 0; k < F; ++k) {
        const Face& f = q.face(live[k]);
        for (int j = 0; j < 3; ++j) tv[k][j] = f.v[j], tn[k][j] = id[f.nb[j]];
    }
    const i64 nt = (F + kFacetChunk - 1) / kFacetChunk;
    std::vector<std::vector<std::pair<i32, i32>>> flat(static_cast<size_t>(nt));
    run_tasks(nt, split && F >= kFacetChunk, [&](i64 t) {
        for (i32 k = static_cast<i32>(t * kFacetChunk), e = static_cast<i32>(std::min<i64>(F, (t + 1) * kFacetChunk));
             k < e; ++k)
            for (int j = 0; j < 3; ++j) {
                const i32 g = tn[k][j];
                if (g < k) continue;
                int opp = 0;
                while (tn[g][opp] != k) ++opp;
                if (P.orient3d(at(tv[k][0]), at(tv[k][1]), at(tv[k][2]), at(tv[g][opp])) == 0)
                    flat[static_cast<size_t>(t)].emplace_back(k, g);
            }
    });
    DisjointSets ds(F);
    bool any_flat = false;
    for (const auto& v : flat)
        for (const auto& [x, y] : v) ds.unite(x, y), any_flat = true;

    auto rotate_min = [](std::array<i32, 3> t) {
        int m = 0;
        for (int j = 1; j < 3; ++j)
            if (t[j] < t[m]) m = j;
        return std::array<i32, 3>{t[m], t[(m + 1) % 3], t[(m + 2) % 3]};
    };
    // Groups: faces by root (counting sort), each root's faces in index order.
    std::vector<i32> root(static_cast<size_t>(F)), start(static_cast<size_t>(F) + 1, 0),
        members(static_cast<size_t>(F));
    for (i32 k = 0; k < F; ++k) root[k] = any_flat ? ds.find(k) : k, ++start[root[k] + 1];
    for (i32 k = 0; k < F; ++k) start[k + 1] += start[k];
    {
        std::vector<i32> fill(start.begin(), start.end() - 1);
        for (i32 k = 0; k < F; ++k) members[fill[root[k]]++] = k;
    }
    out.tris.clear();
    out.plane.clear();
    std::vector<std::pair<i32, i32>> bedges;
    std::vector<i32> cycle, corners;
    for (i32 r = 0; r < F; ++r) {
        const i32 s = start[r], e = start[r + 1];
        if (s == e) continue;
        const i32 g = static_cast<i32>(out.plane.size());
        if (e - s == 1) {
            const auto t = rotate_min(tv[members[s]]);
            out.tris.push_back({{t[0], t[1], t[2]}, g});
            out.plane.push_back(t);
            continue;
        }
        // The polygon's boundary: the group's edges whose neighbour is in another group, as directed edges.
        bedges.clear();
        for (i32 m = s; m < e; ++m) {
            const i32 k = members[m];
            for (int j = 0; j < 3; ++j)
                if (root[tn[k][j]] != r) bedges.emplace_back(tv[k][(j + 1) % 3], tv[k][(j + 2) % 3]);
        }
        std::sort(bedges.begin(), bedges.end());
        cycle.clear();
        i32 v = bedges[0].first;
        for (size_t step = 0; step < bedges.size(); ++step) {
            cycle.push_back(v);
            const auto it = std::lower_bound(bedges.begin(), bedges.end(), std::make_pair(v, i32{-1}));
            if (it == bedges.end() || it->first != v) throw std::logic_error("ConvexHull(): internal error");
            v = it->second;
        }
        if (v != cycle[0]) throw std::logic_error("ConvexHull(): internal error");
        // Exact collinearity in the facet's plane: a coordinate projection that keeps the plane's triangles flat-free.
        const auto& t0 = tv[members[s]];
        int axis = 0;
        for (; axis < 3; ++axis) {
            const int i = axis, j = (axis + 1) % 3;
            const double A[2] = {at(t0[0])[i], at(t0[0])[j]}, B[2] = {at(t0[1])[i], at(t0[1])[j]};
            const double C[2] = {at(t0[2])[i], at(t0[2])[j]};
            if (P.orient2d(A, B, C) != 0) break;
        }
        const int pi = axis, pj = (axis + 1) % 3;
        corners.clear();
        const size_t L = cycle.size();
        for (size_t k = 0; k < L; ++k) {
            const i32 u = cycle[(k + L - 1) % L], w = cycle[k], x = cycle[(k + 1) % L];
            const double A[2] = {at(u)[pi], at(u)[pj]}, B[2] = {at(w)[pi], at(w)[pj]}, C[2] = {at(x)[pi], at(x)[pj]};
            if (P.orient2d(A, B, C) != 0) corners.push_back(w);
        }
        if (corners.size() < 3) throw std::logic_error("ConvexHull(): internal error");
        const size_t m0 = static_cast<size_t>(std::min_element(corners.begin(), corners.end()) - corners.begin());
        std::rotate(corners.begin(), corners.begin() + static_cast<std::ptrdiff_t>(m0), corners.end());
        for (size_t k = 1; k + 1 < corners.size(); ++k)
            out.tris.push_back({{corners[0], corners[k], corners[k + 1]}, g});
        out.plane.push_back({corners[0], corners[1], corners[2]});
    }
    return true;
}

// ---------------------------------------------------------------- results

struct Result {
    int d = 0;
    i64 n = 0;
    std::vector<i64> vertices, simplices, neighbors;
    std::vector<double> equations;
    double area = 0.0, volume = 0.0;
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
};

// The unit outward normal of the 2-D edge a -> b (its inside on the left): (dy, -dx) / |b - a|.
void normal2(const double* a, const double* b, double* c) {
    double dx = b[0] - a[0], dy = b[1] - a[1];
    double L = euclid(a, b, 2);
    if (!(std::isfinite(dx) && std::isfinite(dy) && L >= kBoundMin && L <= std::numeric_limits<double>::max())) {
        Range r;
        r.add(a, 2);
        r.add(b, 2);
        const WideInt wx = sub(WideInt::scaled(b[0], r.lo), WideInt::scaled(a[0], r.lo));
        const WideInt wy = sub(WideInt::scaled(b[1], r.lo), WideInt::scaled(a[1], r.lo));
        int top = INT_MIN;
        if (wx.sign() != 0) top = std::max(top, wide_top(wx, r.lo));
        if (wy.sign() != 0) top = std::max(top, wide_top(wy, r.lo));
        dx = wx.to_double(i64{r.lo} - top);
        dy = wy.to_double(i64{r.lo} - top);
        const double zero[2] = {0.0, 0.0}, w[2] = {dx, dy};
        L = euclid(w, zero, 2);
    }
    c[0] = dy / L;
    c[1] = -dx / L;
}

// The cross product (b - a) x (c - a) by the value kernel (each component a det2 of a projection).
void cross3(const double* a, const double* b, const double* c, bool safe2, double* nv) {
    for (int k = 0; k < 3; ++k) {
        const int i = (k + 1) % 3, j = (k + 2) % 3;
        const double A[2] = {a[i], a[j]}, B[2] = {b[i], b[j]}, C[2] = {c[i], c[j]};
        nv[k] = safe2 ? det2(B, C, A) : det2_any(B, C, A);
    }
}

// The unit outward normal of the facet a, b, c (counterclockwise from outside).
void normal3(const double* a, const double* b, const double* c, bool safe2, double* u) {
    double nv[3];
    cross3(a, b, c, safe2, nv);
    const double big = std::max({std::fabs(nv[0]), std::fabs(nv[1]), std::fabs(nv[2])});
    if (!(big >= 0x1p-960 && big <= 0x1p960)) {
        WideInt w[3];
        i64 e[3];
        int top = INT_MIN;
        for (int k = 0; k < 3; ++k) {
            const int i = (k + 1) % 3, j = (k + 2) % 3;
            const double A[2] = {a[i], a[j]}, B[2] = {b[i], b[j]}, C[2] = {c[i], c[j]};
            const double* q[3] = {B, C, A};
            WideInt v[4];
            int lo = 0;
            if (!detail::wide_differences<3, 2>(q, v, &lo)) {
                e[k] = 0;
                continue;
            }
            w[k] = orient2d_det(v[0], v[1], v[2], v[3]);
            e[k] = 2 * i64{lo};
            if (w[k].sign() != 0) top = std::max(top, wide_top(w[k], e[k]));
        }
        for (int k = 0; k < 3; ++k) nv[k] = w[k].sign() == 0 ? 0.0 : w[k].to_double(e[k] - top);
    }
    const double zero[3] = {0.0, 0.0, 0.0};
    const double L = euclid(nv, zero, 3);
    for (int k = 0; k < 3; ++k) u[k] = nv[k] / L;
}

template <int D>
double offset(const double* c, const double* x) {
    double s = c[0] * x[0];
    for (int k = 1; k < D; ++k) s = s + c[k] * x[k];
    if (std::isfinite(s)) return -s;
    // A partial sum overflowed (coordinates near DBL_MAX): the same sum of terms scaled by 2^-4 (exact but for
    // terms below 2^-1018, within the subnormal allowance of §4), scaled back; inf only when -c.x overflows.
    s = c[0] * (x[0] * 0x1p-4);
    for (int k = 1; k < D; ++k) s = s + c[k] * (x[k] * 0x1p-4);
    return -(s * 0x1p4);
}

HullError build2(const double* p, i64 n, const Scan<2>& sc, bool split, Result& R) {
    const Range rg = range_of(sc);
    const Predicates P(rg);
    const bool safe = rg.safe(2);
    std::vector<i64> ring = hull2(p, n, sc, P, safe, split);
    if (ring.size() < 3) {
        HullError err;
        err.distinct = count_distinct(p, n, 2);
        err.kind = err.distinct < 3 ? HullError::TooFew : HullError::Collinear;
        return err;
    }
    const size_t h = ring.size();
    const size_t m0 = static_cast<size_t>(std::min_element(ring.begin(), ring.end()) - ring.begin());
    std::rotate(ring.begin(), ring.begin() + static_cast<std::ptrdiff_t>(m0), ring.end());
    R.vertices = ring;
    // Facets: the edges (ring[k], ring[k + 1]) in order of their first point; pos[k] is edge k's row.
    std::vector<size_t> order(h);
    for (size_t k = 0; k < h; ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return ring[x] < ring[y]; });
    std::vector<size_t> row(h);
    for (size_t r = 0; r < h; ++r) row[order[r]] = r;
    R.simplices.resize(2 * h);
    R.neighbors.resize(2 * h);
    R.equations.resize(3 * h);
    std::vector<double> len(h);
    for (size_t r = 0; r < h; ++r) {
        const size_t k = order[r];
        const i64 i = ring[k], j = ring[(k + 1) % h];
        R.simplices[2 * r] = i, R.simplices[2 * r + 1] = j;
        R.neighbors[2 * r] = static_cast<i64>(row[(k + 1) % h]);    // opposite i: the edge from j
        R.neighbors[2 * r + 1] = static_cast<i64>(row[(k + h - 1) % h]);  // opposite j: the edge into i
        double c[2];
        normal2(p + 2 * i, p + 2 * j, c);
        R.equations[3 * r] = c[0], R.equations[3 * r + 1] = c[1];
        R.equations[3 * r + 2] = offset<2>(c, p + 2 * std::min(i, j));
        len[r] = euclid(p + 2 * i, p + 2 * j, 2);
    }
    R.area = blocked_sum(len, split);  // the perimeter
    std::vector<double> tri(h - 2);
    const double* o = p + 2 * ring[0];
    for (size_t k = 1; k + 1 < h; ++k) {
        const double *b = p + 2 * ring[k], *c = p + 2 * ring[k + 1];
        tri[k - 1] = rg.safe(2) ? det2(b, c, o) : det2_any(b, c, o);
    }
    R.volume = blocked_sum(tri, split) * 0.5;  // the enclosed area
    return {};
}

HullError build3(const double* p, i64 n, const Scan<3>& sc, bool split, Result& R) {
    const Range rg = range_of(sc);
    const Predicates P(rg);
    const bool safe2 = rg.safe(2), safe3 = rg.safe(3);
    Hull3Out H;
    if (!hull3(p, n, sc, P, safe3, split, H)) {
        HullError err;
        err.distinct = count_distinct(p, n, 3);
        err.kind = err.distinct < 4 ? HullError::TooFew : HullError::Coplanar;
        return err;
    }
    // The extreme points (ascending) and their ranks.
    std::vector<unsigned char> mark(static_cast<size_t>(n), 0);
    for (const Tri& t : H.tris)
        for (const i32 v : t.v) mark[static_cast<size_t>(v)] = 1;
    std::vector<i32> rank(static_cast<size_t>(n), -1);
    for (i64 i = 0; i < n; ++i)
        if (mark[static_cast<size_t>(i)])
            rank[static_cast<size_t>(i)] = static_cast<i32>(R.vertices.size()), R.vertices.push_back(i);
    const i64 h = static_cast<i64>(R.vertices.size());
    // Rows in lexicographic order: by first point (counting sort over the ranks), then (second, third).
    const i64 F = static_cast<i64>(H.tris.size());
    std::vector<i64> start(static_cast<size_t>(h) + 1, 0);
    for (const Tri& t : H.tris) ++start[static_cast<size_t>(rank[t.v[0]]) + 1];
    for (i64 k = 0; k < h; ++k) start[k + 1] += start[k];
    std::vector<i32> ord(static_cast<size_t>(F));
    {
        std::vector<i64> fill(start.begin(), start.end() - 1);
        for (i64 k = 0; k < F; ++k) ord[static_cast<size_t>(fill[rank[H.tris[k].v[0]]]++)] = static_cast<i32>(k);
    }
    for (i64 k = 0; k < h; ++k)
        std::sort(ord.begin() + start[k], ord.begin() + start[k + 1], [&](i32 x, i32 y) {
            const Tri &a = H.tris[x], &b = H.tris[y];
            return a.v[1] != b.v[1] ? a.v[1] < b.v[1] : a.v[2] < b.v[2];
        });
    R.simplices.resize(static_cast<size_t>(3 * F));
    for (i64 r = 0; r < F; ++r)
        for (int j = 0; j < 3; ++j) R.simplices[static_cast<size_t>(3 * r + j)] = H.tris[ord[r]].v[j];
    // Neighbours: directed edges bucketed by their first point's rank, each bucket by its second point.
    struct DEdge {
        i32 to, row;
        int k;  // the vertex opposite the edge
    };
    std::vector<i64> es(static_cast<size_t>(h) + 1, 0);
    for (i64 r = 0; r < F; ++r)
        for (int j = 0; j < 3; ++j) ++es[static_cast<size_t>(rank[R.simplices[3 * r + j]]) + 1];
    for (i64 k = 0; k < h; ++k) es[k + 1] += es[k];
    std::vector<DEdge> de(static_cast<size_t>(3 * F));
    {
        std::vector<i64> fill(es.begin(), es.end() - 1);
        for (i64 r = 0; r < F; ++r)
            for (int j = 0; j < 3; ++j) {
                const i64 from = R.simplices[3 * r + (j + 1) % 3], to = R.simplices[3 * r + (j + 2) % 3];
                de[static_cast<size_t>(fill[rank[from]]++)] = {static_cast<i32>(to), static_cast<i32>(r), j};
            }
    }
    for (i64 k = 0; k < h; ++k)
        std::sort(de.begin() + es[k], de.begin() + es[k + 1], [](const DEdge& x, const DEdge& y) {
            return x.to < y.to;
        });
    R.neighbors.assign(static_cast<size_t>(3 * F), -1);
    for (i64 r = 0; r < F; ++r)
        for (int j = 0; j < 3; ++j) {
            const i64 from = R.simplices[3 * r + (j + 1) % 3], to = R.simplices[3 * r + (j + 2) % 3];
            // The twin edge to -> from, in the bucket of `to`.
            const auto b = de.begin() + es[rank[to]], e = de.begin() + es[rank[to] + 1];
            const auto it = std::lower_bound(b, e, from, [](const DEdge& x, i64 v) { return x.to < v; });
            if (it == e || it->to != from) throw std::logic_error("ConvexHull(): internal error");
            R.neighbors[static_cast<size_t>(3 * r + j)] = it->row;
        }
    // Equations (one per polygon), facet areas and the volume terms, by tasks of facets.
    const i64 G = static_cast<i64>(H.plane.size());
    std::vector<double> geq(static_cast<size_t>(4 * G));
    const i64 ng = (G + kFacetChunk - 1) / kFacetChunk;
    run_tasks(ng, split && G >= kFacetChunk, [&](i64 t) {
        for (i64 g = t * kFacetChunk, e = std::min(G, (t + 1) * kFacetChunk); g < e; ++g) {
            const auto& tr = H.plane[static_cast<size_t>(g)];
            double* c = geq.data() + 4 * g;
            normal3(p + 3 * i64{tr[0]}, p + 3 * i64{tr[1]}, p + 3 * i64{tr[2]}, safe2, c);
            c[3] = offset<3>(c, p + 3 * i64{tr[0]});
        }
    });
    R.equations.resize(static_cast<size_t>(4 * F));
    std::vector<double> area(static_cast<size_t>(F)), vol(static_cast<size_t>(F));
    const double* o = p + 3 * R.vertices[0];
    const i64 nf = (F + kFacetChunk - 1) / kFacetChunk;
    run_tasks(nf, split && F >= kFacetChunk, [&](i64 t) {
        const double zero[3] = {0.0, 0.0, 0.0};
        for (i64 r = t * kFacetChunk, e = std::min(F, (t + 1) * kFacetChunk); r < e; ++r) {
            const Tri& tr = H.tris[ord[r]];
            std::memcpy(R.equations.data() + 4 * r, geq.data() + 4 * i64{tr.group}, 4 * sizeof(double));
            const double *a = p + 3 * i64{tr.v[0]}, *b = p + 3 * i64{tr.v[1]}, *c = p + 3 * i64{tr.v[2]};
            double nv[3];
            cross3(a, b, c, safe2, nv);
            area[static_cast<size_t>(r)] = euclid(nv, zero, 3);
            vol[static_cast<size_t>(r)] = safe3 ? det3(o, a, b, c) : det3_any(o, a, b, c);
        }
    });
    R.area = blocked_sum(area, split) * 0.5;
    R.volume = blocked_sum(vol, split) / 6.0;
    return {};
}

NDArray f64_copy(const std::vector<double>& v, std::initializer_list<i64> shape) {
    const std::vector<i64> s(shape);
    NDArray a = new_array(DType::Float64, s);
    if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(double));
    return a;
}

NDArray i64_copy(const std::vector<i64>& v, std::initializer_list<i64> shape) {
    const std::vector<i64> s(shape);
    NDArray a = new_array(DType::Int64, s);
    if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(i64));
    return a;
}

nb::tuple convex_hull(nb::handle points) {
    if (!is_ndarray(points)) throw nb::type_error("_convex_hull(): expected an sd.ndarray");
    const NDArray& a = as_ndarray(points);
    if (a.ndim != 2 || a.dtype != DType::Float64 || (a.shape[1] != 2 && a.shape[1] != 3))
        throw nb::type_error("_convex_hull(): wrong array layout");
    const i64 n = a.shape[0];
    const int d = static_cast<int>(a.shape[1]);
    if (n > std::numeric_limits<i32>::max() / 2) throw nb::value_error("ConvexHull(): too many points");
    const i64 sh[] = {n, d};
    NDArray copy = new_array(DType::Float64, sh);
    auto* p = reinterpret_cast<double*>(copy.data);
    const bool split = n * d >= kSplitWork;
    Result R;
    R.d = d;
    R.n = n;
    HullError err;
    {
        const parallel::Unlocked unlocked(split);
        const i64 nt = (n + kChunk - 1) / kChunk;
        run_tasks(nt, split, [&](i64 t) {
            for (i64 i = t * kChunk, e = std::min(n, (t + 1) * kChunk); i < e; ++i)
                for (int k = 0; k < d; ++k)
                    std::memcpy(p + i * d + k, a.data + i * a.strides[0] + k * a.strides[1], sizeof(double));
        });
        if (d == 2) {
            const Scan<2> sc = scan<2>(p, n, split);
            if (!sc.finite) {
                err.kind = HullError::NotFinite;
            } else {
                for (int k = 0; k < 2; ++k) R.lo[k] = sc.lo[k], R.hi[k] = sc.hi[k];
                err = build2(p, n, sc, split, R);
            }
        } else {
            const Scan<3> sc = scan<3>(p, n, split);
            if (!sc.finite) {
                err.kind = HullError::NotFinite;
            } else {
                for (int k = 0; k < 3; ++k) R.lo[k] = sc.lo[k], R.hi[k] = sc.hi[k];
                err = build3(p, n, sc, split, R);
            }
        }
    }
    switch (err.kind) {
        case HullError::None: break;
        case HullError::NotFinite: throw nb::value_error("ConvexHull(): 'points' must be finite");
        case HullError::TooFew:
            throw nb::value_error(("ConvexHull(): needs at least " + std::to_string(d + 1) + " distinct points, got " +
                                   std::to_string(err.distinct))
                                      .c_str());
        case HullError::Collinear: throw nb::value_error("ConvexHull(): 'points' are collinear");
        case HullError::Coplanar: throw nb::value_error("ConvexHull(): 'points' are coplanar");
    }
    const i64 h = static_cast<i64>(R.vertices.size()), F = static_cast<i64>(R.simplices.size()) / d;
    std::vector<double> lo(R.lo, R.lo + d), hi(R.hi, R.hi + d);
    return nb::make_tuple(
        wrap(std::move(copy)), wrap(i64_copy(R.vertices, {h})), wrap(i64_copy(R.simplices, {F, d})),
        wrap(i64_copy(R.neighbors, {F, d})), wrap(f64_copy(R.equations, {F, d + 1})), R.area, R.volume,
        wrap(f64_copy(lo, {d})), wrap(f64_copy(hi, {d}))
    );
}

}  // namespace

void bind_hull(nb::module_& m) { m.def("_convex_hull", &convex_hull, nb::arg("points")); }

}  // namespace sd::spatial
