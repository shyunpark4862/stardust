// What the Delaunay triangulations of d = 2 (delaunay2.cpp) and d = 3 (delaunay3.cpp) share
// (docs/design/spatial_geometry.md §2.2, §3.5, §3.10, §3.11; milestone SPT9, decisions SPT124-SPT139):
//
//   lex_order_parallel   the indices sorted by (coordinates, index): a merge sort whose runs and merge
//                        rounds depend on n alone (SPT124), so the order is the total order's whatever
//                        the thread count. The convex hull (hull.cpp) can use it too.
//   range_of             the Range (predicates.hpp) of many coordinates, by chunks.
//   Pt<D>                a point with its index, the record the builders partition.
//   Triangulation        the result: the points, the canonical simplices and neighbours, the vertex to
//                        simplex map, the representatives of equal points, the predicates of the set
//                        and the location grid.
//   canonical_form<D>    any positively oriented simplices with their neighbours -> the canonical form of
//                        §2.2 (rows rotated by an even permutation to the smallest index first, rows in
//                        lexicographic order, neighbours renumbered): a counting sort on the first vertex
//                        and a sort inside each bucket, so the arrays are the same for any thread count.
//   vertex_to_simplex    the smallest simplex of every vertex (an atomic minimum: order-free).
//   build_locator,       the location grid and the walk of find_simplex (§3.5): a closed simplex holding
//   find_simplex         the point, the smallest such simplex (by the facets through the point), -1 outside.
//   sort_parallel,       a merge sort of any records by a total order (runs and rounds fixed by n), and the
//   brio_order_parallel  BRIO insertion order of sfc.hpp's brio_order made with it (d = 3, SPT157).
//   convex_hull, coplanar_rows, vertex_neighbors2/3, transform2/3: the attributes made on first use.
//
// Vertex indices are int32 inside a build (n < 2^31; the builders check) and int64 in the results (SPT93).
#pragma once

#include "parallel/pool.hpp"
#include "spatial/predicates.hpp"
#include "spatial/sfc.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sd::spatial::tri {

// The array area's rule: from 2^15 units of work a pass splits and runs without the GIL.
inline constexpr std::int64_t kParallelWork = 1 << 15;
inline constexpr std::int64_t kSortRun = 1 << 14;    // points a task sorts before the merge rounds
inline constexpr std::int64_t kChunk = 1 << 13;      // simplices or points a task of a linear pass takes
inline constexpr std::int64_t kQueryBlock = 256;     // find_simplex: query points per task (§3.5)
inline constexpr std::int64_t kQueryWork = 128;      // work units of one query point
inline constexpr std::int64_t kPointsPerCell = 4;    // the location grid has about m / 4 cells (§3.5)
inline constexpr std::int64_t kQueryCellSort = 1 << 12;  // find_simplex: from here the points go cell by cell

template <class F>
inline void run_tasks(std::int64_t ntasks, bool split, F&& fn) {
    if (!split || ntasks <= 1) {
        for (std::int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

inline std::int64_t chunks_of(std::int64_t n, std::int64_t chunk) { return (n + chunk - 1) / chunk; }

// Relaxed atomics on plain integers (the counting sorts and the minima): the results do not depend on
// the order in which the tasks get there.
inline std::int64_t fetch_add(std::int64_t& x, std::int64_t v) { return __atomic_fetch_add(&x, v, __ATOMIC_RELAXED); }

inline void fetch_min(std::int64_t& x, std::int64_t v) {
    std::int64_t cur = __atomic_load_n(&x, __ATOMIC_RELAXED);
    while (v < cur && !__atomic_compare_exchange_n(&x, &cur, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

// ---------------------------------------------------------------- the lexicographic order

// The indices 0..n-1 sorted by (coordinates, index) (-0.0 equals 0.0): runs of kSortRun sorted by tasks,
// then merge rounds of pairs of runs. Every run and every round depends on n alone and the order is total,
// so the result is lex_order's (sfc.hpp) for any thread count.
inline std::vector<std::int64_t> lex_order_parallel(const double* pts, std::int64_t n, int d, bool split) {
    std::vector<std::int64_t> a(static_cast<std::size_t>(n)), b;
    for (std::int64_t i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = i;
    auto less = [=](std::int64_t i, std::int64_t j) {
        const double *p = pts + i * d, *q = pts + j * d;
        for (int k = 0; k < d; ++k)
            if (p[k] != q[k]) return p[k] < q[k];
        return i < j;
    };
    const std::int64_t runs = chunks_of(n, kSortRun);
    run_tasks(runs, split, [&](std::int64_t r) {
        std::sort(a.begin() + r * kSortRun, a.begin() + std::min(n, (r + 1) * kSortRun), less);
    });
    if (runs <= 1) return a;
    b.resize(a.size());
    std::vector<std::int64_t>*src = &a, *dst = &b;
    for (std::int64_t w = kSortRun; w < n; w *= 2) {
        const std::int64_t pairs = chunks_of(n, 2 * w);
        run_tasks(pairs, split, [&](std::int64_t p) {
            const std::int64_t lo = 2 * w * p, mid = std::min(n, lo + w), hi = std::min(n, lo + 2 * w);
            std::merge(
                src->begin() + lo, src->begin() + mid, src->begin() + mid, src->begin() + hi, dst->begin() + lo,
                less
            );  // fmt: skip
        });
        std::swap(src, dst);
    }
    return std::move(*src);
}

// Any records sorted by a total order `less`: runs of kSortRun sorted by tasks, then merge rounds of pairs of
// runs, all fixed by the size alone, so the result is std::sort's for any thread count.
template <class T, class Less>
inline void sort_parallel(std::vector<T>& a, Less less, bool split) {
    const auto n = static_cast<std::int64_t>(a.size());
    const std::int64_t runs = chunks_of(n, kSortRun);
    run_tasks(runs, split, [&](std::int64_t r) {
        std::sort(a.begin() + r * kSortRun, a.begin() + std::min(n, (r + 1) * kSortRun), less);
    });
    if (runs <= 1) return;
    std::vector<T> b(a.size());
    std::vector<T>*src = &a, *dst = &b;
    for (std::int64_t w = kSortRun; w < n; w *= 2) {
        const std::int64_t pairs = chunks_of(n, 2 * w);
        run_tasks(pairs, split, [&](std::int64_t p) {
            const std::int64_t lo = 2 * w * p, mid = std::min(n, lo + w), hi = std::min(n, lo + 2 * w);
            std::merge(
                src->begin() + lo, src->begin() + mid, src->begin() + mid, src->begin() + hi, dst->begin() + lo,
                less
            );  // fmt: skip
        });
        std::swap(src, dst);
    }
    if (src != &a) a.swap(b);
}

// sfc.hpp's brio_order with the keys made by tasks and the records sorted by sort_parallel: the same total
// order (round, Hilbert key, lexicographic rank), so the same insertion order for any thread count.
inline std::vector<std::int64_t> brio_order_parallel(
    const double* pts, int d, const std::vector<std::int64_t>& distinct, bool split
) {  // fmt: skip
    const auto m = static_cast<std::int64_t>(distinct.size());
    std::vector<std::int64_t> out(static_cast<std::size_t>(m));
    if (m == 0) return out;
    double lo[3], hi[3];
    for (int k = 0; k < d; ++k) lo[k] = hi[k] = pts[distinct[0] * d + k];
    for (std::int64_t r = 1; r < m; ++r)
        for (int k = 0; k < d; ++k) {
            const double v = pts[distinct[static_cast<std::size_t>(r)] * d + k];
            lo[k] = std::min(lo[k], v);
            hi[k] = std::max(hi[k], v);
        }
    struct Item {
        int round;
        std::uint64_t key;
        std::int64_t rank;
    };
    const int rounds = brio_rounds(m);
    std::vector<Item> items(static_cast<std::size_t>(m));
    run_tasks(chunks_of(m, kChunk), split, [&](std::int64_t c) {
        for (std::int64_t r = c * kChunk, e = std::min(m, (c + 1) * kChunk); r < e; ++r) {
            const double* p = pts + distinct[static_cast<std::size_t>(r)] * d;
            std::uint32_t cell[3] = {0, 0, 0};
            for (int k = 0; k < d; ++k) cell[k] = grid_cell(p[k], lo[k], hi[k], kHilbertBits);
            const int zeros = std::countr_zero(splitmix64(static_cast<std::uint64_t>(r)) | (std::uint64_t{1} << 63));
            const int round = rounds - 1 - std::min(zeros, rounds - 1);  // 0 is inserted first
            const std::uint64_t key =
                d == 3 ? hilbert_key_of<3>(cell, kHilbertBits) : hilbert_key_of<2>(cell, kHilbertBits);
            items[static_cast<std::size_t>(r)] = {round, key, r};
        }
    });
    sort_parallel(
        items,
        [](const Item& a, const Item& b) {
            if (a.round != b.round) return a.round < b.round;
            if (a.key != b.key) return a.key < b.key;
            return a.rank < b.rank;
        },
        split
    );  // fmt: skip
    for (std::int64_t r = 0; r < m; ++r)
        out[static_cast<std::size_t>(r)] = distinct[static_cast<std::size_t>(items[static_cast<std::size_t>(r)].rank)];
    return out;
}

// The Range of x[0..count), by chunks (the largest hi and the smallest lo: order-free).
inline Range range_of(const double* x, std::int64_t count, bool split) {
    const std::int64_t nc = chunks_of(count, 8 * kChunk);
    std::vector<Range> part(static_cast<std::size_t>(std::max<std::int64_t>(nc, 1)));
    run_tasks(nc, split, [&](std::int64_t c) {
        Range r;
        r.add(x + c * 8 * kChunk, std::min(count, (c + 1) * 8 * kChunk) - c * 8 * kChunk);
        part[static_cast<std::size_t>(c)] = r;
    });
    Range out;
    for (const Range& r : part) {
        out.hi = std::max(out.hi, r.hi);
        out.lo = std::min(out.lo, r.lo);
    }
    return out;
}

// ---------------------------------------------------------------- points and the result

template <int D>
struct Pt {
    double c[D];
    std::int64_t id;  // the index in the input
};

template <int D>
struct Locator {
    std::int64_t g[D] = {};  // cells per axis
    double lo[D] = {}, hi[D] = {};
    std::vector<std::int64_t> start;  // a simplex near each cell's centre (row-major, axis 0 fastest)
};

struct Triangulation {
    int d = 2;
    std::int64_t n = 0;                // input points
    std::int64_t m = 0;                // distinct points
    std::int64_t t = 0;                // simplices
    const double* pts = nullptr;       // (n, d), input order (the points array of the result)
    std::int64_t* simplices = nullptr;  // (t, d + 1), canonical
    std::int64_t* neighbors = nullptr;  // (t, d + 1)
    std::int64_t* v2s = nullptr;       // (n,) the smallest simplex of each point (the representative's)
    std::vector<std::int64_t> rep;     // (n,) the smallest index with the same coordinates
    Range range;                       // of the coordinates
    Predicates pred;                   // chosen from range
    Locator<2> loc2;
    Locator<3> loc3;
};

// ---------------------------------------------------------------- the canonical form (§2.2)

namespace detail {

// The even permutations of 4 positions with each first position: then the other three are rotated.
inline constexpr int kEven4[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};

// p[j]: the position in the simplex s of the canonical row's j-th vertex. The permutation is even
// (orientation kept): the smallest vertex first, and for d = 3 the smallest of the other three second.
template <int D>
inline void canonical_perm(const std::int32_t* s, int* p) {
    int m = 0;
    for (int k = 1; k <= D; ++k)
        if (s[k] < s[m]) m = k;
    if constexpr (D == 2) {
        p[0] = m;
        p[1] = (m + 1) % 3;
        p[2] = (m + 2) % 3;
    } else {
        const int* b = kEven4[m];
        int r = 1;
        if (s[b[2]] < s[b[r]]) r = 2;
        if (s[b[3]] < s[b[r]]) r = 3;
        p[0] = b[0];
        p[1] = b[r];
        p[2] = b[r == 3 ? 1 : r + 1];
        p[3] = b[r == 1 ? 3 : r - 1];
    }
}

}  // namespace detail

// s (t, D + 1): positively oriented simplices of the input indices; nb (t, D + 1): nb[i][k] the simplex
// across the facet opposite vertex k, -1 for none. Writes the canonical rows S and neighbours NB.
template <int D>
inline void canonical_form(
    std::int64_t n,
    std::int64_t t,
    const std::int32_t* s,
    const std::int32_t* nb,
    std::int64_t* S,
    std::int64_t* NB,
    bool split
) {
    constexpr int K = D + 1;
    std::vector<std::int64_t> start(static_cast<std::size_t>(n) + 1, 0);
    const std::int64_t nc = chunks_of(t, kChunk);
    run_tasks(nc, split, [&](std::int64_t c) {
        for (std::int64_t i = c * kChunk, e = std::min(t, (c + 1) * kChunk); i < e; ++i) {
            std::int32_t lo = s[i * K];
            for (int k = 1; k < K; ++k) lo = std::min(lo, s[i * K + k]);
            fetch_add(start[static_cast<std::size_t>(lo) + 1], 1);
        }
    });
    for (std::int64_t v = 0; v < n; ++v) start[static_cast<std::size_t>(v) + 1] += start[static_cast<std::size_t>(v)];
    std::vector<std::int64_t> cursor(start.begin(), start.end() - 1);
    std::vector<std::int64_t> order(static_cast<std::size_t>(t));  // order[slot]: the simplex at that slot
    std::vector<std::int32_t> rows(static_cast<std::size_t>(t * K));  // canonical rows, in the input's order
    run_tasks(nc, split, [&](std::int64_t c) {
        int p[K];
        for (std::int64_t i = c * kChunk, e = std::min(t, (c + 1) * kChunk); i < e; ++i) {
            detail::canonical_perm<D>(s + i * K, p);
            for (int j = 0; j < K; ++j) rows[static_cast<std::size_t>(i * K + j)] = s[i * K + p[j]];
            order[static_cast<std::size_t>(fetch_add(cursor[static_cast<std::size_t>(s[i * K + p[0]])], 1))] = i;
        }
    });
    // Each bucket (one first vertex) in the order of its rows' other vertices: rows are distinct.
    const std::int64_t vc = chunks_of(n, kChunk);
    run_tasks(vc, split, [&](std::int64_t c) {
        for (std::int64_t v = c * kChunk, e = std::min(n, (c + 1) * kChunk); v < e; ++v) {
            const std::int64_t a = start[static_cast<std::size_t>(v)], b = start[static_cast<std::size_t>(v) + 1];
            if (b - a < 2) continue;
            std::sort(order.begin() + a, order.begin() + b, [&](std::int64_t x, std::int64_t y) {
                const std::int32_t *rx = rows.data() + x * K, *ry = rows.data() + y * K;
                for (int j = 1; j < K; ++j)
                    if (rx[j] != ry[j]) return rx[j] < ry[j];
                return false;
            });
        }
    });
    std::vector<std::int64_t> slot(static_cast<std::size_t>(t));
    run_tasks(nc, split, [&](std::int64_t c) {
        for (std::int64_t q = c * kChunk, e = std::min(t, (c + 1) * kChunk); q < e; ++q)
            slot[static_cast<std::size_t>(order[static_cast<std::size_t>(q)])] = q;
    });
    run_tasks(nc, split, [&](std::int64_t c) {
        int p[K];
        for (std::int64_t q = c * kChunk, e = std::min(t, (c + 1) * kChunk); q < e; ++q) {
            const std::int64_t i = order[static_cast<std::size_t>(q)];
            detail::canonical_perm<D>(s + i * K, p);
            for (int j = 0; j < K; ++j) {
                S[q * K + j] = rows[static_cast<std::size_t>(i * K + j)];
                const std::int32_t o = nb[i * K + p[j]];
                NB[q * K + j] = o < 0 ? -1 : slot[static_cast<std::size_t>(o)];
            }
        }
    });
}

// v2s[i]: the smallest simplex holding point i's representative.
inline void vertex_to_simplex(Triangulation& T, bool split) {
    const int K = T.d + 1;
    std::int64_t* v2s = T.v2s;
    std::fill(v2s, v2s + T.n, INT64_MAX);
    run_tasks(chunks_of(T.t, kChunk), split, [&](std::int64_t c) {
        for (std::int64_t q = c * kChunk, e = std::min(T.t, (c + 1) * kChunk); q < e; ++q)
            for (int j = 0; j < K; ++j) fetch_min(v2s[T.simplices[q * K + j]], q);
    });
    for (std::int64_t i = 0; i < T.n; ++i) {
        const std::int64_t r = T.rep[static_cast<std::size_t>(i)];
        if (r != i) v2s[i] = v2s[r];
    }
}

// ---------------------------------------------------------------- locating a point (§3.5)

namespace detail {

// The sign of the simplex s with vertex k replaced by x: >= 0 for every k when the closed simplex holds x.
template <int D, bool Safe>
inline int orient_with(const Triangulation& T, const std::int64_t* s, int k, const double* x) {
    const double* q[D + 1];
    for (int j = 0; j <= D; ++j) q[j] = j == k ? x : T.pts + s[j] * D;
    if constexpr (D == 2) {
        return Safe ? orient2d(q[0], q[1], q[2]) : orient2d_wide(q[0], q[1], q[2]);
    } else {
        return Safe ? orient3d(q[0], q[1], q[2], q[3]) : orient3d_wide(q[0], q[1], q[2], q[3]);
    }
}

// The visibility walk from simplex t towards x: across a facet whose other side holds x strictly. A
// Delaunay triangulation has no cycle of such steps, whatever facet is taken first (Edelsbrunner), so the
// walk ends. Returns the simplex whose closed simplex holds x, or -1 when it leaves the hull (*last: the
// simplex it stopped in). The facets are tried from a position that turns with each step.
template <int D, bool Safe>
inline std::int64_t walk(const Triangulation& T, std::int64_t t, const double* x, std::int64_t* last) {
    constexpr int K = D + 1;
    std::int64_t prev = -1;
    int turn = 0;
    for (;;) {
        const std::int64_t* s = T.simplices + t * K;
        const std::int64_t* nb = T.neighbors + t * K;
        std::int64_t next = -2;
        for (int j = 0; j < K; ++j) {
            const int k = (j + turn) % K;
            if (prev >= 0 && nb[k] == prev) continue;
            if (orient_with<D, Safe>(T, s, k, x) < 0) {
                next = nb[k];
                break;
            }
        }
        if (last) *last = t;
        if (next == -2) return t;
        if (next < 0) return -1;
        prev = t;
        t = next;
        turn = turn + 1 == K ? 0 : turn + 1;
    }
}

// The smallest simplex whose closed simplex holds x, given one (t): through the facets x lies on.
template <int D, bool Safe>
inline std::int64_t smallest_holding(const Triangulation& T, std::int64_t t, const double* x) {
    constexpr int K = D + 1;
    const std::int64_t* s = T.simplices + t * K;
    int zeros = 0, other = -1;
    for (int k = 0; k < K; ++k) {
        if (orient_with<D, Safe>(T, s, k, x) == 0) ++zeros;
        else other = k;
    }
    if (zeros == 0) return t;
    if (zeros == D) return T.v2s[s[other]];  // x is the vertex `other`: every simplex of its star holds it
    std::int64_t best = t;
    std::int64_t seen[64];
    int nseen = 0, top = 0;
    std::vector<std::int64_t> more;  // an edge's star past 64 simplices (d = 3)
    auto visited = [&](std::int64_t u) {
        for (int i = 0; i < nseen; ++i)
            if (seen[i] == u) return true;
        return std::find(more.begin(), more.end(), u) != more.end();
    };
    std::vector<std::int64_t> stack{t};
    seen[nseen++] = t;
    while (top < static_cast<int>(stack.size())) {
        const std::int64_t u = stack[static_cast<std::size_t>(top++)];
        const std::int64_t* su = T.simplices + u * K;
        for (int k = 0; k < K; ++k) {
            const std::int64_t v = T.neighbors[u * K + k];
            if (v < 0 || visited(v) || orient_with<D, Safe>(T, su, k, x) != 0) continue;
            if (nseen < 64) seen[nseen++] = v;
            else more.push_back(v);
            stack.push_back(v);
            best = std::min(best, v);
        }
    }
    return best;
}

template <int D>
inline std::int64_t cell_on(double x, double lo, double hi, std::int64_t g) {
    const double width = 0.5 * hi - 0.5 * lo;
    if (!(width > 0.0)) return 0;
    const double f = (0.5 * x - 0.5 * lo) / width * static_cast<double>(g);
    if (!(f > 0.0)) return 0;
    return f >= static_cast<double>(g - 1) ? g - 1 : static_cast<std::int64_t>(f);
}

template <int D>
inline const Locator<D>& locator(const Triangulation& T) {
    if constexpr (D == 2) return T.loc2;
    else return T.loc3;
}

template <int D>
inline Locator<D>& locator(Triangulation& T) {
    if constexpr (D == 2) return T.loc2;
    else return T.loc3;
}

template <int D>
inline std::int64_t cell_of(const Locator<D>& L, const double* x) {
    std::int64_t c = 0;
    for (int k = D - 1; k >= 0; --k) c = c * L.g[k] + cell_on<D>(x[k], L.lo[k], L.hi[k], L.g[k]);
    return c;
}

// Whether the predicates may take their fast stages for the triangulation's points with x (SPT113).
template <int D>
inline bool safe_with(const Triangulation& T, const double* x) {
    Range r = T.range;
    for (int k = 0; k < D; ++k) r.add(x[k]);
    return r.safe(D);
}

template <int D>
inline std::int64_t walk_any(const Triangulation& T, std::int64_t t, const double* x, std::int64_t* last) {
    return safe_with<D>(T, x) ? walk<D, true>(T, t, x, last) : walk<D, false>(T, t, x, last);
}

}  // namespace detail

// The location grid (§3.5): about m / kPointsPerCell cells over the bounding box, the cells' numbers of
// each axis in proportion to its extent; each cell starts at a simplex holding (or nearest to) its centre,
// found by walking from the smallest point in the cell (or the cell before it on its row).
template <int D>
inline void build_locator(Triangulation& T, bool split) {
    Locator<D>& L = detail::locator<D>(T);
    for (int k = 0; k < D; ++k) {
        L.lo[k] = L.hi[k] = T.pts[k];
        for (std::int64_t i = 1; i < T.n; ++i) {
            L.lo[k] = std::min(L.lo[k], T.pts[i * D + k]);
            L.hi[k] = std::max(L.hi[k], T.pts[i * D + k]);
        }
    }
    // Cells per axis: the extents' ratios in log2 (finite for any finite box).
    const double cells = std::max(1.0, static_cast<double>(T.m) / static_cast<double>(kPointsPerCell));
    double logw[D], mean = 0.0;
    for (int k = 0; k < D; ++k) {
        logw[k] = std::log2(0.5 * L.hi[k] - 0.5 * L.lo[k]);
        mean += logw[k] / D;
    }
    std::int64_t total = 1;
    for (int k = 0; k < D; ++k) {
        const double g = std::exp2(std::log2(cells) / D + std::clamp(logw[k] - mean, -40.0, 40.0));
        L.g[k] = static_cast<std::int64_t>(std::clamp(std::round(g), 1.0, 1e6));
        total *= L.g[k];
    }
    while (total > 4 * static_cast<std::int64_t>(cells) + 16) {  // a very flat box: cut the longest axis
        int k = 0;
        for (int j = 1; j < D; ++j)
            if (L.g[j] > L.g[k]) k = j;
        total /= L.g[k];
        L.g[k] = std::max<std::int64_t>(1, L.g[k] / 2);
        total *= L.g[k];
    }
    L.start.assign(static_cast<std::size_t>(total), INT64_MAX);
    std::vector<std::int64_t>& first = L.start;  // the smallest representative in each cell, then the starts
    run_tasks(chunks_of(T.n, kChunk), split, [&](std::int64_t c) {
        for (std::int64_t i = c * kChunk, e = std::min(T.n, (c + 1) * kChunk); i < e; ++i)
            if (T.rep[static_cast<std::size_t>(i)] == i)
                fetch_min(first[static_cast<std::size_t>(detail::cell_of<D>(L, T.pts + i * D))], i);
    });
    const std::int64_t row = L.g[0], nrows = total / row;
    run_tasks(chunks_of(nrows, 64), split, [&](std::int64_t c) {
        for (std::int64_t r = c * 64, e = std::min(nrows, (c + 1) * 64); r < e; ++r) {
            std::int64_t cur = 0;
            for (std::int64_t j = 0; j < row; ++j) {
                const std::int64_t v = first[static_cast<std::size_t>(r * row + j)];
                if (v != INT64_MAX) {
                    cur = T.v2s[v];
                    break;
                }
            }
            std::int64_t rest = r;
            double centre[D];
            for (int k = 1; k < D; ++k) {
                const std::int64_t i = rest % L.g[k];
                rest /= L.g[k];
                const double h = 0.5 * L.lo[k] + (static_cast<double>(i) + 0.5) / static_cast<double>(L.g[k]) *
                                                     (0.5 * L.hi[k] - 0.5 * L.lo[k]);
                centre[k] = std::clamp(2.0 * h, L.lo[k], L.hi[k]);
            }
            for (std::int64_t j = 0; j < row; ++j) {
                const std::int64_t v = first[static_cast<std::size_t>(r * row + j)];
                if (v != INT64_MAX) cur = T.v2s[v];
                const double h = 0.5 * L.lo[0] + (static_cast<double>(j) + 0.5) / static_cast<double>(row) *
                                                     (0.5 * L.hi[0] - 0.5 * L.lo[0]);
                centre[0] = std::clamp(2.0 * h, L.lo[0], L.hi[0]);
                std::int64_t last = cur;
                const std::int64_t got = detail::walk_any<D>(T, cur, centre, &last);
                cur = got >= 0 ? got : last;
                first[static_cast<std::size_t>(r * row + j)] = cur;
            }
        }
    });
}

// find_simplex (SPT103): out[i] the smallest simplex whose closed simplex holds x[i], -1 outside the hull
// or for a non-finite point. Blocks of kQueryBlock points per task: every answer is the definition's, so
// the order the points are taken in changes nothing but the time. On one thread, from kQueryCellSort points,
// they are taken cell by cell of the location grid (a counting sort; SPT188): the points of one cell walk over
// the same simplices while those are in cache (10^5 3-D points: 29 ms -> 18 ms). On more threads the
// counting's atomics cost more than the cache gains, and the points go in their order.
//
// locate_each calls done(i, t) once per point with that simplex (-1 outside or non-finite), in the order above,
// from the task that located it: the scattered-data interpolants of sd.interpolate (interpolate/scattered.cpp)
// evaluate each point there, while its simplex is in cache.
template <int D, class Done>
inline void locate_each(const Triangulation& T, const double* x, std::int64_t q, bool split, Done&& done) {
    const Locator<D>& L = detail::locator<D>(T);
    auto one = [&](std::int64_t i) {
        const double* p = x + i * D;
        bool finite = true;
        for (int k = 0; k < D; ++k) finite = finite && std::isfinite(p[k]);
        if (!finite) {
            done(i, std::int64_t{-1});
            return;
        }
        const std::int64_t t0 = L.start[static_cast<std::size_t>(detail::cell_of<D>(L, p))];
        if (detail::safe_with<D>(T, p)) {
            const std::int64_t t = detail::walk<D, true>(T, t0, p, nullptr);
            done(i, t < 0 ? std::int64_t{-1} : detail::smallest_holding<D, true>(T, t, p));
        } else {
            const std::int64_t t = detail::walk<D, false>(T, t0, p, nullptr);
            done(i, t < 0 ? std::int64_t{-1} : detail::smallest_holding<D, false>(T, t, p));
        }
    };
    const std::int64_t nb = chunks_of(q, kQueryBlock);
    if (q < kQueryCellSort || (split && parallel::num_threads() > 1)) {  // threads: the counts' atomics cost more
        run_tasks(nb, split, [&](std::int64_t c) {
            for (std::int64_t i = c * kQueryBlock, e = std::min(q, (c + 1) * kQueryBlock); i < e; ++i) one(i);
        });
        return;
    }
    const auto cells = static_cast<std::int64_t>(L.start.size());
    std::vector<std::int64_t> start(static_cast<std::size_t>(cells) + 1, 0), cell(static_cast<std::size_t>(q));
    run_tasks(nb, split, [&](std::int64_t c) {
        for (std::int64_t i = c * kQueryBlock, e = std::min(q, (c + 1) * kQueryBlock); i < e; ++i) {
            const std::int64_t k = detail::cell_of<D>(L, x + i * D);  // any cell for a non-finite point
            cell[static_cast<std::size_t>(i)] = k;
            fetch_add(start[static_cast<std::size_t>(k) + 1], 1);
        }
    });
    for (std::int64_t k = 0; k < cells; ++k)
        start[static_cast<std::size_t>(k) + 1] += start[static_cast<std::size_t>(k)];
    std::vector<std::int64_t> order(static_cast<std::size_t>(q));
    run_tasks(nb, split, [&](std::int64_t c) {
        for (std::int64_t i = c * kQueryBlock, e = std::min(q, (c + 1) * kQueryBlock); i < e; ++i)
            order[static_cast<std::size_t>(
                fetch_add(start[static_cast<std::size_t>(cell[static_cast<std::size_t>(i)])], 1)
            )] = i;
    });
    run_tasks(nb, split, [&](std::int64_t c) {
        for (std::int64_t j = c * kQueryBlock, e = std::min(q, (c + 1) * kQueryBlock); j < e; ++j)
            one(order[static_cast<std::size_t>(j)]);
    });
}

template <int D>
inline void find_simplex(const Triangulation& T, const double* x, std::int64_t q, std::int64_t* out, bool split) {
    locate_each<D>(T, x, q, split, [out](std::int64_t i, std::int64_t t) { out[i] = t; });
}

// ---------------------------------------------------------------- the attributes made on first use

// The boundary facets (f, d) of §2.2: outward (d = 2: the interior on the left; d = 3: counterclockwise
// seen from outside), the smallest index first for d = 3 (d = 2 keeps the direction's order), rows sorted.
// For a positive simplex the facet opposite k with its vertices in position order faces outward when k is
// even and inward when k is odd (one swap away).
inline std::vector<std::int64_t> convex_hull(const Triangulation& T) {
    const int K = T.d + 1, d = T.d;
    std::vector<std::int64_t> rows;
    for (std::int64_t q = 0; q < T.t; ++q)
        for (int k = 0; k < K; ++k) {
            if (T.neighbors[q * K + k] >= 0) continue;
            std::int64_t f[3];
            int m = 0;
            for (int j = 0; j < K; ++j)
                if (j != k) f[m++] = T.simplices[q * K + j];
            if (k & 1) std::swap(f[d - 2], f[d - 1]);
            if (d == 3) {
                while (f[0] > f[1] || f[0] > f[2]) std::rotate(f, f + 1, f + 3);
            }
            rows.insert(rows.end(), f, f + d);
        }
    const std::int64_t nf = static_cast<std::int64_t>(rows.size()) / d;
    std::vector<std::int64_t> idx(static_cast<std::size_t>(nf));
    for (std::int64_t i = 0; i < nf; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::sort(idx.begin(), idx.end(), [&](std::int64_t a, std::int64_t b) {
        return std::lexicographical_compare(
            rows.begin() + a * d, rows.begin() + (a + 1) * d, rows.begin() + b * d,
            rows.begin() + (b + 1) * d
        );  // fmt: skip
    });
    std::vector<std::int64_t> out(rows.size());
    for (std::int64_t i = 0; i < nf; ++i)
        std::copy(
            rows.begin() + idx[static_cast<std::size_t>(i)] * d,
            rows.begin() + (idx[static_cast<std::size_t>(i)] + 1) * d,
            out.begin() + i * d
        );  // fmt: skip
    return out;
}

// The rows (point, simplex, vertex) of the points left out of the triangulation: the equal points that
// are not their representative (SPT96), in increasing order.
inline std::vector<std::int64_t> coplanar_rows(const Triangulation& T) {
    std::vector<std::int64_t> out;
    for (std::int64_t i = 0; i < T.n; ++i) {
        const std::int64_t r = T.rep[static_cast<std::size_t>(i)];
        if (r != i) out.insert(out.end(), {i, T.v2s[r], r});
    }
    return out;
}

// vertex_neighbor_vertices for d = 2: (indptr (n + 1), indices), each row increasing. Every directed edge
// (a, b) of a counterclockwise triangle puts b in a's row; a boundary edge also puts a in b's row (its
// other direction has no triangle). Rows are filled in any order and then sorted, so the arrays do not
// depend on the thread count.
inline void vertex_neighbors2(
    const Triangulation& T,
    std::int64_t* indptr,
    std::vector<std::int64_t>& indices,
    bool split
) {  // fmt: skip
    const std::int64_t n = T.n, t = T.t;
    std::fill(indptr, indptr + n + 1, 0);
    const std::int64_t nc = chunks_of(t, kChunk);
    auto each = [&](auto&& put) {
        run_tasks(nc, split, [&](std::int64_t c) {
            for (std::int64_t q = c * kChunk, e = std::min(t, (c + 1) * kChunk); q < e; ++q) {
                const std::int64_t* s = T.simplices + q * 3;
                for (int k = 0; k < 3; ++k) {
                    const std::int64_t a = s[(k + 1) % 3], b = s[(k + 2) % 3];  // the edge opposite k
                    put(a, b);
                    if (T.neighbors[q * 3 + k] < 0) put(b, a);
                }
            }
        });
    };
    each([&](std::int64_t a, std::int64_t) { fetch_add(indptr[a + 1], 1); });
    for (std::int64_t i = 0; i < n; ++i) indptr[i + 1] += indptr[i];
    indices.assign(static_cast<std::size_t>(indptr[n]), 0);
    std::vector<std::int64_t> cursor(indptr, indptr + n);
    each([&](std::int64_t a, std::int64_t b) {
        indices[static_cast<std::size_t>(fetch_add(cursor[static_cast<std::size_t>(a)], 1))] = b;
    });
    run_tasks(chunks_of(n, kChunk), split, [&](std::int64_t c) {
        for (std::int64_t i = c * kChunk, e = std::min(n, (c + 1) * kChunk); i < e; ++i)
            std::sort(indices.begin() + indptr[i], indices.begin() + indptr[i + 1]);
    });
}

// transform for d = 2 (SPT106): out (t, 3, 2) with out[q, :2, :] = T^-1 = adj(T) / D and out[q, 2, :] the
// vertex 2, T's columns the vertices 0 and 1 minus vertex 2. The adjugate's entries are differences
// (relative error u), D is det2 (1.002 u), one division: 3.01 u per entry (§4: 4 u).
inline void transform2(const Triangulation& T, double* out, bool split) {
    run_tasks(chunks_of(T.t, kChunk), split, [&](std::int64_t c) {
        for (std::int64_t q = c * kChunk, e = std::min(T.t, (c + 1) * kChunk); q < e; ++q) {
            const std::int64_t* s = T.simplices + q * 3;
            const double *a = T.pts + s[0] * 2, *b = T.pts + s[1] * 2, *r = T.pts + s[2] * 2;
            const double D = T.range.safe(2) ? det2(a, b, r) : det2_wide(a, b, r);
            double* o = out + q * 6;
            o[0] = (b[1] - r[1]) / D;
            o[1] = -(b[0] - r[0]) / D;
            o[2] = -(a[1] - r[1]) / D;
            o[3] = (a[0] - r[0]) / D;
            o[4] = r[0];
            o[5] = r[1];
        }
    });
}

// vertex_neighbor_vertices for d = 3: (indptr (n + 1), indices), each row increasing. The star of every
// vertex (its tetrahedra, a counting sort in any order) gives its neighbours: the other vertices of the
// star, each kept once by a small open-addressing set (a star of k tetrahedra names about 3 k / 5 distinct
// vertices five times each; SPT189), then sorted, by tasks of kChunk vertices whose rows are copied in chunk order.
// So the arrays do not depend on the thread count. Memory: the stars (4 t int32) and the rows.
inline void vertex_neighbors3(
    const Triangulation& T,
    std::int64_t* indptr,
    std::vector<std::int64_t>& indices,
    bool split
) {  // fmt: skip
    const std::int64_t n = T.n, t = T.t;
    std::vector<std::int64_t> start(static_cast<std::size_t>(n) + 1, 0);
    const std::int64_t tc = chunks_of(t, kChunk);
    run_tasks(tc, split, [&](std::int64_t c) {
        for (std::int64_t q = c * kChunk, e = std::min(t, (c + 1) * kChunk); q < e; ++q)
            for (int j = 0; j < 4; ++j) fetch_add(start[static_cast<std::size_t>(T.simplices[q * 4 + j]) + 1], 1);
    });
    for (std::int64_t i = 0; i < n; ++i) start[static_cast<std::size_t>(i) + 1] += start[static_cast<std::size_t>(i)];
    std::vector<std::int32_t> star(static_cast<std::size_t>(4 * t));
    {
        std::vector<std::int64_t> cursor(start.begin(), start.end() - 1);
        run_tasks(tc, split, [&](std::int64_t c) {
            for (std::int64_t q = c * kChunk, e = std::min(t, (c + 1) * kChunk); q < e; ++q)
                for (int j = 0; j < 4; ++j)
                    star[static_cast<std::size_t>(
                        fetch_add(cursor[static_cast<std::size_t>(T.simplices[q * 4 + j])], 1)
                    )] = static_cast<std::int32_t>(q);  // fmt: skip
        });
    }
    constexpr std::int64_t kRows = 1 << 10;  // vertices per task: about 30 tetrahedra each (SPT189)
    const std::int64_t vc = chunks_of(n, kRows);
    std::vector<std::vector<std::int64_t>> part(static_cast<std::size_t>(vc));
    run_tasks(vc, split, [&](std::int64_t c) {
        std::vector<std::int64_t>& rows = part[static_cast<std::size_t>(c)];
        std::vector<std::int64_t> buf, set(64, -1);
        int bits = 6;
        for (std::int64_t v = c * kRows, e = std::min(n, (c + 1) * kRows); v < e; ++v) {
            const std::int64_t lo = start[static_cast<std::size_t>(v)], hi = start[static_cast<std::size_t>(v) + 1];
            while ((std::int64_t{1} << bits) < 8 * (hi - lo)) ++bits;  // load at most 3/8
            if (set.size() < (std::size_t{1} << bits)) set.assign(std::size_t{1} << bits, -1);
            const std::uint64_t mask = (std::uint64_t{1} << bits) - 1;
            buf.clear();
            for (std::int64_t k = lo; k < hi; ++k) {
                const std::int64_t* s = T.simplices + static_cast<std::int64_t>(star[static_cast<std::size_t>(k)]) * 4;
                for (int j = 0; j < 4; ++j) {
                    const std::int64_t u = s[j];
                    if (u == v) continue;
                    std::uint64_t h = (static_cast<std::uint64_t>(u) * 0x9e3779b97f4a7c15ULL) >> (64 - bits);
                    while (set[h] >= 0 && set[h] != u) h = (h + 1) & mask;
                    if (set[h] < 0) {
                        set[h] = u;
                        buf.push_back(u);
                    }
                }
            }
            for (const std::int64_t u : buf) {  // empty the set: each key is ahead of its home slot
                std::uint64_t h = (static_cast<std::uint64_t>(u) * 0x9e3779b97f4a7c15ULL) >> (64 - bits);
                while (set[h] != u) h = (h + 1) & mask;
                set[h] = -1;
            }
            std::sort(buf.begin(), buf.end());
            indptr[v + 1] = static_cast<std::int64_t>(buf.size());
            rows.insert(rows.end(), buf.begin(), buf.end());
        }
    });
    indptr[0] = 0;
    for (std::int64_t i = 0; i < n; ++i) indptr[i + 1] += indptr[i];
    indices.assign(static_cast<std::size_t>(indptr[n]), 0);
    run_tasks(vc, split, [&](std::int64_t c) {
        const std::vector<std::int64_t>& rows = part[static_cast<std::size_t>(c)];
        std::copy(rows.begin(), rows.end(), indices.begin() + indptr[c * kRows]);
    });
}

// transform for d = 3 (SPT106): out (t, 4, 3) with out[q, :3, :] = T^-1 and out[q, 3, :] the vertex 3 (r),
// T's columns e_k = vertex k - r. Row i of T^-1 is (e_(i+1) x e_(i+2)) / D with D = det T = det3(r, v0, v1,
// v2); a component of a cross product is det2 of two coordinates of two vertices against r (1.002 u), D is
// det3 (1.002 u), one division: 3.01 u per entry (§4: 4 u).
inline void transform3(const Triangulation& T, double* out, bool split) {
    const bool s2 = T.range.safe(2), s3 = T.range.safe(3);
    run_tasks(chunks_of(T.t, kChunk), split, [&](std::int64_t c) {
        for (std::int64_t q = c * kChunk, e = std::min(T.t, (c + 1) * kChunk); q < e; ++q) {
            const std::int64_t* s = T.simplices + q * 4;
            const double* v[3] = {T.pts + s[0] * 3, T.pts + s[1] * 3, T.pts + s[2] * 3};
            const double* r = T.pts + s[3] * 3;
            const double D = s3 ? det3(r, v[0], v[1], v[2]) : det3_wide(r, v[0], v[1], v[2]);
            double* o = out + q * 12;
            for (int i = 0; i < 3; ++i) {
                const double *a = v[(i + 1) % 3], *b = v[(i + 2) % 3];
                for (int k = 0; k < 3; ++k) {
                    const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
                    const double pa[2] = {a[k1], a[k2]}, pb[2] = {b[k1], b[k2]}, pr[2] = {r[k1], r[k2]};
                    o[3 * i + k] = (s2 ? det2(pa, pb, pr) : det2_wide(pa, pb, pr)) / D;
                }
            }
            o[9] = r[0];
            o[10] = r[1];
            o[11] = r[2];
        }
    });
}

}  // namespace sd::spatial::tri
