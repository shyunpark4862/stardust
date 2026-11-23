// sd.spatial's Delaunay for d = 2 (docs/design/spatial_geometry.md §3.3, §3.5, §3.10, §3.11; milestone SPT9,
// decisions SPT104, SPT124-SPT139) and the native class _Delaunay for d = 2 and 3 (the 3-D build is
// delaunay3.cpp, milestone SPT10; the Python class is src/stardust/spatial/_delaunay.py, which checks every
// argument).
//
//   _Delaunay(points, max_simplices=-1)
//                              points (n, d) float64 of any strides, finite (checked by the caller), d = 2 or 3.
//                              ValueError for fewer than d + 1 distinct points or points on one line (d = 2) or
//                              plane (d = 3) (SPT97, exact). d = 3: MemoryError when the tetrahedra (ghosts
//                              included) pass max_simplices (-1: what physical memory holds; tests pass less).
//   t.n, t.ndim, t.nsimplex, t.ndistinct
//   t.points(), t.simplices(), t.neighbors(), t.vertex_to_simplex()   the arrays (the same objects each call)
//   t.convex_hull(), t.vertex_neighbor_vertices(), t.coplanar(), t.transform()   new arrays (made on demand)
//   t.find_simplex(x)          x (q, d) C-contiguous float64 -> (q,) int64
//   t._grid()                  the location grid's cells per axis (tests)
//
// THE BUILD (§3.3). The distinct points (sorted lexicographically, equal points merged into the smallest
// index, SPT96) are triangulated by Guibas and Stolfi's divide and conquer with Dwyer's alternating cuts:
// a node of the recursion splits its points at the median of its axis's order (x then y for axis 0; y then
// -x for axis 1, which is the order of the points turned a quarter clockwise), the children triangulate
// their halves and the node merges them. Turning the plane by a quarter keeps every orientation and every
// circle test, so the merge in axis 1 is the merge of the turned points in axis 0, and Guibas and Stolfi's
// argument (which needs the halves separated in lexicographic order, collinear points included) holds
// unchanged. A child returns its hull edges at the extreme points of its own axis; the parent walks along
// the child's hull to the extremes of its axis (the order along a convex boundary rises and falls once).
//
// The edges are a quad-edge structure without the dual (SPT126): half-edge e and its twin e ^ 1, each with
// its origin and its successor and predecessor around the origin (onext, oprev). Splice, connect and delete
// are Guibas and Stolfi's operations on the origin rings.
//
// The tie rule is the predicates' (§3.2): incircle_sos never answers 0 for the merge's tests, whose first
// three points turn counterclockwise, so the result is the one triangulation of the perturbed points,
// whatever the order of the merges. One change to the published merge: a candidate is deleted only when
// the next candidate is valid too (SPT127). Without it the next candidate may be the base edge's twin (its
// destination the base edge's origin), whose repeated point the tie rule would call "inside".
//
// PARALLEL (§3.10). The recursion's nodes are fixed by the sizes alone (the median position). Nodes of more
// than kLeaf points form the top of the tree: their partitions run level by level (a level's nodes side by
// side), the subtrees of at most kLeaf points run as tasks, and the merges run level by level upwards. Each
// node owns the edge slots 3 lo .. 3 hi of its point positions lo .. hi (a planar graph on V vertices has at
// most 3V - 6 edges, and the merge keeps the graph planar), so the tasks touch disjoint memory. The edge
// numbers and the canonical arrays do not depend on the thread count.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/delaunay3.hpp"
#include "spatial/predicates.hpp"
#include "spatial/sfc.hpp"
#include "spatial/spatial.hpp"
#include "spatial/triangulation.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

using tri::Pt;
using tri::Triangulation;

constexpr std::int64_t kLeaf = 1 << 14;  // a subtree of at most this many points is one task (SPT128)

// ---------------------------------------------------------------- the edge structure

struct FreeList {
    std::int32_t head = -1, tail = -1;  // edge numbers (half-edges 2u, 2u + 1), linked through onext[2u]
};

struct Hull {
    std::int32_t le = -1;  // the counterclockwise hull edge out of the node axis's smallest point
    std::int32_t re = -1;  // the clockwise hull edge out of its largest point
    FreeList free;
};

// Axis 0: (x, y); axis 1: (y, -x), the lexicographic order of the points turned a quarter clockwise.
inline bool less_on(const Pt<2>& a, const Pt<2>& b, int axis) {
    if (axis == 0) return a.c[0] < b.c[0] || (a.c[0] == b.c[0] && a.c[1] < b.c[1]);
    return a.c[1] < b.c[1] || (a.c[1] == b.c[1] && a.c[0] > b.c[0]);
}

struct Builder {
    Pt<2>* P;  // the distinct points, partitioned in place; a vertex is a position in P
    std::int32_t* org;
    std::int32_t* onext;
    std::int32_t* oprev;
    Predicates pred;

    // ---- the operations

    std::int32_t dest(std::int32_t e) const { return org[e ^ 1]; }
    std::int32_t lnext(std::int32_t e) const { return oprev[e ^ 1]; }
    std::int32_t rprev(std::int32_t e) const { return onext[e ^ 1]; }

    bool ccw(std::int32_t a, std::int32_t b, std::int32_t c) const { return pred.orient2d(P[a].c, P[b].c, P[c].c) > 0; }
    bool leftof(std::int32_t x, std::int32_t e) const { return ccw(x, org[e], dest(e)); }
    bool rightof(std::int32_t x, std::int32_t e) const { return ccw(x, dest(e), org[e]); }
    // d strictly inside the perturbed circle through a, b, c (counterclockwise).
    bool incircle(std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t d) const {
        return pred.incircle_sos(P[a].c, P[b].c, P[c].c, P[d].c) > 0;
    }

    std::int32_t make_edge(FreeList& fl, std::int32_t a, std::int32_t b) {
        const std::int32_t u = fl.head;
        fl.head = onext[2 * u];
        if (fl.head < 0) fl.tail = -1;
        const std::int32_t e = 2 * u;
        org[e] = a;
        org[e + 1] = b;
        onext[e] = oprev[e] = e;
        onext[e + 1] = oprev[e + 1] = e + 1;
        return e;
    }

    void splice(std::int32_t a, std::int32_t b) {
        const std::int32_t an = onext[a], bn = onext[b];
        onext[a] = bn;
        onext[b] = an;
        oprev[bn] = a;
        oprev[an] = b;
    }

    std::int32_t connect(FreeList& fl, std::int32_t a, std::int32_t b) {
        const std::int32_t e = make_edge(fl, dest(a), org[b]);
        splice(e, lnext(a));
        splice(e ^ 1, b);
        return e;
    }

    void kill(FreeList& fl, std::int32_t e) {
        splice(e, oprev[e]);
        splice(e ^ 1, oprev[e ^ 1]);
        const std::int32_t u = e >> 1;
        org[2 * u] = org[2 * u + 1] = -1;
        onext[2 * u] = fl.head;
        if (fl.head < 0) fl.tail = u;
        fl.head = u;
    }

    static FreeList join(FreeList a, const FreeList& b, std::int32_t* onext) {
        if (a.head < 0) return b;
        if (b.head >= 0) {
            onext[2 * a.tail] = b.head;
            a.tail = b.tail;
        }
        return a;
    }

    // ---- the recursion

    // Two or three points at positions lo..hi, sorted in the node's axis order (Guibas and Stolfi).
    Hull base(std::int64_t lo, std::int64_t hi, int axis) {
        std::sort(P + lo, P + hi, [axis](const Pt<2>& a, const Pt<2>& b) { return less_on(a, b, axis); });
        // The node's edge slots: the first ones for the edges, the rest free.
        FreeList fl;
        const auto first = static_cast<std::int32_t>(3 * lo), end = static_cast<std::int32_t>(3 * hi);
        for (std::int32_t u = first; u < end; ++u) {
            org[2 * u] = org[2 * u + 1] = -1;
            onext[2 * u] = u + 1 < end ? u + 1 : -1;
        }
        fl.head = first;
        fl.tail = end - 1;
        const auto s1 = static_cast<std::int32_t>(lo);
        Hull h;
        if (hi - lo == 2) {
            const std::int32_t a = make_edge(fl, s1, s1 + 1);
            h.le = a;
            h.re = a ^ 1;
        } else {
            const std::int32_t a = make_edge(fl, s1, s1 + 1), b = make_edge(fl, s1 + 1, s1 + 2);
            splice(a ^ 1, b);
            const int o = pred.orient2d(P[s1].c, P[s1 + 1].c, P[s1 + 2].c);
            if (o > 0) {
                connect(fl, b, a);
                h.le = a;
                h.re = b ^ 1;
            } else if (o < 0) {
                const std::int32_t c = connect(fl, b, a);
                h.le = c ^ 1;
                h.re = c;
            } else {
                h.le = a;
                h.re = b ^ 1;
            }
        }
        h.free = fl;
        return h;
    }

    // The counterclockwise hull edge out of the smallest point in `axis` order, from any counterclockwise
    // hull edge c: along the hull towards the smaller neighbour (the order rises and falls once).
    std::int32_t lowest(std::int32_t c, int axis) const {
        if (less_on(P[dest(c)], P[org[c]], axis)) {
            while (less_on(P[dest(c)], P[org[c]], axis)) c = rprev(c);
            return c;
        }
        std::int32_t w = oprev[c];  // the clockwise hull edge out of the same point
        if (!less_on(P[dest(w)], P[org[w]], axis)) return c;
        while (less_on(P[dest(w)], P[org[w]], axis)) w = lnext(w);
        return onext[w];
    }

    // The clockwise hull edge out of the largest point in `axis` order, from any clockwise hull edge w.
    std::int32_t highest(std::int32_t w, int axis) const {
        if (less_on(P[org[w]], P[dest(w)], axis)) {
            while (less_on(P[org[w]], P[dest(w)], axis)) w = lnext(w);
            return w;
        }
        std::int32_t c = onext[w];  // the counterclockwise hull edge out of the same point
        if (!less_on(P[org[c]], P[dest(c)], axis)) return w;
        while (less_on(P[org[c]], P[dest(c)], axis)) c = rprev(c);
        return oprev[c];
    }

    // Merges the triangulations of the left (smaller in `axis` order) and right halves.
    Hull merge(const Hull& L, const Hull& R, int axis) {
        FreeList fl = join(L.free, R.free, onext);
        std::int32_t ldo = lowest(L.le, axis), ldi = highest(L.re, axis);
        std::int32_t rdi = lowest(R.le, axis), rdo = highest(R.re, axis);
        for (;;) {  // the lower common tangent
            if (leftof(org[rdi], ldi)) ldi = lnext(ldi);
            else if (rightof(org[ldi], rdi)) rdi = rprev(rdi);
            else break;
        }
        std::int32_t basel = connect(fl, rdi ^ 1, ldi);
        if (org[ldi] == org[ldo]) ldo = basel ^ 1;
        if (org[rdi] == org[rdo]) rdo = basel;
        auto valid = [&](std::int32_t e) { return ccw(dest(e), dest(basel), org[basel]); };
        for (;;) {
            std::int32_t lcand = onext[basel ^ 1];
            if (valid(lcand)) {
                for (;;) {
                    const std::int32_t nx = onext[lcand];
                    if (!valid(nx) || !incircle(dest(basel), org[basel], dest(lcand), dest(nx))) break;
                    kill(fl, lcand);
                    lcand = nx;
                }
            }
            std::int32_t rcand = oprev[basel];
            if (valid(rcand)) {
                for (;;) {
                    const std::int32_t nx = oprev[rcand];
                    if (!valid(nx) || !incircle(dest(basel), org[basel], dest(rcand), dest(nx))) break;
                    kill(fl, rcand);
                    rcand = nx;
                }
            }
            const bool lv = valid(lcand), rv = valid(rcand);
            if (!lv && !rv) break;
            if (!lv || (rv && incircle(dest(lcand), org[lcand], org[rcand], dest(rcand))))
                basel = connect(fl, rcand, basel ^ 1);
            else basel = connect(fl, basel ^ 1, lcand ^ 1);
        }
        Hull h;
        h.le = ldo;
        h.re = rdo;
        h.free = fl;
        return h;
    }

    // Splits positions lo..hi at the median of the node's axis order (depth 0 is sorted already).
    void partition(std::int64_t lo, std::int64_t hi, int depth) {
        if (depth == 0) return;
        const int axis = depth & 1;
        std::nth_element(P + lo, P + lo + (hi - lo) / 2, P + hi, [axis](const Pt<2>& a, const Pt<2>& b) {
            return less_on(a, b, axis);
        });  // fmt: skip
    }

    Hull build(std::int64_t lo, std::int64_t hi, int depth) {
        const int axis = depth & 1;
        if (hi - lo <= 3) return base(lo, hi, axis);
        partition(lo, hi, depth);
        const std::int64_t mid = lo + (hi - lo) / 2;
        const Hull L = build(lo, mid, depth + 1);
        const Hull R = build(mid, hi, depth + 1);
        return merge(L, R, axis);
    }
};

// ---------------------------------------------------------------- the whole build

struct Node {
    Node(std::int64_t lo_, std::int64_t hi_, int depth_) : lo(lo_), hi(hi_), depth(depth_) {}
    std::int64_t lo, hi;
    int depth;
    std::int32_t left = -1, right = -1;  // children (nodes of the top), -1 for a subtree task
    Hull hull;
};

enum class Status { Ok, TooFew, Collinear, TooMany };

struct Internal {
    std::int64_t m = 0, t = 0;
    std::vector<std::int32_t> s, nb;  // (t, 3) triangles of input indices, counterclockwise; neighbours
};

// Triangulates the representatives; fills T.rep, T.range, T.pred, T.m and the internal triangles.
Status build_internal(Triangulation& T, Internal& out, bool split) {
    const std::int64_t n = T.n;
    const double* pts = T.pts;
    const std::vector<std::int64_t> order = tri::lex_order_parallel(pts, n, 2, split);
    const std::vector<std::int64_t> distinct = representatives(pts, 2, order, T.rep);
    const auto m = static_cast<std::int64_t>(distinct.size());
    T.m = out.m = m;
    if (m < 3) return Status::TooFew;
    if (m > (INT32_MAX - 8) / 6) return Status::TooMany;
    T.range = tri::range_of(pts, 2 * n, split);
    T.pred = Predicates(T.range);
    // Every point on the line through the lexicographically first and last points: collinear.
    {
        const double *a = pts + 2 * distinct.front(), *b = pts + 2 * distinct.back();
        bool flat = true;
        for (std::int64_t r = 1; r + 1 < m && flat; ++r)
            flat = T.pred.orient2d(a, b, pts + 2 * distinct[static_cast<std::size_t>(r)]) == 0;
        if (flat) return Status::Collinear;
    }
    std::vector<Pt<2>> P(static_cast<std::size_t>(m));
    for (std::int64_t r = 0; r < m; ++r) {
        const std::int64_t i = distinct[static_cast<std::size_t>(r)];
        P[static_cast<std::size_t>(r)] = {{pts[2 * i], pts[2 * i + 1]}, i};
    }
    const std::int64_t halves = 6 * m;  // 3 m edge slots
    std::vector<std::int32_t> org(static_cast<std::size_t>(halves)), onext(static_cast<std::size_t>(halves)),
        oprev(static_cast<std::size_t>(halves));
    Builder B{P.data(), org.data(), onext.data(), oprev.data(), T.pred};

    // The top of the tree (nodes of more than kLeaf points), level by level.
    std::vector<Node> nodes{Node(0, m, 0)};
    std::vector<std::vector<std::int32_t>> levels{{0}};
    std::vector<std::int32_t> tasks;  // subtrees of at most kLeaf points
    for (std::size_t l = 0; l < levels.size(); ++l) {
        std::vector<std::int32_t> next;
        for (const std::int32_t k : levels[l]) {
            Node& nd = nodes[static_cast<std::size_t>(k)];
            if (nd.hi - nd.lo <= kLeaf) {
                tasks.push_back(k);
                continue;
            }
            const std::int64_t mid = nd.lo + (nd.hi - nd.lo) / 2;
            const int depth = nd.depth;
            const std::int64_t lo = nd.lo, hi = nd.hi;
            const auto a = static_cast<std::int32_t>(nodes.size());
            nodes.emplace_back(lo, mid, depth + 1);
            nodes.emplace_back(mid, hi, depth + 1);
            nodes[static_cast<std::size_t>(k)].left = a;
            nodes[static_cast<std::size_t>(k)].right = a + 1;
            next.push_back(a);
            next.push_back(a + 1);
        }
        if (!next.empty()) levels.push_back(std::move(next));
    }
    auto internal_of = [&](const std::vector<std::int32_t>& level) {
        std::vector<std::int32_t> out;
        for (const std::int32_t k : level)
            if (nodes[static_cast<std::size_t>(k)].left >= 0) out.push_back(k);
        return out;
    };
    for (const auto& level : levels) {  // partitions, top down
        const std::vector<std::int32_t> inner = internal_of(level);
        tri::run_tasks(static_cast<std::int64_t>(inner.size()), split, [&](std::int64_t j) {
            const Node& nd = nodes[static_cast<std::size_t>(inner[static_cast<std::size_t>(j)])];
            B.partition(nd.lo, nd.hi, nd.depth);
        });
    }
    tri::run_tasks(static_cast<std::int64_t>(tasks.size()), split, [&](std::int64_t j) {
        Node& nd = nodes[static_cast<std::size_t>(tasks[static_cast<std::size_t>(j)])];
        Builder b = B;
        nd.hull = b.build(nd.lo, nd.hi, nd.depth);
    });
    for (std::size_t l = levels.size(); l-- > 0;) {  // merges, bottom up
        const std::vector<std::int32_t> inner = internal_of(levels[l]);
        tri::run_tasks(static_cast<std::int64_t>(inner.size()), split, [&](std::int64_t j) {
            Node& nd = nodes[static_cast<std::size_t>(inner[static_cast<std::size_t>(j)])];
            Builder b = B;
            nd.hull = b.merge(
                nodes[static_cast<std::size_t>(nd.left)].hull, nodes[static_cast<std::size_t>(nd.right)].hull,
                nd.depth & 1
            );  // fmt: skip
        });
    }
    const std::int32_t le = nodes[0].hull.le;

    // The outer face: left of the twin of the counterclockwise hull edge, walked by lnext.
    std::vector<std::uint8_t> outer(static_cast<std::size_t>(halves), 0);
    {
        const std::int32_t e0 = le ^ 1;
        std::int32_t e = e0;
        do {
            outer[static_cast<std::size_t>(e)] = 1;
            e = B.lnext(e);
        } while (e != e0);
    }
    // Every inner face is a triangle; its smallest half-edge number names it. Tasks over half-edge chunks
    // count their faces, then number them in chunk order.
    const std::int64_t nc = tri::chunks_of(halves, tri::kChunk);
    std::vector<std::int64_t> count(static_cast<std::size_t>(nc) + 1, 0);
    auto names_face = [&](std::int32_t e) {
        if (org[static_cast<std::size_t>(e)] < 0 || outer[static_cast<std::size_t>(e)]) return false;
        const std::int32_t f = B.lnext(e), g = B.lnext(f);
        return e < f && e < g;
    };
    tri::run_tasks(nc, split, [&](std::int64_t c) {
        std::int64_t k = 0;
        for (std::int64_t e = c * tri::kChunk, end = std::min(halves, (c + 1) * tri::kChunk); e < end; ++e)
            k += names_face(static_cast<std::int32_t>(e));
        count[static_cast<std::size_t>(c) + 1] = k;
    });
    for (std::int64_t c = 0; c < nc; ++c) count[static_cast<std::size_t>(c) + 1] += count[static_cast<std::size_t>(c)];
    const std::int64_t t = count[static_cast<std::size_t>(nc)];
    out.t = t;
    out.s.assign(static_cast<std::size_t>(3 * t), 0);
    out.nb.assign(static_cast<std::size_t>(3 * t), -1);
    std::vector<std::int32_t> face(static_cast<std::size_t>(halves), -1);  // the triangle left of each half-edge
    std::vector<std::int32_t> first(static_cast<std::size_t>(t));          // its half-edge (vertex 0 -> vertex 1)
    tri::run_tasks(nc, split, [&](std::int64_t c) {
        std::int64_t k = count[static_cast<std::size_t>(c)];
        for (std::int64_t e = c * tri::kChunk, end = std::min(halves, (c + 1) * tri::kChunk); e < end; ++e) {
            const auto e0 = static_cast<std::int32_t>(e);
            if (!names_face(e0)) continue;
            const std::int32_t e1 = B.lnext(e0), e2 = B.lnext(e1);
            const auto q = static_cast<std::int32_t>(k++);
            face[static_cast<std::size_t>(e0)] = face[static_cast<std::size_t>(e1)] =
                face[static_cast<std::size_t>(e2)] = q;
            first[static_cast<std::size_t>(q)] = e0;
            out.s[static_cast<std::size_t>(3 * q)] = static_cast<std::int32_t>(P[static_cast<std::size_t>(org[e0])].id);
            out.s[static_cast<std::size_t>(3 * q + 1)] =
                static_cast<std::int32_t>(P[static_cast<std::size_t>(org[e1])].id);
            out.s[static_cast<std::size_t>(3 * q + 2)] =
                static_cast<std::int32_t>(P[static_cast<std::size_t>(org[e2])].id);
        }
    });
    // Across the edge opposite vertex k: e1 (1 -> 2) is opposite 0, e2 (2 -> 0) opposite 1, e0 opposite 2.
    tri::run_tasks(tri::chunks_of(t, tri::kChunk), split, [&](std::int64_t c) {
        for (std::int64_t q = c * tri::kChunk, end = std::min(t, (c + 1) * tri::kChunk); q < end; ++q) {
            const std::int32_t e0 = first[static_cast<std::size_t>(q)], e1 = B.lnext(e0), e2 = B.lnext(e1);
            out.nb[static_cast<std::size_t>(3 * q)] = face[static_cast<std::size_t>(e1 ^ 1)];
            out.nb[static_cast<std::size_t>(3 * q + 1)] = face[static_cast<std::size_t>(e2 ^ 1)];
            out.nb[static_cast<std::size_t>(3 * q + 2)] = face[static_cast<std::size_t>(e0 ^ 1)];
        }
    });
    return Status::Ok;
}

// ---------------------------------------------------------------- the native class

nb::object array_of(DType t, std::initializer_list<std::int64_t> shape) {
    const std::vector<std::int64_t> s(shape);
    return wrap(new_array(t, s));
}

template <class T>
T* data_of(const nb::object& a) {
    return reinterpret_cast<T*>(as_ndarray(a).data);
}

struct DelaunayObj {
    nb::object points, simplices, neighbors, v2s;
    Triangulation T;
};

const NDArray& points_arg(nb::handle h, const char* func, bool contiguous = true) {
    if (!is_ndarray(h)) throw nb::type_error((std::string(func) + "(): expected an sd.ndarray").c_str());
    const NDArray& a = as_ndarray(h);
    if (a.ndim != 2 || a.dtype != DType::Float64 || (contiguous && !a.c_contiguous))
        throw nb::type_error((std::string(func) + "(): wrong layout").c_str());
    return a;
}

// The points' own copy (n, d), C-contiguous, from any strides (SPT98: the points are always copied).
nb::object points_copy(const NDArray& a) {
    const std::int64_t n = a.shape[0], d = a.shape[1];
    nb::object out = array_of(DType::Float64, {n, d});
    auto* o = data_of<double>(out);
    for (std::int64_t i = 0; i < n; ++i)
        for (std::int64_t k = 0; k < d; ++k)
            std::memcpy(o + i * d + k, a.data + i * a.strides[0] + k * a.strides[1], 8);
    return out;
}

// d = 3 (delaunay3.cpp): the build, then the canonical form, the vertex to simplex map and the grid.
void new_delaunay3(DelaunayObj& obj, std::int64_t max_simplices) {
    Triangulation& T = obj.T;
    const std::int64_t n = T.n;
    const bool unlock = n * 64 >= tri::kParallelWork, split = n > kLeaf;
    const std::int64_t max_tets = max_simplices < 0 ? tri::default_max_tets(n) : max_simplices;
    tri::Internal3 in;
    tri::Build3 st;
    {
        const parallel::Unlocked unlocked(unlock);
        st = tri::build_delaunay3(T, in, max_tets, split);
    }
    if (st == tri::Build3::TooFew)
        throw nb::value_error(("Delaunay(): needs at least 4 distinct points, got " + std::to_string(T.m)).c_str());
    if (st == tri::Build3::Coplanar) throw nb::value_error("Delaunay(): 'points' are coplanar");
    if (st == tri::Build3::Memory) {
        PyErr_SetString(
            PyExc_MemoryError, "Delaunay(): the triangulation is larger than this machine's physical memory"
        );
        throw nb::python_error();
    }
    T.t = in.t;
    obj.simplices = array_of(DType::Int64, {in.t, 4});
    obj.neighbors = array_of(DType::Int64, {in.t, 4});
    obj.v2s = array_of(DType::Int64, {n});
    T.simplices = data_of<std::int64_t>(obj.simplices);
    T.neighbors = data_of<std::int64_t>(obj.neighbors);
    T.v2s = data_of<std::int64_t>(obj.v2s);
    {
        const parallel::Unlocked unlocked(unlock);
        tri::canonical_form<3>(n, in.t, in.s.data(), in.nb.data(), T.simplices, T.neighbors, split);
        in = tri::Internal3();
        tri::vertex_to_simplex(T, split);
        tri::build_locator<3>(T, split);
    }
}

void new_delaunay(DelaunayObj* self, nb::handle ph, std::int64_t max_simplices) {
    const NDArray& a = points_arg(ph, "_Delaunay", false);
    const std::int64_t n = a.shape[0];
    const int d = static_cast<int>(a.shape[1]);
    if (d != 2 && d != 3) throw nb::value_error("Delaunay(): 'points' must have 2 or 3 columns");
    if (n >= INT32_MAX) throw nb::value_error("Delaunay(): too many points");
    if (d == 3) {
        DelaunayObj obj;  // moved into self once complete: an exception leaves nothing half built
        obj.points = points_copy(a);
        obj.T.d = 3;
        obj.T.n = n;
        obj.T.pts = data_of<const double>(obj.points);
        new_delaunay3(obj, max_simplices);
        new (self) DelaunayObj(std::move(obj));
        return;
    }
    DelaunayObj obj;  // moved into self once complete: an exception leaves nothing half built
    obj.points = points_copy(a);
    Triangulation& T = obj.T;
    T.d = d;
    T.n = n;
    T.pts = data_of<const double>(obj.points);
    // Without the GIL from the array area's threshold; split only past one subtree task (SPT128): below it the
    // linear passes have too few chunks to pay for waking the pool.
    const bool unlock = n * 64 >= tri::kParallelWork, split = n > kLeaf;
    Internal in;
    Status st;
    {
        const parallel::Unlocked unlocked(unlock);
        st = build_internal(T, in, split);
    }
    if (st == Status::TooFew)
        throw nb::value_error(("Delaunay(): needs at least 3 distinct points, got " + std::to_string(T.m)).c_str());
    if (st == Status::Collinear) throw nb::value_error("Delaunay(): 'points' are collinear");
    if (st == Status::TooMany) throw nb::value_error("Delaunay(): too many points");
    T.t = in.t;
    obj.simplices = array_of(DType::Int64, {in.t, 3});
    obj.neighbors = array_of(DType::Int64, {in.t, 3});
    obj.v2s = array_of(DType::Int64, {n});
    T.simplices = data_of<std::int64_t>(obj.simplices);
    T.neighbors = data_of<std::int64_t>(obj.neighbors);
    T.v2s = data_of<std::int64_t>(obj.v2s);
    {
        const parallel::Unlocked unlocked(unlock);
        tri::canonical_form<2>(n, in.t, in.s.data(), in.nb.data(), T.simplices, T.neighbors, split);
        in = Internal();
        tri::vertex_to_simplex(T, split);
        tri::build_locator<2>(T, split);
    }
    new (self) DelaunayObj(std::move(obj));
}

nb::object find_simplex(const DelaunayObj& self, nb::handle xh) {
    const Triangulation& T = self.T;
    const NDArray& x = points_arg(xh, "_Delaunay.find_simplex");
    if (x.shape[1] != T.d) throw nb::type_error("_Delaunay.find_simplex(): wrong number of columns");
    const std::int64_t q = x.shape[0];
    nb::object out = array_of(DType::Int64, {q});
    const bool split = q * tri::kQueryWork >= tri::kParallelWork;
    const parallel::Unlocked unlocked(split);
    const auto* px = reinterpret_cast<const double*>(x.data);
    if (T.d == 2) tri::find_simplex<2>(T, px, q, data_of<std::int64_t>(out), split);
    else tri::find_simplex<3>(T, px, q, data_of<std::int64_t>(out), split);
    return out;
}

nb::object int64_rows(const std::vector<std::int64_t>& v, std::int64_t cols) {
    const auto rows = static_cast<std::int64_t>(v.size()) / cols;
    nb::object out = array_of(DType::Int64, {rows, cols});
    if (!v.empty()) std::memcpy(data_of<std::int64_t>(out), v.data(), 8 * v.size());
    return out;
}

}  // namespace

void bind_delaunay(nb::module_& m) {
    nb::class_<DelaunayObj>(m, "_Delaunay")
        .def("__init__", &new_delaunay, nb::arg("points"), nb::arg("max_simplices") = -1)
        .def_prop_ro("n", [](const DelaunayObj& o) { return o.T.n; })
        .def_prop_ro("ndim", [](const DelaunayObj& o) { return o.T.d; })
        .def_prop_ro("nsimplex", [](const DelaunayObj& o) { return o.T.t; })
        .def_prop_ro("ndistinct", [](const DelaunayObj& o) { return o.T.m; })
        .def("points", [](const DelaunayObj& o) { return o.points; })
        .def("simplices", [](const DelaunayObj& o) { return o.simplices; })
        .def("neighbors", [](const DelaunayObj& o) { return o.neighbors; })
        .def("vertex_to_simplex", [](const DelaunayObj& o) { return o.v2s; })
        .def("convex_hull", [](const DelaunayObj& o) { return int64_rows(tri::convex_hull(o.T), o.T.d); })
        .def("coplanar", [](const DelaunayObj& o) { return int64_rows(tri::coplanar_rows(o.T), 3); })
        .def(
            "vertex_neighbor_vertices",
            [](const DelaunayObj& o) {
                const Triangulation& T = o.T;
                nb::object indptr = array_of(DType::Int64, {T.n + 1});
                std::vector<std::int64_t> idx;
                const bool split = T.t > 4 * tri::kChunk;
                {
                    const parallel::Unlocked unlocked(T.t * 8 >= tri::kParallelWork);
                    if (T.d == 2) tri::vertex_neighbors2(T, data_of<std::int64_t>(indptr), idx, split);
                    else tri::vertex_neighbors3(T, data_of<std::int64_t>(indptr), idx, split);
                }
                nb::object indices = array_of(DType::Int64, {static_cast<std::int64_t>(idx.size())});
                if (!idx.empty()) std::memcpy(data_of<std::int64_t>(indices), idx.data(), 8 * idx.size());
                return nb::make_tuple(indptr, indices);
            }
        )
        .def(
            "transform",
            [](const DelaunayObj& o) {
                const Triangulation& T = o.T;
                nb::object out = array_of(DType::Float64, {T.t, T.d + 1, T.d});
                const bool split = T.t > 4 * tri::kChunk;
                const parallel::Unlocked unlocked(T.t * 16 >= tri::kParallelWork);
                if (T.d == 2) tri::transform2(T, data_of<double>(out), split);
                else tri::transform3(T, data_of<double>(out), split);
                return out;
            }
        )
        .def("find_simplex", &find_simplex, nb::arg("x"))
        .def("_grid", [](const DelaunayObj& o) {
            if (o.T.d == 2) return nb::make_tuple(o.T.loc2.g[0], o.T.loc2.g[1]);
            return nb::make_tuple(o.T.loc3.g[0], o.T.loc3.g[1], o.T.loc3.g[2]);
        });
}

}  // namespace sd::spatial
