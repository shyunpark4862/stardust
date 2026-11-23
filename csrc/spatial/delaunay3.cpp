// sd.spatial's Delaunay for d = 3: the build (docs/design/spatial_geometry.md §3.4, §3.10, §3.11; milestone
// SPT10, decisions SPT107, SPT156-SPT171). The native class _Delaunay (delaunay2.cpp) calls build_delaunay3
// for points with 3 columns and then makes the canonical form, the vertex to simplex map and the location
// grid of triangulation.hpp, which d = 2 and d = 3 share.
//
// THE ORDER. The distinct points (sorted lexicographically, equal points merged into the smallest index,
// SPT96) start with one tetrahedron: the lexicographically first two points, the first point after them not
// on their line and the first point after that not on their plane (none: ValueError, SPT97; the tests are
// exact). The others are inserted in BRIO order (sfc.hpp; rounds of doubling size from a fixed hash of the
// lexicographic rank, each along the Hilbert curve of the bounding box), whose keys and sort run by tasks
// (brio_order_parallel; the same order for any thread count).
//
// THE STRUCTURE. Tetrahedra with their four vertices and four neighbours (int32; n[k] = 4 * the tetrahedron
// across the facet opposite v[k] + that facet's index there, SPT158). The hull is closed by ghosts: a ghost
// has the point at infinity (kInf) as one vertex and its finite facet on the hull. Every tetrahedron is
// positive in the one sense the code uses: replacing its vertex k by a point x on the same side of the facet
// opposite k as v[k] gives orient3d > 0. For a finite tetrahedron that is orient3d(v) > 0; for a ghost with
// kInf at k it says the points beyond the hull facet (outside) have orient3d > 0 with x in kInf's place.
//
// INSERTION (Bowyer and Watson). A visibility walk from a tetrahedron of the previous point (the facets tried
// from a position that turns at each step, triangulation.hpp's rule; a Delaunay triangulation has no cycle of
// such steps) ends in a finite tetrahedron whose closed tetrahedron holds x, or steps into a ghost whose hull
// facet x is strictly beyond. Either is in conflict with x: a point of a closed inscribed tetrahedron that is
// not a vertex lies strictly inside the sphere. The conflict region grows breadth first over the neighbours:
//   finite tetrahedron   insphere_sos(v, x) > 0                 (the tie rule of §3.2)
//   ghost (kInf at k)    orient3d with x at k > 0, or = 0 and its finite neighbour is in conflict
// The second line is the tie rule on the hull (SPT159): for x on the plane of a hull facet abc the lifted
// determinant of abcd and x does not depend on d's perturbation (its cofactor is orient3d(a, b, c, x) = 0),
// so the finite neighbour's perturbed sphere decides abc as the 2-D tie rule in that plane would. The region
// is the set of tetrahedra whose perturbed sphere holds x, star-shaped from x, and its boundary facets with
// x make the new tetrahedra (a conflict tetrahedron with its vertex across the boundary facet replaced by
// x). The new tetrahedra around x are linked by the cavity boundary's edges (each in exactly two boundary
// facets; a small hash table per insertion). The slots of the deleted tetrahedra are reused.
//
// Every insphere_sos call has five distinct points (equal points are merged before the build and x is new),
// so SPT127's case (a repeated point the tie rule may call "inside") cannot arise in 3-D (SPT160).
//
// MEMORY (§3.4). The tetrahedra (ghosts included) are counted as they are made; past max_tets (what physical
// memory holds at kTetBytes each after kPointBytes3 per input point, or a test's smaller number) the build
// stops with Build3::Memory (MemoryError) and frees everything. The build is one thread (SPT107); the sort,
// the canonical form and the queries are parallel.
#include "spatial/delaunay3.hpp"

#include "base/buffer.hpp"
#include "spatial/predicates.hpp"
#include "spatial/sfc.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace sd::spatial::tri {

namespace {

constexpr std::int32_t kInf = -1;   // the point at infinity
constexpr std::int32_t kDead = -2;  // v[0] of a free slot

struct Tet {
    std::int32_t v[4];  // insertion positions; kInf at one position for a ghost
    std::int32_t n[4];  // n[k] = 4 * (the tetrahedron across the facet opposite v[k]) + (that facet's index there)
};

struct MemoryLimit {};

inline int inf_pos(const Tet& t) {
    for (int k = 0; k < 4; ++k)
        if (t.v[k] == kInf) return k;
    return -1;
}

// The edges of the cavity boundary -> the new tetrahedron's facet (4 * slot + facet) waiting for its twin.
class EdgeTable {
public:
    void prepare(std::size_t entries) {
        std::size_t cap = 64;
        while (cap < 2 * entries) cap *= 2;
        if (cap > key_.size()) {
            key_.assign(cap, kEmpty);
            val_.assign(cap, 0);
        }
    }
    // The waiting facet of edge k, or -1 after storing f for it.
    std::int32_t take(std::uint64_t k, std::int32_t f) {
        const std::uint64_t mask = key_.size() - 1;
        std::uint64_t h = k * 0x9e3779b97f4a7c15ULL;
        std::uint64_t i = (h ^ (h >> 29)) & mask;
        while (key_[i] != kEmpty) {
            if (key_[i] == k) return val_[i];
            i = (i + 1) & mask;
        }
        key_[i] = k;
        val_[i] = f;
        used_.push_back(i);
        return -1;
    }
    void clear() {
        for (const std::uint64_t i : used_) key_[i] = kEmpty;
        used_.clear();
    }

private:
    static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
    std::vector<std::uint64_t> key_;
    std::vector<std::int32_t> val_;
    std::vector<std::uint64_t> used_;
};

inline std::uint64_t edge_key(std::int32_t a, std::int32_t b) {
    const auto u = static_cast<std::uint32_t>(a < 0 ? INT32_MAX : a);
    const auto w = static_cast<std::uint32_t>(b < 0 ? INT32_MAX : b);
    return u < w ? (std::uint64_t{u} << 32) | w : (std::uint64_t{w} << 32) | u;
}

class Builder3 {
public:
    Builder3(const double* X, const Predicates& pred, std::int64_t m, std::int64_t max_tets)
        : X_(X), pred_(pred), max_tets_(max_tets) {
        const std::int64_t want = std::min<std::int64_t>(max_tets, 7 * m + 64);
        tets_.reserve(static_cast<std::size_t>(want));
        stamp_.reserve(static_cast<std::size_t>(want));
    }

    // The first tetrahedron: positions 0..3, positively oriented, and its four ghosts.
    void start() {
        const std::int32_t t0 = alloc();
        tets_[t0] = Tet{{0, 1, 2, 3}, {}};
        std::int32_t g[4];
        for (int k = 0; k < 4; ++k) {
            g[k] = alloc();
            Tet& G = tets_[g[k]];
            G = tets_[t0];
            G.v[k] = kInf;
            std::swap(G.v[(k + 1) & 3], G.v[(k + 2) & 3]);  // the other side of the facet
            G.n[k] = 4 * t0 + k;
            tets_[t0].n[k] = 4 * g[k] + k;
        }
        for (int i = 0; i < 4; ++i)  // the ghosts' facets through kInf: by their vertex sets
            for (int a = 0; a < 4; ++a) {
                if (a == i) continue;
                for (int j = i + 1; j < 4; ++j)
                    for (int b = 0; b < 4; ++b) {
                        if (b == j || !same_facet(tets_[g[i]], a, tets_[g[j]], b)) continue;
                        tets_[g[i]].n[a] = 4 * g[j] + b;
                        tets_[g[j]].n[b] = 4 * g[i] + a;
                    }
            }
        hint_ = t0;
    }

    void insert(std::int32_t v) {
        const double* x = pt(v);
        const std::int32_t t0 = locate(hint_, x);
        ++gen_;
        const std::uint32_t in = 2 * gen_, out = 2 * gen_ + 1;
        cavity_.clear();
        bnd_.clear();
        stamp_[static_cast<std::size_t>(t0)] = in;
        cavity_.push_back(t0);
        for (std::size_t h = 0; h < cavity_.size(); ++h) {  // breadth first, the cavity list as the queue
            const std::int32_t c = cavity_[h];
            for (int k = 0; k < 4; ++k) {
                const std::int32_t nb = tets_[static_cast<std::size_t>(c)].n[k] >> 2;
                const std::uint32_t s = stamp_[static_cast<std::size_t>(nb)];
                if (s == in) continue;
                if (s != out) {
                    if (conflict(nb, x)) {
                        stamp_[static_cast<std::size_t>(nb)] = in;
                        cavity_.push_back(nb);
                        continue;
                    }
                    stamp_[static_cast<std::size_t>(nb)] = out;
                }
                bnd_.emplace_back(c, k);
            }
        }
        // The new tetrahedra: each boundary facet's conflict tetrahedron with x across the facet.
        const std::size_t B = bnd_.size(), C = cavity_.size();
        newt_.resize(B);
        slot_.resize(B);
        for (std::size_t i = 0; i < B; ++i) {
            const auto [c, k] = bnd_[i];
            newt_[i] = tets_[static_cast<std::size_t>(c)];
            newt_[i].v[k] = v;
        }
        for (std::size_t i = 0; i < B; ++i) slot_[i] = i < C ? cavity_[i] : alloc();
        for (std::size_t i = B; i < C; ++i) {
            tets_[static_cast<std::size_t>(cavity_[i])].v[0] = kDead;
            free_.push_back(cavity_[i]);
        }
        table_.prepare(3 * B / 2 + 1);
        hint_ = -1;
        for (std::size_t i = 0; i < B; ++i) {
            const std::int32_t sl = slot_[i];
            const int k = bnd_[i].second;
            Tet& T = tets_[static_cast<std::size_t>(sl)];
            T = newt_[i];
            const std::int32_t outer = T.n[k];
            tets_[static_cast<std::size_t>(outer >> 2)].n[outer & 3] = 4 * sl + k;
            if (hint_ < 0 && inf_pos(T) < 0) hint_ = sl;
        }
        for (std::size_t i = 0; i < B; ++i) {
            const std::int32_t sl = slot_[i];
            const int k = bnd_[i].second;
            for (int j = 0; j < 4; ++j) {
                if (j == k) continue;
                int e[2], ne = 0;
                for (int a = 0; a < 4; ++a)
                    if (a != j && a != k) e[ne++] = a;
                const Tet& T = tets_[static_cast<std::size_t>(sl)];
                const std::int32_t f = table_.take(edge_key(T.v[e[0]], T.v[e[1]]), 4 * sl + j);
                if (f >= 0) {
                    tets_[static_cast<std::size_t>(sl)].n[j] = f;
                    tets_[static_cast<std::size_t>(f >> 2)].n[f & 3] = 4 * sl + j;
                }
            }
        }
        table_.clear();
    }

    // The finite tetrahedra in slot order: input indices (ids[position]) and neighbours (-1 across the hull).
    void extract(const std::vector<std::int32_t>& ids, Internal3& out) const {
        std::vector<std::int32_t> fid(tets_.size(), -1);
        std::int32_t t = 0;
        for (std::size_t s = 0; s < tets_.size(); ++s)
            if (tets_[s].v[0] != kDead && inf_pos(tets_[s]) < 0) fid[s] = t++;
        out.t = t;
        out.s.assign(static_cast<std::size_t>(4) * static_cast<std::size_t>(t), 0);
        out.nb.assign(static_cast<std::size_t>(4) * static_cast<std::size_t>(t), -1);
        for (std::size_t s = 0; s < tets_.size(); ++s) {
            const std::int32_t q = fid[s];
            if (q < 0) continue;
            for (int j = 0; j < 4; ++j) {
                out.s[static_cast<std::size_t>(4 * q + j)] = ids[static_cast<std::size_t>(tets_[s].v[j])];
                out.nb[static_cast<std::size_t>(4 * q + j)] = fid[static_cast<std::size_t>(tets_[s].n[j] >> 2)];
            }
        }
    }

private:
    const double* pt(std::int32_t v) const { return X_ + 3 * static_cast<std::int64_t>(v); }

    std::int32_t alloc() {
        if (!free_.empty()) {
            const std::int32_t s = free_.back();
            free_.pop_back();
            return s;
        }
        if (static_cast<std::int64_t>(tets_.size()) >= max_tets_) throw MemoryLimit{};
        tets_.push_back(Tet{});
        stamp_.push_back(0);
        return static_cast<std::int32_t>(tets_.size() - 1);
    }

    static bool same_facet(const Tet& s, int a, const Tet& t, int b) {
        std::int32_t x[3], y[3];
        for (int j = 0, p = 0, q = 0; j < 4; ++j) {
            if (j != a) x[p++] = s.v[j];
            if (j != b) y[q++] = t.v[j];
        }
        std::sort(x, x + 3);
        std::sort(y, y + 3);
        return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
    }

    // The orientation of t with its vertex k replaced by x (t's other vertices finite).
    int orient_at(const Tet& t, int k, const double* x) const {
        const double* q[4];
        for (int j = 0; j < 4; ++j) q[j] = j == k ? x : pt(t.v[j]);
        return pred_.orient3d(q[0], q[1], q[2], q[3]);
    }

    bool conflict(std::int32_t s, const double* x) const {
        const Tet& t = tets_[static_cast<std::size_t>(s)];
        const int k = inf_pos(t);
        if (k < 0) return pred_.insphere_sos(pt(t.v[0]), pt(t.v[1]), pt(t.v[2]), pt(t.v[3]), x) > 0;
        const int o = orient_at(t, k, x);
        if (o != 0) return o > 0;
        const Tet& f = tets_[static_cast<std::size_t>(t.n[k] >> 2)];  // the finite tetrahedron across the hull facet
        return pred_.insphere_sos(pt(f.v[0]), pt(f.v[1]), pt(f.v[2]), pt(f.v[3]), x) > 0;
    }

    // A tetrahedron in conflict with x, by the visibility walk from the finite tetrahedron s.
    std::int32_t locate(std::int32_t s, const double* x) const {
        std::int32_t prev = -1;
        int turn = 0;
        for (;;) {
            const Tet& t = tets_[static_cast<std::size_t>(s)];
            std::int32_t next = -1;
            for (int j = 0; j < 4; ++j) {
                const int k = (j + turn) & 3;
                const std::int32_t nb = t.n[k] >> 2;
                if (nb == prev) continue;
                if (orient_at(t, k, x) < 0) {
                    next = nb;
                    break;
                }
            }
            if (next < 0) return s;
            if (inf_pos(tets_[static_cast<std::size_t>(next)]) >= 0) return next;
            prev = s;
            s = next;
            turn = (turn + 1) & 3;
        }
    }

    const double* X_;
    Predicates pred_;
    std::int64_t max_tets_;
    std::vector<Tet> tets_;
    std::vector<std::uint32_t> stamp_;
    std::vector<std::int32_t> free_;
    std::vector<std::int32_t> cavity_, slot_;
    std::vector<std::pair<std::int32_t, int>> bnd_;
    std::vector<Tet> newt_;
    EdgeTable table_;
    std::uint32_t gen_ = 0;
    std::int32_t hint_ = 0;
};

}  // namespace

std::int64_t default_max_tets(std::int64_t n) {
    const auto mem = static_cast<double>(physical_memory());
    const double fit = (mem - static_cast<double>(kPointBytes3) * static_cast<double>(n)) / kTetBytes;
    return fit <= 0.0 ? 0 : static_cast<std::int64_t>(std::min(fit, static_cast<double>(kMaxTets)));
}

Build3 build_delaunay3(Triangulation& T, Internal3& out, std::int64_t max_tets, bool split) {
    const std::int64_t n = T.n;
    const double* pts = T.pts;
    std::vector<std::int64_t> distinct;
    {
        const std::vector<std::int64_t> order = lex_order_parallel(pts, n, 3, split);
        distinct = representatives(pts, 3, order, T.rep);
    }
    const auto m = static_cast<std::int64_t>(distinct.size());
    T.m = m;
    if (m < 4) return Build3::TooFew;
    T.range = range_of(pts, 3 * n, split);
    T.pred = Predicates(T.range);
    const Predicates& pred = T.pred;

    // The first tetrahedron (SPT156): the first two points, the first after them off their line, the first
    // after that off their plane (every point before it is on the line, so on the plane too).
    auto P = [&](std::int64_t r) { return pts + 3 * distinct[static_cast<std::size_t>(r)]; };
    auto collinear = [&](const double* a, const double* b, const double* c) {
        for (int k = 0; k < 3; ++k) {
            const int k1 = (k + 1) % 3;
            const double pa[2] = {a[k], a[k1]}, pb[2] = {b[k], b[k1]}, pc[2] = {c[k], c[k1]};
            if (pred.orient2d(pa, pb, pc) != 0) return false;
        }
        return true;
    };
    std::int64_t rc = 2;
    while (rc < m && collinear(P(0), P(1), P(rc))) ++rc;
    if (rc == m) return Build3::Coplanar;
    std::int64_t rd = rc + 1;
    int o = 0;
    while (rd < m && (o = pred.orient3d(P(0), P(1), P(rc), P(rd))) == 0) ++rd;
    if (rd == m) return Build3::Coplanar;
    std::int64_t first[4] = {0, 1, rc, rd};
    if (o < 0) std::swap(first[2], first[3]);

    // The insertion order: the first four, then BRIO without them.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(m));
    std::vector<double> X(static_cast<std::size_t>(3 * m));
    {
        std::int64_t pos = 0;
        auto put = [&](std::int64_t i) {
            ids[static_cast<std::size_t>(pos)] = static_cast<std::int32_t>(i);
            std::memcpy(X.data() + 3 * pos, pts + 3 * i, 24);
            ++pos;
        };
        for (const std::int64_t r : first) put(distinct[static_cast<std::size_t>(r)]);
        const std::int64_t i0 = distinct[0], i1 = distinct[1], i2 = distinct[static_cast<std::size_t>(rc)],
                           i3 = distinct[static_cast<std::size_t>(rd)];
        const std::vector<std::int64_t> brio = brio_order_parallel(pts, 3, distinct, split);
        for (const std::int64_t i : brio)
            if (i != i0 && i != i1 && i != i2 && i != i3) put(i);
    }

    try {
        Builder3 B(X.data(), pred, m, max_tets);
        B.start();
        for (std::int64_t r = 4; r < m; ++r) B.insert(static_cast<std::int32_t>(r));
        B.extract(ids, out);
    } catch (const MemoryLimit&) {
        out = Internal3();
        return Build3::Memory;
    }
    return Build3::Ok;
}

}  // namespace sd::spatial::tri
