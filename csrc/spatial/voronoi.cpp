// sd.spatial's Voronoi diagram for d = 2 and 3 (docs/design/spatial_geometry.md §2.5, §3.7, §3.10, §3.11, §4;
// milestone SPT12, decisions SPT110, SPT111, SPT172-SPT189). The Python class is src/stardust/spatial/_voronoi.py,
// which checks the arguments.
//
//   _voronoi(points, max_simplices=-1, simplex_vertex=False)
//       points (n, d) float64 of any strides, finite (checked by the caller), d = 2 or 3. Returns (points,
//       vertices, ridge_points, ridge_indptr, ridge_indices, region_indptr, region_indices, point_region) and,
//       with simplex_vertex, the Voronoi vertex of every simplex of the canonical Delaunay triangulation
//       (tests). The errors of Delaunay with "Voronoi()" (ValueError for fewer than d + 1 distinct points or
//       points on one line / plane; d = 3: MemoryError past max_simplices tetrahedra, -1 for physical memory).
//   _circumcenter(rows, wide)
//       rows (m, d + 1, d) float64, C-contiguous, positive simplices: (centres (m, d), numerators (m, d),
//       denominators (m,)) of the kernel below (tests); wide forces the exact path.
//
// THE DIAGRAM IS THE DUAL OF THE DELAUNAY SUBDIVISION (§3.7). The canonical Delaunay triangulation (§3.3, §3.4;
// its build and canonical form are Delaunay's) is a refinement of the subdivision into cells whose vertices lie
// on one empty sphere. Two neighbouring simplices are in one cluster when the opposite vertex of one is exactly
// on the other's sphere (the unperturbed incircle / insphere is 0, SPT173); a cluster is one cell, so all its
// simplices have one circumsphere (the cell is convex and triangulated by them, hence connected across
// facets), and every simplex with that sphere is in it. A cluster is one Voronoi vertex, numbered in the order
// of its smallest simplex (§2.5); its coordinates are that simplex's circumcentre.
//
// THE RIDGES (SPT176). A Delaunay edge (i, j) has the simplices around it in a cycle (d = 2: the one or two
// triangles at the edge; d = 3: the tetrahedra around the edge). On the hull the cycle is closed by the ghosts,
// the facets' outside: two ghosts at one edge are one cluster when their hull facets are coplanar (the same
// half-space, the dual of a sphere through infinity). After merging consecutive equal clusters the ridge has
// as many vertices as runs, and it has a (d - 1)-dimensional size exactly when there are at least d runs:
// d = 2: two different sides; d = 3: a polygon of at least three distinct points. Consecutive vertices of a
// 3-D ridge are never collinear with their neighbours (two consecutive dual segments are perpendicular to the
// facets i j x and i j y of one tetrahedron, which are not parallel), so three runs are a true polygon; and a
// ray-only ridge (one cluster between two coplanar hull triangles: the diagonal of a flat hull face) has one
// ghost run and is dropped. The ghosts are -1 in the output, once.
//
// THE ORDERS (§2.5, SPT177). Regions: d = 2 counterclockwise around the point (the star's triangles in that
// order), -1 first when the point is on the hull, else the smallest vertex first; d = 3 increasing. Ridges:
// the pairs i < j in lexicographic order; d = 2 the two vertices increasing, or (-1, v); d = 3 counterclockwise
// seen from i towards j (the tetrahedra around the edge in the negative sense of the axis j - i), -1 first or
// the smallest first. Every order is fixed by the canonical arrays, not by the walk's start.
//
// THE VERTEX KERNEL (§4, SPT174). With a the first vertex of the cluster's smallest simplex, v - a = N / (2 D):
// D = det[b - a; c - a(; d - a)] (det2 / det3, 1.002 u), N_k the same determinant with column k replaced by
// the squared lengths L = |p - a|^2. N is computed as the value kernels compute D: the differences as exact
// pairs h + t (two_diff), the squared lengths as Lh + Ll, the head products error free, the first-order terms
// in float64. Counted as in predicates.hpp (every term's magnitude times the roundings it passes, the
// neglected terms of second order, Pi the exact permanent sum |L_i h h| of the head expansion):
//   lengths:  L = Lh + Ll + eps, |Ll| <= 5.01 u Lh (d = 3; 4.01 u for d = 2), |eps| <= 21 u^2 Lh;
//   d = 2:    first-order terms 7.03 u Pi, at most 4 roundings: 28.2 u^2 Pi; eps 13.3; second order 4.1:
//             |r - x| <= u |r| + 46 u^2 Pi;
//   d = 3:    first-order terms 12.1 u Pi, at most 7 roundings: 85 u^2 Pi; minor2's low part 4.1; eps 21.1;
//             second order 21.2: |r - x| <= u |r| + 132 u^2 Pi.
// r is accepted when |r| >= 2^-30 P^ (P^ the float permanent, Pi <= 1.0001 P^): the second-order part is then
// below 132 * 2^-23 u |r| < 0.0001 u |r|, so |r - x| <= 1.002 u |x| as for det2 and det3. 2^-30 instead of
// det's 2^-36 leaves a factor of 100 over the counts. Otherwise (or outside the range promise for degree
// d + 1) the wide path computes N and D exactly and divides their correctly rounded mantissas (u each, the
// quotient u): |w - (v - a)| <= 3.01 u |v - a| with the exponent applied at the end, so the quotient
// overflows to +-inf only when v - a does; past 2^1022 the sum is taken halved (v = 2 (a / 2 + w / 2)), so
// v - a may pass DBL_MAX when v does not. Then v = a + w: |v^_k - v_k| <= u |v_k| + 4 u |v_k - a_k| (§4;
// near the subnormals 2^-1074 more). A centre that overflows is +-inf (SPT175).
//
// PARALLEL (§3.10, SPT180). The cluster tests and the centres go by simplex chunks, the regions and ridges by
// point chunks of kPointChunk (fixed sizes) whose rows are copied in chunk order; the union-find is one
// thread over the flagged facets in simplex order. Without the GIL from n * 64 >= 2^15, split past 2^14
// points (Delaunay's rule, SPT128). Every array is the same for any thread count.
#include "array/ndarray.hpp"
#include "base/buffer.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/delaunay2.hpp"
#include "spatial/delaunay3.hpp"
#include "spatial/predicates.hpp"
#include "spatial/spatial.hpp"
#include "spatial/triangulation.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

using i64 = std::int64_t;
using i32 = std::int32_t;
using tri::chunks_of;
using tri::run_tasks;
using tri::Triangulation;

constexpr i64 kLeaf = 1 << 14;              // split past this many points (SPT128)
constexpr i64 kPointChunk = 1 << 10;        // points a task of the region and ridge pass takes
constexpr double kNumeratorAccept = 0x1p-30;  // the vertex kernel's acceptance (SPT174)
// The bytes one tetrahedron stands for after a 3-D build (SPT181): the canonical rows and neighbours 64, the
// cluster arrays 17, the stars 16, the vertex 24, the ridges and regions with their chunk copies about 210.
constexpr i64 kVoronoiTetBytes = 352;

// ---------------------------------------------------------------- the vertex kernel (SPT174)

// |h + t|^2 = Lh + Ll + eps for the exact pairs h + t of D coordinates.
template <int D>
inline void squared_length(const double* h, const double* t, double& Lh, double& Ll) {
    double p0, e0, p1, e1;
    two_prod(h[0], h[0], p0, e0);
    two_prod(h[1], h[1], p1, e1);
    if constexpr (D == 2) {
        double g;
        two_sum(p0, p1, Lh, g);
        Ll = ((e0 + e1) + g) + 2.0 * (h[0] * t[0] + h[1] * t[1]);
    } else {
        double p2, e2, s, g1, g2;
        two_prod(h[2], h[2], p2, e2);
        two_sum(p0, p1, s, g1);
        two_sum(s, p2, Lh, g2);
        Ll = (((e0 + e1) + e2) + (g1 + g2)) + 2.0 * ((h[0] * t[0] + h[1] * t[1]) + h[2] * t[2]);
    }
}

// The numerators N (d = 2) of the circumcentre of a, b, c: N_x = L_b c_y - L_c b_y, N_y = L_c b_x - L_b c_x
// (differences from a). False when one is not certified (the wide path decides). The points are safe for q = 3.
inline bool numerators2(const double* a, const double* b, const double* c, double* N) {
    double hb[2], tb[2], hc[2], tc[2];
    for (int k = 0; k < 2; ++k) {
        two_diff(b[k], a[k], hb[k], tb[k]);
        two_diff(c[k], a[k], hc[k], tc[k]);
    }
    double Lb, lb, Lc, lc;
    squared_length<2>(hb, tb, Lb, lb);
    squared_length<2>(hc, tc, Lc, lc);
    for (int comp = 0; comp < 2; ++comp) {
        const int k = 1 - comp;  // N_x uses the y column, N_y the x column with the sign flipped
        double P, E, Q, F, s, g;
        two_prod(Lb, hc[k], P, E);
        two_prod(Lc, hb[k], Q, F);
        two_diff(P, Q, s, g);
        const double low = ((E - F) + g) + ((Lb * tc[k] + lb * hc[k]) - (Lc * tb[k] + lc * hb[k]));
        const double r = s + low;
        const double perm = std::fabs(P) + std::fabs(Q);
        if (perm == 0.0) {
            N[comp] = 0.0;  // both head products 0: the coordinate k of b and c equals a's, exactly
            continue;
        }
        if (!(std::fabs(r) >= kNumeratorAccept * perm)) return false;
        N[comp] = comp == 0 ? r : -r;
    }
    return true;
}

// The numerators N (d = 3) of the circumcentre of a, b, c, d: N_x = det[L, y, z], N_y = -det[L, x, z],
// N_z = det[L, x, y] over the rows b, c, d minus a. The points are safe for q = 4.
inline bool numerators3(const double* a, const double* b, const double* c, const double* d, double* N) {
    double h[3][3], t[3][3];
    const double* p[3] = {b, c, d};
    double tails = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            two_diff(p[i][k], a[k], h[i][k], t[i][k]);
            tails += std::fabs(t[i][k]);
        }
    double L[3], l[3];
    for (int i = 0; i < 3; ++i) squared_length<3>(h[i], t[i], L[i], l[i]);
    constexpr int kJ[3] = {1, 0, 0}, kK[3] = {2, 2, 1};
    for (int comp = 0; comp < 3; ++comp) {
        const int j = kJ[comp], k = kK[comp];
        double s1, m1, q1, s2, m2, q2, s3, m3, q3;
        detail::minor2(h[1][j], h[2][k], h[1][k], h[2][j], s1, m1, q1);  // c_j d_k - c_k d_j
        detail::minor2(h[0][j], h[2][k], h[0][k], h[2][j], s2, m2, q2);  // b_j d_k - b_k d_j
        detail::minor2(h[0][j], h[1][k], h[0][k], h[1][j], s3, m3, q3);  // b_j c_k - b_k c_j
        double P1, E1, P2, E2, P3, E3, S1, ga, S, gb;
        two_prod(L[0], s1, P1, E1);
        two_prod(L[1], s2, P2, E2);
        two_prod(L[2], s3, P3, E3);
        two_diff(P1, P2, S1, ga);
        two_sum(S1, P3, S, gb);
        double low = ((ga + gb) + ((E1 - E2) + E3)) +
                     (((L[0] * m1 - L[1] * m2) + L[2] * m3) + ((l[0] * s1 - l[1] * s2) + l[2] * s3));
        if (tails != 0.0) {
            const double T1 = (t[1][j] * h[2][k] + h[1][j] * t[2][k]) - (t[1][k] * h[2][j] + h[1][k] * t[2][j]);
            const double T2 = (t[0][j] * h[2][k] + h[0][j] * t[2][k]) - (t[0][k] * h[2][j] + h[0][k] * t[2][j]);
            const double T3 = (t[0][j] * h[1][k] + h[0][j] * t[1][k]) - (t[0][k] * h[1][j] + h[0][k] * t[1][j]);
            low += (L[0] * T1 - L[1] * T2) + L[2] * T3;
        }
        const double r = S + low;
        const double perm = (L[0] * q1 + L[1] * q2) + L[2] * q3;
        if (perm == 0.0) {
            N[comp] = 0.0;  // every head minor's products are 0: the minors are exactly 0
            continue;
        }
        if (!(std::fabs(r) >= kNumeratorAccept * perm)) return false;
        N[comp] = comp == 1 ? -r : r;
    }
    return true;
}

// num / den * 2^e from the correctly rounded mantissas (num, den != 0): within 3.01 u, +-inf on overflow.
inline double scaled_quotient(const WideInt& num, const WideInt& den, i64 e) {
    if (num.sign() == 0) return 0.0;
    const i64 bn = num.bits(), bd = den.bits();
    const double q = num.to_double(1 - bn) / den.to_double(1 - bd);  // mantissas in [1, 2]
    const i64 shift = std::clamp<i64>((bn - 1) - (bd - 1) + e, -4000, 4000);
    return std::ldexp(q, static_cast<int>(shift));
}

inline WideInt det2w(const WideInt& a, const WideInt& b, const WideInt& c, const WideInt& d) {
    return sub(mul(a, d), mul(b, c));
}

inline WideInt det3w(const WideInt (&m)[3][3]) {
    return add(
        sub(mul(m[0][0], det2w(m[1][1], m[1][2], m[2][1], m[2][2])),
            mul(m[0][1], det2w(m[1][0], m[1][2], m[2][0], m[2][2]))),
        mul(m[0][2], det2w(m[1][0], m[1][1], m[2][0], m[2][1]))
    );  // fmt: skip
}

// The exact numerators and denominator of the circumcentre of p[0..D] (coordinates over 2^e, e the lowest set
// bit): v - p[0] = N / (2 D) * 2^e.
template <int D>
inline void centre_exact(const double* const* p, WideInt* N, WideInt& den, int& e) {
    Range r;
    for (int i = 0; i <= D; ++i) r.add(p[i], D);
    e = r.lo;
    WideInt m[D][D], L[D];
    for (int i = 0; i < D; ++i) {
        for (int k = 0; k < D; ++k) m[i][k] = sub(WideInt::scaled(p[i + 1][k], e), WideInt::scaled(p[0][k], e));
        L[i] = mul(m[i][0], m[i][0]);
        for (int k = 1; k < D; ++k) L[i] = add(L[i], mul(m[i][k], m[i][k]));
    }
    if constexpr (D == 2) {
        den = det2w(m[0][0], m[0][1], m[1][0], m[1][1]);
        N[0] = det2w(L[0], m[0][1], L[1], m[1][1]);
        N[1] = det2w(m[0][0], L[0], m[1][0], L[1]);
    } else {
        den = det3w(m);
        for (int k = 0; k < 3; ++k) {
            WideInt c[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) c[i][j] = j == k ? L[i] : m[i][j];
            N[k] = det3w(c);
        }
    }
}

template <int D>
inline void centre_wide(const double* const* p, double* v) {
    WideInt N[D], den;
    int e = 0;
    centre_exact<D>(p, N, den, e);
    for (int k = 0; k < D; ++k) {
        const double w = scaled_quotient(N[k], den, i64{e} - 1);
        if (std::fabs(w) <= 0x1p1022) {
            v[k] = p[0][k] + w;
        } else {  // v - a may overflow when v does not (a near DBL_MAX): v = 2 (a / 2 + (v - a) / 2)
            v[k] = 2.0 * (0.5 * p[0][k] + scaled_quotient(N[k], den, i64{e} - 2));
        }
    }
}

// The circumcentre of the positive simplex p[0..D] (§4). `safe`: the point set's range keeps the fast path
// exact for degree D + 1 (otherwise the simplex's own range is checked).
template <int D>
inline void circumcentre(const double* const* p, double* v, bool safe) {
    if (!safe) {
        Range r;
        for (int i = 0; i <= D; ++i) r.add(p[i], D);
        safe = r.safe(D + 1);
    }
    if (safe) {
        double N[D];
        bool ok;
        if constexpr (D == 2) ok = numerators2(p[0], p[1], p[2], N);
        else ok = numerators3(p[0], p[1], p[2], p[3], N);
        if (ok) {
            double det;
            if constexpr (D == 2) det = det2(p[1], p[2], p[0]);  // det[b - a; c - a]
            else det = det3(p[0], p[1], p[2], p[3]);
            const double den = 2.0 * det;
            for (int k = 0; k < D; ++k) v[k] = p[0][k] + N[k] / den;
            return;
        }
    }
    centre_wide<D>(p, v);
}

// ---------------------------------------------------------------- the build

enum class Error { None, TooFew, Collinear, Coplanar, TooMany, Memory };

// An array left uninitialized (every element is written before it is read).
template <class T>
class Raw {
public:
    void resize(std::size_t n) { p_.reset(new T[n]); }
    T* data() { return p_.get(); }
    const T* data() const { return p_.get(); }
    T& operator[](std::size_t i) { return p_[i]; }
    const T& operator[](std::size_t i) const { return p_[i]; }

private:
    std::unique_ptr<T[]> p_;
};

struct Diagram {
    i64 t = 0;                  // simplices of the triangulation
    Raw<i64> S, NB, v2s;  // the canonical triangulation
    Raw<i64> vid;         // the Voronoi vertex of each simplex
    i64 nv = 0;                  // Voronoi vertices
    std::vector<i64> roots;      // the smallest simplex of each vertex's cluster, in vertex order
};

Error triangulate(Triangulation& T, Diagram& G, i64 max_tets, bool split) {
    const int d = T.d, K = d + 1;
    const i64 n = T.n;
    i64 t = 0;
    if (d == 2) {
        tri::Internal2 in;
        switch (tri::build_delaunay2(T, in, split)) {
            case tri::Build2::Ok: break;
            case tri::Build2::TooFew: return Error::TooFew;
            case tri::Build2::Collinear: return Error::Collinear;
            case tri::Build2::TooMany: return Error::TooMany;
        }
        t = in.t;
        G.S.resize(static_cast<std::size_t>(K * t));
        G.NB.resize(static_cast<std::size_t>(K * t));
        tri::canonical_form<2>(n, t, in.s.data(), in.nb.data(), G.S.data(), G.NB.data(), split);
    } else {
        tri::Internal3 in;
        switch (tri::build_delaunay3(T, in, max_tets, split)) {
            case tri::Build3::Ok: break;
            case tri::Build3::TooFew: return Error::TooFew;
            case tri::Build3::Coplanar: return Error::Coplanar;
            case tri::Build3::Memory: return Error::Memory;
        }
        t = in.t;
        // The arrays after the build hold kVoronoiTetBytes per tetrahedron (SPT181).
        if (max_tets >= 0 && t > max_tets * tri::kTetBytes / kVoronoiTetBytes) return Error::Memory;
        G.S.resize(static_cast<std::size_t>(K * t));
        G.NB.resize(static_cast<std::size_t>(K * t));
        tri::canonical_form<3>(n, t, in.s.data(), in.nb.data(), G.S.data(), G.NB.data(), split);
    }
    G.t = T.t = t;
    T.simplices = G.S.data();
    T.neighbors = G.NB.data();
    G.v2s.resize(static_cast<std::size_t>(n));
    T.v2s = G.v2s.data();
    tri::vertex_to_simplex(T, split);
    return Error::None;
}

// The clusters (SPT173): flag the facets whose two simplices have one circumsphere, join them (union-find with
// the smaller root kept, so parent[x] <= x and one forward pass gives every root), number the roots.
void clusters(const Triangulation& T, Diagram& G, bool split) {
    const int d = T.d, K = d + 1;
    const i64 t = G.t;
    const i64* S = G.S.data();
    const i64* NB = G.NB.data();
    std::vector<std::uint8_t> flag(static_cast<std::size_t>(t));
    const i64 nc = chunks_of(t, tri::kChunk);
    std::vector<std::uint8_t> any(static_cast<std::size_t>(std::max<i64>(nc, 1)), 0);
    run_tasks(nc, split, [&](i64 c) {
        std::uint8_t seen = 0;
        for (i64 q = c * tri::kChunk, e = std::min(t, (c + 1) * tri::kChunk); q < e; ++q) {
            std::uint8_t f = 0;
            const double* p[4];
            for (int k = 0; k < K; ++k) p[k] = T.pts + S[q * K + k] * d;
            for (int k = 0; k < K; ++k) {
                const i64 r = NB[q * K + k];
                if (r <= q) continue;
                int j = 0;
                while (NB[r * K + j] != q) ++j;
                const double* x = T.pts + S[r * K + j] * d;
                const int s =
                    d == 2 ? T.pred.incircle(p[0], p[1], p[2], x) : T.pred.insphere(p[0], p[1], p[2], p[3], x);
                if (s == 0) f = static_cast<std::uint8_t>(f | (1u << k));
            }
            flag[static_cast<std::size_t>(q)] = f;
            seen |= f;
        }
        any[static_cast<std::size_t>(c)] = seen;
    });
    G.vid.resize(static_cast<std::size_t>(t));
    i64* vid = G.vid.data();
    bool joined = false;
    for (const std::uint8_t a : any) joined = joined || a != 0;
    if (!joined) {  // every simplex is its own vertex
        G.nv = t;
        run_tasks(nc, split, [&](i64 c) {
            for (i64 q = c * tri::kChunk, e = std::min(t, (c + 1) * tri::kChunk); q < e; ++q) vid[q] = q;
        });
        G.roots.clear();
        return;
    }
    std::vector<i64> parent(static_cast<std::size_t>(t));
    for (i64 q = 0; q < t; ++q) parent[static_cast<std::size_t>(q)] = q;
    auto find = [&](i64 x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            const i64 up = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            parent[static_cast<std::size_t>(x)] = up;
            x = up;
        }
        return x;
    };
    for (i64 q = 0; q < t; ++q) {
        const std::uint8_t f = flag[static_cast<std::size_t>(q)];
        if (f == 0) continue;
        for (int k = 0; k < K; ++k)
            if (f & (1u << k)) {
                const i64 a = find(q), b = find(NB[q * K + k]);
                if (a < b) parent[static_cast<std::size_t>(b)] = a;
                else if (b < a) parent[static_cast<std::size_t>(a)] = b;
            }
    }
    for (i64 q = 0; q < t; ++q) {  // parent[q] <= q: its root is already final
        const i64 p = parent[static_cast<std::size_t>(q)];
        if (p != q) parent[static_cast<std::size_t>(q)] = parent[static_cast<std::size_t>(p)];
    }
    std::vector<i64> count(static_cast<std::size_t>(nc) + 1, 0);
    run_tasks(nc, split, [&](i64 c) {
        i64 k = 0;
        for (i64 q = c * tri::kChunk, e = std::min(t, (c + 1) * tri::kChunk); q < e; ++q)
            k += parent[static_cast<std::size_t>(q)] == q;
        count[static_cast<std::size_t>(c) + 1] = k;
    });
    for (i64 c = 0; c < nc; ++c) count[static_cast<std::size_t>(c) + 1] += count[static_cast<std::size_t>(c)];
    G.nv = count[static_cast<std::size_t>(nc)];
    G.roots.resize(static_cast<std::size_t>(G.nv));
    run_tasks(nc, split, [&](i64 c) {
        i64 k = count[static_cast<std::size_t>(c)];
        for (i64 q = c * tri::kChunk, e = std::min(t, (c + 1) * tri::kChunk); q < e; ++q)
            if (parent[static_cast<std::size_t>(q)] == q) {
                vid[q] = k;
                G.roots[static_cast<std::size_t>(k++)] = q;
            }
    });
    run_tasks(nc, split, [&](i64 c) {
        for (i64 q = c * tri::kChunk, e = std::min(t, (c + 1) * tri::kChunk); q < e; ++q) {
            const i64 p = parent[static_cast<std::size_t>(q)];
            if (p != q) vid[q] = vid[p];
        }
    });
}

// The vertices (nv, d): the circumcentre of each cluster's smallest simplex.
void centres(const Triangulation& T, const Diagram& G, double* out, bool split) {
    const int d = T.d, K = d + 1;
    const bool safe = T.range.safe(d + 1);
    const i64 nv = G.nv;
    run_tasks(chunks_of(nv, tri::kChunk), split, [&](i64 c) {
        for (i64 v = c * tri::kChunk, e = std::min(nv, (c + 1) * tri::kChunk); v < e; ++v) {
            const i64 q = G.roots.empty() ? v : G.roots[static_cast<std::size_t>(v)];
            const double* p[4];
            for (int k = 0; k < K; ++k) p[k] = T.pts + G.S[static_cast<std::size_t>(q * K + k)] * d;
            if (d == 2) circumcentre<2>(p, out + v * 2, safe);
            else circumcentre<3>(p, out + v * 3, safe);
        }
    });
}

// What one task of points produces: the regions' rows (their lengths go to region_indptr[i + 1]) and the
// ridges of the pairs (i, j > i) of its points, in order.
struct Part {
    std::vector<i64> region;
    std::vector<i64> pairs;     // (i, j) per ridge
    std::vector<i64> lengths;   // vertices per ridge
    std::vector<i64> vertices;  // the ridges' vertices
};

inline int position(const i64* s, i64 v) {
    int k = 0;
    while (s[k] != v) ++k;
    return k;
}

// The runs of a cyclic list of clusters (consecutive equal ones merged, the last with the first too).
inline void cyclic_runs(std::vector<i64>& c) {
    c.erase(std::unique(c.begin(), c.end()), c.end());
    while (c.size() > 1 && c.front() == c.back()) c.pop_back();
}

inline void smallest_first(std::vector<i64>& c) {
    std::rotate(c.begin(), std::min_element(c.begin(), c.end()), c.end());
}

// A task's working lists, reused from point to point.
struct Side {
    i64 v, b, c;  // d = 2: a triangle's vertex; its edges (i, b) and (i, c), counterclockwise
};
struct Ridge2 {
    i64 j, a, b;  // a = -1 for an unbounded ridge
};
struct Scratch {
    std::vector<Side> star;
    std::vector<Ridge2> out;
    std::vector<i64> tmp, nbrs, cyc, back;
};

// d = 2: the region of point i and its ridges to j > i.
void point2(const Triangulation& T, const Diagram& G, i64 i, Part& P, Scratch& W) {
    const i64* S = G.S.data();
    const i64* NB = G.NB.data();
    const i64* vid = G.vid.data();
    // Turn clockwise to the hull (or all the way round), then walk counterclockwise.
    const i64 q0 = G.v2s[static_cast<std::size_t>(i)];
    i64 q = q0;
    bool hull = false;
    for (;;) {
        const int p = position(S + q * 3, i);
        const i64 nq = NB[q * 3 + (p + 2) % 3];  // across the edge (i, next vertex): clockwise
        if (nq < 0) {
            hull = true;
            break;
        }
        if (nq == q0) break;
        q = nq;
    }
    const i64 start = hull ? q : q0;
    std::vector<Side>& star = W.star;
    std::vector<i64>& tmp = W.tmp;
    star.clear();
    q = start;
    do {
        const int p = position(S + q * 3, i);
        star.push_back({vid[q], S[q * 3 + (p + 1) % 3], S[q * 3 + (p + 2) % 3]});
        q = NB[q * 3 + (p + 1) % 3];  // across the edge (i, c): counterclockwise
    } while (q >= 0 && q != start);
    const auto m = static_cast<i64>(star.size());
    // The region.
    tmp.clear();
    for (const Side& s : star) tmp.push_back(s.v);
    if (hull) {
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        P.region.push_back(-1);
    } else {
        cyclic_runs(tmp);
        smallest_first(tmp);
    }
    P.region.insert(P.region.end(), tmp.begin(), tmp.end());
    // The ridges: edge k sits between triangles k - 1 and k (edge 0 and, on the hull, edge m on one only).
    std::vector<Ridge2>& out = W.out;
    out.clear();
    for (i64 k = 0; k <= m; ++k) {
        if (k == m && !hull) break;
        const i64 j = k == 0 ? star[0].b : star[static_cast<std::size_t>(k - 1)].c;
        if (j < i) continue;
        if (hull && (k == 0 || k == m)) {
            out.push_back({j, -1, star[static_cast<std::size_t>(k == 0 ? 0 : m - 1)].v});
            continue;
        }
        const i64 u = star[static_cast<std::size_t>(k == 0 ? m - 1 : k - 1)].v, w = star[static_cast<std::size_t>(k)].v;
        if (u != w) out.push_back({j, std::min(u, w), std::max(u, w)});
    }
    std::sort(out.begin(), out.end(), [](const Ridge2& x, const Ridge2& y) { return x.j < y.j; });
    for (const Ridge2& r : out) {
        P.pairs.insert(P.pairs.end(), {i, r.j});
        P.lengths.push_back(2);
        P.vertices.insert(P.vertices.end(), {r.a, r.b});
    }
}

// The parity of a permutation of 0..3 (true: odd).
inline bool odd4(const int* p) {
    int inv = 0;
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b) inv += p[a] > p[b];
    return inv & 1;
}

struct Stars {
    std::vector<i64> start;  // (n + 1)
    std::vector<i32> tets;   // the tetrahedra holding each point (a counting sort; the order is not used)
};

Stars stars3(const Diagram& G, i64 n, bool split) {
    Stars st;
    const i64 t = G.t;
    st.start.assign(static_cast<std::size_t>(n) + 1, 0);
    const i64 tc = chunks_of(t, tri::kChunk);
    run_tasks(tc, split, [&](i64 c) {
        for (i64 q = c * tri::kChunk, e = std::min(t, (c + 1) * tri::kChunk); q < e; ++q)
            for (int j = 0; j < 4; ++j) tri::fetch_add(st.start[static_cast<std::size_t>(G.S[q * 4 + j]) + 1], 1);
    });
    for (i64 i = 0; i < n; ++i) st.start[static_cast<std::size_t>(i) + 1] += st.start[static_cast<std::size_t>(i)];
    st.tets.resize(static_cast<std::size_t>(4 * t));
    std::vector<i64> cursor(st.start.begin(), st.start.end() - 1);
    run_tasks(tc, split, [&](i64 c) {
        for (i64 q = c * tri::kChunk, e = std::min(t, (c + 1) * tri::kChunk); q < e; ++q)
            for (int j = 0; j < 4; ++j)
                st.tets[static_cast<std::size_t>(tri::fetch_add(cursor[static_cast<std::size_t>(G.S[q * 4 + j])], 1))] =
                    static_cast<i32>(q);
    });
    return st;
}

// d = 3: the region of point i and its ridges to j > i.
void point3(
    const Triangulation& T,
    const Diagram& G,
    const Stars& st,
    i64 i,
    Part& P,
    std::vector<i64>& tmp,
    std::vector<i64>& nbrs,
    std::vector<i64>& cyc,
    std::vector<i64>& back
) {  // fmt: skip
    const i64* S = G.S.data();
    const i64* NB = G.NB.data();
    const i64* vid = G.vid.data();
    const i64 lo = st.start[static_cast<std::size_t>(i)], hi = st.start[static_cast<std::size_t>(i) + 1];
    // The region: the clusters of the star, -1 when a hull facet holds i.
    tmp.clear();
    nbrs.clear();
    bool hull = false;
    for (i64 k = lo; k < hi; ++k) {
        const i64 q = st.tets[static_cast<std::size_t>(k)];
        tmp.push_back(vid[q]);
        for (int j = 0; j < 4; ++j) {
            const i64 v = S[q * 4 + j];
            if (v == i) continue;
            if (NB[q * 4 + j] < 0) hull = true;  // the facet opposite v holds i
            if (v > i) nbrs.push_back(v << 32 | q);  // the neighbour with a tetrahedron of the edge
        }
    }
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    if (hull) P.region.push_back(-1);
    P.region.insert(P.region.end(), tmp.begin(), tmp.end());
    // The ridges to the neighbours j > i, in increasing j, each from its smallest tetrahedron.
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(
        std::unique(nbrs.begin(), nbrs.end(), [](i64 x, i64 y) { return x >> 32 == y >> 32; }), nbrs.end()
    );  // fmt: skip
    const double* pi = T.pts + i * 3;
    for (const i64 packed : nbrs) {
        const i64 j = packed >> 32;
        i64 q = packed & 0xffffffff;
        // In q: i, j and a, b with orient3d(i, j, a, b) > 0 (b turns from a positively about j - i).
        auto frame = [&](i64 u, i64& a, i64& b) {
            int pos[4];
            pos[0] = position(S + u * 4, i);
            pos[1] = position(S + u * 4, j);
            int r = 2;
            for (int m = 0; m < 4; ++m)
                if (m != pos[0] && m != pos[1]) pos[r++] = m;
            if (odd4(pos)) std::swap(pos[2], pos[3]);
            a = S[u * 4 + pos[2]];
            b = S[u * 4 + pos[3]];
        };
        i64 a, b;
        frame(q, a, b);
        const i64 q0 = q, a0 = a, b0 = b;
        // Walk negatively (across the facet i j a): counterclockwise seen from i towards j, all the way round
        // or to the hull (the facet i j last_a).
        cyc.clear();
        bool open = false;
        i64 last_a = a, first_b = b;
        for (;;) {
            cyc.push_back(vid[q]);
            last_a = a;
            const i64 nq = NB[q * 4 + position(S + q * 4, b)];
            if (nq < 0) {
                open = true;
                break;
            }
            if (nq == q0) break;
            q = nq;
            frame(q, a, b);
        }
        if (open) {  // the tetrahedra before q0: positively (across the facet i j b) to the facet i j first_b
            back.clear();
            q = q0, a = a0, b = b0;
            for (;;) {
                const i64 nq = NB[q * 4 + position(S + q * 4, a)];
                if (nq < 0) break;
                q = nq;
                frame(q, a, b);
                back.push_back(vid[q]);
            }
            first_b = b;
            cyc.insert(cyc.begin(), back.rbegin(), back.rend());
        }
        if (open) {
            cyc.erase(std::unique(cyc.begin(), cyc.end()), cyc.end());
            const double* pj = T.pts + j * 3;
            const int ghosts =
                T.pred.orient3d(pi, pj, T.pts + first_b * 3, T.pts + last_a * 3) == 0 ? 1 : 2;  // one or two facets
            if (static_cast<i64>(cyc.size()) + ghosts < 3) continue;
            P.vertices.push_back(-1);
            P.lengths.push_back(static_cast<i64>(cyc.size()) + 1);
        } else {
            cyclic_runs(cyc);
            if (cyc.size() < 3) continue;
            smallest_first(cyc);
            P.lengths.push_back(static_cast<i64>(cyc.size()));
        }
        P.pairs.insert(P.pairs.end(), {i, j});
        P.vertices.insert(P.vertices.end(), cyc.begin(), cyc.end());
    }
}

// The arrays of the result (made without the GIL, wrapped after).
struct Output {
    NDArray vertices, ridge_points, ridge_indptr, ridge_indices, region_indptr, region_indices, point_region;
};

void regions_and_ridges(const Triangulation& T, Diagram& G, Output& O, bool split) {
    const i64 n = T.n;
    const int d = T.d;
    Stars st;
    if (d == 3) st = stars3(G, n, split);
    const i64 pc = chunks_of(n, kPointChunk);
    std::vector<Part> parts(static_cast<std::size_t>(std::max<i64>(pc, 1)));
    const i64 rsh[] = {n + 1};
    O.region_indptr = new_array(DType::Int64, rsh);
    auto* rind = reinterpret_cast<i64*>(O.region_indptr.data);
    rind[0] = 0;
    run_tasks(pc, split, [&](i64 c) {
        Part& P = parts[static_cast<std::size_t>(c)];
        Scratch W;
        for (i64 i = c * kPointChunk, e = std::min(n, (c + 1) * kPointChunk); i < e; ++i) {
            const auto before = static_cast<i64>(P.region.size());
            if (T.rep[static_cast<std::size_t>(i)] == i) {
                if (d == 2) point2(T, G, i, P, W);
                else point3(T, G, st, i, P, W.tmp, W.nbrs, W.cyc, W.back);
            }
            rind[i + 1] = static_cast<i64>(P.region.size()) - before;
        }
    });
    st = Stars();
    G = Diagram();  // the triangulation is done with: its memory goes before the result's arrays come
    for (i64 i = 0; i < n; ++i) rind[i + 1] += rind[i];
    // Offsets of the parts, then their rows copied in chunk order.
    std::vector<i64> roff(static_cast<std::size_t>(pc) + 1, 0), voff(static_cast<std::size_t>(pc) + 1, 0);
    for (i64 c = 0; c < pc; ++c) {
        const Part& P = parts[static_cast<std::size_t>(c)];
        roff[static_cast<std::size_t>(c) + 1] = roff[static_cast<std::size_t>(c)] + static_cast<i64>(P.lengths.size());
        voff[static_cast<std::size_t>(c) + 1] = voff[static_cast<std::size_t>(c)] + static_cast<i64>(P.vertices.size());
    }
    const i64 R = roff[static_cast<std::size_t>(pc)], RV = voff[static_cast<std::size_t>(pc)];
    const i64 ish[] = {rind[n]}, psh[] = {R, 2}, ipsh[] = {R + 1}, vsh[] = {RV};
    O.region_indices = new_array(DType::Int64, ish);
    O.ridge_points = new_array(DType::Int64, psh);
    O.ridge_indptr = new_array(DType::Int64, ipsh);
    O.ridge_indices = new_array(DType::Int64, vsh);
    auto* regi = reinterpret_cast<i64*>(O.region_indices.data);
    auto* rp = reinterpret_cast<i64*>(O.ridge_points.data);
    auto* rptr = reinterpret_cast<i64*>(O.ridge_indptr.data);
    auto* rv = reinterpret_cast<i64*>(O.ridge_indices.data);
    rptr[0] = 0;
    run_tasks(pc, split, [&](i64 c) {
        Part& P = parts[static_cast<std::size_t>(c)];
        if (!P.region.empty()) std::memcpy(regi + rind[c * kPointChunk], P.region.data(), 8 * P.region.size());
        const i64 r0 = roff[static_cast<std::size_t>(c)];
        if (!P.pairs.empty()) std::memcpy(rp + 2 * r0, P.pairs.data(), 8 * P.pairs.size());
        if (!P.vertices.empty())
            std::memcpy(rv + voff[static_cast<std::size_t>(c)], P.vertices.data(), 8 * P.vertices.size());
        i64 at = voff[static_cast<std::size_t>(c)];
        for (std::size_t k = 0; k < P.lengths.size(); ++k) {
            at += P.lengths[k];
            rptr[r0 + static_cast<i64>(k) + 1] = at;
        }
        P = Part();
    });
}

nb::object wrap_array(NDArray&& a) { return wrap(std::move(a)); }

nb::tuple voronoi(nb::handle ph, i64 max_simplices, bool simplex_vertex) {
    if (!is_ndarray(ph)) throw nb::type_error("_voronoi(): expected an sd.ndarray");
    const NDArray& a = as_ndarray(ph);
    if (a.ndim != 2 || a.dtype != DType::Float64 || (a.shape[1] != 2 && a.shape[1] != 3))
        throw nb::type_error("_voronoi(): wrong array layout");
    const i64 n = a.shape[0];
    const int d = static_cast<int>(a.shape[1]);
    if (n >= INT32_MAX) throw nb::value_error("Voronoi(): too many points");
    const i64 sh[] = {n, d};
    NDArray copy = new_array(DType::Float64, sh);
    auto* pts = reinterpret_cast<double*>(copy.data);
    const bool unlock = n * 64 >= tri::kParallelWork, split = n > kLeaf;
    i64 max_tets = max_simplices;
    if (d == 3 && max_tets < 0) {
        const auto mem = static_cast<double>(physical_memory());
        const double fit = (mem - static_cast<double>(tri::kPointBytes3) * static_cast<double>(n)) / tri::kTetBytes;
        max_tets = fit <= 0.0 ? 0 : static_cast<i64>(std::min(fit, static_cast<double>(tri::kMaxTets)));
    }
    Triangulation T;
    T.d = d;
    T.n = n;
    T.pts = pts;
    Diagram G;
    Output O;
    Error err = Error::None;
    NDArray sv;
    {
        const parallel::Unlocked unlocked(unlock);
        run_tasks(chunks_of(n, tri::kChunk), split, [&](i64 c) {
            for (i64 i = c * tri::kChunk, e = std::min(n, (c + 1) * tri::kChunk); i < e; ++i)
                for (int k = 0; k < d; ++k)
                    std::memcpy(pts + i * d + k, a.data + i * a.strides[0] + k * a.strides[1], sizeof(double));
        });
        err = triangulate(T, G, max_tets, split);
        if (err == Error::None) {
            clusters(T, G, split);
            if (simplex_vertex) {
                const i64 ssh[] = {G.t};
                sv = new_array(DType::Int64, ssh);
                if (G.t) std::memcpy(sv.data, G.vid.data(), 8 * static_cast<std::size_t>(G.t));
            }
            const i64 vsh[] = {G.nv, d};
            O.vertices = new_array(DType::Float64, vsh);
            centres(T, G, reinterpret_cast<double*>(O.vertices.data), split);
            G.roots = std::vector<i64>();
            regions_and_ridges(T, G, O, split);
            const i64 nsh[] = {n};
            O.point_region = new_array(DType::Int64, nsh);
            std::memcpy(O.point_region.data, T.rep.data(), 8 * static_cast<std::size_t>(n));
        }
    }
    switch (err) {
        case Error::None: break;
        case Error::TooFew:
            throw nb::value_error(("Voronoi(): needs at least " + std::to_string(d + 1) + " distinct points, got " +
                                   std::to_string(T.m))
                                      .c_str());
        case Error::Collinear: throw nb::value_error("Voronoi(): 'points' are collinear");
        case Error::Coplanar: throw nb::value_error("Voronoi(): 'points' are coplanar");
        case Error::TooMany: throw nb::value_error("Voronoi(): too many points");
        case Error::Memory:
            PyErr_SetString(
                PyExc_MemoryError, "Voronoi(): the triangulation is larger than this machine's physical memory"
            );
            throw nb::python_error();
    }
    nb::object out[8] = {
        wrap_array(std::move(copy)),
        wrap_array(std::move(O.vertices)),
        wrap_array(std::move(O.ridge_points)),
        wrap_array(std::move(O.ridge_indptr)),
        wrap_array(std::move(O.ridge_indices)),
        wrap_array(std::move(O.region_indptr)),
        wrap_array(std::move(O.region_indices)),
        wrap_array(std::move(O.point_region))
    };
    if (simplex_vertex)
        return nb::make_tuple(
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], wrap_array(std::move(sv))
        );
    return nb::make_tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
}

// The kernel on rows of simplices (tests): centres, the fast path's numerators (NaN when not certified or
// outside the range promise) and det, or with `wide` the exact path's correctly rounded N and D.
nb::tuple circumcenter_rows(nb::handle rh, bool wide) {
    if (!is_ndarray(rh)) throw nb::type_error("_circumcenter(): expected an sd.ndarray");
    const NDArray& r = as_ndarray(rh);
    if (r.ndim != 3 || r.dtype != DType::Float64 || !r.c_contiguous || (r.shape[2] != 2 && r.shape[2] != 3) ||
        r.shape[1] != r.shape[2] + 1)
        throw nb::type_error("_circumcenter(): wrong array layout");
    const i64 m = r.shape[0];
    const int d = static_cast<int>(r.shape[2]);
    const i64 vsh[] = {m, d}, dsh[] = {m};
    NDArray v = new_array(DType::Float64, vsh), N = new_array(DType::Float64, vsh), D = new_array(DType::Float64, dsh);
    const auto* x = reinterpret_cast<const double*>(r.data);
    auto* pv = reinterpret_cast<double*>(v.data);
    auto* pn = reinterpret_cast<double*>(N.data);
    auto* pd = reinterpret_cast<double*>(D.data);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (i64 i = 0; i < m; ++i) {
        const double* p[4];
        for (int k = 0; k <= d; ++k) p[k] = x + (i * (d + 1) + k) * d;
        Range rg;
        for (int k = 0; k <= d; ++k) rg.add(p[k], d);
        if (wide) {
            WideInt num[3], den;
            int e = 0;
            if (d == 2) centre_exact<2>(p, num, den, e);
            else centre_exact<3>(p, num, den, e);
            for (int k = 0; k < d; ++k) pn[i * d + k] = num[k].to_double(i64{d + 1} * e);
            pd[i] = den.to_double(i64{d} * e);
            if (d == 2) centre_wide<2>(p, pv + i * d);
            else centre_wide<3>(p, pv + i * d);
            continue;
        }
        bool ok = false;
        if (rg.safe(d + 1)) {
            ok = d == 2 ? numerators2(p[0], p[1], p[2], pn + i * d) : numerators3(p[0], p[1], p[2], p[3], pn + i * d);
            pd[i] = d == 2 ? det2(p[1], p[2], p[0]) : det3(p[0], p[1], p[2], p[3]);
        }
        if (!ok) {
            for (int k = 0; k < d; ++k) pn[i * d + k] = nan;
            pd[i] = nan;
        }
        if (d == 2) circumcentre<2>(p, pv + i * d, false);
        else circumcentre<3>(p, pv + i * d, false);
    }
    return nb::make_tuple(wrap(std::move(v)), wrap(std::move(N)), wrap(std::move(D)));
}

}  // namespace

void bind_voronoi(nb::module_& m) {
    m.def("_voronoi", &voronoi, nb::arg("points"), nb::arg("max_simplices") = -1, nb::arg("simplex_vertex") = false);
    m.def("_circumcenter", &circumcenter_rows, nb::arg("rows"), nb::arg("wide") = false);
}

}  // namespace sd::spatial
