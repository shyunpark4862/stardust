// Exact geometric predicates and the determinant value kernels (docs/design/spatial_geometry.md §3.1,
// §3.2; decisions SPT90, SPT95, SPT100, SPT112-SPT116).
//
//   orient2d(a, b, c)        sign of det[a - c; b - c]: +1 when a, b, c turn counterclockwise
//   orient3d(a, b, c, d)     sign of det[b - a, c - a, d - a] (the positive tetrahedron of §2.2)
//   incircle(a, b, c, d)     +1 when d is strictly inside the circle through a, b, c, for a, b, c
//                            counterclockwise (the sign flips with orient2d(a, b, c)); 0 on the circle
//   insphere(a, b, c, d, e)  +1 when e is strictly inside the sphere through a, b, c, d, for
//                            orient3d(a, b, c, d) > 0; 0 on the sphere
//   incircle_sos, insphere_sos   the same with the tie rule of §3.2: never 0 for distinct points whose
//                            first d + 1 are not degenerate
//   det2(a, b, c), det3(a, b, c, d)   the VALUES of the two orientation determinants
//
// Every sign is the sign of the determinant of the inputs read as rationals, for every finite float64.
//
// THREE STAGES (SPT112)
//   A. filter       the determinant of the rounded differences in float64 against K * (its permanent).
//   B. expansion    when every difference a - c is exact (its two_diff tail is 0): the determinant of
//                   those doubles as an expansion (expansion.hpp). Grids, integer data and clustered
//                   points (Sterbenz) land here; the general case does not.
//   W. wide         integers (wideint.hpp): any finite input.
// A and B need the range promise below; W needs nothing.
//
// THE RANGE PROMISE (Range::safe(q), q the degree: 2 orient2d, 3 orient3d, 4 incircle, 5 insphere).
// Let 2^hi bound the coordinates (|x| < 2^(hi + 1)) and 2^lo divide them all. If
//     q (hi + 2) + 8 <= 1023   and   q lo >= -1074 (or lo >= 0)
// then in stages A and B (i) nothing overflows: a difference is below 2^(hi + 2), a product of at most q
// of them below 2^(q (hi + 2)), and a sum of at most 72 such products (insphere: 4 lifts of 3 squares times
// 6 products; every partial sum, every permanent and every expansion component is bounded by that sum of
// magnitudes) below 2^(q (hi + 2) + 7), with one more bit for the roundings; (ii) nothing underflows: every
// exact intermediate value is a multiple of 2^(k lo) for some k <= q, hence of 2^-1074, and so is its
// rounding (the float grid at its size is a coarser power of two, or it is representable). A product that
// is a multiple of 2^-1074 and below 2^-1022 is representable, so
//     fl(x op y) = (x op y)(1 + delta), |delta| <= u = 2^-53,   and   |fl(z) - z| <= u |fl(z)|        (M)
// hold for every operation, and two_prod's error term is representable (exact).
// Predicates built from a point set's Range take the plain functions (orient2d, ...) when the set is
// safe for that degree; otherwise the *_any functions, which check the points of the one call (SPT113).
//
// THE FILTER CONSTANTS (SPT114). Lemma. Evaluate a polynomial of differences by a tree of float64
// operations and give every node a count c: a difference fl(a - b) has c = 1, a sum or difference of two
// nodes c = max(c_l, c_r) + 1, a product c = c_l + c_r + 1. Let S be the node's exact "permanent": |d| for
// a difference, S_l + S_r for a sum or difference, S_l S_r for a product. Under (M) the computed value is
//     x^ = sum over the monomials m_j of m_j prod_(i <= c)(1 + delta_ji),  so  |x^ - x| <= ((1 + u)^c - 1) S.
// (Induction: a rounded sum multiplies every monomial of both sides by one more factor; a rounded product
// pairs every monomial of one side with every monomial of the other, c_l + c_r factors, and one more.)
// The same tree with |.| at the products' leaves and + for every - computes P^ >= (1 - u)^(c_P) S. With the
// last operation a sum (|x^ - (l^ +- r^)| <= u |x^| by (M)) and theta_k = (1 + u)^k - 1:
//     |x^ - x| <= u |x^| + theta_(c - 1) S,  and the sign of x^ is the sign of x when  (1 - u)|x^| > theta_(c - 1) S,
// which |x^| > fl(K P^) >= K P^ (1 - u) gives as soon as  K >= theta_(c - 1) / (1 - u)^(c_P + 2).
// (When K P^ is below 2^-1022, fl(K P^) may lose more than u relatively; but x^ and fl(K P^) are multiples
// of 2^-1074, so |x^| > fl(K P^) means |x^| >= fl(K P^) + 2^-1074 > K P^, and the conclusion stands.)
// The counts of the evaluation orders below (every operation is written out; -ffp-contract=off):
//     orient2d   products 3, det 4;                                   permanent 4    K >= 3u + 21.0 u^2
//     orient3d   2 x 2 minors 4, times z 6, two sums 7, 8;             permanent 8    K >= 7u + 91.0 u^2
//     incircle   lifts 4, minors 4, products 9, two sums 10, 11;       permanent 11   K >= 10u + 175.0 u^2
//     insphere   lifts 5, 2 x 2 minors 4, 3 x 3 minors 8, products 14,
//                pair differences 15, sum 16;                          permanent 17   K >= 15u + 390.1 u^2
// The constants are the next (k + m u) u with m a multiple of 16; each is a float64 exactly, and
// tests/spatial/test_predicates.py checks the inequality in rational arithmetic against the compiled values.
// (Shewchuk's (3 + 16u)u, (7 + 56u)u, (10 + 96u)u, (16 + 224u)u are for his evaluation order and a finer
// count; ours differ in the second-order term and, for insphere, in the leading 15.)
//
// THE VALUE KERNELS (SPT115). det2 and det3 return r with |r - x| <= 1.002 u |x| + 2^-1075 (0 exactly when
// x = 0, +-inf when x overflows); the design's contract is 2u. Fast path (range promise for q = 2, 3): the
// differences as exact pairs h + t (two_diff), the head determinant by error-free products and sums, the
// first-order tail terms and all rounding errors of the head part in plain float64:
//   det2: x = s + C + N with (p, e) = two_prod(h1, h2), (q, f) = two_prod(h3, h4), (s, g) = two_diff(p, q),
//     C = (e - f + g) + (h1 t2 + t1 h2 - h3 t4 - t3 h4), N = t1 t2 - t3 t4. With Pm = |h1 h2| + |h3 h4| and
//     |t| <= u |h|: the terms of C sum to at most (4u + u^2) Pm in magnitude and pass at most 4 roundings,
//     so |c^ - C| <= 17 u^2 Pm; |N| <= u^2 Pm; r = fl(s + c^) has |r - x| <= u |r| + 18 u^2 Pm.
//   det3 (rows A, B, C, x = sum of z_i M_i over the three 2 x 2 minors M_i of the x, y columns, Pi the
//     permanent of |H|): M_i = s_i + (e_i - f_i + g_i) as in det2 and m_i = fl(fl(e_i - f_i) + g_i) is
//     within 4.1 u^2 mu_i; (P_i, E_i) = two_prod(z_i, s_i), l_i = fl(z_i m_i) within 6.2 u^2 |z_i| mu_i;
//     P_1 + P_2 + P_3 = S + g_a + g_b by two two_sums; the nine first-order tail terms t_ij cof_ij(H), each
//     with at most 3 roundings, are within 9.3 u^2 Pi in total; the terms with two or three tails are below
//     3.1 u^2 Pi; the float sum of the 17 low terms (magnitudes summing to at most 8.4 u Pi, at most 16
//     additions) loses at most 136 u^2 Pi. So r = fl(S + low) has |r - x| <= u |r| + 160 u^2 Pi.
//   Both accept r when |r| >= 2^-36 P^ (P^ the float permanent, >= (1 - u)^5 of the exact one): then the
//   second-order part is below 0.002 u |r| and |r - x| <= 1.002 u |x|. Otherwise (a determinant 2^36
//   times smaller than its terms, or an input outside the range) the wide path computes the determinant
//   exactly and rounds it to nearest.
//
// THE TIE RULE (SPT95, SPT116). The lift height of the k-th largest point in lexicographic order (x, then
// y, then z) gets + eps^k. The lifted determinant is linear in the heights, so with value 0 the sign is
// the sign of the first non-zero cofactor, taking the d + 2 points from the lexicographically largest
// down: for the point at argument position i (0-based)
//     incircle_sos:  (-1)^i      * orient2d(the other three, in argument order)
//     insphere_sos:  (-1)^(i+1)  * orient3d(the other four, in argument order).
// Only the order among the d + 2 points matters, so the predicate needs no global rank. For distinct
// points with a non-degenerate first simplex the result is never 0 (the last point's cofactor is that
// simplex's orientation). tests/spatial/exactgeom.py computes the same determinant as a polynomial in eps.
#pragma once

#include "spatial/expansion.hpp"
#include "spatial/wideint.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sd::spatial {

inline constexpr double kU = 0x1p-53;
inline constexpr double kOrient2dFilter = (3.0 + 32.0 * kU) * kU;
inline constexpr double kOrient3dFilter = (7.0 + 96.0 * kU) * kU;
inline constexpr double kIncircleFilter = (10.0 + 176.0 * kU) * kU;
inline constexpr double kInsphereFilter = (15.0 + 400.0 * kU) * kU;
inline constexpr double kValueAccept = 0x1p-36;

// ---------------------------------------------------------------- the range promise

struct Range {
    int hi = INT_MIN;  // the largest floor(log2 |x|) of the non-zero coordinates
    int lo = INT_MAX;  // the smallest exponent of a lowest set bit

    void add(double x) {
        if (x == 0.0) return;
        const FloatParts p = float_parts(x);
        const int h = p.e + 63 - std::countl_zero(p.m), l = p.e + std::countr_zero(p.m);
        if (h > hi) hi = h;
        if (l < lo) lo = l;
    }
    void add(const double* x, std::int64_t n) {
        for (std::int64_t i = 0; i < n; ++i) add(x[i]);
    }
    // True when stages A and B are exact for a predicate of degree q on coordinates in this range.
    bool safe(int q) const {
        if (hi == INT_MIN) return true;
        return q * (hi + 2) + 8 <= 1023 && (lo >= 0 || q * lo >= -1074);
    }
};

// ---------------------------------------------------------------- the determinants, once for B and W

// T is double (stage B; products are expansions) or WideInt.
template <class T>
inline auto orient2d_det(const T& acx, const T& acy, const T& bcx, const T& bcy) {
    return sub(mul(acx, bcy), mul(acy, bcx));
}

template <class T>
inline auto orient3d_det(
    const T& ax, const T& ay, const T& az, const T& bx, const T& by, const T& bz, const T& cx, const T& cy, const T& cz
) {
    return add(
        add(mul(sub(mul(cx, by), mul(bx, cy)), az), mul(sub(mul(ax, cy), mul(cx, ay)), bz)),
        mul(sub(mul(bx, ay), mul(ax, by)), cz)
    );
}

template <class T>
inline auto incircle_det(const T& ax, const T& ay, const T& bx, const T& by, const T& cx, const T& cy) {
    const auto al = add(mul(ax, ax), mul(ay, ay));
    const auto bl = add(mul(bx, bx), mul(by, by));
    const auto cl = add(mul(cx, cx), mul(cy, cy));
    return add(
        add(mul(sub(mul(bx, cy), mul(cx, by)), al), mul(sub(mul(cx, ay), mul(ax, cy)), bl)),
        mul(sub(mul(ax, by), mul(bx, ay)), cl)
    );
}

template <class T>
inline auto insphere_det(
    const T& ax,
    const T& ay,
    const T& az,
    const T& bx,
    const T& by,
    const T& bz,
    const T& cx,
    const T& cy,
    const T& cz,
    const T& dx,
    const T& dy,
    const T& dz
) {
    const auto ab = sub(mul(ax, by), mul(bx, ay));
    const auto bc = sub(mul(bx, cy), mul(cx, by));
    const auto cd = sub(mul(cx, dy), mul(dx, cy));
    const auto da = sub(mul(dx, ay), mul(ax, dy));
    const auto ac = sub(mul(ax, cy), mul(cx, ay));
    const auto bd = sub(mul(bx, dy), mul(dx, by));
    const auto abc = add(sub(mul(bc, az), mul(ac, bz)), mul(ab, cz));
    const auto bcd = add(sub(mul(cd, bz), mul(bd, cz)), mul(bc, dz));
    const auto cda = add(add(mul(da, cz), mul(ac, dz)), mul(cd, az));
    const auto dab = add(add(mul(ab, dz), mul(bd, az)), mul(da, bz));
    const auto al = add(add(mul(ax, ax), mul(ay, ay)), mul(az, az));
    const auto bl = add(add(mul(bx, bx), mul(by, by)), mul(bz, bz));
    const auto cl = add(add(mul(cx, cx), mul(cy, cy)), mul(cz, cz));
    const auto dl = add(add(mul(dx, dx), mul(dy, dy)), mul(dz, dz));
    return add(sub(mul(dab, cl), mul(abc, dl)), sub(mul(bcd, al), mul(cda, bl)));
}

namespace detail {

// The differences p[i] - p[np - 1] of np points in d dimensions as integers (coordinates divided by
// 2^e, e their lowest set bit). Returns false when every coordinate is 0; *e_out is e.
template <int NP, int D>
inline bool wide_differences(const double* const* p, WideInt* out, int* e_out = nullptr) {
    Range r;
    for (int i = 0; i < NP; ++i) r.add(p[i], D);
    if (r.hi == INT_MIN) return false;
    WideInt last[D];
    for (int k = 0; k < D; ++k) last[k] = WideInt::scaled(p[NP - 1][k], r.lo);
    for (int i = 0; i + 1 < NP; ++i)
        for (int k = 0; k < D; ++k) out[i * D + k] = sub(WideInt::scaled(p[i][k], r.lo), last[k]);
    if (e_out) *e_out = r.lo;
    return true;
}

template <int NP, int D>
inline bool points_safe(const double* const* p, int q) {
    Range r;
    for (int i = 0; i < NP; ++i) r.add(p[i], D);
    return r.safe(q);
}

}  // namespace detail

// ---------------------------------------------------------------- orient2d

// Stage A: 1 or -1, or 2 when the filter does not decide.
inline int orient2d_filter(const double* a, const double* b, const double* c) {
    const double acx = a[0] - c[0], bcx = b[0] - c[0];
    const double acy = a[1] - c[1], bcy = b[1] - c[1];
    const double l = acx * bcy, r = acy * bcx;
    const double det = l - r;
    const double bound = kOrient2dFilter * (std::fabs(l) + std::fabs(r));
    if (det > bound) return 1;
    if (-det > bound) return -1;
    return 2;
}

// Stage B: the sign, or 2 when a difference is not exact (or the sign was not certified).
inline int orient2d_expansion(const double* a, const double* b, const double* c) {
    double acx, acy, bcx, bcy, t0, t1, t2, t3;
    two_diff(a[0], c[0], acx, t0);
    two_diff(a[1], c[1], acy, t1);
    two_diff(b[0], c[0], bcx, t2);
    two_diff(b[1], c[1], bcy, t3);
    if (t0 != 0.0 || t1 != 0.0 || t2 != 0.0 || t3 != 0.0) return 2;
    auto det = orient2d_det(acx, acy, bcx, bcy);
    return sign(det);
}

inline int orient2d_wide(const double* a, const double* b, const double* c) {
    const double* p[3] = {a, b, c};
    WideInt v[4];
    if (!detail::wide_differences<3, 2>(p, v)) return 0;
    return orient2d_det(v[0], v[1], v[2], v[3]).sign();
}

[[gnu::noinline]] inline int orient2d_slow(const double* a, const double* b, const double* c) {
    const int s = orient2d_expansion(a, b, c);
    return s != 2 ? s : orient2d_wide(a, b, c);
}

// The points satisfy Range::safe(2).
inline int orient2d(const double* a, const double* b, const double* c) {
    const int s = orient2d_filter(a, b, c);
    return s != 2 ? s : orient2d_slow(a, b, c);
}

// Any finite points.
inline int orient2d_any(const double* a, const double* b, const double* c) {
    const double* p[3] = {a, b, c};
    return detail::points_safe<3, 2>(p, 2) ? orient2d(a, b, c) : orient2d_wide(a, b, c);
}

// ---------------------------------------------------------------- orient3d

inline int orient3d_filter(const double* a, const double* b, const double* c, const double* d) {
    const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
    const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
    const double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
    const double cxby = cdx * bdy, bxcy = bdx * cdy;
    const double axcy = adx * cdy, cxay = cdx * ady;
    const double bxay = bdx * ady, axby = adx * bdy;
    const double det = (adz * (cxby - bxcy) + bdz * (axcy - cxay)) + cdz * (bxay - axby);
    const double perm =
        (std::fabs(adz) * (std::fabs(cxby) + std::fabs(bxcy)) + std::fabs(bdz) * (std::fabs(axcy) + std::fabs(cxay))) +
        std::fabs(cdz) * (std::fabs(bxay) + std::fabs(axby));
    const double bound = kOrient3dFilter * perm;
    if (det > bound) return 1;
    if (-det > bound) return -1;
    return 2;
}

inline int orient3d_expansion(const double* a, const double* b, const double* c, const double* d) {
    const double* p[3] = {a, b, c};
    double h[9], tail = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            double t;
            two_diff(p[i][k], d[k], h[3 * i + k], t);
            tail += std::fabs(t);
        }
    if (tail != 0.0) return 2;
    auto det = orient3d_det(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    return sign(det);
}

inline int orient3d_wide(const double* a, const double* b, const double* c, const double* d) {
    const double* p[4] = {a, b, c, d};
    WideInt v[9];
    if (!detail::wide_differences<4, 3>(p, v)) return 0;
    return orient3d_det(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]).sign();
}

[[gnu::noinline]] inline int orient3d_slow(const double* a, const double* b, const double* c, const double* d) {
    const int s = orient3d_expansion(a, b, c, d);
    return s != 2 ? s : orient3d_wide(a, b, c, d);
}

inline int orient3d(const double* a, const double* b, const double* c, const double* d) {
    const int s = orient3d_filter(a, b, c, d);
    return s != 2 ? s : orient3d_slow(a, b, c, d);
}

inline int orient3d_any(const double* a, const double* b, const double* c, const double* d) {
    const double* p[4] = {a, b, c, d};
    return detail::points_safe<4, 3>(p, 3) ? orient3d(a, b, c, d) : orient3d_wide(a, b, c, d);
}

// ---------------------------------------------------------------- incircle

inline int incircle_filter(const double* a, const double* b, const double* c, const double* d) {
    const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
    const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
    const double bxcy = bdx * cdy, cxby = cdx * bdy;
    const double cxay = cdx * ady, axcy = adx * cdy;
    const double axby = adx * bdy, bxay = bdx * ady;
    const double al = adx * adx + ady * ady;
    const double bl = bdx * bdx + bdy * bdy;
    const double cl = cdx * cdx + cdy * cdy;
    const double det = (al * (bxcy - cxby) + bl * (cxay - axcy)) + cl * (axby - bxay);
    const double perm = (al * (std::fabs(bxcy) + std::fabs(cxby)) + bl * (std::fabs(cxay) + std::fabs(axcy))) +
                        cl * (std::fabs(axby) + std::fabs(bxay));
    const double bound = kIncircleFilter * perm;
    if (det > bound) return 1;
    if (-det > bound) return -1;
    return 2;
}

inline int incircle_expansion(const double* a, const double* b, const double* c, const double* d) {
    const double* p[3] = {a, b, c};
    double h[6], tail = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 2; ++k) {
            double t;
            two_diff(p[i][k], d[k], h[2 * i + k], t);
            tail += std::fabs(t);
        }
    if (tail != 0.0) return 2;
    auto det = incircle_det(h[0], h[1], h[2], h[3], h[4], h[5]);
    return sign(det);
}

inline int incircle_wide(const double* a, const double* b, const double* c, const double* d) {
    const double* p[4] = {a, b, c, d};
    WideInt v[6];
    if (!detail::wide_differences<4, 2>(p, v)) return 0;
    return incircle_det(v[0], v[1], v[2], v[3], v[4], v[5]).sign();
}

[[gnu::noinline]] inline int incircle_slow(const double* a, const double* b, const double* c, const double* d) {
    const int s = incircle_expansion(a, b, c, d);
    return s != 2 ? s : incircle_wide(a, b, c, d);
}

// The points satisfy Range::safe(4).
inline int incircle(const double* a, const double* b, const double* c, const double* d) {
    const int s = incircle_filter(a, b, c, d);
    return s != 2 ? s : incircle_slow(a, b, c, d);
}

inline int incircle_any(const double* a, const double* b, const double* c, const double* d) {
    const double* p[4] = {a, b, c, d};
    return detail::points_safe<4, 2>(p, 4) ? incircle(a, b, c, d) : incircle_wide(a, b, c, d);
}

// ---------------------------------------------------------------- insphere

inline int insphere_filter(const double* a, const double* b, const double* c, const double* d, const double* e) {
    const double aex = a[0] - e[0], bex = b[0] - e[0], cex = c[0] - e[0], dex = d[0] - e[0];
    const double aey = a[1] - e[1], bey = b[1] - e[1], cey = c[1] - e[1], dey = d[1] - e[1];
    const double aez = a[2] - e[2], bez = b[2] - e[2], cez = c[2] - e[2], dez = d[2] - e[2];
    const double axby = aex * bey, bxay = bex * aey;
    const double bxcy = bex * cey, cxby = cex * bey;
    const double cxdy = cex * dey, dxcy = dex * cey;
    const double dxay = dex * aey, axdy = aex * dey;
    const double axcy = aex * cey, cxay = cex * aey;
    const double bxdy = bex * dey, dxby = dex * bey;
    const double ab = axby - bxay, bc = bxcy - cxby, cd = cxdy - dxcy;
    const double da = dxay - axdy, ac = axcy - cxay, bd = bxdy - dxby;
    const double abc = (aez * bc - bez * ac) + cez * ab;
    const double bcd = (bez * cd - cez * bd) + dez * bc;
    const double cda = (cez * da + dez * ac) + aez * cd;
    const double dab = (dez * ab + aez * bd) + bez * da;
    const double al = (aex * aex + aey * aey) + aez * aez;
    const double bl = (bex * bex + bey * bey) + bez * bez;
    const double cl = (cex * cex + cey * cey) + cez * cez;
    const double dl = (dex * dex + dey * dey) + dez * dez;
    const double det = (cl * dab - dl * abc) + (al * bcd - bl * cda);
    const double pab = std::fabs(axby) + std::fabs(bxay), pbc = std::fabs(bxcy) + std::fabs(cxby);
    const double pcd = std::fabs(cxdy) + std::fabs(dxcy), pda = std::fabs(dxay) + std::fabs(axdy);
    const double pac = std::fabs(axcy) + std::fabs(cxay), pbd = std::fabs(bxdy) + std::fabs(dxby);
    const double fa = std::fabs(aez), fb = std::fabs(bez), fc = std::fabs(cez), fd = std::fabs(dez);
    const double pabc = (fa * pbc + fb * pac) + fc * pab;
    const double pbcd = (fb * pcd + fc * pbd) + fd * pbc;
    const double pcda = (fc * pda + fd * pac) + fa * pcd;
    const double pdab = (fd * pab + fa * pbd) + fb * pda;
    const double perm = ((cl * pdab + dl * pabc) + al * pbcd) + bl * pcda;
    const double bound = kInsphereFilter * perm;
    if (det > bound) return 1;
    if (-det > bound) return -1;
    return 2;
}

inline int insphere_expansion(const double* a, const double* b, const double* c, const double* d, const double* e) {
    const double* p[4] = {a, b, c, d};
    double h[12], tail = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) {
            double t;
            two_diff(p[i][k], e[k], h[3 * i + k], t);
            tail += std::fabs(t);
        }
    if (tail != 0.0) return 2;
    auto det = insphere_det(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11]);
    return sign(det);
}

inline int insphere_wide(const double* a, const double* b, const double* c, const double* d, const double* e) {
    const double* p[5] = {a, b, c, d, e};
    WideInt v[12];
    if (!detail::wide_differences<5, 3>(p, v)) return 0;
    return insphere_det(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]).sign();
}

[[gnu::noinline]] inline int insphere_slow(
    const double* a, const double* b, const double* c, const double* d, const double* e
) {
    const int s = insphere_expansion(a, b, c, d, e);
    return s != 2 ? s : insphere_wide(a, b, c, d, e);
}

// The points satisfy Range::safe(5).
inline int insphere(const double* a, const double* b, const double* c, const double* d, const double* e) {
    const int s = insphere_filter(a, b, c, d, e);
    return s != 2 ? s : insphere_slow(a, b, c, d, e);
}

inline int insphere_any(const double* a, const double* b, const double* c, const double* d, const double* e) {
    const double* p[5] = {a, b, c, d, e};
    return detail::points_safe<5, 3>(p, 5) ? insphere(a, b, c, d, e) : insphere_wide(a, b, c, d, e);
}

// ---------------------------------------------------------------- the tie rule (§3.2)

// p > q in lexicographic order of the coordinates (-0.0 is 0.0).
inline bool lex_greater(const double* p, const double* q, int d) {
    for (int k = 0; k < d; ++k)
        if (p[k] != q[k]) return p[k] > q[k];
    return false;
}

namespace detail {

// The positions 0..n-1 of p sorted from the lexicographically largest point down (equal points keep
// their argument order).
template <int NP, int D>
inline void descending(const double* const* p, int* idx) {
    for (int i = 0; i < NP; ++i) {
        int j = i;
        while (j > 0 && lex_greater(p[i], p[idx[j - 1]], D)) {
            idx[j] = idx[j - 1];
            --j;
        }
        idx[j] = i;
    }
}

template <class In, class Orient>
inline int incircle_tie(const double* a, const double* b, const double* c, const double* d, In in, Orient orient) {
    const int s = in(a, b, c, d);
    if (s != 0) return s;
    const double* p[4] = {a, b, c, d};
    int idx[4];
    descending<4, 2>(p, idx);
    for (int k = 0; k < 4; ++k) {
        const int i = idx[k];
        const double* q[3];
        for (int j = 0, m = 0; j < 4; ++j)
            if (j != i) q[m++] = p[j];
        const int o = orient(q[0], q[1], q[2]);
        if (o != 0) return (i & 1) ? -o : o;
    }
    return 0;
}

template <class In, class Orient>
inline int insphere_tie(
    const double* a, const double* b, const double* c, const double* d, const double* e, In in, Orient orient
) {
    const int s = in(a, b, c, d, e);
    if (s != 0) return s;
    const double* p[5] = {a, b, c, d, e};
    int idx[5];
    descending<5, 3>(p, idx);
    for (int k = 0; k < 5; ++k) {
        const int i = idx[k];
        const double* q[4];
        for (int j = 0, m = 0; j < 5; ++j)
            if (j != i) q[m++] = p[j];
        const int o = orient(q[0], q[1], q[2], q[3]);
        if (o != 0) return (i & 1) ? o : -o;
    }
    return 0;
}

}  // namespace detail

// The points satisfy Range::safe(4) / safe(5).
inline int incircle_sos(const double* a, const double* b, const double* c, const double* d) {
    return detail::incircle_tie(a, b, c, d, incircle, orient2d);
}
inline int insphere_sos(const double* a, const double* b, const double* c, const double* d, const double* e) {
    return detail::insphere_tie(a, b, c, d, e, insphere, orient3d);
}
inline int incircle_sos_any(const double* a, const double* b, const double* c, const double* d) {
    return detail::incircle_tie(a, b, c, d, incircle_any, orient2d_any);
}
inline int insphere_sos_any(const double* a, const double* b, const double* c, const double* d, const double* e) {
    return detail::insphere_tie(a, b, c, d, e, insphere_any, orient3d_any);
}

// ---------------------------------------------------------------- the predicates of one point set

// Chooses the path once from the range of a point set's coordinates (and, for find_simplex, of the query
// point: build another Predicates from the merged Range, or call the *_any functions).
class Predicates {
public:
    Predicates() = default;
    explicit Predicates(const Range& r) : s2_(r.safe(2)), s3_(r.safe(3)), s4_(r.safe(4)), s5_(r.safe(5)) {}

    int orient2d(const double* a, const double* b, const double* c) const {
        return s2_ ? spatial::orient2d(a, b, c) : orient2d_any(a, b, c);
    }
    int orient3d(const double* a, const double* b, const double* c, const double* d) const {
        return s3_ ? spatial::orient3d(a, b, c, d) : orient3d_any(a, b, c, d);
    }
    int incircle(const double* a, const double* b, const double* c, const double* d) const {
        return s4_ ? spatial::incircle(a, b, c, d) : incircle_any(a, b, c, d);
    }
    int insphere(const double* a, const double* b, const double* c, const double* d, const double* e) const {
        return s5_ ? spatial::insphere(a, b, c, d, e) : insphere_any(a, b, c, d, e);
    }
    int incircle_sos(const double* a, const double* b, const double* c, const double* d) const {
        return s4_ ? spatial::incircle_sos(a, b, c, d) : incircle_sos_any(a, b, c, d);
    }
    int insphere_sos(const double* a, const double* b, const double* c, const double* d, const double* e) const {
        return s5_ ? spatial::insphere_sos(a, b, c, d, e) : insphere_sos_any(a, b, c, d, e);
    }

private:
    bool s2_ = false, s3_ = false, s4_ = false, s5_ = false;
};

// ---------------------------------------------------------------- the value kernels

// det[a - c; b - c] exactly, rounded to nearest.
inline double det2_wide(const double* a, const double* b, const double* c) {
    const double* p[3] = {a, b, c};
    WideInt v[4];
    int e = 0;
    if (!detail::wide_differences<3, 2>(p, v, &e)) return 0.0;
    return orient2d_det(v[0], v[1], v[2], v[3]).to_double(2 * std::int64_t{e});
}

// The fast path (points safe for q = 2): the value, or NaN when it is not certified.
inline double det2_fast(const double* a, const double* b, const double* c) {
    double h1, h2, h3, h4, t1, t2, t3, t4;
    two_diff(a[0], c[0], h1, t1);
    two_diff(b[1], c[1], h2, t2);
    two_diff(a[1], c[1], h3, t3);
    two_diff(b[0], c[0], h4, t4);
    double p, e, q, f, s, g;
    two_prod(h1, h2, p, e);
    two_prod(h3, h4, q, f);
    two_diff(p, q, s, g);
    const double low = ((e - f) + g) + ((h1 * t2 + t1 * h2) - (h3 * t4 + t3 * h4));
    const double r = s + low;
    const double perm = std::fabs(p) + std::fabs(q);
    if (perm == 0.0) return 0.0;
    if (std::fabs(r) >= kValueAccept * perm) return r;
    return std::numeric_limits<double>::quiet_NaN();
}

inline double det2(const double* a, const double* b, const double* c) {
    const double r = det2_fast(a, b, c);
    return r == r ? r : det2_wide(a, b, c);
}

inline double det2_any(const double* a, const double* b, const double* c) {
    const double* p[3] = {a, b, c};
    return detail::points_safe<3, 2>(p, 2) ? det2(a, b, c) : det2_wide(a, b, c);
}

// det[b - a, c - a, d - a] exactly, rounded to nearest.
inline double det3_wide(const double* a, const double* b, const double* c, const double* d) {
    const double* p[4] = {a, b, c, d};
    WideInt v[9];
    int e = 0;
    if (!detail::wide_differences<4, 3>(p, v, &e)) return 0.0;
    return orient3d_det(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]).to_double(3 * std::int64_t{e});
}

namespace detail {

// u1 v1 - u2 v2 = s + (a float64 within 4.1 u^2 of the rest); *perm is fl(|u1 v1| + |u2 v2|).
inline void minor2(double u1, double v1, double u2, double v2, double& s, double& low, double& perm) {
    double p, e, q, f, g;
    two_prod(u1, v1, p, e);
    two_prod(u2, v2, q, f);
    two_diff(p, q, s, g);
    low = (e - f) + g;
    perm = std::fabs(p) + std::fabs(q);
}

}  // namespace detail

// The fast path (points safe for q = 3): the value, or NaN when it is not certified.
inline double det3_fast(const double* a, const double* b, const double* c, const double* d) {
    double ax, ay, az, bx, by, bz, cx, cy, cz;
    double tax, tay, taz, tbx, tby, tbz, tcx, tcy, tcz;
    two_diff(a[0], d[0], ax, tax);
    two_diff(a[1], d[1], ay, tay);
    two_diff(a[2], d[2], az, taz);
    two_diff(b[0], d[0], bx, tbx);
    two_diff(b[1], d[1], by, tby);
    two_diff(b[2], d[2], bz, tbz);
    two_diff(c[0], d[0], cx, tcx);
    two_diff(c[1], d[1], cy, tcy);
    two_diff(c[2], d[2], cz, tcz);
    double sa, la, pa, sb, lb, pb, sc, lc, pc;
    detail::minor2(cx, by, bx, cy, sa, la, pa);  // times az
    detail::minor2(ax, cy, cx, ay, sb, lb, pb);  // times bz
    detail::minor2(bx, ay, ax, by, sc, lc, pc);  // times cz
    double P1, E1, P2, E2, P3, E3, S1, ga, S, gb;
    two_prod(az, sa, P1, E1);
    two_prod(bz, sb, P2, E2);
    two_prod(cz, sc, P3, E3);
    two_sum(P1, P2, S1, ga);
    two_sum(S1, P3, S, gb);
    double low = ((ga + gb) + ((E1 + E2) + E3)) + ((az * la + bz * lb) + cz * lc);
    const double tails =
        (((std::fabs(tax) + std::fabs(tay)) + std::fabs(taz)) + ((std::fabs(tbx) + std::fabs(tby)) + std::fabs(tbz))) +
        ((std::fabs(tcx) + std::fabs(tcy)) + std::fabs(tcz));
    if (tails != 0.0) {
        double t = taz * sa;
        t += tbz * sb;
        t += tcz * sc;
        t += tax * (bz * cy - cz * by);
        t += tay * (cz * bx - bz * cx);
        t += tbx * (cz * ay - az * cy);
        t += tby * (az * cx - cz * ax);
        t += tcx * (az * by - bz * ay);
        t += tcy * (bz * ax - az * bx);
        low += t;
    }
    const double r = S + low;
    const double perm = (std::fabs(az) * pa + std::fabs(bz) * pb) + std::fabs(cz) * pc;
    if (perm == 0.0) return 0.0;
    if (std::fabs(r) >= kValueAccept * perm) return r;
    return std::numeric_limits<double>::quiet_NaN();
}

inline double det3(const double* a, const double* b, const double* c, const double* d) {
    const double r = det3_fast(a, b, c, d);
    return r == r ? r : det3_wide(a, b, c, d);
}

inline double det3_any(const double* a, const double* b, const double* c, const double* d) {
    const double* p[4] = {a, b, c, d};
    return detail::points_safe<4, 3>(p, 3) ? det3(a, b, c, d) : det3_wide(a, b, c, d);
}

}  // namespace sd::spatial
