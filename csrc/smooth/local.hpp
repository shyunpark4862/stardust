// The sorted-window local line of sd.smooth (docs/design/smooth.md §3.1; decisions SM27, SM36-SM38): lowess's
// kernel (smooth/local.cpp) and the one loess (milestone SM2) takes for one variable and degree 1, so that the two
// give the same bits.
//
// Data: x sorted ascending (stable, ties ordered by y, the weight and the index: decision SM36), y, and the prior
// weights pw (the robust weight times the user's weight, >= 0). At an evaluation point x0:
//
//   window   the k nearest points [l, l + k): l is the first left end with l = n - k or
//            x[l + k] - x0 >= x0 - x[l] (a binary search; the predicate is monotone in l because a rounded
//            difference is monotone in its operands), the radius h = max(x0 - x[l], x[l + k - 1] - x0)
//   weights  v_j = (1 - |d_j|^3)^3 pw_j for |x_j - x0| < h, 0 otherwise, d_j = (x_j - x0) / h computed as
//            ((x_j - x0) sc) inv with inv = 1 / (h sc); sc = 2^1000 when h < 2^-900 (subnormal spacing: 1 / h
//            would overflow; the product by a power of two is exact), else 1. The factor 1 - |d|^3 is computed as
//            ((h - |x_j - x0|) sc inv) ((1 + |d|) + |d|^2): no cancellation at the window's edge, where
//            1 - |d|^3 would lose the relative accuracy that the edge points' weights need (decision SM38). The
//            gap h - |x_j - x0| is taken from the window's end on x_j's side, (x_r - x_j) + g_r or (x_j - x_l) + g_l,
//            with g the exact radius less that end's exact distance (each distance a rounded difference plus its
//            rounding error, once per window; decision SM111): a rounded x_j - x0 would move the gap of a point
//            1e-8 inside a radius of 2.5 by 1e-8 of itself
//   sums     a = sum v, b = sum v d, c = sum v d^2, s0 = sum v y, s1 = sum v d y over [f, g], the first and the last
//            point of positive weight, in NEON lanes paired by the range's length alone (two lane pairs over four
//            points, then a pair, then the last point): the sums are a function of the data and x0, whichever of
//            two points at distance h the window took (their weight is 0)
//   value    (c s0 - b s1) / (a c - b^2) when kappa_1 = a c / (a c - b^2) <= 2^20; otherwise the recentred form
//            s0 / a - dbar s1' / c' with dbar = b / a, c' = sum v (d - dbar)^2, s1' = sum v (d - dbar) y (a second
//            pass, no cancellation; SM27). Every positive weight at one x: the local mean s0 / a. h = 0 (the
//            window is k copies of x0): the weighted mean of every point at x0, v_j = pw_j. Fewer than two
//            positive weights: no fit (Few; the caller gives y_i at a data point, NaN elsewhere, as statsmodels)
//   hat row  l_j = v_j (P + Q d_j): P = c / det, Q = -b / det (closed form), P = 1 / a + dbar^2 / c',
//            Q = -dbar / c' (recentred), P = 1 / a, Q = 0 (a mean)
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sd::smooth {

struct Sorted1D {
    const double* x;
    const double* y;
    const double* pw;  // prior weights >= 0
    std::int64_t n;
};

enum class LineKind : std::int8_t { Line, Mean, Few, Nan };

// The window's ends and their gaps (scaled by sc): the gap h - |x - x0| of a point is (xr - x) sc + gr on the
// right of x0 and (x - xl) sc + gl on the left (decision SM111).
struct Edge {
    double xl = 0, xr = 0, gl = 0, gr = 0;
};

// a - b = s + e exactly (Knuth's two-sum of a and -b).
inline void two_diff(double a, double b, double& s, double& e) {
    s = a - b;
    const double bv = s - a;
    const double av = s - bv;
    e = (a - av) + (-b - bv);
}

// The ends xl <= x0 <= xr of a window and its radius: exact (the larger end distance, widen = 1) or h (sc: the
// scale of the gaps; hs = h sc).
inline Edge window_edge(double xl, double xr, double x0, double sc, double hs, bool exact) {
    double sl, el, sr, er;
    two_diff(x0, xl, sl, el);
    two_diff(xr, x0, sr, er);
    Edge E{xl, xr, 0.0, 0.0};
    if (exact) {
        if (sr > sl || (sr == sl && er >= el)) E.gl = ((sr - sl) + (er - el)) * sc;
        else E.gr = ((sl - sr) + (el - er)) * sc;
    } else {
        E.gl = (hs - sl * sc) - el * sc;
        E.gr = (hs - sr * sc) - er * sc;
    }
    return E;
}

struct LineFit {
    double value = std::numeric_limits<double>::quiet_NaN();
    double x0 = 0, h = 0, sc = 1, hs = 0, inv = 0;  // hs = h sc, inv = 1 / hs
    Edge edge;
    double P = 0, Q = 0;
    std::int64_t lo = 0, hi = 0;  // the points [lo, hi) that may carry a positive weight
    std::int64_t self = -1;       // the data point fitted (sorted index), -1 for a new point
    LineKind kind = LineKind::Few;
};

inline constexpr double kRecentre = 0x1p20;  // kappa_1 above which the recentred pass is taken (SM27)
inline constexpr double kTinyRadius = 0x1p-900;
inline constexpr double kRadiusScale = 0x1p1000;

// The left end of the k nearest points of x0 (1 <= k <= n).
inline std::int64_t window_left(const double* x, std::int64_t n, std::int64_t k, double x0) {
    std::int64_t lo = 0, hi = n - k;
    while (lo < hi) {
        const std::int64_t mid = lo + (hi - lo) / 2;
        if (x[mid + k] - x0 >= x0 - x[mid]) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// The tricube factor (1 - |d|^3)^3 of an inside point (|x - x0| < h), the same operations as the lanes of
// line_moments (a gap below 0, a point whose exact distance is not inside, gives 0).
inline double tricube_of(double x, double x0, double sc, double inv, const Edge& E) {
    const double diff = x - x0;
    const double ad = std::fabs(diff * sc) * inv;
    const double gap = std::max(diff >= 0 ? (E.xr - x) * sc + E.gr : (x - E.xl) * sc + E.gl, 0.0);
    const double t = (gap * inv) * ((1.0 + ad) + ad * ad);
    return t * t * t;
}

// v_j of the fit F.
inline double window_weight(const Sorted1D& D, const LineFit& F, std::int64_t j) {
    if (F.h == 0) return D.x[j] == F.x0 ? D.pw[j] : 0.0;
    if (!(std::fabs(D.x[j] - F.x0) < F.h)) return 0.0;
    return tricube_of(D.x[j], F.x0, F.sc, F.inv, F.edge) * D.pw[j];
}

struct Moments {
    double a, b, c, s0, s1;
};

template <bool Scaled>
inline Moments line_moments(
    const Sorted1D& D, std::int64_t lo, std::int64_t hi, double x0, double sc, double inv, const Edge& E
) {
    const float64x2_t vx0 = vdupq_n_f64(x0), vsc = vdupq_n_f64(sc), vinv = vdupq_n_f64(inv);
    const float64x2_t vxl = vdupq_n_f64(E.xl), vxr = vdupq_n_f64(E.xr), vgl = vdupq_n_f64(E.gl),
                      vgr = vdupq_n_f64(E.gr);
    const float64x2_t one = vdupq_n_f64(1.0), zero = vdupq_n_f64(0.0);
    float64x2_t a0 = zero, b0 = zero, c0 = zero, s00 = zero, s10 = zero;
    float64x2_t a1 = zero, b1 = zero, c1 = zero, s01 = zero, s11 = zero;
    const auto step = [&](std::int64_t j, float64x2_t& a, float64x2_t& b, float64x2_t& c, float64x2_t& s0,
                          float64x2_t& s1) {
        const float64x2_t xx = vld1q_f64(D.x + j);
        const float64x2_t raw = vsubq_f64(xx, vx0);
        float64x2_t diff = raw, right = vsubq_f64(vxr, xx), left = vsubq_f64(xx, vxl);
        if constexpr (Scaled) {
            diff = vmulq_f64(diff, vsc);
            right = vmulq_f64(right, vsc), left = vmulq_f64(left, vsc);
        }
        const float64x2_t d = vmulq_f64(diff, vinv);
        const float64x2_t ad = vabsq_f64(d);
        const float64x2_t gap =
            vmaxq_f64(vbslq_f64(vcgezq_f64(raw), vaddq_f64(right, vgr), vaddq_f64(left, vgl)), zero);
        const float64x2_t edge = vmulq_f64(gap, vinv);
        const float64x2_t t = vmulq_f64(edge, vaddq_f64(vaddq_f64(one, ad), vmulq_f64(ad, ad)));
        const float64x2_t v = vmulq_f64(vmulq_f64(vmulq_f64(t, t), t), vld1q_f64(D.pw + j));
        const float64x2_t vd = vmulq_f64(v, d);
        const float64x2_t yy = vld1q_f64(D.y + j);
        a = vaddq_f64(a, v);
        b = vaddq_f64(b, vd);
        c = vaddq_f64(c, vmulq_f64(vd, d));
        s0 = vaddq_f64(s0, vmulq_f64(v, yy));
        s1 = vaddq_f64(s1, vmulq_f64(vd, yy));
    };
    std::int64_t j = lo;
    for (; j + 4 <= hi; j += 4) {
        step(j, a0, b0, c0, s00, s10);
        step(j + 2, a1, b1, c1, s01, s11);
    }
    if (j + 2 <= hi) {
        step(j, a0, b0, c0, s00, s10);
        j += 2;
    }
    a0 = vaddq_f64(a0, a1), b0 = vaddq_f64(b0, b1), c0 = vaddq_f64(c0, c1);
    s00 = vaddq_f64(s00, s01), s10 = vaddq_f64(s10, s11);
    Moments m{
        vgetq_lane_f64(a0, 0) + vgetq_lane_f64(a0, 1), vgetq_lane_f64(b0, 0) + vgetq_lane_f64(b0, 1),
        vgetq_lane_f64(c0, 0) + vgetq_lane_f64(c0, 1), vgetq_lane_f64(s00, 0) + vgetq_lane_f64(s00, 1),
        vgetq_lane_f64(s10, 0) + vgetq_lane_f64(s10, 1)
    };
    if (j < hi) {  // the last point: the lanes' operations
        const double xx = D.x[j], raw = xx - x0;
        double diff = raw, right = E.xr - xx, left = xx - E.xl;
        if constexpr (Scaled) diff = diff * sc, right = right * sc, left = left * sc;
        const double d = diff * inv;
        const double ad = std::fabs(d);
        const double gap = std::max(raw >= 0 ? right + E.gr : left + E.gl, 0.0);
        const double t = (gap * inv) * ((1.0 + ad) + ad * ad);
        const double v = t * t * t * D.pw[j];
        const double vd = v * d;
        m.a += v, m.b += vd, m.c += vd * d, m.s0 += v * D.y[j], m.s1 += vd * D.y[j];
    }
    return m;
}

// The local line at x0 from its k nearest points (decision SM27). self: the sorted index of the data point fitted,
// or -1. A Few fit leaves the value NaN (the caller decides). widen: a factor on the radius (loess's span > 1 with
// k = n: sqrt(span) times the farthest distance, decision SM29); 1 leaves every operation as lowess's.
inline LineFit fit_line(const Sorted1D& D, std::int64_t k, double x0, std::int64_t self = -1, double widen = 1.0) {
    LineFit F;
    F.x0 = x0, F.self = self;
    const double* x = D.x;
    const std::int64_t n = D.n;
    if (!(x0 == x0)) {
        F.kind = LineKind::Nan;
        return F;
    }
    const std::int64_t l = window_left(x, n, k, x0), r = l + k;
    const double hfar = std::max(x0 - x[l], x[r - 1] - x0);
    const double h = widen != 1.0 ? hfar * widen : hfar;
    F.h = h;
    if (!(h < std::numeric_limits<double>::infinity())) {  // a distance overflowed
        F.kind = LineKind::Nan;
        return F;
    }
    if (h == 0) {  // k copies of x0: the weighted mean of every point at x0
        std::int64_t t0 = l, t1 = r;
        while (t0 > 0 && x[t0 - 1] == x0) --t0;
        while (t1 < n && x[t1] == x0) ++t1;
        F.lo = t0, F.hi = t1;
        double a = 0, s0 = 0;
        std::int64_t npos = 0;
        for (std::int64_t j = t0; j < t1; ++j) {
            a += D.pw[j], s0 += D.pw[j] * D.y[j];
            npos += D.pw[j] > 0;
        }
        if (npos < 2) return F;
        F.kind = LineKind::Mean, F.value = s0 / a, F.P = 1.0 / a;
        return F;
    }
    F.sc = h < kTinyRadius ? kRadiusScale : 1.0;
    F.hs = widen != 1.0 ? (hfar * F.sc) * widen : h * F.sc;  // widened after the scaling: no subnormal rounding
    F.inv = 1.0 / F.hs;
    F.edge = window_edge(x[l], x[r - 1], x0, F.sc, F.hs, widen == 1.0);
    std::int64_t lo = l, hi = r;  // the points strictly inside the radius
    while (lo < hi && !(std::fabs(x[lo] - x0) < h)) ++lo;
    while (hi > lo && !(std::fabs(x[hi - 1] - x0) < h)) --hi;
    std::int64_t f = lo;
    while (f < hi && window_weight(D, F, f) == 0) ++f;
    if (f == hi) return F;
    std::int64_t g = hi - 1;
    while (g > f && window_weight(D, F, g) == 0) --g;
    if (g == f) return F;
    F.lo = f, F.hi = g + 1;
    const Moments m = F.sc == 1.0 ? line_moments<false>(D, f, g + 1, x0, 1.0, F.inv, F.edge)
                                  : line_moments<true>(D, f, g + 1, x0, F.sc, F.inv, F.edge);
    if (x[f] == x[g]) {  // every positive weight at one x
        F.kind = LineKind::Mean, F.value = m.s0 / m.a, F.P = 1.0 / m.a;
        return F;
    }
    const double ac = m.a * m.c, det = ac - m.b * m.b;
    if (det > 0 && ac <= kRecentre * det) {
        F.kind = LineKind::Line;
        F.value = (m.c * m.s0 - m.b * m.s1) / det;
        F.P = m.c / det, F.Q = -m.b / det;
        return F;
    }
    // recentred: d - dbar has no cancellation
    const double dbar = m.b / m.a;
    double cc = 0, s1 = 0;
    for (std::int64_t j = f; j <= g; ++j) {
        const double v = window_weight(D, F, j);
        const double e = ((x[j] - x0) * F.sc) * F.inv - dbar;
        cc += v * e * e, s1 += v * e * D.y[j];
    }
    if (!(cc > 0)) {
        F.kind = LineKind::Mean, F.value = m.s0 / m.a, F.P = 1.0 / m.a;
        return F;
    }
    F.kind = LineKind::Line;
    F.value = m.s0 / m.a - dbar * (s1 / cc);
    F.P = 1.0 / m.a + dbar * dbar / cc, F.Q = -dbar / cc;
    return F;
}

// l_j of the fit F: the weight of data point j in the fitted value (sorted index).
inline double hat_at(const Sorted1D& D, const LineFit& F, std::int64_t j) {
    switch (F.kind) {
        case LineKind::Nan: return std::numeric_limits<double>::quiet_NaN();
        case LineKind::Few: return j == F.self ? 1.0 : 0.0;
        default: break;
    }
    if (j < F.lo || j >= F.hi) return 0.0;
    const double v = window_weight(D, F, j);
    if (v == 0) return 0.0;
    if (F.kind == LineKind::Mean) return v * F.P;
    return v * (F.P + F.Q * (((D.x[j] - F.x0) * F.sc) * F.inv));
}

// sum_j l_j^2 / w_j over the hat row (w null: the weights are 1; a point of weight 0 has l_j = 0): the variance of
// the fitted value in units of sigma^2 when Var y_j = sigma^2 / w_j.
inline double hat_norm2(const Sorted1D& D, const LineFit& F, const double* w) {
    if (F.kind == LineKind::Nan || F.kind == LineKind::Few) return std::numeric_limits<double>::quiet_NaN();
    double s = 0;
    for (std::int64_t j = F.lo; j < F.hi; ++j) {
        const double l = hat_at(D, F, j);
        if (l == 0) continue;
        s += w ? l * l / w[j] : l * l;
    }
    return s;
}

}  // namespace sd::smooth
