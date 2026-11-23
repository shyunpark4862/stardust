// The scattered-data interpolants of sd.interpolate on sd.spatial's Delaunay triangulation
// (docs/design/interpolate.md §3.8; docs/design/spatial_geometry.md §3.8, §3.9; milestone IP9):
// LinearNDInterpolator (d = 2, 3) and CloughTocher2DInterpolator (d = 2).
//
//   locate      tri::locate_each (spatial/triangulation.hpp): find_simplex's exact walk, its blocks and its order;
//               each point is evaluated in the task that located it, while its simplex is in cache. Outside the
//               convex hull the value is fill_value; a point with a NaN coordinate is NaN in every column
//   linear      lambda_k = D_k / D with D the simplex's determinant and D_k the same with vertex k replaced by the
//               point (the value kernels det2 / det3 of spatial/predicates.hpp: within 1.002 u, exactly 0 when the
//               point is on the facet opposite k), then sum_k lambda_k f_k in vertex order: the first non-zero
//               weight's product, then one fma per term. A zero weight is skipped: at a data point the weight is
//               D / D = 1 (D_k is the same call as D) and the value comes out to the bit, and a value that is NaN or
//               inf off the point's face does not reach it. When D leaves [2^-900, 2^900] (points of subnormal size,
//               or near the largest doubles) the determinants are those of the points scaled by one power of two
//               (scaled_copies), whose ratios are the same
//   clough-     scipy's piecewise cubic (its _clough_tocher_2d_single): the triangle split at its centroid, the
//   tocher      Bezier coefficients from the vertex values and gradients, the inner ones from the C1 condition across
//               each edge in the direction of the neighbour's centroid, g_k = (c_(k+2) - c_k) / (3 c_k - 1) with c the
//               neighbour's centroid in this triangle's barycentric coordinates (no neighbour: g_k = -1/2). Here
//               c is not rounded through a centroid: with w the neighbour's vertex off the shared edge,
//                   g_k = (D + D_(k+2)(w) - D_k(w)) / (3 (D_k(w) - D)),
//               three value kernels on the data points (the denominator is a sum of two negative terms, so it is
//               relatively exact; scaled copies as above out of range). The point's barycentric coordinates are the
//               linear path's lambda; the sub-triangle is the one opposite the smallest (scipy's first minimum),
//               evaluated by de Casteljau's steps in a fixed order; at a data point (weights 1, 0, 0) the value is f
//               itself, -0.0 included. A triangle with a coordinate past 2^1000 takes its edges of the points times
//               2^-e and the gradients times 2^e (an edge could overflow)
//   gradients   scipy's global estimate (Nielson 1983, Renka-Cline 1984: the sum over the edges of the squared
//               curvature of the edge's cubic Hermite curve, made smaller vertex by vertex): Gauss-Seidel sweeps in
//               index order, each vertex solving its 2 x 2 system over its neighbours in increasing order; per
//               directed edge e and w = e / |e| / |e| / |e| are computed once, the vertex's Q = sum 4 e w^T is
//               inverted once and its data part sum 6 (f_i - f_j) w once per column. The column stops after the sweep
//               whose largest change max |y_new - y_old| is below tol times the largest gradient component max |y_new|
//               (a relative rule, so the sweeps do not depend on the units of the points or the values), or after
//               maxiter sweeps. Edges past 2^+-500 scale the points by a power of two first (edge_scale). The columns
//               (a complex value is two) are independent tasks; a column runs on one thread, so the gradients do not
//               depend on the thread count. A point that is not its representative (an equal point) gets its
//               representative's gradient
//
// Private bindings (the Python layer checks the arguments; `tri` is sd.spatial.Delaunay's native object; values
// (n, k) float64 or complex128, C-contiguous):
//   scattered_duplicate(tri, values) -> (i, j)    the smallest point j whose representative i (the smallest equal
//                                                 point) has other bytes in its values row, or (-1, -1)
//   linear_nd(tri, x, values, fill_re, fill_im) -> (q, k) of values' dtype
//   ct_gradients(tri, values, tol, maxiter, indptr, indices) -> (grad (n, k, 2) of values' dtype, sweeps, converged)
//       indptr, indices: the triangulation's vertex_neighbor_vertices (kept by the Delaunay once made);
//       sweeps: the most any column took; converged: every column stopped by tol
//   ct_eval(tri, x, values, grad, fill_re, fill_im) -> (q, k) of values' dtype

#include "interpolate/interpolate.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/delaunay2.hpp"
#include "spatial/predicates.hpp"
#include "spatial/triangulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

using spatial::DelaunayObj;
using spatial::tri::Triangulation;
namespace tri = spatial::tri;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const Triangulation& triangulation_of(nb::handle h) {
    try {
        return nb::cast<const DelaunayObj&>(h).T;
    } catch (const nb::cast_error&) { throw nb::type_error("'tri' must be the native Delaunay object"); }
}

// The values (n, k) as doubles: k2 = k (float64) or 2 k (complex128, the parts side by side).
struct Values {
    const double* f = nullptr;
    std::int64_t k = 0, k2 = 0;
    bool complex = false;
};

Values values_arg(nb::handle h, std::int64_t n) {
    const NDArray& a = require_ndarray(h, "values");
    if (a.ndim != 2 || !a.c_contiguous || (a.dtype != DType::Float64 && a.dtype != DType::Complex128))
        throw nb::type_error("'values' must be a C-contiguous 2-D float64 or complex128 array");
    if (a.shape[0] != n) throw nb::value_error("'values' must have one row per point");
    Values v;
    v.f = reinterpret_cast<const double*>(a.data);
    v.k = a.shape[1];
    v.complex = a.dtype == DType::Complex128;
    v.k2 = v.complex ? 2 * v.k : v.k;
    return v;
}

const double* points_arg(nb::handle h, int d, std::int64_t& q) {
    const NDArray& a = require_ndarray(h, "x");
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[1] != d)
        throw nb::type_error("'x' must be a C-contiguous (q, d) float64 array");
    q = a.shape[0];
    return reinterpret_cast<const double*>(a.data);
}

nb::object new_out(const Values& v, std::int64_t rows, std::int64_t cols3 = 0) {
    std::vector<std::int64_t> shape{rows, v.k};
    if (cols3) shape.push_back(cols3);
    return wrap(new_array(v.complex ? DType::Complex128 : DType::Float64, shape));
}

double* data_of(const nb::object& o) { return reinterpret_cast<double*>(as_ndarray(o).data); }

// The fill row: fill_re in every float64 column, (fill_re, fill_im) in every complex one.
std::vector<double> fill_row(const Values& v, double re, double im) {
    std::vector<double> out(static_cast<std::size_t>(v.k2));
    for (std::int64_t c = 0; c < v.k2; ++c) out[static_cast<std::size_t>(c)] = v.complex && (c & 1) ? im : re;
    return out;
}

// A point outside (t < 0): NaN in every column when a coordinate is NaN, else the fill row.
template <int D>
void outside(const double* p, const std::vector<double>& fill, double* row, std::int64_t k2) {
    bool nan = false;
    for (int j = 0; j < D; ++j) nan = nan || p[j] != p[j];
    if (nan) std::fill(row, row + k2, kNaN);
    else std::copy(fill.begin(), fill.end(), row);
}

// The determinant of the simplex q[0..D] (value kernels): the fast stages under the range promise (safe), else the
// wide path; with `any`, the range of the call's own points decides (scaled copies, below).
template <int D>
inline double det_of(const double* const* q, bool safe, bool any = false) {
    if constexpr (D == 2) {
        if (any) return spatial::det2_any(q[0], q[1], q[2]);
        return safe ? spatial::det2(q[0], q[1], q[2]) : spatial::det2_wide(q[0], q[1], q[2]);
    } else {
        if (any) return spatial::det3_any(q[0], q[1], q[2], q[3]);
        return safe ? spatial::det3(q[0], q[1], q[2], q[3]) : spatial::det3_wide(q[0], q[1], q[2], q[3]);
    }
}

// A determinant whose value keeps its relative accuracy, and whose cofactors can neither overflow nor vanish.
inline bool in_range(double v) {
    const double a = std::fabs(v);
    return a >= 0x1p-900 && a <= 0x1p900;
}

// Copies of m points scaled by one power of two s1 s2 = 2^-e (two factors: 2^-e may not be a double) that brings
// the largest coordinate near 1: for a computation whose determinants leave the normal range (points of
// subnormal size, or near the largest doubles). The scaling is exact unless a coordinate falls below the normal
// range, and a ratio of two determinants of the copies is the ratio of the originals'.
template <int D>
inline void scaled_copies(const double* const* p, int m, double (*out)[D]) {
    double big = 0.0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < D; ++j) big = std::max(big, std::fabs(p[i][j]));
    const int e = big > 0.0 ? std::ilogb(big) + 1 : 0;
    const double s1 = std::ldexp(1.0, -e / 2), s2 = std::ldexp(1.0, -e - (-e / 2));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < D; ++j) out[i][j] = p[i][j] * s1 * s2;
}

// lambda[k] = D_k / D for the point p in simplex t.
template <int D>
inline void barycentric(const Triangulation& T, std::int64_t t, const double* p, bool safe, double* lambda) {
    const std::int64_t* s = T.simplices + t * (D + 1);
    const double* q[D + 2];
    for (int j = 0; j <= D; ++j) q[j] = T.pts + s[j] * D;
    double det = det_of<D>(q, safe);
    bool any = false;
    double copies[D + 2][D];
    if (!in_range(det)) {  // inside the simplex |D_k| <= |D|: D alone decides
        q[D + 1] = p;
        scaled_copies<D>(q, D + 2, copies);
        for (int j = 0; j <= D; ++j) q[j] = copies[j];
        p = copies[D + 1];
        any = true;
        det = det_of<D>(q, safe, any);
    }
    for (int k = 0; k <= D; ++k) {
        const double* keep = q[k];
        q[k] = p;
        lambda[k] = det_of<D>(q, safe, any) / det;
        q[k] = keep;
    }
}

// ---------------------------------------------------------------- the linear interpolant

template <int D>
void linear_all(
    const Triangulation& T,
    const double* x,
    std::int64_t q,
    const Values& v,
    const std::vector<double>& fill,
    double* out,
    bool split
) {
    const std::int64_t k2 = v.k2;
    tri::locate_each<D>(T, x, q, split, [&](std::int64_t i, std::int64_t t) {
        const double* p = x + i * D;
        double* row = out + i * k2;
        if (t < 0) {
            outside<D>(p, fill, row, k2);
            return;
        }
        double lambda[D + 1];
        barycentric<D>(T, t, p, tri::detail::safe_with<D>(T, p), lambda);
        const std::int64_t* s = T.simplices + t * (D + 1);
        bool first = true;
        for (int k = 0; k <= D; ++k) {
            const double w = lambda[k];
            if (w == 0.0) continue;
            const double* f = v.f + s[k] * k2;
            if (first) {
                for (std::int64_t c = 0; c < k2; ++c) row[c] = w * f[c];
                first = false;
            } else {
                for (std::int64_t c = 0; c < k2; ++c) row[c] = std::fma(w, f[c], row[c]);
            }
        }
    });
}

// ---------------------------------------------------------------- Clough-Tocher: the cubic

// What one point needs of its triangle, column-independent: the edges, g_k, and the weights of its sub-triangle.
struct CTGeometry {
    double e12x, e12y, e23x, e23y, e31x, e31y;
    double g[3];
    double b[3];  // the sub-triangle's barycentric weights: the two vertices, then the centroid
    int sub;      // the vertex the sub-triangle leaves out (the smallest coordinate, the first one on a tie)
    double gs1 = 1.0, gs2 = 1.0;  // the gradients' factor when the edges are scaled (coordinates past 2^1000)
};

inline void ct_geometry(const Triangulation& T, std::int64_t t, const double* p, CTGeometry& G) {
    const std::int64_t* s = T.simplices + t * 3;
    const double* v[3] = {T.pts + s[0] * 2, T.pts + s[1] * 2, T.pts + s[2] * 2};
    const bool safe = tri::detail::safe_with<2>(T, p), safe_data = T.range.safe(2);
    double big = 0.0;
    for (int k = 0; k < 3; ++k) big = std::max(big, std::max(std::fabs(v[k][0]), std::fabs(v[k][1])));
    if (big <= 0x1p1000) {
        G.e12x = v[1][0] - v[0][0];
        G.e12y = v[1][1] - v[0][1];
        G.e23x = v[2][0] - v[1][0];
        G.e23y = v[2][1] - v[1][1];
        G.e31x = v[0][0] - v[2][0];
        G.e31y = v[0][1] - v[2][1];
    } else {  // an edge may overflow: the edges of the points times 2^-e and the gradients times 2^e (exact)
        const int e = std::ilogb(big) + 1;
        const double s = std::ldexp(1.0, -e);
        G.gs1 = std::ldexp(1.0, e / 2);
        G.gs2 = std::ldexp(1.0, e - e / 2);
        G.e12x = v[1][0] * s - v[0][0] * s;
        G.e12y = v[1][1] * s - v[0][1] * s;
        G.e23x = v[2][0] * s - v[1][0] * s;
        G.e23y = v[2][1] * s - v[1][1] * s;
        G.e31x = v[0][0] * s - v[2][0] * s;
        G.e31y = v[0][1] * s - v[2][1] * s;
    }
    const double* q0[3] = {v[0], v[1], v[2]};
    const double det0 = det_of<2>(q0, safe_data);
    for (int k = 0; k < 3; ++k) {
        const std::int64_t u = T.neighbors[t * 3 + k];
        if (u < 0) {
            G.g[k] = -0.5;
            continue;
        }
        int j = 0;
        while (T.neighbors[u * 3 + j] != t) ++j;
        const double* w = T.pts + T.simplices[u * 3 + j] * 2;
        const int k2 = (k + 2) % 3;
        const double* q[4] = {v[0], v[1], v[2], w};
        double det = det0, copies[4][2];
        bool any = false;
        auto dets = [&](double& dk, double& dk2) {
            const double* a = q[k];
            q[k] = q[3];
            dk = det_of<2>(q, safe_data, any);
            q[k] = a;
            const double* b = q[k2];
            q[k2] = q[3];
            dk2 = det_of<2>(q, safe_data, any);
            q[k2] = b;
        };
        double dk, dk2;
        dets(dk, dk2);
        if (!in_range(det) || !(std::fabs(dk) <= 0x1p900) || !(std::fabs(dk2) <= 0x1p900)) {
            scaled_copies<2>(q, 4, copies);
            for (int i = 0; i < 4; ++i) q[i] = copies[i];
            any = true;
            det = det_of<2>(q, safe_data, any);
            dets(dk, dk2);
        }
        G.g[k] = ((det + dk2) - dk) / (3.0 * (dk - det));
    }
    double lambda[3];
    barycentric<2>(T, t, p, safe, lambda);
    int m = 0;
    for (int k = 1; k < 3; ++k)
        if (lambda[k] < lambda[m]) m = k;
    const double low = lambda[m];
    G.sub = m;
    G.b[0] = lambda[(m + 1) % 3] - low;
    G.b[1] = lambda[(m + 2) % 3] - low;
    G.b[2] = 3.0 * low;
}

// One column: the value at the point from the vertex values f[3] and gradients df[3][2] (scipy's coefficients and
// their order), the cubic of the sub-triangle by de Casteljau.
inline double ct_value(const CTGeometry& G, const double* f, const double (*df)[2]) {
    const double df12 = +(df[0][0] * G.e12x + df[0][1] * G.e12y);
    const double df21 = -(df[1][0] * G.e12x + df[1][1] * G.e12y);
    const double df23 = +(df[1][0] * G.e23x + df[1][1] * G.e23y);
    const double df32 = -(df[2][0] * G.e23x + df[2][1] * G.e23y);
    const double df31 = +(df[2][0] * G.e31x + df[2][1] * G.e31y);
    const double df13 = -(df[0][0] * G.e31x + df[0][1] * G.e31y);
    // c[i][j][k][l]: the coefficient of b1^i b2^j b3^k b4^l (b4 the centroid's), named cijkl in scipy
    const double c3000 = f[0], c0300 = f[1], c0030 = f[2];
    const double c2100 = (df12 + 3 * c3000) / 3, c2010 = (df13 + 3 * c3000) / 3;
    const double c1200 = (df21 + 3 * c0300) / 3, c0210 = (df23 + 3 * c0300) / 3;
    const double c1020 = (df31 + 3 * c0030) / 3, c0120 = (df32 + 3 * c0030) / 3;
    const double c2001 = (c2100 + c2010 + c3000) / 3;
    const double c0201 = (c1200 + c0300 + c0210) / 3;
    const double c0021 = (c1020 + c0120 + c0030) / 3;
    const double c0111 =
        (G.g[0] * (-c0300 + 3 * c0210 - 3 * c0120 + c0030) + (-c0300 + 2 * c0210 - c0120 + c0021 + c0201)) / 2;
    const double c1011 =
        (G.g[1] * (-c0030 + 3 * c1020 - 3 * c2010 + c3000) + (-c0030 + 2 * c1020 - c2010 + c2001 + c0021)) / 2;
    const double c1101 =
        (G.g[2] * (-c3000 + 3 * c2100 - 3 * c1200 + c0300) + (-c3000 + 2 * c2100 - c1200 + c2001 + c0201)) / 2;
    const double c1002 = (c1101 + c1011 + c2001) / 3;
    const double c0102 = (c1101 + c0111 + c0201) / 3;
    const double c0012 = (c1011 + c0111 + c0021) / 3;
    const double c0003 = (c1002 + c0102 + c0012) / 3;
    // The sub-triangle (A, B, C) = (vertex sub + 1, vertex sub + 2, the centroid): its ten coefficients
    // a[i][j] (A^i B^j C^(3 - i - j)), in the order A^3, A^2 B, A B^2, B^3, A^2 C, A B C, B^2 C, A C^2, B C^2, C^3.
    double c[10];
    if (G.sub == 0) {  // A = vertex 2, B = vertex 3
        const double r[10] = {c0300, c0210, c0120, c0030, c0201, c0111, c0021, c0102, c0012, c0003};
        std::copy(r, r + 10, c);
    } else if (G.sub == 1) {  // A = vertex 3, B = vertex 1
        const double r[10] = {c0030, c1020, c2010, c3000, c0021, c1011, c2001, c0012, c1002, c0003};
        std::copy(r, r + 10, c);
    } else {  // A = vertex 1, B = vertex 2
        const double r[10] = {c3000, c2100, c1200, c0300, c2001, c1101, c0201, c1002, c0102, c0003};
        std::copy(r, r + 10, c);
    }
    const double a = G.b[0], b = G.b[1], z = G.b[2];
    if (z == 0.0 && (a == 0.0 || b == 0.0)) return a == 0.0 ? c[3] : c[0];  // a data point: its value, -0.0 too
    auto step = [&](double p, double qv, double r) { return std::fma(z, r, std::fma(b, qv, a * p)); };
    // level 1 (degree 2): AA, AB, BB, AC, BC, CC
    const double d0 = step(c[0], c[1], c[4]), d1 = step(c[1], c[2], c[5]), d2 = step(c[2], c[3], c[6]);
    const double d3 = step(c[4], c[5], c[7]), d4 = step(c[5], c[6], c[8]), d5 = step(c[7], c[8], c[9]);
    // level 2 (degree 1): A, B, C
    const double e0 = step(d0, d1, d3), e1 = step(d1, d2, d4), e2 = step(d3, d4, d5);
    return step(e0, e1, e2);
}

void ct_all(
    const Triangulation& T,
    const double* x,
    std::int64_t q,
    const Values& v,
    const double* grad,
    const std::vector<double>& fill,
    double* out,
    bool split
) {
    const std::int64_t k2 = v.k2;
    tri::locate_each<2>(T, x, q, split, [&](std::int64_t i, std::int64_t t) {
        const double* p = x + i * 2;
        double* row = out + i * k2;
        if (t < 0) {
            outside<2>(p, fill, row, k2);
            return;
        }
        CTGeometry G;
        ct_geometry(T, t, p, G);
        const std::int64_t* s = T.simplices + t * 3;
        for (std::int64_t c = 0; c < k2; ++c) {
            double f[3], df[3][2];
            for (int j = 0; j < 3; ++j) {
                f[j] = v.f[s[j] * k2 + c];
                // grad (n, k, 2) of the values' dtype: a complex column c is part c & 1 of value c >> 1
                for (int a = 0; a < 2; ++a) {
                    df[j][a] = v.complex ? grad[((s[j] * v.k + (c >> 1)) * 2 + a) * 2 + (c & 1)]
                                         : grad[(s[j] * v.k + c) * 2 + a];
                    if (G.gs1 != 1.0) df[j][a] = df[j][a] * G.gs1 * G.gs2;
                }
            }
            row[c] = ct_value(G, f, df);
        }
    });
}

// ---------------------------------------------------------------- Clough-Tocher: the gradients

struct EdgeData {
    const std::int64_t* indptr = nullptr;  // the triangulation's vertex_neighbor_vertices (indices increasing)
    const std::int64_t* idx = nullptr;
    std::int64_t nedges = 0;
    std::vector<double> ew;  // per directed edge (ex, ey, wx, wy), w = e / |e|^3, e in the scaled coordinates
    std::vector<double> iq;  // per vertex Q^-1 (q0, q1, q3)
    double s1 = 1.0, s2 = 1.0;  // the coordinates' power-of-two factor s1 s2 (the gradients come back times it)
};

// Powers of two s1 s2 = 2^-e (two factors: 2^-e may not be a double) for the coordinates when an edge length is
// past 2^+-500 (|e|^2 and w = e / |e|^3 would leave the doubles; the squared length overflowing or underflowing
// says so) or a difference overflows: e centres the edges' lengths in log scale (the points' extent when a
// difference overflows). An exact scaling in the range of normal numbers,
// so the estimate is the same function of the scaled points; points of ordinary extent keep 1, scipy's arithmetic.
void edge_scale(const Triangulation& T, EdgeData& E, bool split) {
    const std::int64_t n = T.n, nc = tri::chunks_of(n, tri::kChunk);
    std::vector<double> lo(static_cast<std::size_t>(nc), INFINITY), hi(static_cast<std::size_t>(nc), 0.0);
    tri::run_tasks(nc, split, [&](std::int64_t ch) {
        double a = INFINITY, b = 0.0;
        for (std::int64_t i = ch * tri::kChunk, e = std::min(n, (ch + 1) * tri::kChunk); i < e; ++i) {
            const double* pi = T.pts + i * 2;
            for (std::int64_t k = E.indptr[i]; k < E.indptr[i + 1]; ++k) {
                const double* pj = T.pts + E.idx[k] * 2;
                const double ex = pj[0] - pi[0], ey = pj[1] - pi[1];
                const double L2 = ex * ex + ey * ey;  // |e|^2: inf or below 2^-1000 sends to the scaled path
                a = std::min(a, L2);
                b = L2 > b || L2 != L2 ? L2 : b;
            }
        }
        lo[static_cast<std::size_t>(ch)] = a;
        hi[static_cast<std::size_t>(ch)] = b;
    });
    double a = INFINITY, b = 0.0;
    for (std::int64_t c = 0; c < nc; ++c) {
        a = std::min(a, lo[static_cast<std::size_t>(c)]);
        b = std::max(b, hi[static_cast<std::size_t>(c)]);
    }
    if (a >= 0x1p-1000 && b <= 0x1p1000) return;
    int e;
    if (std::isfinite(b) && a > 0.0) {
        e = (std::ilogb(a) + std::ilogb(b)) / 4;  // the squares' exponents: twice the lengths'
    } else {
        const spatial::tri::Locator<2>& L = T.loc2;
        double span = 0.0;
        for (int k = 0; k < 2; ++k) span = std::max(span, 0.5 * L.hi[k] - 0.5 * L.lo[k]);
        e = std::ilogb(span) + 1;
    }
    E.s1 = std::ldexp(1.0, -e / 2);
    E.s2 = std::ldexp(1.0, -e - (-e / 2));
}

EdgeData edge_data(const Triangulation& T, const std::int64_t* indptr, const std::int64_t* idx, bool split) {
    EdgeData E;
    const std::int64_t n = T.n;
    E.indptr = indptr;
    E.idx = idx;
    E.nedges = indptr[n];
    edge_scale(T, E, split);
    const double s1 = E.s1, s2 = E.s2;
    E.ew.assign(static_cast<std::size_t>(4 * E.nedges), 0.0);
    E.iq.assign(static_cast<std::size_t>(3 * n), 0.0);
    tri::run_tasks(tri::chunks_of(n, tri::kChunk), split, [&](std::int64_t ch) {
        for (std::int64_t i = ch * tri::kChunk, e = std::min(n, (ch + 1) * tri::kChunk); i < e; ++i) {
            const double* pi = T.pts + i * 2;
            double q0 = 0, q1 = 0, q3 = 0;
            for (std::int64_t a = E.indptr[i]; a < E.indptr[i + 1]; ++a) {
                const double* pj = T.pts + E.idx[a] * 2;
                const double ex = pj[0] * s1 * s2 - pi[0] * s1 * s2, ey = pj[1] * s1 * s2 - pi[1] * s1 * s2;
                const double r = 1.0 / std::sqrt(ex * ex + ey * ey);  // |e| in [2^-500, 2^500] here
                const double wx = ex * r * r * r, wy = ey * r * r * r;  // (e / |e|) / |e|^2: not |e|^3, out of range
                double* o = E.ew.data() + 4 * a;
                o[0] = ex, o[1] = ey, o[2] = wx, o[3] = wy;
                q0 += 4 * ex * wx;
                q1 += 4 * ex * wy;
                q3 += 4 * ey * wy;
            }
            const double det = q0 * q3 - q1 * q1;
            double* o = E.iq.data() + 3 * i;
            o[0] = q3 / det, o[1] = -q1 / det, o[2] = q0 / det;
        }
    });
    return E;
}

// One column (f with stride k2): the gradients y (n, 2), returns the sweeps it took and whether tol stopped it.
std::pair<std::int64_t, bool> ct_column(
    const Triangulation& T,
    const EdgeData& E,
    const double* f,
    std::int64_t k2,
    double tol,
    std::int64_t maxiter,
    std::vector<double>& y
) {
    const std::int64_t n = T.n;
    std::vector<double> cdata(static_cast<std::size_t>(2 * n), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        const double fi = f[i * k2];
        double c0 = 0, c1 = 0;
        for (std::int64_t a = E.indptr[i]; a < E.indptr[i + 1]; ++a) {
            const double* o = E.ew.data() + 4 * a;
            const double df = 6 * (fi - f[E.idx[a] * k2]);
            c0 = std::fma(df, o[2], c0);
            c1 = std::fma(df, o[3], c1);
        }
        cdata[static_cast<std::size_t>(2 * i)] = c0;
        cdata[static_cast<std::size_t>(2 * i + 1)] = c1;
    }
    y.assign(static_cast<std::size_t>(2 * n), 0.0);
    double* Y = y.data();
    for (std::int64_t it = 0; it < maxiter; ++it) {
        double change = 0.0, size = 0.0;  // the sweep's largest |y_new - y_old| and largest |y_new|
        for (std::int64_t i = 0; i < n; ++i) {
            if (T.rep[static_cast<std::size_t>(i)] != i) continue;
            double s0 = cdata[static_cast<std::size_t>(2 * i)], s1 = cdata[static_cast<std::size_t>(2 * i + 1)];
            for (std::int64_t a = E.indptr[i]; a < E.indptr[i + 1]; ++a) {
                const double* o = E.ew.data() + 4 * a;
                const double* yj = Y + 2 * E.idx[a];
                const double t = 2 * std::fma(o[1], yj[1], o[0] * yj[0]);
                s0 = std::fma(t, o[2], s0);
                s1 = std::fma(t, o[3], s1);
            }
            const double* iq = E.iq.data() + 3 * i;
            const double r0 = std::fma(iq[1], s1, iq[0] * s0), r1 = std::fma(iq[2], s1, iq[1] * s0);
            const double c = std::max(std::fabs(Y[2 * i] + r0), std::fabs(Y[2 * i + 1] + r1));
            const double z = std::max(std::fabs(r0), std::fabs(r1));
            Y[2 * i] = -r0;
            Y[2 * i + 1] = -r1;
            if (c > change || c != c) change = c;
            if (z > size || z != z) size = z;
        }
        if (change < tol * size) return {it + 1, true};
    }
    return {maxiter, false};
}

// ---------------------------------------------------------------- bindings

nb::tuple scattered_duplicate(nb::handle tri_h, nb::handle values_h) {
    const Triangulation& T = triangulation_of(tri_h);
    const NDArray& a = require_ndarray(values_h, "values");
    if (a.ndim != 2 || !a.c_contiguous || a.shape[0] != T.n)
        throw nb::type_error("'values' must be (n, k) C-contiguous");
    const std::size_t row = static_cast<std::size_t>(a.shape[1]) * static_cast<std::size_t>(a.itemsize());
    for (std::int64_t j = 0; j < T.n; ++j) {
        const std::int64_t r = T.rep[static_cast<std::size_t>(j)];
        if (r != j && row && std::memcmp(a.data + r * row, a.data + j * row, row) != 0) return nb::make_tuple(r, j);
    }
    return nb::make_tuple(-1, -1);
}

nb::object linear_nd(nb::handle tri_h, nb::handle x_h, nb::handle values_h, double fill_re, double fill_im) {
    const Triangulation& T = triangulation_of(tri_h);
    std::int64_t q = 0;
    const double* x = points_arg(x_h, T.d, q);
    const Values v = values_arg(values_h, T.n);
    nb::object out = new_out(v, q);
    if (q == 0 || v.k == 0) return out;
    const std::vector<double> fill = fill_row(v, fill_re, fill_im);
    const bool split = q * (tri::kQueryWork + 16 * (T.d + 2) + 2 * v.k2) >= tri::kParallelWork;
    double* o = data_of(out);
    const parallel::Unlocked unlocked(split);
    if (T.d == 2) linear_all<2>(T, x, q, v, fill, o, split);
    else linear_all<3>(T, x, q, v, fill, o, split);
    return out;
}

nb::tuple ct_gradients(
    nb::handle tri_h, nb::handle values_h, double tol, std::int64_t maxiter, nb::handle indptr_h, nb::handle indices_h
) {
    const Triangulation& T = triangulation_of(tri_h);
    if (T.d != 2) throw nb::value_error("ct_gradients: the triangulation must be 2-D");
    const Values v = values_arg(values_h, T.n);
    const NDArray& ip = require_ndarray(indptr_h, "indptr");
    const NDArray& ix = require_ndarray(indices_h, "indices");
    if (ip.ndim != 1 || ix.ndim != 1 || !ip.c_contiguous || !ix.c_contiguous || ip.dtype != DType::Int64 ||
        ix.dtype != DType::Int64 || ip.shape[0] != T.n + 1)
        throw nb::type_error("'indptr' and 'indices' must be the triangulation's vertex_neighbor_vertices");
    const auto* indptr = reinterpret_cast<const std::int64_t*>(ip.data);
    const auto* indices = reinterpret_cast<const std::int64_t*>(ix.data);
    nb::object grad = new_out(v, T.n, 2);
    double* g = data_of(grad);
    const std::int64_t n = T.n, k2 = v.k2;
    std::vector<std::int64_t> sweeps(static_cast<std::size_t>(k2), 0);
    std::vector<char> done(static_cast<std::size_t>(k2), 0);
    {
        // the work of the precomputation and of one sweep per column: the threshold of the array area
        const bool unlock = n * 64 * std::max<std::int64_t>(1, k2) >= tri::kParallelWork;
        const parallel::Unlocked unlocked(unlock);
        const EdgeData E = edge_data(T, indptr, indices, n > tri::kChunk && unlock);
        tri::run_tasks(k2, unlock && k2 > 1, [&](std::int64_t c) {
            std::vector<double> y;
            const auto r = ct_column(T, E, v.f + c, k2, tol, maxiter, y);
            sweeps[static_cast<std::size_t>(c)] = r.first;
            done[static_cast<std::size_t>(c)] = r.second;
            for (std::int64_t i = 0; i < n; ++i) {
                const std::int64_t src = T.rep[static_cast<std::size_t>(i)];
                for (int a = 0; a < 2; ++a) {
                    const double val = y[static_cast<std::size_t>(2 * src + a)] * E.s1 * E.s2;
                    if (v.complex) g[((i * v.k + (c >> 1)) * 2 + a) * 2 + (c & 1)] = val;
                    else g[(i * v.k + c) * 2 + a] = val;
                }
            }
        });
    }
    std::int64_t most = 0;
    bool all = true;
    for (std::int64_t c = 0; c < k2; ++c) {
        most = std::max(most, sweeps[static_cast<std::size_t>(c)]);
        all = all && done[static_cast<std::size_t>(c)];
    }
    return nb::make_tuple(grad, most, all);
}

nb::object ct_eval(
    nb::handle tri_h, nb::handle x_h, nb::handle values_h, nb::handle grad_h, double fill_re, double fill_im
) {
    const Triangulation& T = triangulation_of(tri_h);
    if (T.d != 2) throw nb::value_error("ct_eval: the triangulation must be 2-D");
    std::int64_t q = 0;
    const double* x = points_arg(x_h, 2, q);
    const Values v = values_arg(values_h, T.n);
    const NDArray& ga = require_ndarray(grad_h, "grad");
    if (ga.ndim != 3 || !ga.c_contiguous || ga.shape[0] != T.n || ga.shape[1] != v.k || ga.shape[2] != 2 ||
        ga.dtype != (v.complex ? DType::Complex128 : DType::Float64))
        throw nb::type_error("'grad' must be a C-contiguous (n, k, 2) array of the values' dtype");
    nb::object out = new_out(v, q);
    if (q == 0 || v.k == 0) return out;
    const std::vector<double> fill = fill_row(v, fill_re, fill_im);
    const bool split = q * (tri::kQueryWork + 160 + 40 * v.k2) >= tri::kParallelWork;
    double* o = data_of(out);
    const parallel::Unlocked unlocked(split);
    ct_all(T, x, q, v, reinterpret_cast<const double*>(ga.data), fill, o, split);
    return out;
}

}  // namespace

void bind_scattered(nb::module_& m) {
    m.def("scattered_duplicate", &scattered_duplicate, nb::arg("tri"), nb::arg("values"));
    m.def(
        "linear_nd", &linear_nd, nb::arg("tri"), nb::arg("x"), nb::arg("values"), nb::arg("fill_re"), nb::arg("fill_im")
    );
    m.def(
        "ct_gradients", &ct_gradients, nb::arg("tri"), nb::arg("values"), nb::arg("tol"), nb::arg("maxiter"),
        nb::arg("indptr"), nb::arg("indices")
    );
    m.def(
        "ct_eval", &ct_eval, nb::arg("tri"), nb::arg("x"), nb::arg("values"), nb::arg("grad"), nb::arg("fill_re"),
        nb::arg("fill_im")
    );
}

}  // namespace sd::interpolate
