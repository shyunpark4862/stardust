// The B-splines of sd.interpolate (docs/design/interpolate.md §2.5, §3.4, §6; milestone IP3): evaluation (de Boor),
// the knot rules and the collocation systems of make_interp_spline, the derivative, the antiderivative, knot
// insertion and the two conversions with the power basis. Knots t (N = n + k + 1, nondecreasing, finite),
// coefficients c (rows, D) row-major with rows >= n (only the first n count; complex coefficients are (rows, 2 ncol)
// doubles: every operation here is linear in the coefficients with real factors).
//
//   interval     the l in [k, n - 1] with t[l] <= x < t[l + 1], x >= t[n] giving n - 1 and x < t[k] giving k
//                (scipy's walk, found here by base/interval.hpp's count of the knots at or below x, so random
//                points cost O(log N) each: scipy walks from the previous point's interval, O(n) per point)
//   basis        the k + 1 functions B_{l-k} .. B_l of degree k at x, or their derivatives of order nu: scipy's
//                recursion (_deBoor_D) with the weights as two quotients (t[l+m] - x) / (t[l+m] - t[l+m-j]) and
//                (x - t[l+m-j]) / (...), each in [0, 1] inside the base interval, so every step is a convex
//                combination and no step overflows (a reciprocal of a subnormal knot spacing would); the
//                derivative steps are scipy's, w = j h / (t[l+m] - t[l+m-j])
//   value        c[l - k] B_{l-k} first, then one fma per further term, in the order of the coefficients
//
// The interval and the basis are interpolate/deboor.hpp's, which the grid's tensor-product splines (grid.cpp) share.
//
// The accuracy contracts (docs/design/interpolate.md §4): evaluation within gamma_{6k+1} sum_j |c_j| B~_j(x)
// plus a floor of subnormal roundings, B~ the recursion run on the absolute values of its terms (B itself inside
// the base interval without derivatives); construction componentwise within 2 gamma_{8k+8} |A^-1| |A| |c^|.
//
// Private bindings (the Python layer checks the arguments):
//   check_knots(t) -> 0 finite and nondecreasing, -1 not finite, -2 decreasing somewhere
//   bspline_eval(t, c, k, x, nu, rule) -> (q, ncol)          rule 0 NaN outside [t[k], t[n]], 1 extend, 2 periodic
//   bspline_basis(t, k, x, extend) -> (data, indices, indptr) design_matrix's CSR arrays (rows of k + 1 entries)
//   bspline_knots(x, k, kind) -> t                           kind 0 not-a-knot, 1 the nodes with k more at each
//                                                            end (derivative conditions), 2 periodic (scipy's rules)
//   bspline_interp(x, y, t, k, lo, lv, ro, rv, pivot) -> (c, status)  the collocation solve; lo/ro the derivative
//                                                            orders in row order, lv/rv their values (rows, D);
//                                                            pivot: partial pivoting at once; status 0 no
//                                                            pivoting, 1 partial pivoting, -1 singular, -2 a
//                                                            derivative condition's row overflows
//   bspline_periodic(x, y, t, k) -> (c, status)              the periodic solve (n - 1 coefficients, repeated)
//   bspline_derivative(t, c, k, nu) -> (t2, c2) or None      None: a zero knot spacing (repeated internal knots)
//   bspline_antiderivative(t, c, k, nu) -> (t2, c2)
//   bspline_insert(t, c, k, x, m, periodic) -> (t2, c2) or None   None: too few internal knots (periodic)
//   bspline_to_ppoly(t, c, k) -> (cp, breaks)                the distinct knots of [t[k], t[n]] and the Taylor
//                                                            coefficients there (highest degree first), from the
//                                                            piece's coefficients differenced d times
//   bspline_from_power(x, cp, t, k) -> c                     Marsden's identity (blossoms) from a CubicSpline

#include "interpolate/interpolate.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "base/interval.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "interpolate/banded.hpp"
#include "interpolate/deboor.hpp"
#include "interpolate/search.hpp"
#include "parallel/pool.hpp"

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

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// fn(begin, end) over [0, total) in pieces of at least `grain`, on the pool when the work is large enough; the
// pieces depend on the sizes only.
template <typename F>
void split_range(std::int64_t total, std::int64_t work, std::int64_t grain, const F& fn) {
    if (total <= 0) return;
    if (work >= kParallelThreshold && parallel::available() && total > grain) {
        const std::int64_t per_unit = std::max<std::int64_t>(1, work / total);
        const std::int64_t per = std::max(grain, kParallelTask * 4 / per_unit);
        const std::int64_t tasks = (total + per - 1) / per;
        if (tasks > 1) {
            parallel::parallel_for(
                tasks, [&](std::int64_t t) { fn(t * per, std::min(total, (t + 1) * per)); }, work_ns(work / tasks)
            );
            return;
        }
    }
    const parallel::Unlocked unlocked(work >= kParallelThreshold);
    fn(0, total);
}

// ---------------------------------------------------------------- arguments

struct Knots {
    const double* t;
    std::int64_t N, k, n;  // N knots, degree k, n = N - k - 1 basis functions
};

Knots knots_arg(nb::handle o, std::int64_t k) {
    const NDArray& a = require_ndarray(o, "t");
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error("'t' must be a C-contiguous 1-D float64 array");
    if (k < 0 || a.shape[0] < 2 * k + 2) throw nb::value_error("'t' must have 2 k + 2 knots or more");
    return {reinterpret_cast<const double*>(a.data), a.shape[0], k, a.shape[0] - k - 1};
}

struct Coefs {
    const double* c;
    std::int64_t rows, ncol, D;  // D doubles per row
    DType dtype;
};

Coefs coefs_arg(nb::handle o, std::int64_t need) {
    const NDArray& a = require_ndarray(o, "c");
    if (a.ndim != 2 || !a.c_contiguous || a.shape[0] < need)
        throw nb::value_error("'c' must be a C-contiguous (rows, ncol) array with a row per basis function");
    if (a.dtype != DType::Float64 && a.dtype != DType::Complex128)
        throw nb::type_error("'c' must be float64 or complex128");
    const std::int64_t per = a.dtype == DType::Complex128 ? 2 : 1;
    return {reinterpret_cast<const double*>(a.data), a.shape[0], a.shape[1], a.shape[1] * per, a.dtype};
}

const NDArray& vector_arg(nb::handle o, const char* what) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous 1-D float64 array").c_str());
    return a;
}

NDArray new_values(DType dtype, std::int64_t rows, std::int64_t ncol, bool zeroed = false) {
    const std::int64_t shape[2] = {rows, ncol};
    return new_array(dtype, shape, zeroed);
}

NDArray new_vector(std::int64_t n) {
    const std::int64_t shape[1] = {n};
    return new_array(DType::Float64, shape);
}

// 0: finite and nondecreasing; -1 not finite; -2 decreasing somewhere
int check_knots(nb::handle o) {
    const NDArray& a = vector_arg(o, "t");
    const auto* t = reinterpret_cast<const double*>(a.data);
    const std::int64_t n = a.shape[0];
    for (std::int64_t i = 0; i < n; ++i)
        if (!std::isfinite(t[i])) return -1;
    for (std::int64_t i = 1; i < n; ++i)
        if (t[i] < t[i - 1]) return -2;
    return 0;
}

// ---------------------------------------------------------------- the interval and the basis

inline std::int64_t knot_interval_of(const Knots& T, double x) {
    std::int64_t c;
    count_le_lanes<1>(T.t, T.N, &x, &c);
    return knot_interval(c, T.k, T.n);
}

// sum_r c[l - k + r][col] h[r]: the first product, then one fma per term.
inline double combine(const Coefs& C, std::int64_t l, std::int64_t k, std::int64_t col, const double* h) {
    const double* p = C.c + (l - k) * C.D + col;
    double acc = p[0] * h[0];
    for (std::int64_t r = 1; r <= k; ++r) acc = std::fma(p[r * C.D], h[r], acc);
    return acc;
}

// ---------------------------------------------------------------- evaluation

// K: the degree when it is a compile-time constant (nu = 0), else -1
template <int K>
void eval_range(
    const Knots& T,
    const Coefs& C,
    const double* xq,
    double* out,
    std::int64_t nu,
    Extrapolate rule,
    std::int64_t begin,
    std::int64_t end
) {
    constexpr std::int64_t kBlock = 256;
    double v[kBlock];
    std::int64_t cnt[kBlock];
    std::vector<double> work(2 * (T.k + 1));
    double* h = work.data();
    double* hh = h + T.k + 1;
    GallopState state;
    const double lo = T.t[T.k], hi = T.t[T.n];
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t mm = std::min(kBlock, end - b0);
        for (std::int64_t i = 0; i < mm; ++i) {
            const double p = xq[b0 + i];
            v[i] = rule == Extrapolate::Periodic ? wrap_periodic(p, lo, hi) : p;
        }
        count_le_block(T.t, T.N, v, mm, cnt, state);
        for (std::int64_t i = 0; i < mm; ++i) {
            double* row = out + (b0 + i) * C.D;
            const double p = v[i];
            if (p != p || (rule == Extrapolate::Nan && (p < lo || p > hi))) {
                std::fill(row, row + C.D, kNaN);
                continue;
            }
            if (nu > T.k) {
                std::fill(row, row + C.D, 0.0);
                continue;
            }
            const std::int64_t l = knot_interval(cnt[i], T.k, T.n);
            if constexpr (K >= 0) basis_fixed<K>(T.t, l, p, h);
            else basis_values(T.t, T.k, l, p, nu, h, hh);
            for (std::int64_t col = 0; col < C.D; ++col) row[col] = combine(C, l, T.k, col, h);
        }
    }
}

nb::object bspline_eval(nb::handle to, nb::handle co, std::int64_t k, nb::handle qo, std::int64_t nu, int rule) {
    const Knots T = knots_arg(to, k);
    const Coefs C = coefs_arg(co, T.n);
    const NDArray& q = vector_arg(qo, "x");
    if (nu < 0) throw nb::value_error("'nu' must be nonnegative");
    if (rule < 0 || rule > 2) throw nb::value_error("'rule' must be 0, 1 or 2");
    const std::int64_t nq = q.shape[0];
    NDArray out = new_values(C.dtype, nq, C.ncol);
    if (nq == 0 || C.D == 0) return wrap(std::move(out));
    const auto* xq = reinterpret_cast<const double*>(q.data);
    auto* o = reinterpret_cast<double*>(out.data);
    const std::int64_t log_n = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::log2(T.N + 1)));
    const std::int64_t per_point = log_n + 3 * (k + 1) * (k + 1) + 2 * (k + 1) * C.D;
    split_range(nq, nq * per_point, 256, [&](std::int64_t b, std::int64_t e) {
        const auto r = static_cast<Extrapolate>(rule);
        switch (nu == 0 ? k : -1) {
            case 2: eval_range<2>(T, C, xq, o, nu, r, b, e); break;
            case 3: eval_range<3>(T, C, xq, o, nu, r, b, e); break;
            case 4: eval_range<4>(T, C, xq, o, nu, r, b, e); break;
            case 5: eval_range<5>(T, C, xq, o, nu, r, b, e); break;
            default: eval_range<-1>(T, C, xq, o, nu, r, b, e);
        }
    });
    return wrap(std::move(out));
}

// design_matrix: per point the k + 1 basis values of its interval (zeros kept) and their columns.
nb::object bspline_basis(nb::handle to, std::int64_t k, nb::handle qo, bool extend) {
    const Knots T = knots_arg(to, k);
    const NDArray& q = vector_arg(qo, "x");
    const std::int64_t nq = q.shape[0], w = k + 1;
    const auto* xq = reinterpret_cast<const double*>(q.data);
    NDArray data = new_vector(nq * w);
    const std::int64_t ishape[1] = {nq * w}, pshape[1] = {nq + 1};
    NDArray indices = new_array(DType::Int64, ishape);
    NDArray indptr = new_array(DType::Int64, pshape);
    auto* d = reinterpret_cast<double*>(data.data);
    auto* ind = reinterpret_cast<std::int64_t*>(indices.data);
    auto* ptr = reinterpret_cast<std::int64_t*>(indptr.data);
    for (std::int64_t i = 0; i <= nq; ++i) ptr[i] = i * w;
    split_range(nq, nq * (3 * w * w + 20), 256, [&](std::int64_t b, std::int64_t e) {
        std::vector<double> hh(w);
        for (std::int64_t i = b; i < e; ++i) {
            const double x = xq[i];
            const std::int64_t l = knot_interval_of(T, x);
            double* h = d + i * w;
            if (!extend && (x < T.t[k] || x > T.t[T.n])) {
                std::fill(h, h + w, kNaN);  // the caller checks the bounds first
            } else {
                basis_values(T.t, k, l, x, 0, h, hh.data());
            }
            for (std::int64_t r = 0; r < w; ++r) ind[i * w + r] = l - k + r;
        }
    });
    nb::list out;
    out.append(wrap(std::move(data)));
    out.append(wrap(std::move(indices)));
    out.append(wrap(std::move(indptr)));
    return nb::tuple(out);
}

// ---------------------------------------------------------------- knots

nb::object bspline_knots(nb::handle xo, std::int64_t k, int kind) {
    const NDArray& xa = vector_arg(xo, "x");
    const auto* x = reinterpret_cast<const double*>(xa.data);
    const std::int64_t n = xa.shape[0];
    if (n < 2 || k < 0) throw nb::value_error("'x' must have 2 points or more, 'k' must be nonnegative");
    std::vector<double> t;
    if (kind == 0) {  // scipy's _not_a_knot: odd k the nodes, even k the midpoints in doubles; k2 dropped per end
        std::vector<double> mid;
        std::int64_t k2;
        if (k % 2 == 1) {
            k2 = (k + 1) / 2;
            mid.assign(x, x + n);
        } else {
            k2 = k / 2;
            for (std::int64_t i = 0; i + 1 < n; ++i) mid.push_back((x[i + 1] + x[i]) / 2);
        }
        t.assign(k + 1, x[0]);
        const std::int64_t m = static_cast<std::int64_t>(mid.size());
        for (std::int64_t i = k2; i < m - k2; ++i) t.push_back(mid[i]);
        t.insert(t.end(), k + 1, x[n - 1]);
    } else if (kind == 1) {  // scipy's _augknt
        t.assign(k, x[0]);
        t.insert(t.end(), x, x + n);
        t.insert(t.end(), k, x[n - 1]);
    } else if (kind == 2) {  // scipy's _periodic_knots
        std::vector<double> xc(x, x + n), dx(n - 1);
        if (k % 2 == 0) {
            for (std::int64_t i = 0; i + 1 < n; ++i) dx[i] = xc[i + 1] - xc[i];
            for (std::int64_t i = 1; i + 1 < n; ++i) xc[i] = xc[i] - dx[i - 1] / 2;
        }
        for (std::int64_t i = 0; i + 1 < n; ++i) dx[i] = xc[i + 1] - xc[i];
        t.assign(n + 2 * k, 0.0);
        for (std::int64_t i = 0; i < n; ++i) t[k + i] = xc[i];
        const std::int64_t len = n + 2 * k;
        for (std::int64_t i = 0; i < k; ++i) {
            t[k - i - 1] = t[k - i] - dx[(n - 1) - (i % (n - 1)) - 1];
            t[len - k + i] = t[len - k + i - 1] + dx[i % (n - 1)];
        }
    } else {
        throw nb::value_error("'kind' must be 0, 1 or 2");
    }
    NDArray out = new_vector(static_cast<std::int64_t>(t.size()));
    if (!t.empty()) std::memcpy(out.data, t.data(), t.size() * sizeof(double));
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- the collocation systems

// Solves the band for every column of R (rows, D) on the pool, columns by groups of a fixed size.
void solve_columns(const BandSolver& S, double* R, std::int64_t D) {
    const std::int64_t rows = S.rows();
    constexpr std::int64_t kGroup = 8;
    const std::int64_t groups = (D + kGroup - 1) / kGroup;
    split_range(groups, groups * rows * 16, 1, [&](std::int64_t g0, std::int64_t g1) {
        for (std::int64_t g = g0; g < g1; ++g) S.solve(R, D, g * kGroup, std::min(D, (g + 1) * kGroup));
    });
}

nb::object pair_result(NDArray&& c, int status) {
    nb::list out;
    out.append(wrap(std::move(c)));
    out.append(nb::int_(status));
    return nb::tuple(out);
}

nb::object bspline_interp(
    nb::handle xo,
    nb::handle yo,
    nb::handle to,
    std::int64_t k,
    nb::handle loo,
    nb::handle lvo,
    nb::handle roo,
    nb::handle rvo,
    bool pivot
) {
    const NDArray& xa = vector_arg(xo, "x");
    const auto* x = reinterpret_cast<const double*>(xa.data);
    const std::int64_t n = xa.shape[0];
    const Knots T = knots_arg(to, k);
    const Coefs Y = coefs_arg(yo, n);
    if (Y.rows != n) throw nb::value_error("'y' must have one row per point");
    const NDArray& lo = require_ndarray(loo, "lo");
    const NDArray& ro = require_ndarray(roo, "ro");
    if (lo.dtype != DType::Int64 || ro.dtype != DType::Int64 || lo.ndim != 1 || ro.ndim != 1 || !lo.c_contiguous ||
        !ro.c_contiguous)
        throw nb::value_error("'lo' and 'ro' must be C-contiguous 1-D int64 arrays");
    const std::int64_t nl = lo.shape[0], nr = ro.shape[0];
    const Coefs LV = coefs_arg(lvo, nl), RV = coefs_arg(rvo, nr);
    if (LV.D != Y.D || RV.D != Y.D || LV.dtype != Y.dtype || RV.dtype != Y.dtype)
        throw nb::value_error("the derivative values must have the values' columns and dtype");
    const std::int64_t N = T.n;
    if (n < 2 || N != n + nl + nr) throw nb::value_error("the conditions must be as many as the coefficients");
    const auto* lord = reinterpret_cast<const std::int64_t*>(lo.data);
    const auto* rord = reinterpret_cast<const std::int64_t*>(ro.data);
    // the rows (IP35): x[0], the left derivatives, x[1] .. x[n-2], the right derivatives, x[n-1]
    struct Row {
        double x;
        std::int64_t nu;
        const double* rhs;
    };
    std::vector<Row> rows;
    rows.reserve(N);
    rows.push_back({x[0], 0, Y.c});
    for (std::int64_t i = 0; i < nl; ++i) rows.push_back({x[0], lord[i], LV.c + i * Y.D});
    for (std::int64_t i = 1; i + 1 < n; ++i) rows.push_back({x[i], 0, Y.c + i * Y.D});
    for (std::int64_t i = 0; i < nr; ++i) rows.push_back({x[n - 1], rord[i], RV.c + i * Y.D});
    rows.push_back({x[n - 1], 0, Y.c + (n - 1) * Y.D});
    std::vector<std::int64_t> left(N);
    std::int64_t p = 0, q = 0, from = 0;
    for (std::int64_t r = 0; r < N; ++r) {  // the rows' points never decrease: a gallop from the previous count
        if (rows[r].nu < 0 || rows[r].nu > k) throw nb::value_error("a derivative order is outside [0, k]");
        from = count_le_from(T.t, T.N, rows[r].x, from);
        left[r] = knot_interval(from, k, T.n);
        p = std::max(p, r - (left[r] - k));
        q = std::max(q, left[r] - r);
    }
    {  // a derivative condition's row overflows when the knot spacing is tiny (subnormal): -2, not "singular"
        std::vector<double> h(k + 1), hh(k + 1);
        for (std::int64_t r = 0; r < N; ++r) {
            if (rows[r].nu == 0) continue;
            basis_values(T.t, k, left[r], rows[r].x, rows[r].nu, h.data(), hh.data());
            for (double v : h)
                if (!std::isfinite(v)) return pair_result(new_values(Y.dtype, N, Y.ncol), -2);
        }
    }
    BandSolver S(N, p, q);
    NDArray out = new_values(Y.dtype, N, Y.ncol);
    auto* R = reinterpret_cast<double*>(out.data);
    for (std::int64_t r = 0; r < N; ++r)
        for (std::int64_t col = 0; col < Y.D; ++col) R[r * Y.D + col] = rows[r].rhs[col];
    // the rows are independent: filled in pieces on the pool (each entry the same operations on any thread)
    const auto fill_rows = [&](BandSolver& B, std::int64_t r0, std::int64_t r1) {
        std::vector<double> h(k + 1), hh(k + 1);
        for (std::int64_t r = r0; r < r1; ++r) {
            const std::int64_t l = left[r];
            if (rows[r].nu == 0 && k == 3) basis_fixed<3>(T.t, l, rows[r].x, h.data());
            else if (rows[r].nu == 0 && k == 5) basis_fixed<5>(T.t, l, rows[r].x, h.data());
            else basis_values(T.t, k, l, rows[r].x, rows[r].nu, h.data(), hh.data());
            for (std::int64_t s = 0; s <= k; ++s) B.at(r, l - k + s) = h[s];
        }
    };
    std::int64_t bad;
    {
        const parallel::Unlocked unlocked(N * (p + 1) * (q + 1) >= kParallelThreshold);
        bad = S.factor(
            [&](BandSolver& B) {
                split_range(N, N * 3 * (k + 1) * (k + 1), 4096, [&](std::int64_t r0, std::int64_t r1) {
                    fill_rows(B, r0, r1);
                });
            },
            pivot
        );
    }
    if (bad >= 0) return pair_result(std::move(out), -1);
    if (Y.D > 0) solve_columns(S, R, Y.D);
    return pair_result(std::move(out), S.pivoted() ? 1 : 0);
}

// The periodic system (scipy's _make_periodic_spline): n - 1 = m unknowns d with c_j = d_{j mod m}, one row per
// node but the last, B_j(x_i) added into column j mod m. With the columns turned by s = k / 2 the matrix is a band
// A_b (a submatrix of the periodic collocation matrix, so totally nonnegative with a positive diagonal) plus a few
// corner entries U V^T; Woodbury's formula (scipy's choice too): A_b Z = U, (I + V^T Z) g = V^T A_b^-1 y, d =
// A_b^-1 y - Z g, the small system by partial pivoting. n <= k: scipy's full system of n + k - 1 unknowns (the k - 1
// continuity conditions of the derivatives at the ends and the n interpolation conditions), dense.
nb::object bspline_periodic(nb::handle xo, nb::handle yo, nb::handle to, std::int64_t k) {
    const NDArray& xa = vector_arg(xo, "x");
    const auto* x = reinterpret_cast<const double*>(xa.data);
    const std::int64_t n = xa.shape[0];
    const Knots T = knots_arg(to, k);
    const Coefs Y = coefs_arg(yo, n);
    if (n < 2 || T.n != n + k - 1) throw nb::value_error("'t' must be the periodic knots of 'x'");
    const std::int64_t nt = T.n, D = Y.D;
    NDArray out = new_values(Y.dtype, nt, Y.ncol, true);
    auto* c = reinterpret_cast<double*>(out.data);
    std::vector<double> h(k + 1), hh(k + 1);
    if (n <= k) {
        const std::int64_t m = nt;
        std::vector<double> M(m * m, 0.0), R(m * D, 0.0);
        for (std::int64_t i = 0; i + 1 < k; ++i) {
            basis_values(T.t, k, k, x[0], i + 1, h.data(), hh.data());
            for (std::int64_t s = 0; s <= k; ++s) M[i * m + s] += h[s];
            basis_values(T.t, k, nt, x[n - 1], i + 1, h.data(), hh.data());  // the interval right of x[n-1]
            for (std::int64_t s = 0; s < k; ++s) M[i * m + (m - k + s)] -= h[s];
        }
        for (std::int64_t i = 0; i < n; ++i) {
            const std::int64_t l = knot_interval_of(T, x[i]);
            basis_values(T.t, k, l, x[i], 0, h.data(), hh.data());
            for (std::int64_t s = 0; s <= k; ++s) M[(i + k - 1) * m + (l - k + s)] = h[s];
            std::memcpy(R.data() + (i + k - 1) * D, Y.c + i * D, D * sizeof(double));
        }
        if (!dense_solve(M.data(), m, R.data(), D)) return pair_result(std::move(out), -1);
        std::memcpy(c, R.data(), m * D * sizeof(double));
        return pair_result(std::move(out), 0);
    }
    const std::int64_t m = n - 1, s = k / 2;
    // the entries: row i, unwrapped column u = l - k + r - s (the band when 0 <= u < m, else a corner at u mod m)
    struct Entry {
        std::int64_t i, col;
        double v;
        bool corner;
    };
    std::vector<Entry> entries;
    entries.reserve(m * (k + 1));
    std::int64_t p = 0, q = 0, from = 0;
    for (std::int64_t i = 0; i < m; ++i) {
        from = count_le_from(T.t, T.N, x[i], from);
        const std::int64_t l = knot_interval(from, k, T.n);
        basis_values(T.t, k, l, x[i], 0, h.data(), hh.data());
        for (std::int64_t r = 0; r <= k; ++r) {
            const std::int64_t u = l - k + r - s;
            if (u >= 0 && u < m) {
                entries.push_back({i, u, h[r], false});
                p = std::max(p, i - u);
                q = std::max(q, u - i);
            } else {
                entries.push_back({i, ((u % m) + m) % m, h[r], true});
            }
        }
    }
    std::vector<std::int64_t> cols;  // the corner columns, in increasing order
    for (const Entry& e : entries)
        if (e.corner && std::find(cols.begin(), cols.end(), e.col) == cols.end()) cols.push_back(e.col);
    std::sort(cols.begin(), cols.end());
    const std::int64_t nk = static_cast<std::int64_t>(cols.size());
    BandSolver S(m, p, q);
    // right sides: the corner columns U (m, nk), then the values (m, D)
    const std::int64_t W = nk + D;
    std::vector<double> R(m * W, 0.0);
    for (const Entry& e : entries) {
        if (e.corner) {
            const std::int64_t j = std::find(cols.begin(), cols.end(), e.col) - cols.begin();
            R[e.i * W + j] += e.v;
        }
    }
    for (std::int64_t i = 0; i < m; ++i) std::memcpy(R.data() + i * W + nk, Y.c + i * D, D * sizeof(double));
    std::int64_t bad;
    {
        const parallel::Unlocked unlocked(m * (p + 1) * (q + 1) >= kParallelThreshold);
        bad = S.factor([&](BandSolver& B) {
            for (const Entry& e : entries)
                if (!e.corner) B.at(e.i, e.col) += e.v;
        });
    }
    if (bad >= 0) return pair_result(std::move(out), -1);
    solve_columns(S, R.data(), W);
    // H = I + V^T Z (nk, nk); g = H^-1 V^T y; d = y' - Z g
    if (nk > 0 && D > 0) {
        std::vector<double> H(nk * nk), G(nk * D);
        for (std::int64_t a = 0; a < nk; ++a) {
            for (std::int64_t b = 0; b < nk; ++b) H[a * nk + b] = (a == b ? 1.0 : 0.0) + R[cols[a] * W + b];
            for (std::int64_t col = 0; col < D; ++col) G[a * D + col] = R[cols[a] * W + nk + col];
        }
        if (!dense_solve(H.data(), nk, G.data(), D)) return pair_result(std::move(out), -1);
        for (std::int64_t i = 0; i < m; ++i) {
            double* r = R.data() + i * W;
            for (std::int64_t col = 0; col < D; ++col) {
                double acc = r[nk + col];
                for (std::int64_t b = 0; b < nk; ++b) acc = acc - r[b] * G[b * D + col];
                r[nk + col] = acc;
            }
        }
    }
    // c_j = d_{j mod m} = d'_{(j - s) mod m}
    for (std::int64_t j = 0; j < nt; ++j) {
        const std::int64_t src = (((j - s) % m) + m) % m;
        std::memcpy(c + j * D, R.data() + src * W + nk, D * sizeof(double));
    }
    return pair_result(std::move(out), S.pivoted() ? 1 : 0);
}

// ---------------------------------------------------------------- calculus

// scipy's splder, nu times: dt = t[k+1 : -1] - t[1 : -k-1]; c = (c[1 : -1-k] - c[: -2-k]) k / dt, padded with k
// zero rows; t = t[1 : -1]; k -= 1. A zero dt (repeated internal knots) is None.
nb::object bspline_derivative(nb::handle to, nb::handle co, std::int64_t k, std::int64_t nu) {
    const Knots T0 = knots_arg(to, k);
    const Coefs C = coefs_arg(co, T0.n);
    if (nu < 0 || nu > k) throw nb::value_error("'nu' must be in [0, k]");
    const std::int64_t D = C.D;
    std::vector<double> t(T0.t, T0.t + T0.N);
    std::vector<double> c(T0.N * D, 0.0);  // padded to the number of knots (scipy's FITPACK convention)
    std::memcpy(c.data(), C.c, std::min(C.rows, T0.N) * D * sizeof(double));
    std::int64_t kk = k;
    for (std::int64_t step = 0; step < nu; ++step) {
        const std::int64_t N = static_cast<std::int64_t>(t.size()), m = N - kk - 2;
        std::vector<double> c2((N - 2) * D, 0.0);
        const double fk = static_cast<double>(kk);
        for (std::int64_t i = 0; i < m; ++i) {
            const double dt = t[i + kk + 1] - t[i + 1];
            if (dt == 0.0) return nb::none();
            for (std::int64_t col = 0; col < D; ++col)
                c2[i * D + col] = (c[(i + 1) * D + col] - c[i * D + col]) * fk / dt;
        }
        t = std::vector<double>(t.begin() + 1, t.end() - 1);
        c = std::move(c2);
        --kk;
    }
    NDArray tout = new_vector(static_cast<std::int64_t>(t.size()));
    std::memcpy(tout.data, t.data(), t.size() * sizeof(double));
    NDArray cout_ = new_values(C.dtype, static_cast<std::int64_t>(t.size()), C.ncol);
    if (!c.empty()) std::memcpy(cout_.data, c.data(), c.size() * sizeof(double));
    nb::list out;
    out.append(wrap(std::move(tout)));
    out.append(wrap(std::move(cout_)));
    return nb::tuple(out);
}

// scipy's splantider, nu times: dt = t[k+1:] - t[:-k-1]; c = cumsum(c[:n] dt) / (k + 1) in order (a plain running
// sum per column); c = [0, c, c[-1] (k + 2 times)]; t = [t[0], t, t[-1]]; k += 1.
nb::object bspline_antiderivative(nb::handle to, nb::handle co, std::int64_t k, std::int64_t nu) {
    const Knots T0 = knots_arg(to, k);
    const Coefs C = coefs_arg(co, T0.n);
    if (nu < 0) throw nb::value_error("'nu' must be nonnegative");
    const std::int64_t D = C.D;
    std::vector<double> t(T0.t, T0.t + T0.N);
    std::vector<double> c(T0.N * D, 0.0);
    std::memcpy(c.data(), C.c, std::min(C.rows, T0.N) * D * sizeof(double));
    std::int64_t kk = k;
    for (std::int64_t step = 0; step < nu; ++step) {
        const std::int64_t N = static_cast<std::int64_t>(t.size()), n = N - kk - 1;
        std::vector<double> c2((N + 2) * D, 0.0);
        const double div = static_cast<double>(kk + 1);
        split_range(D, D * n * 3, 1, [&](std::int64_t c0, std::int64_t c1) {
            for (std::int64_t col = c0; col < c1; ++col) {
                double run = 0.0;
                for (std::int64_t i = 0; i < n; ++i) {
                    const double term = c[i * D + col] * (t[i + kk + 1] - t[i]);
                    run = i == 0 ? term : run + term;
                    c2[(i + 1) * D + col] = run / div;
                }
                for (std::int64_t i = n + 1; i < N + 2; ++i) c2[i * D + col] = c2[n * D + col];
            }
        });
        std::vector<double> t2;
        t2.reserve(N + 2);
        t2.push_back(t.front());
        t2.insert(t2.end(), t.begin(), t.end());
        t2.push_back(t.back());
        t = std::move(t2);
        c = std::move(c2);
        ++kk;
    }
    NDArray tout = new_vector(static_cast<std::int64_t>(t.size()));
    std::memcpy(tout.data, t.data(), t.size() * sizeof(double));
    NDArray cout_ = new_values(C.dtype, static_cast<std::int64_t>(t.size()), C.ncol);
    if (!c.empty()) std::memcpy(cout_.data, c.data(), c.size() * sizeof(double));
    nb::list out;
    out.append(wrap(std::move(tout)));
    out.append(wrap(std::move(cout_)));
    return nb::tuple(out);
}

// ---------------------------------------------------------------- knot insertion

// scipy's _insert (FITPACK's insert, Boehm's algorithm), m times: the interval l of x (x in [t[k], t[n]]; one less
// when t[l] == t[l + k + 1]), the knot after t[l], the coefficients c[l-k+1 .. l] replaced by
// fac c[i] + (1 - fac) c[i - 1], fac = (x - tt[i]) / (tt[i+k+1] - tt[i]); a periodic spline copies its boundary
// knots and coefficients. None: a periodic spline without k internal knots on either side of x.
nb::object bspline_insert(nb::handle to, nb::handle co, std::int64_t k, double xv, std::int64_t times, bool periodic) {
    const Knots T0 = knots_arg(to, k);
    const Coefs C = coefs_arg(co, T0.n);
    const std::int64_t D = C.D;
    std::vector<double> t(T0.t, T0.t + T0.N);
    std::vector<double> c(C.c, C.c + C.rows * D);
    for (std::int64_t rep = 0; rep < times; ++rep) {
        const std::int64_t N = static_cast<std::int64_t>(t.size()), n = N - k - 1;
        const std::int64_t rows = static_cast<std::int64_t>(c.size()) / std::max<std::int64_t>(D, 1);
        std::int64_t l = k;
        while (l + 1 < n && t[l + 1] <= xv) ++l;  // scipy's find_interval from k (the largest l <= n - 1)
        if (t[l] == t[l + k + 1]) --l;
        if (periodic && l + 1 <= 2 * k && l + 1 >= N - 2 * k) return nb::none();
        std::vector<double> tt;
        tt.reserve(N + 1);
        tt.insert(tt.end(), t.begin(), t.begin() + l + 1);
        tt.push_back(xv);
        tt.insert(tt.end(), t.begin() + l + 1, t.end());
        std::vector<double> cc((rows + 1) * D, 0.0);
        for (std::int64_t i = l; i < rows; ++i)
            for (std::int64_t col = 0; col < D; ++col) cc[(i + 1) * D + col] = c[i * D + col];
        for (std::int64_t i = l; i > l - k; --i) {
            const double fac = (xv - tt[i]) / (tt[i + k + 1] - tt[i]);
            const std::int64_t prev = i - 1 >= 0 ? i - 1 : rows - 1;  // numpy's c[-1] in scipy's edge case
            for (std::int64_t col = 0; col < D; ++col)
                cc[i * D + col] = fac * c[i * D + col] + (1.0 - fac) * c[prev * D + col];
        }
        for (std::int64_t i = 0; i < l - k + 1; ++i)
            for (std::int64_t col = 0; col < D; ++col) cc[i * D + col] = c[i * D + col];
        if (periodic) {
            const std::int64_t nn = static_cast<std::int64_t>(tt.size()), nk = nn - k - 1, n2k = nn - 2 * k - 1;
            const double per = tt[nk] - tt[k];
            if (l >= nk - k) {
                for (std::int64_t i = 0; i < k; ++i) tt[i] = tt[nk - k + i] - per;
                for (std::int64_t i = 0; i < k; ++i)
                    for (std::int64_t col = 0; col < D; ++col) cc[i * D + col] = cc[(n2k + i) * D + col];
            }
            if (l <= 2 * k - 1) {
                for (std::int64_t i = 0; i < k; ++i) tt[nn - k + i] = tt[k + 1 + i] + per;
                for (std::int64_t i = 0; i < k; ++i)
                    for (std::int64_t col = 0; col < D; ++col) cc[(n2k + i) * D + col] = cc[i * D + col];
            }
        }
        t = std::move(tt);
        c = std::move(cc);
    }
    NDArray tout = new_vector(static_cast<std::int64_t>(t.size()));
    std::memcpy(tout.data, t.data(), t.size() * sizeof(double));
    const std::int64_t rows = D > 0 ? static_cast<std::int64_t>(c.size()) / D : C.rows + times;
    NDArray cout_ = new_values(C.dtype, rows, C.ncol);
    if (!c.empty()) std::memcpy(cout_.data, c.data(), c.size() * sizeof(double));
    nb::list out;
    out.append(wrap(std::move(tout)));
    out.append(wrap(std::move(cout_)));
    return nb::tuple(out);
}

// ---------------------------------------------------------------- the power basis

// PPoly.from_spline: the breakpoints are the distinct knots of the base interval [t[k], t[n]] (so the piecewise
// polynomial is the spline's function under every outside rule); piece j's coefficient of degree d is the d-th
// derivative at its left breakpoint (from the right) over d!: the spline of degree k - d whose coefficients are the
// piece's k + 1 coefficients differenced d times (scipy's splder formula, (a_j - a_{j-1}) kk / (t[j+kk] - t[j]); its
// spacings are positive there), evaluated at the breakpoint.
nb::object bspline_to_ppoly(nb::handle to, nb::handle co, std::int64_t k) {
    const Knots T = knots_arg(to, k);
    const Coefs C = coefs_arg(co, T.n);
    std::vector<double> br;
    for (std::int64_t i = k; i <= T.n; ++i)
        if (br.empty() || T.t[i] != br.back()) br.push_back(T.t[i]);
    const std::int64_t m = static_cast<std::int64_t>(br.size()) - 1;
    if (m < 1) throw nb::value_error("the base interval is empty");
    const std::int64_t D = C.D;
    const std::int64_t shape[3] = {k + 1, m, C.ncol};
    NDArray cp = new_array(C.dtype, shape);
    auto* o = reinterpret_cast<double*>(cp.data);
    std::vector<double> fact(k + 1, 1.0);
    for (std::int64_t d = 1; d <= k; ++d) fact[d] = fact[d - 1] * static_cast<double>(d);
    split_range(m, m * (k + 1) * ((k + 1) * (k + 1) + 3 * D), 256, [&](std::int64_t j0, std::int64_t j1) {
        std::vector<double> h(k + 1), hh(k + 1), a((k + 1) * D);
        std::int64_t from = 0;
        for (std::int64_t j = j0; j < j1; ++j) {
            const double x = br[j];
            from = count_le_from(T.t, T.N, x, from);
            const std::int64_t l = knot_interval(from, k, T.n);
            std::memcpy(a.data(), C.c + (l - k) * D, (k + 1) * D * sizeof(double));
            for (std::int64_t d = 0; d <= k; ++d) {
                const std::int64_t kk = k - d;  // a[r] belongs to B_{l-kk+r} of degree kk
                switch (kk) {  // the unrolled basis for the small degrees: the same operations (deboor.hpp)
                    case 0: h[0] = 1.0; break;
                    case 1: basis_fixed<1>(T.t, l, x, h.data()); break;
                    case 2: basis_fixed<2>(T.t, l, x, h.data()); break;
                    case 3: basis_fixed<3>(T.t, l, x, h.data()); break;
                    case 4: basis_fixed<4>(T.t, l, x, h.data()); break;
                    case 5: basis_fixed<5>(T.t, l, x, h.data()); break;
                    default: basis_values(T.t, kk, l, x, 0, h.data(), hh.data());
                }
                double* row = o + ((k - d) * m + j) * D;
                // d! = 1 or 2 divides exactly as a product with its reciprocal (the same rounding of the same
                // quotient): the divisions that remain are those by 6, 24, ...
                const double f = fact[d], inv = d <= 2 ? 1.0 / f : 0.0;
                for (std::int64_t col = 0; col < D; ++col) {
                    double acc = a[col] * h[0];
                    for (std::int64_t r = 1; r <= kk; ++r) acc = std::fma(a[r * D + col], h[r], acc);
                    row[col] = d <= 2 ? acc * inv : acc / f;
                }
                if (kk == 0) break;
                const double fk = static_cast<double>(kk);
                for (std::int64_t r = 0; r < kk; ++r) {
                    const std::int64_t jj = l - kk + 1 + r;
                    const double dt = T.t[jj + kk] - T.t[jj];
                    for (std::int64_t col = 0; col < D; ++col)
                        a[r * D + col] = (a[(r + 1) * D + col] - a[r * D + col]) * fk / dt;
                }
            }
        }
    });
    NDArray bout = new_vector(m + 1);
    std::memcpy(bout.data, br.data(), (m + 1) * sizeof(double));
    nb::list out;
    out.append(wrap(std::move(cp)));
    out.append(wrap(std::move(bout)));
    return nb::tuple(out);
}

// BSpline.from_power_basis (scipy's formula through Marsden's identity): c_j = sum_d a_d e_d(u) / binom(k, d), the
// blossom of piece i at the knots t[j+1 .. j+k], u_r = t[j+r] - x[i], a_d piece i's coefficient of degree d and
// e_d the elementary symmetric polynomial; piece j for j < n - 2 (n breakpoints), piece n - 2 for the rest.
nb::object bspline_from_power(nb::handle xo, nb::handle co, nb::handle to, std::int64_t k) {
    const NDArray& xa = vector_arg(xo, "x");
    const auto* x = reinterpret_cast<const double*>(xa.data);
    const std::int64_t nb_ = xa.shape[0];
    const NDArray& ca = require_ndarray(co, "c");
    if (ca.ndim != 3 || !ca.c_contiguous || ca.shape[0] != k + 1 || ca.shape[1] != nb_ - 1)
        throw nb::value_error("'c' must be a C-contiguous (k + 1, m, ncol) array");
    const Knots T = knots_arg(to, k);
    const std::int64_t per = ca.dtype == DType::Complex128 ? 2 : 1, D = ca.shape[2] * per, m = ca.shape[1];
    const auto* cp = reinterpret_cast<const double*>(ca.data);
    NDArray out = new_values(ca.dtype, T.n, ca.shape[2]);
    auto* o = reinterpret_cast<double*>(out.data);
    std::vector<double> binom(k + 1, 1.0), e(k + 1);
    for (std::int64_t d = 1; d <= k; ++d)
        binom[d] = binom[d - 1] * static_cast<double>(k - d + 1) / static_cast<double>(d);
    {  // no Python object inside
        const parallel::Unlocked unlocked(T.n * (k + 1) * (k + 1 + D) >= kParallelThreshold);
        for (std::int64_t j = 0; j < T.n; ++j) {
            const std::int64_t i = std::min(j, nb_ - 2);
            std::fill(e.begin(), e.end(), 0.0);
            e[0] = 1.0;
            for (std::int64_t r = 1; r <= k; ++r) {
                const double u = T.t[j + r] - x[i];
                for (std::int64_t d = r; d >= 1; --d) e[d] = e[d] + u * e[d - 1];
            }
            for (std::int64_t col = 0; col < D; ++col) {
                double acc = 0.0;
                for (std::int64_t d = 0; d <= k; ++d) acc = acc + cp[((k - d) * m + i) * D + col] * e[d] / binom[d];
                o[j * D + col] = acc;
            }
        }
    }
    return wrap(std::move(out));
}

}  // namespace

void bind_bspline(nb::module_& m) {
    m.def("check_knots", &check_knots, nb::arg("t").none());
    m.def(
        "bspline_eval", &bspline_eval, nb::arg("t").none(), nb::arg("c").none(), nb::arg("k"), nb::arg("x").none(),
        nb::arg("nu"), nb::arg("rule")
    );  // fmt: skip
    m.def("bspline_basis", &bspline_basis, nb::arg("t").none(), nb::arg("k"), nb::arg("x").none(), nb::arg("extend"));
    m.def("bspline_knots", &bspline_knots, nb::arg("x").none(), nb::arg("k"), nb::arg("kind"));
    m.def(
        "bspline_interp", &bspline_interp, nb::arg("x").none(), nb::arg("y").none(), nb::arg("t").none(), nb::arg("k"),
        nb::arg("lo").none(), nb::arg("lv").none(), nb::arg("ro").none(), nb::arg("rv").none(), nb::arg("pivot")
    );  // fmt: skip
    m.def(
        "bspline_periodic", &bspline_periodic, nb::arg("x").none(), nb::arg("y").none(), nb::arg("t").none(),
        nb::arg("k")
    );  // fmt: skip
    m.def(
        "bspline_derivative", &bspline_derivative, nb::arg("t").none(), nb::arg("c").none(), nb::arg("k"),
        nb::arg("nu")
    );  // fmt: skip
    m.def(
        "bspline_antiderivative", &bspline_antiderivative, nb::arg("t").none(), nb::arg("c").none(), nb::arg("k"),
        nb::arg("nu")
    );  // fmt: skip
    m.def(
        "bspline_insert", &bspline_insert, nb::arg("t").none(), nb::arg("c").none(), nb::arg("k"), nb::arg("x"),
        nb::arg("m"), nb::arg("periodic")
    );  // fmt: skip
    m.def("bspline_to_ppoly", &bspline_to_ppoly, nb::arg("t").none(), nb::arg("c").none(), nb::arg("k"));
    m.def(
        "bspline_from_power", &bspline_from_power, nb::arg("x").none(), nb::arg("c").none(), nb::arg("t").none(),
        nb::arg("k")
    );  // fmt: skip
}

}  // namespace sd::interpolate
