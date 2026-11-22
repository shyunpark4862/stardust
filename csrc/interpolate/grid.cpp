// The regular grids of sd.interpolate (docs/design/interpolate.md §2.7, §3.7, §6; milestone IP5):
// RegularGridInterpolator's six methods on a grid of d ascending axes x_a (n_a nodes each) whose values are rows of a
// (n_0 ... n_{d-1}, K) array, row-major (axis 0 slowest), K columns of a batch.
//
//   linear    per axis the interval and t of base/interval.hpp and base/lerp.hpp (the 1-D linear path's), then the
//             2^d corner values reduced axis by axis from the last to the first with the linear kernel: exact at the
//             nodes, between the corner values, and in 1-D the bits of sd.interp and of the 1-D linear path
//             (linear.cpp). An axis of one node takes no part (the values are constant along it, scipy's rule)
//   nearest   per axis the left node when t <= 0.5, else the right one (scipy's rule, the same t)
//   spline    the tensor product of make_interp_spline's not-a-knot B-splines (degree k): the coefficients are the
//             1-D collocation solve (banded.hpp's BandSolver, the rows of bspline.cpp's) along each axis in turn, on
//             every line of that axis (the collocation matrix of a tensor product is the Kronecker product of the
//             axes' matrices, so this is the exact solve; no iteration, decision IP23); evaluation is the (k + 1)^d
//             coefficients around the point reduced axis by axis from the last to the first, each step de Boor's
//             basis of that axis (deboor.hpp) and bspline.cpp's sum (the first product, then one fma per term). In
//             1-D both give make_interp_spline's bits
//   pchip     PCHIP's value in an interval reads the four nodes around it only (its node slopes are local), so per
//             point the 4^d values of the window around it are reduced axis by axis from the last to the first, each
//             step the 1-D PCHIP of that axis's full node array at the window (pchip.hpp's slopes, cubic.cpp's Hermite
//             coefficients, ppoly.cpp's Horner): the function of scipy's RegularGridInterpolator(method='pchip'),
//             which builds a 1-D PchipInterpolator per point along each axis, and in 1-D PchipInterpolator's bits
//
// The outside rule (rule): 0 a point outside the grid on any axis gets the fill value; 1 every point extends the end
// pieces; 2 extends too and reports the lowest axis where a point lies outside (bounds_error: the caller raises). A
// point with a NaN coordinate is NaN under every rule and lies nowhere (decision IP24).
//
// Every point is one kernel, the points split by count alone: the results do not depend on the thread count. The
// coefficients' solves run per line and column, the same operations on any thread.
//
// Private bindings (the Python layer checks the arguments; points ascending, values C-contiguous):
//   grid_linear(points, values, xi, nearest, rule, fill_re, fill_im) -> (out, bad)  values (N, K) float64, complex128
//   grid_spline_coefs(points, knots, k, values) -> (c, status)       status 0, 1 (partial pivoting), -1 singular
//   grid_spline_eval(knots, k, c, xi, nu, rule, fill_re, fill_im) -> (out, bad)  c (N, K), n_a = len(t_a) - k - 1
//   grid_pchip(points, values, xi, rule, fill) -> (out, bad)         values float64, every axis 4 nodes or more
// xi (m, d) float64; the fill value's imaginary part counts for complex values only; bad the lowest axis outside
// (rule 2), else -1.

#include "interpolate/interpolate.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "base/interval.hpp"
#include "base/lerp.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "interpolate/banded.hpp"
#include "interpolate/deboor.hpp"
#include "interpolate/pchip.hpp"
#include "interpolate/search.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::int64_t kBlock = 256;

// fn(begin, end) over [0, total) in pieces of at least `grain`, on the pool when the work is large enough; the pieces
// depend on the sizes only (bspline.cpp's).
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

struct Axis {
    const double* x;  // ascending nodes (or knots)
    std::int64_t n;
};

const NDArray& vector_arg(nb::handle o, const char* what) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error(("'" + std::string(what) + "' must be C-contiguous 1-D float64 arrays").c_str());
    return a;
}

std::vector<Axis> axes_arg(nb::handle o, const char* what) {
    std::vector<Axis> out;
    for (nb::handle h : nb::borrow<nb::sequence>(o)) {
        const NDArray& a = vector_arg(h, what);
        if (a.shape[0] < 1) throw nb::value_error(("'" + std::string(what) + "' must not be empty").c_str());
        out.push_back({reinterpret_cast<const double*>(a.data), a.shape[0]});
    }
    if (out.empty()) throw nb::value_error(("'" + std::string(what) + "' must have an axis or more").c_str());
    return out;
}

struct Values {
    const double* v;
    std::int64_t rows, K, D;  // K columns, D doubles per row
    DType dtype;
};

Values values_arg(nb::handle o, std::int64_t rows, const char* what = "values") {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 2 || !a.c_contiguous || a.shape[0] != rows)
        throw nb::value_error(
            ("'" + std::string(what) + "' must be a C-contiguous (rows, K) array, a row per node").c_str()
        );
    if (a.dtype != DType::Float64 && a.dtype != DType::Complex128)
        throw nb::type_error(("'" + std::string(what) + "' must be float64 or complex128").c_str());
    const std::int64_t per = a.dtype == DType::Complex128 ? 2 : 1;
    return {reinterpret_cast<const double*>(a.data), a.shape[0], a.shape[1], a.shape[1] * per, a.dtype};
}

const NDArray& xi_arg(nb::handle o, std::int64_t d) {
    const NDArray& a = require_ndarray(o, "xi");
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[1] != d)
        throw nb::value_error("'xi' must be a C-contiguous (m, d) float64 array");
    return a;
}

// The row strides of the axes (in rows) and the number of rows; a count past 2^62 is a MemoryError.
std::vector<std::int64_t> strides_of(const std::vector<std::int64_t>& n, std::int64_t& rows) {
    std::vector<std::int64_t> s(n.size());
    rows = 1;
    for (std::int64_t a = static_cast<std::int64_t>(n.size()) - 1; a >= 0; --a) {
        s[a] = rows;
        if (n[a] > 0 && rows > (std::int64_t{1} << 62) / n[a]) throw std::bad_alloc();
        rows *= n[a];
    }
    return s;
}

// base^d, a MemoryError when the workspace of a point would pass 2^30 entries
std::int64_t corners(std::int64_t base, std::int64_t d) {
    std::int64_t c = 1;
    for (std::int64_t a = 0; a < d; ++a) {
        if (c > (std::int64_t{1} << 30) / base) throw std::bad_alloc();
        c *= base;
    }
    return c;
}

nb::object pair(NDArray&& out, std::int64_t second) { return nb::make_tuple(wrap(std::move(out)), second); }

NDArray new_values(DType dtype, std::int64_t rows, std::int64_t ncol) {
    const std::int64_t shape[2] = {rows, ncol};
    return new_array(dtype, shape);
}

// The lowest axis reported outside (rule 2), merged over the tasks.
struct Bad {
    std::atomic<std::int64_t> dim{std::numeric_limits<std::int64_t>::max()};
    void merge(std::int64_t a) {
        std::int64_t cur = dim.load();
        while (a < cur && !dim.compare_exchange_weak(cur, a)) {}
    }
    std::int64_t result() const {
        const std::int64_t v = dim.load();
        return v == std::numeric_limits<std::int64_t>::max() ? -1 : v;
    }
};

// What a point's row gets before any interpolation: 0 compute it, 1 NaN, 2 the fill value. The lowest axis where a
// point lies outside goes to `low` under rule 2.
inline int classify(const double* p, const std::vector<Axis>& ax, int rule, std::int64_t& low) {
    const std::int64_t d = static_cast<std::int64_t>(ax.size());
    std::int64_t out = -1;
    for (std::int64_t a = 0; a < d; ++a) {
        if (p[a] != p[a]) return 1;
        if (out < 0 && (p[a] < ax[a].x[0] || p[a] > ax[a].x[ax[a].n - 1])) out = a;
    }
    if (out < 0) return 0;
    if (rule == 0) return 2;
    if (rule == 2) low = std::min(low, out);
    return 0;
}

// The count of entries at or below each point's coordinate on every axis of `search` (the axes with `use[a]`), for a
// block of points: cnt[a * kBlock + i].
void count_block(
    const std::vector<Axis>& search,
    const std::vector<char>& use,
    const double* xi,
    std::int64_t d,
    std::int64_t b0,
    std::int64_t m,
    std::int64_t* cnt,
    std::vector<GallopState>& state
) {
    double v[kBlock];
    for (std::int64_t a = 0; a < d; ++a) {
        if (!use[a]) continue;
        for (std::int64_t i = 0; i < m; ++i) v[i] = xi[(b0 + i) * d + a];
        count_le_block(search[a].x, search[a].n, v, m, cnt + a * kBlock, state[a]);
    }
}

template <typename T>
constexpr T nan_value() {
    if constexpr (std::is_same_v<T, double>) return kNaN;
    else return T(kNaN, kNaN);
}

std::int64_t log2_of(std::int64_t n) { return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::log2(n + 1))); }

// ---------------------------------------------------------------- linear and nearest

template <typename T>
struct LinearJob {
    const std::vector<Axis>& ax;
    const std::vector<std::int64_t>& stride;
    const T* v;
    std::int64_t K;
    const double* xi;
    T* out;
    bool nearest;
    int rule;
    T fill;
};

// linear for real values, a block of points at a time: the axes' intervals and t, the corners' offsets, the corner
// values of every point gathered side by side (independent loads, so many grid reads are in flight at once), then
// each reduction step over the whole block in base/lerp.hpp's vectorized kernel. Per point the same operations as
// linear_range's (so the same bits); a block with a non-finite corner value redoes those steps with lerp.
void linear_block_range(const LinearJob<double>& J, Bad& bad, std::int64_t begin, std::int64_t end) {
    const std::int64_t d = static_cast<std::int64_t>(J.ax.size()), K = J.K;
    std::vector<std::int64_t> act;  // the axes of two nodes or more
    std::vector<char> use(d);
    for (std::int64_t a = 0; a < d; ++a)
        if (J.ax[a].n >= 2) act.push_back(a), use[a] = 1;
    const std::int64_t da = static_cast<std::int64_t>(act.size());
    const std::int64_t nc = corners(2, da);
    std::vector<std::int64_t> cnt(d * kBlock), off(nc * kBlock), live(kBlock);
    std::vector<double> t(std::max<std::int64_t>(da, 1) * kBlock), w(nc * kBlock), tmp(kBlock);
    std::vector<GallopState> state(d);
    std::int64_t low = std::numeric_limits<std::int64_t>::max();
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t mm = std::min(kBlock, end - b0);
        count_block(J.ax, use, J.xi, d, b0, mm, cnt.data(), state);
        std::int64_t nl = 0;  // the points computed, packed: live[0 .. nl)
        for (std::int64_t i = 0; i < mm; ++i) {
            const double* p = J.xi + (b0 + i) * d;
            double* row = J.out + (b0 + i) * K;
            const int what = classify(p, J.ax, J.rule, low);
            if (what) {
                std::fill(row, row + K, what == 1 ? kNaN : J.fill);
                continue;
            }
            // the corners: axis act[0] the highest bit, the last active axis the lowest (linear_range's order)
            std::int64_t* o = off.data() + nl * nc;
            o[0] = 0;
            std::int64_t size = 1;
            for (std::int64_t q = 0; q < da; ++q) {
                const std::int64_t a = act[q];
                const Axis& A = J.ax[a];
                const std::int64_t j = interval_of(cnt[a * kBlock + i], A.n);
                t[q * kBlock + nl] = lerp_t(p[a], A.x[j], A.x[j + 1]);
                const std::int64_t s = J.stride[a];
                for (std::int64_t c = size - 1; c >= 0; --c) {
                    o[2 * c + 1] = o[c] + (j + 1) * s;
                    o[2 * c] = o[c] + j * s;
                }
                size *= 2;
            }
            live[nl++] = i;
        }
        if (nl == 0) continue;
        for (std::int64_t col = 0; col < K; ++col) {
            for (std::int64_t c = 0; c < nc; ++c)  // w[c * kBlock + l]: corner c of point l
                for (std::int64_t l = 0; l < nl; ++l) w[c * kBlock + l] = J.v[off[l * nc + c] * K + col];
            std::int64_t n = nc;
            for (std::int64_t q = da - 1; q >= 0; --q) {
                n /= 2;
                const double* tq = t.data() + q * kBlock;
                for (std::int64_t c = 0; c < n; ++c) {
                    const double* wa = w.data() + 2 * c * kBlock;
                    const double* wb = wa + kBlock;
                    if (!lerp_ab_block(wa, wb, tq, tmp.data(), nl))
                        for (std::int64_t l = 0; l < nl; ++l) tmp[l] = lerp(wa[l], wb[l], tq[l]);
                    std::copy(tmp.begin(), tmp.begin() + nl, w.begin() + c * kBlock);
                }
            }
            for (std::int64_t l = 0; l < nl; ++l) J.out[(b0 + live[l]) * K + col] = w[l];
        }
    }
    bad.merge(low);
}

template <typename T>
void linear_range(const LinearJob<T>& J, Bad& bad, std::int64_t begin, std::int64_t end) {
    const std::int64_t d = static_cast<std::int64_t>(J.ax.size()), K = J.K;
    std::vector<std::int64_t> act;  // the axes of two nodes or more
    std::vector<char> use(d);
    for (std::int64_t a = 0; a < d; ++a)
        if (J.ax[a].n >= 2) act.push_back(a), use[a] = 1;
    const std::int64_t da = static_cast<std::int64_t>(act.size());
    const std::int64_t nc = J.nearest ? 1 : corners(2, da);
    std::vector<std::int64_t> cnt(d * kBlock), off(nc);
    std::vector<double> t(da);
    std::vector<T> w(nc);
    std::vector<GallopState> state(d);
    std::int64_t low = std::numeric_limits<std::int64_t>::max();
    const T nan_row = nan_value<T>();
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t mm = std::min(kBlock, end - b0);
        count_block(J.ax, use, J.xi, d, b0, mm, cnt.data(), state);
        for (std::int64_t i = 0; i < mm; ++i) {
            const double* p = J.xi + (b0 + i) * d;
            T* row = J.out + (b0 + i) * K;
            const int what = classify(p, J.ax, J.rule, low);
            if (what) {
                std::fill(row, row + K, what == 1 ? nan_row : J.fill);
                continue;
            }
            if (J.nearest) {
                std::int64_t o = 0;
                for (std::int64_t a : act) {
                    const Axis& A = J.ax[a];
                    const std::int64_t j = interval_of(cnt[a * kBlock + i], A.n);
                    const double tt = lerp_t(p[a], A.x[j], A.x[j + 1]);
                    o += (tt <= 0.5 ? j : j + 1) * J.stride[a];
                }
                std::copy(J.v + o * K, J.v + (o + 1) * K, row);
                continue;
            }
            // the corners: axis act[0] the highest bit, the last active axis the lowest
            off[0] = 0;
            std::int64_t size = 1;
            for (std::int64_t q = 0; q < da; ++q) {
                const std::int64_t a = act[q];
                const Axis& A = J.ax[a];
                const std::int64_t j = interval_of(cnt[a * kBlock + i], A.n);
                t[q] = lerp_t(p[a], A.x[j], A.x[j + 1]);
                const std::int64_t s = J.stride[a];
                for (std::int64_t c = size - 1; c >= 0; --c) {
                    off[2 * c + 1] = off[c] + (j + 1) * s;
                    off[2 * c] = off[c] + j * s;
                }
                size *= 2;
            }
            for (std::int64_t col = 0; col < K; ++col) {
                for (std::int64_t c = 0; c < nc; ++c) w[c] = J.v[off[c] * K + col];
                std::int64_t n = nc;
                for (std::int64_t q = da - 1; q >= 0; --q) {
                    n /= 2;
                    for (std::int64_t c = 0; c < n; ++c) w[c] = lerp(w[2 * c], w[2 * c + 1], t[q]);
                }
                row[col] = w[0];
            }
        }
    }
    bad.merge(low);
}

nb::object grid_linear(
    nb::handle po, nb::handle vo, nb::handle xo, bool nearest, int rule, double fill_re, double fill_im
) {
    const std::complex<double> fill(fill_re, fill_im);
    const std::vector<Axis> ax = axes_arg(po, "points");
    const std::int64_t d = static_cast<std::int64_t>(ax.size());
    std::vector<std::int64_t> n(d);
    for (std::int64_t a = 0; a < d; ++a) n[a] = ax[a].n;
    std::int64_t rows;
    const std::vector<std::int64_t> stride = strides_of(n, rows);
    const Values V = values_arg(vo, rows);
    const NDArray& xi = xi_arg(xo, d);
    if (rule < 0 || rule > 2) throw nb::value_error("'rule' must be 0, 1 or 2");
    const std::int64_t m = xi.shape[0];
    NDArray out = new_values(V.dtype, m, V.K);
    if (m == 0) return pair(std::move(out), -1);
    std::int64_t da = 0, logs = 0;
    for (std::int64_t a = 0; a < d; ++a)
        if (n[a] >= 2) ++da, logs += log2_of(n[a]) + 8;
    const std::int64_t per = logs + (nearest ? 1 : corners(2, da)) * std::max<std::int64_t>(V.K, 1) * 4 + 4 * d;
    const auto* x = reinterpret_cast<const double*>(xi.data);
    Bad bad;
    if (V.dtype == DType::Complex128) {
        const LinearJob<complex128> J{ax,      stride, reinterpret_cast<const complex128*>(V.v),
                                      V.K,     x,      reinterpret_cast<complex128*>(out.data),
                                      nearest, rule,   fill};  // fmt: skip
        split_range(m, m * per, kBlock, [&](std::int64_t b, std::int64_t e) { linear_range(J, bad, b, e); });
    } else {
        const LinearJob<double> J{ax,      stride, V.v,        V.K, x, reinterpret_cast<double*>(out.data),
                                  nearest, rule,   fill.real()};  // fmt: skip
        split_range(m, m * per, kBlock, [&](std::int64_t b, std::int64_t e) {
            // the block path's workspace costs more than it saves below a few dozen points (the same bits)
            if (nearest || corners(2, da) > 64 || e - b < 32) linear_range(J, bad, b, e);
            else linear_block_range(J, bad, b, e);
        });
    }
    return pair(std::move(out), bad.result());
}

// ---------------------------------------------------------------- tensor-product splines

// The knots of every axis: n_a = N_a - k - 1 coefficients.
std::vector<Axis> knots_arg(nb::handle o, std::int64_t k, std::vector<std::int64_t>& n) {
    std::vector<Axis> t = axes_arg(o, "knots");
    n.resize(t.size());
    for (std::size_t a = 0; a < t.size(); ++a) {
        if (t[a].n < 2 * k + 2) throw nb::value_error("'knots' must have 2 k + 2 knots or more per axis");
        n[a] = t[a].n - k - 1;
    }
    return t;
}

// The 1-D collocation matrix of not-a-knot interpolation at the nodes x (n of them) with the knots t, as
// bspline.cpp's bspline_interp builds it (the same rows, bands and basis), factored; status 0, 1 (partial pivoting)
// or -1 (singular).
BandSolver collocation(const Axis& X, const Axis& T, std::int64_t k, int& status) {
    const std::int64_t n = X.n;
    std::vector<std::int64_t> left(n);
    std::int64_t p = 0, q = 0, from = 0;
    for (std::int64_t r = 0; r < n; ++r) {
        from = count_le_from(T.x, T.n, X.x[r], from);
        left[r] = knot_interval(from, k, n);
        p = std::max(p, r - (left[r] - k));
        q = std::max(q, left[r] - r);
    }
    BandSolver S(n, p, q);
    const std::int64_t bad = S.factor([&](BandSolver& B) {
        std::vector<double> h(k + 1), hh(k + 1);
        for (std::int64_t r = 0; r < n; ++r) {
            const std::int64_t l = left[r];
            if (k == 3) basis_fixed<3>(T.x, l, X.x[r], h.data());
            else if (k == 5) basis_fixed<5>(T.x, l, X.x[r], h.data());
            else basis_values(T.x, k, l, X.x[r], 0, h.data(), hh.data());
            for (std::int64_t s = 0; s <= k; ++s) B.at(r, l - k + s) = h[s];
        }
    });
    status = bad >= 0 ? -1 : (S.pivoted() ? 1 : 0);
    return S;
}

nb::object grid_spline_coefs(nb::handle po, nb::handle to, std::int64_t k, nb::handle vo) {
    const std::vector<Axis> ax = axes_arg(po, "points");
    std::vector<std::int64_t> n;
    if (k < 1) throw nb::value_error("'k' must be 1 or more");
    const std::vector<Axis> kt = knots_arg(to, k, n);
    const std::int64_t d = static_cast<std::int64_t>(ax.size());
    if (static_cast<std::int64_t>(kt.size()) != d) throw nb::value_error("'knots' must have one array per axis");
    for (std::int64_t a = 0; a < d; ++a)
        if (n[a] != ax[a].n) throw nb::value_error("'knots' must be the not-a-knot knots of 'points'");
    std::int64_t rows;
    const std::vector<std::int64_t> stride = strides_of(n, rows);
    const Values V = values_arg(vo, rows);
    NDArray out = new_values(V.dtype, rows, V.K);
    auto* C = reinterpret_cast<double*>(out.data);
    std::copy(V.v, V.v + rows * V.D, C);
    int status = 0;
    if (rows == 0 || V.D == 0) return pair(std::move(out), 0);
    for (std::int64_t a = 0; a < d; ++a) {
        int st;
        const BandSolver S = [&] {  // the GIL released for the factorization alone (no Python object inside)
            const parallel::Unlocked unlocked(n[a] * (k + 1) * (k + 1) >= kParallelThreshold);
            return collocation(ax[a], kt[a], k, st);
        }();
        if (st < 0) return pair(std::move(out), -1);
        status = std::max(status, st);
        // the lines of axis a: P slabs of (n_a, Q) doubles, row-major; the columns of a slab by groups of 8
        const std::int64_t P = rows / (n[a] * stride[a]), Q = stride[a] * V.D;
        constexpr std::int64_t kGroup = 8;
        const std::int64_t groups = (Q + kGroup - 1) / kGroup, units = P * groups;
        const std::int64_t width = 2 * k + 2;
        split_range(units, units * n[a] * width * 2 * std::min(Q, kGroup), 1, [&](std::int64_t u0, std::int64_t u1) {
            for (std::int64_t u = u0; u < u1; ++u) {
                const std::int64_t slab = u / groups, g = u % groups;
                S.solve(C + slab * n[a] * Q, Q, g * kGroup, std::min(Q, (g + 1) * kGroup));
            }
        });
    }
    return pair(std::move(out), status);
}

struct SplineJob {
    const std::vector<Axis>& t;  // knots per axis
    const std::vector<Axis>& dom;  // [t[k], t[n]] per axis, as an axis of 2 entries (the outside test)
    const std::vector<std::int64_t>& n;
    const std::vector<std::int64_t>& stride;
    const std::vector<std::int64_t>& nu;
    std::int64_t k;
    const double* c;
    std::int64_t D;
    const double* xi;
    double* out;
    int rule;
    const std::vector<double>& fill;  // D doubles
};

void spline_range(const SplineJob& J, Bad& bad, std::int64_t begin, std::int64_t end) {
    const std::int64_t d = static_cast<std::int64_t>(J.t.size()), k = J.k, K1 = k + 1, D = J.D;
    const std::int64_t nc = corners(K1, d);
    std::vector<char> use(d, 1);
    std::vector<std::int64_t> cnt(d * kBlock), off(nc);
    std::vector<double> h(d * K1), hh(K1), w(nc);
    std::vector<GallopState> state(d);
    bool zero = false;
    for (std::int64_t a = 0; a < d; ++a) zero |= J.nu[a] > k;
    std::int64_t low = std::numeric_limits<std::int64_t>::max();
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t mm = std::min(kBlock, end - b0);
        count_block(J.t, use, J.xi, d, b0, mm, cnt.data(), state);
        for (std::int64_t i = 0; i < mm; ++i) {
            const double* p = J.xi + (b0 + i) * d;
            double* row = J.out + (b0 + i) * D;
            const int what = classify(p, J.dom, J.rule, low);
            if (what == 1) {
                std::fill(row, row + D, kNaN);
                continue;
            }
            if (what == 2) {
                std::copy(J.fill.begin(), J.fill.end(), row);
                continue;
            }
            if (zero) {
                std::fill(row, row + D, 0.0);
                continue;
            }
            off[0] = 0;
            std::int64_t size = 1;
            for (std::int64_t a = 0; a < d; ++a) {
                const std::int64_t l = knot_interval(cnt[a * kBlock + i], k, J.n[a]);
                double* ha = h.data() + a * K1;
                const double* ta = J.t[a].x;
                if (J.nu[a] == 0 && k == 3) basis_fixed<3>(ta, l, p[a], ha);
                else if (J.nu[a] == 0 && k == 5) basis_fixed<5>(ta, l, p[a], ha);
                else basis_values(ta, k, l, p[a], J.nu[a], ha, hh.data());
                const std::int64_t s = J.stride[a];
                for (std::int64_t c = size - 1; c >= 0; --c)
                    for (std::int64_t r = K1 - 1; r >= 0; --r) off[c * K1 + r] = off[c] + (l - k + r) * s;
                size *= K1;
            }
            for (std::int64_t col = 0; col < D; ++col) {
                for (std::int64_t c = 0; c < nc; ++c) w[c] = J.c[off[c] * D + col];
                std::int64_t len = nc;
                for (std::int64_t a = d - 1; a >= 0; --a) {
                    const double* ha = h.data() + a * K1;
                    len /= K1;
                    for (std::int64_t g = 0; g < len; ++g) {
                        const double* wg = w.data() + g * K1;
                        double acc = wg[0] * ha[0];
                        for (std::int64_t r = 1; r < K1; ++r) acc = std::fma(wg[r], ha[r], acc);
                        w[g] = acc;
                    }
                }
                row[col] = w[0];
            }
        }
    }
    bad.merge(low);
}

nb::object grid_spline_eval(
    nb::handle to,
    std::int64_t k,
    nb::handle co,
    nb::handle xo,
    nb::handle nuo,
    int rule,
    double fill_re,
    double fill_im
) {
    const std::complex<double> fill(fill_re, fill_im);
    if (k < 1) throw nb::value_error("'k' must be 1 or more");
    std::vector<std::int64_t> n;
    const std::vector<Axis> t = knots_arg(to, k, n);
    const std::int64_t d = static_cast<std::int64_t>(t.size());
    std::int64_t rows;
    const std::vector<std::int64_t> stride = strides_of(n, rows);
    const Values C = values_arg(co, rows, "c");
    const NDArray& xi = xi_arg(xo, d);
    const NDArray& nua = require_ndarray(nuo, "nu");
    if (nua.ndim != 1 || !nua.c_contiguous || nua.dtype != DType::Int64 || nua.shape[0] != d)
        throw nb::value_error("'nu' must be a C-contiguous (d,) int64 array");
    const auto* nup = reinterpret_cast<const std::int64_t*>(nua.data);
    const std::vector<std::int64_t> nu(nup, nup + d);
    for (std::int64_t v : nu)
        if (v < 0) throw nb::value_error("'nu' must be nonnegative");
    if (rule < 0 || rule > 2) throw nb::value_error("'rule' must be 0, 1 or 2");
    std::vector<double> dom_x(2 * d);
    std::vector<Axis> dom(d);
    for (std::int64_t a = 0; a < d; ++a) {
        dom_x[2 * a] = t[a].x[k];
        dom_x[2 * a + 1] = t[a].x[n[a]];
        dom[a] = {dom_x.data() + 2 * a, 2};
    }
    std::vector<double> fv(C.D);
    for (std::int64_t col = 0; col < C.K; ++col) {
        if (C.dtype == DType::Complex128) fv[2 * col] = fill.real(), fv[2 * col + 1] = fill.imag();
        else fv[col] = fill.real();
    }
    const std::int64_t m = xi.shape[0];
    NDArray out = new_values(C.dtype, m, C.K);
    if (m == 0) return pair(std::move(out), -1);
    std::int64_t per = 0;
    for (std::int64_t a = 0; a < d; ++a) per += log2_of(t[a].n) + 3 * (k + 1) * (k + 1);
    per += corners(k + 1, d) * std::max<std::int64_t>(C.D, 1) * 2;
    const SplineJob J{t,
                      dom,
                      n,
                      stride,
                      nu,
                      k,
                      C.v,
                      C.D,
                      reinterpret_cast<const double*>(xi.data),
                      reinterpret_cast<double*>(out.data),
                      rule,
                      fv};  // fmt: skip
    Bad bad;
    split_range(m, m * per, kBlock, [&](std::int64_t b, std::int64_t e) { spline_range(J, bad, b, e); });
    return pair(std::move(out), bad.result());
}

// ---------------------------------------------------------------- pchip

// The 1-D PCHIP of the nodes X (n >= 4) at s = x - X[j] in the interval j, from the values y of the window of four
// nodes w0 .. w0 + 3 (w0 = clamp(j - 1, 0, n - 4)): the slopes of cubic.cpp's pchip_slopes at j and j + 1 (the
// window holds every value they read), cubic.cpp's Hermite coefficients and ppoly.cpp's Horner, the same operations.
inline double pchip_1d(const double* X, std::int64_t n, const double* y, std::int64_t w0, std::int64_t j, double s) {
    auto m = [&](std::int64_t i) { return (y[i + 1 - w0] - y[i - w0]) / (X[i + 1] - X[i]); };
    auto slope = [&](std::int64_t i) {
        if (i == 0) return pchip_edge(X[1] - X[0], X[2] - X[1], m(0), m(1));
        if (i == n - 1) return pchip_edge(X[n - 1] - X[n - 2], X[n - 2] - X[n - 3], m(n - 2), m(n - 3));
        const double hl = X[i] - X[i - 1], hr = X[i + 1] - X[i];
        return pchip_interior(2.0 * hr + hl, hr + 2.0 * hl, m(i - 1), m(i));
    };
    const double s0 = slope(j), s1 = slope(j + 1);
    const double h = X[j + 1] - X[j];
    const double y0 = y[j - w0], y1 = y[j + 1 - w0];
    const double dd = (y1 - y0) / h;
    const double t = ((s0 + s1) - 2.0 * dd) / h;
    const double c3 = t / h, c2 = (dd - s0) / h - t;
    return std::fma(std::fma(std::fma(c3, s, c2), s, s0), s, y0);
}

struct PchipJob {
    const std::vector<Axis>& ax;
    const std::vector<std::int64_t>& stride;
    const double* v;
    std::int64_t K;
    const double* xi;
    double* out;
    int rule;
    double fill;
};

void pchip_range(const PchipJob& J, Bad& bad, std::int64_t begin, std::int64_t end) {
    const std::int64_t d = static_cast<std::int64_t>(J.ax.size()), K = J.K;
    const std::int64_t nc = corners(4, d);
    std::vector<char> use(d, 1);
    std::vector<std::int64_t> cnt(d * kBlock), off(nc), jj(d), ww(d);
    std::vector<double> ss(d), w(nc);
    std::vector<GallopState> state(d);
    std::int64_t low = std::numeric_limits<std::int64_t>::max();
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t mm = std::min(kBlock, end - b0);
        count_block(J.ax, use, J.xi, d, b0, mm, cnt.data(), state);
        for (std::int64_t i = 0; i < mm; ++i) {
            const double* p = J.xi + (b0 + i) * d;
            double* row = J.out + (b0 + i) * K;
            const int what = classify(p, J.ax, J.rule, low);
            if (what) {
                std::fill(row, row + K, what == 1 ? kNaN : J.fill);
                continue;
            }
            off[0] = 0;
            std::int64_t size = 1;
            for (std::int64_t a = 0; a < d; ++a) {
                const Axis& A = J.ax[a];
                const std::int64_t j = interval_of(cnt[a * kBlock + i], A.n);
                const std::int64_t w0 = std::clamp<std::int64_t>(j - 1, 0, A.n - 4);
                jj[a] = j, ww[a] = w0, ss[a] = p[a] - A.x[j];
                const std::int64_t s = J.stride[a];
                for (std::int64_t c = size - 1; c >= 0; --c)
                    for (std::int64_t r = 3; r >= 0; --r) off[c * 4 + r] = off[c] + (w0 + r) * s;
                size *= 4;
            }
            for (std::int64_t col = 0; col < K; ++col) {
                for (std::int64_t c = 0; c < nc; ++c) w[c] = J.v[off[c] * K + col];
                std::int64_t len = nc;
                for (std::int64_t a = d - 1; a >= 0; --a) {
                    const Axis& A = J.ax[a];
                    len /= 4;
                    for (std::int64_t g = 0; g < len; ++g)
                        w[g] = pchip_1d(A.x, A.n, w.data() + 4 * g, ww[a], jj[a], ss[a]);
                }
                row[col] = w[0];
            }
        }
    }
    bad.merge(low);
}

nb::object grid_pchip(nb::handle po, nb::handle vo, nb::handle xo, int rule, double fill) {
    const std::vector<Axis> ax = axes_arg(po, "points");
    const std::int64_t d = static_cast<std::int64_t>(ax.size());
    std::vector<std::int64_t> n(d);
    for (std::int64_t a = 0; a < d; ++a) {
        n[a] = ax[a].n;
        if (n[a] < 4) throw nb::value_error("'points' must have 4 nodes or more per axis");
    }
    std::int64_t rows;
    const std::vector<std::int64_t> stride = strides_of(n, rows);
    const Values V = values_arg(vo, rows);
    if (V.dtype != DType::Float64) throw nb::type_error("'values' must be float64");
    const NDArray& xi = xi_arg(xo, d);
    if (rule < 0 || rule > 2) throw nb::value_error("'rule' must be 0, 1 or 2");
    const std::int64_t m = xi.shape[0];
    NDArray out = new_values(V.dtype, m, V.K);
    if (m == 0) return pair(std::move(out), -1);
    std::int64_t per = 0;
    for (std::int64_t a = 0; a < d; ++a) per += log2_of(n[a]) + 8;
    per += corners(4, d) * std::max<std::int64_t>(V.K, 1) * 20;
    const PchipJob J{
        ax,   stride, V.v, V.K, reinterpret_cast<const double*>(xi.data), reinterpret_cast<double*>(out.data),
        rule, fill
    };  // fmt: skip
    Bad bad;
    split_range(m, m * per, kBlock, [&](std::int64_t b, std::int64_t e) { pchip_range(J, bad, b, e); });
    return pair(std::move(out), bad.result());
}

}  // namespace

void bind_grid(nb::module_& m) {
    m.def(
        "grid_linear", &grid_linear, nb::arg("points"), nb::arg("values").none(), nb::arg("xi").none(),
        nb::arg("nearest"), nb::arg("rule"), nb::arg("fill_re"), nb::arg("fill_im")
    );  // fmt: skip
    m.def(
        "grid_spline_coefs", &grid_spline_coefs, nb::arg("points"), nb::arg("knots"), nb::arg("k"),
        nb::arg("values").none()
    );  // fmt: skip
    m.def(
        "grid_spline_eval", &grid_spline_eval, nb::arg("knots"), nb::arg("k"), nb::arg("c").none(),
        nb::arg("xi").none(), nb::arg("nu").none(), nb::arg("rule"), nb::arg("fill_re"), nb::arg("fill_im")
    );  // fmt: skip
    m.def(
        "grid_pchip", &grid_pchip, nb::arg("points"), nb::arg("values").none(), nb::arg("xi").none(), nb::arg("rule"),
        nb::arg("fill")
    );  // fmt: skip
}

}  // namespace sd::interpolate
