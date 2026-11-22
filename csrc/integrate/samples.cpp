// The sample rules trapezoid, simpson, romb, cumulative_trapezoid and cumulative_simpson, and the
// weighted sum of fixed_quad (docs/design/integrate.md §3.7; scipy 1.18.1's
// integrate/_quadrature.py).
//
// Each rule works lane by lane: a lane is y along the axis (and x, the spacing dx and the initial
// value of that lane). The terms are scipy's expressions, in scipy's order of operations; the
// sum of a lane's terms is the library's pairwise sum (base/pairwise.hpp: 16 accumulators,
// blocks of 512, the tree of sd.sum along a contiguous axis) in the index order, and a cumulative
// rule's running sum is cumulative_sum's scan (array/kernels/reduce.hpp, scan_lane). So a lane's
// value depends on its numbers only: not on the axis, the memory layout or the thread count.
//
// A complex y is scaled by the real coefficients part by part (double * complex), where numpy
// promotes the coefficient to complex first; the two differ only when a part is infinite or NaN.
//
// Threads: lanes in parallel once the work (lanes x length) reaches the array's threshold
// (2^15); a single long lane sums its terms in parallel subtrees of the same tree
// (pairwise_sum_mt) and scans in cumulative_sum's blocks. Work at or above the threshold runs
// without the GIL (API.md §3).
#include "integrate/integrate.hpp"

#include "array/kernels/reduce.hpp"
#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "base/pairwise.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"

#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace sd::integrate {

namespace nb = nanobind;

namespace {

enum Rule : int {
    kTrapezoid = 0,
    kSimpson = 1,
    kRomb = 2,
    kCumulativeTrapezoid = 3,
    kCumulativeSimpson = 4,
    kWeighted = 5,  // fixed_quad: sum of x_i y_i, x holding the weights
};

bool reduces(int rule) { return rule != kCumulativeTrapezoid && rule != kCumulativeSimpson; }

// One lane's operands. x is null without x; dx and init are read through their pointer when the
// argument is an array, else the scalar.
struct Lane {
    const std::byte* y;
    const std::byte* x;
    const std::byte* dx;
    const std::byte* init;
    std::byte* o;
};

// Where every lane starts: the axes other than `axis` (in order) with each operand's strides.
struct Lanes {
    std::int64_t n = 0;                   // y's length along the axis
    std::int64_t ys = 0, xs = 0, os = 0;  // strides along the axis (out: cumulative rules)
    int nd = 0;                           // the other axes
    std::int64_t shape[kMaxNdim]{}, sy[kMaxNdim]{}, sx[kMaxNdim]{}, sdx[kMaxNdim]{}, si[kMaxNdim]{}, so[kMaxNdim]{};
    std::int64_t count = 1;
    Lane base{};

    Lane at(std::int64_t l) const {
        Lane p = base;
        for (int d = nd - 1; d >= 0; --d) {
            const std::int64_t i = l % shape[d];
            l /= shape[d];
            p.y += i * sy[d];
            if (p.x) p.x += i * sx[d];
            if (p.dx) p.dx += i * sdx[d];
            if (p.init) p.init += i * si[d];
            p.o += i * so[d];
        }
        return p;
    }
};

template <typename T>
inline T ld(const std::byte* p, std::int64_t stride, std::int64_t i) {
    return load_as<T>(p + i * stride);
}

template <typename T, typename Get>
T lane_sum(std::int64_t n, const Get& get, bool mt) {
    if constexpr (std::is_same_v<T, double>) return mt ? pairwise_sum_mt(n, get) : pairwise_sum(0, n, get);
    else return mt ? pairwise_sum_complex_mt(n, get) : pairwise_sum_complex(0, n, get);
}

template <typename T>
struct Kernel {
    const Lanes& L;
    double dx_scalar;
    T init_scalar;
    bool has_init;
    int rule;

    double dx(const Lane& p) const { return p.dx ? load_as<double>(p.dx) : dx_scalar; }
    T init(const Lane& p) const { return p.init ? load_as<T>(p.init) : init_scalar; }
    T y(const Lane& p, std::int64_t i) const { return ld<T>(p.y, L.ys, i); }
    double x(const Lane& p, std::int64_t i) const { return ld<double>(p.x, L.xs, i); }
    // the spacing between points i and i + 1: x's difference, else dx
    double h(const Lane& p, std::int64_t i, double d) const { return p.x ? x(p, i + 1) - x(p, i) : d; }

    // trapezoid's term i: d (y[i + 1] + y[i]) / 2.0
    T trapezoid_term(const Lane& p, std::int64_t i, double d) const {
        return h(p, i, d) * (y(p, i + 1) + y(p, i)) / 2.0;
    }

    // scipy's _basic_simpson over the pairs of intervals starting at 0, 2, ..., below `stop`
    T simpson_basic(const Lane& p, std::int64_t stop, double d, bool mt) const {
        const std::int64_t pairs = stop > 0 ? (stop + 1) / 2 : 0;
        if (!p.x) {
            if constexpr (std::is_same_v<T, double>) {
                if (L.ys == static_cast<std::int64_t>(sizeof(double))) {  // typed loads (the same bits)
                    const double* yp = reinterpret_cast<const double*>(p.y);
                    const T s = lane_sum<T>(
                        pairs, [yp](std::int64_t j) { return yp[2 * j] + 4.0 * yp[2 * j + 1] + yp[2 * j + 2]; }, mt
                    );
                    return d / 3.0 * s;
                }
            }
            const T s = lane_sum<T>(
                pairs, [&](std::int64_t j) { return y(p, 2 * j) + 4.0 * y(p, 2 * j + 1) + y(p, 2 * j + 2); }, mt
            );
            return d / 3.0 * s;
        }
        return lane_sum<T>(
            pairs,
            [&](std::int64_t j) {
                const std::int64_t k = 2 * j;
                const double h0 = x(p, k + 1) - x(p, k), h1 = x(p, k + 2) - x(p, k + 1);
                const double hsum = h0 + h1, hprod = h0 * h1;
                const double h0divh1 = h1 != 0 ? h0 / h1 : 0.;
                const double c0 = 2.0 - (h0divh1 != 0 ? 1.0 / h0divh1 : 0.);
                const double c1 = hsum * (hprod != 0 ? hsum / hprod : 0.);
                const double c2 = 2.0 - h0divh1;
                return hsum / 6.0 * (y(p, k) * c0 + y(p, k + 1) * c1 + y(p, k + 2) * c2);
            },
            mt
        );
    }

    T simpson(const Lane& p, bool mt) const {
        const std::int64_t n = L.n;
        const double d = dx(p);
        if (n <= 1) return T(0);
        if (n % 2 == 1) return simpson_basic(p, n - 2, d, mt);
        if (n == 2) {
            const T val = 0.5 * h(p, 0, d) * (y(p, 1) + y(p, 0));
            return T(0) + val;
        }
        T result = simpson_basic(p, n - 3, d, mt);
        // the last interval by Cartwright's correction (scipy 1.11's form)
        const double h0 = h(p, n - 3, d), h1 = h(p, n - 2, d);
        double num = 2 * (h1 * h1) + 3 * h0 * h1, den = 6 * (h1 + h0);
        const double alpha = den != 0 ? num / den : 0.;
        num = h1 * h1 + 3.0 * h0 * h1;
        den = 6 * h0;
        const double beta = den != 0 ? num / den : 0.;
        num = 1 * std::pow(h1, 3.0);
        den = 6 * h0 * (h0 + h1);
        const double eta = den != 0 ? num / den : 0.;
        result = result + (alpha * y(p, n - 1) + beta * y(p, n - 2) - eta * y(p, n - 3));
        return result + T(0);
    }

    T romb(const Lane& p, bool mt) const {
        const std::int64_t intervals = L.n - 1;  // a power of 2 (checked by the caller)
        int k = 0;
        while ((std::int64_t{1} << k) < intervals) ++k;
        double hh = static_cast<double>(intervals) * dx(p);
        std::vector<T> prev(static_cast<std::size_t>(k) + 1), cur(prev.size());
        prev[0] = (y(p, 0) + y(p, intervals)) / 2.0 * hh;
        std::int64_t start = intervals, step = intervals;
        for (int i = 1; i <= k; ++i) {
            start >>= 1;
            const std::int64_t terms = (intervals - start + step - 1) / step;
            const std::int64_t s0 = start, st = step;
            const T s = lane_sum<T>(terms, [&](std::int64_t j) { return y(p, s0 + j * st); }, mt);
            step >>= 1;
            cur[0] = 0.5 * (prev[0] + hh * s);
            for (int j = 1; j <= i; ++j) {
                const T pr = cur[static_cast<std::size_t>(j) - 1];
                cur[static_cast<std::size_t>(j)] =
                    pr + (pr - prev[static_cast<std::size_t>(j) - 1]) / (std::ldexp(1.0, 2 * j) - 1.0);
            }
            hh /= 2.0;
            std::swap(prev, cur);
        }
        return prev[static_cast<std::size_t>(k)];
    }

    T weighted(const Lane& p, bool mt) const {
        if constexpr (std::is_same_v<T, double>) {
            if (L.ys == static_cast<std::int64_t>(sizeof(double)) &&
                L.xs == static_cast<std::int64_t>(sizeof(double))) {
                const double* yp = reinterpret_cast<const double*>(p.y);
                const double* xp = reinterpret_cast<const double*>(p.x);
                return lane_sum<T>(L.n, [xp, yp](std::int64_t i) { return xp[i] * yp[i]; }, mt);
            }
        }
        return lane_sum<T>(L.n, [&](std::int64_t i) { return x(p, i) * y(p, i); }, mt);
    }

    T reduce(const Lane& p, bool mt) const {
        switch (rule) {
            case kTrapezoid: {
                const double d = dx(p);
                const std::int64_t m = std::max<std::int64_t>(L.n - 1, 0);
                if constexpr (std::is_same_v<T, double>) {
                    // contiguous float64: typed loads, so the pairwise leaf's chains load vectors
                    // (the same terms in the same tree: the same bits)
                    if (L.ys == static_cast<std::int64_t>(sizeof(double))) {
                        const double* yp = reinterpret_cast<const double*>(p.y);
                        if (!p.x)
                            return lane_sum<T>(
                                m, [yp, d](std::int64_t i) { return d * (yp[i + 1] + yp[i]) / 2.0; }, mt
                            );
                        if (L.xs == static_cast<std::int64_t>(sizeof(double))) {
                            const double* xp = reinterpret_cast<const double*>(p.x);
                            return lane_sum<T>(
                                m, [yp, xp](std::int64_t i) { return (xp[i + 1] - xp[i]) * (yp[i + 1] + yp[i]) / 2.0; },
                                mt
                            );
                        }
                    }
                }
                return lane_sum<T>(m, [&](std::int64_t i) { return trapezoid_term(p, i, d); }, mt);
            }
            case kSimpson: return simpson(p, mt);
            case kRomb: return romb(p, mt);
            default: return weighted(p, mt);
        }
    }

    // cumulative_simpson's integral over interval k from three points (Cartwright's formulas,
    // scipy's _cumulative_simpson_{equal,unequal}_intervals): `left` uses points k, k + 1, k + 2
    // (scipy's h1 sub-integrals), else points k - 1, k, k + 1 (h2, from the reversed arrays).
    T simpson_piece(const Lane& p, std::int64_t k, bool left, double d) const {
        T f1, f2, f3;
        double x21, x32;
        if (left) {
            f1 = y(p, k), f2 = y(p, k + 1), f3 = y(p, k + 2);
            x21 = h(p, k, d), x32 = h(p, k + 1, d);
        } else {
            f1 = y(p, k + 1), f2 = y(p, k), f3 = y(p, k - 1);
            x21 = h(p, k, d), x32 = h(p, k - 1, d);
        }
        if (!p.x) return x21 / 3 * (5.0 * f1 / 4.0 + 2.0 * f2 - f3 / 4.0);
        const double x31 = x21 + x32;
        const double x21_x31 = x21 / x31, x21_x32 = x21 / x32;
        const double x21x21_x31x32 = x21_x31 * x21_x32;
        const double c1 = 3 - x21_x31, c2 = 3 + x21x21_x31x32 + x21_x31, c3 = -x21x21_x31x32;
        return x21 / 6 * (c1 * f1 + c2 * f2 + c3 * f3);
    }

    // the cumulative rules' term k (the integral over interval k, before the running sum)
    T piece(const Lane& p, std::int64_t k, double d) const {
        const std::int64_t n = L.n;
        if (rule == kCumulativeTrapezoid || n < 3) return trapezoid_term(p, k, d);
        const bool left = k % 2 == 0 && k != n - 2;  // the last interval only has a right formula
        return simpson_piece(p, k, left, d);
    }

    void cumulative(const Lane& p, bool mt) const {
        const std::int64_t m = L.n - 1;  // the intervals
        const double d = dx(p);
        std::byte* first = has_init ? p.o + L.os : p.o;
        auto put = [&](std::int64_t k) { *reinterpret_cast<T*>(first + k * L.os) = piece(p, k, d); };
        if (mt && m >= kParallelThreshold) {
            const std::int64_t tasks = (m + kParallelTask - 1) / kParallelTask;
            parallel::parallel_for(
                tasks,
                [&](std::int64_t t) {
                    const std::int64_t end = std::min(m, (t + 1) * kParallelTask);
                    for (std::int64_t k = t * kParallelTask; k < end; ++k) put(k);
                },
                kParallelTask  // about 1 ns per interval
            );
        } else {
            for (std::int64_t k = 0; k < m; ++k) put(k);
        }
        if (m > 0) scan_lane<ops::Add, T>(first, L.os, first, L.os, m);
        if (!has_init) return;
        const T v = init(p);
        *reinterpret_cast<T*>(p.o) = v;
        if (rule == kCumulativeSimpson)  // scipy adds the initial value to every integral
            for (std::int64_t k = 0; k < m; ++k) {
                T* q = reinterpret_cast<T*>(first + k * L.os);
                *q = *q + v;
            }
    }
};

// The lanes of y along `axis`, with the other operands' strides (x: y's shape; dx and init: y's
// shape with extent 1 along the axis; out: the result).
Lanes lanes_of(
    const NDArray& y,
    const NDArray* x,
    const NDArray* dx,
    const NDArray* init,
    const NDArray& out,
    int axis,
    bool cumulative
) {
    Lanes L;
    L.n = y.shape[axis];
    L.ys = y.strides[axis];
    L.xs = x ? x->strides[axis] : 0;
    L.os = cumulative ? out.strides[axis] : 0;
    L.base = {y.data, x ? x->data : nullptr, dx ? dx->data : nullptr, init ? init->data : nullptr, out.data};
    int od = 0;
    for (int d = 0; d < y.ndim; ++d) {
        if (d == axis) continue;
        L.shape[L.nd] = y.shape[d];
        L.sy[L.nd] = y.strides[d];
        L.sx[L.nd] = x ? x->strides[d] : 0;
        L.sdx[L.nd] = dx ? dx->strides[d] : 0;
        L.si[L.nd] = init ? init->strides[d] : 0;
        L.so[L.nd] = cumulative ? out.strides[d] : (out.ndim ? out.strides[od] : 0);
        ++od;
        L.count *= y.shape[d];
        ++L.nd;
    }
    return L;
}

// trapezoid on lanes that lie side by side in memory (the reduced axis outer, as axis=0 of a
// C-ordered matrix): a block of `w` adjacent lanes is summed row by row, each lane through the
// very same pairwise tree and leaf as pairwise_sum (base/pairwise.hpp: the 16 chains, their fold,
// the tail, the splits), so every lane has the bits of the lane-at-a-time sum; the rows are read
// contiguously across the block instead of one strided element per lane.
struct SideBySide {
    const std::byte* y;  // lane 0's first element
    std::int64_t ys;     // bytes between rows (the reduced axis)
    std::int64_t w;      // lanes in the block (adjacent doubles)
    double d;

    double term(std::int64_t i, std::int64_t l) const {
        const double* r0 = reinterpret_cast<const double*>(y + i * ys);
        const double* r1 = reinterpret_cast<const double*>(y + (i + 1) * ys);
        return d * (r1[l] + r0[l]) / 2.0;
    }
    // out[l] = the pairwise leaf of terms [b, b + m), m <= kBlock
    void leaf(std::int64_t b, std::int64_t m, double* out, double* r) const {
        constexpr int K = pairwise::kLanes;
        if (m < K) {
            for (std::int64_t l = 0; l < w; ++l) out[l] = 0.;
            for (std::int64_t i = 0; i < m; ++i) {
                const double* r0 = reinterpret_cast<const double*>(y + (b + i) * ys);
                const double* r1 = reinterpret_cast<const double*>(y + (b + i + 1) * ys);
                for (std::int64_t l = 0; l < w; ++l) out[l] += d * (r1[l] + r0[l]) / 2.0;
            }
            return;
        }
        // r[j * w + l]: chain j of lane l
        for (int j = 0; j < K; ++j) {
            const double* r0 = reinterpret_cast<const double*>(y + (b + j) * ys);
            const double* r1 = reinterpret_cast<const double*>(y + (b + j + 1) * ys);
            double* rj = r + j * w;
            for (std::int64_t l = 0; l < w; ++l) rj[l] = d * (r1[l] + r0[l]) / 2.0;
        }
        std::int64_t i = K;
        for (; i + K <= m; i += K)
            for (int j = 0; j < K; ++j) {
                const double* r0 = reinterpret_cast<const double*>(y + (b + i + j) * ys);
                const double* r1 = reinterpret_cast<const double*>(y + (b + i + j + 1) * ys);
                double* rj = r + j * w;
                for (std::int64_t l = 0; l < w; ++l) rj[l] += d * (r1[l] + r0[l]) / 2.0;
            }
        for (int st = 1; st < K; st *= 2)  // fold_lanes, lane by lane
            for (int j = 0; j < K; j += 2 * st) {
                double* a = r + j * w;
                const double* c = r + (j + st) * w;
                for (std::int64_t l = 0; l < w; ++l) a[l] += c[l];
            }
        for (std::int64_t l = 0; l < w; ++l) out[l] = r[l];
        for (; i < m; ++i) {
            const double* r0 = reinterpret_cast<const double*>(y + (b + i) * ys);
            const double* r1 = reinterpret_cast<const double*>(y + (b + i + 1) * ys);
            for (std::int64_t l = 0; l < w; ++l) out[l] += d * (r1[l] + r0[l]) / 2.0;
        }
    }
    // pairwise::tree over terms [b, b + m), lane by lane; scratch: w * (16 + 2 * depth) doubles
    void tree(std::int64_t b, std::int64_t m, double* out, double* scratch) const {
        if (m <= pairwise::kBlock) {
            leaf(b, m, out, scratch);
            return;
        }
        const std::int64_t m2 = pairwise::split(m);
        double* left = scratch;
        double* right = scratch + w;
        tree(b, m2, left, scratch + 2 * w);
        tree(b + m2, m - m2, right, scratch + 2 * w);
        for (std::int64_t l = 0; l < w; ++l) out[l] = left[l] + right[l];
    }
};

// Whether trapezoid can run on blocks of side-by-side lanes: float64, no x, a scalar dx, the
// lanes' innermost axis contiguous in y and in the result, and more than one lane there.
bool side_by_side(const Lanes& L, int rule, bool has_x, bool has_dx_array) {
    return rule == kTrapezoid && !has_x && !has_dx_array && L.nd >= 1 && L.n >= 2 &&
           L.sy[L.nd - 1] == static_cast<std::int64_t>(sizeof(double)) &&
           L.so[L.nd - 1] == static_cast<std::int64_t>(sizeof(double)) && L.shape[L.nd - 1] >= 2;
}

template <typename T>
nb::object run(
    int rule,
    const NDArray& y,
    const NDArray* x,
    double dx_scalar,
    const NDArray* dx,
    T init_scalar,
    const NDArray* init,
    bool has_init,
    int axis
) {
    const bool cumulative = !reduces(rule);
    const std::int64_t n = y.shape[axis];
    std::vector<std::int64_t> shape;
    for (int d = 0; d < y.ndim; ++d) {
        if (d != axis) shape.push_back(y.shape[d]);
        else if (cumulative) shape.push_back(n - 1 + (has_init ? 1 : 0));
    }
    const bool scalar = shape.empty();
    NDArray out;
    T one{};
    if (!scalar) out = new_array(dtype_of<T>, shape);
    else out.data = reinterpret_cast<std::byte*>(&one);
    const Lanes L = lanes_of(y, x, dx, init, out, axis, cumulative);
    const Kernel<T> K{L, dx_scalar, init_scalar, has_init, rule};
    const std::int64_t work = L.count * std::max<std::int64_t>(n, 1);
    if constexpr (std::is_same_v<T, double>) {
        if (side_by_side(L, rule, x != nullptr, dx != nullptr)) {
            // blocks of up to kSideLanes adjacent lanes within each run of the innermost axis
            // 64 lanes: one 512-byte run per row, and 16 blocks (tasks) for 1000 lanes
            constexpr std::int64_t kSideLanes = 64;
            const std::int64_t inner = L.shape[L.nd - 1];
            const std::int64_t runs = L.count / inner;
            const std::int64_t per_run = (inner + kSideLanes - 1) / kSideLanes;
            const std::int64_t blocks = runs * per_run;
            int depth = 0;
            for (std::int64_t m = n - 1; m > pairwise::kBlock; m = m - pairwise::split(m)) ++depth;
            const std::size_t scratch = static_cast<std::size_t>(kSideLanes * (pairwise::kLanes + 2 * depth + 2));
            auto block = [&](std::int64_t bi, std::vector<double>& buf) {
                const std::int64_t run = bi / per_run, l0 = (bi % per_run) * kSideLanes;
                const Lane p = L.at(run * inner + l0);
                const SideBySide S{p.y, L.ys, std::min(kSideLanes, inner - l0), dx_scalar};
                S.tree(0, n - 1, reinterpret_cast<double*>(p.o), buf.data());
            };
            {
                // the GIL comes back at the end of this scope: wrap() below makes a Python object
                parallel::Unlocked unlocked(work >= kParallelThreshold);
                if (blocks >= 2 && work >= kParallelThreshold && parallel::available()) {
                    parallel::parallel_for(
                        blocks,
                        [&](std::int64_t bi) {
                            std::vector<double> buf(scratch);
                            block(bi, buf);
                        },
                        kSideLanes * (n - 1) / 4  // about 0.25 ns per term
                    );
                } else {
                    std::vector<double> buf(scratch);
                    for (std::int64_t bi = 0; bi < blocks; ++bi) block(bi, buf);
                }
            }
            return wrap(std::move(out));
        }
    }
    {
        parallel::Unlocked unlocked(work >= kParallelThreshold);
        auto lane = [&](std::int64_t l, bool mt) {
            const Lane p = L.at(l);
            if (cumulative) K.cumulative(p, mt);
            else *reinterpret_cast<T*>(p.o) = K.reduce(p, mt);
        };
        if (L.count >= 2 && work >= kParallelThreshold && parallel::available()) {
            const std::int64_t per = std::max<std::int64_t>(1, kParallelTask / std::max<std::int64_t>(n, 1));
            const std::int64_t tasks = (L.count + per - 1) / per;
            parallel::parallel_for(
                tasks,
                [&](std::int64_t t) {
                    const std::int64_t end = std::min(L.count, (t + 1) * per);
                    for (std::int64_t l = t * per; l < end; ++l) lane(l, false);
                },
                per * std::max<std::int64_t>(n, 1) / 2  // about 0.5 ns per term (reductions 0.2, cumulative 1.7)
            );
        } else {
            for (std::int64_t l = 0; l < L.count; ++l) lane(l, L.count == 1);
        }
    }
    if (scalar) return nb::cast(one);
    return wrap(std::move(out));
}

const NDArray* optional_array(nb::handle o) { return is_ndarray(o) ? &as_ndarray(o) : nullptr; }

// samples(func, rule, y, x, dx, axis, initial): the rule on y (float64 or complex128) along
// `axis` (normalized). x: None or float64 of y's shape (a broadcast view); dx: a float, or float64
// of y's shape with extent 1 along the axis; initial: None, a number, or y's dtype with that
// lane shape. Arguments are checked by stardust/integrate/_samples.py.
nb::object samples(
    const std::string& func,
    int rule,
    nb::handle y_obj,
    nb::handle x_obj,
    nb::handle dx_obj,
    int axis,
    nb::handle init_obj
) {
    return call_named(func.c_str(), [&]() -> nb::object {
        const NDArray& y = require_ndarray(y_obj, "y");
        const NDArray* x = optional_array(x_obj);
        const NDArray* dx = optional_array(dx_obj);
        const double dx_scalar = dx || dx_obj.is_none() ? 1.0 : nb::cast<double>(dx_obj);
        const NDArray* init = optional_array(init_obj);
        const bool has_init = !init_obj.is_none();
        if (y.dtype == DType::Complex128) {
            const complex128 iv =
                has_init && !init
                    ? complex128(nb::cast<double>(init_obj.attr("real")), nb::cast<double>(init_obj.attr("imag")))
                    : complex128(0);
            return run<complex128>(rule, y, x, dx_scalar, dx, iv, init, has_init, axis);
        }
        const double iv = has_init && !init ? nb::cast<double>(init_obj) : 0.;
        return run<double>(rule, y, x, dx_scalar, dx, iv, init, has_init, axis);
    });
}

}  // namespace

void bind_samples(nb::module_& m) {
    m.def(
        "samples", &samples, nb::arg("func"), nb::arg("rule"), nb::arg("y"), nb::arg("x").none(), nb::arg("dx").none(),
        nb::arg("axis"), nb::arg("initial").none()
    );
}

}  // namespace sd::integrate
