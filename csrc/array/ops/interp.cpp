// interp: numpy's piecewise linear interpolation (docs/api/array.md §8.5; docs/design/interpolate.md §2.2,
// §3.1, §3.2; decisions IP2, IP5, IP37, IP39).
//
// The value between two nodes is base/lerp.hpp's kernel, the one every linear path of sd.interpolate uses,
// and the interval is base/interval.hpp's count of nodes at or below the point. Each point is computed
// alone, so the result does not depend on how the points are split across threads (tasks by size only).

#include "array/ops/interp.hpp"

#include "array/convert.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/ops/copy.hpp"
#include "array/parallel_limits.hpp"
#include "base/interval.hpp"
#include "base/lerp.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace sd {

namespace {

// A 1-D argument as a C-contiguous array of dtype `d` (the array itself when it is one already).
NDArray contiguous_as(const NDArray& a, DType d) {
    if (a.dtype == d && a.c_contiguous) return a;
    NDArray out = new_array(d, a.shape_span());
    if (out.size() > 0) assign(out, a);
    return out;
}

// The polyline: nodes xp (nondecreasing, finite), values fp, and the values left of xp[0] and right of
// xp[n - 1]. With a period the points are first wrapped into [0, period].
template <typename T>
struct Line {
    const double* xp;
    const T* fp;
    std::int64_t n;
    T left, right;
    double period;  // 0: none
};

// The value at x given c, the number of nodes <= x (numpy's binary_search_with_guess + 1): left of
// everything, at or right of the last node, or inside the interval [xp[c - 1], xp[c]) of positive width
// (a repeated node is a step: at the node, the value on its right).
template <typename T>
inline T value_at(const Line<T>& L, double x, std::int64_t c) {
    if (x != x) {  // NaN: NaN (numpy: NaN + 0j for complex values; decision IP39 for one node)
        if constexpr (std::is_same_v<T, double>) return x;
        else return T(x, 0.0);
    }
    if (c == 0) return L.left;
    if (c == L.n) return x == L.xp[L.n - 1] ? L.fp[L.n - 1] : L.right;
    const std::int64_t j = c - 1;
    return lerp(L.fp[j], L.fp[j + 1], lerp_t(x, L.xp[j], L.xp[j + 1]));
}

// numpy's x % period (npy_divmod's remainder): in [0, period] for a positive period
inline double wrap_period(double x, double period) { return ops::Remainder::f(x, period); }

// out[i] for the points x[begin, end), in blocks of kBlock (base/interval.hpp's count_le_block: sorted
// blocks gallop, others search in lanes; either gives the same count, so the path only changes the time).
template <typename T>
void eval_range(const Line<T>& L, const double* x, T* out, std::int64_t begin, std::int64_t end) {
    constexpr std::int64_t kBlock = 256;
    double v[kBlock];
    std::int64_t c[kBlock];
    GallopState state;
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t m = std::min(kBlock, end - b0);
        if (L.period > 0.0)
            for (std::int64_t i = 0; i < m; ++i) v[i] = wrap_period(x[b0 + i], L.period);
        else std::copy(x + b0, x + b0 + m, v);
        count_le_block(L.xp, L.n, v, m, c, state);
        if constexpr (std::is_same_v<T, double>) {
            if (L.n >= 2) {
                // The interval's nodes and values side by side, then base/lerp.hpp's block kernel (the same bits as
                // value_at); the points left of everything, at or right of the last node, and those the block
                // kernel leaves out (NaN, infinite or non-finite values) take value_at.
                double x0[kBlock], x1[kBlock], a[kBlock], b[kBlock];
                bool inside = true;
                for (std::int64_t i = 0; i < m; ++i) {
                    inside &= (c[i] != 0) & (c[i] != L.n);
                    const std::int64_t j = std::clamp<std::int64_t>(c[i] - 1, 0, L.n - 2);
                    x0[i] = L.xp[j], x1[i] = L.xp[j + 1], a[i] = L.fp[j], b[i] = L.fp[j + 1];
                }
                T* o = out + b0;
                if (!(lerp_block(v, x0, x1, a, b, o, m) & inside))
                    for (std::int64_t i = 0; i < m; ++i)
                        if (c[i] == 0 || c[i] == L.n || !lerp_block_covers(v[i], x0[i], x1[i], a[i], b[i]))
                            o[i] = value_at(L, v[i], c[i]);
                continue;
            }
        }
        for (std::int64_t i = 0; i < m; ++i) out[b0 + i] = value_at(L, v[i], c[i]);
    }
}

// One value of the line (a Python scalar x), held to the same rules.
template <typename T>
T eval_one(const Line<T>& L, double x) {
    const double v = L.period > 0.0 ? wrap_period(x, L.period) : x;
    std::int64_t c = 0;
    count_le_lanes<1>(L.xp, L.n, &v, &c);
    return value_at(L, v, c);
}

template <typename T>
void eval_all(const Line<T>& L, const double* x, T* out, std::int64_t m) {
    const std::int64_t log_n = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::log2(L.n + 1)));
    // the search and the kernel, ~ns x 6 per point (measured: 4 ns a point in random order among 10^3 nodes), in
    // tasks of 1024 points: 5000 points then wake the pool (decision IP102)
    const std::int64_t work = m * (2 * log_n + 8);
    constexpr std::int64_t kTask = 1024;
    if (work >= kParallelThreshold && parallel::available()) {
        const std::int64_t tasks = (m + kTask - 1) / kTask;
        parallel::parallel_for(
            tasks, [&](std::int64_t t) { eval_range(L, x, out, t * kTask, std::min(m, (t + 1) * kTask)); },
            work_ns(work / tasks)
        );
    } else {
        const parallel::Unlocked unlocked(work >= kParallelThreshold);
        eval_range(L, x, out, 0, m);
    }
}

// left= / right=: a number of fp's kind (real values take an int or a float; complex values also a complex)
template <typename T>
T parse_end(nb::handle o, const char* what, T dflt) {
    if (o.is_none()) return dflt;
    if (std::optional<double> v = real_value(o, what)) return T(*v);
    if constexpr (std::is_same_v<T, complex128>) {
        if (scalar_dtype(o) == DType::Complex128) return scalar_from_python<complex128>(o, DType::Complex128);
        throw nb::type_error(("'" + std::string(what) + "' must be a number, not " + py_type_name(o)).c_str());
    } else {
        throw nb::type_error(("'" + std::string(what) + "' must be an int or a float, not " + py_type_name(o)).c_str());
    }
}

// The line of xp, fp (float64 and fp's dtype, contiguous) with numpy's periodic normalisation: xp wrapped
// into [0, period], sorted (stable: nodes that wrap onto each other keep their order) and extended by one
// node on each side.
template <typename T>
struct Periodic {
    std::vector<double> xp;
    std::vector<T> fp;
    Periodic(const double* x, const T* f, std::int64_t n, double period) : xp(n + 2), fp(n + 2) {
        std::vector<double> w(n);
        for (std::int64_t i = 0; i < n; ++i) w[i] = wrap_period(x[i], period);
        std::vector<std::int64_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) { return w[a] < w[b]; });
        for (std::int64_t i = 0; i < n; ++i) {
            xp[i + 1] = w[order[i]];
            fp[i + 1] = f[order[i]];
        }
        xp[0] = xp[n] - period;
        fp[0] = fp[n];
        xp[n + 1] = xp[1] + period;
        fp[n + 1] = fp[1];
    }
};

template <typename T>
nb::object interp_typed(
    const NDArray* xa,
    double x_scalar,
    const NDArray& xp,
    const NDArray& fp,
    nb::handle left_obj,
    nb::handle right_obj,
    double period
) {
    const std::int64_t n = xp.shape[0];
    const double* xpd = reinterpret_cast<const double*>(xp.data);
    const T* fpd = reinterpret_cast<const T*>(fp.data);
    std::optional<Periodic<T>> per;
    Line<T> L{xpd, fpd, n, fpd[0], fpd[n - 1], period};
    if (period > 0.0) {
        per.emplace(xpd, fpd, n, period);
        L = Line<T>{per->xp.data(), per->fp.data(), n + 2, per->fp.front(), per->fp.back(), period};
    } else {
        L.left = parse_end<T>(left_obj, "left", fpd[0]);
        L.right = parse_end<T>(right_obj, "right", fpd[n - 1]);
    }
    if (xa == nullptr) return to_python(eval_one(L, x_scalar));
    NDArray out = new_array(dtype_of<T>, xa->shape_span());
    const std::int64_t m = out.size();
    if (m > 0) eval_all(L, reinterpret_cast<const double*>(xa->data), reinterpret_cast<T*>(out.data), m);
    return wrap(std::move(out));
}

nb::object interp(
    nb::handle x_obj,
    nb::handle xp_obj,
    nb::handle fp_obj,
    nb::handle left_obj,
    nb::handle right_obj,
    nb::handle period_obj
) {
    // x: an array or a Python scalar, real
    NDArray xc;
    double x_scalar = 0.0;
    const bool scalar = !is_ndarray(x_obj);
    if (scalar) {
        const std::optional<DType> d = scalar_dtype(x_obj);
        if (!d)
            throw nb::type_error(
                ("'x' must be an sd.ndarray or a bool, int or float, not " + py_type_name(x_obj)).c_str()
            );
        if (*d == DType::Complex128) throw nb::type_error("'x' must be real, not complex");
        x_scalar = scalar_from_python<double>(x_obj, *d);
    } else {
        const NDArray& xa = as_ndarray(x_obj);
        if (xa.dtype == DType::Complex128) throw nb::type_error("'x' must be real, not complex128");
    }
    // xp and fp: 1-D arrays of one length, xp real
    const NDArray& xp0 = require_ndarray(xp_obj, "xp");
    const NDArray& fp0 = require_ndarray(fp_obj, "fp");
    if (xp0.ndim != 1) throw nb::value_error("'xp' must be 1-D");
    if (fp0.ndim != 1) throw nb::value_error("'fp' must be 1-D");
    if (xp0.dtype == DType::Complex128) throw nb::type_error("'xp' must be real, not complex128");
    if (xp0.shape[0] == 0) throw nb::value_error("'xp' must not be empty");
    if (fp0.shape[0] != xp0.shape[0]) throw nb::value_error("'fp' must be as long as 'xp'");
    // period: a finite nonzero number (its absolute value), or None
    double period = 0.0;
    if (!period_obj.is_none()) {
        const std::optional<double> p = real_value(period_obj, "period");
        if (!p) throw nb::type_error(("'period' must be an int or a float, not " + py_type_name(period_obj)).c_str());
        if (*p == 0.0 || !std::isfinite(*p)) throw nb::value_error("'period' must be finite and nonzero");
        period = std::fabs(*p);
    }
    const NDArray xp = contiguous_as(xp0, DType::Float64);
    const DType vd = fp0.dtype == DType::Complex128 ? DType::Complex128 : DType::Float64;
    const NDArray fp = contiguous_as(fp0, vd);
    // xp finite, and nondecreasing unless a period sorts it (decision IP5)
    {
        const double* p = reinterpret_cast<const double*>(xp.data);
        const std::int64_t n = xp.shape[0];
        bool finite = true, sorted = true;
        for (std::int64_t i = 0; i < n; ++i) finite &= std::isfinite(p[i]);
        for (std::int64_t i = 1; i < n; ++i) sorted &= p[i - 1] <= p[i];
        if (!finite) throw nb::value_error("'xp' must be finite");
        if (!sorted && period == 0.0) throw nb::value_error("'xp' must be nondecreasing");
    }
    const NDArray* xa = nullptr;
    if (!scalar) {
        xc = contiguous_as(as_ndarray(x_obj), DType::Float64);
        xa = &xc;
    }
    if (vd == DType::Complex128) return interp_typed<complex128>(xa, x_scalar, xp, fp, left_obj, right_obj, period);
    return interp_typed<double>(xa, x_scalar, xp, fp, left_obj, right_obj, period);
}

}  // namespace

void bind_interp(nb::module_& m) {
    using nb::arg;
    m.def(
        "interp", named("interp", &interp), nb::sig("def interp(x, xp, fp, /, *, left=None, right=None, period=None)"),
        arg().none(), arg().none(), arg().none(), nb::kw_only(), arg("left").none() = nb::none(),
        arg("right").none() = nb::none(), arg("period").none() = nb::none()
    );
}

}  // namespace sd
