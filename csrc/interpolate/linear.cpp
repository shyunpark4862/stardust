// The linear path of sd.interpolate (docs/design/interpolate.md §3.2, §6): the piecewise linear function
// through (x[i], y[i]) with a batch of columns, under the three outside rules. The k = 1 B-spline
// (milestone IP3) and the 1-D linear grid (milestone IP5) evaluate through it, so that they, sd.interp and
// each other give the same bits on [x[0], x[n - 1]] (§2.2): one interval count (base/interval.hpp), one
// t per point for all columns, one kernel (base/lerp.hpp).
//
// _core.interpolate.linear(x, xp, fp, extrapolate) -> (m, k): x (m,) float64, xp (n,) float64 strictly
// increasing (n >= 2; the Python layer checks the order), fp (n, k) float64 or complex128, all
// C-contiguous; extrapolate 0 (NaN outside), 1 (extend the end pieces) or 2 (periodic). A NaN point gives
// NaN in every column (NaN in both parts of a complex value). Nothing public calls it yet.

#include "interpolate/interpolate.hpp"

#include "array/convert.hpp"
#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "base/interval.hpp"
#include "base/lerp.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "interpolate/search.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

template <typename T>
struct Piecewise {
    const double* x;
    const T* y;  // (n, k) row-major
    std::int64_t n, k;
    Extrapolate rule;
};

template <typename T>
constexpr T nan_value() {
    constexpr double q = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) return q;
    else return T(q, q);
}

template <typename T>
void linear_range(const Piecewise<T>& P, const double* x, T* out, std::int64_t begin, std::int64_t end) {
    constexpr std::int64_t kBlock = 256;
    double v[kBlock];
    std::int64_t c[kBlock];
    GallopState state;
    const double lo = P.x[0], hi = P.x[P.n - 1];
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t m = std::min(kBlock, end - b0);
        if (P.rule == Extrapolate::Periodic)
            for (std::int64_t i = 0; i < m; ++i) v[i] = wrap_periodic(x[b0 + i], lo, hi);
        else std::copy(x + b0, x + b0 + m, v);
        count_le_block(P.x, P.n, v, m, c, state);
        if constexpr (std::is_same_v<T, double>) {
            if (P.k == 1) {
                // One column: the interval's nodes and values side by side, then base/lerp.hpp's block kernel (the
                // same bits as the loop below); NaN points, points outside under the NaN rule and those the block
                // kernel leaves out take that loop's way.
                double x0[kBlock], x1[kBlock], a[kBlock], bb[kBlock];
                bool inside = true;
                for (std::int64_t i = 0; i < m; ++i) {
                    inside &= (v[i] >= lo) & (v[i] <= hi);
                    const std::int64_t j = interval_of(c[i], P.n);
                    x0[i] = P.x[j], x1[i] = P.x[j + 1], a[i] = P.y[j], bb[i] = P.y[j + 1];
                }
                T* o = out + b0;
                const bool all = lerp_block(v, x0, x1, a, bb, o, m);
                if (!(all & (inside | (P.rule != Extrapolate::Nan)))) {
                    for (std::int64_t i = 0; i < m; ++i) {
                        const double p = v[i];
                        if (p != p || (P.rule == Extrapolate::Nan && (p < lo || p > hi))) o[i] = nan_value<T>();
                        else if (!lerp_block_covers(p, x0[i], x1[i], a[i], bb[i]))
                            o[i] = lerp(a[i], bb[i], lerp_t(p, x0[i], x1[i]));
                    }
                }
                continue;
            }
        }
        for (std::int64_t i = 0; i < m; ++i) {
            T* row = out + (b0 + i) * P.k;
            const double p = v[i];
            if (p != p || (P.rule == Extrapolate::Nan && (p < lo || p > hi))) {
                std::fill(row, row + P.k, nan_value<T>());
                continue;
            }
            const std::int64_t j = interval_of(c[i], P.n);
            const double t = lerp_t(p, P.x[j], P.x[j + 1]);
            const T* a = P.y + j * P.k;
            const T* b = a + P.k;
            for (std::int64_t col = 0; col < P.k; ++col) row[col] = lerp(a[col], b[col], t);
        }
    }
}

template <typename T>
void linear_all(const Piecewise<T>& P, const double* x, T* out, std::int64_t m) {
    const std::int64_t log_n = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::log2(P.n + 1)));
    const std::int64_t work = m * (2 * log_n + 8 * P.k);  // sd.interp's estimate and task size (decision IP102)
    if (work >= kParallelThreshold && parallel::available()) {
        const std::int64_t per = std::max<std::int64_t>(1, 1024 / std::max<std::int64_t>(1, P.k));
        const std::int64_t tasks = (m + per - 1) / per;
        parallel::parallel_for(
            tasks, [&](std::int64_t t) { linear_range(P, x, out, t * per, std::min(m, (t + 1) * per)); },
            work_ns(work / tasks)
        );
    } else {
        const parallel::Unlocked unlocked(work >= kParallelThreshold);
        linear_range(P, x, out, 0, m);
    }
}

const NDArray& contiguous_arg(nb::handle o, const char* what, int ndim) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != ndim || !a.c_contiguous)
        throw nb::value_error(
            ("'" + std::string(what) + "' must be a C-contiguous " + std::to_string(ndim) + "-D array").c_str()
        );
    return a;
}

nb::object linear(nb::handle x_obj, nb::handle xp_obj, nb::handle fp_obj, int extrapolate) {
    const NDArray& x = contiguous_arg(x_obj, "x", 1);
    const NDArray& xp = contiguous_arg(xp_obj, "xp", 1);
    const NDArray& fp = contiguous_arg(fp_obj, "fp", 2);
    if (x.dtype != DType::Float64 || xp.dtype != DType::Float64) throw nb::type_error("'x' and 'xp' must be float64");
    if (fp.dtype != DType::Float64 && fp.dtype != DType::Complex128)
        throw nb::type_error("'fp' must be float64 or complex128");
    const std::int64_t n = xp.shape[0];
    if (n < 2 || fp.shape[0] != n) throw nb::value_error("'xp' must have 2 points or more, 'fp' one row per point");
    if (extrapolate < 0 || extrapolate > 2) throw nb::value_error("'extrapolate' must be 0, 1 or 2");
    const std::int64_t m = x.shape[0], k = fp.shape[1];
    const std::int64_t shape[2] = {m, k};
    NDArray out = new_array(fp.dtype, shape);
    if (m == 0 || k == 0) return wrap(std::move(out));
    const auto* xd = reinterpret_cast<const double*>(x.data);
    const auto* xpd = reinterpret_cast<const double*>(xp.data);
    const auto rule = static_cast<Extrapolate>(extrapolate);
    if (fp.dtype == DType::Complex128) {
        const Piecewise<complex128> P{xpd, reinterpret_cast<const complex128*>(fp.data), n, k, rule};
        linear_all(P, xd, reinterpret_cast<complex128*>(out.data), m);
    } else {
        const Piecewise<double> P{xpd, reinterpret_cast<const double*>(fp.data), n, k, rule};
        linear_all(P, xd, reinterpret_cast<double*>(out.data), m);
    }
    return wrap(std::move(out));
}

}  // namespace

void bind_linear(nb::module_& m) {
    m.def(
        "linear", named("linear", &linear), nb::arg("x").none(), nb::arg("xp").none(), nb::arg("fp").none(),
        nb::arg("extrapolate")
    );  // fmt: skip
}

}  // namespace sd::interpolate
