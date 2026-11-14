// cov and corrcoef (docs/api/array.md §7.11).
//
// numpy's steps in numpy's order: each variable's mean, the centred matrix
// X, X @ X^H (our matmul), times 1/(N - correction). The mean and the matmul
// are ours, so the bits are not numpy's (docs/design/array.md §6.12).

#include "array/ops/covariance.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/matmul.hpp"
#include "array/ops/reduction.hpp"
#include "array/parallel_limits.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sd {

namespace {

double parse_correction(nb::handle o) {
    if (const std::optional<double> c = real_value(o)) return *c;
    throw nb::type_error(("'correction' must be an int or a float, not " + py_type_name(o)).c_str());
}

// The covariance matrix (p, p): float64, or complex128 for complex x.
NDArray cov_matrix(nb::handle x_obj, bool rowvar, double correction) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.ndim != 1 && x.ndim != 2) throw nb::value_error("'x' must be 1-D or 2-D");
    const DType d = x.dtype == DType::Complex128 ? DType::Complex128 : DType::Float64;

    // Variables in rows: a 1-D x is one variable; rowvar=False transposes a 2-D x.
    NDArray view = x;
    if (x.ndim == 1) {
        view.ndim = 2;
        view.shape[0] = 1;
        view.strides[0] = 0;
        view.shape[1] = x.shape[0];
        view.strides[1] = x.strides[0];
    } else if (!rowvar) {
        std::swap(view.shape[0], view.shape[1]);
        std::swap(view.strides[0], view.strides[1]);
    }
    view.update_flags();
    const std::int64_t p = view.shape[0], n = view.shape[1];
    const std::int64_t shape[] = {p, n};
    NDArray xc = new_array(d, shape);
    assign(xc, view);

    // Centre each row on its mean (sd.mean, so the same bits as mean(x, axis=1)).
    const NDArray mu = as_ndarray(mean(wrap(NDArray(xc)), nb::int_(1), nb::bool_(true)));
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            T* a = reinterpret_cast<T*>(xc.data);
            const T* m = reinterpret_cast<const T*>(mu.data);
            auto centre = [&](std::int64_t first, std::int64_t last) {
                for (std::int64_t i = first; i < last; ++i)
                    for (std::int64_t k = 0; k < n; ++k) a[i * n + k] = a[i * n + k] - m[i];
            };
            // Rows in tasks of at least kParallelThreshold elements: smaller ones cost more to hand out than they save.
            const std::int64_t rows = std::max<std::int64_t>(1, kParallelThreshold / std::max<std::int64_t>(n, 1));
            const std::int64_t tasks = (p + rows - 1) / rows;
            if (tasks > 1 && parallel::available()) {
                parallel::parallel_for(tasks, [&](std::int64_t t) { centre(t * rows, std::min(p, (t + 1) * rows)); });
            } else {
                centre(0, p);
            }
        }
    });

    // X^H as a transposed view (of a conjugated copy for complex).
    NDArray xh = xc;
    if (d == DType::Complex128) {
        xh = new_array(d, shape);
        const complex128* a = reinterpret_cast<const complex128*>(xc.data);
        complex128* h = reinterpret_cast<complex128*>(xh.data);
        for (std::int64_t i = 0; i < p * n; ++i) h[i] = {a[i].real(), -a[i].imag()};
    }
    std::swap(xh.shape[0], xh.shape[1]);
    std::swap(xh.strides[0], xh.strides[1]);
    xh.update_flags();
    NDArray c = as_ndarray(matmul(wrap(std::move(xc)), wrap(std::move(xh))));

    // numpy multiplies by 1/fact; fact <= 0 gives NaN, as var does
    // (docs/design/array.md §6.8).
    const double fact = static_cast<double>(n) - correction;
    const double scale = fact > 0 ? 1.0 / fact : std::numeric_limits<double>::quiet_NaN();
    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            T* a = reinterpret_cast<T*>(c.data);
            for (std::int64_t i = 0; i < c.size(); ++i) a[i] = a[i] * scale;
        }
    });
    return c;
}

nb::object cov(nb::handle x, nb::handle rowvar_obj, nb::handle correction_obj) {
    require_ndarray(x);
    const bool rowvar = parse_bool(rowvar_obj, "rowvar");
    const double correction = parse_correction(correction_obj);
    NDArray c = cov_matrix(x, rowvar, correction);
    if (c.shape[0] == 1) return load_python(c.dtype, c.data);  // numpy squeezes one variable to a scalar
    return wrap(std::move(c));
}

nb::object corrcoef(nb::handle x, nb::handle rowvar_obj) {
    require_ndarray(x);
    const bool rowvar = parse_bool(rowvar_obj, "rowvar");
    NDArray c = cov_matrix(x, rowvar, 0.0);  // the factor cancels
    const std::int64_t p = c.shape[0];
    if (p == 1) {  // numpy: c / c
        if (c.dtype == DType::Complex128) {
            const complex128 v = *reinterpret_cast<const complex128*>(c.data);
            return to_python(complex128(v / v));
        }
        const double v = *reinterpret_cast<const double*>(c.data);
        return to_python(v / v);
    }
    // c / sd[:, None] / sd[None, :], then the real (and imaginary) parts clipped to [-1, 1].
    auto clip = [](double v) { return std::isnan(v) ? v : std::min(1.0, std::max(-1.0, v)); };
    visit_dtype(c.dtype, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            T* a = reinterpret_cast<T*>(c.data);
            std::vector<double> sd(p);
            for (std::int64_t i = 0; i < p; ++i) {
                if constexpr (ops::is_cplx<T>) sd[i] = std::sqrt(a[i * p + i].real());
                else sd[i] = std::sqrt(a[i * p + i]);
            }
            for (std::int64_t i = 0; i < p; ++i)
                for (std::int64_t j = 0; j < p; ++j) {
                    T v = a[i * p + j];
                    if constexpr (ops::is_cplx<T>) {
                        const double re = v.real() / sd[i] / sd[j], im = v.imag() / sd[i] / sd[j];
                        v = {clip(re), clip(im)};
                    } else {
                        v = clip(v / sd[i] / sd[j]);
                    }
                    a[i * p + j] = v;
                }
        }
    });
    return wrap(std::move(c));
}

}  // namespace

void bind_covariance(nb::module_& m) {
    using nb::arg;
    m.def("cov", named("cov", &cov), arg().none(), nb::kw_only(), arg("rowvar").none() = true,
          arg("correction").none() = 0);
    m.def("corrcoef", named("corrcoef", &corrcoef), arg().none(), nb::kw_only(), arg("rowvar").none() = true);
}

}  // namespace sd
