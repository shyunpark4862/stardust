// cov and corrcoef (docs/api/array.md §7.11).
//
// numpy's steps in numpy's order: each variable's mean, the centred matrix
// X, X X^H, times 1/(N - correction). The mean and the BLAS call are ours,
// so the bits are not numpy's (docs/design/array.md §6.12).

#include "array/ops/covariance.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "accelerate/blas.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/ufunc.hpp"
#include "array/ops/reduction.hpp"
#include "array/ops/pieces.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sd {

namespace {

// The covariance matrix (p, p): float64, or complex128 for complex x.
//   1. each variable's mean: sd.mean of x itself along the observations;
//   2. the centred matrix X (p, n), C-ordered in the result dtype, written
//      by one subtract (x - mean, out=X) -- the copy and the centring in one pass;
//   3. X X^H by BLAS: syrk / herk (the upper triangle, half of gemm's work;
//      numpy's dot(X, X.T) takes syrk too) or gemm, by the shape;
//   4. the lower triangle copied from the upper one (so the result is exactly
//      symmetric / Hermitian, with a real diagonal) and everything times
//      1/fact, in one pass.
NDArray cov_matrix(nb::handle x_obj, bool rowvar, double correction) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.ndim != 1 && x.ndim != 2) throw nb::value_error("'x' must be 1-D or 2-D");
    const DType d = x.dtype == DType::Complex128 ? DType::Complex128 : DType::Float64;

    // Variables in rows: a 1-D x is one variable; rowvar=False transposes a 2-D x.
    NDArray view = view_of(x_obj);
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
    const std::int64_t cshape[] = {p, p};
    NDArray c = new_array(d, cshape);
    if (p == 0) return c;

    nb::object v = wrap(NDArray(view));
    const nb::object mu = mean(v, nb::int_(1), nb::bool_(true));  // (p, 1)
    const std::int64_t xshape[] = {p, n};
    NDArray xc = new_array(d, xshape);
    nb::object xc_obj = wrap(NDArray(xc));
    binary<ops::Subtract>(v, mu, xc_obj);

    visit_dtype(d, [&]<typename T>(std::type_identity<T>) {
        if constexpr (ops::is_float<T> || ops::is_cplx<T>) {
            T* cp = reinterpret_cast<T*>(c.data);
            const T* xp = reinterpret_cast<const T*>(xc.data);
            if (n > 0) {
                const parallel::Unlocked unlocked(true);  // BLAS may take a while; it touches no Python objects
                // syrk does half of gemm's multiply-adds, but Accelerate's syrk
                // is slower when there are many more observations than
                // variables (p = 20, n = 1e5: 3.5 ms against 2.0 ms). Measured:
                // syrk wins or ties while n <= 4p (p = 2000, n = 200: 2.4x),
                // gemm from there on.
                if (n <= 4 * p) blas::syrk_upper<T>(p, n, xp, n, cp, p);
                else
                    blas::gemm<T>(
                        CblasNoTrans, ops::is_cplx<T> ? CblasConjTrans : CblasTrans, p, p, n, xp, n, xp, n, cp, p
                    );
            } else {
                std::fill(cp, cp + p * p, T(0));
            }
            // numpy multiplies by 1/fact; fact <= 0 gives NaN, as var does
            // (docs/design/array.md §6.8). The lower triangle is the upper one
            // mirrored (conjugated for complex).
            const double fact = static_cast<double>(n) - correction;
            const double scale = fact > 0 ? 1.0 / fact : std::numeric_limits<double>::quiet_NaN();
            // In 64 x 64 tiles: writing the lower triangle column by column
            // (stride p) missed the cache on every element (p = 2000: 1.7 ms
            // of a 2.7 ms cov). Each task owns the tiles of one block row of
            // the upper triangle and their mirrors, which no other task touches.
            constexpr std::int64_t kTile = 64;
            const std::int64_t blocks = (p + kTile - 1) / kTile;
            auto block_row = [&](std::int64_t bi) {
                const std::int64_t i0 = bi * kTile, i1 = std::min(p, i0 + kTile);
                for (std::int64_t j0 = i0; j0 < p; j0 += kTile) {
                    const std::int64_t j1 = std::min(p, j0 + kTile);
                    for (std::int64_t i = i0; i < i1; ++i) {
                        if constexpr (ops::is_cplx<T>) {
                            if (j0 == i0) cp[i * p + i] = T(cp[i * p + i].real(), 0.0);  // |x|^2 sums: real
                        }
                        for (std::int64_t j = std::max(j0, i); j < j1; ++j) {
                            const T val = cp[i * p + j] * scale;
                            cp[i * p + j] = val;
                            if constexpr (ops::is_cplx<T>) cp[j * p + i] = T(val.real(), -val.imag());
                            else cp[j * p + i] = val;
                        }
                    }
                }
            };
            for_pieces(blocks, 1, p * p, [&](std::int64_t b, std::int64_t, std::int64_t) { block_row(b); });
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
            for_pieces(p, kParallelTask / p, p * p, [&](std::int64_t, std::int64_t r0, std::int64_t r1) {
                for (std::int64_t i = r0; i < r1; ++i)
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
            });
        }
    });
    return wrap(std::move(c));
}

}  // namespace

void bind_covariance(nb::module_& m) {
    using nb::arg;
    m.def(
        "cov", named("cov", &cov), arg().none(), nb::kw_only(), arg("rowvar").none() = true,
        arg("correction").none() = 0
    );
    m.def("corrcoef", named("corrcoef", &corrcoef), arg().none(), nb::kw_only(), arg("rowvar").none() = true);
}

}  // namespace sd
