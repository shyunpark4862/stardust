// QR and least squares: qr, qr_pivoted, lstsq (API.md §3, §4; ROADMAP.md M4).
//
// qr        : geqrf, then orgqr/ungqr builds Q in place in its output slot
// qr_pivoted: geqp3 (column pivoting), then the same
// lstsq     : gelsd, chosen by measurement in M4 (experiments/m4_lstsq_driver,
//             DESIGN.md §2.13): faster than gelss everywhere that matters, and
//             unlike gelsy it gives the singular values the result carries (S6)
#include "ops/qr.hpp"

#include "core/args.hpp"
#include "core/outblock.hpp"
#include "core/pyrefs.hpp"
#include "ops/common.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace sdla {

namespace {

// mode='reduced' | 'complete' (API.md §0.4). true for 'complete'.
bool parse_mode(nb::handle o, const char* func) {
    if (!PyUnicode_Check(o.ptr()))
        throw nb::type_error((std::string(func) + "(): 'mode' must be 'reduced' or 'complete'").c_str());
    const std::string s = nb::cast<std::string>(o);
    if (s == "reduced") return false;
    if (s == "complete") return true;
    throw nb::value_error((std::string(func) + "(): 'mode' must be 'reduced' or 'complete', got '" + s + "'").c_str());
}

nb::object qr_impl(nb::handle xo, nb::handle mode_o, bool pivoted) {
    const char* func = pivoted ? "qr_pivoted" : "qr";
    Input x(xo, func, "x");
    require_matrix(x);
    const bool complete = parse_mode(mode_o, func);
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    // reduced: Q (m, k), R (k, n); complete: Q (m, m), R (m, n)
    const lint qcols = complete ? m : k, rrows = complete ? m : k;
    const lint lda = std::max<lint>(m, 1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, func);
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out qo = alloc_matrices(b.shape(), m, qcols, dtype_of<T>());
        Out ro = alloc_matrices(b.shape(), rrows, n, dtype_of<T>());
        Out po;
        if (pivoted) po = alloc_vectors(b.shape(), n, DType::Int64);
        T* Q = qo.as<T>();
        T* R = ro.as<T>();
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* a = s.get<T>(static_cast<std::size_t>(lda * n));
            T* tau = s.get<T>(static_cast<std::size_t>(std::max<lint>(k, 1)), 1);
            if (!pack(x.dtype(), x.data() + off[0], m, n, x.stride(-2), x.stride(-1), a, lda, true)) return kNonFinite;
            lint* jpvt = nullptr;
            if (pivoted) {
                jpvt = s.get_ints(static_cast<std::size_t>(n));
                std::fill(jpvt, jpvt + n, 0);  // every column free to move
                if (m > 0 && n > 0) lapack::geqp3(m, n, a, lda, jpvt, tau, s);
                else
                    for (lint j = 0; j < n; ++j) jpvt[j] = j + 1;
            } else if (m > 0 && n > 0) {
                lapack::geqrf(m, n, a, lda, tau, s);
            }
            // R: the upper triangle of the factored matrix (rows beyond k are zero).
            T* ri = R + i * rrows * n;
            for (lint j = 0; j < n; ++j)
                for (lint r = 0; r < rrows; ++r) ri[j * rrows + r] = (r <= j && r < k) ? a[j * lda + r] : T(0);
            // Q: the reflectors go to Q's own slot, and orgqr turns them into Q there.
            LapackSlot<T> qs(Q + i * m * qcols, m * qcols, s, Scratch::kAlign0);
            T* qi = qs.get();
            if (m > 0 && qcols > 0) {
                for (lint j = 0; j < k; ++j) std::copy(a + j * lda, a + j * lda + m, qi + j * m);
                lapack::orgqr(m, qcols, k, qi, m, tau, s);
            }
            if (pivoted) {
                // jpvt[j] (1-based) is the column of x that went to position j: x[:, perm] = Q @ R.
                std::int64_t* p = po.as<std::int64_t>() + i * n;
                for (lint j = 0; j < n; ++j) p[j] = jpvt[j] - 1;
            }
            return 0;
        });
        if (f) raise(b, f, func, "x");
        if (pivoted) return make_result(Result::QRPivoted, {to_sdnp(qo), to_sdnp(ro), to_sdnp(po)});
        return make_result(Result::QR, {to_sdnp(qo), to_sdnp(ro)});
    });
}

nb::object qr(nb::handle xo, nb::handle mode_o) { return qr_impl(xo, mode_o, false); }
nb::object qr_pivoted(nb::handle xo, nb::handle mode_o) { return qr_impl(xo, mode_o, true); }

nb::object lstsq(nb::handle ao, nb::handle bo, nb::handle rtol_o) {
    Input a(ao, "lstsq", "a");
    Input y(bo, "lstsq", "b");
    require_matrix(a);
    const lint m = a.shape(-2), n = a.shape(-1), mn = std::min(m, n);
    const Rhs r = rhs_of(y, m, "lstsq", "a", false);
    const double rcond = parse_rtol(rtol_o, m, n, "lstsq");
    const lint lda = std::max<lint>(m, 1), ldb = std::max<lint>({m, n, 1});
    Batch b({{a.batch_shape(2), a.batch_strides(2)}, {y.batch_shape(r.core), y.batch_strides(r.core)}}, "lstsq");
    const bool complex = computes_complex(a.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out xo = r.vec ? alloc_vectors(b.shape(), n, dtype_of<T>()) : alloc_matrices(b.shape(), n, r.k, dtype_of<T>());
        Out so = alloc_vectors(b.shape(), mn, DType::Float64);
        Out rk = alloc_c(b.shape(), DType::Int64);
        T* X = xo.as<T>();
        double* S = so.as<double>();
        std::int64_t* RK = rk.as<std::int64_t>();
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* A = s.get<T>(static_cast<std::size_t>(lda * n));
            if (!pack(a.dtype(), a.data() + off[0], m, n, a.stride(-2), a.stride(-1), A, lda, true)) return kNonFinite;
            // b needs max(m, n) rows: gelsd leaves the n-row solution where b was.
            // With no right-hand side (k = 0) the rank and singular values are still
            // wanted, but gelsd's helper xLALSD rejects nrhs = 0: give it one zero column.
            const lint nrhs = std::max<lint>(r.k, 1);
            T* B = s.get<T>(static_cast<std::size_t>(ldb * nrhs), 1);
            if (r.k == 0) std::fill(B, B + ldb, T(0));
            if (!pack(y.dtype(), y.data() + off[1], m, r.k, r.rs, r.cs, B, ldb, true)) return kNonFiniteB;
            T* xi = X + i * n * r.k;
            lint rank = 0;
            if (m > 0 && n > 0) {
                // xLALSD silently replaces rcond >= 1 by machine epsilon. By our
                // definition (API.md §0.4: singular values <= rtol * max are zero)
                // rtol >= 1 zeroes all of them: rank 0 and solution 0 (DESIGN.md §2.13).
                const bool all_zero = rcond >= 1.0;
                LapackSlot<double> sv(S + i * mn, mn, s, Scratch::kAlign0);
                if (lapack::gelsd(m, n, nrhs, A, lda, B, ldb, sv.get(), all_zero ? 0.5 : rcond, rank, s) > 0)
                    return kNoConvergence;
                if (all_zero) {
                    rank = 0;
                    std::fill(xi, xi + n * r.k, T(0));
                } else {
                    for (lint j = 0; j < r.k; ++j) std::copy(B + j * ldb, B + j * ldb + n, xi + j * n);
                }
            } else {
                std::fill(xi, xi + n * r.k, T(0));  // the minimum-norm solution of an empty system
            }
            RK[i] = rank;
            return 0;
        });
        if (f) raise(b, f, "lstsq", "a");
        nb::object rank_obj = b.ndim() == 0 ? py_scalar(RK[0]) : to_sdnp(rk);
        return make_result(Result::Lstsq, {to_sdnp(xo), rank_obj, to_sdnp(so)});
    });
}

}  // namespace

void bind_qr(nb::module_& m) {
    m.def("qr", &qr, nb::arg().none(), nb::kw_only(), nb::arg("mode") = "reduced");
    m.def("qr_pivoted", &qr_pivoted, nb::arg().none(), nb::kw_only(), nb::arg("mode") = "reduced");
    m.def("lstsq", &lstsq, nb::arg().none(), nb::arg().none(), nb::kw_only(), nb::arg("rtol").none() = nb::none());
}

}  // namespace sdla
