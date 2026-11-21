// QR and least squares: qr, qr_pivoted, lstsq (docs/api/linalg.md §4, §5).
//
// qr        : geqrf, then orgqr/ungqr builds Q in place in its output slot
// qr_pivoted: geqp3 (column pivoting), then the same
// lstsq     : gelsd, chosen by measurement (docs/design/linalg.md §10.4):
//             faster than gelss everywhere that matters, and unlike gelsy it
//             gives the singular values the result carries (docs/api/linalg.md §4)
#include "linalg/ops/qr.hpp"

#include <limits>
#include "bind/errors.hpp"

#include "linalg/args.hpp"
#include "linalg/cost.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pyrefs.hpp"
#include "linalg/ops/common.hpp"
#include "linalg/ops/svals.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <limits>
#include <cmath>
#include <string>

namespace sd::linalg {

namespace {


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
        const Failure f =
            run_batch(b, cost(kQR, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                T* a = s.get<T>(static_cast<std::size_t>(lda * n));
                T* tau = s.get<T>(static_cast<std::size_t>(std::max<lint>(k, 1)), 1);
                if (!pack(x.dtype(), x.data() + off[0], m, n, x.stride(-2), x.stride(-1), a, lda, true))
                    return kNonFinite;
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
        if (pivoted) return make_result(Result::QRPivoted, {to_array(qo), to_array(ro), to_array(po)});
        return make_result(Result::QR, {to_array(qo), to_array(ro)});
    });
}

nb::object qr(nb::handle xo, nb::handle mode_o) { return qr_impl(xo, mode_o, false); }
nb::object qr_pivoted(nb::handle xo, nb::handle mode_o) { return qr_impl(xo, mode_o, true); }

// rtol below this goes through the SVD (see the loop below).
constexpr double kEps = 0x1p-52;

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
        const Failure f = run_batch(
            b, cost(kSVD, double(m) * n) + rhs_cost(m, r.k), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                T* A = s.get<T>(static_cast<std::size_t>(lda * n));
                if (!pack(a.dtype(), a.data() + off[0], m, n, a.stride(-2), a.stride(-1), A, lda, true))
                    return kNonFinite;
                // b needs max(m, n) rows: gelsd leaves the n-row solution where b was.
                // With no right-hand side (k = 0) the rank and singular values are still
                // wanted, but gelsd's helper xLALSD rejects nrhs = 0: give it one zero column.
                const lint nrhs = std::max<lint>(r.k, 1);
                // B where numpy has it (numpy and scipy's buffer: A, then B, then S in one malloc):
                // 8 m n bytes past a 16-byte boundary. Accelerate's dgelsd gives other last bits
                // for a B off 16 bytes on its underdetermined path (the same routine called with
                // only B moved by 8 bytes: 35 of 40 random 9 x 11 systems and 34 of 40 5 x 7 ones
                // changed bits, 3 x 8 and 11 x 9 none), so B keeps numpy's offset (2026-11-19; the
                // array here not 16-byte aligned on purpose; docs/design/linalg.md §6.4 lists the
                // scratch sub-arrays elsewhere that may be unaligned as well).
                T* B = s.get<T>(static_cast<std::size_t>(ldb * nrhs) + 1, 1);
                if constexpr (std::is_same_v<T, double>) B += (m * n) % 2;
                if (r.k == 0) std::fill(B, B + ldb, T(0));
                if (!pack(y.dtype(), y.data() + off[1], m, r.k, r.rs, r.cs, B, ldb, true)) return kNonFiniteB;
                T* xi = X + i * n * r.k;
                lint rank = 0;
                // x = V S_r^-1 U^H b from gesdd's U, Vh with the `rank` largest values inverted
                // (A and B packed); the values below 2^-1022 apart (svals.hpp: inf times an
                // exact zero of a singular vector was NaN, lstsq(diag(1, 5e-324), 1) [nan, inf]).
                auto svd_solve = [&](lint keep) -> int {
                    // U (m x mn), Vh (mn x n) and gesdd's own singular values in slot 2; U^H b in slot 3
                    T* u = s.get<T>(static_cast<std::size_t>(m * mn + mn * n + mn), 2);
                    T* vh = u + m * mn;
                    double* sd = reinterpret_cast<double*>(vh + mn * n);
                    if (lapack::gesdd('S', m, n, A, lda, sd, u, m, vh, mn, s) > 0) return kNoConvergence;
                    if (r.k == 0) return 0;
                    T* w = s.get<T>(static_cast<std::size_t>(mn * r.k), 3);
                    constexpr char h = std::is_same_v<T, cplx> ? 'C' : 'T';
                    lapack::gemm<T>(h, 'N', mn, r.k, m, u, m, B, ldb, w, mn);
                    const lint rn = normal_rank(sd, keep), nh = keep - rn;
                    std::vector<T> wh(static_cast<std::size_t>(nh * r.k));
                    for (lint j = 0; j < r.k; ++j)
                        for (lint q = rn; q < keep; ++q) wh[j * nh + q - rn] = w[j * mn + q] / (sd[q] * kSingularShift);
                    for (lint j = 0; j < r.k; ++j)
                        for (lint q = 0; q < mn; ++q) w[j * mn + q] = q < rn ? w[j * mn + q] / sd[q] : T(0);
                    // the solution's slot in the output, 16-byte aligned for gemm (kAlign0 holds sv)
                    LapackSlot<T> xs(xi, n * r.k, s, Scratch::kAlign1);
                    lapack::gemm<T>(h, 'N', n, r.k, mn, vh, mn, w, mn, xs.get(), n);
                    if (nh > 0) {  // rare: a kept singular value in the subnormals
                        std::vector<T> xh(static_cast<std::size_t>(n * r.k));
                        lapack::gemm<T>(h, 'N', n, r.k, nh, vh + rn, mn, wh.data(), nh, xh.data(), n);
                        add_shifted(xs.get(), xh.data(), n * r.k);
                    }
                    return 0;
                };
                if (m > 0 && n > 0 && rcond < kEps) {
                    // rtol below eps: gelsd's xLALSD moves the bidiagonal's entries below eps (of the
                    // largest) to +-eps before its divide and conquer (min(m, n) > 25), so a zero
                    // singular value counted as one (rtol = 0: a 30 x 30 with a zero column had rank
                    // 30, matrix_rank 29). Here the singular values are matrix_rank's own (gesvd,
                    // values only: the same rank to the bit), and the solution is
                    // V S_r^-1 U^H b from gesdd, inverting the `rank` largest (user decision
                    // 2026-11-19; docs/design/linalg.md §10.4).
                    LapackSlot<double> sv(S + i * mn, mn, s, Scratch::kAlign0);
                    double* svp = sv.get();
                    if (const int code = singular_values<T>(a, off[0], m, n, s, svp)) return code;
                    const double tol = rcond * svp[0];
                    for (lint j = 0; j < mn; ++j) rank += svp[j] > tol;
                    if (const int code = svd_solve(rank)) return code;
                } else if (m > 0 && n > 0) {
                    // xLALSD silently replaces rcond >= 1 by machine epsilon. By our
                    // definition (docs/api/linalg.md §4: singular values <= rtol * max are zero)
                    // rtol >= 1 zeroes all of them: rank 0 and solution 0 (docs/design/linalg.md §10.4).
                    // (rcond >= eps here: the values it moves to eps, below eps before, stay at or
                    // below rcond, zero as they were.)
                    const bool all_zero = rcond >= 1.0;
                    const double rc = all_zero ? 0.5 : rcond;
                    LapackSlot<double> sv(S + i * mn, mn, s, Scratch::kAlign0);
                    if (lapack::gelsd(m, n, nrhs, A, lda, B, ldb, sv.get(), rc, rank, s) > 0) return kNoConvergence;
                    if (all_zero) {
                        rank = 0;
                        std::fill(xi, xi + n * r.k, T(0));
                    } else {
                        // (A kept value in the subnormals needs nothing here: with rcond >= eps it
                        // means sigma_max < 2^-970, so the largest entry is below gelsd's smlnum
                        // (sfmin / eps = 2^-970; dgelsd.f): it scales the matrix into range and
                        // the solution back, so exact zeros
                        // stay zero and overflow is inf: 9000 random tiny systems gave no NaN,
                        // the same as numpy's; 2026-11-19.)
                        for (lint j = 0; j < r.k; ++j) std::copy(B + j * ldb, B + j * ldb + n, xi + j * n);
                    }
                } else {
                    std::fill(xi, xi + n * r.k, T(0));  // the minimum-norm solution of an empty system
                }
                RK[i] = rank;
                return 0;
            }
        );
        if (f) raise(b, f, "lstsq", "a");
        nb::object rank_obj = b.ndim() == 0 ? py_scalar(RK[0]) : to_array(rk);
        return make_result(Result::Lstsq, {to_array(xo), rank_obj, to_array(so)});
    });
}

}  // namespace

void bind_qr(nb::module_& m) {
    m.def(
        "qr", named("qr", &qr), nb::sig("def qr(x, /, *, mode='reduced')"), nb::arg().none(), nb::kw_only(),
        nb::arg("mode").none() = "reduced"
    );
    m.def(
        "qr_pivoted", named("qr_pivoted", &qr_pivoted), nb::sig("def qr_pivoted(x, /, *, mode='reduced')"),
        nb::arg().none(), nb::kw_only(), nb::arg("mode").none() = "reduced"
    );
    m.def(
        "lstsq", named("lstsq", &lstsq), nb::sig("def lstsq(a, b, /, *, rtol=None)"), nb::arg().none(),
        nb::arg().none(), nb::kw_only(), nb::arg("rtol").none() = nb::none()
    );
}

}  // namespace sd::linalg
