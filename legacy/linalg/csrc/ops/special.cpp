// Special structure (API.md §5; ROADMAP.md M7):
// solve_tridiagonal, solve_banded, solve_sylvester.
//
// Drivers chosen by measurement (experiments/m7_drivers, DESIGN.md §2.16):
//   solve_tridiagonal : gtsv  (1.2-3x faster than gttrf + gttrs, same bits)
//   solve_banded      : gbsv  (as fast as gbtrf + gbtrs, same bits); gtsv when
//                       both bandwidths are 1 (user decision), so that
//                       solve_banded(.., 1, 1) == solve_tridiagonal bit for bit
//   solve_sylvester   : Bartels-Stewart: gees of a and b, trsyl3 (up to 4.8x
//                       faster than trsyl), and back with four gemm
#include "ops/special.hpp"

#include "core/args.hpp"
#include "core/cost.hpp"
#include "core/outblock.hpp"
#include "core/pyrefs.hpp"
#include "ops/common.hpp"

#include <algorithm>
#include <string>

namespace sdla {

namespace {

// One strided vector of an input: dtype, first element, byte stride.
struct Vec {
    DType dt;
    const std::byte* p;
    std::int64_t stride;
};

// b must have the matrix's n rows; the message names the matrix as n x n,
// since the structured arguments do not have that shape themselves.
void check_rhs_rows(const Input& y, lint n, const char* func) {
    const std::int64_t rows = y.ndim() == 1 ? y.shape(0) : y.shape(-2);
    if (rows != n)
        throw nb::value_error((std::string(func) + "(): the matrix is " + std::to_string(n) + "x" + std::to_string(n) +
                               " but 'b' has " + std::to_string(rows) + " rows")
                                  .c_str());
}

// ------------------------------------------------------- tridiagonal (gtsv) --

// One tridiagonal solve: packs dl, d, du into scratch and b into the output
// slot `x`, then gtsv. The checks run in argument order with the matrix's
// singularity right after the matrix (common.hpp): gtsv's pivots do not depend
// on b, so a NaN in b is reported after a singular matrix, as in solve.
template <class T>
int tri_body(Vec dl, Vec d, Vec du, lint n, const Input& y, std::int64_t yoff, const Rhs& r, T* x, Scratch& s,
             int c_dl, int c_d, int c_du) {
    const lint nm1 = std::max<lint>(n - 1, 0);
    // three sub-arrays, each on a 16-byte boundary (common.hpp: aligned16)
    const std::size_t pn = pad2(n), pm = pad2(nm1);
    T* w = s.get<T>(pn + 2 * pm);
    T *wd = w, *wdl = w + pn, *wdu = w + pn + pm;
    if (!pack(dl.dt, dl.p, nm1, 1, dl.stride, 0, wdl, nm1, true)) return c_dl;
    if (!pack(d.dt, d.p, n, 1, d.stride, 0, wd, n, true)) return c_d;
    if (!pack(du.dt, du.p, nm1, 1, du.stride, 0, wdu, nm1, true)) return c_du;
    const bool b_ok = pack(y.dtype(), y.data() + yoff, n, r.k, r.rs, r.cs, x, n, true);
    if (n > 0 && lapack::gtsv(n, r.k, wdl, wd, wdu, x, n) > 0) return kSingular;
    return b_ok ? 0 : kNonFiniteB;
}

void require_vector(const Input& v, const char* func, const char* name) {
    if (v.ndim() < 1)
        throw nb::value_error((std::string(func) + "(): '" + name + "' must have at least one dimension").c_str());
}

nb::object solve_tridiagonal(nb::handle dlo, nb::handle do_, nb::handle duo, nb::handle bo) {
    const char* func = "solve_tridiagonal";
    Input dl(dlo, func, "dl"), d(do_, func, "d"), du(duo, func, "du"), y(bo, func, "b");
    require_vector(dl, func, "dl");
    require_vector(d, func, "d");
    require_vector(du, func, "du");
    const lint n = d.shape(-1);
    const lint nm1 = std::max<lint>(n - 1, 0);
    for (const Input* v : {&dl, &du})
        if (v->shape(-1) != nm1)
            throw nb::value_error((std::string(func) + "(): '" + v->arg() + "' must have length " +
                                   std::to_string(nm1) + " (one less than 'd'), got " + std::to_string(v->shape(-1)))
                                      .c_str());
    check_rhs_rows(y, n, func);
    const Rhs r = rhs_of(y, n, func, "d");
    Batch b({{dl.batch_shape(1), dl.batch_strides(1)},
             {d.batch_shape(1), d.batch_strides(1)},
             {du.batch_shape(1), du.batch_strides(1)},
             {y.batch_shape(r.core), y.batch_strides(r.core)}},
            func);
    const bool complex = computes_complex(dl.dtype()) || computes_complex(d.dtype()) ||
                         computes_complex(du.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_rhs(b, r, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, cost(kBanded, 3.0 * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            return tri_body<T>({dl.dtype(), dl.data() + off[0], dl.stride(-1)}, {d.dtype(), d.data() + off[1], d.stride(-1)},
                               {du.dtype(), du.data() + off[2], du.stride(-1)}, n, y, off[3], r,
                               LapackSlot<T>(res + i * n * r.k, n * r.k, s, Scratch::kAlign0).get(), s, kNonFinite,
                               kNonFinite2, kNonFinite3);
        });
        if (f) raise(b, f, func, "dl", "d", "du");
        return to_sdnp(out);
    });
}

// -------------------------------------------------------------- banded (gbsv) --

nb::object solve_banded(nb::handle abo, nb::handle bo, nb::handle klo, nb::handle kuo) {
    const char* func = "solve_banded";
    Input a(abo, func, "ab"), y(bo, func, "b");
    require_matrix(a);
    const std::int64_t kl = parse_int(klo, func, "lower_bandwidth");
    const std::int64_t ku = parse_int(kuo, func, "upper_bandwidth");
    if (kl < 0 || ku < 0)
        throw nb::value_error((std::string(func) + "(): the bandwidths must be >= 0, got lower_bandwidth=" +
                               std::to_string(kl) + ", upper_bandwidth=" + std::to_string(ku))
                                  .c_str());
    // (each bandwidth is below the row count first, so the sum cannot overflow)
    if (kl >= a.shape(-2) || ku >= a.shape(-2) || a.shape(-2) != kl + ku + 1)
        throw nb::value_error((std::string(func) + "(): 'ab' must have lower_bandwidth + upper_bandwidth + 1 rows (" +
                               std::to_string(kl) + " + " + std::to_string(ku) + " + 1), got " +
                               std::to_string(a.shape(-2)))
                                  .c_str());
    const lint n = a.shape(-1);
    check_rhs_rows(y, n, func);
    const Rhs r = rhs_of(y, n, func, "ab");
    Batch b({{a.batch_shape(2), a.batch_strides(2)}, {y.batch_shape(r.core), y.batch_strides(r.core)}}, func);
    const bool complex = computes_complex(a.dtype()) || computes_complex(y.dtype());
    const std::int64_t rs = a.stride(-2), cs = a.stride(-1);
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_rhs(b, r, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, (kl == 1 && ku == 1) || double(n) * (kl + ku + 1) >= kBandSplitMin ? cost(kBanded, double(n) * (kl + ku + 1)) : 0.0, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            const std::byte* base = a.data() + off[0];
            LapackSlot<T> xs(res + i * n * r.k, n * r.k, s, Scratch::kAlign0);
            T* x = xs.get();
            if (kl == 1 && ku == 1) {
                // ab[u + i - j, j] = a[i, j]: du = ab[0, 1:], d = ab[1, :], dl = ab[2, :-1].
                return tri_body<T>({a.dtype(), base + 2 * rs, cs}, {a.dtype(), base + rs, cs},
                                   {a.dtype(), base + cs, cs}, n, y, off[1], r, x, s, kNonFinite, kNonFinite,
                                   kNonFinite);
            }
            // gbsv's layout: the band in rows kl .. 2kl+ku, rows 0 .. kl-1 room for the
            // fill-in. Only the band's elements inside the matrix are read and checked
            // (DESIGN.md §2.12); the rest is zeroed.
            const lint ldab = 2 * kl + ku + 1;
            T* w = s.get<T>(static_cast<std::size_t>(ldab * n));
            std::fill(w, w + ldab * n, T(0));
            for (lint j = 0; j < n; ++j) {
                const lint r0 = std::max<lint>(0, ku - j), r1 = std::min<lint>(kl + ku, ku + n - 1 - j);
                if (!pack(a.dtype(), base + r0 * rs + j * cs, r1 - r0 + 1, 1, rs, 0, w + j * ldab + kl + r0, ldab, true))
                    return kNonFinite;
            }
            const bool b_ok = pack(y.dtype(), y.data() + off[1], n, r.k, r.rs, r.cs, x, n, true);
            if (n > 0) {
                lint* ipiv = s.get_ints(static_cast<std::size_t>(n));
                if (lapack::gbsv(n, kl, ku, r.k, w, ldab, ipiv, x, n) > 0) return kSingular;
            }
            return b_ok ? 0 : kNonFiniteB;
        });
        if (f) raise(b, f, func, "ab");
        return to_sdnp(out);
    });
}

// ------------------------------------------------------------------ sylvester --

// a x + x b = c by Bartels-Stewart: a = U R U^H, b = W S W^H (Schur), then
// R Y + Y S = U^H c W (trsyl3) and x = U Y W^H. A real a, b give the real
// quasi-triangular Schur forms, which trsyl3 handles directly.
nb::object solve_sylvester(nb::handle ao, nb::handle bo, nb::handle co) {
    const char* func = "solve_sylvester";
    Input a(ao, func, "a"), bb(bo, func, "b"), c(co, func, "c");
    require_square(a);
    require_square(bb);
    require_matrix(c);
    const lint m = a.shape(-1), n = bb.shape(-1);
    if (c.shape(-2) != m || c.shape(-1) != n)
        throw nb::value_error((std::string(func) + "(): 'c' must be " + std::to_string(m) + "x" + std::to_string(n) +
                               " to match 'a' and 'b', got " + std::to_string(c.shape(-2)) + "x" +
                               std::to_string(c.shape(-1)))
                                  .c_str());
    Batch b({{a.batch_shape(2), a.batch_strides(2)}, {bb.batch_shape(2), bb.batch_strides(2)}, {c.batch_shape(2), c.batch_strides(2)}},
            func);
    const bool complex = computes_complex(a.dtype()) || computes_complex(bb.dtype()) || computes_complex(c.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), m, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, cost(kSylvester, (double(m) * m + double(n) * n) / 2), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            const std::size_t mm = pad2(m * m), nn = pad2(n * n);  // 16-byte boundaries (aligned16)
            T* rs_ = s.get<T>(mm + nn, 0);  // R, then S
            T* uw = s.get<T>(mm + nn, 1);   // U, then W
            T *R = rs_, *S = rs_ + mm, *U = uw, *W = uw + mm;
            LapackSlot<T> xs(res + i * m * n, m * n, s, Scratch::kAlign0);
            T* x = xs.get();
            if (!pack(a.dtype(), a.data() + off[0], m, m, a.stride(-2), a.stride(-1), R, m, true)) return kNonFinite;
            if (m > 0 && lapack::gees(m, R, m, U, m, s) > 0) return kNoConvergenceEig;
            if (!pack(bb.dtype(), bb.data() + off[1], n, n, bb.stride(-2), bb.stride(-1), S, n, true))
                return kNonFiniteB;
            if (n > 0 && lapack::gees(n, S, n, W, n, s) > 0) return kNoConvergenceEig;
            if (!pack(c.dtype(), c.data() + off[2], m, n, c.stride(-2), c.stride(-1), x, m, true)) return kNonFinite2;
            if (m == 0 || n == 0) return 0;
            T* tmp = s.get<T>(static_cast<std::size_t>(m * n), 2);
            lapack::gemm('C', 'N', m, n, m, U, m, x, m, tmp, m);  // U^H c
            lapack::gemm('N', 'N', m, n, n, tmp, m, W, n, x, m);  // F = U^H c W
            double scale = 1.0;
            if (lapack::trsyl3(m, n, R, m, S, n, x, m, scale, s) == 1) return kCloseEigenvalues;
            if (scale != 1.0)  // trsyl3 solved R Y + Y S = scale F (scale < 1 avoids overflow)
                for (lint k = 0; k < m * n; ++k) x[k] /= scale;
            lapack::gemm('N', 'N', m, n, m, U, m, x, m, tmp, m);  // U Y
            lapack::gemm('N', 'C', m, n, n, tmp, m, W, n, x, m);  // x = U Y W^H
            return 0;
        });
        if (f) raise(b, f, func, "a", "c");
        return to_sdnp(out);
    });
}

}  // namespace

void bind_special(nb::module_& m) {
    m.def("solve_tridiagonal", &solve_tridiagonal, nb::arg().none(), nb::arg().none(), nb::arg().none(),
          nb::arg().none());
    m.def("solve_banded", &solve_banded, nb::arg().none(), nb::arg().none(), nb::kw_only(), nb::arg("lower_bandwidth"),
          nb::arg("upper_bandwidth"));
    m.def("solve_sylvester", &solve_sylvester, nb::arg().none(), nb::arg().none(), nb::arg().none());
}

}  // namespace sdla
