// Triangular solves and the Cholesky and LU factorizations (docs/api/linalg.md §4, §5):
// solve_triangular, cholesky, cholesky_solve, lu, lu_solve.
//
// Functions that read only one triangle pack and check only that triangle
// (docs/design/linalg.md §5.4). A zero on the diagonal of a triangular factor
// is a LinAlgError everywhere, also where LAPACK itself would just divide by it.
#include "linalg/ops/factor.hpp"

#include "linalg/cost.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pyrefs.hpp"
#include "linalg/ops/common.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace sd::linalg {

namespace {


// -------------------------------------------------------- solve_triangular --

nb::object solve_triangular(nb::handle ao, nb::handle bo, bool upper, bool unit_diagonal) {
    Input a(ao, "solve_triangular", "a");
    Input y(bo, "solve_triangular", "b");
    require_square(a);
    const lint n = a.shape(-1);
    const Rhs r = rhs_of(y, n, "solve_triangular", "a");
    Batch b({{a.batch_shape(2), a.batch_strides(2)}, {y.batch_shape(r.core), y.batch_strides(r.core)}},
            "solve_triangular");
    const bool complex = computes_complex(a.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_rhs(b, r, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, cost(kTriangular, double(n) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            // Only the triangle trtrs reads; the diagonal too unless it is taken as 1.
            const T* t;
            lint ld;
            if (!triangle_operand(a, off[0], n, !upper, !unit_diagonal, s.get<T>(static_cast<std::size_t>(n * n)), t, ld))
                return kNonFinite;
            if (!unit_diagonal && zero_on_diagonal(t, n, ld)) return kSingular;
            LapackSlot<T> xs(res + i * n * r.k, n * r.k, s, Scratch::kAlign0);
            T* x = xs.get();
            if (!pack(y.dtype(), y.data() + off[1], n, r.k, r.rs, r.cs, x, n, true)) return kNonFiniteB;
            if (n > 0) lapack::trtrs(upper ? 'U' : 'L', 'N', unit_diagonal ? 'U' : 'N', n, r.k, t, ld, x, n);
            return 0;
        }, lu_task_cap(n, r.k, false));
        if (f) raise(b, f, "solve_triangular", "a");
        return to_array(out);
    });
}

// ---------------------------------------------------------------- cholesky --

nb::object cholesky(nb::handle xo, bool upper) {
    Input x(xo, "cholesky", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "cholesky");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, cost(kLU, double(n) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            LapackSlot<T> os(res + i * n * n, n * n, s, Scratch::kAlign0);
            T* o = os.get();
            // L is computed in the output slot itself; for upper=True in scratch, then U = L^H.
            T* l = upper ? s.get<T>(static_cast<std::size_t>(n * n)) : o;
            // The input's lower triangle only (docs/api/linalg.md §1.4).
            if (!pack_triangle(x.dtype(), x.data() + off[0], n, x.stride(-2), x.stride(-1), l, n, true, true, true))
                return kNonFinite;
            if (n > 0 && lapack::potrf('L', n, l, n) > 0) return kNotPD;
            for (lint j = 0; j < n; ++j)
                for (lint r = 0; r < n; ++r) {
                    if (upper) o[j * n + r] = r <= j ? conj_if_complex(l[r * n + j]) : T(0);  // U(r,j) = conj(L(j,r))
                    else if (r < j) o[j * n + r] = T(0);                                    // clear L's upper part
                }
            return 0;
        });
        if (f) raise(b, f, "cholesky", "x");
        return to_array(out);
    });
}

nb::object cholesky_solve(nb::handle lo, nb::handle bo, bool upper) {
    Input l(lo, "cholesky_solve", "l");
    Input y(bo, "cholesky_solve", "b");
    require_square(l);
    const lint n = l.shape(-1);
    const Rhs r = rhs_of(y, n, "cholesky_solve", "l");
    Batch b({{l.batch_shape(2), l.batch_strides(2)}, {y.batch_shape(r.core), y.batch_strides(r.core)}},
            "cholesky_solve");
    const bool complex = computes_complex(l.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_rhs(b, r, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, cost(kTriangular, double(n) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            const T* t;
            lint ld;
            if (!triangle_operand(l, off[0], n, !upper, true, s.get<T>(static_cast<std::size_t>(n * n)), t, ld))
                return kNonFinite;
            // potrs would divide by a zero diagonal without a word (docs/design/linalg.md §10.3).
            if (zero_on_diagonal(t, n, ld)) return kSingular;
            LapackSlot<T> xs(res + i * n * r.k, n * r.k, s, Scratch::kAlign0);
            T* x = xs.get();
            if (!pack(y.dtype(), y.data() + off[1], n, r.k, r.rs, r.cs, x, n, true)) return kNonFiniteB;
            if (n > 0 && r.k > 0) lapack::potrs(upper ? 'U' : 'L', n, r.k, t, ld, x, n);
            return 0;
        }, lu_task_cap(n, r.k, false));
        if (f) raise(b, f, "cholesky_solve", "l");
        return to_array(out);
    });
}

// ---------------------------------------------------------------------- lu --

nb::object lu(nb::handle xo) {
    Input x(xo, "lu", "x");
    require_matrix(x);
    const lint m = x.shape(-2), n = x.shape(-1), kk = std::min(m, n);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "lu");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out lo = alloc_matrices(b.shape(), m, kk, dtype_of<T>());
        Out uo = alloc_matrices(b.shape(), kk, n, dtype_of<T>());
        Out po = alloc_vectors(b.shape(), m, DType::Int64);
        T* L = lo.as<T>();
        T* U = uo.as<T>();
        std::int64_t* P = po.as<std::int64_t>();
        const Failure f = run_batch(b, cost(kLU, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* a = s.get<T>(static_cast<std::size_t>(m * n));
            lint* ipiv = s.get_ints(static_cast<std::size_t>(kk));
            if (!pack(x.dtype(), x.data() + off[0], m, n, x.stride(-2), x.stride(-1), a, m, true)) return kNonFinite;
            // A singular matrix still has an LU factorization; U just has a zero
            // on its diagonal (docs/design/linalg.md §10.2), so info > 0 is not an error here.
            if (m > 0 && n > 0) lapack::getrf(m, n, a, m, ipiv);
            T* li = L + i * m * kk;
            T* ui = U + i * kk * n;
            for (lint j = 0; j < kk; ++j)
                for (lint r = 0; r < m; ++r) li[j * m + r] = r > j ? a[j * m + r] : (r == j ? T(1) : T(0));
            for (lint j = 0; j < n; ++j)
                for (lint r = 0; r < kk; ++r) ui[j * kk + r] = r <= j ? a[j * m + r] : T(0);
            // ipiv records swaps (step r swapped rows r and ipiv[r]-1); replaying them
            // on 0..m-1 gives the rows of x that L @ U reproduces: x[perm] = L @ U.
            std::int64_t* p = P + i * m;
            for (lint r = 0; r < m; ++r) p[r] = r;
            for (lint r = 0; r < kk; ++r) std::swap(p[r], p[ipiv[r] - 1]);
            return 0;
        }, lu_task_cap(std::min(m, n), 0, true));
        if (f) raise(b, f, "lu", "x");
        return make_result(Result::LU, {to_array(lo), to_array(uo), to_array(po)});
    });
}

nb::object lu_solve(nb::handle lo, nb::handle uo, nb::handle po, nb::handle bo) {
    Input l(lo, "lu_solve", "l");
    Input u(uo, "lu_solve", "u");
    Input p(po, "lu_solve", "perm");
    Input y(bo, "lu_solve", "b");
    require_square(l);
    require_square(u);
    const lint n = l.shape(-1);
    if (u.shape(-1) != n)
        throw nb::value_error(("lu_solve(): 'l' is " + std::to_string(n) + "x" + std::to_string(n) + " but 'u' is " +
                               std::to_string(u.shape(-1)) + "x" + std::to_string(u.shape(-1)))
                                  .c_str());
    if (p.dtype() != DType::Int64) throw nb::type_error("lu_solve(): 'perm' must be an int64 array");
    if (p.shape(-1) != n)
        throw nb::value_error(("lu_solve(): 'perm' must have length " + std::to_string(n) + ", got " +
                               std::to_string(p.shape(-1)))
                                  .c_str());
    const Rhs r = rhs_of(y, n, "lu_solve", "l");
    Batch b({{l.batch_shape(2), l.batch_strides(2)},
             {u.batch_shape(2), u.batch_strides(2)},
             {p.batch_shape(1), p.batch_strides(1)},
             {y.batch_shape(r.core), y.batch_strides(r.core)}},
            "lu_solve");
    const bool complex = computes_complex(l.dtype()) || computes_complex(u.dtype()) || computes_complex(y.dtype());
    const std::int64_t ps = p.stride(-1);
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_rhs(b, r, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(b, cost(kTriangular, double(n) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            // L strictly below the diagonal (its unit diagonal is not read), U on
            // and above it: read in place when column-major, else packed
            // (docs/design/linalg.md §5.5).
            const T *tl, *tu;
            lint ldl, ldu;
            if (!triangle_operand(l, off[0], n, true, false, s.get<T>(static_cast<std::size_t>(n * n)), tl, ldl))
                return kNonFinite;
            if (!triangle_operand(u, off[1], n, false, true, s.get<T>(static_cast<std::size_t>(n * n), 3), tu, ldu))
                return kNonFiniteU;
            if (zero_on_diagonal(tu, n, ldu)) return kSingular;
            // perm must be a permutation of 0..n-1.
            lint* perm = s.get_ints(static_cast<std::size_t>(n));
            std::uint8_t* seen = s.get<std::uint8_t>(static_cast<std::size_t>(n), 1);
            std::fill(seen, seen + n, 0);
            for (lint k = 0; k < n; ++k) {
                const std::int64_t v = load<std::int64_t>(p.data() + off[2] + k * ps);
                if (v < 0 || v >= n || seen[v]) return kBadPerm;
                seen[v] = 1;
                perm[k] = v;
            }
            // x starts as b[perm] (an exact row gather), then L and U are solved
            // with no further swaps: the same steps as solve()'s getrs after getrf.
            LapackSlot<T> xs(res + i * n * r.k, n * r.k, s, Scratch::kAlign0);
            T* x = xs.get();
            const bool ok = visit_storage(y.dtype(), [&]<class S>(std::type_identity<S>) {
                if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) {
                    return false;  // unreachable
                } else {
                    bool good = true;
                    for (lint j = 0; j < r.k; ++j)
                        for (lint k = 0; k < n; ++k) {
                            const S v = load<S>(y.data() + off[3] + perm[k] * r.rs + j * r.cs);
                            if constexpr (std::is_same_v<S, double> || std::is_same_v<S, cplx>) good &= finite(v);
                            x[j * n + k] = convert<T>(v);
                        }
                    return good;
                }
            });
            if (!ok) return kNonFiniteB;
            // getrs without swaps is trtrs with L (unit) and then with U: the same
            // calls, the same bits (checked, docs/design/linalg.md §10.2), and no
            // packed L\U matrix needed.
            if (n > 0 && r.k > 0) {
                lapack::trtrs('L', 'N', 'U', n, r.k, tl, ldl, x, n);
                lapack::trtrs('U', 'N', 'N', n, r.k, tu, ldu, x, n);
            }
            return 0;
        }, lu_task_cap(n, r.k, false));
        if (f) raise(b, f, "lu_solve", "l");
        return to_array(out);
    });
}

}  // namespace

void bind_factor(nb::module_& m) {
    m.def("solve_triangular", &solve_triangular, nb::arg().none(), nb::arg().none(), nb::kw_only(),
          nb::arg("upper"), nb::arg("unit_diagonal") = false);
    m.def("cholesky", &cholesky, nb::arg().none(), nb::kw_only(), nb::arg("upper") = false);
    m.def("cholesky_solve", &cholesky_solve, nb::arg().none(), nb::arg().none(), nb::kw_only(),
          nb::arg("upper") = false);
    m.def("lu", &lu, nb::arg().none());
    m.def("lu_solve", &lu_solve, nb::arg().none(), nb::arg().none(), nb::arg().none(), nb::arg().none());
}

}  // namespace sd::linalg
