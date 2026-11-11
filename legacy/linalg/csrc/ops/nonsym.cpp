// General eigenvalues, Schur, Hessenberg and LDL (API.md §4; ROADMAP.md M6):
// eig, eigvals, schur, hessenberg, ldl.
//
//   eig, eigvals : geev ('V' / 'N': values-only mode for eigvals, as in M5)
//   schur        : gees, no sorting
//   hessenberg   : gehrd + orghr/unghr
//   ldl          : sytrf/hetrf (Bunch-Kaufman), chosen over sytrf_rook by
//                  measurement (experiments/m6_ldl_driver, DESIGN.md §2.15)
#include "ops/nonsym.hpp"

#include "core/outblock.hpp"
#include "core/pyrefs.hpp"
#include "ops/common.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace sdla {

namespace {

template <class T>
T conj_if_complex(T v) {
    if constexpr (std::is_same_v<T, cplx>) return std::conj(v);
    else return v;
}

// ------------------------------------------------------------ eig, eigvals --

// Eigenvalues (and vectors) of one matrix in `w` / `v` (complex, unsorted).
// Returns 0, kNonFinite or kNoConvergenceEig.
template <class T>
int eig_one(const Input& x, std::int64_t off, lint n, bool vectors, Scratch& s, cplx* w, cplx* v) {
    T* a = s.get<T>(static_cast<std::size_t>(n * n));
    if (!pack(x.dtype(), x.data() + off, n, n, x.stride(-2), x.stride(-1), a, n, true)) return kNonFinite;
    if (n == 0) return 0;
    const char jobvr = vectors ? 'V' : 'N';
    if constexpr (std::is_same_v<T, cplx>) {
        cplx dummy{};
        if (lapack::geev(jobvr, n, a, n, w, vectors ? v : &dummy, vectors ? n : 1, s) > 0) return kNoConvergenceEig;
    } else {
        double* wr = s.get<double>(static_cast<std::size_t>(2 * n), 1);
        double* wi = wr + n;
        double dummy = 0;
        double* vr = vectors ? s.get<double>(static_cast<std::size_t>(n * n), 2) : &dummy;
        if (lapack::geev(jobvr, n, a, n, wr, wi, vr, vectors ? n : 1, s) > 0) return kNoConvergenceEig;
        // A real matrix's complex eigenvalues come in conjugate pairs (wi > 0 first);
        // the pair's vectors are stored as two real columns: re in j, im in j+1.
        for (lint j = 0; j < n; ++j) {
            w[j] = cplx(wr[j], wi[j]);
            if (!vectors) continue;
            if (wi[j] == 0) {
                for (lint r = 0; r < n; ++r) v[j * n + r] = cplx(vr[j * n + r], 0.0);
            } else {
                for (lint r = 0; r < n; ++r) {
                    v[j * n + r] = cplx(vr[j * n + r], vr[(j + 1) * n + r]);
                    v[(j + 1) * n + r] = cplx(vr[j * n + r], -vr[(j + 1) * n + r]);
                }
                w[j + 1] = cplx(wr[j + 1], wi[j + 1]);
                ++j;
            }
        }
    }
    return 0;
}

nb::object eig_impl(nb::handle xo, bool vectors) {
    const char* func = vectors ? "eig" : "eigvals";
    Input x(xo, func, "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, func);
    // Always complex128, whatever the input (API.md §0.3).
    Out wo = alloc_vectors(b.shape(), n, DType::Complex128);
    Out vo;
    if (vectors) vo = alloc_matrices(b.shape(), n, n, DType::Complex128);
    const bool complex = computes_complex(x.dtype());
    const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
        // Unsorted results in scratch (slot 3 and the int-free complex area), then sorted out.
        cplx* w = s.get<cplx>(static_cast<std::size_t>(n * (vectors ? n + 1 : 1)), 3);
        cplx* v = w + n;
        const int code = complex ? eig_one<cplx>(x, off[0], n, vectors, s, w, v)
                                 : eig_one<double>(x, off[0], n, vectors, s, w, v);
        if (code) return code;
        // Sorted by (real, imag), stably -- stardust-numpy's complex order (S4).
        std::vector<lint> idx(static_cast<std::size_t>(n));
        std::iota(idx.begin(), idx.end(), lint(0));
        std::stable_sort(idx.begin(), idx.end(), [&](lint p, lint q) {
            return w[p].real() < w[q].real() || (w[p].real() == w[q].real() && w[p].imag() < w[q].imag());
        });
        cplx* wout = wo.as<cplx>() + i * n;
        for (lint j = 0; j < n; ++j) wout[j] = w[idx[j]];
        if (vectors) {
            cplx* vout = vo.as<cplx>() + i * n * n;
            for (lint j = 0; j < n; ++j) std::copy(v + idx[j] * n, v + idx[j] * n + n, vout + j * n);
        }
        return 0;
    });
    if (f) raise(b, f, func, "x");
    if (vectors) return make_result(Result::Eig, {to_sdnp(wo), to_sdnp(vo)});
    return to_sdnp(wo);
}

nb::object eig(nb::handle x) { return eig_impl(x, true); }
nb::object eigvals(nb::handle x) { return eig_impl(x, false); }

// ------------------------------------------------------ schur, hessenberg --

nb::object schur(nb::handle xo) {
    Input x(xo, "schur", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "schur");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out to = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        Out zo = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* t = to.as<T>() + i * n * n;  // packed into T's slot; gees overwrites it with T
            if (!pack(x.dtype(), x.data() + off[0], n, n, x.stride(-2), x.stride(-1), t, n, true)) return kNonFinite;
            if (n == 0) return 0;
            return lapack::gees(n, t, n, zo.as<T>() + i * n * n, n, s) > 0 ? kNoConvergenceEig : 0;
        });
        if (f) raise(b, f, "schur", "x");
        return make_result(Result::Schur, {to_sdnp(to), to_sdnp(zo)});
    });
}

nb::object hessenberg(nb::handle xo) {
    Input x(xo, "hessenberg", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "hessenberg");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out ho = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        Out qo = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* a = s.get<T>(static_cast<std::size_t>(n * n));
            if (!pack(x.dtype(), x.data() + off[0], n, n, x.stride(-2), x.stride(-1), a, n, true)) return kNonFinite;
            if (n == 0) return 0;
            T* tau = s.get<T>(static_cast<std::size_t>(std::max<lint>(n - 1, 1)), 1);
            lapack::gehrd(n, a, n, tau, s);
            // H: on and above the subdiagonal; below it are the reflectors.
            T* h = ho.as<T>() + i * n * n;
            for (lint j = 0; j < n; ++j)
                for (lint r = 0; r < n; ++r) h[j * n + r] = r <= j + 1 ? a[j * n + r] : T(0);
            // Q: the reflectors go to Q's slot, and orghr turns them into Q there.
            T* q = qo.as<T>() + i * n * n;
            std::copy(a, a + n * n, q);
            lapack::orghr(n, q, n, tau, s);
            return 0;
        });
        if (f) raise(b, f, "hessenberg", "x");
        return make_result(Result::Hessenberg, {to_sdnp(ho), to_sdnp(qo)});
    });
}

// --------------------------------------------------------------------- ldl --

// sytrf (lower) gives A = M D M^H with M = P1 L1 P2 L2 ..., a product of
// row interchanges and unit lower block columns, recorded in ipiv:
//   ipiv[k] > 0           : a 1x1 block at k; rows k and ipiv[k]-1 were swapped
//   ipiv[k] = ipiv[k+1] < 0: a 2x2 block at k, k+1; rows k+1 and -ipiv[k+1]-1 swapped
// We return the triangular L = M[perm] with x[perm][:, perm] = L D L^H (API.md §4).
// The same reading as scipy.linalg.ldl (its _ldl_sanitize_ipiv and
// _ldl_construct_tri_factor), so perm, L and D agree with scipy.
template <class T>
void ldl_unpack(const T* a, const lint* ipiv, lint n, Scratch& s, T* L, T* D, std::int64_t* perm_out) {
    lint* swap = s.get<lint>(static_cast<std::size_t>(n), 1);
    lint* piv = s.get<lint>(static_cast<std::size_t>(n), 2);  // 1: 1x1, 2: first of a 2x2, 0: its second
    for (lint k = 0; k < n; ++k) swap[k] = k;
    for (lint k = 0; k < n;) {
        if (ipiv[k] > 0) {
            swap[k] = ipiv[k] - 1;
            piv[k] = 1;
            k += 1;
        } else {
            swap[k + 1] = -ipiv[k + 1] - 1;
            piv[k] = 2;
            piv[k + 1] = 0;
            k += 2;
        }
    }
    // D: the diagonal blocks. M (in L's slot): identity plus the multipliers
    // below the blocks (a 2x2 block's off-diagonal element belongs to D).
    std::fill(D, D + n * n, T(0));
    std::fill(L, L + n * n, T(0));
    for (lint j = 0; j < n; ++j) {
        D[j * n + j] = a[j * n + j];
        L[j * n + j] = T(1);
        const bool block2 = piv[j] == 2;
        if (block2) {
            D[j * n + j + 1] = a[j * n + j + 1];
            D[(j + 1) * n + j] = conj_if_complex(a[j * n + j + 1]);
        }
        for (lint r = j + (block2 ? 2 : 1); r < n; ++r) L[j * n + r] = a[j * n + r];
    }
    // Replay the interchanges from the last step back to the first, on M's rows
    // (only the columns at and right of the step) and on a running permutation.
    std::int64_t* perm = reinterpret_cast<std::int64_t*>(s.get<lint>(static_cast<std::size_t>(n), 4));
    for (lint k = 0; k < n; ++k) perm[k] = k;
    for (lint k = n - 1; k >= 0; --k) {
        const lint t = swap[k];
        if (t == k) continue;
        const lint c0 = piv[k] == 0 ? k - 1 : k;
        for (lint c = c0; c < n; ++c) std::swap(L[c * n + k], L[c * n + t]);
        std::swap(perm[k], perm[t]);
    }
    // perm_out = argsort(perm); M[perm_out] is lower triangular: gather its rows.
    for (lint k = 0; k < n; ++k) perm_out[perm[k]] = k;
    T* m = s.get<T>(static_cast<std::size_t>(n * n), 3);
    std::copy(L, L + n * n, m);
    for (lint c = 0; c < n; ++c)
        for (lint r = 0; r < n; ++r) L[c * n + r] = m[c * n + perm_out[r]];
}

nb::object ldl(nb::handle xo) {
    Input x(xo, "ldl", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "ldl");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out lo = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        Out dout = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        Out po = alloc_vectors(b.shape(), n, DType::Int64);
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* a = s.get<T>(static_cast<std::size_t>(n * n));
            // The lower triangle only (API.md §0.2), checked as read (DESIGN.md §2.12).
            if (!pack_triangle(x.dtype(), x.data() + off[0], n, x.stride(-2), x.stride(-1), a, n, true, true, true))
                return kNonFinite;
            if (n == 0) return 0;
            lint* ipiv = s.get_ints(static_cast<std::size_t>(n));
            lapack::sytrf(n, a, n, ipiv, s);  // a zero block of D is not an error (like lu, DESIGN.md §2.15)
            ldl_unpack(a, ipiv, n, s, lo.as<T>() + i * n * n, dout.as<T>() + i * n * n, po.as<std::int64_t>() + i * n);
            return 0;
        });
        if (f) raise(b, f, "ldl", "x");
        return make_result(Result::LDL, {to_sdnp(lo), to_sdnp(dout), to_sdnp(po)});
    });
}

}  // namespace

void bind_nonsym(nb::module_& m) {
    m.def("eig", &eig, nb::arg().none());
    m.def("eigvals", &eigvals, nb::arg().none());
    m.def("schur", &schur, nb::arg().none());
    m.def("hessenberg", &hessenberg, nb::arg().none());
    m.def("ldl", &ldl, nb::arg().none());
}

}  // namespace sdla
