// Hermitian eigenvalues and the SVD, and what is built on them
// (docs/api/linalg.md §3, §4, §5): eigh, eigvalsh, svd, svdvals, matrix_rank, pinv, cond.
//
// Drivers, chosen by measurement (docs/design/linalg.md §10.1, §10.5):
//   eigh            syevd/heevd  (vectors: fastest at every size; 2x syevr at n=1000)
//   eigh(a, b)      sygvd/hegvd  (vectors: 8x sygv at n=1000)
//   eigvalsh        syevd 'N'    (values only; all drivers alike, the same family as eigh)
//   eigvalsh(a, b)  sygv 'N'     (values only: 2x faster than sygvd 'N' at n=1000)
//   svd             gesdd        (10x gesvd at n=1000)
//   svdvals         gesvd 'N'    (values only: 5-25% faster than gesdd 'N')
// Values-only modes give values that differ from the ones computed with the
// vectors at rounding level (docs/design/linalg.md §10.5).
#include "linalg/ops/spectral.hpp"

#include "linalg/args.hpp"
#include "linalg/cost.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pairwise.hpp"
#include "linalg/pyrefs.hpp"
#include "linalg/ops/common.hpp"
#include "linalg/ops/svals.hpp"
#include "linalg/ops/vnorm.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace sd::linalg {

namespace {

// ----------------------------------------------------------- eigh, eigvalsh --

nb::object eigh_impl(nb::handle ao, nb::handle bo, bool vectors) {
    const char* func = vectors ? "eigh" : "eigvalsh";
    Input a(ao, func, "a");
    require_square(a);
    const lint n = a.shape(-1);
    const bool gen = !bo.is_none();
    std::optional<Input> bin;
    if (gen) {
        bin.emplace(bo, func, "b");
        require_square(*bin);
        if (bin->shape(-1) != n)
            throw nb::value_error((std::string(func) + "(): 'a' is " + std::to_string(n) + "x" + std::to_string(n) +
                                   " but 'b' is " + std::to_string(bin->shape(-1)) + "x" +
                                   std::to_string(bin->shape(-1)))
                                      .c_str());
    }
    Batch b = gen ? Batch({{a.batch_shape(2), a.batch_strides(2)}, {bin->batch_shape(2), bin->batch_strides(2)}}, func)
                  : Batch({{a.batch_shape(2), a.batch_strides(2)}}, func);
    const bool complex = computes_complex(a.dtype()) || (gen && computes_complex(bin->dtype()));
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out wo = alloc_vectors(b.shape(), n, DType::Float64);
        Out vo;
        if (vectors) vo = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        double* W = wo.as<double>();
        const Failure f = run_batch(b, cost(kEigh, double(n) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            // With vectors, `a` is packed straight into the eigenvector slot; LAPACK
            // overwrites it with the eigenvectors (docs/design/linalg.md §5.2).
            LapackSlot<T> vs(vectors ? vo.as<T>() + i * n * n : s.get<T>(static_cast<std::size_t>(n * n)), n * n, s,
                             Scratch::kAlign0);
            T* av = vs.get();
            // Lower triangles only (docs/api/linalg.md §1.4), checked as read
            // (docs/design/linalg.md §5.4).
            if (!pack_triangle(a.dtype(), a.data() + off[0], n, a.stride(-2), a.stride(-1), av, n, true, true, true))
                return kNonFinite;
            T* bv = nullptr;
            if (gen) {
                bv = s.get<T>(static_cast<std::size_t>(n * n), 1);
                if (!pack_triangle(bin->dtype(), bin->data() + off[1], n, bin->stride(-2), bin->stride(-1), bv, n,
                                   true, true, true))
                    return kNonFiniteB;
            }
            if (n == 0) return 0;
            const char jobz = vectors ? 'V' : 'N';
            LapackSlot<double> ws(W + i * n, n, s, Scratch::kAlign1);
            double* w = ws.get();
            if (!gen) return lapack::syevd(jobz, n, av, n, w, s) > 0 ? kNoConvergenceEig : 0;
            const lint info = lapack::sygv(vectors, jobz, n, av, n, bv, n, w, s);
            if (info > n) return kNotPD;  // b is not positive definite
            return info > 0 ? kNoConvergenceEig : 0;
        });
        if (f) raise(b, f, func, "a");
        if (vectors) return make_result(Result::Eigh, {to_sdnp(wo), to_sdnp(vo)});
        return to_sdnp(wo);
    });
}

nb::object eigh(nb::handle a, nb::handle b) { return eigh_impl(a, b, true); }
nb::object eigvalsh(nb::handle a, nb::handle b) { return eigh_impl(a, b, false); }

// ---------------------------------------------------------- svd, svdvals --

bool parse_mode(nb::handle o, const char* func) {
    if (!PyUnicode_Check(o.ptr()))
        throw nb::type_error((std::string(func) + "(): 'mode' must be 'reduced' or 'complete'").c_str());
    const std::string s = nb::cast<std::string>(o);
    if (s == "reduced") return false;
    if (s == "complete") return true;
    throw nb::value_error((std::string(func) + "(): 'mode' must be 'reduced' or 'complete', got '" + s + "'").c_str());
}

template <class T>
void set_identity(T* a, lint n) {
    std::fill(a, a + n * n, T(0));
    for (lint d = 0; d < n; ++d) a[d * n + d] = T(1);
}

nb::object svd(nb::handle xo, nb::handle mode_o) {
    Input x(xo, "svd", "x");
    require_matrix(x);
    const bool complete = parse_mode(mode_o, "svd");
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    const lint ucols = complete ? m : k, vrows = complete ? n : k;
    const lint lda = std::max<lint>(m, 1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "svd");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out uo = alloc_matrices(b.shape(), m, ucols, dtype_of<T>());
        Out so = alloc_vectors(b.shape(), k, DType::Float64);
        Out vo = alloc_matrices(b.shape(), vrows, n, dtype_of<T>());
        const Failure f = run_batch(b, cost(kSVD, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* a = s.get<T>(static_cast<std::size_t>(lda * n));
            if (!pack(x.dtype(), x.data() + off[0], m, n, x.stride(-2), x.stride(-1), a, lda, true)) return kNonFinite;
            LapackSlot<T> us(uo.as<T>() + i * m * ucols, m * ucols, s, Scratch::kAlign0);
            LapackSlot<T> vs(vo.as<T>() + i * vrows * n, vrows * n, s, Scratch::kAlign1);
            LapackSlot<double> ss(so.as<double>() + i * k, k, s, Scratch::kAlign2);
            T* u = us.get();
            T* vh = vs.get();
            if (k == 0) {  // no singular values; complete U and Vh are identities
                if (complete) {
                    set_identity(u, m);
                    set_identity(vh, n);
                }
                return 0;
            }
            // gesdd writes U and Vh straight into their output slots.
            return lapack::gesdd(complete ? 'A' : 'S', m, n, a, lda, ss.get(), u, m, vh, vrows, s) > 0
                       ? kNoConvergence
                       : 0;
        });
        if (f) raise(b, f, "svd", "x");
        return make_result(Result::SVD, {to_sdnp(uo), to_sdnp(so), to_sdnp(vo)});
    });
}

nb::object svdvals(nb::handle xo) {
    Input x(xo, "svdvals", "x");
    require_matrix(x);
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "svdvals");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out so = alloc_vectors(b.shape(), k, DType::Float64);
        const Failure f = run_batch(b, cost(kSVD, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            LapackSlot<double> ss(so.as<double>() + i * k, k, s, Scratch::kAlign0);
            return singular_values<T>(x, off[0], m, n, s, ss.get());
        });
        if (f) raise(b, f, "svdvals", "x");
        return to_sdnp(so);
    });
}

// ------------------------------------------------------ matrix_rank, pinv --

nb::object matrix_rank(nb::handle xo, nb::handle rtol_o) {
    Input x(xo, "matrix_rank", "x");
    require_matrix(x);
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    const double rtol = parse_rtol(rtol_o, m, n, "matrix_rank");
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "matrix_rank");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out ro = alloc_c(b.shape(), DType::Int64);
        std::int64_t* R = ro.as<std::int64_t>();
        const Failure f = run_batch(b, cost(kSVD, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            double* sv = s.get<double>(static_cast<std::size_t>(std::max<lint>(k, 1)), 1);
            if (const int code = singular_values<T>(x, off[0], m, n, s, sv)) return code;
            // Singular values <= rtol * the largest count as zero (docs/api/linalg.md §3,
            // docs/design/linalg.md §10.5);
            // with rtol >= 1 that is all of them.
            const double tol = k > 0 ? rtol * sv[0] : 0.0;
            std::int64_t r = 0;
            for (lint j = 0; j < k; ++j) r += sv[j] > tol;
            R[i] = r;
            return 0;
        });
        if (f) raise(b, f, "matrix_rank", "x");
        return b.ndim() == 0 ? py_scalar(R[0]) : to_sdnp(ro);
    });
}

template <class T>
T conj_if_complex(T v) {
    if constexpr (std::is_same_v<T, cplx>) return std::conj(v);
    else return v;
}

nb::object pinv(nb::handle xo, nb::handle rtol_o) {
    Input x(xo, "pinv", "x");
    require_matrix(x);
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    const double rtol = parse_rtol(rtol_o, m, n, "pinv");
    const lint lda = std::max<lint>(m, 1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "pinv");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out po = alloc_matrices(b.shape(), n, m, dtype_of<T>());
        const Failure f = run_batch(b, cost(kSVD, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            LapackSlot<T> os(po.as<T>() + i * n * m, n * m, s, Scratch::kAlign0);
            T* out = os.get();
            T* a = s.get<T>(static_cast<std::size_t>(lda * n));
            if (!pack(x.dtype(), x.data() + off[0], m, n, x.stride(-2), x.stride(-1), a, lda, true)) return kNonFinite;
            if (k == 0) {
                std::fill(out, out + n * m, T(0));
                return 0;
            }
            // x = U S Vh, so pinv(x) = Vh^H S^+ U^H with S^+ inverting only the
            // singular values above rtol * the largest (the same rule as matrix_rank).
            T* u = s.get<T>(static_cast<std::size_t>(m * k), 1);
            T* vh = s.get<T>(static_cast<std::size_t>(k * n), 2);
            double* sv = s.get<double>(static_cast<std::size_t>(k), 3);
            if (lapack::gesdd('S', m, n, a, lda, sv, u, m, vh, k, s) > 0) return kNoConvergence;
            const double tol = rtol * sv[0];
            // W = Vh^H S^+ (n x k), built in `a`'s storage (no longer needed).
            T* w = a;
            for (lint j = 0; j < k; ++j) {
                const double inv = sv[j] > tol ? 1.0 / sv[j] : 0.0;
                for (lint r = 0; r < n; ++r) w[j * n + r] = conj_if_complex(vh[r * k + j]) * inv;
            }
            lapack::gemm<T>('N', std::is_same_v<T, cplx> ? 'C' : 'T', n, m, k, w, n, u, m, out, n);
            return 0;
        });
        if (f) raise(b, f, "pinv", "x");
        return to_sdnp(po);
    });
}

// ------------------------------------------------------------------- cond --

enum class CondOrd { Two, NegTwo, Fro, Nuc, One, NegOne, Inf, NegInf };

CondOrd parse_cond_ord(nb::handle o) {
    if (PyUnicode_Check(o.ptr())) {
        const std::string s = nb::cast<std::string>(o);
        if (s == "fro") return CondOrd::Fro;
        if (s == "nuc") return CondOrd::Nuc;
    } else if (!PyBool_Check(o.ptr()) && (PyLong_Check(o.ptr()) || PyFloat_Check(o.ptr()))) {
        const double p = PyFloat_AsDouble(o.ptr());
        if (p == -1.0 && PyErr_Occurred()) throw nb::python_error();  // an int too large for a double
        if (p == 2) return CondOrd::Two;
        if (p == -2) return CondOrd::NegTwo;
        if (p == 1) return CondOrd::One;
        if (p == -1) return CondOrd::NegOne;
        if (p == kInf) return CondOrd::Inf;
        if (p == -kInf) return CondOrd::NegInf;
    }
    throw nb::value_error("cond(): 'ord' must be 'fro', 'nuc', 1, -1, 2, -2, inf or -inf");
}

// The `ord` norm of an n x n column-major buffer (fro, nuc, +-1, +-inf).
template <class T>
int square_norm(CondOrd ord, const T* a, lint n, Scratch& s, double& out) {
    // The same sums as matrix_norm (pairwise, logical C order), on the
    // column-major buffer: element (i, j) at a[j * n + i].
    const std::byte* base = reinterpret_cast<const std::byte*>(a);
    constexpr std::int64_t sz = sizeof(T);
    if (ord == CondOrd::Fro) {
        if constexpr (std::is_same_v<T, cplx>) {
            auto w = make_walk<double>(base, {n, n, 2}, {sz, n * sz, 8}, [](double v) { return std::fabs(v); });
            out = vnorm({Ord::Two, 2.0}, w.count(), w);
        } else {
            auto w = make_walk<double>(base, {n, n}, {sz, n * sz}, [](double v) { return std::fabs(v); });
            out = vnorm({Ord::Two, 2.0}, w.count(), w);
        }
        return 0;
    }
    if (ord == CondOrd::Nuc) {
        T* c = s.get<T>(static_cast<std::size_t>(n * n), 3);  // gesvd destroys its input
        std::copy(a, a + n * n, c);
        double* sv = s.get<double>(static_cast<std::size_t>(n), 1);
        if (lapack::gesvd_values(n, n, c, n, sv, s) > 0) return kNoConvergence;
        out = pairwise_sum<double>(n, [&](std::int64_t b0, std::int64_t len, double* o) {
            std::copy(sv + b0, sv + b0 + len, o);
        });
        return 0;
    }
    const bool cols = ord == CondOrd::One || ord == CondOrd::NegOne;
    const bool max = ord == CondOrd::One || ord == CondOrd::Inf;
    double best = max ? 0.0 : kInf;
    for (lint l = 0; l < n; ++l) {
        // column l: (0..n-1, l), step 1 element; row l: (l, 0..n-1), step n elements
        auto w = make_walk<T>(base + (cols ? l * n * sz : l * sz), {n}, {cols ? sz : n * sz},
                              [](T v) { return magnitude(v); });
        const double sum = vnorm({Ord::One, 1.0}, n, w);
        best = max ? std::max(best, sum) : std::min(best, sum);
    }
    out = best;
    return 0;
}

nb::object cond(nb::handle xo, nb::handle ord_o) {
    Input x(xo, "cond", "x");
    require_matrix(x);
    const CondOrd ord = parse_cond_ord(ord_o);
    const lint m = x.shape(-2), n = x.shape(-1), k = std::min(m, n);
    const bool by_svd = ord == CondOrd::Two || ord == CondOrd::NegTwo;
    if (!by_svd) require_square(x);  // norm(x) * norm(inv(x)) needs an inverse
    if (k == 0) throw nb::value_error("cond(): the condition number of an empty matrix is not defined");
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "cond");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out co = alloc_c(b.shape(), DType::Float64);
        double* C = co.as<double>();
        const Failure f = run_batch(b, cost(kSVD, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            if (by_svd) {
                double* sv = s.get<double>(static_cast<std::size_t>(k), 1);
                if (const int code = singular_values<T>(x, off[0], m, n, s, sv)) return code;
                // A singular matrix has an infinite condition number (docs/design/linalg.md §10.5);
                // ord=-2 is its reciprocal, 0. (sv[0] = 0 means x = 0: singular too.)
                const double hi = sv[0], lo = sv[k - 1];
                if (ord == CondOrd::Two) C[i] = lo == 0 ? kInf : hi / lo;
                else C[i] = lo == 0 ? 0.0 : lo / hi;
                return 0;
            }
            // norm(x) * norm(inv(x)); the inverse as inv() computes it.
            T* lu;
            lint* ipiv;
            const int code = lu_factor<T>(x, off[0], n, s, lu, ipiv);
            if (code == kNonFinite) return code;
            if (code == kSingular) {
                C[i] = kInf;
                return 0;
            }
            // x itself again (lu_factor overwrote its copy), then its inverse.
            T* xa = s.get<T>(static_cast<std::size_t>(n * n), 2);
            pack(x.dtype(), x.data() + off[0], n, n, x.stride(-2), x.stride(-1), xa, n, false);
            double nx = 0, ni = 0;
            if (const int c = square_norm(ord, xa, n, s, nx)) return c;
            T* xi = s.get<T>(static_cast<std::size_t>(n * n), 2);
            lu_inverse(lu, ipiv, n, xi);
            if (const int c = square_norm(ord, xi, n, s, ni)) return c;
            C[i] = nx * ni;
            return 0;
        }, by_svd ? 0 : lu_task_cap(n, n, true));  // the other ords invert like inv
        if (f) raise(b, f, "cond", "x");
        return b.ndim() == 0 ? py_scalar(C[0]) : to_sdnp(co);
    });
}

}  // namespace

void bind_spectral(nb::module_& m) {
    m.def("eigh", &eigh, nb::arg().none(), nb::arg().none() = nb::none());
    m.def("eigvalsh", &eigvalsh, nb::arg().none(), nb::arg().none() = nb::none());
    m.def("svd", &svd, nb::arg().none(), nb::kw_only(), nb::arg("mode") = "reduced");
    m.def("svdvals", &svdvals, nb::arg().none());
    m.def("matrix_rank", &matrix_rank, nb::arg().none(), nb::kw_only(), nb::arg("rtol").none() = nb::none());
    m.def("pinv", &pinv, nb::arg().none(), nb::kw_only(), nb::arg("rtol").none() = nb::none());
    m.def("cond", &cond, nb::arg().none(), nb::kw_only(), nb::arg("ord") = 2);
}

}  // namespace sd::linalg
