// The LU family: solve, inv, det, slogdet (docs/api/linalg.md §3, §4).
//
// Every function factors with getrf. Paths chosen by measurement
// (docs/design/linalg.md §10.2):
//   solve : getrf + getrs  (Accelerate's gesv is never faster, and 2x slower at n=2048)
//   inv   : getrf + getrs against the identity  (fastest up to n=128; inv(a) == solve(a, I) bit for bit)
#include "linalg/ops/lu.hpp"
#include "bind/errors.hpp"

#include "linalg/cost.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pyrefs.hpp"
#include "linalg/ops/common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace sd::linalg {

namespace {

// sign * exp(logabs) = det, from an LU factorization. Singular: (0, -inf).
template <class T>
void sign_logabs(const T* lu, const lint* ipiv, lint n, bool singular, T& sign, double& logabs) {
    if (singular) {
        sign = T(0);
        logabs = -std::numeric_limits<double>::infinity();
        return;
    }
    sign = T(1);
    logabs = 0.0;
    for (lint i = 0; i < n; ++i) {
        const T u = lu[i * n + i];
        const double au = std::abs(u);
        logabs += std::log(au);
        sign *= u / au;
        if (ipiv[i] != i + 1) sign = -sign;  // ipiv is 1-based: row i was swapped with ipiv[i]
    }
}

// d = m * 2^e with m's larger component in [0.5, 1) (0 stays 0).
inline void normalize(double& m, std::int64_t& e) {
    int k;
    m = std::frexp(m, &k);
    e += k;
}
inline void normalize(cplx& m, std::int64_t& e) {
    int k;
    std::frexp(std::max(std::fabs(m.real()), std::fabs(m.imag())), &k);
    m = cplx(std::ldexp(m.real(), -k), std::ldexp(m.imag(), -k));
    e += k;
}
inline bool normal(double v) { return std::isnormal(v); }
inline bool normal(cplx v) {
    // a component may be an exact 0 (a real matrix in complex arithmetic)
    return (std::isnormal(v.real()) || v.real() == 0) && (std::isnormal(v.imag()) || v.imag() == 0) && v != cplx(0);
}

// The determinant itself: the product of U's diagonal, with the swaps' sign.
template <class T>
T det_of(const T* lu, const lint* ipiv, lint n, bool singular) {
    if (singular) return T(0);
    T d = T(1);
    bool ok = true;
    for (lint i = 0; i < n; ++i) {
        d *= lu[i * n + i];
        if (ipiv[i] != i + 1) d = -d;
        ok &= normal(d);
    }
    if (ok) return d;
    // A partial product overflowed or left the normal range, although the determinant
    // may not (diag(1e200, 1e200, 1e-300)). Again with every factor and partial product
    // scaled into [0.5, 1) and the powers of 2 summed apart: scaling by a power of 2 is
    // exact, so the products round as above while they stay normal, and only the final
    // ldexp can overflow or underflow.
    d = T(1);
    std::int64_t e = 0;
    for (lint i = 0; i < n; ++i) {
        T u = lu[i * n + i];
        normalize(u, e);
        d *= u;
        normalize(d, e);
        if (ipiv[i] != i + 1) d = -d;
    }
    const int k = static_cast<int>(std::clamp<std::int64_t>(e, -4000, 4000));  // past either end: 0 or inf
    if constexpr (std::is_same_v<T, cplx>) return cplx(std::ldexp(d.real(), k), std::ldexp(d.imag(), k));
    else return std::ldexp(d, k);
}

// A body for det/slogdet: factor, tolerate singular matrices (docs/design/linalg.md §2.2).
template <class T, class Emit>
int det_body(const Input& x, lint n, std::int64_t off, Scratch& s, Emit&& emit) {
    T* a;
    lint* ipiv;
    const int code = lu_factor<T>(x, off, n, s, a, ipiv);
    if (code == kNonFinite) return code;
    emit(a, ipiv, code == kSingular);
    return 0;
}

nb::object det(nb::handle xo) {
    Input x(xo, "det", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "det");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        if (b.ndim() == 0) {
            T d{};
            const Failure f = run_batch(
                b, cost(kLU, double(n) * n),
                [&](std::int64_t, const std::int64_t* off, Scratch& s) {
                    return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                        d = det_of(lu, ipiv, n, sing);
                    });
                },
                lu_task_cap(n, 0)
            );
            if (f) raise(b, f, "det", "x");
            return py_scalar(d);
        }
        Out out = alloc_c(b.shape(), dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(
            b, cost(kLU, double(n) * n),
            [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                    res[i] = det_of(lu, ipiv, n, sing);
                });
            },
            lu_task_cap(n, 0)
        );
        if (f) raise(b, f, "det", "x");
        return to_array(out);
    });
}

nb::object slogdet(nb::handle xo) {
    Input x(xo, "slogdet", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "slogdet");
    auto make_slogdet = [](nb::object sign, nb::object logabs) { return make_result(Result::Slogdet, {sign, logabs}); };
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        if (b.ndim() == 0) {
            T sign{};
            double logabs = 0;
            const Failure f = run_batch(
                b, cost(kLU, double(n) * n),
                [&](std::int64_t, const std::int64_t* off, Scratch& s) {
                    return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                        sign_logabs(lu, ipiv, n, sing, sign, logabs);
                    });
                },
                lu_task_cap(n, 0)
            );
            if (f) raise(b, f, "slogdet", "x");
            return make_slogdet(py_scalar(sign), py_scalar(logabs));
        }
        Out sign_out = alloc_c(b.shape(), dtype_of<T>());
        Out logabs_out = alloc_c(b.shape(), DType::Float64);
        T* sign = sign_out.as<T>();
        double* logabs = logabs_out.as<double>();
        const Failure f = run_batch(
            b, cost(kLU, double(n) * n),
            [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                    sign_logabs(lu, ipiv, n, sing, sign[i], logabs[i]);
                });
            },
            lu_task_cap(n, 0)
        );
        if (f) raise(b, f, "slogdet", "x");
        return make_slogdet(to_array(sign_out), to_array(logabs_out));
    });
}

nb::object inv(nb::handle xo) {
    Input x(xo, "inv", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "inv");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), n, n, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(
            b, cost(kLU, double(n) * n),
            [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                T* a;
                lint* ipiv;
                if (const int code = lu_factor<T>(x, off[0], n, s, a, ipiv)) return code;
                LapackSlot<T> rs(res + i * n * n, n * n, s, Scratch::kAlign0);
                lu_inverse(a, ipiv, n, rs.get());  // this element's slot, filled in place
                return 0;
            },
            lu_task_cap(n, n)
        );
        if (f) raise(b, f, "inv", "x");
        return to_array(out);
    });
}

nb::object solve(nb::handle ao, nb::handle bo) {
    Input a(ao, "solve", "a");
    Input y(bo, "solve", "b");
    require_square(a);
    const lint n = a.shape(-1);
    const Rhs r = rhs_of(y, n, "solve", "a");
    const lint k = r.k;
    Batch b({{a.batch_shape(2), a.batch_strides(2)}, {y.batch_shape(r.core), y.batch_strides(r.core)}}, "solve");
    const bool complex = computes_complex(a.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = r.vec ? alloc_vectors(b.shape(), n, dtype_of<T>()) : alloc_matrices(b.shape(), n, k, dtype_of<T>());
        T* res = out.as<T>();
        const Failure f = run_batch(
            b, cost(kLU, double(n) * n) + rhs_cost(n, k),
            [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                T* lu;
                lint* ipiv;
                if (const int code = lu_factor<T>(a, off[0], n, s, lu, ipiv)) return code;
                LapackSlot<T> xs(res + i * n * k, n * k, s, Scratch::kAlign0);
                T* x = xs.get();  // b is packed straight into the output slot (docs/design/linalg.md §5.2)
                if (!pack(y.dtype(), y.data() + off[1], n, k, r.rs, r.cs, x, n, true)) return kNonFiniteB;
                lapack::lu_solve_factored(n, k, lu, ipiv, x, n);  // trsv for one or two columns
                return 0;
            },
            lu_task_cap(n, k)
        );
        if (f) raise(b, f, "solve", "a");
        return to_array(out);
    });
}

}  // namespace

void bind_lu(nb::module_& m) {
    // Unnamed args are positional-only (docs/api/linalg.md §1.1). .none() lets None reach
    // Input, which rejects every non-sd.ndarray argument with the same TypeError.
    m.def("solve", named("solve", &solve), nb::sig("def solve(a, b, /)"), nb::arg().none(), nb::arg().none());
    m.def("inv", named("inv", &inv), nb::sig("def inv(x, /)"), nb::arg().none());
    m.def("det", named("det", &det), nb::sig("def det(x, /)"), nb::arg().none());
    m.def("slogdet", named("slogdet", &slogdet), nb::sig("def slogdet(x, /)"), nb::arg().none());
}

}  // namespace sd::linalg
