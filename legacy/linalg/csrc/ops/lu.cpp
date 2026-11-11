// The LU family: solve, inv, det, slogdet (API.md §2, §3; ROADMAP.md M1).
//
// Every function factors with getrf. Paths chosen by measurement in M1
// (experiments/m1_solve_driver, DESIGN.md §2.10):
//   solve : getrf + getrs  (Accelerate's gesv is never faster, and 2x slower at n=2048)
//   inv   : getrf + getrs against the identity  (fastest up to n=128; inv(a) == solve(a, I) bit for bit)
#include "ops/lu.hpp"

#include "core/batch.hpp"
#include "core/errors.hpp"
#include "core/input.hpp"
#include "core/outblock.hpp"
#include "core/pack.hpp"
#include "core/pyrefs.hpp"
#include "lapack/lapack.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

namespace sdla {

namespace {

// Failure codes of the batch bodies.
constexpr int kNonFinite = 1;   // in the matrix being factored
constexpr int kSingular = 2;
constexpr int kNonFiniteB = 3;  // in solve's right-hand side

template <class F>
nb::object with_type(bool complex, F&& f) {
    if (complex) return f(std::type_identity<cplx>{});
    return f(std::type_identity<double>{});
}

// Turns a batch failure into the Python exception (DESIGN.md §2.4).
[[noreturn]] void raise(const Batch& b, const Failure& f, const char* func, const char* arg) {
    if (f.code == kNonFinite)
        throw nb::value_error(
            failure_message(b, f.index, (std::string(func) + "(): '" + arg + "' contains NaN or infinity").c_str())
                .c_str());
    throw LinAlgError(failure_message(b, f.index, "singular matrix"));
}

// Packs the square matrix of `x` at byte offset `off` into scratch and factors it.
// Returns the failure code (0, kNonFinite or kSingular); `a` and `ipiv` point into scratch.
template <class T>
int factor(const Input& x, std::int64_t off, lint n, Scratch& s, T*& a, lint*& ipiv) {
    a = s.get<T>(static_cast<std::size_t>(n * n));
    ipiv = s.get_ints(static_cast<std::size_t>(n));
    if (!pack(x.dtype(), x.data() + off, n, n, x.stride(-2), x.stride(-1), a, n, true)) return kNonFinite;
    if (n == 0) return 0;
    return lapack::getrf(n, n, a, n, ipiv) > 0 ? kSingular : 0;
}

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

// The determinant itself: the product of U's diagonal, with the swaps' sign.
template <class T>
T det_of(const T* lu, const lint* ipiv, lint n, bool singular) {
    if (singular) return T(0);
    T d = T(1);
    for (lint i = 0; i < n; ++i) {
        d *= lu[i * n + i];
        if (ipiv[i] != i + 1) d = -d;
    }
    return d;
}

// A body for det/slogdet: factor, tolerate singular matrices (DESIGN.md §2.4).
template <class T, class Emit>
int det_body(const Input& x, lint n, std::int64_t off, Scratch& s, Emit&& emit) {
    T* a;
    lint* ipiv;
    const int code = factor<T>(x, off, n, s, a, ipiv);
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
            const Failure f = run_batch(b, [&](std::int64_t, const std::int64_t* off, Scratch& s) {
                return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                    d = det_of(lu, ipiv, n, sing);
                });
            });
            if (f) raise(b, f, "det", "x");
            return py_scalar(d);
        }
        Out out = alloc_scalars(b.shape(), std::is_same_v<T, cplx>);
        T* res = reinterpret_cast<T*>(out.data);
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                res[i] = det_of(lu, ipiv, n, sing);
            });
        });
        if (f) raise(b, f, "det", "x");
        return to_sdnp(out);
    });
}

nb::object slogdet(nb::handle xo) {
    Input x(xo, "slogdet", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "slogdet");
    auto make_result = [](nb::object sign, nb::object logabs) {
        PyObject* args[] = {sign.ptr(), logabs.ptr()};
        PyObject* r = PyObject_Vectorcall(refs().slogdet_result, args, 2, nullptr);
        if (!r) throw nb::python_error();
        return nb::steal(r);
    };
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        if (b.ndim() == 0) {
            T sign{};
            double logabs = 0;
            const Failure f = run_batch(b, [&](std::int64_t, const std::int64_t* off, Scratch& s) {
                return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                    sign_logabs(lu, ipiv, n, sing, sign, logabs);
                });
            });
            if (f) raise(b, f, "slogdet", "x");
            return make_result(py_scalar(sign), py_scalar(logabs));
        }
        Out sign_out = alloc_scalars(b.shape(), std::is_same_v<T, cplx>);
        Out logabs_out = alloc_scalars(b.shape(), false);
        T* sign = reinterpret_cast<T*>(sign_out.data);
        double* logabs = reinterpret_cast<double*>(logabs_out.data);
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            return det_body<T>(x, n, off[0], s, [&](const T* lu, const lint* ipiv, bool sing) {
                sign_logabs(lu, ipiv, n, sing, sign[i], logabs[i]);
            });
        });
        if (f) raise(b, f, "slogdet", "x");
        return make_result(to_sdnp(sign_out), to_sdnp(logabs_out));
    });
}

nb::object inv(nb::handle xo) {
    Input x(xo, "inv", "x");
    require_square(x);
    const lint n = x.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "inv");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), n, n, std::is_same_v<T, cplx>);
        T* res = reinterpret_cast<T*>(out.data);
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* a;
            lint* ipiv;
            if (const int code = factor<T>(x, off[0], n, s, a, ipiv)) return code;
            if (n == 0) return 0;
            T* r = res + i * n * n;  // this element's slot; the output is filled in place
            for (lint k = 0; k < n * n; ++k) r[k] = T(0);
            for (lint k = 0; k < n; ++k) r[k * n + k] = T(1);
            lapack::getrs('N', n, n, a, n, ipiv, r, n);
            return 0;
        });
        if (f) raise(b, f, "inv", "x");
        return to_sdnp(out);
    });
}

nb::object solve(nb::handle ao, nb::handle bo) {
    Input a(ao, "solve", "a");
    Input y(bo, "solve", "b");
    require_square(a);
    const lint n = a.shape(-1);
    // API.md §0.2: a 1-D b is one vector (M,); anything else is a stack of matrices (..., M, K).
    const bool vec = y.ndim() == 1;
    const lint rows = vec ? y.shape(0) : y.shape(-2);
    const lint k = vec ? 1 : y.shape(-1);
    if (rows != n)
        throw nb::value_error(("solve(): 'a' is " + std::to_string(n) + "x" + std::to_string(n) + " but 'b' has " +
                               std::to_string(rows) + " rows")
                                  .c_str());
    const int bcore = vec ? 1 : 2;
    Batch b({{a.batch_shape(2), a.batch_strides(2)}, {y.batch_shape(bcore), y.batch_strides(bcore)}}, "solve");
    const std::int64_t rs = vec ? y.stride(0) : y.stride(-2);
    const std::int64_t cs = vec ? 0 : y.stride(-1);
    const bool complex = computes_complex(a.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = vec ? alloc_vectors(b.shape(), n, std::is_same_v<T, cplx>)
                      : alloc_matrices(b.shape(), n, k, std::is_same_v<T, cplx>);
        T* res = reinterpret_cast<T*>(out.data);
        const Failure f = run_batch(b, [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* lu;
            lint* ipiv;
            if (const int code = factor<T>(a, off[0], n, s, lu, ipiv)) return code;
            T* x = res + i * n * k;  // b is packed straight into the output slot (ARCHITECTURE.md §3.2)
            if (!pack(y.dtype(), y.data() + off[1], n, k, rs, cs, x, n, true)) return kNonFiniteB;
            if (n > 0 && k > 0) lapack::getrs('N', n, k, lu, n, ipiv, x, n);
            return 0;
        });
        if (f.code == kNonFiniteB) raise(b, {f.index, kNonFinite}, "solve", "b");
        if (f) raise(b, f, "solve", "a");
        return to_sdnp(out);
    });
}

}  // namespace

void bind_lu(nb::module_& m) {
    // Unnamed args are positional-only (API.md §0.1). .none() lets None reach
    // Input, which rejects every non-sdnp argument with the same TypeError.
    m.def("solve", &solve, nb::arg().none(), nb::arg().none());
    m.def("inv", &inv, nb::arg().none());
    m.def("det", &det, nb::arg().none());
    m.def("slogdet", &slogdet, nb::arg().none());
}

}  // namespace sdla
