// The Krylov kernels' shared pieces (docs/design/sparse.md §3.7, §3.8): numpy's scalar arithmetic,
// the deterministic vector operations (sd.linalg.vecdot's sums, sd.linalg.vector_norm's 2-norm,
// element-wise updates), the operators (a CSR matrix through the sparse SpMV, a Python callable,
// the identity) and LAPACK 3.10+'s lartg. The iterative solvers (iterative.cpp) and the eigen
// solvers (arpack.cpp, svds.cpp) use them, so the same operations give the same bits in both.
//
// Every vector operation is split by size alone, so a result is the same bit for bit for any
// thread count (API.md §3).
#pragma once

#include "array/kernels/scalar_ops.hpp"
#include "bind/pyutil.hpp"
#include "linalg/ops/vnorm.hpp"
#include "linalg/pairwise.hpp"
#include "sparse/arith.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sd::sparse::krylov {

namespace nb = nanobind;

using std::int64_t;

constexpr double kEps = DBL_EPSILON;
constexpr double kInf = std::numeric_limits<double>::infinity();

template <class T>
inline constexpr bool kComplex = std::is_same_v<T, complex128>;

// ---- scalars: numpy's arithmetic ----

template <class T>
inline T mul(T a, T b) {
    if constexpr (kComplex<T>) return ops::cmul(a, b);
    else return a * b;
}
template <class T>
inline T dvd(T a, T b) {
    if constexpr (kComplex<T>) return ops::cdiv(a, b);
    else return a / b;
}
template <class T>
inline T add(T a, T b) {
    if constexpr (kComplex<T>) return {a.real() + b.real(), a.imag() + b.imag()};
    else return a + b;
}
template <class T>
inline T sub(T a, T b) {
    if constexpr (kComplex<T>) return {a.real() - b.real(), a.imag() - b.imag()};
    else return a - b;
}
template <class T>
inline T neg(T a) {
    if constexpr (kComplex<T>) return {-a.real(), -a.imag()};
    else return -a;
}
template <class T>
inline T conj(T a) {
    if constexpr (kComplex<T>) return {a.real(), -a.imag()};
    else return a;
}
template <class T>
inline double re(T a) {
    if constexpr (kComplex<T>) return a.real();
    else return a;
}
template <class T>
inline double im(T a) {
    if constexpr (kComplex<T>) return a.imag();
    else return 0.0;
}
template <class T>
inline double mag(T a) {  // numpy's abs (hypot for complex)
    if constexpr (kComplex<T>) return std::hypot(a.real(), a.imag());
    else return std::fabs(a);
}
template <class T>
inline T of(double v) {
    return T(v);
}
template <class T>
inline bool is_zero(T a) {
    return re(a) == 0.0 && im(a) == 0.0;
}
// numpy's order for complex: a < b, a <= b (real parts first)
template <class T>
inline bool lt(T a, double b) {
    return re(a) < b || (re(a) == b && im(a) < 0.0);
}
template <class T>
inline bool le(T a, double b) {
    return re(a) < b || (re(a) == b && im(a) <= 0.0);
}
template <class T>
inline bool ge(T a, double b) {
    return !lt(a, b);
}

// ---- vectors ----

template <class T>
using Vec = std::vector<T>;

// f(i) for every i < n, in fixed chunks over the pool when long (each element on its own).
template <class F>
void each(int64_t n, const F& f) {
    for_chunks(n, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) f(i);
    });
}

// sum_i a_i b_i (real) or sum_i conj(a_i) b_i (complex): sd.linalg.vecdot's sum.
template <class T>
T dot(const T* a, const T* b, int64_t n) {
    if constexpr (kComplex<T>) {
        return linalg::pairwise_tree_mt<complex128>(n, [a, b](int64_t b0, int64_t m) {
            return linalg::detail::fma_cdot_leaf(a + b0, b + b0, m);
        });
    } else {
        return linalg::pairwise_tree_mt<double>(n, [a, b](int64_t b0, int64_t m) {
            return linalg::detail::fma_dot_leaf(a + b0, b + b0, m);
        });
    }
}

// ||x||_2: sd.linalg.vector_norm's (a complex vector as the real vector of its parts).
template <class T>
double nrm2(const T* x, int64_t n) {
    const double* d = reinterpret_cast<const double*>(x);
    const int64_t count = kComplex<T> ? 2 * n : n;
    auto fill = [d](int64_t b, int64_t len, double* out) {
        for (int64_t k = 0; k < len; ++k) out[k] = std::fabs(d[b + k]);
    };
    return linalg::vnorm(linalg::VecOrd{linalg::Ord::Two, 2.0}, count, fill, d);
}

template <class T>
bool any_nonzero(const Vec<T>& x) {
    for (const T& v : x)
        if (!is_zero(v)) return true;
    return false;
}

// y += a x
template <class T>
void axpy(T a, const T* x, T* y, int64_t n) {
    if constexpr (kComplex<T>) each(n, [&](int64_t i) { y[i] = add(y[i], ops::cmul(a, x[i])); });
    else each(n, [&](int64_t i) { y[i] = std::fma(a, x[i], y[i]); });
}

// y = a y + x  (numpy's y *= a; y += x, fused for real)
template <class T>
void aypx(T a, T* y, const T* x, int64_t n) {
    if constexpr (kComplex<T>) each(n, [&](int64_t i) { y[i] = add(ops::cmul(a, y[i]), x[i]); });
    else each(n, [&](int64_t i) { y[i] = std::fma(a, y[i], x[i]); });
}

// x *= a
template <class T>
void scal(T a, T* x, int64_t n) {
    each(n, [&](int64_t i) { x[i] = mul(a, x[i]); });
}

// ---- operators ----

template <class T>
struct Op {
    enum Kind { Identity, Matrix, Python } kind = Identity;
    int64_t rows = 0, cols = 0;
    const int64_t* ip = nullptr;
    const int64_t* ix = nullptr;
    const T* v = nullptr;
    std::vector<int64_t> bounds;
    nb::object fn;

    bool native() const { return kind != Python; }
    int64_t work() const { return kind == Matrix ? ip[rows] + rows : rows; }

    // y = op(x); x has cols entries, y rows.
    void apply(const T* x, T* y) const {
        if (kind == Identity) {
            std::memcpy(y, x, static_cast<std::size_t>(rows) * sizeof(T));
            return;
        }
        if (kind == Matrix) {
            const int64_t nt = static_cast<int64_t>(bounds.size()) - 1;
            for_tasks(nt, work(), [&](int64_t t) {
                std::vector<T> parts;
                for (int64_t r = bounds[t]; r < bounds[t + 1]; ++r) {
                    auto step = [this, x](int64_t p, T acc) { return madd(v[p], x[ix[p]], acc); };
                    y[r] = ordered_sum<T>(ip[r], ip[r + 1], step, parts);
                }
            });
            return;
        }
        nb::gil_scoped_acquire gil;
        const int64_t shape[] = {cols};
        NDArray in = new_array(dtype_of<T>, shape);
        std::memcpy(in.data, x, static_cast<std::size_t>(cols) * sizeof(T));
        nb::object out = fn(wrap(std::move(in)));
        if (!is_ndarray(out)) throw nb::type_error("the operator's function did not return an sd.ndarray");
        const NDArray& o = as_ndarray(out);
        if (o.dtype != dtype_of<T> || o.ndim != 1 || o.shape[0] != rows || !o.c_contiguous)
            throw nb::value_error("the operator's function returned an array of the wrong shape or dtype");
        std::memcpy(y, o.data, static_cast<std::size_t>(rows) * sizeof(T));
    }
};

template <class T>
Op<T> parse_op(nb::handle o, int64_t rows, int64_t cols, const char* func) {
    Op<T> op;
    op.rows = rows;
    op.cols = cols;
    if (o.is_none()) return op;
    if (nb::isinstance<nb::tuple>(o)) {
        nb::tuple t = nb::borrow<nb::tuple>(o);
        if (t.size() != 3) throw nb::value_error((std::string(func) + "(): an operator tuple has 3 arrays").c_str());
        const NDArray& indptr = vector_arg(t[0], func, DType::Int64);
        const NDArray& indices = vector_arg(t[1], func, DType::Int64);
        const NDArray& data = vector_arg(t[2], func, dtype_of<T>);
        if (indptr.shape[0] != rows + 1 || indices.shape[0] != data.shape[0])
            throw nb::value_error((std::string(func) + "(): the operator's arrays do not match").c_str());
        op.kind = Op<T>::Matrix;
        op.ip = i64(indptr);
        op.ix = i64(indices);
        op.v = typed<T>(data);
        if (op.ip[rows] != data.shape[0])
            throw nb::value_error((std::string(func) + "(): the operator's arrays do not match").c_str());
        op.bounds = line_bounds(op.ip, rows, kChunk);
        return op;
    }
    if (!PyCallable_Check(o.ptr()))
        throw nb::type_error((std::string(func) + "(): an operator must be callable").c_str());
    op.kind = Op<T>::Python;
    op.fn = nb::borrow<nb::object>(o);
    return op;
}

template <class T>
const T* vec_arg(nb::handle o, int64_t n, const char* func, const char* what) {
    const NDArray& a = vector_arg(o, func, dtype_of<T>);
    if (a.shape[0] != n) throw nb::value_error((std::string(func) + "(): '" + what + "' has the wrong length").c_str());
    return typed<T>(a);
}

// ---- rotations ----

// LAPACK 3.10+'s lartg (Anderson 2017, la_xlartg.f90; Accelerate's LAPACK 3.12 has the same): c, s,
// r with [c s; -conj(s) c] [f; g] = [r; 0], c real. scipy calls it through get_lapack_funcs('lartg').
struct Rot {
    double c;
    complex128 s, r;
};

inline Rot lartg(double f, double g) {
    const double safmin = 0x1p-1022, safmax = 0x1p1022;
    const double rtmin = std::sqrt(safmin), rtmax = std::sqrt(safmax / 2);
    const double f1 = std::fabs(f), g1 = std::fabs(g);
    if (g == 0) return {1.0, 0.0, f};
    if (f == 0) return {0.0, std::copysign(1.0, g), g1};
    if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
        const double d = std::sqrt(f * f + g * g);
        const double c = f1 / d, r = std::copysign(d, f);
        return {c, g / r, r};
    }
    const double u = std::min(safmax, std::max(safmin, std::max(f1, g1)));
    const double fs = f / u, gs = g / u;
    const double d = std::sqrt(fs * fs + gs * gs);
    const double c = std::fabs(fs) / d, r = std::copysign(d, f);
    return {c, gs / r, r * u};
}

inline double abssq(complex128 z) { return z.real() * z.real() + z.imag() * z.imag(); }
inline complex128 cscale(complex128 z, double a) { return {z.real() * a, z.imag() * a}; }
inline complex128 cdivr(complex128 z, double a) { return {z.real() / a, z.imag() / a}; }

inline Rot lartg(complex128 f, complex128 g) {
    const double safmin = 0x1p-1022, safmax = 0x1p1022;
    const double rtmin = std::sqrt(safmin);
    const auto fabsmax = [](complex128 z) { return std::max(std::fabs(z.real()), std::fabs(z.imag())); };
    if (g.real() == 0 && g.imag() == 0) return {1.0, {0.0, 0.0}, f};
    if (f.real() == 0 && f.imag() == 0) {
        if (g.real() == 0) {
            const double r = std::fabs(g.imag());
            return {0.0, cdivr(std::conj(g), r), r};
        }
        if (g.imag() == 0) {
            const double r = std::fabs(g.real());
            return {0.0, cdivr(std::conj(g), r), r};
        }
        const double g1 = fabsmax(g);
        const double rtmax = std::sqrt(safmax / 2);
        if (g1 > rtmin && g1 < rtmax) {
            const double d = std::sqrt(abssq(g));
            return {0.0, cdivr(std::conj(g), d), d};
        }
        const double u = std::min(safmax, std::max(safmin, g1));
        const complex128 gs = cdivr(g, u);
        const double d = std::sqrt(abssq(gs));
        return {0.0, cdivr(std::conj(gs), d), d * u};
    }
    const double f1 = fabsmax(f), g1 = fabsmax(g);
    double rtmax = std::sqrt(safmax / 4);
    Rot out;
    if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
        const double f2 = abssq(f), g2 = abssq(g), h2 = f2 + g2;
        if (f2 >= h2 * safmin) {
            out.c = std::sqrt(f2 / h2);
            out.r = cdivr(f, out.c);
            rtmax *= 2;
            if (f2 > rtmin && h2 < rtmax) out.s = ops::cmul(std::conj(g), cdivr(f, std::sqrt(f2 * h2)));
            else out.s = ops::cmul(std::conj(g), cdivr(out.r, h2));
        } else {
            const double d = std::sqrt(f2 * h2);
            out.c = f2 / d;
            out.r = out.c >= safmin ? cdivr(f, out.c) : cscale(f, h2 / d);
            out.s = ops::cmul(std::conj(g), cdivr(f, d));
        }
        return out;
    }
    const double u = std::min(safmax, std::max(safmin, std::max(f1, g1)));
    const complex128 gs = cdivr(g, u);
    const double g2 = abssq(gs);
    double w, f2, h2;
    complex128 fs;
    if (f1 / u < rtmin) {
        const double v = std::min(safmax, std::max(safmin, f1));
        w = v / u;
        fs = cdivr(f, v);
        f2 = abssq(fs);
        h2 = f2 * (w * w) + g2;
    } else {
        w = 1.0;
        fs = cdivr(f, u);
        f2 = abssq(fs);
        h2 = f2 + g2;
    }
    if (f2 >= h2 * safmin) {
        out.c = std::sqrt(f2 / h2);
        out.r = cdivr(fs, out.c);
        rtmax *= 2;
        if (f2 > rtmin && h2 < rtmax) out.s = ops::cmul(std::conj(gs), cdivr(fs, std::sqrt(f2 * h2)));
        else out.s = ops::cmul(std::conj(gs), cdivr(out.r, h2));
    } else {
        const double d = std::sqrt(f2 * h2);
        out.c = f2 / d;
        out.r = out.c >= safmin ? cdivr(fs, out.c) : cscale(fs, h2 / d);
        out.s = ops::cmul(std::conj(gs), cdivr(fs, d));
    }
    out.c *= w;
    out.r = cscale(out.r, u);
    return out;
}

}  // namespace sd::sparse::krylov
