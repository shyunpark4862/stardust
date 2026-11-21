// The iterative solvers of sd.sparse (docs/design/sparse.md §3.7): cg, bicgstab, gmres, minres,
// lsqr and lsmr, scipy 1.18.1's algorithms and stopping tests in its order of operations (scipy's
// Python of scipy/sparse/linalg/_isolve: iterative.py, minres.py, lsqr.py, lsmr.py).
//
//   _cg(A, M, b, x0, /, *, rtol, atol, max_iter)
//   _bicgstab(A, M, b, x0, /, *, rtol, atol, max_iter)
//   _gmres(A, M, b, x0, /, *, rtol, atol, restart, max_iter)
//   _minres(A, M, b, x0, /, *, rtol, shift, max_iter)
//   _lsqr(A, AH, b, x0, /, *, damp, atol, btol, conlim, max_iter)
//   _lsmr(A, AH, b, x0, /, *, damp, atol, btol, conlim, max_iter)
//
// An operator (A, its adjoint AH, the preconditioner M) is one of
//   - a tuple (indptr, indices, data): canonical CSR arrays of the matrix itself (a dense matrix
//     comes as the CSR of all its entries), data of the system's dtype; y = A x is the sparse
//     area's SpMV (every output entry one sum in ascending column order, fma per term, pieces of
//     kLine past kLine terms: sparse/arith.hpp), so the bits are those of `A @ x`;
//   - a Python callable f(x) -> y on sd.ndarrays (a LinearOperator's product, checked by the
//     Python layer: y is a new C-contiguous array of the system's dtype and the right length);
//   - None (M only): the identity.
// b is a C-contiguous 1-D array of the system's dtype (float64 or complex128), x0 one or None.
// The result is a dict: x, status, n_iter and the method's estimates (below), and `history`, the
// residual norm the method tracks after each step (a float64 array; tests compare it with scipy).
//
// The vector operations are this library's (docs/design/sparse.md §3.7): inner products are
// sd.linalg.vecdot's sums (pairwise trees of fused leaves, conjugating the first vector for
// complex; numpy's dot and vdot), norms sd.linalg.vector_norm's (the fused sum of squares, its
// rescaled pass on overflow; a complex vector's real and imaginary parts as one real vector),
// updates element by element (fma for a real axpy, ops::cmul then a sum for complex, as the array
// area's complex product). Every one is split by size alone, so the result is the same bit for
// bit for any thread count (API.md §3). Scalars follow numpy's arithmetic: a complex quantity
// stays complex (ops::cmul, ops::cdiv), comparisons of complex values are numpy's order (the real
// parts, then the imaginary ones).
//
// The GIL: with native operators the whole solve runs without it once the work per step (the
// stored entries and the length) reaches the area's threshold; a Python operator keeps the GIL
// for the solve (its calls need it), and the kernels release it as every sparse kernel does.
#include "array/kernels/scalar_ops.hpp"
#include "bind/pyutil.hpp"
#include "linalg/errors.hpp"
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

namespace nb = nanobind;

namespace sd::sparse {

namespace {

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

// ---- results ----

template <class T>
struct Out {
    Vec<T> x;
    int64_t status = 0, n_iter = 0, n_inner = 0, stop = 0;
    double normr = 0, normr2 = 0, norma = 0, conda = 0, normar = 0, normx = 0;
    Vec<double> var;
    Vec<double> history;
};

template <class T>
nb::dict to_dict(Out<T>& o) {
    nb::dict d;
    {
        const int64_t shape[] = {static_cast<int64_t>(o.x.size())};
        NDArray x = new_array(dtype_of<T>, shape);
        if (!o.x.empty()) std::memcpy(x.data, o.x.data(), o.x.size() * sizeof(T));
        d["x"] = wrap(std::move(x));
    }
    auto doubles = [](const Vec<double>& v) {
        const int64_t shape[] = {static_cast<int64_t>(v.size())};
        NDArray a = new_array(DType::Float64, shape);
        if (!v.empty()) std::memcpy(a.data, v.data(), v.size() * sizeof(double));
        return wrap(std::move(a));
    };
    d["status"] = o.status;
    d["n_iter"] = o.n_iter;
    d["n_inner"] = o.n_inner;
    d["stop"] = o.stop;
    d["normr"] = o.normr;
    d["normr2"] = o.normr2;
    d["norma"] = o.norma;
    d["conda"] = o.conda;
    d["normar"] = o.normar;
    d["normx"] = o.normx;
    d["var"] = doubles(o.var);
    d["history"] = doubles(o.history);
    return d;
}

// The two operators' GIL: without it for the whole solve when both are native and a step is
// large; with it (the kernels release it on their own) otherwise.
template <class T>
bool solve_unlocked(const Op<T>& a, const Op<T>& b, int64_t n) {
    return a.native() && b.native() && a.work() + b.work() + n >= kParallelThreshold;
}

// x0 (or zeros) as the start.
template <class T>
Vec<T> start(const T* x0, int64_t n) {
    Vec<T> x(static_cast<std::size_t>(n), T{});
    if (x0) std::memcpy(x.data(), x0, static_cast<std::size_t>(n) * sizeof(T));
    return x;
}

// r = b - A x, or b when x is all zero (scipy: `b - matvec(x) if x.any() else b.copy()`).
template <class T>
void residual(const Op<T>& A, const T* b, const Vec<T>& x, Vec<T>& r, Vec<T>& tmp, bool check_zero = true) {
    const int64_t n = static_cast<int64_t>(r.size());
    if (check_zero && !any_nonzero(x)) {
        std::memcpy(r.data(), b, static_cast<std::size_t>(n) * sizeof(T));
        return;
    }
    A.apply(x.data(), tmp.data());
    each(n, [&](int64_t i) { r[i] = sub(b[i], tmp[i]); });
}

// ---- cg (scipy's iterative.py) ----

template <class T>
void cg(
    const Op<T>& A,
    const Op<T>& M,
    const T* b,
    const T* x0,
    int64_t n,
    double rtol,
    double atol_in,
    int64_t maxiter,
    Out<T>& o
) {
    const double bnrm2 = nrm2(b, n);
    const double atol = std::max(atol_in, rtol * bnrm2);
    if (bnrm2 == 0) {
        o.x.assign(b, b + n);
        o.history.push_back(0.0);
        return;
    }
    Vec<T> x = start(x0, n), r(n), z(n), p(n), q(n);
    residual(A, b, x, r, q);
    T rho_prev{};
    for (int64_t it = 0; it < maxiter; ++it) {
        const double rn = nrm2(r.data(), n);
        o.history.push_back(rn);
        if (rn < atol) {
            o.x = std::move(x);
            o.normr = rn;
            o.n_iter = it;
            return;
        }
        const T* zp = r.data();
        if (M.kind != Op<T>::Identity) {
            M.apply(r.data(), z.data());
            zp = z.data();
        }
        const T rho = dot(r.data(), zp, n);
        if (it > 0) aypx(dvd(rho, rho_prev), p.data(), zp, n);
        else std::memcpy(p.data(), zp, static_cast<std::size_t>(n) * sizeof(T));
        A.apply(p.data(), q.data());
        const T alpha = dvd(rho, dot(p.data(), q.data(), n));
        axpy(alpha, p.data(), x.data(), n);
        axpy(neg(alpha), q.data(), r.data(), n);
        rho_prev = rho;
    }
    o.normr = nrm2(r.data(), n);
    o.history.push_back(o.normr);
    o.x = std::move(x);
    o.status = maxiter;
    o.n_iter = maxiter;
}

// ---- bicgstab ----

template <class T>
void bicgstab(
    const Op<T>& A,
    const Op<T>& M,
    const T* b,
    const T* x0,
    int64_t n,
    double rtol,
    double atol_in,
    int64_t maxiter,
    Out<T>& o
) {
    const double bnrm2 = nrm2(b, n);
    const double atol = std::max(atol_in, rtol * bnrm2);
    if (bnrm2 == 0) {
        o.x.assign(b, b + n);
        o.history.push_back(0.0);
        return;
    }
    // "These values make no sense but coming from original Fortran code" (scipy)
    const double rhotol = kEps * kEps, omegatol = rhotol;
    Vec<T> x = start(x0, n), r(n), p(n), v(n), s(n), t(n), phat(n), shat(n);
    residual(A, b, x, r, v);
    const Vec<T> rtilde = r;
    T rho_prev{}, omega{}, alpha{};
    auto finish = [&](int64_t status, int64_t it, double rn) {
        o.x = std::move(x);
        o.status = status;
        o.n_iter = it;
        o.normr = rn;
    };
    double rn = 0;
    for (int64_t it = 0; it < maxiter; ++it) {
        rn = nrm2(r.data(), n);
        o.history.push_back(rn);
        if (rn < atol) return finish(0, it, rn);
        const T rho = dot(rtilde.data(), r.data(), n);
        if (mag(rho) < rhotol) return finish(-10, it, rn);
        if (it > 0) {
            if (mag(omega) < omegatol) return finish(-11, it, rn);
            const T beta = mul(dvd(rho, rho_prev), dvd(alpha, omega));
            axpy(neg(omega), v.data(), p.data(), n);
            aypx(beta, p.data(), r.data(), n);
        } else {
            p = r;
        }
        const T* ph = p.data();
        if (M.kind != Op<T>::Identity) {
            M.apply(p.data(), phat.data());
            ph = phat.data();
        }
        A.apply(ph, v.data());
        const T rv = dot(rtilde.data(), v.data(), n);
        if (is_zero(rv)) return finish(-11, it, rn);
        alpha = dvd(rho, rv);
        axpy(neg(alpha), v.data(), r.data(), n);
        s = r;
        const double sn = nrm2(s.data(), n);
        if (sn < atol) {
            o.history.push_back(sn);
            axpy(alpha, ph, x.data(), n);
            return finish(0, it + 1, sn);
        }
        const T* sh = s.data();
        if (M.kind != Op<T>::Identity) {
            M.apply(s.data(), shat.data());
            sh = shat.data();
        }
        A.apply(sh, t.data());
        omega = dvd(dot(t.data(), s.data(), n), dot(t.data(), t.data(), n));
        axpy(alpha, ph, x.data(), n);
        axpy(omega, sh, x.data(), n);
        axpy(neg(omega), t.data(), r.data(), n);
        rho_prev = rho;
    }
    rn = nrm2(r.data(), n);
    o.history.push_back(rn);
    finish(maxiter, maxiter, rn);
}

// ---- gmres ----

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

template <class T>
void gmres(
    const Op<T>& A,
    const Op<T>& M,
    const T* b,
    const T* x0,
    int64_t n,
    double rtol,
    double atol_in,
    int64_t restart,
    int64_t maxiter,
    Out<T>& o
) {
    const double bnrm2 = nrm2(b, n);
    const double atol = std::max(atol_in, rtol * bnrm2);
    if (bnrm2 == 0) {
        o.x.assign(b, b + n);
        return;
    }
    Vec<T> x = start(x0, n), r(n), w(n), tmp(n), xs(n);
    const std::size_t un = static_cast<std::size_t>(n), ur = static_cast<std::size_t>(restart);
    double Mb_nrm2;
    if (M.kind == Op<T>::Identity) {
        Mb_nrm2 = bnrm2;
    } else {
        M.apply(b, tmp.data());
        Mb_nrm2 = nrm2(tmp.data(), n);
    }
    double ptol_max_factor = 1.0;
    double ptol = Mb_nrm2 * std::min(ptol_max_factor, atol / bnrm2);
    double presid = 0.0, rnorm = 0.0;
    Vec<T> v((ur + 1) * un);                 // the Krylov basis, row k = v[k * n ...]
    Vec<T> h(ur * (ur + 1), T{});            // h[col][k] = h[col * (restart + 1) + k]
    Vec<double> gc(ur, 0.0);                 // the rotations' c
    Vec<T> gs(ur, T{});                      // and s
    Vec<T> S(ur + 1), y(ur);
    auto H = [&](int64_t col, int64_t k) -> T& { return h[static_cast<std::size_t>(col * (restart + 1) + k)]; };
    auto V = [&](int64_t k) { return v.data() + static_cast<std::size_t>(k) * un; };
    int64_t iteration = 0;
    for (; iteration < maxiter; ++iteration) {
        if (iteration == 0) {
            residual(A, b, x, r, tmp);
            rnorm = nrm2(r.data(), n);
            if (rnorm < atol) {
                o.x = std::move(x);
                o.normr = rnorm;
                return;
            }
        }
        M.apply(r.data(), V(0));
        const double t0 = nrm2(V(0), n);
        scal(of<T>(1 / t0), V(0), n);
        std::fill(S.begin(), S.end(), T{});
        S[0] = of<T>(t0);
        bool breakdown = false;
        int64_t col = 0;
        for (col = 0; col < restart; ++col) {
            A.apply(V(col), tmp.data());
            M.apply(tmp.data(), w.data());
            // modified Gram-Schmidt
            const double h0 = nrm2(w.data(), n);
            for (int64_t k = 0; k <= col; ++k) {
                const T t = dot(V(k), w.data(), n);
                H(col, k) = t;
                axpy(neg(t), V(k), w.data(), n);
            }
            const double h1 = nrm2(w.data(), n);
            H(col, col + 1) = of<T>(h1);
            std::memcpy(V(col + 1), w.data(), un * sizeof(T));
            if (h1 <= kEps * h0) {  // the exact solution indicator
                H(col, col + 1) = T{};
                breakdown = true;
            } else {
                scal(of<T>(1 / h1), V(col + 1), n);
            }
            // the past rotations on the new column of h
            for (int64_t k = 0; k < col; ++k) {
                const T c = of<T>(gc[k]), s = gs[k];
                const T n0 = H(col, k), n1 = H(col, k + 1);
                H(col, k) = add(mul(c, n0), mul(s, n1));
                H(col, k + 1) = add(mul(neg(conj(s)), n0), mul(c, n1));
            }
            // the new rotation, on h and S
            const Rot g = lartg(H(col, col), H(col, col + 1));
            T gsv, gr;
            if constexpr (kComplex<T>) {
                gsv = g.s;
                gr = g.r;
            } else {
                gsv = g.s.real();
                gr = g.r.real();
            }
            gc[col] = g.c;
            gs[col] = gsv;
            H(col, col) = gr;
            H(col, col + 1) = T{};
            const T t = mul(neg(conj(gsv)), S[col]);
            S[col] = mul(of<T>(g.c), S[col]);
            S[col + 1] = t;
            presid = mag(t);
            ++o.n_inner;
            o.history.push_back(presid);
            if (presid <= ptol || breakdown) break;
        }
        if (col == restart) col = restart - 1;
        // the upper triangular h(col, col) system, singular cases pseudo-solved (scipy)
        if (is_zero(H(col, col))) S[col] = T{};
        for (int64_t k = 0; k <= col; ++k) y[k] = S[k];
        for (int64_t k = col; k > 0; --k) {
            if (!is_zero(y[k])) {
                y[k] = dvd(y[k], H(k, k));
                const T t = y[k];
                for (int64_t j = 0; j < k; ++j) y[j] = sub(y[j], mul(t, H(k, j)));
            }
        }
        if (!is_zero(y[0])) y[0] = dvd(y[0], H(0, 0));
        // x += y @ v[:col + 1]
        each(n, [&](int64_t i) {
            T acc{};
            for (int64_t k = 0; k <= col; ++k) acc = madd(y[k], V(k)[i], acc);
            x[i] = add(x[i], acc);
        });
        residual(A, b, x, r, tmp, false);
        rnorm = nrm2(r.data(), n);
        if (rnorm <= atol) break;
        if (breakdown) {  // the exact solution, but the outer test failed
            o.stop = 1;
            break;
        }
        if (presid <= ptol) ptol_max_factor = std::max(kEps, 0.25 * ptol_max_factor);
        else ptol_max_factor = std::min(1.0, 1.5 * ptol_max_factor);
        ptol = presid * std::min(ptol_max_factor, atol / rnorm);
    }
    o.n_iter = std::min(iteration + 1, maxiter);
    o.status = rnorm <= atol ? 0 : maxiter;
    o.normr = rnorm;
    o.x = std::move(x);
}

// ---- minres (scipy's minres.py) ----

template <class T>
double vec2norm(T a, T b) {  // numpy's norm([a, b])
    if constexpr (kComplex<T>) return std::sqrt(abssq(a) + abssq(b));
    else return std::sqrt(a * a + b * b);
}

template <class T>
void minres(
    const Op<T>& A,
    const Op<T>& M,
    const T* b,
    const T* x0,
    int64_t n,
    double rtol,
    double shift,
    int64_t maxiter,
    Out<T>& o
) {
    int64_t istop = 0, itn = 0;
    double Anorm = 0, Acond = 0, ynorm = 0;
    T rnorm{}, Arnorm{};
    Vec<T> x = start(x0, n), r1(n), y(n), tmp(n), v(n), w(n), w1(n), w2(n), r2(n);
    if (x0) residual(A, b, x, r1, tmp, false);
    else std::memcpy(r1.data(), b, static_cast<std::size_t>(n) * sizeof(T));
    M.apply(r1.data(), y.data());
    const T beta1t = dot(r1.data(), y.data(), n);
    if (lt(beta1t, 0.0)) throw linalg::LinAlgError("minres(): the preconditioner is not positive definite");
    if (is_zero(beta1t)) {
        o.x = std::move(x);
        o.stop = 0;
        return;
    }
    const double bnorm = nrm2(b, n);
    if (bnorm == 0) {
        o.x.assign(b, b + n);
        o.stop = 0;
        return;
    }
    const double beta1 = std::sqrt(re(beta1t));
    double oldb = 0, beta = beta1;
    T dbar{}, epsln{}, phibar = of<T>(beta1), rhs1 = of<T>(beta1), rhs2{}, tnorm2{};
    double gmax = 0, gmin = std::numeric_limits<double>::max();
    T cs = of<T>(-1.0), sn{};
    r2 = r1;
    o.history.push_back(beta1);
    while (itn < maxiter) {
        ++itn;
        const double s = 1.0 / beta;
        each(n, [&](int64_t i) { v[i] = mul(of<T>(s), y[i]); });
        A.apply(v.data(), y.data());
        if (shift != 0) axpy(of<T>(-shift), v.data(), y.data(), n);
        if (itn >= 2) axpy(of<T>(-(beta / oldb)), r1.data(), y.data(), n);
        const T alfa = dot(v.data(), y.data(), n);
        axpy(neg(dvd(alfa, of<T>(beta))), r2.data(), y.data(), n);
        std::swap(r1, r2);  // r1 = r2
        r2 = y;             // r2 = y
        M.apply(r2.data(), y.data());
        oldb = beta;
        const T betat = dot(r2.data(), y.data(), n);
        if (lt(betat, 0.0)) throw linalg::LinAlgError("minres(): the matrix is not symmetric");
        beta = std::sqrt(re(betat));
        tnorm2 = add(tnorm2, add(add(mul(alfa, alfa), of<T>(oldb * oldb)), of<T>(beta * beta)));
        if (itn == 1 && beta / beta1 <= 10 * kEps) istop = -1;  // terminate later
        // the previous rotation Q_{k-1}
        const T oldeps = epsln;
        const T delta = add(mul(cs, dbar), mul(sn, alfa));
        const T gbar = sub(mul(sn, dbar), mul(cs, alfa));
        epsln = mul(sn, of<T>(beta));
        dbar = mul(neg(cs), of<T>(beta));
        const double root = vec2norm(gbar, dbar);
        Arnorm = mul(phibar, of<T>(root));
        // the next rotation Q_k (gamma: a real value, numpy's norm cast to the dtype)
        double gamma = vec2norm(gbar, of<T>(beta));
        if (gamma < kEps) gamma = kEps;
        cs = dvd(gbar, of<T>(gamma));
        sn = dvd(of<T>(beta), of<T>(gamma));
        const T phi = mul(cs, phibar);
        phibar = mul(sn, phibar);
        // update x
        const T denom = dvd(of<T>(1.0), of<T>(gamma));
        std::swap(w1, w2);  // w1 = w2
        std::swap(w2, w);   // w2 = w (w now holds the old w1's storage, overwritten next)
        each(n, [&](int64_t i) {
            const T t = sub(sub(v[i], mul(oldeps, w1[i])), mul(delta, w2[i]));
            w[i] = mul(t, denom);
        });
        axpy(phi, w.data(), x.data(), n);
        gmax = std::max(gmax, gamma);
        gmin = std::min(gmin, gamma);
        const T z = dvd(rhs1, of<T>(gamma));
        rhs1 = sub(rhs2, mul(delta, z));
        rhs2 = mul(neg(epsln), z);
        // the norms and the stopping tests
        Anorm = std::sqrt(re(tnorm2));
        ynorm = nrm2(x.data(), n);
        const double epsx = Anorm * ynorm * kEps;
        rnorm = phibar;
        T test1, test2;
        if (ynorm == 0 || Anorm == 0) test1 = of<T>(kInf);
        else test1 = dvd(rnorm, of<T>(Anorm * ynorm));
        if (Anorm == 0) test2 = of<T>(kInf);
        else test2 = of<T>(root / Anorm);
        Acond = gmax / gmin;
        if (istop == 0) {
            const T t1 = add(of<T>(1.0), test1), t2 = add(of<T>(1.0), test2);
            if (le(t2, 1.0)) istop = 2;
            if (le(t1, 1.0)) istop = 1;
            if (itn >= maxiter) istop = 6;
            if (Acond >= 0.1 / kEps) istop = 4;
            if (epsx >= beta1) istop = 3;
            if (le(test2, rtol)) istop = 2;
            if (le(test1, rtol)) istop = 1;
        }
        o.history.push_back(mag(rnorm));
        if (istop != 0) break;
    }
    o.x = std::move(x);
    o.stop = istop;
    o.status = istop == 6 ? maxiter : 0;
    o.n_iter = itn;
    o.normr = mag(rnorm);
    o.norma = Anorm;
    o.conda = Acond;
    o.normar = mag(Arnorm);
    o.normx = ynorm;
}

// ---- lsqr, lsmr: the stable Givens rotation (scipy's _sym_ortho, S.-C. Choi) ----

struct Sym {
    double c, s, r;
};

inline double sign(double a) { return a > 0 ? 1.0 : (a < 0 ? -1.0 : a); }  // np.sign (NaN stays)

inline Sym sym_ortho(double a, double b) {
    if (b == 0) return {sign(a), 0.0, std::fabs(a)};
    if (a == 0) return {0.0, sign(b), std::fabs(b)};
    if (std::fabs(b) > std::fabs(a)) {
        const double tau = a / b;
        const double s = sign(b) / std::sqrt(1 + tau * tau);
        const double c = s * tau;
        return {c, s, b / s};
    }
    const double tau = b / a;
    const double c = sign(a) / std::sqrt(1 + tau * tau);
    const double s = c * tau;
    return {c, s, a / c};
}

// y = A x - a y (lsqr: `u = A.matvec(v) - alfa * u`; lsmr: `u *= -alpha; u += A.matvec(v)`)
template <class T>
void bidiag_step(const Op<T>& A, const T* x, T* y, double a, T* tmp, int64_t m) {
    A.apply(x, tmp);
    if constexpr (kComplex<T>)
        each(m, [&](int64_t i) { y[i] = add(tmp[i], complex128{y[i].real() * -a, y[i].imag() * -a}); });
    else each(m, [&](int64_t i) { y[i] = std::fma(-a, y[i], tmp[i]); });
}

template <class T>
void scale_real(T* x, double a, int64_t n) {
    if constexpr (kComplex<T>) each(n, [&](int64_t i) { x[i] = {x[i].real() * a, x[i].imag() * a}; });
    else each(n, [&](int64_t i) { x[i] = x[i] * a; });
}

// y += a x, a real
template <class T>
void axpy_real(double a, const T* x, T* y, int64_t n) {
    if constexpr (kComplex<T>)
        each(n, [&](int64_t i) {
            y[i] = {std::fma(a, x[i].real(), y[i].real()), std::fma(a, x[i].imag(), y[i].imag())};
        });
    else each(n, [&](int64_t i) { y[i] = std::fma(a, x[i], y[i]); });
}

// y = x + a y, a real
template <class T>
void aypx_real(double a, T* y, const T* x, int64_t n) {
    if constexpr (kComplex<T>)
        each(n, [&](int64_t i) {
            y[i] = {std::fma(a, y[i].real(), x[i].real()), std::fma(a, y[i].imag(), x[i].imag())};
        });
    else each(n, [&](int64_t i) { y[i] = std::fma(a, y[i], x[i]); });
}

template <class T>
void lsqr(
    const Op<T>& A,
    const Op<T>& AH,
    const T* b,
    const T* x0,
    int64_t m,
    int64_t n,
    double damp,
    double atol,
    double btol,
    double conlim,
    int64_t iter_lim,
    Out<T>& o
) {
    int64_t itn = 0, istop = 0;
    const double ctol = conlim > 0 ? 1 / conlim : 0.0;
    double anorm = 0, acond = 0, ddnorm = 0, res2 = 0, xnorm = 0, xxnorm = 0, z = 0, cs2 = -1, sn2 = 0;
    const double dampsq = damp * damp;
    Vec<T> x = start(x0, n), u(m), v(n), w(n), dk(n), tmpm(m), tmpn(n);
    o.var.assign(static_cast<std::size_t>(n), 0.0);
    std::memcpy(u.data(), b, static_cast<std::size_t>(m) * sizeof(T));
    const double bnorm = nrm2(b, m);
    double beta, alfa;
    if (!x0) {
        beta = bnorm;
    } else {
        A.apply(x.data(), tmpm.data());
        each(m, [&](int64_t i) { u[i] = sub(u[i], tmpm[i]); });
        beta = nrm2(u.data(), m);
    }
    if (beta > 0) {
        scale_real(u.data(), 1 / beta, m);
        AH.apply(u.data(), v.data());
        alfa = nrm2(v.data(), n);
    } else {
        v = x;
        alfa = 0;
    }
    if (alfa > 0) scale_real(v.data(), 1 / alfa, n);
    w = v;
    double rhobar = alfa, phibar = beta, rnorm = beta, r1norm = rnorm, r2norm = rnorm;
    double arnorm = alfa * beta;
    o.history.push_back(r1norm);
    auto finish = [&]() {
        o.x = std::move(x);
        o.stop = istop;
        o.status = istop;
        o.n_iter = itn;
        o.normr = r1norm;
        o.normr2 = r2norm;
        o.norma = anorm;
        o.conda = acond;
        o.normar = arnorm;
        o.normx = xnorm;
    };
    if (arnorm == 0) return finish();
    while (itn < iter_lim) {
        ++itn;
        // the next step of the bidiagonalization: beta u = A v - alfa u, alfa v = A^H u - beta v
        bidiag_step(A, v.data(), u.data(), alfa, tmpm.data(), m);
        beta = nrm2(u.data(), m);
        if (beta > 0) {
            scale_real(u.data(), 1 / beta, m);
            anorm = std::sqrt(anorm * anorm + alfa * alfa + beta * beta + dampsq);
            bidiag_step(AH, u.data(), v.data(), beta, tmpn.data(), n);
            alfa = nrm2(v.data(), n);
            if (alfa > 0) scale_real(v.data(), 1 / alfa, n);
        }
        // a rotation eliminates the damping parameter
        double rhobar1, psi;
        if (damp > 0) {
            rhobar1 = std::sqrt(rhobar * rhobar + dampsq);
            const double cs1 = rhobar / rhobar1, sn1 = damp / rhobar1;
            psi = sn1 * phibar;
            phibar = cs1 * phibar;
        } else {
            rhobar1 = rhobar;
            psi = 0.0;
        }
        // a rotation eliminates the subdiagonal element beta
        const Sym g = sym_ortho(rhobar1, beta);
        const double cs = g.c, sn = g.s, rho = g.r;
        const double theta = sn * alfa;
        rhobar = -cs * alfa;
        const double phi = cs * phibar;
        phibar = sn * phibar;
        const double tau = sn * phi;
        // update x and w
        const double t1 = phi / rho, t2 = -theta / rho;
        const double rr = 1 / rho;
        each(n, [&](int64_t i) { dk[i] = mul(of<T>(rr), w[i]); });
        axpy_real(t1, w.data(), x.data(), n);
        aypx_real(t2, w.data(), v.data(), n);
        const double dkn = nrm2(dk.data(), n);
        ddnorm = ddnorm + dkn * dkn;
        each(n, [&](int64_t i) {
            const double a = re(dk[i]), c = im(dk[i]);
            o.var[static_cast<std::size_t>(i)] += kComplex<T> ? a * a + c * c : a * a;
        });
        // a rotation on the right eliminates theta; then the estimate of norm(x)
        const double delta = sn2 * rho, gambar = -cs2 * rho;
        const double rhs = phi - delta * z;
        const double zbar = rhs / gambar;
        xnorm = std::sqrt(xxnorm + zbar * zbar);
        const double gamma = std::sqrt(gambar * gambar + theta * theta);
        cs2 = gambar / gamma;
        sn2 = theta / gamma;
        z = rhs / gamma;
        xxnorm = xxnorm + z * z;
        // the stopping tests
        acond = anorm * std::sqrt(ddnorm);
        const double res1 = phibar * phibar;
        res2 = res2 + psi * psi;
        rnorm = std::sqrt(res1 + res2);
        arnorm = alfa * std::fabs(tau);
        if (damp > 0) {
            const double r1sq = rnorm * rnorm - dampsq * xxnorm;
            r1norm = std::sqrt(std::fabs(r1sq));
            if (r1sq < 0) r1norm = -r1norm;
        } else {
            r1norm = rnorm;
        }
        r2norm = rnorm;
        const double test1 = rnorm / bnorm;
        const double test2 = arnorm / (anorm * rnorm + kEps);
        const double test3 = 1 / (acond + kEps);
        const double t1s = test1 / (1 + anorm * xnorm / bnorm);
        const double rtol = btol + atol * anorm * xnorm / bnorm;
        if (itn >= iter_lim) istop = 7;
        if (1 + test3 <= 1) istop = 6;
        if (1 + test2 <= 1) istop = 5;
        if (1 + t1s <= 1) istop = 4;
        if (test3 <= ctol) istop = 3;
        if (test2 <= atol) istop = 2;
        if (test1 <= rtol) istop = 1;
        o.history.push_back(r1norm);
        if (istop != 0) break;
    }
    finish();
}

template <class T>
void lsmr(
    const Op<T>& A,
    const Op<T>& AH,
    const T* b,
    const T* x0,
    int64_t m,
    int64_t n,
    double damp,
    double atol,
    double btol,
    double conlim,
    int64_t maxiter,
    Out<T>& o
) {
    Vec<T> x = start(x0, n), u(m), v(n, T{}), h(n), hbar(n, T{}), tmpm(m), tmpn(n);
    std::memcpy(u.data(), b, static_cast<std::size_t>(m) * sizeof(T));
    const double normb = nrm2(b, m);
    double beta, alpha;
    if (!x0) {
        beta = normb;
    } else {
        A.apply(x.data(), tmpm.data());
        each(m, [&](int64_t i) { u[i] = sub(u[i], tmpm[i]); });
        beta = nrm2(u.data(), m);
    }
    if (beta > 0) {
        scale_real(u.data(), 1 / beta, m);
        AH.apply(u.data(), v.data());
        alpha = nrm2(v.data(), n);
    } else {
        alpha = 0;
    }
    if (alpha > 0) scale_real(v.data(), 1 / alpha, n);
    int64_t itn = 0, istop = 0;
    double zetabar = alpha * beta, alphabar = alpha, rho = 1, rhobar = 1, cbar = 1, sbar = 0;
    h = v;
    double betadd = beta, betad = 0, rhodold = 1, tautildeold = 0, thetatilde = 0, zeta = 0, d = 0;
    double normA2 = alpha * alpha, maxrbar = 0, minrbar = 1e100;
    double normA = std::sqrt(normA2), condA = 1, normx = 0;
    const double ctol = conlim > 0 ? 1 / conlim : 0.0;
    double normr = beta, normar = alpha * beta;
    o.history.push_back(normr);
    auto finish = [&]() {
        o.x = std::move(x);
        o.stop = istop;
        o.status = istop;
        o.n_iter = itn;
        o.normr = normr;
        o.normar = normar;
        o.norma = normA;
        o.conda = condA;
        o.normx = normx;
    };
    if (normar == 0) return finish();
    if (normb == 0) {
        std::fill(x.begin(), x.end(), T{});
        return finish();
    }
    while (itn < maxiter) {
        ++itn;
        // the next step of the bidiagonalization
        bidiag_step(A, v.data(), u.data(), alpha, tmpm.data(), m);
        beta = nrm2(u.data(), m);
        if (beta > 0) {
            scale_real(u.data(), 1 / beta, m);
            bidiag_step(AH, u.data(), v.data(), beta, tmpn.data(), n);
            alpha = nrm2(v.data(), n);
            if (alpha > 0) scale_real(v.data(), 1 / alpha, n);
        }
        // construct the rotation Qhat_{k, 2k+1}
        const Sym hat = sym_ortho(alphabar, damp);
        const double chat = hat.c, shat = hat.s, alphahat = hat.r;
        // use a plane rotation (Q_i) to turn B_i to R_i
        const double rhoold = rho;
        const Sym q = sym_ortho(alphahat, beta);
        const double c = q.c, s = q.s;
        rho = q.r;
        const double thetanew = s * alpha;
        alphabar = c * alpha;
        // use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
        const double rhobarold = rhobar, zetaold = zeta;
        const double thetabar = sbar * rho, rhotemp = cbar * rho;
        const Sym qb = sym_ortho(cbar * rho, thetanew);
        cbar = qb.c;
        sbar = qb.s;
        rhobar = qb.r;
        zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;
        // update h, h_hat, x
        aypx_real(-(thetabar * rho / (rhoold * rhobarold)), hbar.data(), h.data(), n);
        axpy_real(zeta / (rho * rhobar), hbar.data(), x.data(), n);
        aypx_real(-(thetanew / rho), h.data(), v.data(), n);
        // estimate ||r||: apply the rotation Qhat_{k, 2k+1}
        const double betaacute = chat * betadd, betacheck = -shat * betadd;
        // apply the rotation Q_{k, k+1}
        const double betahat = c * betaacute;
        betadd = -s * betaacute;
        // apply the rotation Qtilde_{k-1}; betad = betad_{k-1} here
        const double thetatildeold = thetatilde;
        const Sym qt = sym_ortho(rhodold, thetabar);
        const double ctildeold = qt.c, stildeold = qt.s, rhotildeold = qt.r;
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;
        // betad = betad_k here, rhodold = rhod_k here
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        const double taud = (zeta - thetatilde * tautildeold) / rhodold;
        d = d + betacheck * betacheck;
        normr = std::sqrt(d + (betad - taud) * (betad - taud) + betadd * betadd);
        // estimate ||A||
        normA2 = normA2 + beta * beta;
        normA = std::sqrt(normA2);
        normA2 = normA2 + alpha * alpha;
        // estimate cond(A)
        maxrbar = std::max(maxrbar, rhobarold);
        if (itn > 1) minrbar = std::min(minrbar, rhobarold);
        condA = std::max(maxrbar, rhotemp) / std::min(minrbar, rhotemp);
        // the stopping tests: ||A^H r||, ||x||
        normar = std::fabs(zetabar);
        normx = nrm2(x.data(), n);
        const double test1 = normr / normb;
        const double test2 = (normA * normr) != 0 ? normar / (normA * normr) : kInf;
        const double test3 = 1 / condA;
        const double t1 = test1 / (1 + normA * normx / normb);
        const double rtol = btol + atol * normA * normx / normb;
        if (itn >= maxiter) istop = 7;
        if (1 + test3 <= 1) istop = 6;
        if (1 + test2 <= 1) istop = 5;
        if (1 + t1 <= 1) istop = 4;
        if (test3 <= ctol) istop = 3;
        if (test2 <= atol) istop = 2;
        if (test1 <= rtol) istop = 1;
        o.history.push_back(normr);
        if (istop > 0) break;
    }
    finish();
}

// ---- entries ----

enum class Method { CG, BiCGStab, GMRES, MINRES };

struct SquareArgs {
    double rtol = 0, atol = 0, shift = 0;
    int64_t restart = 0, maxiter = 0;
};

template <class T>
nb::dict square_typed(
    Method method,
    const char* func,
    nb::handle A_o,
    nb::handle M_o,
    const NDArray& b,
    nb::handle x0_o,
    const SquareArgs& a
) {
    const int64_t n = b.shape[0];
    const Op<T> A = parse_op<T>(A_o, n, n, func), M = parse_op<T>(M_o, n, n, func);
    const T* bp = typed<T>(b);
    const T* x0 = x0_o.is_none() ? nullptr : vec_arg<T>(x0_o, n, func, "x0");
    Out<T> o;
    {
        const parallel::Unlocked unlocked(solve_unlocked(A, M, n));
        switch (method) {
            case Method::CG: cg(A, M, bp, x0, n, a.rtol, a.atol, a.maxiter, o); break;
            case Method::BiCGStab: bicgstab(A, M, bp, x0, n, a.rtol, a.atol, a.maxiter, o); break;
            case Method::GMRES: gmres(A, M, bp, x0, n, a.rtol, a.atol, a.restart, a.maxiter, o); break;
            case Method::MINRES: minres(A, M, bp, x0, n, a.rtol, a.shift, a.maxiter, o); break;
        }
    }
    return to_dict(o);
}

nb::dict square_entry(
    Method method, const char* func, nb::handle A, nb::handle M, nb::handle b_o, nb::handle x0, const SquareArgs& a
) {
    const NDArray& b = vector_arg(b_o, func);
    if (b.dtype == DType::Float64) return square_typed<double>(method, func, A, M, b, x0, a);
    if (b.dtype == DType::Complex128) return square_typed<complex128>(method, func, A, M, b, x0, a);
    throw nb::type_error((std::string(func) + "(): 'b' must be float64 or complex128").c_str());
}

template <class T>
nb::dict lsq_typed(
    bool lsmr_method,
    nb::handle A_o,
    nb::handle AH_o,
    const NDArray& b,
    nb::handle x0_o,
    int64_t n,
    double damp,
    double atol,
    double btol,
    double conlim,
    int64_t maxiter
) {
    const char* func = lsmr_method ? "lsmr" : "lsqr";
    const int64_t m = b.shape[0];
    const Op<T> A = parse_op<T>(A_o, m, n, func), AH = parse_op<T>(AH_o, n, m, func);
    if (A.kind == Op<T>::Identity || AH.kind == Op<T>::Identity)
        throw nb::value_error((std::string(func) + "(): the operators are required").c_str());
    const T* bp = typed<T>(b);
    const T* x0 = x0_o.is_none() ? nullptr : vec_arg<T>(x0_o, n, func, "x0");
    Out<T> o;
    {
        const parallel::Unlocked unlocked(solve_unlocked(A, AH, m + n));
        if (lsmr_method) lsmr(A, AH, bp, x0, m, n, damp, atol, btol, conlim, maxiter, o);
        else lsqr(A, AH, bp, x0, m, n, damp, atol, btol, conlim, maxiter, o);
    }
    return to_dict(o);
}

nb::dict lsq_entry(
    bool lsmr_method,
    nb::handle A,
    nb::handle AH,
    nb::handle b_o,
    nb::handle x0,
    int64_t n,
    double damp,
    double atol,
    double btol,
    double conlim,
    int64_t maxiter
) {
    const char* func = lsmr_method ? "lsmr" : "lsqr";
    const NDArray& b = vector_arg(b_o, func);
    if (b.dtype == DType::Float64)
        return lsq_typed<double>(lsmr_method, A, AH, b, x0, n, damp, atol, btol, conlim, maxiter);
    if (b.dtype == DType::Complex128)
        return lsq_typed<complex128>(lsmr_method, A, AH, b, x0, n, damp, atol, btol, conlim, maxiter);
    throw nb::type_error((std::string(func) + "(): 'b' must be float64 or complex128").c_str());
}

}  // namespace

void bind_iterative(nb::module_& m) {
    m.def(
        "_cg",
        [](nb::handle A, nb::handle M, nb::handle b, nb::handle x0, double rtol, double atol, int64_t max_iter) {
            SquareArgs a;
            a.rtol = rtol;
            a.atol = atol;
            a.maxiter = max_iter;
            return square_entry(Method::CG, "cg", A, M, b, x0, a);
        },
        nb::arg("A").none(), nb::arg("M").none(), nb::arg("b"), nb::arg("x0").none(), nb::kw_only(), nb::arg("rtol"),
        nb::arg("atol"), nb::arg("max_iter")
    );
    m.def(
        "_bicgstab",
        [](nb::handle A, nb::handle M, nb::handle b, nb::handle x0, double rtol, double atol, int64_t max_iter) {
            SquareArgs a;
            a.rtol = rtol;
            a.atol = atol;
            a.maxiter = max_iter;
            return square_entry(Method::BiCGStab, "bicgstab", A, M, b, x0, a);
        },
        nb::arg("A").none(), nb::arg("M").none(), nb::arg("b"), nb::arg("x0").none(), nb::kw_only(), nb::arg("rtol"),
        nb::arg("atol"), nb::arg("max_iter")
    );
    m.def(
        "_gmres",
        [](nb::handle A, nb::handle M, nb::handle b, nb::handle x0, double rtol, double atol, int64_t restart,
           int64_t max_iter) {
            SquareArgs a;
            a.rtol = rtol;
            a.atol = atol;
            a.restart = restart;
            a.maxiter = max_iter;
            return square_entry(Method::GMRES, "gmres", A, M, b, x0, a);
        },
        nb::arg("A").none(), nb::arg("M").none(), nb::arg("b"), nb::arg("x0").none(), nb::kw_only(), nb::arg("rtol"),
        nb::arg("atol"), nb::arg("restart"), nb::arg("max_iter")
    );
    m.def(
        "_minres",
        [](nb::handle A, nb::handle M, nb::handle b, nb::handle x0, double rtol, double shift, int64_t max_iter) {
            SquareArgs a;
            a.rtol = rtol;
            a.shift = shift;
            a.maxiter = max_iter;
            return square_entry(Method::MINRES, "minres", A, M, b, x0, a);
        },
        nb::arg("A").none(), nb::arg("M").none(), nb::arg("b"), nb::arg("x0").none(), nb::kw_only(), nb::arg("rtol"),
        nb::arg("shift"), nb::arg("max_iter")
    );
    for (const bool is_lsmr : {false, true}) {
        m.def(
            is_lsmr ? "_lsmr" : "_lsqr",
            [is_lsmr](
                nb::handle A, nb::handle AH, nb::handle b, nb::handle x0, int64_t n, double damp, double atol,
                double btol, double conlim, int64_t max_iter
            ) { return lsq_entry(is_lsmr, A, AH, b, x0, n, damp, atol, btol, conlim, max_iter); },
            nb::arg("A"), nb::arg("AH"), nb::arg("b"), nb::arg("x0").none(), nb::kw_only(), nb::arg("n"),
            nb::arg("damp"), nb::arg("atol"), nb::arg("btol"), nb::arg("conlim"), nb::arg("max_iter")
        );
    }
}

}  // namespace sd::sparse
