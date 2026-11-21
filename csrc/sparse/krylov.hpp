// The Krylov kernels' shared pieces (docs/design/sparse.md §3.7, §3.8): numpy's scalar arithmetic,
// the deterministic vector operations (sd.linalg.vecdot's sums, sd.linalg.vector_norm's 2-norm,
// element-wise updates), the operators (a CSR matrix through the sparse SpMV, a Python callable,
// the identity) and LAPACK 3.10+'s lartg. The iterative solvers (iterative.cpp) and the eigen
// solvers (arpack.cpp, svds.cpp) use them, so the same operations give the same bits in both.
//
// Every vector operation is split by size alone, so a result is the same bit for bit for any
// thread count (API.md §3).
#pragma once

#include <arm_neon.h>

#include "array/kernels/scalar_ops.hpp"
#include "base/pairwise.hpp"
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

// ---- several vectors in one pass (the same bits as the one-vector operations above) ----

// The leaves of dot()'s pairwise tree over n terms (pairwise::tree: runs of at most kBlock doubles,
// kBlock / 2 complex), left to right; the leaf sums combined along the same tree give dot()'s sum
// exactly (pairwise::combine), whichever task computed each leaf.
inline const pairwise::Leaves& dot_leaves(int64_t n, bool complex) {
    thread_local pairwise::Leaves cache[2];
    thread_local int64_t cached[2] = {-1, -1};
    const int c = complex ? 1 : 0;
    if (cached[c] != n) {
        cache[c] = pairwise::plan(n, complex ? pairwise::kBlock / 2 : pairwise::kBlock, complex);
        cached[c] = n;
    }
    return cache[c];
}

template <class T>
inline T dot_leaf(const T* a, const T* b, int64_t m) {
    if constexpr (kComplex<T>) return linalg::detail::fma_cdot_leaf(a, b, m);
    else return linalg::detail::fma_dot_leaf(a, b, m);
}

// fma_dot_leaf of two vectors against one b at once (b's loads shared): each result has the very
// arithmetic of fma_dot_leaf (the same lanes, the same lane tree, the same tail).
inline void dot_leaf2(const double* a0, const double* a1, const double* b, int64_t n, double& r0, double& r1) {
    int64_t i = 0;
    r0 = 0.;
    r1 = 0.;
    if (n >= 16) {
        float64x2_t c[8], d[8];
        for (int j = 0; j < 8; ++j) {
            const float64x2_t bv = vld1q_f64(b + 2 * j);
            c[j] = vmulq_f64(vld1q_f64(a0 + 2 * j), bv);
            d[j] = vmulq_f64(vld1q_f64(a1 + 2 * j), bv);
        }
        for (i = 16; i + 16 <= n; i += 16)
            for (int j = 0; j < 8; ++j) {
                const float64x2_t bv = vld1q_f64(b + i + 2 * j);
                c[j] = vfmaq_f64(c[j], vld1q_f64(a0 + i + 2 * j), bv);
                d[j] = vfmaq_f64(d[j], vld1q_f64(a1 + i + 2 * j), bv);
            }
        const float64x2_t s = vaddq_f64(
            vaddq_f64(vaddq_f64(c[0], c[1]), vaddq_f64(c[2], c[3])),
            vaddq_f64(vaddq_f64(c[4], c[5]), vaddq_f64(c[6], c[7]))
        );
        const float64x2_t t = vaddq_f64(
            vaddq_f64(vaddq_f64(d[0], d[1]), vaddq_f64(d[2], d[3])),
            vaddq_f64(vaddq_f64(d[4], d[5]), vaddq_f64(d[6], d[7]))
        );
        r0 = vgetq_lane_f64(s, 0) + vgetq_lane_f64(s, 1);
        r1 = vgetq_lane_f64(t, 0) + vgetq_lane_f64(t, 1);
    }
    for (; i < n; ++i) {
        r0 = std::fma(a0[i], b[i], r0);
        r1 = std::fma(a1[i], b[i], r1);
    }
}

// dot()'s 0 + tree, from the leaf sums parts[0, nl) of n terms
template <class T>
T dot_of_leaves(int64_t n, const T* parts, int64_t nl) {
    const std::vector<T> v(parts, parts + nl);
    const T s = pairwise::combine(n, kComplex<T> ? pairwise::kBlock / 2 : pairwise::kBlock, v);
    if constexpr (kComplex<T>) return {0. + s.real(), 0. + s.imag()};
    else return 0. + s;
}

// y[l] = dot(V[:, l], x) for l < j (V's columns of n entries one after another): leaf by leaf over
// all the columns, so x's leaf is read once from cache, two real columns per sweep; tasks of whole
// leaves cut by n and j alone. Bit for bit j calls of dot() (SP8: the Lanczos and Arnoldi steps).
template <class T>
void dots(const T* V, int64_t n, int64_t j, const T* x, T* y) {
    if (j <= 0) return;
    if (j == 1 || n == 0) {
        for (int64_t l = 0; l < j; ++l) y[l] = dot(V + l * n, x, n);
        return;
    }
    const pairwise::Leaves& lv = dot_leaves(n, kComplex<T>);
    const int64_t nl = static_cast<int64_t>(lv.size());
    std::vector<T> parts(static_cast<std::size_t>(nl * j));
    const int64_t per = std::max<int64_t>(1, kChunk / (j * pairwise::kBlock));  // leaves a task
    const int64_t nt = (nl + per - 1) / per;
    for_tasks(nt, n * j, [&](int64_t t) {
        for (int64_t q = t * per; q < std::min(nl, (t + 1) * per); ++q) {
            const int64_t b0 = lv[static_cast<std::size_t>(q)].first, m = lv[static_cast<std::size_t>(q)].second;
            int64_t l = 0;
            if constexpr (!kComplex<T>)
                for (; l + 2 <= j; l += 2)
                    dot_leaf2(
                        V + l * n + b0, V + (l + 1) * n + b0, x + b0, m, parts[l * nl + q], parts[(l + 1) * nl + q]
                    );
            for (; l < j; ++l) parts[l * nl + q] = dot_leaf(V + l * n + b0, x + b0, m);
        }
    });
    for (int64_t l = 0; l < j; ++l) y[l] = dot_of_leaves(n, parts.data() + l * nl, nl);
}

// y += a x, then dot(z, y) (z null: no dot, returns 0): one pass over y, the update and the dot's
// leaves together. The same bits as axpy() then dot() (SP8: the Gram-Schmidt steps of gmres).
template <class T>
T axpy_dot(T a, const T* x, T* y, const T* z, int64_t n) {
    const pairwise::Leaves& lv = dot_leaves(n, kComplex<T>);
    const int64_t nl = static_cast<int64_t>(lv.size());
    std::vector<T> parts(static_cast<std::size_t>(nl));
    const int64_t per = std::max<int64_t>(1, kChunk / pairwise::kBlock);
    const int64_t nt = (nl + per - 1) / per;
    for_tasks(nt, n, [&](int64_t t) {
        for (int64_t q = t * per; q < std::min(nl, (t + 1) * per); ++q) {
            const int64_t b0 = lv[static_cast<std::size_t>(q)].first, m = lv[static_cast<std::size_t>(q)].second;
            for (int64_t i = b0; i < b0 + m; ++i) {
                if constexpr (kComplex<T>) y[i] = add(y[i], ops::cmul(a, x[i]));
                else y[i] = std::fma(a, x[i], y[i]);
            }
            if (z) parts[q] = dot_leaf(z + b0, y + b0, m);
        }
    });
    return z ? dot_of_leaves(n, parts.data(), nl) : T{};
}

// y = (zero ? 0 : y) + sum_l coef[l] V[:, l], each element's terms in column order (blocks of rows
// that keep y's block in L1 across the columns, four columns a pass); tasks of rows cut by n and j
// alone, small enough that a basis of n = 10^4 shares out (SP8).
template <class T>
void gaxpy(const T* V, int64_t n, int64_t j, const T* coef, T* y, bool zero) {
    auto madd4 = [](T c, T v, T acc) {
        if constexpr (kComplex<T>) return add(acc, ops::cmul(c, v));
        else return std::fma(c, v, acc);
    };
    const int64_t rows = std::max<int64_t>(256, (kChunk / std::max<int64_t>(j, 1)) & ~int64_t(255));
    for_chunks(
        n,
        [&](int64_t, int64_t lo, int64_t hi) {
            constexpr int64_t kRows = 256;
            for (int64_t b0 = lo; b0 < hi; b0 += kRows) {
                const int64_t b1 = std::min(hi, b0 + kRows);
                if (zero)
                    for (int64_t i = b0; i < b1; ++i) y[i] = T{};
                int64_t l = 0;
                for (; l + 4 <= j; l += 4) {
                    const T c0 = coef[l], c1 = coef[l + 1], c2 = coef[l + 2], c3 = coef[l + 3];
                    const T *v0 = V + l * n, *v1 = v0 + n, *v2 = v1 + n, *v3 = v2 + n;
                    for (int64_t i = b0; i < b1; ++i)
                        y[i] = madd4(c3, v3[i], madd4(c2, v2[i], madd4(c1, v1[i], madd4(c0, v0[i], y[i]))));
                }
                for (; l < j; ++l) {
                    const T c = coef[l];
                    const T* v = V + l * n;
                    for (int64_t i = b0; i < b1; ++i) y[i] = madd4(c, v[i], y[i]);
                }
            }
        },
        rows, n * std::max<int64_t>(j, 1)
    );
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
