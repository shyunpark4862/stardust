// The polynomial and rational interpolants of sd.interpolate (docs/design/interpolate.md §3.6, §6): the
// barycentric weights, the one barycentric evaluator (BarycentricInterpolator, FloaterHormannInterpolator and a
// real AAA), the derivative values at the nodes, Krogh's divided-difference table and its evaluation, the
// Floater-Hormann weights, and the evaluator of a complex barycentric rational (AAA, Floater-Hormann at complex
// points).
//
// Values are (n, ncol) row-major; complex values are (n, 2 ncol) doubles (real and imaginary parts side by side):
// with real nodes and real weights every formula here is linear in the values with real factors, so a complex
// column is two real ones. A column is computed the same way wherever it runs, alone or in a batch, on any thread.
//
//   weights      w_j = 1 / prod_{k != j} (x_j - x_k), the product in the order of the input, each multiplication
//                rounded to 53 bits with an unbounded exponent (the exponents are counted apart, so nothing
//                overflows or underflows; decision IP21), then every weight scaled by the same power of two so
//                that the largest lies in [0.5, 1) (the formula does not depend on a common factor)
//   evaluation   the second (true) barycentric formula: s_j = w_j / (x - x_j), num = sum_j s_j y_j (one fma per
//                term), den = sum_j s_j, both in the order of the nodes; num / den. At a node its value as it is;
//                where s_j overflows (x within a subnormal distance of a node) the nearest node's value. At an
//                infinite point the rational's limit sum w y / sum w (limit = true) or IEEE's NaN (the polynomial).
//                One node is a constant: its value as it is at every finite point (and as the limit), not
//                (s y) / s rounded twice
//   derivatives  scipy's recursion: D_ij = (w_j / w_i) / (x_i - x_j) computed as w_j / ((x_i - x_j) w_i),
//                D_ii = -sum_{j != i} D_ij; the derivative of order k is the barycentric interpolant of D^k y
//   Krogh        scipy's Hermite divided-difference table (a repeated node carries the next derivative, divided
//                by s!), column by column; evaluation by Horner on the Newton form, and the derivatives by
//                repeated Horner passes that move the centres to x (pass m leaves p^(m)(x) / m! in place m)
//   Floater-Hormann  scipy's closed form w_k = (-1)^(k - d) sum_{i in J_k} 1 / prod_{j = i..i+d, j != k} |x_k - x_j|
//
// Private bindings (the Python layer checks the arguments):
//   bary_weights(x) -> (w, dup)                 dup: a node equal to an earlier one, or -1
//   bary_eval(x, w, y, xq, limit) -> (q, ncol)
//   bary_node_derivative(x, w, v) -> (n, ncol)  D v
//   krogh_table(x, y) -> (c, bad)                bad: a repeated node not next to its equal, or -1
//   krogh_eval(x, c, xq, first, count) -> (count, q, ncol)   the derivatives of orders first .. first + count - 1
//   fh_weights(x, d) -> w                        x sorted
//   rat_eval(zj, wj, fj, zq) -> (q, ncol)        all complex128

#include "interpolate/interpolate.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// fn(begin, end) over [0, total) in pieces of at least `grain`, on the pool when the work (units of a float64
// operation) is large enough; the pieces depend on the sizes only.
template <typename F>
void split_range(std::int64_t total, std::int64_t work, std::int64_t grain, const F& fn) {
    if (total <= 0) return;
    if (work >= kParallelThreshold && parallel::available() && total > grain) {
        const std::int64_t per_unit = std::max<std::int64_t>(1, work / total);
        const std::int64_t per = std::max(grain, kParallelTask * 4 / per_unit);
        const std::int64_t tasks = (total + per - 1) / per;
        if (tasks > 1) {
            parallel::parallel_for(
                tasks, [&](std::int64_t t) { fn(t * per, std::min(total, (t + 1) * per)); }, work_ns(work / tasks)
            );
            return;
        }
    }
    const parallel::Unlocked unlocked(work >= kParallelThreshold);
    fn(0, total);
}

// ---------------------------------------------------------------- arguments

struct Vec {
    const double* p;
    std::int64_t n;
};

Vec vec_arg(nb::handle o, const char* name) {
    const NDArray& a = require_ndarray(o, name);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64)
        throw nb::value_error((std::string("'") + name + "' must be a C-contiguous 1-D float64 array").c_str());
    return {reinterpret_cast<const double*>(a.data), a.shape[0]};
}

// (n, ncol) float64 or complex128 values as doubles: D per row (ncol, or 2 ncol)
struct Cols {
    const double* p;
    std::int64_t n, ncol, D;
    DType dtype;
};

Cols cols_arg(nb::handle o, const char* name, std::int64_t rows) {
    const NDArray& a = require_ndarray(o, name);
    if (a.ndim != 2 || !a.c_contiguous || (rows >= 0 && a.shape[0] != rows))
        throw nb::value_error((std::string("'") + name + "' must be a C-contiguous (n, ncol) array").c_str());
    if (a.dtype != DType::Float64 && a.dtype != DType::Complex128)
        throw nb::type_error((std::string("'") + name + "' must be float64 or complex128").c_str());
    const std::int64_t per = a.dtype == DType::Complex128 ? 2 : 1;
    return {reinterpret_cast<const double*>(a.data), a.shape[0], a.shape[1], a.shape[1] * per, a.dtype};
}

// ---------------------------------------------------------------- weights

// d = f 2^e with |f| in [0.5, 1): returns f and adds e to `exp` (d finite and nonzero)
inline double split_exponent(double d, std::int64_t& exp) {
    const auto b = std::bit_cast<std::uint64_t>(d);
    const int field = static_cast<int>((b >> 52) & 0x7ff);
    if (field == 0) {  // subnormal
        int k = 0;
        const double f = std::frexp(d, &k);
        exp += k;
        return f;
    }
    exp += field - 1022;
    return std::bit_cast<double>((b & ~(std::uint64_t{0x7ff} << 52)) | (std::uint64_t{1022} << 52));
}

nb::object bary_weights(nb::handle xo) {
    const Vec X = vec_arg(xo, "x");
    const std::int64_t n = X.n;
    const std::int64_t shape[1] = {n};
    NDArray out = new_array(DType::Float64, shape);
    auto* w = reinterpret_cast<double*>(out.data);
    std::vector<std::int64_t> te(n);
    std::vector<std::int64_t> dup(n, -1);
    split_range(n, n * n * 3, 64, [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t j = b; j < e; ++j) {
            const double xj = X.p[j];
            double mant = 1.0;
            std::int64_t ex = 0;
            int since = 0;
            for (std::int64_t k = 0; k < n; ++k) {
                if (k == j) continue;
                const double d = xj - X.p[k];
                if (d == 0) {
                    dup[j] = std::max(j, k);
                    break;
                }
                mant *= split_exponent(d, ex);  // |mant| >= 2^-256 between the renormalizations: always normal
                if (++since == 256) {
                    mant = split_exponent(mant, ex);
                    since = 0;
                }
            }
            const double f = split_exponent(mant, ex);  // the product is f 2^ex, |f| in [0.5, 1)
            std::int64_t t = -ex;
            w[j] = split_exponent(1.0 / f, t);  // 1/f in (1, 2]: w_j = w[j] 2^t, |w[j]| in [0.5, 1)
            te[j] = t;
        }
    });
    std::int64_t bad = -1;
    for (std::int64_t j = 0; j < n; ++j)
        if (dup[j] >= 0 && (bad < 0 || dup[j] < bad)) bad = dup[j];
    if (bad >= 0) return nb::make_tuple(wrap(std::move(out)), bad);
    std::int64_t top = std::numeric_limits<std::int64_t>::min();
    for (std::int64_t j = 0; j < n; ++j) top = std::max(top, te[j]);
    for (std::int64_t j = 0; j < n; ++j) {
        const std::int64_t k = te[j] - top;  // <= 0; below -1100 the weight is 0 (2^-1074 of the largest)
        w[j] = k < -1100 ? 0.0 * w[j] : std::ldexp(w[j], static_cast<int>(k));
    }
    return nb::make_tuple(wrap(std::move(out)), std::int64_t{-1});
}

// ---------------------------------------------------------------- the real barycentric evaluator

struct Bary {
    const double* x;
    const double* w;
    std::int64_t n;
    const Cols& Y;
    bool limit;
    std::vector<double> lim;  // the limit at infinity per double column (limit only)
};

// the value at a point that is a node, or where an s_j overflowed: the nearest node's (the first of equals)
void nearest_row(const Bary& B, double v, double* row) {
    std::int64_t best = 0;
    double bd = std::fabs(v - B.x[0]);
    for (std::int64_t j = 1; j < B.n; ++j) {
        const double d = std::fabs(v - B.x[j]);
        if (d < bd) bd = d, best = j;
    }
    std::memcpy(row, B.Y.p + best * B.Y.D, sizeof(double) * B.Y.D);
}

// one point of any column count; s: n doubles of scratch
void bary_point(const Bary& B, double v, double* row, double* s) {
    const std::int64_t n = B.n, D = B.Y.D;
    if (v != v) {
        std::fill(row, row + D, kNaN);
        return;
    }
    if (std::isinf(v)) {
        if (B.limit) std::memcpy(row, B.lim.data(), sizeof(double) * D);
        else std::fill(row, row + D, kNaN);
        return;
    }
    double den = 0.0;
    bool hit = false;
    for (std::int64_t j = 0; j < n; ++j) {
        const double d = v - B.x[j];
        hit |= d == 0;
        s[j] = B.w[j] / d;
        den += s[j];
    }
    if (hit || !std::isfinite(den)) {
        nearest_row(B, v, row);
        return;
    }
    for (std::int64_t c = 0; c < D; ++c) {
        double num = 0.0;
        const double* y = B.Y.p + c;
        for (std::int64_t j = 0; j < n; ++j) num = std::fma(s[j], y[j * D], num);
        row[c] = num / den;
    }
}

// one column, eight points at a time: the same operations per point as bary_point
void bary_block8(const Bary& B, const double* v, double* out) {
    constexpr int L = 8;
    double num[L] = {}, den[L] = {};
    bool bad = false;
    for (int p = 0; p < L; ++p) bad |= !std::isfinite(v[p]);
    if (!bad) {
        for (std::int64_t j = 0; j < B.n; ++j) {
            const double xj = B.x[j], wj = B.w[j], yj = B.Y.p[j];
            for (int p = 0; p < L; ++p) {
                const double s = wj / (v[p] - xj);
                num[p] = std::fma(s, yj, num[p]);
                den[p] += s;
            }
        }
    }
    for (int p = 0; p < L; ++p) {
        if (bad || !std::isfinite(den[p])) {  // a node, an overflow or a point that is not finite
            std::vector<double> s(B.n);
            bary_point(B, v[p], out + p, s.data());
        } else {
            out[p] = num[p] / den[p];
        }
    }
}

nb::object bary_eval(nb::handle xo, nb::handle wo, nb::handle yo, nb::handle qo, bool limit) {
    const Vec X = vec_arg(xo, "x");
    const Vec W = vec_arg(wo, "w");
    const Cols Y = cols_arg(yo, "y", X.n);
    const Vec Q = vec_arg(qo, "xq");
    if (W.n != X.n || X.n < 1) throw nb::value_error("'w' must have one weight per node");
    const std::int64_t n = X.n, nq = Q.n, D = Y.D;
    const std::int64_t shape[2] = {nq, Y.ncol};
    NDArray out = new_array(Y.dtype, shape);
    if (nq == 0 || D == 0) return wrap(std::move(out));
    Bary B{X.p, W.p, n, Y, limit, {}};
    if (limit) {
        double sw = 0.0;
        for (std::int64_t j = 0; j < n; ++j) sw += W.p[j];
        B.lim.resize(D);
        for (std::int64_t c = 0; c < D; ++c) {
            double num = 0.0;
            for (std::int64_t j = 0; j < n; ++j) num = std::fma(W.p[j], Y.p[j * D + c], num);
            B.lim[c] = num / sw;
        }
    }
    auto* o = reinterpret_cast<double*>(out.data);
    if (n == 1) {  // a constant
        for (std::int64_t i = 0; i < nq; ++i) {
            const double v = Q.p[i];
            double* row = o + i * D;
            if (v != v || (std::isinf(v) && !limit)) std::fill(row, row + D, kNaN);
            else std::memcpy(row, Y.p, sizeof(double) * D);
        }
        return wrap(std::move(out));
    }
    split_range(nq, nq * n * (3 + D), 64, [&](std::int64_t b, std::int64_t e) {
        std::vector<double> s(n);
        std::int64_t i = b;
        if (D == 1)
            for (; i + 8 <= e; i += 8) bary_block8(B, Q.p + i, o + i);
        for (; i < e; ++i) bary_point(B, Q.p[i], o + i * D, s.data());
    });
    return wrap(std::move(out));
}

nb::object bary_node_derivative(nb::handle xo, nb::handle wo, nb::handle vo) {
    const Vec X = vec_arg(xo, "x");
    const Vec W = vec_arg(wo, "w");
    const Cols V = cols_arg(vo, "v", X.n);
    if (W.n != X.n) throw nb::value_error("'w' must have one weight per node");
    const std::int64_t n = X.n, D = V.D;
    const std::int64_t shape[2] = {n, V.ncol};
    NDArray out = new_array(V.dtype, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    split_range(n, n * n * (4 + D), 16, [&](std::int64_t b, std::int64_t e) {
        std::vector<double> row(n);
        for (std::int64_t i = b; i < e; ++i) {
            double diag = 0.0;
            for (std::int64_t j = 0; j < n; ++j) {
                if (j == i) continue;
                row[j] = W.p[j] / ((X.p[i] - X.p[j]) * W.p[i]);
                diag += row[j];
            }
            row[i] = -diag;
            for (std::int64_t c = 0; c < D; ++c) {
                double acc = 0.0;
                for (std::int64_t j = 0; j < n; ++j) acc = std::fma(row[j], V.p[j * D + c], acc);
                o[i * D + c] = acc;
            }
        }
    });
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- Krogh

double float_factorial(std::int64_t s) {  // scipy's float_factorial: inf from 171 on
    if (s > 170) return std::numeric_limits<double>::infinity();
    double f = 1.0;
    for (std::int64_t t = 2; t <= s; ++t) f *= static_cast<double>(t);
    return f;
}

nb::object krogh_table(nb::handle xo, nb::handle yo) {
    const Vec X = vec_arg(xo, "x");
    const Cols Y = cols_arg(yo, "y", X.n);
    const std::int64_t n = X.n, D = Y.D;
    const std::int64_t shape[2] = {n, Y.ncol};
    NDArray out = new_array(Y.dtype, shape);
    auto* c = reinterpret_cast<double*>(out.data);
    // a repeated node must follow its equals (scipy: "Elements of `xi` can't be equal.")
    std::int64_t bad = -1;
    for (std::int64_t k = 1; k < n && bad < 0; ++k) {
        std::int64_t s = 0;
        while (s <= k && X.p[k - s] == X.p[k]) ++s;
        --s;
        for (std::int64_t i = 0; i < k - s; ++i)
            if (X.p[i] == X.p[k]) {
                bad = k;
                break;
            }
    }
    if (bad >= 0 || n == 0 || D == 0) return nb::make_tuple(wrap(std::move(out)), bad);
    split_range(D, n * n * 3 * D, 1, [&](std::int64_t b, std::int64_t e) {
        std::vector<double> V(n);
        for (std::int64_t col = b; col < e; ++col) {
            c[col] = Y.p[col];
            for (std::int64_t k = 1; k < n; ++k) {
                std::int64_t s = 0;
                while (s <= k && X.p[k - s] == X.p[k]) ++s;
                --s;
                V[0] = Y.p[k * D + col] / float_factorial(s);
                for (std::int64_t i = 0; i < k - s; ++i) {
                    const double den = X.p[i] - X.p[k];
                    V[i + 1] = s == 0 ? (c[i * D + col] - V[i]) / den : (V[i + 1] - V[i]) / den;
                }
                c[k * D + col] = V[k - s];
            }
        }
    });
    return nb::make_tuple(wrap(std::move(out)), std::int64_t{-1});
}

nb::object krogh_eval(nb::handle xo, nb::handle co, nb::handle qo, std::int64_t first, std::int64_t count) {
    const Vec X = vec_arg(xo, "x");
    const Cols C = cols_arg(co, "c", X.n);
    const Vec Q = vec_arg(qo, "xq");
    if (first < 0 || count < 0) throw nb::value_error("'first' and 'count' must be nonnegative");
    const std::int64_t n = X.n, nq = Q.n, D = C.D;
    const std::int64_t shape[3] = {count, nq, C.ncol};
    NDArray out = new_array(C.dtype, shape, true);
    auto* o = reinterpret_cast<double*>(out.data);
    const std::int64_t last = std::min(first + count, n) - 1;  // the highest order that is not 0
    if (nq == 0 || D == 0) return wrap(std::move(out));
    const std::int64_t plane = nq * D;
    // the orders from n on are 0, and NaN at a NaN point (decision IP64)
    for (std::int64_t m = std::max(first, n); m < first + count; ++m)
        for (std::int64_t i = 0; i < nq; ++i)
            if (Q.p[i] != Q.p[i])
                std::fill(o + (m - first) * plane + i * D, o + (m - first) * plane + (i + 1) * D, kNaN);
    if (n == 0 || last < first) return wrap(std::move(out));
    std::vector<double> fact(last + 1);
    for (std::int64_t m = 0; m <= last; ++m) fact[m] = float_factorial(m);
    const bool value_only = first == 0 && last == 0;
    split_range(nq, nq * n * D * 2 * (last + 1), 64, [&](std::int64_t b, std::int64_t e) {
        std::vector<double> s(n), bb(n);
        std::int64_t i = b;
        if (value_only && D == 1) {  // Horner on eight points at once: the same fma sequence per point
            for (; i + 8 <= e; i += 8) {
                double acc[8], v[8];
                for (int p = 0; p < 8; ++p) acc[p] = C.p[n - 1], v[p] = Q.p[i + p];
                for (std::int64_t k = n - 2; k >= 0; --k) {
                    const double xk = X.p[k], ck = C.p[k];
                    for (int p = 0; p < 8; ++p) acc[p] = std::fma(v[p] - xk, acc[p], ck);
                }
                for (int p = 0; p < 8; ++p) o[i + p] = acc[p];
            }
        }
        for (; i < e; ++i) {
            const double v = Q.p[i];
            for (std::int64_t k = 0; k + 1 < n; ++k) s[k] = v - X.p[k];
            for (std::int64_t col = 0; col < D; ++col) {
                for (std::int64_t k = 0; k < n; ++k) bb[k] = C.p[k * D + col];
                for (std::int64_t m = 0; m <= last; ++m) {
                    // pass m: the centres (x, ..., x, x_0, x_1, ...) with x in places 0 .. m - 1
                    for (std::int64_t k = n - 2; k >= m; --k) bb[k] = std::fma(s[k - m], bb[k + 1], bb[k]);
                    if (m >= first) o[(m - first) * plane + i * D + col] = m == 0 ? bb[0] : bb[m] * fact[m];
                }
            }
        }
    });
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- Floater-Hormann

nb::object fh_weights(nb::handle xo, std::int64_t d) {
    const Vec X = vec_arg(xo, "x");
    const std::int64_t n = X.n;
    if (d < 0 || d >= n) throw nb::value_error("'d' must satisfy 0 <= d < n");
    const std::int64_t shape[1] = {n};
    NDArray out = new_array(DType::Float64, shape);
    auto* w = reinterpret_cast<double*>(out.data);
    split_range(n, n * (d + 1) * (d + 1) * 2, 64, [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t k = b; k < e; ++k) {
            double acc = 0.0;
            for (std::int64_t i = std::max<std::int64_t>(k - d, 0); i < std::min(k + 1, n - d); ++i) {
                double prod = 1.0;
                for (std::int64_t j = i; j <= i + d; ++j)
                    if (j != k) prod *= std::fabs(X.p[k] - X.p[j]);
                acc += 1.0 / prod;
            }
            w[k] = (k - d) % 2 == 0 ? acc : -acc;
        }
    });
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- the complex barycentric evaluator

// 1 / (r + i m) without overflow or underflow in the squares: scaled by a power of two first
inline void cinv(double r, double m, double& ir, double& im) {
    const double big = std::max(std::fabs(r), std::fabs(m));
    if (big == 0 || !std::isfinite(big)) {  // 1/0 and 1/inf as IEEE's real division gives them
        ir = 1.0 / r;
        im = big == 0 ? -1.0 / m : -0.0 * m;
        return;
    }
    std::int64_t e = 0;
    split_exponent(big, e);  // big = f 2^e: (r, m) 2^-e has modulus in [0.5, sqrt 2), exactly
    const int k = static_cast<int>(-e);
    const double rs = std::ldexp(r, k), ms = std::ldexp(m, k);
    const double q = std::fma(rs, rs, ms * ms);
    ir = std::ldexp(rs / q, k);  // 1 / (r + i m) = conj(rs + i ms) / q 2^-e
    im = std::ldexp(-ms / q, k);
}

// (a + i b)(c + i d) with one fma per part
inline void cmul(double a, double b, double c, double d, double& re, double& im) {
    re = std::fma(a, c, -(b * d));
    im = std::fma(a, d, b * c);
}

nb::object rat_eval(nb::handle zjo, nb::handle wjo, nb::handle fjo, nb::handle zqo) {
    const NDArray& zj = require_ndarray(zjo, "zj");
    const NDArray& wj = require_ndarray(wjo, "wj");
    const Cols F = cols_arg(fjo, "fj", -1);
    const NDArray& zq = require_ndarray(zqo, "zq");
    for (const NDArray* a : {&zj, &wj, &zq})
        if (a->ndim != 1 || !a->c_contiguous || a->dtype != DType::Complex128)
            throw nb::value_error("'zj', 'wj' and 'zq' must be C-contiguous 1-D complex128 arrays");
    if (F.dtype != DType::Complex128 || F.n != zj.shape[0] || wj.shape[0] != zj.shape[0])
        throw nb::value_error("'fj' must be a complex128 (m, ncol) array");
    const std::int64_t m = zj.shape[0], nq = zq.shape[0], K = F.ncol;
    const auto* Z = reinterpret_cast<const double*>(zj.data);
    const auto* Wt = reinterpret_cast<const double*>(wj.data);
    const auto* Qz = reinterpret_cast<const double*>(zq.data);
    const std::int64_t shape[2] = {nq, K};
    NDArray out = new_array(DType::Complex128, shape);
    if (nq == 0 || K == 0) return wrap(std::move(out));
    auto* o = reinterpret_cast<double*>(out.data);
    if (m == 1) {  // one support point: a constant, its value as it is at every point but NaN (and as the limit)
        for (std::int64_t q = 0; q < nq; ++q) {
            double* row = o + q * 2 * K;
            const double zr = Qz[2 * q], zi = Qz[2 * q + 1];
            if (!std::isinf(zr) && !std::isinf(zi) && (zr != zr || zi != zi)) std::fill(row, row + 2 * K, kNaN);
            else std::memcpy(row, F.p, sizeof(double) * 2 * K);
        }
        return wrap(std::move(out));
    }
    // the limit at infinity: sum w f / sum w
    std::vector<double> lim(2 * K);
    {
        double sr = 0.0, si = 0.0;
        for (std::int64_t j = 0; j < m; ++j) sr += Wt[2 * j], si += Wt[2 * j + 1];
        double ir, im;
        cinv(sr, si, ir, im);
        for (std::int64_t c = 0; c < K; ++c) {
            double nr = 0.0, ni = 0.0;
            for (std::int64_t j = 0; j < m; ++j) {
                const double fr = F.p[j * 2 * K + 2 * c], fi = F.p[j * 2 * K + 2 * c + 1];
                nr = std::fma(Wt[2 * j], fr, nr);
                nr = std::fma(-Wt[2 * j + 1], fi, nr);
                ni = std::fma(Wt[2 * j], fi, ni);
                ni = std::fma(Wt[2 * j + 1], fr, ni);
            }
            cmul(nr, ni, ir, im, lim[2 * c], lim[2 * c + 1]);
        }
    }
    split_range(nq, nq * m * (12 + 4 * K), 64, [&](std::int64_t b, std::int64_t e) {
        std::vector<double> lr(m), li(m);
        for (std::int64_t q = b; q < e; ++q) {
            double* row = o + q * 2 * K;
            const double zr = Qz[2 * q], zi = Qz[2 * q + 1];
            if (std::isinf(zr) || std::isinf(zi)) {
                std::memcpy(row, lim.data(), sizeof(double) * 2 * K);
                continue;
            }
            if (zr != zr || zi != zi) {
                std::fill(row, row + 2 * K, kNaN);
                continue;
            }
            double dr_ = 0.0, di_ = 0.0;
            bool hit = false;
            for (std::int64_t j = 0; j < m; ++j) {
                const double dr = zr - Z[2 * j], di = zi - Z[2 * j + 1];
                if (dr == 0 && di == 0) {
                    hit = true;
                    break;
                }
                double ir, im;
                cinv(dr, di, ir, im);
                cmul(Wt[2 * j], Wt[2 * j + 1], ir, im, lr[j], li[j]);
                dr_ += lr[j];
                di_ += li[j];
            }
            if (hit || !std::isfinite(dr_) || !std::isfinite(di_)) {  // a support point, or an overflow next to one
                std::int64_t best = 0;
                double bd = std::hypot(zr - Z[0], zi - Z[1]);
                for (std::int64_t j = 1; j < m; ++j) {
                    const double d = std::hypot(zr - Z[2 * j], zi - Z[2 * j + 1]);
                    if (d < bd) bd = d, best = j;
                }
                std::memcpy(row, F.p + best * 2 * K, sizeof(double) * 2 * K);
                continue;
            }
            double ir, im;
            cinv(dr_, di_, ir, im);
            for (std::int64_t c = 0; c < K; ++c) {
                double nr = 0.0, ni = 0.0;
                for (std::int64_t j = 0; j < m; ++j) {
                    const double fr = F.p[j * 2 * K + 2 * c], fi = F.p[j * 2 * K + 2 * c + 1];
                    nr = std::fma(lr[j], fr, nr);
                    nr = std::fma(-li[j], fi, nr);
                    ni = std::fma(lr[j], fi, ni);
                    ni = std::fma(li[j], fr, ni);
                }
                cmul(nr, ni, ir, im, row[2 * c], row[2 * c + 1]);
            }
        }
    });
    return wrap(std::move(out));
}

}  // namespace

void bind_polyint(nb::module_& m) {
    m.def("bary_weights", &bary_weights, nb::arg("x").none());
    m.def(
        "bary_eval", &bary_eval, nb::arg("x").none(), nb::arg("w").none(), nb::arg("y").none(), nb::arg("xq").none(),
        nb::arg("limit")
    );
    m.def("bary_node_derivative", &bary_node_derivative, nb::arg("x").none(), nb::arg("w").none(), nb::arg("v").none());
    m.def("krogh_table", &krogh_table, nb::arg("x").none(), nb::arg("y").none());
    m.def(
        "krogh_eval", &krogh_eval, nb::arg("x").none(), nb::arg("c").none(), nb::arg("xq").none(), nb::arg("first"),
        nb::arg("count")
    );
    m.def("fh_weights", &fh_weights, nb::arg("x").none(), nb::arg("d"));
    m.def(
        "rat_eval", &rat_eval, nb::arg("zj").none(), nb::arg("wj").none(), nb::arg("fj").none(), nb::arg("zq").none()
    );
}

}  // namespace sd::interpolate
