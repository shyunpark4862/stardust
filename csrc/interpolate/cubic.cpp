// The cubic family of sd.interpolate (docs/design/interpolate.md §3.3, §6): the node slopes of CubicSpline
// (its boundary conditions; the no-pivot tridiagonal solve of banded.hpp), PchipInterpolator and
// Akima1DInterpolator (akima, makima), and the one Hermite routine every member builds its coefficients with
// (decision IP15), in scipy's order of operations (decision IP42):
//
//   D = (y1 - y0) / h;  t = ((s0 + s1) - 2 D) / h;  c3 = t / h;  c2 = (D - s0) / h - t;  c1 = s0;  c0 = y0
//
// The slope formulas are scipy 1.18.1's, operation for operation (tools/interpolate_ref.py evaluates the same
// definitions exactly). Complex values (CubicSpline, CubicHermiteSpline) are their real and imaginary parts
// side by side: an (n, k) complex128 array is an (n, 2k) float64 one, and every formula here is linear in the
// values with real coefficients. A batch's columns are independent: every column gets the same operations
// whichever task computes it, so the results do not depend on the thread count.
//
// Private bindings (the Python layer checks the arguments: x strictly increasing and finite, y finite and
// C-contiguous (n, k), n >= 2):
//   spline_slopes(x, y, kind0, v0, kind1, v1) -> (s, bad)  kinds 0 not-a-knot, 1 first derivative v,
//                                                         2 second derivative v, 3 periodic; bad: the first
//                                                         row whose pivot is not positive, or -1
//   pchip_slopes(x, y) -> s;  akima_slopes(x, y, makima) -> s
//   hermite(x, y, s) -> (c, bad)   c (4, n - 1, k); bad: the first interval with a coefficient that is not
//                                  finite (decision IP36), or -1

#include "interpolate/interpolate.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "interpolate/banded.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

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

struct Values {
    const double* x;
    const double* y;  // (n, K) row-major doubles
    std::int64_t n, K;
};

const NDArray& values_arg(nb::handle o, const char* what, std::int64_t n) {
    const NDArray& a = require_ndarray(o, what);
    if (a.ndim != 2 || !a.c_contiguous || a.shape[0] != n)
        throw nb::value_error(("'" + std::string(what) + "' must be a C-contiguous (n, k) array").c_str());
    if (a.dtype != DType::Float64 && a.dtype != DType::Complex128)
        throw nb::type_error(("'" + std::string(what) + "' must be float64 or complex128").c_str());
    return a;
}

const NDArray& nodes_arg(nb::handle o) {
    const NDArray& a = require_ndarray(o, "x");
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[0] < 2)
        throw nb::value_error("'x' must be a C-contiguous 1-D float64 array of 2 points or more");
    return a;
}

std::int64_t doubles_per(const NDArray& a) { return a.dtype == DType::Complex128 ? 2 : 1; }

// ---------------------------------------------------------------- CubicSpline

enum Kind : int { NotAKnot = 0, First = 1, Second = 2, Periodic = 3 };

struct Bc {
    int kind;
    std::vector<double> v;  // K doubles (First, Second)
};

// D[i, :] = (y[i + 1, :] - y[i, :]) / h[i] for the columns [c0, c1)
void divided(const Values& V, const double* h, double* D, std::int64_t c0, std::int64_t c1) {
    for (std::int64_t i = 0; i + 1 < V.n; ++i) {
        const double* y0 = V.y + i * V.K;
        const double* y1 = y0 + V.K;
        double* d = D + i * V.K;
        for (std::int64_t j = c0; j < c1; ++j) d[j] = (y1[j] - y0[j]) / h[i];
    }
}

// The periodic system for n >= 4 (scipy's condensed form: the (n - 2) x (n - 2) leading block Ac of the
// (n - 1) x (n - 1) cyclic matrix, solved for the data and for the coupling column, and the last slope from
// the Schur complement).
std::int64_t periodic_slopes(const Values& V, const double* h, double* S) {
    const std::int64_t n = V.n, K = V.K, M = n - 2;
    std::vector<double> a(M), b(M), c(M), w(M), s2(M, 0.0), D((n - 1) * K);
    for (std::int64_t i = 0; i < M; ++i) {
        const double hl = i == 0 ? h[n - 2] : h[i - 1];
        a[i] = h[i];
        b[i] = 2.0 * (hl + h[i]);
        c[i] = hl;
    }
    const std::int64_t bad = tridiagonal_factor(a.data(), b.data(), c.data(), w.data(), M);
    if (bad >= 0) return bad;
    s2[0] = -h[0];
    s2[M - 1] = -h[n - 4];
    tridiagonal_solve(a.data(), w.data(), c.data(), M, s2.data(), 1, 0, 1);
    const double am10 = h[n - 3], am1m2 = h[n - 2], am1m1 = 2.0 * (h[n - 2] + h[n - 3]);
    const double den = (am1m1 + am10 * s2[0]) + am1m2 * s2[M - 1];
    if (!(den > 0.0 && std::isfinite(den))) return n - 2;
    split_range(K, n * K * 12, 8, [&](std::int64_t c0, std::int64_t c1) {
        divided(V, h, D.data(), c0, c1);
        for (std::int64_t i = 0; i < M; ++i) {
            const double* dl = D.data() + (i == 0 ? n - 2 : i - 1) * K;
            const double* dr = D.data() + i * K;
            const double hl = i == 0 ? h[n - 2] : h[i - 1];
            double* r = S + i * K;
            for (std::int64_t j = c0; j < c1; ++j) r[j] = 3.0 * (h[i] * dl[j] + hl * dr[j]);
        }
        tridiagonal_solve(a.data(), w.data(), c.data(), M, S, K, c0, c1);
        const double* dl = D.data() + (n - 3) * K;
        const double* dr = D.data() + (n - 2) * K;
        double* last = S + (n - 2) * K;
        double* wrap = S + (n - 1) * K;
        for (std::int64_t j = c0; j < c1; ++j) {
            const double blast = 3.0 * (h[n - 2] * dl[j] + h[n - 3] * dr[j]);
            const double sm1 = ((blast - am10 * S[j]) - am1m2 * S[(M - 1) * K + j]) / den;
            last[j] = sm1;
        }
        for (std::int64_t i = 0; i < M; ++i) {
            double* r = S + i * K;
            for (std::int64_t j = c0; j < c1; ++j) r[j] = r[j] + last[j] * s2[i];
        }
        for (std::int64_t j = c0; j < c1; ++j) wrap[j] = S[j];
    });
    return -1;
}

std::int64_t spline(const Values& V, Bc b0, Bc b1, double* S) {
    const std::int64_t n = V.n, K = V.K;
    const double* x = V.x;
    std::vector<double> h(n - 1);
    for (std::int64_t i = 0; i + 1 < n; ++i) h[i] = x[i + 1] - x[i];
    if (n == 2) {  // scipy: not-a-knot and periodic become the first derivative equal to the slope
        for (Bc* bc : {&b0, &b1}) {
            if (bc->kind == NotAKnot || bc->kind == Periodic) {
                bc->kind = First;
                bc->v.resize(K);
                for (std::int64_t j = 0; j < K; ++j) bc->v[j] = (V.y[K + j] - V.y[j]) / h[0];
            }
        }
    }
    if (n == 3 && b0.kind == Periodic) {  // scipy's closed form: one slope for the three nodes
        const double w = 1.0 / h[0] + 1.0 / h[1];
        for (std::int64_t j = 0; j < K; ++j) {
            const double d0 = (V.y[K + j] - V.y[j]) / h[0], d1 = (V.y[2 * K + j] - V.y[K + j]) / h[1];
            S[j] = S[K + j] = S[2 * K + j] = (d0 / h[0] + d1 / h[1]) / w;
        }
        return -1;
    }
    if (b0.kind == Periodic) return periodic_slopes(V, h.data(), S);
    // the tridiagonal matrix (a: sub, b: diagonal, c: super)
    std::vector<double> a(n, 0.0), b(n, 0.0), c(n, 0.0), w(n, 0.0);
    for (std::int64_t i = 1; i + 1 < n; ++i) {
        a[i] = h[i];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i - 1];
    }
    const bool parabola = n == 3 && b0.kind == NotAKnot && b1.kind == NotAKnot;
    if (parabola) {  // scipy's 3 x 3 system of the parabola through the three points
        b[0] = 1.0, c[0] = 1.0;
        a[2] = 1.0, b[2] = 1.0;
    } else {
        if (b0.kind == NotAKnot) b[0] = h[1], c[0] = x[2] - x[0];
        else if (b0.kind == First) b[0] = 1.0, c[0] = 0.0;
        else b[0] = 2.0 * h[0], c[0] = h[0];
        if (b1.kind == NotAKnot) b[n - 1] = h[n - 3], a[n - 1] = x[n - 1] - x[n - 3];
        else if (b1.kind == First) b[n - 1] = 1.0, a[n - 1] = 0.0;
        else b[n - 1] = 2.0 * h[n - 2], a[n - 1] = h[n - 2];
    }
    // the end rows' right sides for column j (d0, d1: the first two divided differences; dm, dl: the last two)
    const double d0x = n >= 3 ? x[2] - x[0] : 0.0, dnx = n >= 3 ? x[n - 1] - x[n - 3] : 0.0;
    auto start_rhs = [&](std::int64_t j, double d0, double d1) {
        if (parabola) return 2.0 * d0;
        if (b0.kind == NotAKnot) return (((h[0] + 2.0 * d0x) * h[1]) * d0 + (h[0] * h[0]) * d1) / d0x;
        if (b0.kind == First) return b0.v[j];
        return (-0.5 * b0.v[j]) * (h[0] * h[0]) + 3.0 * (V.y[K + j] - V.y[j]);
    };
    auto end_rhs = [&](std::int64_t j, double dm, double dl) {
        if (parabola) return 2.0 * dl;
        if (b1.kind == NotAKnot) {
            const double hl = h[n - 2], hm = h[n - 3];
            return ((hl * hl) * dm + ((2.0 * dnx + hl) * hm) * dl) / dnx;
        }
        if (b1.kind == First) return b1.v[j];
        return (0.5 * b1.v[j]) * (h[n - 2] * h[n - 2]) + 3.0 * (V.y[(n - 1) * K + j] - V.y[(n - 2) * K + j]);
    };
    auto delta = [&](std::int64_t i, std::int64_t j) { return (V.y[(i + 1) * K + j] - V.y[i * K + j]) / h[i]; };
    if (K <= 8) {
        // Few columns: one forward pass makes the factorization, the right side (the divided differences on the
        // fly) and the forward elimination, then the back substitution: the operations of the batched path below,
        // in the same order for every column, without its passes over memory.
        double dprev[8], dcur[8];
        if (!(b[0] > 0.0 && std::isfinite(b[0]))) return 0;
        w[0] = 1.0 / b[0];
        for (std::int64_t j = 0; j < K; ++j) {
            dprev[j] = delta(0, j);
            S[j] = start_rhs(j, dprev[j], n >= 3 ? delta(1, j) : 0.0);
        }
        for (std::int64_t i = 1; i < n; ++i) {
            const double l = a[i] / b[i - 1];
            b[i] = b[i] - l * c[i - 1];
            if (!(b[i] > 0.0 && std::isfinite(b[i]))) return i;
            w[i] = 1.0 / b[i];
            a[i] = l;
            double* r = S + i * K;
            const double* p = r - K;
            if (i + 1 < n) {
                for (std::int64_t j = 0; j < K; ++j) {
                    dcur[j] = delta(i, j);
                    r[j] = 3.0 * (h[i] * dprev[j] + h[i - 1] * dcur[j]);
                    r[j] = r[j] - l * p[j];
                    dprev[j] = dcur[j];
                }
            } else {
                for (std::int64_t j = 0; j < K; ++j) {
                    r[j] = end_rhs(j, n >= 3 ? delta(n - 3, j) : 0.0, dprev[j]);
                    r[j] = r[j] - l * p[j];
                }
            }
        }
        double* rn = S + (n - 1) * K;
        for (std::int64_t j = 0; j < K; ++j) rn[j] = rn[j] * w[n - 1];
        for (std::int64_t i = n - 2; i >= 0; --i) {
            double* r = S + i * K;
            const double* q = r + K;
            for (std::int64_t j = 0; j < K; ++j) r[j] = (r[j] - c[i] * q[j]) * w[i];
        }
        return -1;
    }
    const std::int64_t bad = tridiagonal_factor(a.data(), b.data(), c.data(), w.data(), n);
    if (bad >= 0) return bad;
    std::vector<double> D((n - 1) * K);
    split_range(K, n * K * 12, 8, [&](std::int64_t c0, std::int64_t c1) {
        divided(V, h.data(), D.data(), c0, c1);
        for (std::int64_t i = 1; i + 1 < n; ++i) {
            const double* dl = D.data() + (i - 1) * K;
            const double* dr = dl + K;
            double* r = S + i * K;
            for (std::int64_t j = c0; j < c1; ++j) r[j] = 3.0 * (h[i] * dl[j] + h[i - 1] * dr[j]);
        }
        const double* D0 = D.data();
        const double* Dl = D.data() + (n - 2) * K;  // the last interval's
        for (std::int64_t j = c0; j < c1; ++j) {
            S[j] = start_rhs(j, D0[j], n >= 3 ? D0[K + j] : 0.0);
            S[(n - 1) * K + j] = end_rhs(j, n >= 3 ? Dl[j - K] : 0.0, Dl[j]);
        }
        tridiagonal_solve(a.data(), w.data(), c.data(), n, S, K, c0, c1);
    });
    return -1;
}

Bc bc_arg(int kind, nb::handle v, std::int64_t K, const NDArray& y) {
    if (kind < 0 || kind > 3) throw nb::value_error("boundary kind must be 0, 1, 2 or 3");
    Bc bc{kind, {}};
    if (kind == First || kind == Second) {
        const NDArray& a = require_ndarray(v, "value");
        if (a.ndim != 1 || !a.c_contiguous || a.dtype != y.dtype || a.shape[0] * doubles_per(a) != K)
            throw nb::value_error("a boundary value must be a C-contiguous (k,) array of y's dtype");
        const auto* p = reinterpret_cast<const double*>(a.data);
        bc.v.assign(p, p + K);
    }
    return bc;
}

nb::object spline_slopes(nb::handle xo, nb::handle yo, int k0, nb::handle v0, int k1, nb::handle v1) {
    const NDArray& x = nodes_arg(xo);
    const std::int64_t n = x.shape[0];
    const NDArray& y = values_arg(yo, "y", n);
    const std::int64_t K = y.shape[1] * doubles_per(y);
    if ((k0 == Periodic) != (k1 == Periodic)) throw nb::value_error("'periodic' applies to both ends");
    Bc b0 = bc_arg(k0, v0, K, y), b1 = bc_arg(k1, v1, K, y);
    const std::int64_t shape[2] = {n, y.shape[1]};
    NDArray s = new_array(y.dtype, shape);
    std::int64_t bad = -1;
    if (K > 0) {
        const Values V{reinterpret_cast<const double*>(x.data), reinterpret_cast<const double*>(y.data), n, K};
        bad = spline(V, std::move(b0), std::move(b1), reinterpret_cast<double*>(s.data));
    }
    return nb::make_tuple(wrap(std::move(s)), bad);
}

// ---------------------------------------------------------------- PCHIP

inline int sgn(double v) { return (v > 0.0) - (v < 0.0); }

// scipy's PchipInterpolator._edge_case: the one-sided three-point estimate, kept monotone
inline double pchip_edge(double h0, double h1, double m0, double m1) {
    const double d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (sgn(d) != sgn(m0)) return 0.0;
    if (sgn(m0) != sgn(m1) && std::fabs(d) > 3.0 * std::fabs(m0)) return 3.0 * m0;
    return d;
}

nb::object pchip_slopes(nb::handle xo, nb::handle yo) {
    const NDArray& x = nodes_arg(xo);
    const std::int64_t n = x.shape[0];
    const NDArray& y = values_arg(yo, "y", n);
    if (y.dtype != DType::Float64) throw nb::type_error("'y' must be float64");
    const std::int64_t K = y.shape[1];
    const std::int64_t shape[2] = {n, K};
    NDArray out = new_array(DType::Float64, shape);
    const auto* X = reinterpret_cast<const double*>(x.data);
    const auto* Y = reinterpret_cast<const double*>(y.data);
    auto* S = reinterpret_cast<double*>(out.data);
    if (K == 0) return wrap(std::move(out));
    auto slope = [&](std::int64_t i, std::int64_t j) {
        return (Y[(i + 1) * K + j] - Y[i * K + j]) / (X[i + 1] - X[i]);
    };
    if (n == 2) {
        for (std::int64_t j = 0; j < K; ++j) S[j] = S[K + j] = slope(0, j);
        return wrap(std::move(out));
    }
    split_range(n, n * K * 16, 1024, [&](std::int64_t i0, std::int64_t i1) {
        for (std::int64_t i = std::max<std::int64_t>(i0, 1); i < std::min(i1, n - 1); ++i) {
            const double hl = X[i] - X[i - 1], hr = X[i + 1] - X[i];
            const double w1 = 2.0 * hr + hl, w2 = hr + 2.0 * hl;
            for (std::int64_t j = 0; j < K; ++j) {
                const double ml = slope(i - 1, j), mr = slope(i, j);
                if (sgn(mr) != sgn(ml) || mr == 0.0 || ml == 0.0) {
                    S[i * K + j] = 0.0;
                    continue;
                }
                const double whmean = (w1 / ml + w2 / mr) / (w1 + w2);
                S[i * K + j] = 1.0 / whmean;
            }
        }
    });
    const double h0 = X[1] - X[0], h1 = X[2] - X[1], hl = X[n - 1] - X[n - 2], hm = X[n - 2] - X[n - 3];
    for (std::int64_t j = 0; j < K; ++j) {
        S[j] = pchip_edge(h0, h1, slope(0, j), slope(1, j));
        S[(n - 1) * K + j] = pchip_edge(hl, hm, slope(n - 2, j), slope(n - 3, j));
    }
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- Akima, makima

constexpr double kAkimaBreak = 1e-9;  // scipy's cutoff: f12 > 1e-9 max(f12), the max per column (IP33)

nb::object akima_slopes(nb::handle xo, nb::handle yo, bool makima) {
    const NDArray& x = nodes_arg(xo);
    const std::int64_t n = x.shape[0];
    const NDArray& y = values_arg(yo, "y", n);
    if (y.dtype != DType::Float64) throw nb::type_error("'y' must be float64");
    const std::int64_t K = y.shape[1];
    const std::int64_t shape[2] = {n, K};
    NDArray out = new_array(DType::Float64, shape);
    const auto* X = reinterpret_cast<const double*>(x.data);
    const auto* Y = reinterpret_cast<const double*>(y.data);
    auto* S = reinterpret_cast<double*>(out.data);
    if (K == 0) return wrap(std::move(out));
    if (n == 2) {  // the line (scipy)
        for (std::int64_t j = 0; j < K; ++j) S[j] = S[K + j] = (Y[K + j] - Y[j]) / (X[1] - X[0]);
        return wrap(std::move(out));
    }
    // m[0 .. n + 3) per column: two extrapolated slopes on each side of the n - 1 divided differences
    std::vector<double> m((n + 3) * K);
    auto M = [&](std::int64_t i, std::int64_t j) -> double& { return m[i * K + j]; };
    split_range(n - 1, (n - 1) * K * 3, 4096, [&](std::int64_t i0, std::int64_t i1) {
        for (std::int64_t i = i0; i < i1; ++i)
            for (std::int64_t j = 0; j < K; ++j) M(i + 2, j) = (Y[(i + 1) * K + j] - Y[i * K + j]) / (X[i + 1] - X[i]);
    });
    for (std::int64_t j = 0; j < K; ++j) {
        M(1, j) = 2.0 * M(2, j) - M(3, j);
        M(0, j) = 2.0 * M(1, j) - M(2, j);
        M(n + 1, j) = 2.0 * M(n, j) - M(n - 1, j);
        M(n + 2, j) = 2.0 * M(n + 1, j) - M(n, j);
    }
    // f1 = |m[i+3] - m[i+2]| (+ |m[i+3] + m[i+2]| / 2), f2 = |m[i+1] - m[i]| (+ |m[i+1] + m[i]| / 2)
    auto f1f2 = [&](std::int64_t i, std::int64_t j, double& f1, double& f2) {
        f1 = std::fabs(M(i + 3, j) - M(i + 2, j));
        f2 = std::fabs(M(i + 1, j) - M(i, j));
        if (makima) {
            f1 = f1 + 0.5 * std::fabs(M(i + 3, j) + M(i + 2, j));
            f2 = f2 + 0.5 * std::fabs(M(i + 1, j) + M(i, j));
        }
    };
    std::vector<double> cut(K, -INFINITY);
    for (std::int64_t i = 0; i < n; ++i)
        for (std::int64_t j = 0; j < K; ++j) {
            double f1, f2;
            f1f2(i, j, f1, f2);
            cut[j] = std::fmax(cut[j], f1 + f2);
        }
    for (std::int64_t j = 0; j < K; ++j) cut[j] = kAkimaBreak * cut[j];
    split_range(n, n * K * 14, 1024, [&](std::int64_t i0, std::int64_t i1) {
        for (std::int64_t i = i0; i < i1; ++i)
            for (std::int64_t j = 0; j < K; ++j) {
                double f1, f2;
                f1f2(i, j, f1, f2);
                const double f12 = f1 + f2;
                if (f12 > cut[j]) S[i * K + j] = M(i + 1, j) + (f2 / f12) * (M(i + 2, j) - M(i + 1, j));
                else S[i * K + j] = 0.5 * (M(i + 3, j) + M(i, j));
            }
    });
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- Hermite coefficients

nb::object hermite(nb::handle xo, nb::handle yo, nb::handle so) {
    const NDArray& x = nodes_arg(xo);
    const std::int64_t n = x.shape[0], m = n - 1;
    const NDArray& y = values_arg(yo, "y", n);
    const NDArray& s = values_arg(so, "dydx", n);
    if (s.dtype != y.dtype || s.shape[1] != y.shape[1]) throw nb::value_error("'y' and 'dydx' must match");
    const std::int64_t K = y.shape[1] * doubles_per(y);
    const std::int64_t shape[3] = {4, m, y.shape[1]};
    NDArray out = new_array(y.dtype, shape);
    const auto* X = reinterpret_cast<const double*>(x.data);
    const auto* Y = reinterpret_cast<const double*>(y.data);
    const auto* S = reinterpret_cast<const double*>(s.data);
    auto* C = reinterpret_cast<double*>(out.data);
    std::atomic<std::int64_t> bad{m};
    split_range(m, m * K * 12, 1024, [&](std::int64_t i0, std::int64_t i1) {
        std::int64_t first = m;
        for (std::int64_t i = i0; i < i1; ++i) {
            const double h = X[i + 1] - X[i];
            const double *y0 = Y + i * K, *y1 = y0 + K, *s0 = S + i * K, *s1 = s0 + K;
            double* c3 = C + i * K;
            double* c2 = c3 + m * K;
            double* c1 = c2 + m * K;
            double* c0 = c1 + m * K;
            bool finite = true;
            for (std::int64_t j = 0; j < K; ++j) {
                const double d = (y1[j] - y0[j]) / h;
                const double t = ((s0[j] + s1[j]) - 2.0 * d) / h;
                c3[j] = t / h;
                c2[j] = (d - s0[j]) / h - t;
                c1[j] = s0[j];
                c0[j] = y0[j];
                finite = finite && std::isfinite(c3[j]) && std::isfinite(c2[j]) && std::isfinite(c1[j]);
            }
            if (!finite && first == m) first = i;
        }
        std::int64_t cur = bad.load();
        while (first < cur && !bad.compare_exchange_weak(cur, first)) {}
    });
    const std::int64_t b = bad.load();
    return nb::make_tuple(wrap(std::move(out)), b == m ? -1 : b);
}

}  // namespace

void bind_cubic(nb::module_& m) {
    m.def(
        "spline_slopes", &spline_slopes, nb::arg("x").none(), nb::arg("y").none(), nb::arg("kind0"),
        nb::arg("v0").none(), nb::arg("kind1"), nb::arg("v1").none()
    );  // fmt: skip
    m.def("pchip_slopes", &pchip_slopes, nb::arg("x").none(), nb::arg("y").none());
    m.def("akima_slopes", &akima_slopes, nb::arg("x").none(), nb::arg("y").none(), nb::arg("makima"));
    m.def("hermite", &hermite, nb::arg("x").none(), nb::arg("y").none(), nb::arg("dydx").none());
}

}  // namespace sd::interpolate
