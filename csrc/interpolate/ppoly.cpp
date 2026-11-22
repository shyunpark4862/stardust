// The piecewise polynomials of sd.interpolate (docs/design/interpolate.md §2.4, §3.5, §6): evaluation with
// derivatives, the antiderivative's continuity constants, the definite integral and the real roots. The
// coefficients c are (k + 1, m, ncol) row-major, highest degree first, piece i in the local variable
// s = x - x[i]; complex coefficients are (k + 1, m, 2 ncol) doubles (real and imaginary parts side by side:
// every operation here is linear in the coefficients with real factors, except the roots, which are real only).
//
// Breakpoints are strictly increasing or strictly decreasing. A point's piece is the right-open
// x[i] <= v < x[i + 1] (the last one closed), and for decreasing breakpoints its mirror image
// x[i + 1] < v <= x[i] (scipy's two rules): the search runs on the breakpoints, or on their negatives when they
// decrease, with base/interval.hpp's count (the one sd.interp and the linear path use).
//
//   evaluation   s = fl(v - x[i]), then Horner with one fma per step; the derivative of order nu multiplies
//                each coefficient by d! / (d - nu)! as it goes (one rounding more per term)
//   antiderivative  coefficients divided by (d + 1) ... (d + nu), then the constants of degree q < nu, from
//                q = nu - 1 down: c_q[i] = c_q[i - 1] + sum_{d > q} c_d[i - 1] binom(d, q) h[i - 1]^(d - q), a
//                prefix sum over pieces taken in blocks of a fixed length (the same bits on any thread count)
//   integrate    per piece the local antiderivative's difference, summed over the pieces with the pairwise tree
//                of base/pairwise.hpp
//   roots        per piece: the critical points (the derivative's roots, recursively; a quadratic in closed form)
//                split it into monotone sections; a section whose ends differ in sign is bisected over the
//                ordered doubles (at most 64 steps) to two neighbouring doubles of opposite signs; an extremum
//                whose value is zero within the evaluation contract is a double root (decisions IP18, IP45).
//                Signs come from a compensated Horner (an error-free product and sum per step), exact to the
//                sign wherever |p| exceeds a few u^2 sum |c_d s^d|: two roots 2^-30 apart are both found.
//
// Private bindings (the Python layer checks shapes and dtypes):
//   check_x(x, decreasing_ok) -> 0 increasing, 1 decreasing, -1 not finite, -2 not strictly monotone
//   all_finite(a) -> bool
//   ppoly_eval(c, x, xq, nu, rule) -> (q, ncol)        rule 0 NaN outside, 1 extend, 2 periodic
//   ppoly_antiderivative(c, x, nu) -> c2 (k + 1 + nu, m, ncol)
//   ppoly_integrate(c, x, a, b, extend) -> (ncol,)    a <= b; NaN when outside without extend
//   ppoly_roots(c, x, y, discontinuity, extend) -> list of ncol 1-D arrays

#include "interpolate/interpolate.hpp"

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "base/interval.hpp"
#include "base/pairwise.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "interpolate/search.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sd::interpolate {

namespace nb = nanobind;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kU = 0x1p-53;

// ---------------------------------------------------------------- arguments

struct Coef {
    const double* c;
    std::int64_t K1, m, ncol, D;  // D: doubles per row of a piece (ncol, or 2 ncol for complex)
    DType dtype;
    double at(std::int64_t j, std::int64_t i, std::int64_t col) const { return c[(j * m + i) * D + col]; }
};

Coef coef_arg(nb::handle o) {
    const NDArray& a = require_ndarray(o, "c");
    if (a.ndim != 3 || !a.c_contiguous || a.shape[0] < 1 || a.shape[1] < 1)
        throw nb::value_error("'c' must be a C-contiguous (k + 1, m, ncol) array");
    if (a.dtype != DType::Float64 && a.dtype != DType::Complex128)
        throw nb::type_error("'c' must be float64 or complex128");
    const std::int64_t per = a.dtype == DType::Complex128 ? 2 : 1;
    return {reinterpret_cast<const double*>(a.data), a.shape[0], a.shape[1], a.shape[2], a.shape[2] * per, a.dtype};
}

// Breakpoints (n = m + 1 >= 2, strictly monotone, finite) and their search keys: x itself when increasing,
// -x when decreasing (so that the right-open rule on the keys is scipy's mirrored rule on x).
struct Breaks {
    const double* x;
    std::int64_t n;
    bool ascending;
    std::vector<double> neg;
    const double* key;

    Breaks(nb::handle o, std::int64_t m) {
        const NDArray& a = require_ndarray(o, "x");
        if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[0] != m + 1)
            throw nb::value_error("'x' must be a C-contiguous 1-D float64 array of m + 1 breakpoints");
        x = reinterpret_cast<const double*>(a.data);
        n = a.shape[0];
        ascending = x[n - 1] > x[0];
        key = x;
        if (!ascending) {
            neg.resize(n);
            for (std::int64_t i = 0; i < n; ++i) neg[i] = -x[i];
            key = neg.data();
        }
    }
    double kv(double v) const { return ascending ? v : -v; }
    // outside [min, max] (a NaN is not outside: the caller handles it)
    bool outside(double v) const { return kv(v) < key[0] || kv(v) > key[n - 1]; }
    double wrap(double v) const { return wrap_periodic(v, x[0], x[n - 1]); }
    std::int64_t piece_of(double v) const {  // one point, any position (NaN gives 0: the caller checks)
        const double k = kv(v);
        std::int64_t c;
        count_le_lanes<1>(key, n, &k, &c);
        return interval_of(c, n);
    }
};

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

// d! / (d - nu)! for the degrees of the K1 coefficients (j = 0 is degree K1 - 1)
std::vector<double> derivative_factors(std::int64_t K1, std::int64_t nu) {
    std::vector<double> f(std::max<std::int64_t>(K1, 1), 1.0);
    for (std::int64_t j = 0; j < K1; ++j) {
        const std::int64_t d = K1 - 1 - j;
        double v = 1.0;
        for (std::int64_t t = 0; t < nu; ++t) v *= static_cast<double>(d - t);
        f[j] = v;
    }
    return f;
}

// ---------------------------------------------------------------- checks

int check_x(nb::handle o, bool decreasing_ok) {
    const NDArray& a = require_ndarray(o, "x");
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64) throw nb::value_error("'x' must be 1-D float64");
    const auto* x = reinterpret_cast<const double*>(a.data);
    const std::int64_t n = a.shape[0];
    const parallel::Unlocked unlocked(n >= kParallelThreshold);  // no Python object below
    for (std::int64_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return -1;
    if (n < 2) return 0;
    bool up = true, down = true;
    for (std::int64_t i = 1; i < n; ++i) {
        up &= x[i] > x[i - 1];
        down &= x[i] < x[i - 1];
    }
    if (up) return 0;
    if (down && decreasing_ok) return 1;
    return -2;
}

bool all_finite(nb::handle o) {
    const NDArray& a = require_ndarray(o, "a");
    if (!a.c_contiguous) throw nb::value_error("'a' must be C-contiguous");
    if (a.dtype != DType::Float64 && a.dtype != DType::Complex128) throw nb::type_error("'a' must be inexact");
    const std::int64_t n = a.size() * (a.dtype == DType::Complex128 ? 2 : 1);
    const auto* p = reinterpret_cast<const double*>(a.data);
    const parallel::Unlocked unlocked(n >= kParallelThreshold);
    bool ok = true;
    for (std::int64_t i = 0; i < n; ++i) ok &= std::isfinite(p[i]);
    return ok;
}

// ---------------------------------------------------------------- evaluation

// The value of piece i's derivative of order nu at s, for column col.
inline double horner(const Coef& C, std::int64_t i, std::int64_t col, std::int64_t nu, const double* f, double s) {
    const std::int64_t top = C.K1 - nu;
    if (top <= 0) return 0.0;
    const std::int64_t plane = C.m * C.D;
    const double* p = C.c + i * C.D + col;
    if (nu == 0) {
        double r = p[0];
        for (std::int64_t j = 1; j < top; ++j) r = std::fma(r, s, p[j * plane]);
        return r;
    }
    double r = p[0] * f[0];
    for (std::int64_t j = 1; j < top; ++j) r = std::fma(r, s, p[j * plane] * f[j]);
    return r;
}

// the cubic without derivative, the common case, unrolled
inline double horner4(const double* p, std::int64_t plane, double s) {
    return std::fma(std::fma(std::fma(p[0], s, p[plane]), s, p[2 * plane]), s, p[3 * plane]);
}

void eval_range(
    const Coef& C,
    const Breaks& B,
    const double* xq,
    double* out,
    std::int64_t nu,
    const double* f,
    Extrapolate rule,
    std::int64_t begin,
    std::int64_t end
) {
    constexpr std::int64_t kBlock = 256;
    double v[kBlock], k[kBlock];
    std::int64_t cnt[kBlock];
    GallopState state;
    const std::int64_t plane = C.m * C.D;
    const bool cubic = C.K1 == 4 && nu == 0;
    for (std::int64_t b0 = begin; b0 < end; b0 += kBlock) {
        const std::int64_t mm = std::min(kBlock, end - b0);
        for (std::int64_t i = 0; i < mm; ++i) {
            const double p = xq[b0 + i];
            v[i] = rule == Extrapolate::Periodic ? B.wrap(p) : p;
            k[i] = B.kv(v[i]);
        }
        count_le_block(B.key, B.n, k, mm, cnt, state);
        for (std::int64_t i = 0; i < mm; ++i) {
            double* row = out + (b0 + i) * C.D;
            const double p = v[i];
            if (p != p || (rule == Extrapolate::Nan && B.outside(p))) {
                std::fill(row, row + C.D, kNaN);
                continue;
            }
            const std::int64_t j = interval_of(cnt[i], B.n);
            const double s = p - B.x[j];
            if (cubic) {
                const double* base = C.c + j * C.D;
                for (std::int64_t col = 0; col < C.D; ++col) row[col] = horner4(base + col, plane, s);
            } else {
                for (std::int64_t col = 0; col < C.D; ++col) row[col] = horner(C, j, col, nu, f, s);
            }
        }
    }
}

nb::object ppoly_eval(nb::handle co, nb::handle xo, nb::handle qo, std::int64_t nu, int rule) {
    const Coef C = coef_arg(co);
    const Breaks B(xo, C.m);
    const NDArray& q = require_ndarray(qo, "xq");
    if (q.ndim != 1 || !q.c_contiguous || q.dtype != DType::Float64)
        throw nb::value_error("'xq' must be a C-contiguous 1-D float64 array");
    if (nu < 0) throw nb::value_error("'nu' must be nonnegative");
    if (rule < 0 || rule > 2) throw nb::value_error("'rule' must be 0, 1 or 2");
    const std::int64_t nq = q.shape[0];
    const std::int64_t shape[2] = {nq, C.ncol};
    NDArray out = new_array(C.dtype, shape);
    if (nq == 0 || C.D == 0) return wrap(std::move(out));
    const std::vector<double> f = derivative_factors(C.K1, nu);
    const auto* xq = reinterpret_cast<const double*>(q.data);
    auto* o = reinterpret_cast<double*>(out.data);
    const std::int64_t log_n = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::log2(B.n + 1)));
    const std::int64_t work = nq * (2 * log_n + 2 * C.K1 * C.D);  // as sd.interp's (decision IP102)
    if (work >= kParallelThreshold && parallel::available()) {
        const std::int64_t per = std::max<std::int64_t>(256, 4096 / std::max<std::int64_t>(1, C.K1 * C.D));
        const std::int64_t tasks = (nq + per - 1) / per;
        parallel::parallel_for(
            tasks,
            [&](std::int64_t t) {
                eval_range(
                    C, B, xq, o, nu, f.data(), static_cast<Extrapolate>(rule), t * per, std::min(nq, (t + 1) * per)
                );
            },
            work_ns(work / tasks)
        );
    } else {
        const parallel::Unlocked unlocked(work >= kParallelThreshold);
        eval_range(C, B, xq, o, nu, f.data(), static_cast<Extrapolate>(rule), 0, nq);
    }
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- antiderivative

constexpr std::int64_t kPrefixBlock = 4096;  // pieces per block of the continuity prefix sums (by length only)

double binom(std::int64_t d, std::int64_t q) {
    double v = 1.0;
    for (std::int64_t t = 1; t <= q; ++t) v = v * static_cast<double>(d - q + t) / static_cast<double>(t);
    return v;
}

nb::object ppoly_antiderivative(nb::handle co, nb::handle xo, std::int64_t nu) {
    const Coef C = coef_arg(co);
    const Breaks B(xo, C.m);
    if (nu < 1) throw nb::value_error("'nu' must be positive");
    const std::int64_t K2 = C.K1 + nu, m = C.m, Dc = C.D;
    const std::int64_t shape[3] = {K2, m, C.ncol};
    NDArray out = new_array(C.dtype, shape, true);
    auto* c2 = reinterpret_cast<double*>(out.data);
    const std::int64_t plane = m * Dc;
    // the shifted coefficients: degree d -> d + nu, divided by (d + 1) ... (d + nu)
    std::vector<double> div(C.K1);
    for (std::int64_t j = 0; j < C.K1; ++j) {
        const std::int64_t d = C.K1 - 1 - j;
        double v = 1.0;
        for (std::int64_t t = 1; t <= nu; ++t) v *= static_cast<double>(d + t);
        div[j] = v;
    }
    split_range(m, m * Dc * C.K1, 1024, [&](std::int64_t i0, std::int64_t i1) {
        for (std::int64_t j = 0; j < C.K1; ++j)
            for (std::int64_t i = i0; i < i1; ++i)
                for (std::int64_t col = 0; col < Dc; ++col)
                    c2[j * plane + i * Dc + col] = C.c[j * plane + i * Dc + col] / div[j];
    });
    if (m < 2 || Dc == 0) return wrap(std::move(out));
    // the constants, degree q = nu - 1 down to 0 (row K2 - 1 - q); w[i] for piece i's right end
    std::vector<double> w((m - 1) * Dc);
    const std::int64_t Dg = K2 - 1;  // the antiderivative's degree
    const std::int64_t nblocks = (m - 1 + kPrefixBlock - 1) / kPrefixBlock;
    std::vector<double> totals(nblocks * Dc);
    for (std::int64_t q = nu - 1; q >= 0; --q) {
        const std::int64_t row = Dg - q;
        std::vector<double> bc(Dg - q + 1);
        for (std::int64_t t = 1; t <= Dg - q; ++t) bc[t] = binom(q + t, q);
        // w[i] = sum_{t >= 1} c2[degree q + t][i] binom(q + t, q) h^t, Horner in h
        split_range(m - 1, (m - 1) * Dc * (Dg - q) * 2, 1024, [&](std::int64_t i0, std::int64_t i1) {
            for (std::int64_t i = i0; i < i1; ++i) {
                const double h = B.x[i + 1] - B.x[i];
                for (std::int64_t col = 0; col < Dc; ++col) {
                    // degree q + t is row Dg - q - t
                    double r = c2[(row - (Dg - q)) * plane + i * Dc + col] * bc[Dg - q];
                    for (std::int64_t t = Dg - q - 1; t >= 1; --t)
                        r = std::fma(r, h, c2[(row - t) * plane + i * Dc + col] * bc[t]);
                    w[i * Dc + col] = r * h;
                }
            }
        });
        // c_q[i + 1] = w[0] + ... + w[i]: within each block a running sum from 0, then the blocks' offsets
        split_range(nblocks, (m - 1) * Dc, 1, [&](std::int64_t b0, std::int64_t b1) {
            for (std::int64_t b = b0; b < b1; ++b) {
                const std::int64_t i0 = b * kPrefixBlock, i1 = std::min(m - 1, i0 + kPrefixBlock);
                for (std::int64_t col = 0; col < Dc; ++col) {
                    double acc = 0.0;
                    for (std::int64_t i = i0; i < i1; ++i) {
                        acc = i == i0 ? w[i * Dc + col] : acc + w[i * Dc + col];
                        c2[row * plane + (i + 1) * Dc + col] = acc;
                    }
                    totals[b * Dc + col] = acc;
                }
            }
        });
        if (nblocks > 1) {
            std::vector<double> offset(nblocks * Dc, 0.0);
            for (std::int64_t b = 1; b < nblocks; ++b)
                for (std::int64_t col = 0; col < Dc; ++col)
                    offset[b * Dc + col] =
                        b == 1 ? totals[col] : offset[(b - 1) * Dc + col] + totals[(b - 1) * Dc + col];
            split_range(nblocks - 1, (m - 1) * Dc, 1, [&](std::int64_t b0, std::int64_t b1) {
                for (std::int64_t b = b0 + 1; b < b1 + 1; ++b) {
                    const std::int64_t i0 = b * kPrefixBlock, i1 = std::min(m - 1, i0 + kPrefixBlock);
                    for (std::int64_t i = i0; i < i1; ++i)
                        for (std::int64_t col = 0; col < Dc; ++col)
                            c2[row * plane + (i + 1) * Dc + col] =
                                offset[b * Dc + col] + c2[row * plane + (i + 1) * Dc + col];
                }
            });
        }
    }
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- integrate

// piece i's local antiderivative F(s) = sum_d c_d s^(d + 1) / (d + 1) for column col
inline double local_integral(const Coef& C, std::int64_t i, std::int64_t col, double s) {
    const std::int64_t plane = C.m * C.D;
    const double* p = C.c + i * C.D + col;
    double r = p[0] / static_cast<double>(C.K1);
    for (std::int64_t j = 1; j < C.K1; ++j) r = std::fma(r, s, p[j * plane] / static_cast<double>(C.K1 - j));
    return r * s;
}

nb::object ppoly_integrate(nb::handle co, nb::handle xo, double a, double b, bool extend) {
    const Coef C = coef_arg(co);
    const Breaks B(xo, C.m);
    const std::int64_t shape[1] = {C.ncol};
    NDArray out = new_array(C.dtype, shape);
    auto* o = reinterpret_cast<double*>(out.data);
    if (C.D == 0) return wrap(std::move(out));
    if (a != a || b != b || (!extend && (B.outside(a) || B.outside(b)))) {
        std::fill(o, o + C.D, kNaN);
        return wrap(std::move(out));
    }
    const std::int64_t ia = B.piece_of(a), ib = B.piece_of(b);
    // the pieces from first to last in index order; for decreasing breakpoints b (the larger) comes first
    const std::int64_t first = std::min(ia, ib), last = std::max(ia, ib), count = last - first + 1;
    auto term = [&](std::int64_t i, std::int64_t col) {
        const double xi = B.x[i];
        double lo, hi;  // local s of the lower and the upper end of the integral over piece i
        if (B.ascending) {
            lo = i == ia ? a - xi : 0.0;
            hi = i == ib ? b - xi : B.x[i + 1] - xi;
        } else {
            lo = i == ia ? a - xi : B.x[i + 1] - xi;
            hi = i == ib ? b - xi : 0.0;
        }
        const double fl = lo == 0.0 ? 0.0 : local_integral(C, i, col, lo);
        const double fh = hi == 0.0 ? 0.0 : local_integral(C, i, col, hi);
        return fh - fl;
    };
    constexpr std::int64_t kTask = 1 << 14;
    const pairwise::Leaves leaves = pairwise::plan(count, kTask, false);
    const std::int64_t L = static_cast<std::int64_t>(leaves.size());
    std::vector<double> parts(L * C.D);
    split_range(L * C.D, count * C.D * C.K1 * 2, 1, [&](std::int64_t t0, std::int64_t t1) {
        for (std::int64_t t = t0; t < t1; ++t) {
            const std::int64_t col = t / L, leaf = t % L;
            const auto [lb, ln] = leaves[leaf];
            parts[col * L + leaf] = pairwise::tree(lb, ln, [&](std::int64_t s0, std::int64_t sn) {
                return pairwise::leaf(sn, [&](std::int64_t r) { return term(first + s0 + r, col); });
            });
        }
    });
    for (std::int64_t col = 0; col < C.D; ++col) {
        const std::vector<double> mine(parts.begin() + col * L, parts.begin() + (col + 1) * L);
        o[col] = pairwise::combine(count, kTask, mine);
    }
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- roots

// one polynomial (highest degree first) in a local variable s = fl(t - o)
struct Poly {
    std::vector<double> a;
    int deg() const { return static_cast<int>(a.size()) - 1; }
};

inline void two_sum(double x, double y, double& s, double& e) {
    s = x + y;
    const double z = s - x;
    e = (x - (s - z)) + (y - z);
}

// p(s) by compensated Horner (Graillat, Langlois and Louvet): the value as if computed in twice the precision
inline double eval_comp(const Poly& P, double s) {
    double r = P.a[0], c = 0.0;
    for (std::size_t j = 1; j < P.a.size(); ++j) {
        const double p = r * s;
        const double pe = std::fma(r, s, -p);
        double sum, se;
        two_sum(p, P.a[j], sum, se);
        r = sum;
        c = std::fma(c, s, pe + se);
    }
    return r + c;
}

inline double eval_plain(const Poly& P, double s) {
    double r = P.a[0];
    for (std::size_t j = 1; j < P.a.size(); ++j) r = std::fma(r, s, P.a[j]);
    return r;
}

// the evaluation contract's scale at s: gamma_{2d+2} sum_j |a_j| |s|^(d - j) (docs/design/interpolate.md §4)
inline double eval_bound(const Poly& P, double s) {
    const int d = P.deg();
    double r = std::fabs(P.a[0]);
    const double as = std::fabs(s);
    for (std::size_t j = 1; j < P.a.size(); ++j) r = r * as + std::fabs(P.a[j]);
    const double g = (2 * d + 2) * kU / (1 - (2 * d + 2) * kU);
    return 1.01 * g * r + (2 * d + 2) * 0x1p-1074;
}

// the sign of p at t (s = fl(t - o)); an overflowed value takes the leading term's sign
inline int sign_at(const Poly& P, double t, double o) {
    const double s = t - o;
    double v = eval_comp(P, s);
    if (!std::isfinite(v)) {
        v = eval_plain(P, s);
        if (v != v) {
            const int d = P.deg();
            const double lead = P.a[0] * ((d % 2 == 1 && s < 0) ? -1.0 : 1.0);
            return (lead > 0) - (lead < 0);
        }
    }
    return (v > 0) - (v < 0);
}

// the doubles in order as integers: key(-0) == key(+0) == 0
inline std::int64_t okey(double t) {
    const std::int64_t b = std::bit_cast<std::int64_t>(t);
    return b >= 0 ? b : -(b & 0x7fffffffffffffffLL);
}
inline double ofrom(std::int64_t k) {
    return k >= 0 ? std::bit_cast<double>(k) : std::bit_cast<double>((-k) | std::numeric_limits<std::int64_t>::min());
}

// the root in (lo, hi) of a section whose ends have the signs slo, shi (opposite, nonzero)
double bisect(const Poly& P, double o, double lo, double hi, int slo) {
    std::int64_t kl = okey(lo), kh = okey(hi);
    while (static_cast<std::uint64_t>(kh) - static_cast<std::uint64_t>(kl) > 1) {  // at most 64 steps
        const std::int64_t km =
            kl + static_cast<std::int64_t>((static_cast<std::uint64_t>(kh) - static_cast<std::uint64_t>(kl)) / 2);
        const double t = ofrom(km);
        const int s = sign_at(P, t, o);
        if (s == 0) return t;
        if (s == slo) kl = km;
        else kh = km;
    }
    const double tl = ofrom(kl), th = ofrom(kh);
    const double vl = std::fabs(eval_comp(P, tl - o)), vh = std::fabs(eval_comp(P, th - o));
    return vh < vl ? th : tl;
}

void find_roots(const Poly& P, double o, double lo, double hi, std::vector<double>& out);

// the critical points of P in the local range [slo, shi]: the derivative's roots, sorted
void critical_points(const Poly& P, double slo, double shi, std::vector<double>& out) {
    const int d = P.deg();
    if (d < 2) return;
    Poly Q;
    Q.a.resize(d);
    for (int j = 0; j < d; ++j) Q.a[j] = P.a[j] * static_cast<double>(d - j);
    if (d == 2) {
        const double r = -Q.a[1] / Q.a[0];
        if (r > slo && r < shi) out.push_back(r);
        return;
    }
    if (d == 3) {  // a quadratic: the cancellation-free formula
        const double A = Q.a[0], Bq = Q.a[1], Cq = Q.a[2];
        const double disc = Bq * Bq - 4.0 * A * Cq;
        if (disc < 0) return;
        const double sq = std::sqrt(disc);
        const double qq = -0.5 * (Bq + (Bq < 0 ? -sq : sq));
        double r1, r2;
        if (qq == 0.0) r1 = r2 = 0.0;
        else r1 = qq / A, r2 = Cq / qq;
        if (r1 > r2) std::swap(r1, r2);
        if (r1 > slo && r1 < shi) out.push_back(r1);
        if (r2 > slo && r2 < shi && r2 != r1) out.push_back(r2);
        return;
    }
    find_roots(Q, 0.0, slo, shi, out);
    out.erase(std::remove_if(out.begin(), out.end(), [&](double r) { return !(r > slo && r < shi); }), out.end());
}

// The roots of P(fl(t - o)) for t in [lo, hi], ascending, without duplicates (P's degree >= 1, leading
// coefficient nonzero).
void find_roots(const Poly& P, double o, double lo, double hi, std::vector<double>& out) {
    std::vector<double> crit;
    critical_points(P, lo - o, hi - o, crit);
    std::vector<double> E;
    E.reserve(crit.size() + 2);
    E.push_back(lo);
    for (double sc : crit) {
        const double t = o + sc;
        if (t > E.back() && t < hi) E.push_back(t);
    }
    if (hi > E.back()) E.push_back(hi);
    const std::size_t ne = E.size();
    std::vector<int> sg(ne);
    for (std::size_t j = 0; j < ne; ++j) sg[j] = sign_at(P, E[j], o);
    const std::size_t start = out.size();
    for (std::size_t j = 0; j < ne; ++j) {
        if (sg[j] == 0) out.push_back(E[j]);
        else if (j > 0 && j + 1 < ne && sg[j - 1] * sg[j] >= 0 && sg[j] * sg[j + 1] >= 0 && sg[j - 1] != 0 &&
                 sg[j + 1] != 0) {
            // an extremum that does not cross: a double root when its value is zero within the contract
            const double s = E[j] - o;
            if (std::fabs(eval_comp(P, s)) <= eval_bound(P, s)) out.push_back(E[j]);
        }
        if (j + 1 < ne && sg[j] * sg[j + 1] < 0) out.push_back(bisect(P, o, E[j], E[j + 1], sg[j]));
    }
    std::sort(out.begin() + start, out.end());
    out.erase(std::unique(out.begin() + start, out.end()), out.end());
}

// A bound past which a polynomial has no roots (Cauchy's, doubled), as a local s
double root_bound(const Poly& P) {
    double mx = 0.0;
    for (std::size_t j = 1; j < P.a.size(); ++j) mx = std::fmax(mx, std::fabs(P.a[j] / P.a[0]));
    const double b = 2.0 * (1.0 + mx);
    return std::isfinite(b) ? std::fmin(b, 0x1p1000) : 0x1p1000;
}

struct PieceRoots {
    std::vector<double> r;
    bool zero = false;
    bool sign_change = false;  // across the breakpoint at the piece's left end
};

nb::object ppoly_roots(nb::handle co, nb::handle xo, double y, bool discontinuity, bool extend) {
    const Coef C = coef_arg(co);
    if (C.dtype != DType::Float64) throw nb::value_error("root finding is only for real-valued polynomials");
    const Breaks B(xo, C.m);
    const std::int64_t m = C.m;
    nb::list result;
    std::vector<double> f(1, 1.0);
    for (std::int64_t col = 0; col < C.ncol; ++col) {
        std::vector<PieceRoots> pr(m);
        split_range(m, m * C.K1 * 200, 64, [&](std::int64_t i0, std::int64_t i1) {
            for (std::int64_t i = i0; i < i1; ++i) {
                PieceRoots& R = pr[i];
                if (i > 0 && discontinuity) {
                    const double va = horner(C, i - 1, col, 0, f.data(), B.x[i] - B.x[i - 1]) - y;
                    const double vb = C.at(C.K1 - 1, i, col) - y;
                    R.sign_change = (va < 0 && vb > 0) || (va > 0 && vb < 0);
                }
                Poly P;
                std::int64_t j0 = 0;
                while (j0 < C.K1 && C.at(j0, i, col) == 0.0 && j0 < C.K1 - 1) ++j0;
                for (std::int64_t j = j0; j < C.K1; ++j) P.a.push_back(C.at(j, i, col));
                P.a.back() = P.a.back() - y;
                if (P.deg() == 0) {
                    if (P.a[0] == 0.0) R.zero = true;
                    continue;
                }
                if (P.a[0] == 0.0) {  // only the constant was nonzero before subtracting y
                    while (P.a.size() > 1 && P.a[0] == 0.0) P.a.erase(P.a.begin());
                    if (P.deg() == 0) {
                        if (P.a[0] == 0.0) R.zero = true;
                        continue;
                    }
                }
                const double o = B.x[i];
                const double far = root_bound(P);
                // the piece's range in t, extended outward at the ends when extrapolating
                double lo, hi;
                const double xa = B.x[i], xb = B.x[i + 1];
                if (B.ascending) {
                    lo = xa, hi = xb;
                    if (extend && i == 0) lo = std::fmin(xa, std::fmax(o - far, -std::numeric_limits<double>::max()));
                    if (extend && i == m - 1)
                        hi = std::fmax(xb, std::fmin(o + far, std::numeric_limits<double>::max()));
                } else {
                    lo = xb, hi = xa;
                    if (extend && i == 0) hi = std::fmax(xa, std::fmin(o + far, std::numeric_limits<double>::max()));
                    if (extend && i == m - 1)
                        lo = std::fmin(xb, std::fmax(o - far, -std::numeric_limits<double>::max()));
                }
                find_roots(P, o, lo, hi, R.r);
            }
        });
        std::vector<double> roots;
        double last = kNaN;
        for (std::int64_t i = 0; i < m; ++i) {
            const PieceRoots& R = pr[i];
            if (R.sign_change && B.x[i] != last) {
                last = B.x[i];
                roots.push_back(last);
            }
            if (R.zero) {
                roots.push_back(B.x[i]);
                roots.push_back(kNaN);
                last = kNaN;
                continue;
            }
            for (double r : R.r)
                if (r != last) {
                    last = r;
                    roots.push_back(r);
                }
        }
        const std::int64_t shape[1] = {static_cast<std::int64_t>(roots.size())};
        NDArray arr = new_array(DType::Float64, shape);
        if (!roots.empty()) std::memcpy(arr.data, roots.data(), roots.size() * sizeof(double));
        result.append(wrap(std::move(arr)));
    }
    return result;
}

}  // namespace

void bind_ppoly(nb::module_& m) {
    m.def("check_x", &check_x, nb::arg("x").none(), nb::arg("decreasing_ok"));
    m.def("all_finite", &all_finite, nb::arg("a").none());
    m.def(
        "ppoly_eval", &ppoly_eval, nb::arg("c").none(), nb::arg("x").none(), nb::arg("xq").none(), nb::arg("nu"),
        nb::arg("rule")
    );  // fmt: skip
    m.def("ppoly_antiderivative", &ppoly_antiderivative, nb::arg("c").none(), nb::arg("x").none(), nb::arg("nu"));
    m.def(
        "ppoly_integrate", &ppoly_integrate, nb::arg("c").none(), nb::arg("x").none(), nb::arg("a"), nb::arg("b"),
        nb::arg("extend")
    );  // fmt: skip
    m.def(
        "ppoly_roots", &ppoly_roots, nb::arg("c").none(), nb::arg("x").none(), nb::arg("y"), nb::arg("discontinuity"),
        nb::arg("extend")
    );  // fmt: skip
}

}  // namespace sd::interpolate
