// Savitzky-Golay filters (docs/design/smooth.md §3.7, §4.6; decisions SM32, SM70-SM76): the exact least-squares
// coefficients from the orthonormal discrete Chebyshev (Gram) polynomials in double-double, and the filter along
// one axis of an n-D array under scipy's five modes.
//
// _core.smooth.savgol_coeffs(window_length, polyorder, deriv, delta, pos) -> (window_length,) float64
//     the coefficients in window order (scipy's use='dot'); pos -1 is the centre ((N - 1) / 2: half a step off
//     for an even window), otherwise 0 <= pos < N.
// _core.smooth.savgol_filter(x, window_length, polyorder, deriv, delta, axis, mode, cval) -> array of x's shape
//     x a C-contiguous float64 or complex128 array (ndim >= 1, axis in [0, ndim)), mode 0 'interp' (the axis at
//     least window_length long), 1 'mirror', 2 'nearest', 3 'constant' (cval; 0 for the imaginary parts), 4 'wrap'.
//     The Python layer checks the arguments and the work space's size (stardust/smooth/_savgol.py).
//
// Coefficients (§3.7). With s_i = i - (N - 1)/2 at the window's points i = 0 .. N - 1, the polynomials q_0 .. q_d
// orthonormal on them and the position p, the least-squares polynomial's nu-th derivative at p is
//     y = sum_i h_i x_i,   h_i = delta^-nu sum_{j = nu .. d} q_j(s_i) q_j^(nu)(s_p).
// The node values q_j(s_i) come from the recurrence along i (Mukundan 2004): the closed form at the window's
// first point and its neighbour, the three-term recurrence in i up to the middle, the symmetry
// q_j(s_{N-1-i}) = (-1)^j q_j(s_i) for the rest. The recurrence in the degree j, evaluated at the points, loses
// everything once d passes ~N/4 at 106 bits (the window's ends sit in the high degrees' exponentially small
// region); the one along i grows from the small end values towards the middle and kept every (j, i) within
// 6e-14 u of the exact values (N <= 1001, every degree; decision SM70). q_j^(nu)(s_p) comes from the degree
// recurrence at the one point p, whose derivative values stay within ~1e-13 u of their running maximum at every
// p, which bounds the coefficients' error (||h||_2 = ||c||_2); for nu = 0 at a whole-number p it is a node
// value. All of it in double-double; each coefficient is one rounding of the double-double sum (decision SM32).
//
// The filter (§3.7). An output whose window lies inside the axis is the fma chain
//     acc = 0; acc = fma(h_j, x_{i - first + j}, acc) for j = 0 .. N - 1      (first = N/2, or N/2 - 1 for even N)
// with the centre coefficients rounded to float64; NEON computes 16 outputs (inner == 1) or 16 columns (axis not
// last) at once with that chain per output, so every output has the same bits whatever the layout, the block or the
// thread. Outside the axis the modes read scipy.ndimage's extension (mirror without repeating the end point,
// nearest, constant, wrap) through folded indices, no padded copy. 'interp' replaces the first and the last N/2
// outputs by the polynomial fitted to the first and the last window: the projections a_j = sum_i q_j(s_i) x_i of
// the window in double-double, then y_p = delta^-nu sum_j q_j^(nu)(s_p) a_j rounded once (decision SM72; scipy
// fits with polyfit on every line).
//
// Every output is a function of its window and the coefficients alone: the results have the same bits for every
// thread count (decision SM3). The work is cut by shape alone (rows, columns, lines).

#include "array/ndarray.hpp"
#include "array/parallel_limits.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "smooth/smooth.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::smooth {

namespace nb = nanobind;

namespace {

// ---------------------------------------------------------------- double-double

struct DD {
    double hi = 0.0, lo = 0.0;
};

inline DD two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
inline DD quick_two_sum(double a, double b) {  // |a| >= |b| or a == 0
    const double s = a + b;
    return {s, b - (s - a)};
}
inline DD two_prod(double a, double b) {
    const double p = a * b;
    return {p, std::fma(a, b, -p)};
}
inline DD add(DD a, DD b) {
    const DD s = two_sum(a.hi, b.hi);
    const DD t = two_sum(a.lo, b.lo);
    DD r = quick_two_sum(s.hi, s.lo + t.hi);
    return quick_two_sum(r.hi, r.lo + t.lo);
}
inline DD neg(DD a) { return {-a.hi, -a.lo}; }
inline DD sub(DD a, DD b) { return add(a, neg(b)); }
inline DD mul(DD a, DD b) {
    DD p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}
inline DD mul(DD a, double b) {
    DD p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    return quick_two_sum(p.hi, p.lo);
}
inline DD div(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    const DD p = mul(b, q1);
    const DD r = sub(a, p);
    const double q2 = r.hi / b.hi;
    const DD p2 = mul(b, q2);
    const DD r2 = sub(r, p2);
    return add(quick_two_sum(q1, q2), DD{r2.hi / b.hi, 0.0});
}
inline DD sqrt_dd(DD a) {
    if (a.hi <= 0.0) return {0.0, 0.0};
    const double s = std::sqrt(a.hi);
    const DD r = sub(a, two_prod(s, s));
    return quick_two_sum(s, r.hi / (2.0 * s));
}
inline DD ldexp_dd(DD a, std::int64_t e) {
    const int k = static_cast<int>(std::clamp<std::int64_t>(e, -4000, 4000));
    return {std::ldexp(a.hi, k), std::ldexp(a.lo, k)};
}
inline DD exact(double v) { return {v, 0.0}; }  // a whole number below 2^53
// The product of two whole numbers below 2^53, exactly.
inline DD iprod(double a, double b) { return two_prod(a, b); }

// ---------------------------------------------------------------- the Gram polynomials

struct Gram {
    std::int64_t N = 0, d = 0;
    std::vector<DD> T;    // (N, d + 1) row-major: T[i (d + 1) + j] = q_j(s_i)
    std::vector<DD> sb;   // sqrt(beta_k), k = 1 .. d (sb[0] = 0)
    std::vector<DD> isb;  // 1 / sqrt(beta_k)

    DD node(std::int64_t i, std::int64_t j) const { return T[static_cast<size_t>(i * (d + 1) + j)]; }
};

// Gram(N, d) with the node values when `nodes` (the only O(N d) part).
Gram make_gram(std::int64_t N, std::int64_t d, bool nodes) {
    Gram G;
    G.N = N;
    G.d = d;
    const double Nd = static_cast<double>(N);
    G.sb.assign(static_cast<size_t>(d + 1), DD{});
    G.isb.assign(static_cast<size_t>(d + 1), DD{});
    for (std::int64_t k = 1; k <= d; ++k) {
        // beta_k = k^2 (N^2 - k^2) / (4 (4 k^2 - 1)): sqrt(beta_k) = (k / 2) sqrt((N - k)(N + k) / (4 k^2 - 1))
        const double kd = static_cast<double>(k);
        const DD ratio = div(iprod(Nd - kd, Nd + kd), iprod(2 * kd - 1, 2 * kd + 1));
        G.sb[static_cast<size_t>(k)] = mul(sqrt_dd(ratio), 0.5 * kd);
        G.isb[static_cast<size_t>(k)] = div(exact(1.0), G.sb[static_cast<size_t>(k)]);
    }
    if (!nodes) return G;
    const std::int64_t D1 = d + 1;
    G.T.assign(static_cast<size_t>(N * D1), DD{});
    if (N == 1) {
        G.T[0] = exact(1.0);
        return G;
    }
    const std::int64_t half = (N + 1) / 2;  // rows 0 .. half - 1 by the recurrence, the rest by symmetry
    std::vector<DD> anum(static_cast<size_t>(half)), inv(static_cast<size_t>(half)), g2(static_cast<size_t>(half));
    for (std::int64_t x = 2; x < half; ++x) {
        // t(x) = g1 t(x - 1) + g2 t(x - 2), g1 = ((2x - 1)(N + 1 - x) - x - j(j + 1)) / (x (N - x)),
        // g2 = -(x - 1)(N + 1 - x) / (x (N - x))
        const double xd = static_cast<double>(x);
        inv[static_cast<size_t>(x)] = div(exact(1.0), iprod(xd, Nd - xd));
        anum[static_cast<size_t>(x)] = sub(iprod(2 * xd - 1, Nd + 1 - xd), exact(xd));
        g2[static_cast<size_t>(x)] = neg(mul(iprod(xd - 1, Nd + 1 - xd), inv[static_cast<size_t>(x)]));
    }
    // t_j(0) = m 2^e: t_0(0) = 1 / sqrt(N), t_j(0) = -t_{j-1}(0) sqrt((N - j)(2j + 1) / ((N + j)(2j - 1)))
    DD m = div(exact(1.0), sqrt_dd(exact(Nd)));
    std::int64_t e = 0;
    const std::int64_t centre = (N % 2 == 1) ? (N - 1) / 2 : -1;
    for (std::int64_t j = 0; j <= d; ++j) {
        const double jd = static_cast<double>(j);
        if (j > 0) {
            const DD r = div(iprod(Nd - jd, 2 * jd + 1), iprod(Nd + jd, 2 * jd - 1));
            m = neg(mul(m, sqrt_dd(r)));
            int k = 0;
            std::frexp(m.hi, &k);
            m = ldexp_dd(m, -k);
            e += k;
        }
        const double jj = jd * (jd + 1);
        DD u0 = m;
        DD u1 = mul(m, div(exact(Nd - 1 - jj), exact(Nd - 1)));
        std::int64_t E = e;
        G.T[static_cast<size_t>(j)] = ldexp_dd(u0, E);
        if (half > 1) G.T[static_cast<size_t>(D1 + j)] = ldexp_dd(u1, E);
        for (std::int64_t x = 2; x < half; ++x) {
            const DD g1 = mul(sub(anum[static_cast<size_t>(x)], exact(jj)), inv[static_cast<size_t>(x)]);
            DD u2 = add(mul(g1, u1), mul(g2[static_cast<size_t>(x)], u0));
            if (std::fabs(u2.hi) > 0x1p600) {  // the mantissa grows by up to 2^-e: keep it in range
                u2 = ldexp_dd(u2, -600);
                u1 = ldexp_dd(u1, -600);
                E += 600;
            }
            G.T[static_cast<size_t>(x * D1 + j)] = ldexp_dd(u2, E);
            u0 = u1;
            u1 = u2;
        }
        if (centre >= 0 && j % 2 == 1) G.T[static_cast<size_t>(centre * D1 + j)] = DD{};  // odd about the middle
        const bool odd = j % 2 == 1;
        for (std::int64_t x = half; x < N; ++x) {
            const DD v = G.T[static_cast<size_t>((N - 1 - x) * D1 + j)];
            G.T[static_cast<size_t>(x * D1 + j)] = odd ? neg(v) : v;
        }
    }
    return G;
}

// q_j^(nu)(s) for j = 0 .. d by the degree recurrence at the point s (exact):
//     sqrt(beta_{k+1}) q_{k+1}^(v) = s q_k^(v) + v q_k^(v-1) - sqrt(beta_k) q_{k-1}^(v).
std::vector<DD> derivatives_at(const Gram& G, double s, std::int64_t nu) {
    const std::int64_t d = G.d, V = nu + 1;
    std::vector<DD> out(static_cast<size_t>(d + 1));
    std::vector<DD> prev(static_cast<size_t>(V)), cur(static_cast<size_t>(V)), next(static_cast<size_t>(V));
    cur[0] = div(exact(1.0), sqrt_dd(exact(static_cast<double>(G.N))));
    out[0] = cur[static_cast<size_t>(nu)];
    for (std::int64_t k = 0; k < d; ++k) {
        for (std::int64_t v = 0; v < V; ++v) {
            DD t = mul(cur[static_cast<size_t>(v)], s);
            if (v > 0) t = add(t, mul(cur[static_cast<size_t>(v - 1)], static_cast<double>(v)));
            if (k > 0) t = sub(t, mul(G.sb[static_cast<size_t>(k)], prev[static_cast<size_t>(v)]));
            next[static_cast<size_t>(v)] = mul(t, G.isb[static_cast<size_t>(k + 1)]);
        }
        std::swap(prev, cur);
        std::swap(cur, next);
        out[static_cast<size_t>(k + 1)] = cur[static_cast<size_t>(nu)];
    }
    return out;
}

// 1 / delta^nu as a double-double mantissa and a power of two (no overflow or underflow on the way).
struct Scale {
    DD m{1.0, 0.0};
    std::int64_t e = 0;
};
Scale inverse_power(double delta, std::int64_t nu) {
    Scale S;
    if (nu == 0) return S;
    int k = 0;
    const double f = std::frexp(delta, &k);  // delta = f 2^k, |f| in [1/2, 1)
    DD p{1.0, 0.0};
    std::int64_t pe = 0;
    for (std::int64_t v = 0; v < nu; ++v) {
        p = mul(p, f);
        int kk = 0;
        std::frexp(p.hi, &kk);
        p = ldexp_dd(p, -kk);
        pe += kk;
    }
    S.m = div(exact(1.0), p);
    S.e = -(pe + nu * static_cast<std::int64_t>(k));
    return S;
}
inline double finish(DD v, const Scale& S) {
    const DD r = mul(v, S.m);
    return std::ldexp(r.hi, static_cast<int>(std::clamp<std::int64_t>(S.e, -4000, 4000)));
}

// (h, l) += (ah, al) (bh, bl) in double-double: the product's error by fma, one two-sum on the high parts and the
// low parts added in float64 (~2^-104 of the terms per step); elementwise, so the loops over lines vectorize with
// the same bits.
inline void accumulate(double& h, double& l, double ah, double al, double bh, double bl) {
    const double p = ah * bh;
    const double pe = std::fma(ah, bh, -p) + (ah * bl + al * bh);
    const double s = h + p;
    const double bb = s - h;
    const double se = (h - (s - bb)) + (p - bb);
    const double lo = l + (se + pe);
    const double r = s + lo;
    l = lo - (r - s);
    h = r;
}

// The coefficient factors c_j = q_j^(nu)(s_p), j = 0 .. d (zero below nu), at position pos (-1: the centre).
std::vector<DD> factors(const Gram& G, std::int64_t pos, std::int64_t nu) {
    if (nu == 0 && pos >= 0) {
        std::vector<DD> c(static_cast<size_t>(G.d + 1));
        for (std::int64_t j = 0; j <= G.d; ++j) c[static_cast<size_t>(j)] = G.node(pos, j);
        return c;
    }
    const double s = pos < 0 ? 0.0 : static_cast<double>(pos) - 0.5 * static_cast<double>(G.N - 1);
    return derivatives_at(G, s, nu);
}

// The coefficients in window order at one position, rounded once each.
void coefficients(const Gram& G, std::int64_t pos, std::int64_t nu, const Scale& S, double* out) {
    const std::vector<DD> c = factors(G, pos, nu);
    for (std::int64_t i = 0; i < G.N; ++i) {
        DD acc{};
        for (std::int64_t j = nu; j <= G.d; ++j) acc = add(acc, mul(G.node(i, j), c[static_cast<size_t>(j)]));
        out[i] = finish(acc, S);
    }
}

nb::object savgol_coeffs(std::int64_t N, std::int64_t d, std::int64_t nu, double delta, std::int64_t pos) {
    if (N < 1 || d < 0 || d >= N || nu < 0 || pos < -1 || pos >= N)
        throw nb::value_error("needs 0 <= polyorder < window_length, deriv >= 0 and -1 <= pos < window_length");
    const std::int64_t shape[1] = {N};
    NDArray out = new_array(DType::Float64, shape, true);
    auto* o = reinterpret_cast<double*>(out.data);
    if (nu > d) return wrap(std::move(out));  // zeros (scipy)
    {
        const parallel::Unlocked unlocked(N * (d + 1) >= kParallelThreshold);
        const Gram G = make_gram(N, d, true);
        coefficients(G, pos, nu, inverse_power(delta, nu), o);
    }
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- the filter

enum Mode : int { kInterp = 0, kMirror = 1, kNearest = 2, kConstant = 3, kWrap = 4 };

struct Plan {
    const double* x = nullptr;
    double* y = nullptr;
    std::int64_t outer = 0, n = 0, inner = 0;  // in doubles: a complex value is two lanes
    std::int64_t N = 0, first = 0, half = 0, nu = 0, d = 0;
    int mode = kInterp;
    const double* h = nullptr;  // the centre coefficients in window order (N)
    const double* pad = nullptr;  // 'constant': the value of each lane (inner)
    std::int64_t ilo = 0, ihi = 0;  // outputs [ilo, ihi) read inside the axis alone
    // 'interp'
    const Gram* G = nullptr;
    const std::vector<DD>* C = nullptr;  // (half, d + 1): c_j at the positions 0 .. half - 1
    Scale S;
    bool zero = false;  // deriv > polyorder: every coefficient is 0
};

// The index the mode reads for k outside [0, n) (scipy.ndimage's extension), -1 for the constant.
inline std::int64_t fold(std::int64_t k, std::int64_t n, int mode) {
    switch (mode) {
        case kNearest: return k < 0 ? 0 : n - 1;
        case kWrap: {
            const std::int64_t r = k % n;
            return r < 0 ? r + n : r;
        }
        case kMirror: {
            if (n == 1) return 0;
            const std::int64_t period = 2 * n - 2;
            std::int64_t r = k % period;
            if (r < 0) r += period;
            return r < n ? r : period - r;
        }
        default: return -1;
    }
}

// inner == 1: the outputs [i0, i1) of one line (x, y at the line's start).
void line_rows(const Plan& P, const double* x, double* y, std::int64_t i0, std::int64_t i1) {
    const std::int64_t N = P.N, n = P.n;
    const double* h = P.h;
    const std::int64_t a = std::max(i0, P.ilo), b = std::min(i1, P.ihi);
    if (P.mode != kInterp) {
        auto edge = [&](std::int64_t i) {
            double acc = 0.0;
            for (std::int64_t j = 0; j < N; ++j) {
                const std::int64_t k = i - P.first + j;
                const std::int64_t f = (k >= 0 && k < n) ? k : fold(k, n, P.mode);
                acc = std::fma(h[j], f < 0 ? P.pad[0] : x[f], acc);
            }
            y[i] = acc;
        };
        for (std::int64_t i = i0; i < std::min(i1, P.ilo); ++i) edge(i);
        for (std::int64_t i = std::max(i0, P.ihi); i < i1; ++i) edge(i);
    }
    std::int64_t i = a;
    for (; i + 16 <= b; i += 16) {
        const double* xp = x + (i - P.first);
        float64x2_t s0 = vdupq_n_f64(0.0), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        for (std::int64_t j = 0; j < N; ++j) {
            const float64x2_t c = vdupq_n_f64(h[j]);
            const double* q = xp + j;
            s0 = vfmaq_f64(s0, vld1q_f64(q), c);
            s1 = vfmaq_f64(s1, vld1q_f64(q + 2), c);
            s2 = vfmaq_f64(s2, vld1q_f64(q + 4), c);
            s3 = vfmaq_f64(s3, vld1q_f64(q + 6), c);
            s4 = vfmaq_f64(s4, vld1q_f64(q + 8), c);
            s5 = vfmaq_f64(s5, vld1q_f64(q + 10), c);
            s6 = vfmaq_f64(s6, vld1q_f64(q + 12), c);
            s7 = vfmaq_f64(s7, vld1q_f64(q + 14), c);
        }
        vst1q_f64(y + i, s0);
        vst1q_f64(y + i + 2, s1);
        vst1q_f64(y + i + 4, s2);
        vst1q_f64(y + i + 6, s3);
        vst1q_f64(y + i + 8, s4);
        vst1q_f64(y + i + 10, s5);
        vst1q_f64(y + i + 12, s6);
        vst1q_f64(y + i + 14, s7);
    }
    for (; i < b; ++i) {
        const double* xp = x + (i - P.first);
        double acc = 0.0;
        for (std::int64_t j = 0; j < N; ++j) acc = std::fma(h[j], xp[j], acc);
        y[i] = acc;
    }
}

// inner > 1: the outputs [i0, i1) of one outer block, columns [k0, k1).
void block_rows(
    const Plan& P,
    const double* x,
    double* y,
    std::int64_t i0,
    std::int64_t i1,
    std::int64_t k0,
    std::int64_t k1,
    std::vector<const double*>& rows
) {
    const std::int64_t N = P.N, n = P.n, L = P.inner;
    const double* h = P.h;
    for (std::int64_t i = i0; i < i1; ++i) {
        const bool inside = i >= P.ilo && i < P.ihi;
        if (!inside && P.mode == kInterp) continue;  // the end fits
        for (std::int64_t j = 0; j < N; ++j) {
            std::int64_t k = i - P.first + j;
            if (!inside && (k < 0 || k >= n)) k = fold(k, n, P.mode);
            rows[static_cast<size_t>(j)] = k < 0 ? P.pad : x + k * L;
        }
        double* yr = y + i * L;
        std::int64_t c = k0;
        for (; c + 16 <= k1; c += 16) {
            float64x2_t s0 = vdupq_n_f64(0.0), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
            for (std::int64_t j = 0; j < N; ++j) {
                const float64x2_t hv = vdupq_n_f64(h[j]);
                const double* q = rows[static_cast<size_t>(j)] + c;
                s0 = vfmaq_f64(s0, vld1q_f64(q), hv);
                s1 = vfmaq_f64(s1, vld1q_f64(q + 2), hv);
                s2 = vfmaq_f64(s2, vld1q_f64(q + 4), hv);
                s3 = vfmaq_f64(s3, vld1q_f64(q + 6), hv);
                s4 = vfmaq_f64(s4, vld1q_f64(q + 8), hv);
                s5 = vfmaq_f64(s5, vld1q_f64(q + 10), hv);
                s6 = vfmaq_f64(s6, vld1q_f64(q + 12), hv);
                s7 = vfmaq_f64(s7, vld1q_f64(q + 14), hv);
            }
            vst1q_f64(yr + c, s0);
            vst1q_f64(yr + c + 2, s1);
            vst1q_f64(yr + c + 4, s2);
            vst1q_f64(yr + c + 6, s3);
            vst1q_f64(yr + c + 8, s4);
            vst1q_f64(yr + c + 10, s5);
            vst1q_f64(yr + c + 12, s6);
            vst1q_f64(yr + c + 14, s7);
        }
        for (; c + 2 <= k1; c += 2) {
            float64x2_t s0 = vdupq_n_f64(0.0);
            for (std::int64_t j = 0; j < N; ++j)
                s0 = vfmaq_f64(s0, vld1q_f64(rows[static_cast<size_t>(j)] + c), vdupq_n_f64(h[j]));
            vst1q_f64(yr + c, s0);
        }
        for (; c < k1; ++c) {
            double acc = 0.0;
            for (std::int64_t j = 0; j < N; ++j) acc = std::fma(h[j], rows[static_cast<size_t>(j)][c], acc);
            yr[c] = acc;
        }
    }
}

// 'interp': the first and the last half outputs of the lines (o, k) for k in [k0, k1) of one outer block.
void end_fits(const Plan& P, const double* x, double* y, std::int64_t k0, std::int64_t k1, std::vector<double>& work) {
    const std::int64_t N = P.N, n = P.n, L = P.inner, half = P.half, nu = P.nu, d = P.d;
    const std::int64_t W = k1 - k0;
    if (P.zero) {  // every coefficient is 0: 0 x, which keeps a NaN or an inf of the window as NaN
        for (int side = 0; side < 2; ++side) {
            const std::int64_t start = side == 0 ? 0 : n - N;
            for (std::int64_t c = k0; c < k1; ++c) {
                double acc = 0.0;
                for (std::int64_t i = 0; i < N; ++i) acc = std::fma(0.0, x[(start + i) * L + c], acc);
                for (std::int64_t q = 0; q < half; ++q) y[(side == 0 ? q : n - 1 - q) * L + c] = acc;
            }
        }
        return;
    }
    const std::int64_t J = d + 1 - nu;  // the degrees nu .. d
    // Structure of arrays (the loops over the lines vectorize): a_j's high and low parts, then the outputs'.
    if (work.size() < static_cast<size_t>(2 * (J + 1) * W)) work.resize(static_cast<size_t>(2 * (J + 1) * W));
    double* const ah = work.data();
    double* const al = ah + J * W;
    double* const yh = al + J * W;
    double* const yl = yh + W;
    for (int side = 0; side < 2; ++side) {
        const std::int64_t start = side == 0 ? 0 : n - N;
        std::fill(ah, ah + 2 * J * W, 0.0);
        for (std::int64_t i = 0; i < N; ++i) {
            const double* __restrict row = x + (start + i) * L + k0;
            for (std::int64_t j = 0; j < J; ++j) {
                const DD t = P.G->node(i, nu + j);
                double* __restrict h = ah + j * W;
                double* __restrict l = al + j * W;
                for (std::int64_t c = 0; c < W; ++c) accumulate(h[c], l[c], t.hi, t.lo, row[c], 0.0);
            }
        }
        // left: position q at output q; right: position N - 1 - q at output n - 1 - q, whose factors are
        // (-1)^(j + nu) those of q (the window's symmetry)
        for (std::int64_t q = 0; q < half; ++q) {
            const DD* cq = P.C->data() + q * (d + 1);
            std::fill(yh, yh + 2 * W, 0.0);
            for (std::int64_t j = 0; j < J; ++j) {
                DD f = cq[nu + j];
                if (side == 1 && (j % 2 == 1)) f = neg(f);
                const double* __restrict h = ah + j * W;
                const double* __restrict l = al + j * W;
                for (std::int64_t c = 0; c < W; ++c) accumulate(yh[c], yl[c], h[c], l[c], f.hi, f.lo);
            }
            double* yr = y + (side == 0 ? q : n - 1 - q) * L + k0;
            for (std::int64_t c = 0; c < W; ++c) yr[c] = finish(DD{yh[c], yl[c]}, P.S);
        }
    }
}

nb::object savgol_filter(
    nb::handle xo,
    std::int64_t N,
    std::int64_t d,
    std::int64_t nu,
    double delta,
    std::int64_t axis,
    int mode,
    double cval
) {
    const NDArray& X = require_ndarray(xo, "x");
    if (!X.c_contiguous || (X.dtype != DType::Float64 && X.dtype != DType::Complex128) || X.ndim < 1)
        throw nb::value_error("'x' must be a C-contiguous float64 or complex128 array of at least one dimension");
    if (axis < 0 || axis >= X.ndim) throw nb::value_error("'axis' is out of range");
    if (N < 1 || d < 0 || d >= N || nu < 0 || mode < 0 || mode > 4)
        throw nb::value_error("needs 0 <= polyorder < window_length, deriv >= 0 and a mode in 0 .. 4");
    const bool cplx = X.dtype == DType::Complex128;
    std::int64_t outer = 1, inner = 1;
    for (int a = 0; a < axis; ++a) outer *= X.shape[a];
    for (int a = axis + 1; a < X.ndim; ++a) inner *= X.shape[a];
    const std::int64_t n = X.shape[axis];
    if (mode == kInterp && n < N) throw nb::value_error("window_length must not exceed the axis for mode 'interp'");
    NDArray out = new_array(X.dtype, X.shape_span());
    if (outer == 0 || inner == 0 || n == 0) return wrap(std::move(out));

    Plan P;
    P.x = reinterpret_cast<const double*>(X.data);
    P.y = reinterpret_cast<double*>(out.data);
    P.outer = outer;
    P.n = n;
    P.inner = cplx ? 2 * inner : inner;
    P.N = N;
    P.d = d;
    P.nu = nu;
    P.first = (N % 2 == 0) ? N / 2 - 1 : N / 2;
    P.half = N / 2;
    P.mode = mode;
    P.zero = nu > d;
    if (mode == kInterp) {
        P.ilo = P.half;
        P.ihi = n - P.half;
    } else {
        P.ilo = std::min(P.first, n);
        P.ihi = std::min(n, std::max(P.ilo, n - N + P.first + 1));
    }
    const std::int64_t L = P.inner;
    const std::int64_t work = outer * n * L * N;
    const bool big = work >= kParallelThreshold;

    std::vector<double> h(static_cast<size_t>(N), 0.0), pad(static_cast<size_t>(std::max<std::int64_t>(L, 1)), 0.0);
    for (std::int64_t c = 0; c < L; ++c) pad[static_cast<size_t>(c)] = (cplx && c % 2 == 1) ? 0.0 : cval;
    P.pad = pad.data();
    Gram G;
    std::vector<DD> C;
    {
        const parallel::Unlocked unlocked(big || N * (d + 1) >= kParallelThreshold);
        if (!P.zero) {
            G = make_gram(N, d, true);
            P.S = inverse_power(delta, nu);
            coefficients(G, -1, nu, P.S, h.data());
            if (mode == kInterp && P.half > 0) {
                C.resize(static_cast<size_t>(P.half * (d + 1)));
                for (std::int64_t q = 0; q < P.half; ++q) {
                    const std::vector<DD> c = factors(G, q, nu);
                    std::copy(c.begin(), c.end(), C.begin() + q * (d + 1));
                }
            }
        }
    }
    P.h = h.data();
    P.G = &G;
    P.C = &C;

    // Tasks by shape alone. A unit is a block of output rows of one outer block (and of 512 columns when the axis is
    // not the last); a task takes consecutive units worth ~2^17 fma. Split from 2^18 fma: below it the wake-up of the
    // workers costs more than it saves (decision SM75).
    constexpr std::int64_t kTaskWork = std::int64_t{1} << 17, kSplitWork = std::int64_t{1} << 18;
    const std::int64_t K = L == 1 ? 1 : std::min<std::int64_t>(L, 512);
    const std::int64_t kblocks = (L + K - 1) / K;
    std::int64_t B = std::max<std::int64_t>(1, kTaskWork / std::max<std::int64_t>(1, K * N));
    if (L == 1) B = (B + 15) / 16 * 16;
    B = std::min(B, n);
    const std::int64_t rblocks = (n + B - 1) / B;
    const std::int64_t units = outer * rblocks * kblocks;
    const std::int64_t per = std::max<std::int64_t>(1, kTaskWork / std::max<std::int64_t>(1, B * K * N));
    const std::int64_t tasks = (units + per - 1) / per;
    auto run = [&](std::int64_t t) {
        std::vector<const double*> rows(static_cast<size_t>(L == 1 ? 0 : N));
        for (std::int64_t u = t * per; u < std::min(units, (t + 1) * per); ++u) {
            const std::int64_t o = u / (rblocks * kblocks), r = (u / kblocks) % rblocks, kb = u % kblocks;
            const double* xo_ = P.x + o * n * L;
            double* yo = P.y + o * n * L;
            const std::int64_t i0 = r * B, i1 = std::min(n, i0 + B);
            if (L == 1) line_rows(P, xo_, yo, i0, i1);
            else block_rows(P, xo_, yo, i0, i1, kb * K, std::min(L, (kb + 1) * K), rows);
        }
    };
    if (work >= kSplitWork && tasks > 1 && parallel::available()) {
        parallel::parallel_for(tasks, run, work_ns(2 * work / tasks));
    } else {
        const parallel::Unlocked unlocked(big);
        for (std::int64_t t = 0; t < tasks; ++t) run(t);
    }

    if (mode == kInterp && P.half > 0) {
        // the end fits: a unit is up to KE lines of one outer block; a task takes units worth ~2^17 flops
        const std::int64_t J = P.zero ? 1 : d + 1 - nu;
        const std::int64_t per_line = 2 * (N + P.half) * J * 12;  // double-double steps of ~12 flops
        const std::int64_t KE = L == 1 ? 1 : std::clamp<std::int64_t>(4096 / (N * J) / 16 * 16, 16, L);
        const std::int64_t eblocks = (L + KE - 1) / KE;
        const std::int64_t eunits = outer * eblocks;
        const std::int64_t eper = std::max<std::int64_t>(1, kTaskWork / std::max<std::int64_t>(1, KE * per_line));
        const std::int64_t etasks = (eunits + eper - 1) / eper;
        const std::int64_t ework = outer * L * per_line;
        auto erun = [&](std::int64_t t) {
            std::vector<double> work;
            for (std::int64_t u = t * eper; u < std::min(eunits, (t + 1) * eper); ++u) {
                const std::int64_t o = u / eblocks, kb = u % eblocks;
                end_fits(P, P.x + o * n * L, P.y + o * n * L, kb * KE, std::min(L, (kb + 1) * KE), work);
            }
        };
        if (ework >= kSplitWork && etasks > 1 && parallel::available()) {
            parallel::parallel_for(etasks, erun, work_ns(ework / etasks));
        } else {
            const parallel::Unlocked unlocked(ework >= kParallelThreshold);
            for (std::int64_t t = 0; t < etasks; ++t) erun(t);
        }
    }
    return wrap(std::move(out));
}

}  // namespace

void bind_savgol(nb::module_& m) {
    m.def(
        "savgol_coeffs", named("savgol_coeffs", &savgol_coeffs), nb::arg("window_length"), nb::arg("polyorder"),
        nb::arg("deriv"), nb::arg("delta"), nb::arg("pos")
    );  // fmt: skip
    m.def(
        "savgol_filter", named("savgol_filter", &savgol_filter), nb::arg("x").none(), nb::arg("window_length"),
        nb::arg("polyorder"), nb::arg("deriv"), nb::arg("delta"), nb::arg("axis"), nb::arg("mode"), nb::arg("cval")
    );  // fmt: skip
}

}  // namespace sd::smooth
