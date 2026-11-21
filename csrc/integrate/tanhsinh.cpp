// tanhsinh and nsum (docs/design/integrate.md §3.5, §3.6, decisions C4, C5, C84-C93; scipy 1.18.1's
// integrate/_tanhsinh.py), the same steps and evaluation points as scipy's.
//
// tanhsinh: the elementwise family's loop (csrc/elementwise/loop.hpp; optimize O38). The first call
// f(c) at every element's midpoint (k,) (scipy's eim._initialize; a NaN value stops the element),
// then one call per level on the elements in progress, (k, p) with p = 2 x the level's pairs: the
// first call of the loop all the pairs of levels 0 ... min_level (66 points for min_level 2), each
// next one the new pairs of the next level (64, 128, ...). The limits are transformed as scipy's
// _transform_integrals (equal limits are 0, reversed ones swapped and the sign changed, both
// infinite x = t / (1 - t^2), one infinite x = 1/t - 1 + a, the left one x -> -x), the pairs come
// from scipy's table (below), and each level is scipy's Euler-Maclaurin sum and error estimate
// ([1] section 5): the most extreme valid abscissae and their terms (d4), non-finite values
// replaced by the value at the nearest finite point (van Engelen), the sum S_n = h sum f_j w_j
// (+ S_{n-1}/2), the error clip(max(d1^(log d1/log d2), d1^2, eps max|f_j w_j|, d4), eps |S_n|, d1).
// Convergence is error < atol or error < rtol |S_n|; a non-finite S_n stops the element with -3,
// max_level with -2. log=True is scipy's: f gives the log of the integrand, the sums are scipy's
// logsumexp (special/_logsumexp.py) and d1, d2 its complex form (the log of a difference).
//
// The table (C84): scipy's _compute_pair with libm's cosh, sinh, exp and the products c * c of
// numpy's **2, so every abscissa complement and weight is scipy's to the bit (§3.5 of the design;
// 0-980 ULP from the exact values of the same formula, §7.12). Each level is computed once, when
// first needed, and kept (a level, once there, is never moved).
//
// The arithmetic is numpy's (scipy's steps are numpy operations): every sum along a row is numpy's
// pairwise sum (DOUBLE_pairwise_sum: 8 accumulators, blocks of 128, after the reduction's 0), not
// the library's 16-accumulator one, since the sums decide the stopping test and so the calls; the
// complex steps of logsumexp are numpy's (the system's cexp and clog, numpy's log1p, division and
// pairwise sum). Each element's arithmetic is the same on any thread.
//
// nsum: scipy's branches in scipy's order after the first call f(a) on every element: invalid
// a, b, step (-1; b < a is 0), the direct sum of at most max_terms terms (one call (k, terms), the
// padding NaN), then the integral bound to the right (a finite), to the left (f(-x)) and on both
// sides (f(x) + f(-x), x = 0 halved: two calls each time): tanhsinh's lower bound, the term below
// the tolerance among a + 2^j step (two calls), the direct sum up to it, tanhsinh for the rest and
// f(b) when b is finite; the sum and error are scipy's.
//
// [1] Bailey, Jeyabalan and Li, "A comparison of three high-precision quadrature schemes",
//     Experimental Mathematics 14.3 (2005).
#include "integrate/integrate.hpp"
#include "integrate/numpy_ops.hpp"

#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "elementwise/loop.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

// The system's complex exp and log (numpy's npy_cexp and npy_clog are these on macOS), as the
// pair of doubles a double _Complex is in the arm64 calling convention.
struct CPair {
    double re, im;
};
extern "C" CPair sd_libm_cexp(CPair) __asm__("_cexp");
extern "C" CPair sd_libm_clog(CPair) __asm__("_clog");

}  // namespace

namespace sd::integrate {

namespace nb = nanobind;

namespace {

using namespace sd::elementwise;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 0x1p-52;
constexpr int kMaxLevel = 16;  // the largest max_level (C88)

// ---- numpy's arithmetic

// the real pairwise sum and np.sum along a row (integrate/numpy_ops.hpp)
using numpy::np_sum;

// CDOUBLE_pairwise_sum over n doubles (re, im interleaved)
void cpairwise(const double* a, std::size_t n, double& rr, double& ri) {
    if (n < 8) {
        rr = -0.0;
        ri = -0.0;
        for (std::size_t i = 0; i < n; i += 2) {
            rr += a[i];
            ri += a[i + 1];
        }
        return;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + static_cast<std::size_t>(j)];
        rr = (r[0] + r[2]) + (r[4] + r[6]);
        ri = (r[1] + r[3]) + (r[5] + r[7]);
        for (; i < n; i += 2) {
            rr += a[i];
            ri += a[i + 1];
        }
        return;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    double r1, i1, r2, i2;
    cpairwise(a, n2, r1, i1);
    cpairwise(a + n2, n - n2, r2, i2);
    rr = r1 + r2;
    ri = i1 + i2;
}

CPair np_csum(const CPair* z, std::size_t n) {
    double rr, ri;
    cpairwise(reinterpret_cast<const double*>(z), 2 * n, rr, ri);
    return {0.0 + rr, 0.0 + ri};
}

// numpy's complex division (in the case of a real divisor m != 0)
CPair np_cdiv(CPair s, CPair m) {
    if (std::abs(m.re) >= std::abs(m.im)) {
        if (m.re == 0 && m.im == 0) return {s.re / std::abs(m.re), s.im / std::abs(m.re)};
        const double rat = m.im / m.re, scl = 1.0 / (m.re + m.im * rat);
        return {(s.re + s.im * rat) * scl, (s.im - s.re * rat) * scl};
    }
    const double rat = m.re / m.im, scl = 1.0 / (m.im + m.re * rat);
    return {(s.re * rat + s.im) * scl, (s.im * rat - s.re) * scl};
}

// np.max along a row (NaN propagates)
double np_max(const double* a, std::size_t n) {
    double m = a[0];
    for (std::size_t i = 1; i < n; ++i) m = npmax(m, a[i]);
    return m;
}

// np.clip(x, lo, hi): numpy's minimum(maximum(x, lo), hi) with NaN propagating
double np_clip(double x, double lo, double hi) {
    const double t = std::isnan(x) ? x : (x > lo ? x : lo);
    return std::isnan(t) ? t : (t < hi ? t : hi);
}

// scipy.special.logsumexp of n reals (b = None, return_sign = False): the largest terms (all that
// equal the maximum) taken out of the sum, log1p(sum exp(a - max) / m) + log m + max, and the direct
// log(sum exp(a)) where that is not finite. tmp: n doubles of scratch.
double logsumexp(const double* a, std::size_t n, double* tmp) {
    if (n == 0) return -kInf;
    for (std::size_t i = 0; i < n; ++i) tmp[i] = std::exp(a[i]);
    const double out_inf = std::log(np_sum(tmp, n));
    const double mx = np_max(a, n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = a[i] == mx ? 1.0 : 0.0;
    double m = np_sum(tmp, n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = std::exp((a[i] == mx ? -kInf : a[i]) - mx);
    double s = np_sum(tmp, n);
    if (s != 0) s = s / m;
    const double sgn = npsign(s + 1) * npsign(m);
    if (s < -1) s = -s - 2;
    m = std::abs(m);
    double out = std::log1p(s) + std::log(m) + mx;
    if (sgn < 0) out = kNaN;
    return std::isfinite(out) ? out : out_inf;
}

double logsumexp2(double x, double y) {
    const double a[2] = {x, y};
    double t[2];
    return logsumexp(a, 2, t);
}

// The real part of scipy's logsumexp of n <= 4 complex numbers (one element's stack: scipy's d1, d2
// and nsum's error), the element with the largest real part (the last of equal ones) taken out.
double logsumexp_re(const CPair* a, std::size_t n) {
    CPair e[4], z[4], ones[4];
    for (std::size_t i = 0; i < n; ++i) e[i] = sd_libm_cexp(a[i]);
    const double out_inf = sd_libm_clog(np_csum(e, n)).re;
    double mx = a[0].re;
    for (std::size_t i = 1; i < n; ++i) mx = npmax(mx, a[i].re);
    std::int64_t sel = -1;  // the largest index of an element with the largest real part
    for (std::size_t i = 0; i < n; ++i)
        if (a[i].re == mx) sel = static_cast<std::int64_t>(i);
    bool pick[4];
    for (std::size_t i = 0; i < n; ++i) {
        pick[i] = (a[i].re == mx ? static_cast<std::int64_t>(i) : -1) == sel;
        z[i] = pick[i] ? a[i] : CPair{0.0, 0.0};
        ones[i] = pick[i] ? CPair{1.0, 0.0} : CPair{0.0, 0.0};
    }
    const CPair amax = np_csum(z, n), m = np_csum(ones, n);
    for (std::size_t i = 0; i < n; ++i) {
        const CPair ai = pick[i] ? CPair{-kInf, 0.0} : a[i];
        e[i] = sd_libm_cexp(CPair{ai.re - amax.re, ai.im - amax.im});
    }
    CPair s = np_csum(e, n);
    if (!(s.re == 0 && s.im == 0)) s = np_cdiv(s, m);
    const CPair lm = sd_libm_clog(m);
    // numpy's complex log1p: log(hypot(re + 1, im)) + i atan2(im, re + 1)
    const double re = std::log(std::hypot(s.re + 1, s.im)) + lm.re + amax.re;
    const double im = std::atan2(s.im, s.re + 1) + lm.im + amax.im;
    return std::isfinite(re) && std::isfinite(im) ? re : out_inf;
}

// ---- the table: scipy's _get_base_step and _compute_pair

struct Table {
    double h0 = 0;
    std::array<vec, kMaxLevel + 1> xjc, wj;     // level k's pairs
    std::array<std::size_t, kMaxLevel + 2> idx{};  // idx[k]: the pairs of levels 0 ... k - 1
    std::atomic<int> ready{-1};
    std::mutex mu;

    Table() {
        // the base step: xjc underflows at tmax ([2] Eq. 13), 8 steps to it
        const double fmin = 4 * std::numeric_limits<double>::min();
        const double tmax = std::asinh(std::log(2 / fmin - 1) / M_PI);
        h0 = tmax / 8;
        idx[0] = 0;
        for (int k = 0; k <= kMaxLevel; ++k)
            idx[static_cast<std::size_t>(k) + 1] =
                idx[static_cast<std::size_t>(k)] + (k == 0 ? 9 : 4 * (std::size_t{1} << k));
    }

    void compute(int k) {
        const double h = h0 / std::ldexp(1.0, k);  // scipy's h0 / 2**k
        const std::size_t top = std::size_t{8} << k;
        const double pi_2 = M_PI / 2;
        vec& X = xjc[static_cast<std::size_t>(k)];
        vec& W = wj[static_cast<std::size_t>(k)];
        for (std::size_t j = k == 0 ? 0 : 1; j <= top; j += k == 0 ? 1 : 2) {
            const double jh = static_cast<double>(j) * h;
            const double u1 = pi_2 * std::cosh(jh), u2 = pi_2 * std::sinh(jh);
            const double c = std::cosh(u2);
            W.push_back(u1 / (c * c));
            X.push_back(1 / (std::exp(u2) * std::cosh(u2)));
        }
        if (k == 0) W[0] = W[0] / 2;  // x = 0 is evaluated twice, each with half its weight
    }
};

// The table with levels 0 ... level ready.
const Table& table(int level) {
    static Table T;
    if (T.ready.load(std::memory_order_acquire) < level) {
        std::lock_guard<std::mutex> lock(T.mu);
        for (int k = T.ready.load(std::memory_order_relaxed) + 1; k <= level; ++k) {
            T.compute(k);
            T.ready.store(k, std::memory_order_release);
        }
    }
    return T;
}

// ---- the function evaluated: the user's f, or nsum's f(-x) and f(x) + f(-x)

enum Form { kPlain, kNegated, kBoth };

// One call of the function on the points of k of a solve's elements: start(k) one point each
// ((k,)), start(k, p) p points each ((k, p)), put, run(local) with local[r] the solve's element of
// row r (its batch element elems[local[r]] for the args; local itself when elems is null), value.
// kBoth calls f twice, at x then at -x, and gives (f(x) + f(-x)) (x == 0 halved), in log
// logsumexp(f(x), f(-x)) (+ log 0.5 at 0): scipy's nsum on both sides.
class Eval {
public:
    Eval(Call& F, Form form, bool log, const std::int64_t* elems) : F_(F), form_(form), log_(log), elems_(elems) {}

    void start(std::size_t k) { begin(k, 1, false); }
    void start(std::size_t k, std::size_t p) { begin(k, p, true); }
    void put(std::size_t r, std::size_t j, double x) {
        if (form_ == kBoth) {
            x_[r * p_ + j] = x;
            return;
        }
        const double y = form_ == kNegated ? -x : x;
        if (rows_) F_.put(r, j, y);
        else F_.put(r, y);
    }
    void run(const std::int64_t* local) {
        const std::int64_t* idx = local;
        if (elems_) {
            idx_.resize(k_);
            for (std::size_t r = 0; r < k_; ++r) idx_[r] = elems_[local[r]];
            idx = idx_.data();
        }
        if (form_ != kBoth) {
            F_.run(idx);
            return;
        }
        const std::size_t n = k_ * p_;
        v_.resize(n);
        call(idx, 1.0);
        for (std::size_t q = 0; q < n; ++q) v_[q] = raw(q);
        call(idx, -1.0);
        const double lh = std::log(0.5);
        for (std::size_t q = 0; q < n; ++q) {
            const bool zero = x_[q] == 0;
            const double u = v_[q], w = raw(q);
            v_[q] = log_ ? logsumexp2(u, w) + (zero ? lh : 0.0) : (u + w) * (zero ? 0.5 : 1.0);
        }
    }
    double value(std::size_t r, std::size_t j) const {
        if (form_ == kBoth) return v_[r * p_ + j];
        return rows_ ? F_.value(r, j) : F_.value(r);
    }

private:
    void begin(std::size_t k, std::size_t p, bool rows) {
        k_ = k;
        p_ = p;
        rows_ = rows;
        if (form_ == kBoth) x_.resize(k * p);
        else if (rows) F_.start(k, p);
        else F_.start(k);
    }
    void call(const std::int64_t* idx, double sign) {
        if (rows_) F_.start(k_, p_);
        else F_.start(k_);
        for (std::size_t r = 0; r < k_; ++r)
            for (std::size_t j = 0; j < p_; ++j) {
                const double y = sign < 0 ? -x_[r * p_ + j] : x_[r * p_ + j];
                if (rows_) F_.put(r, j, y);
                else F_.put(r, y);
            }
        F_.run(idx);
    }
    double raw(std::size_t q) const { return rows_ ? F_.value(q / p_, q % p_) : F_.value(q); }

    Call& F_;
    Form form_;
    bool log_;
    const std::int64_t* elems_;
    std::size_t k_ = 0, p_ = 1;
    bool rows_ = false;
    std::vector<std::int64_t> idx_;
    vec x_, v_;
};

// ---- tanhsinh

struct Options {
    bool log;
    int max_level, min_level;
    double atol, rtol;
};

struct TsOut {
    vec integral, error;
    ivec status, nfev, level;
};

// The limits' transform (scipy's _transform_integrals) as flags.
enum Kind : std::uint8_t { kNeg = 1, kAbInf = 2, kAInf = 4, kBInf = 8, kSame = 16 };

TsOut tanhsinh_run(Eval& ev, const vec& a_in, const vec& b_in, const Options& o) {
    const std::size_t n = a_in.size();
    const bool lg = o.log;
    const double zero = lg ? -kInf : 0.0;
    const Table& T = table(o.max_level);
    const int min_level = std::min(o.min_level, o.max_level);
    const std::int64_t max_iter = o.max_level - min_level + 1;

    // scipy's first evaluation (eim._initialize): f at each interval's midpoint
    ev.start(n);
    for (std::size_t r = 0; r < n; ++r) {
        const double a = a_in[r], b = b_in[r];
        double c = (a + b) / 2;
        if (std::isinf(a)) c = b - 1.;
        if (std::isinf(b)) c = a + 1.;
        if (std::isinf(a) && std::isinf(b)) c = 0.;
        ev.put(r, 0, c);
    }
    Progress P(n);
    ev.run(P.active.data());

    struct E {  // one element's state, together in memory
        double a, b, a0, alpha;
        double Sn, S1, S2, aerr;  // the estimate, Sk[-1], Sk[-2], the error
        double xr0, fr0, wr0, xl0, fl0, wl0, d4;
        std::uint8_t kind;
    };
    std::vector<E> el(n);
    for (std::size_t i = 0; i < n; ++i) {
        double a = a_in[i], b = b_in[i];
        std::uint8_t kind = 0;
        if (a == b) {
            a = b = 1;
            kind |= kSame;
        }
        if (b < a) {
            std::swap(a, b);
            kind |= kNeg;
        }
        if (std::isinf(a) && std::isinf(b)) {
            a = -1;
            b = 1;
            kind |= kAbInf;
        }
        if (std::isinf(a)) {
            const double t = a;
            a = -b;
            b = -t;
            kind |= kAInf;
        }
        const double a0 = a;
        if (std::isinf(b)) {
            a = 0;
            b = 1;
            kind |= kBInf;
        }
        const double fs0 = ev.value(i, 0);
        const double Sn = std::isnan(a) || std::isnan(b) || std::isnan(fs0) ? kNaN : zero;
        el[i] = {a, b, a0, (b - a) / 2, Sn, kNaN, kNaN, kNaN, -kInf, kNaN, 0.0, kInf, kNaN, 0.0, 0.0, kind};
    }
    // scipy's check before the first iteration: equal limits are 0; a NaN limit or midpoint value -3
    std::vector<std::uint32_t> on;
    advance(P, on, [&](std::size_t r) {
        const auto i = static_cast<std::size_t>(P.active[r]);
        E& e = el[i];
        if (e.kind & kSame) {
            if (!std::isnan(e.Sn)) e.Sn = zero;
            e.aerr = e.Sn;
            P.status[i] = sConverged;
        } else if (std::isnan(e.Sn)) {
            P.status[i] = sValueErr;
        } else {
            return true;
        }
        P.stopped(i, 0, 1);
        return false;
    });

    const double log2 = std::log(2.0), log_eps = std::log(kEps);
    std::int64_t nit = 0, nfev = 1;
    vec first_x, first_w;  // the first call's pairs: levels 0 ... min_level
    while (nit < max_iter && !P.active.empty()) {
        const int lev = min_level + static_cast<int>(nit);
        const bool first = nit == 0;
        const double *X, *W;
        std::size_t np;
        if (first) {
            for (int k = 0; k <= lev; ++k) {
                first_x.insert(
                    first_x.end(), T.xjc[static_cast<std::size_t>(k)].begin(), T.xjc[static_cast<std::size_t>(k)].end()
                );
                first_w.insert(
                    first_w.end(), T.wj[static_cast<std::size_t>(k)].begin(), T.wj[static_cast<std::size_t>(k)].end()
                );
            }
            X = first_x.data();
            W = first_w.data();
            np = first_x.size();
        } else {
            X = T.xjc[static_cast<std::size_t>(lev)].data();
            W = T.wj[static_cast<std::size_t>(lev)].data();
            np = T.xjc[static_cast<std::size_t>(lev)].size();
        }
        const std::size_t k = P.active.size(), p = 2 * np;
        const double h = T.h0 / std::ldexp(1.0, lev);  // scipy's h0 / 2**n
        // the abscissae on [a, b] (right half -alpha xjc + b, left half alpha xjc + a) and the
        // substitutions of the infinite limits
        auto abscissa = [&](const E& e, std::size_t q) {
            return q < np ? -e.alpha * X[q] + e.b : e.alpha * X[q - np] + e.a;
        };
        ev.start(k, p);
        for_elements(k, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t r = r0; r < r1; ++r) {
                const E& e = el[static_cast<std::size_t>(P.active[r])];
                for (std::size_t q = 0; q < p; ++q) {
                    double x = abscissa(e, q);
                    if (e.kind & kAbInf) x = x / (1 - x * x);
                    if (e.kind & kBInf) x = 1 / x - 1 + e.a0;
                    if (e.kind & kAInf) x = x * -1;
                    ev.put(r, q, x);
                }
            }
        });
        ev.run(P.active.data());
        nfev += static_cast<std::int64_t>(p);
        nit += 1;
        const std::size_t nx1 = T.idx[static_cast<std::size_t>(lev)];  // levels below lev (Snm1)
        const std::size_t nx2 = lev >= 1 ? T.idx[static_cast<std::size_t>(lev) - 1] : 0;
        advance(P, on, [&](std::size_t r) {
            const auto i = static_cast<std::size_t>(P.active[r]);
            E& e = el[i];
            thread_local vec fw, wv, tmp;
            fw.resize(p);
            wv.resize(p);
            tmp.resize(p);
            // the values with the substitutions' Jacobians, the weights (0 at an abscissa on a
            // limit), and the most extreme valid abscissae of this level (the first of equal ones)
            double xr = -kInf, xl = kInf;
            std::size_t ir = 0, il = np;
            for (std::size_t q = 0; q < p; ++q) {
                const double x = abscissa(e, q);
                const double w = x <= e.a || x >= e.b ? 0.0 : W[q < np ? q : q - np] * e.alpha;
                double f = ev.value(r, q);
                if (e.kind & kAbInf) {
                    if (lg) {
                        f = f + (std::log(1 + x * x) - 2 * std::log(1 - x * x));
                    } else {
                        const double t = 1 - x * x;
                        f = f * ((1 + x * x) / (t * t));
                    }
                } else if (e.kind & kBInf) {
                    f = lg ? f - 2 * std::log(x) : f * std::pow(x, -2.0);
                }
                fw[q] = f;
                wv[q] = w;
                const bool valid = std::isfinite(f) && w != 0;
                if (q < np) {
                    if ((valid ? x : -kInf) > xr) {
                        xr = x;
                        ir = q;
                    }
                } else if ((valid ? x : kInf) < xl) {
                    xl = x;
                    il = q;
                }
            }
            if (xr > e.xr0) {
                e.xr0 = xr;
                e.fr0 = fw[ir];
                e.wr0 = wv[ir];
            }
            if (xl < e.xl0) {
                e.xl0 = xl;
                e.fl0 = fw[il];
                e.wl0 = wv[il];
            }
            // d4: the larger of the outermost terms
            e.d4 = lg ? npmax(e.fl0 + std::log(e.wl0), e.fr0 + std::log(e.wr0))
                      : npmax(std::abs(e.fl0 * e.wl0), std::abs(e.fr0 * e.wr0));
            // a non-finite value (or a zero weight) takes the nearest finite point's value
            // ([3] p. 22), then the terms f_j w_j
            for (std::size_t q = 0; q < p; ++q) {
                double f = fw[q];
                if (!(std::isfinite(f) && wv[q] != 0)) f = q < np ? e.fr0 : e.fl0;
                fw[q] = lg ? f + std::log(wv[q]) : f * wv[q];
            }
            // the Euler-Maclaurin sum
            double Sn;
            if (lg) {
                const double lh = std::log(h);
                for (std::size_t q = 0; q < p; ++q) tmp[q] = fw[q] + lh;
                Sn = logsumexp(tmp.data(), p, wv.data());
            } else {
                Sn = np_sum(fw.data(), p) * h;
            }
            if (!first) Sn = lg ? logsumexp2(e.S1 - log2, Sn) : e.S1 / 2 + Sn;
            e.Sn = Sn;
            // the error estimate ([1] section 5), from level 2 on: the sums of the two levels
            // below, from this call's terms in its first iteration
            double aerr = kNaN, rerr = kNaN;
            if (lev > 0) {
                auto partial = [&](std::size_t nx, double hh) {
                    for (std::size_t q = 0; q < nx; ++q) {
                        tmp[q] = fw[q];
                        tmp[nx + q] = fw[np + q];
                    }
                    return lg ? logsumexp(tmp.data(), 2 * nx, wv.data()) + std::log(hh)
                              : np_sum(tmp.data(), 2 * nx) * hh;
                };
                if (first) e.S1 = partial(nx1, 2 * h);
                if (lev > 1) {
                    if (first) e.S2 = partial(nx2, 4 * h);
                    if (lg) {
                        const CPair s1[2] = {{Sn, 0.0}, {e.S1 + 0.0, 0.0 + M_PI}};
                        const CPair s2[2] = {{Sn, 0.0}, {e.S2 + 0.0, 0.0 + M_PI}};
                        const double d1 = logsumexp_re(s1, 2), d2 = logsumexp_re(s2, 2);
                        const double d3 = log_eps + np_max(fw.data(), p), d5 = log_eps + Sn;
                        const double t = d1 > -kInf ? d1 * d1 / d2 : -kInf;
                        const double ds = npmax(npmax(npmax(t, 2 * d1), d3), e.d4);
                        aerr = np_clip(ds, d5, d1);
                        rerr = aerr - Sn;
                    } else {
                        const double d1 = std::abs(Sn - e.S1), d2 = std::abs(Sn - e.S2);
                        double m = std::abs(fw[0]);
                        for (std::size_t q = 1; q < p; ++q) m = npmax(m, std::abs(fw[q]));
                        const double d3 = kEps * m, d5 = kEps * std::abs(Sn);
                        const double t = d1 > 0 ? std::pow(d1, std::log(d1) / std::log(d2)) : 0.0;
                        const double ds = npmax(npmax(npmax(t, d1 * d1), d3), e.d4);
                        aerr = np_clip(ds, d5, d1);
                        rerr = aerr / std::abs(Sn);
                    }
                }
            }
            e.aerr = aerr;
            int st = sInProgress;
            if (rerr < o.rtol || aerr < o.atol) st = sConverged;
            else if (lg ? Sn == kInf || std::isnan(Sn) : !std::isfinite(Sn)) st = sValueErr;
            if (st == sInProgress) {  // scipy's post_termination_check: Sk gains S_n
                e.S2 = e.S1;
                e.S1 = Sn;
                return true;
            }
            P.status[i] = st;
            P.stopped(i, nit, nfev);
            return false;
        });
        first_x.clear();
        first_w.clear();
    }
    P.finish(nit, nfev);
    TsOut out;
    out.integral.resize(n);
    out.error.resize(n);
    out.level.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const E& e = el[i];
        double I = e.Sn;
        if (e.kind & kNeg) I = lg ? kNaN : I * -1.0;  // log: the log of a negative integral is complex (C85)
        out.integral[i] = I;
        out.error[i] = e.aerr;
        out.level[i] = P.nit[i] == 0 ? -1 : min_level + P.nit[i] - 1;
    }
    out.status = std::move(P.status);
    out.nfev = std::move(P.nfev);
    return out;
}

// ---- nsum

struct Sum {
    vec S, E;
    ivec status, nfev;
};

// scipy's _direct: sum_{j < m} f(a + j step) over a + j step < b (+ step/2 when inclusive), one
// call (k, m) with m the most terms of any element (the rest NaN); nfev the terms evaluated.
Sum direct(Eval& ev, const vec& a, const vec& b, const vec& step, bool inclusive, bool lg) {
    const std::size_t k = a.size();
    const double zero = lg ? -kInf : 0.0;
    double most = -kInf;
    for (std::size_t r = 0; r < k; ++r) {
        const double s = std::nearbyint((b[r] - a[r]) / step[r]) + (inclusive ? 1 : 0);  // np.round
        if (std::isfinite(s)) most = std::max(most, s);  // scipy's int(max(steps)) fails at inf, NaN
    }
    const std::size_t m = most > 0 ? static_cast<std::size_t>(most) : 0;
    vec thr(k);
    for (std::size_t r = 0; r < k; ++r) thr[r] = b[r] + (inclusive ? 1 * step[r] / 2 : 0 * step[r] / 2);
    ev.start(k, m);
    for_elements(k, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r)
            for (std::size_t j = 0; j < m; ++j) {
                const double x = a[r] + static_cast<double>(j) * step[r];
                ev.put(r, j, x >= thr[r] ? kNaN : x);
            }
    });
    std::vector<std::int64_t> rows(k);
    for (std::size_t r = 0; r < k; ++r) rows[r] = static_cast<std::int64_t>(r);
    ev.run(rows.data());
    Sum o{vec(k), vec(k), ivec(k, 0), ivec(k, 0)};
    for_elements(k, [&](std::size_t r0, std::size_t r1) {
        vec fs(m), t(m);
        for (std::size_t r = r0; r < r1; ++r) {
            std::int64_t cnt = 0;
            for (std::size_t j = 0; j < m; ++j) {
                const double x = a[r] + static_cast<double>(j) * step[r];
                if (x >= thr[r]) {
                    fs[j] = zero;
                } else {
                    fs[j] = ev.value(r, j);
                    cnt += 1;
                }
            }
            const double S = lg ? logsumexp(fs.data(), m, t.data()) : np_sum(fs.data(), m);
            o.S[r] = S;
            o.E[r] = lg ? S + std::log(kEps) : kEps * std::abs(S);
            o.nfev[r] = cnt;
        }
    });
    return o;
}

// scipy's _integral_bound: the sum of f(a + j step) for a + j step <= b (b may be infinite) by the
// integral test: tanhsinh's lower bound on the integral from a, the first of a + 2^i step (i <
// log2 max_terms) and a + max_terms step whose term is below the tolerance and decreasing, the
// direct sum up to it, tanhsinh from it and the halves of its term and f(b).
Sum bound(Eval& ev, const vec& a, const vec& b, const vec& step, const Options& o, const vec& n_steps) {
    const std::size_t k = a.size(), L = n_steps.size();
    const bool lg = o.log;
    const double zero = lg ? -kInf : 0.0, log2 = std::log(2.0);
    const TsOut lb = tanhsinh_run(ev, a, b, o);
    vec tol(k);
    std::vector<std::uint8_t> skip(k);
    ivec status = lb.status;
    for (std::size_t r = 0; r < k; ++r) {
        tol[r] = lg ? logsumexp2(o.atol, o.rtol + lb.integral[r]) : o.atol + o.rtol * lb.integral[r];
        skip[r] = lb.status[r] == sValueErr;
        if (skip[r]) tol[r] = kNaN;
    }
    std::vector<std::int64_t> rows(k);
    for (std::size_t r = 0; r < k; ++r) rows[r] = static_cast<std::int64_t>(r);
    // f at a + n step and at the next term (is it decreasing?)
    vec ks(k * L), fks(k * L);
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t j = 0; j < L; ++j) ks[r * L + j] = a[r] + n_steps[j] * step[r];
    ev.start(k, L);
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t j = 0; j < L; ++j) ev.put(r, j, ks[r * L + j]);
    ev.run(rows.data());
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t j = 0; j < L; ++j) fks[r * L + j] = ev.value(r, j);
    ev.start(k, L);
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t j = 0; j < L; ++j) ev.put(r, j, ks[r * L + j] + step[r]);
    ev.run(rows.data());
    vec kk(k), fk(k);
    std::vector<std::uint8_t> short_(k);
    for (std::size_t r = 0; r < k; ++r) {
        std::size_t cnt = 0;
        for (std::size_t j = 0; j < L; ++j) {
            const double f0 = fks[r * L + j], f1 = ev.value(r, j);
            cnt += (f0 > tol[r]) || (f1 > f0);
        }
        const std::size_t nt = std::min(cnt, L - 1);
        short_[r] = cnt == L;
        kk[r] = a[r] + n_steps[nt] * step[r];
        fk[r] = fks[r * L + nt];
    }
    const Sum left = direct(ev, a, kk, step, false, lg);
    for (std::size_t r = 0; r < k; ++r) {
        if (std::isinf(left.S[r]) && left.S[r] > 0) {
            skip[r] = 1;
            status[r] = sValueErr;
        }
        if (skip[r]) kk[r] = kNaN;
    }
    const TsOut right = tanhsinh_run(ev, kk, b, o);
    // f(b) where b is finite (no call when there is none)
    vec fb(k, zero);
    std::vector<std::int64_t> fin;
    for (std::size_t r = 0; r < k; ++r)
        if (std::isfinite(b[r])) fin.push_back(static_cast<std::int64_t>(r));
    if (!fin.empty()) {
        ev.start(fin.size());
        for (std::size_t q = 0; q < fin.size(); ++q) ev.put(q, 0, b[static_cast<std::size_t>(fin[q])]);
        ev.run(fin.data());
        for (std::size_t q = 0; q < fin.size(); ++q) fb[static_cast<std::size_t>(fin[q])] = ev.value(q, 0);
    }
    Sum out{vec(k), vec(k), ivec(k), ivec(k)};
    for (std::size_t r = 0; r < k; ++r) {
        if (lg) {
            const double ls = std::log(step[r]);
            const double s[4] = {left.S[r], right.integral[r] - ls, fk[r] - log2, fb[r] - log2};
            double t[4];
            out.S[r] = logsumexp(s, 4, t);
            const CPair e[4] = {
                {left.E[r], 0.0}, {right.error[r] - ls, 0.0}, {fk[r] - log2, 0.0}, {fb[r] - log2 + 0.0, 0.0 + M_PI}
            };
            out.E[r] = logsumexp_re(e, 4);
        } else {
            out.S[r] = left.S[r] + right.integral[r] / step[r] + fk[r] / 2 + fb[r] / 2;
            out.E[r] = left.E[r] + right.error[r] / step[r] + fk[r] / 2 - fb[r] / 2;
        }
        std::int64_t st = skip[r] ? status[r] : right.status[r];
        if (st == sConverged && short_[r]) st = -4;
        out.status[r] = st;
        out.nfev[r] = left.nfev[r] + right.nfev[r] + static_cast<std::int64_t>(2 * L) + (std::isfinite(b[r]) ? 1 : 0) +
                      lb.nfev[r];
    }
    return out;
}

// A branch's elements (in their order) and their values.
struct Group {
    std::vector<std::int64_t> elems;
    vec a, b, step;
    void add(std::size_t i, double ai, double bi, double si) {
        elems.push_back(static_cast<std::int64_t>(i));
        a.push_back(ai);
        b.push_back(bi);
        step.push_back(si);
    }
};

// ---- the bindings

Options options(bool log, std::int64_t max_level, std::int64_t min_level, double atol, double rtol) {
    return {log, static_cast<int>(max_level), static_cast<int>(min_level), atol, rtol};
}

nb::object tanhsinh_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle args,
    nb::handle a,
    nb::handle b,
    bool log,
    std::int64_t max_level,
    std::int64_t min_level,
    double atol,
    double rtol
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec av = read(a), bv = read(b);
        Call F(f, check, args, av.size());
        Eval ev(F, kPlain, log, nullptr);
        const TsOut o = tanhsinh_run(ev, av, bv, options(log, max_level, min_level, atol, rtol));
        return nb::make_tuple(out(o.integral), out(o.error), out(o.status), out(o.nfev), out(o.level));
    });
}

nb::object nsum_solve(
    const std::string& func,
    nb::callable f,
    nb::callable check,
    nb::handle args,
    nb::handle a,
    nb::handle b,
    nb::handle step,
    bool log,
    nb::handle n_steps,
    std::int64_t max_terms,
    double atol,
    double rtol
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const vec av = read(a), b0 = read(b), sv = read(step), ns = read(n_steps);
        const std::size_t n = av.size();
        const double zero = log ? -kInf : 0.0;
        Call F(f, check, args, n);
        vec fa;
        F.all(av, fa);  // scipy's first evaluation (eim._initialize): f(a) at every element
        Sum o{vec(n), vec(n), ivec(n, 0), ivec(n, 1)};
        Group g1, g2, g3, g4;
        for (std::size_t i = 0; i < n; ++i) {
            const double ai = av[i], si = sv[i];
            const bool valid = b0[i] >= ai && std::isfinite(si) && si > 0;
            const double nterms = std::floor((b0[i] - ai) / si);
            const double bi = std::isfinite(nterms) ? ai + nterms * si : b0[i];
            if (!valid) {  // -1; b < a is the empty sum
                const bool empty = bi < ai;
                o.S[i] = o.E[i] = empty ? zero : kNaN;
                o.status[i] = empty ? 0 : sSignErr;
            } else if (nterms + 1 <= static_cast<double>(max_terms)) {
                g1.add(i, ai, bi, si);
            } else if (std::isfinite(ai)) {
                g2.add(i, ai, bi, si);
            } else if (std::isfinite(bi)) {
                g3.add(i, -bi, -ai, si);
            } else {
                g4.add(i, 0.0, bi, si);
            }
        }
        const Options opt = options(log, 10, 2, atol, rtol);  // scipy's tanhsinh defaults
        auto put = [&](const Group& g, const Sum& s, std::int64_t times) {
            for (std::size_t r = 0; r < g.elems.size(); ++r) {
                const auto i = static_cast<std::size_t>(g.elems[r]);
                o.S[i] = s.S[r];
                o.E[i] = s.E[r];
                o.status[i] = s.status[r];
                o.nfev[i] += times * s.nfev[r];
            }
        };
        if (!g1.elems.empty()) {
            Eval ev(F, kPlain, log, g1.elems.data());
            Sum s = direct(ev, g1.a, g1.b, g1.step, true, log);
            for (std::size_t r = 0; r < s.S.size(); ++r) s.status[r] = std::isfinite(s.S[r]) ? 0 : sValueErr;
            put(g1, s, 1);
        }
        if (!g2.elems.empty()) {
            Eval ev(F, kPlain, log, g2.elems.data());
            put(g2, bound(ev, g2.a, g2.b, g2.step, opt, ns), 1);
        }
        if (!g3.elems.empty()) {
            Eval ev(F, kNegated, log, g3.elems.data());
            put(g3, bound(ev, g3.a, g3.b, g3.step, opt, ns), 1);
        }
        if (!g4.elems.empty()) {
            Eval ev(F, kBoth, log, g4.elems.data());
            put(g4, bound(ev, g4.a, g4.b, g4.step, opt, ns), 2);
        }
        return nb::make_tuple(out(o.S), out(o.E), out(o.status), out(o.nfev));
    });
}

// The pairs of levels 0 ... level (scipy's pair cache): xjc, wj and the base step h0.
nb::object tanhsinh_table(std::int64_t level) {
    const Table& T = table(static_cast<int>(level));
    vec x, w;
    for (std::int64_t k = 0; k <= level; ++k) {
        const auto u = static_cast<std::size_t>(k);
        x.insert(x.end(), T.xjc[u].begin(), T.xjc[u].end());
        w.insert(w.end(), T.wj[u].begin(), T.wj[u].end());
    }
    return nb::make_tuple(out(x), out(w), T.h0);
}

}  // namespace

void bind_tanhsinh(nb::module_& m) {
    m.def(
        "tanhsinh_solve", &tanhsinh_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("args"),
        nb::arg("a"), nb::arg("b"), nb::arg("log"), nb::arg("max_level"), nb::arg("min_level"), nb::arg("atol"),
        nb::arg("rtol")
    );
    m.def(
        "nsum_solve", &nsum_solve, nb::arg("func"), nb::arg("f"), nb::arg("check"), nb::arg("args"), nb::arg("a"),
        nb::arg("b"), nb::arg("step"), nb::arg("log"), nb::arg("n_steps"), nb::arg("max_terms"), nb::arg("atol"),
        nb::arg("rtol")
    );
    m.def("tanhsinh_table", &tanhsinh_table, nb::arg("level"));
}

}  // namespace sd::integrate
