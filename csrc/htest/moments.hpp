// The lane moments of sd.htest (docs/design/htest.md §3.1, §3.2): the count, the mean and the
// sum of squared deviations of every lane (one sample of one test in a batch) as
// double-doubles, the zero-variance rule (H20), and a sort of double-double keys.
//
// The means and the sums of squares are double-doubles because the statistics subtract them:
// a mean difference of two samples at 1e8 (t), the group means against the grand mean (F,
// Levene), a variance against the pooled one (Bartlett's T, ~1e-15 when the variances agree
// to 1e-9). Rounded to doubles these lose up to the offset's digits; as double-doubles the
// subtractions are exact to ~2^-100 and the statistics keep their full precision whatever
// their condition.
//
// The work is split into fixed blocks of kBlock elements per lane, run on the pool and
// combined in order, so the bits do not depend on the thread count and one long lane
// (axis=None) runs in parallel too.
#pragma once

#include "array/ndarray.hpp"
#include "array/kernels/simd_sort.hpp"
#include "array/ops/pieces.hpp"
#include "htest/lanes.hpp"
#include "special/dd.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

namespace sd::htest {

namespace dd = special::dd;
using dd::DD;

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();
// sd.stats skew's zero-variance rule (H20): m2 <= (eps mean)^2 is a constant sample
inline constexpr double kEps = 0x1p-52;
inline constexpr std::int64_t kBlock = 8192;

// A 2-D float64 array, one lane per row (any strides).
struct Rows {
    const std::byte* data = nullptr;
    std::int64_t s0 = 0, s1 = 0, lanes = 0, n = 0;
    Rows() = default;
    explicit Rows(const NDArray& a)
        : data(a.data), s0(a.strides[0]), s1(a.strides[1]), lanes(a.shape[0]), n(a.shape[1]) {}
    double at(std::int64_t r, std::int64_t j) const { return *reinterpret_cast<const double*>(data + r * s0 + j * s1); }
};

// A double-double sum: hi the running double sum, lo the rounding errors and the low parts
// added up (Neumaier's sum with low parts). Once hi is not finite the low part means nothing
// and the value is hi.
struct Acc {
    double hi = 0.0, lo = 0.0;
    void add(double h, double l = 0.0) {
        const DD s = dd::two_sum(hi, h);
        hi = s.hi;
        lo += s.lo + l;
    }
    void add(DD v) { add(v.hi, v.lo); }
    void add(const Acc& o) { add(o.hi, o.lo); }
    DD value() const { return std::isfinite(hi) ? dd::two_sum(hi, lo) : DD{hi, 0.0}; }
};

// x = a + b of two doubles, normalized.
inline DD sum_dd(double a, double b) { return dd::two_sum(a, b); }
// |x| of a normalized double-double.
inline DD abs_dd(DD x) { return (x.hi < 0.0 || (x.hi == 0.0 && x.lo < 0.0)) ? DD{-x.hi, -x.lo} : x; }
inline double to_double(DD x) { return x.hi + x.lo; }
// (x - m)^2 as a double-double, x and m double-doubles
inline DD sq_dev(double xh, double xl, DD m) {
    const DD d = dd::two_sum(xh, -m.hi);
    const double dl = d.lo + (xl - m.lo);
    DD sq = dd::two_prod(d.hi, d.hi);
    sq.lo += dl * (2.0 * d.hi + dl);
    return sq;
}

// One lane's moments. bad: a NaN met under nan_policy='propagate' (every result of the lane is
// NaN); n counts the observations used (the NaNs left out under 'omit').
struct Moments {
    std::int64_t n = 0;
    bool bad = false;
    DD mean{kNaN, 0.0};
    DD ss{kNaN, 0.0};  // sum (x - mean)^2
    // sd.stats skew's rule (H20): the central second moment m2 = ss / n at most (eps mean)^2.
    // A sample of one is constant.
    bool constant() const {
        if (n < 1) return false;
        const double m2 = ss.hi / static_cast<double>(n), e = kEps * mean.hi;
        return m2 <= e * e;
    }
    // the sample variance (n - 1), a double
    double var() const { return n > 1 ? to_double(ss) / static_cast<double>(n - 1) : kNaN; }
};

// The moments of a set of double-doubles (one lane, no NaNs), in the order given: on the calling
// thread (the per-lane tests' groups).
inline Moments moments_of(const double* hi, const double* lo, std::int64_t n) {
    Moments m;
    m.n = n;
    if (n == 0) return m;
    Acc s;
    for (std::int64_t j = 0; j < n; ++j) s.add(hi[j], lo ? lo[j] : 0.0);
    m.mean = s.value() / static_cast<double>(n);
    if (!std::isfinite(m.mean.hi)) {
        m.ss = {kNaN, 0.0};
        return m;
    }
    Acc q;
    for (std::int64_t j = 0; j < n; ++j) q.add(sq_dev(hi[j], lo ? lo[j] : 0.0, m.mean));
    m.ss = q.value();
    return m;
}

// ---- exact zeros -----------------------------------------------------------------------------
//
// A correlation whose true Sxy is 0 comes out of the double-double sums as a rounding of ~1e-32,
// an infinite relative error (H161). For data of short dyadic values (every x an integer times
// 2^-k below 2^40, k common: integers, halves, ...) the sums of up to 2^20 of them and of their
// products are exact in 128 bits, and the zero is decided exactly; other data keep the
// double-double value.
inline constexpr std::int64_t kExactZeroN = 1 << 20;

// k with every x[0..n) times 2^k an integer below 2^40 in magnitude (n <= kExactZeroN), else false.
inline bool small_dyadic(const double* x, std::int64_t n, int& k) {
    if (n > kExactZeroN) return false;
    int low = std::numeric_limits<int>::max(), high = std::numeric_limits<int>::min();
    for (std::int64_t j = 0; j < n; ++j) {
        const double v = x[j];
        if (v == 0.0) continue;
        if (!std::isfinite(v)) return false;
        int e;
        const double f = std::frexp(std::fabs(v), &e);  // v = f 2^e, f in [1/2, 1)
        auto mant = static_cast<std::uint64_t>(std::ldexp(f, 53));
        int tz = 0;
        while (!(mant & 1u)) mant >>= 1, ++tz;
        low = std::min(low, e - 53 + tz), high = std::max(high, e);
    }
    if (low == std::numeric_limits<int>::max()) {  // all zeros
        k = 0;
        return true;
    }
    k = -low;
    return high + k <= 40;
}
// x times 2^k as an integer (small_dyadic's k)
inline std::int64_t dyadic_int(double x, int k) { return static_cast<std::int64_t>(std::ldexp(x, k)); }

// ---- scaling ---------------------------------------------------------------------------------
//
// The statistics of H1 do not change when every sample of a test is multiplied by the same
// number. A lane whose values reach 2^500 (squared deviations would overflow) or stay below
// 2^-400 (they would leave the normal doubles) is computed in a unit of 2^e, e the exponent of
// its largest magnitude: the values times 2^-e, exact (a value 2^1074 times smaller than the
// largest becomes 0, far below the sums' rounding). Estimates and standard errors go back to the
// data's unit at the end.
inline int unit_exponent(double maxabs) {
    if (!(maxabs > 0.0) || !std::isfinite(maxabs)) return 0;
    const int e = std::ilogb(maxabs);
    return (e >= 500 || e <= -400) ? std::max(e, -1000) : 0;
}

// The (lane, block) items of a pass on the pool: body(item, lane, j0, j1). Tasks of about
// kTask elements (more than the pool's wake-up count for a long lane: 1e6 elements are ~60
// tasks), fixed by the shape alone.
inline constexpr std::int64_t kTask = 1 << 14;
template <class F>
void each_block(std::int64_t lanes, std::int64_t n, F&& body) {
    const std::int64_t nb = std::max<std::int64_t>(1, (n + kBlock - 1) / kBlock);
    const std::int64_t per = std::max<std::int64_t>(1, kTask / std::max<std::int64_t>(1, std::min(n, kBlock)));
    for_pieces_woken(lanes * nb, per, lanes * n * 4, [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
        for (std::int64_t it = b0; it < b1; ++it) {
            const std::int64_t r = it / nb, b = it % nb;
            body(it, r, b * kBlock, std::min(n, (b + 1) * kBlock));
        }
    });
}
inline std::int64_t blocks_of(std::int64_t n) { return std::max<std::int64_t>(1, (n + kBlock - 1) / kBlock); }

// max |x| of every lane of a source (NaNs left out) into out (taking the larger of out's value)
template <class Src>
void lane_maxabs(const Src& src, std::int64_t lanes, std::int64_t n, double* out) {
    const std::int64_t nb = blocks_of(n);
    std::vector<double> part(static_cast<std::size_t>(lanes * nb), 0.0);
    each_block(lanes, n, [&](std::int64_t it, std::int64_t r, std::int64_t j0, std::int64_t j1) {
        double m[4] = {0.0, 0.0, 0.0, 0.0};  // (max is exact: four of them give the same value)
        block_max(src, r, j0, j1, m);
        part[static_cast<std::size_t>(it)] = std::max(std::max(m[0], m[1]), std::max(m[2], m[3]));
    });
    for (std::int64_t r = 0; r < lanes; ++r)
        for (std::int64_t b = 0; b < nb; ++b) out[r] = std::max(out[r], part[static_cast<std::size_t>(r * nb + b)]);
}

// Four sums side by side (element j into sum j mod 4, combined in order): one sum is
// latency-bound. The chains depend on j alone, so the bits do not depend on the split.
struct Acc4 {
    Acc c[4];
    void add(std::int64_t j, double h, double l) { c[j & 3].add(h, l); }
    Acc total() const {
        Acc s = c[0];
        s.add(c[1]);
        s.add(c[2]);
        s.add(c[3]);
        return s;
    }
};

// body(j, k) for j in [j0, j1) in order, k = j mod 4 as a constant (std::integral_constant): the
// body's four chains c[k] stay in registers (a chain picked by j & 3 at run time goes through
// memory, ~4 ns an element), the same operations in the same order as one element at a time.
template <class Body>
inline void for4(std::int64_t j0, std::int64_t j1, Body&& body) {
    using K0 = std::integral_constant<int, 0>;
    using K1 = std::integral_constant<int, 1>;
    using K2 = std::integral_constant<int, 2>;
    using K3 = std::integral_constant<int, 3>;
    auto one = [&](std::int64_t j) {
        switch (j & 3) {
            case 0: body(j, K0{}); break;
            case 1: body(j, K1{}); break;
            case 2: body(j, K2{}); break;
            default: body(j, K3{});
        }
    };
    std::int64_t j = j0;
    for (; j < j1 && (j & 3); ++j) one(j);
    for (; j + 4 <= j1; j += 4) {
        body(j, K0{});
        body(j + 1, K1{});
        body(j + 2, K2{});
        body(j + 3, K3{});
    }
    for (; j < j1; ++j) one(j);
}

// ---- one block of a pass: NEON for a contiguous lane, else element by element ------------------
//
// Each is the scalar loop element by element (element j into chain j mod 4, a NaN left out): the
// NEON form holds chains 0 and 1 in one register and 2 and 3 in another and runs the same IEEE
// operations lane by lane (two_sum's six, the FMA of two_prod as vfmaq), a NaN element masked
// out of its chain, so the bits are the scalar loop's. It needs j0 a multiple of 4 (blocks start
// at multiples of kBlock) and a source whose lane is contiguous: contig(src, r, hi, lo, skip) gives
// the lane's values hi[j] + lo[j] (lo null: 0) and whether a NaN element is left out (skip; a
// source that never says NaN adds it), false for any other source or layout.
template <class Src>
inline bool contig(const Src&, std::int64_t, const double*&, const double*&, bool&) {
    return false;
}

namespace neon {
// chains (c[i], c[i + 1]) as registers
inline void load(const Acc4& a, int i, float64x2_t& h, float64x2_t& l) {
    h = float64x2_t{a.c[i].hi, a.c[i + 1].hi}, l = float64x2_t{a.c[i].lo, a.c[i + 1].lo};
}
inline void store(Acc4& a, int i, float64x2_t h, float64x2_t l) {
    a.c[i].hi = vgetq_lane_f64(h, 0), a.c[i + 1].hi = vgetq_lane_f64(h, 1);
    a.c[i].lo = vgetq_lane_f64(l, 0), a.c[i + 1].lo = vgetq_lane_f64(l, 1);
}
// Acc::add(x, y) on the lanes where ok
inline void add(float64x2_t& hi, float64x2_t& lo, float64x2_t x, float64x2_t y, uint64x2_t ok) {
    const float64x2_t s = vaddq_f64(hi, x), bb = vsubq_f64(s, hi);
    const float64x2_t e = vaddq_f64(vsubq_f64(hi, vsubq_f64(s, bb)), vsubq_f64(x, bb));
    const float64x2_t nl = vaddq_f64(lo, vaddq_f64(e, y));
    hi = vbslq_f64(ok, s, hi), lo = vbslq_f64(ok, nl, lo);
}
inline float64x2_t low(const double* xl, std::int64_t j) { return xl ? vld1q_f64(xl + j) : vdupq_n_f64(0.0); }
// the lanes kept: not NaN, or all of them
inline uint64x2_t kept(float64x2_t u, bool skip) { return skip ? vceqq_f64(u, u) : vdupq_n_u64(~0ull); }
}  // namespace neon

// max |h| of the elements (m[j mod 4])
template <class Src>
inline void block_max(const Src& src, std::int64_t r, std::int64_t j0, std::int64_t j1, double (&m)[4]) {
    const double *x, *xl;
    bool skip;
    std::int64_t j = j0;
    if ((j0 & 3) == 0 && contig(src, r, x, xl, skip)) {
        float64x2_t m01 = float64x2_t{m[0], m[1]}, m23 = float64x2_t{m[2], m[3]};
        const float64x2_t zero = vdupq_n_f64(0.0);
        for (; j + 4 <= j1; j += 4) {
            const float64x2_t a = vld1q_f64(x + j), b = vld1q_f64(x + j + 2);
            // (a NaN never wins std::max(m, |h|): masked whatever the source)
            m01 = vmaxq_f64(m01, vbslq_f64(vceqq_f64(a, a), vabsq_f64(a), zero));
            m23 = vmaxq_f64(m23, vbslq_f64(vceqq_f64(b, b), vabsq_f64(b), zero));
        }
        m[0] = vgetq_lane_f64(m01, 0), m[1] = vgetq_lane_f64(m01, 1);
        m[2] = vgetq_lane_f64(m23, 0), m[3] = vgetq_lane_f64(m23, 1);
    }
    for4(j, j1, [&](std::int64_t i, auto k) {
        double h, l;
        if (src.get(r, i, h, l)) m[k] = std::max(m[k], std::fabs(h));
    });
}

// the sum of the elements times f, their count, whether a NaN was met
template <class Src>
inline void block_sum(
    const Src& src, std::int64_t r, std::int64_t j0, std::int64_t j1, double f, Acc4& a, std::int64_t& cnt, bool& nan
) {
    const double *x, *xl;
    bool skip;
    std::int64_t j = j0;
    if ((j0 & 3) == 0 && contig(src, r, x, xl, skip)) {
        float64x2_t h01, l01, h23, l23;
        neon::load(a, 0, h01, l01);
        neon::load(a, 2, h23, l23);
        const float64x2_t vf = vdupq_n_f64(f);
        int64x2_t nc = vdupq_n_s64(0);
        uint64x2_t all = vdupq_n_u64(~0ull);
        for (; j + 4 <= j1; j += 4) {
            const float64x2_t u = vld1q_f64(x + j), v = vld1q_f64(x + j + 2);
            const uint64x2_t oku = neon::kept(u, skip), okv = neon::kept(v, skip);
            neon::add(h01, l01, vmulq_f64(u, vf), vmulq_f64(neon::low(xl, j), vf), oku);
            neon::add(h23, l23, vmulq_f64(v, vf), vmulq_f64(neon::low(xl, j + 2), vf), okv);
            nc = vsubq_s64(nc, vreinterpretq_s64_u64(oku));
            nc = vsubq_s64(nc, vreinterpretq_s64_u64(okv));
            all = vandq_u64(all, vandq_u64(oku, okv));
        }
        neon::store(a, 0, h01, l01);
        neon::store(a, 2, h23, l23);
        cnt += vgetq_lane_s64(nc, 0) + vgetq_lane_s64(nc, 1);
        if (!(vgetq_lane_u64(all, 0) & vgetq_lane_u64(all, 1))) nan = true;
    }
    for4(j, j1, [&](std::int64_t i, auto k) {
        double h, l;
        if (!src.get(r, i, h, l)) {
            nan = true;
            return;
        }
        a.c[k].add(h * f, l * f);
        ++cnt;
    });
}

// the sum of the squared deviations sq_dev(h f, l f, mean)
template <class Src>
inline void block_ss(const Src& src, std::int64_t r, std::int64_t j0, std::int64_t j1, double f, DD mean, Acc4& a) {
    const double *x, *xl;
    bool skip;
    std::int64_t j = j0;
    if ((j0 & 3) == 0 && contig(src, r, x, xl, skip)) {
        float64x2_t h01, l01, h23, l23;
        neon::load(a, 0, h01, l01);
        neon::load(a, 2, h23, l23);
        const float64x2_t vf = vdupq_n_f64(f), nmh = vdupq_n_f64(-mean.hi), mlo = vdupq_n_f64(mean.lo);
        const float64x2_t two = vdupq_n_f64(2.0);
        // sq_dev(x f, l f, mean) as (p, q)
        auto sq = [&](float64x2_t xv, float64x2_t lv, float64x2_t& p, float64x2_t& q) {
            const float64x2_t xh = vmulq_f64(xv, vf), xlf = vmulq_f64(lv, vf);
            const float64x2_t s = vaddq_f64(xh, nmh), bb = vsubq_f64(s, xh);
            const float64x2_t e = vaddq_f64(vsubq_f64(xh, vsubq_f64(s, bb)), vsubq_f64(nmh, bb));
            const float64x2_t dl = vaddq_f64(e, vsubq_f64(xlf, mlo));
            p = vmulq_f64(s, s);
            const float64x2_t pl = vfmaq_f64(vnegq_f64(p), s, s);
            q = vaddq_f64(pl, vmulq_f64(dl, vaddq_f64(vmulq_f64(two, s), dl)));
        };
        for (; j + 4 <= j1; j += 4) {
            const float64x2_t u = vld1q_f64(x + j), v = vld1q_f64(x + j + 2);
            float64x2_t pu, qu, pv, qv;
            sq(u, neon::low(xl, j), pu, qu);
            sq(v, neon::low(xl, j + 2), pv, qv);
            neon::add(h01, l01, pu, qu, neon::kept(u, skip));
            neon::add(h23, l23, pv, qv, neon::kept(v, skip));
        }
        neon::store(a, 0, h01, l01);
        neon::store(a, 2, h23, l23);
    }
    for4(j, j1, [&](std::int64_t i, auto k) {
        double h, l;
        if (src.get(r, i, h, l)) {
            const DD q = sq_dev(h * f, l * f, mean);
            a.c[k].add(q.hi, q.lo);
        }
    });
}

// The moments of every lane of a source: src.get(r, j, h, l) sets element j of lane r to
// h + l and returns false for a NaN element (left out; the lane is bad unless omit). Two
// passes over (lane, block) items on the pool, combined in order. unit: per-lane exponents e
// (the values taken as x 2^-e; unit_exponent), or null.
template <class Src>
void lane_moments(
    const Src& src, std::int64_t lanes, std::int64_t n, bool omit, Moments* out, const int* unit = nullptr
) {
    const std::int64_t nb = blocks_of(n);
    const std::int64_t items = lanes * nb;
    struct Part {
        Acc s;
        std::int64_t cnt = 0;
        bool nan = false;
    };
    std::vector<Part> p1(static_cast<std::size_t>(items));
    std::vector<Acc> p2(static_cast<std::size_t>(items));
    auto factor = [&](std::int64_t r) { return unit && unit[r] ? std::ldexp(1.0, -unit[r]) : 1.0; };
    each_block(lanes, n, [&](std::int64_t it, std::int64_t r, std::int64_t j0, std::int64_t j1) {
        const double f = factor(r);
        Acc4 a;
        std::int64_t cnt = 0;
        bool nan = false;
        block_sum(src, r, j0, j1, f, a, cnt, nan);
        p1[static_cast<std::size_t>(it)] = {a.total(), cnt, nan};
    });
    for (std::int64_t r = 0; r < lanes; ++r) {
        Acc s;
        Moments& m = out[r];
        m = Moments{};
        for (std::int64_t b = 0; b < nb; ++b) {
            const Part& q = p1[static_cast<std::size_t>(r * nb + b)];
            s.add(q.s);
            m.n += q.cnt;
            m.bad = m.bad || q.nan;
        }
        m.bad = m.bad && !omit;
        if (m.n > 0 && !m.bad) m.mean = s.value() / static_cast<double>(m.n);
    }
    each_block(lanes, n, [&](std::int64_t it, std::int64_t r, std::int64_t j0, std::int64_t j1) {
        const Moments& m = out[r];
        if (m.bad || m.n == 0 || !std::isfinite(m.mean.hi)) return;
        const double f = factor(r);
        const DD mean = m.mean;
        Acc4 a;
        block_ss(src, r, j0, j1, f, mean, a);
        p2[static_cast<std::size_t>(it)] = a.total();
    });
    for (std::int64_t r = 0; r < lanes; ++r) {
        Moments& m = out[r];
        if (m.bad || m.n == 0 || !std::isfinite(m.mean.hi)) {
            m.ss = {kNaN, 0.0};
            continue;
        }
        Acc q;
        for (std::int64_t b = 0; b < nb; ++b) q.add(p2[static_cast<std::size_t>(r * nb + b)]);
        m.ss = q.value();
    }
}

// The sum (in order, one Acc) of term(i) for i in [0, n): the terms formed on the pool when the
// lane is long (cost: one term's work in elements, roughly), then added in order, so the bits are
// those of the plain loop on any thread count.
template <class Term>
Acc ordered_sum(std::int64_t n, std::int64_t cost, Term&& term) {
    Acc s;
    if (n * cost < kParallelThreshold || n < 2 * kTask || !parallel::available()) {
        for (std::int64_t i = 0; i < n; ++i) s.add(term(i));
        return s;
    }
    std::vector<double> t(static_cast<std::size_t>(n));
    for_pieces(n, kTask, n * cost, [&](std::int64_t, std::int64_t b, std::int64_t e) {
        for (std::int64_t i = b; i < e; ++i) t[static_cast<std::size_t>(i)] = term(i);
    });
    for (double v : t) s.add(v);
    return s;
}

// The sources: a sample's rows, and paired rows' differences a - b as double-doubles
// (ttest_rel: the difference of two large values is exact, not rounded to a double).
struct Plain {
    Rows a;
    bool get(std::int64_t r, std::int64_t j, double& h, double& l) const {
        h = a.at(r, j);
        l = 0.0;
        return h == h;
    }
};
inline bool contig(const Plain& p, std::int64_t r, const double*& hi, const double*& lo, bool& skip) {
    if (p.a.s1 != static_cast<std::int64_t>(sizeof(double))) return false;
    hi = reinterpret_cast<const double*>(p.a.data + r * p.a.s0), lo = nullptr, skip = true;
    return true;
}
struct Diff {
    Rows a, b;
    bool get(std::int64_t r, std::int64_t j, double& h, double& l) const {
        const double x = a.at(r, j), y = b.at(r, j);
        if (x != x || y != y) return false;
        const DD d = dd::two_sum(x, -y);
        h = d.hi;
        l = std::isfinite(d.hi) ? d.lo : 0.0;
        return true;
    }
};

// One lane of double-doubles (hi[j] + lo[j], no NaNs; lo may be null): levene's deviations.
struct Vec {
    const double *hi, *lo;
    bool get(std::int64_t, std::int64_t j, double& h, double& l) const {
        h = hi[j];
        l = lo ? lo[j] : 0.0;
        return true;
    }
};
inline bool contig(const Vec& v, std::int64_t, const double*& hi, const double*& lo, bool& skip) {
    hi = v.hi, lo = v.lo, skip = false;
    return true;
}

// A lane's values without its NaNs (all of them under propagate: the caller has checked).
inline bool gather(const Rows& a, std::int64_t r, bool omit, std::vector<double>& v) {
    v.clear();
    bool nan = false;
    for (std::int64_t j = 0; j < a.n; ++j) {
        const double x = a.at(r, j);
        if (x != x) {
            nan = true;
            if (omit) continue;
        }
        v.push_back(x);
    }
    return nan && !omit;
}

// A double-double key with a tag (fligner: the group), ordered by value.
struct Key {
    double hi, lo;
    std::int32_t tag;
};
inline bool key_less(const Key& a, const Key& b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline bool key_equal(const Key& a, const Key& b) { return a.hi == b.hi && a.lo == b.lo; }

// v sorted by less on the pool: fixed pieces of kPiece sorted side by side, then merged pairwise,
// each merge cut into output segments of kPiece by the merge path (the segment's co-ranks in the
// two runs; std::merge's rule, the first run's element first among equals), so every merge, the
// last one too, runs in parallel and gives exactly std::merge's output. The pieces depend on the
// length alone, so the order is the same on any thread count.
template <class T, class Less, class Sort>
void sort_on_pool(std::vector<T>& v, Less less, Sort&& sort) {
    constexpr std::int64_t kPiece = 1 << 15;
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    if (n <= kPiece) {
        sort(v.data(), n);
        return;
    }
    const std::int64_t pieces = (n + kPiece - 1) / kPiece;
    for_pieces_woken(pieces, 1, n * 16, [&](std::int64_t, std::int64_t p0, std::int64_t p1) {
        for (std::int64_t p = p0; p < p1; ++p) sort(v.data() + p * kPiece, std::min(n, (p + 1) * kPiece) - p * kPiece);
    });
    std::vector<T> buf(v.size());
    struct Seg {
        std::int64_t lo, mid, hi, k0, k1;
    };
    std::vector<Seg> segs;
    for (std::int64_t w = kPiece; w < n; w *= 2) {
        segs.clear();
        for (std::int64_t lo = 0; lo < n; lo += 2 * w) {
            const std::int64_t mid = std::min(n, lo + w), hi = std::min(n, lo + 2 * w);
            for (std::int64_t k = 0; k < hi - lo; k += kPiece)
                segs.push_back({lo, mid, hi, k, std::min(hi - lo, k + kPiece)});
        }
        const T* src = v.data();
        T* dst = buf.data();
        // the number of the first run's elements among the merge's first k
        auto corank = [&](const Seg& g, std::int64_t k) {
            const T* a = src + g.lo;
            const T* b = src + g.mid;
            const std::int64_t m = g.mid - g.lo, nb = g.hi - g.mid;
            std::int64_t i0 = std::max<std::int64_t>(0, k - nb), i1 = std::min(k, m);
            while (i0 < i1) {
                const std::int64_t i = i0 + (i1 - i0) / 2, j = k - i;  // a[0, i) and b[0, j)
                if (j > 0 && i < m && !less(b[j - 1], a[i])) i0 = i + 1;  // a[i] comes before b[j - 1]
                else i1 = i;
            }
            return i0;
        };
        for_pieces_woken(
            static_cast<std::int64_t>(segs.size()), 1, n * 4, [&](std::int64_t, std::int64_t q0, std::int64_t q1) {
                for (std::int64_t q = q0; q < q1; ++q) {
                    const Seg& g = segs[static_cast<std::size_t>(q)];
                    const std::int64_t i0 = corank(g, g.k0), i1 = corank(g, g.k1);
                    const std::int64_t j0 = g.k0 - i0, j1 = g.k1 - i1;
                    std::merge(
                        src + g.lo + i0, src + g.lo + i1, src + g.mid + j0, src + g.mid + j1, dst + g.lo + g.k0, less
                    );
                }
            }
        );
        v.swap(buf);
    }
}

// v sorted by value on the pool (sort_on_pool; equal keys are equal values: the ranks do not
// depend on their order anyway).
inline void sort_keys(std::vector<Key>& v) {
    sort_on_pool(v, key_less, [](Key* p, std::int64_t m) { std::sort(p, p + m, key_less); });
}

// v (no NaNs) sorted ascending: the array area's NEON sort (array/kernels/simd_sort.hpp) of each
// piece. Equal doubles are equal values, but -0 and +0, whose order std::sort leaves open too
// (here -0 first within a piece): every use here gives them the same value.
inline void sort_values(std::vector<double>& v) {
    sort_on_pool(v, std::less<double>{}, [](double* p, std::int64_t m) { simd_sort::sort<simd_sort::F64>(p, m); });
}

}  // namespace sd::htest
