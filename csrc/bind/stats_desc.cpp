// The descriptive statistics' lane kernels (T8, docs/design/stats.md §5): "desc_*" in
// stardust._core.stats. The Python side (stardust.stats._descriptive) moves the reduced
// axes last and passes a 2-D float64 array, one lane per row (any strides); every kernel
// works lane by lane on the pool (lanes are independent: any split gives the same bits).
//
// NaN: omit skips them (each lane its own count); otherwise a lane with a NaN gives NaN
// (the caller has raised already for nan_policy='raise'). Sums are Neumaier-compensated and
// the moments two-pass (the mean, corrected by the mean of the residuals, then sum (x - m)^k).
#include "bind/stats_bind.hpp"

#include "array/convert.hpp"
#include "array/ops/pieces.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace nb = nanobind;

namespace sd {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// A compensated sum. Once the sum is not finite (an infinite term, or an overflow) the
// correction is inf - inf = NaN and means nothing: the plain sum is the value (inf, or NaN for
// both infinities), as sd.sum's.
struct Neumaier {
    double s = 0.0, c = 0.0;
    void add(double x) {
        const double t = s + x;
        c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    double value() const { return std::isfinite(s) ? s + c : s; }
};

const NDArray& lanes_arg(nb::handle o) {
    const NDArray& a = require_ndarray(o);
    if (a.dtype != DType::Float64 || a.ndim != 2) throw nb::type_error("desc: expects a 2-D float64 array");
    return a;
}
inline double at(const NDArray& a, std::int64_t r, std::int64_t j) {
    return *reinterpret_cast<const double*>(a.data + r * a.strides[0] + j * a.strides[1]);
}

// Lane r as a vector (without the NaNs when omit); has_nan says whether it had any.
void lane(const NDArray& a, std::int64_t r, bool omit, std::vector<double>& v, bool& has_nan) {
    const std::int64_t n = a.shape[1];
    v.clear();
    has_nan = false;
    for (std::int64_t j = 0; j < n; ++j) {
        const double x = at(a, r, j);
        if (std::isnan(x)) {
            has_nan = true;
            if (omit) continue;
        }
        v.push_back(x);
    }
}

NDArray vec(std::int64_t n, DType t = DType::Float64) {
    const std::int64_t shape[] = {n};
    return new_array(t, shape);
}

template <class F>
void each_lane(std::int64_t lanes, std::int64_t n, F&& f) {
    for_pieces(
        lanes, std::max<std::int64_t>(1, kParallelTask / std::max<std::int64_t>(n, 1)), lanes * n,
        [&](std::int64_t, std::int64_t b, std::int64_t e) {
            std::vector<double> v;
            for (std::int64_t r = b; r < e; ++r) f(r, v);
        }
    );
}

// std::sort of a long vector on the pool (a single lane, axis=None, is otherwise one thread:
// rankdata of 10^6 was 60 ms on any number). A samplesort: 63 splitters from 1024 evenly
// spaced elements, every element to the bucket of the splitters below it (equal elements
// share a bucket) by fixed pieces in their original order, the buckets sorted side by side.
// The pieces and splitters depend on the data alone, so the result does not depend on the
// thread count: it is the sorted order, which is unique up to the order of equal elements
// (-0.0 and 0.0), and that order is fixed too. Short vectors: std::sort as before.
template <class T>
void sort_long(std::vector<T>& v) {
    constexpr std::int64_t kMin = 1 << 16, kBuckets = 64, kOver = 16, kPiece = 1 << 14;
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    if (n < kMin) {
        std::sort(v.begin(), v.end());
        return;
    }
    std::vector<T> sample(static_cast<std::size_t>(kBuckets * kOver));
    for (std::int64_t i = 0; i < kBuckets * kOver; ++i)
        sample[i] = v[static_cast<std::size_t>(i * (n / (kBuckets * kOver)))];
    std::sort(sample.begin(), sample.end());
    T split[kBuckets - 1];
    for (std::int64_t b = 1; b < kBuckets; ++b) split[b - 1] = sample[static_cast<std::size_t>(b * kOver)];
    const std::int64_t pieces = (n + kPiece - 1) / kPiece;
    std::vector<std::uint8_t> id(static_cast<std::size_t>(n));
    std::vector<std::int64_t> count(static_cast<std::size_t>(pieces * kBuckets), 0);
    for_pieces(pieces, 1, n * 8, [&](std::int64_t, std::int64_t p0, std::int64_t p1) {
        for (std::int64_t p = p0; p < p1; ++p) {
            std::int64_t* c = count.data() + p * kBuckets;
            for (std::int64_t i = p * kPiece, e = std::min(n, (p + 1) * kPiece); i < e; ++i) {
                const T& x = v[static_cast<std::size_t>(i)];
                std::int64_t lo = 0;  // the number of splitters below x: a branch-free binary search
                for (std::int64_t step = kBuckets / 2; step > 0; step /= 2) lo += split[lo + step - 1] < x ? step : 0;
                id[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(lo);
                ++c[lo];
            }
        }
    });
    // where piece p's elements of bucket b go: after the bucket's elements of the pieces before
    std::int64_t start[kBuckets + 1];
    std::int64_t at = 0;
    for (std::int64_t b = 0; b < kBuckets; ++b) {
        start[b] = at;
        for (std::int64_t p = 0; p < pieces; ++p) {
            const std::int64_t c = count[static_cast<std::size_t>(p * kBuckets + b)];
            count[static_cast<std::size_t>(p * kBuckets + b)] = at;
            at += c;
        }
    }
    start[kBuckets] = n;
    std::vector<T> out(static_cast<std::size_t>(n));
    for_pieces(pieces, 1, n * 8, [&](std::int64_t, std::int64_t p0, std::int64_t p1) {
        for (std::int64_t p = p0; p < p1; ++p) {
            std::int64_t* c = count.data() + p * kBuckets;
            for (std::int64_t i = p * kPiece, e = std::min(n, (p + 1) * kPiece); i < e; ++i)
                out[static_cast<std::size_t>(c[id[static_cast<std::size_t>(i)]]++)] = v[static_cast<std::size_t>(i)];
        }
    });
    for_pieces(kBuckets, 1, n * 32, [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
        for (std::int64_t b = b0; b < b1; ++b) std::sort(out.begin() + start[b], out.begin() + start[b + 1]);
    });
    v.swap(out);
}

// The mean of v as hi + lo: hi the compensated sum over n, lo the mean of the residuals
// x - hi (compensated). The deviations are then (x - hi) - lo: with the mean rounded to a
// double, a mean of 1e8 is off by up to 7e-9 and that offset enters m3 as 3 offset m2 (a
// skewness 1e-8 off at spread 1).
struct Mean {
    double hi, lo;
    double value() const { return hi + lo; }
    double dev(double x) const { return (x - hi) - lo; }
};
Mean mean_of(const std::vector<double>& v) {
    Neumaier s;
    for (double x : v) s.add(x);
    const double n = static_cast<double>(v.size());
    const double m0 = s.value() / n;
    if (!std::isfinite(m0)) return {m0, 0.0};
    Neumaier r;
    for (double x : v) r.add(x - m0);
    return {m0, r.value() / n};
}

// (count, mean, m2, m3, m4, min, max, mean_lo) per lane: m_k = mean((x - mean)^k) for k up to
// kmax (2..4; the others NaN), min and max when asked; mean_lo the mean's low part (mean -
// its double: zscore subtracts both).
//
// Two passes. Pass 1: count, the sum, min, max, NaN; hi = sum / n. Pass 2: S_k = mean(d^k)
// of d = x - hi, k = 1..kmax. The mean's low part is lo = S_1, and the central moments of
// x - (hi + lo) follow without a third pass:
//     m2 = S2 - lo^2,  m3 = S3 - 3 lo S2 + 2 lo^3,  m4 = S4 - 4 lo S3 + 6 lo^2 S2 - 3 lo^4.
//
// Every sum is cut the same way, by the position j in the lane alone: sub-blocks of kSub
// elements, each summed in eight chains (j mod 8, 16 terms each, in order from 0.0); the
// sub-blocks' results added with Neumaier's compensation within blocks of kBlock elements;
// the blocks' (sum, correction) pairs added the same way in order. Two kinds of chains:
//   - plain adds, joined along a fixed tree (tree8): the sum of x in pass 1 (hi need not be
//     accurate: the deviations' mean lo corrects it) and the even powers d^2, d^4 (terms of
//     one sign: a sub-block is off by at most ~16 units of its sum);
//   - TwoSum chains (the error of every addition kept, as Knuth's), whose eight sums and eight
//     corrections go into the block's Neumaier sum (fold16): the odd powers d, d^3, whose sums
//     cancel (the mean of data around 0, a skewness near 0) and keep every digit there.
// The plain chains replaced a TwoSum per element in three of the five sums. The partition
// depends on n alone: the blocks can run on any threads, and the lanes laid out side by side
// (side_*), alone, strided or not, give the same bits.
//
// Short lanes (one block) run whole in one task, without per-block records.
constexpr std::int64_t kBlock = 8192;  // a multiple of kSub
constexpr std::int64_t kSub = 128;

inline double tree8(const double* s) { return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7])); }

// x into chain (s, c) by TwoSum: c gathers each addition's exact error
inline void two_add(double& s, double& c, double x) {
    const double t = s + x, bb = t - s;
    c += (s - (t - bb)) + (x - bb);
    s = t;
}
// A sub-block's eight TwoSum chains (sums s, corrections c) as hi + lo: hi = tree8(s) with the
// error of each of its additions kept (TwoSum), lo = those errors and tree8(c).
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline void comb8(const double* s, const double* c, double& hi, double& lo) {
    double a[4], e[4], b[2], f[2], g;
    for (int q = 0; q < 4; ++q) two_sum(s[2 * q], s[2 * q + 1], a[q], e[q]);
    two_sum(a[0], a[1], b[0], f[0]), two_sum(a[2], a[3], b[1], f[1]);
    two_sum(b[0], b[1], hi, g);
    lo = ((((e[0] + e[1]) + (e[2] + e[3])) + (f[0] + f[1])) + g) + tree8(c);
}
// ... into the block's sum: hi added, lo straight into its correction (where hi is finite: a
// chain that met an infinity has a NaN error)
inline void add_comb(Neumaier& acc, const double* s, const double* c) {
    double hi, lo;
    comb8(s, c, hi, lo);
    acc.add(hi);
    if (std::isfinite(hi)) acc.c += lo;
}
// Which sums are TwoSum chains: bit m of comp for sums[m], the sum of d^(m + 1). The caller
// asks for those whose error a result amplifies: d for the mean and the z-scores, d and d^3
// for m3 (skew, moment), d^2 and d^4 for the excess kurtosis m4 / m2^2 - 3; the others are plain
// chains. comp is one of kCompSets (the pair kernels are compiled for those).
constexpr bool comp_of(int m, int comp) { return (comp >> m) & 1; }
inline constexpr int kCompSets[] = {0, 1, 5, 10, 15};
// f.template operator()<K>() for kmax (2..4) as a constant
template <class F>
inline void with_k(int kmax, F&& f) {
    if (kmax == 2) f.template operator()<2>();
    else if (kmax == 3) f.template operator()<3>();
    else f.template operator()<4>();
}
// f.template operator()<C>() for comp's set as a constant
template <class F>
inline void with_comp(int comp, F&& f) {
    switch (comp) {
        case 0: f.template operator()<0>(); break;
        case 1: f.template operator()<1>(); break;
        case 5: f.template operator()<5>(); break;
        case 10: f.template operator()<10>(); break;
        default: f.template operator()<15>(); break;
    }
}

// The blocks' (sum, correction) pairs in order: the lane's sum. A block whose sum is not finite
// has a meaningless correction (NaN): left out, as Neumaier::value does.
inline double join(const Neumaier* parts, std::int64_t nb, std::int64_t step) {
    Neumaier t;
    for (std::int64_t b = 0; b < nb; ++b) {
        const Neumaier& p = parts[b * step];
        t.add(p.s);
        if (std::isfinite(p.s)) t.add(p.c);
    }
    return t.value();
}

inline double value_of(const Neumaier& s) { return join(&s, 1, 1); }

// The sums of one block [j0, j1) of a lane, sub-block by sub-block. pass 1: sums[0] of x (NaN
// as 0), the NaN count, min and max (when asked, NaNs left out). pass 2: sums[k - 1] of d^k,
// d = x - h (0 for a NaN), k = 1..kmax.
struct Lane {
    const char* base;
    std::int64_t step;  // bytes between elements
    double x(std::int64_t j) const { return *reinterpret_cast<const double*>(base + j * step); }
};

inline void pass1_scalar(
    const Lane& L,
    std::int64_t j0,
    std::int64_t j1,
    Neumaier& sum,
    std::int64_t& bad,
    bool minmax,
    double& lo,
    double& hi
) {
    for (std::int64_t s0 = j0; s0 < j1; s0 += kSub) {
        const std::int64_t s1 = std::min(j1, s0 + kSub);
        double ch[8] = {};
        for (std::int64_t j = s0; j < s1; ++j) {
            const double x = L.x(j);
            const bool nan = std::isnan(x);
            bad += nan;
            ch[j & 7] += nan ? 0.0 : x;
            if (minmax && !nan) lo = std::min(lo, x), hi = std::max(hi, x);
        }
        sum.add(tree8(ch));
    }
}

inline void pass2_scalar(
    const Lane& L, std::int64_t j0, std::int64_t j1, double h, int kmax, int comp, Neumaier* sums
) {
    for (std::int64_t s0 = j0; s0 < j1; s0 += kSub) {
        const std::int64_t s1 = std::min(j1, s0 + kSub);
        double c[4][8] = {}, e[4][8] = {};  // e: the corrections of the TwoSum chains
        for (std::int64_t j = s0; j < s1; ++j) {
            const double x = L.x(j);
            const double d = std::isnan(x) ? 0.0 : x - h, d2 = d * d;
            const int k = static_cast<int>(j & 7);
            const double p[4] = {d, d2, d2 * d, d2 * d2};
            for (int m = 0; m < kmax; ++m)
                if (comp_of(m, comp)) two_add(c[m][k], e[m][k], p[m]);
                else c[m][k] += p[m];
        }
        for (int m = 0; m < kmax; ++m)
            if (comp_of(m, comp)) add_comb(sums[m], c[m], e[m]);
            else sums[m].add(tree8(c[m]));
    }
}

#if defined(__aarch64__)
// The NEON kernels take M (mask): false sums the NaNs too, for data without them -- the same
// bits, as x + 0.0 and the masked 0 add alike where there is no NaN. The callers run M = false
// first and again with M = true only where a sum came out NaN (a NaN in the data, or inf - inf),
// or for pass 2 where the lane had NaNs (omit).
// two_add and Neumaier's add in lanes (the same operations per lane)
inline void two_add(float64x2_t& s, float64x2_t& c, float64x2_t x) {
    const float64x2_t t = vaddq_f64(s, x), bb = vsubq_f64(t, s);
    c = vaddq_f64(c, vaddq_f64(vsubq_f64(s, vsubq_f64(t, bb)), vsubq_f64(x, bb)));
    s = t;
}
inline uint64x2_t finite2(float64x2_t x) { return vcltq_f64(vabsq_f64(x), vdupq_n_f64(kInf)); }
struct NeuV {
    float64x2_t s = vdupq_n_f64(0.0), c = vdupq_n_f64(0.0);
    void add(float64x2_t x) {
        const float64x2_t t = vaddq_f64(s, x);
        const uint64x2_t big = vcgeq_f64(vabsq_f64(s), vabsq_f64(x));
        c = vaddq_f64(c, vbslq_f64(big, vaddq_f64(vsubq_f64(s, t), x), vaddq_f64(vsubq_f64(x, t), s)));
        s = t;
    }
    void add_if(float64x2_t x, uint64x2_t m) {  // add(x) in the lanes of m only
        NeuV n = *this;
        n.add(x);
        s = vbslq_f64(m, n.s, s), c = vbslq_f64(m, n.c, c);
    }
    float64x2_t value() const { return vbslq_f64(finite2(s), vaddq_f64(s, c), s); }
};
inline void two_sum(float64x2_t a, float64x2_t b, float64x2_t& sv, float64x2_t& e) {
    sv = vaddq_f64(a, b);
    const float64x2_t bb = vsubq_f64(sv, a);
    e = vaddq_f64(vsubq_f64(a, vsubq_f64(sv, bb)), vsubq_f64(b, bb));
}
// add_comb into a fresh block sum, then join() of that one block, per lane: a one-sub-block
// lane's compensated sum
inline float64x2_t comb_join(const float64x2_t* s, const float64x2_t* c) {
    float64x2_t a[4], e[4], b[2], f[2], g, hi;
    for (int q = 0; q < 4; ++q) two_sum(s[2 * q], s[2 * q + 1], a[q], e[q]);
    two_sum(a[0], a[1], b[0], f[0]), two_sum(a[2], a[3], b[1], f[1]);
    two_sum(b[0], b[1], hi, g);
    const float64x2_t ct = vaddq_f64(
        vaddq_f64(vaddq_f64(c[0], c[1]), vaddq_f64(c[2], c[3])), vaddq_f64(vaddq_f64(c[4], c[5]), vaddq_f64(c[6], c[7]))
    );
    const float64x2_t lo =
        vaddq_f64(vaddq_f64(vaddq_f64(vaddq_f64(e[0], e[1]), vaddq_f64(e[2], e[3])), vaddq_f64(f[0], f[1])), g);
    NeuV acc;
    acc.add(hi);
    acc.c = vbslq_f64(finite2(hi), vaddq_f64(acc.c, vaddq_f64(lo, ct)), acc.c);
    NeuV t;
    t.add(acc.s);
    t.add_if(acc.c, finite2(acc.s));
    return t.value();
}

template <bool M>
inline float64x2_t keep(float64x2_t x, float64x2_t v) {  // v, 0 where x is NaN (M)
    if constexpr (M) return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v), vceqq_f64(x, x)));
    return v;
}

// The same operations on contiguous data, the chains (2q, 2q + 1) as the lanes of vector q.
template <bool M>
inline void pass1_contig(const double* xs, std::int64_t j0, std::int64_t j1, Neumaier& sum, std::int64_t& bad) {
    for (std::int64_t s0 = j0; s0 < j1; s0 += kSub) {
        const std::int64_t s1 = std::min(j1, s0 + kSub);
        float64x2_t a0 = vdupq_n_f64(0.0), a1 = a0, a2 = a0, a3 = a0;
        uint64x2_t okc = vdupq_n_u64(0);
        std::int64_t j = s0;
        auto masked = [&](const double* p) {
            const float64x2_t x = vld1q_f64(p);
            if constexpr (M) okc = vsubq_u64(okc, vceqq_f64(x, x));  // all ones where not NaN
            return keep<M>(x, x);
        };
        for (; j + 8 <= s1; j += 8) {
            a0 = vaddq_f64(a0, masked(xs + j)), a1 = vaddq_f64(a1, masked(xs + j + 2));
            a2 = vaddq_f64(a2, masked(xs + j + 4)), a3 = vaddq_f64(a3, masked(xs + j + 6));
        }
        if constexpr (M) bad += (j - s0) - static_cast<std::int64_t>(vgetq_lane_u64(okc, 0) + vgetq_lane_u64(okc, 1));
        alignas(16) double ch[8];
        vst1q_f64(ch, a0), vst1q_f64(ch + 2, a1), vst1q_f64(ch + 4, a2), vst1q_f64(ch + 6, a3);
        for (; j < s1; ++j) {
            const bool nan = M && std::isnan(xs[j]);
            bad += nan;
            ch[j & 7] += nan ? 0.0 : xs[j];
        }
        sum.add(tree8(ch));
    }
}

template <bool M, int COMP, int kmax>
inline void pass2_contig(const double* xs, std::int64_t j0, std::int64_t j1, double h, Neumaier* sums) {
    constexpr bool cm[4] = {comp_of(0, COMP), comp_of(1, COMP), comp_of(2, COMP), comp_of(3, COMP)};
    const float64x2_t hv = vdupq_n_f64(h);
    for (std::int64_t s0 = j0; s0 < j1; s0 += kSub) {
        const std::int64_t s1 = std::min(j1, s0 + kSub);
        float64x2_t c[4][4], e[4][4];
        for (auto& m : c)
            for (auto& v : m) v = vdupq_n_f64(0.0);
        for (auto& m : e)
            for (auto& v : m) v = vdupq_n_f64(0.0);
        std::int64_t j = s0;
        for (; j + 8 <= s1; j += 8)
            for (int q = 0; q < 4; ++q) {
                const float64x2_t x = vld1q_f64(xs + j + 2 * q);
                const float64x2_t d = keep<M>(x, vsubq_f64(x, hv));
                const float64x2_t d2 = vmulq_f64(d, d);
                const float64x2_t p[4] = {d, d2, vmulq_f64(d2, d), vmulq_f64(d2, d2)};
                for (int m = 0; m < kmax; ++m)
                    if (cm[m]) two_add(c[m][q], e[m][q], p[m]);
                    else c[m][q] = vaddq_f64(c[m][q], p[m]);
            }
        alignas(16) double ch[4][8], eh[4][8];
        for (int m = 0; m < kmax; ++m)
            for (int q = 0; q < 4; ++q) vst1q_f64(ch[m] + 2 * q, c[m][q]), vst1q_f64(eh[m] + 2 * q, e[m][q]);
        for (; j < s1; ++j) {
            const double d = M && std::isnan(xs[j]) ? 0.0 : xs[j] - h, d2 = d * d;
            const int k = static_cast<int>(j & 7);
            const double p[4] = {d, d2, d2 * d, d2 * d2};
            for (int m = 0; m < kmax; ++m)
                if (cm[m]) two_add(ch[m][k], eh[m][k], p[m]);
                else ch[m][k] += p[m];
        }
        for (int m = 0; m < kmax; ++m)
            if (cm[m]) add_comb(sums[m], ch[m], eh[m]);
            else sums[m].add(tree8(ch[m]));
    }
}

// Two short contiguous lanes (n <= kSub: one sub-block) at p0 and p1 as the two lanes of each
// vector, the eight chains in registers: the operations of pass1_scalar / pass2_scalar and
// the one-sub-block sums of Once, so the bits of those lanes alone.
inline float64x2_t pair_load(const double* p0, const double* p1, std::int64_t j) {
    return vcombine_f64(vld1_f64(p0 + j), vld1_f64(p1 + j));
}
inline float64x2_t pair_tree(const float64x2_t* c) {
    return vaddq_f64(
        vaddq_f64(vaddq_f64(c[0], c[1]), vaddq_f64(c[2], c[3])), vaddq_f64(vaddq_f64(c[4], c[5]), vaddq_f64(c[6], c[7]))
    );
}
// the sums of x (NaN as 0) + 0.0 into sum[0..1], the NaN counts into bad[0..1]
template <bool M>
inline void pair_pass1(const double* p0, const double* p1, std::int64_t n, double* sum, std::int64_t* bad) {
    float64x2_t c[8];
    for (auto& v : c) v = vdupq_n_f64(0.0);
    uint64x2_t okc = vdupq_n_u64(0);
    auto add = [&](int k, std::int64_t j) {
        const float64x2_t x = pair_load(p0, p1, j);
        if constexpr (M) okc = vsubq_u64(okc, vceqq_f64(x, x));
        c[k] = vaddq_f64(c[k], keep<M>(x, x));
    };
    std::int64_t j = 0;
    for (; j + 8 <= n; j += 8)
        for (int k = 0; k < 8; ++k) add(k, j + k);
    for (int k = 0; k < 8; ++k)
        if (j + k < n) add(k, j + k);
    vst1q_f64(sum, vaddq_f64(pair_tree(c), vdupq_n_f64(0.0)));
    bad[0] = M ? n - static_cast<std::int64_t>(vgetq_lane_u64(okc, 0)) : 0;
    bad[1] = M ? n - static_cast<std::int64_t>(vgetq_lane_u64(okc, 1)) : 0;
}
// the sums of d^k (d = x - h, 0 for a NaN) + 0.0 into S[(k - 1) * 2 + lane], k = 1..K
template <int K, bool M, int COMP>
inline void pair_pass2(const double* p0, const double* p1, std::int64_t n, const double* h, double* S) {
    float64x2_t c[K][8], e[K][8];
    for (auto& m : c)
        for (auto& v : m) v = vdupq_n_f64(0.0);
    for (auto& m : e)
        for (auto& v : m) v = vdupq_n_f64(0.0);
    const float64x2_t hv = vld1q_f64(h);
    auto add = [&](int k, std::int64_t j) {
        const float64x2_t x = pair_load(p0, p1, j);
        const float64x2_t d = keep<M>(x, vsubq_f64(x, hv));
        const float64x2_t d2 = vmulq_f64(d, d);
        float64x2_t p[4];
        p[0] = d, p[1] = d2;
        if constexpr (K >= 3) p[2] = vmulq_f64(d2, d);
        if constexpr (K >= 4) p[3] = vmulq_f64(d2, d2);
#pragma clang loop unroll(full)
        for (int m = 0; m < K; ++m)
            if (comp_of(m, COMP)) two_add(c[m][k], e[m][k], p[m]);
            else c[m][k] = vaddq_f64(c[m][k], p[m]);
    };
    std::int64_t j = 0;
    for (; j + 8 <= n; j += 8)
        for (int k = 0; k < 8; ++k) add(k, j + k);
    for (int k = 0; k < 8; ++k)
        if (j + k < n) add(k, j + k);
    for (int m = 0; m < K; ++m)
        vst1q_f64(S + 2 * m, comp_of(m, COMP) ? comb_join(c[m], e[m]) : vaddq_f64(pair_tree(c[m]), vdupq_n_f64(0.0)));
}

// Lanes that lie side by side in memory (strides[0] = 8: the reduced axis is not the array's
// last, a lane is a column): a column walked alone touches a cache line per element and its
// sums are scalar. Here the elements j of L adjacent lanes are one contiguous run, updated
// together: chain j mod 8 of every lane, the lanes as the vector. The same operations on the
// same chains as lane by lane, so the same bits.
constexpr std::int64_t kSideLanes = 64;

// The sum of a lane of at most kSub elements: its one sub-block's tree, as join() of the one
// block's Neumaier sum would give it (v + 0.0: -0.0 comes out 0.0 there).
struct Once {
    double v = 0.0;
    void add(double x) { v = x + 0.0; }
};
inline double value_of(const Once& s) { return s.v; }

// ch: 8 chains x L lanes (chain k at k * L); at a sub-block's end their trees go into sums[l]
// (per lane) and the chains restart from 0.
template <class Acc>
inline void side_flush(double* ch, std::int64_t L, Acc* sums) {
    std::int64_t l = 0;
    alignas(16) double t[2];
    for (; l + 2 <= L; l += 2) {
        float64x2_t v[8];
        for (int k = 0; k < 8; ++k) v[k] = vld1q_f64(ch + k * L + l);
        const float64x2_t r = vaddq_f64(
            vaddq_f64(vaddq_f64(v[0], v[1]), vaddq_f64(v[2], v[3])),
            vaddq_f64(vaddq_f64(v[4], v[5]), vaddq_f64(v[6], v[7]))
        );
        vst1q_f64(t, r);
        sums[l].add(t[0]), sums[l + 1].add(t[1]);
    }
    for (; l < L; ++l) {
        double s[8];
        for (int k = 0; k < 8; ++k) s[k] = ch[k * L + l];
        sums[l].add(tree8(s));
    }
    std::fill(ch, ch + 8 * L, 0.0);
}

// Element j of the side-by-side lanes: row j at base + j * rs, the lanes adjacent doubles.
struct Side {
    const char* base;
    std::int64_t rs;
    const double* row(std::int64_t j) const { return reinterpret_cast<const double*>(base + j * rs); }
};
inline Side side_of(const NDArray& a, std::int64_t r0) {
    return {reinterpret_cast<const char*>(a.data) + r0 * 8, a.strides[1]};
}

// rows j0..j1 of lanes r0..r0+L: the sums of x (NaN as 0) into sums[l], the NaN counts into bad
template <bool M, class Acc>
void side_sum(const Side& a, std::int64_t L, std::int64_t j0, std::int64_t j1, Acc* sums, std::int64_t* bad) {
    std::vector<double> ch(static_cast<std::size_t>(8 * L), 0.0);
    std::vector<std::uint64_t> okc(static_cast<std::size_t>(L + 1), 0);
    for (std::int64_t j = j0; j < j1; ++j) {
        const double* row = a.row(j);
        double* cj = ch.data() + (j & 7) * L;
        std::int64_t l = 0;
        for (; l + 2 <= L; l += 2) {
            const float64x2_t x = vld1q_f64(row + l);
            if constexpr (M) vst1q_u64(okc.data() + l, vsubq_u64(vld1q_u64(okc.data() + l), vceqq_f64(x, x)));
            vst1q_f64(cj + l, vaddq_f64(vld1q_f64(cj + l), keep<M>(x, x)));
        }
        for (; l < L; ++l) {
            const bool nan = M && std::isnan(row[l]);
            okc[l] += !nan;
            cj[l] += nan ? 0.0 : row[l];
        }
        if ((j + 1) % kSub == 0 || j + 1 == j1) side_flush(ch.data(), L, sums);
    }
    for (std::int64_t l = 0; l < L; ++l) bad[l] = M ? (j1 - j0) - static_cast<std::int64_t>(okc[l]) : 0;
}

// d = x - h[lane] (0 for a NaN), d^k into sums[(k - 1) * L + l], k = 1..kmax
template <bool M, int COMP, int kmax, class Acc>
void side_moments(const Side& a, std::int64_t L, std::int64_t j0, std::int64_t j1, const double* h, Acc* sums) {
    constexpr bool cm[4] = {comp_of(0, COMP), comp_of(1, COMP), comp_of(2, COMP), comp_of(3, COMP)};
    std::vector<double> ch(static_cast<std::size_t>(8 * 8 * L), 0.0);
    double *C[4], *Er[4];  // Er: the TwoSum chains' corrections
    for (int m = 0; m < 4; ++m) C[m] = ch.data() + m * 8 * L, Er[m] = ch.data() + (4 + m) * 8 * L;
    for (std::int64_t j = j0; j < j1; ++j) {
        const double* row = a.row(j);
        const std::int64_t k = (j & 7) * L;
        std::int64_t l = 0;
        for (; l + 2 <= L; l += 2) {
            const float64x2_t x = vld1q_f64(row + l);
            const float64x2_t d = keep<M>(x, vsubq_f64(x, vld1q_f64(h + l)));
            const float64x2_t d2 = vmulq_f64(d, d);
            const float64x2_t p[4] = {d, d2, vmulq_f64(d2, d), vmulq_f64(d2, d2)};
            for (int m = 0; m < kmax; ++m) {
                float64x2_t sv = vld1q_f64(C[m] + k + l);
                if (cm[m]) {
                    float64x2_t cv = vld1q_f64(Er[m] + k + l);
                    two_add(sv, cv, p[m]);
                    vst1q_f64(Er[m] + k + l, cv);
                } else {
                    sv = vaddq_f64(sv, p[m]);
                }
                vst1q_f64(C[m] + k + l, sv);
            }
        }
        for (; l < L; ++l) {
            const double d = M && std::isnan(row[l]) ? 0.0 : row[l] - h[l], d2 = d * d;
            const double p[4] = {d, d2, d2 * d, d2 * d2};
            for (int m = 0; m < kmax; ++m)
                if (cm[m]) two_add(C[m][k + l], Er[m][k + l], p[m]);
                else C[m][k + l] += p[m];
        }
        if ((j + 1) % kSub == 0 || j + 1 == j1)
            for (int m = 0; m < kmax; ++m) {
                if (!cm[m]) {
                    side_flush(C[m], L, sums + m * L);
                    continue;
                }
                for (std::int64_t l2 = 0; l2 < L; ++l2) {
                    double sk[8], ck[8];
                    for (int q = 0; q < 8; ++q) sk[q] = C[m][q * L + l2], ck[q] = Er[m][q * L + l2];
                    add_comb(sums[m * L + l2], sk, ck);
                }
                std::fill(C[m], C[m] + 8 * L, 0.0), std::fill(Er[m], Er[m] + 8 * L, 0.0);
            }
    }
}
#endif

nb::tuple moments(nb::handle rows_obj, bool omit, int kmax, bool minmax, int comp) {
    const NDArray& a = lanes_arg(rows_obj);
    const std::int64_t lanes = a.shape[0], n = a.shape[1];
    NDArray cnt = vec(lanes, DType::Int64), mean = vec(lanes), m2 = vec(lanes), m3 = vec(lanes), m4 = vec(lanes),
            mn = vec(lanes), mx = vec(lanes), ml = vec(lanes);
    auto* pc = reinterpret_cast<std::int64_t*>(cnt.data);
    double *pm = reinterpret_cast<double*>(mean.data), *p2 = reinterpret_cast<double*>(m2.data),
           *p3 = reinterpret_cast<double*>(m3.data), *p4 = reinterpret_cast<double*>(m4.data),
           *pmn = reinterpret_cast<double*>(mn.data), *pmx = reinterpret_cast<double*>(mx.data),
           *pml = reinterpret_cast<double*>(ml.data);
    kmax = std::clamp(kmax, 2, 4);
    if (std::find(std::begin(kCompSets), std::end(kCompSets), comp) == std::end(kCompSets))
        throw nb::value_error("desc_moments: comp must be one of 0, 1, 5, 10, 15");
    const std::int64_t nb_ = std::max<std::int64_t>(1, (n + kBlock - 1) / kBlock);
    const std::int64_t s1 = a.strides[1];
    auto lane_of = [&](std::int64_t r) { return Lane{reinterpret_cast<const char*>(a.data) + r * a.strides[0], s1}; };
    // a block's pass 1 / pass 2 sums (contiguous lanes by NEON, the same bits)
    auto block1 = [&](std::int64_t r, std::int64_t j0, std::int64_t j1, Neumaier& sum, std::int64_t& bad, double& lo,
                      double& hi) {
        const Lane L = lane_of(r);
#if defined(__aarch64__)
        if (s1 == 8 && !minmax) {
            const double* xs = reinterpret_cast<const double*>(L.base);
            Neumaier t;
            pass1_contig<false>(xs, j0, j1, t, bad);
            if (std::isnan(t.s)) t = Neumaier{}, pass1_contig<true>(xs, j0, j1, t, bad);
            sum = t;
            return;
        }
#endif
        pass1_scalar(L, j0, j1, sum, bad, minmax, lo, hi);
    };
    auto block2 = [&](std::int64_t r, std::int64_t j0, std::int64_t j1, double h, Neumaier* sums, bool nan) {
        const Lane L = lane_of(r);
#if defined(__aarch64__)
        if (s1 == 8) {
            const double* xs = reinterpret_cast<const double*>(L.base);
            with_comp(comp, [&]<int C>() {
                with_k(kmax, [&]<int K>() {
                    if (nan) pass2_contig<true, C, K>(xs, j0, j1, h, sums);
                    else pass2_contig<false, C, K>(xs, j0, j1, h, sums);
                });
            });
            return;
        }
#endif
        pass2_scalar(L, j0, j1, h, kmax, comp, sums);
    };
    // the mean (as hi from pass 1) of lane r, from its blocks' sums, count and NaN flag
    auto finish1 = [&](std::int64_t r, double sum, std::int64_t c, bool nan, double lo, double hi) {
        pc[r] = omit ? c : n;
        const bool bad = c == 0 || (nan && !omit);
        pmn[r] = bad || !minmax ? kNaN : lo;
        pmx[r] = bad || !minmax ? kNaN : hi;
        return bad ? kNaN : sum / static_cast<double>(c);
    };
    // the results of lane r from hi and the deviations' sums
    auto finish2 = [&](std::int64_t r, double h, const double* S) {
        if (std::isnan(h)) {
            pm[r] = p2[r] = p3[r] = p4[r] = pml[r] = kNaN;
            return;
        }
        if (!std::isfinite(h)) {  // an infinite mean (inf in the data): no moments
            pm[r] = h, pml[r] = 0.0, p2[r] = p3[r] = p4[r] = kNaN;
            return;
        }
        const double c = static_cast<double>(pc[r]);
        const double S1 = S[0] / c, S2 = S[1] / c, S3 = kmax >= 3 ? S[2] / c : 0.0, S4 = kmax >= 4 ? S[3] / c : 0.0;
        const double lo = S1, l2 = lo * lo;
        pm[r] = h + lo;
        pml[r] = lo - (pm[r] - h);
        p2[r] = S2 - l2;
        p3[r] = kmax >= 3 ? S3 - 3.0 * lo * S2 + 2.0 * l2 * lo : kNaN;
        p4[r] = kmax >= 4 ? S4 - 4.0 * lo * S3 + 6.0 * l2 * S2 - 3.0 * l2 * l2 : kNaN;
    };
    bool side = false;
#if defined(__aarch64__)
    side = a.strides[0] == 8 && lanes >= 2 && !minmax && n > 0;
#endif
    {
        nb::gil_scoped_release release;
#if defined(__aarch64__)
        if (!side && !minmax && s1 == 8 && n <= kSub && lanes >= 2) {
            // pair_pass2 for (kmax, mask, comp) at run time
            auto pair2 = [&](bool msk, const double* p0, const double* p1, std::int64_t nn, const double* h,
                             double* S) {
                auto by_k = [&]<bool Mk, int C>() {
                    if (kmax == 2) pair_pass2<2, Mk, C>(p0, p1, nn, h, S);
                    else if (kmax == 3) pair_pass2<3, Mk, C>(p0, p1, nn, h, S);
                    else pair_pass2<4, Mk, C>(p0, p1, nn, h, S);
                };
                auto by_c = [&]<bool Mk>() { with_comp(comp, [&]<int C>() { by_k.template operator()<Mk, C>(); }); };
                if (msk) by_c.template operator()<true>();
                else by_c.template operator()<false>();
            };
            // short contiguous lanes: two at a time as the vectors' lanes (pair_*)
            const std::int64_t pairs = lanes / 2;
            for_pieces(
                pairs, std::max<std::int64_t>(1, (1 << 13) / std::max<std::int64_t>(n, 1)), lanes * n * kmax,
                [&](std::int64_t, std::int64_t q0, std::int64_t q1) {
                    for (std::int64_t q = q0; q < q1; ++q) {
                        const std::int64_t r = 2 * q;
                        const double* p0 =
                            reinterpret_cast<const double*>(reinterpret_cast<const char*>(a.data) + r * a.strides[0]);
                        const double* p1 =
                            reinterpret_cast<const double*>(reinterpret_cast<const char*>(p0) + a.strides[0]);
                        alignas(16) double sum[2], h[2], S[8] = {};
                        std::int64_t bad[2];
                        pair_pass1<false>(p0, p1, n, sum, bad);
                        if (std::isnan(sum[0]) || std::isnan(sum[1])) pair_pass1<true>(p0, p1, n, sum, bad);
                        for (int l = 0; l < 2; ++l) h[l] = finish1(r + l, sum[l], n - bad[l], bad[l] != 0, kInf, -kInf);
                        if (std::isfinite(h[0]) || std::isfinite(h[1])) { pair2(bad[0] || bad[1], p0, p1, n, h, S); }
                        for (int l = 0; l < 2; ++l) {
                            const double Sl[4] = {S[l], S[2 + l], S[4 + l], S[6 + l]};
                            finish2(r + l, h[l], Sl);
                        }
                    }
                }
            );
            if (lanes % 2) {  // the last lane alone (the same bits)
                const std::int64_t r = lanes - 1;
                Neumaier s1v;
                std::int64_t bad = 0;
                double lo = kInf, hi = -kInf;
                block1(r, 0, n, s1v, bad, lo, hi);
                const double h = finish1(r, join(&s1v, 1, 1), n - bad, bad != 0, lo, hi);
                double S[4] = {};
                if (std::isfinite(h)) {
                    Neumaier sums[4];
                    block2(r, 0, n, h, sums, bad != 0);
                    for (int m = 0; m < kmax; ++m) S[m] = join(sums + m, 1, 1);
                }
                finish2(r, h, S);
            }
        } else
#endif
            if (nb_ == 1 && !side) {
            // short lanes: each one whole in its task
            for_pieces(
                lanes, std::max<std::int64_t>(1, (1 << 14) / std::max<std::int64_t>(n, 1)), lanes * n * kmax,
                [&](std::int64_t, std::int64_t r0, std::int64_t r1) {
                    for (std::int64_t r = r0; r < r1; ++r) {
                        Neumaier s1v;
                        std::int64_t bad = 0;
                        double lo = kInf, hi = -kInf;
                        block1(r, 0, n, s1v, bad, lo, hi);
                        const double h = finish1(r, join(&s1v, 1, 1), n - bad, bad != 0, lo, hi);
                        double S[4] = {};
                        if (std::isfinite(h)) {
                            Neumaier sums[4];
                            block2(r, 0, n, h, sums, bad != 0);
                            for (int m = 0; m < kmax; ++m) S[m] = join(sums + m, 1, 1);
                        }
                        finish2(r, h, S);
                    }
                }
            );
        } else {
            // (lane, block) items, or (group of kSideLanes lanes, block) for side-by-side lanes
            const std::int64_t items = lanes * nb_;
            std::vector<Neumaier> q1(static_cast<std::size_t>(items));
            std::vector<std::int64_t> qbad(static_cast<std::size_t>(items), 0);
            std::vector<double> qlo(minmax ? items : 0, kInf), qhi(minmax ? items : 0, -kInf);
            std::vector<Neumaier> q2(static_cast<std::size_t>(items) * 4);  // item * 4 + (k - 1)
            std::vector<double> his(static_cast<std::size_t>(lanes));
            const std::int64_t groups = (lanes + kSideLanes - 1) / kSideLanes;
            auto run = [&](auto&& body) {
                for_pieces(
                    items, std::max<std::int64_t>(1, (1 << 16) / kBlock), lanes * n,
                    [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
                        for (std::int64_t it = b0; it < b1; ++it) {
                            const std::int64_t b = it % nb_;
                            body(it, it / nb_, b * kBlock, std::min(n, (b + 1) * kBlock));
                        }
                    }
                );
            };
            auto run_side = [&](auto&& body) {
                for_pieces(groups * nb_, 1, lanes * n, [&](std::int64_t, std::int64_t t0, std::int64_t t1) {
                    for (std::int64_t t = t0; t < t1; ++t) {
                        const std::int64_t g = t / nb_, b = t % nb_, r0 = g * kSideLanes;
                        body(r0, std::min(kSideLanes, lanes - r0), b, b * kBlock, std::min(n, (b + 1) * kBlock));
                    }
                });
            };
#if defined(__aarch64__)
            if (side)
                run_side([&](std::int64_t r0, std::int64_t L, std::int64_t b, std::int64_t j0, std::int64_t j1) {
                    std::vector<Neumaier> sums(static_cast<std::size_t>(L));
                    std::vector<std::int64_t> bad(static_cast<std::size_t>(L));
                    side_sum<false>(side_of(a, r0), L, j0, j1, sums.data(), bad.data());
                    if (std::any_of(sums.begin(), sums.end(), [](const Neumaier& t) { return std::isnan(t.s); })) {
                        std::fill(sums.begin(), sums.end(), Neumaier{});
                        side_sum<true>(side_of(a, r0), L, j0, j1, sums.data(), bad.data());
                    }
                    for (std::int64_t l = 0; l < L; ++l) {
                        const std::size_t it = static_cast<std::size_t>((r0 + l) * nb_ + b);
                        q1[it] = sums[l], qbad[it] = bad[l];
                    }
                });
            else
#endif
                run([&](std::int64_t it, std::int64_t r, std::int64_t j0, std::int64_t j1) {
                    double lo = kInf, hi = -kInf;
                    block1(r, j0, j1, q1[it], qbad[it], lo, hi);
                    if (minmax) qlo[it] = lo, qhi[it] = hi;
                });
            for (std::int64_t r = 0; r < lanes; ++r) {
                std::int64_t bad = 0;
                double lo = kInf, hi = -kInf;
                for (std::int64_t b = 0; b < nb_; ++b) {
                    const std::size_t it = static_cast<std::size_t>(r * nb_ + b);
                    bad += qbad[it];
                    if (minmax) lo = std::min(lo, qlo[it]), hi = std::max(hi, qhi[it]);
                }
                his[r] = finish1(r, join(q1.data() + r * nb_, nb_, 1), n - bad, bad != 0, lo, hi);
            }
#if defined(__aarch64__)
            if (side)  // a lane without a finite mean gets sums nobody reads
                run_side([&](std::int64_t r0, std::int64_t L, std::int64_t b, std::int64_t j0, std::int64_t j1) {
                    std::vector<Neumaier> sums(static_cast<std::size_t>(4 * L));
                    bool nan = false;
                    for (std::int64_t l = 0; l < L; ++l)
                        nan = nan || qbad[static_cast<std::size_t>((r0 + l) * nb_ + b)] != 0;
                    with_comp(comp, [&]<int C>() {
                        with_k(kmax, [&]<int K>() {
                            if (nan) side_moments<true, C, K>(side_of(a, r0), L, j0, j1, his.data() + r0, sums.data());
                            else side_moments<false, C, K>(side_of(a, r0), L, j0, j1, his.data() + r0, sums.data());
                        });
                    });
                    for (std::int64_t l = 0; l < L; ++l)
                        for (int m = 0; m < kmax; ++m)
                            q2[static_cast<std::size_t>(((r0 + l) * nb_ + b) * 4 + m)] = sums[m * L + l];
                });
            else
#endif
                run([&](std::int64_t it, std::int64_t r, std::int64_t j0, std::int64_t j1) {
                    const double h = his[static_cast<std::size_t>(r)];
                    if (std::isfinite(h))
                        block2(r, j0, j1, h, q2.data() + it * 4, qbad[static_cast<std::size_t>(it)] != 0);
                });
            for (std::int64_t r = 0; r < lanes; ++r) {
                double S[4] = {};
                for (int m = 0; m < kmax; ++m) S[m] = join(q2.data() + r * nb_ * 4 + m, nb_, 4);
                finish2(r, his[static_cast<std::size_t>(r)], S);
            }
        }
    }
    return nb::make_tuple(
        wrap(std::move(cnt)), wrap(std::move(mean)), wrap(std::move(m2)), wrap(std::move(m3)), wrap(std::move(m4)),
        wrap(std::move(mn)), wrap(std::move(mx)), wrap(std::move(ml))
    );
}

// d^k for an integer k (repeated squaring: exact as the products go), else pow
double ipow(double d, double k) {
    if (k == std::floor(k) && std::fabs(k) <= 64.0) {
        auto e = static_cast<long long>(std::fabs(k));
        double r = 1.0, b = d;
        while (e) {
            if (e & 1) r *= b;
            b *= b;
            e >>= 1;
        }
        return k < 0 ? 1.0 / r : r;
    }
    return std::pow(d, k);
}

// mean((x - c)^order) per lane; c the lane's mean where `centers` is None (then order 1
// gives 0 and order 0 gives 1, as scipy), else centers[r].
nb::object moment(nb::handle rows_obj, double order, nb::handle centers_obj, bool omit) {
    const NDArray& a = lanes_arg(rows_obj);
    const std::int64_t lanes = a.shape[0], n = a.shape[1];
    const NDArray* centers = centers_obj.is_none() ? nullptr : &require_ndarray(centers_obj);
    if (centers && (centers->dtype != DType::Float64 || centers->ndim != 1 || centers->shape[0] != lanes))
        throw nb::value_error("desc: centers must be a float64 vector, one per lane");
    NDArray out = vec(lanes);
    double* po = reinterpret_cast<double*>(out.data);
    {
        nb::gil_scoped_release release;
        each_lane(lanes, n, [&](std::int64_t r, std::vector<double>& v) {
            bool has_nan;
            lane(a, r, omit, v, has_nan);
            if (v.empty() || (has_nan && !omit)) {
                po[r] = kNaN;
                return;
            }
            if (order == 0.0) {
                po[r] = 1.0;
                return;
            }
            Mean c{0.0, 0.0};
            if (centers) {
                c.hi = *reinterpret_cast<const double*>(centers->data + r * centers->strides[0]);
            } else {
                if (order == 1.0) {
                    po[r] = 0.0;
                    return;
                }
                c = mean_of(v);
            }
            Neumaier s;
            for (double x : v) s.add(ipow(c.dev(x), order));
            po[r] = s.value() / static_cast<double>(v.size());
        });
    }
    return wrap(std::move(out));
}

// Ranks along each lane (method: 0 average, 1 min, 2 max, 3 dense, 4 ordinal), ties by
// equal values after sorting (value, position) pairs (a stable sort's order). A lane with a NaN is all NaN, or with omit its NaNs are
// NaN and the rest ranked among themselves (scipy).
NDArray rank_rows(const NDArray& a, int method, bool omit) {
    const std::int64_t lanes = a.shape[0], n = a.shape[1];
    const std::int64_t shape[] = {lanes, n};
    NDArray out = new_array(DType::Float64, shape);
    double* po = reinterpret_cast<double*>(out.data);
    {
        nb::gil_scoped_release release;
        each_lane(lanes, n, [&](std::int64_t r, std::vector<double>&) {
            double* o = po + r * n;
            // (value, position) pairs sorted: the same order as a stable sort by value
            std::vector<std::pair<double, std::int64_t>> vs;
            vs.reserve(static_cast<std::size_t>(n));
            bool has_nan = false;
            for (std::int64_t j = 0; j < n; ++j) {
                const double x = at(a, r, j);
                if (std::isnan(x)) {
                    has_nan = true;
                    o[j] = kNaN;
                } else {
                    vs.emplace_back(x, j);
                }
            }
            if (has_nan && !omit) {
                for (std::int64_t j = 0; j < n; ++j) o[j] = kNaN;
                return;
            }
            sort_long(vs);
            const std::int64_t m = static_cast<std::int64_t>(vs.size());
            double dense = 0.0;
            for (std::int64_t i = 0; i < m;) {
                std::int64_t e = i + 1;
                const double v = vs[static_cast<std::size_t>(i)].first;
                while (e < m && vs[static_cast<std::size_t>(e)].first == v) ++e;
                dense += 1.0;
                for (std::int64_t t = i; t < e; ++t) {
                    double rk;
                    switch (method) {
                        case 0: rk = 0.5 * static_cast<double>(i + 1 + e); break;  // (i+1 + e) / 2
                        case 1: rk = static_cast<double>(i + 1); break;
                        case 2: rk = static_cast<double>(e); break;
                        case 3: rk = dense; break;
                        default: rk = static_cast<double>(t + 1); break;
                    }
                    o[vs[static_cast<std::size_t>(t)].second] = rk;
                }
                i = e;
            }
        });
    }
    return out;
}
nb::object rank(nb::handle rows_obj, int method, bool omit) {
    return wrap(rank_rows(lanes_arg(rows_obj), method, omit));
}

// (mode, count) per lane: the most frequent value, the smallest among ties. NaN counts as a
// value (all NaNs equal) unless omit (scipy); an empty lane is (NaN, 0).
nb::tuple mode(nb::handle rows_obj, bool omit) {
    const NDArray& a = lanes_arg(rows_obj);
    const std::int64_t lanes = a.shape[0], n = a.shape[1];
    NDArray md = vec(lanes), ct = vec(lanes, DType::Int64);
    double* pm = reinterpret_cast<double*>(md.data);
    auto* pc = reinterpret_cast<std::int64_t*>(ct.data);
    {
        nb::gil_scoped_release release;
        each_lane(lanes, n, [&](std::int64_t r, std::vector<double>& v) {
            bool has_nan;
            lane(a, r, true, v, has_nan);  // the NaNs apart: counted as one value below
            const std::int64_t nans = has_nan && !omit ? n - static_cast<std::int64_t>(v.size()) : 0;
            sort_long(v);
            double best = kNaN;
            std::int64_t bc = 0;
            for (std::size_t i = 0; i < v.size();) {
                std::size_t e = i + 1;
                while (e < v.size() && v[e] == v[i]) ++e;
                if (static_cast<std::int64_t>(e - i) > bc) bc = static_cast<std::int64_t>(e - i), best = v[i];
                i = e;
            }
            if (nans > bc) best = kNaN, bc = nans;  // NaN sorts last: it wins only outright
            pm[r] = best, pc[r] = bc;
        });
    }
    return nb::make_tuple(wrap(std::move(md)), wrap(std::move(ct)));
}

// The differential entropy estimates of scipy (method: 0 vasicek, 1 van es, 2 ebrahimi,
// 3 correa; -1 auto by the lane's size), window m (0: floor(sqrt(n) + 1/2)); NaN where the
// lane is too small (2 <= 2m < n fails) or had a NaN without omit.
double diffent_lane(std::vector<double>& v, std::int64_t mw, int method) {
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    const std::int64_t m = mw > 0 ? mw : static_cast<std::int64_t>(std::floor(std::sqrt(static_cast<double>(n)) + 0.5));
    if (!(2 <= 2 * m && 2 * m < n)) return kNaN;
    sort_long(v);
    if (method < 0) method = n <= 10 ? 1 : (n <= 1000 ? 2 : 0);
    auto X = [&](std::int64_t i) {  // padded: X[i] for i in [-m, n + m), the ends repeated
        return v[static_cast<std::size_t>(std::clamp<std::int64_t>(i, 0, n - 1))];
    };
    const double nd = static_cast<double>(n), md = static_cast<double>(m);
    Neumaier s;
    // A zero spacing (tied values, a constant lane) is a term log 0 = -inf, and so is the
    // estimate, as in scipy; an infinite spacing (an infinite value) is a term +inf, and the
    // estimate inf, again as in scipy (both: NaN). The compensated sum would turn an infinite
    // term into NaN (inf - inf in its correction), so such terms are counted aside.
    bool zero = false, infinite = false;
    auto add_log = [&](double t) {
        if (t == 0.0) zero = true;
        else if (t == kInf) infinite = true;
        else s.add(std::log(t));
    };
    auto result = [&](double value) {
        if (std::isnan(value)) return kNaN;
        if (zero && infinite) return kNaN;
        if (zero) return value == kInf ? kNaN : -kInf;
        if (infinite) return value == -kInf ? kNaN : kInf;
        return value;
    };
    switch (method) {
        case 0:  // Vasicek: mean log(n / (2m) (X[i+m] - X[i-m]))
            for (std::int64_t i = 0; i < n; ++i) add_log(nd / (2.0 * md) * (X(i + m) - X(i - m)));
            return result(s.value() / nd);
        case 1: {  // van Es
            for (std::int64_t i = 0; i + m < n; ++i) add_log((nd + 1.0) / md * (v[i + m] - v[i]));
            Neumaier h;
            for (std::int64_t k = m; k <= n; ++k) h.add(1.0 / static_cast<double>(k));
            return result(s.value() / (nd - md) + h.value() + std::log(md) - std::log(nd + 1.0));
        }
        case 2:  // Ebrahimi
            for (std::int64_t i = 1; i <= n; ++i) {
                double ci = 2.0;
                if (i <= m) ci = 1.0 + static_cast<double>(i - 1) / md;
                if (i >= n - m + 1) ci = 1.0 + static_cast<double>(n - i) / md;
                add_log(nd * (X(i - 1 + m) - X(i - 1 - m)) / (ci * md));
            }
            return result(s.value() / nd);
        default: {  // Correa (a constant window is 0 / 0: NaN, as in scipy)
            for (std::int64_t i = 1; i <= n; ++i) {
                Neumaier mb;
                for (std::int64_t d = -m; d <= m; ++d) mb.add(X(i + d - 1));
                const double xbar = mb.value() / (2.0 * md + 1.0);
                Neumaier num, den;
                for (std::int64_t d = -m; d <= m; ++d) {
                    const double diff = X(i + d - 1) - xbar;
                    num.add(diff * static_cast<double>(d));
                    den.add(diff * diff);
                }
                s.add(std::log(num.value() / (nd * den.value())));
            }
            return -s.value() / nd;
        }
    }
}
nb::object diffent(nb::handle rows_obj, std::int64_t window, int method, bool omit) {
    const NDArray& a = lanes_arg(rows_obj);
    const std::int64_t lanes = a.shape[0], n = a.shape[1];
    NDArray out = vec(lanes);
    double* po = reinterpret_cast<double*>(out.data);
    {
        nb::gil_scoped_release release;
        each_lane(lanes, n * 4, [&](std::int64_t r, std::vector<double>& v) {
            bool has_nan;
            lane(a, r, omit, v, has_nan);
            po[r] = has_nan && !omit ? kNaN : diffent_lane(v, window, method);
        });
    }
    return wrap(std::move(out));
}

}  // namespace

// zscore's last step, ((x - mean) - mean_lo) / std per lane, in one pass: rows (lanes, n) any
// strides, the three per-lane vectors contiguous float64; a new C-order (lanes, n) array. The
// operations of the array expression it replaces, so the same bits.
nb::object zscore_rows(nb::handle rows_obj, nb::handle mean_obj, nb::handle lo_obj, nb::handle std_obj) {
    const NDArray& a = lanes_arg(rows_obj);
    const std::int64_t lanes = a.shape[0], n = a.shape[1];
    const double* per[3];
    nb::handle hs[3] = {mean_obj, lo_obj, std_obj};
    for (int k = 0; k < 3; ++k) {
        const NDArray& v = require_ndarray(hs[k]);
        if (v.dtype != DType::Float64 || v.ndim != 1 || v.shape[0] != lanes || (lanes > 1 && v.strides[0] != 8))
            throw nb::value_error("desc_zscore: expects contiguous float64 vectors, one value per lane");
        per[k] = reinterpret_cast<const double*>(v.data);
    }
    const std::int64_t shape[] = {lanes, n};
    NDArray out = new_array(DType::Float64, shape);
    double* po = reinterpret_cast<double*>(out.data);
    {
        nb::gil_scoped_release release;
        for_pieces(
            lanes, std::max<std::int64_t>(1, (1 << 14) / std::max<std::int64_t>(n, 1)), lanes * n,
            [&](std::int64_t, std::int64_t r0, std::int64_t r1) {
                for (std::int64_t r = r0; r < r1; ++r) {
                    const double m = per[0][r], lo = per[1][r], sdv = per[2][r];
                    const char* base = reinterpret_cast<const char*>(a.data) + r * a.strides[0];
                    double* o = po + r * n;
                    const std::int64_t s1 = a.strides[1];
                    if (s1 == 8) {
                        const double* xs = reinterpret_cast<const double*>(base);
                        for (std::int64_t j = 0; j < n; ++j) o[j] = ((xs[j] - m) - lo) / sdv;
                    } else {
                        for (std::int64_t j = 0; j < n; ++j)
                            o[j] = ((*reinterpret_cast<const double*>(base + j * s1) - m) - lo) / sdv;
                    }
                }
            }
        );
    }
    return wrap(std::move(out));
}

void bind_stats_desc(nb::module_& m) {
    m.def("desc_moments", &moments);
    m.def("desc_moment", &moment, nb::arg("rows"), nb::arg("order"), nb::arg("centers").none(), nb::arg("omit"));
    m.def("desc_rank", &rank);
    m.def("desc_mode", &mode);
    m.def("desc_diffent", &diffent);
    m.def("desc_zscore", &zscore_rows);
}

}  // namespace sd
