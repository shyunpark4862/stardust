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

struct Neumaier {
    double s = 0.0, c = 0.0;
    void add(double x) {
        const double t = s + x;
        c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    double value() const { return s + c; }
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
    for_pieces(lanes, std::max<std::int64_t>(1, kParallelTask / std::max<std::int64_t>(n, 1)), lanes * n,
               [&](std::int64_t, std::int64_t b, std::int64_t e) {
                   std::vector<double> v;
                   for (std::int64_t r = b; r < e; ++r) f(r, v);
               });
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
    for (std::int64_t i = 0; i < kBuckets * kOver; ++i) sample[i] = v[static_cast<std::size_t>(i * (n / (kBuckets * kOver)))];
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
// Two passes over fixed blocks of kBlock elements (lane, block) on the pool, combined in
// order: the blocks do not depend on the thread count, so neither do the bits, and a single
// long lane (axis=None) runs in parallel too. Pass 1: count, the compensated sum, min, max,
// NaN; hi = sum / n. Pass 2: S_k = mean(d^k) of d = x - hi, k = 1..kmax, each by four
// interleaved compensated sums (one chain is latency-bound). The mean's low part is lo = S_1,
// and the central moments of x - (hi + lo) follow without a third pass:
//     m2 = S2 - lo^2,  m3 = S3 - 3 lo S2 + 2 lo^3,  m4 = S4 - 4 lo S3 + 6 lo^2 S2 - 3 lo^4.
constexpr std::int64_t kBlock = 8192;

// Eight compensated sums side by side, Knuth's branch-free TwoSum (the error of each
// addition exact, as Neumaier's): independent chains the compiler can vectorize.
struct Neu4 {
    double s[8] = {}, c[8] = {};
    void add(std::int64_t j, double x) {
        const int k = static_cast<int>(j & 7);
        const double t = s[k] + x, bb = t - s[k];
        c[k] += (s[k] - (t - bb)) + (x - bb);
        s[k] = t;
    }
    void into(Neumaier& tot) const {
        for (int k = 0; k < 8; ++k) tot.add(s[k]), tot.add(c[k]);
    }
#if defined(__aarch64__)
    // chains 2 q and 2 q + 1 take the two lanes of x: the bits of add(2 q, x[0]), add(2 q + 1, x[1])
    // (the scalar loop is not vectorized by the compiler: 7 flops an element, 0.45 ns)
    void add2(int q, float64x2_t x) {
        const float64x2_t sv = vld1q_f64(s + 2 * q), cv = vld1q_f64(c + 2 * q);
        const float64x2_t t = vaddq_f64(sv, x), bb = vsubq_f64(t, sv);
        vst1q_f64(c + 2 * q, vaddq_f64(cv, vaddq_f64(vsubq_f64(sv, vsubq_f64(t, bb)), vsubq_f64(x, bb))));
        vst1q_f64(s + 2 * q, t);
    }
#endif
};

#if defined(__aarch64__)
// The contiguous blocks' loops, eight elements at a time from j a multiple of eight (the
// kernels' blocks start at multiples of kBlock): the same operations on the same chains as
// the scalar loops, so the same bits.
// xs[j0..j1) into q with the NaNs as 0; their number.
std::int64_t sum_block(Neu4& q, const double* xs, std::int64_t j0, std::int64_t j1) {
    std::int64_t j = j0, bad = 0;
    uint64x2_t okc = vdupq_n_u64(0);
    for (; j + 8 <= j1; j += 8)
        for (int v = 0; v < 4; ++v) {
            const float64x2_t x = vld1q_f64(xs + j + 2 * v);
            const uint64x2_t ok = vceqq_f64(x, x);  // all ones where not NaN
            okc = vsubq_u64(okc, ok);
            q.add2(v, vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(x), ok)));
        }
    bad = (j - j0) - static_cast<std::int64_t>(vgetq_lane_u64(okc, 0) + vgetq_lane_u64(okc, 1));
    for (; j < j1; ++j) {
        const bool nan = std::isnan(xs[j]);
        bad += nan;
        q.add(j, nan ? 0.0 : xs[j]);
    }
    return bad;
}
// d = xs[j] - h (0 for a NaN), d^k into q[k - 1], k = 1..kmax
void moment_block(Neu4* q, const double* xs, std::int64_t j0, std::int64_t j1, double h, int kmax) {
    std::int64_t j = j0;
    const float64x2_t hv = vdupq_n_f64(h);
    for (; j + 8 <= j1; j += 8)
        for (int v = 0; v < 4; ++v) {
            const float64x2_t x = vld1q_f64(xs + j + 2 * v);
            const uint64x2_t ok = vceqq_f64(x, x);
            const float64x2_t d = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vsubq_f64(x, hv)), ok));
            const float64x2_t d2 = vmulq_f64(d, d);
            q[0].add2(v, d), q[1].add2(v, d2);
            if (kmax >= 3) q[2].add2(v, vmulq_f64(d2, d));
            if (kmax >= 4) q[3].add2(v, vmulq_f64(d2, d2));
        }
    for (; j < j1; ++j) {
        const bool nan = std::isnan(xs[j]);
        const double d = nan ? 0.0 : xs[j] - h, d2 = d * d;
        q[0].add(j, d), q[1].add(j, d2);
        if (kmax >= 3) q[2].add(j, d2 * d);
        if (kmax >= 4) q[3].add(j, d2 * d2);
    }
}

// Lanes that lie side by side in memory (strides[0] = 8: the reduced axis is not the array's
// last, a lane is a column): a column walked alone touches a cache line per element and its
// sums are scalar. Here the elements j of L adjacent lanes are one contiguous run, updated
// together: chain j mod 8 of every lane, the lanes as the vector. The same operations on the
// same chains as lane by lane, so the same bits.
constexpr std::int64_t kSideLanes = 64;
// S, C: 8 chains x L lanes (chain k at k * L), zero on entry. x (L doubles) into chain k.
inline void side_add(double* S, double* C, std::int64_t L, int k, const float64x2_t* x, std::int64_t l) {
    double *sp = S + k * L + l, *cp = C + k * L + l;
    const float64x2_t sv = vld1q_f64(sp), cv = vld1q_f64(cp);
    const float64x2_t t = vaddq_f64(sv, *x), bb = vsubq_f64(t, sv);
    vst1q_f64(cp, vaddq_f64(cv, vaddq_f64(vsubq_f64(sv, vsubq_f64(t, bb)), vsubq_f64(*x, bb))));
    vst1q_f64(sp, t);
}
inline void side_add1(double* S, double* C, std::int64_t L, int k, double x, std::int64_t l) {
    double &sv = S[k * L + l], &cv = C[k * L + l];
    const double t = sv + x, bb = t - sv;
    cv += (sv - (t - bb)) + (x - bb);
    sv = t;
}
// rows j0..j1 of lanes r0..r0+L: the sums of x (NaN as 0) into S, C and the NaN counts into bad
void side_sum(const NDArray& a, std::int64_t r0, std::int64_t L, std::int64_t j0, std::int64_t j1, double* S, double* C,
              std::int64_t* bad) {
    std::vector<std::uint64_t> okc(static_cast<std::size_t>(L + 1), 0);
    for (std::int64_t j = j0; j < j1; ++j) {
        const double* row = reinterpret_cast<const double*>(reinterpret_cast<const char*>(a.data) + r0 * 8 + j * a.strides[1]);
        const int k = static_cast<int>(j & 7);
        std::int64_t l = 0;
        for (; l + 2 <= L; l += 2) {
            const float64x2_t x = vld1q_f64(row + l);
            const uint64x2_t ok = vceqq_f64(x, x);
            vst1q_u64(okc.data() + l, vsubq_u64(vld1q_u64(okc.data() + l), ok));
            const float64x2_t v = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(x), ok));
            side_add(S, C, L, k, &v, l);
        }
        for (; l < L; ++l) {
            const bool nan = std::isnan(row[l]);
            okc[l] += !nan;
            side_add1(S, C, L, k, nan ? 0.0 : row[l], l);
        }
    }
    for (std::int64_t l = 0; l < L; ++l) bad[l] = (j1 - j0) - static_cast<std::int64_t>(okc[l]);
}
// d = x - h[lane] (0 for a NaN), d^k into S[k - 1], C[k - 1] (each 8 x L), k = 1..kmax
void side_moments(const NDArray& a, std::int64_t r0, std::int64_t L, std::int64_t j0, std::int64_t j1, const double* h,
                  int kmax, double* const* S, double* const* C) {
    for (std::int64_t j = j0; j < j1; ++j) {
        const double* row = reinterpret_cast<const double*>(reinterpret_cast<const char*>(a.data) + r0 * 8 + j * a.strides[1]);
        const int k = static_cast<int>(j & 7);
        std::int64_t l = 0;
        for (; l + 2 <= L; l += 2) {
            const float64x2_t x = vld1q_f64(row + l);
            const uint64x2_t ok = vceqq_f64(x, x);
            const float64x2_t d = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vsubq_f64(x, vld1q_f64(h + l))), ok));
            const float64x2_t d2 = vmulq_f64(d, d);
            side_add(S[0], C[0], L, k, &d, l), side_add(S[1], C[1], L, k, &d2, l);
            if (kmax >= 3) {
                const float64x2_t d3 = vmulq_f64(d2, d);
                side_add(S[2], C[2], L, k, &d3, l);
            }
            if (kmax >= 4) {
                const float64x2_t d4 = vmulq_f64(d2, d2);
                side_add(S[3], C[3], L, k, &d4, l);
            }
        }
        for (; l < L; ++l) {
            const bool nan = std::isnan(row[l]);
            const double d = nan ? 0.0 : row[l] - h[l], d2 = d * d;
            side_add1(S[0], C[0], L, k, d, l), side_add1(S[1], C[1], L, k, d2, l);
            if (kmax >= 3) side_add1(S[2], C[2], L, k, d2 * d, l);
            if (kmax >= 4) side_add1(S[3], C[3], L, k, d2 * d2, l);
        }
    }
}
#endif

nb::tuple moments(nb::handle rows_obj, bool omit, int kmax, bool minmax) {
    const NDArray& a = lanes_arg(rows_obj);
    const std::int64_t lanes = a.shape[0], n = a.shape[1];
    NDArray cnt = vec(lanes, DType::Int64), mean = vec(lanes), m2 = vec(lanes), m3 = vec(lanes), m4 = vec(lanes),
            mn = vec(lanes), mx = vec(lanes), ml = vec(lanes);
    auto* pc = reinterpret_cast<std::int64_t*>(cnt.data);
    double *pm = reinterpret_cast<double*>(mean.data), *p2 = reinterpret_cast<double*>(m2.data),
           *p3 = reinterpret_cast<double*>(m3.data), *p4 = reinterpret_cast<double*>(m4.data),
           *pmn = reinterpret_cast<double*>(mn.data), *pmx = reinterpret_cast<double*>(mx.data),
           *pml = reinterpret_cast<double*>(ml.data);
    const std::int64_t nb_ = std::max<std::int64_t>(1, (n + kBlock - 1) / kBlock);
    const std::int64_t items = lanes * nb_;
    struct P1 {
        Neu4 s;
        std::int64_t cnt = 0;
        double lo = kInf, hi = -kInf;
        bool nan = false;
    };
    struct P2 {
        Neu4 s[4];
    };
    std::vector<P1> p1(static_cast<std::size_t>(items));
    std::vector<P2> p2v(static_cast<std::size_t>(items));
    std::vector<double> his(static_cast<std::size_t>(lanes));
    const std::int64_t s1 = a.strides[1];
    auto run = [&](auto&& body) {
        for_pieces(items, std::max<std::int64_t>(1, (1 << 16) / kBlock), lanes * n,
                   [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
                       for (std::int64_t it = b0; it < b1; ++it) {
                           const std::int64_t r = it / nb_, b = it % nb_;
                           const char* base = reinterpret_cast<const char*>(a.data) + r * a.strides[0];
                           body(it, r, base, b * kBlock, std::min(n, (b + 1) * kBlock));
                       }
                   });
    };
    // lanes side by side (see side_sum): groups of kSideLanes lanes x blocks on the pool
    bool side = false;
#if defined(__aarch64__)
    side = a.strides[0] == 8 && lanes >= 2 && !minmax && n > 0;
#endif
    const std::int64_t groups = (lanes + kSideLanes - 1) / kSideLanes;
    auto run_side = [&](auto&& body) {
        for_pieces(groups * nb_, 1, lanes * n, [&](std::int64_t, std::int64_t t0, std::int64_t t1) {
            for (std::int64_t t = t0; t < t1; ++t) {
                const std::int64_t g = t / nb_, b = t % nb_, r0 = g * kSideLanes;
                body(r0, std::min(kSideLanes, lanes - r0), b, b * kBlock, std::min(n, (b + 1) * kBlock));
            }
        });
    };
    {
        nb::gil_scoped_release release;
#if defined(__aarch64__)
        if (side)
            run_side([&](std::int64_t r0, std::int64_t L, std::int64_t b, std::int64_t j0, std::int64_t j1) {
                std::vector<double> S(static_cast<std::size_t>(8 * L), 0.0), C(static_cast<std::size_t>(8 * L), 0.0);
                std::vector<std::int64_t> bad(static_cast<std::size_t>(L));
                side_sum(a, r0, L, j0, j1, S.data(), C.data(), bad.data());
                for (std::int64_t l = 0; l < L; ++l) {
                    P1& q = p1[static_cast<std::size_t>((r0 + l) * nb_ + b)];
                    for (int k = 0; k < 8; ++k) q.s.s[k] = S[k * L + l], q.s.c[k] = C[k * L + l];
                    q.nan = bad[l] != 0;
                    q.cnt = (j1 - j0) - bad[l];
                }
            });
        else
#endif
        run([&](std::int64_t it, std::int64_t, const char* base, std::int64_t j0, std::int64_t j1) {
            P1& q = p1[static_cast<std::size_t>(it)];
            if (s1 == 8 && !minmax) {  // contiguous: NaNs counted, summed as 0 (selects, no branch)
                const double* xs = reinterpret_cast<const double*>(base);
                std::int64_t bad = 0;
#if defined(__aarch64__)
                bad = sum_block(q.s, xs, j0, j1);
#else
                for (std::int64_t j = j0; j < j1; ++j) {
                    const bool nan = std::isnan(xs[j]);
                    bad += nan;
                    q.s.add(j, nan ? 0.0 : xs[j]);
                }
#endif
                q.nan = bad != 0;
                q.cnt = (j1 - j0) - bad;
                return;
            }
            for (std::int64_t j = j0; j < j1; ++j) {
                const double x = *reinterpret_cast<const double*>(base + j * s1);
                if (std::isnan(x)) {
                    q.nan = true;
                    continue;
                }
                q.s.add(j, x);
                ++q.cnt;
                if (minmax) q.lo = std::min(q.lo, x), q.hi = std::max(q.hi, x);
            }
        });
        for (std::int64_t r = 0; r < lanes; ++r) {
            Neumaier t;
            std::int64_t c = 0;
            bool nan = false;
            double lo = kInf, hi = -kInf;
            for (std::int64_t b = 0; b < nb_; ++b) {
                const P1& q = p1[static_cast<std::size_t>(r * nb_ + b)];
                q.s.into(t);
                c += q.cnt, nan = nan || q.nan, lo = std::min(lo, q.lo), hi = std::max(hi, q.hi);
            }
            pc[r] = omit ? c : n;
            const bool bad = c == 0 || (nan && !omit);
            his[static_cast<std::size_t>(r)] = bad ? kNaN : t.value() / static_cast<double>(c);
            pmn[r] = bad || !minmax ? kNaN : lo;
            pmx[r] = bad || !minmax ? kNaN : hi;
        }
#if defined(__aarch64__)
        if (side)  // a lane without a finite mean gets sums nobody reads
            run_side([&](std::int64_t r0, std::int64_t L, std::int64_t b, std::int64_t j0, std::int64_t j1) {
                std::vector<double> buf(static_cast<std::size_t>(8 * 8 * L), 0.0);
                double *S[4], *C[4];
                for (int k = 0; k < 4; ++k) S[k] = buf.data() + (2 * k) * 8 * L, C[k] = buf.data() + (2 * k + 1) * 8 * L;
                side_moments(a, r0, L, j0, j1, his.data() + r0, kmax, S, C);
                for (std::int64_t l = 0; l < L; ++l) {
                    P2& q = p2v[static_cast<std::size_t>((r0 + l) * nb_ + b)];
                    for (int m = 0; m < 4; ++m)
                        for (int k = 0; k < 8; ++k) q.s[m].s[k] = S[m][k * L + l], q.s[m].c[k] = C[m][k * L + l];
                }
            });
        else
#endif
        run([&](std::int64_t it, std::int64_t r, const char* base, std::int64_t j0, std::int64_t j1) {
            const double h = his[static_cast<std::size_t>(r)];
            if (!std::isfinite(h)) return;
            P2& q = p2v[static_cast<std::size_t>(it)];
            if (s1 == 8) {
                const double* xs = reinterpret_cast<const double*>(base);
#if defined(__aarch64__)
                moment_block(q.s, xs, j0, j1, h, kmax);
#else
                for (std::int64_t j = j0; j < j1; ++j) {
                    const bool nan = std::isnan(xs[j]);
                    const double d = nan ? 0.0 : xs[j] - h, d2 = d * d;
                    q.s[0].add(j, d), q.s[1].add(j, d2);
                    if (kmax >= 3) q.s[2].add(j, d2 * d);
                    if (kmax >= 4) q.s[3].add(j, d2 * d2);
                }
#endif
                return;
            }
            for (std::int64_t j = j0; j < j1; ++j) {
                const double x = *reinterpret_cast<const double*>(base + j * s1);
                if (std::isnan(x)) continue;
                const double d = x - h, d2 = d * d;
                q.s[0].add(j, d), q.s[1].add(j, d2);
                if (kmax >= 3) q.s[2].add(j, d2 * d);
                if (kmax >= 4) q.s[3].add(j, d2 * d2);
            }
        });
        for (std::int64_t r = 0; r < lanes; ++r) {
            const double h = his[static_cast<std::size_t>(r)];
            if (std::isnan(h)) {
                pm[r] = p2[r] = p3[r] = p4[r] = pml[r] = kNaN;
                continue;
            }
            if (!std::isfinite(h)) {  // an infinite mean (inf in the data): no moments
                pm[r] = h, pml[r] = 0.0, p2[r] = p3[r] = p4[r] = kNaN;
                continue;
            }
            Neumaier t[4];
            for (std::int64_t b = 0; b < nb_; ++b)
                for (int k = 0; k < 4; ++k) p2v[static_cast<std::size_t>(r * nb_ + b)].s[k].into(t[k]);
            const double c = static_cast<double>(pc[r]);
            const double S1 = t[0].value() / c, S2 = t[1].value() / c, S3 = t[2].value() / c, S4 = t[3].value() / c;
            const double lo = S1, l2 = lo * lo;
            pm[r] = h + lo;
            pml[r] = lo - (pm[r] - h);
            p2[r] = S2 - l2;
            p3[r] = kmax >= 3 ? S3 - 3.0 * lo * S2 + 2.0 * l2 * lo : kNaN;
            p4[r] = kmax >= 4 ? S4 - 4.0 * lo * S3 + 6.0 * l2 * S2 - 3.0 * l2 * l2 : kNaN;
        }
    }
    return nb::make_tuple(wrap(std::move(cnt)), wrap(std::move(mean)), wrap(std::move(m2)), wrap(std::move(m3)),
                          wrap(std::move(m4)), wrap(std::move(mn)), wrap(std::move(mx)), wrap(std::move(ml)));
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
nb::object rank(nb::handle rows_obj, int method, bool omit) { return wrap(rank_rows(lanes_arg(rows_obj), method, omit)); }

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
    switch (method) {
        case 0:  // Vasicek: mean log(n / (2m) (X[i+m] - X[i-m]))
            for (std::int64_t i = 0; i < n; ++i) s.add(std::log(nd / (2.0 * md) * (X(i + m) - X(i - m))));
            return s.value() / nd;
        case 1: {  // van Es
            for (std::int64_t i = 0; i + m < n; ++i) s.add(std::log((nd + 1.0) / md * (v[i + m] - v[i])));
            Neumaier h;
            for (std::int64_t k = m; k <= n; ++k) h.add(1.0 / static_cast<double>(k));
            return s.value() / (nd - md) + h.value() + std::log(md) - std::log(nd + 1.0);
        }
        case 2:  // Ebrahimi
            for (std::int64_t i = 1; i <= n; ++i) {
                double ci = 2.0;
                if (i <= m) ci = 1.0 + static_cast<double>(i - 1) / md;
                if (i >= n - m + 1) ci = 1.0 + static_cast<double>(n - i) / md;
                s.add(std::log(nd * (X(i - 1 + m) - X(i - 1 - m)) / (ci * md)));
            }
            return s.value() / nd;
        default: {  // Correa
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

void bind_stats_desc(nb::module_& m) {
    m.def("desc_moments", &moments);
    m.def("desc_moment", &moment, nb::arg("rows"), nb::arg("order"), nb::arg("centers").none(), nb::arg("omit"));
    m.def("desc_rank", &rank);
    m.def("desc_mode", &mode);
    m.def("desc_diffent", &diffent);
}

}  // namespace sd
