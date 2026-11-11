// The norm of a set of magnitudes, shared by vector_norm, matrix_norm and cond
// (DESIGN.md §2.11: fast unscaled pass, rescaled second pass only on overflow
// or when every term is tiny; empty sets have norm 0). Sums are pairwise
// (M8, DESIGN.md §2.17).
#pragma once

#include "core/pairwise.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <initializer_list>

namespace sdla {

inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// Below this, p-th powers may be subnormal and lose bits: 2^-969 = DBL_MIN * 2^53.
inline constexpr double kTiny = 0x1p-969;

enum class Ord { Inf, NegInf, Zero, One, Two, P };

struct VecOrd {
    Ord kind;
    double p;
};

// The magnitudes of a strided region, in logical C order (the last axis
// fastest), whatever the memory layout: so results do not depend on it.
// A fill for pairwise_sum: walk(begin, len, out) writes |x| at flat positions
// begin .. begin+len-1. Elements are of storage type S; `mag` maps one to a double.
// No heap memory: a walk is made per norm, so per row of a batch.
inline constexpr int kMaxWalkDims = 33;  // stardust-numpy's 32 axes, plus a complex split

template <class S, class Mag>
class Walk {
public:
    Walk(const std::byte* base, int nd, const std::int64_t* shape, const std::int64_t* strides, Mag mag)
        : base_(base), nd_(nd), mag_(mag) {
        for (int d = 0; d < nd; ++d) {
            shape_[d] = shape[d];
            strides_[d] = strides[d];
        }
    }

    std::int64_t count() const {
        std::int64_t c = 1;
        for (int d = 0; d < nd_; ++d) c *= shape_[d];
        return c;
    }

    void operator()(std::int64_t begin, std::int64_t len, double* out) const {
        if (nd_ == 0) {
            out[0] = mag_(load<S>(base_));
            return;
        }
        std::int64_t idx[kMaxWalkDims];
        const std::byte* p = base_;
        std::int64_t rem = begin;
        for (int d = nd_; d-- > 0;) {
            idx[d] = rem % shape_[d];
            rem /= shape_[d];
            p += idx[d] * strides_[d];
        }
        const int last = nd_ - 1;
        const std::int64_t s = strides_[last];
        constexpr std::int64_t sz = sizeof(S);
        std::int64_t k = 0;
        while (true) {
            const std::int64_t run = std::min(len - k, shape_[last] - idx[last]);
            double* o = out + k;
            if (s == sz) {
                for (std::int64_t t = 0; t < run; ++t) o[t] = mag_(load<S>(p + t * sz));
            } else {
                for (std::int64_t t = 0; t < run; ++t) o[t] = mag_(load<S>(p + t * s));
            }
            k += run;
            if (k == len) return;
            // the run reached the end of the last axis: carry into the outer ones
            p -= idx[last] * s;
            idx[last] = 0;
            int d = last;
            while (d-- > 0) {
                p += strides_[d];
                if (++idx[d] < shape_[d]) break;
                p -= shape_[d] * strides_[d];
                idx[d] = 0;
            }
        }
    }

private:
    const std::byte* base_;
    int nd_;
    std::int64_t shape_[kMaxWalkDims], strides_[kMaxWalkDims];
    Mag mag_;
};

template <class S, class Mag>
Walk<S, Mag> make_walk(const std::byte* base, std::initializer_list<std::int64_t> shape,
                       std::initializer_list<std::int64_t> strides, Mag mag) {
    return Walk<S, Mag>(base, static_cast<int>(shape.size()), shape.begin(), strides.begin(), mag);
}

// |v| of one stored element, as a double (complex: the modulus, by hypot).
template <class S>
double magnitude(S v) {
    if constexpr (std::is_same_v<S, std::uint8_t>) return v ? 1.0 : 0.0;
    else if constexpr (std::is_same_v<S, std::int64_t>) return std::fabs(static_cast<double>(v));
    else if constexpr (std::is_same_v<S, double>) return std::fabs(v);
    else return std::abs(v);
}

// The largest of a[0..n) and whether any is NaN, with 4 independent lanes (one
// compare chain would bind the loop to the compare latency). max is exact, so
// the lane order does not change the result.
inline void max_and_nan(const double* a, std::int64_t n, double& mx, bool& nan) {
    double m0 = mx, m1 = mx, m2 = mx, m3 = mx;
    std::uint64_t bad = 0;
    std::int64_t k = 0;
    for (; k + 4 <= n; k += 4) {
        bad |= (a[k] != a[k]) | (a[k + 1] != a[k + 1]) | (a[k + 2] != a[k + 2]) | (a[k + 3] != a[k + 3]);
        m0 = a[k] > m0 ? a[k] : m0;
        m1 = a[k + 1] > m1 ? a[k + 1] : m1;
        m2 = a[k + 2] > m2 ? a[k + 2] : m2;
        m3 = a[k + 3] > m3 ? a[k + 3] : m3;
    }
    for (; k < n; ++k) {
        bad |= a[k] != a[k];
        m0 = a[k] > m0 ? a[k] : m0;
    }
    m0 = m1 > m0 ? m1 : m0;
    m2 = m3 > m2 ? m3 : m2;
    mx = m2 > m0 ? m2 : m0;
    nan |= bad != 0;
}

// The norm of `count` magnitudes, given by fill(begin, len, out) as for
// pairwise_sum (at most kPairwiseBlock per call). Empty sets have norm 0 (S11).
template <class Fill>
inline double vnorm(VecOrd ord, std::int64_t count, Fill&& fill) {
    if (count == 0) return 0.0;
    auto each_chunk = [&](auto&& f) {
        double buf[kPairwiseBlock];
        for (std::int64_t b = 0; b < count; b += kPairwiseBlock) {
            const std::int64_t len = std::min(kPairwiseBlock, count - b);
            fill(b, len, buf);
            for (std::int64_t k = 0; k < len; ++k) f(buf[k]);
        }
    };
    switch (ord.kind) {
        case Ord::Inf: {
            double m = 0;
            bool nan = false;
            double buf[kPairwiseBlock];
            for (std::int64_t b = 0; b < count; b += kPairwiseBlock) {
                const std::int64_t len = std::min(kPairwiseBlock, count - b);
                fill(b, len, buf);
                max_and_nan(buf, len, m, nan);
            }
            return nan ? kNaN : m;
        }
        case Ord::NegInf: {
            double m = kInf;
            bool nan = false;
            each_chunk([&](double a) {
                if (a != a) nan = true;
                else if (a < m) m = a;
            });
            return nan ? kNaN : m;
        }
        case Ord::Zero: {  // the number of nonzeros (not a norm; Array API). NaN counts.
            std::int64_t c = 0;
            each_chunk([&](double a) { c += (a != 0); });
            return static_cast<double>(c);
        }
        case Ord::One:
            return pairwise_sum<double>(count, fill);
        case Ord::Two:
        case Ord::P: {
            const bool two = ord.kind == Ord::Two;
            const double p = ord.p;
            double amax = 0;
            bool nan = false;
            // one pass: the terms a^p for the sum, and the largest a on the side
            // (kept in locals: `out` could alias the captured ones for the compiler)
            auto powers = [&](std::int64_t b, std::int64_t len, double* out) {
                fill(b, len, out);
                max_and_nan(out, len, amax, nan);
                if (two)
                    for (std::int64_t k = 0; k < len; ++k) out[k] = out[k] * out[k];
                else
                    for (std::int64_t k = 0; k < len; ++k) out[k] = std::pow(out[k], p);
            };
            const double s = pairwise_sum<double>(count, powers);
            if (nan) return kNaN;
            if (p > 0) {
                if (amax == kInf) return kInf;
                const double top = two ? amax * amax : std::pow(amax, p);
                if (std::isinf(s) || (amax > 0 && top < kTiny)) {
                    // Rescaled: amax * (sum (a/amax)^p)^(1/p). Every term is <= 1.
                    auto scaled = [&](std::int64_t b, std::int64_t len, double* out) {
                        fill(b, len, out);
                        for (std::int64_t k = 0; k < len; ++k) {
                            const double r = out[k] / amax;
                            out[k] = two ? r * r : std::pow(r, p);
                        }
                    };
                    const double t = pairwise_sum<double>(count, scaled);
                    return amax * (two ? std::sqrt(t) : std::pow(t, 1 / p));
                }
            }
            return two ? std::sqrt(s) : std::pow(s, 1 / p);
        }
    }
    return kNaN;
}

}  // namespace sdla
