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
#include "array/ops/pieces.hpp"
#include "special/dd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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
    for_pieces(lanes * nb, per, lanes * n, [&](std::int64_t, std::int64_t b0, std::int64_t b1) {
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
        double m = 0.0;
        for (std::int64_t j = j0; j < j1; ++j) {
            double h, l;
            if (src.get(r, j, h, l)) m = std::max(m, std::fabs(h));
        }
        part[static_cast<std::size_t>(it)] = m;
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
        for (std::int64_t j = j0; j < j1; ++j) {
            double h, l;
            if (!src.get(r, j, h, l)) {
                nan = true;
                continue;
            }
            a.add(j, h * f, l * f);
            ++cnt;
        }
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
        for (std::int64_t j = j0; j < j1; ++j) {
            double h, l;
            if (src.get(r, j, h, l)) {
                const DD q = sq_dev(h * f, l * f, mean);
                a.add(j, q.hi, q.lo);
            }
        }
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

// v sorted by value on the pool: fixed pieces sorted side by side, then merged pairwise. The
// pieces depend on the length alone, so the order is the same on any thread count (and equal
// keys are equal values: the ranks do not depend on their order anyway).
inline void sort_keys(std::vector<Key>& v) {
    constexpr std::int64_t kPiece = 1 << 15;
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    if (n <= kPiece) {
        std::sort(v.begin(), v.end(), key_less);
        return;
    }
    const std::int64_t pieces = (n + kPiece - 1) / kPiece;
    for_pieces(pieces, 1, n * 16, [&](std::int64_t, std::int64_t p0, std::int64_t p1) {
        for (std::int64_t p = p0; p < p1; ++p)
            std::sort(v.begin() + p * kPiece, v.begin() + std::min(n, (p + 1) * kPiece), key_less);
    });
    std::vector<Key> buf(v.size());
    for (std::int64_t w = kPiece; w < n; w *= 2) {
        const std::int64_t pairs = (n + 2 * w - 1) / (2 * w);
        for_pieces(pairs, 1, n * 4, [&](std::int64_t, std::int64_t q0, std::int64_t q1) {
            for (std::int64_t q = q0; q < q1; ++q) {
                const std::int64_t lo = q * 2 * w, mid = std::min(n, lo + w), hi = std::min(n, lo + 2 * w);
                std::merge(
                    v.begin() + lo, v.begin() + mid, v.begin() + mid, v.begin() + hi, buf.begin() + lo, key_less
                );
            }
        });
        v.swap(buf);
    }
}

}  // namespace sd::htest
