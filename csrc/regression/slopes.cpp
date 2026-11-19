// The Theil-Sen and Siegel slopes with exact medians (docs/design/regression.md §3.6, G56):
// the values of scipy.stats.theilslopes and siegelslopes (scipy 1.18.1), bit for bit.
//
// The pairwise slopes are the doubles scipy forms, (y_i - y_j) / (x_i - x_j) for x_i != x_j
// (each difference and the quotient rounded once); the medians and the confidence bounds
// are order statistics of those values, selected exactly, and combined as scipy's
// quantile(method='linear') does (0.5 a + 0.5 b for an even count).
//   - Siegel: per point the median of its n - 1 slopes (and, for method='separate', of
//     the intercepts y_i - slope x_i), by selection: O(n^2) time, O(n) memory per task.
//   - Theil-Sen, up to kDirect slopes: all of them in one array, then selection. Beyond:
//     O(n^2) passes over the pairs without storing them. Pivots around each wanted rank
//     come from a sample of pairs; a pass counts the slopes below, at and above them and
//     keeps those strictly between (or, when too many are expected there, a hashed
//     subsample that gives the next, narrower pivots). Two or three passes usually
//     suffice; the counts are exact, so the result is the same whatever the sample.
// Chunks of rows are fixed by n alone and combined in order: the same bits for any
// thread count.
#include "bind/errors.hpp"
#include "linalg/ops/common.hpp"
#include "parallel/pool.hpp"
#include "random/philox.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sd::regression {

namespace nb = nanobind;
using linalg::Batch;
using linalg::Failure;
using linalg::Input;
using linalg::lint;
using linalg::Out;
using linalg::Scratch;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::int64_t kKeyMin = std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t kKeyMax = std::numeric_limits<std::int64_t>::max();
// Slopes stored outright up to kDirect; values kept between a target's pivots up to
// kCollectMax (else a subsample). Only the tests change them (slopes()'s last arguments), to
// run the passes on small problems.
constexpr std::int64_t kDirectDefault = std::int64_t(1) << 23;
constexpr std::int64_t kCollectDefault = std::int64_t(1) << 22;
struct Limits {
    std::int64_t direct, collect;
};
constexpr std::int64_t kSample = 1 << 16;                 // pivots' sample size
constexpr int kTasks = 128;                               // row blocks of a pass

inline double rd(const std::byte* p) {
    double v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

// An order-preserving integer key of a double (no NaN); -0 counts as +0.
inline std::int64_t okey(double v) {
    v = v + 0.0;
    std::int64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b >= 0 ? b : b ^ std::numeric_limits<std::int64_t>::max();
}
inline double unkey(std::int64_t b) {
    if (b < 0) b ^= std::numeric_limits<std::int64_t>::max();
    double v;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

inline std::uint64_t mix(std::uint64_t z) {  // splitmix64's finalizer
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// scipy's quantile(v, 0.5) (method 'linear') of the values in v (reordered), NaN when empty.
double qmedian(double* v, lint c) {
    if (c == 0) return kNaN;
    const lint m = (c - 1) / 2;
    std::nth_element(v, v + m, v + c);
    const double a = v[m];
    if (c % 2) return a;
    const double b = *std::min_element(v + m + 1, v + c);
    return 0.5 * a + 0.5 * b;
}

// Row blocks with about equal numbers of pairs (i < j), fixed by n.
std::vector<lint> row_blocks(lint n) {
    std::vector<lint> at{0};
    const double total = 0.5 * double(n) * double(n - 1);
    double acc = 0.0;
    int t = 1;
    for (lint i = 0; i < n; ++i) {
        acc += double(n - 1 - i);
        if (acc >= total * t / kTasks && i + 1 < n && t < kTasks) {
            at.push_back(i + 1);
            ++t;
        }
    }
    at.push_back(n);
    return at;
}

template <class F>
void for_blocks(const std::vector<lint>& at, F&& body) {
    const lint nb_ = static_cast<lint>(at.size()) - 1;
    if (nb_ >= 2 && double(at.back()) * double(at.back()) >= 1e6)
        parallel::parallel_for(nb_, [&](std::int64_t b) { body(static_cast<lint>(b), at[b], at[b + 1]); });
    else
        for (lint b = 0; b < nb_; ++b) body(b, at[b], at[b + 1]);
}

// One wanted order statistic of the Theil-Sen slopes.
struct Target {
    std::int64_t r;             // rank (0-based) among the non-NaN slopes
    bool done = false;
    double value = kNaN;
    std::int64_t lo, hi;        // pivots (keys; kKeyMin/kKeyMax: none)
    std::int64_t L = kKeyMin, H = kKeyMax;  // the answer lies strictly between
    std::int64_t cleL = 0, cltH = 0;        // #(key <= L), #(key < H)
    bool collect = true;
    double rate = 1.0;          // subsample rate when not collecting
};

struct Pts {
    lint n;
    const double* x;
    const double* y;
};

// The order statistics of ranks `want` (ascending, distinct) among the N non-NaN pairwise
// slopes; also counts the infinite ones (ninf).
std::vector<double> select_slopes(const Pts& P, std::int64_t N, const std::vector<std::int64_t>& want, std::int64_t& ninf,
                                  const Limits& lim) {
    const lint n = P.n;
    const std::int64_t kDirect = lim.direct, kCollectMax = lim.collect;
    std::vector<double> out(want.size(), kNaN);
    ninf = 0;
    if (N <= kDirect) {
        std::vector<double> v;
        v.reserve(static_cast<std::size_t>(N));
        for (lint i = 0; i + 1 < n; ++i)
            for (lint j = i + 1; j < n; ++j) {
                const double dx = P.x[i] - P.x[j];
                if (dx == 0.0) continue;
                const double s = (P.y[i] - P.y[j]) / dx;
                if (std::isnan(s)) continue;  // inf / inf from overflowing differences: no slope
                if (std::isinf(s)) ++ninf;
                v.push_back(s);
            }
        lint from = 0;
        for (std::size_t t = 0; t < want.size(); ++t) {
            const lint r = static_cast<lint>(want[t]);
            std::nth_element(v.begin() + from, v.begin() + r, v.end());
            out[t] = v[static_cast<std::size_t>(r)] + 0.0;  // +0 for a zero (scipy's sign is its sort's)
            from = r;
        }
        return out;
    }
    // pivots from a sample of pairs (Philox with a fixed key: deterministic)
    random::Stream st({0x7e17c5e9ULL, static_cast<std::uint64_t>(n)});
    std::vector<std::int64_t> samp;
    samp.reserve(kSample);
    for (std::uint64_t e = 0; e < 4 * std::uint64_t(kSample) && std::int64_t(samp.size()) < kSample; ++e) {
        const std::uint64_t w = st.first(e);
        const lint i = static_cast<lint>(random::bounded(w, static_cast<std::uint64_t>(n), [&](std::uint64_t q) { return st.extra(e, 2 * q); }));
        const lint j = static_cast<lint>(random::bounded(st.extra(e, 1), static_cast<std::uint64_t>(n),
                                                         [&](std::uint64_t q) { return st.extra(e, 2 * q + 3); }));
        if (i == j) continue;
        const double dx = P.x[i] - P.x[j];
        if (dx == 0.0) continue;
        const double sv = (P.y[i] - P.y[j]) / dx;
        if (std::isnan(sv)) continue;
        samp.push_back(okey(sv));
    }
    std::sort(samp.begin(), samp.end());
    std::vector<Target> T(want.size());
    auto place = [&](Target& t, const std::vector<std::int64_t>& V, std::int64_t below, std::int64_t count) {
        // pivots around the rank within (L, H) from a sorted sample V of its values
        const double m = double(V.size());
        if (V.size() < 64) {
            t.lo = t.L, t.hi = t.H;
            t.collect = count <= kCollectMax;
            t.rate = t.collect ? 1.0 : double(kSample) / double(count);
            return;
        }
        const double pos = (double(t.r - below) + 0.5) / double(count) * m;
        const double delta = 4.0 * std::sqrt(m) + 4.0;
        const std::int64_t il = static_cast<std::int64_t>(std::floor(pos - delta));
        const std::int64_t ih = static_cast<std::int64_t>(std::ceil(pos + delta));
        t.lo = il < 0 ? t.L : V[static_cast<std::size_t>(il)];
        t.hi = ih >= std::int64_t(V.size()) ? t.H : V[static_cast<std::size_t>(ih)];
        const double span = double(std::min<std::int64_t>(ih, V.size()) - std::max<std::int64_t>(il, -1));
        const double E = double(count) * span / m;
        t.collect = E <= double(kCollectMax);
        t.rate = t.collect ? 1.0 : std::min(1.0, double(kSample) / E);
    };
    for (std::size_t t = 0; t < want.size(); ++t) {
        T[t].r = want[t];
        T[t].cltH = N;
        place(T[t], samp, 0, N);
    }
    const std::vector<lint> at = row_blocks(n);
    const lint nbk = static_cast<lint>(at.size()) - 1;
    const std::size_t nt = want.size();
    for (int round = 0; round < 64; ++round) {
        bool any = false;
        for (auto& t : T) any = any || !t.done;
        if (!any) break;
        // counts per block and target: <lo, ==lo, between, ==hi; kept values
        std::vector<std::int64_t> cnt(static_cast<std::size_t>(nbk) * nt * 4, 0), infs(static_cast<std::size_t>(nbk), 0);
        std::vector<std::vector<std::vector<std::int64_t>>> kept(static_cast<std::size_t>(nbk),
                                                                 std::vector<std::vector<std::int64_t>>(nt));
        // per target #(key < lo), #(key <= lo), #(key < hi), #(key <= hi), counted branch-free
        // over a row's keys (a slope with x_i = x_j, NaN, has the key kKeyMax); the values
        // strictly between a target's pivots are rare and go to a slow path
        std::int64_t lo[4], hi[4];
        for (std::size_t t = 0; t < nt; ++t) {
            lo[t] = T[t].done ? kKeyMin : T[t].lo;
            hi[t] = T[t].done ? kKeyMin : T[t].hi;
        }
        const std::int64_t kinfp = okey(std::numeric_limits<double>::infinity());
        const std::int64_t kinfn = okey(-std::numeric_limits<double>::infinity());
        for_blocks(at, [&](lint b, lint i0, lint i1) {
            std::int64_t* c = cnt.data() + static_cast<std::size_t>(b) * nt * 4;
            auto& kv = kept[static_cast<std::size_t>(b)];
            std::vector<std::int64_t> key(static_cast<std::size_t>(n));
            std::int64_t acc[16] = {};
            std::int64_t ni = 0;
            for (lint i = i0; i < i1; ++i) {
                const double xi = P.x[i], yi = P.y[i];
                std::int64_t* kq = key.data();
                for (lint j = i + 1; j < n; ++j) {
                    const double dx = xi - P.x[j];
                    double v = (yi - P.y[j]) / dx + 0.0;
                    std::int64_t bits;
                    std::memcpy(&bits, &v, sizeof bits);
                    bits = bits >= 0 ? bits : bits ^ std::numeric_limits<std::int64_t>::max();
                    kq[j] = (dx == 0.0 || v != v) ? kKeyMax : bits;  // no slope: above every rank wanted
                }
                if (round == 0)
                    for (lint j = i + 1; j < n; ++j) ni += (kq[j] == kinfp) + (kq[j] == kinfn);
                for (std::size_t t = 0; t < nt; ++t) {
                    const std::int64_t l = lo[t], h = hi[t];
                    std::int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    for (lint j = i + 1; j < n; ++j) {
                        const std::int64_t q = kq[j];
                        a0 += q < l;
                        a1 += q <= l;
                        a2 += q < h;
                        a3 += q <= h;
                    }
                    acc[4 * t] += a0, acc[4 * t + 1] += a1, acc[4 * t + 2] += a2, acc[4 * t + 3] += a3;
                    if (!(l < h && a2 > a1)) continue;
                    const Target& g = T[t];  // the row's values strictly between: the slow path
                    for (lint j = i + 1; j < n; ++j) {
                        const std::int64_t q = kq[j];
                        if (!(q > l && q < h)) continue;
                        if (g.collect ||
                            double(mix(std::uint64_t(i) * std::uint64_t(n) + std::uint64_t(j)) >> 11) * 0x1p-53 < g.rate)
                            kv[t].push_back(q);
                    }
                }
            }
            for (std::size_t e = 0; e < 4 * nt; ++e) c[e] = acc[e];
            infs[static_cast<std::size_t>(b)] = ni;
        });
        if (round == 0)
            for (auto v : infs) ninf += v;
        const std::int64_t nanpairs = std::int64_t(n) * (n - 1) / 2 - N;
        for (std::size_t t = 0; t < nt; ++t) {
            Target& g = T[t];
            if (g.done) continue;
            std::int64_t A0 = 0, A1 = 0, A2 = 0, A3 = 0;
            for (lint b = 0; b < nbk; ++b) {
                const std::int64_t* c = cnt.data() + static_cast<std::size_t>(b) * nt * 4 + t * 4;
                A0 += c[0], A1 += c[1], A2 += c[2], A3 += c[3];
            }
            if (g.hi == kKeyMax) A3 -= nanpairs;  // the NaN slopes' key
            const std::int64_t c0 = A0, c1 = A1 - A0;
            const std::int64_t c2 = g.lo < g.hi ? A2 - A1 : 0, c3 = g.lo < g.hi ? A3 - A2 : 0;
            // counts are over all values; the pivots' "below lo" includes those <= L
            if (g.r < c0) {
                g.H = g.lo, g.cltH = c0;
                g.lo = g.L, g.hi = g.H;
                const std::int64_t E = c0 - g.cleL;
                g.collect = E <= kCollectMax;
                g.rate = g.collect ? 1.0 : double(kSample) / double(E);
            } else if (g.r < c0 + c1) {
                g.done = true, g.value = unkey(g.lo);
            } else if (g.r < c0 + c1 + c2) {
                std::vector<std::int64_t> V;
                for (lint b = 0; b < nbk; ++b) {
                    const auto& kv = kept[static_cast<std::size_t>(b)][t];
                    V.insert(V.end(), kv.begin(), kv.end());
                }
                g.L = g.lo, g.cleL = c0 + c1, g.H = g.hi, g.cltH = c0 + c1 + c2;
                if (g.collect || g.rate >= 1.0) {  // V holds every value between the pivots
                    const std::size_t m = static_cast<std::size_t>(g.r - (c0 + c1));
                    std::nth_element(V.begin(), V.begin() + static_cast<std::ptrdiff_t>(m), V.end());
                    g.done = true, g.value = unkey(V[m]);
                } else {
                    std::sort(V.begin(), V.end());
                    place(g, V, g.cleL, c2);
                }
            } else if (g.r < c0 + c1 + c2 + c3) {
                g.done = true, g.value = unkey(g.hi);
            } else {
                g.L = g.hi, g.cleL = c0 + c1 + c2 + c3;
                g.lo = g.L, g.hi = g.H;
                const std::int64_t E = g.cltH - g.cleL;
                g.collect = E <= kCollectMax;
                g.rate = g.collect ? 1.0 : double(kSample) / double(E);
            }
        }
    }
    for (std::size_t t = 0; t < nt; ++t) out[t] = T[t].value + 0.0;
    return out;
}

// The sum over groups of tied values of t (t - 1) (2t + 5) (sorted v), and the number of
// pairs within groups.
void ties(std::vector<double> v, std::int64_t& sum5, std::int64_t& pairs) {
    std::sort(v.begin(), v.end());
    sum5 = 0, pairs = 0;
    for (std::size_t a = 0; a < v.size();) {
        std::size_t b = a + 1;
        while (b < v.size() && v[b] == v[a]) ++b;
        const std::int64_t t = static_cast<std::int64_t>(b - a);
        sum5 += t * (t - 1) * (2 * t + 5);
        pairs += t * (t - 1) / 2;
        a = b;
    }
}

// scipy's theilslopes: (slope, intercept, low_slope, high_slope); z = ndtri(alpha / 2) with
// alpha folded below 1/2 as scipy does.
void theil_one(const Pts& P, bool joint, double z, const Limits& lim, double* o) {
    const lint n = P.n;
    std::vector<double> xs(P.x, P.x + n), ys(P.y, P.y + n);
    std::int64_t sx, px, sy, py;
    ties(xs, sx, px);
    ties(ys, sy, py);
    // the pairs with a slope: x_i != x_j, and not inf / inf, which needs a difference of x that
    // overflows (counted only then; scipy's slopes are the finite and infinite ones, G56)
    std::int64_t nnan = 0;
    if (n > 1 && std::isinf(*std::max_element(P.x, P.x + n) - *std::min_element(P.x, P.x + n)))
        for (lint i = 0; i + 1 < n; ++i)
            for (lint j = i + 1; j < n; ++j) {
                const double dx = P.x[i] - P.x[j];
                nnan += dx != 0.0 && std::isnan((P.y[i] - P.y[j]) / dx);
            }
    const std::int64_t N = std::int64_t(n) * (n - 1) / 2 - px - nnan;
    if (N == 0) {
        std::fill(o, o + 4, kNaN);
        return;
    }
    const double ny = double(n);
    const double sigsq = 1 / 18. * ((ny * (ny - 1) * (2 * ny + 5) - double(sx)) - double(sy));
    const double sigma = std::sqrt(std::max(sigsq, 0.0));
    auto ci_ranks = [&](double nt, std::int64_t& rl, std::int64_t& ru) {
        ru = std::min(static_cast<std::int64_t>(std::nearbyint((nt - z * sigma) / 2.)), static_cast<std::int64_t>(nt) - 1);
        rl = std::max(static_cast<std::int64_t>(std::nearbyint((nt + z * sigma) / 2.)) - 1, std::int64_t(0));
    };
    std::vector<std::int64_t> want;
    const std::int64_t m = (N - 1) / 2;
    want.push_back(m);
    if (N % 2 == 0) want.push_back(m + 1);
    std::int64_t rl, ru;
    ci_ranks(double(N), rl, ru);
    want.push_back(rl);
    if (ru >= 0) want.push_back(ru);
    std::vector<std::int64_t> uw(want);
    std::sort(uw.begin(), uw.end());
    uw.erase(std::unique(uw.begin(), uw.end()), uw.end());
    std::int64_t ninf = 0;
    std::vector<double> got = select_slopes(P, N, uw, ninf, lim);
    auto value = [&](std::int64_t r) {
        return got[static_cast<std::size_t>(std::lower_bound(uw.begin(), uw.end(), r) - uw.begin())];
    };
    double slope = value(m);
    if (N % 2 == 0) slope = 0.5 * slope + 0.5 * value(m + 1);
    double lo = value(rl), hi = ru >= 0 ? value(ru) : kNaN;
    if (ninf > 0) {  // the bounds' ranks count the finite slopes only (scipy's nt)
        std::int64_t rl2, ru2;
        ci_ranks(double(N - ninf), rl2, ru2);
        std::vector<std::int64_t> w2{rl2};
        if (ru2 >= 0) w2.push_back(ru2);
        std::sort(w2.begin(), w2.end());
        w2.erase(std::unique(w2.begin(), w2.end()), w2.end());
        std::int64_t dummy;
        std::vector<double> g2 = select_slopes(P, N, w2, dummy, lim);
        auto v2 = [&](std::int64_t r) {
            return g2[static_cast<std::size_t>(std::lower_bound(w2.begin(), w2.end(), r) - w2.begin())];
        };
        lo = v2(rl2);
        hi = ru2 >= 0 ? v2(ru2) : kNaN;
    }
    if (sigsq < 0) lo = hi = kNaN;
    double inter;
    if (joint) {
        std::vector<double> v(static_cast<std::size_t>(n));
        for (lint i = 0; i < n; ++i) v[i] = P.y[i] - slope * P.x[i];
        inter = qmedian(v.data(), n);
    } else {
        inter = qmedian(ys.data(), n) - slope * qmedian(xs.data(), n);
    }
    o[0] = slope, o[1] = inter, o[2] = lo, o[3] = hi;
}

// scipy's siegelslopes: (slope, intercept), repeated medians.
void siegel_one(const Pts& P, bool separate, double* o) {
    const lint n = P.n;
    std::vector<double> ms(static_cast<std::size_t>(n)), mi(separate ? static_cast<std::size_t>(n) : 0);
    constexpr lint kRows = 32;
    const lint nb_ = (n + kRows - 1) / kRows;
    auto block = [&](std::int64_t b) {
        std::vector<double> v(static_cast<std::size_t>(n)), w(separate ? static_cast<std::size_t>(n) : 0);
        for (lint i = static_cast<lint>(b) * kRows; i < std::min<lint>(n, (static_cast<lint>(b) + 1) * kRows); ++i) {
            lint c = 0;
            for (lint j = 0; j < n; ++j) {
                const double dx = P.x[i] - P.x[j];
                if (dx == 0.0) continue;
                const double s = (P.y[i] - P.y[j]) / dx;
                if (std::isnan(s)) continue;  // inf / inf: no slope (as theilslopes)
                if (separate) w[c] = P.y[i] - s * P.x[i];
                v[c++] = s;
            }
            ms[i] = qmedian(v.data(), c);
            if (separate) mi[i] = qmedian(w.data(), c);
        }
    };
    if (nb_ >= 2 && double(n) * double(n) >= 1e6) parallel::parallel_for(nb_, block);
    else
        for (lint b = 0; b < nb_; ++b) block(b);
    auto med = [&](std::vector<double>& v) {  // NaN propagates (scipy's median)
        for (double a : v)
            if (std::isnan(a)) return kNaN;
        return qmedian(v.data(), static_cast<lint>(v.size()));
    };
    const double slope = med(ms);
    double inter;
    if (separate) {
        inter = med(mi);
    } else {
        std::vector<double> v(static_cast<std::size_t>(n));
        for (lint i = 0; i < n; ++i) v[i] = P.y[i] - slope * P.x[i];
        inter = med(v);
    }
    o[0] = slope, o[1] = inter;
}

std::vector<std::int64_t> with(std::vector<std::int64_t> shape, std::initializer_list<std::int64_t> tail) {
    shape.insert(shape.end(), tail);
    return shape;
}

// slopes(func, y (..., n), x (..., n), kind (0 Theil-Sen, 1 Siegel), alt (Theil-Sen: joint;
// Siegel: separate), z) -> (..., 4) Theil-Sen, (..., 2) Siegel. NaN in a regression's data
// gives NaN; infinity is an error.
nb::object slopes(const std::string& func, nb::handle yo, nb::handle xo, int kind, bool alt, double z,
                  std::int64_t direct_max, std::int64_t collect_max) {
    const char* fn = func.c_str();
    const Limits lim{direct_max >= 0 ? direct_max : kDirectDefault, collect_max >= 0 ? collect_max : kCollectDefault};
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), x(xo, fn, "x");
        Batch b({{y.batch_shape(1), y.batch_strides(1)}, {x.batch_shape(1), x.batch_strides(1)}}, fn);
        const lint n = y.shape(-1);
        const lint w = kind == 0 ? 4 : 2;
        Out oo = linalg::alloc_c(with(b.shape(), {w}), DType::Float64);
        const double element_ns = 100.0 + 3.0 * double(n) * double(n);
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch&) {
            double* o = oo.as<double>() + i * w;
            std::vector<double> xv(static_cast<std::size_t>(n)), yv(static_cast<std::size_t>(n));
            bool nan = false;
            for (lint t = 0; t < n; ++t) {
                yv[t] = rd(y.data() + off[0] + t * y.stride(-1));
                xv[t] = rd(x.data() + off[1] + t * x.stride(-1));
                if (std::isinf(yv[t])) return 1;
                if (std::isinf(xv[t])) return 2;
                if (std::isnan(yv[t]) || std::isnan(xv[t])) nan = true;
            }
            if (nan) {
                std::fill(o, o + w, kNaN);
                return 0;
            }
            Pts P{n, xv.data(), yv.data()};
            if (kind == 0) theil_one(P, alt, z, lim, o);
            else siegel_one(P, alt, o);
            for (lint c = 0; c < w; ++c) o[c] += 0.0;  // +0 for a zero (scipy's sign is its sort's)
            return 0;
        });
        if (f) {
            const char* what = f.code == 1 ? "'y' contains infinity" : "'x' contains infinity";
            throw nb::value_error(linalg::failure_message(b, f.index, (std::string(fn) + "(): " + what).c_str()).c_str());
        }
        return linalg::to_array(oo);
    });
}

}  // namespace

void bind_slopes(nb::module_& m) {
    m.def("slopes", &slopes, nb::arg("func"), nb::arg("y").none(), nb::arg("x").none(), nb::arg("kind"), nb::arg("alt"),
          nb::arg("z"), nb::arg("direct_max") = -1, nb::arg("collect_max") = -1);
}

}  // namespace sd::regression
