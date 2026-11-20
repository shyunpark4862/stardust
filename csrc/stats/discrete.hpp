// What the discrete distributions share (docs/design/stats.md T25, §3.2):
//
//   * the quantile search: the smallest k with P(X <= k) >= p (or P(X > k) <= p),
//     comparing on the side whose probability is the small one (1 - p is exact for
//     p >= 1/2) and on the log side below the normal range, by galloping from a
//     guess and bisecting -- O(log distance) evaluations, the answer exact up to
//     the cdf's own last bits;
//   * tail sums for the distributions with no closed-form cdf (hypergeometric,
//     beta-binomial): the terms relative to the first by their ratio recurrence, in
//     double-double (tail_sum_dd), log of the sum kept apart from the log pmf so the log
//     tails go below the normal range;
//   * the entropy -sum pmf log pmf over the support, from the mode outwards, the terms by the
//     ratio recurrence in double-double, compensated;
//   * the bounds: every sum and every search here runs for at most a fixed number of steps
//     (kMaxTailTerms, kMaxEntropyTerms, kMaxMomentTerms, kMaxSearchSteps), and a sum that
//     would need more is NaN, never the part that was summed;
//   * the per-thread memo of a method's values at integers (KMemo): data of a discrete
//     distribution repeats its values, and with scalar parameters each distinct k is then
//     worked out once. A memo only ever returns what the method's one definition gave for
//     that k and those parameters, so the bits do not depend on it.
#pragma once

#include "special/ops/distr.hpp"
#include "stats/chunk.hpp"
#include "stats/engine.hpp"
#include "stats/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace sd::stats::disc {

// k is an integer (finite, no fractional part).
inline bool is_int(double k) { return std::isfinite(k) && std::floor(k) == k; }

// l k passes the float64 range (l a log, k a count): a double-double product l k is then
// inf - inf, a NaN, where the log pmf it stands for is -inf.
inline bool overflows(double l, double k) { return !(std::fabs(l * k) < kInf); }
// That needs a k from here on: the log of a float64 probability is at most 745 in size
// (and log(1 - p) at most 37), so below 2^1000 the product is in range and nothing is tested.
inline constexpr double kFarCount = 0x1p1000;

// ---- the bounds of the sums and the searches ----------------------------------------------------
//
// A method of a discrete distribution returns after a fixed number of steps for any parameters
// (docs/api/stats.md §2.3). Inside the guaranteed parameter region no bound below is reached;
// outside it a sum that would pass its bound is not formed and the method returns NaN (a cut
// sum would be a wrong value with nothing to show for it).
//   * kMaxTailTerms: the terms of one tail sum of a cdf (hypergeometric, beta-binomial). The
//     supports of the guaranteed region have at most 3001 points; 2^20 terms take ~7 ms.
//   * kMaxEntropyTerms: the terms of an entropy sum. The widest of the guaranteed region is
//     the negative binomial's at n = 1e4, p = 1e-4 (sd 1e6): 1.9e7 terms, 0.19 s; 2^25 take
//     ~0.35 s.
//   * kMaxEntropyVariance: a sum over a variance above this is not started (it covers at least
//     16 sd, 2^25 terms at sd 2^21; the guaranteed region's largest variance is 1e12).
//   * kMaxMomentTerms: the terms of a sum of central moments (a log pmf each). The sums in
//     use end after ~200 terms (the logarithmic series for p <= 1/2) or 4 (a hypergeometric
//     total up to 3).
//   * kMaxSearchSteps: the doublings of a quantile search in one direction, and its halvings.
//     The distance from the start doubles from 1 and float64 ends below 2^1024, so 1025
//     doublings reach any quantile; the bracket then halves down to two neighbouring
//     float64s, at most 1024 + 53 times.
inline constexpr double kMaxTailTerms = 0x1p20;
inline constexpr int kMaxEntropyTerms = 1 << 25;
inline constexpr double kMaxEntropyVariance = 0x1p42;
inline constexpr int kMaxMomentTerms = 1 << 12;
inline constexpr int kMaxSearchSteps = 1100;
// Integers up to here have float64 neighbours one apart with room for any sum's walk.
inline constexpr double kMaxWalkStart = 0x1p52;
// Counts up to here have float64 neighbours, and so have the sums of two of them. A
// distribution over counts past it (a binomial n, a hypergeometric total) cannot form n - k,
// k + 1 or n + 1 (which the incomplete beta of (n - k, k + 1) works with): the methods that
// need them are NaN there.
inline constexpr double kMaxCount = 0x1p52;

// A cdf of an unbounded count through an incomplete function of k + 1: from k = 2^53 on,
// k + 1 is not a float64 (it rounds to k or k + 2) and the value is P(X <= k - 1) or
// P(X <= k + 1), off by one pmf. The probability v (the log probability l) stands where
// that pmf is below 2^-56 of it (of |l| e^l, the size of a log's change), and is NaN
// otherwise. Far in a tail, where k passes 2^53 in the guaranteed region, the pmf is 0.
inline double checked_past_counts(double v, double pmf_k) { return pmf_k <= 0x1p-56 * v ? v : kNaN; }
inline double checked_log_past_counts(double l, double pmf_k) {
    return pmf_k == 0.0 || pmf_k <= 0x1p-56 * (std::fabs(l) * std::exp(l)) ? l : kNaN;
}

// ---- the memo of values at integers ----------------------------------------------------------
//
// A direct-mapped table k -> value for one function of one parameter set (up to three
// parameters, compared by their bits), one per thread and per (distribution, function):
// memo<Tag>(). A new parameter set empties it. The values stored are the function's own, so
// a hit and a fresh evaluation are the same bits (f(x)[i] == f(x[i]) holds through it); what
// it saves is the evaluation: an array of 1e5 binomial(40, p) counts has 41 distinct values.
struct KMemo {
    static constexpr int kSlots = 1024;
    double par[3];
    double key[kSlots];
    double val[kSlots];
    KMemo() {
        par[0] = par[1] = par[2] = kNaN;
        clear();
    }
    void clear() {
        for (double& k : key) k = kNaN;  // a NaN key matches nothing
    }
    // Point the memo at a parameter set (unused entries of p: 0).
    void bind(const double* p) {
        if (std::memcmp(par, p, sizeof par) == 0) return;
        std::memcpy(par, p, sizeof par);
        clear();
    }
    static int slot(double k) {
        return std::fabs(k) < 0x1p62
                   ? static_cast<int>(static_cast<std::uint64_t>(static_cast<std::int64_t>(k)) & (kSlots - 1))
                   : 0;
    }
    bool find(double k, double& v) const {
        const int s = slot(k);
        if (key[s] != k) return false;
        v = val[s];
        return true;
    }
    void put(double k, double v) {
        const int s = slot(k);
        key[s] = k;
        val[s] = v;
    }
    template <class F>
    double get(double k, F&& f) {
        double v;
        if (find(k, v)) return v;
        v = f();
        put(k, v);
        return v;
    }
};
template <class Tag>
KMemo& memo() {
    thread_local KMemo m;
    return m;
}

// Below this many elements a call does not touch a memo (a scalar call would only pay for
// the lookup; same bits either way).
inline constexpr int kMemoMin = 8;

template <std::size_t NP>
bool all_same(const Chunk<NP>& c) {
    bool s = true;
    for (bool b : c.same) s = s && b;
    return s;
}
template <std::size_t NP>
void bind(KMemo& m, const Chunk<NP>& c) {
    double p[3] = {0.0, 0.0, 0.0};
    for (std::size_t k = 0; k < NP; ++k) p[k] = c.p[k][0];
    m.bind(p);
}

// A chunk kernel from a scalar definition f(x, params...): NaN where the parameters are
// outside the domain or x is NaN (f is only called inside its domain, as by the engine).
template <class D, std::size_t NP, class F, std::size_t... I>
void scalar_kernel_impl(const Chunk<NP>& c, double* out, F&& f, std::index_sequence<I...>) {
    if (all_same(c)) {
        if (!D::valid(c.p[I][0]...)) {
            for (int i = 0; i < c.n; ++i) out[i] = kNaN;
            return;
        }
        for (int i = 0; i < c.n; ++i) out[i] = c.x[i] != c.x[i] ? kNaN : f(c.x[i], c.p[I][0]...);
        return;
    }
    for (int i = 0; i < c.n; ++i) out[i] = !D::valid(c.p[I][i]...) || c.x[i] != c.x[i] ? kNaN : f(c.x[i], c.p[I][i]...);
}
template <class D, std::size_t NP, class F>
void scalar_kernel(const Chunk<NP>& c, double* out, F&& f) {
    scalar_kernel_impl<D>(c, out, f, std::make_index_sequence<NP>{});
}

// The chunk kernel `kernel` (the method's definition; safe on invalid parameters) behind the
// memo of Tag: with scalar parameters each block of 32 elements looks its values up and
// hands the misses to the kernel as one small chunk, whose values go into the memo. For a
// cdf-like method (floored) the key is floor(x), which is all such a method reads of x.
// A chunk that mostly misses after 128 elements (the values do not repeat: a wide
// distribution) goes to the kernel directly.
template <class Tag, std::size_t NP, class Kernel>
void memoized(const Chunk<NP>& c, double* out, bool floored, Kernel&& kernel) {
    if (c.n < kMemoMin || !all_same(c)) {
        kernel(c, out);
        return;
    }
    KMemo& m = memo<Tag>();
    bind(m, c);
    constexpr int kBlock = 32;
    double kx[kBlock], kv[kBlock];
    int at[kBlock];
    int seen = 0, missed = 0;
    for (int off = 0; off < c.n; off += kBlock) {
        const int nb = std::min(kBlock, c.n - off);
        int miss = 0;
        for (int i = 0; i < nb; ++i) {
            const double k = floored ? std::floor(c.x[off + i]) : c.x[off + i];
            if (m.find(k, out[off + i])) continue;
            kx[miss] = k;
            at[miss++] = off + i;
        }
        if (miss > 0) {
            const Chunk<NP> sub{miss, kx, c.p, c.same};
            kernel(sub, kv);
            for (int j = 0; j < miss; ++j) {
                out[at[j]] = kv[j];
                if (kx[j] == kx[j]) m.put(kx[j], kv[j]);
            }
        }
        seen += nb;
        missed += miss;
        if (seen >= 128 && 4 * missed > 3 * seen && off + nb < c.n) {
            const int rest = off + nb;
            const Chunk<NP> tail{c.n - rest, c.x + rest, c.p, c.same};  // the parameters are scalars
            kernel(tail, out + rest);
            return;
        }
    }
}

// Below this a probability is compared on the log side.
inline constexpr double kTinyP = 0x1p-1000;

// The smallest integer k in [lo, hi] with pred(k), pred monotone (false, then true)
// and true at hi (hi may be inf; lo is finite). At most 2 kMaxSearchSteps + 2 evaluations of
// pred: one at the start, one per doubling of the step away from it (the step is inf after
// 1024 doublings, which ends either run), one at the largest float64, and one per halving of
// the bracket (it ends at two neighbouring float64s, 1024 + 53 halvings at most).
//
// pred never sees a NaN: a cdf that is not known (a sum past its bound) is the business of
// the caller's functions (Memoized, below), which then answer without evaluating, so that
// the search runs out in arithmetic alone.
template <class P>
double smallest(double lo, double hi, double guess, P&& pred) {
    double g = std::floor(guess);
    if (!(g >= lo)) g = lo;
    if (g > hi) g = hi;
    double a, b;  // pred(a) false (a = lo - 1 counts as false), pred(b) true
    if (pred(g)) {
        b = g;
        a = lo - 1.0;
        double step = 1.0;
        for (int i = 0; i < kMaxSearchSteps && b > lo; ++i, step *= 2.0) {
            const double c = std::max(lo, b - step);
            if (!pred(c)) {
                a = c;
                break;
            }
            b = c;
        }
    } else {
        a = g;
        b = hi;
        double step = 1.0;
        for (int i = 0; i < kMaxSearchSteps; ++i, step *= 2.0) {
            const double c = a + step;
            if (c >= hi) break;
            if (!(c < 0x1p1023)) {
                // The last doubling would pass 2^1023: the largest double decides whether the
                // quantile is still finite (Zipf a = 2.00008 at q = 1.2e-308: 4.9e307).
                const double top = std::numeric_limits<double>::max();
                if (!pred(top)) return kInf;  // the quantile is beyond float64 (an infinite support)
                b = top;
                break;
            }
            if (pred(c)) {
                b = c;
                break;
            }
            a = c;
        }
    }
    for (int i = 0; b - a > 1.0; ++i) {
        if (i >= kMaxSearchSteps) return kNaN;  // not reached: the bracket halves each time
        const double m = std::floor(a + 0.5 * (b - a));
        if (m <= a || m >= b) break;  // past 2^53 no float64 lies between: b is the answer rounded
        if (pred(m)) b = m;
        else a = m;
    }
    return b;
}

// a <= b for double-doubles (normalized: |lo| <= ULP(hi) / 2).
inline bool dd_le(num::DD a, num::DD b) { return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo); }

// The four inverses of a discrete distribution from its cdf, ccdf and their logs at
// integers (F: struct with cdf(k), ccdf(k), logcdf(k), logccdf(k), lo, hi, guess; with
// logccdf_dd(k), the log of a deep upper tail is compared as a double-double: a large
// quantile there moves by ULP(log q) / (k |d log ccdf / dk|), 1e-13 / (a - 1) for Zipf).
template <class F>
double icdf(double p, const F& f) {
    if (!(p >= 0.0 && p <= 1.0)) return kNaN;
    if (p == 0.0) return f.lo;
    if (p == 1.0) return f.hi;
    if (p > 0.5) {
        const double q = 1.0 - p;  // exact
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.ccdf(k) <= q; });
    }
    if (p >= kTinyP) return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.cdf(k) >= p; });
    const double lp = std::log(p);
    return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logcdf(k) >= lp; });
}
template <class F>
double iccdf(double q, const F& f) {
    if (!(q >= 0.0 && q <= 1.0)) return kNaN;
    if (q == 0.0) return f.hi;
    if (q == 1.0) return f.lo;
    if (q > 0.5) {
        const double p = 1.0 - q;
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.cdf(k) >= p; });
    }
    if (q >= kTinyP) return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.ccdf(k) <= q; });
    if constexpr (requires { f.logccdf_dd(0.0); }) {  // log q ~ -745: compare below its ULP
        const num::DD lq = num::log_dd(q);
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return dd_le(f.logccdf_dd(k), lq); });
    }
    const double lq = std::log(q);
    return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logccdf(k) <= lq; });
}
template <class F>
double ilogcdf(double lp, const F& f) {
    if (!(lp <= 0.0)) return kNaN;
    if (lp == 0.0) return f.hi;
    if (lp == -kInf) return f.lo;
    if (lp <= -num::kLn2.hi) return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logcdf(k) >= lp; });
    return iccdf(-std::expm1(lp), f);  // P(X > k) <= 1 - e^lp, the small side
}
template <class F>
double ilogccdf(double lq, const F& f) {
    if (!(lq <= 0.0)) return kNaN;
    if (lq == 0.0) return f.lo;
    if (lq == -kInf) return f.hi;
    if (lq <= -num::kLn2.hi) {
        if constexpr (requires { f.logccdf_dd(0.0); })
            return smallest(f.lo, f.hi, f.guess, [&](double k) { return dd_le(f.logccdf_dd(k), num::DD{lq, 0.0}); });
        return smallest(f.lo, f.hi, f.guess, [&](double k) { return f.logccdf(k) <= lq; });
    }
    return icdf(-std::expm1(lq), f);
}

// The ratio pmf(j + dir) / pmf(j) as its four factors (a1 a2) / (b1 b2), each exact as a
// double-double (a count; a count plus a shape by two_sum).
struct Ratio4 {
    num::DD a1, a2, b1, b2;
};

// sum_j pmf(j) / pmf(k) over j = k, k + dir, ... up to `end` (inclusive) with the terms by
// their ratio recurrence in double-double: t_{j+dir} = t_j (a1 a2) / (b1 b2), every product
// and quotient to ~2^-104, so thousands of terms need no resynchronizing from the log pmf
// (which cost a double-double log pmf every 8 terms), and the sum compensated. Stops once a
// term is below 2^-60 of the sum unless `full` (a pmf that may rise again). Returns a
// negative value if a term fell below 1e-200 in a full sum (the caller then takes the
// resynchronizing sum, which recovers from an underflow).
//
// At most kMaxTailTerms terms: NaN if the sum has not ended by then (a full sum over more
// points is not started). k and end are integers below 2^53 (the callers' counts are exact).
template <class Ratio>
double tail_sum_dd(double k, double dir, double end, Ratio&& ratio, bool full) {
    // where the walk ends: at `end`, or at the last term the bound allows
    const double last = dir > 0.0 ? std::min(end, k + kMaxTailTerms) : std::max(end, k - kMaxTailTerms);
    if (full && last != end) return kNaN;
    struct {
        double s = 1.0, c = 0.0;
        void add(double x) {
            const double t = s + x;
            c += s >= x ? (s - t) + x : (x - t) + s;  // Neumaier (a full sum's terms may rise)
            s = t;
        }
    } sum;
    num::DD t{1.0, 0.0};
    for (double j = k; dir > 0.0 ? j < last : j > last; j += dir) {
        const Ratio4 r = ratio(j);
        t = t * ((r.a1 * r.a2) / (r.b1 * r.b2));
        sum.add(t.hi);
        sum.c += t.lo;
        if (full) {
            if (t.hi < 1e-200) return -1.0;
        } else if (t.hi <= sum.s * 0x1p-60) {
            return sum.s + sum.c;
        }
    }
    if (last != end) return kNaN;  // the bound, with terms that still count beyond it
    return sum.s + sum.c;
}

// log sum_{j} pmf(j) / pmf(k) over j = k, k + dir, k + 2 dir, ... up to `end` (inclusive),
// the terms decreasing away from k (ratio(j): the factors of pmf(j + dir) / pmf(j)); NaN if
// the sum has not ended after kMaxTailTerms terms (tail_sum_dd).
template <class Ratio>
double log_tail_factor(double k, double dir, double end, Ratio&& ratio) {
    return std::log(tail_sum_dd(k, dir, end, ratio, false));
}

// ---- the inverses behind a memo -----------------------------------------------------------
//
// A search compares cdf(k) or ccdf(k) with the target at a handful of integers near the
// quantile; over an array with scalar parameters the same integers come up again and again.
// InvMemo keeps both functions' values (the search's own cdf and ccdf: the precise scalar
// definitions), Memoized wraps a distribution's Fns with it (m = nullptr: no memo).
struct InvMemo {
    KMemo cdf, ccdf;
};
template <class D>
InvMemo& inv_memo() {
    thread_local InvMemo m;
    return m;
}
// The memo of D for a chunk with scalar parameters (bound to them), else nullptr.
template <class D, std::size_t NP>
InvMemo* inv_memo_for(const Chunk<NP>& c) {
    if (c.n < kMemoMin || !all_same(c)) return nullptr;
    InvMemo& m = inv_memo<D>();
    bind(m.cdf, c);
    bind(m.ccdf, c);
    return &m;
}
template <class F>
struct Memoized {
    F f;
    InvMemo* m;
    double lo, hi, guess;
    // Set when a value comes back NaN (a sum past its bound): the quantile is then NaN
    // (inverse()). From there on nothing more is evaluated: every value asked for is 0, and
    // the search runs out in arithmetic alone (its steps are bounded: smallest()).
    mutable bool unknown = false;
    Memoized(const F& f_, InvMemo* m_) : f(f_), m(m_), lo(f_.lo), hi(f_.hi), guess(f_.guess) {}
    double known(double v) const {
        if (v == v) return v;
        unknown = true;
        return 0.0;
    }
    // A value the memo holds comes back inline (never a NaN: those are not kept). Anything
    // else is evaluated out of line, so the search's loop is the same few instructions
    // whatever the size of the distribution's cdf.
    [[gnu::always_inline]] double cdf(double k) const {
        double v;
        return m && m->cdf.find(k, v) ? v : cdf_evaluated(k);
    }
    [[gnu::always_inline]] double ccdf(double k) const {
        double v;
        return m && m->ccdf.find(k, v) ? v : ccdf_evaluated(k);
    }
    [[gnu::noinline]] double cdf_evaluated(double k) const {
        if (unknown) return 0.0;
        const double v = f.cdf(k);
        if (m && v == v) m->cdf.put(k, v);
        return known(v);
    }
    [[gnu::noinline]] double ccdf_evaluated(double k) const {
        if (unknown) return 0.0;
        const double v = f.ccdf(k);
        if (m && v == v) m->ccdf.put(k, v);
        return known(v);
    }
    double logcdf(double k) const { return unknown ? 0.0 : known(f.logcdf(k)); }
    double logccdf(double k) const { return unknown ? 0.0 : known(f.logccdf(k)); }
    num::DD logccdf_dd(double k) const
        requires requires { f.logccdf_dd(k); }
    {
        if (unknown) return {0.0, 0.0};
        const num::DD v = f.logccdf_dd(k);
        return {known(v.hi), unknown ? 0.0 : v.lo};
    }
};

enum class Inv { icdf, iccdf, ilogcdf, ilogccdf };
// The inverse `which` of f at t; NaN where the search met a cdf that is not known.
template <class F>
double inverse(Inv which, double t, const F& f) {
    double k;
    switch (which) {
        case Inv::icdf: k = icdf(t, f); break;
        case Inv::iccdf: k = iccdf(t, f); break;
        case Inv::ilogcdf: k = ilogcdf(t, f); break;
        default: k = ilogccdf(t, f); break;
    }
    return f.unknown ? kNaN : k;
}

// The quantile k that the search found (from its exact comparisons, T25) checked against
// the methods' own doubles one step below (user decision 2026-11-20, D-13): where the cdf
// method at k - 1 already reaches the target, k - 1 (iccdf: the ccdf at k - 1 already at or
// below it; the log forms likewise). A target that is a rounded cdf(k) then gives k back
// (icdf(cdf(k)) == k). k - 1 is taken only where it is also one of T25's answers (an exact
// quantile of a target within relative 1e-14 of the smaller side): for icdf at p > 1/2 the
// side is 1 - p, which the rounding of cdf(k - 1) near 1 can pass by far (Poisson(0.0025) at
// p = 1 - 7.8e-16: cdf(4) rounds to p, ccdf(4) = 8.1e-16), so there the ccdf at k - 1 must be
// within 1e-15 of 1 - p too (iccdf at q > 1/2 likewise with the cdf). The log forms are
// relative in t itself, which the methods' rounding keeps. A smaller side (for the log forms
// |t|) below kTinyP is left as found: a subnormal tail's double is not relative to it
// (LogSeries at q = 1.2e-320, ilogcdf(-5e-324)). One or two method evaluations; NaN, infinities and the
// support's lower end pass.
inline constexpr double kSettleSlack = 1e-15;
template <class D, class... P>
double settle(Inv which, double t, double k, P... p) {
    if (!(k == k) || std::isinf(k)) return k;
    const double km = k - 1.0;
    if (!(km >= D::support_lo(p...)) || km + 1.0 != k) return k;  // (past 2^53 k - 1 is no double)
    const bool log = which == Inv::ilogcdf || which == Inv::ilogccdf;
    if (!((log ? -t : std::min(t, 1.0 - t)) >= kTinyP) || t == -kInf) return k;  // (the ends are exact)
    switch (which) {
        case Inv::icdf:
            if (!(method_of<D, method::Cdf>(km, p...) >= t)) return k;
            return t <= 0.5 || method_of<D, method::Ccdf>(km, p...) <= (1.0 - t) * (1.0 + kSettleSlack) ? km : k;
        case Inv::iccdf:
            if (!(method_of<D, method::Ccdf>(km, p...) <= t)) return k;
            return t <= 0.5 || method_of<D, method::Cdf>(km, p...) >= (1.0 - t) * (1.0 - kSettleSlack) ? km : k;
        case Inv::ilogcdf: return method_of<D, method::Logcdf>(km, p...) >= t ? km : k;
        default: return method_of<D, method::Logccdf>(km, p...) <= t ? km : k;
    }
}

// The search's start for a distribution that is wide (sd >= kWideSd: a start at the mean
// would be thousands of steps from a tail quantile, ~2 log2 of them evaluations): the
// Cornish-Fisher estimate mean + sd (z + skew (z^2 - 1) / 6), z the standard normal quantile
// of the probability asked for. Only where the search begins: the answer is the search's.
inline constexpr double kWideSd = 8.0;
template <class F>
void normal_start(F& f, Inv which, double t, double mean, double sd, double skew) {
    using special::distr_detail::one;
    double z;
    switch (which) {
        case Inv::icdf: z = one<special::Ndtri>(t); break;
        case Inv::iccdf: z = -one<special::Ndtri>(t); break;
        case Inv::ilogcdf: z = special::NdtriExp::f(t); break;
        default: z = -special::NdtriExp::f(t); break;
    }
    const double g = std::floor(mean + sd * (z + skew * (z * z - 1.0) / 6.0));
    if (g >= f.lo && g <= f.hi && g < 0x1p1023) f.guess = g;
}

// ---- log tails from both tails ------------------------------------------------------------
//
// The chunk kernel of logcdf (upper: logccdf) for a distribution whose two tails come from
// one incomplete beta or gamma: sides(i, cdf, ccdf) gives element i's P(X <= k) and P(X > k),
// each to its own relative accuracy, or returns false for an element left to the scalar
// definition precise(i) (the ends of the support, degenerate parameters). The rule is the
// scalar definitions' (beta_detail::log_tail, gamma_detail::log_tail): the side below 1/2 as
// log(small); the other as log(1 - small) = log(w) + e / w, w = fl(1 - small) and e its
// rounding error (two_sum: exact; the next term is e^2 / 2 < 2^-106); a small side below
// `deep` (the scalar definition's own threshold) goes to the scalar definition (its log
// from the unrounded pieces).
//
// One vForce log for the chunk: 4u of the result on either side (ck::kVfLog), the sum 1u:
// 5 units. The scalar definitions call libm's log or log1p after the incomplete function,
// behind a branch on which side is the small one; over an array that is a coin toss, and
// each misprediction throws away the next element's incomplete function already under way
// (Poisson: 77 ns an element against the incomplete gamma's 50). Here the side is selected
// by index.
template <class Sides, class Precise>
void log_tail_kernel(int n, double* out, bool upper, double deep, Sides&& sides, Precise&& precise) {
    double arg[kChunkSize], corr[kChunkSize];
    bool fast[kChunkSize];
    for (int i = 0; i < n; ++i) {
        double lower = 0.5, higher = 0.5;
        const bool ok = sides(i, lower, higher);
        const int lower_small = lower <= higher;
        const double pair[2] = {higher, lower};
        const double small = pair[lower_small];
        const int large = static_cast<int>(upper) == lower_small;  // the large side is asked for
        const num::DD w = num::dd::two_sum(1.0, -small);
        const double args[2] = {small, w.hi}, corrs[2] = {0.0, w.lo / w.hi};
        arg[i] = ok ? args[large] : 1.0;
        corr[i] = corrs[large];
        fast[i] = ok && static_cast<bool>(large | static_cast<int>(small >= deep));
    }
    ck::log(arg, arg, n);
    for (int i = 0; i < n; ++i) out[i] = arg[i] + corr[i];
    ck::fallback(n, fast, out, precise);
}

// A sum compensated as Neumaier's (thousands of terms, as for the entropy of a spread pmf,
// would otherwise lose ~sqrt(n) ULP).
struct KahanSum {
    double s = 0.0, c = 0.0;
    void add(double x) {
        const double t = s + x;
        c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    double value() const { return s + c; }
};

// The sums of an entropy walk from a start, in units of pmf(start), over the terms beside
// the start itself (t_k = pmf(k) / pmf(start)):
//     h    = sum t_k |log pmf(k)|   (with the start's own term: the entropy over pmf(start)),
//     rest = sum t_k                (1 + rest = 1 / pmf(start) once the walk has the mass),
//     w    = sum t_k (-log t_k).
struct EntropySums {
    KahanSum h, rest, w;
};

// One run of an entropy walk: the terms at k = from + dir, from + 2 dir, ..., `end`
// (dir = +-1), added to the sums. t_k by the ratio recurrence in double-double (step(k, dir)
// = pmf(k + dir) / pmf(k); every step to ~2^-104, so 2^25 of them lose nothing a float64
// sees), log t_k one float64 log, |log pmf(k)| = |l + log t_k| with l = log pmf(from): ~10 ns
// a term against the ~100 of a double-double log pmf.
//
// Unless `full`, the run stops once the rest is negligible: past a term u of h beyond `mode`
// whose ratio to the one before is r < 1, the rest is about u r / (1 - r) (the ratios keep
// falling past the peak), and the run stops when that is below 2^-60 of the sum. A stop at u
// below 2^-60 of the sum alone left 5e-13 of a negative binomial with sd 2.3e5 (8.6 sd out,
// the rest ~3e4 terms of that size). A full run (a pmf whose ratios do not keep falling)
// goes to `end`. Either stops at a t below 1e-290: the rest of a falling run is below that.
//
// `left`: the terms the bound still allows, counted down; false when the run would pass it
// (or a ratio is NaN), and the sum is then not formed.
template <class Step>
bool entropy_run(
    EntropySums& sums, int& left, double from, double dir, double end, num::DD l, double mode, bool full, Step&& step
) {
    const double lf = l.hi + l.lo;
    num::DD t{1.0, 0.0};
    double prev = std::fabs(lf);  // the term of h at `from`
    for (double k = from; dir > 0.0 ? k < end : k > end;) {
        if (--left < 0) return false;
        t = t * step(k, dir);
        k += dir;
        if (!(t.hi >= 1e-290)) return t.hi == t.hi;
        const double lt = std::log(t.hi) + t.lo / t.hi;  // log t_k
        const double u = std::fabs(t.hi * (lf + lt));
        sums.h.add(u);
        sums.rest.add(t.hi);
        sums.rest.c += t.lo;
        sums.w.add(-t.hi * lt);
        if (!full && (dir > 0.0 ? k > mode + 1.0 : k < mode - 1.0)) {
            const double r = prev > 0.0 ? u / prev : 0.0;
            if (r < 1.0 && u * r <= (1.0 - r) * std::fabs(sums.h.s) * 0x1p-60) return true;
        }
        prev = u;
    }
    return true;
}

// The entropy -sum pmf log pmf of a unimodal pmf, from the mode m outwards in both directions
// (entropy_run), the sums compensated. With p_k = t_k / (1 + rest) (the pmf sums to 1, and
// what the walk leaves out of that is below what it leaves out of the entropy),
//     H = log(1 + rest) + w / (1 + rest),
// both parts nonnegative (t_k <= 1 around the mode) and neither through log pmf(m), whose
// own rounding (an absolute error of an ULP of its size) would be the entropy's; log pmf(m)
// only decides where the walk stops. NaN when the sum would pass kMaxEntropyTerms terms, or
// the mode is past 2^52 (the walk needs its float64 neighbours).
template <class LogPmf, class Step>
double entropy_sum(double mode, double lo, double hi, LogPmf&& logpmf, Step&& step) {
    const double m = std::clamp(std::floor(mode), lo, hi);
    if (!(std::fabs(m) < kMaxWalkStart)) return kNaN;
    const num::DD l = logpmf(m);
    EntropySums sums;
    sums.h.add(std::fabs(l.hi + l.lo));
    int left = kMaxEntropyTerms;
    if (!entropy_run(sums, left, m, 1.0, hi, l, mode, false, step)) return kNaN;
    if (!entropy_run(sums, left, m, -1.0, lo, l, mode, false, step)) return kNaN;
    const double rest = sums.rest.value();
    return std::log1p(rest) + sums.w.value() / (1.0 + rest);
}

// The entropy of a pmf on [lo, hi] that falls from both ends to a trough at `trough` (lo: it
// rises all the way, hi: it falls all the way): the runs [lo, trough) from lo and [trough, hi]
// from hi, each in full (toward a trough the ratios rise to 1: the geometric bound of the
// early stop does not hold) and in units of the pmf at its own end, by which its sum h is
// scaled (the pmf as a double-double: its rounding would be common to every term). NaN for
// more than kMaxEntropyTerms points.
template <class LogPmf, class Step>
double entropy_sum_ends(double trough, double lo, double hi, LogPmf&& logpmf, Step&& step) {
    if (!(hi - lo < kMaxEntropyTerms && std::fabs(lo) < kMaxWalkStart && std::fabs(hi) < kMaxWalkStart)) return kNaN;
    auto scaled = [](num::DD l, double sum) {
        const num::DD p = num::exp_full(l);
        return p.hi * sum + p.lo * sum;
    };
    int left = kMaxEntropyTerms;
    double total = 0.0;
    if (trough > lo) {
        const num::DD l = logpmf(lo);
        EntropySums sums;
        sums.h.add(std::fabs(l.hi + l.lo));
        if (!entropy_run(sums, left, lo, 1.0, trough - 1.0, l, lo, true, step)) return kNaN;
        total += scaled(l, sums.h.value());
    }
    const num::DD l = logpmf(hi);
    EntropySums sums;
    sums.h.add(std::fabs(l.hi + l.lo));
    if (!entropy_run(sums, left, hi, -1.0, trough, l, hi, true, step)) return kNaN;
    return total + scaled(l, sums.h.value());
}

// A closed-form estimate of the quantile as the search's start (the search then settles
// it in two or three evaluations); anything not finite or past 2^1023 keeps the default.
template <class F>
F with_guess(F f, double g) {
    if (g >= f.lo && g < 0x1p1023) f.guess = std::min(std::floor(g), f.hi);
    return f;
}

}  // namespace sd::stats::disc

// The chunk kernels of a discrete distribution's value methods (inside the struct `Self`
// with NP parameters): the scalar definition behind its memo, and one whose misses go to a
// kernel `fname##_kernel(const Chunk<NP>&, double*)` (a fast path with its own fallback).
// floored: the method reads only floor(x) (the cdfs).
#define SD_DISC_MEMO_SCALAR(Self, NP, fname, floored)                                                  \
    static void fname##_chunk(const Chunk<NP>& c, double* out) {                                       \
        struct Tag {};                                                                                 \
        disc::memoized<Tag>(c, out, floored, [](const Chunk<NP>& s, double* o) {                       \
            disc::scalar_kernel<Self>(s, o, [](double x, auto... p) { return Self::fname(x, p...); }); \
        });                                                                                            \
    }
// The four inverses of Self from `searcher(which, t, memo, params...)` (the Fns to search,
// with its start): the scalar definition and the chunk kernel that gives the search the memo.
#define SD_DISC_INVERSE(Self, NP, fname)                                                                         \
    template <class... P>                                                                                        \
    static double fname(double t, P... p) {                                                                      \
        const double k = disc::inverse(disc::Inv::fname, t, Self::searcher(disc::Inv::fname, t, nullptr, p...)); \
        return disc::settle<Self>(disc::Inv::fname, t, k, p...);                                                 \
    }                                                                                                            \
    static void fname##_chunk(const Chunk<NP>& c, double* out) {                                                 \
        disc::InvMemo* m = disc::inv_memo_for<Self>(c);                                                          \
        disc::scalar_kernel<Self>(c, out, [m](double t, auto... p) {                                             \
            const double k = disc::inverse(disc::Inv::fname, t, Self::searcher(disc::Inv::fname, t, m, p...));   \
            return disc::settle<Self>(disc::Inv::fname, t, k, p...);                                             \
        });                                                                                                      \
    }
#define SD_DISC_INVERSES(Self, NP)     \
    SD_DISC_INVERSE(Self, NP, icdf)    \
    SD_DISC_INVERSE(Self, NP, iccdf)   \
    SD_DISC_INVERSE(Self, NP, ilogcdf) \
    SD_DISC_INVERSE(Self, NP, ilogccdf)
#define SD_DISC_MEMO_KERNEL(Self, NP, fname, floored)                                                            \
    static void fname##_chunk(const Chunk<NP>& c, double* out) {                                                 \
        struct Tag {};                                                                                           \
        disc::memoized<Tag>(c, out, floored, [](const Chunk<NP>& s, double* o) { Self::fname##_kernel(s, o); }); \
    }
