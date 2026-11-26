// What the cluster area's kernels share (docs/design/cluster.md §3.10, §6; decisions CL7, CL43).
//
// - The key of decision CL7: the points sorted by their coordinates (the first column first) and then by
//   their index; a point's rank in that order is its key. Every tie rule of the area compares keys, so the
//   same points in another order give the same partition. lex_order() gives order[r] = the input index of
//   rank r (a total order, so std::sort's result is one and the same).
// - The condensed index of a pair i < j of n points (scipy's and sd.spatial.pdist's layout).
// - The power of two of decision CL43: data whose largest |x| lies outside [2^-250, 2^250] are scaled into
//   [1, 2) before squared distances are taken (exact), the results scaled back.
// - The chunk sums of decision CL48 (pairwise_chunks): kChunk = 4096 points in key order make a chunk, each
//   chunk's sum runs point by point, and the chunk sums meet by the pairwise tree in chunk order. The tree is
//   fixed by the number of chunks alone, so a sum has the same bits for any thread count.
// - The random elements of decision CL41 (element): the t-th number of start (sample, reference set) r is
//   element r 2^40 + t of the call's Philox4x64-10 stream; uniform() and draw() read it.
#pragma once

#include "parallel/pool.hpp"
#include "random/philox.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::cluster {

using std::int64_t;

inline constexpr double kU = 0x1p-53;
inline constexpr int64_t kSplitWork = int64_t{1} << 15;  // the parallel threshold of API.md §3 (flops)

inline int64_t cidx(int64_t n, int64_t i, int64_t j) {  // i < j
    return n * i - i * (i + 1) / 2 + (j - i - 1);
}

inline int64_t cpair(int64_t n, int64_t i, int64_t j) { return i < j ? cidx(n, i, j) : cidx(n, j, i); }

// A double's place in the order of the values as an unsigned key (-0.0 taken as +0.0, so equal values have
// equal keys; no NaN reaches here).
inline std::uint64_t value_key(double v) {
    if (v == 0.0) v = 0.0;
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return (b >> 63) ? ~b : (b | (std::uint64_t{1} << 63));
}

// order[r]: the input index of the point of rank r (coordinates lexicographically, then the index). The first
// coordinate's keys are sorted by a stable radix sort of 16-bit digits (the index order kept among equal keys), and
// each run of equal first coordinates is then sorted by the other coordinates and the index: the same total order
// as comparing the rows, so the same result.
inline std::vector<int64_t> lex_order(const double* X, int64_t n, int64_t p) {
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    auto rest_less = [&](int64_t a, int64_t b) {
        const double* x = X + a * p;
        const double* y = X + b * p;
        for (int64_t d = 1; d < p; ++d)
            if (x[d] != y[d]) return x[d] < y[d];
        return a < b;
    };
    std::vector<std::uint64_t> key(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) key[static_cast<size_t>(i)] = value_key(X[i * p]);
    if (n < 2048) {
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            const std::uint64_t ka = key[static_cast<size_t>(a)], kb = key[static_cast<size_t>(b)];
            return ka != kb ? ka < kb : rest_less(a, b);
        });
        return order;
    }
    std::vector<int64_t> tmp(static_cast<size_t>(n));
    std::vector<int64_t> count(65536);
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 16 * pass;
        std::fill(count.begin(), count.end(), 0);
        for (int64_t i : order) ++count[(key[static_cast<size_t>(i)] >> shift) & 0xFFFF];
        int64_t sum = 0;
        for (auto& c : count) {
            const int64_t t = c;
            c = sum;
            sum += t;
        }
        for (int64_t i : order) tmp[static_cast<size_t>(count[(key[static_cast<size_t>(i)] >> shift) & 0xFFFF]++)] = i;
        order.swap(tmp);
    }
    if (p > 1) {
        for (int64_t a = 0; a < n;) {
            int64_t b = a + 1;
            const std::uint64_t ka = key[static_cast<size_t>(order[static_cast<size_t>(a)])];
            while (b < n && key[static_cast<size_t>(order[static_cast<size_t>(b)])] == ka) ++b;
            if (b - a > 1) std::sort(order.begin() + a, order.begin() + b, rest_less);
            a = b;
        }
    }
    return order;
}

// The exponent e of CL43: 0 when max |x| is in [2^-250, 2^250] (or 0), else the e with max |x| 2^e in [1, 2).
inline int scale_exponent(double maxabs) {
    if (maxabs == 0.0 || (maxabs >= 0x1p-250 && maxabs <= 0x1p250)) return 0;
    int e;
    (void)std::frexp(maxabs, &e);  // maxabs = f 2^e, f in [1/2, 1)
    return 1 - e;
}

// A union-find over 0 .. n - 1 whose root is the smallest member (so a cluster's root is its key).
struct MinUnionFind {
    std::vector<int64_t> parent;
    explicit MinUnionFind(int64_t n) : parent(static_cast<size_t>(n)) {
        std::iota(parent.begin(), parent.end(), int64_t{0});
    }
    int64_t find(int64_t x) {
        while (parent[static_cast<size_t>(x)] != x) {
            const int64_t g = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            parent[static_cast<size_t>(x)] = g;
            x = g;
        }
        return x;
    }
    int64_t unite(int64_t a, int64_t b) {  // the roots of two different sets; returns the new root
        if (a > b) std::swap(a, b);
        parent[static_cast<size_t>(b)] = a;
        return a;
    }
};

// ---- chunk sums (CL48) ------------------------------------------------------------------------------------

inline constexpr int64_t kChunk = 4096;  // points of a chunk (CL48)

inline int64_t chunks_of(int64_t n) { return (n + kChunk - 1) / kChunk; }

// The pairwise tree of decision CL48 over `nleaves` leaves of `width` doubles: level by level, leaf (or node) c
// takes c + 2^s when c + 2^s < nleaves (c a multiple of 2^(s + 1)), so node 0 ends as the total. A binary counter
// gives the same tree leaf by leaf: a new value meets the stored node of each level it carries through (the stored
// one on the left), and the stack's remainder is added from the lowest level up, each stored node on the left.
class PairwiseStack {
public:
    explicit PairwiseStack(int64_t width) : width_(width) {}
    // the next leaf (or a node standing for 2^level leaves, from a block of that many: see pairwise_chunks)
    void push(const double* v, int level = 0) {
        std::vector<double> carry(v, v + width_);
        for (int s = level;; ++s) {
            if (static_cast<size_t>(s) >= slots_.size()) slots_.resize(static_cast<size_t>(s) + 1);
            Slot& slot = slots_[static_cast<size_t>(s)];
            if (!slot.full) {
                slot.v.swap(carry);
                slot.full = true;
                return;
            }
            for (int64_t q = 0; q < width_; ++q)
                carry[static_cast<size_t>(q)] = slot.v[static_cast<size_t>(q)] + carry[static_cast<size_t>(q)];
            slot.full = false;
        }
    }
    // the total of the leaves pushed (zeros when none)
    void finish(double* out) {
        bool have = false;
        std::vector<double> acc(static_cast<size_t>(width_), 0.0);
        for (Slot& slot : slots_) {
            if (!slot.full) continue;
            if (!have) {
                acc = slot.v;
                have = true;
            } else {
                for (int64_t q = 0; q < width_; ++q)
                    acc[static_cast<size_t>(q)] = slot.v[static_cast<size_t>(q)] + acc[static_cast<size_t>(q)];
            }
            slot.full = false;
        }
        std::copy(acc.begin(), acc.end(), out);
    }

private:
    struct Slot {
        std::vector<double> v;
        bool full = false;
    };
    int64_t width_;
    std::vector<Slot> slots_;
};

// total[0 .. width) = the pairwise tree (CL48) of the leaves leaf(c, out) for c in [0, nleaves) (leaf writes all
// of out[0 .. width)). Tasks take aligned blocks of 2^b leaves (b from width and nleaves only), reduce them with
// a PairwiseStack and the blocks' nodes meet in a stack of their own at level b: the same tree, so the same bits
// for any b and any thread count. leaf_ns: the rough time of one leaf (parallel_for's wake-up hint).
template <class Leaf>
void pairwise_chunks(int64_t nleaves, int64_t width, Leaf&& leaf, double* total, int64_t leaf_ns = 0) {
    if (nleaves <= 0) {
        std::fill(total, total + width, 0.0);
        return;
    }
    int b = 0;  // at most 2^22 doubles of block nodes, and blocks of at least 8 leaves' work when it is tiny
    while ((((nleaves >> b) + 1) * width > (int64_t{1} << 22)) && (int64_t{1} << b) < nleaves) ++b;
    const int64_t g = int64_t{1} << b;
    const int64_t nt = (nleaves + g - 1) / g;
    std::vector<double> nodes(static_cast<size_t>(nt * width));
    auto block = [&](int64_t t) {
        PairwiseStack st(width);
        std::vector<double> buf(static_cast<size_t>(width));
        for (int64_t c = t * g; c < std::min(nleaves, (t + 1) * g); ++c) {
            leaf(c, buf.data());
            st.push(buf.data());
        }
        st.finish(nodes.data() + t * width);
    };
    if (nt > 1) parallel::parallel_for(nt, block, leaf_ns * g);
    else block(0);
    PairwiseStack top(width);
    for (int64_t t = 0; t < nt; ++t) top.push(nodes.data() + t * width, b);
    top.finish(total);
}

// ---- random elements (CL41) ---------------------------------------------------------------------------------

inline constexpr int kStartBits = 40;

inline std::uint64_t element(std::uint64_t r, std::uint64_t t) { return (r << kStartBits) | t; }

inline random::Key key_of(int64_t k0, int64_t k1) {
    return {static_cast<std::uint64_t>(k0), static_cast<std::uint64_t>(k1)};
}

// a uniform number in [0, 1) from element e (the top 53 bits of its first word)
inline double uniform(random::Stream& st, std::uint64_t e) { return random::to_unit(st.first(e)); }

// an index in [0, m) from element e, every index equally likely (Lemire's method; rejections read e's extra words)
inline int64_t draw(random::Stream& st, std::uint64_t e, int64_t m) {
    return static_cast<int64_t>(random::bounded(st.first(e), static_cast<std::uint64_t>(m), [&](std::uint64_t q) {
        return st.extra(e, q);
    }));
}

// The cumulative sums of v[0 .. n) as decision CL48 orders a running sum: within each chunk point by point, the
// chunks' totals one after another (P(i) = O_c + s_c(i), O_c the running sum of the earlier chunks' totals), so the
// values never decrease and P(n - 1) is the total. Used to draw indices with probability v_i / total.
inline void running_sums(const double* v, int64_t n, double* P) {
    const int64_t nc = chunks_of(n);
    std::vector<double> tot(static_cast<size_t>(nc));
    auto local = [&](int64_t c) {
        double s = 0.0;
        for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) P[i] = s = s + v[i];
        tot[static_cast<size_t>(c)] = s;
    };
    if (nc > 1) parallel::parallel_for(nc, local, kChunk);
    else local(0);
    std::vector<double> off(static_cast<size_t>(nc), 0.0);
    for (int64_t c = 1; c < nc; ++c)
        off[static_cast<size_t>(c)] = off[static_cast<size_t>(c - 1)] + tot[static_cast<size_t>(c - 1)];
    auto shift = [&](int64_t c) {
        const double o = off[static_cast<size_t>(c)];
        for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) P[i] = o + P[i];
    };
    if (nc > 1) parallel::parallel_for(nc - 1, [&](int64_t c) { shift(c + 1); }, kChunk);
}

// The index a uniform u picks from running sums P (probability v_i / P(n - 1)): the first i with P(i) > u P(n - 1);
// when rounding leaves none, the last i whose v_i is positive (a zero v_i is never picked otherwise).
inline int64_t pick(const double* P, int64_t n, double u) {
    const double target = u * P[n - 1];
    const int64_t i = std::upper_bound(P, P + n, target) - P;
    if (i < n) return i;
    for (int64_t j = n - 1; j > 0; --j)
        if (P[j] > P[j - 1]) return j;
    return 0;
}

}  // namespace sd::cluster
