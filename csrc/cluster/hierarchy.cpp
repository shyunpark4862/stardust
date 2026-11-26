// Hierarchical clustering (docs/design/cluster.md §2.2, §3.2, §3.10, §3.11, §4.1; decisions CL15-CL22, CL43,
// CL45, CL46, CL50-CL57). Points are numbered by their key (decision CL7: the rank of the lexicographic order;
// the caller passes the rows in that order, and for precomputed distances the key is the input index), and
// every algorithm works in that numbering: a cluster lives in the slot of its smallest key, so "the smaller
// slot" is "the smaller key" in every tie rule. Z is mapped back to the input's numbers at the end.
//
// _core.cluster:
//   lex_order(X) -> (n,) int64             order[r] = the input index of rank r (common.hpp)
//   linkage(X, D, order, method, path) -> (Z, flags)
//       X: the rank-ordered observations (n, p) or None; D: the rank-ordered condensed distances or None;
//       method 0 single, 1 complete, 2 average, 3 weighted, 4 centroid, 5 median, 6 ward. The tree:
//         single    the minimum spanning tree (mst.hpp: Prim or Boruvka on a KD tree by shape (CL54), or Prim on D),
//                   merged in (distance, key) order; a group of equal distances is resolved by the primitive
//                   definition: within each set of clusters its edges join, the cluster of smallest key
//                   absorbs its neighbours at exactly that distance in key order (CL17, CL50).
//         complete, average, weighted (on D), ward (on X or D)
//                   the nearest-neighbour chain, the neighbour of smallest (value, key); a scan that meets
//                   another candidate exactly equal to its minimum (the chain's previous cluster too) flags
//                   the input, which the generic algorithm then solves (CL46; flags bit 0). On D the chain
//                   works on a square copy when 8 n^2 bytes are at most a quarter of the memory, else on a
//                   condensed one (the same scans and updates: the same tree; CL96).
//         centroid, median (and the flagged inputs)
//                   Muellner's generic algorithm with an indexed heap of (value, key) (CL17).
//       On D the values are the Lance-Williams updates (ward, centroid and median on squared distances); on X
//       (ward, centroid, median: the matrix-free path, CL18) the clusters' centres and sizes. path 0 picks by
//       shape; 1 forces the matrix built from X (ward, centroid, median: squared sums), 3 the generic
//       algorithm, 4 Prim and 5 Boruvka (single on X), 6 the condensed copy of the chain -- for the tests'
//       two-path comparisons.
//       The heights (CL45, CL55): single and complete the chosen distances; the others the Lance-Williams recurrence
//       on the final tree (recurrence() below), so every path gives the same bits for the same tree. Rows in
//       the primitive order on the final heights: of the merges whose children exist, the smallest (height,
//       key pair) first; the monotone linkages' heights carried up the tree as running maxima (CL51).
//       ward, centroid and median on X take CL43's power of two.
//   diana(D, order) -> Z                    R's DIANA (divisive) on rank-ordered condensed distances (CL52)
//   cophenet(Z) -> condensed (n (n - 1) / 2,)
//   optimal_leaf_ordering(Z, D, key) -> Z   Bar-Joseph et al. (2001), ties by key (CL53)
//   _mst(X, method) -> (edges (n - 1, 2), weights)  the spanning trees for tools/cluster_probe.py proto.mst
#include "array/ndarray.hpp"
#include "base/buffer.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "cluster/cluster.hpp"
#include "cluster/common.hpp"
#include "cluster/mst.hpp"
#include "parallel/pool.hpp"
#include "spatial/kdtree.hpp"
#include "spatial/metric.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace sd::cluster {

namespace nb = nanobind;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
enum Method { kSingle = 0, kComplete, kAverage, kWeighted, kCentroid, kMedian, kWard, kDiana };

bool squared_method(int m) { return m == kCentroid || m == kMedian || m == kWard; }
bool monotone_method(int m) { return m != kCentroid && m != kMedian; }

// ---- inputs ---------------------------------------------------------------------------------------------------

const double* rows_of(nb::handle o, int64_t& n, int64_t& p) {
    NDArray& a = require_ndarray(o, "X");
    if (a.dtype != DType::Float64 || !a.c_contiguous || a.ndim != 2) throw nb::value_error("X: C (n, p) float64");
    n = a.shape[0];
    p = a.shape[1];
    return reinterpret_cast<const double*>(a.data);
}

const double* vector_of(nb::handle o, const char* what, int64_t& m) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Float64 || !a.c_contiguous || a.ndim != 1) throw nb::value_error("needs a C 1-D float64");
    m = a.shape[0];
    return reinterpret_cast<const double*>(a.data);
}

const int64_t* ints_of(nb::handle o, const char* what, int64_t& m) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Int64 || !a.c_contiguous || a.ndim != 1) throw nb::value_error("needs a C 1-D int64");
    m = a.shape[0];
    return reinterpret_cast<const int64_t*>(a.data);
}

int64_t condensed_n(int64_t m) {
    int64_t n = static_cast<int64_t>((1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(m))) / 2.0);
    while (n * (n - 1) / 2 > m) --n;
    while ((n + 1) * n / 2 <= m) ++n;
    if (n * (n - 1) / 2 != m) throw nb::value_error("D: not a condensed length");
    return n;
}

// ---- the tree an algorithm builds ------------------------------------------------------------------------------

// Merges in the order an algorithm made them (children before parents): node ids 0 .. n - 1 are the points
// (keys), n + t the t-th merge; v the algorithm's own value of the merge (single, complete: the height).
struct Merges {
    int64_t n = 0;
    std::vector<int64_t> a, b;
    std::vector<double> v;
    explicit Merges(int64_t n_) : n(n_) {
        a.reserve(static_cast<size_t>(std::max<int64_t>(n_ - 1, 0)));
        b.reserve(a.capacity());
        v.reserve(a.capacity());
    }
    int64_t add(int64_t x, int64_t y, double val) {
        a.push_back(x);
        b.push_back(y);
        v.push_back(val);
        return n + static_cast<int64_t>(a.size()) - 1;
    }
    int64_t count() const { return static_cast<int64_t>(a.size()); }
};

// The Lance-Williams update of the matrix path: the value of (A u B, K) from (A, K), (B, K) and (A, B).
constexpr double lance_williams(int m, double ak, double bk, double ab, double na, double nb, double nk) {
    switch (m) {
        case kSingle: return std::min(ak, bk);
        case kComplete: return std::max(ak, bk);
        case kAverage: return (na * ak + nb * bk) / (na + nb);
        case kWeighted: return 0.5 * (ak + bk);
        case kCentroid: {
            const double t = na + nb;
            return (na * ak + nb * bk) / t - (na * nb) * ab / (t * t);
        }
        case kMedian: return 0.5 * ak + 0.5 * bk - 0.25 * ab;
        default: {
            const double t = 1.0 / (na + nb + nk);
            return (na + nk) * t * ak + (nb + nk) * t * bk - nk * t * ab;
        }
    }
}

// ---- the nearest-neighbour chain and the generic algorithm on a matrix --------------------------------------

// The Lance-Williams update of the matrix after merging b into a (a < b): D(a, k) for every live k. The three
// ranges of k (below a, between a and b, above b) are walked with their own index steps.
template <int Mth>
void update_of(
    double* D,
    int64_t n,
    int64_t a,
    int64_t b,
    double ab,
    const std::vector<int64_t>& size,
    const std::vector<char>& alive
) {
    const double na = static_cast<double>(size[static_cast<size_t>(a)]);
    const double nb = static_cast<double>(size[static_cast<size_t>(b)]);
    auto lw = [&](double ak, double bk, int64_t k) {
        return lance_williams(Mth, ak, bk, ab, na, nb, static_cast<double>(size[static_cast<size_t>(k)]));
    };
    int64_t ia = a - 1, ib = b - 1;  // (k, a), (k, b) for k < a
    for (int64_t k = 0; k < a; ia += n - k - 2, ib += n - k - 2, ++k)
        if (alive[static_cast<size_t>(k)]) D[ia] = lw(D[ia], D[ib], k);
    double* ra = D + cidx(n, a, a + 1) - (a + 1);  // ra[k] = (a, k), k > a
    int64_t jb = cidx(n, a + 1, b);                 // (k, b) for a < k < b
    for (int64_t k = a + 1; k < b; jb += n - k - 2, ++k)
        if (alive[static_cast<size_t>(k)]) ra[k] = lw(ra[k], D[jb], k);
    if (b + 1 < n) {
        const double* rb = D + cidx(n, b, b + 1) - (b + 1);
        for (int64_t k = b + 1; k < n; ++k)
            if (alive[static_cast<size_t>(k)]) ra[k] = lw(ra[k], rb[k], k);
    }
}

void update(
    std::vector<double>& D,
    int64_t n,
    int method,
    int64_t a,
    int64_t b,
    double ab,
    const std::vector<int64_t>& size,
    const std::vector<char>& alive
) {
    switch (method) {
        case kSingle: update_of<kSingle>(D.data(), n, a, b, ab, size, alive); break;
        case kComplete: update_of<kComplete>(D.data(), n, a, b, ab, size, alive); break;
        case kAverage: update_of<kAverage>(D.data(), n, a, b, ab, size, alive); break;
        case kWeighted: update_of<kWeighted>(D.data(), n, a, b, ab, size, alive); break;
        case kCentroid: update_of<kCentroid>(D.data(), n, a, b, ab, size, alive); break;
        case kMedian: update_of<kMedian>(D.data(), n, a, b, ab, size, alive); break;
        default: update_of<kWard>(D.data(), n, a, b, ab, size, alive);
    }
}

// The Lance-Williams update of the NN-chain after merging b into a (a < b), over the live slots only (live: sorted,
// b still in it): D(a, k) for every live k other than a and b, each from the same three values as update_of.
template <int Mth>
void update_live(
    double* D,
    int64_t n,
    int64_t a,
    int64_t b,
    double ab,
    const std::vector<int64_t>& size,
    const std::vector<int32_t>& live
) {
    const double na = static_cast<double>(size[static_cast<size_t>(a)]);
    const double nb = static_cast<double>(size[static_cast<size_t>(b)]);
    for (const int32_t k32 : live) {
        const int64_t k = k32;
        if (k == a || k == b) continue;
        const int64_t ia = cpair(n, a, k), ib = cpair(n, b, k);
        D[ia] = lance_williams(Mth, D[ia], D[ib], ab, na, nb, static_cast<double>(size[static_cast<size_t>(k)]));
    }
}

void update_live(
    std::vector<double>& D,
    int64_t n,
    int method,
    int64_t a,
    int64_t b,
    double ab,
    const std::vector<int64_t>& size,
    const std::vector<int32_t>& live
) {
    switch (method) {
        case kComplete: update_live<kComplete>(D.data(), n, a, b, ab, size, live); break;
        case kAverage: update_live<kAverage>(D.data(), n, a, b, ab, size, live); break;
        case kWeighted: update_live<kWeighted>(D.data(), n, a, b, ab, size, live); break;
        default: update_live<kWard>(D.data(), n, a, b, ab, size, live);
    }
}

// D: the working copy (condensed, keys), overwritten. Returns false when a scan met an exact tie (CL46). The live
// slots are a sorted list, so a scan and an update touch the live clusters alone (no test of a dead slot, whose
// outcome a branch could not learn); a scan offers the live slots in increasing order: the smallest value, its first
// (smallest) slot and the number of slots at it, as the scan from the chain's previous cluster finds them.
bool nnchain_matrix(std::vector<double>& D, int64_t n, int method, Merges& M) {
    const double* Dp = D.data();
    std::vector<int64_t> size(static_cast<size_t>(n), 1), node(static_cast<size_t>(n));
    std::iota(node.begin(), node.end(), int64_t{0});
    std::vector<int32_t> live(static_cast<size_t>(n));
    std::iota(live.begin(), live.end(), 0);
    std::vector<int64_t> chain;
    chain.reserve(static_cast<size_t>(n));
    for (int64_t step = 0; step < n - 1; ++step) {
        if (chain.empty()) chain.push_back(live[0]);
        int64_t x, y;
        double best;
        while (true) {
            x = chain.back();
            const int64_t prev = chain.size() >= 2 ? chain[chain.size() - 2] : -1;
            best = kInf;
            y = -1;
            int64_t count = 0;
            const size_t mid = static_cast<size_t>(std::lower_bound(live.begin(), live.end(), x) - live.begin());
            // live slots below x: column x above the diagonal, (j, x) at n j - j (j + 1) / 2 + x - j - 1
            for (size_t q = 0; q < mid; ++q) {
                const int64_t j = live[q];
                const double v = Dp[n * j - j * (j + 1) / 2 + x - j - 1];
                if (v <= best || y < 0) {  // rare: the branch is learnt
                    if (v < best || y < 0) best = v, y = j, count = 1;
                    else if (v == best) ++count;
                }
            }
            // live slots above x: row x (contiguous)
            const double* row = Dp + (x + 1 < n ? cidx(n, x, x + 1) - (x + 1) : 0);
            for (size_t q = mid + 1; q < live.size(); ++q) {
                const int64_t j = live[q];
                const double v = row[j];
                if (v <= best || y < 0) {  // rare: the branch is learnt
                    if (v < best || y < 0) best = v, y = j, count = 1;
                    else if (v == best) ++count;
                }
            }
            if (count > 1) return false;
            if (y == prev) break;
            chain.push_back(y);
        }
        chain.pop_back();
        chain.pop_back();
        const int64_t a = std::min(x, y), b = std::max(x, y);
        const auto aa = static_cast<size_t>(a), bb = static_cast<size_t>(b);
        node[aa] = M.add(node[aa], node[bb], best);
        update_live(D, n, method, a, b, best, size, live);
        live.erase(std::lower_bound(live.begin(), live.end(), static_cast<int32_t>(b)));
        size[aa] += size[bb];
    }
    return true;
}

// The square working copy of the NN-chain (S[i n + j] = S[j n + i] = the value of i and j; f squares D for ward on
// given distances): a scan reads row x contiguously and an update writes row a and column a, where the condensed copy
// reads column x with a stride. Tiles of 64 x 64 keep a tile's transposed writes in L1; the row blocks are tasks
// (each writes its own entries).
std::vector<double> square_of(const double* D, int64_t n, bool sq) {
    std::vector<double> S(static_cast<size_t>(n * n));
    constexpr int64_t B = 64;
    const int64_t nb = (n + B - 1) / B;
    auto body = [&](int64_t bi) {
        const int64_t i0 = bi * B, i1 = std::min(n, i0 + B);
        for (int64_t i = i0; i < i1; ++i) S[static_cast<size_t>(i * n + i)] = 0.0;
        for (int64_t bj = bi; bj < nb; ++bj) {
            const int64_t j0 = bj * B, j1 = std::min(n, j0 + B);
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t js = std::max(j0, i + 1);
                if (js >= j1) continue;
                const double* d = D + cidx(n, i, js);
                for (int64_t j = js; j < j1; ++j) {
                    double v = d[j - js];
                    if (sq) v = v * v;
                    S[static_cast<size_t>(i * n + j)] = v;
                    S[static_cast<size_t>(j * n + i)] = v;
                }
            }
        }
    };
    if (nb > 1 && n * n >= kSplitWork) parallel::parallel_for(nb, body, B * n);
    else
        for (int64_t b = 0; b < nb; ++b) body(b);
    return S;
}

template <int Mth>
void update_square(
    double* S,
    int64_t n,
    int64_t a,
    int64_t b,
    double ab,
    const std::vector<int64_t>& size,
    const std::vector<int32_t>& live
) {
    const double na = static_cast<double>(size[static_cast<size_t>(a)]);
    const double nb = static_cast<double>(size[static_cast<size_t>(b)]);
    double* ra = S + a * n;
    const double* rb = S + b * n;
    for (const int32_t k32 : live) {
        const int64_t k = k32;
        if (k == a || k == b) continue;
        const double v =
            lance_williams(Mth, ra[k], rb[k], ab, na, nb, static_cast<double>(size[static_cast<size_t>(k)]));
        ra[k] = v;
        S[k * n + a] = v;
    }
}

// nnchain_matrix on the square copy: the same scans (live slots in increasing order) and the same updates (each value
// from the same three values), so the same tree, merge values and ties.
bool nnchain_square(std::vector<double>& S, int64_t n, int method, Merges& M) {
    const double* Sp = S.data();
    std::vector<int64_t> size(static_cast<size_t>(n), 1), node(static_cast<size_t>(n));
    std::iota(node.begin(), node.end(), int64_t{0});
    std::vector<int32_t> live(static_cast<size_t>(n));
    std::iota(live.begin(), live.end(), 0);
    std::vector<int64_t> chain;
    chain.reserve(static_cast<size_t>(n));
    for (int64_t step = 0; step < n - 1; ++step) {
        if (chain.empty()) chain.push_back(live[0]);
        int64_t x, y;
        double best;
        while (true) {
            x = chain.back();
            const int64_t prev = chain.size() >= 2 ? chain[chain.size() - 2] : -1;
            best = kInf;
            y = -1;
            int64_t count = 0;
            const double* row = Sp + x * n;
            for (const int32_t j32 : live) {
                const int64_t j = j32;
                const double v = row[j];
                if (j == x) continue;
                if (v <= best || y < 0) {  // rare: the branch is learnt
                    if (v < best || y < 0) best = v, y = j, count = 1;
                    else if (v == best) ++count;
                }
            }
            if (count > 1) return false;
            if (y == prev) break;
            chain.push_back(y);
        }
        chain.pop_back();
        chain.pop_back();
        const int64_t a = std::min(x, y), b = std::max(x, y);
        const auto aa = static_cast<size_t>(a), bb = static_cast<size_t>(b);
        node[aa] = M.add(node[aa], node[bb], best);
        switch (method) {
            case kComplete: update_square<kComplete>(S.data(), n, a, b, best, size, live); break;
            case kAverage: update_square<kAverage>(S.data(), n, a, b, best, size, live); break;
            case kWeighted: update_square<kWeighted>(S.data(), n, a, b, best, size, live); break;
            default: update_square<kWard>(S.data(), n, a, b, best, size, live);
        }
        live.erase(std::lower_bound(live.begin(), live.end(), static_cast<int32_t>(b)));
        size[aa] += size[bb];
    }
    return true;
}

// A min-heap of slots by (value[slot], slot), with positions so that a slot's value can change.
class SlotHeap {
public:
    SlotHeap(int64_t n, const std::vector<double>& value) : val_(value), pos_(static_cast<size_t>(n), -1) {}
    bool empty() const { return heap_.empty(); }
    int64_t top() const { return heap_.front(); }
    void push(int64_t s) {
        pos_[static_cast<size_t>(s)] = static_cast<int64_t>(heap_.size());
        heap_.push_back(s);
        up(static_cast<int64_t>(heap_.size()) - 1);
    }
    bool contains(int64_t s) const { return pos_[static_cast<size_t>(s)] >= 0; }
    void update(int64_t s) {  // the slot's value changed either way
        const int64_t i = pos_[static_cast<size_t>(s)];
        up(i);
        down(pos_[static_cast<size_t>(s)]);
    }
    void remove(int64_t s) {
        const int64_t i = pos_[static_cast<size_t>(s)];
        if (i < 0) return;
        const int64_t last = static_cast<int64_t>(heap_.size()) - 1;
        swap(i, last);
        heap_.pop_back();
        pos_[static_cast<size_t>(s)] = -1;
        if (i < last) {
            up(i);
            down(pos_[static_cast<size_t>(heap_[static_cast<size_t>(i)])]);
        }
    }

private:
    const std::vector<double>& val_;
    std::vector<int64_t> heap_, pos_;
    bool less(int64_t x, int64_t y) const {
        const double a = val_[static_cast<size_t>(x)], b = val_[static_cast<size_t>(y)];
        return a < b || (a == b && x < y);
    }
    void swap(int64_t i, int64_t j) {
        std::swap(heap_[static_cast<size_t>(i)], heap_[static_cast<size_t>(j)]);
        pos_[static_cast<size_t>(heap_[static_cast<size_t>(i)])] = i;
        pos_[static_cast<size_t>(heap_[static_cast<size_t>(j)])] = j;
    }
    void up(int64_t i) {
        while (i > 0) {
            const int64_t p = (i - 1) / 2;
            if (!less(heap_[static_cast<size_t>(i)], heap_[static_cast<size_t>(p)])) break;
            swap(i, p);
            i = p;
        }
    }
    void down(int64_t i) {
        const int64_t m = static_cast<int64_t>(heap_.size());
        while (true) {
            const int64_t l = 2 * i + 1, r = l + 1;
            int64_t s = i;
            if (l < m && less(heap_[static_cast<size_t>(l)], heap_[static_cast<size_t>(s)])) s = l;
            if (r < m && less(heap_[static_cast<size_t>(r)], heap_[static_cast<size_t>(s)])) s = r;
            if (s == i) break;
            swap(i, s);
            i = s;
        }
    }
};

// Muellner's generic algorithm (2011, Fig. 3) with the pair order (value, smaller key, larger key). The ops:
//   dist(i, j)       the current value of the clusters in slots i and j
//   rescan(x)        (value, slot) smallest over the live slots after x, slot -1 when there is none
//   before(a, f)     f(x, dist(x, a)) for every live slot x before a
//   join(a, b, v)    merges slot b into slot a (a < b) at value v; afterwards a's values are new, b is gone
// Invariant: (mindist[x], nn[x]) is at most (dist(x, j), j) for every live j > x, so a slot at the heap's top
// whose recorded neighbour is alive at exactly mindist holds the smallest pair.
template <class Ops>
void generic(int64_t n, Ops& ops) {
    std::vector<double> mindist(static_cast<size_t>(n), kInf);
    std::vector<int64_t> nn(static_cast<size_t>(n), -1);
    std::vector<char> alive(static_cast<size_t>(n), 1);
    auto rescan = [&](int64_t x) {
        const auto [v, j] = ops.rescan(x);
        mindist[static_cast<size_t>(x)] = v;
        nn[static_cast<size_t>(x)] = j;
    };
    SlotHeap heap(n, mindist);
    for (int64_t i = 0; i + 1 < n; ++i) {
        rescan(i);
        heap.push(i);
    }
    for (int64_t step = 0; step < n - 1; ++step) {
        int64_t a, b;
        while (true) {
            a = heap.top();
            b = nn[static_cast<size_t>(a)];
            if (b >= 0 && alive[static_cast<size_t>(b)] && ops.dist(a, b) == mindist[static_cast<size_t>(a)]) break;
            rescan(a);
            if (nn[static_cast<size_t>(a)] < 0) heap.remove(a);
            else heap.update(a);
        }
        const double value = mindist[static_cast<size_t>(a)];
        ops.join(a, b, value);
        alive[static_cast<size_t>(b)] = 0;
        heap.remove(b);
        // the slots before a: a is one of their candidates; b is gone
        ops.before(a, [&](int64_t x, double v) {
            const auto xx = static_cast<size_t>(x);
            if (nn[xx] == b) nn[xx] = a;  // still a lower bound: every candidate equal to mindist lies past b
            if (v < mindist[xx] || (v == mindist[xx] && a < nn[xx])) {
                mindist[xx] = v;
                nn[xx] = a;
                if (heap.contains(x)) heap.update(x);
                else heap.push(x);
            }
        });
        // a's own candidates all changed
        rescan(a);
        if (nn[static_cast<size_t>(a)] < 0) heap.remove(a);
        else if (heap.contains(a)) heap.update(a);
        else heap.push(a);
    }
}

// The matrix's ops for the generic algorithm (and its slots' sizes and nodes).
struct MatrixOps {
    std::vector<double>& D;
    int64_t n;
    int method;
    Merges& M;
    std::vector<int64_t> size, node;
    std::vector<char> alive;
    MatrixOps(std::vector<double>& D_, int64_t n_, int m, Merges& M_)
        : D(D_),
          n(n_),
          method(m),
          M(M_),
          size(static_cast<size_t>(n_), 1),
          node(static_cast<size_t>(n_)),
          alive(static_cast<size_t>(n_), 1) {
        std::iota(node.begin(), node.end(), int64_t{0});
    }
    double dist(int64_t i, int64_t j) const { return D[static_cast<size_t>(cpair(n, i, j))]; }
    std::pair<double, int64_t> rescan(int64_t x) const {
        double best = kInf;
        int64_t arg = -1;
        if (x + 1 >= n) return {best, arg};
        const double* row = D.data() + cidx(n, x, x + 1) - (x + 1);
        for (int64_t j = x + 1; j < n; ++j)
            if (alive[static_cast<size_t>(j)] && (arg < 0 || row[j] < best)) best = row[j], arg = j;
        return {best, arg};
    }
    template <class F>
    void before(int64_t a, F f) const {
        int64_t idx = a - 1;
        for (int64_t x = 0; x < a; idx += n - x - 2, ++x)
            if (alive[static_cast<size_t>(x)]) f(x, D[static_cast<size_t>(idx)]);
    }
    void join(int64_t a, int64_t b, double value) {
        const auto aa = static_cast<size_t>(a), bb = static_cast<size_t>(b);
        node[aa] = M.add(node[aa], node[bb], value);
        update(D, n, method, a, b, value, size, alive);
        alive[bb] = 0;
        size[aa] += size[bb];
    }
};

void generic_matrix(std::vector<double>& D, int64_t n, int method, Merges& M) {
    MatrixOps ops(D, n, method, M);
    generic(n, ops);
}

// ---- the matrix-free path: clusters as centres and sizes (ward, centroid, median) ---------------------------

// The live clusters' centres column by column (cs[d n + q] for the cluster at position q < m) so that a scan
// takes two clusters a NEON register; a cluster's slot is its key, its position only where its numbers are
// stored. For the generic algorithm positions follow the slots' order (a merged-away cluster's position is
// marked and dropped when an eighth of them is); for the chain, which scans them all, the last position fills
// the hole. The squared distance of two centres is
// spatial's sum (SqOp: per block of 32 coordinates the chain t = fma(z, z, t) from 0, z = x - y, the blocks
// added in order; the careful path below kTiny), so a lane computes the scalar's bits; ward's value is
// (2 a b / (a + b)) s, symmetric in the two clusters bit for bit.
struct Centres {
    int64_t n, p, m, dead = 0;
    int method;
    bool ordered = true;  // positions in slot order (the generic algorithm's ranges); else filled from the end
    std::vector<double> cs, sz;           // centres (p, n) and sizes by position
    std::vector<int64_t> slot_of, pos_of, node;
    std::vector<char> dead_at;            // by position: merged away (dropped at the next compaction)
    Centres(const double* X, int64_t n_, int64_t p_, int meth)
        : n(n_),
          p(p_),
          m(n_),
          method(meth),
          cs(static_cast<size_t>(n_ * p_)),
          sz(static_cast<size_t>(n_), 1.0),
          slot_of(static_cast<size_t>(n_)),
          pos_of(static_cast<size_t>(n_)),
          node(static_cast<size_t>(n_)),
          dead_at(static_cast<size_t>(n_), 0) {
        for (int64_t i = 0; i < n; ++i)
            for (int64_t d = 0; d < p; ++d) cs[static_cast<size_t>(d * n + i)] = X[i * p + d];
        std::iota(slot_of.begin(), slot_of.end(), int64_t{0});
        std::iota(pos_of.begin(), pos_of.end(), int64_t{0});
        std::iota(node.begin(), node.end(), int64_t{0});
    }
    void centre(int64_t pos, double* c) const {
        for (int64_t d = 0; d < p; ++d) c[d] = cs[static_cast<size_t>(d * n + pos)];
    }
    double finish(double s, const double* x, const double* y) const {
        return s < spatial::kTiny ? spatial::sq_finish<false>(s, x, y, nullptr, p) : s;
    }
    double ward(double a, double b, double s) const { return method == kWard ? (2.0 * a * b / (a + b)) * s : s; }
    double value(int64_t i, int64_t j) const {
        const int64_t pi = pos_of[static_cast<size_t>(i)], pj = pos_of[static_cast<size_t>(j)];
        double x[256], y[256];
        std::vector<double> xs, ys;
        double* xp = x;
        double* yp = y;
        if (p > 256) {
            xs.resize(static_cast<size_t>(p));
            ys.resize(static_cast<size_t>(p));
            xp = xs.data();
            yp = ys.data();
        }
        centre(pi, xp);
        centre(pj, yp);
        double s;
        spatial::pair_scalar<spatial::SqOp, false>(xp, yp, nullptr, p, &s);
        return ward(sz[static_cast<size_t>(pi)], sz[static_cast<size_t>(pj)], finish(s, xp, yp));
    }
    // out[q - q0] = the value of the cluster whose centre is cx (size a) and the cluster at position q
    void values(const double* cx, double a, int64_t q0, int64_t q1, double* out) const {
        const int64_t nb = (p + spatial::kBlock - 1) / spatial::kBlock;
        int64_t q = q0;
        for (; q + 2 <= q1; q += 2) {
            float64x2_t s = vdupq_n_f64(0.0);
            for (int64_t bk = 0; bk < nb; ++bk) {
                float64x2_t t = vdupq_n_f64(0.0);
                const int64_t end = std::min(p, (bk + 1) * spatial::kBlock);
                for (int64_t d = bk * spatial::kBlock; d < end; ++d) {
                    const float64x2_t z = vsubq_f64(vdupq_n_f64(cx[d]), vld1q_f64(cs.data() + d * n + q));
                    t = vfmaq_f64(t, z, z);
                }
                s = bk == 0 ? t : vaddq_f64(s, t);
            }
            double sv[2];
            vst1q_f64(sv, s);
            for (int k = 0; k < 2; ++k) {
                double v = sv[k];
                if (v < spatial::kTiny) {
                    std::vector<double> y(static_cast<size_t>(p));
                    centre(q + k, y.data());
                    v = finish(v, cx, y.data());  // the careful path (rare)
                }
                out[q + k - q0] = ward(a, sz[static_cast<size_t>(q + k)], v);
            }
        }
        if (q < q1) {  // the last odd cluster: one scalar sum
            double yb[64];
            std::vector<double> yv;
            double* y = yb;
            if (p > 64) {
                yv.resize(static_cast<size_t>(p));
                y = yv.data();
            }
            centre(q, y);
            double v;
            spatial::pair_scalar<spatial::SqOp, false>(cx, y, nullptr, p, &v);
            out[q - q0] = ward(a, sz[static_cast<size_t>(q)], finish(v, cx, y));
        }
    }
    void join(int64_t a, int64_t b, double v, Merges& M) {
        const auto aa = static_cast<size_t>(a), bb = static_cast<size_t>(b);
        node[aa] = M.add(node[aa], node[bb], v);
        const int64_t pa = pos_of[aa], pb = pos_of[bb];
        const double na = sz[static_cast<size_t>(pa)], nb = sz[static_cast<size_t>(pb)];
        for (int64_t d = 0; d < p; ++d) {
            double& x = cs[static_cast<size_t>(d * n + pa)];
            const double y = cs[static_cast<size_t>(d * n + pb)];
            x = method == kMedian ? 0.5 * (x + y) : (na * x + nb * y) / (na + nb);
        }
        sz[static_cast<size_t>(pa)] = na + nb;
        pos_of[bb] = -1;
        if (!ordered) {  // the chain scans everything: the last position fills the hole
            const int64_t last = m - 1;
            if (pb != last) {
                for (int64_t d = 0; d < p; ++d)
                    cs[static_cast<size_t>(d * n + pb)] = cs[static_cast<size_t>(d * n + last)];
                sz[static_cast<size_t>(pb)] = sz[static_cast<size_t>(last)];
                const int64_t moved = slot_of[static_cast<size_t>(last)];
                slot_of[static_cast<size_t>(pb)] = moved;
                pos_of[static_cast<size_t>(moved)] = pb;
            }
            --m;
            return;
        }
        dead_at[static_cast<size_t>(pb)] = 1;
        ++dead;
        if (dead * 8 > m) compact();
    }
    // drops the merged-away positions, keeping the order (so positions stay in increasing slot order)
    void compact() {
        int64_t w = 0;
        for (int64_t q = 0; q < m; ++q) {
            if (dead_at[static_cast<size_t>(q)]) continue;
            if (w != q) {
                for (int64_t d = 0; d < p; ++d) cs[static_cast<size_t>(d * n + w)] = cs[static_cast<size_t>(d * n + q)];
                sz[static_cast<size_t>(w)] = sz[static_cast<size_t>(q)];
                slot_of[static_cast<size_t>(w)] = slot_of[static_cast<size_t>(q)];
            }
            pos_of[static_cast<size_t>(slot_of[static_cast<size_t>(w)])] = w;
            dead_at[static_cast<size_t>(w)] = 0;
            ++w;
        }
        m = w;
        dead = 0;
    }
    // the positions [lo, m) of the slots after x, [0, hi) of the slots before x
    int64_t after(int64_t x) const {
        return std::upper_bound(slot_of.begin(), slot_of.begin() + m, x) - slot_of.begin();
    }
    int64_t before(int64_t x) const {
        return std::lower_bound(slot_of.begin(), slot_of.begin() + m, x) - slot_of.begin();
    }
};

// A scan of every live cluster against slot x: (smallest (value, slot) with slot > lo, how many share its value)
// over the clusters passing keep(slot). Positions in chunks fixed by n (in parallel when large); the chunks'
// results combine by the total order, so the result does not depend on the chunks.
struct Scan {
    double v;
    int64_t j, count;
};

// The positions [q0, q1) against slot x, the merged-away ones skipped.
Scan scan_centres(const Centres& C, int64_t x, int64_t q0, int64_t q1, std::vector<double>& buf) {
    const int64_t p = C.p, len = q1 - q0;
    Scan b{kInf, -1, 0};
    if (len <= 0) return b;
    double cxb[64];
    std::vector<double> cxv;
    double* cx = cxb;
    if (p > 64) {
        cxv.resize(static_cast<size_t>(p));
        cx = cxv.data();
    }
    const int64_t px = C.pos_of[static_cast<size_t>(x)];
    C.centre(px, cx);
    const double a = C.sz[static_cast<size_t>(px)];
    const int64_t chunk = std::max<int64_t>(256, (int64_t{1} << 12) / (p + 4));
    const int64_t nchunk = (len + chunk - 1) / chunk;
    buf.resize(static_cast<size_t>(C.m));
    std::vector<Scan> parts(static_cast<size_t>(nchunk));
    auto body = [&](int64_t t) {
        const int64_t c0 = q0 + t * chunk, c1 = std::min(q1, c0 + chunk);
        C.values(cx, a, c0, c1, buf.data() + c0);
        Scan r{kInf, -1, 0};
        for (int64_t q = c0; q < c1; ++q) {
            if (C.dead_at[static_cast<size_t>(q)] || q == px) continue;
            const int64_t j = C.slot_of[static_cast<size_t>(q)];
            const double v = buf[static_cast<size_t>(q)];
            if (r.j < 0 || v < r.v) r = {v, j, 1};
            else if (v == r.v) {
                ++r.count;
                r.j = std::min(r.j, j);
            }
        }
        parts[static_cast<size_t>(t)] = r;
    };
    if (len * (p + 4) >= kSplitWork && nchunk > 1) parallel::parallel_for(nchunk, body, chunk * (p + 4));
    else
        for (int64_t t = 0; t < nchunk; ++t) body(t);
    for (const Scan& q : parts) {
        if (q.j < 0) continue;
        if (b.j < 0 || q.v < b.v) b = q;
        else if (q.v == b.v) {
            b.count += q.count;
            b.j = std::min(b.j, q.j);
        }
    }
    return b;
}

// The chain on centres (ward). Returns false on an exact tie (CL46).
bool nnchain_centres(Centres& C, Merges& M) {
    const int64_t n = C.n;
    C.ordered = false;
    std::vector<int64_t> chain;
    std::vector<double> buf;
    for (int64_t step = 0; step < n - 1; ++step) {
        if (chain.empty()) {
            int64_t first = C.slot_of[0];
            for (int64_t q = 1; q < C.m; ++q) first = std::min(first, C.slot_of[static_cast<size_t>(q)]);
            chain.push_back(first);
        }
        int64_t x, y;
        double best;
        while (true) {
            x = chain.back();
            const int64_t prev = chain.size() >= 2 ? chain[chain.size() - 2] : -1;
            const Scan b = scan_centres(C, x, 0, C.m, buf);
            if (b.count > 1) return false;
            best = b.v;
            y = b.j;
            if (y == prev) break;
            chain.push_back(y);
        }
        chain.pop_back();
        chain.pop_back();
        C.join(std::min(x, y), std::max(x, y), best, M);
    }
    return true;
}

struct CentreOps {
    Centres& C;
    Merges& M;
    std::vector<double> buf;
    double dist(int64_t i, int64_t j) const { return C.value(i, j); }
    std::pair<double, int64_t> rescan(int64_t x) {
        const Scan b = scan_centres(C, x, C.after(x), C.m, buf);
        return {b.v, b.j};
    }
    template <class F>
    void before(int64_t a, F f) {
        const int64_t hi = C.before(a);
        (void)scan_centres(C, a, 0, hi, buf);  // the values of a against the positions before it
        for (int64_t q = 0; q < hi; ++q)
            if (!C.dead_at[static_cast<size_t>(q)]) f(C.slot_of[static_cast<size_t>(q)], buf[static_cast<size_t>(q)]);
    }
    void join(int64_t a, int64_t b, double v) { C.join(a, b, v, M); }
};

void generic_centres(Centres& C, Merges& M) {
    CentreOps ops{C, M, {}};
    generic(C.n, ops);
}

// ---- single linkage from the spanning tree -----------------------------------------------------------------

// Clusters (union-find roots = keys) with their points in linked lists.
struct Clusters {
    MinUnionFind uf;
    std::vector<int64_t> node, next, tail;
    explicit Clusters(int64_t n)
        : uf(n), node(static_cast<size_t>(n)), next(static_cast<size_t>(n), -1), tail(static_cast<size_t>(n)) {
        std::iota(node.begin(), node.end(), int64_t{0});
        std::iota(tail.begin(), tail.end(), int64_t{0});
    }
    int64_t join(int64_t ra, int64_t rb, double w, Merges& M) {
        const int64_t id = M.add(node[static_cast<size_t>(ra)], node[static_cast<size_t>(rb)], w);
        const int64_t r = uf.unite(ra, rb), o = ra + rb - r;
        next[static_cast<size_t>(tail[static_cast<size_t>(r)])] = o;
        tail[static_cast<size_t>(r)] = tail[static_cast<size_t>(o)];
        node[static_cast<size_t>(r)] = id;
        return r;
    }
    template <class F>
    void points(int64_t root, F f) const {
        for (int64_t x = root; x >= 0; x = next[static_cast<size_t>(x)]) f(x);
    }
};

// The pairs of clusters (roots) at exactly distance w among the points of `roots`' clusters: adj(points,
// root_of, w) gives the pairs (x, y) of points of different clusters at distance w.
template <class Adj>
Merges single_tree(std::vector<Edge> E, int64_t n, Adj adj) {
    std::sort(E.begin(), E.end(), edge_less);
    Merges M(n);
    Clusters C(n);
    const int64_t m = static_cast<int64_t>(E.size());
    std::vector<int64_t> roots, pts, root_of(static_cast<size_t>(n), -1);
    for (int64_t i = 0; i < m;) {
        int64_t j = i;
        while (j < m && E[static_cast<size_t>(j)].w == E[static_cast<size_t>(i)].w) ++j;
        const double w = E[static_cast<size_t>(i)].w;
        if (j - i == 1) {
            C.join(C.uf.find(E[static_cast<size_t>(i)].a), C.uf.find(E[static_cast<size_t>(i)].b), w, M);
            i = j;
            continue;
        }
        // the sets of clusters the group's edges join (the components of the graph of the clusters at distance w)
        roots.clear();
        for (int64_t e = i; e < j; ++e) {
            roots.push_back(C.uf.find(E[static_cast<size_t>(e)].a));
            roots.push_back(C.uf.find(E[static_cast<size_t>(e)].b));
        }
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
        const int64_t k = static_cast<int64_t>(roots.size());
        auto local = [&](int64_t r) {
            return static_cast<int64_t>(std::lower_bound(roots.begin(), roots.end(), r) - roots.begin());
        };
        MinUnionFind luf(k);
        for (int64_t e = i; e < j; ++e) {
            const int64_t x = luf.find(local(C.uf.find(E[static_cast<size_t>(e)].a)));
            const int64_t y = luf.find(local(C.uf.find(E[static_cast<size_t>(e)].b)));
            if (x != y) luf.unite(x, y);
        }
        std::vector<std::vector<int64_t>> groups(static_cast<size_t>(k));
        for (int64_t l = 0; l < k; ++l)
            groups[static_cast<size_t>(luf.find(l))].push_back(roots[static_cast<size_t>(l)]);
        for (const auto& g : groups) {
            if (g.size() < 2) continue;
            if (g.size() == 2) {
                C.join(g[0], g[1], w, M);
                continue;
            }
            // the primitive definition inside the set: the cluster of smallest key absorbs its neighbours at
            // distance w, the neighbour of smallest key first (CL50)
            pts.clear();
            for (const int64_t r : g)
                C.points(r, [&](int64_t x) {
                    pts.push_back(x);
                    root_of[static_cast<size_t>(x)] = r;
                });
            std::vector<std::pair<int64_t, int64_t>> pairs = adj(pts, root_of, w);
            const int64_t c = static_cast<int64_t>(g.size());
            auto li = [&](int64_t r) {
                return static_cast<int64_t>(std::lower_bound(g.begin(), g.end(), r) - g.begin());
            };
            std::vector<std::vector<int64_t>> nb(static_cast<size_t>(c));
            for (const auto& [x, y] : pairs) {
                const int64_t u = li(root_of[static_cast<size_t>(x)]), v = li(root_of[static_cast<size_t>(y)]);
                if (u == v) continue;
                nb[static_cast<size_t>(u)].push_back(v);
                nb[static_cast<size_t>(v)].push_back(u);
            }
            std::vector<char> seen(static_cast<size_t>(c), 0);
            std::priority_queue<int64_t, std::vector<int64_t>, std::greater<>> front;
            seen[0] = 1;
            int64_t grown = g[0], absorbed = 1;
            for (const int64_t v : nb[0]) front.push(v);
            while (!front.empty()) {
                const int64_t v = front.top();
                front.pop();
                if (seen[static_cast<size_t>(v)]) continue;
                seen[static_cast<size_t>(v)] = 1;
                grown = C.join(grown, g[static_cast<size_t>(v)], w, M);
                ++absorbed;
                for (const int64_t u : nb[static_cast<size_t>(v)])
                    if (!seen[static_cast<size_t>(u)]) front.push(u);
            }
            if (absorbed != c) throw std::logic_error("single linkage: a tie group is not connected at its distance");
            for (const int64_t x : pts) root_of[static_cast<size_t>(x)] = -1;
        }
        i = j;
    }
    return M;
}

// The pairs at distance exactly w among pts (points of different clusters), by a scan of all pairs.
template <class Dist>
std::vector<std::pair<int64_t, int64_t>> pairs_scan(
    const std::vector<int64_t>& pts, const std::vector<int64_t>& root_of, double w, int64_t cost, Dist dist
) {
    const int64_t m = static_cast<int64_t>(pts.size());
    const int64_t chunk = std::max<int64_t>(16, (int64_t{1} << 14) / std::max<int64_t>(m * cost, 1));
    const int64_t nt = (m + chunk - 1) / chunk;
    std::vector<std::vector<std::pair<int64_t, int64_t>>> part(static_cast<size_t>(nt));
    auto body = [&](int64_t t) {
        for (int64_t q = t * chunk; q < std::min(m, (t + 1) * chunk); ++q) {
            const int64_t x = pts[static_cast<size_t>(q)];
            for (int64_t r = q + 1; r < m; ++r) {
                const int64_t y = pts[static_cast<size_t>(r)];
                if (root_of[static_cast<size_t>(x)] == root_of[static_cast<size_t>(y)]) continue;
                if (dist(x, y, w)) part[static_cast<size_t>(t)].emplace_back(x, y);
            }
        }
    };
    if (m * m * cost >= 2 * kSplitWork && nt > 1) parallel::parallel_for(nt, body, chunk * m * cost);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    std::vector<std::pair<int64_t, int64_t>> out;
    for (auto& v : part) out.insert(out.end(), v.begin(), v.end());
    return out;
}

// The same pairs by radius searches on the KD tree of all points (points outside pts are never at distance w
// from pts' points across clusters: they would join them below w).
std::vector<std::pair<int64_t, int64_t>> pairs_tree(
    const spatial::KdTree& T,
    const double* X,
    const std::vector<int64_t>& pts,
    const std::vector<int64_t>& root_of,
    double w
) {
    const int64_t m = static_cast<int64_t>(pts.size()), p = T.m;
    const spatial::KdEuclid E{p};
    const spatial::KdCut cut(p, false);
    const double T1 = cut(w);
    const int64_t chunk = 64;
    const int64_t nt = (m + chunk - 1) / chunk;
    std::vector<std::vector<std::pair<int64_t, int64_t>>> part(static_cast<size_t>(nt));
    auto body = [&](int64_t t) {
        std::vector<double> clamp(static_cast<size_t>(p));
        std::vector<int64_t> stack;
        for (int64_t q = t * chunk; q < std::min(m, (t + 1) * chunk); ++q) {
            const int64_t x = pts[static_cast<size_t>(q)];
            const double* xv = X + x * p;
            const int64_t rx = root_of[static_cast<size_t>(x)];
            stack.assign(1, 0);
            while (!stack.empty()) {
                const int64_t id = stack.back();
                stack.pop_back();
                const double* lo = T.lo(id);
                const double* hi = T.hi(id);
                for (int64_t d = 0; d < p; ++d) clamp[static_cast<size_t>(d)] = std::min(std::max(xv[d], lo[d]), hi[d]);
                if (spatial::kd_beyond(E, xv, clamp.data(), T1)) continue;
                const spatial::KdNode& nd = T.nodes[static_cast<size_t>(id)];
                if (nd.right >= 0) {
                    stack.push_back(nd.right);
                    stack.push_back(id + 1);
                    continue;
                }
                for (int64_t r = nd.start; r < nd.end; ++r) {
                    const int64_t y = T.idx[static_cast<size_t>(r)];
                    if (y <= x) continue;  // each pair once
                    const int64_t ry = root_of[static_cast<size_t>(y)];
                    if (ry < 0 || ry == rx) continue;
                    const double* yv = T.data.data() + r * p;
                    const double s = E.raw(xv, yv);
                    if (spatial::KdEuclid::plain(s) && s > spatial::KdEuclid::above(w)) continue;
                    if (E.finish(s, xv, yv) == w) part[static_cast<size_t>(t)].emplace_back(x, y);
                }
            }
        }
    };
    if (m >= 256 && nt > 1) parallel::parallel_for(nt, body, chunk * 2000);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    std::vector<std::pair<int64_t, int64_t>> out;
    for (auto& v : part) out.insert(out.end(), v.begin(), v.end());
    return out;
}

// ---- the heights: the Lance-Williams recurrence on the final tree (CL45) -----------------------------------

// One step of the recurrence: the value of (P1 u P2, Q) from (P1, Q) = v1, (P2, Q) = v2, |P1| = n1, |P2| = n2,
// |Q| = nq and P's own value h. Each form is symmetric in (v1, n1) and (v2, n2) bit for bit (IEEE addition and
// multiplication commute), so the order of a merge's children does not matter.
template <int M>
inline double combine(double v1, double v2, double n1, double n2, double nq, double h) {
    if constexpr (M == kAverage) {
        return (n1 * v1 + n2 * v2) / (n1 + n2);
    } else if constexpr (M == kWeighted) {
        return 0.5 * (v1 + v2);
    } else if constexpr (M == kCentroid) {
        const double t = n1 + n2;
        return (n1 * v1 + n2 * v2) / t - (n1 * n2) * h / (t * t);
    } else if constexpr (M == kMedian) {
        return 0.5 * v1 + 0.5 * v2 - 0.25 * h;
    } else {
        const double t = 1.0 / (n1 + n2 + nq);
        return (n1 + nq) * t * v1 + (n2 + nq) * t * v2 - nq * t * h;
    }
}

// The tree's shape: sizes, keys (smallest point), each merge's outer child (the smaller key: expanded first)
// and inner child, and a post-order (outer subtree, inner subtree, node) in which every subtree is a range.
struct Shape {
    int64_t n;
    std::vector<int64_t> size, key, outer, inner, post, pos;
    explicit Shape(const Merges& M) : n(M.n) {
        const int64_t m = M.count(), total = n + m;
        size.assign(static_cast<size_t>(total), 1);
        key.resize(static_cast<size_t>(total));
        std::iota(key.begin(), key.begin() + n, int64_t{0});
        outer.resize(static_cast<size_t>(m));
        inner.resize(static_cast<size_t>(m));
        for (int64_t t = 0; t < m; ++t) {
            const int64_t a = M.a[static_cast<size_t>(t)], b = M.b[static_cast<size_t>(t)];
            const auto v = static_cast<size_t>(n + t);
            size[v] = size[static_cast<size_t>(a)] + size[static_cast<size_t>(b)];
            const int64_t ka = key[static_cast<size_t>(a)], kb = key[static_cast<size_t>(b)];
            key[v] = std::min(ka, kb);
            outer[static_cast<size_t>(t)] = ka < kb ? a : b;
            inner[static_cast<size_t>(t)] = ka < kb ? b : a;
        }
        post.reserve(static_cast<size_t>(total));
        pos.assign(static_cast<size_t>(total), -1);
        if (m == 0) return;
        std::vector<std::pair<int64_t, int>> st{{total - 1, 0}};
        while (!st.empty()) {
            auto& [v, state] = st.back();
            if (v < n || state == 2) {
                pos[static_cast<size_t>(v)] = static_cast<int64_t>(post.size());
                post.push_back(v);
                st.pop_back();
                continue;
            }
            const int64_t t = v - n;
            const int64_t child = state == 0 ? outer[static_cast<size_t>(t)] : inner[static_cast<size_t>(t)];
            ++state;
            st.emplace_back(child, 0);
        }
    }
    int64_t begin(int64_t v) const { return pos[static_cast<size_t>(v)] - (2 * size[static_cast<size_t>(v)] - 2); }
};

// The value of two points for the recurrence: base(x, y), and base.lanes(xs, u, out) for kLanes points xs
// against one point u (each lane the bits of base(xs[k], u)).
template <class F>
struct FnBase {
    F f;
    double operator()(int64_t x, int64_t y) const { return f(x, y); }
    void lanes(const int64_t* xs, int64_t u, double* out) const;
};

// The squared distance of the rows of X (spatial's sum, careful path below kTiny); four lanes in two NEON
// registers, each lane's chain the scalar's.
struct SqBase {
    const double* X;
    int64_t p;
    double operator()(int64_t x, int64_t y) const {
        const double* xv = X + x * p;
        const double* yv = X + y * p;
        double s;
        spatial::pair_scalar<spatial::SqOp, false>(xv, yv, nullptr, p, &s);
        return spatial::sq_finish<false>(s, xv, yv, nullptr, p);
    }
    void lanes(const int64_t* xs, int64_t u, double* out) const;
};

// H[t]: the value of merge t (squared for ward, centroid, median) by the recurrence; base(x, y) the value of
// two points, cost its work in flops. The value of a point x and a cluster B runs over B's subtree in post-order
// with a stack; kLanes points of the outer side go through B together (one pass over B's structure, the lanes'
// operations independent), each lane's operations exactly those of a pass of its own.
constexpr int kLanes = 4;

template <int Mth, class Base>
std::vector<double> recurrence_of(const Merges& M, const Base& base, int64_t cost) {
    const Shape S(M);
    const int64_t n = M.n, m = M.count();
    std::vector<double> H(static_cast<size_t>(m));
    // per merge: the sizes of its children
    std::vector<double> s1(static_cast<size_t>(m)), s2(static_cast<size_t>(m));
    for (int64_t t = 0; t < m; ++t) {
        s1[static_cast<size_t>(t)] = static_cast<double>(S.size[static_cast<size_t>(S.outer[static_cast<size_t>(t)])]);
        s2[static_cast<size_t>(t)] = static_cast<double>(S.size[static_cast<size_t>(S.inner[static_cast<size_t>(t)])]);
    }
    const int64_t* post = S.post.data();
    // the values of the lanes xs[0 .. kLanes) and cluster B into out
    auto scan = [&](const int64_t* xs, int64_t B, double* out, std::vector<double>& stk) {
        if (B < n) {
            for (int j = 0; j < kLanes; ++j) out[j] = base(xs[j], B);
            return;
        }
        int64_t sp = 0;
        double* st = stk.data();
        for (int64_t i = S.begin(B), e = S.pos[static_cast<size_t>(B)]; i <= e; ++i) {
            const int64_t u = post[i];
            if (u < n) {
                double* v = st + sp * kLanes;
                base.lanes(xs, u, v);
                ++sp;
            } else {
                const auto t = static_cast<size_t>(u - n);
                const double a1 = s1[t], a2 = s2[t], h = H[t];
                double* v1 = st + (sp - 2) * kLanes;
                const double* v2 = v1 + kLanes;
                for (int j = 0; j < kLanes; ++j) v1[j] = combine<Mth>(v1[j], v2[j], a1, a2, 1.0, h);
                --sp;
            }
        }
        for (int j = 0; j < kLanes; ++j) out[j] = st[j];
    };
    std::vector<double> w, stk(static_cast<size_t>(kLanes * (n + 1)));
    std::vector<int64_t> leaves;
    for (int64_t t = 0; t < m; ++t) {
        const int64_t A = S.outer[static_cast<size_t>(t)], B = S.inner[static_cast<size_t>(t)];
        if (A < n) {
            const int64_t xs[kLanes] = {A, A, A, A};
            double out[kLanes];
            scan(xs, B, out, stk);
            H[static_cast<size_t>(t)] = out[0];
            continue;
        }
        leaves.clear();
        for (int64_t i = S.begin(A); i <= S.pos[static_cast<size_t>(A)]; ++i) {
            const int64_t u = post[i];
            if (u < n) leaves.push_back(u);
        }
        const int64_t la = static_cast<int64_t>(leaves.size()), lb = S.size[static_cast<size_t>(B)];
        const int64_t groups = (la + kLanes - 1) / kLanes;
        while (static_cast<int64_t>(leaves.size()) < groups * kLanes) leaves.push_back(leaves.back());
        w.resize(static_cast<size_t>(groups * kLanes));
        const int64_t per = kLanes * lb * (cost + 4);
        const int64_t chunk = std::max<int64_t>(1, (int64_t{1} << 15) / per);
        const int64_t nt = (groups + chunk - 1) / chunk;
        auto body = [&](int64_t c) {
            std::vector<double> local;
            std::vector<double>& st = nt > 1 ? local : stk;
            if (nt > 1) local.resize(static_cast<size_t>(kLanes * (lb + 1)));
            for (int64_t q = c * chunk; q < std::min(groups, (c + 1) * chunk); ++q)
                scan(leaves.data() + q * kLanes, B, w.data() + q * kLanes, st);
        };
        if (groups * per >= kSplitWork && nt > 1) parallel::parallel_for(nt, body, chunk * per);
        else
            for (int64_t c = 0; c < nt; ++c) body(c);
        // A's subtree: the points' values w, combined up with |B| as the other side
        const double nb = static_cast<double>(lb);
        int64_t sp = 0, next = 0;
        double* st = stk.data();
        for (int64_t i = S.begin(A); i <= S.pos[static_cast<size_t>(A)]; ++i) {
            const int64_t u = post[i];
            if (u < n) {
                st[sp++] = w[static_cast<size_t>(next++)];
            } else {
                const auto tt = static_cast<size_t>(u - n);
                st[sp - 2] = combine<Mth>(st[sp - 2], st[sp - 1], s1[tt], s2[tt], nb, H[tt]);
                --sp;
            }
        }
        H[static_cast<size_t>(t)] = st[0];
    }
    return H;
}

template <class F>
void FnBase<F>::lanes(const int64_t* xs, int64_t u, double* out) const {
    for (int j = 0; j < 4; ++j) out[j] = f(xs[j], u);
}

void SqBase::lanes(const int64_t* xs, int64_t u, double* out) const {
    const double* y = X + u * p;
    const double* x0 = X + xs[0] * p;
    const double* x1 = X + xs[1] * p;
    const double* x2 = X + xs[2] * p;
    const double* x3 = X + xs[3] * p;
    float64x2_t s01 = vdupq_n_f64(0.0), s23 = vdupq_n_f64(0.0);
    for (int64_t b = 0; b < p; b += spatial::kBlock) {
        float64x2_t t01 = vdupq_n_f64(0.0), t23 = vdupq_n_f64(0.0);
        const int64_t end = std::min(p, b + spatial::kBlock);
        for (int64_t d = b; d < end; ++d) {
            const float64x2_t yd = vdupq_n_f64(y[d]);
            const float64x2_t z01 = vsubq_f64(vsetq_lane_f64(x1[d], vdupq_n_f64(x0[d]), 1), yd);
            const float64x2_t z23 = vsubq_f64(vsetq_lane_f64(x3[d], vdupq_n_f64(x2[d]), 1), yd);
            t01 = vfmaq_f64(t01, z01, z01);
            t23 = vfmaq_f64(t23, z23, z23);
        }
        if (b == 0) s01 = t01, s23 = t23;
        else s01 = vaddq_f64(s01, t01), s23 = vaddq_f64(s23, t23);
    }
    vst1q_f64(out, s01);
    vst1q_f64(out + 2, s23);
    for (int j = 0; j < 4; ++j)
        if (out[j] < spatial::kTiny) out[j] = spatial::sq_finish<false>(out[j], X + xs[j] * p, y, nullptr, p);
}

template <class F>
FnBase<F> fn_base(F f) {
    return FnBase<F>{f};
}

template <class Base>
std::vector<double> recurrence(const Merges& M, int method, const Base& base, int64_t cost) {
    switch (method) {
        case kAverage: return recurrence_of<kAverage>(M, base, cost);
        case kWeighted: return recurrence_of<kWeighted>(M, base, cost);
        case kCentroid: return recurrence_of<kCentroid>(M, base, cost);
        case kMedian: return recurrence_of<kMedian>(M, base, cost);
        default: return recurrence_of<kWard>(M, base, cost);
    }
}

// ---- the rows of Z ------------------------------------------------------------------------------------------

// heights h[t] (the output scale) -> Z (n - 1, 4) in the input's numbering: order[key] = input index.
std::vector<double> rows(const Merges& M, std::vector<double> h, bool monotone, const int64_t* order) {
    const int64_t n = M.n, m = M.count();
    std::vector<int64_t> size(static_cast<size_t>(n + m), 1), key(static_cast<size_t>(n + m));
    std::iota(key.begin(), key.begin() + n, int64_t{0});
    std::vector<int64_t> parent(static_cast<size_t>(n + m), -1);
    for (int64_t t = 0; t < m; ++t) {
        const int64_t a = M.a[static_cast<size_t>(t)], b = M.b[static_cast<size_t>(t)];
        const auto v = static_cast<size_t>(n + t);
        size[v] = size[static_cast<size_t>(a)] + size[static_cast<size_t>(b)];
        key[v] = std::min(key[static_cast<size_t>(a)], key[static_cast<size_t>(b)]);
        parent[static_cast<size_t>(a)] = parent[static_cast<size_t>(b)] = n + t;
        if (monotone) {  // CL51: a height is at least its children's
            for (const int64_t c : {a, b})
                if (c >= n)
                    h[static_cast<size_t>(t)] = std::max(h[static_cast<size_t>(t)], h[static_cast<size_t>(c - n)]);
        }
    }
    // the primitive order on the final heights: the available merge of smallest (height, key pair)
    struct Item {
        double h;
        int64_t k1, k2, t;
        bool operator>(const Item& o) const {
            if (h != o.h) return h > o.h;
            if (k1 != o.k1) return k1 > o.k1;
            return k2 > o.k2;
        }
    };
    auto item = [&](int64_t t) {
        const int64_t ka = key[static_cast<size_t>(M.a[static_cast<size_t>(t)])];
        const int64_t kb = key[static_cast<size_t>(M.b[static_cast<size_t>(t)])];
        return Item{h[static_cast<size_t>(t)], std::min(ka, kb), std::max(ka, kb), t};
    };
    std::priority_queue<Item, std::vector<Item>, std::greater<>> ready;
    std::vector<int> waiting(static_cast<size_t>(m), 0);
    for (int64_t t = 0; t < m; ++t) {
        waiting[static_cast<size_t>(t)] = (M.a[static_cast<size_t>(t)] >= n) + (M.b[static_cast<size_t>(t)] >= n);
        if (waiting[static_cast<size_t>(t)] == 0) ready.push(item(t));
    }
    std::vector<int64_t> id(static_cast<size_t>(n + m));
    for (int64_t i = 0; i < n; ++i) id[static_cast<size_t>(i)] = order ? order[i] : i;
    std::vector<double> Z(static_cast<size_t>(4 * m));
    double* z = Z.data();
    for (int64_t r = 0; r < m; ++r) {
        const Item it = ready.top();
        ready.pop();
        const int64_t t = it.t;
        const int64_t a = id[static_cast<size_t>(M.a[static_cast<size_t>(t)])];
        const int64_t b = id[static_cast<size_t>(M.b[static_cast<size_t>(t)])];
        z[4 * r] = static_cast<double>(std::min(a, b));
        z[4 * r + 1] = static_cast<double>(std::max(a, b));
        z[4 * r + 2] = it.h;
        z[4 * r + 3] = static_cast<double>(size[static_cast<size_t>(n + t)]);
        id[static_cast<size_t>(n + t)] = n + r;
        const int64_t par = parent[static_cast<size_t>(n + t)];
        if (par >= 0 && --waiting[static_cast<size_t>(par - n)] == 0) ready.push(item(par - n));
    }
    return Z;
}

NDArray z_array_of(const std::vector<double>& z) {
    const int64_t shape[2] = {static_cast<int64_t>(z.size()) / 4, 4};
    NDArray Z = new_array(DType::Float64, shape, false);
    std::copy(z.begin(), z.end(), reinterpret_cast<double*>(Z.data));
    return Z;
}

nb::object z_array(const std::vector<double>& z) { return wrap(z_array_of(z)); }

// ---- the driver ------------------------------------------------------------------------------------------------

std::vector<double> squared_matrix(const double* X, int64_t n, int64_t p) {
    std::vector<double> D(static_cast<size_t>(n * (n - 1) / 2));
    const int64_t task = std::max<int64_t>(1, (int64_t{1} << 14) / std::max<int64_t>(n * (p + 2), 1));
    const int64_t nt = (n + task - 1) / task;
    auto body = [&](int64_t t) {
        for (int64_t i = t * task; i < std::min(n, (t + 1) * task); ++i)
            for (int64_t j = i + 1; j < n; ++j) {
                const double* x = X + i * p;
                const double* y = X + j * p;
                double s;
                spatial::pair_scalar<spatial::SqOp, false>(x, y, nullptr, p, &s);
                D[static_cast<size_t>(cidx(n, i, j))] = spatial::sq_finish<false>(s, x, y, nullptr, p);
            }
    };
    if (n * n * p >= kSplitWork && nt > 1) parallel::parallel_for(nt, body);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    return D;
}

struct Linkage {
    std::vector<double> Z;
    int flags = 0;
};

Linkage run_linkage(
    const double* X, int64_t n, int64_t p, const double* D, int method, int path, const int64_t* order
) {
    Linkage out;
    Merges M(n);
    // CL43: the data of the squared methods scaled by a power of two
    std::vector<double> Xs;
    int e = 0;
    const double* Xw = X;
    if (X && squared_method(method)) {
        double mx = 0.0;
        for (int64_t i = 0; i < n * p; ++i) mx = std::max(mx, std::fabs(X[i]));
        e = scale_exponent(mx);
        if (e != 0) {
            Xs.resize(static_cast<size_t>(n * p));
            for (int64_t i = 0; i < n * p; ++i) Xs[static_cast<size_t>(i)] = std::ldexp(X[i], e);
            Xw = Xs.data();
        }
    }
    if (method == kSingle) {
        std::vector<Edge> E;
        if (D) {
            E = mst_prim_matrix(D, n);
            M = single_tree(
                std::move(E), n, [&](const std::vector<int64_t>& pts, const std::vector<int64_t>& root_of, double w) {
                    return pairs_scan(pts, root_of, w, 2, [&](int64_t x, int64_t y, double ww) {
                        return D[static_cast<size_t>(cpair(n, x, y))] == ww;
                    });
                }
            );
        } else {
            const bool tree = path == 5 || (path != 4 && boruvka_shape(n, p));
            spatial::KdTree T;
            if (tree) {
                T = spatial::kd_build(X, n, p, p, kLeafSize);
                E = mst_boruvka(T);
            } else {
                E = mst_prim(X, n, p);
            }
            const spatial::KdEuclid Eu{p};
            M = single_tree(
                std::move(E), n, [&](const std::vector<int64_t>& pts, const std::vector<int64_t>& root_of, double w) {
                    if (tree) return pairs_tree(T, X, pts, root_of, w);
                    return pairs_scan(pts, root_of, w, p + 2, [&](int64_t x, int64_t y, double ww) {
                        const double* xv = X + x * p;
                        const double* yv = X + y * p;
                        const double s = Eu.raw(xv, yv);
                        if (spatial::KdEuclid::plain(s) && s > spatial::KdEuclid::above(ww)) return false;
                        return Eu.finish(s, xv, yv) == ww;
                    });
                }
            );
        }
        std::vector<double> h(M.v);
        out.Z = rows(M, std::move(h), true, order);
        return out;
    }
    const bool vector_path = X && !D && squared_method(method) && path != 1;
    if (vector_path) {
        Centres C(Xw, n, p, method);
        bool done = false;
        if (method == kWard && path != 3) {
            done = nnchain_centres(C, M);
            if (!done) {
                out.flags |= 1;
                M = Merges(n);
                C = Centres(Xw, n, p, method);
            }
        }
        if (!done) generic_centres(C, M);
    } else {
        // the matrix: D as given, or (ward, centroid, median on X) the squared sums
        std::vector<double> W;
        auto fresh = [&]() {
            if (X && squared_method(method)) {
                W = squared_matrix(Xw, n, p);
            } else if (squared_method(method)) {
                W.assign(D, D + n * (n - 1) / 2);
                for (double& v : W) v = v * v;
            } else {
                W.assign(D, D + n * (n - 1) / 2);
            }
        };
        bool done = false;
        const bool chain =
            (method == kComplete || method == kAverage || method == kWeighted || method == kWard) && path != 3;
        // the square copy when it takes at most a quarter of the memory (the same tree; CL96)
        const bool square =
            chain && D && !(X && squared_method(method)) && path != 6 &&
            static_cast<double>(n) * static_cast<double>(n) * 8.0 <= static_cast<double>(physical_memory()) / 4.0;
        if (square) {
            std::vector<double> S = square_of(D, n, squared_method(method));
            done = nnchain_square(S, n, method, M);
            if (!done) {
                out.flags |= 1;
                M = Merges(n);
            }
        } else {
            fresh();
            if (chain) {
                done = nnchain_matrix(W, n, method, M);
                if (!done) {
                    out.flags |= 1;
                    M = Merges(n);
                    fresh();
                }
            }
        }
        if (!done) {
            if (W.empty()) fresh();
            generic_matrix(W, n, method, M);
        }
    }
    std::vector<double> h;
    if (method == kComplete) {
        h = M.v;
    } else if (X && (squared_method(method) || !D)) {
        h = recurrence(M, method, SqBase{Xw, p}, p + 2);
    } else if (squared_method(method)) {
        h = recurrence(
            M, method, fn_base([&](int64_t x, int64_t y) {
                const double d = D[static_cast<size_t>(cpair(n, x, y))];
                return d * d;
            }),
            2
        );
    } else {
        h = recurrence(
            M, method, fn_base([&](int64_t x, int64_t y) { return D[static_cast<size_t>(cpair(n, x, y))]; }), 1
        );
    }
    if (squared_method(method))
        for (double& v : h) v = std::ldexp(std::sqrt(std::max(v, 0.0)), -e);
    out.Z = rows(M, std::move(h), monotone_method(method), order);
    return out;
}

nb::object linkage_py(nb::handle Xo, nb::handle Do, nb::handle oo, int method, int path) {
    int64_t n = 0, p = 0, m = 0, no = 0;
    const double* X = Xo.is_none() ? nullptr : rows_of(Xo, n, p);
    const double* D = nullptr;
    if (!Do.is_none()) {
        D = vector_of(Do, "D", m);
        const int64_t nd = condensed_n(m);
        if (X && nd != n) throw nb::value_error("X and D disagree");
        n = nd;
    }
    const int64_t* order = ints_of(oo, "order", no);
    if (no != n || n < 2) throw nb::value_error("order: one per point, n >= 2");
    if (method < 0 || method > 6) throw nb::value_error("method");
    if (!X && !D) throw nb::value_error("X or D");
    if (!D && (method == kComplete || method == kAverage || method == kWeighted)) throw nb::value_error("needs D");
    Linkage L;
    {
        const parallel::Unlocked unlocked(n * n >= kSplitWork);
        L = run_linkage(X, n, p, D, method, path, order);
    }
    return nb::make_tuple(z_array(L.Z), L.flags);
}

nb::object lex_order_py(nb::handle Xo) {
    int64_t n, p;
    const double* X = rows_of(Xo, n, p);
    std::vector<int64_t> order;
    {
        const parallel::Unlocked unlocked(n * p >= kSplitWork);
        order = lex_order(X, n, p);
    }
    const int64_t shape[1] = {n};
    NDArray o = new_array(DType::Int64, shape, false);
    std::copy(order.begin(), order.end(), reinterpret_cast<int64_t*>(o.data));
    return wrap(std::move(o));
}

// ---- DIANA ----------------------------------------------------------------------------------------------------

// R's diana (Kaufman and Rousseeuw 1990, ch. 6): a cluster of two or more points is split by moving to a
// splinter group first the point of largest mean distance to the others, then, while some point of the rest
// is on average strictly closer to the splinter group than to the rest of the rest, the point where that
// difference is largest (ties: the smaller key). The split's height is the cluster's diameter. The tree does not
// depend on the order the clusters are split in; they are split top down and merged bottom up.
Merges diana(const double* D, int64_t n, std::vector<double>& h) {
    auto d = [&](int64_t i, int64_t j) { return D[static_cast<size_t>(cpair(n, i, j))]; };
    struct Part {
        std::vector<int64_t> pts;  // sorted keys; emptied once split
        int64_t point = -1;        // a single point's key
        double diam = 0.0;
        int64_t left = -1, right = -1;
    };
    std::vector<Part> parts(1);
    parts[0].pts.resize(static_cast<size_t>(n));
    std::iota(parts[0].pts.begin(), parts[0].pts.end(), int64_t{0});
    std::vector<double> rest, spl;
    std::vector<char> in;
    for (size_t i = 0; i < parts.size(); ++i) {
        std::vector<int64_t> pts = std::move(parts[i].pts);
        const int64_t m = static_cast<int64_t>(pts.size());
        if (m == 1) {
            parts[i].point = pts[0];
            continue;
        }
        double diam = 0.0;
        rest.assign(static_cast<size_t>(m), 0.0);  // the sum of distances to the rest (the others, at first)
        for (int64_t a = 0; a < m; ++a)
            for (int64_t b = a + 1; b < m; ++b) {
                const double v = d(pts[static_cast<size_t>(a)], pts[static_cast<size_t>(b)]);
                rest[static_cast<size_t>(a)] += v;
                rest[static_cast<size_t>(b)] += v;
                diam = std::max(diam, v);
            }
        in.assign(static_cast<size_t>(m), 0);
        spl.assign(static_cast<size_t>(m), 0.0);  // the sum of distances to the splinter group
        int64_t first = 0;
        for (int64_t a = 1; a < m; ++a)
            if (rest[static_cast<size_t>(a)] > rest[static_cast<size_t>(first)]) first = a;
        auto move = [&](int64_t q) {
            in[static_cast<size_t>(q)] = 1;
            for (int64_t a = 0; a < m; ++a) {
                if (in[static_cast<size_t>(a)]) continue;
                const double v = d(pts[static_cast<size_t>(a)], pts[static_cast<size_t>(q)]);
                rest[static_cast<size_t>(a)] -= v;
                spl[static_cast<size_t>(a)] += v;
            }
        };
        move(first);
        int64_t ns = 1, no = m - 1;
        while (no > 1) {
            int64_t arg = -1;
            double best = 0.0;
            for (int64_t a = 0; a < m; ++a) {
                if (in[static_cast<size_t>(a)]) continue;
                const double diff = rest[static_cast<size_t>(a)] / static_cast<double>(no - 1) -
                                    spl[static_cast<size_t>(a)] / static_cast<double>(ns);
                if (arg < 0 || diff > best) best = diff, arg = a;
            }
            if (!(best > 0.0)) break;
            move(arg);
            ++ns;
            --no;
        }
        Part l, r;
        for (int64_t a = 0; a < m; ++a) (in[static_cast<size_t>(a)] ? l : r).pts.push_back(pts[static_cast<size_t>(a)]);
        parts[i].diam = diam;
        parts[i].left = static_cast<int64_t>(parts.size());
        parts[i].right = parts[i].left + 1;
        parts.push_back(std::move(l));
        parts.push_back(std::move(r));
    }
    Merges M(n);
    std::vector<int64_t> node(parts.size());
    for (size_t i = parts.size(); i-- > 0;) {
        const Part& q = parts[i];
        if (q.point >= 0) {
            node[i] = q.point;
            continue;
        }
        node[i] = M.add(node[static_cast<size_t>(q.left)], node[static_cast<size_t>(q.right)], q.diam);
        h.push_back(q.diam);
    }
    return M;
}

nb::object diana_py(nb::handle Do, nb::handle oo) {
    int64_t m, no;
    const double* D = vector_of(Do, "D", m);
    const int64_t n = condensed_n(m);
    const int64_t* order = ints_of(oo, "order", no);
    if (no != n || n < 2) throw nb::value_error("order");
    std::vector<double> z;
    {
        const parallel::Unlocked unlocked(n * n >= kSplitWork);
        std::vector<double> h;
        Merges M = diana(D, n, h);
        z = rows(M, std::move(h), true, order);
    }
    return z_array(z);
}

// ---- tools on Z --------------------------------------------------------------------------------------------------

// The children of Z's rows as ids (validated by the caller: a valid linkage).
void z_children(const double* z, int64_t m, std::vector<int64_t>& a, std::vector<int64_t>& b) {
    a.resize(static_cast<size_t>(m));
    b.resize(static_cast<size_t>(m));
    for (int64_t t = 0; t < m; ++t) {
        a[static_cast<size_t>(t)] = static_cast<int64_t>(z[4 * t]);
        b[static_cast<size_t>(t)] = static_cast<int64_t>(z[4 * t + 1]);
    }
}

nb::object cophenet_py(nb::handle Zo) {
    int64_t m, four;
    NDArray& za = require_ndarray(Zo, "Z");
    if (za.dtype != DType::Float64 || !za.c_contiguous || za.ndim != 2 || za.shape[1] != 4)
        throw nb::value_error("Z: C (n - 1, 4) float64");
    m = za.shape[0];
    four = 4;
    (void)four;
    const auto* z = reinterpret_cast<const double*>(za.data);
    const int64_t n = m + 1;
    const int64_t shape[1] = {n * (n - 1) / 2};
    NDArray out = new_array(DType::Float64, shape, false);
    auto* c = reinterpret_cast<double*>(out.data);
    {
        const parallel::Unlocked unlocked(n * n >= kSplitWork);
        std::vector<int64_t> a, b;
        z_children(z, m, a, b);
        // each cluster's points as a sorted list (two lists merged per row), so that a merge writes its pairs row by
        // row: (i, j) with i < j at row[i] + j, the columns of a row in increasing order
        std::vector<std::vector<int64_t>> pts(static_cast<size_t>(n + m));
        for (int64_t i = 0; i < n; ++i) pts[static_cast<size_t>(i)] = {i};
        std::vector<int64_t> row(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) row[static_cast<size_t>(i)] = cidx(n, i, std::min(i + 1, n - 1)) - (i + 1);
        // the pairs of u's points with the larger points of the other list, row u's run
        auto rows_of = [&](const std::vector<int64_t>& P, const std::vector<int64_t>& Q, int64_t q0, int64_t q1,
                           double h) {
            for (int64_t q = q0; q < q1; ++q) {
                const int64_t u = P[static_cast<size_t>(q)];
                double* r = c + row[static_cast<size_t>(u)];
                for (auto it = std::upper_bound(Q.begin(), Q.end(), u); it != Q.end(); ++it) r[*it] = h;
            }
        };
        for (int64_t t = 0; t < m; ++t) {
            std::vector<int64_t>& A = pts[static_cast<size_t>(a[static_cast<size_t>(t)])];
            std::vector<int64_t>& B = pts[static_cast<size_t>(b[static_cast<size_t>(t)])];
            const double h = z[4 * t + 2];
            const int64_t la = static_cast<int64_t>(A.size()), lb = static_cast<int64_t>(B.size());
            if (la * lb >= kSplitWork && la + lb > 2) {
                const int64_t chunk = std::max<int64_t>(1, (int64_t{1} << 14) / std::max(la, lb));
                const int64_t na = (la + chunk - 1) / chunk, nb = (lb + chunk - 1) / chunk;
                parallel::parallel_for(na + nb, [&](int64_t tt) {
                    if (tt < na) rows_of(A, B, tt * chunk, std::min(la, (tt + 1) * chunk), h);
                    else rows_of(B, A, (tt - na) * chunk, std::min(lb, (tt - na + 1) * chunk), h);
                });
            } else {
                rows_of(A, B, 0, la, h);
                rows_of(B, A, 0, lb, h);
            }
            std::vector<int64_t>& V = pts[static_cast<size_t>(n + t)];
            V.resize(static_cast<size_t>(la + lb));
            std::merge(A.begin(), A.end(), B.begin(), B.end(), V.begin());
            std::vector<int64_t>().swap(A);
            std::vector<int64_t>().swap(B);
        }
    }
    return wrap(std::move(out));
}

// Bar-Joseph, Gifford and Jaakkola (2001): the leaf order of smallest sum of distances between neighbouring
// leaves among the orders the tree allows (flipping any merge's children). M[u][w], u and w on the two sides of
// their lowest common merge, is the smallest cost of that merge's leaves ordered from u to w; at the root the
// first child stays on the left. Ties: the smaller keys (u, then w at the root; m, then k inside).
nb::object optimal_leaf_ordering_py(nb::handle Zo, nb::handle Do, nb::handle ko) {
    NDArray& za = require_ndarray(Zo, "Z");
    if (za.dtype != DType::Float64 || !za.c_contiguous || za.ndim != 2 || za.shape[1] != 4)
        throw nb::value_error("Z: C (n - 1, 4) float64");
    const int64_t m = za.shape[0], n = m + 1;
    int64_t md, nk;
    const double* D = vector_of(Do, "D", md);
    const int64_t* key = ints_of(ko, "key", nk);
    if (md != n * (n - 1) / 2 || nk != n) throw nb::value_error("D, key");
    const auto* z = reinterpret_cast<const double*>(za.data);
    const int64_t shape[2] = {m, 4};
    NDArray out = new_array(DType::Float64, shape, false);
    auto* zo = reinterpret_cast<double*>(out.data);
    std::copy(z, z + 4 * m, zo);
    if (n <= 2) return wrap(std::move(out));
    {
        const parallel::Unlocked unlocked(n * n >= kSplitWork);
        std::vector<int64_t> a, b;
        z_children(z, m, a, b);
        auto dist = [&](int64_t i, int64_t j) { return D[cpair(n, i, j)]; };
        // leaves of every node, in Z's child order (contiguous ranges of one array)
        std::vector<int64_t> lo(static_cast<size_t>(n + m)), hi(static_cast<size_t>(n + m));
        std::vector<int64_t> leaf;  // in-order of the tree as given
        leaf.reserve(static_cast<size_t>(n));
        {
            std::vector<std::pair<int64_t, int>> st{{n + m - 1, 0}};
            while (!st.empty()) {
                auto& [v, s] = st.back();
                if (v < n) {
                    lo[static_cast<size_t>(v)] = static_cast<int64_t>(leaf.size());
                    leaf.push_back(v);
                    hi[static_cast<size_t>(v)] = static_cast<int64_t>(leaf.size());
                    st.pop_back();
                    continue;
                }
                const int64_t t = v - n;
                if (s == 0) {
                    lo[static_cast<size_t>(v)] = static_cast<int64_t>(leaf.size());
                    s = 1;
                    st.emplace_back(a[static_cast<size_t>(t)], 0);
                } else if (s == 1) {
                    s = 2;
                    st.emplace_back(b[static_cast<size_t>(t)], 0);
                } else {
                    hi[static_cast<size_t>(v)] = static_cast<int64_t>(leaf.size());
                    st.pop_back();
                }
            }
        }
        std::vector<int64_t> where(static_cast<size_t>(n));  // a leaf's position in `leaf`
        for (int64_t i = 0; i < n; ++i) where[static_cast<size_t>(leaf[static_cast<size_t>(i)])] = i;
        // M over positions: M[i * n + j]
        std::vector<double> C(static_cast<size_t>(n * n), 0.0);
        auto cost = [&](int64_t i, int64_t j) -> double& { return C[static_cast<size_t>(i * n + j)]; };
        // the other child of node v's child containing position i: the range of candidates
        auto other = [&](int64_t v, int64_t i, int64_t& l0, int64_t& l1) {
            if (v < n) {
                l0 = lo[static_cast<size_t>(v)];
                l1 = hi[static_cast<size_t>(v)];
                return;
            }
            const int64_t t = v - n, ca = a[static_cast<size_t>(t)], cb = b[static_cast<size_t>(t)];
            const int64_t c = i < hi[static_cast<size_t>(ca)] ? cb : ca;
            l0 = lo[static_cast<size_t>(c)];
            l1 = hi[static_cast<size_t>(c)];
        };
        for (int64_t t = 0; t < m; ++t) {
            const int64_t L = a[static_cast<size_t>(t)], R = b[static_cast<size_t>(t)];
            const int64_t l0 = lo[static_cast<size_t>(L)], l1 = hi[static_cast<size_t>(L)];
            const int64_t r0 = lo[static_cast<size_t>(R)], r1 = hi[static_cast<size_t>(R)];
            const int64_t nl = l1 - l0, nr = r1 - r0;
            auto body = [&](int64_t u) {
                std::vector<double> Tk(static_cast<size_t>(nr));
                int64_t m0, m1;
                other(L, u, m0, m1);
                for (int64_t k = r0; k < r1; ++k) {
                    double best = kInf;
                    for (int64_t q = m0; q < m1; ++q) {
                        const double v = (L < n ? 0.0 : cost(u, q)) +
                                         dist(leaf[static_cast<size_t>(q)], leaf[static_cast<size_t>(k)]);
                        best = std::min(best, v);
                    }
                    Tk[static_cast<size_t>(k - r0)] = best;
                }
                for (int64_t w = r0; w < r1; ++w) {
                    int64_t k0, k1;
                    other(R, w, k0, k1);
                    double best = kInf;
                    for (int64_t k = k0; k < k1; ++k)
                        best = std::min(best, Tk[static_cast<size_t>(k - r0)] + (R < n ? 0.0 : cost(k, w)));
                    cost(u, w) = best;
                    cost(w, u) = best;
                }
            };
            const int64_t work = nl * nr * (nl + nr);
            if (work >= kSplitWork && nl > 1) parallel::parallel_for(nl, [&](int64_t q) { body(l0 + q); }, work / nl);
            else
                for (int64_t u = l0; u < l1; ++u) body(u);
        }
        // reconstruction: node v ordered from position u to position w (u, w on its two sides, or v a leaf)
        auto kk = [&](int64_t pos) { return key[static_cast<size_t>(leaf[static_cast<size_t>(pos)])]; };
        const int64_t root = n + m - 1;
        int64_t bu = -1, bw = -1;
        {
            const int64_t L = a[static_cast<size_t>(m - 1)], R = b[static_cast<size_t>(m - 1)];
            double best = kInf;
            for (int64_t u = lo[static_cast<size_t>(L)]; u < hi[static_cast<size_t>(L)]; ++u)
                for (int64_t w = lo[static_cast<size_t>(R)]; w < hi[static_cast<size_t>(R)]; ++w) {
                    const double v = cost(u, w);
                    if (bu < 0 || v < best || (v == best && (kk(u) < kk(bu) || (kk(u) == kk(bu) && kk(w) < kk(bw)))))
                        best = v, bu = u, bw = w;
                }
        }
        std::vector<std::tuple<int64_t, int64_t, int64_t>> st{{root, bu, bw}};
        while (!st.empty()) {
            const auto [v, u, w] = st.back();
            st.pop_back();
            if (v < n) continue;
            const int64_t t = v - n, ca = a[static_cast<size_t>(t)], cb = b[static_cast<size_t>(t)];
            // the child holding u goes first
            const bool u_in_a = u >= lo[static_cast<size_t>(ca)] && u < hi[static_cast<size_t>(ca)];
            const int64_t first = u_in_a ? ca : cb, second = u_in_a ? cb : ca;
            if (!u_in_a) {
                zo[4 * t] = z[4 * t + 1];
                zo[4 * t + 1] = z[4 * t];
            }
            int64_t f0, f1, s0, s1;
            other(first, u, f0, f1);
            other(second, w, s0, s1);
            int64_t bm = -1, bk = -1;
            double best = kInf;
            for (int64_t q = f0; q < f1; ++q)
                for (int64_t k = s0; k < s1; ++k) {
                    const double val = (first < n ? 0.0 : cost(u, q)) +
                                       dist(leaf[static_cast<size_t>(q)], leaf[static_cast<size_t>(k)]) +
                                       (second < n ? 0.0 : cost(k, w));
                    if (bm < 0 || val < best ||
                        (val == best && (kk(q) < kk(bm) || (kk(q) == kk(bm) && kk(k) < kk(bk)))))
                        best = val, bm = q, bk = k;
                }
            st.emplace_back(first, u, bm);
            st.emplace_back(second, bk, w);
        }
    }
    return wrap(std::move(out));
}

// The spanning trees alone, for the probe (tools/cluster_probe.py proto.mst): method 0 Prim, 1 Boruvka (the
// per-point search of linkage), 2 Boruvka by the dual tree.
nb::object mst_py(nb::handle Xo, int method) {
    int64_t n, p;
    const double* X = rows_of(Xo, n, p);
    std::vector<Edge> E;
    {
        const parallel::Unlocked unlocked(true);
        if (method == 0) E = mst_prim(X, n, p);
        else E = mst_boruvka(spatial::kd_build(X, n, p, p, kLeafSize), method == 2);
        std::sort(E.begin(), E.end(), edge_less);
    }
    const int64_t k = static_cast<int64_t>(E.size());
    const int64_t s2[2] = {k, 2}, s1[1] = {k};
    NDArray e = new_array(DType::Int64, s2, false), w = new_array(DType::Float64, s1, false);
    auto* ep = reinterpret_cast<int64_t*>(e.data);
    auto* wp = reinterpret_cast<double*>(w.data);
    for (int64_t i = 0; i < k; ++i) {
        ep[2 * i] = E[static_cast<size_t>(i)].a;
        ep[2 * i + 1] = E[static_cast<size_t>(i)].b;
        wp[i] = E[static_cast<size_t>(i)].w;
    }
    return nb::make_tuple(wrap(std::move(e)), wrap(std::move(w)));
}

}  // namespace

void bind_hierarchy(nb::module_& m) {
    m.def("lex_order", &lex_order_py, nb::arg("X"));
    m.def(
        "linkage", &linkage_py, nb::arg("X").none(), nb::arg("D").none(), nb::arg("order"), nb::arg("method"),
        nb::arg("path")
    );
    m.def("diana", &diana_py, nb::arg("D"), nb::arg("order"));
    m.def("cophenet", &cophenet_py, nb::arg("Z"));
    m.def("optimal_leaf_ordering", &optimal_leaf_ordering_py, nb::arg("Z"), nb::arg("D"), nb::arg("key"));
    m.def("_mst", &mst_py, nb::arg("X"), nb::arg("method"));
}

}  // namespace sd::cluster
