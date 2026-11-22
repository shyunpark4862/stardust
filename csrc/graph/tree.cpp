// The minimum spanning forest (docs/design/graph.md §2.4, §3.4; milestone GR3).
//
//   _mst(indptr, indices, data, n) -> (indptr, indices, data) of the tree
//       The minimum spanning forest of the undirected reading of a graph whose stored weights are finite
//       (the Python layer checks them): every undirected edge {i, j}, i != j, weighs the smaller of the two
//       directions stored (undirected_weight), and the edges are compared by the total order of the key
//       (w, min(i, j), max(i, j)). Under a total order the minimum spanning forest is unique (the theorem
//       for distinct weights), so the answer does not depend on the algorithm: Kruskal on the edges sorted
//       by that key. A chosen edge keeps the place of the stored entry that gave its weight (both
//       directions of the same bits: the upper triangle, i < j), so the result is the input's entries
//       filtered: canonical CSR, the stored values as they are (-0.0 stays -0.0).
//
// The sort is a stable LSD radix sort of (key of w, edge number) with 11-bit digits, the edges numbered in
// (min(i, j), max(i, j)) order: stability makes ties of w fall in that order, the key's own. A digit that
// every key shares is skipped. -0.0 and 0.0 sort as equal (the key of w + 0.0), as the order compares w
// by value. Serial: one pass per digit over the edges (decision GR43 records Borůvka's measurement).
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using std::int64_t;
using std::uint64_t;

nb::object vec1(DType t, int64_t n) {
    const int64_t shape[] = {n};
    return wrap(new_array(t, shape));
}

template <class T>
T* ptr(nb::object& o) {
    return reinterpret_cast<T*>(as_ndarray(o).data);
}

// The order of doubles as unsigned integers (-0.0 read as 0.0; no NaN reaches here).
inline uint64_t order_key(double w) {
    const uint64_t b = std::bit_cast<uint64_t>(w + 0.0);
    return (b >> 63) ? ~b : (b | (uint64_t{1} << 63));
}

struct Item {
    uint64_t key;
    int64_t edge;
};

// A stable sort of items by key (ties keep their order).
void radix_sort(std::vector<Item>& a) {
    const size_t m = a.size();
    if (m < 2) return;
    if (m <= 256) {
        std::stable_sort(a.begin(), a.end(), [](const Item& x, const Item& y) { return x.key < y.key; });
        return;
    }
    constexpr int kBits = 11;
    constexpr uint64_t kMask = (uint64_t{1} << kBits) - 1;
    uint64_t all_and = ~uint64_t{0}, all_or = 0;
    for (const Item& it : a) {
        all_and &= it.key;
        all_or |= it.key;
    }
    const uint64_t varying = all_and ^ all_or;  // the bits not every key shares
    std::vector<Item> b(m);
    std::vector<int64_t> count(size_t{1} << kBits);
    for (int shift = 0; shift < 64; shift += kBits) {
        if (((varying >> shift) & kMask) == 0) continue;
        std::fill(count.begin(), count.end(), 0);
        for (const Item& it : a) ++count[(it.key >> shift) & kMask];
        int64_t sum = 0;
        for (int64_t& c : count) {
            const int64_t t = c;
            c = sum;
            sum += t;
        }
        for (const Item& it : a) b[static_cast<size_t>(count[(it.key >> shift) & kMask]++)] = it;
        a.swap(b);
    }
}

int64_t find(std::vector<int64_t>& parent, int64_t x) {
    while (parent[static_cast<size_t>(x)] != x) {
        const int64_t g = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
        parent[static_cast<size_t>(x)] = g;  // path halving
        x = g;
    }
    return x;
}

// mark[k] = 1 for the stored entries k of the forest's edges.
void spanning_forest(const View& g, std::vector<uint8_t>& mark) {
    const int64_t n = g.n, nnz = g.nnz();
    // The lower triangle (stored (r, c), r > c) by column: row c of `low` lists the r ascending, with the
    // entry's number, so row i meets the edges {i, j}, j > i, stored as (j, i).
    std::vector<int64_t> lp(static_cast<size_t>(n + 1), 0);
    for (int64_t r = 0; r < n; ++r)
        for (int64_t k = g.indptr[r]; k < g.indptr[r + 1] && g.indices[k] < r; ++k)
            ++lp[static_cast<size_t>(g.indices[k] + 1)];
    for (int64_t i = 0; i < n; ++i) lp[static_cast<size_t>(i + 1)] += lp[static_cast<size_t>(i)];
    std::vector<int64_t> low(static_cast<size_t>(lp[static_cast<size_t>(n)])), lowr(low.size());
    {
        std::vector<int64_t> next(lp.begin(), lp.end() - 1);
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t k = g.indptr[r]; k < g.indptr[r + 1] && g.indices[k] < r; ++k) {
                const size_t at = static_cast<size_t>(next[static_cast<size_t>(g.indices[k])]++);
                low[at] = k;
                lowr[at] = r;
            }
        }
    }
    // The undirected edges in (min, max) order: endpoints, the entry that gives the weight, the key.
    std::vector<int64_t> ea, eb, ek;
    std::vector<Item> items;
    ea.reserve(static_cast<size_t>(nnz));
    eb.reserve(static_cast<size_t>(nnz));
    ek.reserve(static_cast<size_t>(nnz));
    items.reserve(static_cast<size_t>(nnz));
    auto add = [&](int64_t i, int64_t j, int64_t k, double w) {
        items.push_back(Item{order_key(w), static_cast<int64_t>(ek.size())});
        ea.push_back(i);
        eb.push_back(j);
        ek.push_back(k);
    };
    for (int64_t i = 0; i < n; ++i) {
        int64_t a = g.indptr[i];
        const int64_t ae = g.indptr[i + 1];
        while (a < ae && g.indices[a] <= i) ++a;  // the upper part of row i: (i, j), j > i
        int64_t b = lp[static_cast<size_t>(i)];
        const int64_t be = lp[static_cast<size_t>(i + 1)];
        while (a < ae || b < be) {
            const int64_t kb = b < be ? low[static_cast<size_t>(b)] : -1;
            const int64_t ja = a < ae ? g.indices[a] : INT64_MAX;
            const int64_t jb = b < be ? lowr[static_cast<size_t>(b)] : INT64_MAX;
            if (ja < jb) {
                add(i, ja, a, g.data[a]);
                ++a;
            } else if (jb < ja) {
                add(i, jb, kb, g.data[kb]);
                ++b;
            } else {
                const double wu = g.data[a], wl = g.data[kb], w = undirected_weight(wu, wl);
                const bool upper = std::memcmp(&wu, &w, sizeof(double)) == 0;
                add(i, ja, upper ? a : kb, w);
                ++a;
                ++b;
            }
        }
    }
    radix_sort(items);
    std::vector<int64_t> parent(static_cast<size_t>(n));
    std::iota(parent.begin(), parent.end(), int64_t{0});
    int64_t chosen = 0;
    for (const Item& it : items) {
        const size_t e = static_cast<size_t>(it.edge);
        const int64_t ra = find(parent, ea[e]), rb = find(parent, eb[e]);
        if (ra == rb) continue;
        parent[static_cast<size_t>(std::max(ra, rb))] = std::min(ra, rb);
        mark[static_cast<size_t>(ek[e])] = 1;
        if (++chosen == n - 1) break;
    }
}

nb::object mst(nb::handle ip, nb::handle ix, nb::handle dh, int64_t n) {
    const View g = view_arg(ip, ix, dh, n, "_mst");
    if (!g.data) throw nb::type_error("_mst(): wrong data");
    const int64_t nnz = g.nnz();
    std::vector<uint8_t> mark(static_cast<size_t>(nnz), 0);
    {
        const parallel::Unlocked unlocked(nnz >= kParallelWork);
        spanning_forest(g, mark);
    }
    int64_t m = 0;
    for (uint8_t x : mark) m += x;
    nb::object op = vec1(DType::Int64, n + 1), oi = vec1(DType::Int64, m), od = vec1(DType::Float64, m);
    int64_t* P = ptr<int64_t>(op);
    int64_t* I = ptr<int64_t>(oi);
    double* D = ptr<double>(od);
    {
        const parallel::Unlocked unlocked(nnz >= kParallelWork);
        int64_t t = 0;
        P[0] = 0;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t k = g.indptr[i]; k < g.indptr[i + 1]; ++k) {
                if (!mark[static_cast<size_t>(k)]) continue;
                I[t] = g.indices[k];
                D[t] = g.data[k];
                ++t;
            }
            P[i + 1] = t;
        }
    }
    return nb::make_tuple(op, oi, od);
}

}  // namespace

void bind_tree(nb::module_& m) {
    m.def("_mst", &mst, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("n"));
}

}  // namespace sd::graph
