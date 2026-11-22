// The articulation points, bridges, biconnected components and core numbers of sd.graph
// (docs/design/graph.md §2.6, §3.7; milestone GR5). The Python layer (src/stardust/graph/_structure.py)
// checks every argument.
//
//   _biconnected(indptr, indices, n, want)
//       want 0: the articulation points (int64, ascending); 1: the bridges ((k, 2) int64, rows (i, j) with
//       i < j in lexicographic order); 2: (number of components, edges, labels): every edge as a row (i, j),
//       i < j, in lexicographic order, and its biconnected component, numbered by the order of each
//       component's smallest edge.
//   _core_number(indptr, indices, n)
//       the core number of every node (int64 (n,)).
//
// The four read the graph undirected: the union of the two directions' patterns with the diagonal left out
// (simple()). Their answers are sets and a partition, each unique, so the search order changes nothing:
// one iterative depth-first search (Hopcroft-Tarjan's low points with a stack of edges) and
// Batagelj-Zaversnik's bucket peeling, both O(n + E) and serial. A call of kParallelWork entries or more
// runs without the GIL.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using std::int64_t;

// The undirected pattern of a graph without its diagonal, canonical (rows ascending, each neighbour once):
// row v is the union of row v and column v of g. A symmetric g without a stored diagonal is read in place;
// otherwise the rows are made here.
struct Simple {
    int64_t n = 0;
    const int64_t* ip = nullptr;
    const int64_t* ix = nullptr;
    std::vector<int64_t> own_ip, own_ix;
    int64_t nnz() const { return ip[n]; }
};

Simple simple(const View& g) {
    const int64_t n = g.n, nnz = g.nnz();
    Simple s;
    s.n = n;
    if (symmetric(g, false, true)) {
        s.ip = g.indptr;
        s.ix = g.indices;
        return s;
    }
    std::vector<int64_t> tp(static_cast<size_t>(n + 1), 0), ti(static_cast<size_t>(nnz));
    for (int64_t k = 0; k < nnz; ++k) ++tp[static_cast<size_t>(g.indices[k] + 1)];
    for (int64_t i = 0; i < n; ++i) tp[static_cast<size_t>(i + 1)] += tp[static_cast<size_t>(i)];
    {
        std::vector<int64_t> next(tp.begin(), tp.end() - 1);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t k = g.indptr[i]; k < g.indptr[i + 1]; ++k)
                ti[static_cast<size_t>(next[static_cast<size_t>(g.indices[k])]++)] = i;
    }
    s.own_ip.resize(static_cast<size_t>(n + 1));
    s.own_ix.reserve(static_cast<size_t>(2 * nnz));
    s.own_ip[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t* a = g.indices + g.indptr[i];
        const int64_t* ae = g.indices + g.indptr[i + 1];
        const int64_t* b = ti.data() + tp[static_cast<size_t>(i)];
        const int64_t* be = ti.data() + tp[static_cast<size_t>(i + 1)];
        while (a < ae || b < be) {
            int64_t j;
            if (b == be || (a < ae && *a < *b)) j = *a++;
            else if (a == ae || *b < *a) j = *b++;
            else {
                j = *a++;
                ++b;
            }
            if (j != i) s.own_ix.push_back(j);
        }
        s.own_ip[static_cast<size_t>(i + 1)] = static_cast<int64_t>(s.own_ix.size());
    }
    s.ip = s.own_ip.data();
    s.ix = s.own_ix.data();
    return s;
}

// The edges of a Simple: edge e is the e-th entry (i, j) with i < j in row order, so the ids follow the
// lexicographic order of the pairs; id[k] is the edge of entry k (both directions of an edge share it).
struct Edges {
    std::vector<int64_t> id, lo, hi;
    int64_t count() const { return static_cast<int64_t>(lo.size()); }
};

Edges edges_of(const Simple& s) {
    // The entry (v, u) mirroring (u, v) is the next unseen entry of row v when the rows are scanned in
    // order: row v holds its columns ascending, and the rows u that hold v come in that same order.
    Edges e;
    e.id.resize(static_cast<size_t>(s.nnz()));
    e.lo.reserve(static_cast<size_t>(s.nnz() / 2));
    e.hi.reserve(static_cast<size_t>(s.nnz() / 2));
    std::vector<int64_t> cursor(s.ip, s.ip + s.n);
    for (int64_t u = 0; u < s.n; ++u) {
        for (int64_t k = s.ip[u]; k < s.ip[u + 1]; ++k) {
            const int64_t v = s.ix[k];
            const int64_t m = cursor[static_cast<size_t>(v)]++;
            if (v > u) {
                const int64_t id = e.count();
                e.lo.push_back(u);
                e.hi.push_back(v);
                e.id[static_cast<size_t>(k)] = id;
                e.id[static_cast<size_t>(m)] = id;
            }
        }
    }
    return e;
}

// One iterative depth-first search over every component (roots ascending, neighbours ascending) with
// Hopcroft and Tarjan's low points: low(v) is the smallest discovery time reachable from v's subtree by
// one back edge. A tree edge p - v closes a biconnected component when low(v) >= disc(p) (the edges
// pushed since it, it included, form the component), is a bridge when low(v) > disc(p), and makes p an
// articulation point (a root only with two children or more). The edges and their stack only when e is
// given (the components); the tree edge back to the parent is skipped by its node (no parallel edges).
struct Blocks {
    std::vector<uint8_t> articulation;  // by node
    std::vector<int64_t> bridges;       // (lo, hi) pairs, in the order found
    std::vector<int64_t> component;     // by edge (e given), in the order the search closes them
    int64_t count = 0;
};

Blocks blocks(const Simple& s, const Edges* e) {
    const int64_t n = s.n;
    Blocks out;
    out.articulation.assign(static_cast<size_t>(n), 0);
    if (e) out.component.assign(static_cast<size_t>(e->count()), -1);
    std::vector<int64_t> disc(static_cast<size_t>(n), -1), low(static_cast<size_t>(n)), next(static_cast<size_t>(n)),
        up(static_cast<size_t>(n)), stack, edges, up_edge(e ? static_cast<size_t>(n) : 0);
    stack.reserve(static_cast<size_t>(n));
    if (e) edges.reserve(static_cast<size_t>(e->count()));
    int64_t time = 0;
    for (int64_t r = 0; r < n; ++r) {
        if (disc[static_cast<size_t>(r)] >= 0) continue;
        disc[static_cast<size_t>(r)] = low[static_cast<size_t>(r)] = time++;
        next[static_cast<size_t>(r)] = s.ip[r];
        up[static_cast<size_t>(r)] = -1;
        stack.push_back(r);
        int64_t children = 0;
        while (!stack.empty()) {
            const int64_t u = stack.back();
            int64_t& k = next[static_cast<size_t>(u)];
            if (k < s.ip[u + 1]) {
                const int64_t v = s.ix[k], at = k++;
                if (v == up[static_cast<size_t>(u)]) continue;  // the tree edge back to the parent
                const int64_t dv = disc[static_cast<size_t>(v)];
                if (dv < 0) {
                    if (e) {
                        edges.push_back(e->id[static_cast<size_t>(at)]);
                        up_edge[static_cast<size_t>(v)] = e->id[static_cast<size_t>(at)];
                    }
                    up[static_cast<size_t>(v)] = u;
                    disc[static_cast<size_t>(v)] = low[static_cast<size_t>(v)] = time++;
                    next[static_cast<size_t>(v)] = s.ip[v];
                    stack.push_back(v);
                    if (u == r) ++children;
                } else if (dv < disc[static_cast<size_t>(u)]) {  // a back edge
                    if (e) edges.push_back(e->id[static_cast<size_t>(at)]);
                    low[static_cast<size_t>(u)] = std::min(low[static_cast<size_t>(u)], dv);
                }
                continue;
            }
            stack.pop_back();
            if (stack.empty()) break;
            const int64_t p = stack.back();
            low[static_cast<size_t>(p)] = std::min(low[static_cast<size_t>(p)], low[static_cast<size_t>(u)]);
            if (low[static_cast<size_t>(u)] >= disc[static_cast<size_t>(p)]) {
                if (p != r) out.articulation[static_cast<size_t>(p)] = 1;
                if (low[static_cast<size_t>(u)] > disc[static_cast<size_t>(p)]) {
                    out.bridges.push_back(std::min(p, u));
                    out.bridges.push_back(std::max(p, u));
                }
                if (e) {  // the edges pushed since the tree edge p - u, it included
                    const int64_t tree = up_edge[static_cast<size_t>(u)];
                    int64_t f;
                    do {
                        f = edges.back();
                        edges.pop_back();
                        out.component[static_cast<size_t>(f)] = out.count;
                    } while (f != tree);
                    ++out.count;
                }
            }
        }
        if (children >= 2) out.articulation[static_cast<size_t>(r)] = 1;
    }
    return out;
}

nb::object int64_array(std::initializer_list<int64_t> shape) {
    return wrap(new_array(DType::Int64, std::span<const int64_t>(shape.begin(), shape.size())));
}

int64_t* data_of(nb::object& o) { return reinterpret_cast<int64_t*>(as_ndarray(o).data); }

nb::object biconnected(nb::handle ip, nb::handle ix, int64_t n, int want) {
    const View g = view_arg(ip, ix, nb::none(), n, "_biconnected");
    if (want < 0 || want > 2) throw nb::value_error("_biconnected(): wrong want");
    Edges e;
    Blocks b;
    std::vector<int64_t> picked;  // want 0: the nodes; want 1: the bridges' pairs, sorted
    std::vector<int64_t> label;   // want 2: the components renumbered by their smallest edge
    {
        const parallel::Unlocked unlocked(n + g.nnz() >= kParallelWork);
        const Simple s = simple(g);
        if (want == 2) e = edges_of(s);
        b = blocks(s, want == 2 ? &e : nullptr);
        if (want == 0) {
            for (int64_t v = 0; v < n; ++v)
                if (b.articulation[static_cast<size_t>(v)]) picked.push_back(v);
        } else if (want == 1) {
            const size_t m = b.bridges.size() / 2;
            std::vector<std::pair<int64_t, int64_t>> rows(m);
            for (size_t i = 0; i < m; ++i) rows[i] = {b.bridges[2 * i], b.bridges[2 * i + 1]};
            std::sort(rows.begin(), rows.end());
            for (const auto& [lo, hi] : rows) {
                picked.push_back(lo);
                picked.push_back(hi);
            }
        } else {
            std::vector<int64_t> renumber(static_cast<size_t>(b.count), -1);
            label.resize(static_cast<size_t>(e.count()));
            int64_t c = 0;
            for (int64_t f = 0; f < e.count(); ++f) {
                int64_t& r = renumber[static_cast<size_t>(b.component[static_cast<size_t>(f)])];
                if (r < 0) r = c++;
                label[static_cast<size_t>(f)] = r;
            }
        }
    }
    const int64_t k = static_cast<int64_t>(picked.size());
    if (want == 0) {
        nb::object out = int64_array({k});
        if (k) std::memcpy(data_of(out), picked.data(), static_cast<size_t>(k) * sizeof(int64_t));
        return out;
    }
    if (want == 1) {
        nb::object out = int64_array({k / 2, 2});
        if (k) std::memcpy(data_of(out), picked.data(), static_cast<size_t>(k) * sizeof(int64_t));
        return out;
    }
    nb::object edges = int64_array({e.count(), 2});
    nb::object labels = int64_array({e.count()});
    int64_t* p = data_of(edges);
    int64_t* l = data_of(labels);
    {
        const parallel::Unlocked unlocked(e.count() >= kParallelWork);
        for (int64_t f = 0; f < e.count(); ++f) {
            p[2 * f] = e.lo[static_cast<size_t>(f)];
            p[2 * f + 1] = e.hi[static_cast<size_t>(f)];
        }
        if (e.count()) std::memcpy(l, label.data(), static_cast<size_t>(e.count()) * sizeof(int64_t));
    }
    return nb::make_tuple(b.count, edges, labels);
}

// Batagelj and Zaversnik's O(n + E) peeling: the nodes in buckets by current degree, taken in increasing
// degree; taking v lowers the degree of each neighbour of larger degree by one (moved to the front of
// its bucket). The degree a node has when taken is its core number.
nb::object core_number(nb::handle ip, nb::handle ix, int64_t n) {
    const View g = view_arg(ip, ix, nb::none(), n, "_core_number");
    nb::object out = int64_array({n});
    int64_t* deg = data_of(out);
    {
        const parallel::Unlocked unlocked(n + g.nnz() >= kParallelWork);
        const Simple s = simple(g);
        int64_t top = 0;
        for (int64_t v = 0; v < n; ++v) {
            deg[v] = s.ip[v + 1] - s.ip[v];
            top = std::max(top, deg[v]);
        }
        std::vector<int64_t> bin(static_cast<size_t>(top) + 2, 0), pos(static_cast<size_t>(n)),
            vert(static_cast<size_t>(n));
        for (int64_t v = 0; v < n; ++v) ++bin[static_cast<size_t>(deg[v]) + 1];
        for (size_t d = 1; d < bin.size(); ++d) bin[d] += bin[d - 1];  // bin[d]: where degree d starts
        for (int64_t v = 0; v < n; ++v) {
            pos[static_cast<size_t>(v)] = bin[static_cast<size_t>(deg[v])]++;
            vert[static_cast<size_t>(pos[static_cast<size_t>(v)])] = v;
        }
        for (size_t d = bin.size() - 1; d > 0; --d) bin[d] = bin[d - 1];
        bin[0] = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t v = vert[static_cast<size_t>(i)];
            for (int64_t k = s.ip[v]; k < s.ip[v + 1]; ++k) {
                const int64_t u = s.ix[k];
                if (deg[u] > deg[v]) {
                    const int64_t du = deg[u], pu = pos[static_cast<size_t>(u)];
                    const int64_t pw = bin[static_cast<size_t>(du)], w = vert[static_cast<size_t>(pw)];
                    if (u != w) {
                        pos[static_cast<size_t>(u)] = pw;
                        vert[static_cast<size_t>(pu)] = w;
                        pos[static_cast<size_t>(w)] = pu;
                        vert[static_cast<size_t>(pw)] = u;
                    }
                    ++bin[static_cast<size_t>(du)];
                    --deg[u];
                }
            }
        }
    }
    return out;
}

}  // namespace

void bind_structure(nb::module_& m) {
    m.def("_biconnected", &biconnected, nb::arg("indptr"), nb::arg("indices"), nb::arg("n"), nb::arg("want"));
    m.def("_core_number", &core_number, nb::arg("indptr"), nb::arg("indices"), nb::arg("n"));
}

}  // namespace sd::graph
