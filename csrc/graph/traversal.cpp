// Traversals, connected components and the topological sort of sd.graph (docs/design/graph.md §2.2, §3.3;
// milestone GR1). The Python layer (src/stardust/graph/_traversal.py) checks every argument.
//
//   _traverse(indptr, indices, data, n, start, directed, depth, tree)
//       the breadth-first (depth false) or depth-first search from `start`, the neighbours of a node in
//       ascending order (directed: its row; undirected: the union of its row and its column). tree false:
//       (order, predecessors), the nodes in the order found and each node's parent (kNoNode for the start
//       and the nodes not reached). tree true: (indptr, indices, data) of the search tree, an edge
//       parent -> child per node found with the weight of the edge it was found by (data, required then).
//   _components(indptr, indices, n, strong)
//       (number of components, labels): weak (the union of the two directions' patterns; also the
//       undirected reading) or strong, each component numbered by the order of its smallest node.
//   _toposort(indptr, indices, n)
//       the lexicographically smallest topological order, or None when the graph has a cycle.
//
// The answers are fixed by the definitions (the neighbour order, the partition, the smallest order), so
// they do not depend on the thread count. Every search is serial (one sweep); the weak components split
// the edges into ranges fixed by the size alone once they are many (a union-find that any order of
// unions leaves with the same partition, whose root is each set's smallest node). A call of
// kParallelWork entries or more runs without the GIL.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using std::int64_t;

nb::object new_vector(DType t, int64_t n) {
    const int64_t shape[] = {n};
    return wrap(new_array(t, shape));
}

int64_t* i64_of(nb::object& o) { return reinterpret_cast<int64_t*>(as_ndarray(o).data); }

// ---- searches ----
//
// The undirected reading of a graph that is not symmetric is g with its transpose t: the neighbours of v
// are the union of row v of g and row v of t, ascending, a node in both once with the undirected_weight of
// the two directions. The rows are never merged element by element (each step would wait on the last
// comparison): the searches skip the nodes already found in each row on its own, and only the few new
// nodes are ordered.

// The nodes found, one bit each (n / 8 bytes: it stays in the first-level cache where a byte per node
// would not, and every entry a search scans tests it).
struct Bits {
    std::vector<uint64_t> w;
    explicit Bits(int64_t n) : w(static_cast<size_t>((n + 63) / 64), 0) {}
    bool test(int64_t i) const { return (w[static_cast<size_t>(i >> 6)] >> (i & 63)) & 1; }
    void set(int64_t i) { w[static_cast<size_t>(i >> 6)] |= uint64_t{1} << (i & 63); }
};

struct Search {
    std::vector<int64_t> order;  // the nodes in the order found
    std::vector<double> weight;  // tree: the weight of the edge each node was found by (by node)
};

// Breadth first: a queue (the order itself). With t, the nodes v finds in its row of g and in its row of
// t are two ascending runs at the end of the queue, merged into one.
void bfs(const View& g, const View* t, int64_t s, int64_t* pred, Bits& seen, Search& out, bool tree) {
    std::vector<int64_t>& q = out.order;
    double* W = tree ? out.weight.data() : nullptr;
    std::vector<int64_t> run;
    q.push_back(s);
    seen.set(s);
    for (size_t head = 0; head < q.size(); ++head) {
        const int64_t v = q[head];
        const size_t qa = q.size();
        for (int64_t k = g.indptr[v], e = g.indptr[v + 1]; k < e; ++k) {
            const int64_t w = g.indices[k];
            if (seen.test(w)) continue;
            seen.set(w);
            pred[w] = v;
            if (W) W[w] = g.data[k];
            q.push_back(w);
        }
        if (!t) continue;
        const size_t qb = q.size();
        for (int64_t k = t->indptr[v], e = t->indptr[v + 1]; k < e; ++k) {
            const int64_t w = t->indices[k];
            if (seen.test(w)) {
                if (W && pred[w] == v) W[w] = undirected_weight(W[w], t->data[k]);  // found from v in g too
                continue;
            }
            seen.set(w);
            pred[w] = v;
            if (W) W[w] = t->data[k];
            q.push_back(w);
        }
        if (qb == qa || q.size() == qb) continue;
        run.assign(q.begin() + static_cast<std::ptrdiff_t>(qa), q.begin() + static_cast<std::ptrdiff_t>(qb));
        std::merge(
            run.begin(), run.end(), q.begin() + static_cast<std::ptrdiff_t>(qb), q.end(),
            q.begin() + static_cast<std::ptrdiff_t>(qa)
        );
    }
}

// Depth first in the order of the recursive search (preorder; at each node the smallest neighbour not yet
// found next): an explicit stack of nodes, each with the position of its next entry in its row and the
// row's end. dfs_one reads g's rows; dfs_two the undirected reading of g with its transpose t: each row
// skips its found nodes on its own and the next node is the smaller of the two heads.
void dfs_one(const View& g, int64_t s, int64_t* pred, Bits& seen, Search& out, bool tree) {
    struct Frame {
        int64_t v, a, ae;
    };
    double* W = tree ? out.weight.data() : nullptr;
    std::vector<Frame> stack;
    stack.push_back({s, g.indptr[s], g.indptr[s + 1]});
    seen.set(s);
    out.order.push_back(s);
    while (!stack.empty()) {
        Frame& f = stack.back();
        int64_t a = f.a;
        const int64_t ae = f.ae;
        while (a < ae && seen.test(g.indices[a])) ++a;
        if (a == ae) {
            stack.pop_back();
            continue;
        }
        const int64_t v = f.v, w = g.indices[a];
        if (W) W[w] = g.data[a];
        f.a = a + 1;
        seen.set(w);
        pred[w] = v;
        out.order.push_back(w);
        stack.push_back({w, g.indptr[w], g.indptr[w + 1]});  // f is not used past this point
    }
}

void dfs_two(const View& g, const View& t, int64_t s, int64_t* pred, Bits& seen, Search& out, bool tree) {
    struct Frame {
        int64_t v, a, ae, b, be;
    };
    double* W = tree ? out.weight.data() : nullptr;
    std::vector<Frame> stack;
    stack.push_back({s, g.indptr[s], g.indptr[s + 1], t.indptr[s], t.indptr[s + 1]});
    seen.set(s);
    out.order.push_back(s);
    while (!stack.empty()) {
        Frame& f = stack.back();
        int64_t a = f.a, b = f.b;
        const int64_t ae = f.ae, be = f.be;
        while (a < ae && seen.test(g.indices[a])) ++a;
        while (b < be && seen.test(t.indices[b])) ++b;
        const int64_t ja = a < ae ? g.indices[a] : INT64_MAX;
        const int64_t jb = b < be ? t.indices[b] : INT64_MAX;
        const int64_t w = std::min(ja, jb);  // the next node: no branch on which row holds it
        if (w == INT64_MAX) {
            stack.pop_back();
            continue;
        }
        const bool in_a = ja == w, in_b = jb == w;
        double wt = 0.0;
        if (W) wt = !in_b ? g.data[a] : !in_a ? t.data[b] : undirected_weight(g.data[a], t.data[b]);
        a += in_a;
        b += in_b;
        const int64_t v = f.v;
        f.a = a;
        f.b = b;
        seen.set(w);
        pred[w] = v;
        if (W) W[w] = wt;
        out.order.push_back(w);
        // f is not used past this point
        stack.push_back({w, g.indptr[w], g.indptr[w + 1], t.indptr[w], t.indptr[w + 1]});
    }
}

void dfs(const View& g, const View* t, int64_t s, int64_t* pred, Bits& seen, Search& out, bool tree) {
    if (t) dfs_two(g, *t, s, pred, seen, out, tree);
    else dfs_one(g, s, pred, seen, out, tree);
}

nb::object traverse(
    nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, int64_t s, bool directed, bool depth, bool tree
) {
    const View g = view_arg(ip, ix, dh, n, "_traverse");
    if (s < 0 || s >= n) throw nb::value_error("_traverse(): wrong start");
    if (tree && !g.data) throw nb::type_error("_traverse(): wrong data");
    const bool unlock = g.nnz() + n >= kParallelWork;
    nb::object op = new_vector(DType::Int64, n);
    int64_t* pred = i64_of(op);
    Search out;
    // The undirected reading: g alone when it is symmetric, else g with its transpose (the pattern's when no
    // weight is read; the indices stand in for the values then).
    bool both = false;
    if (!directed) {
        const parallel::Unlocked unlocked(unlock);
        both = !symmetric(g, tree);
    }
    std::optional<Transposed> t;
    if (both) t.emplace(transpose(g, as_ndarray(ip), as_ndarray(ix), as_ndarray(tree ? dh : ix), tree));
    {
        const parallel::Unlocked unlocked(unlock);
        const View* tv = t ? &t->view : nullptr;
        std::fill(pred, pred + n, kNoNode);
        Bits seen(n);
        out.order.reserve(static_cast<size_t>(n));
        if (tree) out.weight.assign(static_cast<size_t>(n), 0.0);
        if (depth) dfs(g, tv, s, pred, seen, out, tree);
        else bfs(g, tv, s, pred, seen, out, tree);
    }
    const int64_t found = static_cast<int64_t>(out.order.size());
    if (!tree) {
        nb::object oo = new_vector(DType::Int64, found);
        std::memcpy(i64_of(oo), out.order.data(), static_cast<size_t>(found) * sizeof(int64_t));
        return nb::make_tuple(oo, op);
    }
    // The tree: row p holds p's children, ascending (nodes are placed in ascending order).
    nb::object tp = new_vector(DType::Int64, n + 1), ti = new_vector(DType::Int64, found - 1),
               td = new_vector(DType::Float64, found - 1);
    int64_t* P = i64_of(tp);
    int64_t* I = i64_of(ti);
    double* D = reinterpret_cast<double*>(as_ndarray(td).data);
    {
        const parallel::Unlocked unlocked(unlock);
        std::fill(P, P + n + 1, 0);
        for (int64_t v = 0; v < n; ++v)
            if (pred[v] >= 0) ++P[pred[v] + 1];
        for (int64_t i = 0; i < n; ++i) P[i + 1] += P[i];
        std::vector<int64_t> next(P, P + n);
        for (int64_t v = 0; v < n; ++v) {
            if (pred[v] < 0) continue;
            const int64_t dst = next[static_cast<size_t>(pred[v])]++;
            I[dst] = v;
            D[dst] = out.weight[static_cast<size_t>(v)];
        }
    }
    return nb::make_tuple(tp, ti, td);
}

// ---- connected components ----

// Union-find whose root is the smallest node of its set: a union links the larger root under the smaller.
// find halves the path.
int64_t find(int64_t* parent, int64_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

void unite_serial(const View& g, int64_t* parent) {
    for (int64_t i = 0; i < g.n; ++i) {
        for (int64_t k = g.indptr[i], e = g.indptr[i + 1]; k < e; ++k) {
            int64_t a = find(parent, i), b = find(parent, g.indices[k]);
            if (a == b) continue;
            if (a < b) parent[b] = a;
            else parent[a] = b;
        }
    }
}

// The same partition by tasks over ranges of rows (about kUniteTask entries each): a root is linked under
// a smaller one by compare-and-swap, so a root's parent only ever moves to a smaller node and every
// pointer stays inside its set. The partition the edges define is unique, so any interleaving ends with it.
constexpr int64_t kUniteTask = 1 << 16;    // stored entries per task
constexpr int64_t kUniteSplit = 1 << 18;  // from this many stored entries the unions split (measured, GR1)

int64_t find_shared(int64_t* parent, int64_t x) {
    for (;;) {
        const int64_t p = std::atomic_ref<int64_t>(parent[x]).load(std::memory_order_relaxed);
        if (p == x) return x;
        const int64_t gp = std::atomic_ref<int64_t>(parent[p]).load(std::memory_order_relaxed);
        if (gp != p) std::atomic_ref<int64_t>(parent[x]).store(gp, std::memory_order_relaxed);
        x = gp;
    }
}

void unite_shared(int64_t* parent, int64_t a, int64_t b) {
    for (;;) {
        a = find_shared(parent, a);
        b = find_shared(parent, b);
        if (a == b) return;
        if (a < b) std::swap(a, b);  // link the larger root a under b
        int64_t expected = a;
        if (std::atomic_ref<int64_t>(parent[a]).compare_exchange_weak(
                expected, b, std::memory_order_relaxed, std::memory_order_relaxed
            ))
            return;
    }
}

void unite_parallel(const View& g, int64_t* parent) {
    // row ranges with about kUniteTask entries each, found by size alone
    std::vector<int64_t> bounds{0};
    const int64_t nnz = g.nnz();
    for (int64_t t = 1; t * kUniteTask < nnz; ++t) {
        const int64_t r = std::upper_bound(g.indptr, g.indptr + g.n + 1, t * kUniteTask) - g.indptr - 1;
        if (r > bounds.back()) bounds.push_back(r);
    }
    bounds.push_back(g.n);
    const int64_t ntasks = static_cast<int64_t>(bounds.size()) - 1;
    parallel::parallel_for(ntasks, [&](int64_t t) {
        for (int64_t i = bounds[static_cast<size_t>(t)]; i < bounds[static_cast<size_t>(t + 1)]; ++i)
            for (int64_t k = g.indptr[i], e = g.indptr[i + 1]; k < e; ++k) unite_shared(parent, i, g.indices[k]);
    });
}

int64_t weak_labels(const View& g, int64_t* label) {
    const int64_t n = g.n;
    std::vector<int64_t> parent(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) parent[static_cast<size_t>(i)] = i;
    if (g.nnz() >= kUniteSplit && parallel::available()) unite_parallel(g, parent.data());
    else unite_serial(g, parent.data());
    // The root of a set is its smallest node, met first in ascending order.
    int64_t c = 0;
    for (int64_t v = 0; v < n; ++v) {
        const int64_t r = find(parent.data(), v);
        label[v] = r == v ? c++ : label[r];
    }
    return c;
}

// Tarjan's strongly connected components, iterative: per node its discovery index and low link, the
// position of its next edge, and the stack of nodes not yet assigned. Each component gets a temporary
// number (the order it closes); the labels are renumbered by the smallest node afterwards.
int64_t strong_labels(const View& g, int64_t* label) {
    const int64_t n = g.n;
    constexpr int64_t kNone = -1;
    std::vector<int64_t> index(static_cast<size_t>(n), kNone), low(static_cast<size_t>(n)), pos(static_cast<size_t>(n));
    std::vector<int64_t> comp(static_cast<size_t>(n), kNone);
    std::vector<int64_t> calls, stack;
    int64_t counter = 0, ncomp = 0;
    for (int64_t s = 0; s < n; ++s) {
        if (index[static_cast<size_t>(s)] != kNone) continue;
        index[static_cast<size_t>(s)] = low[static_cast<size_t>(s)] = counter++;
        pos[static_cast<size_t>(s)] = g.indptr[s];
        stack.push_back(s);
        calls.push_back(s);
        while (!calls.empty()) {
            const int64_t v = calls.back();
            const size_t vi = static_cast<size_t>(v);
            int64_t& k = pos[vi];
            const int64_t e = g.indptr[v + 1];
            bool descended = false;
            while (k < e) {
                const int64_t w = g.indices[k++];
                const size_t wi = static_cast<size_t>(w);
                if (index[wi] == kNone) {
                    index[wi] = low[wi] = counter++;
                    pos[wi] = g.indptr[w];
                    stack.push_back(w);
                    calls.push_back(w);
                    descended = true;
                    break;
                }
                if (comp[wi] == kNone && index[wi] < low[vi]) low[vi] = index[wi];  // w on the stack
            }
            if (descended) continue;
            calls.pop_back();
            if (low[vi] == index[vi]) {
                int64_t w;
                do {
                    w = stack.back();
                    stack.pop_back();
                    comp[static_cast<size_t>(w)] = ncomp;
                } while (w != v);
                ++ncomp;
            }
            if (!calls.empty()) {
                const size_t ui = static_cast<size_t>(calls.back());
                if (low[vi] < low[ui]) low[ui] = low[vi];
            }
        }
    }
    std::vector<int64_t> renum(static_cast<size_t>(ncomp), kNone);
    int64_t c = 0;
    for (int64_t v = 0; v < n; ++v) {
        int64_t& r = renum[static_cast<size_t>(comp[static_cast<size_t>(v)])];
        if (r == kNone) r = c++;
        label[v] = r;
    }
    return c;
}

nb::object components(nb::handle ip, nb::handle ix, int64_t n, bool strong) {
    const View g = view_arg(ip, ix, nb::none(), n, "_components");
    nb::object ol = new_vector(DType::Int64, n);
    int64_t* label = i64_of(ol);
    int64_t c;
    {
        const parallel::Unlocked unlocked(g.nnz() + n >= kParallelWork);
        c = strong ? strong_labels(g, label) : weak_labels(g, label);
    }
    return nb::make_tuple(c, ol);
}

// ---- topological sort ----

// The set of ready nodes as a bitmap of levels (each bit of a level says whether the 64-bit word below it
// has a bit set): insert and remove-the-smallest in a few word operations for any n.
class MinSet {
public:
    explicit MinSet(int64_t n) {
        int64_t len = n;
        do {
            len = (len + 63) / 64;
            levels_.emplace_back(static_cast<size_t>(len), 0);
        } while (len > 1);
    }
    void insert(int64_t v) {
        for (auto& lv : levels_) {
            const uint64_t bit = uint64_t{1} << (v & 63);
            v >>= 6;
            const bool was = lv[static_cast<size_t>(v)] != 0;
            lv[static_cast<size_t>(v)] |= bit;
            if (was) return;
        }
    }
    bool empty() const { return levels_.back()[0] == 0; }
    int64_t pop_min() {
        int64_t v = 0;
        for (size_t l = levels_.size(); l-- > 0;) v = (v << 6) | std::countr_zero(levels_[l][static_cast<size_t>(v)]);
        const int64_t out = v;
        for (auto& lv : levels_) {
            uint64_t& w = lv[static_cast<size_t>(v >> 6)];
            w &= ~(uint64_t{1} << (v & 63));
            if (w) break;
            v >>= 6;
        }
        return out;
    }

private:
    std::vector<std::vector<uint64_t>> levels_;
};

// Kahn's algorithm taking the smallest ready node each time: the lexicographically smallest order. Nodes
// left over (a cycle, self loops included) make it fail.
nb::object toposort(nb::handle ip, nb::handle ix, int64_t n) {
    const View g = view_arg(ip, ix, nb::none(), n, "_toposort");
    nb::object oo = new_vector(DType::Int64, n);
    int64_t* order = i64_of(oo);
    int64_t done = 0;
    {
        const parallel::Unlocked unlocked(g.nnz() + n >= kParallelWork);
        std::vector<int64_t> indeg(static_cast<size_t>(n), 0);
        for (int64_t k = 0, e = g.nnz(); k < e; ++k) ++indeg[static_cast<size_t>(g.indices[k])];
        if (n > 0) {
            MinSet ready(n);
            for (int64_t v = 0; v < n; ++v)
                if (indeg[static_cast<size_t>(v)] == 0) ready.insert(v);
            while (!ready.empty()) {
                const int64_t v = ready.pop_min();
                order[done++] = v;
                for (int64_t k = g.indptr[v], e = g.indptr[v + 1]; k < e; ++k)
                    if (--indeg[static_cast<size_t>(g.indices[k])] == 0) ready.insert(g.indices[k]);
            }
        }
    }
    if (done < n) return nb::none();
    return oo;
}

}  // namespace

void bind_traversal(nb::module_& m) {
    m.def(
        "_traverse", &traverse, nb::arg("indptr"), nb::arg("indices"), nb::arg("data").none(), nb::arg("n"),
        nb::arg("start"), nb::arg("directed"), nb::arg("depth"), nb::arg("tree")
    );
    m.def("_components", &components, nb::arg("indptr"), nb::arg("indices"), nb::arg("n"), nb::arg("strong"));
    m.def("_toposort", &toposort, nb::arg("indptr"), nb::arg("indices"), nb::arg("n"));
}

}  // namespace sd::graph
