// The maximum flow (docs/design/graph.md §2.4, §3.5; milestone GR3).
//
//   _max_flow(indptr, indices, caps, n, source, sink, dinic)
//       -> (flow_value, flow_indptr, flow_indices, flow_data) or None when the capacities out of the source
//       (the diagonal left out) sum past int64. caps: the int64 capacities of the stored entries, none
//       negative (the Python layer checks them and turns bool into int64).
//
// The residual graph is a CSR of the pattern of G and its transpose (a stored (u, v) or (v, u) gives the
// entry (u, v); the diagonal entries stay, with capacity 0, as scipy's flow keeps them), each entry with
// the place of its reverse. Its rows are ascending, so every search looks at the neighbours in ascending
// order and the flow returned depends on the data alone (decision GR16):
//   - Edmonds-Karp: a breadth-first search from the source that stops when it reaches the sink, then the
//     path of the predecessors is augmented by its bottleneck.
//   - Dinic: the breadth-first levels from the source, then a depth-first search on the current arcs (the
//     smallest neighbour first) augments one path at a time, retreating from dead ends, until no path
//     is left in the level graph.
// The flow f is antisymmetric (f(u, v) = -f(v, u)) and returned on the residual pattern, as scipy's. A
// residual capacity c - f can reach c(u, v) + c(v, u) and pass int64, so the residual capacities are kept
// in uint64 (exact: 0 <= c - f < 2^64) and f = c - (c - f) is read back at the end; a bottleneck is at
// most what is left out of the source, so every flow stays in int64.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cstdint>
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

struct Residual {
    int64_t n = 0;
    std::vector<int64_t> ip, ix, rev, cap;
    std::vector<uint64_t> res;  // the residual capacities c - f
    uint64_t left(int64_t k) const { return res[static_cast<size_t>(k)]; }
    void push(int64_t k, uint64_t d) {
        res[static_cast<size_t>(k)] -= d;
        res[static_cast<size_t>(rev[static_cast<size_t>(k)])] += d;
    }
    int64_t flow(size_t k) const { return static_cast<int64_t>(static_cast<uint64_t>(cap[k]) - res[k]); }
};

// The residual CSR of g (caps: g's capacities) into r.
void residual(const View& g, const int64_t* caps, Residual& r) {
    const int64_t n = g.n, nnz = g.nnz();
    std::vector<int64_t> tp(static_cast<size_t>(n + 1), 0), ti(static_cast<size_t>(nnz));
    for (int64_t k = 0; k < nnz; ++k) ++tp[static_cast<size_t>(g.indices[k] + 1)];
    for (int64_t i = 0; i < n; ++i) tp[static_cast<size_t>(i + 1)] += tp[static_cast<size_t>(i)];
    {
        std::vector<int64_t> next(tp.begin(), tp.end() - 1);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t k = g.indptr[i]; k < g.indptr[i + 1]; ++k)
                ti[static_cast<size_t>(next[static_cast<size_t>(g.indices[k])]++)] = i;
    }
    r.n = n;
    r.ip.assign(static_cast<size_t>(n + 1), 0);
    r.ix.clear();
    r.cap.clear();
    r.ix.reserve(static_cast<size_t>(2 * nnz));
    r.cap.reserve(static_cast<size_t>(2 * nnz));
    for (int64_t i = 0; i < n; ++i) {
        int64_t a = g.indptr[i], b = tp[static_cast<size_t>(i)];
        const int64_t ae = g.indptr[i + 1], be = tp[static_cast<size_t>(i + 1)];
        while (a < ae || b < be) {
            const int64_t ja = a < ae ? g.indices[a] : INT64_MAX, jb = b < be ? ti[static_cast<size_t>(b)] : INT64_MAX;
            const int64_t j = std::min(ja, jb);
            r.ix.push_back(j);
            r.cap.push_back(ja == j && j != i ? caps[a] : 0);
            if (ja == j) ++a;
            if (jb == j) ++b;
        }
        r.ip[static_cast<size_t>(i + 1)] = static_cast<int64_t>(r.ix.size());
    }
    // The pattern is symmetric: row v lists the u with (u, v) ascending, the order rows are walked in here.
    const size_t m = r.ix.size();
    r.rev.assign(m, 0);
    r.res.assign(r.cap.begin(), r.cap.end());
    std::vector<int64_t> next(r.ip.begin(), r.ip.end() - 1);
    for (int64_t u = 0; u < n; ++u)
        for (int64_t k = r.ip[static_cast<size_t>(u)]; k < r.ip[static_cast<size_t>(u + 1)]; ++k)
            r.rev[static_cast<size_t>(k)] = next[static_cast<size_t>(r.ix[static_cast<size_t>(k)])]++;
}

void edmonds_karp(Residual& r, int64_t s, int64_t t) {
    const int64_t n = r.n;
    std::vector<int64_t> pred(static_cast<size_t>(n)), queue(static_cast<size_t>(n)), seen(static_cast<size_t>(n), -1);
    for (int64_t round = 0;; ++round) {
        int64_t head = 0, tail = 0;
        queue[static_cast<size_t>(tail++)] = s;
        seen[static_cast<size_t>(s)] = round;
        bool reached = false;
        while (head < tail && !reached) {
            const int64_t u = queue[static_cast<size_t>(head++)];
            for (int64_t k = r.ip[static_cast<size_t>(u)]; k < r.ip[static_cast<size_t>(u + 1)]; ++k) {
                const int64_t v = r.ix[static_cast<size_t>(k)];
                if (seen[static_cast<size_t>(v)] == round || r.left(k) == 0) continue;
                seen[static_cast<size_t>(v)] = round;
                pred[static_cast<size_t>(v)] = k;
                if (v == t) {
                    reached = true;
                    break;
                }
                queue[static_cast<size_t>(tail++)] = v;
            }
        }
        if (!reached) return;
        uint64_t d = ~uint64_t{0};
        for (int64_t v = t; v != s;
             v = r.ix[static_cast<size_t>(r.rev[static_cast<size_t>(pred[static_cast<size_t>(v)])])])
            d = std::min(d, r.left(pred[static_cast<size_t>(v)]));
        for (int64_t v = t; v != s;
             v = r.ix[static_cast<size_t>(r.rev[static_cast<size_t>(pred[static_cast<size_t>(v)])])])
            r.push(pred[static_cast<size_t>(v)], d);
    }
}

void dinic(Residual& r, int64_t s, int64_t t) {
    const int64_t n = r.n;
    std::vector<int64_t> level(static_cast<size_t>(n)), queue(static_cast<size_t>(n)), arc(static_cast<size_t>(n));
    std::vector<int64_t> path;  // the entries of the current path from s
    path.reserve(static_cast<size_t>(n));
    for (;;) {
        std::fill(level.begin(), level.end(), -1);
        int64_t head = 0, tail = 0;
        queue[static_cast<size_t>(tail++)] = s;
        level[static_cast<size_t>(s)] = 0;
        while (head < tail) {
            const int64_t u = queue[static_cast<size_t>(head++)];
            if (level[static_cast<size_t>(u)] >= level[static_cast<size_t>(t)] && level[static_cast<size_t>(t)] >= 0)
                break;
            for (int64_t k = r.ip[static_cast<size_t>(u)]; k < r.ip[static_cast<size_t>(u + 1)]; ++k) {
                const int64_t v = r.ix[static_cast<size_t>(k)];
                if (level[static_cast<size_t>(v)] >= 0 || r.left(k) == 0) continue;
                level[static_cast<size_t>(v)] = level[static_cast<size_t>(u)] + 1;
                queue[static_cast<size_t>(tail++)] = v;
            }
        }
        if (level[static_cast<size_t>(t)] < 0) return;
        for (int64_t v = 0; v < n; ++v) arc[static_cast<size_t>(v)] = r.ip[static_cast<size_t>(v)];
        path.clear();
        int64_t u = s;
        for (;;) {
            if (u == t) {
                uint64_t d = ~uint64_t{0};
                for (int64_t k : path) d = std::min(d, r.left(k));
                size_t cut = path.size();
                for (size_t q = 0; q < path.size(); ++q) {
                    r.push(path[q], d);
                    if (cut == path.size() && r.left(path[q]) == 0) cut = q;  // the first saturated entry
                }
                path.resize(cut);  // go back to its tail
                u = cut ? r.ix[static_cast<size_t>(path[cut - 1])] : s;
                continue;
            }
            int64_t& k = arc[static_cast<size_t>(u)];
            const int64_t end = r.ip[static_cast<size_t>(u + 1)], next = level[static_cast<size_t>(u)] + 1;
            while (k < end && (level[static_cast<size_t>(r.ix[static_cast<size_t>(k)])] != next || r.left(k) == 0)) ++k;
            if (k < end) {
                path.push_back(k);
                u = r.ix[static_cast<size_t>(k)];
                continue;
            }
            // a dead end: no path to t through u in this level graph
            if (u == s) break;
            level[static_cast<size_t>(u)] = -1;
            path.pop_back();
            u = path.empty() ? s : r.ix[static_cast<size_t>(path.back())];
            ++arc[static_cast<size_t>(u)];
        }
    }
}

nb::object max_flow(nb::handle ip, nb::handle ix, nb::handle ch, int64_t n, int64_t s, int64_t t, bool use_dinic) {
    const char* func = "_max_flow";
    const View g = view_arg(ip, ix, nb::none(), n, func);
    if (!is_ndarray(ch)) throw nb::type_error("_max_flow(): wrong caps");
    const NDArray& c = as_ndarray(ch);
    if (c.ndim != 1 || !c.c_contiguous || c.dtype != DType::Int64 || c.shape[0] != g.nnz())
        throw nb::type_error("_max_flow(): wrong caps");
    if (s < 0 || s >= n || t < 0 || t >= n || s == t) throw nb::value_error("_max_flow(): wrong nodes");
    const int64_t* caps = reinterpret_cast<const int64_t*>(c.data);
    int64_t out_of_source = 0;
    for (int64_t k = g.indptr[s]; k < g.indptr[s + 1]; ++k)
        if (g.indices[k] != s && __builtin_add_overflow(out_of_source, caps[k], &out_of_source)) return nb::none();
    Residual r;
    {
        const parallel::Unlocked unlocked(g.nnz() >= kParallelWork);
        residual(g, caps, r);
        if (use_dinic) dinic(r, s, t);
        else edmonds_karp(r, s, t);
    }
    int64_t value = 0;
    for (int64_t k = r.ip[static_cast<size_t>(s)]; k < r.ip[static_cast<size_t>(s + 1)]; ++k)
        value += r.flow(static_cast<size_t>(k));
    const int64_t m = static_cast<int64_t>(r.ix.size());
    nb::object op = vec1(DType::Int64, n + 1), oi = vec1(DType::Int64, m), od = vec1(DType::Int64, m);
    std::copy(r.ip.begin(), r.ip.end(), ptr<int64_t>(op));
    std::copy(r.ix.begin(), r.ix.end(), ptr<int64_t>(oi));
    int64_t* F = ptr<int64_t>(od);
    for (size_t k = 0; k < static_cast<size_t>(m); ++k) F[k] = r.flow(k);
    return nb::make_tuple(value, op, oi, od);
}

}  // namespace

void bind_flow(nb::module_& m) {
    m.def(
        "_max_flow", &max_flow, nb::arg("indptr"), nb::arg("indices"), nb::arg("caps"), nb::arg("n"), nb::arg("source"),
        nb::arg("sink"), nb::arg("dinic")
    );
}

}  // namespace sd::graph
