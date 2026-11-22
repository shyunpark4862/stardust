// Yen's K shortest loopless paths of sd.graph (docs/design/graph.md §2.3, §3.2 Yen; milestone GR2).
//
//   _yen(indptr, indices, data, n, source, sink, K, directed, preds, negative)
//       -> (dist (M,), pred (M, n) or None, cycle): the first M <= K loopless source-sink paths in the
//       order (left-to-right sum from the source, edge count, node sequence), each path's predecessors
//       in a row (kNoNode off the path); cycle the node of a negative cycle reachable from the source
//       (else -1; the arrays are None then). data None: every edge weighs 1. negative: whether a stored
//       weight is negative (then the potentials of the shared check order the spur searches).
//
// A spur search starts at the spur node with the root path's left-to-right sum x, so the sums it
// compares are those of whole paths from the source; it avoids the root's other nodes and the edges out
// of the spur node that the paths found with the same root take. Among the spur paths of the least sum,
// it returns the one of fewest edges and, among those, the smallest node sequence: the tight edges from
// the spur node, layered by breadth-first edge count, and from the spur node the smallest next node that
// still reaches the sink in that layering. The candidates are kept in that total order, so the K paths
// depend on the data alone. Serial; a call of kParallelWork or more runs without the GIL.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph_mod.hpp"
#include "graph/paths.hpp"
#include "parallel/pool.hpp"

#include <cstring>
#include <set>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using paths::kInf;
using paths::Owned;
using paths::WGraph;
using paths::Work;
using std::int64_t;

struct Path {
    double length;
    std::vector<int64_t> nodes;
};

struct Order {
    bool operator()(const Path& a, const Path& b) const {
        if (a.length != b.length) return a.length < b.length;
        if (a.nodes.size() != b.nodes.size()) return a.nodes.size() < b.nodes.size();
        return a.nodes < b.nodes;
    }
};

class Spur {
public:
    Spur(const WGraph& g, const double* h) : g_(g), h_(h), w_(g.n), d_(static_cast<size_t>(g.n)) {}

    // The best spur path from s (sum x there) to t avoiding the nodes marked in `removed` and the edges
    // s -> v for v in `banned` (sorted). False when t is not reached; else `out` gets s .. t and `length`
    // the sum at t.
    bool search(
        int64_t s,
        double x,
        int64_t t,
        const std::vector<uint8_t>& removed,
        const std::vector<int64_t>& banned,
        std::vector<int64_t>& out,
        double& length
    ) {
        const int64_t n = g_.n;
        double* d = d_.data();
        std::fill(d, d + n, kInf);
        auto allowed = [&](int64_t u, int64_t v) {
            if (v == u || removed[static_cast<size_t>(v)]) return false;
            return u != s || !std::binary_search(banned.begin(), banned.end(), v);
        };
        std::vector<uint8_t>& done = w_.flag;
        std::fill(done.begin(), done.end(), uint8_t{0});
        d[s] = x;
        if (!h_) {
            // Dijkstra; once t is taken, the nodes of the same distance still come (zero-weight edges may
            // join them to t), and nothing past it can be on a tight path to t
            w_.q_set(s, x);
            while (!w_.q_empty()) {
                if (done[static_cast<size_t>(t)] && w_.hk[0] > d[t]) break;
                const int64_t u = w_.q_pop().v;
                done[static_cast<size_t>(u)] = 1;
                const double du = d[u];
                for (int64_t k = g_.ip[u]; k < g_.ip[u + 1]; ++k) {
                    const int64_t v = g_.ix[k];
                    if (!allowed(u, v)) continue;
                    const double tt = du + g_.weight(k);
                    if (tt < d[v]) {
                        d[v] = tt;
                        w_.q_set(v, tt);
                    }
                }
            }
            w_.q_clear();
            for (int64_t v = 0; v < n; ++v)
                if (!done[static_cast<size_t>(v)]) d[v] = kInf;
        } else {
            // negative weights: the label-correcting search on the potentials (a node taken comes back
            // when its distance drops), to the end
            w_.q_set(s, x - h_[s]);
            while (!w_.q_empty()) {
                const int64_t u = w_.q_pop().v;
                const double du = d[u];
                for (int64_t k = g_.ip[u]; k < g_.ip[u + 1]; ++k) {
                    const int64_t v = g_.ix[k];
                    if (!allowed(u, v)) continue;
                    const double tt = du + g_.weight(k);
                    if (tt < d[v]) {
                        d[v] = tt;
                        w_.q_set(v, tt - h_[v]);
                    }
                }
            }
        }
        if (!(d[t] < kInf)) return false;
        // the layering of the tight edges by breadth-first edge count from s (stopping at t's level)
        int64_t* level = w_.level.data();
        int64_t* order = w_.order.data();
        std::fill(level, level + n, int64_t{-1});
        level[s] = 0;
        int64_t qn = 0;
        order[qn++] = s;
        for (int64_t q = 0; q < qn; ++q) {
            const int64_t u = order[q];
            if (u == t || (level[t] >= 0 && level[u] >= level[t])) continue;
            for (int64_t k = g_.ip[u]; k < g_.ip[u + 1]; ++k) {
                const int64_t v = g_.ix[k];
                if (level[v] >= 0 || !(d[v] < kInf) || !allowed(u, v) || d[u] + g_.weight(k) != d[v]) continue;
                level[v] = level[u] + 1;
                order[qn++] = v;
            }
        }
        // good: the nodes from which the layering reaches t, found backward
        std::vector<uint8_t>& good = done;  // reused
        std::fill(good.begin(), good.end(), uint8_t{0});
        good[static_cast<size_t>(t)] = 1;
        auto layered = [&](int64_t u, int64_t k) {
            const int64_t v = g_.ix[k];
            return level[v] == level[u] + 1 && allowed(u, v) && d[u] + g_.weight(k) == d[v];
        };
        for (int64_t q = qn; q-- > 0;) {
            const int64_t u = order[q];
            if (u == t) continue;
            for (int64_t k = g_.ip[u]; k < g_.ip[u + 1]; ++k) {
                if (good[static_cast<size_t>(g_.ix[k])] && layered(u, k)) {
                    good[static_cast<size_t>(u)] = 1;
                    break;
                }
            }
        }
        out.clear();
        out.push_back(s);
        for (int64_t u = s; u != t;) {
            int64_t next = -1;
            for (int64_t k = g_.ip[u]; k < g_.ip[u + 1]; ++k) {  // ascending: the first is the smallest
                if (good[static_cast<size_t>(g_.ix[k])] && layered(u, k)) {
                    next = g_.ix[k];
                    break;
                }
            }
            out.push_back(next);
            u = next;
        }
        length = d[t];
        return true;
    }

private:
    const WGraph& g_;
    const double* h_;
    Work w_;
    std::vector<double> d_;
};

nb::object run_yen(
    nb::handle ip,
    nb::handle ix,
    nb::handle dh,
    int64_t n,
    int64_t s,
    int64_t t,
    int64_t K,
    bool directed,
    bool want_pred,
    bool negative
) {
    const View g0 = view_arg(ip, ix, dh, n, "_yen");
    if (s < 0 || s >= n || t < 0 || t >= n || K < 0) throw nb::value_error("_yen(): wrong arguments");
    std::vector<Path> found;
    int64_t cycle = -1;
    {
        const parallel::Unlocked unlocked(n + g0.nnz() >= kParallelWork);
        Owned own;
        const WGraph g = paths::reading(g0, directed, g0.data != nullptr, own);
        std::vector<double> h;
        if (negative) {
            std::vector<uint8_t> in;
            paths::reach(g, &s, 1, in);
            cycle = paths::negative_cycle(g, in, h);
        }
        if (cycle < 0 && K > 0) {
            if (s == t) {
                found.push_back({0.0, {s}});
            } else {
                Spur spur(g, negative ? h.data() : nullptr);
                std::vector<uint8_t> removed(static_cast<size_t>(n), 0);
                std::vector<int64_t> banned, part;
                double length;
                if (spur.search(s, 0.0, t, removed, banned, part, length)) found.push_back({length, part});
                std::set<Path, Order> candidates;
                while (!found.empty() && static_cast<int64_t>(found.size()) < K) {
                    const std::vector<int64_t> prev = found.back().nodes;
                    double root_sum = 0.0;
                    for (size_t i = 0; i + 1 < prev.size(); ++i) {
                        if (i > 0) {
                            double w = 0.0;
                            paths::edge_weight(g, prev[i - 1], prev[i], w);
                            root_sum += w;
                        }
                        banned.clear();
                        for (const Path& p : found)
                            if (p.nodes.size() > i + 1 &&
                                std::equal(
                                    prev.begin(), prev.begin() + static_cast<std::ptrdiff_t>(i) + 1, p.nodes.begin()
                                ))
                                banned.push_back(p.nodes[i + 1]);
                        std::sort(banned.begin(), banned.end());
                        for (size_t r = 0; r < i; ++r) removed[static_cast<size_t>(prev[r])] = 1;
                        if (spur.search(prev[i], root_sum, t, removed, banned, part, length)) {
                            Path c{
                                length,
                                std::vector<int64_t>(prev.begin(), prev.begin() + static_cast<std::ptrdiff_t>(i))
                            };
                            c.nodes.insert(c.nodes.end(), part.begin(), part.end());
                            candidates.insert(std::move(c));
                        }
                        for (size_t r = 0; r < i; ++r) removed[static_cast<size_t>(prev[r])] = 0;
                    }
                    if (candidates.empty()) break;
                    found.push_back(*candidates.begin());
                    candidates.erase(candidates.begin());
                }
            }
        }
    }
    if (cycle >= 0) return nb::make_tuple(nb::none(), nb::none(), cycle);
    const int64_t M = static_cast<int64_t>(found.size());
    const int64_t s1[] = {M};
    nb::object od = wrap(new_array(DType::Float64, s1));
    double* D = reinterpret_cast<double*>(as_ndarray(od).data);
    for (int64_t m = 0; m < M; ++m) D[m] = found[static_cast<size_t>(m)].length;
    nb::object op = nb::none();
    if (want_pred) {
        const int64_t s2[] = {M, n};
        op = wrap(new_array(DType::Int64, s2));
        int64_t* P = reinterpret_cast<int64_t*>(as_ndarray(op).data);
        std::fill(P, P + M * n, kNoNode);
        for (int64_t m = 0; m < M; ++m) {
            const std::vector<int64_t>& v = found[static_cast<size_t>(m)].nodes;
            for (size_t i = 1; i < v.size(); ++i) P[m * n + v[i]] = v[i - 1];
        }
    }
    return nb::make_tuple(od, op, int64_t{-1});
}

}  // namespace

void bind_yen(nb::module_& m) {
    m.def(
        "_yen", &run_yen, nb::arg("indptr"), nb::arg("indices"), nb::arg("data").none(), nb::arg("n"),
        nb::arg("source"), nb::arg("sink"), nb::arg("K"), nb::arg("directed"), nb::arg("preds"), nb::arg("negative")
    );
}

}  // namespace sd::graph
