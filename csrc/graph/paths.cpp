// The shortest paths of sd.graph (docs/design/graph.md §2.3, §3.2; milestone GR2). The Python layer
// (src/stardust/graph/_paths.py) checks every argument, scans the weights and picks the method.
//
//   _scan(data)
//       (nonfinite, negative): whether a stored weight is NaN or +-inf, and whether one is below 0.
//   _paths(indptr, indices, data, n, sources, method, directed, preds, limit, min_only, flat, negative)
//       method 'D' (Dijkstra on an indexed 4-ary heap), 'BF' (queue Bellman-Ford), 'J' (Johnson: the potentials of the
//       shared negative-cycle check order a Dijkstra that queues a node again when its distance drops),
//       'dense' (the heap-free Dijkstra on the dense matrix; auto's path for dense graphs) or 'U' (edge
//       counts: unweighted, data None). sources None: every node. -> (dist, pred or None, source_of or
//       None, cycle): dist (m, n) ((n,) when flat or min_only), the predecessors of §3.2, min_only's
//       sources; cycle the node a negative cycle reachable from the sources goes through (else -1; then
//       the arrays are None). negative: whether a stored weight is negative (BF and J check for cycles
//       then). One source is one task; the answer of a source is fixed by the definition.
//   _floyd_warshall(indptr, indices, data, n, directed, preds)
//       the blocked Floyd-Warshall (64 x 64 blocks; decisions GR25, GR27) -> (dist (n, n), pred or None,
//       cycle).
//   _construct(indptr, indices, data, n, pred, directed, null_value)
//       (dist, None) with dist[i, j] the left-to-right sum along the path pred[i] draws to j (0 on the
//       diagonal, null_value where pred[i, j] is -9999), or (None, (kind, i, j, value)) for predecessors
//       that are not paths of the graph: kind 1 a value that is neither a node nor -9999, 2 pred[i, j] = p
//       with no edge p -> j, 3 a cycle through j, 4 a path to j that ends at value instead of i.
//   _reconstruct(indptr, indices, data, n, pred, directed)
//       (indptr, indices, data) of the tree of edges pred[v] -> v, or (None, (kind, v, value)) with kind 1
//       or 2 as above and 5 for pred[v] = v.
//
// Every function that does kParallelWork or more runs without the GIL.
#include "graph/paths.hpp"

#include <arm_neon.h>

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"

#include <nanobind/stl/string.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>

namespace nb = nanobind;

namespace sd::graph {

namespace paths {

// ---- the graph read for paths

Owned undirected(const View& g, bool weights) {
    const int64_t n = g.n, nnz = g.nnz();
    // the transpose by a counting sort (rows of the transpose ascending, as g's rows are visited in order)
    std::vector<int64_t> tp(static_cast<size_t>(n + 1), 0), ti(static_cast<size_t>(nnz));
    std::vector<double> tw(weights ? static_cast<size_t>(nnz) : 0);
    for (int64_t k = 0; k < nnz; ++k) ++tp[static_cast<size_t>(g.indices[k] + 1)];
    for (int64_t i = 0; i < n; ++i) tp[static_cast<size_t>(i + 1)] += tp[static_cast<size_t>(i)];
    {
        std::vector<int64_t> next(tp.begin(), tp.end() - 1);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t k = g.indptr[i]; k < g.indptr[i + 1]; ++k) {
                const int64_t dst = next[static_cast<size_t>(g.indices[k])]++;
                ti[static_cast<size_t>(dst)] = i;
                if (weights) tw[static_cast<size_t>(dst)] = g.data[k];
            }
        }
    }
    Owned o;
    o.ip.resize(static_cast<size_t>(n + 1));
    o.ix.reserve(static_cast<size_t>(nnz));
    if (weights) o.w.reserve(static_cast<size_t>(nnz));
    o.ip[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t a = g.indptr[i], b = tp[static_cast<size_t>(i)];
        const int64_t ae = g.indptr[i + 1], be = tp[static_cast<size_t>(i + 1)];
        while (a < ae || b < be) {
            const int64_t ja = a < ae ? g.indices[a] : INT64_MAX, jb = b < be ? ti[static_cast<size_t>(b)] : INT64_MAX;
            if (ja < jb) {
                o.ix.push_back(ja);
                if (weights) o.w.push_back(g.data[a]);
                ++a;
            } else if (jb < ja) {
                o.ix.push_back(jb);
                if (weights) o.w.push_back(tw[static_cast<size_t>(b)]);
                ++b;
            } else {
                o.ix.push_back(ja);
                if (weights) o.w.push_back(undirected_weight(g.data[a], tw[static_cast<size_t>(b)]));
                ++a;
                ++b;
            }
        }
        o.ip[static_cast<size_t>(i + 1)] = static_cast<int64_t>(o.ix.size());
    }
    return o;
}

WGraph reading(const View& g, bool directed, bool weights, Owned& own) {
    if (directed || symmetric(g, weights)) return WGraph{g.n, g.indptr, g.indices, weights ? g.data : nullptr};
    own = undirected(g, weights);
    return WGraph{g.n, own.ip.data(), own.ix.data(), weights ? own.w.data() : nullptr};
}

bool edge_weight(const WGraph& g, int64_t u, int64_t v, double& w) {
    const int64_t* b = g.ix + g.ip[u];
    const int64_t* e = g.ix + g.ip[u + 1];
    const int64_t* p = std::lower_bound(b, e, v);
    if (p == e || *p != v) return false;
    w = g.weight(p - g.ix);
    return true;
}

void reach(const WGraph& g, const int64_t* src, int64_t m, std::vector<uint8_t>& mark) {
    mark.assign(static_cast<size_t>(g.n), 0);
    std::vector<int64_t> q;
    for (int64_t i = 0; i < m; ++i) {
        if (!mark[static_cast<size_t>(src[i])]) {
            mark[static_cast<size_t>(src[i])] = 1;
            q.push_back(src[i]);
        }
    }
    for (size_t h = 0; h < q.size(); ++h) {
        const int64_t u = q[h];
        for (int64_t k = g.ip[u]; k < g.ip[u + 1]; ++k) {
            const int64_t v = g.ix[k];
            if (!mark[static_cast<size_t>(v)]) {
                mark[static_cast<size_t>(v)] = 1;
                q.push_back(v);
            }
        }
    }
}

namespace {

// The smallest node of the cycle the parent pointers lead into from v, or -1 when they end at the root.
int64_t parent_cycle(const std::vector<int64_t>& parent, int64_t v, std::vector<int64_t>& stamp, int64_t id) {
    int64_t x = v;
    while (x >= 0 && stamp[static_cast<size_t>(x)] != id) {
        stamp[static_cast<size_t>(x)] = id;
        x = parent[static_cast<size_t>(x)];
    }
    if (x < 0) return -1;
    int64_t low = x;
    for (int64_t y = parent[static_cast<size_t>(x)]; y != x; y = parent[static_cast<size_t>(y)]) low = std::min(low, y);
    return low;
}

}  // namespace

int64_t negative_cycle(const WGraph& g, const std::vector<uint8_t>& in, std::vector<double>& h) {
    const int64_t n = g.n;
    h.assign(static_cast<size_t>(n), 0.0);
    std::vector<int64_t> parent(static_cast<size_t>(n), -1), count(static_cast<size_t>(n), 0);
    std::vector<int64_t> stamp(static_cast<size_t>(n), -1);
    std::vector<int64_t> ring(static_cast<size_t>(std::max<int64_t>(n, 1)));
    std::vector<uint8_t> inq(static_cast<size_t>(n), 0);
    int64_t head = 0, size = 0, r = 0, checks = 0;
    for (int64_t v = 0; v < n; ++v) {
        if (!in[static_cast<size_t>(v)]) continue;
        ++r;
        ring[static_cast<size_t>(size++)] = v;
        inq[static_cast<size_t>(v)] = 1;
    }
    while (size > 0) {
        const int64_t u = ring[static_cast<size_t>(head)];
        head = head + 1 == n ? 0 : head + 1;
        --size;
        inq[static_cast<size_t>(u)] = 0;
        const double hu = h[static_cast<size_t>(u)];
        for (int64_t k = g.ip[u]; k < g.ip[u + 1]; ++k) {
            const int64_t v = g.ix[k];
            const double w = g.weight(k);
            if (v == u) {
                if (w < 0) return u;  // a negative self loop is a cycle by itself
                continue;
            }
            const double t = hu + w;
            if (!(t < h[static_cast<size_t>(v)])) continue;
            h[static_cast<size_t>(v)] = t;
            parent[static_cast<size_t>(v)] = u;
            if (inq[static_cast<size_t>(v)]) continue;
            if (++count[static_cast<size_t>(v)] > r) {
                // queued more often than a path of the marked nodes has edges: the parent pointers hold a
                // cycle once the values keep dropping; look for it, and go on when it is not there yet
                const int64_t c = parent_cycle(parent, v, stamp, checks++);
                if (c >= 0) return c;
                if (checks > r + 1) return v;  // never found: report the node that keeps dropping
                count[static_cast<size_t>(v)] = 0;
            }
            inq[static_cast<size_t>(v)] = 1;
            int64_t tail = head + size;
            if (tail >= n) tail -= n;
            ring[static_cast<size_t>(tail)] = v;
            ++size;
        }
    }
    return -1;
}

// ---- single searches

void dijkstra(const WGraph& g, const int64_t* src, int64_t m, double limit, double* d, Work& w) {
    const int64_t n = g.n;
    std::fill(d, d + n, kInf);
    for (int64_t i = 0; i < m; ++i) {
        d[src[i]] = 0.0;
        w.q_set(src[i], 0.0);
    }
    bool cut = false;
    while (!w.q_empty()) {
        const Work::Item it = w.q_pop();
        const int64_t u = it.v;
        const double du = it.key;
        if (du > limit) {
            d[u] = kInf;
            cut = true;
            break;
        }
        for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k) {
            const int64_t v = g.ix[k];
            const double t = du + g.weight(k);
            if (t < d[v]) {  // never a node already taken: fl(du + w) >= du >= d(v) for w >= 0
                d[v] = t;
                w.q_set(v, t);
            }
        }
    }
    if (cut) {  // the nodes still queued are past the limit
        for (int64_t i = 0; i < w.hn; ++i) d[w.hv[static_cast<size_t>(i)]] = kInf;
        w.q_clear();
    }
}

void bfs(const WGraph& g, const int64_t* src, int64_t m, double limit, double* d, Work& w) {
    std::fill(d, d + g.n, kInf);
    int64_t qn = 0;
    for (int64_t i = 0; i < m; ++i) {
        if (d[src[i]] != 0.0) {
            d[src[i]] = 0.0;
            w.order[static_cast<size_t>(qn++)] = src[i];
        }
    }
    for (int64_t h = 0; h < qn; ++h) {
        const int64_t u = w.order[static_cast<size_t>(h)];
        const double next = d[u] + 1.0;
        if (next > limit) continue;
        for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k) {
            const int64_t v = g.ix[k];
            if (d[v] == kInf) {
                d[v] = next;
                w.order[static_cast<size_t>(qn++)] = v;
            }
        }
    }
}

bool bellman_ford(const WGraph& g, int64_t s, double* d, Work& w) {
    const int64_t n = g.n;
    std::fill(d, d + n, kInf);
    std::fill(w.flag.begin(), w.flag.end(), uint8_t{0});
    std::fill(w.count.begin(), w.count.end(), int64_t{0});
    int64_t* ring = w.order.data();
    int64_t head = 0, size = 1;
    ring[0] = s;
    d[s] = 0.0;
    w.flag[static_cast<size_t>(s)] = 1;
    while (size > 0) {
        const int64_t u = ring[head];
        head = head + 1 == n ? 0 : head + 1;
        --size;
        w.flag[static_cast<size_t>(u)] = 0;
        const double du = d[u];
        for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k) {
            const int64_t v = g.ix[k];
            if (v == u) continue;
            const double t = du + g.weight(k);
            if (!(t < d[v])) continue;
            d[v] = t;
            if (w.flag[static_cast<size_t>(v)]) continue;
            if (++w.count[static_cast<size_t>(v)] > n) return false;
            w.flag[static_cast<size_t>(v)] = 1;
            int64_t tail = head + size;
            if (tail >= n) tail -= n;
            ring[tail] = v;
            ++size;
        }
    }
    return true;
}

bool johnson(const WGraph& g, int64_t s, const double* h, double* d, Work& w) {
    const int64_t n = g.n;
    std::fill(d, d + n, kInf);
    std::fill(w.count.begin(), w.count.end(), int64_t{0});
    d[s] = 0.0;
    w.q_set(s, 0.0 - h[s]);
    while (!w.q_empty()) {
        const int64_t u = w.q_pop().v;  // a node taken leaves the queue and comes back if d(u) drops
        if (++w.count[static_cast<size_t>(u)] > n + 1) {
            w.q_clear();
            return false;
        }
        const double du = d[u];
        for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k) {
            const int64_t v = g.ix[k];
            if (v == u) continue;
            const double t = du + g.weight(k);
            if (!(t < d[v])) continue;
            d[v] = t;
            w.q_set(v, t - h[v]);
        }
    }
    return true;
}

void predecessors(
    const WGraph& g, const double* d, const int64_t* src, int64_t m, int64_t* pred, int64_t* source_of, Work& w
) {
    const int64_t n = g.n;
    int64_t* level = w.level.data();
    int64_t* order = w.order.data();
    std::fill(level, level + n, int64_t{-1});
    std::fill(pred, pred + n, kNoNode);
    int64_t qn = 0;
    for (int64_t i = 0; i < m; ++i) {
        if (level[src[i]] != 0) {
            level[src[i]] = 0;
            order[qn++] = src[i];
        }
    }
    for (int64_t h = 0; h < qn; ++h) {
        const int64_t u = order[h];
        const double du = d[u];
        const int64_t lu = level[u];
        for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k) {
            const int64_t v = g.ix[k];
            const double dv = d[v];
            if (v == u || !(dv < kInf) || du + g.weight(k) != dv) continue;
            const int64_t lv = level[v];
            if (lv < 0) {
                level[v] = lu + 1;
                pred[v] = u;
                order[qn++] = v;
            } else if (lv == lu + 1 && u < pred[v]) {
                pred[v] = u;
            }
        }
    }
    if (source_of) {
        std::fill(source_of, source_of + n, kNoNode);
        for (int64_t i = 0; i < qn; ++i) {
            const int64_t v = order[i];
            source_of[v] = pred[v] == kNoNode ? v : source_of[pred[v]];
        }
    }
}

}  // namespace paths

namespace {

using paths::kInf;
using paths::Owned;
using paths::WGraph;
using paths::Work;
using std::int64_t;

nb::object new_array1(DType t, int64_t a) {
    const int64_t shape[] = {a};
    return wrap(new_array(t, shape));
}

nb::object new_array2(DType t, int64_t a, int64_t b) {
    const int64_t shape[] = {a, b};
    return wrap(new_array(t, shape));
}

template <class T>
T* ptr(nb::object& o) {
    return reinterpret_cast<T*>(as_ndarray(o).data);
}

// ---- the weight scan

nb::object scan(nb::handle dh) {
    if (!is_ndarray(dh)) throw nb::type_error("_scan(): wrong data");
    const NDArray& a = as_ndarray(dh);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64) throw nb::type_error("_scan(): wrong data");
    const double* x = reinterpret_cast<const double*>(a.data);
    const int64_t m = a.shape[0];
    bool nonfinite = false, negative = false;
    {
        const parallel::Unlocked unlocked(m >= kParallelWork);
        uint64_t bad = 0, neg = 0;
        for (int64_t k = 0; k < m; ++k) {
            const double v = x[k];
            bad |= static_cast<uint64_t>(!(std::fabs(v) < kInf));
            neg |= static_cast<uint64_t>(v < 0.0);
        }
        nonfinite = bad != 0;
        negative = neg != 0;
    }
    return nb::make_tuple(nonfinite, negative);
}

// ---- the heap-free dense Dijkstra (auto on dense graphs; decision GR26)

// One source on the dense matrix W (n x n, +inf where there is no edge and on the diagonal): key holds the
// tentative distance of a node not yet taken and NaN for one taken, so the relax min(key, du + w) (FMIN
// keeps a NaN) leaves a taken node alone with no mask and FMINNM, which passes over a NaN, gives each block
// of 64 keys' minimum; the next node is the first index of the first block holding the smallest key. The
// same fixed point as the heap's Dijkstra (docs/design/graph.md §3.2).
void dense_dijkstra(const double* W, int64_t n, int64_t s, double* d, double* key) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill(key, key + n, kInf);
    std::fill(d, d + n, kInf);
    key[s] = 0.0;
    int64_t u = s;
    double du = 0.0;
    for (int64_t step = 0; step < n; ++step) {
        d[u] = du;
        key[u] = nan;
        const double* row = W + u * n;
        const float64x2_t vdu = vdupq_n_f64(du);
        double best = kInf;
        int64_t best_block = -1, arg = -1;
        int64_t j = 0;
        for (; j + 64 <= n; j += 64) {
            float64x2_t m0 = vdupq_n_f64(kInf), m1 = m0, m2 = m0, m3 = m0;
            for (int q = 0; q < 64; q += 8) {
                const int64_t o = j + q;
                const float64x2_t k0 = vminq_f64(vld1q_f64(key + o), vaddq_f64(vdu, vld1q_f64(row + o)));
                const float64x2_t k1 = vminq_f64(vld1q_f64(key + o + 2), vaddq_f64(vdu, vld1q_f64(row + o + 2)));
                const float64x2_t k2 = vminq_f64(vld1q_f64(key + o + 4), vaddq_f64(vdu, vld1q_f64(row + o + 4)));
                const float64x2_t k3 = vminq_f64(vld1q_f64(key + o + 6), vaddq_f64(vdu, vld1q_f64(row + o + 6)));
                vst1q_f64(key + o, k0);
                vst1q_f64(key + o + 2, k1);
                vst1q_f64(key + o + 4, k2);
                vst1q_f64(key + o + 6, k3);
                m0 = vminnmq_f64(m0, k0);
                m1 = vminnmq_f64(m1, k1);
                m2 = vminnmq_f64(m2, k2);
                m3 = vminnmq_f64(m3, k3);
            }
            const double bm = vminnmvq_f64(vminnmq_f64(vminnmq_f64(m0, m1), vminnmq_f64(m2, m3)));
            if (bm < best) {
                best = bm;
                best_block = j;
            }
        }
        for (int64_t t = j; t < n; ++t) {  // the tail, one at a time (a NaN key stays NaN)
            const double kt = key[t];
            if (kt != kt) continue;
            const double k = std::min(kt, du + row[t]);
            key[t] = k;
            if (k < best) {
                best = k;
                best_block = -1;
                arg = t;
            }
        }
        if (!(best < kInf)) break;  // the rest is unreachable
        if (best_block >= 0)
            for (arg = best_block; !(key[arg] == best); ++arg) {}
        u = arg;
        du = best;
    }
}

// ---- _paths

struct Sources {
    std::vector<int64_t> own;  // every node when the caller passed None
    const int64_t* p = nullptr;
    int64_t m = 0;
};

Sources sources_arg(nb::handle sh, int64_t n) {
    Sources s;
    if (sh.is_none()) {
        s.own.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) s.own[static_cast<size_t>(i)] = i;
        s.p = s.own.data();
        s.m = n;
        return s;
    }
    if (!is_ndarray(sh)) throw nb::type_error("_paths(): wrong sources");
    const NDArray& a = as_ndarray(sh);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Int64) throw nb::type_error("_paths(): wrong sources");
    s.p = reinterpret_cast<const int64_t*>(a.data);
    s.m = a.shape[0];
    for (int64_t i = 0; i < s.m; ++i)
        if (s.p[i] < 0 || s.p[i] >= n) throw nb::value_error("_paths(): wrong sources");
    return s;
}

enum class Method { D, BF, J, Dense, U };

Method method_arg(const std::string& m) {
    if (m == "D") return Method::D;
    if (m == "BF") return Method::BF;
    if (m == "J") return Method::J;
    if (m == "dense") return Method::Dense;
    if (m == "U") return Method::U;
    throw nb::value_error("_paths(): wrong method");
}

nb::object run_paths(
    nb::handle ip,
    nb::handle ix,
    nb::handle dh,
    int64_t n,
    nb::handle sh,
    const std::string& method_name,
    bool directed,
    bool want_pred,
    double limit,
    bool min_only,
    bool flat,
    bool negative
) {
    const View g0 = view_arg(ip, ix, dh, n, "_paths");
    const Method method = method_arg(method_name);
    if ((method == Method::U) != (g0.data == nullptr)) throw nb::type_error("_paths(): wrong data");
    const Sources src = sources_arg(sh, n);
    const int64_t rows = min_only ? 1 : src.m;
    const bool one_row = flat || min_only;
    nb::object od = one_row ? new_array1(DType::Float64, n) : new_array2(DType::Float64, rows, n);
    nb::object op =
        want_pred ? (one_row ? new_array1(DType::Int64, n) : new_array2(DType::Int64, rows, n)) : nb::none();
    nb::object os = want_pred && min_only ? new_array1(DType::Int64, n) : nb::none();
    double* D = ptr<double>(od);
    int64_t* P = want_pred ? ptr<int64_t>(op) : nullptr;
    int64_t* SO = want_pred && min_only ? ptr<int64_t>(os) : nullptr;
    const int64_t work = rows * (n + g0.nnz());
    int64_t cycle = -1;
    {
        const parallel::Unlocked unlocked(work >= kParallelWork || (!directed && g0.nnz() >= kParallelWork));
        Owned own;
        const WGraph g = paths::reading(g0, directed, method != Method::U, own);
        std::vector<double> h;
        if (negative && (method == Method::BF || method == Method::J)) {
            std::vector<uint8_t> in;
            paths::reach(g, src.p, src.m, in);
            cycle = paths::negative_cycle(g, in, h);
        }
        if (cycle < 0 && min_only) {
            Work w(n);
            if (method == Method::U) paths::bfs(g, src.p, src.m, limit, D, w);
            else paths::dijkstra(g, src.p, src.m, limit, D, w);
            if (P) paths::predecessors(g, D, src.p, src.m, P, SO, w);
        } else if (cycle < 0) {
            std::vector<double> W;  // the dense matrix of the 'dense' path
            if (method == Method::Dense) {
                W.assign(static_cast<size_t>(n * n), kInf);
                for (int64_t i = 0; i < n; ++i)
                    for (int64_t k = g.ip[i]; k < g.ip[i + 1]; ++k)
                        if (g.ix[k] != i) W[static_cast<size_t>(i * n + g.ix[k])] = g.w[k];
            }
            std::atomic<int64_t> overflow{INT64_MAX};
            // a task takes a few sources when one is small (the split depends on the sizes alone)
            const int64_t per = std::max<int64_t>(1, std::min<int64_t>(64, 4096 / std::max<int64_t>(1, n + g.nnz())));
            const int64_t ntasks = (rows + per - 1) / per;
            auto task = [&](int64_t t) {
                Work w(n);
                std::vector<double> key(method == Method::Dense ? static_cast<size_t>(n) : 0);
                for (int64_t r = t * per, e = std::min(rows, (t + 1) * per); r < e; ++r) {
                    const int64_t s = src.p[r];
                    double* d = D + r * n;
                    bool ok = true;
                    switch (method) {
                        case Method::D: paths::dijkstra(g, &s, 1, limit, d, w); break;
                        case Method::U: paths::bfs(g, &s, 1, limit, d, w); break;
                        case Method::BF: ok = paths::bellman_ford(g, s, d, w); break;
                        case Method::J:
                            if (negative) ok = paths::johnson(g, s, h.data(), d, w);
                            else paths::dijkstra(g, &s, 1, kInf, d, w);
                            break;
                        case Method::Dense: dense_dijkstra(W.data(), n, s, d, key.data()); break;
                    }
                    if (!ok) {
                        int64_t cur = overflow.load();
                        while (s < cur && !overflow.compare_exchange_weak(cur, s)) {}
                        continue;
                    }
                    if (P) paths::predecessors(g, d, &s, 1, P + r * n, nullptr, w);
                }
            };
            if (ntasks > 1 && work >= kParallelWork) parallel::parallel_for(ntasks, task);
            else
                for (int64_t t = 0; t < ntasks; ++t) task(t);
            // The shared check found no cycle but a search kept going (a cycle whose exact sum is not
            // negative but whose rounded sums keep dropping; docs/design/graph.md §4): report it.
            if (overflow.load() != INT64_MAX) {
                std::vector<uint8_t> in;
                const int64_t s = overflow.load();
                paths::reach(g, &s, 1, in);
                std::vector<double> h2;
                cycle = paths::negative_cycle(g, in, h2);
                if (cycle < 0) cycle = s;
            }
        }
    }
    if (cycle >= 0) return nb::make_tuple(nb::none(), nb::none(), nb::none(), cycle);
    return nb::make_tuple(od, op, os, int64_t{-1});
}

// ---- the blocked Floyd-Warshall (decisions GR25, GR27)

constexpr int64_t B = 64;  // the block: fixed, since another size changes the last bits

// C = min(C, A (x) Bm) where A or Bm may be C (the diagonal block and its row and column): k in order. The
// row of C a step reads in A is broadcast before the step writes it; the diagonal of the block holding k
// is 0 (or the cycle is being reported), so the entries a step reads do not change in it.
void fw_dependent(double* C, const double* A, const double* Bm, int64_t N) {
    for (int64_t k = 0; k < B; ++k) {
        const double* b = Bm + k * N;
        for (int64_t i = 0; i < B; ++i) {
            const float64x2_t a = vdupq_n_f64(A[i * N + k]);
            double* c = C + i * N;
            for (int64_t j = 0; j < B; j += 2)
                vst1q_f64(c + j, vminq_f64(vld1q_f64(c + j), vaddq_f64(a, vld1q_f64(b + j))));
        }
    }
}

// The same with the predecessors: where a sum is smaller, PC[i][j] = PB[k][j] (Floyd-Warshall's rule; an
// equal sum leaves it). The distances are those of fw_dependent, bit for bit.
void fw_dependent_p(double* C, const double* A, const double* Bm, int64_t* PC, const int64_t* PB, int64_t N) {
    for (int64_t k = 0; k < B; ++k) {
        const double* b = Bm + k * N;
        const int64_t* pb = PB + k * N;
        for (int64_t i = 0; i < B; ++i) {
            const float64x2_t a = vdupq_n_f64(A[i * N + k]);
            double* c = C + i * N;
            int64_t* p = PC + i * N;
            for (int64_t j = 0; j < B; j += 2) {
                const float64x2_t cv = vld1q_f64(c + j), t = vaddq_f64(a, vld1q_f64(b + j));
                const uint64x2_t lt = vcltq_f64(t, cv);
                vst1q_f64(c + j, vminq_f64(cv, t));
                vst1q_s64(p + j, vbslq_s64(lt, vld1q_s64(pb + j), vld1q_s64(p + j)));
            }
        }
    }
}

// C = min(C, A (x) Bm) with C apart from A and Bm: 4 x 8 register tiles, k in order for every entry.
void fw_independent(double* C, const double* A, const double* Bm, int64_t N) {
    for (int64_t i = 0; i < B; i += 4) {
        for (int64_t j = 0; j < B; j += 8) {
            float64x2_t c[4][4];
            for (int r = 0; r < 4; ++r)
                for (int q = 0; q < 4; ++q) c[r][q] = vld1q_f64(C + (i + r) * N + j + 2 * q);
            for (int64_t k = 0; k < B; ++k) {
                const double* b = Bm + k * N + j;
                const float64x2_t b0 = vld1q_f64(b), b1 = vld1q_f64(b + 2), b2 = vld1q_f64(b + 4),
                                  b3 = vld1q_f64(b + 6);
                for (int r = 0; r < 4; ++r) {
                    const float64x2_t a = vdupq_n_f64(A[(i + r) * N + k]);
                    c[r][0] = vminq_f64(c[r][0], vaddq_f64(a, b0));
                    c[r][1] = vminq_f64(c[r][1], vaddq_f64(a, b1));
                    c[r][2] = vminq_f64(c[r][2], vaddq_f64(a, b2));
                    c[r][3] = vminq_f64(c[r][3], vaddq_f64(a, b3));
                }
            }
            for (int r = 0; r < 4; ++r)
                for (int q = 0; q < 4; ++q) vst1q_f64(C + (i + r) * N + j + 2 * q, c[r][q]);
        }
    }
}

// The same with the predecessors (4 x 4 tiles: the distances and the predecessors in registers).
void fw_independent_p(double* C, const double* A, const double* Bm, int64_t* PC, const int64_t* PB, int64_t N) {
    for (int64_t i = 0; i < B; i += 4) {
        for (int64_t j = 0; j < B; j += 4) {
            float64x2_t c[4][2];
            int64x2_t p[4][2];
            for (int r = 0; r < 4; ++r)
                for (int q = 0; q < 2; ++q) {
                    c[r][q] = vld1q_f64(C + (i + r) * N + j + 2 * q);
                    p[r][q] = vld1q_s64(PC + (i + r) * N + j + 2 * q);
                }
            for (int64_t k = 0; k < B; ++k) {
                const double* b = Bm + k * N + j;
                const int64_t* pb = PB + k * N + j;
                const float64x2_t b0 = vld1q_f64(b), b1 = vld1q_f64(b + 2);
                const int64x2_t q0 = vld1q_s64(pb), q1 = vld1q_s64(pb + 2);
                for (int r = 0; r < 4; ++r) {
                    const float64x2_t a = vdupq_n_f64(A[(i + r) * N + k]);
                    const float64x2_t t0 = vaddq_f64(a, b0), t1 = vaddq_f64(a, b1);
                    p[r][0] = vbslq_s64(vcltq_f64(t0, c[r][0]), q0, p[r][0]);
                    p[r][1] = vbslq_s64(vcltq_f64(t1, c[r][1]), q1, p[r][1]);
                    c[r][0] = vminq_f64(c[r][0], t0);
                    c[r][1] = vminq_f64(c[r][1], t1);
                }
            }
            for (int r = 0; r < 4; ++r)
                for (int q = 0; q < 2; ++q) {
                    vst1q_f64(C + (i + r) * N + j + 2 * q, c[r][q]);
                    vst1q_s64(PC + (i + r) * N + j + 2 * q, p[r][q]);
                }
        }
    }
}

// D (N x N, N a multiple of B) holds the weights, +inf for no edge and 0 on the diagonal (or a negative
// self loop's weight); P (or null) the predecessors (i for an edge i -> j, kNoNode elsewhere). Both are
// stored block by block (block (bi, bj) is B * B contiguous entries at (bi * nb + bj) * B * B, its rows B
// apart): a block's 64 rows then lie in one 32 KB run instead of 64 rows N apart, which in the row-major
// matrix fell into a few cache sets (N a multiple of 1024 bytes) and kept missing. The layout changes no
// operation. Step kb updates the diagonal block, then its row and column (one task per block), then every
// other block (one task per block): what a block computes depends on the block size alone, not on the
// thread count.
void floyd_warshall_blocks(double* D, int64_t* P, int64_t N, bool split) {
    const int64_t nb = N / B;
    auto run = [&](int64_t ntasks, const std::function<void(int64_t)>& fn) {
        if (split && ntasks > 1) parallel::parallel_for(ntasks, fn);
        else
            for (int64_t t = 0; t < ntasks; ++t) fn(t);
    };
    auto at = [&](int64_t bi, int64_t bj) { return (bi * nb + bj) * B * B; };
    for (int64_t kb = 0; kb < nb; ++kb) {
        const int64_t kk = at(kb, kb);
        if (P) fw_dependent_p(D + kk, D + kk, D + kk, P + kk, P + kk, B);
        else fw_dependent(D + kk, D + kk, D + kk, B);
        run(2 * (nb - 1), [&](int64_t x) {
            int64_t m = x % (nb - 1);
            m += m >= kb;
            if (x < nb - 1) {  // the row of block kb: C = D[kb][m], A the diagonal block, B = C
                const int64_t c = at(kb, m);
                if (P) fw_dependent_p(D + c, D + kk, D + c, P + c, P + c, B);
                else fw_dependent(D + c, D + kk, D + c, B);
            } else {  // the column: C = D[m][kb], A = C, B the diagonal block
                const int64_t c = at(m, kb);
                if (P) fw_dependent_p(D + c, D + c, D + kk, P + c, P + kk, B);
                else fw_dependent(D + c, D + c, D + kk, B);
            }
        });
        run((nb - 1) * (nb - 1), [&](int64_t x) {
            int64_t i = x / (nb - 1), j = x % (nb - 1);
            i += i >= kb;
            j += j >= kb;
            const int64_t c = at(i, j), a = at(i, kb), b = at(kb, j);
            if (P) fw_independent_p(D + c, D + a, D + b, P + c, P + b, B);
            else fw_independent(D + c, D + a, D + b, B);
        });
    }
}

nb::object run_floyd_warshall(nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, bool directed, bool want_pred) {
    const View g0 = view_arg(ip, ix, dh, n, "_floyd_warshall");
    const int64_t N = (n + B - 1) / B * B, nblk = N / B;
    nb::object od = new_array2(DType::Float64, n, n);
    nb::object op = want_pred ? new_array2(DType::Int64, n, n) : nb::none();
    double* out = ptr<double>(od);
    int64_t* pout = want_pred ? ptr<int64_t>(op) : nullptr;
    int64_t cycle = -1;
    {
        const bool big = N * N >= kParallelWork;
        const parallel::Unlocked unlocked(big);
        Owned own;
        const WGraph g = paths::reading(g0, directed, g0.data != nullptr, own);
        // the matrices block by block (floyd_warshall_blocks); entry (i, j) at place(i, j)
        auto place = [&](int64_t i, int64_t j) { return ((i / B) * nblk + j / B) * B * B + (i % B) * B + j % B; };
        std::vector<double> Dv(static_cast<size_t>(N * N), kInf);
        std::vector<int64_t> Pv(want_pred ? static_cast<size_t>(N * N) : 0, kNoNode);
        double* D = Dv.data();
        int64_t* P = want_pred ? Pv.data() : nullptr;
        for (int64_t i = 0; i < N; ++i) D[place(i, i)] = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t k = g.ip[i]; k < g.ip[i + 1]; ++k) {
                const int64_t j = g.ix[k];
                const double w = g.weight(k);
                if (j == i) {
                    if (w < 0) D[place(i, i)] = w;  // a negative self loop: a negative cycle
                    continue;
                }
                D[place(i, j)] = w;
                if (P) P[place(i, j)] = i;
            }
        }
        floyd_warshall_blocks(D, P, N, big && parallel::available());
        int64_t first_negative = -1;
        for (int64_t i = 0; i < n && first_negative < 0; ++i)
            if (D[place(i, i)] < 0) first_negative = i;
        if (first_negative >= 0) {
            std::vector<uint8_t> all(static_cast<size_t>(n), 1);
            std::vector<double> h;
            cycle = paths::negative_cycle(g, all, h);
            if (cycle < 0) cycle = first_negative;
        } else {
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t bj = 0; bj < nblk; ++bj) {
                    const int64_t j0 = bj * B, len = std::min(B, n - j0), from = place(i, j0);
                    std::memcpy(out + i * n + j0, D + from, static_cast<size_t>(len) * sizeof(double));
                    if (P) std::memcpy(pout + i * n + j0, P + from, static_cast<size_t>(len) * sizeof(int64_t));
                }
            }
        }
    }
    if (cycle >= 0) return nb::make_tuple(nb::none(), nb::none(), cycle);
    return nb::make_tuple(od, op, int64_t{-1});
}

// ---- construct_dist_matrix, reconstruct_path

const NDArray& index_matrix(nb::handle h, int64_t ndim, const char* func) {
    if (!is_ndarray(h)) throw nb::type_error((std::string(func) + "(): wrong predecessors").c_str());
    const NDArray& a = as_ndarray(h);
    if (a.ndim != ndim || !a.c_contiguous || a.dtype != DType::Int64)
        throw nb::type_error((std::string(func) + "(): wrong predecessors").c_str());
    return a;
}

nb::object construct(
    nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, nb::handle ph, bool directed, double null
) {
    const View g0 = view_arg(ip, ix, dh, n, "_construct");
    if (!g0.data) throw nb::type_error("_construct(): wrong data");
    const NDArray& pa = index_matrix(ph, 2, "_construct");
    if (pa.shape[0] != n || pa.shape[1] != n) throw nb::value_error("_construct(): wrong predecessors");
    const int64_t* pred = reinterpret_cast<const int64_t*>(pa.data);
    nb::object od = new_array2(DType::Float64, n, n);
    double* D = ptr<double>(od);
    // the first failure in row order (rows run in any order): (row * n + column) and its kind and value
    parallel::FirstRow first;
    std::vector<int64_t> kind(static_cast<size_t>(std::max<int64_t>(n, 1))),
        value(static_cast<size_t>(std::max<int64_t>(n, 1)));
    {
        const bool big = n * n >= kParallelWork;
        const parallel::Unlocked unlocked(big);
        Owned own;
        const WGraph g = paths::reading(g0, directed, true, own);
        auto row = [&](int64_t i) {
            if (first.failed_before(i * n)) return;
            const int64_t* p = pred + i * n;
            double* d = D + i * n;
            for (int64_t j = 0; j < n; ++j) {
                if (p[j] != kNoNode && (p[j] < 0 || p[j] >= n)) {
                    kind[static_cast<size_t>(i)] = 1;
                    value[static_cast<size_t>(i)] = p[j];
                    first.offer(i * n + j);
                    return;
                }
            }
            // state: 0 unknown, 1 on the walk being followed, 2 done with a path, 3 done without one
            std::vector<uint8_t> state(static_cast<size_t>(n), 0);
            std::vector<int64_t> walk;
            state[static_cast<size_t>(i)] = 2;
            d[i] = 0.0;
            auto fail = [&](int64_t k, int64_t col, int64_t val) {
                kind[static_cast<size_t>(i)] = k;
                value[static_cast<size_t>(i)] = val;
                first.offer(i * n + col);
            };
            for (int64_t j = 0; j < n; ++j) {
                if (state[static_cast<size_t>(j)] >= 2) continue;
                walk.clear();
                int64_t x = j;
                for (;;) {
                    const uint8_t st = state[static_cast<size_t>(x)];
                    if (st == 1) return fail(3, x, x);  // the predecessors go round
                    if (st >= 2) break;
                    state[static_cast<size_t>(x)] = 1;
                    walk.push_back(x);
                    if (p[x] == kNoNode) break;
                    x = p[x];
                }
                if (state[static_cast<size_t>(x)] == 1) {  // x has no predecessor and is not i
                    if (walk.size() > 1) return fail(4, j, x);
                    state[static_cast<size_t>(x)] = 3;
                    d[x] = null;
                    continue;
                }
                if (state[static_cast<size_t>(x)] == 3) return fail(4, j, x);
                for (size_t q = walk.size(); q-- > 0;) {  // from the end the path reached, toward j
                    const int64_t v = walk[q], u = p[v];
                    double w;
                    if (!paths::edge_weight(g, u, v, w)) return fail(2, v, u);
                    d[v] = d[u] + w;
                    state[static_cast<size_t>(v)] = 2;
                }
            }
        };
        if (big && n > 1) parallel::parallel_for(n, row);
        else
            for (int64_t i = 0; i < n; ++i) row(i);
    }
    if (first.any()) {
        const int64_t at = first.row(), i = at / n;
        return nb::make_tuple(
            nb::none(), nb::make_tuple(kind[static_cast<size_t>(i)], i, at % n, value[static_cast<size_t>(i)])
        );
    }
    return nb::make_tuple(od, nb::none());
}

nb::object reconstruct(nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, nb::handle ph, bool directed) {
    const View g0 = view_arg(ip, ix, dh, n, "_reconstruct");
    if (!g0.data) throw nb::type_error("_reconstruct(): wrong data");
    const NDArray& pa = index_matrix(ph, 1, "_reconstruct");
    if (pa.shape[0] != n) throw nb::value_error("_reconstruct(): wrong predecessors");
    const int64_t* pred = reinterpret_cast<const int64_t*>(pa.data);
    const bool big = n + g0.nnz() >= kParallelWork;
    // the first predecessor that is not an edge of the graph: (kind, v, value), kind 0 when there is none
    int64_t kind = 0, bad_v = 0, bad_value = 0, edges = 0;
    std::vector<double> wv(static_cast<size_t>(n), 0.0);
    {
        const parallel::Unlocked unlocked(big);
        for (int64_t v = 0; v < n && !kind; ++v) {
            if (pred[v] != kNoNode && (pred[v] < 0 || pred[v] >= n)) kind = 1, bad_v = v, bad_value = pred[v];
            else if (pred[v] == v) kind = 5, bad_v = v, bad_value = v;
        }
        if (!kind) {
            Owned own;
            const WGraph g = paths::reading(g0, directed, true, own);
            for (int64_t v = 0; v < n; ++v) {
                if (pred[v] == kNoNode) continue;
                if (!paths::edge_weight(g, pred[v], v, wv[static_cast<size_t>(v)])) {
                    kind = 2, bad_v = v, bad_value = pred[v];
                    break;
                }
                ++edges;
            }
        }
    }
    if (kind) return nb::make_tuple(nb::none(), nb::make_tuple(kind, bad_v, bad_value));
    nb::object tp = new_array1(DType::Int64, n + 1), ti = new_array1(DType::Int64, edges),
               td = new_array1(DType::Float64, edges);
    int64_t* P = ptr<int64_t>(tp);
    int64_t* I = ptr<int64_t>(ti);
    double* W = ptr<double>(td);
    {
        const parallel::Unlocked unlocked(big);
        std::fill(P, P + n + 1, 0);
        for (int64_t v = 0; v < n; ++v)
            if (pred[v] != kNoNode) ++P[pred[v] + 1];
        for (int64_t i = 0; i < n; ++i) P[i + 1] += P[i];
        std::vector<int64_t> next(P, P + n);
        for (int64_t v = 0; v < n; ++v) {  // children placed in ascending order: canonical rows
            if (pred[v] == kNoNode) continue;
            const int64_t at = next[static_cast<size_t>(pred[v])]++;
            I[at] = v;
            W[at] = wv[static_cast<size_t>(v)];
        }
    }
    return nb::make_tuple(tp, ti, td);
}

}  // namespace

void bind_paths(nb::module_& m) {
    m.def("_scan", &scan, nb::arg("data"));
    m.def(
        "_paths", &run_paths, nb::arg("indptr"), nb::arg("indices"), nb::arg("data").none(), nb::arg("n"),
        nb::arg("sources").none(), nb::arg("method"), nb::arg("directed"), nb::arg("preds"), nb::arg("limit"),
        nb::arg("min_only"), nb::arg("flat"), nb::arg("negative")
    );
    m.def(
        "_floyd_warshall", &run_floyd_warshall, nb::arg("indptr"), nb::arg("indices"), nb::arg("data").none(),
        nb::arg("n"), nb::arg("directed"), nb::arg("preds")
    );
    m.def(
        "_construct", &construct, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("n"), nb::arg("pred"),
        nb::arg("directed"), nb::arg("null_value")
    );
    m.def(
        "_reconstruct", &reconstruct, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("n"),
        nb::arg("pred"), nb::arg("directed")
    );
}

}  // namespace sd::graph
