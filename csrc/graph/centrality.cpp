// PageRank and betweenness centrality of sd.graph (docs/design/graph.md §2.6, §3.7; milestone GR5). The
// Python layer (src/stardust/graph/_centrality.py) checks every argument.
//
//   _pagerank(indptr, indices, data, n, directed, alpha, personalization, dangling, tol) -> (x, iterations)
//       The iterates x_{k+1} = T(x_k) = alpha (G^T x_k) + (1 - alpha) v from x_0 = v, G the transition
//       matrix (each row of positive out-weight divided by it; data None: every edge weighs 1; a row of
//       out-weight 0 replaced by the dangling distribution), v and the dangling distribution the given
//       non-negative arrays divided by their sums. T contracts by alpha in l1, so ||x - x*|| <= (alpha
//       ||x_{k+1} - x_k|| + eps) / (1 - alpha) for the computed x_{k+1} (eps a bound of one step's
//       rounding): the iteration stops once that bound is at most tol, or at the a priori count where
//       2 alpha^k <= tol (decisions GR56, GR57). Undirected (directed false): the undirected reading
//       (docs/design/graph.md §2.1), a self loop once.
//   _betweenness(indptr, indices, data, n, directed, normalized, endpoints) -> float64 (n,)
//       Brandes's betweenness on the shortest-path DAG of every source: the distances of the shortest
//       paths' definition (docs/design/graph.md §3.2: left-to-right sums; data None: edge counts), the
//       DAG edges u -> v with fl(d(u) + w) == d(v) and (d(u), h(u)) before (d(v), h(v)) (h the breadth-first
//       edge count over those tight edges: it orders the absorbed steps of equal distance), the path counts sigma
//       pulled along the in-edges in ascending order, the dependencies delta pulled along the out-edges
//       in ascending order (decision GR58); networkx's endpoints and scaling.
//
// Thread independence (§3.8): PageRank's rows are summed one row per task in blocks fixed by the size,
// each row by the shared pairwise tree, the vector sums by the same tree cut at fixed leaves; the sources
// of the betweenness are tasks of a size fixed by n, whose partial sums are added in task order.
#include "array/ndarray.hpp"
#include "base/pairwise.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "graph/paths.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using paths::Owned;
using paths::WGraph;
using std::int64_t;

constexpr double kU = 0x1p-53;

double gamma(int64_t k) { return static_cast<double>(k) * kU / (1.0 - static_cast<double>(k) * kU); }

[[noreturn]] void wrong(const char* func, const char* what) {
    throw nb::type_error((std::string(func) + "(): wrong " + what).c_str());
}

const double* float_vector(nb::handle o, int64_t n, const char* func, const char* what) {
    if (!is_ndarray(o)) wrong(func, what);
    const NDArray& a = as_ndarray(o);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != DType::Float64 || a.shape[0] != n) wrong(func, what);
    return reinterpret_cast<const double*>(a.data);
}

nb::object new_vector(DType t, int64_t n) {
    const int64_t shape[] = {n};
    return wrap(new_array(t, shape));
}

// ---- sums ----

// The depth of the pairwise tree of base/pairwise.hpp over m terms: the most roundings any term goes
// through (a leaf of fewer than 16 terms is one chain from 0.0, whose first addition is exact; a longer
// leaf 16 chains, their 4-level fold and a sequential tail).
int64_t pairwise_depth(int64_t m) {
    if (m <= pairwise::kBlock) {
        if (m < pairwise::kLanes) return std::max<int64_t>(m - 1, 0);
        return (m / pairwise::kLanes - 1) + 4 + (m % pairwise::kLanes);
    }
    const int64_t a = pairwise::split(m);
    return 1 + std::max(pairwise_depth(a), pairwise_depth(m - a));
}

// get(0) + ... + get(n - 1) by the shared pairwise tree; the subtrees of kSumTask terms on the pool when
// split (pairwise::plan and combine: the serial tree's bits).
constexpr int64_t kSumTask = 1 << 16;

template <class Get>
double tree_sum(int64_t n, const Get& get, bool split) {
    auto leaf_sum = [&](int64_t b, int64_t m) { return pairwise::leaf(m, [&](int64_t i) { return get(b + i); }); };
    if (!split || n <= kSumTask) return pairwise::tree(0, n, leaf_sum);
    const pairwise::Leaves leaves = pairwise::plan(n, kSumTask, false);
    std::vector<double> parts(leaves.size());
    parallel::parallel_for(static_cast<int64_t>(leaves.size()), [&](int64_t t) {
        const auto [b, m] = leaves[static_cast<size_t>(t)];
        parts[static_cast<size_t>(t)] = pairwise::tree(b, m, leaf_sum);
    });
    return pairwise::combine(n, kSumTask, parts);
}

// ---- PageRank ----

// Row blocks of about kRowWork entries and rows each, fixed by the pattern alone.
constexpr int64_t kRowWork = 1 << 15;

std::vector<int64_t> row_blocks(int64_t n, const int64_t* ip) {
    std::vector<int64_t> cut{0};
    int64_t since = 0;
    for (int64_t i = 0; i < n; ++i) {
        since += ip[i + 1] - ip[i] + 1;
        if (since >= kRowWork) {
            cut.push_back(i + 1);
            since = 0;
        }
    }
    if (cut.back() != n) cut.push_back(n);
    return cut;
}

// The transition probability of an edge of weight w from a node of out-weight `out`: 0 from a dangling node
// (its stored weights all 0; its mass moves by the dangling distribution instead).
inline double share(double w, double out) { return out > 0.0 ? w / out : 0.0; }

nb::object pagerank(
    nb::handle ip,
    nb::handle ix,
    nb::handle dh,
    int64_t n,
    bool directed,
    double alpha,
    nb::handle ph,
    nb::handle qh,
    double tol
) {
    const char* func = "_pagerank";
    const View g = view_arg(ip, ix, dh, n, func);
    const double* pers = float_vector(ph, n, func, "personalization");
    const double* dang = float_vector(qh, n, func, "dangling");
    if (!(alpha >= 0.0 && alpha < 1.0) || !(tol > 0.0)) throw nb::value_error("_pagerank(): wrong alpha or tol");
    const bool weighted = g.data != nullptr;
    nb::object ox = new_vector(DType::Float64, n);
    double* x = reinterpret_cast<double*>(as_ndarray(ox).data);
    int64_t iterations = 0;
    const bool split = parallel::available() && n + g.nnz() >= kParallelWork;
    // Directed: the in-edges of every node are the transpose's rows. On the pool, sparse's (parallel)
    // transpose, made with the GIL held, whose weights turn into transition probabilities in place (data
    // None: the pattern's); on one thread one counting sort that writes the probabilities as it goes. Both
    // are the canonical transpose, so the rows and their terms are the same.
    std::optional<Transposed> tr;
    if (directed && split)
        tr.emplace(transpose(g, as_ndarray(ip), as_ndarray(ix), as_ndarray(weighted ? dh : ix), weighted));
    {
        const parallel::Unlocked unlocked(n + g.nnz() >= kParallelWork);
        auto over = [&](const std::vector<int64_t>& cut, const auto& fn) {  // fn(t, lo, hi) per row block t
            const int64_t nblk = static_cast<int64_t>(cut.size()) - 1;
            auto task = [&](int64_t t) { fn(t, cut[static_cast<size_t>(t)], cut[static_cast<size_t>(t + 1)]); };
            if (split && nblk > 1) parallel::parallel_for(nblk, task);
            else
                for (int64_t t = 0; t < nblk; ++t) task(t);
        };
        Owned own;
        const WGraph r = paths::reading(g, directed, weighted, own);
        const int64_t nnz = r.nnz();
        // The out-weights (data None: the out-degrees) and the dangling nodes; a row per task block.
        std::vector<double> out(static_cast<size_t>(n));
        const std::vector<int64_t> out_cut = row_blocks(n, r.ip);
        std::vector<int64_t> block_depth(out_cut.size(), 0);
        over(out_cut, [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t deepest = 0;
            for (int64_t j = lo; j < hi; ++j) {
                const int64_t b = r.ip[j], m = r.ip[j + 1] - b;
                double s;
                if (weighted) {
                    s = pairwise::tree(0, m, [&](int64_t bb, int64_t mm) {
                        return pairwise::leaf(mm, [&](int64_t i) { return r.w[b + bb + i]; });
                    });
                    deepest = std::max(deepest, pairwise_depth(m));
                } else {
                    s = static_cast<double>(m);
                }
                out[static_cast<size_t>(j)] = s;
            }
            block_depth[static_cast<size_t>(t)] = deepest;
        });
        const int64_t depth_out = *std::max_element(block_depth.begin(), block_depth.end());  // the deepest sum
        std::vector<int64_t> dangling;
        for (int64_t j = 0; j < n; ++j)
            if (out[static_cast<size_t>(j)] == 0.0) dangling.push_back(j);
        // The in-edges of every node (the transpose; the undirected reading is its own) with the transition
        // probabilities w / out(j) of their sources j (weighted only: unweighted divides x(j) by the degree).
        std::vector<double> own_tw;
        std::vector<int64_t> tp, tj;
        const double* tw = nullptr;
        const int64_t* in_p = r.ip;
        const int64_t* in_j = r.ix;
        if (directed && !tr) {
            tp.assign(static_cast<size_t>(n + 1), 0);
            tj.resize(static_cast<size_t>(nnz));
            if (weighted) own_tw.resize(static_cast<size_t>(nnz));
            for (int64_t k = 0; k < nnz; ++k) ++tp[static_cast<size_t>(r.ix[k] + 1)];
            for (int64_t i = 0; i < n; ++i) tp[static_cast<size_t>(i + 1)] += tp[static_cast<size_t>(i)];
            std::vector<int64_t> next(tp.begin(), tp.end() - 1);
            for (int64_t j = 0; j < n; ++j)
                for (int64_t k = r.ip[j]; k < r.ip[j + 1]; ++k) {
                    const int64_t at = next[static_cast<size_t>(r.ix[k])]++;
                    tj[static_cast<size_t>(at)] = j;
                    if (weighted) own_tw[static_cast<size_t>(at)] = share(r.w[k], out[static_cast<size_t>(j)]);
                }
            in_p = tp.data();
            in_j = tj.data();
            tw = weighted ? own_tw.data() : nullptr;
        } else if (directed) {
            in_p = tr->view.indptr;
            in_j = tr->view.indices;
        }
        if (weighted && !tw) {
            double* p = nullptr;
            const double* w = nullptr;
            if (directed) {
                p = reinterpret_cast<double*>(tr->arrays.data.data);  // the transpose's weights, ours to change
                w = p;
            } else {
                own_tw.resize(static_cast<size_t>(nnz));
                p = own_tw.data();
                w = r.w;
            }
            over(row_blocks(n, in_p), [&](int64_t, int64_t lo, int64_t hi) {
                for (int64_t k = in_p[lo]; k < in_p[hi]; ++k) p[k] = share(w[k], out[static_cast<size_t>(in_j[k])]);
            });
            tw = p;
        }
        int64_t depth_in = 0;
        for (int64_t i = 0; i < n; ++i) depth_in = std::max(depth_in, pairwise_depth(in_p[i + 1] - in_p[i]));
        // v and the dangling distribution, each divided by its pairwise sum; c = (1 - alpha) v.
        const double psum = tree_sum(n, [&](int64_t i) { return pers[i]; }, split);
        const double qsum = tree_sum(n, [&](int64_t i) { return dang[i]; }, split);
        std::vector<double> v(static_cast<size_t>(n)), d(static_cast<size_t>(n)), c(static_cast<size_t>(n));
        const double beta = 1.0 - alpha;
        for (int64_t i = 0; i < n; ++i) {
            v[static_cast<size_t>(i)] = pers[i] / psum;
            d[static_cast<size_t>(i)] = dang[i] / qsum;
            c[static_cast<size_t>(i)] = beta * v[static_cast<size_t>(i)];
        }
        // The rounding of one step, relative to each entry of T(x) (every term is non-negative): a row term
        // (weighted: w / out(j) then times x(j); else x(j) / degree) through its row's tree, then the
        // dangling term added, times alpha, plus c; the dangling term D d(i) (D the tree sum over the
        // dangling nodes, d(i) divided by its sum) the same three; c from v(i) = p(i) / sum.
        const int64_t hn = pairwise_depth(n);
        const int64_t term = weighted ? depth_out + 2 : 1;
        const int64_t hd = pairwise_depth(static_cast<int64_t>(dangling.size()));
        const int64_t K = std::max({term + depth_in + 3, hd + hn + 5, hn + 4});
        const double gK = gamma(K), margin = 1.0 + 0x1p-40;
        // The a priori count: ||x_k - x*|| <= 2 alpha^k + eps / (1 - alpha), so 2 alpha^k <= tol is enough.
        int64_t kmax = 1;
        if (alpha > 0.0) {
            const double k = std::ceil((std::log(tol) - std::log(2.0)) / std::log(alpha)) + 1.0;
            kmax = k > 1.0 ? (k < 9.0e15 ? static_cast<int64_t>(k) : int64_t{9000000000000000}) : 1;
        }
        std::vector<double> y(static_cast<size_t>(n)), z(weighted ? 0 : static_cast<size_t>(n));
        std::memcpy(x, v.data(), static_cast<size_t>(n) * sizeof(double));
        double* cur = x;  // the iterate x_k and the next one, swapped each step
        double* nxt = y.data();
        const std::vector<int64_t> cut = row_blocks(n, in_p);
        const int64_t nblocks = static_cast<int64_t>(cut.size()) - 1;
        const int64_t nd = static_cast<int64_t>(dangling.size());
        for (int64_t it = 1; n > 0; ++it) {
            const double D = tree_sum(nd, [&](int64_t i) { return cur[dangling[static_cast<size_t>(i)]]; }, split);
            if (!weighted)
                over(out_cut, [&](int64_t, int64_t lo, int64_t hi) {
                    for (int64_t j = lo; j < hi; ++j)
                        z[static_cast<size_t>(j)] =
                            out[static_cast<size_t>(j)] > 0.0 ? cur[j] / out[static_cast<size_t>(j)] : 0.0;
                });
            auto block = [&](int64_t t) {
                for (int64_t i = cut[static_cast<size_t>(t)]; i < cut[static_cast<size_t>(t + 1)]; ++i) {
                    const int64_t b = in_p[i], m = in_p[i + 1] - b;
                    double s = 0.0;
                    if (m < pairwise::kLanes) {  // a short row: the leaf's one chain, written out
                        if (weighted)
                            for (int64_t k = b; k < b + m; ++k) s += tw[k] * cur[in_j[k]];
                        else
                            for (int64_t k = b; k < b + m; ++k) s += z[static_cast<size_t>(in_j[k])];
                    } else if (weighted) {
                        s = pairwise::tree(0, m, [&](int64_t bb, int64_t mm) {
                            return pairwise::leaf(mm, [&](int64_t q) {
                                const int64_t k = b + bb + q;
                                return tw[k] * cur[in_j[k]];
                            });
                        });
                    } else {
                        s = pairwise::tree(0, m, [&](int64_t bb, int64_t mm) {
                            return pairwise::leaf(mm, [&](int64_t q) {
                                return z[static_cast<size_t>(in_j[b + bb + q])];
                            });
                        });
                    }
                    nxt[i] = alpha * (s + D * d[static_cast<size_t>(i)]) + c[static_cast<size_t>(i)];
                }
            };
            if (split && nblocks > 1) parallel::parallel_for(nblocks, block);
            else
                for (int64_t t = 0; t < nblocks; ++t) block(t);
            const double delta = tree_sum(n, [&](int64_t i) { return std::fabs(nxt[i] - cur[i]); }, split);
            const double total = tree_sum(n, [&](int64_t i) { return nxt[i]; }, split);
            std::swap(cur, nxt);
            iterations = it;
            // eps >= gamma_K ||T(x_k)||, ||T(x_k)|| <= total (1 + gamma_hn) / (1 - gamma_K); the exact l1
            // distance <= delta (1 + gamma_hn) / (1 - u).
            const double eps = gK * total * (1.0 + gamma(hn)) / (1.0 - gK);
            const double dist = delta * (1.0 + gamma(hn)) / (1.0 - kU);
            if ((alpha * dist + eps) * margin <= tol * beta || it >= kmax) break;
        }
        if (cur != x) std::memcpy(x, cur, static_cast<size_t>(n) * sizeof(double));
    }
    return nb::make_tuple(ox, iterations);
}

// ---- betweenness ----

// A path count with an exponent of its own (m in [0.5, 1) or 0, value m 2^e): the sums round like doubles
// (one rounding each) without overflowing. Doubles hold the counts below kWide; a source whose counts
// reach it is counted again with these, which give the same bits where both apply.
struct Wide {
    double m = 0.0;
    int64_t e = 0;
};

constexpr double kWide = 0x1p960;

inline Wide wide(double x) {
    Wide w;
    int ex = 0;
    w.m = std::frexp(x, &ex);
    w.e = ex;
    return w;
}

inline Wide add(const Wide& a, const Wide& b) {
    if (a.m == 0.0) return b;
    if (b.m == 0.0) return a;
    const int64_t e = std::max(a.e, b.e);
    const int64_t da = a.e - e, db = b.e - e;
    const double s = std::ldexp(a.m, static_cast<int>(std::max<int64_t>(da, -2000))) +
                     std::ldexp(b.m, static_cast<int>(std::max<int64_t>(db, -2000)));
    Wide w = wide(s);
    w.e += e;
    return w;
}

// sigma(u) / sigma(w) as a double (at most 1: u is one of w's predecessors).
inline double ratio(const Wide& a, const Wide& b) {
    return std::ldexp(a.m / b.m, static_cast<int>(std::clamp<int64_t>(a.e - b.e, -4000, 4000)));
}
inline double ratio(double a, double b) { return a / b; }
inline double add(double a, double b) { return a + b; }

// The scratch of one source.
struct Pass {
    paths::Work work;
    std::vector<double> d, delta, sigma;
    std::vector<Wide> wsigma;
    std::vector<int64_t> order, h;
    bool use_h = false;
    explicit Pass(int64_t n)
        : work(n),
          d(static_cast<size_t>(n)),
          delta(static_cast<size_t>(n)),
          sigma(static_cast<size_t>(n)),
          order(static_cast<size_t>(n)),
          h(static_cast<size_t>(n)) {}
};

// The reached nodes in a topological order of the DAG and their distances: a breadth-first search (data
// None: every DAG edge goes one level down) or Dijkstra on the indexed heap. Equal distances (absorbed
// weights, or ties of sums) are told apart by h, the breadth-first edge count over the tight edges from s
// (the shortest paths' predecessor rule, docs/design/graph.md §3.2): a run of equal distances is put in the
// order (h, index) and p.h is set; it stays empty (every reached distance distinct) otherwise. Returns how
// many nodes were reached.
int64_t distances(const WGraph& g, int64_t s, Pass& p) {
    const int64_t n = g.n;
    double* d = p.d.data();
    int64_t* order = p.order.data();
    std::fill(d, d + n, paths::kInf);
    d[s] = 0.0;
    p.use_h = false;
    int64_t found = 0;
    if (!g.w) {
        order[found++] = s;
        for (int64_t h = 0; h < found; ++h) {
            const int64_t u = order[h];
            const double next = d[u] + 1.0;
            for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k)
                if (d[g.ix[k]] == paths::kInf) {
                    d[g.ix[k]] = next;
                    order[found++] = g.ix[k];
                }
        }
        return found;
    }
    paths::Work& w = p.work;
    w.q_set(s, 0.0);
    while (!w.q_empty()) {
        const paths::Work::Item it = w.q_pop();
        const int64_t u = it.v;
        order[found++] = u;
        for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k) {
            const int64_t v = g.ix[k];
            const double t = it.key + g.w[k];
            if (t < d[v]) {
                d[v] = t;
                w.q_set(v, t);
            }
        }
    }
    bool runs = false;
    for (int64_t a = 1; a < found && !runs; ++a) runs = d[order[a]] == d[order[a - 1]];
    if (!runs) return found;
    // h by a breadth-first search over the tight edges, then each run of equal distances by (h, index).
    int64_t* h = p.h.data();
    int64_t* queue = p.work.order.data();
    for (int64_t a = 0; a < found; ++a) h[order[a]] = -1;
    h[s] = 0;
    int64_t qn = 0;
    queue[qn++] = s;
    for (int64_t q = 0; q < qn; ++q) {
        const int64_t u = queue[q];
        for (int64_t k = g.ip[u], e = g.ip[u + 1]; k < e; ++k) {
            const int64_t v = g.ix[k];
            if (h[v] < 0 && d[v] != paths::kInf && d[u] + g.w[k] == d[v]) {
                h[v] = h[u] + 1;
                queue[qn++] = v;
            }
        }
    }
    for (int64_t a = 0; a < found;) {
        int64_t b = a + 1;
        while (b < found && d[order[b]] == d[order[a]]) ++b;
        if (b - a > 1)
            std::sort(order + a, order + b, [&](int64_t x, int64_t y) {
                return h[x] < h[y] || (h[x] == h[y] && x < y);
            });
        a = b;
    }
    p.use_h = true;
    return found;
}

// Whether u -> v (weight wt) is an edge of the DAG: tight, and (d(u), h(u)) before (d(v), h(v)) (h only
// where distances repeat). v is reached.
inline bool dag_edge(const double* d, const int64_t* h, int64_t u, int64_t v, double wt) {
    const double du = d[u], dv = d[v];
    return du + wt == dv && (du < dv || (h && h[u] < h[v]));
}

// sigma along the in-edges (rows of `in`, ascending), delta along the out-edges (rows of `out`), into
// acc: delta(w) for w != s (endpoints: delta(w) + 1, and the reached count - 1 at s). False (nothing
// added) when a count reaches kWide in doubles.
template <class S>
bool source(
    const WGraph& out,
    const WGraph& in,
    int64_t s,
    int64_t found,
    bool endpoints,
    Pass& p,
    std::vector<S>& sigma,
    double* acc
) {
    const double* d = p.d.data();
    const int64_t* h = p.use_h ? p.h.data() : nullptr;
    const int64_t* order = p.order.data();
    double* delta = p.delta.data();
    S one;
    if constexpr (std::is_same_v<S, double>) one = 1.0;
    else one = wide(1.0);
    sigma[static_cast<size_t>(s)] = one;
    for (int64_t a = 1; a < found; ++a) {
        const int64_t v = order[a];
        S t{};
        for (int64_t k = in.ip[v], e = in.ip[v + 1]; k < e; ++k) {
            const int64_t u = in.ix[k];
            if (dag_edge(d, h, u, v, in.weight(k))) t = add(t, sigma[static_cast<size_t>(u)]);
        }
        if constexpr (std::is_same_v<S, double>)
            if (!(t < kWide)) return false;
        sigma[static_cast<size_t>(v)] = t;
    }
    for (int64_t a = found; a-- > 0;) {
        const int64_t u = order[a];
        double t = 0.0;
        for (int64_t k = out.ip[u], e = out.ip[u + 1]; k < e; ++k) {
            const int64_t w = out.ix[k];
            if (d[w] != paths::kInf && dag_edge(d, h, u, w, out.weight(k)))
                t += ratio(sigma[static_cast<size_t>(u)], sigma[static_cast<size_t>(w)]) * (1.0 + delta[w]);
        }
        delta[u] = t;
    }
    for (int64_t a = 1; a < found; ++a) {
        const int64_t w = order[a];
        acc[w] += endpoints ? delta[w] + 1.0 : delta[w];
    }
    if (endpoints) acc[s] += static_cast<double>(found - 1);
    return true;
}

nb::object betweenness(
    nb::handle ip, nb::handle ix, nb::handle dh, int64_t n, bool directed, bool normalized, bool endpoints
) {
    const View g = view_arg(ip, ix, dh, n, "_betweenness");
    const bool weighted = g.data != nullptr;
    nb::object ob = new_vector(DType::Float64, n);
    double* bc = reinterpret_cast<double*>(as_ndarray(ob).data);
    std::fill(bc, bc + n, 0.0);
    {
        const parallel::Unlocked unlocked(n * (n + g.nnz()) >= kParallelWork);
        Owned own;
        const WGraph out = paths::reading(g, directed, weighted, own);
        Owned tr;  // the in-edges: the transpose of a directed graph (the undirected reading is its own)
        WGraph in = out;
        if (directed) {
            const int64_t nnz = out.nnz();
            tr.ip.assign(static_cast<size_t>(n + 1), 0);
            tr.ix.resize(static_cast<size_t>(nnz));
            if (weighted) tr.w.resize(static_cast<size_t>(nnz));
            for (int64_t k = 0; k < nnz; ++k) ++tr.ip[static_cast<size_t>(out.ix[k] + 1)];
            for (int64_t i = 0; i < n; ++i) tr.ip[static_cast<size_t>(i + 1)] += tr.ip[static_cast<size_t>(i)];
            std::vector<int64_t> next(tr.ip.begin(), tr.ip.end() - 1);
            for (int64_t j = 0; j < n; ++j)
                for (int64_t k = out.ip[j]; k < out.ip[j + 1]; ++k) {
                    const int64_t at = next[static_cast<size_t>(out.ix[k])]++;
                    tr.ix[static_cast<size_t>(at)] = j;
                    if (weighted) tr.w[static_cast<size_t>(at)] = out.w[k];
                }
            in = WGraph{n, tr.ip.data(), tr.ix.data(), weighted ? tr.w.data() : nullptr};
        }
        // Tasks of `per` sources (fixed by n), run `wave` at a time; their partial sums are added to bc in
        // task order, so neither the wave nor the thread count changes a bit.
        const int64_t per = std::clamp<int64_t>((n + 255) / 256, 1, 64);
        const int64_t ntasks = (n + per - 1) / per;
        const bool split = parallel::available() && n * (n + out.nnz()) >= kParallelWork;
        const int64_t threads = split ? parallel::num_threads() : 1;
        const int64_t cap = std::max<int64_t>(1, (int64_t{1} << 24) / std::max<int64_t>(n, 1));  // 128 MB of partials
        const int64_t wave = std::clamp<int64_t>(4 * threads, 1, std::min(cap, std::max<int64_t>(ntasks, 1)));
        std::vector<double> partial(static_cast<size_t>(wave * n));
        for (int64_t first = 0; first < ntasks; first += wave) {
            const int64_t count = std::min(wave, ntasks - first);
            auto task = [&](int64_t t) {
                double* acc = partial.data() + t * n;
                std::fill(acc, acc + n, 0.0);
                Pass p(n);
                const int64_t lo = (first + t) * per, hi = std::min(n, lo + per);
                for (int64_t s = lo; s < hi; ++s) {
                    const int64_t found = distances(out, s, p);
                    if (!source(out, in, s, found, endpoints, p, p.sigma, acc)) {
                        p.wsigma.resize(static_cast<size_t>(n));
                        source(out, in, s, found, endpoints, p, p.wsigma, acc);
                    }
                }
            };
            if (split && count > 1) parallel::parallel_for(count, task);
            else
                for (int64_t t = 0; t < count; ++t) task(t);
            for (int64_t t = 0; t < count; ++t) {
                const double* acc = partial.data() + t * n;
                for (int64_t v = 0; v < n; ++v) bc[v] += acc[v];
            }
        }
        // networkx's _rescale: N valid (s, t) pairs per v; none below 2 (every value is 0 then).
        const int64_t N = endpoints ? n : n - 1;
        double scale = 1.0;
        if (N >= 2) {
            if (normalized) scale = 1.0 / static_cast<double>(N * (N - 1));
            else if (!directed) scale = 0.5;
        }
        if (scale != 1.0)
            for (int64_t v = 0; v < n; ++v) bc[v] *= scale;
    }
    return ob;
}

}  // namespace

void bind_centrality(nb::module_& m) {
    m.def(
        "_pagerank", &pagerank, nb::arg("indptr"), nb::arg("indices"), nb::arg("data").none(), nb::arg("n"),
        nb::arg("directed"), nb::arg("alpha"), nb::arg("personalization"), nb::arg("dangling"), nb::arg("tol")
    );
    m.def(
        "_betweenness", &betweenness, nb::arg("indptr"), nb::arg("indices"), nb::arg("data").none(), nb::arg("n"),
        nb::arg("directed"), nb::arg("normalized"), nb::arg("endpoints")
    );
}

}  // namespace sd::graph
