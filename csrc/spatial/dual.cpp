// The dual-tree queries of sd.spatial's KDTree (docs/design/spatial.md §3.4, §3.5, §3.6; milestone SPT4):
// query_ball_tree (and sparse_distance_matrix), query_pairs and count_neighbors. The Python layer
// (src/stardust/spatial/_kdtree.py) checks every argument.
//
//   _ball_tree(A, B, r, p)    -> (indptr (A.n + 1), indices, distances): row i (a point of A, input order)
//                                holds every point j of B with d(A_i, B_j) <= r, in index order
//   _pairs(A, r, p)           -> (K, 2) int64: every (i, j), i < j, with d(A_i, A_j) <= r, sorted
//   _count(A, B, radii, order, p, wA, wB, cumulative)
//                             radii (R,) sorted nondecreasing, order (R,) int64 their places in the result;
//                             wA, wB None or (A.n,) / (B.n,) float64 -> (R,) int64 (both None) or float64:
//                             out[order[k]] = the count (or the weight wA_i wB_j summed) of the pairs with
//                             d <= radii[k] (cumulative) or radii[k - 1] < d <= radii[k] (k = 0: d <= radii[0])
//
// The two trees are walked together from node pairs: a pair whose near bound exceeds KdCut(r) is cut, and
// (counting) a pair whose far bound proves every point pair inside is added whole (KdPair, kdtree.hpp, with
// the proofs). Otherwise the larger node (A's on a tie) is split, and two leaves compare every point pair
// with the pair kernel of the metric of p: every returned distance and every boundary test is the kernel's
// value d^ <= r, so the results are the brute force's (radius_neighbors, cdist, pdist with the same kernel).
//
// Tasks (SPT25): each tree is cut into its nodes at a fixed depth (kDepth, or shallower leaves) and every
// pair of pieces (a, b) is a task, numbered a * (pieces of B) + b (query_pairs: a <= b only). The cut
// depends on the trees alone, never on the thread count; the lists are gathered row by row and sorted, and
// the weighted sums of the tasks are added in task order, so every result is the same bytes for any
// thread count (and run serially below the parallel threshold, the same cut). count_neighbors cuts less
// deep when there are many radii (the tasks' sums take tasks x (R + 1) doubles, at most kCountSlots).
#include "array/ndarray.hpp"
#include "base/buffer.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/kdtree.hpp"
#include "spatial/spatial.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

constexpr int64_t kParallelWork = 1 << 15;  // array.md §10.4: from here a call splits and drops the GIL
constexpr int64_t kWorkPerPoint = 64;       // the work units of one point per coordinate
constexpr int kDepth = 4;                   // the depth of the pieces: up to 16 per tree, 256 tasks
constexpr int64_t kCountSlots = 1 << 22;    // the doubles of count_neighbors' per-task sums (at most)
constexpr int64_t kListBytes = 64;          // a found pair of query_ball_tree while gathering, rounded up
constexpr int64_t kPairBytes = 32;          // a found pair of query_pairs while gathering

template <class F>
void run_tasks(int64_t ntasks, bool split, F&& fn) {
    if (!split || ntasks <= 1) {
        for (int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

[[noreturn]] void too_large(const char* func) {
    const std::string msg = std::string(func) + "(): the result is larger than this machine's physical memory";
    PyErr_SetString(PyExc_MemoryError, msg.c_str());
    throw nb::python_error();
}

nb::object new_array_of(DType t, std::initializer_list<int64_t> shape) {
    const std::vector<int64_t> s(shape);
    return wrap(new_array(t, s));
}

// The nodes at `depth` below `node` (or the leaves above it), in preorder.
void pieces(const KdTree& t, int64_t node, int depth, std::vector<int64_t>& out) {
    const KdNode& nd = t.nodes[static_cast<size_t>(node)];
    if (depth == 0 || nd.right < 0) {
        out.push_back(node);
        return;
    }
    pieces(t, node + 1, depth - 1, out);
    pieces(t, nd.right, depth - 1, out);
}

std::vector<int64_t> pieces(const KdTree& t, int depth) {
    std::vector<int64_t> out;
    pieces(t, 0, depth, out);
    return out;
}

bool split_of(const KdTree& A, const KdTree& B) {
    return (A.n + B.n) * std::max<int64_t>(A.m, 1) * kWorkPerPoint >= kParallelWork;
}

int64_t count_of(const KdTree& t, int64_t node) {
    const KdNode& nd = t.nodes[static_cast<size_t>(node)];
    return nd.end - nd.start;
}

// The common walk: from (na, nb), cut by `cut(na, nb)`, two leaves to `leaves(na, nb)`, else the larger
// node split (A's on a tie, a leaf never).
template <class Cut, class Leaves>
void walk(const KdTree& A, const KdTree& B, int64_t na, int64_t nb, Cut& cut, Leaves& leaves) {
    if (cut(na, nb)) return;
    const KdNode &a = A.nodes[static_cast<size_t>(na)], &b = B.nodes[static_cast<size_t>(nb)];
    const bool la = a.right < 0, lb = b.right < 0;
    if (la && lb) {
        leaves(na, nb);
    } else if (lb || (!la && a.end - a.start >= b.end - b.start)) {
        walk(A, B, na + 1, nb, cut, leaves);
        walk(A, B, a.right, nb, cut, leaves);
    } else {
        walk(A, B, na, nb + 1, cut, leaves);
        walk(A, B, na, b.right, cut, leaves);
    }
}

// ---- query_ball_tree --------------------------------------------------------------------------------

struct Trip {
    int64_t pos;  // the point of A, in A's leaf order
    int64_t j;    // the point of B, its input index
    double d;
};

template <class Dist>
nb::object ball_tree_with(const KdTree& A, const KdTree& B, const Dist& dist, const KdCut& cut, double r) {
    const std::vector<int64_t> pa = pieces(A, kDepth), pb = pieces(B, kDepth);
    const int64_t SA = static_cast<int64_t>(pa.size()), SB = static_cast<int64_t>(pb.size()), m = A.m;
    const bool split = split_of(A, B);
    const double T1 = cut(r), above = Dist::above(r);
    const int64_t budget = static_cast<int64_t>(physical_memory() / kListBytes);
    std::atomic<int64_t> total{0};
    std::atomic<bool> over{false};
    std::vector<std::vector<Trip>> found(static_cast<size_t>(SA * SB));
    // per piece of A: its rows' counts (leaf order) and their (j, d), each row in index order
    std::vector<std::vector<int64_t>> rows(static_cast<size_t>(SA));
    std::vector<std::vector<int64_t>> js(static_cast<size_t>(SA));
    std::vector<std::vector<double>> ds(static_cast<size_t>(SA));
    {
        const parallel::Unlocked unlocked(split);
        run_tasks(SA * SB, split, [&](int64_t task) {
            std::vector<Trip>& out = found[static_cast<size_t>(task)];
            KdPair pr(m);
            auto cutf = [&](int64_t na, int64_t nb) {
                if (over.load(std::memory_order_relaxed)) return true;
                pr.near(A.lo(na), A.hi(na), B.lo(nb), B.hi(nb), m);
                return kd_beyond(dist, pr.a.data(), pr.b.data(), T1);
            };
            auto leaves = [&](int64_t na, int64_t nb) {
                const KdNode &a = A.nodes[static_cast<size_t>(na)], &b = B.nodes[static_cast<size_t>(nb)];
                const size_t from = out.size();
                for (int64_t i = a.start; i < a.end; ++i) {
                    const double* x = A.data.data() + i * m;
                    const double* y = B.data.data() + b.start * m;
                    for (int64_t q = b.start; q < b.end; ++q, y += m) {
                        const double s = dist.raw(x, y);
                        if (Dist::plain(s) && s > above) continue;
                        const double d = dist.finish(s, x, y);
                        if (d <= r) out.push_back({i, B.idx[static_cast<size_t>(q)], d});
                    }
                }
                const int64_t c = static_cast<int64_t>(out.size() - from);
                if (c && total.fetch_add(c, std::memory_order_relaxed) + c > budget) over.store(true);
            };
            walk(A, B, pa[static_cast<size_t>(task / SB)], pb[static_cast<size_t>(task % SB)], cutf, leaves);
        });
        if (!over.load()) {
            run_tasks(SA, split, [&](int64_t a) {  // gather each piece of A's rows, in index order
                const KdNode& nd = A.nodes[static_cast<size_t>(pa[static_cast<size_t>(a)])];
                const int64_t len = nd.end - nd.start;
                std::vector<int64_t>& cnt = rows[static_cast<size_t>(a)];
                cnt.assign(static_cast<size_t>(len + 1), 0);
                int64_t tot = 0;
                for (int64_t b = 0; b < SB; ++b)
                    for (const Trip& e : found[static_cast<size_t>(a * SB + b)]) {
                        ++cnt[static_cast<size_t>(e.pos - nd.start + 1)];
                        ++tot;
                    }
                for (int64_t k = 0; k < len; ++k) cnt[static_cast<size_t>(k + 1)] += cnt[static_cast<size_t>(k)];
                std::vector<int64_t>& J = js[static_cast<size_t>(a)];
                std::vector<double>& D = ds[static_cast<size_t>(a)];
                J.resize(static_cast<size_t>(tot));
                D.resize(static_cast<size_t>(tot));
                std::vector<int64_t> at(cnt.begin(), cnt.end() - 1);
                for (int64_t b = 0; b < SB; ++b) {
                    std::vector<Trip>& f = found[static_cast<size_t>(a * SB + b)];
                    for (const Trip& e : f) {
                        const int64_t k = at[static_cast<size_t>(e.pos - nd.start)]++;
                        J[static_cast<size_t>(k)] = e.j;
                        D[static_cast<size_t>(k)] = e.d;
                    }
                    std::vector<Trip>().swap(f);
                }
                std::vector<std::pair<int64_t, double>> row;
                for (int64_t k = 0; k < len; ++k) {
                    const int64_t s0 = cnt[static_cast<size_t>(k)], s1 = cnt[static_cast<size_t>(k + 1)];
                    if (s1 - s0 < 2) continue;
                    row.clear();
                    for (int64_t e = s0; e < s1; ++e)
                        row.emplace_back(J[static_cast<size_t>(e)], D[static_cast<size_t>(e)]);
                    std::sort(row.begin(), row.end(), [](const auto& u, const auto& v) { return u.first < v.first; });
                    for (int64_t e = s0; e < s1; ++e) {
                        J[static_cast<size_t>(e)] = row[static_cast<size_t>(e - s0)].first;
                        D[static_cast<size_t>(e)] = row[static_cast<size_t>(e - s0)].second;
                    }
                }
            });
        }
    }
    if (over.load()) too_large("query_ball_tree");
    // the rows in input order: row idx[pos] is the piece's row of pos
    std::vector<int64_t> indptr(static_cast<size_t>(A.n + 1), 0);
    for (int64_t a = 0; a < SA; ++a) {
        const KdNode& nd = A.nodes[static_cast<size_t>(pa[static_cast<size_t>(a)])];
        const std::vector<int64_t>& cnt = rows[static_cast<size_t>(a)];
        for (int64_t k = 0; k < nd.end - nd.start; ++k)
            indptr[static_cast<size_t>(A.idx[static_cast<size_t>(nd.start + k)] + 1)] =
                cnt[static_cast<size_t>(k + 1)] - cnt[static_cast<size_t>(k)];
    }
    for (int64_t i = 0; i < A.n; ++i) indptr[static_cast<size_t>(i + 1)] += indptr[static_cast<size_t>(i)];
    const int64_t nnz = indptr[static_cast<size_t>(A.n)];
    nb::object op = new_array_of(DType::Int64, {A.n + 1}), oi = new_array_of(DType::Int64, {nnz}),
               od = new_array_of(DType::Float64, {nnz});
    std::copy(indptr.begin(), indptr.end(), reinterpret_cast<int64_t*>(as_ndarray(op).data));
    int64_t* I = reinterpret_cast<int64_t*>(as_ndarray(oi).data);
    double* D = reinterpret_cast<double*>(as_ndarray(od).data);
    {
        const bool par = split && nnz >= kParallelWork;
        const parallel::Unlocked unlocked(par);
        run_tasks(SA, par, [&](int64_t a) {
            const KdNode& nd = A.nodes[static_cast<size_t>(pa[static_cast<size_t>(a)])];
            const std::vector<int64_t>& cnt = rows[static_cast<size_t>(a)];
            const std::vector<int64_t>& J = js[static_cast<size_t>(a)];
            const std::vector<double>& DD = ds[static_cast<size_t>(a)];
            for (int64_t k = 0; k < nd.end - nd.start; ++k) {
                const int64_t s0 = cnt[static_cast<size_t>(k)], s1 = cnt[static_cast<size_t>(k + 1)];
                const int64_t dst = indptr[static_cast<size_t>(A.idx[static_cast<size_t>(nd.start + k)])];
                std::copy(J.begin() + s0, J.begin() + s1, I + dst);
                std::copy(DD.begin() + s0, DD.begin() + s1, D + dst);
            }
        });
    }
    return nb::make_tuple(op, oi, od);
}

// ---- query_pairs --------------------------------------------------------------------------------------

struct Pair {
    int64_t i, j;  // input indices, i < j
};

template <class Dist>
nb::object pairs_with(const KdTree& A, const Dist& dist, const KdCut& cut, double r) {
    const std::vector<int64_t> pa = pieces(A, kDepth);
    const int64_t S = static_cast<int64_t>(pa.size()), m = A.m;
    std::vector<std::pair<int64_t, int64_t>> tasks;  // the pieces (a, b), a <= b
    for (int64_t a = 0; a < S; ++a)
        for (int64_t b = a; b < S; ++b) tasks.emplace_back(a, b);
    const int64_t ntasks = static_cast<int64_t>(tasks.size());
    const bool split = split_of(A, A);
    const double T1 = cut(r), above = Dist::above(r);
    const int64_t budget = static_cast<int64_t>(physical_memory() / kPairBytes);
    std::atomic<int64_t> total{0};
    std::atomic<bool> over{false};
    std::vector<std::vector<Pair>> found(static_cast<size_t>(ntasks));
    {
        const parallel::Unlocked unlocked(split);
        run_tasks(ntasks, split, [&](int64_t task) {
            std::vector<Pair>& out = found[static_cast<size_t>(task)];
            KdPair pr(m);
            auto test = [&](int64_t p, int64_t q) {
                const double* x = A.data.data() + p * m;
                const double* y = A.data.data() + q * m;
                const double s = dist.raw(x, y);
                if (Dist::plain(s) && s > above) return;
                if (dist.finish(s, x, y) <= r) {
                    const int64_t i = A.idx[static_cast<size_t>(p)], j = A.idx[static_cast<size_t>(q)];
                    out.push_back(i < j ? Pair{i, j} : Pair{j, i});
                }
            };
            auto account = [&](size_t from) {
                const int64_t c = static_cast<int64_t>(out.size() - from);
                if (c && total.fetch_add(c, std::memory_order_relaxed) + c > budget) over.store(true);
            };
            auto cutf = [&](int64_t na, int64_t nb) {
                if (over.load(std::memory_order_relaxed)) return true;
                pr.near(A.lo(na), A.hi(na), A.lo(nb), A.hi(nb), m);
                return kd_beyond(dist, pr.a.data(), pr.b.data(), T1);
            };
            auto leaves = [&](int64_t na, int64_t nb) {  // two disjoint leaves
                const KdNode &a = A.nodes[static_cast<size_t>(na)], &b = A.nodes[static_cast<size_t>(nb)];
                const size_t from = out.size();
                for (int64_t p = a.start; p < a.end; ++p)
                    for (int64_t q = b.start; q < b.end; ++q) test(p, q);
                account(from);
            };
            // a node with itself: its two halves with themselves and with each other (no bound: distance 0)
            auto self = [&](auto&& me, int64_t node) -> void {
                if (over.load(std::memory_order_relaxed)) return;
                const KdNode& nd = A.nodes[static_cast<size_t>(node)];
                if (nd.right < 0) {
                    const size_t from = out.size();
                    for (int64_t p = nd.start; p < nd.end; ++p)
                        for (int64_t q = p + 1; q < nd.end; ++q) test(p, q);
                    account(from);
                    return;
                }
                me(me, node + 1);
                walk(A, A, node + 1, nd.right, cutf, leaves);
                me(me, nd.right);
            };
            const auto [a, b] = tasks[static_cast<size_t>(task)];
            if (a == b) self(self, pa[static_cast<size_t>(a)]);
            else walk(A, A, pa[static_cast<size_t>(a)], pa[static_cast<size_t>(b)], cutf, leaves);
        });
    }
    if (over.load()) too_large("query_pairs");
    // rows by i, then each row by j (SPT77): the pairs are cut by blocks of kRowBlock rows straight into the
    // output (a count per task and block; each task's place in a block from the prefix sums in (block, task)
    // order), then each block is put in (i, j) order on its own (a counting sort by i, then each row by j).
    // Every step but the prefix sums runs on the pool; the result is the sorted list whatever the order.
    constexpr int64_t kRowBlock = 1 << 12;
    const int64_t ntask = static_cast<int64_t>(found.size());
    const int64_t nblk = std::max<int64_t>(1, (A.n + kRowBlock - 1) / kRowBlock);
    int64_t K = 0;
    for (const auto& f : found) K += static_cast<int64_t>(f.size());
    nb::object out = new_array_of(DType::Int64, {K, 2});
    int64_t* O = reinterpret_cast<int64_t*>(as_ndarray(out).data);
    {
        const bool par = split && K >= kParallelWork;
        const parallel::Unlocked unlocked(par);
        std::vector<int64_t> at(static_cast<size_t>(ntask * nblk), 0);
        run_tasks(ntask, par, [&](int64_t t) {
            int64_t* c = at.data() + t * nblk;
            for (const Pair& e : found[static_cast<size_t>(t)]) ++c[e.i / kRowBlock];
        });
        std::vector<int64_t> bstart(static_cast<size_t>(nblk + 1), 0);
        int64_t acc = 0;
        for (int64_t b = 0; b < nblk; ++b) {
            bstart[static_cast<size_t>(b)] = acc;
            for (int64_t t = 0; t < ntask; ++t) {
                const int64_t c = at[static_cast<size_t>(t * nblk + b)];
                at[static_cast<size_t>(t * nblk + b)] = acc;
                acc += c;
            }
        }
        bstart[static_cast<size_t>(nblk)] = acc;
        run_tasks(ntask, par, [&](int64_t t) {
            int64_t* c = at.data() + t * nblk;
            std::vector<Pair>& f = found[static_cast<size_t>(t)];
            for (const Pair& e : f) {
                const int64_t k = c[e.i / kRowBlock]++;
                O[2 * k] = e.i;
                O[2 * k + 1] = e.j;
            }
            std::vector<Pair>().swap(f);
        });
        run_tasks(nblk, par, [&](int64_t b) {
            const int64_t s0 = bstart[static_cast<size_t>(b)], s1 = bstart[static_cast<size_t>(b + 1)];
            if (s1 - s0 < 2) return;
            const int64_t r0 = b * kRowBlock, rows = std::min(kRowBlock, A.n - r0);
            std::vector<int64_t> pos(static_cast<size_t>(rows + 1), 0), tmp(static_cast<size_t>(2 * (s1 - s0)));
            for (int64_t k = s0; k < s1; ++k) ++pos[static_cast<size_t>(O[2 * k] - r0 + 1)];
            for (int64_t r = 0; r < rows; ++r) pos[static_cast<size_t>(r + 1)] += pos[static_cast<size_t>(r)];
            std::vector<int64_t> put(pos.begin(), pos.end() - 1);
            for (int64_t k = s0; k < s1; ++k) {
                const int64_t q = put[static_cast<size_t>(O[2 * k] - r0)]++;
                tmp[static_cast<size_t>(2 * q)] = O[2 * k];
                tmp[static_cast<size_t>(2 * q + 1)] = O[2 * k + 1];
            }
            std::copy(tmp.begin(), tmp.end(), O + 2 * s0);
            for (int64_t r = 0; r < rows; ++r) {
                const int64_t a0 = s0 + pos[static_cast<size_t>(r)], a1 = s0 + pos[static_cast<size_t>(r + 1)];
                if (a1 - a0 < 2) continue;
                std::vector<int64_t> row;
                row.reserve(static_cast<size_t>(a1 - a0));
                for (int64_t k = a0; k < a1; ++k) row.push_back(O[2 * k + 1]);
                std::sort(row.begin(), row.end());
                for (int64_t k = a0; k < a1; ++k) O[2 * k + 1] = row[static_cast<size_t>(k - a0)];
            }
        });
    }
    return out;
}

// ---- count_neighbors ----------------------------------------------------------------------------------

// The weight of every node: its points' weights summed (a leaf in leaf order, a node as left + right), or
// the point count without weights.
double node_weights(const KdTree& t, const double* w, int64_t node, std::vector<double>& W) {
    const KdNode& nd = t.nodes[static_cast<size_t>(node)];
    double v = 0.0;
    if (nd.right < 0) {
        if (!w) v = static_cast<double>(nd.end - nd.start);
        else
            for (int64_t i = nd.start; i < nd.end; ++i) v += w[i];
    } else {
        const double l = node_weights(t, w, node + 1, W);
        v = l + node_weights(t, w, nd.right, W);
    }
    W[static_cast<size_t>(node)] = v;
    return v;
}

// The weights in leaf order (null for none).
std::vector<double> leaf_weights(const KdTree& t, nb::handle h) {
    std::vector<double> w;
    if (h.is_none()) return w;
    if (!is_ndarray(h)) throw nb::type_error("_count(): weights must be an sd.ndarray");
    const NDArray& a = as_ndarray(h);
    if (a.ndim != 1 || a.dtype != DType::Float64 || a.shape[0] != t.n) throw nb::type_error("_count(): wrong weights");
    w.resize(static_cast<size_t>(t.n));
    for (int64_t i = 0; i < t.n; ++i)
        std::memcpy(&w[static_cast<size_t>(i)], a.data + t.idx[static_cast<size_t>(i)] * a.strides[0], 8);
    return w;
}

// One task's counts: delta[k] the pairs whose first radius with d <= radii[k] is k (k = R: none).
template <class Dist, class Acc>
struct Counter {
    const KdTree &A, &B;
    const Dist& dist;
    const KdCut& cut;
    const double *r, *T1;  // the radii and their cut thresholds KdCut(radii[k])
    const double *wa, *wb;  // leaf-order weights or null
    const double *WA, *WB;  // node weights (weighted counts only)
    Acc* delta;
    KdPair pr;

    Acc whole(int64_t na, int64_t nb) const {
        if constexpr (std::is_same_v<Acc, double>) return WA[na] * WB[nb];
        else return count_of(A, na) * count_of(B, nb);
    }

    void go(int64_t na, int64_t nb, int64_t lo, int64_t hi) {
        const int64_t m = A.m;
        pr.near(A.lo(na), A.hi(na), B.lo(nb), B.hi(nb), m);
        const double dn = kd_distance(dist, pr.a.data(), pr.b.data());
        lo = std::partition_point(T1 + lo, T1 + hi, [dn](double t) { return t < dn; }) - T1;  // cut below lo
        if (lo < hi) {
            pr.far(A.lo(na), A.hi(na), B.lo(nb), B.hi(nb), m);
            const double F = cut(kd_distance(dist, pr.a.data(), pr.b.data()));
            hi = std::partition_point(r + lo, r + hi, [F](double v) { return v < F; }) - r;  // inside from hi
        }
        if (lo == hi) {
            delta[hi] += whole(na, nb);
            return;
        }
        const KdNode &a = A.nodes[static_cast<size_t>(na)], &b = B.nodes[static_cast<size_t>(nb)];
        const bool la = a.right < 0, lb = b.right < 0;
        if (la && lb) {
            leaves(a, b, lo, hi);
        } else if (lb || (!la && a.end - a.start >= b.end - b.start)) {
            go(na + 1, nb, lo, hi);
            go(a.right, nb, lo, hi);
        } else {
            go(na, nb + 1, lo, hi);
            go(na, b.right, lo, hi);
        }
    }

    void leaves(const KdNode& a, const KdNode& b, int64_t lo, int64_t hi) {
        const int64_t m = A.m;
        const double above = Dist::above(r[hi - 1]), below = Dist::below(r[lo]);
        for (int64_t i = a.start; i < a.end; ++i) {
            const double* x = A.data.data() + i * m;
            const double* y = B.data.data() + b.start * m;
            for (int64_t q = b.start; q < b.end; ++q, y += m) {
                const double s = dist.raw(x, y);
                int64_t k;
                if (Dist::plain(s) && s > above) {
                    k = hi;
                } else if (Dist::plain(s) && s <= below) {
                    k = lo;
                } else {
                    const double d = dist.finish(s, x, y);
                    k = std::partition_point(r + lo, r + hi, [d](double v) { return v < d; }) - r;
                }
                if constexpr (std::is_same_v<Acc, double>) delta[k] += (wa ? wa[i] : 1.0) * (wb ? wb[q] : 1.0);
                else ++delta[k];
            }
        }
    }
};

template <class Dist, class Acc>
void count_with(
    const KdTree& A,
    const KdTree& B,
    const Dist& dist,
    const KdCut& cut,
    const std::vector<double>& r,
    const std::vector<double>& wa,
    const std::vector<double>& wb,
    bool weighted,
    Acc* sum
) {
    const int64_t R = static_cast<int64_t>(r.size());
    int depth = kDepth;
    while (depth > 0 && (int64_t{1} << (2 * depth)) * (R + 1) > kCountSlots) --depth;
    const std::vector<int64_t> pa = pieces(A, depth), pb = pieces(B, depth);
    const int64_t SA = static_cast<int64_t>(pa.size()), SB = static_cast<int64_t>(pb.size());
    std::vector<double> T1(static_cast<size_t>(R));
    for (int64_t k = 0; k < R; ++k) T1[static_cast<size_t>(k)] = cut(r[static_cast<size_t>(k)]);
    std::vector<double> WA, WB;
    if (weighted) {
        WA.resize(A.nodes.size());
        WB.resize(B.nodes.size());
        node_weights(A, wa.empty() ? nullptr : wa.data(), 0, WA);
        node_weights(B, wb.empty() ? nullptr : wb.data(), 0, WB);
    }
    std::vector<Acc> deltas(static_cast<size_t>(SA * SB * (R + 1)), Acc{0});
    const bool split = split_of(A, B);
    {
        const parallel::Unlocked unlocked(split);
        run_tasks(SA * SB, split, [&](int64_t task) {
            Counter<Dist, Acc> c{
                A,
                B,
                dist,
                cut,
                r.data(),
                T1.data(),
                wa.empty() ? nullptr : wa.data(),
                wb.empty() ? nullptr : wb.data(),
                WA.data(),
                WB.data(),
                deltas.data() + task * (R + 1),
                KdPair(A.m)
            };
            c.go(pa[static_cast<size_t>(task / SB)], pb[static_cast<size_t>(task % SB)], 0, R);
        });
    }
    for (int64_t t = 0; t < SA * SB; ++t)  // in task order (SPT25)
        for (int64_t k = 0; k < R; ++k) sum[k] += deltas[static_cast<size_t>(t * (R + 1) + k)];
}

template <class Dist>
nb::object count_dist(
    const KdTree& A,
    const KdTree& B,
    const Dist& dist,
    const KdCut& cut,
    const std::vector<double>& r,
    const std::vector<int64_t>& order,
    const std::vector<double>& wa,
    const std::vector<double>& wb,
    bool weighted,
    bool cumulative
) {
    const int64_t R = static_cast<int64_t>(r.size());
    nb::object out = new_array_of(weighted ? DType::Float64 : DType::Int64, {R});
    auto finish = [&](auto* sum, auto* o) {
        if (cumulative)
            for (int64_t k = 1; k < R; ++k) sum[k] = sum[k - 1] + sum[k];
        for (int64_t k = 0; k < R; ++k) o[order[static_cast<size_t>(k)]] = sum[k];
    };
    if (weighted) {
        std::vector<double> sum(static_cast<size_t>(R), 0.0);
        count_with(A, B, dist, cut, r, wa, wb, true, sum.data());
        finish(sum.data(), reinterpret_cast<double*>(as_ndarray(out).data));
    } else {
        std::vector<int64_t> sum(static_cast<size_t>(R), 0);
        count_with(A, B, dist, cut, r, wa, wb, false, sum.data());
        finish(sum.data(), reinterpret_cast<int64_t*>(as_ndarray(out).data));
    }
    return out;
}

// ---- bindings -----------------------------------------------------------------------------------------

void check_pair(const KdTree& A, const KdTree& B, double p, const char* func) {
    if (A.m != B.m) throw nb::type_error((std::string(func) + "(): the trees' m differ").c_str());
    if (!(p >= 1.0)) throw nb::value_error((std::string(func) + "(): wrong p").c_str());
}

template <class F>
nb::object by_p(int64_t m, double p, F&& f) {
    if (p == 2.0) return f(KdEuclid{m}, KdCut(m, false));
    if (p == 1.0) return f(KdCity{m}, KdCut(m, false));
    if (std::isinf(p)) return f(KdCheb{m}, KdCut(m, true));
    return f(KdMink{m, Root(p)}, KdCut(m, false));
}

nb::object ball_tree(const KdTree& A, const KdTree& B, double r, double p) {
    check_pair(A, B, p, "_ball_tree");
    if (!(r >= 0.0)) throw nb::value_error("_ball_tree(): wrong r");
    return by_p(A.m, p, [&](const auto& dist, const KdCut& cut) { return ball_tree_with(A, B, dist, cut, r); });
}

nb::object pairs(const KdTree& A, double r, double p) {
    check_pair(A, A, p, "_pairs");
    if (!(r >= 0.0)) throw nb::value_error("_pairs(): wrong r");
    return by_p(A.m, p, [&](const auto& dist, const KdCut& cut) { return pairs_with(A, dist, cut, r); });
}

nb::object count(
    const KdTree& A,
    const KdTree& B,
    nb::handle radii,
    nb::handle order_h,
    double p,
    nb::handle wa_h,
    nb::handle wb_h,
    bool cumulative
) {
    check_pair(A, B, p, "_count");
    if (!is_ndarray(radii) || !is_ndarray(order_h)) throw nb::type_error("_count(): expected sd.ndarrays");
    const NDArray& ra = as_ndarray(radii);
    const NDArray& oa = as_ndarray(order_h);
    if (ra.ndim != 1 || ra.dtype != DType::Float64 || oa.ndim != 1 || oa.dtype != DType::Int64 ||
        oa.shape[0] != ra.shape[0])
        throw nb::type_error("_count(): wrong radii");
    const int64_t R = ra.shape[0];
    std::vector<double> r(static_cast<size_t>(R));
    std::vector<int64_t> order(static_cast<size_t>(R));
    std::vector<char> seen(static_cast<size_t>(R), 0);
    for (int64_t k = 0; k < R; ++k) {
        std::memcpy(&r[static_cast<size_t>(k)], ra.data + k * ra.strides[0], 8);
        std::memcpy(&order[static_cast<size_t>(k)], oa.data + k * oa.strides[0], 8);
        const double v = r[static_cast<size_t>(k)];
        const int64_t o = order[static_cast<size_t>(k)];
        if (!(v >= 0.0) || (k && v < r[static_cast<size_t>(k - 1)]) || o < 0 || o >= R || seen[static_cast<size_t>(o)])
            throw nb::value_error("_count(): wrong radii");
        seen[static_cast<size_t>(o)] = 1;
    }
    const bool weighted = !wa_h.is_none() || !wb_h.is_none();
    const std::vector<double> wa = leaf_weights(A, wa_h), wb = leaf_weights(B, wb_h);
    return by_p(A.m, p, [&](const auto& dist, const KdCut& cut) {
        return count_dist(A, B, dist, cut, r, order, wa, wb, weighted, cumulative);
    });
}

}  // namespace

void bind_dual(nb::module_& m) {
    m.def("_ball_tree", &ball_tree, nb::arg("A"), nb::arg("B"), nb::arg("r"), nb::arg("p"));
    m.def("_pairs", &pairs, nb::arg("A"), nb::arg("r"), nb::arg("p"));
    m.def(
        "_count", &count, nb::arg("A"), nb::arg("B"), nb::arg("radii"), nb::arg("order"), nb::arg("p"),
        nb::arg("wa").none(), nb::arg("wb").none(), nb::arg("cumulative")
    );
}

}  // namespace sd::spatial
