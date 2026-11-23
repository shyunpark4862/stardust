// KDTree.query_ball_point of sd.spatial (docs/design/spatial.md §3.4, §3.5, §3.6; milestone SPT4). The Python
// layer (src/stardust/spatial/_kdtree.py) checks every argument.
//
//   _ball_point(tree, x, r, p, count_only, func="query_ball_point", max_pairs=-1)
//       x (Q, m) float64 C-contiguous and finite, r (Q,) float64 >= 0 or inf, p >= 1 or inf
//       -> (indptr (Q + 1), indices, distances): row q holds every data point j with d(x_q, j) <= r_q, in
//          index order; count_only: the counts (Q,) int64.
//
// Every distance returned and every boundary test is the pair kernel of the metric of p (kdtree.hpp), so
// the rows are radius_neighbors' (the brute force with the same kernel). A node is cut only when its near
// bound exceeds KdCut(r) (SPT51, kdtree.hpp); counting adds a whole node once its far bound proves every
// point inside (KdPair::far). One task searches kQueryBlock query points from start to end (size alone),
// so the result does not depend on the thread count. A result larger than this machine's physical memory
// stops the search with MemoryError.
#include "array/ndarray.hpp"
#include "base/buffer.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/kdtree.hpp"
#include "spatial/spatial.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

constexpr int64_t kParallelWork = 1 << 15;  // array.md §10.4: from here a call splits and drops the GIL
constexpr int64_t kQueryBlock = 64;         // query points a task searches (as KDTree.query, SPT54)
constexpr int64_t kQueryWorkPerPoint = 64;  // the work units of one query point per coordinate
constexpr int64_t kEntryBytes = 40;         // a found pair while collecting (16) and in the result (16), rounded up

template <class F>
void run_tasks(int64_t ntasks, bool split, F&& fn) {
    if (!split || ntasks <= 1) {
        for (int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

struct Hit {
    int64_t j;
    double d;
};

// One query point's search: the points within r (listing) or their count.
template <class Dist>
class Ball {
public:
    Ball(const KdTree& t, const Dist& dist, const KdCut& cut, std::vector<Hit>* out)
        : t_(t), dist_(dist), cut_(cut), out_(out), pr_(t.m) {}

    // Lists (or counts, out_ null) the points within r of x.
    int64_t run(const double* x, double r) {
        x_ = x;
        r_ = r;
        T1_ = cut_(r);
        count_ = 0;
        if (t_.n > 0 && !beyond(0)) visit(0);
        return count_;
    }

private:
    const KdTree& t_;
    const Dist& dist_;
    const KdCut& cut_;
    std::vector<Hit>* out_;
    KdPair pr_;
    const double* x_ = nullptr;
    double r_ = 0.0, T1_ = 0.0;
    int64_t count_ = 0;

    bool beyond(int64_t node) {
        pr_.near(x_, x_, t_.lo(node), t_.hi(node), t_.m);
        return kd_beyond(dist_, pr_.a.data(), pr_.b.data(), T1_);
    }

    bool inside(int64_t node) {
        pr_.far(x_, x_, t_.lo(node), t_.hi(node), t_.m);
        return cut_(kd_distance(dist_, pr_.a.data(), pr_.b.data())) <= r_;
    }

    void visit(int64_t node) {
        const KdNode& nd = t_.nodes[static_cast<size_t>(node)];
        if (!out_ && inside(node)) {
            count_ += nd.end - nd.start;
            return;
        }
        if (nd.right < 0) {
            const double below = Dist::below(r_), above = Dist::above(r_);
            const double* y = t_.data.data() + nd.start * t_.m;
            for (int64_t i = nd.start; i < nd.end; ++i, y += t_.m) {
                const double s = dist_.raw(x_, y);
                if (Dist::plain(s)) {
                    if (s > above) continue;  // d^ > r
                    if (!out_ && s <= below) {  // d^ <= r
                        ++count_;
                        continue;
                    }
                }
                const double d = dist_.finish(s, x_, y);
                if (!(d <= r_)) continue;
                ++count_;
                if (out_) out_->push_back({t_.idx[static_cast<size_t>(i)], d});
            }
            return;
        }
        const int64_t l = node + 1, rt = nd.right;
        if (!beyond(l)) visit(l);
        if (!beyond(rt)) visit(rt);
    }
};

[[noreturn]] void too_large(const char* func) {
    const std::string msg = std::string(func) + "(): the result is larger than this machine's physical memory";
    PyErr_SetString(PyExc_MemoryError, msg.c_str());
    throw nb::python_error();
}

nb::object new_array_of(DType t, std::initializer_list<int64_t> shape) {
    const std::vector<int64_t> s(shape);
    return wrap(new_array(t, s));
}

// What ball_with needs besides the search: the public function's name for the MemoryError and the
// number of found pairs it stops at (the pairs physical memory holds, or the caller's smaller number).
struct Limit {
    const char* func;
    int64_t pairs;
};

template <class Dist>
nb::object ball_with(
    const KdTree& t,
    const Dist& dist,
    const KdCut& cut,
    const double* X,
    int64_t q,
    const double* R,
    bool count_only,
    const Limit& limit
) {
    const int64_t ntasks = (q + kQueryBlock - 1) / kQueryBlock;
    const bool split = q * std::max<int64_t>(t.m, 1) * kQueryWorkPerPoint >= kParallelWork;
    const int64_t m = t.m;
    if (count_only) {
        nb::object oc = new_array_of(DType::Int64, {q});
        int64_t* C = reinterpret_cast<int64_t*>(as_ndarray(oc).data);
        {
            const parallel::Unlocked unlocked(split);
            run_tasks(ntasks, split, [&](int64_t task) {
                Ball<Dist> b(t, dist, cut, nullptr);
                for (int64_t i = task * kQueryBlock, e = std::min(q, (task + 1) * kQueryBlock); i < e; ++i)
                    C[i] = b.run(X + i * m, R[i]);
            });
        }
        return oc;
    }
    // Per task the hits of its rows one after another (each row in index order) and the rows' counts.
    std::vector<std::vector<Hit>> hits(static_cast<size_t>(ntasks));
    std::vector<int64_t> counts(static_cast<size_t>(q), 0);
    const int64_t budget = limit.pairs;
    std::atomic<int64_t> total{0};
    std::atomic<bool> over{false};
    {
        const parallel::Unlocked unlocked(split);
        run_tasks(ntasks, split, [&](int64_t task) {
            std::vector<Hit>& h = hits[static_cast<size_t>(task)];
            Ball<Dist> b(t, dist, cut, &h);
            for (int64_t i = task * kQueryBlock, e = std::min(q, (task + 1) * kQueryBlock); i < e; ++i) {
                if (over.load(std::memory_order_relaxed)) return;
                const size_t from = h.size();
                b.run(X + i * m, R[i]);
                std::sort(h.begin() + static_cast<std::ptrdiff_t>(from), h.end(), [](const Hit& u, const Hit& v) {
                    return u.j < v.j;
                });
                const int64_t c = static_cast<int64_t>(h.size() - from);
                counts[static_cast<size_t>(i)] = c;
                if (c && total.fetch_add(c, std::memory_order_relaxed) + c > budget) over.store(true);
            }
        });
    }
    if (over.load()) too_large(limit.func);
    int64_t nnz = 0;
    for (int64_t c : counts) nnz += c;
    nb::object op = new_array_of(DType::Int64, {q + 1}), oi = new_array_of(DType::Int64, {nnz}),
               od = new_array_of(DType::Float64, {nnz});
    int64_t* P = reinterpret_cast<int64_t*>(as_ndarray(op).data);
    int64_t* I = reinterpret_cast<int64_t*>(as_ndarray(oi).data);
    double* D = reinterpret_cast<double*>(as_ndarray(od).data);
    P[0] = 0;
    for (int64_t i = 0; i < q; ++i) P[i + 1] = P[i] + counts[static_cast<size_t>(i)];
    {
        const parallel::Unlocked unlocked(split && nnz >= kParallelWork);
        run_tasks(ntasks, split && nnz >= kParallelWork, [&](int64_t task) {
            const std::vector<Hit>& h = hits[static_cast<size_t>(task)];
            int64_t dst = P[task * kQueryBlock];
            for (const Hit& e : h) {
                I[dst] = e.j;
                D[dst++] = e.d;
            }
        });
    }
    return nb::make_tuple(op, oi, od);
}

nb::object ball_point(
    const KdTree& t, nb::handle xh, nb::handle rh, double p, bool count_only, const std::string& func, int64_t max_pairs
) {
    if (!is_ndarray(xh) || !is_ndarray(rh)) throw nb::type_error("_ball_point(): expected sd.ndarrays");
    const NDArray& x = as_ndarray(xh);
    const NDArray& r = as_ndarray(rh);
    if (x.ndim != 2 || x.dtype != DType::Float64 || !x.c_contiguous || x.shape[1] != t.m)
        throw nb::type_error("_ball_point(): wrong x");
    const int64_t q = x.shape[0];
    if (r.ndim != 1 || r.dtype != DType::Float64 || !r.c_contiguous || r.shape[0] != q)
        throw nb::type_error("_ball_point(): wrong r");
    if (!(p >= 1.0)) throw nb::value_error("_ball_point(): wrong p");
    const auto* X = reinterpret_cast<const double*>(x.data);
    const auto* R = reinterpret_cast<const double*>(r.data);
    for (int64_t i = 0; i < q; ++i)
        if (!(R[i] >= 0.0)) throw nb::value_error("_ball_point(): wrong r");
    const int64_t m = t.m;
    const int64_t fits = static_cast<int64_t>(physical_memory() / kEntryBytes);
    const Limit limit{func.c_str(), max_pairs < 0 ? fits : std::min(max_pairs, fits)};
    if (p == 2.0) return ball_with(t, KdEuclid{m}, KdCut(m, false), X, q, R, count_only, limit);
    if (p == 1.0) return ball_with(t, KdCity{m}, KdCut(m, false), X, q, R, count_only, limit);
    if (std::isinf(p)) return ball_with(t, KdCheb{m}, KdCut(m, true), X, q, R, count_only, limit);
    return ball_with(t, KdMink{m, Root(p)}, KdCut(m, false), X, q, R, count_only, limit);
}

}  // namespace

void bind_query(nb::module_& m) {
    m.def(
        "_ball_point", &ball_point, nb::arg("tree"), nb::arg("x"), nb::arg("r"), nb::arg("p"), nb::arg("count_only"),
        nb::arg("func") = "query_ball_point", nb::arg("max_pairs") = -1
    );
}

}  // namespace sd::spatial
