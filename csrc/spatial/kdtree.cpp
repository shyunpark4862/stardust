// sd.spatial's KDTree: the build and the k-nearest query (docs/design/spatial.md §3.3, §3.4, §3.5; the
// tree and the search are in kdtree.hpp). The Python layer (src/stardust/spatial/_kdtree.py) checks every
// argument; these functions only check the layout they are handed and that the points are finite.
//
//   _KDTree(data, leafsize)       data (n, m) float64 (any strides), copied into leaf order
//   t.n, t.m, t.leafsize, t.size  the counts (size: the nodes)
//   t.mins(), t.maxes()           new (m,) float64: the range of the data (zeros when n = 0, as scipy)
//   t.query(x, ranks, p, ub)      x (Q, m) float64, ranks (R,) int64 >= 1 (1-based), p >= 1 or inf,
//                                 ub >= 0 or inf -> (distances (Q, R) float64, indices (Q, R) int64):
//                                 the ranks-th nearest by (distance, index) with distance < ub, the
//                                 missing ones inf and n
//   t._state()                    (indices, starts, ends, rights, boxes): the tree's arrays, for the tests
//
// Every query point is searched by one task from start to end; the tasks are blocks of kQueryBlock points
// (size alone), so the results do not depend on the thread count. From kParallelWork units of work a call
// splits and runs without the GIL.
#include "spatial/kdtree.hpp"

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

// The array area's rule (docs/design/array.md §10.4): from 2^15 units of work a call splits and runs
// without the GIL, whatever the thread count.
constexpr int64_t kParallelWork = 1 << 15;
constexpr int64_t kBoxChunk = 1 << 14;     // points a task scans for the box of a node at the top
constexpr int64_t kSubtreeMin = 1 << 12;   // the smallest subtree built as one task
constexpr int64_t kSubtreeShare = 64;      // ... or n / 64 points, whichever is larger (SPT53)
constexpr int64_t kQueryBlock = 64;        // query points a task searches (SPT54)
constexpr int64_t kQueryWorkPerPoint = 64;  // the work units of one query point per coordinate

template <class F>
void run_tasks(int64_t ntasks, bool split, F&& fn) {
    if (!split || ntasks <= 1) {
        for (int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

// The build works on the tree's own copy of the rows: a node's points are the rows [s, e) of t.data
// (with their original indices in t.idx), moved there by the splits, so every scan is sequential.
struct Builder {
    int64_t m, leafsize;
    KdTree& t;
    std::unordered_map<int64_t, int64_t> counts;
    struct Key {
        double key;
        int64_t pos;  // the row (its original index is t.idx[pos], read for ties only)
    };
    std::vector<Key> keys;      // scratch of split and leaf
    std::vector<double> rows;   // scratch of move_rows
    std::vector<int64_t> ids;   // scratch of move_rows

    int64_t count(int64_t c) {
        if (c <= leafsize) return 1;
        const auto it = counts.find(c);
        if (it != counts.end()) return it->second;
        const int64_t v = 1 + count(c / 2) + count(c - c / 2);
        counts.emplace(c, v);
        return v;
    }

    double* row(int64_t i) { return t.data.data() + i * m; }

    // The range of rows [s, e) into the node's box; `split`: by chunks on the pool (min and max are
    // exact, so the chunks give the same box).
    void box_of(int64_t node, int64_t s, int64_t e, bool split) {
        double* lo = t.box.data() + 2 * m * node;
        double* hi = lo + m;
        if (s == e || m == 0) {
            std::fill(lo, lo + 2 * m, 0.0);
            return;
        }
        auto scan = [&](int64_t a, int64_t b, double* l, double* h) {
            const double* p0 = row(a);
            std::copy(p0, p0 + m, l);
            std::copy(p0, p0 + m, h);
            for (int64_t i = a + 1; i < b; ++i) {
                const double* p = row(i);
                for (int64_t d = 0; d < m; ++d) {
                    l[d] = std::min(l[d], p[d]);
                    h[d] = std::max(h[d], p[d]);
                }
            }
        };
        const int64_t chunks = (e - s + kBoxChunk - 1) / kBoxChunk;
        if (!split || chunks < 2) {
            scan(s, e, lo, hi);
            return;
        }
        std::vector<double> part(static_cast<size_t>(2 * m * chunks));
        run_tasks(chunks, true, [&](int64_t c) {
            double* l = part.data() + 2 * m * c;
            scan(s + c * kBoxChunk, std::min(e, s + (c + 1) * kBoxChunk), l, l + m);
        });
        std::copy(part.begin(), part.begin() + m, lo);
        std::copy(part.begin() + m, part.begin() + 2 * m, hi);
        for (int64_t c = 1; c < chunks; ++c)
            for (int64_t d = 0; d < m; ++d) {
                lo[d] = std::min(lo[d], part[static_cast<size_t>(2 * m * c + d)]);
                hi[d] = std::max(hi[d], part[static_cast<size_t>(2 * m * c + m + d)]);
            }
    }

    // Rows [s, e) rearranged as keys[0 .. e - s) say (keys[j].pos the row now at s + j); `split`: by
    // chunks on the pool.
    void move_rows(int64_t s, int64_t e, bool split) {
        const int64_t c = e - s;
        rows.resize(static_cast<size_t>(c * m));
        ids.resize(static_cast<size_t>(c));
        const int64_t chunks = (c + kBoxChunk - 1) / kBoxChunk;
        run_tasks(chunks, split, [&](int64_t k) {
            for (int64_t j = k * kBoxChunk, end = std::min(c, (k + 1) * kBoxChunk); j < end; ++j) {
                const int64_t from = keys[static_cast<size_t>(j)].pos;
                const double* p = row(from);
                std::copy(p, p + m, rows.data() + j * m);
                ids[static_cast<size_t>(j)] = t.idx[static_cast<size_t>(from)];
            }
        });
        run_tasks(chunks, split, [&](int64_t k) {
            const int64_t j0 = k * kBoxChunk, j1 = std::min(c, (k + 1) * kBoxChunk);
            std::copy(rows.data() + j0 * m, rows.data() + j1 * m, row(s + j0));
            std::copy(ids.begin() + j0, ids.begin() + j1, t.idx.begin() + s + j0);
        });
    }

    // A leaf: its points in index order (the tree does not depend on nth_element's arrangement).
    void leaf(int64_t node, int64_t s, int64_t e) {
        t.nodes[static_cast<size_t>(node)] = KdNode{s, e, -1};
        bool sorted = true;
        for (int64_t i = s + 1; i < e && sorted; ++i)
            sorted = t.idx[static_cast<size_t>(i - 1)] < t.idx[static_cast<size_t>(i)];
        if (sorted) return;
        keys.resize(static_cast<size_t>(e - s));
        for (int64_t i = s; i < e; ++i) keys[static_cast<size_t>(i - s)] = Key{0.0, i};
        const int64_t* idx = t.idx.data();
        std::sort(keys.begin(), keys.end(), [idx](const Key& a, const Key& b) { return idx[a.pos] < idx[b.pos]; });
        move_rows(s, e, false);
    }

    // Splits rows [s, e) at s + (e - s) / 2 on the node's widest axis by (coordinate, index); the node's
    // entry is set. Returns the middle.
    int64_t split(int64_t node, int64_t s, int64_t e, bool par) {
        const double *lo = t.box.data() + 2 * m * node, *hi = lo + m;
        int64_t dim = 0;
        double best = -1.0;
        for (int64_t d = 0; d < m; ++d) {
            const double w = hi[d] - lo[d];  // inf for a range beyond the doubles: still the widest
            if (w > best) {
                best = w;
                dim = d;
            }
        }
        const int64_t c = e - s, mid = s + c / 2;
        keys.resize(static_cast<size_t>(c));
        const int64_t chunks = (c + kBoxChunk - 1) / kBoxChunk;
        run_tasks(chunks, par, [&](int64_t k) {
            for (int64_t j = k * kBoxChunk, end = std::min(c, (k + 1) * kBoxChunk); j < end; ++j)
                keys[static_cast<size_t>(j)] = Key{m ? row(s + j)[dim] : 0.0, s + j};
        });
        const int64_t* idx = t.idx.data();
        std::nth_element(keys.begin(), keys.begin() + (mid - s), keys.end(), [idx](const Key& a, const Key& b) {
            return a.key < b.key || (a.key == b.key && idx[a.pos] < idx[b.pos]);
        });
        move_rows(s, e, par);
        t.nodes[static_cast<size_t>(node)] = KdNode{s, e, node + 1 + count(mid - s)};
        return mid;
    }

    void serial(int64_t node, int64_t s, int64_t e) {
        box_of(node, s, e, false);
        if (e - s <= leafsize) {
            leaf(node, s, e);
            return;
        }
        const int64_t mid = split(node, s, e, false);
        const int64_t right = t.nodes[static_cast<size_t>(node)].right;
        serial(node + 1, s, mid);
        serial(right, mid, e);
    }

    struct Task {
        int64_t node, s, e;
    };

    // The top of the tree on the calling thread down to subtrees of at most `grain` points, which are
    // collected (in preorder) for the pool.
    void top(int64_t node, int64_t s, int64_t e, int64_t grain, std::vector<Task>& tasks) {
        if (e - s <= grain || e - s <= leafsize) {
            tasks.push_back({node, s, e});
            return;
        }
        box_of(node, s, e, true);
        const int64_t mid = split(node, s, e, true);
        const int64_t right = t.nodes[static_cast<size_t>(node)].right;
        top(node + 1, s, mid, grain, tasks);
        top(right, mid, e, grain, tasks);
    }
};

}  // namespace

int64_t kd_node_count(int64_t c, int64_t leafsize) {
    if (c <= leafsize) return 1;
    return 1 + kd_node_count(c / 2, leafsize) + kd_node_count(c - c / 2, leafsize);
}

KdTree kd_build(const double* src, int64_t n, int64_t m, int64_t ld, int64_t leafsize) {
    KdTree t;
    t.n = n;
    t.m = m;
    t.leafsize = leafsize;
    const bool split = n * std::max<int64_t>(m, 1) >= kParallelWork;
    t.idx.resize(static_cast<size_t>(n));
    t.data.resize(static_cast<size_t>(n * m));
    const int64_t chunks = (n + kBoxChunk - 1) / kBoxChunk;
    run_tasks(chunks, split, [&](int64_t c) {
        for (int64_t i = c * kBoxChunk, e = std::min(n, (c + 1) * kBoxChunk); i < e; ++i) {
            t.idx[static_cast<size_t>(i)] = i;
            std::copy(src + i * ld, src + i * ld + m, t.data.data() + i * m);
        }
    });
    Builder b{m, leafsize, t, {}, {}, {}, {}};
    const int64_t nodes = b.count(n);
    t.nodes.assign(static_cast<size_t>(nodes), KdNode{});
    t.box.assign(static_cast<size_t>(2 * m * nodes), 0.0);
    if (!split) {
        b.serial(0, 0, n);
        return t;
    }
    std::vector<Builder::Task> tasks;
    b.top(0, 0, n, std::max(kSubtreeMin, n / kSubtreeShare), tasks);
    run_tasks(static_cast<int64_t>(tasks.size()), true, [&](int64_t i) {
        const auto& k = tasks[static_cast<size_t>(i)];
        Builder local{m, leafsize, t, b.counts, {}, {}, {}};
        local.serial(k.node, k.s, k.e);
    });
    return t;
}

namespace {

// A 2-D float64 array as rows of m doubles with a row stride (a copy when the columns are strided).
struct Rows {
    const double* p = nullptr;
    int64_t rows = 0, cols = 0, ld = 0;
    std::vector<double> own;
};

Rows rows_arg(nb::handle o, const char* func) {
    if (!is_ndarray(o)) throw nb::type_error((std::string(func) + "(): expected an sd.ndarray").c_str());
    const NDArray& a = as_ndarray(o);
    if (a.ndim != 2 || a.dtype != DType::Float64)
        throw nb::type_error((std::string(func) + "(): wrong array layout").c_str());
    Rows r;
    r.rows = a.shape[0];
    r.cols = a.shape[1];
    if (r.rows * r.cols == 0) {
        r.ld = r.cols;
        r.p = reinterpret_cast<const double*>(a.data);
        return r;
    }
    if (a.strides[1] == 8 && a.strides[0] % 8 == 0 && a.strides[0] >= 0) {
        r.p = reinterpret_cast<const double*>(a.data);
        r.ld = a.strides[0] / 8;
        return r;
    }
    r.own.resize(static_cast<size_t>(r.rows * r.cols));
    for (int64_t i = 0; i < r.rows; ++i)
        for (int64_t j = 0; j < r.cols; ++j)
            std::memcpy(&r.own[static_cast<size_t>(i * r.cols + j)], a.data + i * a.strides[0] + j * a.strides[1], 8);
    r.p = r.own.data();
    r.ld = r.cols;
    return r;
}

bool all_finite(const Rows& r) {
    for (int64_t i = 0; i < r.rows; ++i) {
        const double* p = r.p + i * r.ld;
        for (int64_t j = 0; j < r.cols; ++j)
            if (!std::isfinite(p[j])) return false;
    }
    return true;
}

NDArray f64_vector(const double* v, int64_t m) {
    const int64_t shape[] = {m};
    NDArray out = new_array(DType::Float64, shape);
    if (m) std::memcpy(out.data, v, static_cast<size_t>(8 * m));
    return out;
}

KdTree* new_tree(nb::handle data, int64_t leafsize) {
    const Rows r = rows_arg(data, "_KDTree");
    if (leafsize < 1) throw nb::value_error("_KDTree(): leafsize < 1");
    bool finite = true;
    auto* t = new KdTree();
    {
        const parallel::Unlocked unlocked(r.rows * std::max<int64_t>(r.cols, 1) >= kParallelWork);
        finite = all_finite(r);
        if (finite) *t = kd_build(r.p, r.rows, r.cols, r.ld, leafsize);
    }
    if (!finite) {
        delete t;
        throw nb::value_error("KDTree(): 'data' must be finite");
    }
    return t;
}

template <class Dist>
void query_with(
    const KdTree& t,
    const Dist& dist,
    const KdCut& cut,
    const Rows& x,
    const int64_t* ranks,
    int64_t nr,
    int64_t kmax,
    double ub,
    double* od,
    int64_t* oi
) {
    const int64_t q = x.rows;
    const int64_t ntasks = (q + kQueryBlock - 1) / kQueryBlock;
    const bool split = q * std::max<int64_t>(t.m, 1) * kQueryWorkPerPoint >= kParallelWork;
    const parallel::Unlocked unlocked(split);
    run_tasks(ntasks, split, [&](int64_t task) {
        KdSearch<Dist> s(t, dist, cut, kmax);
        std::vector<std::pair<double, int64_t>> found;
        for (int64_t i = task * kQueryBlock, e = std::min(q, (task + 1) * kQueryBlock); i < e; ++i) {
            s.run(x.p + i * x.ld, ub, found);
            const int64_t got = static_cast<int64_t>(found.size());
            for (int64_t j = 0; j < nr; ++j) {
                const int64_t r = ranks[j] - 1;
                od[i * nr + j] =
                    r < got ? found[static_cast<size_t>(r)].first : std::numeric_limits<double>::infinity();
                oi[i * nr + j] = r < got ? found[static_cast<size_t>(r)].second : t.n;
            }
        }
    });
}

nb::tuple tree_query(const KdTree& t, nb::handle xh, nb::handle ranks_h, double p, double ub) {
    const Rows x = rows_arg(xh, "_KDTree.query");
    if (x.cols != t.m) throw nb::type_error("_KDTree.query(): wrong number of columns");
    if (!is_ndarray(ranks_h)) throw nb::type_error("_KDTree.query(): ranks must be an sd.ndarray");
    const NDArray& rk = as_ndarray(ranks_h);
    if (rk.ndim != 1 || rk.dtype != DType::Int64 || rk.shape[0] < 1)
        throw nb::type_error("_KDTree.query(): wrong ranks");
    const int64_t nr = rk.shape[0];
    std::vector<int64_t> ranks(static_cast<size_t>(nr));
    for (int64_t j = 0; j < nr; ++j) {
        std::memcpy(&ranks[static_cast<size_t>(j)], rk.data + j * rk.strides[0], 8);
        if (ranks[static_cast<size_t>(j)] < 1) throw nb::value_error("_KDTree.query(): ranks start at 1");
    }
    if (!(p >= 1.0) || !(ub >= 0.0)) throw nb::value_error("_KDTree.query(): wrong p or bound");
    const int64_t kmax = *std::max_element(ranks.begin(), ranks.end());
    const int64_t shape[] = {x.rows, nr};
    NDArray d = new_array(DType::Float64, shape);
    NDArray i = new_array(DType::Int64, shape);
    bool finite = true;
    {
        const parallel::Unlocked unlocked(x.rows * std::max<int64_t>(x.cols, 1) >= kParallelWork);
        finite = all_finite(x);
    }
    if (!finite) throw nb::value_error("query(): 'x' must be finite");
    auto* od = reinterpret_cast<double*>(d.data);
    auto* oi = reinterpret_cast<int64_t*>(i.data);
    const int64_t m = t.m;
    if (p == 2.0) {
        query_with(t, KdEuclid{m}, KdCut(m, false), x, ranks.data(), nr, kmax, ub, od, oi);
    } else if (p == 1.0) {
        query_with(t, KdCity{m}, KdCut(m, false), x, ranks.data(), nr, kmax, ub, od, oi);
    } else if (std::isinf(p)) {
        query_with(t, KdCheb{m}, KdCut(m, true), x, ranks.data(), nr, kmax, ub, od, oi);
    } else {
        query_with(t, KdMink{m, Root(p)}, KdCut(m, false), x, ranks.data(), nr, kmax, ub, od, oi);
    }
    return nb::make_tuple(wrap(std::move(d)), wrap(std::move(i)));
}

nb::tuple tree_state(const KdTree& t) {
    const int64_t nn = static_cast<int64_t>(t.nodes.size());
    const int64_t s1[] = {t.n}, s2[] = {nn}, s3[] = {nn, 2 * t.m};
    NDArray idx = new_array(DType::Int64, s1);
    NDArray st = new_array(DType::Int64, s2), en = new_array(DType::Int64, s2), rt = new_array(DType::Int64, s2);
    NDArray bx = new_array(DType::Float64, s3);
    if (t.n) std::memcpy(idx.data, t.idx.data(), static_cast<size_t>(8 * t.n));
    for (int64_t k = 0; k < nn; ++k) {
        reinterpret_cast<int64_t*>(st.data)[k] = t.nodes[static_cast<size_t>(k)].start;
        reinterpret_cast<int64_t*>(en.data)[k] = t.nodes[static_cast<size_t>(k)].end;
        reinterpret_cast<int64_t*>(rt.data)[k] = t.nodes[static_cast<size_t>(k)].right;
    }
    if (!t.box.empty()) std::memcpy(bx.data, t.box.data(), 8 * t.box.size());
    return nb::make_tuple(
        wrap(std::move(idx)), wrap(std::move(st)), wrap(std::move(en)), wrap(std::move(rt)), wrap(std::move(bx))
    );
}

}  // namespace

void bind_kdtree(nb::module_& m) {
    nb::class_<KdTree>(m, "_KDTree")
        .def(
            "__init__",
            [](KdTree* self, nb::handle data, int64_t leafsize) {
                KdTree* t = new_tree(data, leafsize);
                new (self) KdTree(std::move(*t));
                delete t;
            },
            nb::arg("data"), nb::arg("leafsize")
        )
        .def_ro("n", &KdTree::n)
        .def_ro("m", &KdTree::m)
        .def_ro("leafsize", &KdTree::leafsize)
        .def_prop_ro("size", [](const KdTree& t) { return static_cast<int64_t>(t.nodes.size()); })
        .def(
            "mins",
            [](const KdTree& t) {
                return wrap(f64_vector(t.box.data(), t.m));  // the root's box (zeros when n = 0)
            }
        )
        .def("maxes", [](const KdTree& t) { return wrap(f64_vector(t.box.data() + t.m, t.m)); })
        .def("query", &tree_query, nb::arg("x"), nb::arg("ranks"), nb::arg("p"), nb::arg("ub"))
        .def("_state", &tree_state);
}

}  // namespace sd::spatial
