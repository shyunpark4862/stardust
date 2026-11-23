// kneighbors and radius_neighbors with metric='haversine' through a KD tree on the unit sphere
// (docs/design/spatial.md §3.4 "haversine의 트리 길"; decisions SPT83-SPT88). The Python layer
// (src/stardust/spatial/_neighbors.py) checks every argument.
//
//   _sphere_kneighbors(XA, XB, k, upper)
//       -> (distances (mA, k), indices (mA, k)) as _kneighbors', or None when a point is outside the
//          domain below (the caller then takes the brute force).
//   _sphere_radius(XA, XB, r, count_only)
//       -> (indptr, indices, distances) or the counts, as _radius_neighbors', or None (as above).
//       XA, XB: (rows, 2) float64 C-contiguous and finite, (latitude, longitude) in radians.
//
// Every (latitude, longitude) becomes the point P = (cos phi cos lam, cos phi sin lam, sin phi) of the
// unit sphere, and the KD tree of csrc/spatial/kdtree.hpp is built over the computed points P^ of XB. The
// tree only selects candidates: every returned distance and every test that decides membership or order is
// haversine_pair (csrc/spatial/metric.hpp) of the original rows, called as cdist calls it (x the row of XA,
// y the row of XB, cos phi by std::cos), so the distances are cdist's bits and the order is (d, j).
//
// The margin (SPT84). The chord C = |P(x) - P(y)| of the exact points satisfies C = 2 sqrt(h) = 2 sin(d / 2)
// for the exact haversine d (an identity for all real phi, lam), so d grows with C; after rounding it
// need not, and the cut below covers the three errors:
//   (H) the haversine kernel: |d^ - d| <= 2^-46 (1 + d) on the domain |phi| <= fl(pi/2), |lam| <= 8 (the
//       proof is in docs/design/spatial.md §3.4: 16u d + 2^-99 on the near branch, 43u on the far one,
//       with libm's sin, cos and asin within 1 ULP);
//   (P) the conversion: |P^ - P| <= 5.1u (each component one or two libm values and one product, so its
//       relative error is at most (1 + 2u)^2 (1 + u) - 1 <= 5.1u, and |P| = 1);
//   (E) the euclidean kernel over the 3 coordinates: |b^ - D| <= kappa D + 2^-1074, kappa = gamma_6 / 2 + u.
// The cut is computed with every constant at least 4 times the bound it covers (2^-44 for H, 2^-48 for P),
// so it also holds for libm errors up to 4 ULP:
//     cut(T) = fl(fl(2 sin(fl(fl(T c) + 2^-42) / 2)) c) + 2^-46),  c = 1 + 2^-41,
//     and inf when fl(fl(T c) + 2^-42) >= 3.14 (or T = inf).
// If d^(x, y) <= T then d <= T' = (T + 2^-44) / (1 - 2^-44) <= t := fl(fl(T c) + 2^-42) < pi, so
// C = 2 sin(d / 2) <= 2 sin(t / 2) (sin grows on [0, pi / 2]), |P^(x) - P^(y)| <= C + 2^-47, and the kernel
// distance b^ from P^(x) to any box holding P^(y) (the clamp point is no farther than P^(y)) is at most
// (1 + kappa) (2 sin(t / 2) + 2^-47) + 2^-1074 <= cut(T). So a box (or a point) whose b^ exceeds cut(T)
// holds no y with d^ <= T: nothing pruned can be a neighbour or a tie (equal d^, smaller index).
// The squared-sum shortcut of kdtree.hpp (KdEuclid::above) applies to b^ as in a euclidean search.
//
// Outside the domain (a latitude past fl(pi/2) or a longitude past 8 in magnitude) the bound H is not
// proven (half the latitude difference may pass pi/2, and the second-order terms of the longitude
// correction of SPT88 and the rounding of phi1 + phi2 grow with the inputs), so the functions return None.
// One task searches a block of query points from start to end (its size from the number of query points
// alone): the result does not depend on the thread count.
#include "array/ndarray.hpp"
#include "base/buffer.hpp"
#include "base/sincos.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/kdtree.hpp"
#include "spatial/metric.hpp"
#include "spatial/spatial.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

using std::int64_t;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int64_t kParallelWork = 1 << 15;  // array.md §10.4: from here a call splits and drops the GIL
constexpr int64_t kQueryBlock = 64;         // query points a task searches (as KDTree.query, SPT54) ...
constexpr int64_t kQueryBlockMin = 4;       // ... halved down to this while there are fewer than kQueryTasks tasks
constexpr int64_t kQueryTasks = 32;
constexpr int64_t kQueryWorkPerPoint = 64;  // the work units of one query point per coordinate
constexpr int64_t kRowChunk = 1 << 14;      // rows a task converts
constexpr int64_t kLeafsize = 10;           // as the euclidean tree path (SPT73, SPT78)
constexpr int64_t kEntryBytes = 40;         // a found pair while collecting and in the result, rounded up
constexpr double kLatMax = 0x1.921fb54442d18p+0;  // fl(pi / 2)
constexpr double kLonMax = 8.0;

// The query points of a task: kQueryBlock, halved while that leaves fewer than kQueryTasks tasks (size alone;
// as the direct path's chunks, SPT74), so few query points still spread over the threads.
int64_t query_block(int64_t q) {
    int64_t b = kQueryBlock;
    while (b > kQueryBlockMin && q < kQueryTasks * b) b /= 2;
    return b;
}

template <class F>
void run_tasks(int64_t ntasks, bool split, F&& fn) {
    if (!split || ntasks <= 1) {
        for (int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

// The threshold a 3-D bound must exceed so that every point behind it has d^ > T (the top of this file).
inline double sphere_cut(double T) {
    constexpr double c = 1.0 + 0x1p-41;
    const double t = T * c + 0x1p-42;
    if (!(t < 3.14)) return kInf;
    return (2.0 * std::sin(t * 0.5)) * c + 0x1p-46;
}

// The rows of one side: (latitude, longitude, cos latitude) and the point on the sphere.
struct Side {
    const double* geo = nullptr;  // rows of 2 doubles
    int64_t rows = 0;
    std::vector<double> cosv;  // std::cos(latitude): cdist's value (distance.cpp, cosines)
    std::vector<double> pts;   // (rows, 3): P^
    bool inside = true;
};

Side side_of(const NDArray& a, bool split) {
    Side s;
    s.geo = reinterpret_cast<const double*>(a.data);
    s.rows = a.shape[0];
    s.cosv.resize(static_cast<size_t>(s.rows));
    s.pts.resize(static_cast<size_t>(3 * s.rows));
    const int64_t chunks = (s.rows + kRowChunk - 1) / kRowChunk;
    std::vector<char> ok(static_cast<size_t>(chunks), 1);
    run_tasks(chunks, split, [&](int64_t c) {
        for (int64_t i = c * kRowChunk, e = std::min(s.rows, (c + 1) * kRowChunk); i < e; ++i) {
            const double f = s.geo[2 * i], l = s.geo[2 * i + 1];
            if (!(std::fabs(f) <= kLatMax && std::fabs(l) <= kLonMax)) ok[static_cast<size_t>(c)] = 0;
            s.cosv[static_cast<size_t>(i)] = std::cos(f);
            const SinCos a = sincos(f), b = sincos(l);
            double* p = s.pts.data() + 3 * i;
            p[0] = a.c * b.c;
            p[1] = a.c * b.s;
            p[2] = a.s;
        }
    });
    for (char v : ok) s.inside = s.inside && v;
    return s;
}

// The data side: the tree over P^ of XB and, per leaf position, (latitude, longitude, cos latitude).
struct Globe {
    KdTree tree;
    std::vector<double> geo;  // (n, 3) in leaf order
};

Globe globe_of(const Side& b, bool split) {
    Globe g;
    g.tree = kd_build(b.pts.data(), b.rows, 3, 3, kLeafsize);
    g.geo.resize(static_cast<size_t>(3 * b.rows));
    const int64_t chunks = (b.rows + kRowChunk - 1) / kRowChunk;
    run_tasks(chunks, split, [&](int64_t c) {
        for (int64_t i = c * kRowChunk, e = std::min(b.rows, (c + 1) * kRowChunk); i < e; ++i) {
            const int64_t j = g.tree.idx[static_cast<size_t>(i)];
            double* o = g.geo.data() + 3 * i;
            o[0] = b.geo[2 * j];
            o[1] = b.geo[2 * j + 1];
            o[2] = b.cosv[static_cast<size_t>(j)];
        }
    });
    return g;
}

// One query point: its row (latitude, longitude), cos latitude and P^.
struct Query {
    double f, l, c;
    const double* p;
};

// The kernel's squared sum from P^(x) to the point of a node's box nearest it, and whether a bound with
// that sum exceeds T1 (the squared-sum shortcut when the sum is plain, else the distance itself).
class Bounds {
public:
    explicit Bounds(const KdTree& t) : t_(t) {}
    double sum(const double* x, int64_t node) {
        const double *lo = t_.lo(node), *hi = t_.hi(node);
        for (int i = 0; i < 3; ++i) c_[i] = std::min(std::max(x[i], lo[i]), hi[i]);
        return e_.raw(x, c_);
    }
    bool beyond(double s, const double* x, double T1, double above) const {
        if (KdEuclid::plain(s)) return s > above;
        return e_.finish(s, x, c_) > T1;
    }

private:
    const KdTree& t_;
    KdEuclid e_{3};
    double c_[3] = {0.0, 0.0, 0.0};
};

// One query's k nearest by (d^, j) with d^ < ub (KdSearch's walk with the haversine at the leaves).
class Nearest {
public:
    Nearest(const Globe& g, int64_t k) : g_(g), t_(g.tree), b_(g.tree), k_(std::min(k, g.tree.n)) {
        heap_.reserve(static_cast<size_t>(std::max<int64_t>(k_, 1)));
    }

    void run(const Query& q, double ub, std::vector<std::pair<double, int64_t>>& out) {
        heap_.clear();
        q_ = q;
        ub_ = ub;
        retarget();
        if (k_ > 0 && !t_.nodes.empty()) {
            const double s = b_.sum(q_.p, 0);
            if (!b_.beyond(s, q_.p, T1_, above_)) visit(0);
        }
        out.assign(heap_.begin(), heap_.end());
        std::sort(out.begin(), out.end());
    }

private:
    const Globe& g_;
    const KdTree& t_;
    Bounds b_;
    KdEuclid e_{3};
    int64_t k_;
    std::vector<std::pair<double, int64_t>> heap_;  // max-heap by (distance, index)
    Query q_{};
    double ub_ = 0.0, T_ = 0.0, T1_ = 0.0, above_ = 0.0;

    // T: the distance a point must beat; T1, above: its cut and the cut's squared-sum test (cached: the
    // cut takes a sine, and T only moves when the heap's top does).
    void retarget() {
        T_ = static_cast<int64_t>(heap_.size()) < k_ ? ub_ : heap_.front().first;
        T1_ = sphere_cut(T_);
        above_ = KdEuclid::above(T1_);
    }

    void offer(int64_t i) {
        const double* y = t_.data.data() + 3 * i;
        const double s = e_.raw(q_.p, y);
        if (KdEuclid::plain(s) && s > above_) return;  // d^ > T: neither below ub nor before the top
        const double* gy = g_.geo.data() + 3 * i;
        const std::pair<double, int64_t> e{
            haversine_pair(q_.f, q_.l, q_.c, gy[0], gy[1], gy[2]), t_.idx[static_cast<size_t>(i)]
        };
        if (static_cast<int64_t>(heap_.size()) < k_) {
            if (!(e.first < ub_)) return;
            heap_.push_back(e);
            std::push_heap(heap_.begin(), heap_.end());
            if (static_cast<int64_t>(heap_.size()) == k_) retarget();
        } else if (e < heap_.front()) {
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.back() = e;
            std::push_heap(heap_.begin(), heap_.end());
            if (heap_.front().first != T_) retarget();
        }
    }

    void visit(int64_t node) {
        const KdNode& nd = t_.nodes[static_cast<size_t>(node)];
        if (nd.right < 0) {
            for (int64_t i = nd.start; i < nd.end; ++i) offer(i);
            return;
        }
        const int64_t l = node + 1, r = nd.right;
        const double sl = b_.sum(q_.p, l), sr = b_.sum(q_.p, r);
        const bool left_first = !(sr < sl);
        const int64_t first = left_first ? l : r, second = left_first ? r : l;
        const double s1 = left_first ? sl : sr, s2 = left_first ? sr : sl;
        if (!cut(s1, first)) visit(first);
        if (!cut(s2, second)) visit(second);
    }

    // Whether a child whose bound sum s was computed before is cut at the current T (the careful path
    // needs the clamp point again).
    bool cut(double s, int64_t node) {
        if (KdEuclid::plain(s)) return s > above_;
        (void)b_.sum(q_.p, node);
        return b_.beyond(s, q_.p, T1_, above_);
    }
};

struct Hit {
    int64_t j;
    double d;
};

// One query's points with d^ <= r (listing) or their count.
class Within {
public:
    Within(const Globe& g, std::vector<Hit>* out) : g_(g), t_(g.tree), b_(g.tree), out_(out) {}

    int64_t run(const Query& q, double r) {
        q_ = q;
        r_ = r;
        T1_ = sphere_cut(r);
        above_ = KdEuclid::above(T1_);
        count_ = 0;
        if (t_.n > 0) visit(0);
        return count_;
    }

private:
    const Globe& g_;
    const KdTree& t_;
    Bounds b_;
    KdEuclid e_{3};
    std::vector<Hit>* out_;
    Query q_{};
    double r_ = 0.0, T1_ = 0.0, above_ = 0.0;
    int64_t count_ = 0;

    void visit(int64_t node) {
        const double s = b_.sum(q_.p, node);
        if (b_.beyond(s, q_.p, T1_, above_)) return;
        const KdNode& nd = t_.nodes[static_cast<size_t>(node)];
        if (nd.right >= 0) {
            visit(node + 1);
            visit(nd.right);
            return;
        }
        for (int64_t i = nd.start; i < nd.end; ++i) {
            const double sy = e_.raw(q_.p, t_.data.data() + 3 * i);
            if (KdEuclid::plain(sy) && sy > above_) continue;  // d^ > r
            const double* gy = g_.geo.data() + 3 * i;
            const double d = haversine_pair(q_.f, q_.l, q_.c, gy[0], gy[1], gy[2]);
            if (!(d <= r_)) continue;
            ++count_;
            if (out_) out_->push_back({t_.idx[static_cast<size_t>(i)], d});
        }
    }
};

const NDArray& latlon(nb::handle h, const char* func) {
    if (!is_ndarray(h)) throw nb::type_error((std::string(func) + "(): expected sd.ndarrays").c_str());
    const NDArray& a = as_ndarray(h);
    if (a.ndim != 2 || a.dtype != DType::Float64 || !a.c_contiguous || a.shape[1] != 2)
        throw nb::type_error((std::string(func) + "(): wrong layout").c_str());
    return a;
}

nb::object new_array_of(DType t, std::initializer_list<int64_t> shape) {
    const std::vector<int64_t> s(shape);
    return wrap(new_array(t, s));
}

Query query_of(const Side& a, int64_t i) {
    return {a.geo[2 * i], a.geo[2 * i + 1], a.cosv[static_cast<size_t>(i)], a.pts.data() + 3 * i};
}

nb::object sphere_kneighbors(nb::handle xa, nb::handle xb, int64_t k, double upper) {
    const NDArray& A = latlon(xa, "_sphere_kneighbors");
    const NDArray& B = latlon(xb, "_sphere_kneighbors");
    if (k < 1 || !(upper >= 0.0)) throw nb::value_error("_sphere_kneighbors(): wrong k or upper");
    const int64_t ma = A.shape[0], mb = B.shape[0];
    nb::object od = new_array_of(DType::Float64, {ma, k}), oi = new_array_of(DType::Int64, {ma, k});
    double* D = reinterpret_cast<double*>(as_ndarray(od).data);
    int64_t* I = reinterpret_cast<int64_t*>(as_ndarray(oi).data);
    const bool split = (ma + mb) * 3 * kQueryWorkPerPoint >= kParallelWork;
    bool inside = true;
    {
        const parallel::Unlocked unlocked(split);
        const Side a = side_of(A, split);
        const Side b = side_of(B, split);
        inside = a.inside && b.inside;
        if (inside) {
            const Globe g = globe_of(b, split);
            const int64_t qb = query_block(ma), ntasks = (ma + qb - 1) / qb;
            run_tasks(ntasks, split, [&](int64_t task) {
                Nearest s(g, k);
                std::vector<std::pair<double, int64_t>> found;
                for (int64_t i = task * qb, e = std::min(ma, (task + 1) * qb); i < e; ++i) {
                    s.run(query_of(a, i), upper, found);
                    const int64_t got = static_cast<int64_t>(found.size());
                    for (int64_t j = 0; j < k; ++j) {
                        D[i * k + j] = j < got ? found[static_cast<size_t>(j)].first : kInf;
                        I[i * k + j] = j < got ? found[static_cast<size_t>(j)].second : mb;
                    }
                }
            });
        }
    }
    if (!inside) return nb::none();
    return nb::make_tuple(od, oi);
}

[[noreturn]] void too_large() {
    PyErr_SetString(PyExc_MemoryError, "radius_neighbors(): the result is larger than this machine's physical memory");
    throw nb::python_error();
}

nb::object sphere_radius(nb::handle xa, nb::handle xb, nb::handle rh, bool count_only) {
    const NDArray& A = latlon(xa, "_sphere_radius");
    const NDArray& B = latlon(xb, "_sphere_radius");
    if (!is_ndarray(rh)) throw nb::type_error("_sphere_radius(): expected sd.ndarrays");
    const NDArray& rr = as_ndarray(rh);
    const int64_t ma = A.shape[0];
    if (rr.ndim != 1 || rr.dtype != DType::Float64 || !rr.c_contiguous || rr.shape[0] != ma)
        throw nb::type_error("_sphere_radius(): wrong r");
    const auto* R = reinterpret_cast<const double*>(rr.data);
    for (int64_t i = 0; i < ma; ++i)
        if (!(R[i] >= 0.0)) throw nb::value_error("_sphere_radius(): wrong r");
    const bool split = (ma + B.shape[0]) * 3 * kQueryWorkPerPoint >= kParallelWork;
    const int64_t qb = query_block(ma), ntasks = (ma + qb - 1) / qb;
    nb::object oc;
    int64_t* C = nullptr;
    if (count_only) {
        oc = new_array_of(DType::Int64, {ma});
        C = reinterpret_cast<int64_t*>(as_ndarray(oc).data);
    }
    std::vector<std::vector<Hit>> hits(count_only ? 0 : static_cast<size_t>(ntasks));
    std::vector<int64_t> counts(count_only ? 0 : static_cast<size_t>(ma), 0);
    const int64_t budget = static_cast<int64_t>(physical_memory() / kEntryBytes);
    std::atomic<int64_t> total{0};
    std::atomic<bool> over{false};
    bool inside = true;
    {
        const parallel::Unlocked unlocked(split);
        const Side a = side_of(A, split);
        const Side b = side_of(B, split);
        inside = a.inside && b.inside;
        if (inside) {
            const Globe g = globe_of(b, split);
            run_tasks(ntasks, split, [&](int64_t task) {
                const int64_t i0 = task * qb, i1 = std::min(ma, (task + 1) * qb);
                if (count_only) {
                    Within w(g, nullptr);
                    for (int64_t i = i0; i < i1; ++i) C[i] = w.run(query_of(a, i), R[i]);
                    return;
                }
                std::vector<Hit>& h = hits[static_cast<size_t>(task)];
                Within w(g, &h);
                for (int64_t i = i0; i < i1; ++i) {
                    if (over.load(std::memory_order_relaxed)) return;
                    const size_t from = h.size();
                    w.run(query_of(a, i), R[i]);
                    std::sort(h.begin() + static_cast<std::ptrdiff_t>(from), h.end(), [](const Hit& u, const Hit& v) {
                        return u.j < v.j;
                    });
                    const int64_t c = static_cast<int64_t>(h.size() - from);
                    counts[static_cast<size_t>(i)] = c;
                    if (c && total.fetch_add(c, std::memory_order_relaxed) + c > budget) over.store(true);
                }
            });
        }
    }
    if (!inside) return nb::none();
    if (count_only) return oc;
    if (over.load()) too_large();
    int64_t nnz = 0;
    for (int64_t c : counts) nnz += c;
    nb::object op = new_array_of(DType::Int64, {ma + 1}), oi = new_array_of(DType::Int64, {nnz}),
               od = new_array_of(DType::Float64, {nnz});
    int64_t* P = reinterpret_cast<int64_t*>(as_ndarray(op).data);
    int64_t* I = reinterpret_cast<int64_t*>(as_ndarray(oi).data);
    double* D = reinterpret_cast<double*>(as_ndarray(od).data);
    P[0] = 0;
    for (int64_t i = 0; i < ma; ++i) P[i + 1] = P[i] + counts[static_cast<size_t>(i)];
    {
        const bool copy_split = split && nnz >= kParallelWork;
        const parallel::Unlocked unlocked(copy_split);
        run_tasks(ntasks, copy_split, [&](int64_t task) {
            int64_t dst = P[task * qb];
            for (const Hit& e : hits[static_cast<size_t>(task)]) {
                I[dst] = e.j;
                D[dst++] = e.d;
            }
        });
    }
    return nb::make_tuple(op, oi, od);
}

}  // namespace

void bind_sphere(nb::module_& m) {
    m.def("_sphere_kneighbors", &sphere_kneighbors, nb::arg("XA"), nb::arg("XB"), nb::arg("k"), nb::arg("upper"));
    m.def("_sphere_radius", &sphere_radius, nb::arg("XA"), nb::arg("XB"), nb::arg("r"), nb::arg("count_only"));
}

}  // namespace sd::spatial
