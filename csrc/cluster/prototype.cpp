// Prototypes of sd.cluster's kernels for milestone CL0's measurements (docs/design/cluster.md §3.2, §3.3, §5),
// timed and checked by tools/cluster_probe.py (section proto). Not public: the private names _proto_* of
// _core.cluster; milestones CL1 and CL2 replace them by the area's kernels and remove this file.
//
// _proto_assign(X, C, form) -> (n,) int64
//     the nearest centre of every point, the smaller centre on a tie. form 0: the difference form, the sums
//     sum_j (x_j - c_j)^2 of spatial's pair kernel (csrc/spatial/metric.hpp: tile<SqOp> on panels of 8 centres,
//     so a sum has the bits of sd.spatial.cdist(metric='sqeuclidean')); form 1: the expanded form
//     ||x||^2 - 2 x.c + ||c||^2 with Accelerate's dgemm for x.c (scikit-learn's form); form 2: the difference
//     form one pair at a time (pair_scalar<SqOp>, the same bits as form 0); form 3: the expanded form as a filter
//     with a proved error bound and the difference form on the centres it cannot rule out (the bits of form 0;
//     _proto_candidates() counts those). Points in tasks of 1024 (forms 0, 2) or blocks of 4096 rows per dgemm
//     (form 1, Accelerate's own threads; form 3 tasks of 4096 points, each its own dgemm on the pool).
// _proto_kmeans(X, C0, max_iter, accel, margin) -> (labels, centres, n_iter, n_dist)
//     Lloyd's iteration from C0 until no label changes (or max_iter): accel 0 every distance (form 0), accel 1
//     Hamerly's bounds (2010) on the true distances, kept with the rounding margin gamma = margin (p + 4) u of
//     the computed squared distances (U rounded up, L down) and a point skipped only when L > U (1 + 2 gamma)
//     (1 + 4u), which proves every other centre's computed sum larger: then both give the same labels and,
//     with one summation for the centres, the same bits. margin 0 drops the rounding (an unproved skip).
//     The centres are the means of chunk sums of 4096 points added in chunk order (a pairwise tree), so they do
//     not depend on the thread count. An empty cluster keeps its centre. n_dist counts the distances computed.
// _proto_mst(X, method) -> (edges (n - 1, 2) int64, weights (n - 1,))
//     the Euclidean minimum spanning tree as squared distances (spatial's pair kernel), the edges sorted by
//     (weight, min, max): method 0 Prim without a matrix (O(n^2 p), the vertex of smallest (distance, index)
//     next), 1 Boruvka on a KD tree (each round every point's nearest point of another component, the nodes
//     inside its own component or farther than the best cut with a rounding margin), 2 Boruvka by a scan.
//     All three pick the same edges: every comparison is of the same computed sums under one total order.
// _proto_nnchain(X, method, check) -> (Z (n - 1, 4), tie)
//     the nearest-neighbour chain (Murtagh 1983, scipy's nn_chain) on the condensed matrix of Euclidean
//     distances with the Lance-Williams updates: method 1 complete, 2 average, 3 weighted, 4 ward. The
//     neighbour is the smallest (distance, index); check 1 also records whether a scan met an exact tie at
//     its minimum (tie = 1). Z sorted by height (stable), as scipy.
// _proto_heights(X, Z, method) -> (n - 1,)
//     the heights of a tree recomputed by the definition: method 2 average (the sum of the |A||B| Euclidean
//     distances: a row of A's point against B's points in index order, the rows by a pairwise tree, over
//     |A||B|), 4 ward (the centroids by pairwise sums, sqrt(2 |A||B| / (|A| + |B|)) ||c_A - c_B||). Distances
//     here and in the chain are spatial's Euclidean with its careful path for tiny and huge sums.
#include "array/ndarray.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "cluster/cluster.hpp"
#include "parallel/pool.hpp"
#include "spatial/metric.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace sd::cluster {

namespace nb = nanobind;

namespace {

using std::int64_t;
constexpr double kU = 0x1p-53;
constexpr int64_t kChunk = 4096;
constexpr int64_t kAssignTask = 1024;
int64_t last_candidates = 0;  // form 3's difference-form distances in the last call (_proto_candidates)

const double* rows_of(nb::handle o, const char* what, int64_t& n, int64_t& p) {
    NDArray& a = require_ndarray(o, what);
    if (a.dtype != DType::Float64 || !a.c_contiguous || a.ndim != 2) throw nb::value_error("needs C (n, p) float64");
    n = a.shape[0];
    p = a.shape[1];
    return reinterpret_cast<const double*>(a.data);
}

inline double sq(const double* x, const double* y, int64_t p) {
    double s;
    spatial::pair_scalar<spatial::SqOp, false>(x, y, nullptr, p, &s);
    return s;
}

// The Euclidean distance with spatial's careful path (the bits of sd.spatial.cdist): subnormal differences scaled.
inline double dist(const double* x, const double* y, int64_t p) {
    return spatial::euclid_finish<false>(sq(x, y, p), x, y, nullptr, p);
}

// ---------------------------------------------------------------- assignment

// labels[i0 .. i1) by the difference form: 4 points against panels of 8 centres (tile<SqOp, 4>).
void assign_tile(
    const double* X, int64_t p, const double* panels, int64_t k, int64_t i0, int64_t i1, int64_t* lab, double* best_out
) {
    const int64_t np = (k + spatial::kPanel - 1) / spatial::kPanel;
    double out[4 * spatial::kPanel];
    for (int64_t i = i0; i < i1; i += 4) {
        const int R = static_cast<int>(std::min<int64_t>(4, i1 - i));
        const double* xr[4];
        for (int r = 0; r < 4; ++r) xr[r] = X + (i + std::min(r, R - 1)) * p;
        double best[4];
        int64_t arg[4];
        for (int r = 0; r < 4; ++r) best[r] = std::numeric_limits<double>::infinity(), arg[r] = 0;
        for (int64_t b = 0; b < np; ++b) {
            spatial::tile<spatial::SqOp, 4, false>(xr, panels + b * p * spatial::kPanel, nullptr, p, out);
            const int64_t cmax = std::min<int64_t>(spatial::kPanel, k - b * spatial::kPanel);
            for (int r = 0; r < R; ++r)
                for (int64_t c = 0; c < cmax; ++c) {
                    const double v = out[r * spatial::kPanel + c];
                    if (v < best[r]) best[r] = v, arg[r] = b * spatial::kPanel + c;
                }
        }
        for (int r = 0; r < R; ++r) {
            lab[i + r] = arg[r];
            if (best_out) best_out[i + r] = best[r];
        }
    }
}

std::vector<double> pack_centres(const double* C, int64_t k, int64_t p) {
    const int64_t np = (k + spatial::kPanel - 1) / spatial::kPanel;
    std::vector<double> panels(static_cast<size_t>(np * p * spatial::kPanel));
    for (int64_t b = 0; b < np; ++b)
        spatial::pack_panel(
            C + b * spatial::kPanel * p, p, std::min<int64_t>(spatial::kPanel, k - b * spatial::kPanel), p,
            panels.data() + b * p * spatial::kPanel
        );
    return panels;
}

void assign_expanded(const double* X, int64_t n, int64_t p, const double* C, int64_t k, int64_t* lab) {
    std::vector<double> cn(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
        double s = 0;
        for (int64_t d = 0; d < p; ++d) s += C[j * p + d] * C[j * p + d];
        cn[static_cast<size_t>(j)] = s;
    }
    std::vector<double> G(static_cast<size_t>(kChunk * k));
    for (int64_t i0 = 0; i0 < n; i0 += kChunk) {
        const int64_t m = std::min(kChunk, n - i0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, p, 1.0, X + i0 * p, p, C, p, 0.0, G.data(), k);
        for (int64_t r = 0; r < m; ++r) {
            const double* x = X + (i0 + r) * p;
            double xn = 0;
            for (int64_t d = 0; d < p; ++d) xn += x[d] * x[d];
            double best = std::numeric_limits<double>::infinity();
            int64_t arg = 0;
            for (int64_t j = 0; j < k; ++j) {
                const double v = xn - 2.0 * G[static_cast<size_t>(r * k + j)] + cn[static_cast<size_t>(j)];
                if (v < best) best = v, arg = j;
            }
            lab[i0 + r] = arg;
        }
    }
}

// form 3: the expanded form (dgemm) as a filter, then the difference form on the candidates it cannot rule out.
// The expanded value v~ of a true squared distance d is within E = (p + 4) u (||x|| + ||c||)^2 of d (the dot
// product's p terms, the norms' and the two additions); the difference form's d^ within (p + 4) u d. A centre j is
// ruled out when v~_j - E_j > (v~_b + E_b) (1 + 2 (p + 4) u) + tiny for the centre b of smallest v~: then
// d_j > d_b (1 + 2 g), so d^_j > d^_b and j is not the difference form's argmin. Every other centre gets the
// difference form, and the label is their smallest (d^, index): the bits of form 0.
void assign_filtered(const double* X, int64_t n, int64_t p, const double* C, int64_t k, int64_t* lab, int64_t* cands) {
    const double g = static_cast<double>(p + 4) * kU;
    std::vector<double> cn(static_cast<size_t>(k)), cr(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
        double s = 0;
        for (int64_t d = 0; d < p; ++d) s += C[j * p + d] * C[j * p + d];
        cn[static_cast<size_t>(j)] = s;
        cr[static_cast<size_t>(j)] = std::sqrt(s) * (1.0 + g);
    }
    constexpr int64_t kRows = 4096;  // a task's points: one dgemm of kRows x k into the task's own buffer
    const int64_t nt = (n + kRows - 1) / kRows;
    std::vector<int64_t> counts(static_cast<size_t>(nt), 0);
    parallel::parallel_for(nt, [&](int64_t t) {
        const int64_t i0 = t * kRows, m = std::min(kRows, n - i0);
        std::vector<double> G(static_cast<size_t>(m * k)), e(static_cast<size_t>(k));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, p, 1.0, X + i0 * p, p, C, p, 0.0, G.data(), k);
        int64_t cnt = 0;
        for (int64_t r = 0; r < m; ++r) {
            const double* x = X + (i0 + r) * p;
            double* v = G.data() + r * k;
            double xn = 0;
            for (int64_t d = 0; d < p; ++d) xn += x[d] * x[d];
            const double xr = std::sqrt(xn) * (1.0 + g);
            int64_t b = 0;
            for (int64_t j = 0; j < k; ++j) {
                v[j] = xn - 2.0 * v[j] + cn[static_cast<size_t>(j)];
                const double rr = xr + cr[static_cast<size_t>(j)];
                e[static_cast<size_t>(j)] = 2.0 * g * rr * rr;
                if (v[j] < v[b]) b = j;
            }
            const double lim = (v[b] + e[static_cast<size_t>(b)]) * (1.0 + 4.0 * g) + 0x1p-1000;
            double best = std::numeric_limits<double>::infinity();
            int64_t arg = 0;
            for (int64_t j = 0; j < k; ++j) {
                if (v[j] - e[static_cast<size_t>(j)] > lim) continue;
                ++cnt;
                const double s = sq(x, C + j * p, p);
                if (s < best) best = s, arg = j;
            }
            lab[i0 + r] = arg;
        }
        counts[static_cast<size_t>(t)] = cnt;
    });
    *cands = std::accumulate(counts.begin(), counts.end(), int64_t{0});
}

nb::object proto_assign(nb::handle Xo, nb::handle Co, int form) {
    int64_t n, p, k, pc;
    const double* X = rows_of(Xo, "X", n, p);
    const double* C = rows_of(Co, "C", k, pc);
    if (pc != p || k < 1) throw nb::value_error("C must be (k, p)");
    const int64_t shape[1] = {n};
    NDArray out = new_array(DType::Int64, shape, false);
    auto* lab = reinterpret_cast<int64_t*>(out.data);
    {
        const parallel::Unlocked unlocked(true);
        if (form == 1) {
            assign_expanded(X, n, p, C, k, lab);
        } else if (form == 3) {
            int64_t cands = 0;
            assign_filtered(X, n, p, C, k, lab, &cands);
            last_candidates = cands;
        } else {
            const std::vector<double> panels = pack_centres(C, k, p);
            const int64_t ntask = (n + kAssignTask - 1) / kAssignTask;
            parallel::parallel_for(ntask, [&](int64_t t) {
                const int64_t i0 = t * kAssignTask, i1 = std::min(n, i0 + kAssignTask);
                if (form == 0) {
                    assign_tile(X, p, panels.data(), k, i0, i1, lab, nullptr);
                } else {
                    for (int64_t i = i0; i < i1; ++i) {
                        double best = std::numeric_limits<double>::infinity();
                        int64_t arg = 0;
                        for (int64_t j = 0; j < k; ++j) {
                            const double v = sq(X + i * p, C + j * p, p);
                            if (v < best) best = v, arg = j;
                        }
                        lab[i] = arg;
                    }
                }
            });
        }
    }
    return wrap(std::move(out));
}

// ---------------------------------------------------------------- k-means

// The centres: chunk sums (chunk c: points [c kChunk, (c + 1) kChunk)), added by a pairwise tree in chunk order.
void centres(const double* X, int64_t n, int64_t p, const int64_t* lab, int64_t k, const double* old, double* C) {
    const int64_t nc = (n + kChunk - 1) / kChunk;
    std::vector<double> sums(static_cast<size_t>(nc * k * p), 0.0);
    std::vector<int64_t> counts(static_cast<size_t>(nc * k), 0);
    parallel::parallel_for(nc, [&](int64_t c) {
        double* s = sums.data() + c * k * p;
        int64_t* m = counts.data() + c * k;
        for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) {
            const int64_t j = lab[i];
            ++m[j];
            for (int64_t d = 0; d < p; ++d) s[j * p + d] += X[i * p + d];
        }
    });
    for (int64_t step = 1; step < nc; step *= 2)  // pairwise: chunk c += chunk c + step
        for (int64_t c = 0; c + step < nc; c += 2 * step) {
            for (int64_t q = 0; q < k * p; ++q)
                sums[static_cast<size_t>(c * k * p + q)] += sums[static_cast<size_t>((c + step) * k * p + q)];
            for (int64_t q = 0; q < k; ++q)
                counts[static_cast<size_t>(c * k + q)] += counts[static_cast<size_t>((c + step) * k + q)];
        }
    for (int64_t j = 0; j < k; ++j)
        for (int64_t d = 0; d < p; ++d)
            C[j * p + d] = counts[static_cast<size_t>(j)] ? sums[static_cast<size_t>(j * p + d)] /
                                                                static_cast<double>(counts[static_cast<size_t>(j)])
                                                          : old[j * p + d];
}

inline double up(double v, int times = 2) {
    for (int i = 0; i < times; ++i) v = std::nextafter(v, std::numeric_limits<double>::infinity());
    return v;
}
inline double down(double v, int times = 2) {
    for (int i = 0; i < times; ++i) v = std::nextafter(v, -std::numeric_limits<double>::infinity());
    return v > 0 ? v : 0.0;
}

nb::object proto_kmeans(nb::handle Xo, nb::handle Co, int64_t max_iter, int accel, double margin) {
    int64_t n, p, k, pc;
    const double* X = rows_of(Xo, "X", n, p);
    const double* C0 = rows_of(Co, "C0", k, pc);
    if (pc != p || k < 1) throw nb::value_error("C0 must be (k, p)");
    const int64_t s1[1] = {n};
    const int64_t s2[2] = {k, p};
    NDArray lab_a = new_array(DType::Int64, s1, false);
    NDArray cen_a = new_array(DType::Float64, s2, false);
    auto* lab = reinterpret_cast<int64_t*>(lab_a.data);
    auto* C = reinterpret_cast<double*>(cen_a.data);
    int64_t iters = 0;
    std::atomic<int64_t> ndist{0};
    {
        const parallel::Unlocked unlocked(true);
        std::copy(C0, C0 + k * p, C);
        std::vector<double> Cold(static_cast<size_t>(k * p)), U(static_cast<size_t>(n)), L(static_cast<size_t>(n)),
            delta(static_cast<size_t>(k)), best(static_cast<size_t>(n));
        const double gamma = margin * static_cast<double>(p + 4) * kU;
        const double skip = (1.0 + 2.0 * gamma) * (1.0 + 4.0 * kU);
        const int64_t ntask = (n + kAssignTask - 1) / kAssignTask;
        // a full scan of point i: label, upper bound of the nearest's distance, lower bound of the second's
        auto scan = [&](int64_t i, const double* Cc) {
            double b1 = std::numeric_limits<double>::infinity(), b2 = b1;
            int64_t a = 0;
            for (int64_t j = 0; j < k; ++j) {
                const double v = sq(X + i * p, Cc + j * p, p);
                if (v < b1) b2 = b1, b1 = v, a = j;
                else if (v < b2) b2 = v;
            }
            lab[i] = a;
            U[static_cast<size_t>(i)] = up(std::sqrt(b1) * (1.0 + gamma));
            L[static_cast<size_t>(i)] =
                k > 1 ? down(std::sqrt(b2) * (1.0 - gamma)) : std::numeric_limits<double>::infinity();
        };
        if (accel) {
            parallel::parallel_for(ntask, [&](int64_t t) {
                for (int64_t i = t * kAssignTask; i < std::min(n, (t + 1) * kAssignTask); ++i) scan(i, C);
            });
            ndist += n * k;
        } else {
            const std::vector<double> panels = pack_centres(C, k, p);
            parallel::parallel_for(ntask, [&](int64_t t) {
                assign_tile(X, p, panels.data(), k, t * kAssignTask, std::min(n, (t + 1) * kAssignTask), lab, nullptr);
            });
            ndist += n * k;
        }
        std::vector<int64_t> prev(lab, lab + n);
        for (iters = 1; iters <= max_iter; ++iters) {
            std::copy(C, C + k * p, Cold.begin());
            centres(X, n, p, lab, k, Cold.data(), C);
            if (accel) {
                double m1 = 0, m2 = 0;
                int64_t a1 = -1;
                for (int64_t j = 0; j < k; ++j) {
                    const double s = sq(Cold.data() + j * p, C + j * p, p);
                    const double dj = margin > 0 ? up(std::sqrt(s) * (1.0 + gamma)) : std::sqrt(s);
                    delta[static_cast<size_t>(j)] = dj;
                    if (dj > m1) m2 = m1, m1 = dj, a1 = j;
                    else if (dj > m2) m2 = dj;
                }
                std::atomic<int64_t> local{0};
                parallel::parallel_for(ntask, [&](int64_t t) {
                    int64_t cnt = 0;
                    for (int64_t i = t * kAssignTask; i < std::min(n, (t + 1) * kAssignTask); ++i) {
                        const auto ii = static_cast<size_t>(i);
                        const int64_t a = lab[i];
                        U[ii] = margin > 0 ? up(U[ii] + delta[static_cast<size_t>(a)], 1)
                                           : U[ii] + delta[static_cast<size_t>(a)];
                        const double dm = a == a1 ? m2 : m1;
                        L[ii] = margin > 0 ? down(L[ii] - dm, 1) : std::max(0.0, L[ii] - dm);
                        if (L[ii] > U[ii] * skip) continue;
                        // tighten U with the computed distance to the own centre first
                        const double s = sq(X + i * p, C + a * p, p);
                        ++cnt;
                        U[ii] = margin > 0 ? up(std::sqrt(s) * (1.0 + gamma)) : std::sqrt(s);
                        if (L[ii] > U[ii] * skip) continue;
                        scan(i, C);
                        cnt += k;
                    }
                    local += cnt;
                });
                ndist += local.load();
            } else {
                const std::vector<double> panels = pack_centres(C, k, p);
                parallel::parallel_for(ntask, [&](int64_t t) {
                    assign_tile(
                        X, p, panels.data(), k, t * kAssignTask, std::min(n, (t + 1) * kAssignTask), lab, nullptr
                    );
                });
                ndist += n * k;
            }
            if (std::equal(prev.begin(), prev.end(), lab)) break;
            std::copy(lab, lab + n, prev.begin());
        }
        if (iters > max_iter) iters = max_iter;
    }
    return nb::make_tuple(wrap(std::move(lab_a)), wrap(std::move(cen_a)), iters, ndist.load());
}

// ---------------------------------------------------------------- minimum spanning trees

struct Edge {
    double w;
    int64_t a, b;  // a < b
};
inline bool less(const Edge& x, const Edge& y) {
    if (x.w != y.w) return x.w < y.w;
    if (x.a != y.a) return x.a < y.a;
    return x.b < y.b;
}

nb::object edges_out(std::vector<Edge>& E) {
    std::sort(E.begin(), E.end(), less);
    const int64_t m = static_cast<int64_t>(E.size());
    const int64_t s2[2] = {m, 2}, s1[1] = {m};
    NDArray e = new_array(DType::Int64, s2, false), w = new_array(DType::Float64, s1, false);
    auto* ep = reinterpret_cast<int64_t*>(e.data);
    auto* wp = reinterpret_cast<double*>(w.data);
    for (int64_t i = 0; i < m; ++i) {
        ep[2 * i] = E[static_cast<size_t>(i)].a;
        ep[2 * i + 1] = E[static_cast<size_t>(i)].b;
        wp[i] = E[static_cast<size_t>(i)].w;
    }
    return nb::make_tuple(wrap(std::move(e)), wrap(std::move(w)));
}

std::vector<Edge> prim(const double* X, int64_t n, int64_t p) {
    std::vector<Edge> E;
    if (n < 2) return E;
    std::vector<double> dmin(static_cast<size_t>(n), std::numeric_limits<double>::infinity());
    std::vector<int64_t> from(static_cast<size_t>(n), -1);
    std::vector<char> in(static_cast<size_t>(n), 0);
    int64_t cur = 0;
    in[0] = 1;
    const int64_t task = std::max<int64_t>(256, (1 << 15) / std::max<int64_t>(p, 1));
    const int64_t ntask = (n + task - 1) / task;
    std::vector<double> tb(static_cast<size_t>(ntask));
    std::vector<int64_t> ti(static_cast<size_t>(ntask));
    for (int64_t step = 1; step < n; ++step) {
        auto body = [&](int64_t t) {
            double b = std::numeric_limits<double>::infinity();
            int64_t bi = -1;
            for (int64_t v = t * task; v < std::min(n, (t + 1) * task); ++v) {
                const auto vv = static_cast<size_t>(v);
                if (in[vv]) continue;
                const double s = sq(X + cur * p, X + v * p, p);
                if (s < dmin[vv] || (s == dmin[vv] && cur < from[vv])) dmin[vv] = s, from[vv] = cur;
                if (bi < 0 || dmin[vv] < b) b = dmin[vv], bi = v;  // the smallest (distance, index)
            }
            tb[static_cast<size_t>(t)] = b;
            ti[static_cast<size_t>(t)] = bi;
        };
        if (n * p >= (1 << 15)) parallel::parallel_for(ntask, body);
        else
            for (int64_t t = 0; t < ntask; ++t) body(t);
        double b = std::numeric_limits<double>::infinity();
        int64_t bi = -1;
        for (int64_t t = 0; t < ntask; ++t)
            if (ti[static_cast<size_t>(t)] >= 0 && (bi < 0 || tb[static_cast<size_t>(t)] < b))
                b = tb[static_cast<size_t>(t)], bi = ti[static_cast<size_t>(t)];
        in[static_cast<size_t>(bi)] = 1;
        E.push_back({b, std::min(bi, from[static_cast<size_t>(bi)]), std::max(bi, from[static_cast<size_t>(bi)])});
        cur = bi;
    }
    return E;
}

struct Tree {
    int64_t p = 0;
    std::vector<double> pts;  // leaf order
    std::vector<int64_t> idx;
    struct Node {
        int64_t lo, hi, left, right;
    };
    std::vector<Node> nodes;
    std::vector<double> box;  // 2p per node
    int64_t build(int64_t lo, int64_t hi, std::vector<int64_t>& order, const double* X) {
        const int64_t id = static_cast<int64_t>(nodes.size());
        nodes.push_back({lo, hi, -1, -1});
        box.resize(static_cast<size_t>((id + 1) * 2 * p));
        double* bl = box.data() + id * 2 * p;
        for (int64_t d = 0; d < p; ++d) bl[d] = std::numeric_limits<double>::infinity(), bl[p + d] = -bl[d];
        for (int64_t i = lo; i < hi; ++i)
            for (int64_t d = 0; d < p; ++d) {
                const double v = X[order[static_cast<size_t>(i)] * p + d];
                bl[d] = std::min(bl[d], v);
                bl[p + d] = std::max(bl[p + d], v);
            }
        if (hi - lo > 16) {
            int64_t ax = 0;
            for (int64_t d = 1; d < p; ++d)
                if (bl[p + d] - bl[d] > bl[p + ax] - bl[ax]) ax = d;
            const int64_t mid = (lo + hi) / 2;
            std::nth_element(order.begin() + lo, order.begin() + mid, order.begin() + hi, [&](int64_t a, int64_t b) {
                const double va = X[a * p + ax], vb = X[b * p + ax];
                return va < vb || (va == vb && a < b);
            });
            const int64_t l = build(lo, mid, order, X);
            const int64_t r = build(mid, hi, order, X);
            nodes[static_cast<size_t>(id)].left = l;
            nodes[static_cast<size_t>(id)].right = r;
        }
        return id;
    }
};

int64_t find(std::vector<int64_t>& par, int64_t x) {
    while (par[static_cast<size_t>(x)] != x) {
        par[static_cast<size_t>(x)] = par[static_cast<size_t>(par[static_cast<size_t>(x)])];
        x = par[static_cast<size_t>(x)];
    }
    return x;
}

std::vector<Edge> boruvka(const double* X, int64_t n, int64_t p, bool tree) {
    std::vector<Edge> E;
    if (n < 2) return E;
    Tree T;
    T.p = p;
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    if (tree) T.build(0, n, order, X);
    std::vector<int64_t> par(static_cast<size_t>(n)), comp(static_cast<size_t>(n));
    std::iota(par.begin(), par.end(), 0);
    const double cut = 1.0 + 2.0 * static_cast<double>(p + 4) * kU;  // a node's computed bound vs the best sum
    std::vector<Edge> best_of(static_cast<size_t>(n));
    std::vector<int64_t> ncomp;  // per node: the component of all its points, or -1
    int64_t components = n;
    while (components > 1) {
        for (int64_t i = 0; i < n; ++i) comp[static_cast<size_t>(i)] = find(par, i);
        if (tree) {
            ncomp.assign(T.nodes.size(), -1);
            for (int64_t id = static_cast<int64_t>(T.nodes.size()) - 1; id >= 0; --id) {
                const auto& nd = T.nodes[static_cast<size_t>(id)];
                if (nd.left < 0) {
                    int64_t c = comp[static_cast<size_t>(order[static_cast<size_t>(nd.lo)])];
                    for (int64_t i = nd.lo + 1; i < nd.hi && c >= 0; ++i)
                        if (comp[static_cast<size_t>(order[static_cast<size_t>(i)])] != c) c = -1;
                    ncomp[static_cast<size_t>(id)] = c;
                } else {
                    const int64_t a = ncomp[static_cast<size_t>(nd.left)], b = ncomp[static_cast<size_t>(nd.right)];
                    ncomp[static_cast<size_t>(id)] = (a >= 0 && a == b) ? a : -1;
                }
            }
        }
        const int64_t task = 256, ntask = (n + task - 1) / task;
        parallel::parallel_for(ntask, [&](int64_t t) {
            std::vector<int64_t> stack;
            std::vector<double> c(static_cast<size_t>(p));
            for (int64_t i = t * task; i < std::min(n, (t + 1) * task); ++i) {
                const int64_t ci = comp[static_cast<size_t>(i)];
                const double* x = X + i * p;
                Edge b{std::numeric_limits<double>::infinity(), -1, -1};
                auto offer = [&](int64_t j) {
                    if (comp[static_cast<size_t>(j)] == ci) return;
                    const Edge e{sq(x, X + j * p, p), std::min(i, j), std::max(i, j)};
                    if (b.a < 0 || less(e, b)) b = e;
                };
                if (!tree) {
                    for (int64_t j = 0; j < n; ++j) offer(j);
                } else {
                    stack.assign(1, 0);
                    while (!stack.empty()) {
                        const int64_t id = stack.back();
                        stack.pop_back();
                        if (ncomp[static_cast<size_t>(id)] == ci) continue;
                        const double* bl = T.box.data() + id * 2 * p;
                        for (int64_t d = 0; d < p; ++d)
                            c[static_cast<size_t>(d)] = std::min(std::max(x[d], bl[d]), bl[p + d]);
                        if (b.a >= 0 && sq(x, c.data(), p) > b.w * cut) continue;
                        const auto& nd = T.nodes[static_cast<size_t>(id)];
                        if (nd.left < 0) {
                            for (int64_t q = nd.lo; q < nd.hi; ++q) offer(order[static_cast<size_t>(q)]);
                        } else {
                            // the nearer child last (popped first)
                            const double* l = T.box.data() + nd.left * 2 * p;
                            const double* r = T.box.data() + nd.right * 2 * p;
                            double dl = 0, dr = 0;
                            for (int64_t d = 0; d < p; ++d) {
                                const double zl = x[d] < l[d] ? l[d] - x[d] : (x[d] > l[p + d] ? x[d] - l[p + d] : 0);
                                const double zr = x[d] < r[d] ? r[d] - x[d] : (x[d] > r[p + d] ? x[d] - r[p + d] : 0);
                                dl += zl * zl;
                                dr += zr * zr;
                            }
                            if (dl <= dr) stack.push_back(nd.right), stack.push_back(nd.left);
                            else stack.push_back(nd.left), stack.push_back(nd.right);
                        }
                    }
                }
                best_of[static_cast<size_t>(i)] = b;
            }
        });
        // the smallest edge of each component (the points in index order: a fixed reduction)
        std::vector<Edge> cb(static_cast<size_t>(n), Edge{std::numeric_limits<double>::infinity(), -1, -1});
        for (int64_t i = 0; i < n; ++i) {
            const Edge& e = best_of[static_cast<size_t>(i)];
            Edge& c = cb[static_cast<size_t>(comp[static_cast<size_t>(i)])];
            if (e.a >= 0 && (c.a < 0 || less(e, c))) c = e;
        }
        for (int64_t r = 0; r < n; ++r) {
            const Edge& e = cb[static_cast<size_t>(r)];
            if (e.a < 0) continue;
            const int64_t a = find(par, e.a), b = find(par, e.b);
            if (a == b) continue;
            par[static_cast<size_t>(std::max(a, b))] = std::min(a, b);
            E.push_back(e);
            --components;
        }
    }
    return E;
}

nb::object proto_mst(nb::handle Xo, int method) {
    int64_t n, p;
    const double* X = rows_of(Xo, "X", n, p);
    std::vector<Edge> E;
    {
        const parallel::Unlocked unlocked(true);
        E = method == 0 ? prim(X, n, p) : boruvka(X, n, p, method == 1);
    }
    return edges_out(E);
}

// ---------------------------------------------------------------- NN-chain and heights

inline int64_t cidx(int64_t n, int64_t i, int64_t j) {  // condensed index, i < j
    return n * i - i * (i + 1) / 2 + (j - i - 1);
}

nb::object proto_nnchain(nb::handle Xo, int method, int check) {
    int64_t n, p;
    const double* X = rows_of(Xo, "X", n, p);
    const int64_t s2[2] = {n - 1, 4};
    NDArray Za = new_array(DType::Float64, s2, false);
    auto* Z = reinterpret_cast<double*>(Za.data);
    int tie = 0;
    {
        const parallel::Unlocked unlocked(true);
        std::vector<double> D(static_cast<size_t>(n * (n - 1) / 2));
        const int64_t task = 64;
        parallel::parallel_for((n + task - 1) / task, [&](int64_t t) {
            for (int64_t i = t * task; i < std::min(n, (t + 1) * task); ++i)
                for (int64_t j = i + 1; j < n; ++j)
                    D[static_cast<size_t>(cidx(n, i, j))] = dist(X + i * p, X + j * p, p);
        });
        std::vector<int64_t> size(static_cast<size_t>(n), 1), chain;
        std::vector<char> alive(static_cast<size_t>(n), 1);
        struct M {
            int64_t a, b;
            double h;
            int64_t s;
        };
        std::vector<M> merges;
        auto dist = [&](int64_t i, int64_t j) -> double& {
            return D[static_cast<size_t>(i < j ? cidx(n, i, j) : cidx(n, j, i))];
        };
        for (int64_t k = 0; k < n - 1; ++k) {
            if (chain.empty()) {
                for (int64_t i = 0; i < n; ++i)
                    if (alive[static_cast<size_t>(i)]) {
                        chain.push_back(i);
                        break;
                    }
            }
            int64_t x = 0, y = 0;
            double cur;
            while (true) {
                x = chain.back();
                int64_t yb = -1;
                double best = std::numeric_limits<double>::infinity();
                if (chain.size() > 1) yb = chain[chain.size() - 2], best = dist(x, yb);
                for (int64_t i = 0; i < n; ++i) {
                    if (!alive[static_cast<size_t>(i)] || i == x) continue;
                    const double v = dist(x, i);
                    if (v < best || (v == best && i < yb)) {
                        if (check && v == best && yb >= 0) tie = 1;
                        best = v, yb = i;
                    } else if (check && v == best && i != yb) {
                        tie = 1;
                    }
                }
                y = yb;
                cur = best;
                if (chain.size() > 1 && y == chain[chain.size() - 2]) break;
                chain.push_back(y);
            }
            chain.pop_back();
            chain.pop_back();
            if (x > y) std::swap(x, y);
            const double nx = static_cast<double>(size[static_cast<size_t>(x)]),
                         ny = static_cast<double>(size[static_cast<size_t>(y)]);
            merges.push_back({x, y, cur, size[static_cast<size_t>(x)] + size[static_cast<size_t>(y)]});
            alive[static_cast<size_t>(x)] = 0;  // the merged cluster lives at y
            for (int64_t i = 0; i < n; ++i) {
                if (!alive[static_cast<size_t>(i)] || i == y) continue;
                const double dx = dist(x, i), dy = dist(y, i), ni = static_cast<double>(size[static_cast<size_t>(i)]);
                double v;
                switch (method) {
                    case 1: v = std::max(dx, dy); break;
                    case 2: v = (nx * dx + ny * dy) / (nx + ny); break;
                    case 3: v = 0.5 * (dx + dy); break;
                    default: {
                        const double t = 1.0 / (nx + ny + ni);
                        v = std::sqrt((nx + ni) * t * dx * dx + (ny + ni) * t * dy * dy - ni * t * cur * cur);
                    }
                }
                dist(y, i) = v;
            }
            size[static_cast<size_t>(y)] += size[static_cast<size_t>(x)];
        }
        // label the clusters as scipy does: sort by height (stable), then union-find numbering
        std::stable_sort(merges.begin(), merges.end(), [](const M& a, const M& b) { return a.h < b.h; });
        std::vector<int64_t> par(static_cast<size_t>(2 * n - 1)), lab(static_cast<size_t>(n));
        std::iota(par.begin(), par.end(), 0);
        std::iota(lab.begin(), lab.end(), 0);
        std::vector<int64_t> root(static_cast<size_t>(n));
        std::iota(root.begin(), root.end(), 0);
        auto fr = [&](int64_t v) {
            while (par[static_cast<size_t>(v)] != v) v = par[static_cast<size_t>(v)];
            return v;
        };
        std::vector<int64_t> sz(static_cast<size_t>(2 * n - 1), 1);
        for (int64_t i = 0; i < n - 1; ++i) {
            int64_t a = fr(merges[static_cast<size_t>(i)].a), b = fr(merges[static_cast<size_t>(i)].b);
            if (a > b) std::swap(a, b);
            Z[4 * i] = static_cast<double>(a);
            Z[4 * i + 1] = static_cast<double>(b);
            Z[4 * i + 2] = merges[static_cast<size_t>(i)].h;
            sz[static_cast<size_t>(n + i)] = sz[static_cast<size_t>(a)] + sz[static_cast<size_t>(b)];
            Z[4 * i + 3] = static_cast<double>(sz[static_cast<size_t>(n + i)]);
            par[static_cast<size_t>(a)] = par[static_cast<size_t>(b)] = n + i;
        }
    }
    return nb::make_tuple(wrap(std::move(Za)), tie);
}

double pairwise_sum(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    for (size_t step = 1; step < v.size(); step *= 2)
        for (size_t c = 0; c + step < v.size(); c += 2 * step) v[c] += v[c + step];
    return v[0];
}

nb::object proto_heights(nb::handle Xo, nb::handle Zo, int method) {
    int64_t n, p, m, four;
    const double* X = rows_of(Xo, "X", n, p);
    const double* Z = rows_of(Zo, "Z", m, four);
    if (m != n - 1 || four != 4) throw nb::value_error("Z must be (n - 1, 4)");
    const int64_t s1[1] = {m};
    NDArray ha = new_array(DType::Float64, s1, false);
    auto* H = reinterpret_cast<double*>(ha.data);
    {
        const parallel::Unlocked unlocked(true);
        std::vector<std::vector<int64_t>> members(static_cast<size_t>(2 * n - 1));
        for (int64_t i = 0; i < n; ++i) members[static_cast<size_t>(i)] = {i};
        std::vector<std::vector<double>> sums(method == 4 ? static_cast<size_t>(2 * n - 1) : 0);
        for (int64_t q = 0; q < m; ++q) {
            const auto a = static_cast<size_t>(Z[4 * q]), b = static_cast<size_t>(Z[4 * q + 1]);
            const auto& A = members[a];
            const auto& B = members[b];
            const double na = static_cast<double>(A.size()), nb_ = static_cast<double>(B.size());
            if (method == 2) {
                std::vector<double> rows(A.size());
                const int64_t na_i = static_cast<int64_t>(A.size());
                auto body = [&](int64_t r) {
                    double s = 0;
                    const double* x = X + A[static_cast<size_t>(r)] * p;
                    for (const int64_t j : B) s += dist(x, X + j * p, p);
                    rows[static_cast<size_t>(r)] = s;
                };
                if (A.size() * B.size() * static_cast<size_t>(p) >= (1u << 15)) parallel::parallel_for(na_i, body);
                else
                    for (int64_t r = 0; r < na_i; ++r) body(r);
                H[q] = pairwise_sum(rows) / (na * nb_);
            } else {
                // centroids: the points' coordinates by a pairwise tree in member order
                auto centroid = [&](const std::vector<int64_t>& S, std::vector<double>& c) {
                    c.assign(static_cast<size_t>(p), 0.0);
                    std::vector<double> col(S.size());
                    for (int64_t d = 0; d < p; ++d) {
                        for (size_t r = 0; r < S.size(); ++r) col[r] = X[S[r] * p + d];
                        c[static_cast<size_t>(d)] = pairwise_sum(col) / static_cast<double>(S.size());
                    }
                };
                std::vector<double> ca, cb;
                centroid(A, ca);
                centroid(B, cb);
                H[q] = std::sqrt(2.0 * na * nb_ / (na + nb_)) * dist(ca.data(), cb.data(), p);
            }
            std::vector<int64_t> merged;
            merged.reserve(A.size() + B.size());
            std::merge(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(merged));
            members[static_cast<size_t>(n + q)] = std::move(merged);
            members[a].clear();
            members[a].shrink_to_fit();
            members[b].clear();
            members[b].shrink_to_fit();
        }
    }
    return wrap(std::move(ha));
}

}  // namespace

void bind_prototype(nb::module_& m) {
    m.def("_proto_assign", &proto_assign, nb::arg("X"), nb::arg("C"), nb::arg("form"));
    m.def("_proto_candidates", [] { return last_candidates; });
    m.def(
        "_proto_kmeans", &proto_kmeans, nb::arg("X"), nb::arg("C0"), nb::arg("max_iter"), nb::arg("accel"),
        nb::arg("margin")
    );
    m.def("_proto_mst", &proto_mst, nb::arg("X"), nb::arg("method"));
    m.def("_proto_nnchain", &proto_nnchain, nb::arg("X"), nb::arg("method"), nb::arg("check"));
    m.def("_proto_heights", &proto_heights, nb::arg("X"), nb::arg("Z"), nb::arg("method"));
}

}  // namespace sd::cluster
