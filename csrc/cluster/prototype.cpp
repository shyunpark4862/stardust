// Prototypes of sd.cluster's k-means kernels for milestone CL0's measurements (docs/design/cluster.md §3.3, §5),
// timed and checked by tools/cluster_probe.py (section proto). Not public: the private names _proto_* of
// _core.cluster; milestone CL2 replaces them by the area's kernels and removes this file (milestone CL1 replaced
// the hierarchy's prototypes -- the spanning trees, the chain and the heights -- by csrc/cluster/hierarchy.cpp).
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

}  // namespace

void bind_prototype(nb::module_& m) {
    m.def("_proto_assign", &proto_assign, nb::arg("X"), nb::arg("C"), nb::arg("form"));
    m.def("_proto_candidates", [] { return last_candidates; });
    m.def(
        "_proto_kmeans", &proto_kmeans, nb::arg("X"), nb::arg("C0"), nb::arg("max_iter"), nb::arg("accel"),
        nb::arg("margin")
    );
}

}  // namespace sd::cluster
