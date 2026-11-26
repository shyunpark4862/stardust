// k-medoids (docs/design/cluster.md §2.3, §3.4, §4.3; decisions CL26, CL41 and milestone CL2's in §1.1): R cluster's
// PAM (Kaufman and Rousseeuw 1990, pam(variant = "original")) and CLARA. Points are numbered by their key (CL7: the
// caller passes observations in key order; precomputed distances keep the input index), and every tie goes to the
// smaller key.
//
//   distances  observations: sd.spatial's euclidean or cityblock of a pair (the pair kernel and its finish: the bits
//              of pdist), computed when needed or held as a condensed matrix when it fits in kCache bytes (the same
//              bits either way); or the given condensed matrix.
//   BUILD      the point of smallest distance sum, then k - 1 times the point of largest gain sum_j max(D_j - d(j, i),
//              0) (D_j the distance to the nearest medoid so far), each sum point by point in key order.
//   SWAP       the change of swapping medoid m for point o is sum_j of min(d(o, j), D2_j) - D_j for the points j
//              whose nearest medoid is m and min(d(o, j) - D_j, 0) for the others (D_j, D2_j: the nearest and second
//              nearest medoid's distances). It is computed as FastPAM1 (Schubert and Rousseeuw 2019) does, in O(1) a
//              point: S_o = sum_j min(d(o, j) - D_j, 0) and for each m the correction sum over its own points of
//              min(d(o, j), D2_j) - D_j - min(d(o, j) - D_j, 0), each sum point by point in key order, then
//              change = S_o + correction. The swap of smallest (change, key of m, key of o) is made while its change
//              is negative, at most max_iter times.
//   nearest    a point's nearest medoid: the smallest (distance, key of the medoid).
//   CLARA      R's clara: `samples` samples of sample_size points, each but the first holding the best medoids so
//              far and drawn otherwise uniformly without replacement (sample s: element s 2^40 + key of the
//              point, CL41); PAM on each sample; the sample whose medoids give the smallest mean distance over all
//              points (the first on a tie) is kept. Its BUILD objective is its BUILD medoids' mean distance over all
//              points.
//
// _core.cluster:
//   kmedoids(X, D, k, method, metric, max_iter, samples, sample_size, key0, key1)
//       -> (labels (n,) medoid slot of each point, medoids (k,) keys, objective, build objective, n_swaps,
//           converged)       X: rows in key order or None; D: condensed distances or None; metric 0 euclidean,
//                             1 cityblock; method 0 PAM, 1 CLARA
#include "cluster/cluster.hpp"
#include "cluster/partition.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sd::cluster {

namespace {

constexpr int64_t kTaskPts = 256;            // candidate points of a SWAP task
constexpr int64_t kCache = int64_t{1} << 28;  // bytes of a cached condensed matrix

// The distances of a set of points numbered 0 .. n - 1.
struct Dist {
    const double* X = nullptr;  // rows (n, p), or null
    int64_t n = 0, p = 0;
    int metric = 0;             // 0 euclidean, 1 cityblock
    const double* D = nullptr;  // a condensed matrix, or null
    std::vector<double> own;    // the cached matrix

    double operator()(int64_t i, int64_t j) const {
        if (i == j) return 0.0;
        if (D) return D[cpair(n, i, j)];
        return pair(X + i * p, X + j * p);
    }
    double pair(const double* x, const double* y) const {
        double s;
        if (metric == 1) {
            spatial::pair_scalar<spatial::AbsOp, false>(x, y, nullptr, p, &s);
            return s;
        }
        spatial::pair_scalar<spatial::SqOp, false>(x, y, nullptr, p, &s);
        return spatial::euclid_finish<false>(s, x, y, nullptr, p);
    }
    void cache() {
        if (D || n < 2 || 8 * (n * (n - 1) / 2) > kCache) return;
        own.resize(static_cast<size_t>(n * (n - 1) / 2));
        auto body = [&](int64_t i) {
            for (int64_t j = i + 1; j < n; ++j) own[static_cast<size_t>(cidx(n, i, j))] = pair(X + i * p, X + j * p);
        };
        if (n * n * p >= kSplitWork) parallel::parallel_for(n, body, n * p);
        else
            for (int64_t i = 0; i < n; ++i) body(i);
        D = own.data();
    }
};

// The total of v[0 .. n) by CL48's chunks (the objective's sums)
double chunk_total(const std::vector<double>& v) {
    double t = 0.0;
    const int64_t n = static_cast<int64_t>(v.size());
    pairwise_chunks(
        chunks_of(n), 1,
        [&](int64_t c, double* out) {
            double s = 0.0;
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) s += v[static_cast<size_t>(i)];
            out[0] = s;
        },
        &t
    );
    return t;
}

struct Pam {
    std::vector<int64_t> med;  // the medoids' keys
    std::vector<int64_t> built;  // BUILD's medoids
    std::vector<int64_t> near;  // each point's nearest medoid (its slot)
    std::vector<double> d1, d2;
    double build = 0.0, total = 0.0;
    int64_t swaps = 0;
    bool converged = true;
};

// nearest and second nearest medoid of every point
void nearest(const Dist& d, Pam& P) {
    const int64_t n = d.n, k = static_cast<int64_t>(P.med.size());
    P.near.resize(static_cast<size_t>(n));
    P.d1.resize(static_cast<size_t>(n));
    P.d2.resize(static_cast<size_t>(n));
    // medoid slots in key order: the tie goes to the smaller key
    std::vector<int64_t> slots(static_cast<size_t>(k));
    std::iota(slots.begin(), slots.end(), int64_t{0});
    std::sort(slots.begin(), slots.end(), [&](int64_t a, int64_t b) {
        return P.med[static_cast<size_t>(a)] < P.med[static_cast<size_t>(b)];
    });
    auto body = [&](int64_t t) {
        for (int64_t j = t * kTaskPts; j < std::min(n, (t + 1) * kTaskPts); ++j) {
            double b1 = kInfD, b2 = kInfD;
            int64_t a = 0;
            for (int64_t s : slots) {
                const double v = d(j, P.med[static_cast<size_t>(s)]);
                if (v < b1) b2 = b1, b1 = v, a = s;
                else if (v < b2) b2 = v;
            }
            P.near[static_cast<size_t>(j)] = a, P.d1[static_cast<size_t>(j)] = b1, P.d2[static_cast<size_t>(j)] = b2;
        }
    };
    const int64_t nt = (n + kTaskPts - 1) / kTaskPts;
    if (nt > 1 && n * k >= kSplitWork / 8) parallel::parallel_for(nt, body, kTaskPts * k * 8);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    P.total = chunk_total(P.d1);
}

// the best (value, key) over tasks: value then the smaller key
struct Best {
    double v = kInfD;
    int64_t a = -1, b = -1;
    bool less(double v2, int64_t a2, int64_t b2) const {
        if (a < 0) return true;
        if (v2 != v) return v2 < v;
        if (a2 != a) return a2 < a;
        return b2 < b;
    }
};

Pam pam(const Dist& d, int64_t k, int64_t max_iter) {
    const int64_t n = d.n;
    Pam P;
    // BUILD: the smallest distance sum first
    std::vector<double> sums(static_cast<size_t>(n));
    const int64_t nt = (n + kTaskPts - 1) / kTaskPts;
    auto par = [&](auto&& body) {
        if (nt > 1 && n * n >= kSplitWork / 8) parallel::parallel_for(nt, body, kTaskPts * n * 8);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    };
    par([&](int64_t t) {
        for (int64_t i = t * kTaskPts; i < std::min(n, (t + 1) * kTaskPts); ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += d(i, j);
            sums[static_cast<size_t>(i)] = s;
        }
    });
    int64_t first = 0;
    for (int64_t i = 1; i < n; ++i)
        if (sums[static_cast<size_t>(i)] < sums[static_cast<size_t>(first)]) first = i;
    P.med.push_back(first);
    std::vector<double> Dm(static_cast<size_t>(n));
    std::vector<char> is_med(static_cast<size_t>(n), 0);
    is_med[static_cast<size_t>(first)] = 1;
    for (int64_t j = 0; j < n; ++j) Dm[static_cast<size_t>(j)] = d(j, first);
    std::vector<double> gain(static_cast<size_t>(n));
    while (static_cast<int64_t>(P.med.size()) < k) {
        par([&](int64_t t) {
            for (int64_t i = t * kTaskPts; i < std::min(n, (t + 1) * kTaskPts); ++i) {
                if (is_med[static_cast<size_t>(i)]) continue;
                double g = 0.0;
                for (int64_t j = 0; j < n; ++j) {
                    const double z = Dm[static_cast<size_t>(j)] - d(j, i);
                    if (z > 0.0) g += z;
                }
                gain[static_cast<size_t>(i)] = g;
            }
        });
        int64_t best = -1;
        for (int64_t i = 0; i < n; ++i)
            if (!is_med[static_cast<size_t>(i)] &&
                (best < 0 || gain[static_cast<size_t>(i)] > gain[static_cast<size_t>(best)]))
                best = i;
        P.med.push_back(best);
        is_med[static_cast<size_t>(best)] = 1;
        for (int64_t j = 0; j < n; ++j) Dm[static_cast<size_t>(j)] = std::min(Dm[static_cast<size_t>(j)], d(j, best));
    }
    nearest(d, P);
    P.build = P.total;
    P.built = P.med;
    // SWAP
    std::vector<Best> bests(static_cast<size_t>(nt));
    P.converged = false;
    for (;;) {
        par([&](int64_t t) {
            Best B;
            std::vector<double> corr(static_cast<size_t>(k));
            for (int64_t o = t * kTaskPts; o < std::min(n, (t + 1) * kTaskPts); ++o) {
                if (is_med[static_cast<size_t>(o)]) continue;
                std::fill(corr.begin(), corr.end(), 0.0);
                double S = 0.0;
                for (int64_t j = 0; j < n; ++j) {
                    const double doj = d(o, j), dj = P.d1[static_cast<size_t>(j)];
                    const double shared = std::min(doj - dj, 0.0);
                    S += shared;
                    const double own = std::min(doj, P.d2[static_cast<size_t>(j)]) - dj;
                    corr[static_cast<size_t>(P.near[static_cast<size_t>(j)])] += own - shared;
                }
                for (int64_t m = 0; m < k; ++m) {
                    const double v = S + corr[static_cast<size_t>(m)];
                    if (B.less(v, P.med[static_cast<size_t>(m)], o))
                        B.v = v, B.a = P.med[static_cast<size_t>(m)], B.b = o;
                }
            }
            bests[static_cast<size_t>(t)] = B;
        });
        Best B;
        for (const Best& b : bests)
            if (b.a >= 0 && B.less(b.v, b.a, b.b)) B = b;
        if (B.a < 0 || !(B.v < 0.0)) {
            P.converged = true;
            break;
        }
        if (P.swaps >= max_iter) break;
        for (auto& mm : P.med)
            if (mm == B.a) mm = B.b;
        is_med[static_cast<size_t>(B.a)] = 0;
        is_med[static_cast<size_t>(B.b)] = 1;
        ++P.swaps;
        nearest(d, P);
    }
    return P;
}

// The mean distance of all points to their nearest of the medoids meds (keys), with their slots
double assign_all(const Dist& d, const std::vector<int64_t>& meds, std::vector<int64_t>* lab) {
    Pam P;
    P.med = meds;
    nearest(d, P);
    if (lab) *lab = P.near;
    return P.total / static_cast<double>(d.n);
}

struct Out {
    std::vector<int64_t> lab, med;
    double objective = 0.0, build = 0.0;
    int64_t swaps = 0;
    bool converged = true;
};

Out run(Dist& d, int64_t k, int method, int64_t max_iter, int64_t samples, int64_t sample_size, random::Key key) {
    Out out;
    const int64_t n = d.n;
    if (method == 0 || sample_size >= n) {
        d.cache();
        Pam P = pam(d, k, max_iter);
        out.med = P.med, out.lab = P.near, out.swaps = P.swaps, out.converged = P.converged;
        out.objective = P.total / static_cast<double>(n);
        out.build = P.build / static_cast<double>(n);
        return out;
    }
    std::vector<int64_t> best_meds, built;
    double best = kInfD;
    for (int64_t s = 0; s < samples; ++s) {
        std::vector<char> in(static_cast<size_t>(n), 0);
        std::vector<int64_t> pts = best_meds;
        for (int64_t m : best_meds) in[static_cast<size_t>(m)] = 1;
        // the rest drawn uniformly without replacement (Efraimidis-Spirakis keys of unit weights: the largest u)
        std::vector<double> u(static_cast<size_t>(n));
        random::Stream st(key);
        for (int64_t i = 0; i < n; ++i)
            u[static_cast<size_t>(i)] =
                in[static_cast<size_t>(i)]
                    ? -1.0
                    : uniform(st, element(static_cast<std::uint64_t>(s), static_cast<std::uint64_t>(i)));
        std::vector<int64_t> idx(static_cast<size_t>(n));
        std::iota(idx.begin(), idx.end(), int64_t{0});
        const auto need = static_cast<std::ptrdiff_t>(sample_size - static_cast<int64_t>(pts.size()));
        std::partial_sort(idx.begin(), idx.begin() + need, idx.end(), [&](int64_t a, int64_t b) {
            const double x = u[static_cast<size_t>(a)], y = u[static_cast<size_t>(b)];
            return x != y ? x > y : a < b;
        });
        pts.insert(pts.end(), idx.begin(), idx.begin() + need);
        std::sort(pts.begin(), pts.end());  // the sample in key order
        const int64_t m = static_cast<int64_t>(pts.size());
        Dist ds;
        ds.n = m, ds.p = d.p, ds.metric = d.metric;
        std::vector<double> rows, sub;
        if (d.X) {
            rows.resize(static_cast<size_t>(m * d.p));
            for (int64_t q = 0; q < m; ++q)
                std::copy(
                    d.X + pts[static_cast<size_t>(q)] * d.p, d.X + (pts[static_cast<size_t>(q)] + 1) * d.p,
                    rows.begin() + q * d.p
                );
            ds.X = rows.data();
            ds.cache();
        } else {
            sub.resize(static_cast<size_t>(m * (m - 1) / 2));
            for (int64_t a = 0; a < m; ++a)
                for (int64_t b = a + 1; b < m; ++b)
                    sub[static_cast<size_t>(cidx(m, a, b))] =
                        d(pts[static_cast<size_t>(a)], pts[static_cast<size_t>(b)]);
            ds.D = sub.data();
        }
        Pam P = pam(ds, k, max_iter);
        std::vector<int64_t> meds(static_cast<size_t>(k));
        for (int64_t j = 0; j < k; ++j)
            meds[static_cast<size_t>(j)] = pts[static_cast<size_t>(P.med[static_cast<size_t>(j)])];
        const double obj = assign_all(d, meds, nullptr);
        if (obj < best) {
            best = obj;
            best_meds = meds;
            out.med = meds;
            out.swaps = P.swaps, out.converged = P.converged;
            built.resize(static_cast<size_t>(k));
            for (int64_t j = 0; j < k; ++j)
                built[static_cast<size_t>(j)] = pts[static_cast<size_t>(P.built[static_cast<size_t>(j)])];
        }
    }
    out.objective = assign_all(d, out.med, &out.lab);
    out.build = assign_all(d, built, nullptr);  // the kept sample's BUILD medoids over all points
    return out;
}

nb::object kmedoids_py(
    nb::handle Xo,
    nb::handle Do,
    int64_t k,
    int method,
    int metric,
    int64_t max_iter,
    int64_t samples,
    int64_t sample_size,
    int64_t key0,
    int64_t key1
) {
    Dist d;
    int64_t n = 0;
    if (!Xo.is_none()) {
        d.X = matrix_arg(Xo, "X", d.n, d.p);
        n = d.n;
    } else {
        int64_t m;
        d.D = vector_arg(Do, "D", m);
        n = (1 + static_cast<int64_t>(std::llround(std::sqrt(1.0 + 8.0 * static_cast<double>(m))))) / 2;
        if (n * (n - 1) / 2 != m) throw nb::value_error("D: condensed");
        d.n = n;
    }
    d.metric = metric;
    if (k < 1 || k > n) throw nb::value_error("k");
    Out out;
    {
        const parallel::Unlocked unlocked(n * n >= kSplitWork);
        out = run(d, k, method, max_iter, samples, sample_size, key_of(key0, key1));
    }
    return nb::make_tuple(
        wrap(int_array(out.lab, n)), wrap(int_array(out.med, k)), out.objective, out.build, out.swaps, out.converged
    );
}

}  // namespace

void bind_kmedoids(nb::module_& m) {
    m.def(
        "kmedoids", &kmedoids_py, nb::arg("X").none(), nb::arg("D").none(), nb::arg("k"), nb::arg("method"),
        nb::arg("metric"), nb::arg("max_iter"), nb::arg("samples"), nb::arg("sample_size"), nb::arg("key0"),
        nb::arg("key1")
    );
}

}  // namespace sd::cluster
