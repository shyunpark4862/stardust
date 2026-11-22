// pdist, cdist, the distance functions and squareform of sd.spatial (docs/design/spatial.md §2.2, §3.1,
// §3.5). The Python layer (src/stardust/spatial/_distance.py) checks every argument; these functions only
// check the layout they are handed.
//
//   _cdist(XA, XB, metric, p, w, aux)   (mA, n), (mB, n) -> (mA, mB) float64
//   _pdist(X, metric, p, w, aux)        (m, n) -> (m (m - 1) / 2,) float64, pairs i < j row by row
//   _paired(U, V, metric, p, w, aux)    (P, n), (P, n) -> (P,) float64: the distance of U[k] and V[k]
//       metric: one of the fourteen names (minkowski's p = 1, 2, inf already sent to cityblock,
//       euclidean, chebyshev; correlation(centered=False) to cosine). p: minkowski's. w: None or the
//       weights (n,). aux: seuclidean's V (n,), mahalanobis' VI (n, n), else None. The data are
//       float64, except hamming and jaccard: bool, int64 or float64 (both sides the same dtype).
//   _squareform_tomatrix(v, d)          (d (d - 1) / 2,) -> (d, d), the dtype kept
//   _squareform_tovector(X, checks)     (d, d) -> (code, vector or None): code 0, or 1 not symmetric,
//                                       2 not symmetric because of a NaN, 3 a nonzero diagonal
//
// Every pair goes through one kernel of csrc/spatial/metric.hpp: pdist and cdist through the tile kernel
// (the remainder rows and columns padded, their results dropped), the distance functions through the
// one-pair kernel, which gives the same bits. Work is cut into tasks by size alone (row blocks and panel
// chunks; pdist's row groups balanced by their pairs), each pair written by one task, so the result does
// not depend on the thread count. From kParallelWork pair-terms a call splits and runs without the GIL.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/metric.hpp"
#include "spatial/spatial.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

using std::int64_t;

// The array area's rule (docs/design/array.md §10.4): from 2^15 units of work a call splits and runs
// without the GIL, whatever the thread count.
constexpr int64_t kParallelWork = 1 << 15;
// Pair-terms per task (about 0.25 ms of the tile kernel).
constexpr int64_t kTaskWork = 1 << 20;
// Doubles of packed panels a cdist task reads again for each row group (256 KB).
constexpr int64_t kPanelChunk = 1 << 15;

template <class F>
void for_tasks(int64_t ntasks, int64_t work, F&& fn) {
    if (ntasks <= 0) return;
    const parallel::Unlocked unlocked(work >= kParallelWork);
    if (work < kParallelWork || ntasks == 1) {
        for (int64_t t = 0; t < ntasks; ++t) fn(t);
    } else {
        parallel::parallel_for(ntasks, fn);
    }
}

// ---- arguments -------------------------------------------------------------------------------

// A 1-D or 2-D array as rows x cols contiguous elements (a strided one copied).
struct Mat {
    const std::byte* data = nullptr;
    int64_t rows = 0, cols = 0;
    DType dtype = DType::Float64;
    std::vector<std::byte> own;

    Mat() = default;
    Mat(Mat&&) = default;
    Mat& operator=(Mat&&) = default;
    Mat(const Mat&) = delete;
    Mat& operator=(const Mat&) = delete;

    template <class T>
    const T* row(int64_t i) const {
        return reinterpret_cast<const T*>(data) + i * cols;
    }
    const double* f64() const { return reinterpret_cast<const double*>(data); }
};

Mat mat_arg(nb::handle o, const char* func, int ndim) {
    if (!is_ndarray(o)) throw nb::type_error((std::string(func) + "(): expected an sd.ndarray").c_str());
    const NDArray& a = as_ndarray(o);
    if (a.ndim != ndim || (a.dtype != DType::Float64 && a.dtype != DType::Int64 && a.dtype != DType::Bool))
        throw nb::type_error((std::string(func) + "(): wrong array layout").c_str());
    Mat m;
    m.dtype = a.dtype;
    m.rows = ndim == 2 ? a.shape[0] : 1;
    m.cols = ndim == 2 ? a.shape[1] : a.shape[0];
    const int64_t isz = itemsize(a.dtype);
    if (a.c_contiguous || m.rows * m.cols == 0) {
        m.data = a.data;
        return m;
    }
    m.own.resize(static_cast<size_t>(m.rows * m.cols * isz));
    const int64_t s0 = ndim == 2 ? a.strides[0] : 0, s1 = a.strides[ndim - 1];
    for (int64_t i = 0; i < m.rows; ++i)
        for (int64_t j = 0; j < m.cols; ++j)
            std::memcpy(m.own.data() + (i * m.cols + j) * isz, a.data + i * s0 + j * s1, static_cast<size_t>(isz));
    m.data = m.own.data();
    return m;
}

enum class Metric {
    Euclidean,
    SqEuclidean,
    Cityblock,
    Chebyshev,
    Minkowski,
    SEuclidean,
    Mahalanobis,
    Cosine,
    Correlation,
    Canberra,
    BrayCurtis,
    Hamming,
    Jaccard,
    Haversine
};

Metric parse_metric(const std::string& s, const char* func) {
    static const std::pair<const char*, Metric> names[] = {
        {"euclidean", Metric::Euclidean},     {"sqeuclidean", Metric::SqEuclidean},
        {"cityblock", Metric::Cityblock},     {"chebyshev", Metric::Chebyshev},
        {"minkowski", Metric::Minkowski},     {"seuclidean", Metric::SEuclidean},
        {"mahalanobis", Metric::Mahalanobis}, {"cosine", Metric::Cosine},
        {"correlation", Metric::Correlation}, {"canberra", Metric::Canberra},
        {"braycurtis", Metric::BrayCurtis},   {"hamming", Metric::Hamming},
        {"jaccard", Metric::Jaccard},         {"haversine", Metric::Haversine},
    };
    for (const auto& [name, m] : names)
        if (s == name) return m;
    throw nb::value_error((std::string(func) + "(): unknown metric " + s).c_str());
}

// What a call computes: the metric, its parameters, the weights (seuclidean: 1 / V).
struct Plan {
    Metric metric = Metric::Euclidean;
    int64_t n = 0;
    bool weighted = false;
    std::vector<double> w;  // weights, or 1 / V
    const double* vi = nullptr;
    double denom = 0.0;  // hamming: n or the sum of the weights
    Root root{2.0};
    DType dtype = DType::Float64;

    const double* wp() const { return weighted ? w.data() : nullptr; }
};

Plan make_plan(
    const std::string& metric,
    double p,
    nb::handle w,
    nb::handle aux,
    int64_t n,
    DType dtype,
    Mat& vi_store,
    const char* func
) {
    Plan plan;
    plan.metric = parse_metric(metric, func);
    plan.n = n;
    plan.dtype = dtype;
    if (!w.is_none()) {
        const Mat wm = mat_arg(w, func, 1);
        if (wm.cols != n || wm.dtype != DType::Float64)
            throw nb::type_error((std::string(func) + "(): wrong weights").c_str());
        plan.weighted = true;
        plan.w.assign(wm.f64(), wm.f64() + n);
    }
    if (plan.metric == Metric::SEuclidean) {
        const Mat v = mat_arg(aux, func, 1);
        if (v.cols != n || v.dtype != DType::Float64) throw nb::type_error((std::string(func) + "(): wrong V").c_str());
        plan.weighted = true;
        plan.w.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) plan.w[static_cast<size_t>(i)] = 1.0 / v.f64()[i];
    }
    if (plan.metric == Metric::Mahalanobis) {
        vi_store = mat_arg(aux, func, 2);
        if (vi_store.rows != n || vi_store.cols != n || vi_store.dtype != DType::Float64)
            throw nb::type_error((std::string(func) + "(): wrong VI").c_str());
        plan.vi = vi_store.f64();
    }
    if (plan.metric == Metric::Minkowski) plan.root = Root(p);
    if (plan.metric == Metric::Hamming)
        plan.denom = plan.weighted ? weight_sum(plan.w.data(), n) : static_cast<double>(n);
    const bool counting = plan.metric == Metric::Hamming || plan.metric == Metric::Jaccard;
    if (!counting && dtype != DType::Float64) throw nb::type_error((std::string(func) + "(): wrong dtype").c_str());
    if (plan.metric == Metric::Haversine && n != 2)
        throw nb::type_error((std::string(func) + "(): not 2 columns").c_str());
    return plan;
}

// ---- prepared rows ---------------------------------------------------------------------------

// rows x n doubles, prepared row by row
template <class F>
void each_row(int64_t rows, int64_t n, F&& f) {
    const int64_t per = std::max<int64_t>(1, kTaskWork / std::max<int64_t>(1, 8 * n));
    for_tasks((rows + per - 1) / per, rows * std::max<int64_t>(n, 1) * 8, [&](int64_t t) {
        for (int64_t i = t * per, e = std::min(rows, (t + 1) * per); i < e; ++i) f(i);
    });
}

// cosine and correlation: the rows normalized (centered first for correlation)
std::vector<double> prepared(const Mat& x, const Plan& plan) {
    const int64_t n = plan.n;
    std::vector<double> out(static_cast<size_t>(x.rows * n));
    const bool center = plan.metric == Metric::Correlation;
    const double sumw = plan.weighted ? weight_sum(plan.w.data(), n) : 0.0;
    each_row(x.rows, n, [&](int64_t i) {
        double* o = out.data() + i * n;
        if (plan.weighted) prepare_row<true>(x.row<double>(i), plan.w.data(), n, center, sumw, o);
        else prepare_row<false>(x.row<double>(i), nullptr, n, center, sumw, o);
    });
    return out;
}

std::vector<double> cosines(const Mat& x) {  // haversine: cos(latitude) per row
    std::vector<double> c(static_cast<size_t>(x.rows));
    for (int64_t i = 0; i < x.rows; ++i) c[static_cast<size_t>(i)] = std::cos(x.row<double>(i)[0]);
    return c;
}

std::vector<std::uint64_t> packed_bits(const Mat& x, int64_t n) {
    const int64_t words = (n + 63) / 64;
    std::vector<std::uint64_t> out(static_cast<size_t>(x.rows * words));
    each_row(x.rows, words, [&](int64_t i) { pack_bits(x.row<std::uint8_t>(i), n, out.data() + i * words); });
    return out;
}

inline int64_t popcount_xor(const std::uint64_t* a, const std::uint64_t* b, int64_t words) {
    int64_t c = 0;
    for (int64_t k = 0; k < words; ++k) c += std::popcount(a[k] ^ b[k]);
    return c;
}
inline int64_t popcount_or(const std::uint64_t* a, const std::uint64_t* b, int64_t words) {
    int64_t c = 0;
    for (int64_t k = 0; k < words; ++k) c += std::popcount(a[k] | b[k]);
    return c;
}

// ---- the finishers (per pair, from the K sums; x and y the pair's rows) ----------------------
//
// operator() finishes one pair; vec() two pairs at once from NEON registers, the same IEEE operations
// (sqrt, div, min are correctly rounded either way), and marks in `care` the lanes it cannot finish (a sum
// that needs the careful path), which the driver then finishes one by one.

inline uint64x2_t no_care() { return vdupq_n_u64(0); }
inline uint64x2_t care_of(float64x2_t s) {  // !(kTiny <= s <= kHuge) and not NaN
    const uint64x2_t in = vandq_u64(vcgeq_f64(s, vdupq_n_f64(kTiny)), vcleq_f64(s, vdupq_n_f64(kHuge)));
    return vbicq_u64(vceqq_f64(s, s), in);
}

template <bool W>
struct EuclidFin {
    const double* w;
    int64_t n;
    double operator()(const double* s, const double* x, const double* y) const {
        return euclid_finish<W>(s[0], x, y, w, n);
    }
    float64x2_t vec(const float64x2_t* s, uint64x2_t& care) const {
        care = care_of(s[0]);
        return vsqrtq_f64(s[0]);
    }
};
template <bool W>
struct SqFin {
    const double* w;
    int64_t n;
    double operator()(const double* s, const double* x, const double* y) const {
        return sq_finish<W>(s[0], x, y, w, n);
    }
    float64x2_t vec(const float64x2_t* s, uint64x2_t& care) const {
        care = vcltq_f64(s[0], vdupq_n_f64(kTiny));
        return s[0];
    }
};
struct IdFin {
    double operator()(const double* s, const double*, const double*) const { return s[0]; }
    float64x2_t vec(const float64x2_t* s, uint64x2_t& care) const {
        care = no_care();
        return s[0];
    }
};
struct BrayFin {
    double operator()(const double* s, const double*, const double*) const { return s[0] / s[1]; }
    float64x2_t vec(const float64x2_t* s, uint64x2_t& care) const {
        care = no_care();
        return vdivq_f64(s[0], s[1]);
    }
};
struct HammingFin {
    double d;
    double operator()(const double* s, const double*, const double*) const { return s[0] / d; }
    float64x2_t vec(const float64x2_t* s, uint64x2_t& care) const {
        care = no_care();
        return vdivq_f64(s[0], vdupq_n_f64(d));
    }
};
struct JaccardFin {
    double operator()(const double* s, const double*, const double*) const { return s[1] == 0 ? 0.0 : s[0] / s[1]; }
    float64x2_t vec(const float64x2_t* s, uint64x2_t& care) const {
        care = no_care();
        return vbslq_f64(vceqzq_f64(s[1]), f2_zero(), vdivq_f64(s[0], s[1]));
    }
};
template <bool W>
struct CosFin {  // on the prepared rows
    const double* w;
    int64_t n;
    double operator()(const double* s, const double* x, const double* y) const {
        if (n == 0) return std::numeric_limits<double>::quiet_NaN();
        const double d = sq_finish<W>(s[0], x, y, w, n) * 0.5;
        return d > 2.0 ? 2.0 : d;
    }
    float64x2_t vec(const float64x2_t* s, uint64x2_t& care) const {
        care = n == 0 ? vdupq_n_u64(~0ull) : vcltq_f64(s[0], vdupq_n_f64(kTiny));
        const float64x2_t d = vmulq_f64(s[0], vdupq_n_f64(0.5));
        return vbslq_f64(vcgtq_f64(d, vdupq_n_f64(2.0)), vdupq_n_f64(2.0), d);
    }
};

template <bool W>
EuclidFin<W> euclid_fin(const Plan& p) {
    return {p.wp(), p.n};
}
template <bool W>
SqFin<W> sq_fin(const Plan& p) {
    return {p.wp(), p.n};
}
inline IdFin id_fin() { return {}; }
inline BrayFin bray_fin() { return {}; }
inline HammingFin hamming_fin(const Plan& p) { return {p.denom}; }
inline JaccardFin jaccard_fin() { return {}; }
template <bool W>
CosFin<W> cos_fin(const Plan& p) {
    return {p.wp(), p.n};
}

// The results of tile row r (of R) for columns c0 .. cols into o[0 .. cols - c0): two at a time, the lanes
// that need care then one by one. x: the row's data; y(c): column c's data.
template <int K, int R, class Fin, class Y>
inline void finish_row(const Fin& fin, const double* buf, int r, const double* x, Y y, int c0, int cols, double* o) {
    uint64x2_t care[kPanel / 2];
    uint64x2_t any = vdupq_n_u64(0);
    for (int c = 0; c < kPanel; c += 2) {
        float64x2_t sv[K];
        for (int k = 0; k < K; ++k) sv[k] = vld1q_f64(buf + (k * R + r) * kPanel + c);
        const float64x2_t v = fin.vec(sv, care[c / 2]);
        any = vorrq_u64(any, care[c / 2]);
        if (c >= c0 && c + 1 < cols) {
            vst1q_f64(o + (c - c0), v);
        } else {
            if (c >= c0 && c < cols) o[c - c0] = vgetq_lane_f64(v, 0);
            if (c + 1 >= c0 && c + 1 < cols) o[c + 1 - c0] = vgetq_lane_f64(v, 1);
        }
    }
    if (vmaxvq_u32(vreinterpretq_u32_u64(any)) == 0) return;
    for (int c = c0; c < cols; ++c) {
        if ((c % 2 == 0 ? vgetq_lane_u64(care[c / 2], 0) : vgetq_lane_u64(care[c / 2], 1)) == 0) continue;
        double s1[K];
        for (int k = 0; k < K; ++k) s1[k] = buf[(k * R + r) * kPanel + c];
        o[c - c0] = fin(s1, x, y(c));
    }
}

// ---- the drivers -----------------------------------------------------------------------------

std::vector<double> pack_all(const double* y, int64_t rows, int64_t n) {
    const int64_t panels = (rows + kPanel - 1) / kPanel;
    std::vector<double> packed(static_cast<size_t>(panels * kPanel * n));
    if (n == 0) return packed;
    each_row(panels, n * kPanel, [&](int64_t p) {
        pack_panel(
            y + p * kPanel * n, n, std::min<int64_t>(kPanel, rows - p * kPanel), n, packed.data() + p * kPanel * n
        );
    });
    return packed;
}

// cdist through the tile kernel: a (ma, n), b (mb, n), out (ma, mb)
template <class Op, int R, bool W, class Fin>
void cdist_tile(
    const double* a, int64_t ma, const double* b, int64_t mb, int64_t n, const double* w, double* out, Fin fin
) {
    if (ma == 0 || mb == 0) return;
    const int64_t panels = (mb + kPanel - 1) / kPanel;
    const std::vector<double> packed = pack_all(b, mb, n);
    const int64_t nn = std::max<int64_t>(n, 1);
    const int64_t pchunk = std::max<int64_t>(1, kPanelChunk / (kPanel * nn));
    const int64_t rblock = std::max<int64_t>(R, (kTaskWork / (pchunk * kPanel * nn)) / R * R);
    const int64_t nrb = (ma + rblock - 1) / rblock, npc = (panels + pchunk - 1) / pchunk;
    for_tasks(nrb * npc, ma * mb * nn, [&](int64_t t) {
        const int64_t rb = t / npc, pc = t % npc;
        double buf[Op::K * R * kPanel];
        const double* xs[R];
        const int64_t i1 = std::min(ma, (rb + 1) * rblock), p1 = std::min(panels, (pc + 1) * pchunk);
        for (int64_t i0 = rb * rblock; i0 < i1; i0 += R) {
            for (int r = 0; r < R; ++r) xs[r] = a + std::min(i0 + r, ma - 1) * n;
            for (int64_t p = pc * pchunk; p < p1; ++p) {
                tile<Op, R, W>(xs, packed.data() + p * kPanel * n, w, n, buf);
                const int cols = static_cast<int>(std::min<int64_t>(kPanel, mb - p * kPanel));
                const double* yb = b + p * kPanel * n;
                for (int r = 0; r < R && i0 + r < i1; ++r) {
                    const int64_t i = i0 + r;
                    finish_row<Op::K, R>(
                        fin, buf, r, a + i * n, [&](int c) { return yb + c * n; }, 0, cols, out + i * mb + p * kPanel
                    );
                }
            }
        }
    });
}

// Task bounds over pdist's rows (or row groups of `group` rows): about kTaskWork pair-terms a task.
std::vector<int64_t> pdist_bounds(int64_t m, int64_t group, int64_t cost) {
    const int64_t groups = (m + group - 1) / group;
    std::vector<int64_t> bounds{0};
    int64_t acc = 0;
    for (int64_t g = 0; g < groups; ++g) {
        acc += std::max<int64_t>(0, m - g * group) * group * cost;
        if (acc >= kTaskWork && g + 1 < groups) {
            bounds.push_back(g + 1);
            acc = 0;
        }
    }
    bounds.push_back(groups);
    return bounds;
}

inline int64_t cond_index(int64_t i, int64_t j, int64_t m) { return i * m - i * (i + 1) / 2 + (j - i - 1); }

template <class Op, int R, bool W, class Fin>
void pdist_tile(const double* x, int64_t m, int64_t n, const double* w, double* out, Fin fin) {
    if (m < 2) return;
    const int64_t panels = (m + kPanel - 1) / kPanel;
    const std::vector<double> packed = pack_all(x, m, n);
    const int64_t nn = std::max<int64_t>(n, 1);
    const std::vector<int64_t> bounds = pdist_bounds(m, R, nn);
    for_tasks(static_cast<int64_t>(bounds.size()) - 1, m * (m - 1) / 2 * nn, [&](int64_t t) {
        double buf[Op::K * R * kPanel];
        const double* xs[R];
        for (int64_t g = bounds[t]; g < bounds[t + 1]; ++g) {
            const int64_t i0 = g * R;
            for (int r = 0; r < R; ++r) xs[r] = x + std::min(i0 + r, m - 1) * n;
            for (int64_t p = (i0 + 1) / kPanel; p < panels; ++p) {
                tile<Op, R, W>(xs, packed.data() + p * kPanel * n, w, n, buf);
                const int cols = static_cast<int>(std::min<int64_t>(kPanel, m - p * kPanel));
                const double* yb = x + p * kPanel * n;
                for (int r = 0; r < R && i0 + r < m; ++r) {
                    const int64_t i = i0 + r;
                    const int64_t c0 = std::max<int64_t>(0, i + 1 - p * kPanel);  // the first j > i
                    if (c0 >= cols) continue;
                    finish_row<Op::K, R>(
                        fin, buf, r, x + i * n, [&](int c) { return yb + c * n; }, static_cast<int>(c0), cols,
                        out + cond_index(i, p * kPanel + c0, m)
                    );
                }
            }
        }
    });
}

// The scalar drivers: f(i, j, scratch) -> the distance of row i of the first side and row j of the
// second; scratch holds n doubles per task. cost: a pair's work in pair-terms.
template <class F>
void cdist_scalar(int64_t ma, int64_t mb, int64_t n, int64_t cost, double* out, F f) {
    if (ma == 0 || mb == 0) return;
    const int64_t per = std::max<int64_t>(1, kTaskWork / std::max<int64_t>(1, mb * cost));
    for_tasks((ma + per - 1) / per, ma * mb * cost, [&](int64_t t) {
        std::vector<double> scratch(static_cast<size_t>(n));
        for (int64_t i = t * per, e = std::min(ma, (t + 1) * per); i < e; ++i)
            for (int64_t j = 0; j < mb; ++j) out[i * mb + j] = f(i, j, scratch.data());
    });
}

template <class F>
void pdist_scalar(int64_t m, int64_t n, int64_t cost, double* out, F f) {
    if (m < 2) return;
    const std::vector<int64_t> bounds = pdist_bounds(m, 1, cost);
    for_tasks(static_cast<int64_t>(bounds.size()) - 1, m * (m - 1) / 2 * cost, [&](int64_t t) {
        std::vector<double> scratch(static_cast<size_t>(n));
        for (int64_t i = bounds[t]; i < bounds[t + 1]; ++i)
            for (int64_t j = i + 1; j < m; ++j) out[cond_index(i, j, m)] = f(i, j, scratch.data());
    });
}

// ---- one call: cdist and pdist share the dispatch ---------------------------------------------

// The cost of a pair in pair-terms, for the task sizes and the parallel threshold.
int64_t pair_cost(const Plan& plan) {
    const int64_t n = std::max<int64_t>(plan.n, 1);
    switch (plan.metric) {
        case Metric::Minkowski: return 16 * n;
        case Metric::Mahalanobis: return n * n + n;
        case Metric::Haversine: return 64;
        default: return n;
    }
}

// Side: what a driver needs of one side (cdist: two, pdist: one). Calls the right driver with the right
// kernel. `two`: cdist (a, b) or pdist (a only).
struct Call {
    const Mat* a;
    const Mat* b;  // null: pdist
    double* out;
    const Plan* plan;

    template <class Op, int R, bool W, class Fin>
    void tile_on(const double* xa, const double* xb, Fin fin) const {
        const int64_t n = plan->n;
        if (b) cdist_tile<Op, R, W>(xa, a->rows, xb, b->rows, n, plan->wp(), out, fin);
        else pdist_tile<Op, R, W>(xa, a->rows, n, plan->wp(), out, fin);
    }
    template <class Op, int R, class Fin>
    void tile(Fin fin) const {
        const double* xa = a->f64();
        const double* xb = b ? b->f64() : nullptr;
        if (plan->weighted) tile_on<Op, R, true>(xa, xb, fin);
        else tile_on<Op, R, false>(xa, xb, fin);
    }
    template <class F>
    void scalar(F f) const {
        const int64_t cost = pair_cost(*plan);
        if (b) cdist_scalar(a->rows, b->rows, plan->n, cost, out, f);
        else pdist_scalar(a->rows, plan->n, cost, out, f);
    }
    const Mat& second() const { return b ? *b : *a; }
};

template <class Op, bool W, class T>
double count_pair(const T* x, const T* y, const Plan& p) {
    double s[Op::K];
    pair_scalar<Op, W, T>(x, y, p.w.data(), p.n, s);
    if constexpr (std::is_same_v<Op, HammingOp>) return s[0] / p.denom;
    else return s[1] == 0 ? 0.0 : s[0] / s[1];
}

// hamming and jaccard on int64 data, and on bool data with weights: the one-pair kernel per pair
template <class Op, class T>
void count_scalar(const Call& c) {
    const Plan& p = *c.plan;
    const Mat &A = *c.a, &B = c.second();
    if (p.weighted)
        c.scalar([&](int64_t i, int64_t j, double*) { return count_pair<Op, true, T>(A.row<T>(i), B.row<T>(j), p); });
    else c.scalar([&](int64_t i, int64_t j, double*) { return count_pair<Op, false, T>(A.row<T>(i), B.row<T>(j), p); });
}

void run(const Call& c) {
    const Plan& p = *c.plan;
    const int64_t n = p.n;
    const bool w = p.weighted;
    switch (p.metric) {
        case Metric::Euclidean:
        case Metric::SEuclidean:
            if (w) c.tile<WSqOp, 4>(euclid_fin<true>(p));
            else c.tile<SqOp, 4>(euclid_fin<false>(p));
            return;
        case Metric::SqEuclidean:
            if (w) c.tile<WSqOp, 4>(sq_fin<true>(p));
            else c.tile<SqOp, 4>(sq_fin<false>(p));
            return;
        case Metric::Cityblock:
            if (w) c.tile<WAbsOp, 4>(id_fin());
            else c.tile<AbsOp, 4>(id_fin());
            return;
        case Metric::Chebyshev:
            if (w) c.tile<WMaxOp, 4>(id_fin());
            else c.tile<MaxOp, 4>(id_fin());
            return;
        case Metric::Canberra:
            if (w) c.tile<WCanberraOp, 4>(id_fin());
            else c.tile<CanberraOp, 4>(id_fin());
            return;
        case Metric::BrayCurtis:
            if (w) c.tile<WBrayOp, 2>(bray_fin());
            else c.tile<BrayOp, 2>(bray_fin());
            return;
        case Metric::Hamming:
        case Metric::Jaccard: {
            const bool ham = p.metric == Metric::Hamming;
            if (p.dtype == DType::Float64) {
                if (ham) c.tile<HammingOp, 4>(hamming_fin(p));
                else c.tile<JaccardOp, 2>(jaccard_fin());
            } else if (p.dtype == DType::Int64) {
                if (ham) count_scalar<HammingOp, int64_t>(c);
                else count_scalar<JaccardOp, int64_t>(c);
            } else if (w) {
                if (ham) count_scalar<HammingOp, std::uint8_t>(c);
                else count_scalar<JaccardOp, std::uint8_t>(c);
            } else {  // bool, unweighted: the rows packed into words, popcounts (SPT23)
                const int64_t words = (n + 63) / 64;
                const std::vector<std::uint64_t> pa = packed_bits(*c.a, n);
                const std::vector<std::uint64_t> pb = c.b ? packed_bits(*c.b, n) : std::vector<std::uint64_t>();
                const std::uint64_t* qa = pa.data();
                const std::uint64_t* qb = c.b ? pb.data() : pa.data();
                const double dn = static_cast<double>(n);
                if (ham)
                    c.scalar([&](int64_t i, int64_t j, double*) {
                        return static_cast<double>(popcount_xor(qa + i * words, qb + j * words, words)) / dn;
                    });
                else
                    c.scalar([&](int64_t i, int64_t j, double*) {
                        const int64_t o = popcount_or(qa + i * words, qb + j * words, words);
                        return o == 0 ? 0.0
                                      : static_cast<double>(popcount_xor(qa + i * words, qb + j * words, words)) /
                                            static_cast<double>(o);
                    });
            }
            return;
        }
        case Metric::Cosine:
        case Metric::Correlation: {
            const std::vector<double> pa = prepared(*c.a, p);
            const std::vector<double> pb = c.b ? prepared(*c.b, p) : std::vector<double>();
            if (w) c.tile_on<WSqOp, 4, true>(pa.data(), c.b ? pb.data() : nullptr, cos_fin<true>(p));
            else c.tile_on<SqOp, 4, false>(pa.data(), c.b ? pb.data() : nullptr, cos_fin<false>(p));
            return;
        }
        case Metric::Minkowski: {
            const Mat &A = *c.a, &B = c.second();
            const double* wp = p.wp();
            if (w)
                c.scalar([&](int64_t i, int64_t j, double*) {
                    return minkowski_pair<true>(A.row<double>(i), B.row<double>(j), wp, n, p.root);
                });
            else
                c.scalar([&](int64_t i, int64_t j, double*) {
                    return minkowski_pair<false>(A.row<double>(i), B.row<double>(j), nullptr, n, p.root);
                });
            return;
        }
        case Metric::Mahalanobis: {
            const Mat &A = *c.a, &B = c.second();
            c.scalar([&](int64_t i, int64_t j, double* z) {
                return mahalanobis_pair(A.row<double>(i), B.row<double>(j), p.vi, n, z);
            });
            return;
        }
        case Metric::Haversine: {
            const Mat &A = *c.a, &B = c.second();
            const std::vector<double> ca = cosines(A);
            const std::vector<double> cb = c.b ? cosines(B) : std::vector<double>();
            const double* qb = c.b ? cb.data() : ca.data();
            c.scalar([&](int64_t i, int64_t j, double*) {
                const double *x = A.row<double>(i), *y = B.row<double>(j);
                return haversine_pair(x[0], x[1], ca[static_cast<size_t>(i)], y[0], y[1], qb[j]);
            });
            return;
        }
    }
}

// ---- the distance functions: one pair at a time ------------------------------------------------

template <class Op, bool W, class Fin>
double one(const double* x, const double* y, const Plan& p, Fin fin) {
    double s[Op::K];
    pair_scalar<Op, W>(x, y, p.wp(), p.n, s);
    return fin(s, x, y);
}

template <class Op, class Fin>
double one_w(const double* x, const double* y, const Plan& p, Fin fin) {
    return p.weighted ? one<Op, true>(x, y, p, fin) : one<Op, false>(x, y, p, fin);
}

// The distance of U[k] and V[k] for k < P, by the one-pair kernel of the metric.
void paired_run(const Mat& U, const Mat& V, const Plan& p, double* out) {
    const int64_t P = U.rows, n = p.n;
    const int64_t cost = pair_cost(p);
    const int64_t per = std::max<int64_t>(1, kTaskWork / cost);
    const bool w = p.weighted;
    std::vector<double> pu, pv;  // cosine, correlation
    std::vector<double> cu, cv;  // haversine
    if (p.metric == Metric::Cosine || p.metric == Metric::Correlation) {
        pu = prepared(U, p);
        pv = prepared(V, p);
    }
    if (p.metric == Metric::Haversine) {
        cu = cosines(U);
        cv = cosines(V);
    }
    for_tasks((P + per - 1) / per, P * cost, [&](int64_t t) {
        std::vector<double> scratch(static_cast<size_t>(p.metric == Metric::Mahalanobis ? n : 0));
        for (int64_t k = t * per, e = std::min(P, (t + 1) * per); k < e; ++k) {
            double d = 0.0;
            if (p.metric == Metric::Hamming || p.metric == Metric::Jaccard) {
                const bool ham = p.metric == Metric::Hamming;
                auto go = [&]<class T>(std::type_identity<T>) {
                    const T *x = U.row<T>(k), *y = V.row<T>(k);
                    if (ham)
                        return w ? count_pair<HammingOp, true, T>(x, y, p) : count_pair<HammingOp, false, T>(x, y, p);
                    return w ? count_pair<JaccardOp, true, T>(x, y, p) : count_pair<JaccardOp, false, T>(x, y, p);
                };
                if (p.dtype == DType::Float64) d = go(std::type_identity<double>{});
                else if (p.dtype == DType::Int64) d = go(std::type_identity<int64_t>{});
                else d = go(std::type_identity<std::uint8_t>{});
                out[k] = d;
                continue;
            }
            const double *x = U.row<double>(k), *y = V.row<double>(k);
            switch (p.metric) {
                case Metric::Euclidean:
                case Metric::SEuclidean:
                    d = w ? one<WSqOp, true>(x, y, p, euclid_fin<true>(p))
                          : one<SqOp, false>(x, y, p, euclid_fin<false>(p));
                    break;
                case Metric::SqEuclidean:
                    d = w ? one<WSqOp, true>(x, y, p, sq_fin<true>(p)) : one<SqOp, false>(x, y, p, sq_fin<false>(p));
                    break;
                case Metric::Cityblock:
                    d = w ? one<WAbsOp, true>(x, y, p, id_fin()) : one<AbsOp, false>(x, y, p, id_fin());
                    break;
                case Metric::Chebyshev:
                    d = w ? one<WMaxOp, true>(x, y, p, id_fin()) : one<MaxOp, false>(x, y, p, id_fin());
                    break;
                case Metric::Canberra:
                    d = w ? one<WCanberraOp, true>(x, y, p, id_fin()) : one<CanberraOp, false>(x, y, p, id_fin());
                    break;
                case Metric::BrayCurtis:
                    d = w ? one<WBrayOp, true>(x, y, p, bray_fin()) : one<BrayOp, false>(x, y, p, bray_fin());
                    break;
                case Metric::Cosine:
                case Metric::Correlation: {
                    const double *xh = pu.data() + k * n, *yh = pv.data() + k * n;
                    d = w ? one<WSqOp, true>(xh, yh, p, cos_fin<true>(p))
                          : one<SqOp, false>(xh, yh, p, cos_fin<false>(p));
                    break;
                }
                case Metric::Minkowski:
                    d = w ? minkowski_pair<true>(x, y, p.wp(), n, p.root)
                          : minkowski_pair<false>(x, y, nullptr, n, p.root);
                    break;
                case Metric::Mahalanobis: d = mahalanobis_pair(x, y, p.vi, n, scratch.data()); break;
                case Metric::Haversine:
                    d = haversine_pair(x[0], x[1], cu[static_cast<size_t>(k)], y[0], y[1], cv[static_cast<size_t>(k)]);
                    break;
                default: break;
            }
            out[k] = d;
        }
    });
}

// ---- bindings --------------------------------------------------------------------------------

NDArray new_f64(std::initializer_list<int64_t> shape) {
    const std::vector<int64_t> s(shape);
    return new_array(DType::Float64, s);
}

nb::object cdist_py(nb::handle xa, nb::handle xb, const std::string& metric, double p, nb::handle w, nb::handle aux) {
    const char* f = "_cdist";
    const Mat A = mat_arg(xa, f, 2), B = mat_arg(xb, f, 2);
    if (A.cols != B.cols || A.dtype != B.dtype) throw nb::type_error("_cdist(): sides differ");
    Mat vi;
    const Plan plan = make_plan(metric, p, w, aux, A.cols, A.dtype, vi, f);
    NDArray out = new_f64({A.rows, B.rows});
    run(Call{&A, &B, reinterpret_cast<double*>(out.data), &plan});
    return wrap(std::move(out));
}

nb::object pdist_py(nb::handle x, const std::string& metric, double p, nb::handle w, nb::handle aux) {
    const char* f = "_pdist";
    const Mat X = mat_arg(x, f, 2);
    Mat vi;
    const Plan plan = make_plan(metric, p, w, aux, X.cols, X.dtype, vi, f);
    const int64_t m = X.rows;
    NDArray out = new_f64({m < 2 ? 0 : m * (m - 1) / 2});
    run(Call{&X, nullptr, reinterpret_cast<double*>(out.data), &plan});
    return wrap(std::move(out));
}

nb::object paired_py(nb::handle u, nb::handle v, const std::string& metric, double p, nb::handle w, nb::handle aux) {
    const char* f = "_paired";
    const Mat U = mat_arg(u, f, 2), V = mat_arg(v, f, 2);
    if (U.cols != V.cols || U.rows != V.rows || U.dtype != V.dtype) throw nb::type_error("_paired(): sides differ");
    Mat vi;
    const Plan plan = make_plan(metric, p, w, aux, U.cols, U.dtype, vi, f);
    NDArray out = new_f64({U.rows});
    paired_run(U, V, plan, reinterpret_cast<double*>(out.data));
    return wrap(std::move(out));
}

template <class T>
void tomatrix_typed(const T* v, int64_t d, T* o) {
    for (int64_t i = 0; i < d; ++i) {
        o[i * d + i] = T(0);
        for (int64_t j = i + 1; j < d; ++j) {
            const T x = v[cond_index(i, j, d)];
            o[i * d + j] = x;
            o[j * d + i] = x;
        }
    }
}

nb::object squareform_tomatrix(nb::handle v, int64_t d) {
    const Mat V = mat_arg(v, "_squareform_tomatrix", 1);
    if (V.cols != (d < 2 ? 0 : d * (d - 1) / 2)) throw nb::type_error("_squareform_tomatrix(): wrong length");
    const int64_t shape[] = {d, d};
    NDArray out = new_array(V.dtype, shape);
    {
        const parallel::Unlocked unlocked(d * d >= kParallelWork);
        visit_dtype(V.dtype, [&]<class T>(std::type_identity<T>) {
            if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int64_t> || std::is_same_v<T, double>)
                tomatrix_typed<T>(reinterpret_cast<const T*>(V.data), d, reinterpret_cast<T*>(out.data));
        });
    }
    return wrap(std::move(out));
}

template <class T>
int tovector_typed(const T* x, int64_t d, bool checks, T* o) {
    if (checks) {
        for (int64_t i = 0; i < d; ++i)
            for (int64_t j = i + 1; j < d; ++j) {
                const T a = x[i * d + j], b = x[j * d + i];
                if (!(a == b)) {
                    if constexpr (std::is_same_v<T, double>)
                        if (a != a || b != b) return 2;
                    return 1;
                }
            }
        for (int64_t i = 0; i < d; ++i)
            if (!(x[i * d + i] == T(0))) return 3;
    }
    for (int64_t i = 0; i < d; ++i)
        for (int64_t j = i + 1; j < d; ++j) o[cond_index(i, j, d)] = x[i * d + j];
    return 0;
}

nb::tuple squareform_tovector(nb::handle x, bool checks) {
    const Mat X = mat_arg(x, "_squareform_tovector", 2);
    const int64_t d = X.rows;
    if (X.cols != d) throw nb::type_error("_squareform_tovector(): not square");
    const int64_t shape[] = {d < 2 ? 0 : d * (d - 1) / 2};
    NDArray out = new_array(X.dtype, shape);
    int code = 0;
    {
        const parallel::Unlocked unlocked(d * d >= kParallelWork);
        visit_dtype(X.dtype, [&]<class T>(std::type_identity<T>) {
            if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int64_t> || std::is_same_v<T, double>)
                code = tovector_typed<T>(reinterpret_cast<const T*>(X.data), d, checks, reinterpret_cast<T*>(out.data));
        });
    }
    if (code != 0) return nb::make_tuple(code, nb::none());
    return nb::make_tuple(0, wrap(std::move(out)));
}

}  // namespace

void bind_distance(nb::module_& m) {
    m.def(
        "_cdist", &cdist_py, nb::arg("XA"), nb::arg("XB"), nb::arg("metric"), nb::arg("p"), nb::arg("w").none(),
        nb::arg("aux").none()
    );
    m.def(
        "_pdist", &pdist_py, nb::arg("X"), nb::arg("metric"), nb::arg("p"), nb::arg("w").none(), nb::arg("aux").none()
    );
    m.def(
        "_paired", &paired_py, nb::arg("U"), nb::arg("V"), nb::arg("metric"), nb::arg("p"), nb::arg("w").none(),
        nb::arg("aux").none()
    );
    m.def("_squareform_tomatrix", &squareform_tomatrix, nb::arg("v"), nb::arg("d"));
    m.def("_squareform_tovector", &squareform_tovector, nb::arg("X"), nb::arg("checks"));
}

}  // namespace sd::spatial
