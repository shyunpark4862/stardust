// kneighbors and radius_neighbors of sd.spatial (docs/design/spatial.md §2.3, §3.2, §3.4, §3.6; milestone
// SPT2). The Python layer (src/stardust/spatial/_neighbors.py) checks every argument.
//
//   _kneighbors(XA, XB, metric, p, w, aux, k, upper, path)
//       -> (distances (mA, k) float64, indices (mA, k) int64): row i holds the k smallest (d, j) of the
//          pairs (XA[i], XB[j]) with d < upper, in (d, j) order; missing slots are (inf, mB).
//   _radius_neighbors(XA, XB, metric, p, w, aux, r, count_only, path)
//       -> (indptr (mA + 1), indices, distances): row i holds every j with d <= r[i], in j order;
//          count_only: the counts (mA,) int64. r: (mA,) float64.
//       metric, p, w, aux: as _cdist's (csrc/spatial/distance.cpp). path: 0 chooses by the metric and
//       the shapes, 1 forces the direct path, 2 the GEMM pre-selection when the metric allows it (tests).
//
// The answer is fixed by the definition: every returned distance and every comparison that decides
// membership or order uses d, _cdist's value of the pair (the one pair kernel), and the order is the
// total order (d, j). Two paths compute it:
//
//   direct  the cdist of a block of XA rows against a block of XB rows (CdistBlocks, _cdist's own
//           drivers), then a heap or a list per row. Every metric.
//   GEMM    unweighted euclidean and sqeuclidean (decisions SPT2, SPT28): G = A' B'^T (Accelerate's
//           dgemm) of the rows moved by c, the column mean of XB, gives the lower bound L~ of §3.2
//           (theorem 3); a pair whose L~ exceeds the current threshold T is dropped, every other pair is
//           recomputed by the pair kernel on the original rows (its d decides). So the answer does not
//           depend on the GEMM, its blocks or Accelerate's thread mode.
//
// Work is cut by size alone: chunks of XA rows and, when they are few, segments of XB, each (chunk,
// segment) a task with its own heaps; segments are merged in (d, j) order afterwards. Since (d, j) is
// a total order, the answer is the same for any split and any thread count. Working memory is bounded
// (§3.6): per task a block of distances (direct) or the moved rows, the GEMM block and the bounds
// (GEMM), and the heaps (at most kHeapEntries slots, or k for one row); the GEMM path keeps two doubles
// per XB row (the moved row's squared norm and R).
#include "accelerate/blas.hpp"
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "spatial/metric.hpp"
#include "spatial/neighbors.hpp"
#include "spatial/spatial.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::spatial {

namespace {

using std::int64_t;

constexpr double kInf = std::numeric_limits<double>::infinity();
// The array area's rule (docs/design/array.md §10.4): from 2^15 units of work a call splits and runs
// without the GIL, whatever the thread count.
constexpr int64_t kParallelWork = 1 << 15;
constexpr int64_t kMinTasks = 32;           // tasks wanted (XA chunks, then XB segments)
constexpr int64_t kSegmentMin = 4096;       // fewest XB rows of a segment
constexpr int64_t kHeapEntries = 1 << 16;   // (row, rank) slots of a task's heaps
constexpr int64_t kSegmentSlots = 1 << 22;  // slots of kneighbors' segment buffers
constexpr int64_t kDirectRows = 64;         // XA rows of a direct chunk
constexpr int64_t kDirectBlock = 1 << 16;   // doubles of a direct block of distances
constexpr int64_t kDirectPack = 1 << 18;    // doubles of the XB rows a direct block packs
constexpr int64_t kGemmRows = 256;          // XA rows of a GEMM chunk (at most)
constexpr int64_t kGemmCols = 2048;         // XB rows of a GEMM block (at most)
constexpr int64_t kGemmData = 1 << 19;      // doubles of a chunk's or a block's moved rows (at most)
// The GEMM pre-selection is used for kGemmMinN <= n <= kGemmMaxN (SPT28 allows it up to 2^21 - 2: longer
// rows gain nothing), from kGemmMinRowsA rows of XA and kGemmMinRowsB rows of XB (kGemmMinRowsBWide for
// n >= kGemmWideN): below them the moved copy, the norms and the small GEMM blocks cost more than they save
// (decision SPT74, measured in SPT6; it was 12 <= n, 64 rows of XB in SPT40).
constexpr int64_t kGemmMinN = 1;
constexpr int64_t kGemmMaxN = 1 << 16;
constexpr int64_t kGemmMinRowsA = 64;
constexpr int64_t kGemmMinRowsB = 2048;
constexpr int64_t kGemmWideN = 8;
constexpr int64_t kGemmMinRowsBWide = 1024;
constexpr int64_t kMeanRows = 4096;  // rows of a partial column sum of XB

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

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ma x mb pairs of `cost` pair-terms, saturated (no output bounds ma mb here, so the product may pass int64)
inline int64_t work_of(int64_t ma, int64_t mb, int64_t cost) {
    const double w =
        static_cast<double>(ma) * static_cast<double>(std::max<int64_t>(mb, 1)) * static_cast<double>(cost);
    return w >= 0x1p62 ? (int64_t{1} << 62) : static_cast<int64_t>(w);
}
inline int64_t down8(int64_t v) { return std::max<int64_t>(8, v / 8 * 8); }

// ---- (d, j) and the heaps ------------------------------------------------------------------------

struct Nb {
    double d;
    int64_t j;
};

// The total order of the answer (SPT4): distance, then index. No NaN gets here (d < upper or d <= r
// is false for it).
inline bool before(const Nb& a, const Nb& b) { return a.d < b.d || (a.d == b.d && a.j < b.j); }

// The k best (d, j) so far: a max-heap on (d, j), h[0] the worst.
struct Heap {
    Nb* h = nullptr;
    int64_t k = 0, size = 0;

    bool full() const { return size == k; }
    double bound() const { return size == k ? h[0].d : kInf; }  // T of §3.2
    void push(Nb x) {
        if (size < k) {
            int64_t i = size++;
            while (i > 0) {
                const int64_t up = (i - 1) / 2;
                if (!before(h[up], x)) break;
                h[i] = h[up];
                i = up;
            }
            h[i] = x;
            return;
        }
        if (!before(x, h[0])) return;
        int64_t i = 0;
        for (;;) {
            const int64_t l = 2 * i + 1;
            if (l >= k) break;
            const int64_t c = (l + 1 < k && before(h[l], h[l + 1])) ? l + 1 : l;
            if (!before(x, h[c])) break;
            h[i] = h[c];
            i = c;
        }
        h[i] = x;
    }
    void sort() { std::sort(h, h + size, before); }
};

// ---- the split -----------------------------------------------------------------------------------

struct Layout {
    int64_t qa = 1, nA = 0;  // XA rows per chunk, chunks
    int64_t seg = 8, S = 1;  // XB rows per segment, segments
};

// By the sizes alone. k: kneighbors' k (0: radius_neighbors).
Layout layout(int64_t ma, int64_t mb, int64_t qa_max, int64_t k) {
    Layout L;
    L.qa = k > 0 ? std::clamp<int64_t>(kHeapEntries / k, 1, qa_max) : qa_max;
    L.nA = ma == 0 ? 0 : ceil_div(ma, L.qa);
    int64_t S = 1;
    if (L.nA > 0 && L.nA < kMinTasks) {
        S = std::min(ceil_div(kMinTasks, L.nA), std::max<int64_t>(1, mb / kSegmentMin));
        if (k > 0) S = std::min(S, std::max<int64_t>(1, kSegmentSlots / std::max<int64_t>(1, ma * k)));
    }
    L.seg = std::max<int64_t>(8, ceil_div(std::max<int64_t>(mb, 1), S) + 7) / 8 * 8;
    L.S = mb == 0 ? 1 : ceil_div(mb, L.seg);
    return L;
}

// XA rows of a direct chunk: kDirectRows, fewer for a small XA so that it still makes kMinTasks chunks.
int64_t direct_rows(int64_t ma) {
    return std::clamp<int64_t>((ceil_div(std::max<int64_t>(ma, 1), kMinTasks) + 7) / 8 * 8, 8, kDirectRows);
}

int64_t direct_block(int64_t rows, int64_t n) {
    const int64_t bb = std::min<int64_t>(4096, kDirectBlock / std::max<int64_t>(rows, 1));
    return down8(std::min(bb, kDirectPack / std::max<int64_t>(n, 1)));
}

// ---- the GEMM pre-selection (§3.2, decision SPT28: the constants are exact doubles) ------------

struct Consts {
    double c1, eta0, c2, c5, c3;
    explicit Consts(int64_t n)
        : c1(static_cast<double>(n + 3) * 0x1p-53),
          eta0(static_cast<double>(n + 4) * 0x1p-1070),
          c2(1.0 - static_cast<double>(n + 16) * 0x1p-53),
          c5(1.0 - static_cast<double>(n + 24) * 0x1p-53),
          c3(1.0 + static_cast<double>(2 * (n + 6)) * 0x1p-53) {}
};
constexpr double kRho = 0x1p-525;
constexpr double kVMin = 0x1p-500;

// sum x_i^2 in the tile order (the direct kernel's sum of squares of x - 0)
inline double sumsq_tile(const double* x, int64_t n) {
    double s = 0.0;
    for (int64_t b = 0; b < n; b += kBlock) {
        double t = 0.0;
        const int64_t e = std::min(n, b + kBlock);
        for (int64_t i = b; i < e; ++i) t = std::fma(x[i], x[i], t);
        s = b == 0 ? t : s + t;
    }
    return s;
}

// R = fl(fl(fl(sqrt(nhat)) c3) + rho) >= ||x'|| (theorem 3, step 1)
inline double radius_of(double nhat, const Consts& K) {
    const double r = std::sqrt(nhat);
    const double rc = r * K.c3;
    return rc + kRho;
}

// The lower bound L~ of a pair from G = x'.y', or -inf when the pair cannot be dropped (t > 0 fails,
// NaN included, or v < 2^-500). sq: sqeuclidean (L~ bounds s^), else euclidean (bounds d^).
inline double lower_one(double g, double ny, double ry, double nx, double rx, const Consts& K, bool sq) {
    const double sn = nx + ny;
    const double e_hat = sn - (g + g);
    const double a = rx + ry;
    const double aa = a * a;
    const double delta = K.c1 * aa + K.eta0;
    const double t = e_hat - delta;
    if (!(t > 0.0)) return -kInf;
    const double e = 0x1p-50 * a + 0x1p-1074;
    const double v = std::sqrt(t) - e;
    if (!(v >= kVMin)) return -kInf;
    if (sq) {
        const double vv = v * v;
        return vv * K.c5;
    }
    return K.c2 * v;
}

// lt[c] = lower_one(g[c], ny[c], ry[c], nx, rx) for c < w, two lanes at a time (the same IEEE
// operations, so the same values).
void lower_row(
    const double* g,
    const double* ny,
    const double* ry,
    int64_t w,
    double nx,
    double rx,
    const Consts& K,
    bool sq,
    double* lt
) {
    const float64x2_t vnx = vdupq_n_f64(nx), vrx = vdupq_n_f64(rx), c1 = vdupq_n_f64(K.c1), eta0 = vdupq_n_f64(K.eta0),
                      zero = vdupq_n_f64(0.0), s50 = vdupq_n_f64(0x1p-50), tiny = vdupq_n_f64(0x1p-1074),
                      vmin = vdupq_n_f64(kVMin), c2 = vdupq_n_f64(K.c2), c5 = vdupq_n_f64(K.c5),
                      ninf = vdupq_n_f64(-kInf);
    int64_t c = 0;
    for (; c + 2 <= w; c += 2) {
        const float64x2_t gv = vld1q_f64(g + c);
        const float64x2_t sn = vaddq_f64(vnx, vld1q_f64(ny + c));
        const float64x2_t e_hat = vsubq_f64(sn, vaddq_f64(gv, gv));
        const float64x2_t a = vaddq_f64(vrx, vld1q_f64(ry + c));
        const float64x2_t delta = vaddq_f64(vmulq_f64(c1, vmulq_f64(a, a)), eta0);
        const float64x2_t t = vsubq_f64(e_hat, delta);
        const uint64x2_t pos = vcgtq_f64(t, zero);
        const float64x2_t e = vaddq_f64(vmulq_f64(s50, a), tiny);
        const float64x2_t v = vsubq_f64(vsqrtq_f64(t), e);
        const uint64x2_t ok = vandq_u64(pos, vcgeq_f64(v, vmin));
        const float64x2_t l = sq ? vmulq_f64(vmulq_f64(v, v), c5) : vmulq_f64(c2, v);
        vst1q_f64(lt + c, vbslq_f64(ok, l, ninf));
    }
    for (; c < w; ++c) lt[c] = lower_one(g[c], ny[c], ry[c], nx, rx, K, sq);
}

// The distance of one pair by the pair kernel: _cdist's bits (the tile kernel's lane is pair_scalar's,
// and the finish is the same function).
inline double pair_d(const double* x, const double* y, int64_t n, bool sq) {
    double s;
    pair_scalar<SqOp, false>(x, y, nullptr, n, &s);
    return sq ? sq_finish<false>(s, x, y, nullptr, n) : euclid_finish<false>(s, x, y, nullptr, n);
}

// The block-wide bound of the pre-selection (SPT71). With a_max = fl(R_x + max R_y over a block of XB rows),
// every step of L~ is monotone in a (the products, sums and square roots round monotonically; Delta~ and
// e grow with a, t and v shrink as they grow), so L'(t') = fl(c2 fl(fl(sqrt(t')) - e_max)) with t' =
// fl(E^ - Delta~(a_max)) is at most the pair's own L~, and t' > 0, v' >= 2^-500 imply the pair's t > 0,
// v >= 2^-500. L' is nondecreasing in t', so for a threshold T a value tau with L'(next(tau)) > T proves
// L~ > T for every t' > tau: such a pair is dropped after an add, a subtract and a compare. The other
// pairs get their own L~ (lower_one). The answer is the same (only provably dropped pairs are skipped).
struct Fast {
    double dmax = 0.0, emax = 0.0;
    bool sq = false;
    const Consts* K = nullptr;

    void set(double amax, const Consts& k, bool s) {
        K = &k;
        sq = s;
        const double aa = amax * amax;
        dmax = k.c1 * aa + k.eta0;
        emax = 0x1p-50 * amax + 0x1p-1074;
    }
    // L'(t) (-inf when the pair could not be dropped)
    double lower(double t) const {
        if (!(t > 0.0)) return -kInf;
        const double v = std::sqrt(t) - emax;
        if (!(v >= kVMin)) return -kInf;
        if (sq) {
            const double vv = v * v;
            return vv * K->c5;
        }
        return K->c2 * v;
    }
    // A tau with L'(next(tau)) > T (inf: none found, nothing is dropped fast).
    double tau(double T) const {
        if (!(T < kInf) || !(T >= 0.0)) return kInf;
        const double r = sq ? std::sqrt(T / K->c5) : T / K->c2;
        double t = (r * (1.0 + 0x1p-40) + emax) * (1.0 + 0x1p-40);
        t = t * t * (1.0 + 0x1p-40);
        for (int it = 0; it < 64 && t < kInf; ++it) {
            if (lower(std::nextafter(t, kInf)) > T) return t;
            t = t * (1.0 + 0x1p-20) + 0x1p-1000;
        }
        return kInf;
    }
};

// What the GEMM path keeps for a call: the column mean c of XB, per XB row the moved row's squared norm
// and R, and (when it fits in kMovedCopy doubles) the moved rows themselves, made once for all tasks.
struct Hybrid {
    int64_t n = 0;
    bool sq = false;
    Consts K{1};
    std::vector<double> c, nb, rb, moved;
    int64_t qa = 1, bb = 8;  // chunk rows (at most), block rows
};

constexpr int64_t kMovedCopy = int64_t{1} << 27;  // doubles of the moved XB kept for a call (1 GB)

void prepare_hybrid(Hybrid& hy, const CdistBlocks& cb, bool sq) {
    const int64_t mb = cb.rows_b(), n = cb.cols();
    const double* B = cb.b_f64();
    hy.n = n;
    hy.sq = sq;
    hy.K = Consts(n);
    // chunks of XA: at most kGemmRows rows, fewer for a small XA so that it still makes kMinTasks chunks
    hy.qa = std::clamp<int64_t>(std::min(kGemmData / n, (ceil_div(cb.rows_a(), kMinTasks) + 7) / 8 * 8), 8, kGemmRows);
    hy.bb = down8(std::min<int64_t>(kGemmCols, kGemmData / n));
    // column mean: partial sums over fixed row chunks, added in chunk order
    const int64_t chunks = ceil_div(mb, kMeanRows);
    std::vector<double> part(static_cast<size_t>(chunks * n), 0.0);
    for_tasks(chunks, mb * n, [&](int64_t t) {
        double* p = part.data() + t * n;
        for (int64_t j = t * kMeanRows, e = std::min(mb, (t + 1) * kMeanRows); j < e; ++j)
            for (int64_t i = 0; i < n; ++i) p[i] = p[i] + B[j * n + i];
    });
    hy.c.assign(static_cast<size_t>(n), 0.0);
    for (int64_t t = 0; t < chunks; ++t)
        for (int64_t i = 0; i < n; ++i) hy.c[static_cast<size_t>(i)] = hy.c[static_cast<size_t>(i)] + part[t * n + i];
    for (int64_t i = 0; i < n; ++i)
        hy.c[static_cast<size_t>(i)] = hy.c[static_cast<size_t>(i)] / static_cast<double>(mb);
    hy.nb.resize(static_cast<size_t>(mb));
    hy.rb.resize(static_cast<size_t>(mb));
    const bool keep = mb * n <= kMovedCopy;
    if (keep) hy.moved.resize(static_cast<size_t>(mb * n));
    for_tasks(chunks, mb * n, [&](int64_t t) {
        std::vector<double> scratch(keep ? 0 : static_cast<size_t>(n));
        for (int64_t j = t * kMeanRows, e = std::min(mb, (t + 1) * kMeanRows); j < e; ++j) {
            double* y = keep ? hy.moved.data() + j * n : scratch.data();
            for (int64_t i = 0; i < n; ++i) y[i] = B[j * n + i] - hy.c[static_cast<size_t>(i)];
            const double s = sumsq_tile(y, n);
            hy.nb[static_cast<size_t>(j)] = s;
            hy.rb[static_cast<size_t>(j)] = radius_of(s, hy.K);
        }
    });
}

// One task of the GEMM path: rows [i0, i1) of XA against rows [s0, s1) of XB. `visit(r, j0, w, g, nx, rx,
// fast, x)` is called per row r and block [j0, j0 + w) with the row of G = x' y'^T (null: the block cannot
// use the pre-selection, every pair is a candidate), the moved row's squared norm and R, the block-wide
// bound (Fast), and the row itself; it decides the candidates and recomputes them.
template <class Visit>
void gemm_task(const CdistBlocks& cb, const Hybrid& hy, int64_t i0, int64_t i1, int64_t s0, int64_t s1, Visit visit) {
    const int64_t n = hy.n, rows = i1 - i0;
    const double *A = cb.a_f64(), *B = cb.b_f64();
    const double* c = hy.c.data();
    std::vector<double> ap(static_cast<size_t>(rows * n)), na(static_cast<size_t>(rows)), ra(static_cast<size_t>(rows));
    bool a_ok = true;
    double ra_max = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        double* o = ap.data() + r * n;
        for (int64_t i = 0; i < n; ++i) o[i] = A[(i0 + r) * n + i] - c[i];
        na[static_cast<size_t>(r)] = sumsq_tile(o, n);
        ra[static_cast<size_t>(r)] = radius_of(na[static_cast<size_t>(r)], hy.K);
        a_ok = a_ok && std::isfinite(na[static_cast<size_t>(r)]);
        ra_max = std::max(ra_max, ra[static_cast<size_t>(r)]);
    }
    const int64_t bb = hy.bb;
    const bool keep = !hy.moved.empty();
    std::vector<double> bp(keep ? 0 : static_cast<size_t>(bb * n)), g(static_cast<size_t>(rows * bb));
    Fast fast;
    for (int64_t j0 = s0; j0 < s1; j0 += bb) {
        const int64_t w = std::min(s1, j0 + bb) - j0;
        bool ok = a_ok;
        double rb_max = 0.0;
        for (int64_t q = 0; q < w; ++q) {
            ok = ok && std::isfinite(hy.nb[static_cast<size_t>(j0 + q)]);
            rb_max = std::max(rb_max, hy.rb[static_cast<size_t>(j0 + q)]);
        }
        const double am = ra_max + rb_max;
        ok = ok && std::isfinite(am * am);  // the fallback of §3.2: coordinates beyond about 1e154
        if (ok) {
            const double* yb = hy.moved.data() + j0 * n;
            if (!keep) {
                for (int64_t q = 0; q < w; ++q) {
                    double* o = bp.data() + q * n;
                    for (int64_t i = 0; i < n; ++i) o[i] = B[(j0 + q) * n + i] - c[i];
                }
                yb = bp.data();
            }
            blas::gemm<double>(CblasNoTrans, CblasTrans, rows, w, n, ap.data(), n, yb, n, g.data(), w);
        }
        for (int64_t r = 0; r < rows; ++r) {
            const double rx = ra[static_cast<size_t>(r)];
            if (ok) fast.set(rx + rb_max, hy.K, hy.sq);
            visit(r, j0, w, ok ? g.data() + r * w : nullptr, na[static_cast<size_t>(r)], rx, fast, A + (i0 + r) * n);
        }
    }
}

// The pairs q < w of a row that the block-wide bound cannot drop (t' <= tau, or NaN), into idx.
inline int64_t survivors(
    const double* g, const double* ny, int64_t w, double nx, const Fast& f, double tau, int64_t* idx
) {
    int64_t cnt = 0;
    const float64x2_t vnx = vdupq_n_f64(nx), vd = vdupq_n_f64(f.dmax), vt = vdupq_n_f64(tau);
    int64_t q = 0;
    for (; q + 2 <= w; q += 2) {
        const float64x2_t gv = vld1q_f64(g + q);
        const float64x2_t e_hat = vsubq_f64(vaddq_f64(vnx, vld1q_f64(ny + q)), vaddq_f64(gv, gv));
        const uint64x2_t drop = vcgtq_f64(vsubq_f64(e_hat, vd), vt);
        if (vgetq_lane_u64(drop, 0) == 0) idx[cnt++] = q;
        if (vgetq_lane_u64(drop, 1) == 0) idx[cnt++] = q + 1;
    }
    for (; q < w; ++q) {
        const double e_hat = (nx + ny[q]) - (g[q] + g[q]);
        if (!(e_hat - f.dmax > tau)) idx[cnt++] = q;
    }
    return cnt;
}

// ---- kneighbors ---------------------------------------------------------------------------------

bool use_gemm(const CdistBlocks& cb, int path) {
    const int64_t n = cb.cols();
    if (cb.gemm_kind() == 0 || n < 1 || n > kGemmMaxN || cb.rows_b() == 0 || path == 1) return false;
    if (path == 2) return true;
    const int64_t min_b = n >= kGemmWideN ? kGemmMinRowsBWide : kGemmMinRowsB;
    return n >= kGemmMinN && cb.rows_a() >= kGemmMinRowsA && cb.rows_b() >= min_b;
}

// The direct path for one task: heaps[r] for XA row i0 + r over XB rows [s0, s1).
void knn_direct(const CdistBlocks& cb, int64_t i0, int64_t i1, int64_t s0, int64_t s1, double upper, Heap* heaps) {
    const int64_t rows = i1 - i0, bb = direct_block(rows, cb.cols());
    std::vector<double> buf(static_cast<size_t>(rows * std::min(bb, std::max<int64_t>(s1 - s0, 1))));
    for (int64_t j0 = s0; j0 < s1; j0 += bb) {
        const int64_t j1 = std::min(s1, j0 + bb), w = j1 - j0;
        cb.block(i0, i1, j0, j1, buf.data());
        for (int64_t r = 0; r < rows; ++r) {
            Heap& h = heaps[r];
            const double* row = buf.data() + r * w;
            for (int64_t q = 0; q < w; ++q) {
                const double d = row[q];
                if (!(d < upper) || (h.full() && !(d <= h.h[0].d))) continue;
                h.push({d, j0 + q});
            }
        }
    }
}

// The GEMM path for one task. A block whose heap is not full first recomputes the pairs it still needs of
// smallest E^ (the first block's rule of §3.2, decision SPT41, with E^ for L~: SPT72), so that T is set
// early; if the heap is still not full, every other pair's L~ is compared with T. A block whose heap is
// full drops most pairs by the block-wide bound (Fast) and collects the rest with L~ <= T; when they are
// many (data sorted from far to near, where every block lowers T), the k of smallest L~ are recomputed
// first, which sets T for the others (SPT72). Any of these orders gives the same answer: the heap ends
// with the k smallest (d, j) of the pairs not proven beyond T.
void knn_gemm(
    const CdistBlocks& cb, const Hybrid& hy, int64_t i0, int64_t i1, int64_t s0, int64_t s1, double upper, Heap* heaps
) {
    const int64_t n = hy.n, k = heaps[0].k;
    const double* B = cb.b_f64();
    std::vector<Nb> pick(static_cast<size_t>(std::min(hy.bb, k)));
    std::vector<double> lt(static_cast<size_t>(hy.bb));
    std::vector<int64_t> idx(static_cast<size_t>(hy.bb));
    std::vector<char> taken(static_cast<size_t>(hy.bb), 0);
    std::vector<Nb> cand;
    gemm_task(
        cb, hy, i0, i1, s0, s1,
        [&](int64_t r, int64_t j0, int64_t w, const double* g, double nx, double rx, const Fast& fast,
            const double* x) {
            Heap& h = heaps[r];
            auto take = [&](int64_t j) {
                const double d = pair_d(x, B + j * n, n, hy.sq);
                if (d < upper) h.push({d, j});
            };
            if (!g) {
                for (int64_t q = 0; q < w; ++q) take(j0 + q);
                return;
            }
            const double* ny = hy.nb.data() + j0;
            const double* ry = hy.rb.data() + j0;
            int64_t picked = 0;  // pairs taken first (marked in `taken`)
            if (!h.full()) {
                // the pairs of smallest E^ first: any choice gives the same answer, E^ is cheap and close to L~
                Heap sel{pick.data(), std::min(h.k - h.size, w), 0};
                for (int64_t q = 0; q < w; ++q) {
                    const double e_hat = (nx + ny[q]) - (g[q] + g[q]);
                    if (!sel.full() || e_hat < sel.h[0].d) sel.push({e_hat, q});
                }
                picked = sel.size;
                for (int64_t e = 0; e < picked; ++e) {
                    const int64_t q = sel.h[e].j;
                    take(j0 + q);
                    taken[static_cast<size_t>(q)] = 1;
                }
                if (!h.full()) {  // still fewer than k below the bound: every L~ against T
                    lower_row(g, ny, ry, w, nx, rx, hy.K, hy.sq, lt.data());
                    double T = h.bound();
                    for (int64_t q = 0; q < w; ++q) {
                        if (taken[static_cast<size_t>(q)]) continue;
                        const double l = lt[q];
                        if (!(l < upper) || l > T) continue;  // L~ <= d^: the pair cannot qualify
                        take(j0 + q);
                        T = h.bound();
                    }
                    for (int64_t e = 0; e < picked; ++e) taken[static_cast<size_t>(sel.h[e].j)] = 0;
                    return;
                }
            }
            double T = h.bound();
            const int64_t cnt = survivors(g, ny, w, nx, fast, fast.tau(T), idx.data());
            cand.clear();
            for (int64_t e = 0; e < cnt; ++e) {
                const int64_t q = idx[static_cast<size_t>(e)];
                if (picked && taken[static_cast<size_t>(q)]) continue;
                const double l = lower_one(g[q], ny[q], ry[q], nx, rx, hy.K, hy.sq);
                if (l <= T) cand.push_back({l, q});  // -inf too: a pair that cannot be dropped
            }
            if (picked)
                for (int64_t q = 0; q < w; ++q) taken[static_cast<size_t>(q)] = 0;
            if (static_cast<int64_t>(cand.size()) > 2 * k) {
                std::nth_element(cand.begin(), cand.begin() + (k - 1), cand.end(), before);
                for (int64_t e = 0; e < k; ++e) take(j0 + cand[static_cast<size_t>(e)].j);
                T = h.bound();
                for (size_t e = static_cast<size_t>(k); e < cand.size(); ++e) {
                    if (cand[e].d > T) continue;
                    take(j0 + cand[e].j);
                    T = h.bound();
                }
                return;
            }
            for (const Nb& e : cand) {
                if (e.d > T) continue;
                take(j0 + e.j);
                T = h.bound();
            }
        }
    );
}

nb::object new_array_of(DType t, std::initializer_list<int64_t> shape) {
    const std::vector<int64_t> s(shape);
    return wrap(new_array(t, s));
}

nb::tuple kneighbors_py(
    nb::handle xa,
    nb::handle xb,
    const std::string& metric,
    double p,
    nb::handle w,
    nb::handle aux,
    int64_t k,
    double upper,
    int path
) {
    const char* f = "_kneighbors";
    if (k < 1) throw nb::value_error("_kneighbors(): k < 1");
    const CdistBlocks cb(xa, xb, metric, p, w, aux, f);
    const int64_t ma = cb.rows_a(), mb = cb.rows_b(), n = cb.cols();
    nb::object od = new_array_of(DType::Float64, {ma, k}), oi = new_array_of(DType::Int64, {ma, k});
    double* D = reinterpret_cast<double*>(as_ndarray(od).data);
    int64_t* I = reinterpret_cast<int64_t*>(as_ndarray(oi).data);
    const bool gemm = use_gemm(cb, path);
    Hybrid hy;
    if (gemm) prepare_hybrid(hy, cb, cb.gemm_kind() == 2);
    const Layout L = layout(ma, mb, gemm ? hy.qa : direct_rows(ma), k);
    std::vector<Nb> segs;
    std::vector<int64_t> segn;
    if (L.S > 1) {
        segs.resize(static_cast<size_t>(L.S * ma * k));
        segn.resize(static_cast<size_t>(L.S * ma));
    }
    auto write = [&](int64_t i, const Nb* e, int64_t cnt) {
        for (int64_t q = 0; q < k; ++q) {
            D[i * k + q] = q < cnt ? e[q].d : kInf;
            I[i * k + q] = q < cnt ? e[q].j : mb;
        }
    };
    const int64_t work = work_of(ma, mb, gemm ? std::max<int64_t>(n, 1) : cb.pair_cost());
    for_tasks(L.nA * L.S, work, [&](int64_t t) {
        const int64_t a = t / L.S, s = t % L.S;
        const int64_t i0 = a * L.qa, i1 = std::min(ma, i0 + L.qa), s0 = std::min(mb, s * L.seg),
                      s1 = std::min(mb, s0 + L.seg), rows = i1 - i0;
        std::vector<Nb> store(static_cast<size_t>(rows * k));
        std::vector<Heap> heaps(static_cast<size_t>(rows));
        for (int64_t r = 0; r < rows; ++r) heaps[static_cast<size_t>(r)] = Heap{store.data() + r * k, k, 0};
        if (gemm) knn_gemm(cb, hy, i0, i1, s0, s1, upper, heaps.data());
        else knn_direct(cb, i0, i1, s0, s1, upper, heaps.data());
        for (int64_t r = 0; r < rows; ++r) {
            Heap& h = heaps[static_cast<size_t>(r)];
            h.sort();
            if (L.S == 1) {
                write(i0 + r, h.h, h.size);
            } else {
                std::copy(h.h, h.h + h.size, segs.data() + (s * ma + i0 + r) * k);
                segn[static_cast<size_t>(s * ma + i0 + r)] = h.size;
            }
        }
    });
    if (L.S > 1) {  // merge the segments' sorted lists, row by row
        const int64_t per = std::max<int64_t>(1, (1 << 14) / (k * L.S));
        for_tasks(ceil_div(ma, per), ma * k * L.S, [&](int64_t t) {
            std::vector<int64_t> pos(static_cast<size_t>(L.S));
            std::vector<Nb> out(static_cast<size_t>(k));
            for (int64_t i = t * per, e = std::min(ma, (t + 1) * per); i < e; ++i) {
                std::fill(pos.begin(), pos.end(), 0);
                int64_t cnt = 0;
                while (cnt < k) {
                    int64_t best = -1;
                    for (int64_t s = 0; s < L.S; ++s) {
                        if (pos[static_cast<size_t>(s)] >= segn[static_cast<size_t>(s * ma + i)]) continue;
                        const Nb& c = segs[static_cast<size_t>((s * ma + i) * k + pos[static_cast<size_t>(s)])];
                        if (best < 0 ||
                            before(c, segs[static_cast<size_t>((best * ma + i) * k + pos[static_cast<size_t>(best)])]))
                            best = s;
                    }
                    if (best < 0) break;
                    out[static_cast<size_t>(cnt++)] =
                        segs[static_cast<size_t>((best * ma + i) * k + pos[static_cast<size_t>(best)]++)];
                }
                write(i, out.data(), cnt);
            }
        });
    }
    return nb::make_tuple(od, oi);
}

// ---- radius_neighbors -------------------------------------------------------------------------

// What a task found: per row the count, and (unless counting) the (j, d) of the rows one after another.
struct Found {
    std::vector<int64_t> cnt;
    std::vector<int64_t> idx;
    std::vector<double> dist;
};

nb::object radius_py(
    nb::handle xa,
    nb::handle xb,
    const std::string& metric,
    double p,
    nb::handle w,
    nb::handle aux,
    nb::handle r_arr,
    bool count_only,
    int path
) {
    const char* f = "_radius_neighbors";
    const CdistBlocks cb(xa, xb, metric, p, w, aux, f);
    const int64_t ma = cb.rows_a(), mb = cb.rows_b(), n = cb.cols();
    if (!is_ndarray(r_arr)) throw nb::type_error("_radius_neighbors(): r must be an sd.ndarray");
    const NDArray& ra = as_ndarray(r_arr);
    if (ra.dtype != DType::Float64 || ra.ndim != 1 || ra.shape[0] != ma || !ra.c_contiguous)
        throw nb::type_error("_radius_neighbors(): wrong r");
    const double* R = reinterpret_cast<const double*>(ra.data);
    const bool gemm = use_gemm(cb, path);
    Hybrid hy;
    if (gemm) prepare_hybrid(hy, cb, cb.gemm_kind() == 2);
    const Layout L = layout(ma, mb, gemm ? hy.qa : direct_rows(ma), 0);
    std::vector<Found> found(static_cast<size_t>(L.nA * L.S));
    const int64_t work = work_of(ma, mb, gemm ? std::max<int64_t>(n, 1) : cb.pair_cost());
    for_tasks(L.nA * L.S, work, [&](int64_t t) {
        const int64_t a = t / L.S, s = t % L.S;
        const int64_t i0 = a * L.qa, i1 = std::min(ma, i0 + L.qa), s0 = std::min(mb, s * L.seg),
                      s1 = std::min(mb, s0 + L.seg), rows = i1 - i0;
        Found& fo = found[static_cast<size_t>(t)];
        fo.cnt.assign(static_cast<size_t>(rows), 0);
        std::vector<std::vector<Nb>> lists(count_only ? 0 : static_cast<size_t>(rows));
        auto hit = [&](int64_t r, int64_t j, double d) {
            ++fo.cnt[static_cast<size_t>(r)];
            if (!count_only) lists[static_cast<size_t>(r)].push_back({d, j});
        };
        if (gemm) {
            const double* B = cb.b_f64();
            std::vector<int64_t> idx(static_cast<size_t>(hy.bb));
            gemm_task(
                cb, hy, i0, i1, s0, s1,
                [&](int64_t r, int64_t j0, int64_t wd, const double* g, double nx, double rx, const Fast& fast,
                    const double* x) {
                    const double rr = R[i0 + r];
                    auto test = [&](int64_t q) {
                        const double d = pair_d(x, B + (j0 + q) * n, n, hy.sq);
                        if (d <= rr) hit(r, j0 + q, d);
                    };
                    if (!g) {
                        for (int64_t q = 0; q < wd; ++q) test(q);
                        return;
                    }
                    const double *ny = hy.nb.data() + j0, *ry = hy.rb.data() + j0;
                    const int64_t cnt = survivors(g, ny, wd, nx, fast, fast.tau(rr), idx.data());
                    for (int64_t e = 0; e < cnt; ++e) {
                        const int64_t q = idx[static_cast<size_t>(e)];
                        if (lower_one(g[q], ny[q], ry[q], nx, rx, hy.K, hy.sq) > rr) continue;  // L~ <= d^: d^ > r
                        test(q);
                    }
                }
            );
        } else {
            const int64_t bb = direct_block(rows, n);
            std::vector<double> buf(static_cast<size_t>(rows * std::min(bb, std::max<int64_t>(s1 - s0, 1))));
            for (int64_t j0 = s0; j0 < s1; j0 += bb) {
                const int64_t j1 = std::min(s1, j0 + bb), wd = j1 - j0;
                cb.block(i0, i1, j0, j1, buf.data());
                for (int64_t r = 0; r < rows; ++r) {
                    const double rr = R[i0 + r];
                    const double* row = buf.data() + r * wd;
                    for (int64_t q = 0; q < wd; ++q)
                        if (row[q] <= rr) hit(r, j0 + q, row[q]);
                }
            }
        }
        if (!count_only) {
            int64_t total = 0;
            for (int64_t c : fo.cnt) total += c;
            fo.idx.reserve(static_cast<size_t>(total));
            fo.dist.reserve(static_cast<size_t>(total));
            for (const auto& l : lists)
                for (const Nb& e : l) {
                    fo.idx.push_back(e.j);
                    fo.dist.push_back(e.d);
                }
        }
    });
    // row counts: the segments of a chunk in order
    std::vector<int64_t> indptr(static_cast<size_t>(ma + 1), 0);
    for (int64_t a = 0; a < L.nA; ++a)
        for (int64_t s = 0; s < L.S; ++s) {
            const Found& fo = found[static_cast<size_t>(a * L.S + s)];
            for (size_t r = 0; r < fo.cnt.size(); ++r) indptr[static_cast<size_t>(a * L.qa) + r + 1] += fo.cnt[r];
        }
    if (count_only) {
        nb::object oc = new_array_of(DType::Int64, {ma});
        int64_t* C = reinterpret_cast<int64_t*>(as_ndarray(oc).data);
        for (int64_t i = 0; i < ma; ++i) C[i] = indptr[static_cast<size_t>(i + 1)];
        return oc;
    }
    for (int64_t i = 0; i < ma; ++i) indptr[static_cast<size_t>(i + 1)] += indptr[static_cast<size_t>(i)];
    const int64_t nnz = indptr[static_cast<size_t>(ma)];
    nb::object op = new_array_of(DType::Int64, {ma + 1}), oi = new_array_of(DType::Int64, {nnz}),
               od = new_array_of(DType::Float64, {nnz});
    std::copy(indptr.begin(), indptr.end(), reinterpret_cast<int64_t*>(as_ndarray(op).data));
    int64_t* I = reinterpret_cast<int64_t*>(as_ndarray(oi).data);
    double* D = reinterpret_cast<double*>(as_ndarray(od).data);
    for_tasks(L.nA * L.S, 2 * nnz, [&](int64_t t) {
        const int64_t a = t / L.S, s = t % L.S;
        const Found& fo = found[static_cast<size_t>(t)];
        int64_t src = 0;
        for (size_t r = 0; r < fo.cnt.size(); ++r) {
            const int64_t i = a * L.qa + static_cast<int64_t>(r);
            int64_t dst = indptr[static_cast<size_t>(i)];
            for (int64_t s2 = 0; s2 < s; ++s2) dst += found[static_cast<size_t>(a * L.S + s2)].cnt[r];
            std::copy(fo.idx.begin() + src, fo.idx.begin() + src + fo.cnt[r], I + dst);
            std::copy(fo.dist.begin() + src, fo.dist.begin() + src + fo.cnt[r], D + dst);
            src += fo.cnt[r];
        }
    });
    return nb::make_tuple(op, oi, od);
}

}  // namespace

void bind_neighbors(nb::module_& m) {
    m.def(
        "_kneighbors", &kneighbors_py, nb::arg("XA"), nb::arg("XB"), nb::arg("metric"), nb::arg("p"),
        nb::arg("w").none(), nb::arg("aux").none(), nb::arg("k"), nb::arg("upper"), nb::arg("path")
    );
    m.def(
        "_radius_neighbors", &radius_py, nb::arg("XA"), nb::arg("XB"), nb::arg("metric"), nb::arg("p"),
        nb::arg("w").none(), nb::arg("aux").none(), nb::arg("r"), nb::arg("count_only"), nb::arg("path")
    );
}

}  // namespace sd::spatial
