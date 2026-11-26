// k-means and its kin (docs/design/cluster.md §2.3, §3.1, §3.3, §3.10, §4.2; decisions CL10, CL23-CL25, CL41-CL43,
// CL48, CL49). The rows arrive in key order (CL7: the rank of the lexicographic order), so every tie rule below that
// says "the smaller point" means the smaller key, and the same points in another order give the same result.
//
// The definition is Lloyd's iteration on spatial's sqeuclidean (partition.hpp: sqd, the difference form) of the data
// scaled by CL43's power of two. Every path below computes it, so they all give the same labels and the same bits:
//
//   scans      a point's nearest centre: the smallest (sqd, centre number). direct: spatial's tile kernel (4 points
//              against panels of 8 centres; a lane has pair_scalar's bits). filtered (CL49): the expanded form
//              ||x||^2 - 2 x.c + ||c||^2 (Accelerate's dgemm for x.c) with a proven error bound rules out the
//              centres it shows to be farther than another, and the difference form decides among the rest
//              (§4.2 proves that a ruled-out centre's sqd is larger than another's, whatever order dgemm sums in).
//   bounds     Hamerly's (2010) upper bound of the own centre's distance and lower bound of the others', kept on
//              true distances with CL42's margin (partition.hpp Margin); a point whose bounds clear the margin
//              keeps its label without a scan (§4.2: its scan would give it again).
//   centres    the weighted means of the labels: chunk sums of 4096 points in key order met by the pairwise tree
//              in chunk order (CL48, common.hpp), then one division a coordinate.
//
// _core.cluster:
//   kmeans(X, w, k, init, C0, n_init, max_iter, tol, key0, key1, path)
//       -> (labels (n,), centres (k, p), scaled centres (k, p), e, inertia, the clusters' inertias (k,), n_iter,
//           converged, best_init, relocated)
//       X: the rows in key order; w: weights in that order or None; init 0 'k-means++', 1 'random', 2 the array
//       C0 (k, p). n_init starts (CL41: start r draws element r 2^40 + t of the stream keyed by key0, key1); the
//       best is the smallest (inertia, start). tol: sklearn's relative tolerance (times the mean of the columns'
//       variances). path 0 by shape; 1 Lloyd direct, 2 Lloyd filtered, 3 bounds direct, 4 bounds filtered (the
//       tests' comparisons). labels are the centres' numbers in the order the start made them.
//   kmeans_predict(X, C, Cs, e, path) -> (labels, sqd)   the nearest centres of new rows (predict)
//   kmeans_distinct(X, w) -> int                          distinct rows of positive weight (rows in key order)
//   first_seen(labels, order, k) -> (labels, numbers)     key-order labels to the input's order, renumbered (CL6)
//   kmeans_plusplus(X, order, k, key0, key1, r) -> (k,)   the key ranks k-means++ picks (start r; the mixtures' start)
//   minibatch_kmeans(...), bisecting_kmeans(...)          below
//   _kmeans_assign_expanded(X, C) -> labels               Accelerate's expanded form alone (measurements only)
#include "cluster/cluster.hpp"
#include "cluster/partition.hpp"
#include "parallel/pool.hpp"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sd::cluster {

namespace {

constexpr int64_t kTask = 4096;  // points of an assignment task
constexpr int kPanel = spatial::kPanel;

// ||x||^2 for the filter: four running sums (the filter's bound holds for any order of the sum, §4.2)
inline double norm2_any(const double* x, int64_t p) {
    float64x2_t a = vdupq_n_f64(0.0), b = a;
    int64_t d = 0;
    for (; d + 4 <= p; d += 4) {
        const float64x2_t u = vld1q_f64(x + d), v = vld1q_f64(x + d + 2);
        a = vfmaq_f64(a, u, u), b = vfmaq_f64(b, v, v);
    }
    double s = vaddvq_f64(vaddq_f64(a, b));
    for (; d < p; ++d) s += x[d] * x[d];
    return s;
}

// ---- the centres as the scans read them ----------------------------------------------------------------------

struct Centres {
    int64_t k = 0, p = 0;
    const double* C = nullptr;
    std::vector<double> panels;  // full panels of 8 centres, then the tail's panel of tw (scan_direct)
    int64_t full = 0, tw = 0;    // full panels, the tail's width (0, 2, 4 or 8 centres)
    std::vector<double> norm2;   // computed ||c||^2 (filter)
    std::vector<double> radius;  // >= ||c||^2 (filter)
    std::vector<double> half;    // <= half the true distance to the nearest other centre (bounds)

    void set(const double* c, int64_t kk, int64_t pp, bool filter, bool bounds, const Margin& M) {
        C = c, k = kk, p = pp;
        full = k / kPanel;
        const int64_t t = k - full * kPanel;
        tw = t == 0 ? 0 : t <= 2 ? 2 : t <= 4 ? 4 : kPanel;
        panels.assign(static_cast<size_t>((full * kPanel + tw) * p), 0.0);
        for (int64_t b = 0; b < full; ++b)
            spatial::pack_panel(C + b * kPanel * p, p, kPanel, p, panels.data() + b * p * kPanel);
        double* tp = panels.data() + full * kPanel * p;  // the tail: panel[d * tw + c] = C[full * 8 + c][d]
        for (int64_t d = 0; d < p; ++d)
            for (int64_t c = 0; c < tw; ++c) tp[d * tw + c] = c < t ? C[(full * kPanel + c) * p + d] : 0.0;
        if (filter) {
            norm2.resize(static_cast<size_t>(k));
            radius.resize(static_cast<size_t>(k));
            for (int64_t j = 0; j < k; ++j) {
                const double s = norm2_any(C + j * p, p);
                norm2[static_cast<size_t>(j)] = s;
                radius[static_cast<size_t>(j)] = up2(s * (1.0 + M.g));
            }
        }
        if (bounds) {
            half.assign(static_cast<size_t>(k), kInfD);
            for (int64_t a = 0; a < k; ++a)
                for (int64_t b = a + 1; b < k; ++b) {
                    const double h = 0.5 * M.down(sqd(C + a * p, C + b * p, p));
                    half[static_cast<size_t>(a)] = std::min(half[static_cast<size_t>(a)], h);
                    half[static_cast<size_t>(b)] = std::min(half[static_cast<size_t>(b)], h);
                }
        }
    }
};

// The data of one fit: the rows (scaled, key order), weights, the margin and the filter's norms.
struct Data {
    const double* X = nullptr;
    const double* w = nullptr;  // null: every weight 1
    int64_t n = 0, p = 0;
    Margin M{1};
    bool filter = false;
    std::vector<double> xn2, xrad;  // computed ||x||^2 and an upper bound of it (filter)

    double weight(int64_t i) const { return w ? w[i] : 1.0; }

    void prepare_filter() {
        if (!filter || !xn2.empty()) return;
        xn2.resize(static_cast<size_t>(n));
        xrad.resize(static_cast<size_t>(n));
        auto body = [&](int64_t t) {
            for (int64_t i = t * kTask; i < std::min(n, (t + 1) * kTask); ++i) {
                const double s = norm2_any(X + i * p, p);
                xn2[static_cast<size_t>(i)] = s;
                xrad[static_cast<size_t>(i)] = up2(s * (1.0 + M.g));
            }
        };
        const int64_t nt = (n + kTask - 1) / kTask;
        if (nt > 1 && n * p >= kSplitWork) parallel::parallel_for(nt, body, kTask * p);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    }
};

// The filter pays (CL49) where the scan's products are many: by shape alone (CL95, §5's measurements).
bool filter_shape(int64_t p, int64_t k) { return p >= 10 && p * k >= 400; }

// ---- scans ---------------------------------------------------------------------------------------------------

// A scan's result for one point: its nearest centre and that centre's sqd, and what bounds the true distance to every
// other centre below (lower(); infinity with one centre): b2 the second smallest sqd among the centres scanned by the
// difference form, le the smallest v - E of the centres the filter ruled out (infinity for none). The bound is made
// only where Hamerly's bounds read it (its square roots cost about as much as a small scan).
struct Scan {
    int64_t a;
    double s1, b2, le;
};

// One point's scan by sqd alone (the careful finish on every centre): the slow path below 2^-968.
Scan scan_one(const Data& D, const Centres& Cs, const double* x) {
    double b1 = kInfD, b2 = kInfD;
    int64_t a = 0;
    for (int64_t j = 0; j < Cs.k; ++j) {
        const double v = sqd(x, Cs.C + j * D.p, D.p);
        b2 = std::min(b2, std::max(b1, v));
        a = v < b1 ? j : a;
        b1 = std::min(b1, v);
    }
    return {a, b1, b2, kInfD};
}

// direct (CL95): the difference form in spatial's tile order (each lane one pair's chain: blocks of 32 terms, each a
// sequential fma chain from 0, the block sums added in block order; the bits of pair_scalar and of spatial::tile)
// for R points against a panel of 2C centres (R C = 16 accumulators): groups of 16 points take the full panels of 8
// centres four points at a time and the tail's 2 or 4 centres sixteen or eight points at a time, so no lane is spent
// on a missing centre but in a tail of 5-7. The running smallest, second smallest and its centre are kept lane by
// lane on NEON (strict <: a lane keeps its first centre on a tie) and the two lanes met at the end (the smaller
// value, the smaller centre on a tie): the smallest (sqd, centre) and the second smallest of the scalar scan. A point
// whose two smallest sums are not both above 2^-968 is scanned again by sqd (only sums below it change in the
// careful finish, so above it the lanes' sums are sqd's).
struct Best {
    float64x2_t b1, b2;
    uint64x2_t a;
};

template <int R, int C>
inline void direct_panel(
    const double* const* x, const double* panel, int64_t p, int64_t c0, int64_t valid, Best* best
) {
    constexpr int W = 2 * C;
    double s[R][W];
    for (int64_t b = 0; b < p; b += spatial::kBlock) {
        float64x2_t t[R][C];
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) t[r][c] = vdupq_n_f64(0.0);
        const int64_t e = std::min(p, b + spatial::kBlock);
        for (int64_t i = b; i < e; ++i) {
            float64x2_t yv[C];
            for (int c = 0; c < C; ++c) yv[c] = vld1q_f64(panel + i * W + 2 * c);
            for (int r = 0; r < R; ++r) {
                const float64x2_t xv = vld1q_dup_f64(x[r] + i);
                for (int c = 0; c < C; ++c) {
                    const float64x2_t z = vsubq_f64(xv, yv[c]);
                    t[r][c] = vfmaq_f64(t[r][c], z, z);
                }
            }
        }
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                vst1q_f64(s[r] + 2 * c, b == 0 ? t[r][c] : vaddq_f64(vld1q_f64(s[r] + 2 * c), t[r][c]));
    }
    const float64x2_t inf = vdupq_n_f64(kInfD);
    for (int c = 0; c < C; ++c) {
        const uint64x2_t idx = {static_cast<uint64_t>(c0 + 2 * c), static_cast<uint64_t>(c0 + 2 * c + 1)};
        const uint64x2_t ok = vcltq_u64(
            vsubq_u64(idx, vdupq_n_u64(static_cast<uint64_t>(c0))), vdupq_n_u64(static_cast<uint64_t>(valid))
        );
        for (int r = 0; r < R; ++r) {
            const float64x2_t v = vbslq_f64(ok, vld1q_f64(s[r] + 2 * c), inf);
            Best& B = best[r];
            const uint64x2_t lt = vcltq_f64(v, B.b1);
            B.b2 = vminq_f64(B.b2, vmaxq_f64(B.b1, v));
            B.a = vbslq_u64(lt, idx, B.a);
            B.b1 = vminq_f64(B.b1, v);
        }
    }
}

template <int R>
inline void direct_reset(Best* best) {
    for (int r = 0; r < R; ++r) best[r] = {vdupq_n_f64(kInfD), vdupq_n_f64(kInfD), vdupq_n_u64(0)};
}

// the two lanes of a row met: (smallest, second smallest, its centre)
inline void direct_meet(const Best& B, double& b1, double& b2, int64_t& a) {
    const double v0 = vgetq_lane_f64(B.b1, 0), v1 = vgetq_lane_f64(B.b1, 1);
    const auto a0 = static_cast<int64_t>(vgetq_lane_u64(B.a, 0)), a1 = static_cast<int64_t>(vgetq_lane_u64(B.a, 1));
    const bool second = (v1 < v0) | ((v1 == v0) & (a1 < a0));  // no branch: either lane wins half the time
    b1 = second ? v1 : v0;
    a = second ? a1 : a0;
    b2 = std::min(std::min(vgetq_lane_f64(B.b2, 0), vgetq_lane_f64(B.b2, 1)), std::max(v0, v1));
}

void scan_direct(const Data& D, const Centres& Cs, const int64_t* idx, int64_t m, Scan* out) {
    const int64_t p = D.p, k = Cs.k;
    const int64_t tail = k - Cs.full * kPanel;
    const double* tp = Cs.panels.data() + Cs.full * kPanel * p;
    constexpr int G = 16;
    for (int64_t q = 0; q < m; q += G) {
        const int R = static_cast<int>(std::min<int64_t>(G, m - q));
        const double* xr[G];
        for (int r = 0; r < G; ++r) xr[r] = D.X + idx[q + std::min(r, R - 1)] * p;
        Best best[G];
        direct_reset<G>(best);
        for (int g = 0; g < R; g += 4)
            for (int64_t b = 0; b < Cs.full; ++b)
                direct_panel<4, 4>(xr + g, Cs.panels.data() + b * p * kPanel, p, b * kPanel, kPanel, best + g);
        if (Cs.tw == 2) direct_panel<16, 1>(xr, tp, p, Cs.full * kPanel, tail, best);
        else if (Cs.tw == 4)
            for (int g = 0; g < R; g += 8) direct_panel<8, 2>(xr + g, tp, p, Cs.full * kPanel, tail, best + g);
        else if (Cs.tw == kPanel)
            for (int g = 0; g < R; g += 4) direct_panel<4, 4>(xr + g, tp, p, Cs.full * kPanel, tail, best + g);
        for (int r = 0; r < R; ++r) {
            double b1, b2;
            int64_t a;
            direct_meet(best[r], b1, b2, a);
            if (b1 >= spatial::kTiny && b2 >= spatial::kTiny) out[q + r] = {a, b1, b2, kInfD};
            else out[q + r] = scan_one(D, Cs, xr[r]);
        }
    }
}

// a lower bound of the true d^2's square root from A, a computed v - E of a ruled-out centre (§4.2)
inline double lower_from_expanded(double A) {
    return A > 0x1p-950 ? down2(std::sqrt(A * (1.0 - 4.0 * kU)) * (1.0 - 2.0 * kU)) : 0.0;
}

// The lower bound of a scan (k > 1): from b2 by the difference form's bound and from le by the expanded form's; a
// missing one (infinity) adds nothing (down(inf) and lower_from_expanded(inf) are both the double below the largest,
// above any bound from a finite value: the bits of min(lower_from_expanded(le), down(b2))).
inline double scan_lower(const Margin& M, const Scan& r, int64_t k) {
    if (k <= 1) return kInfD;
    const double a = r.b2 < kInfD ? M.down(r.b2) : kInfD, b = r.le < kInfD ? lower_from_expanded(r.le) : kInfD;
    return a < kInfD || b < kInfD ? std::min(a, b) : M.down(kInfD);
}

// filtered (CL49): the expanded form rules out centres, the difference form decides among the rest. The expanded
// value v = (||x||^2 - 2 x.c) + ||c||^2 (the norms and x.c summed in any order: dgemm) is within
// E = cE (||x||^2 + ||c||^2) of the true d^2 (cE = 2 (2 gamma(p + 2)) with room for E's own roundings); a centre
// j with v_j - E_j > (v_b + E_b) kappa + tau is farther than b by more than sqd's error, so
// its sqd is larger than b's (§4.2) and it cannot be the nearest; b is taken as the centre of smallest v + E, which
// is never ruled out. tau absorbs the underflows' absolute errors. Both passes run on NEON, two centres a step.
void scan_filtered(
    const Data& D,
    const Centres& Cs,
    const int64_t* idx,
    int64_t m,
    Scan* out,
    std::vector<double>& Xg,
    std::vector<double>& G
) {
    const int64_t p = D.p, k = Cs.k;
    const double* rows = D.X + idx[0] * p;
    bool contiguous = true;
    for (int64_t q = 1; q < m && contiguous; ++q) contiguous = idx[q] == idx[0] + q;
    if (!contiguous) {
        Xg.resize(static_cast<size_t>(m * p));
        for (int64_t q = 0; q < m; ++q) std::copy(D.X + idx[q] * p, D.X + (idx[q] + 1) * p, Xg.begin() + q * p);
        rows = Xg.data();
    }
    G.resize(static_cast<size_t>(m * k));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, p, 1.0, rows, p, Cs.C, p, 0.0, G.data(), k);
    const double cE = 8.0 * static_cast<double>(p + 4) * kU;
    const double kappa = 1.0 + 8.0 * static_cast<double>(p + 4) * kU;
    const double tau = 0x1p-960;
    const double* nc = Cs.norm2.data();
    const double* rc = Cs.radius.data();  // >= ||c||^2
    const int64_t k2 = k & ~int64_t{1};
    const float64x2_t cEv = vdupq_n_f64(cE), infv = vdupq_n_f64(kInfD);
    struct Cand {
        int64_t q, j;
    };
    std::vector<Cand> cand;
    cand.reserve(static_cast<size_t>(2 * m));
    std::vector<double> lows(static_cast<size_t>(m));
    for (int64_t q = 0; q < m; ++q) {
        const int64_t i = idx[q];
        const double* x = D.X + i * p;
        double* v = G.data() + q * k;  // x.c, then A = v - E
        const double nx = D.xn2[static_cast<size_t>(i)], rx = D.xrad[static_cast<size_t>(i)];
        // pass 1: v = (||x||^2 - 2 x.c) + ||c||^2, E = cE (rx + rc), A = v - E: A is stored over x.c, and the
        // smallest v + E, the smallest A with its centre and the second smallest A are kept lane by lane
        const float64x2_t nxv = vdupq_n_f64(nx), rxv = vdupq_n_f64(rx);
        float64x2_t umin = infv, a1 = infv, a2 = infv;
        uint64x2_t aj = vdupq_n_u64(0), jv = {0, 1};
        const uint64x2_t two = vdupq_n_u64(2);
        for (int64_t j = 0; j < k2; j += 2, jv = vaddq_u64(jv, two)) {
            const float64x2_t g = vld1q_f64(v + j);
            const float64x2_t vv = vaddq_f64(vsubq_f64(nxv, vaddq_f64(g, g)), vld1q_f64(nc + j));
            const float64x2_t e = vmulq_f64(cEv, vaddq_f64(rxv, vld1q_f64(rc + j)));
            umin = vminq_f64(umin, vaddq_f64(vv, e));
            const float64x2_t A = vsubq_f64(vv, e);
            vst1q_f64(v + j, A);
            const uint64x2_t lt = vcltq_f64(A, a1);
            a2 = vminq_f64(a2, vmaxq_f64(a1, A));
            aj = vbslq_u64(lt, jv, aj);
            a1 = vminq_f64(a1, A);
        }
        double um = vminvq_f64(umin);
        double A1, A2;
        int64_t J1;
        {
            const double l0 = vgetq_lane_f64(a1, 0), l1 = vgetq_lane_f64(a1, 1);
            const auto j0 = static_cast<int64_t>(vgetq_lane_u64(aj, 0)),
                       j1 = static_cast<int64_t>(vgetq_lane_u64(aj, 1));
            const bool second = (l1 < l0) | ((l1 == l0) & (j1 < j0));
            A1 = second ? l1 : l0, J1 = second ? j1 : j0;
            A2 = std::min(std::min(vgetq_lane_f64(a2, 0), vgetq_lane_f64(a2, 1)), std::max(l0, l1));
        }
        for (int64_t j = k2; j < k; ++j) {
            const double vv = (nx - (v[j] + v[j])) + nc[j], e = cE * (rx + rc[j]);
            um = std::min(um, vv + e);
            const double A = vv - e;
            v[j] = A;
            A2 = std::min(A2, std::max(A1, A));
            J1 = A < A1 ? j : J1;
            A1 = std::min(A1, A);
        }
        // a centre with A > lim is ruled out (the smallest A among them bounds their distances below). The usual
        // case: the second smallest A is above lim, so the one kept centre is that of the smallest A (it is the
        // centre of smallest v + E or has its A at most that) and the ruled-out centres' smallest A is the second
        // smallest; otherwise pass 2 walks the stored A.
        const double lim = um * kappa + tau;
        double lo;
        if (A2 > lim) {
            cand.push_back({q, J1});
            lo = A2;
        } else {
            lo = kInfD;
            for (int64_t j = 0; j < k; ++j) {
                const double A = v[j];
                if (A > lim) lo = std::min(lo, A);
                else cand.push_back({q, j});
            }
        }
        lows[static_cast<size_t>(q)] = lo;
    }
    // the kept centres by the difference form, eight pairs side by side (sqdN: sqd's bits), then each point's
    // smallest (sqd, centre) and second smallest in centre order
    const size_t nc_ = cand.size();
    std::vector<double> sv(nc_);
    for (size_t c = 0; c < nc_; c += 8) {
        const double* xr[8];
        const double* cr[8];
        for (size_t r = 0; r < 8; ++r) {
            const Cand& cc = cand[std::min(c + r, nc_ - 1)];
            xr[r] = D.X + idx[cc.q] * p;
            cr[r] = Cs.C + cc.j * p;
        }
        double o8[8];
        sqdN<8>(xr, cr, p, o8);
        for (size_t r = 0; r < 8 && c + r < nc_; ++r) sv[c + r] = o8[r];
    }
    size_t c = 0;
    for (int64_t q = 0; q < m; ++q) {
        double b1 = kInfD, b2 = kInfD;
        int64_t a = 0;
        for (; c < nc_ && cand[c].q == q; ++c) {
            const double s = sv[c];
            b2 = std::min(b2, std::max(b1, s));
            a = s < b1 ? cand[c].j : a;
            b1 = std::min(b1, s);
        }
        out[q] = {a, b1, b2, lows[static_cast<size_t>(q)]};
    }
}

struct ScanBuffers {
    std::vector<int64_t> idx;
    std::vector<Scan> res;
    std::vector<double> Xg, G;
};

void scan(const Data& D, const Centres& Cs, bool filter, ScanBuffers& B) {
    const int64_t m = static_cast<int64_t>(B.idx.size());
    B.res.resize(static_cast<size_t>(m));
    if (m == 0) return;
    if (filter) scan_filtered(D, Cs, B.idx.data(), m, B.res.data(), B.Xg, B.G);
    else scan_direct(D, Cs, B.idx.data(), m, B.res.data());
}

// ---- Lloyd's iteration -----------------------------------------------------------------------------------------

struct Fit {
    std::vector<int64_t> lab;
    std::vector<double> C;
    std::vector<double> per;  // each cluster's inertia
    double inertia = 0.0;
    int64_t n_iter = 0;
    bool converged = false;
};

// The bounds of one fit (Hamerly): U >= the own centre's true distance, L <= every other's.
struct Bounds {
    std::vector<double> U, L;
    std::vector<double> delta;  // >= each centre's last move
    double d1 = 0.0, d2 = 0.0;  // the largest delta and the largest but its centre's
    int64_t a1 = -1;

    void moves(const Margin& M, const double* Cold, const double* Cnew, int64_t k, int64_t p) {
        delta.resize(static_cast<size_t>(k));
        d1 = d2 = 0.0, a1 = -1;
        for (int64_t j = 0; j < k; ++j) {
            const double s = sqd(Cold + j * p, Cnew + j * p, p);
            const double dj = s == 0.0 ? 0.0 : M.up(s);
            delta[static_cast<size_t>(j)] = dj;
            if (dj > d1) d2 = d1, d1 = dj, a1 = j;
            else if (dj > d2) d2 = dj;
        }
    }
};

// Assigns every point to its nearest centre (the bounds skip the points they prove); true when a label changed.
bool assign(const Data& D, const Centres& Cs, std::vector<int64_t>& lab, Bounds* Bd, bool fresh, bool filter) {
    const int64_t n = D.n, p = D.p;
    const int64_t nt = (n + kTask - 1) / kTask;
    std::vector<char> changed(static_cast<size_t>(nt), 0);
    auto body = [&](int64_t t) {
        ScanBuffers B;
        const int64_t i0 = t * kTask, i1 = std::min(n, (t + 1) * kTask);
        B.idx.reserve(static_cast<size_t>(i1 - i0));
        for (int64_t i = i0; i < i1; ++i) {
            if (Bd && !fresh) {
                const auto ii = static_cast<size_t>(i);
                const int64_t a = lab[ii];
                double U = next_up(Bd->U[ii] + Bd->delta[static_cast<size_t>(a)]);
                const double L = next_down(Bd->L[ii] - (a == Bd->a1 ? Bd->d2 : Bd->d1));
                Bd->L[ii] = L;
                const double z = std::max(L, Cs.half[static_cast<size_t>(a)]);
                if (D.M.clear(z, U)) {
                    Bd->U[ii] = U;
                    continue;
                }
                U = D.M.up(sqd(D.X + i * p, Cs.C + a * p, p));
                Bd->U[ii] = U;
                if (D.M.clear(z, U)) continue;
            }
            B.idx.push_back(i);
        }
        scan(D, Cs, filter, B);
        char ch = 0;
        for (size_t q = 0; q < B.idx.size(); ++q) {
            const auto ii = static_cast<size_t>(B.idx[q]);
            const Scan& r = B.res[q];
            ch |= static_cast<char>(r.a != lab[ii]);
            lab[ii] = r.a;
            if (Bd) Bd->U[ii] = D.M.up(r.s1), Bd->L[ii] = scan_lower(D.M, r, Cs.k);
        }
        changed[static_cast<size_t>(t)] = ch;
    };
    if (nt > 1 && n * p * Cs.k >= kSplitWork) parallel::parallel_for(nt, body, kTask * p * Cs.k);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    return std::any_of(changed.begin(), changed.end(), [](char c) { return c != 0; });
}

// The weighted sums and weights of the labels (CL48): S (k, p) and W (k)
void label_sums(const Data& D, const int64_t* lab, int64_t k, std::vector<double>& S, std::vector<double>& W) {
    const int64_t n = D.n, p = D.p, width = k * p + k;
    std::vector<double> tot(static_cast<size_t>(width));
    pairwise_chunks(
        chunks_of(n), width,
        [&](int64_t c, double* out) {
            std::fill(out, out + width, 0.0);
            const int64_t i1 = std::min(n, (c + 1) * kChunk);
            if (!D.w) {  // weight 1: w x is x exactly
                for (int64_t i = c * kChunk; i < i1; ++i) {
                    const int64_t j = lab[i];
                    out[k * p + j] += 1.0;
                    const double* x = D.X + i * p;
                    double* s = out + j * p;
                    for (int64_t d = 0; d < p; ++d) s[d] += x[d];
                }
                return;
            }
            for (int64_t i = c * kChunk; i < i1; ++i) {
                const int64_t j = lab[i];
                const double wi = D.w[i];
                out[k * p + j] += wi;
                const double* x = D.X + i * p;
                double* s = out + j * p;
                for (int64_t d = 0; d < p; ++d) s[d] += wi * x[d];
            }
        },
        tot.data(), kChunk * p
    );
    S.assign(tot.begin(), tot.begin() + k * p);
    W.assign(tot.begin() + k * p, tot.end());
}

// sum_i w_i sqd(x_i, c_lab(i)) by CL48's chunks, and the same sum of each cluster's points (per, k values)
double inertia_of(const Data& D, const int64_t* lab, const double* C, int64_t k, std::vector<double>* per = nullptr) {
    const int64_t n = D.n, p = D.p, width = k + 1;
    std::vector<double> tot(static_cast<size_t>(width));
    pairwise_chunks(
        chunks_of(n), width,
        [&](int64_t c, double* out) {
            std::fill(out, out + width, 0.0);
            const int64_t i0 = c * kChunk, i1 = std::min(n, (c + 1) * kChunk);
            std::vector<double> sq(static_cast<size_t>(i1 - i0));
            sqd_rows(D.X, p, i0, i1, C, lab, sq.data());
            for (int64_t i = i0; i < i1; ++i) {
                const double v = D.weight(i) * sq[static_cast<size_t>(i - i0)];
                out[0] += v;
                out[1 + lab[i]] += v;
            }
        },
        tot.data(), kChunk * p
    );
    if (per) per->assign(tot.begin() + 1, tot.end());
    return tot[0];
}

// The empty clusters (no weight) take the points farthest from their centres (sqd to the centre of this
// assignment, the larger first, the smaller key on a tie), among the points of positive weight: scikit-learn's
// relocation (_relocate_empty_clusters_dense). Returns the number moved; the moved points' bounds are reset.
int64_t relocate(
    const Data& D, const double* C, int64_t k, std::vector<int64_t>& lab, const std::vector<double>& W, Bounds* Bd
) {
    std::vector<int64_t> empty;
    for (int64_t j = 0; j < k; ++j)
        if (!(W[static_cast<size_t>(j)] > 0.0)) empty.push_back(j);
    if (empty.empty()) return 0;
    const int64_t n = D.n, p = D.p;
    std::vector<double> s(static_cast<size_t>(n));
    auto body = [&](int64_t t) {
        const int64_t i0 = t * kTask, i1 = std::min(n, (t + 1) * kTask);
        sqd_rows(D.X, p, i0, i1, C, lab.data(), s.data() + i0);
        for (int64_t i = i0; i < i1; ++i)
            if (!(D.weight(i) > 0.0)) s[static_cast<size_t>(i)] = -1.0;
    };
    const int64_t nt = (n + kTask - 1) / kTask;
    if (nt > 1 && n * p >= kSplitWork) parallel::parallel_for(nt, body, kTask * p);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    const auto m = std::min<size_t>(empty.size(), static_cast<size_t>(n));
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(m), order.end(), [&](int64_t a, int64_t b) {
            const double x = s[static_cast<size_t>(a)], y = s[static_cast<size_t>(b)];
            return x != y ? x > y : a < b;
        }
    );
    int64_t moved = 0;
    for (size_t q = 0; q < m; ++q) {
        const int64_t i = order[q];
        if (s[static_cast<size_t>(i)] < 0.0) break;  // no point of positive weight left
        lab[static_cast<size_t>(i)] = empty[q];
        if (Bd) Bd->U[static_cast<size_t>(i)] = kInfD, Bd->L[static_cast<size_t>(i)] = 0.0;
        ++moved;
    }
    return moved;
}

// Lloyd's iteration from C (k, p) as scikit-learn's _kmeans_single_lloyd runs it: assign, relocate the empty
// clusters, the new centres; stop when no label changed (strict) or the centres moved by at most tol_abs in
// all (sum of the sqd of the moves); after a stop that is not strict the labels are made again from the last
// centres. bounds: Hamerly's; filter: the filtered scans.
Fit lloyd(
    Data& D,
    std::vector<double> C,
    int64_t k,
    int64_t max_iter,
    double tol_abs,
    bool bounds,
    bool filter,
    int64_t* relocated = nullptr
) {
    const int64_t n = D.n, p = D.p;
    Fit F;
    F.lab.assign(static_cast<size_t>(n), -1);
    if (filter) D.prepare_filter();
    Bounds Bd;
    if (bounds) Bd.U.assign(static_cast<size_t>(n), kInfD), Bd.L.assign(static_cast<size_t>(n), 0.0);
    Centres Cs;
    std::vector<double> Cnew(static_cast<size_t>(k * p)), S, W;
    bool fresh = true, strict = false;
    int64_t it = 0;
    for (it = 1; it <= max_iter; ++it) {
        Cs.set(C.data(), k, p, filter, bounds, D.M);
        bool changed = assign(D, Cs, F.lab, bounds ? &Bd : nullptr, fresh, filter) || fresh;
        fresh = false;
        label_sums(D, F.lab.data(), k, S, W);
        const int64_t moved = relocate(D, C.data(), k, F.lab, W, bounds ? &Bd : nullptr);
        if (moved) {
            changed = true;
            if (relocated) *relocated += moved;
            label_sums(D, F.lab.data(), k, S, W);
        }
        for (int64_t j = 0; j < k; ++j) {
            const double wj = W[static_cast<size_t>(j)];
            for (int64_t d = 0; d < p; ++d)
                Cnew[static_cast<size_t>(j * p + d)] =
                    wj > 0.0 ? S[static_cast<size_t>(j * p + d)] / wj : C[static_cast<size_t>(j * p + d)];
        }
        double shift = 0.0;
        for (int64_t j = 0; j < k; ++j) shift += sqd(C.data() + j * p, Cnew.data() + j * p, p);
        if (bounds) Bd.moves(D.M, C.data(), Cnew.data(), k, p);
        C.swap(Cnew);
        if (!changed) {
            strict = true;
            break;
        }
        if (shift <= tol_abs) break;
    }
    F.converged = it <= max_iter;
    F.n_iter = std::min(it, max_iter);
    if (!strict) {
        Cs.set(C.data(), k, p, filter, bounds, D.M);
        assign(D, Cs, F.lab, bounds ? &Bd : nullptr, false, filter);
    }
    F.C = std::move(C);
    F.inertia = inertia_of(D, F.lab.data(), F.C.data(), k, &F.per);
    return F;
}

// ---- starts ----------------------------------------------------------------------------------------------------

// sqd of every point to centre c (the point's row): out[i]
void sqd_all(const Data& D, const double* c, double* out) {
    const int64_t n = D.n, p = D.p;
    auto body = [&](int64_t t) {
        const int64_t i0 = t * kTask, i1 = std::min(n, (t + 1) * kTask);
        sqd_rows(D.X, p, i0, i1, c, nullptr, out + i0);
    };
    const int64_t nt = (n + kTask - 1) / kTask;
    if (nt > 1 && n * p >= kSplitWork) parallel::parallel_for(nt, body, kTask * p);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
}

// scikit-learn's greedy k-means++ (_kmeans_plusplus: 2 + floor(ln k) candidates a step, each drawn with
// probability w D^2 by the running sums of CL48's order, the one of smallest potential sum w min(D^2, d^2) kept,
// the smaller candidate number on a tie). Start r draws element r 2^40 + t: t = 0 the first centre, then
// t = 1 + (c - 1) L + q for candidate q of centre c.
std::vector<double> plusplus(
    const Data& D, int64_t k, random::Key key, std::uint64_t r, std::vector<int64_t>* picked = nullptr
) {
    const int64_t n = D.n, p = D.p;
    random::Stream st(key);
    std::vector<double> C(static_cast<size_t>(k * p)), Dm(static_cast<size_t>(n)), P(static_cast<size_t>(n)),
        v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = D.weight(i);
    running_sums(v.data(), n, P.data());
    int64_t first = pick(P.data(), n, uniform(st, element(r, 0)));
    if (picked) picked->assign(1, first);
    std::copy(D.X + first * p, D.X + (first + 1) * p, C.begin());
    sqd_all(D, D.X + first * p, Dm.data());
    const int64_t L = 2 + static_cast<int64_t>(std::log(static_cast<double>(k)));
    std::vector<int64_t> cand(static_cast<size_t>(L));
    std::vector<double> pots(static_cast<size_t>(L));
    for (int64_t c = 1; c < k; ++c) {
        for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = D.weight(i) * Dm[static_cast<size_t>(i)];
        running_sums(v.data(), n, P.data());
        for (int64_t q = 0; q < L; ++q)
            cand[static_cast<size_t>(q)] =
                pick(P.data(), n, uniform(st, element(r, static_cast<std::uint64_t>(1 + (c - 1) * L + q))));
        // the candidates' potentials: CL48's chunk sums of w min(D^2, sqd(x, candidate))
        const int64_t np = (L + kPanel - 1) / kPanel;
        std::vector<double> panels(static_cast<size_t>(np * p * kPanel), 0.0), rows(static_cast<size_t>(L * p));
        for (int64_t q = 0; q < L; ++q)
            std::copy(
                D.X + cand[static_cast<size_t>(q)] * p, D.X + (cand[static_cast<size_t>(q)] + 1) * p,
                rows.begin() + q * p
            );
        for (int64_t b = 0; b < np; ++b)
            spatial::pack_panel(
                rows.data() + b * kPanel * p, p, std::min<int64_t>(kPanel, L - b * kPanel), p,
                panels.data() + b * p * kPanel
            );
        pairwise_chunks(
            chunks_of(n), L,
            [&](int64_t ch, double* out) {
                std::fill(out, out + L, 0.0);
                double buf[4 * kPanel];
                const int64_t i0 = ch * kChunk, i1 = std::min(n, (ch + 1) * kChunk);
                for (int64_t i = i0; i < i1; i += 4) {
                    const int R = static_cast<int>(std::min<int64_t>(4, i1 - i));
                    const double* xr[4];
                    for (int rr = 0; rr < 4; ++rr) xr[rr] = D.X + (i + std::min(rr, R - 1)) * p;
                    for (int64_t b = 0; b < np; ++b) {
                        spatial::tile<spatial::SqOp, 4, false>(xr, panels.data() + b * p * kPanel, nullptr, p, buf);
                        const int64_t cmax = std::min<int64_t>(kPanel, L - b * kPanel);
                        for (int rr = 0; rr < R; ++rr)
                            for (int64_t cc = 0; cc < cmax; ++cc) {
                                const int64_t q = b * kPanel + cc;
                                double s = buf[rr * kPanel + cc];
                                if (!(s >= spatial::kTiny))
                                    s = spatial::sq_finish<false>(s, xr[rr], rows.data() + q * p, nullptr, p);
                                const double dm = Dm[static_cast<size_t>(i + rr)];
                                out[q] += D.weight(i + rr) * (s < dm ? s : dm);
                            }
                    }
                }
            },
            pots.data(), kChunk * p * L
        );
        int64_t best = 0;
        for (int64_t q = 1; q < L; ++q)
            if (pots[static_cast<size_t>(q)] < pots[static_cast<size_t>(best)]) best = q;
        const int64_t pick_i = cand[static_cast<size_t>(best)];
        if (picked) picked->push_back(pick_i);
        std::copy(D.X + pick_i * p, D.X + (pick_i + 1) * p, C.begin() + c * p);
        std::vector<double> dn(static_cast<size_t>(n));
        sqd_all(D, D.X + pick_i * p, dn.data());
        for (int64_t i = 0; i < n; ++i)
            Dm[static_cast<size_t>(i)] = std::min(Dm[static_cast<size_t>(i)], dn[static_cast<size_t>(i)]);
    }
    return C;
}

// k points drawn without replacement with probability proportional to the weights (successive sampling), by
// Efraimidis and Spirakis's keys: point i's key is log(u_i) / w_i, u_i in (0, 1) from element r 2^40 + i, and the
// k largest (key, then the smaller point on a tie) in decreasing order are the draws in order. Points of weight 0
// are never drawn.
std::vector<int64_t> sample_without_replacement(
    const double* w,
    int64_t n,
    int64_t k,
    random::Key key,
    std::uint64_t r,
    const std::vector<char>* exclude = nullptr,
    int64_t t0 = 0
) {
    std::vector<double> kv(static_cast<size_t>(n));
    auto body = [&](int64_t t) {
        random::Stream st(key);
        for (int64_t i = t * kTask; i < std::min(n, (t + 1) * kTask); ++i) {
            const double u =
                (static_cast<double>(st.first(element(r, static_cast<std::uint64_t>(t0 + i))) >> 11) + 0.5) * 0x1p-53;
            const double wi = w ? w[i] : 1.0;
            const bool out = exclude && (*exclude)[static_cast<size_t>(i)];
            kv[static_cast<size_t>(i)] = (wi > 0.0 && !out) ? std::log(u) / wi : -kInfD;
        }
    };
    const int64_t nt = (n + kTask - 1) / kTask;
    if (nt > 1 && n >= kSplitWork) parallel::parallel_for(nt, body, kTask * 20);
    else
        for (int64_t t = 0; t < nt; ++t) body(t);
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    const auto m = static_cast<std::ptrdiff_t>(std::min(k, n));
    std::partial_sort(order.begin(), order.begin() + m, order.end(), [&](int64_t a, int64_t b) {
        const double x = kv[static_cast<size_t>(a)], y = kv[static_cast<size_t>(b)];
        return x != y ? x > y : a < b;
    });
    order.resize(static_cast<size_t>(m));
    return order;
}

std::vector<double> random_start(const Data& D, int64_t k, random::Key key, std::uint64_t r) {
    const std::vector<int64_t> idx = sample_without_replacement(D.w, D.n, k, key, r);
    std::vector<double> C(static_cast<size_t>(k * D.p));
    for (int64_t j = 0; j < k; ++j)
        std::copy(
            D.X + idx[static_cast<size_t>(j)] * D.p, D.X + (idx[static_cast<size_t>(j)] + 1) * D.p, C.begin() + j * D.p
        );
    return C;
}

// sklearn's tolerance: tol times the mean of the columns' variances (unweighted, divided by n), by CL48's sums
double tolerance(const Data& D, double tol) {
    if (tol == 0.0) return 0.0;
    const int64_t n = D.n, p = D.p;
    std::vector<double> mean(static_cast<size_t>(p)), var(static_cast<size_t>(p));
    pairwise_chunks(
        chunks_of(n), p,
        [&](int64_t c, double* out) {
            std::fill(out, out + p, 0.0);
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i)
                for (int64_t d = 0; d < p; ++d) out[d] += D.X[i * p + d];
        },
        mean.data(), kChunk * p
    );
    for (auto& m : mean) m /= static_cast<double>(n);
    pairwise_chunks(
        chunks_of(n), p,
        [&](int64_t c, double* out) {
            std::fill(out, out + p, 0.0);
            for (int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i)
                for (int64_t d = 0; d < p; ++d) {
                    const double z = D.X[i * p + d] - mean[static_cast<size_t>(d)];
                    out[d] += z * z;
                }
        },
        var.data(), kChunk * p
    );
    double s = 0.0;
    for (int64_t d = 0; d < p; ++d) s += var[static_cast<size_t>(d)] / static_cast<double>(n);
    return s / static_cast<double>(p) * tol;
}

// The path by shape: Hamerly's bounds where a scan costs more than their check (k (p + 4) >= 100: below it the
// check of a point's two bounds, with its roundings, costs about what its scan does; CL95 and the measurements in
// docs/design/cluster.md §5), the filter by filter_shape. Forced paths (tests): 1 Lloyd direct, 2 Lloyd filtered,
// 3 bounds direct, 4 bounds filtered.
bool bounds_shape(int64_t p, int64_t k) { return k >= 2 && k * (p + 4) >= 100; }

void choose_path(int path, int64_t p, int64_t k, bool& bounds, bool& filter) {
    if (path == 0) {
        bounds = bounds_shape(p, k);
        filter = filter_shape(p, k);
        return;
    }
    bounds = path == 3 || path == 4;
    filter = path == 2 || path == 4;
}

// The scaled copy of rows (CL43) and the exponent; rows itself when the exponent is 0.
struct Scaled {
    std::vector<double> copy;
    const double* X = nullptr;
    int e = 0;
    Scaled(const double* X0, int64_t m, int e0) : X(X0), e(e0) {
        if (e != 0) copy = scaled_copy(X0, m, e), X = copy.data();
    }
};

// The rows of a fit in key order: X[order[r]] scaled by 2^e (CL43; e from the rows' and extra's largest |value|)
// and their weights, gathered in tasks.
struct KeyRows {
    std::vector<double> X, w;
    int64_t n = 0, p = 0;
    int e = 0;
    KeyRows(const double* X0, const int64_t* order, const double* w0, int64_t nn, int64_t pp, double extra_max = 0.0)
        : n(nn), p(pp) {
        e = scale_exponent(std::max(max_abs_of(X0, n * p), extra_max));
        X.resize(static_cast<size_t>(n * p));
        if (w0) w.resize(static_cast<size_t>(n));
        const double f = std::ldexp(1.0, std::clamp(e, -1000, 1000));
        auto body = [&](int64_t t) {
            for (int64_t r = t * kTask; r < std::min(n, (t + 1) * kTask); ++r) {
                const double* x = X0 + order[r] * p;
                double* y = X.data() + r * p;
                if (e == 0) std::copy(x, x + p, y);
                else if (e >= -1000 && e <= 1000)
                    for (int64_t d = 0; d < p; ++d) y[d] = x[d] * f;
                else
                    for (int64_t d = 0; d < p; ++d) y[d] = spatial::scaled(x[d], e);
                if (w0) w[static_cast<size_t>(r)] = w0[order[r]];
            }
        };
        const int64_t nt = (n + kTask - 1) / kTask;
        if (nt > 1 && n * p >= kSplitWork) parallel::parallel_for(nt, body, kTask * p);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    }
    Data data() const {
        Data D;
        D.X = X.data(), D.w = w.empty() ? nullptr : w.data(), D.n = n, D.p = p, D.M = Margin(p);
        return D;
    }
};

struct KMeansOut {
    Fit best;
    int64_t best_init = 0;
    int64_t relocated = 0;
};

// n_init starts r0 + r of Lloyd's iteration (the best: the smallest (inertia, start))
KMeansOut kmeans_run(
    Data& D,
    int64_t k,
    int init,
    const double* C0,
    int64_t n_init,
    int64_t max_iter,
    double tol_abs,
    random::Key key,
    int path,
    std::uint64_t r0 = 0
) {
    bool bounds, filter;
    choose_path(path, D.p, k, bounds, filter);
    D.filter = filter;
    KMeansOut out;
    bool have = false;
    for (int64_t r = 0; r < n_init; ++r) {
        std::vector<double> C;
        const std::uint64_t rr = r0 + static_cast<std::uint64_t>(r);
        if (init == 0) C = plusplus(D, k, key, rr);
        else if (init == 1) C = random_start(D, k, key, rr);
        else C.assign(C0, C0 + k * D.p);
        int64_t moved = 0;
        Fit F = lloyd(D, std::move(C), k, max_iter, tol_abs, bounds, filter, &moved);
        if (!have || F.inertia < out.best.inertia) {
            out.best = std::move(F);
            out.best_init = r;
            out.relocated = moved;
            have = true;
        }
    }
    return out;
}

struct Args {
    const double* X = nullptr;
    const int64_t* order = nullptr;
    const double* w = nullptr;
    int64_t n = 0, p = 0;
};

Args fit_args(nb::handle Xo, nb::handle ordo, nb::handle wo) {
    Args A;
    A.X = matrix_arg(Xo, "X", A.n, A.p);
    int64_t m;
    A.order = index_arg(ordo, "order", m);
    if (m != A.n) throw nb::value_error("order: one index a row");
    if (!wo.is_none()) {
        A.w = vector_arg(wo, "w", m);
        if (m != A.n) throw nb::value_error("w: one weight a row");
    }
    return A;
}

// A fit's centres back from the grid (exact but for subnormal results), its inertias, and the tuple's head:
// (labels in key order, centres, the centres on the grid, e, inertia, per-cluster inertia)
nb::object fit_head(const Fit& F, int64_t n, int64_t k, int64_t p, int e) {
    std::vector<double> Cback(F.C.size()), per(F.per);
    for (size_t q = 0; q < Cback.size(); ++q) Cback[q] = e ? spatial::scaled(F.C[q], -e) : F.C[q];
    for (auto& v : per) v = e ? std::ldexp(v, -2 * e) : v;
    const double inertia = e ? std::ldexp(F.inertia, -2 * e) : F.inertia;
    return nb::make_tuple(
        wrap(int_array(F.lab, n)), wrap(float_array(Cback, k, p)), wrap(float_array(F.C, k, p)), e, inertia,
        wrap(float_array(per, k))
    );
}

nb::object kmeans_py(
    nb::handle Xo,
    nb::handle ordo,
    nb::handle wo,
    int64_t k,
    int init,
    nb::handle C0o,
    int64_t n_init,
    int64_t max_iter,
    double tol,
    int64_t key0,
    int64_t key1,
    int path
) {
    const Args A = fit_args(Xo, ordo, wo);
    const int64_t n = A.n, p = A.p;
    const double* C0 = nullptr;
    if (init == 2) {
        int64_t kk, pp;
        C0 = matrix_arg(C0o, "C0", kk, pp);
        if (kk != k || pp != p) throw nb::value_error("C0 must be (k, p)");
    }
    if (k < 1 || k > n) throw nb::value_error("k");
    KMeansOut out;
    int e = 0;
    {
        const parallel::Unlocked unlocked(n * p >= kSplitWork);
        const KeyRows R(A.X, A.order, A.w, n, p, C0 ? max_abs_of(C0, k * p) : 0.0);
        e = R.e;
        std::vector<double> C0s;
        if (C0) C0s = e ? scaled_copy(C0, k * p, e) : std::vector<double>(C0, C0 + k * p);
        Data D = R.data();
        out = kmeans_run(
            D, k, init, C0 ? C0s.data() : nullptr, n_init, max_iter, tolerance(D, tol), key_of(key0, key1), path
        );
    }
    nb::object head = fit_head(out.best, n, k, p, e);
    return nb::make_tuple(head, out.best.n_iter, out.best.converged, out.best_init, out.relocated);
}

}  // namespace

std::vector<int64_t> kmeans_labels(const double* X, int64_t n, int64_t p, int64_t k, random::Key key) {
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    const KeyRows R(X, order.data(), nullptr, n, p);
    Data D = R.data();
    KMeansOut out = kmeans_run(D, k, 0, nullptr, 1, 300, tolerance(D, 1e-4), key, 0);
    return std::move(out.best.lab);
}

namespace {

// ---- mini-batch k-means ------------------------------------------------------------------------------------------

// scikit-learn's MiniBatchKMeans (fit): the starts on subsamples of init_size points, the one of smallest inertia on
// a validation sample kept; then steps of batch_size points drawn with probability proportional to the weights (with
// replacement, unit weights in the batch): each step labels its batch by the centres, moves every centre to the
// mean of its old mean (its accumulated count) and its new points, and every 10 k points processed (or while a
// centre has no count) moves the centres of count below reassignment_ratio times the largest to points of the
// batch. It stops after (max_iter n) / batch_size steps, when the centres moved by at most tol's absolute value, or
// after max_no_improvement steps without a smaller smoothed batch inertia. Random elements (CL41): start s draws its
// subsample from r = 2^22 + s and its centres from r = s; the validation sample r = 2^23; the batches r = 2^23 + 1
// (t = step batch_size + q); the reassignments r = 2^23 + 2 (t = event batch_size + q).
struct MiniOut {
    Fit F;
    int64_t best_init = 0, n_steps = 0;
};

MiniOut minibatch(
    Data& D,
    int64_t k,
    int init,
    const double* C0,
    int64_t n_init,
    int64_t batch,
    int64_t max_iter,
    int64_t max_no_improvement,
    double tol_abs,
    int64_t init_size,
    double ratio,
    random::Key key,
    int path
) {
    const int64_t n = D.n, p = D.p;
    bool bounds, filter;
    choose_path(path, p, k, bounds, filter);
    D.filter = filter;
    D.prepare_filter();
    random::Stream st(key);
    constexpr std::uint64_t kValid = std::uint64_t{1} << 23;
    // the validation sample (uniform, with replacement) and its weights
    std::vector<int64_t> valid(static_cast<size_t>(init_size));
    for (int64_t t = 0; t < init_size; ++t)
        valid[static_cast<size_t>(t)] = draw(st, element(kValid, static_cast<std::uint64_t>(t)), n);
    auto valid_inertia = [&](const std::vector<double>& C) {
        Centres Cs;
        Cs.set(C.data(), k, p, false, false, D.M);
        ScanBuffers B;
        B.idx = valid;
        scan(D, Cs, false, B);
        double s = 0.0;
        for (size_t q = 0; q < valid.size(); ++q) s += D.weight(valid[q]) * B.res[q].s1;
        return s;
    };
    MiniOut out;
    std::vector<double> C;
    double best = kInfD;
    for (int64_t r = 0; r < n_init; ++r) {
        std::vector<double> Ci;
        if (init == 2) {
            Ci.assign(C0, C0 + k * p);
        } else if (init_size < n) {
            // the start's subsample (uniform, with replacement), its rows and weights
            std::vector<double> Xs(static_cast<size_t>(init_size * p)), ws(static_cast<size_t>(init_size));
            for (int64_t t = 0; t < init_size; ++t) {
                const int64_t i = draw(
                    st,
                    element((std::uint64_t{1} << 22) + static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(t)), n
                );
                std::copy(D.X + i * p, D.X + (i + 1) * p, Xs.begin() + t * p);
                ws[static_cast<size_t>(t)] = D.weight(i);
            }
            Data Ds;
            Ds.X = Xs.data(), Ds.w = ws.data(), Ds.n = init_size, Ds.p = p, Ds.M = D.M;
            Ci = init == 0 ? plusplus(Ds, k, key, static_cast<std::uint64_t>(r))
                           : random_start(Ds, k, key, static_cast<std::uint64_t>(r));
        } else {
            Ci = init == 0 ? plusplus(D, k, key, static_cast<std::uint64_t>(r))
                           : random_start(D, k, key, static_cast<std::uint64_t>(r));
        }
        const double v = valid_inertia(Ci);
        if (r == 0 || v < best) best = v, C = std::move(Ci), out.best_init = r;
    }
    // the steps
    std::vector<double> P;
    if (D.w) {
        P.resize(static_cast<size_t>(n));
        running_sums(D.w, n, P.data());
    }
    const int64_t n_steps = (max_iter * n) / batch;
    std::vector<double> counts(static_cast<size_t>(k), 0.0), Cnew(static_cast<size_t>(k * p));
    double ewa = 0.0, ewa_min = 0.0;
    bool have_ewa = false, have_min = false, stopped = false;
    int64_t no_improvement = 0, since = 0, events = 0, step = 0;
    ScanBuffers B;
    B.idx.resize(static_cast<size_t>(batch));
    Centres Cs;
    for (step = 0; step < n_steps; ++step) {
        for (int64_t q = 0; q < batch; ++q) {
            const std::uint64_t e = element(kValid + 1, static_cast<std::uint64_t>(step * batch + q));
            B.idx[static_cast<size_t>(q)] = D.w ? pick(P.data(), n, uniform(st, e)) : draw(st, e, n);
        }
        // the reassignment decision (scikit-learn's _random_reassign, before the step)
        since += batch;
        bool reassign = false;
        if (std::any_of(counts.begin(), counts.end(), [](double c) { return c == 0.0; }) || since >= 10 * k)
            since = 0, reassign = true;
        Cs.set(C.data(), k, p, filter, false, D.M);
        scan(D, Cs, filter, B);
        double batch_inertia = 0.0;
        for (int64_t q = 0; q < batch; ++q) batch_inertia += B.res[static_cast<size_t>(q)].s1;
        // the update: each centre with new points, its old mean times its count plus the points, over the new count
        std::vector<double> add(static_cast<size_t>(k), 0.0);
        for (int64_t q = 0; q < batch; ++q) add[static_cast<size_t>(B.res[static_cast<size_t>(q)].a)] += 1.0;
        for (int64_t j = 0; j < k; ++j)
            for (int64_t d = 0; d < p; ++d)
                Cnew[static_cast<size_t>(j * p + d)] =
                    add[static_cast<size_t>(j)] > 0.0
                        ? C[static_cast<size_t>(j * p + d)] * counts[static_cast<size_t>(j)]
                        : C[static_cast<size_t>(j * p + d)];
        for (int64_t q = 0; q < batch; ++q) {
            const int64_t j = B.res[static_cast<size_t>(q)].a;
            const double* x = D.X + B.idx[static_cast<size_t>(q)] * p;
            for (int64_t d = 0; d < p; ++d) Cnew[static_cast<size_t>(j * p + d)] += x[d];
        }
        for (int64_t j = 0; j < k; ++j) {
            if (!(add[static_cast<size_t>(j)] > 0.0)) continue;
            counts[static_cast<size_t>(j)] += add[static_cast<size_t>(j)];
            for (int64_t d = 0; d < p; ++d) Cnew[static_cast<size_t>(j * p + d)] /= counts[static_cast<size_t>(j)];
        }
        if (reassign && ratio > 0.0) {
            const double mx = *std::max_element(counts.begin(), counts.end());
            std::vector<int64_t> low;
            for (int64_t j = 0; j < k; ++j)
                if (counts[static_cast<size_t>(j)] < ratio * mx) low.push_back(j);
            const auto cap = static_cast<size_t>(batch / 2);
            if (low.size() > cap) {  // keep the centres of smallest counts (the smaller centre on a tie)
                std::stable_sort(low.begin(), low.end(), [&](int64_t a, int64_t b) {
                    return counts[static_cast<size_t>(a)] < counts[static_cast<size_t>(b)];
                });
                low.resize(cap);
                std::sort(low.begin(), low.end());
            }
            if (!low.empty()) {
                std::vector<char> keep(static_cast<size_t>(k), 1);
                for (int64_t j : low) keep[static_cast<size_t>(j)] = 0;
                double mn = kInfD;
                for (int64_t j = 0; j < k; ++j)
                    if (keep[static_cast<size_t>(j)]) mn = std::min(mn, counts[static_cast<size_t>(j)]);
                const std::vector<int64_t> rows = sample_without_replacement(
                    nullptr, batch, static_cast<int64_t>(low.size()), key, kValid + 2, nullptr, events * batch
                );
                ++events;
                for (size_t q = 0; q < low.size(); ++q) {
                    const double* x = D.X + B.idx[static_cast<size_t>(rows[q])] * p;
                    std::copy(x, x + p, Cnew.begin() + low[q] * p);
                    counts[static_cast<size_t>(low[q])] = mn;
                }
            }
        }
        double diff = 0.0;
        if (tol_abs > 0.0)
            for (int64_t q = 0; q < k * p; ++q) {
                const double z = Cnew[static_cast<size_t>(q)] - C[static_cast<size_t>(q)];
                diff += z * z;
            }
        C.swap(Cnew);
        // scikit-learn's _mini_batch_convergence
        const double bi = batch_inertia / static_cast<double>(batch);
        if (step == 0) continue;
        if (!have_ewa) {
            ewa = bi, have_ewa = true;
        } else {
            const double alpha = std::min(static_cast<double>(batch) * 2.0 / static_cast<double>(n + 1), 1.0);
            ewa = ewa * (1.0 - alpha) + bi * alpha;
        }
        if (tol_abs > 0.0 && diff <= tol_abs) {
            stopped = true;
            break;
        }
        if (!have_min || ewa < ewa_min) no_improvement = 0, ewa_min = ewa, have_min = true;
        else ++no_improvement;
        if (max_no_improvement >= 0 && no_improvement >= max_no_improvement) {
            stopped = true;
            break;
        }
    }
    out.n_steps = std::min(step + 1, n_steps);
    // the partition of every point by the final centres
    Fit& F = out.F;
    F.lab.assign(static_cast<size_t>(n), -1);
    Cs.set(C.data(), k, p, filter, false, D.M);
    assign(D, Cs, F.lab, nullptr, true, filter);
    F.inertia = inertia_of(D, F.lab.data(), C.data(), k, &F.per);
    F.C = std::move(C);
    F.converged = stopped;
    F.n_iter = (out.n_steps * batch + n - 1) / n;
    return out;
}

nb::object minibatch_py(
    nb::handle Xo,
    nb::handle ordo,
    nb::handle wo,
    int64_t k,
    int init,
    nb::handle C0o,
    int64_t n_init,
    int64_t batch,
    int64_t max_iter,
    int64_t max_no_improvement,
    double tol,
    int64_t init_size,
    double ratio,
    int64_t key0,
    int64_t key1,
    int path
) {
    const Args A = fit_args(Xo, ordo, wo);
    const int64_t n = A.n, p = A.p;
    const double* C0 = nullptr;
    if (init == 2) {
        int64_t kk, pp;
        C0 = matrix_arg(C0o, "C0", kk, pp);
        if (kk != k || pp != p) throw nb::value_error("C0 must be (k, p)");
    }
    if (k < 1 || k > n || batch < 1 || init_size < 1) throw nb::value_error("k, batch_size, init_size");
    MiniOut out;
    int e = 0;
    {
        const parallel::Unlocked unlocked(n * p >= kSplitWork);
        const KeyRows R(A.X, A.order, A.w, n, p, C0 ? max_abs_of(C0, k * p) : 0.0);
        e = R.e;
        std::vector<double> C0s;
        if (C0) C0s = e ? scaled_copy(C0, k * p, e) : std::vector<double>(C0, C0 + k * p);
        Data D = R.data();
        out = minibatch(
            D, k, init, C0 ? C0s.data() : nullptr, n_init, batch, max_iter, max_no_improvement, tolerance(D, tol),
            init_size, ratio, key_of(key0, key1), path
        );
    }
    nb::object head = fit_head(out.F, n, k, p, e);
    return nb::make_tuple(head, out.F.n_iter, out.F.converged, out.best_init, out.n_steps);
}

// ---- bisecting k-means -------------------------------------------------------------------------------------------

// scikit-learn's BisectingKMeans: k - 1 times the leaf of largest score (its inertia, or its number of points; the
// first leaf in the tree's left-to-right order on a tie; a leaf of fewer than two distinct points of positive weight
// is never split) is split by k-means with k = 2 (n_init starts, the best the smallest (inertia, start)), the
// children's centres those of the split. The tolerance is the whole data's (sklearn's _tolerance on X). Split s's
// starts draw r = s n_init + i (CL41). Each node keeps its centre for predict (down the tree, the nearer child; the
// left on a tie).
struct BisectNode {
    std::vector<int64_t> rows;  // the points (keys, increasing)
    std::vector<double> centre;
    double inertia = 0.0, score = 0.0;
    int64_t left = -1, right = -1, split = -1;  // children (node numbers) and the split's number
};

struct BisectOut {
    std::vector<BisectNode> nodes;  // node 0 the root
    std::vector<int64_t> leaves;    // in the tree's left-to-right order
    std::vector<int64_t> best_init;
    int64_t n_iter = 0;
    bool converged = true;
};

void leaves_of(const std::vector<BisectNode>& N, int64_t v, std::vector<int64_t>& out) {
    if (N[static_cast<size_t>(v)].left < 0) {
        out.push_back(v);
        return;
    }
    leaves_of(N, N[static_cast<size_t>(v)].left, out);
    leaves_of(N, N[static_cast<size_t>(v)].right, out);
}

int64_t distinct_rows(const Data& D, const std::vector<int64_t>& rows, int64_t enough) {
    int64_t count = 0;
    const double* last = nullptr;
    for (int64_t i : rows) {
        if (!(D.weight(i) > 0.0)) continue;
        const double* x = D.X + i * D.p;
        if (!last || !std::equal(x, x + D.p, last)) {
            if (++count >= enough) return count;
        }
        last = x;
    }
    return count;
}

BisectOut bisect(
    Data& D,
    int64_t k,
    int init,
    int64_t n_init,
    int64_t max_iter,
    double tol_abs,
    bool by_size,
    random::Key key,
    int path
) {
    const int64_t n = D.n, p = D.p;
    BisectOut out;
    out.nodes.emplace_back();
    out.nodes[0].rows.resize(static_cast<size_t>(n));
    std::iota(out.nodes[0].rows.begin(), out.nodes[0].rows.end(), int64_t{0});
    for (int64_t s = 0; s + 1 < k; ++s) {
        std::vector<int64_t> lv;
        leaves_of(out.nodes, 0, lv);
        int64_t pickv = -1;
        for (int64_t v : lv) {
            const BisectNode& L = out.nodes[static_cast<size_t>(v)];
            if (distinct_rows(D, L.rows, 2) < 2) continue;
            if (pickv < 0 || L.score > out.nodes[static_cast<size_t>(pickv)].score) pickv = v;
        }
        if (pickv < 0) throw std::runtime_error("bisecting_kmeans: no leaf to split");
        const std::vector<int64_t> rows = out.nodes[static_cast<size_t>(pickv)].rows;
        const int64_t m = static_cast<int64_t>(rows.size());
        std::vector<double> Xs(static_cast<size_t>(m * p)), ws;
        if (D.w) ws.resize(static_cast<size_t>(m));
        for (int64_t q = 0; q < m; ++q) {
            const int64_t i = rows[static_cast<size_t>(q)];
            std::copy(D.X + i * p, D.X + (i + 1) * p, Xs.begin() + q * p);
            if (D.w) ws[static_cast<size_t>(q)] = D.w[i];
        }
        Data Ds;
        Ds.X = Xs.data(), Ds.w = D.w ? ws.data() : nullptr, Ds.n = m, Ds.p = p, Ds.M = D.M;
        KMeansOut ko = kmeans_run(
            Ds, 2, init, nullptr, n_init, max_iter, tol_abs, key, path, static_cast<std::uint64_t>(s * n_init)
        );
        out.best_init.push_back(ko.best_init);
        out.n_iter += ko.best.n_iter;
        out.converged = out.converged && ko.best.converged;
        for (int c = 0; c < 2; ++c) {
            BisectNode child;
            for (int64_t q = 0; q < m; ++q)
                if (ko.best.lab[static_cast<size_t>(q)] == c) child.rows.push_back(rows[static_cast<size_t>(q)]);
            child.centre.assign(ko.best.C.begin() + c * p, ko.best.C.begin() + (c + 1) * p);
            child.inertia = ko.best.per[static_cast<size_t>(c)];
            child.score = by_size ? static_cast<double>(child.rows.size()) : child.inertia;
            out.nodes.push_back(std::move(child));
        }
        BisectNode& P = out.nodes[static_cast<size_t>(pickv)];
        P.left = static_cast<int64_t>(out.nodes.size()) - 2;
        P.right = static_cast<int64_t>(out.nodes.size()) - 1;
        P.split = s;
        P.rows.clear();
        P.rows.shrink_to_fit();
    }
    leaves_of(out.nodes, 0, out.leaves);
    return out;
}

nb::object bisecting_py(
    nb::handle Xo,
    nb::handle ordo,
    nb::handle wo,
    int64_t k,
    int init,
    int64_t n_init,
    int64_t max_iter,
    double tol,
    bool by_size,
    int64_t key0,
    int64_t key1,
    int path
) {
    const Args A = fit_args(Xo, ordo, wo);
    const int64_t n = A.n, p = A.p;
    if (k < 1 || k > n) throw nb::value_error("k");
    BisectOut out;
    Fit F;
    int e = 0;
    std::vector<double> node_centres;  // (2k - 1, p), unscaled: node v's centre (the root's: zeros)
    std::vector<int64_t> children;     // (k - 1, 2): each split node's (left, right) node numbers, by split
    std::vector<double> heights;       // (k - 1,): the split node's inertia, unscaled
    std::vector<int64_t> split_node(static_cast<size_t>(std::max<int64_t>(k - 1, 0)));
    std::vector<int64_t> leaf_of_node;
    {
        const parallel::Unlocked unlocked(n * p >= kSplitWork);
        const KeyRows R(A.X, A.order, A.w, n, p);
        e = R.e;
        Data D = R.data();
        out = bisect(D, k, init, n_init, max_iter, tolerance(D, tol), by_size, key_of(key0, key1), path);
        // the leaves numbered in the tree's order: labels, centres, inertias
        const int64_t nn = static_cast<int64_t>(out.nodes.size());
        leaf_of_node.assign(static_cast<size_t>(nn), -1);
        F.lab.assign(static_cast<size_t>(n), 0);
        F.C.assign(static_cast<size_t>(k * p), 0.0);
        for (size_t j = 0; j < out.leaves.size(); ++j) {
            const BisectNode& L = out.nodes[static_cast<size_t>(out.leaves[j])];
            leaf_of_node[static_cast<size_t>(out.leaves[j])] = static_cast<int64_t>(j);
            for (int64_t i : L.rows) F.lab[static_cast<size_t>(i)] = static_cast<int64_t>(j);
            if (!L.centre.empty())
                std::copy(L.centre.begin(), L.centre.end(), F.C.begin() + static_cast<int64_t>(j) * p);
        }
        if (k == 1) {  // one leaf, the root: its centre the weighted mean
            std::vector<double> S, W;
            label_sums(D, F.lab.data(), 1, S, W);
            for (int64_t d = 0; d < p; ++d) F.C[static_cast<size_t>(d)] = S[static_cast<size_t>(d)] / W[0];
        }
        D.filter = filter_shape(p, k);
        F.inertia = inertia_of(D, F.lab.data(), F.C.data(), k, &F.per);
        node_centres.assign(static_cast<size_t>(nn * p), 0.0);
        for (int64_t v = 0; v < nn; ++v) {
            const BisectNode& N = out.nodes[static_cast<size_t>(v)];
            for (size_t d = 0; d < N.centre.size(); ++d)
                node_centres[static_cast<size_t>(v * p) + d] = e ? spatial::scaled(N.centre[d], -e) : N.centre[d];
            if (N.split >= 0)
                split_node[static_cast<size_t>(N.split)] = v, leaf_of_node[static_cast<size_t>(v)] = -2 - N.split;
        }
        children.resize(static_cast<size_t>(2 * std::max<int64_t>(k - 1, 0)));
        heights.resize(static_cast<size_t>(std::max<int64_t>(k - 1, 0)));
        for (int64_t s = 0; s + 1 < k; ++s) {
            const BisectNode& N = out.nodes[static_cast<size_t>(split_node[static_cast<size_t>(s)])];
            children[static_cast<size_t>(2 * s)] = N.left;
            children[static_cast<size_t>(2 * s + 1)] = N.right;
            const double h = N.inertia;  // the split node's inertia about its centre (the root: the total below)
            heights[static_cast<size_t>(s)] = e ? std::ldexp(h, -2 * e) : h;
        }
        if (k > 1) {  // the root's inertia about the weighted mean of all points
            std::vector<double> S, W, mean(static_cast<size_t>(p));
            std::vector<int64_t> zero(static_cast<size_t>(n), 0);
            label_sums(D, zero.data(), 1, S, W);
            for (int64_t d = 0; d < p; ++d) mean[static_cast<size_t>(d)] = S[static_cast<size_t>(d)] / W[0];
            const double h = inertia_of(D, zero.data(), mean.data(), 1);
            heights[0] = e ? std::ldexp(h, -2 * e) : h;
        }
    }
    nb::object head = fit_head(F, n, k, p, e);
    const int64_t nn = static_cast<int64_t>(out.nodes.size());
    std::vector<int64_t> sizes(static_cast<size_t>(nn), 0);
    for (int64_t v = nn - 1; v >= 0; --v) {
        const BisectNode& N = out.nodes[static_cast<size_t>(v)];
        sizes[static_cast<size_t>(v)] = N.left < 0
                                            ? static_cast<int64_t>(N.rows.size())
                                            : sizes[static_cast<size_t>(N.left)] + sizes[static_cast<size_t>(N.right)];
    }
    return nb::make_tuple(
        head, out.n_iter, out.converged, wrap(int_array(out.best_init, std::max<int64_t>(k - 1, 0))),
        wrap(float_array(node_centres, nn, p)), wrap(int_array(children, std::max<int64_t>(k - 1, 0), 2)),
        wrap(float_array(heights, std::max<int64_t>(k - 1, 0))), wrap(int_array(leaf_of_node, nn)),
        wrap(int_array(sizes, nn))
    );
}

// bisecting k-means' predict: down the tree from the root (node 0), at each split the child of smaller sqd (the left
// on a tie), to a leaf; X and the node centres take the exponent of their largest |value| (CL43). leaf[v]: node
// v's leaf number, or -2 - s for the node of split s, whose children are children[s]. -> (m,) leaf numbers
nb::object bisect_predict_py(nb::handle Xo, nb::handle Co, nb::handle cho, nb::handle leafo) {
    int64_t m, p, nn, pc, nl;
    const double* X = matrix_arg(Xo, "X", m, p);
    const double* C = matrix_arg(Co, "centres", nn, pc);
    const int64_t* leaf = index_arg(leafo, "leaf", nl);
    NDArray& cha = require_ndarray(cho, "children");
    if (cha.dtype != DType::Int64 || !cha.c_contiguous || cha.ndim != 2 || cha.shape[1] != 2)
        throw nb::value_error("children: (s, 2) int64");
    const auto* ch = reinterpret_cast<const int64_t*>(cha.data);
    if (pc != p || nl != nn) throw nb::value_error("centres");
    std::vector<int64_t> lab(static_cast<size_t>(m));
    {
        const parallel::Unlocked unlocked(m * p >= kSplitWork);
        const int e = scale_exponent(std::max(max_abs_of(X, m * p), max_abs_of(C, nn * p)));
        Scaled Xs(X, m * p, e);
        Scaled Cs(C, nn * p, e);
        auto body = [&](int64_t t) {
            for (int64_t i = t * kTask; i < std::min(m, (t + 1) * kTask); ++i) {
                const double* x = Xs.X + i * p;
                int64_t v = 0;
                while (leaf[v] < 0) {
                    const int64_t sp = -2 - leaf[v], a = ch[2 * sp], b = ch[2 * sp + 1];
                    v = sqd(x, Cs.X + b * p, p) < sqd(x, Cs.X + a * p, p) ? b : a;
                }
                lab[static_cast<size_t>(i)] = leaf[v];
            }
        };
        const int64_t nt = (m + kTask - 1) / kTask;
        if (nt > 1 && m * p >= kSplitWork) parallel::parallel_for(nt, body, kTask * p);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    }
    return wrap(int_array(lab, m));
}

// The nearest centres of new rows X (m, p) (predict): C the centres (k, p) in the fit's numbering, Cs the same on
// the fit's grid (exponent e). The rows and centres take the exponent of their largest |value| (CL43); when it is
// the fit's, the fit's own centres Cs, so the fit's rows get the fit's labels. -> (labels, sqd on that grid)
nb::object kmeans_predict_py(nb::handle Xo, nb::handle Co, nb::handle Cso, int e_fit, int path) {
    int64_t m, p, k, pc, k2, p2;
    const double* X = matrix_arg(Xo, "X", m, p);
    const double* C = matrix_arg(Co, "C", k, pc);
    const double* Cfit = matrix_arg(Cso, "Cs", k2, p2);
    if (pc != p || k < 1 || k2 != k || p2 != p) throw nb::value_error("C must be (k, p)");
    std::vector<int64_t> lab(static_cast<size_t>(m));
    std::vector<double> s(static_cast<size_t>(m));
    {
        const parallel::Unlocked unlocked(m * p * k >= kSplitWork);
        const int e = scale_exponent(std::max(max_abs_of(X, m * p), max_abs_of(C, k * p)));
        Scaled Xs(X, m * p, e);
        std::vector<double> Cown;
        const double* Cg = Cfit;
        if (e != e_fit) Cown = e ? scaled_copy(C, k * p, e) : std::vector<double>(C, C + k * p), Cg = Cown.data();
        Data D;
        D.X = Xs.X, D.n = m, D.p = p, D.M = Margin(p);
        bool bounds, filter;
        choose_path(path, p, k, bounds, filter);
        D.filter = filter;
        D.prepare_filter();
        Centres Cs;
        Cs.set(Cg, k, p, filter, false, D.M);
        const int64_t nt = (m + kTask - 1) / kTask;
        auto body = [&](int64_t t) {
            ScanBuffers B;
            for (int64_t i = t * kTask; i < std::min(m, (t + 1) * kTask); ++i) B.idx.push_back(i);
            scan(D, Cs, filter, B);
            for (size_t q = 0; q < B.idx.size(); ++q)
                lab[static_cast<size_t>(B.idx[q])] = B.res[q].a, s[static_cast<size_t>(B.idx[q])] = B.res[q].s1;
        };
        if (nt > 1 && m * p * k >= kSplitWork) parallel::parallel_for(nt, body, kTask * p * k);
        else
            for (int64_t t = 0; t < nt; ++t) body(t);
    }
    return nb::make_tuple(wrap(int_array(lab, m)), wrap(float_array(s, m)));
}

// The number of distinct rows of positive weight (X[order[r]] in key order: equal rows are neighbours).
int64_t kmeans_distinct_py(nb::handle Xo, nb::handle ordo, nb::handle wo) {
    const Args A = fit_args(Xo, ordo, wo);
    int64_t count = 0;
    const double* last = nullptr;
    for (int64_t r = 0; r < A.n; ++r) {
        const int64_t i = A.order[r];
        if (A.w && !(A.w[i] > 0.0)) continue;
        const double* x = A.X + i * A.p;
        if (!last || !std::equal(x, x + A.p, last, [](double a, double b) { return a == b; })) ++count;
        last = x;
    }
    return count;
}

// Labels in key order (lab[r] the centre of the row of rank r, 0 .. k - 1) to the input's order (order[r] the input
// index of rank r), renumbered by first appearance (CL6); a centre no point takes comes after the others in its own
// order. -> (labels (n,), new number of each old one (k,))
nb::object first_seen_py(nb::handle labo, nb::handle ordo, int64_t k) {
    int64_t n, m;
    const int64_t* lab = index_arg(labo, "labels", n);
    const int64_t* order = index_arg(ordo, "order", m);
    if (m != n) throw nb::value_error("order");
    std::vector<int64_t> in(static_cast<size_t>(n)), num(static_cast<size_t>(k), -1);
    for (int64_t r = 0; r < n; ++r) in[static_cast<size_t>(order[r])] = lab[r];
    int64_t next = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t j = in[static_cast<size_t>(i)];
        if (j < 0) continue;
        if (num[static_cast<size_t>(j)] < 0) num[static_cast<size_t>(j)] = next++;
        in[static_cast<size_t>(i)] = num[static_cast<size_t>(j)];
    }
    for (int64_t j = 0; j < k; ++j)
        if (num[static_cast<size_t>(j)] < 0) num[static_cast<size_t>(j)] = next++;
    return nb::make_tuple(wrap(int_array(in, n)), wrap(int_array(num, k)));
}

// The rows k-means++ picks (start r of the stream keyed by key0, key1; CL63): their ranks in key order, in the order
// picked (gaussian_mixture's init='k-means++', as scikit-learn's GaussianMixture takes kmeans_plusplus's indices).
nb::object plusplus_py(nb::handle Xo, nb::handle ordo, int64_t k, int64_t key0, int64_t key1, int64_t r) {
    const Args A = fit_args(Xo, ordo, nb::none());
    if (k < 1 || k > A.n || r < 0) throw nb::value_error("k");
    std::vector<int64_t> idx;
    {
        const parallel::Unlocked unlocked(A.n * A.p >= kSplitWork);
        const KeyRows R(A.X, A.order, nullptr, A.n, A.p);
        const Data D = R.data();
        (void)plusplus(D, k, key_of(key0, key1), static_cast<std::uint64_t>(r), &idx);
    }
    return wrap(int_array(idx, k));
}

// Accelerate's expanded form alone (sklearn's): the measurements' baseline (tools/cluster_probe.py); single: with
// Accelerate held to one thread for the call (BLASSetThreading), else Accelerate's own threads
nb::object assign_expanded_py(nb::handle Xo, nb::handle Co, bool single) {
    int64_t n, p, k, pc;
    const double* X = matrix_arg(Xo, "X", n, p);
    const double* C = matrix_arg(Co, "C", k, pc);
    std::vector<int64_t> lab(static_cast<size_t>(n));
    {
        const parallel::Unlocked unlocked(true);
        std::vector<double> cn(static_cast<size_t>(k));
        for (int64_t j = 0; j < k; ++j) {
            double s = 0;
            for (int64_t d = 0; d < p; ++d) s += C[j * p + d] * C[j * p + d];
            cn[static_cast<size_t>(j)] = s;
        }
        const BLAS_THREADING before = BLASGetThreading();
        if (single) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        constexpr int64_t kRows = 4096;
        std::vector<double> G(static_cast<size_t>(kRows * k));
        for (int64_t i0 = 0; i0 < n; i0 += kRows) {
            const int64_t m = std::min(kRows, n - i0);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, p, 1.0, X + i0 * p, p, C, p, 0.0, G.data(), k);
            for (int64_t r = 0; r < m; ++r) {
                const double* x = X + (i0 + r) * p;
                double xn = 0;
                for (int64_t d = 0; d < p; ++d) xn += x[d] * x[d];
                double best = kInfD;
                int64_t arg = 0;
                for (int64_t j = 0; j < k; ++j) {
                    const double v = xn - 2.0 * G[static_cast<size_t>(r * k + j)] + cn[static_cast<size_t>(j)];
                    if (v < best) best = v, arg = j;
                }
                lab[static_cast<size_t>(i0 + r)] = arg;
            }
        }
        if (single) BLASSetThreading(before);
    }
    return wrap(int_array(lab, n));
}

}  // namespace

void bind_kmeans(nb::module_& m) {
    m.def(
        "kmeans", &kmeans_py, nb::arg("X"), nb::arg("order"), nb::arg("w").none(), nb::arg("k"), nb::arg("init"),
        nb::arg("C0").none(), nb::arg("n_init"), nb::arg("max_iter"), nb::arg("tol"), nb::arg("key0"), nb::arg("key1"),
        nb::arg("path")
    );
    m.def(
        "minibatch_kmeans", &minibatch_py, nb::arg("X"), nb::arg("order"), nb::arg("w").none(), nb::arg("k"),
        nb::arg("init"), nb::arg("C0").none(), nb::arg("n_init"), nb::arg("batch_size"), nb::arg("max_iter"),
        nb::arg("max_no_improvement"), nb::arg("tol"), nb::arg("init_size"), nb::arg("reassignment_ratio"),
        nb::arg("key0"), nb::arg("key1"), nb::arg("path")
    );
    m.def(
        "bisecting_kmeans", &bisecting_py, nb::arg("X"), nb::arg("order"), nb::arg("w").none(), nb::arg("k"),
        nb::arg("init"), nb::arg("n_init"), nb::arg("max_iter"), nb::arg("tol"), nb::arg("by_size"), nb::arg("key0"),
        nb::arg("key1"), nb::arg("path")
    );
    m.def(
        "bisecting_predict", &bisect_predict_py, nb::arg("X"), nb::arg("centres"), nb::arg("children"), nb::arg("leaf")
    );
    m.def(
        "kmeans_predict", &kmeans_predict_py, nb::arg("X"), nb::arg("C"), nb::arg("Cs"), nb::arg("e"), nb::arg("path")
    );
    m.def("kmeans_distinct", &kmeans_distinct_py, nb::arg("X"), nb::arg("order"), nb::arg("w").none());
    m.def("first_seen", &first_seen_py, nb::arg("labels"), nb::arg("order"), nb::arg("k"));
    m.def(
        "kmeans_plusplus", &plusplus_py, nb::arg("X"), nb::arg("order"), nb::arg("k"), nb::arg("key0"), nb::arg("key1"),
        nb::arg("r")
    );
    m.def("_kmeans_assign_expanded", &assign_expanded_py, nb::arg("X"), nb::arg("C"), nb::arg("single") = false);
}

}  // namespace sd::cluster
