// Quantile regression's numerical core (docs/design/regression.md §3.6).
//
// One regression per batch element, through linalg's run_batch; the quantiles of one
// regression (q an array, G62) are solved in the given order, each from the previous one's
// basis.
//   1. Checks (inf is an error, NaN gives an all-NaN result) and least squares through the
//      linear models' core (regression/lsq.hpp): the aliased columns (G9, G21), R and
//      (R^T R)^-1 for the covariances, and the interior point method's start.
//   2. The kept columns (intercept first) are packed row-major, and exact duplicate rows
//      (the same y and x, bit for bit up to the sign of zero) are merged into one row with a
//      multiplicity m_i: the objective sum m_i rho_q(y_i - x_i b) is the same function.
//   3. Cold start (the first quantile; G60): the Frisch-Newton primal-dual interior point
//      method on the dual linear program max y^T a s.t. X^T a = (1 - q) X^T m, 0 <= a <= m
//      (Portnoy and Koenker 1997, quantreg's rq.fit.fnb): Mehrotra's predictor-corrector,
//      steps 0.99995 of the way to the boundary, from the least squares coefficients with
//      the intercept moved to their residuals' q-quantile (dual slacks floored, Newton
//      solves refined: see ipm); stopped at the relative duality gap `tol`, at `max_iter`, or
//      when the Newton system cannot be factored. Crossover: the rows in the order of |r_i|
//      at its coefficients (ties by row), the first k linearly independent ones are the basis.
//   4. Simplex pivots from that basis, Barrodale and Roberts' steps: along the edge of the
//      basis row whose multiplier is most out of its box, with the exact line search over
//      the breakpoints, until the dual certificate holds: lambda solves X_B^T lambda = -g,
//      g = sum_{i not in B} psi_q(r_i) m_i x_i, and every lambda_j lies in
//      [m_j (q - 1), m_j q]. Degenerate steps (zero residuals blocking) are taken too; when a
//      state (basis and residual signs) repeats, Bland's rule until the objective decreases.
//   5. Non-unique solutions (G63): a multiplier at a bound leaves the optimal set F, cut out
//      by the certificate's sign constraints, possibly larger than a point. The returned
//      vertex is F's lexicographically smallest point (params[0] smallest, then params[1],
//      ...), found by pivots inside F with Bland's rule; `nonunique` when F has a second
//      point (a pivot of positive length towards F's lexicographic maximum exists).
//   6. The vertex's basis is made canonical: of the rows with zero residual, the first k
//      linearly independent ones in row order. beta solves X_B beta = y_B by QR with up to
//      three corrections on the residual summed in double-double, so the result depends on
//      the vertex only, not on the path to it (warm or cold start, thread count).
//   7. Covariances as R's summary.rq (quantreg 6.1), with its eps 2^-26 times max |y_i|
//      (G67): 'iid' (Koenker-Bassett, the sparsity from a median regression on order
//      statistics of the residuals), 'nid' (Hendricks-Koenker, the fits at q +- h) and 'ker'
//      (Powell's kernel); the sandwiches through the QR of sqrt(f) X.
//
// Row sums are cut into chunks fixed by n alone and added in order; selections and sorts
// compare with fixed keys. Nothing depends on the thread count.
#include "regression/quantile.hpp"

#include "regression/lsq.hpp"

#include "bind/errors.hpp"
#include "linalg/ops/common.hpp"
#include "parallel/pool.hpp"
#include "special/dd.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sd::regression {

namespace nb = nanobind;
using namespace lsq;
using linalg::Batch;
using linalg::Failure;
using linalg::Input;
using linalg::lint;
using linalg::Out;
using linalg::Scratch;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 0x1p-52;
constexpr double kU = 0x1p-53;

// The Python layer maps the names (stardust/regression/_quantile.py).
enum CovType : int { cIid = 0, cNid, cKer };

// The per-(regression, quantile) scalars, in this order (stardust/regression/_quantile.py
// reads them). qNGroups and qLags are NaN (the shared result code asks for them).
enum QStat : int {
    qNobs = 0, qRank, qObjective, qNullObjective, qNIter, qNPivots, qConverged, qNonunique, qIsNaN, qBandwidth,
    qNGroups, qLags, qNStat
};

constexpr int eInfY = 1, eInfX = 2;

constexpr double kStep = 0.99995;        // rq.fit.fnb's beta: the fraction of the step to the boundary
constexpr double kRZero = 0.001;         // the start's dual slack floor, relative to the mean |r|
constexpr double kSqrtEps = 0x1p-26;     // R's .Machine$double.eps^(1/2), summary.rq's eps (times max |y_i|, G67)
constexpr int kRefine = 3;               // corrections of a basis solve (G15)
constexpr double kDualTol = 32.0;        // a multiplier within 32 u (times its scale) of a bound is at it
constexpr double kLexTol = 64.0;         // a direction's component below 64 u ||d|| is zero (lexicographic sign)

inline double rd(const std::byte* p) {
    double v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    e = std::fma(a, b, -p);
}

// Row passes: chunks of rows fixed by n alone, partial results combined in order (the same
// bits for any thread count), on the pool from kRowParallelMin rows (serially inside a task).
constexpr lint kRowChunk = 4096;
constexpr lint kRowParallelMin = 1 << 15;

inline lint row_chunks(lint n) { return std::max<lint>(1, (n + kRowChunk - 1) / kRowChunk); }

template <class F>
void for_row_chunks(lint n, F&& body) {
    const lint nc = (n + kRowChunk - 1) / kRowChunk;
    auto one = [&](lint c) { body(c, c * kRowChunk, std::min(n, (c + 1) * kRowChunk)); };
    if (nc >= 2 && n >= kRowParallelMin) {
        parallel::parallel_for(nc, [&](std::int64_t c) { one(static_cast<lint>(c)); });
    } else {
        for (lint c = 0; c < nc; ++c) one(c);
    }
}

// Compensated sums (Ogita, Rump and Oishi's Sum2): s + c carries about twice the precision.
struct Sum2 {
    double s = 0.0, c = 0.0;
    void add(double v) {
        double e;
        two_sum(s, v, s, e);
        c += e;
    }
    void add_prod(double a, double b) {
        double p, e;
        two_prod(a, b, p, e);
        add(p);
        c += e;
    }
    void add(const Sum2& o) {
        add(o.s);
        c += o.c;
    }
    double value() const { return s + c; }
};

// Per-chunk vectors of `width` Sum2 slots, combined in chunk order into out.
template <class F>
void chunk_sum2(lint n, lint width, std::vector<Sum2>& out, F&& body) {
    const lint nc = row_chunks(n);
    std::vector<Sum2> part(static_cast<std::size_t>(nc * width));
    for_row_chunks(n, [&](lint c, lint i0, lint i1) { body(i0, i1, part.data() + c * width); });
    out.assign(static_cast<std::size_t>(width), Sum2{});
    for (lint c = 0; c < nc; ++c)
        for (lint j = 0; j < width; ++j) out[j].add(part[c * width + j]);
}

// Per-chunk plain double sums (the interior point method's: its bits only need to be
// deterministic), combined in chunk order.
template <class F>
void chunk_sum(lint n, lint width, std::vector<double>& out, F&& body) {
    const lint nc = row_chunks(n);
    std::vector<double> part(static_cast<std::size_t>(nc * width), 0.0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) { body(i0, i1, part.data() + c * width); });
    out.assign(static_cast<std::size_t>(width), 0.0);
    for (lint c = 0; c < nc; ++c)
        for (lint j = 0; j < width; ++j) out[j] += part[c * width + j];
}

// ---- the original design (strided input, intercept column 0) ----------------------------
struct Design {
    lint n, p, k;
    bool intercept;
    const std::byte* X;
    std::int64_t xrs, xcs;
    double at(lint i, lint c) const {
        return (intercept && c == 0) ? 1.0 : rd(X + i * xrs + (c - (intercept ? 1 : 0)) * xcs);
    }
};

// ---- the packed problem ------------------------------------------------------------------
// min sum_i m_i rho_q(y_i - x_i b) over n rows (duplicates merged) and k columns (the kept
// columns of the design, intercept first), X row-major.
struct QP {
    lint n = 0, k = 0;
    const double* X = nullptr;
    const double* y = nullptr;
    const double* m = nullptr;
    bool icept = false;          // column 0 is the intercept's
    double itol = 0.0;           // a row is independent of others when its part orthogonal to them exceeds itol ||x_i||
    std::vector<Sum2> sall;      // sum_i m_i x_i
    std::vector<double> l1;      // ||x_i||_1
    const double* row(lint i) const { return X + i * k; }
};

void make_sall(QP& P) {
    P.l1.resize(static_cast<std::size_t>(P.n));
    for_row_chunks(P.n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            const double* x = P.row(i);
            double a = 0.0;
            for (lint c = 0; c < P.k; ++c) a += std::fabs(x[c]);
            P.l1[i] = a;
        }
    });
    chunk_sum2(P.n, P.k, P.sall, [&](lint i0, lint i1, Sum2* acc) {
        for (lint i = i0; i < i1; ++i) {
            const double* x = P.row(i);
            const double mi = P.m[i];
            for (lint c = 0; c < P.k; ++c) acc[c].add_prod(mi, x[c]);
        }
    });
}

// o1[i - i0] = x_i b1 (and o2[i - i0] = x_i b2) for the rows i0 .. i1, each row summed in
// column order with fused multiply-adds; four rows at a time (independent chains: the same
// bits as one row at a time).
inline void dots(const QP& P, lint i0, lint i1, const double* b1, const double* b2, double* o1, double* o2) {
    const lint k = P.k;
    lint i = i0;
    for (; i + 4 <= i1; i += 4) {
        const double* __restrict x0 = P.row(i);
        const double* __restrict x1 = x0 + k;
        const double* __restrict x2 = x1 + k;
        const double* __restrict x3 = x2 + k;
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        if (b2) {
            double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
            for (lint c = 0; c < k; ++c) {
                const double u = b1[c], v = b2[c];
                a0 = std::fma(x0[c], u, a0);
                a1 = std::fma(x1[c], u, a1);
                a2 = std::fma(x2[c], u, a2);
                a3 = std::fma(x3[c], u, a3);
                c0 = std::fma(x0[c], v, c0);
                c1 = std::fma(x1[c], v, c1);
                c2 = std::fma(x2[c], v, c2);
                c3 = std::fma(x3[c], v, c3);
            }
            o2[i - i0] = c0, o2[i - i0 + 1] = c1, o2[i - i0 + 2] = c2, o2[i - i0 + 3] = c3;
        } else {
            for (lint c = 0; c < k; ++c) {
                const double u = b1[c];
                a0 = std::fma(x0[c], u, a0);
                a1 = std::fma(x1[c], u, a1);
                a2 = std::fma(x2[c], u, a2);
                a3 = std::fma(x3[c], u, a3);
            }
        }
        o1[i - i0] = a0, o1[i - i0 + 1] = a1, o1[i - i0 + 2] = a2, o1[i - i0 + 3] = a3;
    }
    for (; i < i1; ++i) {
        const double* x = P.row(i);
        double a = 0.0, v = 0.0;
        for (lint c = 0; c < k; ++c) {
            a = std::fma(x[c], b1[c], a);
            if (b2) v = std::fma(x[c], b2[c], v);
        }
        o1[i - i0] = a;
        if (b2) o2[i - i0] = v;
    }
}

// ---- a basis: the k x k matrix X_B of the basis rows, its QR, refined solves --------------
struct Basis {
    lint k = 0;
    std::vector<double> a, f, tau;  // X_B (column-major, exact), its geqrf factors

    bool factor(const QP& P, const std::vector<lint>& B, Scratch& sc) {
        k = P.k;
        a.resize(static_cast<std::size_t>(k * k));
        for (lint r = 0; r < k; ++r) {
            const double* x = P.row(B[r]);
            for (lint c = 0; c < k; ++c) a[c * k + r] = x[c];
        }
        f = a;
        tau.resize(static_cast<std::size_t>(k));
        linalg::lapack::geqrf(k, k, f.data(), k, tau.data(), sc);
        for (lint c = 0; c < k; ++c) {
            const double d = f[c * k + c];
            if (!(d != 0.0) || !std::isfinite(d)) return false;
        }
        return true;
    }
    void solve(double* b, Scratch& sc) const {  // X_B v = b, in place
        linalg::lapack::ormqr('T', k, 1, k, f.data(), k, tau.data(), b, k, sc);
        linalg::lapack::trtrs('U', 'N', 'N', k, 1, f.data(), k, b, k);
    }
    void solve_t(double* b, Scratch& sc) const {  // X_B^T v = b, in place
        linalg::lapack::trtrs('U', 'T', 'N', k, 1, f.data(), k, b, k);
        linalg::lapack::ormqr('N', k, 1, k, f.data(), k, tau.data(), b, k, sc);
    }
    // X_B v = bh + bl (trans false) or X_B^T v = bh + bl (true): the QR solution, then up to
    // kRefine corrections from the residual in double-double (Dot2), stopping when a
    // correction changes no component.
    void refined(const double* bh, const double* bl, double* v, bool trans, Scratch& sc) const {
        std::vector<double> d(static_cast<std::size_t>(k));
        for (lint r = 0; r < k; ++r) v[r] = bh[r];
        trans ? solve_t(v, sc) : solve(v, sc);
        for (int it = 0; it < kRefine; ++it) {
            for (lint r = 0; r < k; ++r) {
                double P = bh[r], S = bl ? bl[r] : 0.0;
                for (lint c = 0; c < k; ++c) {
                    const double e = trans ? a[r * k + c] : a[c * k + r];
                    double hp, ep, q;
                    two_prod(-e, v[c], hp, ep);
                    two_sum(P, hp, P, q);
                    S += q + ep;
                }
                d[r] = P + S;
            }
            trans ? solve_t(d.data(), sc) : solve(d.data(), sc);
            bool changed = false;
            for (lint r = 0; r < k; ++r) {
                const double nv = v[r] + d[r];
                if (nv != v[r]) changed = true;
                v[r] = nv;
            }
            if (!changed) break;
        }
    }
};

// Greedy linear independence of rows (modified Gram-Schmidt, twice): add(x) keeps x when its
// part orthogonal to the rows kept so far exceeds tol ||x||.
struct Indep {
    lint k;
    double tol;
    std::vector<double> Q, v;
    lint m = 0;
    Indep(lint k_, double tol_) : k(k_), tol(tol_), Q(static_cast<std::size_t>(k_ * k_)), v(static_cast<std::size_t>(k_)) {}
    bool add(const double* x) {
        double nx = 0.0;
        for (lint c = 0; c < k; ++c) nx += x[c] * x[c];
        nx = std::sqrt(nx);
        if (!(nx > 0.0) || m >= k) return false;
        for (lint c = 0; c < k; ++c) v[c] = x[c];
        for (int pass = 0; pass < 2; ++pass)
            for (lint j = 0; j < m; ++j) {
                const double* q = Q.data() + j * k;
                double d = 0.0;
                for (lint c = 0; c < k; ++c) d += q[c] * v[c];
                for (lint c = 0; c < k; ++c) v[c] -= d * q[c];
            }
        double nv = 0.0;
        for (lint c = 0; c < k; ++c) nv += v[c] * v[c];
        nv = std::sqrt(nv);
        if (!(nv > tol * nx)) return false;
        double* q = Q.data() + m * k;
        for (lint c = 0; c < k; ++c) q[c] = v[c] / nv;
        ++m;
        return true;
    }
};

// ---- exact duplicates ---------------------------------------------------------------------
inline std::uint64_t mix64(std::uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
inline std::uint64_t bits0(double v) {  // -0 hashes as +0 (they compare equal)
    if (v == 0.0) return 0;
    std::uint64_t b;
    std::memcpy(&b, &v, 8);
    return b;
}

// rep[i]: the first row (smallest index) equal to row i in y and every packed column.
// Rows hash into buckets by their hash's top bits (a stable counting sort, so each bucket
// holds its rows in row order); each bucket is deduplicated with its own table.
std::vector<lint> duplicates(lint n, lint k, const double* X, const double* y) {
    std::vector<std::uint64_t> h(static_cast<std::size_t>(n));
    for_row_chunks(n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            std::uint64_t z = mix64(bits0(y[i]));
            const double* x = X + i * k;
            for (lint c = 0; c < k; ++c) z = mix64(z ^ bits0(x[c]));
            h[i] = z;
        }
    });
    int bb = 0;
    while (bb < 14 && (lint(1) << bb) * 1024 < n) ++bb;
    const lint nbk = lint(1) << bb;
    auto bucket = [&](lint i) -> lint { return bb == 0 ? 0 : static_cast<lint>(h[i] >> (64 - bb)); };
    const lint nc = row_chunks(n);
    std::vector<lint> cnt(static_cast<std::size_t>(nc * nbk), 0);
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        lint* cc = cnt.data() + c * nbk;
        for (lint i = i0; i < i1; ++i) ++cc[bucket(i)];
    });
    std::vector<lint> start(static_cast<std::size_t>(nbk + 1), 0);
    {
        lint run = 0;
        for (lint b = 0; b < nbk; ++b) {
            start[b] = run;
            for (lint c = 0; c < nc; ++c) {
                const lint v = cnt[c * nbk + b];
                cnt[c * nbk + b] = run;
                run += v;
            }
        }
        start[nbk] = run;
    }
    std::vector<lint> order(static_cast<std::size_t>(n));
    for_row_chunks(n, [&](lint c, lint i0, lint i1) {
        lint* cc = cnt.data() + c * nbk;
        for (lint i = i0; i < i1; ++i) order[cc[bucket(i)]++] = i;
    });
    std::vector<lint> rep(static_cast<std::size_t>(n));
    auto same = [&](lint a, lint b) {
        if (y[a] != y[b]) return false;
        const double* xa = X + a * k;
        const double* xb = X + b * k;
        for (lint c = 0; c < k; ++c)
            if (xa[c] != xb[c]) return false;
        return true;
    };
    auto one_bucket = [&](lint b, std::vector<lint>& table) {
        const lint s0 = start[b], s1 = start[b + 1], m = s1 - s0;
        if (m == 0) return;
        lint sz = 1;
        while (sz < 2 * m) sz <<= 1;
        table.assign(static_cast<std::size_t>(sz), -1);
        for (lint t = s0; t < s1; ++t) {
            const lint i = order[t];
            lint slot = static_cast<lint>(mix64(h[i]) & static_cast<std::uint64_t>(sz - 1));
            for (;;) {
                const lint j = table[slot];
                if (j < 0) {
                    table[slot] = i;
                    rep[i] = i;
                    break;
                }
                if (h[j] == h[i] && same(j, i)) {
                    rep[i] = j;
                    break;
                }
                slot = (slot + 1) & (sz - 1);
            }
        }
    };
    const lint ntask = std::min<lint>(nbk, 64);
    auto task = [&](lint t) {
        std::vector<lint> table;
        for (lint b = t * nbk / ntask; b < (t + 1) * nbk / ntask; ++b) one_bucket(b, table);
    };
    if (ntask >= 2 && n >= kRowParallelMin) {
        parallel::parallel_for(ntask, [&](std::int64_t t) { task(static_cast<lint>(t)); });
    } else {
        for (lint t = 0; t < ntask; ++t) task(t);
    }
    return rep;
}

// ---- the interior point method (G60) -----------------------------------------------------
// lp_fnm (Koenker's Frisch-Newton; quantreg's rqfnb) on min c^T x s.t. A x = b, 0 <= x <= u
// with A = X^T, c = -y, b = (1 - q) X^T m, u = m: x is the dual a, y_lp = -beta. Returns
// the iterations; beta (in: the least squares start, out: the last iterate's coefficients).
int ipm(const QP& P, double tau, int max_iter, double tol, double* beta) {
    const lint n = P.n, k = P.k;
    const double* m = P.m;
    std::vector<double> x(static_cast<std::size_t>(n)), s(x.size()), z(x.size()), w(x.size()), q(x.size()), r(x.size()),
        dx(x.size()), dz(x.size()), dw(x.size()), xinv(x.size()), sinv(x.size());
    std::vector<double> yl(static_cast<std::size_t>(k)), b(yl.size()), dy(yl.size()), M(static_cast<std::size_t>(k * k));
    for (lint c = 0; c < k; ++c) {
        yl[c] = -beta[c];
        b[c] = (1.0 - tau) * P.sall[c].value();
    }
    // the dual slacks z - w = c - A^T y_lp = X beta - y: z = max(r, 0) + delta, w = max(-r, 0) +
    // delta with delta kRZero times the mean |r|, so that no slack starts at (or within rounding
    // of) 0 and q = 1/(z/x + w/s) stays bounded (lp_fnm's z = max(r, 0) with r nudged where it
    // is exactly 0 leaves tiny residuals, the shifted start's quantile row among them, with
    // q ~ 1e15 and a singular Newton matrix, and the nudge leaves the start dual infeasible)
    std::vector<double> asum;
    chunk_sum(n, 1, asum, [&](lint i0, lint i1, double* acc) {
        std::vector<double> av(static_cast<std::size_t>(i1 - i0));
        dots(P, i0, i1, beta, nullptr, av.data(), nullptr);
        double a = 0.0;
        for (lint i = i0; i < i1; ++i) {
            r[i] = av[i - i0] - P.y[i];
            a += std::fabs(r[i]);
        }
        acc[0] = a;
    });
    const double delta = asum[0] > 0.0 ? kRZero * asum[0] / static_cast<double>(n) : kRZero;
    for_row_chunks(n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            const double ri = r[i];
            x[i] = (1.0 - tau) * m[i];
            s[i] = tau * m[i];
            z[i] = (ri > 0.0 ? ri : 0.0) + delta;
            w[i] = (ri < 0.0 ? -ri : 0.0) + delta;
        }
    });
    // gap = c^T x - y_lp^T b + w^T u; the dual objective of the quantile regression y^T lambda,
    // lambda = x - (1 - q) m (the scale of the relative gap)
    auto gaps = [&](double& gap, double& dual) {
        std::vector<double> g;
        chunk_sum(n, 2, g, [&](lint i0, lint i1, double* acc) {
            double a0 = 0.0, a1 = 0.0;
            for (lint i = i0; i < i1; ++i) {
                a0 += -P.y[i] * x[i] + w[i] * m[i];
                a1 += P.y[i] * (x[i] - (1.0 - tau) * m[i]);
            }
            acc[0] = a0;
            acc[1] = a1;
        });
        double yb = 0.0;
        for (lint c = 0; c < k; ++c) yb += yl[c] * b[c];
        gap = g[0] - yb;
        dual = g[1];
    };
    double gap, dual;
    gaps(gap, dual);
    std::vector<double> part, sums, mins;
    const lint slot = k * k + k + 1;
    int it = 0;
    while (it < max_iter && std::isfinite(gap) && !(gap <= tol * std::fabs(dual))) {
        // affine (predictor) direction: (A Q A^T) dy = A Q r, Q = diag(1/(z/x + w/s)), r = z - w
        const lint nc = row_chunks(n);
        part.assign(static_cast<std::size_t>(nc * slot), 0.0);
        for_row_chunks(n, [&](lint ci, lint i0, lint i1) {
            const lint len = i1 - i0;
            std::vector<double> A(static_cast<std::size_t>(len * k)), v(static_cast<std::size_t>(len));
            double mu = 0.0;
            for (lint i = i0; i < i1; ++i) {
                xinv[i] = 1.0 / x[i];
                sinv[i] = 1.0 / s[i];
                q[i] = 1.0 / (z[i] * xinv[i] + w[i] * sinv[i]);
                r[i] = z[i] - w[i];
                mu += z[i] * x[i] + w[i] * s[i];
                const double sq = std::sqrt(q[i]);
                const double* xi = P.row(i);
                double* ai = A.data() + (i - i0) * k;
                for (lint c = 0; c < k; ++c) ai[c] = sq * xi[c];
                v[i - i0] = sq * r[i];
            }
            double* G = part.data() + ci * slot;
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, k, len, 1.0, A.data(), k, 0.0, G, k);
            cblas_dgemv(CblasColMajor, CblasNoTrans, k, len, 1.0, A.data(), k, v.data(), 1, 0.0, G + k * k, 1);
            G[k * k + k] = mu;
        });
        sums.assign(static_cast<std::size_t>(slot), 0.0);
        for (lint ci = 0; ci < nc; ++ci)
            for (lint e = 0; e < slot; ++e) sums[e] += part[ci * slot + e];
        for (lint e = 0; e < k * k; ++e) M[e] = sums[e];
        std::vector<double> rhs(sums.begin() + k * k, sums.begin() + k * k + k);
        double mu = sums[k * k + k];
        // the Newton matrix is ill-conditioned (q spans many decades): the solves are
        // refined against the assembled matrix, so that X^T dx = 0 holds to rounding and the
        // iterates stay primal feasible (else the gap stalls at the drift)
        std::vector<double> Ms(M);
        for (lint a = 0; a < k; ++a)
            for (lint c = a + 1; c < k; ++c) Ms[a * k + c] = Ms[c * k + a];  // the full symmetric matrix
        if (linalg::lapack::potrf('U', k, M.data(), k) != 0) break;
        auto newton = [&](const std::vector<double>& h, std::vector<double>& v) {
            v = h;
            linalg::lapack::potrs('U', k, 1, M.data(), k, v.data(), k);
            std::vector<double> e(static_cast<std::size_t>(k));
            for (int pass = 0; pass < 2; ++pass) {
                for (lint a = 0; a < k; ++a) {
                    double t = h[a];
                    for (lint c = 0; c < k; ++c) t -= Ms[c * k + a] * v[c];
                    e[a] = t;
                }
                linalg::lapack::potrs('U', k, 1, M.data(), k, e.data(), k);
                for (lint a = 0; a < k; ++a) v[a] += e[a];
            }
        };
        newton(rhs, dy);
        bool finite = true;
        for (lint c = 0; c < k; ++c) finite = finite && std::isfinite(dy[c]);
        if (!finite) break;
        // the directions with the step lengths to the boundary (min over each block: exact, any
        // order), in one pass; `corr` adds Mehrotra's corrector terms (from the predictor's
        // dx, dz, dw still in place) with the centering mu
        const lint ncc = row_chunks(n);
        std::vector<double> mn(static_cast<std::size_t>(ncc * 4));
        std::vector<double>&dxv = dx, &dzv = dz, &dwv = dw, &xv = x, &sv = s, &zv = z, &wv = w, &qv = q, &rv = r,
                            &xinvv = xinv, &sinvv = sinv;
        auto direction = [&](bool corr, double mu_c, double& fp, double& fd) {
            std::fill(mn.begin(), mn.end(), kInf);
            for_row_chunks(n, [&](lint ci, lint i0, lint i1) {
                double* o = mn.data() + ci * 4;
                std::vector<double> av(static_cast<std::size_t>(i1 - i0));
                dots(P, i0, i1, dy.data(), nullptr, av.data(), nullptr);
                double* __restrict dx = dxv.data();
                double* __restrict dz = dzv.data();
                double* __restrict dw = dwv.data();
                const double* __restrict x = xv.data();
                const double* __restrict s = sv.data();
                const double* __restrict z = zv.data();
                const double* __restrict w = wv.data();
                const double* __restrict q = qv.data();
                const double* __restrict r = rv.data();
                const double* __restrict xinv = xinvv.data();
                const double* __restrict sinv = sinvv.data();
                double o0 = kInf, o1 = kInf, o2 = kInf, o3 = kInf;
                for (lint i = i0; i < i1; ++i) {
                    const double acc = av[i - i0];
                    double ndx, ndz, ndw;
                    if (corr) {
                        const double dxdz = dx[i] * dz[i], dsdw = -dx[i] * dw[i];
                        const double ct = dxdz - dsdw - mu_c * (xinv[i] - sinv[i]);
                        ndx = q[i] * (acc - r[i] - ct);
                        ndz = mu_c * xinv[i] - z[i] - z[i] * ndx * xinv[i] - dxdz;
                        ndw = mu_c * sinv[i] - w[i] + w[i] * ndx * sinv[i] - dsdw;
                    } else {
                        ndx = q[i] * (acc - r[i]);
                        ndz = -z[i] * (ndx * xinv[i] + 1.0);
                        ndw = -w[i] * (-ndx * sinv[i] + 1.0);
                    }
                    dx[i] = ndx;
                    dz[i] = ndz;
                    dw[i] = ndw;
                    // the ratios to the boundary, branch-free (ds = -dx)
                    const double tx = (ndx < 0.0 ? -x[i] : s[i]) / ndx;
                    const double tz = -z[i] / ndz, tw = -w[i] / ndw;
                    o0 = std::min(o0, ndx < 0.0 ? tx : kInf);
                    o1 = std::min(o1, ndx > 0.0 ? tx : kInf);
                    o2 = std::min(o2, ndz < 0.0 ? tz : kInf);
                    o3 = std::min(o3, ndw < 0.0 ? tw : kInf);
                }
                o[0] = o0, o[1] = o1, o[2] = o2, o[3] = o3;
            });
            double f[4] = {kInf, kInf, kInf, kInf};
            for (lint ci = 0; ci < ncc; ++ci)
                for (int e = 0; e < 4; ++e) f[e] = std::min(f[e], mn[ci * 4 + e]);
            fp = std::min(kStep * std::min(f[0], f[1]), 1.0);
            fd = std::min(kStep * std::min(f[2], f[3]), 1.0);
        };
        double fp, fd;
        direction(false, 0.0, fp, fd);
        if (std::min(fp, fd) < 1.0) {
            // Mehrotra's corrector with the centering mu (g/mu)^3 / (2n)
            std::vector<double> gs;
            chunk_sum(n, 1, gs, [&](lint i0, lint i1, double* acc) {
                double a = 0.0;
                for (lint i = i0; i < i1; ++i)
                    a += (z[i] + fd * dz[i]) * (x[i] + fp * dx[i]) + (w[i] + fd * dw[i]) * (s[i] - fp * dx[i]);
                acc[0] = a;
            });
            mu = mu * std::pow(gs[0] / mu, 3.0) / (2.0 * static_cast<double>(n));
            std::vector<double> rhs2;
            chunk_sum(n, k, rhs2, [&](lint i0, lint i1, double* acc) {
                // q (dx dz - ds dw - xi) per row, then X^T of it by BLAS
                std::vector<double> t(static_cast<std::size_t>(i1 - i0));
                for (lint i = i0; i < i1; ++i) {
                    const double dxdz = dx[i] * dz[i], dsdw = -dx[i] * dw[i];
                    t[i - i0] = q[i] * (dxdz - dsdw - mu * (xinv[i] - sinv[i]));
                }
                cblas_dgemv(CblasColMajor, CblasNoTrans, k, i1 - i0, 1.0, P.row(i0), k, t.data(), 1, 0.0, acc, 1);
            });
            std::vector<double> h2(static_cast<std::size_t>(k));
            for (lint c = 0; c < k; ++c) h2[c] = rhs[c] + rhs2[c];
            newton(h2, dy);
            finite = true;
            for (lint c = 0; c < k; ++c) finite = finite && std::isfinite(dy[c]);
            if (!finite) break;
            direction(true, mu, fp, fd);
        }
        if (!(fp > 0.0) || !(fd >= 0.0) || !std::isfinite(fp) || !std::isfinite(fd)) break;
        for (lint c = 0; c < k; ++c) yl[c] += fd * dy[c];
        std::vector<double> g;
        chunk_sum(n, 2, g, [&](lint i0, lint i1, double* acc) {
            double a0 = 0.0, a1 = 0.0;
            for (lint i = i0; i < i1; ++i) {
                const double xn = x[i] + fp * dx[i], wn = w[i] + fd * dw[i];
                s[i] -= fp * dx[i];
                z[i] += fd * dz[i];
                x[i] = xn;
                w[i] = wn;
                a0 += -P.y[i] * xn + wn * m[i];
                a1 += P.y[i] * (xn - (1.0 - tau) * m[i]);
            }
            acc[0] = a0;
            acc[1] = a1;
        });
        double yb = 0.0;
        for (lint c = 0; c < k; ++c) yb += yl[c] * b[c];
        gap = g[0] - yb;
        dual = g[1];
        ++it;
    }
    for (lint c = 0; c < k; ++c) beta[c] = -yl[c];
    return it;
}

// ---- the simplex state -------------------------------------------------------------------
enum Allow : std::uint8_t { aPlus = 1, aMinus = 2 };  // face phase: the directions a basis row may leave by

struct State {
    bool valid = false;                // B is a basis (a warm start may use it)
    std::vector<lint> B;               // the basis rows (k)
    std::vector<std::int8_t> side;     // per row: 0 basis, +1 residual >= 0, -1 residual <= 0
    std::vector<std::uint8_t> allow;   // per basis position (face phase)
};

struct Work {
    std::vector<double> r;    // residuals at the current vertex (0 on the basis and snapped rows)
    std::vector<double> beta, lam;
    std::vector<Sum2> neg;    // sum of m_i x_i over the rows of side -1 (kept up to date by the simplex)
    std::vector<double> av, vv;   // row products (scratch of the passes)
    Basis F;
};

// Factors the basis and solves beta (refined). bn: ||beta||_inf.
bool solve_basis(const QP& P, const State& S, Work& W, Scratch& sc) {
    const lint k = P.k;
    if (!W.F.factor(P, S.B, sc)) return false;
    std::vector<double> yb(static_cast<std::size_t>(k));
    for (lint a = 0; a < k; ++a) yb[a] = P.y[S.B[a]];
    W.beta.resize(static_cast<std::size_t>(k));
    W.F.refined(yb.data(), nullptr, W.beta.data(), false, sc);
    return true;
}

inline double inf_norm(const std::vector<double>& v) {
    double m = 0.0;
    for (double e : v) m = std::max(m, std::fabs(e));
    return m;
}

// The snapping tolerance of a residual: (k + 16) u (|y_i| + ||x_i||_1 ||beta||_inf). A
// residual within it is taken as exactly 0 and keeps its side (residuals are plain double;
// they decide signs only beyond it).
inline double snap_tol(const QP& P, lint i, double bn) {
    return (static_cast<double>(P.k) + 16.0) * kU * (std::fabs(P.y[i]) + P.l1[i] * bn);
}

// The multipliers lambda: X_B^T lambda = -g, g = q (sum over nonbasic rows of m x) - neg
// (in double-double), solved refined.
void multipliers(const QP& P, double tau, const State& S, Work& W, Scratch& sc) {
    const lint k = P.k;
    std::vector<Sum2> sb(static_cast<std::size_t>(k));
    for (lint a = 0; a < k; ++a) {
        const double* xb = P.row(S.B[a]);
        for (lint c = 0; c < k; ++c) sb[c].add_prod(P.m[S.B[a]], xb[c]);
    }
    std::vector<double> gh(static_cast<std::size_t>(k)), gl(gh.size());
    for (lint c = 0; c < k; ++c) {
        Sum2 t;
        t.add_prod(-tau, P.sall[c].s);
        t.add_prod(-tau, P.sall[c].c);
        t.add_prod(tau, sb[c].s);
        t.add_prod(tau, sb[c].c);
        t.add(W.neg[c]);
        gh[c] = t.s;
        gl[c] = t.c;
    }
    W.lam.resize(static_cast<std::size_t>(k));
    W.F.refined(gh.data(), gl.data(), W.lam.data(), true, sc);
}

// The basis solve, every row's residual and side, and with want_lam neg from scratch and the
// multipliers.
bool evaluate(const QP& P, double tau, State& S, Work& W, bool want_lam, Scratch& sc) {
    const lint n = P.n, k = P.k;
    if (!solve_basis(P, S, W, sc)) return false;
    const double bn = inf_norm(W.beta);
    const double* beta = W.beta.data();
    chunk_sum2(n, want_lam ? k : 0, W.neg, [&](lint i0, lint i1, Sum2* acc) {
        std::vector<double> av(static_cast<std::size_t>(i1 - i0));
        dots(P, i0, i1, beta, nullptr, av.data(), nullptr);
        for (lint i = i0; i < i1; ++i) {
            if (S.side[i] == 0) {
                W.r[i] = 0.0;
                continue;
            }
            double ri = P.y[i] - av[i - i0];
            if (std::fabs(ri) <= snap_tol(P, i, bn)) ri = 0.0;
            else S.side[i] = ri > 0.0 ? 1 : -1;
            W.r[i] = ri;
            if (want_lam && S.side[i] < 0) {
                const double* xi = P.row(i);
                for (lint c = 0; c < k; ++c) acc[c].add_prod(P.m[i], xi[c]);
            }
        }
    });
    if (want_lam) multipliers(P, tau, S, W, sc);
    return true;
}

// neg += sign m_i x_i
inline void neg_update(const QP& P, Work& W, lint i, double sign) {
    const double* xi = P.row(i);
    for (lint c = 0; c < P.k; ++c) W.neg[c].add_prod(sign * P.m[i], xi[c]);
}
// Row i's side becomes s (neg follows).
inline void set_side(const QP& P, State& S, Work& W, lint i, std::int8_t s, bool track) {
    if (track && S.side[i] != s) {
        if (S.side[i] < 0) neg_update(P, W, i, -1.0);
        if (s < 0) neg_update(P, W, i, 1.0);
    }
    S.side[i] = s;
}

// The reduced costs of leaving basis position a in direction sigma (+1: its residual turns
// negative, -1: positive): lambda_a + m (1 - q) and m q - lambda_a; and the tolerance under
// which they count as 0.
inline double rcost(const QP& P, double tau, const Work& W, const State& S, lint a, int sigma) {
    const double ma = P.m[S.B[a]];
    return sigma > 0 ? W.lam[a] + ma * (1.0 - tau) : ma * tau - W.lam[a];
}
inline double rtolr(const QP& P, const Work& W, const State& S, lint a) {
    return kDualTol * kU * std::max(P.m[S.B[a]], std::fabs(W.lam[a]));
}

struct Blk {
    double t;
    lint i;
    double w;
};
inline bool blk_less(const Blk& a, const Blk& b) { return a.t < b.t || (a.t == b.t && a.i < b.i); }

// The breakpoints along d (X_B d = sigma e_a): the nonbasic rows whose residual moves towards
// and through 0, t_i = r_i / (x_i d) >= 0, weight m_i |x_i d|; |x_i d| below
// (k + 16) u ||x_i||_1 ||d||_inf is taken as 0 (the row does not move). With `recompute` the
// residuals at beta are computed in the same pass (their sides updated, neg following);
// otherwise W.r is used.
void blockers(const QP& P, State& S, Work& W, const double* d, bool recompute, std::vector<Blk>& out) {
    const lint n = P.n, k = P.k;
    double dn = 0.0;
    for (lint c = 0; c < k; ++c) dn = std::max(dn, std::fabs(d[c]));
    const double tv = (static_cast<double>(k) + 16.0) * kU * dn;
    const double bn = inf_norm(W.beta);
    const double* beta = W.beta.data();
    const lint nc = row_chunks(n);
    out.resize(static_cast<std::size_t>(n));
    W.av.resize(static_cast<std::size_t>(n));
    W.vv.resize(static_cast<std::size_t>(n));
    std::vector<lint> cnt(static_cast<std::size_t>(nc), 0);
    std::vector<std::vector<std::pair<lint, std::int8_t>>> changes(static_cast<std::size_t>(nc));
    for_row_chunks(n, [&](lint ci, lint i0, lint i1) {
        Blk* o = out.data() + i0;
        lint no = 0;
        double* av = W.av.data() + i0;
        double* vv = W.vv.data() + i0;
        if (recompute) dots(P, i0, i1, beta, d, av, vv);
        else dots(P, i0, i1, d, nullptr, vv, nullptr);
        for (lint i = i0; i < i1; ++i) {
            const int sd0 = S.side[i];
            const double v = vv[i - i0];
            double ri;
            int sd = sd0;
            if (recompute && sd0 != 0) {
                ri = P.y[i] - av[i - i0];
                if (std::fabs(ri) <= snap_tol(P, i, bn)) ri = 0.0;
                else sd = ri > 0.0 ? 1 : -1;
                W.r[i] = ri;
                if (sd != sd0) changes[static_cast<std::size_t>(ci)].push_back({i, static_cast<std::int8_t>(sd)});
            } else {
                ri = W.r[i];
            }
            // a breakpoint (branch-free: written always, kept when it is one)
            const double sv = static_cast<double>(sd) * v;
            const bool keep = sv > tv * P.l1[i];
            const double t = ri / v;
            o[no] = {t > 0.0 ? t : 0.0, i, P.m[i] * sv};
            no += keep ? 1 : 0;
        }
        cnt[static_cast<std::size_t>(ci)] = no;
    });
    for (auto& ch : changes)
        for (auto [i, s] : ch) set_side(P, S, W, i, s, true);
    std::size_t m = 0;
    for (lint ci = 0; ci < nc; ++ci) {
        const std::size_t c = static_cast<std::size_t>(cnt[ci]);
        if (c) std::memmove(out.data() + m, out.data() + ci * kRowChunk, c * sizeof(Blk));
        m += c;
    }
    out.resize(m);
}

// The position (in v, after the call) of the first breakpoint in (t, row) order at which the
// running sum of the weights reaches W; v[0, p) are the ones before it (in some order).
std::size_t weighted_select(std::vector<Blk>& v, double Wt) {
    std::size_t lo = 0, hi = v.size();
    double acc = 0.0;
    if (hi > 4096) {
        // a threshold from a regular sample (its weights scaled up), a little beyond where the
        // sample's running sum reaches W; one partition at it usually holds the answer
        const std::size_t st = hi / 1024;
        std::vector<Blk> smp;
        smp.reserve(1024);
        for (std::size_t j = 0; j < hi; j += st) smp.push_back(v[j]);
        std::sort(smp.begin(), smp.end(), blk_less);
        double cum = 0.0;
        std::size_t pos = smp.size();
        for (std::size_t j = 0; j < smp.size(); ++j) {
            cum += smp[j].w * static_cast<double>(st);
            if (cum >= Wt) {
                pos = j;
                break;
            }
        }
        if (pos < smp.size()) {
            const Blk T = smp[std::min(smp.size() - 1, pos + 8 + pos / 2)];
            auto it = std::partition(v.begin(), v.end(), [&](const Blk& e) { return !blk_less(T, e); });
            const std::size_t f = static_cast<std::size_t>(it - v.begin());
            double sf = 0.0;
            for (std::size_t j = 0; j < f; ++j) sf += v[j].w;
            if (sf >= Wt) {
                hi = f;
            } else {
                acc = sf;
                lo = f;
            }
        }
    }
    while (hi - lo > 64) {
        const std::size_t mid = lo + (hi - lo) / 2;
        std::nth_element(v.begin() + lo, v.begin() + mid, v.begin() + hi, blk_less);
        double sl = 0.0;
        for (std::size_t i = lo; i < mid; ++i) sl += v[i].w;
        if (acc + sl >= Wt) {
            hi = mid;
        } else {
            acc += sl;
            lo = mid;
        }
    }
    std::sort(v.begin() + lo, v.begin() + hi, blk_less);
    for (std::size_t p = lo; p < hi; ++p) {
        acc += v[p].w;
        if (acc >= Wt) return p;
    }
    return hi - 1;
}

std::uint64_t state_hash(const State& S) {
    std::vector<lint> b(S.B);
    std::sort(b.begin(), b.end());
    std::uint64_t h = 0x12345678ULL;
    for (lint v : b) h = mix64(h ^ static_cast<std::uint64_t>(v));
    const std::size_t n = S.side.size();
    for (std::size_t i = 0; i < n; i += 8) {
        std::uint64_t word = 0;
        for (std::size_t j = i; j < std::min(n, i + 8); ++j)
            word = (word << 8) | static_cast<std::uint8_t>(S.side[j]);
        h = mix64(h ^ word);
    }
    return h;
}

// Main simplex: pivots until the certificate holds. Returns true when it does (W.lam is the
// certificate's, from residuals and sums recomputed from scratch), false at the pivot cap or
// a singular basis. `pivots` counts. Between pivots neg is updated by the rows whose side
// changes and the residuals are recomputed in the ratio test's pass.
bool simplex(const QP& P, double tau, State& S, Work& W, lint cap, lint& pivots, Scratch& sc) {
    const lint k = P.k;
    bool bland = false, fresh = true;
    std::vector<std::uint64_t> seen;
    std::vector<Blk> blk;
    std::vector<double> d(static_cast<std::size_t>(k));
    if (!evaluate(P, tau, S, W, true, sc)) return false;
    for (;;) {
        lint ba = -1;
        int bs = 0;
        double best = 0.0;
        lint bvar = std::numeric_limits<lint>::max();
        for (lint a = 0; a < k; ++a) {
            const double tl = rtolr(P, W, S, a);
            for (int sg : {-1, 1}) {
                const double rc = rcost(P, tau, W, S, a, sg);
                if (!(rc < -tl)) continue;
                const lint var = 2 * S.B[a] + (sg > 0 ? 1 : 0);
                if (bland ? var < bvar : (rc < best || (rc == best && var < bvar))) {
                    best = rc;
                    bvar = var;
                    ba = a;
                    bs = sg;
                }
            }
        }
        if (ba < 0) {
            if (fresh) return true;
            // confirm the certificate with everything recomputed
            if (!evaluate(P, tau, S, W, true, sc)) return false;
            fresh = true;
            continue;
        }
        if (pivots >= cap) return false;
        std::fill(d.begin(), d.end(), 0.0);
        d[ba] = static_cast<double>(bs);
        W.F.solve(d.data(), sc);
        blockers(P, S, W, d.data(), !fresh, blk);
        fresh = false;
        if (blk.empty()) return false;  // a ray: impossible for a full rank design
        const double rc = rcost(P, tau, W, S, ba, bs);
        std::size_t p = weighted_select(blk, -rc);
        lint enter = blk[p].i;
        bool flips = true;
        if (!(blk[p].t > 0.0)) {
            // degenerate: no move. A repeated state switches to Bland's rule (the leaving
            // row: the smallest blocking row at t = 0) until the objective decreases.
            const std::uint64_t hs = state_hash(S);
            if (std::find(seen.begin(), seen.end(), hs) != seen.end()) bland = true;
            seen.push_back(hs);
            if (bland) {
                lint mi = std::numeric_limits<lint>::max();
                for (const Blk& b : blk)
                    if (!(b.t > 0.0) && b.i < mi) mi = b.i;
                enter = mi;
                flips = false;
            }
        } else {
            bland = false;
            seen.clear();
        }
        if (flips)
            for (std::size_t j = 0; j < p; ++j) set_side(P, S, W, blk[j].i, static_cast<std::int8_t>(-S.side[blk[j].i]), true);
        set_side(P, S, W, S.B[ba], static_cast<std::int8_t>(bs > 0 ? -1 : 1), true);
        set_side(P, S, W, enter, 0, true);
        S.B[ba] = enter;
        ++pivots;
        if (!solve_basis(P, S, W, sc)) return false;
        multipliers(P, tau, S, W, sc);
    }
}

// The lexicographic sign of d: the sign of its first component beyond kLexTol u ||d||_inf.
int lexsign(const double* d, lint k) {
    double dn = 0.0;
    for (lint c = 0; c < k; ++c) dn = std::max(dn, std::fabs(d[c]));
    for (lint c = 0; c < k; ++c)
        if (std::fabs(d[c]) > kLexTol * kU * dn) return d[c] > 0.0 ? 1 : -1;
    return 0;
}

// Pivots inside the optimal set F (the sign constraints of the certificate and the allowed
// directions of the basis rows) towards its lexicographic minimum (dir = 1) or maximum
// (dir = -1), Bland's rule. moved: a pivot of positive length was taken; with stop_on_move
// the walk ends there. Returns false at the cap or a singular basis.
bool face(const QP& P, double tau, State& S, Work& W, int dir, bool stop_on_move, bool& moved, lint cap, lint& pivots,
          Scratch& sc) {
    const lint k = P.k;
    std::vector<Blk> blk;
    std::vector<double> d(static_cast<std::size_t>(k)), dbest(d.size());
    for (;;) {
        if (!evaluate(P, tau, S, W, false, sc)) return false;
        lint ba = -1;
        int bs = 0;
        lint bvar = std::numeric_limits<lint>::max();
        for (lint a = 0; a < k; ++a) {
            for (int sg : {-1, 1}) {
                if (!(S.allow[a] & (sg > 0 ? aPlus : aMinus))) continue;
                const lint var = 2 * S.B[a] + (sg > 0 ? 1 : 0);
                if (var >= bvar) continue;
                std::fill(d.begin(), d.end(), 0.0);
                d[a] = static_cast<double>(sg);
                W.F.solve(d.data(), sc);
                if (dir * lexsign(d.data(), k) < 0) {
                    bvar = var;
                    ba = a;
                    bs = sg;
                    dbest = d;
                }
            }
        }
        if (ba < 0) return true;
        if (pivots >= cap) return false;
        blockers(P, S, W, dbest.data(), false, blk);
        if (blk.empty()) return true;  // a ray of F: impossible for a full rank design
        std::size_t p = 0;
        for (std::size_t j = 1; j < blk.size(); ++j)
            if (blk_less(blk[j], blk[p])) p = j;
        const lint enter = blk[p].i;
        if (blk[p].t > 0.0) {
            moved = true;
            if (stop_on_move) return true;
        }
        const std::int8_t es = S.side[enter];
        S.side[S.B[ba]] = static_cast<std::int8_t>(bs > 0 ? -1 : 1);
        S.side[enter] = 0;
        S.B[ba] = enter;
        // the entering row may leave again only towards its side's constraint
        S.allow[ba] = es > 0 ? aMinus : aPlus;
        ++pivots;
    }
}

// The canonical basis of the current vertex: the first k linearly independent rows (in row
// order) among those with zero residual; then beta from it.
bool canonical(const QP& P, double tau, State& S, Work& W, Scratch& sc) {
    const lint n = P.n, k = P.k;
    Indep ind(k, P.itol);
    std::vector<lint> nb;
    nb.reserve(static_cast<std::size_t>(k));
    for (lint i = 0; i < n && static_cast<lint>(nb.size()) < k; ++i)
        if (S.side[i] == 0 || W.r[i] == 0.0)
            if (ind.add(P.row(i))) nb.push_back(i);
    if (static_cast<lint>(nb.size()) < k) {
        nb = S.B;
        std::sort(nb.begin(), nb.end());
    }
    for (lint a = 0; a < k; ++a)
        if (S.side[S.B[a]] == 0) S.side[S.B[a]] = 1;
    for (lint i : nb) S.side[i] = 0;
    S.B = nb;
    return evaluate(P, tau, S, W, false, sc);
}

// ---- one quantile ------------------------------------------------------------------------
struct QOpts {
    int max_iter;
    double tol;
};

struct QOut {
    int iters = 0;
    lint pivots = 0;
    bool converged = false, nonunique = false;
};

// The crossover basis from coefficients beta: rows by |r_i| (ties by row), the first k
// linearly independent (by the tolerance itol). False when fewer than k are.
bool crossover(const QP& P, const double* beta, State& S, double itol) {
    const lint n = P.n, k = P.k;
    std::vector<std::pair<double, lint>> ord(static_cast<std::size_t>(n));
    S.side.assign(static_cast<std::size_t>(n), 1);
    for_row_chunks(n, [&](lint, lint i0, lint i1) {
        for (lint i = i0; i < i1; ++i) {
            const double* xi = P.row(i);
            double a = 0.0;
            for (lint c = 0; c < k; ++c) a += xi[c] * beta[c];
            const double ri = P.y[i] - a;
            S.side[i] = ri >= 0.0 ? 1 : -1;
            ord[i] = {std::fabs(ri), i};
        }
    });
    Indep ind(k, itol);
    S.B.clear();
    std::size_t done = 0, want = static_cast<std::size_t>(std::min<lint>(n, 4 * k + 8));
    while (static_cast<lint>(S.B.size()) < k && done < ord.size()) {
        if (want > done) {
            std::nth_element(ord.begin() + done, ord.begin() + (want - 1), ord.end());
            std::sort(ord.begin() + done, ord.begin() + want);
        }
        for (std::size_t j = done; j < want && static_cast<lint>(S.B.size()) < k; ++j)
            if (ind.add(P.row(ord[j].second))) S.B.push_back(ord[j].second);
        done = want;
        want = std::min(ord.size(), 2 * want);
    }
    if (static_cast<lint>(S.B.size()) < k) return false;
    for (lint i : S.B) S.side[i] = 0;
    return true;
}

lint pivot_cap(const QP& P) { return 4 * P.n + 64 * P.k + 1000; }

// Solves one quantile: from the interior point method (cold) or the state's basis (warm), to
// the canonical vertex of the rule of G63 in S and W.
QOut solve_q(const QP& P, double tau, State& S, Work& W, bool warm, const QOpts& op, const double* start, Scratch& sc) {
    QOut out;
    const lint k = P.k;
    W.r.assign(static_cast<std::size_t>(P.n), 0.0);
    if (!warm || !S.valid) {
        std::vector<double> b(start, start + k);
        if (P.icept) {
            // the least squares fit moved to the q-quantile of its residuals (weighted by m)
            std::vector<Blk> v(static_cast<std::size_t>(P.n));
            std::vector<double> av(static_cast<std::size_t>(P.n));
            double tw = 0.0;
            for_row_chunks(P.n, [&](lint, lint i0, lint i1) { dots(P, i0, i1, b.data(), nullptr, av.data() + i0, nullptr); });
            for (lint i = 0; i < P.n; ++i) {
                v[i] = {P.y[i] - av[i], i, P.m[i]};
                tw += P.m[i];
            }
            const std::size_t j = weighted_select(v, tau * tw);
            b[0] += v[j].t;
        }
        out.iters = ipm(P, tau, op.max_iter, op.tol, b.data());
        // a column kept by G21's rule may leave fewer than k rows independent by the rows'
        // tolerance: then any nonzero orthogonal part counts
        if (!crossover(P, b.data(), S, P.itol) && !crossover(P, b.data(), S, 0.0)) {
            S.valid = false;
            W.beta.assign(static_cast<std::size_t>(k), kNaN);
            return out;
        }
    }
    S.valid = true;
    const lint cap = pivot_cap(P);
    if (!simplex(P, tau, S, W, cap, out.pivots, sc)) {
        if (!canonical(P, tau, S, W, sc)) {
            S.valid = false;
            W.beta.assign(static_cast<std::size_t>(k), kNaN);
        }
        return out;
    }
    out.converged = true;
    // the multipliers at a bound: the directions F extends in from this vertex
    S.allow.assign(static_cast<std::size_t>(k), 0);
    bool any = false;
    for (lint a = 0; a < k; ++a) {
        const double tl = rtolr(P, W, S, a);
        if (rcost(P, tau, W, S, a, 1) <= tl) S.allow[a] |= aPlus;
        if (rcost(P, tau, W, S, a, -1) <= tl) S.allow[a] |= aMinus;
        any = any || S.allow[a];
    }
    if (any) {
        bool moved = false;
        if (!face(P, tau, S, W, 1, false, moved, cap, out.pivots, sc)) out.converged = false;
        if (!moved && out.converged) {
            State T = S;
            Work V;
            V.r = W.r;
            lint pv = 0;
            bool moved2 = false;
            if (!face(P, tau, T, V, -1, true, moved2, cap, pv, sc)) out.converged = false;
            moved = moved2;
        }
        out.nonunique = moved;
    }
    if (!canonical(P, tau, S, W, sc)) {
        out.converged = false;
        S.valid = false;
        W.beta.assign(static_cast<std::size_t>(k), kNaN);
    }
    return out;
}

// ---- covariances -------------------------------------------------------------------------
// R of the QR of sqrt(f) X_K over the rows with f > 0 (kk x kk upper, row-major); false when
// it is singular or not finite.
bool weighted_r(const Design& D, const std::vector<lint>& K, const double* f, double* Rf, Scratch& sc) {
    const lint n = D.n, kk = static_cast<lint>(K.size());
    lint nf = 0;
    for (lint i = 0; i < n; ++i) {
        if (!(f[i] >= 0.0) || std::isinf(f[i])) return false;
        if (f[i] > 0.0) ++nf;
    }
    if (nf < kk) return false;
    std::vector<double> A(static_cast<std::size_t>(nf * kk)), tau(static_cast<std::size_t>(kk));
    lint row = 0;
    for (lint i = 0; i < n; ++i) {
        if (!(f[i] > 0.0)) continue;
        const double s = std::sqrt(f[i]);
        for (lint a = 0; a < kk; ++a) A[a * nf + row] = s * D.at(i, K[a]);
        ++row;
    }
    linalg::lapack::geqrf(nf, kk, A.data(), nf, tau.data(), sc);
    for (lint a = 0; a < kk; ++a)
        for (lint b = 0; b < kk; ++b) Rf[a * kk + b] = b >= a ? A[b * nf + a] : 0.0;
    for (lint a = 0; a < kk; ++a)
        if (!(Rf[a * kk + a] != 0.0) || !std::isfinite(Rf[a * kk + a])) return false;
    return true;
}

// q (1 - q) H^-1 J H^-1 with H = Rf^T Rf (X^T F X) and J = R0^T R0 (X^T X): N = R0 Rf^-1 Rf^-T,
// the covariance q (1 - q) N^T N (no product of X^T X with itself is formed).
void sandwich(lint kk, const double* R0, const double* Rf, double tau, double* V) {
    std::vector<double> N(R0, R0 + kk * kk);
    for (lint i = 0; i < kk; ++i) {
        double* m = N.data() + i * kk;
        // m Rf = r0_i: forward substitution with Rf^T
        for (lint j = 0; j < kk; ++j) {
            double s = m[j];
            for (lint l = 0; l < j; ++l) s -= m[l] * Rf[l * kk + j];
            m[j] = s / Rf[j * kk + j];
        }
        // n Rf^T = m: back substitution with Rf
        for (lint j = kk - 1; j >= 0; --j) {
            double s = m[j];
            for (lint l = j + 1; l < kk; ++l) s -= m[l] * Rf[j * kk + l];
            m[j] = s / Rf[j * kk + j];
        }
    }
    const double c = tau * (1.0 - tau);
    for (lint a = 0; a < kk; ++a)
        for (lint b = 0; b < kk; ++b) {
            double s = 0.0;
            for (lint i = 0; i < kk; ++i) s += N[i * kk + a] * N[i * kk + b];
            V[a * kk + b] = c * s;
        }
}

// R's quantile(x, p) (type 7) of v (reordered).
double quantile7(std::vector<double>& v, double p) {
    const lint n = static_cast<lint>(v.size());
    const double h = static_cast<double>(n - 1) * p;
    const lint lo = static_cast<lint>(std::floor(h));
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    const double a = v[lo];
    if (lo + 1 >= n) return a;
    const double b = *std::min_element(v.begin() + lo + 1, v.end());
    return a + (h - static_cast<double>(lo)) * (b - a);
}

// ---- the regression ----------------------------------------------------------------------
struct Opts {
    bool intercept;
    double rtol;
    int cov;
    int max_iter;
    double tol;
    std::vector<double> qs, hraw, hhalf, kerc;
};

struct Outs {
    double *params, *cov, *R, *fitted, *resid, *stats;
    std::uint8_t* kept;
    std::int64_t* basis;
};

// Packs a problem (rows duplicated merged) from row-major X (n x k) and y.
struct Packed {
    std::vector<double> X, y, m;
    std::vector<lint> map;   // original row -> packed row
    std::vector<lint> orig;  // packed row -> its first original row
    QP P;
};

void pack(Packed& pk, lint n, lint k, std::vector<double>&& X, std::vector<double>&& y, double itol) {
    std::vector<lint> rep = duplicates(n, k, X.data(), y.data());
    lint nu = 0;
    pk.map.resize(static_cast<std::size_t>(n));
    for (lint i = 0; i < n; ++i) {
        if (rep[i] == i) pk.map[i] = nu++;
        else pk.map[i] = pk.map[rep[i]];
    }
    if (nu == n) {
        pk.X = std::move(X);
        pk.y = std::move(y);
        pk.m.assign(static_cast<std::size_t>(n), 1.0);
        pk.orig.resize(static_cast<std::size_t>(n));
        for (lint i = 0; i < n; ++i) pk.orig[i] = i;
    } else {
        pk.X.resize(static_cast<std::size_t>(nu * k));
        pk.y.resize(static_cast<std::size_t>(nu));
        pk.m.assign(static_cast<std::size_t>(nu), 0.0);
        pk.orig.resize(static_cast<std::size_t>(nu));
        for (lint i = 0; i < n; ++i) {
            const lint u = pk.map[i];
            if (rep[i] == i) {
                pk.orig[u] = i;
                pk.y[u] = y[i];
                std::copy(X.begin() + i * k, X.begin() + (i + 1) * k, pk.X.begin() + u * k);
            }
            pk.m[u] += 1.0;
        }
    }
    pk.P.n = nu;
    pk.P.k = k;
    pk.P.X = pk.X.data();
    pk.P.y = pk.y.data();
    pk.P.m = pk.m.data();
    pk.P.itol = itol;
    make_sall(pk.P);
}

// A small dense quantile regression (iid's sparsity): the slope of the q = 1/2 regression of
// v on (1, t).
double median_slope(const std::vector<double>& v, const std::vector<double>& t, double itol, const QOpts& op,
                    Scratch& sc) {
    const lint n = static_cast<lint>(v.size());
    std::vector<double> X(static_cast<std::size_t>(2 * n)), y(v);
    for (lint i = 0; i < n; ++i) X[2 * i] = 1.0, X[2 * i + 1] = t[i];
    // least squares start
    double mt = 0.0, mv = 0.0;
    for (lint i = 0; i < n; ++i) mt += t[i], mv += v[i];
    mt /= static_cast<double>(n);
    mv /= static_cast<double>(n);
    double sxy = 0.0, sxx = 0.0;
    for (lint i = 0; i < n; ++i) sxy += (t[i] - mt) * (v[i] - mv), sxx += (t[i] - mt) * (t[i] - mt);
    const double b1 = sxx > 0.0 ? sxy / sxx : 0.0;
    const double start[2] = {mv - b1 * mt, b1};
    Packed pk;
    pack(pk, n, 2, std::move(X), std::move(y), itol);
    State S;
    Work W;
    const QOut o = solve_q(pk.P, 0.5, S, W, false, op, start, sc);
    if (!o.converged) return kNaN;
    return W.beta[1];
}

void fill_nan(lint n, lint k, lint L, const Outs& o) {
    std::fill(o.params, o.params + L * k, kNaN);
    std::fill(o.cov, o.cov + L * k * k, kNaN);
    std::fill(o.R, o.R + L * k * k, kNaN);
    std::fill(o.fitted, o.fitted + L * n, kNaN);
    std::fill(o.resid, o.resid + L * n, kNaN);
    std::fill(o.kept, o.kept + L * k, std::uint8_t(0));
    std::fill(o.basis, o.basis + L * k, std::int64_t(-1));
    for (lint l = 0; l < L; ++l) {
        double* st = o.stats + l * qNStat;
        std::fill(st, st + qNStat, kNaN);
        st[qRank] = 0;
        st[qIsNaN] = 1;
    }
}

int quantreg_one(const Opts& op, lint n, lint p, const std::byte* yb, std::int64_t ys, const std::byte* Xb,
                 std::int64_t xrs, std::int64_t xcs, const Outs& o, Scratch& sc) {
    const lint k = p + (op.intercept ? 1 : 0);
    const lint L = static_cast<lint>(op.qs.size());
    bool nan = false;
    std::vector<double> y(static_cast<std::size_t>(std::max<lint>(n, 1)));
    for (lint i = 0; i < n; ++i) {
        y[i] = rd(yb + i * ys);
        if (std::isnan(y[i])) nan = true;
        else if (std::isinf(y[i])) return eInfY;
    }
    Data d0{reinterpret_cast<const std::byte*>(y.data()), 8, Xb, xrs, xcs, nullptr, 0, nullptr, 0, nullptr, 0};
    if (scan_x(d0, n, p, nan)) return eInfX;
    if (nan) {
        fill_nan(n, k, L, o);
        return 0;
    }
    Design D{n, p, k, op.intercept, Xb, xrs, xcs};
    // ---- 1. least squares: the aliased columns, R, (R^T R)^-1, the start
    std::vector<double> cov0(static_cast<std::size_t>(std::max<lint>(k * k, 1))), R0(cov0.size()),
        r0(static_cast<std::size_t>(std::max<lint>(n, 1))), f0(r0.size()), bols(static_cast<std::size_t>(std::max<lint>(k, 1)));
    std::vector<std::uint8_t> kept(bols.size());
    double st0[kNStat];
    {
        Problem pb{n, p, k, op.intercept, false, false, false, op.rtol, kNonrobust, 3, -1, 0, false};
        Dest od{bols.data(), cov0.data(), R0.data(), f0.data(), r0.data(), kept.data(), st0, nullptr};
        fit_one(pb, d0, od, sc);
    }
    std::vector<lint> K;
    for (lint c = 0; c < k; ++c)
        if (kept[c]) K.push_back(c);
    const lint kk = static_cast<lint>(K.size());
    const double dn = static_cast<double>(n), dfr = dn - static_cast<double>(kk);
    const double itol = op.rtol >= 0 ? op.rtol : static_cast<double>(std::max(n, k)) * kEps;
    // R0, (R^T R)^-1 on the kept columns
    std::vector<double> R0k(static_cast<std::size_t>(std::max<lint>(kk * kk, 1))), C0k(R0k.size());
    for (lint a = 0; a < kk; ++a)
        for (lint b = 0; b < kk; ++b) {
            R0k[a * kk + b] = b >= a ? R0[K[a] * k + K[b]] : 0.0;
            C0k[a * kk + b] = cov0[K[a] * k + K[b]];
        }
    // ---- 2. the packed problem
    Packed pk;
    if (kk > 0) {
        std::vector<double> Xk(static_cast<std::size_t>(n * kk)), yk(y.begin(), y.begin() + n);
        for_row_chunks(n, [&](lint, lint i0, lint i1) {
            for (lint i = i0; i < i1; ++i)
                for (lint a = 0; a < kk; ++a) Xk[i * kk + a] = D.at(i, K[a]);
        });
        pack(pk, n, kk, std::move(Xk), std::move(yk), itol);
        pk.P.icept = op.intercept && K[0] == 0;
    }
    std::vector<double> start(static_cast<std::size_t>(std::max<lint>(kk, 1)));
    for (lint a = 0; a < kk; ++a) start[a] = bols[K[a]];
    const QOpts qo{op.max_iter, op.tol};
    State S;
    Work W;
    std::vector<double> beta(static_cast<std::size_t>(std::max<lint>(k, 1)));
    std::vector<double> ysort(y.begin(), y.begin() + n);
    // summary.rq's eps relative to the scale of y (G67): 2^-26 max |y_i| over the fit's rows
    // (1 for y all 0), so that y scaled by c scales params and bse by c
    double ymax = 0.0;
    for (lint i = 0; i < n; ++i) ymax = std::max(ymax, std::fabs(y[i]));
    const double eps_y = kSqrtEps * (ymax > 0.0 ? ymax : 1.0);
    for (lint l = 0; l < L; ++l) {
        const double tau = op.qs[l];
        double* params = o.params + l * k;
        double* cov = o.cov + l * k * k;
        double* fitted = o.fitted + l * n;
        double* resid = o.resid + l * n;
        double* st = o.stats + l * qNStat;
        std::fill(st, st + qNStat, kNaN);
        std::copy(R0.begin(), R0.begin() + k * k, o.R + l * k * k);
        for (lint c = 0; c < k; ++c) o.kept[l * k + c] = kept[c];
        std::fill(o.basis + l * k, o.basis + (l + 1) * k, std::int64_t(-1));
        QOut qr;
        std::fill(beta.begin(), beta.end(), kNaN);
        if (kk > 0) {
            qr = solve_q(pk.P, tau, S, W, l > 0, qo, start.data(), sc);
            for (lint a = 0; a < kk; ++a) {
                beta[K[a]] = W.beta[a];
                if (S.valid) o.basis[l * k + a] = static_cast<std::int64_t>(pk.orig[S.B[a]]);
            }
        } else {
            qr.converged = true;
        }
        // the fit on every original row (Dot2)
        for_row_chunks(n, [&](lint, lint i0, lint i1) {
            for (lint i = i0; i < i1; ++i) {
                double P = 0.0, Sx = 0.0;
                for (lint c = 0; c < k; ++c) {
                    const double b = beta[c];
                    if (std::isnan(b)) continue;
                    double hp, ep, q;
                    two_prod(D.at(i, c), b, hp, ep);
                    two_sum(P, hp, P, q);
                    Sx += q + ep;
                }
                fitted[i] = P + Sx;
                double hh, e;
                two_sum(y[i], -P, hh, e);
                resid[i] = hh + (e - Sx);
            }
        });
        for (lint c = 0; c < k; ++c) params[c] = beta[c];
        std::vector<Sum2> ob;
        chunk_sum2(n, 1, ob, [&](lint i0, lint i1, Sum2* acc) {
            for (lint i = i0; i < i1; ++i) acc[0].add_prod(resid[i], resid[i] < 0.0 ? tau - 1.0 : tau);
        });
        // the restricted model of Koenker and Machado's R1: the intercept alone (its
        // minimizer, the type-1 sample quantile), or nothing without an intercept
        double c0 = 0.0;
        if (op.intercept && n > 0) {
            lint j = static_cast<lint>(std::ceil(static_cast<double>(n) * tau)) - 1;
            j = std::min(std::max<lint>(j, 0), n - 1);
            std::nth_element(ysort.begin(), ysort.begin() + j, ysort.end());
            c0 = ysort[j];
        }
        std::vector<Sum2> nb;
        chunk_sum2(n, 1, nb, [&](lint i0, lint i1, Sum2* acc) {
            for (lint i = i0; i < i1; ++i) {
                double e1, r1;
                two_sum(y[i], -c0, r1, e1);
                const double rr = r1 + e1;
                acc[0].add_prod(rr, rr < 0.0 ? tau - 1.0 : tau);
            }
        });
        // ---- the covariance
        std::fill(cov, cov + k * k, kNaN);
        double hused = kNaN;
        std::vector<double> V(static_cast<std::size_t>(std::max<lint>(kk * kk, 1)), kNaN);
        bool have = false;
        if (kk > 0 && qr.converged && dfr > 0) {
            if (op.cov == cIid) {
                hused = op.hraw[l];
                lint pz = 0;
                for (lint i = 0; i < n; ++i) pz += std::fabs(resid[i]) < eps_y;
                const lint h = std::max<lint>(kk + 1, static_cast<lint>(std::ceil(dn * op.hraw[l])));
                if (h + pz + 1 <= n) {
                    std::vector<std::pair<double, lint>> ord(static_cast<std::size_t>(n));
                    for (lint i = 0; i < n; ++i) ord[i] = {std::fabs(resid[i]), i};
                    std::nth_element(ord.begin(), ord.begin() + pz, ord.end());
                    std::nth_element(ord.begin() + pz, ord.begin() + pz + h, ord.end());
                    std::sort(ord.begin() + pz, ord.begin() + pz + h + 1);
                    std::vector<double> v(static_cast<std::size_t>(h + 1)), t(v.size());
                    for (lint j = 0; j <= h; ++j) {
                        v[j] = resid[ord[pz + j].second];
                        t[j] = static_cast<double>(pz + 1 + j) / dfr;
                    }
                    std::sort(v.begin(), v.end());
                    const double s = median_slope(v, t, static_cast<double>(h + 1) * kEps, qo, sc);
                    if (std::isfinite(s)) {
                        for (lint e = 0; e < kk * kk; ++e) V[e] = s * s * tau * (1.0 - tau) * C0k[e];
                        have = true;
                    }
                }
            } else {
                hused = op.hhalf[l];
                std::vector<double> f(static_cast<std::size_t>(n));
                bool ok = true;
                if (op.cov == cNid) {
                    std::vector<double> bh(static_cast<std::size_t>(kk)), bl(bh.size());
                    for (int sgn : {1, -1}) {
                        State T = S;
                        Work U;
                        const QOut o2 = solve_q(pk.P, tau + sgn * hused, T, U, true, qo, start.data(), sc);
                        ok = ok && o2.converged;
                        (sgn > 0 ? bh : bl) = U.beta;
                    }
                    for (lint i = 0; i < n; ++i) {
                        double dyh = 0.0;
                        for (lint a = 0; a < kk; ++a) dyh += D.at(i, K[a]) * (bh[a] - bl[a]);
                        f[i] = std::max(0.0, (2.0 * hused) / (dyh - eps_y));
                    }
                } else {
                    // Powell's kernel: h in the residuals' units, as summary.rq
                    double mean = 0.0;
                    {
                        Sum2 s;
                        for (lint i = 0; i < n; ++i) s.add(resid[i]);
                        mean = s.value() / dn;
                    }
                    Sum2 ss;
                    for (lint i = 0; i < n; ++i) {
                        const double e = resid[i] - mean;
                        ss.add_prod(e, e);
                    }
                    const double sdv = std::sqrt(ss.value() / (dn - 1.0));
                    std::vector<double> u(resid, resid + n);
                    const double q3 = quantile7(u, 0.75), q1 = quantile7(u, 0.25);
                    const double hr = op.kerc[l] * std::min(sdv, (q3 - q1) / 1.34);
                    const double c = 1.0 / std::sqrt(2.0 * M_PI);
                    for (lint i = 0; i < n; ++i) {
                        const double zz = resid[i] / hr;
                        f[i] = c * std::exp(-0.5 * zz * zz) / hr;
                    }
                }
                std::vector<double> Rf(static_cast<std::size_t>(kk * kk));
                if (ok && weighted_r(D, K, f.data(), Rf.data(), sc)) {
                    sandwich(kk, R0k.data(), Rf.data(), tau, V.data());
                    have = true;
                }
            }
        }
        if (have)
            for (lint a = 0; a < kk; ++a)
                for (lint b = 0; b < kk; ++b) cov[K[a] * k + K[b]] = V[a * kk + b];
        st[qNobs] = dn;
        st[qRank] = static_cast<double>(kk);
        st[qObjective] = ob[0].value();
        st[qNullObjective] = nb[0].value();
        st[qNIter] = qr.iters;
        st[qNPivots] = static_cast<double>(qr.pivots);
        st[qConverged] = qr.converged ? 1.0 : 0.0;
        st[qNonunique] = qr.nonunique ? 1.0 : 0.0;
        st[qIsNaN] = 0.0;
        st[qBandwidth] = hused;
    }
    return 0;
}

std::vector<std::int64_t> with(std::vector<std::int64_t> shape, bool qaxis, lint L,
                               std::initializer_list<std::int64_t> tail) {
    if (qaxis) shape.push_back(L);
    shape.insert(shape.end(), tail);
    return shape;
}

// quantreg_fit(func, y (..., n), X (..., n, p), qs [L], intercept, rtol, cov, hraw [L], hhalf [L], kerc [L],
//              max_iter, tol, qaxis)
// -> (params, cov, R, fitted, resid, kept, stats, basis): per regression and quantile (an
// axis of length L after the batch axes when qaxis). cov is the final covariance; R the least
// squares R (k x k); basis the canonical basis rows (-1 for aliased columns' places).
nb::object quantreg_fit(const std::string& func, nb::handle yo, nb::handle xo, const std::vector<double>& qs,
                        bool intercept, double rtol, int cov, const std::vector<double>& hraw,
                        const std::vector<double>& hhalf, const std::vector<double>& kerc, int max_iter, double tol,
                        bool qaxis) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X");
        Batch b({{y.batch_shape(1), y.batch_strides(1)}, {X.batch_shape(2), X.batch_strides(2)}}, fn);
        const lint n = X.shape(-2), p = X.shape(-1), k = p + (intercept ? 1 : 0);
        const lint L = static_cast<lint>(qs.size());
        Opts op{intercept, rtol, cov, max_iter, tol, qs, hraw, hhalf, kerc};
        const std::vector<std::int64_t>& bs = b.shape();
        Out po = linalg::alloc_c(with(bs, qaxis, L, {k}), DType::Float64);
        Out co = linalg::alloc_c(with(bs, qaxis, L, {k, k}), DType::Float64);
        Out ro = linalg::alloc_c(with(bs, qaxis, L, {k, k}), DType::Float64);
        Out fo = linalg::alloc_c(with(bs, qaxis, L, {n}), DType::Float64);
        Out eo = linalg::alloc_c(with(bs, qaxis, L, {n}), DType::Float64);
        Out kpo = linalg::alloc_c(with(bs, qaxis, L, {k}), DType::Bool);
        Out so = linalg::alloc_c(with(bs, qaxis, L, {qNStat}), DType::Float64);
        Out bo = linalg::alloc_c(with(bs, qaxis, L, {k}), DType::Int64);
        const double nk = double(n) * double(std::max<lint>(k, 1));
        const double element_ns = 2000.0 + double(L) * nk * (double(k) + 20.0) * 40.0 * (cov == cNid ? 2.0 : 1.0);
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            Outs o{po.as<double>() + i * L * k, co.as<double>() + i * L * k * k, ro.as<double>() + i * L * k * k,
                   fo.as<double>() + i * L * n, eo.as<double>() + i * L * n, so.as<double>() + i * L * qNStat,
                   kpo.as<std::uint8_t>() + i * L * k, bo.as<std::int64_t>() + i * L * k};
            return quantreg_one(op, n, p, y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2),
                                X.stride(-1), o, sc);
        });
        if (f) {
            const char* what = f.code == eInfY ? "'y' contains infinity" : "'X' contains infinity";
            throw nb::value_error(linalg::failure_message(b, f.index, (std::string(fn) + "(): " + what).c_str()).c_str());
        }
        return nb::make_tuple(linalg::to_array(po), linalg::to_array(co), linalg::to_array(ro), linalg::to_array(fo),
                              linalg::to_array(eo), linalg::to_array(kpo), linalg::to_array(so), linalg::to_array(bo));
    });
}

}  // namespace

void bind_quantile(nb::module_& m) {
    m.def("quantreg_fit", &quantreg_fit, nb::arg("func"), nb::arg("y").none(), nb::arg("X").none(), nb::arg("qs"),
          nb::arg("intercept"), nb::arg("rtol"), nb::arg("cov"), nb::arg("hraw"), nb::arg("hhalf"), nb::arg("kerc"),
          nb::arg("max_iter"), nb::arg("tol"), nb::arg("qaxis"));
}

}  // namespace sd::regression
