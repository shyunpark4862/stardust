// The linear models' numerical core (docs/design/regression.md §3.1-§3.3).
//
// One regression per batch element, through linalg's run_batch (one task per regression,
// the same bits for any thread count, docs/design/linalg.md §6):
//   1. The rows with a positive weight are the fit's (R's lm, G20). Each is whitened by
//      sqrt(w); the design, intercept column first, is packed column-major.
//   2. Unpivoted Householder QR (geqrf). A column with |R_jj| <= rtol * ||x_j|| is aliased
//      (G9, G21; the rule of R's dqrdc2, so the earlier column stays): it is dropped and the
//      columns after it are factored again. When anything was dropped, the kept columns are
//      factored once more from scratch, so the fit is the reduced problem's, bit for bit.
//   3. beta from R beta = (Q^T b)_1, then `steps` corrections on the augmented system
//      [I X; X^T W 0] [r; beta] = [y; 0] (Bjorck 1967): both blocks of its residual,
//      f = y - r - X beta and g = -X^T W r, are summed in double-double (Dot2, Ogita, Rump
//      and Oishi 2005), and the correction is solved with the same QR (G15).
//   4. Fitted values and residuals (every row, the zero-weight ones too) and the sums of
//      squares in double-double; the covariance (R^T R)^-1, or the HC sandwich
//      R^-1 (Q^T diag(omega) Q) R^-T (docs/design/regression.md §3.3).
//
// Row sums are cut into chunks whose size depends on the number of columns only, and the
// chunks' partial sums are added in order: the result does not depend on the thread count
// or on whether the regression runs alone or inside a batch.
#include "regression/linear.hpp"

#include "bind/errors.hpp"
#include "linalg/ops/common.hpp"
#include "parallel/pool.hpp"
#include "special/dd.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace sd::regression {

namespace nb = nanobind;
using linalg::Batch;
using linalg::Failure;
using linalg::Input;
using linalg::lint;
using linalg::Out;
using linalg::Scratch;
using DD = special::dd::DD;
using special::dd::operator+;
using special::dd::operator-;
using special::dd::operator*;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kEps = 0x1p-52;

// cov_type codes (the Python layer maps the names)
enum Cov : int { kNonrobust = 0, kHC0 = 1, kHC1 = 2, kHC2 = 3, kHC3 = 4 };

// The per-regression scalars, in this order (stardust/regression/_linear.py reads them).
enum Stat : int {
    kNobs = 0, kRank, kSsr, kCtss, kUtss, kEssC, kEssU, kSumLogW, kWaldF, kWaldQ, kIsNaN, kNStat
};

// Failure codes of one regression.
constexpr int kInfY = 1, kInfX = 2, kInfW = 3, kNegW = 4, kInfU = 5;

inline double rd(const std::byte* p) {
    double v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

// Error-free transformations.
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
    e = std::fma(a, b, -p);
}
inline DD dd_of(double hi, double lo) {  // normalized
    double s, e;
    two_sum(hi, lo, s, e);
    return {s, e};
}

// A column of the design as the kernel reads it: element i at base + i * stride. The
// intercept of OLS/WLS is a column of ones (stride 0 on a constant).
struct Col {
    const std::byte* base;
    std::int64_t stride;
};
const double kOneValue = 1.0;

// Rows per chunk of a row sum: about 64K elements of the design per chunk, so a chunk's
// columns stay in cache between the two sweeps of a pass. Depends on k only.
inline lint chunk_rows(lint k) {
    lint c = 4096;
    while (c > 256 && c * k > 65536) c /= 2;
    return c;
}

// Work (rows x columns) from which a regression's row passes go to the pool.
constexpr double kPassParallelMin = 65536;

// Runs body(chunk) for every chunk, on the pool when the pass is large.
template <class F>
void for_chunks(lint nchunks, double work, F&& body) {
    if (nchunks >= 2 && work >= kPassParallelMin) {
        parallel::parallel_for(nchunks, [&](std::int64_t c) { body(static_cast<lint>(c)); });
    } else {
        for (lint c = 0; c < nchunks; ++c) body(c);
    }
}

// ---- Householder reflectors on one vector -----------------------------------------------
// c := Q^T c (qt) or Q c with Q = H_0 ... H_{kk-1} from geqrf (H_j = I - tau_j v_j v_j^T,
// v_j = 1 at row j and A[j*m + t] below). In place of dormqr, which on a single vector spent
// most of a refinement step (docs/design/regression.md §3.1): one pass per reflector applies
// it and forms the next reflector's dot product. The dot products are summed in chunks of
// rows fixed by m alone and added in order, so the bits do not depend on the thread count.
constexpr lint kReflChunk = 4096;
constexpr lint kReflParallelMin = 1 << 15;  // rows from which a pass goes to the pool

void apply_reflectors(const double* A, lint m, lint kk, const double* tau, double* c, bool qt, double* part) {
    if (kk == 0 || m == 0) return;
    auto order = [&](lint q) { return qt ? q : kk - 1 - q; };
    // one pass: apply reflector j with coefficient a (j < 0: none), return the dot product of
    // reflector nj (nj < 0: none) with the updated c
    auto pass = [&](lint j, double a, lint nj) -> double {
        const double* vj = j >= 0 ? A + j * m : nullptr;
        const double* vn = nj >= 0 ? A + nj * m : nullptr;
        const lint start = j < 0 ? nj : (nj < 0 ? j : std::min(j, nj));
        const lint bulk = std::max(j, nj) + 1;
        double head = 0.0;
        for (lint t = start; t < std::min(bulk, m); ++t) {
            if (vj && t >= j) c[t] -= a * (t == j ? 1.0 : vj[t]);
            if (vn && t >= nj) head += (t == nj ? 1.0 : vn[t]) * c[t];
        }
        if (bulk >= m) return head;
        const lint c0 = bulk / kReflChunk, c1 = (m - 1) / kReflChunk + 1;
        auto chunk = [&](lint ch) {
            const lint t0 = std::max(bulk, ch * kReflChunk), t1 = std::min(m, (ch + 1) * kReflChunk);
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            lint t = t0;
            if (vj && vn) {
                for (; t + 4 <= t1; t += 4) {
                    c[t] -= a * vj[t];
                    c[t + 1] -= a * vj[t + 1];
                    c[t + 2] -= a * vj[t + 2];
                    c[t + 3] -= a * vj[t + 3];
                    s0 += vn[t] * c[t];
                    s1 += vn[t + 1] * c[t + 1];
                    s2 += vn[t + 2] * c[t + 2];
                    s3 += vn[t + 3] * c[t + 3];
                }
                for (; t < t1; ++t) {
                    c[t] -= a * vj[t];
                    s0 += vn[t] * c[t];
                }
            } else if (vj) {
                for (; t < t1; ++t) c[t] -= a * vj[t];
            } else {
                for (; t + 4 <= t1; t += 4) {
                    s0 += vn[t] * c[t];
                    s1 += vn[t + 1] * c[t + 1];
                    s2 += vn[t + 2] * c[t + 2];
                    s3 += vn[t + 3] * c[t + 3];
                }
                for (; t < t1; ++t) s0 += vn[t] * c[t];
            }
            part[ch] = (s0 + s1) + (s2 + s3);
        };
        if (m - bulk >= kReflParallelMin && c1 - c0 >= 2) {
            parallel::parallel_for(c1 - c0, [&](std::int64_t q) { chunk(c0 + static_cast<lint>(q)); });
        } else {
            for (lint ch = c0; ch < c1; ++ch) chunk(ch);
        }
        double dot = head;
        if (vn)
            for (lint ch = c0; ch < c1; ++ch) dot += part[ch];
        return dot;
    };
    double dot = pass(-1, 0.0, order(0));
    for (lint q = 0; q < kk; ++q) {
        const lint j = order(q);
        dot = pass(j, tau[j] * dot, q + 1 < kk ? order(q + 1) : -1);
    }
}

// Indices into the accumulators of a pass's chunk (each a double-double).
enum Acc : int { kSwuy = 0, kSwuu, kSlogw, kAccInit };
enum AccF : int { fSsr = 0, fCtss, fUtss, fEssC, fEssU, kAccFinal };

// The factor over LAPACK's condition estimate in the refinement's stopping test.
constexpr double kSafety = 16.0;

struct Problem {
    lint n, p, k;   // rows, columns of X, columns of the design (p + intercept)
    bool intercept, has_w, has_u;
    double rtol;    // < 0: max(nobs, k) * eps
    int cov;
    int steps;
};

// The inputs of one regression (byte pointers and strides).
struct Data {
    const std::byte* y;
    std::int64_t ys;
    const std::byte* X;
    std::int64_t xrs, xcs;
    const std::byte* w;
    std::int64_t ws;
    const std::byte* u;
    std::int64_t us;
};

// Outputs of one regression.
struct Dest {
    double* params;  // k
    double* cov;     // k x k, row-major
    double* R;       // k x k, row-major (kept rows/columns in their places)
    double* fitted;  // n
    double* resid;   // n
    std::uint8_t* kept;  // k
    double* stats;   // kNStat
};

void fill_nan(const Problem& pb, const Dest& o) {
    std::fill(o.params, o.params + pb.k, kNaN);
    std::fill(o.cov, o.cov + pb.k * pb.k, kNaN);
    std::fill(o.R, o.R + pb.k * pb.k, kNaN);
    std::fill(o.fitted, o.fitted + pb.n, kNaN);
    std::fill(o.resid, o.resid + pb.n, kNaN);
    std::fill(o.kept, o.kept + pb.k, std::uint8_t(0));
    std::fill(o.stats, o.stats + kNStat, kNaN);
    o.stats[kRank] = 0;
    o.stats[kIsNaN] = 1;
}

// One regression. Returns 0 or a failure code.
int fit_one(const Problem& pb, const Data& d, const Dest& o, Scratch& sc) {
    const lint n = pb.n, k = pb.k;
    // ---- 1. checks, used rows, whitening factors
    bool nan = false;
    for (lint i = 0; i < n; ++i) {
        const double v = rd(d.y + i * d.ys);
        if (std::isnan(v)) nan = true;
        else if (std::isinf(v)) return kInfY;
    }
    // X in memory order (rows outside for a row-major X), on the pool when large; NaN and
    // infinity are told apart only when a value is not finite
    {
        const bool rows_outer = std::llabs(d.xcs) <= std::llabs(d.xrs);
        const lint outer = rows_outer ? n : pb.p, inner = rows_outer ? pb.p : n;
        const std::int64_t os = rows_outer ? d.xrs : d.xcs, is = rows_outer ? d.xcs : d.xrs;
        constexpr lint kBlock = 1024;
        const lint nblocks = (outer + kBlock - 1) / kBlock;
        std::vector<std::uint8_t> flags(static_cast<std::size_t>(std::max<lint>(nblocks, 1)), 0);
        auto block = [&](lint b) {
            std::uint64_t bad = 0;
            for (lint o = b * kBlock; o < std::min(outer, (b + 1) * kBlock); ++o) {
                const std::byte* row = d.X + o * os;
                for (lint q = 0; q < inner; ++q) {
                    std::uint64_t u;
                    std::memcpy(&u, row + q * is, sizeof u);
                    bad |= static_cast<std::uint64_t>((u & 0x7ff0000000000000ull) == 0x7ff0000000000000ull);
                }
            }
            if (!bad) return;
            std::uint8_t f = 1;  // NaN
            for (lint o = b * kBlock; o < std::min(outer, (b + 1) * kBlock); ++o)
                for (lint q = 0; q < inner; ++q)
                    if (std::isinf(rd(d.X + o * os + q * is))) f = 2;
            flags[static_cast<std::size_t>(b)] = f;
        };
        if (nblocks >= 2 && double(n) * double(pb.p) >= double(1 << 20)) {
            parallel::parallel_for(nblocks, [&](std::int64_t b) { block(static_cast<lint>(b)); });
        } else {
            for (lint b = 0; b < nblocks; ++b) block(b);
        }
        for (const std::uint8_t f : flags) {
            if (f == 2) return kInfX;
            if (f == 1) nan = true;
        }
    }
    if (pb.has_w)
        for (lint i = 0; i < n; ++i) {
            const double v = rd(d.w + i * d.ws);
            if (std::isnan(v)) nan = true;
            else if (std::isinf(v)) return kInfW;
            else if (v < 0) return kNegW;
        }
    if (pb.has_u)
        for (lint i = 0; i < n; ++i) {
            const double v = rd(d.u + i * d.us);
            if (std::isnan(v)) nan = true;
            else if (std::isinf(v)) return kInfU;
        }
    if (nan) {
        fill_nan(pb, o);
        return 0;
    }
    lint m = n;
    lint* idx = sc.get_ints(static_cast<std::size_t>(std::max<lint>(n, 1)) + static_cast<std::size_t>(k));
    lint* K = idx + std::max<lint>(n, 1);  // kept columns, in order
    if (pb.has_w) {
        m = 0;
        for (lint i = 0; i < n; ++i)
            if (rd(d.w + i * d.ws) > 0) idx[m++] = i;
    } else {
        for (lint i = 0; i < n; ++i) idx[i] = i;
    }
    auto weight = [&](lint i) { return pb.has_w ? rd(d.w + i * d.ws) : 1.0; };
    auto uval = [&](lint i) { return pb.has_u ? rd(d.u + i * d.us) : 1.0; };
    auto col = [&](lint c) -> Col {
        if (pb.intercept && c == 0) {
            if (pb.has_u) return {d.u, d.us};
            return {reinterpret_cast<const std::byte*>(&kOneValue), 0};
        }
        const lint j = c - (pb.intercept ? 1 : 0);
        return {d.X + j * d.xcs, d.xrs};
    };

    // ---- scratch: one block of doubles, carved in 16-byte units
    const lint mm = std::max<lint>(m, 1), kk0 = std::max<lint>(k, 1);
    const lint chunk = chunk_rows(k);
    const lint nchunks = (m + chunk - 1) / chunk;
    const bool robust = pb.cov != kNonrobust;
    std::size_t total = 0;
    auto reserve = [&](std::int64_t cnt) {
        const std::size_t at = total;
        total += linalg::pad2(cnt);
        return at;
    };
    const std::size_t o_s = reserve(mm), o_A = reserve(mm * kk0), o_tau = reserve(kk0), o_nrm = reserve(kk0),
                      o_beta = reserve(kk0), o_c = reserve(mm), o_f = reserve(mm), o_r = reserve(mm),
                      o_P = reserve(mm), o_S = reserve(mm), o_h = reserve(kk0), o_g = reserve(kk0),
                      o_part = reserve(std::max<lint>(nchunks, 1) * 2 * (kk0 + kAccInit + kAccFinal)),
                      o_work = reserve(kk0), o_RR = reserve(kk0 * kk0), o_M = reserve(robust ? kk0 * kk0 : 0),
                      o_Q = reserve(robust ? mm * kk0 : 0), o_hat = reserve(robust ? mm : 0),
                      o_refl = reserve(mm / kReflChunk + 1);
    double* base = sc.get<double>(total);
    double *s = base + o_s, *A = base + o_A, *tau = base + o_tau, *nrm = base + o_nrm, *beta = base + o_beta,
           *cv = base + o_c, *f = base + o_f, *r = base + o_r, *P = base + o_P, *S = base + o_S, *h = base + o_h,
           *g = base + o_g, *part = base + o_part, *work = base + o_work, *RR = base + o_RR, *M = base + o_M,
           *Q = base + o_Q, *hat = base + o_hat, *refl = base + o_refl;
    for (lint t = 0; t < m; ++t) s[t] = pb.has_w ? std::sqrt(weight(idx[t])) : 1.0;

    // ---- 2. QR with aliased columns dropped
    // whitened columns cols[0..nc) into dst (m rows each)
    // in blocks of rows, so that a row-major X is read a few cache lines at a time
    auto pack = [&](double* dst, const lint* cols, lint nc) {
        constexpr lint kRows = 512;
        const lint nb = (m + kRows - 1) / kRows;
        auto rows = [&](lint b) {
            const lint t0 = b * kRows, t1 = std::min(m, t0 + kRows);
            for (lint jj = 0; jj < nc; ++jj) {
                const Col c = col(cols[jj]);
                double* out = dst + jj * m;
                for (lint t = t0; t < t1; ++t) out[t] = s[t] * rd(c.base + idx[t] * c.stride);
            }
        };
        if (nb >= 2 && double(m) * nc >= double(1 << 20)) {
            parallel::parallel_for(nb, [&](std::int64_t b) { rows(static_cast<lint>(b)); });
        } else {
            for (lint b = 0; b < nb; ++b) rows(b);
        }
    };
    lint kk = k;
    for (lint c = 0; c < k; ++c) K[c] = c;
    double tol = pb.rtol >= 0 ? pb.rtol : static_cast<double>(std::max(m, k)) * kEps;
    if (m > 0 && k > 0) {
        pack(A, K, k);
        for (lint c = 0; c < k; ++c) nrm[c] = cblas_dnrm2(m, A + c * m, 1);
        linalg::lapack::geqrf(m, k, A, m, tau, sc);
        bool dropped = false;
        int restarts = 0;
        // the first aliased column at or after `from` among the factored ones, or -1
        auto first_aliased = [&](lint from) -> lint {
            const lint lim = std::min(m, kk);
            for (lint j = from; j < lim; ++j)
                if (!(std::fabs(A[j * m + j]) > tol * nrm[K[j]])) return j;
            return -1;
        };
        lint j = first_aliased(0);
        // blocked restarts: drop column j, factor the columns after it again
        while (j >= 0 && restarts < 8) {
            dropped = true;
            ++restarts;
            std::copy(K + j + 1, K + kk, K + j);
            --kk;
            if (kk > j) {
                pack(A + j * m, K + j, kk - j);
                if (j > 0) linalg::lapack::ormqr('T', m, kk - j, j, A, m, tau, A + j * m, m, sc);
                if (j < m) linalg::lapack::geqrf(m - j, kk - j, A + j * m + j, m, tau + j, sc);
            }
            j = first_aliased(j);
        }
        if (j >= 0) {
            // many aliased columns: R's dqrdc2 column by column from j on (level 2), so the
            // cost stays one QR whatever their number
            std::copy(K + j + 1, K + kk, K + j);
            --kk;
            if (kk > j) {
                pack(A + j * m, K + j, kk - j);
                if (j > 0) linalg::lapack::ormqr('T', m, kk - j, j, A, m, tau, A + j * m, m, sc);
            }
            lint c = j;
            while (c < kk && c < m) {
                double* cc = A + c * m;
                const double nr = cblas_dnrm2(m - c, cc + c, 1);
                if (!(nr > tol * nrm[K[c]])) {
                    std::memmove(cc, cc + m, static_cast<std::size_t>((kk - c - 1) * m) * sizeof(double));
                    std::copy(K + c + 1, K + kk, K + c);
                    --kk;
                    continue;
                }
                double alpha = cc[c];
                linalg::lapack::larfg(m - c, &alpha, cc + c + 1, tau + c);
                if (c + 1 < kk) {
                    cc[c] = 1.0;
                    linalg::lapack::larf_left(m - c, kk - c - 1, cc + c, tau[c], A + (c + 1) * m + c, m, work);
                }
                cc[c] = alpha;
                ++c;
            }
        }
        if (kk > m) {  // m kept columns span every row: the rest are aliased
            kk = m;
            dropped = true;
        }
        if (dropped && kk > 0) {
            pack(A, K, kk);
            linalg::lapack::geqrf(m, kk, A, m, tau, sc);
        }
    } else {
        kk = 0;
    }

    // ---- 3. solve and refine
    auto colK = [&](lint jj) { return col(K[jj]); };
    // x beta for rows [t0, t1) into (P, S) (Dot2)
    auto xbeta = [&](lint t0, lint t1, const double* b) {
        for (lint t = t0; t < t1; ++t) P[t] = S[t] = 0.0;
        for (lint jj = 0; jj < kk; ++jj) {
            const Col c = colK(jj);
            const double bj = b[jj];
            for (lint t = t0; t < t1; ++t) {
                double hp, ep, q;
                two_prod(rd(c.base + idx[t] * c.stride), bj, hp, ep);
                two_sum(P[t], hp, P[t], q);
                S[t] += q + ep;
            }
        }
    };
    // g's partial sums for chunk ch: sum_t (w_t r_t) x_tj in (hi, lo) pairs
    auto gsum = [&](lint ch, lint t0, lint t1) {
        double* gp = part + ch * 2 * (kk0 + kAccInit + kAccFinal);
        for (lint jj = 0; jj < kk; ++jj) {
            const Col c = colK(jj);
            double hi = 0, lo = 0;
            for (lint t = t0; t < t1; ++t) {
                double ah = r[t], al = 0.0;
                if (pb.has_w) two_prod(weight(idx[t]), r[t], ah, al);
                const double x = rd(c.base + idx[t] * c.stride);
                double hp, ep, q;
                two_prod(ah, x, hp, ep);
                two_sum(hi, hp, hi, q);
                lo += q + ep + al * x;
            }
            gp[2 * jj] = hi;
            gp[2 * jj + 1] = lo;
        }
    };
    auto gcombine = [&]() {
        for (lint jj = 0; jj < kk; ++jj) {
            DD acc{0.0, 0.0};
            for (lint ch = 0; ch < nchunks; ++ch) {
                const double* gp = part + ch * 2 * (kk0 + kAccInit + kAccFinal);
                acc = acc + dd_of(gp[2 * jj], gp[2 * jj + 1]);
            }
            g[jj] = -(acc.hi + acc.lo);
        }
    };
    auto accs = [&](lint ch) { return part + ch * 2 * (kk0 + kAccInit + kAccFinal) + 2 * kk0; };
    const double work_rows = double(m) * double(std::max<lint>(kk, 1));

    // initial solution
    if (kk > 0) {
        for (lint t = 0; t < m; ++t) cv[t] = s[t] * rd(d.y + idx[t] * d.ys);
        apply_reflectors(A, m, kk, tau, cv, true, refl);
        std::copy(cv, cv + kk, beta);
        linalg::lapack::trtrs('U', 'N', 'N', kk, 1, A, m, beta, kk);
    }
    // the first pass: r = y - X beta rounded, f its remainder, g; and the centering sums
    for_chunks(nchunks, work_rows, [&](lint ch) {
        const lint t0 = ch * chunk, t1 = std::min(m, t0 + chunk);
        xbeta(t0, t1, beta);
        DD a[kAccInit] = {};
        for (lint t = t0; t < t1; ++t) {
            const lint i = idx[t];
            const double yv = rd(d.y + i * d.ys);
            double hh, e;
            two_sum(yv, -P[t], hh, e);
            const double v = e - S[t];
            r[t] = hh + v;
            f[t] = (hh - r[t]) + v;
            const double wt = weight(i), ut = uval(i);
            double ph, pl;
            two_prod(wt, ut, ph, pl);
            const DD wu = dd_of(ph, pl);
            a[kSwuy] = a[kSwuy] + wu * yv;
            a[kSwuu] = a[kSwuu] + wu * ut;
            if (pb.has_w) a[kSlogw] = a[kSlogw] + std::log(wt);
        }
        double* ap = accs(ch);
        for (int q = 0; q < kAccInit; ++q) ap[2 * q] = a[q].hi, ap[2 * q + 1] = a[q].lo;
        if (pb.steps > 0) gsum(ch, t0, t1);
    });
    DD init[kAccInit] = {};
    for (lint ch = 0; ch < nchunks; ++ch) {
        const double* ap = accs(ch);
        for (int q = 0; q < kAccInit; ++q) init[q] = init[q] + DD{ap[2 * q], ap[2 * q + 1]};
    }
    if (pb.steps > 0) gcombine();
    // The refinement stops once the next correction cannot matter (user decision 2026-11-18:
    // as many steps as needed, at most pb.steps): each step shrinks the error by about
    // kappa u (measured, bench/regression/g1/refine.log), so after a correction d the error
    // left is about |d| kappa u, and below 2^-54 |beta| a further step changes nothing. kappa
    // is LAPACK's estimate from R (trcon, within a small factor; kSafety covers it). The test
    // depends only on the data, so the result is the same for any thread count.
    const double kappa_u = kk > 0 ? kSafety * 0x1p-53 / linalg::lapack::trcon_1('U', kk, A, m) : 0.0;
    for (int step = 0; step < pb.steps && kk > 0; ++step) {
        // correction: h = R^-T g, d = Q^T (s f), dbeta = R^-1 (d_1 - h), dr~ = Q [h; d_2]
        std::copy(g, g + kk, h);
        linalg::lapack::trtrs('U', 'T', 'N', kk, 1, A, m, h, kk);
        for (lint t = 0; t < m; ++t) cv[t] = s[t] * f[t];
        apply_reflectors(A, m, kk, tau, cv, true, refl);
        for (lint jj = 0; jj < kk; ++jj) work[jj] = cv[jj] - h[jj];
        linalg::lapack::trtrs('U', 'N', 'N', kk, 1, A, m, work, kk);
        double dn = 0.0, bn = 0.0;
        for (lint jj = 0; jj < kk; ++jj) {
            beta[jj] += work[jj];
            dn = std::max(dn, std::fabs(work[jj]));
            bn = std::max(bn, std::fabs(beta[jj]));
        }
        std::copy(h, h + kk, cv);
        apply_reflectors(A, m, kk, tau, cv, false, refl);
        for (lint t = 0; t < m; ++t) r[t] += cv[t] / s[t];
        const bool converged = !(dn * kappa_u > 0x1p-54 * bn);  // NaN: stop (nothing to refine)
        if (step + 1 < pb.steps && !converged) {
            // the residuals of the augmented system at the new (r, beta)
            for_chunks(nchunks, work_rows, [&](lint ch) {
                const lint t0 = ch * chunk, t1 = std::min(m, t0 + chunk);
                xbeta(t0, t1, beta);
                for (lint t = t0; t < t1; ++t) {
                    double h1, e1, h2, e2;
                    two_sum(rd(d.y + idx[t] * d.ys), -r[t], h1, e1);
                    two_sum(h1, -P[t], h2, e2);
                    f[t] = h2 + ((e1 + e2) - S[t]);
                }
                gsum(ch, t0, t1);
            });
            gcombine();
        }
        if (converged) break;
    }

    // ---- 4. fitted values, residuals and the sums of squares
    const DD wuu = init[kSwuu];
    const DD ybar = wuu.hi > 0 ? special::dd::operator/(init[kSwuy], wuu) : DD{kNaN, 0.0};
    for_chunks(nchunks, work_rows, [&](lint ch) {
        const lint t0 = ch * chunk, t1 = std::min(m, t0 + chunk);
        xbeta(t0, t1, beta);
        DD a[kAccFinal] = {};
        for (lint t = t0; t < t1; ++t) {
            const lint i = idx[t];
            const double yv = rd(d.y + i * d.ys);
            double hh, e;
            two_sum(yv, -P[t], hh, e);
            const double v = e - S[t];
            const double rr = hh + v;
            const DD res = dd_of(rr, (hh - rr) + v);
            const DD fit = dd_of(P[t], S[t]);
            o.fitted[i] = fit.hi;
            o.resid[i] = rr;
            r[t] = rr;
            const double wt = weight(i), ut = uval(i);
            a[fSsr] = a[fSsr] + (res * res) * wt;
            const DD yc = DD{yv, 0.0} - ybar * ut;
            a[fCtss] = a[fCtss] + (yc * yc) * wt;
            double yh, yl;
            two_prod(yv, yv, yh, yl);
            a[fUtss] = a[fUtss] + dd_of(yh, yl) * wt;
            const DD fc = fit - ybar * ut;
            a[fEssC] = a[fEssC] + (fc * fc) * wt;
            a[fEssU] = a[fEssU] + (fit * fit) * wt;
        }
        double* ap = accs(ch) + 2 * kAccInit;
        for (int q = 0; q < kAccFinal; ++q) ap[2 * q] = a[q].hi, ap[2 * q + 1] = a[q].lo;
    });
    DD fin[kAccFinal] = {};
    for (lint ch = 0; ch < nchunks; ++ch) {
        const double* ap = accs(ch) + 2 * kAccInit;
        for (int q = 0; q < kAccFinal; ++q) fin[q] = fin[q] + DD{ap[2 * q], ap[2 * q + 1]};
    }
    if (pb.has_w && m < n) {  // zero-weight rows: fitted values and residuals only
        std::vector<char> used(static_cast<std::size_t>(n), 0);
        for (lint t = 0; t < m; ++t) used[static_cast<std::size_t>(idx[t])] = 1;
        for (lint i = 0; i < n; ++i) {
            if (used[static_cast<std::size_t>(i)]) continue;
            double Pi = 0, Si = 0;
            for (lint jj = 0; jj < kk; ++jj) {
                const Col c = colK(jj);
                double hp, ep, q;
                two_prod(rd(c.base + i * c.stride), beta[jj], hp, ep);
                two_sum(Pi, hp, Pi, q);
                Si += q + ep;
            }
            const double yv = rd(d.y + i * d.ys);
            double hh, e;
            two_sum(yv, -Pi, hh, e);
            o.fitted[i] = Pi + Si;
            o.resid[i] = hh + (e - Si);
        }
    }

    // ---- 5. parameters, R, covariance
    double* st = o.stats;
    std::fill(st, st + kNStat, 0.0);
    st[kNobs] = static_cast<double>(m);
    st[kRank] = static_cast<double>(kk);
    st[kSsr] = fin[fSsr].hi;
    st[kCtss] = fin[fCtss].hi;
    st[kUtss] = fin[fUtss].hi;
    st[kEssC] = fin[fEssC].hi;
    st[kEssU] = fin[fEssU].hi;
    st[kSumLogW] = init[kSlogw].hi;
    st[kWaldF] = kNaN;
    st[kWaldQ] = kNaN;
    st[kIsNaN] = 0;
    std::fill(o.params, o.params + k, kNaN);
    std::fill(o.kept, o.kept + k, std::uint8_t(0));
    std::fill(o.R, o.R + k * k, 0.0);
    std::fill(o.cov, o.cov + k * k, kNaN);
    for (lint jj = 0; jj < kk; ++jj) {
        o.params[K[jj]] = beta[jj];
        o.kept[K[jj]] = 1;
        for (lint ii = 0; ii <= jj; ++ii) o.R[K[ii] * k + K[jj]] = A[jj * m + ii];
    }
    if (kk == 0) return 0;
    // (R^T R)^-1 into RR (column-major kk x kk, full)
    for (lint jj = 0; jj < kk; ++jj)
        for (lint ii = 0; ii < kk; ++ii) RR[jj * kk + ii] = ii <= jj ? A[jj * m + ii] : 0.0;
    if (!robust) {
        if (linalg::lapack::potri_upper(kk, RR, kk) > 0) {
            std::fill(RR, RR + kk * kk, kNaN);  // an exact zero on R's diagonal (cannot happen once kept)
        }
        for (lint jj = 0; jj < kk; ++jj)
            for (lint ii = 0; ii <= jj; ++ii) {
                const double v = RR[jj * kk + ii];
                o.cov[K[ii] * k + K[jj]] = v;
                o.cov[K[jj] * k + K[ii]] = v;
            }
        return 0;
    }
    // HC: thin Q, hat values, omega, meat Q^T diag(omega) Q, cov = R^-1 meat R^-T
    std::copy(A, A + m * kk, Q);
    linalg::lapack::orgqr(m, kk, kk, Q, m, tau, sc);
    if (pb.cov == kHC2 || pb.cov == kHC3) {
        std::fill(hat, hat + m, 0.0);
        for (lint jj = 0; jj < kk; ++jj)
            for (lint t = 0; t < m; ++t) hat[t] += Q[jj * m + t] * Q[jj * m + t];
    }
    for (lint t = 0; t < m; ++t) {
        const double e = std::fabs(s[t] * r[t]);  // the whitened residual
        double sq = e;                              // sqrt(omega)
        if (pb.cov == kHC2) sq = e / std::sqrt(1.0 - hat[t]);
        else if (pb.cov == kHC3) sq = e / (1.0 - hat[t]);
        cv[t] = sq;
    }
    for (lint jj = 0; jj < kk; ++jj)
        for (lint t = 0; t < m; ++t) Q[jj * m + t] *= cv[t];
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, kk, m, 1.0, Q, m, 0.0, M, kk);
    for (lint jj = 0; jj < kk; ++jj)
        for (lint ii = jj + 1; ii < kk; ++ii) M[jj * kk + ii] = M[ii * kk + jj];
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, kk, kk, 1.0, RR, kk, M, kk);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, kk, kk, 1.0, RR, kk, M, kk);
    const double c1 = pb.cov == kHC1 ? static_cast<double>(m) / static_cast<double>(m - kk) : 1.0;
    for (lint jj = 0; jj < kk; ++jj)
        for (lint ii = 0; ii <= jj; ++ii) {
            const double v = M[jj * kk + ii] * c1;  // the upper triangle, mirrored
            o.cov[K[ii] * k + K[jj]] = v;
            o.cov[K[jj] * k + K[ii]] = v;
        }
    // the Wald F statistic of every kept parameter but the intercept (statsmodels' fvalue
    // under a robust covariance)
    lint q = 0;
    for (lint jj = 0; jj < kk; ++jj)
        if (!(pb.intercept && K[jj] == 0)) work[q++] = static_cast<double>(jj);
    st[kWaldQ] = static_cast<double>(q);
    if (q == 0) return 0;
    double* V = RR;  // R is no longer needed
    double* bq = cv;
    for (lint b2 = 0; b2 < q; ++b2) {
        const lint jb = static_cast<lint>(work[b2]);
        bq[b2] = beta[jb];
        for (lint a2 = 0; a2 < q; ++a2) V[b2 * q + a2] = M[jb * kk + static_cast<lint>(work[a2])] * c1;
    }
    if (linalg::lapack::potrf('U', q, V, q) > 0) {
        st[kWaldF] = kNaN;
        return 0;
    }
    linalg::lapack::trtrs('U', 'T', 'N', q, 1, V, q, bq, q);  // U^T z = b: b^T V^-1 b = |z|^2
    double ss = 0;
    for (lint b2 = 0; b2 < q; ++b2) ss += bq[b2] * bq[b2];
    st[kWaldF] = ss / static_cast<double>(q);
    return 0;
}

[[noreturn]] void raise_failure(const Batch& b, const Failure& f, const char* func) {
    std::string what;
    switch (f.code) {
        case kInfY: what = "'y' contains infinity"; break;
        case kInfX: what = "'X' contains infinity"; break;
        case kInfW: what = "'weights' contains infinity"; break;
        case kNegW: what = "'weights' must be >= 0"; break;
        default: what = "'sigma' gives a whitened intercept with infinity"; break;
    }
    throw nb::value_error(linalg::failure_message(b, f.index, (std::string(func) + "(): " + what).c_str()).c_str());
}

std::vector<std::int64_t> with(std::vector<std::int64_t> shape, std::initializer_list<std::int64_t> tail) {
    shape.insert(shape.end(), tail);
    return shape;
}

// fit(func, y (..., n), X (..., n, p), w (..., n) | None, u (..., n) | None, intercept, rtol, cov, steps)
// -> (params, cov, R, fitted, resid, kept, stats). Inputs are float64 sd.ndarrays (the Python
// layer converts); u is GLS's whitened column of ones (it replaces the intercept column and
// centers the sums of squares).
nb::object fit(const std::string& func, nb::handle yo, nb::handle xo, nb::handle wo, nb::handle uo, bool intercept,
               double rtol, int cov, int steps) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X");
        const bool has_w = !wo.is_none(), has_u = !uo.is_none();
        std::optional<Input> w, u;
        if (has_w) w.emplace(wo, fn, "weights");
        if (has_u) u.emplace(uo, fn, "sigma");
        Problem pb;
        pb.n = X.shape(-2);
        pb.p = X.shape(-1);
        pb.intercept = intercept;
        pb.k = pb.p + (intercept ? 1 : 0);
        pb.has_w = has_w;
        pb.has_u = has_u;
        pb.rtol = rtol;
        pb.cov = cov;
        pb.steps = steps;
        const linalg::BatchOperand none{};
        Batch b({{y.batch_shape(1), y.batch_strides(1)},
                 {X.batch_shape(2), X.batch_strides(2)},
                 has_w ? linalg::BatchOperand{w->batch_shape(1), w->batch_strides(1)} : none,
                 has_u ? linalg::BatchOperand{u->batch_shape(1), u->batch_strides(1)} : none},
                fn);
        const lint n = pb.n, k = pb.k;
        Out po = linalg::alloc_c(with(b.shape(), {k}), DType::Float64);
        Out co = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out ro = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out fo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out eo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out ko = linalg::alloc_c(with(b.shape(), {k}), DType::Bool);
        Out so = linalg::alloc_c(with(b.shape(), {kNStat}), DType::Float64);
        // estimated serial ns of one regression: the QR, the passes and the fixed cost
        const double nk = double(n) * double(std::max<lint>(k, 1));
        const double element_ns = 1500.0 + nk * (0.6 * double(k) + 3.0 * (2 + steps)) + (cov ? 2.0 * nk * k : 0.0);
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            Data d{y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2), X.stride(-1),
                   has_w ? w->data() + off[2] : nullptr, has_w ? w->stride(-1) : 0,
                   has_u ? u->data() + off[3] : nullptr, has_u ? u->stride(-1) : 0};
            Dest o{po.as<double>() + i * k, co.as<double>() + i * k * k, ro.as<double>() + i * k * k,
                   fo.as<double>() + i * n, eo.as<double>() + i * n, ko.as<std::uint8_t>() + i * k,
                   so.as<double>() + i * kNStat};
            return fit_one(pb, d, o, sc);
        });
        if (f) raise_failure(b, f, fn);
        return nb::make_tuple(linalg::to_array(po), linalg::to_array(co), linalg::to_array(ro), linalg::to_array(fo),
                              linalg::to_array(eo), linalg::to_array(ko), linalg::to_array(so));
    });
}

// apply(func, X (..., m, p), beta (..., k), intercept, y (..., m) | None) -> (X beta, y - X beta | None),
// each in double-double rounded once (Dot2, the column order of the fit: predict on the
// fit's own X gives its fitted values bit for bit). beta has 0 for aliased columns.
nb::object apply(const std::string& func, nb::handle xo, nb::handle bo, bool intercept, nb::handle yo) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input X(xo, fn, "X"), be(bo, fn, "params");
        const bool has_y = !yo.is_none();
        std::optional<Input> y;
        if (has_y) y.emplace(yo, fn, "y");
        const lint m = X.shape(-2), p = X.shape(-1), k = be.shape(-1);
        const linalg::BatchOperand none{};
        Batch b({{X.batch_shape(2), X.batch_strides(2)}, {be.batch_shape(1), be.batch_strides(1)},
                 has_y ? linalg::BatchOperand{y->batch_shape(1), y->batch_strides(1)} : none},
                fn);
        Out fo = linalg::alloc_c(with(b.shape(), {m}), DType::Float64);
        Out eo;
        if (has_y) eo = linalg::alloc_c(with(b.shape(), {m}), DType::Float64);
        const double element_ns = 200.0 + 3.0 * double(m) * double(std::max<lint>(k, 1));
        linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch&) {
            const std::byte* xb = X.data() + off[0];
            const std::byte* bb = be.data() + off[1];
            double* fi = fo.as<double>() + i * m;
            double* ei = has_y ? eo.as<double>() + i * m : nullptr;
            for (lint t = 0; t < m; ++t) {
                double Pv = 0, Sv = 0;
                for (lint c = 0; c < k; ++c) {
                    const double bj = rd(bb + c * be.stride(-1));
                    if (bj == 0.0) continue;  // aliased: no term (the fit has none either)
                    const double x = (intercept && c == 0) ? 1.0 : rd(xb + t * X.stride(-2) + (c - (intercept ? 1 : 0)) * X.stride(-1));
                    double hp, ep, q;
                    two_prod(x, bj, hp, ep);
                    two_sum(Pv, hp, Pv, q);
                    Sv += q + ep;
                }
                fi[t] = Pv + Sv;
                if (ei) {
                    const double yv = rd(y->data() + off[2] + t * y->stride(-1));
                    double hh, e;
                    two_sum(yv, -Pv, hh, e);
                    ei[t] = hh + (e - Sv);
                }
            }
            return 0;
        });
        (void)p;
        nb::object eobj = has_y ? linalg::to_array(eo) : nb::none();
        return nb::make_tuple(linalg::to_array(fo), eobj);
    });
}

// cond(R (..., k, k), kept (..., k)) -> the 2-norm condition number of the kept columns'
// R (that of the whitened design), NaN with no kept column.
nb::object cond(nb::handle ro, nb::handle ko) {
    return call_named("cond", [&]() -> nb::object {
        Input R(ro, "cond", "R"), kp(ko, "cond", "kept");
        const lint k = R.shape(-1);
        Batch b({{R.batch_shape(2), R.batch_strides(2)}, {kp.batch_shape(1), kp.batch_strides(1)}}, "cond");
        Out oo = linalg::alloc_c(b.shape(), DType::Float64);
        const double element_ns = 500.0 + 20.0 * double(k) * double(k) * double(k) / 64.0;
        linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            const std::byte* rb = R.data() + off[0];
            const std::byte* kb = kp.data() + off[1];
            lint* cols = sc.get_ints(static_cast<std::size_t>(std::max<lint>(k, 1)));
            lint kk = 0;
            for (lint c = 0; c < k; ++c) {
                std::uint8_t v;
                std::memcpy(&v, kb + c * kp.stride(-1), 1);
                if (v) cols[kk++] = c;
            }
            double* out = oo.as<double>() + i;
            if (kk == 0) {
                *out = kNaN;
                return 0;
            }
            double* a = sc.get<double>(static_cast<std::size_t>(kk * kk));
            double* sv = sc.get<double>(static_cast<std::size_t>(linalg::pad2(kk)), 1);
            bool finite = true;
            for (lint jj = 0; jj < kk; ++jj)
                for (lint ii = 0; ii < kk; ++ii) {
                    const double v = ii <= jj ? rd(rb + cols[ii] * R.stride(-2) + cols[jj] * R.stride(-1)) : 0.0;
                    finite &= std::isfinite(v);
                    a[jj * kk + ii] = v;
                }
            if (!finite) {
                *out = kNaN;
                return 0;
            }
            linalg::lapack::gesvd_values(kk, kk, a, kk, sv, sc);
            *out = sv[0] / sv[kk - 1];
            return 0;
        });
        return b.ndim() == 0 ? nb::cast(oo.as<double>()[0]) : linalg::to_array(oo);
    });
}

}  // namespace

void bind_linear(nb::module_& m) {
    m.def("fit", &fit, nb::arg("func"), nb::arg("y").none(), nb::arg("X").none(), nb::arg("w").none(),
          nb::arg("u").none(), nb::arg("intercept"), nb::arg("rtol"), nb::arg("cov"), nb::arg("steps"));
    m.def("apply", &apply, nb::arg("func"), nb::arg("X").none(), nb::arg("beta").none(), nb::arg("intercept"),
          nb::arg("y").none());
    m.def("cond", &cond, nb::arg("R").none(), nb::arg("kept").none());
}

}  // namespace sd::regression
