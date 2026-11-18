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
//      squares in double-double; the covariance (R^T R)^-1, or a sandwich R^-1 S R^-T whose
//      meat S is formed from the scores u_t = q_t e_t (q_t a row of the thin Q, e_t the
//      whitened residual): HC (sum omega_t q_t q_t^T), HAC (Newey-West lag sums with
//      Bartlett weights) or cluster (sums over groups) (docs/design/regression.md §3.3).
//   A second mode (hat_only) stops after step 2 and gives the hat values ||q_t||^2, the
//   leverages of the result's diagnostics: the same columns, the same QR.
//
// Row sums are cut into chunks whose size depends on the number of columns only, and the
// chunks' partial sums are added in order: the result does not depend on the thread count
// or on whether the regression runs alone or inside a batch.
#include "regression/linear.hpp"
#include "regression/lsq.hpp"

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
using namespace lsq;
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

// Rows per chunk of the HAC lag sums (fixed: the bits do not depend on the thread count).
constexpr lint kHacChunk = 4096;
// Groups x columns up to which the cluster sums go into one matrix (else blocks of groups).
constexpr lint kClusterDirect = 1 << 22;

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

inline std::int64_t rd_i64(const std::byte* p) {
    std::int64_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

void fill_nan(const Problem& pb, const Dest& o) {
    if (pb.hat_only) {
        std::fill(o.hat, o.hat + pb.n, kNaN);
        return;
    }
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

}  // namespace

// X scanned in memory order (rows outside for a row-major X), on the pool when large; NaN
// and infinity are told apart only when a value is not finite. Returns kInfX or 0 (nan set
// when X has a NaN).
int lsq::scan_x(const Data& d, lint n, lint p, bool& nan) {
    const bool rows_outer = std::llabs(d.xcs) <= std::llabs(d.xrs);
    const lint outer = rows_outer ? n : p, inner = rows_outer ? p : n;
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
    if (nblocks >= 2 && double(n) * double(p) >= double(1 << 20)) {
        parallel::parallel_for(nblocks, [&](std::int64_t b) { block(static_cast<lint>(b)); });
    } else {
        for (lint b = 0; b < nblocks; ++b) block(b);
    }
    for (const std::uint8_t f : flags) {
        if (f == 2) return kInfX;
        if (f == 1) nan = true;
    }
    return 0;
}

// One regression. Returns 0 or a failure code.
int lsq::fit_one(const Problem& pb, const Data& d, const Dest& o, Scratch& sc) {
    const lint n = pb.n, k = pb.k;
    // ---- 1. checks, used rows, whitening factors
    bool nan = false;
    for (lint i = 0; i < n && !pb.lean; ++i) {
        const double v = rd(d.y + i * d.ys);
        if (std::isnan(v)) nan = true;
        else if (std::isinf(v)) return kInfY;
    }
    if (!pb.lean)
        if (const int f = scan_x(d, n, pb.p, nan)) return f;
    if (pb.has_w && !pb.lean)
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
    if (pb.has_g)  // a missing cluster label is a NaN of the data
        for (lint i = 0; i < n; ++i)
            if (rd_i64(d.g + i * d.gs) < 0) nan = true;
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
    const bool robust = pb.cov != kNonrobust && !pb.hat_only;
    const bool want_T = o.T != nullptr && d.hdiff != nullptr && !pb.hat_only;
    const bool newton = d.newton_b0 != nullptr && d.newton_eta0 != nullptr && d.hdiff != nullptr && !pb.hat_only;
    const bool needQ = robust || pb.hat_only || want_T || newton;
    std::size_t total = 0;
    auto reserve = [&](std::int64_t cnt) {
        const std::size_t at = total;
        total += linalg::pad2(cnt);
        return at;
    };
    const std::size_t o_s = reserve(mm), o_A = reserve(mm * kk0), o_tau = reserve(kk0), o_nrm = reserve(kk0),
                      o_beta = reserve(kk0), o_c = reserve(mm), o_f = reserve(mm), o_r = reserve(mm),
                      o_P = reserve(mm), o_S = reserve(mm), o_h = reserve(kk0), o_g = reserve(kk0), o_AH = reserve(newton ? mm : 0), o_AL = reserve(newton ? mm : 0),
                      o_part = reserve(std::max<lint>(nchunks, 1) * 2 * (kk0 + kAccInit + kAccFinal)),
                      o_work = reserve(kk0), o_RR = reserve(kk0 * kk0), o_M = reserve(robust ? kk0 * kk0 : 0), o_W2 = reserve(robust ? kk0 * kk0 : 0),
                      o_Q = reserve(needQ ? mm * kk0 : 0), o_hat = reserve(needQ ? mm : 0),
                      o_refl = reserve(mm / kReflChunk + 1);
    double* base = sc.get<double>(total);
    double *s = base + o_s, *A = base + o_A, *tau = base + o_tau, *nrm = base + o_nrm, *beta = base + o_beta,
           *cv = base + o_c, *f = base + o_f, *r = base + o_r, *P = base + o_P, *S = base + o_S, *h = base + o_h,
           *g = base + o_g, *part = base + o_part, *work = base + o_work, *RR = base + o_RR, *M = base + o_M, *work2 = base + o_W2,
           *Q = base + o_Q, *hat = base + o_hat, *refl = base + o_refl, *AH = base + o_AH, *AL = base + o_AL;
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

    // the hat values h_t = ||q_t||^2 from the thin Q of the fit's own QR (0 for the rows out
    // of the fit: their weighted leverage w_t x_t (X^T W X)^-1 x_t^T is 0)
    auto hat_values = [&]() {
        std::copy(A, A + m * kk, Q);
        linalg::lapack::orgqr(m, kk, kk, Q, m, tau, sc);
        std::fill(hat, hat + m, 0.0);
        for (lint jj = 0; jj < kk; ++jj)
            for (lint t = 0; t < m; ++t) hat[t] += Q[jj * m + t] * Q[jj * m + t];
    };
    // T = Q^T diag(hdiff/w) Q (want_T, newton) into Tk (kk x kk column-major), summed by dgemm
    // over fixed chunks of rows in order; copied to o.T when asked
    std::vector<double> Tk;
    auto make_T = [&](bool form_q = true) {
        Tk.assign(static_cast<std::size_t>(kk * kk), 0.0);
        if (kk == 0) return;
        if (form_q) {
            std::copy(A, A + m * kk, Q);
            linalg::lapack::orgqr(m, kk, kk, Q, m, tau, sc);
        }
        // each chunk's Q^T diag(c/w) Q into its own slot (on the pool when large), the slots
        // added in chunk order: the same bits for any thread count
        constexpr lint kTChunk = 4096;
        const lint ntc = (m + kTChunk - 1) / kTChunk;
        const std::size_t slot = static_cast<std::size_t>(kk * kk);
        std::vector<double> parts(static_cast<std::size_t>(ntc) * slot);
        auto chunkT = [&](lint c) {
            const lint t0 = c * kTChunk, t1 = std::min(m, t0 + kTChunk), len = t1 - t0;
            std::vector<double> Bc(static_cast<std::size_t>(len * kk));
            for (lint jj = 0; jj < kk; ++jj)
                for (lint t = t0; t < t1; ++t) Bc[jj * len + (t - t0)] = Q[jj * m + t] * (d.hdiff[idx[t]] / (s[t] * s[t]));
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, kk, kk, len, 1.0, Q + t0, m, Bc.data(), len, 0.0,
                        parts.data() + static_cast<std::size_t>(c) * slot, kk);
        };
        if (ntc >= 2 && double(m) * double(kk) >= kPassParallelMin) {
            parallel::parallel_for(ntc, [&](std::int64_t c) { chunkT(static_cast<lint>(c)); });
        } else {
            for (lint c = 0; c < ntc; ++c) chunkT(c);
        }
        for (lint c = 0; c < ntc; ++c) {
            const double* q = parts.data() + static_cast<std::size_t>(c) * slot;
            for (std::size_t e = 0; e < slot; ++e) Tk[e] += q[e];
        }
        for (lint jj = 0; jj < kk; ++jj)
            for (lint ii = 0; ii < jj; ++ii) Tk[jj * kk + ii] = Tk[ii * kk + jj] = 0.5 * (Tk[jj * kk + ii] + Tk[ii * kk + jj]);
    };
    auto copy_T = [&]() {
        std::fill(o.T, o.T + k * k, 0.0);
        if (Tk.size() != static_cast<std::size_t>(kk * kk)) make_T();
        for (lint jj = 0; jj < kk; ++jj)
            for (lint ii = 0; ii < kk; ++ii) o.T[K[ii] * k + K[jj]] = Tk[jj * kk + ii];
    };
    if (pb.hat_only) {
        std::fill(o.hat, o.hat + n, 0.0);
        if (kk > 0) {
            hat_values();
            for (lint t = 0; t < m; ++t) o.hat[idx[t]] = hat[t];
        }
        return 0;
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
    // Newton's step (Data::newton_b0): its coefficients' origin b0 in the kept columns, the
    // Cholesky factor of I - T and whether it is taken (set after the first solution)
    bool use_newton = false;
    std::vector<double> b0K, Mc;
    double kappa_m = 1.0;
    // g's partial sums for chunk ch: sum_t (w_t r_t) x_tj in (hi, lo) pairs; with Newton's
    // step sum_t (w_t r_t + c_t x_t (beta - b0)) x_tj, the row factor formed in double-double
    auto gsum = [&](lint ch, lint t0, lint t1) {
        double* gp = part + ch * 2 * (kk0 + kAccInit + kAccFinal);
        if (use_newton) {
            for (lint t = t0; t < t1; ++t) {
                // x_t beta - eta0_t in double-double
                double Pd = 0.0, Sd = 0.0;
                for (lint jj = 0; jj < kk; ++jj) {
                    const Col c = colK(jj);
                    double hp, ep, q;
                    two_prod(rd(c.base + idx[t] * c.stride), beta[jj], hp, ep);
                    two_sum(Pd, hp, Pd, q);
                    Sd += q + ep;
                }
                double dh, de;
                two_sum(Pd, -d.newton_eta0[idx[t]], dh, de);
                const DD del = dd_of(dh, de + Sd);
                double ah, al;
                two_prod(weight(idx[t]), r[t], ah, al);
                const DD a = dd_of(ah, al) + del * d.hdiff[idx[t]];
                AH[t] = a.hi;
                AL[t] = a.lo;
            }
            for (lint jj = 0; jj < kk; ++jj) {
                const Col c = colK(jj);
                double hi = 0, lo = 0;
                for (lint t = t0; t < t1; ++t) {
                    const double x = rd(c.base + idx[t] * c.stride);
                    double hp, ep, q;
                    two_prod(AH[t], x, hp, ep);
                    two_sum(hi, hp, hi, q);
                    lo += q + ep + AL[t] * x;
                }
                gp[2 * jj] = hi;
                gp[2 * jj + 1] = lo;
            }
            return;
        }
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
    // Newton's step: I - T = U^T U (else Fisher's stays), beta = b0 + R^-1 (I - T)^-1 (c_1 - R b0)
    // as the start the refinement corrects
    if (newton && kk > 0) {
        make_T();
        Mc.assign(static_cast<std::size_t>(kk * kk), 0.0);
        for (lint jj = 0; jj < kk; ++jj)
            for (lint ii = 0; ii <= jj; ++ii) Mc[jj * kk + ii] = (ii == jj ? 1.0 : 0.0) - Tk[jj * kk + ii];
        if (linalg::lapack::potrf('U', kk, Mc.data(), kk) == 0) {
            const double rc = linalg::lapack::trcon_1('U', kk, Mc.data(), kk);
            if (rc > 0.0) {
                use_newton = true;
                kappa_m = 1.0 / (rc * rc);
            }
        }
        if (use_newton) {
            b0K.assign(static_cast<std::size_t>(kk), 0.0);
            for (lint jj = 0; jj < kk; ++jj) b0K[jj] = std::isnan(d.newton_b0[K[jj]]) ? 0.0 : d.newton_b0[K[jj]];
            std::vector<double> y(static_cast<std::size_t>(kk));
            for (lint ii = 0; ii < kk; ++ii) {
                double acc = 0.0;
                for (lint jj = ii; jj < kk; ++jj) acc += A[jj * m + ii] * b0K[jj];
                y[ii] = cv[ii] - acc;
            }
            linalg::lapack::trtrs('U', 'T', 'N', kk, 1, Mc.data(), kk, y.data(), kk);
            linalg::lapack::trtrs('U', 'N', 'N', kk, 1, Mc.data(), kk, y.data(), kk);
            linalg::lapack::trtrs('U', 'N', 'N', kk, 1, A, m, y.data(), kk);
            for (lint jj = 0; jj < kk; ++jj) beta[jj] = b0K[jj] + y[jj];
        }
    }
    if (o.newton) *o.newton = use_newton ? 1 : 0;
    // an inner IRLS solve (lean): X beta for every row (Dot2), params and kept
    auto lean_output = [&]() {
        for_chunks(nchunks, work_rows, [&](lint ch) {
            const lint t0 = ch * chunk, t1 = std::min(m, t0 + chunk);
            xbeta(t0, t1, beta);
            for (lint t = t0; t < t1; ++t) o.fitted[idx[t]] = P[t] + S[t];
        });
        if (m < n) {
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
                o.fitted[i] = Pi + Si;
            }
        }
        o.stats[kNobs] = static_cast<double>(m);
        o.stats[kRank] = static_cast<double>(kk);
        std::fill(o.params, o.params + k, kNaN);
        std::fill(o.kept, o.kept + k, std::uint8_t(0));
        for (lint jj = 0; jj < kk; ++jj) {
            o.params[K[jj]] = beta[jj];
            o.kept[K[jj]] = 1;
        }
        if (o.R) {  // R of the kept columns in their places (row-major k x k), when asked
            std::fill(o.R, o.R + k * k, 0.0);
            for (lint jj = 0; jj < kk; ++jj)
                for (lint ii = 0; ii <= jj; ++ii) o.R[K[ii] * k + K[jj]] = A[jj * m + ii];
        }
    };
    if (pb.lean && pb.steps == 0 && !use_newton) {
        lean_output();
        if (want_T) copy_T();
        return 0;
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
    const double kappa_u =
        kk > 0 ? kSafety * 0x1p-53 / linalg::lapack::trcon_1('U', kk, A, m) * (use_newton ? kappa_m : 1.0) : 0.0;
    for (int step = 0; step < pb.steps && kk > 0; ++step) {
        // correction: h = R^-T g, d = Q^T (s f), dbeta = R^-1 (d_1 - h), dr~ = Q [h; d_2]
        std::copy(g, g + kk, h);
        linalg::lapack::trtrs('U', 'T', 'N', kk, 1, A, m, h, kk);
        for (lint t = 0; t < m; ++t) cv[t] = s[t] * f[t];
        apply_reflectors(A, m, kk, tau, cv, true, refl);
        for (lint jj = 0; jj < kk; ++jj) work[jj] = cv[jj] - h[jj];
        if (use_newton) {
            // y = (I - T)^-1 (d_1 - h); dbeta = R^-1 y; dr~ = Q [d_1 - y; d_2]
            linalg::lapack::trtrs('U', 'T', 'N', kk, 1, Mc.data(), kk, work, kk);
            linalg::lapack::trtrs('U', 'N', 'N', kk, 1, Mc.data(), kk, work, kk);
            for (lint jj = 0; jj < kk; ++jj) cv[jj] -= work[jj];
        }
        linalg::lapack::trtrs('U', 'N', 'N', kk, 1, A, m, work, kk);
        double dn = 0.0, bn = 0.0;
        for (lint jj = 0; jj < kk; ++jj) {
            beta[jj] += work[jj];
            dn = std::max(dn, std::fabs(work[jj]));
            bn = std::max(bn, std::fabs(beta[jj]));
        }
        if (!use_newton) std::copy(h, h + kk, cv);
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
    if (pb.lean) {
        lean_output();
        if (want_T) copy_T();
        return 0;
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
    st[kNGroups] = kNaN;
    st[kLags] = kNaN;
    std::fill(o.params, o.params + k, kNaN);
    std::fill(o.kept, o.kept + k, std::uint8_t(0));
    std::fill(o.R, o.R + k * k, 0.0);
    std::fill(o.cov, o.cov + k * k, kNaN);
    for (lint jj = 0; jj < kk; ++jj) {
        o.params[K[jj]] = beta[jj];
        o.kept[K[jj]] = 1;
        for (lint ii = 0; ii <= jj; ++ii) o.R[K[ii] * k + K[jj]] = A[jj * m + ii];
    }
    if (want_T) copy_T();
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
    // robust: thin Q, the scores U = diag(e~) Q (e~ the whitened residuals), the meat S from
    // them, cov = R^-1 S R^-T (no X^T X is formed)
    const bool hc = pb.cov <= kHC3;
    if (hc && (pb.cov == kHC2 || pb.cov == kHC3)) {
        hat_values();
    } else {
        std::copy(A, A + m * kk, Q);
        linalg::lapack::orgqr(m, kk, kk, Q, m, tau, sc);
    }
    // GLM: T = Q^T diag(hdiff / w) Q, the observed information's correction in Q's coordinates
    // (into work2: the Wald F, which uses it otherwise, is not computed for a GLM)
    const bool oim = pb.glm && d.hdiff != nullptr;
    if (oim) {  // Q is the thin Q here (orgqr above)
        if (Tk.size() != static_cast<std::size_t>(kk * kk)) make_T(false);
        std::copy(Tk.begin(), Tk.end(), work2);
    }
    // the n of the small-sample factors: the rows used, or the GLM's sum of frequency weights
    const double n_eff = d.nobs_eff >= 0 ? d.nobs_eff : static_cast<double>(m);
    for (lint t = 0; t < m; ++t) {
        double e = s[t] * r[t];  // the whitened residual
        // GLM frequency weights: a replicated row adds its score's square once per copy (HC,
        // HAC), not the square of the summed copies (statsmodels' convention)
        if (d.smult && pb.cov != kCluster) e *= d.smult[idx[t]];
        double sq = e;                 // HC: sqrt(omega) (the sign drops out of the squares)
        if (pb.cov == kHC2) sq = e / std::sqrt(1.0 - hat[t]);
        else if (pb.cov == kHC3) sq = e / (1.0 - hat[t]);
        cv[t] = sq;
    }
    for (lint jj = 0; jj < kk; ++jj)
        for (lint t = 0; t < m; ++t) Q[jj * m + t] *= cv[t];
    double c1 = 1.0;  // the small-sample factor
    if (hc) {
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, kk, m, 1.0, Q, m, 0.0, M, kk);
        if (pb.cov == kHC1) c1 = n_eff / (n_eff - static_cast<double>(kk));
    } else if (pb.cov == kHAC) {
        // Newey-West: S = U^T U + sum_l w_l (U_{l:}^T U_{:-l} + its transpose), w_l = 1 - l/(L+1)
        // (Bartlett; statsmodels' S_hac_simple). The lag sums are U^T V with V_t = sum_l w_l
        // u_{t-l} (a filter down each column), formed in chunks of rows and added in order.
        const std::int64_t L =
            pb.maxlags >= 0 ? pb.maxlags
                            : static_cast<std::int64_t>(std::floor(4.0 * std::pow(static_cast<double>(m) / 100.0, 2.0 / 9.0)));
        st[kLags] = static_cast<double>(L);
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, kk, m, 1.0, Q, m, 0.0, M, kk);
        const lint Le = static_cast<lint>(std::min<std::int64_t>(L, m - 1));  // lags with a pair of rows
        if (Le > 0) {
            double* wl = sc.get<double>(static_cast<std::size_t>(Le + 1) + static_cast<std::size_t>(kHacChunk * kk) +
                                            static_cast<std::size_t>(kk * kk),
                                        3);
            double* Vc = wl + Le + 1;
            double* C = Vc + kHacChunk * kk;
            for (lint l = 1; l <= Le; ++l) wl[l] = 1.0 - static_cast<double>(l) / (static_cast<double>(L) + 1.0);
            std::fill(C, C + kk * kk, 0.0);
            for (lint t0 = 0; t0 < m; t0 += kHacChunk) {
                const lint t1 = std::min(m, t0 + kHacChunk), len = t1 - t0;
                auto column = [&](lint jj) {
                    const double* uj = Q + jj * m;
                    double* vj = Vc + jj * kHacChunk;
                    std::fill(vj, vj + len, 0.0);
                    for (lint l = 1; l <= Le; ++l) {
                        const double a = wl[l];
                        for (lint t = std::max(t0, l); t < t1; ++t) vj[t - t0] += a * uj[t - l];
                    }
                };
                if (kk >= 2 && double(len) * double(kk) * double(Le) >= double(1 << 18)) {
                    parallel::parallel_for(kk, [&](std::int64_t jj) { column(static_cast<lint>(jj)); });
                } else {
                    for (lint jj = 0; jj < kk; ++jj) column(jj);
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, kk, kk, len, 1.0, Q + t0, m, Vc, kHacChunk, 1.0, C,
                            kk);
            }
            for (lint jj = 0; jj < kk; ++jj)
                for (lint ii = 0; ii <= jj; ++ii) M[jj * kk + ii] += C[jj * kk + ii] + C[ii * kk + jj];
        }
    } else {
        // one-way cluster: S = sum_g s_g s_g^T, s_g the sum of the group's scores (rows in
        // order), with Stata's and statsmodels' factor G/(G-1) (m-1)/(m-k); G counts the groups
        // among the fit's rows (zero-weight rows are out, G20)
        const std::size_t gt = static_cast<std::size_t>(std::max<std::int64_t>(pb.gtot, 1));
        // ints: the codes' local numbers, each row's group, G + 1 offsets (G <= m), a permutation
        lint* loc = sc.get<lint>(gt + 3 * static_cast<std::size_t>(mm) + 1, 2);
        lint* lab = loc + gt;
        lint* off = lab + mm;
        lint* perm = off + mm + 1;
        std::fill(loc, loc + gt, lint(-1));
        lint G = 0;
        for (lint t = 0; t < m; ++t) {
            const std::int64_t code = rd_i64(d.g + idx[t] * d.gs);
            lint& l = loc[code];
            if (l < 0) l = G++;
            lab[t] = l;
        }
        st[kNGroups] = static_cast<double>(G);
        if (G * kk <= kClusterDirect) {
            // all group sums in one G x kk matrix, rows read in order
            double* Sg = sc.get<double>(static_cast<std::size_t>(std::max<lint>(G, 1) * kk), 3);
            auto column = [&](lint jj) {
                double* sj = Sg + jj * G;
                const double* uj = Q + jj * m;
                std::fill(sj, sj + G, 0.0);
                for (lint t = 0; t < m; ++t) sj[lab[t]] += uj[t];
            };
            if (kk >= 2 && double(m) * double(kk) >= double(1 << 18)) {
                parallel::parallel_for(kk, [&](std::int64_t jj) { column(static_cast<lint>(jj)); });
            } else {
                for (lint jj = 0; jj < kk; ++jj) column(jj);
            }
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, kk, G, 1.0, Sg, std::max<lint>(G, 1), 0.0, M, kk);
        } else {
            // many groups: rows sorted by group (stable), sums for blocks of groups; each sum
            // adds its rows in the same order as above, so the bits are the same
            std::fill(off, off + G + 1, lint(0));
            for (lint t = 0; t < m; ++t) ++off[lab[t] + 1];
            for (lint q = 0; q < G; ++q) off[q + 1] += off[q];
            for (lint t = 0; t < m; ++t) perm[off[lab[t]]++] = t;
            for (lint q = G; q > 0; --q) off[q] = off[q - 1];
            off[0] = 0;
            const lint B = std::max<lint>(1, kClusterDirect / kk);
            double* Sb = sc.get<double>(static_cast<std::size_t>(B * kk), 3);
            std::fill(M, M + kk * kk, 0.0);
            for (lint g0 = 0; g0 < G; g0 += B) {
                const lint g1 = std::min(G, g0 + B), nb2 = g1 - g0;
                auto column = [&](lint jj) {
                    const double* uj = Q + jj * m;
                    double* sj = Sb + jj * B;
                    for (lint q = g0; q < g1; ++q) {
                        double acc = 0.0;
                        for (lint p2 = off[q]; p2 < off[q + 1]; ++p2) acc += uj[perm[p2]];
                        sj[q - g0] = acc;
                    }
                };
                if (kk >= 2) {
                    parallel::parallel_for(kk, [&](std::int64_t jj) { column(static_cast<lint>(jj)); });
                } else {
                    column(0);
                }
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, kk, nb2, 1.0, Sb, B, 1.0, M, kk);
            }
        }
        // one cluster: no estimate (the scores sum to 0 and G/(G-1) is infinite)
        c1 = G < 2 ? kNaN
                   : static_cast<double>(G) / static_cast<double>(G - 1) * (n_eff - 1.0) /
                         (n_eff - static_cast<double>(kk));
    }
    for (lint jj = 0; jj < kk; ++jj)
        for (lint ii = jj + 1; ii < kk; ++ii) M[jj * kk + ii] = M[ii * kk + jj];
    // The Wald F statistic of every kept parameter but the intercept (statsmodels' fvalue
    // under a robust covariance), b2^T V22^-1 b2 / q, formed in Q's coordinates: the intercept
    // is the first column, so rows 2.. of R^-1 are [0, R22^-1], V22 = c R22^-1 S22 R22^-T and
    // b2^T V22^-1 b2 = z^T S22^-1 z / c with z = R22 b2. No inverse of the covariance (whose
    // condition is kappa(X)^2 times the meat's) is formed.
    const lint q0 = (pb.intercept && K[0] == 0) ? 1 : 0, q = kk - q0;
    st[kWaldQ] = static_cast<double>(q);
    if (q > 0 && !pb.glm) {
        double* V = work2;  // S22, then its Cholesky factor
        double* z = cv;
        for (lint b2 = 0; b2 < q; ++b2) {
            double acc = 0.0;
            for (lint jj = q0 + b2; jj < kk; ++jj) acc += A[jj * m + (q0 + b2)] * beta[jj];
            z[b2] = acc;
            for (lint a2 = 0; a2 < q; ++a2) V[b2 * q + a2] = M[(q0 + b2) * kk + (q0 + a2)];
        }
        if (linalg::lapack::potrf('U', q, V, q) > 0) {
            st[kWaldF] = kNaN;
        } else {
            linalg::lapack::trtrs('U', 'T', 'N', q, 1, V, q, z, q);  // U^T v = z: z^T S22^-1 z = |v|^2
            double ss = 0;
            for (lint b2 = 0; b2 < q; ++b2) ss += z[b2] * z[b2];
            st[kWaldF] = ss / c1 / static_cast<double>(q);
        }
    }
    if (oim) {
        // the bread (R^T (I - T) R)^-1 = R^-1 (I - T)^-1 R^-T: with I - T = U^T U (Cholesky; it is
        // positive definite at a maximum of the likelihood), the meat becomes U^-1 U^-T S U^-1 U^-T
        for (lint jj = 0; jj < kk; ++jj)
            for (lint ii = 0; ii < kk; ++ii) work2[jj * kk + ii] = (ii == jj ? 1.0 : 0.0) - work2[jj * kk + ii];
        if (linalg::lapack::potrf('U', kk, work2, kk) > 0) {
            std::fill(M, M + kk * kk, kNaN);  // not a maximum: no sandwich
        } else {
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, kk, kk, 1.0, work2, kk, M, kk);
            cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, kk, kk, 1.0, work2, kk, M, kk);
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, kk, kk, 1.0, work2, kk, M, kk);
            cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, kk, kk, 1.0, work2, kk, M, kk);
        }
    }
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, kk, kk, 1.0, RR, kk, M, kk);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, kk, kk, 1.0, RR, kk, M, kk);
    for (lint jj = 0; jj < kk; ++jj)
        for (lint ii = 0; ii <= jj; ++ii) {
            const double v = M[jj * kk + ii] * c1;  // the upper triangle, mirrored
            o.cov[K[ii] * k + K[jj]] = v;
            o.cov[K[jj] * k + K[ii]] = v;
        }
    return 0;
}

namespace {

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

// fit(func, y (..., n), X (..., n, p), w (..., n) | None, u (..., n) | None, intercept, rtol, cov, steps,
//     groups (..., n) | None, gtot, maxlags, hat_only)
// -> (params, cov, R, fitted, resid, kept, stats), or with hat_only the hat values (..., n).
// Inputs are float64 sd.ndarrays (the Python layer converts); u is GLS's whitened column of
// ones (it replaces the intercept column and centers the sums of squares); groups are int64
// codes 0 .. gtot - 1, -1 for a missing label (cov = cluster).
nb::object fit(const std::string& func, nb::handle yo, nb::handle xo, nb::handle wo, nb::handle uo, bool intercept,
               double rtol, int cov, int steps, nb::handle go, std::int64_t gtot, std::int64_t maxlags, bool hat_only) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input y(yo, fn, "y"), X(xo, fn, "X");
        const bool has_w = !wo.is_none(), has_u = !uo.is_none(), has_g = !go.is_none();
        std::optional<Input> w, u, g;
        if (has_w) w.emplace(wo, fn, "weights");
        if (has_u) u.emplace(uo, fn, "sigma");
        if (has_g) g.emplace(go, fn, "groups");
        Problem pb;
        pb.n = X.shape(-2);
        pb.p = X.shape(-1);
        pb.intercept = intercept;
        pb.k = pb.p + (intercept ? 1 : 0);
        pb.has_w = has_w;
        pb.has_u = has_u;
        pb.has_g = has_g;
        pb.rtol = rtol;
        pb.cov = cov;
        pb.steps = steps;
        pb.maxlags = maxlags;
        pb.gtot = gtot;
        pb.hat_only = hat_only;
        if (cov == kCluster && !has_g) throw nb::value_error("fit(): cluster needs groups");
        if (has_w && has_u) throw nb::value_error("fit(): weights and sigma together");
        const linalg::BatchOperand none{};
        Batch b({{y.batch_shape(1), y.batch_strides(1)},
                 {X.batch_shape(2), X.batch_strides(2)},
                 // the weights (WLS) and the whitened ones (GLS) never come together: one operand
                 has_w   ? linalg::BatchOperand{w->batch_shape(1), w->batch_strides(1)}
                 : has_u ? linalg::BatchOperand{u->batch_shape(1), u->batch_strides(1)}
                         : none,
                 has_g ? linalg::BatchOperand{g->batch_shape(1), g->batch_strides(1)} : none},
                fn);
        const lint n = pb.n, k = pb.k;
        const double nk = double(n) * double(std::max<lint>(k, 1));
        if (hat_only) {
            Out ho = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
            const double element_ns = 1500.0 + nk * (1.2 * double(k) + 3.0);
            const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
                Data d{y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2), X.stride(-1),
                       has_w ? w->data() + off[2] : nullptr, has_w ? w->stride(-1) : 0,
                       has_u ? u->data() + off[2] : nullptr, has_u ? u->stride(-1) : 0,
                       has_g ? g->data() + off[3] : nullptr, has_g ? g->stride(-1) : 0};
                Dest o{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ho.as<double>() + i * n};
                return fit_one(pb, d, o, sc);
            });
            if (f) raise_failure(b, f, fn);
            return linalg::to_array(ho);
        }
        Out po = linalg::alloc_c(with(b.shape(), {k}), DType::Float64);
        Out co = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out ro = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
        Out fo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out eo = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
        Out ko = linalg::alloc_c(with(b.shape(), {k}), DType::Bool);
        Out so = linalg::alloc_c(with(b.shape(), {kNStat}), DType::Float64);
        // estimated serial ns of one regression: the QR, the passes and the fixed cost
        double element_ns = 1500.0 + nk * (0.6 * double(k) + 3.0 * (2 + steps)) + (cov ? 2.0 * nk * k : 0.0);
        if (cov == kHAC) element_ns += nk * (maxlags >= 0 ? double(maxlags) : 4.0 * std::pow(double(n) / 100.0, 2.0 / 9.0));
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            Data d{y.data() + off[0], y.stride(-1), X.data() + off[1], X.stride(-2), X.stride(-1),
                   has_w ? w->data() + off[2] : nullptr, has_w ? w->stride(-1) : 0,
                   has_u ? u->data() + off[2] : nullptr, has_u ? u->stride(-1) : 0,
                   has_g ? g->data() + off[3] : nullptr, has_g ? g->stride(-1) : 0};
            Dest o{po.as<double>() + i * k, co.as<double>() + i * k * k, ro.as<double>() + i * k * k,
                   fo.as<double>() + i * n, eo.as<double>() + i * n, ko.as<std::uint8_t>() + i * k,
                   so.as<double>() + i * kNStat, nullptr};
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

// resid_stats(e (..., n), w (..., n) | None) -> (..., 4): over the rows with w > 0 (every row
// without w), in order: Durbin-Watson's sum (e_t - e_{t-1})^2 over consecutive such rows and
// the central sums sum d^p, d = e - mean(e), p = 2, 3, 4 (Jarque-Bera), all in double-double
// (special/dd.hpp) and rounded once. Chunks of rows fixed by n are added in order: the same
// bits for any thread count. NaN in e gives NaN.
nb::object resid_stats(nb::handle eo, nb::handle wo) {
    const char* fn = "resid_stats";
    return call_named(fn, [&]() -> nb::object {
        Input e(eo, fn, "e");
        const bool has_w = !wo.is_none();
        std::optional<Input> w;
        if (has_w) w.emplace(wo, fn, "weights");
        const lint n = e.shape(-1);
        const linalg::BatchOperand none{};
        Batch b({{e.batch_shape(1), e.batch_strides(1)},
                 has_w ? linalg::BatchOperand{w->batch_shape(1), w->batch_strides(1)} : none},
                fn);
        Out so = linalg::alloc_c(with(b.shape(), {4}), DType::Float64);
        constexpr lint kChunk = 4096;
        const double element_ns = 100.0 + 40.0 * double(n);
        linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            const lint nch = (n + kChunk - 1) / kChunk;
            double* x = sc.get<double>(static_cast<std::size_t>(std::max<lint>(n, 1)) +
                                       static_cast<std::size_t>(std::max<lint>(nch, 1)) * 8);
            double* part = x + std::max<lint>(n, 1);
            lint m = 0;  // the rows of the fit, compacted
            for (lint t = 0; t < n; ++t) {
                if (has_w && !(rd(w->data() + off[1] + t * w->stride(-1)) > 0)) continue;
                x[m++] = rd(e.data() + off[0] + t * e.stride(-1));
            }
            double* out = so.as<double>() + i * 4;
            if (m == 0) {
                std::fill(out, out + 4, kNaN);
                return 0;
            }
            const lint nc = (m + kChunk - 1) / kChunk;
            auto chunks = [&](auto&& body) {
                if (nc >= 2 && m >= (1 << 16)) {
                    parallel::parallel_for(nc, [&](std::int64_t c) { body(static_cast<lint>(c)); });
                } else {
                    for (lint c = 0; c < nc; ++c) body(c);
                }
            };
            auto total = [&](int q) {
                DD acc{0.0, 0.0};
                for (lint c = 0; c < nc; ++c) acc = acc + DD{part[c * 8 + 2 * q], part[c * 8 + 2 * q + 1]};
                return acc;
            };
            // the sum (for the mean) and Durbin-Watson's sum
            chunks([&](lint c) {
                const lint t0 = c * kChunk, t1 = std::min(m, t0 + kChunk);
                DD s1{0.0, 0.0}, dw{0.0, 0.0};
                for (lint t = t0; t < t1; ++t) {
                    s1 = s1 + DD{x[t], 0.0};
                    if (t > 0) {
                        double h, l;
                        two_sum(x[t], -x[t - 1], h, l);
                        const DD d = dd_of(h, l);
                        dw = dw + d * d;
                    }
                }
                part[c * 8 + 0] = s1.hi, part[c * 8 + 1] = s1.lo, part[c * 8 + 2] = dw.hi, part[c * 8 + 3] = dw.lo;
            });
            const DD mean = special::dd::operator/(total(0), DD{static_cast<double>(m), 0.0});
            out[0] = total(1).hi;
            chunks([&](lint c) {
                const lint t0 = c * kChunk, t1 = std::min(m, t0 + kChunk);
                DD a2{0.0, 0.0}, a3{0.0, 0.0}, a4{0.0, 0.0};
                for (lint t = t0; t < t1; ++t) {
                    const DD d = DD{x[t], 0.0} - mean;
                    const DD d2 = d * d;
                    a2 = a2 + d2;
                    a3 = a3 + d2 * d;
                    a4 = a4 + d2 * d2;
                }
                part[c * 8 + 0] = a2.hi, part[c * 8 + 1] = a2.lo, part[c * 8 + 2] = a3.hi, part[c * 8 + 3] = a3.lo;
                part[c * 8 + 4] = a4.hi, part[c * 8 + 5] = a4.lo;
            });
            out[1] = total(0).hi;
            out[2] = total(1).hi;
            out[3] = total(2).hi;
            return 0;
        });
        return linalg::to_array(so);
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
          nb::arg("u").none(), nb::arg("intercept"), nb::arg("rtol"), nb::arg("cov"), nb::arg("steps"),
          nb::arg("groups").none() = nb::none(), nb::arg("gtot") = 0, nb::arg("maxlags") = -1,
          nb::arg("hat_only") = false);
    m.def("apply", &apply, nb::arg("func"), nb::arg("X").none(), nb::arg("beta").none(), nb::arg("intercept"),
          nb::arg("y").none());
    m.def("cond", &cond, nb::arg("R").none(), nb::arg("kept").none());
    m.def("resid_stats", &resid_stats, nb::arg("e").none(), nb::arg("w").none());
}

}  // namespace sd::regression
