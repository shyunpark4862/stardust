// The linear models' numerical core (docs/design/regression.md §3.1-§3.3).
//
// One regression per batch element, through linalg's run_batch (one task per regression,
// the same bits for any thread count, docs/design/linalg.md §6):
//   1. The rows with a positive weight are the fit's (R's lm, G20). Each is whitened by
//      sqrt(w); the design, intercept column first, is packed column-major.
//   2. A tall, well-conditioned design with no aliased column: the Cholesky QR twice
//      (CholeskyQR2, G200; see cholesky_qr), R in A's top rows and the thin Q explicit.
//      Otherwise unpivoted Householder QR (geqrf). A column with |R_jj| <= rtol * ||x_j|| is
//      aliased (G9, G21; the rule of R's dqrdc2, so the earlier column stays): it is dropped
//      and the columns after it are factored again. When anything was dropped, the kept
//      columns are factored once more from scratch, so the fit is the reduced problem's, bit
//      for bit.
//   3. beta from R beta = (Q^T b)_1, then up to `steps` corrections (fewer when a further one
//      cannot matter, G15) on the augmented system
//      [I X; X^T W 0] [r; beta] = [y; 0] (Bjorck 1967): both blocks of its residual,
//      f = y - r - X beta and g = -X^T W r, are summed in double-double (Dot2, Ogita, Rump
//      and Oishi 2005), and the correction is solved with the same QR (G15).
//   4. Fitted values and residuals (every row, the zero-weight ones too) and the sums of
//      squares in double-double; the covariance (R^T R)^-1, or a sandwich R^-1 S R^-T whose
//      meat S is formed from the scores u_t = q_t e_t (q_t a row of the thin Q, e_t the
//      whitened residual): HC (sum omega_t q_t q_t^T), HAC (Newey-West lag sums with
//      Bartlett weights) or cluster (sums over groups) (docs/design/regression.md §3.3).
//   A second mode (hat_only) stops after step 2 and gives the hat values ||q_t||^2, the
//   leverages of the result's diagnostics: the same columns, the same factorization.
//
// Row sums are cut into chunks whose size depends on the number of columns only, and the
// chunks' partial sums are added in order: the result does not depend on the thread count
// or on whether the regression runs alone or inside a batch.
#include "regression/linear.hpp"
#include "regression/lsq.hpp"
#include "regression/dot2.hpp"
#include "regression/dd2.hpp"

#include "bind/errors.hpp"
#include "htest/normality.hpp"
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
// The Cholesky QR (fit_one): from this many rows x columns, and for a first factor's condition
// estimate (LAPACK's 1-norm) up to kCholQRMaxCond (CholeskyQR2's Q is orthonormal to rounding
// for kappa well below u^(-1/2) ~ 7e7; the margin covers the estimate's factor).
constexpr double kCholQRMinWork = 16384;
constexpr double kCholQRMaxCond = 1e5;
constexpr lint kCholQRDotCols = 32;  // Q^T Q by ddot up to this many columns (measured: (10^5, 51) dsyrk wins)
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
        if (o.qmat) {
            std::fill(o.qmat, o.qmat + pb.n * pb.k, kNaN);
            std::fill(o.R, o.R + pb.k * pb.k, kNaN);
            std::fill(o.kept, o.kept + pb.k, std::uint8_t(0));
        }
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
    const bool scan = !pb.lean && !pb.checked;
    for (lint i = 0; i < n && scan; ++i) {
        const double v = rd(d.y + i * d.ys);
        if (std::isnan(v)) nan = true;
        else if (std::isinf(v)) return kInfY;
    }
    if (scan)
        if (const int f = scan_x(d, n, pb.p, nan)) return f;
    if (pb.has_w && scan)
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
    if (m == 0 && !pb.lean && !pb.checked && !pb.hat_only) {
        // no row (none given, or every weight 0): a NaN result with nobs 0 (G96)
        fill_nan(pb, o);
        o.stats[kNobs] = 0.0;
        return 0;
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
    // tall enough for the Cholesky QR (below; its explicit thin Q lives in Q's slot)
    const bool try_cholqr = k >= 1 && m >= 2 * k && double(m) * double(k) >= kCholQRMinWork;
    std::size_t total = 0;
    auto reserve = [&](std::int64_t cnt) {
        const std::size_t at = total;
        total += linalg::pad2(cnt);
        return at;
    };
    const std::size_t o_s = reserve(mm), o_A = reserve(mm * kk0), o_tau = reserve(kk0), o_nrm = reserve(kk0),
                      o_beta = reserve(kk0), o_c = reserve(mm), o_f = reserve(mm), o_r = reserve(mm), o_P = reserve(mm),
                      o_S = reserve(mm), o_h = reserve(kk0), o_g = reserve(kk0), o_AH = reserve(newton ? mm : 0),
                      o_AL = reserve(newton ? mm : 0),
                      o_part = reserve(std::max<lint>(nchunks, 1) * 2 * (kk0 + kAccInit + kAccFinal)),
                      o_work = reserve(kk0), o_RR = reserve(kk0 * kk0), o_M = reserve(robust ? kk0 * kk0 : 0),
                      o_W2 = reserve(robust ? kk0 * kk0 : 0), o_Q = reserve(needQ || try_cholqr ? mm * kk0 : 0),
                      o_hat = reserve(needQ ? mm : 0), o_refl = reserve(mm / kReflChunk + 1),
                      o_sb = reserve(try_cholqr ? mm : 0), o_d1 = reserve(try_cholqr ? kk0 : 0);
    double* base = sc.get<double>(total);
    double *s = base + o_s, *A = base + o_A, *tau = base + o_tau, *nrm = base + o_nrm, *beta = base + o_beta,
           *cv = base + o_c, *f = base + o_f, *r = base + o_r, *P = base + o_P, *S = base + o_S, *h = base + o_h,
           *g = base + o_g, *part = base + o_part, *work = base + o_work, *RR = base + o_RR, *M = base + o_M,
           *work2 = base + o_W2, *Q = base + o_Q, *hat = base + o_hat, *refl = base + o_refl, *AH = base + o_AH,
           *AL = base + o_AL, *sbuf = base + o_sb, *dsave = base + o_d1;
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
    // The Cholesky QR twice (CholeskyQR2) of the packed A when it is well conditioned: R1 from
    // the Cholesky factor of A^T A, Q1 = A R1^-1, then the same on Q1 (Q = Q1 R2^-1, R = R2 R1),
    // by dsyrk, potrf and dtrsm (BLAS 3; the QR of a tall matrix is level-2 bound). For
    // kappa(A) u^(1/2) << 1 the second pass leaves Q orthonormal and A - Q R at rounding, as the
    // Householder QR. Taken only when the first factor's condition estimate is at most
    // kCholQRMaxCond and no column meets the aliasing rule (with a margin of 2); otherwise (or
    // when a factorization fails) the Householder QR below decides, so aliased and
    // ill-conditioned designs (NIST StRD) keep that path. R goes into A's top rows (where the
    // Householder QR leaves it) and the thin Q into Q; tau is unused.
    bool explicit_q = false;
    auto cholesky_qr = [&]() -> bool {
        double* Qx = Q;
        std::copy(A, A + m * k, Qx);
        std::vector<double> R1(static_cast<std::size_t>(k * k), 0.0), R2(static_cast<std::size_t>(k * k), 0.0);
        auto gram_chol = [&](std::vector<double>& G) {
            // Q^T Q: one ddot per pair of columns for a few columns (Accelerate's dsyrk on a
            // tall, narrow matrix ran 6-7x slower: (10^5, 11) 860 us against 129), else dsyrk
            if (k <= kCholQRDotCols) {
                for (lint b = 0; b < k; ++b)
                    for (lint a = 0; a <= b; ++a)
                        G[b * k + a] = cblas_ddot(static_cast<int>(m), Qx + a * m, 1, Qx + b * m, 1);
            } else {
                cblas_dsyrk(
                    CblasColMajor, CblasUpper, CblasTrans, static_cast<int>(k), static_cast<int>(m), 1.0, Qx,
                    static_cast<int>(m), 0.0, G.data(), static_cast<int>(k)
                );
            }
            return linalg::lapack::potrf('U', k, G.data(), k) == 0;
        };
        if (!gram_chol(R1)) return false;
        // the norms from the factor: ||a_j||^2 = sum_i R1_ij^2 (column j of R1)
        for (lint j = 0; j < k; ++j) {
            double s2 = 0.0;
            for (lint i = 0; i <= j; ++i) s2 += R1[j * k + i] * R1[j * k + i];
            nrm[j] = std::sqrt(s2);
        }
        const double rc = linalg::lapack::trcon_1('U', k, R1.data(), k);
        if (!(rc > 0.0) || 1.0 / rc > kCholQRMaxCond) return false;
        for (lint j = 0; j < k; ++j)
            if (!(std::fabs(R1[j * k + j]) > 2.0 * tol * nrm[j])) return false;
        cblas_dtrsm(
            CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, static_cast<int>(m), static_cast<int>(k),
            1.0, R1.data(), static_cast<int>(k), Qx, static_cast<int>(m)
        );
        if (!gram_chol(R2)) return false;
        cblas_dtrsm(
            CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, static_cast<int>(m), static_cast<int>(k),
            1.0, R2.data(), static_cast<int>(k), Qx, static_cast<int>(m)
        );
        // R = R2 R1 into A's top rows (upper triangle)
        for (lint j = 0; j < k; ++j)
            for (lint i = j + 1; i < k; ++i) R1[j * k + i] = 0.0, R2[j * k + i] = 0.0;
        cblas_dtrmm(
            CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, static_cast<int>(k), static_cast<int>(k),
            1.0, R2.data(), static_cast<int>(k), R1.data(), static_cast<int>(k)
        );
        for (lint j = 0; j < k; ++j)
            for (lint i = 0; i < k; ++i) A[j * m + i] = i <= j ? R1[j * k + i] : 0.0;
        return true;
    };
    if (m > 0 && k > 0) {
        pack(A, K, k);
        explicit_q = try_cholqr && cholesky_qr();
        // the column norms of the aliasing rule (the Cholesky QR takes them from A^T A's diagonal):
        // one dnrm2 call per column (on the pool when large), the same bits for any thread count
        if (!explicit_q) {
            if (k >= 2 && double(m) * k >= double(1 << 20)) {
                parallel::parallel_for(k, [&](std::int64_t c) { nrm[c] = cblas_dnrm2(m, A + c * m, 1); });
            } else {
                for (lint c = 0; c < k; ++c) nrm[c] = cblas_dnrm2(m, A + c * m, 1);
            }
        }
    }
    if (m > 0 && k > 0 && !explicit_q) {
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
    } else if (!(m > 0 && k > 0)) {
        kk = 0;
    }

    // the hat values h_t = ||q_t||^2 from the thin Q of the fit's own QR (0 for the rows out
    // of the fit: their weighted leverage w_t x_t (X^T W X)^-1 x_t^T is 0)
    // the thin Q into Q (the Cholesky QR's is there already)
    auto form_q = [&]() {
        if (explicit_q) return;
        std::copy(A, A + m * kk, Q);
        linalg::lapack::orgqr(m, kk, kk, Q, m, tau, sc);
    };
    auto hat_values = [&]() {
        form_q();
        std::fill(hat, hat + m, 0.0);
        for (lint jj = 0; jj < kk; ++jj)
            for (lint t = 0; t < m; ++t) hat[t] += Q[jj * m + t] * Q[jj * m + t];
    };
    // T = Q^T diag(hdiff/w) Q (want_T, newton) into Tk (kk x kk column-major), summed by dgemm
    // over fixed chunks of rows in order; copied to o.T when asked
    std::vector<double> Tk;
    auto make_T = [&](bool with_q = true) {
        Tk.assign(static_cast<std::size_t>(kk * kk), 0.0);
        if (kk == 0) return;
        if (with_q) form_q();
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
                for (lint t = t0; t < t1; ++t)
                    Bc[jj * len + (t - t0)] = Q[jj * m + t] * (d.hdiff[idx[t]] / (s[t] * s[t]));
            cblas_dgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, kk, kk, len, 1.0, Q + t0, m, Bc.data(), len, 0.0,
                parts.data() + static_cast<std::size_t>(c) * slot, kk
            );
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
            for (lint ii = 0; ii < jj; ++ii)
                Tk[jj * kk + ii] = Tk[ii * kk + jj] = 0.5 * (Tk[jj * kk + ii] + Tk[ii * kk + jj]);
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
        if (o.qmat) {  // the thin Q, R and the kept columns, in their places
            std::fill(o.qmat, o.qmat + n * k, 0.0);
            std::fill(o.R, o.R + k * k, 0.0);
            std::fill(o.kept, o.kept + k, std::uint8_t(0));
            for (lint jj = 0; jj < kk; ++jj) {
                o.kept[K[jj]] = 1;
                for (lint t = 0; t < m; ++t) o.qmat[idx[t] * k + K[jj]] = Q[jj * m + t];
                for (lint ii = 0; ii <= jj; ++ii) o.R[K[ii] * k + K[jj]] = A[jj * m + ii];
            }
        }
        return 0;
    }

    // ---- 3. solve and refine
    auto colK = [&](lint jj) { return col(K[jj]); };
    std::vector<dot2::ColRef> CK(static_cast<std::size_t>(std::max<lint>(kk, 1)));
    for (lint jj = 0; jj < kk; ++jj) CK[jj] = {colK(jj).base, colK(jj).stride};
    // a row-major X: the row passes go along rows (NEON lanes over rows for x beta), else
    // along the columns (each a contiguous run); the same bits either way
    const bool rowwise = std::llabs(d.xcs) < std::llabs(d.xrs);
    // x beta for rows [t0, t1) into (P, S) (Dot2)
    auto xbeta = [&](lint t0, lint t1, const double* b) {
        if (rowwise) {
            dot2::dot2_rows(CK.data(), b, kk, t1 - t0, [&](lint t) { return idx[t0 + t]; }, P + t0, S + t0);
            return;
        }
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
            // each column's sum over the rows in order (NEON lanes over columns)
            dot2::dot2_cols(
                CK.data(), kk, t1 - t0, [&](lint t) { return idx[t0 + t]; },
                [&](lint t, double& ah, double& al) { ah = AH[t0 + t], al = AL[t0 + t]; }, gp
            );
            return;
        }
        dot2::dot2_cols(
            CK.data(), kk, t1 - t0, [&](lint t) { return idx[t0 + t]; },
            [&](lint t, double& ah, double& al) {
                ah = r[t0 + t], al = 0.0;
                if (pb.has_w) two_prod(weight(idx[t0 + t]), r[t0 + t], ah, al);
            },
            gp
        );
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

    // c := Q^T c and c := Q c of the factorization: the reflectors, or with the Cholesky QR's
    // thin Q1 a pair: Q^T c gives d1 = Q1^T c (keeping c), and Q [c1; d2] (d2 the rest of Q^T c,
    // untouched) is c + Q1 (c1 - d1), the identity Q2 Q2^T = I - Q1 Q1^T
    std::vector<double> work2q(static_cast<std::size_t>(kk0));
    auto qt_apply = [&](double* c) {
        if (!explicit_q) return apply_reflectors(A, m, kk, tau, c, true, refl);
        std::copy(c, c + m, sbuf);
        cblas_dgemv(
            CblasColMajor, CblasTrans, static_cast<int>(m), static_cast<int>(kk), 1.0, Q, static_cast<int>(m), sbuf, 1,
            0.0, dsave, 1
        );
        std::copy(dsave, dsave + kk, c);
    };
    auto q_apply = [&](double* c) {
        if (!explicit_q) return apply_reflectors(A, m, kk, tau, c, false, refl);
        for (lint jj = 0; jj < kk; ++jj) work2q[jj] = c[jj] - dsave[jj];
        std::copy(sbuf, sbuf + m, c);
        cblas_dgemv(
            CblasColMajor, CblasNoTrans, static_cast<int>(m), static_cast<int>(kk), 1.0, Q, static_cast<int>(m),
            work2q.data(), 1, 1.0, c, 1
        );
    };
    // initial solution
    if (kk > 0) {
        for (lint t = 0; t < m; ++t) cv[t] = s[t] * rd(d.y + idx[t] * d.ys);
        qt_apply(cv);
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
        // four interleaved partial sums per quantity (rows t mod 4), added in order at the end:
        // independent chains instead of one double-double chain through every row
        DD a4[kAccInit][4] = {};
        for (lint t = t0; t < t1; ++t) {
            const lint i = idx[t];
            const double yv = rd(d.y + i * d.ys);
            double hh, e;
            two_sum(yv, -P[t], hh, e);
            const double v = e - S[t];
            r[t] = hh + v;
            f[t] = (hh - r[t]) + v;
            DD* a = &a4[0][t & 3];
            if (!pb.has_w && !pb.has_u) {
                a[kSwuy * 4] = a[kSwuy * 4] + yv;  // w u = 1: the same terms as below
                a[kSwuu * 4] = a[kSwuu * 4] + 1.0;
                continue;
            }
            const double wt = weight(i), ut = uval(i);
            double ph, pl;
            two_prod(wt, ut, ph, pl);
            const DD wu = dd_of(ph, pl);
            a[kSwuy * 4] = a[kSwuy * 4] + wu * yv;
            a[kSwuu * 4] = a[kSwuu * 4] + wu * ut;
            if (pb.has_w) a[kSlogw * 4] = a[kSlogw * 4] + std::log(wt);
        }
        DD a[kAccInit];
        for (int q = 0; q < kAccInit; ++q) a[q] = (a4[q][0] + a4[q][1]) + (a4[q][2] + a4[q][3]);
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
    // kappa u (measured, bench/regression/g1/refine.log at commit 4151d2a), so after a correction d the error
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
        qt_apply(cv);
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
        q_apply(cv);
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
        // four interleaved partial sums per quantity (rows t mod 4), added in order at the end;
        // without weights (and u) the factors w and u are 1 and their products are skipped (the
        // same values)
        DD a4[kAccFinal][4] = {};
        const bool unit = !pb.has_w && !pb.has_u;
        lint t = t0;
        if (unit) {
            // two rows per vector (lanes), two vectors: the same partial sums by rows mod 4
            using namespace dd2;
            const float64x2_t z = vdupq_n_f64(0.0);
            DD2 v4[kAccFinal][2];
            for (auto& q : v4) q[0] = q[1] = DD2{z, z};
            const DD2 nyb{vdupq_n_f64(-ybar.hi), vdupq_n_f64(-ybar.lo)};
            for (; t + 4 <= t1; t += 4) {
                for (int h = 0; h < 2; ++h) {
                    const lint ta = t + 2 * h;
                    const lint i0 = idx[ta], i1 = idx[ta + 1];
                    const float64x2_t yv = {rd(d.y + i0 * d.ys), rd(d.y + i1 * d.ys)};
                    const float64x2_t Pv = vld1q_f64(P + ta), Sv = vld1q_f64(S + ta);
                    const DD2 hs = two_sum(yv, vnegq_f64(Pv));
                    const float64x2_t v = vsubq_f64(hs.lo, Sv);
                    const float64x2_t rr = vaddq_f64(hs.hi, v);
                    const DD2 res = two_sum(rr, vaddq_f64(vsubq_f64(hs.hi, rr), v));
                    const DD2 fit = two_sum(Pv, Sv);
                    o.fitted[i0] = vgetq_lane_f64(fit.hi, 0), o.fitted[i1] = vgetq_lane_f64(fit.hi, 1);
                    o.resid[i0] = vgetq_lane_f64(rr, 0), o.resid[i1] = vgetq_lane_f64(rr, 1);
                    vst1q_f64(r + ta, rr);
                    DD2* q = &v4[0][h];
                    q[fSsr * 2] = add(q[fSsr * 2], mul(res, res));
                    const DD2 yc = add(DD2{yv, z}, nyb);
                    q[fCtss * 2] = add(q[fCtss * 2], mul(yc, yc));
                    const DD2 yy = two_prod(yv, yv);
                    q[fUtss * 2] = add(q[fUtss * 2], two_sum(yy.hi, yy.lo));
                    const DD2 fc = add(fit, nyb);
                    q[fEssC * 2] = add(q[fEssC * 2], mul(fc, fc));
                    q[fEssU * 2] = add(q[fEssU * 2], mul(fit, fit));
                }
            }
            // the lanes: rows mod 4 = 0, 1 (vector 0) and 2, 3 (vector 1)
            for (int q = 0; q < kAccFinal; ++q)
                for (int h = 0; h < 2; ++h) {
                    a4[q][2 * h] = DD{vgetq_lane_f64(v4[q][h].hi, 0), vgetq_lane_f64(v4[q][h].lo, 0)};
                    a4[q][2 * h + 1] = DD{vgetq_lane_f64(v4[q][h].hi, 1), vgetq_lane_f64(v4[q][h].lo, 1)};
                }
        }
        for (; t < t1; ++t) {
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
            DD* a = &a4[0][t & 3];
            double yh, yl;
            two_prod(yv, yv, yh, yl);
            if (unit) {
                a[fSsr * 4] = a[fSsr * 4] + res * res;
                const DD yc = DD{yv, 0.0} - ybar;
                a[fCtss * 4] = a[fCtss * 4] + yc * yc;
                a[fUtss * 4] = a[fUtss * 4] + dd_of(yh, yl);
                const DD fc = fit - ybar;
                a[fEssC * 4] = a[fEssC * 4] + fc * fc;
                a[fEssU * 4] = a[fEssU * 4] + fit * fit;
                continue;
            }
            const double wt = weight(i), ut = uval(i);
            a[fSsr * 4] = a[fSsr * 4] + (res * res) * wt;
            const DD yc = DD{yv, 0.0} - ybar * ut;
            a[fCtss * 4] = a[fCtss * 4] + (yc * yc) * wt;
            a[fUtss * 4] = a[fUtss * 4] + dd_of(yh, yl) * wt;
            const DD fc = fit - ybar * ut;
            a[fEssC * 4] = a[fEssC * 4] + (fc * fc) * wt;
            a[fEssU * 4] = a[fEssU * 4] + (fit * fit) * wt;
        }
        DD a[kAccFinal];
        for (int q = 0; q < kAccFinal; ++q) a[q] = (a4[q][0] + a4[q][1]) + (a4[q][2] + a4[q][3]);
        double* ap = accs(ch) + 2 * kAccInit;
        for (int q = 0; q < kAccFinal; ++q) ap[2 * q] = a[q].hi, ap[2 * q + 1] = a[q].lo;
    });
    DD fin[kAccFinal] = {};
    for (lint ch = 0; ch < nchunks; ++ch) {
        const double* ap = accs(ch) + 2 * kAccInit;
        for (int q = 0; q < kAccFinal; ++q) fin[q] = fin[q] + DD{ap[2 * q], ap[2 * q + 1]};
    }
    // A perfect fit (G95): every used row's residual within rounding of its terms,
    // |r_t| <= (kk + 16) u (|y_t| + sum_j |x_tj| |beta_j|) (rlm's and quantreg's test): rounding
    // beta moves x_t beta by about u sum_j |x_tj beta_j|, and the residual itself is computed in
    // double-double. Its residuals are rounding, not data: the sum of squares is 0 (scale 0,
    // bse 0 and llf +inf when df_resid > 0; t and p NaN, G22) and the robust covariances'
    // scores vanish. The test stops at the first row outside it (usually the first).
    if (kk > 0 && fin[fSsr].hi > 0.0) {
        const double c = static_cast<double>(kk + 16) * 0x1p-53;
        bool perfect = true;
        for (lint jj = 0; jj < kk; ++jj) perfect = perfect && std::isfinite(beta[jj]);
        for (lint t = 0; t < m && perfect; ++t) {
            const lint i = idx[t];
            double xb = 0.0;
            for (lint jj = 0; jj < kk; ++jj) {
                const Col cl = colK(jj);
                xb += std::fabs(rd(cl.base + i * cl.stride) * beta[jj]);
            }
            perfect = std::fabs(r[t]) <= c * (std::fabs(rd(d.y + i * d.ys)) + xb);
        }
        if (perfect) {
            fin[fSsr] = DD{0.0, 0.0};
            std::fill(r, r + m, 0.0);
        }
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
        form_q();
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
        // a GLM row of frequency f stands for f copies, each with the leverage h / f (the
        // repeated data's value)
        const double ht = d.smult ? hat[t] * (d.smult[idx[t]] * d.smult[idx[t]]) : hat[t];
        if (pb.cov == kHC2) sq = e / std::sqrt(1.0 - ht);
        else if (pb.cov == kHC3) sq = e / (1.0 - ht);
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
        // the default from nobs: the rows used, or the GLM's sum of frequency weights (with
        // frequency weights the lags stay those of the rows, statsmodels' convention)
        const std::int64_t L = pb.maxlags >= 0
                                   ? pb.maxlags
                                   : static_cast<std::int64_t>(std::floor(4.0 * std::pow(n_eff / 100.0, 2.0 / 9.0)));
        st[kLags] = static_cast<double>(L);
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, kk, m, 1.0, Q, m, 0.0, M, kk);
        const lint Le = static_cast<lint>(std::min<std::int64_t>(L, m - 1));  // lags with a pair of rows
        if (Le > 0) {
            double* wl = sc.get<double>(
                static_cast<std::size_t>(Le + 1) + static_cast<std::size_t>(kHacChunk * kk) +
                    static_cast<std::size_t>(kk * kk),
                3
            );
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
                cblas_dgemm(
                    CblasColMajor, CblasTrans, CblasNoTrans, kk, kk, len, 1.0, Q + t0, m, Vc, kHacChunk, 1.0, C, kk
                );
            }
            for (lint jj = 0; jj < kk; ++jj)
                for (lint ii = 0; ii <= jj; ++ii) M[jj * kk + ii] += C[jj * kk + ii] + C[ii * kk + jj];
        }
    } else {
        // one-way cluster: S = sum_g s_g s_g^T, s_g the sum of the group's scores (rows in
        // order), with Stata's and statsmodels' factor G/(G-1) (m-1)/(m-k); G counts the groups
        // among the fit's rows (zero-weight rows are out, G20)
        // A batch's codes run over the labels of every regression (gtot), so a table of gtot
        // entries per regression would cost the batch's labels for each one: above m labels,
        // the codes are first made dense by sorting this regression's own (O(m log m)). Either
        // way the local numbers are in order of first appearance (the same sums, the same bits).
        const bool dense = pb.gtot > m;
        const std::size_t gt =
            dense ? static_cast<std::size_t>(mm) : static_cast<std::size_t>(std::max<std::int64_t>(pb.gtot, 1));
        // ints: the codes' local numbers, each row's group, G + 1 offsets (G <= m), a permutation
        lint* loc = sc.get<lint>(gt + 3 * static_cast<std::size_t>(mm) + 1, 2);
        lint* lab = loc + gt;
        lint* off = lab + mm;
        lint* perm = off + mm + 1;
        std::fill(loc, loc + gt, lint(-1));
        std::vector<std::int64_t> uniq;
        if (dense) {
            uniq.resize(static_cast<std::size_t>(m));
            for (lint t = 0; t < m; ++t) uniq[static_cast<std::size_t>(t)] = rd_i64(d.g + idx[t] * d.gs);
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        }
        lint G = 0;
        for (lint t = 0; t < m; ++t) {
            std::int64_t code = rd_i64(d.g + idx[t] * d.gs);
            if (dense) code = std::lower_bound(uniq.begin(), uniq.end(), code) - uniq.begin();
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
    // a cluster meat has rank at most G - 1 (the group sums of the scores add to X^T e = 0):
    // with fewer than q + 1 groups S22 is singular and the test undefined (api §2), though its
    // rounding can still pass the Cholesky
    if (q > 0 && !pb.glm && pb.cov == kCluster && st[kNGroups] - 1.0 < static_cast<double>(q)) {
        st[kWaldF] = kNaN;
    } else if (q > 0 && !pb.glm) {
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
            cblas_dtrsm(
                CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, kk, kk, 1.0, work2, kk, M, kk
            );
            cblas_dtrsm(
                CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, kk, kk, 1.0, work2, kk, M, kk
            );
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
//     groups (..., n) | None, gtot, maxlags, hat_only, thin_q)
// -> (params, cov, R, fitted, resid, kept, stats), or with hat_only the hat values (..., n);
// with hat_only and thin_q (hat, Q (..., n, k), R (..., k, k), kept (..., k)): the thin Q of the
// fit's QR, its R and the kept columns (the influence measures and the sequential sums of
// squares, docs/design/regression.md §3.8).
// Inputs are float64 sd.ndarrays (the Python layer converts); u is GLS's whitened column of
// ones (it replaces the intercept column and centers the sums of squares); groups are int64
// codes 0 .. gtot - 1, -1 for a missing label (cov = cluster).
nb::object fit(
    const std::string& func,
    nb::handle yo,
    nb::handle xo,
    nb::handle wo,
    nb::handle uo,
    bool intercept,
    double rtol,
    int cov,
    int steps,
    nb::handle go,
    std::int64_t gtot,
    std::int64_t maxlags,
    bool hat_only,
    bool thin_q
) {
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
        Batch b(
            {{y.batch_shape(1), y.batch_strides(1)},
             {X.batch_shape(2), X.batch_strides(2)},
             // the weights (WLS) and the whitened ones (GLS) never come together: one operand
             has_w   ? linalg::BatchOperand{w->batch_shape(1), w->batch_strides(1)}
             : has_u ? linalg::BatchOperand{u->batch_shape(1), u->batch_strides(1)}
                     : none,
             has_g ? linalg::BatchOperand{g->batch_shape(1), g->batch_strides(1)} : none},
            fn
        );
        const lint n = pb.n, k = pb.k;
        const double nk = double(n) * double(std::max<lint>(k, 1));
        if (hat_only) {
            Out ho = linalg::alloc_c(with(b.shape(), {n}), DType::Float64);
            Out qo, rqo, kqo;
            if (thin_q) {
                qo = linalg::alloc_c(with(b.shape(), {n, k}), DType::Float64);
                rqo = linalg::alloc_c(with(b.shape(), {k, k}), DType::Float64);
                kqo = linalg::alloc_c(with(b.shape(), {k}), DType::Bool);
            }
            const double element_ns = 1500.0 + nk * (1.2 * double(k) + 3.0);
            const Failure f =
                linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
                    Data d{
                        y.data() + off[0],
                        y.stride(-1),
                        X.data() + off[1],
                        X.stride(-2),
                        X.stride(-1),
                        has_w ? w->data() + off[2] : nullptr,
                        has_w ? w->stride(-1) : 0,
                        has_u ? u->data() + off[2] : nullptr,
                        has_u ? u->stride(-1) : 0,
                        has_g ? g->data() + off[3] : nullptr,
                        has_g ? g->stride(-1) : 0
                    };
                    Dest o{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ho.as<double>() + i * n};
                    if (thin_q) {
                        o.qmat = qo.as<double>() + i * n * k;
                        o.R = rqo.as<double>() + i * k * k;
                        o.kept = kqo.as<std::uint8_t>() + i * k;
                    }
                    return fit_one(pb, d, o, sc);
                });
            if (f) raise_failure(b, f, fn);
            if (thin_q)
                return nb::make_tuple(
                    linalg::to_array(ho), linalg::to_array(qo), linalg::to_array(rqo), linalg::to_array(kqo)
                );
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
        if (cov == kHAC)
            element_ns += nk * (maxlags >= 0 ? double(maxlags) : 4.0 * std::pow(double(n) / 100.0, 2.0 / 9.0));
        const Failure f = linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            Data d{
                y.data() + off[0],
                y.stride(-1),
                X.data() + off[1],
                X.stride(-2),
                X.stride(-1),
                has_w ? w->data() + off[2] : nullptr,
                has_w ? w->stride(-1) : 0,
                has_u ? u->data() + off[2] : nullptr,
                has_u ? u->stride(-1) : 0,
                has_g ? g->data() + off[3] : nullptr,
                has_g ? g->stride(-1) : 0
            };
            Dest o{po.as<double>() + i * k,      co.as<double>() + i * k * k,
                   ro.as<double>() + i * k * k,  fo.as<double>() + i * n,
                   eo.as<double>() + i * n,      ko.as<std::uint8_t>() + i * k,
                   so.as<double>() + i * kNStat, nullptr};
            return fit_one(pb, d, o, sc);
        });
        if (f) raise_failure(b, f, fn);
        return nb::make_tuple(
            linalg::to_array(po), linalg::to_array(co), linalg::to_array(ro), linalg::to_array(fo),
            linalg::to_array(eo), linalg::to_array(ko), linalg::to_array(so)
        );
    });
}

// apply(func, X (..., m, p), beta (..., k), intercept, y (..., m) | None, mask (..., k) | None)
// -> (X beta, y - X beta | None), each in double-double rounded once (Dot2, the column order
// of the fit: predict on the fit's own X gives its fitted values bit for bit). mask: 1.0 for
// the columns in the model, 0.0 for the aliased ones, which take no part (R: a NaN in X_new
// there does not reach the prediction); None: every column. A column in the model takes part
// whatever its coefficient (a NaN in X_new under a coefficient 0 gives NaN).
nb::object apply(const std::string& func, nb::handle xo, nb::handle bo, bool intercept, nb::handle yo, nb::handle mo) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        Input X(xo, fn, "X"), be(bo, fn, "params");
        const bool has_y = !yo.is_none(), has_m = !mo.is_none();
        std::optional<Input> y, mk;
        if (has_y) y.emplace(yo, fn, "y");
        if (has_m) mk.emplace(mo, fn, "mask");
        const lint m = X.shape(-2), p = X.shape(-1), k = be.shape(-1);
        const linalg::BatchOperand none{};
        Batch b(
            {{X.batch_shape(2), X.batch_strides(2)},
             {be.batch_shape(1), be.batch_strides(1)},
             has_y ? linalg::BatchOperand{y->batch_shape(1), y->batch_strides(1)} : none,
             has_m ? linalg::BatchOperand{mk->batch_shape(1), mk->batch_strides(1)} : none},
            fn
        );
        Out fo = linalg::alloc_c(with(b.shape(), {m}), DType::Float64);
        Out eo;
        if (has_y) eo = linalg::alloc_c(with(b.shape(), {m}), DType::Float64);
        const double element_ns = 200.0 + 3.0 * double(m) * double(std::max<lint>(k, 1));
        linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch&) {
            const std::byte* xb = X.data() + off[0];
            const std::byte* bb = be.data() + off[1];
            double* fi = fo.as<double>() + i * m;
            double* ei = has_y ? eo.as<double>() + i * m : nullptr;
            const std::byte* mb = has_m ? mk->data() + off[3] : nullptr;
            // the columns with a term (an aliased one has none, nor in the fit) and their
            // coefficients; each row's Dot2 sum over them in order, eight rows at a time (each
            // lane the scalar operations of its row)
            std::vector<dot2::ColRef> cols;
            std::vector<double> bv;
            for (lint c = 0; c < k; ++c) {
                if (mb && rd(mb + c * mk->stride(-1)) == 0.0) continue;
                bv.push_back(rd(bb + c * be.stride(-1)));
                if (intercept && c == 0) cols.push_back({reinterpret_cast<const std::byte*>(&kOneValue), 0});
                else cols.push_back({xb + (c - (intercept ? 1 : 0)) * X.stride(-1), X.stride(-2)});
            }
            constexpr lint kBlk = 256;
            double P[kBlk], S[kBlk];
            for (lint t0 = 0; t0 < m; t0 += kBlk) {
                const lint nt = std::min(kBlk, m - t0);
                dot2::dot2_rows(
                    cols.data(), bv.data(), static_cast<lint>(cols.size()), nt, [&](lint t) { return t0 + t; }, P, S
                );
                for (lint u = 0; u < nt; ++u) {
                    const lint t = t0 + u;
                    const double Pv = P[u], Sv = S[u];
                    fi[t] = Pv + Sv;
                    if (ei) {
                        const double yv = rd(y->data() + off[2] + t * y->stride(-1));
                        double hh, e;
                        two_sum(yv, -Pv, hh, e);
                        ei[t] = hh + (e - Sv);
                    }
                }
            }
            return 0;
        });
        (void)p;
        nb::object eobj = has_y ? linalg::to_array(eo) : nb::none();
        return nb::make_tuple(linalg::to_array(fo), eobj);
    });
}

// resid_stats(e (..., n), w (..., n) | None) -> (..., 5): over the rows with w > 0 (every row
// without w), in order: Durbin-Watson's sum (e_t - e_{t-1})^2 over consecutive such rows, the
// central sums sum d^p, d = e - mean(e), p = 2, 3, 4 (Jarque-Bera) and the mean, all in
// double-double (special/dd.hpp) and rounded once. The central sums are sd.htest's kernel
// (htest/normality.hpp, H17): jarque_bera() takes its statistic from them as
// sd.htest.jarque_bera does. Chunks of rows fixed by n are added in order: the same bits for
// any thread count. NaN in e gives NaN.
nb::object resid_stats(nb::handle eo, nb::handle wo) {
    const char* fn = "resid_stats";
    return call_named(fn, [&]() -> nb::object {
        Input e(eo, fn, "e");
        const bool has_w = !wo.is_none();
        std::optional<Input> w;
        if (has_w) w.emplace(wo, fn, "weights");
        const lint n = e.shape(-1);
        const linalg::BatchOperand none{};
        Batch b(
            {{e.batch_shape(1), e.batch_strides(1)},
             has_w ? linalg::BatchOperand{w->batch_shape(1), w->batch_strides(1)} : none},
            fn
        );
        Out so = linalg::alloc_c(with(b.shape(), {5}), DType::Float64);
        constexpr lint kChunk = htest::kSumChunk;
        const double element_ns = 100.0 + 40.0 * double(n);
        linalg::run_batch(b, element_ns, [&](std::int64_t i, const std::int64_t* off, Scratch& sc) {
            const lint nch = (n + kChunk - 1) / kChunk;
            double* x = sc.get<double>(
                static_cast<std::size_t>(std::max<lint>(n, 1)) + static_cast<std::size_t>(std::max<lint>(nch, 1)) * 8
            );
            double* part = x + std::max<lint>(n, 1);
            lint m = 0;  // the rows of the fit, compacted
            for (lint t = 0; t < n; ++t) {
                if (has_w && !(rd(w->data() + off[1] + t * w->stride(-1)) > 0)) continue;
                x[m++] = rd(e.data() + off[0] + t * e.stride(-1));
            }
            double* out = so.as<double>() + i * 5;
            if (m == 0) {
                std::fill(out, out + 5, kNaN);
                return 0;
            }
            const lint nc = (m + kChunk - 1) / kChunk;
            // Durbin-Watson's sum
            htest::sum_chunks(m, [&](lint c) {
                const lint t0 = c * kChunk, t1 = std::min(m, t0 + kChunk);
                DD dw{0.0, 0.0};
                for (lint t = std::max<lint>(t0, 1); t < t1; ++t) {
                    double h, l;
                    two_sum(x[t], -x[t - 1], h, l);
                    const DD d = dd_of(h, l);
                    dw = dw + d * d;
                }
                part[c * 8 + 2] = dw.hi, part[c * 8 + 3] = dw.lo;
            });
            DD dw{0.0, 0.0};
            for (lint c = 0; c < nc; ++c) dw = dw + DD{part[c * 8 + 2], part[c * 8 + 3]};
            out[0] = dw.hi;
            const htest::CentralSums cs = htest::central_sums(x, m, part);
            out[1] = cs.s2.hi;
            out[2] = cs.s3.hi;
            out[3] = cs.s4.hi;
            out[4] = cs.mean.hi;
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

void bind_select(nb::module_& m);  // regression/select.cpp

// same_values(a, b) -> whether two float64 arrays of the same shape hold the same values
// (a == b, or both NaN) element by element, stopping at the first difference: the model
// comparisons' test of the same data (their columns usually differ in the first element).
nb::object same_values(nb::handle ao, nb::handle bo) {
    const char* fn = "same_values";
    return call_named(fn, [&]() -> nb::object {
        Input a(ao, fn, "a"), b(bo, fn, "b");
        const int nd = a.ndim();
        if (b.ndim() != nd) return nb::cast(false);
        std::int64_t total = 1;
        for (int d = 0; d < nd; ++d) {
            if (a.shape(d) != b.shape(d)) return nb::cast(false);
            total *= a.shape(d);
        }
        if (total == 0) return nb::cast(true);
        if (a.data() == b.data()) {
            bool same = true;
            for (int d = 0; d < nd && same; ++d) same = a.shape(d) == 1 || a.stride(d) == b.stride(d);
            if (same) return nb::cast(true);
        }
        std::vector<std::int64_t> idx(static_cast<std::size_t>(nd), 0);
        const lint inner = nd ? a.shape(nd - 1) : 1;
        const std::int64_t sa = nd ? a.stride(nd - 1) : 0, sb = nd ? b.stride(nd - 1) : 0;
        for (std::int64_t done = 0; done < total; done += inner) {
            std::int64_t oa = 0, ob = 0;
            for (int d = 0; d + 1 < nd; ++d) oa += idx[d] * a.stride(d), ob += idx[d] * b.stride(d);
            const std::byte* pa = a.data() + oa;
            const std::byte* pb = b.data() + ob;
            for (lint t = 0; t < inner; ++t) {
                const double x = rd(pa + t * sa), y = rd(pb + t * sb);
                if (!(x == y || (std::isnan(x) && std::isnan(y)))) return nb::cast(false);
            }
            for (int d = nd - 2; d >= 0; --d) {
                if (++idx[d] < a.shape(d)) break;
                idx[d] = 0;
            }
        }
        return nb::cast(true);
    });
}

void bind_linear(nb::module_& m) {
    m.def(
        "fit", &fit, nb::arg("func"), nb::arg("y").none(), nb::arg("X").none(), nb::arg("w").none(),
        nb::arg("u").none(), nb::arg("intercept"), nb::arg("rtol"), nb::arg("cov"), nb::arg("steps"),
        nb::arg("groups").none() = nb::none(), nb::arg("gtot") = 0, nb::arg("maxlags") = -1,
        nb::arg("hat_only") = false, nb::arg("thin_q") = false
    );
    m.def(
        "apply", &apply, nb::arg("func"), nb::arg("X").none(), nb::arg("beta").none(), nb::arg("intercept"),
        nb::arg("y").none(), nb::arg("mask").none() = nb::none()
    );
    m.def("cond", &cond, nb::arg("R").none(), nb::arg("kept").none());
    m.def("resid_stats", &resid_stats, nb::arg("e").none(), nb::arg("w").none());
    m.def("same_values", &same_values, nb::arg("a").none(), nb::arg("b").none());
    bind_select(m);
}

}  // namespace sd::regression
