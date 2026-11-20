// Double-double linear algebra for the multivariate and matrix distributions (T9, T10;
// docs/design/stats.md §6): a matrix factored in double already moves a quadratic form or
// trace by ~kappa eps (it is the factor of A + E, |E| ~ d eps |A|); in double-double that is
// ~kappa 1e-32.
#pragma once

#include "stats/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace sd::stats::ddla {

using num::DD;
namespace dd = num::dd;

// sqrt of a positive double-double: s = sqrt(hi), then one Newton correction (a - s^2) / 2s.
inline DD sqrt(DD a) {
    const double s = std::sqrt(a.hi);
    const DD s2 = dd::two_prod(s, s);
    const double corr = ((a.hi - s2.hi) - s2.lo + a.lo) / (2.0 * s);
    return dd::two_sum(s, corr);
}

// log1p(u) for a double-double u > -1: the series for |u| < 2^-10 (1 + u would drop u's
// low bits), else log of 1 + u.
inline DD log1p(DD u) {
    if (std::fabs(u.hi) < 0x1p-10) {  // sum (-1)^(k+1) u^k / k
        DD sum = u, pw = u;
        for (int k = 2; k < 40; ++k) {
            pw = pw * u;
            const DD t = pw / static_cast<double>(k);
            sum = (k % 2 == 0) ? sum - t : sum + t;
            if (std::fabs(t.hi) <= 0x1p-110 * std::fabs(sum.hi)) break;
        }
        return sum;
    }
    return dd::log(u + 1.0);
}

// The lower Cholesky factor L (d x d, row-major, the upper part left 0) of the matrix whose
// lower triangle a(i, j) gives (i >= j), and log det = 2 sum log L_jj. False where a pivot is
// not positive or not finite.
template <class A>
bool cholesky(std::int64_t d, A&& a, std::vector<DD>& L, DD& logdet) {
    L.assign(static_cast<std::size_t>(d * d), DD{0.0, 0.0});
    logdet = DD{0.0, 0.0};
    for (std::int64_t j = 0; j < d; ++j) {
        DD s{a(j, j), 0.0};
        for (std::int64_t k = 0; k < j; ++k) s = s - L[j * d + k] * L[j * d + k];
        if (!(s.hi > 0.0) || !std::isfinite(s.hi)) return false;
        const DD ljj = sqrt(s);
        L[j * d + j] = ljj;
        logdet = logdet + dd::log(ljj) * 2.0;
        for (std::int64_t i = j + 1; i < d; ++i) {
            DD t{a(i, j), 0.0};
            for (std::int64_t k = 0; k < j; ++k) t = t - L[i * d + k] * L[j * d + k];
            if (!std::isfinite(t.hi)) return false;
            L[i * d + j] = t / ljj;
        }
    }
    return true;
}

// b <- L^-1 b (forward substitution), L lower d x d row-major, b a vector of d (stride s).
inline void solve_lower(const std::vector<DD>& L, std::int64_t d, DD* b, std::int64_t s = 1) {
    for (std::int64_t i = 0; i < d; ++i) {
        DD t = b[i * s];
        for (std::int64_t k = 0; k < i; ++k) t = t - L[i * d + k] * b[k * s];
        b[i * s] = t / L[i * d + i];
    }
}

// The inverse of L L^T (d x d, row-major, full): L^-T L^-1 from the columns of L^-1.
inline void inverse_from_cholesky(const std::vector<DD>& L, std::int64_t d, std::vector<DD>& inv) {
    std::vector<DD> Li(static_cast<std::size_t>(d * d), DD{0.0, 0.0});  // L^-1, lower
    for (std::int64_t c = 0; c < d; ++c) {
        std::vector<DD> e(static_cast<std::size_t>(d), DD{0.0, 0.0});
        e[c] = DD{1.0, 0.0};
        solve_lower(L, d, e.data());
        for (std::int64_t i = 0; i < d; ++i) Li[i * d + c] = e[i];
    }
    inv.assign(static_cast<std::size_t>(d * d), DD{0.0, 0.0});
    for (std::int64_t i = 0; i < d; ++i)
        for (std::int64_t j = 0; j <= i; ++j) {
            DD s{0.0, 0.0};
            for (std::int64_t k = i; k < d; ++k) s = s + Li[k * d + i] * Li[k * d + j];
            inv[i * d + j] = s;
            inv[j * d + i] = s;
        }
}

// ---- the fast path of the quadratic forms ------------------------------------------------------
//
// |L^-1 r|^2 by double-double forward substitution costs ~1 ns per entry of L (a chain of
// compensated updates per row). The fast path multiplies by W = L^-1 instead, W computed once
// per factor in double-double and rounded to double: z = W r is then d dot products in plain
// double with no recurrence, and every rounding error enters z once, not amplified along the
// substitution. What is left is the cancellation inside each dot product, and that is bounded
// per point, rigorously: the caller accepts the plain result only where the bound fits its
// budget and runs the double-double substitution for every other point.
//
// The bound. u = 2^-53. Wh is the rounded W; a row's dot product is eight interleaved fma
// chains (terms k = j mod 8) summed by a fixed tree, so a row in block b = i / 8 (at most
// 8 (b + 1) terms) makes at most m = b + 4 roundings on any term's path (Higham, Accuracy and
// Stability of Numerical Algorithms, §3.1: the error is at most gamma_m sum |terms|); the
// first eight rows have one term per chain and the tree adds zeros exactly, which leaves
// m = 1 + ceil(log2(i + 1)) (kFirstM). With
// the input rh given to within er (|rh - r| <= er componentwise)
//     |zh_i - z_i| <= gamma_m v_i + u v_i + (|Wh| er)_i + eta (|W| |L| |W| |r|)_i,
//     v_i = sum_k |Wh_ik| |rh_k|,
// the second term W's rounding, the last the double-double substitution that made W
// (eta = d 2^-96: 2^8 times the nominal d 2^-104 of d double-double operations). The sums over
// k < i are bounded without a second pass by Cauchy-Schwarz,
//     sum_{k<i} |Wh_ik| |rh_k| <= |Wh_i,<i|_2 |rh_<i|_2,
// the row norms kept per factor and the prefix norm of rh taken through the end of i's block
// (an upper bound; one sqrt per block). The diagonal term is kept apart so a diagonal factor
// gives v_i exactly. (|W||L| v)_i <= max_k (|W||L|)_ik sum_k v_k, and sum_k v_k <= K |rh|_2
// with a constant K of the factor. Underflow adds at most m 2^-1074 to a row: nothing against a budget of
// 1e-15 max(1, |l|).
//
// Layout: W's rows and the vectors r, z, e are padded with zeros to dp = padded(d), a multiple
// of eight, so the dot products have no remainders; a zero term adds nothing and rounds nothing.

constexpr double kU = 0x1p-53;
// The error budget per point of the densities built on this path (multivariate normal and t,
// matrix normal; logpdf in the contract's mixed units, pdf relative): 40 units of 2^-53 =
// 4.4e-15 of the contract's 1e-14, a user decision of 2026-11-16 for these methods alone
// (every other fast path keeps the engine's kFastBudget = 16). A worst-case bound of a dot
// product cannot go below (m + 2) u sum |terms| (m = 4..16 roundings on a path, W's and r's
// own), which is 14-22 units for points drawn from the distribution itself and twice that
// through the matrix normal's two products, while the measured difference from the
// double-double path stays below 5e-16 (bench/stats/opt/e/mv_sweep.py at commit 4151d2a).
constexpr double kQuadBudget = 40.0 * kU;
constexpr double kBoundSlack = 1.0 + 0x1p-20;  // over the rounding of the bounds' own arithmetic
// roundings on a term's path in dot8 over n <= 8 nonzero leading terms, by n - 1
constexpr double kFirstM[8] = {1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0};

inline std::int64_t padded(std::int64_t d) { return (d + 7) & ~static_cast<std::int64_t>(7); }
// doubles per factor: W (d rows of dp), then |W_ii|, the off-diagonal row norms and
// eta K max_k (|W||L|)_ik, dp each
inline std::int64_t fast_size(std::int64_t d) { return padded(d) * (d + 3); }

// sum a_k b_k over 8 * blocks terms: eight fma chains, the chains in adjacent pairs as the
// vector lanes, ((01 + 23) + (45 + 67)) and the two lanes last. The scalar form gives the
// same bits.
inline double dot8(const double* a, const double* b, std::int64_t blocks) {
#if defined(__aarch64__)
    float64x2_t s0 = vdupq_n_f64(0.0), s1 = s0, s2 = s0, s3 = s0;
    for (std::int64_t q = 0; q < blocks; ++q, a += 8, b += 8) {
        s0 = vfmaq_f64(s0, vld1q_f64(a), vld1q_f64(b));
        s1 = vfmaq_f64(s1, vld1q_f64(a + 2), vld1q_f64(b + 2));
        s2 = vfmaq_f64(s2, vld1q_f64(a + 4), vld1q_f64(b + 4));
        s3 = vfmaq_f64(s3, vld1q_f64(a + 6), vld1q_f64(b + 6));
    }
    const float64x2_t t = vaddq_f64(vaddq_f64(s0, s1), vaddq_f64(s2, s3));
    return vgetq_lane_f64(t, 0) + vgetq_lane_f64(t, 1);
#else
    double s[8] = {};
    for (std::int64_t q = 0; q < blocks; ++q, a += 8, b += 8)
        for (int j = 0; j < 8; ++j) s[j] = std::fma(a[j], b[j], s[j]);
    return ((s[0] + s[2]) + (s[4] + s[6])) + ((s[1] + s[3]) + (s[5] + s[7]));
#endif
}

// sum a_k b_k over any n terms: dot8 of the whole blocks, then the rest in order.
inline double dotn(const double* a, const double* b, std::int64_t n) {
    const std::int64_t blocks = n / 8;
    double s = dot8(a, b, blocks);
    for (std::int64_t k = 8 * blocks; k < n; ++k) s += a[k] * b[k];
    return s;
}

// dot8 of four rows of the same length against one b (the rows' loads are the bottleneck of a
// row alone: two per fma; four rows share b's: 0.11 ns per fma of two terms, the load ports'
// limit). With half, the rows' last four terms of the last block are known to be zero and
// skipped (adding them changes nothing). Each out[q] has the bits of dot8(a_q, b, blocks + half).
inline void dot8x4(
    const double* a0,
    const double* a1,
    const double* a2,
    const double* a3,
    const double* b,
    std::int64_t blocks,
    bool half,
    double* out
) {
#if defined(__aarch64__)
    const float64x2_t zero = vdupq_n_f64(0.0);
    float64x2_t s00 = zero, s01 = zero, s02 = zero, s03 = zero, s10 = zero, s11 = zero, s12 = zero, s13 = zero;
    float64x2_t s20 = zero, s21 = zero, s22 = zero, s23 = zero, s30 = zero, s31 = zero, s32 = zero, s33 = zero;
    for (std::int64_t q = 0; q < blocks; ++q, a0 += 8, a1 += 8, a2 += 8, a3 += 8, b += 8) {
        const float64x2_t b0 = vld1q_f64(b), b1 = vld1q_f64(b + 2), b2 = vld1q_f64(b + 4), b3 = vld1q_f64(b + 6);
        s00 = vfmaq_f64(s00, vld1q_f64(a0), b0), s01 = vfmaq_f64(s01, vld1q_f64(a0 + 2), b1);
        s02 = vfmaq_f64(s02, vld1q_f64(a0 + 4), b2), s03 = vfmaq_f64(s03, vld1q_f64(a0 + 6), b3);
        s10 = vfmaq_f64(s10, vld1q_f64(a1), b0), s11 = vfmaq_f64(s11, vld1q_f64(a1 + 2), b1);
        s12 = vfmaq_f64(s12, vld1q_f64(a1 + 4), b2), s13 = vfmaq_f64(s13, vld1q_f64(a1 + 6), b3);
        s20 = vfmaq_f64(s20, vld1q_f64(a2), b0), s21 = vfmaq_f64(s21, vld1q_f64(a2 + 2), b1);
        s22 = vfmaq_f64(s22, vld1q_f64(a2 + 4), b2), s23 = vfmaq_f64(s23, vld1q_f64(a2 + 6), b3);
        s30 = vfmaq_f64(s30, vld1q_f64(a3), b0), s31 = vfmaq_f64(s31, vld1q_f64(a3 + 2), b1);
        s32 = vfmaq_f64(s32, vld1q_f64(a3 + 4), b2), s33 = vfmaq_f64(s33, vld1q_f64(a3 + 6), b3);
    }
    if (half) {
        const float64x2_t b0 = vld1q_f64(b), b1 = vld1q_f64(b + 2);
        s00 = vfmaq_f64(s00, vld1q_f64(a0), b0), s01 = vfmaq_f64(s01, vld1q_f64(a0 + 2), b1);
        s10 = vfmaq_f64(s10, vld1q_f64(a1), b0), s11 = vfmaq_f64(s11, vld1q_f64(a1 + 2), b1);
        s20 = vfmaq_f64(s20, vld1q_f64(a2), b0), s21 = vfmaq_f64(s21, vld1q_f64(a2 + 2), b1);
        s30 = vfmaq_f64(s30, vld1q_f64(a3), b0), s31 = vfmaq_f64(s31, vld1q_f64(a3 + 2), b1);
    }
    const float64x2_t t0 = vaddq_f64(vaddq_f64(s00, s01), vaddq_f64(s02, s03));
    const float64x2_t t1 = vaddq_f64(vaddq_f64(s10, s11), vaddq_f64(s12, s13));
    const float64x2_t t2 = vaddq_f64(vaddq_f64(s20, s21), vaddq_f64(s22, s23));
    const float64x2_t t3 = vaddq_f64(vaddq_f64(s30, s31), vaddq_f64(s32, s33));
    vst1q_f64(out, vpaddq_f64(t0, t1));  // lane 0 + lane 1 of each
    vst1q_f64(out + 2, vpaddq_f64(t2, t3));
#else
    const double* a[4] = {a0, a1, a2, a3};
    for (int q = 0; q < 4; ++q) out[q] = dot8(a[q], b, blocks + half);
#endif
}

// W = L^-1 and the bound's constants into out (fast_size(d) doubles). False where the inverse
// is not finite (the fast path is then off for this factor).
inline bool fast_prep(const std::vector<DD>& L, std::int64_t d, double* out) {
    const std::int64_t dp = padded(d);
    double *W = out, *wd = out + d * dp, *rn = wd + dp, *ec = rn + dp;
    std::fill(out, out + fast_size(d), 0.0);
    std::vector<DD> col(static_cast<std::size_t>(d));
    for (std::int64_t c = 0; c < d; ++c) {  // column c of L^-1: L y = e_c, rows c..d-1
        for (std::int64_t i = c; i < d; ++i) {
            DD t{i == c ? 1.0 : 0.0, 0.0};
            for (std::int64_t k = c; k < i; ++k) t = t - L[i * d + k] * col[k];
            col[i] = t / L[i * d + i];
            const double w = col[i].hi + col[i].lo;
            if (!std::isfinite(w)) return false;
            W[i * dp + c] = w;
        }
    }
    // sum_k v_k <= |r|_2 K, K = |diag W|_2 + sum_i |W_i,<i|_2 (v_i <= |W_ii| |r_i| + |W_i,<i|_2 |r|_2)
    double sd2 = 0.0, srn = 0.0;
    for (std::int64_t i = 0; i < d; ++i) {
        double s = 0.0;
        for (std::int64_t k = 0; k < i; ++k) s += W[i * dp + k] * W[i * dp + k];
        wd[i] = std::fabs(W[i * dp + i]);
        rn[i] = std::sqrt(s) * kBoundSlack;
        sd2 += wd[i] * wd[i], srn += rn[i];
    }
    const double eta_k = static_cast<double>(d) * 0x1p-96 * (std::sqrt(sd2) + srn) * kBoundSlack;
    std::vector<double> cl(static_cast<std::size_t>(d));
    for (std::int64_t i = 0; i < d; ++i) {
        double cmax = 0.0;
        std::fill(cl.begin(), cl.end(), 0.0);  // row i of |W||L|
        for (std::int64_t j = 0; j <= i; ++j) {
            const double w = std::fabs(W[i * dp + j]);
            for (std::int64_t k = 0; k <= j; ++k) cl[k] += w * std::fabs(L[j * d + k].hi);
        }
        for (std::int64_t k = 0; k <= i; ++k) cmax = std::max(cmax, cl[k]);
        ec[i] = eta_k * cmax * kBoundSlack;
        if (!std::isfinite(rn[i]) || !std::isfinite(ec[i])) return false;
    }
    return true;
}

// z = W r in plain double with its error bound. r, z, e: dp doubles (r zero past d). er:
// |r - r_exact| <= er componentwise (dp doubles), or nullptr for u |r| (each r_i rounded
// once). On return |z_i - (L^-1 r_exact)_i| <= e_i. Any non-finite value makes a bound
// non-finite or NaN: the caller's test then fails.
inline void fast_solve(const double* f, std::int64_t d, const double* r, const double* er, double* z, double* e) {
    const std::int64_t dp = padded(d);
    const double *W = f, *wd = f + d * dp, *rn = wd + dp, *ec = rn + dp;
    double sr = 0.0, se = 0.0, nr = 0.0;  // sums of r^2 and er^2 through the block, sqrt(sr)
    auto row = [&](std::int64_t i) { return W + std::min(i, d - 1) * dp; };  // past d: any row, the result dropped
    for (std::int64_t i0 = 0; i0 < d; i0 += 8) {
        const std::int64_t blocks = i0 / 8 + 1, n = std::min<std::int64_t>(8, d - i0);
        // rows i0..i0+3 have at most 8 blocks - 4 terms
        double zz[8] = {};
        dot8x4(row(i0), row(i0 + 1), row(i0 + 2), row(i0 + 3), r, blocks - 1, true, zz);
        if (n > 4) dot8x4(row(i0 + 4), row(i0 + 5), row(i0 + 6), row(i0 + 7), r, blocks, false, zz + 4);
        for (std::int64_t j = 0; j < 8; ++j) z[i0 + j] = j < n ? zz[j] : 0.0;
        sr += dot8(r + i0, r + i0, 1);
        // gamma_m + u (W's rounding; + u for r's own), second-order terms in the half
        const double g0 = (er ? 1.5 : 2.5), gm = static_cast<double>(blocks) + 3.0;
        double g[8];
        for (int j = 0; j < 8; ++j) g[j] = ((i0 == 0 ? kFirstM[j] : gm) + g0) * (kU * kBoundSlack);
        nr = std::sqrt(sr);  // |r|_2 through this block
        // e_i = g (|W_ii| |r_i| + |W_i,<i|_2 |r|_2) (+ |W_ii| er_i + |W_i,<i|_2 |er|_2); the
        // constants are zero past d
        const double *w = wd + i0, *rr = rn + i0, *ri = r + i0;
        double* eo = e + i0;
        if (er) {
            se += dot8(er + i0, er + i0, 1);
            const double ne = std::sqrt(se) * kBoundSlack;
            for (int j = 0; j < 8; ++j)
                eo[j] =
                    g[j] * (rr[j] * nr + w[j] * std::fabs(ri[j])) + (rr[j] * ne + w[j] * (er[i0 + j] * kBoundSlack));
        } else {
            for (int j = 0; j < 8; ++j) eo[j] = g[j] * (rr[j] * nr + w[j] * std::fabs(ri[j]));
        }
    }
    const double c2 = nr * kBoundSlack;  // |r|_2: ec_i c2 >= eta (|W||L| v)_i
    for (std::int64_t i = 0; i < dp; ++i) e[i] += ec[i] * c2;
}

// Q = |z|^2 and a bound on |Q - |z_exact|^2| from z's bounds e (dp doubles each, zero past
// d): sum (2 |z_i| + e_i) e_i plus the sum's own gamma_m Q, m = dp / 8 + 3 (kFirstM for d <= 8).
inline double fast_norm2(const double* z, const double* e, std::int64_t d, double& err) {
    const std::int64_t dp = padded(d);
    const double q = dot8(z, z, dp / 8);
    double b = 0.0;
    for (std::int64_t i0 = 0; i0 < dp; i0 += 8) {
        double t[8];
        for (int j = 0; j < 8; ++j) t[j] = 2.0 * std::fabs(z[i0 + j]) + e[i0 + j];
        b += dot8(t, e + i0, 1);
    }
    const double m = d <= 8 ? kFirstM[d - 1] : static_cast<double>(dp / 8) + 3.0;
    err = (b + (m + 0.5) * kU * q) * kBoundSlack;
    return q;
}

// log Gamma_p(a) = p (p - 1) / 4 log pi + sum_{j=1}^p log Gamma(a + (1 - j) / 2), double-double.
inline DD log_mvgamma(std::int64_t p, double a) {
    DD s = num::kLogPi * (0.25 * static_cast<double>(p * (p - 1)));
    for (std::int64_t j = 1; j <= p; ++j) s = s + num::log_gamma(dd::two_sum(a, 0.5 * static_cast<double>(1 - j)));
    return s;
}

}  // namespace sd::stats::ddla
