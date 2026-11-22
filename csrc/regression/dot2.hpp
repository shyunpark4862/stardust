// Row passes of the regression area in double-double (Ogita, Rump and Oishi's Dot2), with
// NEON lanes over independent chains: each lane does exactly the scalar operations of its
// row (dot2_rows) or column (dot2_cols) in the scalar order, so the results are the same
// bits as the plain loops (docs/design/regression.md §3.1; compiled with -ffp-contract=off,
// so no product is fused that the scalar code does not fuse).
#pragma once

#include <arm_neon.h>

#include <Accelerate/Accelerate.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sd::regression::dot2 {

using lint = std::int64_t;

// A column of the design: element of row i at base + i * stride.
struct ColRef {
    const std::byte* base;
    std::int64_t stride;
};

inline double rd(const std::byte* p) {
    double v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

// One step of a row's sum: (hp, ep) = two_prod(x, b); (P, q) = two_sum(P, hp); S += q + ep.
inline void row_step(double& P, double& S, double x, double b) {
    const double hp = x * b, ep = std::fma(x, b, -hp);
    const double s = P + hp, bb = s - P;
    const double q = (P - (s - bb)) + (hp - bb);
    P = s;
    S += q + ep;
}
inline void row_step(float64x2_t& P, float64x2_t& S, float64x2_t x, float64x2_t b) {
    const float64x2_t hp = vmulq_f64(x, b), ep = vfmaq_f64(vnegq_f64(hp), x, b);
    const float64x2_t s = vaddq_f64(P, hp), bb = vsubq_f64(s, P);
    const float64x2_t q = vaddq_f64(vsubq_f64(P, vsubq_f64(s, bb)), vsubq_f64(hp, bb));
    P = s;
    S = vaddq_f64(S, vaddq_f64(q, ep));
}

// One step of a column's sum with the row factor (ah, al): (hp, ep) = two_prod(ah, x);
// (hi, q) = two_sum(hi, hp); lo += q + ep + al x.
inline void col_step(double& hi, double& lo, double ah, double al, double x) {
    const double hp = ah * x, ep = std::fma(ah, x, -hp);
    const double s = hi + hp, bb = s - hi;
    const double q = (hi - (s - bb)) + (hp - bb);
    hi = s;
    lo += q + ep + al * x;
}
inline void col_step(float64x2_t& hi, float64x2_t& lo, float64x2_t ah, float64x2_t al, float64x2_t x) {
    const float64x2_t hp = vmulq_f64(ah, x), ep = vfmaq_f64(vnegq_f64(hp), ah, x);
    const float64x2_t s = vaddq_f64(hi, hp), bb = vsubq_f64(s, hi);
    const float64x2_t q = vaddq_f64(vsubq_f64(hi, vsubq_f64(s, bb)), vsubq_f64(hp, bb));
    hi = s;
    lo = vaddq_f64(lo, vaddq_f64(vaddq_f64(q, ep), vmulq_f64(al, x)));
}

// x beta over the columns c[0 .. nc) for the rows row(t), t in [0, nt): P[t] + S[t], each
// row's sum from (p0, 0) over the columns in order (eight rows at a time, then four).
template <class Row>
inline void dot2_rows(
    const ColRef* c, const double* b, lint nc, lint nt, Row&& row, double* P, double* S, double p0 = 0.0
) {
    lint t = 0;
    // eight rows at a time: four independent chains hide the additions' latency (each lane's
    // operations are still its row's scalar ones, in order)
    for (; t + 8 <= nt; t += 8) {
        lint i[8];
        for (int u = 0; u < 8; ++u) i[u] = row(t + u);
        float64x2_t PA = vdupq_n_f64(p0), SA = vdupq_n_f64(0.0), PB = PA, SB = SA, PC = PA, SC = SA, PD = PA, SD = SA;
        for (lint j = 0; j < nc; ++j) {
            const std::byte* base = c[j].base;
            const std::int64_t st = c[j].stride;
            const float64x2_t bj = vdupq_n_f64(b[j]);
            const float64x2_t xa = {rd(base + i[0] * st), rd(base + i[1] * st)};
            const float64x2_t xb = {rd(base + i[2] * st), rd(base + i[3] * st)};
            const float64x2_t xc = {rd(base + i[4] * st), rd(base + i[5] * st)};
            const float64x2_t xd = {rd(base + i[6] * st), rd(base + i[7] * st)};
            row_step(PA, SA, xa, bj);
            row_step(PB, SB, xb, bj);
            row_step(PC, SC, xc, bj);
            row_step(PD, SD, xd, bj);
        }
        vst1q_f64(P + t, PA);
        vst1q_f64(P + t + 2, PB);
        vst1q_f64(P + t + 4, PC);
        vst1q_f64(P + t + 6, PD);
        vst1q_f64(S + t, SA);
        vst1q_f64(S + t + 2, SB);
        vst1q_f64(S + t + 4, SC);
        vst1q_f64(S + t + 6, SD);
    }
    for (; t + 4 <= nt; t += 4) {
        const lint i0 = row(t), i1 = row(t + 1), i2 = row(t + 2), i3 = row(t + 3);
        float64x2_t PA = vdupq_n_f64(p0), SA = vdupq_n_f64(0.0), PB = PA, SB = SA;
        for (lint j = 0; j < nc; ++j) {
            const std::byte* base = c[j].base;
            const std::int64_t st = c[j].stride;
            const float64x2_t bj = vdupq_n_f64(b[j]);
            const float64x2_t xa = {rd(base + i0 * st), rd(base + i1 * st)};
            const float64x2_t xb = {rd(base + i2 * st), rd(base + i3 * st)};
            row_step(PA, SA, xa, bj);
            row_step(PB, SB, xb, bj);
        }
        vst1q_f64(P + t, PA);
        vst1q_f64(P + t + 2, PB);
        vst1q_f64(S + t, SA);
        vst1q_f64(S + t + 2, SB);
    }
    for (; t < nt; ++t) {
        const lint i = row(t);
        double p = p0, s = 0.0;
        for (lint j = 0; j < nc; ++j) row_step(p, s, rd(c[j].base + i * c[j].stride), b[j]);
        P[t] = p;
        S[t] = s;
    }
}

// The column sums sum_t (ah_t + al_t) x_{row(t), j} over t in [0, nt), each column's (hi, lo)
// from (0, 0) over the rows in order (four columns at a time), into out[2 j], out[2 j + 1].
// (Interleaving the rows of the last columns made the refinement's fixed point drift on
// kappa = 1e11 designs, fitted values 1.3e-10 -> 2.3e-10 relative: kept in order.)
// fac(t, ah, al) gives row t's factor.
template <class Row, class Fac>
inline void dot2_cols(const ColRef* c, lint nc, lint nt, Row&& row, Fac&& fac, double* out) {
    lint j = 0;
    for (; j + 4 <= nc; j += 4) {
        const ColRef c0 = c[j], c1 = c[j + 1], c2 = c[j + 2], c3 = c[j + 3];
        float64x2_t hA = vdupq_n_f64(0.0), lA = hA, hB = hA, lB = hA;
        for (lint t = 0; t < nt; ++t) {
            const lint i = row(t);
            double ah, al;
            fac(t, ah, al);
            const float64x2_t vh = vdupq_n_f64(ah), vl = vdupq_n_f64(al);
            const float64x2_t xa = {rd(c0.base + i * c0.stride), rd(c1.base + i * c1.stride)};
            const float64x2_t xb = {rd(c2.base + i * c2.stride), rd(c3.base + i * c3.stride)};
            col_step(hA, lA, vh, vl, xa);
            col_step(hB, lB, vh, vl, xb);
        }
        double h[4], l[4];
        vst1q_f64(h, hA);
        vst1q_f64(h + 2, hB);
        vst1q_f64(l, lA);
        vst1q_f64(l + 2, lB);
        for (int q = 0; q < 4; ++q) out[2 * (j + q)] = h[q], out[2 * (j + q) + 1] = l[q];
    }
    for (; j < nc; ++j) {
        double hi = 0.0, lo = 0.0;
        for (lint t = 0; t < nt; ++t) {
            double ah, al;
            fac(t, ah, al);
            col_step(hi, lo, ah, al, rd(c[j].base + row(t) * c[j].stride));
        }
        out[2 * j] = hi;
        out[2 * j + 1] = lo;
    }
}

// x h in plain double over the columns, rows row(t), t in [0, nt): each row's sum from 0
// over the columns in order (eight rows at a time, then four: the independent chains hide
// the additions' latency), into out[t].
template <class Row>
inline void plain_rows(const ColRef* c, const double* h, lint nc, lint nt, Row&& row, double* out) {
    lint t = 0;
    for (; t + 8 <= nt; t += 8) {
        lint i[8];
        for (int u = 0; u < 8; ++u) i[u] = row(t + u);
        float64x2_t A = vdupq_n_f64(0.0), B = A, C = A, D = A;
        for (lint j = 0; j < nc; ++j) {
            const std::byte* base = c[j].base;
            const std::int64_t st = c[j].stride;
            const float64x2_t hj = vdupq_n_f64(h[j]);
            const float64x2_t xa = {rd(base + i[0] * st), rd(base + i[1] * st)};
            const float64x2_t xb = {rd(base + i[2] * st), rd(base + i[3] * st)};
            const float64x2_t xc = {rd(base + i[4] * st), rd(base + i[5] * st)};
            const float64x2_t xd = {rd(base + i[6] * st), rd(base + i[7] * st)};
            A = vaddq_f64(A, vmulq_f64(xa, hj));
            B = vaddq_f64(B, vmulq_f64(xb, hj));
            C = vaddq_f64(C, vmulq_f64(xc, hj));
            D = vaddq_f64(D, vmulq_f64(xd, hj));
        }
        vst1q_f64(out + t, A);
        vst1q_f64(out + t + 2, B);
        vst1q_f64(out + t + 4, C);
        vst1q_f64(out + t + 6, D);
    }
    for (; t + 4 <= nt; t += 4) {
        const lint i0 = row(t), i1 = row(t + 1), i2 = row(t + 2), i3 = row(t + 3);
        float64x2_t A = vdupq_n_f64(0.0), B = A;
        for (lint j = 0; j < nc; ++j) {
            const std::byte* base = c[j].base;
            const std::int64_t st = c[j].stride;
            const float64x2_t hj = vdupq_n_f64(h[j]);
            const float64x2_t xa = {rd(base + i0 * st), rd(base + i1 * st)};
            const float64x2_t xb = {rd(base + i2 * st), rd(base + i3 * st)};
            A = vaddq_f64(A, vmulq_f64(xa, hj));
            B = vaddq_f64(B, vmulq_f64(xb, hj));
        }
        vst1q_f64(out + t, A);
        vst1q_f64(out + t + 2, B);
    }
    for (; t < nt; ++t) {
        const lint i = row(t);
        double acc = 0.0;
        for (lint j = 0; j < nc; ++j) acc += rd(c[j].base + i * c[j].stride) * h[j];
        out[t] = acc;
    }
}

// out[t] = h0 + x_t h (h0 = h[0] with an intercept, the columns' coefficients after it) for
// rows [i0, i1) of X (p columns, all with a term) by one dgemv on X read in place, when X has a
// unit stride along its rows or its columns (C or F order); false (nothing written) otherwise.
// BLAS's summation order, not the scalar loop's: the bits depend on the layout (not on the
// thread count: one call per fixed chunk of rows).
inline bool gemv_rows(
    const std::byte* X,
    std::int64_t xrs,
    std::int64_t xcs,
    lint p,
    bool intercept,
    const double* h,
    lint i0,
    lint i1,
    double* out
) {
    constexpr std::int64_t e = static_cast<std::int64_t>(sizeof(double));
    const lint len = i1 - i0;
    if (p <= 0 || len <= 0) return false;
    const double* a = reinterpret_cast<const double*>(X + i0 * xrs);
    const double* b = h + (intercept ? 1 : 0);
    if (xcs == e && xrs % e == 0 && xrs / e >= p) {
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, static_cast<int>(len), static_cast<int>(p), 1.0, a, static_cast<int>(xrs / e),
            b, 1, 0.0, out, 1
        );
    } else if (xrs == e && xcs % e == 0 && xcs / e >= len) {
        cblas_dgemv(
            CblasColMajor, CblasNoTrans, static_cast<int>(len), static_cast<int>(p), 1.0, a, static_cast<int>(xcs / e),
            b, 1, 0.0, out, 1
        );
    } else {
        return false;
    }
    if (intercept)
        for (lint t = 0; t < len; ++t) out[t] += h[0];
    return true;
}

}  // namespace sd::regression::dot2
