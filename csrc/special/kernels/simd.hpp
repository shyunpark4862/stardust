// One source for the NEON lanes and the scalar tail (docs/design/special.md
// §3.4). A kernel stage is written once as a template over V:
//   V = double   one element (the last odd element of a chunk)
//   V = F2       two float64 lanes of a NEON register
// Both overload the same IEEE operations -- add, multiply, divide, fused
// multiply-add only where written, sqrt, compare, select -- each rounding
// exactly as its scalar twin, so a lane and the scalar give the same bits.
#pragma once

#include <arm_neon.h>

#include <array>
#include <cmath>
#include <cstddef>

namespace sd::special::simd {

struct F2 {
    float64x2_t v;
};
struct M2 {
    uint64x2_t m;
};

inline F2 operator+(F2 a, F2 b) { return {vaddq_f64(a.v, b.v)}; }
inline F2 operator-(F2 a, F2 b) { return {vsubq_f64(a.v, b.v)}; }
inline F2 operator*(F2 a, F2 b) { return {vmulq_f64(a.v, b.v)}; }
inline F2 operator/(F2 a, F2 b) { return {vdivq_f64(a.v, b.v)}; }
inline F2 operator-(F2 a) { return {vnegq_f64(a.v)}; }
inline M2 operator<(F2 a, F2 b) { return {vcltq_f64(a.v, b.v)}; }
inline M2 operator<=(F2 a, F2 b) { return {vcleq_f64(a.v, b.v)}; }
inline M2 operator>(F2 a, F2 b) { return {vcgtq_f64(a.v, b.v)}; }
inline M2 operator>=(F2 a, F2 b) { return {vcgeq_f64(a.v, b.v)}; }
inline M2 operator==(F2 a, F2 b) { return {vceqq_f64(a.v, b.v)}; }
inline M2 operator&&(M2 a, M2 b) { return {vandq_u64(a.m, b.m)}; }

// Constants and memory
template <class V> V cst(double c);
template <> inline double cst<double>(double c) { return c; }
template <> inline F2 cst<F2>(double c) { return {vdupq_n_f64(c)}; }
inline double load(const double* p, double) { return *p; }
inline F2 load(const double* p, F2) { return {vld1q_f64(p)}; }
inline void store(double* p, double x) { *p = x; }
inline void store(double* p, F2 x) { vst1q_f64(p, x.v); }

// Elementwise functions
inline double fabs(double x) { return std::fabs(x); }
inline F2 fabs(F2 x) { return {vabsq_f64(x.v)}; }
inline double sqrt(double x) { return std::sqrt(x); }
inline F2 sqrt(F2 x) { return {vsqrtq_f64(x.v)}; }
inline double fma(double a, double b, double c) { return std::fma(a, b, c); }  // a * b + c, one rounding
inline F2 fma(F2 a, F2 b, F2 c) { return {vfmaq_f64(c.v, a.v, b.v)}; }
inline double select(bool m, double a, double b) { return m ? a : b; }
inline F2 select(M2 m, F2 a, F2 b) { return {vbslq_f64(m.m, a.v, b.v)}; }

// c[0] x^(N-1) + ... + c[N-1] by Horner's rule (Cephes polevl), and with a
// leading 1 (p1evl). Each step one fused multiply-add, r x + c[i] rounded once
// (the same helper for lanes and the scalar, so both give the same bits; one
// rounding fewer a step than a multiply and an add, and the chain about 1.6x
// shorter).
template <class V, std::size_t N>
V horner(V x, const std::array<double, N>& c) {
    V r = cst<V>(c[0]);
    for (std::size_t i = 1; i < N; ++i) r = fma(r, x, cst<V>(c[i]));
    return r;
}
template <class V, std::size_t N>
V horner1(V x, const std::array<double, N>& c) {
    V r = x + cst<V>(c[0]);
    for (std::size_t i = 1; i < N; ++i) r = fma(r, x, cst<V>(c[i]));
    return r;
}

}  // namespace sd::special::simd
