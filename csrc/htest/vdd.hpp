// Double-double arithmetic on two lanes of a NEON register (sd.htest's vectorized sums): the
// operations of special/dd.hpp (two_sum, quick_two_sum, two_prod by an FMA, DD + double, DD + DD,
// DD * DD) and htest/moments.hpp's Acc::add (acc), each the same IEEE operations as the scalar one in
// the same order, lane by lane (vfmaq_f64 is std::fma's single rounding), so a loop that keeps
// each chain's order gives the scalar loop's bits.
#pragma once

#include <arm_neon.h>

namespace sd::htest::vdd {

using V = float64x2_t;
struct D2 {
    V hi, lo;
};
inline D2 two_sum(V a, V b) {
    const V s = vaddq_f64(a, b), bb = vsubq_f64(s, a);
    return {s, vaddq_f64(vsubq_f64(a, vsubq_f64(s, bb)), vsubq_f64(b, bb))};
}
inline D2 quick_two_sum(V a, V b) {
    const V s = vaddq_f64(a, b);
    return {s, vsubq_f64(b, vsubq_f64(s, a))};
}
inline D2 add(D2 a, V b) {  // operator+(DD, double)
    const D2 s = two_sum(a.hi, b);
    return quick_two_sum(s.hi, vaddq_f64(s.lo, a.lo));
}
inline D2 add(D2 a, D2 b) {  // operator+(DD, DD)
    const D2 s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
    const D2 r = quick_two_sum(s.hi, vaddq_f64(s.lo, t.hi));
    return quick_two_sum(r.hi, vaddq_f64(r.lo, t.lo));
}
inline D2 neg(D2 a) { return {vnegq_f64(a.hi), vnegq_f64(a.lo)}; }
inline D2 mul(D2 a, D2 b) {  // operator*(DD, DD)
    const V p = vmulq_f64(a.hi, b.hi);
    V pl = vfmaq_f64(vnegq_f64(p), a.hi, b.hi);
    pl = vaddq_f64(pl, vaddq_f64(vmulq_f64(a.hi, b.lo), vmulq_f64(a.lo, b.hi)));
    return quick_two_sum(p, pl);
}
// Acc::add(x, y) of chains (hi, lo)
inline void acc(V& hi, V& lo, V x, V y) {
    const D2 s = two_sum(hi, x);
    hi = s.hi, lo = vaddq_f64(lo, vaddq_f64(s.lo, y));
}

}  // namespace sd::htest::vdd
