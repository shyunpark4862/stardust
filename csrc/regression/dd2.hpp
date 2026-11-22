// Double-double arithmetic on two lanes (NEON float64x2): the operations of special/dd.hpp
// lane by lane, for row passes that accumulate several sums per row (each lane a partial sum
// of every other row). Compiled with -ffp-contract=off: the fused products are the explicit ones.
#pragma once

#include <arm_neon.h>

namespace sd::regression::dd2 {

struct DD2 {
    float64x2_t hi, lo;
};

inline DD2 two_sum(float64x2_t a, float64x2_t b) {
    const float64x2_t s = vaddq_f64(a, b), bb = vsubq_f64(s, a);
    return {s, vaddq_f64(vsubq_f64(a, vsubq_f64(s, bb)), vsubq_f64(b, bb))};
}
inline DD2 quick_two_sum(float64x2_t a, float64x2_t b) {
    const float64x2_t s = vaddq_f64(a, b);
    return {s, vsubq_f64(b, vsubq_f64(s, a))};
}
inline DD2 two_prod(float64x2_t a, float64x2_t b) {
    const float64x2_t p = vmulq_f64(a, b);
    return {p, vfmaq_f64(vnegq_f64(p), a, b)};
}
inline DD2 add(DD2 a, DD2 b) {
    const DD2 s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
    const DD2 r = quick_two_sum(s.hi, vaddq_f64(s.lo, t.hi));
    return quick_two_sum(r.hi, vaddq_f64(r.lo, t.lo));
}
inline DD2 add(DD2 a, float64x2_t b) {
    const DD2 s = two_sum(a.hi, b);
    return quick_two_sum(s.hi, vaddq_f64(s.lo, a.lo));
}
inline DD2 mul(DD2 a, DD2 b) {
    DD2 p = two_prod(a.hi, b.hi);
    p.lo = vaddq_f64(p.lo, vaddq_f64(vmulq_f64(a.hi, b.lo), vmulq_f64(a.lo, b.hi)));
    return quick_two_sum(p.hi, p.lo);
}

}  // namespace sd::regression::dd2
