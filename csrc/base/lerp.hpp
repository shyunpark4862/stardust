// The linear kernel that the array's sd.interp (array/ops/interp.cpp) and sd.interpolate's linear paths
// (interpolate/linear.cpp) share, so that every linear path gives the same bits (docs/design/interpolate.md
// §3.2; decisions IP4, IP26, IP27, IP38).
//
// A point x of the interval [x0, x1] whose ends carry the values a and b:
//
//   t = (x - x0) / (x1 - x0)                      lerp_t; halved when x1 - x0 or x - x0 overflows
//   t == 0 -> a, t == 1 -> b                      the ends, to the sign of zero
//   a, b of opposite signs (0 counts as either)   t b + (1 - t) a
//   otherwise                                     a + t (b - a), clamped at b on the side past b
//
// §3.2 proves what this keeps: the ends are exact, a == b gives a, the value is monotone in x and lies
// between a and b for 0 <= t <= 1, and |r - y| <= gamma_6 (|a| + |b|) + 3 eta there (gamma_10 |t| (|a| + |b|)
// + 3 eta outside), y the exact value and eta = 2^-1075. So the interval a point is given at a node does not
// change its value. Non-finite ends: inside, the infinite end (NaN for opposite infinities, NaN for a NaN
// end); outside, a + t (b - a) in IEEE arithmetic.
//
// Both forms are computed and the result selected (decision IP26): a branch on the signs of a and b and on t
// mispredicts on points in random order (3.7x numpy's time; this form is 1.2x, with the same bits). The
// compiler must not fuse the products into the sums (-ffp-contract=off, CMakeLists.txt): the proof counts
// one rounding per operation.
#pragma once

#include <cmath>
#include <complex>
#include <limits>

namespace sd {

// The position of x in [x0, x1] (x0 < x1, finite): fl(fl(x - x0) / fl(x1 - x0)), nondecreasing in x, 0 at
// x0 and 1 at x1. When the width or x - x0 overflows (a width near DBL_MAX, or a point far outside), the
// same with x, x0 and x1 halved: the halved differences do not overflow, and the halving changes neither
// the ends nor the order.
inline double lerp_t(double x, double x0, double x1) {
    const double w = x1 - x0, d = x - x0;
    if (std::isinf(w) || std::isinf(d)) [[unlikely]] {
        if (std::isinf(x)) return x;  // a point at +-inf: t is +-inf (the width is positive)
        return (0.5 * x - 0.5 * x0) / (0.5 * x1 - 0.5 * x0);
    }
    return d / w;
}

namespace detail {

// a or b is not finite (the caller checked). Inside: the sum picks the infinite end, NaN for opposite
// infinities or a NaN end (numpy's interp gives the same, tests/interpolate/data/numpy.json nonfinite/*).
inline double lerp_nonfinite(double a, double b, double t) {
    if (t == 0.0) return a;
    if (t == 1.0) return b;
    if (t > 0.0 && t < 1.0) return a + b;
    return a + t * (b - a);  // outside (or a NaN t): IEEE
}

}  // namespace detail

// The value at t between a (t = 0) and b (t = 1); outside [0, 1] the line through them.
inline double lerp(double a, double b, double t) {
    if (!(std::isfinite(a) && std::isfinite(b))) [[unlikely]]
        return detail::lerp_nonfinite(a, b, t);
    const double r1 = t * b + (1.0 - t) * a;  // opposite signs: no cancellation
    double r2 = a + t * (b - a);              // same sign: exact for a == b
    // r2 past b is cut back to b: for t <= 1 on b's side of a, for t > 1 on the other side
    r2 = ((t > 1.0) == (b > a)) ? std::fmax(r2, b) : std::fmin(r2, b);
    const bool opposite = ((a <= 0.0) & (b >= 0.0)) | ((a >= 0.0) & (b <= 0.0));  // bitwise: no branches
    double r = opposite ? r1 : r2;
    r = (t == 1.0) ? b : r;
    r = (t == 0.0) ? a : r;
    return (t != t) ? t : r;  // a NaN t (a NaN point) stays NaN: fmin and fmax would drop it
}

// complex values: the real and imaginary parts with the same t (numpy's interp does the same)
inline std::complex<double> lerp(std::complex<double> a, std::complex<double> b, double t) {
    return {lerp(a.real(), b.real(), t), lerp(a.imag(), b.imag(), t)};
}

}  // namespace sd
