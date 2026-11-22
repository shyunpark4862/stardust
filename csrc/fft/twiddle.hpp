// Twiddle factors w = e^(-2 pi i k / n) from the exact fraction k / n (docs/design/fft.md §3.3).
//
// k mod n is folded by exact integer arithmetic onto an angle theta in [0, pi/4]: with
// o = floor(8k / n) and r = 8k - o n (0 <= r < n), the angle 2 pi k / n is o pi/4 + (pi/4)(r / n).
// An even octant o is the quarter turn o/2 plus theta = (pi/4)(r / n); an odd one is the quarter
// turn (o + 1)/2 minus theta' = (pi/4)((n - r) / n). theta is formed as a double-double (pi/4 as
// a double-double times the double-double of r / n), sin and cos of its high part come from one
// call (base/sincos.hpp) and the low part corrects them to first order (sin(h + l) = sin h +
// l cos h, cos(h + l) = cos h - l sin h; the dropped l^2 terms are below 2^-106). A quarter turn is
// an exact swap and negation. So w depends on k mod n and n alone, and each part is within about
// one ULP of the true value (milestone FT1 measures every plan's table against Arb).
//
// Milestone FT0 wrote this for the prototype of csrc/fft/cfft.hpp; FT1 keeps it for the plans.
#pragma once

#include <cmath>
#include <cstdint>

#include "base/sincos.hpp"

namespace sd::fft {

struct Twiddle {
    double re, im;
};

namespace detail {

// pi/4 as a double-double: hi = RN(pi/4), lo = RN(pi/4 - hi)
inline constexpr double kQuarterPiHi = 0x1.921fb54442d18p-1;
inline constexpr double kQuarterPiLo = 0x1.1a62633145c07p-55;

// (s, c) of theta = (pi/4)(num / den), 0 <= num <= den, den < 2^53: the double-double of num / den
// (q + e, e from the exact remainder num - q den), times pi/4 as a double-double.
inline SinCos octant_sincos(int64_t num, int64_t den) {
    if (num == 0) return {0.0, 1.0};
    const double dn = static_cast<double>(num), dd = static_cast<double>(den);
    const double q = dn / dd;
    const double e = std::fma(-q, dd, dn) / dd;  // num - q den is exact (num, den < 2^53)
    const double h = kQuarterPiHi * q;
    const double l = std::fma(kQuarterPiHi, q, -h) + (kQuarterPiHi * e + kQuarterPiLo * q);
    const double th = h + l;
    const double tl = l - (th - h);
    const SinCos sc = sd::sincos(th);
    return {sc.s + tl * sc.c, sc.c - tl * sc.s};
}

}  // namespace detail

// e^(-2 pi i k / n) for any integer k and n >= 1 (n < 2^53 / 8).
inline Twiddle twiddle(int64_t k, int64_t n) {
    k %= n;
    if (k < 0) k += n;
    const int64_t o = (8 * k) / n;
    const int64_t r = 8 * k - o * n;
    SinCos a;     // (sin, cos) of the angle measured from the quarter turn q
    int64_t q;    // quarter turns
    if (o % 2 == 0) {
        q = o / 2;
        a = detail::octant_sincos(r, n);
    } else {
        q = (o + 1) / 2;
        const SinCos b = detail::octant_sincos(n - r, n);
        a = {-b.s, b.c};
    }
    double c, s;  // cos and sin of 2 pi k / n
    switch (q & 3) {
        case 0: c = a.c, s = a.s; break;
        case 1: c = -a.s, s = a.c; break;
        case 2: c = -a.c, s = -a.s; break;
        default: c = a.s, s = -a.c; break;
    }
    return {c, -s};
}

}  // namespace sd::fft
