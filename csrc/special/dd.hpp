// Double-double arithmetic for the exponents of the incomplete functions
// (docs/design/special.md §5.4; S4).
//
// x^a e^-x / Gamma(a) and x^a (1-x)^b / B(a, b) are e^-E times a modest
// factor, with E up to ~745 while the result is still a normal number. A
// relative error d in E becomes an error of E d in the result: float64's
// 2^-53 would cost up to 745 * 2^-53 / 2^-52 ~ 370 ULP. So E is formed as an
// unevaluated sum hi + lo (about 100 bits) from the exact inputs, and only
// e^-hi (1 - lo) is rounded.
//
// Only what the exponents need: sums and products (exact transformations),
// quotients, and log in two forms -- log(lambda) and
// mu(t) = t - log(1 + t) -- with ~2^-64 relative accuracy.
#pragma once

#include "special/coeffs/dd_log_table.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace sd::special::dd {

struct DD {
    double hi, lo;
};

inline DD two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
inline DD quick_two_sum(double a, double b) {  // |a| >= |b|
    const double s = a + b;
    return {s, b - (s - a)};
}
inline DD two_prod(double a, double b) {
    const double p = a * b;
    return {p, std::fma(a, b, -p)};
}

inline DD operator+(DD a, DD b) {
    const DD s = two_sum(a.hi, b.hi);
    const DD t = two_sum(a.lo, b.lo);
    DD r = quick_two_sum(s.hi, s.lo + t.hi);
    return quick_two_sum(r.hi, r.lo + t.lo);
}
inline DD operator-(DD a) { return {-a.hi, -a.lo}; }
inline DD operator-(DD a, DD b) { return a + (-b); }
inline DD operator+(DD a, double b) {
    const DD s = two_sum(a.hi, b);
    return quick_two_sum(s.hi, s.lo + a.lo);
}
inline DD operator-(DD a, double b) { return a + (-b); }

inline DD operator*(DD a, DD b) {
    DD p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}
inline DD operator*(DD a, double b) {
    DD p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    return quick_two_sum(p.hi, p.lo);
}

// Quotients: q1 and one correction from the exact remainder (fma), ~2^-104
// relative (a third correction term, 8 ns of 31, bought nothing the exponents use).
inline DD operator/(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    const double r = (std::fma(-q1, b.hi, a.hi) + a.lo) - q1 * b.lo;
    return quick_two_sum(q1, r / b.hi);
}
inline DD operator/(DD a, double b) {
    const double q1 = a.hi / b;
    return quick_two_sum(q1, (std::fma(-q1, b, a.hi) + a.lo) / b);
}

inline DD ldexp(DD a, int e) { return {std::ldexp(a.hi, e), std::ldexp(a.lo, e)}; }

// 2/3, 2/5 and log 2 as hi + lo.
inline constexpr DD kTwoThirds = {0.66666666666666663, 3.7007434154171886e-17};
inline constexpr DD kTwoFifths = {0.40000000000000002, -2.2204460492503132e-17};
inline constexpr DD kLn2 = {0.69314718055994529, 2.3190468138462996e-17};

// mu(t) = t - log(1 + t) for t in [-1/2, 1], t = hi + lo.
// With s = t / (2 + t), log(1 + t) = 2 atanh(s) = 2 (s + s^3/3 + s^5/5 + ...)
// and t - 2 s = s t exactly, so
//     mu = s t - 2 s^3 / 3 - 2 s^5 / 5 - 2 s^7 (1/7 + s^2/9 + ...)
// |s| <= 1/3: the first three terms in double-double, the tail (at most
// 2^-12 of mu) in float64, to s^45 ((1/9)^19 below 2^-58 of the tail).
inline DD log1pmx_series(DD t) {  // returns mu(t) >= 0
    const DD s = t / (t + 2.0);
    const DD s2 = s * s;
    const DD s3 = s2 * s;
    const DD s5 = s3 * s2;
    const double w = s2.hi;
    double tail = 1.0 / 45;
    for (int k = 43; k >= 7; k -= 2) tail = tail * w + 1.0 / k;
    const DD s7tail = s5 * (2.0 * w * tail);
    return s * t - s3 * kTwoThirds - s5 * kTwoFifths - s7tail;
}

// log(lambda) for lambda = hi + lo > 0 (normal): lambda = m 2^k with
// m in [1/sqrt 2, sqrt 2), log m = 2 atanh((m - 1) / (m + 1)).
inline DD log_series(DD lam) {
    int k = 0;
    double mh = std::frexp(lam.hi, &k);
    if (mh < 0.70710678118654752) {
        mh *= 2.0;
        --k;
    }
    const double ml = std::ldexp(lam.lo, -k);
    const DD num = DD{mh - 1.0, 0.0} + ml;  // mh - 1 is exact (Sterbenz)
    const DD den = two_sum(mh, 1.0) + ml;
    const DD s = num / den;  // |s| <= 0.1716
    const DD s2 = s * s;
    const DD s3 = s2 * s;
    const double w = s2.hi;
    double tail = 1.0 / 27;
    for (int j = 25; j >= 5; j -= 2) tail = tail * w + 1.0 / j;
    // log m = 2 s + 2 s^3 / 3 + 2 s^3 (s^2 tail)
    const DD logm = s * 2.0 + s3 * kTwoThirds + s3 * (2.0 * w * tail);
    return kLn2 * static_cast<double>(k) + logm;
}

// ---- the fast forms (the ones used): a table instead of the long series
//
// lambda = 2^k c (1 + r) with c = i/128 the table point nearest the mantissa m in
// [3/4, 3/2) and |r| <= 2^-7.58, r = (m - c)/c as hi + lo (m - c is exact; the
// table's rounded 1/c is corrected by one fma). Then
//     log(1 + r) = r - r^2/2 + r^3/3 - r^4 (1/4 - r/5 + ... + r^6/10),
// r^2 and r^3 as hi + lo, the rest in float64 (below 2^-30 of r^2), the series cut
// at r^11/11 < 2^-86. The series versions above took 31 ns (three divisions and 12
// to 20 terms); these take a quarter of it (7 ns) and are closer to the true value:
// against mpmath on 60 000 points, log 2^-77 (the series 2^-64.5) and mu 2^-68
// (the series 2^-62) relative (docs/design/special.md §5.4).
struct Reduced {
    double k;      // the binary exponent
    DD logc;       // log c
    DD r;
};
inline bool reduce(DD lam, Reduced& out) {
    std::uint64_t bits;
    std::memcpy(&bits, &lam.hi, sizeof bits);
    const int e = static_cast<int>(bits >> 52);  // sign 0 (lam > 0) and the biased exponent
    if (e <= 23 || e >= 2023) return false;  // 0, negative, subnormal, huge, inf, NaN: the series versions
    int k = e - 1023;
    bits = (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m;
    std::memcpy(&m, &bits, sizeof m);  // in [1, 2)
    if (m >= 1.5) {
        m *= 0.5;
        ++k;
    }
    const std::uint64_t sbits = static_cast<std::uint64_t>(1023 - k) << 52;
    double scale;
    std::memcpy(&scale, &sbits, sizeof scale);  // 2^-k
    const int i = static_cast<int>(m * 128.0 + 0.5);  // 96 .. 192
    const LogPoint& c = kLogTable[static_cast<std::size_t>(i - 96)];
    const double cv = i * 0x1p-7;
    const double d = m - cv;  // exact
    const double rh = d * c.inv;
    const double rl = (std::fma(-rh, cv, d) + lam.lo * scale) * c.inv;
    out = {static_cast<double>(k), {c.hi, c.lo}, quick_two_sum(rh, rl)};
    return true;
}
inline constexpr DD kThird = {0.33333333333333331, 1.8503717077085941e-17};
// r^2/2 and r^3/3 - r^4/4 + ... of log(1 + r), each as hi + lo (not normalised).
inline void log1p_parts(DD r, DD& half_r2, DD& tail) {
    DD r2 = two_prod(r.hi, r.hi);
    r2.lo += 2.0 * r.hi * r.lo;
    DD r3 = two_prod(r.hi, r2.hi);
    r3.lo += r.hi * r2.lo + r.lo * r2.hi;
    const double x = r.hi;
    const double q = 0.25 + x * (-0.2 + x * (1.0 / 6 + x * (-1.0 / 7 + x * (0.125 + x * (-1.0 / 9 + x * 0.1)))));
    half_r2 = {0.5 * r2.hi, 0.5 * r2.lo};
    const double th = r3.hi * kThird.hi;
    tail = {th, std::fma(r3.hi, kThird.hi, -th) + r3.hi * kThird.lo + r3.lo * kThird.hi - (r2.hi * r2.hi) * q};
}
// A sum of hi + lo terms: the high parts by two_sum, every low part and error into one
// double (they are 2^-53 of the terms: their own rounding is 2^-106).
struct Acc {
    double hi = 0.0, lo = 0.0;
    void add(double h, double l) {
        const DD s = two_sum(hi, h);
        hi = s.hi;
        lo += s.lo + l;
    }
    DD value() const { return quick_two_sum(hi, lo); }
};

// log(lambda) for lambda = hi + lo > 0.
inline DD log(DD lam) {
    Reduced z;
    if (!reduce(lam, z)) return log_series(lam);
    DD half_r2, tail;
    log1p_parts(z.r, half_r2, tail);
    const DD kl = two_prod(kLn2.hi, z.k);
    Acc s;
    s.add(kl.hi, kl.lo + kLn2.lo * z.k);
    s.add(z.logc.hi, z.logc.lo);
    s.add(z.r.hi, z.r.lo);
    s.add(-half_r2.hi, -half_r2.lo);
    s.add(tail.hi, tail.lo);
    return s.value();
}
inline DD log(double x) { return log(DD{x, 0.0}); }

// mu(t) = t - log(1 + t) >= 0 for t = hi + lo > -1 (any size: the reduction is of 1 + t).
// At c = 1, k = 0 (t small) r is t itself and t - r vanishes exactly, leaving
// r^2/2 - r^3/3 + ... with its own relative accuracy.
inline DD log1pmx_neg(DD t) {
    Reduced z;
    if (!reduce(two_sum(1.0, t.hi) + t.lo, z)) return log1pmx_series(t);
    DD half_r2, tail;
    log1p_parts(z.r, half_r2, tail);
    const DD kl = two_prod(kLn2.hi, z.k);
    Acc s;
    s.add(t.hi, t.lo);
    s.add(-kl.hi, -(kl.lo + kLn2.lo * z.k));
    s.add(-z.logc.hi, -z.logc.lo);
    s.add(-z.r.hi, -z.r.lo);
    s.add(half_r2.hi, half_r2.lo);
    s.add(-tail.hi, -tail.lo);
    return s.value();
}

// a * mu(lambda - 1) = a (lambda - 1 - log lambda) >= 0 for lambda = hi + lo:
// the atanh series near 1, a (lambda - 1) - a log(lambda) elsewhere (there
// |log lambda| >= log 2, so the subtraction loses at most 2 bits).
inline DD a_mu(double a, DD lam) {
    if (lam.hi >= 0.5 && lam.hi <= 2.0) return log1pmx_neg(lam - 1.0) * a;
    return (lam - 1.0) * a - log(lam) * a;
}

}  // namespace sd::special::dd
