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

#include <cmath>

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

inline DD operator/(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    DD r = a - b * q1;
    const double q2 = r.hi / b.hi;
    r = r - b * q2;
    const double q3 = r.hi / b.hi;
    return quick_two_sum(q1, q2) + q3;
}
inline DD operator/(DD a, double b) { return a / DD{b, 0.0}; }

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
inline DD log1pmx_neg(DD t) {  // returns mu(t) >= 0
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
inline DD log(DD lam) {
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
inline DD log(double x) { return log(DD{x, 0.0}); }

// a * mu(lambda - 1) = a (lambda - 1 - log lambda) >= 0 for lambda = hi + lo:
// the atanh series near 1, a (lambda - 1) - a log(lambda) elsewhere (there
// |log lambda| >= log 2, so the subtraction loses at most 2 bits).
inline DD a_mu(double a, DD lam) {
    if (lam.hi >= 0.5 && lam.hi <= 2.0) return log1pmx_neg(lam - 1.0) * a;
    return (lam - 1.0) * a - log(lam) * a;
}

}  // namespace sd::special::dd
